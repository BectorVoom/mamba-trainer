//! Completion-conditioned molecule model: substructure-set encoder, exact
//! teacher forcing and trainer.
//!
//! [`CompletionModel`] generates a complete molecule from its composition and
//! a set of supplied typed substructures. It reuses the spectrum-conditioned
//! [`Ms2Decoder`](super::decoder::Ms2Decoder) unchanged, but feeds it a
//! different memory: [`SubstructureEncoder`] encodes the supplied pattern set
//! permutation-invariantly (a pooled composition row plus per-atom message
//! passing inside each pattern, with no pattern-index or position embedding),
//! and [`CompletionTrainer`] trains it by teacher forcing under the
//! exact-completion masks (`meta` flag 2, as written by
//! [`TargetBatch::build_exact`](super::targets_batch::TargetBatch::build_exact)).
//!
//! Per-step remaining-composition and containment-progress features from the
//! design document (`docs/MOLECULAR_COMPLETION_DESIGN.md`, "Model
//! architecture") are deferred: the decoder sees the pattern memory and the
//! composition embedding only. [`Pattern::parent_atoms`](super::completion_data::Pattern::parent_atoms)
//! is audit metadata and never reaches the device: only the five
//! [`PatternBatch`] arrays do.

use std::collections::HashMap;
use std::path::Path;

use cubecl::prelude::Runtime;
use serde::{Deserialize, Serialize};

use crate::autograd::{Var, cat, no_grad};
use crate::backend::{DType, Device, FloatElem};
use crate::error::{Error, Result};
use crate::nn::init::Initializer;
use crate::nn::linear::{Linear, LinearConfig};
use crate::nn::module::{Module, ModuleVisitor, StateDict};
use crate::nn::norm::{RmsNorm, RmsNormConfig};
use crate::nn::param::Param;
use crate::ssm::SsmConfig;
use crate::tensor::Tensor;
use crate::tensor::ops::index::IdTensor;
use crate::tensor::ops::ms2::{self, Ms2Constants, ReplayBuffers};
use crate::tensor::ops::random::Rng;
use crate::train::optim::{AdamW, AdamWConfig, Optimizer, grad_scale};

use super::chem::{CHEMISTRY_VERSION, Composition, HYDROGEN, atom_type};
use super::completion::{contains_pattern, contains_patterns_disjoint};
use super::completion_data::{CompletionExample, CompletionSet, ExtractionConfig, PatternSource, SplitMix64, same_identity};
use super::completion_fingerprint::{
    FINGERPRINT_SLOTS, FingerprintBatch, FingerprintEncoder, FingerprintMode, FingerprintNoise,
    FingerprintNoiseLevel, FingerprintStore, SparseFingerprint,
};
use super::contain::Containment;
use super::functional_groups::functional_groups;
use super::contract::ModelConfig;
use super::contract::candidate_status;
use super::formula_enum::{EnumDomain, RatioBounds};
use super::decoder::{Ms2Decoder, ReplayView, TeacherOutput, graph_loss};
use super::encoder::EncoderOutput;
use super::generate::composed_decode_step;
use super::grammar::{COMPLETION_GRAMMAR_VERSION, Limits, Token, replay_exact};
use super::graph::MolGraph;
use super::targets_batch::TargetBatch;
use super::twin;
use super::workspace::Ms2Capabilities;

/// Config version of [`CompletionModelConfig`].
pub const COMPLETION_MODEL_VERSION: &str = "completion-model-v1";
/// Pattern-atom slots per query (the request layer's limit).
pub const PATTERN_SLOTS: usize = 24;
/// Patterns per query (the request layer's limit).
pub const MAX_PATTERNS: usize = 8;
/// Checkpoint header format of [`CompletionTrainer::save`].
pub const COMPLETION_CHECKPOINT_FORMAT: &str = "completion-checkpoint-v1";

/// Hyperparameters of [`CompletionModel`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CompletionModelConfig {
    /// Config version ([`COMPLETION_MODEL_VERSION`]).
    pub version: String,
    /// Grammar version ([`COMPLETION_GRAMMAR_VERSION`]).
    pub grammar: String,
    /// Chemistry domain version ([`CHEMISTRY_VERSION`]).
    pub chemistry: String,
    /// Residual stream width.
    pub d_model: u32,
    /// Trace-decoding SSM.
    pub decoder: SsmConfig,
    /// Decoder blocks (`1..=4`, the [`ModelConfig`] bound).
    pub decoder_blocks: u32,
    /// Cross-attention heads (must divide `d_model`).
    pub attention_heads: u32,
    /// Maximum atoms per molecule (`1..=32`).
    pub max_atoms: u32,
    /// Maximum ring closures per molecule (`<= 8`).
    pub max_ring_closures: u32,
    /// Message-passing rounds of the substructure encoder (`1..=8`).
    pub message_rounds: u32,
    /// Fingerprint token slots per query (`0` means no fingerprint encoder:
    /// existing checkpoints and fixtures load and behave bit-identically).
    /// When non-zero the model owns a [`FingerprintEncoder`](super::completion_fingerprint::FingerprintEncoder)
    /// and the decoder memory becomes `[context; pattern atoms (24);
    /// fingerprint tokens (slots)]`.
    #[serde(default)]
    pub fingerprint_slots: u32,
    /// Compute dtype.
    pub dtype: DType,
}

impl CompletionModelConfig {
    /// Small config for tests: `d = 64`, 2 decoder blocks, 4 heads, 16 atoms,
    /// 4 closures, 3 message rounds.
    pub fn small() -> Self {
        Self {
            version: COMPLETION_MODEL_VERSION.to_string(),
            grammar: COMPLETION_GRAMMAR_VERSION.to_string(),
            chemistry: CHEMISTRY_VERSION.to_string(),
            d_model: 64,
            decoder: SsmConfig {
                d_model: 64,
                n_heads: 4,
                head_dim: 16,
                d_state: 16,
                n_groups: 4,
                ..SsmConfig::default()
            },
            decoder_blocks: 2,
            attention_heads: 4,
            max_atoms: 16,
            max_ring_closures: 4,
            message_rounds: 3,
            fingerprint_slots: 0,
            dtype: DType::F32,
        }
    }

    /// Base config: `d = 128`, the [`ModelConfig::v0`] decoder SSM, 4 decoder
    /// blocks, 32 atoms, 6 closures, 3 message rounds.
    pub fn base() -> Self {
        Self {
            version: COMPLETION_MODEL_VERSION.to_string(),
            grammar: COMPLETION_GRAMMAR_VERSION.to_string(),
            chemistry: CHEMISTRY_VERSION.to_string(),
            d_model: 128,
            decoder: ModelConfig::v0().decoder.clone(),
            decoder_blocks: 4,
            attention_heads: 4,
            max_atoms: 32,
            max_ring_closures: 6,
            message_rounds: 3,
            fingerprint_slots: 0,
            dtype: DType::F32,
        }
    }

    /// Base config with the fingerprint encoder: [`base`](Self::base) with
    /// [`FINGERPRINT_SLOTS`] slots.
    pub fn base_fingerprint() -> Self {
        let mut config = Self::base();
        config.fingerprint_slots = FINGERPRINT_SLOTS as u32;
        config
    }

    /// Tiny config for the committed fixture: `d = 16`, 1 decoder block, 2
    /// heads, a matching small SSM, 12 atoms, 2 closures, 2 message rounds.
    /// `d = 16` (not 32) keeps the checkpoint well under 300 KB; the
    /// wider `d = 32` draft exceeded it at ~600 KB.
    pub fn tiny() -> Self {
        Self {
            version: COMPLETION_MODEL_VERSION.to_string(),
            grammar: COMPLETION_GRAMMAR_VERSION.to_string(),
            chemistry: CHEMISTRY_VERSION.to_string(),
            d_model: 16,
            decoder: SsmConfig {
                d_model: 16,
                n_heads: 2,
                head_dim: 8,
                d_state: 8,
                n_groups: 2,
                conv_kernel: None,
                ..SsmConfig::default()
            },
            decoder_blocks: 1,
            attention_heads: 2,
            max_atoms: 12,
            max_ring_closures: 2,
            message_rounds: 2,
            fingerprint_slots: 0,
            dtype: DType::F32,
        }
    }

    /// Check the versions and ranges: the three version strings must match
    /// their constants, the decoder SSM must validate with
    /// `decoder.d_model == d_model`, `d_model` must be a positive multiple
    /// of `attention_heads`, and the block/atom/closure/round counts must fit
    /// their documented ranges. `fingerprint_slots` is `0` (no fingerprint
    /// encoder) or `1..=4096`. Anything else is [`Error::Config`].
    pub fn validate(&self) -> Result<()> {
        if self.version != COMPLETION_MODEL_VERSION {
            return Err(Error::config(format!(
                "CompletionModelConfig::validate: version {:?} does not match {COMPLETION_MODEL_VERSION:?}",
                self.version
            )));
        }
        if self.grammar != COMPLETION_GRAMMAR_VERSION {
            return Err(Error::config(format!(
                "CompletionModelConfig::validate: grammar {:?} does not match {COMPLETION_GRAMMAR_VERSION:?}",
                self.grammar
            )));
        }
        if self.chemistry != CHEMISTRY_VERSION {
            return Err(Error::config(format!(
                "CompletionModelConfig::validate: chemistry {:?} does not match {CHEMISTRY_VERSION:?}",
                self.chemistry
            )));
        }
        self.decoder.validate()?;
        if self.decoder.d_model != self.d_model as usize {
            return Err(Error::config(format!(
                "CompletionModelConfig::validate: decoder d_model {} must equal d_model {}",
                self.decoder.d_model, self.d_model
            )));
        }
        if self.d_model == 0 || self.attention_heads == 0 {
            return Err(Error::config(format!(
                "CompletionModelConfig::validate: d_model {} and attention_heads {} must be non-zero",
                self.d_model, self.attention_heads
            )));
        }
        if !self.d_model.is_multiple_of(self.attention_heads) {
            return Err(Error::config(format!(
                "CompletionModelConfig::validate: d_model {} is not a multiple of attention_heads {}",
                self.d_model, self.attention_heads
            )));
        }
        if !(1..=4).contains(&self.decoder_blocks) {
            return Err(Error::config(format!(
                "CompletionModelConfig::validate: decoder_blocks {} is not in 1..=4",
                self.decoder_blocks
            )));
        }
        if !(1..=32).contains(&self.max_atoms) {
            return Err(Error::config(format!(
                "CompletionModelConfig::validate: max_atoms {} is not in 1..=32",
                self.max_atoms
            )));
        }
        if self.max_ring_closures > 8 {
            return Err(Error::config(format!(
                "CompletionModelConfig::validate: max_ring_closures {} exceeds 8",
                self.max_ring_closures
            )));
        }
        if !(1..=8).contains(&self.message_rounds) {
            return Err(Error::config(format!(
                "CompletionModelConfig::validate: message_rounds {} is not in 1..=8",
                self.message_rounds
            )));
        }
        if self.fingerprint_slots > 4096 {
            return Err(Error::config(format!(
                "CompletionModelConfig::validate: fingerprint_slots {} exceeds 4096",
                self.fingerprint_slots
            )));
        }
        Ok(())
    }

    /// The [`ModelConfig`] that [`Ms2Decoder::init`] needs: [`ModelConfig::v0`]
    /// with the decoder-relevant fields overridden (width, both SSM widths,
    /// blocks, heads, structure limits, dtype). Only those fields carry
    /// meaning here; the encoder SSM is width-matched but otherwise V0, and
    /// the formula table reference is untouched because the completion model
    /// has no formula head.
    fn decoder_model_config(&self) -> ModelConfig {
        let mut model = ModelConfig::v0();
        model.chemistry = self.chemistry.clone();
        model.d_model = self.d_model;
        model.encoder.d_model = self.d_model as usize;
        model.decoder = self.decoder.clone();
        model.decoder_blocks = self.decoder_blocks;
        model.attention_heads = self.attention_heads;
        model.max_atoms = self.max_atoms;
        model.max_ring_closures = self.max_ring_closures;
        model.dtype = self.dtype;
        model
    }
}

/// Host-side pattern batch: the queries' substructure sets laid into
/// [`PATTERN_SLOTS`] slots each, plus the queries' compositions.
///
/// The patterns of one query occupy its slots one after another (no pattern
/// index is stored); padding slots hold type id 0, open value 0, validity 0
/// and zero adjacency. Only these five arrays reach the device: no
/// per-pattern key, no parent atom index, no pattern index and no target
/// token.
#[derive(Clone, Debug)]
pub struct PatternBatch {
    /// Queries per batch.
    pub queries: usize,
    /// `[B*P]` atom type ids (0 in padding).
    pub types: Vec<u32>,
    /// `[B*P]` open attachment valences, `1 + residual` clamped to 8 (0 in
    /// padding).
    pub open: Vec<u32>,
    /// `[B*P]` slot validity, 1/0.
    pub valid: Vec<f32>,
    /// `[B*3*P*P]` bond adjacency: `adjacency[b, o-1, i, j] = 1` when slots
    /// `i` and `j` of query `b` are bonded with order `o` (symmetric,
    /// block-diagonal over the patterns), else 0.
    pub adjacency: Vec<f32>,
    /// `[B*10]` composition features, `ln(1 + count)` in
    /// [`ELEMENTS`](super::chem::ELEMENTS) order.
    pub features: Vec<f32>,
}

impl PatternBatch {
    /// Lay each query's patterns into its [`PATTERN_SLOTS`] slots one after
    /// another: `types` is the atom type id, `open` is `1 +` the atom's
    /// residual valence inside its own pattern
    /// ([`MolGraph::residual_valence`]) clamped to 8, `adjacency` the
    /// block-diagonal bond indicator per order, and `features` the
    /// `ln(1 + count)` composition. [`Error::Config`] when a query holds
    /// more than [`MAX_PATTERNS`] patterns, more than [`PATTERN_SLOTS`]
    /// pattern atoms in total, an empty pattern or a bond order outside
    /// `1..=3`.
    pub fn build(patterns: &[&[MolGraph]], compositions: &[Composition]) -> Result<Self> {
        if patterns.len() != compositions.len() {
            return Err(Error::config(format!(
                "PatternBatch::build: {} pattern entries for {} compositions",
                patterns.len(),
                compositions.len()
            )));
        }
        if patterns.is_empty() {
            return Err(Error::config(
                "PatternBatch::build: needs at least one query".to_string(),
            ));
        }
        let queries = patterns.len();
        let slots = PATTERN_SLOTS;
        let mut types = vec![0u32; queries * slots];
        let mut open = vec![0u32; queries * slots];
        let mut valid = vec![0.0f32; queries * slots];
        let mut adjacency = vec![0.0f32; queries * 3 * slots * slots];
        let mut features = vec![0.0f32; queries * 10];
        for (b, (pats, composition)) in patterns.iter().zip(compositions.iter()).enumerate() {
            if pats.len() > MAX_PATTERNS {
                return Err(Error::config(format!(
                    "PatternBatch::build: query {b} holds {} patterns, past the limit of {MAX_PATTERNS}",
                    pats.len()
                )));
            }
            let total: usize = pats.iter().map(|graph| graph.atoms().len()).sum();
            if total > slots {
                return Err(Error::config(format!(
                    "PatternBatch::build: query {b} holds {total} pattern atoms, past the limit of {slots} slots"
                )));
            }
            for (g, graph) in pats.iter().enumerate() {
                if graph.atoms().is_empty() {
                    return Err(Error::config(format!(
                        "PatternBatch::build: query {b} pattern {g} is empty"
                    )));
                }
            }
            let mut slot = 0usize;
            for graph in pats.iter() {
                let residual = graph.residual_valence();
                for (i, &atom_type) in graph.atoms().iter().enumerate() {
                    let at = b * slots + slot;
                    types[at] = u32::from(atom_type);
                    open[at] = (1u32 + u32::from(residual[i])).min(8);
                    valid[at] = 1.0;
                    slot += 1;
                }
            }
            let mut base = 0usize;
            for (g, graph) in pats.iter().enumerate() {
                for (a, other, order) in graph.bonds() {
                    if !matches!(order, 1..=3) {
                        return Err(Error::config(format!(
                            "PatternBatch::build: query {b} pattern {g} has bond order {order} (needs 1..=3)"
                        )));
                    }
                    let o = (*order as usize) - 1;
                    let (i, j) = (base + a, base + other);
                    adjacency[(b * 3 + o) * slots * slots + i * slots + j] = 1.0;
                    adjacency[(b * 3 + o) * slots * slots + j * slots + i] = 1.0;
                }
                base += graph.atoms().len();
            }
            for (e, count) in composition.iter().enumerate() {
                features[b * 10 + e] = (1.0 + f32::from(*count)).ln();
            }
        }
        Ok(Self {
            queries,
            types,
            open,
            valid,
            adjacency,
            features,
        })
    }

    /// Check the host arrays against the encoder's contract before upload.
    ///
    /// Array lengths must fit `queries` queries of [`PATTERN_SLOTS`] slots
    /// (`types`/`open`/`valid` are `[B*P]`, `adjacency` is `[B*3*P*P]` and
    /// `features` is `[B*10]`); every `valid` entry must be exactly 0.0 or
    /// 1.0; `types` must be below 18 and `open` below 9 (in padding slots too,
    /// apart from which padded `types`/`open` values are inert by
    /// selection); every `adjacency` and `features` entry must be finite;
    /// `adjacency` must be exactly 0 wherever either endpoint slot is
    /// invalid, symmetric, and zero on the diagonal. Anything else is
    /// [`Error::Config`] naming the first offending flat index.
    pub fn validate(&self) -> Result<()> {
        let (b, p) = (self.queries, PATTERN_SLOTS);
        if self.types.len() != b * p {
            return Err(Error::config(format!(
                "PatternBatch::validate: types holds {} entries for {b} queries of {p} slots",
                self.types.len()
            )));
        }
        if self.open.len() != b * p {
            return Err(Error::config(format!(
                "PatternBatch::validate: open holds {} entries for {b} queries of {p} slots",
                self.open.len()
            )));
        }
        if self.valid.len() != b * p {
            return Err(Error::config(format!(
                "PatternBatch::validate: valid holds {} entries for {b} queries of {p} slots",
                self.valid.len()
            )));
        }
        if self.adjacency.len() != b * 3 * p * p {
            return Err(Error::config(format!(
                "PatternBatch::validate: adjacency holds {} entries for {b} queries of 3 orders and {p} slots",
                self.adjacency.len()
            )));
        }
        if self.features.len() != b * 10 {
            return Err(Error::config(format!(
                "PatternBatch::validate: features holds {} entries for {b} queries",
                self.features.len()
            )));
        }
        for (s, &v) in self.valid.iter().enumerate() {
            if v != 0.0 && v != 1.0 {
                return Err(Error::config(format!(
                    "PatternBatch::validate: valid[{s}] is {v} (needs exactly 0.0 or 1.0)"
                )));
            }
        }
        for (s, &t) in self.types.iter().enumerate() {
            if t >= 18 {
                return Err(Error::config(format!(
                    "PatternBatch::validate: types[{s}] is {t} (needs below 18)"
                )));
            }
        }
        for (s, &o) in self.open.iter().enumerate() {
            if o >= 9 {
                return Err(Error::config(format!(
                    "PatternBatch::validate: open[{s}] is {o} (needs below 9)"
                )));
            }
        }
        for (i, &x) in self.adjacency.iter().enumerate() {
            if !x.is_finite() {
                return Err(Error::config(format!(
                    "PatternBatch::validate: adjacency[{i}] is not finite"
                )));
            }
        }
        for (i, &x) in self.features.iter().enumerate() {
            if !x.is_finite() {
                return Err(Error::config(format!(
                    "PatternBatch::validate: features[{i}] is not finite"
                )));
            }
        }
        for q in 0..b {
            for o in 0..3 {
                for i in 0..p {
                    for j in 0..p {
                        let flat = (q * 3 + o) * p * p + i * p + j;
                        let x = self.adjacency[flat];
                        if i == j && x != 0.0 {
                            return Err(Error::config(format!(
                                "PatternBatch::validate: adjacency[{flat}] (query {q} order {o} slot {i}) is {x} on the diagonal (needs 0)"
                            )));
                        }
                        let ok_i = self.valid[q * p + i] != 0.0;
                        let ok_j = self.valid[q * p + j] != 0.0;
                        if (!ok_i || !ok_j) && x != 0.0 {
                            return Err(Error::config(format!(
                                "PatternBatch::validate: adjacency[{flat}] (query {q} order {o} slots {i},{j}) is {x} touching an invalid slot (needs 0)"
                            )));
                        }
                        let mirror = self.adjacency[(q * 3 + o) * p * p + j * p + i];
                        if x != mirror {
                            return Err(Error::config(format!(
                                "PatternBatch::validate: adjacency[{flat}] (query {q} order {o} slots {i},{j}) is {x} but its mirror is {mirror} (needs symmetry)"
                            )));
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

/// Device copy of a [`PatternBatch`]: exactly the five host arrays, no key,
/// no parent atom index, no pattern index, no target token.
struct DevicePatterns<R: Runtime, E: FloatElem> {
    /// `[B, P]` atom type ids.
    types: IdTensor<R>,
    /// `[B, P]` open attachment valences.
    open: IdTensor<R>,
    /// `[B, P]` slot validity.
    valid: Tensor<R, E>,
    /// `[B, 3, P, P]` bond adjacency per order.
    adjacency: Tensor<R, E>,
    /// `[B, 10]` composition features.
    features: Tensor<R, E>,
}

impl PatternBatch {
    /// Upload the five arrays: 5 uploads, no launch, no read. Mismatched
    /// lengths (e.g. from hand-edited host arrays) are [`Error::Shape`].
    fn upload<R: Runtime, E: FloatElem>(&self, device: &Device<R>) -> Result<DevicePatterns<R, E>> {
        let (b, p) = (self.queries, PATTERN_SLOTS);
        if self.types.len() != b * p
            || self.open.len() != b * p
            || self.valid.len() != b * p
            || self.adjacency.len() != b * 3 * p * p
            || self.features.len() != b * 10
        {
            return Err(Error::shape(format!(
                "PatternBatch::upload: lengths {} {} {} {} {} do not fit {b} queries of {p} slots",
                self.types.len(),
                self.open.len(),
                self.valid.len(),
                self.adjacency.len(),
                self.features.len()
            )));
        }
        Ok(DevicePatterns {
            types: IdTensor::from_slice(&self.types, vec![b, p], device)?,
            open: IdTensor::from_slice(&self.open, vec![b, p], device)?,
            valid: Tensor::<R, E>::from_f32(&self.valid, vec![b, p], device)?,
            adjacency: Tensor::<R, E>::from_f32(&self.adjacency, vec![b, 3, p, p], device)?,
            features: Tensor::<R, E>::from_f32(&self.features, vec![b, 10], device)?,
        })
    }
}

/// One message-passing round of [`SubstructureEncoder`]: per-order bond
/// projections, the self update, the output projection and the norm.
struct EncoderRound<R: Runtime, E: FloatElem> {
    /// `bond[o]`: `Linear(d, d, no bias)` applied before the order-`o + 1`
    /// adjacency product.
    bond: [Linear<R, E>; 3],
    /// `Linear(d, d)` self term added to the incoming messages.
    own: Linear<R, E>,
    /// `Linear(d, d)` output projection of the residual update.
    out: Linear<R, E>,
    /// Per-round normalisation.
    norm: RmsNorm<R, E>,
}

impl<R: Runtime, E: FloatElem> Module<R, E> for EncoderRound<R, E> {
    fn visit(&self, visitor: &mut ModuleVisitor<'_, R, E>) {
        for (o, bond) in self.bond.iter().enumerate() {
            visitor.child_at("bond", o, bond);
        }
        visitor.child("own", &self.own);
        visitor.child("out", &self.out);
        visitor.child("norm", &self.norm);
    }
}

/// Permutation-invariant encoder of the supplied substructure set.
///
/// Atom embeddings (type plus open attachment valence) are refined by shared
/// message passing inside each pattern — the block-diagonal adjacency never
/// mixes patterns — and pooled with the composition embedding. There is no
/// pattern-index or position embedding: the output is invariant to the order
/// of the patterns and equivariant to the order of the atoms inside a
/// pattern. Padding slots are selected to exact zeros after every round, so
/// in-range padded `types`/`open` values cannot reach any output bit (see
/// the padding test); [`PatternBatch::validate`] rejects any adjacency
/// touching an invalid slot, any out-of-range id and any non-finite entry
/// before upload.
pub struct SubstructureEncoder<R: Runtime, E: FloatElem> {
    /// `[18, d]` atom-type table (row 0 is padding).
    type_emb: Param<R, E>,
    /// `[9, d]` open-valence table (row 0 is padding, rows `1..=8`).
    open_emb: Param<R, E>,
    /// Message-passing rounds.
    rounds: Vec<EncoderRound<R, E>>,
    /// `Linear(10, d)` composition row network input.
    row_in: Linear<R, E>,
    /// `Linear(d, d)` composition row network output.
    row_out: Linear<R, E>,
    /// `Linear(d, d)` projection of the composition vector into memory.
    memory_in: Linear<R, E>,
    /// Residual width.
    d_model: usize,
}

impl<R: Runtime, E: FloatElem> Module<R, E> for SubstructureEncoder<R, E> {
    fn visit(&self, visitor: &mut ModuleVisitor<'_, R, E>) {
        visitor.param("type_emb", &self.type_emb);
        visitor.param("open_emb", &self.open_emb);
        for (r, round) in self.rounds.iter().enumerate() {
            visitor.child_at("round", r, round);
        }
        visitor.child("row_in", &self.row_in);
        visitor.child("row_out", &self.row_out);
        visitor.child("memory_in", &self.memory_in);
    }
}

/// One `[rows, d]` embedding table with the decoder's standard init.
fn encoder_table<R: Runtime, E: FloatElem>(
    rows: usize,
    d: usize,
    device: &Device<R>,
    rng: &mut Rng,
) -> Param<R, E> {
    Param::new(
        Initializer::Normal {
            mean: 0.0,
            std: 0.02,
        }
        .init(vec![rows, d], device, rng),
    )
}

impl<R: Runtime, E: FloatElem> SubstructureEncoder<R, E> {
    /// Build the encoder for width `d` with `rounds` message-passing rounds.
    fn init(d_model: usize, rounds: usize, device: &Device<R>, rng: &mut Rng) -> Self {
        let type_emb = encoder_table(18, d_model, device, rng);
        let open_emb = encoder_table(9, d_model, device, rng);
        let mut round_vec = Vec::with_capacity(rounds);
        for _ in 0..rounds {
            let bond = [
                LinearConfig::new(d_model, d_model)
                    .with_bias(false)
                    .init(device, rng),
                LinearConfig::new(d_model, d_model)
                    .with_bias(false)
                    .init(device, rng),
                LinearConfig::new(d_model, d_model)
                    .with_bias(false)
                    .init(device, rng),
            ];
            round_vec.push(EncoderRound {
                bond,
                own: LinearConfig::new(d_model, d_model).init(device, rng),
                out: LinearConfig::new(d_model, d_model).init(device, rng),
                norm: RmsNormConfig::new(d_model).init(device, rng),
            });
        }
        Self {
            type_emb,
            open_emb,
            rounds: round_vec,
            row_in: LinearConfig::new(10, d_model).init(device, rng),
            row_out: LinearConfig::new(d_model, d_model).init(device, rng),
            memory_in: LinearConfig::new(d_model, d_model).init(device, rng),
            d_model,
        }
    }

    /// Pattern states and composition row without memory assembly: `h`
    /// (`[B, P, d]`, message-passed and reselected) with its validity, and
    /// `g` (`[B, d]`, the composition row network). Shared by
    /// [`encode`](Self::encode) and the fingerprint-combined path, so the
    /// pattern branch is bit-identical in both.
    fn encode_intermediates(
        &self,
        batch: &PatternBatch,
        device: &Device<R>,
    ) -> Result<(Var<R, E>, Tensor<R, E>, Var<R, E>, Tensor<R, E>, Tensor<R, E>)> {
        batch.validate()?;
        let uploaded = batch.upload(device)?;
        let (b, p, d) = (batch.queries, PATTERN_SLOTS, self.d_model);
        if b == 0 {
            let h = Var::constant(Tensor::<R, E>::zeros(vec![0, p, d], device));
            let valid = Tensor::<R, E>::zeros(vec![0, p], device);
            let g = Var::constant(Tensor::<R, E>::zeros(vec![0, d], device));
            let features = Tensor::<R, E>::zeros(vec![0, 10], device);
            let adjacency = Tensor::<R, E>::zeros(vec![0, 3, p, p], device);
            return Ok((h, valid, g, features, adjacency));
        }
        let flat = b * p;
        let type_ids = uploaded.types.reshape(vec![flat])?;
        let open_ids = uploaded.open.reshape(vec![flat])?;
        let mut h = Var::ms2_lookup(&self.type_emb.var_standalone(), &type_ids)?
            .reshape(vec![b, p, d])?
            .add(
                &Var::ms2_lookup(&self.open_emb.var_standalone(), &open_ids)?
                    .reshape(vec![b, p, d])?,
            )?;
        h = h.ms2_select_valid(&uploaded.valid)?;
        for round in &self.rounds {
            let mut messages = Var::constant(Tensor::<R, E>::zeros(vec![b, p, d], device));
            for o in 0..3 {
                let projected = round.bond[o].apply(&h)?;
                let adj = crate::tensor::ops::movement::slice(&uploaded.adjacency, 1, o, 1)?
                    .reshape(vec![b, p, p])?;
                messages = messages.add(&Var::constant(adj).matmul(&projected)?)?;
            }
            let update = round
                .out
                .apply(&messages.add(&round.own.apply(&h)?)?.silu()?)?;
            h = h.add(&update)?;
            h = round.norm.apply(&h)?.ms2_select_valid(&uploaded.valid)?;
        }
        let g = self.row_out.apply(
            &self
                .row_in
                .apply(&Var::constant(uploaded.features.clone()))?
                .silu()?,
        )?;
        Ok((h, uploaded.valid, g, uploaded.features, uploaded.adjacency))
    }

    /// Encode `batch` into the decoder memory, with no device read:
    /// `h = lookup(type) + lookup(open)`, selected to exact zeros in
    /// padding; per round, adjacency-masked message passing plus a residual
    /// update, renormalised and reselected; the composition row network `g`;
    /// `memory = [Linear(g); h]`, `mask = [1; valid]`, and the pool is the
    /// clamped mean of `h` plus `g`, exactly as
    /// [`Ms2Encoder::encode`](super::encoder::Ms2Encoder::encode) assembles
    /// it (so an empty pattern set gives `pool == g` bit for bit).
    pub fn encode(&self, batch: &PatternBatch, device: &Device<R>) -> Result<EncoderOutput<R, E>> {
        let (b, p, d) = (batch.queries, PATTERN_SLOTS, self.d_model);
        if b == 0 {
            let x = Var::constant(Tensor::<R, E>::zeros(vec![0, p, d], device));
            let valid = Tensor::<R, E>::zeros(vec![0, p], device);
            let memory = Var::constant(Tensor::<R, E>::zeros(vec![0, 1 + p, d], device));
            let memory_mask = Tensor::<R, E>::zeros(vec![0, 1 + p], device);
            let pool = Var::constant(Tensor::<R, E>::zeros(vec![0, d], device));
            let context = Var::constant(Tensor::<R, E>::zeros(vec![0, d], device));
            return Ok(EncoderOutput {
                x,
                valid,
                memory,
                memory_mask,
                pool,
                context,
            });
        }
        let (h, valid, g, _, _) = self.encode_intermediates(batch, device)?;
        // Re-upload for the adjacency-free assembly below is unnecessary: the
        // intermediates carry every device value; validity is `valid`.
        let mem0 = self.memory_in.apply(&g)?.unsqueeze(1)?;
        let memory = cat(&[mem0, h.clone()], 1)?;
        let ones = Tensor::<R, E>::ones(vec![b, 1], device);
        let memory_mask = crate::tensor::ops::movement::cat(&[ones, valid.clone()], 1)?;
        // The clamped mean exactly as `Ms2Encoder::encode` does it, so the
        // mean part is exactly zero when no slot is valid.
        let x_sum = h.sum_dim(1)?;
        let len_t =
            crate::tensor::ops::reduce::sum_dim(&valid, 1)?.reshape(vec![b, 1, 1])?;
        let len_var = Var::constant(len_t);
        let one_var = Var::constant(Tensor::<R, E>::ones(vec![b, 1, 1], device));
        let den = len_var.maximum(&one_var)?;
        let mean = x_sum.div(&den)?.squeeze(1)?;
        // `g + mean`, not `mean + g`: elementwise addition commutes
        // bitwise for finite values (all values here are finite by
        // construction), and this order keeps `g`'s tape the accumulator.
        // Tape merges drain every parent tape but the first parent's into
        // it, so with `mean` first the `cat` above would leave `memory`
        // and `context` pointing at a drained tape: their values would
        // stay exact, but no gradient could reach the encoder through
        // them. With `g` first, `memory`, `pool` and `context` all stay
        // rooted on the full tape. (`x` keeps the atom branch's tape,
        // which this drains; its values stay exact, and no training path
        // differentiates through `x` — the decoder reads only `memory`,
        // `memory_mask` and the context.)
        let pool = g.add(&mean)?;
        Ok(EncoderOutput {
            x: h,
            valid,
            memory,
            memory_mask,
            pool,
            context: g,
        })
    }
}

/// Completion-conditioned molecule model: the substructure-set encoder plus
/// the reused graph-action decoder.
pub struct CompletionModel<R: Runtime, E: FloatElem> {
    /// The hyperparameters the two parts were built for.
    pub config: CompletionModelConfig,
    /// Substructure-set encoder (the decoder's memory source).
    encoder: SubstructureEncoder<R, E>,
    /// Fingerprint-set encoder (`None` when `fingerprint_slots == 0`).
    fingerprint_encoder: Option<FingerprintEncoder<R, E>>,
    /// Graph-action decoder (architecture §4.3, reused unchanged).
    decoder: Ms2Decoder<R, E>,
}

impl<R: Runtime, E: FloatElem> Module<R, E> for CompletionModel<R, E> {
    fn visit(&self, visitor: &mut ModuleVisitor<'_, R, E>) {
        visitor.child("encoder", &self.encoder);
        if let Some(fp) = &self.fingerprint_encoder {
            visitor.child("fingerprint_encoder", fp);
        }
        visitor.child("decoder", &self.decoder);
    }
}

impl<R: Runtime, E: FloatElem> CompletionModel<R, E> {
    /// Build the encoder and decoder for `config` on `device`. The config
    /// must validate (versions and ranges), the mapped [`ModelConfig`] must
    /// validate, and the neural element type must equal the configured dtype
    /// (refused with [`Error::Config`] before any allocation, as in
    /// [`Ms2Model::init`](super::generate::Ms2Model::init)).
    pub fn init(config: &CompletionModelConfig, device: &Device<R>, rng: &mut Rng) -> Result<Self> {
        config.validate()?;
        let mapped = config.decoder_model_config();
        mapped.validate().map_err(|e| {
            Error::config(format!(
                "CompletionModel::init: mapped model config rejected: {e}"
            ))
        })?;
        Ms2Capabilities::check_device::<R, E>(device, &mapped)?;
        let d = config.d_model as usize;
        let encoder = SubstructureEncoder::init(d, config.message_rounds as usize, device, rng);
        let fingerprint_encoder = if config.fingerprint_slots > 0 {
            Some(FingerprintEncoder::init(d, device, rng))
        } else {
            None
        };
        let decoder = Ms2Decoder::init(&mapped, device, rng)?;
        Ok(Self {
            config: config.clone(),
            encoder,
            fingerprint_encoder,
            decoder,
        })
    }

    /// Whether the model owns a fingerprint encoder (`fingerprint_slots > 0`).
    pub fn has_fingerprint(&self) -> bool {
        self.fingerprint_encoder.is_some()
    }

    /// Fingerprint slots of the model (`0` without the encoder).
    pub fn fingerprint_slots(&self) -> usize {
        self.config.fingerprint_slots as usize
    }

    /// Encode `batch` into the decoder memory, pool and context. No device
    /// read. A model with a fingerprint encoder encodes an empty token set
    /// here; use [`encode_with_fingerprints`](Self::encode_with_fingerprints)
    /// to supply one.
    pub fn encode(&self, batch: &PatternBatch, device: &Device<R>) -> Result<EncoderOutput<R, E>> {
        match &self.fingerprint_encoder {
            None => self.encoder.encode(batch, device),
            Some(_) => {
                let empty = FingerprintBatch::empty(batch.queries, self.fingerprint_slots())?;
                self.encode_with_fingerprints(batch, &empty, device)
            }
        }
    }

    /// Encode patterns with a fingerprint set into the decoder memory
    /// (`[context; pattern atoms; fingerprint tokens]` with the matching
    /// mask). The composition row `g` adds the pooled fingerprint vector
    /// (masked mean through a `Linear`) to the composition network, with `g`
    /// first in every addition (the tape-order discipline of the pattern
    /// encoder). No device read.
    ///
    /// `fingerprints.slots` must equal the model's `fingerprint_slots` and
    /// the query counts must match; a model without the encoder rejects any
    /// non-empty fingerprint set with [`Error::Config`].
    pub fn encode_with_fingerprints(
        &self,
        patterns: &PatternBatch,
        fingerprints: &FingerprintBatch,
        device: &Device<R>,
    ) -> Result<EncoderOutput<R, E>> {
        let Some(fp_encoder) = &self.fingerprint_encoder else {
            if fingerprints.valid.iter().any(|&v| v != 0.0) {
                return Err(Error::config(
                    "CompletionModel::encode: model without a fingerprint encoder was given a fingerprint (fingerprint_slots = 0)".to_string(),
                ));
            }
            return self.encoder.encode(patterns, device);
        };
        let slots = self.fingerprint_slots();
        if fingerprints.slots != slots {
            return Err(Error::config(format!(
                "CompletionModel::encode: fingerprint batch holds {} slots for a model with {slots}",
                fingerprints.slots
            )));
        }
        if fingerprints.queries != patterns.queries {
            return Err(Error::config(format!(
                "CompletionModel::encode: {} fingerprint queries for {} pattern queries",
                fingerprints.queries, patterns.queries
            )));
        }
        let (b, p) = (patterns.queries, PATTERN_SLOTS);
        let d = self.config.d_model as usize;
        if b == 0 {
            let x = Var::constant(Tensor::<R, E>::zeros(vec![0, p, d], device));
            let valid = Tensor::<R, E>::zeros(vec![0, p], device);
            let memory = Var::constant(Tensor::<R, E>::zeros(vec![0, 1 + p + slots, d], device));
            let memory_mask = Tensor::<R, E>::zeros(vec![0, 1 + p + slots], device);
            let pool = Var::constant(Tensor::<R, E>::zeros(vec![0, d], device));
            let context = Var::constant(Tensor::<R, E>::zeros(vec![0, d], device));
            return Ok(EncoderOutput {
                x,
                valid,
                memory,
                memory_mask,
                pool,
                context,
            });
        }
        let (h_pat, valid_pat, g0, _, _) = self.encoder.encode_intermediates(patterns, device)?;
        let (h_fp, valid_fp) = fp_encoder.encode_states(fingerprints, device)?;
        let fp_pooled = fp_encoder.encode_pooled_from_states(&h_fp, &valid_fp, device)?;
        // `g0 + fp`, not `fp + g0`: same tape discipline as the pattern
        // encoder (the first parent keeps the accumulator tape).
        let g = g0.add(&fp_pooled)?;
        let mem0 = self.encoder.memory_in.apply(&g)?.unsqueeze(1)?;
        let memory = cat(&[mem0, h_pat.clone(), h_fp], 1)?;
        let ones = Tensor::<R, E>::ones(vec![b, 1], device);
        let memory_mask = crate::tensor::ops::movement::cat(
            &[ones, valid_pat.clone(), valid_fp],
            1,
        )?;
        let x_sum = h_pat.sum_dim(1)?;
        let len_t =
            crate::tensor::ops::reduce::sum_dim(&valid_pat, 1)?.reshape(vec![b, 1, 1])?;
        let len_var = Var::constant(len_t);
        let one_var = Var::constant(Tensor::<R, E>::ones(vec![b, 1, 1], device));
        let den = len_var.maximum(&one_var)?;
        let mean = x_sum.div(&den)?.squeeze(1)?;
        let pool = g.add(&mean)?;
        Ok(EncoderOutput {
            x: h_pat,
            valid: valid_pat,
            memory,
            memory_mask,
            pool,
            context: g,
        })
    }

    /// Fingerprint batch for `generate`: one [`SparseFingerprint`] per
    /// feasible request (`None` means an empty token set). A model without
    /// the encoder given any fingerprint is [`Error::Config`]; a model with
    /// it given none encodes empty sets.
    fn fingerprint_batch_for(
        &self,
        requests: &[CompletionRequest],
        feasible: &[usize],
    ) -> Result<FingerprintBatch> {
        let slots = self.fingerprint_slots();
        match &self.fingerprint_encoder {
            None => {
                for &i in feasible {
                    if requests[i].fingerprint.is_some() {
                        return Err(Error::config(
                            "CompletionModel::generate: request carries a fingerprint but the model has no fingerprint encoder (fingerprint_slots = 0)".to_string(),
                        ));
                    }
                }
                let empty_fp = SparseFingerprint { entries: Vec::new() };
                let fps: Vec<SparseFingerprint> =
                    feasible.iter().map(|_| empty_fp.clone()).collect();
                FingerprintBatch::build(&fps, 0)
            }
            Some(_) => {
                let mut fps: Vec<SparseFingerprint> = Vec::with_capacity(feasible.len());
                let empty = SparseFingerprint { entries: Vec::new() };
                for &i in feasible {
                    match requests[i].fingerprint {
                        Some(fp) => {
                            fp.validate()?;
                            fps.push(SparseFingerprint {
                                entries: fp.entries.clone(),
                            });
                        }
                        None => fps.push(empty.clone()),
                    }
                }
                FingerprintBatch::build(&fps, slots)
            }
        }
    }

    /// Teacher forcing under the exact-completion masks: upload the patterns
    /// (5 uploads) and the targets (4 uploads), run
    /// [`grammar_replay`](ms2::grammar_replay) (flag 2 is in the meta), then
    /// [`Ms2Decoder::teacher`] with the encoder's context as the composition
    /// embedding. The loss is [`graph_loss`] (the mean over queries of the
    /// summed sequence NLL). No device read.
    ///
    /// `targets` must hold one exact-completion slot per query (`slots = 1`,
    /// as
    /// [`TargetBatch::build_exact`](super::targets_batch::TargetBatch::build_exact)
    /// writes): one slot per query, `max_steps` equal to the model's `T`,
    /// and every occupied row's budget flag equal to 2. A
    /// [`TargetBatch::build`](super::targets_batch::TargetBatch::build)
    /// batch (flag 1) or any other shape is [`Error::Config`], checked before
    /// any upload. The replay's first-illegal-step column is `u32::MAX` for
    /// every row whenever the targets passed `build_exact`'s host
    /// validation, so the training path performs no device read for it.
    pub fn teacher(
        &self,
        patterns: &PatternBatch,
        targets: &TargetBatch,
        constants: &Ms2Constants<R>,
        device: &Device<R>,
    ) -> Result<(TeacherOutput<R, E>, Var<R, E>)> {
        if self.fingerprint_encoder.is_some() {
            let empty =
                FingerprintBatch::empty(patterns.queries, self.fingerprint_slots())?;
            return self.teacher_with_fingerprints(patterns, &empty, targets, constants, device);
        }
        let queries = patterns.queries;
        if queries == 0 {
            return Err(Error::config(
                "CompletionModel::teacher: needs at least one query".to_string(),
            ));
        }
        if targets.spectra != queries || targets.slots != 1 {
            return Err(Error::config(format!(
                "CompletionModel::teacher: needs one exact target slot per query ({queries} queries), got spectra {} slots {}",
                targets.spectra, targets.slots
            )));
        }
        let limits = Limits::new(
            self.config.max_atoms as usize,
            self.config.max_ring_closures as usize,
        )
        .map_err(|e| {
            Error::config(format!(
                "CompletionModel::teacher: model limits rejected: {e}"
            ))
        })?;
        if targets.max_steps != limits.max_steps() {
            return Err(Error::config(format!(
                "CompletionModel::teacher: targets hold T = {} steps, the model needs {}",
                targets.max_steps,
                limits.max_steps()
            )));
        }
        let rows = targets.spectra * targets.slots;
        if targets.meta.len() != rows * 12 {
            return Err(Error::config(format!(
                "CompletionModel::teacher: targets meta holds {} words for {rows} slots",
                targets.meta.len()
            )));
        }
        for row in 0..rows {
            if targets.meta[row * 12] == 0 {
                continue;
            }
            if targets.meta[row * 12 + 1] != 2 {
                return Err(Error::config(format!(
                    "CompletionModel::teacher: targets row {row} carries budget flag {} (needs 2, the exact-completion flag)",
                    targets.meta[row * 12 + 1]
                )));
            }
        }
        let encoded = self.encode(patterns, device)?;
        let uploaded = targets.upload(device)?;
        let atoms = self.config.max_atoms as usize;
        let buffers = ReplayBuffers::new(queries, targets.max_steps, atoms, device);
        ms2::grammar_replay(
            &uploaded.tokens,
            &uploaded.meta,
            constants,
            self.config.max_atoms,
            self.config.max_ring_closures,
            &buffers,
        )?;
        let replay = ReplayView {
            replay: &buffers.replay,
            atoms: &buffers.atoms,
        };
        let out = self
            .decoder
            .teacher(&encoded, &encoded.context, &uploaded, &replay)?;
        let loss = graph_loss(&out, &uploaded.q, queries)?;
        Ok((out, loss))
    }

    /// Teacher forcing with a fingerprint set: like
    /// [`teacher`](Self::teacher) but encoding `patterns` with
    /// `fingerprints` (see
    /// [`encode_with_fingerprints`](Self::encode_with_fingerprints)). A model
    /// without the encoder rejects a non-empty set; `fingerprints.slots`
    /// must equal the model's slots. No device read.
    pub fn teacher_with_fingerprints(
        &self,
        patterns: &PatternBatch,
        fingerprints: &FingerprintBatch,
        targets: &TargetBatch,
        constants: &Ms2Constants<R>,
        device: &Device<R>,
    ) -> Result<(TeacherOutput<R, E>, Var<R, E>)> {
        let queries = patterns.queries;
        if queries == 0 {
            return Err(Error::config(
                "CompletionModel::teacher: needs at least one query".to_string(),
            ));
        }
        if targets.spectra != queries || targets.slots != 1 {
            return Err(Error::config(format!(
                "CompletionModel::teacher: needs one exact target slot per query ({queries} queries), got spectra {} slots {}",
                targets.spectra, targets.slots
            )));
        }
        let limits = Limits::new(
            self.config.max_atoms as usize,
            self.config.max_ring_closures as usize,
        )
        .map_err(|e| {
            Error::config(format!(
                "CompletionModel::teacher: model limits rejected: {e}"
            ))
        })?;
        if targets.max_steps != limits.max_steps() {
            return Err(Error::config(format!(
                "CompletionModel::teacher: targets hold T = {} steps, the model needs {}",
                targets.max_steps,
                limits.max_steps()
            )));
        }
        let rows = targets.spectra * targets.slots;
        if targets.meta.len() != rows * 12 {
            return Err(Error::config(format!(
                "CompletionModel::teacher: targets meta holds {} words for {rows} slots",
                targets.meta.len()
            )));
        }
        for row in 0..rows {
            if targets.meta[row * 12] == 0 {
                continue;
            }
            if targets.meta[row * 12 + 1] != 2 {
                return Err(Error::config(format!(
                    "CompletionModel::teacher: targets row {row} carries budget flag {} (needs 2, the exact-completion flag)",
                    targets.meta[row * 12 + 1]
                )));
            }
        }
        let encoded = self.encode_with_fingerprints(patterns, fingerprints, device)?;
        let uploaded = targets.upload(device)?;
        let atoms = self.config.max_atoms as usize;
        let buffers = ReplayBuffers::new(queries, targets.max_steps, atoms, device);
        ms2::grammar_replay(
            &uploaded.tokens,
            &uploaded.meta,
            constants,
            self.config.max_atoms,
            self.config.max_ring_closures,
            &buffers,
        )?;
        let replay = ReplayView {
            replay: &buffers.replay,
            atoms: &buffers.atoms,
        };
        let out = self
            .decoder
            .teacher(&encoded, &encoded.context, &uploaded, &replay)?;
        let loss = graph_loss(&out, &uploaded.q, queries)?;
        Ok((out, loss))
    }
}

/// Training hyperparameters of [`CompletionTrainer`].
pub struct CompletionTrainConfig {
    /// AdamW base learning rate.
    pub lr: f32,
    /// AdamW decoupled weight decay.
    pub weight_decay: f32,
    /// Global gradient-norm clip (`None` disables it), applied with the
    /// device-side scale, so clipping costs no device read.
    pub grad_clip: Option<f32>,
    /// Seed for weight initialisation.
    pub seed: u64,
    /// How patterns are cut from each example's target.
    pub extraction: ExtractionConfig,
    /// Sampling seed for pattern extraction.
    pub extraction_seed: u64,
    /// Where the patterns of a query come from. Old checkpoints without this
    /// field load as `RandomPatches` of `extraction`.
    pub pattern_source: PatternSource,
    /// How the fingerprint evidence of a training query is produced (`None`
    /// means no fingerprint: patterns only, as today). Old checkpoints
    /// without this field load as `None`.
    pub fingerprint_mode: Option<FingerprintMode>,
    /// Token threshold for fingerprint evidence: entries below it are
    /// dropped (default 0.1). Always validated in `(0, 1]`.
    pub fingerprint_threshold: f32,
}

/// Default fingerprint token threshold (0.1).
fn default_fingerprint_threshold() -> f32 {
    0.1
}

/// The serializable fields of [`CompletionTrainConfig`].
#[derive(Serialize, Deserialize)]
struct CompletionTrainConfigFields {
    /// AdamW base learning rate.
    lr: f32,
    /// AdamW decoupled weight decay.
    weight_decay: f32,
    /// Global gradient-norm clip.
    grad_clip: Option<f32>,
    /// Seed for weight initialisation.
    seed: u64,
    /// Fewest patterns per parent.
    min_patterns: usize,
    /// Most patterns per parent.
    max_patterns: usize,
    /// Fewest atoms per pattern.
    min_pattern_atoms: usize,
    /// Most atoms per pattern.
    max_pattern_atoms: usize,
    /// Most pattern atoms in total across one parent's patterns.
    max_total_atoms: usize,
    /// Sampling seed for pattern extraction.
    extraction_seed: u64,
    /// Pattern source; absent in pre-task checkpoints.
    #[serde(default)]
    pattern_source: Option<PatternSource>,
    /// Fingerprint mode; absent (`None`) in pre-task checkpoints.
    #[serde(default)]
    fingerprint_mode: Option<FingerprintMode>,
    /// Fingerprint token threshold; absent in pre-task checkpoints (0.1).
    #[serde(default = "default_fingerprint_threshold")]
    fingerprint_threshold: f32,
}

impl Clone for CompletionTrainConfig {
    fn clone(&self) -> Self {
        Self {
            lr: self.lr,
            weight_decay: self.weight_decay,
            grad_clip: self.grad_clip,
            seed: self.seed,
            extraction: ExtractionConfig {
                min_patterns: self.extraction.min_patterns,
                max_patterns: self.extraction.max_patterns,
                min_pattern_atoms: self.extraction.min_pattern_atoms,
                max_pattern_atoms: self.extraction.max_pattern_atoms,
                max_total_atoms: self.extraction.max_total_atoms,
            },
            extraction_seed: self.extraction_seed,
            pattern_source: self.pattern_source.clone(),
            fingerprint_mode: self.fingerprint_mode,
            fingerprint_threshold: self.fingerprint_threshold,
        }
    }
}

impl PartialEq for CompletionTrainConfig {
    fn eq(&self, other: &Self) -> bool {
        self.lr == other.lr
            && self.weight_decay == other.weight_decay
            && self.grad_clip == other.grad_clip
            && self.seed == other.seed
            && self.extraction.min_patterns == other.extraction.min_patterns
            && self.extraction.max_patterns == other.extraction.max_patterns
            && self.extraction.min_pattern_atoms == other.extraction.min_pattern_atoms
            && self.extraction.max_pattern_atoms == other.extraction.max_pattern_atoms
            && self.extraction.max_total_atoms == other.extraction.max_total_atoms
            && self.extraction_seed == other.extraction_seed
            && self.pattern_source == other.pattern_source
            && self.fingerprint_mode == other.fingerprint_mode
            && self.fingerprint_threshold == other.fingerprint_threshold
    }
}

impl core::fmt::Debug for CompletionTrainConfig {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CompletionTrainConfig")
            .field("lr", &self.lr)
            .field("weight_decay", &self.weight_decay)
            .field("grad_clip", &self.grad_clip)
            .field("seed", &self.seed)
            .field("min_patterns", &self.extraction.min_patterns)
            .field("max_patterns", &self.extraction.max_patterns)
            .field("min_pattern_atoms", &self.extraction.min_pattern_atoms)
            .field("max_pattern_atoms", &self.extraction.max_pattern_atoms)
            .field("max_total_atoms", &self.extraction.max_total_atoms)
            .field("extraction_seed", &self.extraction_seed)
            .field("pattern_source", &self.pattern_source)
            .field("fingerprint_mode", &self.fingerprint_mode)
            .field("fingerprint_threshold", &self.fingerprint_threshold)
            .finish()
    }
}

impl Serialize for CompletionTrainConfig {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> core::result::Result<S::Ok, S::Error> {
        CompletionTrainConfigFields {
            lr: self.lr,
            weight_decay: self.weight_decay,
            grad_clip: self.grad_clip,
            seed: self.seed,
            min_patterns: self.extraction.min_patterns,
            max_patterns: self.extraction.max_patterns,
            min_pattern_atoms: self.extraction.min_pattern_atoms,
            max_pattern_atoms: self.extraction.max_pattern_atoms,
            max_total_atoms: self.extraction.max_total_atoms,
            extraction_seed: self.extraction_seed,
            pattern_source: Some(self.pattern_source.clone()),
            fingerprint_mode: self.fingerprint_mode,
            fingerprint_threshold: self.fingerprint_threshold,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for CompletionTrainConfig {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> core::result::Result<Self, D::Error> {
        let fields = CompletionTrainConfigFields::deserialize(deserializer)?;
        let extraction = ExtractionConfig {
            min_patterns: fields.min_patterns,
            max_patterns: fields.max_patterns,
            min_pattern_atoms: fields.min_pattern_atoms,
            max_pattern_atoms: fields.max_pattern_atoms,
            max_total_atoms: fields.max_total_atoms,
        };
        let pattern_source = fields
            .pattern_source
            .unwrap_or_else(|| PatternSource::RandomPatches(extraction.clone()));
        Ok(Self {
            lr: fields.lr,
            weight_decay: fields.weight_decay,
            grad_clip: fields.grad_clip,
            seed: fields.seed,
            extraction,
            extraction_seed: fields.extraction_seed,
            pattern_source,
            fingerprint_mode: fields.fingerprint_mode,
            fingerprint_threshold: fields.fingerprint_threshold,
        })
    }
}

impl CompletionTrainConfig {
    /// Check the documented ranges: finite positive `lr`, finite
    /// non-negative `weight_decay`, finite positive `grad_clip` when set,
    /// and valid extraction and pattern-source configs.
    pub fn validate(&self) -> Result<()> {
        if !(self.lr.is_finite() && self.lr > 0.0) {
            return Err(Error::config(format!(
                "CompletionTrainConfig::validate: lr {} is not finite and positive",
                self.lr
            )));
        }
        if !(self.weight_decay.is_finite() && self.weight_decay >= 0.0) {
            return Err(Error::config(format!(
                "CompletionTrainConfig::validate: weight_decay {} is not finite and non-negative",
                self.weight_decay
            )));
        }
        if let Some(clip) = self.grad_clip
            && !(clip.is_finite() && clip > 0.0)
        {
            return Err(Error::config(format!(
                "CompletionTrainConfig::validate: grad_clip {clip} is not finite and positive"
            )));
        }
        self.extraction.validate()?;
        match &self.pattern_source {
            PatternSource::RandomPatches(config) => config.validate()?,
            PatternSource::FunctionalGroups(config) => config.validate()?,
        }
        if !(self.fingerprint_threshold.is_finite()
            && self.fingerprint_threshold > 0.0
            && self.fingerprint_threshold <= 1.0)
        {
            return Err(Error::config(format!(
                "CompletionTrainConfig::validate: fingerprint_threshold {} is not in (0, 1]",
                self.fingerprint_threshold
            )));
        }
        Ok(())
    }
}

/// Fit provenance of [`FormulaArtifacts`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FormulaFitInfo {
    /// Training molecules the domain and bounds were fitted on.
    pub molecules: u64,
    /// Domain margin (`0` from the experiment driver).
    pub margin: u16,
    /// Bounds widening margin (`0` from the experiment driver).
    pub quantile_margin: u16,
    /// Free-text fit source (for example the train export name).
    pub source: String,
}

/// One training composition count behind [`FormulaArtifacts`]: the canonical
/// formula text (see [`artifact_formula_text`]) with the number of training
/// molecules carrying exactly that composition.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompositionCount {
    /// Canonical formula text (for example `C2H6O`).
    pub formula: String,
    /// Training molecules with exactly this composition.
    pub count: u64,
}

/// Canonical formula text for [`CompositionCount`]: C, always H (even zero),
/// then N, O, F, P, S, Cl, Br, I when present, count `1` omitted.
///
/// This must render exactly like
/// [`formula_text`](super::completion_formula::formula_text) (which cannot be
/// shared: that module depends on this one); the model test
/// `artifact_text_matches_formula_text` checks them against each other.
fn artifact_formula_text(c: &Composition) -> String {
    let suffix = |symbol: &str, count: u16| {
        if count == 1 {
            symbol.to_string()
        } else {
            format!("{symbol}{count}")
        }
    };
    const SYMBOLS: [&str; 10] = ["C", "H", "N", "O", "F", "P", "S", "Cl", "Br", "I"];
    let mut out = String::new();
    if c[0] > 0 {
        out.push_str(&suffix(SYMBOLS[0], c[0]));
    }
    out.push_str(&suffix(SYMBOLS[1], c[1]));
    for e in [2usize, 3, 4, 5, 6, 7, 8, 9] {
        if c[e] > 0 {
            out.push_str(&suffix(SYMBOLS[e], c[e]));
        }
    }
    out
}

/// Formula-search artifacts bound to a checkpoint.
///
/// The domain and bounds are fitted on the training compositions only; the
/// mass path enumerates with every exact chemical filter on and these bounds.
/// There is no learned formula ranker: formulas are ordered by absolute mass
/// residual with no learned prior.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FormulaArtifacts {
    /// Enumeration domain fitted on the training compositions.
    pub domain: EnumDomain,
    /// Train-fit pruning bounds fitted on the training compositions.
    pub bounds: RatioBounds,
    /// SHA-256 (hex) of the domain JSON.
    pub domain_sha256: String,
    /// SHA-256 (hex) of the bounds JSON.
    pub bounds_sha256: String,
    /// Fit provenance.
    pub fit: FormulaFitInfo,
    /// Training composition counts as a compact list sorted by formula text:
    /// one entry per distinct training composition. Used only as an explicit
    /// training-frequency prior for the `train_frequency` trajectory
    /// allocation (weight `(count + 1) / sum(count + 1)` over the selected
    /// formulas), never as a calibrated probability. Absent (`None`) in old
    /// checkpoints, which load with an empty list.
    #[serde(default)]
    pub composition_counts: Vec<CompositionCount>,
}

impl FormulaArtifacts {
    /// Build artifacts from training compositions, clamping the domain heavy
    /// total to `max_atoms`.
    pub fn fit(
        compositions: &[Composition],
        max_atoms: u32,
        margin: u16,
        quantile_margin: u16,
        source: String,
    ) -> Result<Self> {
        let mut domain = EnumDomain::from_compositions(compositions.iter().copied(), margin)?;
        let heavy_cap = max_atoms.min(u32::from(u16::MAX)) as u16;
        if domain.heavy_max > heavy_cap {
            domain.heavy_max = heavy_cap;
        }
        let bounds = RatioBounds::fit(compositions.iter().copied(), quantile_margin)?;
        let domain_json = domain.to_json();
        let bounds_json = bounds.to_json();
        let domain_sha256 = super::experiment::sha256_hex(domain_json.as_bytes());
        let bounds_sha256 = super::experiment::sha256_hex(bounds_json.as_bytes());
        // Training composition counts, sorted by formula text.
        let mut counts: std::collections::BTreeMap<String, u64> =
            std::collections::BTreeMap::new();
        for c in compositions {
            *counts.entry(artifact_formula_text(c)).or_default() += 1;
        }
        let composition_counts = counts
            .into_iter()
            .map(|(formula, count)| CompositionCount { formula, count })
            .collect();
        Ok(Self {
            domain,
            bounds,
            domain_sha256,
            bounds_sha256,
            fit: FormulaFitInfo {
                molecules: compositions.len() as u64,
                margin,
                quantile_margin,
                source,
            },
            composition_counts,
        })
    }

    /// Training count of `formula` (canonical text, `0` when the training set
    /// never saw it). The list is sorted by text, so this is a binary search.
    pub fn composition_count(&self, formula: &str) -> u64 {
        self.composition_counts
            .binary_search_by(|entry| entry.formula.as_str().cmp(formula))
            .map(|i| self.composition_counts[i].count)
            .unwrap_or(0)
    }
}

/// The completion trainer: the model, its optimizer and the resident grammar
/// constants.
///
/// Like [`Ms2Trainer`](super::train::Ms2Trainer), a step runs three phases —
/// forward pass and report packing, gradient computation with the device-side
/// clip scale, optimizer update — and performs no device read unless
/// [`CompletionTrainer::request_report`] was called, in which case exactly
/// one batched read of the loss is performed.
pub struct CompletionTrainer<R: Runtime, E: FloatElem> {
    /// The completion-conditioned model.
    model: CompletionModel<R, E>,
    /// AdamW over every model parameter.
    optimizer: AdamW<R, E>,
    /// Parameters in named-parameter order, for the optimizer step.
    params: Vec<Param<R, E>>,
    /// Training hyperparameters.
    train: CompletionTrainConfig,
    /// Grammar limits from the model config.
    limits: Limits,
    /// Resident wavelength and atom-type tables of the replay kernel.
    constants: Ms2Constants<R>,
    /// Device the model lives on.
    device: Device<R>,
    /// Whether the next [`CompletionTrainer::step`] reports its loss.
    report_pending: bool,
    /// Optimizer steps completed so far.
    steps: u64,
    /// Optional formula-search artifacts bound to the checkpoint.
    formula_artifacts: Option<FormulaArtifacts>,
    /// True fingerprint bits by molecule index (set by the driver when the
    /// model owns the encoder).
    fingerprint_store: Option<FingerprintStore>,
    /// MIST noise histograms for `MistLike` sampling.
    fingerprint_noise: Option<FingerprintNoise>,
    /// Which noise histogram set `MistLike` sampling draws from (set by the
    /// driver with the run's `--fp-noise-level`; not part of the checkpoint
    /// header, so every run sets it explicitly after load).
    fingerprint_noise_level: FingerprintNoiseLevel,
}

impl<R: Runtime, E: FloatElem> CompletionTrainer<R, E> {
    /// Build the model and prepare the optimizer. Validates both configs and
    /// the neural dtype before any allocation, upload or launch.
    pub fn new(
        model_config: &CompletionModelConfig,
        train_config: &CompletionTrainConfig,
        device: &Device<R>,
    ) -> Result<Self> {
        train_config.validate()?;
        let mut rng = Rng::seeded(train_config.seed);
        let model = CompletionModel::init(model_config, device, &mut rng)?;
        let limits = Limits::new(
            model_config.max_atoms as usize,
            model_config.max_ring_closures as usize,
        )
        .map_err(|e| {
            Error::config(format!(
                "CompletionTrainer::new: model limits rejected: {e}"
            ))
        })?;
        let optimizer = AdamWConfig {
            learning_rate: train_config.lr,
            weight_decay: train_config.weight_decay,
            ..Default::default()
        }
        .init();
        let params = model
            .named_parameters()
            .into_iter()
            .map(|(_, p)| p)
            .collect();
        Ok(Self {
            model,
            optimizer,
            params,
            train: train_config.clone(),
            limits,
            constants: Ms2Constants::new(device),
            device: device.clone(),
            report_pending: false,
            steps: 0,
            formula_artifacts: None,
            fingerprint_store: None,
            fingerprint_noise: None,
            fingerprint_noise_level: FingerprintNoiseLevel::default(),
        })
    }

    /// The model being trained.
    pub fn model(&self) -> &CompletionModel<R, E> {
        &self.model
    }

    /// The effective training configuration: the checkpoint's own config
    /// after [`load`](Self::load), the fresh config otherwise. Continued
    /// training always follows this, never the resume CLI values (see the
    /// resume-conflict check in the experiment driver).
    pub fn train_config(&self) -> &CompletionTrainConfig {
        &self.train
    }

    /// Optimizer steps completed so far.
    pub fn step_count(&self) -> u64 {
        self.steps
    }

    /// Formula-search artifacts bound to this trainer, if any.
    pub fn formula_artifacts(&self) -> Option<&FormulaArtifacts> {
        self.formula_artifacts.as_ref()
    }

    /// Attach formula-search artifacts fitted on the training compositions.
    ///
    /// Overwrites any previous block; the weights are untouched, so an
    /// already-trained checkpoint gains the block without retraining.
    pub fn set_formula_artifacts(&mut self, artifacts: FormulaArtifacts) {
        self.formula_artifacts = Some(artifacts);
    }

    /// Attach fingerprint evidence for training/evaluation with a fingerprint
    /// encoder: true on-bit lists by molecule index. Overwrites any previous
    /// store. Required by [`step`](Self::step) when the model owns the
    /// encoder; ignored otherwise.
    pub fn set_fingerprint_store(&mut self, store: FingerprintStore) {
        self.fingerprint_store = Some(store);
    }

    /// Attach the MIST noise histograms behind `MistLike` sampling.
    /// Overwrites any previous noise. Required when the train config's
    /// `fingerprint_mode` is `MistLike`.
    pub fn set_fingerprint_noise(&mut self, noise: FingerprintNoise) {
        self.fingerprint_noise = Some(noise);
    }

    /// Select the noise histogram set `MistLike` sampling draws from (the
    /// run's `--fp-noise-level`; default `spectrum`). Not persisted in the
    /// checkpoint: the driver sets it on every run after load.
    pub fn set_fingerprint_noise_level(&mut self, level: FingerprintNoiseLevel) {
        self.fingerprint_noise_level = level;
    }

    /// The selected noise histogram set.
    pub fn fingerprint_noise_level(&self) -> FingerprintNoiseLevel {
        self.fingerprint_noise_level
    }

    /// Ask the next [`CompletionTrainer::step`] to report its loss (exactly
    /// one batched read of the pre-update loss).
    pub fn request_report(&mut self) {
        self.report_pending = true;
    }

    /// The forward half of a training step: exact targets, substructure
    /// encoding, replay, teacher forcing and the graph loss, plus the
    /// single-scalar report packing. No device read.
    #[allow(clippy::type_complexity)]
    fn forward_batch(
        &self,
        patterns: &[&[MolGraph]],
        traces: &[&[Token]],
        compositions: &[Composition],
    ) -> Result<(TeacherOutput<R, E>, Var<R, E>, Var<R, E>)> {
        if patterns.is_empty() {
            return Err(Error::config(
                "CompletionTrainer: cannot train on an empty query list".to_string(),
            ));
        }
        let batch = PatternBatch::build(patterns, compositions)?;
        let targets = TargetBatch::build_exact(traces, compositions, self.limits)?;
        let (out, loss) = self
            .model
            .teacher(&batch, &targets, &self.constants, &self.device)?;
        let packed = loss.reshape(vec![1])?;
        Ok((out, loss, packed))
    }

    /// The forward half with fingerprint evidence: like
    /// [`forward_batch`](Self::forward_batch) but encoding `fingerprints`
    /// (one per query) with the patterns. No device read.
    #[allow(clippy::type_complexity)]
    fn forward_batch_with_fingerprints(
        &self,
        patterns: &[&[MolGraph]],
        fingerprints: &[SparseFingerprint],
        traces: &[&[Token]],
        compositions: &[Composition],
    ) -> Result<(TeacherOutput<R, E>, Var<R, E>, Var<R, E>)> {
        if patterns.is_empty() {
            return Err(Error::config(
                "CompletionTrainer: cannot train on an empty query list".to_string(),
            ));
        }
        if patterns.len() != fingerprints.len() {
            return Err(Error::config(format!(
                "CompletionTrainer: {} pattern entries for {} fingerprints",
                patterns.len(),
                fingerprints.len()
            )));
        }
        let batch = PatternBatch::build(patterns, compositions)?;
        let slots = self.model.fingerprint_slots();
        let fp_batch = FingerprintBatch::build(fingerprints, slots)?;
        let targets = TargetBatch::build_exact(traces, compositions, self.limits)?;
        let (out, loss) = self.model.teacher_with_fingerprints(
            &batch,
            &fp_batch,
            &targets,
            &self.constants,
            &self.device,
        )?;
        let packed = loss.reshape(vec![1])?;
        Ok((out, loss, packed))
    }

    /// Fingerprint evidence of one example under the train config: exact true
    /// bits at probability 1, or a MIST-like sample from the attached noise.
    /// Requires the attached store (by `source_index`); `MistLike` requires
    /// the attached noise. Deterministic in (`extraction_seed`, key, `draw`).
    fn fingerprint_for_example(
        &self,
        example: &CompletionExample,
        draw: u64,
    ) -> Result<SparseFingerprint> {
        let mode = self.train.fingerprint_mode.ok_or_else(|| {
            Error::config(
                "CompletionTrainer: model owns a fingerprint encoder but the train config sets no fingerprint_mode".to_string(),
            )
        })?;
        let store = self.fingerprint_store.as_ref().ok_or_else(|| {
            Error::config(
                "CompletionTrainer: model owns a fingerprint encoder but no fingerprint store was attached (set_fingerprint_store)".to_string(),
            )
        })?;
        let true_bits = store.get_by_index(example.source_index)?;
        match mode {
            FingerprintMode::Exact => SparseFingerprint::from_bits(true_bits),
            FingerprintMode::MistLike => {
                let noise = self.fingerprint_noise.as_ref().ok_or_else(|| {
                    Error::config(
                        "CompletionTrainer: fingerprint_mode is mist_like but no fingerprint noise was attached (set_fingerprint_noise)".to_string(),
                    )
                })?;
                noise.sample_at_level(
                    true_bits,
                    self.train.extraction_seed,
                    &example.key,
                    draw,
                    self.train.fingerprint_threshold,
                    self.fingerprint_noise_level,
                )
            }
        }
    }

    /// Fingerprints of these examples in order (see
    /// [`fingerprint_for_example`](Self::fingerprint_for_example)).
    fn fingerprints_for(
        &self,
        set: &CompletionSet,
        indices: &[usize],
        draw: u64,
    ) -> Result<Vec<SparseFingerprint>> {
        let mut out = Vec::with_capacity(indices.len());
        for &i in indices {
            let example = set.examples.get(i).ok_or_else(|| {
                Error::config(format!(
                    "CompletionTrainer: example index {i} outside {} examples",
                    set.examples.len()
                ))
            })?;
            out.push(self.fingerprint_for_example(example, draw)?);
        }
        Ok(out)
    }

    /// One optimizer step on caller-supplied patterns, traces and
    /// compositions: the shared core behind [`CompletionTrainer::step`]
    /// (which extracts the patterns from a [`CompletionSet`]) and the path
    /// tests and controls use for fixed patterns.
    ///
    /// No device read happens unless [`CompletionTrainer::request_report`]
    /// was called, in which case exactly one batched read of the pre-update
    /// loss is performed and the loss is returned.
    pub fn step_with(
        &mut self,
        patterns: &[&[MolGraph]],
        traces: &[&[Token]],
        compositions: &[Composition],
    ) -> Result<Option<f32>> {
        // Forward pass and report packing.
        let (_, loss, packed) = self.forward_batch(patterns, traces, compositions)?;
        // Gradients with the device-side clip scale.
        let grads = loss.backward_retain()?;
        let scale = match self.train.grad_clip {
            Some(max_norm) => grad_scale(&grads, max_norm, 1.0)?,
            None => None,
        };
        // Optimizer update.
        match &scale {
            Some(scale) => {
                self.optimizer
                    .step_scaled(&self.params, &grads, Some(&scale.factor))?;
            }
            None => {
                self.optimizer.step(&self.params, &grads)?;
            }
        }
        // Close out: count the step, and read back the report only when
        // requested. This is the only phase that ever reads the device.
        self.steps += 1;
        if !self.report_pending {
            return Ok(None);
        }
        self.report_pending = false;
        Ok(Some(packed.try_to_f32()?[0]))
    }

    /// One optimizer step on caller-supplied patterns, fingerprints, traces
    /// and compositions. No device read happens unless
    /// [`request_report`](Self::request_report) was called.
    pub fn step_with_fingerprints(
        &mut self,
        patterns: &[&[MolGraph]],
        fingerprints: &[SparseFingerprint],
        traces: &[&[Token]],
        compositions: &[Composition],
    ) -> Result<Option<f32>> {
        let (_, loss, packed) =
            self.forward_batch_with_fingerprints(patterns, fingerprints, traces, compositions)?;
        let grads = loss.backward_retain()?;
        let scale = match self.train.grad_clip {
            Some(max_norm) => grad_scale(&grads, max_norm, 1.0)?,
            None => None,
        };
        match &scale {
            Some(scale) => {
                self.optimizer
                    .step_scaled(&self.params, &grads, Some(&scale.factor))?;
            }
            None => {
                self.optimizer.step(&self.params, &grads)?;
            }
        }
        self.steps += 1;
        if !self.report_pending {
            return Ok(None);
        }
        self.report_pending = false;
        Ok(Some(packed.try_to_f32()?[0]))
    }

    /// One optimizer step on these examples: the patterns of example `i` are
    /// `set.examples[i].patterns_from(&pattern_source, extraction_seed, draw)`,
    /// and the traces and compositions are the examples' own. Molecules
    /// without any functional group train with an empty pattern set. Returns
    /// the pre-update loss only when [`CompletionTrainer::request_report`]
    /// was called before (`None` otherwise, with no device read at all).
    ///
    /// With a fingerprint encoder the fingerprints come from the attached
    /// store under the train config's mode (see
    /// [`fingerprint_for_example`](Self::fingerprint_for_example)); without
    /// one the patterns-only path runs as before.
    pub fn step(
        &mut self,
        set: &CompletionSet,
        indices: &[usize],
        draw: u64,
    ) -> Result<Option<f32>> {
        if indices.is_empty() {
            return Err(Error::config(
                "CompletionTrainer::step: cannot train on an empty index list".to_string(),
            ));
        }
        let mut owned: Vec<Vec<MolGraph>> = Vec::with_capacity(indices.len());
        let mut traces: Vec<&[Token]> = Vec::with_capacity(indices.len());
        let mut compositions: Vec<Composition> = Vec::with_capacity(indices.len());
        for &i in indices {
            let example = set.examples.get(i).ok_or_else(|| {
                Error::config(format!(
                    "CompletionTrainer::step: example index {i} outside {} examples",
                    set.examples.len()
                ))
            })?;
            owned.push(
                example
                    .patterns_from(
                        &self.train.pattern_source,
                        self.train.extraction_seed,
                        draw,
                    )?
                    .patterns
                    .into_iter()
                    .map(|pattern| pattern.graph)
                    .collect(),
            );
            traces.push(&example.trace);
            compositions.push(example.composition);
        }
        let refs: Vec<&[MolGraph]> = owned.iter().map(Vec::as_slice).collect();
        if self.model.has_fingerprint() {
            let fps = self.fingerprints_for(set, indices, draw)?;
            return self.step_with_fingerprints(&refs, &fps, &traces, &compositions);
        }
        self.step_with(&refs, &traces, &compositions)
    }

    /// Teacher-forced evaluation of caller-supplied patterns, traces and
    /// compositions: the per-query summed NLL under `no_grad`, in one
    /// batched read. Tests and controls use it; [`CompletionTrainer::step`]
    /// training pairs go through [`CompletionTrainer::teacher_eval`].
    pub fn teacher_eval_with(
        &mut self,
        patterns: &[&[MolGraph]],
        traces: &[&[Token]],
        compositions: &[Composition],
    ) -> Result<Vec<f32>> {
        let _guard = no_grad();
        let (out, _, _) = self.forward_batch(patterns, traces, compositions)?;
        out.nll.try_to_f32()
    }

    /// Teacher-forced evaluation with fingerprint evidence: like
    /// [`teacher_eval_with`](Self::teacher_eval_with) but encoding
    /// `fingerprints` with the patterns.
    pub fn teacher_eval_with_fingerprints(
        &mut self,
        patterns: &[&[MolGraph]],
        fingerprints: &[SparseFingerprint],
        traces: &[&[Token]],
        compositions: &[Composition],
    ) -> Result<Vec<f32>> {
        let _guard = no_grad();
        let (out, _, _) =
            self.forward_batch_with_fingerprints(patterns, fingerprints, traces, compositions)?;
        out.nll.try_to_f32()
    }

    /// Teacher-forced evaluation of these examples with the trainer's
    /// extraction at `draw`: the per-example summed NLL under `no_grad`, in
    /// one batched read.
    pub fn teacher_eval(
        &mut self,
        set: &CompletionSet,
        indices: &[usize],
        draw: u64,
    ) -> Result<Vec<f32>> {
        if indices.is_empty() {
            return Err(Error::config(
                "CompletionTrainer::teacher_eval: cannot evaluate an empty index list".to_string(),
            ));
        }
        let mut owned: Vec<Vec<MolGraph>> = Vec::with_capacity(indices.len());
        let mut traces: Vec<&[Token]> = Vec::with_capacity(indices.len());
        let mut compositions: Vec<Composition> = Vec::with_capacity(indices.len());
        for &i in indices {
            let example = set.examples.get(i).ok_or_else(|| {
                Error::config(format!(
                    "CompletionTrainer::teacher_eval: example index {i} outside {} examples",
                    set.examples.len()
                ))
            })?;
            owned.push(
                example
                    .patterns(&self.train.extraction, self.train.extraction_seed, draw)?
                    .into_iter()
                    .map(|pattern| pattern.graph)
                    .collect(),
            );
            traces.push(&example.trace);
            compositions.push(example.composition);
        }
        let refs: Vec<&[MolGraph]> = owned.iter().map(Vec::as_slice).collect();
        if self.model.has_fingerprint() {
            let fps = self.fingerprints_for(set, indices, draw)?;
            return self.teacher_eval_with_fingerprints(&refs, &fps, &traces, &compositions);
        }
        self.teacher_eval_with(&refs, &traces, &compositions)
    }

    /// Save the weights plus a JSON header (`format`, the model and train
    /// configs, the step count, and the optional formula artifacts).
    ///
    /// The optimizer state is not stored (as in [`Ms2Trainer`](super::train::Ms2Trainer),
    /// whose `save` documents the same): [`CompletionTrainer::load`]
    /// rebuilds a fresh AdamW, so bias correction restarts. This is said in
    /// the file too.
    ///
    /// The write is atomic: the bytes go to a sibling temporary file in the
    /// same directory (flushed and `sync_all`-ed), which is then renamed over
    /// `path`. On any error the temporary file is removed and `path` is left
    /// untouched, so a full disk or an interruption never truncates a
    /// previous good checkpoint.
    pub fn save(&self, path: &Path) -> Result<()> {
        let checkpoint = CompletionCheckpoint {
            format: COMPLETION_CHECKPOINT_FORMAT.to_string(),
            note: "Completion checkpoint: weights, model config and train config. The optimizer state is not stored; load rebuilds a fresh AdamW."
                .to_string(),
            model_config: self.model.config.clone(),
            train_config: self.train.clone(),
            steps: self.steps,
            weights: self.model.state_dict(),
            formula_artifacts: self.formula_artifacts.clone(),
        };
        let text = serde_json::to_string_pretty(&checkpoint)?;
        let parent = match path.parent() {
            Some(p) if !p.as_os_str().is_empty() => p.to_path_buf(),
            _ => std::path::PathBuf::from("."),
        };
        let stem = path
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "checkpoint".to_string());
        static SAVE_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let uniq = SAVE_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let temp = parent.join(format!(".{stem}.tmp-{}-{uniq}", std::process::id()));
        let write_result = (|| {
            use std::io::Write;
            let mut file = std::fs::File::create(&temp)?;
            file.write_all(text.as_bytes())?;
            file.flush()?;
            file.sync_all()?;
            drop(file);
            std::fs::rename(&temp, path)?;
            Ok::<(), crate::error::Error>(())
        })();
        if write_result.is_err() {
            let _ = std::fs::remove_file(&temp);
        }
        write_result?;
        Ok(())
    }

    /// Load a checkpoint saved by [`CompletionTrainer::save`] onto `device`.
    ///
    /// A different header format or config version is [`Error::Config`]; the
    /// weights must match the rebuilt model exactly (strict load). The
    /// optimizer is fresh, not restored (see [`CompletionTrainer::save`]).
    pub fn load(path: &Path, device: &Device<R>) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        Self::load_bytes(&bytes, device)
    }

    /// Load a checkpoint from already-read bytes onto `device`.
    ///
    /// [`CompletionTrainer::load`] reads the file once and calls this, and
    /// [`CompletionService`](super::completion_api::CompletionService::load)
    /// hashes and deserializes the same buffer, so a concurrent replacement
    /// cannot make the provenance hash describe other bytes. A truncated or
    /// foreign buffer is an error, never a panic.
    pub fn load_bytes(bytes: &[u8], device: &Device<R>) -> Result<Self> {
        let checkpoint: CompletionCheckpoint = serde_json::from_slice(bytes)?;
        if checkpoint.format != COMPLETION_CHECKPOINT_FORMAT {
            return Err(Error::config(format!(
                "CompletionTrainer::load: unknown format {:?} (expected {COMPLETION_CHECKPOINT_FORMAT:?})",
                checkpoint.format
            )));
        }
        checkpoint.model_config.validate().map_err(|e| {
            Error::config(format!(
                "CompletionTrainer::load: checkpoint model config rejected: {e}"
            ))
        })?;
        let mut trainer = Self::new(&checkpoint.model_config, &checkpoint.train_config, device)?;
        trainer.model.load_state_dict(&checkpoint.weights, true)?;
        trainer.steps = checkpoint.steps;
        trainer.formula_artifacts = checkpoint.formula_artifacts;
        Ok(trainer)
    }
}

/// A saved [`CompletionTrainer`]: weights plus the JSON header (format, model
/// and train configs, step count, optional formula artifacts) that binds
/// them. The optimizer state is deliberately absent.
#[derive(Serialize, Deserialize)]
struct CompletionCheckpoint {
    /// Checkpoint header format ([`COMPLETION_CHECKPOINT_FORMAT`]).
    format: String,
    /// Human-readable note, including the missing optimizer state.
    note: String,
    /// Model hyperparameters.
    model_config: CompletionModelConfig,
    /// Training hyperparameters.
    train_config: CompletionTrainConfig,
    /// Optimizer steps completed when saved.
    steps: u64,
    /// Model weights by parameter path.
    weights: StateDict,
    /// Optional formula-search artifacts bound to the checkpoint. Absent
    /// (`None`) in old checkpoints, which still load.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    formula_artifacts: Option<FormulaArtifacts>,
}

// ---------------------------------------------------------------------------
// Exact-rule sampling with host acceptance (`completion-generation-v1`)
// ---------------------------------------------------------------------------

/// How the supplied substructures constrain host acceptance in
/// [`CompletionModel::generate`](CompletionModel::generate).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SubstructureSemantics {
    /// Every pattern is contained somewhere; patterns may share atoms. This
    /// is the historical rule and the default.
    #[default]
    Contained,
    /// The patterns are distinct occurrences: one joint injective embedding
    /// maps them onto pairwise disjoint atom sets.
    DisjointOccurrences,
    /// The patterns are the complete list of the molecule's functional
    /// groups (`functional-groups-ertl-v1`): the candidate's own functional
    /// groups, as a multiset of typed graphs up to isomorphism, equal the
    /// supplied multiset. This implies
    /// [`DisjointOccurrences`](SubstructureSemantics::DisjointOccurrences)
    /// (equal multisets of whole groups are disjoint by construction); only
    /// the multiset comparison runs, never both.
    CompleteFunctionalGroups,
}

impl SubstructureSemantics {
    /// The protocol spelling of the semantics.
    pub fn as_str(self) -> &'static str {
        match self {
            SubstructureSemantics::Contained => "contained",
            SubstructureSemantics::DisjointOccurrences => "disjoint_occurrences",
            SubstructureSemantics::CompleteFunctionalGroups => "complete_functional_groups",
        }
    }

    /// Parse a protocol spelling (see [`as_str`](SubstructureSemantics::as_str));
    /// `None` for anything else.
    pub fn parse(text: &str) -> Option<Self> {
        match text {
            "contained" => Some(SubstructureSemantics::Contained),
            "disjoint_occurrences" => Some(SubstructureSemantics::DisjointOccurrences),
            "complete_functional_groups" => Some(SubstructureSemantics::CompleteFunctionalGroups),
            _ => None,
        }
    }

    /// One-line meaning of the semantics, echoed in API responses.
    pub fn meaning(self) -> &'static str {
        match self {
            SubstructureSemantics::Contained => {
                "every pattern is contained somewhere; patterns may share atoms"
            }
            SubstructureSemantics::DisjointOccurrences => {
                "patterns are distinct occurrences on pairwise disjoint atom sets"
            }
            SubstructureSemantics::CompleteFunctionalGroups => {
                "patterns are the molecule's complete functional-group list (functional-groups-ertl-v1)"
            }
        }
    }
}

/// Host acceptance verdict of one finished candidate under one
/// [`SubstructureSemantics`]; see [`accepts`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Acceptance {
    /// The candidate satisfies the rule.
    Accepted,
    /// A required pattern is not contained (the `Contained` and
    /// `DisjointOccurrences` rules).
    RejectedContainment,
    /// Under `CompleteFunctionalGroups`: every supplied group is present
    /// disjointly, but the candidate carries additional functional groups.
    RejectedExtraGroups,
    /// Under `CompleteFunctionalGroups`: the candidate lacks a supplied
    /// group (its group multiset does not cover the supplied one).
    RejectedMissingGroups,
    /// A containment or identity check hit its work limit: never acceptance.
    ContainmentUnresolved,
}

/// Multiset comparison behind the `CompleteFunctionalGroups` rule.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CompleteVerdict {
    /// The candidate's groups equal the supplied patterns one-to-one.
    Equal,
    /// Every supplied group is present disjointly, with groups left over.
    Extra,
    /// Some supplied group has no disjoint identical group.
    Missing,
    /// An identity comparison hit its work limit: never acceptance.
    Unresolved,
}

/// Multiset comparison of the candidate's own functional groups against the
/// supplied patterns: each side's groups become typed induced subgraphs,
/// bucketed by the cheap invariant behind [`identity_bucket`] (sorted atom
/// types plus sorted bond triples; different buckets are certainly
/// non-identical), then matched one-to-one with
/// [`same_identity`](super::completion_data::same_identity) per pair.
///
/// A certified full match is `Equal` (same length) or `Extra` (candidate
/// groups left over); no certified full match is `Missing`, unless some
/// comparison was unresolved, which is `Unresolved`. Whether the supplied
/// patterns are what the extraction would produce from any molecule (for
/// example a pattern containing an unmarked atom) is not this function's
/// concern: it compares multisets.
fn complete_verdict(
    candidate: &MolGraph,
    patterns: &[MolGraph],
    identity_work_limit: usize,
) -> CompleteVerdict {
    let groups = match functional_groups(candidate) {
        Ok(groups) => groups,
        Err(_) => return CompleteVerdict::Unresolved,
    };
    let mut group_graphs: Vec<MolGraph> = Vec::with_capacity(groups.len());
    for group in &groups {
        match candidate.induced(&group.atoms) {
            Ok(graph) => group_graphs.push(graph),
            Err(_) => return CompleteVerdict::Unresolved,
        }
    }
    let group_buckets: Vec<_> = group_graphs.iter().map(identity_bucket).collect();
    let pattern_buckets: Vec<_> = patterns.iter().map(identity_bucket).collect();
    let mut used = vec![false; group_graphs.len()];
    let mut saw_unresolved = false;
    let mut all_matched = true;
    for (pattern, pattern_bucket) in patterns.iter().zip(pattern_buckets.iter()) {
        let mut matched = false;
        for (index, group) in group_graphs.iter().enumerate() {
            if used[index] || group_buckets[index] != *pattern_bucket {
                continue;
            }
            match same_identity(pattern, group, identity_work_limit) {
                Some(true) => {
                    used[index] = true;
                    matched = true;
                    break;
                }
                Some(false) => {}
                None => {
                    saw_unresolved = true;
                }
            }
        }
        if !matched {
            all_matched = false;
        }
    }
    if all_matched {
        if group_graphs.len() == patterns.len() {
            CompleteVerdict::Equal
        } else {
            CompleteVerdict::Extra
        }
    } else if saw_unresolved {
        CompleteVerdict::Unresolved
    } else {
        CompleteVerdict::Missing
    }
}

/// Host acceptance of one finished candidate under `semantics`.
///
/// `Contained` checks every pattern independently with
/// [`contains_pattern`](super::completion::contains_pattern) (first failure
/// in pattern order wins, as before); `DisjointOccurrences` runs one joint
/// injective embedding with disjoint images through the audit's joint search
/// ([`contains_patterns_disjoint`](super::completion::contains_patterns_disjoint));
/// `CompleteFunctionalGroups` compares the candidate's own functional-group
/// multiset against the supplied one. A work-limit outcome is
/// [`ContainmentUnresolved`](Acceptance::ContainmentUnresolved), never
/// acceptance. Pure host computation: no device work, deterministic.
pub fn accepts(
    candidate: &MolGraph,
    patterns: &[MolGraph],
    semantics: SubstructureSemantics,
    containment_node_limit: usize,
    identity_work_limit: usize,
) -> Acceptance {
    match semantics {
        SubstructureSemantics::Contained => {
            for pattern in patterns {
                match contains_pattern(candidate, pattern, containment_node_limit) {
                    Containment::Contained => {}
                    Containment::NotContained => return Acceptance::RejectedContainment,
                    Containment::WorkLimit => return Acceptance::ContainmentUnresolved,
                }
            }
            Acceptance::Accepted
        }
        SubstructureSemantics::DisjointOccurrences => {
            match contains_patterns_disjoint(candidate, patterns, containment_node_limit) {
                Containment::Contained => Acceptance::Accepted,
                Containment::NotContained => Acceptance::RejectedContainment,
                Containment::WorkLimit => Acceptance::ContainmentUnresolved,
            }
        }
        SubstructureSemantics::CompleteFunctionalGroups => {
            match complete_verdict(candidate, patterns, identity_work_limit) {
                CompleteVerdict::Equal => Acceptance::Accepted,
                CompleteVerdict::Extra => Acceptance::RejectedExtraGroups,
                CompleteVerdict::Missing => Acceptance::RejectedMissingGroups,
                CompleteVerdict::Unresolved => Acceptance::ContainmentUnresolved,
            }
        }
    }
}

/// Necessary composition fit of the supplied patterns under `semantics`,
/// checked per request before sampling.
///
/// `None` means feasible; `Some(reason)` means no molecule with
/// `composition` can satisfy the rule, so the request needs no device work.
/// The `Contained` rule never fails this check (patterns may share atoms).
/// Under `DisjointOccurrences` and `CompleteFunctionalGroups` the summed
/// element counts and hydrogens of all patterns must fit the composition
/// (summed, because the occurrences are disjoint); under
/// `CompleteFunctionalGroups` every heteroatom of the composition must
/// additionally be accounted for by the patterns
/// (`functional-groups-ertl-v1` marks every heteroatom), so the patterns'
/// heteroatom counts must equal the composition's. Pure host computation.
pub fn check_feasibility(
    composition: &Composition,
    patterns: &[MolGraph],
    semantics: SubstructureSemantics,
) -> Option<String> {
    if semantics == SubstructureSemantics::Contained {
        return None;
    }
    let mut need_heavy = [0u32; 10];
    let mut need_h = 0u32;
    for (index, pattern) in patterns.iter().enumerate() {
        for &id in pattern.atoms() {
            let Some(atom) = atom_type(id) else {
                return Some(format!(
                    "infeasible: pattern {index} uses unknown atom type id {id}"
                ));
            };
            if atom.element == HYDROGEN {
                need_h += 1;
            } else {
                need_heavy[atom.element] += 1;
                need_h += u32::from(atom.hydrogens);
            }
        }
    }
    const SYMBOLS: [&str; 10] = ["C", "H", "N", "O", "F", "P", "S", "Cl", "Br", "I"];
    for e in [0usize, 2, 3, 4, 5, 6, 7, 8, 9] {
        if need_heavy[e] > u32::from(composition[e]) {
            return Some(format!(
                "infeasible: summed pattern {} count {} exceeds composition {} count {} (disjoint occurrences consume disjoint atoms)",
                SYMBOLS[e], need_heavy[e], SYMBOLS[e], composition[e]
            ));
        }
    }
    if need_h > u32::from(composition[HYDROGEN]) {
        return Some(format!(
            "infeasible: summed pattern hydrogen count {need_h} exceeds composition hydrogen count {} (disjoint occurrences consume disjoint atoms)",
            composition[HYDROGEN]
        ));
    }
    if semantics == SubstructureSemantics::CompleteFunctionalGroups {
        for e in [2usize, 3, 4, 5, 6, 7, 8, 9] {
            if need_heavy[e] != u32::from(composition[e]) {
                return Some(format!(
                    "infeasible: summed pattern {} count {} does not equal composition {} count {} (functional-groups-ertl-v1 marks every heteroatom)",
                    SYMBOLS[e], need_heavy[e], SYMBOLS[e], composition[e]
                ));
            }
        }
    }
    None
}

/// One completion query: an exact target composition with required
/// substructures.
///
/// The `id` keys the sampler's random draws (like the spectrum id in
/// [`Ms2Model::generate`](super::generate::Ms2Model::generate)): the same
/// `(id, seed, temperature)` always samples the same traces. Each `id` is
/// mixed through one SplitMix64 round before its low/high words reach the
/// sampler, so structured ids (sequential, or sharing one half) do not
/// collide by construction; a 2^-32 chance collision between unrelated ids
/// remains. Two requests in one call must not share an `id`
/// ([`Error::Config`] naming both indices): sharing an `id` and a
/// composition would sample identically on the device, so the formula-only
/// control compares separate calls instead.
pub struct CompletionRequest<'a> {
    /// Per-query draw key (see the struct docs): unique within a call, mixed
    /// through one SplitMix64 round (a 2^-32 collision chance between
    /// unrelated ids remains).
    pub id: u64,
    /// Exact target composition, hydrogens included.
    pub composition: Composition,
    /// Required substructures (confirmed present in the target).
    pub patterns: &'a [MolGraph],
    /// Substructures host acceptance checks (`None` means acceptance uses
    /// `patterns`). The model still conditions on `patterns`: with
    /// `CompleteFunctionalGroups` over functional groups, `patterns` is the
    /// seeded subset that fits the encoder while `acceptance_patterns` is
    /// the untruncated full group list acceptance uses.
    pub acceptance_patterns: Option<&'a [MolGraph]>,
    /// Fingerprint evidence (`None` means no fingerprint: a model with the
    /// encoder encodes an empty token set; a model without it given `Some`
    /// is [`Error::Config`]).
    pub fingerprint: Option<&'a SparseFingerprint>,
}

/// Sampling hyperparameters of
/// [`CompletionModel::generate`](CompletionModel::generate).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CompletionGenerationConfig {
    /// Trajectories per query (`K`, `1..=1024`).
    pub trajectories: u32,
    /// Softmax temperature (`> 0`, finite).
    pub temperature: f32,
    /// RNG seed: mixed through one SplitMix64 round into the low/high halves
    /// the sampler hashes (so structured seeds such as `1` and `1 << 32` no
    /// longer share a stream by construction; a 2^-32 chance collision
    /// between unrelated seeds remains).
    pub seed: u64,
    /// Shortlist size (`1..=25`).
    pub returned: u32,
    /// Node budget per
    /// [`contains_pattern`](super::completion::contains_pattern) call.
    pub containment_node_limit: u32,
    /// Work budget per [`same_identity`](super::completion_data::same_identity)
    /// call.
    pub identity_work_limit: u32,
    /// Whether the model conditions on the patterns. `false` is the
    /// formula-only control: the device encodes an empty pattern set (so two
    /// requests with the same composition and `id` sample identically) while
    /// acceptance still requires each request's own patterns.
    pub condition_on_patterns: bool,
    /// How the supplied substructures constrain host acceptance (default
    /// [`Contained`](SubstructureSemantics::Contained), the historical rule;
    /// absent in old serialized configs, which load as `Contained`).
    #[serde(default)]
    pub substructure_semantics: SubstructureSemantics,
}

impl Default for CompletionGenerationConfig {
    /// `K = 64`, temperature 1, seed 0, shortlist 25, both work limits
    /// 100,000, patterns conditioned on. The `K` default is a choice of
    /// this module (the task fixes every other default): large enough for
    /// the shortlist to fill on the overfit molecules, small enough for a
    /// fast host pass.
    fn default() -> Self {
        Self {
            trajectories: 64,
            temperature: 1.0,
            seed: 0,
            returned: 25,
            containment_node_limit: 100_000,
            identity_work_limit: 100_000,
            condition_on_patterns: true,
            substructure_semantics: SubstructureSemantics::Contained,
        }
    }
}

impl CompletionGenerationConfig {
    /// Check the documented ranges: `trajectories` in `1..=1024`,
    /// `temperature` finite and positive, `returned` in `1..=25`. The two
    /// work limits accept any `u32` (0 only makes its check resolve
    /// unresolved immediately). Anything else is [`Error::Config`].
    pub fn validate(&self) -> Result<()> {
        if !(1..=1024).contains(&self.trajectories) {
            return Err(Error::config(format!(
                "CompletionGenerationConfig::validate: trajectories {} is not in 1..=1024",
                self.trajectories
            )));
        }
        if !(self.temperature.is_finite() && self.temperature > 0.0) {
            return Err(Error::config(format!(
                "CompletionGenerationConfig::validate: temperature {} is not finite and positive",
                self.temperature
            )));
        }
        if !(1..=25).contains(&self.returned) {
            return Err(Error::config(format!(
                "CompletionGenerationConfig::validate: returned {} is not in 1..=25",
                self.returned
            )));
        }
        Ok(())
    }
}

/// One ranked shortlist entry: a distinct accepted molecule.
pub struct CompletionCandidate {
    /// The molecule, replayed from [`trace`](CompletionCandidate::trace) on
    /// the host under the exact-completion rule.
    pub graph: MolGraph,
    /// The first sampled trace that produced this identity.
    pub trace: Vec<Token>,
    /// Trajectories that produced this identity (repeats counted).
    pub samples: u32,
    /// The highest trace log-probability among them.
    pub best_log_prob: f32,
}

/// One sampled trajectory of a query, in trajectory order.
///
/// The full device readout, kept so tests can tie every recorded trace
/// log-probability to the teacher and compare the formula-only control arm
/// trace for trace. Acceptance (`candidates`) is the filtered, deduplicated,
/// ranked view of these rows.
pub struct SampledTrajectory {
    /// Trajectory index within the query (`0..K`).
    pub trajectory: u32,
    /// Decoded trace tokens (`length` words of the record; empty when the
    /// record's length or a token field was out of range).
    pub trace: Vec<Token>,
    /// Recorded trace log-probability (the `f32` bits of the record's spare
    /// word).
    pub log_prob: f32,
    /// Raw device status word (the [`candidate_status`] bits).
    pub status: u32,
}

/// The accepted shortlist and accounting of one query.
pub struct QueryOutcome {
    /// Ranked shortlist: at most `returned` entries, never padded. Ranked by
    /// `samples` descending, then `best_log_prob` descending
    /// ([`f32::total_cmp`]), then `trace` lexicographically (the derived
    /// [`Token`] order) — fully deterministic. Holds certified-distinct
    /// identities only: duplicates never occupy slots.
    pub candidates: Vec<CompletionCandidate>,
    /// Accepted trajectories the identity search could not certify, one entry
    /// per distinct unresolved trace (`samples` counting repeats of that
    /// exact trace). Never part of [`candidates`][QueryOutcome::candidates]
    /// and never a hit: [`score_query`](super::completion_eval::score_query)
    /// scans the shortlist only.
    pub unresolved: Vec<CompletionCandidate>,
    /// Every sampled trajectory of the query, in trajectory order (see
    /// [`SampledTrajectory`]).
    pub sampled: Vec<SampledTrajectory>,
    /// Accepted identities before the `returned` cut.
    pub distinct: u32,
    /// Trajectories sampled (`K`).
    pub trajectories: u32,
    /// Device status FINISHED.
    pub finished: u32,
    /// Device status `no_valid_action`.
    pub dead_end: u32,
    /// Device status TRUNCATED.
    pub truncated: u32,
    /// FINISHED on the device but the host exact replay did not end stopped
    /// and complete (or the replayed graph was disconnected or off
    /// composition). Counted, never hidden; expected 0.
    pub rejected_replay: u32,
    /// Complete but a required pattern is not contained. Under
    /// [`DisjointOccurrences`](SubstructureSemantics::DisjointOccurrences)
    /// and
    /// [`CompleteFunctionalGroups`](SubstructureSemantics::CompleteFunctionalGroups)
    /// this covers the chosen rule instead of independent containment.
    pub rejected_containment: u32,
    /// A containment check hit its node limit (not accepted).
    pub containment_unresolved: u32,
    /// Under `CompleteFunctionalGroups`: finished candidates that contain
    /// every supplied group disjointly but carry additional functional
    /// groups. Counted inside `rejected_containment` (a split of it); 0
    /// under the other rules.
    pub rejected_extra_groups: u32,
    /// Under `CompleteFunctionalGroups`: finished candidates whose group
    /// multiset does not cover the supplied one. Counted inside
    /// `rejected_containment` (a split of it); 0 under the other rules.
    pub rejected_missing_groups: u32,
    /// Finished trajectories whose replayed graph passes the `Contained`
    /// rule. Recorded under every semantics, so the cost of each rule is
    /// visible from one run.
    pub pass_contained: u32,
    /// Finished trajectories whose replayed graph passes the
    /// `DisjointOccurrences` rule (recorded under every semantics).
    pub pass_disjoint: u32,
    /// Finished trajectories whose replayed graph passes the
    /// `CompleteFunctionalGroups` rule (recorded under every semantics).
    pub pass_complete: u32,
    /// An identity comparison hit its work limit: the trajectory is kept in
    /// [`unresolved`][QueryOutcome::unresolved] (one entry per distinct
    /// trace) and counted here, once per trajectory.
    pub identity_unresolved: u32,
    /// Trajectories whose status word carries none of FINISHED, TRUNCATED or
    /// `no_valid_action` (e.g. a never-started row). With the rows this
    /// sampler uploads every trajectory starts, so this is expected 0, but
    /// the accounting `trajectories == finished + dead_end + truncated +
    /// other_status` holds exactly either way (asserted in debug builds), as
    /// does the identity accounting `finished == rejected_replay +
    /// rejected_containment + containment_unresolved + sum(samples over all
    /// identities before the cut) + sum(samples over unresolved)`.
    pub other_status: u32,
    /// Pre-check failure reason: no trajectory was sampled (no device work)
    /// because no molecule with the composition can satisfy the rule under
    /// the active [`SubstructureSemantics`]. `None` means the query ran.
    pub infeasible: Option<String>,
}

/// Cheap order-independent identity bucket behind the grouping in
/// [`CompletionModel::generate`]: sorted atom types plus sorted
/// bond-order/type triples. Two graphs with different keys are certainly
/// non-identical, so [`same_identity`](super::completion_data::same_identity)
/// never runs for them; equal keys prove nothing and fall through to the
/// exact check. A bond triple is the two endpoint atom types in nondecreasing
/// order plus the bond order.
fn identity_bucket(graph: &MolGraph) -> (Vec<u8>, Vec<(u8, u8, u8)>) {
    let mut types = graph.atoms().to_vec();
    types.sort_unstable();
    let mut triples: Vec<(u8, u8, u8)> = graph
        .bonds()
        .iter()
        .map(|(a, b, order)| {
            let (left, right) = (graph.atoms()[*a], graph.atoms()[*b]);
            if left <= right {
                (left, right, *order)
            } else {
                (right, left, *order)
            }
        })
        .collect();
    triples.sort_unstable();
    (types, triples)
}

/// One accepted identity while grouping a query's trajectories.
struct AcceptedIdentity {
    /// Replayed molecule of the first trace.
    graph: MolGraph,
    /// First sampled trace of this identity.
    trace: Vec<Token>,
    /// Every trace merged into this identity, the first trace included: an
    /// exact trace repeat is this identity without any search.
    traces: Vec<Vec<Token>>,
    /// Bucket key of [`identity_bucket`].
    bucket: (Vec<u8>, Vec<(u8, u8, u8)>),
    /// Trajectories merged into this identity.
    samples: u32,
    /// Highest trace log-probability among them.
    best: f32,
}

impl<R: Runtime, E: FloatElem> CompletionModel<R, E> {
    /// Sample a ranked shortlist of at most `returned` distinct complete
    /// molecules per query under the exact-completion rule.
    ///
    /// Device part (under `no_grad`; exactly one device read, the final
    /// action record; no read inside the step loop): encode the queries'
    /// [`PatternBatch`] (empty pattern sets when `condition_on_patterns` is
    /// false, so the device inputs do not depend on the patterns), expand
    /// the composition embedding `g` to one row per trajectory (`[B*K, d]`)
    /// while the memory stays `[B, 1 + P, d]` with `rows_per_spectrum = K`,
    /// upload the host-built initial rows of
    /// [`twin::init_completion_trajectories`] (START applied, started word
    /// 2), then `T - 1` steps of [`ms2::step_token`] plus
    /// [`composed_decode_step`] with `closures = max_ring_closures`,
    /// `steps = T`, the config's temperature
    /// and the mixed seed halves. The host re-check below is the authority, so no
    /// validation kernel runs before the read.
    ///
    /// Host part, per query, over its `K` action records in trajectory
    /// order: a FINISHED record decodes its tokens and replays them with
    /// [`replay_exact`](super::grammar::replay_exact) (required stopped and
    /// complete, connected, on composition, else `rejected_replay`); the
    /// finished graph must satisfy the active
    /// [`SubstructureSemantics`](SubstructureSemantics) via
    /// [`accepts`] over the request's `acceptance_patterns` (or `patterns`
    /// when `None`): `RejectedContainment` (and, under
    /// `CompleteFunctionalGroups`, `RejectedExtraGroups` and
    /// `RejectedMissingGroups`, both counted inside `rejected_containment`)
    /// reject the trajectory, while `ContainmentUnresolved` counts
    /// `containment_unresolved` and never accepts. Under `Contained` the
    /// first failure in pattern order wins, as before. Every replayed graph
    /// additionally records whether it passes each of the three rules
    /// (`pass_contained`, `pass_disjoint`, `pass_complete`), regardless of
    /// the active one;
    /// survivors group by [`same_identity`](super::completion_data::same_identity)
    /// behind the [`identity_bucket`] pre-filter, in this order per accepted
    /// trajectory: an exact trace repeat of an accepted identity (or of any
    /// trace already merged into it) merges with no search; otherwise every
    /// accepted identity in the bucket is compared (`Some(true)` merges and
    /// records the trace, even after an unresolved comparison); no `Some(true)`
    /// with at least one `None` keeps the trajectory in
    /// [`unresolved`][QueryOutcome::unresolved] (one entry per distinct trace,
    /// repeats counted, never in the shortlist) and flags
    /// `identity_unresolved`; all `Some(false)` starts a new identity. The
    /// ranking is `samples` descending, then `best_log_prob` descending, then
    /// `trace` lexicographically, cut to `returned` without padding. A
    /// recorded FINISHED status is never trusted on its own.
    ///
    /// Request ids must be distinct within the call ([`Error::Config` naming
    /// both indices); each id is mixed through one SplitMix64 round and the
    /// seed through one round into its low/high halves, so structured
    /// ids/seeds do not collide by construction (a 2^-32 chance collision
    /// between unrelated values remains).
    ///
    /// Before sampling, every request passes the necessary composition fit
    /// [`check_feasibility`] under the active semantics: an infeasible
    /// request returns no candidates with
    /// [`infeasible`](QueryOutcome::infeasible) set and every counter at 0,
    /// without sampling (no device work for it). Only feasible requests
    /// reach the device; when none is feasible there is no device work at
    /// all.
    ///
    /// An empty request list returns no outcomes without touching the
    /// device.
    pub fn generate(
        &self,
        requests: &[CompletionRequest],
        config: &CompletionGenerationConfig,
        constants: &Ms2Constants<R>,
        device: &Device<R>,
    ) -> Result<Vec<QueryOutcome>> {
        config.validate()?;
        if requests.is_empty() {
            return Ok(Vec::new());
        }
        let mut seen_ids: HashMap<u64, usize> = HashMap::with_capacity(requests.len());
        for (i, request) in requests.iter().enumerate() {
            if let Some(&first) = seen_ids.get(&request.id) {
                return Err(Error::config(format!(
                    "CompletionModel::generate: requests {first} and {i} share id {} (ids must be distinct within a call)",
                    request.id
                )));
            }
            seen_ids.insert(request.id, i);
        }
        let _guard = no_grad();
        let limits = Limits::new(
            self.config.max_atoms as usize,
            self.config.max_ring_closures as usize,
        )
        .map_err(|e| {
            Error::config(format!(
                "CompletionModel::generate: model limits rejected: {e}"
            ))
        })?;
        let k = config.trajectories as usize;
        // Necessary composition fit before sampling: an infeasible request
        // returns no candidates with `infeasible` set and no device work.
        // Acceptance (and hence the fit) runs over `acceptance_patterns`
        // when present, else `patterns`.
        let mut infeasible: Vec<Option<String>> = Vec::with_capacity(requests.len());
        for request in requests.iter() {
            infeasible.push(check_feasibility(
                &request.composition,
                request.acceptance_patterns.unwrap_or(request.patterns),
                config.substructure_semantics,
            ));
        }
        let feasible: Vec<usize> = (0..requests.len())
            .filter(|&i| infeasible[i].is_none())
            .collect();
        let mut outcomes: Vec<Option<QueryOutcome>> = Vec::with_capacity(requests.len());
        for _ in 0..requests.len() {
            outcomes.push(None);
        }
        for (i, reason) in infeasible.iter().enumerate() {
            if let Some(reason) = reason {
                outcomes[i] = Some(QueryOutcome {
                    candidates: Vec::new(),
                    unresolved: Vec::new(),
                    sampled: Vec::new(),
                    distinct: 0,
                    trajectories: 0,
                    finished: 0,
                    dead_end: 0,
                    truncated: 0,
                    rejected_replay: 0,
                    rejected_containment: 0,
                    containment_unresolved: 0,
                    rejected_extra_groups: 0,
                    rejected_missing_groups: 0,
                    pass_contained: 0,
                    pass_disjoint: 0,
                    pass_complete: 0,
                    identity_unresolved: 0,
                    other_status: 0,
                    infeasible: Some(reason.clone()),
                });
            }
        }
        if feasible.is_empty() {
            return Ok(outcomes.into_iter().map(|o| o.expect("every outcome filled")).collect());
        }
        let b = feasible.len();
        let rows = b * k;
        let atoms = self.config.max_atoms as usize;
        let closures = self.config.max_ring_closures;
        let steps = limits.max_steps();
        let compositions: Vec<Composition> =
            feasible.iter().map(|&i| requests[i].composition).collect();
        let empty_patterns: Vec<MolGraph> = Vec::new();
        let pattern_refs: Vec<&[MolGraph]> = if config.condition_on_patterns {
            feasible.iter().map(|&i| requests[i].patterns).collect()
        } else {
            feasible.iter().map(|_| empty_patterns.as_slice()).collect()
        };
        let batch = PatternBatch::build(&pattern_refs, &compositions)?;
        let fp_batch = self.fingerprint_batch_for(requests, &feasible)?;
        let encoded = self.encode_with_fingerprints(&batch, &fp_batch, device)?;
        let d = self.config.d_model as usize;
        let traj_formula = encoded
            .context
            .clone()
            .reshape(vec![b, 1, d])?
            .expand(vec![b, k, d])?
            .reshape(vec![rows, d])?;
        let ids: Vec<u64> = feasible
            .iter()
            .map(|&i| SplitMix64::new(requests[i].id).next())
            .collect();
        let (traj_meta_host, state_host, actions_host) =
            twin::init_completion_trajectories(&ids, &compositions, k, steps, atoms);
        let traj_meta =
            IdTensor::from_slice(&traj_meta_host, vec![rows, ms2::TRAJ_META_WIDTH], device)?;
        let mut replay = IdTensor::from_slice(
            &state_host,
            vec![rows, ms2::replay_state_width(atoms)],
            device,
        )?;
        let mut actions = IdTensor::from_slice(
            &actions_host,
            vec![rows, ms2::sample_record_width(steps, atoms)],
            device,
        )?;
        let mut step_token = IdTensor::empty(vec![rows, 4], device);
        let mut logits = Tensor::<R, E>::empty(vec![rows, ms2::sample_logits_width(atoms)], device);
        let mut decoder_state = self.decoder.start_state(&encoded, rows, device)?;
        let bond_table = self.decoder.bond_by_type_value();
        // One SplitMix64 round of the seed into the halves the sampler
        // hashes: structured seeds (1 and 1 << 32, for example) no longer
        // share a stream by construction.
        let seed_mix = SplitMix64::new(config.seed).next();
        let seed_lo = seed_mix as u32;
        let seed_hi = (seed_mix >> 32) as u32;
        // Step 0 is not sampled: initialisation wrote START at position 0,
        // the same range `generate_with_hook` uses.
        for step in 1..steps {
            let _tally = crate::backend::tally_scope("ms2.step");
            ms2::step_token(&actions, &mut step_token, steps, atoms)?;
            composed_decode_step(
                &self.decoder,
                &encoded,
                &traj_formula,
                &mut actions,
                &mut step_token,
                &mut replay,
                &mut logits,
                &traj_meta,
                &mut decoder_state,
                &bond_table,
                &constants.atom_table,
                step,
                seed_lo,
                seed_hi,
                config.temperature,
                steps,
                atoms,
                closures,
                k,
                rows,
            )?;
        }
        // The single device read of the call: the final action record.
        let record = actions.try_to_vec()?;
        let width = ms2::sample_record_width(steps, atoms);
        let returned = config.returned as usize;
        let semantics = config.substructure_semantics;
        let containment_limit = config.containment_node_limit as usize;
        let identity_limit = config.identity_work_limit as usize;
        for (q, &qi) in feasible.iter().enumerate() {
            let request = &requests[qi];
            let mut outcome = QueryOutcome {
                candidates: Vec::new(),
                unresolved: Vec::new(),
                sampled: Vec::with_capacity(k),
                distinct: 0,
                trajectories: k as u32,
                finished: 0,
                dead_end: 0,
                truncated: 0,
                rejected_replay: 0,
                rejected_containment: 0,
                containment_unresolved: 0,
                rejected_extra_groups: 0,
                rejected_missing_groups: 0,
                pass_contained: 0,
                pass_disjoint: 0,
                pass_complete: 0,
                identity_unresolved: 0,
                other_status: 0,
                infeasible: None,
            };
            let mut identities: Vec<AcceptedIdentity> = Vec::new();
            // Unresolved accepted trajectories, one entry per distinct trace.
            let mut unresolved: Vec<CompletionCandidate> = Vec::new();
            for kk in 0..k {
                let base = (q * k + kk) * width;
                let tail = base + steps * 4 + atoms;
                let length = record[tail] as usize;
                let status = record[tail + 1];
                let log_prob = f32::from_bits(record[tail + 2]);
                if status & candidate_status::FINISHED != 0 {
                    outcome.finished += 1;
                } else if status & candidate_status::NO_VALID_ACTION != 0 {
                    outcome.dead_end += 1;
                } else if status & candidate_status::TRUNCATED != 0 {
                    outcome.truncated += 1;
                } else {
                    outcome.other_status += 1;
                }
                let mut trace = Vec::new();
                let mut decodable = length <= steps;
                if decodable {
                    for i in 0..length {
                        let words = &record[base + i * 4..base + i * 4 + 4];
                        let fields: Vec<u8> = words
                            .iter()
                            .map(|word| u8::try_from(*word).ok())
                            .collect::<Option<Vec<u8>>>()
                            .unwrap_or_default();
                        if fields.len() != 4 {
                            decodable = false;
                            break;
                        }
                        trace.push(Token {
                            kind: fields[0],
                            atom_type: fields[1],
                            bond: fields[2],
                            pointer: fields[3],
                        });
                    }
                }
                if !decodable {
                    trace.clear();
                }
                outcome.sampled.push(SampledTrajectory {
                    trajectory: kk as u32,
                    trace: trace.clone(),
                    log_prob,
                    status,
                });
                if status & candidate_status::FINISHED == 0 {
                    continue;
                }
                if !decodable {
                    // A finished row whose record cannot be decoded is a
                    // rejected candidate, never a silently dropped one.
                    outcome.rejected_replay += 1;
                    continue;
                }
                let end = match replay_exact(&trace, limits, request.composition) {
                    Ok(end) => end,
                    Err(_) => {
                        outcome.rejected_replay += 1;
                        continue;
                    }
                };
                if !end.stopped() || !end.is_complete() {
                    outcome.rejected_replay += 1;
                    continue;
                }
                let graph = match end.graph() {
                    Ok(graph) => graph,
                    Err(_) => {
                        outcome.rejected_replay += 1;
                        continue;
                    }
                };
                if !graph.is_connected() || graph.composition() != request.composition {
                    outcome.rejected_replay += 1;
                    continue;
                }
                let accept_patterns = request.acceptance_patterns.unwrap_or(request.patterns);
                // Per-rule pass counts over the replayed graph, for every
                // rule regardless of the active one: all three verdicts run
                // immediately after successful replay (a candidate rejected
                // by the active rule still counts its passes under the other
                // rules), then the active verdict is applied.
                let rules = [
                    SubstructureSemantics::Contained,
                    SubstructureSemantics::DisjointOccurrences,
                    SubstructureSemantics::CompleteFunctionalGroups,
                ];
                let mut verdicts = [Acceptance::RejectedContainment; 3];
                for (i, rule) in rules.iter().enumerate() {
                    verdicts[i] = accepts(
                        &graph,
                        accept_patterns,
                        *rule,
                        containment_limit,
                        identity_limit,
                    );
                }
                if verdicts[0] == Acceptance::Accepted {
                    outcome.pass_contained += 1;
                }
                if verdicts[1] == Acceptance::Accepted {
                    outcome.pass_disjoint += 1;
                }
                if verdicts[2] == Acceptance::Accepted {
                    outcome.pass_complete += 1;
                }
                let active = match semantics {
                    SubstructureSemantics::Contained => 0,
                    SubstructureSemantics::DisjointOccurrences => 1,
                    SubstructureSemantics::CompleteFunctionalGroups => 2,
                };
                match verdicts[active] {
                    Acceptance::Accepted => {}
                    Acceptance::RejectedContainment => {
                        outcome.rejected_containment += 1;
                        continue;
                    }
                    Acceptance::RejectedExtraGroups => {
                        outcome.rejected_containment += 1;
                        outcome.rejected_extra_groups += 1;
                        continue;
                    }
                    Acceptance::RejectedMissingGroups => {
                        outcome.rejected_containment += 1;
                        outcome.rejected_missing_groups += 1;
                        continue;
                    }
                    Acceptance::ContainmentUnresolved => {
                        outcome.containment_unresolved += 1;
                        continue;
                    }
                }
                let bucket = identity_bucket(&graph);
                // (a) An exact trace repeat of an accepted identity (or of any
                // trace already merged into it) is that identity: no search.
                // With a spent work budget every copy of one graph lands here
                // instead of taking its own slot.
                let mut merged = false;
                for accepted in identities.iter_mut() {
                    if accepted.traces.contains(&trace) {
                        accepted.samples += 1;
                        if log_prob > accepted.best {
                            accepted.best = log_prob;
                        }
                        merged = true;
                        break;
                    }
                }
                if merged {
                    continue;
                }
                // (b) Otherwise compare with every accepted identity in the
                // bucket: `Some(true)` anywhere merges (and records the
                // trace), even after an unresolved comparison.
                let mut hit: Option<usize> = None;
                let mut saw_unresolved = false;
                for (index, accepted) in identities.iter().enumerate() {
                    if accepted.bucket != bucket {
                        continue;
                    }
                    match same_identity(
                        &graph,
                        &accepted.graph,
                        config.identity_work_limit as usize,
                    ) {
                        Some(true) => {
                            hit = Some(index);
                            break;
                        }
                        Some(false) => {}
                        None => {
                            saw_unresolved = true;
                        }
                    }
                }
                match hit {
                    Some(index) => {
                        let accepted = &mut identities[index];
                        accepted.samples += 1;
                        if log_prob > accepted.best {
                            accepted.best = log_prob;
                        }
                        accepted.traces.push(trace);
                    }
                    None if saw_unresolved => {
                        // (c) No certified merge but an unresolved comparison:
                        // the trajectory waits in `unresolved` (one entry per
                        // distinct trace), never in the shortlist.
                        outcome.identity_unresolved += 1;
                        match unresolved.iter_mut().find(|entry| entry.trace == trace) {
                            Some(entry) => {
                                entry.samples += 1;
                                if log_prob > entry.best_log_prob {
                                    entry.best_log_prob = log_prob;
                                }
                            }
                            None => unresolved.push(CompletionCandidate {
                                graph,
                                trace,
                                samples: 1,
                                best_log_prob: log_prob,
                            }),
                        }
                    }
                    None => {
                        // (d) Certified distinct from every accepted identity.
                        identities.push(AcceptedIdentity {
                            graph,
                            trace: trace.clone(),
                            traces: vec![trace],
                            bucket,
                            samples: 1,
                            best: log_prob,
                        });
                    }
                }
            }
            debug_assert_eq!(
                outcome.trajectories,
                outcome.finished + outcome.dead_end + outcome.truncated + outcome.other_status,
                "query {q}: every trajectory carries exactly one status class"
            );
            let identity_samples: u32 = identities.iter().map(|id| id.samples).sum();
            let unresolved_samples: u32 = unresolved.iter().map(|entry| entry.samples).sum();
            debug_assert_eq!(
                outcome.identity_unresolved, unresolved_samples,
                "query {q}: identity_unresolved counts every unresolved trajectory once"
            );
            debug_assert_eq!(
                outcome.finished,
                outcome.rejected_replay
                    + outcome.rejected_containment
                    + outcome.containment_unresolved
                    + identity_samples
                    + unresolved_samples,
                "query {q}: finished trajectories are rejected, certified or unresolved"
            );
            outcome.distinct = identities.len() as u32;
            outcome.unresolved = unresolved;
            identities.sort_by(|a, c| {
                c.samples
                    .cmp(&a.samples)
                    .then_with(|| c.best.total_cmp(&a.best))
                    .then_with(|| a.trace.cmp(&c.trace))
            });
            outcome.candidates = identities
                .into_iter()
                .take(returned)
                .map(|accepted| CompletionCandidate {
                    graph: accepted.graph,
                    trace: accepted.trace,
                    samples: accepted.samples,
                    best_log_prob: accepted.best,
                })
                .collect();
            outcomes[qi] = Some(outcome);
        }
        Ok(outcomes
            .into_iter()
            .map(|outcome| outcome.expect("every outcome filled"))
            .collect())
    }
}
