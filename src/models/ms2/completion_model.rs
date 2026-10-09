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
use crate::tensor::ops::index::{IdTensor, gather_rows};
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
use super::completion_spectrum::{SPECTRUM_SLOTS, SpectrumBatch, SpectrumEncoder, SpectrumEvidence};
use super::contain::Containment;
use super::functional_groups::functional_groups;
use super::contract::ModelConfig;
use super::contract::candidate_status;
use super::formula_enum::{EnumDomain, RatioBounds};
use super::decoder::{Ms2Decoder, ReplayView, TeacherOutput, graph_loss};
use super::encoder::EncoderOutput;
use super::generate::{
    StepLogProbs, apply_chosen_action, composed_decode_step, composed_step_heads,
    gather_generation_rows, step_field_log_probs,
};
use super::grammar::{COMPLETION_GRAMMAR_VERSION, Limits, Token, TraceState, replay_exact};
use super::graph::MolGraph;
use super::targets_batch::TargetBatch;
use super::twin;
use super::workspace::Ms2Capabilities;

/// Config version of [`CompletionModelConfig`].
pub const COMPLETION_MODEL_VERSION: &str = "completion-model-v1";
/// Pattern-atom slots per query (the request layer's limit, and the width
/// [`PatternBatch::build`] lays out).
///
/// This is the decoder's own atom limit (`max_atoms`), so every substructure
/// that can occur inside a molecule the model can build — a Bemis-Murcko
/// scaffold included — fits in one query's slots. Nothing learned is shaped
/// by it: [`SubstructureEncoder`]'s tables are per atom type and per open
/// valence, its message passing and projections are per `d_model`, and the
/// decoder reads the memory width off the tensor
/// ([`Ms2Decoder::teacher`](super::decoder::Ms2Decoder::teacher) pads it to
/// a whole number of vectors), so a checkpoint saved at a narrower width
/// loads unchanged and simply sees more slots. The cost is the
/// `[B, 3, P, P]` adjacency, quadratic in the width: 12 KiB per query in
/// `f32` here, against 6.75 KiB at the 24 slots this model was first trained
/// with. [`PatternBatch::build_with_slots`] lays out a narrower batch for
/// tests that pin that older behaviour.
pub const PATTERN_SLOTS: usize = 32;
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
    /// Spectrum peak slots per query (`0` means no spectrum encoder:
    /// existing checkpoints and fixtures load and behave bit-identically).
    /// When non-zero the model owns a [`SpectrumEncoder`](super::completion_spectrum::SpectrumEncoder)
    /// and the peak states follow the fingerprint tokens in the decoder
    /// memory, with the pooled peak, adduct and neutral-mass vector added to
    /// the composition row.
    #[serde(default)]
    pub spectrum_slots: u32,
    /// Whether the decoder is told, at every step, how much of the requested
    /// composition and of the supplied substructure it has already built
    /// (`false` means no progress projection: existing checkpoints and
    /// fixtures load and behave bit-identically).
    ///
    /// When true the model owns a [`ProgressConditioner`], whose
    /// `Linear(32, d)` projection of
    /// [`ms2::PROGRESS_FEATURES`](crate::tensor::ops::ms2::PROGRESS_FEATURES)
    /// words is added to the per-row conditioning the decoder takes at each
    /// step — in the sampling and beam paths and in teacher forcing alike, so
    /// training and inference see the same inputs.
    #[serde(default)]
    pub progress_features: bool,
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
            spectrum_slots: 0,
            progress_features: false,
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
            spectrum_slots: 0,
            progress_features: false,
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

    /// Base config with the fingerprint and the spectrum encoder:
    /// [`base_fingerprint`](Self::base_fingerprint) with [`SPECTRUM_SLOTS`]
    /// peak slots.
    pub fn base_spectrum() -> Self {
        let mut config = Self::base_fingerprint();
        config.spectrum_slots = SPECTRUM_SLOTS as u32;
        config
    }

    /// Base config with the fingerprint encoder, the spectrum encoder and the
    /// per-step progress features: [`base_spectrum`](Self::base_spectrum)
    /// with `progress_features` on.
    pub fn base_progress() -> Self {
        let mut config = Self::base_spectrum();
        config.progress_features = true;
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
            spectrum_slots: 0,
            progress_features: false,
            dtype: DType::F32,
        }
    }

    /// Check the versions and ranges: the three version strings must match
    /// their constants, the decoder SSM must validate with
    /// `decoder.d_model == d_model`, `d_model` must be a positive multiple
    /// of `attention_heads`, and the block/atom/closure/round counts must fit
    /// their documented ranges. `fingerprint_slots` is `0` (no fingerprint
    /// encoder) or `1..=4096`; `spectrum_slots` is `0` (no spectrum encoder)
    /// or `1..=512`. Anything else is [`Error::Config`].
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
        if self.spectrum_slots > 512 {
            return Err(Error::config(format!(
                "CompletionModelConfig::validate: spectrum_slots {} exceeds 512",
                self.spectrum_slots
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
/// its own `slots` slots each ([`PATTERN_SLOTS`] from
/// [`build`](PatternBatch::build)), plus the queries' compositions.
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
    /// Pattern-atom slots per query, the `P` of every shape below
    /// ([`PATTERN_SLOTS`] from [`build`](PatternBatch::build)).
    pub slots: usize,
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
        Self::build_with_slots(patterns, compositions, PATTERN_SLOTS)
    }

    /// [`build`](Self::build) at an explicit width: the same layout in
    /// `slots` pattern-atom slots per query instead of [`PATTERN_SLOTS`].
    ///
    /// The encoder reads the width off the batch, so a narrower batch encodes
    /// through the same path; this exists to pin that a query whose atoms fit
    /// the narrower width encodes bit-identically at either one (the padding
    /// slots an extra width adds are selected to exact zeros and masked out
    /// of the decoder memory). `slots` outside `1..=PATTERN_SLOTS` is
    /// [`Error::Config`]; everything else is [`build`](Self::build)'s
    /// contract with `slots` in place of [`PATTERN_SLOTS`].
    pub fn build_with_slots(
        patterns: &[&[MolGraph]],
        compositions: &[Composition],
        slots: usize,
    ) -> Result<Self> {
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
        if slots == 0 || slots > PATTERN_SLOTS {
            return Err(Error::config(format!(
                "PatternBatch::build: {slots} slots per query is not within 1..={PATTERN_SLOTS}"
            )));
        }
        let queries = patterns.len();
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
            slots,
            types,
            open,
            valid,
            adjacency,
            features,
        })
    }

    /// Per-query counts of each atom type among the *valid* pattern slots:
    /// `[B*18]`, index `b * 18 + t` holding how many of query `b`'s supplied
    /// pattern atoms carry type `t`.
    ///
    /// Index 0 of every query is always 0: padding slots hold type id 0 and
    /// are excluded by their `valid` entry, and no real atom type is 0. This
    /// is the pattern half of the progress features
    /// ([`ms2::progress_features`](crate::tensor::ops::ms2::progress_features)),
    /// taken on the host because it depends on nothing the device changes:
    /// the kernel then reads 18 words instead of scanning
    /// `slots` slots per lane. A slot whose type is at or past 18
    /// is skipped rather than counted — [`validate`](Self::validate) refuses
    /// it before any upload, and this never panics on a hand-built batch.
    pub fn type_counts(&self) -> Vec<u32> {
        let (b, p) = (self.queries, self.slots);
        let mut counts = vec![0u32; b * 18];
        for q in 0..b {
            for s in 0..p {
                let at = q * p + s;
                if self.valid.get(at).copied().unwrap_or(0.0) == 0.0 {
                    continue;
                }
                let Some(&t) = self.types.get(at) else {
                    continue;
                };
                if t < 18 {
                    counts[q * 18 + t as usize] += 1;
                }
            }
        }
        counts
    }

    /// Check the host arrays against the encoder's contract before upload.
    ///
    /// `slots` must be within `1..=`[`PATTERN_SLOTS`], and the array lengths
    /// must fit `queries` queries of `slots` slots
    /// (`types`/`open`/`valid` are `[B*P]`, `adjacency` is `[B*3*P*P]` and
    /// `features` is `[B*10]`); every `valid` entry must be exactly 0.0 or
    /// 1.0; `types` must be below 18 and `open` below 9 (in padding slots too,
    /// apart from which padded `types`/`open` values are inert by
    /// selection); every `adjacency` and `features` entry must be finite;
    /// `adjacency` must be exactly 0 wherever either endpoint slot is
    /// invalid, symmetric, and zero on the diagonal. Anything else is
    /// [`Error::Config`] naming the first offending flat index.
    pub fn validate(&self) -> Result<()> {
        let (b, p) = (self.queries, self.slots);
        if p == 0 || p > PATTERN_SLOTS {
            return Err(Error::config(format!(
                "PatternBatch::validate: {p} slots per query is not within 1..={PATTERN_SLOTS}"
            )));
        }
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
        let (b, p) = (self.queries, self.slots);
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
        let (b, p, d) = (batch.queries, batch.slots, self.d_model);
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
        let (b, p, d) = (batch.queries, batch.slots, self.d_model);
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

/// The per-step structural progress conditioning: one `Linear(32, d)` over
/// the [`ms2::PROGRESS_FEATURES`] words of
/// [`ms2::progress_features`], added to the per-row conditioning vector the
/// decoder already takes at every step.
///
/// Present exactly when
/// [`CompletionModelConfig::progress_features`] is set. The projection is
/// the whole parameter cost (`32 * d` weights, no bias), and it is
/// **zero-initialised**: a model built with the feature on starts numerically
/// identical to the same model with it off, which is what lets an existing
/// checkpoint be resumed with the feature switched on (a non-strict load
/// leaves these weights at zero, so step 0 of the resumed run is the
/// checkpoint). The weight gradient is still non-zero from the first step —
/// it is `x` against the incoming adjoint — so the projection learns from
/// there.
///
/// No bias, because a bias would be a constant added to every position of
/// every row: the formula row the decoder already adds carries that, and a
/// second copy of it would only be a reparametrisation.
pub struct ProgressConditioner<R: Runtime, E: FloatElem> {
    /// `Linear(PROGRESS_FEATURES, d, no bias)`, zero-initialised.
    proj: Linear<R, E>,
    /// Residual width of the model the projection feeds.
    d_model: usize,
    /// Atom slots per candidate (`max_atoms`).
    atoms: usize,
    /// Ring-closure budget per candidate (`max_ring_closures`).
    closures: u32,
}

impl<R: Runtime, E: FloatElem> Module<R, E> for ProgressConditioner<R, E> {
    fn visit(&self, visitor: &mut ModuleVisitor<'_, R, E>) {
        visitor.child("proj", &self.proj);
    }
}

impl<R: Runtime, E: FloatElem> ProgressConditioner<R, E> {
    /// Build the zero-initialised projection for a model of width `d_model`
    /// with `atoms` atom slots and `closures` ring closures.
    fn init(
        d_model: usize,
        atoms: usize,
        closures: u32,
        device: &Device<R>,
        rng: &mut Rng,
    ) -> Self {
        let proj = LinearConfig::new(ms2::PROGRESS_FEATURES, d_model)
            .with_bias(false)
            .with_initializer(Initializer::Zeros)
            .init::<R, E>(device, rng);
        Self {
            proj,
            d_model,
            atoms,
            closures,
        }
    }

    /// The per-row conditioning of one decode step: `base` (`[rows, d]`, the
    /// expanded composition row) plus the projected progress features of the
    /// current grammar rows.
    ///
    /// `replay` is the `[rows, 3A + 16]` state row **after** the last emitted
    /// token, which is exactly what
    /// [`Ms2Decoder::step_logits`](super::decoder::Ms2Decoder::step_logits)
    /// reads at the same step, so the features describe the prefix the step
    /// conditions on. `pattern_counts` is `[rows / rows_per_spectrum, 18]`
    /// from [`PatternBatch::type_counts`]. 1 launch for the features plus the
    /// projection's own; no device read.
    #[allow(clippy::too_many_arguments)]
    pub fn step_conditioning(
        &self,
        base: &Var<R, E>,
        replay: &IdTensor<R>,
        traj_meta: &IdTensor<R>,
        pattern_counts: &IdTensor<R>,
        rows_per_spectrum: usize,
        steps: usize,
        device: &Device<R>,
    ) -> Result<Var<R, E>> {
        let rows = replay.shape().dim(0);
        let want_base: &[usize] = &[rows, self.d_model];
        if base.dims() != want_base {
            return Err(Error::shape(format!(
                "ProgressConditioner::step_conditioning needs base [{rows}, {}], got {}",
                self.d_model,
                base.shape()
            )));
        }
        let mut features =
            Tensor::<R, E>::empty(vec![rows, ms2::PROGRESS_FEATURES], device);
        ms2::progress_features(
            replay,
            traj_meta,
            pattern_counts,
            &mut features,
            self.atoms,
            rows_per_spectrum,
            self.closures,
            steps,
        )?;
        base.add(&self.proj.apply(&Var::constant(features))?)
    }

    /// The `[rows, T, d]` per-step conditioning of a teacher pass: the
    /// projected progress features of every position, in one feature launch.
    ///
    /// `tokens` and `target_meta` are the uploaded target words
    /// ([`TargetBatch::upload`](super::targets_batch::TargetBatch)), and
    /// `pattern_counts` is `[rows, 18]` — one row per target row, which the
    /// completion teacher path has because it carries one exact-completion
    /// slot per query. Position `i` describes the state after tokens `0..=i`,
    /// the state the stepping path's row carries at decode step `i + 1`, so
    /// training and sampling see the same features at the same prefix. 1
    /// launch for the features plus the projection's own; no device read.
    pub fn teacher_conditioning(
        &self,
        tokens: &IdTensor<R>,
        target_meta: &IdTensor<R>,
        pattern_counts: &IdTensor<R>,
        constants: &Ms2Constants<R>,
        device: &Device<R>,
    ) -> Result<Var<R, E>> {
        let rows = tokens.shape().dim(0);
        let steps = tokens.shape().dim(1);
        let mut scratch =
            IdTensor::empty(vec![rows, ms2::replay_state_width(self.atoms)], device);
        let mut features =
            Tensor::<R, E>::empty(vec![rows, steps, ms2::PROGRESS_FEATURES], device);
        ms2::progress_features_teacher(
            tokens,
            target_meta,
            pattern_counts,
            constants,
            &mut scratch,
            &mut features,
            self.atoms,
            self.closures,
        )?;
        let flat = Var::constant(features.reshape(vec![rows * steps, ms2::PROGRESS_FEATURES])?);
        self.proj
            .apply(&flat)?
            .reshape(vec![rows, steps, self.d_model])
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
    /// Spectrum encoder (`None` when `spectrum_slots == 0`).
    spectrum_encoder: Option<SpectrumEncoder<R, E>>,
    /// Graph-action decoder (architecture §4.3, reused unchanged).
    decoder: Ms2Decoder<R, E>,
    /// Per-step structural progress conditioning (`None` when
    /// `progress_features` is off).
    progress: Option<ProgressConditioner<R, E>>,
}

impl<R: Runtime, E: FloatElem> Module<R, E> for CompletionModel<R, E> {
    fn visit(&self, visitor: &mut ModuleVisitor<'_, R, E>) {
        visitor.child("encoder", &self.encoder);
        if let Some(fp) = &self.fingerprint_encoder {
            visitor.child("fingerprint_encoder", fp);
        }
        if let Some(spectrum) = &self.spectrum_encoder {
            visitor.child("spectrum_encoder", spectrum);
        }
        visitor.child("decoder", &self.decoder);
        // Last, so a config without the progress projection names exactly the
        // parameters it named before this conditioner existed.
        if let Some(progress) = &self.progress {
            visitor.child("progress", progress);
        }
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
        // Built after the decoder, so a config without spectrum slots draws
        // exactly the weights it drew before this encoder existed.
        let spectrum_encoder = if config.spectrum_slots > 0 {
            Some(SpectrumEncoder::init(d, device, rng))
        } else {
            None
        };
        // Built last for the same reason the spectrum encoder is: a config
        // without the progress projection draws exactly the weights it drew
        // before this conditioner existed.
        let progress = if config.progress_features {
            Some(ProgressConditioner::init(
                d,
                config.max_atoms as usize,
                config.max_ring_closures,
                device,
                rng,
            ))
        } else {
            None
        };
        Ok(Self {
            config: config.clone(),
            encoder,
            fingerprint_encoder,
            spectrum_encoder,
            decoder,
            progress,
        })
    }

    /// Whether the model owns the per-step progress conditioning
    /// (`progress_features` on).
    pub fn has_progress(&self) -> bool {
        self.progress.is_some()
    }

    /// The per-step progress conditioning, for a caller that drives the
    /// decode loop itself (`None` when the feature is off).
    pub fn progress(&self) -> Option<&ProgressConditioner<R, E>> {
        self.progress.as_ref()
    }

    /// Whether the model owns a spectrum encoder (`spectrum_slots > 0`).
    pub fn has_spectrum(&self) -> bool {
        self.spectrum_encoder.is_some()
    }

    /// Spectrum peak slots of the model (`0` without the encoder).
    pub fn spectrum_slots(&self) -> usize {
        self.config.spectrum_slots as usize
    }

    /// Whether the model owns a fingerprint encoder (`fingerprint_slots > 0`).
    pub fn has_fingerprint(&self) -> bool {
        self.fingerprint_encoder.is_some()
    }

    /// Fingerprint slots of the model (`0` without the encoder).
    pub fn fingerprint_slots(&self) -> usize {
        self.config.fingerprint_slots as usize
    }

    /// The graph-action decoder, for a caller that drives the decode loop
    /// itself.
    ///
    /// [`generate`](Self::generate) is the sampling loop; a search over the
    /// same model needs the decoder to build its own state
    /// ([`Ms2Decoder::start_state`]) and to drive
    /// [`composed_step_log_probs`](super::generate::composed_step_log_probs)
    /// and
    /// [`composed_decode_step`](super::generate::composed_decode_step)
    /// step by step.
    pub fn decoder(&self) -> &Ms2Decoder<R, E> {
        &self.decoder
    }

    /// Encode `batch` into the decoder memory, pool and context. No device
    /// read. A model with a fingerprint encoder encodes an empty token set
    /// here; use [`encode_with_fingerprints`](Self::encode_with_fingerprints)
    /// to supply one.
    pub fn encode(&self, batch: &PatternBatch, device: &Device<R>) -> Result<EncoderOutput<R, E>> {
        if self.fingerprint_encoder.is_none() && self.spectrum_encoder.is_none() {
            return self.encoder.encode(batch, device);
        }
        let empty = FingerprintBatch::empty(batch.queries, self.fingerprint_slots())?;
        self.encode_with_fingerprints(batch, &empty, device)
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
        self.encode_with_evidence(patterns, fingerprints, None, device)
    }

    /// Encode patterns, a fingerprint set and spectral evidence into the
    /// decoder memory: `[context; pattern atoms; fingerprint tokens; peaks]`
    /// with the matching mask, each optional part present only when the
    /// model owns its encoder. The composition row `g` adds the pooled
    /// fingerprint vector and the pooled peak, adduct and neutral-mass
    /// vector, with `g` first in every addition (the tape-order discipline
    /// of the pattern encoder). No device read.
    ///
    /// `spectra` is `None` for no spectral evidence (a model with the
    /// spectrum encoder then encodes an empty set, which contributes exact
    /// zeros); a model without the encoder rejects any present evidence with
    /// [`Error::Config`], as does a slot or query count that differs from
    /// the model's. The fingerprint rules are those of
    /// [`encode_with_fingerprints`](Self::encode_with_fingerprints).
    pub fn encode_with_evidence(
        &self,
        patterns: &PatternBatch,
        fingerprints: &FingerprintBatch,
        spectra: Option<&SpectrumBatch>,
        device: &Device<R>,
    ) -> Result<EncoderOutput<R, E>> {
        if self.spectrum_encoder.is_none()
            && let Some(spectra) = spectra
            && spectra.present.iter().any(|&v| v != 0.0)
        {
            return Err(Error::config(
                "CompletionModel::encode: model without a spectrum encoder was given spectral evidence (spectrum_slots = 0)".to_string(),
            ));
        }
        if self.fingerprint_encoder.is_none() && fingerprints.valid.iter().any(|&v| v != 0.0) {
            return Err(Error::config(
                "CompletionModel::encode: model without a fingerprint encoder was given a fingerprint (fingerprint_slots = 0)".to_string(),
            ));
        }
        if self.fingerprint_encoder.is_none() && self.spectrum_encoder.is_none() {
            return self.encoder.encode(patterns, device);
        }
        let fp_slots = self.fingerprint_slots();
        if self.fingerprint_encoder.is_some() {
            if fingerprints.slots != fp_slots {
                return Err(Error::config(format!(
                    "CompletionModel::encode: fingerprint batch holds {} slots for a model with {fp_slots}",
                    fingerprints.slots
                )));
            }
            if fingerprints.queries != patterns.queries {
                return Err(Error::config(format!(
                    "CompletionModel::encode: {} fingerprint queries for {} pattern queries",
                    fingerprints.queries, patterns.queries
                )));
            }
        }
        let spec_slots = self.spectrum_slots();
        if self.spectrum_encoder.is_some()
            && let Some(spectra) = spectra
        {
            if spectra.slots != spec_slots {
                return Err(Error::config(format!(
                    "CompletionModel::encode: spectrum batch holds {} slots for a model with {spec_slots}",
                    spectra.slots
                )));
            }
            if spectra.queries != patterns.queries {
                return Err(Error::config(format!(
                    "CompletionModel::encode: {} spectrum queries for {} pattern queries",
                    spectra.queries, patterns.queries
                )));
            }
        }
        let (b, p) = (patterns.queries, patterns.slots);
        let d = self.config.d_model as usize;
        let extra = fp_slots + spec_slots;
        if b == 0 {
            let x = Var::constant(Tensor::<R, E>::zeros(vec![0, p, d], device));
            let valid = Tensor::<R, E>::zeros(vec![0, p], device);
            let memory = Var::constant(Tensor::<R, E>::zeros(vec![0, 1 + p + extra, d], device));
            let memory_mask = Tensor::<R, E>::zeros(vec![0, 1 + p + extra], device);
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
        let mut g = g0;
        let mut memory_parts: Vec<Var<R, E>> = Vec::with_capacity(3);
        let mut mask_parts: Vec<Tensor<R, E>> = Vec::with_capacity(3);
        if let Some(fp_encoder) = &self.fingerprint_encoder {
            let (h_fp, valid_fp) = fp_encoder.encode_states(fingerprints, device)?;
            let fp_pooled = fp_encoder.encode_pooled_from_states(&h_fp, &valid_fp, device)?;
            // `g + fp`, not `fp + g`: same tape discipline as the pattern
            // encoder (the first parent keeps the accumulator tape).
            g = g.add(&fp_pooled)?;
            memory_parts.push(h_fp);
            mask_parts.push(valid_fp);
        }
        if let Some(spectrum_encoder) = &self.spectrum_encoder {
            let empty;
            let batch = match spectra {
                Some(batch) => batch,
                None => {
                    empty = SpectrumBatch::empty(b, spec_slots)?;
                    &empty
                }
            };
            let encoded = spectrum_encoder.encode(batch, device)?;
            g = g.add(&encoded.pooled)?;
            memory_parts.push(encoded.states);
            mask_parts.push(encoded.valid);
        }
        let mem0 = self.encoder.memory_in.apply(&g)?.unsqueeze(1)?;
        let mut memory_all = vec![mem0, h_pat.clone()];
        memory_all.extend(memory_parts);
        let memory = cat(&memory_all, 1)?;
        let ones = Tensor::<R, E>::ones(vec![b, 1], device);
        let mut mask_all = vec![ones, valid_pat.clone()];
        mask_all.extend(mask_parts);
        let memory_mask = crate::tensor::ops::movement::cat(&mask_all, 1)?;
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

    /// Spectrum batch for `generate`: one optional [`SpectrumEvidence`] per
    /// feasible request. `None` when the model has no spectrum encoder (any
    /// present evidence is then [`Error::Config`]).
    fn spectrum_batch_for(
        &self,
        spectra: &[Option<&SpectrumEvidence>],
        feasible: &[usize],
    ) -> Result<Option<SpectrumBatch>> {
        if self.spectrum_encoder.is_none() {
            if feasible.iter().any(|&i| spectra[i].is_some()) {
                return Err(Error::config(
                    "CompletionModel::generate: request carries spectral evidence but the model has no spectrum encoder (spectrum_slots = 0)".to_string(),
                ));
            }
            return Ok(None);
        }
        let picked: Vec<Option<&SpectrumEvidence>> = feasible.iter().map(|&i| spectra[i]).collect();
        Ok(Some(SpectrumBatch::build(&picked, self.spectrum_slots())?))
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
        if self.fingerprint_encoder.is_some() || self.spectrum_encoder.is_some() {
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
        // The per-step progress conditioning, when the model owns it: one
        // `[rows, T, 32]` feature launch plus its projection, added to every
        // position's input. With the feature off nothing is built and the
        // pass is the one it has always been.
        let progress = match &self.progress {
            Some(conditioner) => {
                let counts = IdTensor::from_slice(
                    &patterns.type_counts(),
                    vec![queries, 18],
                    device,
                )?;
                Some(conditioner.teacher_conditioning(
                    &uploaded.tokens,
                    &uploaded.meta,
                    &counts,
                    constants,
                    device,
                )?)
            }
            None => None,
        };
        let out = self.decoder.teacher_with_progress(
            &encoded,
            &encoded.context,
            &uploaded,
            &replay,
            progress.as_ref(),
        )?;
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
        self.teacher_with_evidence(patterns, fingerprints, None, targets, constants, device)
    }

    /// Teacher forcing with a fingerprint set and spectral evidence: like
    /// [`teacher_with_fingerprints`](Self::teacher_with_fingerprints) but
    /// encoding through
    /// [`encode_with_evidence`](Self::encode_with_evidence). No device read.
    pub fn teacher_with_evidence(
        &self,
        patterns: &PatternBatch,
        fingerprints: &FingerprintBatch,
        spectra: Option<&SpectrumBatch>,
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
        let encoded = self.encode_with_evidence(patterns, fingerprints, spectra, device)?;
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
        // The per-step progress conditioning, when the model owns it: one
        // `[rows, T, 32]` feature launch plus its projection, added to every
        // position's input. With the feature off nothing is built and the
        // pass is the one it has always been.
        let progress = match &self.progress {
            Some(conditioner) => {
                let counts = IdTensor::from_slice(
                    &patterns.type_counts(),
                    vec![queries, 18],
                    device,
                )?;
                Some(conditioner.teacher_conditioning(
                    &uploaded.tokens,
                    &uploaded.meta,
                    &counts,
                    constants,
                    device,
                )?)
            }
            None => None,
        };
        let out = self.decoder.teacher_with_progress(
            &encoded,
            &encoded.context,
            &uploaded,
            &replay,
            progress.as_ref(),
        )?;
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
    /// training follows this, including explicit changes made through
    /// [`set_learning_rate`](Self::set_learning_rate).
    pub fn train_config(&self) -> &CompletionTrainConfig {
        &self.train
    }

    /// Set the learning rate for subsequent updates and saved checkpoints.
    /// Preserves optimizer moments and the step counter. A non-finite or
    /// non-positive rate is rejected before changing either copy.
    pub fn set_learning_rate(&mut self, lr: f32) -> Result<()> {
        if !(lr.is_finite() && lr > 0.0) {
            return Err(Error::config(
                "CompletionTrainer::set_learning_rate: lr must be finite and positive".to_string(),
            ));
        }
        self.optimizer.set_learning_rate(lr);
        self.train.lr = lr;
        Ok(())
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

    /// The forward half with fingerprint and spectral evidence: like
    /// [`forward_batch_with_fingerprints`](Self::forward_batch_with_fingerprints)
    /// with one optional [`SpectrumEvidence`] per query. `fingerprints` is
    /// `None` for no fingerprint evidence (empty token sets). No device
    /// read.
    #[allow(clippy::type_complexity)]
    fn forward_batch_with_evidence(
        &self,
        patterns: &[&[MolGraph]],
        fingerprints: Option<&[SparseFingerprint]>,
        spectra: &[Option<&SpectrumEvidence>],
        traces: &[&[Token]],
        compositions: &[Composition],
    ) -> Result<(TeacherOutput<R, E>, Var<R, E>, Var<R, E>)> {
        if patterns.is_empty() {
            return Err(Error::config(
                "CompletionTrainer: cannot train on an empty query list".to_string(),
            ));
        }
        if patterns.len() != spectra.len() {
            return Err(Error::config(format!(
                "CompletionTrainer: {} pattern entries for {} spectra",
                patterns.len(),
                spectra.len()
            )));
        }
        let batch = PatternBatch::build(patterns, compositions)?;
        let slots = self.model.fingerprint_slots();
        let fp_batch = match fingerprints {
            Some(fingerprints) => {
                if patterns.len() != fingerprints.len() {
                    return Err(Error::config(format!(
                        "CompletionTrainer: {} pattern entries for {} fingerprints",
                        patterns.len(),
                        fingerprints.len()
                    )));
                }
                FingerprintBatch::build(fingerprints, slots)?
            }
            None => FingerprintBatch::empty(patterns.len(), slots)?,
        };
        let spectrum_batch = if self.model.has_spectrum() {
            Some(SpectrumBatch::build(spectra, self.model.spectrum_slots())?)
        } else if spectra.iter().any(Option::is_some) {
            return Err(Error::config(
                "CompletionTrainer: spectral evidence for a model without a spectrum encoder (spectrum_slots = 0)".to_string(),
            ));
        } else {
            None
        };
        let targets = TargetBatch::build_exact(traces, compositions, self.limits)?;
        let (out, loss) = self.model.teacher_with_evidence(
            &batch,
            &fp_batch,
            spectrum_batch.as_ref(),
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

    /// One optimizer step on caller-supplied patterns, fingerprints
    /// (`None` for none), spectral evidence, traces and compositions. No
    /// device read happens unless [`request_report`](Self::request_report)
    /// was called.
    pub fn step_with_evidence(
        &mut self,
        patterns: &[&[MolGraph]],
        fingerprints: Option<&[SparseFingerprint]>,
        spectra: &[Option<&SpectrumEvidence>],
        traces: &[&[Token]],
        compositions: &[Composition],
    ) -> Result<Option<f32>> {
        let (_, loss, packed) = self.forward_batch_with_evidence(
            patterns,
            fingerprints,
            spectra,
            traces,
            compositions,
        )?;
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

    /// Teacher-forced evaluation with fingerprint and spectral evidence: the
    /// per-query summed NLL under `no_grad`, in one batched read.
    pub fn teacher_eval_with_evidence(
        &mut self,
        patterns: &[&[MolGraph]],
        fingerprints: Option<&[SparseFingerprint]>,
        spectra: &[Option<&SpectrumEvidence>],
        traces: &[&[Token]],
        compositions: &[Composition],
    ) -> Result<Vec<f32>> {
        let _guard = no_grad();
        let (out, _, _) = self.forward_batch_with_evidence(
            patterns,
            fingerprints,
            spectra,
            traces,
            compositions,
        )?;
        out.nll.try_to_f32()
    }

    /// Per-step log-probabilities of the target's own actions under teacher
    /// forcing: the `[queries, T, 4]` field log-probabilities of
    /// [`TeacherOutput`], flattened, with `T` returned alongside. Summing a
    /// step's four fields gives the log-probability the model assigns to the
    /// target action at that step, which is what a target-survival curve
    /// needs. One batched read, under `no_grad`.
    pub fn teacher_steps_with_evidence(
        &mut self,
        patterns: &[&[MolGraph]],
        fingerprints: Option<&[SparseFingerprint]>,
        spectra: &[Option<&SpectrumEvidence>],
        traces: &[&[Token]],
        compositions: &[Composition],
    ) -> Result<(Vec<f32>, usize)> {
        let _guard = no_grad();
        let (out, _, _) = self.forward_batch_with_evidence(
            patterns,
            fingerprints,
            spectra,
            traces,
            compositions,
        )?;
        let steps = self.limits.max_steps();
        Ok((out.field_log_prob.try_to_f32()?, steps))
    }

    /// The resident replay tables of this trainer (for
    /// [`CompletionModel::generate`]).
    pub fn constants(&self) -> &Ms2Constants<R> {
        &self.constants
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

    /// [`CompletionTrainer::load`] with the per-step progress features
    /// switched on, so a checkpoint trained without them can be resumed with
    /// them.
    ///
    /// A strict load would refuse: the rebuilt model names a `progress.proj`
    /// weight the checkpoint has no entry for. This load is therefore
    /// **non-strict** when (and only when) the override adds the projection,
    /// which leaves that one parameter at its initialisation — zero, by
    /// [`ProgressConditioner::init`] — and loads every other parameter by
    /// name and shape exactly as the strict load does. The resumed model is
    /// therefore numerically the checkpoint at step 0, and the projection
    /// learns from there.
    ///
    /// A checkpoint that already carries the feature loads strictly, as does
    /// one asked for the setting it already has. Switching the feature
    /// *off* is refused: the checkpoint's `progress.proj` entry would be an
    /// unexpected one the rebuilt model cannot place, and dropping a trained
    /// parameter silently is not something this should do quietly.
    pub fn load_with_progress_features(
        path: &Path,
        device: &Device<R>,
        progress_features: bool,
    ) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        let checkpoint: CompletionCheckpoint = serde_json::from_slice(&bytes)?;
        if checkpoint.format != COMPLETION_CHECKPOINT_FORMAT {
            return Err(Error::config(format!(
                "CompletionTrainer::load_with_progress_features: unknown format {:?} (expected {COMPLETION_CHECKPOINT_FORMAT:?})",
                checkpoint.format
            )));
        }
        let had = checkpoint.model_config.progress_features;
        if had && !progress_features {
            return Err(Error::config(
                "CompletionTrainer::load_with_progress_features: the checkpoint carries a trained progress projection; loading it with the feature off would drop that parameter"
                    .to_string(),
            ));
        }
        let mut model_config = checkpoint.model_config.clone();
        model_config.progress_features = progress_features;
        model_config.validate().map_err(|e| {
            Error::config(format!(
                "CompletionTrainer::load_with_progress_features: checkpoint model config rejected: {e}"
            ))
        })?;
        let mut trainer = Self::new(&model_config, &checkpoint.train_config, device)?;
        // Non-strict exactly when the override added the projection.
        trainer
            .model
            .load_state_dict(&checkpoint.weights, had == progress_features)?;
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
    /// Shortlist size (`1..=1024`): how many distinct accepted identities
    /// the outcome keeps. The competition shortlist is 25, which the JSON
    /// protocol enforces on its own (`request.generation.returned`); a
    /// larger value here lets an evaluation keep the whole accepted pool and
    /// re-rank it before cutting to 25, instead of losing candidates to an
    /// early cut.
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
    /// `temperature` finite and positive, `returned` in `1..=1024`. The two
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
        if !(1..=1024).contains(&self.returned) {
            return Err(Error::config(format!(
                "CompletionGenerationConfig::validate: returned {} is not in 1..=1024",
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

/// Per-query accounting of one
/// [`generate_beam_with_spectra`](CompletionModel::generate_beam_with_spectra)
/// search, beside its [`QueryOutcome`].
///
/// The identities hold for every query:
///
/// * `candidates_scored == admitted_live + finished + candidates_dropped +
///   refused_tokens` — every legal action the search scored either entered
///   the beam, left it finished, or fell below the width's cut;
/// * `1 + admitted_live == rows_expanded + live_at_limit` — every row the
///   search ever held (the one it started with plus the admitted ones) was
///   either expanded or still live when the search stopped.
///
/// `candidates_dropped` is the number a wider beam would have had room for:
/// it is what the width cost.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BeamStats {
    /// Live rows the search expanded, summed over steps.
    pub rows_expanded: u64,
    /// Expanded rows with no legal action at all (a dead end).
    pub dead_end_rows: u64,
    /// Legal actions scored, summed over steps.
    pub candidates_scored: u64,
    /// Scored actions that entered the beam as live rows.
    pub admitted_live: u64,
    /// Scored actions below the width's cut.
    pub candidates_dropped: u64,
    /// Admitted actions the host grammar then refused (expected 0: the masks
    /// the candidate came from say it is legal).
    pub refused_tokens: u64,
    /// STOP actions taken: finished candidates handed to the host path.
    pub finished: u32,
    /// Live rows left when the search stopped.
    pub live_at_limit: u32,
    /// Steps run (the last step index the loop executed).
    pub steps_run: u32,
    /// Decoder rows actually executed: `width` times the steps run, whether
    /// or not the width was busy, because the device block keeps its uniform
    /// width. This is the work to compare against a sampler's `K` times its
    /// steps.
    pub row_steps: u64,
    /// The useful part of [`row_steps`](BeamStats::row_steps): live rows
    /// summed over steps.
    pub live_row_steps: u64,
}

/// One action a [`BeamAudit`] recorded: the host score the search ranked it
/// by, and the device score of the same action at the same context.
#[derive(Clone, Debug, PartialEq)]
pub struct AuditedAction {
    /// Step the action was committed at.
    pub step: u32,
    /// Device row the action was committed to (a child slot), or the parent
    /// row for a STOP, which takes no slot.
    pub row: u32,
    /// The action.
    pub token: Token,
    /// The joint log-probability the host scoring produced.
    pub host: f32,
    /// The same quantity from
    /// [`step_field_log_probs`](super::generate::step_field_log_probs) over
    /// the same step's heads: the four fields summed at the action's
    /// indices.
    pub device: f32,
    /// Whether this was a STOP (scored from the kind field alone).
    pub stop: bool,
}

/// Collector of [`generate_beam_audited`](CompletionModel::generate_beam_audited).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BeamAudit {
    /// Every committed action, in step order.
    pub actions: Vec<AuditedAction>,
}

impl BeamAudit {
    /// The largest `|host - device|` over the recorded actions, and the
    /// action it belongs to.
    pub fn worst(&self) -> Option<(f32, &AuditedAction)> {
        self.actions
            .iter()
            .map(|a| ((a.host - a.device).abs(), a))
            .fold(None, |worst, item| match worst {
                Some((d, _)) if d >= item.0 => worst,
                _ => Some(item),
            })
    }
}

/// One live row of a beam, host side: its cumulative trace log-probability,
/// the trace itself and the grammar state that trace reached.
struct BeamRow {
    /// Summed step log-probabilities of `trace`.
    cum: f32,
    /// The tokens applied so far, starting with START.
    trace: Vec<Token>,
    /// The grammar state after `trace`, which enumerates the legal actions.
    state: TraceState,
}

/// One legal continuation of one live row.
struct BeamCandidate {
    /// Device row the continuation extends.
    slot: usize,
    /// The action.
    token: Token,
    /// Its log-probability under this step's distribution.
    joint: f32,
    /// The parent's cumulative log-probability plus `joint`: what the beam
    /// ranks by.
    score: f32,
}

/// `log_softmax` over the legal set of one field, on the host: the maximum
/// and the sum run over the legal indices in increasing order, exactly as
/// [`ms2::field_log_probs`] computes them on the device, so the two agree to
/// floating-point rounding.
fn masked_shift(mask: u32, width: usize, value: impl Fn(usize) -> f32) -> f32 {
    let mut top = f32::NEG_INFINITY;
    for i in 0..width {
        if mask & (1 << i) != 0 {
            let v = value(i);
            if v > top {
                top = v;
            }
        }
    }
    let mut sum = 0.0f32;
    for i in 0..width {
        if mask & (1 << i) != 0 {
            sum += (value(i) - top).exp();
        }
    }
    top + sum.ln()
}

/// The pointer score of atom `i` under conditioning row `c` and bond row `b`,
/// from one packed sampler row: `pointer_base + pointer_by_type[c] +
/// pointer_by_bond[b]`, the sum [`ms2::field_log_probs`] masks.
fn pointer_score(head: &[f32], atoms: usize, c: usize, b: usize, i: usize) -> f32 {
    head[27 + i] + head[27 + atoms + c * atoms + i] + head[27 + atoms + 19 * atoms + b * atoms + i]
}

/// Every legal continuation of one live row, scored by the hierarchical
/// composition of architecture §3.6 and pushed onto `out`.
///
/// The legal sets come from the host grammar
/// ([`TraceState::masks`](super::grammar::TraceState::masks)), which is the
/// reference the device's [`ms2::step_masks`] is tested against, so no rule
/// is restated here; only the masked log-softmax is host arithmetic, and
/// [`generate_beam_audited`](CompletionModel::generate_beam_audited) checks
/// it against the device for every action the search commits.
///
/// Returns false when the row has no legal action at all — a dead end, which
/// takes no slot and ends that hypothesis.
fn beam_expand(
    slot: usize,
    row: &BeamRow,
    head: &[f32],
    bond_table: &[f32],
    atoms: usize,
    scratch: &mut BeamScratch,
    out: &mut Vec<BeamCandidate>,
) -> bool {
    let probe = |kind: u8, atom_type: u8, bond: u8| Token {
        kind,
        atom_type,
        bond,
        pointer: 0,
    };
    let kinds = row.state.masks(probe(super::grammar::PAD, 0, 0)).kinds;
    if kinds == 0 {
        return false;
    }
    let kind_shift = masked_shift(kinds, 5, |i| head[i]);
    let lp_kind = |kind: u8| head[usize::from(kind)] - kind_shift;
    let mut any = false;
    if kinds & (1 << super::grammar::STOP) != 0 {
        let joint = lp_kind(super::grammar::STOP);
        out.push(BeamCandidate {
            slot,
            token: probe(super::grammar::STOP, 0, 0),
            joint,
            score: row.cum + joint,
        });
        any = true;
    }
    if kinds & (1 << super::grammar::ADD_ATOM) != 0 {
        let kind = super::grammar::ADD_ATOM;
        // The root uses the kind and the atom type only (architecture §3.8);
        // it is the state's step 1, the same test the device makes.
        if row.state.step() == 1 {
            let types = row.state.masks(probe(kind, 0, 0)).atom_types;
            if types != 0 {
                let type_shift = masked_shift(types, 18, |i| head[5 + i]);
                for ty in 0..18usize {
                    if types & (1 << ty) == 0 {
                        continue;
                    }
                    let joint = lp_kind(kind) + (head[5 + ty] - type_shift);
                    out.push(BeamCandidate {
                        slot,
                        token: probe(kind, ty as u8, 0),
                        joint,
                        score: row.cum + joint,
                    });
                    any = true;
                }
            }
        } else {
            // Every `(type, bond, pointers)` of this row in one sweep of the
            // valence and feasibility checks: `masks` per pair would sweep
            // them again for every other pair.
            row.state.add_continuations(&mut scratch.adds);
            let mut types = 0u32;
            for &(ty, _, _) in scratch.adds.iter() {
                types |= 1 << u32::from(ty);
            }
            if types != 0 {
                let type_shift = masked_shift(types, 18, |i| head[5 + i]);
                let mut i = 0usize;
                while i < scratch.adds.len() {
                    // The rows of one atom type are contiguous, so its bond
                    // mask and its normaliser are built once.
                    let ty = scratch.adds[i].0;
                    let mut end = i;
                    let mut bonds = 0u32;
                    while end < scratch.adds.len() && scratch.adds[end].0 == ty {
                        bonds |= 1 << u32::from(scratch.adds[end].1);
                        end += 1;
                    }
                    let c = usize::from(ty);
                    let bond_of = |j: usize| head[23 + j] + bond_table[c * 4 + j];
                    let bond_shift = masked_shift(bonds, 4, bond_of);
                    let lp_type = head[5 + c] - type_shift;
                    for &(_, bond, pointers) in &scratch.adds[i..end] {
                        let b = usize::from(bond);
                        let lp_bond = bond_of(b) - bond_shift;
                        let ptr_of = |j: usize| pointer_score(head, atoms, c, b, j);
                        let ptr_shift = masked_shift(pointers, atoms, ptr_of);
                        for pointer in 0..atoms {
                            if pointers & (1 << pointer) == 0 {
                                continue;
                            }
                            let joint = lp_kind(kind)
                                + lp_type
                                + lp_bond
                                + (ptr_of(pointer) - ptr_shift);
                            out.push(BeamCandidate {
                                slot,
                                token: Token {
                                    kind,
                                    atom_type: ty,
                                    bond,
                                    pointer: pointer as u8,
                                },
                                joint,
                                score: row.cum + joint,
                            });
                            any = true;
                        }
                    }
                    i = end;
                }
            }
        }
    }
    if kinds & (1 << super::grammar::CLOSE_RING) != 0 {
        // CLOSE_RING uses the kind, bond and pointer; its conditioning row is
        // 18 and its atom-type field is unused (log-probability exactly 0 at
        // index 0).
        let kind = super::grammar::CLOSE_RING;
        let c = ms2::SAMPLE_COND_ROWS - 1;
        row.state.close_continuations(&mut scratch.closes);
        let mut bonds = 0u32;
        for &(bond, _) in scratch.closes.iter() {
            bonds |= 1 << u32::from(bond);
        }
        if bonds != 0 {
            let bond_of = |j: usize| head[23 + j] + bond_table[c * 4 + j];
            let bond_shift = masked_shift(bonds, 4, bond_of);
            for &(bond, pointers) in scratch.closes.iter() {
                let b = usize::from(bond);
                let lp_bond = bond_of(b) - bond_shift;
                let ptr_of = |j: usize| pointer_score(head, atoms, c, b, j);
                let ptr_shift = masked_shift(pointers, atoms, ptr_of);
                for pointer in 0..atoms {
                    if pointers & (1 << pointer) == 0 {
                        continue;
                    }
                    let joint = lp_kind(kind) + lp_bond + (ptr_of(pointer) - ptr_shift);
                    out.push(BeamCandidate {
                        slot,
                        token: Token {
                            kind,
                            atom_type: 0,
                            bond,
                            pointer: pointer as u8,
                        },
                        joint,
                        score: row.cum + joint,
                    });
                    any = true;
                }
            }
        }
    }
    any
}

/// Reused enumeration scratch of one beam step, so expanding a row allocates
/// nothing.
#[derive(Default)]
struct BeamScratch {
    /// `(atom type, bond, pointer mask)` of
    /// [`TraceState::add_continuations`].
    adds: Vec<(u8, u8, u32)>,
    /// `(bond, pointer mask)` of [`TraceState::close_continuations`].
    closes: Vec<(u8, u32)>,
}

/// The host half of a completion search over one query's trajectories:
/// status accounting, exact replay, acceptance under the active
/// [`SubstructureSemantics`], identity merging and the ranked cut.
///
/// This is the body [`CompletionModel::generate_with_spectra`] ran inline,
/// extracted unchanged so that a search over the same model
/// ([`CompletionModel::generate_beam_with_spectra`]) reaches the same
/// acceptance and ranking instead of a second copy of them.
///
/// `trajectories` is every trajectory of the query, in the order the caller
/// wants them reported (the sampler's `K` rows in trajectory order, or a
/// beam's finished candidates in the order it found them): it becomes
/// [`QueryOutcome::sampled`], and its length
/// [`QueryOutcome::trajectories`]. A FINISHED trajectory whose trace is
/// empty is counted `rejected_replay` — for the sampler that is a record
/// whose words did not decode, and a decodable FINISHED record always holds
/// at least the START token, so this is the inline behaviour unchanged.
/// `query` names the query in the debug assertions only.
#[allow(clippy::too_many_arguments)]
fn accept_trajectories(
    trajectories: Vec<SampledTrajectory>,
    request: &CompletionRequest,
    limits: Limits,
    semantics: SubstructureSemantics,
    containment_limit: usize,
    identity_limit: usize,
    returned: usize,
    query: usize,
) -> QueryOutcome {
    let mut outcome = QueryOutcome {
        candidates: Vec::new(),
        unresolved: Vec::new(),
        sampled: Vec::new(),
        distinct: 0,
        trajectories: trajectories.len() as u32,
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
    for traj in trajectories.iter() {
        let status = traj.status;
        let log_prob = traj.log_prob;
        let trace = traj.trace.clone();
            if status & candidate_status::FINISHED != 0 {
                outcome.finished += 1;
            } else if status & candidate_status::NO_VALID_ACTION != 0 {
                outcome.dead_end += 1;
            } else if status & candidate_status::TRUNCATED != 0 {
                outcome.truncated += 1;
            } else {
                outcome.other_status += 1;
            }
            if status & candidate_status::FINISHED == 0 {
                continue;
            }
            if trace.is_empty() {
                // A finished trajectory with no trace (a record whose
                // words did not decode, or a caller that supplied none)
                // is a rejected candidate, never a silently dropped one.
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
                    identity_limit,
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
            "query {query}: every trajectory carries exactly one status class"
        );
        let identity_samples: u32 = identities.iter().map(|id| id.samples).sum();
        let unresolved_samples: u32 = unresolved.iter().map(|entry| entry.samples).sum();
        debug_assert_eq!(
            outcome.identity_unresolved, unresolved_samples,
            "query {query}: identity_unresolved counts every unresolved trajectory once"
        );
        debug_assert_eq!(
            outcome.finished,
            outcome.rejected_replay
                + outcome.rejected_containment
                + outcome.containment_unresolved
                + identity_samples
                + unresolved_samples,
            "query {query}: finished trajectories are rejected, certified or unresolved"
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
    outcome.sampled = trajectories;
    outcome
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
        let none: Vec<Option<&SpectrumEvidence>> = vec![None; requests.len()];
        self.generate_with_spectra(requests, &none, config, constants, device)
    }

    /// [`generate`](Self::generate) with one optional [`SpectrumEvidence`]
    /// per request (`spectra[i]` conditions `requests[i]`). The evidence only
    /// conditions the decoder: legality, acceptance, ranking and accounting
    /// are exactly those of `generate`. A model without a spectrum encoder
    /// given any evidence is [`Error::Config`], as is a length mismatch.
    pub fn generate_with_spectra(
        &self,
        requests: &[CompletionRequest],
        spectra: &[Option<&SpectrumEvidence>],
        config: &CompletionGenerationConfig,
        constants: &Ms2Constants<R>,
        device: &Device<R>,
    ) -> Result<Vec<QueryOutcome>> {
        config.validate()?;
        if spectra.len() != requests.len() {
            return Err(Error::config(format!(
                "CompletionModel::generate: {} spectra for {} requests",
                spectra.len(),
                requests.len()
            )));
        }
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
        let spectrum_batch = self.spectrum_batch_for(spectra, &feasible)?;
        let encoded =
            self.encode_with_evidence(&batch, &fp_batch, spectrum_batch.as_ref(), device)?;
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
        // The per-query pattern type counts the progress features read, when
        // the model owns them: one upload before the loop, nothing per step
        // beyond the feature kernel and its projection.
        let progress_counts = match &self.progress {
            Some(_) => Some(IdTensor::from_slice(
                &batch.type_counts(),
                vec![b, 18],
                device,
            )?),
            None => None,
        };
        for step in 1..steps {
            let _tally = crate::backend::tally_scope("ms2.step");
            ms2::step_token(&actions, &mut step_token, steps, atoms)?;
            // The step's conditioning: the composition row, plus the
            // projected progress of the prefix this step continues. With the
            // feature off this is `traj_formula` itself, unchanged.
            let conditioning = match (&self.progress, &progress_counts) {
                (Some(conditioner), Some(counts)) => conditioner.step_conditioning(
                    &traj_formula,
                    &replay,
                    &traj_meta,
                    counts,
                    k,
                    steps,
                    device,
                )?,
                _ => traj_formula.clone(),
            };
            composed_decode_step(
                &self.decoder,
                &encoded,
                &conditioning,
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
            // The record words of this query's `k` rows, decoded into the
            // trajectory list the shared host path takes. A row whose words
            // do not decode keeps an empty trace, which that path counts as
            // `rejected_replay`, exactly as this loop did inline.
            let mut sampled: Vec<SampledTrajectory> = Vec::with_capacity(k);
            for kk in 0..k {
                let base = (q * k + kk) * width;
                let tail = base + steps * 4 + atoms;
                let length = record[tail] as usize;
                let status = record[tail + 1];
                let log_prob = f32::from_bits(record[tail + 2]);
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
                sampled.push(SampledTrajectory {
                    trajectory: kk as u32,
                    trace,
                    log_prob,
                    status,
                });
            }
            let outcome = accept_trajectories(
                sampled,
                request,
                limits,
                semantics,
                containment_limit,
                identity_limit,
                returned,
                q,
            );
            outcomes[qi] = Some(outcome);
        }
        Ok(outcomes
            .into_iter()
            .map(|outcome| outcome.expect("every outcome filled"))
            .collect())
    }

    /// Search a ranked shortlist of complete molecules with a **beam** of
    /// `width` rows per query instead of `trajectories` independent samples.
    ///
    /// Same conditioning as [`generate_with_spectra`](Self::generate_with_spectra)
    /// (patterns, fingerprints, spectral evidence and the request's
    /// composition), the same grammar, and the same host acceptance, identity
    /// merging and ranking — [`accept_trajectories`] is the one copy of that
    /// path — behind the same [`QueryOutcome`], with one [`BeamStats`] per
    /// request beside it. `config.temperature` and `config.seed` are unused:
    /// nothing is drawn.
    ///
    /// Why it reaches molecules sampling misses: a sample's trace probability
    /// is the product of its per-step probabilities, so a single unlikely
    /// step costs the whole trace, while a beam keeps an action by *rank* and
    /// pays no product. Per step, per query, the search
    ///
    /// 1. runs the one decoder pass of the step
    ///    ([`composed_step_heads`](super::generate::composed_step_heads)) and
    ///    reads its packed head row;
    /// 2. enumerates every legal action of every live row with the host
    ///    grammar
    ///    ([`add_continuations`](super::grammar::TraceState::add_continuations)
    ///    and [`close_continuations`](super::grammar::TraceState::close_continuations),
    ///    the one-sweep form of
    ///    [`TraceState::masks`](super::grammar::TraceState::masks) — the
    ///    reference the device's [`ms2::step_masks`] is tested against) and
    ///    scores each one with the hierarchical composition of architecture
    ///    §3.6: `log P(kind) + log P(type | kind) + log P(bond | kind, type)
    ///    + log P(pointer | kind, type, bond)`, each over its own legal set
    ///    at temperature 1;
    /// 3. ranks the query's candidates by cumulative trace log-probability
    ///    and walks them in that order: a STOP leaves the beam as a finished
    ///    candidate (freeing its slot, so the width stays busy with live
    ///    continuations), any other action takes the next slot, and the walk
    ///    stops once `width` live rows are filled — the candidates below that
    ///    cut are [`BeamStats::candidates_dropped`], which is what the width
    ///    cost;
    /// 4. gathers the per-row state to the chosen parents
    ///    ([`DecoderState::gather`](super::decoder::DecoderState::gather) and
    ///    [`gather_generation_rows`](super::generate::gather_generation_rows))
    ///    and commits the chosen actions
    ///    ([`apply_chosen_action`](super::generate::apply_chosen_action)).
    ///
    /// The search stops when no query has a live row, or at the step limit.
    /// Slots of a query with fewer than `width` live hypotheses hold an inert
    /// copy of a live row: the device block stays `B * width` rows so
    /// `rows_per_spectrum` is uniform, and the host never reads them. That is
    /// also why [`BeamStats::row_steps`] — the decoder rows actually executed
    /// — is `width` times the steps run whether or not the width was busy,
    /// with [`BeamStats::live_row_steps`] beside it as the useful part.
    ///
    /// A STOP takes no device slot: the finished trace is complete on the
    /// host, so the device action records hold the live rows only, and the
    /// outcome is built from the host traces.
    ///
    /// Reads: one of the `bond_by_type` table before the loop, then three per
    /// step (the packed heads, and the parent vector each of the two gathers
    /// validates).
    ///
    /// Why the enumeration is host side. The action is factored, so a row's
    /// candidates need the conditionals at many contexts, while
    /// [`composed_step_log_probs`](super::generate::composed_step_log_probs)
    /// masks one context per row. The alternative is one scoring call per
    /// field level — four decoder passes per step, with the rows re-indexed
    /// between levels because a row expands into several partial actions.
    /// This search instead runs the decoder **once** per step and composes
    /// the four conditionals from the packed heads on the host: the legal
    /// sets still come from the shared host grammar, so no rule is restated,
    /// and the only host arithmetic is the masked log-softmax, which
    /// [`generate_beam_audited`](Self::generate_beam_audited) checks against
    /// the device for every action the search commits (measured equal to the
    /// last bit in `tests/ms2_completion_beam.rs`). Four passes would have
    /// cost four times the decoder work to avoid twenty lines of host
    /// arithmetic that a test can pin exactly.
    ///
    /// Infeasible requests behave exactly as in
    /// [`generate_with_spectra`](Self::generate_with_spectra): no device
    /// work, [`infeasible`](QueryOutcome::infeasible) set, every counter 0.
    pub fn generate_beam_with_spectra(
        &self,
        requests: &[CompletionRequest],
        spectra: &[Option<&SpectrumEvidence>],
        config: &CompletionGenerationConfig,
        width: u32,
        constants: &Ms2Constants<R>,
        device: &Device<R>,
    ) -> Result<(Vec<QueryOutcome>, Vec<BeamStats>)> {
        self.beam_inner(requests, spectra, config, width, constants, device, None)
    }

    /// [`generate_beam_with_spectra`](Self::generate_beam_with_spectra)
    /// without spectral evidence, as [`generate`](Self::generate) is to
    /// [`generate_with_spectra`](Self::generate_with_spectra).
    pub fn generate_beam(
        &self,
        requests: &[CompletionRequest],
        config: &CompletionGenerationConfig,
        width: u32,
        constants: &Ms2Constants<R>,
        device: &Device<R>,
    ) -> Result<(Vec<QueryOutcome>, Vec<BeamStats>)> {
        let none: Vec<Option<&SpectrumEvidence>> = vec![None; requests.len()];
        self.generate_beam_with_spectra(requests, &none, config, width, constants, device)
    }

    /// [`generate_beam_with_spectra`](Self::generate_beam_with_spectra) with
    /// a [`BeamAudit`]: for every action the search commits it records the
    /// host score the action was ranked by beside the device score of the
    /// same action at the same context, from
    /// [`step_field_log_probs`](super::generate::step_field_log_probs) over
    /// the heads of that very step — the arithmetic of
    /// [`composed_step_log_probs`](super::generate::composed_step_log_probs),
    /// without a second decoder pass.
    ///
    /// This is what pins the host composition to the device one. A search
    /// whose composition is wrong ranks actions wrongly while every other
    /// property still holds, so nothing else would reveal it. Auditing costs
    /// one row gather, ten launches and two reads per step; it is for tests
    /// and diagnostics, not production.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_beam_audited(
        &self,
        requests: &[CompletionRequest],
        spectra: &[Option<&SpectrumEvidence>],
        config: &CompletionGenerationConfig,
        width: u32,
        constants: &Ms2Constants<R>,
        device: &Device<R>,
        audit: &mut BeamAudit,
    ) -> Result<(Vec<QueryOutcome>, Vec<BeamStats>)> {
        self.beam_inner(
            requests,
            spectra,
            config,
            width,
            constants,
            device,
            Some(audit),
        )
    }

    /// The body behind the three beam entry points.
    #[allow(clippy::too_many_arguments)]
    fn beam_inner(
        &self,
        requests: &[CompletionRequest],
        spectra: &[Option<&SpectrumEvidence>],
        config: &CompletionGenerationConfig,
        width: u32,
        constants: &Ms2Constants<R>,
        device: &Device<R>,
        mut audit: Option<&mut BeamAudit>,
    ) -> Result<(Vec<QueryOutcome>, Vec<BeamStats>)> {
        config.validate()?;
        if spectra.len() != requests.len() {
            return Err(Error::config(format!(
                "CompletionModel::generate_beam: {} spectra for {} requests",
                spectra.len(),
                requests.len()
            )));
        }
        if width == 0 {
            return Err(Error::config(
                "CompletionModel::generate_beam: width 0 keeps no row".to_string(),
            ));
        }
        if requests.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }
        let mut seen_ids: HashMap<u64, usize> = HashMap::with_capacity(requests.len());
        for (i, request) in requests.iter().enumerate() {
            if let Some(&first) = seen_ids.get(&request.id) {
                return Err(Error::config(format!(
                    "CompletionModel::generate_beam: requests {first} and {i} share id {} (ids must be distinct within a call)",
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
                "CompletionModel::generate_beam: model limits rejected: {e}"
            ))
        })?;
        let k = width as usize;
        let returned = config.returned as usize;
        let semantics = config.substructure_semantics;
        let containment_limit = config.containment_node_limit as usize;
        let identity_limit = config.identity_work_limit as usize;
        // The same necessary composition fit the sampler applies, with the
        // same consequence: an infeasible request does no device work.
        let mut infeasible: Vec<Option<String>> = Vec::with_capacity(requests.len());
        for request in requests.iter() {
            infeasible.push(check_feasibility(
                &request.composition,
                request.acceptance_patterns.unwrap_or(request.patterns),
                semantics,
            ));
        }
        let feasible: Vec<usize> = (0..requests.len())
            .filter(|&i| infeasible[i].is_none())
            .collect();
        let mut outcomes: Vec<Option<QueryOutcome>> = Vec::with_capacity(requests.len());
        let mut reports: Vec<BeamStats> = Vec::with_capacity(requests.len());
        for _ in 0..requests.len() {
            outcomes.push(None);
            reports.push(BeamStats::default());
        }
        for (i, reason) in infeasible.iter().enumerate() {
            if let Some(reason) = reason {
                let mut outcome = accept_trajectories(
                    Vec::new(),
                    &requests[i],
                    limits,
                    semantics,
                    containment_limit,
                    identity_limit,
                    returned,
                    i,
                );
                outcome.infeasible = Some(reason.clone());
                outcomes[i] = Some(outcome);
            }
        }
        if feasible.is_empty() {
            return Ok((
                outcomes
                    .into_iter()
                    .map(|outcome| outcome.expect("every outcome filled"))
                    .collect(),
                reports,
            ));
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
        let spectrum_batch = self.spectrum_batch_for(spectra, &feasible)?;
        let encoded =
            self.encode_with_evidence(&batch, &fp_batch, spectrum_batch.as_ref(), device)?;
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
        let mut traj_meta =
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
        let logits_width = ms2::sample_logits_width(atoms);
        let mut scored = StepLogProbs::<R, E>::new(rows, atoms, device);
        let mut state = self.decoder.start_state(&encoded, rows, device)?;
        let bond_table = self.decoder.bond_by_type_value();
        // The one table the host scoring needs beyond the packed heads, read
        // once before the loop.
        let bond_host = bond_table.try_to_f32()?;
        let audit_pack = match audit {
            Some(_) => {
                let stop_row: Vec<u32> = (0..rows)
                    .flat_map(|_| [u32::from(super::grammar::STOP), 0, 0, 0])
                    .collect();
                Some((
                    StepLogProbs::<R, E>::new(rows, atoms, device),
                    IdTensor::from_slice(&stop_row, vec![rows, 4], device)?,
                ))
            }
            None => None,
        };
        let mut audit_pack = audit_pack;
        // The per-query pattern type counts the progress features read, when
        // the model owns them (see the sampler).
        let progress_counts = match &self.progress {
            Some(_) => Some(IdTensor::from_slice(
                &batch.type_counts(),
                vec![b, 18],
                device,
            )?),
            None => None,
        };
        // Host rows: slot `q * k + i` of the device block. Row 0 of every
        // query starts live with START applied, which is what the uploaded
        // rows carry; the others are inert until the width fills.
        let mut live: Vec<Option<BeamRow>> = (0..rows).map(|_| None).collect();
        let start_token = Token {
            kind: super::grammar::START,
            atom_type: 0,
            bond: 0,
            pointer: 0,
        };
        for (q, composition) in compositions.iter().enumerate() {
            let mut host_state = TraceState::new_exact(limits, *composition);
            host_state.apply(start_token).map_err(|e| {
                Error::config(format!(
                    "CompletionModel::generate_beam: the grammar refused START for query {q}: {e}"
                ))
            })?;
            live[q * k] = Some(BeamRow {
                cum: 0.0,
                trace: vec![start_token],
                state: host_state,
            });
        }
        let mut finished: Vec<Vec<(Vec<Token>, f32)>> = (0..b).map(|_| Vec::new()).collect();
        let mut stats: Vec<BeamStats> = (0..b).map(|_| BeamStats::default()).collect();
        let mut candidates: Vec<BeamCandidate> = Vec::new();
        let mut scratch = BeamScratch::default();
        for step in 1..steps {
            if live.iter().all(|row| row.is_none()) {
                break;
            }
            ms2::step_token(&actions, &mut step_token, steps, atoms)?;
            // The step's conditioning, exactly as the sampler builds it, so
            // the search scores what the sampler would draw from.
            let conditioning = match (&self.progress, &progress_counts) {
                (Some(conditioner), Some(counts)) => conditioner.step_conditioning(
                    &traj_formula,
                    &replay,
                    &traj_meta,
                    counts,
                    k,
                    steps,
                    device,
                )?,
                _ => traj_formula.clone(),
            };
            composed_step_heads(
                &self.decoder,
                &encoded,
                &conditioning,
                &step_token,
                &replay,
                &mut state,
                &mut scored.logits,
                step,
                atoms,
                k,
                rows,
            )?;
            // The step's heads on the host: every legal action of every live
            // row is scored from this one packed row.
            let heads = scored.logits.try_to_f32()?;
            let mut parents = vec![0u32; rows];
            let mut tokens = vec![0u32; rows * 4];
            let mut step_lp = vec![0.0f32; rows];
            let mut next: Vec<Option<BeamRow>> = (0..rows).map(|_| None).collect();
            // Admitted STOPs of this step, for the audit: they take no slot,
            // so they are checked against the parent row's kind field.
            let mut stopped: Vec<(usize, f32)> = Vec::new();
            for q in 0..b {
                let base = q * k;
                candidates.clear();
                let mut live_here = 0usize;
                for i in 0..k {
                    let Some(row) = &live[base + i] else { continue };
                    live_here += 1;
                    stats[q].rows_expanded += 1;
                    let head = &heads[(base + i) * logits_width..(base + i + 1) * logits_width];
                    if !beam_expand(
                        base + i,
                        row,
                        head,
                        &bond_host,
                        atoms,
                        &mut scratch,
                        &mut candidates,
                    ) {
                        stats[q].dead_end_rows += 1;
                    }
                }
                stats[q].live_row_steps += live_here as u64;
                stats[q].candidates_scored += candidates.len() as u64;
                // A total order — score, then the parent slot, then the token
                // words — so the same inputs give the same beam twice.
                candidates.sort_by(|x, y| {
                    y.score
                        .total_cmp(&x.score)
                        .then_with(|| x.slot.cmp(&y.slot))
                        .then_with(|| x.token.cmp(&y.token))
                });
                let mut filled = 0usize;
                let mut walked = 0usize;
                for candidate in candidates.iter() {
                    if filled == k {
                        break;
                    }
                    walked += 1;
                    let Some(parent) = &live[candidate.slot] else {
                        continue;
                    };
                    let mut trace = parent.trace.clone();
                    trace.push(candidate.token);
                    if candidate.token.kind == super::grammar::STOP {
                        // Out of the beam as a finished candidate; its slot
                        // goes to the next-best live continuation.
                        finished[q].push((trace, candidate.score));
                        stats[q].finished += 1;
                        stopped.push((candidate.slot, candidate.joint));
                        continue;
                    }
                    let mut host_state = parent.state.clone();
                    if host_state.apply(candidate.token).is_err() {
                        // The mask the candidate came from says it is legal,
                        // so this cannot happen; count it instead of
                        // trusting it.
                        stats[q].refused_tokens += 1;
                        continue;
                    }
                    let slot = base + filled;
                    parents[slot] = candidate.slot as u32;
                    tokens[slot * 4] = u32::from(candidate.token.kind);
                    tokens[slot * 4 + 1] = u32::from(candidate.token.atom_type);
                    tokens[slot * 4 + 2] = u32::from(candidate.token.bond);
                    tokens[slot * 4 + 3] = u32::from(candidate.token.pointer);
                    step_lp[slot] = candidate.joint;
                    next[slot] = Some(BeamRow {
                        cum: candidate.score,
                        trace,
                        state: host_state,
                    });
                    filled += 1;
                    stats[q].admitted_live += 1;
                }
                stats[q].candidates_dropped += (candidates.len() - walked) as u64;
                // The query's unused slots hold an inert copy of its first
                // live child (of its row 0 when it has no live row left):
                // the device block keeps its uniform width, and the host
                // never reads these rows again.
                let (pad_parent, pad_token) = if filled > 0 {
                    (
                        parents[base],
                        [
                            tokens[base * 4],
                            tokens[base * 4 + 1],
                            tokens[base * 4 + 2],
                            tokens[base * 4 + 3],
                        ],
                    )
                } else {
                    (base as u32, [u32::from(super::grammar::STOP), 0, 0, 0])
                };
                for i in filled..k {
                    let slot = base + i;
                    parents[slot] = pad_parent;
                    tokens[slot * 4..slot * 4 + 4].copy_from_slice(&pad_token);
                }
            }
            let parent_ids = IdTensor::from_slice(&parents, vec![rows], device)?;
            let token_ids = IdTensor::from_slice(&tokens, vec![rows, 4], device)?;
            if let (Some(audit), Some((pack, stop_context))) =
                (audit.as_mut(), audit_pack.as_mut())
            {
                // Before the gather the rows are the parents: the device's
                // own `log P(STOP | state)` per parent row, against the host
                // score of every STOP this step admitted.
                pack.logits = scored.logits.clone();
                step_field_log_probs(
                    &replay,
                    &traj_meta,
                    stop_context,
                    &bond_table,
                    &constants.atom_table,
                    pack,
                    atoms,
                    closures,
                )?;
                let kind_lp = pack.kind.try_to_f32()?;
                for &(slot, host) in stopped.iter() {
                    audit.actions.push(AuditedAction {
                        step: step as u32,
                        row: slot as u32,
                        token: Token {
                            kind: super::grammar::STOP,
                            atom_type: 0,
                            bond: 0,
                            pointer: 0,
                        },
                        host,
                        device: kind_lp[slot * ms2::SAMPLE_KIND_WIDTH
                            + usize::from(super::grammar::STOP)],
                        stop: true,
                    });
                }
            }
            let gathered = gather_generation_rows(
                &actions,
                &replay,
                &step_token,
                &traj_meta,
                &parent_ids,
                // Children of one parent need distinct draw keys only for a
                // sampler; nothing is drawn here, so every child keeps its
                // query's key and the trajectory rows stay as uploaded.
                &(0..rows).map(|slot| ids[slot / k]).collect::<Vec<u64>>(),
                steps,
                atoms,
                device,
            )?;
            state = state.gather(&parent_ids, device)?;
            actions = gathered.actions;
            replay = gathered.replay;
            step_token = gathered.step_token;
            traj_meta = gathered.traj_meta;
            if let (Some(audit), Some((pack, _))) = (audit.as_mut(), audit_pack.as_mut()) {
                // After the gather the rows are the children: each one's
                // grammar row is its parent's, so the four fields at the
                // committed token's indices are the device's score for the
                // action the search chose. The heads are gathered the same
                // way, so row `i` scores child `i`.
                pack.logits = gather_rows(&scored.logits, &parent_ids)?;
                step_field_log_probs(
                    &replay,
                    &traj_meta,
                    &token_ids,
                    &bond_table,
                    &constants.atom_table,
                    pack,
                    atoms,
                    closures,
                )?;
                let kind_lp = pack.kind.try_to_f32()?;
                let type_lp = pack.atom_type.try_to_f32()?;
                let bond_lp = pack.bond.try_to_f32()?;
                let ptr_lp = pack.pointer.try_to_f32()?;
                for (slot, row) in next.iter().enumerate() {
                    if row.is_none() {
                        continue;
                    }
                    let token = Token {
                        kind: tokens[slot * 4] as u8,
                        atom_type: tokens[slot * 4 + 1] as u8,
                        bond: tokens[slot * 4 + 2] as u8,
                        pointer: tokens[slot * 4 + 3] as u8,
                    };
                    let device_lp = kind_lp
                        [slot * ms2::SAMPLE_KIND_WIDTH + usize::from(token.kind)]
                        + type_lp[slot * ms2::SAMPLE_TYPE_WIDTH + usize::from(token.atom_type)]
                        + bond_lp[slot * ms2::SAMPLE_BOND_WIDTH + usize::from(token.bond)]
                        + ptr_lp[slot * atoms + usize::from(token.pointer)];
                    audit.actions.push(AuditedAction {
                        step: step as u32,
                        row: slot as u32,
                        token,
                        host: step_lp[slot],
                        device: device_lp,
                        stop: false,
                    });
                }
            }
            let step_lp_t = Tensor::<R, E>::from_f32(&step_lp, vec![rows], device)?;
            apply_chosen_action(
                &token_ids,
                &mut actions,
                &mut step_token,
                &mut replay,
                &traj_meta,
                &mut state,
                &step_lp_t,
                &constants.atom_table,
                step,
                steps,
                atoms,
                closures,
            )?;
            live = next;
            for q in 0..b {
                stats[q].row_steps += k as u64;
                stats[q].steps_run = step as u32;
            }
        }
        for q in 0..b {
            stats[q].live_at_limit = live[q * k..(q + 1) * k]
                .iter()
                .filter(|row| row.is_some())
                .count() as u32;
        }
        for (q, &qi) in feasible.iter().enumerate() {
            // The finished candidates in the order the search found them,
            // each carrying its whole-trace log-probability: the shared host
            // path replays, accepts, merges identities and ranks them.
            let trajectories: Vec<SampledTrajectory> = finished[q]
                .iter()
                .enumerate()
                .map(|(i, (trace, log_prob))| SampledTrajectory {
                    trajectory: i as u32,
                    trace: trace.clone(),
                    log_prob: *log_prob,
                    status: candidate_status::FINISHED,
                })
                .collect();
            outcomes[qi] = Some(accept_trajectories(
                trajectories,
                &requests[qi],
                limits,
                semantics,
                containment_limit,
                identity_limit,
                returned,
                q,
            ));
            reports[qi] = stats[q].clone();
        }
        Ok((
            outcomes
                .into_iter()
                .map(|outcome| outcome.expect("every outcome filled"))
                .collect(),
            reports,
        ))
    }
}
