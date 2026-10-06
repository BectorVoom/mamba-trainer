//! Graph-action decoder (architecture §4.3): teacher forcing, factor heads and
//! the graph loss (§4.4).
//!
//! [`Ms2Decoder::teacher`] runs the parallel teacher-forced pass over
//! `[B*G, T, d]`: static input embeddings, [`Mamba3Block`] layers with
//! per-layer cross-attention into the spectrum memory, and the four factor
//! heads scored against the replay masks. [`graph_loss`] is the
//! `q`-weighted mean negative log-likelihood. [`Ms2Decoder::step_logits`] is
//! the single-position form the sampler drives: it advances the recurrent
//! caches and returns the unmasked head outputs of architecture §3.6.
//!
//! No device read happens in `teacher` or `graph_loss`: ids stay on the
//! device, tables are read with [`Var::ms2_lookup`], and unused fields use
//! the all-masked rule of architecture §3.8 (mask "index 0 only", target id
//! 0, gathered value multiplied by the field's use indicator).

use cubecl::prelude::Runtime;

use crate::autograd::{Var, cat};
use crate::backend::{Device, FloatElem};
use crate::error::{Error, Result};
use crate::models::mamba3::{Mamba3Block, Mamba3BlockConfig, MixerCache};
use crate::tensor::ops::mixer_step::MixerStepBuffers;
use crate::nn::init::Initializer;
use crate::nn::linear::{Linear, LinearConfig};
use crate::nn::module::{Module, ModuleVisitor};
use crate::nn::norm::{RmsNorm, RmsNormConfig};
use crate::nn::param::Param;
use crate::tensor::Shape;
use crate::tensor::Tensor;
use crate::tensor::ops::index::{IdTensor, slice_ids_along};
use crate::tensor::ops::matmul::{matmul, matmul_nt};
use crate::tensor::ops::ms2::{
    TEACHER_PLAN_HEAD, atom_key_update, atom_memory_update, attn_context, attn_weights,
    effective_mask, pred_ids, step_embed, step_head_layout, step_logits_pack, teacher_plan,
};
use crate::tensor::ops::random::Rng;
use crate::tensor::ops::{elemwise, movement};

use super::contract::ModelConfig;
use super::encoder::EncoderOutput;
use super::targets_batch::{DevicePacking, DeviceTargets};

/// Step-embedding rows: `T <= 64` by contract §3.4, so a fixed table covers
/// every generation config without knowing `T` at init.
const STEP_ROWS: usize = 64;

/// Atom-type rows of the pointer conditioning tables: the 17 types plus row 0
/// (unused) and row 18 (CLOSE_RING).
const COND_ROWS: usize = 19;

/// Whether `l` is a bare weight and bias: no LoRA adapter and no quantizer,
/// so a fused kernel reading its parameters computes what `apply` does.
fn plain_linear<R: Runtime, E: FloatElem>(l: &Linear<R, E>) -> bool {
    l.lora().is_none() && l.weight_quantizer().is_none() && l.activation_quantizer().is_none()
}

/// How the rows of a teacher pass relate to the spectra of the encoder output.
#[derive(Clone, Copy)]
enum TeacherLayout<'a, R: Runtime, E: FloatElem> {
    /// `B * G` rows, `G` consecutive ones per spectrum.
    Padded,
    /// Virtual spectra: the owner `[B']` of each ([`Ms2Decoder::teacher_grouped`]).
    Grouped(&'a IdTensor<R>),
    /// One trace per row, the layers over packed rows
    /// ([`Ms2Decoder::teacher_packed`]).
    Packed(&'a DevicePacking<R, E>),
}

/// Rows of `x` (`[B, ..]`) picked by `owner` (`[B']`): `[B', ..]`, with the
/// gradient of a row summed over the virtual spectra that share it.
fn gather_spectra<R: Runtime, E: FloatElem>(
    x: &Var<R, E>,
    owner: &IdTensor<R>,
) -> Result<Var<R, E>> {
    let mut dims = x.dims().to_vec();
    let stored = dims[0];
    let width: usize = dims[1..].iter().product();
    dims[0] = owner.len();
    Var::ms2_lookup(&x.reshape(vec![stored, width])?, owner)?.reshape(dims)
}

/// One decoder layer: a [`Mamba3Block`] followed by cross-attention
/// (`q, k, v, o`: `Linear(d → d)`, no bias, `attention_heads` heads) over the
/// spectrum memory.
struct DecoderLayer<R: Runtime, E: FloatElem> {
    mixer: Mamba3Block<R, E>,
    norm: RmsNorm<R, E>,
    q: Linear<R, E>,
    k: Linear<R, E>,
    v: Linear<R, E>,
    o: Linear<R, E>,
}

impl<R: Runtime, E: FloatElem> Module<R, E> for DecoderLayer<R, E> {
    fn visit(&self, visitor: &mut ModuleVisitor<'_, R, E>) {
        visitor.child("mixer", &self.mixer);
        visitor.child("norm", &self.norm);
        visitor.child("q", &self.q);
        visitor.child("k", &self.k);
        visitor.child("v", &self.v);
        visitor.child("o", &self.o);
    }
}

/// The graph-action decoder of architecture §4.3.
pub struct Ms2Decoder<R: Runtime, E: FloatElem> {
    kind_emb: Param<R, E>,
    type_emb: Param<R, E>,
    bond_emb: Param<R, E>,
    ptr_emb: Param<R, E>,
    step_emb: Param<R, E>,
    layers: Vec<DecoderLayer<R, E>>,
    kind_head: Linear<R, E>,
    type_head: Linear<R, E>,
    bond_head: Linear<R, E>,
    mem_proj: Linear<R, E>,
    ptr_query: Linear<R, E>,
    bond_by_type: Param<R, E>,
    e_residual: Param<R, E>,
    e_ptr_type: Param<R, E>,
    e_ptr_bond: Param<R, E>,
    d_model: usize,
    n_heads: usize,
    max_atoms: usize,
}

/// Output of [`Ms2Decoder::teacher`].
pub struct TeacherOutput<R: Runtime, E: FloatElem> {
    /// `[B*G]` per-target negative log-likelihood.
    pub nll: Var<R, E>,
    /// `[B*G, T, 4]` gathered per-field log-probabilities (0 for unused
    /// fields and unscored positions).
    pub field_log_prob: Var<R, E>,
}

/// Per-field log-softmax rows of the teacher pass (inspection support for the
/// normalisation test): `[B*G, T, W]` per field.
pub struct FieldDistributions<R: Runtime, E: FloatElem> {
    /// `[B*G, T, 5]` kind log-probabilities.
    pub kind_log_prob: Var<R, E>,
    /// `[B*G, T, 18]` atom-type log-probabilities.
    pub type_log_prob: Var<R, E>,
    /// `[B*G, T, 4]` bond log-probabilities.
    pub bond_log_prob: Var<R, E>,
    /// `[B*G, T, A]` pointer log-probabilities.
    pub pointer_log_prob: Var<R, E>,
}

/// Unmasked single-position head outputs of architecture §3.6, as returned by
/// [`Ms2Decoder::step_logits`]: the caller adds `bond_by_type` and applies
/// the legality masks.
pub struct StepHeads<R: Runtime, E: FloatElem> {
    /// `[rows, 5]` kind logits.
    pub kind: Var<R, E>,
    /// `[rows, 18]` atom-type logits.
    pub atom_type: Var<R, E>,
    /// `[rows, 4]` bond logits before `bond_by_type`.
    pub bond_base: Var<R, E>,
    /// `[rows, A]` pointer scores from the query projection alone.
    pub pointer_base: Var<R, E>,
    /// `[rows, 19, A]` pointer scores from the type conditioning alone.
    pub pointer_by_type: Var<R, E>,
    /// `[rows, 4, A]` pointer scores from the bond conditioning alone.
    pub pointer_by_bond: Var<R, E>,
}

/// Recurrent state for [`Ms2Decoder::step_logits`]: the mixer caches plus the
/// atom memory the pointer head reads. The grammar state itself (atom counts
/// and residual valences) lives in the trajectory `state` buffer the caller
/// passes to [`Ms2Decoder::step_logits`], so this state performs no device
/// read and needs a launch count independent of the rows.
pub struct DecoderState<R: Runtime, E: FloatElem> {
    /// Per-layer mixer caches, `[rows, ..]`. Empty in a state built by
    /// [`Ms2Decoder::start_state_unobserved`], whose recurrent state lives in
    /// the fused step's in-place buffers and is not readable between steps.
    pub caches: Vec<MixerCache<R, E>>,
    /// `[rows, A, d]` atom memory: the row of the atom added by a token.
    pub atom_memory: Tensor<R, E>,
    /// Previous decoder output per row: the atom added by the current token
    /// was predicted by the previous output, so that is what its memory row
    /// stores.
    pub prev_h: Tensor<R, E>,
    /// `[rows * A]` clamped residual ids (`min(residual, 7)`) the pointer
    /// head's residual lookup reads, refreshed by
    /// [`crate::tensor::ops::ms2::atom_memory_update`].
    pub resid_ids: IdTensor<R>,
    /// Per-layer cross-attention keys `[B, 1 + N, d]`, computed once per
    /// generation call by [`Ms2Decoder::start_state`].
    pub keys: Vec<Tensor<R, E>>,
    /// Per-layer cross-attention values `[B, 1 + N, d]`, computed once per
    /// generation call by [`Ms2Decoder::start_state`].
    pub values: Vec<Tensor<R, E>>,
    /// Constants and scratch of the fused step ([`Ms2Decoder::step_packed`]),
    /// present only in a state built by [`Ms2Decoder::start_state_fused`].
    /// Such a state is driven by `step_packed` alone: its `atom_memory` is
    /// empty (the fused step keeps the projected rows in
    /// [`FusedStep::atom_keys`] instead), so [`Ms2Decoder::step_logits`]
    /// refuses it with a shape error.
    pub fused: Option<FusedStep<R, E>>,
}

/// What [`Ms2Decoder::step_packed`] needs beyond the recurrent state: the
/// parameter tables concatenated once per generation call, so each stage of
/// the step binds one table, and the buffers the step reuses.
pub struct FusedStep<R: Runtime, E: FloatElem> {
    /// `[27 + A + 64, d]`: the kind, atom-type, bond, pointer and step
    /// embedding tables, row-concatenated in that order.
    embed: Tensor<R, E>,
    /// `[d, W]` ([`step_head_layout`]): the kind, atom-type and bond head
    /// weights; the pointer query projection and its product with the
    /// residual rows; the atom-memory projection and its product with the
    /// conditioning rows; zero columns up to a multiple of 4.
    head_w: Tensor<R, E>,
    /// `[27]`: the kind, atom-type and bond head biases (no other column of
    /// the head row has one).
    head_b: Tensor<R, E>,
    /// `[23, 8]`: every conditioning row of the pointer head (`E_ptr_type`
    /// (19), then `E_ptr_bond` (4)) against every `E_residual` row.
    ptr_cross: Tensor<R, E>,
    /// `[rows, A, d + 23]` projected atom memory: row `j` is
    /// `mem_proj(atom_memory[j])` followed by its product with each of the
    /// 23 conditioning rows, written when the atom is added.
    pub atom_keys: Tensor<R, E>,
    /// `[rows, W]` head row of the previous step; its key segment is the
    /// atom-memory projection of the previous output.
    prev_heads: Tensor<R, E>,
    /// `[rows, heads, M]` attention weights, overwritten by every layer.
    attn_w: Tensor<R, E>,
    /// Per-layer recurrent state of a loop whose carries nobody reads
    /// ([`Ms2Decoder::start_state_unobserved`]): the mixers step it where it
    /// lies ([`Mamba3Block::step_in_place`]) and `DecoderState::caches` is
    /// empty. `None` keeps the caches, which a caller may read, freeze or
    /// snapshot after every step.
    carries: Option<Vec<MixerStepBuffers<R, E>>>,
}

impl<R: Runtime, E: FloatElem> FusedStep<R, E> {
    /// Clones of the live in-place carry tensors per layer, when this step
    /// holds them; `None` otherwise. Test support; see
    /// [`DecoderState::in_place_carries`].
    pub fn in_place_carries(
        &self,
    ) -> Option<Vec<crate::tensor::ops::mixer_step::InPlaceCarryTensors<R, E>>> {
        self.carries
            .as_ref()
            .map(|cs| cs.iter().map(|c| c.carry_tensors()).collect())
    }

    /// The previous step's head row (`[rows, W]`); its key segment is the
    /// atom-memory projection the next step may store. Test support for the
    /// in-place/functional comparison.
    pub fn prev_heads(&self) -> &Tensor<R, E> {
        &self.prev_heads
    }
}

impl<R: Runtime, E: FloatElem> DecoderState<R, E> {
    /// Whether the recurrent state is stepped in place
    /// ([`Ms2Decoder::start_state_unobserved`]): there are then no per-step
    /// caches to read or to freeze.
    pub fn carries_in_place(&self) -> bool {
        self.fused.as_ref().is_some_and(|f| f.carries.is_some())
    }

    /// Clones of the live in-place carry tensors per layer, when this state
    /// steps its carries in place ([`Ms2Decoder::start_state_unobserved`]);
    /// `None` otherwise. Test support for the in-place/functional carry
    /// comparison (`tests/ms2_fused_step.rs`).
    pub fn in_place_carries(
        &self,
    ) -> Option<Vec<crate::tensor::ops::mixer_step::InPlaceCarryTensors<R, E>>> {
        self.fused.as_ref()?.in_place_carries()
    }
}

/// Full teacher internals: the loss quantities plus the per-field
/// distributions, computed in one pass.
struct TeacherRun<R: Runtime, E: FloatElem> {
    nll: Var<R, E>,
    fields: Var<R, E>,
    dkind: Var<R, E>,
    dtype: Var<R, E>,
    dbond: Var<R, E>,
    dptr: Var<R, E>,
}

/// One `[rows, d]` embedding table with the decoder's standard init.
fn decoder_table<R: Runtime, E: FloatElem>(
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

impl<R: Runtime, E: FloatElem> Ms2Decoder<R, E> {
    /// Build the decoder for `model` on `device`.
    pub fn init(model: &ModelConfig, device: &Device<R>, rng: &mut Rng) -> Result<Self> {
        let d = model.d_model as usize;
        let heads = model.attention_heads as usize;
        let a = model.max_atoms as usize;
        let layers_n = model.decoder_blocks as usize;
        if d == 0 {
            return Err(Error::config("Ms2Decoder::init: d_model is 0".to_string()));
        }
        if heads == 0 {
            return Err(Error::config(
                "Ms2Decoder::init: attention_heads is 0".to_string(),
            ));
        }
        if !d.is_multiple_of(heads) {
            return Err(Error::config(format!(
                "Ms2Decoder::init: d_model {d} is not a multiple of attention_heads {heads}"
            )));
        }
        if !(1..=32).contains(&a) {
            return Err(Error::config(format!(
                "Ms2Decoder::init: max_atoms {a} is not in 1..=32"
            )));
        }
        let kind_emb = decoder_table(5, d, device, rng);
        let type_emb = decoder_table(18, d, device, rng);
        let bond_emb = decoder_table(4, d, device, rng);
        let ptr_emb = decoder_table(a, d, device, rng);
        let step_emb = decoder_table(STEP_ROWS, d, device, rng);
        let mut layers = Vec::with_capacity(layers_n);
        for _ in 0..layers_n {
            layers.push(DecoderLayer {
                mixer: Mamba3BlockConfig::new(model.decoder.clone()).init(device, rng)?,
                norm: RmsNormConfig::new(d).init(device, rng),
                q: LinearConfig::new(d, d).with_bias(false).init(device, rng),
                k: LinearConfig::new(d, d).with_bias(false).init(device, rng),
                v: LinearConfig::new(d, d).with_bias(false).init(device, rng),
                o: LinearConfig::new(d, d).with_bias(false).init(device, rng),
            });
        }
        let kind_head = LinearConfig::new(d, 5).init(device, rng);
        let type_head = LinearConfig::new(d, 18).init(device, rng);
        let bond_head = LinearConfig::new(d, 4).init(device, rng);
        let mem_proj = LinearConfig::new(d, d).with_bias(false).init(device, rng);
        let ptr_query = LinearConfig::new(d, d).with_bias(false).init(device, rng);
        let bond_by_type = decoder_table(COND_ROWS, 4, device, rng);
        let e_residual = decoder_table(8, d, device, rng);
        let e_ptr_type = decoder_table(COND_ROWS, d, device, rng);
        let e_ptr_bond = decoder_table(4, d, device, rng);
        Ok(Self {
            kind_emb,
            type_emb,
            bond_emb,
            ptr_emb,
            step_emb,
            layers,
            kind_head,
            type_head,
            bond_head,
            mem_proj,
            ptr_query,
            bond_by_type,
            e_residual,
            e_ptr_type,
            e_ptr_bond,
            d_model: d,
            n_heads: heads,
            max_atoms: a,
        })
    }

    /// Cross-attention of one layer over precomputed keys and values: queries
    /// `[b, qt, d]` attend over `k`/`v` (`[b, m, d]`), gated by `mask`
    /// (`[b, m]` 1/0). Masked scores take no weight. The teacher pass and the stepped pass share this helper, so
    /// both paths stay arithmetically identical; only who computed `k`/`v`
    /// differs (once per pass in [`Ms2Decoder::teacher`], once per generation
    /// call in [`Ms2Decoder::start_state`]).
    ///
    /// [`Ms2Decoder::teacher`]: Ms2Decoder::teacher
    /// [`Ms2Decoder::start_state`]: Ms2Decoder::start_state
    fn attend_cached(
        &self,
        layer: &DecoderLayer<R, E>,
        q_in: &Var<R, E>,
        k: &Var<R, E>,
        v: &Var<R, E>,
        mask: &Tensor<R, E>,
    ) -> Result<Var<R, E>> {
        let (kh, vh) = self.split_heads(k, v)?;
        self.attend_heads(layer, q_in, &kh, &vh, mask)
    }

    /// Keys and values `[b, m, d]` laid out per head for [`attend_heads`]:
    /// keys `[b, h, hd, m]`, values `[b, h, m, hd]`.
    ///
    /// [`attend_heads`]: Ms2Decoder::attend_heads
    fn split_heads(&self, k: &Var<R, E>, v: &Var<R, E>) -> Result<(Var<R, E>, Var<R, E>)> {
        let (b, m) = (k.dims()[0], k.dims()[1]);
        let h = self.n_heads;
        let hd = self.d_model / h;
        Ok((
            k.reshape(vec![b, m, h, hd])?.permute(&[0, 2, 3, 1])?,
            v.reshape(vec![b, m, h, hd])?.permute(&[0, 2, 1, 3])?,
        ))
    }

    /// [`attend_cached`] over keys and values already split per head
    /// ([`split_heads`]). The scale, the key mask and the softmax are one
    /// launch ([`Var::masked_softmax`]), reading the `[b, m]` mask as it is.
    ///
    /// [`attend_cached`]: Ms2Decoder::attend_cached
    /// [`split_heads`]: Ms2Decoder::split_heads
    fn attend_heads(
        &self,
        layer: &DecoderLayer<R, E>,
        q_in: &Var<R, E>,
        kh: &Var<R, E>,
        vh: &Var<R, E>,
        mask: &Tensor<R, E>,
    ) -> Result<Var<R, E>> {
        let b = q_in.dims()[0];
        let qt = q_in.dims()[1];
        let m = kh.dims()[3];
        let d = self.d_model;
        let h = self.n_heads;
        let hd = d / h;
        let q = layer.q.apply(&layer.norm.apply(q_in)?)?;
        let qh = q.reshape(vec![b, qt, h, hd])?.permute(&[0, 2, 1, 3])?;
        let weights = qh
            .matmul(kh)?
            .masked_softmax(&mask.reshape(Shape::new(vec![b, m]))?, 1.0 / (hd as f32).sqrt())?;
        let ctx = weights
            .matmul(vh)?
            .permute(&[0, 2, 1, 3])?
            .reshape(vec![b, qt, d])?;
        layer.o.apply(&ctx)
    }

    /// Static input embeddings `[rows, T, d]` for the teacher pass: token
    /// field lookups plus the position and formula embeddings. Unused token
    /// fields are 0 and look up row 0.
    fn embed_teacher(
        &self,
        tokens: &IdTensor<R>,
        formula: &Var<R, E>,
        rows: usize,
        t: usize,
        spectra: usize,
    ) -> Result<Var<R, E>> {
        let device = tokens.device();
        let col = |c: usize| -> Result<IdTensor<R>> {
            slice_ids_along(tokens, 2, c, 1)?.reshape(vec![rows * t])
        };
        let embed_col = |table: &Param<R, E>, ids: &IdTensor<R>| -> Result<Var<R, E>> {
            Var::ms2_lookup(&table.var_standalone(), ids)?.reshape(vec![rows, t, self.d_model])
        };
        let e_kind = embed_col(&self.kind_emb, &col(0)?)?;
        let e_type = embed_col(&self.type_emb, &col(1)?)?;
        let e_bond = embed_col(&self.bond_emb, &col(2)?)?;
        let e_ptr = embed_col(&self.ptr_emb, &col(3)?)?;
        let tokens_sum = e_kind.add(&e_type)?.add(&e_bond)?.add(&e_ptr)?;
        // Position `i` of every row reads step row `i`: the first `T` rows
        // of the table, broadcast over the rows by the sum. (A lookup with
        // ids `i` gives the same values, but its adjoint scans `rows * T`
        // ids for each of the table's elements.) A horizon beyond the
        // table keeps the lookup, whose out-of-range rows are zero.
        let with_step = if t <= STEP_ROWS {
            let e_step = self
                .step_emb
                .var_standalone()
                .slice(0, 0, t)?
                .reshape(vec![1, t, self.d_model])?;
            tokens_sum.add(&e_step)?
        } else {
            let mut step_ids = vec![0u32; rows * t];
            for r in 0..rows {
                for i in 0..t {
                    step_ids[r * t + i] = i as u32;
                }
            }
            let step_t = IdTensor::from_slice(&step_ids, vec![rows * t], device)?;
            tokens_sum.add(&embed_col(&self.step_emb, &step_t)?)?
        };
        // The formula embedding broadcasts over the spectrum's G targets
        // and the positions, again by the sum.
        let slots = rows / spectra;
        with_step
            .reshape(vec![spectra, slots, t, self.d_model])?
            .add(&formula.reshape(vec![spectra, 1, 1, self.d_model])?)?
            .reshape(vec![rows, t, self.d_model])
    }

    /// Decoder layers over `[rows, T, d]`: each `Mamba3Block`, then residual
    /// cross-attention. For cross-attention the queries are viewed as
    /// `[B, G*T, d]` (each query attends independently) and restored after.
    fn apply_layers(
        &self,
        x: &Var<R, E>,
        encoded: &EncoderOutput<R, E>,
        rows: usize,
        t: usize,
        spectra: usize,
        owner: Option<&IdTensor<R>>,
    ) -> Result<Var<R, E>> {
        let slots = rows / spectra;
        let mut x = x.clone();
        let mask = match owner {
            Some(owner) => crate::tensor::ops::ms2::lookup(&encoded.memory_mask, owner)?,
            None => encoded.memory_mask.clone(),
        };
        for layer in &self.layers {
            x = layer.mixer.apply(&x)?;
            let q = x.reshape(vec![spectra, slots * t, self.d_model])?;
            // Keys and values once per pass (the stepped path reuses the ones
            // `start_state` computed); the attention arithmetic itself is the
            // shared `attend_cached` helper. With virtual spectra they are
            // projected once per spectrum and gathered, not projected once
            // per virtual spectrum.
            // The head split is taken per spectrum too, before the gather:
            // the gather moves whole rows either way, and the split is the
            // copy that swaps the contiguous axis.
            let (mut k, mut v) = self.split_heads(
                &layer.k.apply(&encoded.memory)?,
                &layer.v.apply(&encoded.memory)?,
            )?;
            if let Some(owner) = owner {
                k = gather_spectra(&k, owner)?;
                v = gather_spectra(&v, owner)?;
            }
            let ctx = self.attend_heads(layer, &q, &k, &v, &mask)?;
            let back = ctx.reshape(vec![rows, t, self.d_model])?;
            x = x.add(&back)?;
        }
        Ok(x)
    }

    /// The full teacher pass behind [`teacher`] and [`field_distributions`].
    ///
    /// [`teacher`]: Ms2Decoder::teacher
    /// [`field_distributions`]: Ms2Decoder::field_distributions
    ///
    /// `owner` (`[B']`) turns the rows into `B'` *virtual spectra*: virtual
    /// spectrum `v` holds `rows / B'` target rows of spectrum `owner[v]`,
    /// whose memory, mask and formula embedding it takes
    /// ([`Ms2Decoder::teacher_grouped`]).
    fn run(
        &self,
        encoded: &EncoderOutput<R, E>,
        formula_embedding: &Var<R, E>,
        targets: &DeviceTargets<R, E>,
        replay: &ReplayView<'_, R>,
        layout: TeacherLayout<'_, R, E>,
    ) -> Result<TeacherRun<R, E>> {
        // Every input rank is checked before any dimension is read.
        if targets.tokens.shape().rank() != 3
            || targets.meta.shape().rank() != 2
            || targets.q.rank() != 1
            || targets.use_mask.rank() != 3
            || formula_embedding.rank() != 2
            || encoded.memory.rank() != 3
            || encoded.memory_mask.rank() != 2
        {
            return Err(Error::shape(format!(
                "Ms2Decoder::teacher needs tokens [rows, T, 4], meta [rows, 12], q [rows], use [rows, T, 4], formula [B, d], memory [B, 1+N, d] and mask [B, 1+N], got {} and {} and {} and {} and {} and {} and {}",
                targets.tokens.shape(),
                targets.meta.shape(),
                targets.q.shape(),
                targets.use_mask.shape(),
                formula_embedding.shape(),
                encoded.memory.shape(),
                encoded.memory_mask.shape()
            )));
        }
        let rows = targets.tokens.shape().dim(0);
        let t = targets.tokens.shape().dim(1);
        let stored = encoded.memory.shape().dim(0);
        let owner = match layout {
            TeacherLayout::Grouped(owner) => Some(owner),
            _ => None,
        };
        if owner.is_some_and(|o| o.shape().rank() != 1) {
            return Err(Error::shape(format!(
                "Ms2Decoder::teacher needs the owner of each virtual spectrum as [B'], got {}",
                owner.map(|o| o.shape().clone()).unwrap_or_default()
            )));
        }
        // Packed: the rows are traces, in no spectrum grouping the heads need.
        let spectra = match layout {
            TeacherLayout::Packed(_) => rows.max(1),
            _ => owner.map_or(stored, |o| o.len()),
        };
        let a = self.max_atoms;
        if spectra == 0 || !rows.is_multiple_of(spectra) {
            return Err(Error::shape(format!(
                "Ms2Decoder::teacher needs rows B*G with B = {}, got rows {rows}",
                spectra
            )));
        }
        let want_tokens: &[usize] = &[rows, t, 4];
        let want_meta: &[usize] = &[rows, 12];
        let want_use: &[usize] = &[rows, t, 4];
        let want_formula: &[usize] = &[stored, self.d_model];
        if targets.tokens.shape().dims() != want_tokens
            || targets.meta.shape().dims() != want_meta
            || targets.q.len() != rows
            || targets.use_mask.shape().dims() != want_use
            || formula_embedding.dims() != want_formula
            || replay.widths() != (t, a)
            || rows != replay.rows()
        {
            return Err(Error::shape(format!(
                "Ms2Decoder::teacher has mismatched batch shapes: tokens {}, meta {}, q {}, use {}, formula {}, replay T/A {:?}, replay rows {}",
                targets.tokens.shape(),
                targets.meta.shape(),
                targets.q.shape(),
                targets.use_mask.shape(),
                formula_embedding.shape(),
                replay.widths(),
                replay.rows()
            )));
        }
        let device = targets.tokens.device();
        // Atom memory `[rows, A, d]`, gathered after the parallel pass with
        // `gather_tokens(h, atom_step - 1)`; unused slots keep `u32::MAX` (the
        // gather's zero row).
        let h = match layout {
            TeacherLayout::Packed(packing) => {
                self.packed_hidden(encoded, formula_embedding, packing, rows, t)?
            }
            _ => {
                let gathered;
                let formula_embedding = match owner {
                    Some(owner) => {
                        gathered = Var::ms2_lookup(formula_embedding, owner)?;
                        &gathered
                    }
                    None => formula_embedding,
                };
                let x = self.embed_teacher(&targets.tokens, formula_embedding, rows, t, spectra)?;
                self.apply_layers(&x, encoded, rows, t, spectra, owner)?
            }
        };
        let atom_cols = slice_ids_along(replay.atoms, 1, 0, a)?.reshape(vec![rows * a])?;
        let pred = pred_ids(&atom_cols)?;
        let atom_mem = Var::gather_tokens(&h, &pred, a)?;
        let d = self.d_model;
        let ptr_scale = 1.0 / (d as f32).sqrt();
        // "Index 0 only" rows, one per field width.
        let index0 = |width: usize| -> Result<Tensor<R, E>> {
            let mut row = vec![0.0f32; width];
            row[0] = 1.0;
            Tensor::from_f32(&row, vec![1, width], device)
        };
        let index0_kind = index0(5)?;
        let index0_type = index0(18)?;
        let index0_bond = index0(4)?;
        let index0_ptr = index0(a)?;
        let mut nll = Var::constant(Tensor::zeros(vec![rows], device));
        let mut field_rows: Vec<Var<R, E>> = Vec::with_capacity(2);
        let mut kind_rows: Vec<Var<R, E>> = Vec::with_capacity(2);
        let mut type_rows: Vec<Var<R, E>> = Vec::with_capacity(2);
        let mut bond_rows: Vec<Var<R, E>> = Vec::with_capacity(2);
        let mut ptr_rows: Vec<Var<R, E>> = Vec::with_capacity(2);
        let positions = t.saturating_sub(1);
        let n = rows * positions;
        if n > 0 {
            // Every output position `i = 0..T-1` is scored in one pass over
            // `n = rows * (T - 1)` flattened rows (row `r * (T - 1) + i`),
            // not one position at a time: the heads are position-wise, so
            // this changes no value, and the launch count no longer grows
            // with `T`. The masks for output position `i` are
            // `replay[.., i + 1, 0..4]` (the state before token `i + 1`); the
            // target ids come from the in-range conditioning plan.
            let plan = teacher_plan(&targets.tokens, &targets.meta, replay.replay, a)?;
            let plan_col = |c: usize| -> Result<IdTensor<R>> {
                slice_ids_along(&plan, 1, c, 1)?.reshape(vec![n])
            };
            let kind_id = plan_col(0)?;
            let type_id = plan_col(1)?;
            let bond_id = plan_col(2)?;
            let ptr_id = plan_col(3)?;
            let cond_id = plan_col(4)?;
            let resid = slice_ids_along(&plan, 1, TEACHER_PLAN_HEAD, a)?;
            let eff = |field: usize, width: usize| -> Result<Tensor<R, E>> {
                effective_mask(&plan, &targets.use_mask, field, width)
            };
            let h_p = h.slice(1, 0, positions)?.reshape(vec![n, d])?;
            // Kind, atom type: plain head logits.
            let kind_lp = self
                .kind_head
                .apply(&h_p)?
                .mask_logits(&eff(0, 5)?)?
                .log_softmax(1)?;
            let kind_got = kind_lp.take_along_last(&kind_id)?;
            let type_lp = self
                .type_head
                .apply(&h_p)?
                .mask_logits(&eff(1, 18)?)?
                .log_softmax(1)?;
            let type_got = type_lp.take_along_last(&type_id)?;
            // Bond: the learned row `bond_by_type[c]` joins before masking.
            let bond_base = self.bond_head.apply(&h_p)?;
            let bond_corr = Var::ms2_lookup(&self.bond_by_type.var_standalone(), &cond_id)?;
            let bond_lp = bond_base
                .add(&bond_corr)?
                .mask_logits(&eff(2, 4)?)?
                .log_softmax(1)?;
            let bond_got = bond_lp.take_along_last(&bond_id)?;
            // Pointer: `(Linear(h) + E_ptr_type[c] + E_ptr_bond[b]) · k_j /
            // sqrt(d)` with `k_j = Linear(memory_j) +
            // E_residual[min(residual_j, 7)]`, added before masking. The
            // product is taken term by term, `q · Linear(memory_j) +
            // q · E_residual[..]`, so no `[n, A, d]` key tensor exists: the
            // atom memory does not depend on the position and is projected
            // once (`[rows, A, d]`), and the residual term is the query
            // against the 8 residual rows (`[n, 8]`) gathered per atom. Per
            // row that replaces `A * d` key values by `A + 8` scores, in the
            // forward pass and in every adjoint (the residual table's
            // gradient is one product, not a scan of `n * A` ids).
            let q0 = self.ptr_query.apply(&h_p)?;
            let qt = Var::ms2_lookup(&self.e_ptr_type.var_standalone(), &cond_id)?;
            let qb = Var::ms2_lookup(&self.e_ptr_bond.var_standalone(), &bond_id)?;
            let query = q0.add(&qt)?.add(&qb)?;
            let keys0 = self.mem_proj.apply(&atom_mem)?;
            let by_memory = query
                .reshape(vec![rows, positions, d])?
                .matmul_nt(&keys0)?
                .reshape(vec![n, a])?;
            let residual_rows = self.e_residual.var_standalone();
            let residual_n = residual_rows.dims()[0];
            let by_residual = query
                .matmul_nt(&residual_rows)?
                .reshape(vec![n, 1, residual_n])?
                .expand(vec![n, a, residual_n])?
                .take_along_last(&resid)?;
            let ptr_scores = by_memory.add(&by_residual)?.mul_scalar(ptr_scale);
            let ptr_lp = ptr_scores.mask_logits(&eff(3, a)?)?.log_softmax(1)?;
            let ptr_got = ptr_lp.take_along_last(&ptr_id)?;
            // A field the target does not use, or an unscored position,
            // contributes exactly 0: its id is already 0 (the "index 0 only"
            // value, exactly 0 after log-softmax) and the product with
            // `use` zeroes any residual.
            let widen = |got: &Var<R, E>| got.reshape(vec![n, 1]);
            let got = cat(
                &[
                    widen(&kind_got)?,
                    widen(&type_got)?,
                    widen(&bond_got)?,
                    widen(&ptr_got)?,
                ],
                1,
            )?;
            let use_p = Var::constant(
                movement::slice(&targets.use_mask, 1, 0, positions)?.reshape(vec![n, 4])?,
            );
            // `0 - sum`, not a negation: an empty target's sum is exactly 0
            // and its nll must be `+0.0`, as the per-position subtraction
            // from a zero start gave.
            nll = nll.sub(
                &got.mul(&use_p)?
                    .reshape(vec![rows, positions * 4])?
                    .sum_dim(1)?
                    .reshape(vec![rows])?,
            )?;
            field_rows.push(got.reshape(vec![rows, positions, 4])?);
            kind_rows.push(kind_lp.reshape(vec![rows, positions, 5])?);
            type_rows.push(type_lp.reshape(vec![rows, positions, 18])?);
            bond_rows.push(bond_lp.reshape(vec![rows, positions, 4])?);
            ptr_rows.push(ptr_lp.reshape(vec![rows, positions, a])?);
        }
        // Position `T - 1` never reads a token or replay row `T`: its ids are
        // 0, its masks are "index 0 only" and its `use` is 0, so every
        // gathered field value is exactly 0 while every distribution row is
        // the index-0-only distribution (log-probability 0 at index 0, the
        // masked value elsewhere), so every row is a distribution.
        let zero_field = Var::constant(Tensor::zeros(vec![rows, 1, 4], device));
        field_rows.push(zero_field);
        let index_dist = |w: usize, index0: &Tensor<R, E>| -> Result<Var<R, E>> {
            let logits = Var::constant(Tensor::zeros(vec![rows, w], device));
            let mask = elemwise::expand(
                &index0.reshape(Shape::new(vec![1, w]))?,
                &Shape::new(vec![rows, w]),
            )?;
            logits
                .mask_logits(&mask)?
                .log_softmax(1)?
                .reshape(vec![rows, 1, w])
        };
        kind_rows.push(index_dist(5, &index0_kind)?);
        type_rows.push(index_dist(18, &index0_type)?);
        bond_rows.push(index_dist(4, &index0_bond)?);
        ptr_rows.push(index_dist(a, &index0_ptr)?);
        Ok(TeacherRun {
            nll,
            fields: cat(&field_rows, 1)?,
            dkind: cat(&kind_rows, 1)?,
            dtype: cat(&type_rows, 1)?,
            dbond: cat(&bond_rows, 1)?,
            dptr: cat(&ptr_rows, 1)?,
        })
    }

    /// Teacher forcing over positions `i = 0..T-1`: input `i` is built from
    /// token `i` and predicts token `i + 1`. `nll[target]` is minus the
    /// `use`-weighted sum of the gathered per-field log-probabilities.
    /// Neither this nor [`graph_loss`] reads the device.
    ///
    /// [`graph_loss`]: graph_loss
    pub fn teacher(
        &self,
        encoded: &EncoderOutput<R, E>,
        formula_embedding: &Var<R, E>,
        targets: &DeviceTargets<R, E>,
        replay: &ReplayView<'_, R>,
    ) -> Result<TeacherOutput<R, E>> {
        let run = self.run(encoded, formula_embedding, targets, replay, TeacherLayout::Padded)?;
        Ok(TeacherOutput {
            nll: run.nll,
            field_log_prob: run.fields,
        })
    }

    /// [`Ms2Decoder::teacher`] over *virtual spectra*: `targets` holds
    /// `B' * G'` rows, and virtual spectrum `v` — rows `v * G'..(v + 1) * G'`
    /// — belongs to spectrum `owner[v]` of `encoded` and `formula_embedding`
    /// (both still `B` spectra). A row's result is the one
    /// [`Ms2Decoder::teacher`] gives it in any batch that pairs it with the
    /// same spectrum, so a batch that leaves out the empty target slots
    /// ([`super::targets_batch::TargetBatch::compact`]) scores the occupied
    /// ones identically at a fraction of the rows. Keys and values are
    /// projected once per spectrum and gathered per virtual spectrum.
    pub fn teacher_grouped(
        &self,
        encoded: &EncoderOutput<R, E>,
        formula_embedding: &Var<R, E>,
        targets: &DeviceTargets<R, E>,
        replay: &ReplayView<'_, R>,
        owner: &IdTensor<R>,
    ) -> Result<TeacherOutput<R, E>> {
        let run = self.run(
            encoded,
            formula_embedding,
            targets,
            replay,
            TeacherLayout::Grouped(owner),
        )?;
        Ok(TeacherOutput {
            nll: run.nll,
            field_log_prob: run.fields,
        })
    }

    /// [`Ms2Decoder::teacher`] over ragged sequences: `targets` holds one
    /// occupied trace per row, and `packing`
    /// ([`super::targets_batch::TargetBatch::pack`]) lays those traces end
    /// to end in rows of its own length. The layers — where nearly all of
    /// the pass's arithmetic is — run over the packed rows, with the state
    /// reset where each trace begins and each row attending its spectrum's
    /// memory; their output is taken back to one trace per row for the
    /// heads. A trace's positions see the tokens before them in the same
    /// trace and nothing else, as they do in a row of their own, so the
    /// result is [`Ms2Decoder::teacher`]'s for the same traces while the
    /// positions past a trace's end are never computed.
    pub fn teacher_packed(
        &self,
        encoded: &EncoderOutput<R, E>,
        formula_embedding: &Var<R, E>,
        targets: &DeviceTargets<R, E>,
        replay: &ReplayView<'_, R>,
        packing: &DevicePacking<R, E>,
    ) -> Result<TeacherOutput<R, E>> {
        let run = self.run(
            encoded,
            formula_embedding,
            targets,
            replay,
            TeacherLayout::Packed(packing),
        )?;
        Ok(TeacherOutput {
            nll: run.nll,
            field_log_prob: run.fields,
        })
    }

    /// The decoder output `[rows, T, d]` of a packed pass: embed and run the
    /// layers over the packed rows, then take every trace position from its
    /// packed cell (zero past a trace's end).
    ///
    /// Each layer works in two layouts of the same cells. Its block scans
    /// rows that mix spectra, with the state reset where a trace begins; its
    /// cross-attention regroups the cells into rows of one spectrum each, so
    /// a row's queries share that spectrum's keys and values, and the context
    /// is taken back to the scan layout for the residual.
    fn packed_hidden(
        &self,
        encoded: &EncoderOutput<R, E>,
        formula: &Var<R, E>,
        packing: &DevicePacking<R, E>,
        rows: usize,
        t: usize,
    ) -> Result<Var<R, E>> {
        let d = self.d_model;
        if packing.tokens.shape().rank() != 3 {
            return Err(Error::shape(format!(
                "Ms2Decoder::teacher_packed needs packed tokens [P, L, 4], got {}",
                packing.tokens.shape()
            )));
        }
        let scan = &packing.scan;
        let (packed, len) = (scan.rows, scan.row_len);
        let cells = packed * len;
        let attn_rows = packing.attn_owner.len();
        let attn_cells = attn_rows * packing.attn_len;
        if packing.tokens.shape().dims() != [packed, len, 4]
            || scan.reset.dims() != [packed, len]
            || packing.steps.len() != cells
            || packing.cell_owner.len() != cells
            || packing.to_scan.len() != cells
            || packing.to_attn.len() != attn_cells
            || scan.pack.len() != cells
            || scan.unpack.len() != rows * t
        {
            return Err(Error::shape(format!(
                "Ms2Decoder::teacher_packed has a packing that does not fit {rows} traces of {t}: tokens {}, reset {}, steps {}, owners {}, to_scan {}, to_attn {} for {attn_rows} attention rows of {}, pack {}, unpack {}",
                packing.tokens.shape(),
                scan.reset.shape(),
                packing.steps.shape(),
                packing.cell_owner.shape(),
                packing.to_scan.shape(),
                packing.to_attn.shape(),
                packing.attn_len,
                scan.pack.shape(),
                scan.unpack.shape()
            )));
        }
        let col = |c: usize| -> Result<IdTensor<R>> {
            slice_ids_along(&packing.tokens, 2, c, 1)?.reshape(vec![cells])
        };
        let embed = |table: &Param<R, E>, ids: &IdTensor<R>| -> Result<Var<R, E>> {
            Var::ms2_lookup(&table.var_standalone(), ids)
        };
        // A cell's step row is its position inside its own trace, and its
        // formula embedding its own spectrum's.
        let mut x = embed(&self.kind_emb, &col(0)?)?
            .add(&embed(&self.type_emb, &col(1)?)?)?
            .add(&embed(&self.bond_emb, &col(2)?)?)?
            .add(&embed(&self.ptr_emb, &col(3)?)?)?
            .add(&embed(&self.step_emb, &packing.steps)?)?
            .add(&Var::ms2_lookup(formula, &packing.cell_owner)?)?
            .reshape(vec![packed, len, d])?;
        // The memory is one metadata slot and `N` peaks, and `1 + N` is not a
        // whole number of vectors: the attention products over it — batched,
        // small, and with the memory as their `n` or `k` — could then only
        // run on the unvectorised plans. A few masked slots of zeros round it
        // up; a masked slot takes no attention weight, so nothing else moves.
        let slots = encoded.memory.shape().dim(1);
        let spectra = encoded.memory.shape().dim(0);
        let pad = slots.next_multiple_of(4) - slots;
        let (memory, memory_mask) = if pad > 0 && spectra > 0 {
            let device = encoded.memory_mask.device();
            (
                cat(
                    &[
                        encoded.memory.clone(),
                        Var::constant(Tensor::zeros(vec![spectra, pad, d], device)),
                    ],
                    1,
                )?,
                movement::cat(
                    &[
                        encoded.memory_mask.clone(),
                        Tensor::zeros(vec![spectra, pad], device),
                    ],
                    1,
                )?,
            )
        } else {
            (encoded.memory.clone(), encoded.memory_mask.clone())
        };
        let mask = crate::tensor::ops::ms2::lookup(&memory_mask, &packing.attn_owner)?;
        for layer in &self.layers {
            x = layer
                .mixer
                .apply_with_state_masked(&x, None, Some(&scan.reset))?
                .0;
            let q = Var::ms2_take_rows(&x.reshape(vec![cells, d])?, &packing.to_attn, &packing.to_scan)?
                .reshape(vec![attn_rows, packing.attn_len, d])?;
            // Keys and values: projected and split per head once per
            // spectrum, gathered per attention row.
            let (k, v) =
                self.split_heads(&layer.k.apply(&memory)?, &layer.v.apply(&memory)?)?;
            let k = gather_spectra(&k, &packing.attn_owner)?;
            let v = gather_spectra(&v, &packing.attn_owner)?;
            let ctx = self.attend_heads(layer, &q, &k, &v, &mask)?;
            let back = Var::ms2_take_rows(
                &ctx.reshape(vec![attn_cells, d])?,
                &packing.to_scan,
                &packing.to_attn,
            )?
            .reshape(vec![packed, len, d])?;
            x = x.add(&back)?;
        }
        scan.unpack(&x, &[rows, t, d])
    }

    /// Per-field log-softmax rows of the teacher pass, for the normalisation
    /// test. Inspection support: same values [`teacher`] scores. Every row
    /// is a distribution: unscored positions and position `T - 1` carry the
    /// index-0-only distribution (log-probability 0 at index 0, the masked
    /// value elsewhere).
    ///
    /// [`teacher`]: Ms2Decoder::teacher
    pub fn field_distributions(
        &self,
        encoded: &EncoderOutput<R, E>,
        formula_embedding: &Var<R, E>,
        targets: &DeviceTargets<R, E>,
        replay: &ReplayView<'_, R>,
    ) -> Result<FieldDistributions<R, E>> {
        let run = self.run(encoded, formula_embedding, targets, replay, TeacherLayout::Padded)?;
        Ok(FieldDistributions {
            kind_log_prob: run.dkind,
            type_log_prob: run.dtype,
            bond_log_prob: run.dbond,
            pointer_log_prob: run.dptr,
        })
    }

    /// Decoder output `h` (`[B*G, T, d]`) without the heads. Inspection
    /// support for the causality test.
    pub fn hidden(
        &self,
        encoded: &EncoderOutput<R, E>,
        formula_embedding: &Var<R, E>,
        targets: &DeviceTargets<R, E>,
    ) -> Result<Var<R, E>> {
        if targets.tokens.shape().rank() != 3
            || formula_embedding.rank() != 2
            || encoded.memory.rank() != 3
        {
            return Err(Error::shape(format!(
                "Ms2Decoder::hidden needs tokens [rows, T, 4], formula [B, d] and memory [B, 1+N, d], got {} and {} and {}",
                targets.tokens.shape(),
                formula_embedding.shape(),
                encoded.memory.shape()
            )));
        }
        let rows = targets.tokens.shape().dim(0);
        let t = targets.tokens.shape().dim(1);
        let spectra = encoded.memory.shape().dim(0);
        if spectra == 0 || !rows.is_multiple_of(spectra) {
            return Err(Error::shape(format!(
                "Ms2Decoder::hidden needs rows B*G with B = {spectra}, got rows {rows}"
            )));
        }
        let x = self.embed_teacher(&targets.tokens, formula_embedding, rows, t, spectra)?;
        self.apply_layers(&x, encoded, rows, t, spectra, None)
    }

    /// Empty recurrent state for `rows` trajectories over `encoded`: fresh
    /// mixer caches, zeroed atom memory, previous outputs and residual ids,
    /// plus the per-layer cross-attention keys and values (`[B, 1 + N, d]`)
    /// computed once here so [`Ms2Decoder::step_logits`] never recomputes
    /// them. No device read.
    ///
    /// [`Ms2Decoder::step_logits`]: Ms2Decoder::step_logits
    pub fn start_state(
        &self,
        encoded: &EncoderOutput<R, E>,
        rows: usize,
        device: &Device<R>,
    ) -> Result<DecoderState<R, E>> {
        self.start_state_inner(encoded, rows, device, false, false)
    }

    /// [`Ms2Decoder::start_state`] for the fused step
    /// ([`Ms2Decoder::step_packed`]): the same recurrent state plus the
    /// per-call tables of [`FusedStep`], built with a handful of launches
    /// here so the step itself binds one table per stage. When a head or a
    /// projection carries a LoRA adapter, a quantizer or an unexpected bias,
    /// the fused step does not apply and the composed state is returned
    /// (`fused` is `None`). No device read.
    pub fn start_state_fused(
        &self,
        encoded: &EncoderOutput<R, E>,
        rows: usize,
        device: &Device<R>,
    ) -> Result<DecoderState<R, E>> {
        self.start_state_inner(encoded, rows, device, self.fusable(), false)
    }

    /// [`Ms2Decoder::start_state_fused`] for a loop whose recurrent state
    /// nobody reads between steps or afterwards — every caller that only
    /// wants the trajectories.
    ///
    /// The mixers then step their state in place
    /// ([`Mamba3Block::step_in_place`]): no state-sized tensor is allocated
    /// or copied per step, and `caches` is empty. A stopped row's state is
    /// not held still either — nothing reads it: the sampler ignores the
    /// row's logits from then on, and no kernel of the step mixes rows. A
    /// caller that reads the carries (the carry trace of the parity tests)
    /// uses `start_state_fused` and freezes them. Falls back to
    /// `start_state_fused` when the fused step or the in-place mixer step
    /// does not apply. No device read.
    pub fn start_state_unobserved(
        &self,
        encoded: &EncoderOutput<R, E>,
        rows: usize,
        device: &Device<R>,
    ) -> Result<DecoderState<R, E>> {
        // The unobserved loop is the composed-off, capture-off case of the
        // shared predicate, so support changes land in one place.
        let in_place = self.steps_carries_in_place(device, false, false);
        self.start_state_latched(encoded, rows, device, in_place)
    }

    /// [`Ms2Decoder::start_state_unobserved`] with the in-place decision
    /// latched by the caller (task F10 item A3): the generator passes its
    /// per-call [`GeneratePreflight::decode_in_place`](super::generate::GeneratePreflight::decode_in_place)
    /// value instead of re-evaluating the predicate, so a toggle flipped
    /// after preflight cannot change the state under construction. The
    /// staged [`Ms2Decoder::start_state_unobserved`] entry point evaluates
    /// the predicate fresh, for callers with no call to latch.
    pub(crate) fn start_state_latched(
        &self,
        encoded: &EncoderOutput<R, E>,
        rows: usize,
        device: &Device<R>,
        in_place: bool,
    ) -> Result<DecoderState<R, E>> {
        self.start_state_inner(encoded, rows, device, self.fusable(), in_place)
    }

    /// Whether the decode loop steps this decoder's carries in place on
    /// `device` under these execution flags: the single predicate behind
    /// both the state the generator builds and the carry numbers the memory
    /// estimate charges, so the two cannot drift.
    ///
    /// The generator evaluates this ONCE per call, at preflight, and latches
    /// the value into [`GeneratePreflight::decode_in_place`](super::generate::GeneratePreflight::decode_in_place)
    /// (task F10 item A3): decoder init and every step use the latched
    /// value, so a setter invoked from a hook mid-call changes the NEXT call
    /// only.
    ///
    /// True exactly when the generator will actually step in place: the
    /// composed reference step is off, nobody reads the carries (no carry
    /// capture), and the in-place fused step covers every layer on this
    /// backend (`fusable` tables plus
    /// [`Mamba3Block::step_in_place_supported`], which itself includes the
    /// `MAMBA3_FUSED_STEP` toggle and the backend's binding limit). Any
    /// other combination keeps per-step caches: the old bank stays live
    /// while the new bank is constructed, and the freeze runs in place over
    /// the new bank (see `decode_functional_step` in
    /// [`Ms2MemoryEstimate::generation_for_decode_mode`]).
    ///
    /// [`Ms2MemoryEstimate::generation_for_decode_mode`]: super::workspace::Ms2MemoryEstimate::generation_for_decode_mode
    /// [`Mamba3Block::step_in_place_supported`]: crate::models::mamba3::Mamba3Block::step_in_place_supported
    pub fn steps_carries_in_place(
        &self,
        device: &Device<R>,
        composed_step: bool,
        capture_carry_trace: bool,
    ) -> bool {
        !composed_step
            && !capture_carry_trace
            && self.fusable()
            && self
                .layers
                .iter()
                .all(|l| l.mixer.step_in_place_supported(device))
    }

    /// Whether the fused step computes exactly what the composed step does:
    /// every head and projection is a plain `Linear`, and the two pointer
    /// projections have no bias (an untouched atom-memory row then projects
    /// to exactly zero, which is what the fused state starts from).
    fn fusable(&self) -> bool {
        let plain = plain_linear::<R, E>;
        plain(&self.kind_head)
            && plain(&self.type_head)
            && plain(&self.bond_head)
            && plain(&self.mem_proj)
            && plain(&self.ptr_query)
            && self.mem_proj.bias().is_none()
            && self.ptr_query.bias().is_none()
    }

    /// The per-call tables and buffers of the fused step.
    fn fused_step(
        &self,
        encoded: &EncoderOutput<R, E>,
        rows: usize,
        device: &Device<R>,
    ) -> Result<FusedStep<R, E>> {
        let a = self.max_atoms;
        let d = self.d_model;
        let embed = movement::cat(
            &[
                self.kind_emb.value(),
                self.type_emb.value(),
                self.bond_emb.value(),
                self.ptr_emb.value(),
                self.step_emb.value(),
            ],
            0,
        )?;
        // The conditioning rows of the pointer head (19 atom types, then 4
        // bonds) and their products with the residual rows: a pointer score
        // is `query · (key + E_residual[r])`, and with a conditioning row as
        // the query both halves are known long before the step that reads
        // them — the first when the atom's key is stored, the second now.
        let e_cond = movement::cat(&[self.e_ptr_type.value(), self.e_ptr_bond.value()], 0)?;
        let e_residual = self.e_residual.value();
        let ptr_cross = matmul_nt(&e_cond, &e_residual)?;
        // The head row of one output, in the layout of `step_head_layout`:
        // the logits, the pointer query and its products with the residual
        // rows, then the atom-memory key and its products with the
        // conditioning rows, padded with zero columns to a whole number of
        // vectors.
        let layout = step_head_layout(d);
        let query_w = self.ptr_query.weight().value();
        let key_w = self.mem_proj.weight().value();
        let query_resid = matmul_nt(&query_w, &e_residual)?;
        let key_cond = matmul_nt(&key_w, &e_cond)?;
        let mut head_parts = vec![
            self.kind_head.weight().value(),
            self.type_head.weight().value(),
            self.bond_head.weight().value(),
            query_w,
            query_resid,
            key_w,
            key_cond,
        ];
        let pad = layout.width - layout.used;
        if pad > 0 {
            head_parts.push(Tensor::zeros(vec![d, pad], device));
        }
        let head_w = movement::cat(&head_parts, 1)?;
        let bias = |l: &Linear<R, E>| -> Tensor<R, E> {
            match l.bias() {
                Some(b) => b.value(),
                None => Tensor::zeros(vec![l.out_features()], device),
            }
        };
        let head_b = movement::cat(
            &[
                bias(&self.kind_head),
                bias(&self.type_head),
                bias(&self.bond_head),
            ],
            0,
        )?;
        let mem = encoded.memory.shape().dim(1);
        Ok(FusedStep {
            embed,
            head_w,
            head_b,
            ptr_cross,
            atom_keys: Tensor::zeros(vec![rows, a, layout.key_width], device),
            prev_heads: Tensor::zeros(vec![rows, layout.width], device),
            attn_w: Tensor::empty(vec![rows, self.n_heads, mem], device),
            carries: None,
        })
    }

    /// The state behind [`Ms2Decoder::start_state`] and
    /// [`Ms2Decoder::start_state_fused`].
    fn start_state_inner(
        &self,
        encoded: &EncoderOutput<R, E>,
        rows: usize,
        device: &Device<R>,
        fused: bool,
        in_place: bool,
    ) -> Result<DecoderState<R, E>> {
        if encoded.memory.rank() != 3 {
            return Err(Error::shape(format!(
                "Ms2Decoder::start_state needs memory [B, 1+N, d], got {}",
                encoded.memory.shape()
            )));
        }
        let mut keys = Vec::with_capacity(self.layers.len());
        let mut values = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            keys.push(layer.k.apply(&encoded.memory)?.tensor().clone());
            values.push(layer.v.apply(&encoded.memory)?.tensor().clone());
        }
        let a = self.max_atoms;
        let d = self.d_model;
        let resid = IdTensor::from_slice(&vec![0u32; rows * a], vec![rows * a], device)?;
        let mut fused_step = if fused {
            Some(self.fused_step(encoded, rows, device)?)
        } else {
            None
        };
        // Carries stepped in place replace the caches rather than join them.
        let mut caches = Vec::new();
        match fused_step.as_mut() {
            Some(step) if in_place => {
                step.carries = Some(
                    self.layers
                        .iter()
                        .map(|l| l.mixer.empty_step_buffers(rows, device))
                        .collect(),
                );
            }
            _ => {
                caches = self
                    .layers
                    .iter()
                    .map(|l| l.mixer.empty_cache(rows, device))
                    .collect();
            }
        }
        Ok(DecoderState {
            caches,
            // The fused step keeps projected rows in `FusedStep::atom_keys`;
            // its raw memory stays empty rather than a second zeroed bank.
            atom_memory: if fused {
                Tensor::empty(vec![0, a, d], device)
            } else {
                Tensor::zeros(vec![rows, a, d], device)
            },
            prev_h: Tensor::zeros(vec![rows, d], device),
            resid_ids: resid,
            keys,
            values,
            fused: fused_step,
        })
    }

    /// The `bond_by_type[c]` correction table (`[19, 4]`) the sampler adds to
    /// the bond head before masking. Generation support: the table travels to
    /// the sampling kernel as a plain tensor, with no device read.
    pub fn bond_by_type_value(&self) -> Tensor<R, E> {
        self.bond_by_type.value()
    }

    /// One sampler position: embed `token` (`[rows, 4]`) at `position`,
    /// advance every layer with [`Mamba3Block::step`] and cross-attention over
    /// the cached keys and values, refresh the atom-memory row when `token`
    /// added an atom (row = atom count before the token; the content is the
    /// previous output, which predicted the token, matching the teacher's
    /// `gather_tokens(h, atom_step - 1)`), and return the unmasked head
    /// outputs of architecture §3.6 (`bond_by_type` added by the caller from
    /// the parameter table). `rows_per_spectrum` groups the rows into spectra
    /// for cross-attention. `grammar_state` is the `[rows, 3A + 16]` grammar
    /// row after `token` was applied (in generation the sampler applies each
    /// token before the next call reads it, and START is applied by
    /// initialisation); the atom count and residual valences come from it, so
    /// this performs no device read and its launch count is independent of
    /// the rows and the token values.
    pub fn step_logits(
        &self,
        encoded: &EncoderOutput<R, E>,
        formula_embedding: &Var<R, E>,
        token: &IdTensor<R>,
        position: usize,
        grammar_state: &IdTensor<R>,
        state: &mut DecoderState<R, E>,
        rows_per_spectrum: usize,
    ) -> Result<StepHeads<R, E>> {
        if token.shape().rank() != 2
            || formula_embedding.rank() != 2
            || encoded.memory.rank() != 3
            || encoded.memory_mask.rank() != 2
            || grammar_state.shape().rank() != 2
        {
            return Err(Error::shape(format!(
                "Ms2Decoder::step_logits needs token [rows, 4], formula [rows, d], memory [B, 1+N, d], mask [B, 1+N] and grammar_state [rows, 3A + 16], got {} and {} and {} and {} and {}",
                token.shape(),
                formula_embedding.shape(),
                encoded.memory.shape(),
                encoded.memory_mask.shape(),
                grammar_state.shape()
            )));
        }
        let rows = token.shape().dim(0);
        let want_token: &[usize] = &[rows, 4];
        let want_formula: &[usize] = &[rows, self.d_model];
        let want_grammar: &[usize] = &[
            rows,
            crate::tensor::ops::ms2::replay_state_width(self.max_atoms),
        ];
        if token.shape().dims() != want_token
            || formula_embedding.dims() != want_formula
            || grammar_state.shape().dims() != want_grammar
            || rows_per_spectrum == 0
            || !rows.is_multiple_of(rows_per_spectrum)
            || encoded.memory.shape().dim(0) != rows / rows_per_spectrum
            || state.caches.len() != self.layers.len()
            || state.keys.len() != self.layers.len()
            || state.values.len() != self.layers.len()
        {
            return Err(Error::shape(format!(
                "Ms2Decoder::step_logits has mismatched batch shapes: token {}, formula {}, memory {}, grammar_state {} and rows_per_spectrum {rows_per_spectrum}",
                token.shape(),
                formula_embedding.shape(),
                encoded.memory.shape(),
                grammar_state.shape()
            )));
        }
        if position >= STEP_ROWS {
            return Err(Error::shape(format!(
                "Ms2Decoder::step_logits: position {position} exceeds the {STEP_ROWS}-row step table"
            )));
        }
        let d = self.d_model;
        let a = self.max_atoms;
        let spectra = rows / rows_per_spectrum;
        // Static input embedding: the same six lookups as the teacher pass.
        let field = |c: usize| -> Result<IdTensor<R>> {
            slice_ids_along(token, 1, c, 1)?.reshape(vec![rows])
        };
        let lookup = |table: &Param<R, E>, ids: &IdTensor<R>| -> Result<Var<R, E>> {
            Var::ms2_lookup(&table.var_standalone(), ids)
        };
        let e_step = self
            .step_emb
            .var_standalone()
            .slice(0, position, 1)?
            .reshape(vec![1, d])?
            .expand(vec![rows, d])?;
        let mut x = lookup(&self.kind_emb, &field(0)?)?
            .add(&lookup(&self.type_emb, &field(1)?)?)?
            .add(&lookup(&self.bond_emb, &field(2)?)?)?
            .add(&lookup(&self.ptr_emb, &field(3)?)?)?
            .add(&e_step)?
            .add(formula_embedding)?;
        x = x.unsqueeze(1)?;
        for (l, layer) in self.layers.iter().enumerate() {
            let (y1, cache) = layer.mixer.step(&x, &state.caches[l])?;
            state.caches[l] = cache;
            let y = y1.reshape(vec![spectra, rows_per_spectrum, d])?;
            // The keys and values `start_state` computed once per generation
            // call; the attention arithmetic is the shared helper, so this
            // path matches the teacher exactly.
            let k = Var::constant(state.keys[l].clone());
            let v = Var::constant(state.values[l].clone());
            let ctx = self.attend_cached(layer, &y, &k, &v, &encoded.memory_mask)?;
            let back = ctx.reshape(vec![rows, 1, d])?;
            x = y1.add(&back)?;
        }
        let h = x.reshape(vec![rows, d])?;
        // Refresh the atom memory and residual ids on the device from the
        // grammar row, then roll the previous output: O(1) launches per
        // step, no device read.
        atom_memory_update(
            token,
            grammar_state,
            &state.prev_h,
            &mut state.atom_memory,
            &mut state.resid_ids,
            self.max_atoms,
        )?;
        state.prev_h = h.tensor().clone();
        // Keys carry the clamped residuals from the preallocated id buffer.
        let keys0 = self
            .mem_proj
            .apply(&Var::constant(state.atom_memory.clone()))?;
        let keys_r = Var::ms2_lookup(&self.e_residual.var_standalone(), &state.resid_ids)?
            .reshape(vec![rows, a, d])?;
        let keys = keys0.add(&keys_r)?;
        let scale = 1.0 / (d as f32).sqrt();
        let kind = self.kind_head.apply(&h)?;
        let atom_type = self.type_head.apply(&h)?;
        let bond_base = self.bond_head.apply(&h)?;
        let q0 = self.ptr_query.apply(&h)?;
        let pointer_base = q0
            .unsqueeze(1)?
            .matmul_nt(&keys)?
            .squeeze(1)?
            .mul_scalar(scale);
        let expand_table = |table: &Param<R, E>, n: usize| -> Result<Var<R, E>> {
            table
                .var_standalone()
                .unsqueeze(0)?
                .expand(vec![rows, n, d])
        };
        let pointer_by_type = expand_table(&self.e_ptr_type, COND_ROWS)?
            .matmul_nt(&keys)?
            .mul_scalar(scale);
        let pointer_by_bond = expand_table(&self.e_ptr_bond, 4)?
            .matmul_nt(&keys)?
            .mul_scalar(scale);
        Ok(StepHeads {
            kind,
            atom_type,
            bond_base,
            pointer_base,
            pointer_by_type,
            pointer_by_bond,
        })
    }
}

impl<R: Runtime, E: FloatElem> Ms2Decoder<R, E> {
    /// [`Ms2Decoder::step_logits`] as the fused step: the same position,
    /// the same state transition and the same head values, written straight
    /// into the packed sampler row `logits` (`[rows, 27 + 24 A]`, the layout
    /// of [`crate::tensor::ops::ms2::sample_logits_offsets`]) with one launch
    /// per stage instead of the composed ops.
    ///
    /// Per step: one embedding kernel; per layer the mixer step, the query
    /// projection, two attention kernels and the output projection; one
    /// atom-key kernel; one product for every head and projection; one
    /// kernel for the packed logits. The atom memory is kept projected
    /// ([`FusedStep::atom_keys`]): the projection of the previous output is a
    /// segment of the previous step's head row, so the pointer head never
    /// re-projects the whole memory. `state` must come from
    /// [`Ms2Decoder::start_state_fused`] with `fused` present. No device
    /// read; the launch count is independent of the rows and the tokens.
    #[allow(clippy::too_many_arguments)]
    pub fn step_packed(
        &self,
        encoded: &EncoderOutput<R, E>,
        formula_embedding: &Var<R, E>,
        token: &IdTensor<R>,
        position: usize,
        grammar_state: &IdTensor<R>,
        state: &mut DecoderState<R, E>,
        rows_per_spectrum: usize,
        logits: &mut Tensor<R, E>,
    ) -> Result<()> {
        let _no_grad = crate::autograd::no_grad();
        if token.shape().rank() != 2
            || formula_embedding.rank() != 2
            || encoded.memory.rank() != 3
            || encoded.memory_mask.rank() != 2
            || grammar_state.shape().rank() != 2
        {
            return Err(Error::shape(format!(
                "Ms2Decoder::step_packed needs token [rows, 4], formula [rows, d], memory [B, 1+N, d], mask [B, 1+N] and grammar_state [rows, 3A + 16], got {} and {} and {} and {} and {}",
                token.shape(),
                formula_embedding.shape(),
                encoded.memory.shape(),
                encoded.memory_mask.shape(),
                grammar_state.shape()
            )));
        }
        let rows = token.shape().dim(0);
        let d = self.d_model;
        let a = self.max_atoms;
        let want_token: &[usize] = &[rows, 4];
        let want_formula: &[usize] = &[rows, d];
        let want_grammar: &[usize] = &[rows, crate::tensor::ops::ms2::replay_state_width(a)];
        if token.shape().dims() != want_token
            || formula_embedding.dims() != want_formula
            || grammar_state.shape().dims() != want_grammar
            || rows_per_spectrum == 0
            || !rows.is_multiple_of(rows_per_spectrum)
            || encoded.memory.shape().dim(0) != rows / rows_per_spectrum
            || (!state.carries_in_place() && state.caches.len() != self.layers.len())
            || state.keys.len() != self.layers.len()
            || state.values.len() != self.layers.len()
        {
            return Err(Error::shape(format!(
                "Ms2Decoder::step_packed has mismatched batch shapes: token {}, formula {}, memory {}, grammar_state {} and rows_per_spectrum {rows_per_spectrum}",
                token.shape(),
                formula_embedding.shape(),
                encoded.memory.shape(),
                grammar_state.shape()
            )));
        }
        if position >= STEP_ROWS {
            return Err(Error::shape(format!(
                "Ms2Decoder::step_packed: position {position} exceeds the {STEP_ROWS}-row step table"
            )));
        }
        let DecoderState {
            caches,
            prev_h,
            resid_ids,
            keys,
            values,
            fused,
            ..
        } = state;
        let Some(fused) = fused.as_mut() else {
            return Err(Error::config(
                "Ms2Decoder::step_packed needs a state from start_state_fused with the fused tables present"
                    .to_string(),
            ));
        };
        let x0 = step_embed(&fused.embed, token, formula_embedding.tensor(), position, a)?;
        let mut x = Var::constant(x0.reshape(vec![rows, 1, d])?);
        for (l, layer) in self.layers.iter().enumerate() {
            let y1 = match fused.carries.as_mut() {
                Some(carries) => layer.mixer.step_in_place(&x, &mut carries[l])?,
                None => {
                    let (y1, cache) = layer.mixer.step(&x, &caches[l])?;
                    caches[l] = cache;
                    y1
                }
            };
            // Cross-attention of the single query of each row over the keys
            // and values `start_state_fused` computed once: the arithmetic of
            // `attend_cached`, with the head split folded into the kernels'
            // indexing instead of per-step permutes of the whole memory.
            let q = layer
                .q
                .apply(&layer.norm.apply(&y1)?)?
                .tensor()
                .reshape(vec![rows, d])?;
            attn_weights(
                &q,
                &keys[l],
                &encoded.memory_mask,
                &mut fused.attn_w,
                self.n_heads,
                rows_per_spectrum,
            )?;
            let ctx = attn_context(&fused.attn_w, &values[l], rows_per_spectrum)?;
            let back = layer
                .o
                .apply(&Var::constant(ctx.reshape(vec![rows, 1, d])?))?;
            x = y1.add(&back)?;
        }
        let h = x.tensor().reshape(vec![rows, d])?;
        // The atom added by `token` was predicted by the previous output:
        // its key row is the projection segment of the previous head row.
        atom_key_update(
            token,
            grammar_state,
            &fused.prev_heads,
            step_head_layout(d).key,
            &mut fused.atom_keys,
            resid_ids,
            a,
        )?;
        // Every head and projection of this output in one product
        // (`step_head_layout`): the kind, atom-type and bond-base logits, the
        // pointer query with its residual products, and the atom-memory key
        // with its conditioning products, which the next step may store.
        // The bare product: only the 27 logits have a bias, and the kernel
        // that packs them adds it.
        let heads = matmul(&h, &fused.head_w)?;
        *prev_h = h;
        step_logits_pack(
            &heads,
            &fused.atom_keys,
            resid_ids,
            &fused.ptr_cross,
            &fused.head_b,
            logits,
            a,
        )?;
        fused.prev_heads = heads;
        Ok(())
    }
}

impl<R: Runtime, E: FloatElem> Module<R, E> for Ms2Decoder<R, E> {
    fn visit(&self, visitor: &mut ModuleVisitor<'_, R, E>) {
        visitor.param("kind_emb", &self.kind_emb);
        visitor.param("type_emb", &self.type_emb);
        visitor.param("bond_emb", &self.bond_emb);
        visitor.param("ptr_emb", &self.ptr_emb);
        visitor.param("step_emb", &self.step_emb);
        for (i, l) in self.layers.iter().enumerate() {
            visitor.child_at("layers", i, l);
        }
        visitor.child("kind_head", &self.kind_head);
        visitor.child("type_head", &self.type_head);
        visitor.child("bond_head", &self.bond_head);
        visitor.child("mem_proj", &self.mem_proj);
        visitor.child("ptr_query", &self.ptr_query);
        visitor.param("bond_by_type", &self.bond_by_type);
        visitor.param("e_residual", &self.e_residual);
        visitor.param("e_ptr_type", &self.e_ptr_type);
        visitor.param("e_ptr_bond", &self.e_ptr_bond);
    }
}

/// The replay buffers a teacher pass scores, borrowing the device tensors.
pub struct ReplayView<'a, R: Runtime> {
    /// `[rows, T, 4 + A]` masks and residuals.
    pub replay: &'a IdTensor<R>,
    /// `[rows, A + 1]` atom steps and first illegal step.
    pub atoms: &'a IdTensor<R>,
}

impl<R: Runtime> ReplayView<'_, R> {
    fn rows(&self) -> usize {
        self.replay.shape().dim(0)
    }

    fn widths(&self) -> (usize, usize) {
        (self.replay.shape().dim(1), self.replay.shape().dim(2) - 4)
    }
}

/// `(1 / B) * sum_g q_g * nll_g` (architecture §4.4): an unlabeled spectrum
/// contributes 0 through its zero `q` weights, and the divisor is `B`, the
/// spectra count including unlabeled ones. No device read.
pub fn graph_loss<R: Runtime, E: FloatElem>(
    out: &TeacherOutput<R, E>,
    q: &Tensor<R, E>,
    spectra: usize,
) -> Result<Var<R, E>> {
    if out.nll.rank() != 1 || q.rank() != 1 {
        return Err(Error::shape(format!(
            "graph_loss needs nll [B*G] and q [B*G], got {} and {}",
            out.nll.shape(),
            q.shape()
        )));
    }
    if out.nll.dims() != q.dims() {
        return Err(Error::shape(format!(
            "graph_loss needs nll and q of one length, got {} and {}",
            out.nll.shape(),
            q.shape()
        )));
    }
    if spectra == 0 {
        return Err(Error::shape(
            "graph_loss needs at least one spectrum".to_string(),
        ));
    }
    let qv = Var::constant(q.clone());
    Ok(out.nll.mul(&qv)?.sum()?.mul_scalar(1.0 / spectra as f32))
}
