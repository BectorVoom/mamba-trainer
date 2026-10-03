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
use crate::nn::init::Initializer;
use crate::nn::linear::{Linear, LinearConfig};
use crate::nn::module::{Module, ModuleVisitor};
use crate::nn::norm::{RmsNorm, RmsNormConfig};
use crate::nn::param::Param;
use crate::tensor::Shape;
use crate::tensor::Tensor;
use crate::tensor::ops::index::{IdTensor, slice_ids_along};
use crate::tensor::ops::ms2::{atom_memory_update, bits_to_mask, pred_ids, teacher_ids};
use crate::tensor::ops::random::Rng;
use crate::tensor::ops::{elemwise, movement};

use super::contract::ModelConfig;
use super::encoder::EncoderOutput;
use super::targets_batch::DeviceTargets;

/// Step-embedding rows: `T <= 64` by contract §3.4, so a fixed table covers
/// every generation config without knowing `T` at init.
const STEP_ROWS: usize = 64;

/// Atom-type rows of the pointer conditioning tables: the 17 types plus row 0
/// (unused) and row 18 (CLOSE_RING).
const COND_ROWS: usize = 19;

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
    /// Per-layer mixer caches, `[rows, ..]`.
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
    /// (`[b, m]` 1/0). Scores are masked with `mask_logits` before the
    /// softmax. The teacher pass and the stepped pass share this helper, so
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
        let b = q_in.dims()[0];
        let qt = q_in.dims()[1];
        let m = k.dims()[1];
        let d = self.d_model;
        let h = self.n_heads;
        let hd = d / h;
        let q = layer.q.apply(&layer.norm.apply(q_in)?)?;
        let qh = q.reshape(vec![b, qt, h, hd])?.permute(&[0, 2, 1, 3])?;
        let kh = k.reshape(vec![b, m, h, hd])?.permute(&[0, 2, 3, 1])?;
        let scores = qh.matmul(&kh)?.mul_scalar(1.0 / (hd as f32).sqrt());
        let flat = mask.reshape(Shape::new(vec![b, 1, 1, m]))?;
        let full = elemwise::expand(&flat, &Shape::new(vec![b, h, qt, m]))?;
        let weights = scores.mask_logits(&full)?.softmax(3)?;
        let vh = v.reshape(vec![b, m, h, hd])?.permute(&[0, 2, 1, 3])?;
        let ctx = weights
            .matmul(&vh)?
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
        // Position ids `[rows * T]`: position `i` at every row's slot `i`.
        let mut step_ids = vec![0u32; rows * t];
        for r in 0..rows {
            for i in 0..t {
                step_ids[r * t + i] = i as u32;
            }
        }
        let step_t = IdTensor::from_slice(&step_ids, vec![rows * t], device)?;
        let e_step = embed_col(&self.step_emb, &step_t)?;
        // The formula embedding broadcasts over the spectrum's G targets.
        let slots = rows / spectra;
        let e_formula = formula
            .reshape(vec![spectra, 1, 1, self.d_model])?
            .expand(vec![spectra, slots, t, self.d_model])?
            .reshape(vec![rows, t, self.d_model])?;
        e_kind
            .add(&e_type)?
            .add(&e_bond)?
            .add(&e_ptr)?
            .add(&e_step)?
            .add(&e_formula)
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
    ) -> Result<Var<R, E>> {
        let slots = rows / spectra;
        let mut x = x.clone();
        for layer in &self.layers {
            x = layer.mixer.apply(&x)?;
            let q = x.reshape(vec![spectra, slots * t, self.d_model])?;
            // Keys and values once per pass (the stepped path reuses the ones
            // `start_state` computed); the attention arithmetic itself is the
            // shared `attend_cached` helper.
            let k = layer.k.apply(&encoded.memory)?;
            let v = layer.v.apply(&encoded.memory)?;
            let ctx = self.attend_cached(layer, &q, &k, &v, &encoded.memory_mask)?;
            let back = ctx.reshape(vec![rows, t, self.d_model])?;
            x = x.add(&back)?;
        }
        Ok(x)
    }

    /// Effective float mask `[rows, W]` for one field: the replay bit mask
    /// where the field is used, "index 0 only" elsewhere (architecture §3.8).
    fn effective_mask(
        bits: &IdTensor<R>,
        use_col: &Tensor<R, E>,
        width: usize,
        index0: &Tensor<R, E>,
        rows: usize,
    ) -> Result<Tensor<R, E>> {
        let mask = bits_to_mask(bits, width)?;
        let use_full = elemwise::expand(
            &use_col.reshape(Shape::new(vec![rows, 1]))?,
            &Shape::new(vec![rows, width]),
        )?;
        let idle = elemwise::expand(index0, &Shape::new(vec![rows, width]))?;
        let on = elemwise::mul(&mask, &use_full)?;
        let off = elemwise::mul(&idle, &elemwise::rsub_scalar(&use_full, 1.0))?;
        elemwise::add(&on, &off)
    }

    /// The full teacher pass behind [`teacher`] and [`field_distributions`].
    ///
    /// [`teacher`]: Ms2Decoder::teacher
    /// [`field_distributions`]: Ms2Decoder::field_distributions
    fn run(
        &self,
        encoded: &EncoderOutput<R, E>,
        formula_embedding: &Var<R, E>,
        targets: &DeviceTargets<R, E>,
        replay: &ReplayView<'_, R>,
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
        let spectra = encoded.memory.shape().dim(0);
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
        let want_formula: &[usize] = &[spectra, self.d_model];
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
        let x = self.embed_teacher(&targets.tokens, formula_embedding, rows, t, spectra)?;
        let h = self.apply_layers(&x, encoded, rows, t, spectra)?;
        let atom_cols = slice_ids_along(replay.atoms, 1, 0, a)?.reshape(vec![rows * a])?;
        let pred = pred_ids(&atom_cols)?;
        let atom_mem = Var::gather_tokens(&h, &pred, a)?;
        let lengths = slice_ids_along(&targets.meta, 1, 0, 1)?.reshape(vec![rows])?;
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
        let mut field_rows: Vec<Var<R, E>> = Vec::with_capacity(t);
        let mut kind_rows: Vec<Var<R, E>> = Vec::with_capacity(t);
        let mut type_rows: Vec<Var<R, E>> = Vec::with_capacity(t);
        let mut bond_rows: Vec<Var<R, E>> = Vec::with_capacity(t);
        let mut ptr_rows: Vec<Var<R, E>> = Vec::with_capacity(t);
        let positions = t.saturating_sub(1);
        for i in 0..positions {
            let pos = i + 1;
            // The masks for output position `i` are `replay[.., i + 1, 0..4]`
            // (the state before token `i + 1`); the target ids come from the
            // in-range conditioning kernel.
            let tgt_col = slice_ids_along(&targets.tokens, 1, pos, 1)?.reshape(vec![rows, 4])?;
            let cond = teacher_ids(&tgt_col, &lengths, pos, a)?;
            let cond_col = |c: usize| -> Result<IdTensor<R>> {
                slice_ids_along(&cond, 1, c, 1)?.reshape(vec![rows])
            };
            let replay_row =
                slice_ids_along(replay.replay, 1, pos, 1)?.reshape(vec![rows, 4 + a])?;
            let bits = |f: usize| -> Result<IdTensor<R>> {
                slice_ids_along(&replay_row, 1, f, 1)?.reshape(vec![rows])
            };
            let resid = slice_ids_along(&replay_row, 1, 4, a)?.reshape(vec![rows * a])?;
            let use_row = movement::slice(&targets.use_mask, 1, i, 1)?.reshape(vec![rows, 4])?;
            let use_col = |f: usize| movement::slice(&use_row, 1, f, 1)?.reshape(vec![rows]);
            let h_i = h.slice(1, i, 1)?.reshape(vec![rows, d])?;
            // Kind, atom type: plain head logits.
            let kind_logits = self.kind_head.apply(&h_i)?;
            let kind_eff = Self::effective_mask(&bits(0)?, &use_col(0)?, 5, &index0_kind, rows)?;
            let kind_lp = kind_logits.mask_logits(&kind_eff)?.log_softmax(1)?;
            let kind_got = kind_lp.take_along_last(&cond_col(0)?)?;
            let type_logits = self.type_head.apply(&h_i)?;
            let type_eff = Self::effective_mask(&bits(1)?, &use_col(1)?, 18, &index0_type, rows)?;
            let type_lp = type_logits.mask_logits(&type_eff)?.log_softmax(1)?;
            let type_got = type_lp.take_along_last(&cond_col(1)?)?;
            // Bond: the learned row `bond_by_type[c]` joins before masking.
            let bond_base = self.bond_head.apply(&h_i)?;
            let bond_corr = Var::ms2_lookup(&self.bond_by_type.var_standalone(), &cond_col(4)?)?;
            let bond_logits = bond_base.add(&bond_corr)?;
            let bond_eff = Self::effective_mask(&bits(2)?, &use_col(2)?, 4, &index0_bond, rows)?;
            let bond_lp = bond_logits.mask_logits(&bond_eff)?.log_softmax(1)?;
            let bond_got = bond_lp.take_along_last(&cond_col(2)?)?;
            // Pointer: `(Linear(h) + E_ptr_type[c] + E_ptr_bond[b]) · k_j /
            // sqrt(d)` with `k_j = Linear(memory_j) +
            // E_residual[min(residual_j, 7)]`, added before masking.
            let q0 = self.ptr_query.apply(&h_i)?;
            let qt = Var::ms2_lookup(&self.e_ptr_type.var_standalone(), &cond_col(4)?)?;
            let qb = Var::ms2_lookup(&self.e_ptr_bond.var_standalone(), &cond_col(2)?)?;
            let query = q0.add(&qt)?.add(&qb)?.unsqueeze(1)?;
            let keys0 = self.mem_proj.apply(&atom_mem)?;
            let keys_r = Var::ms2_lookup(&self.e_residual.var_standalone(), &resid)?
                .reshape(vec![rows, a, d])?;
            let keys = keys0.add(&keys_r)?;
            let ptr_scores = query.matmul_nt(&keys)?.squeeze(1)?.mul_scalar(ptr_scale);
            let ptr_eff = Self::effective_mask(&bits(3)?, &use_col(3)?, a, &index0_ptr, rows)?;
            let ptr_lp = ptr_scores.mask_logits(&ptr_eff)?.log_softmax(1)?;
            let ptr_got = ptr_lp.take_along_last(&cond_col(3)?)?;
            // A field the target does not use, or an unscored position,
            // contributes exactly 0: its id is already 0 (the "index 0 only"
            // value, exactly 0 after log-softmax) and the product with
            // `use` zeroes any residual.
            let parts = [&kind_got, &type_got, &bond_got, &ptr_got];
            for (f, got) in parts.iter().enumerate() {
                let use_v = Var::constant(use_col(f)?);
                nll = nll.sub(&got.mul(&use_v)?)?;
            }
            let widen = |got: &Var<R, E>| got.reshape(vec![rows, 1]);
            field_rows.push(
                cat(
                    &[
                        widen(&kind_got)?,
                        widen(&type_got)?,
                        widen(&bond_got)?,
                        widen(&ptr_got)?,
                    ],
                    1,
                )?
                .unsqueeze(1)?,
            );
            let widen_t = |lp: &Var<R, E>, w: usize| lp.reshape(vec![rows, 1, w]);
            kind_rows.push(widen_t(&kind_lp, 5)?);
            type_rows.push(widen_t(&type_lp, 18)?);
            bond_rows.push(widen_t(&bond_lp, 4)?);
            ptr_rows.push(widen_t(&ptr_lp, a)?);
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
        let run = self.run(encoded, formula_embedding, targets, replay)?;
        Ok(TeacherOutput {
            nll: run.nll,
            field_log_prob: run.fields,
        })
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
        let run = self.run(encoded, formula_embedding, targets, replay)?;
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
        self.apply_layers(&x, encoded, rows, t, spectra)
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
        Ok(DecoderState {
            caches: self
                .layers
                .iter()
                .map(|l| l.mixer.empty_cache(rows, device))
                .collect(),
            atom_memory: Tensor::zeros(vec![rows, a, d], device),
            prev_h: Tensor::zeros(vec![rows, d], device),
            resid_ids: resid,
            keys,
            values,
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
