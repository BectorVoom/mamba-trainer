//! Fragment-ion assignment head (architecture §2.2) and its loss (§2.3).
//!
//! [`AssignmentHead::log_prob`] scores, for each `(b, f, p)` with the
//! hypotheses of [`ion_assign`](crate::tensor::ops::ms2_ion::ion_assign) in
//! `ion [B, F, N, J, 12]` / `ion_meta [B, F, N, 4]`, a distribution over the
//! kept hypotheses and an explicit **unassigned** class: `logit_j =
//! (e(u_j, h_j) · W x_p) / sqrt(d)` with `e` the
//! [`FormulaHead`](super::formula_head::FormulaHead)'s row network on the
//! `ln(1 + count)` hypothesis features (through
//! [`count_features`](crate::tensor::ops::ms2::count_features), so the
//! gradients of `L_assign` reach the formula head's row network too) and `x_p`
//! the encoder output of the peak; `logit_unassigned = w · x_p + b`; masked
//! `log_softmax` over the classes that exist (the
//! [`ion_class_mask`](crate::tensor::ops::ms2_assign::ion_class_mask) of
//! `ion_meta`). A peak with no kept hypothesis, with `ion_unavailable`, or a
//! padding peak (the lane marks all three with `ion_unavailable` and an empty
//! row) has probability 1 on unassigned. When the support is incomplete the
//! probabilities are conditional on the retained hypotheses and are reported
//! with that status. The unassigned class is a modelled remainder under
//! partial pseudo-label supervision; it is not a measured noise probability.
//!
//! [`AssignmentHead::loss`] is `L_assign` of §2.3 for the training case
//! `F = 1`: per peak `− log sum_j label_mask_j p_j`, computed stably in log
//! space (`logsumexp` of the masked log-probabilities through the crate's
//! masked-logits primitive — never `log(0)`), gated by the 0/1 eligibility
//! `label_state == 1`, summed and divided by `max(1, eligible count)`.
//! Neither [`log_prob`](AssignmentHead::log_prob) nor
//! [`loss`](AssignmentHead::loss) reads the device: ids stay on the device,
//! the eligible/partial/dropped counts are device reductions packed into one
//! `[3]` tensor the trainer reads inside its single report read.
//!
//! Standalone module (plan item P4.2, neural part): no `Ms2Model`, trainer or
//! driver integration. What the integration task must wire: `x` is the
//! encoder output `[B, N, d]`; `ion`/`ion_meta` come from `ms2_ion_assign`
//! over the retained formulas in generation (one row per `(b, f, p)`) and
//! over the true parent composition (`F = 1`, `top_counts = gold_counts`) in
//! training; `label_mask [B, N, J + 1]` / `label_state [B, N]` come from
//! `ms2_ion_label_mask`; the training objective adds `lambda_a * L_assign`
//! with `lambda_a = 0.1`; the report reads `counts` (eligible, partial,
//! dropped) next to `assignment_label_overflow`.
//!
//! Counts: `eligible` is the number of peaks with `label_state == 1` or `3`
//! (the loss denominator); `dropped` the number with `label_state == 2`
//! (labels exist but none kept, `assignment_label_dropped`); `partial` the
//! number with `label_state == 3` (true partial: at least one label of the
//! peak is kept and at least one label of the peak is not, computed from the
//! label matching itself in `ms2_ion_label_mask`).

use cubecl::prelude::Runtime;

use crate::autograd::{Var, cat};
use crate::backend::{Device, FloatElem};
use crate::error::{Error, Result};
use crate::nn::linear::{Linear, LinearConfig};
use crate::nn::module::{Module, ModuleVisitor};
use crate::tensor::Tensor;
use crate::tensor::ops::index::{IdTensor, ids_to_float};
use crate::tensor::ops::ms2::count_features;
use crate::tensor::ops::ms2_assign::{ION_CLASS_MAX, ion_class_mask};
use crate::tensor::ops::{elemwise, movement, reduce};
use crate::tensor::ops::random::Rng;

use super::contract::ModelConfig;
use super::formula_head::FormulaHead;

/// The fragment-ion assignment head of architecture §2.2.
///
/// `proj` is `Linear(d → d)` with no bias (the peak projection `W`,
/// architecture §2.2 `logit_j = (e_j . W x_p) / sqrt(d)`); `unassigned` is
/// `Linear(d → 1)` (the unassigned logit `w · x_p + b`). The head does NOT
/// own the hypothesis row network: [`log_prob`](AssignmentHead::log_prob)
/// takes `&FormulaHead` and uses its row network on the hypothesis features.
pub struct AssignmentHead<R: Runtime, E: FloatElem> {
    /// `Linear(d → d)` peak projection.
    proj: Linear<R, E>,
    /// `Linear(d → 1)` unassigned logit.
    unassigned: Linear<R, E>,
    /// Residual width.
    d_model: usize,
}

/// Output of [`AssignmentHead::log_prob`].
pub struct AssignOutput<R: Runtime, E: FloatElem> {
    /// `[B, F, N, J + 1]` masked log-softmax over the kept hypotheses and the
    /// unassigned class `J`: log-probability 0 at class `J` and the masked
    /// value elsewhere for peaks with no kept hypothesis, with
    /// `ion_unavailable`, or padding peaks.
    pub log_prob: Var<R, E>,
    /// `[B, F, N, J + 1]` 1/0 class mask the logits were scored under (1 on
    /// the kept hypothesis classes and on unassigned; unassigned-only under
    /// `ion_unavailable`).
    pub mask: Tensor<R, E>,
}

/// Output of [`AssignmentHead::loss`].
pub struct AssignLoss<R: Runtime, E: FloatElem> {
    /// `L_assign`: the eligibility-gated mean negative log masked-sum,
    /// divided by `max(1, eligible count)`. Exactly 0 when no peak is
    /// eligible.
    pub loss: Var<R, E>,
    /// `[3]` device scalars: the eligible count (`label_state == 1 or 3`),
    /// the true-partial count (`label_state == 3`) and the
    /// dropped count (`label_state == 2`). Read by the trainer inside its
    /// single report read; never read here.
    pub counts: Tensor<R, E>,
}

impl<R: Runtime, E: FloatElem> AssignmentHead<R, E> {
    /// Build the head for `model` on `device`.
    pub fn init(model: &ModelConfig, device: &Device<R>, rng: &mut Rng) -> Result<Self> {
        let d = model.d_model as usize;
        if d == 0 {
            return Err(Error::config(
                "AssignmentHead::init: d_model is 0".to_string(),
            ));
        }
        let proj = LinearConfig::new(d, d).with_bias(false).init(device, rng);
        let unassigned = LinearConfig::new(d, 1).init(device, rng);
        Ok(Self {
            proj,
            unassigned,
            d_model: d,
        })
    }

    /// Residual width.
    pub fn d_model(&self) -> usize {
        self.d_model
    }

    /// Assignment log-probabilities over the kept hypotheses and unassigned
    /// (architecture §2.2).
    ///
    /// `ion [B, F, N, J, 12]` holds the kept hypothesis records (10 element
    /// counts with the ion's own hydrogen count, mass, residual offset by
    /// 2^31), `ion_meta [B, F, N, 4]` the per-peak (accepted, ambiguous,
    /// kept, status) words, `x [B, N, d]` the encoder output and `log_table`
    /// the resident `[1024]` `ln(1 + n)` table. `1 <= J <= 8`.
    ///
    /// `logit_j = (e_j · W x_p) / sqrt(d)` for `j < kept` (masked out for
    /// `j >= kept`), `logit_unassigned = w · x_p + b` always present, masked
    /// `log_softmax` over the classes that exist. No device read.
    pub fn log_prob(
        &self,
        formula: &FormulaHead<R, E>,
        log_table: &Tensor<R, E>,
        ion: &IdTensor<R>,
        ion_meta: &IdTensor<R>,
        x: &Var<R, E>,
    ) -> Result<AssignOutput<R, E>> {
        // Every input rank is checked before any dimension is read, so a
        // malformed shape is `Error::Shape` rather than a panic.
        if ion.shape().rank() != 5
            || ion_meta.shape().rank() != 4
            || x.rank() != 3
            || log_table.rank() != 1
        {
            return Err(Error::shape(format!(
                "AssignmentHead::log_prob needs ion [B, F, N, J, 12], ion_meta [B, F, N, 4], x [B, N, d] and log_table [1024], got {} and {} and {} and {}",
                ion.shape(),
                ion_meta.shape(),
                x.shape(),
                log_table.shape()
            )));
        }
        let batch = ion.shape().dim(0);
        let f = ion.shape().dim(1);
        let n = ion.shape().dim(2);
        let j = ion.shape().dim(3);
        if j == 0 || j > ION_CLASS_MAX {
            return Err(Error::shape(format!(
                "AssignmentHead::log_prob needs 1 <= J <= {ION_CLASS_MAX} (architecture §2.2), got J = {j}"
            )));
        }
        if ion.shape().dims() != [batch, f, n, j, 12]
            || ion_meta.shape().dims() != [batch, f, n, 4]
        {
            return Err(Error::shape(format!(
                "AssignmentHead::log_prob needs ion [{batch}, {f}, {n}, {j}, 12] and ion_meta [{batch}, {f}, {n}, 4], got {} and {}",
                ion.shape(),
                ion_meta.shape()
            )));
        }
        let want_x: &[usize] = &[batch, n, self.d_model];
        if x.dims() != want_x {
            return Err(Error::shape(format!(
                "AssignmentHead::log_prob needs x [B, N, d] with d = {}, got {}",
                self.d_model,
                x.shape()
            )));
        }
        if log_table.len() != 1024 {
            return Err(Error::shape(format!(
                "AssignmentHead::log_prob needs log_table [1024], got {}",
                log_table.shape()
            )));
        }
        let rows = batch
            .checked_mul(f)
            .and_then(|v| v.checked_mul(n))
            .and_then(|v| v.checked_mul(j))
            .ok_or_else(|| {
                Error::shape("AssignmentHead::log_prob: B * F * N * J overflows usize".to_string())
            })?;
        // Hypothesis features: `ln(1 + count)` of each record's first 10
        // words through the shared table kernel (integers carry no gradient;
        // the row network below is differentiated).
        let flat = ion.reshape(vec![rows, 12])?;
        let mut feats = Tensor::empty(vec![rows, 10], ion.device());
        count_features(&flat, log_table, &mut feats, 12)?;
        let feats5 = Var::constant(feats).reshape(vec![batch, f, n, j, 10])?;
        // `e(u_j, h_j)`: the formula head's row network, so `L_assign`
        // gradients reach its weights too.
        let e = formula.embed_rows(&feats5)?;
        // Peak queries broadcast over the formula slots and hypotheses.
        let q = self.proj.apply(x)?;
        let qe = q
            .unsqueeze(1)?
            .expand(vec![batch, f, n, self.d_model])?
            .unsqueeze(3)?
            .expand(vec![batch, f, n, j, self.d_model])?;
        let scores = e
            .mul(&qe)?
            .sum_dim(4)?
            .squeeze(4)?
            .mul_scalar(1.0 / (self.d_model as f32).sqrt());
        // The unassigned logit joins as the last class.
        let u = self
            .unassigned
            .apply(x)?
            .unsqueeze(1)?
            .expand(vec![batch, f, n, 1])?;
        let logits = cat(&[scores, u], 3)?;
        // Classes that exist: kept hypotheses plus unassigned, unassigned
        // only under `ion_unavailable` (padding peaks carry the same bit, so
        // they need no separate input).
        let mask = ion_class_mask(ion_meta, j)?;
        let log_prob = logits.mask_logits(&mask)?.log_softmax(3)?;
        Ok(AssignOutput { log_prob, mask })
    }

    /// `L_assign` of architecture §2.3, for the training case `F = 1`.
    ///
    /// Per peak `− log sum_j label_mask_j p_j`, computed stably in log space
    /// (`logsumexp` of the masked log-probabilities through the crate's
    /// masked-logits primitive — never `log(0)`), gated by the 0/1
    /// eligibility `label_state == 1 or 3`, summed and divided by `max(1, eligible
    /// count)`. `label_mask` is `[B, N, J + 1]` float 0/1 (class `J` is
    /// unassigned; rows outside states 1 and 3 carry its one-hot, so their masked
    /// log-sum is finite and then multiplied by the 0 eligibility indicator),
    /// `label_state` is `[B, N]` (`0` no label, `1` some label kept with every
    /// label kept, `2` labels exist but none kept, `3` true partial: some label
    /// kept while some label of the same peak is not). No device read.
    pub fn loss(
        &self,
        out: &AssignOutput<R, E>,
        label_mask: &Tensor<R, E>,
        label_state: &IdTensor<R>,
    ) -> Result<AssignLoss<R, E>> {
        // Every input rank is checked before any dimension is read, so a
        // malformed shape is `Error::Shape` rather than a panic.
        if out.log_prob.rank() != 4 || label_mask.rank() != 3 || label_state.shape().rank() != 2 {
            return Err(Error::shape(format!(
                "AssignmentHead::loss needs log_prob [B, F, N, J + 1] with F = 1, label_mask [B, N, J + 1] and label_state [B, N], got {} and {} and {}",
                out.log_prob.shape(),
                label_mask.shape(),
                label_state.shape()
            )));
        }
        let lp_dims = out.log_prob.dims().to_vec();
        let (batch, f, n, width) = (lp_dims[0], lp_dims[1], lp_dims[2], lp_dims[3]);
        if f != 1 {
            return Err(Error::shape(format!(
                "AssignmentHead::loss needs the training case F = 1, got F = {f}"
            )));
        }
        if width < 2 {
            return Err(Error::shape(format!(
                "AssignmentHead::loss needs J + 1 >= 2 classes, got width {width}"
            )));
        }
        let _j = width - 1;
        let want_mask: &[usize] = &[batch, n, width];
        let want_state: &[usize] = &[batch, n];
        if label_mask.dims() != want_mask || label_state.shape().dims() != want_state {
            return Err(Error::shape(format!(
                "AssignmentHead::loss needs label_mask [{batch}, {n}, {width}] and label_state [{batch}, {n}], got {} and {}",
                label_mask.shape(),
                label_state.shape()
            )));
        }
        // `− log sum_j mask_j p_j` in log space: the masked logits carry the
        // label mask (unlabelled classes at the masked value, whose
        // exponentials are exactly 0), the max-subtracted `logsumexp` never
        // forms `log(0)` — not even for degenerate all-masked caller rows,
        // whose shifted exponentials are all 1.
        let lp = out.log_prob.reshape(vec![batch, n, width])?;
        let masked = lp.mask_logits(label_mask)?;
        let peak_max = masked.max_dim(2)?.detach();
        let log_sum = masked
            .sub(&peak_max)?
            .exp()
            .sum_dim(2)?
            .log()
            .add(&peak_max)?
            .squeeze(2)?
            .neg();
        // Eligibility without a read, as in `FormulaHead::loss`: states 1
        // (fully retained) and 3 (true partial) are eligible; state 3 is the
        // partial count; state 2 is dropped.
        let state_f: Tensor<R, E> = ids_to_float(label_state);
        let elig1 = elemwise::eq_scalar(&state_f, 1.0);
        let partial_t = elemwise::eq_scalar(&state_f, 3.0);
        let elig_t = elemwise::add(&elig1, &partial_t)?;
        let elig = Var::constant(elig_t.clone());
        let total = log_sum.mul(&elig)?.sum()?;
        let denom_unclamped = elig.sum()?;
        let one = Var::constant(Tensor::full(
            denom_unclamped.shape().clone(),
            1.0,
            out.log_prob.device(),
        ));
        let denom = denom_unclamped.maximum(&one)?;
        let loss = total.div(&denom)?;
        // Report counts, still without a read: eligible (`state == 1 or 3`),
        // partial (`state == 3`, emitted by `ms2_ion_label_mask` from the
        // label matching itself) and dropped (`state == 2`), packed into one
        // `[3]` tensor for the trainer's single report read.
        let dropped_t = elemwise::eq_scalar(&state_f, 2.0);
        let e_sum = reduce::sum_all(&elig_t)?.reshape(vec![1])?;
        let p_sum = reduce::sum_all(&partial_t)?.reshape(vec![1])?;
        let d_sum = reduce::sum_all(&dropped_t)?.reshape(vec![1])?;
        let counts = movement::cat(&[e_sum, p_sum, d_sum], 0)?;
        Ok(AssignLoss { loss, counts })
    }
}

impl<R: Runtime, E: FloatElem> Module<R, E> for AssignmentHead<R, E> {
    fn visit(&self, visitor: &mut ModuleVisitor<'_, R, E>) {
        visitor.child("proj", &self.proj);
        visitor.child("unassigned", &self.unassigned);
    }
}

/// Host twin of
/// [`ion_class_mask`](crate::tensor::ops::ms2_assign::ion_class_mask): the
/// kernel-expressible form the `#[cube]` kernel copies.
///
/// `ion_meta` is flat `[B, F, N, 4]` (accepted, ambiguous, kept, status);
/// returns flat `[B, F, N, J + 1]` floats: 1 on the kept hypothesis classes
/// (`j < min(kept, J)`) and on unassigned (`J`), 0 elsewhere;
/// unassigned-only when bit 2 (`ion_unavailable`) of the status is set.
/// Status bits 0 and 1 do not change the mask.
pub fn ion_class_mask_host(
    ion_meta: &[u32],
    batch: usize,
    f: usize,
    n: usize,
    j: usize,
) -> Vec<f32> {
    let width = j + 1;
    let mut out = vec![0.0f32; batch * f * n * width];
    for row in 0..batch * f * n {
        let kept = ion_meta[row * 4 + 2];
        let status = ion_meta[row * 4 + 3];
        let mut eff = kept.min(j as u32) as usize;
        if status & 4 != 0 {
            eff = 0;
        }
        for col in 0..width {
            let mut v = 0.0f32;
            if col < eff {
                v = 1.0;
            }
            if col == j {
                v = 1.0;
            }
            out[row * width + col] = v;
        }
    }
    out
}

/// Checked activation bytes this head materialises for `(b, f, n, j, d)` at
/// `elem_bytes` bytes per float: the hypothesis features `[B, F, N, J, 10]`,
/// two row-network activations `[B, F, N, J, d]`, and the logits and
/// log-probabilities `[B, F, N, J + 1]` — for the memory estimate.
pub fn activation_bytes(
    b: usize,
    f: usize,
    n: usize,
    j: usize,
    d: usize,
    elem_bytes: usize,
) -> Result<u64> {
    fn mul(a: u64, v: usize, item: &'static str) -> Result<u64> {
        a.checked_mul(v as u64)
            .ok_or_else(|| Error::config(format!("memory estimate overflow: {item}")))
    }
    fn add(a: u64, v: u64, item: &'static str) -> Result<u64> {
        a.checked_add(v)
            .ok_or_else(|| Error::config(format!("memory estimate overflow: {item}")))
    }
    let bfn = mul(
        mul(mul(b as u64, f, "assign_features")?, n, "assign_features")?,
        j,
        "assign_features",
    )?;
    let features = mul(mul(bfn, 10, "assign_features")?, elem_bytes, "assign_features")?;
    let row_one = mul(mul(bfn, d, "assign_row")?, elem_bytes, "assign_row")?;
    let row_acts = add(row_one, row_one, "assign_row")?;
    let j1 = j
        .checked_add(1)
        .ok_or_else(|| Error::config("memory estimate overflow: assign_logits".to_string()))?;
    let cells = mul(
        mul(
            mul(b as u64, f, "assign_logits")?,
            n,
            "assign_logits",
        )?,
        j1,
        "assign_logits",
    )?;
    let logits = mul(cells, elem_bytes, "assign_logits")?;
    let log_prob = mul(cells, elem_bytes, "assign_log_prob")?;
    add(
        add(add(features, row_acts, "assign")?, logits, "assign")?,
        log_prob,
        "assign",
    )
}
