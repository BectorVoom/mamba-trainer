//! Reranker features, model and training examples (`docs/MS2_V1_ARCHITECTURE.md`
//! §4.3; plan item P6.3, machinery only).
//!
//! Standalone components: no hook into [`crate::models::ms2::generate`]. The
//! integration task (a later task) fills the two input buffers this module
//! cannot derive on the device and calls the kernel in
//! [`crate::tensor::ops::ms2_rerank`]:
//!
//! * `scores [B*K, 2]` — `(trace_log_prob, formula_log_prob)` per trajectory.
//!   Today the trace term lives in the `actions` record and the formula term in
//!   `top_log_prob [B, F]`, so the integration copies the trace bits and
//!   indexes `top_log_prob` with the trajectory's formula slot (exactly as the
//!   `ms2_rank` integration does; see [`crate::models::ms2::pack`]).
//! * `evidence_f [B*K, 2]` — `(largest evidence log-probability, smallest
//!   |residual| in units of the fragment tolerance)` per trajectory, `(0, 1)`
//!   when the candidate has no evidence. These need the assignment head's
//!   log-probabilities and a division by the per-peak fragment tolerance, so
//!   they are host inputs here, not device derivations.
//!
//! Everything else (atom count, open-valence sum, evidence count, the
//! incomplete-support flag) is computed on the device from `actions`,
//! `scores` and `evidence`.
//!
//! Kernel-expressible form (the twin was written first, in the same subset as
//! [`crate::models::ms2::pack`]: `u32` loop counters from literals, `while`
//! loops, full buffers with explicit bases/strides, one exit per lane, no
//! `wrapping_*`, no fixed-size local arrays): [`feature_lane`] is copied line
//! for line by the `#[cube]` kernel, with the 8 feature scalars written out
//! one by one on both sides.

use std::path::Path;

use cubecl::prelude::Runtime;
use serde::{Deserialize, Serialize};

use crate::autograd::Var;
use crate::backend::{Device, FloatElem};
use crate::error::{Error, Result};
use crate::nn::linear::{Linear, LinearConfig};
use crate::nn::module::{Module, ModuleVisitor, StateDict};
use crate::tensor::Tensor;
use crate::tensor::ops::random::Rng;
use crate::train::optim::{AdamW, AdamWConfig, Optimizer};

use super::contain::Containment;
use super::contract::{CandidateBatch, candidate_status};
use super::identity::DUPLICATE_GRAPH;
use super::pack::SCORE_FINITE_MAX;

// ---------------------------------------------------------------------------
// Feature layout
// ---------------------------------------------------------------------------

/// Feature 0: the trace log-probability (`scores[.., 0]`).
pub const F_TRACE_LP: usize = 0;
/// Feature 1: the formula log-probability (`scores[.., 1]`).
pub const F_FORMULA_LP: usize = 1;
/// Feature 2: atom count over `A` (ADD_ATOM tokens below `length`).
pub const F_ATOM_FRAC: usize = 2;
/// Feature 3: open-valence sum over `2A` (the `atoms_cap` record words).
pub const F_OPEN_FRAC: usize = 3;
/// Feature 4: retained evidence count over `E` ([`EVIDENCE_CAP`]).
///
/// The evidence buffer's count word reports the TOTAL number of qualifying
/// peaks (unbounded), while architecture §2.4 retains at most `E = 4`
/// evidence records: the feature uses `min(count, E) / E`, so six qualifying
/// peaks give 1.0, never 1.5.
pub const F_EVID_COUNT: usize = 4;
/// Feature 5: largest evidence log-probability (`0` when there is no
/// evidence; from `evidence_f[.., 0]`).
pub const F_EVID_MAX_LP: usize = 5;
/// Feature 6: smallest absolute evidence residual in units of the fragment
/// tolerance (`1` when there is no evidence; from `evidence_f[.., 1]`).
pub const F_EVID_MIN_RESID: usize = 6;
/// Feature 7: the `evidence_support_incomplete` flag (bit 7 of the evidence
/// status word).
pub const F_EVID_INCOMPLETE: usize = 7;
/// Features per candidate.
pub const N_FEATURES: usize = 8;
/// Evidence capacity `E` of architecture §2.4: feature 4 divides by this.
pub const EVIDENCE_CAP: f32 = 4.0;
/// Words per `evidence` row; words 0 (status) and 1 (count) are read here.
pub const EVIDENCE_STRIDE: usize = 18;
/// Bit 7 of the evidence status: the assignment support behind it is
/// incomplete (architecture §2.4).
pub const EVIDENCE_SUPPORT_INCOMPLETE: u32 = 1 << 7;
/// Token kind of an atom addition (grammar `ADD_ATOM`); spelled as a literal
/// here and in the kernel, as `ms2_identity` does.
pub const ADD_ATOM_KIND: u32 = 2;

// ---------------------------------------------------------------------------
// Host twin (copied line for line by `ops::ms2_rerank`)
// ---------------------------------------------------------------------------

/// Numerically stable binary cross-entropy with logits, per element:
/// `softplus(x) − x·z`.
///
/// Shared by the reranker and the fingerprint head (which reaches it as
/// `super::rerank::bce_with_logits`, so there is exactly one implementation).
/// The derivative is `sigmoid(x) − z` EVERYWHERE, including `x = 0`: the
/// fused [`Var::softplus`] adjoint is `grad * sigmoid(x)`, exact at zero
/// (`1 / (1 + exp(0)) = 0.5`), unlike the old `maximum(x, 0) − x·z +
/// log(1 + exp(−|x|))` composition whose `maximum`/`abs` routing gives `−z`
/// at exactly zero logits (where an untrained head with zero bias sits).
/// Finite for extreme logits (±80 and beyond: `softplus(80) ≈ 80`,
/// `softplus(−80) ≈ 0`).
pub fn bce_with_logits<R: Runtime, E: FloatElem>(
    logits: &Var<R, E>,
    targets: &Var<R, E>,
) -> Result<Var<R, E>> {
    logits.softplus()?.sub(&logits.mul(targets)?)
}

/// One guarded full-buffer read: `buf[base + idx]` when it exists, else `0`.
fn slot(buf: &[u32], base: u32, idx: u32) -> u32 {
    let addr = base + idx;
    let mut out = 0u32;
    if (addr as usize) < buf.len() {
        out = buf[addr as usize];
    }
    out
}

/// One guarded float read: `buf[base + idx]` when it exists, else `0.0`.
fn fslot(buf: &[f32], base: u32, idx: u32) -> f32 {
    let addr = (base + idx) as usize;
    let mut out = 0.0f32;
    if addr < buf.len() {
        out = buf[addr];
    }
    out
}

/// One guarded float write.
fn fput(buf: &mut [f32], base: u32, idx: u32, value: f32) {
    let addr = (base + idx) as usize;
    if addr < buf.len() {
        buf[addr] = value;
    }
}

/// One guarded status-word write.
fn oput(buf: &mut [u32], idx: u32, value: u32) {
    if (idx as usize) < buf.len() {
        buf[idx as usize] = value;
    }
}

/// Whether a log-probability is finite: inside ±[`SCORE_FINITE_MAX`]. A range
/// test, never a NaN self-comparison (fast-math backends fold those to false,
/// so NaN would pass).
fn finite_lane(x: f32) -> u32 {
    let mut ok = 0u32;
    if x > -SCORE_FINITE_MAX && x < SCORE_FINITE_MAX {
        ok = 1;
    }
    ok
}

/// Per-trajectory feature lane: the twin of `ms2_rerank_features`'s lane.
///
/// `actions` holds device trajectory records of `record_stride` words
/// (`steps * 4` token words, `atoms_cap` open-valence words, then length,
/// status and two spare words); `scores` is `[rows, 2]` flat
/// `(trace_log_prob, formula_log_prob)`; `evidence` is `[rows, 18]` flat
/// (word 0 status, word 1 count); `evidence_f` is `[rows, 2]` flat (largest
/// evidence log-probability, smallest |residual| in tolerance units).
/// Writes the 8 features at `record * 8` and the status word (`1` finite,
/// `0` the score was non-finite and the row is all zeros).
///
/// Finite by construction: a non-finite input score zeroes the row; with no
/// evidence (`count == 0`) the incomplete flag is 0 and features 5 and 6 take
/// the spec defaults `(0, 1)` whatever `evidence_f` holds; a non-finite
/// `evidence_f` entry with evidence present falls back to the same defaults.
/// Feature 4 uses the RETAINED evidence count `min(count, E) / E`: the buffer's
/// count word is the total number of qualifying peaks (unbounded), while §2.4
/// retains at most `E = 4` records.
#[allow(clippy::too_many_arguments)]
pub fn feature_lane(
    actions: &[u32],
    record_stride: u32,
    steps: u32,
    atoms_cap: u32,
    scores: &[f32],
    evidence: &[u32],
    evidence_f: &[f32],
    record: u32,
    features: &mut [f32],
    ok: &mut [u32],
) {
    let abase = record * record_stride;
    let len_field = steps * 4 + atoms_cap;
    let length = slot(actions, abase, len_field);
    let trace_lp = fslot(scores, record * 2, 0);
    let formula_lp = fslot(scores, record * 2, 1);
    let mut good = 0u32;
    if finite_lane(trace_lp) == 1 && finite_lane(formula_lp) == 1 {
        good = 1;
    }
    let mut f0 = 0.0f32;
    let mut f1 = 0.0f32;
    let mut f2 = 0.0f32;
    let mut f3 = 0.0f32;
    let mut f4 = 0.0f32;
    let mut f5 = 0.0f32;
    let mut f6 = 0.0f32;
    let mut f7 = 0.0f32;
    if good == 1 {
        let mut atoms = 0u32;
        let mut s = 0u32;
        while s < steps {
            let kind = slot(actions, abase, s * 4);
            if s < length && kind == ADD_ATOM_KIND {
                atoms += 1;
            }
            s += 1;
        }
        let mut open = 0u32;
        let mut a = 0u32;
        while a < atoms_cap {
            open += slot(actions, abase, steps * 4 + a);
            a += 1;
        }
        let ev_status = slot(evidence, record * EVIDENCE_STRIDE as u32, 0);
        let ev_count = slot(evidence, record * EVIDENCE_STRIDE as u32, 1);
        let mut max_lp = fslot(evidence_f, record * 2, 0);
        let mut min_resid = fslot(evidence_f, record * 2, 1);
        if ev_count == 0 {
            max_lp = 0.0;
            min_resid = 1.0;
        } else {
            if finite_lane(max_lp) == 0 {
                max_lp = 0.0;
            }
            if finite_lane(min_resid) == 0 {
                min_resid = 1.0;
            }
        }
        let mut incomplete = 0.0f32;
        if ev_count != 0 && ev_status & EVIDENCE_SUPPORT_INCOMPLETE != 0 {
            incomplete = 1.0;
        }
        f0 = trace_lp;
        f1 = formula_lp;
        f2 = atoms as f32 / atoms_cap as f32;
        f3 = open as f32 / (2 * atoms_cap) as f32;
        let mut retained = ev_count;
        if retained > EVIDENCE_CAP as u32 {
            retained = EVIDENCE_CAP as u32;
        }
        f4 = retained as f32 / EVIDENCE_CAP;
        f5 = max_lp;
        f6 = min_resid;
        f7 = incomplete;
    }
    let out_base = record * N_FEATURES as u32;
    fput(features, out_base, 0, f0);
    fput(features, out_base, 1, f1);
    fput(features, out_base, 2, f2);
    fput(features, out_base, 3, f3);
    fput(features, out_base, 4, f4);
    fput(features, out_base, 5, f5);
    fput(features, out_base, 6, f6);
    fput(features, out_base, 7, f7);
    oput(ok, record, good);
}

/// Host features of one `(B, K)` bucket: the twin of `ms2_rerank_features`
/// over already-read buffers. Used by the kernel tests as the expected value;
/// the integration task calls the kernel instead.
///
/// `actions` is `[rows, steps * 4 + atoms + 4]`, `scores` is `[rows, 2]`,
/// `evidence` is `[rows, 18]`, `evidence_f` is `[rows, 2]`. Returns
/// `(features [rows, 8], feature_ok [rows])`. Shape violations are
/// [`Error::Shape`].
#[allow(clippy::too_many_arguments)]
pub fn compute_features(
    actions: &[u32],
    steps: usize,
    atoms_cap: u32,
    scores: &[f32],
    evidence: &[u32],
    evidence_f: &[f32],
) -> Result<(Vec<f32>, Vec<u32>)> {
    if atoms_cap == 0 || atoms_cap > 32 {
        return Err(Error::shape(format!(
            "compute_features needs 1 <= atoms_cap <= 32, got {atoms_cap}"
        )));
    }
    if steps == 0 {
        return Err(Error::shape(
            "compute_features needs steps >= 1".to_string(),
        ));
    }
    let stride = steps
        .checked_mul(4)
        .and_then(|v| v.checked_add(atoms_cap as usize))
        .and_then(|v| v.checked_add(4))
        .ok_or_else(|| Error::shape("compute_features: steps * 4 + atoms + 4 overflows usize".to_string()))?;
    // `steps >= 1`, so `stride >= 9`: the division below cannot divide by
    // zero, and the product check rejects ragged tails.
    let rows = actions.len() / stride;
    if actions.len() != rows * stride {
        return Err(Error::shape(format!(
            "compute_features needs actions [rows, {stride}] (steps {steps}, atoms {atoms_cap}), got length {}",
            actions.len()
        )));
    }
    if scores.len() != rows * 2 {
        return Err(Error::shape(format!(
            "compute_features needs scores [{rows}, 2], got length {}",
            scores.len()
        )));
    }
    if evidence.len() != rows * EVIDENCE_STRIDE {
        return Err(Error::shape(format!(
            "compute_features needs evidence [{rows}, {EVIDENCE_STRIDE}], got length {}",
            evidence.len()
        )));
    }
    if evidence_f.len() != rows * 2 {
        return Err(Error::shape(format!(
            "compute_features needs evidence_f [{rows}, 2], got length {}",
            evidence_f.len()
        )));
    }
    let mut features = vec![0.0f32; rows * N_FEATURES];
    let mut ok = vec![0u32; rows];
    let mut r = 0u32;
    while (r as usize) < rows {
        feature_lane(
            actions,
            stride as u32,
            steps as u32,
            atoms_cap,
            scores,
            evidence,
            evidence_f,
            r,
            &mut features,
            &mut ok,
        );
        r += 1;
    }
    Ok((features, ok))
}

// ---------------------------------------------------------------------------
// Reranker model
// ---------------------------------------------------------------------------

/// Hidden width of the reranker (architecture §4.3: `Linear(8 → 16)`, SiLU,
/// `Linear(16 → 1)`).
pub const RERANKER_HIDDEN: usize = 16;

/// The reranker of architecture §4.3: a logit of containment in the true
/// parent from the 8 features above.
pub struct Reranker<R: Runtime, E: FloatElem> {
    /// `Linear(8 → 16)`.
    hidden: Linear<R, E>,
    /// `Linear(16 → 1)`.
    out: Linear<R, E>,
}

impl<R: Runtime, E: FloatElem> Reranker<R, E> {
    /// Build the reranker on `device`, weights from `rng`.
    pub fn init(device: &Device<R>, rng: &mut Rng) -> Self {
        Self {
            hidden: LinearConfig::new(N_FEATURES, RERANKER_HIDDEN).init(device, rng),
            out: LinearConfig::new(RERANKER_HIDDEN, 1).init(device, rng),
        }
    }

    /// Containment logits of `features` (`[rows, 8]`), one per row (`[rows]`).
    /// No device read.
    pub fn logits(&self, features: &Tensor<R, E>) -> Result<Var<R, E>> {
        if features.rank() != 2 || features.shape().dim(1) != N_FEATURES {
            return Err(Error::shape(format!(
                "Reranker::logits needs features [rows, {N_FEATURES}], got {}",
                features.shape()
            )));
        }
        let rows = features.shape().dim(0);
        let x = Var::constant(features.clone());
        let h = self.hidden.apply(&x)?.silu()?;
        let y = self.out.apply(&h)?;
        y.reshape(vec![rows])
    }

    /// Weighted binary cross-entropy with logits via the shared
    /// [`bce_with_logits`] (`softplus(x) − x·z`, whose derivative is
    /// `sigmoid(x) − z` everywhere including zero), summed with `weights` and
    /// divided by the weight sum (0 when the weight sum is 0, via
    /// `max(sum, 1)`: the numerator is 0 then, with no device read).
    ///
    /// `features` is `[rows, 8]`, `labels` and `weights` are `[rows]` floats
    /// (`0`/`1` by contract; values are not re-checked here). No device read.
    pub fn loss(
        &self,
        features: &Tensor<R, E>,
        labels: &Tensor<R, E>,
        weights: &Tensor<R, E>,
    ) -> Result<Var<R, E>> {
        if features.rank() != 2 || features.shape().dim(1) != N_FEATURES {
            return Err(Error::shape(format!(
                "Reranker::loss needs features [rows, {N_FEATURES}], got {}",
                features.shape()
            )));
        }
        let rows = features.shape().dim(0);
        if labels.rank() != 1 || labels.len() != rows {
            return Err(Error::shape(format!(
                "Reranker::loss needs labels [{rows}], got {}",
                labels.shape()
            )));
        }
        if weights.rank() != 1 || weights.len() != rows {
            return Err(Error::shape(format!(
                "Reranker::loss needs weights [{rows}], got {}",
                weights.shape()
            )));
        }
        if rows == 0 {
            return Ok(Var::constant(Tensor::full(
                Vec::<usize>::new(),
                0.0,
                features.device(),
            )));
        }
        let x = self.logits(features)?;
        let z = Var::constant(labels.clone());
        let w = Var::constant(weights.clone());
        let per = bce_with_logits(&x, &z)?;
        let num = per.mul(&w)?.sum()?;
        let den = w.sum()?;
        let one = Var::constant(Tensor::full(Vec::<usize>::new(), 1.0, features.device()));
        let denom = den.maximum(&one)?;
        num.div(&denom)
    }
}

impl<R: Runtime, E: FloatElem> Module<R, E> for Reranker<R, E> {
    fn visit(&self, visitor: &mut ModuleVisitor<'_, R, E>) {
        visitor.child("hidden", &self.hidden);
        visitor.child("out", &self.out);
    }
}

// ---------------------------------------------------------------------------
// Trainer
// ---------------------------------------------------------------------------

/// AdamW trainer for the reranker: labels are produced on the host by the
/// caller (containment, resolved only) and passed in; the trainer takes them
/// as inputs alongside the features.
pub struct RerankTrainer<R: Runtime, E: FloatElem> {
    /// AdamW with no weight decay (a decayed step on an all-zero-weight batch
    /// would move the parameters, breaking the zero-loss/no-change rule).
    optim: AdamW<R, E>,
    /// Loss is downloaded only every `report_every` steps.
    report_every: usize,
    /// Updates applied so far.
    steps: u64,
}

impl<R: Runtime, E: FloatElem> RerankTrainer<R, E> {
    /// Build the trainer: AdamW at `learning_rate` (defaults otherwise, weight
    /// decay forced to 0), reporting every `report_every` steps (at least 1).
    pub fn new(learning_rate: f32, report_every: usize) -> Self {
        let config = AdamWConfig::builder()
            .learning_rate(learning_rate)
            .weight_decay(0.0)
            .build();
        Self {
            optim: config.init(),
            report_every: report_every.max(1),
            steps: 0,
        }
    }

    /// One gradient step on this batch. `eligible` is the host-known count of
    /// eligible examples in the batch (from [`eligible_examples`]); when it is
    /// 0 the batch carries no learning signal, so the call returns `Ok(None)`
    /// without touching parameters, without advancing Adam's moments or the
    /// step counter, and without any device read. (Adam retains momentum after
    /// any nonzero-gradient step, so running the optimizer on an all-zero-weight
    /// batch would still move the parameters.)
    ///
    /// Otherwise returns the loss only at report boundaries
    /// (`steps % report_every == 0`); otherwise `None`, with no device read
    /// (the loss tensor is built and differentiated but never downloaded).
    pub fn step(
        &mut self,
        model: &Reranker<R, E>,
        features: &Tensor<R, E>,
        labels: &Tensor<R, E>,
        weights: &Tensor<R, E>,
        eligible: usize,
    ) -> Result<Option<f32>> {
        if eligible == 0 {
            return Ok(None);
        }
        let loss = model.loss(features, labels, weights)?;
        let grads = loss.backward()?;
        self.optim.step(&model.parameters(), &grads)?;
        self.steps += 1;
        if self.steps.is_multiple_of(self.report_every as u64) {
            Ok(Some(loss.try_to_f32()?[0]))
        } else {
            Ok(None)
        }
    }

    /// The optimizer's per-parameter moments for `model`'s parameters, for
    /// tests that assert a zero-eligible batch leaves optimizer state
    /// bit-identical.
    pub fn optimizer_state_dict(&self, model: &Reranker<R, E>) -> StateDict {
        self.optim.state_dict(&model.named_parameters())
    }

    /// The optimizer's internal step counter (Adam's bias correction clock):
    /// unchanged by zero-eligible batches.
    pub fn optimizer_steps(&self) -> u64 {
        self.optim.step_count()
    }

    /// Updates applied so far.
    pub fn steps(&self) -> u64 {
        self.steps
    }
}

// ---------------------------------------------------------------------------
// Training examples
// ---------------------------------------------------------------------------

/// How many candidates [`eligible_examples`] excluded, by reason.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ExcludedCounts {
    /// Not `FINISHED`.
    pub not_finished: usize,
    /// Finished but `INVALID_FINAL`, `TRUNCATED` or `REQUEST_FAILED`.
    pub invalid: usize,
    /// Finished and valid but `DUPLICATE_TRACE` or `DUPLICATE_GRAPH` (an
    /// unresolved identity, bit 8, stays eligible: it is not a duplicate
    /// flag).
    pub duplicate: usize,
    /// Finished, valid, non-duplicate, but containment hit its work limit:
    /// excluded and counted, never used as negatives.
    pub work_limit: usize,
}

/// Training examples of architecture §4.3: finished, valid, non-duplicate
/// candidates with a resolved containment label.
///
/// Returns the record indices, the `0`/`1` labels (`1` for
/// [`Containment::Contained`]) and the exclusion counts. `containment` holds
/// one [`Containment`] per record of `batch` (a length mismatch is
/// [`Error::Shape`).
pub fn eligible_examples(
    batch: &CandidateBatch,
    containment: &[Containment],
) -> Result<(Vec<usize>, Vec<f32>, ExcludedCounts)> {
    let n = batch.batch * batch.trajectories;
    if containment.len() != n {
        return Err(Error::shape(format!(
            "eligible_examples needs one containment outcome per record ({n}), got {}",
            containment.len()
        )));
    }
    if batch.status.len() != n {
        return Err(Error::shape(format!(
            "eligible_examples needs one status per record ({n}), got {}",
            batch.status.len()
        )));
    }
    let mut indices = Vec::new();
    let mut labels = Vec::new();
    let mut excluded = ExcludedCounts::default();
    for (r, (&st, &outcome)) in batch.status.iter().zip(containment.iter()).enumerate() {
        if st & candidate_status::FINISHED == 0 {
            excluded.not_finished += 1;
            continue;
        }
        if st & candidate_status::INVALID_FINAL != 0
            || st & candidate_status::TRUNCATED != 0
            || st & candidate_status::REQUEST_FAILED != 0
        {
            excluded.invalid += 1;
            continue;
        }
        if st & candidate_status::DUPLICATE_TRACE != 0 || st & DUPLICATE_GRAPH != 0 {
            excluded.duplicate += 1;
            continue;
        }
        match outcome {
            Containment::Contained => {
                indices.push(r);
                labels.push(1.0);
            }
            Containment::NotContained => {
                indices.push(r);
                labels.push(0.0);
            }
            Containment::WorkLimit => {
                excluded.work_limit += 1;
            }
        }
    }
    Ok((indices, labels, excluded))
}

// ---------------------------------------------------------------------------
// Checkpoint
// ---------------------------------------------------------------------------

/// Version string stored with every reranker artifact.
pub const RERANKER_VERSION: &str = "ms2-reranker-v1";

/// A saved [`Reranker`]: the version string, the weights, and the
/// generation/dedup configuration JSON the reranker was trained under
/// (architecture §4.3: stored with the artifact), in the same JSON style as
/// `Ms2Trainer::save` (parameter paths via [`StateDict`]).
#[derive(Serialize, Deserialize)]
struct RerankerCheckpoint {
    /// Artifact version ([`RERANKER_VERSION`]).
    version: String,
    /// Human-readable note.
    note: String,
    /// The generation, deduplication and sampling configuration, as supplied
    /// by the caller.
    gen_config: serde_json::Value,
    /// Weights by parameter path.
    weights: StateDict,
}

impl<R: Runtime, E: FloatElem> Reranker<R, E> {
    /// Save the parameters with [`RERANKER_VERSION`] and `gen_config` (the
    /// generation/dedup configuration this reranker was trained under).
    pub fn save(&self, path: &Path, gen_config: &serde_json::Value) -> Result<()> {
        let checkpoint = RerankerCheckpoint {
            version: RERANKER_VERSION.to_string(),
            note: "MS2 reranker artifact: weights plus the generation/dedup configuration they were trained under."
                .to_string(),
            gen_config: gen_config.clone(),
            weights: self.state_dict(),
        };
        let text = serde_json::to_string_pretty(&checkpoint)?;
        std::fs::write(path, text)?;
        Ok(())
    }

    /// Load a checkpoint saved by [`Reranker::save`] onto `device`, returning
    /// the model and its stored configuration JSON. A version mismatch is
    /// [`Error::Config`]; weights load strictly.
    pub fn load(path: &Path, device: &Device<R>) -> Result<(Self, serde_json::Value)> {
        let text = std::fs::read_to_string(path)?;
        let checkpoint: RerankerCheckpoint = serde_json::from_str(&text)?;
        if checkpoint.version != RERANKER_VERSION {
            return Err(Error::config(format!(
                "Reranker::load: unknown version {} (expected {RERANKER_VERSION})",
                checkpoint.version
            )));
        }
        // Placeholders, immediately overwritten by the strict load below.
        let mut rng = Rng::seeded(0);
        let model = Self::init(device, &mut rng);
        model.load_state_dict(&checkpoint.weights, true)?;
        Ok((model, checkpoint.gen_config))
    }
}
