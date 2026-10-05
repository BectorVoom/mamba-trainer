//! The V0 training and evaluation driver (architecture §§4.4 and 7).
//!
//! [`Ms2Trainer`] composes the encoder, formula head and decoder exactly as
//! the `overfit_smoke` test assembles a training step, with AdamW and reads
//! only at reporting boundaries: [`Ms2Trainer::step`] performs no device read
//! unless [`Ms2Trainer::request_report`] was called, in which case it performs
//! exactly one batched read of `[L, L_graph, L_formula]`.
//!
//! The graph loss conditions on the true parent formula (contracts §7.2): the
//! decoder input is the gold row's formula embedding (the oracle form), and a
//! spectrum whose gold formula is absent from the scored window conditions on
//! the zero vector and contributes 0 to `L_formula` (counted in
//! [`LossReport`]).

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use cubecl::prelude::Runtime;
use serde::{Deserialize, Serialize};

use crate::autograd::{Grads, Var, cat, no_grad};
use crate::backend::{Device, FloatElem};
use crate::error::{Error, Result};
use crate::nn::Module;
use crate::nn::module::StateDict;
use crate::nn::param::Param;
use crate::tensor::Tensor;
use crate::tensor::ops::index::{IdTensor, ids_to_float, read_all};
use crate::tensor::ops::ms2::{
    self, FormulaBuffers, Ms2Constants, PeakBuffers, ReplayBuffers, safe_ids,
};
use crate::tensor::ops::random::Rng;
use crate::train::optim::{AdamW, AdamWConfig, GradScale, Optimizer, grad_scale};

use super::batch::DeviceSpectra;
use super::chem::Composition;
use super::contract::{Control, FormulaFeatures, GenerationConfig, ModelConfig, SpectrumBatch};
use super::decoder::{ReplayView, TeacherOutput, graph_loss};
use super::encoder::EncoderOutput;
use super::enum_cache::{EnumCache, EnumCacheHeader, run_device_enumeration_into};
use super::experiment::{
    ExperimentSet, apply_precursor_jitter, donor_stats, jitter_variant_index,
    spectrum_batch_for, spectrum_batch_with_donors, target_batch_for,
};
use super::formula_evidence_ref::jitter_precursor_mz;
use super::formula::FormulaTable;
use super::formula_head::{DeviceFormulaTable, FormulaOutput};
use super::generate::{GENERATION_WINDOW_M, GenerationWorkspace, Ms2Model};
use super::grammar::Limits;
use super::metrics::{SpectrumEval, evaluate_candidates};
use super::targets_batch::TargetBatch;
use super::workspace::Ms2Capabilities;

/// Device window width `M` of a training batch (architecture §2).
pub const TRAIN_WINDOW_M: usize = GENERATION_WINDOW_M;

/// Formula rows scored per training batch: unbounded on the host side, so the
/// device window width `M` is the only cap.
pub const TRAIN_ROWS_SCORED_MAX: u32 = 4096;

/// Containment work limit of [`Ms2Trainer::generate_eval`].
pub const TRAIN_EVAL_WORK_LIMIT: usize = 1_000_000;

/// How the decoder conditions on the true parent formula during training
/// (V1 §1.2, teacher forcing).
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
pub enum GoldFormulaConditioning {
    /// Condition on the head embedding of the true parent composition
    /// (`gold_counts` through `count_features` and the head's row network),
    /// whether or not the search scored it.
    Composition,
    /// V0 behaviour: condition on the scored row's embedding when the gold
    /// is in the window, else the zero vector.
    ScoredRowOrZero,
}

fn default_gold_conditioning() -> GoldFormulaConditioning {
    GoldFormulaConditioning::ScoredRowOrZero
}

/// What [`Ms2Trainer::conditioning_for_test`] hands back: the actual trainer
/// conditioning embedding, the full scored embedding and the device gold
/// slots (see that method; test support only).
#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct TrainerConditioning {
    /// Spectra in the batch.
    pub spectra: usize,
    /// Residual width.
    pub d_model: usize,
    /// Device gold slots (`u32::MAX` when absent).
    pub slots: Vec<u32>,
    /// Conditioning embedding (`[B, d]` flat) the decoder actually receives.
    pub e_cond: Vec<f32>,
    /// Scored row embeddings (`[B, M, d]` flat, `M = window_m`).
    pub scored_embedding: Vec<f32>,
    /// Scored-candidate capacity of this call.
    pub window_m: usize,
}

/// Teacher-forcing conditioning embedding shared by `forward` and the test
/// hook: one code path, so the test exercises the actual trainer
/// conditioning rather than a reconstruction.
///
/// `Composition` embeds the gold counts through `count_features` and the
/// head's row network (`[B, 10]` → `[B, d]` with the same weights and op
/// order as the scored `[B, M, 10]` path, so the same features give the same
/// bits); `ScoredRowOrZero` gathers the scored row's embedding and zeroes
/// absent rows on the device, launching no gold-network kernel. A free
/// function (not a method) so `forward` can hold its buffer borrow while
/// using only the disjoint model/table/device fields.
///
/// Bit-equality note: the `[B, 10]` and `[B, M, 10]` `embed_rows` matmuls
/// take the same tuned path on the supported backends (pinned by
/// `trainer_conditioning_bit_equality_at_m32_and_m128` on CPU; the GPU run
/// pins it there). If a backend ever tunes them differently, this gold path
/// must be reworked to run in the same shape class — e.g. by scoring the
/// gold as one extra row of the same batched tensor — and documented here,
/// never by loosening that test.
///
/// Launch note: the `Composition` branch runs under the `ms2.gold_embed`
/// tally scope, so a launch test can prove structurally — on every backend —
/// that `ScoredRowOrZero` launches zero gold-network kernels (that branch
/// never enters this scope) while `Composition` launches more than zero.
#[allow(clippy::too_many_arguments)]
fn condition_embeddings<R: Runtime, E: FloatElem>(
    head: &super::formula_head::FormulaHead<R, E>,
    log_table: &Tensor<R, E>,
    device: &Device<R>,
    mode: GoldFormulaConditioning,
    scored: &super::formula_head::FormulaOutput<R, E>,
    gold_counts_t: &IdTensor<R>,
    gold_slot_t: &IdTensor<R>,
    batch: usize,
    d: usize,
) -> Result<Var<R, E>> {
    match mode {
        GoldFormulaConditioning::Composition => {
            let _tally = crate::backend::tally_scope("ms2.gold_embed");
            let gold_feat_t = Tensor::empty(vec![batch, 10], device);
            // `count_features` needs `&mut` out; allocate then fill.
            let mut gold_feat_buf = gold_feat_t;
            ms2::count_features(gold_counts_t, log_table, &mut gold_feat_buf, 10)?;
            let gold_feat_var = Var::constant(gold_feat_buf).reshape(vec![batch, 10])?;
            head.embed_rows(&gold_feat_var)?.reshape(vec![batch, d])
        }
        GoldFormulaConditioning::ScoredRowOrZero => {
            let safe = safe_ids(gold_slot_t, 0)?;
            let e_gold =
                Var::gather_tokens(&scored.embedding, &safe, 1)?.reshape(vec![batch, d])?;
            // Validity without a read: `u32::MAX` casts to 2^32 in `f32`,
            // every real slot well below it (the same device-side rule as
            // `FormulaHead::loss`).
            let gold_f: Tensor<R, E> = ids_to_float(gold_slot_t);
            let is_absent = crate::tensor::ops::elemwise::eq_scalar(&gold_f, u32::MAX as f32);
            let valid_mask = crate::tensor::ops::elemwise::rsub_scalar(&is_absent, 1.0);
            let valid_var = Var::constant(valid_mask)
                .reshape(vec![batch, 1])?
                .expand(vec![batch, d])?;
            e_gold.mul(&valid_var)
        }
    }
}

/// Assignment half of the training forward prefix: `ion_assign` on the gold
/// composition (`F = 1`), host label sets, `ion_label_mask`, the head's
/// `log_prob` and `loss`. Returns `(None, None, 0)` when disabled
/// (`lambda_assign <= 0` or no head): zero launches, zero reads. The third
/// element is `assignment_label_overflow`: labels beyond `L` counted on the
/// host at upload (no read).
///
/// No device read. The spectrum's m/z uncertainty reaches the device in the
/// `spec [B, 2]` buffer the kernel expects. A free function (not a method)
/// so the bucket borrow never overlaps a whole-`self` borrow.
#[allow(clippy::too_many_arguments)]
fn forward_assign<R: Runtime, E: FloatElem>(
    model: &super::generate::Ms2Model<R, E>,
    train: &TrainConfig,
    device: &Device<R>,
    log_table: &Tensor<R, E>,
    set: &ExperimentSet,
    indices: &[usize],
    prep: &Prepared,
    spectra: &DeviceSpectra<R, E>,
    bucket: &mut TrainBucket<R, E>,
    encoded: &EncoderOutput<R, E>,
    b: usize,
) -> Result<(Option<Var<R, E>>, Option<Tensor<R, E>>, usize)> {
    let Some(head) = model.assignment.as_ref() else {
        return Ok((None, None, 0));
    };
    let Some(cfg) = model.config.assignment.as_ref() else {
        return Ok((None, None, 0));
    };
    if !(train.lambda_assign > 0.0) {
        return Ok((None, None, 0));
    }
    let j = cfg.hypotheses as usize;
    let work_max = cfg.work_max;
    let l_cap = cfg.labels as usize;
    let n = model.config.n_peaks as usize;
    let work = (b as u64)
        .checked_mul(n as u64)
        .and_then(|v| v.checked_mul(u64::from(work_max)))
        .ok_or_else(|| {
            Error::config(
                "forward_assign: B * F * N * work_max overflows u64".to_string(),
            )
        })?;
    if work > u64::from(train.ion_request_work_max) {
        return Err(Error::config(format!(
            "forward_assign: B * F * N * ion_work_max {work} exceeds ion_request_work_max {} (refused before dispatch)",
            train.ion_request_work_max
        )));
    }
    // The m/z uncertainty travels with the uploaded peaks (donor assembly
    // included): one shared source with the formula-evidence stage.
    let spec_t = spectra.evidence_spec(device)?;
    let top_counts_t = IdTensor::from_slice(&prep.gold_counts, vec![b, 1, 10], device)?;
    let mut ion_t = IdTensor::empty(vec![b, 1, n, j, 12], device);
    let mut ion_meta_t = IdTensor::empty(vec![b, 1, n, 4], device);
    crate::tensor::ops::ms2_ion::ion_assign(
        &top_counts_t,
        &bucket.peaks.kept,
        &spectra.meta,
        &spec_t,
        &mut ion_t,
        &mut ion_meta_t,
        work_max,
    )?;
    let n_raw = prep.n_raw;
    let mut lab_host = vec![0u32; b * l_cap * 12];
    let mut overflow = 0usize;
    for (bi, &idx) in indices.iter().enumerate() {
        let Some(lbls) = set.spectra.get(idx).and_then(|s| s.labels.as_ref()) else {
            continue;
        };
        let adduct_id = prep.spectra.adduct[bi];
        let base = bi * n_raw;
        let count = (prep.spectra.peak_count[bi] as usize).min(n_raw);
        let raw_of = |pid: u32| -> Option<u32> {
            for k in 0..count {
                if prep.spectra.peak_id[base + k] == pid {
                    return Some(k as u32);
                }
            }
            None
        };
        let sets = super::ion::ion_labels(lbls, adduct_id, raw_of, l_cap);
        overflow += sets.overflow;
        for (li, lab) in sets.labels.iter().enumerate() {
            let lb = (bi * l_cap + li) * 12;
            lab_host[lb] = lab.raw_index;
            for e in 0..10 {
                lab_host[lb + 1 + e] = u32::from(lab.counts[e]);
            }
            lab_host[lb + 11] = 1;
        }
    }
    let lab_t = IdTensor::from_slice(&lab_host, vec![b, l_cap, 12], device)?;
    let mut mask_t = Tensor::empty(vec![b, n, j + 1], device);
    let mut state_t = IdTensor::empty(vec![b, n], device);
    crate::tensor::ops::ms2_ion::ion_label_mask(
        &lab_t,
        &ion_t,
        &ion_meta_t,
        &bucket.peaks.kept,
        &mut mask_t,
        &mut state_t,
        0,
    )?;
    let out = head.log_prob(&model.formula, log_table, &ion_t, &ion_meta_t, &encoded.x)?;
    let al = head.loss(&out, &mask_t, &state_t)?;
    Ok((Some(al.loss), Some(al.counts), overflow))
}

/// Training hyperparameters of [`Ms2Trainer`].
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct TrainConfig {
    /// Spectra per optimizer step.
    pub batch: usize,
    /// Target slots per spectrum (16 in V0: the recipe keeps at most 16).
    pub slots: usize,
    /// AdamW base learning rate.
    pub lr: f32,
    /// AdamW decoupled weight decay.
    pub weight_decay: f32,
    /// Formula-loss weight (`L = L_graph + formula_weight * L_formula`).
    pub formula_weight: f32,
    /// Seed for weight initialisation and (in the driver) epoch shuffles.
    pub seed: u64,
    /// The control the model trains under (and is evaluated under).
    pub control: Control,
    /// Global gradient-norm clip (`None` disables it). Applied with the
    /// device-side scale, so clipping costs no device read.
    pub grad_clip: Option<f32>,
    /// How the decoder conditions on the true parent formula (V1 §1.2).
    /// A serialized config without the field means `ScoredRowOrZero`, so a
    /// V0 run is reproduced; new runs choose `Composition` explicitly (the
    /// experiment driver's default). The Rust `Default` stays V0-compatible
    /// (`ScoredRowOrZero`).
    #[serde(default = "default_gold_conditioning")]
    pub gold_formula_conditioning: GoldFormulaConditioning,
    /// Formula candidate source (V1 §1.4). A serialized config without the
    /// field means `Table`, so a V0 run is reproduced.
    #[serde(default = "default_train_formula_source")]
    pub formula_source: super::contract::FormulaSource,
    /// Scored-candidate capacity per spectrum (`M`, V1 §1.2): one of 32,
    /// 128, 512, 2048. A serialized config without the field means 32.
    #[serde(default = "default_train_formula_window")]
    pub formula_window: u32,
    /// Submitted lanes (`B * P`) refused before any upload, allocation or
    /// encoder launch when above this (V1 §1.4, `Enumerate` only). A
    /// serialized config without the field means 262,144.
    #[serde(default = "default_train_enum_lanes_max")]
    pub enum_lanes_max: u32,
    /// Per-lane visit budget of the enumerating source (V1 §1.4, `Enumerate`
    /// only). A serialized config without the field means 4,096.
    #[serde(default = "default_train_enum_lane_visits_max")]
    pub enum_lane_visits_max: u32,
    /// Worst-case visits covered by one count or fill launch (V1 §1.4,
    /// `Enumerate` only). A serialized config without the field means
    /// 4,000,000. Must be non-zero.
    #[serde(default = "default_train_enum_dispatch_visits_max")]
    pub enum_dispatch_visits_max: u32,
    /// Enumeration fitting source identity (D6, experiment driver only):
    /// file name, SHA-256 and export subset the EnumDomain/RatioBounds were
    /// fitted on. `None` for table-only or checkpoints predating provenance.
    #[serde(default)]
    pub enum_fit_name: Option<String>,
    #[serde(default)]
    pub enum_fit_sha256: Option<String>,
    #[serde(default)]
    pub enum_fit_subset: Option<String>,
    /// Assignment-loss weight (`L = L_graph + formula_weight * L_formula +
    /// lambda_assign * L_assign`, architecture §2.3). Default 0.0 = off; the
    /// spec value 0.1 is what the experiment driver uses with `--assign`.
    #[serde(default = "default_train_lambda_assign")]
    pub lambda_assign: f32,
    /// Request-level ion work bound of architecture §2.1:
    /// `B * F * N * ion_work_max` above this is refused before dispatch.
    /// Default `2^28` (the `GenerationConfig::ion_request_work_max` default).
    /// Must be non-zero. A serialized config without the field means `2^28`.
    #[serde(default = "default_train_ion_request_work_max")]
    pub ion_request_work_max: u32,
    /// Per-lane visit budget of the evidence walk (architecture §1.6, `W`):
    /// at most this many sub-composition visits per `(b, m)` lane. Default
    /// 2,048. Must be non-zero. A serialized config without the field means
    /// 2,048.
    #[serde(default = "default_train_formula_evidence_work_max")]
    pub formula_evidence_work_max: u32,
    /// Worst-case hydrogen trials covered by one evidence dispatch launch
    /// (architecture §1.6, task E4F item 2). Default `2^28`. Must be non-zero.
    /// A serialized config without the field means `2^28`.
    #[serde(default = "default_train_formula_evidence_dispatch_max")]
    pub formula_evidence_dispatch_max: u64,
    /// Precursor jitter of architecture §1.6: standard deviation `sigma` in
    /// ppm of the per-spectrum precursor-m/z training noise (`e ~ Normal(0,
    /// sigma)`, truncated at `3 sigma`, via
    /// `formula_evidence_ref::jitter_precursor_mz` keyed by
    /// `(seed, 1 + step, spectrum index)`). Default 0.0 (no jitter). Must be
    /// finite and in `[0, 5]`.
    #[serde(default = "default_train_precursor_jitter_ppm")]
    pub precursor_jitter_ppm: f32,
    /// Fixed jitter-draw pool of architecture §1.6 (task T6): `0` means a
    /// fresh draw per `(seed, step, spectrum)` as before (never cached);
    /// `V > 0` means each spectrum has `V` fixed draws (`split_tag = 1 + v`,
    /// `v < V`) and step `s` uses `v = hash(seed, s, spectrum index) mod V`
    /// (see [`jitter_variant_index`](super::experiment::jitter_variant_index)),
    /// so the driver can cache all `V` variants of every training spectrum
    /// up front. With `V > 0` the jitter is drawn from a fixed pool of `V`
    /// draws per spectrum, not a fresh draw per step.
    #[serde(default = "default_train_precursor_jitter_variants")]
    pub precursor_jitter_variants: u32,
}

fn default_train_formula_source() -> super::contract::FormulaSource {
    super::contract::FormulaSource::Table
}

fn default_train_formula_window() -> u32 {
    32
}

fn default_train_enum_lanes_max() -> u32 {
    262_144
}

fn default_train_enum_lane_visits_max() -> u32 {
    4_096
}

fn default_train_enum_dispatch_visits_max() -> u32 {
    4_000_000
}

fn default_train_lambda_assign() -> f32 {
    0.0
}

fn default_train_ion_request_work_max() -> u32 {
    268_435_456
}

/// Default per-lane visit budget of the evidence walk (architecture §1.6).
fn default_train_formula_evidence_work_max() -> u32 {
    2_048
}

/// Default worst-case hydrogen trials per evidence dispatch launch.
fn default_train_formula_evidence_dispatch_max() -> u64 {
    268_435_456
}

/// Default precursor jitter in ppm (no jitter).
fn default_train_precursor_jitter_ppm() -> f32 {
    0.0
}

/// Default fixed jitter-draw pool (0: a fresh draw per step, never cached).
fn default_train_precursor_jitter_variants() -> u32 {
    0
}

impl Default for TrainConfig {
    /// The V0 defaults: batch 16, 16 slots, lr 3e-4, decay 0.1, formula
    /// weight 0.2, seed 1, no control, no clip, `ScoredRowOrZero` (V0
    /// behaviour), `Table` source with `M = 32`, lane limits 262,144 /
    /// 4,096 / 4,000,000.
    fn default() -> Self {
        Self {
            batch: 16,
            slots: 16,
            lr: 3e-4,
            weight_decay: 0.1,
            formula_weight: 0.2,
            seed: 1,
            control: Control::None,
            grad_clip: None,
            gold_formula_conditioning: GoldFormulaConditioning::ScoredRowOrZero,
            formula_source: super::contract::FormulaSource::Table,
            formula_window: 32,
            enum_lanes_max: default_train_enum_lanes_max(),
            enum_lane_visits_max: default_train_enum_lane_visits_max(),
            enum_dispatch_visits_max: default_train_enum_dispatch_visits_max(),
            enum_fit_name: None,
            enum_fit_sha256: None,
            enum_fit_subset: None,
            lambda_assign: default_train_lambda_assign(),
            ion_request_work_max: default_train_ion_request_work_max(),
            formula_evidence_work_max: default_train_formula_evidence_work_max(),
            formula_evidence_dispatch_max: default_train_formula_evidence_dispatch_max(),
            precursor_jitter_ppm: default_train_precursor_jitter_ppm(),
            precursor_jitter_variants: default_train_precursor_jitter_variants(),
        }
    }
}

impl TrainConfig {
    /// Check the documented ranges: `batch >= 1`, `1 <= slots <= 16` (the
    /// contract cap), finite positive `lr`, finite non-negative
    /// `weight_decay` and `formula_weight`, finite positive `grad_clip` when
    /// set.
    pub fn validate(&self) -> Result<()> {
        if self.batch == 0 {
            return Err(Error::config(
                "TrainConfig::validate: batch is 0 (needs at least one spectrum)".to_string(),
            ));
        }
        if !(1..=16).contains(&self.slots) {
            return Err(Error::config(format!(
                "TrainConfig::validate: slots {} is not in 1..=16",
                self.slots
            )));
        }
        if !(self.lr.is_finite() && self.lr > 0.0) {
            return Err(Error::config(format!(
                "TrainConfig::validate: lr {} is not finite and positive",
                self.lr
            )));
        }
        for (name, value) in [
            ("weight_decay", self.weight_decay),
            ("formula_weight", self.formula_weight),
            ("lambda_assign", self.lambda_assign),
        ] {
            if !(value.is_finite() && value >= 0.0) {
                return Err(Error::config(format!(
                    "TrainConfig::validate: {name} {value} is not finite and non-negative"
                )));
            }
        }
        if let Some(clip) = self.grad_clip
            && !(clip.is_finite() && clip > 0.0)
        {
            return Err(Error::config(format!(
                "TrainConfig::validate: grad_clip {clip} is not finite and positive"
            )));
        }
        if !matches!(self.formula_window, 32 | 128 | 512 | 2048) {
            return Err(Error::config(format!(
                "TrainConfig::validate: formula_window {} is not one of 32, 128, 512, 2048",
                self.formula_window
            )));
        }
        if self.enum_lanes_max == 0 {
            return Err(Error::config(format!(
                "TrainConfig::validate: enum_lanes_max {} is not non-zero",
                self.enum_lanes_max
            )));
        }
        if self.enum_lane_visits_max == 0 {
            return Err(Error::config(format!(
                "TrainConfig::validate: enum_lane_visits_max {} is not non-zero",
                self.enum_lane_visits_max
            )));
        }
        if self.enum_dispatch_visits_max == 0 {
            return Err(Error::config(format!(
                "TrainConfig::validate: enum_dispatch_visits_max {} is not non-zero",
                self.enum_dispatch_visits_max
            )));
        }
        if self.ion_request_work_max == 0 {
            return Err(Error::config(
                "TrainConfig::validate: ion_request_work_max 0 is not non-zero".to_string(),
            ));
        }
        if self.formula_evidence_work_max == 0 {
            return Err(Error::config(
                "TrainConfig::validate: formula_evidence_work_max 0 is not non-zero".to_string(),
            ));
        }
        if self.formula_evidence_dispatch_max == 0 {
            return Err(Error::config(
                "TrainConfig::validate: formula_evidence_dispatch_max 0 is not non-zero".to_string(),
            ));
        }
        if !(self.precursor_jitter_ppm.is_finite()
            && self.precursor_jitter_ppm >= 0.0
            && self.precursor_jitter_ppm <= 5.0)
        {
            return Err(Error::config(format!(
                "TrainConfig::validate: precursor_jitter_ppm {} is not finite and in [0, 5]",
                self.precursor_jitter_ppm
            )));
        }
        Ok(())
    }
}

/// One reported optimizer step: the pre-update losses with the formula
/// coverage that produced them.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct LossReport {
    /// Optimizer steps completed after this step (1-based).
    pub step: u64,
    /// `L = L_graph + formula_weight * L_formula + lambda_assign * L_assign`.
    pub loss: f32,
    /// Graph loss before the update.
    pub graph: f32,
    /// Formula loss before the update.
    pub formula: f32,
    /// Assignment loss `L_assign` (pseudo-label, oracle formula) before the
    /// update; 0 when assignment is disabled.
    #[serde(default)]
    pub assign: f32,
    /// Spectra in the step (the divisor of `L_graph`, labeled or not).
    pub spectra: usize,
    /// Spectra whose gold formula was scored (the `L_formula` divisor).
    pub formula_present: usize,
    /// Spectra whose gold formula was absent from the window (`L_formula` 0).
    /// V0 name, kept as an alias of `gold_not_scored` for the experiment
    /// report schema (see `examples/ms2_experiment.rs`); new code should read
    /// `gold_not_scored`.
    #[serde(default)]
    pub formula_absent: usize,
    /// Spectra whose gold is not in the scored support (V1 §1.2 training
    /// metric; the request status `formula_absent` keeps its contract meaning
    /// and is never used for it).
    #[serde(default)]
    pub gold_not_scored: usize,
    /// Assignment eligible peaks (`label_state == 1 or 3`, the `L_assign`
    /// denominator).
    #[serde(default)]
    pub assign_eligible: usize,
    /// True-partial peaks (`label_state == 3`: some label kept while some
    /// label of the same peak is not).
    #[serde(default)]
    pub assign_partial: usize,
    /// Peaks with labels but none kept (`label_state == 2`).
    #[serde(default)]
    pub assign_dropped: usize,
    /// Labels beyond `L` counted on the host at upload (no read).
    #[serde(default)]
    pub assignment_label_overflow: usize,
}

/// Teacher-forced evaluation of one spectrum batch, read in one batched read.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct TeacherEval {
    /// Per-target negative log-likelihood `[B*G]`.
    pub nll: Vec<f32>,
    /// Per-target weights `[B*G]` (0 for empty slots).
    pub q: Vec<f32>,
    /// Scored positions per target `[B*G]` (the `use` rows of the target).
    pub scored_tokens: Vec<u32>,
    /// Gold window slot per spectrum `[B]` (`u32::MAX` when absent).
    pub gold_slot: Vec<u32>,
    /// Gold log-probability per spectrum `[B]` (0 when absent).
    pub gold_log_prob: Vec<f32>,
    /// Spectra evaluated.
    pub spectra: usize,
    /// Target slots per spectrum.
    pub slots: usize,
    /// Molecule index per spectrum `[B]`, for
    /// [`teacher_nll_per_token`](super::metrics::teacher_nll_per_token).
    pub molecules: Vec<usize>,
    /// Donor rows of the same molecule as the recipient (0 for
    /// molecule-aware donors; reported by every evaluation).
    pub donor_same_molecule: usize,
    /// Recipients with no donor peak passing the mass filter
    /// (`0 < mz <= precursor + 2 Da`); they stay empty, never falling back
    /// to own peaks.
    pub donor_no_eligible_peaks: usize,
}

/// Teacher-forced field detail of one spectrum batch, for `--diagnose`.
///
/// The same forward pass as [`TeacherEval`] plus the read-back per-field
/// log-probabilities and the host target buffers the field split needs.
/// Aggregates only are reported; per-spectrum rows never leave the host loop
/// except inside this struct for the caller to aggregate.
pub struct TeacherFieldEval {
    /// Per-target negative log-likelihood `[B*G]`.
    pub nll: Vec<f32>,
    /// Per-target weights `[B*G]` (0 for empty slots).
    pub q: Vec<f32>,
    /// Scored positions per target `[B*G]`.
    pub scored_tokens: Vec<u32>,
    /// Gathered per-field log-probabilities `[B*G*T*4]` row-major
    /// (`TeacherOutput.field_log_prob` read back).
    pub field_log_prob: Vec<f32>,
    /// Field-use indicators `[B*G*T*4]` (host `use_mask`).
    pub use_mask: Vec<f32>,
    /// Target tokens `[B*G*T*4]` (host tokens; kind at `pos` decides the
    /// STOP split of the kind field).
    pub tokens: Vec<u32>,
    /// Trace length `T`.
    pub max_steps: usize,
    /// Spectra evaluated.
    pub spectra: usize,
    /// Target slots per spectrum.
    pub slots: usize,
    /// Molecule index per spectrum `[B]`.
    pub molecules: Vec<usize>,
    /// Donor rows of the same molecule (0 for molecule-aware donors).
    pub donor_same_molecule: usize,
    /// Recipients with no eligible donor peak.
    pub donor_no_eligible_peaks: usize,
}

/// One `(B, n_raw)` bucket of preallocated training buffers, reused across
/// steps with the same shapes.
struct TrainBucket<R: Runtime, E: FloatElem> {
    /// Bucket key: batch, raw peak capacity, scored-candidate capacity,
    /// enum lanes `P`, formula-features layout (as `u8`: 0 `Counts`,
    /// 1 `Evidence`).
    key: (usize, usize, usize, usize, u8),
    /// Peak-selection scratch for `(B, n_raw, N)`.
    peaks: PeakBuffers<R, E>,
    /// Formula window for `(B, M)`.
    formula: FormulaBuffers<R, E>,
    /// `[B * P, 2]` lane stats for the enumerating source (empty for table).
    lane_stats: IdTensor<R>,
    /// `[B * P]` offsets for the enumerating source (empty for table).
    offsets: IdTensor<R>,
    /// Grammar replay for `(B*G, T, A)`.
    replay: ReplayBuffers<R>,
}

/// Host-side batch preparation shared by [`Ms2Trainer::step`] and
/// [`Ms2Trainer::teacher_eval`].
struct Prepared {
    /// Contract batch (donor-substituted under `ShuffledSpectrum`, own peaks
    /// otherwise).
    spectra: SpectrumBatch,
    /// Packed training targets (always the recipients').
    targets: TargetBatch,
    /// Gold composition counts per spectrum (`[B, 10]` flat, parent
    /// composition in `ELEMENTS` order), uploaded with the batch for
    /// `gold_slot` and teacher forcing (V1 §1.2).
    gold_counts: Vec<u32>,
    /// Scored positions per target, from the `use` mask.
    scored_tokens: Vec<u32>,
    /// Raw peak capacity the batch was bucketed to.
    n_raw: usize,
    /// Donor rows of the same molecule (0 for molecule-aware donors).
    donor_same_molecule: usize,
    /// Recipients with no eligible donor peak.
    donor_no_eligible_peaks: usize,
}

/// One boundary of [`Ms2Trainer::step_with_boundaries`]: the hook runs after
/// the forward/loss build, after the backward pass, and after the optimizer
/// update, so a profiling driver observes the real phases of one training
/// step instead of subtracting an evaluation-only pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepPhase {
    /// Forward pass, loss build and report packing are enqueued.
    AfterForward,
    /// Gradients are computed (with the device-side clip scale, when set).
    AfterBackward,
    /// The optimizer update is enqueued.
    AfterOptimizer,
}

/// The forward half of a training step, between
/// [`Ms2Trainer::forward_state`] and [`Ms2Trainer::backward_state`].
///
/// Fields stay private: a profiling driver names the type to stash it
/// between device-timing spans without touching its contents. Holding it
/// keeps the autograd graph alive for the backward phase.
pub struct ForwardState<R: Runtime, E: FloatElem> {
    // `tout`, `graph` and `formula_loss` are never read directly: holding
    // them pins the forward values' lifetimes exactly as `step` did before
    // the phase split (one code path, bit-identical results).
    #[allow(dead_code)]
    tout: TeacherOutput<R, E>,
    #[allow(dead_code)]
    graph: Var<R, E>,
    #[allow(dead_code)]
    formula_loss: Var<R, E>,
    // `e_cond` is never read by the step either: holding it pins the exact
    // conditioning tensor handed to `decoder.teacher`, which the test hook
    // observes through the same prefix.
    #[allow(dead_code)]
    e_cond: Var<R, E>,
    #[allow(dead_code)]
    assign_loss: Option<Var<R, E>>,
    total: Var<R, E>,
    packed: Var<R, E>,
    prep: Prepared,
    spectra: usize,
    /// Labels beyond `L` counted on the host at upload (no read).
    #[allow(dead_code)]
    assign_overflow: usize,
}

/// Gradients with the device-side clip scale, between
/// [`Ms2Trainer::backward_state`] and [`Ms2Trainer::optimizer_step`].
///
/// Fields stay private (see [`ForwardState`]).
pub struct BackwardState<R: Runtime, E: FloatElem> {
    grads: Grads<R, E>,
    scale: Option<GradScale<R, E>>,
}

/// Slots per virtual spectrum of the grouped teacher pass
/// ([`TargetBatch::compact`]).
const COMPACT_GROUP: usize = 4;
/// Virtual spectra are padded to a multiple of this, so a training run sees
/// row counts in steps of `COMPACT_GROUP * COMPACT_BUCKET` and the shapes
/// its kernels and tuned products meet stay few.
const COMPACT_BUCKET: usize = 8;
/// Positions per packed row of the ragged teacher pass, in horizons
/// ([`TargetBatch::pack`]): long enough that a spectrum's traces fill a row
/// with little left over.
const PACK_HORIZONS: usize = 2;
/// Scan rows (any spectrum's traces) come in multiples of this.
const PACK_SCAN_BUCKET: usize = 4;
/// Attention rows (one spectrum's traces each) come in multiples of this.
const PACK_ATTN_BUCKET: usize = 8;
/// Trace rows (what the heads see) come in multiples of this.
const PACK_TRACE_BUCKET: usize = 32;

/// How a training step lays out its teacher pass. Every layout gives the
/// same losses and gradients up to rounding; they differ in how much of the
/// padding they compute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TeacherPass {
    /// `B * slots` rows of the full horizon, empty slots included: the
    /// layout the evaluation paths use.
    Padded,
    /// The occupied slots only, in virtual spectra
    /// ([`TargetBatch::compact`]), each row still the full horizon.
    Slots,
    /// The occupied slots as ragged sequences ([`TargetBatch::pack`]): the
    /// decoder layers skip the positions past a trace's end as well.
    Ragged,
}

/// `-1` not yet read from `MAMBA3_MS2_COMPACT_TEACHER`, else a [`TeacherPass`].
static TEACHER_PASS: core::sync::atomic::AtomicI8 = core::sync::atomic::AtomicI8::new(-1);

/// The teacher-pass layout of a training step: [`TeacherPass::Ragged`]
/// unless `MAMBA3_MS2_COMPACT_TEACHER` is `0` (padded) or `slots`, or
/// [`set_teacher_pass`] chose another.
pub fn teacher_pass() -> TeacherPass {
    use core::sync::atomic::Ordering;
    let code = match TEACHER_PASS.load(Ordering::Relaxed) {
        -1 => {
            let code = match std::env::var("MAMBA3_MS2_COMPACT_TEACHER").as_deref() {
                Ok("0") => 0,
                Ok("slots") => 1,
                _ => 2,
            };
            TEACHER_PASS.store(code, Ordering::Relaxed);
            code
        }
        code => code,
    };
    match code {
        0 => TeacherPass::Padded,
        1 => TeacherPass::Slots,
        _ => TeacherPass::Ragged,
    }
}

/// Choose the teacher-pass layout for this process (for comparing them).
pub fn set_teacher_pass(pass: TeacherPass) {
    let code = match pass {
        TeacherPass::Padded => 0,
        TeacherPass::Slots => 1,
        TeacherPass::Ragged => 2,
    };
    TEACHER_PASS.store(code, core::sync::atomic::Ordering::Relaxed);
}

/// The compact teacher pass on (the default, [`TeacherPass::Ragged`]) or off
/// ([`TeacherPass::Padded`]).
pub fn set_compact_teacher(on: bool) {
    set_teacher_pass(if on { TeacherPass::Ragged } else { TeacherPass::Padded });
}

/// The V0 trainer: the composed model, its optimizer, the resident formula
/// table and preallocated per-batch-shape buffers.
pub struct Ms2Trainer<R: Runtime, E: FloatElem> {
    /// The composed model (encoder, formula head, decoder).
    pub model: Ms2Model<R, E>,
    /// AdamW over every model parameter.
    optimizer: AdamW<R, E>,
    /// Parameters in named-parameter order, for the optimizer step.
    params: Vec<Param<R, E>>,
    /// The host formula table (for the host-side gold-slot reference).
    table: FormulaTable,
    /// Composition to table row, for gold lookups without a scan.
    gold_index: HashMap<Composition, u32>,
    /// The resident device formula table.
    device_table: DeviceFormulaTable<R, E>,
    /// Resident Fourier wavelengths and atom-type table.
    constants: Ms2Constants<R>,
    /// Device the model lives on.
    device: Device<R>,
    /// Training hyperparameters.
    train: TrainConfig,
    /// Grammar limits from the model config.
    limits: Limits,
    /// Cached buffer buckets by `(batch, n_raw, window_m)`, oldest first.
    buckets: Vec<TrainBucket<R, E>>,
    /// Replay buffers of the compact teacher pass by row count, oldest first.
    compact_replay: Vec<(usize, ReplayBuffers<R>)>,
    /// Preallocated generation state for [`Ms2Trainer::generate_eval`].
    workspace: RefCell<GenerationWorkspace<R, E>>,
    /// Whether the next [`Ms2Trainer::step`] reports its losses.
    report_pending: bool,
    /// Optimizer steps completed so far.
    steps: u64,
    /// Host batch uploaded by the most recent forward prefix (test hook for
    /// the jitter-propagation test: the precursors recorded here are what
    /// enumeration, residuals, metadata and the peak filter all saw).
    /// Explicit opt-in via [`Self::capture_prep_batch`], default off: when
    /// off no clone is performed and nothing is retained (zero production
    /// cost).
    #[doc(hidden)]
    pub last_prep_batch: Option<SpectrumBatch>,
    /// Whether the test hook above retains the prepared batch. Default
    /// false (task E5F Part B: no clone, nothing retained).
    capture_prep: bool,
}

/// The shared forward prefix of [`Ms2Trainer::forward_with_donors`]: the
/// prepared host batch, the encoder output, the scored formula window, the
/// device gold slots and the teacher-forcing conditioning embedding.
///
/// The `e_cond` here is the exact tensor the finish hands to
/// `decoder.teacher`: [`Ms2Trainer::forward_state`] keeps it in the returned
/// [`ForwardState`], so the test hook observes production.
struct ForwardPrefix<R: Runtime, E: FloatElem> {
    /// Host-side batch (donor-substituted under `ShuffledSpectrum`).
    prep: Prepared,
    /// Encoder output (`pool` feeds the formula head).
    encoded: EncoderOutput<R, E>,
    /// Scored formula window (`log_prob` feeds the loss, `embedding` feeds
    /// the `ScoredRowOrZero` conditioning branch).
    scored: FormulaOutput<R, E>,
    /// Device gold slots (`u32::MAX` when absent).
    gold_slot_t: IdTensor<R>,
    /// Conditioning embedding handed to `decoder.teacher`.
    e_cond: Var<R, E>,
    /// Spectra in the batch.
    spectra: usize,
    /// Assignment loss `L_assign` (pseudo-label, oracle formula); `None`
    /// when assignment is disabled (`lambda_assign == 0` or no head).
    assign_loss: Option<Var<R, E>>,
    /// Assignment report counts `[eligible, partial, dropped]` on the
    /// device; `None` when disabled (no read, no launch).
    assign_counts: Option<Tensor<R, E>>,
    /// Labels beyond `L` counted on the host at upload
    /// (`assignment_label_overflow`; no read).
    assign_overflow: usize,
}

/// Evidence diagnostics of one evaluation batch (architecture §1.6, report
/// boundary).
///
/// Under `FormulaFeatures::Counts` there is nothing to diagnose (the driver
/// reports `null`). Under `Evidence`, computed from one extra device read
/// of `cand_ev` and the gold slot per evaluation batch; a warmed
/// non-report training step still reads nothing.
#[derive(Clone, Debug, PartialEq)]
pub struct EvidenceDiagnostics {
    /// Scored `(spectrum, slot)` pairs examined.
    pub scored: usize,
    /// Of them, with `complete = 0`.
    pub incomplete: usize,
    /// Spectra examined.
    pub spectra: usize,
    /// Sum over spectra of the evidence-peak count.
    pub peaks_sum: f64,
    /// Sum over spectra whose gold formula is in the scored support of the
    /// gold slot's explained-count fraction (`explained / n_ev`, `0` when
    /// the spectrum has no evidence peak).
    pub gold_sum: f64,
    /// Spectra whose gold formula is in the scored support.
    pub gold_spectra: usize,
    /// Sum over the other scored slots of those spectra of their
    /// explained-count fractions.
    pub other_sum: f64,
    /// Other scored slots summed.
    pub other_slots: usize,
}

impl EvidenceDiagnostics {
    /// Fraction of scored candidates with `complete = 0` (`0` when none
    /// scored).
    pub fn incomplete_fraction(&self) -> f64 {
        if self.scored == 0 {
            0.0
        } else {
            self.incomplete as f64 / self.scored as f64
        }
    }

    /// Mean number of evidence peaks per spectrum (`0` when empty).
    pub fn peaks_mean(&self) -> f64 {
        if self.spectra == 0 {
            0.0
        } else {
            self.peaks_sum / self.spectra as f64
        }
    }

    /// Mean explained-count fraction of the gold slot (`None` when no
    /// spectrum has its gold formula in the scored support).
    pub fn gold_explained_fraction(&self) -> Option<f64> {
        if self.gold_spectra == 0 {
            None
        } else {
            Some(self.gold_sum / self.gold_spectra as f64)
        }
    }

    /// Mean explained-count fraction over the other scored slots of those
    /// spectra (`None` when there is no such slot).
    pub fn other_explained_fraction(&self) -> Option<f64> {
        if self.other_slots == 0 {
            None
        } else {
            Some(self.other_sum / self.other_slots as f64)
        }
    }
}

impl<R: Runtime, E: FloatElem> Ms2Trainer<R, E> {
    /// Build the model, upload the table and prepare the optimizer.
    ///
    /// The model config's table reference is stamped with the uploaded
    /// table's row count and SHA-256, binding the weights to the exact rows;
    /// [`Ms2Trainer::save`] records that reference and [`Ms2Trainer::load`]
    /// checks it.
    pub fn new(
        model_config: &ModelConfig,
        table: &FormulaTable,
        train: &TrainConfig,
        device: &Device<R>,
    ) -> Result<Self> {
        train.validate()?;
        // Dtype gate (contracts §3.3): the actual neural element type must
        // equal the configured dtype and be in the validated set — refused
        // here (Error::Config on mismatch), before any allocation, upload or
        // launch.
        Ms2Capabilities::check_device::<R, E>(device, model_config)?;
        let device_table = DeviceFormulaTable::upload(table, device)?;
        let mut stamped = model_config.clone();
        stamped.formula_table.rows = device_table.rows as u32;
        stamped.formula_table.sha256 = device_table.sha256.clone();
        let mut rng = Rng::seeded(train.seed);
        let model = Ms2Model::init(&stamped, device, &mut rng)?;
        let ssm_atoms = stamped.max_atoms as usize;
        let ssm_closures = stamped.max_ring_closures as usize;
        let limits = Limits::new(ssm_atoms, ssm_closures)
            .map_err(|e| Error::config(format!("Ms2Trainer::new: model limits rejected: {e}")))?;
        let optimizer = AdamWConfig {
            learning_rate: train.lr,
            weight_decay: train.weight_decay,
            ..Default::default()
        }
        .init();
        let params = model
            .named_parameters()
            .into_iter()
            .map(|(_, p)| p)
            .collect();
        let mut gold_index = HashMap::new();
        for row in 0..table.len() {
            gold_index.insert(*table.composition(row), row as u32);
        }
        Ok(Self {
            model,
            optimizer,
            params,
            table: table.clone(),
            gold_index,
            device_table,
            constants: Ms2Constants::new(device),
            device: device.clone(),
            train: train.clone(),
            limits,
            buckets: Vec::new(),
            compact_replay: Vec::new(),
            workspace: RefCell::new(GenerationWorkspace::new()),
            report_pending: false,
            steps: 0,
            last_prep_batch: None,
            capture_prep: false,
        })
    }

    /// Test hook switch for [`Self::last_prep_batch`] (task E5F Part B):
    /// explicit opt-in, default off. When off, the forward prefix performs
    /// no clone and retains nothing; when on, it records the exact host
    /// batch that is uploaded below. Turning off clears any retained batch.
    #[doc(hidden)]
    pub fn capture_prep_batch(&mut self, on: bool) {
        self.capture_prep = on;
        if !on {
            self.last_prep_batch = None;
        }
    }

    /// Test accessor for the retained prepared batch: `None` unless
    /// [`Self::capture_prep_batch`] was switched on (task E5F Part B).
    #[doc(hidden)]
    pub fn captured_prep_batch(&self) -> Option<&SpectrumBatch> {
        self.last_prep_batch.as_ref()
    }

    /// Training hyperparameters.
    pub fn train_config(&self) -> &TrainConfig {
        &self.train
    }

    /// Optimizer steps completed so far.
    pub fn step_count(&self) -> u64 {
        self.steps
    }

    /// SHA-256 of the resident formula table.
    pub fn table_sha256(&self) -> &str {
        &self.device_table.sha256
    }

    /// Upload enumeration artifacts once and bind them to the model config
    /// (V1 §1.4). Training with `formula_source = Enumerate` requires it;
    /// a mismatch at load is `Error::Config`.
    pub fn upload_enum_artifacts(
        &mut self,
        domain: &super::formula_enum::EnumDomain,
        bounds: &super::formula_enum::RatioBounds,
    ) -> Result<()> {
        let device = self.device.clone();
        self.model.upload_enum_artifacts(domain, bounds, &device)
    }

    /// Attach a memoised device enumeration (task T6), forwarded to the
    /// model: with `FormulaSource::Enumerate`, the training prefix and
    /// `generate` serve a fully cached batch from it (two uploads, no
    /// enumeration kernel) and run the device enumeration otherwise.
    /// `None` (the default) means today's behaviour. Attaching a cache
    /// adds no device read to a training step or to `generate`.
    pub fn set_enum_cache(&mut self, cache: Option<Arc<EnumCache>>) {
        self.model.set_enum_cache(cache);
    }

    /// Cache lookups attempted and served since init, as `(lookups, hits)`
    /// (see [`Ms2Model::enum_cache_stats`](super::generate::Ms2Model::enum_cache_stats)).
    /// Both the training prefix and `generate` count here.
    pub fn enum_cache_stats(&self) -> (u64, u64) {
        self.model.enum_cache_stats()
    }

    /// The cache header this trainer enumerates with at window `M`: the
    /// resident artifacts' SHA-256, their depth `P`, the window, the scored
    /// cap ([`TRAIN_ROWS_SCORED_MAX`] bounded by `M`) and the lane visit
    /// budget. [`Error::Config`] without resident artifacts.
    pub fn enum_cache_header(&self, window_m: usize) -> Result<EnumCacheHeader> {
        let Some(artifacts) = self.model.enum_artifacts.as_ref() else {
            return Err(Error::config(
                "Ms2Trainer::enum_cache_header: formula_source Enumerate needs resident enum artifacts".to_string(),
            ));
        };
        let p = u32::try_from(artifacts.p).map_err(|_| {
            Error::config(format!(
                "Ms2Trainer::enum_cache_header: rare-table depth {} exceeds u32",
                artifacts.p
            ))
        })?;
        let window = u32::try_from(window_m).map_err(|_| {
            Error::config(format!(
                "Ms2Trainer::enum_cache_header: window {window_m} exceeds u32"
            ))
        })?;
        Ok(EnumCacheHeader::new(
            artifacts.domain_sha256.clone(),
            artifacts.bounds_sha256.clone(),
            p,
            window,
            TRAIN_ROWS_SCORED_MAX,
            self.train.enum_lane_visits_max,
        ))
    }

    /// Fill `cache` from the EXISTING device enumeration (count → offsets →
    /// fill → pad) for each batch, reading `cand` and `counters` back (one
    /// batched read per batch — this is a precompute pass, reads are
    /// expected) and inserting every spectrum not yet present. Fully cached
    /// batches are skipped without any launch or read. The cache header
    /// must match [`Ms2Trainer::enum_cache_header`] at `window_m`, else
    /// `Error::Config` naming the field. Reuses the production launch
    /// functions, not a copy.
    pub fn build_enum_cache<'a>(
        &self,
        batches: impl Iterator<Item = &'a SpectrumBatch>,
        window_m: usize,
        cache: &mut EnumCache,
    ) -> Result<()> {
        let Some(artifacts) = self.model.enum_artifacts.as_ref() else {
            return Err(Error::config(
                "Ms2Trainer::build_enum_cache: formula_source Enumerate needs resident enum artifacts".to_string(),
            ));
        };
        artifacts.check(&self.model.config)?;
        cache
            .header()
            .check_compatible(&self.enum_cache_header(window_m)?)?;
        let scored_cap = TRAIN_ROWS_SCORED_MAX.min(window_m as u32);
        for batch in batches {
            run_device_enumeration_into::<R, E>(
                &self.device,
                artifacts,
                batch,
                scored_cap,
                self.train.enum_lanes_max,
                self.train.enum_dispatch_visits_max,
                self.train.enum_lane_visits_max,
                window_m,
                cache,
            )?;
        }
        Ok(())
    }

    /// Ask the next [`Ms2Trainer::step`] to report its losses (exactly one
    /// batched read of `[L, L_graph, L_formula]`).
    pub fn request_report(&mut self) {
        self.report_pending = true;
    }

    /// Raw peak capacity bucket for these spectra: the smallest of 64, 128,
    /// 256, 512 covering the longest stored peak list.
    fn bucket_n_raw(&self, set: &ExperimentSet, indices: &[usize]) -> Result<usize> {
        let mut longest = 0usize;
        for &i in indices {
            let s = set.spectra.get(i).ok_or_else(|| {
                Error::config(format!(
                    "Ms2Trainer: spectrum index {i} outside {} spectra",
                    set.spectra.len()
                ))
            })?;
            longest = longest.max(s.spectrum.peak_id.len());
        }
        Self::bucket_for_len(longest)
    }

    /// Raw peak capacity bucket covering both recipients and their donors:
    /// the donor peak lists travel in the batch, so the bucket must cover
    /// the longest of either side.
    fn bucket_n_raw_with_donors(
        &self,
        set: &ExperimentSet,
        indices: &[usize],
        donors: &[usize],
    ) -> Result<usize> {
        let mut longest = 0usize;
        for &i in indices.iter().chain(donors.iter()) {
            let s = set.spectra.get(i).ok_or_else(|| {
                Error::config(format!(
                    "Ms2Trainer: spectrum index {i} outside {} spectra",
                    set.spectra.len()
                ))
            })?;
            longest = longest.max(s.spectrum.peak_id.len());
        }
        Self::bucket_for_len(longest)
    }

    /// Bucket for a peak-list length.
    fn bucket_for_len(longest: usize) -> Result<usize> {
        for bucket in [64usize, 128, 256, 512] {
            if longest <= bucket {
                return Ok(bucket);
            }
        }
        Err(Error::config(format!(
            "Ms2Trainer: longest peak list {longest} exceeds the 512-peak bucket"
        )))
    }

    /// Donor indices parallel to `indices` from the set-wide
    /// [`ExperimentSet::donor_map`] under the trainer seed.
    fn donors_for(&self, set: &ExperimentSet, indices: &[usize]) -> Result<Vec<usize>> {
        let map = set.donor_map(self.train.seed)?;
        indices
            .iter()
            .map(|&i| {
                map.get(i).copied().ok_or_else(|| {
                    Error::config(format!(
                        "Ms2Trainer: spectrum index {i} outside donor map of {} spectra",
                        map.len()
                    ))
                })
            })
            .collect()
    }

    /// Table row of a parent composition (`u32::MAX` when the table lacks it:
    /// the spectrum then contributes 0 to `L_formula` and counts as absent).
    fn gold_row(&self, parent: &Composition) -> u32 {
        self.gold_index.get(parent).copied().unwrap_or(u32::MAX)
    }

    /// Assemble the host batches, explicitly choosing donor peaks or own
    /// peaks. `use_donors` substitutes molecule-aware donor peaks (the
    /// `ShuffledSpectrum` control as contracts §10 defines it, via
    /// [`ExperimentSet::donor_map`] under the trainer seed); `false` keeps
    /// the spectra's own peaks. The targets, gold slots and scored counts are
    /// always the recipients'. The control-following forward pass uses
    /// `use_donors = (control == ShuffledSpectrum)`.
    fn prepare_with_donors(
        &self,
        set: &ExperimentSet,
        indices: &[usize],
        use_donors: bool,
        train_jitter: bool,
    ) -> Result<Prepared> {
        if indices.is_empty() {
            return Err(Error::config(
                "Ms2Trainer: cannot prepare an empty index list".to_string(),
            ));
        }
        let (mut spectra, donor_same_molecule, donor_no_eligible_peaks, n_raw) = if use_donors {
            let donors = self.donors_for(set, indices)?;
            let n_raw = self.bucket_n_raw_with_donors(set, indices, &donors)?;
            let batch = spectrum_batch_with_donors(set, indices, &donors, n_raw as u32)?;
            let (same, no_eligible) = donor_stats(set, indices, &donors)?;
            (batch, same, no_eligible, n_raw)
        } else {
            let n_raw = self.bucket_n_raw(set, indices)?;
            let batch = spectrum_batch_for(set, indices, n_raw as u32)?;
            (batch, 0, 0, n_raw)
        };
        // Precursor jitter (architecture §1.6, training): host side, before
        // the batch is uploaded. With `precursor_jitter_variants == 0` the
        // draw is keyed by `(seed, 1 + step, spectrum index)` via
        // [`apply_precursor_jitter`], so a run is reproducible and the
        // draw for a spectrum does not depend on batch composition. With
        // `precursor_jitter_variants = V > 0` the jitter comes from a fixed
        // pool: step `s` uses draw `v = hash(seed, s, spectrum index) mod V`
        // (`split_tag = 1 + v`), so the driver can cache all `V` variants
        // up front. The shuffled-spectrum control keeps the recipient's
        // precursor (jitter keys by the recipient indices), while the peaks
        // are the donor's. `sigma = 0` leaves every byte of the batch
        // unchanged. Evaluation passes `train_jitter = false` and always
        // sees the stored precursor.
        if train_jitter && self.train.precursor_jitter_ppm > 0.0 {
            let step_tag = self.steps + 1;
            if self.train.precursor_jitter_variants == 0 {
                apply_precursor_jitter(
                    &mut spectra,
                    indices,
                    self.train.precursor_jitter_ppm,
                    self.train.seed,
                    step_tag,
                );
            } else {
                let variants = self.train.precursor_jitter_variants;
                for (b, &idx) in indices.iter().enumerate() {
                    if b >= spectra.precursor_mz_udalton.len() {
                        break;
                    }
                    let v = jitter_variant_index(
                        self.train.seed,
                        step_tag,
                        idx as u64,
                        variants,
                    );
                    let mz = spectra.precursor_mz_udalton[b];
                    spectra.precursor_mz_udalton[b] = jitter_precursor_mz(
                        mz,
                        f64::from(self.train.precursor_jitter_ppm),
                        self.train.seed,
                        1 + u64::from(v),
                        idx as u64,
                    );
                }
            }
        }
        let b = indices.len();
        let slots = self.train.slots;
        let targets = target_batch_for(set, indices, slots, self.limits)?;
        // V1 §1.2: `gold_counts` (parent compositions) travel with the batch
        // for the device `gold_slot` and teacher forcing. No host window
        // search here: the device kernel replaces it in the step
        // (`gold_slots_host` stays as its twin and for tests).
        let mut gold_counts = vec![0u32; b * 10];
        for (i, &idx) in indices.iter().enumerate() {
            let comp = set.spectra[idx].parent_composition;
            for e in 0..10 {
                gold_counts[i * 10 + e] = u32::from(comp[e]);
            }
        }
        // Scored positions per target: the `use` rows of the target (the kind
        // column counts every scored position once, START excluded).
        let t = self.limits.max_steps();
        let mut scored_tokens = vec![0u32; b * slots];
        for (row, n) in scored_tokens.iter_mut().enumerate() {
            let mut count = 0u32;
            for i in 0..t {
                count += targets.use_mask[(row * t + i) * 4] as u32;
            }
            *n = count;
        }
        Ok(Prepared {
            spectra,
            targets,
            gold_counts,
            scored_tokens,
            n_raw,
            donor_same_molecule,
            donor_no_eligible_peaks,
        })
    }

    /// The buffer bucket for this batch shape, allocating on a miss (and
    /// evicting the oldest past four shapes, as in generation). A free
    /// function so the bucket borrow never overlaps the model borrow in
    /// [`Ms2Trainer::forward_prefix`].
    #[allow(clippy::too_many_arguments)]
    fn bucket_for<'a>(
        buckets: &'a mut Vec<TrainBucket<R, E>>,
        device: &Device<R>,
        n_peaks: usize,
        atoms: usize,
        steps: usize,
        batch: usize,
        n_raw: usize,
        slots: usize,
        window_m: usize,
        enum_p: usize,
        formula_features: FormulaFeatures,
    ) -> Result<&'a mut TrainBucket<R, E>> {
        let key = (
            batch,
            n_raw,
            window_m,
            enum_p,
            u8::from(formula_features == FormulaFeatures::Evidence),
        );
        if let Some(pos) = buckets.iter().position(|bk| bk.key == key) {
            return Ok(&mut buckets[pos]);
        }
        let lanes = batch * enum_p;
        buckets.push(TrainBucket {
            key,
            peaks: PeakBuffers::new(batch, n_raw, n_peaks, device),
            formula: if formula_features == FormulaFeatures::Evidence {
                FormulaBuffers::new_evidence(batch, window_m, 1, device)
            } else {
                FormulaBuffers::new(batch, window_m, 1, device)
            },
            lane_stats: IdTensor::empty(vec![lanes, 2], device),
            offsets: IdTensor::empty(vec![lanes], device),
            replay: ReplayBuffers::new(batch * slots, steps, atoms, device),
        });
        while buckets.len() > 4 {
            buckets.remove(0);
        }
        Ok(buckets
            .iter_mut()
            .find(|bk| bk.key == key)
            .expect("the bucket just pushed is cached"))
    }

    /// Upload one spectrum batch and run the forward pass through the graph
    /// loss, returning everything [`Ms2Trainer::step`] and
    /// [`Ms2Trainer::teacher_eval`] share. No device read.
    ///
    /// Under `ShuffledSpectrum` the batch already carries molecule-aware
    /// donor peaks (see [`Ms2Trainer::prepare`]); it is passed to the model
    /// as an ordinary request with [`Control::None`], so `generate`'s
    /// in-batch rotation is never applied a second time.
    ///
    /// The conditioning embedding travels last: the exact tensor handed to
    /// `decoder.teacher`.
    #[allow(clippy::type_complexity)]
    fn forward(
        &mut self,
        set: &ExperimentSet,
        indices: &[usize],
    ) -> Result<(
        crate::models::ms2::decoder::TeacherOutput<R, E>,
        Var<R, E>,
        Var<R, E>,
        Var<R, E>,
        Var<R, E>,
        Var<R, E>,
        IdTensor<R>,
        Prepared,
        Var<R, E>,
        Option<Var<R, E>>,
        Option<Tensor<R, E>>,
        usize,
    )> {
        self.forward_with_donors(
            set,
            indices,
            self.train.control == Control::ShuffledSpectrum,
            false,
            false,
        )
    }

    /// The shared forward prefix of every training forward pass: upload,
    /// encode, formula window/gather/features/score, gold slot and the
    /// teacher-forcing conditioning embedding.
    ///
    /// [`Ms2Trainer::step`] (via `forward_state`), [`Ms2Trainer::teacher_eval`]
    /// and the test hook all run this one function, and the returned `e_cond`
    /// is the exact tensor the finish hands to `decoder.teacher` — so a test
    /// observing the hook's embedding observes production, never a
    /// reconstruction. The bucket borrow ends with the prefix; the finish
    /// re-borrows the same `(batch, n_raw, window_m)` bucket for the replay
    /// buffers, so no buffer is allocated twice. No device read.
    fn forward_prefix(
        &mut self,
        set: &ExperimentSet,
        indices: &[usize],
        use_donors: bool,
        window_m: usize,
        train_jitter: bool,
    ) -> Result<ForwardPrefix<R, E>> {
        // Dtype gate (contracts §3.3): every `step` entry point checks the
        // actual neural element type against the model dtype (Error::Config on
        // mismatch) and the validated set, before any upload, allocation or
        // launch.
        Ms2Capabilities::check_device::<R, E>(&self.device, &self.model.config)?;
        let prep = self.prepare_with_donors(set, indices, use_donors, train_jitter)?;
        // Test hook (task E5F Part B, explicit opt-in, default off): record
        // the exact host batch that is uploaded below only when enabled, so
        // production pays no clone and retains nothing.
        if self.capture_prep {
            self.last_prep_batch = Some(prep.spectra.clone());
        } else {
            self.last_prep_batch = None;
        }
        let b = indices.len();
        let atoms = self.model.config.max_atoms as usize;
        // D3: resolve/check artifacts and lane products BEFORE any upload,
        // bucket allocation or encoder launch, so an excessive-lane request
        // is refused with counters unchanged.
        let enum_p = match self.train.formula_source {
            super::contract::FormulaSource::Table => 0,
            super::contract::FormulaSource::Enumerate => {
                let Some(artifacts) = self.model.enum_artifacts.as_ref() else {
                    return Err(Error::config(
                        "Ms2Trainer::forward_prefix: formula_source Enumerate needs resident enum artifacts".to_string(),
                    ));
                };
                artifacts.check(&self.model.config)?;
                let lanes = (b as u64)
                    .checked_mul(artifacts.p as u64)
                    .ok_or_else(|| {
                        Error::config(format!(
                            "Ms2Trainer::forward_prefix: batch {b} times P {} overflows u64",
                            artifacts.p
                        ))
                    })?;
                if lanes > u64::from(self.train.enum_lanes_max) {
                    return Err(Error::config(format!(
                        "Ms2Trainer::forward_prefix: B * P {lanes} exceeds enum_lanes_max {} (refused before any upload, allocation or launch)",
                        self.train.enum_lanes_max
                    )));
                }
                artifacts.p
            }
        };
        // The batch already carries donor peaks under the shuffled control;
        // it is an ordinary request from here on (`None`), so no second
        // rotation happens in the encoder or in `generate`. Every other
        // control keeps its own blinding with donor peaks: the `--diagnose`
        // peak sensitivity of a metadata-only or structure-prior model must
        // compare the same model on two inputs, not switch its peak path on.
        let encode_control = if self.train.control == Control::ShuffledSpectrum {
            Control::None
        } else {
            self.train.control
        };
        let spectra = DeviceSpectra::upload(&prep.spectra, &self.device)?;
        let bucket = Self::bucket_for(
            &mut self.buckets,
            &self.device,
            self.model.config.n_peaks as usize,
            atoms,
            self.limits.max_steps(),
            b,
            prep.n_raw,
            self.train.slots,
            window_m,
            enum_p,
            self.model.config.formula_features,
        )?;
        let encoded = self
            .model
            .encoder
            .encode(&spectra, &bucket.peaks, encode_control)?;
        match self.train.formula_source {
            super::contract::FormulaSource::Table => {
                ms2::formula_window(
                    &self.device_table.table,
                    &spectra.meta,
                    self.device_table.max_error,
                    u32::MAX,
                    TRAIN_ROWS_SCORED_MAX,
                    &bucket.formula,
                )?;
                ms2::formula_gather(
                    &bucket.formula.window,
                    &self.device_table.table,
                    &self.device_table.counts,
                    &mut bucket.formula.cand,
                )?;
            }
            super::contract::FormulaSource::Enumerate => {
                use crate::tensor::ops::ms2_enum::{EnumLaunch, cand_pad, enum_offsets};
                use super::formula_enum::{build_enum_meta, validate_enum_dispatch};
                let Some(artifacts) = self.model.enum_artifacts.as_ref() else {
                    return Err(Error::config(
                        "Ms2Trainer::forward_prefix: formula_source Enumerate needs resident enum artifacts".to_string(),
                    ));
                };
                artifacts.check(&self.model.config)?;
                // D3: lanes already refused before upload using
                // `self.train.enum_lanes_max`; re-check here defensively
                // with the same configured limit (no hardcoded constant).
                let lanes = (b as u64)
                    .checked_mul(artifacts.p as u64)
                    .ok_or_else(|| {
                        Error::config(format!(
                            "Ms2Trainer::forward_prefix: batch {b} times P {} overflows u64",
                            artifacts.p
                        ))
                    })?;
                if lanes > u64::from(self.train.enum_lanes_max) {
                    return Err(Error::config(format!(
                        "Ms2Trainer::forward_prefix: B * P {lanes} exceeds enum_lanes_max {} (refused)",
                        self.train.enum_lanes_max
                    )));
                }
                let scored_cap = TRAIN_ROWS_SCORED_MAX.min(window_m as u32);
                let meta_host = build_enum_meta(
                    &prep.spectra,
                    artifacts.domain_max_error,
                    self.train.enum_lane_visits_max,
                    scored_cap,
                );
                // T6 memo: a fully cached batch uploads `cand` and
                // `counters` (two uploads) and launches no enumeration
                // kernel; a partially cached batch takes the device path
                // (no mixing). Everything downstream sees bit-identical
                // `cand` and `counters`, and no device read is added to the
                // step by the cache.
                let mut served = false;
                if let Some(cache) = self.model.enum_cache_ref() {
                    let keys: Vec<[u32; 8]> = meta_host
                        .chunks_exact(8)
                        .map(|r| {
                            let mut k = [0u32; 8];
                            k.copy_from_slice(r);
                            k
                        })
                        .collect();
                    let mut hit = false;
                    if let Some((cand_host, counters_host)) =
                        cache.expand_batch(&keys, window_m)
                    {
                        // The lane preflight and every other refusal of the
                        // uncached path still applies with a cache: refuse
                        // here exactly as the wrappers would, so behaviour
                        // does not depend on cache state.
                        validate_enum_dispatch(
                            b,
                            artifacts.p,
                            window_m,
                            self.train.enum_lanes_max,
                        )
                        .map_err(|e| match e {
                            Error::Shape(msg) => Error::shape(format!(
                                "Ms2Trainer::forward_prefix (cached): {msg}"
                            )),
                            other => other,
                        })?;
                        bucket.formula.cand = IdTensor::from_slice(
                            &cand_host,
                            vec![b, window_m, 13],
                            &self.device,
                        )?;
                        bucket.formula.counters = IdTensor::from_slice(
                            &counters_host,
                            vec![b, 5],
                            &self.device,
                        )?;
                        hit = true;
                        served = true;
                    }
                    self.model.note_enum_cache_lookup(hit);
                }
                if !served {
                    let meta_t =
                        IdTensor::from_slice(&meta_host, vec![b, 8], &self.device)?;
                    let launch = EnumLaunch::from_chemistry();
                    launch.count(
                        &meta_t,
                        &artifacts.rare,
                        &artifacts.bounds,
                        &bucket.lane_stats,
                        self.train.enum_lanes_max,
                        self.train.enum_dispatch_visits_max,
                        self.train.enum_lane_visits_max,
                    )?;
                    enum_offsets(
                        &bucket.lane_stats,
                        &meta_t,
                        &bucket.offsets,
                        &bucket.formula.counters,
                        scored_cap,
                        window_m,
                        self.train.enum_lanes_max,
                    )?;
                    launch.fill(
                        &meta_t,
                        &artifacts.rare,
                        &artifacts.bounds,
                        &bucket.offsets,
                        &bucket.formula.cand,
                        scored_cap,
                        self.train.enum_lanes_max,
                        self.train.enum_dispatch_visits_max,
                        self.train.enum_lane_visits_max,
                    )?;
                    cand_pad(
                        &bucket.formula.counters,
                        &bucket.formula.cand,
                        artifacts.p,
                        self.train.enum_lanes_max,
                    )?;
                }
            }
        }
        if matches!(
            self.model.config.formula_features,
            FormulaFeatures::Evidence
        ) {
            // Task E5F: host-known bounds without a read (same rule
            // as the generation path; `spectra` was uploaded from the
            // exact donor-assembled batch, so its stored bound pairs each
            // uploaded peak with its uploaded row's ppm).
            let h_cap_max = match self.train.formula_source {
                super::contract::FormulaSource::Table => self.device_table.hydrogen_cap_max(),
                super::contract::FormulaSource::Enumerate => self
                    .model
                    .enum_artifacts
                    .as_ref()
                    .map(|a| a.hydrogen_cap_max())
                    .unwrap_or(u32::MAX),
            };
            let tol_max = spectra.uploaded_tol_max();
            Ms2Model::generate_search_evidence(
                &spectra,
                &self.device_table,
                &mut bucket.formula,
                &bucket.peaks,
                b,
                window_m,
                self.train.formula_evidence_work_max,
                self.train.formula_evidence_dispatch_max,
                h_cap_max,
                tol_max,
            )?;
        } else {
            ms2::count_features(
                &bucket.formula.cand.reshape(vec![b * window_m, 13])?,
                &self.device_table.log_table,
                &mut bucket.formula.cand_feat.reshape(vec![b * window_m, 10])?,
                13,
            )?;
        }
        let scored = self.model.formula.score(&bucket.formula, &encoded.pool)?;
        // V1 §1.2: `gold_counts` travel with the batch; `gold_slot` on the
        // device replaces the host window search (no host read, no host
        // search). `gold_slots_host` stays as this kernel's twin and for
        // tests.
        let gold_counts_t = IdTensor::from_slice(&prep.gold_counts, vec![b, 10], &self.device)?;
        let mut gold_slot_t = IdTensor::empty(vec![b], &self.device);
        ms2::gold_slot(&bucket.formula.cand, &gold_counts_t, &mut gold_slot_t)?;
        // Teacher forcing (V1 §1.2): `Composition` conditions on the head
        // embedding of the true parent composition (through
        // `count_features` and the head's row network), whether or not the
        // search scored it; `ScoredRowOrZero` (V0) conditions on the scored
        // row's embedding when scored, else zero. The gold row network is
        // launched only in `Composition` mode: the `ScoredRowOrZero` branch
        // of `condition_embeddings` gathers the scored embedding and never
        // calls `embed_rows` on the gold features, so no gold-network launch
        // happens there (proved by the `ms2.gold_embed` tally scope).
        let d_model = self.model.config.d_model as usize;
        let e_cond = condition_embeddings(
            &self.model.formula,
            &self.device_table.log_table,
            &self.device,
            self.train.gold_formula_conditioning,
            &scored,
            &gold_counts_t,
            &gold_slot_t,
            b,
            d_model,
        )?;
        // Fragment-ion assignment (architecture §2.3): only when
        // `lambda_assign > 0` and the model owns the head. Otherwise zero
        // launches, zero reads, bit-identical results.
        let (assign_loss, assign_counts, assign_overflow) = forward_assign(
            &self.model,
            &self.train,
            &self.device,
            &self.device_table.log_table,
            set,
            indices,
            &prep,
            &spectra,
            bucket,
            &encoded,
            b,
        )?;
        Ok(ForwardPrefix {
            prep,
            encoded,
            scored,
            gold_slot_t,
            e_cond,
            spectra: b,
            assign_loss,
            assign_counts,
            assign_overflow,
        })
    }

    /// Forward pass with an explicit peak choice: `use_donors` substitutes
    /// molecule-aware donor peaks, `false` keeps own peaks. Shared by the
    /// control-following [`Ms2Trainer::forward`] and the `--diagnose` peak
    /// sensitivity, which needs both inputs for every model.
    ///
    /// The conditioning embedding travels in the return value (last): it is
    /// the exact tensor handed to `decoder.teacher` below, and
    /// [`Ms2Trainer::forward_state`] keeps it in the returned forward state.
    #[allow(clippy::type_complexity)]
    fn forward_with_donors(
        &mut self,
        set: &ExperimentSet,
        indices: &[usize],
        use_donors: bool,
        compact: bool,
        train_jitter: bool,
    ) -> Result<(
        crate::models::ms2::decoder::TeacherOutput<R, E>,
        Var<R, E>,
        Var<R, E>,
        Var<R, E>,
        Var<R, E>,
        Var<R, E>,
        IdTensor<R>,
        Prepared,
        Var<R, E>,
        Option<Var<R, E>>,
        Option<Tensor<R, E>>,
        usize,
    )> {
        let window_m = self.train.formula_window as usize;
        let prefix = self.forward_prefix(set, indices, use_donors, window_m, train_jitter)?;
        let ForwardPrefix {
            prep,
            encoded,
            scored,
            gold_slot_t,
            e_cond,
            spectra: b,
            assign_loss,
            assign_counts,
            assign_overflow,
        } = prefix;
        let atoms = self.model.config.max_atoms as usize;
        let closures = self.model.config.max_ring_closures;
        let formula_loss = self.model.formula.loss(&scored, &gold_slot_t)?;
        // Scored-gold count without a read, for the report packing: the same
        // validity rule as the loss, reduced on the device.
        let gold_f_count: Tensor<R, E> = ids_to_float(&gold_slot_t);
        let is_absent_count =
            crate::tensor::ops::elemwise::eq_scalar(&gold_f_count, u32::MAX as f32);
        let valid_count = crate::tensor::ops::elemwise::rsub_scalar(&is_absent_count, 1.0);
        let present_var = Var::constant(valid_count).sum()?;
        // The training layouts leave out what the padded pass computes for
        // nothing: empty slots, and with `Ragged` the positions past a
        // trace's end. The evaluation paths (`compact == false`) keep the
        // padded layout, which has a result per slot.
        let slots = self.train.slots;
        let steps = self.limits.max_steps();
        let pass = if compact { teacher_pass() } else { TeacherPass::Padded };
        if pass == TeacherPass::Ragged {
            let host = prep.targets.pack(
                PACK_HORIZONS * steps,
                PACK_SCAN_BUCKET,
                PACK_ATTN_BUCKET,
                PACK_TRACE_BUCKET,
            );
            let rows = host.traces.spectra;
            let targets = host.traces.upload(&self.device)?;
            let packing = host.upload::<R, E>(&self.device)?;
            let buffers = Self::replay_for(&mut self.compact_replay, rows, steps, atoms, &self.device);
            ms2::grammar_replay(
                &targets.tokens,
                &targets.meta,
                &self.constants,
                atoms as u32,
                closures,
                buffers,
            )?;
            let replay = ReplayView {
                replay: &buffers.replay,
                atoms: &buffers.atoms,
            };
            let tout = self
                .model
                .decoder
                .teacher_packed(&encoded, &e_cond, &targets, &replay, &packing)?;
            // The divisor is the spectra of the batch.
            let graph = graph_loss(&tout, &targets.q, b)?;
            return self.finish_forward(
                tout, graph, formula_loss, scored, present_var, gold_slot_t, prep, e_cond,
                assign_loss, assign_counts, assign_overflow,
            );
        }
        let packed = if pass == TeacherPass::Slots && slots > COMPACT_GROUP {
            let (targets, owner) = prep.targets.compact(COMPACT_GROUP, COMPACT_BUCKET);
            (targets.spectra * targets.slots < b * slots).then_some((targets, owner))
        } else {
            None
        };
        let (tout, graph) = match packed {
            Some((host, owner)) => {
                let rows = host.spectra * host.slots;
                let targets = host.upload(&self.device)?;
                let owner = IdTensor::from_slice(&owner, vec![host.spectra], &self.device)?;
                let buffers =
                    Self::replay_for(&mut self.compact_replay, rows, steps, atoms, &self.device);
                ms2::grammar_replay(
                    &targets.tokens,
                    &targets.meta,
                    &self.constants,
                    atoms as u32,
                    closures,
                    buffers,
                )?;
                let replay = ReplayView {
                    replay: &buffers.replay,
                    atoms: &buffers.atoms,
                };
                let tout = self
                    .model
                    .decoder
                    .teacher_grouped(&encoded, &e_cond, &targets, &replay, &owner)?;
                // The divisor stays the spectra of the batch, not the
                // virtual ones.
                let graph = graph_loss(&tout, &targets.q, b)?;
                (tout, graph)
            }
            None => {
                let targets = prep.targets.upload(&self.device)?;
                let enum_p = match self.train.formula_source {
                    super::contract::FormulaSource::Table => 0,
                    super::contract::FormulaSource::Enumerate => {
                        self.model.enum_artifacts.as_ref().map(|a| a.p).unwrap_or(0)
                    }
                };
                // The same bucket the prefix used (same key, so this re-borrow
                // allocates nothing): only the replay buffers are still needed.
                let bucket = Self::bucket_for(
                    &mut self.buckets,
                    &self.device,
                    self.model.config.n_peaks as usize,
                    atoms,
                    self.limits.max_steps(),
                    b,
                    prep.n_raw,
                    slots,
                    window_m,
                    enum_p,
                    self.model.config.formula_features,
                )?;
                ms2::grammar_replay(
                    &targets.tokens,
                    &targets.meta,
                    &self.constants,
                    atoms as u32,
                    closures,
                    &bucket.replay,
                )?;
                let replay = ReplayView {
                    replay: &bucket.replay.replay,
                    atoms: &bucket.replay.atoms,
                };
                let tout = self
                    .model
                    .decoder
                    .teacher(&encoded, &e_cond, &targets, &replay)?;
                let graph = graph_loss(&tout, &targets.q, b)?;
                (tout, graph)
            }
        };
        self.finish_forward(
            tout, graph, formula_loss, scored, present_var, gold_slot_t, prep, e_cond,
            assign_loss, assign_counts, assign_overflow,
        )
    }

    /// The replay buffers of a compact teacher pass with `rows` rows, cached
    /// by row count (the eight most recent).
    fn replay_for<'a>(
        cache: &'a mut Vec<(usize, ReplayBuffers<R>)>,
        rows: usize,
        steps: usize,
        atoms: usize,
        device: &Device<R>,
    ) -> &'a ReplayBuffers<R> {
        if !cache.iter().any(|(r, _)| *r == rows) {
            cache.push((rows, ReplayBuffers::new(rows, steps, atoms, device)));
            while cache.len() > 8 {
                cache.remove(0);
            }
        }
        &cache
            .iter()
            .find(|(r, _)| *r == rows)
            .expect("the replay buffers just cached")
            .1
    }

    /// The total loss and the forward pass's return value, whatever layout
    /// the teacher pass took.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn finish_forward(
        &self,
        tout: TeacherOutput<R, E>,
        graph: Var<R, E>,
        formula_loss: Var<R, E>,
        scored: FormulaOutput<R, E>,
        present_var: Var<R, E>,
        gold_slot_t: IdTensor<R>,
        prep: Prepared,
        e_cond: Var<R, E>,
        assign_loss: Option<Var<R, E>>,
        assign_counts: Option<Tensor<R, E>>,
        assign_overflow: usize,
    ) -> Result<(
        TeacherOutput<R, E>,
        Var<R, E>,
        Var<R, E>,
        Var<R, E>,
        Var<R, E>,
        Var<R, E>,
        IdTensor<R>,
        Prepared,
        Var<R, E>,
        Option<Var<R, E>>,
        Option<Tensor<R, E>>,
        usize,
    )> {
        let mut total = graph.add(&formula_loss.mul_scalar(self.train.formula_weight))?;
        // Assignment term: `lambda_assign * L_assign` (pseudo-label, oracle
        // formula). Disabled means zero launches and bit-identical totals.
        if let Some(al) = assign_loss.as_ref() {
            total = total.add(&al.mul_scalar(self.train.lambda_assign))?;
        }
        let log_prob = scored.log_prob;
        Ok((
            tout,
            graph,
            formula_loss,
            total,
            log_prob,
            present_var,
            gold_slot_t,
            prep,
            e_cond,
            assign_loss,
            assign_counts,
            assign_overflow,
        ))
    }

    /// Test hook: the actual trainer conditioning for these spectra.
    ///
    /// Runs the same [`Ms2Trainer::forward_prefix`] that `step` and
    /// `teacher_eval` run and reads back the conditioning embedding (the
    /// exact tensor the production finish hands to `decoder.teacher`), the
    /// full scored embedding and the gold slots. Tests compare the
    /// in-window conditioning rows against the scored rows by `to_bits()`
    /// instead of reconstructing the forward pass. Performs device reads;
    /// test support only.
    #[doc(hidden)]
    pub fn conditioning_for_test(
        &mut self,
        set: &ExperimentSet,
        indices: &[usize],
    ) -> Result<TrainerConditioning> {
        let window_m = self.train.formula_window as usize;
        self.conditioning_for_test_with_window(set, indices, window_m)
    }

    /// Test hook with an explicit scored-candidate capacity `window_m`
    /// (32, 128, 512, 2048): the same [`Ms2Trainer::forward_prefix`] that
    /// `step` runs, at the requested M, so an M = 128 check exercises the
    /// actual trainer conditioning at another capacity. Test support only;
    /// performs device reads.
    #[doc(hidden)]
    pub fn conditioning_for_test_with_window(
        &mut self,
        set: &ExperimentSet,
        indices: &[usize],
        window_m: usize,
    ) -> Result<TrainerConditioning> {
        if !matches!(window_m, 32 | 128 | 512 | 2048) {
            return Err(Error::config(format!(
                "conditioning_for_test_with_window: window_m {window_m} is not one of 32, 128, 512, 2048"
            )));
        }
        let use_donors = self.train.control == Control::ShuffledSpectrum;
        let prefix = self.forward_prefix(set, indices, use_donors, window_m, false)?;
        let b = prefix.spectra;
        let d = self.model.config.d_model as usize;
        let (ids, floats) = read_all(
            &[&prefix.gold_slot_t],
            &[prefix.e_cond.tensor(), prefix.scored.embedding.tensor()],
        )?;
        Ok(TrainerConditioning {
            spectra: b,
            d_model: d,
            slots: ids[0].clone(),
            e_cond: floats[0].clone(),
            scored_embedding: floats[1].clone(),
            window_m,
        })
    }

    /// One optimizer step on these spectra.
    ///
    /// No device read happens unless [`Ms2Trainer::request_report`] was
    /// called, in which case exactly one batched read of
    /// `[L, L_graph, L_formula]` is performed (the pre-update losses) and the
    /// report is returned. The gradient clip uses the device-side scale, so
    /// clipping costs no read either way. Unlabeled spectra may appear in the
    /// batch: they contribute 0 to `L_graph` through their zero `q` weights,
    /// and the divisor is `B` including them.
    ///
    /// This is [`Ms2Trainer::step_with_boundaries`] with a no-op hook: there
    /// is one code path, so profiling the boundaries observes exactly what
    /// this runs.
    pub fn step(&mut self, set: &ExperimentSet, indices: &[usize]) -> Result<Option<LossReport>> {
        self.step_with_boundaries(set, indices, &mut |_| {})
    }

    /// One optimizer step with a hook at the three phase boundaries.
    ///
    /// Runs exactly what [`Ms2Trainer::step`] runs — forward pass, loss
    /// build and report packing ([`StepPhase::AfterForward`]), gradient
    /// computation with the device-side clip scale
    /// ([`StepPhase::AfterBackward`]), optimizer update
    /// ([`StepPhase::AfterOptimizer`]) — calling `on_boundary` after each
    /// phase, then the (read-free unless reported) step close-out. A driver
    /// profiling stage boundaries snapshots counters and synchronised wall
    /// time in the hook, so forward, backward and optimizer each get real
    /// launches, allocations and wall time instead of a `step − teacher_eval`
    /// subtraction (the teacher pass performs evaluation-only work and a
    /// device read `step` never does).
    pub fn step_with_boundaries(
        &mut self,
        set: &ExperimentSet,
        indices: &[usize],
        on_boundary: &mut dyn FnMut(StepPhase),
    ) -> Result<Option<LossReport>> {
        let fwd = self.forward_state(set, indices)?;
        on_boundary(StepPhase::AfterForward);
        let bwd = self.backward_state(&fwd)?;
        on_boundary(StepPhase::AfterBackward);
        self.optimizer_step(&bwd)?;
        on_boundary(StepPhase::AfterOptimizer);
        self.finish_step(fwd)
    }

    /// Run the forward pass with loss build and report packing, without any
    /// gradient or update: the first phase of [`Ms2Trainer::step`].
    ///
    /// Public so the device-timing harness can run one phase per
    /// `client.profile` span, stashing the returned state between spans (the
    /// backward phase consumes it). No device read.
    pub fn forward_state(
        &mut self,
        set: &ExperimentSet,
        indices: &[usize],
    ) -> Result<ForwardState<R, E>> {
        // The training forward pass: the teacher runs on the occupied slots
        // only (the evaluation passes, which read a result per slot, keep
        // the padded layout).
        let (tout, graph, formula_loss, total, _log_prob, present, _gold_slot, prep, e_cond, assign_loss, assign_counts, assign_overflow) =
            self.forward_with_donors(
                set,
                indices,
                self.train.control == Control::ShuffledSpectrum,
                true,
                true,
            )?;
        let b = indices.len();
        // Report read: `[L, L_graph, L_formula, present]` plus, when
        // assignment is enabled, `[L_assign, eligible, partial, dropped]` —
        // still a single batched `try_to_f32` (one read when reported, zero
        // otherwise). Disabled means the 4-scalar form, bit-identical to V0.
        let packed = match (assign_loss.as_ref(), assign_counts.as_ref()) {
            (Some(al), Some(cc)) => {
                let cc_var = Var::constant(cc.clone()).reshape(vec![3])?;
                cat(
                    &[
                        total.reshape(vec![1])?,
                        graph.reshape(vec![1])?,
                        formula_loss.reshape(vec![1])?,
                        present.reshape(vec![1])?,
                        al.reshape(vec![1])?,
                        cc_var,
                    ],
                    0,
                )?
            }
            _ => cat(
                &[
                    total.reshape(vec![1])?,
                    graph.reshape(vec![1])?,
                    formula_loss.reshape(vec![1])?,
                    present.reshape(vec![1])?,
                ],
                0,
            )?,
        };
        Ok(ForwardState {
            tout,
            graph,
            formula_loss,
            e_cond,
            assign_loss,
            total,
            packed,
            prep,
            spectra: b,
            assign_overflow,
        })
    }

    /// Compute gradients (and the device-side clip scale when configured)
    /// from a [`ForwardState`]: the second phase of [`Ms2Trainer::step`]. No
    /// device read.
    pub fn backward_state(&self, fwd: &ForwardState<R, E>) -> Result<BackwardState<R, E>> {
        let grads = fwd.total.backward_retain()?;
        let scale = match self.train.grad_clip {
            Some(max_norm) => grad_scale(&grads, max_norm, 1.0)?,
            None => None,
        };
        Ok(BackwardState { grads, scale })
    }

    /// Apply the optimizer update from a [`BackwardState`]: the third phase
    /// of [`Ms2Trainer::step`]. No device read.
    pub fn optimizer_step(&mut self, bwd: &BackwardState<R, E>) -> Result<()> {
        match &bwd.scale {
            Some(scale) => {
                self.optimizer
                    .step_scaled(&self.params, &bwd.grads, Some(&scale.factor))?;
            }
            None => {
                self.optimizer.step(&self.params, &bwd.grads)?;
            }
        }
        Ok(())
    }

    /// Close out a step: count it and, when [`Ms2Trainer::request_report`]
    /// was called, perform the single batched report read. This is the only
    /// phase that ever reads the device.
    fn finish_step(&mut self, fwd: ForwardState<R, E>) -> Result<Option<LossReport>> {
        self.steps += 1;
        if !self.report_pending {
            return Ok(None);
        }
        self.report_pending = false;
        let values = fwd.packed.try_to_f32()?;
        // V1 §1.2: the scored-gold count comes from the device-packed scalar
        // (inside the existing single read), never from a host search.
        let present = values[3].round().clamp(0.0, fwd.spectra as f32) as usize;
        let absent = fwd.spectra - present;
        let (assign, eligible, partial, dropped) = if values.len() >= 8 {
            (
                values[4],
                values[5].round().clamp(0.0, 1e9) as usize,
                values[6].round().clamp(0.0, 1e9) as usize,
                values[7].round().clamp(0.0, 1e9) as usize,
            )
        } else {
            (0.0, 0, 0, 0)
        };
        Ok(Some(LossReport {
            step: self.steps,
            loss: values[0],
            graph: values[1],
            formula: values[2],
            assign,
            spectra: fwd.spectra,
            formula_present: present,
            formula_absent: absent,
            gold_not_scored: absent,
            assign_eligible: eligible,
            assign_partial: partial,
            assign_dropped: dropped,
            assignment_label_overflow: fwd.assign_overflow,
        }))
    }

    /// Teacher-forced evaluation of these spectra: no gradient, one batched
    /// read per evaluated batch (the per-target NLLs and the gold
    /// log-probabilities travel in one [`read_all`]). Runs under the control
    /// the trainer was built with; under `ShuffledSpectrum` the peaks are
    /// molecule-aware donors (see [`Ms2Trainer::prepare`]) and the evaluation
    /// reports `donor_same_molecule` (0) and `donor_no_eligible_peaks`.
    pub fn teacher_eval(&mut self, set: &ExperimentSet, indices: &[usize]) -> Result<TeacherEval> {
        let _guard = no_grad();
        let (tout, _, _, _, log_prob, _, gold_slot_t, prep, _, _, _, _) = self.forward(set, indices)?;
        let b = indices.len();
        let slots = self.train.slots;
        // Gold log-probabilities without an extra read: the safe slot picks a
        // value, the absent rows are masked back to 0 on the device. The
        // `gold_slot` itself joins the existing single batched read, so a
        // warmed eval still performs exactly one runtime read per batch.
        let safe = safe_ids(&gold_slot_t, 0)?;
        let picked = log_prob.take_along_last(&safe)?;
        let gold_f: Tensor<R, E> = ids_to_float(&gold_slot_t);
        let is_absent = crate::tensor::ops::elemwise::eq_scalar(&gold_f, u32::MAX as f32);
        let valid_mask = crate::tensor::ops::elemwise::rsub_scalar(&is_absent, 1.0);
        let picked = picked.mul(&Var::constant(valid_mask))?;
        let (ids, floats) = read_all(&[&gold_slot_t], &[tout.nll.tensor(), picked.tensor()])?;
        let molecules = indices.iter().map(|&i| set.spectra[i].molecule).collect();
        Ok(TeacherEval {
            nll: floats[0].clone(),
            q: prep.targets.q.clone(),
            scored_tokens: prep.scored_tokens,
            gold_slot: ids[0].clone(),
            gold_log_prob: floats[1].clone(),
            spectra: b,
            slots,
            molecules,
            donor_same_molecule: prep.donor_same_molecule,
            donor_no_eligible_peaks: prep.donor_no_eligible_peaks,
        })
    }

    /// Assignment evaluation on these spectra (architecture §2.3, pseudo-label,
    /// oracle formula): `L_assign` under the true parent composition plus the
    /// eligible/partial/dropped counts and the fraction of anchored peaks
    /// whose labelled hypothesis has the largest probability
    /// (`assign_top1_pseudo_label`). Returns `(loss, eligible, partial,
    /// dropped, top1)` with `top1 = None` when no peak is eligible. Performs
    /// device reads (evaluation only, never on the training hot path).
    pub fn assign_eval(
        &mut self,
        set: &ExperimentSet,
        indices: &[usize],
    ) -> Result<(f32, usize, usize, usize, Option<f64>)> {
        let _guard = no_grad();
        if self.model.assignment.is_none() {
            return Err(Error::config(
                "Ms2Trainer::assign_eval: assignment disabled (no head)".to_string(),
            ));
        }
        let Some(cfg) = self.model.config.assignment.clone() else {
            return Err(Error::config(
                "Ms2Trainer::assign_eval: assignment disabled (no config)".to_string(),
            ));
        };
        if indices.is_empty() {
            return Err(Error::config(
                "Ms2Trainer::assign_eval: cannot evaluate an empty index list".to_string(),
            ));
        }
        // Reuse the training prefix for the exact production assignment path
        // (ion_assign on gold, labels, mask, log_prob, loss), then read back
        // what the report needs plus the rows for top-1.
        let window_m = self.train.formula_window as usize;
        let use_donors = self.train.control == Control::ShuffledSpectrum;
        let prefix = self.forward_prefix(set, indices, use_donors, window_m, false)?;
        let b = prefix.spectra;
        let (Some(al), Some(cc)) = (prefix.assign_loss, prefix.assign_counts) else {
            return Err(Error::config(
                "Ms2Trainer::assign_eval: assignment did not run (lambda_assign <= 0?)".to_string(),
            ));
        };
        // Read loss + counts (one batched read) for the NLL and denominators.
        let packed = cat(
            &[
                al.reshape(vec![1])?,
                Var::constant(cc.clone()).reshape(vec![3])?,
            ],
            0,
        )?;
        let values = packed.try_to_f32()?;
        let loss = values[0];
        let eligible = values[1].round().clamp(0.0, 1e9) as usize;
        let partial = values[2].round().clamp(0.0, 1e9) as usize;
        let dropped = values[3].round().clamp(0.0, 1e9) as usize;
        let Some(head) = self.model.assignment.as_ref() else {
            return Err(Error::config(
                "Ms2Trainer::assign_eval: assignment disabled (no head)".to_string(),
            ));
        };
        // Top-1 needs the per-peak winners: re-run the assignment host-side?
        // No: read the device log-probs/mask/state from a second assignment
        // pass that keeps them alive. For small validation batches the extra
        // work is negligible next to generation.
        //
        // To avoid duplicating the whole prefix, recompute the host twins for
        // top-1 from the already-read loss? Not enough. Instead, run a small
        // host-side top-1 using the twins and the head parameters read back?
        // Simpler and exact: read back the log_prob/mask/state device buffers
        // by re-running forward_assign's pieces here (same launches, plus
        // reads). The duplication is evaluation-only.
        let j = cfg.hypotheses as usize;
        let n = self.model.config.n_peaks as usize;
        let l_cap = cfg.labels as usize;
        // Rebuild what forward_assign built (same code path, plus reads).
        // Note: this duplicates launches; evaluation is not launch-pinned.
        let prep = prefix.prep;
        // Re-upload spectra to get meta/kept (same as prefix did).
        let spectra = DeviceSpectra::upload(&prep.spectra, &self.device)?;
        let bucket = Self::bucket_for(
            &mut self.buckets,
            &self.device,
            self.model.config.n_peaks as usize,
            self.model.config.max_atoms as usize,
            self.limits.max_steps(),
            b,
            prep.n_raw,
            self.train.slots,
            window_m,
            match self.train.formula_source {
                super::contract::FormulaSource::Table => 0,
                super::contract::FormulaSource::Enumerate => {
                    self.model.enum_artifacts.as_ref().map(|a| a.p).unwrap_or(0)
                }
            },
            self.model.config.formula_features,
        )?;
        // Re-encode to get x (same as prefix; extra launches, eval-only).
        let encode_control = if self.train.control == Control::ShuffledSpectrum {
            Control::None
        } else {
            self.train.control
        };
        let encoded = self.model.encoder.encode(&spectra, &bucket.peaks, encode_control)?;
        // Same shared uncertainty source as the training prefix.
        let spec_t = spectra.evidence_spec(&self.device)?;
        let top_counts_t = IdTensor::from_slice(&prep.gold_counts, vec![b, 1, 10], &self.device)?;
        let mut ion_t = IdTensor::empty(vec![b, 1, n, j, 12], &self.device);
        let mut ion_meta_t = IdTensor::empty(vec![b, 1, n, 4], &self.device);
        crate::tensor::ops::ms2_ion::ion_assign(
            &top_counts_t,
            &bucket.peaks.kept,
            &spectra.meta,
            &spec_t,
            &mut ion_t,
            &mut ion_meta_t,
            cfg.work_max,
        )?;
        let n_raw = prep.n_raw;
        let mut lab_host = vec![0u32; b * l_cap * 12];
        for (bi, &idx) in indices.iter().enumerate() {
            let Some(lbls) = set.spectra.get(idx).and_then(|s| s.labels.as_ref()) else {
                continue;
            };
            let adduct_id = prep.spectra.adduct[bi];
            let base = bi * n_raw;
            let count = (prep.spectra.peak_count[bi] as usize).min(n_raw);
            let raw_of = |pid: u32| -> Option<u32> {
                for k in 0..count {
                    if prep.spectra.peak_id[base + k] == pid {
                        return Some(k as u32);
                    }
                }
                None
            };
            let sets = super::ion::ion_labels(lbls, adduct_id, raw_of, l_cap);
            for (li, lab) in sets.labels.iter().enumerate() {
                let lb = (bi * l_cap + li) * 12;
                lab_host[lb] = lab.raw_index;
                for e in 0..10 {
                    lab_host[lb + 1 + e] = u32::from(lab.counts[e]);
                }
                lab_host[lb + 11] = 1;
            }
        }
        let lab_t = IdTensor::from_slice(&lab_host, vec![b, l_cap, 12], &self.device)?;
        let mut mask_t = Tensor::<R, E>::empty(vec![b, n, j + 1], &self.device);
        let mut state_t = IdTensor::empty(vec![b, n], &self.device);
        crate::tensor::ops::ms2_ion::ion_label_mask(
            &lab_t,
            &ion_t,
            &ion_meta_t,
            &bucket.peaks.kept,
            &mut mask_t,
            &mut state_t,
            0,
        )?;
        let out = head.log_prob(&self.model.formula, &self.device_table.log_table, &ion_t, &ion_meta_t, &encoded.x)?;
        let lp_h = out.log_prob.try_to_f32()?;
        let mask_h = mask_t.to_f32();
        let (state_ids, _) = read_all::<R, E>(&[&state_t], &[])?;
        let state_h = &state_ids[0];
        // Fraction of eligible peaks whose labelled hypothesis (mask) holds
        // the largest probability (ties count as success when a labelled class
        // ties for the max; unassigned-only rows are never eligible).
        let width = j + 1;
        let mut top = 0usize;
        let mut elig = 0usize;
        for bi in 0..b {
            for pi in 0..n {
                let row = bi * n + pi;
                if state_h[row] != 1 {
                    continue;
                }
                elig += 1;
                let base = row * width;
                let mut best = f32::NEG_INFINITY;
                for c in 0..width {
                    let v = lp_h[base + c];
                    if v > best {
                        best = v;
                    }
                }
                // Any labelled class attaining the max counts.
                let mut hit = false;
                for c in 0..width {
                    if mask_h[base + c] != 0.0 && lp_h[base + c] == best {
                        // Exclude the degenerate case where the max is shared
                        // with an unlabelled class but a labelled class also
                        // attains it: still a success (labelled hypothesis is
                        // among the winners).
                        hit = true;
                        break;
                    }
                }
                if hit {
                    top += 1;
                }
            }
        }
        let top1 = if elig == 0 { None } else { Some(top as f64 / elig as f64) };
        Ok((loss, eligible, partial, dropped, top1))
    }

    /// Teacher-forced field detail for `--diagnose`: the same forward pass
    /// with an explicit peak choice plus a read-back of the per-field
    /// log-probabilities.
    ///
    /// `use_donors` substitutes molecule-aware donor peaks (section 1
    /// donors under the trainer seed); `false` keeps the spectra's own peaks.
    /// Everything else (metadata, targets, formula conditioning) is fixed, so
    /// the paired own/donor comparison is a pure peak-sensitivity check.
    pub fn teacher_field_eval(
        &mut self,
        set: &ExperimentSet,
        indices: &[usize],
        use_donors: bool,
    ) -> Result<TeacherFieldEval> {
        let _guard = no_grad();
        let (tout, _, _, _, _, _, _, prep, _, _, _, _) =
            self.forward_with_donors(set, indices, use_donors, false, false)?;
        let b = indices.len();
        let slots = self.train.slots;
        let t = self.limits.max_steps();
        let (_, floats) = read_all(&[], &[tout.nll.tensor(), tout.field_log_prob.tensor()])?;
        let molecules = indices.iter().map(|&i| set.spectra[i].molecule).collect();
        Ok(TeacherFieldEval {
            nll: floats[0].clone(),
            q: prep.targets.q.clone(),
            scored_tokens: prep.scored_tokens,
            field_log_prob: floats[1].clone(),
            use_mask: prep.targets.use_mask.clone(),
            tokens: prep.targets.tokens.clone(),
            max_steps: t,
            spectra: b,
            slots,
            molecules,
            donor_same_molecule: prep.donor_same_molecule,
            donor_no_eligible_peaks: prep.donor_no_eligible_peaks,
        })
    }

    /// Evidence diagnostics of these spectra (architecture §1.6, evaluation
    /// only).
    ///
    /// Returns `None` under `FormulaFeatures::Counts` (no extra work, no
    /// read). Under `Evidence` runs the training forward prefix at the
    /// stored precursor and performs one extra device read of `cand_ev`
    /// with the gold slot, the search counters and the valid flags of
    /// `ev_peaks` in the same batched [`read_all`], then aggregates on the
    /// host: the fraction of scored candidates with `complete = 0`, the
    /// mean evidence-peak count per spectrum (from the valid `ev_peaks`
    /// flags, independently of candidate support), and — over spectra whose
    /// gold formula is in the scored support — the mean explained-count
    /// fraction of the gold slot next to the mean over the other scored
    /// slots. A warmed non-report training step still reads nothing (this
    /// method is never called there).
    pub fn evidence_diagnostics(
        &mut self,
        set: &ExperimentSet,
        indices: &[usize],
    ) -> Result<Option<EvidenceDiagnostics>> {
        if self.model.config.formula_features != FormulaFeatures::Evidence {
            return Ok(None);
        }
        if indices.is_empty() {
            return Err(Error::config(
                "Ms2Trainer::evidence_diagnostics: cannot evaluate an empty index list".to_string(),
            ));
        }
        let _guard = no_grad();
        let window_m = self.train.formula_window as usize;
        let use_donors = self.train.control == Control::ShuffledSpectrum;
        let prefix = self.forward_prefix(set, indices, use_donors, window_m, false)?;
        let b = prefix.spectra;
        let enum_p = match self.train.formula_source {
            super::contract::FormulaSource::Table => 0,
            super::contract::FormulaSource::Enumerate => {
                self.model.enum_artifacts.as_ref().map(|a| a.p).unwrap_or(0)
            }
        };
        let bucket = Self::bucket_for(
            &mut self.buckets,
            &self.device,
            self.model.config.n_peaks as usize,
            self.model.config.max_atoms as usize,
            self.limits.max_steps(),
            b,
            prefix.prep.n_raw,
            self.train.slots,
            window_m,
            enum_p,
            self.model.config.formula_features,
        )?;
        let Some(cand_ev) = bucket.formula.cand_ev.as_ref() else {
            return Err(Error::shape(
                "Ms2Trainer::evidence_diagnostics: Evidence layout needs cand_ev".to_string(),
            ));
        };
        let Some(ev_peaks) = bucket.formula.ev_peaks.as_ref() else {
            return Err(Error::shape(
                "Ms2Trainer::evidence_diagnostics: Evidence layout needs ev_peaks".to_string(),
            ));
        };
        // One batched diagnostic read: gold slots, counters, the valid flags
        // of `ev_peaks` and the candidate evidence. The evidence-peak count
        // per spectrum comes from the valid flags alone, independently of
        // candidate support; the candidate-based denominators below are
        // unchanged.
        let (ids, floats) =
            read_all(&[&prefix.gold_slot_t, &bucket.formula.counters, ev_peaks], &[cand_ev])?;
        let gold_slot = &ids[0];
        let counters = &ids[1];
        let ev_ids = &ids[2];
        let ev = &floats[0];
        // Valid evidence peaks per spectrum: `ev_peaks [B, P, 4]` carries the
        // valid flag in word 3 (`1` valid, `0` padding).
        let stride = if b == 0 { 0 } else { ev_ids.len() / b };
        debug_assert_eq!(stride % 4, 0);
        let mut valid_counts = vec![0usize; b];
        for s in 0..b {
            let mut n = 0usize;
            for w in (3..stride).step_by(4) {
                if ev_ids[s * stride + w] == 1 {
                    n += 1;
                }
            }
            valid_counts[s] = n;
        }
        let mut out = EvidenceDiagnostics {
            scored: 0,
            incomplete: 0,
            spectra: b,
            peaks_sum: 0.0,
            gold_sum: 0.0,
            gold_spectra: 0,
            other_sum: 0.0,
            other_slots: 0,
        };
        for s in 0..b {
            let scored = counters[s * 5 + 2].min(window_m as u32) as usize;
            // Evidence peaks independent of candidate support: the valid
            // `ev_peaks` flags counted above (an empty-support spectrum still
            // contributes its selected peaks). The candidate-based
            // denominators below keep their existing definitions.
            out.peaks_sum += valid_counts[s] as f64;
            let nev = if scored > 0 { ev[(s * window_m) * 4 + 2] } else { 0.0 };
            for m in 0..scored {
                let base = (s * window_m + m) * 4;
                out.scored += 1;
                if ev[base + 3] == 0.0 {
                    out.incomplete += 1;
                }
            }
            let gold = gold_slot[s];
            if gold != u32::MAX && (gold as usize) < scored {
                let gb = (s * window_m + gold as usize) * 4;
                let expl = ev[gb];
                let frac = if nev > 0.0 { f64::from(expl / nev) } else { 0.0 };
                out.gold_sum += frac;
                out.gold_spectra += 1;
                for m in 0..scored {
                    if m == gold as usize {
                        continue;
                    }
                    let ob = (s * window_m + m) * 4;
                    let ofrac = if nev > 0.0 {
                        f64::from(ev[ob] / nev)
                    } else {
                        0.0
                    };
                    out.other_sum += ofrac;
                    out.other_slots += 1;
                }
            }
        }
        Ok(Some(out))
    }

    /// Generation evaluation of these spectra: `generate` under `config` plus
    /// [`evaluate_candidates`], with formula recall set per spectrum from the
    /// candidate formula rows against the gold row of the parent composition
    /// in the table. A gold composition absent from the table counts as a
    /// miss (`Some(false)`); the caller counts such spectra separately from
    /// [`ExperimentSet`] and the table.
    ///
    /// Under `ShuffledSpectrum` (the trainer control or the generation
    /// control) the batch carries molecule-aware donors (section 1) and is
    /// passed as an ordinary request with [`Control::None`], so `generate`'s
    /// in-batch rotation is never applied a second time. This lets stored
    /// pilot checkpoints be re-evaluated without retraining.
    pub fn generate_eval(
        &self,
        set: &ExperimentSet,
        indices: &[usize],
        config: &GenerationConfig,
    ) -> Result<Vec<SpectrumEval>> {
        Ok(self.generate_eval_with_work(set, indices, config)?.0)
    }

    /// Generate candidates for these spectra (evaluation helper for evidence
    /// metrics): the same request batch as [`Ms2Trainer::generate_eval`]
    /// through [`Ms2Model::generate`], returning the validated
    /// [`CandidateBatch`] with evidence fields when `evidence` is on.
    pub fn generate_candidates(
        &self,
        set: &ExperimentSet,
        indices: &[usize],
        config: &GenerationConfig,
    ) -> Result<super::contract::CandidateBatch> {
        let _guard = no_grad();
        let (batch, gen_control) = self.eval_request_batch(set, indices, config)?;
        let mut workspace = self.workspace.borrow_mut();
        let out = self.model.generate(
            &batch,
            &self.device_table,
            &gen_control,
            &mut workspace,
            &self.constants,
        )?;
        Ok(out)
    }

    /// The request batch of [`generate_eval`]: molecule-aware donor peaks
    /// under `ShuffledSpectrum` (passed as an ordinary request with
    /// [`Control::None`]), else the plain batch. Shared by the unpacked and
    /// packed evaluation paths so both generate the same spectra.
    fn eval_request_batch(
        &self,
        set: &ExperimentSet,
        indices: &[usize],
        config: &GenerationConfig,
    ) -> Result<(SpectrumBatch, GenerationConfig)> {
        let shuffled = self.train.control == Control::ShuffledSpectrum
            || config.control == Control::ShuffledSpectrum;
        if shuffled {
            let donors = {
                let map = set.donor_map(self.train.seed)?;
                indices
                    .iter()
                    .map(|&i| {
                        map.get(i).copied().ok_or_else(|| {
                            Error::config(format!(
                                "Ms2Trainer::generate_eval: spectrum index {i} outside donor map"
                            ))
                        })
                    })
                    .collect::<Result<Vec<usize>>>()?
            };
            let n_raw = self.bucket_n_raw_with_donors(set, indices, &donors)?;
            let batch = spectrum_batch_with_donors(set, indices, &donors, n_raw as u32)?;
            let mut cfg = config.clone();
            cfg.control = Control::None;
            Ok((batch, cfg))
        } else {
            let n_raw = self.bucket_n_raw(set, indices)?;
            let batch = spectrum_batch_for(set, indices, n_raw as u32)?;
            Ok((batch, config.clone()))
        }
    }

    /// [`generate_eval`], plus the dispatch work of the generation call
    /// (V1 §3.3): one [`GenerationWork`] for the batch, so the experiment
    /// driver can record `generation_work` aggregates in its report without
    /// re-running generation.
    ///
    /// [`generate_eval`]: Ms2Trainer::generate_eval
    /// [`GenerationWork`]: super::contract::GenerationWork
    pub fn generate_eval_with_work(
        &self,
        set: &ExperimentSet,
        indices: &[usize],
        config: &GenerationConfig,
    ) -> Result<(Vec<SpectrumEval>, super::contract::GenerationWork, Vec<u32>)> {
        let _guard = no_grad();
        let (batch, gen_control) = self.eval_request_batch(set, indices, config)?;
        let mut workspace = self.workspace.borrow_mut();
        let candidates = self.model.generate(
            &batch,
            &self.device_table,
            &gen_control,
            &mut workspace,
            &self.constants,
        )?;
        drop(workspace);
        let work = candidates.work();
        let k = gen_control.trajectories as usize;
        let mut evals = evaluate_candidates(set, indices, &candidates, TRAIN_EVAL_WORK_LIMIT)?;
        for (pos, eval) in evals.iter_mut().enumerate() {
            let entry = &set.spectra[indices[pos]];
            let src = candidates
                .formula_source
                .get(pos)
                .copied()
                .unwrap_or(0);
            let hit = if src == 1 {
                let gold = entry.parent_composition;
                (0..k).any(|kk| {
                    let r = pos * k + kk;
                    candidates.formula_rank[r] != u32::MAX
                        && candidates.formula_counts[r * 10..r * 10 + 10]
                            .iter()
                            .zip(gold.iter())
                            .all(|(&a, &b)| a == b)
                })
            } else {
                let gold = self.gold_row(&entry.parent_composition);
                gold != u32::MAX
                    && (0..k).any(|kk| candidates.formula_row[pos * k + kk] == gold)
            };
            eval.formula_recall = Some(hit);
        }
        // D5: per-spectrum request statuses for exhaustion statistics, from the
        // same donor-path evaluation (no extra probe).
        let request_status = candidates.request_status.clone();
        Ok((evals, work, request_status))
    }

    /// Packed generation evaluation of these spectra (V1 §4.4, driver
    /// support): `generate` plus `generate_packed` over the same request
    /// batch, with the packed top-R re-inflated through
    /// [`to_candidate_batch`](super::pack::PackedCandidateBatch::to_candidate_batch)
    /// for `evaluate_candidates` precision/coverage at R.
    ///
    /// Returns `(packed_evals, packed_batch, dup_graph_rate,
    /// unresolved_rate, work, request_status)`. The rates are fractions over
    /// the unpacked batch's non-`request_failed` records: `duplicate_graph`
    /// (bit 7) and `identity_unresolved` (bit 8). The packed evals'
    /// `formula_recall` is set per spectrum by composition match of the
    /// packed formulas against the gold parent composition (source-
    /// independent; a spectrum with no packed formula is a miss).
    pub fn generate_eval_packed(
        &self,
        set: &ExperimentSet,
        indices: &[usize],
        config: &GenerationConfig,
    ) -> Result<(
        Vec<SpectrumEval>,
        super::pack::PackedCandidateBatch,
        f64,
        f64,
        super::contract::GenerationWork,
        Vec<u32>,
    )> {
        let _guard = no_grad();
        let (batch, gen_control) = self.eval_request_batch(set, indices, config)?;
        let mut workspace = self.workspace.borrow_mut();
        let candidates = self.model.generate(
            &batch,
            &self.device_table,
            &gen_control,
            &mut workspace,
            &self.constants,
        )?;
        let packed = self.model.generate_packed(
            &batch,
            &self.device_table,
            &gen_control,
            &mut workspace,
            &self.constants,
        )?;
        drop(workspace);
        let work = candidates.work();
        let request_status = candidates.request_status.clone();
        // Duplicate-graph and unresolved rates over the unpacked batch's
        // live records.
        let mut live = 0usize;
        let mut dup = 0usize;
        let mut unres = 0usize;
        for st in &candidates.status {
            if st & super::contract::candidate_status::REQUEST_FAILED != 0 {
                continue;
            }
            live += 1;
            if st & super::contract::candidate_status::DUPLICATE_GRAPH != 0 {
                dup += 1;
            }
            if st & super::contract::candidate_status::IDENTITY_UNRESOLVED != 0 {
                unres += 1;
            }
        }
        let dup_rate = if live == 0 { 0.0 } else { dup as f64 / live as f64 };
        let unres_rate = if live == 0 { 0.0 } else { unres as f64 / live as f64 };
        // Packed top-R evals through the re-inflated trajectory-ordered
        // view (trajectories = R).
        let reinflated = packed.to_candidate_batch()?;
        let mut evals = evaluate_candidates(set, indices, &reinflated, TRAIN_EVAL_WORK_LIMIT)?;
        let r = reinflated.trajectories;
        for (pos, eval) in evals.iter_mut().enumerate() {
            let entry = &set.spectra[indices[pos]];
            let gold = entry.parent_composition;
            let hit = (0..r).any(|rr| {
                let s = pos * r + rr;
                packed.formula_rank[s] != u32::MAX
                    && packed.formula_counts[s * 10..s * 10 + 10]
                        .iter()
                        .zip(gold.iter())
                        .all(|(&a, &b)| a == b)
            });
            eval.formula_recall = Some(hit);
        }
        Ok((evals, packed, dup_rate, unres_rate, work, request_status))
    }

    /// Save the weights, the model and train configs and the table reference.
    ///
    /// The optimizer state is not stored in V0: [`Ms2Trainer::load`] rebuilds
    /// a fresh AdamW (bias correction restarts), which is why this says so in
    /// the file too.
    pub fn save(&self, path: &Path) -> Result<()> {
        let checkpoint = Checkpoint {
            schema_version: 1,
            note: "V0 checkpoint: weights, model config, table reference and train config. The optimizer state is not stored in V0; load rebuilds a fresh AdamW."
                .to_string(),
            model_config: self.model.config.clone(),
            train_config: self.train.clone(),
            table_rows: self.device_table.rows as u32,
            table_sha256: self.device_table.sha256.clone(),
            steps: self.steps,
            weights: self.model.state_dict(),
            domain_json: self
                .model
                .enum_artifacts
                .as_ref()
                .map(|a| a.domain_json.clone()),
            bounds_json: self
                .model
                .enum_artifacts
                .as_ref()
                .map(|a| a.bounds_json.clone()),
        };
        let text = serde_json::to_string_pretty(&checkpoint)?;
        std::fs::write(path, text)?;
        Ok(())
    }

    /// Load a checkpoint saved by [`Ms2Trainer::save`] onto `device`.
    ///
    /// `table` must be the same formula table the checkpoint was saved with
    /// (row count and SHA-256 are both checked); the optimizer is fresh, not
    /// restored (see [`Ms2Trainer::save`]).
    pub fn load(path: &Path, table: &FormulaTable, device: &Device<R>) -> Result<Self> {
        let text = std::fs::read_to_string(path)?;
        let checkpoint: Checkpoint = serde_json::from_str(&text)?;
        let model_config = checkpoint.model_config.clone();
        Self::load_inner(checkpoint, &model_config, table, device)
    }

    /// Load a checkpoint saved by [`Ms2Trainer::save`] with an explicit
    /// model config instead of the checkpoint's own.
    ///
    /// Everything [`Ms2Trainer::load`] checks still applies; additionally a
    /// `formula_features` mismatch between the checkpoint and `model_config`
    /// is [`Error::Config`] naming `formula_features` (an `Evidence`
    /// checkpoint into a `Counts` config or the reverse never loads
    /// silently).
    pub fn load_with_config(
        path: &Path,
        model_config: &ModelConfig,
        table: &FormulaTable,
        device: &Device<R>,
    ) -> Result<Self> {
        let text = std::fs::read_to_string(path)?;
        let checkpoint: Checkpoint = serde_json::from_str(&text)?;
        if checkpoint.model_config.formula_features != model_config.formula_features {
            return Err(Error::config(format!(
                "Ms2Trainer::load_with_config: checkpoint formula_features {:?} does not match model config formula_features {:?} (an Evidence checkpoint into a Counts config or the reverse is refused)",
                checkpoint.model_config.formula_features, model_config.formula_features
            )));
        }
        Self::load_inner(checkpoint, model_config, table, device)
    }

    /// Shared body of [`Ms2Trainer::load`] and
    /// [`Ms2Trainer::load_with_config`]: the checkpoint is already parsed and
    /// the `formula_features` layout already matches `model_config`.
    fn load_inner(
        checkpoint: Checkpoint,
        model_config: &ModelConfig,
        table: &FormulaTable,
        device: &Device<R>,
    ) -> Result<Self> {
        if checkpoint.schema_version != 1 {
            return Err(Error::config(format!(
                "Ms2Trainer::load: unknown schema_version {} (expected 1)",
                checkpoint.schema_version
            )));
        }
        let uploaded = DeviceFormulaTable::<R, E>::upload(table, device)?;
        if uploaded.rows as u32 != checkpoint.table_rows {
            return Err(Error::config(format!(
                "Ms2Trainer::load: table has {} rows but the checkpoint names {}",
                uploaded.rows, checkpoint.table_rows
            )));
        }
        if uploaded.sha256 != checkpoint.table_sha256 {
            return Err(Error::config(format!(
                "Ms2Trainer::load: table sha256 {} does not match the checkpoint {}",
                uploaded.sha256, checkpoint.table_sha256
            )));
        }
        if checkpoint.model_config.formula_table.rows != checkpoint.table_rows
            || checkpoint.model_config.formula_table.sha256 != checkpoint.table_sha256
        {
            return Err(Error::config(
                "Ms2Trainer::load: the checkpoint's model config names a different table than its table reference"
                    .to_string(),
            ));
        }
        // D8: a checkpoint whose training source is Enumerate (or whose
        // ModelConfig names formula artifacts) without both artifact JSONs is
        // Error::Config at load (the first forward would otherwise fail for
        // missing resident artifacts).
        let needs_artifacts = matches!(
            checkpoint.train_config.formula_source,
            super::contract::FormulaSource::Enumerate
        ) || checkpoint.model_config.formula_artifacts.is_some();
        if needs_artifacts
            && (checkpoint.domain_json.is_none() || checkpoint.bounds_json.is_none())
        {
            return Err(Error::config(
                "Ms2Trainer::load: checkpoint trains with Enumerate (or names formula_artifacts) but lacks both artifact JSONs".to_string(),
            ));
        }
        let mut trainer = Self::new(
            model_config,
            table,
            &checkpoint.train_config,
            device,
        )?;
        if let Err(e) = trainer.model.load_state_dict(&checkpoint.weights, true) {
            // A head-less checkpoint into an assignment config must be
            // `Error::Config` naming the missing parameters (no silent random
            // initialisation). `load_state_dict` reports `Error::StateDict`
            // with `missing entry for ...`; map assignment misses to Config.
            let msg = e.to_string();
            if trainer.model.assignment.is_some() && msg.contains("missing entry for `assignment") {
                return Err(Error::config(format!(
                    "Ms2Trainer::load: checkpoint lacks assignment parameters for an assignment config (no silent random initialisation): {msg}"
                )));
            }
            // A layout mismatch across `formula_features` must be
            // `Error::Config` naming it: an `Evidence` checkpoint into a
            // `Counts` model (or the reverse) leaves `evidence_in` /
            // `evidence_out` unexpectedly present or missing.
            if msg.contains("evidence_in") || msg.contains("evidence_out") {
                return Err(Error::config(format!(
                    "Ms2Trainer::load: checkpoint formula_features layout does not match the model config (evidence_in/evidence_out present or missing): {msg}"
                )));
            }
            return Err(e);
        }
        trainer.steps = checkpoint.steps;
        match (&checkpoint.domain_json, &checkpoint.bounds_json) {
            (Some(domain_json), Some(bounds_json)) => {
                let domain: super::formula_enum::EnumDomain =
                    serde_json::from_str(domain_json).map_err(|e| {
                        Error::config(format!(
                            "Ms2Trainer::load: cannot parse checkpoint domain_json: {e}"
                        ))
                    })?;
                let bounds: super::formula_enum::RatioBounds =
                    serde_json::from_str(bounds_json).map_err(|e| {
                        Error::config(format!(
                            "Ms2Trainer::load: cannot parse checkpoint bounds_json: {e}"
                        ))
                    })?;
                let artifacts = super::formula_head::DeviceEnumArtifacts::upload(
                    &domain, &bounds, device,
                )?;
                trainer.model.set_enum_artifacts(artifacts)?;
            }
            (None, None) => {}
            _ => {
                return Err(Error::config(
                    "Ms2Trainer::load: checkpoint has only one of domain_json/bounds_json (expected both or neither)".to_string(),
                ));
            }
        }
        Ok(trainer)
    }
}

/// A saved [`Ms2Trainer`]: weights plus the configs and table reference that
/// bind them. The optimizer state is deliberately absent in V0.
#[derive(Serialize, Deserialize)]
struct Checkpoint {
    /// Checkpoint schema version (1).
    schema_version: u32,
    /// Human-readable note, including the missing optimizer state.
    note: String,
    /// Model hyperparameters (with the table reference).
    model_config: ModelConfig,
    /// Training hyperparameters.
    train_config: TrainConfig,
    /// Formula-table row count.
    table_rows: u32,
    /// SHA-256 of the formula-table JSON.
    table_sha256: String,
    /// Optimizer steps completed when saved.
    steps: u64,
    /// Model weights by parameter path.
    weights: StateDict,
    /// Enum domain JSON (`None` for table-only checkpoints).
    #[serde(default)]
    domain_json: Option<String>,
    /// Ratio bounds JSON (`None` for table-only checkpoints).
    #[serde(default)]
    bounds_json: Option<String>,
}
