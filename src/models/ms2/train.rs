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

use cubecl::prelude::Runtime;
use serde::{Deserialize, Serialize};

use crate::autograd::{Var, cat, no_grad};
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
use crate::train::optim::{AdamW, AdamWConfig, Optimizer, grad_scale};

use super::batch::DeviceSpectra;
use super::chem::Composition;
use super::contract::{Control, GenerationConfig, ModelConfig, SpectrumBatch};
use super::decoder::{ReplayView, graph_loss};
use super::experiment::{
    ExperimentSet, donor_stats, spectrum_batch_for, spectrum_batch_with_donors, target_batch_for,
};
use super::formula::{FormulaTable, WindowQuery};
use super::formula_head::{DeviceFormulaTable, gold_slots_host};
use super::generate::{GENERATION_WINDOW_M, GenerationWorkspace, Ms2Model};
use super::grammar::Limits;
use super::metrics::{SpectrumEval, evaluate_candidates};
use super::targets_batch::TargetBatch;

/// Device window width `M` of a training batch (architecture §2).
pub const TRAIN_WINDOW_M: usize = GENERATION_WINDOW_M;

/// Formula rows scored per training batch: unbounded on the host side, so the
/// device window width `M` is the only cap, exactly as
/// [`gold_slots_host`] assumes.
pub const TRAIN_ROWS_SCORED_MAX: u32 = 4096;

/// Containment work limit of [`Ms2Trainer::generate_eval`].
pub const TRAIN_EVAL_WORK_LIMIT: usize = 1_000_000;

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
}

impl Default for TrainConfig {
    /// The V0 defaults: batch 16, 16 slots, lr 3e-4, decay 0.1, formula
    /// weight 0.2, seed 1, no control, no clip.
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
        Ok(())
    }
}

/// One reported optimizer step: the pre-update losses with the formula
/// coverage that produced them.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct LossReport {
    /// Optimizer steps completed after this step (1-based).
    pub step: u64,
    /// `L = L_graph + formula_weight * L_formula` before the update.
    pub loss: f32,
    /// Graph loss before the update.
    pub graph: f32,
    /// Formula loss before the update.
    pub formula: f32,
    /// Spectra in the step (the divisor of `L_graph`, labeled or not).
    pub spectra: usize,
    /// Spectra whose gold formula was scored (the `L_formula` divisor).
    pub formula_present: usize,
    /// Spectra whose gold formula was absent from the window (`L_formula` 0).
    pub formula_absent: usize,
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
    /// Bucket key: batch, raw peak capacity.
    key: (usize, usize),
    /// Peak-selection scratch for `(B, n_raw, N)`.
    peaks: PeakBuffers<R, E>,
    /// Formula window for `(B, M)`.
    formula: FormulaBuffers<R, E>,
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
    /// Gold window slot per spectrum (`u32::MAX` when absent).
    gold_slots: Vec<u32>,
    /// Scored positions per target, from the `use` mask.
    scored_tokens: Vec<u32>,
    /// Raw peak capacity the batch was bucketed to.
    n_raw: usize,
    /// Donor rows of the same molecule (0 for molecule-aware donors).
    donor_same_molecule: usize,
    /// Recipients with no eligible donor peak.
    donor_no_eligible_peaks: usize,
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
    /// Cached buffer buckets by `(batch, n_raw)`, oldest first.
    buckets: Vec<TrainBucket<R, E>>,
    /// Preallocated generation state for [`Ms2Trainer::generate_eval`].
    workspace: RefCell<GenerationWorkspace<R, E>>,
    /// Whether the next [`Ms2Trainer::step`] reports its losses.
    report_pending: bool,
    /// Optimizer steps completed so far.
    steps: u64,
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
            workspace: RefCell::new(GenerationWorkspace::new()),
            report_pending: false,
            steps: 0,
        })
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
    ) -> Result<Prepared> {
        if indices.is_empty() {
            return Err(Error::config(
                "Ms2Trainer: cannot prepare an empty index list".to_string(),
            ));
        }
        let (spectra, donor_same_molecule, donor_no_eligible_peaks, n_raw) = if use_donors {
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
        let b = indices.len();
        let slots = self.train.slots;
        let targets = target_batch_for(set, indices, slots, self.limits)?;
        let queries: Vec<WindowQuery> = (0..b)
            .map(|i| WindowQuery {
                precursor_mz: spectra.precursor_mz_udalton[i],
                adduct: spectra.adduct[i],
                ppm_tenths: spectra.precursor_tolerance(i),
                precursor_uncertainty: spectra.precursor_uncertainty_udalton[i],
                rows_visited_max: u32::MAX,
                rows_scored_max: TRAIN_ROWS_SCORED_MAX,
            })
            .collect();
        let gold_rows: Vec<u32> = indices
            .iter()
            .map(|&i| self.gold_row(&set.spectra[i].parent_composition))
            .collect();
        let gold_slots = gold_slots_host(&self.table, &queries, &gold_rows, TRAIN_WINDOW_M);
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
            gold_slots,
            scored_tokens,
            n_raw,
            donor_same_molecule,
            donor_no_eligible_peaks,
        })
    }

    /// The buffer bucket for this batch shape, allocating on a miss (and
    /// evicting the oldest past four shapes, as in generation). A free
    /// function so the bucket borrow never overlaps the model borrow in
    /// [`Ms2Trainer::forward`].
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
    ) -> Result<&'a mut TrainBucket<R, E>> {
        let key = (batch, n_raw);
        if let Some(pos) = buckets.iter().position(|bk| bk.key == key) {
            return Ok(&mut buckets[pos]);
        }
        buckets.push(TrainBucket {
            key,
            peaks: PeakBuffers::new(batch, n_raw, n_peaks, device),
            formula: FormulaBuffers::new(batch, TRAIN_WINDOW_M, 1, device),
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
        Prepared,
    )> {
        self.forward_with_donors(
            set,
            indices,
            self.train.control == Control::ShuffledSpectrum,
        )
    }

    /// Forward pass with an explicit peak choice: `use_donors` substitutes
    /// molecule-aware donor peaks, `false` keeps own peaks. Shared by the
    /// control-following [`Ms2Trainer::forward`] and the `--diagnose` peak
    /// sensitivity, which needs both inputs for every model.
    #[allow(clippy::type_complexity)]
    fn forward_with_donors(
        &mut self,
        set: &ExperimentSet,
        indices: &[usize],
        use_donors: bool,
    ) -> Result<(
        crate::models::ms2::decoder::TeacherOutput<R, E>,
        Var<R, E>,
        Var<R, E>,
        Var<R, E>,
        Var<R, E>,
        Prepared,
    )> {
        let prep = self.prepare_with_donors(set, indices, use_donors)?;
        let b = indices.len();
        let slots = self.train.slots;
        let d = self.model.config.d_model as usize;
        let atoms = self.model.config.max_atoms as usize;
        let closures = self.model.config.max_ring_closures;
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
        let targets = prep.targets.upload(&self.device)?;
        let bucket = Self::bucket_for(
            &mut self.buckets,
            &self.device,
            self.model.config.n_peaks as usize,
            atoms,
            self.limits.max_steps(),
            b,
            prep.n_raw,
            slots,
        )?;
        let encoded = self
            .model
            .encoder
            .encode(&spectra, &bucket.peaks, encode_control)?;
        ms2::formula_window(
            &self.device_table.table,
            &spectra.meta,
            self.device_table.max_error,
            u32::MAX,
            TRAIN_ROWS_SCORED_MAX,
            &bucket.formula,
        )?;
        let scored =
            self.model
                .formula
                .score(&self.device_table, &bucket.formula, &encoded.pool)?;
        let gold_t = IdTensor::from_slice(&prep.gold_slots, vec![b], &self.device)?;
        let formula_loss = self.model.formula.loss(&scored, &gold_t)?;
        // Oracle conditioning on the true parent formula: the gold row's
        // embedding, or the zero vector when the gold is absent (masked out
        // below, so absent spectra still train their graph loss on nothing —
        // their `q` weights are what zeroes it — with a neutral input).
        let safe = safe_ids(&gold_t, 0)?;
        let e_gold = Var::gather_tokens(&scored.embedding, &safe, 1)?.reshape(vec![b, d])?;
        // Validity without a read: `u32::MAX` casts to 2^32 in `f32`, every
        // real slot well below it (the same device-side rule as
        // `FormulaHead::loss`).
        let gold_f: Tensor<R, E> = ids_to_float(&gold_t);
        let is_absent = crate::tensor::ops::elemwise::eq_scalar(&gold_f, u32::MAX as f32);
        let valid_mask = crate::tensor::ops::elemwise::rsub_scalar(&is_absent, 1.0);
        let valid_var = Var::constant(valid_mask)
            .reshape(vec![b, 1])?
            .expand(vec![b, d])?;
        let e_cond = e_gold.mul(&valid_var)?;
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
        let total = graph.add(&formula_loss.mul_scalar(self.train.formula_weight))?;
        let log_prob = scored.log_prob;
        Ok((tout, graph, formula_loss, total, log_prob, prep))
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
    pub fn step(&mut self, set: &ExperimentSet, indices: &[usize]) -> Result<Option<LossReport>> {
        let (tout, graph, formula_loss, total, _log_prob, prep) = self.forward(set, indices)?;
        let _ = tout;
        let b = indices.len();
        // One `[3]` tensor for the report read: a single `try_to_f32` is a
        // single batched read of all three scalars.
        let packed = cat(
            &[
                total.reshape(vec![1])?,
                graph.reshape(vec![1])?,
                formula_loss.reshape(vec![1])?,
            ],
            0,
        )?;
        let grads = total.backward_retain()?;
        match self.train.grad_clip {
            Some(max_norm) => match grad_scale(&grads, max_norm, 1.0)? {
                Some(scale) => {
                    self.optimizer
                        .step_scaled(&self.params, &grads, Some(&scale.factor))?;
                }
                None => {
                    self.optimizer.step(&self.params, &grads)?;
                }
            },
            None => {
                self.optimizer.step(&self.params, &grads)?;
            }
        }
        self.steps += 1;
        if !self.report_pending {
            return Ok(None);
        }
        self.report_pending = false;
        let values = packed.try_to_f32()?;
        let present = prep.gold_slots.iter().filter(|&&s| s != u32::MAX).count();
        Ok(Some(LossReport {
            step: self.steps,
            loss: values[0],
            graph: values[1],
            formula: values[2],
            spectra: b,
            formula_present: present,
            formula_absent: b - present,
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
        let (tout, _, _, _, log_prob, prep) = self.forward(set, indices)?;
        let b = indices.len();
        let slots = self.train.slots;
        // Gold log-probabilities without a read: the safe slot picks a value,
        // the absent rows are masked back to 0 on the device.
        let gold_t = IdTensor::from_slice(&prep.gold_slots, vec![b], &self.device)?;
        let safe = safe_ids(&gold_t, 0)?;
        let picked = log_prob.take_along_last(&safe)?;
        let gold_f: Tensor<R, E> = ids_to_float(&gold_t);
        let is_absent = crate::tensor::ops::elemwise::eq_scalar(&gold_f, u32::MAX as f32);
        let valid_mask = crate::tensor::ops::elemwise::rsub_scalar(&is_absent, 1.0);
        let picked = picked.mul(&Var::constant(valid_mask))?;
        let (_, floats) = read_all(&[], &[tout.nll.tensor(), picked.tensor()])?;
        let molecules = indices.iter().map(|&i| set.spectra[i].molecule).collect();
        Ok(TeacherEval {
            nll: floats[0].clone(),
            q: prep.targets.q.clone(),
            scored_tokens: prep.scored_tokens,
            gold_slot: prep.gold_slots,
            gold_log_prob: floats[1].clone(),
            spectra: b,
            slots,
            molecules,
            donor_same_molecule: prep.donor_same_molecule,
            donor_no_eligible_peaks: prep.donor_no_eligible_peaks,
        })
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
        let (tout, _, _, _, _, prep) = self.forward_with_donors(set, indices, use_donors)?;
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
        let _guard = no_grad();
        let shuffled = self.train.control == Control::ShuffledSpectrum
            || config.control == Control::ShuffledSpectrum;
        let (batch, gen_control) = if shuffled {
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
            (batch, cfg)
        } else {
            let n_raw = self.bucket_n_raw(set, indices)?;
            let batch = spectrum_batch_for(set, indices, n_raw as u32)?;
            (batch, config.clone())
        };
        let mut workspace = self.workspace.borrow_mut();
        let candidates = self.model.generate(
            &batch,
            &self.device_table,
            &gen_control,
            &mut workspace,
            &self.constants,
        )?;
        drop(workspace);
        let k = gen_control.trajectories as usize;
        let mut evals = evaluate_candidates(set, indices, &candidates, TRAIN_EVAL_WORK_LIMIT)?;
        for (pos, eval) in evals.iter_mut().enumerate() {
            let entry = &set.spectra[indices[pos]];
            let gold = self.gold_row(&entry.parent_composition);
            let hit =
                gold != u32::MAX && (0..k).any(|kk| candidates.formula_row[pos * k + kk] == gold);
            eval.formula_recall = Some(hit);
        }
        Ok(evals)
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
        let mut trainer = Self::new(
            &checkpoint.model_config,
            table,
            &checkpoint.train_config,
            device,
        )?;
        trainer.model.load_state_dict(&checkpoint.weights, true)?;
        trainer.steps = checkpoint.steps;
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
}
