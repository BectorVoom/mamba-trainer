//! Sampling loop, validation and the packed readout (architecture §5).
//!
//! [`Ms2Model`] composes the encoder, formula head and decoder.
//! [`GenerationWorkspace`] holds one `(B, K)` bucket of preallocated buffers
//! (peak and formula buffers plus the trajectory buffers), reused across
//! calls. [`Ms2Model::generate`] runs upload → peak selection → encoder →
//! formula window → formula head → top-F → trajectory initialisation →
//! `T - 1` sampling steps → validation → a single [`read_all`] of the packed
//! buffers, then builds the [`CandidateBatch`] on the host.
//!
//! [`read_all`]: crate::tensor::ops::index::read_all
//! [`CandidateBatch`]: crate::models::ms2::contract::CandidateBatch

use cubecl::prelude::Runtime;

use crate::autograd::Var;
use crate::backend::{Device, FloatElem};
use crate::error::{Error, Result};
use crate::models::mamba3::MixerCache;
use crate::nn::module::{Module, ModuleVisitor};
use crate::ssm::scan::SsmState;
use crate::tensor::Shape;
use crate::tensor::Tensor;
use crate::tensor::ops::elemwise;
use crate::tensor::ops::index::{IdTensor, ids_to_float, read_all, slice_ids_along};
use crate::tensor::ops::ms2::{self, Ms2Constants};
use crate::tensor::ops::random::Rng;

use super::batch::{DeviceSpectra, rotate_peaks};
use super::contract::{
    CandidateBatch, Control, GenerationConfig, ModelConfig, NO_FORMULA, SCHEMA_VERSION,
    SpectrumBatch, candidate_status, request_status,
};
use super::decoder::{DecoderState, Ms2Decoder};
use super::encoder::Ms2Encoder;
use super::formula_head::{DeviceFormulaTable, FormulaHead};
use super::workspace::Ms2MemoryEstimate;

/// Formula-window capacity of a generation bucket (architecture §2: `M`).
pub const GENERATION_WINDOW_M: usize = 32;

/// Cached `(B, K)` buckets are reused across calls; at most this many shapes
/// are kept, so an alternating-bucket sequence stops allocating after
/// warm-up while memory stays bounded.
const BUCKET_CACHE_LIMIT: usize = 4;

/// The composed MS2 model of architecture §4: spectrum encoder, formula head
/// and graph-action decoder, initialised together from one [`ModelConfig`].
pub struct Ms2Model<R: Runtime, E: FloatElem> {
    /// The configuration the three parts were built for.
    pub config: ModelConfig,
    /// Bidirectional spectrum encoder (architecture §4.1).
    pub encoder: Ms2Encoder<R, E>,
    /// Window scorer over the resident formula table (architecture §4.2).
    pub formula: FormulaHead<R, E>,
    /// Graph-action decoder (architecture §4.3).
    pub decoder: Ms2Decoder<R, E>,
}

impl<R: Runtime, E: FloatElem> Module<R, E> for Ms2Model<R, E> {
    /// Visit the encoder, formula head and decoder, in that order.
    fn visit(&self, visitor: &mut ModuleVisitor<'_, R, E>) {
        visitor.child("encoder", &self.encoder);
        visitor.child("formula", &self.formula);
        visitor.child("decoder", &self.decoder);
    }
}

/// One `(B, K)` bucket of preallocated generation buffers, reused across
/// calls with the same shapes.
struct GenBucket<R: Runtime, E: FloatElem> {
    /// Bucket key: batch, trajectories, steps, raw capacity, formulas, window.
    key: (usize, usize, usize, usize, usize, usize),
    /// Peak-selection scratch for `(B, n_raw, N)`.
    peaks: ms2::PeakBuffers<R, E>,
    /// Formula window, counters and top-F for `(B, M, F)`.
    formula: ms2::FormulaBuffers<R, E>,
    /// `[B*K, 14]` trajectory metadata (in/out).
    traj_meta: IdTensor<R>,
    /// `[B*K, 3A + 16]` grammar state rows (in/out).
    state: IdTensor<R>,
    /// `[B*K, T*4 + A + 4]` action records (in/out).
    actions: IdTensor<R>,
    /// `[B*K, 3A + 16]` validation replay scratch.
    scratch: IdTensor<R>,
    /// `[B*K, 27 + 24A]` packed sampler logits.
    logits: Tensor<R, E>,
    /// `[B*K, 4]` next input tokens.
    step_token: IdTensor<R>,
    /// `[B*K, d]` per-trajectory conditioning formula embeddings.
    traj_formula: Tensor<R, E>,
}

/// One decoder layer's recurrent tensors as host floats, for the carry-freeze
/// inspection hook.
#[derive(Clone, Debug)]
pub struct LayerCarryFloats {
    /// `h` state.
    pub h: Vec<f32>,
    /// `last_u` state.
    pub last_u: Vec<f32>,
    /// Rotational angle, when the mixer is rotational.
    pub angle: Option<Vec<f32>>,
    /// Convolution history, when the mixer has a convolution.
    pub conv: Option<Vec<f32>>,
}

/// One sampling step's recurrent tensors as host floats, for the carry-freeze
/// inspection hook.
#[derive(Clone, Debug)]
pub struct StepCarries {
    /// The sampling step (`1..T`) these post-freeze caches belong to.
    pub step: usize,
    /// One entry per decoder layer.
    pub layers: Vec<LayerCarryFloats>,
}

/// Preallocated generation state for one `(B, K)` bucket, reused across
/// calls.
///
/// The trajectory buffers are allocated once per bucket shape and rewritten
/// in place by the kernels. The mixer step and the composed attention
/// allocate their outputs functionally in V0 (a known departure recorded in
/// architecture §5, measured by the footprint test rather than claimed away).
/// Set [`GenerationWorkspace::capture_carry_trace`] to record the post-freeze
/// caches of every step as host floats in [`GenerationWorkspace::carry_trace`]
/// (extra device reads; test support only).
pub struct GenerationWorkspace<R: Runtime, E: FloatElem> {
    /// Cached buckets by shape, oldest first.
    buckets: Vec<GenBucket<R, E>>,
    /// Whether `generate` records the post-freeze caches of every step.
    pub capture_carry_trace: bool,
    /// One entry per sampling step of the last `generate` call, present only
    /// when [`GenerationWorkspace::capture_carry_trace`] was set.
    pub carry_trace: Vec<StepCarries>,
    /// The per-trajectory conditioning formula embeddings of the last
    /// `generate` call (`[B*K, d]`), for inspection tests (a handle clone,
    /// no device read).
    pub last_traj_formula: Option<Tensor<R, E>>,
}

impl<R: Runtime, E: FloatElem> GenerationWorkspace<R, E> {
    /// An empty workspace: no bucket, no capture.
    pub fn new() -> Self {
        Self {
            buckets: Vec::new(),
            capture_carry_trace: false,
            carry_trace: Vec::new(),
            last_traj_formula: None,
        }
    }

    /// Bucket shapes currently cached, oldest first.
    pub fn bucket_keys(&self) -> Vec<(usize, usize, usize, usize, usize, usize)> {
        self.buckets.iter().map(|b| b.key).collect()
    }

    /// The bucket for these shapes, allocating (and evicting the oldest past
    /// the cache limit) on a miss.
    #[allow(clippy::too_many_arguments)]
    fn bucket(
        &mut self,
        batch: usize,
        trajectories: usize,
        steps: usize,
        n_raw: usize,
        formulas: usize,
        window_m: usize,
        atoms: usize,
        d_model: usize,
        n_peaks: usize,
        device: &Device<R>,
    ) -> Result<&mut GenBucket<R, E>> {
        let key = (batch, trajectories, steps, n_raw, formulas, window_m);
        if let Some(pos) = self.buckets.iter().position(|b| b.key == key) {
            return Ok(&mut self.buckets[pos]);
        }
        let rows = batch * trajectories;
        let state_width = ms2::replay_state_width(atoms);
        let record_width = ms2::sample_record_width(steps, atoms);
        let logits_width = ms2::sample_logits_width(atoms);
        let bucket = GenBucket {
            key,
            peaks: ms2::PeakBuffers::new(batch, n_raw, n_peaks, device),
            formula: ms2::FormulaBuffers::new(batch, window_m, formulas, device),
            traj_meta: IdTensor::empty(vec![rows, ms2::TRAJ_META_WIDTH], device),
            state: IdTensor::empty(vec![rows, state_width], device),
            actions: IdTensor::empty(vec![rows, record_width], device),
            scratch: IdTensor::empty(vec![rows, state_width], device),
            logits: Tensor::empty(vec![rows, logits_width], device),
            step_token: IdTensor::empty(vec![rows, 4], device),
            traj_formula: Tensor::empty(vec![rows, d_model], device),
        };
        self.buckets.push(bucket);
        while self.buckets.len() > BUCKET_CACHE_LIMIT {
            self.buckets.remove(0);
        }
        Ok(self
            .buckets
            .iter_mut()
            .find(|b| b.key == key)
            .expect("the bucket just pushed is cached"))
    }
}

impl<R: Runtime, E: FloatElem> Default for GenerationWorkspace<R, E> {
    /// An empty workspace; see [`GenerationWorkspace::new`].
    fn default() -> Self {
        Self::new()
    }
}

impl<R: Runtime, E: FloatElem> Ms2Model<R, E> {
    /// Build the encoder, formula head and decoder for `config` on `device`.
    pub fn init(config: &ModelConfig, device: &Device<R>, rng: &mut Rng) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            config: config.clone(),
            encoder: Ms2Encoder::init(config, device, rng)?,
            formula: FormulaHead::init(config, device, rng)?,
            decoder: Ms2Decoder::init(config, device, rng)?,
        })
    }

    /// Freeze one cache tensor of finished rows back to its previous value:
    /// `new` where the row is alive, `old` where it is not, selected by
    /// comparison (never by multiplying with a mask).
    fn freeze_tensor(
        new_t: &Tensor<R, E>,
        old_t: &Tensor<R, E>,
        alive: &Tensor<R, E>,
        dead: &Tensor<R, E>,
    ) -> Result<Tensor<R, E>> {
        let dims = new_t.shape().dims().to_vec();
        let rows = dims[0];
        let mut vdims = dims;
        vdims.pop();
        let mut ashape = vec![rows];
        ashape.extend(vec![1; vdims.len() - 1]);
        let alive_v = elemwise::expand(
            &alive.reshape(Shape::new(ashape.clone()))?,
            &Shape::new(vdims.clone()),
        )?;
        let dead_v = elemwise::expand(&dead.reshape(Shape::new(ashape))?, &Shape::new(vdims))?;
        let kept_new = ms2::select_valid(new_t, &alive_v)?;
        let kept_old = ms2::select_valid(old_t, &dead_v)?;
        elemwise::add(&kept_new, &kept_old)
    }

    /// Freeze every recurrent tensor of one layer: a finished row's `h`,
    /// `last_u`, `angle` and convolution history stay exactly as they were.
    fn freeze_cache(
        old: &MixerCache<R, E>,
        new_cache: &MixerCache<R, E>,
        alive: &Tensor<R, E>,
        dead: &Tensor<R, E>,
    ) -> Result<MixerCache<R, E>> {
        let h = Var::constant(Self::freeze_tensor(
            new_cache.ssm.h.tensor(),
            old.ssm.h.tensor(),
            alive,
            dead,
        )?);
        let last_u = Var::constant(Self::freeze_tensor(
            new_cache.ssm.last_u.tensor(),
            old.ssm.last_u.tensor(),
            alive,
            dead,
        )?);
        let angle = match (&new_cache.ssm.angle, &old.ssm.angle) {
            (Some(new_a), Some(old_a)) => Some(Var::constant(Self::freeze_tensor(
                new_a.tensor(),
                old_a.tensor(),
                alive,
                dead,
            )?)),
            _ => None,
        };
        let conv = match (&new_cache.conv, &old.conv) {
            (Some(new_c), Some(old_c)) => Some(Var::constant(Self::freeze_tensor(
                new_c.tensor(),
                old_c.tensor(),
                alive,
                dead,
            )?)),
            _ => None,
        };
        Ok(MixerCache {
            ssm: SsmState { h, last_u, angle },
            conv,
        })
    }

    /// Snapshot one step's post-freeze caches as host floats (test support;
    /// performs device reads, only called when capture is enabled).
    fn snapshot_caches(state: &DecoderState<R, E>, step: usize) -> Result<StepCarries> {
        let mut layers = Vec::with_capacity(state.caches.len());
        for cache in &state.caches {
            layers.push(LayerCarryFloats {
                h: cache.ssm.h.try_to_f32()?,
                last_u: cache.ssm.last_u.try_to_f32()?,
                angle: cache
                    .ssm
                    .angle
                    .as_ref()
                    .map(|a| a.try_to_f32())
                    .transpose()?,
                conv: cache.conv.as_ref().map(|c| c.try_to_f32()).transpose()?,
            });
        }
        Ok(StepCarries { step, layers })
    }

    /// Run the full generation pipeline of architecture §5 and return the
    /// validated [`CandidateBatch`].
    ///
    /// Upload once (applying `rotate_peaks` first for
    /// [`Control::ShuffledSpectrum`]) → peak selection → encoder → formula
    /// window → formula head → top-F → trajectory initialisation → `T - 1`
    /// sampling steps of step-token, stepped heads, packed logits, sampling,
    /// atom-memory write and carry freeze → validation → one [`read_all`].
    /// Rows of a spectrum share its memory without copies
    /// (`rows_per_spectrum = K`). Failed requests keep K records with
    /// `request_failed`, `length 0` and `formula_row = u32::MAX`.
    ///
    /// Under [`Control::ShuffledSpectrum`] a batch of fewer than 2 spectra is
    /// [`Error::Config`], never a silent identity: with one row the rotation
    /// would return the spectrum's own peaks. In-batch rotation can also pair
    /// spectra of the same molecule, so experiments must use molecule-aware
    /// donors instead (see `batch::rotate_peaks`); the trainer passes
    /// donor-substituted batches as ordinary requests and never reaches this
    /// rotation a second time.
    ///
    /// Preflight: the [`Ms2MemoryEstimate`] comparison with
    /// `config.max_device_bytes` runs before any upload, allocation or
    /// launch, so a refused configuration leaves the launch and allocation
    /// counters unchanged. Buckets: a workspace built for one bucket used
    /// with another bucket's batch transparently allocates (and evicts the
    /// oldest past the cache limit) rather than erroring; see
    /// [`GenerationWorkspace::bucket_keys`].
    ///
    /// [`read_all`]: crate::tensor::ops::index::read_all
    /// [`CandidateBatch`]: crate::models::ms2::contract::CandidateBatch
    #[allow(clippy::too_many_lines)]
    pub fn generate(
        &self,
        batch: &SpectrumBatch,
        table: &DeviceFormulaTable<R, E>,
        config: &GenerationConfig,
        workspace: &mut GenerationWorkspace<R, E>,
        constants: &Ms2Constants<R>,
    ) -> Result<CandidateBatch> {
        // No gradient tape across the loop: every step would otherwise extend
        // the graph and leak memory over the trajectory.
        let _no_grad = crate::autograd::no_grad();
        let atoms = self.config.max_atoms as usize;
        let closures = self.config.max_ring_closures as usize;
        config.validate(atoms, closures)?;
        if config.oracle_formula {
            return Err(Error::Unsupported(
                "Ms2Model::generate: oracle_formula is not implemented in V0".to_string(),
            ));
        }
        let device = table.table.device().clone();
        let spectra_n = batch.len();
        let trajectories = config.trajectories as usize;
        let formulas = config.formulas as usize;
        let steps = config.max_steps as usize;
        let rows = spectra_n * trajectories;
        Ms2MemoryEstimate::generation(
            &self.config,
            table.rows as u64,
            spectra_n as u64,
            trajectories as u64,
            batch.n_raw as u64,
            steps as u64,
        )?
        .check_limit(config.max_device_bytes)?;
        // Upload once; the shuffled control rotates the peak buffers first.
        // Fewer than 2 spectra under ShuffledSpectrum is a config error,
        // never a silent identity (a one-row rotation returns own peaks).
        if config.control == Control::ShuffledSpectrum && spectra_n < 2 {
            return Err(Error::config(format!(
                "Ms2Model::generate: ShuffledSpectrum needs at least 2 spectra, got {spectra_n} (in-batch rotation cannot shuffle a single row; use molecule-aware donors)"
            )));
        }
        let owned;
        let upload_batch = if config.control == Control::ShuffledSpectrum {
            owned = rotate_peaks(batch);
            &owned
        } else {
            batch
        };
        let spectra = {
            let _tally = crate::backend::tally_scope("ms2.preprocess");
            DeviceSpectra::upload(upload_batch, &device)?
        };
        let host_status = spectra.host_status.clone();
        let spectrum_ids = spectra.spectrum_id.clone();
        let n_raw = spectra.n_raw;
        workspace.carry_trace.clear();
        let (encoded, bucket_ptr) = {
            let bucket = workspace.bucket(
                spectra_n,
                trajectories,
                steps,
                n_raw,
                formulas,
                GENERATION_WINDOW_M,
                atoms,
                self.config.d_model as usize,
                self.config.n_peaks as usize,
                &device,
            )?;
            let encoded = {
                let _tally = crate::backend::tally_scope("ms2.encoder");
                self.encoder
                    .encode(&spectra, &bucket.peaks, config.control)?
            };
            crate::backend::check_launches(&device)?;
            {
                let _tally = crate::backend::tally_scope("ms2.search");
                ms2::formula_window(
                    &table.table,
                    &spectra.meta,
                    table.max_error,
                    config.formula_rows_visited_max,
                    config.formula_rows_scored_max,
                    &bucket.formula,
                )?;
                let scored = self.formula.score(table, &bucket.formula, &encoded.pool)?;
                ms2::formula_top(
                    &scored.log_prob.tensor().clone(),
                    &bucket.formula.window,
                    &bucket.formula,
                )?;
                ms2::init_trajectories(
                    &bucket.formula.top,
                    &spectra.meta,
                    &table.counts,
                    &mut bucket.traj_meta,
                    &mut bucket.state,
                    &mut bucket.actions,
                    spectra_n,
                    formulas,
                    trajectories,
                    table.rows,
                    steps,
                    atoms,
                    matches!(
                        config.control,
                        Control::MetadataOnly | Control::StructurePrior
                    ),
                )?;
                ms2::trajectory_formula(
                    &scored.embedding.tensor().clone(),
                    &bucket.formula.top,
                    &bucket.formula.top_count,
                    &mut bucket.traj_formula,
                    spectra_n,
                    GENERATION_WINDOW_M,
                    formulas,
                    trajectories,
                )?;
            }
            let bucket_ptr: *mut GenBucket<R, E> = bucket;
            (encoded, bucket_ptr)
        };
        // The workspace borrow ends here so the sampling loop can hold the
        // bucket while pushing carry snapshots; re-borrow per step below.
        let mut decoder_state = self.decoder.start_state(&encoded, rows, &device)?;
        let bond_table = self.decoder.bond_by_type_value();
        let traj_formula = Var::constant(unsafe { &*bucket_ptr }.traj_formula.clone());
        workspace.last_traj_formula = Some(unsafe { &*bucket_ptr }.traj_formula.clone());
        let off = ms2::sample_logits_offsets(atoms);
        let logits_width = ms2::sample_logits_width(atoms);
        let seed_lo = (config.seed & 0xFFFF_FFFF) as u32;
        let seed_hi = (config.seed >> 32) as u32;
        // Step 0 is not sampled: initialisation wrote START at position 0.
        // Each step reads the last emitted token, scores it, packs the heads,
        // samples the next token, and freezes the carries of rows that had
        // already finished (their `h`, `last_u` and `angle` stay exactly as
        // they were).
        for step in 1..steps {
            let _tally = crate::backend::tally_scope("ms2.step");
            let bucket = unsafe { &mut *bucket_ptr };
            ms2::step_token(&bucket.actions, &mut bucket.step_token, steps, atoms)?;
            let old_caches: Vec<MixerCache<R, E>> = decoder_state.caches.to_vec();
            let heads = self.decoder.step_logits(
                &encoded,
                &traj_formula,
                &bucket.step_token,
                step - 1,
                &bucket.state,
                &mut decoder_state,
                trajectories,
            )?;
            ms2::pack_copy(
                heads.kind.tensor(),
                &mut bucket.logits,
                rows,
                5,
                off[0],
                logits_width,
            )?;
            ms2::pack_copy(
                heads.atom_type.tensor(),
                &mut bucket.logits,
                rows,
                18,
                off[1],
                logits_width,
            )?;
            ms2::pack_copy(
                heads.bond_base.tensor(),
                &mut bucket.logits,
                rows,
                4,
                off[2],
                logits_width,
            )?;
            ms2::pack_copy(
                heads.pointer_base.tensor(),
                &mut bucket.logits,
                rows,
                atoms,
                off[3],
                logits_width,
            )?;
            ms2::pack_copy(
                heads.pointer_by_type.tensor(),
                &mut bucket.logits,
                rows,
                19 * atoms,
                off[4],
                logits_width,
            )?;
            ms2::pack_copy(
                heads.pointer_by_bond.tensor(),
                &mut bucket.logits,
                rows,
                4 * atoms,
                off[5],
                logits_width,
            )?;
            ms2::sample_step(
                &bucket.logits,
                &bond_table,
                &bucket.traj_meta,
                &mut bucket.state,
                &mut bucket.actions,
                step as u32,
                seed_lo,
                seed_hi,
                config.temperature,
                steps,
                atoms,
                self.config.max_ring_closures,
                &constants.atom_table,
            )?;
            // Rows stopped before or at this step keep their old carries.
            let stopped_ids =
                slice_ids_along(&bucket.state, 1, 3 * atoms + 5, 1)?.reshape(vec![rows])?;
            let stopped_f = ids_to_float(&stopped_ids);
            let alive = elemwise::eq_scalar(&stopped_f, 0.0);
            let dead = elemwise::rsub_scalar(&alive, 1.0);
            let mut frozen = Vec::with_capacity(decoder_state.caches.len());
            for (old, new_cache) in old_caches.iter().zip(decoder_state.caches.iter()) {
                frozen.push(Self::freeze_cache(old, new_cache, &alive, &dead)?);
            }
            decoder_state.caches = frozen;
            if workspace.capture_carry_trace {
                workspace
                    .carry_trace
                    .push(Self::snapshot_caches(&decoder_state, step)?);
            }
        }
        let bucket = unsafe { &mut *bucket_ptr };
        let _tally = crate::backend::tally_scope("ms2.finalize");
        ms2::validate_trajectories(
            &mut bucket.actions,
            &bucket.traj_meta,
            &mut bucket.scratch,
            &constants.atom_table,
            spectra_n,
            trajectories,
            steps,
            atoms,
            self.config.max_ring_closures,
        )?;
        // The single batched read of the whole call.
        let (ids, floats) = read_all(
            &[
                &bucket.actions,
                &bucket.formula.top,
                &bucket.formula.top_count,
                &bucket.formula.counters,
                &bucket.peaks.summary,
            ],
            &[&bucket.formula.top_log_prob, &bucket.peaks.stats],
        )?;
        let actions_h = &ids[0];
        let top_lp_h = &floats[0];
        let stats_h = &floats[1];
        let top_count_h = &ids[2];
        let counters_h = &ids[3];
        let summary_h = &ids[4];
        let record_width = ms2::sample_record_width(steps, atoms);
        let mut out = CandidateBatch {
            schema_version: SCHEMA_VERSION,
            batch: spectra_n,
            trajectories,
            max_steps: steps,
            max_atoms: atoms,
            max_ring_closures: closures,
            spectrum_id: vec![0; rows],
            trajectory: vec![0; rows],
            actions: vec![0; rows * steps * 4],
            length: vec![0; rows],
            formula_row: vec![NO_FORMULA; rows],
            formula_log_prob: vec![0.0; rows],
            trace_log_prob: vec![0.0; rows],
            open_valence: vec![0; rows * atoms],
            attachment_partition: vec![0; rows],
            status: vec![0; rows],
            evidence_status: vec![0; rows],
            identity_resolution: vec![0; rows],
            request_status: vec![0; spectra_n],
            rows_visited: vec![0; spectra_n],
            rows_joined: vec![0; spectra_n],
            rows_scored: vec![0; spectra_n],
            formula_support_complete: vec![0; spectra_n],
            formula_mass_retained: vec![0.0; spectra_n],
            peaks_kept: vec![0; spectra_n],
            intensity_retained: vec![0.0; spectra_n],
        };
        for r in 0..rows {
            let b = r / trajectories;
            let kk = r % trajectories;
            let abase = r * record_width;
            out.spectrum_id[r] = spectrum_ids[b];
            out.trajectory[r] = kk as u32;
            for s in 0..steps {
                for c in 0..4 {
                    out.actions[(r * steps + s) * 4 + c] = actions_h[abase + s * 4 + c];
                }
            }
            out.length[r] = actions_h[abase + steps * 4 + atoms];
            out.status[r] = actions_h[abase + steps * 4 + atoms + 1];
            out.trace_log_prob[r] = f32::from_bits(actions_h[abase + steps * 4 + atoms + 2]);
            out.formula_row[r] = actions_h[abase + steps * 4 + atoms + 3];
            for j in 0..atoms {
                out.open_valence[r * atoms + j] = actions_h[abase + steps * 4 + j] as u8;
            }
            let count = top_count_h[b] as usize;
            out.formula_log_prob[r] = if count > 0 {
                top_lp_h[b * formulas + kk % count]
            } else {
                0.0
            };
        }
        for b in 0..spectra_n {
            // The union of the host validation bits, the peak-selection
            // summary and the formula-search counters. A spectrum with no
            // scored formula carries no conditioning hypothesis, so it is
            // marked `formula_absent` (fatal): its records are failed by
            // construction, and `validate` requires the fatal bit for them.
            let mut rs = host_status[b] | summary_h[b * 2 + 1] | counters_h[b * 5 + 3];
            if top_count_h[b] == 0 {
                rs |= request_status::FORMULA_ABSENT;
            }
            if matches!(
                config.control,
                Control::MetadataOnly | Control::StructurePrior
            ) {
                // The controls bypass the empty-spectrum abstention, which
                // exists for real requests only.
                rs &= !request_status::EMPTY_SPECTRUM;
            }
            out.request_status[b] = rs;
            out.rows_visited[b] = counters_h[b * 5];
            out.rows_joined[b] = counters_h[b * 5 + 1];
            out.rows_scored[b] = counters_h[b * 5 + 2];
            out.formula_support_complete[b] = counters_h[b * 5 + 4].min(1) as u8;
            let count = top_count_h[b] as usize;
            out.formula_mass_retained[b] = if out.formula_support_complete[b] == 1 {
                (0..count.min(formulas))
                    .map(|f| top_lp_h[b * formulas + f].exp())
                    .sum()
            } else {
                0.0
            };
            out.peaks_kept[b] = summary_h[b * 2];
            out.intensity_retained[b] = stats_h[b * 3 + 2];
            // Host-side failure enforcement: a fatal request carries no
            // trace, so its device records (possibly started under the
            // metadata-only bypass) are replaced by failed records here.
            if rs & request_status::FATAL_MASK != 0 {
                for kk in 0..trajectories {
                    let r = b * trajectories + kk;
                    out.length[r] = 0;
                    out.formula_row[r] = NO_FORMULA;
                    out.formula_log_prob[r] = 0.0;
                    out.trace_log_prob[r] = 0.0;
                    out.status[r] = candidate_status::REQUEST_FAILED;
                    for v in &mut out.actions[r * steps * 4..(r + 1) * steps * 4] {
                        *v = 0;
                    }
                    for v in &mut out.open_valence[r * atoms..(r + 1) * atoms] {
                        *v = 0;
                    }
                }
            }
        }
        out.validate()?;
        Ok(out)
    }
}
