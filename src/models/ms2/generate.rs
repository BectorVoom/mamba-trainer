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

use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use crate::autograd::Var;
use crate::backend::{Device, FloatElem};
use crate::error::{Error, Result};
use crate::models::mamba3::MixerCache;
use crate::nn::module::{Module, ModuleVisitor};
use crate::ssm::scan::SsmState;
use crate::tensor::Shape;
use crate::tensor::Tensor;
use crate::tensor::ops::elemwise;
use crate::tensor::ops::index::{IdTensor, ids_to_float, read_all, read_all_mixed, slice_ids_along};
use crate::tensor::ops::ms2::{self, Ms2Constants};
use crate::tensor::ops::random::Rng;

use super::batch::{DeviceSpectra, rotate_peaks};
use super::chem::{parent_mass, tolerance_u32};
use super::contract::{
    AllocationMode, CandidateBatch, Control, FormulaFeatures, FormulaSource, GenerationConfig,
    IdentityMode, ModelConfig, NO_FORMULA, SCHEMA_VERSION, SpectrumBatch, candidate_status,
    request_status, EVIDENCE_CAP,
};
use super::decoder::{DecoderState, Ms2Decoder};
use super::encoder::{EncoderOutput, Ms2Encoder};
use super::enum_cache::{EnumCache, EnumCacheHeader};
use super::formula_enum::{DEVICE_HALF_MAX, build_enum_meta, validate_enum_dispatch};
use super::formula_head::{DeviceEnumArtifacts, DeviceFormulaTable, FormulaHead};
use super::identity::IDENTITY_REQUEST_WORK_MAX;
use super::pack::{PackedCandidateBatch, assemble};
use super::workspace::{Ms2Capabilities, Ms2MemoryEstimate};
use crate::tensor::ops::ms2_enum::{EnumLaunch, cand_pad, enum_offsets};
use crate::tensor::ops::ms2_formula_evidence;
use crate::tensor::ops::ms2_identity;
use crate::tensor::ops::ms2_pack;
use crate::tensor::ops::movement;

/// Formula-window capacity of a generation bucket (architecture §2: `M`).
/// V0 default; V1 §1.2 buckets are keyed by the request's `formula_window`
/// (32, 128, 512, 2048).
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
    /// Resident enumeration artifacts for `FormulaSource::Enumerate` (V1
    /// §1.4, `None` for the table-only path). Set with
    /// [`Ms2Model::upload_enum_artifacts`]; `generate` with `Enumerate`
    /// requires it.
    pub enum_artifacts: Option<DeviceEnumArtifacts<R>>,
    /// Memoised device enumeration (task T6): when set, `generate` and the
    /// training prefix serve fully cached batches from it (two uploads, no
    /// enumeration kernel) and run the device enumeration otherwise. Set
    /// with [`Ms2Model::set_enum_cache`]. `None` means today's behaviour.
    enum_cache: Option<Arc<EnumCache>>,
    /// Cache lookups served or attempted since init (see
    /// [`Ms2Model::enum_cache_stats`]).
    enum_cache_lookups: AtomicU64,
    /// Of them, served from the cache.
    enum_cache_hits: AtomicU64,
    /// Fragment-ion assignment head (architecture §2.2). `None` means
    /// assignment disabled: exactly today's behaviour and results.
    pub assignment: Option<super::assign::AssignmentHead<R, E>>,
}

impl<R: Runtime, E: FloatElem> Module<R, E> for Ms2Model<R, E> {
    /// Visit the encoder, formula head, decoder and (when present) the
    /// assignment head, in that order.
    fn visit(&self, visitor: &mut ModuleVisitor<'_, R, E>) {
        visitor.child("encoder", &self.encoder);
        visitor.child("formula", &self.formula);
        visitor.child("decoder", &self.decoder);
        if let Some(assign) = &self.assignment {
            visitor.child("assignment", assign);
        }
    }
}

/// One `(B, K)` bucket of preallocated generation buffers, reused across
/// calls with the same shapes.
struct GenBucket<R: Runtime, E: FloatElem> {
    /// Bucket key: batch, trajectories, steps, raw capacity, formulas,
    /// window, enum lanes `P`, packed slots `R`, evidence flag (as `u8`),
    /// formula-features layout (as `u8`: 0 `Counts`, 1 `Evidence`).
    /// The evidence flag is part of the key so ON and OFF calls never share
    /// buffers: OFF always sees its own creation-zero evidence rows. The
    /// layout flag is part of the key so `Counts` and `Evidence` calls never
    /// share formula buffers.
    key: (usize, usize, usize, usize, usize, usize, usize, usize, u8, u8),
    /// Peak-selection scratch for `(B, n_raw, N)`.
    peaks: ms2::PeakBuffers<R, E>,
    /// Formula window, counters and top-F for `(B, M, F)`.
    formula: ms2::FormulaBuffers<R, E>,
    /// `[B, K, 12]` trajectory-to-formula allocation (V1 §3.2).
    traj_alloc: IdTensor<R>,
    /// `[B, K, 12]` window-slot translation of the allocation (V1 §4.4):
    /// word 0 is the window slot (the formula rank the packed record
    /// carries), the other words copied.
    traj_window: IdTensor<R>,
    /// `[B * P, 2]` lane stats for the enumerating source (`[0, 2]` when
    /// `P == 0`).
    lane_stats: IdTensor<R>,
    /// `[B * P]` clamped offsets for the enumerating source (empty when
    /// `P == 0`).
    offsets: IdTensor<R>,
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
    /// `[B*K]` graph hashes (V1 §4.2; written when `identity` is `Graph`).
    graph_hash: IdTensor<R>,
    /// `[B*K, G]` hash scratch (`G = 3(A-1+R) + 2A`).
    graph_scratch: IdTensor<R>,
    /// `[B*K, 2]` identity bits and resolution (V1 §4.2).
    identity: IdTensor<R>,
    /// `[B*K, 3A]` identity search scratch.
    identity_scratch: IdTensor<R>,
    /// `[B*K, 2]` ranking scores: trace and formula log-probabilities.
    /// Always f32, on every neural dtype: the trace term is the stored f32
    /// bits unchanged, the formula term is widened into it, so device and
    /// host order agree (spec §4.4).
    scores: Tensor<R, f32>,
    /// `[B*K]` per-trajectory ranks (`u32::MAX` when ineligible).
    rank: IdTensor<R>,
    /// `[B*K]` caller reranker scores (all zero until P6.3 builds it).
    rerank: Tensor<R, E>,
    /// `[B*K, W]` integer pack records.
    record: IdTensor<R>,
    /// `[B*K, 3]` float pack records: always f32, on every neural dtype, so
    /// device words equal the host `pack` exactly (spec §4.4).
    record_f: Tensor<R, f32>,
    /// `[B, R, W]` packed integer records.
    packed: IdTensor<R>,
    /// `[B, R, 3]` packed float records: always f32, on every neural dtype.
    packed_f: Tensor<R, f32>,
    /// `[B]` filled slots per spectrum.
    returned_count: IdTensor<R>,
    /// `[B*K, 18]` evidence rows (status, total count, then the `E = 4`
    /// records of kept-peak position, hypothesis index, shift and residual
    /// offsets). Written by `generate_evidence` when `evidence` is on (as in
    /// `generate`); zero otherwise.
    evidence: IdTensor<R>,
    /// `[B*R, 18]` packed evidence rows in `(spectrum, rank)` order
    /// (kept-peak positions; the host maps them to original peak ids at
    /// readout exactly as `generate` does). Written by `pack_evidence` when
    /// `evidence` is on; zero otherwise. Buckets are keyed by the evidence
    /// flag, so an OFF bucket always holds its creation-zero rows.
    packed_ev: IdTensor<R>,
    /// `[B*R, 4]` packed assignment log-probabilities per evidence record
    /// (`0` beyond the stored count). Written with `packed_ev`.
    packed_ev_f: Tensor<R, E>,
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

/// One boundary of [`Ms2Model::generate_with_hook`]: the hook runs after each
/// pipeline stage, so a profiling driver observes the real stages of one
/// generation call instead of subtracting separately timed runs.
///
/// When no hook is installed ([`Ms2Model::generate`]) each check is a single
/// `None` comparison: no launch, no read and no allocation is added, and the
/// footprint counters are exactly what the hook-free call produces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenerateStage {
    /// Request upload (with `ShuffledSpectrum` rotation applied first).
    AfterPreprocess,
    /// Spectrum encoder.
    AfterEncoder,
    /// Formula window, scoring, top-F, trajectory initialisation and the
    /// formula broadcast.
    AfterSearch,
    /// Decoder start state (K/V projections, cache fills) and conditioning
    /// setup, before the first sampling step.
    AfterDecoderInit,
    /// Sampling step `step` (`1..max_steps`) with carry freeze.
    AfterDecodeStep(usize),
    /// Trajectory validation, before the readout.
    AfterValidate,
    /// Ranking and compaction (`scores`, `rank`, `record_pack`,
    /// `record_pack_f`, `pack`): emitted only by the packed and resident
    /// paths, between validation and the readout.
    AfterPack,
    /// The single batched readout, before host-side batch building.
    AfterReadout,
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
    /// Drive the decode loop with the composed reference step
    /// (`Ms2Decoder::step_logits` plus the pack copies) instead of the fused
    /// step. The two compute the same values; this is what the parity tests
    /// compare the fused step against.
    pub composed_step: bool,
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
            composed_step: false,
            carry_trace: Vec::new(),
            last_traj_formula: None,
        }
    }

    /// Bucket shapes currently cached, oldest first.
    pub fn bucket_keys(
        &self,
    ) -> Vec<(usize, usize, usize, usize, usize, usize, usize, usize, u8, u8)> {
        self.buckets.iter().map(|b| b.key).collect()
    }

    /// Test-only snapshot of the most recently cached bucket's search-stage
    /// buffers: clones (no device read) of the kept peaks, the kept-peak
    /// features, the gathered candidates, the evidence-peak buffers and the
    /// scored feature buffers. A test drives production (`generate` or the
    /// staged `generate_*_ws` functions) on a fresh workspace, then compares
    /// these device buffers against the host twins applied to the same
    /// request. Returns `None` when no bucket is cached.
    #[doc(hidden)]
    #[allow(clippy::type_complexity)]
    pub fn debug_search_buffers(
        &self,
    ) -> Option<(
        IdTensor<R>,
        Tensor<R, E>,
        IdTensor<R>,
        Option<IdTensor<R>>,
        Option<Tensor<R, E>>,
        Option<Tensor<R, E>>,
        Tensor<R, E>,
        Option<Tensor<R, E>>,
    )> {
        let bucket = self.buckets.last()?;
        Some((
            bucket.peaks.kept.clone(),
            bucket.peaks.kept_f.clone(),
            bucket.formula.cand.clone(),
            bucket.formula.ev_peaks.clone(),
            bucket.formula.ev_w.clone(),
            bucket.formula.cand_ev.clone(),
            bucket.formula.cand_feat.clone(),
            bucket.formula.cand_xfeat.clone(),
        ))
    }

    /// The bucket for these shapes, allocating (and evicting the oldest past
    /// the cache limit) on a miss. `enum_p` is the rare-table rows `P` (0
    /// for the table source); the bucket owns `lane_stats` and `offsets`
    /// keyed by `B` and `P` (V1 §1.4). `returned` is the packed slots `R`
    /// (V1 §4.4). `formula_features` selects the `Counts` or `Evidence`
    /// formula buffers (architecture §1.6).
    #[allow(clippy::too_many_arguments)]
    fn bucket(
        &mut self,
        batch: usize,
        trajectories: usize,
        steps: usize,
        n_raw: usize,
        formulas: usize,
        window_m: usize,
        enum_p: usize,
        returned: usize,
        evidence: bool,
        formula_features: FormulaFeatures,
        atoms: usize,
        closures: usize,
        d_model: usize,
        n_peaks: usize,
        device: &Device<R>,
    ) -> Result<&mut GenBucket<R, E>> {
        let key = (
            batch, trajectories, steps, n_raw, formulas, window_m, enum_p, returned,
            u8::from(evidence),
            u8::from(formula_features == FormulaFeatures::Evidence),
        );
        if let Some(pos) = self.buckets.iter().position(|b| b.key == key) {
            return Ok(&mut self.buckets[pos]);
        }
        let bucket = Self::make_bucket(
            key, batch, trajectories, steps, n_raw, formulas, window_m, enum_p, returned,
            atoms, closures, d_model, n_peaks, device,
        )?;
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

    /// Allocate one bucket for these shapes (the cold path of [`bucket`]).
    ///
    /// [`bucket`]: GenerationWorkspace::bucket
    #[allow(clippy::too_many_arguments)]
    fn make_bucket(
        key: (usize, usize, usize, usize, usize, usize, usize, usize, u8, u8),
        batch: usize,
        trajectories: usize,
        steps: usize,
        n_raw: usize,
        formulas: usize,
        window_m: usize,
        enum_p: usize,
        returned: usize,
        atoms: usize,
        closures: usize,
        d_model: usize,
        n_peaks: usize,
        device: &Device<R>,
    ) -> Result<GenBucket<R, E>> {
        let rows = batch * trajectories;
        let state_width = ms2::replay_state_width(atoms);
        let record_width = ms2::sample_record_width(steps, atoms);
        let logits_width = ms2::sample_logits_width(atoms);
        let lanes = batch * enum_p;
        let graph_stride = super::identity::graph_scratch_len(atoms as u32, closures as u32);
        let stack_stride = super::identity::identity_stack_len(atoms as u32);
        let pack_width = super::pack::record_width(steps, atoms);
        let rerank = if rows == 0 {
            Tensor::empty(vec![0], device)
        } else {
            Tensor::from_f32(&vec![0.0f32; rows], vec![rows], device)?
        };
        let evidence_len = rows * super::pack::EVIDENCE_STRIDE as usize;
        let evidence = if evidence_len == 0 {
            IdTensor::empty(vec![rows, super::pack::EVIDENCE_STRIDE as usize], device)
        } else {
            // Zero-filled once at bucket creation (a cold-path upload);
            // `generate_evidence` overwrites these rows when `evidence` is
            // on, and the pack stage compacts them into `packed_ev`.
            IdTensor::from_slice(&vec![0u32; evidence_len], vec![rows, super::pack::EVIDENCE_STRIDE as usize], device)?
        };
        let slots = batch * returned;
        let packed_ev = if slots == 0 {
            IdTensor::empty(vec![slots, super::pack::EVIDENCE_STRIDE as usize], device)
        } else {
            IdTensor::from_slice(
                &vec![0u32; slots * super::pack::EVIDENCE_STRIDE as usize],
                vec![slots, super::pack::EVIDENCE_STRIDE as usize],
                device,
            )?
        };
        let packed_ev_f = if slots == 0 {
            Tensor::empty(vec![slots, super::contract::EVIDENCE_CAP], device)
        } else {
            Tensor::from_f32(
                &vec![0.0f32; slots * super::contract::EVIDENCE_CAP],
                vec![slots, super::contract::EVIDENCE_CAP],
                device,
            )?
        };
        let bucket = GenBucket {
            key,
            peaks: ms2::PeakBuffers::new(batch, n_raw, n_peaks, device),
            formula: if key.9 == 1 {
                ms2::FormulaBuffers::new_evidence(batch, window_m, formulas, device)
            } else {
                ms2::FormulaBuffers::new(batch, window_m, formulas, device)
            },
            traj_alloc: IdTensor::empty(vec![batch, trajectories, 12], device),
            traj_window: IdTensor::empty(vec![batch, trajectories, 12], device),
            lane_stats: IdTensor::empty(vec![lanes, 2], device),
            offsets: IdTensor::empty(vec![lanes], device),
            traj_meta: IdTensor::empty(vec![rows, ms2::TRAJ_META_WIDTH], device),
            state: IdTensor::empty(vec![rows, state_width], device),
            actions: IdTensor::empty(vec![rows, record_width], device),
            scratch: IdTensor::empty(vec![rows, state_width], device),
            logits: Tensor::empty(vec![rows, logits_width], device),
            step_token: IdTensor::empty(vec![rows, 4], device),
            traj_formula: Tensor::empty(vec![rows, d_model], device),
            graph_hash: IdTensor::empty(vec![rows], device),
            graph_scratch: IdTensor::empty(vec![rows, graph_stride], device),
            // Zero-filled once at bucket creation (a cold-path upload): with
            // `TraceOnly` no identity kernel runs, and the readout reads
            // these zeros (bits 0, resolution 0).
            identity: IdTensor::from_slice(&vec![0u32; rows * 2], vec![rows, 2], device)?,
            identity_scratch: IdTensor::empty(vec![rows, stack_stride], device),
            scores: Tensor::empty(vec![rows, 2], device),
            rank: IdTensor::empty(vec![rows], device),
            rerank,
            record: IdTensor::empty(vec![rows, pack_width], device),
            record_f: Tensor::empty(vec![rows, super::pack::WF], device),
            packed: IdTensor::empty(vec![batch, returned, pack_width], device),
            packed_f: Tensor::empty(vec![batch, returned, super::pack::WF], device),
            returned_count: IdTensor::empty(vec![batch], device),
            evidence,
            packed_ev,
            packed_ev_f,
        };
        Ok(bucket)
    }

    /// Remove the bucket for these shapes from the cache, allocating on a
    /// miss: the lease primitive of `generate_resident`. A later call on the
    /// same workspace finds no bucket and allocates another one rather than
    /// overwriting the leased result. [`unlease_bucket`] returns it.
    ///
    /// [`unlease_bucket`]: GenerationWorkspace::unlease_bucket
    fn lease_bucket(
        &mut self,
        pre: &GeneratePreflight,
        d_model: usize,
        n_peaks: usize,
        device: &Device<R>,
    ) -> Result<GenBucket<R, E>> {
        let key = (
            pre.spectra_n,
            pre.trajectories,
            pre.steps,
            pre.n_raw as usize,
            pre.formulas,
            pre.window_m,
            pre.enum_p,
            pre.returned,
            u8::from(pre.evidence),
            u8::from(pre.formula_features == FormulaFeatures::Evidence),
        );
        if let Some(pos) = self.buckets.iter().position(|b| b.key == key) {
            return Ok(self.buckets.remove(pos));
        }
        Self::make_bucket(
            key,
            pre.spectra_n,
            pre.trajectories,
            pre.steps,
            pre.n_raw as usize,
            pre.formulas,
            pre.window_m,
            pre.enum_p,
            pre.returned,
            pre.atoms,
            pre.closures as usize,
            d_model,
            n_peaks,
            device,
        )
    }

    /// Return a leased bucket to the cache, evicting the oldest past the
    /// cache limit. See [`lease_bucket`].
    ///
    /// [`lease_bucket`]: GenerationWorkspace::lease_bucket
    fn unlease_bucket(&mut self, bucket: GenBucket<R, E>) {
        self.buckets.push(bucket);
        while self.buckets.len() > BUCKET_CACHE_LIMIT {
            self.buckets.remove(0);
        }
    }
}

impl<R: Runtime, E: FloatElem> Default for GenerationWorkspace<R, E> {
    /// An empty workspace; see [`GenerationWorkspace::new`].
    fn default() -> Self {
        Self::new()
    }
}

/// Per-spectrum fields shared by the trajectory-ordered and packed readouts:
/// built once from a batched read by
/// [`Ms2Model::per_spectrum_output`].
struct PerSpectrumOut {
    /// Request bits of [`request_status`], per spectrum.
    request_status: Vec<u32>,
    /// Table rows compared against the mass window, per spectrum.
    rows_visited: Vec<u32>,
    /// Rows inside the window, per spectrum.
    rows_joined: Vec<u32>,
    /// Rows given a neural score, per spectrum.
    rows_scored: Vec<u32>,
    /// `1` when every joined row was scored, else `0`.
    formula_support_complete: Vec<u8>,
    /// Probability mass of the retained formulas within the scored window.
    formula_mass_retained: Vec<f32>,
    /// Peaks kept after device selection, per spectrum.
    peaks_kept: Vec<u32>,
    /// Fraction of filtered intensity the kept peaks hold, per spectrum.
    intensity_retained: Vec<f32>,
}

/// Validated shape dims of one generation call, from [`Ms2Model::generate_preflight`].
///
/// The preflight runs the config validation and the [`Ms2MemoryEstimate`]
/// comparison with `config.max_device_bytes` before any upload, allocation
/// or launch, so a refused configuration leaves the launch and allocation
/// counters unchanged — in production and in the device-mode harness alike.
pub struct GeneratePreflight {    /// Spectra per batch.
    pub spectra_n: usize,
    /// Total trajectories per spectrum.
    pub trajectories: usize,
    /// Formula hypotheses per spectrum.
    pub formulas: usize,
    /// Maximum trace steps.
    pub steps: usize,
    /// Trajectory rows (`spectra_n * trajectories`).
    pub rows: usize,
    /// Maximum atoms per candidate.
    pub atoms: usize,
    /// Maximum ring closures per candidate.
    pub closures: u32,
    /// Scored-candidate capacity (`formula_window`).
    pub window_m: usize,
    /// Raw peak capacity of the batch.
    pub n_raw: u32,
    /// Enum lanes `P` (0 for the table source).
    pub enum_p: usize,
    /// Packed slots per spectrum (`R = config.effective_returned()`).
    pub returned: usize,
    /// Fragment-ion evidence flag: buckets are keyed by it, so a bucket
    /// reused across evidence flags can never surface stale evidence rows
    /// (OFF always sees its own creation-zero buffers).
    pub evidence: bool,
    /// What the formula head ranks with (architecture §1.6): buckets are
    /// keyed by it, so `Counts` and `Evidence` calls never share formula
    /// buffers.
    pub formula_features: FormulaFeatures,
}

/// One composed (non-fused) sampling step (`1..max_steps`) with carry
/// freeze: the shared body behind [`Ms2Model::generate_decode_step`]'s
/// composed branch and the completion sampler.
///
/// The caller has already run [`ms2::step_token`] (the shared prologue that
/// also feeds the fused branch): `decoder.step_logits` scores the last
/// emitted token, the six `pack_copy` calls land the head fields in the
/// packed sampler row, [`ms2::sample_step`] draws the next token under the
/// exact-completion rule when the trajectory's started word is 2, and rows
/// stopped before or at this step keep their old carries. No device read;
/// the launch count is independent of the rows.
#[allow(clippy::too_many_arguments)]
pub(crate) fn composed_decode_step<R: Runtime, E: FloatElem>(
    decoder: &Ms2Decoder<R, E>,
    encoded: &EncoderOutput<R, E>,
    traj_formula: &Var<R, E>,
    actions: &mut IdTensor<R>,
    step_token: &mut IdTensor<R>,
    replay: &mut IdTensor<R>,
    logits: &mut Tensor<R, E>,
    traj_meta: &IdTensor<R>,
    decoder_state: &mut DecoderState<R, E>,
    bond_table: &Tensor<R, E>,
    atom_table: &IdTensor<R>,
    step: usize,
    seed_lo: u32,
    seed_hi: u32,
    temperature: f32,
    steps: usize,
    atoms: usize,
    closures: u32,
    trajectories: usize,
    rows: usize,
) -> Result<()> {
    let off = ms2::sample_logits_offsets(atoms);
    let logits_width = ms2::sample_logits_width(atoms);
    let old_caches: Vec<MixerCache<R, E>> = decoder_state.caches.to_vec();
    let heads = decoder.step_logits(
        encoded,
        traj_formula,
        step_token,
        step - 1,
        replay,
        decoder_state,
        trajectories,
    )?;
    ms2::pack_copy(heads.kind.tensor(), logits, rows, 5, off[0], logits_width)?;
    ms2::pack_copy(heads.atom_type.tensor(), logits, rows, 18, off[1], logits_width)?;
    ms2::pack_copy(heads.bond_base.tensor(), logits, rows, 4, off[2], logits_width)?;
    ms2::pack_copy(
        heads.pointer_base.tensor(),
        logits,
        rows,
        atoms,
        off[3],
        logits_width,
    )?;
    ms2::pack_copy(
        heads.pointer_by_type.tensor(),
        logits,
        rows,
        19 * atoms,
        off[4],
        logits_width,
    )?;
    ms2::pack_copy(
        heads.pointer_by_bond.tensor(),
        logits,
        rows,
        4 * atoms,
        off[5],
        logits_width,
    )?;
    ms2::sample_step(
        logits,
        bond_table,
        traj_meta,
        replay,
        actions,
        step as u32,
        seed_lo,
        seed_hi,
        temperature,
        steps,
        atoms,
        closures,
        atom_table,
    )?;
    // Rows stopped before or at this step keep their old carries.
    let stopped_ids =
        slice_ids_along(replay, 1, 3 * atoms + 5, 1)?.reshape(vec![rows])?;
    let stopped_f = ids_to_float(&stopped_ids);
    let alive = elemwise::eq_scalar(&stopped_f, 0.0);
    let dead = elemwise::rsub_scalar(&alive, 1.0);
    let mut frozen = Vec::with_capacity(decoder_state.caches.len());
    for (old, new_cache) in old_caches.iter().zip(decoder_state.caches.iter()) {
        frozen.push(Ms2Model::<R, E>::freeze_cache(old, new_cache, &alive, &dead)?);
    }
    decoder_state.caches = frozen;
    Ok(())
}

impl<R: Runtime, E: FloatElem> Ms2Model<R, E> {
    /// Build the encoder, formula head and decoder for `config` on `device`.
    pub fn init(config: &ModelConfig, device: &Device<R>, rng: &mut Rng) -> Result<Self> {
        config.validate()?;
        // Dtype gate (contracts §3.3): the actual neural element type must
        // equal the configured dtype and be in the validated set — refused
        // here (Error::Config on mismatch), so an unsupported dtype can
        // never reach allocation, upload or launch.
        Ms2Capabilities::check_device::<R, E>(device, config)?;
        let assignment = match &config.assignment {
            Some(a) => {
                a.validate()?;
                Some(super::assign::AssignmentHead::init(config, device, rng)?)
            }
            None => None,
        };
        Ok(Self {
            config: config.clone(),
            encoder: Ms2Encoder::init(config, device, rng)?,
            formula: FormulaHead::init(config, device, rng)?,
            decoder: Ms2Decoder::init(config, device, rng)?,
            enum_artifacts: None,
            enum_cache: None,
            enum_cache_lookups: AtomicU64::new(0),
            enum_cache_hits: AtomicU64::new(0),
            assignment,
        })
    }

    /// Upload enumeration artifacts once and bind them to the config
    /// (V1 §1.4): validates with `validate_device_artifacts`, uploads `rare`
    /// and the packed `bounds`, records versions and SHA-256, and stamps
    /// `ModelConfig::formula_artifacts`. A mismatch at load is
    /// `Error::Config` via [`DeviceEnumArtifacts::check`].
    pub fn upload_enum_artifacts(
        &mut self,
        domain: &super::formula_enum::EnumDomain,
        bounds: &super::formula_enum::RatioBounds,
        device: &Device<R>,
    ) -> Result<()> {
        let artifacts = DeviceEnumArtifacts::upload(domain, bounds, device)?;
        self.config.formula_artifacts = Some(super::contract::FormulaArtifactsRef {
            domain_version: artifacts.domain_version.clone(),
            domain_sha256: artifacts.domain_sha256.clone(),
            bounds_version: artifacts.bounds_version.clone(),
            bounds_sha256: artifacts.bounds_sha256.clone(),
        });
        self.enum_artifacts = Some(artifacts);
        Ok(())
    }

    /// Set pre-uploaded enumeration artifacts, checking them against the
    /// config (mismatch is `Error::Config`). For `--load` paths that
    /// re-upload from stored JSON.
    pub fn set_enum_artifacts(&mut self, artifacts: DeviceEnumArtifacts<R>) -> Result<()> {
        artifacts.check(&self.config)?;
        self.enum_artifacts = Some(artifacts);
        Ok(())
    }

    /// Attach a memoised device enumeration (task T6): with
    /// `FormulaSource::Enumerate`, `generate` (all readout modes) and the
    /// training prefix serve a fully cached batch from it — building the
    /// meta rows as today, then uploading `cand` and `counters` (two
    /// uploads) and launching no enumeration kernel — and run the device
    /// enumeration exactly as today otherwise. A partially cached batch
    /// takes the device path (no mixing). `None` (the default) means
    /// today's behaviour: no cache, unchanged launches and reads.
    ///
    /// The cache is exact or absent: everything downstream sees
    /// bit-identical `cand` and `counters`, and no device read is added to
    /// a training step or to `generate` by the cache.
    pub fn set_enum_cache(&mut self, cache: Option<Arc<EnumCache>>) {
        self.enum_cache = cache;
    }

    /// Cache lookups attempted and served since init, as `(lookups, hits)`:
    /// every `Enumerate` search with a cache set counts one lookup, and one
    /// hit when the whole batch was served from the cache.
    pub fn enum_cache_stats(&self) -> (u64, u64) {
        (
            self.enum_cache_lookups.load(Ordering::Relaxed),
            self.enum_cache_hits.load(Ordering::Relaxed),
        )
    }

    /// The attached cache, if any (crate-visible for the training prefix,
    /// which serves cached batches the same way as the search stage).
    pub(crate) fn enum_cache_ref(&self) -> Option<&Arc<EnumCache>> {
        self.enum_cache.as_ref()
    }

    /// Record one cache lookup (`hit` when the whole batch was served from
    /// the cache). Crate-visible for the training prefix.
    pub(crate) fn note_enum_cache_lookup(&self, hit: bool) {
        self.enum_cache_lookups.fetch_add(1, Ordering::Relaxed);
        if hit {
            self.enum_cache_hits.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// The cache header this model enumerates with under `config`: the
    /// resident artifacts' SHA-256, their depth `P`, the window `M` and the
    /// scored/visit budgets. [`Error::Config`] without resident artifacts.
    /// The driver stamps fresh caches with this and loads stored ones
    /// against it (mismatch is never silently rebuilt over).
    pub fn enum_cache_header(&self, config: &GenerationConfig) -> Result<EnumCacheHeader> {
        let Some(artifacts) = self.enum_artifacts.as_ref() else {
            return Err(Error::config(
                "Ms2Model::enum_cache_header: formula_source Enumerate needs resident enum artifacts".to_string(),
            ));
        };
        let p = u32::try_from(artifacts.p).map_err(|_| {
            Error::config(format!(
                "Ms2Model::enum_cache_header: rare-table depth {} exceeds u32",
                artifacts.p
            ))
        })?;
        Ok(EnumCacheHeader::new(
            artifacts.domain_sha256.clone(),
            artifacts.bounds_sha256.clone(),
            p,
            config.formula_window,
            config.formula_rows_scored_max,
            config.enum_lane_visits_max,
        ))
    }

    /// Fill `cache` from the EXISTING device enumeration (count → offsets →
    /// fill → pad) for each batch, reading `cand` and `counters` back (one
    /// batched read per batch — this is a precompute pass, reads are
    /// expected) and inserting every spectrum not yet present. Fully cached
    /// batches are skipped without any launch or read. The cache header
    /// must match what this model enumerates with under `config`
    /// ([`Ms2Model::enum_cache_header`]), else `Error::Config` naming the
    /// field. Reuses the production launch functions, not a copy.
    pub fn build_enum_cache<'a>(
        &self,
        batches: impl Iterator<Item = &'a SpectrumBatch>,
        config: &GenerationConfig,
        cache: &mut EnumCache,
    ) -> Result<()> {
        let Some(artifacts) = self.enum_artifacts.as_ref() else {
            return Err(Error::config(
                "Ms2Model::build_enum_cache: formula_source Enumerate needs resident enum artifacts".to_string(),
            ));
        };
        artifacts.check(&self.config)?;
        cache
            .header()
            .check_compatible(&self.enum_cache_header(config)?)?;
        let scored_cap = config
            .formula_rows_scored_max
            .min(config.formula_window);
        let window_m = config.formula_window as usize;
        let device = artifacts.rare.device().clone();
        for batch in batches {
            super::enum_cache::run_device_enumeration_into::<R, E>(
                &device,
                artifacts,
                batch,
                scored_cap,
                config.enum_lanes_max,
                config.enum_dispatch_visits_max,
                config.enum_lane_visits_max,
                window_m,
                cache,
            )?;
        }
        Ok(())
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
    /// `pub(crate)` so the shared composed step ([`composed_decode_step`])
    /// can freeze the caches it stepped.
    pub(crate) fn freeze_cache(
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

    /// [`Ms2Model::freeze_cache`] in place, for the fused step: every
    /// recurrent tensor of every layer of `new_caches` takes the old row
    /// where the trajectory has stopped (grammar state column `3A + 5`), two
    /// tensors to a launch and no allocation.
    fn freeze_caches_in_place(
        old: &[MixerCache<R, E>],
        new_caches: &[MixerCache<R, E>],
        grammar_state: &IdTensor<R>,
        atoms: usize,
    ) -> Result<()> {
        let mut carries: Vec<(Tensor<R, E>, &Tensor<R, E>)> = Vec::with_capacity(4 * old.len());
        for (old, new_cache) in old.iter().zip(new_caches) {
            carries.push((new_cache.ssm.h.tensor().clone(), old.ssm.h.tensor()));
            carries.push((new_cache.ssm.last_u.tensor().clone(), old.ssm.last_u.tensor()));
            if let (Some(new_a), Some(old_a)) = (&new_cache.ssm.angle, &old.ssm.angle) {
                carries.push((new_a.tensor().clone(), old_a.tensor()));
            }
            if let (Some(new_c), Some(old_c)) = (&new_cache.conv, &old.conv) {
                carries.push((new_c.tensor().clone(), old_c.tensor()));
            }
        }
        ms2::freeze_rows_all(&mut carries, grammar_state, atoms)
    }

    /// Snapshot one step's post-freeze caches as host floats (test support;
    /// performs device reads, only called when capture is enabled).
    fn snapshot_caches(state: &DecoderState<R, E>, step: usize) -> Result<StepCarries> {
        if state.carries_in_place() {
            return Err(Error::config(
                "the carry trace needs GenerationWorkspace::capture_carry_trace set before the decoder state is built: this state steps its carries in place"
                    .to_string(),
            ));
        }
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


    /// Validate the config and preflight the memory estimate, allocating
    /// nothing: the first stage of [`Ms2Model::generate_with_hook`], shared
    /// with the device-mode harness.
    pub fn generate_preflight(
        &self,
        batch: &SpectrumBatch,
        table: &DeviceFormulaTable<R, E>,
        config: &GenerationConfig,
    ) -> Result<GeneratePreflight> {
        let _no_grad = crate::autograd::no_grad();
        let atoms = self.config.max_atoms as usize;
        let closures = self.config.max_ring_closures;
        config.validate(atoms, closures as usize)?;
        // Dtype gate (contracts §3.3): every `generate` entry point checks
        // the actual neural element type against the model dtype (Error::Config
        // on mismatch) and the validated set, before any upload, allocation
        // or launch.
        Ms2Capabilities::check_device::<R, E>(table.table.device(), &self.config)?;
        if config.oracle_formula {
            return Err(Error::Unsupported(
                "Ms2Model::generate: oracle_formula is not implemented in V0".to_string(),
            ));
        }
        let spectra_n = batch.len();
        let trajectories = config.trajectories as usize;
        let formulas = config.formulas as usize;
        let steps = config.max_steps as usize;
        let rows = spectra_n * trajectories;
        // Enum lanes `P` (0 for the table source); `B * P > enum_lanes_max`
        // is refused before any launch (V1 §1.4).
        let enum_p = match config.formula_source {
            FormulaSource::Table => 0,
            FormulaSource::Enumerate => {
                let Some(artifacts) = self.enum_artifacts.as_ref() else {
                    return Err(Error::config(
                        "Ms2Model::generate_preflight: formula_source Enumerate needs resident enum artifacts (upload_enum_artifacts)".to_string(),
                    ));
                };
                artifacts.check(&self.config)?;
                let lanes = (spectra_n as u64)
                    .checked_mul(artifacts.p as u64)
                    .ok_or_else(|| {
                        Error::config(format!(
                            "Ms2Model::generate_preflight: batch {spectra_n} times P {} overflows u64",
                            artifacts.p
                        ))
                    })?;
                if lanes > u64::from(config.enum_lanes_max) {
                    return Err(Error::config(format!(
                        "Ms2Model::generate_preflight: B * P {lanes} exceeds enum_lanes_max {} (refused before any launch)",
                        config.enum_lanes_max
                    )));
                }
                artifacts.p
            }
        };
        let enum_bounds_words = match config.formula_source {
            FormulaSource::Table => 0,
            FormulaSource::Enumerate => {
                self.enum_artifacts.as_ref().map(|a| a.bounds.len()).unwrap_or(0) as u64
            }
        };
        // Identity request bound of V1 §4.2, checked before dispatch: at most
        // `K * (K - 1) / 2` pairs per spectrum, each bounded by
        // `identity_work_max` assignments.
        if config.identity == IdentityMode::Graph {
            let k = trajectories as u64;
            let pairs = k
                .checked_mul(k.saturating_sub(1))
                .and_then(|v| v.checked_div(2))
                .ok_or_else(|| {
                    Error::config(format!(
                        "Ms2Model::generate_preflight: K {k} pairs overflow u64"
                    ))
                })?;
            let work = (spectra_n as u64)
                .checked_mul(pairs)
                .and_then(|v| v.checked_mul(u64::from(config.identity_work_max)))
                .ok_or_else(|| {
                    Error::config(
                        "Ms2Model::generate_preflight: identity request work overflows u64".to_string(),
                    )
                })?;
            if work > IDENTITY_REQUEST_WORK_MAX {
                return Err(Error::config(format!(
                    "Ms2Model::generate_preflight: B * K * (K - 1) / 2 * identity_work_max {work} exceeds identity_request_work_max {IDENTITY_REQUEST_WORK_MAX} (refused before any launch)"
                )));
            }
        }
        // Fragment-ion evidence (architecture §2): `evidence = true` requires
        // `ModelConfig::assignment`, else `Error::Config`. The ion work bound
        // `B * F * N * ion_work_max <= ion_request_work_max` (§2.1) is checked
        // before dispatch (before any upload, allocation or launch).
        if config.evidence {
            let Some(assign) = self.config.assignment.as_ref() else {
                return Err(Error::config(
                    "Ms2Model::generate_preflight: evidence needs ModelConfig::assignment (assignment disabled)".to_string(),
                ));
            };
            assign.validate()?;
            let n = u64::from(self.config.n_peaks);
            let work = (spectra_n as u64)
                .checked_mul(formulas as u64)
                .and_then(|v| v.checked_mul(n))
                .and_then(|v| v.checked_mul(u64::from(assign.work_max)))
                .ok_or_else(|| {
                    Error::config(
                        "Ms2Model::generate_preflight: B * F * N * ion_work_max overflows u64".to_string(),
                    )
                })?;
            if work > u64::from(config.ion_request_work_max) {
                return Err(Error::config(format!(
                    "Ms2Model::generate_preflight: B * F * N * ion_work_max {work} exceeds ion_request_work_max {} (refused before dispatch)",
                    config.ion_request_work_max
                )));
            }
        }
        Ms2MemoryEstimate::generation_with_enum(
            &self.config,
            table.rows as u64,
            spectra_n as u64,
            trajectories as u64,
            batch.n_raw as u64,
            steps as u64,
            config.formula_window as u64,
            formulas as u64,
            enum_p as u64,
            enum_bounds_words,
        )?
        .check_limit(config.max_device_bytes)?;
        // Worst-case evidence work of the request (architecture §1.6):
        // `B * M * formula_evidence_work_max * 32` peak tests, checked in
        // `u64`. No refusal is added for it; the dispatch chunking of
        // `formula_evidence` bounds one launch instead.
        let _evidence_work = (spectra_n as u64)
            .checked_mul(config.formula_window as u64)
            .and_then(|v| v.checked_mul(u64::from(config.formula_evidence_work_max)))
            .and_then(|v| v.checked_mul(32))
            .ok_or_else(|| {
                Error::config(format!(
                    "Ms2Model::generate_preflight: B * M * formula_evidence_work_max * 32 overflows u64 (batch {spectra_n}, M {}, work_max {})",
                    config.formula_window, config.formula_evidence_work_max
                ))
            })?;
        Ok(GeneratePreflight {
            spectra_n,
            trajectories,
            formulas,
            steps,
            rows,
            atoms,
            closures,
            window_m: config.formula_window as usize,
            n_raw: batch.n_raw,
            enum_p,
            returned: config.effective_returned() as usize,
            evidence: config.evidence,
            formula_features: self.config.formula_features,
        })
    }

    /// Upload the request (applying `rotate_peaks` first for
    /// [`Control::ShuffledSpectrum`]): the preprocess stage of
    /// [`Ms2Model::generate_with_hook`], shared with the device-mode harness.
    ///
    /// Under [`Control::ShuffledSpectrum`] a batch of fewer than 2 spectra is
    /// [`Error::Config`], never a silent identity.
    pub fn generate_preprocess(
        &self,
        batch: &SpectrumBatch,
        config: &GenerationConfig,
        device: &Device<R>,
    ) -> Result<DeviceSpectra<R, E>> {
        let _no_grad = crate::autograd::no_grad();
        let spectra_n = batch.len();
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
        {
            let _tally = crate::backend::tally_scope("ms2.preprocess");
            DeviceSpectra::upload(upload_batch, device)
        }
    }

    /// Run the spectrum encoder over the selected peaks: the encoder stage of
    /// [`Ms2Model::generate_with_hook`], shared with the device-mode harness.
    pub fn generate_encode(
        &self,
        spectra: &DeviceSpectra<R, E>,
        peaks: &ms2::PeakBuffers<R, E>,
        control: Control,
    ) -> Result<EncoderOutput<R, E>> {
        let _no_grad = crate::autograd::no_grad();
        {
            let _tally = crate::backend::tally_scope("ms2.encoder");
            self.encoder.encode(spectra, peaks, control)
        }
    }

    /// The `Evidence` feature path of the search stage (architecture §1.6):
    /// after `cand` is complete, `evidence_peaks`, `formula_evidence` (with
    /// the config limits), `formula_features` into `cand_feat16`, then two
    /// slices filling `cand_feat` (columns 0..10) and `cand_xfeat` (columns
    /// 10..16). `count_features` is not launched in this layout. The
    /// spectrum's m/z uncertainty travels in `spec [B, 2]`, built by
    /// [`DeviceSpectra::evidence_spec`] from the uploaded rows (so donor
    /// peaks travel with the donor's uncertainty), the same source
    /// [`Ms2Model::generate_ion`] builds it from.
    ///
    /// Shared with the training prefix (which passes its own config limits).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn generate_search_evidence(
        spectra: &DeviceSpectra<R, E>,
        table: &DeviceFormulaTable<R, E>,
        formula: &mut ms2::FormulaBuffers<R, E>,
        peaks: &ms2::PeakBuffers<R, E>,
        batch_len: usize,
        window_m: usize,
        work_max: u32,
        dispatch_max: u64,
        h_cap_max: u32,
        tol_max: u32,
    ) -> Result<()> {
        let device = spectra.meta.device().clone();
        debug_assert_eq!(batch_len, spectra.batch);
        debug_assert_eq!(formula.cand.shape().dim(1), window_m);
        let spec_t = spectra.evidence_spec(&device)?;
        {
            let Some(ev_peaks) = formula.ev_peaks.as_mut() else {
                return Err(Error::shape(
                    "Ms2Model::generate_search_evidence: Evidence layout needs ev_peaks (bucket without evidence buffers)".to_string(),
                ));
            };
            let Some(ev_w) = formula.ev_w.as_mut() else {
                return Err(Error::shape(
                    "Ms2Model::generate_search_evidence: Evidence layout needs ev_w (bucket without evidence buffers)".to_string(),
                ));
            };
            ms2_formula_evidence::evidence_peaks(
                &peaks.kept,
                &peaks.kept_f,
                &spectra.meta,
                &spec_t,
                ev_peaks,
                ev_w,
            )?;
        }
        {
            let Some(ev_peaks) = formula.ev_peaks.as_ref() else {
                return Err(Error::shape(
                    "Ms2Model::generate_search_evidence: Evidence layout needs ev_peaks (bucket without evidence buffers)".to_string(),
                ));
            };
            let Some(ev_w) = formula.ev_w.as_ref() else {
                return Err(Error::shape(
                    "Ms2Model::generate_search_evidence: Evidence layout needs ev_w (bucket without evidence buffers)".to_string(),
                ));
            };
            let Some(cand_ev) = formula.cand_ev.as_mut() else {
                return Err(Error::shape(
                    "Ms2Model::generate_search_evidence: Evidence layout needs cand_ev (bucket without evidence buffers)".to_string(),
                ));
            };
            ms2_formula_evidence::formula_evidence(
                &formula.cand,
                ev_peaks,
                ev_w,
                &spectra.meta,
                &spec_t,
                cand_ev,
                work_max,
                dispatch_max,
                h_cap_max,
                tol_max,
            )?;
        }
        {
            let Some(cand_ev) = formula.cand_ev.as_ref() else {
                return Err(Error::shape(
                    "Ms2Model::generate_search_evidence: Evidence layout needs cand_ev (bucket without evidence buffers)".to_string(),
                ));
            };
            let Some(feat16) = formula.cand_feat16.as_mut() else {
                return Err(Error::shape(
                    "Ms2Model::generate_search_evidence: Evidence layout needs cand_feat16 (bucket without evidence buffers)".to_string(),
                ));
            };
            ms2_formula_evidence::formula_features(
                &formula.cand,
                cand_ev,
                &spectra.meta,
                &table.log_table,
                feat16,
            )?;
        }
        let feat16_t = formula.cand_feat16.as_ref().expect("checked above").clone();
        formula.cand_feat = movement::slice(&feat16_t, 2, 0, 10)?;
        formula.cand_xfeat = Some(movement::slice(&feat16_t, 2, 10, 6)?);
        Ok(())
    }

    /// Run the formula window, scoring, top-F, trajectory allocation,
    /// trajectory initialisation and the formula broadcast: the search stage
    /// of [`Ms2Model::generate_with_hook`], shared with the device-mode
    /// harness.
    ///
    /// In the `Evidence` layout (architecture §1.6) the search stage runs,
    /// after `cand` is complete, `evidence_peaks`, `formula_evidence` (with
    /// the config limits), `formula_features` into `cand_feat16`, then two
    /// slices filling `cand_feat` (columns 0..10) and `cand_xfeat` (columns
    /// 10..16); `count_features` is not launched in that layout. With
    /// `Counts` the launch sequence is exactly today's.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_search(
        &self,
        spectra: &DeviceSpectra<R, E>,
        pool: &Var<R, E>,
        table: &DeviceFormulaTable<R, E>,
        formula: &mut ms2::FormulaBuffers<R, E>,
        peaks: &ms2::PeakBuffers<R, E>,
        traj_alloc: &mut IdTensor<R>,
        lane_stats: &IdTensor<R>,
        offsets: &IdTensor<R>,
        host_batch: &SpectrumBatch,
        traj_meta: &mut IdTensor<R>,
        replay: &mut IdTensor<R>,
        actions: &mut IdTensor<R>,
        traj_formula: &mut Tensor<R, E>,
        batch_len: usize,
        window_m: usize,
        formulas: usize,
        trajectories: usize,
        steps: usize,
        atoms: usize,
        metadata_blind: bool,
        rows_visited_max: u32,
        rows_scored_max: u32,
        config: &GenerationConfig,
    ) -> Result<()> {
        let _no_grad = crate::autograd::no_grad();
        {
            let _tally = crate::backend::tally_scope("ms2.search");
            match config.formula_source {
                FormulaSource::Table => {
                    ms2::formula_window(
                        &table.table,
                        &spectra.meta,
                        table.max_error,
                        rows_visited_max,
                        rows_scored_max,
                        formula,
                    )?;
                    ms2::formula_gather(
                        &formula.window,
                        &table.table,
                        &table.counts,
                        &mut formula.cand,
                    )?;
                }
                FormulaSource::Enumerate => {
                    let Some(artifacts) = self.enum_artifacts.as_ref() else {
                        return Err(Error::config(
                            "Ms2Model::generate_search: formula_source Enumerate needs resident enum artifacts".to_string(),
                        ));
                    };
                    artifacts.check(&self.config)?;
                    let scored_cap =
                        config.formula_rows_scored_max.min(config.formula_window);
                    let meta_host = build_enum_meta(
                        host_batch,
                        artifacts.domain_max_error,
                        config.enum_lane_visits_max,
                        scored_cap,
                    );
                    // T6 memo: a fully cached batch uploads `cand` and
                    // `counters` (two uploads) and launches no enumeration
                    // kernel (count, offsets, fill and pad are all skipped;
                    // `lane_stats` / `offsets` are not touched). A partially
                    // cached batch takes the device path (no mixing), and
                    // everything downstream sees bit-identical `cand` and
                    // `counters`. No device read is added by the cache.
                    let mut served = false;
                    if let Some(cache) = self.enum_cache.as_ref() {
                        let keys: Vec<[u32; 8]> = meta_host
                            .chunks_exact(8)
                            .map(|r| {
                                let mut k = [0u32; 8];
                                k.copy_from_slice(r);
                                k
                            })
                            .collect();
                        self.enum_cache_lookups.fetch_add(1, Ordering::Relaxed);
                        if let Some((cand_host, counters_host)) =
                            cache.expand_batch(&keys, window_m)
                        {
                            // The lane preflight and every other refusal of
                            // the uncached path still applies with a cache:
                            // refuse here exactly as the wrappers would, so
                            // behaviour does not depend on cache state.
                            validate_enum_dispatch(
                                batch_len,
                                artifacts.p,
                                window_m,
                                config.enum_lanes_max,
                            )
                            .map_err(|e| match e {
                                Error::Shape(msg) => Error::shape(format!(
                                    "Ms2Model::generate_search (cached): {msg}"
                                )),
                                other => other,
                            })?;
                            let device = table.table.device().clone();
                            formula.cand = IdTensor::from_slice(
                                &cand_host,
                                vec![batch_len, window_m, 13],
                                &device,
                            )?;
                            formula.counters = IdTensor::from_slice(
                                &counters_host,
                                vec![batch_len, 5],
                                &device,
                            )?;
                            self.enum_cache_hits.fetch_add(1, Ordering::Relaxed);
                            served = true;
                        }
                    }
                    if !served {
                        let device = table.table.device().clone();
                        let meta_t =
                            IdTensor::from_slice(&meta_host, vec![batch_len, 8], &device)?;
                        let launch = EnumLaunch::from_chemistry();
                        launch.count(
                            &meta_t,
                            &artifacts.rare,
                            &artifacts.bounds,
                            lane_stats,
                            config.enum_lanes_max,
                            config.enum_dispatch_visits_max,
                            config.enum_lane_visits_max,
                        )?;
                        enum_offsets(
                            lane_stats,
                            &meta_t,
                            offsets,
                            &formula.counters,
                            scored_cap,
                            window_m,
                            config.enum_lanes_max,
                        )?;
                        launch.fill(
                            &meta_t,
                            &artifacts.rare,
                            &artifacts.bounds,
                            offsets,
                            &formula.cand,
                            scored_cap,
                            config.enum_lanes_max,
                            config.enum_dispatch_visits_max,
                            config.enum_lane_visits_max,
                        )?;
                        cand_pad(
                            &formula.counters,
                            &formula.cand,
                            artifacts.p,
                            config.enum_lanes_max,
                        )?;
                    }
                }
            }
            if matches!(self.config.formula_features, FormulaFeatures::Evidence) {
                // Task E5F: host-known bounds without a read. The
                // hydrogen bound comes from the scoring source (table rows
                // or enum artifacts); the tolerance bound is the EXACT
                // uploaded batch's bound (`DeviceSpectra::uploaded_tol_max`),
                // so under `ShuffledSpectrum` a row's donor peaks are sized
                // with that row's own ppm. Never size from the pre-rotation
                // request batch.
                let h_cap_max = match config.formula_source {
                    FormulaSource::Table => table.hydrogen_cap_max(),
                    FormulaSource::Enumerate => self
                        .enum_artifacts
                        .as_ref()
                        .map(|a| a.hydrogen_cap_max())
                        .unwrap_or(u32::MAX),
                };
                let tol_max = spectra.uploaded_tol_max();
                Self::generate_search_evidence(
                    spectra,
                    table,
                    formula,
                    peaks,
                    batch_len,
                    window_m,
                    config.formula_evidence_work_max,
                    config.formula_evidence_dispatch_max,
                    h_cap_max,
                    tol_max,
                )?;
            } else {
                ms2::count_features(
                    &formula.cand.reshape(vec![batch_len * window_m, 13])?,
                    &table.log_table,
                    &mut formula.cand_feat.reshape(vec![batch_len * window_m, 10])?,
                    13,
                )?;
            }
            let scored = self.formula.score(formula, pool)?;
            ms2::formula_top(&scored.log_prob.tensor().clone(), &formula.cand, formula)?;
            ms2::formula_top_counts(&formula.top, &formula.cand, &mut formula.top_counts)?;
            // Trajectory allocation (V1 §3.2): exactly one launch per call,
            // writing the per-trajectory slot/source/counts the stages below
            // read. With `RoundRobin` the slot is `k mod count`, the V0 rule.
            let alloc_mode = match config.allocation {
                AllocationMode::RoundRobin => {
                    crate::models::ms2::allocate::ALLOC_ROUND_ROBIN
                }
                AllocationMode::Proportional => {
                    crate::models::ms2::allocate::ALLOC_PROPORTIONAL
                }
            };
            ms2_identity::allocate(
                &formula.top,
                &formula.top_counts,
                &formula.top_log_prob,
                &formula.top_count,
                traj_alloc,
                alloc_mode,
            )?;
            ms2::init_trajectories(
                traj_alloc,
                &spectra.meta,
                traj_meta,
                replay,
                actions,
                batch_len,
                trajectories,
                steps,
                atoms,
                metadata_blind,
            )?;
            ms2::trajectory_formula(
                &scored.embedding.tensor().clone(),
                traj_alloc,
                &formula.top,
                traj_formula,
                batch_len,
                window_m,
                formulas,
                trajectories,
            )?;
        }
        Ok(())
    }

    /// Fragment-ion assignment for generation (architecture §2.1–§2.2): after
    /// top-F, `ion_assign` for the `F` retained formulas plus the head's
    /// `log_prob` (no labels at inference). Returns `(None, None, None)` when
    /// `evidence` is false: zero launches. Otherwise one `ion_assign` plus the
    /// head launches. The spectrum's m/z uncertainty travels in `spec [B, 2]`,
    /// built by [`DeviceSpectra::evidence_spec`] from the uploaded rows (the
    /// same source the formula-evidence stage uses).
    #[allow(clippy::too_many_arguments)]
    fn generate_ion(
        &self,
        spectra: &DeviceSpectra<R, E>,
        encoded: &EncoderOutput<R, E>,
        table: &DeviceFormulaTable<R, E>,
        top_counts: &IdTensor<R>,
        kept: &IdTensor<R>,
        batch_len: usize,
        formulas: usize,
        config: &GenerationConfig,
        device: &Device<R>,
    ) -> Result<(
        Option<IdTensor<R>>,
        Option<IdTensor<R>>,
        Option<Var<R, E>>,
    )> {
        if !config.evidence {
            return Ok((None, None, None));
        }
        let Some(head) = self.assignment.as_ref() else {
            return Err(Error::config(
                "Ms2Model::generate_ion: evidence needs ModelConfig::assignment".to_string(),
            ));
        };
        let Some(acfg) = self.config.assignment.as_ref() else {
            return Err(Error::config(
                "Ms2Model::generate_ion: evidence needs ModelConfig::assignment".to_string(),
            ));
        };
        let j = acfg.hypotheses as usize;
        let work_max = acfg.work_max;
        let n = self.config.n_peaks as usize;
        // Bound already checked in preflight; re-check defensively with the
        // request's configured limit (no hardcoded constant).
        let work = (batch_len as u64)
            .checked_mul(formulas as u64)
            .and_then(|v| v.checked_mul(n as u64))
            .and_then(|v| v.checked_mul(u64::from(work_max)))
            .ok_or_else(|| {
                Error::config(
                    "Ms2Model::generate_ion: B * F * N * work_max overflows u64".to_string(),
                )
            })?;
        if work > u64::from(config.ion_request_work_max) {
            return Err(Error::config(format!(
                "Ms2Model::generate_ion: B * F * N * ion_work_max {work} exceeds ion_request_work_max {} (refused before dispatch)",
                config.ion_request_work_max
            )));
        }
        let spec_t = spectra.evidence_spec(device)?;
        let mut ion_t = IdTensor::empty(vec![batch_len, formulas, n, j, 12], device);
        let mut ion_meta_t = IdTensor::empty(vec![batch_len, formulas, n, 4], device);
        {
            let _tally = crate::backend::tally_scope("ms2.ion_assign");
            crate::tensor::ops::ms2_ion::ion_assign(
                top_counts,
                kept,
                &spectra.meta,
                &spec_t,
                &mut ion_t,
                &mut ion_meta_t,
                work_max,
            )?;
        }
        let out = {
            let _tally = crate::backend::tally_scope("ms2.assign");
            head.log_prob(&self.formula, &table.log_table, &ion_t, &ion_meta_t, &encoded.x)?
        };
        Ok((Some(ion_t), Some(ion_meta_t), Some(out.log_prob)))
    }

    /// Evidence for candidates (architecture §2.4): after validation,
    /// `ion_evidence` per trajectory using its formula slot (`traj_formula`
    /// allocation), then the reranker's future `evidence_f [B*K, 2]` with the
    /// small kernel. Uses the scored (best-E) selection kernel: the at most
    /// `E = 4` records of largest assignment probability (ties by smaller
    /// peak position). Zero launches when `evidence` is false (the bucket's
    /// all-zero buffer stays, status 0 = unassigned).
    #[allow(clippy::too_many_arguments)]
    fn generate_evidence(
        &self,
        actions: &IdTensor<R>,
        traj_alloc: &IdTensor<R>,
        meta: &IdTensor<R>,
        kept: &IdTensor<R>,
        ion: &IdTensor<R>,
        ion_meta: &IdTensor<R>,
        log_prob: &Tensor<R, E>,
        evidence: &mut IdTensor<R>,
        evidence_f: &mut Tensor<R, E>,
        traj_slot: &mut IdTensor<R>,
        spectra_n: usize,
        trajectories: usize,
        steps: usize,
        atoms: usize,
        config: &GenerationConfig,
    ) -> Result<()> {
        if !config.evidence {
            return Ok(());
        }
        let _tally = crate::backend::tally_scope("ms2.finalize");
        // `[R, 2]` slots from the allocation + spectrum adducts.
        crate::tensor::ops::ms2_ion::ion_traj_slot(
            traj_alloc,
            meta,
            traj_slot,
            trajectories as u32,
        )?;
        // Best-E evidence rows (second selection kernel).
        crate::tensor::ops::ms2_ion::ion_evidence_scored(
            actions,
            traj_slot,
            ion,
            ion_meta,
            log_prob,
            evidence,
            steps as u32,
            atoms as u32,
            trajectories as u32,
        )?;
        // `evidence_f` for the future reranker (small kernel, ≤6 arrays).
        crate::tensor::ops::ms2_ion::ion_evidence_features(
            evidence,
            log_prob,
            traj_slot,
            kept,
            meta,
            evidence_f,
            trajectories as u32,
        )?;
        let _ = spectra_n;
        Ok(())
    }

    /// Build the decoder start state (K/V projections, cache fills) and the
    /// conditioning setup: the decoder-init stage of
    /// [`Ms2Model::generate_with_hook`], shared with the device-mode harness.
    pub fn generate_decoder_init(
        &self,
        encoded: &EncoderOutput<R, E>,
        rows: usize,
        device: &Device<R>,
    ) -> Result<(DecoderState<R, E>, Tensor<R, E>)> {
        self.generate_decoder_init_mode(encoded, rows, device, false)
    }

    /// [`Ms2Model::generate_decoder_init`] with the step form chosen:
    /// `composed = false` builds the fused state the production loop drives
    /// (`Ms2Decoder::step_packed`), `composed = true` the composed reference
    /// state (`Ms2Decoder::step_logits` plus the pack copies), which the
    /// parity tests compare against.
    pub fn generate_decoder_init_mode(
        &self,
        encoded: &EncoderOutput<R, E>,
        rows: usize,
        device: &Device<R>,
        composed: bool,
    ) -> Result<(DecoderState<R, E>, Tensor<R, E>)> {
        self.decoder_init(encoded, rows, device, composed, true)
    }

    /// The decoder state of a generation call. `observed` says whether the
    /// caller reads the recurrent carries (a carry trace, or a harness that
    /// holds the state): when nobody does, the fused loop steps them in place
    /// ([`Ms2Decoder::start_state_unobserved`]) and has nothing to freeze.
    fn decoder_init(
        &self,
        encoded: &EncoderOutput<R, E>,
        rows: usize,
        device: &Device<R>,
        composed: bool,
        observed: bool,
    ) -> Result<(DecoderState<R, E>, Tensor<R, E>)> {
        let _no_grad = crate::autograd::no_grad();
        let state = if composed {
            self.decoder.start_state(encoded, rows, device)?
        } else if observed {
            self.decoder.start_state_fused(encoded, rows, device)?
        } else {
            self.decoder.start_state_unobserved(encoded, rows, device)?
        };
        let bond_table = self.decoder.bond_by_type_value();
        Ok((state, bond_table))
    }
    /// Run one sampling step (`1..max_steps`) with carry freeze: a single
    /// decode step of [`Ms2Model::generate_with_hook`], shared with the
    /// device-mode harness.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_decode_step(
        &self,
        encoded: &EncoderOutput<R, E>,
        traj_formula: &Var<R, E>,
        actions: &mut IdTensor<R>,
        step_token: &mut IdTensor<R>,
        replay: &mut IdTensor<R>,
        logits: &mut Tensor<R, E>,
        traj_meta: &IdTensor<R>,
        decoder_state: &mut DecoderState<R, E>,
        bond_table: &Tensor<R, E>,
        atom_table: &IdTensor<R>,
        step: usize,
        seed_lo: u32,
        seed_hi: u32,
        temperature: f32,
        steps: usize,
        atoms: usize,
        closures: u32,
        trajectories: usize,
        rows: usize,
    ) -> Result<()> {
        let _no_grad = crate::autograd::no_grad();
        let _tally = crate::backend::tally_scope("ms2.step");
        // Shared prologue: the next input token feeds both the fused and
        // the composed step.
        ms2::step_token(actions, step_token, steps, atoms)?;
        let old_caches: Vec<MixerCache<R, E>> = decoder_state.caches.to_vec();
        if decoder_state.fused.is_some() {
            // The fused step: the head values land in the packed row
            // directly, and stopped rows are frozen in place afterwards.
            self.decoder.step_packed(
                encoded,
                traj_formula,
                step_token,
                step - 1,
                replay,
                decoder_state,
                trajectories,
                logits,
            )?;
            ms2::sample_step(
                logits,
                bond_table,
                traj_meta,
                replay,
                actions,
                step as u32,
                seed_lo,
                seed_hi,
                temperature,
                steps,
                atoms,
                closures,
                atom_table,
            )?;
            // Rows stopped before or at this step keep their old carries;
            // carries stepped in place are read by nobody and have no old
            // copy to keep.
            if !decoder_state.carries_in_place() {
                Self::freeze_caches_in_place(&old_caches, &decoder_state.caches, replay, atoms)?;
            }
            return Ok(());
        }
        composed_decode_step(
            &self.decoder,
            encoded,
            traj_formula,
            actions,
            step_token,
            replay,
            logits,
            traj_meta,
            decoder_state,
            bond_table,
            atom_table,
            step,
            seed_lo,
            seed_hi,
            temperature,
            steps,
            atoms,
            closures,
            trajectories,
            rows,
        )
    }

    /// Validate the trajectories in place, then resolve graph identity when
    /// requested: the validate stage of
    /// [`Ms2Model::generate_with_hook`], shared with the device-mode harness.
    ///
    /// With `identity = Graph` (V1 §4.2) `graph_hash` runs after validation,
    /// then `graph_identity`: two launches writing `identity [B*K, 2]` (bits
    /// to OR into the candidates' `status` on the host side of the readout,
    /// resolution 1 or 2). With `TraceOnly` nothing is launched and the
    /// readout keeps `identity_resolution` 0.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_validate(
        &self,
        actions: &mut IdTensor<R>,
        traj_meta: &IdTensor<R>,
        scratch: &mut IdTensor<R>,
        atom_table: &IdTensor<R>,
        graph_hash: &mut IdTensor<R>,
        graph_scratch: &mut IdTensor<R>,
        identity: &mut IdTensor<R>,
        identity_scratch: &mut IdTensor<R>,
        spectra_n: usize,
        trajectories: usize,
        steps: usize,
        atoms: usize,
        closures: u32,
        identity_mode: IdentityMode,
        identity_work_max: u32,
    ) -> Result<()> {
        let _no_grad = crate::autograd::no_grad();
        let _tally = crate::backend::tally_scope("ms2.finalize");
        ms2::validate_trajectories(
            actions,
            traj_meta,
            scratch,
            atom_table,
            spectra_n,
            trajectories,
            steps,
            atoms,
            closures,
        )?;
        if identity_mode == IdentityMode::Graph {
            let atoms_cap = atoms as u32;
            let bonds_cap = atoms.saturating_sub(1) as u32 + closures;
            ms2_identity::graph_hash(
                actions,
                graph_hash,
                graph_scratch,
                steps,
                atoms_cap,
                closures,
                u32::MAX,
            )?;
            ms2_identity::graph_identity(
                actions,
                graph_hash,
                graph_scratch,
                identity,
                identity_scratch,
                steps,
                atoms_cap,
                bonds_cap,
                trajectories,
                identity_work_max,
            )?;
        }
        Ok(())
    }

    /// Per-spectrum fields shared by the trajectory-ordered and packed
    /// readouts, from one batched read: the D4 host reconciliation for
    /// enumeration statuses plus the retained-mass and peak summaries.
    /// `batch_opt` carries the host batch for enumeration reconciliation
    /// (`None` keeps the device counters; the profiler is table-only).
    #[allow(clippy::too_many_arguments)]
    fn per_spectrum_output(
        &self,
        host_status: &[u32],
        counters_h: &[u32],
        summary_h: &[u32],
        top_count_h: &[u32],
        top_lp_h: &[f32],
        stats_h: &[f32],
        batch_opt: Option<&SpectrumBatch>,
        config: &GenerationConfig,
        spectra_n: usize,
        formulas: usize,
    ) -> Result<PerSpectrumOut> {
        // D4: single host reconciliation for enumeration statuses, run
        // before validation and used by every readout path. The device
        // counters already carry the correct absent/exhausted distinction
        // (absent only for a completed empty search); only the unknown,
        // overflow and wide-window scope cases need host truth, and no
        // `formula_absent` is manufactured for an incomplete search.
        let enum_domain_error: Option<u32> = match config.formula_source {
            FormulaSource::Table => None,
            FormulaSource::Enumerate => batch_opt.and_then(|_| {
                self.enum_artifacts
                    .as_ref()
                    .map(|a| a.domain_max_error)
            }),
        };
        let mut out = PerSpectrumOut {
            request_status: vec![0; spectra_n],
            rows_visited: vec![0; spectra_n],
            rows_joined: vec![0; spectra_n],
            rows_scored: vec![0; spectra_n],
            formula_support_complete: vec![0; spectra_n],
            formula_mass_retained: vec![0.0; spectra_n],
            peaks_kept: vec![0; spectra_n],
            intensity_retained: vec![0.0; spectra_n],
        };
        for b in 0..spectra_n {
            // The union of the host validation bits, the peak-selection
            // summary and the formula-search counters. No absent is
            // manufactured here: `formula_absent` means a completed empty
            // search (device counters), plus the explicit unknown-precision
            // rule below.
            let mut rs = host_status[b] | summary_h[b * 2 + 1] | counters_h[b * 5 + 3];
            let mut complete = counters_h[b * 5 + 4].min(1) as u8;
            if let Some(domain_max_error) = enum_domain_error {
                let batch = batch_opt.expect("readout: Enumerate reconciliation needs the host batch");
                let unc = batch.precursor_uncertainty_udalton[b];
                if unc == u32::MAX {
                    // Unknown precision: unavailable (host) + absent
                    // (counters), complete = 0 (contracts §9).
                    complete = 0;
                } else {
                    let precursor = batch.precursor_mz_udalton[b];
                    let adduct = batch.adduct[b];
                    match parent_mass(precursor, adduct) {
                        Err(_) => {
                            // Arithmetic overflow: mass_overflow with
                            // complete = 0 and no absent.
                            rs |= request_status::MASS_OVERFLOW;
                            rs &= !request_status::FORMULA_ABSENT;
                            complete = 0;
                        }
                        Ok(parent) => {
                            let ppm = batch.precursor_tolerance(b);
                            let tol = tolerance_u32(precursor, ppm).unwrap_or(0);
                            let bound = unc.saturating_add(1);
                            let half = tol
                                .saturating_add(bound)
                                .saturating_add(domain_max_error);
                            if half > DEVICE_HALF_MAX {
                                // Too-wide window: exhausted with complete = 0
                                // and no absent (device lanes with budget 0
                                // already exhaust; clear absent defensively).
                                rs &= !request_status::FORMULA_ABSENT;
                                rs |= request_status::FORMULA_SEARCH_EXHAUSTED;
                                complete = 0;
                            }
                            let _ = parent;
                        }
                    }
                }
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
            out.formula_support_complete[b] = complete;
            // V1 §1.2: `formula_mass_retained` sums only the first `top_count`
            // entries (a padding entry's `top_log_prob` is 0 and is never
            // summed).
            let count = top_count_h[b] as usize;
            out.formula_mass_retained[b] = if out.formula_support_complete[b] == 1 {
                (0..count)
                    .map(|f| top_lp_h[b * formulas + f].exp())
                    .sum()
            } else {
                0.0
            };
            out.peaks_kept[b] = summary_h[b * 2];
            out.intensity_retained[b] = stats_h[b * 3 + 2];
        }
        Ok(out)
    }

    /// The single batched [`read_all`] of the whole call plus the host-side
    /// batch building: the readout stage of [`Ms2Model::generate_with_hook`],
    /// shared with the device-mode harness.
    ///
    /// Each trajectory's formula fields come from its allocation record
    /// (`traj_formula [B, K, 12]`: retained slot → rank/row/counts/log-prob)
    /// rather than recomputing `k mod count`. With `identity = Graph` the
    /// identity bits are ORed into the candidates' `status` and
    /// `identity_resolution` is filled (1 or 2); with `TraceOnly` nothing was
    /// launched and the resolution stays 0.
    ///
    /// [`read_all`]: crate::tensor::ops::index::read_all
    #[allow(clippy::too_many_arguments)]
    pub fn generate_readout(
        &self,
        actions: &IdTensor<R>,
        formula: &ms2::FormulaBuffers<R, E>,
        peaks_summary: &IdTensor<R>,
        traj_alloc: &IdTensor<R>,
        identity: &IdTensor<R>,
        use_graph: bool,
        top_log_prob: &Tensor<R, E>,
        peaks_stats: &Tensor<R, E>,
        host_status: &[u32],
        spectrum_ids: &[u64],
        batch_opt: Option<&SpectrumBatch>,
        config: &GenerationConfig,
        spectra_n: usize,
        trajectories: usize,
        formulas: usize,
        steps: usize,
        atoms: usize,
        closures: usize,
    ) -> Result<CandidateBatch> {
        let _no_grad = crate::autograd::no_grad();
        let rows = spectra_n * trajectories;
        // The buffer list and the `readout` estimate item share one layout
        // ([`Ms2MemoryEstimate::generation_readout_counts`]): the lengths
        // below are asserted against it, so the two cannot drift.
        let (want_ids, want_floats) = Ms2MemoryEstimate::generation_readout_counts(
            spectra_n as u64,
            trajectories as u64,
            formulas as u64,
            steps as u64,
            atoms as u64,
        )?;
        let got_id_lens = [
            actions.len(),
            formula.top.len(),
            formula.top_count.len(),
            formula.counters.len(),
            peaks_summary.len(),
            traj_alloc.len(),
            identity.len(),
        ];
        for (i, (got, want)) in got_id_lens.iter().zip(want_ids.iter()).enumerate() {
            assert_eq!(
                *got as u64, *want,
                "generate readout id buffer {i} length drifted from the shared layout"
            );
        }
        let got_float_lens = [top_log_prob.len(), peaks_stats.len()];
        for (i, (got, want)) in got_float_lens.iter().zip(want_floats.iter()).enumerate() {
            assert_eq!(
                *got as u64, *want,
                "generate readout float buffer {i} length drifted from the shared layout"
            );
        }
        let (ids, floats) = read_all(
            &[
                actions,
                &formula.top,
                &formula.top_count,
                &formula.counters,
                peaks_summary,
                traj_alloc,
                identity,
            ],
            &[top_log_prob, peaks_stats],
        )?;
        let actions_h = &ids[0];
        let top_h = &ids[1];
        let top_lp_h = &floats[0];
        let stats_h = &floats[1];
        let top_count_h = &ids[2];
        let counters_h = &ids[3];
        let summary_h = &ids[4];
        let traj_h = &ids[5];
        let identity_h = &ids[6];
        let record_width = ms2::sample_record_width(steps, atoms);
        // V1 §1.2: `formula_source` is 0 (table) for every spectrum here;
        // `Enumerate` is rejected by config validation until §1.4.
        let source_id = match config.formula_source {
            FormulaSource::Table => 0u8,
            FormulaSource::Enumerate => 1u8,
        };
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
            formula_counts: vec![0; rows * 10],
            formula_source: vec![source_id; spectra_n],
            formula_rank: vec![NO_FORMULA; rows],
            evidence_count: vec![0; rows],
            evidence_peak_id: vec![0; rows * super::contract::EVIDENCE_CAP],
            evidence_hypothesis: vec![0; rows * super::contract::EVIDENCE_CAP],
            evidence_shift: vec![0; rows * super::contract::EVIDENCE_CAP],
            evidence_residual: vec![0; rows * super::contract::EVIDENCE_CAP],
            evidence_log_prob: vec![0.0; rows * super::contract::EVIDENCE_CAP],
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
            // The trajectory's formula fields from its allocation record
            // (V1 §3.2): the retained slot indexes `top` (rank/row/log-prob)
            // and carries the 10 counts. With `RoundRobin` the slot is
            // `k mod count`, the V0 provenance, bit for bit.
            let tbase = r * 12;
            let slot = traj_h[tbase];
            if slot == u32::MAX || (slot as usize) >= formulas {
                out.formula_log_prob[r] = 0.0;
                out.formula_rank[r] = NO_FORMULA;
            } else {
                let fslot = slot as usize;
                out.formula_log_prob[r] = top_lp_h[b * formulas + fslot];
                out.formula_rank[r] = top_h[(b * formulas + fslot) * 2 + 1];
                for e in 0..10 {
                    out.formula_counts[r * 10 + e] = traj_h[tbase + 2 + e] as u16;
                }
            }
            // Graph identity (V1 §4.2): the bits are ORed into the
            // candidate status and the resolution is filled (1 or 2). With
            // `TraceOnly` nothing was launched and the resolution stays 0.
            if use_graph {
                out.status[r] |= identity_h[r * 2];
                out.identity_resolution[r] = identity_h[r * 2 + 1] as u8;
            }
        }
        let per_spectrum = self.per_spectrum_output(
            host_status,
            counters_h,
            summary_h,
            top_count_h,
            top_lp_h,
            stats_h,
            batch_opt,
            config,
            spectra_n,
            formulas,
        )?;
        for b in 0..spectra_n {
            out.request_status[b] = per_spectrum.request_status[b];
            out.rows_visited[b] = per_spectrum.rows_visited[b];
            out.rows_joined[b] = per_spectrum.rows_joined[b];
            out.rows_scored[b] = per_spectrum.rows_scored[b];
            out.formula_support_complete[b] = per_spectrum.formula_support_complete[b];
            out.formula_mass_retained[b] = per_spectrum.formula_mass_retained[b];
            out.peaks_kept[b] = per_spectrum.peaks_kept[b];
            out.intensity_retained[b] = per_spectrum.intensity_retained[b];
            let rs = out.request_status[b];
            // Host-side failure enforcement: a fatal request carries no
            // trace, and a spectrum with no scored formula abstains for any
            // reason (contracts §9: no trajectory starts, every record
            // carries request_failed), so device records (possibly started
            // under the metadata-only bypass) are replaced by failed records
            // here.
            if rs & request_status::FATAL_MASK != 0 || out.rows_scored[b] == 0 {
                for kk in 0..trajectories {
                    let r = b * trajectories + kk;
                    out.length[r] = 0;
                    out.formula_row[r] = NO_FORMULA;
                    out.formula_log_prob[r] = 0.0;
                    out.trace_log_prob[r] = 0.0;
                    out.status[r] = candidate_status::REQUEST_FAILED;
                    // No graph to resolve on a failed record: the resolution
                    // stays 0 whatever the identity mode.
                    out.identity_resolution[r] = 0;
                    out.evidence_status[r] = 0;
                    out.evidence_count[r] = 0;
                    for v in &mut out.actions[r * steps * 4..(r + 1) * steps * 4] {
                        *v = 0;
                    }
                    for v in &mut out.open_valence[r * atoms..(r + 1) * atoms] {
                        *v = 0;
                    }
                    for v in &mut out.formula_counts[r * 10..(r + 1) * 10] {
                        *v = 0;
                    }
                    out.formula_rank[r] = NO_FORMULA;
                    let ecap = super::contract::EVIDENCE_CAP;
                    for v in &mut out.evidence_peak_id[r * ecap..(r + 1) * ecap] {
                        *v = 0;
                    }
                    for v in &mut out.evidence_hypothesis[r * ecap..(r + 1) * ecap] {
                        *v = 0;
                    }
                    for v in &mut out.evidence_shift[r * ecap..(r + 1) * ecap] {
                        *v = 0;
                    }
                    for v in &mut out.evidence_residual[r * ecap..(r + 1) * ecap] {
                        *v = 0;
                    }
                    for v in &mut out.evidence_log_prob[r * ecap..(r + 1) * ecap] {
                        *v = 0.0;
                    }
                }
            }
        }
        out.validate()?;
        Ok(out)
    }

    /// Readout with fragment-ion evidence (architecture §2.4): the single
    /// batched `read_all` of the whole call plus host-side evidence assembly.
    ///
    /// Reads the base buffers of [`Ms2Model::generate_readout`] plus
    /// `evidence [B*K, 18]`, `kept [B, N, 3]` and the assignment `log_prob
    /// [B, F, N, J + 1]` in the same batched read (still exactly one runtime
    /// read), then fills `evidence_count`, `evidence_peak_id` (ORIGINAL peak
    /// ids: kept position → raw index → `peak_id` on the host at readout),
    /// `evidence_hypothesis`, `evidence_shift`, `evidence_residual` and
    /// `evidence_log_prob` (`[B*K, E]`, zero beyond the count) plus
    /// `evidence_status` (0, 1, 2, bit 7 for incomplete support). With
    /// `evidence = false` this is never called (the V0 path stays).
    #[allow(clippy::too_many_arguments)]
    fn generate_readout_evidence(
        &self,
        workspace: &mut GenerationWorkspace<R, E>,
        host_status: &[u32],
        spectrum_ids: &[u64],
        batch: &SpectrumBatch,
        config: &GenerationConfig,
        log_prob: Option<&Tensor<R, E>>,
        pre: &GeneratePreflight,
        device: &Device<R>,
    ) -> Result<CandidateBatch> {
        let _no_grad = crate::autograd::no_grad();
        let bucket = self.bucket_for(workspace, pre, device)?;
        let spectra_n = pre.spectra_n;
        let trajectories = pre.trajectories;
        let formulas = pre.formulas;
        let steps = pre.steps;
        let atoms = pre.atoms;
        let closures = pre.closures as usize;
        let rows = spectra_n * trajectories;
        let Some(lp_t) = log_prob else {
            return Err(Error::config(
                "Ms2Model::generate_readout_evidence: evidence needs log_prob".to_string(),
            ));
        };
        let acfg = self.config.assignment.as_ref().ok_or_else(|| {
            Error::config(
                "Ms2Model::generate_readout_evidence: evidence needs ModelConfig::assignment"
                    .to_string(),
            )
        })?;
        let j = acfg.hypotheses as usize;
        let n = self.config.n_peaks as usize;
        // Single batched read: base (7 id + 2 float) + evidence, kept (2 id)
        // + log_prob (1 float). Still exactly one runtime read.
        let (ids, floats) = read_all(
            &[
                &bucket.actions,
                &bucket.formula.top,
                &bucket.formula.top_count,
                &bucket.formula.counters,
                &bucket.peaks.summary,
                &bucket.traj_alloc,
                &bucket.identity,
                &bucket.evidence,
                &bucket.peaks.kept,
            ],
            &[
                &bucket.formula.top_log_prob,
                &bucket.peaks.stats,
                lp_t,
            ],
        )?;
        let actions_h = &ids[0];
        let top_h = &ids[1];
        let top_count_h = &ids[2];
        let counters_h = &ids[3];
        let summary_h = &ids[4];
        let traj_h = &ids[5];
        let identity_h = &ids[6];
        let evidence_h = &ids[7];
        let kept_h = &ids[8];
        let top_lp_h = &floats[0];
        let stats_h = &floats[1];
        let log_prob_h = &floats[2];
        let record_width = ms2::sample_record_width(steps, atoms);
        let source_id = match config.formula_source {
            FormulaSource::Table => 0u8,
            FormulaSource::Enumerate => 1u8,
        };
        let use_graph = config.identity == IdentityMode::Graph;
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
            formula_counts: vec![0; rows * 10],
            formula_source: vec![source_id; spectra_n],
            formula_rank: vec![NO_FORMULA; rows],
            evidence_count: vec![0; rows],
            evidence_peak_id: vec![0; rows * EVIDENCE_CAP],
            evidence_hypothesis: vec![0; rows * EVIDENCE_CAP],
            evidence_shift: vec![0; rows * EVIDENCE_CAP],
            evidence_residual: vec![0; rows * EVIDENCE_CAP],
            evidence_log_prob: vec![0.0; rows * EVIDENCE_CAP],
        };
        // Original peak ids for the evidence mapping: the uploaded batch
        // (rotated under ShuffledSpectrum, own peaks otherwise).
        let owned;
        let map_batch = if config.control == Control::ShuffledSpectrum {
            owned = rotate_peaks(batch);
            &owned
        } else {
            batch
        };
        let n_raw = map_batch.n_raw as usize;
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
            for jj in 0..atoms {
                out.open_valence[r * atoms + jj] = actions_h[abase + steps * 4 + jj] as u8;
            }
            let tbase = r * 12;
            let slot = traj_h[tbase];
            if slot == u32::MAX || (slot as usize) >= formulas {
                out.formula_log_prob[r] = 0.0;
                out.formula_rank[r] = NO_FORMULA;
            } else {
                let fslot = slot as usize;
                out.formula_log_prob[r] = top_lp_h[b * formulas + fslot];
                out.formula_rank[r] = top_h[(b * formulas + fslot) * 2 + 1];
                for e in 0..10 {
                    out.formula_counts[r * 10 + e] = traj_h[tbase + 2 + e] as u16;
                }
            }
            if use_graph {
                out.status[r] |= identity_h[r * 2];
                out.identity_resolution[r] = identity_h[r * 2 + 1] as u8;
            }
            // Evidence from the scored rows: status/count plus the ranked
            // records mapped to original peak ids with log-probs.
            let ebase = r * 18;
            let ev_status = evidence_h[ebase];
            let ev_total = evidence_h[ebase + 1];
            let kept_n = (ev_total as usize).min(EVIDENCE_CAP);
            out.evidence_status[r] = (ev_status & 0xFF) as u8;
            out.evidence_count[r] = kept_n as u8;
            if slot == u32::MAX || (slot as usize) >= formulas {
                // No formula: evidence must be empty (kernel writes zero).
                out.evidence_status[r] = 0;
                out.evidence_count[r] = 0;
            } else {
                let fslot = slot as usize;
                for q in 0..kept_n {
                    let w = ebase + 2 + q * 4;
                    let p_pos = evidence_h[w] as usize;
                    let hyp = evidence_h[w + 1];
                    let shift_ob = evidence_h[w + 2];
                    let resid_ob = evidence_h[w + 3];
                    // Original peak id: kept position -> raw -> peak_id.
                    let mut pid = 0u32;
                    if p_pos < n {
                        let raw = kept_h[(b * n + p_pos) * 3];
                        if raw != u32::MAX && (raw as usize) < n_raw {
                            pid = map_batch.peak_id[b * n_raw + raw as usize];
                            if pid == u32::MAX {
                                pid = 0;
                            }
                        }
                    }
                    let shift = (shift_ob.wrapping_sub(0x8000_0000)) as i32;
                    let resid = (resid_ob.wrapping_sub(0x8000_0000)) as i32;
                    let lp_idx = ((b * formulas + fslot) * n + p_pos) * (j + 1) + (hyp as usize);
                    let mut lp = 0.0f32;
                    if lp_idx < log_prob_h.len() {
                        lp = log_prob_h[lp_idx];
                    }
                    out.evidence_peak_id[r * EVIDENCE_CAP + q] = pid;
                    out.evidence_hypothesis[r * EVIDENCE_CAP + q] = hyp as u8;
                    out.evidence_shift[r * EVIDENCE_CAP + q] = shift as i8;
                    out.evidence_residual[r * EVIDENCE_CAP + q] = resid;
                    out.evidence_log_prob[r * EVIDENCE_CAP + q] = lp;
                }
            }
        }
        let per_spectrum = self.per_spectrum_output(
            host_status,
            counters_h,
            summary_h,
            top_count_h,
            top_lp_h,
            stats_h,
            Some(batch),
            config,
            spectra_n,
            formulas,
        )?;
        for b in 0..spectra_n {
            out.request_status[b] = per_spectrum.request_status[b];
            out.rows_visited[b] = per_spectrum.rows_visited[b];
            out.rows_joined[b] = per_spectrum.rows_joined[b];
            out.rows_scored[b] = per_spectrum.rows_scored[b];
            out.formula_support_complete[b] = per_spectrum.formula_support_complete[b];
            out.formula_mass_retained[b] = per_spectrum.formula_mass_retained[b];
            out.peaks_kept[b] = per_spectrum.peaks_kept[b];
            out.intensity_retained[b] = per_spectrum.intensity_retained[b];
            let rs = out.request_status[b];
            if rs & request_status::FATAL_MASK != 0 || out.rows_scored[b] == 0 {
                for kk in 0..trajectories {
                    let r = b * trajectories + kk;
                    out.length[r] = 0;
                    out.formula_row[r] = NO_FORMULA;
                    out.formula_log_prob[r] = 0.0;
                    out.trace_log_prob[r] = 0.0;
                    out.status[r] = candidate_status::REQUEST_FAILED;
                    out.identity_resolution[r] = 0;
                    out.evidence_status[r] = 0;
                    out.evidence_count[r] = 0;
                    for v in &mut out.actions[r * steps * 4..(r + 1) * steps * 4] {
                        *v = 0;
                    }
                    for v in &mut out.open_valence[r * atoms..(r + 1) * atoms] {
                        *v = 0;
                    }
                    for v in &mut out.formula_counts[r * 10..(r + 1) * 10] {
                        *v = 0;
                    }
                    out.formula_rank[r] = NO_FORMULA;
                    for v in &mut out.evidence_peak_id[r * EVIDENCE_CAP..(r + 1) * EVIDENCE_CAP] {
                        *v = 0;
                    }
                    for v in &mut out.evidence_hypothesis[r * EVIDENCE_CAP..(r + 1) * EVIDENCE_CAP] {
                        *v = 0;
                    }
                    for v in &mut out.evidence_shift[r * EVIDENCE_CAP..(r + 1) * EVIDENCE_CAP] {
                        *v = 0;
                    }
                    for v in &mut out.evidence_residual[r * EVIDENCE_CAP..(r + 1) * EVIDENCE_CAP] {
                        *v = 0;
                    }
                    for v in &mut out.evidence_log_prob[r * EVIDENCE_CAP..(r + 1) * EVIDENCE_CAP] {
                        *v = 0.0;
                    }
                }
            }
        }
        out.validate()?;
        Ok(out)
    }

    /// The workspace bucket for these shapes, allocating on a miss: the piece
    /// of orchestration every workspace-level stage shares. A lookup is
    /// host-side only (no launch, read or allocation on a hit), so per-stage
    /// counter windows are unaffected.
    fn bucket_for<'w>(
        &self,
        workspace: &'w mut GenerationWorkspace<R, E>,
        pre: &GeneratePreflight,
        device: &Device<R>,
    ) -> Result<&'w mut GenBucket<R, E>> {
        workspace.bucket(
            pre.spectra_n,
            pre.trajectories,
            pre.steps,
            pre.n_raw as usize,
            pre.formulas,
            pre.window_m,
            pre.enum_p,
            pre.returned,
            pre.evidence,
            pre.formula_features,
            pre.atoms,
            pre.closures as usize,
            self.config.d_model as usize,
            self.config.n_peaks as usize,
            device,
        )
    }

    /// [`Ms2Model::generate_encode`] over the session's warmed bucket: the
    /// encoder stage as the device-mode harness runs it, one span at a time.
    pub fn generate_encode_ws(
        &self,
        workspace: &mut GenerationWorkspace<R, E>,
        spectra: &DeviceSpectra<R, E>,
        control: Control,
        pre: &GeneratePreflight,
        device: &Device<R>,
    ) -> Result<EncoderOutput<R, E>> {
        let _no_grad = crate::autograd::no_grad();
        let bucket = self.bucket_for(workspace, pre, device)?;
        self.generate_encode(spectra, &bucket.peaks, control)
    }

    /// [`Ms2Model::generate_search`] over the session's warmed bucket: the
    /// search stage as the device-mode harness runs it.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_search_ws(
        &self,
        workspace: &mut GenerationWorkspace<R, E>,
        spectra: &DeviceSpectra<R, E>,
        host_batch: &SpectrumBatch,
        pool: &Var<R, E>,
        table: &DeviceFormulaTable<R, E>,
        batch_len: usize,
        trajectories: usize,
        formulas: usize,
        metadata_blind: bool,
        rows_visited_max: u32,
        rows_scored_max: u32,
        config: &GenerationConfig,
        pre: &GeneratePreflight,
        device: &Device<R>,
    ) -> Result<()> {
        let _no_grad = crate::autograd::no_grad();
        let bucket = self.bucket_for(workspace, pre, device)?;
        self.generate_search(
            spectra,
            pool,
            table,
            &mut bucket.formula,
            &bucket.peaks,
            &mut bucket.traj_alloc,
            &bucket.lane_stats,
            &bucket.offsets,
            host_batch,
            &mut bucket.traj_meta,
            &mut bucket.state,
            &mut bucket.actions,
            &mut bucket.traj_formula,
            batch_len,
            pre.window_m,
            formulas,
            trajectories,
            pre.steps,
            pre.atoms,
            metadata_blind,
            rows_visited_max,
            rows_scored_max,
            config,
        )
    }

    /// [`Ms2Model::generate_decoder_init`] plus the conditioning setup over
    /// the session's warmed bucket: the decoder-init stage as the device-mode
    /// harness runs it.
    pub fn generate_decoder_init_ws(
        &self,
        workspace: &mut GenerationWorkspace<R, E>,
        encoded: &EncoderOutput<R, E>,
        pre: &GeneratePreflight,
        device: &Device<R>,
    ) -> Result<(DecoderState<R, E>, Tensor<R, E>, Var<R, E>)> {
        let _no_grad = crate::autograd::no_grad();
        // The carry trace is the one reader of the recurrent state.
        let (state, bonds) = self.decoder_init(
            encoded,
            pre.rows,
            device,
            workspace.composed_step,
            workspace.capture_carry_trace,
        )?;
        let bucket = self.bucket_for(workspace, pre, device)?;
        let traj = Var::constant(bucket.traj_formula.clone());
        let last = bucket.traj_formula.clone();
        workspace.last_traj_formula = Some(last);
        Ok((state, bonds, traj))
    }

    /// One [`Ms2Model::generate_decode_step`] over the session's warmed
    /// bucket: a single decode step as the device-mode harness runs it. The
    /// carry snapshot (test support; performs device reads) is returned for
    /// the caller to store rather than stored here, keeping the bucket borrow
    /// short.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_decode_step_ws(
        &self,
        workspace: &mut GenerationWorkspace<R, E>,
        encoded: &EncoderOutput<R, E>,
        traj_formula: &Var<R, E>,
        decoder_state: &mut DecoderState<R, E>,
        bond_table: &Tensor<R, E>,
        atom_table: &IdTensor<R>,
        step: usize,
        seed_lo: u32,
        seed_hi: u32,
        temperature: f32,
        trajectories: usize,
        pre: &GeneratePreflight,
        device: &Device<R>,
    ) -> Result<Option<StepCarries>> {
        let _no_grad = crate::autograd::no_grad();
        let capture = workspace.capture_carry_trace;
        {
            let bucket = self.bucket_for(workspace, pre, device)?;
            self.generate_decode_step(
                encoded,
                traj_formula,
                &mut bucket.actions,
                &mut bucket.step_token,
                &mut bucket.state,
                &mut bucket.logits,
                &bucket.traj_meta,
                decoder_state,
                bond_table,
                atom_table,
                step,
                seed_lo,
                seed_hi,
                temperature,
                pre.steps,
                pre.atoms,
                pre.closures,
                trajectories,
                pre.rows,
            )?;
        }
        if capture {
            Ok(Some(Self::snapshot_caches(decoder_state, step)?))
        } else {
            Ok(None)
        }
    }

    /// [`Ms2Model::generate_validate`] over the session's warmed bucket: the
    /// validate stage as the device-mode harness runs it.
    pub fn generate_validate_ws(
        &self,
        workspace: &mut GenerationWorkspace<R, E>,
        atom_table: &IdTensor<R>,
        trajectories: usize,
        config: &GenerationConfig,
        pre: &GeneratePreflight,
        device: &Device<R>,
    ) -> Result<()> {
        let _no_grad = crate::autograd::no_grad();
        let bucket = self.bucket_for(workspace, pre, device)?;
        self.generate_validate(
            &mut bucket.actions,
            &bucket.traj_meta,
            &mut bucket.scratch,
            atom_table,
            &mut bucket.graph_hash,
            &mut bucket.graph_scratch,
            &mut bucket.identity,
            &mut bucket.identity_scratch,
            pre.spectra_n,
            trajectories,
            pre.steps,
            pre.atoms,
            pre.closures,
            config.identity,
            config.identity_work_max,
        )
    }

    /// [`Ms2Model::generate_readout`] over the session's warmed bucket: the
    /// table/profiler path without a host batch. Only the modes it can serve
    /// are accepted: `identity = Graph` and the enumerating source are
    /// refused with [`Error::Config`] (instead of silently dropping identity
    /// bits or skipping the reconciliation) — call
    /// [`Ms2Model::generate_readout_ws_batch`] for the complete readout.
    pub fn generate_readout_ws(
        &self,
        workspace: &mut GenerationWorkspace<R, E>,
        host_status: &[u32],
        spectrum_ids: &[u64],
        config: &GenerationConfig,
        trajectories: usize,
        formulas: usize,
        pre: &GeneratePreflight,
        device: &Device<R>,
    ) -> Result<CandidateBatch> {
        let _no_grad = crate::autograd::no_grad();
        if config.identity == IdentityMode::Graph {
            return Err(Error::config(
                "Ms2Model::generate_readout_ws: identity Graph needs the batch-aware readout (generate_readout_ws_batch); this profiler path would silently drop the identity bits".to_string(),
            ));
        }
        if config.formula_source == FormulaSource::Enumerate {
            return Err(Error::config(
                "Ms2Model::generate_readout_ws: formula_source Enumerate needs the batch-aware readout (generate_readout_ws_batch); this profiler path would skip the enumeration reconciliation".to_string(),
            ));
        }
        if config.evidence {
            return Err(Error::config(
                "Ms2Model::generate_readout_ws: evidence=true needs the complete evidence readout; this profiler path would silently emit zero evidence".to_string(),
            ));
        }
        let bucket = self.bucket_for(workspace, pre, device)?;
        self.generate_readout(
            &bucket.actions,
            &bucket.formula,
            &bucket.peaks.summary,
            &bucket.traj_alloc,
            &bucket.identity,
            false,
            &bucket.formula.top_log_prob,
            &bucket.peaks.stats,
            host_status,
            spectrum_ids,
            None,
            config,
            pre.spectra_n,
            trajectories,
            formulas,
            pre.steps,
            pre.atoms,
            pre.closures as usize,
        )
    }

    /// [`Ms2Model::generate_readout`] over the session's warmed bucket with the
    /// host batch for enumeration reconciliation (D4): the path `generate`
    /// uses, so every readout shares the single reconciliation in
    /// `generate_readout`.
    pub fn generate_readout_ws_batch(
        &self,
        workspace: &mut GenerationWorkspace<R, E>,
        host_status: &[u32],
        spectrum_ids: &[u64],
        batch: &SpectrumBatch,
        config: &GenerationConfig,
        trajectories: usize,
        formulas: usize,
        pre: &GeneratePreflight,
        device: &Device<R>,
    ) -> Result<CandidateBatch> {
        let _no_grad = crate::autograd::no_grad();
        let use_graph = config.identity == IdentityMode::Graph;
        if config.evidence {
            return Err(Error::config(
                "Ms2Model::generate_readout_ws_batch: evidence=true needs the complete evidence readout; this profiler path would silently emit zero evidence".to_string(),
            ));
        }
        let bucket = self.bucket_for(workspace, pre, device)?;
        self.generate_readout(
            &bucket.actions,
            &bucket.formula,
            &bucket.peaks.summary,
            &bucket.traj_alloc,
            &bucket.identity,
            use_graph,
            &bucket.formula.top_log_prob,
            &bucket.peaks.stats,
            host_status,
            spectrum_ids,
            Some(batch),
            config,
            pre.spectra_n,
            trajectories,
            formulas,
            pre.steps,
            pre.atoms,
            pre.closures as usize,
        )
    }

    /// Run the full generation pipeline of architecture §5 and return the
    /// validated [`CandidateBatch`].
    ///
    /// This is [`Ms2Model::generate_with_hook`] with no hook: one code path,
    /// so hook-instrumented profiling observes exactly what this runs.
    pub fn generate(
        &self,
        batch: &SpectrumBatch,
        table: &DeviceFormulaTable<R, E>,
        config: &GenerationConfig,
        workspace: &mut GenerationWorkspace<R, E>,
        constants: &Ms2Constants<R>,
    ) -> Result<CandidateBatch> {
        self.generate_with_hook(batch, table, config, workspace, constants, None)
    }

    /// Run the full generation pipeline with a hook at the stage boundaries.
    ///
    /// Runs exactly what [`Ms2Model::generate`] runs, calling `hook` after
    /// each [`GenerateStage`]. With `None` (what `generate` passes) the only
    /// cost is a `None` comparison per boundary: no launch, no read and no
    /// allocation. A profiling driver installs a hook that snapshots counters
    /// and synchronised wall time at each boundary, which isolates decoder
    /// initialisation and the decode loop directly: no `total − prefix`
    /// subtraction, and no combined tail reported as decoder time.
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
    pub fn generate_with_hook(
        &self,
        batch: &SpectrumBatch,
        table: &DeviceFormulaTable<R, E>,
        config: &GenerationConfig,
        workspace: &mut GenerationWorkspace<R, E>,
        constants: &Ms2Constants<R>,
        mut hook: Option<&mut dyn FnMut(GenerateStage)>,
    ) -> Result<CandidateBatch> {
        // No gradient tape across the loop: every step would otherwise extend
        // the graph and leak memory over the trajectory. The stage functions
        // below each hold the same guard, so the device-mode harness (which
        // calls them one span at a time) runs the identical workload.
        let _no_grad = crate::autograd::no_grad();
        let pre = self.generate_preflight(batch, table, config)?;
        let device = table.table.device().clone();
        let spectra = self.generate_preprocess(batch, config, &device)?;
        if let Some(hook) = hook.as_mut() {
            hook(GenerateStage::AfterPreprocess);
        }
        let host_status = spectra.host_status.clone();
        let spectrum_ids = spectra.spectrum_id.clone();
        workspace.carry_trace.clear();
        // V1 §1.2: the workspace bucket is keyed by M as well; the request's
        // `formula_window` is the scored-candidate capacity of this call.
        // Every stage below runs over that warmed bucket through the same
        // workspace-level functions the device-mode harness profiles.
        let encoded = self.generate_encode_ws(workspace, &spectra, config.control, &pre, &device)?;
        if let Some(hook) = hook.as_mut() {
            hook(GenerateStage::AfterEncoder);
        }
        crate::backend::check_launches(&device)?;
        let metadata_blind = matches!(
            config.control,
            Control::MetadataOnly | Control::StructurePrior
        );
        self.generate_search_ws(
            workspace,
            &spectra,
            batch,
            &encoded.pool,
            table,
            pre.spectra_n,
            pre.trajectories,
            pre.formulas,
            metadata_blind,
            config.formula_rows_visited_max,
            config.formula_rows_scored_max,
            config,
            &pre,
            &device,
        )?;
        if let Some(hook) = hook.as_mut() {
            hook(GenerateStage::AfterSearch);
        }
        // Fragment-ion assignment for evidence (architecture §2): after top-F,
        // `ion_assign` for the retained formulas plus the head's log-probs.
        // Zero launches when `evidence` is false.
        let mut ion_opt: Option<IdTensor<R>> = None;
        let mut ion_meta_opt: Option<IdTensor<R>> = None;
        let mut ion_log_prob_opt: Option<Tensor<R, E>> = None;
        if config.evidence {
            let (ion_t, ion_meta_t, lp_var) = {
                let bucket = self.bucket_for(workspace, &pre, &device)?;
                self.generate_ion(
                    &spectra,
                    &encoded,
                    table,
                    &bucket.formula.top_counts,
                    &bucket.peaks.kept,
                    pre.spectra_n,
                    pre.formulas,
                    config,
                    &device,
                )?
            };
            if let (Some(it), Some(imt), Some(lpv)) = (ion_t, ion_meta_t, lp_var) {
                let lp_t = lpv.tensor().clone();
                ion_opt = Some(it);
                ion_meta_opt = Some(imt);
                ion_log_prob_opt = Some(lp_t);
            }
        }
        // The workspace borrow ends here so the sampling loop can hold the
        // bucket while pushing carry snapshots; re-borrow per step below.
        let (mut decoder_state, bond_table, traj_formula) =
            self.generate_decoder_init_ws(workspace, &encoded, &pre, &device)?;
        let seed_lo = (config.seed & 0xFFFF_FFFF) as u32;
        let seed_hi = (config.seed >> 32) as u32;
        if let Some(hook) = hook.as_mut() {
            hook(GenerateStage::AfterDecoderInit);
        }
        // Step 0 is not sampled: initialisation wrote START at position 0.
        // Each step reads the last emitted token, scores it, packs the heads,
        // samples the next token, and freezes the carries of rows that had
        // already finished (their `h`, `last_u` and `angle` stay exactly as
        // they were).
        for step in 1..pre.steps {
            let carry = self.generate_decode_step_ws(
                workspace,
                &encoded,
                &traj_formula,
                &mut decoder_state,
                &bond_table,
                &constants.atom_table,
                step,
                seed_lo,
                seed_hi,
                config.temperature,
                pre.trajectories,
                &pre,
                &device,
            )?;
            if let Some(carry) = carry {
                workspace.carry_trace.push(carry);
            }
            if let Some(hook) = hook.as_mut() {
                hook(GenerateStage::AfterDecodeStep(step));
            }
        }
        self.generate_validate_ws(
            workspace,
            &constants.atom_table,
            pre.trajectories,
            config,
            &pre,
            &device,
        )?;
        if let Some(hook) = hook.as_mut() {
            hook(GenerateStage::AfterValidate);
        }
        // Evidence for candidates (architecture §2.4): after validation,
        // scored evidence rows plus `evidence_f`. Zero launches when OFF.
        let mut traj_slot_opt: Option<IdTensor<R>> = None;
        let mut evidence_f_opt: Option<Tensor<R, E>> = None;
        if config.evidence {
            let (ts_t, ef_t) = {
                let bucket = self.bucket_for(workspace, &pre, &device)?;
                let rows = pre.spectra_n * pre.trajectories;
                let mut ts = IdTensor::empty(vec![rows, 2], &device);
                let mut ef = Tensor::empty(vec![rows, 2], &device);
                let (Some(ion_t), Some(ion_meta_t), Some(lp_t)) =
                    (ion_opt.as_ref(), ion_meta_opt.as_ref(), ion_log_prob_opt.as_ref())
                else {
                    return Err(Error::config(
                        "Ms2Model::generate_with_hook: evidence needs ion buffers (ion stage)".to_string(),
                    ));
                };
                self.generate_evidence(
                    &bucket.actions,
                    &bucket.traj_alloc,
                    &spectra.meta,
                    &bucket.peaks.kept,
                    ion_t,
                    ion_meta_t,
                    lp_t,
                    &mut bucket.evidence,
                    &mut ef,
                    &mut ts,
                    pre.spectra_n,
                    pre.trajectories,
                    pre.steps,
                    pre.atoms,
                    config,
                )?;
                (ts, ef)
            };
            traj_slot_opt = Some(ts_t);
            evidence_f_opt = Some(ef_t);
        }
        // The single batched read of the whole call (V1 §1.2: `top_counts`
        // joins the same batched read, so a warmed `generate` still performs
        // exactly one runtime read). D4: enumeration reconciliation already ran
        // inside `generate_readout` before validation, so every readout path
        // (generate and the workspace stage readout) shares it.
        let out = if config.evidence {
            self.generate_readout_evidence(
                workspace,
                &host_status,
                &spectrum_ids,
                batch,
                config,
                ion_log_prob_opt.as_ref(),
                &pre,
                &device,
            )?
        } else {
            self.generate_readout_ws_batch(
                workspace,
                &host_status,
                &spectrum_ids,
                batch,
                config,
                pre.trajectories,
                pre.formulas,
                &pre,
                &device,
            )?
        };
        // Keep evidence temporaries alive through the readout sync (the packed
        // read above synchronizes; dropping before would free device memory
        // while kernels are pending).
        let _keep = (ion_opt, ion_meta_opt, ion_log_prob_opt, traj_slot_opt, evidence_f_opt);
        if let Some(hook) = hook.as_mut() {
            hook(GenerateStage::AfterReadout);
        }
        Ok(out)
    }

    /// Fill `scores`, translate the allocation slots to window slots, rank,
    /// gather records and compact: the pack stage of the packed and resident
    /// paths (V1 §4.4).
    ///
    /// Six launches per call, constant per mode: `scores_fill`,
    /// `allocate_window`, `rank`, `record_pack` (+ `record_pack_f`), `pack` —
    /// plus one `pack_evidence` launch when `evidence_pack` is `Some` (I3b:
    /// the packed evidence rows and log-probabilities). With `None` no
    /// evidence kernel runs and the creation-zero `packed_ev` buffers stay
    /// untouched, so `evidence = false` launches nothing extra. Runs under
    /// `ms2.finalize`, so the profile driver measures allocation (search),
    /// identity and pack (finalize) stages with the launch budget still
    /// reconciling exactly.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_pack_ws(
        &self,
        workspace: &mut GenerationWorkspace<R, E>,
        config: &GenerationConfig,
        pre: &GeneratePreflight,
        device: &Device<R>,
        evidence_pack: Option<(&IdTensor<R>, &Tensor<R, E>)>,
    ) -> Result<()> {
        let _no_grad = crate::autograd::no_grad();
        let _tally = crate::backend::tally_scope("ms2.finalize");
        let bucket = self.bucket_for(workspace, pre, device)?;
        Self::pack_into(
            &bucket.actions,
            &bucket.traj_alloc,
            &bucket.formula.top,
            &bucket.formula.top_log_prob,
            &bucket.identity,
            &bucket.evidence,
            &bucket.rerank,
            &mut bucket.traj_window,
            &mut bucket.scores,
            &mut bucket.rank,
            &mut bucket.record,
            &mut bucket.record_f,
            &mut bucket.packed,
            &mut bucket.packed_f,
            &mut bucket.returned_count,
            config,
            pre,
        )?;
        if let Some((traj_slot, log_prob)) = evidence_pack {
            let Some(acfg) = self.config.assignment.as_ref() else {
                return Err(Error::config(
                    "Ms2Model::generate_pack_ws: evidence needs ModelConfig::assignment".to_string(),
                ));
            };
            let j = acfg.hypotheses as usize;
            let n = self.config.n_peaks as usize;
            Self::pack_evidence_into(
                &bucket.rank,
                &bucket.evidence,
                traj_slot,
                log_prob,
                &mut bucket.packed_ev,
                &mut bucket.packed_ev_f,
                j,
                n,
                pre,
            )?;
        }
        Ok(())
    }

    /// Compact the per-trajectory evidence rows into the packed evidence
    /// buffers (I3b): one `pack_evidence` launch. The caller runs this only
    /// when `evidence` is on with the assignment head configured (`j` is its
    /// hypothesis capacity `J`, `n` the kept-peak count `N`).
    #[allow(clippy::too_many_arguments)]
    fn pack_evidence_into(
        rank: &IdTensor<R>,
        evidence: &IdTensor<R>,
        traj_slot: &IdTensor<R>,
        log_prob: &Tensor<R, E>,
        packed_ev: &mut IdTensor<R>,
        packed_ev_f: &mut Tensor<R, E>,
        j: usize,
        n: usize,
        pre: &GeneratePreflight,
    ) -> Result<()> {
        let _no_grad = crate::autograd::no_grad();
        ms2_pack::pack_evidence(
            rank,
            evidence,
            traj_slot,
            log_prob,
            packed_ev,
            packed_ev_f,
            pre.formulas,
            n,
            j,
            pre.trajectories,
            pre.returned,
        )?;
        Ok(())
    }

    /// The lane work of [`Ms2Model::generate_pack_ws`] over explicit buffers:
    /// shared by the bucket path and the resident path (which owns its
    /// buffers rather than borrowing the bucket's).
    #[allow(clippy::too_many_arguments)]
    fn pack_into(
        actions: &IdTensor<R>,
        traj_alloc: &IdTensor<R>,
        top: &IdTensor<R>,
        top_log_prob: &Tensor<R, E>,
        identity: &IdTensor<R>,
        evidence: &IdTensor<R>,
        rerank: &Tensor<R, E>,
        traj_window: &mut IdTensor<R>,
        scores: &mut Tensor<R, f32>,
        rank: &mut IdTensor<R>,
        record: &mut IdTensor<R>,
        record_f: &mut Tensor<R, f32>,
        packed: &mut IdTensor<R>,
        packed_f: &mut Tensor<R, f32>,
        returned_count: &mut IdTensor<R>,
        config: &GenerationConfig,
        pre: &GeneratePreflight,
    ) -> Result<()> {
        let _no_grad = crate::autograd::no_grad();
        let use_graph = u32::from(config.identity == IdentityMode::Graph);
        let atoms_cap = pre.atoms as u32;
        // The retained slot each trajectory was assigned (V1 §3.2) maps to
        // its window slot through `top`: the rank the packed record carries.
        ms2_identity::allocate_window(traj_alloc, top, traj_window)?;
        // Trace log-probability from the trajectory record, formula
        // log-probability gathered by the trajectory's slot.
        ms2_pack::scores_fill(
            actions,
            traj_alloc,
            top_log_prob,
            scores,
            pre.steps,
            atoms_cap,
            pre.trajectories,
            pre.formulas,
        )?;
        ms2_pack::rank(
            actions,
            identity,
            scores,
            rerank,
            rank,
            pre.steps,
            atoms_cap,
            pre.trajectories,
            use_graph,
            0,
        )?;
        ms2_pack::record_pack(
            actions,
            traj_window,
            evidence,
            identity,
            record,
            pre.steps,
            atoms_cap,
            pre.trajectories,
            use_graph,
        )?;
        ms2_pack::record_pack_f(scores, rerank, record_f, 0)?;
        ms2_pack::pack(
            rank,
            record,
            record_f,
            packed,
            packed_f,
            returned_count,
            pre.steps,
            atoms_cap,
            pre.trajectories,
            pre.returned,
        )?;
        Ok(())
    }

    /// Assemble a [`PackedCandidateBatch`] from one batched read of the
    /// packed buffers plus the per-spectrum fields: the readout shared by
    /// `generate_packed` and [`ResidentCandidates::read`].
    ///
    /// `packed_ev_h`/`packed_ev_f_h` are the packed evidence rows and their
    /// assignment log-probabilities (kept-peak positions); `kept_h` is the
    /// `[B, N, 3]` kept peaks. Original peak ids are mapped on the host
    /// exactly as in `generate` (kept position → raw index → `peak_id`,
    /// with the same `ShuffledSpectrum` rotation). With `evidence = false`
    /// all three are the zero buffers and every evidence field stays zero.
    #[allow(clippy::too_many_arguments)]
    fn assemble_packed(
        &self,
        packed_h: &[u32],
        packed_f_h: &[f32],
        returned_count_h: &[u32],
        counters_h: &[u32],
        summary_h: &[u32],
        top_count_h: &[u32],
        top_lp_h: &[f32],
        stats_h: &[f32],
        packed_ev_h: &[u32],
        packed_ev_f_h: &[f32],
        kept_h: &[u32],
        host_status: &[u32],
        spectrum_ids: &[u64],
        batch_opt: Option<&SpectrumBatch>,
        config: &GenerationConfig,
        spectra_n: usize,
        trajectories: usize,
        formulas: usize,
        steps: usize,
        atoms: usize,
        closures: usize,
        returned: usize,
    ) -> Result<PackedCandidateBatch> {
        let per = self.per_spectrum_output(
            host_status,
            counters_h,
            summary_h,
            top_count_h,
            top_lp_h,
            stats_h,
            batch_opt,
            config,
            spectra_n,
            formulas,
        )?;
        let source_id = match config.formula_source {
            FormulaSource::Table => 0u8,
            FormulaSource::Enumerate => 1u8,
        };
        let mut stub =
            CandidateBatch::empty(spectrum_ids, trajectories, steps, atoms, closures);
        stub.request_status = per.request_status;
        stub.rows_visited = per.rows_visited;
        stub.rows_joined = per.rows_joined;
        stub.rows_scored = per.rows_scored;
        stub.formula_support_complete = per.formula_support_complete;
        stub.formula_mass_retained = per.formula_mass_retained;
        stub.peaks_kept = per.peaks_kept;
        stub.intensity_retained = per.intensity_retained;
        stub.formula_source = vec![source_id; spectra_n];
        let mut out = assemble(&stub, packed_h, packed_f_h, returned_count_h, returned)?;
        // Packed evidence details (I3b): kept positions become original peak
        // ids on the host, exactly as in `generate`. Unfilled slots stay
        // zero (as `assemble` left them). With `evidence = false` the OFF
        // mode returns all-zero evidence even if the buffers held stale
        // rows: the block below is skipped entirely.
        if config.evidence {
        {
            let ecap = super::contract::EVIDENCE_CAP;
            let n = self.config.n_peaks as usize;
            let slots = spectra_n * returned;
            if !packed_ev_h.iter().all(|&v| v == 0)
                || !packed_ev_f_h.iter().all(|&v| v == 0.0)
            {
                let map_batch_owned;
                let map_batch = match batch_opt {
                    Some(batch) => {
                        if config.control == Control::ShuffledSpectrum {
                            map_batch_owned = rotate_peaks(batch);
                            &map_batch_owned
                        } else {
                            batch
                        }
                    }
                    None => {
                        return Err(Error::config(
                            "Ms2Model::assemble_packed: non-zero packed evidence needs the host batch".to_string(),
                        ));
                    }
                };
                let n_raw = map_batch.n_raw as usize;
                for s in 0..slots {
                    if out.trajectory[s] == u32::MAX {
                        continue;
                    }
                    let b = s / returned;
                    let ebase = s * super::pack::EVIDENCE_STRIDE as usize;
                    let total = packed_ev_h[ebase + 1] as usize;
                    let kept_n = total.min(ecap);
                    out.evidence_count[s] = kept_n as u8;
                    for q in 0..kept_n {
                        let w = ebase + 2 + q * 4;
                        let p_pos = packed_ev_h[w] as usize;
                        let hyp = packed_ev_h[w + 1];
                        let shift_ob = packed_ev_h[w + 2];
                        let resid_ob = packed_ev_h[w + 3];
                        let mut pid = 0u32;
                        if p_pos < n {
                            let raw = kept_h[(b * n + p_pos) * 3];
                            if raw != u32::MAX && (raw as usize) < n_raw {
                                pid = map_batch.peak_id[b * n_raw + raw as usize];
                                if pid == u32::MAX {
                                    pid = 0;
                                }
                            }
                        }
                        let shift = (shift_ob.wrapping_sub(0x8000_0000)) as i32;
                        let resid = (resid_ob.wrapping_sub(0x8000_0000)) as i32;
                        out.evidence_peak_id[s * ecap + q] = pid;
                        out.evidence_hypothesis[s * ecap + q] = hyp as u8;
                        out.evidence_shift[s * ecap + q] = shift as i8;
                        out.evidence_residual[s * ecap + q] = resid;
                        out.evidence_log_prob[s * ecap + q] = packed_ev_f_h[s * ecap + q];
                    }
                }
            }
        }
        }
        out.validate()?;
        Ok(out)
    }

    /// Run the full generation pipeline and return the ranked, compacted
    /// [`PackedCandidateBatch`] (V1 §4.4): `B * R` records in rank order.
    ///
    /// This is [`Ms2Model::generate_packed_with_hook`] with no hook: one code
    /// path, so hook-instrumented profiling observes exactly what this runs.
    /// Performs exactly one device read (the packed buffers plus the
    /// per-spectrum fields). For the same request and seed the result equals
    /// `pack(&generate(..), identity bits, Raw, R)`.
    pub fn generate_packed(
        &self,
        batch: &SpectrumBatch,
        table: &DeviceFormulaTable<R, E>,
        config: &GenerationConfig,
        workspace: &mut GenerationWorkspace<R, E>,
        constants: &Ms2Constants<R>,
    ) -> Result<PackedCandidateBatch> {
        self.generate_packed_with_hook(batch, table, config, workspace, constants, None)
    }

    /// Run the full packed pipeline with a hook at the stage boundaries:
    /// the [`GenerateStage`] sequence of `generate`, plus [`AfterPack`]
    /// between validation and the readout.
    ///
    /// [`AfterPack`]: GenerateStage::AfterPack
    #[allow(clippy::too_many_lines)]
    pub fn generate_packed_with_hook(
        &self,
        batch: &SpectrumBatch,
        table: &DeviceFormulaTable<R, E>,
        config: &GenerationConfig,
        workspace: &mut GenerationWorkspace<R, E>,
        constants: &Ms2Constants<R>,
        mut hook: Option<&mut dyn FnMut(GenerateStage)>,
    ) -> Result<PackedCandidateBatch> {
        let _no_grad = crate::autograd::no_grad();
        let pre = self.generate_preflight(batch, table, config)?;
        let device = table.table.device().clone();
        let spectra = self.generate_preprocess(batch, config, &device)?;
        if let Some(hook) = hook.as_mut() {
            hook(GenerateStage::AfterPreprocess);
        }
        let host_status = spectra.host_status.clone();
        let spectrum_ids = spectra.spectrum_id.clone();
        workspace.carry_trace.clear();
        let encoded = self.generate_encode_ws(workspace, &spectra, config.control, &pre, &device)?;
        if let Some(hook) = hook.as_mut() {
            hook(GenerateStage::AfterEncoder);
        }
        crate::backend::check_launches(&device)?;
        let metadata_blind = matches!(
            config.control,
            Control::MetadataOnly | Control::StructurePrior
        );
        self.generate_search_ws(
            workspace,
            &spectra,
            batch,
            &encoded.pool,
            table,
            pre.spectra_n,
            pre.trajectories,
            pre.formulas,
            metadata_blind,
            config.formula_rows_visited_max,
            config.formula_rows_scored_max,
            config,
            &pre,
            &device,
        )?;
        if let Some(hook) = hook.as_mut() {
            hook(GenerateStage::AfterSearch);
        }
        // Fragment-ion assignment for evidence (architecture §2): after top-F,
        // `ion_assign` for the retained formulas plus the head's log-probs.
        // Zero launches when `evidence` is false.
        let mut ion_opt: Option<IdTensor<R>> = None;
        let mut ion_meta_opt: Option<IdTensor<R>> = None;
        let mut ion_log_prob_opt: Option<Tensor<R, E>> = None;
        if config.evidence {
            let (ion_t, ion_meta_t, lp_var) = {
                let bucket = self.bucket_for(workspace, &pre, &device)?;
                self.generate_ion(
                    &spectra,
                    &encoded,
                    table,
                    &bucket.formula.top_counts,
                    &bucket.peaks.kept,
                    pre.spectra_n,
                    pre.formulas,
                    config,
                    &device,
                )?
            };
            if let (Some(it), Some(imt), Some(lpv)) = (ion_t, ion_meta_t, lp_var) {
                let lp_t = lpv.tensor().clone();
                ion_opt = Some(it);
                ion_meta_opt = Some(imt);
                ion_log_prob_opt = Some(lp_t);
            }
        }
        let (mut decoder_state, bond_table, traj_formula) =
            self.generate_decoder_init_ws(workspace, &encoded, &pre, &device)?;
        let seed_lo = (config.seed & 0xFFFF_FFFF) as u32;
        let seed_hi = (config.seed >> 32) as u32;
        if let Some(hook) = hook.as_mut() {
            hook(GenerateStage::AfterDecoderInit);
        }
        for step in 1..pre.steps {
            let carry = self.generate_decode_step_ws(
                workspace,
                &encoded,
                &traj_formula,
                &mut decoder_state,
                &bond_table,
                &constants.atom_table,
                step,
                seed_lo,
                seed_hi,
                config.temperature,
                pre.trajectories,
                &pre,
                &device,
            )?;
            if let Some(carry) = carry {
                workspace.carry_trace.push(carry);
            }
            if let Some(hook) = hook.as_mut() {
                hook(GenerateStage::AfterDecodeStep(step));
            }
        }
        self.generate_validate_ws(
            workspace,
            &constants.atom_table,
            pre.trajectories,
            config,
            &pre,
            &device,
        )?;
        if let Some(hook) = hook.as_mut() {
            hook(GenerateStage::AfterValidate);
        }
        // Evidence rows plus the packed evidence compaction (architecture
        // §2.4): `ion_traj_slot` and the scored (best-E) selection into the
        // bucket's `evidence` rows, then the usual pack plus one
        // `pack_evidence` launch. Zero launches when `evidence` is false.
        let mut traj_slot_opt: Option<IdTensor<R>> = None;
        if config.evidence {
            let (Some(ion_t), Some(ion_meta_t), Some(lp_t)) =
                (ion_opt.as_ref(), ion_meta_opt.as_ref(), ion_log_prob_opt.as_ref())
            else {
                return Err(Error::config(
                    "Ms2Model::generate_packed_with_hook: evidence needs ion buffers (ion stage)".to_string(),
                ));
            };
            let mut ts = {
                let rows = pre.spectra_n * pre.trajectories;
                IdTensor::empty(vec![rows, 2], &device)
            };
            {
                let _tally = crate::backend::tally_scope("ms2.finalize");
                let bucket = self.bucket_for(workspace, &pre, &device)?;
                crate::tensor::ops::ms2_ion::ion_traj_slot(
                    &bucket.traj_alloc,
                    &spectra.meta,
                    &mut ts,
                    pre.trajectories as u32,
                )?;
                crate::tensor::ops::ms2_ion::ion_evidence_scored(
                    &bucket.actions,
                    &ts,
                    ion_t,
                    ion_meta_t,
                    lp_t,
                    &mut bucket.evidence,
                    pre.steps as u32,
                    pre.atoms as u32,
                    pre.trajectories as u32,
                )?;
            }
            self.generate_pack_ws(workspace, config, &pre, &device, Some((&ts, lp_t)))?;
            traj_slot_opt = Some(ts);
        } else {
            self.generate_pack_ws(workspace, config, &pre, &device, None)?;
        }
        if let Some(hook) = hook.as_mut() {
            hook(GenerateStage::AfterPack);
        }
        // The single batched read of the whole call: the packed buffers plus
        // the per-spectrum fields, and nothing else (in particular not the
        // uncompacted `B * K` records). With evidence the packed evidence
        // rows, their log-probabilities and the kept peaks join the same
        // batched read (still exactly one runtime read); the host maps kept
        // positions to original peak ids exactly as `generate` does.
        let bucket = self.bucket_for(workspace, &pre, &device)?;
        // One runtime read: the packed buffers (whose float records are
        // always f32) plus the per-spectrum fields in the neural dtype.
        let (ids, floats_e, floats_f32) = read_all_mixed(
            &[
                &bucket.packed,
                &bucket.returned_count,
                &bucket.formula.counters,
                &bucket.peaks.summary,
                &bucket.formula.top_count,
                &bucket.packed_ev,
                &bucket.peaks.kept,
            ],
            &[
                &bucket.formula.top_log_prob,
                &bucket.peaks.stats,
                &bucket.packed_ev_f,
            ],
            &[&bucket.packed_f],
        )?;
        let out = self.assemble_packed(
            &ids[0],
            &floats_f32[0],
            &ids[1],
            &ids[2],
            &ids[3],
            &ids[4],
            &floats_e[0],
            &floats_e[1],
            &ids[5],
            &floats_e[2],
            &ids[6],
            &host_status,
            &spectrum_ids,
            Some(batch),
            config,
            pre.spectra_n,
            pre.trajectories,
            pre.formulas,
            pre.steps,
            pre.atoms,
            pre.closures as usize,
            pre.returned,
        )?;
        // Keep evidence temporaries alive through the readout sync (dropping
        // before would free device memory while kernels are pending).
        let _keep = (ion_opt, ion_meta_opt, ion_log_prob_opt, traj_slot_opt);
        if let Some(hook) = hook.as_mut() {
            hook(GenerateStage::AfterReadout);
        }
        Ok(out)
    }

    /// Run the full packed pipeline with no device read and return the
    /// device-resident result (V1 §4.4).
    ///
    /// This is [`Ms2Model::generate_resident_with_hook`] with no hook.
    /// The workspace bucket is leased to the result: a later call on the
    /// same workspace allocates another bucket rather than overwriting the
    /// leased one. [`ResidentCandidates::read`] performs exactly one read;
    /// [`ResidentCandidates::release_into`] (or drop) returns the bucket.
    pub fn generate_resident(
        &self,
        batch: &SpectrumBatch,
        table: &DeviceFormulaTable<R, E>,
        config: &GenerationConfig,
        workspace: &mut GenerationWorkspace<R, E>,
        constants: &Ms2Constants<R>,
    ) -> Result<ResidentCandidates<R, E>> {
        self.generate_resident_with_hook(batch, table, config, workspace, constants, None)
    }

    /// Run the full resident pipeline with a hook at the stage boundaries:
    /// the [`GenerateStage`] sequence of `generate_packed_with_hook` without
    /// the final readout (no read is performed).
    #[allow(clippy::too_many_lines)]
    pub fn generate_resident_with_hook(
        &self,
        batch: &SpectrumBatch,
        table: &DeviceFormulaTable<R, E>,
        config: &GenerationConfig,
        workspace: &mut GenerationWorkspace<R, E>,
        constants: &Ms2Constants<R>,
        mut hook: Option<&mut dyn FnMut(GenerateStage)>,
    ) -> Result<ResidentCandidates<R, E>> {
        let _no_grad = crate::autograd::no_grad();
        let pre = self.generate_preflight(batch, table, config)?;
        let device = table.table.device().clone();
        let spectra = self.generate_preprocess(batch, config, &device)?;
        if let Some(hook) = hook.as_mut() {
            hook(GenerateStage::AfterPreprocess);
        }
        let host_status = spectra.host_status.clone();
        let spectrum_ids = spectra.spectrum_id.clone();
        workspace.carry_trace.clear();
        // The leased bucket: removed from the workspace cache, so a later
        // call on the same workspace allocates another bucket rather than
        // overwriting this result. Scratch stages below run directly over
        // its buffers (the same stage functions production uses).
        let mut bucket = workspace.lease_bucket(
            &pre,
            self.config.d_model as usize,
            self.config.n_peaks as usize,
            &device,
        )?;
        let encoded = self.generate_encode(&spectra, &bucket.peaks, config.control)?;
        if let Some(hook) = hook.as_mut() {
            hook(GenerateStage::AfterEncoder);
        }
        crate::backend::check_launches(&device)?;
        let metadata_blind = matches!(
            config.control,
            Control::MetadataOnly | Control::StructurePrior
        );
        self.generate_search(
            &spectra,
            &encoded.pool,
            table,
            &mut bucket.formula,
            &bucket.peaks,
            &mut bucket.traj_alloc,
            &bucket.lane_stats,
            &bucket.offsets,
            batch,
            &mut bucket.traj_meta,
            &mut bucket.state,
            &mut bucket.actions,
            &mut bucket.traj_formula,
            pre.spectra_n,
            pre.window_m,
            pre.formulas,
            pre.trajectories,
            pre.steps,
            pre.atoms,
            metadata_blind,
            config.formula_rows_visited_max,
            config.formula_rows_scored_max,
            config,
        )?;
        if let Some(hook) = hook.as_mut() {
            hook(GenerateStage::AfterSearch);
        }
        // Fragment-ion assignment for evidence (architecture §2): after top-F,
        // `ion_assign` for the retained formulas plus the head's log-probs.
        // The buffers live in this scope and are consumed by the evidence
        // stage below; the packed evidence rows outlive them in the leased
        // bucket. Zero launches when `evidence` is false.
        let n_peaks = self.config.n_peaks as usize;
        let mut ion_bufs: Option<(IdTensor<R>, IdTensor<R>, Tensor<R, E>)> = None;
        if config.evidence {
            let (ion_t, ion_meta_t, lp_var) = self.generate_ion(
                &spectra,
                &encoded,
                table,
                &bucket.formula.top_counts,
                &bucket.peaks.kept,
                pre.spectra_n,
                pre.formulas,
                config,
                &device,
            )?;
            if let (Some(it), Some(imt), Some(lpv)) = (ion_t, ion_meta_t, lp_var) {
                ion_bufs = Some((it, imt, lpv.tensor().clone()));
            } else {
                return Err(Error::config(
                    "Ms2Model::generate_resident_with_hook: evidence needs ion buffers (ion stage)".to_string(),
                ));
            }
        }
        // The state does not outlive the loop: nobody reads its carries.
        let (mut decoder_state, bond_table) =
            self.decoder_init(&encoded, pre.rows, &device, workspace.composed_step, false)?;
        let traj = Var::constant(bucket.traj_formula.clone());
        workspace.last_traj_formula = Some(bucket.traj_formula.clone());
        let seed_lo = (config.seed & 0xFFFF_FFFF) as u32;
        let seed_hi = (config.seed >> 32) as u32;
        if let Some(hook) = hook.as_mut() {
            hook(GenerateStage::AfterDecoderInit);
        }
        for step in 1..pre.steps {
            self.generate_decode_step(
                &encoded,
                &traj,
                &mut bucket.actions,
                &mut bucket.step_token,
                &mut bucket.state,
                &mut bucket.logits,
                &bucket.traj_meta,
                &mut decoder_state,
                &bond_table,
                &constants.atom_table,
                step,
                seed_lo,
                seed_hi,
                config.temperature,
                pre.steps,
                pre.atoms,
                pre.closures,
                pre.trajectories,
                pre.rows,
            )?;
            if let Some(hook) = hook.as_mut() {
                hook(GenerateStage::AfterDecodeStep(step));
            }
        }
        self.generate_validate(
            &mut bucket.actions,
            &bucket.traj_meta,
            &mut bucket.scratch,
            &constants.atom_table,
            &mut bucket.graph_hash,
            &mut bucket.graph_scratch,
            &mut bucket.identity,
            &mut bucket.identity_scratch,
            pre.spectra_n,
            pre.trajectories,
            pre.steps,
            pre.atoms,
            pre.closures,
            config.identity,
            config.identity_work_max,
        )?;
        if let Some(hook) = hook.as_mut() {
            hook(GenerateStage::AfterValidate);
        }
        {
            let _tally = crate::backend::tally_scope("ms2.finalize");
            // Evidence rows plus the packed compaction (architecture §2.4):
            // `ion_traj_slot` and the scored (best-E) selection into the
            // bucket's `evidence` rows, then the usual pack plus one
            // `pack_evidence` launch. Zero launches when `evidence` is false
            // (the creation-zero evidence buffers stay untouched).
            if config.evidence {
                let Some((ref ion_t, ref ion_meta_t, ref lp_t)) = ion_bufs else {
                    return Err(Error::config(
                        "Ms2Model::generate_resident_with_hook: evidence needs ion buffers (ion stage)".to_string(),
                    ));
                };
                let rows = pre.spectra_n * pre.trajectories;
                let mut ts = IdTensor::empty(vec![rows, 2], &device);
                crate::tensor::ops::ms2_ion::ion_traj_slot(
                    &bucket.traj_alloc,
                    &spectra.meta,
                    &mut ts,
                    pre.trajectories as u32,
                )?;
                crate::tensor::ops::ms2_ion::ion_evidence_scored(
                    &bucket.actions,
                    &ts,
                    ion_t,
                    ion_meta_t,
                    lp_t,
                    &mut bucket.evidence,
                    pre.steps as u32,
                    pre.atoms as u32,
                    pre.trajectories as u32,
                )?;
                // The usual pack first (record_pack reads the fresh evidence
                // status words and computes `rank`), then the evidence
                // compaction over those ranks. The leased bucket owns the
                // packed evidence buffers, but the slot/log-probability
                // inputs are scope-local: compact now, before they drop (no
                // read is performed here).
                Self::pack_into(
                    &bucket.actions,
                    &bucket.traj_alloc,
                    &bucket.formula.top,
                    &bucket.formula.top_log_prob,
                    &bucket.identity,
                    &bucket.evidence,
                    &bucket.rerank,
                    &mut bucket.traj_window,
                    &mut bucket.scores,
                    &mut bucket.rank,
                    &mut bucket.record,
                    &mut bucket.record_f,
                    &mut bucket.packed,
                    &mut bucket.packed_f,
                    &mut bucket.returned_count,
                    config,
                    &pre,
                )?;
                let Some(acfg) = self.config.assignment.as_ref() else {
                    return Err(Error::config(
                        "Ms2Model::generate_resident_with_hook: evidence needs ModelConfig::assignment".to_string(),
                    ));
                };
                let j = acfg.hypotheses as usize;
                Self::pack_evidence_into(
                    &bucket.rank,
                    &bucket.evidence,
                    &ts,
                    lp_t,
                    &mut bucket.packed_ev,
                    &mut bucket.packed_ev_f,
                    j,
                    n_peaks,
                    &pre,
                )?;
            } else {
                Self::pack_into(
                    &bucket.actions,
                    &bucket.traj_alloc,
                    &bucket.formula.top,
                    &bucket.formula.top_log_prob,
                    &bucket.identity,
                    &bucket.evidence,
                    &bucket.rerank,
                    &mut bucket.traj_window,
                    &mut bucket.scores,
                    &mut bucket.rank,
                    &mut bucket.record,
                    &mut bucket.record_f,
                    &mut bucket.packed,
                    &mut bucket.packed_f,
                    &mut bucket.returned_count,
                    config,
                    &pre,
                )?;
            }
        }
        if let Some(hook) = hook.as_mut() {
            hook(GenerateStage::AfterPack);
        }
        Ok(ResidentCandidates {
            bucket,
            host_status,
            spectrum_ids,
            host_batch: batch.clone(),
            config: config.clone(),
            spectra_n: pre.spectra_n,
            trajectories: pre.trajectories,
            formulas: pre.formulas,
            steps: pre.steps,
            atoms: pre.atoms,
            closures: pre.closures as usize,
            returned: pre.returned,
        })
    }
}

/// Device-resident packed candidates (V1 §4.4): the result of
/// [`Ms2Model::generate_resident`], owning its device buffers.
///
/// The workspace bucket is leased to this value: it was removed from the
/// workspace cache, so a later call on the same workspace allocates another
/// bucket rather than overwriting these buffers. [`read`] performs exactly
/// one device read and returns the [`PackedCandidateBatch`];
/// [`release_into`] returns the bucket to a workspace (and [`release`]
/// drops it explicitly; plain drop frees the buffers the same way).
///
/// [`read`]: ResidentCandidates::read
/// [`release_into`]: ResidentCandidates::release_into
/// [`release`]: ResidentCandidates::release
pub struct ResidentCandidates<R: Runtime, E: FloatElem> {
    /// The leased workspace bucket: packed buffers plus the per-spectrum
    /// fields the deferred read assembles.
    bucket: GenBucket<R, E>,
    /// Host request-validation bits per spectrum (no read needed).
    host_status: Vec<u32>,
    /// Provenance per spectrum (no read needed).
    spectrum_ids: Vec<u64>,
    /// Host batch for enumeration reconciliation at read time.
    host_batch: SpectrumBatch,
    /// Generation config (identity mode was already applied on device;
    /// control and formula source steer the host assembly).
    config: GenerationConfig,
    /// Spectra per batch.
    spectra_n: usize,
    /// Trajectories per spectrum.
    trajectories: usize,
    /// Formula hypotheses per spectrum.
    formulas: usize,
    /// Maximum trace steps.
    steps: usize,
    /// Maximum atoms per candidate.
    atoms: usize,
    /// Maximum ring closures per candidate.
    closures: usize,
    /// Packed slots per spectrum.
    returned: usize,
}

impl<R: Runtime, E: FloatElem> ResidentCandidates<R, E> {
    /// Perform exactly one device read — the packed buffers plus the
    /// per-spectrum fields — and return the [`PackedCandidateBatch`].
    /// With evidence the packed evidence rows, their log-probabilities and
    /// the kept peaks join the same batched read (still exactly one runtime
    /// read); the host maps kept positions to original peak ids exactly as
    /// `generate` does.
    pub fn read(&self, model: &Ms2Model<R, E>) -> Result<PackedCandidateBatch> {
        let _no_grad = crate::autograd::no_grad();
        let (ids, floats_e, floats_f32) = read_all_mixed(
            &[
                &self.bucket.packed,
                &self.bucket.returned_count,
                &self.bucket.formula.counters,
                &self.bucket.peaks.summary,
                &self.bucket.formula.top_count,
                &self.bucket.packed_ev,
                &self.bucket.peaks.kept,
            ],
            &[
                &self.bucket.formula.top_log_prob,
                &self.bucket.peaks.stats,
                &self.bucket.packed_ev_f,
            ],
            &[&self.bucket.packed_f],
        )?;
        model.assemble_packed(
            &ids[0],
            &floats_f32[0],
            &ids[1],
            &ids[2],
            &ids[3],
            &ids[4],
            &floats_e[0],
            &floats_e[1],
            &ids[5],
            &floats_e[2],
            &ids[6],
            &self.host_status,
            &self.spectrum_ids,
            Some(&self.host_batch),
            &self.config,
            self.spectra_n,
            self.trajectories,
            self.formulas,
            self.steps,
            self.atoms,
            self.closures,
            self.returned,
        )
    }

    /// Return the leased bucket to `workspace` (reusable by later calls)
    /// and drop the host side.
    pub fn release_into(self, workspace: &mut GenerationWorkspace<R, E>) {
        workspace.unlease_bucket(self.bucket);
    }

    /// Drop the result explicitly, freeing its device buffers to the
    /// allocator (plain drop does the same).
    pub fn release(self) {}
}
