//! MS2 device capabilities and memory estimates (architecture §6.1, §6.2).
//!
//! [`Ms2Capabilities::probe`] reports what the device can do and
//! [`Ms2Capabilities::check`] refuses a model it cannot run before any launch.
//! [`Ms2MemoryEstimate::generation`] and [`Ms2MemoryEstimate::training`] return
//! named byte counts for a configuration, computed with checked arithmetic so
//! an overflow is an error rather than a wrapped size. `GenerationConfig`'s
//! `max_device_bytes` is compared with the estimate through
//! [`Ms2MemoryEstimate::check_limit`] before allocation.

use cubecl::prelude::Runtime;

use crate::backend::{Device, FloatElem, memory_snapshot, reserved_bytes, supports_dtype};
use crate::error::{Error, Result};
use crate::ssm::SsmConfig;

use super::contract::ModelConfig;

/// At most 6 array bindings per MS2 kernel (architecture §1).
///
/// Related fields are packed into one buffer and addressed by offsets passed
/// as scalars, because wgpu's default limit is 8 storage buffers per shader
/// stage and CubeCL spends one on scalar and shape metadata. The capability
/// probe refuses a device with `max_bindings < 7` before any launch.
pub const MS2_MAX_KERNEL_ARRAYS: usize = 6;

/// What an MS2 device can do (architecture §6.1).
///
/// Probed from the runtime, then enforced by [`Ms2Capabilities::check`]
/// before any launch: a device that cannot hold the model dtype or bind the
/// kernels' arrays fails fast instead of failing mid-kernel.
///
/// The `timing` field records what `client.profile` actually returns on the
/// device, probed once with a trivial closure (see [`Ms2Capabilities::probe`]):
/// CubeCL's `ComputeClient::profile` brackets `func` with `start_profile` /
/// `end_profile` and hands back a `ProfileDuration` whose `timing_method` is
/// `Device` for hardware timestamps or `System` for host wall time
/// (`cubecl-common-0.10.0/src/profile.rs`, `ProfileDuration::timing_method`;
/// `cubecl-runtime-0.10.0/src/client.rs`, `ComputeClient::profile`;
/// `cubecl-runtime-0.10.0/src/timestamp_profiler.rs`, `TimestampProfiler::stop`
/// returns system time; `cubecl-wgpu-0.10.0/src/runtime.rs` picks `Device`
/// only with `TIMESTAMP_QUERY`, else `System`; `cubecl-cpu-0.10.0/src/runtime.rs`
/// declares `Device` but its stream (`compute/stream.rs`) timestamps through
/// `TimestampProfiler`, so the probe observes `SystemTime`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ms2Capabilities {
    /// Runtime name, e.g. `"cpu"` or `"wgpu"`.
    pub backend: String,
    /// Whether `f32` buffers and arithmetic are available.
    pub f32_supported: bool,
    /// Whether `f16` buffers and arithmetic are available.
    pub f16_supported: bool,
    /// Whether `bf16` buffers and arithmetic are available.
    pub bf16_supported: bool,
    /// Storage bindings per kernel the hardware allows.
    pub max_bindings: usize,
    /// Whether the runtime reports allocator state.
    pub reports_memory: bool,
    /// How `client.profile` times a stage on this device.
    pub timing: TimingMethod,
    /// Whether the runtime reports reserved bytes (`reserved_bytes` is `Some`).
    pub reports_reserved_bytes: bool,
    /// Hardware plane width (1 on the CPU runtime).
    pub plane_size_max: u32,
}

/// How `client.profile` times a stage (architecture §6.1, P2.4).
///
/// Probed from what the device actually returns, never inferred from the
/// backend name: `DeviceTimestamps` is CubeCL `TimingMethod::Device`
/// (hardware timestamps), `SystemTime` is `TimingMethod::System` (host wall
/// time around a sync), and `Unavailable` means the probe itself errored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimingMethod {
    /// Hardware device timestamps (`TimingMethod::Device`).
    DeviceTimestamps,
    /// Host wall time around a sync (`TimingMethod::System`).
    SystemTime,
    /// The `client.profile` probe errored, so no timing is available.
    Unavailable,
}

impl Ms2Capabilities {
    /// Report the capabilities of `device`.
    ///
    /// The timing probe calls `client.profile` once with a trivial closure
    /// and records the returned `ProfileDuration::timing_method`; an error
    /// maps to [`TimingMethod::Unavailable`] rather than failing the probe.
    /// The method string (`"device"` / `"system"`) is matched rather than
    /// the enum type so no extra crate dependency is needed.
    pub fn probe<R: Runtime>(device: &Device<R>) -> Self {
        let hardware = &device.client().properties().hardware;
        let timing = match device.client().profile(|| {}, "ms2-timing-probe") {
            Ok((_, duration)) => match duration.timing_method().to_string().as_str() {
                "device" => TimingMethod::DeviceTimestamps,
                "system" => TimingMethod::SystemTime,
                _ => TimingMethod::Unavailable,
            },
            Err(_) => TimingMethod::Unavailable,
        };
        Self {
            backend: device.name().to_string(),
            f32_supported: supports_dtype(device, crate::backend::DType::F32),
            f16_supported: supports_dtype(device, crate::backend::DType::F16),
            bf16_supported: supports_dtype(device, crate::backend::DType::BF16),
            max_bindings: hardware.max_bindings as usize,
            reports_memory: memory_snapshot(device).is_some(),
            timing,
            reports_reserved_bytes: reserved_bytes(device).is_some(),
            plane_size_max: hardware.plane_size_max,
        }
    }

    /// Refuse a model this device cannot run.
    ///
    /// The validated dtype set (contracts §3.3), independent of what the
    /// hardware reports: f32 on every backend; bf16 on the CPU backend only;
    /// f16 nowhere — f16 is not validated for the MS2 model (NaN loss from
    /// step 0 and a `generate` that fails its own validation on the CPU
    /// runtime; MS2 kernel compilation failure on wgpu, where the `3e38`
    /// finiteness literal is not representable in f16). A failure is
    /// [`Error::Unsupported`] naming the dtype, the backend and the reason.
    /// `max_bindings` must cover the [`MS2_MAX_KERNEL_ARRAYS`] array bindings
    /// plus the metadata binding.
    pub fn check(&self, model: &ModelConfig) -> Result<()> {
        Self::check_dtype(&self.backend, model.dtype)?;
        let required = MS2_MAX_KERNEL_ARRAYS + 1;
        if self.max_bindings < required {
            return Err(Error::Unsupported(format!(
                "Ms2Capabilities::check: max_bindings {} is below the required {required} \
                 (MS2_MAX_KERNEL_ARRAYS={MS2_MAX_KERNEL_ARRAYS} plus one metadata binding)",
                self.max_bindings,
            )));
        }
        Ok(())
    }

    /// Refuse a model dtype outside the validated set, without a probe.
    ///
    /// The validated allowlist (contracts §3.3) is independent of hardware
    /// capability: f32 on every backend; bf16 on the CPU backend only; f16
    /// nowhere. This one function is used by production, tests and
    /// `examples/ms2_dtype_report.rs` alike, so a combination refused here is
    /// refused everywhere. A refusal is [`Error::Unsupported`] naming the
    /// dtype and saying it is "not validated for this backend".
    /// Exact-mass decisions stay independent of the neural dtype
    /// (contracts §3.3): this gate only covers the neural computation.
    pub fn check_dtype(backend: &str, dtype: crate::backend::DType) -> Result<()> {
        use crate::backend::DType;
        match dtype {
            DType::F32 => Ok(()),
            DType::BF16 => {
                if backend == "cpu" {
                    Ok(())
                } else {
                    Err(Error::Unsupported(format!(
                        "Ms2Capabilities::check: backend {backend} refuses dtype bf16: bf16 is not validated \
                         for this backend (validated on the CPU backend only; use f32 on this backend)"
                    )))
                }
            }
            DType::F16 => Err(Error::Unsupported(format!(
                "Ms2Capabilities::check: backend {backend} refuses dtype f16: f16 is not validated \
                 for this backend (not validated on any backend: NaN on the CPU runtime, kernel compilation failure on wgpu)"
            ))),
        }
    }

    /// Require the actual neural element type to equal the configured dtype
    /// and to be in the validated set ([`Ms2Capabilities::check_dtype`]).
    ///
    /// A mismatch is [`Error::Config`] (not `Unsupported`): the caller wired
    /// the wrong element type to this config. Called with the live backend
    /// name by `Ms2Model::init`, the generation/training preflights and
    /// `Ms2Trainer::new`, before any allocation, upload or launch, so an
    /// unsupported combination never reaches a kernel. Exact-mass decisions
    /// stay independent of the neural dtype (contracts §3.3).
    pub fn check_neural_dtype<E: FloatElem>(backend: &str, dtype: crate::backend::DType) -> Result<()> {
        if E::DTYPE != dtype {
            return Err(Error::config(format!(
                "Ms2Capabilities::check: neural element type {} does not equal the configured dtype {} \
                 (E::DTYPE must equal config.dtype)",
                E::DTYPE.name(),
                dtype.name()
            )));
        }
        Self::check_dtype(backend, E::DTYPE)
    }

    /// Probe `device` for the backend name and refuse a model dtype outside
    /// the validated set ([`Ms2Capabilities::check_dtype`]).
    ///
    /// Reads only client properties (no launch, no read, no allocation), so
    /// it is safe on the hot path before dispatch.
    pub fn check_device<R: Runtime, E: FloatElem>(
        device: &Device<R>,
        model: &ModelConfig,
    ) -> Result<()> {
        Self::check_neural_dtype::<E>(&device.name(), model.dtype)
    }
}

/// Named byte counts for an MS2 configuration, in a fixed documented order.
///
/// Shared items first (`weights`, `formula_table`, `raw_peaks`,
/// `peak_selection`, `encoder_activations`, `spectrum_memory`), then the
/// generation-only or training-only items. `d = d_model`, `N = n_peaks`,
/// `Ld`/`Le` = decoder/encoder blocks, `A = max_atoms`, `T = max_steps`,
/// `W` = encoder.in_proj_width()`. Generation additionally carries
/// `cache_gather`: the per-decode-step bytes read for the K/V caches and
/// the atom memory (V0 sampling does no beam gather).
///
/// Training sizes the peak of a step, the end of the backward pass: the
/// forward values the tape keeps and one gradient per node of the tape. The
/// forward values are `encoder_activations` and `encoder_scan_retained`
/// (per encoder mixer and packed peak cell), `spectrum_memory`,
/// `decoder_embed_retained`, `decoder_activations` and
/// `decoder_mixer_retained` (per decoder layer and position),
/// `attention_scores` (the attention weights; the raw scores are not kept),
/// `decoder_attention_retained` (queries, context and per-head keys and
/// values), `atom_memory`, `decoder_head_positions_retained` and
/// `head_scratch` (the teacher pass's outputs); `activation_gradients` is
/// the retained gradient of every node, `gradients` the parameters' and
/// `optimizer_moments` the AdamW state. A value nothing captures (the
/// output of an add, a reshape or a permute once its consumer has run) and
/// a gradient that shares its consumer's buffer are counted as allocating
/// nothing; the teacher path never materialises the `[rows, 19, A]`
/// pointer-by-type table nor a `[positions, A, d]` key tensor. The shapes
/// are those of the padded teacher pass with every target slot occupied,
/// and of encoder scans over the kept peaks. See P2.2
/// (`tests/ms2_footprint.rs`) for the reconciliation.
///
/// V1 §1.3 (B1-fix): both estimates carry the complete formula workspace —
/// `window`, `counters`, `cand`, `cand_feat`, the head activations, the
/// scores, the `cand` gate mask (`formula_mask`), `formula_top`,
/// `formula_top_log_prob`, `formula_top_counts` and `top_count` — and the
/// generation `readout` packs `top_counts`. Training additionally carries
/// the gold path: `gold_counts`, the `count_features` output with its
/// gradient (`gold_feat`), the row-network activations with their gradients
/// (`gold_formula_head`: 3 `[B, d]` tensors, forward plus gradient) and
/// `gold_slot`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ms2MemoryEstimate {
    /// Named byte counts in the documented order.
    pub items: Vec<(&'static str, u64)>,
}

/// Multiply with an overflow error naming the item under construction.
fn checked_mul(a: u64, b: u64, item: &'static str) -> Result<u64> {
    a.checked_mul(b)
        .ok_or_else(|| Error::Config(format!("memory estimate overflow: {item}")))
}

/// Add with an overflow error naming the item under construction.
fn checked_add(a: u64, b: u64, item: &'static str) -> Result<u64> {
    a.checked_add(b)
        .ok_or_else(|| Error::Config(format!("memory estimate overflow: {item}")))
}

impl Ms2MemoryEstimate {
    /// Checked sum of the items.
    pub fn total(&self) -> Result<u64> {
        let mut sum = 0u64;
        for (_, bytes) in &self.items {
            sum = checked_add(sum, *bytes, "total")?;
        }
        Ok(sum)
    }

    /// Element counts of the single packed `generate` readout, in `read_all`
    /// order: seven id buffers (`actions`, `top`, `top_count`, `counters`,
    /// `summary`, `traj_alloc`, `identity`) then two float buffers
    /// (`top_log_prob`, `stats`).
    ///
    /// This is the one place the read layout lives: `Ms2Model::generate`
    /// asserts its bucket tensors have exactly these lengths before the
    /// batched read, and [`Ms2MemoryEstimate::generation_readout_bytes`]
    /// (used by the `readout` estimate item) prices exactly them, so the
    /// two cannot drift. Widths: `actions` is `[B*K, T*4 + A + 4]` u32,
    /// `top` is `[B, F, 2]`, `top_count`/`counters`/`summary`/`stats` are
    /// `[B]`/`[B, 5]`/`[B, 2]`/`[B, 3]`, `traj_alloc` is `[B, K, 12]`,
    /// `identity` is `[B*K, 2]`, `top_log_prob` is `[B, F]`.
    pub fn generation_readout_counts(
        batch: u64,
        trajectories: u64,
        formulas: u64,
        steps: u64,
        atoms: u64,
    ) -> Result<([u64; 7], [u64; 2])> {
        let rows = checked_mul(batch, trajectories, "readout")?;
        let record = checked_add(
            checked_add(
                checked_mul(steps, 4, "readout")?,
                atoms,
                "readout",
            )?,
            4,
            "readout",
        )?;
        let actions = checked_mul(rows, record, "readout")?;
        let top = checked_mul(
            checked_mul(batch, formulas, "readout")?,
            2,
            "readout",
        )?;
        let top_count = batch;
        let counters = checked_mul(batch, 5, "readout")?;
        let summary = checked_mul(batch, 2, "readout")?;
        let traj_alloc = checked_mul(
            checked_mul(rows, 12, "readout")?,
            1,
            "readout",
        )?;
        let identity = checked_mul(rows, 2, "readout")?;
        let top_log_prob = checked_mul(batch, formulas, "readout")?;
        let stats = checked_mul(batch, 3, "readout")?;
        Ok((
            [actions, top, top_count, counters, summary, traj_alloc, identity],
            [top_log_prob, stats],
        ))
    }

    /// Byte size of the packed `generate` readout: the id buffers of
    /// [`Ms2MemoryEstimate::generation_readout_counts`] at 4 bytes per
    /// element plus the float buffers at `elem` bytes per element.
    pub fn generation_readout_bytes(
        batch: u64,
        trajectories: u64,
        formulas: u64,
        steps: u64,
        atoms: u64,
        elem: u64,
    ) -> Result<u64> {
        let (ids, floats) = Self::generation_readout_counts(
            batch,
            trajectories,
            formulas,
            steps,
            atoms,
        )?;
        let mut sum = 0u64;
        for len in ids {
            sum = checked_add(sum, checked_mul(len, 4, "readout")?, "readout")?;
        }
        for len in floats {
            sum = checked_add(sum, checked_mul(len, elem, "readout")?, "readout")?;
        }
        Ok(sum)
    }

    /// Bytes of the named item, if present.
    pub fn get(&self, name: &str) -> Option<u64> {
        self.items
            .iter()
            .find(|(item, _)| *item == name)
            .map(|(_, bytes)| *bytes)
    }

    /// Estimate for generation with `batch` spectra, `trajectories`
    /// trajectories per spectrum, `formula_rows` resident table rows,
    /// `n_raw` raw peak capacity, `max_steps` trace steps, `window_m` (M)
    /// scored-candidate capacity and `formulas` (F) retained hypotheses.
    ///
    /// V1 §1.3 adds the complete formula workspace: `window`, `counters`,
    /// `cand` (`B*M*13` u32), `cand_feat` (`B*M*10` floats), the head
    /// activations (3 `[B, M, d]` tensors in generation) and the scores
    /// (`[B, M]`), the `cand` gate mask (`[B, M]` floats), plus the retained
    /// `top` (`B*F*2` u32), `top_log_prob` (`B*F` floats,
    /// `formula_top_log_prob`), `top_counts` (`B*F*10` u32) and `top_count`
    /// (`B` u32). The resident `log_table [1024]` joins `formula_table`,
    /// and `readout` carries the single batched read (actions, top,
    /// top_count, counters, summary, `traj_alloc`, `identity`, top_log_prob,
    /// stats). V1 I2 adds the allocation (`traj_alloc`), identity
    /// (`graph_hash`, `graph_scratch`, `identity`, `identity_scratch`) and
    /// pack (`scores`, `rerank`, `record`, `record_f`, `packed` at `R = K`,
    /// `packed_f`, `returned_count`, `evidence`) workspace.
    pub fn generation(
        model: &ModelConfig,
        formula_rows: u64,
        batch: u64,
        trajectories: u64,
        n_raw: u64,
        max_steps: u64,
        window_m: u64,
        formulas: u64,
    ) -> Result<Self> {
        let d = u64::from(model.d_model);
        let n = u64::from(model.n_peaks);
        let ld = u64::from(model.decoder_blocks);
        let a = u64::from(model.max_atoms);
        let t = max_steps;
        let heads = model.decoder.n_heads as u64;
        let head_dim = model.decoder.head_dim as u64;
        let d_state = model.decoder.d_state as u64;
        let w_enc = model.encoder.in_proj_width() as u64;
        let elem = model.dtype.size() as u64;

        // `weights`: parameter_count * elem.
        let weights = checked_mul(parameter_count(model)?, elem, "weights")?;
        // `formula_table`: formula_rows * (8 + 40 + 40) (mass and bound as
        // u32, ten float features, ten u32 counts) plus the resident
        // `log_table [1024]` (V1 §1.2, uploaded once per table/model).
        let per_row = checked_add(checked_add(8, 40, "formula_table")?, 40, "formula_table")?;
        let formula_table = checked_add(
            checked_mul(formula_rows, per_row, "formula_table")?,
            checked_mul(1024, elem, "formula_table")?,
            "formula_table",
        )?;
        // `raw_peaks`: batch * n_raw * (4 + elem) + batch * 8 * 4 (meta).
        let mz_and_intensity = checked_mul(batch, n_raw, "raw_peaks")?;
        let mz_and_intensity = checked_mul(
            mz_and_intensity,
            checked_add(4, elem, "raw_peaks")?,
            "raw_peaks",
        )?;
        let meta = checked_mul(checked_mul(batch, 8, "raw_peaks")?, 4, "raw_peaks")?;
        let raw_peaks = checked_add(mz_and_intensity, meta, "raw_peaks")?;
        // `peak_selection`: batch * n_raw * 8 (rank, position) + batch * N * (12 + 2 * elem) + features batch * N * 71 * elem.
        let ranks = checked_mul(
            checked_mul(batch, n_raw, "peak_selection")?,
            8,
            "peak_selection",
        )?;
        let kept = checked_mul(
            checked_mul(batch, n, "peak_selection")?,
            checked_add(
                12,
                checked_mul(2, elem, "peak_selection")?,
                "peak_selection",
            )?,
            "peak_selection",
        )?;
        let features = checked_mul(
            checked_mul(
                checked_mul(batch, n, "peak_selection")?,
                71,
                "peak_selection",
            )?,
            elem,
            "peak_selection",
        )?;
        let peak_selection = checked_add(
            checked_add(ranks, kept, "peak_selection")?,
            features,
            "peak_selection",
        )?;
        // Plus the per-spectrum stats (3 floats) and summary (2 u32) of PeakBuffers.
        let peak_selection = checked_add(
            peak_selection,
            checked_mul(
                batch,
                checked_add(checked_mul(3, elem, "peak_selection")?, 8, "peak_selection")?,
                "peak_selection",
            )?,
            "peak_selection",
        )?;
        // `encoder_activations`: 2 * batch * N * d * elem (ping-pong) + 2 * batch * N * W * elem (the two directions' input projections of one block).
        let ping_pong = checked_mul(
            checked_mul(
                checked_mul(
                    checked_mul(2, batch, "encoder_activations")?,
                    n,
                    "encoder_activations",
                )?,
                d,
                "encoder_activations",
            )?,
            elem,
            "encoder_activations",
        )?;
        let projections = checked_mul(
            checked_mul(
                checked_mul(
                    checked_mul(2, batch, "encoder_activations")?,
                    n,
                    "encoder_activations",
                )?,
                w_enc,
                "encoder_activations",
            )?,
            elem,
            "encoder_activations",
        )?;
        let encoder_activations = checked_add(ping_pong, projections, "encoder_activations")?;
        // `spectrum_memory`: batch * (1 + N) * d * elem * (1 + 2 * Ld) (memory plus per-layer keys and values).
        let spectrum_memory = checked_mul(
            checked_mul(
                checked_mul(
                    checked_mul(
                        batch,
                        checked_add(1, n, "spectrum_memory")?,
                        "spectrum_memory",
                    )?,
                    d,
                    "spectrum_memory",
                )?,
                elem,
                "spectrum_memory",
            )?,
            checked_add(1, checked_mul(2, ld, "spectrum_memory")?, "spectrum_memory")?,
            "spectrum_memory",
        )?;
        // `decoder_carries`: 2 * carry_bytes(batch, trajectories, Ld, ...) — two banks while the step is functional.
        let conv_history = decoder_conv_history(model)?;
        let one_bank = carry_bytes(
            batch,
            trajectories,
            ld,
            heads,
            head_dim,
            d_state,
            conv_history,
            elem,
        )?;
        let decoder_carries = checked_mul(2, one_bank, "decoder_carries")?;
        // `graph_state`: batch * trajectories * (3 * A + 16) * 4.
        let state_words = checked_add(checked_mul(3, a, "graph_state")?, 16, "graph_state")?;
        let graph_state = checked_mul(
            checked_mul(
                checked_mul(batch, trajectories, "graph_state")?,
                state_words,
                "graph_state",
            )?,
            4,
            "graph_state",
        )?;
        // `actions`: batch * trajectories * (T * 4 + A + 4) * 4.
        let trace_words = checked_add(
            checked_add(checked_mul(t, 4, "actions")?, a, "actions")?,
            4,
            "actions",
        )?;
        let actions = checked_mul(
            checked_mul(
                checked_mul(batch, trajectories, "actions")?,
                trace_words,
                "actions",
            )?,
            4,
            "actions",
        )?;
        // `atom_memory`: batch * trajectories * A * d * elem.
        let atom_memory = checked_mul(
            checked_mul(
                checked_mul(
                    checked_mul(batch, trajectories, "atom_memory")?,
                    a,
                    "atom_memory",
                )?,
                d,
                "atom_memory",
            )?,
            elem,
            "atom_memory",
        )?;
        // `head_scratch`: batch * trajectories * (5 + 18 + 4 + A + 19 * A + 4 * A) * elem.
        let per_step = checked_add(
            checked_add(
                checked_add(5 + 18 + 4, a, "head_scratch")?,
                checked_mul(19, a, "head_scratch")?,
                "head_scratch",
            )?,
            checked_mul(4, a, "head_scratch")?,
            "head_scratch",
        )?;
        let head_scratch = checked_mul(
            checked_mul(
                checked_mul(batch, trajectories, "head_scratch")?,
                per_step,
                "head_scratch",
            )?,
            elem,
            "head_scratch",
        )?;
        // `readout`: the single batched read — exactly the buffers
        // of [`Ms2MemoryEstimate::generation_readout_counts`] (actions, top,
        // top_count, counters, summary, traj_alloc, identity, top_log_prob,
        // stats), priced by
        // [`Ms2MemoryEstimate::generation_readout_bytes`] so the estimate
        // and the read share one layout.
        let readout = Self::generation_readout_bytes(
            batch,
            trajectories,
            formulas,
            t,
            a,
            elem,
        )?;
        // `cache_gather`: per decode-step bytes read for the K/V caches and
        // the atom memory. V0 sampling does no beam gather: no trajectory
        // state is moved between slots, so this is a read volume, not an
        // allocation. K/V: every trajectory attends to its spectrum's
        // per-layer keys and values (`B * K * (1 + N) * d * elem * 2 * Ld`);
        // atom memory: the pointer head reads every trajectory's atom rows
        // (`B * K * A * d * elem`).
        let cache_gather = cache_gather_per_step(batch, trajectories, n, d, ld, a, elem)?;
        // V1 §1.3 items, all checked: the complete formula workspace —
        // `window` (`B*M*2` u32), `counters` (`B*5` u32), `cand`
        // (`B*M*13` u32), `cand_feat` (`B*M*10` floats), the head
        // activations (3 `[B, M, d]` float tensors in generation) and the
        // scores (`[B, M]` floats), plus the retained `top` (`B*F*2` u32),
        // `top_log_prob` (`B*F` floats), `top_counts` (`B*F*10` u32),
        // `top_count` (`B` u32) and the `cand` gate mask (`[B, M]`
        // floats, V1 §1.2 architecture §3.8).
        let m = window_m;
        let f = formulas;
        let window = checked_mul(
            checked_mul(checked_mul(batch, m, "window")?, 2, "window")?,
            4,
            "window",
        )?;
        let counters = checked_mul(
            checked_mul(batch, 5, "counters")?,
            4,
            "counters",
        )?;
        let cand = checked_mul(
            checked_mul(checked_mul(batch, m, "cand")?, 13, "cand")?,
            4,
            "cand",
        )?;
        let cand_feat = checked_mul(
            checked_mul(checked_mul(batch, m, "cand_feat")?, 10, "cand_feat")?,
            elem,
            "cand_feat",
        )?;
        let head_act_one = checked_mul(
            checked_mul(checked_mul(batch, m, "formula_head")?, d, "formula_head")?,
            elem,
            "formula_head",
        )?;
        let formula_head = checked_mul(3, head_act_one, "formula_head")?;
        let formula_scores = checked_mul(checked_mul(batch, m, "formula_scores")?, elem, "formula_scores")?;
        let formula_mask = checked_mul(checked_mul(batch, m, "formula_mask")?, elem, "formula_mask")?;
        let top_buf = checked_mul(
            checked_mul(checked_mul(batch, f, "formula_top")?, 2, "formula_top")?,
            4,
            "formula_top",
        )?;
        let top_lp = checked_mul(
            checked_mul(batch, f, "formula_top_log_prob")?,
            elem,
            "formula_top_log_prob",
        )?;
        let top_counts = checked_mul(
            checked_mul(checked_mul(batch, f, "formula_top_counts")?, 10, "formula_top_counts")?,
            4,
            "formula_top_counts",
        )?;
        let top_count = checked_mul(batch, 4, "top_count")?;
        // V1 I2 items: the allocation, identity and pack workspace, all
        // checked. `packed`/`packed_f` are priced at `R = K` (the upper
        // bound; the bucket is keyed by the request's `R <= K`).
        let traj_alloc = checked_mul(
            checked_mul(
                checked_mul(batch, trajectories, "traj_alloc")?,
                12,
                "traj_alloc",
            )?,
            4,
            "traj_alloc",
        )?;
        // `traj_window` translates the allocation to window slots (V1 §4.4):
        // same shape as `traj_alloc`.
        let traj_window = traj_alloc;
        let graph_hash = checked_mul(
            checked_mul(batch, trajectories, "graph_hash")?,
            4,
            "graph_hash",
        )?;
        let bonds_u64 = a.saturating_sub(1).saturating_add(u64::from(model.max_ring_closures));
        let graph_stride = checked_add(
            checked_mul(3, bonds_u64, "graph_scratch")?,
            checked_mul(2, a, "graph_scratch")?,
            "graph_scratch",
        )?;
        let graph_scratch = checked_mul(
            checked_mul(
                checked_mul(batch, trajectories, "graph_scratch")?,
                graph_stride,
                "graph_scratch",
            )?,
            4,
            "graph_scratch",
        )?;
        let identity = checked_mul(
            checked_mul(
                checked_mul(batch, trajectories, "identity")?,
                2,
                "identity",
            )?,
            4,
            "identity",
        )?;
        let identity_scratch = checked_mul(
            checked_mul(
                checked_mul(batch, trajectories, "identity_scratch")?,
                checked_mul(3, a, "identity_scratch")?,
                "identity_scratch",
            )?,
            4,
            "identity_scratch",
        )?;
        let scores = checked_mul(
            checked_mul(
                checked_mul(batch, trajectories, "scores")?,
                2,
                "scores",
            )?,
            elem,
            "scores",
        )?;
        let rank = checked_mul(
            checked_mul(batch, trajectories, "rank")?,
            4,
            "rank",
        )?;
        let rerank = checked_mul(
            checked_mul(batch, trajectories, "rerank")?,
            elem,
            "rerank",
        )?;
        let pack_w = checked_add(
            checked_add(19, checked_mul(t, 4, "record")?, "record")?,
            a,
            "record",
        )?;
        let record = checked_mul(
            checked_mul(
                checked_mul(batch, trajectories, "record")?,
                pack_w,
                "record",
            )?,
            4,
            "record",
        )?;
        // `record_f`/`packed_f` are always f32 (4 bytes), on every neural
        // dtype, so device words equal the host `pack` exactly (spec §4.4).
        let record_f = checked_mul(
            checked_mul(
                checked_mul(batch, trajectories, "record_f")?,
                3,
                "record_f",
            )?,
            4,
            "record_f",
        )?;
        let packed = checked_mul(
            checked_mul(
                checked_mul(batch, trajectories, "packed")?,
                pack_w,
                "packed",
            )?,
            4,
            "packed",
        )?;
        let packed_f = checked_mul(
            checked_mul(
                checked_mul(batch, trajectories, "packed_f")?,
                3,
                "packed_f",
            )?,
            4,
            "packed_f",
        )?;
        let returned_count = checked_mul(batch, 4, "returned_count")?;
        let evidence = checked_mul(
            checked_mul(
                checked_mul(batch, trajectories, "evidence")?,
                18,
                "evidence",
            )?,
            4,
            "evidence",
        )?;
        // Packed evidence rows and log-probabilities (I3b, architecture
        // §2.4): `[B*R, 18]` u32 and `[B*R, E]` floats, priced at `R = K`
        // like `packed`.
        let packed_ev = checked_mul(
            checked_mul(
                checked_mul(batch, trajectories, "packed_ev")?,
                18,
                "packed_ev",
            )?,
            4,
            "packed_ev",
        )?;
        let packed_ev_f = checked_mul(
            checked_mul(
                checked_mul(batch, trajectories, "packed_ev_f")?,
                4,
                "packed_ev_f",
            )?,
            elem,
            "packed_ev_f",
        )?;
        // Fragment-ion assignment (architecture §2): `ion`, `ion_meta`,
        // hypothesis features, the head's activations, logits/log-probs,
        // `evidence_f` and `traj_slot`/`spec`. Zero when disabled, so V0
        // totals are bit-identical.
        let (assign_ion, assign_ion_meta, assign_features, assign_row, assign_logits, assign_log_prob, assign_evidence_f, assign_traj_slot, assign_spec) =
            if let Some(acfg) = &model.assignment {
                let jj = u64::from(acfg.hypotheses);
                let bfn = checked_mul(
                    checked_mul(checked_mul(batch, formulas, "assign_ion")?, n, "assign_ion")?,
                    jj,
                    "assign_ion",
                )?;
                let ai = checked_mul(
                    checked_mul(bfn, 12, "assign_ion")?,
                    4,
                    "assign_ion",
                )?;
                let aim = checked_mul(
                    checked_mul(
                        checked_mul(batch, formulas, "assign_ion_meta")?,
                        n,
                        "assign_ion_meta",
                    )?,
                    checked_mul(4, 4, "assign_ion_meta")?,
                    "assign_ion_meta",
                )?;
                // Hypothesis features `[B,F,N,J,10]`, two row-network
                // activations `[B,F,N,J,d]`, logits/log-probs `[B,F,N,J+1]`.
                let afeats = checked_mul(
                    checked_mul(bfn, 10, "assign_features")?,
                    elem,
                    "assign_features",
                )?;
                let arow_one = checked_mul(
                    checked_mul(bfn, d, "assign_row")?,
                    elem,
                    "assign_row",
                )?;
                let arow = checked_mul(2, arow_one, "assign_row")?;
                let j1 = jj.checked_add(1).ok_or_else(|| {
                    Error::Config("memory estimate overflow: assign_logits".to_string())
                })?;
                let acells = checked_mul(
                    checked_mul(checked_mul(batch, formulas, "assign_logits")?, n, "assign_logits")?,
                    j1,
                    "assign_logits",
                )?;
                let alogits = checked_mul(acells, elem, "assign_logits")?;
                let alogprob = checked_mul(acells, elem, "assign_log_prob")?;
                let aevf = checked_mul(
                    checked_mul(
                        checked_mul(batch, trajectories, "assign_evidence_f")?,
                        2,
                        "assign_evidence_f",
                    )?,
                    elem,
                    "assign_evidence_f",
                )?;
                let atraj = checked_mul(
                    checked_mul(
                        checked_mul(batch, trajectories, "assign_traj_slot")?,
                        2,
                        "assign_traj_slot",
                    )?,
                    4,
                    "assign_traj_slot",
                )?;
                let aspec = checked_mul(checked_mul(batch, 2, "assign_spec")?, 4, "assign_spec")?;
                (ai, aim, afeats, arow, alogits, alogprob, aevf, atraj, aspec)
            } else {
                (0, 0, 0, 0, 0, 0, 0, 0, 0)
            };

        Ok(Self {
            items: vec![
                ("weights", weights),
                ("formula_table", formula_table),
                ("raw_peaks", raw_peaks),
                ("peak_selection", peak_selection),
                ("encoder_activations", encoder_activations),
                ("spectrum_memory", spectrum_memory),
                ("decoder_carries", decoder_carries),
                ("graph_state", graph_state),
                ("actions", actions),
                ("atom_memory", atom_memory),
                ("head_scratch", head_scratch),
                ("cache_gather", cache_gather),
                ("readout", readout),
                ("window", window),
                ("counters", counters),
                ("cand", cand),
                ("cand_feat", cand_feat),
                ("formula_head", formula_head),
                ("formula_scores", formula_scores),
                ("formula_mask", formula_mask),
                ("formula_top", top_buf),
                ("formula_top_log_prob", top_lp),
                ("formula_top_counts", top_counts),
                ("top_count", top_count),
                ("traj_alloc", traj_alloc),
                ("traj_window", traj_window),
                ("graph_hash", graph_hash),
                ("graph_scratch", graph_scratch),
                ("identity", identity),
                ("identity_scratch", identity_scratch),
                ("scores", scores),
                ("rank", rank),
                ("rerank", rerank),
                ("record", record),
                ("record_f", record_f),
                ("packed", packed),
                ("packed_f", packed_f),
                ("returned_count", returned_count),
                ("evidence", evidence),
                ("packed_ev", packed_ev),
                ("packed_ev_f", packed_ev_f),
                ("assign_ion", assign_ion),
                ("assign_ion_meta", assign_ion_meta),
                ("assign_features", assign_features),
                ("assign_row", assign_row),
                ("assign_logits", assign_logits),
                ("assign_log_prob", assign_log_prob),
                ("assign_evidence_f", assign_evidence_f),
                ("assign_traj_slot", assign_traj_slot),
                ("assign_spec", assign_spec),
            ],
        })
    }

    /// Estimate for generation with enumeration resident artifacts (V1 §1.4):
    /// the base [`Ms2MemoryEstimate::generation`] items plus `rare` (`32 P`
    /// bytes), `bounds` (words `* 4`, under 64 KiB), `lane_stats`
    /// (`8 B P` bytes), `offsets` (`4 B P` bytes, together `12 B P`) and
    /// `enum_meta` (`[B, 8]` u32, `32 B` bytes, workspace-owned).
    /// `enum_p` is the rare-table rows `P`, `enum_bounds_words` the packed
    /// bounds length in `u32` words. With both zero the total equals
    /// [`Ms2MemoryEstimate::generation`].
    #[allow(clippy::too_many_arguments)]
    pub fn generation_with_enum(
        model: &ModelConfig,
        formula_rows: u64,
        batch: u64,
        trajectories: u64,
        n_raw: u64,
        max_steps: u64,
        window_m: u64,
        formulas: u64,
        enum_p: u64,
        enum_bounds_words: u64,
    ) -> Result<Self> {
        let mut est = Self::generation(
            model,
            formula_rows,
            batch,
            trajectories,
            n_raw,
            max_steps,
            window_m,
            formulas,
        )?;
        let rare = checked_mul(enum_p, 32, "rare")?;
        let bounds = checked_mul(enum_bounds_words, 4, "bounds")?;
        let lanes = checked_mul(batch, enum_p, "lane_stats")?;
        let lane_stats = checked_mul(lanes, 8, "lane_stats")?;
        let offsets = checked_mul(lanes, 4, "offsets")?;
        let enum_meta = checked_mul(batch, 32, "enum_meta")?;
        est.items.push(("rare", rare));
        est.items.push(("bounds", bounds));
        est.items.push(("lane_stats", lane_stats));
        est.items.push(("offsets", offsets));
        est.items.push(("enum_meta", enum_meta));
        Ok(est)
    }

    /// Estimate for training with `batch` spectra, `targets` target slots per
    /// spectrum, `formula_rows` resident table rows, `n_raw` raw peak capacity,
    /// `max_steps` trace steps and `window_m` (M) scored-candidate capacity.
    ///
    /// The forward items are the values the tape keeps through the backward
    /// pass, `activation_gradients` the retained gradient of every node,
    /// `gradients` the parameter gradients and `optimizer_moments` the AdamW
    /// state (see the type docs for the per-item shapes).
    /// V1 §1.3 adds the complete formula workspace (`window`, `counters`,
    /// `cand`, `cand_feat`, the head activations with their gradients, the
    /// scores with their gradient, the `cand` gate mask, the retained `top`,
    /// `top_log_prob`, `top_counts` and `top_count`) plus the gold path
    /// (`gold_counts`, the `count_features` output and the row-network
    /// activations, each forward plus gradient, and `gold_slot`).
    pub fn training(
        model: &ModelConfig,
        formula_rows: u64,
        batch: u64,
        targets: u64,
        n_raw: u64,
        max_steps: u64,
        window_m: u64,
    ) -> Result<Self> {
        let d = u64::from(model.d_model);
        let n = u64::from(model.n_peaks);
        let t = max_steps;
        let ld = u64::from(model.decoder_blocks);
        let a = u64::from(model.max_atoms);
        let attn_heads = u64::from(model.attention_heads);
        let w_enc = model.encoder.in_proj_width() as u64;
        let w_dec = model.decoder.in_proj_width() as u64;
        let elem = model.dtype.size() as u64;

        // `weights`: parameter_count * elem.
        let weights = checked_mul(parameter_count(model)?, elem, "weights")?;
        // `formula_table`: formula_rows * (8 + 40 + 40) (mass and bound as
        // u32, ten float features, ten u32 counts) plus the resident
        // `log_table [1024]` (V1 §1.2, uploaded once per table/model).
        let per_row = checked_add(checked_add(8, 40, "formula_table")?, 40, "formula_table")?;
        let formula_table = checked_add(
            checked_mul(formula_rows, per_row, "formula_table")?,
            checked_mul(1024, elem, "formula_table")?,
            "formula_table",
        )?;
        // `raw_peaks`: batch * n_raw * (4 + elem) + batch * 8 * 4 (meta).
        let mz_and_intensity = checked_mul(batch, n_raw, "raw_peaks")?;
        let mz_and_intensity = checked_mul(
            mz_and_intensity,
            checked_add(4, elem, "raw_peaks")?,
            "raw_peaks",
        )?;
        let meta = checked_mul(checked_mul(batch, 8, "raw_peaks")?, 4, "raw_peaks")?;
        let raw_peaks = checked_add(mz_and_intensity, meta, "raw_peaks")?;
        // `peak_selection`: batch * n_raw * 8 (rank, position) + batch * N * (12 + 2 * elem) + features batch * N * 71 * elem.
        let ranks = checked_mul(
            checked_mul(batch, n_raw, "peak_selection")?,
            8,
            "peak_selection",
        )?;
        let kept = checked_mul(
            checked_mul(batch, n, "peak_selection")?,
            checked_add(
                12,
                checked_mul(2, elem, "peak_selection")?,
                "peak_selection",
            )?,
            "peak_selection",
        )?;
        let features = checked_mul(
            checked_mul(
                checked_mul(batch, n, "peak_selection")?,
                71,
                "peak_selection",
            )?,
            elem,
            "peak_selection",
        )?;
        let peak_selection = checked_add(
            checked_add(ranks, kept, "peak_selection")?,
            features,
            "peak_selection",
        )?;
        // Plus the per-spectrum stats (3 floats) and summary (2 u32) of PeakBuffers.
        let peak_selection = checked_add(
            peak_selection,
            checked_mul(
                batch,
                checked_add(checked_mul(3, elem, "peak_selection")?, 8, "peak_selection")?,
                "peak_selection",
            )?,
            "peak_selection",
        )?;
        // Everything from here to the formula workspace is what a training
        // step holds at its peak, which is the end of the backward pass: the
        // forward values the tape keeps (a buffer stays alive when an
        // adjoint rule captured it or a later operand still reads it; the
        // output of an add, a reshape or a permute that nothing captures is
        // freed as soon as its consumer has run), and one gradient buffer for
        // every node of the tape (`Var::backward_retain`, minus the nodes
        // that share their consumer's buffer: a view, or either operand of
        // an add). The forward items below hold the first, in the order of
        // the pass; `activation_gradients` holds the second. Each count was
        // reconciled against the live bytes at the boundaries of one step
        // and against the list of retained gradients by shape (V0 shapes,
        // P2.2, `tests/ms2_footprint.rs`).
        //
        // Sizes are those of the layout that allocates most: every target
        // slot occupied (the padded teacher pass; the compact and ragged
        // passes allocate for occupied slots and trace cells only). The
        // encoder scans run over the kept peaks laid end to end
        // (`Ms2Encoder::peak_packing`, the default), at most
        // `min(n_raw, N)` cells a spectrum.
        let sum = |terms: &[u64], item: &'static str| -> Result<u64> {
            terms.iter().try_fold(0u64, |acc, &term| checked_add(acc, term, item))
        };
        let prod = |factors: &[u64], item: &'static str| -> Result<u64> {
            factors.iter().try_fold(1u64, |acc, &factor| checked_mul(acc, factor, item))
        };
        let le = u64::from(model.encoder_blocks);
        let mixers_e = checked_mul(2, le, "encoder_activations")?;
        let mem = checked_add(1, n, "spectrum_memory")?;
        let rows = checked_mul(batch, targets, "decoder_activations")?;
        let cells = checked_mul(rows, t, "decoder_activations")?;
        let positions = checked_mul(rows, max_steps.saturating_sub(1), "head_scratch")?;
        let cells_e = checked_mul(batch, n.min(n_raw), "encoder_activations")?;
        let unpacked = prod(&[batch, n, d], "encoder_activations")?;
        let memory = prod(&[batch, mem, d], "spectrum_memory")?;
        if attn_heads == 0 {
            return Err(Error::Config(
                "memory estimate: attention_heads is 0".to_string(),
            ));
        }
        // Mixer widths: inner width, one of `B`/`C`, heads, head width and
        // the rotation angles (zero when the dynamics are real).
        let widths = |ssm: &SsmConfig| -> (u64, u64, u64, u64, u64) {
            (
                ssm.d_inner() as u64,
                ssm.bc_width() as u64,
                ssm.n_heads as u64,
                ssm.head_dim as u64,
                ssm.theta_width() as u64,
            )
        };
        let (di_e, bc_e, h_e, _, theta_e) = widths(&model.encoder);
        let (di_d, bc_d, h_d, p_d, theta_d) = widths(&model.decoder);
        // Scan chunking matches `ssd_chunked`: the chunk is the config size
        // clamped to the sequence, splitting the sequence into `chunks`.
        let chunk_d = (model.decoder.chunk_size as u64).min(t.max(1)).max(1);
        let chunks_d = t.div_ceil(chunk_d);

        // `encoder_activations`, per mixer (two directions a block) and
        // packed cell: the block input and its norm, the mixer output and
        // the residual sum (`4 d`), the fused projection and the pieces it
        // is split into (`2 W`), the norm's scale and the scan's per-cell
        // scalars (6); per mixer one unpacked `[B, N, d]` output; and once
        // the peak embedding, the final norm and the selected rows
        // (`3 [B, N, d]`).
        let encoder_activations = checked_mul(
            sum(
                &[
                    prod(
                        &[
                            mixers_e,
                            cells_e,
                            sum(&[4 * d, 6, 2 * w_enc], "encoder_activations")?,
                        ],
                        "encoder_activations",
                    )?,
                    checked_mul(mixers_e, unpacked, "encoder_activations")?,
                    checked_mul(3, unpacked, "encoder_activations")?,
                ],
                "encoder_activations",
            )?,
            elem,
            "encoder_activations",
        )?;
        // `encoder_scan_retained`, per mixer and packed cell: the activated
        // `x`, `B`, `C` (`d_inner + 2 bc`), the normed `B` and `C` (`2 bc`),
        // the scan output and the gated output (`2 d_inner`), and `dt`,
        // `lambda` with the two norm scales (`4 H`). The reset-aware scan of
        // the packed rows keeps no band of its own.
        let encoder_scan_retained = prod(
            &[
                mixers_e,
                cells_e,
                sum(&[3 * di_e, 4 * bc_e, 4 * h_e], "encoder_scan_retained")?,
                elem,
            ],
            "encoder_scan_retained",
        )?;
        // `spectrum_memory`: the memory `[B, 1 + N, d]` and its mask. The
        // per-layer key and value projections are not kept; their per-head
        // copies are, in `decoder_attention_retained`.
        let spectrum_memory = checked_mul(
            checked_add(memory, checked_mul(batch, mem, "spectrum_memory")?, "spectrum_memory")?,
            elem,
            "spectrum_memory",
        )?;
        // `targets`: batch * targets * T * (4 + 4 + A) * 4 (tokens and replay).
        let per_position = checked_add(checked_add(4, 4, "targets")?, a, "targets")?;
        let targets_bytes = prod(&[cells, per_position, 4], "targets")?;
        // `decoder_embed_retained`: the summed input embedding `[rows, T, d]`.
        // The five lookups and the adds between them are freed as they are
        // consumed.
        let decoder_embed_retained = prod(&[cells, d, elem], "decoder_embed_retained")?;
        // `decoder_activations`, per layer and cell: the block's norm output
        // and the mixer's output (`2 d`), the fused projection and the
        // pieces it is split into (`2 W`).
        let decoder_activations = prod(
            &[
                ld,
                cells,
                checked_mul(2, checked_add(d, w_dec, "decoder_activations")?, "decoder_activations")?,
                elem,
            ],
            "decoder_activations",
        )?;
        // `decoder_mixer_retained`, per layer: per cell the activated `x`,
        // `B`, `C`, the normed `B` and `C`, the scan output and the gated
        // output (`3 d_inner + 4 bc`), the scan's two angle tables
        // (`2 theta`), `dt`, `lambda` and the norm scales (`4 H`) and two
        // scalars; per row and chunk the scan's intra-chunk band
        // `[H, C, C]` and one state row `[H, P]`.
        let decoder_mixer_retained = prod(
            &[
                ld,
                sum(
                    &[
                        checked_mul(
                            cells,
                            sum(
                                &[3 * di_d, 4 * bc_d, 2 * theta_d, 4 * h_d, 2],
                                "decoder_mixer_retained",
                            )?,
                            "decoder_mixer_retained",
                        )?,
                        prod(
                            &[
                                rows,
                                chunks_d,
                                h_d,
                                checked_add(
                                    checked_mul(chunk_d, chunk_d, "decoder_mixer_retained")?,
                                    p_d,
                                    "decoder_mixer_retained",
                                )?,
                            ],
                            "decoder_mixer_retained",
                        )?,
                    ],
                    "decoder_mixer_retained",
                )?,
                elem,
            ],
            "decoder_mixer_retained",
        )?;
        // `attention_scores`: the attention weights `[B, h, G*T, 1 + N]` of
        // every layer. The raw scores are not kept: the fused weights' adjoint
        // reads the weights alone.
        let weights_cells = prod(&[batch, attn_heads, targets, t, mem], "attention_scores")?;
        let attention_scores = prod(&[ld, weights_cells, elem], "attention_scores")?;
        // `decoder_attention_retained`, per layer: per query the normed
        // input, the per-head queries, the context, the residual sum and one
        // more `d`-wide buffer, with the norm's scale (`5 d + 1`), and the
        // per-head keys and values (`2 [B, 1 + N, d]`). The key mask is read
        // as `[B, 1 + N]` and never expanded.
        let decoder_attention_retained = prod(
            &[
                ld,
                checked_add(
                    checked_mul(
                        cells,
                        checked_add(5 * d, 1, "decoder_attention_retained")?,
                        "decoder_attention_retained",
                    )?,
                    checked_mul(2, memory, "decoder_attention_retained")?,
                    "decoder_attention_retained",
                )?,
                elem,
            ],
            "decoder_attention_retained",
        )?;
        // `atom_memory` (training): batch * targets * A * d * elem, the creation-state rows
        // gathered once per target after the parallel decoder pass.
        let atom_rows = prod(&[rows, a, d], "atom_memory")?;
        let train_atom_memory = checked_mul(atom_rows, elem, "atom_memory")?;
        // `decoder_head_positions_retained`: per scored position the hidden
        // row and the pointer query (`2 d`) and, per field, the effective
        // mask and the log-probabilities (`2 (5 + 18 + 4 + A)`); per target
        // the projected atom memory `[A, d]`. No `[positions, A, d]` key
        // tensor exists (the pointer scores are taken term by term), nor
        // the `[rows, 19, A]` by-type table.
        let field_width = checked_add(5 + 18 + 4, a, "head_scratch")?;
        let decoder_head_positions_retained = checked_mul(
            checked_add(
                checked_mul(
                    positions,
                    checked_add(
                        2 * d,
                        checked_mul(2, field_width, "decoder_head_positions_retained")?,
                        "decoder_head_positions_retained",
                    )?,
                    "decoder_head_positions_retained",
                )?,
                atom_rows,
                "decoder_head_positions_retained",
            )?,
            elem,
            "decoder_head_positions_retained",
        )?;
        // `head_scratch` (training): the teacher pass's outputs, the four
        // distributions and the gathered fields `[rows, T, 5 + 18 + 4 + A + 4]`.
        let outputs = checked_mul(
            cells,
            checked_add(field_width, 4, "head_scratch")?,
            "head_scratch",
        )?;
        let train_head_scratch = checked_mul(outputs, elem, "head_scratch")?;
        // `gradients`: equal to `weights`.
        let gradients = weights;
        // `optimizer_moments`: 2 * `weights`.
        let optimizer_moments = checked_mul(2, weights, "optimizer_moments")?;
        // `activation_gradients`: the retained gradient of every node.
        //
        // Encoder, per mixer and packed cell: the projection (`W`), the
        // convolved `x B C` (`d_inner + 2 bc`), the gate, `x`, the scan
        // output in both layouts and the gated output (`5 d_inner`), `B`
        // and `C` before and after their norm (`4 bc`), four `d`-wide
        // buffers of the block, the angles (`3 theta`) and the per-head
        // scalars (`9 H`); `2 mixers + 3` unpacked `[B, N, d]` buffers; the
        // memory.
        let enc_grad = sum(
            &[
                prod(
                    &[
                        mixers_e,
                        cells_e,
                        sum(
                            &[w_enc, 6 * di_e, 6 * bc_e, 4 * d, 3 * theta_e, 9 * h_e],
                            "activation_gradients",
                        )?,
                    ],
                    "activation_gradients",
                )?,
                checked_mul(
                    checked_add(checked_mul(2, mixers_e, "activation_gradients")?, 3, "activation_gradients")?,
                    unpacked,
                    "activation_gradients",
                )?,
                memory,
            ],
            "activation_gradients",
        )?;
        // Decoder, per layer. Mixer, per cell: as the encoder's with one
        // scan-output layout fewer (`W + 5 d_inner + 6 bc + 4 d + 3 theta
        // + 7 H`). Attention: scores and weights (`2 [B, h, G*T, 1 + N]`),
        // seven `d`-wide buffers per query (normed input, queries in both
        // layouts, context in both layouts, output projection, residual)
        // and the keys and values in both layouts (`4 [B, 1 + N, d]`).
        let dec_grad = checked_mul(
            ld,
            sum(
                &[
                    checked_mul(
                        cells,
                        sum(
                            &[w_dec, 5 * di_d, 6 * bc_d, 4 * d, 3 * theta_d, 7 * h_d],
                            "activation_gradients",
                        )?,
                        "activation_gradients",
                    )?,
                    checked_mul(2, weights_cells, "activation_gradients")?,
                    prod(&[7, cells, d], "activation_gradients")?,
                    checked_mul(4, memory, "activation_gradients")?,
                ],
                "activation_gradients",
            )?,
            "activation_gradients",
        )?;
        // Heads, per scored position: the hidden row, the pointer query and
        // its sum (`3 d`); the residual-row scores gathered per atom
        // (`8 A`), four pointer-width and one more `A`-wide buffer
        // (`5 A`); four type-width, four kind-width and six bond-width
        // buffers (`4 * 18 + 4 * 5 + 6 * 4`); the residual scores (8), eight
        // gathered scalars and the four field values (`8 + 4`). Per target
        // the atom memory and its projection (`2 [A, d]`). Then the summed
        // embedding and the outputs.
        let head_grad = sum(
            &[
                checked_mul(
                    positions,
                    sum(
                        &[3 * d, 13 * a, 4 * 18 + 4 * 5 + 6 * 4, 8 + 8 + 4],
                        "activation_gradients",
                    )?,
                    "activation_gradients",
                )?,
                checked_mul(2, atom_rows, "activation_gradients")?,
                checked_mul(cells, d, "activation_gradients")?,
                outputs,
            ],
            "activation_gradients",
        )?;
        let activation_gradients = checked_mul(
            sum(&[enc_grad, dec_grad, head_grad], "activation_gradients")?,
            elem,
            "activation_gradients",
        )?;

        Ok(Self {
            items: vec![
                ("weights", weights),
                ("formula_table", formula_table),
                ("raw_peaks", raw_peaks),
                ("peak_selection", peak_selection),
                ("encoder_activations", encoder_activations),
                ("spectrum_memory", spectrum_memory),
                ("targets", targets_bytes),
                ("decoder_activations", decoder_activations),
                ("attention_scores", attention_scores),
                ("gradients", gradients),
                ("optimizer_moments", optimizer_moments),
                ("activation_gradients", activation_gradients),
                ("atom_memory", train_atom_memory),
                ("head_scratch", train_head_scratch),
                ("decoder_mixer_retained", decoder_mixer_retained),
                ("decoder_attention_retained", decoder_attention_retained),
                (
                    "decoder_head_positions_retained",
                    decoder_head_positions_retained,
                ),
                ("decoder_embed_retained", decoder_embed_retained),
                ("encoder_scan_retained", encoder_scan_retained),
                // V1 §1.3 complete formula workspace (training holds the
                // same `FormulaBuffers` with F = 1, plus the `cand` gate
                // mask): `window` (`B*M*2` u32), `counters` (`B*5` u32),
                // `cand`, `cand_feat`, the scored head activations with
                // their gradients (6 `[B, M, d]` floats) and the scores
                // with their gradient (2 `[B, M]` floats), the retained
                // `top` (`B*1*2` u32), `top_log_prob` (`B*1` floats),
                // `top_counts` (`B*1*10` u32), `top_count` (`B` u32) and
                // `formula_mask` (`[B, M]` floats).
                (
                    "window",
                    checked_mul(
                        checked_mul(
                            checked_mul(batch, window_m, "window")?,
                            2,
                            "window",
                        )?,
                        4,
                        "window",
                    )?,
                ),
                (
                    "counters",
                    checked_mul(
                        checked_mul(batch, 5, "counters")?,
                        4,
                        "counters",
                    )?,
                ),
                (
                    "cand",
                    checked_mul(
                        checked_mul(
                            checked_mul(batch, window_m, "cand")?,
                            13,
                            "cand",
                        )?,
                        4,
                        "cand",
                    )?,
                ),
                (
                    "cand_feat",
                    checked_mul(
                        checked_mul(checked_mul(batch, window_m, "cand_feat")?, 10, "cand_feat")?,
                        elem,
                        "cand_feat",
                    )?,
                ),
                (
                    "formula_head",
                    checked_mul(
                        checked_mul(
                            checked_mul(
                                checked_mul(batch, window_m, "formula_head")?,
                                d,
                                "formula_head",
                            )?,
                            elem,
                            "formula_head",
                        )?,
                        6,
                        "formula_head",
                    )?,
                ),
                (
                    "formula_scores",
                    checked_mul(
                        checked_mul(
                            checked_mul(batch, window_m, "formula_scores")?,
                            elem,
                            "formula_scores",
                        )?,
                        2,
                        "formula_scores",
                    )?,
                ),
                (
                    "formula_mask",
                    checked_mul(
                        checked_mul(batch, window_m, "formula_mask")?,
                        elem,
                        "formula_mask",
                    )?,
                ),
                (
                    "formula_top",
                    checked_mul(
                        checked_mul(
                            checked_mul(batch, 1, "formula_top")?,
                            2,
                            "formula_top",
                        )?,
                        4,
                        "formula_top",
                    )?,
                ),
                (
                    "formula_top_log_prob",
                    checked_mul(
                        checked_mul(batch, 1, "formula_top_log_prob")?,
                        elem,
                        "formula_top_log_prob",
                    )?,
                ),
                (
                    "formula_top_counts",
                    checked_mul(
                        checked_mul(
                            checked_mul(batch, 1, "formula_top_counts")?,
                            10,
                            "formula_top_counts",
                        )?,
                        4,
                        "formula_top_counts",
                    )?,
                ),
                (
                    "top_count",
                    checked_mul(batch, 4, "top_count")?,
                ),
                (
                    "gold_counts",
                    checked_mul(
                        checked_mul(batch, 10, "gold_counts")?,
                        4,
                        "gold_counts",
                    )?,
                ),
                // Gold path (V1 §1.2 teacher forcing): the `count_features`
                // output (`[B, 10]` floats, forward plus gradient) and the
                // row-network activations (3 `[B, d]` tensors — `row_in`
                // output, SiLU output, `row_out` output — forward plus
                // gradient, the same 3 the scored `formula_head` counts per
                // `[B, M, d]` slot).
                (
                    "gold_feat",
                    checked_mul(
                        checked_mul(
                            checked_mul(batch, 10, "gold_feat")?,
                            elem,
                            "gold_feat",
                        )?,
                        2,
                        "gold_feat",
                    )?,
                ),
                (
                    "gold_formula_head",
                    checked_mul(
                        checked_mul(
                            checked_mul(
                                checked_mul(3, batch, "gold_formula_head")?,
                                d,
                                "gold_formula_head",
                            )?,
                            elem,
                            "gold_formula_head",
                        )?,
                        2,
                        "gold_formula_head",
                    )?,
                ),
                (
                    "gold_slot",
                    checked_mul(batch, 4, "gold_slot")?,
                ),
                (
                    "assign_ion",
                    if let Some(acfg) = &model.assignment {
                        let jj = u64::from(acfg.hypotheses);
                        checked_mul(
                            checked_mul(
                                checked_mul(
                                    checked_mul(batch, 1, "assign_ion")?,
                                    n,
                                    "assign_ion",
                                )?,
                                jj,
                                "assign_ion",
                            )?,
                            checked_mul(12, 4, "assign_ion")?,
                            "assign_ion",
                        )?
                    } else {
                        0
                    },
                ),
                (
                    "assign_ion_meta",
                    if let Some(acfg) = &model.assignment {
                        let _ = acfg;
                        checked_mul(
                            checked_mul(
                                checked_mul(batch, 1, "assign_ion_meta")?,
                                n,
                                "assign_ion_meta",
                            )?,
                            checked_mul(4, 4, "assign_ion_meta")?,
                            "assign_ion_meta",
                        )?
                    } else {
                        0
                    },
                ),
                (
                    "assign_features",
                    if let Some(acfg) = &model.assignment {
                        let jj = u64::from(acfg.hypotheses);
                        checked_mul(
                            checked_mul(
                                checked_mul(
                                    checked_mul(batch, 1, "assign_features")?,
                                    n,
                                    "assign_features",
                                )?,
                                jj,
                                "assign_features",
                            )?,
                            checked_mul(
                                checked_mul(10, elem, "assign_features")?,
                                2,
                                "assign_features",
                            )?,
                            "assign_features",
                        )?
                    } else {
                        0
                    },
                ),
                (
                    "assign_row",
                    if let Some(acfg) = &model.assignment {
                        let jj = u64::from(acfg.hypotheses);
                        checked_mul(
                            checked_mul(
                                checked_mul(
                                    checked_mul(batch, 1, "assign_row")?,
                                    n,
                                    "assign_row",
                                )?,
                                jj,
                                "assign_row",
                            )?,
                            checked_mul(
                                checked_mul(
                                    checked_mul(2, d, "assign_row")?,
                                    elem,
                                    "assign_row",
                                )?,
                                2,
                                "assign_row",
                            )?,
                            "assign_row",
                        )?
                    } else {
                        0
                    },
                ),
                (
                    "assign_logits",
                    if let Some(acfg) = &model.assignment {
                        let j1 = u64::from(acfg.hypotheses).checked_add(1).ok_or_else(|| {
                            Error::Config("memory estimate overflow: assign_logits".to_string())
                        })?;
                        checked_mul(
                            checked_mul(
                                checked_mul(batch, n, "assign_logits")?,
                                j1,
                                "assign_logits",
                            )?,
                            checked_mul(elem, 2, "assign_logits")?,
                            "assign_logits",
                        )?
                    } else {
                        0
                    },
                ),
                (
                    "assign_labels",
                    if let Some(acfg) = &model.assignment {
                        let ll = u64::from(acfg.labels);
                        let lab = checked_mul(
                            checked_mul(
                                checked_mul(batch, ll, "assign_labels")?,
                                12,
                                "assign_labels",
                            )?,
                            4,
                            "assign_labels",
                        )?;
                        let mask = checked_mul(
                            checked_mul(
                                checked_mul(batch, n, "assign_labels")?,
                                u64::from(acfg.hypotheses).checked_add(1).ok_or_else(|| {
                                    Error::Config(
                                        "memory estimate overflow: assign_labels".to_string(),
                                    )
                                })?,
                                "assign_labels",
                            )?,
                            checked_mul(elem, 2, "assign_labels")?,
                            "assign_labels",
                        )?;
                        let state = checked_mul(
                            checked_mul(
                                checked_mul(batch, n, "assign_labels")?,
                                4,
                                "assign_labels",
                            )?,
                            2,
                            "assign_labels",
                        )?;
                        checked_add(
                            checked_add(lab, mask, "assign_labels")?,
                            state,
                            "assign_labels",
                        )?
                    } else {
                        0
                    },
                ),
            ],
        })
    }

    /// Estimate for training with enumeration resident artifacts (V1 §1.4):
    /// the base [`Ms2MemoryEstimate::training`] items plus `rare`, `bounds`,
    /// `lane_stats`, `offsets` and `enum_meta` (`32 B` bytes, same shapes as in
    /// [`Ms2MemoryEstimate::generation_with_enum`]).
    pub fn training_with_enum(
        model: &ModelConfig,
        formula_rows: u64,
        batch: u64,
        targets: u64,
        n_raw: u64,
        max_steps: u64,
        window_m: u64,
        enum_p: u64,
        enum_bounds_words: u64,
    ) -> Result<Self> {
        let mut est = Self::training(
            model, formula_rows, batch, targets, n_raw, max_steps, window_m,
        )?;
        let rare = checked_mul(enum_p, 32, "rare")?;
        let bounds = checked_mul(enum_bounds_words, 4, "bounds")?;
        let lanes = checked_mul(batch, enum_p, "lane_stats")?;
        let lane_stats = checked_mul(lanes, 8, "lane_stats")?;
        let offsets = checked_mul(lanes, 4, "offsets")?;
        let enum_meta = checked_mul(batch, 32, "enum_meta")?;
        est.items.push(("rare", rare));
        est.items.push(("bounds", bounds));
        est.items.push(("lane_stats", lane_stats));
        est.items.push(("offsets", offsets));
        est.items.push(("enum_meta", enum_meta));
        Ok(est)
    }

    /// Refuse a configuration whose estimate exceeds `max_device_bytes`.
    ///
    /// A configuration that does not fit is refused before allocation, never
    /// silently reduced. The error names the total, the limit and the largest
    /// item so the caller knows what dominates.
    pub fn check_limit(&self, max_device_bytes: u64) -> Result<()> {
        let total = self.total()?;
        if total <= max_device_bytes {
            return Ok(());
        }
        let (name, bytes) = self
            .items
            .iter()
            .max_by_key(|(_, b)| *b)
            .copied()
            .unwrap_or(("none", 0));
        Err(Error::Config(format!(
            "memory estimate {total} bytes exceeds limit {max_device_bytes} bytes: \
             largest item {name} = {bytes} bytes"
        )))
    }
}

/// Recurrent convolution history elements per decoder layer and trajectory.
///
/// One bank keeps the last `k - 1` inputs of the fused `x`/`B`/`C` projection;
/// a mixer without convolution keeps none.
fn decoder_conv_history(model: &ModelConfig) -> Result<u64> {
    let dec = &model.decoder;
    match dec.conv_kernel {
        None => Ok(0),
        Some(k) => {
            let back = (k as u64).checked_sub(1).ok_or_else(|| {
                Error::Config("memory estimate overflow: decoder_carries".to_string())
            })?;
            let channels = (dec.d_inner() + 2 * dec.bc_width()) as u64;
            checked_mul(back, channels, "decoder_carries")
        }
    }
}

/// One recurrent-state bank in bytes: `batch * trajectories * layers * (2 *
/// heads * head_dim * d_state + heads * d_state / 2 + conv_history) *
/// elem_bytes`.
///
/// The `2 * heads * head_dim * d_state` terms are the `h` and `last_u` states
/// and `heads * d_state / 2` the rotational `angle`; `conv_history` is the
/// convolution history per layer (zero without convolution). Generation holds
/// two banks while the step is functional.
pub fn carry_bytes(
    batch: u64,
    trajectories: u64,
    layers: u64,
    heads: u64,
    head_dim: u64,
    d_state: u64,
    conv_history: u64,
    elem_bytes: u64,
) -> Result<u64> {
    const ITEM: &str = "decoder_carries";
    let hu = checked_mul(
        checked_mul(checked_mul(2, heads, ITEM)?, head_dim, ITEM)?,
        d_state,
        ITEM,
    )?;
    let angle = checked_mul(heads, d_state, ITEM)? / 2;
    let per_layer = checked_add(checked_add(hu, angle, ITEM)?, conv_history, ITEM)?;
    let elems = checked_mul(
        checked_mul(checked_mul(batch, trajectories, ITEM)?, layers, ITEM)?,
        per_layer,
        ITEM,
    )?;
    checked_mul(elems, elem_bytes, ITEM)
}

/// Per-decode-step cache-gather read volume in bytes: the K/V caches plus the
/// atom memory a sampling step reads.
///
/// V0 sampling does no beam gather (no ancestry movement, no slot reorder),
/// so this item is a read volume, not an allocation: every trajectory attends
/// to its spectrum's per-layer keys and values
/// (`batch * trajectories * (1 + N) * d * elem * 2 * Ld`) and the pointer
/// head reads every trajectory's atom rows (`batch * trajectories * A * d *
/// elem`). Computed with checked arithmetic; an overflow is an error.
pub fn cache_gather_per_step(
    batch: u64,
    trajectories: u64,
    n_peaks: u64,
    d_model: u64,
    decoder_layers: u64,
    max_atoms: u64,
    elem_bytes: u64,
) -> Result<u64> {
    const ITEM: &str = "cache_gather";
    // K/V: batch * trajectories * (1 + N) * d * elem * (2 * Ld).
    let kv_elems = checked_mul(
        checked_mul(
            checked_mul(
                checked_mul(batch, trajectories, ITEM)?,
                checked_add(1, n_peaks, ITEM)?,
                ITEM,
            )?,
            d_model,
            ITEM,
        )?,
        checked_mul(2, decoder_layers, ITEM)?,
        ITEM,
    )?;
    let kv = checked_mul(kv_elems, elem_bytes, ITEM)?;
    // Atom memory: batch * trajectories * A * d * elem.
    let atom_elems = checked_mul(
        checked_mul(
            checked_mul(checked_mul(batch, trajectories, ITEM)?, max_atoms, ITEM)?,
            d_model,
            ITEM,
        )?,
        1,
        ITEM,
    )?;
    let atom = checked_mul(atom_elems, elem_bytes, ITEM)?;
    checked_add(kv, atom, ITEM)
}

/// Parameters of one pre-norm [`crate::models::mamba3::Mamba3Block`] for an
/// [`SsmConfig`]: the block norm weight, the fused input projection and the
/// output projection (with biases only where the config enables them), the
/// per-head `dt`/`A`/`skip` parameters, the per-group `B`/`C` biases and norms
/// only where the config enables them, and the convolution weight and bias
/// only when the config has one.
pub fn block_parameter_count(ssm: &SsmConfig) -> Result<u64> {
    const ITEM: &str = "block parameters";
    let d_model = ssm.d_model as u64;
    let heads = ssm.n_heads as u64;
    let state = ssm.d_state as u64;
    let d_inner = ssm.d_inner() as u64;
    let in_width = ssm.in_proj_width() as u64;

    // Block pre-norm gain.
    let mut total = d_model;
    // Fused input projection d_model -> in_proj_width, bias iff ssm.bias.
    total = checked_add(total, checked_mul(d_model, in_width, ITEM)?, ITEM)?;
    if ssm.bias {
        total = checked_add(total, in_width, ITEM)?;
    }
    // Output projection d_inner -> d_model, bias iff ssm.bias.
    total = checked_add(total, checked_mul(d_inner, d_model, ITEM)?, ITEM)?;
    if ssm.bias {
        total = checked_add(total, d_model, ITEM)?;
    }
    // Short causal convolution over x, B and C together, iff configured.
    if let Some(k) = ssm.conv_kernel {
        let channels = (ssm.d_inner() + 2 * ssm.bc_width()) as u64;
        total = checked_add(total, checked_mul(k as u64, channels, ITEM)?, ITEM)?;
        total = checked_add(total, channels, ITEM)?;
    }
    // Per-head dt bias, log-A and direct skip iff configured.
    total = checked_add(total, heads, ITEM)?;
    total = checked_add(total, heads, ITEM)?;
    if ssm.skip_connection {
        total = checked_add(total, heads, ITEM)?;
    }
    // Per-head channel biases on B and C iff configured.
    if ssm.bc_bias {
        total = checked_add(total, checked_mul(heads, state, ITEM)?, ITEM)?;
        total = checked_add(total, checked_mul(heads, state, ITEM)?, ITEM)?;
    }
    // RMS gain on B/C iff configured (one norm shared by both).
    if ssm.bc_norm {
        total = checked_add(total, state, ITEM)?;
    }
    // Post-gate norm iff configured.
    if ssm.post_gate_norm {
        total = checked_add(total, d_inner, ITEM)?;
    }
    Ok(total)
}

/// Weights of the architecture of `docs/MS2_V0_ARCHITECTURE.md` section 4.
///
/// Breakdown: peak embedding `71*d + d + d*d + d`; metadata embeddings
/// `(3 + 2 + 9 + 2) * d` and `34*d + d`; conditioning `d*d`; encoder `2 * Le`
/// blocks + final RmsNorm `d` + memory projection `d*d + d`; formula head
/// `10*d + d + d*d + d + d*d`; decoder embeddings `(5 + 18 + 4 + A + T) * d`
/// with `T = 2 + A + R_max`; per decoder layer one block + RmsNorm `d` +
/// attention `4 * d * d`; heads `d*5 + 5`, `d*18 + 18`, `d*4 + 4 + 19*4`,
/// pointer `2 * d * d + 8 * d + 19 * d + 4 * d`.
pub fn parameter_count(model: &ModelConfig) -> Result<u64> {
    const ITEM: &str = "weights";
    let d = u64::from(model.d_model);
    let a = u64::from(model.max_atoms);
    let r = u64::from(model.max_ring_closures);
    let le = u64::from(model.encoder_blocks);
    let ld = u64::from(model.decoder_blocks);
    // Trace-step embedding rows: START/STOP framing plus atoms and closures.
    let t = checked_add(checked_add(2, a, ITEM)?, r, ITEM)?;

    // Peak embedding: Linear(71 -> d) plus Linear(d -> d), both biased.
    let mut total = checked_add(
        checked_add(checked_mul(71, d, ITEM)?, d, ITEM)?,
        checked_add(checked_mul(d, d, ITEM)?, d, ITEM)?,
        ITEM,
    )?;
    // Metadata embeddings (adduct, polarity, energy count, energy flag) and the Linear(34 -> d).
    total = checked_add(total, checked_mul(3 + 2 + 9 + 2, d, ITEM)?, ITEM)?;
    total = checked_add(
        total,
        checked_add(checked_mul(34, d, ITEM)?, d, ITEM)?,
        ITEM,
    )?;
    // Conditioning projection of the metadata context.
    total = checked_add(total, checked_mul(d, d, ITEM)?, ITEM)?;
    // Encoder: two directions per block, then the final norm and the memory projection.
    total = checked_add(
        total,
        checked_mul(
            checked_mul(2, le, ITEM)?,
            block_parameter_count(&model.encoder)?,
            ITEM,
        )?,
        ITEM,
    )?;
    total = checked_add(total, d, ITEM)?;
    total = checked_add(total, checked_add(checked_mul(d, d, ITEM)?, d, ITEM)?, ITEM)?;
    // Formula head: Linear(10 -> d), Linear(d -> d), then the pool projection.
    total = checked_add(
        total,
        checked_add(checked_mul(10, d, ITEM)?, d, ITEM)?,
        ITEM,
    )?;
    total = checked_add(total, checked_add(checked_mul(d, d, ITEM)?, d, ITEM)?, ITEM)?;
    total = checked_add(total, checked_mul(d, d, ITEM)?, ITEM)?;
    // Decoder token embeddings: kind, type, bond, pointer, step.
    total = checked_add(
        total,
        checked_mul(
            checked_add(
                checked_add(checked_add(checked_add(5, 18, ITEM)?, 4, ITEM)?, a, ITEM)?,
                t,
                ITEM,
            )?,
            d,
            ITEM,
        )?,
        ITEM,
    )?;
    // Per decoder layer: one block, one norm and the Q/K/V/O projections.
    let per_layer = checked_add(
        checked_add(block_parameter_count(&model.decoder)?, d, ITEM)?,
        checked_mul(checked_mul(4, d, ITEM)?, d, ITEM)?,
        ITEM,
    )?;
    total = checked_add(total, checked_mul(ld, per_layer, ITEM)?, ITEM)?;
    // Factor heads: kind, atom type, bond plus the bond-by-type table.
    total = checked_add(total, checked_add(checked_mul(d, 5, ITEM)?, 5, ITEM)?, ITEM)?;
    total = checked_add(
        total,
        checked_add(checked_mul(d, 18, ITEM)?, 18, ITEM)?,
        ITEM,
    )?;
    total = checked_add(
        total,
        checked_add(
            checked_add(checked_mul(d, 4, ITEM)?, 4, ITEM)?,
            checked_mul(19, 4, ITEM)?,
            ITEM,
        )?,
        ITEM,
    )?;
    // Pointer head: key and query projections plus the residual/type/bond tables.
    total = checked_add(
        total,
        checked_add(
            checked_add(
                checked_add(
                    checked_mul(checked_mul(2, d, ITEM)?, d, ITEM)?,
                    checked_mul(8, d, ITEM)?,
                    ITEM,
                )?,
                checked_mul(19, d, ITEM)?,
                ITEM,
            )?,
            checked_mul(4, d, ITEM)?,
            ITEM,
        )?,
        ITEM,
    )?;
    Ok(total)
}
