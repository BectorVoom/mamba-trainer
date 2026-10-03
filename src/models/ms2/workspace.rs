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

use crate::backend::{Device, memory_snapshot, reserved_bytes, supports_dtype};
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
    /// The model dtype must be supported, and `max_bindings` must cover the
    /// [`MS2_MAX_KERNEL_ARRAYS`] array bindings plus the metadata binding. A
    /// failure is [`Error::Unsupported`] naming the capability.
    pub fn check(&self, model: &ModelConfig) -> Result<()> {
        let supported = match model.dtype {
            crate::backend::DType::F32 => self.f32_supported,
            crate::backend::DType::F16 => self.f16_supported,
            crate::backend::DType::BF16 => self.bf16_supported,
        };
        if !supported {
            return Err(Error::Unsupported(format!(
                "Ms2Capabilities::check: backend {} does not support dtype {} \
                 (f32_supported={}, f16_supported={}, bf16_supported={})",
                self.backend,
                model.dtype.name(),
                self.f32_supported,
                self.f16_supported,
                self.bf16_supported,
            )));
        }
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
/// Training additionally carries the autograd-retained tensors the coarse
/// forward items miss, each counted forward-plus-gradient (both live at
/// peak backward): `decoder_mixer_retained` (split pieces, output
/// projection, gate and scan working set per decoder layer; the input
/// projection output itself stays in `decoder_activations`),
/// `decoder_attention_retained` (softmax weights, materialized mask and
/// Q/K/V/context copies per layer; the scores stay in `attention_scores`),
/// `decoder_head_positions_retained` (per teacher-forced position the factor
/// logits, masks, pointer query/keys/scores and log-probabilities, plus the
/// concatenated field rows), `decoder_embed_retained` (token-embedding
/// lookups, adds and the formula broadcast) and `encoder_scan_retained`
/// (split pieces and scan working set per encoder direction and block; the
/// projection outputs stay in `encoder_activations`). The teacher path never
/// materialises the `[rows, 19, A]` pointer-by-type table (it gathers the
/// type row by lookup), so no item counts it. Shape-only views
/// (reshape, permute, slice, unsqueeze, squeeze) are counted as allocating
/// nothing. See P2.2 (`tests/ms2_footprint.rs`).
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

    /// Bytes of the named item, if present.
    pub fn get(&self, name: &str) -> Option<u64> {
        self.items
            .iter()
            .find(|(item, _)| *item == name)
            .map(|(_, bytes)| *bytes)
    }

    /// Estimate for generation with `batch` spectra, `trajectories`
    /// trajectories per spectrum, `formula_rows` resident table rows,
    /// `n_raw` raw peak capacity and `max_steps` trace steps.
    pub fn generation(
        model: &ModelConfig,
        formula_rows: u64,
        batch: u64,
        trajectories: u64,
        n_raw: u64,
        max_steps: u64,
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
        // `formula_table`: formula_rows * (8 + 40) (mass and bound as u32, ten f32 features).
        let formula_table = checked_mul(formula_rows, 8 + 40, "formula_table")?;
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
        // `readout`: equal to `actions` plus batch * trajectories * elem plus batch * 16 * 4.
        let readout = checked_add(
            checked_add(
                actions,
                checked_mul(
                    checked_mul(batch, trajectories, "readout")?,
                    elem,
                    "readout",
                )?,
                "readout",
            )?,
            checked_mul(checked_mul(batch, 16, "readout")?, 4, "readout")?,
            "readout",
        )?;
        // `cache_gather`: per decode-step bytes read for the K/V caches and
        // the atom memory. V0 sampling does no beam gather: no trajectory
        // state is moved between slots, so this is a read volume, not an
        // allocation. K/V: every trajectory attends to its spectrum's
        // per-layer keys and values (`B * K * (1 + N) * d * elem * 2 * Ld`);
        // atom memory: the pointer head reads every trajectory's atom rows
        // (`B * K * A * d * elem`).
        let cache_gather = cache_gather_per_step(batch, trajectories, n, d, ld, a, elem)?;

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
            ],
        })
    }

    /// Estimate for training with `batch` spectra, `targets` target slots per
    /// spectrum, `formula_rows` resident table rows, `n_raw` raw peak capacity
    /// and `max_steps` trace steps.
    ///
    /// Beyond the forward-pass items (shared with generation plus `targets`,
    /// `decoder_activations`, `attention_scores`, `atom_memory` and
    /// `head_scratch`), the retained teacher-path tensors the coarse items
    /// miss are counted forward-plus-gradient in `decoder_mixer_retained`,
    /// `decoder_attention_retained`, `decoder_head_positions_retained`,
    /// `decoder_embed_retained` and `encoder_scan_retained` (see the type
    /// docs for the per-item shapes); `gradients` holds the parameter
    /// gradients, `optimizer_moments` the AdamW state and
    /// `activation_gradients` the gradients of the coarse forward items.
    pub fn training(
        model: &ModelConfig,
        formula_rows: u64,
        batch: u64,
        targets: u64,
        n_raw: u64,
        max_steps: u64,
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
        // `formula_table`: formula_rows * (8 + 40) (mass and bound as u32, ten f32 features).
        let formula_table = checked_mul(formula_rows, 8 + 40, "formula_table")?;
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
        // `targets`: batch * targets * T * (4 + 4 + A) * 4 (tokens and replay).
        let per_position = checked_add(checked_add(4, 4, "targets")?, a, "targets")?;
        let targets_bytes = checked_mul(
            checked_mul(checked_mul(batch, targets, "targets")?, t, "targets")?,
            per_position,
            "targets",
        )?;
        let targets_bytes = checked_mul(targets_bytes, 4, "targets")?;
        // `decoder_activations`: batch * targets * T * (2 * Ld * (d + decoder.in_proj_width())) * elem.
        let per_position_dec = checked_mul(
            checked_mul(2, ld, "decoder_activations")?,
            checked_add(d, w_dec, "decoder_activations")?,
            "decoder_activations",
        )?;
        let decoder_activations = checked_mul(
            checked_mul(
                checked_mul(
                    checked_mul(batch, targets, "decoder_activations")?,
                    t,
                    "decoder_activations",
                )?,
                per_position_dec,
                "decoder_activations",
            )?,
            elem,
            "decoder_activations",
        )?;
        // `attention_scores`: Ld * batch * attention_heads * targets * T * (1 + N) * elem.
        let attention_scores = checked_mul(
            checked_mul(
                checked_mul(
                    checked_mul(
                        checked_mul(
                            checked_mul(ld, batch, "attention_scores")?,
                            attn_heads,
                            "attention_scores",
                        )?,
                        targets,
                        "attention_scores",
                    )?,
                    t,
                    "attention_scores",
                )?,
                checked_add(1, n, "attention_scores")?,
                "attention_scores",
            )?,
            elem,
            "attention_scores",
        )?;
        // `gradients`: equal to `weights`.
        let gradients = weights;
        // `optimizer_moments`: 2 * `weights`.
        let optimizer_moments = checked_mul(2, weights, "optimizer_moments")?;
        // `activation_gradients`: equal to `encoder_activations` + `decoder_activations` + `attention_scores`.
        let activation_gradients = checked_add(
            checked_add(
                encoder_activations,
                decoder_activations,
                "activation_gradients",
            )?,
            attention_scores,
            "activation_gradients",
        )?;

        // `atom_memory` (training): batch * targets * A * d * elem, the creation-state rows
        // gathered once per target after the parallel decoder pass.
        let train_atom_memory = checked_mul(
            checked_mul(
                checked_mul(
                    checked_mul(batch, targets, "atom_memory")?,
                    a,
                    "atom_memory",
                )?,
                d,
                "atom_memory",
            )?,
            elem,
            "atom_memory",
        )?;
        // `head_scratch` (training): batch * targets * T * (5 + 18 + 4 + A + 19A + 4A) * elem,
        // the factor-head logits of every teacher-forced position.
        let head_width = checked_add(
            checked_add(
                checked_add(5 + 18 + 4, a, "head_scratch")?,
                checked_mul(19, a, "head_scratch")?,
                "head_scratch",
            )?,
            checked_mul(4, a, "head_scratch")?,
            "head_scratch",
        )?;
        let train_head_scratch = checked_mul(
            checked_mul(
                checked_mul(
                    checked_mul(batch, targets, "head_scratch")?,
                    max_steps,
                    "head_scratch",
                )?,
                head_width,
                "head_scratch",
            )?,
            elem,
            "head_scratch",
        )?;

        // Retained teacher-path tensors the coarse items above miss, each
        // counted forward-plus-gradient (both live at peak backward, when
        // every retained buffer has its gradient beside it). Shapes follow
        // `Ms2Decoder::run`/`apply_layers`/`attend_cached` (decoder.rs) and
        // the chunked SSD (`ssm/scan.rs`, `ssd_chunked`); widths come from
        // the config (`in_proj_width`, `d_inner`, heads, state), never from
        // measurement. `rows` is B*G teacher rows, `positions` the T-1 scored
        // positions, `mem` the 1+N attended rows, `queries` the G*T queries
        // per spectrum.
        let rows = checked_mul(batch, targets, "decoder_mixer_retained")?;
        let positions = max_steps.saturating_sub(1);
        let mem = checked_add(1, n, "decoder_attention_retained")?;
        let queries = checked_mul(targets, t, "decoder_attention_retained")?;
        let di_dec = model.decoder.d_inner() as u64;
        let h_mix = model.decoder.n_heads as u64;
        let p_dim = model.decoder.head_dim as u64;
        let s_state = model.decoder.d_state as u64;
        let h_e = model.encoder.n_heads as u64;
        let p_e = model.encoder.head_dim as u64;
        let s_e = model.encoder.d_state as u64;
        let le = u64::from(model.encoder_blocks);
        let hd_attn = d
            .checked_div(attn_heads)
            .ok_or_else(|| Error::Config("memory estimate: attention_heads is 0".to_string()))?;
        // Scan chunking matches `ssd_chunked`: the chunk is the config size
        // clamped to the sequence, splitting the sequence into `chunks`.
        let chunk_d = (model.decoder.chunk_size as u64).min(t.max(1)).max(1);
        let chunks_d = t.div_ceil(chunk_d);
        let chunk_e = (model.encoder.chunk_size as u64).min(n.max(1)).max(1);
        let chunks_e = n.div_ceil(chunk_e);

        // `decoder_mixer_retained`, per decoder layer: the split projection
        // pieces (`[rows, T, W]`, tiling the fused output on the fused-split
        // copy path), the output projection and gate (`[rows, T, d]`,
        // `[rows, T, d_inner]`), and the scan working set (intra-chunk band
        // `[rows*chunks*H, C, C]`, intra-chunk output
        // `[rows*chunks*H, C, P]`, chunk-edge states `[rows, chunks, H, P,
        // S]`). The fused input projection output itself stays in
        // `decoder_activations`.
        let mixer_per_layer = checked_add(
            checked_add(
                checked_add(
                    checked_mul(
                        checked_mul(rows, t, "decoder_mixer_retained")?,
                        w_dec,
                        "decoder_mixer_retained",
                    )?,
                    checked_mul(
                        checked_mul(rows, t, "decoder_mixer_retained")?,
                        d,
                        "decoder_mixer_retained",
                    )?,
                    "decoder_mixer_retained",
                )?,
                checked_mul(
                    checked_mul(rows, t, "decoder_mixer_retained")?,
                    di_dec,
                    "decoder_mixer_retained",
                )?,
                "decoder_mixer_retained",
            )?,
            checked_add(
                checked_add(
                    checked_mul(
                        checked_mul(
                            checked_mul(rows, chunks_d, "decoder_mixer_retained")?,
                            h_mix,
                            "decoder_mixer_retained",
                        )?,
                        checked_mul(chunk_d, chunk_d, "decoder_mixer_retained")?,
                        "decoder_mixer_retained",
                    )?,
                    checked_mul(
                        checked_mul(
                            checked_mul(rows, chunks_d, "decoder_mixer_retained")?,
                            h_mix,
                            "decoder_mixer_retained",
                        )?,
                        checked_mul(chunk_d, p_dim, "decoder_mixer_retained")?,
                        "decoder_mixer_retained",
                    )?,
                    "decoder_mixer_retained",
                )?,
                checked_mul(
                    checked_mul(
                        checked_mul(
                            checked_mul(rows, chunks_d, "decoder_mixer_retained")?,
                            h_mix,
                            "decoder_mixer_retained",
                        )?,
                        p_dim,
                        "decoder_mixer_retained",
                    )?,
                    s_state,
                    "decoder_mixer_retained",
                )?,
                "decoder_mixer_retained",
            )?,
            "decoder_mixer_retained",
        )?;
        let decoder_mixer_retained = checked_mul(
            checked_mul(
                checked_mul(ld, mixer_per_layer, "decoder_mixer_retained")?,
                elem,
                "decoder_mixer_retained",
            )?,
            2,
            "decoder_mixer_retained",
        )?;

        // `decoder_attention_retained`, per decoder layer: the softmax
        // weights and the materialized broadcast mask (`[B, h, G*T, 1+N]`
        // each; `elemwise::expand` allocates), the Q/K/V/context copies
        // (`[B, h, G*T, hd]`, `[B, h, hd, 1+N]`, `[B, h, 1+N, hd]`,
        // `[B, h, G*T, hd]`) and the per-layer key/value/output
        // projections (`[B, 1+N, d]` twice, `[B, G*T, d]`). The scores stay
        // in `attention_scores`.
        let attn_per_layer = checked_add(
            checked_add(
                checked_add(
                    checked_mul(
                        checked_mul(
                            checked_mul(
                                checked_mul(batch, attn_heads, "decoder_attention_retained")?,
                                queries,
                                "decoder_attention_retained",
                            )?,
                            mem,
                            "decoder_attention_retained",
                        )?,
                        2,
                        "decoder_attention_retained",
                    )?,
                    checked_add(
                        checked_add(
                            checked_mul(
                                checked_mul(
                                    checked_mul(
                                        checked_mul(
                                            batch,
                                            attn_heads,
                                            "decoder_attention_retained",
                                        )?,
                                        queries,
                                        "decoder_attention_retained",
                                    )?,
                                    hd_attn,
                                    "decoder_attention_retained",
                                )?,
                                2,
                                "decoder_attention_retained",
                            )?,
                            checked_mul(
                                checked_mul(
                                    checked_mul(
                                        checked_mul(
                                            batch,
                                            attn_heads,
                                            "decoder_attention_retained",
                                        )?,
                                        hd_attn,
                                        "decoder_attention_retained",
                                    )?,
                                    mem,
                                    "decoder_attention_retained",
                                )?,
                                2,
                                "decoder_attention_retained",
                            )?,
                            "decoder_attention_retained",
                        )?,
                        checked_mul(
                            checked_mul(batch, mem, "decoder_attention_retained")?,
                            d,
                            "decoder_attention_retained",
                        )?,
                        "decoder_attention_retained",
                    )?,
                    "decoder_attention_retained",
                )?,
                checked_mul(
                    checked_mul(batch, mem, "decoder_attention_retained")?,
                    d,
                    "decoder_attention_retained",
                )?,
                "decoder_attention_retained",
            )?,
            checked_mul(
                checked_mul(batch, queries, "decoder_attention_retained")?,
                d,
                "decoder_attention_retained",
            )?,
            "decoder_attention_retained",
        )?;
        let decoder_attention_retained = checked_mul(
            checked_mul(
                checked_mul(ld, attn_per_layer, "decoder_attention_retained")?,
                elem,
                "decoder_attention_retained",
            )?,
            2,
            "decoder_attention_retained",
        )?;

        // `decoder_head_positions_retained`: every scored position keeps its
        // factor logits, effective-mask chain (six temporaries plus mask,
        // masked logits and log-probabilities per field), pointer query,
        // keys and scores (`[rows, A]`; the `[rows, 19, A]` by-type table is
        // never materialised on the teacher path — the type row is gathered
        // by lookup — so it is counted nowhere), gathered values and the
        // NLL/cat pieces: per position `rows * (298 + 5*d + 3*A*d + 10*A)`
        // floats, plus the concatenated field rows `[rows, T, 5+18+4+A+4]`.
        let head_per_pos = checked_add(
            checked_add(
                298,
                checked_mul(5, d, "decoder_head_positions_retained")?,
                "decoder_head_positions_retained",
            )?,
            checked_add(
                checked_mul(
                    checked_mul(3, a, "decoder_head_positions_retained")?,
                    d,
                    "decoder_head_positions_retained",
                )?,
                checked_mul(10, a, "decoder_head_positions_retained")?,
                "decoder_head_positions_retained",
            )?,
            "decoder_head_positions_retained",
        )?;
        let head_positions = checked_add(
            checked_mul(
                checked_mul(
                    checked_mul(rows, positions, "decoder_head_positions_retained")?,
                    head_per_pos,
                    "decoder_head_positions_retained",
                )?,
                elem,
                "decoder_head_positions_retained",
            )?,
            checked_mul(
                checked_mul(
                    checked_mul(rows, t, "decoder_head_positions_retained")?,
                    checked_add(
                        checked_add(5 + 18 + 4, a, "decoder_head_positions_retained")?,
                        4,
                        "decoder_head_positions_retained",
                    )?,
                    "decoder_head_positions_retained",
                )?,
                elem,
                "decoder_head_positions_retained",
            )?,
            "decoder_head_positions_retained",
        )?;
        let decoder_head_positions_retained =
            checked_mul(head_positions, 2, "decoder_head_positions_retained")?;

        // `decoder_embed_retained`: the five token-field embedding lookups,
        // their five chained adds and the broadcast formula embedding:
        // eleven `[rows, T, d]` buffers (`expand` allocates).
        let decoder_embed_retained = checked_mul(
            checked_mul(
                checked_mul(
                    checked_mul(
                        checked_mul(11, rows, "decoder_embed_retained")?,
                        t,
                        "decoder_embed_retained",
                    )?,
                    d,
                    "decoder_embed_retained",
                )?,
                elem,
                "decoder_embed_retained",
            )?,
            2,
            "decoder_embed_retained",
        )?;

        // `encoder_scan_retained`, per direction and block: the split
        // projection pieces (`[B, N, W]`) and the scan working set (band,
        // intra-chunk output and chunk-edge states, shaped as the decoder's
        // with the encoder widths and the N-length chunking). The
        // projection outputs and ping-pong buffers stay in
        // `encoder_activations`.
        let enc_scan_per_mixer = checked_add(
            checked_mul(
                checked_mul(batch, n, "encoder_scan_retained")?,
                w_enc,
                "encoder_scan_retained",
            )?,
            checked_add(
                checked_add(
                    checked_mul(
                        checked_mul(
                            checked_mul(batch, chunks_e, "encoder_scan_retained")?,
                            h_e,
                            "encoder_scan_retained",
                        )?,
                        checked_mul(chunk_e, chunk_e, "encoder_scan_retained")?,
                        "encoder_scan_retained",
                    )?,
                    checked_mul(
                        checked_mul(
                            checked_mul(batch, chunks_e, "encoder_scan_retained")?,
                            h_e,
                            "encoder_scan_retained",
                        )?,
                        checked_mul(chunk_e, p_e, "encoder_scan_retained")?,
                        "encoder_scan_retained",
                    )?,
                    "encoder_scan_retained",
                )?,
                checked_mul(
                    checked_mul(
                        checked_mul(
                            checked_mul(batch, chunks_e, "encoder_scan_retained")?,
                            h_e,
                            "encoder_scan_retained",
                        )?,
                        p_e,
                        "encoder_scan_retained",
                    )?,
                    s_e,
                    "encoder_scan_retained",
                )?,
                "encoder_scan_retained",
            )?,
            "encoder_scan_retained",
        )?;
        let encoder_scan_retained = checked_mul(
            checked_mul(
                checked_mul(
                    checked_mul(2, le, "encoder_scan_retained")?,
                    enc_scan_per_mixer,
                    "encoder_scan_retained",
                )?,
                elem,
                "encoder_scan_retained",
            )?,
            2,
            "encoder_scan_retained",
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
            ],
        })
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
