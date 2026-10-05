//! MS2 peak selection, peak features, formula search and adapters.
//!
//! The device half of architecture §3.1, §3.2, §3.3, §3.4 and §3.8. Every
//! kernel is scalar (one value per lane) and runs through
//! [`crate::backend::launch_1d_spans`]: one unit walks a contiguous span of
//! lanes, which is what the CPU runtime needs to keep its workers on disjoint
//! cache lines.
//!
//! Binding budget: no kernel takes more than 6 array arguments (architecture
//! §1). Shape mismatches are [`crate::error::Error::Shape`] before any
//! launch, and no operation reads device memory back to the host. Each kernel
//! writes every element of its outputs and never reads an input slot at or
//! beyond `peak_count`, so poisoned padding cannot reach an output. Integer
//! arithmetic is `u32` throughout; index differences that could underflow are
//! taken by ordered subtraction. Host twins with the same arithmetic live in
//! [`crate::models::ms2::twin`].

#![allow(clippy::collapsible_if)]

use cubecl::prelude::*;

use crate::backend::{Device, FloatElem, launch_1d_spans, line_size_for};
use crate::error::{Error, Result};
use crate::models::ms2::contract::request_status;
use crate::tensor::base::Tensor;
use crate::tensor::ops::fused::plane_segments_per_row;
use crate::tensor::ops::index::IdTensor;
use crate::tensor::ops::random::{hash_u32, hash_unit_f32};
use crate::tensor::shape::Shape;
use grammar::{
    add_pointers, apply_token, budget_of, close_pointers, closures, is_legal, kind_mask, n_atoms,
    replay_masks, resid_of, root_types, step_of, type_fits, used_of,
};

/// Per-spectrum `u32` metadata width: `peak_count`, `precursor`,
/// `precursor_uncertainty`, `adduct`, `fragment_tolerance`,
/// `precursor_tolerance`, `id_lo`, `id_hi`.
pub const META_WIDTH: usize = 8;
/// Peak feature width of [`peak_features`] (architecture §3.2).
pub const PEAK_FEATURES: usize = 71;
/// Metadata feature width of [`meta_features`] (architecture §4.1).
pub const META_FEATURES: usize = 34;
/// Frozen integer Fourier wavelengths in micro-dalton units
/// (`round(10^(4 + 5.3 k / 15))`, architecture §3.2).
pub const MS2_WAVELENGTHS: [u32; 16] = [
    10000, 22560, 50894, 114815, 259020, 584341, 1318257, 2973948, 6709137, 15135612, 34145489,
    77031201, 173780083, 392042666, 884436519, 1995262315,
];

/// Whether two kernel arguments share the same device allocation.
///
/// Compares the debug rendering of the underlying managed-memory binding,
/// which names the allocation id: two clones of one buffer share it, two
/// live distinct buffers do not. Used to refuse input/output aliasing where
/// a kernel reads one buffer while writing the other in the same launch.
/// Buffers of different lengths are different allocations, which settles
/// most pairs of a decoding step without rendering anything.
fn shares_storage<R: Runtime>(a: &ArrayArg<R>, b: &ArrayArg<R>) -> bool {
    match (a, b) {
        (ArrayArg::Handle { handle: ha }, ArrayArg::Handle { handle: hb }) => {
            ha.handle.size() == hb.handle.size()
                && format!("{:?}", ha.handle.memory) == format!("{:?}", hb.handle.memory)
        }
        _ => false,
    }
}
/// Resident device copy of the frozen Fourier wavelengths
/// ([`MS2_WAVELENGTHS`]) and the V0 atom-type table: uploaded once per device
/// and reused by [`peak_features`], [`meta_features`] and [`grammar_replay`],
/// so a warmed encode or teacher pass performs no upload. The caller keeps
/// this struct alongside the model.
pub struct Ms2Constants<R: Runtime> {
    /// The 16 wavelengths as one `[16]` id buffer.
    pub waves: IdTensor<R>,
    /// The 18 atom-type rows as one `[18, 3]` id buffer holding
    /// `(element, hydrogens, valence)` in [`crate::models::ms2::chem`]
    /// order; row 0 is unused (all zeros).
    pub atom_table: IdTensor<R>,
}

impl<R: Runtime> Ms2Constants<R> {
    /// Upload the wavelength and atom-type tables once. Exactly 2 uploads, no
    /// launch.
    pub fn new(device: &Device<R>) -> Self {
        let table = MS2_WAVELENGTHS;
        let waves = IdTensor::from_slice(&table, vec![MS2_WAVELENGTHS.len()], device)
            .expect("a 16-element upload fits");
        let mut types = vec![0u32; 18 * 3];
        for (i, t) in crate::models::ms2::chem::ATOM_TYPES.iter().enumerate() {
            types[(i + 1) * 3] = t.element as u32;
            types[(i + 1) * 3 + 1] = u32::from(t.hydrogens);
            types[(i + 1) * 3 + 2] = u32::from(t.valence);
        }
        let atom_table =
            IdTensor::from_slice(&types, vec![18, 3], device).expect("a 54-element upload fits");
        Self { waves, atom_table }
    }
}

/// Peaks above `precursor + 2 Da` are ineligible (architecture §3.1).
pub const PRECURSOR_MARGIN: u32 = 2_000_000;
/// Relative-intensity floor of the kept set (architecture §3.1).
pub const INTENSITY_FLOOR: f32 = 1e-3;

/// Largest transformed intensity treated as finite. Eligibility is the range test
/// `0 < t < FINITE_MAX` because fast-math backends (Metal) fold `t - t == 0`.
///
/// `FINITE_MAX` (3e38) is the ONE shared bound of the validated score domain:
/// a ranking term or score is "in the validated domain" when strictly inside
/// (−3e38, 3e38); anything else (NaN, infinities, finite extremes at or
/// beyond it, overflowing sums) is treated as invalid and excluded from
/// ranking. This rule is by specification — finiteness tests are not portable
/// across shader backends — and [`crate::models::ms2::pack::SCORE_FINITE_MAX`]
/// and [`crate::models::ms2::allocate::ALLOC_FINITE_MAX`] are aliases of it.
pub const FINITE_MAX: f32 = 3.0e38;

/// `2π` in `f32`, passed to the feature kernel as a scalar so the host twin
/// and the kernel multiply by the same rounded value.
const TWO_PI_F32: f32 = 2.0 * core::f32::consts::PI;

/// Preallocated outputs of peak selection for one `(B, n_raw, n_keep)` bucket.
pub struct PeakBuffers<R: Runtime, E: FloatElem> {
    /// `[B, 3]`: max, total, retained relative intensity.
    pub stats: Tensor<R, E>,
    /// `[B, n_raw]`: intensity rank among kept peaks, `u32::MAX` when not kept.
    pub rank: IdTensor<R>,
    /// `[B, n_raw]`: m/z order among selected peaks, `u32::MAX` when not selected.
    pub position: IdTensor<R>,
    /// `[B, n_keep, 3]`: raw index, m/z, reverse index.
    pub kept: IdTensor<R>,
    /// `[B, n_keep, 2]`: relative intensity, valid flag.
    pub kept_f: Tensor<R, E>,
    /// `[B, 2]`: kept count and status bits (`1 << 19` when truncated).
    pub summary: IdTensor<R>,
}

impl<R: Runtime, E: FloatElem> PeakBuffers<R, E> {
    /// Allocate the six outputs of [`peak_select`] uninitialised: every kernel
    /// writes every element, so there is nothing to initialise.
    pub fn new(batch: usize, n_raw: usize, n_keep: usize, device: &Device<R>) -> Self {
        Self {
            stats: Tensor::empty(vec![batch, 3], device),
            rank: IdTensor::empty(vec![batch, n_raw], device),
            position: IdTensor::empty(vec![batch, n_raw], device),
            kept: IdTensor::empty(vec![batch, n_keep, 3], device),
            kept_f: Tensor::empty(vec![batch, n_keep, 2], device),
            summary: IdTensor::empty(vec![batch, 2], device),
        }
    }

    /// Allocate the outputs filled with poison (NaN floats, `0xDEAD_BEEF`
    /// ids), so a kernel that skips an element is caught by the comparison
    /// with the host twin. Test support only.
    pub fn poisoned(batch: usize, n_raw: usize, n_keep: usize, device: &Device<R>) -> Result<Self> {
        let poison_f =
            |len: usize| Tensor::<R, E>::from_f32(&vec![f32::NAN; len], vec![len], device);
        let poison_u =
            |len: usize| IdTensor::from_slice(&vec![0xDEAD_BEEF; len], vec![len], device);
        Ok(Self {
            stats: poison_f(batch * 3)?.reshape(vec![batch, 3])?,
            rank: poison_u(batch * n_raw)?.reshape(vec![batch, n_raw])?,
            position: poison_u(batch * n_raw)?.reshape(vec![batch, n_raw])?,
            kept: poison_u(batch * n_keep * 3)?.reshape(vec![batch, n_keep, 3])?,
            kept_f: poison_f(batch * n_keep * 2)?.reshape(vec![batch, n_keep, 2])?,
            summary: poison_u(batch * 2)?.reshape(vec![batch, 2])?,
        })
    }
}

/// Lane per spectrum: largest eligible transformed intensity (`max`, 0 when
/// none) and the index-ordered sum of `t / max` over eligible peaks above the
/// floor (`total`, 0 when `max == 0`). Writes `stats[b, 2] = 0`, which the
/// summary kernel overwrites. Arrays: `mz`, `intensity`, `meta`, `stats`.
// Finiteness is a range test, `0 < t < FINITE_MAX`: Metal compiles WGSL with
// fast-math, which folds `t - t == 0` and `t != t` to constants, so an
// overflowed square would otherwise pass as eligible (seen on the M1).
#[cube(launch_unchecked)]
fn ms2_peak_stats_kernel<F: Float + CubeElement>(
    mz: &Array<u32>,
    intensity: &Array<F>,
    meta: &Array<u32>,
    stats: &mut Array<F>,
    n_raw: usize,
    intensity_scale: u32,
    margin: u32,
    floor: F,
    lanes: usize,
    span: usize,
) {
    let zero = F::new(0.0_f32);
    let finite_max = F::new(FINITE_MAX);
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    for pos in start..end {
        // Clamp to `n_raw`: invalid metadata must never index past the inputs.
        let mut count = meta[pos * 8] as usize;
        if count > n_raw {
            count = n_raw;
        }
        let c = meta[pos * 8 + 1];
        let mut max = zero;
        for i in 0..count {
            let m = mz[pos * n_raw + i];
            let x = intensity[pos * n_raw + i];
            let mut t = x;
            if intensity_scale == 1u32 {
                t = x * x;
            }
            // Eligibility (architecture §3.1): non-zero m/z within the
            // precursor bound, and a finite positive transformed intensity.
            // Sequential single-statement selections: the bound subtraction
            // stays behind its own comparison so no sum overflows.
            let mut m_nonzero = false;
            if m != 0u32 {
                m_nonzero = true;
            }
            let mut bound_ok = false;
            if m <= c {
                bound_ok = true;
            }
            if m > c {
                bound_ok = m - c <= margin;
            }
            let mut t_ok = false;
            if t > zero {
                if t < finite_max {
                    t_ok = true;
                }
            }
            let mut eligible = false;
            if m_nonzero && bound_ok && t_ok {
                eligible = true;
            }
            if eligible && t > max {
                max = t;
            }
        }
        let threshold = floor * max;
        let mut total = zero;
        if max > zero {
            for i in 0..count {
                let m = mz[pos * n_raw + i];
                let x = intensity[pos * n_raw + i];
                let mut t = x;
                if intensity_scale == 1u32 {
                    t = x * x;
                }
                let mut m_nonzero = false;
                if m != 0u32 {
                    m_nonzero = true;
                }
                let mut bound_ok = false;
                if m <= c {
                    bound_ok = true;
                }
                if m > c {
                    bound_ok = m - c <= margin;
                }
                let mut t_ok = false;
                if t > zero {
                    if t < finite_max {
                        t_ok = true;
                    }
                }
                let mut eligible = false;
                if m_nonzero && bound_ok && t_ok {
                    eligible = true;
                }
                if eligible && t >= threshold {
                    total += t / max;
                }
            }
        }
        stats[pos * 3] = max;
        stats[pos * 3 + 1] = total;
        stats[pos * 3 + 2] = zero;
    }
}

/// Lane per raw peak: rank among the kept peaks of its spectrum by decreasing
/// transformed intensity, ties by smaller index; `sentinel` when not kept
/// (padding lanes included). Reads `mz`, `intensity`, `meta`, `stats`.
// Finiteness is a range test, `0 < t < FINITE_MAX`: Metal compiles WGSL with
// fast-math, which folds `t - t == 0` and `t != t` to constants, so an
// overflowed square would otherwise pass as eligible (seen on the M1).
#[cube(launch_unchecked)]
fn ms2_peak_rank_kernel<F: Float + CubeElement>(
    mz: &Array<u32>,
    intensity: &Array<F>,
    meta: &Array<u32>,
    stats: &Array<F>,
    rank: &mut Array<u32>,
    n_raw: usize,
    intensity_scale: u32,
    margin: u32,
    floor: F,
    sentinel: u32,
    lanes: usize,
    span: usize,
) {
    let zero = F::new(0.0_f32);
    let finite_max = F::new(FINITE_MAX);
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    for pos in start..end {
        let b = pos / n_raw;
        let i = pos % n_raw;
        // Clamp to `n_raw`: invalid metadata must never index past the inputs.
        let mut count = meta[b * 8] as usize;
        if count > n_raw {
            count = n_raw;
        }
        let c = meta[b * 8 + 1];
        let max_v = stats[b * 3];
        let threshold = floor * max_v;
        let mut r = sentinel;
        if i < count {
            let m = mz[b * n_raw + i];
            let x = intensity[b * n_raw + i];
            let mut t = x;
            if intensity_scale == 1u32 {
                t = x * x;
            }
            let mut m_nonzero = false;
            if m != 0u32 {
                m_nonzero = true;
            }
            let mut bound_ok = false;
            if m <= c {
                bound_ok = true;
            }
            if m > c {
                bound_ok = m - c <= margin;
            }
            let mut t_ok = false;
            if t > zero {
                if t < finite_max {
                    t_ok = true;
                }
            }
            let mut max_pos = false;
            if max_v > zero {
                max_pos = true;
            }
            let mut above_floor = false;
            if t >= threshold {
                above_floor = true;
            }
            let mut keep = false;
            if m_nonzero && bound_ok && t_ok && max_pos && above_floor {
                keep = true;
            }
            if keep {
                let mut cnt = 0u32;
                for j in 0..count {
                    let mj = mz[b * n_raw + j];
                    let xj = intensity[b * n_raw + j];
                    let mut tj = xj;
                    if intensity_scale == 1u32 {
                        tj = xj * xj;
                    }
                    let mut mj_nonzero = false;
                    if mj != 0u32 {
                        mj_nonzero = true;
                    }
                    let mut bound_j = false;
                    if mj <= c {
                        bound_j = true;
                    }
                    if mj > c {
                        bound_j = mj - c <= margin;
                    }
                    let mut tok_j = false;
                    if tj > zero {
                        if tj < finite_max {
                            tok_j = true;
                        }
                    }
                    let mut floor_j = false;
                    if tj >= threshold {
                        floor_j = true;
                    }
                    let mut kept_j = false;
                    if mj_nonzero && bound_j && tok_j && max_pos && floor_j {
                        kept_j = true;
                    }
                    if kept_j {
                        let mut better = false;
                        if tj > t {
                            better = true;
                        }
                        let mut tie = false;
                        if tj == t {
                            tie = j < i;
                        }
                        if tie {
                            better = true;
                        }
                        if better {
                            cnt += 1u32;
                        }
                    }
                }
                r = cnt;
            }
        }
        rank[pos] = r;
    }
}

/// Lane per raw peak: position among the selected peaks (`rank < n_keep`) by
/// increasing m/z, ties by smaller index; `sentinel` when not selected.
/// Reads only `rank`, `mz` and `meta`.
#[cube(launch_unchecked)]
fn ms2_peak_order_kernel(
    rank: &Array<u32>,
    mz: &Array<u32>,
    meta: &Array<u32>,
    position: &mut Array<u32>,
    n_raw: usize,
    n_keep: usize,
    sentinel: u32,
    lanes: usize,
    span: usize,
) {
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    for pos in start..end {
        let b = pos / n_raw;
        let i = pos % n_raw;
        // Clamp to `n_raw`: invalid metadata must never index past the inputs.
        let mut count = meta[b * 8] as usize;
        if count > n_raw {
            count = n_raw;
        }
        let mut p = sentinel;
        if i < count {
            let r = rank[b * n_raw + i];
            if (r as usize) < n_keep {
                let m = mz[b * n_raw + i];
                let mut cnt = 0u32;
                for j in 0..count {
                    let rj = rank[b * n_raw + j];
                    if (rj as usize) < n_keep {
                        let mj = mz[b * n_raw + j];
                        let mut hit = false;
                        if mj < m {
                            hit = true;
                        }
                        let mut tie = false;
                        if mj == m {
                            tie = j < i;
                        }
                        if tie {
                            hit = true;
                        }
                        if hit {
                            cnt += 1u32;
                        }
                    }
                }
                p = cnt;
            }
        }
        position[pos] = p;
    }
}

/// Lane per kept slot: the raw index with `position == p`, the selected count
/// `len`, and the relative intensity. The spectrum maximum is recomputed from
/// `intensity` and `meta` with the same arithmetic as the stats kernel, so
/// this kernel needs no `stats` read and stays within the binding budget;
/// lanes with `position != sentinel` are exactly the valid slots, so no slot
/// at or beyond `peak_count` is ever read. Arrays: `position`, `mz`,
/// `intensity`, `meta`, `kept`, `kept_f`.
// Finiteness is a range test, `0 < t < FINITE_MAX`: Metal compiles WGSL with
// fast-math, which folds `t - t == 0` and `t != t` to constants, so an
// overflowed square would otherwise pass as eligible (seen on the M1).
#[cube(launch_unchecked)]
fn ms2_peak_gather_kernel<F: Float + CubeElement>(
    position: &Array<u32>,
    mz: &Array<u32>,
    intensity: &Array<F>,
    meta: &Array<u32>,
    kept: &mut Array<u32>,
    kept_f: &mut Array<F>,
    n_raw: usize,
    n_keep: usize,
    intensity_scale: u32,
    margin: u32,
    sentinel: u32,
    lanes: usize,
    span: usize,
) {
    let zero = F::new(0.0_f32);
    let finite_max = F::new(FINITE_MAX);
    let one = F::new(1.0_f32);
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    for pos in start..end {
        let b = pos / n_keep;
        let p = pos % n_keep;
        // Clamp to `n_raw`: invalid metadata must never index past the inputs.
        let mut count = meta[b * 8] as usize;
        if count > n_raw {
            count = n_raw;
        }
        let c = meta[b * 8 + 1];
        let mut max = zero;
        for i in 0..count {
            let m = mz[b * n_raw + i];
            let x = intensity[b * n_raw + i];
            let mut t = x;
            if intensity_scale == 1u32 {
                t = x * x;
            }
            let mut m_nonzero = false;
            if m != 0u32 {
                m_nonzero = true;
            }
            let mut bound_ok = false;
            if m <= c {
                bound_ok = true;
            }
            if m > c {
                bound_ok = m - c <= margin;
            }
            let mut t_ok = false;
            if t > zero {
                if t < finite_max {
                    t_ok = true;
                }
            }
            let mut eligible = false;
            if m_nonzero && bound_ok && t_ok {
                eligible = true;
            }
            if eligible && t > max {
                max = t;
            }
        }
        // One mutated u32 per pass: `len` counts the selected slots, then a
        // separate pass finds the slot with `position == p` and stops.
        let mut len = 0u32;
        for i in 0..count {
            let q = position[b * n_raw + i];
            if q != sentinel {
                len += 1u32;
            }
        }
        let p32 = p as u32;
        let mut found = sentinel;
        for i in 0..count {
            if position[b * n_raw + i] == p32 {
                found = i as u32;
                break;
            }
        }
        let base = (b * n_keep + p) * 3;
        let base_f = (b * n_keep + p) * 2;
        if p32 < len {
            let fi = found as usize;
            let mf = mz[b * n_raw + fi];
            let xf = intensity[b * n_raw + fi];
            let mut t = xf;
            if intensity_scale == 1u32 {
                t = xf * xf;
            }
            kept[base] = found;
            kept[base + 1] = mf;
            kept[base + 2] = len - 1u32 - p32;
            kept_f[base_f] = t / max;
            kept_f[base_f + 1] = one;
        } else {
            kept[base] = sentinel;
            kept[base + 1] = 0u32;
            kept[base + 2] = p32;
            kept_f[base_f] = zero;
            kept_f[base_f + 1] = zero;
        }
    }
}

/// Lane per spectrum: the selected count, the truncation bit (set when more
/// peaks were kept than `n_keep` holds), and `stats[b, 2]`, the index-ordered
/// sum of selected `t / max` over `total` (0 when `total == 0`). Selected here
/// means `rank < n_keep`, the same set the order kernel positions. The max and
/// total arrive on a read-only view of `stats` while column 2 is written
/// through a second view of the same buffer. Arrays: `rank`, `intensity`,
/// `meta`, `stats_in`, `stats_out`, `summary`.
#[cube(launch_unchecked)]
fn ms2_peak_summary_kernel<F: Float + CubeElement>(
    rank: &Array<u32>,
    intensity: &Array<F>,
    meta: &Array<u32>,
    stats_in: &Array<F>,
    stats_out: &mut Array<F>,
    summary: &mut Array<u32>,
    n_raw: usize,
    n_keep: usize,
    intensity_scale: u32,
    trunc_bit: u32,
    sentinel: u32,
    lanes: usize,
    span: usize,
) {
    let zero = F::new(0.0_f32);
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    for pos in start..end {
        // Clamp to `n_raw`: invalid metadata must never index past the inputs.
        let mut count = meta[pos * 8] as usize;
        if count > n_raw {
            count = n_raw;
        }
        let max_v = stats_in[pos * 3];
        let total_v = stats_in[pos * 3 + 1];
        let mut kept_n = 0u32;
        for i in 0..count {
            let r = rank[pos * n_raw + i];
            if r != sentinel {
                kept_n += 1u32;
            }
        }
        // The kept ranks are a permutation of `0..kept_n`, so the selected
        // count (`rank < n_keep`) is the capped kept count.
        let mut seln = kept_n;
        if (kept_n as usize) > n_keep {
            seln = n_keep as u32;
        }
        let mut sel = zero;
        if total_v > zero {
            for i in 0..count {
                let rs = rank[pos * n_raw + i];
                if (rs as usize) < n_keep {
                    let x = intensity[pos * n_raw + i];
                    let mut t = x;
                    if intensity_scale == 1u32 {
                        t = x * x;
                    }
                    sel += t / max_v;
                }
            }
        }
        let mut bits = 0u32;
        if (kept_n as usize) > n_keep {
            bits = trunc_bit;
        }
        let mut retained = zero;
        if total_v > zero {
            retained = sel / total_v;
        }
        summary[pos * 2] = seln;
        summary[pos * 2 + 1] = bits;
        stats_out[pos * 3 + 2] = retained;
    }
}

/// Check the six shapes of [`peak_select`] before any launch, returning
/// `(batch, n_raw, n_keep)`. Every rank is checked before any dimension is
/// read, so a malformed shape is [`Error::Shape`] rather than a panic.
fn check_peak_select_shapes<R: Runtime, E: FloatElem>(
    mz: &IdTensor<R>,
    intensity: &Tensor<R, E>,
    meta: &IdTensor<R>,
    out: &PeakBuffers<R, E>,
) -> Result<(usize, usize, usize)> {
    if mz.shape().rank() != 2
        || intensity.shape().rank() != 2
        || meta.shape().rank() != 2
        || out.stats.shape().rank() != 2
        || out.rank.shape().rank() != 2
        || out.position.shape().rank() != 2
        || out.kept.shape().rank() != 3
        || out.kept_f.shape().rank() != 3
        || out.summary.shape().rank() != 2
    {
        return Err(Error::shape(format!(
            "peak_select needs mz [B, n_raw], intensity [B, n_raw], meta [B, 8], stats [B, 3], rank/position [B, n_raw], kept [B, N, 3], kept_f [B, N, 2] and summary [B, 2], got {} and {} and {} and {} and {} and {} and {} and {} and {}",
            mz.shape(),
            intensity.shape(),
            meta.shape(),
            out.stats.shape(),
            out.rank.shape(),
            out.position.shape(),
            out.kept.shape(),
            out.kept_f.shape(),
            out.summary.shape()
        )));
    }
    let batch = mz.shape().dim(0);
    let n_raw = mz.shape().dim(1);
    let n_keep = out.kept.shape().dim(1);
    let kept_last = out.kept.shape().dim(2);
    let kept_f_last = out.kept_f.shape().dim(2);
    // Expected shapes bound once: comparing against `&[...]` literals inline
    // trips `op_ref`.
    let want_input: &[usize] = &[batch, n_raw];
    let want_meta: &[usize] = &[batch, META_WIDTH];
    if intensity.shape().dims() != want_input || meta.shape().dims() != want_meta {
        return Err(Error::shape(format!(
            "peak_select needs intensity [{batch}, {n_raw}] and meta [{batch}, 8], got {} and {}",
            intensity.shape(),
            meta.shape()
        )));
    }
    if kept_last != 3 || kept_f_last != 2 {
        return Err(Error::shape(format!(
            "peak_select needs kept [B, N, 3] and kept_f [B, N, 2], got last dimensions {kept_last} and {kept_f_last}"
        )));
    }
    let want_stats: &[usize] = &[batch, 3];
    let want_raw: &[usize] = &[batch, n_raw];
    let want_kept: &[usize] = &[batch, n_keep, 3];
    let want_kept_f: &[usize] = &[batch, n_keep, 2];
    let want_summary: &[usize] = &[batch, 2];
    if out.stats.shape().dims() != want_stats
        || out.rank.shape().dims() != want_raw
        || out.position.shape().dims() != want_raw
        || out.kept.shape().dims() != want_kept
        || out.kept_f.shape().dims() != want_kept_f
        || out.summary.shape().dims() != want_summary
    {
        return Err(Error::shape(format!(
            "peak_select needs stats [{batch}, 3], rank/position [{batch}, {n_raw}], kept [{batch}, {n_keep}, 3], kept_f [{batch}, {n_keep}, 2] and summary [{batch}, 2], got {} and {} and {} and {} and {} and {}",
            out.stats.shape(),
            out.rank.shape(),
            out.position.shape(),
            out.kept.shape(),
            out.kept_f.shape(),
            out.summary.shape()
        )));
    }
    Ok((batch, n_raw, n_keep))
}

/// Run peak selection (architecture §3.1) into the preallocated `out`.
///
/// `mz` is `[B, n_raw]` m/z in micro-dalton units, `intensity` the
/// untransformed intensities in the scale `intensity_scale` names (0 linear,
/// 1 square root of relative intensity), `meta` the `[B, 8]` per-spectrum
/// metadata. Exactly 5 launches: stats, rank, order, gather, summary.
pub fn peak_select<R: Runtime, E: FloatElem>(
    mz: &IdTensor<R>,
    intensity: &Tensor<R, E>,
    meta: &IdTensor<R>,
    intensity_scale: u32,
    out: &PeakBuffers<R, E>,
) -> Result<()> {
    let (batch, n_raw, n_keep) = check_peak_select_shapes(mz, intensity, meta, out)?;
    if batch == 0 {
        return Ok(());
    }
    let client = mz.client();
    let floor = E::from_scalar(INTENSITY_FLOOR);
    let sentinel = u32::MAX;
    let (count, dim, span) = launch_1d_spans(client, batch, n_raw.max(1));
    unsafe {
        ms2_peak_stats_kernel::launch_unchecked::<E, R>(
            client,
            count,
            dim,
            mz.arg(),
            intensity.arg(),
            meta.arg(),
            out.stats.arg(),
            n_raw,
            intensity_scale,
            PRECURSOR_MARGIN,
            floor,
            batch,
            span,
        );
    }
    let lanes = batch * n_raw;
    if lanes > 0 {
        let (count, dim, span) = launch_1d_spans(client, lanes, n_raw.max(1));
        unsafe {
            ms2_peak_rank_kernel::launch_unchecked::<E, R>(
                client,
                count,
                dim,
                mz.arg(),
                intensity.arg(),
                meta.arg(),
                out.stats.arg(),
                out.rank.arg(),
                n_raw,
                intensity_scale,
                PRECURSOR_MARGIN,
                floor,
                sentinel,
                lanes,
                span,
            );
        }
        let (count, dim, span) = launch_1d_spans(client, lanes, n_raw.max(1));
        unsafe {
            ms2_peak_order_kernel::launch_unchecked::<R>(
                client,
                count,
                dim,
                out.rank.arg(),
                mz.arg(),
                meta.arg(),
                out.position.arg(),
                n_raw,
                n_keep,
                sentinel,
                lanes,
                span,
            );
        }
    }
    let slots = batch * n_keep;
    if slots > 0 {
        let (count, dim, span) = launch_1d_spans(client, slots, n_raw.max(1));
        unsafe {
            ms2_peak_gather_kernel::launch_unchecked::<E, R>(
                client,
                count,
                dim,
                out.position.arg(),
                mz.arg(),
                intensity.arg(),
                meta.arg(),
                out.kept.arg(),
                out.kept_f.arg(),
                n_raw,
                n_keep,
                intensity_scale,
                PRECURSOR_MARGIN,
                sentinel,
                slots,
                span,
            );
        }
    }
    let (count, dim, span) = launch_1d_spans(client, batch, n_raw.max(1));
    unsafe {
        ms2_peak_summary_kernel::launch_unchecked::<E, R>(
            client,
            count,
            dim,
            out.rank.arg(),
            intensity.arg(),
            meta.arg(),
            out.stats.arg(),
            out.stats.arg(),
            out.summary.arg(),
            n_raw,
            n_keep,
            intensity_scale,
            1u32 << 19,
            sentinel,
            batch,
            span,
        );
    }
    Ok(())
}

/// Lane per `(b, p)`: the 71 peak features of architecture §3.2, exact zeros
/// in padding slots. `m` is the kept m/z, `c` the precursor, `r` the relative
/// intensity; the wavelengths arrive as one `[16]` array argument. Arrays:
/// `kept`, `kept_f`, `meta`, `waves`, `out`.
#[cube(launch_unchecked)]
fn ms2_peak_features_kernel<F: Float + CubeElement>(
    kept: &Array<u32>,
    kept_f: &Array<F>,
    meta: &Array<u32>,
    waves: &Array<u32>,
    out: &mut Array<F>,
    n_keep: usize,
    two_pi: F,
    ln101: F,
    sentinel: u32,
    lanes: usize,
    span: usize,
) {
    let zero = F::new(0.0_f32);
    let one = F::new(1.0_f32);
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    for pos in start..end {
        let b = pos / n_keep;
        let p = pos % n_keep;
        let base = (b * n_keep + p) * 3;
        let base_f = (b * n_keep + p) * 2;
        let base_o = pos * 71;
        let raw = kept[base];
        if raw == sentinel {
            for f in 0..71 {
                out[base_o + f] = zero;
            }
        } else {
            let m = kept[base + 1];
            let r = kept_f[base_f];
            let c = meta[b * 8 + 1];
            let mf = F::cast_from(m);
            let cf = F::cast_from(c);
            out[base_o] = mf * F::new(1e-9_f32);
            out[base_o + 1] = (cf - mf) * F::new(1e-9_f32);
            out[base_o + 2] = mf / cf;
            out[base_o + 3] = r;
            out[base_o + 4] = r.sqrt();
            out[base_o + 5] = (one + F::new(100.0_f32) * r).ln() / ln101;
            let mut gap = zero;
            if p > 0 {
                let prev = kept[base - 3 + 1];
                gap = F::cast_from(m - prev) * F::new(1e-6_f32);
            }
            out[base_o + 6] = (one + gap).ln();
            for k in 0..16 {
                let w = waves[k];
                let phase = two_pi * F::cast_from(m % w) / F::cast_from(w);
                out[base_o + 7 + 2 * k] = phase.sin();
                out[base_o + 8 + 2 * k] = phase.cos();
            }
            // The difference by ordered subtraction: each subtraction runs
            // inside the branch that makes it non-negative, so there is no
            // wrapping intermediate even before the correction.
            let mut v = 0u32;
            if m > c {
                v = m - c;
            }
            if m <= c {
                v = c - m;
            }
            for k in 0..16 {
                let w = waves[k];
                let phase = two_pi * F::cast_from(v % w) / F::cast_from(w);
                out[base_o + 39 + 2 * k] = phase.sin();
                out[base_o + 40 + 2 * k] = phase.cos();
            }
        }
    }
}

/// Write the 71 peak features of architecture §3.2 into `out` (`[B, n_keep,
/// 71]`) from the selection outputs `kept` (`[B, n_keep, 3]`), `kept_f`
/// (`[B, n_keep, 2]`) and `meta` (`[B, 8]`), with the wavelengths from the
/// resident [`Ms2Constants`]. Exactly 1 launch and no upload.
pub fn peak_features<R: Runtime, E: FloatElem>(
    kept: &IdTensor<R>,
    kept_f: &Tensor<R, E>,
    meta: &IdTensor<R>,
    waves: &Ms2Constants<R>,
    out: &Tensor<R, E>,
) -> Result<()> {
    // Every rank is checked before any dimension is read, so a malformed
    // shape is `Error::Shape` rather than a panic; the last dimensions ride
    // along in the full-shape comparison below.
    if kept.shape().rank() != 3
        || kept_f.shape().rank() != 3
        || meta.shape().rank() != 2
        || out.shape().rank() != 3
        || waves.waves.len() != MS2_WAVELENGTHS.len()
    {
        return Err(Error::shape(format!(
            "peak_features needs kept [B, N, 3], kept_f [B, N, 2], meta [B, 8], 16 resident wavelengths and out [B, N, 71], got {} and {} and {} and {} wavelengths and {}",
            kept.shape(),
            kept_f.shape(),
            meta.shape(),
            waves.waves.len(),
            out.shape()
        )));
    }
    let batch = kept.shape().dim(0);
    let n_keep = kept.shape().dim(1);
    let want_kept: &[usize] = &[batch, n_keep, 3];
    let want_kept_f: &[usize] = &[batch, n_keep, 2];
    let want_meta: &[usize] = &[batch, META_WIDTH];
    let want_out: &[usize] = &[batch, n_keep, PEAK_FEATURES];
    if kept.shape().dims() != want_kept
        || kept_f.shape().dims() != want_kept_f
        || meta.shape().dims() != want_meta
        || out.shape().dims() != want_out
    {
        return Err(Error::shape(format!(
            "peak_features needs kept [B, N, 3], kept_f [B, N, 2], meta [B, 8] and out [B, N, 71], got {} and {} and {} and {}",
            kept.shape(),
            kept_f.shape(),
            meta.shape(),
            out.shape()
        )));
    }
    if out.is_empty() {
        return Ok(());
    }
    let client = out.client();
    let lanes = batch * n_keep;
    let (count, dim, span) = launch_1d_spans(client, lanes, PEAK_FEATURES);
    unsafe {
        ms2_peak_features_kernel::launch_unchecked::<E, R>(
            client,
            count,
            dim,
            kept.arg(),
            kept_f.arg(),
            meta.arg(),
            waves.waves.arg(),
            out.arg(),
            n_keep,
            E::from_scalar(TWO_PI_F32),
            E::from_scalar(101.0f32.ln()),
            u32::MAX,
            lanes,
            span,
        );
    }
    Ok(())
}

/// Lane per element: `x` where `valid != 0`, exact `+0.0` elsewhere, so a NaN
/// in an unselected slot cannot reach an output.
#[cube(launch_unchecked)]
fn ms2_select_valid_kernel<F: Float + CubeElement>(
    x: &Array<F>,
    valid: &Array<F>,
    out: &mut Array<F>,
    d: usize,
    lanes: usize,
    span: usize,
) {
    let zero = F::new(0.0_f32);
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    for pos in start..end {
        if valid[pos / d] != zero {
            out[pos] = x[pos];
        } else {
            out[pos] = zero;
        }
    }
}

/// Keep `x` (`[.., n, d]`) where `valid` (`[.., n]`) is non-zero, exact zero
/// elsewhere. One launch; the adjoint is the same selection of the gradient.
pub fn select_valid<R: Runtime, E: FloatElem>(
    x: &Tensor<R, E>,
    valid: &Tensor<R, E>,
) -> Result<Tensor<R, E>> {
    if x.rank() < 2
        || valid.shape().rank() + 1 != x.rank()
        || x.shape().dims()[..x.rank() - 2] != valid.shape().dims()[..valid.shape().rank() - 1]
        || x.shape().dim(x.rank() - 2) != valid.shape().dim(valid.shape().rank() - 1)
    {
        return Err(Error::shape(format!(
            "select_valid needs x [.., n, d] and valid [.., n], got {} and {}",
            x.shape(),
            valid.shape()
        )));
    }
    let out = Tensor::empty(x.shape().clone(), x.device());
    if out.is_empty() {
        return Ok(out);
    }
    let d = x.shape().dim_from_end(0);
    let lanes = out.len();
    let (count, dim, span) = launch_1d_spans(x.client(), lanes, 1);
    unsafe {
        ms2_select_valid_kernel::launch_unchecked::<E, R>(
            x.client(),
            count,
            dim,
            x.arg(),
            valid.arg(),
            out.arg(),
            d,
            lanes,
            span,
        );
    }
    Ok(out)
}

/// Lane per output element: bit `col` of `bits[row]` as `1.0` or `0.0`.
#[cube(launch_unchecked)]
fn ms2_bits_to_mask_kernel<F: Float + CubeElement>(
    bits: &Array<u32>,
    out: &mut Array<F>,
    width: usize,
    lanes: usize,
    span: usize,
) {
    let zero = F::new(0.0_f32);
    let one = F::new(1.0_f32);
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    for pos in start..end {
        let row = pos / width;
        let col = (pos % width) as u32;
        let mut v = zero;
        if (bits[row] & (1u32 << col)) != 0u32 {
            v = one;
        }
        out[pos] = v;
    }
}

/// Expand `bits` (`[rows]`) into a `[rows, width]` float 0/1 mask holding bit
/// `i` of `bits[row]` at column `i`. One launch; `width <= 32`.
pub fn bits_to_mask<R: Runtime, E: FloatElem>(
    bits: &IdTensor<R>,
    width: usize,
) -> Result<Tensor<R, E>> {
    if bits.shape().rank() != 1 || width > 32 {
        return Err(Error::shape(format!(
            "bits_to_mask needs bits [rows] and width <= 32, got {} and width {width}",
            bits.shape()
        )));
    }
    let rows = bits.len();
    let out = Tensor::empty(Shape::new(vec![rows, width]), bits.device());
    if out.is_empty() {
        return Ok(out);
    }
    let lanes = out.len();
    let (count, dim, span) = launch_1d_spans(bits.client(), lanes, 1);
    unsafe {
        ms2_bits_to_mask_kernel::launch_unchecked::<E, R>(
            bits.client(),
            count,
            dim,
            bits.arg(),
            out.arg(),
            width,
            lanes,
            span,
        );
    }
    Ok(out)
}

/// Lane per output vector: `table[id, col]`, or `0` when `id` is out of
/// range (which includes `u32::MAX`). The load is unconditional — an
/// out-of-range id reads row 0 and stores zero instead — because a lane
/// waits out a load placed under a branch, and a row is moved a vector at a
/// time rather than an element at a time.
#[cube(launch_unchecked)]
fn ms2_lookup_kernel<F: Float + CubeElement, N: Size>(
    table: &Array<Vector<F, N>>,
    ids: &Array<u32>,
    out: &mut Array<Vector<F, N>>,
    table_rows: usize,
    d_lines: usize,
    lanes: usize,
    span: usize,
) {
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    for pos in start..end {
        let row = pos / d_lines;
        let col = pos % d_lines;
        let id = ids[row] as usize;
        let live = id < table_rows;
        let safe = select(live, id, 0usize);
        let value = table[safe * d_lines + col];
        if live {
            out[pos] = value;
        } else {
            out[pos] = Vector::<F, N>::new(F::new(0.0_f32));
        }
    }
}

/// Look up `ids` (`[rows]`) in `table` (`[V, d]`), giving `[rows, d]`; every
/// id at or above `V` (including `u32::MAX`) gives a zero row. One launch,
/// and neither pass reads ids back to the host.
pub fn lookup<R: Runtime, E: FloatElem>(
    table: &Tensor<R, E>,
    ids: &IdTensor<R>,
) -> Result<Tensor<R, E>> {
    if table.shape().rank() != 2 || ids.shape().rank() != 1 {
        return Err(Error::shape(format!(
            "lookup needs table [V, d] and ids [rows], got {} and {}",
            table.shape(),
            ids.shape()
        )));
    }
    let (rows, d) = (ids.len(), table.shape().dim(1));
    let out = Tensor::empty(Shape::new(vec![rows, d]), table.device());
    if out.is_empty() {
        return Ok(out);
    }
    if table.shape().dim(0) == 0 {
        // No row to read, in range or as the stand-in of one out of range.
        crate::tensor::ops::elemwise::fill_(&out, 0.0);
        return Ok(out);
    }
    let line = line_size_for::<R, E>(table.client(), d);
    let lanes = out.len() / line;
    let (count, dim, span) = launch_1d_spans(table.client(), lanes, line);
    unsafe {
        ms2_lookup_kernel::launch_unchecked::<E, R>(
            table.client(),
            count,
            dim,
            line,
            table.arg(),
            ids.arg(),
            out.arg(),
            table.shape().dim(0),
            d / line,
            lanes,
            span,
        );
    }
    Ok(out)
}

/// Rows one lane of [`lookup_backward`] scans on a device with planes. A lane
/// waits on memory once per eight rows, so a lane that scanned every row of a
/// training batch was bound by that wait, however small the table; in groups
/// of this many rows the scan is eight waits and the groups run side by side.
const LOOKUP_BACKWARD_GROUP_ROWS: usize = 64;

/// Lane per `(group, v, col)`: the sum of `grad[r, col]` over the rows `r` of
/// the lane's group (`group_rows` consecutive rows; one group holding every
/// row when the scan is not split) with `ids[r] == v`, in increasing row
/// order. No atomics, no host read of the ids.
#[allow(clippy::too_many_arguments)]
#[cube(launch_unchecked)]
fn ms2_lookup_backward_kernel<F: Float + CubeElement>(
    grad: &Array<F>,
    ids: &Array<u32>,
    out: &mut Array<F>,
    rows: usize,
    d: usize,
    table_elems: usize,
    group_rows: usize,
    lanes: usize,
    span: usize,
) {
    let zero = F::new(0.0_f32);
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    for pos in start..end {
        let within = pos % table_elems;
        let v = (within / d) as u32;
        let col = within % d;
        let first = (pos / table_elems) * group_rows;
        let mut last = first + group_rows;
        if last > rows {
            last = rows;
        }
        let chunks = (last - first) / 8;
        let mut acc = zero;
        // The ids are loaded eight per round and the gradient only for a
        // matching row, added in increasing row order: the sum is the one a
        // row-by-row scan gives, with an eighth of its waits on the id load.
        for c in 0..chunks {
            let r = first + c * 8;
            let i0 = ids[r];
            let i1 = ids[r + 1];
            let i2 = ids[r + 2];
            let i3 = ids[r + 3];
            let i4 = ids[r + 4];
            let i5 = ids[r + 5];
            let i6 = ids[r + 6];
            let i7 = ids[r + 7];
            if i0 == v {
                acc += grad[r * d + col];
            }
            if i1 == v {
                acc += grad[(r + 1) * d + col];
            }
            if i2 == v {
                acc += grad[(r + 2) * d + col];
            }
            if i3 == v {
                acc += grad[(r + 3) * d + col];
            }
            if i4 == v {
                acc += grad[(r + 4) * d + col];
            }
            if i5 == v {
                acc += grad[(r + 5) * d + col];
            }
            if i6 == v {
                acc += grad[(r + 6) * d + col];
            }
            if i7 == v {
                acc += grad[(r + 7) * d + col];
            }
        }
        for r in first + chunks * 8..last {
            if ids[r] == v {
                acc += grad[r * d + col];
            }
        }
        out[pos] = acc;
    }
}

/// Adjoint of [`lookup`] on the device: accumulate `grad` (`[rows, d]`)
/// back into a `[table_rows, d]` table along `ids`.
///
/// One launch on a runtime without planes. On a device with planes and more
/// than [`LOOKUP_BACKWARD_GROUP_ROWS`] rows the scan is split: one launch
/// sums each group of rows into its own partial table and a reduction adds
/// the partials, so the rows of a table element are summed group by group
/// rather than in one chain.
pub fn lookup_backward<R: Runtime, E: FloatElem>(
    grad: &Tensor<R, E>,
    ids: &IdTensor<R>,
    table_rows: usize,
) -> Result<Tensor<R, E>> {
    if grad.shape().rank() != 2 || ids.shape().rank() != 1 || ids.len() != grad.shape().dim(0) {
        return Err(Error::shape(format!(
            "lookup_backward needs grad [rows, d] and ids [rows], got {} and {}",
            grad.shape(),
            ids.shape()
        )));
    }
    let d = grad.shape().dim(1);
    let rows = ids.len();
    let table = Shape::new(vec![table_rows, d]);
    let table_elems = table_rows * d;
    let split = grad.client().properties().hardware.plane_size_max > 1
        && rows > LOOKUP_BACKWARD_GROUP_ROWS;
    let (groups, group_rows) = if split {
        (
            rows.div_ceil(LOOKUP_BACKWARD_GROUP_ROWS),
            LOOKUP_BACKWARD_GROUP_ROWS,
        )
    } else {
        (1, rows)
    };
    let out = Tensor::empty(Shape::new(vec![groups, table_elems]), grad.device());
    if out.is_empty() {
        return out.reshape(table);
    }
    let lanes = out.len();
    let (count, dim, span) = launch_1d_spans(grad.client(), lanes, 1);
    unsafe {
        ms2_lookup_backward_kernel::launch_unchecked::<E, R>(
            grad.client(),
            count,
            dim,
            grad.arg(),
            ids.arg(),
            out.arg(),
            rows,
            d,
            table_elems,
            group_rows,
            lanes,
            span,
        );
    }
    if groups > 1 {
        return crate::tensor::ops::reduce::sum_dim(&out, 0)?.reshape(table);
    }
    out.reshape(table)
}

/// Lane per element: `fallback` where the id is `u32::MAX`, the id itself
/// elsewhere.
#[cube(launch_unchecked)]
fn ms2_safe_ids_kernel(
    ids: &Array<u32>,
    out: &mut Array<u32>,
    fallback: u32,
    sentinel: u32,
    lanes: usize,
    span: usize,
) {
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    for pos in start..end {
        let id = ids[pos];
        if id == sentinel {
            out[pos] = fallback;
        } else {
            out[pos] = id;
        }
    }
}

/// Replace `u32::MAX` ids by `fallback`, keeping the shape, so unused fields
/// can be handed to a gather. One launch.
pub fn safe_ids<R: Runtime>(ids: &IdTensor<R>, fallback: u32) -> Result<IdTensor<R>> {
    let out = IdTensor::empty(ids.shape().clone(), ids.device());
    if out.is_empty() {
        return Ok(out);
    }
    let lanes = out.len();
    let (count, dim, span) = launch_1d_spans(ids.client(), lanes, 1);
    unsafe {
        ms2_safe_ids_kernel::launch_unchecked::<R>(
            ids.client(),
            count,
            dim,
            ids.arg(),
            out.arg(),
            fallback,
            u32::MAX,
            lanes,
            span,
        );
    }
    Ok(out)
}

/// Lane per spectrum: the 34 metadata features of architecture §4.1, one row
/// of `out` per spectrum. `meta` is `[B, 8]`, `energy` is `[B, 2]` holding the
/// collision energy in eV (0 when unknown) and the known flag, `waves` is the
/// 16 frozen wavelengths and `out` is `[B, 34]`. Feature 0 is
/// `min(ce, 400) / 100` when the flag is non-zero and 0 otherwise, feature 1
/// is `f32(precursor) * 1e-9`, and the rest are `sin`/`cos` of
/// `2π f32(precursor % w) / f32(w)`. Arrays: `meta`, `energy`, `waves`, `out`.
#[cube(launch_unchecked)]
fn ms2_meta_features_kernel<F: Float + CubeElement>(
    meta: &Array<u32>,
    energy: &Array<F>,
    waves: &Array<u32>,
    out: &mut Array<F>,
    two_pi: F,
    lanes: usize,
    span: usize,
) {
    let zero = F::new(0.0_f32);
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    for pos in start..end {
        let ce = energy[pos * 2];
        let known = energy[pos * 2 + 1];
        let mut feat0 = zero;
        if known != zero {
            let clip = F::new(400.0_f32);
            let mut capped = clip;
            if ce < clip {
                capped = ce;
            }
            feat0 = capped / F::new(100.0_f32);
        }
        let precursor = meta[pos * 8 + 1];
        let base = pos * 34;
        out[base] = feat0;
        out[base + 1] = F::cast_from(precursor) * F::new(1e-9_f32);
        for k in 0..16 {
            let w = waves[k];
            let phase = two_pi * F::cast_from(precursor % w) / F::cast_from(w);
            out[base + 2 + 2 * k] = phase.sin();
            out[base + 3 + 2 * k] = phase.cos();
        }
    }
}

/// Write the 34 metadata features of architecture §4.1 into `out` (`[B, 34]`)
/// from `meta` (`[B, 8]`) and `energy` (`[B, 2]`), with the wavelengths from
/// the resident [`Ms2Constants`]. Exactly 1 launch and no upload.
pub fn meta_features<R: Runtime, E: FloatElem>(
    meta: &IdTensor<R>,
    energy: &Tensor<R, E>,
    waves: &Ms2Constants<R>,
    out: &Tensor<R, E>,
) -> Result<()> {
    if meta.shape().rank() != 2
        || energy.shape().rank() != 2
        || out.shape().rank() != 2
        || waves.waves.len() != MS2_WAVELENGTHS.len()
    {
        return Err(Error::shape(format!(
            "meta_features needs meta [B, 8], energy [B, 2], 16 resident wavelengths and out [B, 34], got {} and {} and {} wavelengths and {}",
            meta.shape(),
            energy.shape(),
            waves.waves.len(),
            out.shape()
        )));
    }
    let batch = meta.shape().dim(0);
    let want_meta: &[usize] = &[batch, META_WIDTH];
    let want_energy: &[usize] = &[batch, 2];
    let want_out: &[usize] = &[batch, META_FEATURES];
    if meta.shape().dims() != want_meta
        || energy.shape().dims() != want_energy
        || out.shape().dims() != want_out
    {
        return Err(Error::shape(format!(
            "meta_features needs meta [B, 8], energy [B, 2] and out [B, 34], got {} and {} and {}",
            meta.shape(),
            energy.shape(),
            out.shape()
        )));
    }
    if out.is_empty() {
        return Ok(());
    }
    let client = out.client();
    let (count, dim, span) = launch_1d_spans(client, batch, META_FEATURES);
    unsafe {
        ms2_meta_features_kernel::launch_unchecked::<E, R>(
            client,
            count,
            dim,
            meta.arg(),
            energy.arg(),
            waves.waves.arg(),
            out.arg(),
            E::from_scalar(TWO_PI_F32),
            batch,
            span,
        );
    }
    Ok(())
}

/// Lane per output element: `kept[(b, p), column]`, so one column of the
/// `[B, N, 3]` selection output as a flat `[B * N]` id vector.
#[cube(launch_unchecked)]
fn ms2_kept_column_kernel(
    kept: &Array<u32>,
    out: &mut Array<u32>,
    column: usize,
    lanes: usize,
    span: usize,
) {
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    for pos in start..end {
        out[pos] = kept[pos * 3 + column];
    }
}

/// One column of `kept` (`[B, N, 3]`) as a flat `[B * N]` id vector, for the
/// per-spectrum `reverse` ids that [`Var::gather_tokens`] runs on. One launch.
pub fn kept_column<R: Runtime>(kept: &IdTensor<R>, column: usize) -> Result<IdTensor<R>> {
    if kept.shape().rank() != 3 || kept.shape().dim(2) != 3 || column >= 3 {
        return Err(Error::shape(format!(
            "kept_column needs kept [B, N, 3] and column 0..3, got {} and column {column}",
            kept.shape()
        )));
    }
    let (batch, n_keep) = (kept.shape().dim(0), kept.shape().dim(1));
    let out = IdTensor::empty(vec![batch * n_keep], kept.device());
    if out.is_empty() {
        return Ok(out);
    }
    let lanes = out.len();
    let (count, dim, span) = launch_1d_spans(kept.client(), lanes, 1);
    unsafe {
        ms2_kept_column_kernel::launch_unchecked::<R>(
            kept.client(),
            count,
            dim,
            kept.arg(),
            out.arg(),
            column,
            lanes,
            span,
        );
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Formula window, top-F and window mask (architecture §3.3, §3.4, §3.8)
// ---------------------------------------------------------------------------

/// Preallocated outputs of the formula search for one `(B, M, F)` bucket,
/// reused across calls.
pub struct FormulaBuffers<R: Runtime, E: FloatElem> {
    /// `[B, M, 2]`: table row (`u32::MAX` padding), flag (0 none, 1 accept,
    /// 2 ambiguous).
    pub window: IdTensor<R>,
    /// `[B, 5]`: rows_visited, rows_joined, rows_scored, status bits,
    /// complete (0/1).
    pub counters: IdTensor<R>,
    /// `[B, F, 2]`: table row (`u32::MAX` padding), window slot
    /// (`u32::MAX` padding).
    pub top: IdTensor<R>,
    /// `[B, F]` log-probability of each top entry (0 in padding).
    pub top_log_prob: Tensor<R, E>,
    /// `[B]` number of top entries filled.
    pub top_count: IdTensor<R>,
    /// `[B, M, 13]` per scored candidate (V1 §1.2): 10 element counts in
    /// `ELEMENTS` order, integer mass, flag (0 none, 1 accept, 2 ambiguous),
    /// source id (table row; `u32::MAX` for an enumerated candidate or
    /// padding; padding is all `0` except source `u32::MAX`).
    pub cand: IdTensor<R>,
    /// `[B, M, 10]` float `ln(1 + count)` of `cand` (exact `0` in padding),
    /// via [`count_features`].
    pub cand_feat: Tensor<R, E>,
    /// `[B, F, 10]` counts of the retained formulas (`0` in padding), via
    /// [`formula_top_counts`].
    pub top_counts: IdTensor<R>,
}

impl<R: Runtime, E: FloatElem> FormulaBuffers<R, E> {
    /// Allocate the outputs of [`formula_window`], [`formula_gather`],
    /// [`count_features`], [`formula_top`] and [`formula_top_counts`]
    /// uninitialised: every kernel writes every element, so there is nothing
    /// to initialise.
    pub fn new(batch: usize, m: usize, f: usize, device: &Device<R>) -> Self {
        Self {
            window: IdTensor::empty(vec![batch, m, 2], device),
            counters: IdTensor::empty(vec![batch, 5], device),
            top: IdTensor::empty(vec![batch, f, 2], device),
            top_log_prob: Tensor::empty(vec![batch, f], device),
            top_count: IdTensor::empty(vec![batch], device),
            cand: IdTensor::empty(vec![batch, m, 13], device),
            cand_feat: Tensor::empty(vec![batch, m, 10], device),
            top_counts: IdTensor::empty(vec![batch, f, 10], device),
        }
    }

    /// Allocate the outputs filled with poison (NaN floats, `0xDEAD_BEEF`
    /// ids), so a kernel that skips an element is caught by the comparison
    /// with the host twin. Test support only.
    pub fn poisoned(batch: usize, m: usize, f: usize, device: &Device<R>) -> Result<Self> {
        let poison_f =
            |len: usize| Tensor::<R, E>::from_f32(&vec![f32::NAN; len], vec![len], device);
        let poison_u =
            |len: usize| IdTensor::from_slice(&vec![0xDEAD_BEEF; len], vec![len], device);
        Ok(Self {
            window: poison_u(batch * m * 2)?.reshape(vec![batch, m, 2])?,
            counters: poison_u(batch * 5)?.reshape(vec![batch, 5])?,
            top: poison_u(batch * f * 2)?.reshape(vec![batch, f, 2])?,
            top_log_prob: poison_f(batch * f)?.reshape(vec![batch, f])?,
            top_count: poison_u(batch)?.reshape(vec![batch])?,
            cand: poison_u(batch * m * 13)?.reshape(vec![batch, m, 13])?,
            cand_feat: poison_f(batch * m * 10)?.reshape(vec![batch, m, 10])?,
            top_counts: poison_u(batch * f * 10)?.reshape(vec![batch, f, 10])?,
        })
    }
}

/// `a + b`, saturating at `u32::MAX`: the host reference sums in `u64` where
/// no input pair can overflow, and WGSL has no second limb, so the kernel
/// saturates instead. Every saturated sum here feeds a window bound the host
/// saturates the same way.
#[cube]
fn ms2_sat_add(a: u32, b: u32, max_u32: u32) -> u32 {
    // Statement form only: a value-form `if`/`else` does not lower to the
    // backend IR. The plain sum may wrap, but it is overwritten exactly when
    // it would have overflowed.
    let mut out = a + b;
    if a > max_u32 - b {
        out = max_u32;
    }
    out
}

/// `p - w`, flooring at 0: the low end of the superset window.
#[cube]
fn ms2_sat_sub(p: u32, w: u32) -> u32 {
    let mut out = 0u32;
    if p > w {
        out = p - w;
    }
    out
}

/// `floor(mz * t / 10^7)` in 32 bits (contract §5, the host form is
/// [`crate::models::ms2::chem::tolerance_u32`]): exact for `t <= 1000`, the
/// validated range, where `hi * t` and the second numerator fit in `u32`.
#[cube]
fn ms2_tolerance_u32(mz: u32, t: u32) -> u32 {
    let hi = mz / 10_000u32;
    let lo = mz % 10_000u32;
    let q = hi * t;
    q / 1_000u32 + ((q % 1_000u32) * 10_000u32 + lo * t) / 10_000_000u32
}

/// The verdict of contract §5 as a flag (0 reject, 1 accept, 2 ambiguous).
/// The sums run as saturation tests (`r <= tol && E <= tol - r` for accept,
/// `r > tol && r - tol > E` for reject), which decide exactly like the
/// host's `u64` sums with no `u32` sum ever overflowing.
#[cube]
fn ms2_decide(parent: u32, mass: u32, error: u32, tol: u32) -> u32 {
    let mut r = parent - mass;
    if mass > parent {
        r = mass - parent;
    }
    let mut accept = false;
    if r <= tol {
        if error <= tol - r {
            accept = true;
        }
    }
    let mut reject = false;
    if r > tol {
        if r - tol > error {
            reject = true;
        }
    }
    let mut flag = 2u32;
    if accept {
        flag = 1u32;
    }
    if reject {
        flag = 0u32;
    }
    flag
}

/// Lane per spectrum: the precursor window of
/// [`crate::models::ms2::formula::FormulaTable::window`], reproduced step for
/// step, including the two halving searches and the visit counter. The peak
/// count (`meta[b, 0]`) is never read: the search depends only on the
/// precursor. Arrays: `table`, `meta`, `window`, `counters`.
#[allow(unused_assignments)]
#[allow(clippy::too_many_arguments)]
#[cube(launch_unchecked)]
fn ms2_formula_window_kernel(
    table: &Array<u32>,
    meta: &Array<u32>,
    window: &mut Array<u32>,
    counters: &mut Array<u32>,
    rows: usize,
    m: usize,
    max_error: u32,
    rows_visited_max: u32,
    rows_scored_max: u32,
    h_net: u32,
    bit_absent: u32,
    bit_overflow: u32,
    bit_exhausted: u32,
    bit_exact_absent: u32,
    max_u32: u32,
    lanes: usize,
    span: usize,
) {
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    for pos in start..end {
        let b = pos;
        let prec = meta[b * 8 + 1];
        let unc = meta[b * 8 + 2];
        let adduct = meta[b * 8 + 3];
        let ppm = meta[b * 8 + 5];
        // Every slot starts empty; the scan fills the scored prefix in table
        // order, so the prefix is the first joined rows.
        for mm in 0..m {
            window[(b * m + mm) * 2] = max_u32;
            window[(b * m + mm) * 2 + 1] = 0u32;
        }
        let mut visited = 0u32;
        let mut joined = 0u32;
        let mut scored = 0u32;
        let mut status = 0u32;
        let mut complete = 0u32;
        let mut cap = rows_scored_max;
        if cap > m as u32 {
            cap = m as u32;
        }
        if unc == max_u32 {
            // Unknown precursor precision: nothing is searched (contract §9).
            // The two bits arrive combined: the backend IR has no precedent
            // for `|` inside a kernel, so the host ORs them.
            status = bit_exact_absent;
        } else {
            // Neutral parent with the same failure condition as `parent_mass`:
            // adduct 1 subtracts the net hydrogen shift `m_H - m_e`, adduct 2
            // adds it, and anything else (including unknown 0) is overflow.
            // The ordered form matches the host's `i64` value on success.
            let mut parent = 0u32;
            let mut ok = false;
            if adduct == 1u32 {
                if prec >= h_net {
                    parent = prec - h_net;
                    ok = true;
                }
            }
            if adduct == 2u32 {
                if prec <= max_u32 - h_net {
                    parent = prec + h_net;
                    ok = true;
                }
            }
            if ok {
                let tol = ms2_tolerance_u32(prec, ppm);
                let bound = ms2_sat_add(unc, 1u32, max_u32);
                let width = ms2_sat_add(ms2_sat_add(tol, bound, max_u32), max_error, max_u32);
                let lo_mass = ms2_sat_sub(parent, width);
                let hi_mass = ms2_sat_add(parent, width, max_u32);
                // Lower halving search: one visit per mass read, stopping
                // with the rows joined so far past the visit limit. The
                // search is fully unrolled (40 guarded steps, more than the
                // 33 any `u32` range needs): a bound-derived table index
                // inside a loop body does not lower to the backend IR (seen
                // on the CPU backend), while straight-line guarded reads do.
                // Extra steps past completion or the visit limit read
                // nothing, so the visit count matches the reference exactly.
                let mut lower = 0usize;
                let mut upper = rows;
                let mut cut = false;
                // Halving step 0 of 40: one guarded mass read, exactly like
                // `FormulaTable::window`'s lower-bound loop body.
                if lower < upper {
                    if visited + 1u32 > rows_visited_max {
                        cut = true;
                    } else {
                        visited += 1u32;
                        let lmid_0 = lower + (upper - lower) / 2;
                        let lval_0 = table[lmid_0 * 2];
                        if lval_0 < lo_mass {
                            lower = lmid_0 + 1;
                        }
                        if lval_0 == lo_mass {
                            upper = lmid_0;
                        }
                        if lo_mass < lval_0 {
                            upper = lmid_0;
                        }
                    }
                }
                // Halving step 1 of 40: one guarded mass read, exactly like
                // `FormulaTable::window`'s lower-bound loop body.
                if lower < upper {
                    if visited + 1u32 > rows_visited_max {
                        cut = true;
                    } else {
                        visited += 1u32;
                        let lmid_1 = lower + (upper - lower) / 2;
                        let lval_1 = table[lmid_1 * 2];
                        if lval_1 < lo_mass {
                            lower = lmid_1 + 1;
                        }
                        if lval_1 == lo_mass {
                            upper = lmid_1;
                        }
                        if lo_mass < lval_1 {
                            upper = lmid_1;
                        }
                    }
                }
                // Halving step 2 of 40: one guarded mass read, exactly like
                // `FormulaTable::window`'s lower-bound loop body.
                if lower < upper {
                    if visited + 1u32 > rows_visited_max {
                        cut = true;
                    } else {
                        visited += 1u32;
                        let lmid_2 = lower + (upper - lower) / 2;
                        let lval_2 = table[lmid_2 * 2];
                        if lval_2 < lo_mass {
                            lower = lmid_2 + 1;
                        }
                        if lval_2 == lo_mass {
                            upper = lmid_2;
                        }
                        if lo_mass < lval_2 {
                            upper = lmid_2;
                        }
                    }
                }
                // Halving step 3 of 40: one guarded mass read, exactly like
                // `FormulaTable::window`'s lower-bound loop body.
                if lower < upper {
                    if visited + 1u32 > rows_visited_max {
                        cut = true;
                    } else {
                        visited += 1u32;
                        let lmid_3 = lower + (upper - lower) / 2;
                        let lval_3 = table[lmid_3 * 2];
                        if lval_3 < lo_mass {
                            lower = lmid_3 + 1;
                        }
                        if lval_3 == lo_mass {
                            upper = lmid_3;
                        }
                        if lo_mass < lval_3 {
                            upper = lmid_3;
                        }
                    }
                }
                // Halving step 4 of 40: one guarded mass read, exactly like
                // `FormulaTable::window`'s lower-bound loop body.
                if lower < upper {
                    if visited + 1u32 > rows_visited_max {
                        cut = true;
                    } else {
                        visited += 1u32;
                        let lmid_4 = lower + (upper - lower) / 2;
                        let lval_4 = table[lmid_4 * 2];
                        if lval_4 < lo_mass {
                            lower = lmid_4 + 1;
                        }
                        if lval_4 == lo_mass {
                            upper = lmid_4;
                        }
                        if lo_mass < lval_4 {
                            upper = lmid_4;
                        }
                    }
                }
                // Halving step 5 of 40: one guarded mass read, exactly like
                // `FormulaTable::window`'s lower-bound loop body.
                if lower < upper {
                    if visited + 1u32 > rows_visited_max {
                        cut = true;
                    } else {
                        visited += 1u32;
                        let lmid_5 = lower + (upper - lower) / 2;
                        let lval_5 = table[lmid_5 * 2];
                        if lval_5 < lo_mass {
                            lower = lmid_5 + 1;
                        }
                        if lval_5 == lo_mass {
                            upper = lmid_5;
                        }
                        if lo_mass < lval_5 {
                            upper = lmid_5;
                        }
                    }
                }
                // Halving step 6 of 40: one guarded mass read, exactly like
                // `FormulaTable::window`'s lower-bound loop body.
                if lower < upper {
                    if visited + 1u32 > rows_visited_max {
                        cut = true;
                    } else {
                        visited += 1u32;
                        let lmid_6 = lower + (upper - lower) / 2;
                        let lval_6 = table[lmid_6 * 2];
                        if lval_6 < lo_mass {
                            lower = lmid_6 + 1;
                        }
                        if lval_6 == lo_mass {
                            upper = lmid_6;
                        }
                        if lo_mass < lval_6 {
                            upper = lmid_6;
                        }
                    }
                }
                // Halving step 7 of 40: one guarded mass read, exactly like
                // `FormulaTable::window`'s lower-bound loop body.
                if lower < upper {
                    if visited + 1u32 > rows_visited_max {
                        cut = true;
                    } else {
                        visited += 1u32;
                        let lmid_7 = lower + (upper - lower) / 2;
                        let lval_7 = table[lmid_7 * 2];
                        if lval_7 < lo_mass {
                            lower = lmid_7 + 1;
                        }
                        if lval_7 == lo_mass {
                            upper = lmid_7;
                        }
                        if lo_mass < lval_7 {
                            upper = lmid_7;
                        }
                    }
                }
                // Halving step 8 of 40: one guarded mass read, exactly like
                // `FormulaTable::window`'s lower-bound loop body.
                if lower < upper {
                    if visited + 1u32 > rows_visited_max {
                        cut = true;
                    } else {
                        visited += 1u32;
                        let lmid_8 = lower + (upper - lower) / 2;
                        let lval_8 = table[lmid_8 * 2];
                        if lval_8 < lo_mass {
                            lower = lmid_8 + 1;
                        }
                        if lval_8 == lo_mass {
                            upper = lmid_8;
                        }
                        if lo_mass < lval_8 {
                            upper = lmid_8;
                        }
                    }
                }
                // Halving step 9 of 40: one guarded mass read, exactly like
                // `FormulaTable::window`'s lower-bound loop body.
                if lower < upper {
                    if visited + 1u32 > rows_visited_max {
                        cut = true;
                    } else {
                        visited += 1u32;
                        let lmid_9 = lower + (upper - lower) / 2;
                        let lval_9 = table[lmid_9 * 2];
                        if lval_9 < lo_mass {
                            lower = lmid_9 + 1;
                        }
                        if lval_9 == lo_mass {
                            upper = lmid_9;
                        }
                        if lo_mass < lval_9 {
                            upper = lmid_9;
                        }
                    }
                }
                // Halving step 10 of 40: one guarded mass read, exactly like
                // `FormulaTable::window`'s lower-bound loop body.
                if lower < upper {
                    if visited + 1u32 > rows_visited_max {
                        cut = true;
                    } else {
                        visited += 1u32;
                        let lmid_10 = lower + (upper - lower) / 2;
                        let lval_10 = table[lmid_10 * 2];
                        if lval_10 < lo_mass {
                            lower = lmid_10 + 1;
                        }
                        if lval_10 == lo_mass {
                            upper = lmid_10;
                        }
                        if lo_mass < lval_10 {
                            upper = lmid_10;
                        }
                    }
                }
                // Halving step 11 of 40: one guarded mass read, exactly like
                // `FormulaTable::window`'s lower-bound loop body.
                if lower < upper {
                    if visited + 1u32 > rows_visited_max {
                        cut = true;
                    } else {
                        visited += 1u32;
                        let lmid_11 = lower + (upper - lower) / 2;
                        let lval_11 = table[lmid_11 * 2];
                        if lval_11 < lo_mass {
                            lower = lmid_11 + 1;
                        }
                        if lval_11 == lo_mass {
                            upper = lmid_11;
                        }
                        if lo_mass < lval_11 {
                            upper = lmid_11;
                        }
                    }
                }
                // Halving step 12 of 40: one guarded mass read, exactly like
                // `FormulaTable::window`'s lower-bound loop body.
                if lower < upper {
                    if visited + 1u32 > rows_visited_max {
                        cut = true;
                    } else {
                        visited += 1u32;
                        let lmid_12 = lower + (upper - lower) / 2;
                        let lval_12 = table[lmid_12 * 2];
                        if lval_12 < lo_mass {
                            lower = lmid_12 + 1;
                        }
                        if lval_12 == lo_mass {
                            upper = lmid_12;
                        }
                        if lo_mass < lval_12 {
                            upper = lmid_12;
                        }
                    }
                }
                // Halving step 13 of 40: one guarded mass read, exactly like
                // `FormulaTable::window`'s lower-bound loop body.
                if lower < upper {
                    if visited + 1u32 > rows_visited_max {
                        cut = true;
                    } else {
                        visited += 1u32;
                        let lmid_13 = lower + (upper - lower) / 2;
                        let lval_13 = table[lmid_13 * 2];
                        if lval_13 < lo_mass {
                            lower = lmid_13 + 1;
                        }
                        if lval_13 == lo_mass {
                            upper = lmid_13;
                        }
                        if lo_mass < lval_13 {
                            upper = lmid_13;
                        }
                    }
                }
                // Halving step 14 of 40: one guarded mass read, exactly like
                // `FormulaTable::window`'s lower-bound loop body.
                if lower < upper {
                    if visited + 1u32 > rows_visited_max {
                        cut = true;
                    } else {
                        visited += 1u32;
                        let lmid_14 = lower + (upper - lower) / 2;
                        let lval_14 = table[lmid_14 * 2];
                        if lval_14 < lo_mass {
                            lower = lmid_14 + 1;
                        }
                        if lval_14 == lo_mass {
                            upper = lmid_14;
                        }
                        if lo_mass < lval_14 {
                            upper = lmid_14;
                        }
                    }
                }
                // Halving step 15 of 40: one guarded mass read, exactly like
                // `FormulaTable::window`'s lower-bound loop body.
                if lower < upper {
                    if visited + 1u32 > rows_visited_max {
                        cut = true;
                    } else {
                        visited += 1u32;
                        let lmid_15 = lower + (upper - lower) / 2;
                        let lval_15 = table[lmid_15 * 2];
                        if lval_15 < lo_mass {
                            lower = lmid_15 + 1;
                        }
                        if lval_15 == lo_mass {
                            upper = lmid_15;
                        }
                        if lo_mass < lval_15 {
                            upper = lmid_15;
                        }
                    }
                }
                // Halving step 16 of 40: one guarded mass read, exactly like
                // `FormulaTable::window`'s lower-bound loop body.
                if lower < upper {
                    if visited + 1u32 > rows_visited_max {
                        cut = true;
                    } else {
                        visited += 1u32;
                        let lmid_16 = lower + (upper - lower) / 2;
                        let lval_16 = table[lmid_16 * 2];
                        if lval_16 < lo_mass {
                            lower = lmid_16 + 1;
                        }
                        if lval_16 == lo_mass {
                            upper = lmid_16;
                        }
                        if lo_mass < lval_16 {
                            upper = lmid_16;
                        }
                    }
                }
                // Halving step 17 of 40: one guarded mass read, exactly like
                // `FormulaTable::window`'s lower-bound loop body.
                if lower < upper {
                    if visited + 1u32 > rows_visited_max {
                        cut = true;
                    } else {
                        visited += 1u32;
                        let lmid_17 = lower + (upper - lower) / 2;
                        let lval_17 = table[lmid_17 * 2];
                        if lval_17 < lo_mass {
                            lower = lmid_17 + 1;
                        }
                        if lval_17 == lo_mass {
                            upper = lmid_17;
                        }
                        if lo_mass < lval_17 {
                            upper = lmid_17;
                        }
                    }
                }
                // Halving step 18 of 40: one guarded mass read, exactly like
                // `FormulaTable::window`'s lower-bound loop body.
                if lower < upper {
                    if visited + 1u32 > rows_visited_max {
                        cut = true;
                    } else {
                        visited += 1u32;
                        let lmid_18 = lower + (upper - lower) / 2;
                        let lval_18 = table[lmid_18 * 2];
                        if lval_18 < lo_mass {
                            lower = lmid_18 + 1;
                        }
                        if lval_18 == lo_mass {
                            upper = lmid_18;
                        }
                        if lo_mass < lval_18 {
                            upper = lmid_18;
                        }
                    }
                }
                // Halving step 19 of 40: one guarded mass read, exactly like
                // `FormulaTable::window`'s lower-bound loop body.
                if lower < upper {
                    if visited + 1u32 > rows_visited_max {
                        cut = true;
                    } else {
                        visited += 1u32;
                        let lmid_19 = lower + (upper - lower) / 2;
                        let lval_19 = table[lmid_19 * 2];
                        if lval_19 < lo_mass {
                            lower = lmid_19 + 1;
                        }
                        if lval_19 == lo_mass {
                            upper = lmid_19;
                        }
                        if lo_mass < lval_19 {
                            upper = lmid_19;
                        }
                    }
                }
                // Halving step 20 of 40: one guarded mass read, exactly like
                // `FormulaTable::window`'s lower-bound loop body.
                if lower < upper {
                    if visited + 1u32 > rows_visited_max {
                        cut = true;
                    } else {
                        visited += 1u32;
                        let lmid_20 = lower + (upper - lower) / 2;
                        let lval_20 = table[lmid_20 * 2];
                        if lval_20 < lo_mass {
                            lower = lmid_20 + 1;
                        }
                        if lval_20 == lo_mass {
                            upper = lmid_20;
                        }
                        if lo_mass < lval_20 {
                            upper = lmid_20;
                        }
                    }
                }
                // Halving step 21 of 40: one guarded mass read, exactly like
                // `FormulaTable::window`'s lower-bound loop body.
                if lower < upper {
                    if visited + 1u32 > rows_visited_max {
                        cut = true;
                    } else {
                        visited += 1u32;
                        let lmid_21 = lower + (upper - lower) / 2;
                        let lval_21 = table[lmid_21 * 2];
                        if lval_21 < lo_mass {
                            lower = lmid_21 + 1;
                        }
                        if lval_21 == lo_mass {
                            upper = lmid_21;
                        }
                        if lo_mass < lval_21 {
                            upper = lmid_21;
                        }
                    }
                }
                // Halving step 22 of 40: one guarded mass read, exactly like
                // `FormulaTable::window`'s lower-bound loop body.
                if lower < upper {
                    if visited + 1u32 > rows_visited_max {
                        cut = true;
                    } else {
                        visited += 1u32;
                        let lmid_22 = lower + (upper - lower) / 2;
                        let lval_22 = table[lmid_22 * 2];
                        if lval_22 < lo_mass {
                            lower = lmid_22 + 1;
                        }
                        if lval_22 == lo_mass {
                            upper = lmid_22;
                        }
                        if lo_mass < lval_22 {
                            upper = lmid_22;
                        }
                    }
                }
                // Halving step 23 of 40: one guarded mass read, exactly like
                // `FormulaTable::window`'s lower-bound loop body.
                if lower < upper {
                    if visited + 1u32 > rows_visited_max {
                        cut = true;
                    } else {
                        visited += 1u32;
                        let lmid_23 = lower + (upper - lower) / 2;
                        let lval_23 = table[lmid_23 * 2];
                        if lval_23 < lo_mass {
                            lower = lmid_23 + 1;
                        }
                        if lval_23 == lo_mass {
                            upper = lmid_23;
                        }
                        if lo_mass < lval_23 {
                            upper = lmid_23;
                        }
                    }
                }
                // Halving step 24 of 40: one guarded mass read, exactly like
                // `FormulaTable::window`'s lower-bound loop body.
                if lower < upper {
                    if visited + 1u32 > rows_visited_max {
                        cut = true;
                    } else {
                        visited += 1u32;
                        let lmid_24 = lower + (upper - lower) / 2;
                        let lval_24 = table[lmid_24 * 2];
                        if lval_24 < lo_mass {
                            lower = lmid_24 + 1;
                        }
                        if lval_24 == lo_mass {
                            upper = lmid_24;
                        }
                        if lo_mass < lval_24 {
                            upper = lmid_24;
                        }
                    }
                }
                // Halving step 25 of 40: one guarded mass read, exactly like
                // `FormulaTable::window`'s lower-bound loop body.
                if lower < upper {
                    if visited + 1u32 > rows_visited_max {
                        cut = true;
                    } else {
                        visited += 1u32;
                        let lmid_25 = lower + (upper - lower) / 2;
                        let lval_25 = table[lmid_25 * 2];
                        if lval_25 < lo_mass {
                            lower = lmid_25 + 1;
                        }
                        if lval_25 == lo_mass {
                            upper = lmid_25;
                        }
                        if lo_mass < lval_25 {
                            upper = lmid_25;
                        }
                    }
                }
                // Halving step 26 of 40: one guarded mass read, exactly like
                // `FormulaTable::window`'s lower-bound loop body.
                if lower < upper {
                    if visited + 1u32 > rows_visited_max {
                        cut = true;
                    } else {
                        visited += 1u32;
                        let lmid_26 = lower + (upper - lower) / 2;
                        let lval_26 = table[lmid_26 * 2];
                        if lval_26 < lo_mass {
                            lower = lmid_26 + 1;
                        }
                        if lval_26 == lo_mass {
                            upper = lmid_26;
                        }
                        if lo_mass < lval_26 {
                            upper = lmid_26;
                        }
                    }
                }
                // Halving step 27 of 40: one guarded mass read, exactly like
                // `FormulaTable::window`'s lower-bound loop body.
                if lower < upper {
                    if visited + 1u32 > rows_visited_max {
                        cut = true;
                    } else {
                        visited += 1u32;
                        let lmid_27 = lower + (upper - lower) / 2;
                        let lval_27 = table[lmid_27 * 2];
                        if lval_27 < lo_mass {
                            lower = lmid_27 + 1;
                        }
                        if lval_27 == lo_mass {
                            upper = lmid_27;
                        }
                        if lo_mass < lval_27 {
                            upper = lmid_27;
                        }
                    }
                }
                // Halving step 28 of 40: one guarded mass read, exactly like
                // `FormulaTable::window`'s lower-bound loop body.
                if lower < upper {
                    if visited + 1u32 > rows_visited_max {
                        cut = true;
                    } else {
                        visited += 1u32;
                        let lmid_28 = lower + (upper - lower) / 2;
                        let lval_28 = table[lmid_28 * 2];
                        if lval_28 < lo_mass {
                            lower = lmid_28 + 1;
                        }
                        if lval_28 == lo_mass {
                            upper = lmid_28;
                        }
                        if lo_mass < lval_28 {
                            upper = lmid_28;
                        }
                    }
                }
                // Halving step 29 of 40: one guarded mass read, exactly like
                // `FormulaTable::window`'s lower-bound loop body.
                if lower < upper {
                    if visited + 1u32 > rows_visited_max {
                        cut = true;
                    } else {
                        visited += 1u32;
                        let lmid_29 = lower + (upper - lower) / 2;
                        let lval_29 = table[lmid_29 * 2];
                        if lval_29 < lo_mass {
                            lower = lmid_29 + 1;
                        }
                        if lval_29 == lo_mass {
                            upper = lmid_29;
                        }
                        if lo_mass < lval_29 {
                            upper = lmid_29;
                        }
                    }
                }
                // Halving step 30 of 40: one guarded mass read, exactly like
                // `FormulaTable::window`'s lower-bound loop body.
                if lower < upper {
                    if visited + 1u32 > rows_visited_max {
                        cut = true;
                    } else {
                        visited += 1u32;
                        let lmid_30 = lower + (upper - lower) / 2;
                        let lval_30 = table[lmid_30 * 2];
                        if lval_30 < lo_mass {
                            lower = lmid_30 + 1;
                        }
                        if lval_30 == lo_mass {
                            upper = lmid_30;
                        }
                        if lo_mass < lval_30 {
                            upper = lmid_30;
                        }
                    }
                }
                // Halving step 31 of 40: one guarded mass read, exactly like
                // `FormulaTable::window`'s lower-bound loop body.
                if lower < upper {
                    if visited + 1u32 > rows_visited_max {
                        cut = true;
                    } else {
                        visited += 1u32;
                        let lmid_31 = lower + (upper - lower) / 2;
                        let lval_31 = table[lmid_31 * 2];
                        if lval_31 < lo_mass {
                            lower = lmid_31 + 1;
                        }
                        if lval_31 == lo_mass {
                            upper = lmid_31;
                        }
                        if lo_mass < lval_31 {
                            upper = lmid_31;
                        }
                    }
                }
                // Halving step 32 of 40: one guarded mass read, exactly like
                // `FormulaTable::window`'s lower-bound loop body.
                if lower < upper {
                    if visited + 1u32 > rows_visited_max {
                        cut = true;
                    } else {
                        visited += 1u32;
                        let lmid_32 = lower + (upper - lower) / 2;
                        let lval_32 = table[lmid_32 * 2];
                        if lval_32 < lo_mass {
                            lower = lmid_32 + 1;
                        }
                        if lval_32 == lo_mass {
                            upper = lmid_32;
                        }
                        if lo_mass < lval_32 {
                            upper = lmid_32;
                        }
                    }
                }
                // Halving step 33 of 40: one guarded mass read, exactly like
                // `FormulaTable::window`'s lower-bound loop body.
                if lower < upper {
                    if visited + 1u32 > rows_visited_max {
                        cut = true;
                    } else {
                        visited += 1u32;
                        let lmid_33 = lower + (upper - lower) / 2;
                        let lval_33 = table[lmid_33 * 2];
                        if lval_33 < lo_mass {
                            lower = lmid_33 + 1;
                        }
                        if lval_33 == lo_mass {
                            upper = lmid_33;
                        }
                        if lo_mass < lval_33 {
                            upper = lmid_33;
                        }
                    }
                }
                // Halving step 34 of 40: one guarded mass read, exactly like
                // `FormulaTable::window`'s lower-bound loop body.
                if lower < upper {
                    if visited + 1u32 > rows_visited_max {
                        cut = true;
                    } else {
                        visited += 1u32;
                        let lmid_34 = lower + (upper - lower) / 2;
                        let lval_34 = table[lmid_34 * 2];
                        if lval_34 < lo_mass {
                            lower = lmid_34 + 1;
                        }
                        if lval_34 == lo_mass {
                            upper = lmid_34;
                        }
                        if lo_mass < lval_34 {
                            upper = lmid_34;
                        }
                    }
                }
                // Halving step 35 of 40: one guarded mass read, exactly like
                // `FormulaTable::window`'s lower-bound loop body.
                if lower < upper {
                    if visited + 1u32 > rows_visited_max {
                        cut = true;
                    } else {
                        visited += 1u32;
                        let lmid_35 = lower + (upper - lower) / 2;
                        let lval_35 = table[lmid_35 * 2];
                        if lval_35 < lo_mass {
                            lower = lmid_35 + 1;
                        }
                        if lval_35 == lo_mass {
                            upper = lmid_35;
                        }
                        if lo_mass < lval_35 {
                            upper = lmid_35;
                        }
                    }
                }
                // Halving step 36 of 40: one guarded mass read, exactly like
                // `FormulaTable::window`'s lower-bound loop body.
                if lower < upper {
                    if visited + 1u32 > rows_visited_max {
                        cut = true;
                    } else {
                        visited += 1u32;
                        let lmid_36 = lower + (upper - lower) / 2;
                        let lval_36 = table[lmid_36 * 2];
                        if lval_36 < lo_mass {
                            lower = lmid_36 + 1;
                        }
                        if lval_36 == lo_mass {
                            upper = lmid_36;
                        }
                        if lo_mass < lval_36 {
                            upper = lmid_36;
                        }
                    }
                }
                // Halving step 37 of 40: one guarded mass read, exactly like
                // `FormulaTable::window`'s lower-bound loop body.
                if lower < upper {
                    if visited + 1u32 > rows_visited_max {
                        cut = true;
                    } else {
                        visited += 1u32;
                        let lmid_37 = lower + (upper - lower) / 2;
                        let lval_37 = table[lmid_37 * 2];
                        if lval_37 < lo_mass {
                            lower = lmid_37 + 1;
                        }
                        if lval_37 == lo_mass {
                            upper = lmid_37;
                        }
                        if lo_mass < lval_37 {
                            upper = lmid_37;
                        }
                    }
                }
                // Halving step 38 of 40: one guarded mass read, exactly like
                // `FormulaTable::window`'s lower-bound loop body.
                if lower < upper {
                    if visited + 1u32 > rows_visited_max {
                        cut = true;
                    } else {
                        visited += 1u32;
                        let lmid_38 = lower + (upper - lower) / 2;
                        let lval_38 = table[lmid_38 * 2];
                        if lval_38 < lo_mass {
                            lower = lmid_38 + 1;
                        }
                        if lval_38 == lo_mass {
                            upper = lmid_38;
                        }
                        if lo_mass < lval_38 {
                            upper = lmid_38;
                        }
                    }
                }
                // Halving step 39 of 40: one guarded mass read, exactly like
                // `FormulaTable::window`'s lower-bound loop body.
                if lower < upper {
                    if visited + 1u32 > rows_visited_max {
                        cut = true;
                    } else {
                        visited += 1u32;
                        let lmid_39 = lower + (upper - lower) / 2;
                        let lval_39 = table[lmid_39 * 2];
                        if lval_39 < lo_mass {
                            lower = lmid_39 + 1;
                        }
                        if lval_39 == lo_mass {
                            upper = lmid_39;
                        }
                        if lo_mass < lval_39 {
                            upper = lmid_39;
                        }
                    }
                }
                let first = lower;
                let mut lo = 0usize;
                let mut hi = rows;
                // Halving step 0 of 40: one guarded mass read, exactly like
                // `FormulaTable::window`'s upper-bound loop body.
                if lo < hi {
                    if visited + 1u32 > rows_visited_max {
                        cut = true;
                    } else {
                        visited += 1u32;
                        let umid_0 = lo + (hi - lo) / 2;
                        let uval_0 = table[umid_0 * 2];
                        if uval_0 < hi_mass {
                            lo = umid_0 + 1;
                        }
                        if uval_0 == hi_mass {
                            lo = umid_0 + 1;
                        }
                        if hi_mass < uval_0 {
                            hi = umid_0;
                        }
                    }
                }
                // Halving step 1 of 40: one guarded mass read, exactly like
                // `FormulaTable::window`'s upper-bound loop body.
                if lo < hi {
                    if visited + 1u32 > rows_visited_max {
                        cut = true;
                    } else {
                        visited += 1u32;
                        let umid_1 = lo + (hi - lo) / 2;
                        let uval_1 = table[umid_1 * 2];
                        if uval_1 < hi_mass {
                            lo = umid_1 + 1;
                        }
                        if uval_1 == hi_mass {
                            lo = umid_1 + 1;
                        }
                        if hi_mass < uval_1 {
                            hi = umid_1;
                        }
                    }
                }
                // Halving step 2 of 40: one guarded mass read, exactly like
                // `FormulaTable::window`'s upper-bound loop body.
                if lo < hi {
                    if visited + 1u32 > rows_visited_max {
                        cut = true;
                    } else {
                        visited += 1u32;
                        let umid_2 = lo + (hi - lo) / 2;
                        let uval_2 = table[umid_2 * 2];
                        if uval_2 < hi_mass {
                            lo = umid_2 + 1;
                        }
                        if uval_2 == hi_mass {
                            lo = umid_2 + 1;
                        }
                        if hi_mass < uval_2 {
                            hi = umid_2;
                        }
                    }
                }
                // Halving step 3 of 40: one guarded mass read, exactly like
                // `FormulaTable::window`'s upper-bound loop body.
                if lo < hi {
                    if visited + 1u32 > rows_visited_max {
                        cut = true;
                    } else {
                        visited += 1u32;
                        let umid_3 = lo + (hi - lo) / 2;
                        let uval_3 = table[umid_3 * 2];
                        if uval_3 < hi_mass {
                            lo = umid_3 + 1;
                        }
                        if uval_3 == hi_mass {
                            lo = umid_3 + 1;
                        }
                        if hi_mass < uval_3 {
                            hi = umid_3;
                        }
                    }
                }
                // Halving step 4 of 40: one guarded mass read, exactly like
                // `FormulaTable::window`'s upper-bound loop body.
                if lo < hi {
                    if visited + 1u32 > rows_visited_max {
                        cut = true;
                    } else {
                        visited += 1u32;
                        let umid_4 = lo + (hi - lo) / 2;
                        let uval_4 = table[umid_4 * 2];
                        if uval_4 < hi_mass {
                            lo = umid_4 + 1;
                        }
                        if uval_4 == hi_mass {
                            lo = umid_4 + 1;
                        }
                        if hi_mass < uval_4 {
                            hi = umid_4;
                        }
                    }
                }
                // Halving step 5 of 40: one guarded mass read, exactly like
                // `FormulaTable::window`'s upper-bound loop body.
                if lo < hi {
                    if visited + 1u32 > rows_visited_max {
                        cut = true;
                    } else {
                        visited += 1u32;
                        let umid_5 = lo + (hi - lo) / 2;
                        let uval_5 = table[umid_5 * 2];
                        if uval_5 < hi_mass {
                            lo = umid_5 + 1;
                        }
                        if uval_5 == hi_mass {
                            lo = umid_5 + 1;
                        }
                        if hi_mass < uval_5 {
                            hi = umid_5;
                        }
                    }
                }
                // Halving step 6 of 40: one guarded mass read, exactly like
                // `FormulaTable::window`'s upper-bound loop body.
                if lo < hi {
                    if visited + 1u32 > rows_visited_max {
                        cut = true;
                    } else {
                        visited += 1u32;
                        let umid_6 = lo + (hi - lo) / 2;
                        let uval_6 = table[umid_6 * 2];
                        if uval_6 < hi_mass {
                            lo = umid_6 + 1;
                        }
                        if uval_6 == hi_mass {
                            lo = umid_6 + 1;
                        }
                        if hi_mass < uval_6 {
                            hi = umid_6;
                        }
                    }
                }
                // Halving step 7 of 40: one guarded mass read, exactly like
                // `FormulaTable::window`'s upper-bound loop body.
                if lo < hi {
                    if visited + 1u32 > rows_visited_max {
                        cut = true;
                    } else {
                        visited += 1u32;
                        let umid_7 = lo + (hi - lo) / 2;
                        let uval_7 = table[umid_7 * 2];
                        if uval_7 < hi_mass {
                            lo = umid_7 + 1;
                        }
                        if uval_7 == hi_mass {
                            lo = umid_7 + 1;
                        }
                        if hi_mass < uval_7 {
                            hi = umid_7;
                        }
                    }
                }
                // Halving step 8 of 40: one guarded mass read, exactly like
                // `FormulaTable::window`'s upper-bound loop body.
                if lo < hi {
                    if visited + 1u32 > rows_visited_max {
                        cut = true;
                    } else {
                        visited += 1u32;
                        let umid_8 = lo + (hi - lo) / 2;
                        let uval_8 = table[umid_8 * 2];
                        if uval_8 < hi_mass {
                            lo = umid_8 + 1;
                        }
                        if uval_8 == hi_mass {
                            lo = umid_8 + 1;
                        }
                        if hi_mass < uval_8 {
                            hi = umid_8;
                        }
                    }
                }
                // Halving step 9 of 40: one guarded mass read, exactly like
                // `FormulaTable::window`'s upper-bound loop body.
                if lo < hi {
                    if visited + 1u32 > rows_visited_max {
                        cut = true;
                    } else {
                        visited += 1u32;
                        let umid_9 = lo + (hi - lo) / 2;
                        let uval_9 = table[umid_9 * 2];
                        if uval_9 < hi_mass {
                            lo = umid_9 + 1;
                        }
                        if uval_9 == hi_mass {
                            lo = umid_9 + 1;
                        }
                        if hi_mass < uval_9 {
                            hi = umid_9;
                        }
                    }
                }
                // Halving step 10 of 40: one guarded mass read, exactly like
                // `FormulaTable::window`'s upper-bound loop body.
                if lo < hi {
                    if visited + 1u32 > rows_visited_max {
                        cut = true;
                    } else {
                        visited += 1u32;
                        let umid_10 = lo + (hi - lo) / 2;
                        let uval_10 = table[umid_10 * 2];
                        if uval_10 < hi_mass {
                            lo = umid_10 + 1;
                        }
                        if uval_10 == hi_mass {
                            lo = umid_10 + 1;
                        }
                        if hi_mass < uval_10 {
                            hi = umid_10;
                        }
                    }
                }
                // Halving step 11 of 40: one guarded mass read, exactly like
                // `FormulaTable::window`'s upper-bound loop body.
                if lo < hi {
                    if visited + 1u32 > rows_visited_max {
                        cut = true;
                    } else {
                        visited += 1u32;
                        let umid_11 = lo + (hi - lo) / 2;
                        let uval_11 = table[umid_11 * 2];
                        if uval_11 < hi_mass {
                            lo = umid_11 + 1;
                        }
                        if uval_11 == hi_mass {
                            lo = umid_11 + 1;
                        }
                        if hi_mass < uval_11 {
                            hi = umid_11;
                        }
                    }
                }
                // Halving step 12 of 40: one guarded mass read, exactly like
                // `FormulaTable::window`'s upper-bound loop body.
                if lo < hi {
                    if visited + 1u32 > rows_visited_max {
                        cut = true;
                    } else {
                        visited += 1u32;
                        let umid_12 = lo + (hi - lo) / 2;
                        let uval_12 = table[umid_12 * 2];
                        if uval_12 < hi_mass {
                            lo = umid_12 + 1;
                        }
                        if uval_12 == hi_mass {
                            lo = umid_12 + 1;
                        }
                        if hi_mass < uval_12 {
                            hi = umid_12;
                        }
                    }
                }
                // Halving step 13 of 40: one guarded mass read, exactly like
                // `FormulaTable::window`'s upper-bound loop body.
                if lo < hi {
                    if visited + 1u32 > rows_visited_max {
                        cut = true;
                    } else {
                        visited += 1u32;
                        let umid_13 = lo + (hi - lo) / 2;
                        let uval_13 = table[umid_13 * 2];
                        if uval_13 < hi_mass {
                            lo = umid_13 + 1;
                        }
                        if uval_13 == hi_mass {
                            lo = umid_13 + 1;
                        }
                        if hi_mass < uval_13 {
                            hi = umid_13;
                        }
                    }
                }
                // Halving step 14 of 40: one guarded mass read, exactly like
                // `FormulaTable::window`'s upper-bound loop body.
                if lo < hi {
                    if visited + 1u32 > rows_visited_max {
                        cut = true;
                    } else {
                        visited += 1u32;
                        let umid_14 = lo + (hi - lo) / 2;
                        let uval_14 = table[umid_14 * 2];
                        if uval_14 < hi_mass {
                            lo = umid_14 + 1;
                        }
                        if uval_14 == hi_mass {
                            lo = umid_14 + 1;
                        }
                        if hi_mass < uval_14 {
                            hi = umid_14;
                        }
                    }
                }
                // Halving step 15 of 40: one guarded mass read, exactly like
                // `FormulaTable::window`'s upper-bound loop body.
                if lo < hi {
                    if visited + 1u32 > rows_visited_max {
                        cut = true;
                    } else {
                        visited += 1u32;
                        let umid_15 = lo + (hi - lo) / 2;
                        let uval_15 = table[umid_15 * 2];
                        if uval_15 < hi_mass {
                            lo = umid_15 + 1;
                        }
                        if uval_15 == hi_mass {
                            lo = umid_15 + 1;
                        }
                        if hi_mass < uval_15 {
                            hi = umid_15;
                        }
                    }
                }
                // Halving step 16 of 40: one guarded mass read, exactly like
                // `FormulaTable::window`'s upper-bound loop body.
                if lo < hi {
                    if visited + 1u32 > rows_visited_max {
                        cut = true;
                    } else {
                        visited += 1u32;
                        let umid_16 = lo + (hi - lo) / 2;
                        let uval_16 = table[umid_16 * 2];
                        if uval_16 < hi_mass {
                            lo = umid_16 + 1;
                        }
                        if uval_16 == hi_mass {
                            lo = umid_16 + 1;
                        }
                        if hi_mass < uval_16 {
                            hi = umid_16;
                        }
                    }
                }
                // Halving step 17 of 40: one guarded mass read, exactly like
                // `FormulaTable::window`'s upper-bound loop body.
                if lo < hi {
                    if visited + 1u32 > rows_visited_max {
                        cut = true;
                    } else {
                        visited += 1u32;
                        let umid_17 = lo + (hi - lo) / 2;
                        let uval_17 = table[umid_17 * 2];
                        if uval_17 < hi_mass {
                            lo = umid_17 + 1;
                        }
                        if uval_17 == hi_mass {
                            lo = umid_17 + 1;
                        }
                        if hi_mass < uval_17 {
                            hi = umid_17;
                        }
                    }
                }
                // Halving step 18 of 40: one guarded mass read, exactly like
                // `FormulaTable::window`'s upper-bound loop body.
                if lo < hi {
                    if visited + 1u32 > rows_visited_max {
                        cut = true;
                    } else {
                        visited += 1u32;
                        let umid_18 = lo + (hi - lo) / 2;
                        let uval_18 = table[umid_18 * 2];
                        if uval_18 < hi_mass {
                            lo = umid_18 + 1;
                        }
                        if uval_18 == hi_mass {
                            lo = umid_18 + 1;
                        }
                        if hi_mass < uval_18 {
                            hi = umid_18;
                        }
                    }
                }
                // Halving step 19 of 40: one guarded mass read, exactly like
                // `FormulaTable::window`'s upper-bound loop body.
                if lo < hi {
                    if visited + 1u32 > rows_visited_max {
                        cut = true;
                    } else {
                        visited += 1u32;
                        let umid_19 = lo + (hi - lo) / 2;
                        let uval_19 = table[umid_19 * 2];
                        if uval_19 < hi_mass {
                            lo = umid_19 + 1;
                        }
                        if uval_19 == hi_mass {
                            lo = umid_19 + 1;
                        }
                        if hi_mass < uval_19 {
                            hi = umid_19;
                        }
                    }
                }
                // Halving step 20 of 40: one guarded mass read, exactly like
                // `FormulaTable::window`'s upper-bound loop body.
                if lo < hi {
                    if visited + 1u32 > rows_visited_max {
                        cut = true;
                    } else {
                        visited += 1u32;
                        let umid_20 = lo + (hi - lo) / 2;
                        let uval_20 = table[umid_20 * 2];
                        if uval_20 < hi_mass {
                            lo = umid_20 + 1;
                        }
                        if uval_20 == hi_mass {
                            lo = umid_20 + 1;
                        }
                        if hi_mass < uval_20 {
                            hi = umid_20;
                        }
                    }
                }
                // Halving step 21 of 40: one guarded mass read, exactly like
                // `FormulaTable::window`'s upper-bound loop body.
                if lo < hi {
                    if visited + 1u32 > rows_visited_max {
                        cut = true;
                    } else {
                        visited += 1u32;
                        let umid_21 = lo + (hi - lo) / 2;
                        let uval_21 = table[umid_21 * 2];
                        if uval_21 < hi_mass {
                            lo = umid_21 + 1;
                        }
                        if uval_21 == hi_mass {
                            lo = umid_21 + 1;
                        }
                        if hi_mass < uval_21 {
                            hi = umid_21;
                        }
                    }
                }
                // Halving step 22 of 40: one guarded mass read, exactly like
                // `FormulaTable::window`'s upper-bound loop body.
                if lo < hi {
                    if visited + 1u32 > rows_visited_max {
                        cut = true;
                    } else {
                        visited += 1u32;
                        let umid_22 = lo + (hi - lo) / 2;
                        let uval_22 = table[umid_22 * 2];
                        if uval_22 < hi_mass {
                            lo = umid_22 + 1;
                        }
                        if uval_22 == hi_mass {
                            lo = umid_22 + 1;
                        }
                        if hi_mass < uval_22 {
                            hi = umid_22;
                        }
                    }
                }
                // Halving step 23 of 40: one guarded mass read, exactly like
                // `FormulaTable::window`'s upper-bound loop body.
                if lo < hi {
                    if visited + 1u32 > rows_visited_max {
                        cut = true;
                    } else {
                        visited += 1u32;
                        let umid_23 = lo + (hi - lo) / 2;
                        let uval_23 = table[umid_23 * 2];
                        if uval_23 < hi_mass {
                            lo = umid_23 + 1;
                        }
                        if uval_23 == hi_mass {
                            lo = umid_23 + 1;
                        }
                        if hi_mass < uval_23 {
                            hi = umid_23;
                        }
                    }
                }
                // Halving step 24 of 40: one guarded mass read, exactly like
                // `FormulaTable::window`'s upper-bound loop body.
                if lo < hi {
                    if visited + 1u32 > rows_visited_max {
                        cut = true;
                    } else {
                        visited += 1u32;
                        let umid_24 = lo + (hi - lo) / 2;
                        let uval_24 = table[umid_24 * 2];
                        if uval_24 < hi_mass {
                            lo = umid_24 + 1;
                        }
                        if uval_24 == hi_mass {
                            lo = umid_24 + 1;
                        }
                        if hi_mass < uval_24 {
                            hi = umid_24;
                        }
                    }
                }
                // Halving step 25 of 40: one guarded mass read, exactly like
                // `FormulaTable::window`'s upper-bound loop body.
                if lo < hi {
                    if visited + 1u32 > rows_visited_max {
                        cut = true;
                    } else {
                        visited += 1u32;
                        let umid_25 = lo + (hi - lo) / 2;
                        let uval_25 = table[umid_25 * 2];
                        if uval_25 < hi_mass {
                            lo = umid_25 + 1;
                        }
                        if uval_25 == hi_mass {
                            lo = umid_25 + 1;
                        }
                        if hi_mass < uval_25 {
                            hi = umid_25;
                        }
                    }
                }
                // Halving step 26 of 40: one guarded mass read, exactly like
                // `FormulaTable::window`'s upper-bound loop body.
                if lo < hi {
                    if visited + 1u32 > rows_visited_max {
                        cut = true;
                    } else {
                        visited += 1u32;
                        let umid_26 = lo + (hi - lo) / 2;
                        let uval_26 = table[umid_26 * 2];
                        if uval_26 < hi_mass {
                            lo = umid_26 + 1;
                        }
                        if uval_26 == hi_mass {
                            lo = umid_26 + 1;
                        }
                        if hi_mass < uval_26 {
                            hi = umid_26;
                        }
                    }
                }
                // Halving step 27 of 40: one guarded mass read, exactly like
                // `FormulaTable::window`'s upper-bound loop body.
                if lo < hi {
                    if visited + 1u32 > rows_visited_max {
                        cut = true;
                    } else {
                        visited += 1u32;
                        let umid_27 = lo + (hi - lo) / 2;
                        let uval_27 = table[umid_27 * 2];
                        if uval_27 < hi_mass {
                            lo = umid_27 + 1;
                        }
                        if uval_27 == hi_mass {
                            lo = umid_27 + 1;
                        }
                        if hi_mass < uval_27 {
                            hi = umid_27;
                        }
                    }
                }
                // Halving step 28 of 40: one guarded mass read, exactly like
                // `FormulaTable::window`'s upper-bound loop body.
                if lo < hi {
                    if visited + 1u32 > rows_visited_max {
                        cut = true;
                    } else {
                        visited += 1u32;
                        let umid_28 = lo + (hi - lo) / 2;
                        let uval_28 = table[umid_28 * 2];
                        if uval_28 < hi_mass {
                            lo = umid_28 + 1;
                        }
                        if uval_28 == hi_mass {
                            lo = umid_28 + 1;
                        }
                        if hi_mass < uval_28 {
                            hi = umid_28;
                        }
                    }
                }
                // Halving step 29 of 40: one guarded mass read, exactly like
                // `FormulaTable::window`'s upper-bound loop body.
                if lo < hi {
                    if visited + 1u32 > rows_visited_max {
                        cut = true;
                    } else {
                        visited += 1u32;
                        let umid_29 = lo + (hi - lo) / 2;
                        let uval_29 = table[umid_29 * 2];
                        if uval_29 < hi_mass {
                            lo = umid_29 + 1;
                        }
                        if uval_29 == hi_mass {
                            lo = umid_29 + 1;
                        }
                        if hi_mass < uval_29 {
                            hi = umid_29;
                        }
                    }
                }
                // Halving step 30 of 40: one guarded mass read, exactly like
                // `FormulaTable::window`'s upper-bound loop body.
                if lo < hi {
                    if visited + 1u32 > rows_visited_max {
                        cut = true;
                    } else {
                        visited += 1u32;
                        let umid_30 = lo + (hi - lo) / 2;
                        let uval_30 = table[umid_30 * 2];
                        if uval_30 < hi_mass {
                            lo = umid_30 + 1;
                        }
                        if uval_30 == hi_mass {
                            lo = umid_30 + 1;
                        }
                        if hi_mass < uval_30 {
                            hi = umid_30;
                        }
                    }
                }
                // Halving step 31 of 40: one guarded mass read, exactly like
                // `FormulaTable::window`'s upper-bound loop body.
                if lo < hi {
                    if visited + 1u32 > rows_visited_max {
                        cut = true;
                    } else {
                        visited += 1u32;
                        let umid_31 = lo + (hi - lo) / 2;
                        let uval_31 = table[umid_31 * 2];
                        if uval_31 < hi_mass {
                            lo = umid_31 + 1;
                        }
                        if uval_31 == hi_mass {
                            lo = umid_31 + 1;
                        }
                        if hi_mass < uval_31 {
                            hi = umid_31;
                        }
                    }
                }
                // Halving step 32 of 40: one guarded mass read, exactly like
                // `FormulaTable::window`'s upper-bound loop body.
                if lo < hi {
                    if visited + 1u32 > rows_visited_max {
                        cut = true;
                    } else {
                        visited += 1u32;
                        let umid_32 = lo + (hi - lo) / 2;
                        let uval_32 = table[umid_32 * 2];
                        if uval_32 < hi_mass {
                            lo = umid_32 + 1;
                        }
                        if uval_32 == hi_mass {
                            lo = umid_32 + 1;
                        }
                        if hi_mass < uval_32 {
                            hi = umid_32;
                        }
                    }
                }
                // Halving step 33 of 40: one guarded mass read, exactly like
                // `FormulaTable::window`'s upper-bound loop body.
                if lo < hi {
                    if visited + 1u32 > rows_visited_max {
                        cut = true;
                    } else {
                        visited += 1u32;
                        let umid_33 = lo + (hi - lo) / 2;
                        let uval_33 = table[umid_33 * 2];
                        if uval_33 < hi_mass {
                            lo = umid_33 + 1;
                        }
                        if uval_33 == hi_mass {
                            lo = umid_33 + 1;
                        }
                        if hi_mass < uval_33 {
                            hi = umid_33;
                        }
                    }
                }
                // Halving step 34 of 40: one guarded mass read, exactly like
                // `FormulaTable::window`'s upper-bound loop body.
                if lo < hi {
                    if visited + 1u32 > rows_visited_max {
                        cut = true;
                    } else {
                        visited += 1u32;
                        let umid_34 = lo + (hi - lo) / 2;
                        let uval_34 = table[umid_34 * 2];
                        if uval_34 < hi_mass {
                            lo = umid_34 + 1;
                        }
                        if uval_34 == hi_mass {
                            lo = umid_34 + 1;
                        }
                        if hi_mass < uval_34 {
                            hi = umid_34;
                        }
                    }
                }
                // Halving step 35 of 40: one guarded mass read, exactly like
                // `FormulaTable::window`'s upper-bound loop body.
                if lo < hi {
                    if visited + 1u32 > rows_visited_max {
                        cut = true;
                    } else {
                        visited += 1u32;
                        let umid_35 = lo + (hi - lo) / 2;
                        let uval_35 = table[umid_35 * 2];
                        if uval_35 < hi_mass {
                            lo = umid_35 + 1;
                        }
                        if uval_35 == hi_mass {
                            lo = umid_35 + 1;
                        }
                        if hi_mass < uval_35 {
                            hi = umid_35;
                        }
                    }
                }
                // Halving step 36 of 40: one guarded mass read, exactly like
                // `FormulaTable::window`'s upper-bound loop body.
                if lo < hi {
                    if visited + 1u32 > rows_visited_max {
                        cut = true;
                    } else {
                        visited += 1u32;
                        let umid_36 = lo + (hi - lo) / 2;
                        let uval_36 = table[umid_36 * 2];
                        if uval_36 < hi_mass {
                            lo = umid_36 + 1;
                        }
                        if uval_36 == hi_mass {
                            lo = umid_36 + 1;
                        }
                        if hi_mass < uval_36 {
                            hi = umid_36;
                        }
                    }
                }
                // Halving step 37 of 40: one guarded mass read, exactly like
                // `FormulaTable::window`'s upper-bound loop body.
                if lo < hi {
                    if visited + 1u32 > rows_visited_max {
                        cut = true;
                    } else {
                        visited += 1u32;
                        let umid_37 = lo + (hi - lo) / 2;
                        let uval_37 = table[umid_37 * 2];
                        if uval_37 < hi_mass {
                            lo = umid_37 + 1;
                        }
                        if uval_37 == hi_mass {
                            lo = umid_37 + 1;
                        }
                        if hi_mass < uval_37 {
                            hi = umid_37;
                        }
                    }
                }
                // Halving step 38 of 40: one guarded mass read, exactly like
                // `FormulaTable::window`'s upper-bound loop body.
                if lo < hi {
                    if visited + 1u32 > rows_visited_max {
                        cut = true;
                    } else {
                        visited += 1u32;
                        let umid_38 = lo + (hi - lo) / 2;
                        let uval_38 = table[umid_38 * 2];
                        if uval_38 < hi_mass {
                            lo = umid_38 + 1;
                        }
                        if uval_38 == hi_mass {
                            lo = umid_38 + 1;
                        }
                        if hi_mass < uval_38 {
                            hi = umid_38;
                        }
                    }
                }
                // Halving step 39 of 40: one guarded mass read, exactly like
                // `FormulaTable::window`'s upper-bound loop body.
                if lo < hi {
                    if visited + 1u32 > rows_visited_max {
                        cut = true;
                    } else {
                        visited += 1u32;
                        let umid_39 = lo + (hi - lo) / 2;
                        let uval_39 = table[umid_39 * 2];
                        if uval_39 < hi_mass {
                            lo = umid_39 + 1;
                        }
                        if uval_39 == hi_mass {
                            lo = umid_39 + 1;
                        }
                        if hi_mass < uval_39 {
                            hi = umid_39;
                        }
                    }
                }
                if cut {
                    // Exhausted before or during the halving searches: nothing
                    // joined, the counters as they stand.
                    status = bit_exhausted;
                } else {
                    let mut done = false;
                    for row in first..lo {
                        if visited + 1u32 > rows_visited_max {
                            done = true;
                        } else {
                            visited += 1u32;
                            let e_row = ms2_sat_add(table[row * 2 + 1], bound, max_u32);
                            let flag = ms2_decide(parent, table[row * 2], e_row, tol);
                            if flag != 0u32 {
                                if scored < cap {
                                    window[(b * m + scored as usize) * 2] = row as u32;
                                    window[(b * m + scored as usize) * 2 + 1] = flag;
                                    scored += 1u32;
                                }
                                joined += 1u32;
                            }
                        }
                    }
                    if done {
                        // The visit limit stopped the scan: keep the counters
                        // and the rows joined so far, scored under the cap.
                        let mut sc = joined;
                        if sc > cap {
                            sc = cap;
                        }
                        scored = sc;
                        status = bit_exhausted;
                    } else if joined > cap {
                        // More joined rows than can be scored is exhausted too.
                        status = bit_exhausted;
                    } else {
                        // A finished search is complete even when it joined
                        // nothing (then it is absent instead).
                        if joined == 0u32 {
                            status = bit_absent;
                        }
                        complete = 1u32;
                    }
                }
            } else {
                status = bit_overflow;
            }
        }
        counters[b * 5] = visited;
        counters[b * 5 + 1] = joined;
        counters[b * 5 + 2] = scored;
        counters[b * 5 + 3] = status;
        counters[b * 5 + 4] = complete;
    }
}

/// Run the precursor window search (architecture §3.3) into `out.window` and
/// `out.counters`.
///
/// `table` is `[R, 2]` (mass, per-row arithmetic bound), `meta` the `[B, 8]`
/// per-spectrum metadata; the query per spectrum is `precursor_mz =
/// meta[b, 1]`, `adduct = meta[b, 3]`, `ppm_tenths = meta[b, 5]`,
/// `precursor_uncertainty = meta[b, 2]`, with `rows_scored_max =
/// min(rows_scored_max, M)`. Exactly 1 launch, one lane per spectrum.
pub fn formula_window<R: Runtime, E: FloatElem>(
    table: &IdTensor<R>,
    meta: &IdTensor<R>,
    max_error: u32,
    rows_visited_max: u32,
    rows_scored_max: u32,
    out: &FormulaBuffers<R, E>,
) -> Result<()> {
    // Every rank is checked before any dimension is read, so a malformed
    // shape is `Error::Shape` rather than a panic.
    if table.shape().rank() != 2
        || meta.shape().rank() != 2
        || out.window.shape().rank() != 3
        || out.counters.shape().rank() != 2
        || out.top.shape().rank() != 3
        || out.top_log_prob.shape().rank() != 2
        || out.top_count.shape().rank() != 1
    {
        return Err(Error::shape(format!(
            "formula_window needs table [R, 2], meta [B, 8], window [B, M, 2], counters [B, 5], top [B, F, 2], top_log_prob [B, F] and top_count [B], got {} and {} and {} and {} and {} and {} and {}",
            table.shape(),
            meta.shape(),
            out.window.shape(),
            out.counters.shape(),
            out.top.shape(),
            out.top_log_prob.shape(),
            out.top_count.shape()
        )));
    }
    let batch = meta.shape().dim(0);
    let rows = table.shape().dim(0);
    let m = out.window.shape().dim(1);
    let f = out.top.shape().dim(1);
    let want_table: &[usize] = &[rows, 2];
    let want_meta: &[usize] = &[batch, META_WIDTH];
    let want_window: &[usize] = &[batch, m, 2];
    let want_counters: &[usize] = &[batch, 5];
    let want_top: &[usize] = &[batch, f, 2];
    let want_top_lp: &[usize] = &[batch, f];
    let want_top_count: &[usize] = &[batch];
    if table.shape().dims() != want_table
        || meta.shape().dims() != want_meta
        || out.window.shape().dims() != want_window
        || out.counters.shape().dims() != want_counters
        || out.top.shape().dims() != want_top
        || out.top_log_prob.shape().dims() != want_top_lp
        || out.top_count.shape().dims() != want_top_count
    {
        return Err(Error::shape(format!(
            "formula_window needs table [{rows}, 2], meta [{batch}, 8], window [{batch}, {m}, 2], counters [{batch}, 5], top [{batch}, {f}, 2], top_log_prob [{batch}, {f}] and top_count [{batch}], got {} and {} and {} and {} and {} and {} and {}",
            table.shape(),
            meta.shape(),
            out.window.shape(),
            out.counters.shape(),
            out.top.shape(),
            out.top_log_prob.shape(),
            out.top_count.shape()
        )));
    }
    if batch == 0 {
        return Ok(());
    }
    let client = meta.client();
    // Net hydrogen shift of contract §4.3, `m_H - m_e`, from the domain
    // tables rather than a literal.
    let h_net = crate::models::ms2::chem::ELEMENTS[crate::models::ms2::chem::HYDROGEN].mass
        - crate::models::ms2::chem::ELECTRON_MASS;
    let (count, dim, span) = launch_1d_spans(client, batch, rows.max(1));
    unsafe {
        ms2_formula_window_kernel::launch_unchecked::<R>(
            client,
            count,
            dim,
            table.arg(),
            meta.arg(),
            out.window.arg(),
            out.counters.arg(),
            rows,
            m,
            max_error,
            rows_visited_max,
            rows_scored_max,
            h_net,
            request_status::FORMULA_ABSENT,
            request_status::MASS_OVERFLOW,
            request_status::FORMULA_SEARCH_EXHAUSTED,
            request_status::EXACT_MASS_UNAVAILABLE | request_status::FORMULA_ABSENT,
            u32::MAX,
            batch,
            span,
        );
    }
    Ok(())
}

/// Lane per spectrum: the `F` largest `log_prob` entries among the scored
/// candidates whose flag is non-zero, ties by smaller slot, as `F`
/// successive argmax passes (`O(F^2 * M)` per lane: `F` picks, each
/// candidate checks the `F` taken slots; linear in `M` under the
/// contractual `F <= 8`). Reads `log_prob`
/// and `cand` (the source id is `cand[.., 12]`); writes `top`
/// (source id, window slot), `top_log_prob` and `top_count`; unused entries
/// get row and slot `u32::MAX` and log-probability 0. A slot is a candidate
/// when its flag is non-zero, its score lies in the validated domain
/// `-FINITE_MAX < s < FINITE_MAX` (the crate's fast-math-safe finiteness
/// rule, the same constant and `>`/`<` comparison form as the other MS2
/// kernels), and it was not chosen by an earlier pick (compared against the
/// slots already written to `top`). Each pick scans the slots once in
/// increasing order, streaming its best into its own output slots with a
/// strict `>` (ties keep the smaller slot); the pick stays empty when no
/// candidate remains, so non-empty picks form a dense prefix.
/// `top_count` is the number of non-empty picks, so every slot below it is
/// a real row and `k mod top_count` never selects padding. All loads use
/// unconditional indices with the mask applied at use. Arrays: `log_prob`,
/// `cand`, `top`, `top_log_prob`, `top_count`.
#[allow(clippy::too_many_arguments)]
#[allow(unused_assignments)]
#[cube(launch_unchecked)]
fn ms2_formula_top_kernel<F: Float + CubeElement>(
    log_prob: &Array<F>,
    cand: &Array<u32>,
    top: &mut Array<u32>,
    top_log_prob: &mut Array<F>,
    top_count: &mut Array<u32>,
    m: usize,
    f: usize,
    max_u32: u32,
    lanes: usize,
    span: usize,
) {
    let zero = F::new(0.0_f32);
    let finite_max = F::new(FINITE_MAX);
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    for pos in start..end {
        let b = pos;
        // `F` successive argmax passes, `O(F^2 * M)` per lane (each of the
        // `F` picks scans `M` candidates and checks the `F` taken slots;
        // linear in `M` under the contractual `F <= 8`). Each pick
        // streams its best directly into its output slots (`top`,
        // `top_log_prob`): the scan carries NO register state across `mm`
        // iterations (only fresh per-slot flags), which is what lowers to
        // the backend IR on the pinned toolchain. All buffer loads below
        // use unconditional indices with the mask applied at use, so no
        // load sits inside a branch.
        let mut written: u32 = 0u32;
        for ff in 0..f {
            let ffu = ff as u32;
            // Start this pick empty (padding values); the scan below
            // overwrites them when a candidate is taken.
            top[(b * f + ff) * 2] = max_u32;
            top[(b * f + ff) * 2 + 1] = max_u32;
            top_log_prob[b * f + ff] = zero;
            for mm in 0..m {
                // Unconditional loads at valid indices (`mm < m`), masked
                // at first use. The source id loads here too, stored only
                // on a take below.
                let flag_mm = cand[(b * m + mm) * 13 + 11];
                let score_mm = log_prob[b * m + mm];
                let src_mm = cand[(b * m + mm) * 13 + 12];
                let mm_u32 = mm as u32;
                let mut flagged: bool = false;
                if flag_mm != 0u32 {
                    flagged = true;
                }
                // Validated domain `-FINITE_MAX < s < FINITE_MAX`: the same
                // constant and `>`/`<` comparison form as the other MS2
                // kernels, false for NaN, infinities and out-of-domain
                // scores, so such a slot is never selected. The lower bound
                // runs as `0 - s < FINITE_MAX` (negation is exact, so this
                // is exactly `s > -FINITE_MAX`).
                let mut score_ok: bool = false;
                let neg_score = zero - score_mm;
                if score_mm < finite_max {
                    if neg_score < finite_max {
                        score_ok = true;
                    }
                }
                // Already chosen by an earlier pick: compare its slot with
                // the picks below `ffu` (a miss empties the remaining
                // candidate set, so later picks miss too). The prior picks
                // load unconditionally with the mask applied at use; the
                // current pick slot stays masked out, so reads and writes
                // never overlap.
                let mut taken: bool = false;
                for pp in 0..f {
                    let ppu = pp as u32;
                    let prior = top[(b * f + pp) * 2 + 1];
                    if ppu < ffu {
                        if prior == mm_u32 {
                            taken = true;
                        }
                    }
                }
                let mut not_taken: bool = true;
                if taken {
                    not_taken = false;
                }
                let mut candidate: bool = false;
                if flagged && score_ok && not_taken {
                    candidate = true;
                }
                // Current best of this pick, reloaded from its output slots
                // (array reads, no carried registers). Empty while the slot
                // still holds the sentinel, in which case the first
                // candidate is always taken; later ones only on a strict
                // `>`, which breaks ties by smaller slot in increasing
                // scan order.
                let cur_slot = top[(b * f + ff) * 2 + 1];
                let cur_lp = top_log_prob[b * f + ff];
                let mut take: bool = false;
                if candidate {
                    if cur_slot == max_u32 {
                        take = true;
                    }
                }
                if candidate {
                    if cur_slot != max_u32 {
                        if score_mm > cur_lp {
                            take = true;
                        }
                    }
                }
                // Stream the new best into the pick's output slots.
                if take {
                    top[(b * f + ff) * 2] = src_mm;
                }
                if take {
                    top[(b * f + ff) * 2 + 1] = mm_u32;
                }
                if take {
                    top_log_prob[b * f + ff] = score_mm;
                }
            }
            // A pick that took nothing still holds the sentinel slot. The
            // count is the number of non-empty picks, read back from the
            // output slot (array read, no carried registers), so the
            // written entries compact densely and every slot below
            // `top_count` is a real row.
            let pick_slot = top[(b * f + ff) * 2 + 1];
            if pick_slot != max_u32 {
                written += 1u32;
            }
        }
        top_count[b] = written;
        for ff in (written as usize)..f {
            let base = (b * f + ff) * 2;
            top[base] = max_u32;
            top[base + 1] = max_u32;
            top_log_prob[b * f + ff] = zero;
        }
    }
}

/// Run top-F selection (architecture §3.4, V1 §1.2) from `log_prob`
/// (`[B, M]`) and `cand` (`[B, M, 13]`, source id at `[.., 12]`, flag at
/// `[.., 11]`) into `out.top`, `out.top_log_prob` and `out.top_count`.
/// `F` successive argmax passes (`O(F^2 * M)` per lane: `F` picks, each
/// candidate checks the `F` taken slots; linear in `M` under the contractual
/// `F <= 8`): a slot is a candidate
/// when its flag is non-zero, its score lies in `-FINITE_MAX < s <
/// FINITE_MAX`, and it was not chosen by an earlier pick; ties break by
/// smaller slot. A score outside the validated domain is never selected.
/// `top_count` is the number of picks written; the written entries compact
/// densely so every slot below `top_count` is a real row. Exactly 1 launch,
/// one lane per spectrum.
pub fn formula_top<R: Runtime, E: FloatElem>(
    log_prob: &Tensor<R, E>,
    cand: &IdTensor<R>,
    out: &FormulaBuffers<R, E>,
) -> Result<()> {
    // Every rank is checked before any dimension is read, so a malformed
    // shape is `Error::Shape` rather than a panic.
    if log_prob.shape().rank() != 2
        || cand.shape().rank() != 3
        || out.top.shape().rank() != 3
        || out.top_log_prob.shape().rank() != 2
        || out.top_count.shape().rank() != 1
    {
        return Err(Error::shape(format!(
            "formula_top needs log_prob [B, M], cand [B, M, 13], top [B, F, 2], top_log_prob [B, F] and top_count [B], got {} and {} and {} and {} and {}",
            log_prob.shape(),
            cand.shape(),
            out.top.shape(),
            out.top_log_prob.shape(),
            out.top_count.shape()
        )));
    }
    let batch = log_prob.shape().dim(0);
    let m = log_prob.shape().dim(1);
    let f = out.top.shape().dim(1);
    let want_cand: &[usize] = &[batch, m, 13];
    let want_top: &[usize] = &[batch, f, 2];
    let want_top_lp: &[usize] = &[batch, f];
    let want_top_count: &[usize] = &[batch];
    if cand.shape().dims() != want_cand
        || out.top.shape().dims() != want_top
        || out.top_log_prob.shape().dims() != want_top_lp
        || out.top_count.shape().dims() != want_top_count
    {
        return Err(Error::shape(format!(
            "formula_top needs cand [{batch}, {m}, 13], top [{batch}, {f}, 2], top_log_prob [{batch}, {f}] and top_count [{batch}], got {} and {} and {} and {}",
            cand.shape(),
            out.top.shape(),
            out.top_log_prob.shape(),
            out.top_count.shape()
        )));
    }
    if batch == 0 {
        return Ok(());
    }
    let client = log_prob.client();
    let (count, dim, span) = launch_1d_spans(client, batch, m.max(1) * f.max(1));
    unsafe {
        ms2_formula_top_kernel::launch_unchecked::<E, R>(
            client,
            count,
            dim,
            log_prob.arg(),
            cand.arg(),
            out.top.arg(),
            out.top_log_prob.arg(),
            out.top_count.arg(),
            m,
            f,
            u32::MAX,
            batch,
            span,
        );
    }
    Ok(())
}

/// Lane per `(b, m)`: write padding for an empty table (V1 §1.2, `rows == 0`).
/// No table element is loaded and no table array is bound: every one of the
/// 13 words of every candidate is written as padding (all `0` except source
/// `u32::MAX`). Arrays: `cand` (1).
#[cube(launch_unchecked)]
fn ms2_formula_gather_empty_kernel(cand: &mut Array<u32>, max_u32: u32, lanes: usize, span: usize) {
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    for pos in start..end {
        let base = pos * 13;
        for e in 0..10usize {
            cand[base + e] = 0u32;
        }
        cand[base + 10] = 0u32;
        cand[base + 11] = 0u32;
        cand[base + 12] = max_u32;
    }
}

/// Lane per `(b, m)`: gather the scored candidate composition from the table
/// (V1 §1.2, table source only). `window` is `[B, M, 2]` (row, flag),
/// `table` is `[R, 2]` (mass, bound), `table_counts` is `[R, 10]`; `cand` is
/// `[B, M, 13]` (10 counts, mass, flag, source). A padding slot is all `0`
/// except source `u32::MAX`. Arrays: `window`, `table`, `table_counts`,
/// `cand` (4).
#[cube(launch_unchecked)]
fn ms2_formula_gather_kernel(
    window: &Array<u32>,
    table: &Array<u32>,
    table_counts: &Array<u32>,
    cand: &mut Array<u32>,
    rows: usize,
    m: usize,
    max_u32: u32,
    lanes: usize,
    span: usize,
) {
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    for pos in start..end {
        let b = pos / m;
        let mm = pos % m;
        let row = window[(b * m + mm) * 2];
        let flag = window[(b * m + mm) * 2 + 1];
        let mut ok = false;
        if row != max_u32 {
            if flag != 0u32 {
                if (row as usize) < rows {
                    ok = true;
                }
            }
        }
        // Safe index outside the branch (RADV): load unconditionally, mask
        // at first use.
        let mut safe = 0usize;
        if ok {
            safe = row as usize;
        }
        let mass_safe = table[safe * 2];
        let base = (b * m + mm) * 13;
        if ok {
            for e in 0..10usize {
                cand[base + e] = table_counts[safe * 10 + e];
            }
            cand[base + 10] = mass_safe;
            cand[base + 11] = flag;
            cand[base + 12] = row;
        } else {
            for e in 0..10usize {
                cand[base + e] = 0u32;
            }
            cand[base + 10] = 0u32;
            cand[base + 11] = 0u32;
            cand[base + 12] = max_u32;
        }
    }
}

/// Run [`ms2_formula_gather_kernel`]: table source only. Exactly 1 launch,
/// one lane per `(b, m)`.
pub fn formula_gather<R: Runtime>(
    window: &IdTensor<R>,
    table: &IdTensor<R>,
    table_counts: &IdTensor<R>,
    cand: &mut IdTensor<R>,
) -> Result<()> {
    if window.shape().rank() != 3
        || table.shape().rank() != 2
        || table_counts.shape().rank() != 2
        || cand.shape().rank() != 3
    {
        return Err(Error::shape(format!(
            "formula_gather needs window [B, M, 2], table [R, 2], table_counts [R, 10] and cand [B, M, 13], got {} and {} and {} and {}",
            window.shape(),
            table.shape(),
            table_counts.shape(),
            cand.shape()
        )));
    }
    let batch = window.shape().dim(0);
    let m = window.shape().dim(1);
    let rows = table.shape().dim(0);
    if table.shape().dims() != [rows, 2]
        || table_counts.shape().dims() != [rows, 10]
        || window.shape().dims() != [batch, m, 2]
        || cand.shape().dims() != [batch, m, 13]
    {
        return Err(Error::shape(format!(
            "formula_gather needs window [{batch}, {m}, 2], table [{rows}, 2], table_counts [{rows}, 10] and cand [{batch}, {m}, 13], got {} and {} and {} and {}",
            window.shape(),
            table.shape(),
            table_counts.shape(),
            cand.shape()
        )));
    }
    if batch == 0 || m == 0 {
        return Ok(());
    }
    // Empty table (`rows == 0`): no table element may be loaded on any
    // backend, and a zero-length array is never bound to a kernel that
    // indexes it. The padding-only kernel above binds `cand` alone and
    // writes all 13 words of every candidate as padding.
    if rows == 0 {
        let client = window.client();
        let lanes = batch * m;
        let (count, dim, span) = launch_1d_spans(client, lanes, 13);
        unsafe {
            ms2_formula_gather_empty_kernel::launch_unchecked::<R>(
                client,
                count,
                dim,
                cand.arg(),
                u32::MAX,
                lanes,
                span,
            );
        }
        return Ok(());
    }
    let client = window.client();
    let lanes = batch * m;
    let (count, dim, span) = launch_1d_spans(client, lanes, 13);
    unsafe {
        ms2_formula_gather_kernel::launch_unchecked::<R>(
            client,
            count,
            dim,
            window.arg(),
            table.arg(),
            table_counts.arg(),
            cand.arg(),
            rows,
            m,
            u32::MAX,
            lanes,
            span,
        );
    }
    Ok(())
}

/// Lane per output element `(record, e)`: `out[r, e] = log_table[count]`
/// where `count` is the `e`-th of the record's first 10 words (V1 §1.2).
/// `records` is a `u32` record buffer of any record width `w >= 10` (a scalar
/// argument; `cand` with `w = 13`, `gold_counts` with `w = 10`); `log_table`
/// is `[1024]` with `log_table[n] = ln(1 + n)` uploaded once, so the feature
/// is the same bits as V0's uploaded table features. A count above 1023
/// cannot occur (host-validated); an out-of-range count writes `0` without
/// reading out of bounds. Arrays: `records`, `log_table`, `out` (3).
#[cube(launch_unchecked)]
fn ms2_count_features_kernel<F: Float + CubeElement>(
    records: &Array<u32>,
    log_table: &Array<F>,
    out: &mut Array<F>,
    width: usize,
    lanes: usize,
    span: usize,
) {
    let zero = F::new(0.0_f32);
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    for pos in start..end {
        let r = pos / 10;
        let e = pos % 10;
        let count = records[r * width + e];
        let mut ok = false;
        if count < 1024u32 {
            ok = true;
        }
        let mut safe = 0usize;
        if ok {
            safe = count as usize;
        }
        let v = log_table[safe];
        if ok {
            out[pos] = v;
        } else {
            out[pos] = zero;
        }
    }
}

/// Run [`ms2_count_features_kernel`]. `records` is `[records_n, w]` with
/// `w >= 10`; `log_table` is `[1024]`; `out` is `[records_n, 10]`. Exactly 1
/// launch, one lane per output element.
pub fn count_features<R: Runtime, E: FloatElem>(
    records: &IdTensor<R>,
    log_table: &Tensor<R, E>,
    out: &mut Tensor<R, E>,
    width: usize,
) -> Result<()> {
    if records.shape().rank() != 2 || log_table.shape().rank() != 1 || out.shape().rank() != 2 {
        return Err(Error::shape(format!(
            "count_features needs records [N, w], log_table [1024] and out [N, 10], got {} and {} and {}",
            records.shape(),
            log_table.shape(),
            out.shape()
        )));
    }
    let records_n = records.shape().dim(0);
    let w = records.shape().dim(1);
    if w != width || width < 10 {
        return Err(Error::shape(format!(
            "count_features needs record width w >= 10 (got w = {w}, width arg = {width})"
        )));
    }
    if log_table.len() != 1024 {
        return Err(Error::shape(format!(
            "count_features needs log_table [1024], got {}",
            log_table.shape()
        )));
    }
    if out.shape().dims() != [records_n, 10] {
        return Err(Error::shape(format!(
            "count_features needs out [{records_n}, 10], got {}",
            out.shape()
        )));
    }
    if out.is_empty() {
        return Ok(());
    }
    let lanes = out.len();
    let (count, dim, span) = launch_1d_spans(records.client(), lanes, 1);
    unsafe {
        ms2_count_features_kernel::launch_unchecked::<E, R>(
            records.client(),
            count,
            dim,
            records.arg(),
            log_table.arg(),
            out.arg(),
            width,
            lanes,
            span,
        );
    }
    Ok(())
}

/// Lane per `(b, f)`: copy the retained formula's counts from `cand`
/// (V1 §1.2). `top` is `[B, F, 2]` (source id, window slot), `cand` is
/// `[B, M, 13]`; `top_counts` is `[B, F, 10]` (`0` in padding). Arrays: `top`,
/// `cand`, `top_counts` (3).
#[cube(launch_unchecked)]
fn ms2_formula_top_counts_kernel(
    top: &Array<u32>,
    cand: &Array<u32>,
    top_counts: &mut Array<u32>,
    m: usize,
    f: usize,
    max_u32: u32,
    lanes: usize,
    span: usize,
) {
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    for pos in start..end {
        let b = pos / f;
        let ff = pos % f;
        let slot = top[(b * f + ff) * 2 + 1];
        let mut ok = false;
        if slot != max_u32 {
            if (slot as usize) < m {
                ok = true;
            }
        }
        let mut safe = 0usize;
        if ok {
            safe = slot as usize;
        }
        // The candidate base uses the safe slot; padding writes zeros.
        let cbase = (b * m + safe) * 13;
        let obase = (b * f + ff) * 10;
        if ok {
            for e in 0..10usize {
                top_counts[obase + e] = cand[cbase + e];
            }
        } else {
            for e in 0..10usize {
                top_counts[obase + e] = 0u32;
            }
        }
    }
}

/// Run [`ms2_formula_top_counts_kernel`]. Exactly 1 launch, one lane per
/// `(b, f)`.
pub fn formula_top_counts<R: Runtime>(
    top: &IdTensor<R>,
    cand: &IdTensor<R>,
    top_counts: &mut IdTensor<R>,
) -> Result<()> {
    if top.shape().rank() != 3 || cand.shape().rank() != 3 || top_counts.shape().rank() != 3 {
        return Err(Error::shape(format!(
            "formula_top_counts needs top [B, F, 2], cand [B, M, 13] and top_counts [B, F, 10], got {} and {} and {}",
            top.shape(),
            cand.shape(),
            top_counts.shape()
        )));
    }
    let batch = top.shape().dim(0);
    let f = top.shape().dim(1);
    let m = cand.shape().dim(1);
    if top.shape().dims() != [batch, f, 2]
        || cand.shape().dims() != [batch, m, 13]
        || top_counts.shape().dims() != [batch, f, 10]
    {
        return Err(Error::shape(format!(
            "formula_top_counts needs top [{batch}, {f}, 2], cand [{batch}, {m}, 13] and top_counts [{batch}, {f}, 10], got {} and {} and {}",
            top.shape(),
            cand.shape(),
            top_counts.shape()
        )));
    }
    if batch == 0 || f == 0 {
        return Ok(());
    }
    let client = top.client();
    let lanes = batch * f;
    let (count, dim, span) = launch_1d_spans(client, lanes, 10);
    unsafe {
        ms2_formula_top_counts_kernel::launch_unchecked::<R>(
            client,
            count,
            dim,
            top.arg(),
            cand.arg(),
            top_counts.arg(),
            m,
            f,
            u32::MAX,
            lanes,
            span,
        );
    }
    Ok(())
}

/// Lane per spectrum: the first scored slot whose 10 counts equal the gold
/// composition (V1 §1.2, training and teacher-forced evaluation only).
/// `cand` is `[B, M, 13]`, `gold_counts` is `[B, 10]`; `gold_slot` is `[B]`
/// (the slot, else `u32::MAX`). Only flagged slots (`flag != 0`, i.e. below
/// `rows_scored`) are considered, so padding (all `0`, flag `0`) never
/// matches. Arrays: `cand`, `gold_counts`, `gold_slot` (3).
#[cube(launch_unchecked)]
fn ms2_gold_slot_kernel(
    cand: &Array<u32>,
    gold_counts: &Array<u32>,
    gold_slot: &mut Array<u32>,
    m: usize,
    lanes: usize,
    span: usize,
) {
    let sentinel = 4294967295u32;
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    for pos in start..end {
        let b = pos;
        let mut best = 4294967295u32;
        let mut found = false;
        for mm in 0..m {
            let flag = cand[(b * m + mm) * 13 + 11];
            if flag != 0u32 {
                if !found {
                    let mut eq = true;
                    for e in 0..10usize {
                        if cand[(b * m + mm) * 13 + e] != gold_counts[b * 10 + e] {
                            eq = false;
                        }
                    }
                    if eq {
                        best = mm as u32;
                        found = true;
                    }
                }
            }
        }
        if found {
            gold_slot[b] = best;
        } else {
            gold_slot[b] = sentinel;
        }
    }
}

/// Run [`ms2_gold_slot_kernel`]. Exactly 1 launch, one lane per spectrum.
pub fn gold_slot<R: Runtime>(
    cand: &IdTensor<R>,
    gold_counts: &IdTensor<R>,
    gold_slot_out: &mut IdTensor<R>,
) -> Result<()> {
    if cand.shape().rank() != 3
        || gold_counts.shape().rank() != 2
        || gold_slot_out.shape().rank() != 1
    {
        return Err(Error::shape(format!(
            "gold_slot needs cand [B, M, 13], gold_counts [B, 10] and gold_slot [B], got {} and {} and {}",
            cand.shape(),
            gold_counts.shape(),
            gold_slot_out.shape()
        )));
    }
    let batch = cand.shape().dim(0);
    let m = cand.shape().dim(1);
    if cand.shape().dims() != [batch, m, 13]
        || gold_counts.shape().dims() != [batch, 10]
        || gold_slot_out.len() != batch
    {
        return Err(Error::shape(format!(
            "gold_slot needs cand [{batch}, {m}, 13], gold_counts [{batch}, 10] and gold_slot [{batch}], got {} and {} and {}",
            cand.shape(),
            gold_counts.shape(),
            gold_slot_out.shape()
        )));
    }
    if batch == 0 {
        return Ok(());
    }
    let client = cand.client();
    let (count, dim, span) = launch_1d_spans(client, batch, m.max(1));
    unsafe {
        ms2_gold_slot_kernel::launch_unchecked::<R>(
            client,
            count,
            dim,
            cand.arg(),
            gold_counts.arg(),
            gold_slot_out.arg(),
            m,
            batch,
            span,
        );
    }
    Ok(())
}

/// Lane per output element: `1.0` where the window flag is non-zero, `0.0`
/// where it is zero, so flags 1 (accept) and 2 (ambiguous) both become 1
/// (architecture §3.8).
#[cube(launch_unchecked)]
fn ms2_nonzero_mask_kernel<F: Float + CubeElement>(
    window: &Array<u32>,
    out: &mut Array<F>,
    lanes: usize,
    span: usize,
) {
    let zero = F::new(0.0_f32);
    let one = F::new(1.0_f32);
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    for pos in start..end {
        let mut v = zero;
        if window[pos * 2 + 1] != 0u32 {
            v = one;
        }
        out[pos] = v;
    }
}

/// A `[B, M]` float mask that is 1 where `window[b, m, 1]` is non-zero:
/// the formula head's gate over the joined window slots. One launch.
pub fn nonzero_mask<R: Runtime, E: FloatElem>(window: &IdTensor<R>) -> Result<Tensor<R, E>> {
    if window.shape().rank() != 3 {
        return Err(Error::shape(format!(
            "nonzero_mask needs window [B, M, 2], got {}",
            window.shape()
        )));
    }
    let batch = window.shape().dim(0);
    let m = window.shape().dim(1);
    let want: &[usize] = &[batch, m, 2];
    if window.shape().dims() != want {
        return Err(Error::shape(format!(
            "nonzero_mask needs window [{batch}, {m}, 2], got {}",
            window.shape()
        )));
    }
    let out = Tensor::empty(Shape::new(vec![batch, m]), window.device());
    if out.is_empty() {
        return Ok(out);
    }
    let lanes = out.len();
    let (count, dim, span) = launch_1d_spans(window.client(), lanes, 1);
    unsafe {
        ms2_nonzero_mask_kernel::launch_unchecked::<E, R>(
            window.client(),
            count,
            dim,
            window.arg(),
            out.arg(),
            lanes,
            span,
        );
    }
    Ok(out)
}

/// Lane per output element: `1.0` where the `cand` flag (`[.., 11]`) is
/// non-zero, `0.0` where it is zero (V1 §1.2, architecture §3.8).
#[cube(launch_unchecked)]
fn ms2_cand_mask_kernel<F: Float + CubeElement>(
    cand: &Array<u32>,
    out: &mut Array<F>,
    lanes: usize,
    span: usize,
) {
    let zero = F::new(0.0_f32);
    let one = F::new(1.0_f32);
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    for pos in start..end {
        let mut v = zero;
        if cand[pos * 13 + 11] != 0u32 {
            v = one;
        }
        out[pos] = v;
    }
}

/// Write the `cand` join mask into caller-provided `out` (`[B, M]`
/// floats): the launch path behind [`cand_mask`]. Tests poison `out` first
/// (NaN floats), so a lane the kernel skips fails the every-element
/// comparison against the host twin. One launch.
pub fn cand_mask_into<R: Runtime, E: FloatElem>(
    cand: &IdTensor<R>,
    out: &mut Tensor<R, E>,
) -> Result<()> {
    if cand.shape().rank() != 3 || cand.shape().dim(2) != 13 {
        return Err(Error::shape(format!(
            "cand_mask needs cand [B, M, 13], got {}",
            cand.shape()
        )));
    }
    let batch = cand.shape().dim(0);
    let m = cand.shape().dim(1);
    let want: &[usize] = &[batch, m, 13];
    if cand.shape().dims() != want {
        return Err(Error::shape(format!(
            "cand_mask needs cand [{batch}, {m}, 13], got {}",
            cand.shape()
        )));
    }
    let want_out: &[usize] = &[batch, m];
    if out.shape().dims() != want_out {
        return Err(Error::shape(format!(
            "cand_mask needs out [{batch}, {m}], got {}",
            out.shape()
        )));
    }
    if out.is_empty() {
        return Ok(());
    }
    let lanes = out.len();
    let (count, dim, span) = launch_1d_spans(cand.client(), lanes, 1);
    unsafe {
        ms2_cand_mask_kernel::launch_unchecked::<E, R>(
            cand.client(),
            count,
            dim,
            cand.arg(),
            out.arg(),
            lanes,
            span,
        );
    }
    Ok(())
}

/// A `[B, M]` float mask that is 1 where `cand[b, m, 11]` is non-zero:
/// the formula head's gate over the scored candidates. One launch.
pub fn cand_mask<R: Runtime, E: FloatElem>(cand: &IdTensor<R>) -> Result<Tensor<R, E>> {
    if cand.shape().rank() != 3 || cand.shape().dim(2) != 13 {
        return Err(Error::shape(format!(
            "cand_mask needs cand [B, M, 13], got {}",
            cand.shape()
        )));
    }
    let batch = cand.shape().dim(0);
    let m = cand.shape().dim(1);
    let want: &[usize] = &[batch, m, 13];
    if cand.shape().dims() != want {
        return Err(Error::shape(format!(
            "cand_mask needs cand [{batch}, {m}, 13], got {}",
            cand.shape()
        )));
    }
    let mut out = Tensor::empty(Shape::new(vec![batch, m]), cand.device());
    cand_mask_into(cand, &mut out)?;
    Ok(out)
}

// ---------------------------------------------------------------------------
// Graph legality and replay (architecture §3.5)
// ---------------------------------------------------------------------------

/// Fixed words per grammar state row beyond the three per-atom words: atoms,
/// last parent, last close + 1 (0 = none), closures, step, stopped, and the
/// 10 used element counts.
pub const STATE_WIDTH_BASE: usize = 16;

/// Words of one grammar state row for `max_atoms = a`: `3 * a + 16`.
pub fn replay_state_width(max_atoms: usize) -> usize {
    3 * max_atoms + STATE_WIDTH_BASE
}

/// Device twin of [`crate::models::ms2::grammar::TraceState`]: the legality
/// rules of contract §4.4 as `#[cube]` functions over one state row, shared
/// by training replay (§3.5) and the sampler (§3.6).
///
/// State row layout for `a` atoms (`S = 3a + 16` words): `0..a` atom type,
/// `a..2a` residual valence, `2a..3a` parent pointer (`u32::MAX` for the
/// root), then counters `atoms`, `last parent`, `last close + 1` (0 = none),
/// `closures`, `step`, `stopped`, and 10 used element counts in
/// [`crate::models::ms2::chem::ELEMENTS`] order.
///
/// Adjacency is not stored: a bond exists exactly when one atom is the
/// other's parent or a CLOSE_RING named it, and the grammar makes duplicate
/// bonds impossible by construction (an ADD creates a fresh atom, and a
/// closure from the newest atom must point above its parent and above its
/// previous closure, so neither end can already be bonded). The
/// already-bonded checks of `TraceState` therefore never fire on a prefix the
/// legality rules admit, and the kernel keeps the same rule set without a
/// bond set.
///
/// A module of its own only so that one `missing_docs` allow covers it.
#[allow(missing_docs)]
pub mod grammar {
    use cubecl::prelude::*;

    #[cube]
    pub fn n_atoms(state: &mut Array<u32>, sbase: usize, atoms_n: usize) -> u32 {
        state[sbase + 3 * atoms_n]
    }

    #[cube]
    pub fn last_parent(state: &mut Array<u32>, sbase: usize, atoms_n: usize) -> u32 {
        state[sbase + 3 * atoms_n + 1]
    }

    #[cube]
    pub fn last_close_p1(state: &mut Array<u32>, sbase: usize, atoms_n: usize) -> u32 {
        state[sbase + 3 * atoms_n + 2]
    }

    #[cube]
    pub fn closures(state: &mut Array<u32>, sbase: usize, atoms_n: usize) -> u32 {
        state[sbase + 3 * atoms_n + 3]
    }

    #[cube]
    pub fn step_of(state: &mut Array<u32>, sbase: usize, atoms_n: usize) -> u32 {
        state[sbase + 3 * atoms_n + 4]
    }

    #[cube]
    pub fn stopped(state: &mut Array<u32>, sbase: usize, atoms_n: usize) -> u32 {
        state[sbase + 3 * atoms_n + 5]
    }

    #[cube]
    pub fn used_of(state: &mut Array<u32>, sbase: usize, atoms_n: usize, e: usize) -> u32 {
        state[sbase + 3 * atoms_n + 6 + e]
    }

    #[cube]
    pub fn resid_of(state: &mut Array<u32>, sbase: usize, atoms_n: usize, j: usize) -> u32 {
        state[sbase + atoms_n + j]
    }

    #[cube]
    pub fn budget_of(meta: &Array<u32>, mbase: usize, e: usize) -> u32 {
        meta[mbase + 2 + e]
    }

    /// Whether atom type `id` fits the remaining formula budget
    /// (`TraceState::type_fits`), as 1/0: without a budget every valid id fits.
    #[cube]
    pub fn type_fits(
        state: &mut Array<u32>,
        sbase: usize,
        atypes: &Array<u32>,
        meta: &Array<u32>,
        mbase: usize,
        atoms_n: usize,
        id: u32,
    ) -> u32 {
        let mut fits = 0u32;
        if id >= 1u32 {
            if id <= 17u32 {
                fits = 1u32;
                if meta[mbase + 1] != 0u32 {
                    let elem = atypes[id as usize * 3] as usize;
                    let hyd = atypes[id as usize * 3 + 1];
                    if used_of(state, sbase, atoms_n, elem) >= budget_of(meta, mbase, elem) {
                        fits = 0u32;
                    }
                    if used_of(state, sbase, atoms_n, 1) + hyd > budget_of(meta, mbase, 1) {
                        fits = 0u32;
                    }
                }
            }
        }
        fits
    }

    /// Atom types the root may take (`TraceState::root_types`).
    #[cube]
    pub fn root_types(
        state: &mut Array<u32>,
        sbase: usize,
        atypes: &Array<u32>,
        meta: &Array<u32>,
        mbase: usize,
        atoms_n: usize,
    ) -> u32 {
        let mut types = 0u32;
        for id in 1usize..18usize {
            if type_fits(state, sbase, atypes, meta, mbase, atoms_n, id as u32) != 0u32 {
                types |= 1u32 << (id as u32);
            }
        }
        types
    }

    /// Pointers a non-root child may take for bond order `bond`
    /// (`TraceState::add_pointers`).
    #[cube]
    pub fn add_pointers(
        state: &mut Array<u32>,
        sbase: usize,
        atoms_n: usize,
        bond: u32,
        max_atoms: u32,
    ) -> u32 {
        let mut mask = 0u32;
        let n = n_atoms(state, sbase, atoms_n) as usize;
        if n > 0 && (n as u32) < max_atoms {
            let lp = last_parent(state, sbase, atoms_n) as usize;
            for p in lp..n {
                if resid_of(state, sbase, atoms_n, p) >= bond {
                    mask |= 1u32 << (p as u32);
                }
            }
        }
        mask
    }

    /// Whether any non-root child can be added (`TraceState::has_add`).
    #[cube]
    pub fn has_add(
        state: &mut Array<u32>,
        sbase: usize,
        atypes: &Array<u32>,
        meta: &Array<u32>,
        mbase: usize,
        atoms_n: usize,
        max_atoms: u32,
    ) -> u32 {
        let mut has = 0u32;
        for id in 1usize..18usize {
            if type_fits(state, sbase, atypes, meta, mbase, atoms_n, id as u32) != 0u32 {
                let val = atypes[id * 3 + 2];
                let hyd = atypes[id * 3 + 1];
                for bond in 1u32..4u32 {
                    if val >= hyd && bond <= val - hyd {
                        if add_pointers(state, sbase, atoms_n, bond, max_atoms) != 0u32 {
                            has = 1u32;
                        }
                    }
                }
            }
        }
        has
    }

    /// Pointers a ring closure of order `bond` may take on the newest atom
    /// (`TraceState::close_pointers`, without the bond set: duplicates are
    /// impossible by the ordering rule, see the module docs).
    #[cube]
    pub fn close_pointers(
        state: &mut Array<u32>,
        sbase: usize,
        atoms_n: usize,
        bond: u32,
        max_closures: u32,
        sentinel: u32,
    ) -> u32 {
        let mut mask = 0u32;
        let n = n_atoms(state, sbase, atoms_n) as usize;
        if n >= 2 && closures(state, sbase, atoms_n) < max_closures {
            let newest = n - 1;
            if resid_of(state, sbase, atoms_n, newest) >= bond {
                let par = state[sbase + 2 * atoms_n + newest];
                let mut pp1 = 0u32;
                if par != sentinel {
                    pp1 = par + 1u32;
                }
                let lc = last_close_p1(state, sbase, atoms_n);
                let mut lo = pp1;
                if lc > lo {
                    lo = lc;
                }
                for p in (lo as usize)..newest {
                    if resid_of(state, sbase, atoms_n, p) >= bond {
                        mask |= 1u32 << (p as u32);
                    }
                }
            }
        }
        mask
    }

    /// Whether any ring closure is available (`TraceState::has_close`).
    #[cube]
    pub fn has_close(
        state: &mut Array<u32>,
        sbase: usize,
        atoms_n: usize,
        max_closures: u32,
        sentinel: u32,
    ) -> u32 {
        let mut has = 0u32;
        for bond in 1u32..4u32 {
            if close_pointers(state, sbase, atoms_n, bond, max_closures, sentinel) != 0u32 {
                has = 1u32;
            }
        }
        has
    }

    /// The kind mask at this step (`TraceState::kind_mask`).
    #[cube]
    pub fn kind_mask(
        state: &mut Array<u32>,
        sbase: usize,
        atypes: &Array<u32>,
        meta: &Array<u32>,
        mbase: usize,
        atoms_n: usize,
        max_atoms: u32,
        max_closures: u32,
        sentinel: u32,
    ) -> u32 {
        let mut kinds = 0u32;
        if stopped(state, sbase, atoms_n) == 0u32 {
            let step = step_of(state, sbase, atoms_n);
            if step == 0u32 {
                kinds = 1u32 << 1u32;
            } else if step == 1u32 {
                if root_types(state, sbase, atypes, meta, mbase, atoms_n) != 0u32 {
                    kinds = 1u32 << 2u32;
                }
            } else {
                kinds = 1u32 << 4u32;
                if has_add(state, sbase, atypes, meta, mbase, atoms_n, max_atoms) != 0u32 {
                    kinds |= 1u32 << 2u32;
                }
                if has_close(state, sbase, atoms_n, max_closures, sentinel) != 0u32 {
                    kinds |= 1u32 << 3u32;
                }
            }
        }
        kinds
    }

    /// The four legality masks before a token, conditioned on the earlier
    /// fields of the taken token (`TraceState::masks`).
    #[allow(clippy::too_many_arguments)]
    #[cube]
    pub fn replay_masks(
        state: &mut Array<u32>,
        sbase: usize,
        atypes: &Array<u32>,
        meta: &Array<u32>,
        mbase: usize,
        atoms_n: usize,
        max_atoms: u32,
        max_closures: u32,
        sentinel: u32,
        k: u32,
        steps: u32,
        b: u32,
        out: &mut Array<u32>,
        obase: usize,
    ) {
        if stopped(state, sbase, atoms_n) != 0u32 {
            out[obase] = 0u32;
            out[obase + 1] = 0u32;
            out[obase + 2] = 0u32;
            out[obase + 3] = 0u32;
        } else if step_of(state, sbase, atoms_n) == 0u32 {
            out[obase] = 1u32 << 1u32;
            out[obase + 1] = 0u32;
            out[obase + 2] = 0u32;
            out[obase + 3] = 0u32;
        } else if step_of(state, sbase, atoms_n) == 1u32 {
            out[obase] = kind_mask(
                state,
                sbase,
                atypes,
                meta,
                mbase,
                atoms_n,
                max_atoms,
                max_closures,
                sentinel,
            );
            out[obase + 1] = root_types(state, sbase, atypes, meta, mbase, atoms_n);
            out[obase + 2] = 0u32;
            out[obase + 3] = 0u32;
        } else {
            let kinds = kind_mask(
                state,
                sbase,
                atypes,
                meta,
                mbase,
                atoms_n,
                max_atoms,
                max_closures,
                sentinel,
            );
            out[obase] = kinds;
            out[obase + 1] = 0u32;
            out[obase + 2] = 0u32;
            out[obase + 3] = 0u32;
            if k == 2u32 {
                if has_add(state, sbase, atypes, meta, mbase, atoms_n, max_atoms) != 0u32 {
                    let mut types = 0u32;
                    let mut bonds = 0u32;
                    let mut pointers = 0u32;
                    for id in 1usize..18usize {
                        if type_fits(state, sbase, atypes, meta, mbase, atoms_n, id as u32) != 0u32
                        {
                            let val = atypes[id * 3 + 2];
                            let hyd = atypes[id * 3 + 1];
                            for bond in 1u32..4u32 {
                                if val >= hyd && bond <= val - hyd {
                                    let ptrs = add_pointers(state, sbase, atoms_n, bond, max_atoms);
                                    if ptrs != 0u32 {
                                        // Setting the bit again is idempotent,
                                        // so no completion flag is needed.
                                        types |= 1u32 << (id as u32);
                                        if (id as u32) == steps {
                                            bonds |= 1u32 << bond;
                                            if bond == b {
                                                pointers |= ptrs;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    out[obase + 1] = types;
                    out[obase + 2] = bonds;
                    out[obase + 3] = pointers;
                }
            }
            if k == 3u32 {
                if has_close(state, sbase, atoms_n, max_closures, sentinel) != 0u32 {
                    let mut bonds = 0u32;
                    let mut pointers = 0u32;
                    for bond in 1u32..4u32 {
                        let ptrs =
                            close_pointers(state, sbase, atoms_n, bond, max_closures, sentinel);
                        if ptrs != 0u32 {
                            bonds |= 1u32 << bond;
                            if bond == b {
                                pointers |= ptrs;
                            }
                        }
                    }
                    out[obase + 2] = bonds;
                    out[obase + 3] = pointers;
                }
            }
        }
    }

    /// Whether the token is legal here (`TraceState::is_legal`, including the
    /// rule that fields the kind does not use must be zero).
    ///
    /// The range checks stay in comparison form: `RangeInclusive::contains`
    /// does not lower to the backend IR.
    #[allow(clippy::too_many_arguments)]
    #[cube]
    pub fn is_legal(
        state: &mut Array<u32>,
        sbase: usize,
        atypes: &Array<u32>,
        meta: &Array<u32>,
        mbase: usize,
        atoms_n: usize,
        max_atoms: u32,
        max_closures: u32,
        sentinel: u32,
        k: u32,
        steps: u32,
        b: u32,
        p: u32,
    ) -> u32 {
        let mut legal = 0u32;
        if stopped(state, sbase, atoms_n) == 0u32 {
            let step = step_of(state, sbase, atoms_n);
            if step == 0u32 {
                if k == 1u32 && steps == 0u32 && b == 0u32 && p == 0u32 {
                    legal = 1u32;
                }
            } else if step == 1u32 {
                if k == 2u32 && b == 0u32 && p == 0u32 {
                    if type_fits(state, sbase, atypes, meta, mbase, atoms_n, steps) != 0u32 {
                        legal = 1u32;
                    }
                }
            } else if k == 2u32 {
                if steps >= 1u32 {
                    if steps <= 17u32 {
                        if b >= 1u32 {
                            if b <= 3u32 {
                                let val = atypes[steps as usize * 3 + 2];
                                let hyd = atypes[steps as usize * 3 + 1];
                                if val >= hyd && b <= val - hyd {
                                    if type_fits(state, sbase, atypes, meta, mbase, atoms_n, steps)
                                        != 0u32
                                    {
                                        let n = n_atoms(state, sbase, atoms_n);
                                        if n < max_atoms
                                            && p >= last_parent(state, sbase, atoms_n)
                                            && (p as usize) < (n as usize)
                                            && resid_of(state, sbase, atoms_n, p as usize) >= b
                                        {
                                            legal = 1u32;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            } else if k == 3u32 {
                if steps == 0u32 {
                    if b >= 1u32 {
                        if b <= 3u32 {
                            let n = n_atoms(state, sbase, atoms_n) as usize;
                            if n >= 1 && closures(state, sbase, atoms_n) < max_closures {
                                let newest = n - 1;
                                let par = state[sbase + 2 * atoms_n + newest];
                                let mut pp1 = 0u32;
                                if par != sentinel {
                                    pp1 = par + 1u32;
                                }
                                let lc = last_close_p1(state, sbase, atoms_n);
                                let mut lo = pp1;
                                if lc > lo {
                                    lo = lc;
                                }
                                if p >= lo
                                    && (p as usize) < newest
                                    && resid_of(state, sbase, atoms_n, p as usize) >= b
                                    && resid_of(state, sbase, atoms_n, newest) >= b
                                {
                                    legal = 1u32;
                                }
                            }
                        }
                    }
                }
            } else if k == 4u32 {
                if steps == 0u32 && b == 0u32 && p == 0u32 {
                    legal = 1u32;
                }
            }
        }
        legal
    }

    /// Apply a legal token (`TraceState::apply` without the error path: the
    /// caller checks [`is_legal`] first).
    #[allow(clippy::too_many_arguments)]
    #[cube]
    pub fn apply_token(
        state: &mut Array<u32>,
        sbase: usize,
        atypes: &Array<u32>,
        atoms_n: usize,
        k: u32,
        steps: u32,
        b: u32,
        p: u32,
    ) {
        if k == 2u32 {
            let elem = atypes[steps as usize * 3] as usize;
            let hyd = atypes[steps as usize * 3 + 1];
            let val = atypes[steps as usize * 3 + 2];
            let n = n_atoms(state, sbase, atoms_n) as usize;
            state[sbase + 3 * atoms_n + 6 + elem] = state[sbase + 3 * atoms_n + 6 + elem] + 1u32;
            state[sbase + 3 * atoms_n + 6 + 1] = state[sbase + 3 * atoms_n + 6 + 1] + hyd;
            state[sbase + n] = steps;
            state[sbase + atoms_n + n] = val - hyd;
            if step_of(state, sbase, atoms_n) == 1u32 {
                state[sbase + 2 * atoms_n + n] = 4294967295u32;
            } else {
                let q = p as usize;
                state[sbase + atoms_n + q] = state[sbase + atoms_n + q] - b;
                state[sbase + atoms_n + n] = state[sbase + atoms_n + n] - b;
                state[sbase + 2 * atoms_n + n] = p;
                state[sbase + 3 * atoms_n + 1] = p;
            }
            state[sbase + 3 * atoms_n + 2] = 0u32;
            state[sbase + 3 * atoms_n] = state[sbase + 3 * atoms_n] + 1u32;
        }
        if k == 3u32 {
            let n = n_atoms(state, sbase, atoms_n) as usize;
            let newest = n - 1;
            let q = p as usize;
            state[sbase + atoms_n + q] = state[sbase + atoms_n + q] - b;
            state[sbase + atoms_n + newest] = state[sbase + atoms_n + newest] - b;
            state[sbase + 3 * atoms_n + 2] = p + 1u32;
            state[sbase + 3 * atoms_n + 3] = state[sbase + 3 * atoms_n + 3] + 1u32;
        }
        if k == 4u32 {
            state[sbase + 3 * atoms_n + 5] = 1u32;
        }
        state[sbase + 3 * atoms_n + 4] = state[sbase + 3 * atoms_n + 4] + 1u32;
    }
}

/// Preallocated outputs of [`grammar_replay`] for one `(rows, T, A)` bucket.
pub struct ReplayBuffers<R: Runtime> {
    /// `[rows, 3A + 16]` lane-owned grammar scratch, initialised by the kernel
    /// before it is read.
    pub state: IdTensor<R>,
    /// `[rows, T, 4 + A]` per-step masks and residual valences: for step `t`
    /// the four legality masks before token `t`, then each atom's residual
    /// valence before token `t` (0 for atoms not yet added). Steps at or after
    /// the trace length, and steps after the first illegal token, are all
    /// zero.
    pub replay: IdTensor<R>,
    /// `[rows, A + 1]`: the step at which each atom was added (`u32::MAX`
    /// for unused slots), and the first illegal step (`u32::MAX` for a legal
    /// prefix).
    pub atoms: IdTensor<R>,
}

impl<R: Runtime> ReplayBuffers<R> {
    /// Allocate the three outputs of [`grammar_replay`] uninitialised: the
    /// kernel writes every element, so there is nothing to initialise.
    pub fn new(rows: usize, steps: usize, atoms_n: usize, device: &Device<R>) -> Self {
        Self {
            state: IdTensor::empty(vec![rows, replay_state_width(atoms_n)], device),
            replay: IdTensor::empty(vec![rows, steps, 4 + atoms_n], device),
            atoms: IdTensor::empty(vec![rows, atoms_n + 1], device),
        }
    }

    /// Allocate the outputs filled with poison (`0xDEAD_BEEF`), so a kernel
    /// that skips an element is caught by the comparison with the host
    /// replay. Test support only.
    pub fn poisoned(rows: usize, steps: usize, atoms_n: usize, device: &Device<R>) -> Result<Self> {
        let poison = |len: usize| IdTensor::from_slice(&vec![0xDEAD_BEEF; len], vec![len], device);
        Ok(Self {
            state: poison(rows * replay_state_width(atoms_n))?
                .reshape(vec![rows, replay_state_width(atoms_n)])?,
            replay: poison(rows * steps * (4 + atoms_n))?.reshape(vec![
                rows,
                steps,
                4 + atoms_n,
            ])?,
            atoms: poison(rows * (atoms_n + 1))?.reshape(vec![rows, atoms_n + 1])?,
        })
    }
}

/// Lane per target row: replay `tokens` (`[rows, T, 4]`) with the grammar of
/// contract §4.4 under the budget of `target_meta` (`[rows, 12]`: length, a
/// budget flag, the 10 budget counts in
/// [`crate::models::ms2::chem::ELEMENTS`] order). For each step `steps < length`,
/// before applying token `steps`, write `replay[row, steps, 0..4]` (the four masks
/// [`crate::models::ms2::grammar::TraceState::masks`] would return) and
/// `replay[row, steps, 4..]` (every atom's residual valence, 0 for atoms not yet
/// added); then apply the token if it is legal. `atoms[row, j]` records the
/// step at which atom `j` was added, `atoms[row, A]` the first illegal step.
/// After the first illegal token the row stops applying tokens but still
/// writes all-zero masks for the remaining steps; steps `steps >= length` are all
/// zero. Exactly 1 launch.
pub fn grammar_replay<R: Runtime>(
    tokens: &IdTensor<R>,
    target_meta: &IdTensor<R>,
    constants: &Ms2Constants<R>,
    max_atoms: u32,
    max_closures: u32,
    out: &ReplayBuffers<R>,
) -> Result<()> {
    if tokens.shape().rank() != 3
        || target_meta.shape().rank() != 2
        || out.state.shape().rank() != 2
        || out.replay.shape().rank() != 3
        || out.atoms.shape().rank() != 2
    {
        return Err(Error::shape(format!(
            "grammar_replay needs tokens [rows, T, 4], meta [rows, 12], state [rows, 3A + 16], replay [rows, T, 4 + A] and atoms [rows, A + 1], got {} and {} and {} and {} and {}",
            tokens.shape(),
            target_meta.shape(),
            out.state.shape(),
            out.replay.shape(),
            out.atoms.shape()
        )));
    }
    let rows = tokens.shape().dim(0);
    let steps = tokens.shape().dim(1);
    // Validate limits before any subtraction or launch: the host
    // `TraceState` bounds (`Limits::new`), and the replay/state/atoms
    // widths derived from them.
    if max_atoms == 0 || max_atoms > 32 {
        return Err(Error::config(format!(
            "grammar_replay: max_atoms {max_atoms} is not in 1..=32"
        )));
    }
    if max_closures > 32 {
        return Err(Error::config(format!(
            "grammar_replay: max_closures {max_closures} exceeds 32"
        )));
    }
    let replay_width = out.replay.shape().dim(2);
    if replay_width < 4 {
        return Err(Error::shape(format!(
            "grammar_replay needs replay width 4 + A with A = {max_atoms} (at least 4), got width {replay_width}"
        )));
    }
    let atoms_n = replay_width - 4;
    if max_atoms as usize != atoms_n {
        return Err(Error::shape(format!(
            "grammar_replay needs max_atoms {max_atoms} to match the replay width A = {atoms_n}"
        )));
    }
    if out.state.shape().dim(1) != replay_state_width(atoms_n) {
        return Err(Error::shape(format!(
            "grammar_replay needs state [rows, {}] for A = {atoms_n}, got {}",
            replay_state_width(atoms_n),
            out.state.shape()
        )));
    }
    if out.atoms.shape().dim(1) != atoms_n + 1 {
        return Err(Error::shape(format!(
            "grammar_replay needs atoms [rows, {}] for A = {atoms_n}, got {}",
            atoms_n + 1,
            out.atoms.shape()
        )));
    }
    let want_tokens: &[usize] = &[rows, steps, 4];
    let want_meta: &[usize] = &[rows, 12];
    let want_state: &[usize] = &[rows, replay_state_width(atoms_n)];
    let want_replay: &[usize] = &[rows, steps, 4 + atoms_n];
    let want_atoms: &[usize] = &[rows, atoms_n + 1];
    if tokens.shape().dims() != want_tokens
        || target_meta.shape().dims() != want_meta
        || out.state.shape().dims() != want_state
        || out.replay.shape().dims() != want_replay
        || out.atoms.shape().dims() != want_atoms
    {
        return Err(Error::shape(format!(
            "grammar_replay needs tokens [{rows}, {steps}, 4], meta [{rows}, 12], state [{rows}, {}], replay [{rows}, {steps}, {}] and atoms [{rows}, {}], got {} and {} and {} and {} and {}",
            replay_state_width(atoms_n),
            4 + atoms_n,
            atoms_n + 1,
            tokens.shape(),
            target_meta.shape(),
            out.state.shape(),
            out.replay.shape(),
            out.atoms.shape()
        )));
    }
    if constants.atom_table.len() != 18 * 3 {
        return Err(Error::shape(format!(
            "grammar_replay needs the 54-element resident atom table, got length {}",
            constants.atom_table.len()
        )));
    }
    if rows == 0 {
        return Ok(());
    }
    let client = tokens.client();
    let (count, dim, span) = launch_1d_spans(client, rows, steps * (4 + atoms_n));
    unsafe {
        ms2_replay_kernel::launch_unchecked::<R>(
            client,
            count,
            dim,
            tokens.arg(),
            target_meta.arg(),
            constants.atom_table.arg(),
            out.state.arg(),
            out.replay.arg(),
            out.atoms.arg(),
            steps,
            atoms_n,
            max_atoms,
            max_closures,
            u32::MAX,
            rows,
            span,
        );
    }
    Ok(())
}

/// Lane per target row of [`grammar_replay`]; see its docs for the semantics.
/// Arrays: `tokens`, `target_meta`, `atom_table`, `state`, `replay`, `atoms`.
#[allow(clippy::too_many_arguments)]
#[cube(launch_unchecked)]
fn ms2_replay_kernel(
    tokens: &Array<u32>,
    target_meta: &Array<u32>,
    atom_table: &Array<u32>,
    state: &mut Array<u32>,
    replay: &mut Array<u32>,
    atoms: &mut Array<u32>,
    steps: usize,
    atoms_n: usize,
    max_atoms: u32,
    max_closures: u32,
    sentinel: u32,
    lanes: usize,
    span: usize,
) {
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    for pos in start..end {
        let r = pos;
        let sbase = r * (3 * atoms_n + 16);
        let mbase = r * 12;
        for j in 0..atoms_n {
            state[sbase + j] = 0u32;
            state[sbase + atoms_n + j] = 0u32;
            state[sbase + 2 * atoms_n + j] = sentinel;
        }
        for c in 0..16 {
            state[sbase + 3 * atoms_n + c] = 0u32;
        }
        for j in 0..atoms_n {
            atoms[r * (atoms_n + 1) + j] = sentinel;
        }
        atoms[r * (atoms_n + 1) + atoms_n] = sentinel;
        let length = target_meta[mbase] as usize;
        let width = 4 + atoms_n;
        for tt in 0..steps {
            let rbase = (r * steps + tt) * (4 + atoms_n);
            if tt >= length {
                for w in 0..width {
                    replay[rbase + w] = 0u32;
                }
            } else {
                // The first-illegal marker doubles as the failed flag: once
                // `atoms[row, A]` is set, later steps write zeros without
                // applying anything.
                if atoms[r * (atoms_n + 1) + atoms_n] != sentinel {
                    for w in 0..width {
                        replay[rbase + w] = 0u32;
                    }
                } else {
                    let tbase = (r * steps + tt) * 4;
                    let k = tokens[tbase];
                    let ty = tokens[tbase + 1];
                    let b = tokens[tbase + 2];
                    let p = tokens[tbase + 3];
                    replay_masks(
                        state,
                        sbase,
                        atom_table,
                        target_meta,
                        mbase,
                        atoms_n,
                        max_atoms,
                        max_closures,
                        sentinel,
                        k,
                        ty,
                        b,
                        replay,
                        rbase,
                    );
                    let n = n_atoms(state, sbase, atoms_n) as usize;
                    for j in 0..atoms_n {
                        let mut v = 0u32;
                        if j < n {
                            v = state[sbase + atoms_n + j];
                        }
                        replay[rbase + 4 + j] = v;
                    }
                    if is_legal(
                        state,
                        sbase,
                        atom_table,
                        target_meta,
                        mbase,
                        atoms_n,
                        max_atoms,
                        max_closures,
                        sentinel,
                        k,
                        ty,
                        b,
                        p,
                    ) != 0u32
                    {
                        if k == 2u32 {
                            atoms[r * (atoms_n + 1) + n] = tt as u32;
                        }
                        apply_token(state, sbase, atom_table, atoms_n, k, ty, b, p);
                    } else {
                        atoms[r * (atoms_n + 1) + atoms_n] = tt as u32;
                    }
                }
            }
        }
    }
}

/// Lane per element: `id - 1`, keeping the `u32::MAX` sentinel (and 0) fixed.
/// Turns atom-add steps into the `h` indices the atom memory gathers.
#[cube(launch_unchecked)]
fn ms2_pred_ids_kernel(
    ids: &Array<u32>,
    out: &mut Array<u32>,
    sentinel: u32,
    lanes: usize,
    span: usize,
) {
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    for pos in start..end {
        let id = ids[pos];
        let mut v = id;
        if id != sentinel && id > 0u32 {
            v = id - 1u32;
        }
        out[pos] = v;
    }
}

/// `atom_step - 1` with `u32::MAX` mapping to `u32::MAX`, on the device (the
/// atom added by token `s` has memory `h[s - 1]`, and unused atom slots keep
/// the gather's zero row). One launch.
pub fn pred_ids<R: Runtime>(ids: &IdTensor<R>) -> Result<IdTensor<R>> {
    let out = IdTensor::empty(ids.shape().clone(), ids.device());
    if out.is_empty() {
        return Ok(out);
    }
    let lanes = out.len();
    let (count, dim, span) = launch_1d_spans(ids.client(), lanes, 1);
    unsafe {
        ms2_pred_ids_kernel::launch_unchecked::<R>(
            ids.client(),
            count,
            dim,
            ids.arg(),
            out.arg(),
            u32::MAX,
            lanes,
            span,
        );
    }
    Ok(out)
}

/// Lane per target row: the in-range conditioning ids for the teacher pass at
/// trace position `pos` (predicting token `pos`). `tgt` is the `[rows, 4]`
/// target token there, `len` the `[rows]` trace lengths. Output `[rows, 5]`:
/// kind, atom type, bond and pointer ids with 0 substituted for every unused
/// field and every unscored position (`take_along_last` indexes its id
/// directly and has no sentinel handling), plus the conditioning row `c` (the
/// target atom type for ADD_ATOM, 18 for CLOSE_RING, 0 elsewhere). Fields the
/// target token does not use, out-of-range fields and unscored positions can
/// therefore only select row 0, never read out of bounds.
#[cube(launch_unchecked)]
fn ms2_teacher_ids_kernel(
    tgt: &Array<u32>,
    len: &Array<u32>,
    out: &mut Array<u32>,
    pos: usize,
    atoms_n: usize,
    lanes: usize,
    span: usize,
) {
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    for r in start..end {
        let k = tgt[r * 4];
        let ty = tgt[r * 4 + 1];
        let b = tgt[r * 4 + 2];
        let p = tgt[r * 4 + 3];
        // `pos` is scored when it names a real token; every id below is 0
        // for an unused field or an unscored position, by nested selection
        // rather than a flag (a conditionally assigned local here breaks the
        // kernel macro's type inference).
        let mut kind_id = 0u32;
        let mut type_id = 0u32;
        let mut bond_id = 0u32;
        let mut ptr_id = 0u32;
        let mut c = 0u32;
        if pos >= 1 {
            if (pos as u32) < len[r] {
                if k <= 4u32 {
                    kind_id = k;
                }
                if k == 2u32 {
                    if ty >= 1u32 {
                        if ty <= 17u32 {
                            type_id = ty;
                            c = ty;
                        }
                    }
                    if pos != 1 {
                        if b >= 1u32 {
                            if b <= 3u32 {
                                bond_id = b;
                            }
                        }
                        if (p as usize) < atoms_n {
                            ptr_id = p;
                        }
                    }
                }
                if k == 3u32 {
                    c = 18u32;
                    if b >= 1u32 {
                        if b <= 3u32 {
                            bond_id = b;
                        }
                    }
                    if (p as usize) < atoms_n {
                        ptr_id = p;
                    }
                }
            }
        }
        out[r * 5] = kind_id;
        out[r * 5 + 1] = type_id;
        out[r * 5 + 2] = bond_id;
        out[r * 5 + 3] = ptr_id;
        out[r * 5 + 4] = c;
    }
}

/// The in-range conditioning ids of [`ms2_teacher_ids_kernel`]: `tgt` is
/// `[rows, 4]`, `len` is `[rows]`, output `[rows, 5]`. One launch.
pub fn teacher_ids<R: Runtime>(
    tgt: &IdTensor<R>,
    len: &IdTensor<R>,
    pos: usize,
    atoms_n: usize,
) -> Result<IdTensor<R>> {
    if tgt.shape().rank() != 2 || len.shape().rank() != 1 {
        return Err(Error::shape(format!(
            "teacher_ids needs tgt [rows, 4] and len [rows], got {} and {}",
            tgt.shape(),
            len.shape()
        )));
    }
    let rows = len.len();
    if tgt.len() != rows * 4 {
        return Err(Error::shape(format!(
            "teacher_ids needs tgt [rows, 4] for len [rows], got {} and {}",
            tgt.shape(),
            len.shape()
        )));
    }
    let out = IdTensor::empty(vec![rows, 5], tgt.device());
    if out.is_empty() {
        return Ok(out);
    }
    let (count, dim, span) = launch_1d_spans(tgt.client(), rows, 1);
    unsafe {
        ms2_teacher_ids_kernel::launch_unchecked::<R>(
            tgt.client(),
            count,
            dim,
            tgt.arg(),
            len.arg(),
            out.arg(),
            pos,
            atoms_n,
            rows,
            span,
        );
    }
    Ok(out)
}

/// Columns of one [`teacher_plan`] row before the `A` residual columns: the
/// five conditioning ids of [`teacher_ids`] (kind, atom type, bond, pointer,
/// conditioning row) and the four legality bit fields of the replay row.
pub const TEACHER_PLAN_HEAD: usize = 9;

/// Lane per `(target row, output position)`: everything the teacher heads
/// need from the integer side at output position `i` (predicting token
/// `pos = i + 1`), for every position at once. One plan row holds the five
/// conditioning ids of [`ms2_teacher_ids_kernel`] at `pos`, the four legality
/// bit fields `replay[r, pos, 0..4]` and the `A` clamped residuals
/// `min(replay[r, pos, 4..4 + A], 7)`, so the teacher pass issues one launch where
/// the per-position form issued a slice and a kernel per position.
#[cube(launch_unchecked)]
fn ms2_teacher_plan_kernel(
    tokens: &Array<u32>,
    meta: &Array<u32>,
    replay: &Array<u32>,
    out: &mut Array<u32>,
    steps: usize,
    atoms_n: usize,
    meta_width: usize,
    lanes: usize,
    span: usize,
) {
    let positions = steps - 1;
    let width = 9 + atoms_n;
    let replay_width = 4 + atoms_n;
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    for lane in start..end {
        let r = lane / positions;
        let pos = lane % positions + 1;
        let tok = (r * steps + pos) * 4;
        let k = tokens[tok];
        let ty = tokens[tok + 1];
        let b = tokens[tok + 2];
        let p = tokens[tok + 3];
        let mut kind_id = 0u32;
        let mut type_id = 0u32;
        let mut bond_id = 0u32;
        let mut ptr_id = 0u32;
        let mut c = 0u32;
        if (pos as u32) < meta[r * meta_width] {
            if k <= 4u32 {
                kind_id = k;
            }
            if k == 2u32 {
                if ty >= 1u32 {
                    if ty <= 17u32 {
                        type_id = ty;
                        c = ty;
                    }
                }
                if pos != 1 {
                    if b >= 1u32 {
                        if b <= 3u32 {
                            bond_id = b;
                        }
                    }
                    if (p as usize) < atoms_n {
                        ptr_id = p;
                    }
                }
            }
            if k == 3u32 {
                c = 18u32;
                if b >= 1u32 {
                    if b <= 3u32 {
                        bond_id = b;
                    }
                }
                if (p as usize) < atoms_n {
                    ptr_id = p;
                }
            }
        }
        let o = lane * width;
        out[o] = kind_id;
        out[o + 1] = type_id;
        out[o + 2] = bond_id;
        out[o + 3] = ptr_id;
        out[o + 4] = c;
        let src = (r * steps + pos) * replay_width;
        for j in 0..4 {
            out[o + 5 + j] = replay[src + j];
        }
        // Residuals clamped to the residual table (`min(residual, 7)`,
        // architecture §4.3), so they index it directly.
        for j in 0..atoms_n {
            let v = replay[src + 4 + j];
            let mut c = v;
            if c > 7u32 {
                c = 7u32;
            }
            out[o + 9 + j] = c;
        }
    }
}

/// The integer plan of the whole teacher pass in one launch: for `tokens`
/// (`[rows, T, 4]`), `meta` (`[rows, W]`, trace length in column 0) and
/// `replay` (`[rows, T, 4 + A]`), a `[rows * (T - 1), 9 + A]` buffer whose row
/// `r * (T - 1) + i` describes output position `i` (predicting token
/// `i + 1`): columns `0..5` are exactly [`teacher_ids`] at `pos = i + 1`,
/// columns `5..9` the legality bit fields `replay[r, i + 1, 0..4]` and
/// columns `9..` the residuals `min(replay[r, i + 1, 4..], 7)`.
pub fn teacher_plan<R: Runtime>(
    tokens: &IdTensor<R>,
    meta: &IdTensor<R>,
    replay: &IdTensor<R>,
    atoms_n: usize,
) -> Result<IdTensor<R>> {
    if tokens.shape().rank() != 3 || meta.shape().rank() != 2 || replay.shape().rank() != 3 {
        return Err(Error::shape(format!(
            "teacher_plan needs tokens [rows, T, 4], meta [rows, W] and replay [rows, T, 4 + A], got {} and {} and {}",
            tokens.shape(),
            meta.shape(),
            replay.shape()
        )));
    }
    let rows = tokens.shape().dim(0);
    let steps = tokens.shape().dim(1);
    let want_replay: &[usize] = &[rows, steps, 4 + atoms_n];
    if tokens.shape().dim(2) != 4
        || meta.shape().dim(0) != rows
        || meta.shape().dim(1) == 0
        || replay.shape().dims() != want_replay
        || steps == 0
    {
        return Err(Error::shape(format!(
            "teacher_plan has mismatched shapes: tokens {}, meta {}, replay {} for A = {atoms_n}",
            tokens.shape(),
            meta.shape(),
            replay.shape()
        )));
    }
    let lanes = rows * (steps - 1);
    let out = IdTensor::empty(vec![lanes, TEACHER_PLAN_HEAD + atoms_n], tokens.device());
    if out.is_empty() {
        return Ok(out);
    }
    let (count, dim, span) = launch_1d_spans(tokens.client(), lanes, TEACHER_PLAN_HEAD + atoms_n);
    unsafe {
        ms2_teacher_plan_kernel::launch_unchecked::<R>(
            tokens.client(),
            count,
            dim,
            tokens.arg(),
            meta.arg(),
            replay.arg(),
            out.arg(),
            steps,
            atoms_n,
            meta.shape().dim(1),
            lanes,
            span,
        );
    }
    Ok(out)
}

/// Lane per output element: the effective legality mask of one teacher field
/// (architecture §3.8) for every `(row, position)` at once. With `m` the bit
/// `col` of the field's replay bits, `u` the field's use indicator and `idle`
/// the "index 0 only" row, the value is `m * u + idle * (1 - u)` — the same
/// arithmetic as the composed `bits_to_mask`, `mul`, `rsub_scalar`, `add`
/// chain it replaces.
#[cube(launch_unchecked)]
fn ms2_effective_mask_kernel<F: Float + CubeElement>(
    plan: &Array<u32>,
    use_mask: &Array<F>,
    out: &mut Array<F>,
    width: usize,
    plan_width: usize,
    field: usize,
    positions: usize,
    steps: usize,
    lanes: usize,
    span: usize,
) {
    let zero = F::new(0.0_f32);
    let one = F::new(1.0_f32);
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    for pos in start..end {
        let n = pos / width;
        let col = (pos % width) as u32;
        let r = n / positions;
        let i = n % positions;
        let bits = plan[n * plan_width + 5 + field];
        let u = use_mask[(r * steps + i) * 4 + field];
        let mut m = zero;
        if (bits & (1u32 << col)) != 0u32 {
            m = one;
        }
        let mut idle = zero;
        if col == 0u32 {
            idle = one;
        }
        out[pos] = m * u + idle * (one - u);
    }
}

/// The effective float mask `[rows * (T - 1), width]` of teacher field
/// `field` (0 kind, 1 atom type, 2 bond, 3 pointer) for every output
/// position: the replay bit mask where the field is used, "index 0 only"
/// elsewhere (architecture §3.8). `plan` is [`teacher_plan`]'s buffer and
/// `use_mask` the `[rows, T, 4]` use indicators (position `i` scores token
/// `i + 1`). One launch; `width <= 32`.
pub fn effective_mask<R: Runtime, E: FloatElem>(
    plan: &IdTensor<R>,
    use_mask: &Tensor<R, E>,
    field: usize,
    width: usize,
) -> Result<Tensor<R, E>> {
    if plan.shape().rank() != 2 || use_mask.shape().rank() != 3 || field >= 4 || width > 32 {
        return Err(Error::shape(format!(
            "effective_mask needs plan [rows * (T - 1), 9 + A], use [rows, T, 4], field < 4 and width <= 32, got {} and {} and field {field} and width {width}",
            plan.shape(),
            use_mask.shape()
        )));
    }
    let rows = use_mask.shape().dim(0);
    let steps = use_mask.shape().dim(1);
    let n = plan.shape().dim(0);
    if use_mask.shape().dim(2) != 4
        || steps == 0
        || n != rows * (steps - 1)
        || plan.shape().dim(1) < TEACHER_PLAN_HEAD
    {
        return Err(Error::shape(format!(
            "effective_mask has mismatched shapes: plan {} and use {}",
            plan.shape(),
            use_mask.shape()
        )));
    }
    let out = Tensor::empty(Shape::new(vec![n, width]), use_mask.device());
    if out.is_empty() {
        return Ok(out);
    }
    let lanes = out.len();
    let (count, dim, span) = launch_1d_spans(plan.client(), lanes, 1);
    unsafe {
        ms2_effective_mask_kernel::launch_unchecked::<E, R>(
            plan.client(),
            count,
            dim,
            plan.arg(),
            use_mask.arg(),
            out.arg(),
            width,
            plan.shape().dim(1),
            field,
            steps - 1,
            steps,
            lanes,
            span,
        );
    }
    Ok(out)
}

// Generation: legal sampling, initialisation and validation (architecture
// §§3.6–3.7 and 5)
// ---------------------------------------------------------------------------
// The packed sampler-logits layout below is the one constant layout shared by
// the device packer ([`pack_copy`]) and the sampler ([`sample_step`]): one row
// holds `kind (5) | atom_type (18) | bond_base (4) | pointer_base (A) |
// pointer_by_type (19 * A) | pointer_by_bond (4 * A)`, where the 19
// conditioning rows are the 17 atom types plus row 0 (unused) and row 18
// (CLOSE_RING), and the 4 bond rows are indexed by bond value. The sampler adds
// `bond_by_type[c]` from its `tables` argument and
// `pointer_by_type[c] + pointer_by_bond[b]` before masking, exactly like the
// teacher's heads.

/// Width of the kind field of a packed sampler-logits row.
pub const SAMPLE_KIND_WIDTH: usize = 5;
/// Width of the atom-type field of a packed sampler-logits row.
pub const SAMPLE_TYPE_WIDTH: usize = 18;
/// Width of the bond field of a packed sampler-logits row.
pub const SAMPLE_BOND_WIDTH: usize = 4;
/// Conditioning rows of the pointer tables: the 17 atom types plus row 0
/// (unused) and row 18 (CLOSE_RING).
pub const SAMPLE_COND_ROWS: usize = 19;
/// Bond rows of the pointer table, indexed by bond value.
pub const SAMPLE_PBOND_ROWS: usize = 4;

/// Width of one packed sampler-logits row for `a` atoms: `27 + 24 * a`.
pub fn sample_logits_width(a: usize) -> usize {
    27 + 24 * a
}

/// Field offsets within one packed sampler-logits row: kind, atom type, bond,
/// pointer base, pointer by type, pointer by bond.
pub fn sample_logits_offsets(a: usize) -> [usize; 6] {
    [0, 5, 23, 27, 27 + a, 27 + a + 19 * a]
}

/// Width of the per-trajectory metadata of [`init_trajectories`] and
/// [`sample_step`]: spectrum id low/high, trajectory index, started flag, 10
/// budget counts in [`crate::models::ms2::chem::ELEMENTS`] order.
pub const TRAJ_META_WIDTH: usize = 14;

/// Words of one trajectory action record for `t` steps and `a` atoms:
/// `t * 4` trace words, `a` open-valence words, then length, status,
/// trace-log-probability bits and formula row.
pub fn sample_record_width(t: usize, a: usize) -> usize {
    t * 4 + a + 4
}

/// Draws for [`sample_step`]: architecture §3.6, `s = hash_u32(seed_hi,
/// seed_lo, 0)`, `key = hash_u32(id_lo, s, id_hi)`, `base =
/// hash_u32(trajectory, key, 0)`, `u = hash_unit_f32(4 * step + field, base,
/// 0)`, via [`hash_u32`] and [`hash_unit_f32`].
///
/// Zeroed grammar state rows (`[rows, 3A + 16]`, the layout of
/// [`grammar_replay`]): the starting point [`grammar_apply`] applies tokens
/// onto. No launch.
pub fn grammar_state_zeros<R: Runtime>(
    rows: usize,
    atoms: usize,
    device: &Device<R>,
) -> IdTensor<R> {
    IdTensor::from_slice(
        &vec![0u32; rows * replay_state_width(atoms)],
        vec![rows, replay_state_width(atoms)],
        device,
    )
    .expect("a zeroed grammar state fits")
}

/// Lane per row: apply one token (`[rows, 4]`) onto the grammar `state` row
/// with the same [`apply_token`] helper [`grammar_replay`] uses, under the
/// budget of `target_meta` (`[rows, 12]`, the layout of [`grammar_replay`]).
/// An illegal token leaves the row unchanged and sets its stopped word to the
/// error marker 2 (a legal STOP writes 1). Exactly 1 launch.
///
/// Aliasing: `state` is written while `tokens` and `target_meta` are read in
/// the same launch, so neither input may share storage with `state`; such a
/// call is [`Error::Config`] before any launch. The two inputs are only read
/// and may alias each other.
pub fn grammar_apply<R: Runtime>(
    tokens: &IdTensor<R>,
    state: &mut IdTensor<R>,
    target_meta: &IdTensor<R>,
    constants: &Ms2Constants<R>,
    max_atoms: u32,
    max_closures: u32,
) -> Result<()> {
    if tokens.shape().rank() != 2 || state.shape().rank() != 2 || target_meta.shape().rank() != 2 {
        return Err(Error::shape(format!(
            "grammar_apply needs tokens [rows, 4], state [rows, 3A + 16] and meta [rows, 12], got {} and {} and {}",
            tokens.shape(),
            state.shape(),
            target_meta.shape()
        )));
    }
    let rows = tokens.shape().dim(0);
    let atoms_n = state
        .shape()
        .dim(1)
        .checked_sub(STATE_WIDTH_BASE)
        .ok_or_else(|| {
            Error::shape(format!(
                "grammar_apply needs state [rows, 3A + 16], got {}",
                state.shape()
            ))
        })?
        / 3;
    if max_atoms as usize != atoms_n {
        return Err(Error::shape(format!(
            "grammar_apply needs max_atoms {max_atoms} to match the state width A = {atoms_n}"
        )));
    }
    let want_tokens: &[usize] = &[rows, 4];
    let want_state: &[usize] = &[rows, replay_state_width(atoms_n)];
    let want_meta: &[usize] = &[rows, 12];
    if tokens.shape().dims() != want_tokens
        || state.shape().dims() != want_state
        || target_meta.shape().dims() != want_meta
    {
        return Err(Error::shape(format!(
            "grammar_apply needs tokens [{rows}, 4], state [{rows}, {}] and meta [{rows}, 12], got {} and {} and {}",
            replay_state_width(atoms_n),
            tokens.shape(),
            state.shape(),
            target_meta.shape()
        )));
    }
    if constants.atom_table.len() != 18 * 3 {
        return Err(Error::shape(format!(
            "grammar_apply needs the 54-element resident atom table, got length {}",
            constants.atom_table.len()
        )));
    }
    // The kernel reads `tokens`/`target_meta` while writing `state`: shared
    // storage would let a lane's write corrupt another lane's read, so it is
    // refused before any launch.
    if shares_storage(&tokens.arg(), &state.arg()) {
        return Err(Error::config(
            "grammar_apply: tokens and state share storage; the output must not alias an input"
                .to_string(),
        ));
    }
    if shares_storage(&target_meta.arg(), &state.arg()) {
        return Err(Error::config(
            "grammar_apply: target_meta and state share storage; the output must not alias an input"
                .to_string(),
        ));
    }
    if rows == 0 {
        return Ok(());
    }
    let client = tokens.client();
    let (count, dim, span) = launch_1d_spans(client, rows, 1);
    unsafe {
        ms2_grammar_apply_kernel::launch_unchecked::<R>(
            client,
            count,
            dim,
            tokens.arg(),
            state.arg(),
            target_meta.arg(),
            constants.atom_table.arg(),
            atoms_n,
            max_atoms,
            max_closures,
            u32::MAX,
            rows,
            span,
        );
    }
    Ok(())
}

/// Lane per row of [`grammar_apply`]; see its docs for the semantics.
/// Arrays: `tokens`, `state` (in/out), `target_meta`, `atom_table`.
#[allow(clippy::too_many_arguments)]
#[cube(launch_unchecked)]
fn ms2_grammar_apply_kernel(
    tokens: &Array<u32>,
    state: &mut Array<u32>,
    target_meta: &Array<u32>,
    atom_table: &Array<u32>,
    atoms_n: usize,
    max_atoms: u32,
    max_closures: u32,
    sentinel: u32,
    lanes: usize,
    span: usize,
) {
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    for pos in start..end {
        let r = pos;
        let sbase = r * (3 * atoms_n + 16);
        let mbase = r * 12;
        let k = tokens[r * 4];
        let ty = tokens[r * 4 + 1];
        let b = tokens[r * 4 + 2];
        let p = tokens[r * 4 + 3];
        if is_legal(
            state,
            sbase,
            atom_table,
            target_meta,
            mbase,
            atoms_n,
            max_atoms,
            max_closures,
            sentinel,
            k,
            ty,
            b,
            p,
        ) != 0u32
        {
            apply_token(state, sbase, atom_table, atoms_n, k, ty, b, p);
        } else {
            // The error marker: a legal STOP writes 1, so 2 names an illegal
            // token without touching anything else in the row.
            state[sbase + 3 * atoms_n + 5] = 2u32;
        }
    }
}

/// Write the atom-memory row of an added atom and refresh the clamped residual
/// ids, in place and with no device read.
///
/// When the row's `token` kind is ADD_ATOM, the decoder output that predicted
/// the token (`prev_h`, the previous output, matching the teacher's
/// `gather_tokens(h, atom_step - 1)`) is copied into
/// `atom_memory[row, count - 1, :]` with `count` read from the grammar state
/// row (the state after `token` was applied, so it already counts the new
/// atom); any other kind leaves the memory untouched. The clamped ids
/// `min(residual, 7)` land in the preallocated `[rows * A]` buffer the pointer
/// head's residual lookup reads. Exactly 2 launches however many rows there
/// are, instead of the per-row slice/cat loop this replaces.
///
/// Aliasing: `prev_h` is read while `atom_memory` is written, and
/// `grammar_state` is read while `resid_ids` is written, so those pairs must
/// not share storage; such a call is [`Error::Config`] before any launch.
pub fn atom_memory_update<R: Runtime, E: FloatElem>(
    token: &IdTensor<R>,
    grammar_state: &IdTensor<R>,
    prev_h: &Tensor<R, E>,
    atom_memory: &mut Tensor<R, E>,
    resid_ids: &mut IdTensor<R>,
    max_atoms: usize,
) -> Result<()> {
    if token.shape().rank() != 2
        || grammar_state.shape().rank() != 2
        || prev_h.rank() != 2
        || atom_memory.rank() != 3
    {
        return Err(Error::shape(format!(
            "atom_memory_update needs token [rows, 4], grammar_state [rows, 3A + 16], prev_h [rows, d] and atom_memory [rows, A, d], got {} and {} and {} and {}",
            token.shape(),
            grammar_state.shape(),
            prev_h.shape(),
            atom_memory.shape()
        )));
    }
    let rows = token.shape().dim(0);
    let d = prev_h.shape().dim(1);
    let a = max_atoms;
    if grammar_state.shape().dims() != [rows, replay_state_width(a)]
        || prev_h.shape().dims() != [rows, d]
        || atom_memory.shape().dims() != [rows, a, d]
        || resid_ids.len() != rows * a
    {
        return Err(Error::shape(format!(
            "atom_memory_update has mismatched batch shapes: token {}, grammar_state {}, prev_h {}, atom_memory {}, resid {rows}x{a}",
            token.shape(),
            grammar_state.shape(),
            prev_h.shape(),
            atom_memory.shape()
        )));
    }
    // The first kernel reads `prev_h` while writing `atom_memory` and the
    // second reads `grammar_state` while writing `resid_ids`: shared storage
    // in either pair would corrupt the read, so both are refused up front.
    // (`token`/`grammar_state` are only read and may alias each other.)
    if shares_storage(&prev_h.arg(), &atom_memory.arg()) {
        return Err(Error::config(
            "atom_memory_update: prev_h and atom_memory share storage; the output must not alias an input"
                .to_string(),
        ));
    }
    if shares_storage(&grammar_state.arg(), &resid_ids.arg()) {
        return Err(Error::config(
            "atom_memory_update: grammar_state and resid_ids share storage; the output must not alias an input"
                .to_string(),
        ));
    }
    if shares_storage(&token.arg(), &resid_ids.arg()) {
        return Err(Error::config(
            "atom_memory_update: token and resid_ids share storage; the output must not alias an input"
                .to_string(),
        ));
    }
    if rows == 0 {
        return Ok(());
    }
    let client = token.client();
    let (count, dim, span) = launch_1d_spans(client, rows * d, 1);
    unsafe {
        ms2_atom_memory_kernel::launch_unchecked::<E, R>(
            client,
            count,
            dim,
            token.arg(),
            grammar_state.arg(),
            prev_h.arg(),
            atom_memory.arg(),
            a,
            d,
            replay_state_width(a),
            rows * d,
            span,
        );
    }
    let (count, dim, span) = launch_1d_spans(client, rows * a, 1);
    unsafe {
        ms2_residual_ids_kernel::launch_unchecked::<R>(
            client,
            count,
            dim,
            grammar_state.arg(),
            resid_ids.arg(),
            a,
            replay_state_width(a),
            rows * a,
            span,
        );
    }
    Ok(())
}

/// Lane per `(row, d-element)` of [`atom_memory_update`]: on ADD_ATOM copy
/// `prev_h[row, :]` into `atom_memory[row, count - 1, :]` in place, where
/// `count` is the atom count of the grammar state row (after the token was
/// applied). Arrays: `token`, `grammar_state`, `prev_h`, `atom_memory`.
#[allow(clippy::too_many_arguments)]
#[cube(launch_unchecked)]
fn ms2_atom_memory_kernel<F: Float + CubeElement>(
    token: &Array<u32>,
    grammar_state: &Array<u32>,
    prev_h: &Array<F>,
    atom_memory: &mut Array<F>,
    atoms_n: usize,
    d: usize,
    state_width: usize,
    lanes: usize,
    span: usize,
) {
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    for pos in start..end {
        let r = pos / d;
        let j = pos % d;
        if token[r * 4] == 2u32 {
            let n = grammar_state[(r * state_width) + 3 * atoms_n];
            if n != 0u32 && n <= atoms_n as u32 {
                let slot = (n - 1u32) as usize;
                atom_memory[(r * atoms_n + slot) * d + j] = prev_h[r * d + j];
            }
        }
    }
}

/// Lane per `(row, atom)` of [`atom_memory_update`]: the clamped residual id
/// `min(residual, 7)` of every atom. Arrays: `grammar_state`, `resid_ids`.
#[cube(launch_unchecked)]
fn ms2_residual_ids_kernel(
    grammar_state: &Array<u32>,
    resid_ids: &mut Array<u32>,
    atoms_n: usize,
    state_width: usize,
    lanes: usize,
    span: usize,
) {
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    for pos in start..end {
        let r = pos / atoms_n;
        let j = pos % atoms_n;
        let v = grammar_state[r * state_width + atoms_n + j];
        let mut c = v;
        if c > 7u32 {
            c = 7u32;
        }
        resid_ids[pos] = c;
    }
}

/// Lane per row: the sampler's next input token, read from the trajectory's
/// own last emitted token (`actions` at `(length - 1)`, zeros for a row with
/// `length == 0`). Arrays: `actions`, `out` (`[rows, 4]`). Exactly 1 launch.
pub fn step_token<R: Runtime>(
    actions: &IdTensor<R>,
    out: &mut IdTensor<R>,
    steps: usize,
    atoms: usize,
) -> Result<()> {
    let rows = out.len() / 4;
    if out.shape().dims() != [rows, 4]
        || actions.shape().dims() != [rows, sample_record_width(steps, atoms)]
    {
        return Err(Error::shape(format!(
            "step_token needs actions [rows, {}] and out [rows, 4], got {} and {}",
            sample_record_width(steps, atoms),
            actions.shape(),
            out.shape()
        )));
    }
    if rows == 0 {
        return Ok(());
    }
    let client = actions.client();
    let (count, dim, span) = launch_1d_spans(client, rows, 1);
    unsafe {
        ms2_step_token_kernel::launch_unchecked::<R>(
            client,
            count,
            dim,
            actions.arg(),
            out.arg(),
            steps,
            atoms,
            sample_record_width(steps, atoms),
            rows,
            span,
        );
    }
    Ok(())
}

/// Lane per row of [`step_token`]; see its docs for the semantics.
#[allow(clippy::too_many_arguments)]
#[cube(launch_unchecked)]
fn ms2_step_token_kernel(
    actions: &Array<u32>,
    out: &mut Array<u32>,
    steps: usize,
    atoms_n: usize,
    record_width: usize,
    lanes: usize,
    span: usize,
) {
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    for pos in start..end {
        let r = pos;
        let abase = r * record_width;
        let len = actions[abase + steps * 4 + atoms_n];
        for c in 0..4usize {
            let mut v = 0u32;
            if len != 0u32 {
                v = actions[abase + ((len - 1u32) as usize) * 4 + c];
            }
            out[r * 4 + c] = v;
        }
    }
}

/// Lane per element: copy one head field into its packed-logits slice,
/// `out[row * stride + offset + i] = field[row * width + i]`. The generation
/// loop calls this once per head field (6 launches per step, independent of
/// the rows), which keeps the sampler at 6 array bindings. Exactly 1 launch
/// per call.
pub fn pack_copy<R: Runtime, E: FloatElem>(
    field: &Tensor<R, E>,
    out: &mut Tensor<R, E>,
    rows: usize,
    width: usize,
    offset: usize,
    stride: usize,
) -> Result<()> {
    if field.len() != rows * width || out.len() != rows * stride {
        return Err(Error::shape(format!(
            "pack_copy needs field [rows, {width}] and out [rows, {stride}], got {} and {}",
            field.shape(),
            out.shape()
        )));
    }
    if field.is_empty() {
        return Ok(());
    }
    let lanes = field.len();
    let (count, dim, span) = launch_1d_spans(field.client(), lanes, 1);
    unsafe {
        ms2_pack_copy_kernel::launch_unchecked::<E, R>(
            field.client(),
            count,
            dim,
            field.arg(),
            out.arg(),
            width,
            offset,
            stride,
            lanes,
            span,
        );
    }
    Ok(())
}

/// Lane per element of [`pack_copy`]; see its docs for the semantics.
#[cube(launch_unchecked)]
fn ms2_pack_copy_kernel<F: Float + CubeElement>(
    field: &Array<F>,
    out: &mut Array<F>,
    width: usize,
    offset: usize,
    stride: usize,
    lanes: usize,
    span: usize,
) {
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    for pos in start..end {
        let r = pos / width;
        let i = pos % width;
        out[r * stride + offset + i] = field[pos];
    }
}

/// Lane per `(trajectory, d-element)`: the conditioning formula embedding of
/// one trajectory, `out[r, :] = embedding[b, slot]` with `slot` the window
/// slot of the trajectory's retained formula: `s = traj_formula[(b, k), 0]`
/// (the retained index written by `ms2_allocate`, `u32::MAX` when there is
/// none) maps through `top[(b, s), 1]`; zeros when there is none. Arrays:
/// `embedding` (`[B, M, d]`), `traj_formula` (`[B, K, 12]`), `top`
/// (`[B, F, 2]`), `out` (`[rows, d]`). Exactly 1 launch per generation call:
/// the formula does not change mid-trace.
#[allow(clippy::too_many_arguments)]
pub fn trajectory_formula<R: Runtime, E: FloatElem>(
    embedding: &Tensor<R, E>,
    traj_formula: &IdTensor<R>,
    top: &IdTensor<R>,
    out: &mut Tensor<R, E>,
    spectra: usize,
    window_m: usize,
    formulas: usize,
    per_spectrum: usize,
) -> Result<()> {
    let rows = out.shape().dim(0);
    let d = out.shape().dim(1);
    if embedding.shape().dims() != [spectra, window_m, d]
        || traj_formula.shape().dims() != [spectra, per_spectrum, 12]
        || top.shape().dims() != [spectra, formulas, 2]
        || out.shape().dims() != [rows, d]
        || rows != spectra * per_spectrum
    {
        return Err(Error::shape(format!(
            "trajectory_formula needs embedding [B, M, d], traj_formula [B, K, 12], top [B, F, 2] and out [B*K, d], got {} and {} and {} and {}",
            embedding.shape(),
            traj_formula.shape(),
            top.shape(),
            out.shape()
        )));
    }
    if out.is_empty() {
        return Ok(());
    }
    let lanes = out.len();
    let (count, dim, span) = launch_1d_spans(embedding.client(), lanes, 1);
    unsafe {
        ms2_trajectory_formula_kernel::launch_unchecked::<E, R>(
            embedding.client(),
            count,
            dim,
            embedding.arg(),
            traj_formula.arg(),
            top.arg(),
            out.arg(),
            window_m,
            formulas,
            d,
            per_spectrum,
            u32::MAX,
            lanes,
            span,
        );
    }
    Ok(())
}

/// Lane per element of [`trajectory_formula`]; see its docs for the semantics.
#[allow(clippy::too_many_arguments)]
#[cube(launch_unchecked)]
fn ms2_trajectory_formula_kernel<F: Float + CubeElement>(
    embedding: &Array<F>,
    traj_formula: &Array<u32>,
    top: &Array<u32>,
    out: &mut Array<F>,
    window_m: usize,
    formulas: usize,
    d: usize,
    per_spectrum: usize,
    sentinel: u32,
    lanes: usize,
    span: usize,
) {
    let zero = F::new(0.0f32);
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    for pos in start..end {
        let r = pos / d;
        let j = pos % d;
        let b = r / per_spectrum;
        let k = (r % per_spectrum) as u32;
        // The trajectory's retained formula slot from the allocation buffer
        // (V1 §3.2); the window slot comes from `top`, as the host readout
        // does. With `RoundRobin` the slot is `k mod count`, the V0 rule, so
        // every result is bit-identical to the V0 path.
        let s = traj_formula[(b * per_spectrum + k as usize) * 12];
        let mut v = zero;
        if s != sentinel && (s as usize) < formulas {
            let w = top[(b * formulas + s as usize) * 2 + 1];
            if w != sentinel {
                v = embedding[(b * window_m + w as usize) * d + j];
            }
        }
        out[pos] = v;
    }
}

/// One legal sampling step (architecture §3.6), lane per trajectory.
///
/// Bindings (6): `logits` (float, one packed row per trajectory, the layout of
/// [`sample_logits_offsets`]), `tables` (float, `bond_by_type [19, 4]`),
/// `traj_meta` (`[rows, 14]`: id low/high, trajectory index, started flag, 10
/// budget counts), `state` (in/out, the grammar row), `actions` (in/out, the
/// record of [`sample_record_width`]), `atom_table` (the resident `[18, 3]`
/// atom-type table the legality helpers read).
///
/// In one launch it leaves finished, failed and not-started rows untouched;
/// computes the legal kinds with the same [`kind_mask`] helper
/// [`grammar_replay`] uses (no kind legal sets `no_valid_action`); samples the
/// kind, then the atom type, bond order (`bond_base + bond_by_type[c]`) and
/// pointer (`pointer_base + pointer_by_type[c] + pointer_by_bond[bond]`) from
/// their masked softmaxes at `temperature`, each over its own legal set in
/// increasing index order with the inverse-CDF draw `u = hash_unit_f32(4 *
/// step + field, base, 0)` where `s = hash_u32(seed_hi, seed_lo, 0)`, `key =
/// hash_u32(id_lo, s, id_hi)` and `base = hash_u32(trajectory, key, 0)`
/// (first cumulative probability above `u`, last legal index on
/// rounding); adds the fields' log-softmax at temperature 1 to the
/// trace-log-probability word (stored as `f32` bits in the record's spare
/// word, since the binding budget leaves no room for a separate scores
/// buffer); applies the token, writes it at `length` and bumps `length`; on
/// STOP sets `finished` and the open valence, and at step `T - 1` without STOP
/// sets `truncated`.
pub fn sample_step<R: Runtime, E: FloatElem>(
    logits: &Tensor<R, E>,
    tables: &Tensor<R, E>,
    traj_meta: &IdTensor<R>,
    state: &mut IdTensor<R>,
    actions: &mut IdTensor<R>,
    step: u32,
    seed_lo: u32,
    seed_hi: u32,
    temperature: f32,
    steps: usize,
    atoms: usize,
    max_closures: u32,
    atom_table: &IdTensor<R>,
) -> Result<()> {
    let rows = traj_meta.len() / TRAJ_META_WIDTH;
    let width = sample_logits_width(atoms);
    if logits.shape().dims() != [rows, width]
        || tables.shape().dims() != [SAMPLE_COND_ROWS, SAMPLE_BOND_WIDTH]
        || traj_meta.shape().dims() != [rows, TRAJ_META_WIDTH]
        || state.shape().dims() != [rows, replay_state_width(atoms)]
        || actions.shape().dims() != [rows, sample_record_width(steps, atoms)]
        || atom_table.len() != 18 * 3
    {
        return Err(Error::shape(format!(
            "sample_step needs logits [rows, {width}], tables [19, 4], traj_meta [rows, 14], state [rows, {}] and actions [rows, {}], got {}/{}/{}/{}/{} and table length {}",
            replay_state_width(atoms),
            sample_record_width(steps, atoms),
            logits.shape(),
            tables.shape(),
            traj_meta.shape(),
            state.shape(),
            actions.shape(),
            atom_table.len()
        )));
    }
    if rows == 0 {
        return Ok(());
    }
    let client = logits.client();
    let (count, dim, span) = launch_1d_spans(client, rows, width);
    unsafe {
        ms2_sample_step_kernel::launch_unchecked::<E, R>(
            client,
            count,
            dim,
            logits.arg(),
            tables.arg(),
            traj_meta.arg(),
            state.arg(),
            actions.arg(),
            atom_table.arg(),
            step,
            seed_lo,
            seed_hi,
            E::from_scalar(temperature),
            steps,
            atoms,
            max_closures,
            rows,
            span,
        );
    }
    Ok(())
}

/// `atoms` as the `u32` the sampler's legality helpers take.
fn max_atoms_u32(atoms: usize) -> u32 {
    atoms as u32
}

/// Lane per trajectory of [`sample_step`]; see its docs for the semantics.
#[allow(clippy::too_many_arguments)]
#[cube(launch_unchecked)]
fn ms2_sample_step_kernel<F: Float + CubeElement>(
    logits: &Array<F>,
    tables: &Array<F>,
    traj: &Array<u32>,
    state: &mut Array<u32>,
    actions: &mut Array<u32>,
    atom_table: &Array<u32>,
    step: u32,
    seed_lo: u32,
    seed_hi: u32,
    temperature: F,
    steps: usize,
    atoms_n: usize,
    max_closures: u32,
    lanes: usize,
    span: usize,
) {
    // The sentinel and the atom cap are fixed by the replay layout: the
    // state width always matches `max_atoms`.
    let sentinel = 4294967295u32;
    let max_atoms = atoms_n as u32;
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    for pos in start..end {
        let r = pos;
        let tbase = r * 14;
        let id_lo = traj[tbase];
        let id_hi = traj[tbase + 1];
        let traj_idx = traj[tbase + 2];
        let started = traj[tbase + 3];
        let sbase = r * (3 * atoms_n + 16);
        let abase = r * (steps * 4 + atoms_n + 4);
        let len_off = abase + steps * 4 + atoms_n;
        let st_off = len_off + 1;
        let lp_off = len_off + 2;
        let st = actions[st_off];
        // Finished, truncated, failed, validation-marked and not-started
        // rows are absorbing and stay untouched: the live rows nest below.
        if started != 0u32 {
            if st & 95u32 == 0u32 {
                // The budget view into the trajectory row: flag, then the 10
                // counts, exactly like the replay meta layout.
                let mbase = tbase + 2;
                let kinds = kind_mask(
                    state,
                    sbase,
                    atom_table,
                    traj,
                    mbase,
                    atoms_n,
                    max_atoms,
                    max_closures,
                    sentinel,
                );
                if kinds == 0u32 {
                    actions[st_off] = st | 4u32;
                } else {
                    let lrow = 27 + 24 * atoms_n;
                    let lbase = r * lrow;
                    let s = hash_u32(seed_hi, seed_lo, 0u32);
                    let key = hash_u32(id_lo, s, id_hi);
                    let draw_base = hash_u32(traj_idx, key, 0u32);
                    // Kind draw over the 5 kind logits.
                    let u_kind: F = F::cast_from(hash_unit_f32(step * 4u32, draw_base, 0u32));
                    let mut k_first = true;
                    let mut k_max = F::new(0.0f32);
                    for i in 0..5usize {
                        if kinds & (1u32 << (i as u32)) != 0u32 {
                            if k_first {
                                k_max = logits[lbase + i];
                                k_first = false;
                            } else if logits[lbase + i] > k_max {
                                k_max = logits[lbase + i];
                            }
                        }
                    }
                    let mut k_sum = F::new(0.0f32);
                    for i in 0..5usize {
                        if kinds & (1u32 << (i as u32)) != 0u32 {
                            k_sum += ((logits[lbase + i] - k_max) / temperature).exp();
                        }
                    }
                    let mut k_cum = F::new(0.0f32);
                    let mut k_pick = 0u32;
                    let mut k_last = 0u32;
                    let mut k_pending = true;
                    for i in 0..5usize {
                        if kinds & (1u32 << (i as u32)) != 0u32 {
                            k_last = i as u32;
                            if k_pending {
                                k_cum += ((logits[lbase + i] - k_max) / temperature).exp() / k_sum;
                                if k_cum > u_kind {
                                    k_pick = i as u32;
                                    k_pending = false;
                                }
                            }
                        }
                    }
                    if k_pending {
                        k_pick = k_last;
                    }
                    let mut k_max1 = F::new(0.0f32);
                    let mut k_first1 = true;
                    for i in 0..5usize {
                        if kinds & (1u32 << (i as u32)) != 0u32 {
                            if k_first1 {
                                k_max1 = logits[lbase + i];
                                k_first1 = false;
                            } else if logits[lbase + i] > k_max1 {
                                k_max1 = logits[lbase + i];
                            }
                        }
                    }
                    let mut k_sum1 = F::new(0.0f32);
                    for i in 0..5usize {
                        if kinds & (1u32 << (i as u32)) != 0u32 {
                            k_sum1 += (logits[lbase + i] - k_max1).exp();
                        }
                    }
                    let mut lp = (logits[lbase + k_pick as usize] - k_max1) - k_sum1.ln();
                    let mut ty = 0u32;
                    let mut b = 0u32;
                    let mut p = 0u32;
                    let is_add = k_pick == 2u32;
                    let is_close = k_pick == 3u32;
                    if is_add {
                        let at_root = step_of(state, sbase, atoms_n) == 1u32;
                        if at_root {
                            // The root uses kind and type only.
                            let ty_mask =
                                root_types(state, sbase, atom_table, traj, mbase, atoms_n);
                            let u_ty: F =
                                F::cast_from(hash_unit_f32(step * 4u32 + 1u32, draw_base, 0u32));
                            let mut t_first = true;
                            let mut t_max = F::new(0.0f32);
                            for i in 0..18usize {
                                if ty_mask & (1u32 << (i as u32)) != 0u32 {
                                    if t_first {
                                        t_max = logits[lbase + 5 + i];
                                        t_first = false;
                                    } else if logits[lbase + 5 + i] > t_max {
                                        t_max = logits[lbase + 5 + i];
                                    }
                                }
                            }
                            let mut t_sum = F::new(0.0f32);
                            for i in 0..18usize {
                                if ty_mask & (1u32 << (i as u32)) != 0u32 {
                                    t_sum += ((logits[lbase + 5 + i] - t_max) / temperature).exp();
                                }
                            }
                            let mut t_cum = F::new(0.0f32);
                            let mut t_last = 0u32;
                            let mut t_pending = true;
                            for i in 0..18usize {
                                if ty_mask & (1u32 << (i as u32)) != 0u32 {
                                    t_last = i as u32;
                                    if t_pending {
                                        t_cum += ((logits[lbase + 5 + i] - t_max) / temperature)
                                            .exp()
                                            / t_sum;
                                        if t_cum > u_ty {
                                            ty = i as u32;
                                            t_pending = false;
                                        }
                                    }
                                }
                            }
                            if t_pending {
                                ty = t_last;
                            }
                            let mut t_max1 = F::new(0.0f32);
                            let mut t_first1 = true;
                            for i in 0..18usize {
                                if ty_mask & (1u32 << (i as u32)) != 0u32 {
                                    if t_first1 {
                                        t_max1 = logits[lbase + 5 + i];
                                        t_first1 = false;
                                    } else if logits[lbase + 5 + i] > t_max1 {
                                        t_max1 = logits[lbase + 5 + i];
                                    }
                                }
                            }
                            let mut t_sum1 = F::new(0.0f32);
                            for i in 0..18usize {
                                if ty_mask & (1u32 << (i as u32)) != 0u32 {
                                    t_sum1 += (logits[lbase + 5 + i] - t_max1).exp();
                                }
                            }
                            lp = lp + (logits[lbase + 5 + ty as usize] - t_max1) - t_sum1.ln();
                        } else {
                            // The non-root type mask: every type with some bond
                            // and pointer completion, like TraceState::masks.
                            let mut ty_mask = 0u32;
                            for id in 1usize..18usize {
                                if type_fits(
                                    state, sbase, atom_table, traj, mbase, atoms_n, id as u32,
                                ) != 0u32
                                {
                                    let val = atom_table[id * 3 + 2];
                                    let hyd = atom_table[id * 3 + 1];
                                    for bond in 1u32..4u32 {
                                        if val >= hyd && bond <= val - hyd {
                                            if add_pointers(state, sbase, atoms_n, bond, max_atoms)
                                                != 0u32
                                            {
                                                ty_mask |= 1u32 << (id as u32);
                                            }
                                        }
                                    }
                                }
                            }
                            let u_ty: F =
                                F::cast_from(hash_unit_f32(step * 4u32 + 1u32, draw_base, 0u32));
                            let mut t_first = true;
                            let mut t_max = F::new(0.0f32);
                            for i in 0..18usize {
                                if ty_mask & (1u32 << (i as u32)) != 0u32 {
                                    if t_first {
                                        t_max = logits[lbase + 5 + i];
                                        t_first = false;
                                    } else if logits[lbase + 5 + i] > t_max {
                                        t_max = logits[lbase + 5 + i];
                                    }
                                }
                            }
                            let mut t_sum = F::new(0.0f32);
                            for i in 0..18usize {
                                if ty_mask & (1u32 << (i as u32)) != 0u32 {
                                    t_sum += ((logits[lbase + 5 + i] - t_max) / temperature).exp();
                                }
                            }
                            let mut t_cum = F::new(0.0f32);
                            let mut t_last = 0u32;
                            let mut t_pending = true;
                            for i in 0..18usize {
                                if ty_mask & (1u32 << (i as u32)) != 0u32 {
                                    t_last = i as u32;
                                    if t_pending {
                                        t_cum += ((logits[lbase + 5 + i] - t_max) / temperature)
                                            .exp()
                                            / t_sum;
                                        if t_cum > u_ty {
                                            ty = i as u32;
                                            t_pending = false;
                                        }
                                    }
                                }
                            }
                            if t_pending {
                                ty = t_last;
                            }
                            let mut t_max1 = F::new(0.0f32);
                            let mut t_first1 = true;
                            for i in 0..18usize {
                                if ty_mask & (1u32 << (i as u32)) != 0u32 {
                                    if t_first1 {
                                        t_max1 = logits[lbase + 5 + i];
                                        t_first1 = false;
                                    } else if logits[lbase + 5 + i] > t_max1 {
                                        t_max1 = logits[lbase + 5 + i];
                                    }
                                }
                            }
                            let mut t_sum1 = F::new(0.0f32);
                            for i in 0..18usize {
                                if ty_mask & (1u32 << (i as u32)) != 0u32 {
                                    t_sum1 += (logits[lbase + 5 + i] - t_max1).exp();
                                }
                            }
                            lp = lp + (logits[lbase + 5 + ty as usize] - t_max1) - t_sum1.ln();
                            // The conditioning row for the bond and pointer
                            // tables: the sampled atom type.
                            let c = ty;
                            // The bond order from bond_base + bond_by_type[c].
                            let ty_idx = ty as usize;
                            let val = atom_table[ty_idx * 3 + 2];
                            let hyd = atom_table[ty_idx * 3 + 1];
                            let mut b_mask = 0u32;
                            for bond in 1u32..4u32 {
                                if val >= hyd && bond <= val - hyd {
                                    if add_pointers(state, sbase, atoms_n, bond, max_atoms) != 0u32
                                    {
                                        b_mask |= 1u32 << bond;
                                    }
                                }
                            }
                            let u_bond: F =
                                F::cast_from(hash_unit_f32(step * 4u32 + 2u32, draw_base, 0u32));
                            let mut b_first = true;
                            let mut b_max = F::new(0.0f32);
                            for i in 0..4usize {
                                if b_mask & (1u32 << (i as u32)) != 0u32 {
                                    let li = logits[lbase + 23 + i] + tables[c as usize * 4 + i];
                                    if b_first {
                                        b_max = li;
                                        b_first = false;
                                    } else if li > b_max {
                                        b_max = li;
                                    }
                                }
                            }
                            let mut b_sum = F::new(0.0f32);
                            for i in 0..4usize {
                                if b_mask & (1u32 << (i as u32)) != 0u32 {
                                    let li = logits[lbase + 23 + i] + tables[c as usize * 4 + i];
                                    b_sum += ((li - b_max) / temperature).exp();
                                }
                            }
                            let mut b_cum = F::new(0.0f32);
                            let mut b_last = 0u32;
                            let mut b_pending = true;
                            for i in 0..4usize {
                                if b_mask & (1u32 << (i as u32)) != 0u32 {
                                    b_last = i as u32;
                                    if b_pending {
                                        let li =
                                            logits[lbase + 23 + i] + tables[c as usize * 4 + i];
                                        b_cum += ((li - b_max) / temperature).exp() / b_sum;
                                        if b_cum > u_bond {
                                            b = i as u32;
                                            b_pending = false;
                                        }
                                    }
                                }
                            }
                            if b_pending {
                                b = b_last;
                            }
                            let mut b_max1 = F::new(0.0f32);
                            let mut b_first1 = true;
                            for i in 0..4usize {
                                if b_mask & (1u32 << (i as u32)) != 0u32 {
                                    let li = logits[lbase + 23 + i] + tables[c as usize * 4 + i];
                                    if b_first1 {
                                        b_max1 = li;
                                        b_first1 = false;
                                    } else if li > b_max1 {
                                        b_max1 = li;
                                    }
                                }
                            }
                            let mut b_sum1 = F::new(0.0f32);
                            for i in 0..4usize {
                                if b_mask & (1u32 << (i as u32)) != 0u32 {
                                    let li = logits[lbase + 23 + i] + tables[c as usize * 4 + i];
                                    b_sum1 += (li - b_max1).exp();
                                }
                            }
                            let lb = logits[lbase + 23 + b as usize]
                                + tables[c as usize * 4 + b as usize];
                            lp = lp + (lb - b_max1) - b_sum1.ln();
                            // The pointer from the three query parts.
                            let p_mask = add_pointers(state, sbase, atoms_n, b, max_atoms);
                            let u_ptr: F =
                                F::cast_from(hash_unit_f32(step * 4u32 + 3u32, draw_base, 0u32));
                            let pb = lbase + 27;
                            let pt = lbase + 27 + atoms_n;
                            let pn = lbase + 27 + atoms_n + 19 * atoms_n;
                            let mut p_first = true;
                            let mut p_max = F::new(0.0f32);
                            for i in 0..atoms_n {
                                if p_mask & (1u32 << (i as u32)) != 0u32 {
                                    let li = logits[pb + i]
                                        + logits[pt + c as usize * atoms_n + i]
                                        + logits[pn + b as usize * atoms_n + i];
                                    if p_first {
                                        p_max = li;
                                        p_first = false;
                                    } else if li > p_max {
                                        p_max = li;
                                    }
                                }
                            }
                            let mut p_sum = F::new(0.0f32);
                            for i in 0..atoms_n {
                                if p_mask & (1u32 << (i as u32)) != 0u32 {
                                    let li = logits[pb + i]
                                        + logits[pt + c as usize * atoms_n + i]
                                        + logits[pn + b as usize * atoms_n + i];
                                    p_sum += ((li - p_max) / temperature).exp();
                                }
                            }
                            let mut p_cum = F::new(0.0f32);
                            let mut p_last = 0u32;
                            let mut p_pending = true;
                            for i in 0..atoms_n {
                                if p_mask & (1u32 << (i as u32)) != 0u32 {
                                    p_last = i as u32;
                                    if p_pending {
                                        let li = logits[pb + i]
                                            + logits[pt + c as usize * atoms_n + i]
                                            + logits[pn + b as usize * atoms_n + i];
                                        p_cum += ((li - p_max) / temperature).exp() / p_sum;
                                        if p_cum > u_ptr {
                                            p = i as u32;
                                            p_pending = false;
                                        }
                                    }
                                }
                            }
                            if p_pending {
                                p = p_last;
                            }
                            let mut p_max1 = F::new(0.0f32);
                            let mut p_first1 = true;
                            for i in 0..atoms_n {
                                if p_mask & (1u32 << (i as u32)) != 0u32 {
                                    let li = logits[pb + i]
                                        + logits[pt + c as usize * atoms_n + i]
                                        + logits[pn + b as usize * atoms_n + i];
                                    if p_first1 {
                                        p_max1 = li;
                                        p_first1 = false;
                                    } else if li > p_max1 {
                                        p_max1 = li;
                                    }
                                }
                            }
                            let mut p_sum1 = F::new(0.0f32);
                            for i in 0..atoms_n {
                                if p_mask & (1u32 << (i as u32)) != 0u32 {
                                    let li = logits[pb + i]
                                        + logits[pt + c as usize * atoms_n + i]
                                        + logits[pn + b as usize * atoms_n + i];
                                    p_sum1 += (li - p_max1).exp();
                                }
                            }
                            let lptr = logits[pb + p as usize]
                                + logits[pt + c as usize * atoms_n + p as usize]
                                + logits[pn + b as usize * atoms_n + p as usize];
                            lp = lp + (lptr - p_max1) - p_sum1.ln();
                        }
                    } else if is_close {
                        // CLOSE_RING uses kind, bond and pointer; c is row 18.
                        let c = 18u32;
                        let mut b_mask = 0u32;
                        for bond in 1u32..4u32 {
                            if close_pointers(state, sbase, atoms_n, bond, max_closures, sentinel)
                                != 0u32
                            {
                                b_mask |= 1u32 << bond;
                            }
                        }
                        let u_bond: F =
                            F::cast_from(hash_unit_f32(step * 4u32 + 2u32, draw_base, 0u32));
                        let mut b_first = true;
                        let mut b_max = F::new(0.0f32);
                        for i in 0..4usize {
                            if b_mask & (1u32 << (i as u32)) != 0u32 {
                                let li = logits[lbase + 23 + i] + tables[c as usize * 4 + i];
                                if b_first {
                                    b_max = li;
                                    b_first = false;
                                } else if li > b_max {
                                    b_max = li;
                                }
                            }
                        }
                        let mut b_sum = F::new(0.0f32);
                        for i in 0..4usize {
                            if b_mask & (1u32 << (i as u32)) != 0u32 {
                                let li = logits[lbase + 23 + i] + tables[c as usize * 4 + i];
                                b_sum += ((li - b_max) / temperature).exp();
                            }
                        }
                        let mut b_cum = F::new(0.0f32);
                        let mut b_last = 0u32;
                        let mut b_pending = true;
                        for i in 0..4usize {
                            if b_mask & (1u32 << (i as u32)) != 0u32 {
                                b_last = i as u32;
                                if b_pending {
                                    let li = logits[lbase + 23 + i] + tables[c as usize * 4 + i];
                                    b_cum += ((li - b_max) / temperature).exp() / b_sum;
                                    if b_cum > u_bond {
                                        b = i as u32;
                                        b_pending = false;
                                    }
                                }
                            }
                        }
                        if b_pending {
                            b = b_last;
                        }
                        let mut b_max1 = F::new(0.0f32);
                        let mut b_first1 = true;
                        for i in 0..4usize {
                            if b_mask & (1u32 << (i as u32)) != 0u32 {
                                let li = logits[lbase + 23 + i] + tables[c as usize * 4 + i];
                                if b_first1 {
                                    b_max1 = li;
                                    b_first1 = false;
                                } else if li > b_max1 {
                                    b_max1 = li;
                                }
                            }
                        }
                        let mut b_sum1 = F::new(0.0f32);
                        for i in 0..4usize {
                            if b_mask & (1u32 << (i as u32)) != 0u32 {
                                let li = logits[lbase + 23 + i] + tables[c as usize * 4 + i];
                                b_sum1 += (li - b_max1).exp();
                            }
                        }
                        let lb =
                            logits[lbase + 23 + b as usize] + tables[c as usize * 4 + b as usize];
                        lp = lp + (lb - b_max1) - b_sum1.ln();
                        let p_mask =
                            close_pointers(state, sbase, atoms_n, b, max_closures, sentinel);
                        let u_ptr: F =
                            F::cast_from(hash_unit_f32(step * 4u32 + 3u32, draw_base, 0u32));
                        let pb = lbase + 27;
                        let pt = lbase + 27 + atoms_n;
                        let pn = lbase + 27 + atoms_n + 19 * atoms_n;
                        let mut p_first = true;
                        let mut p_max = F::new(0.0f32);
                        for i in 0..atoms_n {
                            if p_mask & (1u32 << (i as u32)) != 0u32 {
                                let li = logits[pb + i]
                                    + logits[pt + c as usize * atoms_n + i]
                                    + logits[pn + b as usize * atoms_n + i];
                                if p_first {
                                    p_max = li;
                                    p_first = false;
                                } else if li > p_max {
                                    p_max = li;
                                }
                            }
                        }
                        let mut p_sum = F::new(0.0f32);
                        for i in 0..atoms_n {
                            if p_mask & (1u32 << (i as u32)) != 0u32 {
                                let li = logits[pb + i]
                                    + logits[pt + c as usize * atoms_n + i]
                                    + logits[pn + b as usize * atoms_n + i];
                                p_sum += ((li - p_max) / temperature).exp();
                            }
                        }
                        let mut p_cum = F::new(0.0f32);
                        let mut p_last = 0u32;
                        let mut p_pending = true;
                        for i in 0..atoms_n {
                            if p_mask & (1u32 << (i as u32)) != 0u32 {
                                p_last = i as u32;
                                if p_pending {
                                    let li = logits[pb + i]
                                        + logits[pt + c as usize * atoms_n + i]
                                        + logits[pn + b as usize * atoms_n + i];
                                    p_cum += ((li - p_max) / temperature).exp() / p_sum;
                                    if p_cum > u_ptr {
                                        p = i as u32;
                                        p_pending = false;
                                    }
                                }
                            }
                        }
                        if p_pending {
                            p = p_last;
                        }
                        let mut p_max1 = F::new(0.0f32);
                        let mut p_first1 = true;
                        for i in 0..atoms_n {
                            if p_mask & (1u32 << (i as u32)) != 0u32 {
                                let li = logits[pb + i]
                                    + logits[pt + c as usize * atoms_n + i]
                                    + logits[pn + b as usize * atoms_n + i];
                                if p_first1 {
                                    p_max1 = li;
                                    p_first1 = false;
                                } else if li > p_max1 {
                                    p_max1 = li;
                                }
                            }
                        }
                        let mut p_sum1 = F::new(0.0f32);
                        for i in 0..atoms_n {
                            if p_mask & (1u32 << (i as u32)) != 0u32 {
                                let li = logits[pb + i]
                                    + logits[pt + c as usize * atoms_n + i]
                                    + logits[pn + b as usize * atoms_n + i];
                                p_sum1 += (li - p_max1).exp();
                            }
                        }
                        let lptr = logits[pb + p as usize]
                            + logits[pt + c as usize * atoms_n + p as usize]
                            + logits[pn + b as usize * atoms_n + p as usize];
                        lp = lp + (lptr - p_max1) - p_sum1.ln();
                    }
                    // The sampled fields complete a legal token by construction of
                    // the kind mask, so apply it, write it at length and bump.
                    apply_token(state, sbase, atom_table, atoms_n, k_pick, ty, b, p);
                    let len = actions[len_off];
                    actions[abase + (len as usize) * 4] = k_pick;
                    actions[abase + (len as usize) * 4 + 1] = ty;
                    actions[abase + (len as usize) * 4 + 2] = b;
                    actions[abase + (len as usize) * 4 + 3] = p;
                    actions[len_off] = len + 1u32;
                    let old_lp = F::reinterpret(actions[lp_off]);
                    actions[lp_off] = u32::reinterpret(old_lp + lp);
                    if k_pick == 4u32 {
                        actions[st_off] = st | 1u32;
                        let n = n_atoms(state, sbase, atoms_n) as usize;
                        for j in 0..atoms_n {
                            let mut v = 0u32;
                            if j < n {
                                v = resid_of(state, sbase, atoms_n, j);
                            }
                            actions[abase + steps * 4 + j] = v;
                        }
                    } else if step == (steps as u32) - 1u32 {
                        actions[st_off] = st | 2u32;
                    }
                }
            }
        }
    }
}

/// Initialise one `(B, K)` bucket of trajectories, lane per trajectory.
///
/// Zeroes the grammar `state` row and the `actions` record, writes START at
/// position 0 with `length = 1` for started rows, and fills `traj_meta` with
/// the spectrum id, trajectory index, started flag and the 10 budget counts
/// of the conditioning formula (`top_counts[b, k mod count]`, V1 §1.2).
/// Trajectory `k` still uses formula `k mod top_count`. A row whose spectrum
/// failed never starts: `peak_count == 0` (the upload
/// writes 0 for every fatal host status, and an empty spectrum is fatal) or
/// `count == 0` (no scored formula, which is exactly the device-side
/// `formula_absent`), unless `metadata_only` bypasses the empty-spectrum
/// abstention. Failed rows keep `request_failed` with `length = 0` and
/// `formula_row = u32::MAX`.
///
/// Bindings (6): `top` (`[B, F, 2]`), `spectra_meta` (`[B, 8]`, carrying the
/// peak count and the id halves), `top_counts` (u32, `[B, F, 10]` retained
/// counts, so exact chemistry budgets never depend on the
/// neural float path), `traj_meta`, `state` and `actions` (all in/out).
/// Exactly 1 launch.
#[allow(clippy::too_many_arguments)]
/// Initialise one `(B, K)` bucket of trajectories from the allocation
/// buffer (V1 §3.2), lane per trajectory.
///
/// `traj_formula` is the `[B, K, 12]` buffer `ms2_allocate` wrote (retained
/// formula slot, source row, 10 counts; slot `u32::MAX` when there is none),
/// `spectra_meta` is `[B, 8]`. Returns `(traj_meta, state, actions)` as
/// before. A row starts exactly when its allocation slot is not the sentinel
/// and the spectrum has peaks (`metadata_only` bypasses the empty-spectrum
/// abstention, as before); the conditioning budgets come from the
/// allocation's 10 counts and `formula_row` is its source row. With
/// `RoundRobin` the slot is `k mod count`, so every result is bit-identical
/// to the V0 path that read `top`/`top_counts` directly.
///
/// Bindings (5): `traj_formula`, `spectra_meta`, `traj_meta`, `state`,
/// `actions` (the last three in/out). Exactly 1 launch.
pub fn init_trajectories<R: Runtime>(
    traj_formula: &IdTensor<R>,
    spectra_meta: &IdTensor<R>,
    traj_meta: &mut IdTensor<R>,
    state: &mut IdTensor<R>,
    actions: &mut IdTensor<R>,
    spectra: usize,
    per_spectrum: usize,
    steps: usize,
    atoms: usize,
    metadata_only: bool,
) -> Result<()> {
    let rows = spectra * per_spectrum;
    if traj_formula.shape().dims() != [spectra, per_spectrum, 12]
        || spectra_meta.shape().dims() != [spectra, META_WIDTH]
        || traj_meta.shape().dims() != [rows, TRAJ_META_WIDTH]
        || state.shape().dims() != [rows, replay_state_width(atoms)]
        || actions.shape().dims() != [rows, sample_record_width(steps, atoms)]
    {
        return Err(Error::shape(format!(
            "init_trajectories needs traj_formula [B, K, 12], spectra_meta [B, 8], traj_meta [B*K, 14], state [B*K, {}] and actions [B*K, {}], got {} and {} and {} and {} and {}",
            replay_state_width(atoms),
            sample_record_width(steps, atoms),
            traj_formula.shape(),
            spectra_meta.shape(),
            traj_meta.shape(),
            state.shape(),
            actions.shape()
        )));
    }
    if rows == 0 {
        return Ok(());
    }
    let client = traj_formula.client();
    let (count, dim, span) = launch_1d_spans(client, rows, TRAJ_META_WIDTH);
    unsafe {
        ms2_init_trajectories_kernel::launch_unchecked::<R>(
            client,
            count,
            dim,
            traj_formula.arg(),
            spectra_meta.arg(),
            traj_meta.arg(),
            state.arg(),
            actions.arg(),
            per_spectrum,
            steps,
            atoms,
            u32::from(metadata_only),
            rows,
            span,
        );
    }
    Ok(())
}

/// Lane per trajectory of [`init_trajectories`]; see its docs for the
/// semantics. Arrays: `traj_formula`, `spectra_meta`, `traj_meta`,
/// `state`, `actions` (the last three in/out).
#[allow(clippy::too_many_arguments)]
#[allow(clippy::assign_op_pattern)]
#[cube(launch_unchecked)]
fn ms2_init_trajectories_kernel(
    traj_formula: &Array<u32>,
    spectra_meta: &Array<u32>,
    traj_meta: &mut Array<u32>,
    state: &mut Array<u32>,
    actions: &mut Array<u32>,
    per_spectrum: usize,
    steps: usize,
    atoms_n: usize,
    metadata_only: u32,
    lanes: usize,
    span: usize,
) {
    // Widths from the atom and step counts; the sentinel is the padding word.
    let sentinel = 4294967295u32;
    let state_width = 3 * atoms_n + 16;
    let record_width = steps * 4 + atoms_n + 4;
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    for pos in start..end {
        let r = pos;
        let b = r / per_spectrum;
        let k = (r % per_spectrum) as u32;
        // This trajectory's allocation: the retained formula slot, the
        // source row and the 10 conditioning counts (V1 §3.2). A sentinel
        // slot means no scored formula, whatever the spectrum's peaks.
        let tfbase = r * 12;
        let fslot = traj_formula[tfbase];
        let frow = traj_formula[tfbase + 1];
        let peak_count = spectra_meta[b * 8];
        let sbase = r * state_width;
        for w in 0..3 * atoms_n + 16 {
            state[sbase + w] = 0u32;
        }
        let abase = r * record_width;
        for w in 0..steps * 4 + atoms_n + 4 {
            actions[abase + w] = 0u32;
        }
        let tbase = r * 14;
        for w in 0..14usize {
            traj_meta[tbase + w] = 0u32;
        }
        let len_off = abase + steps * 4 + atoms_n;
        // A row starts when the allocation names a formula and the spectrum
        // has peaks (`metadata_only` bypasses the empty-spectrum abstention).
        let mut go = false;
        if fslot != sentinel {
            if peak_count != 0u32 {
                go = true;
            }
            if metadata_only != 0u32 {
                go = true;
            }
        }
        if go {
            // Budgets come from the allocation record (V1 §3.2), never from a
            // table lookup: exact chemistry without a float round-trip.
            state[sbase + 3 * atoms_n + 4] = 1u32;
            actions[abase] = 1u32;
            actions[len_off] = 1u32;
            actions[len_off + 3] = frow;
            traj_meta[tbase] = spectra_meta[b * 8 + 6];
            traj_meta[tbase + 1] = spectra_meta[b * 8 + 7];
            traj_meta[tbase + 2] = k;
            traj_meta[tbase + 3] = 1u32;
            for e in 0..10usize {
                traj_meta[tbase + 4 + e] = traj_formula[tfbase + 2 + e];
            }
        } else {
            actions[len_off + 1] = 64u32;
            actions[len_off + 3] = sentinel;
        }
    }
}

/// Validate one `(B, K)` bucket of trajectories, lane per trajectory.
///
/// Replays the emitted trace from scratch with the shared legality helpers
/// into the lane's own `scratch` row and sets `invalid_final` when any step
/// is illegal or the final rules of contracts §4.5 fail (at least one atom,
/// the atom and closure caps, the composition within the formula hypothesis).
/// Then compares the trace and formula row with every earlier trajectory of
/// the same spectrum and sets `duplicate_trace` on an exact match. Failed
/// requests stay untouched.
///
/// Bindings (4): `actions` (in/out), `traj_meta` (budgets), `atom_table` and
/// `scratch` (in/out). Exactly 1 launch.
#[allow(clippy::too_many_arguments)]
pub fn validate_trajectories<R: Runtime>(
    actions: &mut IdTensor<R>,
    traj_meta: &IdTensor<R>,
    scratch: &mut IdTensor<R>,
    atom_table: &IdTensor<R>,
    spectra: usize,
    per_spectrum: usize,
    steps: usize,
    atoms: usize,
    max_closures: u32,
) -> Result<()> {
    let rows = spectra * per_spectrum;
    if actions.shape().dims() != [rows, sample_record_width(steps, atoms)]
        || traj_meta.shape().dims() != [rows, TRAJ_META_WIDTH]
        || scratch.shape().dims() != [rows, replay_state_width(atoms)]
        || atom_table.len() != 18 * 3
    {
        return Err(Error::shape(format!(
            "validate_trajectories needs actions [rows, {}], traj_meta [rows, 14], scratch [rows, {}] and the 54-element atom table, got {} and {} and {} and length {}",
            sample_record_width(steps, atoms),
            replay_state_width(atoms),
            actions.shape(),
            traj_meta.shape(),
            scratch.shape(),
            atom_table.len()
        )));
    }
    if rows == 0 {
        return Ok(());
    }
    let client = actions.client();
    let (count, dim, span) = launch_1d_spans(client, rows, steps * 4);
    unsafe {
        ms2_validate_kernel::launch_unchecked::<R>(
            client,
            count,
            dim,
            actions.arg(),
            traj_meta.arg(),
            scratch.arg(),
            atom_table.arg(),
            per_spectrum,
            steps,
            atoms,
            max_atoms_u32(atoms),
            max_closures,
            replay_state_width(atoms),
            sample_record_width(steps, atoms),
            u32::MAX,
            rows,
            span,
        );
    }
    Ok(())
}

/// Lane per trajectory of [`validate_trajectories`]; see its docs for the
/// semantics. Arrays: `actions` (in/out), `traj_meta`, `scratch` (in/out),
/// `atom_table`.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::bool_comparison)]
#[allow(clippy::assign_op_pattern)]
#[cube(launch_unchecked)]
fn ms2_validate_kernel(
    actions: &mut Array<u32>,
    traj_meta: &Array<u32>,
    scratch: &mut Array<u32>,
    atom_table: &Array<u32>,
    per_spectrum: usize,
    steps: usize,
    atoms_n: usize,
    max_atoms: u32,
    max_closures: u32,
    state_width: usize,
    record_width: usize,
    sentinel: u32,
    lanes: usize,
    span: usize,
) {
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    for pos in start..end {
        let r = pos;
        let abase = r * record_width;
        let len_off = abase + steps * 4 + atoms_n;
        let st_off = len_off + 1;
        let form_off = len_off + 3;
        let st = actions[st_off];
        if st & 64u32 == 0u32 {
            let ssbase = r * state_width;
            for w in 0..state_width {
                scratch[ssbase + w] = 0u32;
            }
            let mbase = r * 14 + 2;
            let len = actions[len_off] as usize;
            let mut bad = false;
            for tt in 0..len {
                if bad == false {
                    let k = actions[abase + tt * 4];
                    let ty = actions[abase + tt * 4 + 1];
                    let b = actions[abase + tt * 4 + 2];
                    let p = actions[abase + tt * 4 + 3];
                    if is_legal(
                        scratch,
                        ssbase,
                        atom_table,
                        traj_meta,
                        mbase,
                        atoms_n,
                        max_atoms,
                        max_closures,
                        sentinel,
                        k,
                        ty,
                        b,
                        p,
                    ) != 0u32
                    {
                        apply_token(scratch, ssbase, atom_table, atoms_n, k, ty, b, p);
                    } else {
                        bad = true;
                    }
                }
            }
            let n = n_atoms(scratch, ssbase, atoms_n);
            if n == 0u32 {
                bad = true;
            }
            if n > max_atoms {
                bad = true;
            }
            if closures(scratch, ssbase, atoms_n) > max_closures {
                bad = true;
            }
            for e in 0..10usize {
                if used_of(scratch, ssbase, atoms_n, e) > budget_of(traj_meta, mbase, e) {
                    bad = true;
                }
            }
            // A record claimed `finished` (status bit 0) must end in STOP
            // (kind 4): a legal truncated history carries `truncated`, not
            // `finished`, so it is unaffected by this check.
            if st & 1u32 != 0u32 {
                if len == 0usize {
                    bad = true;
                } else {
                    let last = abase + (len - 1usize) * 4;
                    if actions[last] != 4u32 {
                        bad = true;
                    }
                }
            }
            let b = r / per_spectrum;
            let k = r % per_spectrum;
            let len_rel = steps * 4 + atoms_n;
            let st_rel = len_rel + 1;
            let form_rel = len_rel + 3;
            let mut dup = false;
            for k2 in 0..k {
                if dup == false {
                    let r2 = b * per_spectrum + k2;
                    let a2 = r2 * record_width;
                    if actions[a2 + st_rel] & 64u32 == 0u32 {
                        if actions[a2 + form_rel] == actions[form_off] {
                            // D2: compare the conditioning FORMULA, not just
                            // `formula_row` (which is `u32::MAX` for every
                            // enumerated formula). The 10 budget counts in
                            // `traj_meta` are the formula identity the kernel
                            // can bind within its array limit; requiring both
                            // keeps table-source results bit-identical (same
                            // row implies same counts) while distinguishing
                            // different enumerated formulas.
                            let mut formula_same: u32 = 1u32;
                            for e in 0..10usize {
                                if budget_of(traj_meta, mbase, e)
                                    != budget_of(traj_meta, r2 * 14 + 2, e)
                                {
                                    formula_same = 0u32;
                                }
                            }
                            if formula_same != 0u32 {
                                if actions[a2 + len_rel] == actions[len_off] {
                                    let mut same = true;
                                    for w in 0..steps * 4 {
                                        if actions[a2 + w] != actions[abase + w] {
                                            same = false;
                                        }
                                    }
                                    if same {
                                        dup = true;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            let mut out = st;
            if bad {
                out = out | 8u32;
            }
            if dup {
                out = out | 16u32;
            }
            actions[st_off] = out;
        }
    }
}

// ---------------------------------------------------------------------------
// Fused sampler step (P8 / O4): the single-position decoder step as a handful
// of kernels instead of the composed tensor ops
// ---------------------------------------------------------------------------
// One sampling step at a single position is launch-bound, not work-bound: the
// composed form issued about 155 launches for a few hundred thousand
// multiply-adds. The kernels below compute the same values as the composed
// ops they replace (the composed path stays as the reference; the parity
// tests compare the two) with one launch per stage:
//
// * [`step_embed`]: the six input embeddings and their sum;
// * [`attn_weights`], [`attn_context`]: masked single-query cross-attention
//   over the cached keys and values, without the per-step head permutes;
// * [`atom_key_update`]: the projected atom-memory row of an added atom and
//   the clamped residual ids;
// * [`step_logits_pack`]: the head logits and the three pointer score blocks,
//   written straight into the packed sampler row;
// * [`freeze_rows`]: the carry freeze of stopped rows, in place.

/// Rows of the fused embedding table before the pointer rows: kind (5), atom
/// type (18), bond (4).
pub const STEP_EMBED_FIXED_ROWS: usize = 27;
/// Rows of the residual embedding at the head of the fused pointer table,
/// followed by the 19 type-conditioning and the 4 bond-conditioning rows.
pub const STEP_PTR_RESID_ROWS: usize = 8;

/// Lane per output element of [`step_embed`].
#[allow(clippy::too_many_arguments)]
#[cube(launch_unchecked)]
fn ms2_step_embed_kernel<F: Float + CubeElement>(
    tables: &Array<F>,
    token: &Array<u32>,
    formula: &Array<F>,
    out: &mut Array<F>,
    d: usize,
    atoms_n: usize,
    position: usize,
    lanes: usize,
    span: usize,
) {
    let zero = F::new(0.0_f32);
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    for pos in start..end {
        let r = pos / d;
        let col = pos % d;
        let k = token[r * 4] as usize;
        let ty = token[r * 4 + 1] as usize;
        let b = token[r * 4 + 2] as usize;
        let p = token[r * 4 + 3] as usize;
        let mut e_kind = zero;
        if k < 5 {
            e_kind = tables[k * d + col];
        }
        let mut e_type = zero;
        if ty < 18 {
            e_type = tables[(5 + ty) * d + col];
        }
        let mut e_bond = zero;
        if b < 4 {
            e_bond = tables[(23 + b) * d + col];
        }
        let mut e_ptr = zero;
        if p < atoms_n {
            e_ptr = tables[(27 + p) * d + col];
        }
        let e_step = tables[(27 + atoms_n + position) * d + col];
        out[pos] = e_kind + e_type + e_bond + e_ptr + e_step + formula[pos];
    }
}

/// The sampler's input embedding at one position in one launch:
/// `E_kind[kind] + E_type[type] + E_bond[bond] + E_pointer[pointer] +
/// E_step[position] + formula`, summed in that order, with an out-of-range
/// field id contributing a zero row (the composed lookups' rule).
///
/// `tables` is the row concatenation `kind (5) | type (18) | bond (4) |
/// pointer (A) | step (S)` as `[27 + A + S, d]`, `token` is `[rows, 4]` and
/// `formula` `[rows, d]`. Output `[rows, d]`.
pub fn step_embed<R: Runtime, E: FloatElem>(
    tables: &Tensor<R, E>,
    token: &IdTensor<R>,
    formula: &Tensor<R, E>,
    position: usize,
    atoms_n: usize,
) -> Result<Tensor<R, E>> {
    if tables.rank() != 2 || token.shape().rank() != 2 || formula.rank() != 2 {
        return Err(Error::shape(format!(
            "step_embed needs tables [27 + A + S, d], token [rows, 4] and formula [rows, d], got {} and {} and {}",
            tables.shape(),
            token.shape(),
            formula.shape()
        )));
    }
    let d = tables.shape().dim(1);
    let rows = token.shape().dim(0);
    let fixed = STEP_EMBED_FIXED_ROWS + atoms_n;
    if token.shape().dim(1) != 4
        || formula.shape().dims() != [rows, d]
        || tables.shape().dim(0) <= fixed
        || position >= tables.shape().dim(0) - fixed
    {
        return Err(Error::shape(format!(
            "step_embed has mismatched shapes: tables {}, token {}, formula {}, A = {atoms_n}, position {position}",
            tables.shape(),
            token.shape(),
            formula.shape()
        )));
    }
    let out = Tensor::empty(Shape::new(vec![rows, d]), tables.device());
    if out.is_empty() {
        return Ok(out);
    }
    let lanes = out.len();
    let (count, dim, span) = launch_1d_spans(tables.client(), lanes, 1);
    unsafe {
        ms2_step_embed_kernel::launch_unchecked::<E, R>(
            tables.client(),
            count,
            dim,
            tables.arg(),
            token.arg(),
            formula.arg(),
            out.arg(),
            d,
            atoms_n,
            position,
            lanes,
            span,
        );
    }
    Ok(out)
}

/// Lane per `(row, head, memory slot)` of [`attn_weights`]: the masked,
/// scaled score of one query head against one key. The `hd`-long dot product
/// loads eight pairs per round, so a lane waits on memory `hd / 8` times
/// rather than `hd` times.
#[allow(clippy::too_many_arguments)]
#[cube(launch_unchecked)]
fn ms2_attn_scores_kernel<F: Float + CubeElement>(
    q: &Array<F>,
    k: &Array<F>,
    mask: &Array<F>,
    w: &mut Array<F>,
    d: usize,
    heads: usize,
    mem: usize,
    rows_per_spectrum: usize,
    scale: F,
    lanes: usize,
    span: usize,
) {
    let hd = d / heads;
    let chunks = hd / 8;
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    for lane in start..end {
        let m = lane % mem;
        let rh = lane / mem;
        let row = rh / heads;
        let head = rh % heads;
        let b = row / rows_per_spectrum;
        let qbase = row * d + head * hd;
        let kbase = (b * mem + m) * d + head * hd;
        let mut s0 = F::new(0.0_f32);
        let mut s1 = F::new(0.0_f32);
        let mut s2 = F::new(0.0_f32);
        let mut s3 = F::new(0.0_f32);
        let mut s4 = F::new(0.0_f32);
        let mut s5 = F::new(0.0_f32);
        let mut s6 = F::new(0.0_f32);
        let mut s7 = F::new(0.0_f32);
        for c in 0..chunks {
            let i = c * 8;
            s0 += q[qbase + i] * k[kbase + i];
            s1 += q[qbase + i + 1] * k[kbase + i + 1];
            s2 += q[qbase + i + 2] * k[kbase + i + 2];
            s3 += q[qbase + i + 3] * k[kbase + i + 3];
            s4 += q[qbase + i + 4] * k[kbase + i + 4];
            s5 += q[qbase + i + 5] * k[kbase + i + 5];
            s6 += q[qbase + i + 6] * k[kbase + i + 6];
            s7 += q[qbase + i + 7] * k[kbase + i + 7];
        }
        let mut s = ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7));
        for i in chunks * 8..hd {
            s += q[qbase + i] * k[kbase + i];
        }
        s = s * scale;
        // The mask value is loaded unconditionally and applied by selection.
        let gate = mask[b * mem + m];
        w[lane] = select(gate == F::new(0.0_f32), F::min_value(), s);
    }
}

/// Lane per `(row, head)` of [`attn_weights`]: the softmax of the lane's `M`
/// scores in place — the row maximum, the sum of the shifted exponentials,
/// then `exp(s - max) / sum` — each pass loading eight slots per round.
#[cube(launch_unchecked)]
fn ms2_attn_softmax_kernel<F: Float + CubeElement>(
    w: &mut Array<F>,
    mem: usize,
    lanes: usize,
    span: usize,
) {
    let chunks = mem / 8;
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    for lane in start..end {
        let wbase = lane * mem;
        let mut mx = F::min_value();
        for c in 0..chunks {
            let i = wbase + c * 8;
            let v0 = w[i];
            let v1 = w[i + 1];
            let v2 = w[i + 2];
            let v3 = w[i + 3];
            let v4 = w[i + 4];
            let v5 = w[i + 5];
            let v6 = w[i + 6];
            let v7 = w[i + 7];
            let m01 = v0.max(v1);
            let m23 = v2.max(v3);
            let m45 = v4.max(v5);
            let m67 = v6.max(v7);
            mx = mx.max(m01.max(m23).max(m45.max(m67)));
        }
        for m in chunks * 8..mem {
            mx = mx.max(w[wbase + m]);
        }
        let mut sum = F::new(0.0_f32);
        for c in 0..chunks {
            let i = wbase + c * 8;
            let e0 = (w[i] - mx).exp();
            let e1 = (w[i + 1] - mx).exp();
            let e2 = (w[i + 2] - mx).exp();
            let e3 = (w[i + 3] - mx).exp();
            let e4 = (w[i + 4] - mx).exp();
            let e5 = (w[i + 5] - mx).exp();
            let e6 = (w[i + 6] - mx).exp();
            let e7 = (w[i + 7] - mx).exp();
            sum += ((e0 + e1) + (e2 + e3)) + ((e4 + e5) + (e6 + e7));
        }
        for m in chunks * 8..mem {
            sum += (w[wbase + m] - mx).exp();
        }
        for c in 0..chunks {
            let i = wbase + c * 8;
            let e0 = (w[i] - mx).exp() / sum;
            let e1 = (w[i + 1] - mx).exp() / sum;
            let e2 = (w[i + 2] - mx).exp() / sum;
            let e3 = (w[i + 3] - mx).exp() / sum;
            let e4 = (w[i + 4] - mx).exp() / sum;
            let e5 = (w[i + 5] - mx).exp() / sum;
            let e6 = (w[i + 6] - mx).exp() / sum;
            let e7 = (w[i + 7] - mx).exp() / sum;
            w[i] = e0;
            w[i + 1] = e1;
            w[i + 2] = e2;
            w[i + 3] = e3;
            w[i + 4] = e4;
            w[i + 5] = e5;
            w[i + 6] = e6;
            w[i + 7] = e7;
        }
        for m in chunks * 8..mem {
            w[wbase + m] = (w[wbase + m] - mx).exp() / sum;
        }
    }
}

/// [`ms2_attn_scores_kernel`] and [`ms2_attn_softmax_kernel`] in one launch,
/// with one plane (or one aligned segment of a plane) per `(row, head)`, for
/// a device with planes.
///
/// A lane takes the memory slots `slot, slot + width, ..` of its row: the
/// slot's key is a run of whole vectors, so the dot product is `hd / N` vector
/// loads against the row's query (unrolled, so they are issued together and
/// waited for once), and the row maximum and the sum of the
/// exponentials are one plane reduction each instead of a serial pass over
/// the `M` slots. A lane keeps its scores in `w` between the passes (it reads
/// back only what it wrote). The arithmetic is the two kernels': a masked
/// slot takes the most negative finite value before the max shift.
#[allow(clippy::too_many_arguments)]
#[cube(launch_unchecked)]
fn ms2_attn_weights_plane_kernel<F: Float + CubeElement, N: Size>(
    q: &Array<Vector<F, N>>,
    k: &Array<Vector<F, N>>,
    mask: &Array<F>,
    w: &mut Array<F>,
    d_lines: usize,
    heads: usize,
    mem: usize,
    rows_per_spectrum: usize,
    scale: F,
    lanes: usize,
    #[comptime] hd_lines: usize,
    #[comptime] seg_bits: u32,
) {
    let mut width = PLANE_DIM as usize;
    let mut slot = UNIT_POS_PLANE as usize;
    if comptime!(seg_bits > 0) {
        width = comptime!(1usize << seg_bits);
        slot = UNIT_POS_PLANE as usize % width;
    }
    let rh = ABSOLUTE_POS / width;
    let live = rh < lanes;
    let safe = select(live, rh, 0);
    let row = safe / heads;
    let head = safe % heads;
    let b = row / rows_per_spectrum;
    let q_at = row * d_lines + head * hd_lines;
    let w_at = safe * mem;
    let steps = mem.div_ceil(width);

    // A lane past the end of the row keeps the floor, which no maximum takes
    // and whose exponential is never added.
    let mut top = F::min_value();
    for s in 0..steps {
        let m = slot + s * width;
        let ok = m < mem;
        let sm = select(ok, m, 0);
        let k_at = (b * mem + sm) * d_lines + head * hd_lines;
        // A masked slot's key is never read: on real spectra most slots of
        // the memory are padding.
        let gate = mask[b * mem + sm];
        let mut v = F::min_value();
        if ok && gate != F::new(0.0_f32) {
            let mut acc = Vector::<F, N>::new(F::new(0.0_f32));
            #[unroll]
            for i in 0..hd_lines {
                acc += q[q_at + i] * k[k_at + i];
            }
            let mut dot = acc[0];
            #[unroll]
            for l in 1..N::value() {
                dot += acc[l];
            }
            v = dot * scale;
        }
        top = top.max(v);
        if live && ok {
            w[w_at + m] = v;
        }
    }
    let mut row_top = top;
    if comptime!(seg_bits == 0) {
        row_top = plane_max(top);
    } else {
        #[unroll]
        for j in 0..seg_bits {
            row_top = row_top.max(plane_shuffle_xor(row_top, 1u32 << j));
        }
    }

    let mut sum = F::new(0.0_f32);
    for s in 0..steps {
        let m = slot + s * width;
        if live && m < mem {
            let e = (w[w_at + m] - row_top).exp();
            w[w_at + m] = e;
            sum += e;
        }
    }
    let mut row_sum = sum;
    if comptime!(seg_bits == 0) {
        row_sum = plane_sum(sum);
    } else {
        #[unroll]
        for j in 0..seg_bits {
            row_sum += plane_shuffle_xor(row_sum, 1u32 << j);
        }
    }

    for s in 0..steps {
        let m = slot + s * width;
        if live && m < mem {
            w[w_at + m] = w[w_at + m] / row_sum;
        }
    }
}

/// Single-query cross-attention weights in two launches: for the projected
/// queries `q` (`[rows, d]`, one per trajectory) and the cached keys `k`
/// (`[B, M, d]`), the per-head `softmax(mask_logits(q_h · k_h / sqrt(hd)))`
/// over the `M` memory slots of the row's spectrum (`row / rows_per_spectrum`),
/// written to `w` (`[rows, heads, M]`). The arithmetic is the composed
/// `matmul`, `mul_scalar`, `mask_logits`, `softmax` chain's: a masked slot
/// takes the most negative finite value before the max shift. The scores are
/// one lane per `(row, head, slot)`; the softmax is one lane per
/// `(row, head)` over the stored scores. On a device with planes both run as
/// one launch, a plane per `(row, head)` ([`ms2_attn_weights_plane_kernel`]).
pub fn attn_weights<R: Runtime, E: FloatElem>(
    q: &Tensor<R, E>,
    k: &Tensor<R, E>,
    mask: &Tensor<R, E>,
    w: &mut Tensor<R, E>,
    heads: usize,
    rows_per_spectrum: usize,
) -> Result<()> {
    if q.rank() != 2 || k.rank() != 3 || mask.rank() != 2 || w.rank() != 3 {
        return Err(Error::shape(format!(
            "attn_weights needs q [rows, d], k [B, M, d], mask [B, M] and w [rows, heads, M], got {} and {} and {} and {}",
            q.shape(),
            k.shape(),
            mask.shape(),
            w.shape()
        )));
    }
    let rows = q.shape().dim(0);
    let d = q.shape().dim(1);
    let spectra = k.shape().dim(0);
    let mem = k.shape().dim(1);
    if heads == 0
        || !d.is_multiple_of(heads)
        || rows_per_spectrum == 0
        || rows != spectra * rows_per_spectrum
        || k.shape().dim(2) != d
        || mask.shape().dims() != [spectra, mem]
        || w.shape().dims() != [rows, heads, mem]
    {
        return Err(Error::shape(format!(
            "attn_weights has mismatched shapes: q {}, k {}, mask {}, w {}, heads {heads}, rows per spectrum {rows_per_spectrum}",
            q.shape(),
            k.shape(),
            mask.shape(),
            w.shape()
        )));
    }
    if shares_storage(&q.arg(), &w.arg())
        || shares_storage(&k.arg(), &w.arg())
        || shares_storage(&mask.arg(), &w.arg())
    {
        return Err(Error::config(
            "attn_weights: w shares storage with an input; the output must not alias an input"
                .to_string(),
        ));
    }
    if w.is_empty() {
        return Ok(());
    }
    let hd = d / heads;
    let line = line_size_for::<R, E>(q.client(), hd);
    if let Some((count, dim, seg_bits)) =
        plane_segments_per_row::<R>(q.client(), rows * heads, mem)
    {
        unsafe {
            ms2_attn_weights_plane_kernel::launch_unchecked::<E, R>(
                q.client(),
                count,
                dim,
                line,
                q.arg(),
                k.arg(),
                mask.arg(),
                w.arg(),
                d / line,
                heads,
                mem,
                rows_per_spectrum,
                E::from_scalar(1.0 / (hd as f32).sqrt()),
                rows * heads,
                hd / line,
                seg_bits,
            );
        }
        return Ok(());
    }
    let lanes = rows * heads * mem;
    let (count, dim, span) = launch_1d_spans(q.client(), lanes, hd);
    unsafe {
        ms2_attn_scores_kernel::launch_unchecked::<E, R>(
            q.client(),
            count,
            dim,
            q.arg(),
            k.arg(),
            mask.arg(),
            w.arg(),
            d,
            heads,
            mem,
            rows_per_spectrum,
            E::from_scalar(1.0 / (hd as f32).sqrt()),
            lanes,
            span,
        );
    }
    let lanes = rows * heads;
    let (count, dim, span) = launch_1d_spans(q.client(), lanes, 3 * mem);
    unsafe {
        ms2_attn_softmax_kernel::launch_unchecked::<E, R>(
            q.client(),
            count,
            dim,
            w.arg(),
            mem,
            lanes,
            span,
        );
    }
    Ok(())
}

/// Lane per output element of [`attn_context`].
#[allow(clippy::too_many_arguments)]
#[cube(launch_unchecked)]
fn ms2_attn_context_kernel<F: Float + CubeElement>(
    w: &Array<F>,
    v: &Array<F>,
    out: &mut Array<F>,
    d: usize,
    heads: usize,
    mem: usize,
    rows_per_spectrum: usize,
    lanes: usize,
    span: usize,
) {
    let hd = d / heads;
    let chunks = mem / 8;
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    for pos in start..end {
        let row = pos / d;
        let c = pos % d;
        let head = c / hd;
        let b = row / rows_per_spectrum;
        let wbase = (row * heads + head) * mem;
        let vbase = b * mem * d + c;
        let mut a0 = F::new(0.0_f32);
        let mut a1 = F::new(0.0_f32);
        let mut a2 = F::new(0.0_f32);
        let mut a3 = F::new(0.0_f32);
        let mut a4 = F::new(0.0_f32);
        let mut a5 = F::new(0.0_f32);
        let mut a6 = F::new(0.0_f32);
        let mut a7 = F::new(0.0_f32);
        for ch in 0..chunks {
            let m = ch * 8;
            a0 += w[wbase + m] * v[vbase + m * d];
            a1 += w[wbase + m + 1] * v[vbase + (m + 1) * d];
            a2 += w[wbase + m + 2] * v[vbase + (m + 2) * d];
            a3 += w[wbase + m + 3] * v[vbase + (m + 3) * d];
            a4 += w[wbase + m + 4] * v[vbase + (m + 4) * d];
            a5 += w[wbase + m + 5] * v[vbase + (m + 5) * d];
            a6 += w[wbase + m + 6] * v[vbase + (m + 6) * d];
            a7 += w[wbase + m + 7] * v[vbase + (m + 7) * d];
        }
        let mut acc = ((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7));
        for m in chunks * 8..mem {
            acc += w[wbase + m] * v[vbase + m * d];
        }
        out[pos] = acc;
    }
}

/// The attention context of [`attn_weights`]' weights in one launch:
/// `out[row, c] = sum_m w[row, head(c), m] * v[spectrum(row), m, c]`, the
/// composed `weights.matmul(values)` with the head permutes folded into the
/// indexing. `w` is `[rows, heads, M]`, `v` `[B, M, d]`; output `[rows, d]`.
pub fn attn_context<R: Runtime, E: FloatElem>(
    w: &Tensor<R, E>,
    v: &Tensor<R, E>,
    rows_per_spectrum: usize,
) -> Result<Tensor<R, E>> {
    if w.rank() != 3 || v.rank() != 3 {
        return Err(Error::shape(format!(
            "attn_context needs w [rows, heads, M] and v [B, M, d], got {} and {}",
            w.shape(),
            v.shape()
        )));
    }
    let rows = w.shape().dim(0);
    let heads = w.shape().dim(1);
    let mem = w.shape().dim(2);
    let spectra = v.shape().dim(0);
    let d = v.shape().dim(2);
    if heads == 0
        || !d.is_multiple_of(heads)
        || rows_per_spectrum == 0
        || rows != spectra * rows_per_spectrum
        || v.shape().dim(1) != mem
    {
        return Err(Error::shape(format!(
            "attn_context has mismatched shapes: w {}, v {}, rows per spectrum {rows_per_spectrum}",
            w.shape(),
            v.shape()
        )));
    }
    let out = Tensor::empty(Shape::new(vec![rows, d]), w.device());
    if out.is_empty() {
        return Ok(out);
    }
    let lanes = out.len();
    let (count, dim, span) = launch_1d_spans(w.client(), lanes, mem);
    unsafe {
        ms2_attn_context_kernel::launch_unchecked::<E, R>(
            w.client(),
            count,
            dim,
            w.arg(),
            v.arg(),
            out.arg(),
            d,
            heads,
            mem,
            rows_per_spectrum,
            lanes,
            span,
        );
    }
    Ok(out)
}

/// Lane per row of [`atom_key_update`].
#[allow(clippy::too_many_arguments)]
#[cube(launch_unchecked)]
fn ms2_atom_key_kernel<F: Float + CubeElement>(
    token: &Array<u32>,
    grammar_state: &Array<u32>,
    src: &Array<F>,
    atom_keys: &mut Array<F>,
    resid_ids: &mut Array<u32>,
    atoms_n: usize,
    d: usize,
    state_width: usize,
    src_width: usize,
    src_offset: usize,
    lanes: usize,
    span: usize,
) {
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    for r in start..end {
        if token[r * 4] == 2u32 {
            let n = grammar_state[(r * state_width) + 3 * atoms_n];
            if n != 0u32 && n <= atoms_n as u32 {
                let slot = (n - 1u32) as usize;
                for j in 0..d {
                    atom_keys[(r * atoms_n + slot) * d + j] = src[r * src_width + src_offset + j];
                }
            }
        }
        for j in 0..atoms_n {
            let v = grammar_state[r * state_width + atoms_n + j];
            let mut c = v;
            if c > 7u32 {
                c = 7u32;
            }
            resid_ids[r * atoms_n + j] = c;
        }
    }
}

/// [`atom_memory_update`] for the fused step, in one launch: when the row's
/// `token` is ADD_ATOM, the `d` values `src[row, src_offset..src_offset + d]`
/// (the key segment of the previous step's head row: the pointer-key
/// projection of the previous decoder output and whatever the caller stores
/// with it, `d` being the last axis of `atom_keys`) are copied into
/// `atom_keys[row, count - 1, :]`,
/// and the clamped residual ids `min(residual, 7)` are refreshed. The stored
/// row is the projection of what [`atom_memory_update`] stores, so the pointer
/// head needs no per-step projection of the whole memory.
pub fn atom_key_update<R: Runtime, E: FloatElem>(
    token: &IdTensor<R>,
    grammar_state: &IdTensor<R>,
    src: &Tensor<R, E>,
    src_offset: usize,
    atom_keys: &mut Tensor<R, E>,
    resid_ids: &mut IdTensor<R>,
    max_atoms: usize,
) -> Result<()> {
    if token.shape().rank() != 2
        || grammar_state.shape().rank() != 2
        || src.rank() != 2
        || atom_keys.rank() != 3
    {
        return Err(Error::shape(format!(
            "atom_key_update needs token [rows, 4], grammar_state [rows, 3A + 16], src [rows, W] and atom_keys [rows, A, d], got {} and {} and {} and {}",
            token.shape(),
            grammar_state.shape(),
            src.shape(),
            atom_keys.shape()
        )));
    }
    let rows = token.shape().dim(0);
    let a = max_atoms;
    let d = atom_keys.shape().dim(2);
    let src_width = src.shape().dim(1);
    if token.shape().dim(1) != 4
        || grammar_state.shape().dims() != [rows, replay_state_width(a)]
        || src.shape().dim(0) != rows
        || src_offset + d > src_width
        || atom_keys.shape().dims() != [rows, a, d]
        || resid_ids.len() != rows * a
    {
        return Err(Error::shape(format!(
            "atom_key_update has mismatched batch shapes: token {}, grammar_state {}, src {} at offset {src_offset}, atom_keys {}, resid {rows}x{a}",
            token.shape(),
            grammar_state.shape(),
            src.shape(),
            atom_keys.shape()
        )));
    }
    if shares_storage(&src.arg(), &atom_keys.arg())
        || shares_storage(&grammar_state.arg(), &resid_ids.arg())
        || shares_storage(&token.arg(), &resid_ids.arg())
    {
        return Err(Error::config(
            "atom_key_update: an output shares storage with an input; the output must not alias an input"
                .to_string(),
        ));
    }
    if rows == 0 {
        return Ok(());
    }
    let client = token.client();
    let (count, dim, span) = launch_1d_spans(client, rows, d + a);
    unsafe {
        ms2_atom_key_kernel::launch_unchecked::<E, R>(
            client,
            count,
            dim,
            token.arg(),
            grammar_state.arg(),
            src.arg(),
            atom_keys.arg(),
            resid_ids.arg(),
            a,
            d,
            replay_state_width(a),
            src_width,
            src_offset,
            rows,
            span,
        );
    }
    Ok(())
}

/// Where the columns of the fused step's head row sit: one product of the
/// decoder output fills all of them ([`step_head_layout`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StepHeadLayout {
    /// First of the `d` pointer-query columns; the 27 logits precede it.
    pub query: usize,
    /// First of the 8 columns `query · E_residual[r]`.
    pub query_resid: usize,
    /// First of the `key_width` columns an added atom stores.
    pub key: usize,
    /// `d + 23`: the atom-memory key, then `key · E_cond[c]` for the 19
    /// atom-type and the 4 bond conditioning rows.
    pub key_width: usize,
    /// Columns in use.
    pub used: usize,
    /// `used` rounded up to a multiple of 4.
    pub width: usize,
}

/// The head-row layout for model width `d`: `logits (27) | query (d) |
/// query · E_residual (8) | key (d) | key · E_cond (23) | padding`.
pub fn step_head_layout(d: usize) -> StepHeadLayout {
    let cond = SAMPLE_COND_ROWS + SAMPLE_PBOND_ROWS;
    let query = STEP_EMBED_FIXED_ROWS;
    let query_resid = query + d;
    let key = query_resid + STEP_PTR_RESID_ROWS;
    let used = key + d + cond;
    StepHeadLayout {
        query,
        query_resid,
        key,
        key_width: d + cond,
        used,
        width: used.next_multiple_of(4),
    }
}

/// Lane per packed logit of [`step_logits_pack`].
#[allow(clippy::too_many_arguments)]
#[cube(launch_unchecked)]
fn ms2_step_logits_kernel<F: Float + CubeElement>(
    heads: &Array<F>,
    atom_keys: &Array<F>,
    resid_ids: &Array<u32>,
    cross: &Array<F>,
    bias: &Array<F>,
    logits: &mut Array<F>,
    atoms_n: usize,
    d: usize,
    heads_width: usize,
    key_width: usize,
    query_at: usize,
    query_resid_at: usize,
    scale: F,
    lanes: usize,
    span: usize,
) {
    let width = 27 + 24 * atoms_n;
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    for pos in start..end {
        let r = pos / width;
        let col = pos % width;
        if col < 27 {
            logits[pos] = heads[r * heads_width + col] + bias[col];
        } else {
            let rel = col - 27;
            let seg = rel / atoms_n;
            let j = rel % atoms_n;
            let kbase = (r * atoms_n + j) * key_width;
            let mut rid = resid_ids[r * atoms_n + j] as usize;
            if rid > 7 {
                rid = 7;
            }
            if seg == 0 {
                // The row's own query: its product with the key is the one
                // dot product of the step that cannot be known earlier.
                let qbase = r * heads_width + query_at;
                let chunks = d / 8;
                let mut a0 = F::new(0.0_f32);
                let mut a1 = F::new(0.0_f32);
                let mut a2 = F::new(0.0_f32);
                let mut a3 = F::new(0.0_f32);
                let mut a4 = F::new(0.0_f32);
                let mut a5 = F::new(0.0_f32);
                let mut a6 = F::new(0.0_f32);
                let mut a7 = F::new(0.0_f32);
                for ch in 0..chunks {
                    let i = ch * 8;
                    a0 += heads[qbase + i] * atom_keys[kbase + i];
                    a1 += heads[qbase + i + 1] * atom_keys[kbase + i + 1];
                    a2 += heads[qbase + i + 2] * atom_keys[kbase + i + 2];
                    a3 += heads[qbase + i + 3] * atom_keys[kbase + i + 3];
                    a4 += heads[qbase + i + 4] * atom_keys[kbase + i + 4];
                    a5 += heads[qbase + i + 5] * atom_keys[kbase + i + 5];
                    a6 += heads[qbase + i + 6] * atom_keys[kbase + i + 6];
                    a7 += heads[qbase + i + 7] * atom_keys[kbase + i + 7];
                }
                let mut acc = ((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7));
                for i in chunks * 8..d {
                    acc += heads[qbase + i] * atom_keys[kbase + i];
                }
                acc += heads[r * heads_width + query_resid_at + rid];
                logits[pos] = acc * scale;
            } else {
                // A conditioning row as the query: both halves were computed
                // before this step.
                let stored = atom_keys[kbase + d + seg - 1];
                logits[pos] = (stored + cross[(seg - 1) * 8 + rid]) * scale;
            }
        }
    }
}

/// The packed sampler-logits row of one step in one launch (layout of
/// [`sample_logits_offsets`]): the 27 kind, atom-type and bond-base logits
/// are `heads[row, 0..27] + bias`, and the three pointer blocks are
/// `query · (key[row, j] + E_residual[resid[row, j]]) / sqrt(d)` with the
/// query the row's own (pointer base), `E_ptr_type[c]` (19 rows) and
/// `E_ptr_bond[b]` (4 rows).
///
/// Only the row's own query costs a dot product here. Its residual half is a
/// column of the head row; a conditioning row's product with a key was stored
/// with the key, and its product with a residual row is `cross`.
///
/// `heads` is `[rows, W]` in the layout of [`step_head_layout`], `atom_keys`
/// `[rows, A, d + 23]` (the key, then its 23 conditioning products),
/// `resid_ids` `[rows * A]` clamped to `0..=7`, `cross` `[23, 8]` (conditioning
/// row against residual row), `bias` the 27 head biases (the head row is the
/// bare product, so its one biased segment gets its bias here rather than in
/// a launch of its own), and `logits` `[rows, 27 + 24 * A]`, overwritten in
/// place.
pub fn step_logits_pack<R: Runtime, E: FloatElem>(
    heads: &Tensor<R, E>,
    atom_keys: &Tensor<R, E>,
    resid_ids: &IdTensor<R>,
    cross: &Tensor<R, E>,
    bias: &Tensor<R, E>,
    logits: &mut Tensor<R, E>,
    max_atoms: usize,
) -> Result<()> {
    if heads.rank() != 2 || atom_keys.rank() != 3 || cross.rank() != 2 || logits.rank() != 2 {
        return Err(Error::shape(format!(
            "step_logits_pack needs heads [rows, W], atom_keys [rows, A, d + 23], cross [23, 8] and logits [rows, 27 + 24 A], got {} and {} and {} and {}",
            heads.shape(),
            atom_keys.shape(),
            cross.shape(),
            logits.shape()
        )));
    }
    let rows = heads.shape().dim(0);
    let a = max_atoms;
    let cond = SAMPLE_COND_ROWS + SAMPLE_PBOND_ROWS;
    let key_width = atom_keys.shape().dim(2);
    let heads_width = heads.shape().dim(1);
    if key_width <= cond {
        return Err(Error::shape(format!(
            "step_logits_pack needs atom_keys [rows, A, d + {cond}], got {}",
            atom_keys.shape()
        )));
    }
    let d = key_width - cond;
    let layout = step_head_layout(d);
    if heads_width < layout.used
        || atom_keys.shape().dims() != [rows, a, key_width]
        || resid_ids.len() != rows * a
        || cross.shape().dims() != [cond, STEP_PTR_RESID_ROWS]
        || bias.len() < STEP_EMBED_FIXED_ROWS
        || logits.shape().dims() != [rows, sample_logits_width(a)]
    {
        return Err(Error::shape(format!(
            "step_logits_pack has mismatched shapes: heads {}, atom_keys {}, resid {}, cross {}, bias {}, logits {} for A = {a}",
            heads.shape(),
            atom_keys.shape(),
            resid_ids.shape(),
            cross.shape(),
            bias.shape(),
            logits.shape()
        )));
    }
    if shares_storage(&heads.arg(), &logits.arg())
        || shares_storage(&atom_keys.arg(), &logits.arg())
        || shares_storage(&cross.arg(), &logits.arg())
        || shares_storage(&bias.arg(), &logits.arg())
    {
        return Err(Error::config(
            "step_logits_pack: logits shares storage with an input; the output must not alias an input"
                .to_string(),
        ));
    }
    if logits.is_empty() {
        return Ok(());
    }
    let lanes = logits.len();
    let (count, dim, span) = launch_1d_spans(heads.client(), lanes, 8);
    unsafe {
        ms2_step_logits_kernel::launch_unchecked::<E, R>(
            heads.client(),
            count,
            dim,
            heads.arg(),
            atom_keys.arg(),
            resid_ids.arg(),
            cross.arg(),
            bias.arg(),
            logits.arg(),
            a,
            d,
            heads_width,
            key_width,
            layout.query,
            layout.query_resid,
            E::from_scalar(1.0 / (d as f32).sqrt()),
            lanes,
            span,
        );
    }
    Ok(())
}

/// Lane per element of [`freeze_rows`].
#[allow(clippy::too_many_arguments)]
#[cube(launch_unchecked)]
fn ms2_freeze_rows_kernel<F: Float + CubeElement>(
    new_t: &mut Array<F>,
    old_t: &Array<F>,
    state: &Array<u32>,
    width: usize,
    state_width: usize,
    stop_col: usize,
    lanes: usize,
    span: usize,
) {
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    for pos in start..end {
        let r = pos / width;
        // Guarded on purpose: a live row costs one load of its flag and no
        // traffic on the carries, which are the largest buffers of the step
        // (`h` alone is `heads * head_dim * d_state` values per row). A
        // branch-free select would read and rewrite every row every step.
        if state[r * state_width + stop_col] != 0u32 {
            new_t[pos] = old_t[pos];
        }
    }
}

/// Lane per 8 consecutive elements of [`freeze_rows`], for a row width that
/// is a multiple of 8: one flag load decides the lane, and a stopped row's
/// eight values are loaded in one block before they are stored.
#[allow(clippy::too_many_arguments)]
#[cube(launch_unchecked)]
fn ms2_freeze_rows8_kernel<F: Float + CubeElement>(
    new_t: &mut Array<F>,
    old_t: &Array<F>,
    state: &Array<u32>,
    chunks_per_row: usize,
    state_width: usize,
    stop_col: usize,
    lanes: usize,
    span: usize,
) {
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    for lane in start..end {
        let r = lane / chunks_per_row;
        if state[r * state_width + stop_col] != 0u32 {
            let i = lane * 8;
            let v0 = old_t[i];
            let v1 = old_t[i + 1];
            let v2 = old_t[i + 2];
            let v3 = old_t[i + 3];
            let v4 = old_t[i + 4];
            let v5 = old_t[i + 5];
            let v6 = old_t[i + 6];
            let v7 = old_t[i + 7];
            new_t[i] = v0;
            new_t[i + 1] = v1;
            new_t[i + 2] = v2;
            new_t[i + 3] = v3;
            new_t[i + 4] = v4;
            new_t[i + 5] = v5;
            new_t[i + 6] = v6;
            new_t[i + 7] = v7;
        }
    }
}

/// [`ms2_freeze_rows8_kernel`] over two carries in one launch. The two may
/// differ in row width (`h` beside an angle): a lane is chunk `lane` of
/// either, and takes part in each carry only while that carry has such a
/// chunk.
#[allow(clippy::too_many_arguments)]
#[cube(launch_unchecked)]
fn ms2_freeze_pair8_kernel<F: Float + CubeElement>(
    new_a: &mut Array<F>,
    old_a: &Array<F>,
    new_b: &mut Array<F>,
    old_b: &Array<F>,
    state: &Array<u32>,
    chunks_per_row_a: usize,
    chunks_per_row_b: usize,
    lanes_a: usize,
    lanes_b: usize,
    state_width: usize,
    stop_col: usize,
    lanes: usize,
    span: usize,
) {
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    for lane in start..end {
        let i = lane * 8;
        if lane < lanes_a {
            let r = lane / chunks_per_row_a;
            if state[r * state_width + stop_col] != 0u32 {
                let a0 = old_a[i];
                let a1 = old_a[i + 1];
                let a2 = old_a[i + 2];
                let a3 = old_a[i + 3];
                let a4 = old_a[i + 4];
                let a5 = old_a[i + 5];
                let a6 = old_a[i + 6];
                let a7 = old_a[i + 7];
                new_a[i] = a0;
                new_a[i + 1] = a1;
                new_a[i + 2] = a2;
                new_a[i + 3] = a3;
                new_a[i + 4] = a4;
                new_a[i + 5] = a5;
                new_a[i + 6] = a6;
                new_a[i + 7] = a7;
            }
        }
        if lane < lanes_b {
            let r = lane / chunks_per_row_b;
            if state[r * state_width + stop_col] != 0u32 {
                let b0 = old_b[i];
                let b1 = old_b[i + 1];
                let b2 = old_b[i + 2];
                let b3 = old_b[i + 3];
                let b4 = old_b[i + 4];
                let b5 = old_b[i + 5];
                let b6 = old_b[i + 6];
                let b7 = old_b[i + 7];
                new_b[i] = b0;
                new_b[i + 1] = b1;
                new_b[i + 2] = b2;
                new_b[i + 3] = b3;
                new_b[i + 4] = b4;
                new_b[i + 5] = b5;
                new_b[i + 6] = b6;
                new_b[i + 7] = b7;
            }
        }
    }
}

/// Shape checks shared by [`freeze_rows`] and [`freeze_rows_pair`]: `new_t`
/// and `old_t` are `[rows, ..]` of one shape and `grammar_state` is
/// `[rows, 3A + 16]`. Returns the rows.
fn freeze_check<R: Runtime, E: FloatElem>(
    new_t: &Tensor<R, E>,
    old_t: &Tensor<R, E>,
    grammar_state: &IdTensor<R>,
    max_atoms: usize,
) -> Result<usize> {
    if new_t.shape().dims() != old_t.shape().dims()
        || new_t.rank() == 0
        || grammar_state.shape().rank() != 2
    {
        return Err(Error::shape(format!(
            "freeze_rows needs new and old of one shape [rows, ..] and grammar_state [rows, 3A + 16], got {} and {} and {}",
            new_t.shape(),
            old_t.shape(),
            grammar_state.shape()
        )));
    }
    let rows = new_t.shape().dim(0);
    if grammar_state.shape().dims() != [rows, replay_state_width(max_atoms)] {
        return Err(Error::shape(format!(
            "freeze_rows needs grammar_state [{rows}, {}], got {}",
            replay_state_width(max_atoms),
            grammar_state.shape()
        )));
    }
    Ok(rows)
}

/// The sampler's carry freeze in one launch and in place: every row of
/// `new_t` whose trajectory has stopped (grammar state column `3A + 5`
/// non-zero: stopped or failed) is overwritten with the same row of `old_t`;
/// a live row is left as it is, at the cost of one flag load per lane and no
/// traffic on the carry. The row is selected by comparison, never by
/// multiplying with a mask. `new_t` and `old_t` are `[rows, ..]` of one
/// shape, `grammar_state` `[rows, 3A + 16]`. A row width that is a multiple
/// of 8 runs one lane per 8 elements. Two handles on one buffer are already
/// frozen: nothing is launched.
pub fn freeze_rows<R: Runtime, E: FloatElem>(
    new_t: &mut Tensor<R, E>,
    old_t: &Tensor<R, E>,
    grammar_state: &IdTensor<R>,
    max_atoms: usize,
) -> Result<()> {
    let rows = freeze_check(new_t, old_t, grammar_state, max_atoms)?;
    if new_t.is_empty() || shares_storage(&new_t.arg(), &old_t.arg()) {
        return Ok(());
    }
    let elems = new_t.len();
    let width = elems / rows;
    let state_width = replay_state_width(max_atoms);
    let stop_col = 3 * max_atoms + 5;
    if width.is_multiple_of(8) {
        let lanes = elems / 8;
        let (count, dim, span) = launch_1d_spans(new_t.client(), lanes, 8);
        unsafe {
            ms2_freeze_rows8_kernel::launch_unchecked::<E, R>(
                new_t.client(),
                count,
                dim,
                new_t.arg(),
                old_t.arg(),
                grammar_state.arg(),
                width / 8,
                state_width,
                stop_col,
                lanes,
                span,
            );
        }
        return Ok(());
    }
    let (count, dim, span) = launch_1d_spans(new_t.client(), elems, 1);
    unsafe {
        ms2_freeze_rows_kernel::launch_unchecked::<E, R>(
            new_t.client(),
            count,
            dim,
            new_t.arg(),
            old_t.arg(),
            grammar_state.arg(),
            width,
            state_width,
            stop_col,
            elems,
            span,
        );
    }
    Ok(())
}

/// [`freeze_rows`] for two carries in one launch when each row width is a
/// multiple of 8 and the four buffers are distinct; any other case freezes
/// them one after the other. The two carries need not have one shape.
pub fn freeze_rows_pair<R: Runtime, E: FloatElem>(
    new_a: &mut Tensor<R, E>,
    old_a: &Tensor<R, E>,
    new_b: &mut Tensor<R, E>,
    old_b: &Tensor<R, E>,
    grammar_state: &IdTensor<R>,
    max_atoms: usize,
) -> Result<()> {
    let rows = freeze_check(new_a, old_a, grammar_state, max_atoms)?;
    freeze_check(new_b, old_b, grammar_state, max_atoms)?;
    let (elems_a, elems_b) = (new_a.len(), new_b.len());
    let distinct = !shares_storage(&new_a.arg(), &old_a.arg())
        && !shares_storage(&new_b.arg(), &old_b.arg())
        && !shares_storage(&new_a.arg(), &new_b.arg())
        && !shares_storage(&new_a.arg(), &old_b.arg())
        && !shares_storage(&new_b.arg(), &old_a.arg());
    if elems_a == 0
        || elems_b == 0
        || !(elems_a / rows).is_multiple_of(8)
        || !(elems_b / rows).is_multiple_of(8)
        || !distinct
    {
        freeze_rows(new_a, old_a, grammar_state, max_atoms)?;
        return freeze_rows(new_b, old_b, grammar_state, max_atoms);
    }
    let (lanes_a, lanes_b) = (elems_a / 8, elems_b / 8);
    let lanes = lanes_a.max(lanes_b);
    let (count, dim, span) = launch_1d_spans(new_a.client(), lanes, 16);
    unsafe {
        ms2_freeze_pair8_kernel::launch_unchecked::<E, R>(
            new_a.client(),
            count,
            dim,
            new_a.arg(),
            old_a.arg(),
            new_b.arg(),
            old_b.arg(),
            grammar_state.arg(),
            elems_a / rows / 8,
            elems_b / rows / 8,
            lanes_a,
            lanes_b,
            replay_state_width(max_atoms),
            3 * max_atoms + 5,
            lanes,
            span,
        );
    }
    Ok(())
}

/// [`freeze_rows`] over any number of carries, two to a launch
/// ([`freeze_rows_pair`]): the carries of every layer of a decoding step in
/// half as many launches as there are tensors. Each entry is `(new, old)`.
pub fn freeze_rows_all<R: Runtime, E: FloatElem>(
    carries: &mut [(Tensor<R, E>, &Tensor<R, E>)],
    grammar_state: &IdTensor<R>,
    max_atoms: usize,
) -> Result<()> {
    for pair in carries.chunks_mut(2) {
        match pair {
            [(new_a, old_a), (new_b, old_b)] => {
                freeze_rows_pair(new_a, old_a, new_b, old_b, grammar_state, max_atoms)?
            }
            [(new_t, old_t)] => freeze_rows(new_t, old_t, grammar_state, max_atoms)?,
            _ => {}
        }
    }
    Ok(())
}
