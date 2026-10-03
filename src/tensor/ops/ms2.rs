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

use crate::backend::{Device, FloatElem, launch_1d_spans};
use crate::error::{Error, Result};
use crate::models::ms2::contract::request_status;
use crate::tensor::base::Tensor;
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
fn shares_storage<R: Runtime>(a: &ArrayArg<R>, b: &ArrayArg<R>) -> bool {
    match (a, b) {
        (ArrayArg::Handle { handle: ha }, ArrayArg::Handle { handle: hb }) => {
            format!("{:?}", ha.handle.memory) == format!("{:?}", hb.handle.memory)
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

/// Lane per output element: `table[id, col]`, or `0` when `id` is out of
/// range (which includes `u32::MAX`), so padding ids read no table row.
#[cube(launch_unchecked)]
fn ms2_lookup_kernel<F: Float + CubeElement>(
    table: &Array<F>,
    ids: &Array<u32>,
    out: &mut Array<F>,
    table_rows: usize,
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
        let row = pos / d;
        let col = pos % d;
        let id = ids[row];
        if (id as usize) < table_rows {
            out[pos] = table[(id as usize) * d + col];
        } else {
            out[pos] = zero;
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
    let lanes = out.len();
    let (count, dim, span) = launch_1d_spans(table.client(), lanes, 1);
    unsafe {
        ms2_lookup_kernel::launch_unchecked::<E, R>(
            table.client(),
            count,
            dim,
            table.arg(),
            ids.arg(),
            out.arg(),
            table.shape().dim(0),
            d,
            lanes,
            span,
        );
    }
    Ok(out)
}

/// Lane per table element `(v, col)`: the sum of `grad[r, col]` over the rows
/// `r` with `ids[r] == v`, in increasing row order. No atomics, no host
/// read of the ids.
#[cube(launch_unchecked)]
fn ms2_lookup_backward_kernel<F: Float + CubeElement>(
    grad: &Array<F>,
    ids: &Array<u32>,
    out: &mut Array<F>,
    rows: usize,
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
        let v = pos / d;
        let col = pos % d;
        let mut acc = zero;
        for r in 0..rows {
            if (ids[r] as usize) == v {
                acc += grad[r * d + col];
            }
        }
        out[pos] = acc;
    }
}

/// Adjoint of [`lookup`] on the device: accumulate `grad` (`[rows, d]`)
/// back into a `[table_rows, d]` table along `ids`. One launch.
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
    let out = Tensor::empty(Shape::new(vec![table_rows, d]), grad.device());
    if out.is_empty() {
        return Ok(out);
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
            ids.len(),
            d,
            lanes,
            span,
        );
    }
    Ok(out)
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
}

impl<R: Runtime, E: FloatElem> FormulaBuffers<R, E> {
    /// Allocate the five outputs of [`formula_window`] and [`formula_top`]
    /// uninitialised: every kernel writes every element, so there is nothing
    /// to initialise.
    pub fn new(batch: usize, m: usize, f: usize, device: &Device<R>) -> Self {
        Self {
            window: IdTensor::empty(vec![batch, m, 2], device),
            counters: IdTensor::empty(vec![batch, 5], device),
            top: IdTensor::empty(vec![batch, f, 2], device),
            top_log_prob: Tensor::empty(vec![batch, f], device),
            top_count: IdTensor::empty(vec![batch], device),
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

/// Lane per spectrum: the `F` largest `log_prob` entries among the window
/// slots whose flag is non-zero, ties by smaller slot. Writes `top`
/// (table row, window slot), `top_log_prob` and `top_count`; unused entries
/// get row and slot `u32::MAX` and log-probability 0. Floats are compared
/// with `>`/`==` only. Arrays: `log_prob`, `window`, `top`, `top_log_prob`,
/// `top_count`.
#[allow(clippy::too_many_arguments)]
#[cube(launch_unchecked)]
fn ms2_formula_top_kernel<F: Float + CubeElement>(
    log_prob: &Array<F>,
    window: &Array<u32>,
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
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    for pos in start..end {
        let b = pos;
        let mut joined = 0u32;
        for mm in 0..m {
            if window[(b * m + mm) * 2 + 1] != 0u32 {
                joined += 1u32;
            }
        }
        let mut count = joined;
        if count > f as u32 {
            count = f as u32;
        }
        // `top_count` is the number of entries actually written: ranks are
        // unique for finite scores, but a NaN (which compares false both
        // ways) can share or skip a rank, leaving a pick with no slot.
        // Written entries compact densely so every slot below `top_count`
        // is a real row and `k mod top_count` never selects padding.
        let mut written = 0u32;
        for ff in 0..f {
            let ffu = ff as u32;
            if ffu < count {
                // The slot of rank `ffu`: flagged slots ordered by decreasing
                // log-probability, ties by smaller slot. The ranks are a
                // permutation of `0..joined` for finite log-probabilities, so
                // exactly one slot matches; a NaN (which compares false both
                // ways) or any other tie anomaly can share or skip a rank, in
                // which case no slot matches and nothing is written for this
                // pick rather than indexing out of bounds.
                let mut best = max_u32;
                let mut found = false;
                for mm in 0..m {
                    if window[(b * m + mm) * 2 + 1] != 0u32 {
                        let mut rank = 0u32;
                        for nn in 0..m {
                            if window[(b * m + nn) * 2 + 1] != 0u32 {
                                let ln = log_prob[b * m + nn];
                                let lm = log_prob[b * m + mm];
                                let mut better = false;
                                if ln > lm {
                                    better = true;
                                }
                                if ln == lm {
                                    if nn < mm {
                                        better = true;
                                    }
                                }
                                if better {
                                    rank += 1u32;
                                }
                            }
                        }
                        if rank == ffu {
                            best = mm as u32;
                            found = true;
                            break;
                        }
                    }
                }
                if found {
                    let dest = (b * f + written as usize) * 2;
                    top[dest] = window[(b * m + best as usize) * 2];
                    top[dest + 1] = best;
                    top_log_prob[b * f + written as usize] = log_prob[b * m + best as usize];
                    written += 1u32;
                }
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

/// Run top-F selection (architecture §3.4) from `log_prob` (`[B, M]`) and the
/// window of [`formula_window`] into `out.top`, `out.top_log_prob` and
/// `out.top_count`. `top_count` is the number of entries actually written:
/// ranks are unique for finite scores, but a non-finite score can share or
/// skip a rank, in which case the written entries compact densely so every
/// slot below `top_count` is a real row. Exactly 1 launch, one lane per
/// spectrum.
pub fn formula_top<R: Runtime, E: FloatElem>(
    log_prob: &Tensor<R, E>,
    window: &IdTensor<R>,
    out: &FormulaBuffers<R, E>,
) -> Result<()> {
    // Every rank is checked before any dimension is read, so a malformed
    // shape is `Error::Shape` rather than a panic.
    if log_prob.shape().rank() != 2
        || window.shape().rank() != 3
        || out.top.shape().rank() != 3
        || out.top_log_prob.shape().rank() != 2
        || out.top_count.shape().rank() != 1
    {
        return Err(Error::shape(format!(
            "formula_top needs log_prob [B, M], window [B, M, 2], top [B, F, 2], top_log_prob [B, F] and top_count [B], got {} and {} and {} and {} and {}",
            log_prob.shape(),
            window.shape(),
            out.top.shape(),
            out.top_log_prob.shape(),
            out.top_count.shape()
        )));
    }
    let batch = log_prob.shape().dim(0);
    let m = log_prob.shape().dim(1);
    let f = out.top.shape().dim(1);
    let want_window: &[usize] = &[batch, m, 2];
    let want_top: &[usize] = &[batch, f, 2];
    let want_top_lp: &[usize] = &[batch, f];
    let want_top_count: &[usize] = &[batch];
    if window.shape().dims() != want_window
        || out.top.shape().dims() != want_top
        || out.top_log_prob.shape().dims() != want_top_lp
        || out.top_count.shape().dims() != want_top_count
    {
        return Err(Error::shape(format!(
            "formula_top needs window [{batch}, {m}, 2], top [{batch}, {f}, 2], top_log_prob [{batch}, {f}] and top_count [{batch}], got {} and {} and {} and {}",
            window.shape(),
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
            window.arg(),
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
/// slot of formula `k mod count[b]` from `top` (`[B, F, 2]`); zeros for a
/// spectrum with `count == 0`. Arrays: `embedding` (`[B, M, d]`), `top`,
/// `top_count`, `out` (`[rows, d]`). Exactly 1 launch per generation call: the
/// formula does not change mid-trace.
#[allow(clippy::too_many_arguments)]
pub fn trajectory_formula<R: Runtime, E: FloatElem>(
    embedding: &Tensor<R, E>,
    top: &IdTensor<R>,
    top_count: &IdTensor<R>,
    out: &mut Tensor<R, E>,
    spectra: usize,
    window_m: usize,
    formulas: usize,
    per_spectrum: usize,
) -> Result<()> {
    let rows = out.shape().dim(0);
    let d = out.shape().dim(1);
    if embedding.shape().dims() != [spectra, window_m, d]
        || top.shape().dims() != [spectra, formulas, 2]
        || top_count.len() != spectra
        || out.shape().dims() != [rows, d]
        || rows != spectra * per_spectrum
    {
        return Err(Error::shape(format!(
            "trajectory_formula needs embedding [B, M, d], top [B, F, 2], top_count [B] and out [B*K, d], got {} and {} and {} and {}",
            embedding.shape(),
            top.shape(),
            top_count.shape(),
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
            top.arg(),
            top_count.arg(),
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
    top: &Array<u32>,
    top_count: &Array<u32>,
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
        let count = top_count[b];
        let mut v = zero;
        if count != 0u32 {
            let slot = top[(b * formulas + (k % count) as usize) * 2 + 1];
            if slot != sentinel {
                v = embedding[(b * window_m + slot as usize) * d + j];
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
/// of the conditioning formula (`top[b, k mod count]`, architecture §3.4).
/// A row whose spectrum failed never starts: `peak_count == 0` (the upload
/// writes 0 for every fatal host status, and an empty spectrum is fatal) or
/// `count == 0` (no scored formula, which is exactly the device-side
/// `formula_absent`), unless `metadata_only` bypasses the empty-spectrum
/// abstention. Failed rows keep `request_failed` with `length = 0` and
/// `formula_row = u32::MAX`.
///
/// Bindings (6): `top` (`[B, F, 2]`), `spectra_meta` (`[B, 8]`, carrying the
/// peak count and the id halves), `table_counts` (u32, the resident `[R, 10]`
/// exact element counts, so exact chemistry budgets never depend on the
/// neural float path), `traj_meta`, `state` and `actions` (all in/out).
/// Exactly 1 launch.
#[allow(clippy::too_many_arguments)]
pub fn init_trajectories<R: Runtime>(
    top: &IdTensor<R>,
    spectra_meta: &IdTensor<R>,
    table_counts: &IdTensor<R>,
    traj_meta: &mut IdTensor<R>,
    state: &mut IdTensor<R>,
    actions: &mut IdTensor<R>,
    spectra: usize,
    formulas: usize,
    per_spectrum: usize,
    table_rows: usize,
    steps: usize,
    atoms: usize,
    metadata_only: bool,
) -> Result<()> {
    let rows = spectra * per_spectrum;
    if top.shape().dims() != [spectra, formulas, 2]
        || spectra_meta.shape().dims() != [spectra, META_WIDTH]
        || table_counts.shape().dims() != [table_rows, 10]
        || traj_meta.shape().dims() != [rows, TRAJ_META_WIDTH]
        || state.shape().dims() != [rows, replay_state_width(atoms)]
        || actions.shape().dims() != [rows, sample_record_width(steps, atoms)]
    {
        return Err(Error::shape(format!(
            "init_trajectories needs top [B, F, 2], spectra_meta [B, 8], table_counts [R, 10], traj_meta [B*K, 14], state [B*K, {}] and actions [B*K, {}], got {} and {} and {} and {} and {} and {}",
            replay_state_width(atoms),
            sample_record_width(steps, atoms),
            top.shape(),
            spectra_meta.shape(),
            table_counts.shape(),
            traj_meta.shape(),
            state.shape(),
            actions.shape()
        )));
    }
    if rows == 0 {
        return Ok(());
    }
    let client = top.client();
    let (count, dim, span) = launch_1d_spans(client, rows, TRAJ_META_WIDTH);
    unsafe {
        ms2_init_trajectories_kernel::launch_unchecked::<R>(
            client,
            count,
            dim,
            top.arg(),
            spectra_meta.arg(),
            table_counts.arg(),
            traj_meta.arg(),
            state.arg(),
            actions.arg(),
            formulas,
            per_spectrum,
            table_rows,
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
/// semantics. Arrays: `top`, `spectra_meta`, `table_counts`, `traj_meta`,
/// `state`, `actions` (the last three in/out).
#[allow(clippy::too_many_arguments)]
#[allow(clippy::assign_op_pattern)]
#[cube(launch_unchecked)]
fn ms2_init_trajectories_kernel(
    top: &Array<u32>,
    spectra_meta: &Array<u32>,
    table_counts: &Array<u32>,
    traj_meta: &mut Array<u32>,
    state: &mut Array<u32>,
    actions: &mut Array<u32>,
    formulas: usize,
    per_spectrum: usize,
    table_rows: usize,
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
        // The scored-formula count is the packed prefix of valid top rows.
        let mut count = 0u32;
        for ff in 0..formulas {
            if top[(b * formulas + ff) * 2] != sentinel {
                count += 1u32;
            }
        }
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
        // A row starts when the spectrum has peaks and a scored formula
        // (`metadata_only` bypasses the empty-spectrum abstention).
        let mut go = false;
        if count != 0u32 {
            if peak_count != 0u32 {
                go = true;
            }
            if metadata_only != 0u32 {
                go = true;
            }
        }
        if go {
            let fslot = ((k % count) as usize) % formulas;
            let frow = top[(b * formulas + fslot) * 2];
            if frow < table_rows as u32 {
                state[sbase + 3 * atoms_n + 4] = 1u32;
                actions[abase] = 1u32;
                actions[len_off] = 1u32;
                actions[len_off + 3] = frow;
                traj_meta[tbase] = spectra_meta[b * 8 + 6];
                traj_meta[tbase + 1] = spectra_meta[b * 8 + 7];
                traj_meta[tbase + 2] = k;
                traj_meta[tbase + 3] = 1u32;
                for e in 0..10usize {
                    traj_meta[tbase + 4 + e] = table_counts[frow as usize * 10 + e];
                }
            } else {
                actions[len_off + 1] = 64u32;
                actions[len_off + 3] = sentinel;
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
