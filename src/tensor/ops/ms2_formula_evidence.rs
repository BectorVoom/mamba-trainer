//! GPU kernels for formula evidence (task E1).
//!
//! Architecture `docs/MS2_V1_ARCHITECTURE.md` §1.6 (residual and
//! explained-peak features) with the ion hypotheses of §2.1. Each kernel is a
//! copy of its host twin's lane in
//! `crate::models::ms2::formula_evidence`
//! ([`evidence_peaks_lane`](crate::models::ms2::formula_evidence::evidence_peaks_lane),
//! [`formula_evidence_lane`](crate::models::ms2::formula_evidence::formula_evidence_lane),
//! [`formula_features_lane`](crate::models::ms2::formula_evidence::formula_features_lane));
//! the twins were written first in the kernel-expressible form (full buffers
//! with explicit indices, `u32` loop counters from literals, no early `return`
//! inside loops), so both stay identical. Integer buffers are [`IdTensor`]
//! (`u32`); every launch goes through [`crate::backend::launch_1d_spans`]
//! with one lane per output item; shapes are checked to
//! [`crate::error::Error::Shape`] before any launch.
//!
//! CubeCL 0.10 deltas from the twin spelling (same arithmetic, mechanical):
//!
//! * No fixed-size local arrays as kernel values: the visit's nine lane
//!   registers are scalars (`d0..d8` digits, `r0..r8` radices), selected by
//!   slot with if-chains where the twin loops over its register arrays. Each
//!   block cites the twin source.
//! * No `wrapping_*` methods, `abs_diff`, `div_ceil`, `is_multiple_of` or
//!   `u32::MAX`: plain wrapping device ops, manual absolute differences,
//!   quotient/remainder ceilings and `4294967295u32`. `saturating_add` is a
//!   wrap-detect-and-clamp sequence with the same value.
//! * No `!` on booleans: explicit `== 0u32` / `== 1u32` comparisons on `u32`
//!   flags, or comparison-derived booleans. `u32` flags initialised from
//!   literals carry loop state (the macro cannot infer a literal
//!   `true`/`false`).
//! * Loop-carried variables start from literals or buffer loads, never a
//!   plain copy of a scalar argument.
//!
//! At most 6 arrays per kernel. Every output element is written by exactly
//! one lane; padding gets an explicit value; selection is by comparison,
//! never by multiplying with a mask. A buffer load inside an `if` is avoided
//! on hot paths: indices are computed unconditionally (clamped into range
//! first where the slot may be padding) and the guard applies where the
//! value is used.
//!
//! `kept_f [B, N, 2]` carries the relative intensity in column 0 and the
//! valid flag in column 1 (what `peak_select` writes); only column 0 is read
//! here.
//!
//! CubeCL 0.10 has no `!` on booleans, `RangeInclusive::contains`,
//! `usize::is_multiple_of` or `abs_diff` in kernels, so lanes use explicit
//! comparisons; the clippy lints for those patterns are allowed here rather
//! than rewritten.
#![allow(
    clippy::bool_comparison,
    clippy::manual_range_contains,
    clippy::manual_is_multiple_of,
    clippy::manual_abs_diff
)]

use cubecl::prelude::*;

use crate::backend::{FloatElem, launch_1d_spans};
use crate::error::{Error, Result};
use crate::tensor::base::Tensor;
use crate::tensor::ops::index::IdTensor;
use crate::tensor::ops::ms2_identity::{check_device_len, check_device_scalar};

// ---------------------------------------------------------------------------
// Shared `#[cube]` helpers (copies of the twin helpers)
// ---------------------------------------------------------------------------

/// Element mass of heavy slot `s` (C, N, O, F, P, S, Cl, Br, I).
/// Copy of [`crate::models::ms2::ion::heavy_mass_u32`].
#[allow(unused_assignments)]
#[cube]
fn ms2_fe_heavy_mass(slot: u32) -> u32 {
    let mut m: u32 = 12000000u32;
    if slot == 1u32 {
        m = 14003074u32;
    }
    if slot == 2u32 {
        m = 15994915u32;
    }
    if slot == 3u32 {
        m = 18998403u32;
    }
    if slot == 4u32 {
        m = 30973762u32;
    }
    if slot == 5u32 {
        m = 31972071u32;
    }
    if slot == 6u32 {
        m = 34968853u32;
    }
    if slot == 7u32 {
        m = 78918338u32;
    }
    if slot == 8u32 {
        m = 126904472u32;
    }
    m
}

/// Rounding residual of heavy slot `s`.
/// Copy of [`crate::models::ms2::ion::heavy_res_u32`].
#[allow(unused_assignments)]
#[cube]
fn ms2_fe_heavy_res(slot: u32) -> u32 {
    let mut r: u32 = 0u32;
    if slot == 1u32 {
        r = 5u32;
    }
    if slot == 2u32 {
        r = 381u32;
    }
    if slot == 3u32 {
        r = 163u32;
    }
    if slot == 4u32 {
        r = 2u32;
    }
    if slot == 5u32 {
        r = 175u32;
    }
    if slot == 6u32 {
        r = 318u32;
    }
    if slot == 7u32 {
        r = 400u32;
    }
    if slot == 8u32 {
        r = 100u32;
    }
    r
}

/// `floor(mz * t / 10^7)` in 32 bits (contract §5, the host form is
/// [`crate::models::ms2::chem::tolerance_u32`]): exact for `t <= 1000`, the
/// validated range, where `hi * t` and the second numerator fit in `u32`.
/// Copy of the `ms2_tolerance_u32` helper of `crate::tensor::ops::ms2`.
#[cube]
fn ms2_fe_tolerance_u32(mz: u32, t: u32) -> u32 {
    let hi = mz / 10000u32;
    let lo = mz % 10000u32;
    let q = hi * t;
    q / 1000u32 + ((q % 1000u32) * 10000u32 + lo * t) / 10000000u32
}

/// `a + b`, saturating at `u32::MAX`: the plain sum wraps, and is
/// overwritten exactly when it would have overflowed. Copy of the `ms2_sat_add`
/// helper of `crate::tensor::ops::ms2`.
#[cube]
fn ms2_fe_sat_add(a: u32, b: u32) -> u32 {
    let mut out = a + b;
    if out < a {
        out = 4294967295u32;
    }
    out
}

/// `a - b`, flooring at 0. Copy of the `ms2_sat_sub` helper of
/// `crate::tensor::ops::ms2`.
#[cube]
fn ms2_fe_sat_sub(a: u32, b: u32) -> u32 {
    let mut out = 0u32;
    if a > b {
        out = a - b;
    }
    out
}

// ---------------------------------------------------------------------------
// `ms2_evidence_peaks`: lane per spectrum
// ---------------------------------------------------------------------------

/// Lane per spectrum `b` of [`evidence_peaks`]; a copy of
/// [`crate::models::ms2::formula_evidence::evidence_peaks_lane`] over
/// `Array`s.
///
/// `P` passes over the `N` kept positions pick the eligible peaks in
/// (intensity descending, position ascending) order without a taken set; the
/// weights divide by the slot-ordered intensity sum. All other statements
/// are shared verbatim with the twin (modulo the spelling deltas above).
///
/// Selection applies the candidate-independent part of the scope test,
/// `tol_p.saturating_add(U) <= m_H`; whether a selected peak is in scope
/// for a particular candidate (`tol_p + U + E_ion(c) <= m_H`) is decided in
/// kernel 2, where an out-of-scope peak is unexplained for that candidate
/// and still counts in `n_ev`.
#[allow(clippy::too_many_arguments)]
#[allow(unused_assignments)]
#[cube(launch_unchecked)]
fn ms2_evidence_peaks_kernel<F: Float + CubeElement>(
    kept: &Array<u32>,
    kept_f: &Array<F>,
    meta: &Array<u32>,
    spec: &Array<u32>,
    ev_peaks: &mut Array<u32>,
    ev_w: &mut Array<F>,
    n: u32,
    p_dim: u32,
    lanes: usize,
    span: usize,
) {
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    for pos in start..end {
        let b = pos as u32;
        let meta_base = b * 8u32;
        let spec_base = b * 2u32;
        let peak_count = meta[meta_base as usize];
        let mut count = peak_count;
        if count > n {
            count = n;
        }
        let adduct_id = meta[(meta_base + 3u32) as usize];
        let ppm = meta[(meta_base + 4u32) as usize];
        let u_unc = spec[spec_base as usize];
        let known = adduct_id == 1u32 || adduct_id == 2u32;
        let mut prev_int = F::new(0.0_f32);
        let mut prev_pos: u32 = 0u32;
        let mut s: u32 = 0u32;
        while s < p_dim {
            let mut have: u32 = 0u32;
            let mut best_int = F::new(0.0_f32);
            let mut best_pos: u32 = 0u32;
            let mut best_mz: u32 = 0u32;
            let mut p: u32 = 0u32;
            while p < n {
                // Unconditional loads: every index below is in range.
                let mz = kept[((b * n + p) * 3u32 + 1u32) as usize];
                let inten = kept_f[((b * n + p) * 2u32) as usize];
                // `t_ok` and `tol_p` by the same `u32` algorithm as
                // `ion_assign_lane`; the target mass itself is only formed
                // for the winning row below.
                let mut t_ok: u32 = 0u32;
                if adduct_id == 1u32 && mz <= 4294967295u32 - 549u32 {
                    t_ok = 1u32;
                }
                if adduct_id == 2u32 && mz >= 549u32 {
                    t_ok = 1u32;
                }
                let tol_p = ms2_fe_tolerance_u32(mz, ppm);
                let half = ms2_fe_sat_add(tol_p, u_unc);
                let mut base_ok: u32 = 0u32;
                if p < count
                    && mz != 0u32
                    && known
                    && t_ok == 1u32
                    && u_unc != 4294967295u32
                    && half <= 1007825u32
                {
                    base_ok = 1u32;
                }
                // NaN gate by comparison only: NaN fails both arms, every
                // other float passes exactly one (so NaN can never win an
                // empty best slot by default).
                let zero = F::new(0.0_f32);
                let float_ok = inten > zero || inten <= zero;
                let mut after: u32 = 0u32;
                if s == 0u32 {
                    after = 1u32;
                }
                if inten < prev_int {
                    after = 1u32;
                }
                if inten == prev_int && p > prev_pos {
                    after = 1u32;
                }
                let mut beats: u32 = 0u32;
                if have == 0u32 {
                    beats = 1u32;
                }
                if have == 1u32 && inten > best_int {
                    beats = 1u32;
                }
                if have == 1u32 && inten == best_int && p < best_pos {
                    beats = 1u32;
                }
                if base_ok == 1u32 && float_ok && after == 1u32 && beats == 1u32 {
                    have = 1u32;
                    best_int = inten;
                    best_pos = p;
                    best_mz = mz;
                }
                p += 1u32;
            }
            let row = (b * p_dim + s) * 4u32;
            if have == 1u32 {
                // Same `t` / `tol_p` formation as above, for the winner.
                let mut t: u32 = 0u32;
                if adduct_id == 1u32 {
                    t = best_mz + 549u32;
                }
                if adduct_id == 2u32 {
                    t = best_mz - 549u32;
                }
                let tol_p = ms2_fe_tolerance_u32(best_mz, ppm);
                ev_peaks[row as usize] = best_pos;
                ev_peaks[(row + 1u32) as usize] = t;
                ev_peaks[(row + 2u32) as usize] = tol_p;
                ev_peaks[(row + 3u32) as usize] = 1u32;
                prev_int = best_int;
                prev_pos = best_pos;
            } else {
                ev_peaks[row as usize] = 4294967295u32;
                ev_peaks[(row + 1u32) as usize] = 0u32;
                ev_peaks[(row + 2u32) as usize] = 0u32;
                ev_peaks[(row + 3u32) as usize] = 0u32;
            }
            s += 1u32;
        }
        // Weights in two passes that re-read the selected rows: the sum in
        // slot order, then each slot's share. Loads are unconditional (the
        // position is clamped into range first); the valid flag selects.
        let mut sum = F::new(0.0_f32);
        let mut s2: u32 = 0u32;
        while s2 < p_dim {
            let wpos = ev_peaks[((b * p_dim + s2) * 4u32) as usize];
            let mut safe: u32 = 0u32;
            if wpos < n {
                safe = wpos;
            }
            let wi = kept_f[((b * n + safe) * 2u32) as usize];
            if ev_peaks[((b * p_dim + s2) * 4u32 + 3u32) as usize] == 1u32 {
                sum += wi;
            }
            s2 += 1u32;
        }
        let mut s3: u32 = 0u32;
        while s3 < p_dim {
            let rpos = ev_peaks[((b * p_dim + s3) * 4u32) as usize];
            let mut rsafe: u32 = 0u32;
            if rpos < n {
                rsafe = rpos;
            }
            let ri = kept_f[((b * n + rsafe) * 2u32) as usize];
            let mut wv = F::new(0.0_f32);
            if ev_peaks[((b * p_dim + s3) * 4u32 + 3u32) as usize] == 1u32
                && sum > F::new(0.0_f32)
            {
                wv = ri / sum;
            }
            ev_w[(b * p_dim + s3) as usize] = wv;
            s3 += 1u32;
        }
    }
}

/// Evidence peaks of one `(B, N, P)` bucket, lane per spectrum.
///
/// `kept [B, N, 3]` holds the kept peaks (raw index, m/z, reverse),
/// `kept_f [B, N, 2]` the relative intensity in column 0 with the valid flag
/// in column 1 (what `peak_select` writes), `meta [B, 8]` the spectrum words
/// (peak count, precursor, precursor uncertainty, adduct,
/// fragment-tolerance ppm tenths, precursor tolerance, id lo/hi) and
/// `spec [B, 2]` the m/z uncertainty with one reserved word. Writes
/// `ev_peaks [B, P, 4]` (kept position, target mass `t`, fragment tolerance
/// `tol_p`, valid flag `1`; padding `u32::MAX, 0, 0, 0`) and `ev_w [B, P]`
/// (intensity over the slot-ordered sum; `0` in padding, all `0` when the
/// sum is not positive). The evidence mask of [`formula_evidence`] is one
/// `u32`, so `P == 0` and `P > 32` are refused with [`Error::Shape`], as is
/// `N == 0`. `B == 0` returns `Ok(())` with zero launches. Exactly 1 launch,
/// one lane per spectrum.
///
/// Selection applies the candidate-independent part of the scope test,
/// `tol_p.saturating_add(U) <= m_H`; whether a selected peak is in scope
/// for a particular candidate (`tol_p + U + E_ion(c) <= m_H`) is decided in
/// kernel 2, where an out-of-scope peak is unexplained for that candidate
/// and still counts in `n_ev`.
pub fn evidence_peaks<R: Runtime, E: FloatElem>(
    kept: &IdTensor<R>,
    kept_f: &Tensor<R, E>,
    meta: &IdTensor<R>,
    spec: &IdTensor<R>,
    ev_peaks: &mut IdTensor<R>,
    ev_w: &mut Tensor<R, E>,
) -> Result<()> {
    if kept.shape().rank() != 3
        || kept_f.shape().rank() != 3
        || meta.shape().rank() != 2
        || spec.shape().rank() != 2
        || ev_peaks.shape().rank() != 3
        || ev_w.shape().rank() != 2
    {
        return Err(Error::shape(format!(
            "evidence_peaks needs kept [B, N, 3], kept_f [B, N, 2], meta [B, 8], spec [B, 2], ev_peaks [B, P, 4] and ev_w [B, P], got {} and {} and {} and {} and {} and {}",
            kept.shape(),
            kept_f.shape(),
            meta.shape(),
            spec.shape(),
            ev_peaks.shape(),
            ev_w.shape()
        )));
    }
    let batch = kept.shape().dim(0);
    let n = kept.shape().dim(1);
    let p = ev_peaks.shape().dim(1);
    if kept.shape().dim(2) != 3 || kept_f.shape().dim(2) != 2 || ev_peaks.shape().dim(2) != 4 {
        return Err(Error::shape(format!(
            "evidence_peaks needs last dimensions 3, 2 and 4, got {} and {} and {}",
            kept.shape(),
            kept_f.shape(),
            ev_peaks.shape()
        )));
    }
    let want_kept: &[usize] = &[batch, n, 3];
    let want_kept_f: &[usize] = &[batch, n, 2];
    let want_meta: &[usize] = &[batch, 8];
    let want_spec: &[usize] = &[batch, 2];
    let want_peaks: &[usize] = &[batch, p, 4];
    let want_w: &[usize] = &[batch, p];
    if kept.shape().dims() != want_kept
        || kept_f.shape().dims() != want_kept_f
        || meta.shape().dims() != want_meta
        || spec.shape().dims() != want_spec
        || ev_peaks.shape().dims() != want_peaks
        || ev_w.shape().dims() != want_w
    {
        return Err(Error::shape(format!(
            "evidence_peaks needs kept [{batch}, {n}, 3], kept_f [{batch}, {n}, 2], meta [{batch}, 8], spec [{batch}, 2], ev_peaks [{batch}, {p}, 4] and ev_w [{batch}, {p}], got {} and {} and {} and {} and {} and {}",
            kept.shape(),
            kept_f.shape(),
            meta.shape(),
            spec.shape(),
            ev_peaks.shape(),
            ev_w.shape()
        )));
    }
    if p > 32 {
        return Err(Error::shape(format!(
            "evidence_peaks needs P <= 32 for the u32 explained-peak mask, got P = {p}"
        )));
    }
    if n == 0 {
        return Err(Error::shape(
            "evidence_peaks needs N >= 1 (empty kept buffers read out of bounds), got N = 0"
                .to_string(),
        ));
    }
    if p == 0 {
        return Err(Error::shape(
            "evidence_peaks needs P >= 1 (no evidence slot to write), got P = 0".to_string(),
        ));
    }
    if batch == 0 {
        return Ok(());
    }
    check_device_len("kept", kept.len())?;
    check_device_len("kept_f", kept_f.len())?;
    check_device_len("meta", meta.len())?;
    check_device_len("spec", spec.len())?;
    check_device_len("ev_peaks", ev_peaks.len())?;
    check_device_len("ev_w", ev_w.len())?;
    let n_u32 = check_device_scalar("evidence_peaks N", n)?;
    let p_u32 = check_device_scalar("evidence_peaks P", p)?;
    // `batch` fits `u32`, so every `pos < batch` narrows to a `u32` lane.
    check_device_scalar("evidence_peaks B lanes", batch)?;
    let client = kept.client();
    let (count, dim, span) = launch_1d_spans(client, batch, n.saturating_mul(p).max(1));
    unsafe {
        ms2_evidence_peaks_kernel::launch_unchecked::<E, R>(
            client,
            count,
            dim,
            kept.arg(),
            kept_f.arg(),
            meta.arg(),
            spec.arg(),
            ev_peaks.arg(),
            ev_w.arg(),
            n_u32,
            p_u32,
            batch,
            span,
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// `ms2_formula_evidence`: lane per `(b, m)`
// ---------------------------------------------------------------------------

/// Lane per `(b, m)` of [`formula_evidence`]; a copy of
/// [`crate::models::ms2::formula_evidence::formula_evidence_lane`] over
/// `Array`s.
///
/// The walk enumerates the non-carbon heavy vectors `u'` by mixed-radix
/// index `j` over slots 1..8 (slot 1 least significant) with the eight lane
/// registers scalar-expanded (`d1..d8` digits, `r1..r8` radices, selected by
/// slot through an if-chain where the twin loops over its register arrays);
/// the guarded radix-product budget, the per-`h` carbon closed form and the
/// wrapped hydrogen ranges are the twin's blocks with the same guards
/// (division before every product, saturating adds, `t - delta` by
/// saturation instead of any wrapping subtraction). All other statements
/// are shared verbatim with the twin (modulo the spelling deltas above).
///
/// Proof sketch (the twin's doc, cited here): acceptance implies the lane
/// window and 3-slot cap never cut it (an accepted `cand` within `tol <=
/// half_p` of `t`; at most three hydrogen integers under `half_p <= m_H`;
/// carbon residue-free); one multiple of `12,000,000` per `2 delta`
/// interval (`2 delta <= 2 m_H < 12,000,000`); the wrapped fast ranges from
/// `X = t + tol - m'`, `Rm = X mod 1,000,000`, `S = 7,825 h + D = Rm +
/// 1,000,000 s` over `s in 0 ..= s_max` with `s_max = (7,825 h_cap + 2 tol)
/// / 1,000,000` (the old `7,825 h_cap + 2 tol < 1,000,000` fast range is the
/// `s_max = 0` case; ranges may overlap and trying an `h` twice is
/// harmless). Budget: `visits = min(J, W)` over slots 1..8 with `complete
/// = (J <= W)`; `ion_assign`'s visit index is `k = n + (c[C] + 1) j`.
/// Worst-case cost per lane: `W * P * trials` hydrogen trials with `trials`
/// the item-1 bound from the lane's clamped `h_cap` and the peak's `tol`.
/// The lane clamps its `h_cap` to the host `h_cap_max` scalar before sizing
/// its ranges, enforcing what the dispatch sizing assumes.
///
/// Selection (kernel 1) applies the candidate-independent part of the scope
/// test, `tol_p.saturating_add(U) <= m_H`; whether a selected peak is in
/// scope for a particular candidate (`tol_p + U + E_ion(c) <= m_H`) is
/// decided here, where an out-of-scope peak is unexplained for that
/// candidate and still counts in `n_ev`.
#[allow(clippy::too_many_arguments)]
#[allow(unused_assignments)]
#[cube(launch_unchecked)]
fn ms2_formula_evidence_kernel<F: Float + CubeElement>(
    cand: &Array<u32>,
    ev_peaks: &Array<u32>,
    ev_w: &Array<F>,
    meta: &Array<u32>,
    spec: &Array<u32>,
    cand_ev: &mut Array<F>,
    m_dim: u32,
    p_dim: u32,
    work_max: u32,
    h_cap_max: u32,
    first_lane: usize,
    lanes: usize,
    span: usize,
) {
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    for local in start..end {
        // Bounded dispatch: this launch covers the contiguous lane range
        // `[first_lane, first_lane + lanes)`; the absolute lane index keeps
        // results identical to one unchunked launch.
        let pos = first_lane + local;
        let lane = pos as u32;
        let b = lane / m_dim;
        let m = lane % m_dim;
        let cand_base = (b * m_dim + m) * 13u32;
        let ob = (b * m_dim + m) * 4u32;
        // Every output element is written: zero the row first.
        cand_ev[ob as usize] = F::new(0.0_f32);
        cand_ev[(ob + 1u32) as usize] = F::new(0.0_f32);
        cand_ev[(ob + 2u32) as usize] = F::new(0.0_f32);
        cand_ev[(ob + 3u32) as usize] = F::new(0.0_f32);
        let flag = cand[(cand_base + 11u32) as usize];
        if flag != 0u32 {
            let mut n_ev: u32 = 0u32;
            let mut sv: u32 = 0u32;
            while sv < p_dim {
                if ev_peaks[((b * p_dim + sv) * 4u32 + 3u32) as usize] == 1u32 {
                    n_ev += 1u32;
                }
                sv += 1u32;
            }
            // Lane registers: candidate counts and non-carbon radices
            // (`c[e] + 1` over slots 1..8, carbon kept as `c0` for the
            // closed-form `n1 <= c[C]` check).
            let c0 = cand[cand_base as usize];
            let c1 = cand[(cand_base + 1u32) as usize];
            let c2 = cand[(cand_base + 2u32) as usize];
            let c3 = cand[(cand_base + 3u32) as usize];
            let c4 = cand[(cand_base + 4u32) as usize];
            let c5 = cand[(cand_base + 5u32) as usize];
            let c6 = cand[(cand_base + 6u32) as usize];
            let c7 = cand[(cand_base + 7u32) as usize];
            let c8 = cand[(cand_base + 8u32) as usize];
            let c9 = cand[(cand_base + 9u32) as usize];
            let r1 = c2 + 1u32;
            let r2 = c3 + 1u32;
            let r3 = c4 + 1u32;
            let r4 = c5 + 1u32;
            let r5 = c6 + 1u32;
            let r6 = c7 + 1u32;
            let r7 = c8 + 1u32;
            let r8 = c9 + 1u32;
            // Guarded non-carbon radix product `J` against `W` (the twin's
            // budget block with the radix selected by slot through an
            // if-chain): each multiply runs only when `partial <= work_max
            // / r` proves it fits and stays within budget. `visits = min(J,
            // W)`; `complete = (J <= W)`.
            let mut partial: u32 = 1u32;
            let mut cut: u32 = 0u32;
            let mut gi: u32 = 0u32;
            while gi < 8u32 {
                let mut r = r1;
                if gi == 1u32 {
                    r = r2;
                }
                if gi == 2u32 {
                    r = r3;
                }
                if gi == 3u32 {
                    r = r4;
                }
                if gi == 4u32 {
                    r = r5;
                }
                if gi == 5u32 {
                    r = r6;
                }
                if gi == 6u32 {
                    r = r7;
                }
                if gi == 7u32 {
                    r = r8;
                }
                if r == 0u32 {
                    r = 1u32;
                    cut = 1u32;
                }
                if cut == 0u32 && r > 1u32 {
                    if partial > work_max / r {
                        cut = 1u32;
                    } else {
                        partial *= r;
                    }
                }
                gi += 1u32;
            }
            let mut visits = partial;
            if cut == 1u32 {
                visits = work_max;
            }
            if work_max == 0u32 {
                visits = 0u32;
                cut = 1u32;
            }
            let adduct_id = meta[(b * 8u32 + 3u32) as usize];
            let u_unc = spec[(b * 2u32) as usize];
            // `E_ion` upper bound over the candidate residuals (+3 H +
            // electron), as in `ion_assign_lane`.
            let mut nda: u32 = 0u32;
            nda += c0 * ms2_fe_heavy_res(0u32);
            nda += c2 * ms2_fe_heavy_res(1u32);
            nda += c3 * ms2_fe_heavy_res(2u32);
            nda += c4 * ms2_fe_heavy_res(3u32);
            nda += c5 * ms2_fe_heavy_res(4u32);
            nda += c6 * ms2_fe_heavy_res(5u32);
            nda += c7 * ms2_fe_heavy_res(6u32);
            nda += c8 * ms2_fe_heavy_res(7u32);
            nda += c9 * ms2_fe_heavy_res(8u32);
            nda += c1 * 33u32;
            nda += 3u32 * 33u32 + 421u32;
            let mut e_add: u32 = 0u32;
            if nda % 1000u32 != 0u32 {
                e_add = 1u32;
            }
            let e_ion = nda / 1000u32 + e_add;
            let mut h_pos: u32 = 0u32;
            if adduct_id == 1u32 {
                h_pos = 1u32;
            }
            let h_cap = c1 + h_pos + 2u32;
            let mut h_hi_abs = h_cap;
            if h_hi_abs > 65535u32 {
                h_hi_abs = 65535u32;
            }
            if h_hi_abs > h_cap_max {
                h_hi_abs = h_cap_max;
            }
            let mut mask: u32 = 0u32;
            let mut explained: u32 = 0u32;
            let mut j: u32 = 0u32;
            let mut live: u32 = 1u32;
            if n_ev == 0u32 {
                live = 0u32;
            }
            while j < visits && live == 1u32 {
                // One visit: mixed-radix digits of `j` over slots 1..8,
                // slot 1 least significant (the twin's digits block,
                // scalar-expanded). `j = 0` is the all-zero vector.
                let mut tmp = j;
                let d1 = tmp % r1;
                tmp /= r1;
                let d2 = tmp % r2;
                tmp /= r2;
                let d3 = tmp % r3;
                tmp /= r3;
                let d4 = tmp % r4;
                tmp /= r4;
                let d5 = tmp % r5;
                tmp /= r5;
                let d6 = tmp % r6;
                tmp /= r6;
                let d7 = tmp % r7;
                tmp /= r7;
                let d8 = tmp % r8;
                tmp /= r8;
                // Non-carbon heavy mass (division-guarded) and residual sum
                // (saturating): the twin's mass block with the digit, mass
                // and residual selected by slot through if-chains.
                let mut mprime: u32 = 0u32;
                let mut mass_ok: u32 = 1u32;
                let mut resprime: u32 = 0u32;
                let mut zero_u: u32 = 1u32;
                let mut di: u32 = 0u32;
                while di < 8u32 {
                    let mut d = d1;
                    if di == 1u32 {
                        d = d2;
                    }
                    if di == 2u32 {
                        d = d3;
                    }
                    if di == 3u32 {
                        d = d4;
                    }
                    if di == 4u32 {
                        d = d5;
                    }
                    if di == 5u32 {
                        d = d6;
                    }
                    if di == 6u32 {
                        d = d7;
                    }
                    if di == 7u32 {
                        d = d8;
                    }
                    let me = ms2_fe_heavy_mass(di + 1u32);
                    let re = ms2_fe_heavy_res(di + 1u32);
                    let room = 4294967295u32 - mprime;
                    let mut fits: u32 = 0u32;
                    if d <= room / me {
                        fits = 1u32;
                    }
                    let go = mass_ok == 1u32 && fits == 1u32;
                    if mass_ok == 1u32 && fits == 1u32 {
                        mass_ok = 1u32;
                    } else {
                        mass_ok = 0u32;
                    }
                    if go {
                        mprime += d * me;
                    }
                    let rroom = 4294967295u32 - resprime;
                    let mut rfits: u32 = 0u32;
                    if d <= rroom / re {
                        rfits = 1u32;
                    }
                    if rfits == 1u32 {
                        resprime += d * re;
                    } else {
                        resprime = 4294967295u32;
                    }
                    if d != 0u32 {
                        zero_u = 0u32;
                    }
                    di += 1u32;
                }
                // Evidence peaks with unconditional loads: indices always in
                // range; the valid flag, the scope gate and the explained
                // bit select.
                let mut s: u32 = 0u32;
                while s < p_dim {
                    let t_s = ev_peaks[((b * p_dim + s) * 4u32 + 1u32) as usize];
                    let tol_s = ev_peaks[((b * p_dim + s) * 4u32 + 2u32) as usize];
                    let valid_s = ev_peaks[((b * p_dim + s) * 4u32 + 3u32) as usize];
                    // `half_p` by saturating additions (wrap-detect-and-clamp
                    // in `ms2_fe_sat_add`).
                    let half_p = ms2_fe_sat_add(ms2_fe_sat_add(tol_s, u_unc), e_ion);
                    let bit = 1u32 << s;
                    let gated = valid_s == 1u32
                        && half_p <= 1007825u32
                        && (mask & bit) == 0u32
                        && mass_ok == 1u32;
                    if gated {
                        // Wrapped hydrogen trial ranges for this (visit,
                        // peak): `s_max = (7,825 h_hi_abs + 2 tol_s) /
                        // 1,000,000` with guards (`h_hi_abs <= 65535`, so
                        // `p1 <= 512,806,375`; `tol_s <= m_H` under the scope
                        // gate, so `2 * tol_s` fits; the sum is below 2^30).
                        // The wrapped ranges run when `(s_max + 1) * (2 tol /
                        // 7,825 + 2) <= h_hi_abs + 1` (identical result,
                        // proved in the twin's doc); otherwise the plain
                        // `0 ..= h_hi_abs` range runs. Every tried `h` below
                        // goes through the same exact per-`h` test.
                        let mut s_max: u32 = 0u32;
                        let mut range_ok: u32 = 0u32;
                        if h_hi_abs <= 4294967295u32 / 7825u32 {
                            let p1 = h_hi_abs * 7825u32;
                            if tol_s <= (4294967295u32 - p1) / 2u32 {
                                let hsum = p1 + tol_s * 2u32;
                                s_max = hsum / 1000000u32;
                                range_ok = 1u32;
                            }
                        }
                        // `(s_max + 1) * per_s` with its guard (`s_max <=
                        // ~514`, so `s_max + 1` cannot wrap; `per_s >= 2`,
                        // so the division is safe; the product is below
                        // 2^30 for valid inputs).
                        let mut use_wrapped: u32 = 0u32;
                        if range_ok == 1u32 {
                            let two_tol_w = tol_s * 2u32;
                            let per_s = two_tol_w / 7825u32 + 2u32;
                            let s1 = s_max + 1u32;
                            if s1 <= 4294967295u32 / per_s {
                                let wrapped = s1 * per_s;
                                // `h_hi_abs <= 65535`, so `+ 1` cannot wrap.
                                let plain = h_hi_abs + 1u32;
                                if wrapped <= plain {
                                    use_wrapped = 1u32;
                                }
                            }
                        }
                        if use_wrapped == 1u32 {
                            let mut empty: u32 = 0u32;
                            if mprime > t_s && mprime - t_s > tol_s {
                                empty = 1u32;
                            }
                            if empty == 0u32 {
                                // `Rm = (t + tol - mprime) mod 1,000,000`
                                // from residues only (`X` itself can exceed
                                // `u32` for precursor-scale masses).
                                let tm = t_s % 1000000u32;
                                let lm = tol_s % 1000000u32;
                                let mm = mprime % 1000000u32;
                                let mut ssum = tm + lm;
                                if ssum >= 1000000u32 {
                                    ssum -= 1000000u32;
                                }
                                let mut rm = ssum + 1000000u32 - mm;
                                if ssum >= mm {
                                    rm = ssum - mm;
                                }
                                // `2 * tol_s` fits: guarded above.
                                let two_tol = tol_s * 2u32;
                                let mut sw: u32 = 0u32;
                                while sw <= s_max {
                                    // `base = Rm + 1,000,000 sw` with its
                                    // guards (`sw <= s_max <= ~514`, so both
                                    // fit for valid inputs; the dead else
                                    // never engages there).
                                    let mut prod: u32 = 0u32;
                                    if sw <= 4294967295u32 / 1000000u32 {
                                        prod = 1000000u32 * sw;
                                    }
                                    let mut base: u32 = rm;
                                    if prod <= 4294967295u32 - rm {
                                        base = rm + prod;
                                    }
                                    let mut lo_need: u32 = 0u32;
                                    if base > two_tol {
                                        lo_need = base - two_tol;
                                    }
                                    let mut h_lo = lo_need / 7825u32;
                                    if lo_need % 7825u32 != 0u32 {
                                        h_lo += 1u32;
                                    }
                                    let mut h_hi = base / 7825u32;
                                    if h_hi > h_hi_abs {
                                        h_hi = h_hi_abs;
                                    }
                                    if h_lo <= h_hi {
                                        // `h` starts from a literal plus the
                                        // computed start (a plain copy of a
                                        // computed bound would risk the
                                        // launch-drop this crate documents).
                                        // `h_hi <= 65535`, so `h += 1`
                                        // cannot wrap.
                                        let mut h: u32 = 0u32 + h_lo;
                                        while h <= h_hi {
                            // `base = mprime + m_H * h` only under the guard
                            // that proves it fits `u32`.
                            let mut fits_h: u32 = 0u32;
                            if h <= (4294967295u32 - mprime) / 1007825u32 {
                                fits_h = 1u32;
                            }
                            if fits_h == 1u32 {
                                let base = mprime + h * 1007825u32;
                                // `resprime + 33 h + 421` with overflow
                                // saturation (a saturated residual fails the
                                // `bound <= tol` test below).
                                let mut rok: u32 = 1u32;
                                let mut rh = resprime;
                                if h > (4294967295u32 - rh) / 33u32 {
                                    rok = 0u32;
                                } else {
                                    rh += h * 33u32;
                                }
                                if rok == 1u32 && rh > 4294967295u32 - 421u32 {
                                    rok = 0u32;
                                }
                                if rok == 1u32 {
                                    rh += 421u32;
                                }
                                let mut arith_add: u32 = 0u32;
                                if rh % 1000u32 != 0u32 {
                                    arith_add = 1u32;
                                }
                                let arith = rh / 1000u32 + arith_add;
                                let bound = ms2_fe_sat_add(arith, u_unc);
                                if rok == 1u32 && bound <= tol_s {
                                    // The only carbon multiple of the `2
                                    // delta` interval (`2 delta <= 2 m_H <
                                    // 12,000,000`, proved in the twin's
                                    // doc).
                                    let delta = tol_s - bound;
                                    let top = ms2_fe_sat_add(t_s, delta);
                                    if top >= base {
                                        let n1 = (top - base) / 12000000u32;
                                        if n1 <= c0 && n1 <= 4294967295u32 / 12000000u32 {
                                            // `n1 * 12,000,000 <= top -
                                            // base`, so both sums below fit
                                            // `u32`.
                                            let lhs = n1 * 12000000u32 + base;
                                            // `lhs + delta >= t` without a
                                            // wrapping subtraction.
                                            let need = ms2_fe_sat_sub(t_s, delta);
                                            let not_both_zero = n1 != 0u32 || zero_u == 0u32;
                                            if lhs >= need && not_both_zero {
                                                if (mask & bit) == 0u32 {
                                                    mask |= bit;
                                                    explained += 1u32;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            h += 1u32;
                                        }
                                    }
                                    sw += 1u32;
                                }
                            }
                        } else {
                            // Plain `0 ..= h_hi_abs` range with the same
                            // exact per-`h` test as above. `h_hi_abs <=
                            // 65535`, so `h += 1` cannot wrap.
                            let mut h: u32 = 0u32;
                            while h <= h_hi_abs {
                                let mut fits_h: u32 = 0u32;
                                if h <= (4294967295u32 - mprime) / 1007825u32 {
                                    fits_h = 1u32;
                                }
                                if fits_h == 1u32 {
                                    let base = mprime + h * 1007825u32;
                                    let mut rok: u32 = 1u32;
                                    let mut rh = resprime;
                                    if h > (4294967295u32 - rh) / 33u32 {
                                        rok = 0u32;
                                    } else {
                                        rh += h * 33u32;
                                    }
                                    if rok == 1u32 && rh > 4294967295u32 - 421u32 {
                                        rok = 0u32;
                                    }
                                    if rok == 1u32 {
                                        rh += 421u32;
                                    }
                                    let mut arith_add: u32 = 0u32;
                                    if rh % 1000u32 != 0u32 {
                                        arith_add = 1u32;
                                    }
                                    let arith = rh / 1000u32 + arith_add;
                                    let bound = ms2_fe_sat_add(arith, u_unc);
                                    if rok == 1u32 && bound <= tol_s {
                                        let delta = tol_s - bound;
                                        let top = ms2_fe_sat_add(t_s, delta);
                                        if top >= base {
                                            let n1 = (top - base) / 12000000u32;
                                            if n1 <= c0 && n1 <= 4294967295u32 / 12000000u32 {
                                                let lhs = n1 * 12000000u32 + base;
                                                let need = ms2_fe_sat_sub(t_s, delta);
                                                let not_both_zero = n1 != 0u32 || zero_u == 0u32;
                                                if lhs >= need && not_both_zero {
                                                    if (mask & bit) == 0u32 {
                                                        mask |= bit;
                                                        explained += 1u32;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                h += 1u32;
                            }
                        }
                    }
                    s += 1u32;
                }
                if explained == n_ev {
                    live = 0u32;
                }
                j += 1u32;
            }
            // Weight sum over set bits in ascending slot order: each weight
            // is loaded unconditionally, then added under the bit test.
            let mut wsum = F::new(0.0_f32);
            let mut sw: u32 = 0u32;
            while sw < p_dim {
                let bit = 1u32 << sw;
                let wv = ev_w[(b * p_dim + sw) as usize];
                if (mask & bit) != 0u32 {
                    wsum += wv;
                }
                sw += 1u32;
            }
            cand_ev[ob as usize] = F::cast_from(explained);
            cand_ev[(ob + 1u32) as usize] = wsum;
            cand_ev[(ob + 2u32) as usize] = F::cast_from(n_ev);
            let mut comp_v = F::new(0.0_f32);
            if cut == 0u32 {
                comp_v = F::new(1.0_f32);
            }
            cand_ev[(ob + 3u32) as usize] = comp_v;
        }
    }
}

/// Candidate evidence of one `(B, M, P)` bucket, lane per `(b, m)`.
///
/// `cand [B, M, 13]` holds the scored candidates (10 counts in `ELEMENTS`
/// order, integer mass, flag, source id), `ev_peaks [B, P, 4]` and
/// `ev_w [B, P]` the evidence peaks of [`evidence_peaks`], `meta [B, 8]` the
/// spectrum words and `spec [B, 2]` the m/z uncertainty. Writes
/// `cand_ev [B, M, 4]` (explained-peak count, explained weight, evidence
/// count, complete flag; `0, 0, 0, 0` in a padding slot). `work_max`
/// (`W`, the per-lane visit budget) and `dispatch_tests_max` (the worst-case
/// hydrogen trials per launch) must both be non-zero ([`Error::Config`]
/// otherwise); the evidence mask is one `u32`, so `P == 0` and `P > 32`
/// are [`Error::Shape`]. `B == 0` or `M == 0` returns `Ok(())` with zero
/// launches.
///
/// `h_cap_max` is an upper bound of `c[H] + 3` over every candidate the
/// request can score (table source: the table's largest hydrogen count + 3;
/// enumerating source: the artifacts' hydrogen bound + 3; both known on the
/// host without a read) and `tol_max` the largest fragment tolerance of the
/// batch (the tolerance at the largest uploaded peak m/z under the
/// request's ppm, host data). With `s_max = (7,825 h_cap_max + 2 tol_max) /
/// 1,000,000`, `trials_bound = min(h_cap_max + 1, (s_max + 1) * (2 tol_max /
/// 7,825 + 2))`, the per-lane bound is `work_max * P * trials_bound`
/// (checked `u64`).
///
/// Selection (kernel 1) applies the candidate-independent part of the scope
/// test, `tol_p.saturating_add(U) <= m_H`; whether a selected peak is in
/// scope for a particular candidate (`tol_p + U + E_ion(c) <= m_H`) is
/// decided here, where an out-of-scope peak is unexplained for that
/// candidate and still counts in `n_ev`.
///
/// Bounded dispatch (the `enum_count` pattern): the `B * M` lanes run in
/// contiguous chunks of `max(1, dispatch_tests_max / per_lane_bound)`
/// lanes, each chunk its own launch and its own submission, so the worst
/// case of one GPU job is about `dispatch_tests_max` hydrogen trials. The
/// lane takes the absolute lane index, so chunking changes no result. The
/// launch count is `ceil(B * M / lanes_per_launch)`; no launch when `B * M`
/// is 0. The kernel enforces what the sizing assumes by clamping each
/// lane's `h_cap` to `h_cap_max` (a correct bound never engages the clamp).
pub fn formula_evidence<R: Runtime, E: FloatElem>(
    cand: &IdTensor<R>,
    ev_peaks: &IdTensor<R>,
    ev_w: &Tensor<R, E>,
    meta: &IdTensor<R>,
    spec: &IdTensor<R>,
    cand_ev: &mut Tensor<R, E>,
    work_max: u32,
    dispatch_tests_max: u64,
    h_cap_max: u32,
    tol_max: u32,
) -> Result<()> {
    if cand.shape().rank() != 3
        || ev_peaks.shape().rank() != 3
        || ev_w.shape().rank() != 2
        || meta.shape().rank() != 2
        || spec.shape().rank() != 2
        || cand_ev.shape().rank() != 3
    {
        return Err(Error::shape(format!(
            "formula_evidence needs cand [B, M, 13], ev_peaks [B, P, 4], ev_w [B, P], meta [B, 8], spec [B, 2] and cand_ev [B, M, 4], got {} and {} and {} and {} and {} and {}",
            cand.shape(),
            ev_peaks.shape(),
            ev_w.shape(),
            meta.shape(),
            spec.shape(),
            cand_ev.shape()
        )));
    }
    let batch = cand.shape().dim(0);
    let m = cand.shape().dim(1);
    let p = ev_peaks.shape().dim(1);
    if cand.shape().dim(2) != 13 || ev_peaks.shape().dim(2) != 4 || cand_ev.shape().dim(2) != 4 {
        return Err(Error::shape(format!(
            "formula_evidence needs last dimensions 13, 4 and 4, got {} and {} and {}",
            cand.shape(),
            ev_peaks.shape(),
            cand_ev.shape()
        )));
    }
    let want_cand: &[usize] = &[batch, m, 13];
    let want_peaks: &[usize] = &[batch, p, 4];
    let want_w: &[usize] = &[batch, p];
    let want_meta: &[usize] = &[batch, 8];
    let want_spec: &[usize] = &[batch, 2];
    let want_ev: &[usize] = &[batch, m, 4];
    if cand.shape().dims() != want_cand
        || ev_peaks.shape().dims() != want_peaks
        || ev_w.shape().dims() != want_w
        || meta.shape().dims() != want_meta
        || spec.shape().dims() != want_spec
        || cand_ev.shape().dims() != want_ev
    {
        return Err(Error::shape(format!(
            "formula_evidence needs cand [{batch}, {m}, 13], ev_peaks [{batch}, {p}, 4], ev_w [{batch}, {p}], meta [{batch}, 8], spec [{batch}, 2] and cand_ev [{batch}, {m}, 4], got {} and {} and {} and {} and {} and {}",
            cand.shape(),
            ev_peaks.shape(),
            ev_w.shape(),
            meta.shape(),
            spec.shape(),
            cand_ev.shape()
        )));
    }
    if p > 32 {
        return Err(Error::shape(format!(
            "formula_evidence needs P <= 32 for the u32 explained-peak mask, got P = {p}"
        )));
    }
    if p == 0 {
        return Err(Error::shape(
            "formula_evidence needs P >= 1 (no evidence slot for the u32 mask), got P = 0"
                .to_string(),
        ));
    }
    if work_max == 0 {
        return Err(Error::config(
            "formula_evidence needs work_max (formula_evidence_work_max) non-zero".to_string(),
        ));
    }
    if dispatch_tests_max == 0 {
        return Err(Error::config(
            "formula_evidence needs dispatch_tests_max non-zero".to_string(),
        ));
    }
    let lanes = batch.checked_mul(m).ok_or_else(|| {
        Error::shape(format!(
            "formula_evidence: batch {batch} times {m} lanes overflows usize"
        ))
    })?;
    if lanes == 0 {
        return Ok(());
    }
    check_device_len("cand", cand.len())?;
    check_device_len("ev_peaks", ev_peaks.len())?;
    check_device_len("ev_w", ev_w.len())?;
    check_device_len("meta", meta.len())?;
    check_device_len("spec", spec.len())?;
    check_device_len("cand_ev", cand_ev.len())?;
    let m_u32 = check_device_scalar("formula_evidence M", m)?;
    let p_u32 = check_device_scalar("formula_evidence P", p)?;
    // `lanes` fits `u32`, so every absolute lane below narrows without
    // wrapping.
    check_device_scalar("formula_evidence B * M lanes", lanes)?;
    // Bounded dispatch (the `enum_count` pattern): one launch covers at most
    // about `dispatch_tests_max` hydrogen trials (`work_max * P *
    // trials_bound` per lane, with `trials_bound` from the host-known
    // `h_cap_max` / `tol_max`).
    let s_max_host =
        (u64::from(h_cap_max) * 7_825 + 2 * u64::from(tol_max)) / 1_000_000;
    let per_s_host = (2 * u64::from(tol_max)) / 7_825 + 2;
    let wrapped_host = (s_max_host + 1) * per_s_host;
    let trials_bound = wrapped_host.min(u64::from(h_cap_max) + 1).max(1);
    let per_lane = (work_max as u64)
        .checked_mul(p as u64)
        .and_then(|v| v.checked_mul(trials_bound))
        .ok_or_else(|| {
            Error::config(format!(
                "formula_evidence: work_max {work_max} times P {p} times trials_bound {trials_bound} overflows u64"
            ))
        })?
        .max(1);
    let per = (dispatch_tests_max / per_lane).max(1) as usize;
    let n_launches = lanes.div_ceil(per);
    let client = cand.client();
    for i in 0..n_launches {
        let first = i * per;
        let chunk = (lanes - first).min(per);
        let (count, dim, span) = launch_1d_spans(client, chunk, 1024);
        unsafe {
            ms2_formula_evidence_kernel::launch_unchecked::<E, R>(
                client,
                count,
                dim,
                cand.arg(),
                ev_peaks.arg(),
                ev_w.arg(),
                meta.arg(),
                spec.arg(),
                cand_ev.arg(),
                m_u32,
                p_u32,
                work_max,
                h_cap_max,
                first,
                chunk,
                span,
            );
        }
        // Submit this chunk as its own GPU job (one flush per launch; not a
        // read, moves no counters), bounding the work of one GPU job as
        // documented above.
        crate::backend::check_launches(cand_ev.device())?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// `ms2_formula_features`: lane per `(b, m)`
// ---------------------------------------------------------------------------

/// Lane per `(b, m)` of [`formula_features`]; a copy of
/// [`crate::models::ms2::formula_evidence::formula_features_lane`] over
/// `Array`s.
///
/// Words `0..10` read `log_table[count]` with the same out-of-range guard
/// as `count_features`, so the bits equal that kernel's; words `10`/`11`
/// copy the formula-window kernel's parent-mass and tolerance arithmetic
/// (cited there); words `12..15` read `cand_ev` straight through a float to
/// `u32` cast for the `log_table` index. Padding slots write exact `0` in
/// all 16 words. All other statements are shared verbatim with the twin
/// (modulo the spelling deltas above).
#[allow(clippy::too_many_arguments)]
#[allow(unused_assignments)]
#[cube(launch_unchecked)]
fn ms2_formula_features_kernel<F: Float + CubeElement>(
    cand: &Array<u32>,
    cand_ev: &Array<F>,
    meta: &Array<u32>,
    log_table: &Array<F>,
    out: &mut Array<F>,
    m_dim: u32,
    lanes: usize,
    span: usize,
) {
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    for pos in start..end {
        let lane = pos as u32;
        let b = lane / m_dim;
        let m = lane % m_dim;
        let cb = (b * m_dim + m) * 13u32;
        let eb = (b * m_dim + m) * 4u32;
        let ob = (b * m_dim + m) * 16u32;
        let flag = cand[(cb + 11u32) as usize];
        let mut pad: u32 = 0u32;
        if flag == 0u32 {
            pad = 1u32;
        }
        // Words 0..10: `log_table[count]`, as `count_features` writes them.
        let mut e: u32 = 0u32;
        while e < 10u32 {
            let count = cand[(cb + e) as usize];
            let mut ok: u32 = 0u32;
            if count < 1024u32 {
                ok = 1u32;
            }
            let mut safe: usize = 0usize;
            if ok == 1u32 {
                safe = count as usize;
            }
            let mut v = log_table[safe];
            if ok == 0u32 {
                v = F::new(0.0_f32);
            }
            if pad == 1u32 {
                v = F::new(0.0_f32);
            }
            out[(ob + e) as usize] = v;
            e += 1u32;
        }
        // Precursor block: the parent mass the formula-window kernel derives
        // from `meta` (same adduct arithmetic over the net hydrogen shift
        // `m_H − m_e = 1007276`, cited from `ms2_formula_window_kernel`).
        let prec = meta[(b * 8u32 + 1u32) as usize];
        let unc = meta[(b * 8u32 + 2u32) as usize];
        let adduct_id = meta[(b * 8u32 + 3u32) as usize];
        let ppm_pre = meta[(b * 8u32 + 5u32) as usize];
        let mut m_p: u32 = 0u32;
        let mut parent_ok: u32 = 0u32;
        if adduct_id == 1u32 && prec >= 1007276u32 {
            m_p = prec - 1007276u32;
            parent_ok = 1u32;
        }
        if adduct_id == 2u32 && prec <= 4294967295u32 - 1007276u32 {
            m_p = prec + 1007276u32;
            parent_ok = 1u32;
        }
        // `w` is the precursor tolerance at the precursor m/z (same `u32`
        // tolerance algorithm, `meta` word 5 as ppm tenths) plus the
        // precursor uncertainty, saturating, at least 1.
        let tol_pre = ms2_fe_tolerance_u32(prec, ppm_pre);
        let mut w = ms2_fe_sat_add(tol_pre, unc);
        if w == 0u32 {
            w = 1u32;
        }
        let m_c = cand[(cb + 10u32) as usize];
        let mut d: u32 = 0u32;
        if m_c >= m_p {
            d = m_c - m_p;
        } else {
            d = m_p - m_c;
        }
        // The minimum proves the product below fits `u32`.
        let mut cap_w = w;
        if cap_w > 1073741823u32 {
            cap_w = 1073741823u32;
        }
        let lim = cap_w * 4u32;
        if d > lim {
            d = lim;
        }
        let abs_v = F::cast_from(d) / F::cast_from(w);
        let mut av10 = abs_v;
        let mut av11 = abs_v;
        if m_c < m_p {
            av11 = F::new(0.0_f32) - abs_v;
        }
        if parent_ok == 0u32 || unc == 4294967295u32 || pad == 1u32 {
            av10 = F::new(0.0_f32);
            av11 = F::new(0.0_f32);
        }
        out[(ob + 10u32) as usize] = av10;
        out[(ob + 11u32) as usize] = av11;
        // Words 12..15 straight from `cand_ev`.
        let expl = cand_ev[eb as usize];
        let wt = cand_ev[(eb + 1u32) as usize];
        let nev = cand_ev[(eb + 2u32) as usize];
        let comp = cand_ev[(eb + 3u32) as usize];
        let mut f12 = F::new(0.0_f32);
        if nev != F::new(0.0_f32) {
            f12 = expl / nev;
        }
        let expl_u = u32::cast_from(expl);
        // Clamp the index, load unconditionally, then select (no
        // global-buffer load behind a branch).
        let mut expl_safe: usize = 0usize;
        if expl_u < 1024u32 {
            expl_safe = expl_u as usize;
        }
        let expl_v = log_table[expl_safe];
        let mut f14 = F::new(0.0_f32);
        if expl_u < 1024u32 {
            f14 = expl_v;
        }
        let mut o12 = f12;
        let mut o13 = wt;
        let mut o14 = f14;
        let mut o15 = comp;
        if pad == 1u32 {
            o12 = F::new(0.0_f32);
            o13 = F::new(0.0_f32);
            o14 = F::new(0.0_f32);
            o15 = F::new(0.0_f32);
        }
        out[(ob + 12u32) as usize] = o12;
        out[(ob + 13u32) as usize] = o13;
        out[(ob + 14u32) as usize] = o14;
        out[(ob + 15u32) as usize] = o15;
    }
}

/// Evidence features of one `(B, M)` bucket, lane per `(b, m)`.
///
/// `cand [B, M, 13]` holds the scored candidates, `cand_ev [B, M, 4]` their
/// evidence of [`formula_evidence`], `meta [B, 8]` the spectrum words and
/// `log_table [1024]` the resident `ln(1 + count)` table (the same table
/// `count_features` uses). Writes `out [B, M, 16]`: `0..10`
/// `log_table[count]` (the same bits `count_features` writes), `10`
/// `abs_res`, `11` `signed_res`, `12` explained count over evidence count,
/// `13` explained weight, `14` `log_table[expl_count]`, `15` complete; exact
/// `0` in every word of a padding slot. Exactly 1 launch, one lane per
/// `(b, m)`.
pub fn formula_features<R: Runtime, E: FloatElem>(
    cand: &IdTensor<R>,
    cand_ev: &Tensor<R, E>,
    meta: &IdTensor<R>,
    log_table: &Tensor<R, E>,
    out: &mut Tensor<R, E>,
) -> Result<()> {
    if cand.shape().rank() != 3
        || cand_ev.shape().rank() != 3
        || meta.shape().rank() != 2
        || log_table.shape().rank() != 1
        || out.shape().rank() != 3
    {
        return Err(Error::shape(format!(
            "formula_features needs cand [B, M, 13], cand_ev [B, M, 4], meta [B, 8], log_table [1024] and out [B, M, 16], got {} and {} and {} and {} and {}",
            cand.shape(),
            cand_ev.shape(),
            meta.shape(),
            log_table.shape(),
            out.shape()
        )));
    }
    let batch = cand.shape().dim(0);
    let m = cand.shape().dim(1);
    if cand.shape().dim(2) != 13 || cand_ev.shape().dim(2) != 4 || out.shape().dim(2) != 16 {
        return Err(Error::shape(format!(
            "formula_features needs last dimensions 13, 4 and 16, got {} and {} and {}",
            cand.shape(),
            cand_ev.shape(),
            out.shape()
        )));
    }
    let want_cand: &[usize] = &[batch, m, 13];
    let want_ev: &[usize] = &[batch, m, 4];
    let want_meta: &[usize] = &[batch, 8];
    let want_out: &[usize] = &[batch, m, 16];
    if cand.shape().dims() != want_cand
        || cand_ev.shape().dims() != want_ev
        || meta.shape().dims() != want_meta
        || out.shape().dims() != want_out
    {
        return Err(Error::shape(format!(
            "formula_features needs cand [{batch}, {m}, 13], cand_ev [{batch}, {m}, 4], meta [{batch}, 8] and out [{batch}, {m}, 16], got {} and {} and {} and {}",
            cand.shape(),
            cand_ev.shape(),
            meta.shape(),
            out.shape()
        )));
    }
    if log_table.len() != 1024 {
        return Err(Error::shape(format!(
            "formula_features needs log_table [1024], got {}",
            log_table.shape()
        )));
    }
    let lanes = batch.checked_mul(m).ok_or_else(|| {
        Error::shape(format!(
            "formula_features: batch {batch} times {m} lanes overflows usize"
        ))
    })?;
    if lanes == 0 {
        return Ok(());
    }
    check_device_len("cand", cand.len())?;
    check_device_len("cand_ev", cand_ev.len())?;
    check_device_len("meta", meta.len())?;
    check_device_len("log_table", log_table.len())?;
    check_device_len("out", out.len())?;
    let m_u32 = check_device_scalar("formula_features M", m)?;
    // `lanes` fits `u32`, so every `pos < lanes` narrows to a `u32` lane.
    check_device_scalar("formula_features B * M lanes", lanes)?;
    let client = cand.client();
    let (count, dim, span) = launch_1d_spans(client, lanes, 16);
    unsafe {
        ms2_formula_features_kernel::launch_unchecked::<E, R>(
            client,
            count,
            dim,
            cand.arg(),
            cand_ev.arg(),
            meta.arg(),
            log_table.arg(),
            out.arg(),
            m_u32,
            lanes,
            span,
        );
    }
    Ok(())
}
