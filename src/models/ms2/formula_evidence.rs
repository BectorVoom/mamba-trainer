//! Host reference twin of the formula-evidence stage (task E1).
//!
//! Pure host Rust with integer masses only: no tensors, no kernels, no neural
//! code. [`evidence_peaks_lane`], [`formula_evidence_lane`] and
//! [`formula_features_lane`] are the line-for-line twins of the `#[cube]`
//! kernels in `crate::tensor::ops::ms2_formula_evidence`, written first in the
//! kernel-expressible form (full buffers with explicit indices, `u32` loop
//! counters from literals, no early `return` inside loops), so both stay
//! identical. [`evidence_peaks`], [`formula_evidence`] and
//! [`formula_features`] are host wrappers that loop the lanes over whole
//! batches, so every host test exercises the lane code.
//!
//! * [`evidence_peaks_lane`] — one spectrum: the `P` most intense eligible
//!   kept peaks in `(intensity, position)` order, with normalised weights.
//! * [`formula_evidence_lane`] — one `(b, m)` slot: the explained-peak mask,
//!   weight, evidence count and completion flag of a candidate.
//! * [`formula_features_lane`] — one `(b, m)` slot: the 16 `Evidence` feature
//!   values of architecture §1.6.
//! * [`EVIDENCE_PEAKS`] — the evidence-peak capacity `P = 32`.

use super::chem::{ELEMENTS, ELECTRON_MASS, ELECTRON_RESIDUAL_NDA, HYDROGEN};

/// Evidence peaks kept per spectrum (`P`): one `u32` bit per slot fits the
/// explained-peak mask of [`formula_evidence_lane`].
pub const EVIDENCE_PEAKS: usize = 32;

/// Hydrogen mass split `m_H = 1,000,000 + 7,825` (task E4F item 1).
pub const FE_H_MOD: u32 = 7_825;
/// Unit modulus of the residue argument (task E4F item 1).
pub const FE_UNIT: u32 = 1_000_000;

/// Wrap count `s_max = (7,825 h_cap + 2 tol) / 1,000,000` (integer
/// division) for the residue argument of item 1.
///
/// Computed in `u64` (no overflow for any `u32` inputs); the lane only calls
/// it with `h_cap <= 65,535` and `tol <= m_H`, where `7,825 h_cap + 2 tol <
/// 2^30`.
pub fn hydrogen_s_max(h_cap: u32, tol: u32) -> u32 {
    ((u64::from(h_cap) * 7_825 + 2 * u64::from(tol)) / 1_000_000) as u32
}

/// Per-`(visit, peak)` hydrogen trial bound of item 1:
/// `min(h_cap + 1, (s_max + 1) * (2 tol / 7,825 + 2))`.
///
/// Each wrapped range `s` spans `h` with `7,825 h` in
/// `[Rm + 1,000,000 s - 2 tol, Rm + 1,000,000 s]`, an interval of length `2
/// tol`, hence at most `2 tol / 7,825 + 2` integers; there are `s_max + 1`
/// ranges. The plain range has `h_cap + 1` values. The lane tries at most
/// this many hydrogen counts per (visit, peak) (with multiplicity when
/// ranges overlap).
pub fn hydrogen_trials_bound(h_cap: u32, tol: u32) -> u64 {
    let s_max = u64::from(hydrogen_s_max(h_cap, tol));
    let per_s = (2 * u64::from(tol)) / 7_825 + 2;
    let wrapped = (s_max + 1) * per_s;
    wrapped.min(u64::from(h_cap) + 1)
}

/// Unclamped wrapped-range trial count `(s_max + 1) * (2 tol / 7,825 + 2)`
/// (item 1 rule): the lane uses the wrapped ranges exactly when this does
/// not exceed `h_cap + 1`.
pub fn hydrogen_wrapped_trials(h_cap: u32, tol: u32) -> u64 {
    let s_max = u64::from(hydrogen_s_max(h_cap, tol));
    let per_s = (2 * u64::from(tol)) / 7_825 + 2;
    (s_max + 1) * per_s
}

/// Heavy elements in `ELEMENTS` order (hydrogen skipped): the mixed-radix
/// digits of the sub-composition walk, carbon least significant. Copy of the
/// order named `HEAVY` in `super::ion` (which is private there), cited so the
/// two cannot drift without review.
const HEAVY_ELEM: [usize; 9] = [0, 2, 3, 4, 5, 6, 7, 8, 9];

/// Full-buffer lane of `ms2_evidence_peaks`: the kernel twin for spectrum `b`.
///
/// `kept` is `[B, N, 3]` flat kept peaks (raw index, m/z, reverse), `kept_f`
/// is `[B, N, 2]` flat (relative intensity in column 0, valid flag in column
/// 1), `meta` is `[B, 8]` flat spectrum words (peak count, precursor,
/// precursor uncertainty, adduct, fragment-tolerance ppm tenths, precursor
/// tolerance, id lo/hi) and `spec` is `[B, 2]` flat per-spectrum words (m/z
/// uncertainty `U`, reserved). Writes the `[B, P, 4]` flat `ev_peaks` row
/// (kept position `p`, target mass `t`, fragment tolerance `tol_p`, valid
/// flag `1`; padding is `u32::MAX, 0, 0, 0`) and the `[B, P]` flat `ev_w`
/// row (intensity over the slot-ordered sum of the selected intensities; `0`
/// in padding, all `0` when the sum is not positive). Every element of both
/// rows is written.
///
/// A kept position `p` is eligible when `p < meta[b, 0]` (clamped to `N`),
/// its m/z is non-zero, the adduct id is 1 or 2, `t` fits `u32`,
/// `U != u32::MAX` and `tol_p.saturating_add(U) <= m_H`. Selection is `P`
/// passes over the `N` kept positions: pass `s` takes the maximum among the
/// eligible peaks strictly after the previous pick in the strict total order
/// (intensity descending, then position ascending), so no taken set is needed.
/// A NaN intensity is never selected: every float test is an ordering
/// comparison (`>`, `<`, `<=`, `==` between distinct values), which NaN
/// fails, including the `float_ok` gate that keeps NaN from winning an empty
/// best slot by default.
///
/// Selection applies the candidate-independent part of the scope test,
/// `tol_p.saturating_add(U) <= m_H`; whether a selected peak is in scope
/// for a particular candidate (`tol_p + U + E_ion(c) <= m_H`) is decided in
/// [`formula_evidence_lane`], where an out-of-scope peak is unexplained for
/// that candidate and still counts in `n_ev`.
#[allow(clippy::too_many_arguments)]
pub fn evidence_peaks_lane(
    kept: &[u32],
    kept_f: &[f32],
    meta: &[u32],
    spec: &[u32],
    b: u32,
    n: u32,
    p_dim: u32,
    ev_peaks: &mut [u32],
    ev_w: &mut [f32],
) {
    // Full-buffer addresses in u32 exactly as the kernel does (cast to
    // `usize` only at the index expression).
    let meta_base: u32 = b * 8;
    let spec_base: u32 = b * 2;
    let peak_count = meta[meta_base as usize];
    let mut count = peak_count;
    if count > n {
        count = n;
    }
    let adduct_id = meta[(meta_base + 3) as usize];
    let ppm = meta[(meta_base + 4) as usize];
    let u = spec[spec_base as usize];
    let known = adduct_id == 1 || adduct_id == 2;
    let m_h = ELEMENTS[HYDROGEN].mass;
    let mut prev_int = 0.0f32;
    let mut prev_pos = 0u32;
    let mut s: u32 = 0;
    while s < p_dim {
        let mut have = false;
        let mut best_int = 0.0f32;
        let mut best_pos = 0u32;
        let mut best_mz = 0u32;
        let mut p: u32 = 0;
        while p < n {
            // Unconditional loads: every index below is in range.
            let mz = kept[((b * n + p) * 3 + 1) as usize];
            let inten = kept_f[((b * n + p) * 2) as usize];
            // `t_ok` and `tol_p` by the same `u32` algorithm as
            // `ion_assign_lane` (adduct rule, `tolerance_u32` split); the
            // target mass itself is only formed for the winning row below.
            let t_ok = (adduct_id == 1 && mz <= u32::MAX - ELECTRON_MASS)
                || (adduct_id == 2 && mz >= ELECTRON_MASS);
            let hi = mz / 10_000;
            let lo = mz % 10_000;
            let q = hi.wrapping_mul(ppm);
            let tol_p =
                q / 1000 + ((q % 1000) * 10_000 + lo.wrapping_mul(ppm)) / 10_000_000;
            let half = tol_p.saturating_add(u);
            let base_ok =
                p < count && mz != 0 && known && t_ok && u != u32::MAX && half <= m_h;
            // NaN gate by comparison only: NaN fails both arms, every other
            // float passes exactly one.
            let float_ok = inten > 0.0 || inten <= 0.0;
            let after =
                s == 0 || inten < prev_int || (inten == prev_int && p > prev_pos);
            let beats =
                !have || inten > best_int || (inten == best_int && p < best_pos);
            if base_ok && float_ok && after && beats {
                have = true;
                best_int = inten;
                best_pos = p;
                best_mz = mz;
            }
            p += 1;
        }
        let row: u32 = (b * p_dim + s) * 4;
        if have {
            // Same `t` / `tol_p` formation as above, for the winning m/z.
            let mut t = 0u32;
            if adduct_id == 1 {
                t = best_mz + ELECTRON_MASS;
            }
            if adduct_id == 2 {
                t = best_mz - ELECTRON_MASS;
            }
            let hi = best_mz / 10_000;
            let lo = best_mz % 10_000;
            let q = hi.wrapping_mul(ppm);
            let tol_p =
                q / 1000 + ((q % 1000) * 10_000 + lo.wrapping_mul(ppm)) / 10_000_000;
            ev_peaks[row as usize] = best_pos;
            ev_peaks[(row + 1) as usize] = t;
            ev_peaks[(row + 2) as usize] = tol_p;
            ev_peaks[(row + 3) as usize] = 1;
            prev_int = best_int;
            prev_pos = best_pos;
        } else {
            ev_peaks[row as usize] = u32::MAX;
            ev_peaks[(row + 1) as usize] = 0;
            ev_peaks[(row + 2) as usize] = 0;
            ev_peaks[(row + 3) as usize] = 0;
        }
        s += 1;
    }
    // Weights in two passes that re-read the selected rows (as the kernel
    // does): the sum in slot order, then each slot's share.
    let mut sum = 0.0f32;
    let mut s2: u32 = 0;
    while s2 < p_dim {
        let wpos = ev_peaks[((b * p_dim + s2) * 4) as usize];
        let mut safe = 0u32;
        if wpos < n {
            safe = wpos;
        }
        let inten = kept_f[((b * n + safe) * 2) as usize];
        if ev_peaks[((b * p_dim + s2) * 4 + 3) as usize] == 1 {
            sum += inten;
        }
        s2 += 1;
    }
    let mut s3: u32 = 0;
    while s3 < p_dim {
        let rpos = ev_peaks[((b * p_dim + s3) * 4) as usize];
        let mut rsafe = 0u32;
        if rpos < n {
            rsafe = rpos;
        }
        let inten = kept_f[((b * n + rsafe) * 2) as usize];
        let mut wv = 0.0f32;
        if ev_peaks[((b * p_dim + s3) * 4 + 3) as usize] == 1 && sum > 0.0 {
            wv = inten / sum;
        }
        ev_w[(b * p_dim + s3) as usize] = wv;
        s3 += 1;
    }
}

/// Evidence peaks of a whole batch: loops [`evidence_peaks_lane`] over the
/// spectra. `kept` is `batch * n * 3` flat, `kept_f` `batch * n * 2` flat,
/// `meta` `batch * 8` flat, `spec` `batch * 2` flat. Returns
/// (`ev_peaks` flat `batch * p * 4`, `ev_w` flat `batch * p`).
///
/// Preconditions (panics with a message when violated, mirroring the
/// kernel wrapper's [`Error::Shape`](crate::error::Error::Shape) refusals):
/// `n >= 1`, `p >= 1`, `p <= 32`.
pub fn evidence_peaks(
    kept: &[u32],
    kept_f: &[f32],
    meta: &[u32],
    spec: &[u32],
    batch: usize,
    n: usize,
    p: usize,
) -> (Vec<u32>, Vec<f32>) {
    assert!(n >= 1, "evidence_peaks needs N >= 1, got N = {n}");
    assert!(p >= 1, "evidence_peaks needs P >= 1, got P = {p}");
    assert!(
        p <= 32,
        "evidence_peaks needs P <= 32 for the u32 mask, got P = {p}"
    );
    let mut peaks = vec![0u32; batch * p * 4];
    let mut weights = vec![0.0f32; batch * p];
    for b in 0..batch {
        evidence_peaks_lane(
            kept,
            kept_f,
            meta,
            spec,
            b as u32,
            n as u32,
            p as u32,
            &mut peaks,
            &mut weights,
        );
    }
    (peaks, weights)
}

/// Full-buffer lane of `ms2_formula_evidence`: the kernel twin for slot
/// `(b, m)`.
///
/// `cand` is `[B, M, 13]` flat candidates (10 counts in `ELEMENTS` order,
/// integer mass, flag, source id), `ev_peaks`/`ev_w` the evidence peaks of
/// [`evidence_peaks_lane`], `meta`/`spec` as there. Writes the `[B, M, 4]`
/// flat `cand_ev` row: explained-peak count (an exact small integer as
/// float), explained weight (sum of `ev_w` over explained peaks in ascending
/// slot order, each weight loaded unconditionally), number of valid evidence
/// peaks of the spectrum (float), complete flag `1.0`/`0.0`. A padding
/// candidate slot (`cand` flag word 11 `== 0`) writes `0, 0, 0, 0`. Every
/// element of the row is written.
///
/// Predicate (task E4). Notation as in `ion_lane_visit` / `ion_assign_lane`
/// (`super::ion`): candidate counts `c`, heavy slots in `HEAVY` order with
/// slot 0 = carbon (mass `12,000,000`, residual `0`), hydrogen mass
/// `m_H = 1,007,825` with residual `33`, electron residual `421`, target
/// `t`, fragment tolerance `tol`, uncertainty `U`, `E_ion(c)`,
/// `half_p = sat(tol + U + E_ion)`, hydrogen cap
/// `h_cap = min(c[H] + h_pos + 2, 65535)` with `h_pos = 1` for adduct 1 else
/// `0`. An evidence peak is explained by `c` exactly when there are a carbon
/// count `n` and a non-carbon heavy vector `u'` (digits of slots 1..8, each
/// `<= c`), not both zero, and a hydrogen count `h`, with `0 <= n <= c[C]`,
/// `0 <= h <= h_cap`, `cand = m(u') + 12,000,000 n + m_H h` fitting `u32`,
/// `r = |t - cand| <= tol` and `bound(u', h) <= tol - r`, where
/// `bound(u', h) = sat(ceil((res(u') + 33 h + 421) / 1000) + U)` — provided
/// the peak is valid and `half_p <= m_H` (otherwise unexplained).
///
/// Proof sketch. First, acceptance implies the lane's window and 3-slot cap
/// never cut it, so explained-ness here equals `ion_assign(parent = c, ...)`
/// reporting `accepted >= 1` over the same visits: an accepted hypothesis
/// has `cand` within `tol <= half_p` of `t`, hence inside the lane's window
/// `[t - half_p, t + half_p]`, hence inside its hydrogen interval; under
/// `half_p <= m_H` that interval holds at most three integers, so the lane's
/// 3-slot cap never cuts it; carbon contributes no residual, so the bound
/// here equals the lane's own bound term for term. Second, one multiple of
/// `12,000,000` per interval: for a fixed `u'` (mass `m'`, residual sum
/// `res'`) and peak, and a fixed `h` with `bound(u', h) <= tol`, writing
/// `delta = tol - bound` and `base = m' + m_H h`, a carbon count is accepted
/// exactly when `12,000,000 n` lies in `[t - base - delta, t - base +
/// delta]`; because `2 delta <= 2 m_H < 12,000,000` that interval holds at
/// most one multiple, and with `top = t + delta` (saturating) there is none
/// when `top < base`, otherwise `n1 = (top - base) / 12,000,000` is the only
/// candidate, accepted exactly when `12,000,000 n1 + base + delta >= t`
/// (tested as `12,000,000 n1 + base >= t.saturating_sub(delta)`, without any
/// wrapping subtraction), `n1 <= c[C]`, and not (`n1 == 0` with `u'` all
/// zero). Third, the wrapped fast hydrogen ranges (task E4F item 1,
/// generalising the old `7,825 h_cap + 2 tol < 1,000,000` fast range, which
/// is the `s_max = 0` case): with `X = t + tol - m'` (no hypothesis when `t
/// + tol < m'`, i.e. `m' - t > tol`), `Q = X / 1,000,000` and `Rm = X %
/// 1,000,000`, an accepted `(n, h)` has `cand - m' = 1,000,000 (12 n + h) +
/// 7,825 h` (since `12,000,000 = 12 * 1,000,000` and `m_H = 1,000,000 +
/// 7,825`) and `X - (cand - m') = D` with `0 <= D <= 2 tol`. Let `S = 7,825
/// h + D`; then `0 <= S <= 7,825 h_cap + 2 tol` and `S = X - 1,000,000 (12 n
/// + h)`, so `S ≡ X ≡ Rm (mod 1,000,000)` and `S = Rm + 1,000,000 s` for
/// some wrap count `s` in `0 ..= s_max` with `s_max = (7,825 h_cap + 2 tol)
/// / 1,000,000` (integer division; also `s <= Q` because `12 n + h = Q - s
/// >= 0`). For each `s`, `7,825 h` lies in `[Rm + 1,000,000 s - 2 tol, Rm +
/// 1,000,000 s]`, i.e. `h` in `ceil(max(Rm + 1,000,000 s - 2 tol, 0) /
/// 7,825) ..= min((Rm + 1,000,000 s) / 7,825, h_cap)`. The union over `s`
/// of these ranges contains every accepted `h`: an accepted `h` yields its
/// `S`, hence its `s = (S - Rm) / 1,000,000`, and lies in range `s` by
/// construction. Ranges of different `s` cannot miss an accepted `h` at
/// their boundaries: each accepted `h` belongs to at least the range of its
/// own `s` (the argument above is per-`h`, not per-boundary), and ranges may
/// overlap — when `2 tol >= 1,000,000 - 7,825` consecutive ranges share
/// values, trying an `h` twice is harmless because the per-`h` test is exact
/// and idempotent (it only sets an already-set mask bit). The lane uses the
/// wrapped ranges when `(s_max + 1) * (2 tol / 7,825 + 2) <= h_cap + 1`
/// (each range holds at most `2 tol / 7,825 + 2` integers: an interval of
/// length `2 tol` sampled at step `7,825` has at most `floor(2 tol / 7,825)
/// + 1` interior points plus one for each ceiling edge), else the plain
/// range `0 ..= h_cap`; either way each tried `h` goes through the same
/// exact per-`h` test as the slow path (`0 ..= h_cap`), and the narrowing
/// decides nothing by itself. The number of trials per (visit, peak) is at
/// most `min(h_cap + 1, (s_max + 1)(2 tol / 7,825 + 2))`.
///
/// Walk and budget. The lane enumerates the non-carbon heavy vectors `u'`
/// by mixed-radix index `j = 0, 1, 2, ...` over slots 1..8 (slot 1 least
/// significant; `j = 0` is the all-zero vector, for which only `n >= 1`
/// counts), at most `W` of them (`work_max`, the `formula_evidence_work_max`
/// budget; the default stays 2,048). `J` is the product over slots 1..8 of
/// `(c[e] + 1)` (guarded, as `lane_visits_u32` guards it); `visits = min(J,
/// W)`; `complete = (J <= W)`. The lane stops early once every valid
/// evidence peak is explained. Relation to `ion_assign`'s order: its visit
/// index is `k = n + (c[C] + 1) j`, so the first `V` values of `j` are
/// exactly its visits `1 ..= V * (c[C] + 1) - 1`: with `V = min(J, W)`,
/// explained-ness here equals `ion_assign(parent = c, ..., work_max = V *
/// (c[C] + 1) - 1).accepted >= 1` whenever that product fits `u32` and is at
/// least 1.
///
/// Worst-case cost per lane: at most `W * P * trials` hydrogen trials, with
/// `trials = min(h_cap + 1, (s_max + 1)(2 tol / 7,825 + 2))` per (visit,
/// peak) of item 1 (`h_cap` the lane's clamped cap, `tol` the peak's
/// fragment tolerance). The dispatch sizing of
/// `crate::tensor::ops::ms2_formula_evidence::formula_evidence` bounds the
/// launch by `work_max * P * trials_bound` with `trials_bound` from the
/// host-known `h_cap_max` / `tol_max`, and the lane enforces the sizing by
/// clamping its `h_cap` to `h_cap_max` (a correct bound never engages the
/// clamp).
///
/// The per-lane `h_cap_max` bound (task E4F item 2) is an upper bound of
/// `c[H] + 3` over every candidate the request can score (the lane's
/// `c[H] + h_pos + 2` with `h_pos <= 1`); the lane clamps its hydrogen cap
/// to it before sizing its ranges.
///
/// Selection (kernel 1) applies the candidate-independent part of the scope
/// test, `tol_p.saturating_add(U) <= m_H`; whether a selected peak is in
/// scope for a particular candidate (`tol_p + U + E_ion(c) <= m_H`) is
/// decided here, where an out-of-scope peak is unexplained for that
/// candidate and still counts in `n_ev`.
///
/// Preconditions: `p_dim >= 1`, `p_dim <= 32` (the explained set is one
/// `u32` mask); `work_max` is honoured as above (`0` gives zero visits and
/// `complete = 0`; the kernel wrapper refuses it with `Error::Config`).
/// `h_cap_max` clamps the lane's hydrogen cap (item 2); a correct bound
/// never engages the clamp.
#[allow(clippy::too_many_arguments)]
pub fn formula_evidence_lane(
    cand: &[u32],
    ev_peaks: &[u32],
    ev_w: &[f32],
    meta: &[u32],
    spec: &[u32],
    b: u32,
    m: u32,
    m_dim: u32,
    p_dim: u32,
    work_max: u32,
    h_cap_max: u32,
    cand_ev: &mut [f32],
) {
    formula_evidence_lane_inner(
        cand, ev_peaks, ev_w, meta, spec, b, m, m_dim, p_dim, work_max, h_cap_max,
        false, None, cand_ev,
    );
}

/// Slow-path twin of [`formula_evidence_lane`]: forces the `0 ..= h_cap`
/// hydrogen range for every (visit, peak) instead of the narrowed wrapped
/// ranges.
///
/// Test-only oracle for the fast path: on identical inputs its row equals
/// [`formula_evidence_lane`]'s row exactly (integer words bit-for-bit; the
/// weight is the same `f32` sum over the same explained set).
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn formula_evidence_lane_slow(
    cand: &[u32],
    ev_peaks: &[u32],
    ev_w: &[f32],
    meta: &[u32],
    spec: &[u32],
    b: u32,
    m: u32,
    m_dim: u32,
    p_dim: u32,
    work_max: u32,
    h_cap_max: u32,
    cand_ev: &mut [f32],
) {
    formula_evidence_lane_inner(
        cand, ev_peaks, ev_w, meta, spec, b, m, m_dim, p_dim, work_max, h_cap_max,
        true, None, cand_ev,
    );
}

/// Test-only counting variant of [`formula_evidence_lane_slow`]: the slow
/// range (`0 ..= h_cap`) with a hydrogen-trial counter return. Together
/// with [`formula_evidence_lane_trials`] this pins the physical-work change
/// of item 1 (slow count before, wrapped count after).
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn formula_evidence_lane_slow_trials(
    cand: &[u32],
    ev_peaks: &[u32],
    ev_w: &[f32],
    meta: &[u32],
    spec: &[u32],
    b: u32,
    m: u32,
    m_dim: u32,
    p_dim: u32,
    work_max: u32,
    h_cap_max: u32,
    cand_ev: &mut [f32],
    trials: &mut u64,
) {
    formula_evidence_lane_inner(
        cand, ev_peaks, ev_w, meta, spec, b, m, m_dim, p_dim, work_max, h_cap_max,
        true, Some(trials), cand_ev,
    );
}

/// Test-only counting variant of [`formula_evidence_lane`]: the hidden twin
/// with a hydrogen-trial counter return. Runs the same wrapped ranges as
/// [`formula_evidence_lane`] (never forced slow) and adds every tried `h`
/// (with multiplicity when wrapped ranges overlap) into `trials`.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn formula_evidence_lane_trials(
    cand: &[u32],
    ev_peaks: &[u32],
    ev_w: &[f32],
    meta: &[u32],
    spec: &[u32],
    b: u32,
    m: u32,
    m_dim: u32,
    p_dim: u32,
    work_max: u32,
    h_cap_max: u32,
    cand_ev: &mut [f32],
    trials: &mut u64,
) {
    formula_evidence_lane_inner(
        cand, ev_peaks, ev_w, meta, spec, b, m, m_dim, p_dim, work_max, h_cap_max,
        false, Some(trials), cand_ev,
    );
}

/// Shared lane body of [`formula_evidence_lane`],
/// [`formula_evidence_lane_slow`] and [`formula_evidence_lane_trials`]:
/// `force_slow` selects the `0 ..= h_cap` hydrogen range for every (visit,
/// peak); `trials` (when `Some`) counts every tried `h`.
#[allow(clippy::too_many_arguments)]
fn formula_evidence_lane_inner(
    cand: &[u32],
    ev_peaks: &[u32],
    ev_w: &[f32],
    meta: &[u32],
    spec: &[u32],
    b: u32,
    m: u32,
    m_dim: u32,
    p_dim: u32,
    work_max: u32,
    h_cap_max: u32,
    force_slow: bool,
    trials: Option<&mut u64>,
    cand_ev: &mut [f32],
) {
    // Carbon mass step, hydrogen mass and residuals, and the unit split
    // (`m_H = 1,000,000 + 7,825`): values cited from
    // `super::chem::ELEMENTS`, so the kernel's literals cannot drift without
    // review.
    const C_MASS: u32 = 12_000_000;
    const M_H: u32 = 1_007_825;
    const H_RES: u32 = 33;
    const E_RES: u32 = 421;
    const UNIT: u32 = 1_000_000;
    // Full-buffer addresses in u32 exactly as the kernel does (cast to
    // `usize` only at the index expression).
    let cand_base: u32 = (b * m_dim + m) * 13;
    let ob: u32 = (b * m_dim + m) * 4;
    let mut trial_total: u64 = 0;
    // Every output element is written: zero the row first.
    cand_ev[ob as usize] = 0.0;
    cand_ev[(ob + 1) as usize] = 0.0;
    cand_ev[(ob + 2) as usize] = 0.0;
    cand_ev[(ob + 3) as usize] = 0.0;
    let flag = cand[(cand_base + 11) as usize];
    if flag != 0 {
        let mut n_ev: u32 = 0;
        let mut sv: u32 = 0;
        while sv < p_dim {
            if ev_peaks[((b * p_dim + sv) * 4 + 3) as usize] == 1 {
                n_ev += 1;
            }
            sv += 1;
        }
        // Non-carbon radices `c[e] + 1` over slots 1..8 in HEAVY order.
        let mut radices: [u32; 8] = [1; 8];
        let mut rj: u32 = 0;
        while rj < 8 {
            let ce = HEAVY_ELEM[(rj + 1) as usize];
            radices[rj as usize] = cand[(cand_base + ce as u32) as usize].wrapping_add(1);
            rj += 1;
        }
        // Guarded non-carbon radix product `J` against `W`: each multiply
        // runs only when `partial <= work_max / r` proves it fits and stays
        // within budget. `visits = min(J, W)`; `complete = (J <= W)`. A zero
        // radix (a `u32::MAX` count) rides as 1 and cuts the walk; a zero
        // budget visits nothing and is incomplete.
        let mut partial: u32 = 1;
        let mut cut: u32 = 0;
        let mut gi: u32 = 0;
        while gi < 8 {
            let mut r = radices[gi as usize];
            if r == 0 {
                r = 1;
                cut = 1;
            }
            if cut == 0 && r > 1 {
                if partial > work_max / r {
                    cut = 1;
                } else {
                    partial = partial.wrapping_mul(r);
                }
            }
            gi += 1;
        }
        let mut visits = partial;
        if cut == 1 {
            visits = work_max;
        }
        if work_max == 0 {
            visits = 0;
            cut = 1;
        }
        let complete: f32 = if cut == 0 { 1.0 } else { 0.0 };
        let adduct_id = meta[(b * 8 + 3) as usize];
        let u = spec[(b * 2) as usize];
        // `E_ion` exactly as `ion_assign_lane`: parent residuals plus three
        // hydrogens plus the electron residual, ceiled from nano-dalton.
        let mut nda = 0u32;
        let mut hs: u32 = 0;
        while hs < 9 {
            nda = nda.wrapping_add(
                cand[(cand_base + HEAVY_ELEM[hs as usize] as u32) as usize]
                    .wrapping_mul(ELEMENTS[HEAVY_ELEM[hs as usize]].residual_nda),
            );
            hs += 1;
        }
        nda = nda.wrapping_add(
            cand[(cand_base + HYDROGEN as u32) as usize]
                .wrapping_mul(ELEMENTS[HYDROGEN].residual_nda),
        );
        nda = nda.wrapping_add(3 * ELEMENTS[HYDROGEN].residual_nda + ELECTRON_RESIDUAL_NDA);
        let e_ion = nda / 1000 + u32::from(!nda.is_multiple_of(1000));
        let h_pos = if adduct_id == 1 { 1 } else { 0 };
        let h_cap = cand[(cand_base + HYDROGEN as u32) as usize]
            .wrapping_add(h_pos)
            .wrapping_add(2);
        // Item 2: clamp the lane's hydrogen cap to the host-known bound
        // (a correct bound never engages).
        let mut h_hi_abs: u32 = h_cap.min(u32::from(u16::MAX));
        if h_hi_abs > h_cap_max {
            h_hi_abs = h_cap_max;
        }
        let m_h = ELEMENTS[HYDROGEN].mass;
        let c_c = cand[cand_base as usize];
        let mut mask: u32 = 0;
        let mut explained: u32 = 0;
        let mut j: u32 = 0;
        let mut live: u32 = 1;
        if n_ev == 0 {
            live = 0;
        }
        while j < visits && live == 1 {
            // Mixed-radix digits of `j` over slots 1..8, slot 1 least
            // significant; `j = 0` is the all-zero vector.
            let mut digits: [u32; 8] = [0; 8];
            let mut tmp = j;
            let mut dj: u32 = 0;
            while dj < 8 {
                let r = radices[dj as usize];
                digits[dj as usize] = tmp % r;
                tmp /= r;
                dj += 1;
            }
            // Non-carbon heavy mass (division-guarded) and residual sum
            // (saturating: a saturated residual can never pass the per-`h`
            // `bound <= tol` test below). Residuals of slots 1..8 are all
            // non-zero, so both divisions are exact.
            let mut mprime: u32 = 0;
            let mut mass_ok = true;
            let mut resprime: u32 = 0;
            let mut zero_u = true;
            let mut di: u32 = 0;
            while di < 8 {
                let d = digits[di as usize];
                let ce = HEAVY_ELEM[(di + 1) as usize];
                let me = ELEMENTS[ce].mass;
                let re = ELEMENTS[ce].residual_nda;
                let fits = d <= (u32::MAX - mprime) / me;
                let go = mass_ok && fits;
                mass_ok = mass_ok && fits;
                if go {
                    mprime = mprime.wrapping_add(d.wrapping_mul(me));
                }
                if d > (u32::MAX - resprime) / re {
                    resprime = u32::MAX;
                } else {
                    resprime = resprime.wrapping_add(d.wrapping_mul(re));
                }
                if d != 0 {
                    zero_u = false;
                }
                di += 1;
            }
            let mut s: u32 = 0;
            while s < p_dim {
                let row: u32 = (b * p_dim + s) * 4;
                let t = ev_peaks[(row + 1) as usize];
                let tol_p = ev_peaks[(row + 2) as usize];
                let valid = ev_peaks[(row + 3) as usize];
                let half_p = tol_p.saturating_add(u).saturating_add(e_ion);
                let bit = 1u32 << s;
                if valid == 1 && half_p <= m_h && (mask & bit) == 0 && mass_ok {
                    // Hydrogen trial ranges for this (visit, peak): the slow
                    // `0 ..= h_hi_abs` range, or the wrapped ranges below when
                    // `(s_max + 1) * (2 tol / 7,825 + 2) <= h_hi_abs + 1`
                    // (identical result, proved in the lane doc). The
                    // narrowing decides nothing by itself: every tried `h`
                    // below goes through the same exact per-`h` test.
                    let mut trial_ranges: Vec<(u32, u32)> = Vec::new();
                    if !force_slow
                        && hydrogen_wrapped_trials(h_hi_abs, tol_p)
                            <= u64::from(h_hi_abs) + 1
                    {
                        if mprime > t && mprime - t > tol_p {
                            // `t + tol < mprime`: no hypothesis.
                        } else {
                            // `Rm = (t + tol - mprime) mod 1,000,000`
                            // from residues only (`X` itself can exceed
                            // `u32` for precursor-scale masses).
                            let rm = {
                                let tm = t % UNIT;
                                let lm = tol_p % UNIT;
                                let mm = mprime % UNIT;
                                let mut ssum = tm + lm;
                                if ssum >= UNIT {
                                    ssum -= UNIT;
                                }
                                if ssum >= mm {
                                    ssum - mm
                                } else {
                                    ssum + UNIT - mm
                                }
                            };
                            let s_max = hydrogen_s_max(h_hi_abs, tol_p);
                            let two_tol = u64::from(tol_p) * 2;
                            let mut sw: u32 = 0;
                            while sw <= s_max {
                                let base =
                                    u64::from(rm) + 1_000_000u64 * u64::from(sw);
                                let lo_need = base.saturating_sub(two_tol);
                                let h_lo = (lo_need.div_ceil(7_825)) as u32;
                                let mut h_hi = (base / 7_825) as u32;
                                if h_hi > h_hi_abs {
                                    h_hi = h_hi_abs;
                                }
                                if h_lo <= h_hi {
                                    trial_ranges.push((h_lo, h_hi));
                                }
                                sw = sw.wrapping_add(1);
                                if sw == 0 {
                                    break;
                                }
                            }
                        }
                    } else {
                        trial_ranges.push((0, h_hi_abs));
                    }
                    // Hydrogen trials over the fixed ranges above; `h`
                    // starts from a literal plus the computed start.
                    // `h_end <= h_hi_abs <= 65535`, so `h += 1` cannot wrap.
                    for (h_start, h_end) in trial_ranges {
                        let mut h: u32 = 0u32.wrapping_add(h_start);
                        while h <= h_end {
                            trial_total = trial_total.wrapping_add(1);
                            // `base = mprime + m_H * h` only under the guard
                            // that proves it fits `u32`.
                            if h <= (u32::MAX - mprime) / M_H {
                                let base = mprime.wrapping_add(h.wrapping_mul(M_H));
                                // `resprime + 33 h + 421` with overflow
                                // saturation (a saturated residual fails the
                                // `bound <= tol` test below).
                                let mut rok = true;
                                let mut rh = resprime;
                                if h > (u32::MAX - rh) / H_RES {
                                    rok = false;
                                } else {
                                    rh = rh.wrapping_add(h.wrapping_mul(H_RES));
                                }
                                if rok && rh > u32::MAX - E_RES {
                                    rok = false;
                                }
                                if rok {
                                    rh = rh.wrapping_add(E_RES);
                                }
                                let arith = rh / 1000 + u32::from(!rh.is_multiple_of(1000));
                                let bound = arith.saturating_add(u);
                                if rok && bound <= tol_p {
                                    // The only carbon multiple of the `2 delta`
                                    // interval (`2 delta <= 2 m_H <
                                    // 12,000,000`, proved in the lane doc).
                                    let delta = tol_p - bound;
                                    let top = t.saturating_add(delta);
                                    if top >= base {
                                        let n1 = (top - base) / C_MASS;
                                        if n1 <= c_c && n1 <= u32::MAX / C_MASS {
                                            // `n1 * C_MASS <= top - base`, so
                                            // both sums below fit `u32`.
                                            let lhs =
                                                n1.wrapping_mul(C_MASS).wrapping_add(base);
                                            // `lhs + delta >= t` without a
                                            // wrapping subtraction.
                                            let need = t.saturating_sub(delta);
                                            if lhs >= need && !(n1 == 0 && zero_u) {
                                                if (mask & bit) == 0 {
                                                    mask |= bit;
                                                    explained += 1;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            h = h.wrapping_add(1);
                        }
                    }
                }
                s += 1;
            }
            if explained == n_ev {
                live = 0;
            }
            j = j.wrapping_add(1);
        }
        // Weight sum over set bits in ascending slot order: each weight is
        // loaded unconditionally, then added under the bit test (same order
        // as the kernel).
        let mut wsum = 0.0f32;
        let mut sw: u32 = 0;
        while sw < p_dim {
            let wv = ev_w[(b * p_dim + sw) as usize];
            if (mask & (1u32 << sw)) != 0 {
                wsum += wv;
            }
            sw += 1;
        }
        cand_ev[ob as usize] = explained as f32;
        cand_ev[(ob + 1) as usize] = wsum;
        cand_ev[(ob + 2) as usize] = n_ev as f32;
        cand_ev[(ob + 3) as usize] = complete;
    }
    if let Some(tr) = trials {
        *tr = trial_total;
    }
}

/// Candidate evidence of a whole batch: loops [`formula_evidence_lane`] over
/// the `(b, m)` slots. `cand` is `batch * m * 13` flat, `ev_peaks`
/// `batch * p * 4` flat, `ev_w` `batch * p` flat, `meta` `batch * 8` flat,
/// `spec` `batch * 2` flat. Returns `cand_ev` flat `batch * m * 4`.
/// `h_cap_max` clamps every lane's hydrogen cap (item 2); pass a true bound
/// (e.g. `u32::MAX` when no bound is known).
///
/// Preconditions (panics with a message when violated, mirroring the
/// kernel wrapper's [`Error::Shape`](crate::error::Error::Shape) refusals):
/// `p >= 1`, `p <= 32`.
#[allow(clippy::too_many_arguments)]
pub fn formula_evidence(
    cand: &[u32],
    ev_peaks: &[u32],
    ev_w: &[f32],
    meta: &[u32],
    spec: &[u32],
    batch: usize,
    m: usize,
    p: usize,
    work_max: u32,
    h_cap_max: u32,
) -> Vec<f32> {
    assert!(p >= 1, "formula_evidence needs P >= 1, got P = {p}");
    assert!(
        p <= 32,
        "formula_evidence needs P <= 32 for the u32 mask, got P = {p}"
    );
    let mut out = vec![0.0f32; batch * m * 4];
    for b in 0..batch {
        for mm in 0..m {
            formula_evidence_lane(
                cand,
                ev_peaks,
                ev_w,
                meta,
                spec,
                b as u32,
                mm as u32,
                m as u32,
                p as u32,
                work_max,
                h_cap_max,
                &mut out,
            );
        }
    }
    out
}

/// Full-buffer lane of `ms2_formula_features`: the kernel twin for slot
/// `(b, m)`.
///
/// `cand` is `[B, M, 13]` flat, `cand_ev` the `[B, M, 4]` flat evidence of
/// [`formula_evidence_lane`], `meta` `[B, 8]` flat, `log_table` the resident
/// `[1024]` float table with `log_table[n] = ln(1 + n)`. Writes the
/// `[B, M, 16]` flat `out` row: `0..10` is `log_table[count_e]` (the same
/// bits `count_features` writes; exact `0` in a padding slot), `10`
/// `abs_res` and `11` `signed_res` from the candidate mass against the
/// neutral precursor mass, `12` explained count over evidence count (`0`
/// when there is none), `13` explained weight, `14`
/// `log_table[expl_count]`, `15` complete. A padding slot is exact `0` in all
/// 16. Every element of the row is written.
///
/// The precursor block copies the formula-window kernel's arithmetic: `m_p`
/// is the neutral parent mass from `meta` under the adduct rule
/// (`prec − (m_H − m_e)` for adduct 1, `prec + (m_H − m_e)` for adduct 2),
/// `w = tol_prec + precursor_uncertainty` by saturating addition with
/// `tol_prec` the precursor tolerance at the precursor m/z in integer units
/// (the `u32` tolerance algorithm, `meta` word 5 as ppm tenths),
/// `w = max(w, 1)`; `d = |m_c − m_p|` by comparison then subtraction,
/// `d = min(d, 4 * min(w, u32::MAX / 4))`, `abs_res = d / w` as floats,
/// `signed_res` negated when `m_c < m_p`. When the adduct is unknown, the
/// parent mass leaves `u32`, or `precursor_uncertainty == u32::MAX`, both
/// are `0`.
#[allow(clippy::too_many_arguments)]
pub fn formula_features_lane(
    cand: &[u32],
    cand_ev: &[f32],
    meta: &[u32],
    log_table: &[f32],
    b: u32,
    m: u32,
    m_dim: u32,
    out: &mut [f32],
) {
    // Full-buffer addresses in u32 exactly as the kernel does (cast to
    // `usize` only at the index expression).
    let cb: u32 = (b * m_dim + m) * 13;
    let eb: u32 = (b * m_dim + m) * 4;
    let ob: u32 = (b * m_dim + m) * 16;
    let flag = cand[(cb + 11) as usize];
    let pad = flag == 0;
    let mut e: u32 = 0;
    while e < 10 {
        let count = cand[(cb + e) as usize];
        let mut v = 0.0f32;
        if (count as usize) < log_table.len() {
            v = log_table[count as usize];
        }
        if pad {
            v = 0.0;
        }
        out[(ob + e) as usize] = v;
        e += 1;
    }
    // Precursor block (copies the formula-window kernel's parent-mass and
    // tolerance arithmetic, cited there).
    let meta_base: u32 = b * 8;
    let prec = meta[(meta_base + 1) as usize];
    let unc = meta[(meta_base + 2) as usize];
    let adduct_id = meta[(meta_base + 3) as usize];
    let ppm_pre = meta[(meta_base + 5) as usize];
    let h_net = ELEMENTS[HYDROGEN].mass - ELECTRON_MASS;
    let mut m_p = 0u32;
    let mut parent_ok = false;
    if adduct_id == 1 && prec >= h_net {
        m_p = prec - h_net;
        parent_ok = true;
    }
    if adduct_id == 2 && prec <= u32::MAX - h_net {
        m_p = prec + h_net;
        parent_ok = true;
    }
    let pre_ok = parent_ok && unc != u32::MAX;
    let hi = prec / 10_000;
    let lo = prec % 10_000;
    let q = hi.wrapping_mul(ppm_pre);
    let tol_pre = q / 1000 + ((q % 1000) * 10_000 + lo.wrapping_mul(ppm_pre)) / 10_000_000;
    let mut w = tol_pre.saturating_add(unc);
    if w == 0 {
        w = 1;
    }
    let m_c = cand[(cb + 10) as usize];
    let mut d = 0u32;
    if m_c >= m_p {
        d = m_c - m_p;
    }
    if m_c < m_p {
        d = m_p - m_c;
    }
    // The minimum proves the product below fits `u32` (rule: no product
    // before its guard).
    let mut cap_w = w;
    if cap_w > u32::MAX / 4 {
        cap_w = u32::MAX / 4;
    }
    let lim = cap_w * 4;
    if d > lim {
        d = lim;
    }
    let abs_res = (d as f32) / (w as f32);
    let mut signed_res = abs_res;
    if m_c < m_p {
        signed_res = -abs_res;
    }
    let mut a10 = abs_res;
    let mut a11 = signed_res;
    if !pre_ok || pad {
        a10 = 0.0;
        a11 = 0.0;
    }
    out[(ob + 10) as usize] = a10;
    out[(ob + 11) as usize] = a11;
    // Evidence features straight from `cand_ev`.
    let expl = cand_ev[eb as usize];
    let wt = cand_ev[(eb + 1) as usize];
    let nev = cand_ev[(eb + 2) as usize];
    let comp = cand_ev[(eb + 3) as usize];
    let mut f12 = 0.0f32;
    if nev != 0.0 {
        f12 = expl / nev;
    }
    let mut f14 = 0.0f32;
    // Clamp the index, load unconditionally, then select (same order as the
    // kernel).
    let expl_u = expl as u32;
    let expl_safe = if expl_u < 1024 { expl_u as usize } else { 0 };
    let expl_v = if expl_safe < log_table.len() {
        log_table[expl_safe]
    } else {
        0.0
    };
    if (expl as usize) < log_table.len() && expl_u < 1024 {
        f14 = expl_v;
    }
    let mut o12 = f12;
    let mut o13 = wt;
    let mut o14 = f14;
    let mut o15 = comp;
    if pad {
        o12 = 0.0;
        o13 = 0.0;
        o14 = 0.0;
        o15 = 0.0;
    }
    out[(ob + 12) as usize] = o12;
    out[(ob + 13) as usize] = o13;
    out[(ob + 14) as usize] = o14;
    out[(ob + 15) as usize] = o15;
}

/// Evidence features of a whole batch: loops [`formula_features_lane`] over
/// the `(b, m)` slots. `cand` is `batch * m * 13` flat, `cand_ev`
/// `batch * m * 4` flat, `meta` `batch * 8` flat. Returns the feature rows
/// flat `batch * m * 16`.
#[allow(clippy::too_many_arguments)]
pub fn formula_features(
    cand: &[u32],
    cand_ev: &[f32],
    meta: &[u32],
    log_table: &[f32],
    batch: usize,
    m: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; batch * m * 16];
    for b in 0..batch {
        for mm in 0..m {
            formula_features_lane(
                cand,
                cand_ev,
                meta,
                log_table,
                b as u32,
                mm as u32,
                m as u32,
                &mut out,
            );
        }
    }
    out
}
