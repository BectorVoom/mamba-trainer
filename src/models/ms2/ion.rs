//! Host reference twin of the fragment-ion assignment stage (architecture §2).
//!
//! Pure host Rust with integer masses only: no tensors, no kernels, no neural
//! code. [`ion_lane_visit`] is the reviewed visit-level twin (with
//! [`lane_visits_u32`] and [`lane_verdict_u32`]); the full-buffer lanes
//! [`ion_assign_lane`], [`label_mask_lane`], [`evidence_lane`] (with
//! [`evidence_match`]) are the line-for-line twins of the `#[cube]` kernels in
//! `crate::tensor::ops::ms2_ion`, which scalar-expand the visit's fixed arrays
//! (CubeCL 0.10 has no fixed-size local arrays as kernel values) with the
//! same arithmetic, and spell `wrapping_*`/`abs_diff`/`div_ceil` as plain
//! wrapping device ops. [`ion_assign`] and [`label_mask`] are host wrappers
//! that pack single-peak/single-row buffers and call the lanes, so every
//! host test exercises the lane code; [`evidence_status`] packs its decoded
//! inputs into the lane's buffer form and calls [`evidence_match`].
//!
//! * [`ion_lane_visit`] — one visit index in `u32` only (§2.1).
//! * [`lane_visits_u32`] — the guarded visit budget (§2.1).
//! * [`lane_verdict_u32`] — the overflow-free verdict rule (§5).
//! * [`ion_assign_lane`] — the `(b, f, p)` lane twin of `ms2_ion_assign`.
//! * [`ion_assign`] — host wrapper: hypotheses of one peak (spec §2.1).
//! * [`ion_labels`] — deduplicated ion label sets from recipe anchors (§2.3).
//! * [`label_mask_lane`] — the `(b, p)` lane twin of `ms2_ion_label_mask`.
//! * [`label_mask`] — host wrapper over the lane (§2.3).
//! * [`evidence_lane`] — the trajectory lane twin of `ms2_ion_evidence`.
//! * [`evidence_match`] — the shared match block of the lane and [`evidence_status`].
//! * [`evidence_status`] — packed mass-consistency evidence of a candidate (§2.4).
//! * [`mapping_is_supported`] and [`embedding_ion`] — the contract §4.3
//!   parent-to-ion mapping (P6.8).

use std::collections::BTreeSet;

use crate::error::{Error, Result};

use super::chem::{
    Composition, ELECTRON_MASS, ELECTRON_RESIDUAL_NDA, ELEMENTS, HYDROGEN, Ion, adduct, ion,
};
use super::grammar::{ADD_ATOM, Limits, replay};
use super::targets::{Embedding, Labels};
use super::graph::MolGraph;

/// Heavy elements in [`ELEMENTS`] order (hydrogen skipped): the mixed-radix
/// digits of §2.1, carbon least significant.
const HEAVY: [usize; 9] = [0, 2, 3, 4, 5, 6, 7, 8, 9];

/// `ion_meta` bit 0: the radix product exceeds the visit budget (§2.1).
pub const ION_SEARCH_EXHAUSTED: u32 = 1 << 0;
/// `ion_meta` bit 1: more accepted hypotheses than the kept capacity `J`.
pub const ION_CAPACITY_EXCEEDED: u32 = 1 << 1;
/// `ion_meta` bit 2: the peak is not searched at all (unknown precision, a
/// window wider than the hydrogen bound, a padding peak, an empty parent).
pub const ION_UNAVAILABLE: u32 = 1 << 2;
/// Evidence bit 7: the assignment support behind the status is incomplete.
pub const EVIDENCE_SUPPORT_INCOMPLETE: u32 = 1 << 7;

/// Per-peak work and capacity limits of §2.1 (`ion_work_max`, `J`).
pub struct IonLimits {
    /// Most heavy vectors visited per peak (`ion_work_max`).
    pub work_max: u32,
    /// Most accepted hypotheses kept per peak (`J`, the `ion` width).
    pub kept: u32,
}

impl Default for IonLimits {
    /// The spec defaults: `ion_work_max = 4096`, `J = 4`.
    fn default() -> Self {
        Self {
            work_max: 4096,
            kept: 4,
        }
    }
}

/// One accepted ion hypothesis: the ion's own composition (hydrogen is the
/// ion's own count), its atomic mass and its signed residual `mass − t`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IonHypothesis {
    /// Element counts with the ion's own hydrogen count at [`HYDROGEN`].
    pub counts: Composition,
    /// Atomic mass in integer units.
    pub mass: u32,
    /// Signed `mass − t` in integer units (fits `i32` for accepted verdicts).
    pub residual: i32,
}

/// The assignment of one peak: full accepted/ambiguous counts, the kept
/// visit-order prefix and the §2.1 status bits.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IonAssignment {
    /// Accepted hypotheses among the visited vectors (all of them, not just
    /// the kept ones).
    pub accepted: u32,
    /// Ambiguous hypotheses among the visited vectors (counted only).
    pub ambiguous: u32,
    /// The first `J` accepted hypotheses in visit order.
    pub kept: Vec<IonHypothesis>,
    /// Status bits ([`ION_SEARCH_EXHAUSTED`], [`ION_CAPACITY_EXCEEDED`],
    /// [`ION_UNAVAILABLE`]); zero means the support is complete.
    pub status: u32,
}

/// Bias of the lane residual: the signed `mass − t` as offset-binary `u32`
/// (`ob = (mass − t) + 2^31`, wrapping).
pub const LANE_BIAS: u32 = 0x8000_0000;
/// Lane verdict code: the hypothesis is ruled out.
pub const LANE_REJECT: u32 = 0;
/// Lane verdict code: the hypothesis explains the peak.
pub const LANE_ACCEPT: u32 = 1;
/// Lane verdict code: counted, but neither a label nor a rejection.
pub const LANE_AMBIGUOUS: u32 = 2;
/// Heavy slots of the lane (carbon least significant).
pub const LANE_HEAVY: u32 = 9;
/// Hypothesis slots per visit of the lane (a searched window spans at most
/// three hydrogen counts, see [`ion_assign`]).
pub const LANE_H: u32 = 3;

/// Heavy-slot element masses in [`HEAVY`] order (C, N, O, F, P, S, Cl, Br, I).
/// Kernel-portable literals: [`heavy_mass_u32`] selects them with an
/// if-chain (CubeCL 0.10 cannot index a fixed array by a dynamic slot), and
/// `tests/ms2_ion.rs` pins every entry against
/// [`ELEMENTS`](super::chem::ELEMENTS).
pub const HEAVY_MASS_U32: [u32; 9] = [
    12_000_000,
    14_003_074,
    15_994_915,
    18_998_403,
    30_973_762,
    31_972_071,
    34_968_853,
    78_918_338,
    126_904_472,
];

/// Heavy-slot rounding residuals in [`HEAVY`] order (see [`HEAVY_MASS_U32`]).
pub const HEAVY_RES_U32: [u32; 9] = [0, 5, 381, 163, 2, 175, 318, 400, 100];

/// Element mass of heavy slot `s` (see [`HEAVY_MASS_U32`]).
///
/// Kernel-portable if-chain: the `#[cube]` kernel copies this function with
/// the same literals, so no element table is bound as an array.
pub fn heavy_mass_u32(slot: u32) -> u32 {
    let mut m = 12_000_000u32;
    if slot == 1 {
        m = 14_003_074;
    }
    if slot == 2 {
        m = 15_994_915;
    }
    if slot == 3 {
        m = 18_998_403;
    }
    if slot == 4 {
        m = 30_973_762;
    }
    if slot == 5 {
        m = 31_972_071;
    }
    if slot == 6 {
        m = 34_968_853;
    }
    if slot == 7 {
        m = 78_918_338;
    }
    if slot == 8 {
        m = 126_904_472;
    }
    m
}

/// Rounding residual of heavy slot `s` (see [`HEAVY_RES_U32`]).
///
/// Kernel-portable if-chain, copied by the `#[cube]` kernel like
/// [`heavy_mass_u32`].
pub fn heavy_res_u32(slot: u32) -> u32 {
    let mut r = 0u32;
    if slot == 1 {
        r = 5;
    }
    if slot == 2 {
        r = 381;
    }
    if slot == 3 {
        r = 163;
    }
    if slot == 4 {
        r = 2;
    }
    if slot == 5 {
        r = 175;
    }
    if slot == 6 {
        r = 318;
    }
    if slot == 7 {
        r = 400;
    }
    if slot == 8 {
        r = 100;
    }
    r
}

/// Packed atom-type fields of V0 atom type id `ty` (1–17):
/// `(element << 16) | parent_hydrogens`, in [`ELEMENTS`](super::chem::ELEMENTS)
/// order (C 0, H 1, N 2, O 3, F 4, P 5, S 6, Cl 7, Br 8, I 9).
///
/// Kernel-portable if-chain over the frozen V0 domain, copied by the
/// `#[cube]` kernel, so the evidence lane needs no atom-type table binding.
/// Out-of-domain ids yield 0; the lane only calls this for `1 <= ty <= 17`.
pub fn atom_fields_u32(ty: u32) -> u32 {
    let mut fields = 0u32;
    if ty == 1 {
        fields = 0;
    }
    if ty == 2 {
        fields = 1;
    }
    if ty == 3 {
        fields = 2;
    }
    if ty == 4 {
        fields = 3;
    }
    if ty == 5 {
        fields = 2 * 65536;
    }
    if ty == 6 {
        fields = 2 * 65536 + 1;
    }
    if ty == 7 {
        fields = 2 * 65536 + 2;
    }
    if ty == 8 {
        fields = 3 * 65536;
    }
    if ty == 9 {
        fields = 3 * 65536 + 1;
    }
    if ty == 10 {
        fields = 4 * 65536;
    }
    if ty == 11 {
        fields = 7 * 65536;
    }
    if ty == 12 {
        fields = 8 * 65536;
    }
    if ty == 13 {
        fields = 6 * 65536;
    }
    if ty == 14 {
        fields = 6 * 65536 + 1;
    }
    if ty == 15 {
        fields = 6 * 65536;
    }
    if ty == 16 {
        fields = 5 * 65536;
    }
    if ty == 17 {
        fields = 9 * 65536;
    }
    fields
}

/// Fixed-size input of one lane visit: the kernel twin's registers.
///
/// All scalars and arrays are `u32`. `radices`, `heavy_mass` and `heavy_res`
/// run over the 9 heavy slots in [`HEAVY`] order; `h_res` is the hydrogen
/// rounding residual, `electron_res` the electron rounding residual, `t` the
/// target mass, `tol` the ppm tolerance, `uncertainty` the spectrum precision,
/// `lo`/`hi` the searched window, `m_h` the hydrogen mass, `h_hi_abs` the
/// output-type hydrogen cap and `k` the visit index (`1 <= k`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IonLaneIn {
    /// Heavy radices `parent[e] + 1`, each `>= 1`.
    pub radices: [u32; 9],
    /// Heavy element masses, each `>= 1`.
    pub heavy_mass: [u32; 9],
    /// Heavy rounding residuals.
    pub heavy_res: [u32; 9],
    /// Hydrogen rounding residual.
    pub h_res: u32,
    /// Electron rounding residual.
    pub electron_res: u32,
    /// Target mass `t`.
    pub t: u32,
    /// Ppm tolerance `tol_p`.
    pub tol: u32,
    /// Spectrum precision `U`.
    pub uncertainty: u32,
    /// Window lower edge.
    pub lo: u32,
    /// Window upper edge.
    pub hi: u32,
    /// Hydrogen mass, `>= 1`.
    pub m_h: u32,
    /// Hydrogen cap of the output type.
    pub h_hi_abs: u32,
    /// Visit index, `1 <= k`.
    pub k: u32,
}

/// Fixed-size output of one lane visit: every array is `u32`.
///
/// Entry `j` below `n_h` carries one hydrogen hypothesis: its mass, its
/// offset-binary residual and its verdict code ([`LANE_ACCEPT`],
/// [`LANE_AMBIGUOUS`] or [`LANE_REJECT`]). `heavy` holds the mixed-radix
/// digits and `h_lo` the first hydrogen count, so hypothesis `j` has
/// hydrogen `h_lo + j`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IonLaneOut {
    /// Mixed-radix digits of `k`, carbon least significant.
    pub heavy: [u32; 9],
    /// First hydrogen count of the window interval.
    pub h_lo: u32,
    /// Hydrogen hypotheses carried below (at most [`LANE_H`]).
    pub n_h: u32,
    /// Hypothesis masses.
    pub mass: [u32; 3],
    /// Hypothesis residuals as offset-binary `u32`.
    pub residual_ob: [u32; 3],
    /// Hypothesis verdict codes.
    pub verdict: [u32; 3],
}

/// Overflow-free verdict rule of the kernel twin (contract §5).
///
/// The kernel twin is expressed through this helper: accept holds exactly
/// when `r <= tol` and `error <= tol − r`, reject exactly when `r > tol`
/// and `r − tol > error`, else ambiguity, where `r = |observed − computed|`.
/// Each subtraction runs only where its guard holds (short-circuit `&&`)
/// or wraps, so no sum is ever formed and every comparison stays in `u32`.
/// Returns [`LANE_ACCEPT`], [`LANE_REJECT`] or [`LANE_AMBIGUOUS`].
pub fn lane_verdict_u32(observed: u32, computed: u32, error: u32, tolerance: u32) -> u32 {
    let r = observed.abs_diff(computed);
    let in_tol = r <= tolerance;
    let tol_minus_r = tolerance.wrapping_sub(r);
    let hit = in_tol && error <= tol_minus_r;
    let over = r > tolerance;
    let r_minus_tol = r.wrapping_sub(tolerance);
    let miss = over && r_minus_tol > error;
    let mut v = LANE_AMBIGUOUS;
    let set_hit = hit;
    let set_miss = !hit && miss;
    if set_hit {
        v = LANE_ACCEPT;
    }
    if set_miss {
        v = LANE_REJECT;
    }
    v
}

/// Guarded visit budget of the kernel twin.
///
/// The kernel twin is expressed through this helper: `visits` is
/// `min(total − 1, work_max)` where `total` is the radix product, and the
/// second return holds 1 exactly when `total − 1 > work_max`. The product is
/// multiplied only while it fits the remaining headroom, tested with
/// division, so nothing overflows. The `work_max == u32::MAX` budget names
/// the value `2^32`: the `> MAX / radix` step then separates an exact
/// `2^32` product (complete, visits `MAX`) from a larger one (exhausted) by
/// the remainder test, and an exact `2^32` partial rides the `big` flag
/// until a further radix above 1 exhausts it.
pub fn lane_visits_u32(radices: [u32; 9], work_max: u32) -> (u32, u32) {
    let mut partial: u32 = 1;
    let mut cut: u32 = 0;
    let mut big: u32 = 0;
    let mut i: u32 = 0;
    while i < LANE_HEAVY {
        let r = radices[i as usize];
        let top = work_max == u32::MAX;
        let lim = work_max.wrapping_add(1);
        let over_a = partial > lim / r;
        let q = u32::MAX / r;
        let rem = u32::MAX % r;
        let fits_b = partial <= q;
        let plus_one = r != 1;
        let exact_b = plus_one && partial == q.wrapping_add(1) && r == rem.wrapping_add(1);
        let live = cut == 0;
        let was_big = big == 1;
        let one = r == 1;
        let cut_big = live && was_big && !one;
        let cut_a = live && !was_big && !top && over_a;
        let cut_b = live && !was_big && top && !fits_b && !exact_b;
        let cut_now = cut_big || cut_a || cut_b;
        let grow_a = live && !was_big && !top && !over_a;
        let grow_b = live && !was_big && top && fits_b;
        let grow_big = live && !was_big && top && !fits_b && exact_b;
        let grow = grow_a || grow_b;
        if cut_now {
            cut = 1;
        }
        if grow {
            partial = partial.wrapping_mul(r);
        }
        if grow_big {
            big = 1;
            partial = 0;
        }
        i = i.wrapping_add(1);
    }
    let mut visits = partial.wrapping_sub(1);
    let exhausted = cut == 1;
    let at_top = !exhausted && big == 1;
    if exhausted {
        visits = work_max;
    }
    if at_top {
        visits = u32::MAX;
    }
    (visits, cut)
}

/// One visit index of the assignment search: the kernel twin.
///
/// The kernel twin is `ion_lane_visit`: a `#[cube]` kernel copies this
/// function line for line. Every value is `u32` (boolean flags combine with
/// `&&`); `usize` appears only to index the fixed arrays. The caller passes
/// `1 <= k`, radices `>= 1`, masses `>= 1` and a searched window that spans
/// at most three hydrogen counts; the mixed-radix digits come from repeated
/// `u32` div/mod, the heavy mass from division-guarded `u32` accumulation,
/// the hydrogen interval from quotient/remainder splits, each hypothesis
/// verdict from the overflow-free `u32` comparisons of [`lane_verdict_u32`]
/// (called here, never the multi-width rule), and each residual as
/// offset-binary `u32` (`mass − t + 2^31`, wrapping).
pub fn ion_lane_visit(input: IonLaneIn) -> IonLaneOut {
    let mut heavy: [u32; 9] = [0; 9];
    let mut tmp = input.k;
    let mut i: u32 = 0;
    while i < LANE_HEAVY {
        let r = input.radices[i as usize];
        heavy[i as usize] = tmp % r;
        tmp /= r;
        i = i.wrapping_add(1);
    }
    debug_assert!(tmp == 0);
    let mut m: u32 = 0;
    let mut mass_ok = true;
    let mut j: u32 = 0;
    while j < LANE_HEAVY {
        let d = heavy[j as usize];
        let me = input.heavy_mass[j as usize];
        let room = u32::MAX.wrapping_sub(m);
        let fits = d <= room / me;
        let go = mass_ok && fits;
        mass_ok = mass_ok && fits;
        if go {
            m = m.wrapping_add(d.wrapping_mul(me));
        }
        j = j.wrapping_add(1);
    }
    let mut res_heavy: u32 = 0;
    let mut e: u32 = 0;
    while e < LANE_HEAVY {
        let dd = heavy[e as usize];
        res_heavy = res_heavy.wrapping_add(dd.wrapping_mul(input.heavy_res[e as usize]));
        e = e.wrapping_add(1);
    }
    let in_hi = m <= input.hi;
    let gated = mass_ok && in_hi;
    let below = m < input.lo;
    let gap = input.lo.wrapping_sub(m);
    let ceil = gap / input.m_h + u32::from(!gap.is_multiple_of(input.m_h));
    let take_ceil = gated && below;
    let mut h_lo: u32 = 0;
    if take_ceil {
        h_lo = ceil;
    }
    let span = input.hi.wrapping_sub(m);
    let h_hi = (span / input.m_h).min(input.h_hi_abs);
    let has = gated && h_lo <= h_hi;
    let full_n = h_hi.wrapping_sub(h_lo).wrapping_add(1);
    let mut want: u32 = 0;
    if has {
        want = full_n;
    }
    let clamp = want > LANE_H;
    let mut cap_n = want;
    if clamp {
        cap_n = LANE_H;
    }
    let mut mass: [u32; 3] = [0; 3];
    let mut residual_ob: [u32; 3] = [0; 3];
    let mut verdict: [u32; 3] = [LANE_REJECT; 3];
    let mut valid: u32 = 0;
    let mut t: u32 = 0;
    while t < LANE_H {
        let on = has && t < cap_n;
        let h = h_lo.wrapping_add(t);
        let room_h = u32::MAX.wrapping_sub(m);
        let fits_h = h <= room_h / input.m_h;
        let slot = on && fits_h;
        // The candidate mass is formed only inside the slot guard: `fits_h`
        // makes `m + h * m_h` exact in `u32`, while an inactive slot's
        // product can wrap (review: parent C2 N305 at visit 917). The
        // wrapping multiplications above stay defined but unused off-slot.
        if slot {
            let cand = m.wrapping_add(h.wrapping_mul(input.m_h));
            let res = res_heavy
                .wrapping_add(h.wrapping_mul(input.h_res))
                .wrapping_add(input.electron_res);
            let arith = res / 1000 + u32::from(!res.is_multiple_of(1000));
            let bound = arith.saturating_add(input.uncertainty);
            let v = lane_verdict_u32(input.t, cand, bound, input.tol);
            let ob = cand.wrapping_sub(input.t).wrapping_add(LANE_BIAS);
            mass[t as usize] = cand;
            residual_ob[t as usize] = ob;
            verdict[t as usize] = v;
            valid = valid.wrapping_add(1);
        }
        t = t.wrapping_add(1);
    }
    IonLaneOut {
        heavy,
        h_lo,
        n_h: valid,
        mass,
        residual_ob,
        verdict,
    }
}

/// Full-buffer `(b, f, p)` lane of `ms2_ion_assign`: the kernel twin.
///
/// `top_counts` is `[B, F, 10]` flat parent counts, `kept` `[B, N, 3]` flat
/// kept peaks (raw index, m/z, reverse), `meta` `[B, 8]` flat spectrum words
/// (peak count, precursor, precursor uncertainty, adduct, fragment-tolerance
/// ppm tenths, precursor tolerance, id lo/hi) and `spec` `[B, 2]` flat
/// per-spectrum words (m/z uncertainty `U`, reserved). Writes the `[B, F, N,
/// J, 12]` flat `ion` row (10 counts with the ion's own hydrogen count,
/// mass, residual offset by 2^31; zeros in padding) and the `[B, F, N, 4]`
/// flat `ion_meta` row (accepted, ambiguous, kept, status). Every element of
/// both rows is written, padding peaks and unstarted formula slots
/// (all-zero `top_counts`) included; those yield zeros with
/// [`ION_UNAVAILABLE`].
///
/// The visit itself runs through [`ion_lane_visit`] (with [`lane_visits_u32`]
/// and [`lane_verdict_u32`]); the `#[cube]` kernel scalar-expands that call
/// (CubeCL 0.10 has no fixed-size local arrays as kernel values) with the
/// same arithmetic, each block citing the twin. The spectrum's m/z
/// uncertainty `U` lives only on the host (contract `Spectra`;
/// `meta[.., 2]` is the precursor uncertainty), so it travels in `spec`.
/// Preconditions: `ppm <= 1000` (the [`tolerance_u32`](super::chem::tolerance_u32) proof bound; wider
/// values wrap deterministically through `wrapping_*` but are rejected by
/// the wrapper). An unknown adduct or a target mass outside `u32` yields
/// [`ION_UNAVAILABLE`]; the wrapper rejects them as errors instead.
#[allow(clippy::too_many_arguments)]
pub fn ion_assign_lane(
    top_counts: &[u32],
    kept: &[u32],
    meta: &[u32],
    spec: &[u32],
    b: u32,
    f: u32,
    p: u32,
    f_dim: u32,
    n: u32,
    j: u32,
    work_max: u32,
    ion: &mut [u32],
    ion_meta: &mut [u32],
) {
    // Full-buffer addresses in u32 exactly as the kernel does (cast to
    // `usize` only at the index expression).
    let tc_base: u32 = (b * f_dim + f) * 10;
    let kept_base: u32 = (b * n + p) * 3;
    let meta_base: u32 = b * 8;
    let spec_base: u32 = b * 2;
    let pos: u32 = (b * f_dim + f) * n + p;
    let ion_base: u32 = pos * j * 12;
    let im_base: u32 = pos * 4;
    // Every output element is written: zero both rows first.
    let mut zi: u32 = 0;
    while zi < j * 12 {
        ion[(ion_base + zi) as usize] = 0;
        zi += 1;
    }
    ion_meta[im_base as usize] = 0;
    ion_meta[(im_base + 1) as usize] = 0;
    ion_meta[(im_base + 2) as usize] = 0;
    ion_meta[(im_base + 3) as usize] = 0;
    // Parent counts and the empty-formula gate.
    let mut all_zero = true;
    let mut e: u32 = 0;
    while e < 10 {
        if top_counts[(tc_base + e) as usize] != 0 {
            all_zero = false;
        }
        e += 1;
    }
    let peak_count = meta[meta_base as usize];
    let peak_mz = kept[(kept_base + 1) as usize];
    let adduct_id = meta[(meta_base + 3) as usize];
    let ppm = meta[(meta_base + 4) as usize];
    let u = spec[spec_base as usize];
    let known_adduct = adduct_id == 1 || adduct_id == 2;
    // t = mz + z_a * m_e: +549 for [M+H]+, −549 for [M-H]-.
    let mut t = 0u32;
    let mut t_ok = false;
    if adduct_id == 1 && peak_mz <= u32::MAX - ELECTRON_MASS {
        t = peak_mz + ELECTRON_MASS;
        t_ok = true;
    }
    if adduct_id == 2 && peak_mz >= ELECTRON_MASS {
        t = peak_mz - ELECTRON_MASS;
        t_ok = true;
    }
    // Fragment tolerance by the `u32` algorithm of [`tolerance_u32`](super::chem::tolerance_u32).
    let hi = peak_mz / 10_000;
    let lo = peak_mz % 10_000;
    let q = hi.wrapping_mul(ppm);
    let tol_p = q / 1000 + ((q % 1000) * 10_000 + lo.wrapping_mul(ppm)) / 10_000_000;
    // E_ion = ceil((parent residuals + 3 * residual_H + 421) / 1000): an
    // upper bound of every hypothesis's own bound, because residuals are
    // non-negative and no hypothesis carries more than parent[H] + 3
    // hydrogens (max(h_a, 0) <= 1 for the V0 adducts, plus 2). The parent
    // sum runs over all 10 elements, hydrogen included.
    let mut nda = 0u32;
    let mut hs: u32 = 0;
    while hs < 9 {
        nda = nda.wrapping_add(
            top_counts[(tc_base + HEAVY[hs as usize] as u32) as usize]
                .wrapping_mul(heavy_res_u32(hs)),
        );
        hs += 1;
    }
    nda = nda.wrapping_add(
        top_counts[(tc_base + HYDROGEN as u32) as usize].wrapping_mul(ELEMENTS[HYDROGEN].residual_nda),
    );
    nda = nda.wrapping_add(3 * ELEMENTS[HYDROGEN].residual_nda + ELECTRON_RESIDUAL_NDA);
    let e_ion = nda / 1000 + u32::from(!nda.is_multiple_of(1000));
    let half_p = tol_p.saturating_add(u).saturating_add(e_ion);
    // Scope restriction: a wider window makes the peak ion_unavailable, so
    // every visited vector below spans at most three hydrogen counts.
    let searchable = known_adduct
        && t_ok
        && !all_zero
        && peak_mz != 0
        && p < peak_count
        && u != u32::MAX
        && half_p <= ELEMENTS[HYDROGEN].mass;
    if searchable {
        // Lane register tables in HEAVY order (all u32; radices >= 1, masses >= 1).
        let mut radices: [u32; 9] = [1; 9];
        let mut heavy_mass: [u32; 9] = [1; 9];
        let mut heavy_res: [u32; 9] = [0; 9];
        let mut reg: u32 = 0;
        while reg < 9 {
            let ce = HEAVY[reg as usize];
            radices[reg as usize] = top_counts[(tc_base + ce as u32) as usize].wrapping_add(1);
            heavy_mass[reg as usize] = ELEMENTS[ce].mass;
            heavy_res[reg as usize] = ELEMENTS[ce].residual_nda;
            reg += 1;
        }
        // Guarded visit budget: `cut == 1` exactly when total − 1 > work_max.
        let (visits, cut) = lane_visits_u32(radices, work_max);
        let mut status: u32 = 0;
        if cut == 1 {
            status |= ION_SEARCH_EXHAUSTED;
        }
        let lo_w = t.saturating_sub(half_p);
        let hi_w = t.saturating_add(half_p);
        let h_pos = if adduct_id == 1 { 1 } else { 0 };
        let h_cap = top_counts[(tc_base + HYDROGEN as u32) as usize]
            .wrapping_add(h_pos)
            .wrapping_add(2);
        // A kept ion composition stores its hydrogen count in u16, so the
        // search range never exceeds what the output type holds.
        let h_hi_abs: u32 = h_cap.min(u32::from(u16::MAX));
        let mut accepted: u32 = 0;
        let mut ambiguous: u32 = 0;
        let mut stored: u32 = 0;
        let m_h = ELEMENTS[HYDROGEN].mass;
        let h_res = ELEMENTS[HYDROGEN].residual_nda;
        let mut k = 1u32;
        let mut live = 1u32;
        while k <= visits && live == 1 {
            // One kernel-twin visit: digits, masses and verdicts in u32 only.
            let out = ion_lane_visit(IonLaneIn {
                radices,
                heavy_mass,
                heavy_res,
                h_res,
                electron_res: ELECTRON_RESIDUAL_NDA,
                t,
                tol: tol_p,
                uncertainty: u,
                lo: lo_w,
                hi: hi_w,
                m_h,
                h_hi_abs,
                k,
            });
            let mut slot: u32 = 0;
            while slot < out.n_h {
                let v = out.verdict[slot as usize];
                if v == LANE_ACCEPT {
                    accepted = accepted.saturating_add(1);
                }
                if v == LANE_AMBIGUOUS {
                    ambiguous = ambiguous.saturating_add(1);
                }
                if v == LANE_ACCEPT && stored < j {
                    let h = out.h_lo.wrapping_add(slot);
                    let w: u32 = ion_base + stored * 12;
                    let mut ds: u32 = 0;
                    while ds < 9 {
                        ion[(w + HEAVY[ds as usize] as u32) as usize] = out.heavy[ds as usize];
                        ds += 1;
                    }
                    ion[(w + HYDROGEN as u32) as usize] = h;
                    ion[(w + 10) as usize] = out.mass[slot as usize];
                    ion[(w + 11) as usize] = out.residual_ob[slot as usize];
                    stored += 1;
                }
                slot += 1;
            }
            if k == u32::MAX {
                live = 0;
            } else {
                k += 1;
            }
        }
        if accepted > j {
            status |= ION_CAPACITY_EXCEEDED;
        }
        ion_meta[im_base as usize] = accepted;
        ion_meta[(im_base + 1) as usize] = ambiguous;
        ion_meta[(im_base + 2) as usize] = stored;
        ion_meta[(im_base + 3) as usize] = status;
    } else {
        ion_meta[(im_base + 3) as usize] = ION_UNAVAILABLE;
    }
}

/// Hypotheses of one peak under a parent formula (spec §2.1): host wrapper.
///
/// Packs the single peak into the [`ion_assign_lane`] buffer form and calls
/// the lane, so this wrapper exercises exactly the kernel twin's code: it
/// owns the `Vec` allocation and converts the lane's offset-binary residuals
/// back to `i32` (accepted verdicts have `r <= tol_p`, tiny, so the
/// conversion is exact).
///
/// Heavy sub-vectors visit mixed-radix index 1 upward (carbon least
/// significant, index 0 excluded so hydrogen-only "ions" never appear). The
/// radix product is guarded by division and never overflows: `total − 1 >
/// work_max` is decided exactly and visits stop at `work_max`. The window is
/// `t ± half_p` with `half_p = tol_p + U + E_ion` by saturating additions; a
/// `half_p` wider than one hydrogen mass makes the peak [`ION_UNAVAILABLE`],
/// so every visited vector spans at most three hydrogen counts. Each `(u, h)`
/// gets the contracts §5 verdict with its own bound.
///
/// Errors: unknown adduct (`unsupported_adduct`), `ppm_tenths > 1000`
/// (contract §3.1 and the [`tolerance_u32`](super::chem::tolerance_u32) proof bound), a target mass `t`
/// outside `u32` (`mass_overflow`). The unknown-precision sentinel
/// (`mz_uncertainty == u32::MAX`), a padding peak (`peak_mz == 0`) and an
/// empty parent yield [`ION_UNAVAILABLE`], not an error.
pub fn ion_assign(
    parent: &Composition,
    adduct_id: u16,
    peak_mz: u32,
    mz_uncertainty: u32,
    ppm_tenths: u32,
    limits: &IonLimits,
) -> Result<IonAssignment> {
    let unavailable = IonAssignment {
        accepted: 0,
        ambiguous: 0,
        kept: Vec::new(),
        status: ION_UNAVAILABLE,
    };
    let a = adduct(adduct_id).ok_or_else(|| {
        Error::Unsupported(format!("ion_assign: unknown adduct id {adduct_id}"))
    })?;
    if ppm_tenths > 1000 {
        return Err(Error::config(format!(
            "ion_assign: ppm_tenths {ppm_tenths} exceeds the 1000 proof bound"
        )));
    }
    // Padding peaks and unstarted formulas are never searched (§2.1).
    if peak_mz == 0 || parent.iter().all(|&n| n == 0) {
        return Ok(unavailable);
    }
    // The unknown-precision sentinel disables exact-mass decisions (§5).
    if mz_uncertainty == u32::MAX {
        return Ok(unavailable);
    }
    // t = mz + z_a * m_e, checked: +549 for [M+H]+, −549 for [M-H]-. The
    // lane maps an out-of-range target to ION_UNAVAILABLE; the wrapper
    // reports it as an error, as before.
    if a.charge > 0 {
        peak_mz.checked_add(ELECTRON_MASS)
    } else {
        peak_mz.checked_sub(ELECTRON_MASS)
    }
    .ok_or_else(|| {
        Error::Unsupported(format!(
            "mass_overflow: ion target of peak {peak_mz} under adduct {adduct_id} leaves u32 range"
        ))
    })?;
    // Single-peak buffers for the lane: one parent row, one kept peak, one
    // spectrum (peak count 1, the adduct and ppm in their meta words, `U` in
    // spec) and one output row pair.
    let mut top_counts = [0u32; 10];
    for (e, n) in parent.iter().enumerate() {
        top_counts[e] = u32::from(*n);
    }
    let kept = [0u32, peak_mz, 0];
    let meta = [1u32, 0, 0, u32::from(adduct_id), ppm_tenths, 0, 0, 0];
    let spec = [mz_uncertainty, 0];
    let ju = limits.kept as usize;
    let mut ion = vec![0u32; ju * 12];
    let mut ion_meta = vec![0u32; 4];
    ion_assign_lane(
        &top_counts,
        &kept,
        &meta,
        &spec,
        0,
        0,
        0,
        1,
        1,
        limits.kept,
        limits.work_max,
        &mut ion,
        &mut ion_meta,
    );
    let stored = ion_meta[2] as usize;
    let mut kept_hyps = Vec::with_capacity(stored);
    for q in 0..stored {
        let w = q * 12;
        let mut full: Composition = [0; 10];
        for e in 0..10 {
            full[e] = ion[w + e] as u16;
        }
        // Accepted verdicts have r <= tol_p (tiny), so the offset-binary
        // lane residual converts to i32 exactly.
        let residual = ion[w + 11].wrapping_sub(LANE_BIAS) as i32;
        kept_hyps.push(IonHypothesis {
            counts: full,
            mass: ion[w + 10],
            residual,
        });
    }
    Ok(IonAssignment {
        accepted: ion_meta[0],
        ambiguous: ion_meta[1],
        kept: kept_hyps,
        status: ion_meta[3],
    })
}

/// One deduplicated ion label: the raw peak index in the batch and the ion
/// composition (hydrogen is the ion's own count).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IonLabel {
    /// Raw peak index in the uploaded batch (what `kept[.., 0]` holds).
    pub raw_index: u32,
    /// Ion composition: `heavy(g)` with hydrogen `H(g) + h_a + s`.
    pub counts: Composition,
}

/// The label set of one spectrum: deduplicated ion compositions sorted by
/// (raw index, the 10 counts lexicographically), cut at `cap`, with the
/// dropped remainder counted.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IonLabels {
    /// The first `cap` labels in sort order.
    pub labels: Vec<IonLabel>,
    /// Labels beyond `cap` (`assignment_label_overflow`).
    pub overflow: usize,
}

/// Ion label sets from the recipe anchors (spec §2.3).
///
/// For each anchor `(peak_id, s)` of each kept target `g`, the composition
/// `(heavy(g), H(g) + h_a + s)`, where `heavy(g)` and `H(g)` come from the
/// target's canonical trace replayed through the existing grammar/graph code
/// (atom types carry element and parent hydrogens). Peak ids map to raw batch
/// indices through `raw_index_of_peak_id` (no `peak_id` buffer is bound, so
/// the host does this mapping); anchors of peaks outside the batch, with a
/// negative ion hydrogen count, or behind an untraceable target are skipped.
/// An unknown adduct yields no labels.
pub fn ion_labels(
    labels: &Labels,
    adduct_id: u16,
    raw_index_of_peak_id: impl Fn(u32) -> Option<u32>,
    cap: usize,
) -> IonLabels {
    let mut set: BTreeSet<(u32, Composition)> = BTreeSet::new();
    if let Some(a) = adduct(adduct_id) {
        for target in &labels.targets {
            let graph = replay(&target.trace, Limits::V0, None).and_then(|s| s.graph());
            let Ok(graph) = graph else { continue };
            let comp = graph.composition();
            for &(peak_id, s) in &target.anchors {
                let Some(raw) = raw_index_of_peak_id(peak_id) else {
                    continue;
                };
                let h = comp[HYDROGEN] as i64 + i64::from(a.hydrogens) + i64::from(s);
                if h < 0 || h > i64::from(u16::MAX) {
                    continue;
                }
                let mut ion_c = comp;
                ion_c[HYDROGEN] = h as u16;
                set.insert((raw, ion_c));
            }
        }
    }
    let overflow = set.len().saturating_sub(cap);
    let labels = set
        .into_iter()
        .take(cap)
        .map(|(raw_index, counts)| IonLabel { raw_index, counts })
        .collect();
    IonLabels { labels, overflow }
}

/// Full-buffer `(b, p)` lane of `ms2_ion_label_mask`: the kernel twin.
///
/// `ion_labels` is `[B, L, 12]` flat (raw index, the 10 counts with the ion's
/// hydrogen count, a valid flag), `ion`/`ion_meta` the assignment rows of
/// formula slot `f_slot` and `kept` the `[B, N, 3]` kept peaks. Writes the
/// `[B, N, J + 1]` flat float `label_mask` row (class `J` is unassigned) and
/// the `label_state` word (`0` no label, `1` some label kept with every label
/// kept, `2` labels exist but none kept, `3` true partial: some label kept
/// while some label of the same peak is not, computed from the label matching
/// itself rather than from the mask sum). Rows outside states 1 and 3 carry
/// the one-hot of the unassigned class, so their masked log-sum is finite;
/// the caller multiplies them by the 0 eligibility indicator. Label rows
/// with a cleared valid flag (upload overflow, §2.3) never match. Every
/// element written.
#[allow(clippy::too_many_arguments)]
pub fn label_mask_lane(
    ion_labels: &[u32],
    ion: &[u32],
    ion_meta: &[u32],
    kept: &[u32],
    b: u32,
    p: u32,
    f_slot: u32,
    f_dim: u32,
    n: u32,
    j: u32,
    label_cap: u32,
    label_mask: &mut [f32],
    label_state: &mut [u32],
) {
    // Full-buffer addresses in u32 exactly as the kernel does (cast to
    // `usize` only at the index expression).
    let width: u32 = j + 1;
    let lab_base: u32 = b * label_cap * 12;
    let pos: u32 = (b * f_dim + f_slot) * n + p;
    let ion_base: u32 = pos * j * 12;
    let im_base: u32 = pos * 4;
    let kept_base: u32 = (b * n + p) * 3;
    let mask_base: u32 = (b * n + p) * width;
    let state_idx: u32 = b * n + p;
    let mut zi: u32 = 0;
    while zi < width {
        label_mask[(mask_base + zi) as usize] = 0.0;
        zi += 1;
    }
    label_state[state_idx as usize] = 0;
    let raw = kept[kept_base as usize];
    if raw != u32::MAX {
        let kept_n: u32 = ion_meta[(im_base + 2) as usize];
        let mut any_kept = false;
        let mut qq: u32 = 0;
        while qq < j && qq < kept_n {
            let hb: u32 = ion_base + qq * 12;
            let mut hit = false;
            let mut l: u32 = 0;
            while l < label_cap {
                let lb: u32 = lab_base + l * 12;
                if ion_labels[(lb + 11) as usize] != 0 && ion_labels[lb as usize] == raw {
                    let mut eq = true;
                    let mut e: u32 = 0;
                    while e < 10 {
                        if ion[(hb + e) as usize] != ion_labels[(lb + 1 + e) as usize] {
                            eq = false;
                        }
                        e += 1;
                    }
                    if eq {
                        hit = true;
                    }
                }
                l += 1;
            }
            if hit {
                label_mask[(mask_base + qq) as usize] = 1.0;
                any_kept = true;
            }
            qq += 1;
        }
        if any_kept {
            // True partial from the label matching itself (review finding
            // A1): some label of this peak is kept while some label of the
            // same peak is not among the kept hypotheses.
            let mut any_dropped = false;
            let mut l: u32 = 0;
            while l < label_cap {
                let lb: u32 = lab_base + l * 12;
                if ion_labels[(lb + 11) as usize] != 0 && ion_labels[lb as usize] == raw {
                    let mut hit = false;
                    let mut qq: u32 = 0;
                    while qq < j && qq < kept_n {
                        let hb: u32 = ion_base + qq * 12;
                        let mut eq = true;
                        let mut e: u32 = 0;
                        while e < 10 {
                            if ion[(hb + e) as usize] != ion_labels[(lb + 1 + e) as usize] {
                                eq = false;
                            }
                            e += 1;
                        }
                        if eq {
                            hit = true;
                        }
                        qq += 1;
                    }
                    if !hit {
                        any_dropped = true;
                    }
                }
                l += 1;
            }
            if any_dropped {
                label_state[state_idx as usize] = 3;
            } else {
                label_state[state_idx as usize] = 1;
            }
        } else {
            let mut any_label = false;
            let mut l: u32 = 0;
            while l < label_cap {
                let lb: u32 = lab_base + l * 12;
                if ion_labels[(lb + 11) as usize] != 0 && ion_labels[lb as usize] == raw {
                    any_label = true;
                }
                l += 1;
            }
            if any_label {
                label_state[state_idx as usize] = 2;
            }
            label_mask[(mask_base + j) as usize] = 1.0;
        }
    } else {
        label_mask[(mask_base + j) as usize] = 1.0;
    }
}

/// Label mask twin of `ms2_ion_label_mask` (spec §2.3).
///
/// Packs one spectrum into the [`label_mask_lane`] buffer form and calls the
/// lane per kept peak, so this wrapper exercises exactly the kernel twin's
/// code. `kept_raw_index` holds the raw batch index per kept peak slot
/// (`u32::MAX` padding), `hypotheses` the [`IonAssignment`] per slot and `j`
/// the hypothesis capacity `J`. Returns the float mask `[N, J + 1]`
/// (row-major; class `J` is unassigned) and the state `[N]` (`0` no label,
/// `1` some label kept with every label kept, `2` labels exist but none
/// kept, `3` true partial). Peaks outside states 1 and 3 carry the one-hot
/// of the unassigned class, so their masked log-sum is finite; the caller
/// multiplies them by the 0 eligibility indicator.
///
/// Slots without a matching [`IonAssignment`] (shorter `hypotheses`) behave
/// as peaks with no kept hypothesis.
pub fn label_mask(
    kept_raw_index: &[u32],
    hypotheses: &[IonAssignment],
    labels: &IonLabels,
    j: usize,
) -> (Vec<f32>, Vec<u32>) {
    let n = kept_raw_index.len();
    let width = j.saturating_add(1);
    let Some(cells) = n.checked_mul(width) else {
        return (Vec::new(), vec![0; n]);
    };
    let l = labels.labels.len();
    let mut ion = vec![0u32; n * j * 12];
    let mut ion_meta = vec![0u32; n * 4];
    for (p, assign) in hypotheses.iter().enumerate().take(n) {
        let take = assign.kept.len().min(j);
        for (q, hyp) in assign.kept.iter().take(take).enumerate() {
            let w = (p * j + q) * 12;
            for e in 0..10 {
                ion[w + e] = u32::from(hyp.counts[e]);
            }
            ion[w + 10] = hyp.mass;
            ion[w + 11] = (hyp.residual as u32).wrapping_add(LANE_BIAS);
        }
        ion_meta[p * 4] = assign.accepted;
        ion_meta[p * 4 + 1] = assign.ambiguous;
        ion_meta[p * 4 + 2] = take as u32;
        ion_meta[p * 4 + 3] = assign.status;
    }
    let mut lab = vec![0u32; l * 12];
    for (i, label) in labels.labels.iter().enumerate() {
        lab[i * 12] = label.raw_index;
        for e in 0..10 {
            lab[i * 12 + 1 + e] = u32::from(label.counts[e]);
        }
        lab[i * 12 + 11] = 1;
    }
    let mut kept = vec![0u32; n * 3];
    for (p, &raw) in kept_raw_index.iter().enumerate() {
        kept[p * 3] = raw;
    }
    let mut mask = vec![0.0f32; cells];
    let mut state = vec![0u32; n];
    for p in 0..n {
        label_mask_lane(
            &lab,
            &ion,
            &ion_meta,
            &kept,
            0,
            p as u32,
            0,
            1,
            n as u32,
            j as u32,
            l as u32,
            &mut mask,
            &mut state,
        );
    }
    (mask, state)
}

/// Evidence records per candidate (`E = 4`, spec §2.4): the probability-ranked
/// choice of §2.4 needs the assignment head and is done later; the device
/// keeps the first `E` matches in peak order.
pub const EVIDENCE_SLOTS: u32 = 4;

/// Words of one `ms2_ion_evidence` row: status, count, then [`EVIDENCE_SLOTS`]
/// records of (kept-peak position, hypothesis index, shift offset by 2^31,
/// residual offset by 2^31).
pub const EVIDENCE_ROW_WORDS: usize = 18;

/// Packed evidence of one candidate: the host form of one `ms2_ion_evidence`
/// row (spec §2.4).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EvidencePacked {
    /// `0` unassigned (no match), `1` mass-consistent, `2` mass-consistent
    /// for every boundary count, plus [`EVIDENCE_SUPPORT_INCOMPLETE`] when
    /// some compared `(b, f, p)` support is incomplete.
    pub status: u32,
    /// Total matching peaks (possibly more than [`EVIDENCE_SLOTS`]).
    pub count: u32,
    /// Kept-peak position per record.
    pub peak: [u32; EVIDENCE_SLOTS as usize],
    /// Index among the kept `J` per record.
    pub hypothesis: [u32; EVIDENCE_SLOTS as usize],
    /// Shift `s` offset by 2^31 per record.
    pub shift_ob: [u32; EVIDENCE_SLOTS as usize],
    /// Signed hypothesis residual offset by 2^31 per record.
    pub residual_ob: [u32; EVIDENCE_SLOTS as usize],
}

impl EvidencePacked {
    /// The flat `[2 + 4E]` row the kernel writes.
    pub fn to_row(&self) -> Vec<u32> {
        let mut row = vec![0u32; EVIDENCE_ROW_WORDS];
        row[0] = self.status;
        row[1] = self.count;
        for q in 0..EVIDENCE_SLOTS as usize {
            row[2 + q * 4] = self.peak[q];
            row[2 + q * 4 + 1] = self.hypothesis[q];
            row[2 + q * 4 + 2] = self.shift_ob[q];
            row[2 + q * 4 + 3] = self.residual_ob[q];
        }
        row
    }
}

/// Match block shared by [`evidence_lane`] and [`evidence_status`]: the
/// kernel twin's per-trajectory comparison over full buffers.
///
/// `hc` holds the candidate heavy counts in [`HEAVY`] order, `hg` its parent
/// hydrogens `H(g)`, `c_lo`/`c_hi` the boundary-count range, and
/// `ha_pos`/`ha_neg` the adduct hydrogens split (`+1` passes `(1, 0)`, `−1`
/// passes `(0, 1)`). Reads the `ion`/`ion_meta` rows of `(b, slot)` and
/// writes the `[2 + 4E]` flat `evidence` row at `ebase`: the maximum base
/// status over all mass-consistent matched peaks (`0` unassigned, `1`
/// mass-consistent, `2` mass-consistent for every boundary count) with
/// [`EVIDENCE_SUPPORT_INCOMPLETE`] when some compared peak's support is
/// incomplete, the total qualifying-peak count, then the first
/// [`EVIDENCE_SLOTS`] qualifying matches in peak order. A peak matches when
/// some kept hypothesis has `u == u_g` (smallest `|residual|` wins, ties by
/// smaller hypothesis index); its shift is `s = h − H(g) − h_a`, and only
/// shifts with `base > 0` (the §2.4 mass-consistency rule) count, record or
/// raise the status: a candidate with only unsupported shifts keeps status
/// 0, count 0 and an all-zero record area. Every element written; an
/// invalid lane (`ok == 0`, or `slot >= f_dim`) writes a zeroed row.
///
/// The `#[cube]` kernel copies this block with the `hc` array scalar-expanded
/// (nine lane scalars); every other statement is shared verbatim.
// Manual absolute differences: the kernel has no `abs_diff`.
#[allow(clippy::too_many_arguments, clippy::manual_abs_diff)]
pub fn evidence_match(
    hc: [u32; 9],
    hg: u32,
    c_lo: u32,
    c_hi: u32,
    ha_pos: u32,
    ha_neg: u32,
    ion: &[u32],
    ion_meta: &[u32],
    b: u32,
    slot: u32,
    f_dim: u32,
    n: u32,
    j: u32,
    ok: u32,
    evidence: &mut [u32],
    ebase: u32,
) {
    // Full-buffer addresses in u32 exactly as the kernel does (cast to
    // `usize` only at the index expression).
    let mut zi: u32 = 0;
    while zi < EVIDENCE_ROW_WORDS as u32 {
        evidence[(ebase + zi) as usize] = 0;
        zi += 1;
    }
    let valid = if ok == 1 && slot < f_dim { 1 } else { 0 };
    let mut count = 0u32;
    let mut recorded = 0u32;
    let mut best = 0u32;
    let mut incompl = 0u32;
    let mut p = 0u32;
    while p < n {
        if valid == 1 {
            let pos: u32 = (b * f_dim + slot) * n + p;
            let ibase: u32 = pos * j * 12;
            let mbase: u32 = pos * 4;
            if ion_meta[(mbase + 3) as usize] != 0 {
                incompl = 1;
            }
            let kept_n = ion_meta[(mbase + 2) as usize];
            let mut bq = 0u32;
            let mut bres = 0u32;
            let mut found = 0u32;
            let mut q = 0u32;
            while q < kept_n {
                let hb: u32 = ibase + q * 12;
                let mut eq = 1u32;
                if ion[hb as usize] != hc[0] {
                    eq = 0;
                }
                if ion[(hb + 2) as usize] != hc[1] {
                    eq = 0;
                }
                if ion[(hb + 3) as usize] != hc[2] {
                    eq = 0;
                }
                if ion[(hb + 4) as usize] != hc[3] {
                    eq = 0;
                }
                if ion[(hb + 5) as usize] != hc[4] {
                    eq = 0;
                }
                if ion[(hb + 6) as usize] != hc[5] {
                    eq = 0;
                }
                if ion[(hb + 7) as usize] != hc[6] {
                    eq = 0;
                }
                if ion[(hb + 8) as usize] != hc[7] {
                    eq = 0;
                }
                if ion[(hb + 9) as usize] != hc[8] {
                    eq = 0;
                }
                if eq == 1 {
                    let ob = ion[(hb + 11) as usize];
                    let rabs = if ob >= LANE_BIAS {
                        ob - LANE_BIAS
                    } else {
                        LANE_BIAS - ob
                    };
                    if found == 0 || rabs < bres {
                        bq = q;
                        bres = rabs;
                        found = 1;
                    }
                }
                q += 1;
            }
            if found == 1 {
                let hb: u32 = ibase + bq * 12;
                let hh = ion[(hb + 1) as usize];
                let lhs = hh + ha_neg;
                let rhs = hg + ha_pos;
                let ge = if lhs >= rhs { 1 } else { 0 };
                let s_abs = if ge == 1 { lhs - rhs } else { rhs - lhs };
                let lim_lo = if c_lo < 2 { c_lo } else { 2 };
                let lim_hi = if c_hi < 2 { c_hi } else { 2 };
                let base = if s_abs <= lim_lo {
                    2
                } else if s_abs <= lim_hi {
                    1
                } else {
                    0
                };
                // Only mass-consistent shifts are candidate evidence
                // (architecture §2.4): a peak whose shift fails the rule
                // contributes no count, no record and no status, so a
                // candidate with only unsupported shifts stays at status 0,
                // count 0 with an all-zero record area. Records keep first-E
                // order among the qualifying peaks.
                if base > 0 {
                    if base > best {
                        best = base;
                    }
                    count += 1;
                    if recorded < EVIDENCE_SLOTS {
                        let w: u32 = ebase + 2 + recorded * 4;
                        evidence[w as usize] = p;
                        evidence[(w + 1) as usize] = bq;
                        evidence[(w + 2) as usize] = if ge == 1 {
                            LANE_BIAS + s_abs
                        } else {
                            LANE_BIAS - s_abs
                        };
                        evidence[(w + 3) as usize] = ion[(hb + 11) as usize];
                        recorded += 1;
                    }
                }
            }
        }
        p += 1;
    }
    let mut st = best;
    if incompl == 1 {
        st += EVIDENCE_SUPPORT_INCOMPLETE;
    }
    evidence[ebase as usize] = st;
    evidence[(ebase + 1) as usize] = count;
}

/// Match block with probability-ranked record choice (architecture §2.4):
/// the kernel twin's per-trajectory comparison that keeps the at most `E = 4`
/// qualifying peaks of largest assignment log-probability (ties by smaller
/// peak position) instead of the first four.
///
/// `log_prob` is `[B, F, N, J + 1]` flat assignment log-probabilities (the
/// head's `log_prob`, class `J` is unassigned); for the peak's chosen
/// hypothesis `bq` the record's probability is
/// `log_prob[((b * f_dim + slot) * n + p) * (j + 1) + bq]`. All other
/// arguments and the status/count semantics match [`evidence_match`]; only
/// the recorded set differs. Every element written; an invalid lane writes a
/// zeroed row. Per-lane work is `O(N * E)` with `E = 4` (at most `512`
/// comparisons at `N = 128`), never quadratic in an unbounded window.
#[allow(clippy::too_many_arguments)]
pub fn evidence_match_scored(
    hc: [u32; 9],
    hg: u32,
    c_lo: u32,
    c_hi: u32,
    ha_pos: u32,
    ha_neg: u32,
    ion: &[u32],
    ion_meta: &[u32],
    log_prob: &[f32],
    b: u32,
    slot: u32,
    f_dim: u32,
    n: u32,
    j: u32,
    ok: u32,
    evidence: &mut [u32],
    ebase: u32,
) {
    // Full-buffer addresses in u32 exactly as the kernel does (cast to
    // `usize` only at the index expression).
    let mut zi: u32 = 0;
    while zi < EVIDENCE_ROW_WORDS as u32 {
        evidence[(ebase + zi) as usize] = 0;
        zi += 1;
    }
    let valid = if ok == 1 && slot < f_dim { 1 } else { 0 };
    let width: u32 = j + 1;
    let mut count = 0u32;
    let mut best = 0u32;
    let mut incompl = 0u32;
    // Top-E by (log-prob desc, peak asc), kept sorted descending.
    let mut tp: [u32; 4] = [0; 4];
    let mut th: [u32; 4] = [0; 4];
    let mut ts_ob: [u32; 4] = [0; 4];
    let mut tr_ob: [u32; 4] = [0; 4];
    let mut tlp: [f32; 4] = [f32::NEG_INFINITY; 4];
    let mut kept_e = 0u32;
    let mut p = 0u32;
    while p < n {
        if valid == 1 {
            let pos: u32 = (b * f_dim + slot) * n + p;
            let ibase: u32 = pos * j * 12;
            let mbase: u32 = pos * 4;
            if ion_meta[(mbase + 3) as usize] != 0 {
                incompl = 1;
            }
            let kept_n = ion_meta[(mbase + 2) as usize];
            let mut bq = 0u32;
            let mut bres = 0u32;
            let mut found = 0u32;
            let mut q = 0u32;
            while q < kept_n {
                let hb: u32 = ibase + q * 12;
                let mut eq = 1u32;
                if ion[hb as usize] != hc[0] {
                    eq = 0;
                }
                if ion[(hb + 2) as usize] != hc[1] {
                    eq = 0;
                }
                if ion[(hb + 3) as usize] != hc[2] {
                    eq = 0;
                }
                if ion[(hb + 4) as usize] != hc[3] {
                    eq = 0;
                }
                if ion[(hb + 5) as usize] != hc[4] {
                    eq = 0;
                }
                if ion[(hb + 6) as usize] != hc[5] {
                    eq = 0;
                }
                if ion[(hb + 7) as usize] != hc[6] {
                    eq = 0;
                }
                if ion[(hb + 8) as usize] != hc[7] {
                    eq = 0;
                }
                if ion[(hb + 9) as usize] != hc[8] {
                    eq = 0;
                }
                if eq == 1 {
                    let ob = ion[(hb + 11) as usize];
                    let rabs = if ob >= LANE_BIAS {
                        ob - LANE_BIAS
                    } else {
                        LANE_BIAS - ob
                    };
                    if found == 0 || rabs < bres {
                        bq = q;
                        bres = rabs;
                        found = 1;
                    }
                }
                q += 1;
            }
            if found == 1 {
                let hb: u32 = ibase + bq * 12;
                let hh = ion[(hb + 1) as usize];
                let lhs = hh + ha_neg;
                let rhs = hg + ha_pos;
                let ge = if lhs >= rhs { 1 } else { 0 };
                let s_abs = if ge == 1 { lhs - rhs } else { rhs - lhs };
                let lim_lo = if c_lo < 2 { c_lo } else { 2 };
                let lim_hi = if c_hi < 2 { c_hi } else { 2 };
                let base = if s_abs <= lim_lo {
                    2
                } else if s_abs <= lim_hi {
                    1
                } else {
                    0
                };
                if base > 0 {
                    if base > best {
                        best = base;
                    }
                    count += 1;
                    // Log-probability of the chosen hypothesis (unconditional
                    // index; masked by `found`/`base` at use).
                    let lp_idx: u32 = pos * width + bq;
                    let mut lp = f32::NEG_INFINITY;
                    if (lp_idx as usize) < log_prob.len() {
                        lp = log_prob[lp_idx as usize];
                    }
                    if lp.is_nan() {
                        lp = f32::NEG_INFINITY;
                    }
                    let shift_ob = if ge == 1 {
                        LANE_BIAS + s_abs
                    } else {
                        LANE_BIAS - s_abs
                    };
                    let resid_ob = ion[(hb + 11) as usize];
                    // Insert into the sorted top-E (strictly greater displaces;
                    // ties keep the smaller peak position; a candidate larger
                    // than none appends at `kept_e` while records remain).
                    let mut at = 4u32;
                    let mut i = 0u32;
                    while i < kept_e {
                        if lp > tlp[i as usize] {
                            at = i;
                            break;
                        }
                        i += 1;
                    }
                    if at == 4 && kept_e < 4 {
                        at = kept_e;
                    }
                    if at < 4 {
                        if kept_e < 4 {
                            kept_e += 1;
                        }
                        let mut k = kept_e - 1;
                        while k > at {
                            tp[k as usize] = tp[(k - 1) as usize];
                            th[k as usize] = th[(k - 1) as usize];
                            ts_ob[k as usize] = ts_ob[(k - 1) as usize];
                            tr_ob[k as usize] = tr_ob[(k - 1) as usize];
                            tlp[k as usize] = tlp[(k - 1) as usize];
                            k -= 1;
                        }
                        tp[at as usize] = p;
                        th[at as usize] = bq;
                        ts_ob[at as usize] = shift_ob;
                        tr_ob[at as usize] = resid_ob;
                        tlp[at as usize] = lp;
                    }
                }
            }
        }
        p += 1;
    }
    // Write the ranked records (already sorted by probability).
    let mut r = 0u32;
    while r < kept_e {
        let w: u32 = ebase + 2 + r * 4;
        evidence[w as usize] = tp[r as usize];
        evidence[(w + 1) as usize] = th[r as usize];
        evidence[(w + 2) as usize] = ts_ob[r as usize];
        evidence[(w + 3) as usize] = tr_ob[r as usize];
        r += 1;
    }
    let mut st = best;
    if incompl == 1 {
        st += EVIDENCE_SUPPORT_INCOMPLETE;
    }
    evidence[ebase as usize] = st;
    evidence[(ebase + 1) as usize] = count;
}

/// Full-buffer trajectory lane of `ms2_ion_evidence`: the kernel twin.
///
/// `actions` is `[R, S]` flat trajectory rows (`S = steps * 4 + atoms + 4`:
/// `steps` token words, `atoms` open-valence words in trace order, then
/// length, status and two reserved words); `traj_slot` is `[R, 2]` flat
/// (formula slot, adduct id). The lane decodes the finished trace — atom
/// types to heavy counts `u_g` and parent hydrogens `H(g)` through
/// [`atom_fields_u32`], open valence per atom from the record — and runs
/// [`evidence_match`] over the hypotheses of its own formula slot, writing
/// the `[2 + 4E]` flat `evidence` row. An unfinished trajectory, an unknown
/// adduct or a slot at or beyond `f_dim` writes a zeroed row (status 0, no
/// records).
// Manual range check and ceiling division: the kernel has neither
// `RangeInclusive::contains` nor `is_multiple_of`.
#[allow(
    clippy::too_many_arguments,
    clippy::manual_range_contains,
    clippy::manual_is_multiple_of
)]
pub fn evidence_lane(
    actions: &[u32],
    traj_slot: &[u32],
    ion: &[u32],
    ion_meta: &[u32],
    r: u32,
    steps: u32,
    atoms_cap: u32,
    k_dim: u32,
    f_dim: u32,
    n: u32,
    j: u32,
    evidence: &mut [u32],
) {
    let b = r / k_dim;
    let stride = steps * 4 + atoms_cap + 4;
    let abase = r * stride;
    let tbase = r * 2;
    let ebase = r * EVIDENCE_ROW_WORDS as u32;
    let len_field = steps * 4 + atoms_cap;
    let length = actions[(abase + len_field) as usize];
    let status_w = actions[(abase + len_field + 1) as usize];
    // Bit 0 is `finished` (contract `candidate_status::FINISHED`).
    let finished = if status_w & 1 != 0 { 1 } else { 0 };
    let slot = traj_slot[tbase as usize];
    let adduct_id = traj_slot[(tbase + 1) as usize];
    let mut ha_pos = 0u32;
    let mut ha_neg = 0u32;
    let mut adduct_ok = 0u32;
    if adduct_id == 1 {
        ha_pos = 1;
        adduct_ok = 1;
    }
    if adduct_id == 2 {
        ha_neg = 1;
        adduct_ok = 1;
    }
    // Trace decode: heavy counts, parent hydrogens and the open-valence
    // range. Atoms past the cap are skipped; types outside 1–17 are not
    // atoms of the V0 domain and are skipped too.
    let mut hc = [0u32; 9];
    let mut hg = 0u32;
    let mut c_lo = 0u32;
    let mut c_hi = 0u32;
    let mut n_atoms = 0u32;
    let open_base = abase + steps * 4;
    let mut s = 0u32;
    while s < length {
        let tok = abase + s * 4;
        let kind = actions[tok as usize];
        let ty = actions[(tok + 1) as usize];
        if kind == u32::from(ADD_ATOM) && ty >= 1 && ty <= 17 && n_atoms < atoms_cap {
            let fields = atom_fields_u32(ty);
            let el = fields / 65536;
            let ph = fields - el * 65536;
            if el == 0 {
                hc[0] += 1;
            }
            if el == 2 {
                hc[1] += 1;
            }
            if el == 3 {
                hc[2] += 1;
            }
            if el == 4 {
                hc[3] += 1;
            }
            if el == 5 {
                hc[4] += 1;
            }
            if el == 6 {
                hc[5] += 1;
            }
            if el == 7 {
                hc[6] += 1;
            }
            if el == 8 {
                hc[7] += 1;
            }
            if el == 9 {
                hc[8] += 1;
            }
            hg += ph;
            let o = actions[(open_base + n_atoms) as usize];
            c_lo += o / 3 + if o % 3 != 0 { 1 } else { 0 };
            c_hi += o;
            n_atoms += 1;
        }
        s += 1;
    }
    let ok = if finished == 1 && adduct_ok == 1 { 1 } else { 0 };
    evidence_match(
        hc, hg, c_lo, c_hi, ha_pos, ha_neg, ion, ion_meta, b, slot, f_dim, n, j, ok, evidence,
        ebase,
    );
}

/// Full-buffer trajectory lane with probability-ranked records: the kernel
/// twin of the scored evidence kernel.
///
/// Same decoding as [`evidence_lane`], but runs [`evidence_match_scored`]
/// over the hypotheses with the assignment `log_prob [B, F, N, J + 1]` flat
/// log-probabilities, keeping the at most `E = 4` qualifying peaks of
/// largest probability (ties by smaller peak position).
#[allow(
    clippy::too_many_arguments,
    clippy::manual_range_contains,
    clippy::manual_is_multiple_of
)]
#[allow(clippy::too_many_arguments)]
pub fn evidence_lane_scored(
    actions: &[u32],
    traj_slot: &[u32],
    ion: &[u32],
    ion_meta: &[u32],
    log_prob: &[f32],
    r: u32,
    steps: u32,
    atoms_cap: u32,
    k_dim: u32,
    f_dim: u32,
    n: u32,
    j: u32,
    evidence: &mut [u32],
) {
    let b = r / k_dim;
    let stride = steps * 4 + atoms_cap + 4;
    let abase = r * stride;
    let tbase = r * 2;
    let ebase = r * EVIDENCE_ROW_WORDS as u32;
    let len_field = steps * 4 + atoms_cap;
    let length = actions[(abase + len_field) as usize];
    let status_w = actions[(abase + len_field + 1) as usize];
    let finished = if status_w & 1 != 0 { 1 } else { 0 };
    let slot = traj_slot[tbase as usize];
    let adduct_id = traj_slot[(tbase + 1) as usize];
    let mut ha_pos = 0u32;
    let mut ha_neg = 0u32;
    let mut adduct_ok = 0u32;
    if adduct_id == 1 {
        ha_pos = 1;
        adduct_ok = 1;
    }
    if adduct_id == 2 {
        ha_neg = 1;
        adduct_ok = 1;
    }
    let mut hc = [0u32; 9];
    let mut hg = 0u32;
    let mut c_lo = 0u32;
    let mut c_hi = 0u32;
    let mut n_atoms = 0u32;
    let open_base = abase + steps * 4;
    let mut s = 0u32;
    while s < length {
        let tok = abase + s * 4;
        let kind = actions[tok as usize];
        let ty = actions[(tok + 1) as usize];
        if kind == u32::from(ADD_ATOM) && ty >= 1 && ty <= 17 && n_atoms < atoms_cap {
            let fields = atom_fields_u32(ty);
            let el = fields / 65536;
            let ph = fields - el * 65536;
            if el == 0 {
                hc[0] += 1;
            }
            if el == 2 {
                hc[1] += 1;
            }
            if el == 3 {
                hc[2] += 1;
            }
            if el == 4 {
                hc[3] += 1;
            }
            if el == 5 {
                hc[4] += 1;
            }
            if el == 6 {
                hc[5] += 1;
            }
            if el == 7 {
                hc[6] += 1;
            }
            if el == 8 {
                hc[7] += 1;
            }
            if el == 9 {
                hc[8] += 1;
            }
            hg += ph;
            let o = actions[(open_base + n_atoms) as usize];
            c_lo += o / 3 + if o % 3 != 0 { 1 } else { 0 };
            c_hi += o;
            n_atoms += 1;
        }
        s += 1;
    }
    let ok = if finished == 1 && adduct_ok == 1 { 1 } else { 0 };
    evidence_match_scored(
        hc, hg, c_lo, c_hi, ha_pos, ha_neg, ion, ion_meta, log_prob, b, slot, f_dim, n, j,
        ok, evidence, ebase,
    );
}

/// Per-trajectory evidence features `(max log-prob, min |residual| / tol)`:
/// the kernel twin for the reranker's future `evidence_f [B*K, 2]` input.
///
/// `evidence` is `[R, 18]` flat (word 0 status, word 1 count, then records of
/// peak position, hypothesis, shift_ob, residual_ob); `log_prob` is
/// `[B, F, N, J + 1]` flat assignment log-probabilities; `traj_slot` is
/// `[R, 2]` (slot, adduct); `kept` is `[B, N, 3]` (raw, m/z, reverse);
/// `meta` is `[B, 8]` spectrum words (fragment tolerance ppm tenths at word
/// 4). Writes `evidence_f` row `(largest assignment log-probability among
/// the trajectory's evidence records, smallest |residual| in units of the
/// fragment tolerance at that peak; `(0, 1)` when there is no evidence).
/// Per-lane work is `O(E)` with `E = 4`.
#[allow(clippy::too_many_arguments)]
pub fn evidence_features_lane(
    evidence: &[u32],
    log_prob: &[f32],
    traj_slot: &[u32],
    kept: &[u32],
    meta: &[u32],
    r: u32,
    k_dim: u32,
    f_dim: u32,
    n: u32,
    j: u32,
    evidence_f: &mut [f32],
) {
    // Full-buffer addresses in u32 exactly as the kernel does (cast to
    // `usize` only at the index expression).
    let ebase: u32 = r * EVIDENCE_ROW_WORDS as u32;
    let fbase: u32 = r * 2;
    let b = r / k_dim;
    let tbase: u32 = r * 2;
    let slot = traj_slot[tbase as usize];
    let width: u32 = j + 1;
    let count = evidence[(ebase + 1) as usize];
    let mut max_lp = f32::NEG_INFINITY;
    let mut min_unit = f32::INFINITY;
    let mut q = 0u32;
    while q < EVIDENCE_SLOTS && q < count {
        let w: u32 = ebase + 2 + q * 4;
        let p = evidence[w as usize];
        let bq = evidence[(w + 1) as usize];
        let resid_ob = evidence[(w + 3) as usize];
        // Log-probability lookup (unconditional index, masked by count at use
        // here: `q < count` guards, and the buffer holds the written rows).
        let pos: u32 = (b * f_dim + slot) * n + p;
        let lp_idx: u32 = pos * width + bq;
        let mut lp = f32::NEG_INFINITY;
        if (lp_idx as usize) < log_prob.len() {
            lp = log_prob[lp_idx as usize];
        }
        if lp.is_nan() {
            lp = f32::NEG_INFINITY;
        }
        if lp > max_lp {
            max_lp = lp;
        }
        // Tolerance at this peak by the `u32` algorithm, then units.
        let kept_base: u32 = (b * n + p) * 3;
        let mut mz = 0u32;
        if ((kept_base + 1) as usize) < kept.len() {
            mz = kept[(kept_base + 1) as usize];
        }
        let mut ppm = 100u32;
        let meta_base: u32 = b * 8;
        if ((meta_base + 4) as usize) < meta.len() {
            ppm = meta[(meta_base + 4) as usize];
        }
        let hi = mz / 10_000;
        let lo = mz % 10_000;
        let qq = hi.wrapping_mul(ppm);
        let tol = qq / 1000 + ((qq % 1000) * 10_000 + lo.wrapping_mul(ppm)) / 10_000_000;
        let tol_f = if tol == 0 { 1.0f32 } else { tol as f32 };
        let rabs = if resid_ob >= LANE_BIAS {
            resid_ob - LANE_BIAS
        } else {
            LANE_BIAS - resid_ob
        } as f32;
        let unit = rabs / tol_f;
        if unit < min_unit {
            min_unit = unit;
        }
        q += 1;
    }
    if count == 0 {
        evidence_f[fbase as usize] = 0.0;
        evidence_f[(fbase + 1) as usize] = 1.0;
    } else {
        let mut ml = max_lp;
        if !ml.is_finite() {
            ml = 0.0;
        }
        let mut mu = min_unit;
        if !mu.is_finite() {
            mu = 1.0;
        }
        evidence_f[fbase as usize] = ml;
        evidence_f[(fbase + 1) as usize] = mu;
    }
}

/// Host `[R, 2]` trajectory slots from `traj_alloc [B, K, 12]` and
/// `meta [B, 8]`: word 0 is the retained formula slot, word 1 the adduct id
/// (word 3 of the spectrum meta). The twin of the `ion_traj_slot` lane.
pub fn traj_slot_host(traj_alloc: &[u32], meta: &[u32], batch: usize, k: usize) -> Vec<u32> {
    let mut out = vec![0u32; batch * k * 2];
    for b in 0..batch {
        for kk in 0..k {
            let r = b * k + kk;
            out[r * 2] = traj_alloc
                .get((r * 12) as usize)
                .copied()
                .unwrap_or(u32::MAX);
            out[r * 2 + 1] = meta.get(b * 8 + 3).copied().unwrap_or(0);
        }
    }
    out
}

/// Packed mass-consistency evidence of a candidate graph (spec §2.4).
///
/// The candidate is described by its heavy counts (the hydrogen slot of
/// `candidate_heavy` is ignored), its parent hydrogens `H(g)` and its open
/// valence per atom `o_a`. With bond orders up to 3, `c_lo = sum ceil(o_a /
/// 3)` and `c_hi = sum o_a` bound the boundary-bond count of any partition.
/// A kept hypothesis `(u, h)` of peak `p` with `u == u_g` gives the shift `s
/// = h − H(g) − h_a` (with several matches the smallest `|residual|` wins,
/// ties by smaller hypothesis index); the candidate is mass-consistent with
/// `p` when `|s| <= min(c_hi, 2)` and mass-consistent for every boundary
/// count when `|s| <= min(c_lo, 2)`. Neither says that a compatible parent
/// exists. An unknown adduct yields a zeroed row.
///
/// Returns the packed [`EvidencePacked`] row: the maximum base status over
/// all mass-consistent matched peaks with [`EVIDENCE_SUPPORT_INCOMPLETE`]
/// when some compared peak's support is incomplete, the total qualifying
/// match count, then the first [`EVIDENCE_SLOTS`] qualifying matches in peak
/// order (peaks whose shift fails the mass-consistency rule contribute
/// nothing). `j` is the hypothesis capacity `J` of the packed rows.
// Manual ceiling division: the kernel has no `is_multiple_of`.
#[allow(clippy::manual_is_multiple_of)]
pub fn evidence_status(
    candidate_heavy: &Composition,
    candidate_parent_h: u32,
    open_valence: &[u8],
    adduct_id: u16,
    hypotheses_of_peaks: &[IonAssignment],
    j: usize,
) -> EvidencePacked {
    let mut hc = [0u32; 9];
    for (slot, &e) in HEAVY.iter().enumerate() {
        hc[slot] = u32::from(candidate_heavy[e]);
    }
    let mut c_lo = 0u32;
    let mut c_hi = 0u32;
    for &o in open_valence {
        let ou = u32::from(o);
        c_lo += ou / 3 + u32::from(ou % 3 != 0);
        c_hi += ou;
    }
    let (ha_pos, ha_neg, ok) = match adduct_id {
        1 => (1, 0, 1),
        2 => (0, 1, 1),
        _ => (0, 0, 0),
    };
    let n = hypotheses_of_peaks.len();
    let mut ion = vec![0u32; n * j * 12];
    let mut ion_meta = vec![0u32; n * 4];
    for (p, assign) in hypotheses_of_peaks.iter().enumerate() {
        let take = assign.kept.len().min(j);
        for (q, hyp) in assign.kept.iter().take(take).enumerate() {
            let w = (p * j + q) * 12;
            for e in 0..10 {
                ion[w + e] = u32::from(hyp.counts[e]);
            }
            ion[w + 10] = hyp.mass;
            ion[w + 11] = (hyp.residual as u32).wrapping_add(LANE_BIAS);
        }
        ion_meta[p * 4] = assign.accepted;
        ion_meta[p * 4 + 1] = assign.ambiguous;
        ion_meta[p * 4 + 2] = take as u32;
        ion_meta[p * 4 + 3] = assign.status;
    }
    let mut row = vec![0u32; EVIDENCE_ROW_WORDS];
    evidence_match(
        hc,
        candidate_parent_h,
        c_lo,
        c_hi,
        ha_pos,
        ha_neg,
        &ion,
        &ion_meta,
        0,
        0,
        1,
        n as u32,
        j as u32,
        ok,
        &mut row,
        0,
    );
    let mut out = EvidencePacked {
        status: row[0],
        count: row[1],
        peak: [0; EVIDENCE_SLOTS as usize],
        hypothesis: [0; EVIDENCE_SLOTS as usize],
        shift_ob: [0; EVIDENCE_SLOTS as usize],
        residual_ob: [0; EVIDENCE_SLOTS as usize],
    };
    for q in 0..EVIDENCE_SLOTS as usize {
        out.peak[q] = row[2 + q * 4];
        out.hypothesis[q] = row[2 + q * 4 + 1];
        out.shift_ob[q] = row[2 + q * 4 + 2];
        out.residual_ob[q] = row[2 + q * 4 + 3];
    }
    out
}

/// Whether the contract §4.3 mapping supports shift `s` at boundary-bond
/// count `c`: `|s| <= min(c, 2)` (P6.8).
pub fn mapping_is_supported(c: u32, s: i32) -> bool {
    s.unsigned_abs() <= c.min(2)
}

/// The ion of a labeled embedding by the contract formula (P6.8 test hook).
///
/// The embedding carries its atom set and its known boundary-bond count
/// `c(g)`; the composition comes from the induced subgraph (parent
/// hydrogens). Returns `Ok(None)` when the mapping is unsupported
/// ([`mapping_is_supported`]) or the ion's hydrogen count would be negative
/// (or exceed `u16`); otherwise the ion composition (hydrogen is the ion's
/// own count) with its m/z and bound. The `h <= u16::MAX` pre-check keeps the
/// `u16` cast inside [`ion`] exact.
pub fn embedding_ion(
    parent: &MolGraph,
    embedding: &Embedding,
    adduct_id: u16,
    shift: i32,
) -> Result<Option<(Composition, Ion)>> {
    if !mapping_is_supported(embedding.boundary as u32, shift) {
        return Ok(None);
    }
    let sub = parent.induced(&embedding.atoms)?;
    let comp = sub.composition();
    let a = adduct(adduct_id).ok_or_else(|| {
        Error::Unsupported(format!("embedding_ion: unknown adduct id {adduct_id}"))
    })?;
    let h = i64::from(comp[HYDROGEN]) + i64::from(a.hydrogens) + i64::from(shift);
    if h < 0 || h > i64::from(u16::MAX) {
        return Ok(None);
    }
    let Some(hyp) = ion(&comp, adduct_id, shift)? else {
        return Ok(None);
    };
    let mut ion_c = comp;
    ion_c[HYDROGEN] = h as u16;
    Ok(Some((ion_c, hyp)))
}
