//! GPU kernels for fragment-ion assignment (K2).
//!
//! Architecture `docs/MS2_V1_ARCHITECTURE.md` §2 (assignment, labels,
//! evidence). Each kernel is a copy of its host twin's lane in
//! `crate::models::ms2::ion` ([`ion_assign_lane`](crate::models::ms2::ion::ion_assign_lane),
//! [`label_mask_lane`](crate::models::ms2::ion::label_mask_lane),
//! [`evidence_lane`](crate::models::ms2::ion::evidence_lane) with
//! [`evidence_match`](crate::models::ms2::ion::evidence_match)); the twins
//! were written first in the kernel-expressible form (full buffers with
//! explicit indices, `u32` loop counters from literals, no early `return`
//! inside loops), so both stay identical. Integer buffers are [`IdTensor`]
//! (`u32`); every launch goes through [`crate::backend::launch_1d_spans`]
//! with one lane per output item; shapes are checked to
//! [`crate::error::Error::Shape`] before any launch.
//!
//! CubeCL 0.10 deltas from the twin spelling (same arithmetic, mechanical):
//!
//! * No fixed-size local arrays as kernel values: the visit's nine lane
//!   registers are scalars (`d0..d8`, `r0..r8`), selected by slot with
//!   if-chains where the twin loops over its register arrays. Each block
//!   cites the twin source.
//! * No `wrapping_*` methods, `abs_diff`, `div_ceil`, `is_multiple_of` or
//!   `u32::MAX`: plain wrapping device ops, manual absolute differences,
//!   quotient/remainder ceilings and `4294967295u32`. `saturating_add` is a
//!   wrap-detect-and-clamp sequence with the same value.
//! * No `!` on booleans: explicit `== false` comparisons. Booleans
//!   initialised from literals become `u32` 0/1 flags (the macro cannot infer
//!   a literal `true`/`false`); comparison-derived booleans stay as-is.
//! * Loop-carried variables start from literals or buffer loads, never a
//!   plain copy of a scalar argument.
//!
//! At most 6 arrays per kernel. Every output element is written by exactly
//! one lane; padding gets an explicit value; selection is by comparison,
//! never by multiplying with a mask. A buffer load inside an `if` is avoided
//! on hot paths: indices are computed unconditionally and the guard applies
//! where the value is used.
//!
//! [`ms2_ion_assign`] binds 6 arrays rather than the 5 of the spec sketch:
//! the spectrum's m/z uncertainty `U` lives only on the host (contract
//! `Spectra`; `meta[.., 2]` is the precursor uncertainty), so it travels in
//! the small per-spectrum `spec [B, 2]` buffer (uncertainty, reserved).

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
use crate::models::ms2::ion::EVIDENCE_ROW_WORDS;
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
fn ms2_ion_heavy_mass(slot: u32) -> u32 {
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
fn ms2_ion_heavy_res(slot: u32) -> u32 {
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

/// Packed atom-type fields `(element << 16) | parent_hydrogens` of V0 atom
/// type id `ty` (1–17). Copy of
/// [`crate::models::ms2::ion::atom_fields_u32`].
#[allow(unused_assignments)]
#[cube]
fn ms2_ion_atom_fields(ty: u32) -> u32 {
    let mut fields: u32 = 0u32;
    if ty == 1u32 {
        fields = 0u32;
    }
    if ty == 2u32 {
        fields = 1u32;
    }
    if ty == 3u32 {
        fields = 2u32;
    }
    if ty == 4u32 {
        fields = 3u32;
    }
    if ty == 5u32 {
        fields = 2u32 * 65536u32;
    }
    if ty == 6u32 {
        fields = 2u32 * 65536u32 + 1u32;
    }
    if ty == 7u32 {
        fields = 2u32 * 65536u32 + 2u32;
    }
    if ty == 8u32 {
        fields = 3u32 * 65536u32;
    }
    if ty == 9u32 {
        fields = 3u32 * 65536u32 + 1u32;
    }
    if ty == 10u32 {
        fields = 4u32 * 65536u32;
    }
    if ty == 11u32 {
        fields = 7u32 * 65536u32;
    }
    if ty == 12u32 {
        fields = 8u32 * 65536u32;
    }
    if ty == 13u32 {
        fields = 6u32 * 65536u32;
    }
    if ty == 14u32 {
        fields = 6u32 * 65536u32 + 1u32;
    }
    if ty == 15u32 {
        fields = 6u32 * 65536u32;
    }
    if ty == 16u32 {
        fields = 5u32 * 65536u32;
    }
    if ty == 17u32 {
        fields = 9u32 * 65536u32;
    }
    fields
}

/// Overflow-free verdict rule (contract §5): accept exactly when
/// `r <= tol` and `error <= tol − r`, reject exactly when `r > tol` and
/// `r − tol > error`, else ambiguity, with `r = |observed − computed|`.
/// Copy of [`crate::models::ms2::ion::lane_verdict_u32`] with manual
/// absolute differences and plain wrapping ops.
#[allow(unused_assignments)]
#[cube]
fn ms2_ion_verdict(observed: u32, computed: u32, error: u32, tolerance: u32) -> u32 {
    let mut r: u32 = 0u32;
    if observed >= computed {
        r = observed - computed;
    } else {
        r = computed - observed;
    }
    let in_tol = r <= tolerance;
    // Wraps when `r > tolerance`, exactly like the twin's `wrapping_sub`;
    // the value is unused then (`hit` is false through `in_tol`).
    let tol_minus_r = tolerance - r;
    let hit = in_tol && error <= tol_minus_r;
    let over = r > tolerance;
    let r_minus_tol = r - tolerance;
    let miss = over && r_minus_tol > error;
    let mut v: u32 = 2u32;
    if hit {
        v = 1u32;
    }
    if hit == false && miss {
        v = 0u32;
    }
    v
}

// ---------------------------------------------------------------------------
// `ms2_ion_assign`: lane per `(b, f, p)`
// ---------------------------------------------------------------------------

/// Lane per `(b, f, p)` of [`ion_assign`]; a copy of
/// [`crate::models::ms2::ion::ion_assign_lane`] over `Array`s.
///
/// The visit block is [`crate::models::ms2::ion::ion_lane_visit`] with the
/// nine lane registers scalar-expanded (`d0..d8` digits, `r0..r8` radices);
/// the budget block is [`crate::models::ms2::ion::lane_visits_u32`] with the
/// radix selected by slot through an if-chain. All other statements are
/// shared verbatim with the twin (modulo the spelling deltas above).
#[allow(clippy::too_many_arguments)]
#[allow(unused_assignments)]
#[cube(launch_unchecked)]
fn ms2_ion_assign_kernel(
    top_counts: &Array<u32>,
    kept: &Array<u32>,
    meta: &Array<u32>,
    spec: &Array<u32>,
    ion: &mut Array<u32>,
    ion_meta: &mut Array<u32>,
    f_dim: u32,
    n: u32,
    j: u32,
    work_max: u32,
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
        let bf = lane / n;
        let p = lane % n;
        let b = bf / f_dim;
        let f = bf % f_dim;
        let tc_base = (b * f_dim + f) * 10u32;
        let kept_base = (b * n + p) * 3u32;
        let meta_base = b * 8u32;
        let spec_base = b * 2u32;
        let ion_base = lane * j * 12u32;
        let im_base = lane * 4u32;
        // Every output element is written: zero both rows first.
        let mut zi: u32 = 0u32;
        while zi < j * 12u32 {
            ion[(ion_base + zi) as usize] = 0u32;
            zi += 1;
        }
        ion_meta[im_base as usize] = 0u32;
        ion_meta[(im_base + 1u32) as usize] = 0u32;
        ion_meta[(im_base + 2u32) as usize] = 0u32;
        ion_meta[(im_base + 3u32) as usize] = 0u32;
        // Parent counts and the empty-formula gate.
        let mut all_zero: u32 = 1u32;
        let mut e: u32 = 0u32;
        while e < 10u32 {
            if top_counts[(tc_base + e) as usize] != 0u32 {
                all_zero = 0u32;
            }
            e += 1;
        }
        let peak_count = meta[meta_base as usize];
        let peak_mz = kept[(kept_base + 1u32) as usize];
        let adduct_id = meta[(meta_base + 3u32) as usize];
        let ppm = meta[(meta_base + 4u32) as usize];
        let u_unc = spec[spec_base as usize];
        let known_adduct = adduct_id == 1u32 || adduct_id == 2u32;
        // t = mz + z_a * m_e: +549 for [M+H]+, −549 for [M-H]-.
        let mut t: u32 = 0u32;
        let mut t_ok: u32 = 0u32;
        if adduct_id == 1u32 && peak_mz <= 4294967295u32 - 549u32 {
            t = peak_mz + 549u32;
            t_ok = 1u32;
        }
        if adduct_id == 2u32 && peak_mz >= 549u32 {
            t = peak_mz - 549u32;
            t_ok = 1u32;
        }
        // Fragment tolerance by the `u32` algorithm of `tolerance_u32`.
        let hi = peak_mz / 10000u32;
        let lo = peak_mz % 10000u32;
        let q = hi * ppm;
        let tol_p = q / 1000u32 + ((q % 1000u32) * 10000u32 + lo * ppm) / 10000000u32;
        // E_ion upper bound over the parent residuals (+3 H + electron).
        // Element words are C, N, O, F, P, S, Cl, Br, I (twin `HEAVY` order);
        // the parent hydrogen term joins below, as in the twin.
        let mut nda: u32 = 0u32;
        nda += top_counts[tc_base as usize] * ms2_ion_heavy_res(0u32);
        nda += top_counts[(tc_base + 2u32) as usize] * ms2_ion_heavy_res(1u32);
        nda += top_counts[(tc_base + 3u32) as usize] * ms2_ion_heavy_res(2u32);
        nda += top_counts[(tc_base + 4u32) as usize] * ms2_ion_heavy_res(3u32);
        nda += top_counts[(tc_base + 5u32) as usize] * ms2_ion_heavy_res(4u32);
        nda += top_counts[(tc_base + 6u32) as usize] * ms2_ion_heavy_res(5u32);
        nda += top_counts[(tc_base + 7u32) as usize] * ms2_ion_heavy_res(6u32);
        nda += top_counts[(tc_base + 8u32) as usize] * ms2_ion_heavy_res(7u32);
        nda += top_counts[(tc_base + 9u32) as usize] * ms2_ion_heavy_res(8u32);
        nda += top_counts[(tc_base + 1u32) as usize] * 33u32;
        nda += 3u32 * 33u32 + 421u32;
        let mut e_ceil: u32 = 0u32;
        if nda % 1000u32 != 0u32 {
            e_ceil = 1u32;
        }
        let e_ion = nda / 1000u32 + e_ceil;
        // half_p by saturating additions (wrap-detect-and-clamp).
        let mut half_step = tol_p + u_unc;
        if half_step < tol_p {
            half_step = 4294967295u32;
        }
        let mut half_p = half_step + e_ion;
        if half_p < half_step {
            half_p = 4294967295u32;
        }
        // Scope restriction: a wider window makes the peak ion_unavailable.
        let mut searchable: u32 = 0u32;
        if known_adduct && t_ok == 1u32 && all_zero == 0u32 {
            searchable = 1u32;
        }
        if peak_mz == 0u32 {
            searchable = 0u32;
        }
        if p >= peak_count {
            searchable = 0u32;
        }
        if u_unc == 4294967295u32 {
            searchable = 0u32;
        }
        if half_p > 1007825u32 {
            searchable = 0u32;
        }
        if searchable == 1u32 {
            // Lane registers: parent counts and radices in HEAVY order.
            let c0 = top_counts[tc_base as usize];
            let c1 = top_counts[(tc_base + 1u32) as usize];
            let c2 = top_counts[(tc_base + 2u32) as usize];
            let c3 = top_counts[(tc_base + 3u32) as usize];
            let c4 = top_counts[(tc_base + 4u32) as usize];
            let c5 = top_counts[(tc_base + 5u32) as usize];
            let c6 = top_counts[(tc_base + 6u32) as usize];
            let c7 = top_counts[(tc_base + 7u32) as usize];
            let c8 = top_counts[(tc_base + 8u32) as usize];
            let c9 = top_counts[(tc_base + 9u32) as usize];
            let r0 = c0 + 1u32;
            let r1 = c2 + 1u32;
            let r2 = c3 + 1u32;
            let r3 = c4 + 1u32;
            let r4 = c5 + 1u32;
            let r5 = c6 + 1u32;
            let r6 = c7 + 1u32;
            let r7 = c8 + 1u32;
            let r8 = c9 + 1u32;
            // Guarded visit budget (`lane_visits_u32` with the radix
            // selected by slot through an if-chain).
            let mut partial: u32 = 1u32;
            let mut cut: u32 = 0u32;
            let mut big: u32 = 0u32;
            let mut ri: u32 = 0u32;
            while ri < 9u32 {
                let mut r = r0;
                if ri == 1u32 {
                    r = r1;
                }
                if ri == 2u32 {
                    r = r2;
                }
                if ri == 3u32 {
                    r = r3;
                }
                if ri == 4u32 {
                    r = r4;
                }
                if ri == 5u32 {
                    r = r5;
                }
                if ri == 6u32 {
                    r = r6;
                }
                if ri == 7u32 {
                    r = r7;
                }
                if ri == 8u32 {
                    r = r8;
                }
                let top = work_max == 4294967295u32;
                let lim = work_max + 1u32;
                let over_a = partial > lim / r;
                let qmax = 4294967295u32 / r;
                let rem = 4294967295u32 % r;
                let fits_b = partial <= qmax;
                let plus_one = r != 1u32;
                let exact_b = plus_one && partial == qmax + 1u32 && r == rem + 1u32;
                let live_flag = cut == 0u32;
                let was_big = big == 1u32;
                let one = r == 1u32;
                let cut_big = live_flag && was_big && one == false;
                let cut_a = live_flag && was_big == false && top == false && over_a;
                let cut_b =
                    live_flag && was_big == false && top && fits_b == false && exact_b == false;
                let cut_now = cut_big || cut_a || cut_b;
                let grow_a = live_flag && was_big == false && top == false && over_a == false;
                let grow_b = live_flag && was_big == false && top && fits_b;
                let grow_big = live_flag && was_big == false && top && fits_b == false && exact_b;
                let grow = grow_a || grow_b;
                if cut_now {
                    cut = 1u32;
                }
                if grow {
                    partial *= r;
                }
                if grow_big {
                    big = 1u32;
                    partial = 0u32;
                }
                ri += 1u32;
            }
            let mut visits = partial - 1u32;
            let exhausted = cut == 1u32;
            let at_top = exhausted == false && big == 1u32;
            if exhausted {
                visits = work_max;
            }
            if at_top {
                visits = 4294967295u32;
            }
            let mut status: u32 = 0u32;
            if cut == 1u32 {
                status |= 1u32;
            }
            let mut lo_w = t - half_p;
            if half_p > t {
                lo_w = 0u32;
            }
            let mut hi_w = t + half_p;
            if hi_w < t {
                hi_w = 4294967295u32;
            }
            let mut h_pos: u32 = 0u32;
            if adduct_id == 1u32 {
                h_pos = 1u32;
            }
            let h_cap = c1 + h_pos + 2u32;
            let mut h_hi_abs = h_cap;
            if h_hi_abs > 65535u32 {
                h_hi_abs = 65535u32;
            }
            let mut accepted: u32 = 0u32;
            let mut ambiguous: u32 = 0u32;
            let mut stored: u32 = 0u32;
            let mut k: u32 = 1u32;
            let mut live: u32 = 1u32;
            while k <= visits && live == 1u32 {
                // One visit: mixed-radix digits, carbon least significant
                // (`ion_lane_visit` digits block, scalar-expanded).
                let mut tmp = k;
                let d0 = tmp % r0;
                tmp /= r0;
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
                // Heavy mass, division-guarded (`ion_lane_visit` mass block).
                let mut m: u32 = 0u32;
                let mut mass_ok: u32 = 1u32;
                let mut mi: u32 = 0u32;
                while mi < 9u32 {
                    let mut d = d0;
                    if mi == 1u32 {
                        d = d1;
                    }
                    if mi == 2u32 {
                        d = d2;
                    }
                    if mi == 3u32 {
                        d = d3;
                    }
                    if mi == 4u32 {
                        d = d4;
                    }
                    if mi == 5u32 {
                        d = d5;
                    }
                    if mi == 6u32 {
                        d = d6;
                    }
                    if mi == 7u32 {
                        d = d7;
                    }
                    if mi == 8u32 {
                        d = d8;
                    }
                    let me = ms2_ion_heavy_mass(mi);
                    let room = 4294967295u32 - m;
                    let fits = d <= room / me;
                    let go = mass_ok == 1u32 && fits;
                    if mass_ok == 1u32 && fits {
                        mass_ok = 1u32;
                    } else {
                        mass_ok = 0u32;
                    }
                    if go {
                        m += d * me;
                    }
                    mi += 1u32;
                }
                // Heavy rounding residual (`ion_lane_visit` residual block).
                let mut res_heavy: u32 = 0u32;
                let mut ei: u32 = 0u32;
                while ei < 9u32 {
                    let mut d = d0;
                    if ei == 1u32 {
                        d = d1;
                    }
                    if ei == 2u32 {
                        d = d2;
                    }
                    if ei == 3u32 {
                        d = d3;
                    }
                    if ei == 4u32 {
                        d = d4;
                    }
                    if ei == 5u32 {
                        d = d5;
                    }
                    if ei == 6u32 {
                        d = d6;
                    }
                    if ei == 7u32 {
                        d = d7;
                    }
                    if ei == 8u32 {
                        d = d8;
                    }
                    res_heavy += d * ms2_ion_heavy_res(ei);
                    ei += 1u32;
                }
                let in_hi = m <= hi_w;
                let gated = mass_ok == 1u32 && in_hi;
                let below = m < lo_w;
                let gap = lo_w - m;
                let mut ceil_add: u32 = 0u32;
                if gap % 1007825u32 != 0u32 {
                    ceil_add = 1u32;
                }
                let ceil = gap / 1007825u32 + ceil_add;
                let mut h_lo: u32 = 0u32;
                if gated && below {
                    h_lo = ceil;
                }
                let span_h = hi_w - m;
                let mut h_hi = span_h / 1007825u32;
                if h_hi > h_hi_abs {
                    h_hi = h_hi_abs;
                }
                let has = gated && h_lo <= h_hi;
                let full_n = h_hi - h_lo + 1u32;
                let mut want: u32 = 0u32;
                if has {
                    want = full_n;
                }
                let mut cap_n = want;
                if want > 3u32 {
                    cap_n = 3u32;
                }
                // Hydrogen hypotheses, at most three (`ion_lane_visit`
                // hydrogen block, with the slot-guarded mass formation).
                let mut hh: u32 = 0u32;
                while hh < 3u32 {
                    let on = has && hh < cap_n;
                    let h = h_lo + hh;
                    let room_h = 4294967295u32 - m;
                    let fits_h = h <= room_h / 1007825u32;
                    let slot_on = on && fits_h;
                    if slot_on {
                        let cand = m + h * 1007825u32;
                        let res = res_heavy + h * 33u32 + 421u32;
                        let mut arith_add: u32 = 0u32;
                        if res % 1000u32 != 0u32 {
                            arith_add = 1u32;
                        }
                        let arith = res / 1000u32 + arith_add;
                        let mut bound = arith + u_unc;
                        if bound < arith {
                            bound = 4294967295u32;
                        }
                        let v = ms2_ion_verdict(t, cand, bound, tol_p);
                        let ob = cand - t + 2147483648u32;
                        // Saturating counts, like the twin.
                        if v == 1u32 && accepted < 4294967295u32 {
                            accepted += 1u32;
                        }
                        if v == 2u32 && ambiguous < 4294967295u32 {
                            ambiguous += 1u32;
                        }
                        if v == 1u32 && stored < j {
                            let w = ion_base + stored * 12u32;
                            let mut di: u32 = 0u32;
                            while di < 9u32 {
                                let mut d = d0;
                                if di == 1u32 {
                                    d = d1;
                                }
                                if di == 2u32 {
                                    d = d2;
                                }
                                if di == 3u32 {
                                    d = d3;
                                }
                                if di == 4u32 {
                                    d = d4;
                                }
                                if di == 5u32 {
                                    d = d5;
                                }
                                if di == 6u32 {
                                    d = d6;
                                }
                                if di == 7u32 {
                                    d = d7;
                                }
                                if di == 8u32 {
                                    d = d8;
                                }
                                // Digit words are C, N, O, F, P, S, Cl, Br,
                                // I (twin `HEAVY` order).
                                let mut word = 0u32;
                                if di == 1u32 {
                                    word = 2u32;
                                }
                                if di == 2u32 {
                                    word = 3u32;
                                }
                                if di == 3u32 {
                                    word = 4u32;
                                }
                                if di == 4u32 {
                                    word = 5u32;
                                }
                                if di == 5u32 {
                                    word = 6u32;
                                }
                                if di == 6u32 {
                                    word = 7u32;
                                }
                                if di == 7u32 {
                                    word = 8u32;
                                }
                                if di == 8u32 {
                                    word = 9u32;
                                }
                                ion[(w + word) as usize] = d;
                                di += 1u32;
                            }
                            ion[(w + 1u32) as usize] = h;
                            ion[(w + 10u32) as usize] = cand;
                            ion[(w + 11u32) as usize] = ob;
                            stored += 1u32;
                        }
                    }
                    hh += 1u32;
                }
                if k == 4294967295u32 {
                    live = 0u32;
                } else {
                    k += 1u32;
                }
            }
            if accepted > j {
                status |= 2u32;
            }
            ion_meta[im_base as usize] = accepted;
            ion_meta[(im_base + 1u32) as usize] = ambiguous;
            ion_meta[(im_base + 2u32) as usize] = stored;
            ion_meta[(im_base + 3u32) as usize] = status;
        } else {
            ion_meta[(im_base + 3u32) as usize] = 4u32;
        }
    }
}

/// Fragment-ion hypotheses of one `(B, F, N)` bucket, lane per `(b, f, p)`.
///
/// `top_counts [B, F, 10]` holds the retained parent compositions,
/// `kept [B, N, 3]` the kept peaks (raw index, m/z, reverse),
/// `meta [B, 8]` the spectrum words (peak count, precursor, precursor
/// uncertainty, adduct, fragment-tolerance ppm tenths, precursor tolerance,
/// id lo/hi) and `spec [B, 2]` the m/z uncertainty with one reserved word.
/// Writes `ion [B, F, N, J, 12]` (10 counts with the ion's own hydrogen
/// count, mass, residual offset by 2^31; zeros in padding) and
/// `ion_meta [B, F, N, 4]` (accepted, ambiguous, kept, status bits 0
/// exhausted, 1 capacity exceeded, 2 unavailable). The caller guarantees
/// valid adducts and `ppm <= 1000` (request validation on the host; the lane
/// maps violations to unavailable without a device read). Every bound
/// buffer's word count and every narrowed scalar fit `u32` (checked below
/// with checked host arithmetic); larger layouts would wrap device
/// addresses — e.g. `B = 21_846, F = 8, N = 256, J = 8` wraps the last
/// spectrum's base onto earlier lanes — and are refused with [`Error::Shape`]
/// before launch.
#[allow(clippy::too_many_arguments)]
pub fn ion_assign<R: Runtime>(
    top_counts: &IdTensor<R>,
    kept: &IdTensor<R>,
    meta: &IdTensor<R>,
    spec: &IdTensor<R>,
    ion: &mut IdTensor<R>,
    ion_meta: &mut IdTensor<R>,
    work_max: u32,
) -> Result<()> {
    if top_counts.shape().rank() != 3
        || kept.shape().rank() != 3
        || meta.shape().rank() != 2
        || spec.shape().rank() != 2
        || ion.shape().rank() != 5
        || ion_meta.shape().rank() != 4
    {
        return Err(Error::shape(format!(
            "ion_assign needs top_counts [B, F, 10], kept [B, N, 3], meta [B, 8], spec [B, 2], ion [B, F, N, J, 12] and ion_meta [B, F, N, 4], got {} and {} and {} and {} and {} and {}",
            top_counts.shape(),
            kept.shape(),
            meta.shape(),
            spec.shape(),
            ion.shape(),
            ion_meta.shape()
        )));
    }
    let batch = top_counts.shape().dim(0);
    let f = top_counts.shape().dim(1);
    let n = kept.shape().dim(1);
    let j = ion.shape().dim(3);
    if top_counts.shape().dim(2) != 10
        || kept.shape().dim(2) != 3
        || ion.shape().dim(4) != 12
        || ion_meta.shape().dim(3) != 4
    {
        return Err(Error::shape(format!(
            "ion_assign needs last dimensions 10, 3, 12 and 4, got {} and {} and {} and {}",
            top_counts.shape(),
            kept.shape(),
            ion.shape(),
            ion_meta.shape()
        )));
    }
    let want_top: &[usize] = &[batch, f, 10];
    let want_kept: &[usize] = &[batch, n, 3];
    let want_meta: &[usize] = &[batch, 8];
    let want_spec: &[usize] = &[batch, 2];
    let want_ion: &[usize] = &[batch, f, n, j, 12];
    let want_im: &[usize] = &[batch, f, n, 4];
    if top_counts.shape().dims() != want_top
        || kept.shape().dims() != want_kept
        || meta.shape().dims() != want_meta
        || spec.shape().dims() != want_spec
        || ion.shape().dims() != want_ion
        || ion_meta.shape().dims() != want_im
    {
        return Err(Error::shape(format!(
            "ion_assign needs top_counts [{batch}, {f}, 10], kept [{batch}, {n}, 3], meta [{batch}, 8], spec [{batch}, 2], ion [{batch}, {f}, {n}, {j}, 12] and ion_meta [{batch}, {f}, {n}, 4], got {} and {} and {} and {} and {} and {}",
            top_counts.shape(),
            kept.shape(),
            meta.shape(),
            spec.shape(),
            ion.shape(),
            ion_meta.shape()
        )));
    }
    let lanes = batch
        .checked_mul(f)
        .and_then(|v| v.checked_mul(n))
        .ok_or_else(|| Error::shape("ion_assign: B * F * N overflows usize".to_string()))?;
    if lanes == 0 {
        return Ok(());
    }
    check_device_len("top_counts", top_counts.len())?;
    check_device_len("kept", kept.len())?;
    check_device_len("meta", meta.len())?;
    check_device_len("spec", spec.len())?;
    check_device_len("ion", ion.len())?;
    check_device_len("ion_meta", ion_meta.len())?;
    let f_u32 = check_device_scalar("ion_assign F", f)?;
    let n_u32 = check_device_scalar("ion_assign N", n)?;
    let j_u32 = check_device_scalar("ion_assign J", j)?;
    // `lanes` fits `u32` (checked above), so every `pos < lanes` narrows to a
    // `u32` lane without wrapping.
    check_device_scalar("ion_assign B * F * N lanes", lanes)?;
    let client = top_counts.client();
    let per_lane = j.saturating_mul(12).saturating_add(64).max(1);
    let (count, dim, span) = launch_1d_spans(client, lanes, per_lane);
    unsafe {
        ms2_ion_assign_kernel::launch_unchecked::<R>(
            client,
            count,
            dim,
            top_counts.arg(),
            kept.arg(),
            meta.arg(),
            spec.arg(),
            ion.arg(),
            ion_meta.arg(),
            f_u32,
            n_u32,
            j_u32,
            work_max,
            lanes,
            span,
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// `ms2_ion_label_mask`: lane per `(b, p)`
// ---------------------------------------------------------------------------

/// Lane per `(b, p)` of [`ion_label_mask`]; a copy of
/// [`crate::models::ms2::ion::label_mask_lane`] over `Array`s.
#[allow(clippy::too_many_arguments)]
#[allow(unused_assignments)]
#[cube(launch_unchecked)]
fn ms2_ion_label_mask_kernel<F: Float + CubeElement>(
    ion_labels: &Array<u32>,
    ion: &Array<u32>,
    ion_meta: &Array<u32>,
    kept: &Array<u32>,
    label_mask: &mut Array<F>,
    label_state: &mut Array<u32>,
    f_dim: u32,
    n: u32,
    j: u32,
    f_slot: u32,
    label_cap: u32,
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
        let b = lane / n;
        let p = lane % n;
        let width = j + 1u32;
        let lab_base = b * label_cap * 12u32;
        let apos = (b * f_dim + f_slot) * n + p;
        let ion_base = apos * j * 12u32;
        let im_base = apos * 4u32;
        let kept_base = (b * n + p) * 3u32;
        let mask_base = lane * width;
        let mut zi: u32 = 0u32;
        while zi < width {
            label_mask[(mask_base + zi) as usize] = F::new(0.0_f32);
            zi += 1u32;
        }
        label_state[lane as usize] = 0u32;
        let raw = kept[kept_base as usize];
        if raw != 4294967295u32 {
            let kept_n = ion_meta[(im_base + 2u32) as usize];
            let mut any_kept: u32 = 0u32;
            let mut qq: u32 = 0u32;
            while qq < j && qq < kept_n {
                let hb = ion_base + qq * 12u32;
                let mut hit: u32 = 0u32;
                let mut l: u32 = 0u32;
                while l < label_cap {
                    let lb = lab_base + l * 12u32;
                    if ion_labels[(lb + 11u32) as usize] != 0u32
                        && ion_labels[lb as usize] == raw
                    {
                        let mut eq: u32 = 1u32;
                        let mut e: u32 = 0u32;
                        while e < 10u32 {
                            if ion[(hb + e) as usize] != ion_labels[(lb + 1u32 + e) as usize] {
                                eq = 0u32;
                            }
                            e += 1u32;
                        }
                        if eq == 1u32 {
                            hit = 1u32;
                        }
                    }
                    l += 1u32;
                }
                if hit == 1u32 {
                    label_mask[(mask_base + qq) as usize] = F::new(1.0_f32);
                    any_kept = 1u32;
                }
                qq += 1u32;
            }
            if any_kept == 1u32 {
                // True partial from the label matching itself (review
                // finding A1): some label of this peak is kept while some
                // label of the same peak is not among the kept hypotheses.
                let mut any_dropped: u32 = 0u32;
                let mut l2: u32 = 0u32;
                while l2 < label_cap {
                    let lb2 = lab_base + l2 * 12u32;
                    if ion_labels[(lb2 + 11u32) as usize] != 0u32
                        && ion_labels[lb2 as usize] == raw
                    {
                        let mut hit2: u32 = 0u32;
                        let mut qq2: u32 = 0u32;
                        while qq2 < j && qq2 < kept_n {
                            let hb2 = ion_base + qq2 * 12u32;
                            let mut eq2: u32 = 1u32;
                            let mut e2: u32 = 0u32;
                            while e2 < 10u32 {
                                if ion[(hb2 + e2) as usize] != ion_labels[(lb2 + 1u32 + e2) as usize] {
                                    eq2 = 0u32;
                                }
                                e2 += 1u32;
                            }
                            if eq2 == 1u32 {
                                hit2 = 1u32;
                            }
                            qq2 += 1u32;
                        }
                        if hit2 == 0u32 {
                            any_dropped = 1u32;
                        }
                    }
                    l2 += 1u32;
                }
                if any_dropped == 1u32 {
                    label_state[lane as usize] = 3u32;
                } else {
                    label_state[lane as usize] = 1u32;
                }
            } else {
                let mut any_label: u32 = 0u32;
                let mut l: u32 = 0u32;
                while l < label_cap {
                    let lb = lab_base + l * 12u32;
                    if ion_labels[(lb + 11u32) as usize] != 0u32
                        && ion_labels[lb as usize] == raw
                    {
                        any_label = 1u32;
                    }
                    l += 1u32;
                }
                if any_label == 1u32 {
                    label_state[lane as usize] = 2u32;
                }
                label_mask[(mask_base + j) as usize] = F::new(1.0_f32);
            }
        } else {
            label_mask[(mask_base + j) as usize] = F::new(1.0_f32);
        }
    }
}

/// Assignment label masks of one `(B, N)` bucket, lane per `(b, p)`.
///
/// `ion_labels [B, L, 12]` holds (raw index, 10 counts, valid flag),
/// `ion [B, F, N, J, 12]` and `ion_meta [B, F, N, 4]` the assignment rows of
/// formula slot `f_slot`, `kept [B, N, 3]` the kept peaks. Writes
/// `label_mask [B, N, J + 1]` (float 0/1; class `J` is unassigned) and
/// `label_state [B, N]` (0 no label, 1 some label kept with every label
/// kept, 2 labels but none kept, 3 true partial: some label kept while some
/// label of the same peak is not). Training uses `F = 1`. Every bound buffer's word count and every
/// narrowed scalar fit `u32` (checked below with checked host arithmetic);
/// larger layouts would wrap device addresses and are refused with
/// [`Error::Shape`] before launch.
#[allow(clippy::too_many_arguments)]
pub fn ion_label_mask<R: Runtime, E: FloatElem>(
    ion_labels: &IdTensor<R>,
    ion: &IdTensor<R>,
    ion_meta: &IdTensor<R>,
    kept: &IdTensor<R>,
    label_mask: &mut Tensor<R, E>,
    label_state: &mut IdTensor<R>,
    f_slot: u32,
) -> Result<()> {
    if ion_labels.shape().rank() != 3
        || ion.shape().rank() != 5
        || ion_meta.shape().rank() != 4
        || kept.shape().rank() != 3
        || label_mask.shape().rank() != 3
        || label_state.shape().rank() != 2
    {
        return Err(Error::shape(format!(
            "ion_label_mask needs ion_labels [B, L, 12], ion [B, F, N, J, 12], ion_meta [B, F, N, 4], kept [B, N, 3], label_mask [B, N, J + 1] and label_state [B, N], got {} and {} and {} and {} and {} and {}",
            ion_labels.shape(),
            ion.shape(),
            ion_meta.shape(),
            kept.shape(),
            label_mask.shape(),
            label_state.shape()
        )));
    }
    let batch = ion.shape().dim(0);
    let f = ion.shape().dim(1);
    let n = ion.shape().dim(2);
    let j = ion.shape().dim(3);
    let l = ion_labels.shape().dim(1);
    if ion_labels.shape().dim(2) != 12
        || ion.shape().dim(4) != 12
        || ion_meta.shape().dim(3) != 4
        || kept.shape().dim(2) != 3
    {
        return Err(Error::shape(format!(
            "ion_label_mask needs last dimensions 12, 12, 4 and 3, got {} and {} and {} and {}",
            ion_labels.shape(),
            ion.shape(),
            ion_meta.shape(),
            kept.shape()
        )));
    }
    let want_lab: &[usize] = &[batch, l, 12];
    let want_ion: &[usize] = &[batch, f, n, j, 12];
    let want_im: &[usize] = &[batch, f, n, 4];
    let want_kept: &[usize] = &[batch, n, 3];
    let want_mask: &[usize] = &[batch, n, j + 1];
    let want_state: &[usize] = &[batch, n];
    if ion_labels.shape().dims() != want_lab
        || ion.shape().dims() != want_ion
        || ion_meta.shape().dims() != want_im
        || kept.shape().dims() != want_kept
        || label_mask.shape().dims() != want_mask
        || label_state.shape().dims() != want_state
    {
        return Err(Error::shape(format!(
            "ion_label_mask needs ion_labels [{batch}, {l}, 12], ion [{batch}, {f}, {n}, {j}, 12], ion_meta [{batch}, {f}, {n}, 4], kept [{batch}, {n}, 3], label_mask [{batch}, {n}, {}] and label_state [{batch}, {n}], got {} and {} and {} and {} and {} and {}",
            j + 1,
            ion_labels.shape(),
            ion.shape(),
            ion_meta.shape(),
            kept.shape(),
            label_mask.shape(),
            label_state.shape()
        )));
    }
    if f_slot >= f as u32 && n > 0 {
        return Err(Error::shape(format!(
            "ion_label_mask needs f_slot {f_slot} below F {f}"
        )));
    }
    let lanes = batch
        .checked_mul(n)
        .ok_or_else(|| Error::shape("ion_label_mask: B * N overflows usize".to_string()))?;
    if lanes == 0 {
        return Ok(());
    }
    check_device_len("ion_labels", ion_labels.len())?;
    check_device_len("ion", ion.len())?;
    check_device_len("ion_meta", ion_meta.len())?;
    check_device_len("kept", kept.len())?;
    check_device_len("label_mask", label_mask.len())?;
    check_device_len("label_state", label_state.len())?;
    // `lanes` fits `u32`, so every `pos < lanes` narrows to a `u32` lane.
    check_device_scalar("ion_label_mask B * N lanes", lanes)?;
    let f_u32 = check_device_scalar("ion_label_mask F", f)?;
    let n_u32 = check_device_scalar("ion_label_mask N", n)?;
    let j_u32 = check_device_scalar("ion_label_mask J", j)?;
    let l_u32 = check_device_scalar("ion_label_mask L", l)?;
    let client = ion_labels.client();
    let per_lane = l.saturating_add(j + 1).max(1);
    let (count, dim, span) = launch_1d_spans(client, lanes, per_lane);
    unsafe {
        ms2_ion_label_mask_kernel::launch_unchecked::<E, R>(
            client,
            count,
            dim,
            ion_labels.arg(),
            ion.arg(),
            ion_meta.arg(),
            kept.arg(),
            label_mask.arg(),
            label_state.arg(),
            f_u32,
            n_u32,
            j_u32,
            f_slot,
            l_u32,
            lanes,
            span,
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// `ms2_ion_evidence`: lane per trajectory
// ---------------------------------------------------------------------------

/// Lane per trajectory of [`ion_evidence`]; a copy of
/// [`crate::models::ms2::ion::evidence_lane`] (with
/// [`crate::models::ms2::ion::evidence_match`]) over `Array`s.
///
/// The candidate heavy counts are nine lane scalars (`hc0..hc8`,
/// scalar-expanded from the twin's `hc` array); all other statements are
/// shared verbatim with the twin (modulo the spelling deltas above).
#[allow(clippy::too_many_arguments)]
#[allow(unused_assignments)]
#[cube(launch_unchecked)]
fn ms2_ion_evidence_kernel(
    actions: &Array<u32>,
    traj_slot: &Array<u32>,
    ion: &Array<u32>,
    ion_meta: &Array<u32>,
    evidence: &mut Array<u32>,
    steps: u32,
    atoms_cap: u32,
    k_dim: u32,
    f_dim: u32,
    n: u32,
    j: u32,
    lanes: usize,
    span: usize,
) {
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    for pos in start..end {
        let r = pos as u32;
        let b = r / k_dim;
        let stride = steps * 4u32 + atoms_cap + 4u32;
        let abase = r * stride;
        let tbase = r * 2u32;
        let ebase = r * 18u32;
        let len_field = steps * 4u32 + atoms_cap;
        let length = actions[(abase + len_field) as usize];
        let status_w = actions[(abase + len_field + 1u32) as usize];
        // Bit 0 is `finished`.
        let mut finished: u32 = 0u32;
        if status_w & 1u32 != 0u32 {
            finished = 1u32;
        }
        let slot = traj_slot[tbase as usize];
        let adduct_id = traj_slot[(tbase + 1u32) as usize];
        let mut ha_pos: u32 = 0u32;
        let mut ha_neg: u32 = 0u32;
        let mut adduct_ok: u32 = 0u32;
        if adduct_id == 1u32 {
            ha_pos = 1u32;
            adduct_ok = 1u32;
        }
        if adduct_id == 2u32 {
            ha_neg = 1u32;
            adduct_ok = 1u32;
        }
        // Trace decode: heavy counts, parent hydrogens and the
        // open-valence range.
        let mut hc0: u32 = 0u32;
        let mut hc1: u32 = 0u32;
        let mut hc2: u32 = 0u32;
        let mut hc3: u32 = 0u32;
        let mut hc4: u32 = 0u32;
        let mut hc5: u32 = 0u32;
        let mut hc6: u32 = 0u32;
        let mut hc7: u32 = 0u32;
        let mut hc8: u32 = 0u32;
        let mut hg: u32 = 0u32;
        let mut c_lo: u32 = 0u32;
        let mut c_hi: u32 = 0u32;
        let mut n_atoms: u32 = 0u32;
        let open_base = abase + steps * 4u32;
        let mut s: u32 = 0u32;
        while s < length {
            let tok = abase + s * 4u32;
            let kind = actions[tok as usize];
            let ty = actions[(tok + 1u32) as usize];
            if kind == 2u32 && ty >= 1u32 && ty <= 17u32 && n_atoms < atoms_cap {
                let fields = ms2_ion_atom_fields(ty);
                let el = fields / 65536u32;
                let ph = fields - el * 65536u32;
                if el == 0u32 {
                    hc0 += 1u32;
                }
                if el == 2u32 {
                    hc1 += 1u32;
                }
                if el == 3u32 {
                    hc2 += 1u32;
                }
                if el == 4u32 {
                    hc3 += 1u32;
                }
                if el == 5u32 {
                    hc4 += 1u32;
                }
                if el == 6u32 {
                    hc5 += 1u32;
                }
                if el == 7u32 {
                    hc6 += 1u32;
                }
                if el == 8u32 {
                    hc7 += 1u32;
                }
                if el == 9u32 {
                    hc8 += 1u32;
                }
                hg += ph;
                let o = actions[(open_base + n_atoms) as usize];
                let mut ceil_add: u32 = 0u32;
                if o % 3u32 != 0u32 {
                    ceil_add = 1u32;
                }
                c_lo += o / 3u32 + ceil_add;
                c_hi += o;
                n_atoms += 1u32;
            }
            s += 1u32;
        }
        let mut ok: u32 = 0u32;
        if finished == 1u32 && adduct_ok == 1u32 {
            ok = 1u32;
        }
        // Match block (`evidence_match` with scalar-expanded `hc`).
        let mut zi: u32 = 0u32;
        while zi < 18u32 {
            evidence[(ebase + zi) as usize] = 0u32;
            zi += 1u32;
        }
        let mut valid: u32 = 0u32;
        if ok == 1u32 && slot < f_dim {
            valid = 1u32;
        }
        let mut count: u32 = 0u32;
        let mut recorded: u32 = 0u32;
        let mut best: u32 = 0u32;
        let mut incompl: u32 = 0u32;
        let mut p: u32 = 0u32;
        while p < n {
            if valid == 1u32 {
                let apos = (b * f_dim + slot) * n + p;
                let ibase = apos * j * 12u32;
                let mbase = apos * 4u32;
                if ion_meta[(mbase + 3u32) as usize] != 0u32 {
                    incompl = 1u32;
                }
                let kept_n = ion_meta[(mbase + 2u32) as usize];
                let mut bq: u32 = 0u32;
                let mut bres: u32 = 0u32;
                let mut found: u32 = 0u32;
                let mut q: u32 = 0u32;
                while q < kept_n {
                    let hb = ibase + q * 12u32;
                    let mut eq: u32 = 1u32;
                    if ion[hb as usize] != hc0 {
                        eq = 0u32;
                    }
                    if ion[(hb + 2u32) as usize] != hc1 {
                        eq = 0u32;
                    }
                    if ion[(hb + 3u32) as usize] != hc2 {
                        eq = 0u32;
                    }
                    if ion[(hb + 4u32) as usize] != hc3 {
                        eq = 0u32;
                    }
                    if ion[(hb + 5u32) as usize] != hc4 {
                        eq = 0u32;
                    }
                    if ion[(hb + 6u32) as usize] != hc5 {
                        eq = 0u32;
                    }
                    if ion[(hb + 7u32) as usize] != hc6 {
                        eq = 0u32;
                    }
                    if ion[(hb + 8u32) as usize] != hc7 {
                        eq = 0u32;
                    }
                    if ion[(hb + 9u32) as usize] != hc8 {
                        eq = 0u32;
                    }
                    if eq == 1u32 {
                        let ob = ion[(hb + 11u32) as usize];
                        let mut rabs: u32 = 0u32;
                        if ob >= 2147483648u32 {
                            rabs = ob - 2147483648u32;
                        } else {
                            rabs = 2147483648u32 - ob;
                        }
                        if found == 0u32 || rabs < bres {
                            bq = q;
                            bres = rabs;
                            found = 1u32;
                        }
                    }
                    q += 1u32;
                }
                if found == 1u32 {
                    let hb = ibase + bq * 12u32;
                    let hh = ion[(hb + 1u32) as usize];
                    let lhs = hh + ha_neg;
                    let rhs = hg + ha_pos;
                    let ge = lhs >= rhs;
                    let mut s_abs: u32 = 0u32;
                    if ge {
                        s_abs = lhs - rhs;
                    } else {
                        s_abs = rhs - lhs;
                    }
                    let mut lim_lo = c_lo;
                    if lim_lo > 2u32 {
                        lim_lo = 2u32;
                    }
                    let mut lim_hi = c_hi;
                    if lim_hi > 2u32 {
                        lim_hi = 2u32;
                    }
                    let mut base: u32 = 0u32;
                    if s_abs <= lim_lo {
                        base = 2u32;
                    } else if s_abs <= lim_hi {
                        base = 1u32;
                    }
                    if base > 0u32 {
                        if base > best {
                            best = base;
                        }
                        count += 1u32;
                        if recorded < 4u32 {
                            let w = ebase + 2u32 + recorded * 4u32;
                            evidence[w as usize] = p;
                            evidence[(w + 1u32) as usize] = bq;
                            if ge {
                                evidence[(w + 2u32) as usize] = 2147483648u32 + s_abs;
                            } else {
                                evidence[(w + 2u32) as usize] = 2147483648u32 - s_abs;
                            }
                            evidence[(w + 3u32) as usize] = ion[(hb + 11u32) as usize];
                            recorded += 1u32;
                        }
                    }
                }
            }
            p += 1u32;
        }
        let mut st = best;
        if incompl == 1u32 {
            st += 128u32;
        }
        evidence[ebase as usize] = st;
        evidence[(ebase + 1u32) as usize] = count;
    }
}

/// Mass-consistency evidence of one `(B, K)` bucket, lane per trajectory.
///
/// `actions [R, S]` holds the trajectory rows (`S = steps * 4 + atoms + 4`:
/// `steps` token words, `atoms` open-valence words in trace order, then
/// length, status and two reserved words); `traj_slot [R, 2]` the formula
/// slot and adduct id per trajectory. Reads the `ion [B, F, N, J, 12]` and
/// `ion_meta [B, F, N, 4]` rows of the trajectory's own formula slot and
/// writes `evidence [R, 2 + 4E]` (`E = 4`): status, match count, then up to
/// 4 records of (kept-peak position, hypothesis index, shift offset by 2^31,
/// residual offset by 2^31), the first matches in peak order. The
/// probability-ranked record choice of §2.4 needs the assignment head and is
/// done later. An unfinished trajectory, an unknown adduct or an invalid
/// slot writes a zeroed row.
///
/// Producer preconditions (trusted, not re-read): every trajectory's
/// `length <= steps` and every compared `(b, f, p)` row's kept-hypothesis
/// count `kept_n <= J`; larger values would read past the bound rows. Only
/// shifts satisfying the §2.4 mass-consistency rule (`base > 0`) count or
/// record. Every bound buffer's word count and every narrowed scalar fit
/// `u32` (checked below with checked host arithmetic); larger layouts would
/// wrap device addresses and are refused with [`Error::Shape`] before
/// launch.
#[allow(clippy::too_many_arguments)]
pub fn ion_evidence<R: Runtime>(
    actions: &IdTensor<R>,
    traj_slot: &IdTensor<R>,
    ion: &IdTensor<R>,
    ion_meta: &IdTensor<R>,
    evidence: &mut IdTensor<R>,
    steps: u32,
    atoms_cap: u32,
    per_spectrum: u32,
) -> Result<()> {
    if actions.shape().rank() != 2
        || traj_slot.shape().rank() != 2
        || ion.shape().rank() != 5
        || ion_meta.shape().rank() != 4
        || evidence.shape().rank() != 2
    {
        return Err(Error::shape(format!(
            "ion_evidence needs actions [R, S], traj_slot [R, 2], ion [B, F, N, J, 12], ion_meta [B, F, N, 4] and evidence [R, 18], got {} and {} and {} and {} and {}",
            actions.shape(),
            traj_slot.shape(),
            ion.shape(),
            ion_meta.shape(),
            evidence.shape()
        )));
    }
    let rows = actions.shape().dim(0);
    let batch = ion.shape().dim(0);
    let f = ion.shape().dim(1);
    let n = ion.shape().dim(2);
    let j = ion.shape().dim(3);
    let expect_stride = (steps as usize)
        .checked_mul(4)
        .and_then(|v| v.checked_add(atoms_cap as usize))
        .and_then(|v| v.checked_add(4));
    let Some(expect_stride) = expect_stride else {
        return Err(Error::shape("ion_evidence: steps * 4 + atoms + 4 overflows usize".to_string()));
    };
    if ion.shape().dim(4) != 12 || ion_meta.shape().dim(3) != 4 {
        return Err(Error::shape(format!(
            "ion_evidence needs last dimensions 12 and 4, got {} and {}",
            ion.shape(),
            ion_meta.shape()
        )));
    }
    let want_actions: &[usize] = &[rows, expect_stride];
    let want_slot: &[usize] = &[rows, 2];
    let want_ion: &[usize] = &[batch, f, n, j, 12];
    let want_im: &[usize] = &[batch, f, n, 4];
    let want_ev: &[usize] = &[rows, EVIDENCE_ROW_WORDS];
    if actions.shape().dims() != want_actions
        || traj_slot.shape().dims() != want_slot
        || ion.shape().dims() != want_ion
        || ion_meta.shape().dims() != want_im
        || evidence.shape().dims() != want_ev
    {
        return Err(Error::shape(format!(
            "ion_evidence needs actions [{rows}, {expect_stride}], traj_slot [{rows}, 2], ion [{batch}, {f}, {n}, {j}, 12], ion_meta [{batch}, {f}, {n}, 4] and evidence [{rows}, {}], got {} and {} and {} and {} and {}",
            EVIDENCE_ROW_WORDS,
            actions.shape(),
            traj_slot.shape(),
            ion.shape(),
            ion_meta.shape(),
            evidence.shape()
        )));
    }
    let Some(expect_rows) = batch.checked_mul(per_spectrum as usize) else {
        return Err(Error::shape(
            "ion_evidence: B * K overflows usize".to_string(),
        ));
    };
    if per_spectrum == 0 || rows != expect_rows {
        return Err(Error::shape(format!(
            "ion_evidence needs rows {rows} == B {batch} * K {per_spectrum}"
        )));
    }
    if steps > 1_000_000 || atoms_cap > 1_000_000 {
        return Err(Error::shape(format!(
            "ion_evidence needs steps {steps} and atoms {atoms_cap} within u32 lane arithmetic"
        )));
    }
    if rows == 0 {
        return Ok(());
    }
    check_device_len("actions", actions.len())?;
    check_device_len("traj_slot", traj_slot.len())?;
    check_device_len("ion", ion.len())?;
    check_device_len("ion_meta", ion_meta.len())?;
    check_device_len("evidence", evidence.len())?;
    // `rows` fits `u32`, so every `r < rows` narrows to a `u32` lane.
    check_device_scalar("ion_evidence rows", rows)?;
    let f_u32 = check_device_scalar("ion_evidence F", f)?;
    let n_u32 = check_device_scalar("ion_evidence N", n)?;
    let j_u32 = check_device_scalar("ion_evidence J", j)?;
    let client = actions.client();
    let per_lane = expect_stride.saturating_add(n.saturating_mul(j));
    let (count, dim, span) = launch_1d_spans(client, rows, per_lane.max(1));
    unsafe {
        ms2_ion_evidence_kernel::launch_unchecked::<R>(
            client,
            count,
            dim,
            actions.arg(),
            traj_slot.arg(),
            ion.arg(),
            ion_meta.arg(),
            evidence.arg(),
            steps,
            atoms_cap,
            per_spectrum,
            f_u32,
            n_u32,
            j_u32,
            rows,
            span,
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// `ms2_ion_evidence_scored`: lane per trajectory with probability-ranked
// records (architecture §2.4). Second selection kernel (the task's allowed
// alternative to extending `ms2_ion_evidence`): the first kernel keeps
// first-E in peak order; this one keeps best-E by assignment log-probability.
// The twin is `evidence_lane_scored` (`evidence_match_scored`) in
// `crate::models::ms2::ion`, copied line for line (scalar-expanded `hc`,
// manual absolute differences, plain wrapping ops, `u32` flags).
// ---------------------------------------------------------------------------

/// Lane per trajectory of [`ion_evidence_scored`]; a copy of
/// [`crate::models::ms2::ion::evidence_lane_scored`] (with
/// [`crate::models::ms2::ion::evidence_match_scored`]) over `Array`s.
#[allow(clippy::too_many_arguments)]
#[allow(unused_assignments)]
#[cube(launch_unchecked)]
fn ms2_ion_evidence_scored_kernel<F: Float + CubeElement>(
    actions: &Array<u32>,
    traj_slot: &Array<u32>,
    ion: &Array<u32>,
    ion_meta: &Array<u32>,
    log_prob: &Array<F>,
    evidence: &mut Array<u32>,
    steps: u32,
    atoms_cap: u32,
    k_dim: u32,
    f_dim: u32,
    n: u32,
    j: u32,
    lanes: usize,
    span: usize,
) {
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    for pos in start..end {
        let r = pos as u32;
        let b = r / k_dim;
        let stride = steps * 4u32 + atoms_cap + 4u32;
        let abase = r * stride;
        let tbase = r * 2u32;
        let ebase = r * 18u32;
        let len_field = steps * 4u32 + atoms_cap;
        let length = actions[(abase + len_field) as usize];
        let status_w = actions[(abase + len_field + 1u32) as usize];
        let mut finished: u32 = 0u32;
        if status_w & 1u32 != 0u32 {
            finished = 1u32;
        }
        let slot = traj_slot[tbase as usize];
        let adduct_id = traj_slot[(tbase + 1u32) as usize];
        let mut ha_pos: u32 = 0u32;
        let mut ha_neg: u32 = 0u32;
        let mut adduct_ok: u32 = 0u32;
        if adduct_id == 1u32 {
            ha_pos = 1u32;
            adduct_ok = 1u32;
        }
        if adduct_id == 2u32 {
            ha_neg = 1u32;
            adduct_ok = 1u32;
        }
        let mut hc0: u32 = 0u32;
        let mut hc1: u32 = 0u32;
        let mut hc2: u32 = 0u32;
        let mut hc3: u32 = 0u32;
        let mut hc4: u32 = 0u32;
        let mut hc5: u32 = 0u32;
        let mut hc6: u32 = 0u32;
        let mut hc7: u32 = 0u32;
        let mut hc8: u32 = 0u32;
        let mut hg: u32 = 0u32;
        let mut c_lo: u32 = 0u32;
        let mut c_hi: u32 = 0u32;
        let mut n_atoms: u32 = 0u32;
        let open_base = abase + steps * 4u32;
        let mut s: u32 = 0u32;
        while s < length {
            let tok = abase + s * 4u32;
            let kind = actions[tok as usize];
            let ty = actions[(tok + 1u32) as usize];
            if kind == 2u32 && ty >= 1u32 && ty <= 17u32 && n_atoms < atoms_cap {
                let fields = ms2_ion_atom_fields(ty);
                let el = fields / 65536u32;
                let ph = fields - el * 65536u32;
                if el == 0u32 {
                    hc0 += 1u32;
                }
                if el == 2u32 {
                    hc1 += 1u32;
                }
                if el == 3u32 {
                    hc2 += 1u32;
                }
                if el == 4u32 {
                    hc3 += 1u32;
                }
                if el == 5u32 {
                    hc4 += 1u32;
                }
                if el == 6u32 {
                    hc5 += 1u32;
                }
                if el == 7u32 {
                    hc6 += 1u32;
                }
                if el == 8u32 {
                    hc7 += 1u32;
                }
                if el == 9u32 {
                    hc8 += 1u32;
                }
                hg += ph;
                let o = actions[(open_base + n_atoms) as usize];
                let mut ceil_add: u32 = 0u32;
                if o % 3u32 != 0u32 {
                    ceil_add = 1u32;
                }
                c_lo += o / 3u32 + ceil_add;
                c_hi += o;
                n_atoms += 1u32;
            }
            s += 1u32;
        }
        let mut ok: u32 = 0u32;
        if finished == 1u32 && adduct_ok == 1u32 {
            ok = 1u32;
        }
        // Scored match block (`evidence_match_scored` with scalar-expanded
        // `hc` and top-E by log-probability). All buffer loads are
        // unconditional (clamped); masks apply at use.
        let mut zi: u32 = 0u32;
        while zi < 18u32 {
            evidence[(ebase + zi) as usize] = 0u32;
            zi += 1u32;
        }
        let mut valid: u32 = 0u32;
        if ok == 1u32 && slot < f_dim {
            valid = 1u32;
        }
        let mut slot_c = slot;
        if slot >= f_dim {
            slot_c = 0u32;
        }
        let width = j + 1u32;
        let mut count: u32 = 0u32;
        let mut best: u32 = 0u32;
        let mut incompl: u32 = 0u32;
        let mut tp0: u32 = 0u32;
        let mut tp1: u32 = 0u32;
        let mut tp2: u32 = 0u32;
        let mut tp3: u32 = 0u32;
        let mut th0: u32 = 0u32;
        let mut th1: u32 = 0u32;
        let mut th2: u32 = 0u32;
        let mut th3: u32 = 0u32;
        let mut ts0: u32 = 0u32;
        let mut ts1: u32 = 0u32;
        let mut ts2: u32 = 0u32;
        let mut ts3: u32 = 0u32;
        let mut tr0: u32 = 0u32;
        let mut tr1: u32 = 0u32;
        let mut tr2: u32 = 0u32;
        let mut tr3: u32 = 0u32;
        let neg_inf = F::new(-3.4028235e38);
        let mut tl0 = neg_inf;
        let mut tl1 = neg_inf;
        let mut tl2 = neg_inf;
        let mut tl3 = neg_inf;
        let mut kept_e: u32 = 0u32;
        let mut p: u32 = 0u32;
        while p < n {
            let apos = (b * f_dim + slot_c) * n + p;
            let ibase = apos * j * 12u32;
            let mbase = apos * 4u32;
            let st_w = ion_meta[(mbase + 3u32) as usize];
            let kept_n = ion_meta[(mbase + 2u32) as usize];
            let mut is_inc: u32 = 0u32;
            if st_w != 0u32 {
                is_inc = 1u32;
            }
            if valid == 1u32 && is_inc == 1u32 {
                incompl = 1u32;
            }
            let mut bq: u32 = 0u32;
            let mut bres: u32 = 0u32;
            let mut found: u32 = 0u32;
            let mut q: u32 = 0u32;
            while q < kept_n {
                let hb = ibase + q * 12u32;
                let mut eq: u32 = 1u32;
                if ion[hb as usize] != hc0 {
                    eq = 0u32;
                }
                if ion[(hb + 2u32) as usize] != hc1 {
                    eq = 0u32;
                }
                if ion[(hb + 3u32) as usize] != hc2 {
                    eq = 0u32;
                }
                if ion[(hb + 4u32) as usize] != hc3 {
                    eq = 0u32;
                }
                if ion[(hb + 5u32) as usize] != hc4 {
                    eq = 0u32;
                }
                if ion[(hb + 6u32) as usize] != hc5 {
                    eq = 0u32;
                }
                if ion[(hb + 7u32) as usize] != hc6 {
                    eq = 0u32;
                }
                if ion[(hb + 8u32) as usize] != hc7 {
                    eq = 0u32;
                }
                if ion[(hb + 9u32) as usize] != hc8 {
                    eq = 0u32;
                }
                if eq == 1u32 {
                    let ob = ion[(hb + 11u32) as usize];
                    let mut rabs: u32 = 0u32;
                    if ob >= 2147483648u32 {
                        rabs = ob - 2147483648u32;
                    } else {
                        rabs = 2147483648u32 - ob;
                    }
                    if found == 0u32 || rabs < bres {
                        bq = q;
                        bres = rabs;
                        found = 1u32;
                    }
                }
                q += 1u32;
            }
            let hb_best = ibase + bq * 12u32;
            let hh = ion[(hb_best + 1u32) as usize];
            let lhs = hh + ha_neg;
            let rhs = hg + ha_pos;
            let ge = lhs >= rhs;
            let mut s_abs: u32 = 0u32;
            if ge {
                s_abs = lhs - rhs;
            } else {
                s_abs = rhs - lhs;
            }
            let mut lim_lo = c_lo;
            if lim_lo > 2u32 {
                lim_lo = 2u32;
            }
            let mut lim_hi = c_hi;
            if lim_hi > 2u32 {
                lim_hi = 2u32;
            }
            let mut base: u32 = 0u32;
            if s_abs <= lim_lo {
                base = 2u32;
            } else if s_abs <= lim_hi {
                base = 1u32;
            }
            let mut qual: u32 = 0u32;
            if found == 1u32 && base > 0u32 && valid == 1u32 {
                qual = 1u32;
            }
            if qual == 1u32 {
                if base > best {
                    best = base;
                }
                count += 1u32;
            }
            // Log-probability of the chosen hypothesis (unconditional load,
            // masked by `qual` at use via the sentinel below).
            let lp_idx = apos * width + bq;
            let lp_raw = log_prob[lp_idx as usize];
            let mut cand_lp = lp_raw;
            if qual == 0u32 {
                cand_lp = neg_inf;
            }
            // Shift/residual for the record (dummy when unqualified; masked
            // by the same `qual` through `cand_lp` never inserting).
            let mut shift_ob: u32 = 0u32;
            if ge {
                shift_ob = 2147483648u32 + s_abs;
            } else {
                shift_ob = 2147483648u32 - s_abs;
            }
            let resid_ob = ion[(hb_best + 11u32) as usize];
            // Insert into the sorted top-4 when qualified (strictly greater
            // displaces; ties keep the smaller peak position since `p` rises).
            let mut do_ins: u32 = 0u32;
            if qual == 1u32 {
                if kept_e < 4u32 {
                    do_ins = 1u32;
                } else if cand_lp > tl3 {
                    do_ins = 1u32;
                }
            }
            if do_ins == 1u32 {
                // Find insertion position (first strictly smaller entry;
                // append at `kept_e` when larger than none).
                let mut at: u32 = 4u32;
                if kept_e >= 1u32 && cand_lp > tl0 {
                    at = 0u32;
                } else if kept_e >= 2u32 && cand_lp > tl1 {
                    at = 1u32;
                } else if kept_e >= 3u32 && cand_lp > tl2 {
                    at = 2u32;
                } else if kept_e >= 4u32 && cand_lp > tl3 {
                    at = 3u32;
                } else if kept_e < 4u32 {
                    at = kept_e;
                }
                if at == 0u32 {
                    // Shift 2->3, 1->2, 0->1 as needed, then write 0.
                    if kept_e == 4u32 {
                        tp3 = tp2;
                        th3 = th2;
                        ts3 = ts2;
                        tr3 = tr2;
                        tl3 = tl2;
                        tp2 = tp1;
                        th2 = th1;
                        ts2 = ts1;
                        tr2 = tr1;
                        tl2 = tl1;
                        tp1 = tp0;
                        th1 = th0;
                        ts1 = ts0;
                        tr1 = tr0;
                        tl1 = tl0;
                    } else if kept_e == 3u32 {
                        tp3 = tp2;
                        th3 = th2;
                        ts3 = ts2;
                        tr3 = tr2;
                        tl3 = tl2;
                        tp2 = tp1;
                        th2 = th1;
                        ts2 = ts1;
                        tr2 = tr1;
                        tl2 = tl1;
                        tp1 = tp0;
                        th1 = th0;
                        ts1 = ts0;
                        tr1 = tr0;
                        tl1 = tl0;
                        kept_e = 4u32;
                    } else if kept_e == 2u32 {
                        tp2 = tp1;
                        th2 = th1;
                        ts2 = ts1;
                        tr2 = tr1;
                        tl2 = tl1;
                        tp1 = tp0;
                        th1 = th0;
                        ts1 = ts0;
                        tr1 = tr0;
                        tl1 = tl0;
                        kept_e = 3u32;
                    } else if kept_e == 1u32 {
                        tp1 = tp0;
                        th1 = th0;
                        ts1 = ts0;
                        tr1 = tr0;
                        tl1 = tl0;
                        kept_e = 2u32;
                    } else {
                        kept_e = 1u32;
                    }
                    tp0 = p;
                    th0 = bq;
                    ts0 = shift_ob;
                    tr0 = resid_ob;
                    tl0 = cand_lp;
                } else if at == 1u32 {
                    if kept_e == 4u32 {
                        tp3 = tp2;
                        th3 = th2;
                        ts3 = ts2;
                        tr3 = tr2;
                        tl3 = tl2;
                        tp2 = tp1;
                        th2 = th1;
                        ts2 = ts1;
                        tr2 = tr1;
                        tl2 = tl1;
                    } else if kept_e == 3u32 {
                        tp3 = tp2;
                        th3 = th2;
                        ts3 = ts2;
                        tr3 = tr2;
                        tl3 = tl2;
                        tp2 = tp1;
                        th2 = th1;
                        ts2 = ts1;
                        tr2 = tr1;
                        tl2 = tl1;
                        kept_e = 4u32;
                    } else if kept_e == 2u32 {
                        tp2 = tp1;
                        th2 = th1;
                        ts2 = ts1;
                        tr2 = tr1;
                        tl2 = tl1;
                        kept_e = 3u32;
                    } else {
                        kept_e = 2u32;
                    }
                    tp1 = p;
                    th1 = bq;
                    ts1 = shift_ob;
                    tr1 = resid_ob;
                    tl1 = cand_lp;
                } else if at == 2u32 {
                    if kept_e == 4u32 {
                        tp3 = tp2;
                        th3 = th2;
                        ts3 = ts2;
                        tr3 = tr2;
                        tl3 = tl2;
                    } else if kept_e == 3u32 {
                        tp3 = tp2;
                        th3 = th2;
                        ts3 = ts2;
                        tr3 = tr2;
                        tl3 = tl2;
                        kept_e = 4u32;
                    } else {
                        kept_e = 3u32;
                    }
                    tp2 = p;
                    th2 = bq;
                    ts2 = shift_ob;
                    tr2 = resid_ob;
                    tl2 = cand_lp;
                } else if at == 3u32 {
                    if kept_e < 4u32 {
                        kept_e = 4u32;
                    }
                    tp3 = p;
                    th3 = bq;
                    ts3 = shift_ob;
                    tr3 = resid_ob;
                    tl3 = cand_lp;
                }
            }
            p += 1u32;
        }
        // Write the ranked records (already sorted by probability): record
        // `r` at `ebase + 2 + r*4` as (peak, hyp, shift_ob, resid_ob).
        let mut r: u32 = 0u32;
        while r < kept_e {
            let w = ebase + 2u32 + r * 4u32;
            let mut pv: u32 = tp0;
            let mut hv: u32 = th0;
            let mut sv: u32 = ts0;
            let mut rv: u32 = tr0;
            if r == 1u32 {
                pv = tp1;
                hv = th1;
                sv = ts1;
                rv = tr1;
            }
            if r == 2u32 {
                pv = tp2;
                hv = th2;
                sv = ts2;
                rv = tr2;
            }
            if r == 3u32 {
                pv = tp3;
                hv = th3;
                sv = ts3;
                rv = tr3;
            }
            evidence[w as usize] = pv;
            evidence[(w + 1u32) as usize] = hv;
            evidence[(w + 2u32) as usize] = sv;
            evidence[(w + 3u32) as usize] = rv;
            r += 1u32;
        }
        let mut st = best;
        if incompl == 1u32 {
            st += 128u32;
        }
        evidence[ebase as usize] = st;
        evidence[(ebase + 1u32) as usize] = count;
    }
}

/// Mass-consistency evidence with probability-ranked records, lane per
/// trajectory.
///
/// Takes the assignment `log_prob [B, F, N, J + 1]` and keeps the at most 4
/// qualifying peaks of largest probability (ties by smaller peak position).
/// See the kernel above for the lane; the twin is `evidence_lane_scored`.
#[allow(clippy::too_many_arguments)]
pub fn ion_evidence_scored<R: Runtime, E: FloatElem>(
    actions: &IdTensor<R>,
    traj_slot: &IdTensor<R>,
    ion: &IdTensor<R>,
    ion_meta: &IdTensor<R>,
    log_prob: &Tensor<R, E>,
    evidence: &mut IdTensor<R>,
    steps: u32,
    atoms_cap: u32,
    per_spectrum: u32,
) -> Result<()> {
    if actions.shape().rank() != 2
        || traj_slot.shape().rank() != 2
        || ion.shape().rank() != 5
        || ion_meta.shape().rank() != 4
        || log_prob.rank() != 4
        || evidence.shape().rank() != 2
    {
        return Err(Error::shape(format!(
            "ion_evidence_scored needs actions [R, S], traj_slot [R, 2], ion [B, F, N, J, 12], ion_meta [B, F, N, 4], log_prob [B, F, N, J + 1] and evidence [R, 18], got {} and {} and {} and {} and {} and {}",
            actions.shape(),
            traj_slot.shape(),
            ion.shape(),
            ion_meta.shape(),
            log_prob.shape(),
            evidence.shape()
        )));
    }
    let rows = actions.shape().dim(0);
    let batch = ion.shape().dim(0);
    let f = ion.shape().dim(1);
    let n = ion.shape().dim(2);
    let j = ion.shape().dim(3);
    let expect_stride = (steps as usize)
        .checked_mul(4)
        .and_then(|v| v.checked_add(atoms_cap as usize))
        .and_then(|v| v.checked_add(4));
    let Some(expect_stride) = expect_stride else {
        return Err(Error::shape("ion_evidence_scored: steps * 4 + atoms + 4 overflows usize".to_string()));
    };
    if ion.shape().dim(4) != 12 || ion_meta.shape().dim(3) != 4 {
        return Err(Error::shape(format!(
            "ion_evidence_scored needs last dimensions 12 and 4, got {} and {}",
            ion.shape(),
            ion_meta.shape()
        )));
    }
    let want_actions: &[usize] = &[rows, expect_stride];
    let want_slot: &[usize] = &[rows, 2];
    let want_ion: &[usize] = &[batch, f, n, j, 12];
    let want_im: &[usize] = &[batch, f, n, 4];
    let want_lp: &[usize] = &[batch, f, n, j + 1];
    let want_ev: &[usize] = &[rows, EVIDENCE_ROW_WORDS];
    if actions.shape().dims() != want_actions
        || traj_slot.shape().dims() != want_slot
        || ion.shape().dims() != want_ion
        || ion_meta.shape().dims() != want_im
        || log_prob.dims() != want_lp
        || evidence.shape().dims() != want_ev
    {
        return Err(Error::shape(format!(
            "ion_evidence_scored needs actions [{rows}, {expect_stride}], traj_slot [{rows}, 2], ion [{batch}, {f}, {n}, {j}, 12], ion_meta [{batch}, {f}, {n}, 4], log_prob [{batch}, {f}, {n}, {}] and evidence [{rows}, {}], got {} and {} and {} and {} and {} and {}",
            j + 1,
            EVIDENCE_ROW_WORDS,
            actions.shape(),
            traj_slot.shape(),
            ion.shape(),
            ion_meta.shape(),
            log_prob.shape(),
            evidence.shape()
        )));
    }
    let Some(expect_rows) = batch.checked_mul(per_spectrum as usize) else {
        return Err(Error::shape("ion_evidence_scored: B * K overflows usize".to_string()));
    };
    if per_spectrum == 0 || rows != expect_rows {
        return Err(Error::shape(format!(
            "ion_evidence_scored needs rows {rows} == B {batch} * K {per_spectrum}"
        )));
    }
    if rows == 0 {
        return Ok(());
    }
    check_device_len("actions", actions.len())?;
    check_device_len("traj_slot", traj_slot.len())?;
    check_device_len("ion", ion.len())?;
    check_device_len("ion_meta", ion_meta.len())?;
    check_device_len("log_prob", log_prob.len())?;
    check_device_len("evidence", evidence.len())?;
    check_device_scalar("ion_evidence_scored rows", rows)?;
    let f_u32 = check_device_scalar("ion_evidence_scored F", f)?;
    let n_u32 = check_device_scalar("ion_evidence_scored N", n)?;
    let j_u32 = check_device_scalar("ion_evidence_scored J", j)?;
    let client = actions.client();
    let per_lane = expect_stride.saturating_add(n.saturating_mul(j));
    let (count, dim, span) = launch_1d_spans(client, rows, per_lane.max(1));
    unsafe {
        ms2_ion_evidence_scored_kernel::launch_unchecked::<E, R>(
            client,
            count,
            dim,
            actions.arg(),
            traj_slot.arg(),
            ion.arg(),
            ion_meta.arg(),
            log_prob.arg(),
            evidence.arg(),
            steps,
            atoms_cap,
            per_spectrum,
            f_u32,
            n_u32,
            j_u32,
            rows,
            span,
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// `ion_traj_slot`: lane per trajectory building `[R, 2]` (slot, adduct).
// ---------------------------------------------------------------------------

/// Lane per trajectory of [`ion_traj_slot_fill`]: `slot` is word 0 of the
/// trajectory's `traj_alloc [B, K, 12]` row, `adduct` is word 3 of its
/// spectrum's `meta [B, 8]` row.
#[allow(unused_assignments)]
#[cube(launch_unchecked)]
fn ms2_ion_traj_slot_kernel(
    traj_alloc: &Array<u32>,
    meta: &Array<u32>,
    traj_slot: &mut Array<u32>,
    k_dim: u32,
    lanes: usize,
    span: usize,
) {
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    for pos in start..end {
        let r = pos as u32;
        let b = r / k_dim;
        let tbase = r * 12u32;
        let mbase = b * 8u32;
        let slot = traj_alloc[tbase as usize];
        let adduct = meta[(mbase + 3u32) as usize];
        traj_slot[(r * 2u32) as usize] = slot;
        traj_slot[(r * 2u32 + 1u32) as usize] = adduct;
    }
}

/// Build `[R, 2]` (formula slot, adduct id) per trajectory from
/// `traj_alloc [B, K, 12]` and `meta [B, 8]`. One launch, 3 arrays.
pub fn ion_traj_slot<R: Runtime>(
    traj_alloc: &IdTensor<R>,
    meta: &IdTensor<R>,
    traj_slot: &mut IdTensor<R>,
    per_spectrum: u32,
) -> Result<()> {
    if traj_alloc.shape().rank() != 3 || meta.shape().rank() != 2 || traj_slot.shape().rank() != 2
    {
        return Err(Error::shape(format!(
            "ion_traj_slot needs traj_alloc [B, K, 12], meta [B, 8] and traj_slot [R, 2], got {} and {} and {}",
            traj_alloc.shape(),
            meta.shape(),
            traj_slot.shape()
        )));
    }
    let batch = traj_alloc.shape().dim(0);
    let k = traj_alloc.shape().dim(1);
    let rows = batch.checked_mul(k).ok_or_else(|| {
        Error::shape("ion_traj_slot: B * K overflows usize".to_string())
    })?;
    if traj_alloc.shape().dims() != [batch, k, 12]
        || meta.shape().dims() != [batch, 8]
        || traj_slot.shape().dims() != [rows, 2]
    {
        return Err(Error::shape(format!(
            "ion_traj_slot needs traj_alloc [{batch}, {k}, 12], meta [{batch}, 8] and traj_slot [{rows}, 2], got {} and {} and {}",
            traj_alloc.shape(),
            meta.shape(),
            traj_slot.shape()
        )));
    }
    if per_spectrum as usize != k || rows == 0 {
        if rows == 0 {
            return Ok(());
        }
        return Err(Error::shape(format!(
            "ion_traj_slot needs per_spectrum {per_spectrum} == K {k}"
        )));
    }
    check_device_len("traj_alloc", traj_alloc.len())?;
    check_device_len("meta", meta.len())?;
    check_device_len("traj_slot", traj_slot.len())?;
    check_device_scalar("ion_traj_slot rows", rows)?;
    let k_u32 = check_device_scalar("ion_traj_slot K", k)?;
    let client = traj_alloc.client();
    let (count, dim, span) = launch_1d_spans(client, rows, 12);
    unsafe {
        ms2_ion_traj_slot_kernel::launch_unchecked::<R>(
            client,
            count,
            dim,
            traj_alloc.arg(),
            meta.arg(),
            traj_slot.arg(),
            k_u32,
            rows,
            span,
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// `ion_evidence_features`: lane per trajectory filling `evidence_f [R, 2]`.
// ---------------------------------------------------------------------------

/// Lane per trajectory of [`ion_evidence_features`]; a copy of
/// [`crate::models::ms2::ion::evidence_features_lane`] over `Array`s.
#[allow(clippy::too_many_arguments)]
#[allow(unused_assignments)]
#[cube(launch_unchecked)]
fn ms2_ion_evidence_features_kernel<F: Float + CubeElement>(
    evidence: &Array<u32>,
    log_prob: &Array<F>,
    traj_slot: &Array<u32>,
    kept: &Array<u32>,
    meta: &Array<u32>,
    evidence_f: &mut Array<F>,
    k_dim: u32,
    f_dim: u32,
    n: u32,
    j: u32,
    lanes: usize,
    span: usize,
) {
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    for pos in start..end {
        let r = pos as u32;
        let b = r / k_dim;
        let ebase = r * 18u32;
        let fbase = r * 2u32;
        let tbase = r * 2u32;
        let slot = traj_slot[tbase as usize];
        let mut slot_c: u32 = 0u32;
        if slot < f_dim {
            slot_c = slot;
        }
        let width = j + 1u32;
        let count = evidence[(ebase + 1u32) as usize];
        let neg_inf = F::new(-3.4028235e38);
        // Use a very small start for the minimum (large positive).
        let big = F::new(3.4028235e38);
        let mut max_lp = neg_inf;
        let mut min_unit = big;
        let mut q: u32 = 0u32;
        while q < 4u32 && q < count {
            let w = ebase + 2u32 + q * 4u32;
            let p = evidence[w as usize];
            let bq = evidence[(w + 1u32) as usize];
            let resid_ob = evidence[(w + 3u32) as usize];
            let apos = (b * f_dim + slot_c) * n + p;
            let lp_idx = apos * width + bq;
            let lp = log_prob[lp_idx as usize];
            if lp > max_lp {
                max_lp = lp;
            }
            let kept_base = (b * n + p) * 3u32;
            let mz = kept[(kept_base + 1u32) as usize];
            let ppm = meta[(b * 8u32 + 4u32) as usize];
            let hi = mz / 10000u32;
            let lo = mz % 10000u32;
            let qq = hi * ppm;
            let tol = qq / 1000u32 + ((qq % 1000u32) * 10000u32 + lo * ppm) / 10000000u32;
            let mut tol_f = F::new(1.0);
            if tol > 1u32 {
                // `tol` fits f32 exactly below 2^24; larger values lose
                // integer precision but stay finite and positive, which is
                // all the ratio needs. Cast via u32 -> f32 through the
                // backend's conversion (spelled as repeated addition? No:
                // CubeCL casts with `as`? Use `F::new`? We need a value
                // conversion, not a literal. Approximate by `tol as f32`
                // through `F::cast_from`? CubeCL `Float` has `from_int`?
                // Simplest portable form: `F::new(tol as f32)` is host-side;
                // on device use the raw bits? Instead, reuse `tol` in f32
                // arithmetic via `F::new(1.0) * F::new(tol as f32)`? The
                // `tol as f32` is a host cast inside the kernel macro — it
                // captures the host value, not the lane's. Correct approach:
                // `let tol_f = F::cast_from(tol)`? CubeCL 0.10 `Float`
                // provides `cast_from` for int->float. Use it.
                tol_f = F::cast_from(tol);
            }
            let bias = 2147483648u32;
            let mut rabs_u: u32 = 0u32;
            if resid_ob >= bias {
                rabs_u = resid_ob - bias;
            } else {
                rabs_u = bias - resid_ob;
            }
            let rabs_f = F::cast_from(rabs_u);
            let unit = rabs_f / tol_f;
            if unit < min_unit {
                min_unit = unit;
            }
            q += 1u32;
        }
        if count == 0u32 {
            evidence_f[fbase as usize] = F::new(0.0);
            evidence_f[(fbase + 1u32) as usize] = F::new(1.0);
        } else {
            // Non-finite guards by range (fast-math safe): NaN fails both
            // comparisons, so clamp via selection.
            let mut ml = max_lp;
            let mut mu = min_unit;
            // If `max_lp` stayed at the sentinel (no valid lookup), use 0.
            if ml == neg_inf {
                ml = F::new(0.0);
            }
            // If `min_unit` stayed big (should not happen with count>0), use 1.
            if mu == big {
                mu = F::new(1.0);
            }
            evidence_f[fbase as usize] = ml;
            evidence_f[(fbase + 1u32) as usize] = mu;
        }
    }
}

/// Fill `evidence_f [R, 2]` (largest log-prob, smallest |residual|/tol) per
/// trajectory. Six arrays, one launch.
#[allow(clippy::too_many_arguments)]
pub fn ion_evidence_features<R: Runtime, E: FloatElem>(
    evidence: &IdTensor<R>,
    log_prob: &Tensor<R, E>,
    traj_slot: &IdTensor<R>,
    kept: &IdTensor<R>,
    meta: &IdTensor<R>,
    evidence_f: &mut Tensor<R, E>,
    per_spectrum: u32,
) -> Result<()> {
    if evidence.shape().rank() != 2
        || log_prob.rank() != 4
        || traj_slot.shape().rank() != 2
        || kept.shape().rank() != 3
        || meta.shape().rank() != 2
        || evidence_f.rank() != 2
    {
        return Err(Error::shape(format!(
            "ion_evidence_features needs evidence [R, 18], log_prob [B, F, N, J + 1], traj_slot [R, 2], kept [B, N, 3], meta [B, 8] and evidence_f [R, 2], got {} and {} and {} and {} and {} and {}",
            evidence.shape(),
            log_prob.shape(),
            traj_slot.shape(),
            kept.shape(),
            meta.shape(),
            evidence_f.shape()
        )));
    }
    let rows = evidence.shape().dim(0);
    let batch = kept.shape().dim(0);
    let n = kept.shape().dim(1);
    let f = log_prob.shape().dim(1);
    let j = log_prob.shape().dim(3).saturating_sub(1);
    if evidence.shape().dims() != [rows, EVIDENCE_ROW_WORDS]
        || traj_slot.shape().dims() != [rows, 2]
        || evidence_f.shape().dims() != [rows, 2]
        || kept.shape().dims() != [batch, n, 3]
        || meta.shape().dims() != [batch, 8]
        || log_prob.shape().dims() != [batch, f, n, j + 1]
    {
        return Err(Error::shape(format!(
            "ion_evidence_features shape mismatch: got {} and {} and {} and {} and {} and {}",
            evidence.shape(),
            log_prob.shape(),
            traj_slot.shape(),
            kept.shape(),
            meta.shape(),
            evidence_f.shape()
        )));
    }
    let Some(expect_rows) = batch.checked_mul(per_spectrum as usize) else {
        return Err(Error::shape("ion_evidence_features: B * K overflows usize".to_string()));
    };
    if per_spectrum == 0 || rows != expect_rows {
        return Err(Error::shape(format!(
            "ion_evidence_features needs rows {rows} == B {batch} * K {per_spectrum}"
        )));
    }
    if rows == 0 {
        return Ok(());
    }
    check_device_len("evidence", evidence.len())?;
    check_device_len("log_prob", log_prob.len())?;
    check_device_len("traj_slot", traj_slot.len())?;
    check_device_len("kept", kept.len())?;
    check_device_len("meta", meta.len())?;
    check_device_len("evidence_f", evidence_f.len())?;
    check_device_scalar("ion_evidence_features rows", rows)?;
    let k_u32 = check_device_scalar("ion_evidence_features K", per_spectrum as usize)?;
    let f_u32 = check_device_scalar("ion_evidence_features F", f)?;
    let n_u32 = check_device_scalar("ion_evidence_features N", n)?;
    let j_u32 = check_device_scalar("ion_evidence_features J", j)?;
    let client = evidence.client();
    let (count, dim, span) = launch_1d_spans(client, rows, 18);
    unsafe {
        ms2_ion_evidence_features_kernel::launch_unchecked::<E, R>(
            client,
            count,
            dim,
            evidence.arg(),
            log_prob.arg(),
            traj_slot.arg(),
            kept.arg(),
            meta.arg(),
            evidence_f.arg(),
            k_u32,
            f_u32,
            n_u32,
            j_u32,
            rows,
            span,
        );
    }
    Ok(())
}
