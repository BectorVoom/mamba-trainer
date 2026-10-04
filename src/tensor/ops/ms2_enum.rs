//! Bounded-enumeration kernels of spec §1.4 (task B2a): count, offsets, fill
//! and pad over one `(B, P, M)` bucket.
//!
//! Four kernels, all `u32`, all launched through
//! [`crate::backend::launch_1d_spans`] with one lane per output item:
//!
//! * `ms2_enum_count`, lane per `(b, r)`: `meta [B, 8]`, `rare [P, 8]` and
//!   the packed `bounds` buffer → `lane_stats [B * P, 2]` (joined; visited
//!   with the exhausted flag in the top bit). 4 arrays.
//! * `ms2_enum_offsets`, lane per spectrum `b`: `lane_stats`, `meta` →
//!   `offsets [B * P]` (the clamped exclusive prefix sums) and `counters
//!   [B, 5]` (visited, joined, scored, status bits, complete). 4 arrays.
//! * `ms2_enum_fill`, lane per `(b, r)`: `meta`, `rare`, `bounds`,
//!   `offsets` → `cand [B, M, 13]` (13-word candidate records for ranks
//!   below the scored cap; the stats words are untouched in fill mode).
//!   5 arrays.
//! * `ms2_cand_pad`, lane per `(b, m)`: `counters` → `cand` (padding records
//!   for slots at or past `rows_scored`; slots below are owned by fill, so
//!   every element of `cand` has exactly one writer). 2 arrays.
//!
//! Count and fill share ONE [`#[cube]`] lane function with a mode argument,
//! a line-for-line copy of
//! [`crate::models::ms2::formula_enum::kernel_lane`] (with its `k_*` helpers
//! and [`decide_u32`](crate::models::ms2::formula_enum::decide_u32)); the
//! offsets and pad lanes likewise copy their host twins. The copies below
//! differ from the twins only in the buffer parameter types (`&Array<u32>`
//! for `&[u32]`): every body is otherwise identical, including the explicit
//! `u32` word lengths, the `max_u32` bound, the fourteen chemistry scalars
//! and the statement-`if` flag style the device IR can express.
//!
//! [`EnumChem`] carries those fourteen chemistry scalars on the host
//! (element masses and residuals in [`ELEMENTS`](crate::models::ms2::chem::ELEMENTS)
//! order); the wrappers fan them out as scalar launch arguments.
//!
//! Shape contract (checked before any launch, [`crate::error::Error::Shape`]
//! on mismatch, never a panic): `meta [B, 8]`, `rare [P, 8]`, `bounds`
//! rank 1, `lane_stats [B * P, 2]`, `offsets` rank 1 of length `B * P`,
//! `counters [B, 5]`, `cand [B, M, 13]`. The scored cap of a launch is
//! `min(meta word 6, scored_cap arg, M)`: the fill lane reads the
//! per-spectrum meta word and the wrapper-clamped `min(scored_cap, M)`, and
//! the offsets lane clamps the meta word against `min(scored_cap, M)`, so no
//! rank past the candidate buffer is ever addressed and fill/pad ownership
//! never overlaps (a cap-0 spectrum gets no fill write at all). Buffers of
//! one launch must not alias. Every wrapper enforces `B * P <= lanes_max`
//! (spec §1.4 `enum_lanes_max`, default 262,144, passed in) and validates
//! with checked host arithmetic that every dimension and every bound array's
//! largest address fits `u32` before any launch.
//!
//! CPU-runtime note: keep test domains small (lanes execute on the host).

// Kernel-shaped code (same reason as `tensor/ops/ms2.rs`): nested statement
// `if`s instead of `&&`, and guarded divisions/remainders instead of
// `checked_div` / `is_multiple_of` (neither exists on the device); all must
// stay identical to the host twins.
#![allow(clippy::collapsible_if)]
#![allow(clippy::manual_checked_ops)]
#![allow(clippy::manual_is_multiple_of)]

use cubecl::prelude::*;

use crate::backend::launch_1d_spans;
use crate::error::{Error, Result};
use crate::models::ms2::chem::{ELEMENTS, HYDROGEN};
use crate::models::ms2::contract::request_status;
use crate::models::ms2::formula_enum::{
    LANE_EXHAUSTED_BIT, LANE_MODE_COUNT, LANE_MODE_FILL, LANE_RECORD_WORDS, LANE_VISITED_MASK,
    LANE_VISITS_REPRESENTABLE_MAX, META_BOUND, META_BUDGET, META_HI, META_LEN, META_LO,
    META_PARENT, META_SCORED_CAP, META_TOL, PACK_CARBON_WIDTH, PACK_DBE_BIAS, PACK_HEADER_LEN,
    PACK_HEAVY_CAPS, PACK_HEAVY_MAX, PACK_HEAVY_WIDTH, PACK_HYDROGEN_MAX, PACK_HYDROGEN_MIN,
    PACK_N_CARBON_ROWS, PACK_N_HEAVY_ROWS, PACK_RARE_DISTINCT_HI, PACK_RARE_DISTINCT_LO,
    PACK_RARE_TOTAL_HI, PACK_RARE_TOTAL_LO, PACK_RATIO_HI_DEN, PACK_RATIO_HI_NUM,
    PACK_RATIO_LO_DEN, PACK_RATIO_LO_NUM, PACK_ZERO_CARBON,
};
use crate::tensor::ops::index::IdTensor;

/// The fourteen chemistry scalars the enumeration lanes take: the C, N, O
/// and H integer masses, then the ten rounding residuals in `ELEMENTS`
/// order. Built once per device setup with [`EnumChem::from_chemistry`] and
/// fanned out as scalar launch arguments by [`enum_count`] and [`enum_fill`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EnumChem {
    /// Integer mass of carbon.
    pub m_c: u32,
    /// Integer mass of nitrogen.
    pub m_n: u32,
    /// Integer mass of oxygen.
    pub m_o: u32,
    /// Integer mass of hydrogen.
    pub m_h: u32,
    /// Rounding residuals in `ELEMENTS` order.
    pub res: [u32; 10],
}

impl EnumChem {
    /// Read the masses and residuals off the frozen chemistry table.
    pub fn from_chemistry() -> Self {
        Self {
            m_c: ELEMENTS[0].mass,
            m_n: ELEMENTS[2].mass,
            m_o: ELEMENTS[3].mass,
            m_h: ELEMENTS[HYDROGEN].mass,
            res: [
                ELEMENTS[0].residual_nda,
                ELEMENTS[1].residual_nda,
                ELEMENTS[2].residual_nda,
                ELEMENTS[3].residual_nda,
                ELEMENTS[4].residual_nda,
                ELEMENTS[5].residual_nda,
                ELEMENTS[6].residual_nda,
                ELEMENTS[7].residual_nda,
                ELEMENTS[8].residual_nda,
                ELEMENTS[9].residual_nda,
            ],
        }
    }
}

/// One shared launch bundle for the enumerating source (spec §1.4): the
/// fourteen chemistry scalars of [`EnumChem`] plus the explicit buffer
/// lengths the wrappers pass as scalars, so generation and training call
/// [`enum_count`] and [`enum_fill`] the same way.
///
/// The scalars are the C, N, O and H integer masses plus the ten rounding
/// residuals in `ELEMENTS` order; the lengths (`meta_len`, `rare_len`,
/// `bounds_len`, `stats_len`, `cand_len`, …) travel as explicit `u32`
/// scalars because the kernels cannot query an array length. Both callers
/// build this once per device setup with [`EnumLaunch::from_chemistry`]
/// and fan it out through the same two methods.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EnumLaunch {
    /// The fourteen chemistry scalars.
    pub chem: EnumChem,
}

impl EnumLaunch {
    /// Read the scalars off the frozen chemistry table.
    pub fn from_chemistry() -> Self {
        Self {
            chem: EnumChem::from_chemistry(),
        }
    }

    /// Run [`enum_count`]: `meta [B, 8]`, `rare [P, 8]`, packed `bounds` →
    /// `lane_stats [B * P, 2]`. Bounded dispatch (spec §1.4 D10):
    /// `ceil(B * P / lanes_per_dispatch)` launches over contiguous lane
    /// ranges, each at most about `dispatch_visits_max` visits.
    /// `lanes_max` is the `B * P` ceiling (spec §1.4 `enum_lanes_max`).
    /// Each dispatch chunk is submitted on its own, so the chunking bounds
    /// the work of one GPU job to about `dispatch_visits_max` visits
    /// (a precautionary work bound, not a measured cure for any reset).
    /// The lane enforces `min(metadata budget, lane_visits_max)` as its
    /// visit bound, in count and fill identically, so the chunk sizing
    /// assumption always holds.
    pub fn count<R: Runtime>(
        &self,
        meta: &IdTensor<R>,
        rare: &IdTensor<R>,
        bounds: &IdTensor<R>,
        lane_stats: &IdTensor<R>,
        lanes_max: u32,
        dispatch_visits_max: u32,
        lane_visits_max: u32,
    ) -> Result<()> {
        enum_count(
            meta,
            rare,
            bounds,
            lane_stats,
            &self.chem,
            lanes_max,
            dispatch_visits_max,
            lane_visits_max,
        )
    }

    /// Run [`enum_fill`]: `meta [B, 8]`, `rare [P, 8]`, packed `bounds`,
    /// `offsets [B * P]` → `cand [B, M, 13]`. Bounded dispatch as in
    /// [`EnumLaunch::count`]. The effective cap is `min(meta word 6,
    /// scored_cap, M)`, identical to offsets/pad. `lanes_max` is the `B * P`
    /// ceiling. Each dispatch chunk is submitted on its own, so the chunking
    /// bounds the work of one GPU job as in [`EnumLaunch::count`]
    /// (a precautionary work bound).
    pub fn fill<R: Runtime>(
        &self,
        meta: &IdTensor<R>,
        rare: &IdTensor<R>,
        bounds: &IdTensor<R>,
        offsets: &IdTensor<R>,
        cand: &IdTensor<R>,
        scored_cap: u32,
        lanes_max: u32,
        dispatch_visits_max: u32,
        lane_visits_max: u32,
    ) -> Result<()> {
        enum_fill(
            meta,
            rare,
            bounds,
            offsets,
            cand,
            &self.chem,
            scored_cap,
            lanes_max,
            dispatch_visits_max,
            lane_visits_max,
        )
    }
}

/// `len` as a `u32` word count, or [`Error::Shape`] when unrepresentable.
fn u32_len(len: usize, what: &str) -> Result<u32> {
    u32::try_from(len)
        .map_err(|_| Error::shape(format!("ms2_enum: {what} length {len} exceeds u32")))
}

// ---------------------------------------------------------------------------
// Kernel-lane copies (line-for-line copies of the host twins, `&Array<u32>`
// for `&[u32]`; see the module docs).
// ---------------------------------------------------------------------------

#[cube]
fn decide_u32(observed: u32, computed: u32, error: u32, tolerance: u32) -> u32 {
    let mut r: u32 = 0u32;
    if observed >= computed {
        r = observed - computed;
    }
    if computed > observed {
        r = computed - observed;
    }
    let mut accept: u32 = 0u32;
    if r <= tolerance {
        if error <= tolerance - r {
            accept = 1u32;
        }
    }
    let mut reject: u32 = 0u32;
    if r > tolerance {
        if r - tolerance > error {
            reject = 1u32;
        }
    }
    let mut flag: u32 = 2u32;
    if accept != 0u32 {
        flag = 1u32;
    }
    if reject != 0u32 {
        flag = 0u32;
    }
    flag
}

#[cube]
fn sat_add_counter(a: u32, b: u32, max_u32: u32) -> u32 {
    let mut s: u32 = max_u32 - 1u32;
    if a < max_u32 - b {
        s = a + b;
    }
    s
}

#[cube]
fn sat_add_saturates(a: u32, b: u32, max_u32: u32) -> u32 {
    let lim = max_u32 - 1u32;
    let mut sat: u32 = 0u32;
    if b > lim {
        sat = 1u32;
    }
    if b <= lim {
        if a > lim - b {
            sat = 1u32;
        }
    }
    sat
}

#[cube]
fn k_read(words: &Array<u32>, index: u32, len: u32) -> u32 {
    let mut v: u32 = 0u32;
    if index < len {
        v = words[index as usize];
    }
    v
}

#[cube]
fn k_write(words: &mut Array<u32>, index: u32, value: u32, len: u32) {
    if index < len {
        words[index as usize] = value;
    }
}

#[cube]
fn k_span_ok(base: u32, count: u32, stride: u32, max_u32: u32) -> u32 {
    let mut ok: u32 = 0u32;
    if stride == 0u32 {
        ok = 1u32;
    }
    if stride != 0u32 {
        if count <= (max_u32 - base) / stride {
            ok = 1u32;
        }
    }
    ok
}

#[cube]
fn k_span_at(base: u32, count: u32, stride: u32, max_u32: u32) -> u32 {
    let mut addr: u32 = 0u32;
    if stride == 0u32 {
        addr = base;
    }
    if stride != 0u32 {
        if count <= (max_u32 - base) / stride {
            addr = base + count * stride;
        }
    }
    addr
}

#[cube]
fn k_at(base: u32, index: u32, stride: u32, max_u32: u32) -> u32 {
    let mut addr: u32 = max_u32;
    if stride == 0u32 {
        addr = base;
    }
    if stride != 0u32 {
        if index <= (max_u32 - base) / stride {
            addr = base + index * stride;
        }
    }
    addr
}

#[cube]
fn k_allow_ok(hi: u32, base: u32, mass: u32) -> u32 {
    let mut ok: u32 = 0u32;
    if mass != 0u32 {
        if base <= hi {
            ok = 1u32;
        }
    }
    ok
}

#[cube]
fn k_allow_n(hi: u32, base: u32, mass: u32) -> u32 {
    let mut n: u32 = 0u32;
    if mass != 0u32 {
        if base <= hi {
            n = (hi - base) / mass;
        }
    }
    n
}

#[cube]
fn k_over_total(rare_sum: u32, c: u32, n: u32, o: u32, max: u32) -> u32 {
    let mut over: u32 = 1u32;
    if rare_sum <= max {
        if c <= max - rare_sum {
            let t1 = rare_sum + c;
            if n <= max - t1 {
                let t2 = t1 + n;
                if o <= max - t2 {
                    over = 0u32;
                }
            }
        }
    }
    over
}

#[cube]
fn k_total_ok(rare_sum: u32, c: u32, n: u32, o: u32, max_u32: u32) -> u32 {
    let mut ok: u32 = 0u32;
    if c <= max_u32 - rare_sum {
        let t1 = rare_sum + c;
        if n <= max_u32 - t1 {
            let t2 = t1 + n;
            if o <= max_u32 - t2 {
                ok = 1u32;
            }
        }
    }
    ok
}

#[cube]
fn k_total_v(rare_sum: u32, c: u32, n: u32, o: u32, max_u32: u32) -> u32 {
    let mut v: u32 = 0u32;
    if c <= max_u32 - rare_sum {
        let t1 = rare_sum + c;
        if n <= max_u32 - t1 {
            let t2 = t1 + n;
            if o <= max_u32 - t2 {
                v = t2 + o;
            }
        }
    }
    v
}

#[allow(clippy::too_many_arguments)]
#[cube]
fn k_exact_ok(
    c: u32,
    h: u32,
    n: u32,
    f: u32,
    p: u32,
    s: u32,
    cl: u32,
    br: u32,
    ii: u32,
    max_u32: u32,
) -> u32 {
    let mut pos: u32 = 2u32;
    let mut ok: u32 = 1u32;
    if c > (max_u32 - pos) / 2u32 {
        ok = 0u32;
    }
    if ok != 0u32 {
        pos += c * 2u32;
    }
    if n > max_u32 - pos {
        ok = 0u32;
    }
    if ok != 0u32 {
        pos += n;
    }
    if p > (max_u32 - pos) / 3u32 {
        ok = 0u32;
    }
    if ok != 0u32 {
        pos += p * 3u32;
    }
    if s > (max_u32 - pos) / 4u32 {
        ok = 0u32;
    }
    if ok != 0u32 {
        pos += s * 4u32;
    }
    let mut neg: u32 = h;
    if f > max_u32 - neg {
        ok = 0u32;
    }
    if ok != 0u32 {
        neg += f;
    }
    if cl > max_u32 - neg {
        ok = 0u32;
    }
    if ok != 0u32 {
        neg += cl;
    }
    if br > max_u32 - neg {
        ok = 0u32;
    }
    if ok != 0u32 {
        neg += br;
    }
    if ii > max_u32 - neg {
        ok = 0u32;
    }
    if ok != 0u32 {
        neg += ii;
    }
    if neg > pos {
        ok = 0u32;
    }
    if pos % 2u32 != neg % 2u32 {
        ok = 0u32;
    }
    ok
}

#[allow(clippy::too_many_arguments)]
#[cube]
fn k_exact_pos(
    c: u32,
    h: u32,
    n: u32,
    f: u32,
    p: u32,
    s: u32,
    cl: u32,
    br: u32,
    ii: u32,
    max_u32: u32,
) -> u32 {
    let mut pos: u32 = 2u32;
    let mut ok: u32 = 1u32;
    if c > (max_u32 - pos) / 2u32 {
        ok = 0u32;
    }
    if ok != 0u32 {
        pos += c * 2u32;
    }
    if n > max_u32 - pos {
        ok = 0u32;
    }
    if ok != 0u32 {
        pos += n;
    }
    if p > (max_u32 - pos) / 3u32 {
        ok = 0u32;
    }
    if ok != 0u32 {
        pos += p * 3u32;
    }
    if s > (max_u32 - pos) / 4u32 {
        ok = 0u32;
    }
    if ok != 0u32 {
        pos += s * 4u32;
    }
    let mut neg: u32 = h;
    if f > max_u32 - neg {
        ok = 0u32;
    }
    if ok != 0u32 {
        neg += f;
    }
    if cl > max_u32 - neg {
        ok = 0u32;
    }
    if ok != 0u32 {
        neg += cl;
    }
    if br > max_u32 - neg {
        ok = 0u32;
    }
    if ok != 0u32 {
        neg += br;
    }
    if ii > max_u32 - neg {
        ok = 0u32;
    }
    if ok != 0u32 {
        neg += ii;
    }
    if neg > pos {
        ok = 0u32;
    }
    if pos % 2u32 != neg % 2u32 {
        ok = 0u32;
    }
    let mut out: u32 = 0u32;
    if ok != 0u32 {
        out = pos;
    }
    out
}

#[allow(clippy::too_many_arguments)]
#[cube]
fn k_exact_neg(
    c: u32,
    h: u32,
    n: u32,
    f: u32,
    p: u32,
    s: u32,
    cl: u32,
    br: u32,
    ii: u32,
    max_u32: u32,
) -> u32 {
    let mut pos: u32 = 2u32;
    let mut ok: u32 = 1u32;
    if c > (max_u32 - pos) / 2u32 {
        ok = 0u32;
    }
    if ok != 0u32 {
        pos += c * 2u32;
    }
    if n > max_u32 - pos {
        ok = 0u32;
    }
    if ok != 0u32 {
        pos += n;
    }
    if p > (max_u32 - pos) / 3u32 {
        ok = 0u32;
    }
    if ok != 0u32 {
        pos += p * 3u32;
    }
    if s > (max_u32 - pos) / 4u32 {
        ok = 0u32;
    }
    if ok != 0u32 {
        pos += s * 4u32;
    }
    let mut neg: u32 = h;
    if f > max_u32 - neg {
        ok = 0u32;
    }
    if ok != 0u32 {
        neg += f;
    }
    if cl > max_u32 - neg {
        ok = 0u32;
    }
    if ok != 0u32 {
        neg += cl;
    }
    if br > max_u32 - neg {
        ok = 0u32;
    }
    if ok != 0u32 {
        neg += br;
    }
    if ii > max_u32 - neg {
        ok = 0u32;
    }
    if ok != 0u32 {
        neg += ii;
    }
    if neg > pos {
        ok = 0u32;
    }
    if pos % 2u32 != neg % 2u32 {
        ok = 0u32;
    }
    let mut out: u32 = 0u32;
    if ok != 0u32 {
        out = neg;
    }
    out
}

#[allow(clippy::too_many_arguments)]
#[cube]
fn k_verdict(
    parent: u32,
    tol: u32,
    bound: u32,
    mass: u32,
    c: u32,
    h: u32,
    n: u32,
    o: u32,
    f: u32,
    p: u32,
    s: u32,
    cl: u32,
    br: u32,
    ii: u32,
    res_c: u32,
    res_h: u32,
    res_n: u32,
    res_o: u32,
    res_f: u32,
    res_p: u32,
    res_s: u32,
    res_cl: u32,
    res_br: u32,
    res_ii: u32,
    max_u32: u32,
) -> u32 {
    let mut nda: u32 = 0u32;
    let mut have: u32 = 1u32;
    if res_c != 0u32 {
        if c > (max_u32 - nda) / res_c {
            have = 0u32;
        }
    }
    if have != 0u32 {
        nda += c * res_c;
    }
    if res_h != 0u32 {
        if h > (max_u32 - nda) / res_h {
            have = 0u32;
        }
    }
    if have != 0u32 {
        nda += h * res_h;
    }
    if res_n != 0u32 {
        if n > (max_u32 - nda) / res_n {
            have = 0u32;
        }
    }
    if have != 0u32 {
        nda += n * res_n;
    }
    if res_o != 0u32 {
        if o > (max_u32 - nda) / res_o {
            have = 0u32;
        }
    }
    if have != 0u32 {
        nda += o * res_o;
    }
    if res_f != 0u32 {
        if f > (max_u32 - nda) / res_f {
            have = 0u32;
        }
    }
    if have != 0u32 {
        nda += f * res_f;
    }
    if res_p != 0u32 {
        if p > (max_u32 - nda) / res_p {
            have = 0u32;
        }
    }
    if have != 0u32 {
        nda += p * res_p;
    }
    if res_s != 0u32 {
        if s > (max_u32 - nda) / res_s {
            have = 0u32;
        }
    }
    if have != 0u32 {
        nda += s * res_s;
    }
    if res_cl != 0u32 {
        if cl > (max_u32 - nda) / res_cl {
            have = 0u32;
        }
    }
    if have != 0u32 {
        nda += cl * res_cl;
    }
    if res_br != 0u32 {
        if br > (max_u32 - nda) / res_br {
            have = 0u32;
        }
    }
    if have != 0u32 {
        nda += br * res_br;
    }
    if res_ii != 0u32 {
        if ii > (max_u32 - nda) / res_ii {
            have = 0u32;
        }
    }
    if have != 0u32 {
        nda += ii * res_ii;
    }
    // `nda / 1000 <= u32::MAX / 1000`, so the `+ 1` cannot wrap; the query
    // bound adds saturating.
    let mut flag: u32 = 0u32;
    if have != 0u32 {
        let mut error: u32 = nda / 1000u32;
        if nda % 1000u32 != 0u32 {
            error += 1u32;
        }
        if error > max_u32 - bound {
            error = max_u32;
        }
        if error <= max_u32 - bound {
            error += bound;
        }
        flag = decide_u32(parent, mass, error, tol);
    }
    flag
}

#[allow(clippy::too_many_arguments)]
#[cube]
fn k_buckets(
    c: u32,
    n: u32,
    o: u32,
    f: u32,
    p: u32,
    s: u32,
    cl: u32,
    br: u32,
    ii: u32,
    rare_sum: u32,
    bounds: &Array<u32>,
    bounds_len: u32,
    n_carbon: u32,
    n_heavy: u32,
    cb_width: u32,
    hb_width: u32,
    caps_c: u32,
    seen_c: u32,
    caps_h: u32,
    seen_h: u32,
    max_u32: u32,
) -> u32 {
    let total_ok = k_total_ok(rare_sum, c, n, o, max_u32);
    let mut ok: u32 = 0u32;
    if total_ok != 0u32 {
        if cb_width != 0u32 {
            if hb_width != 0u32 {
                ok = 1u32;
            }
        }
    }
    let total_v = k_total_v(rare_sum, c, n, o, max_u32);
    let mut cb: u32 = 0u32;
    if cb_width != 0u32 {
        cb = c / cb_width;
    }
    let mut hb: u32 = 0u32;
    if hb_width != 0u32 {
        hb = total_v / hb_width;
    }
    if cb >= n_carbon {
        ok = 0u32;
    }
    if hb >= n_heavy {
        ok = 0u32;
    }
    let seen_c_addr = k_at(seen_c, cb, 1u32, max_u32);
    let seen_c_v = k_read(bounds, seen_c_addr, bounds_len);
    if seen_c_v != 1u32 {
        ok = 0u32;
    }
    let seen_h_addr = k_at(seen_h, hb, 1u32, max_u32);
    let seen_h_v = k_read(bounds, seen_h_addr, bounds_len);
    if seen_h_v != 1u32 {
        ok = 0u32;
    }
    let crow = k_at(caps_c, cb, 9u32, max_u32);
    let hrow = k_at(caps_h, hb, 9u32, max_u32);
    // The nine cap comparisons as a loop over the element position (the
    // kernel has no local arrays, so the count selects through an `if`
    // chain): reusing the same few variables keeps the backend's value
    // table small, where the unrolled form broke its GVN pass.
    let mut caps_ok: u32 = ok;
    let mut q: u32 = 0u32;
    let mut q_more: u32 = 1u32;
    while q_more != 0u32 {
        let mut have_q: u32 = c;
        if q == 1u32 {
            have_q = n;
        }
        if q == 2u32 {
            have_q = o;
        }
        if q == 3u32 {
            have_q = f;
        }
        if q == 4u32 {
            have_q = p;
        }
        if q == 5u32 {
            have_q = s;
        }
        if q == 6u32 {
            have_q = cl;
        }
        if q == 7u32 {
            have_q = br;
        }
        if q == 8u32 {
            have_q = ii;
        }
        let crow_addr = k_at(crow, q, 1u32, max_u32);
        let crow_cap = k_read(bounds, crow_addr, bounds_len);
        if have_q > crow_cap {
            caps_ok = 0u32;
        }
        if caps_ok != 0u32 {
            let hrow_addr = k_at(hrow, q, 1u32, max_u32);
            let hrow_cap = k_read(bounds, hrow_addr, bounds_len);
            if have_q > hrow_cap {
                caps_ok = 0u32;
            }
        }
        if q == 8u32 {
            q_more = 0u32;
        }
        if q_more != 0u32 {
            q += 1u32;
        }
    }
    caps_ok
}

#[allow(clippy::too_many_arguments)]
#[cube]
fn k_ratios(
    h: u32,
    n: u32,
    o: u32,
    c: u32,
    f: u32,
    p: u32,
    s: u32,
    cl: u32,
    br: u32,
    ii: u32,
    bounds: &Array<u32>,
    bounds_len: u32,
    max_u32: u32,
) -> u32 {
    let mut hal: u32 = 0u32;
    let mut ok: u32 = 1u32;
    if f > max_u32 - cl {
        ok = 0u32;
    }
    if ok != 0u32 {
        hal = f + cl;
        if br > max_u32 - hal {
            ok = 0u32;
        }
    }
    if ok != 0u32 {
        hal += br;
        if ii > max_u32 - hal {
            ok = 0u32;
        }
    }
    if ok != 0u32 {
        hal += ii;
    }
    let hal_v = hal;
    let zc = k_read(bounds, PACK_ZERO_CARBON, bounds_len);
    let mut k: u32 = 0u32;
    let mut k_more: u32 = 1u32;
    while k_more != 0u32 {
        let mut num: u32 = h;
        if k == 1u32 {
            num = n;
        }
        if k == 2u32 {
            num = o;
        }
        if k == 3u32 {
            num = hal_v;
        }
        if k == 4u32 {
            num = s;
        }
        if k == 5u32 {
            num = p;
        }
        let mut stage_ok: u32 = 0u32;
        if c == 0u32 {
            if zc == 1u32 {
                stage_ok = 1u32;
            }
        }
        if c != 0u32 {
            let lo_num = k_read(bounds, PACK_RATIO_LO_NUM + k, bounds_len);
            let lo_den = k_read(bounds, PACK_RATIO_LO_DEN + k, bounds_len);
            let hi_num = k_read(bounds, PACK_RATIO_HI_NUM + k, bounds_len);
            let hi_den = k_read(bounds, PACK_RATIO_HI_DEN + k, bounds_len);
            let mut prod_ok: u32 = 1u32;
            let mut a: u32 = 0u32;
            if lo_den != 0u32 {
                if num <= max_u32 / lo_den {
                    a = num * lo_den;
                }
                if num > max_u32 / lo_den {
                    prod_ok = 0u32;
                }
            }
            if lo_den == 0u32 {
                prod_ok = 0u32;
            }
            let mut b: u32 = 0u32;
            if lo_num != 0u32 {
                if c <= max_u32 / lo_num {
                    b = lo_num * c;
                }
                if c > max_u32 / lo_num {
                    prod_ok = 0u32;
                }
            }
            let mut cc: u32 = 0u32;
            if hi_den != 0u32 {
                if num <= max_u32 / hi_den {
                    cc = num * hi_den;
                }
                if num > max_u32 / hi_den {
                    prod_ok = 0u32;
                }
            }
            if hi_den == 0u32 {
                prod_ok = 0u32;
            }
            let mut d: u32 = 0u32;
            if hi_num != 0u32 {
                if c <= max_u32 / hi_num {
                    d = hi_num * c;
                }
                if c > max_u32 / hi_num {
                    prod_ok = 0u32;
                }
            }
            let mut cmp_ok: u32 = 0u32;
            if prod_ok != 0u32 {
                if a >= b {
                    if cc <= d {
                        cmp_ok = 1u32;
                    }
                }
            }
            if lo_den == 0u32 {
                cmp_ok = 0u32;
            }
            if hi_den == 0u32 {
                cmp_ok = 0u32;
            }
            stage_ok = cmp_ok;
        }
        if stage_ok == 0u32 {
            ok = 0u32;
        }
        if k == 5u32 {
            k_more = 0u32;
        }
        if k_more != 0u32 {
            k += 1u32;
        }
    }
    ok
}

#[allow(clippy::too_many_arguments)]
#[cube]
fn k_dbe(
    pos: u32,
    neg: u32,
    rare_sum: u32,
    c: u32,
    n: u32,
    o: u32,
    bounds: &Array<u32>,
    bounds_len: u32,
    n_heavy: u32,
    hb_width: u32,
    seen_h: u32,
    dbe_base: u32,
    max_u32: u32,
) -> u32 {
    let mut ok: u32 = 0u32;
    if k_total_ok(rare_sum, c, n, o, max_u32) != 0u32 {
        if hb_width != 0u32 {
            ok = 1u32;
        }
    }
    let total_v = k_total_v(rare_sum, c, n, o, max_u32);
    let mut hb: u32 = 0u32;
    if hb_width != 0u32 {
        hb = total_v / hb_width;
    }
    if hb >= n_heavy {
        ok = 0u32;
    }
    if k_read(bounds, k_at(seen_h, hb, 1u32, max_u32), bounds_len) != 1u32 {
        ok = 0u32;
    }
    let lo_addr = k_at(dbe_base, hb, 2u32, max_u32);
    let mut hi_addr: u32 = max_u32;
    if lo_addr < max_u32 {
        hi_addr = lo_addr + 1u32;
    }
    let slo = k_read(bounds, lo_addr, bounds_len);
    let shi = k_read(bounds, hi_addr, bounds_len);
    let mut range_ok: u32 = 0u32;
    if pos <= max_u32 - PACK_DBE_BIAS {
        let lhs = pos + PACK_DBE_BIAS;
        if neg <= max_u32 - slo {
            let rhs_lo = neg + slo;
            if neg <= max_u32 - shi {
                let rhs_hi = neg + shi;
                if lhs >= rhs_lo {
                    if lhs <= rhs_hi {
                        range_ok = 1u32;
                    }
                }
            }
        }
    }
    if ok == 0u32 {
        range_ok = 0u32;
    }
    range_ok
}

#[allow(clippy::too_many_arguments)]
#[cube]
fn kernel_lane(
    meta: &Array<u32>,
    meta_len: u32,
    bounds: &Array<u32>,
    bounds_len: u32,
    rare: &Array<u32>,
    rare_len: u32,
    mbase: u32,
    rbase: u32,
    mode: u32,
    offset: u32,
    cap: u32,
    buf: &mut Array<u32>,
    buf_len: u32,
    out_base: u32,
    stats_base: u32,
    m_c: u32,
    m_n: u32,
    m_o: u32,
    m_h: u32,
    res_c: u32,
    res_h: u32,
    res_n: u32,
    res_o: u32,
    res_f: u32,
    res_p: u32,
    res_s: u32,
    res_cl: u32,
    res_br: u32,
    res_ii: u32,
    max_u32: u32,
    lane_budget: u32,
) -> u32 {
    let parent = k_read(meta, mbase + META_PARENT, meta_len);
    let tol = k_read(meta, mbase + META_TOL, meta_len);
    let bound = k_read(meta, mbase + META_BOUND, meta_len);
    let lo = k_read(meta, mbase + META_LO, meta_len);
    let hi = k_read(meta, mbase + META_HI, meta_len);
    let budget = k_read(meta, mbase + META_BUDGET, meta_len);
    // The lane enforces the visit bound the chunk sizing assumes:
    // `min(metadata budget, lane_budget)`, then the representable clamp.
    // `budget_eff` starts from the buffer-loaded budget (never from a copy
    // of the scalar argument), in count and fill identically so ranks agree.
    let mut budget_eff: u32 = budget;
    if budget_eff > lane_budget {
        budget_eff = lane_budget;
    }
    if budget_eff > LANE_VISITS_REPRESENTABLE_MAX {
        budget_eff = LANE_VISITS_REPRESENTABLE_MAX;
    }
    // One effective scored cap (finding 1): `min(meta word 6, cap arg)`.
    // In count mode `cap` is 0 and unused, so the effective cap equals the
    // arg there; in fill mode the lane reads the per-spectrum word too, so
    // offsets, fill and pad agree slot for slot. A spectrum with cap 0 gets
    // no fill write at all.
    let mcap = k_read(meta, mbase + META_SCORED_CAP, meta_len);
    let mut eff_cap: u32 = cap;
    if mode == LANE_MODE_FILL {
        if mcap < eff_cap {
            eff_cap = mcap;
        }
    }
    let f = k_read(rare, rbase, rare_len);
    let p = k_read(rare, rbase + 1u32, rare_len);
    let s = k_read(rare, rbase + 2u32, rare_len);
    let cl = k_read(rare, rbase + 3u32, rare_len);
    let br = k_read(rare, rbase + 4u32, rare_len);
    let ii = k_read(rare, rbase + 5u32, rare_len);
    let rare_mass = k_read(rare, rbase + 6u32, rare_len);
    let rare_sum = k_read(rare, rbase + 7u32, rare_len);
    let mut distinct: u32 = 0u32;
    if f > 0u32 {
        distinct += 1u32;
    }
    if p > 0u32 {
        distinct += 1u32;
    }
    if s > 0u32 {
        distinct += 1u32;
    }
    if cl > 0u32 {
        distinct += 1u32;
    }
    if br > 0u32 {
        distinct += 1u32;
    }
    if ii > 0u32 {
        distinct += 1u32;
    }
    let n_carbon = k_read(bounds, PACK_N_CARBON_ROWS, bounds_len);
    let n_heavy = k_read(bounds, PACK_N_HEAVY_ROWS, bounds_len);
    let cb_width = k_read(bounds, PACK_CARBON_WIDTH, bounds_len);
    let hb_width = k_read(bounds, PACK_HEAVY_WIDTH, bounds_len);
    let caps_c = PACK_HEADER_LEN;
    let seen_c = k_span_at(caps_c, n_carbon, 9u32, max_u32);
    let caps_h = k_span_at(seen_c, n_carbon, 1u32, max_u32);
    let seen_h = k_span_at(caps_h, n_heavy, 9u32, max_u32);
    let dbe_base = k_span_at(seen_h, n_heavy, 1u32, max_u32);
    let mut lane_ok: u32 = 0u32;
    if cb_width != 0u32 {
        if hb_width != 0u32 {
            if k_span_ok(caps_c, n_carbon, 9u32, max_u32) != 0u32 {
                if k_span_ok(seen_c, n_carbon, 1u32, max_u32) != 0u32 {
                    if k_span_ok(caps_h, n_heavy, 9u32, max_u32) != 0u32 {
                        if k_span_ok(seen_h, n_heavy, 1u32, max_u32) != 0u32 {
                            lane_ok = 1u32;
                        }
                    }
                }
            }
        }
    }
    if rare_sum < k_read(bounds, PACK_RARE_TOTAL_LO, bounds_len) {
        lane_ok = 0u32;
    }
    if rare_sum > k_read(bounds, PACK_RARE_TOTAL_HI, bounds_len) {
        lane_ok = 0u32;
    }
    if distinct < k_read(bounds, PACK_RARE_DISTINCT_LO, bounds_len) {
        lane_ok = 0u32;
    }
    if distinct > k_read(bounds, PACK_RARE_DISTINCT_HI, bounds_len) {
        lane_ok = 0u32;
    }
    if rare_mass > hi {
        lane_ok = 0u32;
    }
    // Fill capacity exit (finding 4): a fill lane whose first rank is at or
    // above the effective cap owns no slots, so it enumerates nothing
    // (visited stays 0). Count mode is unchanged. No early `return` (the
    // twin has none): clearing `lane_ok` skips the same traversal.
    if mode == LANE_MODE_FILL {
        if offset >= eff_cap {
            lane_ok = 0u32;
        }
    }
    let cap_c = k_read(bounds, PACK_HEAVY_CAPS, bounds_len);
    let cap_n = k_read(bounds, PACK_HEAVY_CAPS + 1u32, bounds_len);
    let cap_o = k_read(bounds, PACK_HEAVY_CAPS + 2u32, bounds_len);
    let h_min = k_read(bounds, PACK_HYDROGEN_MIN, bounds_len);
    let h_max = k_read(bounds, PACK_HYDROGEN_MAX, bounds_len);
    let heavy_max = k_read(bounds, PACK_HEAVY_MAX, bounds_len);
    let mut visited: u32 = 0;
    let mut joined: u32 = 0;
    let mut exhausted: u32 = 0u32;
    let mut stop: u32 = 0u32;
    if lane_ok != 0u32 {
        let mut c: u32 = 0;
        let mut c_done: u32 = 0u32;
        let mut c_more: u32 = 1u32;
        while c_more != 0u32 {
            // Guard FIRST: `c` is admitted only while `c * m_c <= hi -
            // rare_mass` (past it the level ends: larger `c` only adds mass
            // and atoms). The product below is formed only in the admitted
            // branch.
            let c_allow_ok = k_allow_ok(hi, rare_mass, m_c);
            let c_allow = k_allow_n(hi, rare_mass, m_c);
            let mut c_over: u32 = 0u32;
            if c_allow_ok == 0u32 {
                c_over = 1u32;
            }
            if c > c_allow {
                c_over = 1u32;
            }
            if k_over_total(rare_sum, c, 0u32, 0u32, heavy_max) != 0u32 {
                c_over = 1u32;
            }
            let mut c_live: u32 = 0u32;
            if c_allow_ok != 0u32 {
                if c_over == 0u32 {
                    c_live = 1u32;
                }
            }
            if c_over != 0u32 {
                c_done = 1u32;
            }
            if c_allow_ok == 0u32 {
                c_done = 1u32;
            }
            if c_done != 0u32 {
                c_live = 0u32;
            }
            let mut c_prod: u32 = 0u32;
            let mut c_prod_ok: u32 = 0u32;
            if c_live != 0u32 {
                if m_c != 0u32 {
                    if c <= max_u32 / m_c {
                        c_prod = c * m_c;
                        c_prod_ok = 1u32;
                    }
                }
            }
            let mut m_after_c: u32 = hi;
            if c_prod_ok != 0u32 {
                if c_prod <= max_u32 - rare_mass {
                    m_after_c = rare_mass + c_prod;
                }
            }
            let mut c_go: u32 = 0u32;
            if c_live != 0u32 {
                if c_prod_ok != 0u32 {
                    if m_after_c <= hi {
                        c_go = 1u32;
                    }
                }
            }
            if c_go != 0u32 {
                let mut n: u32 = 0;
                let mut n_done: u32 = 0u32;
                let mut n_more: u32 = 1u32;
                while n_more != 0u32 {
                    // Same guard shape for `n`: compared first, multiplied
                    // only when admitted.
                    let n_allow_ok = k_allow_ok(hi, m_after_c, m_n);
                    let n_allow = k_allow_n(hi, m_after_c, m_n);
                    let mut n_over: u32 = 0u32;
                    if n_allow_ok == 0u32 {
                        n_over = 1u32;
                    }
                    if n > n_allow {
                        n_over = 1u32;
                    }
                    if k_over_total(rare_sum, c, n, 0u32, heavy_max) != 0u32 {
                        n_over = 1u32;
                    }
                    let mut n_live: u32 = 0u32;
                    if c_go != 0u32 {
                        if n_allow_ok != 0u32 {
                            if n_over == 0u32 {
                                n_live = 1u32;
                            }
                        }
                    }
                    if n_over != 0u32 {
                        n_done = 1u32;
                    }
                    if n_allow_ok == 0u32 {
                        n_done = 1u32;
                    }
                    if n_done != 0u32 {
                        n_live = 0u32;
                    }
                    let mut n_prod: u32 = 0u32;
                    let mut n_prod_ok: u32 = 0u32;
                    if n_live != 0u32 {
                        if m_n != 0u32 {
                            if n <= max_u32 / m_n {
                                n_prod = n * m_n;
                                n_prod_ok = 1u32;
                            }
                        }
                    }
                    let mut m_after_n: u32 = hi;
                    if n_prod_ok != 0u32 {
                        if n_prod <= max_u32 - m_after_c {
                            m_after_n = m_after_c + n_prod;
                        }
                    }
                    let mut n_go: u32 = 0u32;
                    if n_live != 0u32 {
                        if n_prod_ok != 0u32 {
                            if m_after_n <= hi {
                                n_go = 1u32;
                            }
                        }
                    }
                    if n_go != 0u32 {
                        let mut o: u32 = 0;
                        let mut o_done: u32 = 0u32;
                        let mut o_more: u32 = 1u32;
                        while o_more != 0u32 {
                            // Same guard shape for `o`.
                            let o_allow_ok = k_allow_ok(hi, m_after_n, m_o);
                            let o_allow = k_allow_n(hi, m_after_n, m_o);
                            let mut o_over: u32 = 0u32;
                            if o_allow_ok == 0u32 {
                                o_over = 1u32;
                            }
                            if o > o_allow {
                                o_over = 1u32;
                            }
                            if k_over_total(rare_sum, c, n, o, heavy_max) != 0u32 {
                                o_over = 1u32;
                            }
                            let mut o_live: u32 = 0u32;
                            if c_go != 0u32 {
                                if n_go != 0u32 {
                                    if o_allow_ok != 0u32 {
                                        if o_over == 0u32 {
                                            o_live = 1u32;
                                        }
                                    }
                                }
                            }
                            if o_over != 0u32 {
                                o_done = 1u32;
                            }
                            if o_allow_ok == 0u32 {
                                o_done = 1u32;
                            }
                            if o_done != 0u32 {
                                o_live = 0u32;
                            }
                            let mut o_prod: u32 = 0u32;
                            let mut o_prod_ok: u32 = 0u32;
                            if o_live != 0u32 {
                                if m_o != 0u32 {
                                    if o <= max_u32 / m_o {
                                        o_prod = o * m_o;
                                        o_prod_ok = 1u32;
                                    }
                                }
                            }
                            let mut m: u32 = max_u32;
                            if o_prod_ok != 0u32 {
                                if o_prod <= max_u32 - m_after_n {
                                    m = m_after_n + o_prod;
                                }
                            }
                            let mut o_go: u32 = 0u32;
                            if o_live != 0u32 {
                                if o_prod_ok != 0u32 {
                                    if m <= hi {
                                        o_go = 1u32;
                                    }
                                }
                            }
                            // Unconditionally safe closed-form hydrogen ends
                            // (`hi >= m` when admitted, so the saturating
                            // difference is exact); the flags below decide
                            // whether the `h` loop runs.
                            let m_ok = k_allow_ok(hi, m, m_h);
                            let h_allow = k_allow_n(hi, m, m_h);
                            let mut h_hi: u32 = h_max;
                            if h_allow < h_max {
                                h_hi = h_allow;
                            }
                            let mut over: u32 = 0u32;
                            if lo > m {
                                over = lo - m;
                            }
                            let mut h_need: u32 = 0u32;
                            if m_h != 0u32 {
                                h_need = over / m_h;
                                if over % m_h != 0u32 {
                                    h_need += 1u32;
                                }
                            }
                            let mut h_lo: u32 = h_need;
                            if h_min > h_need {
                                h_lo = h_min;
                            }
                            let mut ordered: u32 = 0u32;
                            if o_go != 0u32 {
                                if m_ok != 0u32 {
                                    if h_lo <= h_hi {
                                        ordered = 1u32;
                                    }
                                }
                            }
                            let t_ok = k_total_ok(rare_sum, c, n, o, max_u32);
                            let total_v = k_total_v(rare_sum, c, n, o, max_u32);
                            // One visit: this `(c, n, o)` heavy vector with a
                            // non-empty hydrogen range. Stop BEFORE the visit
                            // that would exceed the budget.
                            let mut visit_this: u32 = 0u32;
                            if o_go != 0u32 {
                                if t_ok != 0u32 {
                                    if ordered != 0u32 {
                                        if total_v > 0u32 {
                                            visit_this = 1u32;
                                        }
                                    }
                                }
                            }
                            if visit_this != 0u32 {
                                if visited >= budget_eff {
                                    exhausted = 1u32;
                                    stop = 1u32;
                                }
                            }
                            let mut counted: u32 = 0u32;
                            if visit_this != 0u32 {
                                if visited < budget_eff {
                                    if stop == 0u32 {
                                        // `visited < budget_eff <= 2^31 - 1`,
                                        // so no wrap.
                                        visited += 1u32;
                                        counted = 1u32;
                                    }
                                }
                            }
                            if counted != 0u32 {
                                // Stage (i) is hydrogen-independent:
                                // hoisted out of the `h` loop (same
                                // value for every hydrogen count of
                                // this visit).
                                let buckets_ok = k_buckets(
                                    c, n, o, f, p, s, cl, br, ii, rare_sum, bounds,
                                    bounds_len, n_carbon, n_heavy, cb_width, hb_width,
                                    caps_c, seen_c, caps_h, seen_h, max_u32,
                                );
                                let mut h: u32 = h_lo;
                                let mut h_done: u32 = 0u32;
                                let mut h_more: u32 = 1u32;
                                while h_more != 0u32 {
                                    // Guard FIRST for `h` as well:
                                    // `h <= (hi - m) / m_h` (then the product
                                    // and the sum cannot wrap).
                                    let mut h_admit: u32 = 0u32;
                                    if h <= h_allow {
                                        h_admit = 1u32;
                                    }
                                    if h_admit == 0u32 {
                                        h_done = 1u32;
                                    }
                                    let mut h_prod: u32 = 0u32;
                                    let mut h_prod_ok: u32 = 0u32;
                                    if h_admit != 0u32 {
                                        if m_h != 0u32 {
                                            if h <= max_u32 / m_h {
                                                h_prod = h * m_h;
                                                h_prod_ok = 1u32;
                                            }
                                        }
                                    }
                                    let mut mass: u32 = max_u32;
                                    if h_prod_ok != 0u32 {
                                        if h_prod <= max_u32 - m {
                                            mass = m + h_prod;
                                        }
                                    }
                                    let mut mass_ok: u32 = 0u32;
                                    if h_admit != 0u32 {
                                        if h_prod_ok != 0u32 {
                                            if mass <= hi {
                                                mass_ok = 1u32;
                                            }
                                        }
                                    }
                                    let vflag = k_verdict(
                                        parent, tol, bound, mass, c, h, n, o, f, p, s,
                                        cl, br, ii, res_c, res_h, res_n, res_o, res_f,
                                        res_p, res_s, res_cl, res_br, res_ii, max_u32,
                                    );
                                    let mut v_join: u32 = 0u32;
                                    if vflag != 0u32 {
                                        v_join = 1u32;
                                    }
                                    let mut v_amb: u32 = 0u32;
                                    if vflag == 2u32 {
                                        v_amb = 1u32;
                                    }
                                    let exact_ok = k_exact_ok(
                                        c, h, n, f, p, s, cl, br, ii, max_u32,
                                    );
                                    let pos = k_exact_pos(
                                        c, h, n, f, p, s, cl, br, ii, max_u32,
                                    );
                                    let neg = k_exact_neg(
                                        c, h, n, f, p, s, cl, br, ii, max_u32,
                                    );
                                    let mut ok_all: u32 = mass_ok;
                                    if v_join == 0u32 {
                                        ok_all = 0u32;
                                    }
                                    if exact_ok == 0u32 {
                                        ok_all = 0u32;
                                    }
                                    if buckets_ok == 0u32 {
                                        ok_all = 0u32;
                                    }
                                    // Stage (iii) minima (the maxima hold by
                                    // the lane gate).
                                    if rare_sum
                                        < k_read(bounds, PACK_RARE_TOTAL_LO, bounds_len)
                                    {
                                        ok_all = 0u32;
                                    }
                                    if distinct
                                        < k_read(bounds, PACK_RARE_DISTINCT_LO, bounds_len)
                                    {
                                        ok_all = 0u32;
                                    }
                                    if k_ratios(
                                        h, n, o, c, f, p, s, cl, br, ii, bounds,
                                        bounds_len, max_u32,
                                    ) == 0u32
                                    {
                                        ok_all = 0u32;
                                    }
                                    if k_dbe(
                                        pos, neg, rare_sum, c, n, o, bounds, bounds_len,
                                        n_heavy, hb_width, seen_h, dbe_base, max_u32,
                                    ) == 0u32
                                    {
                                        ok_all = 0u32;
                                    }
                                    let mut gval: u32 = 0u32;
                                    let mut grank_ok: u32 = 0u32;
                                    if joined <= max_u32 - offset {
                                        gval = offset + joined;
                                        grank_ok = 1u32;
                                    }
                                    let mut wbase: u32 = 0u32;
                                    let mut wend_rel_ok: u32 = 0u32;
                                    if gval <= max_u32 / LANE_RECORD_WORDS {
                                        wbase = gval * LANE_RECORD_WORDS;
                                        if wbase <= max_u32 - LANE_RECORD_WORDS {
                                            wend_rel_ok = 1u32;
                                        }
                                    }
                                    let mut write_it: u32 = 0u32;
                                    if mode == LANE_MODE_FILL {
                                        write_it = 1u32;
                                    }
                                    if ok_all == 0u32 {
                                        write_it = 0u32;
                                    }
                                    if grank_ok == 0u32 {
                                        write_it = 0u32;
                                    }
                                    if gval >= eff_cap {
                                        write_it = 0u32;
                                    }
                                    if wend_rel_ok == 0u32 {
                                        write_it = 0u32;
                                    }
                                    if write_it != 0u32 {
                                        if out_base > max_u32 - (wbase + LANE_RECORD_WORDS) {
                                            write_it = 0u32;
                                        }
                                    }
                                    if write_it != 0u32 {
                                        if out_base + wbase + LANE_RECORD_WORDS > buf_len {
                                            write_it = 0u32;
                                        }
                                    }
                                    if write_it != 0u32 {
                                        let waddr = out_base + wbase;
                                        buf[waddr as usize] = c;
                                        buf[(waddr + 1u32) as usize] = h;
                                        buf[(waddr + 2u32) as usize] = n;
                                        buf[(waddr + 3u32) as usize] = o;
                                        buf[(waddr + 4u32) as usize] = f;
                                        buf[(waddr + 5u32) as usize] = p;
                                        buf[(waddr + 6u32) as usize] = s;
                                        buf[(waddr + 7u32) as usize] = cl;
                                        buf[(waddr + 8u32) as usize] = br;
                                        buf[(waddr + 9u32) as usize] = ii;
                                        buf[(waddr + 10u32) as usize] = mass;
                                        let mut flag: u32 = 1u32;
                                        if v_amb != 0u32 {
                                            flag = 2u32;
                                        }
                                        buf[(waddr + 11u32) as usize] = flag;
                                        buf[(waddr + 12u32) as usize] = max_u32;
                                    }
                                    if ok_all != 0u32 {
                                        // `joined` counts lane visits' joins
                                        // in both modes, so count and fill
                                        // agree.
                                        joined += 1u32;
                                        // Fill capacity stop (finding 4): the
                                        // next joined rank is `offset +
                                        // joined`; once it reaches the
                                        // effective cap no later join can be
                                        // written, so the lane stops. Count
                                        // mode is unchanged. The addition is
                                        // guarded: on overflow the lane stops
                                        // safely.
                                        if mode == LANE_MODE_FILL {
                                            if joined <= max_u32 - offset {
                                                if offset + joined >= eff_cap {
                                                    stop = 1u32;
                                                }
                                            }
                                            if joined > max_u32 - offset {
                                                stop = 1u32;
                                            }
                                        }
                                    }
                                    if h_done != 0u32 {
                                        h_more = 0u32;
                                    }
                                    if stop != 0u32 {
                                        h_more = 0u32;
                                    }
                                    if h == h_hi {
                                        h_more = 0u32;
                                    }
                                    if h_more != 0u32 {
                                        // `h < h_hi <= u32::MAX`, so no wrap.
                                        h += 1u32;
                                    }
                                }
                            }
                            if o_done != 0u32 {
                                o_more = 0u32;
                            }
                            if stop != 0u32 {
                                o_more = 0u32;
                            }
                            if o == cap_o {
                                o_more = 0u32;
                            }
                            if o_more != 0u32 {
                                // `o < cap_o <= u32::MAX`, so no wrap.
                                o += 1u32;
                            }
                        }
                    }
                    if n_done != 0u32 {
                        n_more = 0u32;
                    }
                    if stop != 0u32 {
                        n_more = 0u32;
                    }
                    if n == cap_n {
                        n_more = 0u32;
                    }
                    if n_more != 0u32 {
                        // `n < cap_n <= u32::MAX`, so no wrap.
                        n += 1u32;
                    }
                }
            }
            if c_done != 0u32 {
                c_more = 0u32;
            }
            if stop != 0u32 {
                c_more = 0u32;
            }
            if c == cap_c {
                c_more = 0u32;
            }
            if c_more != 0u32 {
                // `c < cap_c <= u32::MAX`, so no wrap.
                c += 1u32;
            }
        }
    }
    let mut visited_word: u32 = visited;
    if exhausted != 0u32 {
        visited_word = visited + LANE_EXHAUSTED_BIT;
    }
    if mode == LANE_MODE_COUNT {
        k_write(buf, stats_base, joined, buf_len);
        k_write(buf, stats_base + 1u32, visited_word, buf_len);
    }
    visited
}

#[allow(clippy::too_many_arguments)]
#[cube]
fn kernel_offsets(
    lane_stats: &Array<u32>,
    stats_len: u32,
    meta: &Array<u32>,
    meta_len: u32,
    offsets: &mut Array<u32>,
    offsets_len: u32,
    counters: &mut Array<u32>,
    counters_len: u32,
    b: u32,
    n_p: u32,
    cap_arg: u32,
    bit_exhausted: u32,
    bit_absent: u32,
    max_u32: u32,
) {
    let mbase = b * META_LEN;
    let mcap = k_read(meta, mbase + META_SCORED_CAP, meta_len);
    let mut cap: u32 = mcap;
    if cap_arg < cap {
        cap = cap_arg;
    }
    let sbase = b * n_p * 2u32;
    let obase = b * n_p;
    let cbase = b * 5u32;
    let mut running_j: u32 = 0;
    let mut running_v: u32 = 0;
    let mut any_exh: u32 = 0u32;
    let mut sat: u32 = 0u32;
    let mut r: u32 = 0;
    let mut r_more: u32 = 1u32;
    if n_p == 0u32 {
        r_more = 0u32;
    }
    while r_more != 0u32 {
        // `r < n_p`, so `sbase + 2 * r + 1` addresses this spectrum's words.
        let jr = k_read(lane_stats, sbase + r * 2u32, stats_len);
        let vw = k_read(lane_stats, sbase + r * 2u32 + 1u32, stats_len);
        let vr = vw & LANE_VISITED_MASK;
        let mut exh: u32 = 0u32;
        if vw & LANE_EXHAUSTED_BIT != 0u32 {
            exh = 1u32;
        }
        let mut off: u32 = cap;
        if running_j < cap {
            off = running_j;
        }
        k_write(offsets, obase + r, off, offsets_len);
        if sat_add_saturates(running_j, jr, max_u32) != 0u32 {
            sat = 1u32;
        }
        running_j = sat_add_counter(running_j, jr, max_u32);
        if sat_add_saturates(running_v, vr, max_u32) != 0u32 {
            sat = 1u32;
        }
        running_v = sat_add_counter(running_v, vr, max_u32);
        if exh != 0u32 {
            any_exh = 1u32;
        }
        if r + 1u32 >= n_p {
            r_more = 0u32;
        }
        if r_more != 0u32 {
            // `r < n_p - 1 <= u32::MAX - 1`, so no wrap.
            r += 1u32;
        }
    }
    let mut scored: u32 = cap;
    if running_j < cap {
        scored = running_j;
    }
    let mut trunc: u32 = 0u32;
    if running_j > scored {
        trunc = 1u32;
    }
    let mut exhausted: u32 = 0u32;
    if any_exh != 0u32 {
        exhausted = 1u32;
    }
    if trunc != 0u32 {
        exhausted = 1u32;
    }
    if sat != 0u32 {
        exhausted = 1u32;
    }
    let mut status: u32 = 0u32;
    if exhausted != 0u32 {
        status = bit_exhausted;
    }
    if exhausted == 0u32 {
        if running_j == 0u32 {
            status = bit_absent;
        }
    }
    let mut complete: u32 = 0u32;
    if exhausted == 0u32 {
        complete = 1u32;
    }
    k_write(counters, cbase, running_v, counters_len);
    k_write(counters, cbase + 1u32, running_j, counters_len);
    k_write(counters, cbase + 2u32, scored, counters_len);
    k_write(counters, cbase + 3u32, status, counters_len);
    k_write(counters, cbase + 4u32, complete, counters_len);
}

#[cube]
fn kernel_pad(
    out: &mut Array<u32>,
    out_len: u32,
    base: u32,
    slot: u32,
    scored: u32,
    cap: u32,
    max_u32: u32,
) {
    let mut pad: u32 = 0u32;
    if slot >= scored {
        if slot < cap {
            pad = 1u32;
        }
    }
    if k_span_ok(base, slot, LANE_RECORD_WORDS, max_u32) == 0u32 {
        pad = 0u32;
    }
    let wbase = k_span_at(base, slot, LANE_RECORD_WORDS, max_u32);
    if pad != 0u32 {
        // `wbase + 12` is in bounds: the span guard keeps
        // `base + slot * 13 <= max_u32`, and each write is length-guarded.
        k_write(out, wbase, 0u32, out_len);
        k_write(out, wbase + 1u32, 0u32, out_len);
        k_write(out, wbase + 2u32, 0u32, out_len);
        k_write(out, wbase + 3u32, 0u32, out_len);
        k_write(out, wbase + 4u32, 0u32, out_len);
        k_write(out, wbase + 5u32, 0u32, out_len);
        k_write(out, wbase + 6u32, 0u32, out_len);
        k_write(out, wbase + 7u32, 0u32, out_len);
        k_write(out, wbase + 8u32, 0u32, out_len);
        k_write(out, wbase + 9u32, 0u32, out_len);
        k_write(out, wbase + 10u32, 0u32, out_len);
        k_write(out, wbase + 11u32, 0u32, out_len);
        k_write(out, wbase + 12u32, max_u32, out_len);
    }
}







// ---------------------------------------------------------------------------
// Launches (one lane per output item; span loops as in `ops::ms2`).
// ---------------------------------------------------------------------------

/// Lane per `(b, r)`: run the shared lane in count mode, writing
/// `(joined, visited_with_exhausted_bit)` into `stats`. The record output is
/// inert in count mode. Arrays: `meta`, `rare`, `bounds`, `stats`.
#[allow(clippy::too_many_arguments)]
#[cube(launch_unchecked)]
fn ms2_enum_count_kernel(
    meta: &Array<u32>,
    meta_len: u32,
    rare: &Array<u32>,
    rare_len: u32,
    bounds: &Array<u32>,
    bounds_len: u32,
    stats: &mut Array<u32>,
    stats_len: u32,
    n_p: usize,
    m_c: u32,
    m_n: u32,
    m_o: u32,
    m_h: u32,
    res_c: u32,
    res_h: u32,
    res_n: u32,
    res_o: u32,
    res_f: u32,
    res_p: u32,
    res_s: u32,
    res_cl: u32,
    res_br: u32,
    res_ii: u32,
    max_u32: u32,
    lane_budget: u32,
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
        // D10 bounded dispatch: this launch covers the contiguous lane range
        // `[first_lane, first_lane + lanes)`; the absolute lane index keeps
        // results identical to one unchunked launch.
        let pos = first_lane + local;
        let b = (pos / n_p) as u32;
        let r = (pos % n_p) as u32;
        let mbase = b * META_LEN;
        let rbase = r * 8u32;
        let sbase = (b * (n_p as u32) + r) * 2u32;
        kernel_lane(
            meta, meta_len, bounds, bounds_len, rare, rare_len, mbase, rbase,
            LANE_MODE_COUNT, 0u32, 0u32, stats, stats_len, 0u32, sbase,
            m_c, m_n, m_o, m_h, res_c, res_h, res_n, res_o, res_f, res_p, res_s, res_cl,
            res_br, res_ii, max_u32, lane_budget,
        );
    }
}

/// Lane per spectrum `b`: aggregate its `n_p` count results into its clamped
/// offsets and its 5 counters. Arrays: `lane_stats`, `meta`, `offsets`,
/// `counters`.
#[allow(clippy::too_many_arguments)]
#[cube(launch_unchecked)]
fn ms2_enum_offsets_kernel(
    lane_stats: &Array<u32>,
    stats_len: u32,
    meta: &Array<u32>,
    meta_len: u32,
    offsets: &mut Array<u32>,
    offsets_len: u32,
    counters: &mut Array<u32>,
    counters_len: u32,
    n_p: u32,
    cap_arg: u32,
    bit_exhausted: u32,
    bit_absent: u32,
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
        let b = pos as u32;
        kernel_offsets(
            lane_stats, stats_len, meta, meta_len, offsets, offsets_len, counters,
            counters_len, b, n_p, cap_arg, bit_exhausted, bit_absent, max_u32,
        );
    }
}

/// Lane per `(b, r)`: run the shared lane in fill mode at the lane's clamped
/// first rank, writing 13-word records for ranks below `cap` into `cand`.
/// The stats words are untouched in fill mode. Arrays: `meta`, `rare`,
/// `bounds`, `offsets`, `cand`.
#[allow(clippy::too_many_arguments)]
#[cube(launch_unchecked)]
fn ms2_enum_fill_kernel(
    meta: &Array<u32>,
    meta_len: u32,
    rare: &Array<u32>,
    rare_len: u32,
    bounds: &Array<u32>,
    bounds_len: u32,
    offsets: &Array<u32>,
    cand: &mut Array<u32>,
    cand_len: u32,
    n_p: usize,
    m_slots: usize,
    cap: u32,
    m_c: u32,
    m_n: u32,
    m_o: u32,
    m_h: u32,
    res_c: u32,
    res_h: u32,
    res_n: u32,
    res_o: u32,
    res_f: u32,
    res_p: u32,
    res_s: u32,
    res_cl: u32,
    res_br: u32,
    res_ii: u32,
    max_u32: u32,
    lane_budget: u32,
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
        // D10 bounded dispatch: contiguous lane range as in the count kernel.
        let pos = first_lane + local;
        let b = (pos / n_p) as u32;
        let r = (pos % n_p) as u32;
        let mbase = b * META_LEN;
        let rbase = r * 8u32;
        let offset = offsets[(b * (n_p as u32) + r) as usize];
        let out_base = (b * (m_slots as u32)) * LANE_RECORD_WORDS;
        kernel_lane(
            meta, meta_len, bounds, bounds_len, rare, rare_len, mbase, rbase,
            LANE_MODE_FILL, offset, cap, cand, cand_len, out_base, 0u32,
            m_c, m_n, m_o, m_h, res_c, res_h, res_n, res_o, res_f, res_p, res_s, res_cl,
            res_br, res_ii, max_u32, lane_budget,
        );
    }
}

/// Lane per `(b, m)`: pad the slot when `rows_scored <= m < M`, where
/// `rows_scored` is read from `counters` and `M` is the candidate width.
/// Slots below `rows_scored` are owned by fill and are never written here.
/// Arrays: `counters`, `cand`.
#[cube(launch_unchecked)]
fn ms2_cand_pad_kernel(
    counters: &Array<u32>,
    cand: &mut Array<u32>,
    cand_len: u32,
    m_slots: usize,
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
        let b = (pos / m_slots) as u32;
        let m = (pos % m_slots) as u32;
        let scored = counters[(b * 5u32 + 2u32) as usize];
        let base = (b * (m_slots as u32)) * LANE_RECORD_WORDS;
        kernel_pad(
            cand,
            cand_len,
            base,
            m,
            scored,
            m_slots as u32,
            max_u32,
        );
    }
}

// ---------------------------------------------------------------------------
// Safe wrappers (ranks, shapes, geometry; `Error::Shape`, never a panic).
// ---------------------------------------------------------------------------

/// Run [`ms2_enum_count_kernel`]: `meta [B, 8]`, `rare [P, 8]`, packed
/// `bounds` → `lane_stats [B * P, 2]`. Bounded dispatch (spec §1.4 D10):
/// `ceil(B * P / lanes_per_dispatch)` launches over contiguous lane ranges
/// (`lanes_per_dispatch = max(1, dispatch_visits_max / lane_visits_max)`),
/// each taking the absolute lane index so results equal one unchunked launch.
/// No launch when `B * P` is 0. `lanes_max` is the spec §1.4 `B * P` ceiling
/// (`enum_lanes_max`).
///
/// After each dispatch launch the queued work is submitted with
/// [`crate::backend::check_launches`], so each chunk is its own GPU job:
/// the chunking and per-chunk submission bound the work of one GPU job to
/// about `dispatch_visits_max` visits (a precautionary work bound). The number of
/// flushes equals the number of enumeration launches. A flush is not a read
/// and launches nothing: it moves none of the crate's counters
/// (`launch_count`, `read_count`, `synchronize_count`).
pub fn enum_count<R: Runtime>(
    meta: &IdTensor<R>,
    rare: &IdTensor<R>,
    bounds: &IdTensor<R>,
    lane_stats: &IdTensor<R>,
    chem: &EnumChem,
    lanes_max: u32,
    dispatch_visits_max: u32,
    lane_visits_max: u32,
) -> Result<()> {
    if meta.shape().rank() != 2
        || rare.shape().rank() != 2
        || bounds.shape().rank() != 1
        || lane_stats.shape().rank() != 2
    {
        return Err(Error::shape(format!(
            "enum_count needs meta [B, 8], rare [P, 8], bounds [Nb] and lane_stats [B * P, 2], got {} and {} and {} and {}",
            meta.shape(),
            rare.shape(),
            bounds.shape(),
            lane_stats.shape()
        )));
    }
    let batch = meta.shape().dim(0);
    let p = rare.shape().dim(0);
    let lanes = batch.checked_mul(p).ok_or_else(|| {
        Error::shape(format!("enum_count: batch {batch} times {p} lanes overflows usize"))
    })?;
    if meta.shape().dims() != [batch, 8]
        || rare.shape().dims() != [p, 8]
        || lane_stats.shape().dims() != [lanes, 2]
    {
        return Err(Error::shape(format!(
            "enum_count needs meta [{batch}, 8], rare [{p}, 8] and lane_stats [{lanes}, 2], got {} and {} and {}",
            meta.shape(),
            rare.shape(),
            lane_stats.shape()
        )));
    }
    crate::models::ms2::formula_enum::validate_enum_dispatch(batch, p, 0, lanes_max)
        .map_err(|e| match e {
            Error::Shape(msg) => Error::shape(format!("enum_count: {msg}")),
            other => other,
        })?;
    if lanes == 0 {
        return Ok(());
    }
    let meta_len = u32_len(meta.len(), "meta")?;
    let rare_len = u32_len(rare.len(), "rare")?;
    let bounds_len = u32_len(bounds.len(), "bounds")?;
    let stats_len = u32_len(lane_stats.len(), "lane_stats")?;
    let client = meta.client();
    // D10 bounded dispatch: split the `B * P` lanes into contiguous chunks so
    // one launch covers at most about `dispatch_visits_max` visits.
    use crate::models::ms2::formula_enum::{enum_dispatches, enum_lanes_per_dispatch};
    let per = enum_lanes_per_dispatch(dispatch_visits_max, lane_visits_max);
    let n_launches = enum_dispatches(lanes, per);
    for i in 0..n_launches {
        let first = i * per;
        let chunk = (lanes - first).min(per);
        let (count, dim, span) = launch_1d_spans(client, chunk, 1024);
        unsafe {
            ms2_enum_count_kernel::launch_unchecked::<R>(
                client,
                count,
                dim,
                meta.arg(),
                meta_len,
                rare.arg(),
                rare_len,
                bounds.arg(),
                bounds_len,
                lane_stats.arg(),
                stats_len,
                p,
                chem.m_c,
                chem.m_n,
                chem.m_o,
                chem.m_h,
                chem.res[0],
                chem.res[1],
                chem.res[2],
                chem.res[3],
                chem.res[4],
                chem.res[5],
                chem.res[6],
                chem.res[7],
                chem.res[8],
                chem.res[9],
                u32::MAX,
                lane_visits_max,
                first,
                chunk,
                span,
            );
        }
        // Submit this chunk as its own GPU job (one flush per launch; not
        // a read, moves no counters), bounding the work of one GPU job as
        // documented above (a precautionary work bound).
        crate::backend::check_launches(meta.device())?;
    }
    Ok(())
}

/// Run [`ms2_enum_offsets_kernel`]: `lane_stats [B * P, 2]`, `meta [B, 8]`
/// → `offsets [B * P]` and `counters [B, 5]`. The effective cap is
/// `min(meta word 6, scored_cap, M)` with `M = m_slots` (the wrapper clamps
/// its arg to `M`, so a skewed meta word cannot address past the candidate
/// buffer and offsets/fill/pad agree). One launch per call; no launch when
/// `B` is 0. `lanes_max` is the spec §1.4 `B * P` ceiling.
pub fn enum_offsets<R: Runtime>(
    lane_stats: &IdTensor<R>,
    meta: &IdTensor<R>,
    offsets: &IdTensor<R>,
    counters: &IdTensor<R>,
    scored_cap: u32,
    m_slots: usize,
    lanes_max: u32,
) -> Result<()> {
    if lane_stats.shape().rank() != 2
        || meta.shape().rank() != 2
        || offsets.shape().rank() != 1
        || counters.shape().rank() != 2
    {
        return Err(Error::shape(format!(
            "enum_offsets needs lane_stats [B * P, 2], meta [B, 8], offsets [B * P] and counters [B, 5], got {} and {} and {} and {}",
            lane_stats.shape(),
            meta.shape(),
            offsets.shape(),
            counters.shape()
        )));
    }
    let batch = meta.shape().dim(0);
    if batch == 0 {
        if !lane_stats.is_empty() || !offsets.is_empty() {
            return Err(Error::shape(format!(
                "enum_offsets needs empty lane_stats and offsets for an empty batch, got {} and {}",
                lane_stats.shape(),
                offsets.shape()
            )));
        }
        if counters.shape().dims() != [0, 5] {
            return Err(Error::shape(format!(
                "enum_offsets needs counters [0, 5] for an empty batch, got {}",
                counters.shape()
            )));
        }
        return Ok(());
    }
    let stat_rows = lane_stats.shape().dim(0);
    if lane_stats.shape().dim(1) != 2 || !stat_rows.is_multiple_of(batch) {
        return Err(Error::shape(format!(
            "enum_offsets needs lane_stats [B * P, 2] with B = {batch}, got {}",
            lane_stats.shape()
        )));
    }
    let p = stat_rows / batch;
    let lanes = batch;
    if meta.shape().dims() != [batch, 8]
        || offsets.len() != lanes.checked_mul(p).ok_or_else(|| {
            Error::shape(format!("enum_offsets: batch {batch} times {p} lanes overflows usize"))
        })?
        || counters.shape().dims() != [batch, 5]
    {
        return Err(Error::shape(format!(
            "enum_offsets needs meta [{batch}, 8], offsets [{}] and counters [{batch}, 5], got {} and {} and {}",
            lanes * p,
            meta.shape(),
            offsets.shape(),
            counters.shape()
        )));
    }
    crate::models::ms2::formula_enum::validate_enum_dispatch(batch, p, m_slots, lanes_max)
        .map_err(|e| match e {
            Error::Shape(msg) => Error::shape(format!("enum_offsets: {msg}")),
            other => other,
        })?;
    let stats_len = u32_len(lane_stats.len(), "lane_stats")?;
    let meta_len = u32_len(meta.len(), "meta")?;
    let offsets_len = u32_len(offsets.len(), "offsets")?;
    let counters_len = u32_len(counters.len(), "counters")?;
    let n_p = u32_len(p, "lanes per spectrum")?;
    let m_u32 = u32_len(m_slots, "candidate slots")?;
    // Identical effective cap as fill/pad: `min(meta, scored_cap, M)`.
    let cap_clamped = if scored_cap < m_u32 { scored_cap } else { m_u32 };
    let client = meta.client();
    let (count, dim, span) = launch_1d_spans(client, lanes, 64);
    unsafe {
        ms2_enum_offsets_kernel::launch_unchecked::<R>(
            client,
            count,
            dim,
            lane_stats.arg(),
            stats_len,
            meta.arg(),
            meta_len,
            offsets.arg(),
            offsets_len,
            counters.arg(),
            counters_len,
            n_p,
            cap_clamped,
            request_status::FORMULA_SEARCH_EXHAUSTED,
            request_status::FORMULA_ABSENT,
            u32::MAX,
            lanes,
            span,
        );
    }
    Ok(())
}

/// Run [`ms2_enum_fill_kernel`]: `meta [B, 8]`, `rare [P, 8]`, packed
/// `bounds`, `offsets [B * P]` → `cand [B, M, 13]`. The effective cap is
/// `min(meta word 6, scored_cap, M)` (the lane reads the per-spectrum meta
/// word; the wrapper clamps its arg to `M`); only ranks below the spectrum's
/// `rows_scored` are written, so fill owns ranks below `scored` and pad owns
/// the rest. Fill never touches `lane_stats`, so `cand` keeps its single
/// writer per slot. Bounded dispatch as in [`enum_count`]; no launch when
/// `B * P` is 0. `lanes_max` is the spec §1.4 `B * P` ceiling.
///
/// After each dispatch launch the queued work is submitted with
/// [`crate::backend::check_launches`], one flush per launch, as documented
/// on [`enum_count`]: each chunk is its own GPU job, bounding the work of
/// one GPU job (a precautionary work bound). A flush is not a read and launches nothing: it
/// moves none of the crate's counters.
#[allow(clippy::too_many_arguments)]
pub fn enum_fill<R: Runtime>(
    meta: &IdTensor<R>,
    rare: &IdTensor<R>,
    bounds: &IdTensor<R>,
    offsets: &IdTensor<R>,
    cand: &IdTensor<R>,
    chem: &EnumChem,
    scored_cap: u32,
    lanes_max: u32,
    dispatch_visits_max: u32,
    lane_visits_max: u32,
) -> Result<()> {
    if meta.shape().rank() != 2
        || rare.shape().rank() != 2
        || bounds.shape().rank() != 1
        || offsets.shape().rank() != 1
        || cand.shape().rank() != 3
    {
        return Err(Error::shape(format!(
            "enum_fill needs meta [B, 8], rare [P, 8], bounds [Nb], offsets [B * P] and cand [B, M, 13], got {} and {} and {} and {} and {}",
            meta.shape(),
            rare.shape(),
            bounds.shape(),
            offsets.shape(),
            cand.shape()
        )));
    }
    let batch = meta.shape().dim(0);
    let p = rare.shape().dim(0);
    let lanes = batch.checked_mul(p).ok_or_else(|| {
        Error::shape(format!("enum_fill: batch {batch} times {p} lanes overflows usize"))
    })?;
    let m = cand.shape().dim(1);
    if meta.shape().dims() != [batch, 8]
        || rare.shape().dims() != [p, 8]
        || offsets.len() != lanes
        || cand.shape().dims() != [batch, m, 13]
    {
        return Err(Error::shape(format!(
            "enum_fill needs meta [{batch}, 8], rare [{p}, 8], offsets [{lanes}] and cand [{batch}, {m}, 13], got {} and {} and {} and {} and {}",
            meta.shape(),
            rare.shape(),
            bounds.shape(),
            offsets.shape(),
            cand.shape()
        )));
    }
    if lanes == 0 {
        return Ok(());
    }
    crate::models::ms2::formula_enum::validate_enum_dispatch(batch, p, m, lanes_max)
        .map_err(|e| match e {
            Error::Shape(msg) => Error::shape(format!("enum_fill: {msg}")),
            other => other,
        })?;
    let m_u32 = u32_len(m, "candidate slots")?;
    let cap = if scored_cap < m_u32 { scored_cap } else { m_u32 };
    let meta_len = u32_len(meta.len(), "meta")?;
    let rare_len = u32_len(rare.len(), "rare")?;
    let bounds_len = u32_len(bounds.len(), "bounds")?;
    // The fill kernel indexes `offsets` directly (no length scalar travels),
    // so its length is validated as `u32` here even on standalone calls.
    u32_len(offsets.len(), "offsets")?;
    let cand_len = u32_len(cand.len(), "cand")?;
    let client = meta.client();
    use crate::models::ms2::formula_enum::{enum_dispatches, enum_lanes_per_dispatch};
    let per = enum_lanes_per_dispatch(dispatch_visits_max, lane_visits_max);
    let n_launches = enum_dispatches(lanes, per);
    for i in 0..n_launches {
        let first = i * per;
        let chunk = (lanes - first).min(per);
        let (count, dim, span) = launch_1d_spans(client, chunk, 1024);
        unsafe {
            ms2_enum_fill_kernel::launch_unchecked::<R>(
                client,
                count,
                dim,
                meta.arg(),
                meta_len,
                rare.arg(),
                rare_len,
                bounds.arg(),
                bounds_len,
                offsets.arg(),
                cand.arg(),
                cand_len,
                p,
                m,
                cap,
            chem.m_c,
            chem.m_n,
            chem.m_o,
            chem.m_h,
            chem.res[0],
            chem.res[1],
            chem.res[2],
            chem.res[3],
            chem.res[4],
            chem.res[5],
            chem.res[6],
            chem.res[7],
            chem.res[8],
            chem.res[9],
            u32::MAX,
            lane_visits_max,
            first,
            chunk,
            span,
            );
        }
        // Submit this chunk as its own GPU job (one flush per launch; not
        // a read, moves no counters); see `enum_count`.
        crate::backend::check_launches(meta.device())?;
    }
    Ok(())
}

/// Run [`ms2_cand_pad_kernel`]: `counters [B, 5]` → `cand [B, M, 13]`.
/// Slots `m >= rows_scored` (read from `counters`) become padding records;
/// slots below are never written, so pad owns slots at or after `scored`
/// while fill owns ranks below it. One launch; no launch when `B * M` is 0.
/// `p` is the rare-table depth of the dispatch (for the `B * P` ceiling) and
/// `lanes_max` the spec §1.4 ceiling: pad enforces the same dispatch ceiling
/// as count/offsets/fill even though it binds no rare buffer.
pub fn cand_pad<R: Runtime>(
    counters: &IdTensor<R>,
    cand: &IdTensor<R>,
    p: usize,
    lanes_max: u32,
) -> Result<()> {
    if counters.shape().rank() != 2 || cand.shape().rank() != 3 {
        return Err(Error::shape(format!(
            "cand_pad needs counters [B, 5] and cand [B, M, 13], got {} and {}",
            counters.shape(),
            cand.shape()
        )));
    }
    let batch = counters.shape().dim(0);
    let m = cand.shape().dim(1);
    let lanes = batch.checked_mul(m).ok_or_else(|| {
        Error::shape(format!("cand_pad: batch {batch} times {m} slots overflows usize"))
    })?;
    if counters.shape().dims() != [batch, 5] || cand.shape().dims() != [batch, m, 13] {
        return Err(Error::shape(format!(
            "cand_pad needs counters [{batch}, 5] and cand [{batch}, {m}, 13], got {} and {}",
            counters.shape(),
            cand.shape()
        )));
    }
    crate::models::ms2::formula_enum::validate_enum_dispatch(batch, p, m, lanes_max)
        .map_err(|e| match e {
            Error::Shape(msg) => Error::shape(format!("cand_pad: {msg}")),
            other => other,
        })?;
    if lanes == 0 {
        return Ok(());
    }
    // The pad kernel indexes `counters` directly (no length scalar travels),
    // so its length is validated as `u32` here.
    u32_len(counters.len(), "counters")?;
    let cand_len = u32_len(cand.len(), "cand")?;
    let client = counters.client();
    let (count, dim, span) = launch_1d_spans(client, lanes, 16);
    unsafe {
        ms2_cand_pad_kernel::launch_unchecked::<R>(
            client,
            count,
            dim,
            counters.arg(),
            cand.arg(),
            cand_len,
            m,
            u32::MAX,
            lanes,
            span,
        );
    }
    Ok(())
}
