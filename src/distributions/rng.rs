//! Counter-based random numbers that are the same on every device and on the host.
//!
//! Everything in [`super`] draws from one generator: **Philox-4×32-10**, the
//! counter-based design from Salmon et al.'s *Parallel Random Numbers: As Easy as
//! 1, 2, 3*. It is the generator PyTorch itself uses, and it is chosen here for
//! three properties that a distribution library cannot do without.
//!
//! * **Stateless.** A draw is a pure function of *where* it is needed — the element
//!   index and a stream number — and of the seed. No unit carries a generator, no
//!   two launches share a stream, and nothing has to be written back. That is what
//!   lets a sampler be a one-unit-per-element kernel with no synchronisation at all.
//! * **Integer-exact.** Every operation is a 32-bit multiply, shift or xor. Those
//!   are exact on every backend CubeCL targets, so the *bits* a draw produces do not
//!   depend on the device, the vector width, the cube geometry, or the compiler's
//!   licence to reassociate floating point. [`philox4x32_10()`] and its host twin
//!   [`host::philox4x32_10()`] therefore agree bit for bit, which is the foundation the
//!   bit-exactness tests in `tests/distributions_bitexact.rs` are built on.
//! * **Independently verifiable.** Philox has published known-answer vectors, and
//!   the test suite checks against them. That turns "the device agrees with my host
//!   reference" — which two matching bugs would also satisfy — into "both agree with
//!   the published algorithm".
//!
//! # The 64-bit multiply, without a 64-bit multiply
//!
//! Philox's round function needs the *high* word of a 32×32 product. Writing it as
//! `((a as u64 * b as u64) >> 32) as u32` works on CUDA and on the CPU runtime, but
//! WGSL has no 64-bit integer at all, so that spelling would silently cost the crate
//! its wgpu backend. [`mulhi()`] computes the same word from four 16-bit partial
//! products in pure `u32` arithmetic. `tests/distributions_bitexact.rs` asserts the
//! two spellings agree on every input it tries; this one is used because it is the
//! one that exists everywhere.
//!
//! # Two ways to index the stream
//!
//! A Philox call turns a four-word counter into four random words. How those four
//! words are handed out decides how much work a sampler does per element, so there
//! are two conventions and every sampler declares which it uses:
//!
//! | | counter | what element `i` gets | used by |
//! |---|---|---|---|
//! | [`draw_lane()`] | `(i / 4, i_hi, stream, 0)` | lane `i % 4` | samplers that need one uniform |
//! | [`draw_block()`] | `(i, i_hi, stream, 1)` | all four lanes | samplers that need two to four |
//!
//! [`draw_lane()`] amortises one Philox evaluation over four elements, which is why
//! the inverse-CDF samplers process four elements per unit. [`draw_block()`] gives a
//! rejection sampler four fresh uniforms per iteration for one evaluation. The
//! trailing counter word differs between them so the two conventions never hand the
//! same bits to two different callers.
//!
//! Like [`super::special`], the functions in this module root are **device**
//! functions — callable from a `#[cube]` kernel, and panicking if called from
//! ordinary Rust. The `_host` suffixed ones beside them are the host twins.
//!
//! Both are functions of the element index alone, never of the launch geometry, so
//! a `[1024]` tensor sampled as one cube of 1024 units and as 32 cubes of 32 units
//! produce identical bits — and so does element 7 of a `[8]` tensor and element 7 of
//! a `[8192]` one.

// A `#[cube]` function cannot use `if` as an expression, so a branching value is a
// `let mut` initialised before the branch that sets it; the initialiser is then dead
// by construction and the lint that notices has nothing to offer.
#![allow(unused_assignments)]
// Every `#[cube] pub fn` expands to a public module of the same name holding the
// macro's generated `expand` entry points. There is no way to attach documentation
// to a module a proc macro synthesises, so `missing_docs` fires on each of them and
// there is nothing to say in reply. Every item written by hand in this file is
// documented; the allow is for the ones that are not.
#![allow(missing_docs)]

use cubecl::prelude::*;

/// Philox's first multiplier.
pub(crate) const PHILOX_M0: u32 = 0xD251_1F53;
/// Philox's second multiplier.
pub(crate) const PHILOX_M1: u32 = 0xCD9E_8D57;
/// The key bump for the first word: the golden ratio, `⌊2³²/φ⌋`.
pub(crate) const PHILOX_W0: u32 = 0x9E37_79B9;
/// The key bump for the second word: `⌊2³²(√3 − 1)⌋`.
pub(crate) const PHILOX_W1: u32 = 0xBB67_AE85;

/// `2⁻²³`, the width of one cell of the unit interval.
///
/// Twenty-three bits rather than an `f32` significand's twenty-four, so that the
/// midpoint of the last cell — `(2²³ − ½)·2⁻²³` — is still exactly representable.
/// At twenty-four it is not: `(2²⁴ − ½)·2⁻²⁴` rounds up to exactly `1`, and a
/// sampler that takes `ln(1 − u)` would then produce an infinity roughly once in
/// sixteen million draws. Half a bit of resolution is a cheap price for a closed
/// proof that [`unit_open()`] never reaches an endpoint.
pub const UNIT_STEP: f32 = 1.0 / 8_388_608.0;

/// The counter word that separates [`draw_lane()`]'s stream from [`draw_block()`]'s.
const DOMAIN_LANE: u32 = 0;
/// See [`DOMAIN_LANE`].
const DOMAIN_BLOCK: u32 = 1;

/// The four words one Philox evaluation produces.
#[derive(CubeType, Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bits4 {
    /// First word.
    pub a: u32,
    /// Second word.
    pub b: u32,
    /// Third word.
    pub c: u32,
    /// Fourth word.
    pub d: u32,
}

/// The high word of `a * b`, in pure 32-bit arithmetic.
///
/// Four 16-bit partial products, recombined the way long multiplication does it.
/// Every intermediate fits in a `u32`: `p00`, `p01`, `p10` and `p11` are products of
/// 16-bit values, and `mid` sums one 16-bit carry with two 16-bit halves, so it
/// stays below `3 · 2¹⁶`. The result is the exact high word — not an approximation
/// of it — which is what makes it a drop-in for the 64-bit spelling.
#[cube]
pub fn mulhi_split(a: u32, b: u32) -> u32 {
    let a0 = a & 0xffffu32;
    let a1 = a >> 16u32;
    let b0 = b & 0xffffu32;
    let b1 = b >> 16u32;
    let p00 = a0 * b0;
    let p01 = a0 * b1;
    let p10 = a1 * b0;
    let p11 = a1 * b1;
    let mid = (p00 >> 16u32) + (p01 & 0xffffu32) + (p10 & 0xffffu32);
    p11 + (p01 >> 16u32) + (p10 >> 16u32) + (mid >> 16u32)
}

/// The same word, from a 64-bit multiply.
///
/// One instruction where [`mulhi_split()`] is eleven, and the generator calls it
/// twenty times per draw — measured, it is worth about 1.7× on every sampler here.
/// Not every backend has it: WGSL has no 64-bit integer at all, which is why the
/// portable spelling exists and why the choice is made per device rather than once.
#[cube]
pub fn mulhi_wide(a: u32, b: u32) -> u32 {
    (((a as u64) * (b as u64)) >> 32u64) as u32
}

/// The high word of `a * b`, by whichever route this device has.
///
/// `wide` is `#[comptime]`, so a kernel compiled for a device with 64-bit integers
/// contains only the one-instruction form and one compiled for a device without
/// contains only the portable one. The two produce identical bits — asserted in
/// `distributions_bitexact::the_two_wide_multiplies_agree` — so the flag is a pure
/// performance choice and a draw does not depend on which way it was taken.
#[cube]
pub fn mulhi(a: u32, b: u32, #[comptime] wide: bool) -> u32 {
    let mut out: u32 = 0;
    if comptime!(wide) {
        out = mulhi_wide(a, b);
    } else {
        out = mulhi_split(a, b);
    }
    out
}

/// Whether this device can multiply 64-bit integers, and so which [`mulhi()`] a kernel
/// launched against it should be compiled with.
///
/// Queried rather than assumed: a wgpu backend on Vulkan may well have `Int64` while
/// the same backend on WebGPU proper does not, and the answer belongs to the device
/// and not to the build.
pub fn wide_multiply<R: Runtime>(client: &ComputeClient<R>) -> bool {
    use cubecl::ir::features::TypeUsage;
    use cubecl::ir::{ElemType, StorageType, UIntKind};

    client
        .properties()
        .features
        .types
        .storage
        .get(&StorageType::Scalar(ElemType::UInt(UIntKind::U64)))
        .is_some_and(|usage| usage.contains(TypeUsage::Arithmetic))
}

/// Ten Philox rounds over a four-word counter and a two-word key.
///
/// The rounds are unrolled deliberately. CubeCL's CPU backend miscompiles a
/// conditional assignment to a loop-carried variable inside a rolled loop — the
/// generated MLIR fails its own dominance check — and a ten-iteration loop over
/// straight-line integer code is exactly the shape a compiler should unroll anyway.
#[allow(arithmetic_overflow)]
#[cube]
pub fn philox4x32_10(
    c0: u32,
    c1: u32,
    c2: u32,
    c3: u32,
    key_lo: u32,
    key_hi: u32,
    #[comptime] wide: bool,
) -> Bits4 {
    let m0 = PHILOX_M0;
    let m1 = PHILOX_M1;
    let mut x0 = c0;
    let mut x1 = c1;
    let mut x2 = c2;
    let mut x3 = c3;
    let mut k0 = key_lo;
    let mut k1 = key_hi;
    #[unroll]
    for round in 0..10u32 {
        let hi0 = mulhi(m0, x0, wide);
        let lo0 = m0 * x0;
        let hi1 = mulhi(m1, x2, wide);
        let lo1 = m1 * x2;
        x0 = hi1 ^ x1 ^ k0;
        x1 = lo1;
        x2 = hi0 ^ x3 ^ k1;
        x3 = lo0;
        // The last round leaves the key alone; bumping it would be harmless but the
        // published algorithm does not, and matching it is what makes the
        // known-answer vectors an independent check rather than a tautology.
        if round < 9u32 {
            k0 += PHILOX_W0;
            k1 += PHILOX_W1;
        }
    }
    Bits4 {
        a: x0,
        b: x1,
        c: x2,
        d: x3,
    }
}

/// Four random words for element `index` of stream `stream`.
///
/// Use this when a sampler needs more than one uniform per element — a rejection
/// loop's proposal and its acceptance test, or the two draws a ratio of gammas
/// needs. One evaluation, four uniforms.
#[cube]
pub fn draw_block(
    index: u32,
    index_hi: u32,
    stream: u32,
    key_lo: u32,
    key_hi: u32,
    #[comptime] wide: bool,
) -> Bits4 {
    philox4x32_10(index, index_hi, stream, DOMAIN_BLOCK, key_lo, key_hi, wide)
}

/// The four words that elements `4·group ..= 4·group + 3` read from stream `stream`.
#[cube]
pub fn draw_lane_block(
    group: u32,
    index_hi: u32,
    stream: u32,
    key_lo: u32,
    key_hi: u32,
    #[comptime] wide: bool,
) -> Bits4 {
    philox4x32_10(group, index_hi, stream, DOMAIN_LANE, key_lo, key_hi, wide)
}

/// One random word for element `index` of stream `stream`.
///
/// Four consecutive elements share an evaluation and take a lane each, so a kernel
/// that walks four elements per unit pays for one Philox call instead of four. The
/// mapping is a function of `index` only, so which lane an element reads does not
/// depend on how the launch was shaped — which is what lets the batched and the
/// scalar path produce the same bits.
#[cube]
pub fn draw_lane(
    index: u32,
    index_hi: u32,
    stream: u32,
    key_lo: u32,
    key_hi: u32,
    #[comptime] wide: bool,
) -> u32 {
    let block = draw_lane_block(index >> 2u32, index_hi, stream, key_lo, key_hi, wide);
    let lane = index & 3u32;
    let mut out = block.a;
    if lane == 1u32 {
        out = block.b;
    } else if lane == 2u32 {
        out = block.c;
    } else if lane == 3u32 {
        out = block.d;
    }
    out
}

/// The `lane`-th word of a [`Bits4`], for a lane known only at run time.
#[cube]
pub fn lane_of(block: Bits4, lane: u32) -> u32 {
    let mut out = block.a;
    if lane == 1u32 {
        out = block.b;
    } else if lane == 2u32 {
        out = block.c;
    } else if lane == 3u32 {
        out = block.d;
    }
    out
}

/// A draw in the **open** interval `(0, 1)`.
///
/// The top 23 bits of `bits` pick one of `2²³` equal cells and the draw is that
/// cell's midpoint, so it is strictly inside the interval rather than sitting on its
/// left edge. That matters more than it looks: an inverse-CDF sampler evaluates
/// `ln u`, `ln(1 − u)` or `tan(π(u − ½))`, every one of which is infinite at an
/// endpoint. Excluding both ends makes every sampler in [`super`] total — no draw
/// can produce an infinity, so none of them needs a guard on the hot path.
///
/// Every step is exact: `bits >> 9` is below `2²³`, so it converts to `f32` without
/// rounding; `+ ½` at that magnitude needs the twenty-fourth bit and no more; and
/// [`UNIT_STEP`] is a power of two. The result is bit-identical on any IEEE-754
/// device, which is what lets a sampler be bit-exact even though its `exp` may not
/// be.
#[cube]
pub fn unit_open(bits: u32) -> f32 {
    (f32::cast_from(bits >> 9u32) + 0.5f32) * UNIT_STEP
}

/// A draw in the **half-open** interval `[0, 1)`.
///
/// For the comparisons a Bernoulli-style decision makes, where `u < p` must be false
/// for every draw when `p` is zero and true for every draw when `p` is one.
#[cube]
pub fn unit_half_open(bits: u32) -> f32 {
    f32::cast_from(bits >> 9u32) * UNIT_STEP
}

// ---------------------------------------------------------------------------
// Host twins
// ---------------------------------------------------------------------------

/// The host compilation of everything above.
///
/// The signatures match the device functions exactly, which is what lets the
/// samplers in [`super::univariate`] be written once and compiled twice: the
/// generated host copy of that module differs from the device one only in its
/// `use` lines, which point `rng` and `special` at these twins instead.
///
/// Unlike [`super::special`], this module is a hand-written twin rather than a
/// mechanical copy of the same source. Integer code cannot be shared: the device is
/// expected to wrap a `u32` multiply that overflows, and the host panics on it in a
/// debug build. `distributions_bitexact::rng_device_matches_host_bit_for_bit`
/// carries the weight the shared source would otherwise carry.
pub mod host {
    use super::{
        Bits4, DOMAIN_BLOCK, DOMAIN_LANE, PHILOX_M0, PHILOX_M1, PHILOX_W0, PHILOX_W1, UNIT_STEP,
    };

    /// The host twin of [`super::mulhi()`].
    ///
    /// Both spellings, so that `wide` means the same thing here as it does on the
    /// device — and so that the split path, which is the one a `u32` overflow could
    /// bite, is exercised on the host too. It cannot overflow: every partial product
    /// is of two 16-bit halves.
    pub fn mulhi(a: u32, b: u32, wide: bool) -> u32 {
        if wide {
            return ((a as u64 * b as u64) >> 32) as u32;
        }
        let (a0, a1) = (a & 0xffff, a >> 16);
        let (b0, b1) = (b & 0xffff, b >> 16);
        let (p00, p01, p10, p11) = (a0 * b0, a0 * b1, a1 * b0, a1 * b1);
        let mid = (p00 >> 16) + (p01 & 0xffff) + (p10 & 0xffff);
        p11 + (p01 >> 16) + (p10 >> 16) + (mid >> 16)
    }

    /// The host twin of [`super::philox4x32_10()`].
    pub fn philox4x32_10(
        c0: u32,
        c1: u32,
        c2: u32,
        c3: u32,
        key_lo: u32,
        key_hi: u32,
        wide: bool,
    ) -> Bits4 {
        let mut x = [c0, c1, c2, c3];
        let mut k = [key_lo, key_hi];
        for round in 0..10 {
            let hi0 = mulhi(PHILOX_M0, x[0], wide);
            let lo0 = PHILOX_M0.wrapping_mul(x[0]);
            let hi1 = mulhi(PHILOX_M1, x[2], wide);
            let lo1 = PHILOX_M1.wrapping_mul(x[2]);
            x = [hi1 ^ x[1] ^ k[0], lo1, hi0 ^ x[3] ^ k[1], lo0];
            if round < 9 {
                k[0] = k[0].wrapping_add(PHILOX_W0);
                k[1] = k[1].wrapping_add(PHILOX_W1);
            }
        }
        Bits4 {
            a: x[0],
            b: x[1],
            c: x[2],
            d: x[3],
        }
    }

    /// The host twin of [`super::draw_block()`].
    pub fn draw_block(
        index: u32,
        index_hi: u32,
        stream: u32,
        key_lo: u32,
        key_hi: u32,
        wide: bool,
    ) -> Bits4 {
        philox4x32_10(index, index_hi, stream, DOMAIN_BLOCK, key_lo, key_hi, wide)
    }

    /// The host twin of [`super::draw_lane_block()`].
    pub fn draw_lane_block(
        group: u32,
        index_hi: u32,
        stream: u32,
        key_lo: u32,
        key_hi: u32,
        wide: bool,
    ) -> Bits4 {
        philox4x32_10(group, index_hi, stream, DOMAIN_LANE, key_lo, key_hi, wide)
    }

    /// The host twin of [`super::draw_lane()`].
    pub fn draw_lane(
        index: u32,
        index_hi: u32,
        stream: u32,
        key_lo: u32,
        key_hi: u32,
        wide: bool,
    ) -> u32 {
        let block = draw_lane_block(index >> 2, index_hi, stream, key_lo, key_hi, wide);
        match index & 3 {
            0 => block.a,
            1 => block.b,
            2 => block.c,
            _ => block.d,
        }
    }

    /// The host twin of [`super::unit_open()`].
    pub fn unit_open(bits: u32) -> f32 {
        ((bits >> 9) as f32 + 0.5) * UNIT_STEP
    }

    /// The host twin of [`super::unit_half_open()`].
    pub fn unit_half_open(bits: u32) -> f32 {
        (bits >> 9) as f32 * UNIT_STEP
    }
}
