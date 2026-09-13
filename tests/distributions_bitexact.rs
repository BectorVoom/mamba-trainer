//! Bit-exactness of the distribution primitives.
//!
//! Three separate questions, deliberately kept apart because they can fail for
//! different reasons and a single "it matches" assertion would not say which:
//!
//! 1. **Is the generator the published one?** [`philox_matches_known_answer_vectors`]
//!    checks Philox-4×32-10 against Random123's own test vectors. Without this, a
//!    device-versus-host comparison only proves two copies of the same mistake agree.
//! 2. **Does the device compute what the host computes?** Every special function is
//!    compiled twice from one source (see [`mamba3::distributions::special`]) and
//!    [`special_device_matches_host_bit_for_bit`] requires the two to agree on every
//!    bit of every result — no epsilon.
//! 3. **Is what they both compute correct?** [`special_matches_reference_values`]
//!    checks against `f64` references computed independently, in ulp.
//!
//! # What bit-exactness can and cannot mean here
//!
//! Question 2's answer depends on the runtime. Everything integer — the generator,
//! the counter arithmetic, the conversion of bits to a uniform — is exact on any
//! backend, and so are `+`, `−`, `×`, `÷` and `√` under IEEE-754. What is *not*
//! guaranteed across backends is `exp`, `ln`, `sin` and `tan`: those are library
//! functions, and CUDA's need not agree with the host's in the last bit.
//! [`libm_primitives_agree_with_the_host`] measures exactly that and names the
//! culprit, so a failure on some future backend reads as "this device's `exp`
//! differs by one ulp" rather than as an unexplained mismatch several layers up.
//!
//! On the CPU runtime — the reference backend, and the one this suite runs on by
//! default — all of them agree, so bit-exactness holds end to end.

#![cfg(feature = "backend")]

use cubecl::prelude::*;
use mamba3::backends::Auto;
use mamba3::distributions::rng;
use mamba3::distributions::special::{self, host};

type R = Auto;

include!("golden/special.rs");

// ---------------------------------------------------------------------------
// The generator
// ---------------------------------------------------------------------------

/// The generator in the shape the published vectors are written in.
fn host_philox(counter: [u32; 4], key: [u32; 2]) -> [u32; 4] {
    let b = rng::host::philox4x32_10(
        counter[0], counter[1], counter[2], counter[3], key[0], key[1], true,
    );
    [b.a, b.b, b.c, b.d]
}

/// Whether this device gets the 64-bit multiply or the portable one.
fn wide() -> bool {
    rng::wide_multiply(&R::client(&<R as Runtime>::Device::default()))
}

/// Both spellings of the generator's high-word multiply, side by side.
#[cube(launch_unchecked)]
fn mulhi_kernel(x: &Array<u32>, out: &mut Array<u32>, n: u32) {
    if ABSOLUTE_POS < n as usize {
        let a = x[ABSOLUTE_POS];
        let b = x[(n as usize - 1) - ABSOLUTE_POS];
        out[ABSOLUTE_POS] = rng::mulhi_split(a, b);
        out[n as usize + ABSOLUTE_POS] = rng::mulhi_wide(a, b);
    }
}

/// The portable multiply and the 64-bit one are the same function.
///
/// This is what makes [`rng::wide_multiply`] a performance switch rather than a
/// numerical one: a draw is the same draw whichever route the device took.
#[test]
fn the_two_wide_multiplies_agree() {
    let client = R::client(&<R as Runtime>::Device::default());
    let mut xs: Vec<u32> = vec![
        0,
        1,
        2,
        0xffff,
        0x1_0000,
        0x7fff_ffff,
        0x8000_0000,
        0xffff_ffff,
    ];
    for i in 0..2040u32 {
        xs.push(rng::host::draw_lane(i, 0, 0, 0x1234, 0x5678, true));
    }
    let n = xs.len();
    let x = client.create_from_slice(u32::as_bytes(&xs));
    let out = client.empty(n * 2 * 4);
    let dim = CubeDim::new_1d(64);
    let count = cubecl::calculate_cube_count_elemwise(&client, n, dim);
    unsafe {
        mulhi_kernel::launch_unchecked::<R>(
            &client,
            count,
            dim,
            ArrayArg::from_raw_parts(x, n),
            ArrayArg::from_raw_parts(out.clone(), n * 2),
            n as u32,
        );
    }
    let got = u32::from_bytes(&client.read_one_unchecked(out)).to_vec();
    for i in 0..n {
        let host_split = rng::host::mulhi(xs[i], xs[n - 1 - i], false);
        let host_wide = rng::host::mulhi(xs[i], xs[n - 1 - i], true);
        assert_eq!(host_split, host_wide, "host spellings differ at {i}");
        assert_eq!(got[i], host_split, "device split multiply at {i}");
        assert_eq!(got[n + i], host_wide, "device wide multiply at {i}");
    }
}

#[test]
fn philox_matches_known_answer_vectors() {
    // Random123's published vectors for philox4x32-10.
    assert_eq!(
        host_philox([0, 0, 0, 0], [0, 0]),
        [0x6627_e8d5, 0xe169_c58d, 0xbc57_ac4c, 0x9b00_dbd8]
    );
    assert_eq!(
        host_philox([0xffff_ffff; 4], [0xffff_ffff; 2]),
        [0x408f_276d, 0x41c8_3b0e, 0xa20b_c7c6, 0x6d54_51fd]
    );
    assert_eq!(
        host_philox(
            [0x243f_6a88, 0x85a3_08d3, 0x1319_8a2e, 0x0370_7344],
            [0xa409_3822, 0x299f_31d0]
        ),
        [0xd16c_fe09, 0x94fd_cceb, 0x5001_e420, 0x2412_6ea1]
    );
}

/// The device generator, its host twin, and the lane/block indexing on top of both.
#[cube(launch_unchecked)]
fn rng_kernel(
    out: &mut Array<u32>,
    n: u32,
    stream: u32,
    key_lo: u32,
    key_hi: u32,
    #[comptime] wide: bool,
) {
    if ABSOLUTE_POS < n as usize {
        let i = ABSOLUTE_POS as u32;
        let block = rng::draw_block(i, 0u32, stream, key_lo, key_hi, wide);
        out[6 * ABSOLUTE_POS] = block.a;
        out[6 * ABSOLUTE_POS + 1] = block.b;
        out[6 * ABSOLUTE_POS + 2] = block.c;
        out[6 * ABSOLUTE_POS + 3] = block.d;
        out[6 * ABSOLUTE_POS + 4] = rng::draw_lane(i, 0u32, stream, key_lo, key_hi, wide);
        out[6 * ABSOLUTE_POS + 5] = u32::reinterpret(rng::unit_open(block.a));
    }
}

#[test]
fn rng_device_matches_host_bit_for_bit() {
    let client = R::client(&<R as Runtime>::Device::default());
    let n = 1024usize;
    let seed = 0x0123_4567_89ab_cdefu64;
    let stream = 5u32;
    let out = client.empty(n * 6 * 4);
    let dim = CubeDim::new_1d(64);
    let count = cubecl::calculate_cube_count_elemwise(&client, n, dim);
    unsafe {
        rng_kernel::launch_unchecked::<R>(
            &client,
            count,
            dim,
            ArrayArg::from_raw_parts(out.clone(), n * 6),
            n as u32,
            stream,
            seed as u32,
            (seed >> 32) as u32,
            wide(),
        );
    }
    let got = u32::from_bytes(&client.read_one_unchecked(out)).to_vec();
    for i in 0..n {
        let (lo, hi) = (i as u32, 0u32);
        let block = rng::host::draw_block(lo, hi, stream, seed as u32, (seed >> 32) as u32, wide());
        assert_eq!(
            &got[6 * i..6 * i + 4],
            &[block.a, block.b, block.c, block.d][..],
            "draw_block at {i}"
        );
        assert_eq!(
            got[6 * i + 4],
            rng::host::draw_lane(lo, hi, stream, seed as u32, (seed >> 32) as u32, wide()),
            "draw_lane at {i}"
        );
        assert_eq!(
            got[6 * i + 5],
            rng::host::unit_open(block.a).to_bits(),
            "unit_open at {i}"
        );
    }
}

#[test]
fn uniform_draws_stay_strictly_inside_the_unit_interval() {
    // The samplers rely on this: every one of them takes a logarithm or a tangent
    // of a draw, and an endpoint would make one of them infinite.
    for bits in [0u32, 1, 0xff, 0x8000_0000, 0xffff_ffff, 0x7fff_ffff] {
        let u = rng::host::unit_open(bits);
        assert!(u > 0.0 && u < 1.0, "unit_open({bits:#x}) = {u}");
        let h = rng::host::unit_half_open(bits);
        assert!((0.0..1.0).contains(&h), "unit_half_open({bits:#x}) = {h}");
    }
}

// ---------------------------------------------------------------------------
// The special functions
// ---------------------------------------------------------------------------

/// Every one-argument special function, selected at compile time.
///
/// `which` is `#[comptime]`, so each launch compiles to exactly one function with no
/// branch left in the device code — the dispatch costs nothing and the kernel under
/// test is the same shape a real kernel would call.
#[cube(launch_unchecked)]
fn special1_kernel(x: &Array<f32>, out: &mut Array<f32>, n: u32, #[comptime] which: u32) {
    if ABSOLUTE_POS < n as usize {
        let v = x[ABSOLUTE_POS];
        let mut r: f32 = 0.0;
        if comptime!(which == 0) {
            r = special::erf_f32(v);
        } else if comptime!(which == 1) {
            r = special::erfc_f32(v);
        } else if comptime!(which == 2) {
            r = special::erfinv_f32(v);
        } else if comptime!(which == 3) {
            r = special::lgamma_f32(v);
        } else if comptime!(which == 4) {
            r = special::digamma_f32(v);
        } else if comptime!(which == 5) {
            r = special::log_i0_f32(v);
        } else if comptime!(which == 6) {
            r = special::log_i1_f32(v);
        } else if comptime!(which == 7) {
            r = special::log1p_f32(v);
        } else if comptime!(which == 8) {
            r = special::expm1_f32(v);
        } else if comptime!(which == 9) {
            r = special::log1mexp_f32(v);
        } else if comptime!(which == 10) {
            r = special::softplus_f32(v);
        } else if comptime!(which == 11) {
            r = special::log_sigmoid_f32(v);
        } else if comptime!(which == 12) {
            r = special::std_normal_cdf_f32(v);
        } else if comptime!(which == 13) {
            r = special::std_normal_icdf_f32(v);
        } else if comptime!(which == 14) {
            r = special::exp_neg_square(v);
        } else if comptime!(which == 15) {
            r = special::bessel_ratio_f32(v);
        } else if comptime!(which == 16) {
            r = special::trigamma_f32(v);
        }
        out[ABSOLUTE_POS] = r;
    }
}

/// Every two-argument special function, likewise.
#[cube(launch_unchecked)]
fn special2_kernel(
    a: &Array<f32>,
    b: &Array<f32>,
    out: &mut Array<f32>,
    n: u32,
    #[comptime] which: u32,
) {
    if ABSOLUTE_POS < n as usize {
        let x = a[ABSOLUTE_POS];
        let y = b[ABSOLUTE_POS];
        let mut r: f32 = 0.0;
        if comptime!(which == 0) {
            r = special::lbeta_f32(x, y);
        } else if comptime!(which == 1) {
            r = special::log_binom_f32(x, y);
        } else if comptime!(which == 2) {
            r = special::logaddexp_f32(x, y);
        } else if comptime!(which == 3) {
            r = special::xlogy_f32(x, y);
        } else if comptime!(which == 4) {
            r = special::xlog1py_f32(x, y);
        }
        out[ABSOLUTE_POS] = r;
    }
}

/// Name, kernel index, the interval the function is defined on, the magnitude its
/// accuracy is judged against, and the ulp budget it is held to.
///
/// That last column is what makes the ulp figures mean something. `erfc(8)` is
/// `1.1e-28` and every digit of it is expected to be right, so its floor is zero and
/// the error is purely relative. `lgamma(1)` is exactly zero while the series that
/// produces it works with quantities of order one, so *no* implementation gets a
/// relative digit there; judging it against a floor of one asks the honest question,
/// "is the answer within a few ulp of where zero sits on this function's own scale?"
///
/// The three loose budgets are conditioning, not sloppiness, and each is a
/// subtraction the caller could avoid but the function's signature cannot:
/// `std_normal_cdf(−6)` inherits `erfc`'s tail; `lbeta(0.01, 100)` asks for `4.5`
/// as a difference of log-gammas near `359`; `log_binom(1000, 3)` asks for `18.9`
/// as a difference near `5900`. PyTorch computes all three the same way and loses
/// the same digits.
const ONE_ARG: &[(&str, u32, f32, f32, f32, f32)] = &[
    ("erf_f32", 0, -6.0, 6.0, 0.0, 2.0),
    ("erfc_f32", 1, -6.0, 9.0, 0.0, 6.0),
    ("erfinv_f32", 2, -0.999_99, 0.999_99, 0.0, 2.0),
    ("lgamma_f32", 3, -30.0, 200.0, 1.0, 8.0),
    ("digamma_f32", 4, -30.0, 200.0, 1.0, 8.0),
    ("log_i0_f32", 5, 0.001, 60.0, 1.0, 4.0),
    ("log_i1_f32", 6, 0.001, 60.0, 1.0, 4.0),
    ("log1p_f32", 7, -0.999, 100.0, 0.0, 2.0),
    ("expm1_f32", 8, -30.0, 30.0, 0.0, 2.0),
    ("log1mexp_f32", 9, -30.0, -1e-4, 0.0, 2.0),
    ("softplus_f32", 10, -40.0, 40.0, 0.0, 2.0),
    ("log_sigmoid_f32", 11, -40.0, 40.0, 0.0, 2.0),
    ("std_normal_cdf_f32", 12, -9.0, 9.0, 0.0, 20.0),
    ("std_normal_icdf_f32", 13, 1e-6, 1.0 - 1e-6, 1.0, 4.0),
    ("exp_neg_square", 14, 0.0, 9.0, 0.0, 2.0),
    ("bessel_ratio_f32", 15, 0.0, 60.0, 0.0, 4.0),
    ("trigamma_f32", 16, -30.0, 200.0, 0.0, 8.0),
];

/// Name, kernel index, accuracy floor and tolerance — see [`ONE_ARG`].
const TWO_ARG: &[(&str, u32, f32, f32)] = &[
    ("lbeta_f32", 0, 1.0, 128.0),
    ("log_binom_f32", 1, 1.0, 192.0),
    ("logaddexp_f32", 2, 1.0, 2.0),
    ("xlogy_f32", 3, 1.0, 2.0),
    ("xlog1py_f32", 4, 0.0, 2.0),
];

fn host1(which: u32, v: f32) -> f32 {
    match which {
        0 => host::erf_f32(v),
        1 => host::erfc_f32(v),
        2 => host::erfinv_f32(v),
        3 => host::lgamma_f32(v),
        4 => host::digamma_f32(v),
        5 => host::log_i0_f32(v),
        6 => host::log_i1_f32(v),
        7 => host::log1p_f32(v),
        8 => host::expm1_f32(v),
        9 => host::log1mexp_f32(v),
        10 => host::softplus_f32(v),
        11 => host::log_sigmoid_f32(v),
        12 => host::std_normal_cdf_f32(v),
        13 => host::std_normal_icdf_f32(v),
        14 => host::exp_neg_square(v),
        15 => host::bessel_ratio_f32(v),
        16 => host::trigamma_f32(v),
        _ => unreachable!("unknown one-argument special function {which}"),
    }
}

fn host2(which: u32, x: f32, y: f32) -> f32 {
    match which {
        0 => host::lbeta_f32(x, y),
        1 => host::log_binom_f32(x, y),
        2 => host::logaddexp_f32(x, y),
        3 => host::xlogy_f32(x, y),
        4 => host::xlog1py_f32(x, y),
        _ => unreachable!("unknown two-argument special function {which}"),
    }
}

fn eval1_on_device(which: u32, xs: &[f32]) -> Vec<f32> {
    let client = R::client(&<R as Runtime>::Device::default());
    let n = xs.len();
    let x = client.create_from_slice(f32::as_bytes(xs));
    let out = client.empty(n * 4);
    let dim = CubeDim::new_1d(64);
    let count = cubecl::calculate_cube_count_elemwise(&client, n, dim);
    unsafe {
        special1_kernel::launch_unchecked::<R>(
            &client,
            count,
            dim,
            ArrayArg::from_raw_parts(x, n),
            ArrayArg::from_raw_parts(out.clone(), n),
            n as u32,
            which,
        );
    }
    f32::from_bytes(&client.read_one_unchecked(out))[..n].to_vec()
}

fn eval2_on_device(which: u32, xs: &[f32], ys: &[f32]) -> Vec<f32> {
    let client = R::client(&<R as Runtime>::Device::default());
    let n = xs.len();
    let a = client.create_from_slice(f32::as_bytes(xs));
    let b = client.create_from_slice(f32::as_bytes(ys));
    let out = client.empty(n * 4);
    let dim = CubeDim::new_1d(64);
    let count = cubecl::calculate_cube_count_elemwise(&client, n, dim);
    unsafe {
        special2_kernel::launch_unchecked::<R>(
            &client,
            count,
            dim,
            ArrayArg::from_raw_parts(a, n),
            ArrayArg::from_raw_parts(b, n),
            ArrayArg::from_raw_parts(out.clone(), n),
            n as u32,
            which,
        );
    }
    f32::from_bytes(&client.read_one_unchecked(out))[..n].to_vec()
}

/// A sweep that is dense enough to cross every branch and every fit boundary.
fn sweep(lo: f32, hi: f32, n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| lo + (hi - lo) * (i as f32 + 0.5) / n as f32)
        .collect()
}

#[test]
fn special_device_matches_host_bit_for_bit() {
    for &(name, which, lo, hi, _, _) in ONE_ARG {
        let xs = sweep(lo, hi, 997);
        let got = eval1_on_device(which, &xs);
        for (i, &x) in xs.iter().enumerate() {
            let want = host1(which, x);
            assert_eq!(
                got[i].to_bits(),
                want.to_bits(),
                "{name}({x:e}): device {} vs host {}",
                got[i],
                want
            );
        }
    }
    for &(name, which, _, _) in TWO_ARG {
        let xs = sweep(0.05, 40.0, 331);
        let ys: Vec<f32> = xs.iter().rev().map(|v| v * 0.37 + 0.01).collect();
        let got = eval2_on_device(which, &xs, &ys);
        for i in 0..xs.len() {
            let want = host2(which, xs[i], ys[i]);
            assert_eq!(
                got[i].to_bits(),
                want.to_bits(),
                "{name}({}, {}): device {} vs host {}",
                xs[i],
                ys[i],
                got[i],
                want
            );
        }
    }
}

/// The gap between `v` and the next `f32` above it.
fn ulp_at(v: f32) -> f32 {
    let v = v.abs().max(f32::MIN_POSITIVE);
    f32::from_bits(v.to_bits() + 1) - v
}

/// The error between `got` and `want`, in units of the last place of whichever of
/// `want` and `floor` is larger. See [`ONE_ARG`] for why the floor is needed.
fn ulp_error(got: f32, want: f32, floor: f32) -> f32 {
    if got == want {
        return 0.0;
    }
    (got - want).abs() / ulp_at(want.abs().max(floor))
}

#[test]
fn special_matches_reference_values() {
    // Each function is held to its own budget rather than a shared one, because the
    // budgets differ for a reason worth writing down; see [`ONE_ARG`].
    let mut worst: Vec<(&'static str, f32, f32, f32)> = Vec::new();
    let mut note = |name: &'static str, tol: f32, err: f32, x: f32| match worst
        .iter_mut()
        .find(|(n, ..)| *n == name)
    {
        Some(slot) if slot.2 < err => *slot = (name, tol, err, x),
        Some(_) => {}
        None => worst.push((name, tol, err, x)),
    };

    for &(name, x, want) in SPECIAL_GOLDEN {
        let &(name, which, _, _, floor, tol) = ONE_ARG
            .iter()
            .find(|(n, ..)| *n == name)
            .unwrap_or_else(|| panic!("golden data names an unknown function {name}"));
        note(name, tol, ulp_error(host1(which, x), want, floor), x);
    }
    for &(name, a, b, want) in SPECIAL_GOLDEN2 {
        let &(name, which, floor, tol) = TWO_ARG
            .iter()
            .find(|(n, ..)| *n == name)
            .unwrap_or_else(|| panic!("golden data names an unknown function {name}"));
        note(name, tol, ulp_error(host2(which, a, b), want, floor), a);
    }

    let mut over: Vec<String> = Vec::new();
    for (name, tol, err, x) in &worst {
        println!("{name:>22}: worst {err:6.2} ulp (budget {tol:5.0}, at {x:e})");
        if err > tol {
            over.push(format!(
                "{name} is {err:.2} ulp off at {x:e}, over its {tol} budget"
            ));
        }
    }
    assert!(over.is_empty(), "{}", over.join("; "));
}

// ---------------------------------------------------------------------------
// The assumptions the two above rest on
// ---------------------------------------------------------------------------

/// The library calls the special functions are built on, side by side.
#[cube(launch_unchecked)]
fn libm_kernel(x: &Array<f32>, out: &mut Array<f32>, n: u32) {
    if ABSOLUTE_POS < n as usize {
        let v = x[ABSOLUTE_POS];
        out[ABSOLUTE_POS] = f32::exp(v);
        out[n as usize + ABSOLUTE_POS] = f32::ln(f32::abs(v) + 1.0f32);
        out[2 * n as usize + ABSOLUTE_POS] = f32::sqrt(f32::abs(v));
        out[3 * n as usize + ABSOLUTE_POS] = f32::sin(v);
        out[4 * n as usize + ABSOLUTE_POS] = f32::tan(v);
        out[5 * n as usize + ABSOLUTE_POS] = f32::floor(v * 128.0f32);
        out[6 * n as usize + ABSOLUTE_POS] = f32::powf(f32::abs(v) + 0.5f32, 1.5f32);
    }
}

#[test]
fn libm_primitives_agree_with_the_host() {
    let client = R::client(&<R as Runtime>::Device::default());
    let xs = sweep(-6.0, 6.0, 512);
    let n = xs.len();
    let x = client.create_from_slice(f32::as_bytes(&xs));
    let out = client.empty(n * 7 * 4);
    let dim = CubeDim::new_1d(64);
    let count = cubecl::calculate_cube_count_elemwise(&client, n, dim);
    unsafe {
        libm_kernel::launch_unchecked::<R>(
            &client,
            count,
            dim,
            ArrayArg::from_raw_parts(x, n),
            ArrayArg::from_raw_parts(out.clone(), n * 7),
            n as u32,
        );
    }
    let got = f32::from_bytes(&client.read_one_unchecked(out)).to_vec();
    let names = ["exp", "ln", "sqrt", "sin", "tan", "floor", "powf"];
    let mut differ = Vec::new();
    for (k, name) in names.iter().enumerate() {
        let mut bad = 0usize;
        let mut worst = 0.0f32;
        for (i, &v) in xs.iter().enumerate() {
            let want = match k {
                0 => v.exp(),
                1 => (v.abs() + 1.0).ln(),
                2 => v.abs().sqrt(),
                3 => v.sin(),
                4 => v.tan(),
                5 => (v * 128.0).floor(),
                _ => (v.abs() + 0.5).powf(1.5),
            };
            let g = got[k * n + i];
            if g.to_bits() != want.to_bits() {
                bad += 1;
                worst = worst.max(ulp_error(g, want, 0.0));
            }
        }
        if bad > 0 {
            differ.push(format!("{name}: {bad}/{n} differ, worst {worst:.2} ulp"));
        }
    }
    assert!(
        differ.is_empty(),
        "this backend's libm is not the host's, so bit-exactness above it cannot \
         hold: {}",
        differ.join("; ")
    );
}

/// The host module is this crate's device source, mechanically transformed.
///
/// Not a style check: the whole bit-exactness argument is that the two sides are
/// *the same program*, and this is what makes that true rather than aspirational.
#[test]
fn host_twin_is_the_same_source() {
    for (name, device, host) in [
        (
            "special",
            include_str!("../src/distributions/special.rs"),
            include_str!("../src/distributions/special_host.rs"),
        ),
        (
            "univariate",
            include_str!("../src/distributions/univariate.rs"),
            include_str!("../src/distributions/univariate_host.rs"),
        ),
    ] {
        assert_eq!(
            derive_host(shared_region(device)),
            shared_region(host),
            "{name}_host.rs is stale: re-derive it with `python3 tools/host_twins.py`"
        );
    }
}

/// The text between the shared-region markers.
fn shared_region(src: &str) -> &str {
    let start = src
        .find("// >>> shared\n")
        .expect("source is missing its shared-region marker")
        + "// >>> shared\n".len();
    let end = src[start..]
        .find("\n// <<< shared")
        .expect("source is missing its shared-region end marker")
        + start;
    &src[start..end]
}

/// The three edits that turn device source into host source, as `tools/host_twins.py`
/// documents them.
fn derive_host(region: &str) -> String {
    // `split` rather than `lines`, so a trailing newline survives the round trip and
    // the comparison is byte for byte rather than nearly so.
    region
        .split('\n')
        .filter(|line| line.trim() != "#[cube]")
        .collect::<Vec<_>>()
        .join("\n")
        .replace("#[comptime] ", "")
        .replace("comptime!(", "(")
}

// ---------------------------------------------------------------------------
// The distributions themselves
// ---------------------------------------------------------------------------

use mamba3::autograd::Var;
use mamba3::backend::Device;
use mamba3::distributions::univariate::{Kind, host as uni};
use mamba3::distributions::{Distribution, Univariate};
use mamba3::tensor::Tensor;

/// How many batch elements each family is tested over.
///
/// Wide enough that the launch is not one cube, so a disagreement caused by the
/// geometry would show up here rather than only in
/// [`draws_do_not_depend_on_the_launch_shape`].
const BATCH: usize = 137;

/// A deterministic spread over `[0, 1)`, distinct per `(salt, index)`.
fn spread(salt: u32, i: usize) -> f32 {
    mamba3::distributions::rng::host::unit_open(mamba3::distributions::rng::host::draw_lane(
        i as u32,
        0,
        salt,
        0x5EED_1234,
        0x9ABC_DEF0,
        true,
    ))
}

/// Parameters and test values for one family, chosen inside its support.
fn case(kind: Kind) -> (Vec<Vec<f32>>, Vec<f32>) {
    let u = |salt: u32, i: usize| spread(salt, i);
    let mut params: Vec<Vec<f32>> = Vec::new();
    let mut values = Vec::with_capacity(BATCH);
    match kind {
        Kind::Normal | Kind::Laplace | Kind::Cauchy | Kind::Gumbel | Kind::LogNormal => {
            params.push((0..BATCH).map(|i| 6.0 * u(1, i) - 3.0).collect());
            params.push((0..BATCH).map(|i| 0.1 + 3.0 * u(2, i)).collect());
        }
        Kind::Uniform => {
            let low: Vec<f32> = (0..BATCH).map(|i| 6.0 * u(1, i) - 3.0).collect();
            let high: Vec<f32> = low
                .iter()
                .enumerate()
                .map(|(i, l)| l + 0.2 + 4.0 * u(2, i))
                .collect();
            params.push(low);
            params.push(high);
        }
        Kind::Exponential | Kind::HalfNormal | Kind::HalfCauchy | Kind::Poisson => {
            params.push((0..BATCH).map(|i| 0.1 + 8.0 * u(1, i)).collect());
        }
        Kind::Pareto
        | Kind::Weibull
        | Kind::Kumaraswamy
        | Kind::Gamma
        | Kind::InverseGamma
        | Kind::Beta
        | Kind::FisherSnedecor => {
            params.push((0..BATCH).map(|i| 0.2 + 6.0 * u(1, i)).collect());
            params.push((0..BATCH).map(|i| 0.2 + 6.0 * u(2, i)).collect());
        }
        Kind::StudentT => {
            params.push((0..BATCH).map(|i| 0.5 + 20.0 * u(1, i)).collect());
            params.push((0..BATCH).map(|i| 4.0 * u(2, i) - 2.0).collect());
            params.push((0..BATCH).map(|i| 0.2 + 2.0 * u(3, i)).collect());
        }
        Kind::VonMises => {
            params.push((0..BATCH).map(|i| 6.0 * u(1, i) - 3.0).collect());
            params.push((0..BATCH).map(|i| 0.01 + 20.0 * u(2, i)).collect());
        }
        Kind::ContinuousBernoulli | Kind::Bernoulli | Kind::Geometric => {
            params.push((0..BATCH).map(|i| 8.0 * u(1, i) - 4.0).collect());
        }
        Kind::Binomial => {
            params.push(
                (0..BATCH)
                    .map(|i| (1 + (40.0 * u(1, i)) as u32) as f32)
                    .collect(),
            );
            params.push((0..BATCH).map(|i| 6.0 * u(2, i) - 3.0).collect());
        }
        Kind::NegativeBinomial => {
            params.push((0..BATCH).map(|i| 0.5 + 20.0 * u(1, i)).collect());
            params.push((0..BATCH).map(|i| 4.0 * u(2, i) - 3.0).collect());
        }
        Kind::LogitRelaxedBernoulli | Kind::RelaxedBernoulli => {
            params.push((0..BATCH).map(|i| 0.1 + 2.0 * u(1, i)).collect());
            params.push((0..BATCH).map(|i| 6.0 * u(2, i) - 3.0).collect());
        }
    }
    // The loop indexes `params`, which is a different collection from the one it is
    // filling, so there is nothing to iterate over instead.
    #[allow(clippy::needless_range_loop)]
    for i in 0..BATCH {
        let t = u(9, i);
        values.push(match kind {
            Kind::Normal
            | Kind::Laplace
            | Kind::Cauchy
            | Kind::Gumbel
            | Kind::StudentT
            | Kind::LogitRelaxedBernoulli => 8.0 * t - 4.0,
            Kind::Uniform => params[0][i] + (params[1][i] - params[0][i]) * t,
            Kind::VonMises => 6.2 * t - 3.1,
            Kind::Kumaraswamy | Kind::Beta | Kind::RelaxedBernoulli | Kind::ContinuousBernoulli => {
                0.01 + 0.98 * t
            }
            Kind::Bernoulli => f32::from(t > 0.5),
            Kind::Geometric | Kind::Poisson | Kind::NegativeBinomial => (20.0 * t).floor(),
            Kind::Binomial => (params[0][i] * t).floor(),
            Kind::Pareto => params[0][i] * (1.0 + 4.0 * t),
            _ => 0.05 + 6.0 * t,
        });
    }
    (params, values)
}

/// `true` when two `f32`s are the same value, counting every `NaN` as equal — which
/// is what an "is this the same computation" question means, unlike `==`.
fn same(a: f32, b: f32) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

fn build(kind: Kind, params: &[Vec<f32>], device: &Device<Auto>) -> Univariate<Auto, f32> {
    let slots: Vec<_> = params
        .iter()
        .map(|p| {
            Tensor::<Auto, f32>::from_f32(p, vec![BATCH], device)
                .expect("a parameter row fills the batch")
                .into()
        })
        .collect();
    Univariate::new(kind, slots, device).expect("the case table supplies the right arity")
}

/// Every scalar family, every operation, against the host compilation of the very
/// same source.
///
/// This is the test the module is built to pass. It is not "the numbers are close":
/// every one of the twenty-six families is asked for its density, its distribution
/// function, its quantile, its entropy, its first two moments, its mode and a draw,
/// and every result has to match the host's *bit for bit*. A one-ulp disagreement
/// anywhere — in a Lanczos series, in a rejection loop's trip count, in the order a
/// sum was taken — fails it.
#[test]
fn every_distribution_matches_its_host_twin_bit_for_bit() {
    let device = Device::<Auto>::default();
    let seed = 0x1234_5678_9ABC_DEF0u64;
    let (key_lo, key_hi) = (seed as u32, (seed >> 32) as u32);

    for kind in Kind::ALL {
        let (params, values) = case(kind);
        let dist = build(kind, &params, &device);
        let code = kind.code();
        let at = |slot: usize, i: usize| params.get(slot).map_or(0.0, |p| p[i]);

        let value = Var::constant(
            Tensor::<Auto, f32>::from_f32(&values, vec![BATCH], &device).expect("values fill"),
        );
        let got = dist.log_prob(&value).expect("log_prob is total").to_f32();
        for i in 0..BATCH {
            let want = uni::log_prob_of(values[i], at(0, i), at(1, i), at(2, i), code);
            assert!(
                same(got[i], want),
                "{kind:?}.log_prob at {i}: device {} host {}",
                got[i],
                want
            );
        }

        if kind.has_cdf() {
            let got = dist.cdf(value.tensor()).expect("cdf is total").to_f32();
            for i in 0..BATCH {
                let want = uni::cdf_of(values[i], at(0, i), at(1, i), at(2, i), code);
                assert!(same(got[i], want), "{kind:?}.cdf at {i}");
            }
        }

        if kind.has_icdf() {
            let qs: Vec<f32> = (0..BATCH).map(|i| 0.001 + 0.998 * spread(21, i)).collect();
            let q = Var::constant(
                Tensor::<Auto, f32>::from_f32(&qs, vec![BATCH], &device).expect("quantiles fill"),
            );
            let got = dist.icdf(&q).expect("icdf is total").to_f32();
            for i in 0..BATCH {
                let want = uni::icdf_of(qs[i], at(0, i), at(1, i), at(2, i), code);
                assert!(same(got[i], want), "{kind:?}.icdf at {i}");
            }
        }

        if kind.has_entropy() {
            let got = dist.entropy().expect("entropy is total").to_f32();
            for (i, &g) in got.iter().enumerate() {
                let want = uni::entropy_of(at(0, i), at(1, i), at(2, i), code);
                assert!(same(g, want), "{kind:?}.entropy at {i}");
            }
        }

        for (which, name) in [(0u32, "mean"), (1, "variance"), (2, "mode")] {
            let got = match which {
                0 => dist.mean(),
                1 => dist.variance(),
                _ => dist.mode(),
            }
            .expect("a summary is total")
            .to_f32();
            for (i, &g) in got.iter().enumerate() {
                let want = uni::moment_of(at(0, i), at(1, i), at(2, i), code, which);
                assert!(
                    same(g, want),
                    "{kind:?}.{name} at {i}: device {g} host {want}"
                );
            }
        }

        let got = dist.sample(seed).expect("sampling is total").to_f32();
        for (i, &g) in got.iter().enumerate() {
            let want = uni::sample_of(
                at(0, i),
                at(1, i),
                at(2, i),
                i as u32,
                0,
                key_lo,
                key_hi,
                code,
                wide(),
            );
            assert!(
                same(g, want),
                "{kind:?}.sample at {i}: device {g} host {want}"
            );
        }
    }
}

/// A draw depends on where it is and on nothing else.
///
/// Three separate ways of asking for element `i` — a batch of `n`, a batch of
/// `16·n`, and `n` rows of a `sample_n` — go through different cube geometries and
/// different total element counts. All three must return the same bits, because the
/// counter the generator uses is the element index and not anything about the
/// launch.
#[test]
fn draws_do_not_depend_on_the_launch_shape() {
    let device = Device::<Auto>::default();
    let seed = 0xFACE_B00Cu64;
    let small = Univariate::<Auto, f32>::normal(0.0, 1.0, &device).unwrap();

    let a = small.sample_n(7, seed).unwrap().to_f32();
    let b = small.sample_n(4096, seed).unwrap().to_f32();
    for i in 0..7 {
        assert_eq!(
            a[i].to_bits(),
            b[i].to_bits(),
            "element {i} differs between a 7-draw and a 4096-draw"
        );
    }

    // And twice with the same seed is twice the same answer.
    let c = small.sample_n(4096, seed).unwrap().to_f32();
    assert!(
        b.iter().zip(&c).all(|(x, y)| x.to_bits() == y.to_bits()),
        "two identical calls disagreed"
    );

    // A different seed is a different stream.
    let d = small.sample_n(4096, seed ^ 1).unwrap().to_f32();
    let matches = b.iter().zip(&d).filter(|(x, y)| x == y).count();
    assert!(
        matches < 8,
        "flipping one seed bit changed only {matches} draws"
    );
}

/// A row's answer does not depend on the rows beside it.
#[test]
fn rows_are_independent_of_the_batch_they_are_in() {
    let device = Device::<Auto>::default();
    let seed = 0x0BAD_C0DEu64;
    for kind in [Kind::Gamma, Kind::Poisson, Kind::Binomial, Kind::VonMises] {
        let (params, _) = case(kind);
        let wide = build(kind, &params, &device).sample(seed).unwrap().to_f32();
        // The same parameters, but only the first eight rows of them.
        let narrow_params: Vec<Vec<f32>> = params.iter().map(|p| p[..8].to_vec()).collect();
        let slots: Vec<_> = narrow_params
            .iter()
            .map(|p| {
                Tensor::<Auto, f32>::from_f32(p, vec![8], &device)
                    .unwrap()
                    .into()
            })
            .collect();
        let narrow = Univariate::<Auto, f32>::new(kind, slots, &device)
            .unwrap()
            .sample(seed)
            .unwrap()
            .to_f32();
        for i in 0..8 {
            assert_eq!(
                wide[i].to_bits(),
                narrow[i].to_bits(),
                "{kind:?} row {i} changed when the batch around it did"
            );
        }
    }
}
