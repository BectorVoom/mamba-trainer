#!/usr/bin/env python3
"""Regenerate the reference tables in `tests/golden/`.

    python3 tests/golden/generate.py

Everything here is computed in `f64` from series, quadrature or the Python standard
library — never from this crate — so the tables are an *independent* check rather
than a snapshot of whatever the kernels happened to produce. Each reference is
evaluated at the `f32`-rounded argument the test will actually pass, so the ulp
error the test reports is the implementation's and not the argument's.

Requires only the standard library.
"""

import math
import os
import struct


def f32(x):
    """Round a Python float to the nearest `f32`."""
    return struct.unpack("f", struct.pack("f", x))[0]


def erfinv(y):
    """Bisect then Newton-polish on `math.erf`."""
    if y == 0.0:
        return 0.0
    s = 1.0 if y > 0 else -1.0
    y = abs(y)
    lo, hi = 0.0, 7.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if math.erf(mid) < y:
            lo = mid
        else:
            hi = mid
    x = 0.5 * (lo + hi)
    for _ in range(6):
        d = 2.0 / math.sqrt(math.pi) * math.exp(-x * x)
        if d == 0.0:
            break
        x -= (math.erf(x) - y) / d
    return s * x


def digamma(x):
    """Recurrence up to 40, then the Bernoulli asymptotic series."""
    if x < 0.0:
        return digamma(1.0 - x) - math.pi / math.tan(math.pi * x)
    acc, y = 0.0, x
    while y < 40.0:
        acc -= 1.0 / y
        y += 1.0
    inv = 1.0 / y
    i2 = inv * inv
    acc += math.log(y) - 0.5 * inv
    p = i2
    for b in (1 / 12, -1 / 120, 1 / 252, -1 / 240, 1 / 132, -691 / 32760, 1 / 12, -3617 / 8160):
        acc -= b * p
        p *= i2
    return acc


def trigamma(x):
    """Recurrence up to 40, then the Bernoulli asymptotic series."""
    if x < 0.0:
        return math.pi ** 2 / math.sin(math.pi * x) ** 2 - trigamma(1.0 - x)
    acc, y = 0.0, x
    while y < 40.0:
        acc += 1.0 / (y * y)
        y += 1.0
    inv = 1.0 / y
    i2 = inv * inv
    acc += inv + 0.5 * i2
    p = i2 * inv
    for b in (1 / 6, -1 / 30, 1 / 42, -1 / 30, 5 / 66, -691 / 2730, 7 / 6):
        acc += b * p
        p *= i2
    return acc


def log_bessel_i(nu, x):
    """`ln I_nu(x)` from the ascending series, summed in log space.

    The series peaks near `k = x/2`, so the term count has to grow with `x`.
    """
    n = int(x) + 400
    terms = [
        (2 * k + nu) * math.log(x / 2.0) - math.lgamma(k + 1) - math.lgamma(k + nu + 1)
        for k in range(n)
    ]
    m = max(terms)
    return m + math.log(sum(math.exp(t - m) for t in terms))


def bessel_ratio(x):
    """`I1(x)/I0(x)` by Miller's backward recurrence, which is stable for any `x`.

    Taking the ratio of two log-space series instead loses every digit that the two
    logarithms share, which at `x = 700` is all of them.
    """
    n = int(x + 40 * math.sqrt(max(x, 1.0))) + 40
    bip, bi = 0.0, 1.0
    for k in range(n, 0, -1):
        bip, bi = bi, bip + 2.0 * k / x * bi
        if abs(bi) > 1e100:
            bi *= 1e-100
            bip *= 1e-100
    return bip / bi


def log1mexp(x):
    return math.log(-math.expm1(x)) if x > -math.log(2) else math.log1p(-math.exp(x))


ONE_ARG = {
    "erf_f32": (math.erf,
        [-4.5, -3.0, -2.0, -1.5, -1.0, -0.999, -0.5, -0.25, -0.0625, -1e-4, 0.0,
         1e-4, 0.0625, 0.25, 0.5, 0.75, 0.999, 1.0, 1.001, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]),
    "erfc_f32": (math.erfc,
        [-3.0, -1.5, -1.0, -0.5, 0.0, 0.25, 0.5, 0.9, 1.0, 1.1, 1.5, 2.0, 2.5, 2.6,
         3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
    "erfinv_f32": (erfinv,
        [-0.999999, -0.99, -0.9, -0.5, -0.1, -0.001, 0.0, 0.001, 0.1, 0.25, 0.5,
         0.75, 0.9, 0.99, 0.999, 0.9999, 0.999999]),
    "lgamma_f32": (math.lgamma,
        [-4.3, -2.5, -1.7, -0.5, -0.1, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5,
         3.0, 4.5, 7.0, 10.0, 50.0, 200.0, 1000.0]),
    "digamma_f32": (digamma,
        [-4.3, -2.5, -0.5, 0.05, 0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 7.9, 8.0, 8.1,
         20.0, 100.0, 1000.0]),
    "trigamma_f32": (trigamma,
        [-4.3, -2.5, -0.5, 0.05, 0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 7.9, 8.0, 8.1,
         20.0, 100.0, 1000.0]),
    "log_i0_f32": (lambda x: log_bessel_i(0, abs(x)),
        [0.1, 0.5, 1.0, 2.0, 3.0, 3.74, 3.76, 4.0, 6.0, 10.0, 30.0, 100.0, 700.0]),
    "log_i1_f32": (lambda x: log_bessel_i(1, abs(x)),
        [0.1, 0.5, 1.0, 2.0, 3.0, 3.74, 3.76, 4.0, 6.0, 10.0, 30.0, 100.0, 700.0]),
    "bessel_ratio_f32": (lambda x: bessel_ratio(abs(x)),
        [0.1, 0.5, 1.0, 2.0, 3.76, 6.0, 10.0, 30.0, 100.0, 700.0]),
    "log1p_f32": (math.log1p,
        [-0.9999, -0.9, -0.5, -0.1, -1e-4, -1e-7, 0.0, 1e-7, 1e-4, 0.1, 0.5, 1.0,
         10.0, 1e4, 1e10]),
    "expm1_f32": (math.expm1,
        [-20.0, -5.0, -1.0, -0.7, -0.69, -0.1, -1e-4, -1e-8, 0.0, 1e-8, 1e-4, 0.1,
         0.69, 0.71, 1.0, 5.0, 20.0]),
    "log1mexp_f32": (log1mexp,
        [-20.0, -5.0, -1.0, -0.7, -0.6931472, -0.69, -0.5, -0.1, -1e-3, -1e-6]),
    "softplus_f32": (lambda x: math.log1p(math.exp(x)) if x < 30 else x,
        [-30.0, -10.0, -1.0, -0.1, 0.0, 0.1, 1.0, 10.0, 30.0, 50.0]),
    "log_sigmoid_f32": (lambda x: -(math.log1p(math.exp(-x)) if x > -30 else -x),
        [-50.0, -30.0, -10.0, -1.0, 0.0, 1.0, 10.0, 30.0]),
    "std_normal_cdf_f32": (lambda x: 0.5 * math.erfc(-x / math.sqrt(2)),
        [-8.0, -6.0, -4.0, -3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0]),
    "std_normal_icdf_f32": (lambda u: math.sqrt(2) * erfinv(2 * u - 1),
        [1e-6, 1e-4, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.9999, 0.999999]),
    "exp_neg_square": (lambda x: math.exp(-x * x),
        [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]),
}

TWO_ARG = {
    "lbeta_f32": (lambda a, b: math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b),
        [(0.5, 0.5), (1.0, 1.0), (2.0, 3.0), (0.1, 0.1), (5.0, 7.0), (50.0, 2.0),
         (1e3, 1e3), (0.01, 100.0)]),
    "log_binom_f32": (lambda n, k: math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1),
        [(5.0, 2.0), (10.0, 0.0), (10.0, 10.0), (100.0, 50.0), (1000.0, 3.0), (7.0, 3.5)]),
    "logaddexp_f32": (lambda a, b: max(a, b) + math.log1p(math.exp(-abs(a - b))),
        [(0.0, 0.0), (1.0, -1.0), (100.0, 99.0), (-100.0, -101.0), (50.0, -50.0), (1e-8, 0.0)]),
    "xlogy_f32": (lambda x, y: 0.0 if x == 0 else x * math.log(y),
        [(0.0, 0.0), (0.0, 5.0), (1.0, 2.0), (3.0, 0.5), (-2.0, 7.0)]),
    "xlog1py_f32": (lambda x, y: 0.0 if x == 0 else x * math.log1p(y),
        [(0.0, -1.0), (1.0, 1e-8), (3.0, -0.5), (2.0, 10.0)]),
}


def main():
    out = [
        "// Generated by tests/golden/generate.py. Do not edit.",
        "",
        "/// Reference values for the one-argument special functions.",
        "///",
        "/// `(name, argument, reference)`, the reference computed in `f64` at the",
        "/// `f32`-rounded argument and then rounded to `f32`.",
        "#[allow(clippy::approx_constant, clippy::excessive_precision)]",
        "pub const SPECIAL_GOLDEN: &[(&str, f32, f32)] = &[",
    ]
    for name, (fn, xs) in ONE_ARG.items():
        for x in xs:
            xf = f32(x)
            out.append(f'    ("{name}", {xf:.9e}, {f32(fn(xf)):.9e}),')
    out += ["];", "",
            "/// Reference values for the two-argument special functions.",
            "#[allow(clippy::approx_constant, clippy::excessive_precision)]",
            "pub const SPECIAL_GOLDEN2: &[(&str, f32, f32, f32)] = &["]
    for name, (fn, ps) in TWO_ARG.items():
        for a, b in ps:
            af, bf = f32(a), f32(b)
            out.append(f'    ("{name}", {af:.9e}, {bf:.9e}, {f32(fn(af, bf)):.9e}),')
    out += ["];", ""]
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "special.rs")
    with open(path, "w") as handle:
        handle.write("\n".join(out))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
