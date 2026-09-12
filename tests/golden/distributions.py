#!/usr/bin/env python3
"""Reference densities for `tests/golden/distributions.rs`, and the quadrature that
checks them.

    python3 tests/golden/distributions.py

The densities below are transcribed from the standard references, not from the Rust.
That transcription is then *verified numerically* before anything is emitted: every
continuous family's density is integrated over its support and must come to one, and
its mean, variance and entropy are produced by the same quadrature rather than by a
second formula. A density with a wrong constant, a wrong exponent or a wrong support
fails to integrate to one, so the table cannot silently encode the same mistake the
kernels might.

Discrete families are summed rather than integrated, with the same three checks.

The substitution `x = tan(...)` maps each support onto the unit interval, which is
what lets one Simpson rule handle a Cauchy's tails and a beta's endpoints alike.
"""

import math
import os

# ---------------------------------------------------------------------------
# Special functions, in f64
# ---------------------------------------------------------------------------


def digamma(x):
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
    for b in (1 / 12, -1 / 120, 1 / 252, -1 / 240, 1 / 132, -691 / 32760, 1 / 12):
        acc -= b * p
        p *= i2
    return acc


def log_i0(x):
    """`ln I₀(x)` from the ascending series, in log space."""
    x = abs(x)
    if x == 0.0:
        return 0.0
    n = int(x) + 200
    terms = [2 * k * math.log(x / 2.0) - 2 * math.lgamma(k + 1) for k in range(n)]
    m = max(terms)
    return m + math.log(sum(math.exp(t - m) for t in terms))


def lbeta(a, b):
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def softplus(x):
    return max(x, 0.0) + math.log1p(math.exp(-abs(x)))


def log_sigmoid(x):
    return -softplus(-x)


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def cont_bern_log_norm(logits):
    p = sigmoid(logits)
    d = p - 0.5
    if abs(d) > 1e-3:
        return math.log(abs(math.log1p(-p) - math.log(p))) - math.log(abs(1 - 2 * p))
    return math.log(2.0) + (4.0 / 3.0 + 104.0 / 45.0 * d * d) * d * d


# ---------------------------------------------------------------------------
# The densities, one per family, keyed by the discriminant in `Kind`
# ---------------------------------------------------------------------------
#
# Each entry is (name, support, log_pdf(x, a, b, c)). The support is one of
# "real", "positive", "unit", "interval" (uses a and b), "circle", or an integer
# support given as a function of the parameters.

LN_2PI = math.log(2 * math.pi)
EULER = 0.5772156649015328606


def _normal(x, a, b, c):
    z = (x - a) / b
    return -0.5 * z * z - math.log(b) - 0.5 * LN_2PI


def _uniform(x, a, b, c):
    return -math.log(b - a) if a <= x < b else -math.inf


def _exponential(x, a, b, c):
    return math.log(a) - a * x


def _laplace(x, a, b, c):
    return -math.log(2 * b) - abs(x - a) / b


def _cauchy(x, a, b, c):
    z = (x - a) / b
    return -math.log(math.pi) - math.log(b) - math.log1p(z * z)


def _gumbel(x, a, b, c):
    z = (x - a) / b
    return -(z + math.exp(-z)) - math.log(b)


def _half_normal(x, a, b, c):
    return -math.inf if x < 0 else _normal(x, 0.0, a, c) + math.log(2.0)


def _half_cauchy(x, a, b, c):
    return -math.inf if x < 0 else _cauchy(x, 0.0, a, c) + math.log(2.0)


def _log_normal(x, a, b, c):
    return -math.inf if x <= 0 else _normal(math.log(x), a, b, c) - math.log(x)


def _pareto(x, a, b, c):
    return -math.inf if x < a else math.log(b) + b * math.log(a) - (b + 1) * math.log(x)


def _weibull(x, a, b, c):
    if x <= 0:
        return -math.inf
    z = x / a
    return math.log(b) - math.log(a) + (b - 1) * math.log(z) - z ** b


def _kumaraswamy(x, a, b, c):
    if not 0 < x < 1:
        return -math.inf
    return math.log(a) + math.log(b) + (a - 1) * math.log(x) + (b - 1) * math.log1p(-(x ** a))


def _gamma(x, a, b, c):
    if x <= 0:
        return -math.inf
    return a * math.log(b) + (a - 1) * math.log(x) - b * x - math.lgamma(a)


def _inverse_gamma(x, a, b, c):
    if x <= 0:
        return -math.inf
    return a * math.log(b) - math.lgamma(a) - (a + 1) * math.log(x) - b / x


def _beta(x, a, b, c):
    if not 0 < x < 1:
        return -math.inf
    return (a - 1) * math.log(x) + (b - 1) * math.log1p(-x) - lbeta(a, b)


def _student_t(x, a, b, c):
    z = (x - b) / c
    return (
        math.lgamma(0.5 * (a + 1))
        - math.lgamma(0.5 * a)
        - 0.5 * math.log(a * math.pi)
        - math.log(c)
        - 0.5 * (a + 1) * math.log1p(z * z / a)
    )


def _fisher(x, a, b, c):
    if x <= 0:
        return -math.inf
    h1, h2, r = 0.5 * a, 0.5 * b, a / b
    return (
        math.lgamma(h1 + h2)
        - math.lgamma(h1)
        - math.lgamma(h2)
        + h1 * math.log(r)
        + (h1 - 1) * math.log(x)
        - (h1 + h2) * math.log1p(r * x)
    )


def _von_mises(x, a, b, c):
    return b * math.cos(x - a) - math.log(2 * math.pi) - log_i0(b)


def _cont_bernoulli(x, a, b, c):
    if not 0 <= x <= 1:
        return -math.inf
    return x * a - softplus(a) + cont_bern_log_norm(a)


def _bernoulli(x, a, b, c):
    return x * a - softplus(a)


def _geometric(x, a, b, c):
    return x * log_sigmoid(-a) + log_sigmoid(a)


def _poisson(x, a, b, c):
    return (0.0 if x == 0 else x * math.log(a)) - a - math.lgamma(x + 1)


def _binomial(x, a, b, c):
    return (
        math.lgamma(a + 1)
        - math.lgamma(x + 1)
        - math.lgamma(a - x + 1)
        + x * b
        - a * softplus(b)
    )


def _negative_binomial(x, a, b, c):
    return (
        a * log_sigmoid(-b)
        + x * log_sigmoid(b)
        + math.lgamma(a + x)
        - math.lgamma(1 + x)
        - math.lgamma(a)
    )


def _logit_relaxed(x, a, b, c):
    diff = b - x * a
    return math.log(a) + diff - 2 * softplus(diff)


def _relaxed(x, a, b, c):
    if not 0 < x < 1:
        return -math.inf
    y = math.log(x) - math.log1p(-x)
    diff = b - y * a
    return math.log(a) + diff - 2 * softplus(diff) - math.log(x) - math.log1p(-x)


FAMILIES = [
    ("Normal", 0, 2, "real", _normal),
    ("Uniform", 1, 2, "interval", _uniform),
    ("Exponential", 2, 1, "positive", _exponential),
    ("Laplace", 3, 2, "real", _laplace),
    ("Cauchy", 4, 2, "real", _cauchy),
    ("Gumbel", 5, 2, "real", _gumbel),
    ("HalfNormal", 6, 1, "positive", _half_normal),
    ("HalfCauchy", 7, 1, "positive", _half_cauchy),
    ("LogNormal", 8, 2, "positive", _log_normal),
    ("Pareto", 9, 2, "above_a", _pareto),
    ("Weibull", 10, 2, "positive", _weibull),
    ("Kumaraswamy", 11, 2, "unit", _kumaraswamy),
    ("Gamma", 12, 2, "positive", _gamma),
    ("InverseGamma", 13, 2, "positive", _inverse_gamma),
    ("Beta", 14, 2, "unit", _beta),
    ("StudentT", 15, 3, "real", _student_t),
    ("FisherSnedecor", 16, 2, "positive", _fisher),
    ("VonMises", 17, 2, "circle", _von_mises),
    ("ContinuousBernoulli", 18, 1, "unit_closed", _cont_bernoulli),
    ("Bernoulli", 19, 1, "bool", _bernoulli),
    ("Geometric", 20, 1, "counting", _geometric),
    ("Poisson", 21, 1, "counting", _poisson),
    ("Binomial", 22, 2, "bounded", _binomial),
    ("NegativeBinomial", 23, 2, "counting", _negative_binomial),
    ("LogitRelaxedBernoulli", 24, 2, "real", _logit_relaxed),
    ("RelaxedBernoulli", 25, 2, "unit", _relaxed),
]

DISCRETE = {"bool", "counting", "bounded"}

# ---------------------------------------------------------------------------
# Quadrature
# ---------------------------------------------------------------------------
#
# Double-exponential rules, because half of these densities are singular at an
# endpoint (a Beta with concentrations below one, a Gamma with a shape below one)
# and the other half have tails a fixed grid cannot reach (a Cauchy, a Student's t).
# A double-exponential substitution turns both into integrands that decay like
# `exp(-exp(t))`, which the trapezoid rule then integrates to machine precision with
# a few hundred nodes. Simpson on a tangent substitution — the obvious thing — gets
# `Gamma(0.5, 3)` wrong in the third digit.

H = 1.0 / 96.0
LIMIT = 3.6


def _de_nodes():
    k = 0
    while k * H <= LIMIT:
        yield k
        if k > 0:
            yield -k
        k += 1


def _accumulate(term):
    total = 0.0
    for k in _de_nodes():
        t = k * H
        try:
            v = term(t)
        except (ValueError, OverflowError, ZeroDivisionError):
            continue
        if math.isfinite(v):
            total += v
    return total * H


def tanh_sinh(f, lo, hi):
    """`∫ f` over a finite interval, with endpoint singularities allowed."""
    mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
    if half <= 0.0:
        return 0.0

    def term(t):
        s = math.sinh(t)
        u = math.tanh(0.5 * math.pi * s)
        w = 0.5 * math.pi * math.cosh(t) / math.cosh(0.5 * math.pi * s) ** 2
        return f(mid + half * u) * w * half

    return _accumulate(term)


def exp_sinh(f):
    """`∫₀^∞ f`."""

    def term(t):
        s = math.sinh(t)
        x = math.exp(0.5 * math.pi * s)
        w = x * 0.5 * math.pi * math.cosh(t)
        return f(x) * w

    return _accumulate(term)


def over_line(f, kink):
    """`∫₋∞^∞ f`, split at `kink`.

    Two exp-sinh halves rather than one sinh-sinh, so that a density with a corner —
    a Laplace's `|x − μ|` — has that corner on a boundary instead of in the middle of
    a rule that assumes analyticity. Splitting a smooth density costs nothing.
    """
    return exp_sinh(lambda s: f(kink - s)) + exp_sinh(lambda s: f(kink + s))


def over_support(f, support, a, b, kink=0.0):
    """`∫ f` over the whole of a continuous support."""
    if support == "real":
        return over_line(f, kink)
    if support == "positive":
        return exp_sinh(f)
    if support == "above_a":
        return exp_sinh(lambda s: f(a + s))
    if support in ("unit", "unit_closed"):
        return tanh_sinh(f, 0.0, 1.0)
    if support == "interval":
        return tanh_sinh(f, a, b)
    if support == "circle":
        return tanh_sinh(f, -math.pi, math.pi)
    raise ValueError(support)


def up_to(f, support, a, b, x, kink=0.0):
    """`∫ f` over the part of the support at or below `x`."""
    if support == "real":
        # `∫₋∞^x f = ∫₀^∞ f(x − s) ds`, which keeps a heavy tail inside the rule;
        # split at the corner, as `over_line` does, when `x` is past it.
        if x > kink:
            return exp_sinh(lambda s: f(kink - s)) + tanh_sinh(f, kink, x)
        return exp_sinh(lambda s: f(x - s))
    if support == "positive":
        return tanh_sinh(f, 0.0, x) if x > 0 else 0.0
    if support == "above_a":
        return tanh_sinh(f, a, x) if x > a else 0.0
    if support in ("unit", "unit_closed"):
        return tanh_sinh(f, 0.0, min(x, 1.0)) if x > 0 else 0.0
    if support == "interval":
        return tanh_sinh(f, a, min(x, b)) if x > a else 0.0
    if support == "circle":
        return tanh_sinh(f, -math.pi, min(x, math.pi)) if x > -math.pi else 0.0
    raise ValueError(support)


def support_points(support, a, b):
    if support == "bool":
        return [0.0, 1.0]
    if support == "bounded":
        return [float(k) for k in range(int(round(a)) + 1)]
    # counting: far enough out that the tail is below 1e-15
    return [float(k) for k in range(0, 4000)]


#: Families whose density has a corner in the interior of its support, and where it
#: is. Only the Laplace, whose `|x − μ|` is not differentiable at the location.
CORNERS = {_laplace}


def summarise(logpdf, support, a, b, c):
    """Total mass, mean, variance and entropy, from quadrature or summation.

    A von Mises reports its *circular* mean and variance — the location, and
    `1 − E[cos(X − μ)]` — because those are what the distribution is about and what
    PyTorch returns. The linear mean of an angle on `[−π, π)` is an artefact of where
    the branch cut was put.
    """
    if support in DISCRETE:
        pts = support_points(support, a, b)
        ps = []
        for x in pts:
            try:
                ps.append(math.exp(logpdf(x, a, b, c)))
            except (ValueError, OverflowError):
                ps.append(0.0)
        mass = sum(ps)
        mean = sum(x * p for x, p in zip(pts, ps))
        second = sum(x * x * p for x, p in zip(pts, ps))
        ent = -sum(p * logpdf(x, a, b, c) for x, p in zip(pts, ps) if p > 0)
        return mass, mean, second - mean * mean, ent

    def pdf(x):
        try:
            return math.exp(logpdf(x, a, b, c))
        except (ValueError, OverflowError):
            return 0.0

    def plogp(x):
        p = pdf(x)
        return -p * logpdf(x, a, b, c) if p > 0.0 else 0.0

    kink = a if logpdf in CORNERS else 0.0
    mass = over_support(pdf, support, a, b, kink)
    ent = over_support(plogp, support, a, b, kink)
    if support == "circle":
        resultant = over_support(lambda x: math.cos(x - a) * pdf(x), support, a, b, kink)
        return mass, a, 1.0 - resultant, ent
    mean = over_support(lambda x: x * pdf(x), support, a, b, kink)
    second = over_support(lambda x: x * x * pdf(x), support, a, b, kink)
    return mass, mean, second - mean * mean, ent


def cdf_at(logpdf, support, a, b, c, x):
    """`P(X ≤ x)` by quadrature or summation."""
    if support in DISCRETE:
        return sum(
            math.exp(logpdf(k, a, b, c))
            for k in support_points(support, a, b)
            if k <= x + 1e-9
        )

    def pdf(t):
        try:
            return math.exp(logpdf(t, a, b, c))
        except (ValueError, OverflowError):
            return 0.0

    return up_to(pdf, support, a, b, x, a if logpdf in CORNERS else 0.0)


CASES = {
    "Normal": [(0.0, 1.0, 0.0), (1.5, 0.4, 0.0), (-2.0, 3.0, 0.0)],
    "Uniform": [(-1.0, 2.0, 0.0), (0.0, 0.5, 0.0)],
    "Exponential": [(1.0, 0.0, 0.0), (0.3, 0.0, 0.0), (4.0, 0.0, 0.0)],
    "Laplace": [(0.0, 1.0, 0.0), (2.0, 0.5, 0.0)],
    "Cauchy": [(0.0, 1.0, 0.0), (1.0, 2.0, 0.0)],
    "Gumbel": [(0.0, 1.0, 0.0), (-1.0, 0.7, 0.0)],
    "HalfNormal": [(1.0, 0.0, 0.0), (2.5, 0.0, 0.0)],
    "HalfCauchy": [(1.0, 0.0, 0.0), (0.5, 0.0, 0.0)],
    "LogNormal": [(0.0, 1.0, 0.0), (0.5, 0.3, 0.0)],
    "Pareto": [(1.0, 3.0, 0.0), (2.0, 5.0, 0.0)],
    "Weibull": [(1.0, 2.0, 0.0), (2.0, 0.8, 0.0)],
    "Kumaraswamy": [(2.0, 3.0, 0.0), (0.7, 1.5, 0.0)],
    "Gamma": [(2.0, 1.0, 0.0), (0.5, 3.0, 0.0), (7.0, 2.0, 0.0)],
    "InverseGamma": [(3.0, 2.0, 0.0), (5.0, 1.0, 0.0)],
    "Beta": [(2.0, 5.0, 0.0), (0.5, 0.5, 0.0), (3.0, 3.0, 0.0)],
    "StudentT": [(5.0, 0.0, 1.0), (2.5, 1.0, 2.0), (30.0, -1.0, 0.5)],
    "FisherSnedecor": [(6.0, 8.0, 0.0), (10.0, 12.0, 0.0)],
    "VonMises": [(0.0, 1.0, 0.0), (1.0, 5.0, 0.0), (-2.0, 0.3, 0.0)],
    "ContinuousBernoulli": [(0.0, 0.0, 0.0), (1.5, 0.0, 0.0), (-2.0, 0.0, 0.0)],
    "Bernoulli": [(0.0, 0.0, 0.0), (1.2, 0.0, 0.0), (-0.8, 0.0, 0.0)],
    "Geometric": [(0.0, 0.0, 0.0), (-1.5, 0.0, 0.0)],
    "Poisson": [(1.0, 0.0, 0.0), (4.5, 0.0, 0.0), (12.0, 0.0, 0.0)],
    "Binomial": [(10.0, 0.0, 0.0), (25.0, -1.0, 0.0)],
    "NegativeBinomial": [(3.0, -0.5, 0.0), (8.0, -1.5, 0.0)],
    "LogitRelaxedBernoulli": [(0.5, 0.0, 0.0), (1.5, 1.0, 0.0)],
    "RelaxedBernoulli": [(0.5, 0.0, 0.0), (1.2, -1.0, 0.0)],
}


def value_points(support, a, b):
    if support == "bool":
        return [0.0, 1.0]
    if support == "bounded":
        n = int(round(a))
        return [0.0, 1.0, float(n // 2), float(n)]
    if support == "counting":
        return [0.0, 1.0, 3.0, 7.0]
    if support == "real":
        return [-2.0, -0.3, 0.4, 1.7]
    if support == "positive":
        return [0.15, 0.8, 2.0, 5.0]
    if support == "above_a":
        return [a * 1.01, a * 1.5, a * 3.0, a * 9.0]
    if support in ("unit", "unit_closed"):
        return [0.05, 0.3, 0.62, 0.95]
    if support == "interval":
        return [a + 0.01 * (b - a), a + 0.5 * (b - a), a + 0.9 * (b - a)]
    if support == "circle":
        return [-3.0, -0.7, 0.4, 2.9]
    raise ValueError(support)


#: Dirichlet cases: (concentration, value on the simplex).
DIRICHLET_CASES = [
    ([1.0, 1.0, 1.0], [0.2, 0.3, 0.5]),
    ([2.0, 3.0, 4.0], [0.15, 0.25, 0.60]),
    ([0.5, 0.5], [0.3, 0.7]),
    ([7.0, 1.5, 0.8, 3.0], [0.4, 0.1, 0.05, 0.45]),
]


def dirichlet_reference(alpha, x):
    """Log-density, entropy, mean and per-class variance of a Dirichlet."""
    total = sum(alpha)
    log_prob = math.lgamma(total) + sum(
        (a - 1.0) * math.log(v) - math.lgamma(a) for a, v in zip(alpha, x)
    )
    k = len(alpha)
    entropy = (
        sum(math.lgamma(a) for a in alpha)
        - math.lgamma(total)
        + (total - k) * digamma(total)
        - sum((a - 1.0) * digamma(a) for a in alpha)
    )
    mean = [a / total for a in alpha]
    var = [a * (total - a) / (total * total * (total + 1.0)) for a in alpha]
    return log_prob, entropy, mean, var


def emit_dirichlet(out):
    out += [
        "",
        "/// `(concentration, value, log_prob, entropy, mean, variance)` for the",
        "/// Dirichlet, from the closed forms rather than from quadrature — a density",
        "/// on a simplex is not something a one-dimensional rule can integrate.",
        "#[allow(clippy::approx_constant, clippy::excessive_precision)]",
        "pub const DIRICHLET_GOLDEN: &[DirichletRow] = &[",
    ]
    for alpha, x in DIRICHLET_CASES:
        lp, ent, mean, var = dirichlet_reference(alpha, x)
        out.append(
            f"    (&{alpha!r}, &{x!r}, {lp!r}, {ent!r}, &{mean!r}, &{var!r}),".replace(
                "[", "["
            )
        )
    out += ["];", ""]
    return out


def main():
    rows = []
    summaries = []
    worst_mass = []
    for name, code, arity, support, logpdf in FAMILIES:
        for a, b, c in CASES[name]:
            mass, mean, var, ent = summarise(logpdf, support, a, b, c)
            worst_mass.append((abs(mass - 1.0), name, (a, b, c), mass))
            for x in value_points(support, a, b):
                lp = logpdf(x, a, b, c)
                cdf = cdf_at(logpdf, support, a, b, c, x)
                rows.append((code, name, a, b, c, x, lp, cdf))
            summaries.append((code, name, a, b, c, mean, var, ent))

    worst_mass.sort(reverse=True)
    for err, name, params, mass in worst_mass[:6]:
        print(f"  mass check: {name}{params} = {mass:.12f} (off by {err:.2e})")
    assert worst_mass[0][0] < 1e-5, f"{worst_mass[0][1]} integrates to {worst_mass[0][3]}"

    out = [
        "// Generated by tests/golden/distributions.py. Do not edit.",
        "",
        "/// One row of [`DIST_GOLDEN`] or [`DIST_SUMMARY_GOLDEN`]: the kind, its",
        "/// name, its three parameter slots and three numbers whose meaning the",
        "/// table's own documentation gives.",
        "pub type GoldenRow = (u32, &'static str, f64, f64, f64, f64, f64, f64);",
        "",
        "/// One row of [`DIRICHLET_GOLDEN`]: concentration, value, log-density,",
        "/// entropy, mean and per-class variance.",
        "pub type DirichletRow = (",
        "    &'static [f64],",
        "    &'static [f64],",
        "    f64,",
        "    f64,",
        "    &'static [f64],",
        "    &'static [f64],",
        ");",
        "",
        "/// `(kind, name, a, b, c, value, log_prob, cdf)`.",
        "///",
        "/// The log-density is an independent transcription of the family's formula;",
        "/// the CDF is that density integrated numerically. Both are `f64` and were",
        "/// checked by requiring the density to integrate to one over its support.",
        "#[allow(clippy::approx_constant, clippy::excessive_precision)]",
        "pub const DIST_GOLDEN: &[GoldenRow] = &[",
    ]
    for code, name, a, b, c, x, lp, cdf in rows:
        out.append(
            f'    ({code}, "{name}", {a!r}, {b!r}, {c!r}, {x!r}, {lp!r}, {cdf!r}),'
        )
    out += [
        "];",
        "",
        "/// `(kind, name, a, b, c, mean, variance, entropy)`, all from the same",
        "/// quadrature as the CDFs above rather than from a second closed form.",
        "#[allow(clippy::approx_constant, clippy::excessive_precision)]",
        "pub const DIST_SUMMARY_GOLDEN: &[GoldenRow] = &[",
    ]
    for code, name, a, b, c, mean, var, ent in summaries:
        out.append(
            f'    ({code}, "{name}", {a!r}, {b!r}, {c!r}, {mean!r}, {var!r}, {ent!r}),'
        )
    out += ["];"]
    out = emit_dirichlet(out)
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "distributions.rs")
    with open(path, "w") as handle:
        handle.write("\n".join(out))
    print(f"wrote {path}: {len(rows)} density points, {len(summaries)} summaries")


if __name__ == "__main__":
    main()
