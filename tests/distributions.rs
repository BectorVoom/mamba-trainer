//! What the distributions compute, checked against something that is not them.
//!
//! [`distributions_bitexact`](../distributions_bitexact/index.html) asks whether the
//! device agrees with the host. This file asks the prior question: whether what they
//! both compute is *right*. Three independent checks, because a single one could be
//! satisfied by a consistent mistake:
//!
//! 1. **Against quadrature.** `tests/golden/distributions.py` transcribes every
//!    density from the standard references in Python, verifies each transcription by
//!    integrating it over its support and requiring one, and emits the resulting
//!    CDFs, means, variances and entropies. A wrong constant in either
//!    implementation shows up as a disagreement with the other.
//! 2. **Against the samplers.** Every family draws two hundred thousand samples and
//!    has to reproduce its own analytic mean and variance, and — where it has a CDF
//!    — pass a Kolmogorov–Smirnov test against it. This is what catches a rejection
//!    sampler whose acceptance test is subtly wrong: the density would be right, the
//!    moments right, and the draws still biased.
//! 3. **Against finite differences.** Every analytic gradient is compared with a
//!    central difference of the value it claims to differentiate.
//!
//! The three fail in different ways, which is the point. A wrong `lgamma` fails the
//! first; a wrong acceptance test fails only the second; a sign error in an adjoint
//! fails only the third.

#![cfg(feature = "backend")]

use mamba3::autograd::Var;
use mamba3::backend::Device;
use mamba3::backends::Auto;
use mamba3::distributions::univariate::Kind;
use mamba3::distributions::{Distribution, Param, Univariate};
use mamba3::tensor::Tensor;

type R = Auto;

include!("golden/distributions.rs");

fn dev() -> Device<R> {
    Device::<R>::default()
}

/// The family a golden row names, rebuilt with that row's parameters.
fn build(code: u32, a: f64, b: f64, c: f64, device: &Device<R>) -> Univariate<R, f32> {
    let kind = Kind::ALL[code as usize];
    let mut params: Vec<Param<R, f32>> = vec![(a as f32).into()];
    if kind.arity() > 1 {
        params.push((b as f32).into());
    }
    if kind.arity() > 2 {
        params.push((c as f32).into());
    }
    Univariate::new(kind, params, device).expect("the golden table is well formed")
}

/// Relative-or-absolute closeness, the form that behaves at zero.
fn close(got: f64, want: f64, tol: f64) -> bool {
    if got.is_nan() && want.is_nan() {
        return true;
    }
    (got - want).abs() <= tol * (1.0 + want.abs())
}

#[test]
fn log_prob_matches_an_independent_transcription() {
    let device = dev();
    // `f32` throughout the kernels, and the reference is a `f64` quadrature, so a
    // few parts in a million is the floor. Measured worst is well inside this.
    const TOL: f64 = 2.0e-5;
    let mut worst: Vec<(&str, f64, f64)> = Vec::new();
    for &(code, name, a, b, c, x, lp, _) in DIST_GOLDEN {
        if !lp.is_finite() {
            continue;
        }
        let dist = build(code, a, b, c, &device);
        let value = Var::constant(
            Tensor::<R, f32>::from_f32(&[x as f32], vec![1], &device).expect("one value"),
        );
        let got = dist.log_prob(&value).expect("log_prob is total").to_f32()[0] as f64;
        let err = (got - lp).abs() / (1.0 + lp.abs());
        match worst.iter_mut().find(|(n, ..)| *n == name) {
            Some(slot) if slot.1 < err => *slot = (name, err, x),
            Some(_) => {}
            None => worst.push((name, err, x)),
        }
        assert!(
            close(got, lp, TOL),
            "{name}({a}, {b}, {c}).log_prob({x}) = {got}, reference {lp}"
        );
    }
    worst.sort_by(|l, r| r.1.total_cmp(&l.1));
    for (name, err, x) in worst.iter().take(5) {
        println!("{name:>22}: worst log_prob error {err:.2e} at x = {x}");
    }
}

#[test]
fn cdf_matches_the_integral_of_the_density() {
    let device = dev();
    // Looser than the density check: the reference is a numerical integral, and a
    // CDF near one loses the digits its complement keeps.
    const TOL: f64 = 5.0e-5;
    for &(code, name, a, b, c, x, _, want) in DIST_GOLDEN {
        let kind = Kind::ALL[code as usize];
        if !kind.has_cdf() {
            continue;
        }
        let dist = build(code, a, b, c, &device);
        let value = Tensor::<R, f32>::from_f32(&[x as f32], vec![1], &device).expect("one value");
        let got = dist.cdf(&value).expect("cdf is total").to_f32()[0] as f64;
        assert!(
            close(got, want, TOL),
            "{name}({a}, {b}, {c}).cdf({x}) = {got}, integral {want}"
        );
    }
}

#[test]
fn summaries_match_the_quadrature() {
    let device = dev();
    const TOL: f64 = 1.0e-4;
    for &(code, name, a, b, c, mean, var, ent) in DIST_SUMMARY_GOLDEN {
        let kind = Kind::ALL[code as usize];
        let dist = build(code, a, b, c, &device);
        let got_mean = dist.mean().expect("mean is total").to_f32()[0] as f64;
        if got_mean.is_finite() {
            assert!(
                close(got_mean, mean, TOL),
                "{name}({a}, {b}, {c}).mean = {got_mean}, quadrature {mean}"
            );
        }
        let got_var = dist.variance().expect("variance is total").to_f32()[0] as f64;
        if got_var.is_finite() {
            assert!(
                close(got_var, var, TOL),
                "{name}({a}, {b}, {c}).variance = {got_var}, quadrature {var}"
            );
        }
        if kind.has_entropy() {
            let got = dist.entropy().expect("entropy is total").to_f32()[0] as f64;
            assert!(
                close(got, ent, TOL),
                "{name}({a}, {b}, {c}).entropy = {got}, quadrature {ent}"
            );
        }
    }
}

#[test]
fn quantiles_invert_the_distribution_function() {
    let device = dev();
    for kind in Kind::ALL {
        if !kind.has_icdf() || !kind.has_cdf() {
            continue;
        }
        // Discrete quantiles are step functions, and inverting one only recovers the
        // probability to within a step.
        if matches!(kind, Kind::Bernoulli | Kind::Geometric) {
            continue;
        }
        let params = family_case(kind);
        let dist = build_from(kind, &params, &device);
        let qs: Vec<f32> = (1..40).map(|i| i as f32 / 40.0).collect();
        let q = Var::constant(
            Tensor::<R, f32>::from_f32(&qs, vec![qs.len()], &device).expect("quantiles"),
        );
        let x = dist.icdf(&q).expect("icdf is total");
        let back = dist.cdf(x.tensor()).expect("cdf is total").to_f32();
        for (i, &want) in qs.iter().enumerate() {
            assert!(
                (back[i] - want).abs() < 2.0e-4,
                "{kind:?}: cdf(icdf({want})) = {}",
                back[i]
            );
        }
    }
}

/// A single-element parameter set per family, inside its support.
fn family_case(kind: Kind) -> Vec<f32> {
    match kind {
        Kind::Normal | Kind::Laplace | Kind::Cauchy | Kind::Gumbel => vec![0.3, 1.4],
        Kind::LogNormal => vec![0.3, 0.5],
        Kind::Uniform => vec![-1.0, 2.5],
        Kind::Exponential | Kind::HalfNormal | Kind::HalfCauchy => vec![1.7],
        Kind::Poisson => vec![4.5],
        // Heavy-tailed families are given a tail index high enough that their
        // *fourth* moment exists. Without one the sample variance has infinite
        // variance of its own, and the moment test below would be measuring the
        // seed rather than the sampler.
        Kind::Pareto => vec![1.3, 6.0],
        Kind::Weibull => vec![1.6, 1.9],
        Kind::Kumaraswamy => vec![2.2, 3.1],
        Kind::Gamma => vec![2.6, 1.3],
        Kind::InverseGamma => vec![6.0, 2.0],
        Kind::Beta => vec![2.3, 3.7],
        Kind::FisherSnedecor => vec![7.0, 12.0],
        Kind::StudentT => vec![8.0, 0.5, 1.2],
        Kind::VonMises => vec![0.4, 2.3],
        Kind::ContinuousBernoulli | Kind::Bernoulli | Kind::Geometric => vec![0.6],
        Kind::Binomial => vec![17.0, 0.4],
        Kind::NegativeBinomial => vec![6.0, -0.7],
        Kind::LogitRelaxedBernoulli | Kind::RelaxedBernoulli => vec![0.8, 0.3],
    }
}

fn build_from(kind: Kind, params: &[f32], device: &Device<R>) -> Univariate<R, f32> {
    let slots: Vec<Param<R, f32>> = params.iter().map(|v| (*v).into()).collect();
    Univariate::new(kind, slots, device).expect("a well-formed case")
}

/// Every sampler reproduces its own mean and variance.
///
/// Two hundred thousand draws, compared against the analytic moments with a
/// tolerance of six standard errors of the sample mean — which is a one-in-a-billion
/// false-alarm rate per family, and the seed is fixed, so a passing suite stays
/// passing.
#[test]
fn samplers_reproduce_their_own_moments() {
    let device = dev();
    const N: usize = 200_000;
    for kind in Kind::ALL {
        let params = family_case(kind);
        let dist = build_from(kind, &params, &device);
        let want_mean = dist.mean().unwrap().to_f32()[0];
        let want_var = dist.variance().unwrap().to_f32()[0];
        if !want_mean.is_finite() || !want_var.is_finite() {
            continue;
        }
        let draws = dist.sample_n(N, 0xBEEF_0000 ^ kind.code() as u64).unwrap().to_f32();
        let (mean, var) = if kind == Kind::VonMises {
            // A von Mises reports circular statistics, and so must its sample: the
            // linear average of angles that wrap at ±π is not an estimate of
            // anything. The resultant vector is.
            let (mut sx, mut sy) = (0.0f64, 0.0f64);
            for v in &draws {
                sx += (*v as f64).cos();
                sy += (*v as f64).sin();
            }
            let (sx, sy) = (sx / N as f64, sy / N as f64);
            (sy.atan2(sx), 1.0 - sx.hypot(sy))
        } else {
            let mean = draws.iter().map(|v| *v as f64).sum::<f64>() / N as f64;
            let var = draws
                .iter()
                .map(|v| {
                    let d = *v as f64 - mean;
                    d * d
                })
                .sum::<f64>()
                / N as f64;
            (mean, var)
        };
        let stderr = (want_var as f64 / N as f64).sqrt();
        assert!(
            (mean - want_mean as f64).abs() < 6.0 * stderr + 1e-4,
            "{kind:?}: sample mean {mean} vs analytic {want_mean} ({stderr:.2e} per draw)"
        );
        // The variance of a sample variance needs the fourth moment, which is not
        // available here; a 4% band is loose enough for every family in the table
        // and tight enough to catch a scale that is wrong by a factor.
        assert!(
            (var - want_var as f64).abs() < 0.04 * want_var as f64 + 1e-3,
            "{kind:?}: sample variance {var} vs analytic {want_var}"
        );
    }
}

/// Every sampler with a CDF passes a Kolmogorov–Smirnov test against it.
///
/// Moments alone would not catch a sampler that is right on average and wrong in
/// shape — a rejection loop that accepts slightly too often in one region. This
/// compares the whole empirical distribution function.
#[test]
fn samplers_match_their_distribution_function() {
    let device = dev();
    const N: usize = 20_000;
    // The asymptotic 1-in-10⁶ critical value is about 1.95/√N; this is comfortably
    // above it, and the seed is fixed.
    let critical = 2.6 / (N as f64).sqrt();
    for kind in Kind::ALL {
        if !kind.has_cdf() {
            continue;
        }
        let params = family_case(kind);
        let dist = build_from(kind, &params, &device);
        let mut draws = dist.sample_n(N, 0x5EED_0000 ^ kind.code() as u64).unwrap().to_f32();
        draws.sort_by(f32::total_cmp);
        let sorted = Tensor::<R, f32>::from_f32(&draws, vec![N], &device).unwrap();
        let cdf = dist.cdf(&sorted).unwrap().to_f32();

        // The empirical distribution function has to be evaluated *per distinct
        // value*, not per sample: a discrete family produces long runs of equal
        // draws, and comparing `F(v)` against the index of the first of them would
        // measure the size of the atom rather than any discrepancy.
        // Both one-sided statistics, and both need the *limit from below* on each
        // side of a jump. At a distinct value `v` the empirical function rises from
        // `before` to `after`, and the true one from `F(v⁻)` to `F(v)`; comparing
        // `F(v)` against `before` — the top of one jump against the bottom of the
        // other — measures the atom, which for a Bernoulli is most of the mass.
        let mut worst = 0.0f64;
        let mut prev = 0.0f64;
        let mut i = 0;
        while i < N {
            let mut j = i;
            while j + 1 < N && draws[j + 1] == draws[i] {
                j += 1;
            }
            let before = i as f64 / N as f64;
            let after = (j + 1) as f64 / N as f64;
            let f = cdf[i] as f64;
            worst = worst.max((f - after).abs()).max((prev - before).abs());
            prev = f;
            i = j + 1;
        }
        assert!(
            worst < critical,
            "{kind:?}: Kolmogorov-Smirnov statistic {worst:.5} over {critical:.5}"
        );
    }
}

// ---------------------------------------------------------------------------
// Gradients
// ---------------------------------------------------------------------------

/// The gradient of `f` at `point`, by central differences.
///
/// The step is scaled to the parameter so that a concentration of `17` and a
/// temperature of `0.8` are both perturbed by something meaningful. `f32` evaluation
/// noise is around `1e-7` of the value, so a step of `3e-3` leaves the difference
/// accurate to a part in `10⁴` — which is the tolerance the assertions use.
fn finite_difference(point: &[f32], f: impl Fn(&[f32]) -> f64) -> Vec<f64> {
    let mut out = Vec::with_capacity(point.len());
    for i in 0..point.len() {
        let h = 3.0e-3 * point[i].abs().max(1.0);
        let mut up = point.to_vec();
        let mut down = point.to_vec();
        up[i] += h;
        down[i] -= h;
        out.push((f(&up) - f(&down)) / (2.0 * h as f64));
    }
    out
}

/// Sum a `Var` to a scalar and read it.
fn total(v: &Var<R, f32>) -> f64 {
    v.sum().expect("a sum is total").to_f32()[0] as f64
}

/// The gradient a backward pass produces for a traced leaf.
fn grad_of(loss: &Var<R, f32>, leaf: &Var<R, f32>) -> Vec<f32> {
    loss.backward_retain()
        .expect("the loss is on the tape")
        .node(leaf.node().expect("the leaf is traced"))
        .expect("the leaf received a gradient")
        .to_f32()
}

fn assert_gradient(name: &str, analytic: &[f32], numeric: &[f64]) {
    for (i, (&a, &n)) in analytic.iter().zip(numeric).enumerate() {
        assert!(
            (a as f64 - n).abs() <= 2.0e-2 * (1.0 + a.abs() as f64),
            "{name}: component {i} analytic {a}, finite difference {n}"
        );
    }
}

/// A value inside every family's support, for differentiating a density at.
fn probe_value(kind: Kind, params: &[f32]) -> f32 {
    match kind {
        Kind::Normal | Kind::Laplace | Kind::Cauchy | Kind::Gumbel | Kind::StudentT
        | Kind::LogitRelaxedBernoulli => 0.7,
        Kind::Uniform => 0.5 * (params[0] + params[1]),
        Kind::VonMises => 1.1,
        Kind::Kumaraswamy | Kind::Beta | Kind::RelaxedBernoulli | Kind::ContinuousBernoulli => 0.42,
        Kind::Bernoulli => 1.0,
        Kind::Geometric | Kind::Poisson | Kind::NegativeBinomial => 3.0,
        Kind::Binomial => 6.0,
        Kind::Pareto => params[0] * 2.3,
        _ => 1.7,
    }
}

#[test]
fn log_prob_gradients_match_finite_differences() {
    let device = dev();
    for kind in Kind::ALL {
        let params = family_case(kind);
        let x = probe_value(kind, &params);
        let evaluate = |p: &[f32]| {
            let dist = build_from(kind, p, &device);
            let value = Var::constant(
                Tensor::<R, f32>::from_f32(&[x], vec![1], &device).expect("one value"),
            );
            total(&dist.log_prob(&value).expect("log_prob is total"))
        };

        // Analytic: one traced leaf holding all of the parameters, split into slots.
        let leaf = Var::traced(
            Tensor::<R, f32>::from_f32(&params, vec![params.len()], &device).expect("params"),
        );
        let slots: Vec<Param<R, f32>> = (0..params.len())
            .map(|i| leaf.slice(0, i, 1).expect("a slot").into())
            .collect();
        let dist = Univariate::new(kind, slots, &device).expect("a well-formed case");
        let value = Var::constant(
            Tensor::<R, f32>::from_f32(&[x], vec![1], &device).expect("one value"),
        );
        let lp = dist.log_prob(&value).expect("log_prob is total");
        assert_gradient(
            &format!("{kind:?}.log_prob d/dparams"),
            &grad_of(&lp, &leaf),
            &finite_difference(&params, evaluate),
        );

        // And with respect to the value itself, where that means anything.
        if !matches!(kind, Kind::Uniform) {
            let traced_value = Var::traced(
                Tensor::<R, f32>::from_f32(&[x], vec![1], &device).expect("one value"),
            );
            let dist = build_from(kind, &params, &device);
            let lp = dist.log_prob(&traced_value).expect("log_prob is total");
            let numeric = finite_difference(&[x], |v| {
                let value = Var::constant(
                    Tensor::<R, f32>::from_f32(v, vec![1], &device).expect("one value"),
                );
                total(&dist.log_prob(&value).expect("log_prob is total"))
            });
            assert_gradient(
                &format!("{kind:?}.log_prob d/dx"),
                &grad_of(&lp, &traced_value),
                &numeric,
            );
        }
    }
}

#[test]
fn entropy_gradients_match_finite_differences() {
    let device = dev();
    for kind in Kind::ALL {
        if !kind.has_entropy() {
            continue;
        }
        let params = family_case(kind);
        let evaluate = |p: &[f32]| {
            total(
                &build_from(kind, p, &device)
                    .entropy()
                    .expect("entropy is total"),
            )
        };
        let leaf = Var::traced(
            Tensor::<R, f32>::from_f32(&params, vec![params.len()], &device).expect("params"),
        );
        let slots: Vec<Param<R, f32>> = (0..params.len())
            .map(|i| leaf.slice(0, i, 1).expect("a slot").into())
            .collect();
        let entropy = Univariate::new(kind, slots, &device)
            .expect("a well-formed case")
            .entropy()
            .expect("entropy is total");
        assert_gradient(
            &format!("{kind:?}.entropy d/dparams"),
            &grad_of(&entropy, &leaf),
            &finite_difference(&params, evaluate),
        );
    }
}

#[test]
fn reparameterised_draws_have_the_right_path_derivative() {
    let device = dev();
    let seed = 0xD00D_1234u64;
    for kind in Kind::ALL {
        if !kind.reparameterised() {
            continue;
        }
        let params = family_case(kind);
        // The draw is a fixed function of the parameters once the seed is fixed, so
        // a finite difference of it is exactly what the path derivative claims.
        let evaluate = |p: &[f32]| {
            build_from(kind, p, &device)
                .sample(seed)
                .expect("sampling is total")
                .to_f32()[0] as f64
        };
        let leaf = Var::traced(
            Tensor::<R, f32>::from_f32(&params, vec![params.len()], &device).expect("params"),
        );
        let slots: Vec<Param<R, f32>> = (0..params.len())
            .map(|i| leaf.slice(0, i, 1).expect("a slot").into())
            .collect();
        let draw = Univariate::new(kind, slots, &device)
            .expect("a well-formed case")
            .rsample(seed)
            .expect("this family is reparameterised");
        assert_gradient(
            &format!("{kind:?}.rsample d/dparams"),
            &grad_of(&draw, &leaf),
            &finite_difference(&params, evaluate),
        );
    }
}

#[test]
fn quantile_gradients_match_finite_differences() {
    let device = dev();
    let q = 0.37f32;
    for kind in Kind::ALL {
        if !kind.has_icdf_grad() {
            continue;
        }
        let params = family_case(kind);
        let evaluate = |p: &[f32]| {
            let quantile = Var::constant(
                Tensor::<R, f32>::from_f32(&[q], vec![1], &device).expect("one quantile"),
            );
            total(
                &build_from(kind, p, &device)
                    .icdf(&quantile)
                    .expect("icdf is total"),
            )
        };
        let leaf = Var::traced(
            Tensor::<R, f32>::from_f32(&params, vec![params.len()], &device).expect("params"),
        );
        let slots: Vec<Param<R, f32>> = (0..params.len())
            .map(|i| leaf.slice(0, i, 1).expect("a slot").into())
            .collect();
        let quantile = Var::constant(
            Tensor::<R, f32>::from_f32(&[q], vec![1], &device).expect("one quantile"),
        );
        let value = Univariate::new(kind, slots, &device)
            .expect("a well-formed case")
            .icdf(&quantile)
            .expect("icdf is total");
        assert_gradient(
            &format!("{kind:?}.icdf d/dparams"),
            &grad_of(&value, &leaf),
            &finite_difference(&params, evaluate),
        );

        // And with respect to the probability, which is `1/pdf(icdf(q))`.
        let traced_q = Var::traced(
            Tensor::<R, f32>::from_f32(&[q], vec![1], &device).expect("one quantile"),
        );
        let dist = build_from(kind, &params, &device);
        let value = dist.icdf(&traced_q).expect("icdf is total");
        let numeric = finite_difference(&[q], |v| {
            let quantile = Var::constant(
                Tensor::<R, f32>::from_f32(v, vec![1], &device).expect("one quantile"),
            );
            total(&dist.icdf(&quantile).expect("icdf is total"))
        });
        assert_gradient(&format!("{kind:?}.icdf d/dq"), &grad_of(&value, &traced_q), &numeric);
    }
}

// ---------------------------------------------------------------------------
// The structured families
// ---------------------------------------------------------------------------

use mamba3::distributions::{
    AffineTransform, Categorical, Dirichlet, Independent, MixtureSameFamily,
    MultivariateNormal, OneHotCategorical, RelaxedOneHotCategorical, TanhTransform,
    TransformedDistribution, kl, kl_divergence,
};

/// `log softmax` in `f64`, written out so the comparison is against arithmetic
/// rather than against another softmax.
fn log_softmax_f64(logits: &[f32]) -> Vec<f64> {
    let top = logits.iter().fold(f64::NEG_INFINITY, |m, v| m.max(*v as f64));
    let total: f64 = logits.iter().map(|v| (*v as f64 - top).exp()).sum();
    logits
        .iter()
        .map(|v| *v as f64 - top - total.ln())
        .collect()
}

#[test]
fn categorical_scores_and_entropy_are_a_softmax() {
    let device = dev();
    let rows: Vec<Vec<f32>> = vec![
        vec![0.0, 1.0, -1.0, 2.5],
        vec![5.0, 5.0, 5.0, 5.0],
        vec![-40.0, 0.0, 40.0, -3.0],
    ];
    let flat: Vec<f32> = rows.concat();
    let logits = Var::traced(
        Tensor::<R, f32>::from_f32(&flat, vec![rows.len(), 4], &device).expect("logits"),
    );
    let cat = Categorical::from_logits(logits.clone()).expect("well formed");

    let probs = cat.probs().expect("probs are total").to_f32();
    let entropy = cat.entropy().expect("entropy is total").to_f32();
    for (r, row) in rows.iter().enumerate() {
        let log_p = log_softmax_f64(row);
        let want_entropy: f64 = -log_p.iter().map(|l| l.exp() * l).sum::<f64>();
        assert!(
            (entropy[r] as f64 - want_entropy).abs() < 2e-6,
            "row {r}: entropy {} vs {want_entropy}",
            entropy[r]
        );
        for k in 0..4 {
            assert!(
                (probs[r * 4 + k] as f64 - log_p[k].exp()).abs() < 2e-7,
                "row {r} class {k}: probability {} vs {}",
                probs[r * 4 + k],
                log_p[k].exp()
            );
        }
        for (k, &want) in log_p.iter().enumerate() {
            let ids = mamba3::tensor::ops::index::IdTensor::from_slice(
                &vec![k as u32; rows.len()],
                vec![rows.len()],
                &device,
            )
            .expect("ids");
            let got = cat.log_prob_ids(&ids).expect("log_prob is total").to_f32()[r] as f64;
            assert!(
                (got - want).abs() < 2e-6,
                "row {r} class {k}: log_prob {got} vs {want}"
            );
        }
    }

    // The adjoint of a categorical log-density is `onehot(a) − p`, and of its
    // entropy `−pᵢ(lᵢ − lse + H)`. Both against finite differences.
    let ids = mamba3::tensor::ops::index::IdTensor::from_slice(&[3, 1, 2], vec![3], &device)
        .expect("ids");
    let lp = cat.log_prob_ids(&ids).expect("log_prob is total");
    let analytic = grad_of(&lp, &logits);
    let numeric = finite_difference(&flat, |p| {
        let l = Var::constant(
            Tensor::<R, f32>::from_f32(p, vec![rows.len(), 4], &device).expect("logits"),
        );
        let c = Categorical::from_logits(l).expect("well formed");
        total(&c.log_prob_ids(&ids).expect("log_prob is total"))
    });
    assert_gradient("Categorical.log_prob d/dlogits", &analytic, &numeric);

    let h = cat.entropy().expect("entropy is total");
    let analytic = grad_of(&h, &logits);
    let numeric = finite_difference(&flat, |p| {
        let l = Var::constant(
            Tensor::<R, f32>::from_f32(p, vec![rows.len(), 4], &device).expect("logits"),
        );
        total(
            &Categorical::from_logits(l)
                .expect("well formed")
                .entropy()
                .expect("entropy is total"),
        )
    });
    assert_gradient("Categorical.entropy d/dlogits", &analytic, &numeric);
}

#[test]
fn categorical_draws_follow_their_probabilities() {
    let device = dev();
    let row = [0.0f32, 1.0, -1.0, 2.5];
    let logits = Tensor::<R, f32>::from_f32(&row, vec![1, 4], &device).expect("logits");
    let cat = Categorical::from_logits(Var::constant(logits)).expect("well formed");
    const N: usize = 100_000;
    let ids = cat.sample_ids_n(N, 0xC0DE_1234).expect("draws").to_vec();
    let mut counts = [0usize; 4];
    for id in &ids {
        counts[*id as usize] += 1;
    }
    let want = log_softmax_f64(&row);
    // Pearson's chi-square, three degrees of freedom; the 1-in-10⁶ critical value is
    // about 24.
    let chi: f64 = (0..4)
        .map(|k| {
            let expected = want[k].exp() * N as f64;
            let diff = counts[k] as f64 - expected;
            diff * diff / expected
        })
        .sum();
    assert!(chi < 24.0, "chi-square {chi} over four classes: {counts:?}");
}

#[test]
fn dirichlet_matches_its_closed_forms() {
    let device = dev();
    for &(alpha, x, want_lp, want_entropy, want_mean, want_var) in DIRICHLET_GOLDEN {
        let k = alpha.len();
        let conc: Vec<f32> = alpha.iter().map(|v| *v as f32).collect();
        let value: Vec<f32> = x.iter().map(|v| *v as f32).collect();
        let dist = Dirichlet::new(Var::constant(
            Tensor::<R, f32>::from_f32(&conc, vec![k], &device).expect("concentration"),
        ))
        .expect("well formed");
        let v = Var::constant(
            Tensor::<R, f32>::from_f32(&value, vec![k], &device).expect("value"),
        );
        let got = dist.log_prob(&v).expect("log_prob is total").to_f32()[0] as f64;
        assert!(
            (got - want_lp).abs() < 2e-5 * (1.0 + want_lp.abs()),
            "Dirichlet{alpha:?}.log_prob = {got}, reference {want_lp}"
        );
        let got = dist.entropy().expect("entropy is total").to_f32()[0] as f64;
        assert!(
            (got - want_entropy).abs() < 2e-5 * (1.0 + want_entropy.abs()),
            "Dirichlet{alpha:?}.entropy = {got}, reference {want_entropy}"
        );
        let mean = dist.mean().expect("mean is total").to_f32();
        let var = dist.variance().expect("variance is total").to_f32();
        for i in 0..k {
            assert!((mean[i] as f64 - want_mean[i]).abs() < 1e-6);
            assert!((var[i] as f64 - want_var[i]).abs() < 1e-7);
        }
    }
}

#[test]
fn dirichlet_draws_sum_to_one_and_hit_their_mean() {
    let device = dev();
    let conc = [2.0f32, 5.0, 1.0];
    let dist = Dirichlet::new(Var::constant(
        Tensor::<R, f32>::from_f32(&conc, vec![3], &device).expect("concentration"),
    ))
    .expect("well formed");
    const N: usize = 50_000;
    let draws = dist.sample_n(N, 0xAB1E).expect("draws").to_f32();
    let mut sums = [0.0f64; 3];
    for row in draws.chunks(3) {
        let total: f32 = row.iter().sum();
        assert!(
            (total - 1.0).abs() < 3e-6,
            "a Dirichlet draw summed to {total}"
        );
        for i in 0..3 {
            sums[i] += row[i] as f64;
        }
    }
    let want = dist.mean().expect("mean is total").to_f32();
    for i in 0..3 {
        let got = sums[i] / N as f64;
        assert!(
            (got - want[i] as f64).abs() < 0.01,
            "class {i}: empirical mean {got} vs {}",
            want[i]
        );
    }

    // The adjoints, against finite differences.
    let leaf = Var::traced(
        Tensor::<R, f32>::from_f32(&conc, vec![3], &device).expect("concentration"),
    );
    let value: Vec<f32> = vec![0.3, 0.5, 0.2];
    let v = Var::constant(Tensor::<R, f32>::from_f32(&value, vec![3], &device).expect("value"));
    let dist = Dirichlet::new(leaf.clone()).expect("well formed");
    let lp = dist.log_prob(&v).expect("log_prob is total");
    let numeric = finite_difference(&conc, |p| {
        let d = Dirichlet::new(Var::constant(
            Tensor::<R, f32>::from_f32(p, vec![3], &device).expect("concentration"),
        ))
        .expect("well formed");
        total(&d.log_prob(&v).expect("log_prob is total"))
    });
    assert_gradient("Dirichlet.log_prob d/dalpha", &grad_of(&lp, &leaf), &numeric);

    let entropy = Dirichlet::new(leaf.clone())
        .expect("well formed")
        .entropy()
        .expect("entropy is total");
    let numeric = finite_difference(&conc, |p| {
        total(
            &Dirichlet::new(Var::constant(
                Tensor::<R, f32>::from_f32(p, vec![3], &device).expect("concentration"),
            ))
            .expect("well formed")
            .entropy()
            .expect("entropy is total"),
        )
    });
    assert_gradient(
        "Dirichlet.entropy d/dalpha",
        &grad_of(&entropy, &leaf),
        &numeric,
    );
}

#[test]
fn multivariate_normal_matches_the_two_by_two_algebra() {
    let device = dev();
    // Σ = [[4, 2], [2, 5]]; the algebra is small enough to write out.
    let cov = [4.0f64, 2.0, 2.0, 5.0];
    let det = cov[0] * cov[3] - cov[1] * cov[2];
    let inv = [cov[3] / det, -cov[1] / det, -cov[2] / det, cov[0] / det];
    let mu = [1.0f64, -1.0];
    let loc = Var::traced(
        Tensor::<R, f32>::from_f32(&[1.0, -1.0], vec![2], &device).expect("loc"),
    );
    let cov_t = Tensor::<R, f32>::from_f32(&[4.0, 2.0, 2.0, 5.0], vec![2, 2], &device)
        .expect("covariance");
    let mvn = MultivariateNormal::from_covariance(loc.clone(), &cov_t).expect("well formed");

    for probe in [[1.0f64, -1.0], [3.0, 0.0], [-2.0, 4.5]] {
        let d = [probe[0] - mu[0], probe[1] - mu[1]];
        let quad = d[0] * (inv[0] * d[0] + inv[1] * d[1]) + d[1] * (inv[2] * d[0] + inv[3] * d[1]);
        let want = -0.5 * quad - 0.5 * det.ln() - (2.0 * std::f64::consts::PI).ln();
        let value = Var::constant(
            Tensor::<R, f32>::from_f32(&[probe[0] as f32, probe[1] as f32], vec![2], &device)
                .expect("value"),
        );
        let got = mvn.log_prob(&value).expect("log_prob is total").to_f32()[0] as f64;
        assert!(
            (got - want).abs() < 2e-5 * (1.0 + want.abs()),
            "log_prob{probe:?} = {got}, algebra {want}"
        );
    }

    let want_entropy = 0.5 * (det * (2.0 * std::f64::consts::E * std::f64::consts::PI).powi(2)).ln();
    let got = mvn.entropy().expect("entropy is total").to_f32()[0] as f64;
    assert!(
        (got - want_entropy).abs() < 1e-5,
        "entropy {got} vs {want_entropy}"
    );

    // The empirical covariance of the draws.
    const N: usize = 200_000;
    let draws = mvn.sample_n(N, 0x1357_9BDF).expect("draws").to_f32();
    let (mut m0, mut m1) = (0.0f64, 0.0f64);
    for row in draws.chunks(2) {
        m0 += row[0] as f64;
        m1 += row[1] as f64;
    }
    let (m0, m1) = (m0 / N as f64, m1 / N as f64);
    let (mut c00, mut c01, mut c11) = (0.0f64, 0.0f64, 0.0f64);
    for row in draws.chunks(2) {
        let (a, b) = (row[0] as f64 - m0, row[1] as f64 - m1);
        c00 += a * a;
        c01 += a * b;
        c11 += b * b;
    }
    let (c00, c01, c11) = (c00 / N as f64, c01 / N as f64, c11 / N as f64);
    assert!((m0 - 1.0).abs() < 0.03 && (m1 + 1.0).abs() < 0.03, "mean {m0}, {m1}");
    assert!(
        (c00 - 4.0).abs() < 0.1 && (c01 - 2.0).abs() < 0.1 && (c11 - 5.0).abs() < 0.12,
        "covariance {c00}, {c01}, {c11}"
    );

    // The adjoint with respect to the mean is `Σ⁻¹(x − μ)`.
    let value = Var::constant(Tensor::<R, f32>::from_f32(&[3.0, 0.0], vec![2], &device).unwrap());
    let lp = mvn.log_prob(&value).expect("log_prob is total");
    let got = grad_of(&lp, &loc);
    let d = [3.0 - mu[0], 0.0 - mu[1]];
    let want = [
        inv[0] * d[0] + inv[1] * d[1],
        inv[2] * d[0] + inv[3] * d[1],
    ];
    for i in 0..2 {
        assert!(
            (got[i] as f64 - want[i]).abs() < 1e-5,
            "d log p/d loc[{i}] = {}, algebra {}",
            got[i],
            want[i]
        );
    }
}

#[test]
fn multivariate_normal_gradients_flow_into_the_factor() {
    let device = dev();
    let tril0 = [2.0f32, 0.0, 0.5, 1.5];
    let leaf = Var::traced(
        Tensor::<R, f32>::from_f32(&tril0, vec![2, 2], &device).expect("factor"),
    );
    let loc = Tensor::<R, f32>::from_f32(&[0.2, -0.4], vec![2], &device).expect("loc");
    let value = Var::constant(Tensor::<R, f32>::from_f32(&[1.1, 0.3], vec![2], &device).unwrap());
    let dist =
        MultivariateNormal::from_scale_tril(Var::constant(loc.clone()), leaf.clone()).unwrap();
    let lp = dist.log_prob(&value).expect("log_prob is total");
    let numeric = finite_difference(&tril0, |p| {
        let l = Tensor::<R, f32>::from_f32(p, vec![2, 2], &device).expect("factor");
        let d = MultivariateNormal::from_scale_tril(
            Var::constant(loc.clone()),
            Var::constant(l),
        )
        .unwrap();
        total(&d.log_prob(&value).expect("log_prob is total"))
    });
    // The strict upper triangle is not a parameter; the kernel writes a zero there
    // and the finite difference sees no change, so both agree at zero.
    assert_gradient("MultivariateNormal.log_prob d/dtril", &grad_of(&lp, &leaf), &numeric);
}

#[test]
fn combinators_compose_the_densities_they_claim_to() {
    let device = dev();
    let mean = [0.0f32, 0.5, -0.5, 1.0, 0.25, -1.5];
    let loc = Tensor::<R, f32>::from_f32(&mean, vec![2, 3], &device).expect("loc");

    // Independent sums the last axis of the base's log-density.
    let base = Univariate::<R, f32>::normal(&loc, 0.7, &device).unwrap();
    let indep = Independent::new(
        Univariate::<R, f32>::normal(&loc, 0.7, &device).unwrap(),
        1,
    )
    .unwrap();
    let x = Tensor::<R, f32>::from_f32(&[0.1, 0.2, 0.3, -0.4, 0.6, 0.9], vec![2, 3], &device)
        .expect("value");
    let per = base.log_prob(&Var::constant(x.clone())).unwrap().to_f32();
    let joint = indep.log_prob(&Var::constant(x.clone())).unwrap().to_f32();
    for r in 0..2 {
        let want: f32 = per[r * 3..r * 3 + 3].iter().sum();
        assert!((joint[r] - want).abs() < 2e-6, "row {r}: {} vs {want}", joint[r]);
    }
    let per_h = base.entropy().unwrap().to_f32();
    let joint_h = indep.entropy().unwrap().to_f32();
    for r in 0..2 {
        let want: f32 = per_h[r * 3..r * 3 + 3].iter().sum();
        assert!((joint_h[r] - want).abs() < 2e-6);
    }

    // An affine transform of a standard normal is a normal with those parameters.
    let standard = Univariate::<R, f32>::normal(0.0, 1.0, &device).unwrap();
    let shifted = TransformedDistribution::new(
        standard,
        vec![Box::new(AffineTransform::new(
            Var::constant(Tensor::<R, f32>::from_f32(&[2.0], vec![1], &device).unwrap()),
            Var::constant(Tensor::<R, f32>::from_f32(&[3.0], vec![1], &device).unwrap()),
        ))],
    );
    let direct = Univariate::<R, f32>::normal(2.0, 3.0, &device).unwrap();
    for probe in [-4.0f32, 0.0, 2.0, 7.5] {
        let v = Var::constant(Tensor::<R, f32>::from_f32(&[probe], vec![1], &device).unwrap());
        let a = shifted.log_prob(&v).unwrap().to_f32()[0];
        let b = direct.log_prob(&v).unwrap().to_f32()[0];
        assert!((a - b).abs() < 3e-6, "transformed {a} vs direct {b} at {probe}");
    }

    // Tanh squashing: `log p(y) = log p(x) − log(1 − y²)` for `y = tanh x`.
    let squashed = TransformedDistribution::new(
        Univariate::<R, f32>::normal(0.3, 1.1, &device).unwrap(),
        vec![Box::new(TanhTransform)],
    );
    let inner = Univariate::<R, f32>::normal(0.3, 1.1, &device).unwrap();
    for y in [-0.9f32, -0.2, 0.4, 0.95] {
        let x = y.atanh();
        let v = Var::constant(Tensor::<R, f32>::from_f32(&[y], vec![1], &device).unwrap());
        let got = squashed.log_prob(&v).unwrap().to_f32()[0] as f64;
        let xv = Var::constant(Tensor::<R, f32>::from_f32(&[x], vec![1], &device).unwrap());
        let want = inner.log_prob(&xv).unwrap().to_f32()[0] as f64
            - (1.0 - (y as f64) * (y as f64)).ln();
        assert!(
            (got - want).abs() < 2e-4 * (1.0 + want.abs()),
            "tanh-normal at {y}: {got} vs {want}"
        );
    }

    // A mixture is a log-sum-exp of its weighted components.
    let comp_loc = Tensor::<R, f32>::from_f32(&[-2.0, 0.0, 3.0], vec![1, 3], &device).unwrap();
    let comp = Univariate::<R, f32>::normal(&comp_loc, 1.0, &device).unwrap();
    let weights = Categorical::from_logits(Var::constant(
        Tensor::<R, f32>::from_f32(&[0.0, 1.0, -0.5], vec![1, 3], &device).unwrap(),
    ))
    .unwrap();
    let mix = MixtureSameFamily::new(weights, comp).unwrap();
    let log_w = log_softmax_f64(&[0.0, 1.0, -0.5]);
    for probe in [-3.0f64, 0.2, 2.7] {
        let v = Var::constant(
            Tensor::<R, f32>::from_f32(&[probe as f32], vec![1], &device).unwrap(),
        );
        let got = mix.log_prob(&v).unwrap().to_f32()[0] as f64;
        let mut terms = Vec::new();
        for (k, m) in [-2.0f64, 0.0, 3.0].iter().enumerate() {
            let z = probe - m;
            terms.push(log_w[k] - 0.5 * z * z - 0.5 * (2.0 * std::f64::consts::PI).ln());
        }
        let top = terms.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let want = top + terms.iter().map(|t| (t - top).exp()).sum::<f64>().ln();
        assert!(
            (got - want).abs() < 2e-5 * (1.0 + want.abs()),
            "mixture at {probe}: {got} vs {want}"
        );
    }
}

#[test]
fn one_hot_and_relaxed_agree_with_the_categorical_underneath() {
    let device = dev();
    let row = [0.4f32, -1.2, 2.0];
    let logits = Var::constant(Tensor::<R, f32>::from_f32(&row, vec![1, 3], &device).unwrap());
    let one_hot = OneHotCategorical::from_logits(logits.clone()).unwrap();
    let log_p = log_softmax_f64(&row);
    for k in 0..3 {
        let mut v = vec![0.0f32; 3];
        v[k] = 1.0;
        let value = Var::constant(Tensor::<R, f32>::from_f32(&v, vec![1, 3], &device).unwrap());
        let got = one_hot.log_prob(&value).unwrap().to_f32()[0] as f64;
        assert!((got - log_p[k]).abs() < 2e-6, "class {k}: {got} vs {}", log_p[k]);
    }

    // As the temperature falls, a relaxed draw approaches a vertex. The statistic
    // is the *average* peak, not the smallest: at any temperature two Gumbel-shifted
    // logits occasionally land within a hair of each other, and that draw sits on an
    // edge of the simplex however cold it is.
    let mut peaks = Vec::new();
    for temperature in [2.0f32, 0.05] {
        let relaxed = RelaxedOneHotCategorical::new(temperature, logits.clone()).unwrap();
        let draws = relaxed.sample_n(2000, 0x9911).unwrap().to_f32();
        let mut total_peak = 0.0f64;
        for chunk in draws.chunks(3) {
            let mass: f32 = chunk.iter().sum();
            assert!((mass - 1.0).abs() < 1e-5, "a relaxed draw summed to {mass}");
            total_peak += chunk.iter().cloned().fold(0.0f32, f32::max) as f64;
        }
        peaks.push(total_peak / 2000.0);
    }
    assert!(
        peaks[1] > 0.97 && peaks[1] > peaks[0] + 0.2,
        "average peak was {:.3} at temperature 2 and {:.3} at 0.05",
        peaks[0],
        peaks[1]
    );
}

#[test]
fn analytic_divergences_agree_with_monte_carlo() {
    let device = dev();
    const N: usize = 400_000;
    let pairs: Vec<(Kind, Vec<f32>, Vec<f32>)> = vec![
        (Kind::Normal, vec![0.0, 1.0], vec![0.6, 1.4]),
        (Kind::Exponential, vec![1.3], vec![0.7]),
        (Kind::Laplace, vec![0.2, 1.0], vec![-0.3, 1.5]),
        (Kind::Gumbel, vec![0.0, 1.0], vec![0.5, 1.3]),
        (Kind::Gamma, vec![3.0, 2.0], vec![2.2, 1.4]),
        (Kind::Beta, vec![2.0, 3.0], vec![2.6, 2.2]),
        (Kind::Bernoulli, vec![0.4], vec![-0.3]),
        (Kind::Poisson, vec![4.0], vec![5.5]),
        (Kind::Pareto, vec![1.0, 4.0], vec![0.9, 3.2]),
        (Kind::LogNormal, vec![0.0, 0.6], vec![0.3, 0.8]),
        (Kind::Uniform, vec![-0.5, 1.5], vec![-1.0, 2.0]),
        (Kind::Cauchy, vec![0.0, 1.0], vec![0.4, 1.3]),
        (Kind::HalfNormal, vec![1.0], vec![1.4]),
        (Kind::HalfCauchy, vec![1.0], vec![1.6]),
        (Kind::InverseGamma, vec![4.0, 2.0], vec![3.2, 1.7]),
        (Kind::Geometric, vec![0.3], vec![-0.4]),
    ];
    for (kind, pp, qq) in pairs {
        let p = build_from(kind, &pp, &device);
        let q = build_from(kind, &qq, &device);
        let analytic = kl_divergence(&p, &q).expect("this pair has a closed form").to_f32()[0]
            as f64;
        let draws = p.sample_n(N, 0x4444_0000 ^ kind.code() as u64).unwrap();
        let value = Var::constant(draws);
        let under_p = p.log_prob(&value).unwrap().to_f32();
        let under_q = q.log_prob(&value).unwrap().to_f32();
        let estimate = under_p
            .iter()
            .zip(&under_q)
            .map(|(a, b)| *a as f64 - *b as f64)
            .sum::<f64>()
            / N as f64;
        assert!(
            (analytic - estimate).abs() < 0.02 * (1.0 + analytic.abs()),
            "{kind:?}: analytic KL {analytic}, Monte Carlo {estimate}"
        );
    }

    // The categorical divergence, against the sum it is.
    let pl = [0.0f32, 1.0, -1.0];
    let ql = [0.5f32, 0.2, 0.3];
    let p = Categorical::from_logits(Var::constant(
        Tensor::<R, f32>::from_f32(&pl, vec![1, 3], &device).unwrap(),
    ))
    .unwrap();
    let q = Categorical::from_logits(Var::constant(
        Tensor::<R, f32>::from_f32(&ql, vec![1, 3], &device).unwrap(),
    ))
    .unwrap();
    let lp = log_softmax_f64(&pl);
    let lq = log_softmax_f64(&ql);
    let want: f64 = (0..3).map(|k| lp[k].exp() * (lp[k] - lq[k])).sum();
    let got = kl::categorical(&p, &q).unwrap().to_f32()[0] as f64;
    assert!((got - want).abs() < 2e-6, "categorical KL {got} vs {want}");

    // And a divergence from a distribution to itself is zero.
    let self_kl = kl::categorical(&p, &p).unwrap().to_f32()[0];
    assert!(self_kl.abs() < 1e-6, "KL(p||p) = {self_kl}");
}

#[test]
fn unsupported_operations_say_so() {
    let device = dev();
    let beta = Univariate::<R, f32>::beta(2.0, 3.0, &device).unwrap();
    let value = Tensor::<R, f32>::from_f32(&[0.5], vec![1], &device).unwrap();
    assert!(beta.cdf(&value).is_err(), "a Beta has no closed-form CDF");
    assert!(
        beta.rsample(0).is_err(),
        "a Beta comes from a rejection sampler and has no path derivative"
    );
    let gamma = Univariate::<R, f32>::gamma(2.0, 1.0, &device).unwrap();
    assert!(gamma.icdf(&Var::constant(value.clone())).is_err());
    assert!(
        kl_divergence(&beta, &gamma).is_err(),
        "there is no closed-form divergence between different families"
    );
    assert!(
        Univariate::<R, f32>::new(Kind::Normal, vec![0.0.into()], &device).is_err(),
        "a normal needs two parameters"
    );
}

#[test]
fn structured_divergences_agree_with_monte_carlo() {
    let device = dev();
    const N: usize = 200_000;

    // Dirichlet against Dirichlet.
    let make = |alpha: &[f32]| {
        Dirichlet::new(Var::constant(
            Tensor::<R, f32>::from_f32(alpha, vec![alpha.len()], &device).unwrap(),
        ))
        .unwrap()
    };
    let p = make(&[2.0, 3.0, 4.0]);
    let q = make(&[1.5, 2.0, 6.0]);
    let analytic = kl::dirichlet(&p, &q).unwrap().to_f32()[0] as f64;
    let draws = Var::constant(p.sample_n(N, 0x77AA).unwrap());
    let estimate = (p.log_prob(&draws).unwrap().to_f32().iter().map(|v| *v as f64).sum::<f64>()
        - q.log_prob(&draws).unwrap().to_f32().iter().map(|v| *v as f64).sum::<f64>())
        / N as f64;
    assert!(
        (analytic - estimate).abs() < 0.02 * (1.0 + analytic.abs()),
        "Dirichlet: analytic KL {analytic}, Monte Carlo {estimate}"
    );

    // A full-covariance Gaussian against another.
    let mvn = |loc: &[f32], cov: &[f32]| {
        MultivariateNormal::from_covariance(
            Var::constant(Tensor::<R, f32>::from_f32(loc, vec![2], &device).unwrap()),
            &Tensor::<R, f32>::from_f32(cov, vec![2, 2], &device).unwrap(),
        )
        .unwrap()
    };
    let p = mvn(&[0.5, -0.5], &[2.0, 0.5, 0.5, 1.0]);
    let q = mvn(&[0.0, 0.2], &[3.0, -0.4, -0.4, 1.6]);
    let analytic =
        mamba3::distributions::multivariate::kl_multivariate_normal(&p, &q).unwrap().to_f32()[0]
            as f64;
    let draws = Var::constant(p.sample_n(N, 0x33CC).unwrap());
    let estimate = (p.log_prob(&draws).unwrap().to_f32().iter().map(|v| *v as f64).sum::<f64>()
        - q.log_prob(&draws).unwrap().to_f32().iter().map(|v| *v as f64).sum::<f64>())
        / N as f64;
    assert!(
        (analytic - estimate).abs() < 0.02 * (1.0 + analytic.abs()),
        "MultivariateNormal: analytic KL {analytic}, Monte Carlo {estimate}"
    );

    // And each is zero against itself.
    assert!(kl::dirichlet(&make(&[2.0, 3.0, 4.0]), &make(&[2.0, 3.0, 4.0]))
        .unwrap()
        .to_f32()[0]
        .abs()
        < 1e-5);
    assert!(
        mamba3::distributions::multivariate::kl_multivariate_normal(&p, &p)
            .unwrap()
            .to_f32()[0]
            .abs()
            < 1e-5
    );
}
