//! What fusing a distribution buys, measured.
//!
//! A log-density written out of elementwise tensor operations — which is what a
//! straightforward port of `torch.distributions` does, and what PyTorch itself does
//! — is a chain of launches with a full-size intermediate between each pair. Written
//! as one kernel it is one launch and no intermediate. The arithmetic is the same, so
//! whatever the difference turns out to be is pure overhead that was there before.
//!
//! This also measures the two things a reinforcement-learning loop actually spends
//! its time on: scoring a batch of actions, and drawing one.
//!
//! ```text
//! cargo run --release --example bench_distributions
//! cargo run --release --no-default-features --features cuda --example bench_distributions
//! ```

use std::time::Instant;

use mamba3::autograd::Var;
use mamba3::backend::{launch_count, reset_launch_count};
use mamba3::distributions::{Categorical, Distribution, Univariate};
use mamba3::prelude::*;
use mamba3::tensor::ops::elemwise;

type R = mamba3::backends::Auto;

fn best(f: &dyn Fn(), device: &Device<R>) -> f64 {
    for _ in 0..3 {
        f();
    }
    device.synchronize();
    let mut best = f64::INFINITY;
    for _ in 0..20 {
        let t = Instant::now();
        f();
        device.synchronize();
        best = best.min(t.elapsed().as_secs_f64());
    }
    best
}

/// Time `f`, and count the launches one call of it costs.
fn measure(name: &str, n: usize, f: &dyn Fn(), device: &Device<R>) -> f64 {
    reset_launch_count();
    f();
    device.synchronize();
    let launches = launch_count();
    let t = best(f, device);
    println!(
        "  {name:<34} {:>8.3} ms  {:>7.1} M elem/s  {launches:>3} launch{}",
        t * 1e3,
        n as f64 / t / 1e6,
        if launches == 1 { "" } else { "es" }
    );
    t
}

/// A normal's log-density, composed the way a tensor library would compose it.
///
/// Eight operations, seven of which write a full-size intermediate. This is the
/// thing the fused kernel replaces.
fn composed_normal_log_prob(
    x: &Tensor<R, f32>,
    loc: &Tensor<R, f32>,
    scale: &Tensor<R, f32>,
) -> Result<Tensor<R, f32>> {
    let centred = elemwise::sub(x, loc)?;
    let z = elemwise::div(&centred, scale)?;
    let squared = elemwise::mul(&z, &z)?;
    let half = elemwise::mul_scalar(&squared, -0.5);
    let log_scale = elemwise::log(scale);
    let less_scale = elemwise::sub(&half, &log_scale)?;
    Ok(elemwise::add_scalar(&less_scale, -0.918_938_5))
}

fn main() -> Result<()> {
    let device = Device::<R>::default();
    println!("backend {}", device.name());

    // ---------------------------------------------------------------- fusion
    let n = 1 << 20;
    let x = Tensor::<R, f32>::zeros(vec![n], &device);
    let loc = Tensor::<R, f32>::zeros(vec![n], &device);
    let scale = Tensor::<R, f32>::ones(vec![n], &device);
    println!(
        "\nnormal log-density over {n} elements ({} MiB)",
        n * 4 / 1048576
    );
    let composed = measure(
        "composed from elementwise ops",
        n,
        &|| {
            let _ = composed_normal_log_prob(&x, &loc, &scale);
        },
        &device,
    );
    let batched = Univariate::<R, f32>::normal(&loc, &scale, &device)?;
    let value = Var::constant(x.clone());
    let fused = measure(
        "fused, tensor parameters",
        n,
        &|| {
            let _ = batched.log_prob(&value);
        },
        &device,
    );
    let scalar_params = Univariate::<R, f32>::normal(0.0, 1.0, &device)?;
    let value_wide = Var::constant(x.clone());
    let scalarised = measure(
        "fused, scalar parameters",
        n,
        &|| {
            let _ = scalar_params.log_prob(&value_wide);
        },
        &device,
    );
    println!(
        "  fusion is {:.2}x, and a scalar parameter a further {:.2}x",
        composed / fused,
        fused / scalarised
    );

    // ------------------------------------------------------------- gradients
    println!("\nthe same, with a gradient");
    let traced_loc = Var::traced(loc.clone());
    measure(
        "fused forward + backward",
        n,
        &|| {
            let d = Univariate::<R, f32>::normal(&traced_loc, 1.0, &device).unwrap();
            let lp = d.log_prob(&value).unwrap();
            let _ = lp.sum().unwrap().backward();
        },
        &device,
    );

    // -------------------------------------------------------------- sampling
    println!("\nsampling {n} elements");
    for (name, dist) in [
        (
            "normal (inverse CDF)",
            Univariate::<R, f32>::normal(0.0, 1.0, &device)?,
        ),
        (
            "exponential (inverse CDF)",
            Univariate::<R, f32>::exponential(1.0, &device)?,
        ),
        (
            "gamma (rejection)",
            Univariate::<R, f32>::gamma(2.5, 1.0, &device)?,
        ),
        (
            "beta (two rejections)",
            Univariate::<R, f32>::beta(2.0, 3.0, &device)?,
        ),
        (
            "poisson, rate 3 (inversion)",
            Univariate::<R, f32>::poisson(3.0, &device)?,
        ),
        (
            "poisson, rate 40 (PTRS)",
            Univariate::<R, f32>::poisson(40.0, &device)?,
        ),
        (
            "binomial, 50 trials (BTRS)",
            Univariate::<R, f32>::binomial_logits(50.0, 0.0, &device)?,
        ),
        (
            "von Mises (rejection)",
            Univariate::<R, f32>::von_mises(0.0, 4.0, &device)?,
        ),
    ] {
        measure(
            name,
            n,
            &|| {
                let _ = dist.sample_n(n, 7);
            },
            &device,
        );
    }

    // ------------------------------------------------------- a policy's step
    println!("\na categorical policy: 4096 environments over an action space");
    for actions in [4usize, 18, 256] {
        let logits = Tensor::<R, f32>::zeros(vec![4096, actions], &device);
        let policy = Categorical::from_logits(Var::constant(logits.clone()))?;
        let rows = 4096;
        println!("  {actions} actions");
        // What PPO's objective used to cost: a log-softmax, a gather and a
        // five-operation entropy, forward and backward.
        let traced = Var::traced(logits.clone());
        let ids = policy.sample_ids(3)?;
        measure(
            "    composed score + entropy (bwd)",
            rows,
            &|| {
                let log_p = traced.log_softmax(1).unwrap();
                let chosen = log_p.take_along_last(&ids).unwrap();
                let entropy = log_p
                    .exp()
                    .mul(&log_p)
                    .unwrap()
                    .sum_dim(1)
                    .unwrap()
                    .squeeze(1)
                    .unwrap()
                    .neg();
                let _ = chosen
                    .sum()
                    .unwrap()
                    .add(&entropy.sum().unwrap())
                    .unwrap()
                    .backward();
            },
            &device,
        );
        let fused_policy = Categorical::from_logits(traced.clone())?;
        measure(
            "    fused score + entropy (bwd)",
            rows,
            &|| {
                let chosen = fused_policy.log_prob_ids(&ids).unwrap();
                let entropy = fused_policy.entropy().unwrap();
                let _ = chosen
                    .sum()
                    .unwrap()
                    .add(&entropy.sum().unwrap())
                    .unwrap()
                    .backward();
            },
            &device,
        );
        measure(
            "    sample, then score",
            rows,
            &|| {
                let ids = policy.sample_ids(3).unwrap();
                let _ = policy.log_prob_ids(&ids).unwrap();
            },
            &device,
        );
        measure(
            "    sample and score together",
            rows,
            &|| {
                let _ = policy.sample_with_log_prob(1.0, 3).unwrap();
            },
            &device,
        );
        measure(
            "    entropy",
            rows,
            &|| {
                let _ = policy.entropy().unwrap();
            },
            &device,
        );
    }
    Ok(())
}
