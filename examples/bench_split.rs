//! One split, fused against a band per launch, at the shapes a rollout step uses.
//!
//! `profile_rollout` says the projection splits are the largest block of pure
//! dispatch in a policy step, and fusing them removes twelve launches of the
//! seventy-five. Whether that is *worth* anything is a separate question: the fused
//! kernel binds seven buffers where a slice binds two, and on a backend whose
//! per-launch cost is mostly bind-group construction those are not the same trade.
//!
//! A PPO round is too noisy to answer it — the collect phase swings 35-60 ms
//! run to run for identical work — so this measures the operation alone, with the
//! two paths interleaved inside one process and the minimum taken over the
//! repetitions. Interleaving is what makes the comparison survive a laptop GPU's
//! thermal behaviour; the minimum is what makes it survive everything else running
//! on the machine.
//!
//! ```text
//! cargo run --release --features wgpu --example bench_split
//! ```

use std::time::{Duration, Instant};

use mamba3::backend::{launch_count, reset_launch_count};
use mamba3::prelude::*;
use mamba3::tensor::ops::movement;

type R = mamba3::backends::Auto;

/// Repetitions inside one timed sample. Enough that the sample is milliseconds
/// rather than microseconds, so the clock is not what is being measured.
const INNER: usize = 200;
/// Timed samples per path. The minimum of these is the reported number.
const SAMPLES: usize = 30;

/// Time `INNER` splits of `input`, synchronising once at the end.
///
/// The synchronisation is outside the loop on purpose: a rollout never waits for a
/// split, so timing one that does would measure a drain the real loop does not pay.
/// What this charges is host submission, which is the cost under study.
fn sample(input: &Tensor<R, f32>, sizes: &[usize], device: &Device<R>) -> Duration {
    let started = Instant::now();
    for _ in 0..INNER {
        let bands = movement::split(input, sizes, 2).expect("shapes are consistent");
        std::hint::black_box(&bands);
    }
    device.synchronize();
    started.elapsed()
}

/// Launches one split issues on the path currently selected.
fn launches(input: &Tensor<R, f32>, sizes: &[usize]) -> usize {
    reset_launch_count();
    let bands = movement::split(input, sizes, 2).expect("shapes are consistent");
    std::hint::black_box(&bands);
    launch_count()
}

fn main() -> Result<()> {
    let device = Device::<R>::default();
    println!("backend: {}\n", device.name());

    // The two splits `Mamba3Mixer::project` performs, at `profile_ppo`'s shape:
    // 32 environments, one position, d_model 64 with 4 heads of 16.
    let envs = 32;
    let cases: &[(&str, usize, &[usize])] = &[
        ("projection 5-way", 268, &[64, 132, 4, 4, 64]),
        ("xBC 3-way", 132, &[64, 34, 34]),
    ];

    println!(
        "{:<18} {:>9} {:>11} {:>11} {:>9}",
        "split", "launches", "unfused", "fused", "change"
    );
    for (label, width, sizes) in cases {
        let input = Tensor::<R, f32>::zeros(vec![envs, 1, *width], &device);

        // Warm-up compiles both kernels and settles the allocator, so neither path
        // pays for the other's first run inside a timed sample.
        for _ in 0..5 {
            sample(&input, sizes, &device);
        }

        let mut unfused = Duration::MAX;
        let mut fused = Duration::MAX;
        for _ in 0..SAMPLES {
            movement::set_fused_split(false);
            unfused = unfused.min(sample(&input, sizes, &device));
            movement::set_fused_split(true);
            fused = fused.min(sample(&input, sizes, &device));
        }

        let each = |d: Duration| d.as_secs_f64() * 1e6 / INNER as f64;
        println!(
            "{label:<18} {:>4} -> {:<2} {:>9.1}us {:>9.1}us {:>8.1}%",
            {
                movement::set_fused_split(false);
                launches(&input, sizes)
            },
            {
                movement::set_fused_split(true);
                launches(&input, sizes)
            },
            each(unfused),
            each(fused),
            100.0 * (each(fused) - each(unfused)) / each(unfused),
        );
    }
    Ok(())
}
