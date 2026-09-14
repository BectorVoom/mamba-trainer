//! T1: an exponential moving average of the weights.
//!
//! The kernel against its host twin, then [`Ema`] and [`Trainer::with_ema`]
//! against a host recomputation of the documented recurrence.
//!
//! # CPU and GPU
//!
//! Every assertion that compares the device with a **host** recomputation is
//! exact on the CPU runtime, the reference backend. Elsewhere it is exact when
//! the device agrees bit for bit, and otherwise held to `|Δ| ≤ 4·ε·max(1, |x|)`
//! per element, printing a `skipped:` line that says so: the recurrence is only
//! `+ − ×`, so the realistic source of a difference is a shader compiler
//! contracting `e + c·(p − e)` into a fused multiply-add, which the budget covers.
//! Assertions that compare the device with **itself** (with and without the
//! average, two identical runs) are exact everywhere.

#![cfg(feature = "backend")]

use mamba3::backend::Device;
use mamba3::backends::Auto;
use mamba3::tensor::Tensor;
use mamba3::tensor::ops::fused::ema_step;

type R = Auto;

fn dev() -> Device<R> {
    Device::<R>::default()
}

/// The documented step, in the documented order.
fn ema_host(ema: &[f32], param: &[f32], one_minus_decay: f32) -> Vec<f32> {
    ema.iter()
        .zip(param)
        .map(|(&e, &p)| e + one_minus_decay * (p - e))
        .collect()
}

/// Deterministic values of mixed sign and magnitude, zeros of both signs among them.
fn values(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..n)
        .map(|i| {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            let bits = state.wrapping_mul(0x2545_f491_4f6c_dd1d);
            match i % 17 {
                0 => 0.0,
                1 => -0.0,
                _ => {
                    let unit = (bits >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
                    let scale = [1e-3f32, 0.1, 1.0, 30.0][(bits & 3) as usize];
                    unit * scale
                }
            }
        })
        .collect()
}

/// Values that differ from the host twin: none on the CPU runtime, and within
/// the budget elsewhere (see the module docs), or the test fails.
fn twin_differences(what: &str, got: &[f32], want: &[f32]) -> usize {
    assert_eq!(got.len(), want.len(), "{what}: lengths");
    let differ: Vec<usize> = (0..got.len())
        .filter(|&i| got[i].to_bits() != want[i].to_bits())
        .collect();
    if let Some(&first) = differ.first() {
        assert!(
            dev().name() != "cpu",
            "{what}: {} of {} values differ from the host twin on the CPU runtime, from index \
             {first}: {} against {}",
            differ.len(),
            got.len(),
            got[first],
            want[first]
        );
    }
    for &i in &differ {
        let budget = 4.0 * f32::EPSILON * want[i].abs().max(1.0);
        assert!(
            (got[i] - want[i]).abs() <= budget,
            "{what}: index {i} is {} against the twin's {}, over the budget {budget:e}",
            got[i],
            want[i]
        );
    }
    differ.len()
}

/// Tallies [`twin_differences`] over a test and reports once, as a skip of the
/// bit-exact claim, when any value was only within budget.
#[derive(Default)]
struct Twin {
    differ: usize,
    total: usize,
}

impl Twin {
    fn check(&mut self, what: &str, got: &[f32], want: &[f32]) {
        self.differ += twin_differences(what, got, want);
        self.total += got.len();
    }

    fn report(&self, test: &str) {
        if self.differ > 0 {
            println!(
                "skipped: {test} bit-exact on {} ({} of {} values differ from the host twin); \
                 held to 4·ε·max(1, |x|) instead",
                dev().name(),
                self.differ,
                self.total
            );
        }
    }
}

#[test]
fn ema_step_matches_its_host_twin() {
    let device = dev();
    // Lengths around every vector width a device offers (1 to 64) and their
    // tails, and one long enough for the CPU runtime to split across threads.
    let lengths = [
        0usize, 1, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 49, 63, 64, 65, 97, 193, 1000, 100_003,
    ];
    let mut twin = Twin::default();
    for (k, &n) in lengths.iter().enumerate() {
        let ema = values(n, 1 + k as u64);
        let param = values(n, 101 + k as u64);
        let e = Tensor::<R, f32>::from_f32(&ema, vec![n], &device).unwrap();
        let p = Tensor::<R, f32>::from_f32(&param, vec![n], &device).unwrap();
        for decay in [0.0f32, 0.5, 0.9, 0.999, 1.0] {
            let c = 1.0 - decay;
            let out = ema_step(&e, &p, c).unwrap();
            assert_eq!(out.shape().dims(), &[n]);
            twin.check(
                &format!("n={n}, decay={decay}"),
                &out.try_to_f32().unwrap(),
                &ema_host(&ema, &param, c),
            );
        }
        // Neither input is written.
        assert_eq!(
            e.try_to_f32().unwrap(),
            ema,
            "the average was written in place"
        );
        assert_eq!(
            p.try_to_f32().unwrap(),
            param,
            "the weight was written in place"
        );
    }

    twin.report("ema_step_matches_its_host_twin");

    // Shapes that disagree are refused rather than read out of bounds.
    let a = Tensor::<R, f32>::zeros(vec![2, 3], &device);
    let b = Tensor::<R, f32>::zeros(vec![3, 2], &device);
    assert!(ema_step(&a, &b, 0.1).is_err());
}
