//! Paired timing of old against chunked top-F selection (task T7 part 2).
//!
//! Times [`mamba3::tensor::ops::ms2::formula_top`] (the existing kernel, one
//! long serial lane per spectrum) against
//! [`mamba3::tensor::ops::ms2::formula_top_chunked_with`] (the chunked form,
//! `2 * F` short launches) in one process, interleaved: each round times `N`
//! back-to-back calls with one device drain at the end (a pipelined average)
//! for each form in turn, so the two see the same machine state. Shapes: `B`
//! in {1, 8, 16, 32}, `M` in {512, 2048}, `F` in {4, 8}. Prints ms per call
//! for both and the ratio (chunked / old).
//!
//! `--chunk C` selects the chunk size of the new form (default
//! [`mamba3::tensor::ops::ms2::FORMULA_TOP_CHUNK`]); the C measurement of
//! the task ran this bench with `--chunk 32`, `--chunk 64` and
//! `--chunk 128`.
//!
//! ```text
//! cargo run --release --no-default-features --features cpu --example bench_ms2_formula_top -- \
//!   [--n 10] [--rounds 5] [--chunk 64]
//! ```

use std::time::Instant;

use mamba3::backend::Device;
use mamba3::backends::Auto;
use mamba3::models::ms2::twin;
use mamba3::tensor::Tensor;
use mamba3::tensor::ops::index::IdTensor;
use mamba3::tensor::ops::ms2;
use mamba3::tensor::ops::random::Rng;

type R = Auto;
type E = f32;

fn usage() -> ! {
    eprintln!("usage: bench_ms2_formula_top [--n 10] [--rounds 5] [--chunk C]");
    std::process::exit(2);
}

/// Build one `(B, M)` input with many ties: quantised scores (halves in a
/// small range, so ties are dense), random flags 0/1/2, and poisoned scores
/// (NaN, infinities, out-of-domain extremes) every 11th slot. `cand` carries
/// the flag at +11 and the slot as the source id at +12.
fn build_input(b: usize, m: usize, seed: u64) -> (Vec<f32>, Vec<u32>) {
    let poisons = [
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        3.1e38,
        -3.1e38,
        3.0e38,
        -3.0e38,
        2.9e38,
        -2.9e38,
    ];
    let mut rng = Rng::seeded(seed);
    let u = rng.uniform_vec(b * m * 2, 0.0, 1.0);
    let mut log_prob = vec![0.0f32; b * m];
    let mut cand = vec![0u32; b * m * 13];
    for bi in 0..b {
        for slot in 0..m {
            let r0 = u[(bi * m + slot) * 2];
            let r1 = u[(bi * m + slot) * 2 + 1];
            let flag = (r0 * 3.0) as u32;
            cand[(bi * m + slot) * 13 + 11] = flag;
            cand[(bi * m + slot) * 13 + 12] = 5000 + slot as u32;
            log_prob[bi * m + slot] = if slot % 11 == 5 {
                poisons[slot % poisons.len()]
            } else {
                ((r1 * 8.0 - 4.0) * 2.0).round() / 2.0
            };
        }
    }
    (log_prob, cand)
}

fn main() {
    let mut n = 10usize;
    let mut rounds = 5usize;
    let mut chunk = ms2::FORMULA_TOP_CHUNK;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut next = || args.next().unwrap_or_else(|| usage());
        match arg.as_str() {
            "--n" => n = next().parse().unwrap_or_else(|_| usage()),
            "--rounds" => rounds = next().parse().unwrap_or_else(|_| usage()),
            "--chunk" => chunk = next().parse().unwrap_or_else(|_| usage()),
            _ => usage(),
        }
    }
    if n == 0 || rounds == 0 || chunk == 0 {
        usage();
    }
    let device = Device::<R>::default();
    println!(
        "bench_ms2_formula_top: backend={} n={n} rounds={rounds} chunk={chunk} (N back-to-back calls, one drain; interleaved old/new per round)",
        device.name()
    );
    println!(
        "{:>4} {:>5} {:>2} {:>10} {:>10} {:>7}  {}",
        "B", "M", "F", "old_ms", "new_ms", "ratio", "note"
    );
    for &b in &[1usize, 8, 16, 32] {
        for &m in &[512usize, 2048] {
            for &f in &[4usize, 8] {
                let (log_prob, cand) = build_input(b, m, 1000 + b as u64 * 31 + m as u64);
                let lp_t = Tensor::<R, E>::from_f32(&log_prob, vec![b, m], &device).unwrap();
                let cand_t = IdTensor::from_slice(&cand, vec![b, m, 13], &device).unwrap();
                let out_old = ms2::FormulaBuffers::<R, E>::new(b, m, f, &device);
                let out_new = ms2::FormulaBuffers::<R, E>::new(b, m, f, &device);
                // Warm both forms, then check they agree bit-for-bit before
                // timing (one host read each; not part of the timing).
                ms2::set_formula_top_chunked(false);
                ms2::formula_top(&lp_t, &cand_t, &out_old).unwrap();
                ms2::formula_top_chunked_with(&lp_t, &cand_t, &out_new, chunk).unwrap();
                device.synchronize();
                let (t_old, lp_old, c_old) = (
                    out_old.top.try_to_vec().unwrap(),
                    out_old.top_log_prob.try_to_f32().unwrap(),
                    out_old.top_count.try_to_vec().unwrap(),
                );
                let (t_new, lp_new, c_new) = (
                    out_new.top.try_to_vec().unwrap(),
                    out_new.top_log_prob.try_to_f32().unwrap(),
                    out_new.top_count.try_to_vec().unwrap(),
                );
                let (w_top, w_lp, w_count) = twin::formula_top_from_cand(&log_prob, &cand, b, m, f);
                let identical = t_old == t_new
                    && c_old == c_new
                    && lp_old
                        .iter()
                        .zip(lp_new.iter())
                        .all(|(a, c)| a.to_bits() == c.to_bits())
                    && t_old == w_top
                    && c_old == w_count
                    && lp_old
                        .iter()
                        .zip(w_lp.iter())
                        .all(|(a, c)| a.to_bits() == c.to_bits());
                let note = if identical {
                    "bit-identical"
                } else {
                    "MISMATCH"
                };
                // Interleaved rounds: each round times N old then N new (even
                // rounds) or N new then N old (odd rounds), one drain each.
                let mut old_ms = Vec::with_capacity(rounds);
                let mut new_ms = Vec::with_capacity(rounds);
                for r in 0..rounds {
                    for new_first in [r % 2 == 1] {
                        for is_new in [new_first, !new_first] {
                            if is_new {
                                let started = Instant::now();
                                for _ in 0..n {
                                    ms2::formula_top_chunked_with(&lp_t, &cand_t, &out_new, chunk)
                                        .unwrap();
                                }
                                device.synchronize();
                                new_ms.push(started.elapsed().as_secs_f64() * 1e3 / n as f64);
                            } else {
                                ms2::set_formula_top_chunked(false);
                                let started = Instant::now();
                                for _ in 0..n {
                                    ms2::formula_top(&lp_t, &cand_t, &out_old).unwrap();
                                }
                                device.synchronize();
                                old_ms.push(started.elapsed().as_secs_f64() * 1e3 / n as f64);
                            }
                        }
                    }
                }
                old_ms.sort_by(|a, b| a.total_cmp(b));
                new_ms.sort_by(|a, b| a.total_cmp(b));
                let old_med = old_ms[old_ms.len() / 2];
                let new_med = new_ms[new_ms.len() / 2];
                println!(
                    "{b:>4} {m:>5} {f:>2} {old_med:>10.3} {new_med:>10.3} {:>7.2}  {note}",
                    new_med / old_med.max(1e-9)
                );
            }
        }
    }
    ms2::clear_formula_top_chunked_override();
}
