//! The device side of the motif decoder's beam step, on a random model.
//!
//! Each step is what `examples/ms2_motif_decoder.rs` does per beam advance
//! with `--arch encoders`: reorder the state rows by the beam's parents
//! (`MotifModel::gather_host`), run one `MotifModel::step` and read the
//! logits back. Parents and tokens are drawn from a fixed seed, so two
//! binaries run exactly the same work; the printed checksum of the last
//! logits lets a candidate be compared against a base.
//!
//! ```text
//! cargo run --release --no-default-features --features wgpu --example bench_motif_beam -- \
//!     [rows 256] [steps 48] [rounds 5] [d_model 128] [layers 4]
//! ```

use mamba3::backend::Device;
use mamba3::backends::Auto;
use mamba3::models::ms2::completion_fingerprint::{FINGERPRINT_SLOTS, FingerprintBatch, SparseFingerprint};
use mamba3::models::ms2::completion_spectrum::{SPECTRUM_SLOTS, SpectrumBatch, SpectrumEvidence};
use mamba3::models::ms2::motif_model::{MotifModel, MotifModelConfig};
use mamba3::nn::Module;
use mamba3::tensor::ops::index::IdTensor;
use std::time::Instant;

type R = Auto;

fn main() -> mamba3::prelude::Result<()> {
    let args: Vec<usize> = std::env::args().skip(1).map(|a| a.parse().expect("numbers")).collect();
    let arg = |i: usize, default: usize| args.get(i).copied().unwrap_or(default);
    let (rows, steps, rounds, d_model, layers) = (arg(0, 256), arg(1, 48), arg(2, 5), arg(3, 128), arg(4, 4));
    let device = Device::<R>::default();
    let config = MotifModelConfig {
        n_out: 5717,
        d_model,
        layers,
        attention_heads: 4,
        d_state: 32,
        max_tokens: steps + 2,
        fingerprint_slots: FINGERPRINT_SLOTS,
        spectrum_slots: SPECTRUM_SLOTS,
        seed: 7,
    };
    let model: MotifModel<R, f32> = MotifModel::init(&config, &device)?;
    model.set_training(false);
    let _guard = mamba3::autograd::no_grad();
    let fingerprint = SparseFingerprint {
        entries: (0..100u16).map(|i| (i * 37, 0.5 + (i % 5) as f32 * 0.1)).collect(),
    };
    let spectrum = SpectrumEvidence {
        peaks: (0..40).map(|i| (40_000_000 + i * 3_100_000, 0.1 + (i % 7) as f32 * 0.1)).collect(),
        precursor_mz: 180_065_000,
        adduct: 1,
        neutral_mass: 179_057_724,
    };
    let fingerprints = FingerprintBatch::build(std::slice::from_ref(&fingerprint), FINGERPRINT_SLOTS)?;
    let spectra = SpectrumBatch::build(&[Some(&spectrum)], SPECTRUM_SLOTS)?;
    let conditioning = model.encode(&[[9, 8, 0, 4, 0, 0, 0, 0, 0, 0]], &fingerprints, Some(&spectra), &device)?;
    let n_out = config.n_out_padded();

    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    // The same draws for every round.
    let plan: Vec<(Vec<u32>, Vec<u32>)> = (0..steps)
        .map(|s| {
            let live = if s == 0 { 1 } else { rows };
            let mut parents: Vec<u32> = (0..rows).map(|_| (next() % live as u64) as u32).collect();
            parents.sort_unstable();
            let tokens = (0..rows).map(|_| (next() % config.n_out as u64) as u32).collect();
            (parents, tokens)
        })
        .collect();

    let mut checksum = 0.0f64;
    for round in 0..rounds {
        let mut state = model.start(&conditioning, &device)?;
        let start = IdTensor::from_slice(&[model.start_token()], vec![1], &device)?;
        model.step(&mut state, &start, &device)?.to_f32();
        let began = Instant::now();
        let mut logits = Vec::new();
        for (parents, tokens) in &plan {
            model.gather_host(&mut state, parents, &device)?;
            let ids = IdTensor::from_slice(tokens, vec![tokens.len()], &device)?;
            logits = model.step(&mut state, &ids, &device)?.to_f32();
        }
        let elapsed = began.elapsed().as_secs_f64();
        checksum = logits.iter().step_by(97).map(|&v| v as f64).sum();
        assert_eq!(logits.len(), rows * n_out);
        println!(
            "round {round}: {steps} steps x {rows} rows in {:.1} ms, {:.3} ms/step",
            elapsed * 1e3,
            elapsed * 1e3 / steps as f64
        );
    }
    println!("checksum {checksum:.6}");
    Ok(())
}
