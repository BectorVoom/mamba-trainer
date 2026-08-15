//! Incremental decoding with a recurrent state cache.
//!
//! The point this example makes: after the prompt is consumed, every new token
//! costs the same amount of work no matter how much context precedes it. There is
//! no growing key/value cache in a pure Mamba-3 stack — only a fixed-size state per
//! layer.
//!
//! ```text
//! cargo run --release --example generate
//! ```

use std::time::Instant;

use mamba3::prelude::*;
use mamba3::tensor::ops::index::IdTensor;

type R = mamba3::backends::Cpu;

fn main() -> Result<()> {
    let device = Device::<R>::default();

    let model = Mamba3LmConfig::builder()
        .vocab_size(64)
        .d_model(64)
        .n_layers(2)
        .with_ssm(|s| {
            s.n_heads = 4;
            s.n_groups = 4;
            s.head_dim = 16;
            s.d_state = 16;
            s.chunk_size = 32;
        })
        .seed(7)
        .build()?
        .init::<R, f32>(&device)?;

    model.eval();
    println!("{model:?}");

    // Show that the cached path reproduces the parallel path exactly.
    let sequence: Vec<u32> = (0..24).map(|i| (i * 7 % 64) as u32).collect();
    let tokens = IdTensor::from_slice(&sequence, vec![1, sequence.len()], &device)?;
    let parallel = model.forward(&tokens, false)?.to_f32();

    let mut cache = model.empty_cache(1, &device);
    let mut incremental = Vec::new();
    for token in &sequence {
        let step = IdTensor::from_slice(&[*token], vec![1, 1], &device)?;
        incremental.extend(model.forward_cached(&step, &mut cache)?.to_f32());
    }
    let drift = parallel
        .iter()
        .zip(&incremental)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("parallel vs incremental, max |difference|: {drift:.2e}");

    // Per-token cost at two very different context lengths.
    for prompt_len in [8usize, 512] {
        let prompt: Vec<u32> = (0..prompt_len).map(|i| (i % 64) as u32).collect();
        let mut cache = model.empty_cache(1, &device);
        let ids = IdTensor::from_slice(&prompt, vec![1, prompt_len], &device)?;
        model.forward_cached(&ids, &mut cache)?;
        device.synchronize();

        let start = Instant::now();
        let steps = 8;
        for i in 0..steps {
            let step = IdTensor::from_slice(&[(i % 64) as u32], vec![1, 1], &device)?;
            model.forward_cached(&step, &mut cache)?;
        }
        device.synchronize();
        println!(
            "context {prompt_len:>4} tokens -> {:>7.2} ms per decoded token",
            start.elapsed().as_secs_f64() * 1000.0 / steps as f64
        );
    }

    // Sampling.
    for (label, sampler) in [
        ("greedy", SamplerConfig::greedy()),
        ("temp 0.8 + top-k 8", SamplerConfig::temperature(0.8).with_top_k(8)),
        (
            "temp 1.0 + top-p 0.9",
            SamplerConfig::temperature(1.0).with_top_p(0.9),
        ),
    ] {
        let mut generator = Generator::new(
            &model,
            GeneratorConfig::builder()
                .max_new_tokens(12)
                .sampler(sampler)
                .seed(42)
                .build(),
        );
        let out = generator.generate(&[1, 2, 3, 4], &device)?;
        println!("{label:<22} {out:?}");
    }

    Ok(())
}
