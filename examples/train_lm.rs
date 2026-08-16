//! Train a small Mamba-3 language model on a synthetic task.
//!
//! The task is "repeat the sequence you were given, shifted by one" over a tiny
//! vocabulary — small enough to overfit in a few hundred steps on a laptop CPU, so
//! the loss curve is visible without a GPU.
//!
//! ```text
//! cargo run --release --example train_lm
//! ```

use mamba3::prelude::*;
use mamba3::tensor::ops::index::IdTensor;
use mamba3::tensor::ops::random::Rng;
use mamba3::train::{Checkpoint, LmBatch, LmTask};

type R = mamba3::backends::Auto;

const VOCAB: usize = 24;
const SEQ: usize = 32;
const BATCH: usize = 4;

/// Sequences that step through the vocabulary with a per-sequence stride, so the
/// model has to use its state rather than memorise unigrams.
fn make_batch(device: &Device<R>, rng: &mut Rng) -> LmBatch<R> {
    let mut inputs = Vec::with_capacity(BATCH * SEQ);
    let mut targets = Vec::with_capacity(BATCH * SEQ);
    for _ in 0..BATCH {
        let start = rng.next_index(VOCAB) as u32;
        let stride = 1 + rng.next_index(3) as u32;
        for t in 0..SEQ {
            let token = (start + stride * t as u32) % VOCAB as u32;
            let next = (start + stride * (t as u32 + 1)) % VOCAB as u32;
            inputs.push(token);
            targets.push(next);
        }
    }
    LmBatch {
        inputs: IdTensor::from_slice(&inputs, vec![BATCH, SEQ], device).unwrap(),
        targets: IdTensor::from_slice(&targets, vec![BATCH, SEQ], device).unwrap(),
    }
}

fn main() -> Result<()> {
    mamba3::tensor::ops::matmul::set_precision_from_env();
    let device = Device::<R>::default();
    println!("backend: {}", device.name());

    let config = Mamba3LmConfig::builder()
        .vocab_size(VOCAB)
        .d_model(64)
        .n_layers(2)
        .with_ssm(|s| {
            s.n_heads = 4;
            s.n_groups = 4;
            s.head_dim = 16;
            s.d_state = 16;
            s.chunk_size = 16;
            // The Mamba-3 defaults: learned trapezoidal discretization plus
            // rotational (complex) state updates.
        })
        .seed(1234)
        .build()?;

    let model = config.init::<R, f32>(&device)?;
    println!("{model:?}");
    println!("{} parameters", model.num_parameters());

    let steps = 250u64;
    let trainer_config = TrainerConfig::builder()
        .learning_rate(3e-3)
        .schedule(LrSchedule::cosine(steps))
        .max_grad_norm(1.0)
        .max_steps(steps)
        .log_every(25)
        .build()?;

    let mut trainer = Trainer::new(
        trainer_config,
        AdamWConfig::builder()
            .learning_rate(3e-3)
            .weight_decay(0.01)
            .build()
            .init::<R, f32>(),
    )
    .on_step(|info| {
        println!(
            "step {:>4}  loss {:.4}  ppl {:>8.2}  lr {:.2e}  |g| {:.3}",
            info.step,
            info.loss,
            mamba3::train::perplexity(info.loss),
            info.learning_rate,
            info.grad_norm
        );
    });

    let task = LmTask::new(&model);
    let mut rng = Rng::seeded(99);
    let batches: Vec<LmBatch<R>> = (0..steps).map(|_| make_batch(&device, &mut rng)).collect();

    let report = trainer.fit(&task, batches)?;
    println!(
        "\nfinished {} steps, final loss {:.4} (started at {:.4})",
        report.steps,
        report.final_loss,
        report.losses.first().copied().unwrap_or_default()
    );

    // Held-out accuracy.
    let eval = make_batch(&device, &mut rng);
    model.eval();
    let logits = model.forward(&eval.inputs, false)?;
    let accuracy = mamba3::train::accuracy(&logits, &eval.targets)?;
    println!("held-out next-token accuracy: {:.1}%", accuracy * 100.0);

    // Generate a continuation with the recurrent cache.
    let mut generator = Generator::new(
        &model,
        GeneratorConfig::builder()
            .max_new_tokens(16)
            .sampler(SamplerConfig::greedy())
            .build(),
    );
    let prompt = [0u32, 2, 4, 6];
    let continuation = generator.generate(&prompt, &device)?;
    println!("prompt      {prompt:?}");
    println!("continued   {continuation:?}");

    let path = std::env::temp_dir().join("mamba3-train-lm.json");
    Checkpoint::capture(&model, report.steps)
        .with_metadata(serde_json::json!({ "task": "synthetic arithmetic" }))
        .save(&path)?;
    println!("checkpoint written to {}", path.display());

    Ok(())
}
