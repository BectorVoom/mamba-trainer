//! Parameter-efficient fine-tuning with LoRA, and quantization-aware training.
//!
//! Two things this example shows:
//!
//! * a base model is frozen and only the rank-`r` adapters train, so the optimizer
//!   touches a small fraction of the weights and the adapter can be merged back
//!   into the base at the end with no change in behaviour;
//! * the same `Linear` also carries fake quantizers, so QAT is one builder call
//!   away and composes with LoRA.
//!
//! ```text
//! cargo run --release --example finetune_lora
//! ```

use mamba3::nn::quant::QuantConfig;
use mamba3::prelude::*;
use mamba3::tensor::ops::index::IdTensor;
use mamba3::tensor::ops::random::Rng;
use mamba3::train::{Checkpoint, LmBatch, LmTask};

type R = mamba3::backends::Auto;

const VOCAB: usize = 24;
const SEQ: usize = 24;
const BATCH: usize = 4;

fn make_batch(device: &Device<R>, rng: &mut Rng, offset: u32) -> LmBatch<R> {
    let mut inputs = Vec::new();
    let mut targets = Vec::new();
    for _ in 0..BATCH {
        let start = rng.next_index(VOCAB) as u32;
        for t in 0..SEQ {
            inputs.push((start + t as u32) % VOCAB as u32);
            // The fine-tuning task shifts the mapping by `offset`, so the adapter
            // has something specific to learn.
            targets.push((start + t as u32 + 1 + offset) % VOCAB as u32);
        }
    }
    LmBatch {
        inputs: IdTensor::from_slice(&inputs, vec![BATCH, SEQ], device).unwrap(),
        targets: IdTensor::from_slice(&targets, vec![BATCH, SEQ], device).unwrap(),
    }
}

fn train(
    model: &Mamba3Lm<R, f32>,
    task: &LmTask<'_, R, f32>,
    steps: u64,
    lr: f32,
    device: &Device<R>,
    rng: &mut Rng,
    offset: u32,
) -> Result<(f32, f32)> {
    let config = TrainerConfig::builder()
        .learning_rate(lr)
        .schedule(LrSchedule::cosine(steps))
        .max_steps(steps)
        .build()?;
    let mut trainer = Trainer::new(config, AdamW::<R, f32>::new(lr));
    let batches: Vec<LmBatch<R>> = (0..steps).map(|_| make_batch(device, rng, offset)).collect();
    let report = trainer.fit(task, batches)?;
    let _ = model;
    Ok((
        report.losses.first().copied().unwrap_or_default(),
        report.final_loss,
    ))
}

fn main() -> Result<()> {
    let device = Device::<R>::default();
    let mut rng = Rng::seeded(2024);

    // ---- 1. a base model with adapters and 8-bit fake quantization attached ----
    let model = Mamba3LmConfig::builder()
        .vocab_size(VOCAB)
        .d_model(64)
        .n_layers(2)
        .with_ssm(|s| {
            s.n_heads = 4;
            s.n_groups = 4;
            s.head_dim = 16;
            s.d_state = 16;
            s.chunk_size = 16;
        })
        .lora(
            LoraConfig::builder()
                .rank(8)
                .alpha(16.0)
                .dropout(0.0)
                .build()?,
        )
        .weight_quant(QuantConfig::int8_weights())
        .seed(11)
        .build()?
        .init::<R, f32>(&device)?;

    println!("{model:?}");
    println!("total parameters: {}", model.num_parameters());

    // ---- 2. pretrain everything on the base task -----------------------------
    // Adapters start at zero, so this is an ordinary full fine-tune.
    let base_task = LmTask::new(&model);
    let (before, after) = train(&model, &base_task, 100, 3e-3, &device, &mut rng, 0)?;
    println!("pretrain loss {before:.4} -> {after:.4}");

    let pretrained = Checkpoint::capture(&model, 100);

    // ---- 3. freeze the base, train only the adapters -------------------------
    model.freeze_matching(&[]);
    model.unfreeze_matching(&["lora"]);
    println!(
        "trainable after freezing: {} of {} parameters ({:.2}%)",
        model.num_trainable_parameters(),
        model.num_parameters(),
        100.0 * model.num_trainable_parameters() as f32 / model.num_parameters() as f32
    );

    let lora_task = LmTask::new(&model).only(&["lora"]);
    let (before, after) = train(&model, &lora_task, 100, 1e-2, &device, &mut rng, 3)?;
    println!("LoRA fine-tune loss {before:.4} -> {after:.4}");

    // Only adapter weights moved.
    let now = model.state_dict();
    let mut moved = 0;
    let mut frozen_moved = 0;
    for (name, entry) in &now.entries {
        let original = &pretrained.state.entries[name];
        let changed = entry
            .data
            .iter()
            .zip(&original.data)
            .any(|(a, b)| (a - b).abs() > 1e-9);
        if changed {
            if name.contains("lora") {
                moved += 1;
            } else {
                frozen_moved += 1;
            }
        }
    }
    println!("adapter tensors updated: {moved}, frozen tensors updated: {frozen_moved}");
    assert_eq!(frozen_moved, 0);

    // ---- 4. ship the adapter on its own --------------------------------------
    let adapter_only = Checkpoint::capture(&model, 200).filtered("lora");
    println!(
        "adapter checkpoint: {} tensors, {} values (vs {} for the full model)",
        adapter_only.state.entries.len(),
        adapter_only.num_values(),
        Checkpoint::capture(&model, 200).num_values()
    );

    // ---- 5. merge the adapter and confirm the function is unchanged ----------
    let probe = IdTensor::from_slice(&[1, 2, 3, 4, 5, 6], vec![1, 6], &device)?;
    model.eval();
    let before_merge = model.forward(&probe, false)?.to_f32();

    // One traversal folds `A @ B` into every base weight it can find.
    model.merge_lora_adapters()?;

    let after_merge = model.forward(&probe, false)?.to_f32();
    let drift = before_merge
        .iter()
        .zip(&after_merge)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("max logit change after merging adapters: {drift:.2e}");
    println!(
        "  (not exactly zero because weight fake-quantization is on: the grid for\n\
         \x20  W + AB is not the grid for W, so quantize(W + AB) != quantize(W) + AB.\n\
         \x20  Without `weight_quant` the merge is exact — see the `lora_starts_neutral\n\
         \x20  _and_merges_exactly` test.)"
    );

    Ok(())
}
