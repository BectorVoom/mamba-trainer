//! Vision Mamba-3 on a synthetic image task.
//!
//! Two coloured-blob classes on small images: enough to show the patch embedding,
//! the bidirectional scan and the classification head working together, and to see
//! the loss fall on a CPU in a few seconds.
//!
//! ```text
//! cargo run --release --example vision
//! ```

use mamba3::models::vision::{Pooling, ScanDirection};
use mamba3::prelude::*;
use mamba3::tensor::Tensor;
use mamba3::tensor::ops::index::IdTensor;
use mamba3::tensor::ops::random::Rng;
use mamba3::train::{ClassificationTask, ImageBatch};

type R = mamba3::backends::Auto;

const IMAGE: usize = 16;
const PATCH: usize = 4;
const CHANNELS: usize = 3;
const CLASSES: usize = 2;
const BATCH: usize = 8;

/// Class 0 puts a bright square in the top-left quadrant, class 1 in the bottom
/// right. A model that ignores position cannot separate them.
fn make_batch(device: &Device<R>, rng: &mut Rng) -> ImageBatch<R, f32> {
    let mut pixels = vec![0.0f32; BATCH * CHANNELS * IMAGE * IMAGE];
    let mut labels = Vec::with_capacity(BATCH);
    for b in 0..BATCH {
        let class = rng.next_index(CLASSES);
        labels.push(class as u32);
        let (row0, col0) = if class == 0 { (0, 0) } else { (IMAGE / 2, IMAGE / 2) };
        for c in 0..CHANNELS {
            for r in row0..row0 + IMAGE / 2 {
                for col in col0..col0 + IMAGE / 2 {
                    let idx = ((b * CHANNELS + c) * IMAGE + r) * IMAGE + col;
                    pixels[idx] = 1.0 + 0.1 * rng.next_f32();
                }
            }
        }
    }
    ImageBatch {
        images: Tensor::from_f32(
            &pixels,
            vec![BATCH, CHANNELS, IMAGE, IMAGE],
            device,
        )
        .unwrap(),
        labels: IdTensor::from_slice(&labels, vec![BATCH], device).unwrap(),
    }
}

fn main() -> Result<()> {
    let device = Device::<R>::default();

    let model = VisionMamba3Config::builder()
        .image_size(IMAGE)
        .patch_size(PATCH)
        .in_channels(CHANNELS)
        .num_classes(CLASSES)
        .d_model(64)
        .n_layers(2)
        // Images have no causal order, so every block scans both ways and adds the
        // results. Each direction owns its own mixer.
        .direction(ScanDirection::Bidirectional)
        .pooling(Pooling::Mean)
        .with_ssm(|s| {
            s.n_heads = 4;
            s.n_groups = 4;
            s.head_dim = 16;
            s.d_state = 16;
            s.chunk_size = 8;
        })
        .seed(5)
        .build()?
        .init::<R, f32>(&device)?;

    println!("{model:?}");
    println!(
        "{} patches per image, {} parameters",
        model.config().num_patches(),
        model.num_parameters()
    );

    let mut rng = Rng::seeded(17);
    let steps = 60u64;

    let trainer_config = TrainerConfig::builder()
        .learning_rate(3e-3)
        .schedule(LrSchedule::cosine(steps))
        .max_steps(steps)
        .log_every(10)
        .build()?;
    let mut trainer = Trainer::new(trainer_config, AdamW::<R, f32>::new(3e-3)).on_step(|info| {
        println!(
            "step {:>3}  loss {:.4}  lr {:.2e}",
            info.step, info.loss, info.learning_rate
        );
    });

    let task = ClassificationTask::new(&model);
    let batches: Vec<_> = (0..steps).map(|_| make_batch(&device, &mut rng)).collect();
    let report = trainer.fit(&task, batches)?;
    println!(
        "loss {:.4} -> {:.4}",
        report.losses.first().copied().unwrap_or_default(),
        report.final_loss
    );

    // Held-out accuracy.
    model.eval();
    let eval = make_batch(&device, &mut rng);
    let logits = model.forward(&mamba3::autograd::Var::constant(eval.images.clone()))?;
    let accuracy = mamba3::train::accuracy(&logits, &eval.labels)?;
    println!("held-out accuracy: {:.1}%", accuracy * 100.0);

    Ok(())
}
