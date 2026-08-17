//! Vision Mamba-3 against a Vision Transformer, same crate, same trainer, same seed.
//!
//! The Transformer is the textbook encoder: patch embedding, learned positions,
//! pre-norm blocks of non-causal multi-head attention and a SwiGLU MLP, mean
//! pooling, linear head. The Mamba side is [`VisionMamba3`]: the same patch
//! embedding and positions, bidirectional Mamba-3 mixers in place of attention.
//! Everything shared — data, initialisation seed, optimizer, schedule, batch
//! sizes — is identical, so the comparison is about the mixer and nothing else.
//!
//! Three things get measured:
//!
//! 1. **Quality.** Same steps, same batches: the loss curve and held-out accuracy.
//! 2. **Learning speed.** Steps and wall-clock seconds until the smoothed training
//!    loss first drops under a threshold — the number this example exists for.
//! 3. **Scaling.** Time for one optimizer step as the image (and so the patch
//!    sequence) grows. Attention is quadratic in the patch count, the scan is
//!    linear.
//!
//! ```text
//! cargo run --release --example vision_vs_transformer
//! MAMBA3_MATMUL_PRECISION=bf16 cargo run --release --example vision_vs_transformer
//! ```

use std::time::{Duration, Instant};

use mamba3::models::vision::{Pooling, ScanDirection};
use mamba3::nn::Module;
use mamba3::nn::attention::{AttentionConfig, MultiHeadAttention};
use mamba3::nn::conv::{PatchEmbed, PatchEmbedConfig};
use mamba3::nn::embedding::PositionalEmbedding;
use mamba3::nn::linear::{Linear, LinearConfig};
use mamba3::nn::mlp::{Mlp, MlpConfig};
use mamba3::nn::module::ModuleVisitor;
use mamba3::nn::norm::{RmsNorm, RmsNormConfig};
use mamba3::prelude::*;
use mamba3::tensor::Tensor;
use mamba3::tensor::ops::index::IdTensor;
use mamba3::tensor::ops::random::Rng;
use mamba3::train::{ClassificationTask, ImageBatch, TrainStep};

type R = mamba3::backends::Auto;

const IMAGE: usize = 32;
const PATCH: usize = 4;
const CHANNELS: usize = 3;
const CLASSES: usize = 10;
const D_MODEL: usize = 128;
const N_LAYERS: usize = 4;
const N_HEADS: usize = 4;
const HEAD_DIM: usize = 32;
const BATCH: usize = 8;
const STEPS: u64 = 150;
const SEED: u64 = 1234;
const DATA_SEED: u64 = 99;

/// The smoothed training loss both models must reach; learning speed is the time
/// and step count at which they first do.
const TARGET_LOSS: f32 = 0.35;

/// A bright blob in one of `CLASSES` grid cells, over a noisy background. The blob
/// is only modestly brighter than the noise, so separating the classes takes more
/// than one step, and its cell is what the label encodes, so the model has to
/// aggregate over positions rather than react to a single pixel.
fn make_batch(
    device: &Device<R>,
    rng: &mut Rng,
    image: usize,
    batch: usize,
) -> ImageBatch<R, f32> {
    let cells = 5; // 5x2 grid of possible blob positions.
    let cell_w = image / cells;
    let cell_h = image / 2;
    let blob = image / 8;
    let mut pixels = vec![0.0f32; batch * CHANNELS * image * image];
    let mut labels = Vec::with_capacity(batch);
    for b in 0..batch {
        let class = rng.next_index(CLASSES);
        labels.push(class as u32);
        let row0 = (class / cells) * cell_h + cell_h / 2 - blob / 2;
        let col0 = (class % cells) * cell_w + cell_w / 2 - blob / 2;
        for c in 0..CHANNELS {
            for r in 0..image {
                for col in 0..image {
                    let idx = ((b * CHANNELS + c) * image + r) * image + col;
                    let in_blob =
                        r >= row0 && r < row0 + blob && col >= col0 && col < col0 + blob;
                    pixels[idx] = 0.5 * rng.next_f32() + if in_blob { 1.0 } else { 0.0 };
                }
            }
        }
    }
    ImageBatch {
        images: Tensor::from_f32(&pixels, vec![batch, CHANNELS, image, image], device)
            .unwrap(),
        labels: IdTensor::from_slice(&labels, vec![batch], device).unwrap(),
    }
}

fn mamba_config(image: usize) -> Result<VisionMamba3Config> {
    VisionMamba3Config::builder()
        .image_size(image)
        .patch_size(PATCH)
        .in_channels(CHANNELS)
        .num_classes(CLASSES)
        .d_model(D_MODEL)
        .n_layers(N_LAYERS)
        .direction(ScanDirection::Bidirectional)
        .pooling(Pooling::Mean)
        .with_ssm(|s| {
            s.n_heads = N_HEADS;
            s.n_groups = N_HEADS;
            s.head_dim = HEAD_DIM;
            s.d_state = 32;
            s.chunk_size = 16;
        })
        .seed(SEED)
        .build()
}

/// A pre-norm ViT block: non-causal attention, then a gated MLP.
struct VitBlock {
    norm1: RmsNorm<R, f32>,
    attn: MultiHeadAttention<R, f32>,
    norm2: RmsNorm<R, f32>,
    mlp: Mlp<R, f32>,
}

impl Module<R, f32> for VitBlock {
    fn visit(&self, visitor: &mut ModuleVisitor<'_, R, f32>) {
        visitor.child("norm1", &self.norm1);
        visitor.child("attn", &self.attn);
        visitor.child("norm2", &self.norm2);
        visitor.child("mlp", &self.mlp);
    }
}

/// The textbook vision Transformer, assembled from the crate's own modules.
struct VisionTransformer {
    patch_embed: PatchEmbed<R, f32>,
    pos_embed: PositionalEmbedding<R, f32>,
    blocks: Vec<VitBlock>,
    norm: RmsNorm<R, f32>,
    head: Linear<R, f32>,
}

impl VisionTransformer {
    fn new(image: usize, d_ff: usize, device: &Device<R>) -> Result<Self> {
        let mut rng = Rng::seeded(SEED);
        let patch_embed =
            PatchEmbedConfig::new(image, PATCH, CHANNELS, D_MODEL).init(device, &mut rng)?;
        let num_patches = patch_embed.num_patches();
        let mut blocks = Vec::with_capacity(N_LAYERS);
        for _ in 0..N_LAYERS {
            blocks.push(VitBlock {
                norm1: RmsNormConfig::new(D_MODEL).init(device, &mut rng),
                attn: AttentionConfig::new(D_MODEL, N_HEADS)
                    .with_head_dim(HEAD_DIM)
                    .with_causal(false)
                    .with_rope(false)
                    .init(device, &mut rng)?,
                norm2: RmsNormConfig::new(D_MODEL).init(device, &mut rng),
                mlp: MlpConfig::new(D_MODEL, d_ff)
                    .with_gated(true)
                    .init(device, &mut rng),
            });
        }
        Ok(Self {
            patch_embed,
            pos_embed: PositionalEmbedding::new(num_patches, D_MODEL, device, &mut rng),
            blocks,
            norm: RmsNormConfig::new(D_MODEL).init(device, &mut rng),
            head: LinearConfig::new(D_MODEL, CLASSES).init(device, &mut rng),
        })
    }

    fn forward(&self, images: &mamba3::autograd::Var<R, f32>) -> Result<mamba3::autograd::Var<R, f32>> {
        let mut x = self.patch_embed.apply(images)?;
        x = self.pos_embed.add_to(&x, 0)?;
        for block in &self.blocks {
            x = x.add(&block.attn.apply(&block.norm1.apply(&x)?)?)?;
            x = x.add(&block.mlp.apply(&block.norm2.apply(&x)?)?)?;
        }
        let x = self.norm.apply(&x)?;
        let pooled = x.mean_dim(1)?.squeeze(1)?;
        self.head.apply(&pooled)
    }
}

impl Module<R, f32> for VisionTransformer {
    fn visit(&self, visitor: &mut ModuleVisitor<'_, R, f32>) {
        visitor.child("patch_embed", &self.patch_embed);
        visitor.child("pos_embed", &self.pos_embed);
        for (i, block) in self.blocks.iter().enumerate() {
            visitor.child_at("blocks", i, block);
        }
        visitor.child("norm", &self.norm);
        visitor.child("head", &self.head);
    }
}

/// Single-label classification for the example's ViT, mirroring [`ClassificationTask`].
struct VitTask<'a> {
    model: &'a VisionTransformer,
    params: Vec<mamba3::nn::param::Param<R, f32>>,
    training: std::cell::Cell<bool>,
}

impl<'a> VitTask<'a> {
    fn new(model: &'a VisionTransformer) -> Self {
        Self {
            params: model.parameters(),
            model,
            training: std::cell::Cell::new(true),
        }
    }
}

impl TrainStep<R, f32> for VitTask<'_> {
    type Batch = ImageBatch<R, f32>;

    fn parameters(&self) -> Vec<mamba3::nn::param::Param<R, f32>> {
        self.params.clone()
    }

    fn loss(&self, batch: &Self::Batch) -> Result<mamba3::autograd::Var<R, f32>> {
        if !self.training.get() {
            let _guard = mamba3::autograd::no_grad();
            let images = mamba3::autograd::Var::constant(batch.images.clone());
            let logits = self.model.forward(&images)?;
            return mamba3::train::cross_entropy(&logits, &batch.labels);
        }
        let images = mamba3::autograd::Var::traced(batch.images.clone());
        let logits = self.model.forward(&images)?;
        mamba3::train::cross_entropy(&logits, &batch.labels)
    }

    fn set_training(&self, training: bool) {
        self.training.set(training);
        self.model.set_training(training);
    }
}

/// The widest SwiGLU that keeps the Transformer at or below the Mamba stack's
/// parameter count, found by construction rather than by hand.
fn matched_d_ff(target: usize, device: &Device<R>) -> Result<usize> {
    let mut best = 8;
    for d_ff in (8..=1024).step_by(8) {
        let params = VisionTransformer::new(IMAGE, d_ff, device)?.num_parameters();
        if params <= target {
            best = d_ff;
        } else {
            break;
        }
    }
    Ok(best)
}

struct Trained {
    name: &'static str,
    parameters: usize,
    first_loss: f32,
    final_loss: f32,
    accuracy: f32,
    train_time: Duration,
    /// `(step, seconds since training started)` when the smoothed loss first
    /// dropped under [`TARGET_LOSS`]; `None` if it never did.
    reached: Option<(u64, f64)>,
}

/// Drive `task` for [`STEPS`] steps over a shared data stream and record when the
/// smoothed loss crosses the target.
fn train_task<T: TrainStep<R, f32, Batch = ImageBatch<R, f32>>>(
    name: &'static str,
    task: &T,
    parameters: usize,
    device: &Device<R>,
) -> Result<Trained> {
    let trainer_config = TrainerConfig::builder()
        .learning_rate(3e-3)
        .schedule(LrSchedule::cosine(STEPS))
        .max_grad_norm(1.0)
        .max_steps(STEPS)
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
    .on_step(move |info| {
        println!("  {name:<14} step {:>4}  loss {:.4}", info.step, info.loss);
    });

    // Same seed for the data stream, so both models see identical batches.
    let mut rng = Rng::seeded(DATA_SEED);
    let batches: Vec<ImageBatch<R, f32>> =
        (0..STEPS).map(|_| make_batch(device, &mut rng, IMAGE, BATCH)).collect();

    // One step at a time so the clock can be read as each loss comes back; the
    // trainer reads the loss at the end of every step anyway.
    let mut reached = None;
    let mut recent: Vec<f32> = Vec::new();
    let mut first_loss = 0.0f32;
    let mut final_loss = 0.0f32;
    let started = Instant::now();
    for (i, batch) in batches.iter().enumerate() {
        let info = trainer.step(task, std::slice::from_ref(batch))?;
        if i == 0 {
            first_loss = info.loss;
        }
        final_loss = info.loss;
        recent.push(info.loss);
        if recent.len() > 5 {
            recent.remove(0);
        }
        let smoothed = recent.iter().sum::<f32>() / recent.len() as f32;
        if reached.is_none() && recent.len() == 5 && smoothed <= TARGET_LOSS {
            reached = Some((info.step, started.elapsed().as_secs_f64()));
        }
    }
    device.synchronize();
    let train_time = started.elapsed();

    Ok(Trained {
        name,
        parameters,
        first_loss,
        final_loss,
        // Held-out accuracy is filled in by the caller, which owns the model.
        accuracy: 0.0,
        train_time,
        reached,
    })
}

/// One optimizer step at a given image size, for the scaling sweep.
fn mamba_step_time(device: &Device<R>, image: usize) -> Result<Duration> {
    let model = mamba_config(image)?.init::<R, f32>(device)?;
    let task = ClassificationTask::new(&model);
    step_time(&task, device, image)
}

fn vit_step_time(device: &Device<R>, image: usize, d_ff: usize) -> Result<Duration> {
    let model = VisionTransformer::new(image, d_ff, device)?;
    let task = VitTask::new(&model);
    step_time(&task, device, image)
}

fn step_time<T: TrainStep<R, f32, Batch = ImageBatch<R, f32>>>(
    task: &T,
    device: &Device<R>,
    image: usize,
) -> Result<Duration> {
    let mut rng = Rng::seeded(5);
    let data = make_batch(device, &mut rng, image, 2);
    let mut trainer = Trainer::new(
        TrainerConfig::builder().learning_rate(1e-4).build()?,
        AdamW::<R, f32>::new(1e-4),
    );
    trainer.step(task, std::slice::from_ref(&data))?;
    device.synchronize();
    let started = Instant::now();
    trainer.step(task, std::slice::from_ref(&data))?;
    device.synchronize();
    Ok(started.elapsed())
}

fn main() -> Result<()> {
    mamba3::tensor::ops::matmul::set_precision_from_env();
    let device = Device::<R>::default();
    println!("backend: {}\n", device.name());

    let mamba_params = mamba_config(IMAGE)?.init::<R, f32>(&device)?.num_parameters();
    let wide_d_ff = (D_MODEL * 8 / 3).div_ceil(8) * 8;
    let lite_d_ff = matched_d_ff(mamba_params, &device)?;
    println!(
        "vision mamba: {mamba_params} parameters, {} patches\n\
         transformer d_ff: {wide_d_ff} (textbook SwiGLU) and {lite_d_ff} (parameter-matched)\n",
        (IMAGE / PATCH) * (IMAGE / PATCH)
    );

    println!("== training ({CLASSES}-way blob classification, {STEPS} steps) ==");
    let mamba_model = mamba_config(IMAGE)?.init::<R, f32>(&device)?;
    let mamba_task = ClassificationTask::new(&mamba_model);
    let mut mamba = train_task("mamba-3", &mamba_task, mamba_params, &device)?;
    mamba_model.eval();
    mamba.accuracy = held_out_accuracy(&device, |images| mamba_model.forward(images))?;

    let wide_model = VisionTransformer::new(IMAGE, wide_d_ff, &device)?;
    let wide_task = VitTask::new(&wide_model);
    let mut wide = train_task("vit", &wide_task, wide_model.num_parameters(), &device)?;
    wide_model.set_training(false);
    wide.accuracy = held_out_accuracy(&device, |images| wide_model.forward(images))?;

    let lite_model = VisionTransformer::new(IMAGE, lite_d_ff, &device)?;
    let lite_task = VitTask::new(&lite_model);
    let mut lite = train_task("vit-lite", &lite_task, lite_model.num_parameters(), &device)?;
    lite_model.set_training(false);
    lite.accuracy = held_out_accuracy(&device, |images| lite_model.forward(images))?;

    println!("\n== quality and learning speed (identical data, steps, optimizer, seed) ==");
    println!(
        "{:<12}{:>10}{:>10}{:>10}{:>10}{:>12}{:>16}",
        "model", "params", "loss@1", "loss@end", "held-out", "train time", "to loss<=0.35"
    );
    for r in [&mamba, &wide, &lite] {
        let reached = match r.reached {
            Some((step, secs)) => format!("{secs:.2}s @ {step}"),
            None => "never".to_string(),
        };
        println!(
            "{:<12}{:>10}{:>10.4}{:>10.4}{:>9.1}%{:>11.2?}{:>16}",
            r.name,
            r.parameters,
            r.first_loss,
            r.final_loss,
            r.accuracy * 100.0,
            r.train_time,
            reached
        );
    }

    println!("\n== one training step: forward, backward and update (batch 2) ==");
    println!(
        "{:>8}{:>8}{:>16}{:>16}{:>10}",
        "image", "patches", "mamba-3", "vit", "speedup"
    );
    for image in [32usize, 64, 128] {
        let m = mamba_step_time(&device, image)?;
        let t = vit_step_time(&device, image, wide_d_ff)?;
        println!(
            "{image:>8}{:>8}{:>16.2?}{:>16.2?}{:>9.2}x",
            (image / PATCH) * (image / PATCH),
            m,
            t,
            t.as_secs_f64() / m.as_secs_f64()
        );
    }

    Ok(())
}

/// Mean held-out accuracy over a few fresh batches.
fn held_out_accuracy(
    device: &Device<R>,
    forward: impl Fn(&mamba3::autograd::Var<R, f32>) -> Result<mamba3::autograd::Var<R, f32>>,
) -> Result<f32> {
    let _guard = mamba3::autograd::no_grad();
    let mut rng = Rng::seeded(DATA_SEED + 1);
    let mut total = 0.0f32;
    let evals = 4;
    for _ in 0..evals {
        let eval = make_batch(device, &mut rng, IMAGE, BATCH);
        let logits = forward(&mamba3::autograd::Var::constant(eval.images.clone()))?;
        total += mamba3::train::accuracy(&logits, &eval.labels)?;
    }
    Ok(total / evals as f32)
}
