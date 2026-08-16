//! Mamba-3 against a Transformer, same crate, same trainer, same seed.
//!
//! Both models are built from [`Mamba3LmConfig`]: the only difference is the layer
//! pattern. `AllMamba` gives a recurrent stack, `AllAttention` plus an MLP gives an
//! ordinary pre-norm Transformer. Everything else — embeddings, RMS norm, the
//! optimizer, the schedule, the data, the initialisation seed — is shared, so the
//! comparison is about the mixer and nothing else.
//!
//! Three things get measured:
//!
//! 1. **Quality.** Same steps, same batches: final training loss and held-out
//!    next-token accuracy.
//! 2. **Prefill.** Time for one forward pass over a whole sequence, as the sequence
//!    grows. Attention is quadratic in the sequence length; the chunked scan is
//!    linear.
//! 3. **Decoding.** Time per generated token, and cache bytes, as the context
//!    grows. This is the one that matters: a Transformer's KV cache and per-token
//!    cost both grow with the context, a Mamba state does neither.
//!
//! ```text
//! cargo run --release --example vs_transformer
//! cargo run --release --no-default-features --features wgpu --example vs_transformer
//! ```

use std::time::{Duration, Instant};

use mamba3::models::hybrid::LayerPattern;
use mamba3::nn::Module;
use mamba3::nn::attention::AttentionConfig;
use mamba3::nn::mlp::MlpConfig;
use mamba3::prelude::*;
use mamba3::tensor::ops::index::IdTensor;
use mamba3::tensor::ops::random::Rng;
use mamba3::train::{LmBatch, LmTask};

type R = mamba3::backends::Auto;

const VOCAB: usize = 64;
const D_MODEL: usize = 128;
const N_LAYERS: usize = 4;
const N_HEADS: usize = 4;
const HEAD_DIM: usize = 32;
const TRAIN_SEQ: usize = 64;
const BATCH: usize = 8;
const STEPS: u64 = 200;
const SEED: u64 = 1234;

/// The longest context either model is asked to decode at.
const MAX_CTX: usize = 4096;
/// Sequences decoded at once. Attention's per-token work scales with this and with
/// the context; a recurrent state's does not.
const DECODE_BATCH: usize = 8;

/// Sequences that step through the vocabulary with a per-sequence stride. Predicting
/// the next token needs the stride, which is only visible from earlier positions, so
/// a model has to carry something forward rather than memorise unigrams.
fn make_batch(device: &Device<R>, rng: &mut Rng, seq: usize, batch: usize) -> LmBatch<R> {
    let mut inputs = Vec::with_capacity(batch * seq);
    let mut targets = Vec::with_capacity(batch * seq);
    for _ in 0..batch {
        let start = rng.next_index(VOCAB) as u32;
        let stride = 1 + rng.next_index(5) as u32;
        for t in 0..seq {
            inputs.push((start + stride * t as u32) % VOCAB as u32);
            targets.push((start + stride * (t as u32 + 1)) % VOCAB as u32);
        }
    }
    LmBatch {
        inputs: IdTensor::from_slice(&inputs, vec![batch, seq], device).unwrap(),
        targets: IdTensor::from_slice(&targets, vec![batch, seq], device).unwrap(),
    }
}

fn mamba_config() -> Result<Mamba3LmConfig> {
    Mamba3LmConfig::builder()
        .vocab_size(VOCAB)
        .d_model(D_MODEL)
        .n_layers(N_LAYERS)
        .pattern(LayerPattern::AllMamba)
        .with_ssm(|s| {
            s.n_heads = N_HEADS;
            s.n_groups = N_HEADS;
            s.head_dim = HEAD_DIM;
            s.d_state = 32;
            s.chunk_size = 32;
        })
        .seed(SEED)
        .build()
}

/// A pre-norm Transformer: multi-head causal attention with RoPE, then a gated
/// SwiGLU feed-forward of width `d_ff`.
///
/// Two are built. The first uses the textbook `8/3 * d_model` SwiGLU width, which at
/// this `d_model` costs about twice the Mamba stack's parameters — the version with
/// no handicap. The second shrinks `d_ff` until the parameter counts match, which is
/// the like-for-like comparison. Reporting both removes the argument.
fn transformer_config(d_ff: usize) -> Result<Mamba3LmConfig> {
    Mamba3LmConfig::builder()
        .vocab_size(VOCAB)
        .d_model(D_MODEL)
        .n_layers(N_LAYERS)
        .pattern(LayerPattern::AllAttention)
        .attention(
            AttentionConfig::new(D_MODEL, N_HEADS)
                .with_head_dim(HEAD_DIM)
                // Room for the longest context plus the tokens decoded after it.
                .with_max_seq_len(MAX_CTX + 128)
                .with_causal(true),
        )
        .mlp(MlpConfig::new(D_MODEL, d_ff).with_gated(true))
        .seed(SEED)
        .build()
}

/// The widest SwiGLU that keeps the Transformer at or below the Mamba stack's
/// parameter count, found by construction rather than by hand.
fn matched_d_ff(target: usize, device: &Device<R>) -> Result<usize> {
    let mut best = 8;
    for d_ff in (8..=1024).step_by(8) {
        let params = transformer_config(d_ff)?
            .init::<R, f32>(device)?
            .num_parameters();
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
}

fn train(name: &'static str, config: Mamba3LmConfig, device: &Device<R>) -> Result<Trained> {
    let model = config.init::<R, f32>(device)?;
    let parameters = model.num_parameters();

    let trainer_config = TrainerConfig::builder()
        .learning_rate(3e-3)
        .schedule(LrSchedule::cosine(STEPS))
        .max_grad_norm(1.0)
        .max_steps(STEPS)
        .log_every(50)
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
        println!(
            "  {name:<12} step {:>4}  loss {:.4}  ppl {:>8.2}",
            info.step,
            info.loss,
            mamba3::train::perplexity(info.loss)
        );
    });

    // Same seed for the data stream, so both models see identical batches.
    let mut rng = Rng::seeded(99);
    let batches: Vec<LmBatch<R>> = (0..STEPS)
        .map(|_| make_batch(device, &mut rng, TRAIN_SEQ, BATCH))
        .collect();

    let task = LmTask::new(&model);
    let started = Instant::now();
    let report = trainer.fit(&task, batches)?;
    device.synchronize();
    let train_time = started.elapsed();

    model.eval();
    let eval = make_batch(device, &mut rng, TRAIN_SEQ, BATCH);
    let logits = model.forward(&eval.inputs, false)?;
    let accuracy = mamba3::train::accuracy(&logits, &eval.targets)?;

    Ok(Trained {
        name,
        parameters,
        first_loss: report.losses.first().copied().unwrap_or_default(),
        final_loss: report.final_loss,
        accuracy,
        train_time,
    })
}

/// Time one full forward pass over `seq` tokens.
fn prefill_time(model: &Mamba3Lm<R, f32>, device: &Device<R>, seq: usize) -> Result<Duration> {
    let ids: Vec<u32> = (0..seq).map(|i| (i % VOCAB) as u32).collect();
    let ids = IdTensor::from_slice(&ids, vec![1, seq], device)?;
    let _ = model.forward(&ids, false)?;
    device.synchronize();
    let started = Instant::now();
    let out = model.forward(&ids, false)?;
    device.synchronize();
    let elapsed = started.elapsed();
    std::hint::black_box(out);
    Ok(elapsed)
}

/// Time one training step — forward, backward and the optimizer update — at a given
/// sequence length.
fn train_step_time(
    model: &Mamba3Lm<R, f32>,
    device: &Device<R>,
    seq: usize,
    batch: usize,
) -> Result<Duration> {
    let mut rng = Rng::seeded(5);
    let data = make_batch(device, &mut rng, seq, batch);
    let task = LmTask::new(model);
    let mut trainer = Trainer::new(
        TrainerConfig::builder().learning_rate(1e-4).build()?,
        AdamW::<R, f32>::new(1e-4),
    );
    trainer.step(&task, std::slice::from_ref(&data))?;
    device.synchronize();
    let started = Instant::now();
    trainer.step(&task, std::slice::from_ref(&data))?;
    device.synchronize();
    Ok(started.elapsed())
}

/// Decode `steps` tokens after a `context`-token prompt; report per-token time and
/// the elements the cache holds at the end.
fn decode_cost(
    model: &Mamba3Lm<R, f32>,
    device: &Device<R>,
    context: usize,
    steps: usize,
) -> Result<(Duration, usize)> {
    // Fill the cache in blocks. Attention materialises a `[batch, heads, t, t]`
    // score matrix, so prefilling 8x4096 in one pass asks for a 2 GB buffer and a
    // laptop GPU refuses; chunking is what a real serving loop does anyway. Both
    // models are prefilled the same way so the comparison stays like-for-like.
    const CHUNK: usize = 512;
    let mut cache = model.empty_cache(DECODE_BATCH, device);
    for start in (0..context).step_by(CHUNK) {
        let len = CHUNK.min(context - start);
        let block: Vec<u32> = (0..DECODE_BATCH * len)
            .map(|i| ((start + i % len) % VOCAB) as u32)
            .collect();
        let ids = IdTensor::from_slice(&block, vec![DECODE_BATCH, len], device)?;
        model.forward_cached(&ids, &mut cache)?;
    }
    device.synchronize();

    // One untimed step so any kernel needed at this shape is already compiled.
    let warm = IdTensor::from_slice(&[0u32; DECODE_BATCH], vec![DECODE_BATCH, 1], device)?;
    model.forward_cached(&warm, &mut cache)?;
    device.synchronize();

    let started = Instant::now();
    for i in 0..steps {
        let step = IdTensor::from_slice(
            &[(i % VOCAB) as u32; DECODE_BATCH],
            vec![DECODE_BATCH, 1],
            device,
        )?;
        model.forward_cached(&step, &mut cache)?;
    }
    device.synchronize();
    let per_token = started.elapsed() / steps as u32;
    let elements: usize = cache.iter().map(|c| c.num_elements()).sum();
    Ok((per_token, elements))
}

fn main() -> Result<()> {
    let device = Device::<R>::default();
    println!("backend: {}\n", device.name());

    let mamba_params = mamba_config()?.init::<R, f32>(&device)?.num_parameters();
    // 8/3 * d_model, rounded up to a multiple of 8: the usual SwiGLU width.
    let wide_d_ff = (D_MODEL * 8 / 3).div_ceil(8) * 8;
    let lite_d_ff = matched_d_ff(mamba_params, &device)?;
    println!(
        "mamba stack: {mamba_params} parameters\n\
         transformer d_ff: {wide_d_ff} (textbook SwiGLU) and {lite_d_ff} (parameter-matched)\n"
    );

    println!("== training ==");
    let mamba = train("mamba-3", mamba_config()?, &device)?;
    let wide = train("transformer", transformer_config(wide_d_ff)?, &device)?;
    let lite = train("transformer-lite", transformer_config(lite_d_ff)?, &device)?;

    println!("\n== quality (identical data, steps, optimizer and seed) ==");
    println!(
        "{:<18}{:>12}{:>12}{:>12}{:>12}{:>12}",
        "model", "params", "loss@1", "loss@end", "held-out", "train time"
    );
    for r in [&mamba, &wide, &lite] {
        println!(
            "{:<18}{:>12}{:>12.4}{:>12.4}{:>11.1}%{:>12.2?}",
            r.name,
            r.parameters,
            r.first_loss,
            r.final_loss,
            r.accuracy * 100.0,
            r.train_time
        );
    }

    // Rebuild both, untrained, for the timing sweeps: weights do not affect cost.
    let mamba_model = mamba_config()?.init::<R, f32>(&device)?;
    let transformer_model = transformer_config(wide_d_ff)?.init::<R, f32>(&device)?;
    mamba_model.eval();
    transformer_model.eval();
    let mamba_train = mamba_config()?.init::<R, f32>(&device)?;
    let transformer_train = transformer_config(wide_d_ff)?.init::<R, f32>(&device)?;

    println!("\n== prefill: one forward pass over the whole sequence ==");
    println!(
        "{:>8}{:>16}{:>16}{:>10}",
        "tokens", "mamba-3", "transformer", "speedup"
    );
    // Attention materialises a `[batch, heads, seq, seq]` score matrix; past 2048
    // tokens that alone outgrows the buffer limit of a laptop GPU, which is itself
    // part of the point.
    for seq in [128usize, 256, 512, 1024, 2048] {
        let m = prefill_time(&mamba_model, &device, seq)?;
        let t = prefill_time(&transformer_model, &device, seq)?;
        println!(
            "{seq:>8}{:>16.2?}{:>16.2?}{:>9.2}x",
            m,
            t,
            t.as_secs_f64() / m.as_secs_f64()
        );
    }

    println!("\n== one training step: forward, backward and update (batch 2) ==");
    println!(
        "{:>8}{:>16}{:>16}{:>10}",
        "tokens", "mamba-3", "transformer", "speedup"
    );
    for seq in [64usize, 128, 256, 512, 1024] {
        let m = train_step_time(&mamba_train, &device, seq, 2)?;
        let t = train_step_time(&transformer_train, &device, seq, 2)?;
        println!(
            "{seq:>8}{:>16.2?}{:>16.2?}{:>9.2}x",
            m,
            t,
            t.as_secs_f64() / m.as_secs_f64()
        );
    }

    println!(
        "\n== decoding: cost of the next token at a given context (batch {DECODE_BATCH}) =="
    );
    println!(
        "{:>8}{:>14}{:>14}{:>10}{:>14}{:>14}",
        "context", "mamba-3", "transformer", "speedup", "mamba cache", "kv cache"
    );
    for context in [128usize, 256, 512, 1024, 2048, 4096] {
        let (m, m_cache) = decode_cost(&mamba_model, &device, context, 16)?;
        let (t, t_cache) = decode_cost(&transformer_model, &device, context, 16)?;
        println!(
            "{context:>8}{:>14.2?}{:>14.2?}{:>9.2}x{:>12} KB{:>11} KB",
            m,
            t,
            t.as_secs_f64() / m.as_secs_f64(),
            m_cache * 4 / 1024,
            t_cache * 4 / 1024
        );
    }

    println!(
        "\nRead the speedup columns as crossovers, not as a headline. At short\n\
         sequences neither model is doing enough arithmetic to matter and the cost\n\
         is kernel dispatch, of which the recurrence issues more — that is where the\n\
         Transformer wins, and it is an implementation gap, not an architectural one.\n\
         Past the crossover the asymptotics take over: attention pays O(T^2) to train\n\
         and O(T) per decoded token with an O(T) cache, and none of those terms exist\n\
         on the other side."
    );
    Ok(())
}
