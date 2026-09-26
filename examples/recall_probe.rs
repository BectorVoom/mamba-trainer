//! Associative-recall probe: how much can the recurrent state *remember*?
//!
//! Multi-query associative recall (MQAR), the task the "Zoology" line of work
//! uses to measure a recurrent layer's memory capacity. Each sequence is
//!
//! ```text
//! k1 v1 k2 v2 ... kn vn  SEP  q1 a1 q2 a2 ... qm am
//! ```
//!
//! where every `q` is one of the keys and its answer `a` is the value that was
//! paired with it. Keys and values are disjoint token ranges, the keys of one
//! sequence are distinct, and the loss is taken *only* at the query positions
//! (everything else is the ignore index). Nothing here can be solved by unigram
//! or positional statistics: the model has to store `n` key/value bindings in
//! its state and read them back on demand.
//!
//! Several stack shapes are trained on identical batches and compared on
//! held-out query accuracy, wall-clock, parameter count and per-layer decode
//! state. Environment knobs: `PAIRS` (bindings per sequence, default 16),
//! `STEPS` (default 400), `BATCH` (default 8), `SEED` (default 1234), `ONLY`
//! (comma-separated variant-name prefixes to run).
//!
//! ```text
//! cargo run --release --example recall_probe
//! PAIRS=24 cargo run --release --example recall_probe
//! ```

use std::time::Instant;

use mamba3::nn::attention::AttentionConfig;
use mamba3::prelude::*;
use mamba3::tensor::ops::index::IdTensor;
use mamba3::tensor::ops::random::Rng;
use mamba3::train::{CrossEntropyConfig, LmBatch, LmTask};

type R = mamba3::backends::Auto;

const N_KEYS: u32 = 32;
const N_VALS: u32 = 32;
const SEP: u32 = N_KEYS + N_VALS;
const IGNORE: u32 = SEP + 1;
const VOCAB: usize = (IGNORE + 1) as usize;

const D_MODEL: usize = 64;
const N_LAYERS: usize = 2;
const N_HEADS: usize = 4;

fn env(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

struct Task {
    pairs: usize,
    queries: usize,
}

impl Task {
    fn seq_len(&self) -> usize {
        2 * self.pairs + 1 + 2 * self.queries
    }

    /// One batch. Keys within a sequence are distinct; queries are a random
    /// subset of them, so every answer is determined by the context alone.
    fn batch(&self, device: &Device<R>, rng: &mut Rng, batch: usize) -> LmBatch<R> {
        let seq = self.seq_len();
        let mut inputs = Vec::with_capacity(batch * seq);
        let mut targets = Vec::with_capacity(batch * seq);
        for _ in 0..batch {
            // Distinct keys: a partial Fisher-Yates over the key range.
            let mut keys: Vec<u32> = (0..N_KEYS).collect();
            for i in 0..self.pairs {
                let j = i + rng.next_index(keys.len() - i);
                keys.swap(i, j);
            }
            let keys = &keys[..self.pairs];
            let values: Vec<u32> = (0..self.pairs)
                .map(|_| N_KEYS + rng.next_index(N_VALS as usize) as u32)
                .collect();

            let mut tokens = Vec::with_capacity(seq + 1);
            for (k, v) in keys.iter().zip(&values) {
                tokens.push(*k);
                tokens.push(*v);
            }
            tokens.push(SEP);
            // A random permutation of the bindings, truncated to `queries`.
            let mut order: Vec<usize> = (0..self.pairs).collect();
            for i in 0..self.queries {
                let j = i + rng.next_index(order.len() - i);
                order.swap(i, j);
            }
            for &i in &order[..self.queries] {
                tokens.push(keys[i]);
                tokens.push(values[i]);
            }
            debug_assert_eq!(tokens.len(), seq + 1);

            // Next-token format; only the position that reads a query predicts.
            let prefix = 2 * self.pairs + 1;
            for t in 0..seq {
                inputs.push(tokens[t]);
                let is_query = t >= prefix && (t - prefix) % 2 == 0;
                targets.push(if is_query { tokens[t + 1] } else { IGNORE });
            }
        }
        LmBatch {
            inputs: IdTensor::from_slice(&inputs, vec![batch, seq], device).unwrap(),
            targets: IdTensor::from_slice(&targets, vec![batch, seq], device).unwrap(),
        }
    }
}

/// Accuracy over the query positions only.
fn query_accuracy(logits: &Var<R, f32>, targets: &IdTensor<R>) -> Result<(usize, usize)> {
    let last = logits.rank() - 1;
    let predicted = mamba3::tensor::ops::reduce::argmax(logits.tensor(), last)?.to_vec();
    let expected = targets.to_vec();
    let mut hits = 0;
    let mut total = 0;
    for (p, e) in predicted.iter().zip(&expected) {
        if *e == IGNORE {
            continue;
        }
        total += 1;
        if p == e {
            hits += 1;
        }
    }
    Ok((hits, total))
}

struct Variant {
    name: &'static str,
    config: Mamba3LmConfig,
    /// Elements one layer carries while decoding.
    decode_state: usize,
}

fn ssm_variant(
    name: &'static str,
    f: impl FnOnce(&mut SsmConfig),
    pattern: LayerPattern,
) -> Result<Variant> {
    let config = Mamba3LmConfig::builder()
        .vocab_size(VOCAB)
        .d_model(D_MODEL)
        .n_layers(N_LAYERS)
        .pattern(pattern)
        .attention(AttentionConfig::new(D_MODEL, N_HEADS).with_causal(true))
        .with_ssm(|s| {
            s.n_heads = N_HEADS;
            s.n_groups = N_HEADS;
            s.head_dim = D_MODEL / N_HEADS;
            s.d_state = 16;
            s.chunk_size = 16;
            f(s);
        })
        .seed(env("SEED", 1234) as u64)
        .build()?;
    let s = &config.stack.ssm;
    // h + last_u, plus the running angle for rotational dynamics.
    let decode_state = 2 * s.n_heads * s.head_dim * s.d_state
        + if s.dynamics == StateDynamics::Rotational {
            s.n_heads * s.d_state / 2
        } else {
            0
        };
    Ok(Variant {
        name,
        config,
        decode_state,
    })
}

fn variants() -> Result<Vec<Variant>> {
    Ok(vec![
        ssm_variant("N16 G4 (base)", |_| {}, LayerPattern::AllMamba)?,
        ssm_variant("N32 G4", |s| s.d_state = 32, LayerPattern::AllMamba)?,
        ssm_variant("N64 G4", |s| s.d_state = 64, LayerPattern::AllMamba)?,
        ssm_variant(
            "N64 G1",
            |s| {
                s.d_state = 64;
                s.n_groups = 1;
            },
            LayerPattern::AllMamba,
        )?,
        ssm_variant(
            "N128 G1",
            |s| {
                s.d_state = 128;
                s.n_groups = 1;
            },
            LayerPattern::AllMamba,
        )?,
        ssm_variant(
            "MIMO R2 P8 N16",
            |s| {
                s.head_dim = 8;
                s.mode = SsmMode::Mimo { rank: 2 };
            },
            LayerPattern::AllMamba,
        )?,
        ssm_variant(
            "MIMO R2 P8 N32",
            |s| {
                s.head_dim = 8;
                s.d_state = 32;
                s.mode = SsmMode::Mimo { rank: 2 };
            },
            LayerPattern::AllMamba,
        )?,
        ssm_variant(
            "N16 real (no rotation)",
            |s| s.dynamics = StateDynamics::Real,
            LayerPattern::AllMamba,
        )?,
        ssm_variant(
            "N64 G1 real",
            |s| {
                s.d_state = 64;
                s.n_groups = 1;
                s.dynamics = StateDynamics::Real;
            },
            LayerPattern::AllMamba,
        )?,
        ssm_variant(
            "MIMO R2 P8 N16 real",
            |s| {
                s.head_dim = 8;
                s.mode = SsmMode::Mimo { rank: 2 };
                s.dynamics = StateDynamics::Real;
            },
            LayerPattern::AllMamba,
        )?,
        ssm_variant(
            "N16 Euler real (Mamba-2)",
            |s| {
                s.dynamics = StateDynamics::Real;
                s.discretization = Discretization::Euler;
            },
            LayerPattern::AllMamba,
        )?,
        ssm_variant("hybrid attn@1", |_| {}, LayerPattern::AttentionAt(vec![1]))?,
    ])
}

struct Outcome {
    name: &'static str,
    params: usize,
    decode_state: usize,
    final_loss: f32,
    accuracy: f32,
    seconds: f32,
}

fn run(
    variant: Variant,
    task: &Task,
    device: &Device<R>,
    steps: u64,
    batch: usize,
) -> Result<Outcome> {
    let model = variant.config.init::<R, f32>(device)?;
    let params = model.num_parameters();

    let trainer_config = TrainerConfig::builder()
        .learning_rate(3e-3)
        .schedule(LrSchedule::cosine(steps))
        .max_grad_norm(1.0)
        .max_steps(steps)
        .log_every(100)
        .build()?;
    let name = variant.name;
    let mut trainer = Trainer::new(
        trainer_config,
        AdamWConfig::builder()
            .learning_rate(3e-3)
            .weight_decay(0.01)
            .build()
            .init::<R, f32>(),
    )
    .on_step(move |info| {
        println!("  {name:<24} step {:>4}  loss {:.4}", info.step, info.loss);
    });

    // Same data stream for every variant.
    let mut rng = Rng::seeded(99);
    let batches: Vec<LmBatch<R>> = (0..steps)
        .map(|_| task.batch(device, &mut rng, batch))
        .collect();

    let lm_task =
        LmTask::new(&model).with_loss(CrossEntropyConfig::default().with_ignore_index(IGNORE));
    let started = Instant::now();
    let report = trainer.fit(&lm_task, batches)?;
    device.synchronize();
    let seconds = started.elapsed().as_secs_f32();

    model.eval();
    let mut hits = 0;
    let mut total = 0;
    for _ in 0..4 {
        let eval = task.batch(device, &mut rng, batch);
        let logits = model.forward(&eval.inputs, false)?;
        let (h, t) = query_accuracy(&logits, &eval.targets)?;
        hits += h;
        total += t;
    }

    Ok(Outcome {
        name,
        params,
        decode_state: variant.decode_state,
        final_loss: report.final_loss,
        accuracy: hits as f32 / total.max(1) as f32,
        seconds,
    })
}

fn main() -> Result<()> {
    mamba3::tensor::ops::matmul::try_set_precision_from_env::<R>()?;
    let device = Device::<R>::default();
    println!("backend: {}", device.name());

    let pairs = env("PAIRS", 16);
    let task = Task {
        pairs,
        queries: pairs.saturating_sub(1).max(1),
    };
    let steps = env("STEPS", 400) as u64;
    let batch = env("BATCH", 8);
    let only: Option<Vec<String>> = std::env::var("ONLY")
        .ok()
        .map(|v| v.split(',').map(|s| s.trim().to_string()).collect());

    println!(
        "task: {pairs} bindings, {} queries, seq {} tokens, {steps} steps x {batch} sequences",
        task.queries,
        task.seq_len()
    );

    let mut outcomes = Vec::new();
    for variant in variants()? {
        if let Some(only) = &only
            && !only.iter().any(|o| variant.name.starts_with(o.as_str()))
        {
            continue;
        }
        println!("\n== {} ==", variant.name);
        outcomes.push(run(variant, &task, &device, steps, batch)?);
    }

    println!(
        "\n{:<24} {:>8} {:>12} {:>9} {:>10} {:>8}",
        "variant", "params", "state/layer", "loss@end", "query acc", "time"
    );
    for o in &outcomes {
        println!(
            "{:<24} {:>8} {:>12} {:>9.4} {:>9.1}% {:>7.1}s",
            o.name,
            o.params,
            o.decode_state,
            o.final_loss,
            o.accuracy * 100.0,
            o.seconds
        );
    }
    Ok(())
}
