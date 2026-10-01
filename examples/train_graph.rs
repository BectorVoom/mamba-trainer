//! Train a Graph Mamba model on one of two synthetic tasks and print what a
//! training run costs: loss, the task's metric, milliseconds and kernel
//! launches per step.
//!
//! ```text
//! cargo run --release --example train_graph
//! MAMBA3_GRAPH_TASK=triangles cargo run --release --example train_graph
//! ```
//!
//! * `majority` (default): one random graph; each node predicts whether most
//!   of its neighbours carry bit 1. Node classification on node partitions;
//!   it needs the per-node token stage (`MAMBA3_GRAPH_HOPS=0` cannot learn it).
//! * `triangles`: many random 20-node graphs; each graph's triangle count is
//!   regressed. Graph regression on whole-graph batches.
//!
//! Overrides, all optional: `MAMBA3_GRAPH_HOPS` (`m`), `_WALKS` (`M`),
//! `_REPEATS` (`s`), `_TOKEN_LAYERS`, `_NODE_LAYERS`, `_LOCAL` (`mean` |
//! `sgc`), `_MPNN` (`none` | `gine` | `gated_gcn`), `_PARTS` (node parts of the
//! `majority` graph; 1 is full batch), `_BATCH_ROWS` (row budget of the
//! `triangles` batches), `_D_MODEL`, `_NODES` (size of the `majority` graph),
//! `_GRAPHS` (number of `triangles` graphs), `_EPOCHS`, `_LR`.

use std::time::Instant;

use mamba3::backend::{launch_count, reset_launch_count};
use mamba3::models::graph::{
    Epoch, EvalOptions, FeatureSpec, Features, GraphData, GraphDataset, GraphMamba,
    GraphMambaSpec, GraphPool, GraphTaskSpec, GraphTrainConfig, GraphTrainer, Labels,
    LocalEncoder, Metric, MpnnKind, RegressionLoss, Split, Splits,
};
use mamba3::prelude::*;

type R = mamba3::backends::Auto;

fn env<T: std::str::FromStr>(key: &str, fallback: T) -> T {
    std::env::var(format!("MAMBA3_GRAPH_{key}"))
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(fallback)
}

fn env_str(key: &str, fallback: &str) -> String {
    std::env::var(format!("MAMBA3_GRAPH_{key}")).unwrap_or_else(|_| fallback.to_string())
}

/// A small deterministic generator.
struct Lcg(u64);

impl Lcg {
    fn below(&mut self, n: usize) -> usize {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as usize % n
    }
}

/// A 60 / 20 / 20 split of `items`.
fn splits(items: usize, seed: u64) -> Splits {
    let mut rng = Lcg(seed);
    let draw: Vec<usize> = (0..items).map(|_| rng.below(10)).collect();
    Splits {
        train: draw.iter().map(|&d| d < 6).collect(),
        val: draw.iter().map(|&d| (6..8).contains(&d)).collect(),
        test: draw.iter().map(|&d| d >= 8).collect(),
    }
}

/// One random graph of `n` nodes with mean degree 6, a random bit per node,
/// and the label "more than half of my neighbours have bit 1" (ties
/// unlabelled).
fn neighbour_majority(n: usize) -> GraphData {
    let mut rng = Lcg(2024);
    let (mut src, mut dst) = (Vec::new(), Vec::new());
    let mut neighbours = vec![std::collections::BTreeSet::new(); n];
    for _ in 0..n * 3 {
        let (a, b) = (rng.below(n), rng.below(n));
        if a != b {
            src.extend([a as u32, b as u32]);
            dst.extend([b as u32, a as u32]);
            neighbours[a].insert(b);
            neighbours[b].insert(a);
        }
    }
    let bits: Vec<usize> = (0..n).map(|_| rng.below(2)).collect();
    let labels = (0..n)
        .map(|v| {
            let ones: usize = neighbours[v].iter().map(|&u| bits[u]).sum();
            match (2 * ones).cmp(&neighbours[v].len()) {
                std::cmp::Ordering::Greater => 1,
                std::cmp::Ordering::Less => 0,
                std::cmp::Ordering::Equal => -1,
            }
        })
        .collect();
    let mut data = GraphData::new(
        n,
        src,
        dst,
        Features::Float {
            dim: 1,
            data: bits.iter().map(|&b| b as f32 * 2.0 - 1.0).collect(),
        },
    );
    data.y = Labels::Node(labels);
    data.masks = Some(splits(n, 7));
    data
}

/// `graphs` random 20-node graphs with constant features; the target is the
/// triangle count over its standard deviation across the dataset.
fn triangle_counts(graphs: usize) -> GraphData {
    let n = 20usize;
    let mut rng = Lcg(5);
    let (mut src, mut dst, mut ptr) = (Vec::new(), Vec::new(), vec![0u32]);
    let mut counts = Vec::with_capacity(graphs);
    for g in 0..graphs {
        let base = g * n;
        let mut adjacent = vec![false; n * n];
        let density = 15 + rng.below(25);
        for a in 0..n {
            for b in a + 1..n {
                if rng.below(100) < density {
                    adjacent[a * n + b] = true;
                    adjacent[b * n + a] = true;
                    src.extend([(base + a) as u32, (base + b) as u32]);
                    dst.extend([(base + b) as u32, (base + a) as u32]);
                }
            }
        }
        let mut triangles = 0usize;
        for a in 0..n {
            for b in a + 1..n {
                for c in b + 1..n {
                    triangles +=
                        (adjacent[a * n + b] && adjacent[b * n + c] && adjacent[a * n + c]) as usize;
                }
            }
        }
        counts.push(triangles as f32);
        ptr.push((base + n) as u32);
    }
    let mean = counts.iter().sum::<f32>() / graphs as f32;
    let std = (counts.iter().map(|c| (c - mean).powi(2)).sum::<f32>() / graphs as f32).sqrt();
    let total = graphs * n;
    let mut data = GraphData::new(
        total,
        src,
        dst,
        Features::Float {
            dim: 1,
            data: vec![1.0; total],
        },
    );
    data.graph_ptr = ptr;
    data.y = Labels::Graph {
        targets: 1,
        values: counts.iter().map(|c| c / std).collect(),
    };
    data.masks = Some(splits(graphs, 11));
    data
}

fn main() -> Result<()> {
    let task_name = env_str("TASK", "majority");
    let majority = match task_name.as_str() {
        "majority" => true,
        "triangles" => false,
        other => {
            return Err(Error::config(format!(
                "MAMBA3_GRAPH_TASK: unknown task {other:?}; expected majority or triangles"
            )));
        }
    };
    let (task, metric, data) = if majority {
        (
            GraphTaskSpec::NodeClass { classes: 2 },
            Metric::Accuracy,
            neighbour_majority(env("NODES", 2000)),
        )
    } else {
        (
            GraphTaskSpec::GraphRegression {
                targets: 1,
                pool: GraphPool::Mean,
                loss: RegressionLoss::L1,
            },
            Metric::Mae,
            triangle_counts(env("GRAPHS", 500)),
        )
    };
    let local = match env_str("LOCAL", if majority { "mean" } else { "sgc" }).as_str() {
        "mean" => LocalEncoder::Mean,
        "sgc" => LocalEncoder::Sgc { hops: 1 },
        other => {
            return Err(Error::config(format!(
                "MAMBA3_GRAPH_LOCAL: unknown encoder {other:?}; expected mean or sgc"
            )));
        }
    };
    let mpnn = match env_str("MPNN", if majority { "none" } else { "gine" }).as_str() {
        "none" => None,
        "gine" => Some(MpnnKind::Gine),
        "gated_gcn" => Some(MpnnKind::GatedGcn),
        other => {
            return Err(Error::config(format!(
                "MAMBA3_GRAPH_MPNN: unknown layer {other:?}; expected none, gine or gated_gcn"
            )));
        }
    };
    let hops = env("HOPS", if majority { 1 } else { 2 });
    let mut spec = GraphMambaSpec::new(FeatureSpec::Float { dim: 1 }, task)
        .with_d_model(env("D_MODEL", 64))
        .with_tokens(hops, env("WALKS", if majority { 16 } else { 8 }), env("REPEATS", 2))
        .with_local(local)
        .with_mpnn(mpnn)
        .with_node_layers(env("NODE_LAYERS", if majority { 1 } else { 2 }));
    if hops > 0 {
        spec = spec.with_token_layers(env("TOKEN_LAYERS", 1));
    }

    let device = Device::<R>::default();
    let dataset = GraphDataset::<R, f32>::new(&spec, data, &device)?;
    let model = GraphMamba::<R, f32>::init(&spec, &device)?;
    let mut trainer = GraphTrainer::new(&GraphTrainConfig {
        learning_rate: env("LR", 3e-3),
        ..Default::default()
    })?;
    let parts: Option<usize> = std::env::var("MAMBA3_GRAPH_PARTS")
        .ok()
        .and_then(|v| v.parse().ok());
    let batch_rows: Option<usize> = std::env::var("MAMBA3_GRAPH_BATCH_ROWS")
        .ok()
        .and_then(|v| v.parse().ok());
    let epoch_of = |epoch: u64| -> Result<Epoch<R, f32>> {
        if majority {
            dataset.epoch_nodes(parts.or(Some(4)), epoch, Split::Train)
        } else {
            dataset.epoch_graphs(batch_rows.or(Some(2048)), epoch, Split::Train)
        }
    };
    let options = EvalOptions {
        batch_rows: batch_rows.or(Some(2048)),
        parts: parts.or(Some(4)),
    };
    let estimate = dataset.memory_estimate(
        (!majority).then_some(options.batch_rows).flatten(),
        majority.then_some(options.parts).flatten(),
    );
    println!(
        "{} on {}: {} nodes in {} graphs, {} parameters; {} rows per batch, about {:.0} MiB live \
         per step",
        task_name,
        device.name(),
        dataset.num_nodes(),
        dataset.num_graphs(),
        model.num_parameters(),
        estimate.rows,
        estimate.live as f64 / (1 << 20) as f64,
    );

    let epochs: u64 = env("EPOCHS", 30);
    for epoch in 0..epochs {
        let plan = epoch_of(epoch)?;
        reset_launch_count();
        let started = Instant::now();
        let steps = model.train_epoch(&mut trainer, &plan)?;
        // One read for every loss of the epoch; it also drains the queue.
        let losses = trainer.read_losses()?;
        let elapsed = started.elapsed().as_secs_f64() * 1e3;
        let launches = launch_count();
        let loss = losses.iter().map(|info| info.loss).sum::<f32>() / losses.len() as f32;
        let val = model.evaluate(&dataset, Split::Val, metric, &options)?;
        println!(
            "epoch {epoch:>3}  loss {loss:.4}  val {} {:.4}  {:.1} ms/step  {} launches/step",
            metric.name(),
            val.value,
            elapsed / steps as f64,
            launches / steps,
        );
    }
    let test = model.evaluate(&dataset, Split::Test, metric, &options)?;
    println!(
        "test {} {:.4} on {} targets",
        metric.name(),
        test.value,
        test.count
    );
    Ok(())
}
