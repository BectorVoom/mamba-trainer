//! What the indirection costs in the two graph kernels that gather through an
//! index (GRAPH_MAMBA_PLAN.md GM11, protocol item 6): each against a baseline of
//! matched work with no index in it, candidates interleaved in one process.
//!
//! ```text
//! cargo run --release --example bench_graph_kernels
//! ROWS=4800 FEATURES=64 cargo run --release --no-default-features --features wgpu \
//!     --example bench_graph_kernels
//! ```
//!
//! * **`token_features`** — every output row is the weighted mean of the feature
//!   rows of one token's nodes, found through the token table. Baseline:
//!   `mean_dim` over a contiguous `[rows · L, k, F]` tensor, `k` being the mean
//!   number of nodes per token, so the same number of source rows and the same
//!   output.
//! * **`gine_aggregate`** — every output row sums `relu` of its neighbours' rows,
//!   found through the adjacency. Baseline: `sum_dim` over a contiguous
//!   `[rows, degree, d]` tensor of the same `nnz`.
//!
//! Each gather also runs on a *banded* graph, whose neighbours are the next rows
//! in memory, beside a *random* one of the same degree: same kernel, same
//! `nnz`, only the locality differs.
//!
//! `ROWS` (1200), `FEATURES` (64), `DEGREE` (6, even), `HOPS` / `WALKS` /
//! `REPEATS` (4 / 8 / 4), `REPS` launches per timing (10), `ROUNDS` (7). A
//! timing is `REPS` launches and one drain (a synchronisation and a one-element
//! read); the minimum over the rounds and the spread are reported.

use std::time::Instant;

use mamba3::models::graph::{
    CanonicalizeOptions, Features, GraphData, GraphStore, NodeOrder, canonicalize,
};
use mamba3::prelude::*;
use mamba3::tensor::ops::graph::{
    BatchRows, Tokens, WalkShape, batch_rows_subset, gine_aggregate, token_features, walk_tokens,
};
use mamba3::tensor::ops::IGNORE;
use mamba3::tensor::ops::reduce::{mean_dim, sum_dim};

type R = mamba3::backends::Auto;

fn env(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

struct Lcg(u64);

impl Lcg {
    fn below(&mut self, n: usize) -> usize {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as usize % n
    }

    fn unit(&mut self) -> f32 {
        self.below(1 << 20) as f32 / (1 << 19) as f32 - 1.0
    }
}

/// A graph on the device with its one whole-graph batch and its walk tokens.
struct Case {
    store: GraphStore<R, f32>,
    rows: BatchRows<R>,
    tokens: Tokens<R>,
    /// Token slots that hold a node, over all rows.
    filled: usize,
    nnz: usize,
}

/// `rows` nodes, each joined to `degree / 2` others: the next ones in memory
/// (`banded`) or random ones. Symmetrised, kept in the given order.
fn case(
    rows: usize,
    features: usize,
    degree: usize,
    banded: bool,
    shape: WalkShape,
    device: &Device<R>,
) -> Result<Case> {
    let mut rng = Lcg(if banded { 11 } else { 13 });
    let (mut src, mut dst) = (Vec::new(), Vec::new());
    for node in 0..rows {
        for k in 1..=degree / 2 {
            let other = if banded {
                (node + k) % rows
            } else {
                (node + 1 + rng.below(rows - 1)) % rows
            };
            src.push(node as u32);
            dst.push(other as u32);
        }
    }
    let data = GraphData::new(
        rows,
        src,
        dst,
        Features::Float {
            dim: features,
            data: (0..rows * features).map(|_| rng.unit()).collect(),
        },
    );
    let canon = canonicalize::<f32>(
        &data.view(),
        &CanonicalizeOptions {
            order: NodeOrder::Given,
            symmetrize: true,
            reverse_index: true,
        },
    )?;
    let nnz = canon.adj_col.len();
    let store = GraphStore::upload(canon, device)?;
    let batch = batch_rows_subset(store.adjacency(), 1, 0, (0, 0), 1)?;
    let tokens = walk_tokens(store.adjacency(), &batch, shape, (0x51, 0x52), 0)?;
    let filled = tokens.node().to_vec().iter().filter(|&&id| id != IGNORE).count();
    Ok(Case {
        store,
        rows: batch,
        tokens,
        filled,
        nnz,
    })
}

/// `reps` launches of `launch`, then a drain; milliseconds per launch.
fn time(
    device: &Device<R>,
    probe: &Tensor<R, f32>,
    reps: usize,
    launch: &dyn Fn() -> Result<Tensor<R, f32>>,
) -> Result<f64> {
    device.synchronize();
    probe.try_to_f32()?;
    let started = Instant::now();
    for _ in 0..reps {
        launch()?;
    }
    device.synchronize();
    probe.try_to_f32()?;
    Ok(started.elapsed().as_secs_f64() * 1e3 / reps as f64)
}

fn main() -> Result<()> {
    let device = Device::<R>::default();
    let rows = env("ROWS", 1200);
    let features = env("FEATURES", 64);
    let degree = env("DEGREE", 6);
    let shape = WalkShape {
        hops: env("HOPS", 4),
        walks: env("WALKS", 8),
        repeats: env("REPEATS", 4),
        sgc: true,
    };
    let (reps, rounds) = (env("REPS", 10), env("ROUNDS", 7));

    let random = case(rows, features, degree, false, shape, &device)?;
    let banded = case(rows, features, degree, true, shape, &device)?;
    let positions = rows * shape.len();
    // The contiguous baselines: as many source rows as the gathers read.
    let per_token = (random.filled as f64 / positions as f64).round().max(1.0) as usize;
    let per_node = (random.nnz as f64 / rows as f64).round().max(1.0) as usize;
    let mut rng = Lcg(17);
    let dense_tokens = Tensor::<R, f32>::from_f32(
        &(0..positions * per_token * features)
            .map(|_| rng.unit())
            .collect::<Vec<_>>(),
        vec![positions, per_token, features],
        &device,
    )?;
    let dense_edges = Tensor::<R, f32>::from_f32(
        &(0..rows * per_node * features)
            .map(|_| rng.unit())
            .collect::<Vec<_>>(),
        vec![rows, per_node, features],
        &device,
    )?;
    let u = Tensor::<R, f32>::from_f32(
        &(0..rows * features).map(|_| rng.unit()).collect::<Vec<_>>(),
        vec![rows, features],
        &device,
    )?;
    let probe = Tensor::<R, f32>::zeros(vec![1], &device);

    println!(
        "{} rows, {features} features, {} tokens per row of up to {} nodes; backend {}",
        rows,
        shape.len(),
        shape.cap(),
        device.name()
    );
    println!(
        "token slots filled: {} random, {} banded ({:.1} and {:.1} per token); baseline {} per \
         token, {} source rows",
        random.filled,
        banded.filled,
        random.filled as f64 / positions as f64,
        banded.filled as f64 / positions as f64,
        per_token,
        positions * per_token,
    );
    println!(
        "edges: {} random, {} banded; baseline {} per node, {} source rows",
        random.nnz,
        banded.nnz,
        per_node,
        rows * per_node
    );

    type Launch<'a> = Box<dyn Fn() -> Result<Tensor<R, f32>> + 'a>;
    let candidates: Vec<(&str, Launch<'_>)> = vec![
        (
            "token_features, random graph",
            Box::new(|| {
                token_features(&random.store.x().source(), None, &random.tokens, &random.rows, None)
            }),
        ),
        (
            "token_features, banded graph",
            Box::new(|| {
                token_features(&banded.store.x().source(), None, &banded.tokens, &banded.rows, None)
            }),
        ),
        (
            "contiguous mean, same source rows",
            Box::new(|| mean_dim(&dense_tokens, 1)),
        ),
        (
            "gine_aggregate, random graph",
            Box::new(|| gine_aggregate(&u, None, random.store.adjacency(), &random.rows)),
        ),
        (
            "gine_aggregate, banded graph",
            Box::new(|| gine_aggregate(&u, None, banded.store.adjacency(), &banded.rows)),
        ),
        (
            "contiguous sum, same nnz",
            Box::new(|| sum_dim(&dense_edges, 1)),
        ),
    ];

    // Warm: every kernel compiled and every buffer size seen.
    for (_, launch) in &candidates {
        time(&device, &probe, 2, launch.as_ref())?;
    }
    let mut times = vec![Vec::with_capacity(rounds); candidates.len()];
    for _ in 0..rounds {
        for (slot, (_, launch)) in times.iter_mut().zip(&candidates) {
            slot.push(time(&device, &probe, reps, launch.as_ref())?);
        }
    }
    let best: Vec<f64> = times
        .iter()
        .map(|t| t.iter().copied().fold(f64::INFINITY, f64::min))
        .collect();
    println!("\nms per launch, minimum of {rounds} interleaved rounds of {reps} (maximum):");
    for (index, ((name, _), t)) in candidates.iter().zip(&times).enumerate() {
        let worst = t.iter().copied().fold(0.0, f64::max);
        // Each group of three ends with its contiguous baseline.
        let baseline = best[index / 3 * 3 + 2];
        println!(
            "  {name:<36} {:>8.3} ({:>8.3})   {:>5.2}x the contiguous baseline",
            best[index],
            worst,
            best[index] / baseline
        );
    }
    Ok(())
}
