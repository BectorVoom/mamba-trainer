//! Where a Graph Mamba training step goes: launches by model region, the
//! clocks that tell a host-bound step from a device-bound one, transfers, the
//! shapes the matmul tuner has to learn, and memory.
//!
//! ```text
//! cargo run --release --example profile_graph
//! MAMBA3_GRAPH_WORKLOAD=a cargo run --release --no-default-features --features wgpu \
//!     --example profile_graph
//! ```
//!
//! Two reference workloads (GRAPH_MAMBA_PLAN.md §2.7), both at `d = 64`,
//! `m = 4`, `s = 4`, `M = 8`:
//!
//! * `b` (default) — many small graphs: a row budget of 4,800 over graphs of
//!   about 150 nodes, message passing with GINE.
//! * `a` — one large graph of 22,662 nodes with 300 features, in 4 node parts.
//!
//! `MAMBA3_GRAPH_SCALE` (default 1.0) scales the node counts down for a quick
//! run. The switches of the design, for A/B runs inside one process each:
//! `MAMBA3_GRAPH_TAIL` (`forward` | `bidirectional`), `_D_STATE`, `_LOCAL`
//! (`sgc` | `mean`), `_BUCKET` (`1` | `0`), `_SAMPLING` (`step` | `epoch` |
//! `static`), `_MPNN` (`gine` | `gated_gcn` | `none`), `_SEQUENCES` (workload
//! `a`), `_HEADS`. `MAMBA3_TIME_LAUNCHES=1` adds a drained time per region
//! (it serialises the queue: read the split, not the total). `_STEPS` sets the
//! timed steps.
//!
//! Three clocks, never mixed: **host submit** is the time `queue_step` takes
//! to return; **drained step** is queue plus one read; device time per kernel
//! comes from CubeCL's own profiler (`cubecl.toml`, `[profiling]`).

use std::collections::BTreeMap;
use std::time::Instant;

use mamba3::backend::{
    launch_count, launch_tally_detailed, launch_time_tally, meta_miss_count, peak_alloc_bytes,
    read_count, reserved_bytes, reset_launch_count, reset_launch_tally, reset_meta_miss_count,
    reset_peak_alloc, reset_read_count, reset_upload_count, set_launch_timer, start_launch_tally,
    stop_launch_tally, upload_count,
};
use mamba3::models::graph::{
    Epoch, EpochOptions, FeatureSpec, Features, GraphData, GraphDataset, GraphMamba,
    GraphMambaSpec, GraphPool, GraphTask, GraphTaskSpec, Labels, LocalEncoder, MpnnKind, Split,
    Splits, TokenSampling, TokenTail,
};
use mamba3::prelude::*;
use mamba3::tensor::ops::matmul::{
    reset_tune_miss_count, start_matmul_log, take_matmul_log, tune_miss_count, tuned_shape_count,
};
use mamba3::train::{AdamW, TrainStep, Trainer, TrainerConfig};

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

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }

    fn below(&mut self, n: usize) -> usize {
        self.next() as usize % n
    }

    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n)
            .map(|_| self.next() as f32 / (1u64 << 30) as f32 - 1.0)
            .collect()
    }
}

/// Graphs of the given sizes with about three random edges per node.
fn graphs(sizes: &[usize], dim: usize, seed: u64) -> GraphData {
    let mut rng = Lcg(seed);
    let (mut src, mut dst, mut ptr) = (Vec::new(), Vec::new(), vec![0u32]);
    let mut at = 0usize;
    for &size in sizes {
        for _ in 0..3 * size {
            let (a, b) = (at + rng.below(size), at + rng.below(size));
            if a != b {
                src.extend([a as u32, b as u32]);
                dst.extend([b as u32, a as u32]);
            }
        }
        at += size;
        ptr.push(at as u32);
    }
    let mut data = GraphData::new(
        at,
        src,
        dst,
        Features::Float {
            dim,
            data: rng.vec(at * dim),
        },
    );
    data.graph_ptr = ptr;
    data
}

/// A read is the only operation guaranteed to wait for every queued kernel.
fn drain(device: &Device<R>) {
    device.synchronize();
    let _ = Tensor::<R, f32>::zeros(vec![1], device).to_data();
}

fn mib(bytes: u64) -> f64 {
    bytes as f64 / (1 << 20) as f64
}

fn main() -> Result<()> {
    let workload = env_str("WORKLOAD", "b");
    let scale: f64 = env("SCALE", 1.0);
    let large = match workload.as_str() {
        "a" => true,
        "b" => false,
        other => {
            return Err(Error::config(format!(
                "MAMBA3_GRAPH_WORKLOAD: unknown workload {other:?}; expected a or b"
            )));
        }
    };
    let pick = |key: &str, fallback: &str, options: &[&str]| -> Result<String> {
        let value = env_str(key, fallback);
        if options.contains(&value.as_str()) {
            Ok(value)
        } else {
            Err(Error::config(format!(
                "MAMBA3_GRAPH_{key}: unknown value {value:?}; expected one of {options:?}"
            )))
        }
    };
    let tail = pick("TAIL", "forward", &["forward", "bidirectional"])?;
    let local = pick("LOCAL", "sgc", &["sgc", "mean"])?;
    let sampling = pick("SAMPLING", "step", &["step", "epoch", "static"])?;
    let mpnn = pick("MPNN", "gine", &["gine", "gated_gcn", "none"])?;
    let bucket = env("BUCKET", 1usize) == 1;

    let (task, data, dim) = if large {
        let n = ((22_662.0 * scale) as usize).max(64);
        let mut data = graphs(&[n], 300, 1);
        data.y = Labels::Node((0..n).map(|i| (i % 18) as i64).collect());
        data.masks = Some(Splits {
            train: vec![true; n],
            val: vec![false; n],
            test: vec![false; n],
        });
        (GraphTaskSpec::NodeClass { classes: 18 }, data, 300)
    } else {
        // Sizes around 150, so a 4,800-row batch holds about 32 graphs.
        let count = ((320.0 * scale) as usize).max(16);
        let mut rng = Lcg(3);
        let sizes: Vec<usize> = (0..count).map(|_| 60 + rng.below(181)).collect();
        let mut data = graphs(&sizes, 16, 2);
        data.y = Labels::GraphClass((0..count as u32).map(|i| i % 10).collect());
        data.masks = Some(Splits {
            train: vec![true; count],
            val: vec![false; count],
            test: vec![false; count],
        });
        (
            GraphTaskSpec::GraphClass {
                classes: 10,
                pool: GraphPool::Mean,
            },
            data,
            16,
        )
    };
    let spec = GraphMambaSpec::new(FeatureSpec::Float { dim }, task)
        .with_tokens(4, 8, 4)
        .with_local(if local == "sgc" {
            LocalEncoder::Sgc { hops: 1 }
        } else {
            LocalEncoder::Mean
        })
        .with_token_tail(if tail == "forward" {
            TokenTail::Forward
        } else {
            TokenTail::Bidirectional
        })
        .with_token_sampling(match sampling.as_str() {
            "step" => TokenSampling::PerStep,
            "epoch" => TokenSampling::PerEpoch,
            _ => TokenSampling::Static,
        })
        .with_mpnn(match mpnn.as_str() {
            "gine" => Some(MpnnKind::Gine),
            "gated_gcn" => Some(MpnnKind::GatedGcn),
            _ => None,
        })
        .with_d_state(env("D_STATE", 8))
        .with_token_heads(env("HEADS", 1))
        .with_node_heads(env("HEADS", 1))
        .with_node_sequences(if large { env("SEQUENCES", 1) } else { 1 });

    let device = Device::<R>::default();
    let dataset = GraphDataset::<R, f32>::new(&spec, data, &device)?;
    let model = GraphMamba::<R, f32>::init(&spec, &device)?;
    let rows = ((4800.0 * scale) as usize).max(256);
    let epoch_of = |epoch: u64| -> Result<Epoch<R, f32>> {
        if large {
            dataset.epoch_nodes(Some(4), epoch, Split::Train)
        } else {
            dataset.epoch_graphs_with(
                Some(rows),
                epoch,
                Some(Split::Train),
                EpochOptions {
                    shuffle: true,
                    bucket,
                },
            )
        }
    };
    let estimate = if large {
        dataset.memory_estimate(None, Some(4))
    } else {
        dataset.memory_estimate(Some(rows), None)
    };
    println!(
        "workload {workload} on {}: {} nodes, {} graphs, {} parameters",
        device.name(),
        dataset.num_nodes(),
        dataset.num_graphs(),
        model.num_parameters()
    );
    println!(
        "spec: tail {tail}, local {local}, sampling {sampling}, mpnn {mpnn}, d_state {}, \
         heads/direction {}, buckets {bucket}",
        spec.token_ssm.d_state, spec.token_ssm.n_heads
    );
    println!(
        "estimate for {} rows: store {:.1} MiB, parameters {:.1} MiB, live {:.1} MiB, largest \
         allocation {:.1} MiB (threshold {:.1} MiB)",
        estimate.rows,
        mib(estimate.store as u64),
        mib(estimate.parameters as u64),
        mib(estimate.live as u64),
        mib(estimate.largest_allocation as u64),
        mib(estimate.threshold as u64),
    );

    let task = GraphTask::new(&model);
    let mut trainer = Trainer::new(TrainerConfig::default(), AdamW::<R, f32>::new(1e-3));

    // --- three epochs: what the first costs, and what stays -----------------
    for epoch in 0..3u64 {
        reset_upload_count();
        reset_read_count();
        reset_meta_miss_count();
        reset_tune_miss_count();
        reset_launch_count();
        let started = Instant::now();
        let plan = epoch_of(epoch)?;
        let mut queued = Vec::with_capacity(plan.len());
        for index in 0..plan.len() {
            let batch = plan.batch(index)?;
            queued.push(trainer.queue_step(&task, std::slice::from_ref(&batch))?);
        }
        let reads_before = read_count();
        let infos = trainer.read_steps(&queued)?;
        println!(
            "epoch {epoch}: {} steps, {:.1} ms/step drained, {} launches/step | uploads {} | \
             reads while queueing {} | tuner misses {} (shapes held {}) | metadata misses {} | \
             loss {:.4}",
            plan.len(),
            started.elapsed().as_secs_f64() * 1e3 / plan.len() as f64,
            launch_count() / plan.len(),
            upload_count(),
            reads_before,
            tune_miss_count(),
            tuned_shape_count(),
            meta_miss_count(),
            infos.last().map_or(f32::NAN, |info| info.loss),
        );
        println!(
            "         reserved {:.0} MiB after {} steps",
            reserved_bytes(&device).map_or(f64::NAN, mib),
            trainer.step_count()
        );
    }

    // --- one step's matrix products: the tuner's keys -----------------------
    let plan = epoch_of(3)?;
    let batch = plan.batch(0)?;
    drain(&device);
    start_matmul_log();
    let queued = trainer.queue_step(&task, std::slice::from_ref(&batch))?;
    let products = take_matmul_log();
    trainer.read_steps(&[queued])?;
    let mut keys: BTreeMap<(usize, usize, usize, usize, bool, bool), usize> = BTreeMap::new();
    for p in &products {
        *keys.entry((p.batch, p.m, p.n, p.k, p.lhs_t, p.rhs_t)).or_insert(0) += 1;
    }
    let mut row_counts: Vec<usize> = products.iter().map(|p| p.batch * p.m).collect();
    row_counts.sort_unstable();
    row_counts.dedup();
    println!(
        "one step: {} matrix products of {} distinct shapes; row counts {:?}",
        products.len(),
        keys.len(),
        row_counts
    );
    println!(
        "batch: {} rows, {} graph slots, padded length {}, {} edge rows",
        batch.rows,
        batch.graphs,
        batch.lengths.max(),
        batch.edges
    );

    // --- host submit against drained step -----------------------------------
    let steps: usize = env("STEPS", 6);
    let (mut submit, mut drained) = (Vec::with_capacity(steps), Vec::with_capacity(steps));
    for _ in 0..steps {
        drain(&device);
        let started = Instant::now();
        let queued = trainer.queue_step(&task, std::slice::from_ref(&batch))?;
        submit.push(started.elapsed().as_secs_f64() * 1e3);
        trainer.read_steps(&[queued])?;
        drained.push(started.elapsed().as_secs_f64() * 1e3);
    }
    let summary = |times: &mut Vec<f64>| {
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        (times[0], times[times.len() / 2], times[times.len() - 1])
    };
    let (s_min, s_med, s_max) = summary(&mut submit);
    let (d_min, d_med, d_max) = summary(&mut drained);
    println!(
        "host submit   min {s_min:.1} median {s_med:.1} max {s_max:.1} ms ({steps} steps of the \
         same batch)"
    );
    println!("drained step  min {d_min:.1} median {d_med:.1} max {d_max:.1} ms");
    println!(
        "host submit is {:.0}% of the drained step: {}",
        100.0 * s_med / d_med,
        if s_med > 0.8 * d_med {
            "the step is host-bound (or the runtime computes on the calling thread)"
        } else {
            "the device works after the host is done"
        }
    );

    // --- launches by region --------------------------------------------------
    drain(&device);
    reset_launch_tally();
    start_launch_tally();
    let timed = std::env::var_os("MAMBA3_TIME_LAUNCHES").is_some();
    if timed {
        let dev = device.clone();
        let probe = Tensor::<R, f32>::zeros(vec![1], &device);
        set_launch_timer(Some(Box::new(move || {
            dev.synchronize();
            let _ = probe.to_data();
        })));
    }
    reset_launch_count();
    let queued = trainer.queue_step(&task, std::slice::from_ref(&batch))?;
    if timed {
        mamba3::backend::flush_launch_timer();
    }
    stop_launch_tally();
    let total = launch_count();
    trainer.read_steps(&[queued])?;
    let mut by_label: BTreeMap<String, usize> = BTreeMap::new();
    for row in launch_tally_detailed() {
        *by_label.entry(row.label).or_insert(0) += row.count;
    }
    let mut labels: Vec<(String, usize)> = by_label.into_iter().collect();
    labels.sort_by(|a, b| b.1.cmp(&a.1));
    println!("launches of one step by region ({total} in all):");
    for (label, count) in &labels {
        println!("  {count:>5}  {:>5.1}%  {label}", 100.0 * *count as f64 / total as f64);
    }
    if timed {
        let mut ms: BTreeMap<String, f64> = BTreeMap::new();
        for (site, time) in launch_time_tally() {
            let label = site.split(" / ").next().unwrap_or("-").to_string();
            *ms.entry(label).or_insert(0.0) += time;
        }
        let all: f64 = ms.values().sum();
        let mut ms: Vec<(String, f64)> = ms.into_iter().collect();
        ms.sort_by(|a, b| b.1.total_cmp(&a.1));
        println!("drained time of one step by region ({all:.1} ms with the queue serialised):");
        for (label, time) in &ms {
            println!("  {time:>8.2} ms  {:>5.1}%  {label}", 100.0 * time / all);
        }
        set_launch_timer(None);
    }

    // --- memory ---------------------------------------------------------------
    let in_use = || {
        drain(&device);
        device.client().memory_usage().ok().map(|u| u.bytes_in_use)
    };
    let before = in_use();
    reset_peak_alloc();
    let loss = task.loss(&batch)?;
    let live = in_use().zip(before).map(|(now, before)| now.saturating_sub(before));
    let forward_peak = peak_alloc_bytes();
    drop(loss.backward()?);
    drain(&device);
    println!(
        "memory: live after the forward pass {:.1} MiB (estimated {:.1}); largest allocation \
         {:.1} MiB forward, {:.1} MiB with the backward pass; reserved {:.0} MiB",
        live.map_or(f64::NAN, mib),
        mib(estimate.live as u64),
        mib(forward_peak as u64),
        mib(peak_alloc_bytes() as u64),
        reserved_bytes(&device).map_or(f64::NAN, mib),
    );
    Ok(())
}
