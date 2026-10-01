//! Time and device memory of one residual Mamba-3 block at the shapes Graph Mamba
//! runs it at (GRAPH_MAMBA_PLAN.md §2.7): forward plus backward of a `BiBlock` or
//! a `ForwardBlock` over `[ROWS, SEQ, D]`, with the input requiring a gradient, as
//! the token and node stages do.
//!
//! ```text
//! ROWS=4800 SEQ=17 HEADS=1 STATE=8 BIDIR=0 cargo run --release \
//!     --no-default-features --features wgpu --example bench_graph_blocks
//! ```
//!
//! `HEADS` is per direction (a bidirectional block doubles it, as `BiBlock` does).
//! `LAST=1` sums only the last position, the stage-1 read-out. `APPLY_LAST=1`
//! (forward block, one layer) compares that read-out done two ways, interleaved
//! in this one process: `apply` followed by a slice, and
//! `ForwardBlock::apply_last`. Two memory numbers:
//! "tape" is the bytes in use right after the forward pass, while every activation
//! the backward needs is still alive; "reserved" is the pool's high-water mark,
//! which is what a step needs to fit (it includes the backward's temporaries, the
//! matmul tuner's probes on a cold cache, and the pool's page rounding).

use std::time::Instant;

use mamba3::models::entity::blocks::{BiBlock, ForwardBlock};
use mamba3::nn::Param;
use mamba3::prelude::*;
use mamba3::ssm::config::SsmConfig;
use mamba3::tensor::ops::random::Rng;

type R = mamba3::backends::Auto;

fn env(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn mb(bytes: Option<u64>) -> f64 {
    bytes.map_or(f64::NAN, |b| b as f64 / (1024.0 * 1024.0))
}

enum Block {
    Bi(BiBlock<R, f32>),
    Forward(ForwardBlock<R, f32>),
}

impl Block {
    fn apply(&self, x: &Var<R, f32>) -> Result<Var<R, f32>> {
        match self {
            Block::Bi(b) => b.apply(x),
            Block::Forward(b) => b.apply(x),
        }
    }
}

fn main() -> Result<()> {
    let rows = env("ROWS", 4800);
    let seq = env("SEQ", 17);
    let d = env("D", 64);
    let heads = env("HEADS", 1);
    let state = env("STATE", 8);
    let bidir = env("BIDIR", 0) == 1;
    let layers = env("LAYERS", 1);
    let last = env("LAST", 0) == 1;
    let iters = env("ITERS", 6);
    let compare_last = env("APPLY_LAST", 0) == 1;
    if compare_last && (bidir || layers != 1) {
        return Err(Error::config(
            "APPLY_LAST=1 compares one forward block: use BIDIR=0 LAYERS=1",
        ));
    }

    let device = Device::<R>::default();
    let ssm = SsmConfig {
        d_model: d,
        n_heads: heads,
        head_dim: d,
        d_state: state,
        n_groups: heads,
        ..SsmConfig::default()
    };
    let mut rng = Rng::seeded(1);
    let mut blocks = Vec::with_capacity(layers);
    for _ in 0..layers {
        blocks.push(if bidir {
            Block::Bi(BiBlock::new(d, &ssm, 1e-5, layers, &device, &mut rng)?)
        } else {
            Block::Forward(ForwardBlock::new(d, &ssm, 1e-5, layers, &device, &mut rng)?)
        });
    }
    let input = Param::new(Tensor::<R, f32>::from_f32(
        &rng.normal_vec(rows * seq * d, 0.0, 1.0),
        vec![rows, seq, d],
        &device,
    )?);
    let anchor = Var::constant(Tensor::<R, f32>::zeros(vec![1], &device));

    let in_use = || {
        device.synchronize();
        device.client().memory_usage().ok().map(|u| u.bytes_in_use)
    };
    let idle = in_use();
    let forward = || -> Result<Var<R, f32>> {
        let mut x = input.var(&anchor);
        for block in &blocks {
            x = block.apply(&x)?;
        }
        let read_out = if last { x.slice(1, seq - 1, 1)? } else { x };
        read_out.sum()
    };
    let step = || -> Result<()> {
        let grads = forward()?.backward()?;
        drop(grads);
        Ok(())
    };

    if compare_last {
        let Block::Forward(block) = &blocks[0] else {
            unreachable!("checked above");
        };
        // The stage-1 read-out, two ways.
        let sliced = || -> Result<Var<R, f32>> {
            block.apply(&input.var(&anchor))?.slice(1, seq - 1, 1)?.sum()
        };
        let direct = || -> Result<Var<R, f32>> { block.apply_last(&input.var(&anchor))?.sum() };
        let run = |f: &dyn Fn() -> Result<Var<R, f32>>| -> Result<f64> {
            let t = Instant::now();
            drop(f()?.backward()?);
            device.synchronize();
            Ok(t.elapsed().as_secs_f64() * 1e3)
        };
        for _ in 0..3 {
            run(&sliced)?;
            run(&direct)?;
        }
        let rounds = iters.max(5);
        let (mut a, mut b) = (Vec::with_capacity(rounds), Vec::with_capacity(rounds));
        mamba3::backend::reset_launch_count();
        for _ in 0..rounds {
            a.push(run(&sliced)?);
            b.push(run(&direct)?);
        }
        let reserved = mamba3::backend::reserved_bytes(&device);
        let tape_of = |f: &dyn Fn() -> Result<Var<R, f32>>| -> Result<f64> {
            let loss = f()?;
            let live = in_use();
            drop(loss);
            Ok(mb(live.zip(idle).map(|(live, idle)| live.saturating_sub(idle))))
        };
        let (tape_a, tape_b) = (tape_of(&sliced)?, tape_of(&direct)?);
        let report = |name: &str, times: &mut Vec<f64>, tape: f64| {
            times.sort_by(|x, y| x.partial_cmp(y).unwrap());
            println!(
                "{} rows {rows} seq {seq} d {d} | fwd heads {heads} state {state} | {name}: \
                 min {:.1} ms median {:.1} ms max {:.1} ms | tape {tape:.0} MB",
                device.name(),
                times[0],
                times[times.len() / 2],
                times[times.len() - 1],
            );
        };
        report("apply + slice", &mut a, tape_a);
        report("apply_last   ", &mut b, tape_b);
        println!(
            "{rounds} interleaved rounds | ratio of minima {:.3} | reserved {:.0} MB (both)",
            b[0] / a[0],
            mb(reserved)
        );
        return Ok(());
    }

    // Warm up: kernel compilation, matmul tuning and the allocator pool settle.
    for _ in 0..3 {
        step()?;
    }
    device.synchronize();

    mamba3::backend::reset_launch_count();
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        step()?;
        device.synchronize();
        times.push(t.elapsed().as_secs_f64() * 1e3);
    }
    let launches = mamba3::backend::launch_count() / iters;
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let reserved = mamba3::backend::reserved_bytes(&device);
    let tape = {
        let loss = forward()?;
        let live = in_use();
        drop(loss);
        live.zip(idle).map(|(live, idle)| live.saturating_sub(idle))
    };
    println!(
        "{} rows {rows} seq {seq} d {d} | {} heads/dir {heads} state {state} layers {layers}{} | \
         min {:.1} ms median {:.1} ms | {launches} launches | tape {:.0} MB | reserved {:.0} MB",
        device.name(),
        if bidir { "bi" } else { "fwd" },
        if last { " last-row loss" } else { "" },
        times[0],
        times[times.len() / 2],
        mb(tape),
        mb(reserved),
    );
    Ok(())
}
