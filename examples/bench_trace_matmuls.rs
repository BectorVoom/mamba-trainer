//! Time every matmul shape a `MAMBA3_TRACE` log recorded, weighted by count.
//!
//! The kernel profiler says how long `Matmul*Kernel` took in total; it cannot
//! say which call sites' shapes that time belongs to. This replays each
//! distinct `TRACE matmul` line on synthetic operands, so the shapes can be
//! ranked by `count × time`. A broadcast-weight product (`rhs_bstride=0`,
//! untransposed contiguous `lhs`) is also timed folded into one
//! `[batch*m, k] × [k, n]` product, which is the same arithmetic.
//!
//! ```text
//! MAMBA3_TRACE=1 cargo run --release --no-default-features --features wgpu \
//!     --example profile_entity_model 2> trace.txt
//! cargo run --release --no-default-features --features wgpu \
//!     --example bench_trace_matmuls < trace.txt
//! ```

use std::collections::BTreeMap;
use std::io::BufRead;
use std::time::Instant;

use mamba3::prelude::*;
use mamba3::tensor::ops::matmul::matmul_3d_t;

type R = mamba3::backends::Auto;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
struct Shape3 {
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
    lhs_t: bool,
    rhs_t: bool,
    lhs_bstride: usize,
    rhs_bstride: usize,
}

fn parse(line: &str) -> Option<Shape3> {
    let rest = line.strip_prefix("TRACE matmul ")?;
    let mut f: BTreeMap<&str, &str> = BTreeMap::new();
    for kv in rest.split_whitespace() {
        let (k, v) = kv.split_once('=')?;
        f.insert(k, v);
    }
    let u = |k: &str| f.get(k)?.parse::<usize>().ok();
    let b = |k: &str| f.get(k)?.parse::<bool>().ok();
    Some(Shape3 {
        batch: u("batch")?,
        m: u("m")?,
        n: u("n")?,
        k: u("k")?,
        lhs_t: b("lhs_t")?,
        rhs_t: b("rhs_t")?,
        lhs_bstride: u("lhs_bstride")?,
        rhs_bstride: u("rhs_bstride")?,
    })
}

/// Milliseconds per call: median over 5 samples of `iters` back-to-back calls
/// and one synchronise. One synchronise per call would add the ~1.4 ms fixed
/// wgpu wait to every sample and swamp the small shapes.
fn time_ms(device: &Device<R>, iters: usize, mut f: impl FnMut()) -> f64 {
    f();
    device.synchronize();
    let mut samples = Vec::with_capacity(5);
    for _ in 0..5 {
        let t = Instant::now();
        for _ in 0..iters {
            f();
        }
        device.synchronize();
        samples.push(t.elapsed().as_secs_f64() * 1000.0 / iters as f64);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[2]
}

fn main() -> Result<()> {
    let device = Device::<R>::default();
    let iters: usize = std::env::var("MAMBA3_BENCH_ITERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(20);
    let mut counts: BTreeMap<Shape3, usize> = BTreeMap::new();
    for line in std::io::stdin().lock().lines() {
        if let Some(s) = parse(&line?) {
            *counts.entry(s).or_default() += 1;
        }
    }
    let mut rows = Vec::new();
    for (s, count) in &counts {
        let lhs_len = if s.lhs_bstride == 0 {
            s.m * s.k
        } else {
            s.lhs_bstride * s.batch
        };
        let rhs_len = if s.rhs_bstride == 0 {
            s.k * s.n
        } else {
            s.rhs_bstride * s.batch
        };
        let lhs = Tensor::<R, f32>::ones(vec![lhs_len], &device);
        let rhs = Tensor::<R, f32>::ones(vec![rhs_len], &device);
        let batched = time_ms(&device, iters, || {
            matmul_3d_t(
                &lhs,
                &rhs,
                s.batch,
                s.m,
                s.n,
                s.k,
                s.lhs_bstride,
                s.rhs_bstride,
                s.lhs_t,
                s.rhs_t,
            );
        });
        let foldable = s.batch > 1 && s.rhs_bstride == 0 && !s.lhs_t && s.lhs_bstride == s.m * s.k;
        let folded = foldable.then(|| {
            time_ms(&device, iters, || {
                matmul_3d_t(&lhs, &rhs, 1, s.batch * s.m, s.n, s.k, 0, 0, false, s.rhs_t);
            })
        });
        rows.push((*s, *count, batched, folded));
    }
    rows.sort_by(|a, b| (b.1 as f64 * b.2).partial_cmp(&(a.1 as f64 * a.2)).unwrap());
    let total: f64 = rows.iter().map(|r| r.1 as f64 * r.2).sum();
    println!(
        "backend: {}  total {:.1} ms over the trace",
        device.name(),
        total
    );
    println!(
        "{:>6} {:>9} {:>9} {:>9}  shape",
        "count", "ms/call", "total ms", "folded"
    );
    for (s, count, ms, folded) in rows {
        let folded = folded.map_or("-".to_string(), |f| format!("{f:.3}"));
        println!(
            "{count:>6} {ms:>9.3} {:>9.1} {folded:>9}  b={} m={} n={} k={} lt={} rt={} lbs={} rbs={}",
            count as f64 * ms,
            s.batch,
            s.m,
            s.n,
            s.k,
            s.lhs_t,
            s.rhs_t,
            s.lhs_bstride,
            s.rhs_bstride
        );
    }
    Ok(())
}
