//! The Mamba-1 selective scan: a diagonal state per `(channel, state)` pair.
//!
//! ```text
//! h[b,t,d,n] = exp(dt[b,t,d] * A[d,n]) * h[b,t-1,d,n] + dt[b,t,d] * Bm[b,t,n] * x[b,t,d]
//! y[b,t,d]   = sum_n Cm[b,t,n] * h[b,t,d,n] + skip[d] * x[b,t,d]          (h[b,-1] = 0)
//! ```
//!
//! Shapes: `x, dt : [B, T, D]`, `A : [D, N]`, `Bm, Cm : [B, T, N]`, `skip : [D]`.
//! Unlike the Mamba-2/3 scan ([`super::ssd_scan`]) the decay differs per state
//! column, so there is no semiseparable-matmul form; the scan is evaluated as a
//! recurrence. Every element of the state is the affine recurrence
//! `h_t = a_t h_{t-1} + u_t`, whose step pairs compose associatively
//! (`finance/lean/ClmVerify` proves this, and that any chunking of time gives the
//! sequential result).
//!
//! Forward, two schedules with identical results up to float rounding:
//!
//! * **sequential** — one unit per `(b, d)` holds `h[b, d, :]` in registers and
//!   walks time. Enough parallelism when `B * D` is in the thousands.
//! * **chunked** (the parallel associative scan) — time is cut into chunks of `L`
//!   steps. Phase 1: per `(b, chunk, d)` a local scan from zero, keeping the end
//!   state and the product of its step decays. Phase 2:
//!   per `(b, d)` a pass over the chunk summaries, giving each chunk's carried-in
//!   state. Phase 3: per `(b, chunk, d)` the local re-scan from the carried-in
//!   state, writing `y`. `L` times more units, for small batches and long series.
//!   The chunk's decay is the running product of its per-step factors, so the
//!   chunked result rounds like the sequential one rather than like `exp(A sum dt)`.
//!
//! Both leave the state at the start of every `seg_len`-step segment behind when a
//! gradient will be taken. Backward: one unit per `(b, d)`, cubes of [`GROUP`]
//! channels of one batch row, walking segments from last to first: the segment's
//! states are recomputed into registers from its checkpoint, then the adjoint
//!
//! ```text
//! g[t] = Cm[t] * gy[t] + a[t+1] * g[t+1]
//! ```
//!
//! is walked back through it. `dx`, `ddt` are per unit; `dA`, `dskip` are summed
//! per unit and folded over the batch afterwards; `dBm`, `dCm` (sums over `d`) are
//! staged in shared memory for the segment and reduced per cube into one partial
//! per channel tile, folded afterwards.
//!
//! Everything accumulates in `f32` whatever the storage type.

use cubecl::prelude::*;

use crate::backend::{FloatElem, launch_1d};
use crate::error::{Error, Result};
use crate::tensor::base::Tensor;
use crate::tensor::shape::Shape;

/// Largest `d_state` the kernels take; the state row lives in registers.
pub const MAX_SELECTIVE_STATE: usize = 64;

/// Channels per backward cube (one batch row, `GROUP` consecutive channels).
pub const GROUP: usize = 32;

/// State registers one backward unit spends on a segment's history. 64 (segments
/// of 4 steps at `d_state` 16) measured 3x faster than 128 on the Radeon 860M
/// (wgpu), interleaved A/B with `mamba-clm bench --ab-flush`: the longer history
/// spilled registers.
const HIST_REGS: usize = 64;

/// Longest backward segment.
const MAX_SEG: usize = 16;

/// How the forward pass schedules time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SelectiveScanMode {
    /// Sequential when `B * D` fills the device, chunked otherwise.
    #[default]
    Auto,
    /// One unit per `(b, d)` walking all of time.
    Sequential,
    /// The three-phase chunked scan; the chunk is rounded up to a multiple of the
    /// backward segment length.
    Chunked(usize),
}

/// Shapes of one scan problem.
#[derive(Debug, Clone, Copy)]
struct Dims {
    batch: usize,
    seq: usize,
    dim: usize,
    state: usize,
}

fn dims<R: Runtime, E: FloatElem>(
    x: &Tensor<R, E>,
    dt: &Tensor<R, E>,
    a: &Tensor<R, E>,
    bm: &Tensor<R, E>,
    cm: &Tensor<R, E>,
    skip: Option<&Tensor<R, E>>,
) -> Result<Dims> {
    let shape_err = |what: &str| Error::Shape(format!("selective_scan: {what}"));
    let [batch, seq, dim] = <[usize; 3]>::try_from(x.dims())
        .map_err(|_| shape_err(&format!("x must be [B, T, D], got {:?}", x.dims())))?;
    if dt.dims() != x.dims() {
        return Err(shape_err(&format!("dt {:?} must match x {:?}", dt.dims(), x.dims())));
    }
    let [ad, state] = <[usize; 2]>::try_from(a.dims())
        .map_err(|_| shape_err(&format!("A must be [D, N], got {:?}", a.dims())))?;
    if ad != dim {
        return Err(shape_err(&format!("A {:?} must have D = {dim} rows", a.dims())));
    }
    for (name, t) in [("Bm", bm), ("Cm", cm)] {
        if t.dims() != [batch, seq, state] {
            return Err(shape_err(&format!(
                "{name} {:?} must be [B, T, N] = [{batch}, {seq}, {state}]",
                t.dims()
            )));
        }
    }
    if let Some(skip) = skip
        && skip.dims() != [dim]
    {
        return Err(shape_err(&format!("skip {:?} must be [D] = [{dim}]", skip.dims())));
    }
    if !(1..=MAX_SELECTIVE_STATE).contains(&state) {
        return Err(Error::Unsupported(format!(
            "selective_scan: d_state {state} outside 1..={MAX_SELECTIVE_STATE}"
        )));
    }
    Ok(Dims {
        batch,
        seq,
        dim,
        state,
    })
}

/// The checkpoint segment length the kernels use for `state` columns on this
/// device (forward and backward agree on it).
pub fn segment_len<R: Runtime>(client: &ComputeClient<R>, state: usize) -> usize {
    seg_len(client, state)
}

/// Backward segment length for `state` columns: the history fits [`HIST_REGS`]
/// registers and the segment's `Bm`/`Cm` rows plus the reduction staging fit
/// shared memory.
fn seg_len<R: Runtime>(client: &ComputeClient<R>, state: usize) -> usize {
    let o = SEG_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed);
    if o > 0 {
        return o;
    }
    let shared = client.properties().hardware.max_shared_memory_size.min(48 * 1024);
    let by_regs = (HIST_REGS / state).clamp(1, MAX_SEG);
    // The segment's Bm/Cm rows live in shared memory; the reduction staging is a
    // fixed two steps.
    let by_shared = ((shared / 4).saturating_sub(2 * 2 * state * (GROUP + 1)) / (2 * state)).max(1);
    by_regs.min(by_shared)
}

/// Override of [`flush_len`] for A/B timing (0 = default); see
/// [`set_backward_flush`].
static FLUSH_OVERRIDE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Override of [`seg_len`] for A/B timing (0 = default); see [`set_segment_len`].
static SEG_OVERRIDE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Benchmarking only: the checkpoint segment length used by both the forward and
/// the backward (`None` restores the default). Changing it between a forward and
/// its backward only costs a recomputation (the checkpoints are then rejected).
pub fn set_segment_len(steps: Option<usize>) {
    SEG_OVERRIDE.store(steps.unwrap_or(0), std::sync::atomic::Ordering::Relaxed);
}

/// Benchmarking only: reduce the backward's staged sums every `steps` reverse
/// steps (`None` restores the default). `steps` must divide the segment length;
/// otherwise the default is used. Larger values need more shared memory.
pub fn set_backward_flush(steps: Option<usize>) {
    FLUSH_OVERRIDE.store(steps.unwrap_or(0), std::sync::atomic::Ordering::Relaxed);
}

/// Reverse steps between the backward's shared-memory reductions: 2 when it
/// divides the segment, else 1.
fn flush_len(seg: usize) -> usize {
    let o = FLUSH_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed);
    if o > 0 && seg.is_multiple_of(o) {
        return o;
    }
    if seg.is_multiple_of(2) { 2 } else { 1 }
}

/// Units below which the forward takes the chunked schedule under
/// [`SelectiveScanMode::Auto`].
const AUTO_SEQUENTIAL_UNITS: usize = 8192;

/// Chunk length for the chunked schedule (a multiple of `seg`).
fn chunk_len(mode: SelectiveScanMode, d: &Dims, seg: usize) -> Option<usize> {
    let raw = match mode {
        SelectiveScanMode::Sequential => return None,
        SelectiveScanMode::Chunked(l) => l.clamp(1, d.seq.max(1)),
        SelectiveScanMode::Auto => {
            let units = d.batch * d.dim;
            if units >= AUTO_SEQUENTIAL_UNITS || d.seq <= 2 * seg {
                return None;
            }
            // Enough chunks to reach the sequential threshold, but at least two
            // segments per chunk so phase 2 stays short relative to the rest.
            let chunks = AUTO_SEQUENTIAL_UNITS.div_ceil(units).max(2);
            d.seq.div_ceil(chunks).max(2 * seg)
        }
    };
    Some(raw.div_ceil(seg) * seg)
}

/// The per-segment states the forward leaves for the backward.
///
/// They are tied to the input buffers they were computed from: the backward uses
/// them only for those same buffers (and shapes) and recomputes otherwise. Writing
/// into an input buffer in place between the forward and the backward is not
/// detected; the autograd op never does.
pub struct SelectiveScanCheckpoints<R: Runtime> {
    states: Tensor<R, f32>,
    seg_len: usize,
    inputs: [u64; 6],
    dims: [usize; 4],
}

/// Buffer identities of a scan's inputs (0 for an absent skip).
fn input_ids<R: Runtime, E: FloatElem>(
    x: &Tensor<R, E>,
    dt: &Tensor<R, E>,
    a: &Tensor<R, E>,
    bm: &Tensor<R, E>,
    cm: &Tensor<R, E>,
    skip: Option<&Tensor<R, E>>,
) -> [u64; 6] {
    [
        x.buffer_id,
        dt.buffer_id,
        a.buffer_id,
        bm.buffer_id,
        cm.buffer_id,
        skip.map_or(0, |s| s.buffer_id),
    ]
}

#[cube]
fn load_f<F: Float + CubeElement>(buf: &Array<F>, i: usize) -> f32 {
    f32::cast_from(buf[i])
}

#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn selective_scan_kernel<F: Float + CubeElement>(
    x: &Array<F>,
    dt: &Array<F>,
    a: &Array<F>,
    bm: &Array<F>,
    cm: &Array<F>,
    skip: &Array<F>,
    y: &mut Array<F>,
    ckpt: &mut Array<f32>,
    seq: usize,
    dim: usize,
    lanes: usize,
    segs: usize,
    #[comptime] n_state: usize,
    #[comptime] seg_len: usize,
    #[comptime] has_skip: bool,
    #[comptime] save_ckpt: bool,
) {
    if ABSOLUTE_POS < lanes {
        let d = ABSOLUTE_POS % dim;
        let bi = ABSOLUTE_POS / dim;
        let mut s = Array::<f32>::new(n_state);
        let mut ad = Array::<f32>::new(n_state);
        #[unroll]
        for i in 0..n_state {
            s[i] = 0.0f32;
            ad[i] = load_f::<F>(a, d * n_state + i);
        }
        let mut dskip = 0.0f32;
        if comptime!(has_skip) {
            dskip = load_f::<F>(skip, d);
        }
        for t in 0..seq {
            if comptime!(save_ckpt) {
                if t % seg_len == 0 {
                    let base = ((bi * segs + t / seg_len) * n_state) * dim + d;
                    #[unroll]
                    for i in 0..n_state {
                        ckpt[base + i * dim] = s[i];
                    }
                }
            }
            let row = bi * seq + t;
            let xv = load_f::<F>(x, row * dim + d);
            let dtv = load_f::<F>(dt, row * dim + d);
            let u = dtv * xv;
            let mut acc = 0.0f32;
            #[unroll]
            for i in 0..n_state {
                let bv = load_f::<F>(bm, row * n_state + i);
                let cv = load_f::<F>(cm, row * n_state + i);
                s[i] = f32::exp(dtv * ad[i]) * s[i] + u * bv;
                acc += cv * s[i];
            }
            y[row * dim + d] = F::cast_from(acc + dskip * xv);
        }
    }
}

/// Chunked phase 1: per `(b, chunk, d)`, the chunk's local scan from zero and the
/// product of its step decays (the same `f32` factors the sequential recurrence
/// multiplies, not `exp(A * sum dt)`, so long chunks do not drift). Lanes are
/// `((b * chunks + c) * D + d)`.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn selective_chunk_local_kernel<F: Float + CubeElement>(
    x: &Array<F>,
    dt: &Array<F>,
    a: &Array<F>,
    bm: &Array<F>,
    local: &mut Array<f32>,
    decay: &mut Array<f32>,
    seq: usize,
    dim: usize,
    chunks: usize,
    chunk: usize,
    lanes: usize,
    #[comptime] n_state: usize,
) {
    if ABSOLUTE_POS < lanes {
        let d = ABSOLUTE_POS % dim;
        let bc = ABSOLUTE_POS / dim;
        let c = bc % chunks;
        let bi = bc / chunks;
        let mut s = Array::<f32>::new(n_state);
        let mut prod = Array::<f32>::new(n_state);
        let mut ad = Array::<f32>::new(n_state);
        #[unroll]
        for i in 0..n_state {
            s[i] = 0.0f32;
            prod[i] = 1.0f32;
            ad[i] = load_f::<F>(a, d * n_state + i);
        }
        let t0 = c * chunk;
        let t1 = select(t0 + chunk < seq, t0 + chunk, seq);
        for t in t0..t1 {
            let row = bi * seq + t;
            let xv = load_f::<F>(x, row * dim + d);
            let dtv = load_f::<F>(dt, row * dim + d);
            let u = dtv * xv;
            #[unroll]
            for i in 0..n_state {
                let bv = load_f::<F>(bm, row * n_state + i);
                let e = f32::exp(dtv * ad[i]);
                s[i] = e * s[i] + u * bv;
                prod[i] *= e;
            }
        }
        let base = (bc * n_state) * dim + d;
        #[unroll]
        for i in 0..n_state {
            local[base + i * dim] = s[i];
            decay[base + i * dim] = prod[i];
        }
    }
}

/// Chunked phase 2: per `(b, d)`, the carried-in state of every chunk.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn selective_chunk_carry_kernel(
    local: &Array<f32>,
    decay: &Array<f32>,
    carry: &mut Array<f32>,
    dim: usize,
    chunks: usize,
    lanes: usize,
    #[comptime] n_state: usize,
) {
    if ABSOLUTE_POS < lanes {
        let d = ABSOLUTE_POS % dim;
        let bi = ABSOLUTE_POS / dim;
        let mut s = Array::<f32>::new(n_state);
        #[unroll]
        for i in 0..n_state {
            s[i] = 0.0f32;
        }
        for c in 0..chunks {
            let base = ((bi * chunks + c) * n_state) * dim + d;
            #[unroll]
            for i in 0..n_state {
                carry[base + i * dim] = s[i];
                s[i] = decay[base + i * dim] * s[i] + local[base + i * dim];
            }
        }
    }
}

/// Chunked phase 3: per `(b, chunk, d)`, the re-scan from the carried-in state,
/// writing `y` and the backward checkpoints.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn selective_chunk_output_kernel<F: Float + CubeElement>(
    x: &Array<F>,
    dt: &Array<F>,
    a: &Array<F>,
    bm: &Array<F>,
    cm: &Array<F>,
    skip: &Array<F>,
    carry: &Array<f32>,
    y: &mut Array<F>,
    ckpt: &mut Array<f32>,
    seq: usize,
    dim: usize,
    chunks: usize,
    chunk: usize,
    lanes: usize,
    segs: usize,
    #[comptime] n_state: usize,
    #[comptime] seg_len: usize,
    #[comptime] has_skip: bool,
    #[comptime] save_ckpt: bool,
) {
    if ABSOLUTE_POS < lanes {
        let d = ABSOLUTE_POS % dim;
        let bc = ABSOLUTE_POS / dim;
        let c = bc % chunks;
        let bi = bc / chunks;
        let mut s = Array::<f32>::new(n_state);
        let mut ad = Array::<f32>::new(n_state);
        let cbase = (bc * n_state) * dim + d;
        #[unroll]
        for i in 0..n_state {
            s[i] = carry[cbase + i * dim];
            ad[i] = load_f::<F>(a, d * n_state + i);
        }
        let mut dskip = 0.0f32;
        if comptime!(has_skip) {
            dskip = load_f::<F>(skip, d);
        }
        let t0 = c * chunk;
        let t1 = select(t0 + chunk < seq, t0 + chunk, seq);
        for t in t0..t1 {
            if comptime!(save_ckpt) {
                if t % seg_len == 0 {
                    let base = ((bi * segs + t / seg_len) * n_state) * dim + d;
                    #[unroll]
                    for i in 0..n_state {
                        ckpt[base + i * dim] = s[i];
                    }
                }
            }
            let row = bi * seq + t;
            let xv = load_f::<F>(x, row * dim + d);
            let dtv = load_f::<F>(dt, row * dim + d);
            let u = dtv * xv;
            let mut acc = 0.0f32;
            #[unroll]
            for i in 0..n_state {
                let bv = load_f::<F>(bm, row * n_state + i);
                let cv = load_f::<F>(cm, row * n_state + i);
                s[i] = f32::exp(dtv * ad[i]) * s[i] + u * bv;
                acc += cv * s[i];
            }
            y[row * dim + d] = F::cast_from(acc + dskip * xv);
        }
    }
}

/// Backward. Cube `(b, tile)` of [`GROUP`] units, unit `l` is channel
/// `tile * GROUP + l` (padding units past `D` run on channel 0 and contribute 0).
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn selective_scan_backward_kernel<F: Float + CubeElement>(
    dy: &Array<F>,
    x: &Array<F>,
    dt: &Array<F>,
    a: &Array<F>,
    bm: &Array<F>,
    cm: &Array<F>,
    skip: &Array<F>,
    ckpt: &Array<f32>,
    dx: &mut Array<F>,
    ddt: &mut Array<F>,
    db_part: &mut Array<f32>,
    dc_part: &mut Array<f32>,
    da_part: &mut Array<f32>,
    skip_part: &mut Array<f32>,
    seq: usize,
    dim: usize,
    tiles: usize,
    cubes: usize,
    segs: usize,
    #[comptime] n_state: usize,
    #[comptime] seg_len: usize,
    #[comptime] flush: usize,
    #[comptime] group: usize,
    #[comptime] has_skip: bool,
) {
    let cube = CUBE_POS_Y as usize * CUBE_COUNT_X as usize + CUBE_POS_X as usize;
    // Whole cubes past the end leave together, before any barrier.
    if cube < cubes {
        let local_id = UNIT_POS_X as usize;
        let tile = cube % tiles;
        let bi = cube / tiles;
        let d_raw = tile * group + local_id;
        let live = d_raw < dim;
        let d = select(live, d_raw, 0usize);
        let keep = select(live, 1.0f32, 0.0f32);

        let mut ad = Array::<f32>::new(n_state);
        let mut g = Array::<f32>::new(n_state);
        let mut da_acc = Array::<f32>::new(n_state);
        let mut s = Array::<f32>::new(n_state);
        #[unroll]
        for i in 0..n_state {
            ad[i] = load_f::<F>(a, d * n_state + i);
            g[i] = 0.0f32;
            da_acc[i] = 0.0f32;
            s[i] = 0.0f32;
        }
        let mut dskip = 0.0f32;
        if comptime!(has_skip) {
            dskip = load_f::<F>(skip, d);
        }
        let mut skip_acc = 0.0f32;
        // State before each step of the segment.
        let mut hist = Array::<f32>::new(comptime!(seg_len * n_state));
        // The last `flush` steps' dBm and dCm contributions,
        // `[flush * 2 * n_state][group + 1]`, reduced over the cube's channels every
        // `flush` steps (small, so several cubes fit per compute unit). The padding
        // column keeps both the per-step writes (one row, consecutive units) and the
        // reduction's reads (one unit per row) free of bank conflicts.
        let stride = comptime!(group + 1);
        let rows = comptime!(seg_len * 2 * n_state);
        let flush_rows = comptime!(flush * 2 * n_state);
        let mut stage = SharedMemory::<f32>::new(comptime!(flush * 2 * n_state * (group + 1)));
        // The segment's `Bm` and `Cm` rows, loaded once per cube instead of once
        // per unit: `[seg_len][Bm (n_state) | Cm (n_state)]`.
        let mut bc = SharedMemory::<f32>::new(comptime!(seg_len * 2 * n_state));

        for kk in 0..segs {
            let k = segs - 1 - kk;
            let t0 = k * seg_len;
            let mut q = local_id;
            while q < rows {
                let j = q / comptime!(2 * n_state);
                let rest = q % comptime!(2 * n_state);
                let t = t0 + j;
                let row = bi * seq + select(t < seq, t, seq - 1);
                let from_c = rest >= n_state;
                let col = select(from_c, rest - select(from_c, n_state, 0usize), rest);
                let bv = load_f::<F>(bm, row * n_state + col);
                let cv = load_f::<F>(cm, row * n_state + col);
                bc[q] = select(from_c, cv, bv);
                q += group;
            }
            sync_cube();
            let cbase = ((bi * segs + k) * n_state) * dim + d;
            #[unroll]
            for i in 0..n_state {
                s[i] = ckpt[cbase + i * dim];
            }
            // Recompute the segment's states. Steps past the end are never read
            // back (their adjoint step is skipped), so they run on clamped rows.
            #[unroll]
            for j in 0..seg_len {
                let t = t0 + j;
                let tt = select(t < seq, t, seq - 1);
                let row = bi * seq + tt;
                let xv = load_f::<F>(x, row * dim + d);
                let dtv = load_f::<F>(dt, row * dim + d);
                let u = dtv * xv;
                #[unroll]
                for i in 0..n_state {
                    hist[comptime!(j * n_state) + i] = s[i];
                    let bv = bc[comptime!(j * 2 * n_state) + i];
                    s[i] = f32::exp(dtv * ad[i]) * s[i] + u * bv;
                }
            }
            // Walk the segment back.
            #[unroll]
            for jj in 0..seg_len {
                let j = comptime!(seg_len - 1 - jj);
                let t = t0 + j;
                let valid = t < seq;
                let tt = select(valid, t, seq - 1);
                let row = bi * seq + tt;
                let on = select(valid, keep, 0.0f32);
                let gy = load_f::<F>(dy, row * dim + d) * on;
                let xv = load_f::<F>(x, row * dim + d);
                let dtv = load_f::<F>(dt, row * dim + d);
                let mut gx = 0.0f32;
                let mut gdt = 0.0f32;
                #[unroll]
                for i in 0..n_state {
                    let bv = bc[comptime!(j * 2 * n_state) + i];
                    let cv = bc[comptime!(j * 2 * n_state + n_state) + i];
                    let hp = hist[comptime!(j * n_state) + i];
                    let decay = f32::exp(dtv * ad[i]);
                    let h = decay * hp + dtv * bv * xv;
                    // Past the end `gy` is 0 and `g` must pass through untouched.
                    let gi = g[i] + cv * gy;
                    gx += gi * dtv * bv;
                    gdt += gi * (ad[i] * decay * hp + bv * xv);
                    da_acc[i] += gi * dtv * decay * hp;
                    stage[(comptime!((jj % flush) * 2 * n_state) + i) * stride + local_id] = gi * dtv * xv * on;
                    stage[(comptime!((jj % flush) * 2 * n_state + n_state) + i) * stride + local_id] = gy * h;
                    g[i] = select(valid, decay * gi, g[i]);
                }
                if valid && live {
                    let out = row * dim + d;
                    dx[out] = F::cast_from(gx + dskip * gy);
                    ddt[out] = F::cast_from(gdt);
                }
                skip_acc += gy * xv;
                if comptime!((jj + 1) % flush == 0) {
                    // Reduce the last `flush` steps over the cube's channels. Slot
                    // `q` of this window is reverse step `jj + 1 - flush + q`.
                    sync_cube();
                    let mut r = local_id;
                    while r < flush_rows {
                        let mut total = 0.0f32;
                        for l in 0..group {
                            total += stage[r * stride + l];
                        }
                        let slot = r / comptime!(2 * n_state);
                        let rest = r % comptime!(2 * n_state);
                        let t = t0 + comptime!(seg_len - 1 - (jj + 1 - flush)) - slot;
                        if t < seq {
                            let part = ((bi * tiles + tile) * seq + t) * n_state;
                            if rest < n_state {
                                db_part[part + rest] = total;
                            } else {
                                dc_part[part + rest - n_state] = total;
                            }
                        }
                        r += group;
                    }
                    sync_cube();
                }
            }
        }
        if live {
            let pbase = (bi * dim + d) * n_state;
            #[unroll]
            for i in 0..n_state {
                da_part[pbase + i] = da_acc[i];
            }
            if comptime!(has_skip) {
                skip_part[bi * dim + d] = skip_acc;
            }
        }
    }
}

/// [`selective_scan_saving`] without checkpoints.
#[allow(clippy::too_many_arguments)]
pub fn selective_scan<R: Runtime, E: FloatElem>(
    x: &Tensor<R, E>,
    dt: &Tensor<R, E>,
    a: &Tensor<R, E>,
    bm: &Tensor<R, E>,
    cm: &Tensor<R, E>,
    skip: Option<&Tensor<R, E>>,
    mode: SelectiveScanMode,
) -> Result<Tensor<R, E>> {
    Ok(selective_scan_saving(x, dt, a, bm, cm, skip, mode, false)?.0)
}

/// The forward scan, leaving the backward's checkpoints behind when `save`.
#[allow(clippy::too_many_arguments)]
pub fn selective_scan_saving<R: Runtime, E: FloatElem>(
    x: &Tensor<R, E>,
    dt: &Tensor<R, E>,
    a: &Tensor<R, E>,
    bm: &Tensor<R, E>,
    cm: &Tensor<R, E>,
    skip: Option<&Tensor<R, E>>,
    mode: SelectiveScanMode,
    save: bool,
) -> Result<(Tensor<R, E>, Option<SelectiveScanCheckpoints<R>>)> {
    let _op = crate::backend::tally_op_scope("selective_scan");
    let d = dims(x, dt, a, bm, cm, skip)?;
    let y = Tensor::empty(x.shape().clone(), x.device());
    if y.is_empty() {
        return Ok((y, None));
    }
    let client = x.client();
    let device = x.device();
    let seg = seg_len(client, d.state);
    let segs = d.seq.div_ceil(seg);
    let placeholder = Tensor::<R, E>::empty(Shape::new(vec![1]), device);
    let ckpt = Tensor::<R, f32>::empty(
        Shape::new(vec![if save { d.batch * segs * d.state * d.dim } else { 1 }]),
        device,
    );
    let skip_arg = skip.unwrap_or(&placeholder);
    match chunk_len(mode, &d, seg) {
        None => {
            let lanes = d.batch * d.dim;
            let (count, cube) = launch_1d(client, lanes, d.seq * d.state * 4);
            unsafe {
                selective_scan_kernel::launch_unchecked::<E, R>(
                    client,
                    count,
                    cube,
                    x.arg(),
                    dt.arg(),
                    a.arg(),
                    bm.arg(),
                    cm.arg(),
                    skip_arg.arg(),
                    y.arg(),
                    ckpt.arg(),
                    d.seq,
                    d.dim,
                    lanes,
                    segs,
                    d.state,
                    seg,
                    skip.is_some(),
                    save,
                );
            }
        }
        Some(chunk) => {
            let chunks = d.seq.div_ceil(chunk);
            let summaries = d.batch * chunks * d.dim;
            let local = Tensor::<R, f32>::empty(Shape::new(vec![summaries * d.state]), device);
            let decay = Tensor::<R, f32>::empty(Shape::new(vec![summaries * d.state]), device);
            let carry = Tensor::<R, f32>::empty(Shape::new(vec![summaries * d.state]), device);
            let (count, cube) = launch_1d(client, summaries, chunk * d.state * 3);
            unsafe {
                selective_chunk_local_kernel::launch_unchecked::<E, R>(
                    client,
                    count,
                    cube,
                    x.arg(),
                    dt.arg(),
                    a.arg(),
                    bm.arg(),
                    local.arg(),
                    decay.arg(),
                    d.seq,
                    d.dim,
                    chunks,
                    chunk,
                    summaries,
                    d.state,
                );
            }
            let lanes = d.batch * d.dim;
            let (count, cube) = launch_1d(client, lanes, chunks * d.state * 3);
            unsafe {
                selective_chunk_carry_kernel::launch_unchecked::<R>(
                    client,
                    count,
                    cube,
                    local.arg(),
                    decay.arg(),
                    carry.arg(),
                    d.dim,
                    chunks,
                    lanes,
                    d.state,
                );
            }
            let (count, cube) = launch_1d(client, summaries, chunk * d.state * 4);
            unsafe {
                selective_chunk_output_kernel::launch_unchecked::<E, R>(
                    client,
                    count,
                    cube,
                    x.arg(),
                    dt.arg(),
                    a.arg(),
                    bm.arg(),
                    cm.arg(),
                    skip_arg.arg(),
                    carry.arg(),
                    y.arg(),
                    ckpt.arg(),
                    d.seq,
                    d.dim,
                    chunks,
                    chunk,
                    summaries,
                    segs,
                    d.state,
                    seg,
                    skip.is_some(),
                    save,
                );
            }
        }
    }
    Ok((
        y,
        save.then_some(SelectiveScanCheckpoints {
            states: ckpt,
            seg_len: seg,
            inputs: input_ids(x, dt, a, bm, cm, skip),
            dims: [d.batch, d.seq, d.dim, d.state],
        }),
    ))
}

/// Gradients of [`selective_scan`].
pub struct SelectiveScanGrads<R: Runtime, E: FloatElem> {
    /// `[B, T, D]`.
    pub dx: Tensor<R, E>,
    /// `[B, T, D]`.
    pub ddt: Tensor<R, E>,
    /// `[D, N]`.
    pub da: Tensor<R, E>,
    /// `[B, T, N]`.
    pub dbm: Tensor<R, E>,
    /// `[B, T, N]`.
    pub dcm: Tensor<R, E>,
    /// `[D]`, when the forward had a skip.
    pub dskip: Option<Tensor<R, E>>,
}

/// The adjoint of [`selective_scan`]: one launch plus the folds of the partial
/// sums. Runs the forward first when `saved` is `None` or was made from other
/// input buffers, shapes or segment length.
#[allow(clippy::too_many_arguments)]
pub fn selective_scan_backward<R: Runtime, E: FloatElem>(
    dy: &Tensor<R, E>,
    x: &Tensor<R, E>,
    dt: &Tensor<R, E>,
    a: &Tensor<R, E>,
    bm: &Tensor<R, E>,
    cm: &Tensor<R, E>,
    skip: Option<&Tensor<R, E>>,
    saved: Option<&SelectiveScanCheckpoints<R>>,
) -> Result<SelectiveScanGrads<R, E>> {
    let _op = crate::backend::tally_op_scope("selective_scan_backward");
    let d = dims(x, dt, a, bm, cm, skip)?;
    if dy.dims() != x.dims() {
        return Err(Error::Shape(format!(
            "selective_scan_backward: dy {:?} must match x {:?}",
            dy.dims(),
            x.dims()
        )));
    }
    let device = x.device();
    let client = x.client();
    let seg = seg_len(client, d.state);
    let segs = d.seq.div_ceil(seg);
    // The defaults always fit; the benchmarking overrides might not.
    let shared = client.properties().hardware.max_shared_memory_size.min(48 * 1024);
    let need = (seg * 2 * d.state + flush_len(seg) * 2 * d.state * (GROUP + 1)) * 4;
    if need > shared || seg > MAX_SEG {
        return Err(Error::Unsupported(format!(
            "selective_scan_backward: segment {seg} / flush {} needs {need} B of shared memory (limit {shared})",
            flush_len(seg)
        )));
    }
    let dx = Tensor::<R, E>::empty(x.shape().clone(), device);
    let ddt = Tensor::<R, E>::empty(x.shape().clone(), device);
    if x.is_empty() {
        let zeros = |s: Vec<usize>| Tensor::<R, E>::zeros(Shape::new(s), device);
        return Ok(SelectiveScanGrads {
            dx,
            ddt,
            da: zeros(vec![d.dim, d.state]),
            dbm: zeros(vec![d.batch, d.seq, d.state]),
            dcm: zeros(vec![d.batch, d.seq, d.state]),
            dskip: skip.map(|_| zeros(vec![d.dim])),
        });
    }
    let fresh;
    let ckpt = match saved {
        Some(saved)
            if saved.seg_len == seg
                && saved.dims == [d.batch, d.seq, d.dim, d.state]
                && saved.inputs == input_ids(x, dt, a, bm, cm, skip) =>
        {
            &saved.states
        }
        _ => {
            fresh = selective_scan_saving(x, dt, a, bm, cm, skip, SelectiveScanMode::Auto, true)?
                .1
                .expect("save requested");
            &fresh.states
        }
    };
    let tiles = d.dim.div_ceil(GROUP);
    let cubes = d.batch * tiles;
    let db_part = Tensor::<R, f32>::empty(Shape::new(vec![d.batch, tiles, d.seq, d.state]), device);
    let dc_part = Tensor::<R, f32>::empty(Shape::new(vec![d.batch, tiles, d.seq, d.state]), device);
    let da_part = Tensor::<R, f32>::empty(Shape::new(vec![d.batch, d.dim, d.state]), device);
    let skip_part = Tensor::<R, f32>::empty(
        Shape::new(vec![if skip.is_some() { d.batch * d.dim } else { 1 }]),
        device,
    );
    let placeholder = Tensor::<R, E>::empty(Shape::new(vec![1]), device);
    let x_cubes = cubes.min(u16::MAX as usize);
    let count = CubeCount::Static(x_cubes as u32, cubes.div_ceil(x_cubes) as u32, 1);
    crate::backend::count_launch();
    unsafe {
        selective_scan_backward_kernel::launch_unchecked::<E, R>(
            client,
            count,
            CubeDim::new_1d(GROUP as u32),
            dy.arg(),
            x.arg(),
            dt.arg(),
            a.arg(),
            bm.arg(),
            cm.arg(),
            skip.unwrap_or(&placeholder).arg(),
            ckpt.arg(),
            dx.arg(),
            ddt.arg(),
            db_part.arg(),
            dc_part.arg(),
            da_part.arg(),
            skip_part.arg(),
            d.seq,
            d.dim,
            tiles,
            cubes,
            segs,
            d.state,
            seg,
            flush_len(seg),
            GROUP,
            skip.is_some(),
        );
    }
    use crate::tensor::ops::elemwise::cast;
    use crate::tensor::ops::reduce::sum_dim;
    let fold_tiles = |part: &Tensor<R, f32>| -> Result<Tensor<R, E>> {
        let summed = if tiles == 1 {
            part.reshape(Shape::new(vec![d.batch, d.seq, d.state]))?
        } else {
            sum_dim(part, 1)?.reshape(Shape::new(vec![d.batch, d.seq, d.state]))?
        };
        Ok(cast::<R, f32, E>(&summed))
    };
    let dbm = fold_tiles(&db_part)?;
    let dcm = fold_tiles(&dc_part)?;
    let da = cast::<R, f32, E>(&sum_dim(&da_part, 0)?).reshape(Shape::new(vec![d.dim, d.state]))?;
    let dskip = match skip {
        Some(_) => Some(
            cast::<R, f32, E>(&sum_dim(&skip_part.reshape(Shape::new(vec![d.batch, d.dim]))?, 0)?)
                .reshape(Shape::new(vec![d.dim]))?,
        ),
        None => None,
    };
    Ok(SelectiveScanGrads {
        dx,
        ddt,
        da,
        dbm,
        dcm,
        dskip,
    })
}
