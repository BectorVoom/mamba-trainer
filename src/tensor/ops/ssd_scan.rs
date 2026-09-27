//! The structured state space scan as one recurrence kernel per direction.
//!
//! [`crate::ssm::ssd_chunked`] computes
//!
//! ```text
//! S_t = exp(a_t) S_{t-1} + w_t x_t B_tᵀ          (S_{-1} = initial state or 0)
//! y_t = exp(a_t) C_t · S_{t-1} + g_t (C_t · B_t) x_t
//! ```
//!
//! as the chunked (SSD) decomposition: a band matrix per chunk, chunk summaries,
//! an inter-chunk recurrence and a carry-in term, each a handful of batched
//! matmuls, permutation copies and elementwise passes — about 60 launches per
//! scan forward and more backward, most of which stream a full activation-sized
//! tensor through memory. For the state sizes the models here use (`d_state` ≤ 64)
//! the recurrence itself is cheap arithmetic, so evaluating it directly is one
//! pass over the inputs and one over the output.
//!
//! Two neighbours of the scan in a Mamba-3 layer ride along, because each is a
//! broadcasting op with a reduction in its adjoint that costs more than the
//! arithmetic it does: the log decay can be given as `dt_t * A[h]`
//! ([`ScanDecay::Rate`]) instead of a materialised `a`, and the direct skip
//! `y += D[h] x` can be added in the same pass. Their per-head parameter
//! gradients come out of the backward kernel as one partial per (batch, head).
//!
//! Forward: one unit per `(batch, head, p)` holds the state row `S[p, :]` in
//! registers and walks time. Backward: `(batch, head)` pairs of units per `p`
//! (split across up to four lanes, padded to whole planes). A forward sweep stores
//! the state every few steps (at most [`SEG`]); each segment is then recomputed
//! into registers and walked back from them. The
//! per-step reductions over `p` (`dB`, `dC`, `da`, `dg`, `dw`, the skip) are
//! xor butterflies inside a plane combined across a pair's planes in a small
//! double-buffered shared array, so each step costs one barrier.
//!
//! Everything accumulates in `f32` whatever the storage type.

use cubecl::prelude::*;

use crate::backend::{FloatElem, launch_1d};
use crate::error::{Error, Result};
use crate::tensor::base::Tensor;
use crate::tensor::shape::Shape;

/// Largest `d_state` the fused scan takes; the state row lives in registers.
pub const MAX_FUSED_STATE: usize = 64;

/// Largest `head_dim` the fused backward takes; it is the cube width.
pub const MAX_FUSED_HEAD_DIM: usize = 256;

/// Most steps between checkpoints in the backward recomputation, and the most
/// state registers a unit spends on one segment's history (`seg_len * n_per`).
const SEG: usize = 8;
const HIST_REGS: usize = 64;


/// How the scan's log decay `a [B, T, H]` is given.
pub enum ScanDecay<'a, R: Runtime, E: FloatElem> {
    /// Materialised: `a` itself.
    Log(&'a Tensor<R, E>),
    /// `a = dt * a_head[h]`, with `dt [B, T, H]` and `a_head [H]` (the `A` of the
    /// Mamba recurrence, negative).
    Rate {
        /// Time steps, `[B, T, H]`.
        dt: &'a Tensor<R, E>,
        /// Per-head rate, `[H]`.
        a_head: &'a Tensor<R, E>,
    },
}

/// `a_t` for `row`, from whichever form the scan was given.
#[cube]
fn log_decay<F: Float + CubeElement>(
    a: &Array<F>,
    a_head: &Array<F>,
    row: usize,
    h: usize,
    #[comptime] rate: bool,
) -> f32 {
    let mut at = f32::cast_from(a[row]);
    if comptime!(rate) {
        at *= f32::cast_from(a_head[h]);
    }
    at
}

#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn ssd_scan_kernel<F: Float + CubeElement, V: Size>(
    x: &Array<F>,
    b: &Array<Vector<F, V>>,
    c: &Array<Vector<F, V>>,
    a: &Array<F>,
    a_head: &Array<F>,
    g: &Array<F>,
    w: &Array<F>,
    init: &Array<F>,
    skip: &Array<F>,
    y: &mut Array<F>,
    ckpt: &mut Array<f32>,
    seq: usize,
    heads: usize,
    head_dim: usize,
    lanes: usize,
    segs: usize,
    #[comptime] n_state: usize,
    #[comptime] vw: usize,
    #[comptime] has_init: bool,
    #[comptime] rate: bool,
    #[comptime] has_skip: bool,
    #[comptime] seg_len: usize,
    #[comptime] save_ckpt: bool,
) {
    // `b` and `c` are read `vw` state columns per load: every unit of a
    // (batch, head) reads the same row of them each step, so the load count, not
    // the bytes, is what they cost.
    if ABSOLUTE_POS < lanes {
        let p = ABSOLUTE_POS % head_dim;
        let bh = ABSOLUTE_POS / head_dim;
        let h = bh % heads;
        let bi = bh / heads;
        let mut d = 0.0f32;
        if comptime!(has_skip) {
            d = f32::cast_from(skip[h]);
        }

        let mut s = Array::<f32>::new(n_state);
        #[unroll]
        for i in 0..n_state {
            if comptime!(has_init) {
                s[i] = f32::cast_from(init[(bh * head_dim + p) * n_state + i]);
            } else {
                s[i] = 0.0f32;
            }
        }

        // Two steps per iteration, each step's loads issued before the other
        // step's arithmetic: a step that loads and then computes waits out the
        // whole memory latency every time.
        let mut in_a = Array::<f32>::new(comptime!(4 + 2 * n_state));
        let mut in_b = Array::<f32>::new(comptime!(4 + 2 * n_state));
        scan_step_inputs::<F, V>(x, b, c, a, a_head, g, w, &mut in_a, bi, h, p, seq, heads, head_dim, 0, n_state, vw, rate);
        let pairs = seq.div_ceil(2);
        for q in 0..pairs {
            let t = 2 * q;
            scan_step_inputs::<F, V>(x, b, c, a, a_head, g, w, &mut in_b, bi, h, p, seq, heads, head_dim, t + 1, n_state, vw, rate);
            scan_step::<F>(&mut s, &in_a, y, ckpt, bh, bi, h, p, seq, heads, head_dim, segs, d, t, n_state, seg_len, save_ckpt);
            scan_step_inputs::<F, V>(x, b, c, a, a_head, g, w, &mut in_a, bi, h, p, seq, heads, head_dim, t + 2, n_state, vw, rate);
            if t + 1 < seq {
                scan_step::<F>(&mut s, &in_b, y, ckpt, bh, bi, h, p, seq, heads, head_dim, segs, d, t + 1, n_state, seg_len, save_ckpt);
            }
        }
    }
}

/// Load one forward step's inputs for unit `(bh, p)`: `[exp(a), g, w x_p, x_p,
/// b[0..N], c[0..N]]`. Steps past the end read row 0 (and are never used).
#[cube]
#[allow(clippy::too_many_arguments)]
fn scan_step_inputs<F: Float + CubeElement, V: Size>(
    x: &Array<F>,
    b: &Array<Vector<F, V>>,
    c: &Array<Vector<F, V>>,
    a: &Array<F>,
    a_head: &Array<F>,
    g: &Array<F>,
    w: &Array<F>,
    inp: &mut Array<f32>,
    bi: usize,
    h: usize,
    p: usize,
    seq: usize,
    heads: usize,
    head_dim: usize,
    t: usize,
    #[comptime] n_state: usize,
    #[comptime] vw: usize,
    #[comptime] rate: bool,
) {
    let tt = select(t < seq, t, 0usize);
    let row = (bi * seq + tt) * heads + h;
    inp[0] = log_decay::<F>(a, a_head, row, h, rate);
    inp[1] = f32::cast_from(g[row]);
    let xp = f32::cast_from(x[row * head_dim + p]);
    inp[2] = f32::cast_from(w[row]);
    inp[3] = xp;
    let base = row * comptime!(n_state / vw);
    #[unroll]
    for j in 0..comptime!(n_state / vw) {
        let cvec = c[base + j];
        let bvec = b[base + j];
        #[unroll]
        for k in 0..vw {
            inp[comptime!(4 + j * vw + k)] = f32::cast_from(bvec[k]);
            inp[comptime!(4 + n_state + j * vw + k)] = f32::cast_from(cvec[k]);
        }
    }
}

/// One forward step of unit `(bh, p)` from inputs [`scan_step_inputs`] loaded.
#[cube]
#[allow(clippy::too_many_arguments)]
fn scan_step<F: Float + CubeElement>(
    s: &mut Array<f32>,
    inp: &Array<f32>,
    y: &mut Array<F>,
    ckpt: &mut Array<f32>,
    bh: usize,
    bi: usize,
    h: usize,
    p: usize,
    seq: usize,
    heads: usize,
    head_dim: usize,
    segs: usize,
    d: f32,
    t: usize,
    #[comptime] n_state: usize,
    #[comptime] seg_len: usize,
    #[comptime] save_ckpt: bool,
) {
    // The state entering every backward segment, for the adjoint to start from
    // instead of re-running this loop.
    if comptime!(save_ckpt) {
        if t % seg_len == 0 {
            let cbase = ((bh * segs + t / seg_len) * n_state) * head_dim + p;
            #[unroll]
            for i in 0..n_state {
                ckpt[cbase + i * head_dim] = s[i];
            }
        }
    }
    let e = f32::exp(inp[0]);
    let gt = inp[1];
    let xp = inp[3];
    let wx = inp[2] * xp;
    let mut cs = 0.0f32;
    let mut cb = 0.0f32;
    #[unroll]
    for i in 0..n_state {
        let ci = inp[comptime!(4 + n_state + i)];
        let bv = inp[comptime!(4 + i)];
        cs += ci * s[i];
        cb += ci * bv;
        s[i] = e * s[i] + wx * bv;
    }
    let row = (bi * seq + t) * heads + h;
    y[row * head_dim + p] = F::cast_from(e * cs + (gt * cb + d) * xp);
}

#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn ssd_scan_backward_kernel<F: Float + CubeElement, V: Size>(
    dy: &Array<F>,
    x: &Array<F>,
    b: &Array<Vector<F, V>>,
    c: &Array<Vector<F, V>>,
    a: &Array<F>,
    a_head: &Array<F>,
    g: &Array<F>,
    w: &Array<F>,
    init: &Array<F>,
    skip: &Array<F>,
    ckpt: &mut Array<f32>,
    dx: &mut Array<F>,
    db: &mut Array<F>,
    dc: &mut Array<F>,
    da: &mut Array<F>,
    dg: &mut Array<F>,
    dw: &mut Array<F>,
    dinit: &mut Array<F>,
    head_part: &mut Array<f32>,
    skip_part: &mut Array<f32>,
    seq: usize,
    heads: usize,
    head_dim: usize,
    bh_total: usize,
    segs: usize,
    #[comptime] group: usize,
    #[comptime] groups: usize,
    #[comptime] max_planes: usize,
    #[comptime] n_state: usize,
    #[comptime] vw: usize,
    #[comptime] split_bits: u32,
    #[comptime] seg_len: usize,
    #[comptime] has_init: bool,
    #[comptime] use_planes: bool,
    #[comptime] rate: bool,
    #[comptime] has_skip: bool,
    #[comptime] has_ckpt: bool,
) {
    // `groups` (batch, head) pairs per cube, `group` units each. A pair's units are
    // `(p, part)` with `part` fastest: `nsplit` adjacent lanes share one row `p` of
    // the state and hold `n_per` of its columns each, so a row's dot products
    // close with an xor butterfly over the low lane bits and a column's sum over
    // `p` with one over the high bits. `group` is padded to whole planes, so no
    // plane straddles two pairs; padding units and pairs past the end run the same
    // loops on clamped indices — every unit reaches every barrier and shuffle —
    // and never write.
    let nsplit = comptime!(1usize << split_bits);
    let n_per = comptime!(n_state >> split_bits);
    let local = UNIT_POS_X as usize;
    let slot = local / group;
    let l = local % group;
    let p_raw = l / nsplit;
    let part = l % nsplit;
    let cube = CUBE_POS_Y as usize * CUBE_COUNT_X as usize + CUBE_POS_X as usize;
    let bh_raw = cube * groups + slot;
    let live_bh = bh_raw < bh_total;
    let live = live_bh && p_raw < head_dim;
    let bh = select(live_bh, bh_raw, bh_total - 1);
    let p = select(p_raw < head_dim, p_raw, 0usize);
    let h = bh % heads;
    let bi = bh / heads;
    let n0 = part * n_per;
    let mut d_skip = 0.0f32;
    if comptime!(has_skip) {
        d_skip = f32::cast_from(skip[h]);
    }
    let mut rate_h = 1.0f32;
    if comptime!(rate) {
        rate_h = f32::cast_from(a_head[h]);
    }

    // Without hardware planes (the CPU runtime) every unit is its own "plane" of
    // one: no shuffles, and each unit's partials go to shared memory directly.
    let mut width = 1usize;
    let mut lane = 0usize;
    let mut p_steps = 0usize;
    if comptime!(use_planes) {
        width = PLANE_DIM as usize;
        lane = UNIT_POS_PLANE as usize;
        // Butterfly steps over `p` inside a plane: masks nsplit, 2 nsplit, ... < width.
        p_steps = (PLANE_DIM.trailing_zeros() - split_bits) as usize;
    }
    let planes = group / width;
    let plane_in_group = l / width;
    // Columns: dB[N], dC[N], then dg, dw, da and the step's skip term.
    let cols = comptime!(2 * n_state + 4);
    let flush_per_unit = comptime!((seg_len * (2 * n_state + 4)).div_ceil(group));
    // One segment of per-plane partials; also holds the final per-unit sums.
    let mut red = SharedMemory::<f32>::new(comptime!(
        (groups * seg_len * max_planes * (2 * n_state + 4)).max(2 * groups * group)
    ));

    // Forward sweep, keeping S_{t0-1} at the start of every segment — unless the
    // forward pass already left those checkpoints behind.
    let mut s = Array::<f32>::new(n_per);
    #[unroll]
    for i in 0..n_per {
        if comptime!(has_init) {
            s[i] = f32::cast_from(init[(bh * head_dim + p) * n_state + n0 + i]);
        } else {
            s[i] = 0.0f32;
        }
    }
    let sweep = select(comptime!(has_ckpt), 0usize, seq);
    for t in 0..sweep {
        if live && t % seg_len == 0 {
            let base = ((bh * segs + t / seg_len) * n_state + n0) * head_dim + p;
            #[unroll]
            for i in 0..n_per {
                ckpt[base + i * head_dim] = s[i];
            }
        }
        let row = (bi * seq + t) * heads + h;
        let e = f32::exp(log_decay::<F>(a, a_head, row, h, rate));
        let wx = f32::cast_from(w[row]) * f32::cast_from(x[row * head_dim + p]);
        let vbase = (row * n_state + n0) / vw;
        #[unroll]
        for j in 0..comptime!(n_per / vw) {
            let bvec = b[vbase + j];
            #[unroll]
            for k in 0..vw {
                let i = comptime!(j * vw + k);
                s[i] = e * s[i] + wx * f32::cast_from(bvec[k]);
            }
        }
    }

    // `gs` is dL/dS_t for the step being processed; no gradient enters at the end.
    let mut gs = Array::<f32>::new(n_per);
    let mut hist = Array::<f32>::new(comptime!(seg_len * n_per));
    let mut bv = Array::<f32>::new(n_per);
    let mut cv = Array::<f32>::new(n_per);
    let mut pb = Array::<f32>::new(n_per);
    let mut pc = Array::<f32>::new(n_per);
    #[unroll]
    for i in 0..n_per {
        gs[i] = 0.0f32;
    }
    // Per-head parameter gradients, summed over time by the unit that owns the
    // `da` / skip column (the same unit every step).
    let mut head_acc = 0.0f32;
    let mut skip_acc = 0.0f32;
    for rs in 0..segs {
        let sg = segs - 1 - rs;
        let t0 = sg * seg_len;
        let rest = seq - t0;
        let len = select(rest < seg_len, rest, seg_len);

        // Recompute the segment's states from its checkpoint into registers:
        // `hist[k]` is S_{t0-1+k}, the state step `t0 + k` reads. `seg_len` is small
        // enough (see the launcher) for the whole segment to stay in registers, so
        // the walk back below needs no reversal of the recurrence — which is
        // unstable exactly where the decay is strong — and no second recompute.
        let cbase = ((bh * segs + sg) * n_state + n0) * head_dim + p;
        #[unroll]
        for i in 0..n_per {
            s[i] = ckpt[cbase + i * head_dim];
        }
        #[unroll]
        for k in 0..seg_len {
            if k < len {
                #[unroll]
                for i in 0..n_per {
                    hist[comptime!(k * n_per + i)] = s[i];
                }
                let row = (bi * seq + t0 + k) * heads + h;
                let e = f32::exp(log_decay::<F>(a, a_head, row, h, rate));
                let wx = f32::cast_from(w[row]) * f32::cast_from(x[row * head_dim + p]);
                let vbase = (row * n_state + n0) / vw;
                #[unroll]
                for j in 0..comptime!(n_per / vw) {
                    let bvec = b[vbase + j];
                    #[unroll]
                    for k2 in 0..vw {
                        let i = comptime!(j * vw + k2);
                        s[i] = e * s[i] + wx * f32::cast_from(bvec[k2]);
                    }
                }
            }
        }

        #[unroll]
        for kk in 0..seg_len {
            let k = comptime!(seg_len - 1 - kk);
            // Uniform across the cube: every pair has the same `len`.
            if k < len {
            let t = t0 + k;
            let row = (bi * seq + t) * heads + h;
            let base = row * n_state + n0;
            let at = log_decay::<F>(a, a_head, row, h, rate);
            let e = f32::exp(at);
            let gt = f32::cast_from(g[row]);
            let wt = f32::cast_from(w[row]);
            let xp = f32::cast_from(x[row * head_dim + p]);
            let dyp = select(live, f32::cast_from(dy[row * head_dim + p]), 0.0f32);
            #[unroll]
            for j in 0..comptime!(n_per / vw) {
                let bvec = b[base / vw + j];
                let cvec = c[base / vw + j];
                #[unroll]
                for k2 in 0..vw {
                    let i = comptime!(j * vw + k2);
                    bv[i] = f32::cast_from(bvec[k2]);
                    cv[i] = f32::cast_from(cvec[k2]);
                }
            }
            // S_{t-1}.
            #[unroll]
            for i in 0..n_per {
                s[i] = hist[comptime!(k * n_per + i)];
            }

            // Row dot products over this unit's columns, closed over the row's lanes.
            let mut cb = 0.0f32;
            let mut bg = 0.0f32;
            let mut cs = 0.0f32;
            let mut gsum = 0.0f32;
            #[unroll]
            for i in 0..n_per {
                cb += cv[i] * bv[i];
                bg += bv[i] * gs[i];
                cs += cv[i] * s[i];
                gsum += gs[i] * s[i];
            }
            #[unroll]
            for k2 in 0..split_bits {
                let m = 1u32 << k2;
                cb += plane_shuffle_xor(cb, m);
                bg += plane_shuffle_xor(bg, m);
                cs += plane_shuffle_xor(cs, m);
                gsum += plane_shuffle_xor(gsum, m);
            }
            if live && part == 0 {
                dx[row * head_dim + p] = F::cast_from((gt * cb + d_skip) * dyp + wt * bg);
            }

            // Column partials over this plane's rows.
            let xdy = xp * dyp;
            #[unroll]
            for i in 0..n_per {
                pb[i] = gt * cv[i] * xdy + wt * xp * gs[i];
                pc[i] = e * s[i] * dyp + gt * bv[i] * xdy;
            }
            // The row scalars are the same on every lane of a row; count them once.
            let first = select(part == 0, 1.0f32, 0.0f32);
            let mut sg_ = cb * xdy * first;
            let mut sw = xp * bg * first;
            let mut sa = e * (dyp * cs + gsum) * first;
            let mut sk = xdy * first;
            if comptime!(use_planes) {
                for k2 in 0..p_steps {
                    let m = (nsplit as u32) << (k2 as u32);
                    #[unroll]
                    for i in 0..n_per {
                        pb[i] += plane_shuffle_xor(pb[i], m);
                        pc[i] += plane_shuffle_xor(pc[i], m);
                    }
                    sg_ += plane_shuffle_xor(sg_, m);
                    sw += plane_shuffle_xor(sw, m);
                    sa += plane_shuffle_xor(sa, m);
                    sk += plane_shuffle_xor(sk, m);
                }
            }
            // Park this plane's partials for step `k`; nothing on the recurrence
            // waits for them, so the whole segment is combined behind one barrier.
            let rbase = ((slot * seg_len + k) * max_planes + plane_in_group) * cols;
            if lane < nsplit {
                #[unroll]
                for i in 0..n_per {
                    red[rbase + n0 + i] = pb[i];
                    red[rbase + n_state + n0 + i] = pc[i];
                }
                if lane == 0 {
                    red[rbase + 2 * n_state] = sg_;
                    red[rbase + 2 * n_state + 1] = sw;
                    red[rbase + 2 * n_state + 2] = sa;
                    red[rbase + 2 * n_state + 3] = sk;
                }
            }

            // dL/dS_{t-1} = e_t (dL/dS_t + C_t dy_t).
            #[unroll]
            for i in 0..n_per {
                gs[i] = e * (gs[i] + cv[i] * dyp);
            }
            }
        }

        // Combine the segment: its (step, column) sums spread over the pair's units.
        sync_cube();
        #[unroll]
        for j in 0..flush_per_unit {
            let item = l + j * group;
            let k = item / cols;
            let col = item % cols;
            if live_bh && item < comptime!(seg_len * (2 * n_state + 4)) && k < len {
                let row = (bi * seq + t0 + k) * heads + h;
                let mut total = 0.0f32;
                for q in 0..planes {
                    total += red[((slot * seg_len + k) * max_planes + q) * cols + col];
                }
                let obase = row * n_state;
                if col < n_state {
                    db[obase + col] = F::cast_from(total);
                } else if col < 2 * n_state {
                    dc[obase + col - n_state] = F::cast_from(total);
                } else if col == 2 * n_state {
                    dg[row] = F::cast_from(total);
                } else if col == 2 * n_state + 1 {
                    dw[row] = F::cast_from(total);
                } else if col == 2 * n_state + 2 {
                    // d/da; with a = dt * A[h] that is A[h] * d/da for `dt`
                    // and dt * d/da summed over time for A[h].
                    da[row] = F::cast_from(total * rate_h);
                    if comptime!(rate) {
                        head_acc += total * f32::cast_from(a[row]);
                    }
                } else {
                    skip_acc += total;
                }
            }
        }
        // The next segment overwrites the partials.
        sync_cube();
    }

    // The per-head sums over time were accumulated by whichever units owned those
    // columns at each step; gather them through shared memory.
    red[(slot * group + l) * 2] = head_acc;
    red[(slot * group + l) * 2 + 1] = skip_acc;
    sync_cube();
    if live_bh && l == 0 {
        let mut head_total = 0.0f32;
        let mut skip_total = 0.0f32;
        for q in 0..group {
            head_total += red[(slot * group + q) * 2];
            skip_total += red[(slot * group + q) * 2 + 1];
        }
        head_part[bh] = head_total;
        skip_part[bh] = skip_total;
    }

    if comptime!(has_init) {
        if live {
            #[unroll]
            for i in 0..n_per {
                dinit[(bh * head_dim + p) * n_state + n0 + i] = F::cast_from(gs[i]);
            }
        }
    }
}

/// Steps per chunk of [`ssd_scan_backward_chunked_kernel`].
const CHUNK: usize = 16;
/// Units per cube of the chunked backward: one cube per (batch, head) pair.
const CHUNK_UNITS: usize = 256;

/// `acc[i * tn + j] += Σ_k A(i0 + i·i_step, k) · B(k, j0 + j·j_step)` for a
/// `tm × tn` tile, both operands in `sm` at `A(i, k) = sm[a_off + i * a_si + k * a_sk]`
/// and `B(k, j) = sm[b_off + k * b_sk + j * b_sj]`; with `scaled`, `A(i, k)` is also
/// multiplied by `sm[s_off + k]`. Callers spread a tile's columns `j_step` apart so
/// that neighbouring units read neighbouring words (no bank conflicts) while the
/// rows they share are broadcasts.
#[cube]
#[allow(clippy::too_many_arguments)]
fn tile_mm(
    sm: &SharedMemory<f32>,
    a_off: usize,
    a_si: usize,
    a_sk: usize,
    b_off: usize,
    b_sk: usize,
    b_sj: usize,
    s_off: usize,
    i0: usize,
    j0: usize,
    acc: &mut Array<f32>,
    #[comptime] k_len: usize,
    #[comptime] tm: usize,
    #[comptime] i_step: usize,
    #[comptime] tn: usize,
    #[comptime] j_step: usize,
    #[comptime] scaled: bool,
) {
    let mut av = Array::<f32>::new(tm);
    #[unroll]
    for k in 0..k_len {
        let mut sk = 1.0f32;
        if comptime!(scaled) {
            sk = sm[s_off + k];
        }
        #[unroll]
        for i in 0..tm {
            av[i] = sm[a_off + (i0 + comptime!(i * i_step)) * a_si + k * a_sk] * sk;
        }
        #[unroll]
        for j in 0..tn {
            let bv = sm[b_off + k * b_sk + (j0 + comptime!(j * j_step)) * b_sj];
            #[unroll]
            for i in 0..tm {
                acc[comptime!(i * tn + j)] += av[i] * bv;
            }
        }
    }
}

/// Load chunk `sg`'s inputs of one (batch, head) pair into this unit's share of
/// registers (the order [`ssd_scan_backward_chunked_kernel`] parks them in shared
/// memory); rows past the end of the sequence read as zero.
#[cube]
#[allow(clippy::too_many_arguments)]
fn fetch_chunk<F: Float + CubeElement>(
    dy: &Array<F>,
    x: &Array<F>,
    b: &Array<F>,
    c: &Array<F>,
    a: &Array<F>,
    g: &Array<F>,
    w: &Array<F>,
    ckpt: &Array<f32>,
    rx: &mut Array<f32>,
    rbc: &mut Array<f32>,
    rs0: &mut Array<f32>,
    rv: &mut Array<f32>,
    l: usize,
    bi: usize,
    bh: usize,
    h: usize,
    seq: usize,
    heads: usize,
    segs: usize,
    sg: usize,
    #[comptime] pd: usize,
    #[comptime] nn: usize,
) {
    // Every load is unconditional (indices clamped, values zeroed by `select`): a
    // load under a branch makes the compiler wait for it at the join, which
    // serialises the chunk's loads on the full memory latency.
    let t0 = sg * comptime!(CHUNK);
    let rest = seq - t0;
    let len = select(rest < comptime!(CHUNK), rest, comptime!(CHUNK));
    let xn = comptime!(CHUNK * pd / CHUNK_UNITS);
    #[unroll]
    for i in 0..comptime!(2 * CHUNK * pd / CHUNK_UNITS) {
        let rem = l + comptime!((i % (CHUNK * pd / CHUNK_UNITS)) * CHUNK_UNITS);
        let k = rem / pd;
        let live = k < len;
        let idx = ((bi * seq + t0 + select(live, k, 0usize)) * heads + h) * pd + rem % pd;
        if comptime!(i < xn) {
            rx[i] = f32::cast_from(x[idx]);
        } else {
            rx[i] = f32::cast_from(dy[idx]);
        }
    }
    let bn = comptime!(CHUNK * nn / CHUNK_UNITS);
    #[unroll]
    for i in 0..comptime!(2 * CHUNK * nn / CHUNK_UNITS) {
        let rem = l + comptime!((i % (CHUNK * nn / CHUNK_UNITS)) * CHUNK_UNITS);
        let k = rem / nn;
        let live = k < len;
        let idx = ((bi * seq + t0 + select(live, k, 0usize)) * heads + h) * nn + rem % nn;
        if comptime!(i < bn) {
            rbc[i] = f32::cast_from(b[idx]);
        } else {
            rbc[i] = f32::cast_from(c[idx]);
        }
    }
    #[unroll]
    for i in 0..comptime!(pd * nn / CHUNK_UNITS) {
        // Checkpoint layout: [bh, seg, n, p].
        rs0[i] = ckpt[(bh * segs + sg) * comptime!(pd * nn) + l + comptime!(i * CHUNK_UNITS)];
    }
    let row = (bi * seq + t0 + select(l < len, l, 0usize)) * heads + h;
    rv[0] = f32::cast_from(a[row]);
    rv[1] = f32::cast_from(g[row]);
    rv[2] = f32::cast_from(w[row]);
}

/// Shared memory (floats) of [`ssd_scan_backward_chunked_kernel`]: rows of `x` / `dy`
/// padded to `head_dim + 1` and of `b` / `c` / the states to `d_state + 1` words so
/// a column walk across units never lands on one bank.
const fn chunked_shared(head_dim: usize, state: usize) -> usize {
    2 * CHUNK * (head_dim + 1) + 2 * CHUNK * (state + 1) + 2 * head_dim * (state + 1)
        + 4 * CHUNK * CHUNK
        + 8 * CHUNK
        + CHUNK * state
        + 2 * CHUNK_UNITS
        + 2 * CHUNK
}

/// The scan's adjoint in the chunked (SSD) form: one cube per (batch, head) walks
/// the chunks of [`CHUNK`] steps backward, carrying `dS` (the gradient of the state
/// leaving the chunk) in shared memory. Inside a chunk, with `A_t` the cumulative
/// log decay from the chunk start, `S0` the entering state (a forward checkpoint),
/// `M[t, τ] = (C_t·B_τ)·coef[t, τ]`, `coef = exp(A_t − A_τ) w_τ` below the diagonal
/// and `g_t` on it, `u_τ = exp(A_end − A_τ) w_τ`:
///
/// ```text
/// dX = Mᵀ dY + diag(u) B dSᵀ + D dY          dM = dY Xᵀ,  Q = dM ⊙ coef
/// dC = Q B + diag(exp A) dY S0               dB = Qᵀ C + diag(u) X dS
/// dS ← dYᵀ diag(exp A) C + exp(A_end) dS
/// ```
///
/// and the log-decay gradient from the same pieces, reverse-cumsummed over the
/// chunk. Every product is a small register-tiled matmul over shared memory, where
/// the per-step adjoint spent its time on cross-lane reductions.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn ssd_scan_backward_chunked_kernel<F: Float + CubeElement>(
    dy: &Array<F>,
    x: &Array<F>,
    b: &Array<F>,
    c: &Array<F>,
    a: &Array<F>,
    a_head: &Array<F>,
    g: &Array<F>,
    w: &Array<F>,
    skip: &Array<F>,
    ckpt: &Array<f32>,
    dx: &mut Array<F>,
    db: &mut Array<F>,
    dc: &mut Array<F>,
    da: &mut Array<F>,
    dg: &mut Array<F>,
    dw: &mut Array<F>,
    dinit: &mut Array<F>,
    head_part: &mut Array<f32>,
    skip_part: &mut Array<f32>,
    seq: usize,
    heads: usize,
    segs: usize,
    #[comptime] pd: usize,
    #[comptime] nn: usize,
    #[comptime] has_init: bool,
    #[comptime] rate: bool,
    #[comptime] has_skip: bool,
) {
    let ll = comptime!(CHUNK);
    let units = comptime!(CHUNK_UNITS);
    // Padded row strides.
    let xs = comptime!(pd + 1);
    let ns = comptime!(nn + 1);
    let l = UNIT_POS_X as usize;
    let bh = CUBE_POS_Y as usize * CUBE_COUNT_X as usize + CUBE_POS_X as usize;
    let h = bh % heads;
    let bi = bh / heads;
    let mut d_skip = 0.0f32;
    if comptime!(has_skip) {
        d_skip = f32::cast_from(skip[h]);
    }
    let mut rate_h = 1.0f32;
    if comptime!(rate) {
        rate_h = f32::cast_from(a_head[h]);
    }

    // Shared layout (floats): x, dy [L, pd+1]; b, c [L, nn+1]; S0, dS [pd, nn+1];
    // CB/M, dM/Q, K, Kw [L, L]; per-step vectors; row-dot partials; reduction pads.
    let o_x = 0usize;
    let o_dy = comptime!(CHUNK * (pd + 1));
    let o_b = comptime!(2 * CHUNK * (pd + 1));
    let o_c = comptime!(2 * CHUNK * (pd + 1) + CHUNK * (nn + 1));
    let o_s0 = comptime!(2 * CHUNK * (pd + 1) + 2 * CHUNK * (nn + 1));
    let o_ds = comptime!(2 * CHUNK * (pd + 1) + 2 * CHUNK * (nn + 1) + pd * (nn + 1));
    let o_cb = comptime!(2 * CHUNK * (pd + 1) + 2 * CHUNK * (nn + 1) + 2 * pd * (nn + 1));
    let o_dm = o_cb + comptime!(CHUNK * CHUNK);
    let o_k = o_cb + comptime!(2 * CHUNK * CHUNK);
    let o_kw = o_cb + comptime!(3 * CHUNK * CHUNK);
    let o_v = o_cb + comptime!(4 * CHUNK * CHUNK);
    // Per-step vectors: raw a, g, w, A (cumulative), exp A, z = exp(A_end − A), u = z w, dA.
    let o_ar = o_v;
    let o_g = o_v + comptime!(CHUNK);
    let o_w = o_v + comptime!(2 * CHUNK);
    let o_ac = o_v + comptime!(3 * CHUNK);
    let o_ea = o_v + comptime!(4 * CHUNK);
    let o_z = o_v + comptime!(5 * CHUNK);
    let o_u = o_v + comptime!(6 * CHUNK);
    let o_da = o_v + comptime!(7 * CHUNK);
    // Row-dot partials over the column tiles of dC / dB (two columns each).
    let o_ysc = o_v + comptime!(8 * CHUNK);
    let o_xsb = o_ysc + comptime!(CHUNK * nn / 2);
    let o_red = o_xsb + comptime!(CHUNK * nn / 2);
    let o_tail = o_red + comptime!(2 * CHUNK_UNITS);
    let mut sm = SharedMemory::<f32>::new(comptime!(chunked_shared(pd, nn)));

    // dS leaving the last chunk: no gradient enters after the sequence.
    #[unroll]
    for i in 0..comptime!((pd * (nn + 1)).div_ceil(CHUNK_UNITS)) {
        let item = l + i * units;
        if item < comptime!(pd * (nn + 1)) {
            sm[o_ds + item] = 0.0f32;
        }
    }
    let mut head_acc = 0.0f32;
    let mut skip_acc = 0.0f32;
    let mut rx = Array::<f32>::new(comptime!(2 * CHUNK * pd / CHUNK_UNITS));
    let mut rbc = Array::<f32>::new(comptime!(2 * CHUNK * nn / CHUNK_UNITS));
    let mut rs0 = Array::<f32>::new(comptime!(pd * nn / CHUNK_UNITS));
    let mut rv = Array::<f32>::new(3usize);
    fetch_chunk::<F>(
        dy, x, b, c, a, g, w, ckpt, &mut rx, &mut rbc, &mut rs0, &mut rv, l, bi, bh, h, seq,
        heads, segs, segs - 1, pd, nn,
    );

    for rs in 0..segs {
        let sg = segs - 1 - rs;
        let t0 = sg * ll;
        let rest = seq - t0;
        let len = select(rest < ll, rest, ll);

        // P1: park the prefetched chunk in shared memory, then start fetching the
        // next one; its loads are in flight while this chunk computes.
        #[unroll]
        for i in 0..comptime!(2 * CHUNK * pd / CHUNK_UNITS) {
            let rem = l + comptime!((i % (CHUNK * pd / CHUNK_UNITS)) * CHUNK_UNITS);
            let base = comptime!(if i < CHUNK * pd / CHUNK_UNITS { 0 } else { CHUNK * (pd + 1) });
            sm[base + (rem / pd) * xs + rem % pd] = select(rem / pd < len, rx[i], 0.0f32);
        }
        #[unroll]
        for i in 0..comptime!(2 * CHUNK * nn / CHUNK_UNITS) {
            let rem = l + comptime!((i % (CHUNK * nn / CHUNK_UNITS)) * CHUNK_UNITS);
            let base = comptime!(if i < CHUNK * nn / CHUNK_UNITS { 0 } else { CHUNK * (nn + 1) });
            sm[o_b + base + (rem / nn) * ns + rem % nn] = select(rem / nn < len, rbc[i], 0.0f32);
        }
        #[unroll]
        for i in 0..comptime!(pd * nn / CHUNK_UNITS) {
            let item = l + comptime!(i * CHUNK_UNITS);
            sm[o_s0 + (item % pd) * ns + item / pd] = rs0[i];
        }
        // Values fetched for rows past the end are dropped here, where the
        // registers are first read, so the fetch never waits on its loads.
        if l < ll {
            let live = l < len;
            let mut ar = rv[0];
            if comptime!(rate) {
                ar *= rate_h;
            }
            sm[o_ar + l] = select(live, ar, 0.0f32);
            sm[o_g + l] = select(live, rv[1], 0.0f32);
            sm[o_w + l] = select(live, rv[2], 0.0f32);
        }
        sync_cube();
        if rs + 1 < segs {
            fetch_chunk::<F>(
                dy, x, b, c, a, g, w, ckpt, &mut rx, &mut rbc, &mut rs0, &mut rv, l, bi, bh, h, seq,
                heads, segs, sg - 1, pd, nn,
            );
        }

        // P2: cumulative decay; CB = C Bᵀ and dM = dY Xᵀ, one entry per unit.
        if l < ll {
            let mut acum = 0.0f32;
            let mut aend = 0.0f32;
            #[unroll]
            for j in 0..comptime!(CHUNK) {
                let v = sm[o_ar + j];
                if j <= l {
                    acum += v;
                }
                aend += v;
            }
            sm[o_ac + l] = acum;
            sm[o_ea + l] = f32::exp(acum);
            let z = f32::exp(aend - acum);
            sm[o_z + l] = z;
            sm[o_u + l] = z * sm[o_w + l];
        }
        #[unroll]
        for i in 0..comptime!((CHUNK * CHUNK).div_ceil(CHUNK_UNITS)) {
            let item = l + i * units;
            if item < comptime!(CHUNK * CHUNK) {
                let t = item / ll;
                let tau = item % ll;
                let mut cb = 0.0f32;
                #[unroll]
                for n in 0..nn {
                    cb += sm[o_c + t * ns + n] * sm[o_b + tau * ns + n];
                }
                let mut dm = 0.0f32;
                #[unroll]
                for p in 0..pd {
                    dm += sm[o_dy + t * xs + p] * sm[o_x + tau * xs + p];
                }
                sm[o_cb + item] = cb;
                sm[o_dm + item] = dm;
            }
        }
        sync_cube();

        // P3: M = CB ⊙ coef (over CB), Q = dM ⊙ coef (over dM), and the decay
        // gradient's pieces K = dM CB exp(A_t − A_τ) w_τ (and without the w_τ).
        #[unroll]
        for i in 0..comptime!((CHUNK * CHUNK).div_ceil(CHUNK_UNITS)) {
            let item = l + i * units;
            if item < comptime!(CHUNK * CHUNK) {
                let t = item / ll;
                let tau = item % ll;
                let cb = sm[o_cb + item];
                let dm = sm[o_dm + item];
                let mut coef = 0.0f32;
                let mut kw = 0.0f32;
                if tau < t {
                    let dec = f32::exp(sm[o_ac + t] - sm[o_ac + tau]);
                    coef = dec * sm[o_w + tau];
                    kw = dm * cb * dec;
                } else if tau == t {
                    coef = sm[o_g + t];
                    if t < len {
                        let row = (bi * seq + t0 + t) * heads + h;
                        dg[row] = F::cast_from(dm * cb);
                        skip_acc += dm;
                    }
                }
                sm[o_cb + item] = cb * coef;
                sm[o_dm + item] = dm * coef;
                sm[o_k + item] = kw * sm[o_w + tau];
                sm[o_kw + item] = kw;
            }
        }
        sync_cube();

        // P4: dX, dC, dB tiles straight to memory; the row dots the decay gradient
        // needs (dY S0 · C and X dS · B) as per-tile partials.
        // dX: rows {i0, i0 + L/2}, columns {j0, j0 + pd/2}.
        #[unroll]
        for i in 0..comptime!((CHUNK * pd / 4).div_ceil(CHUNK_UNITS)) {
            let tile = l + i * units;
            if tile < comptime!(CHUNK * pd / 4) {
                let i0 = tile / comptime!(pd / 2);
                let j0 = tile % comptime!(pd / 2);
                let mut acc = Array::<f32>::new(4usize);
                let mut acc2 = Array::<f32>::new(4usize);
                #[unroll]
                for q in 0..4usize {
                    acc[q] = 0.0f32;
                    acc2[q] = 0.0f32;
                }
                // Mᵀ dY: A(i, k) = M[k, i].
                tile_mm(&sm, o_cb, 1usize, ll, o_dy, xs, 1usize, 0usize, i0, j0, &mut acc,
                    comptime!(CHUNK), 2usize, comptime!(CHUNK / 2), 2usize, comptime!(pd / 2), false);
                // B dSᵀ: A(i, k) = B[i, k], B(k, j) = dS[j, k].
                tile_mm(&sm, o_b, ns, 1usize, o_ds, 1usize, ns, 0usize, i0, j0, &mut acc2,
                    nn, 2usize, comptime!(CHUNK / 2), 2usize, comptime!(pd / 2), false);
                #[unroll]
                for ii in 0..2usize {
                    let r = i0 + comptime!(ii * (CHUNK / 2));
                    if r < len {
                        let u = sm[o_u + r];
                        let row = (bi * seq + t0 + r) * heads + h;
                        #[unroll]
                        for jj in 0..2usize {
                            let col = j0 + comptime!(jj * (pd / 2));
                            let v = acc[comptime!(ii * 2 + jj)] + u * acc2[comptime!(ii * 2 + jj)]
                                + d_skip * sm[o_dy + r * xs + col];
                            dx[row * pd + col] = F::cast_from(v);
                        }
                    }
                }
            }
        }
        // dC, dB: row i0, columns {j0, j0 + nn/2}.
        #[unroll]
        for i in 0..comptime!((CHUNK * nn / 2).div_ceil(CHUNK_UNITS)) {
            let tile = l + i * units;
            if tile < comptime!(CHUNK * nn / 2) {
                let i0 = tile / comptime!(nn / 2);
                let j0 = tile % comptime!(nn / 2);
                let mut acc = Array::<f32>::new(2usize);
                let mut acc2 = Array::<f32>::new(2usize);
                // dC = Q B + diag(exp A) dY S0.
                acc[0] = 0.0f32;
                acc[1] = 0.0f32;
                acc2[0] = 0.0f32;
                acc2[1] = 0.0f32;
                tile_mm(&sm, o_dm, ll, 1usize, o_b, ns, 1usize, 0usize, i0, j0, &mut acc,
                    comptime!(CHUNK), 1usize, 1usize, 2usize, comptime!(nn / 2), false);
                tile_mm(&sm, o_dy, xs, 1usize, o_s0, ns, 1usize, 0usize, i0, j0, &mut acc2,
                    pd, 1usize, 1usize, 2usize, comptime!(nn / 2), false);
                let ea = sm[o_ea + i0];
                let row = (bi * seq + t0 + i0) * heads + h;
                let mut part = 0.0f32;
                #[unroll]
                for jj in 0..2usize {
                    let col = j0 + comptime!(jj * (nn / 2));
                    part += acc2[jj] * sm[o_c + i0 * ns + col];
                    if i0 < len {
                        dc[row * nn + col] = F::cast_from(acc[jj] + ea * acc2[jj]);
                    }
                }
                sm[o_ysc + i0 * comptime!(nn / 2) + j0] = part;
                // dB = Qᵀ C + diag(u) X dS.
                acc[0] = 0.0f32;
                acc[1] = 0.0f32;
                acc2[0] = 0.0f32;
                acc2[1] = 0.0f32;
                tile_mm(&sm, o_dm, 1usize, ll, o_c, ns, 1usize, 0usize, i0, j0, &mut acc,
                    comptime!(CHUNK), 1usize, 1usize, 2usize, comptime!(nn / 2), false);
                tile_mm(&sm, o_x, xs, 1usize, o_ds, ns, 1usize, 0usize, i0, j0, &mut acc2,
                    pd, 1usize, 1usize, 2usize, comptime!(nn / 2), false);
                let u = sm[o_u + i0];
                let mut part_b = 0.0f32;
                #[unroll]
                for jj in 0..2usize {
                    let col = j0 + comptime!(jj * (nn / 2));
                    part_b += acc2[jj] * sm[o_b + i0 * ns + col];
                    if i0 < len {
                        db[row * nn + col] = F::cast_from(acc[jj] + u * acc2[jj]);
                    }
                }
                sm[o_xsb + i0 * comptime!(nn / 2) + j0] = part_b;
            }
        }
        // <dS, S0>, for the decay of the whole chunk. Only real columns: the pad
        // column of S0 is never written and may hold anything, NaN included.
        let mut sdot = 0.0f32;
        #[unroll]
        for i in 0..comptime!(pd * nn / CHUNK_UNITS) {
            let item = l + comptime!(i * CHUNK_UNITS);
            let at = (item / nn) * ns + item % nn;
            sdot += sm[o_ds + at] * sm[o_s0 + at];
        }
        sm[o_red + l] = sdot;
        sync_cube();

        // P5: the decay gradient per step, then dS for the previous chunk.
        if l < ll {
            let t = l;
            let mut rowk = 0.0f32;
            let mut colk = 0.0f32;
            let mut colkw = 0.0f32;
            #[unroll]
            for j in 0..comptime!(CHUNK) {
                rowk += sm[o_k + t * ll + j];
                colk += sm[o_k + j * ll + t];
                colkw += sm[o_kw + j * ll + t];
            }
            let mut ysc = 0.0f32;
            let mut xsb = 0.0f32;
            #[unroll]
            for j in 0..comptime!(nn / 2) {
                ysc += sm[o_ysc + t * comptime!(nn / 2) + j];
                xsb += sm[o_xsb + t * comptime!(nn / 2) + j];
            }
            let uxsb = sm[o_u + t] * xsb;
            sm[o_da + t] = rowk - colk + sm[o_ea + t] * ysc - uxsb;
            // The chunk-end terms (through the state leaving the chunk) belong to
            // A_end, which every step's `a` feeds: gathered once, added to all.
            sm[o_tail + t] = uxsb;
            if t < len {
                let row = (bi * seq + t0 + t) * heads + h;
                dw[row] = F::cast_from(colkw + sm[o_z + t] * xsb);
            }
        } else if l < comptime!(2 * CHUNK) {
            let q = l - ll;
            let mut sd = 0.0f32;
            #[unroll]
            for j in 0..comptime!(CHUNK_UNITS / CHUNK) {
                sd += sm[o_red + q * comptime!(CHUNK_UNITS / CHUNK) + j];
            }
            sm[o_tail + ll + q] = sd;
        }
        let eend = f32::exp(sm[o_ac + comptime!(CHUNK - 1)]);
        // dS: rows p {i0 + k pd/4}, columns {j0, j0 + nn/2}.
        #[unroll]
        for i in 0..comptime!((pd * nn / 8).div_ceil(CHUNK_UNITS)) {
            let tile = l + i * units;
            if tile < comptime!(pd * nn / 8) {
                let i0 = tile / comptime!(nn / 2);
                let j0 = tile % comptime!(nn / 2);
                let mut acc = Array::<f32>::new(8usize);
                #[unroll]
                for q in 0..8usize {
                    acc[q] = 0.0f32;
                }
                // dYᵀ diag(exp A) C: A(i = p, k = t) = dY[t, p] exp(A_t).
                tile_mm(&sm, o_dy, 1usize, xs, o_c, ns, 1usize, o_ea, i0, j0, &mut acc,
                    comptime!(CHUNK), 4usize, comptime!(pd / 4), 2usize, comptime!(nn / 2), true);
                #[unroll]
                for ii in 0..4usize {
                    #[unroll]
                    for jj in 0..2usize {
                        let idx = o_ds + (i0 + comptime!(ii * (pd / 4))) * ns + j0 + comptime!(jj * (nn / 2));
                        sm[idx] = acc[comptime!(ii * 2 + jj)] + eend * sm[idx];
                    }
                }
            }
        }
        sync_cube();

        // da_j = Σ_{t ≥ j} dA_t; with a = dt · A[h], dt gets A[h] da and A[h] dt da.
        if l < len {
            let mut total = 0.0f32;
            let mut sd = 0.0f32;
            #[unroll]
            for j in 0..comptime!(CHUNK) {
                if j >= l {
                    total += sm[o_da + j];
                }
                total += sm[o_tail + j];
                sd += sm[o_tail + ll + j];
            }
            total += sm[o_ea + comptime!(CHUNK - 1)] * sd;
            let row = (bi * seq + t0 + l) * heads + h;
            da[row] = F::cast_from(total * rate_h);
            if comptime!(rate) {
                head_acc += total * f32::cast_from(a[row]);
            }
        }
    }

    if comptime!(has_init) {
        #[unroll]
        for i in 0..comptime!((pd * nn).div_ceil(CHUNK_UNITS)) {
            let item = l + i * units;
            if item < comptime!(pd * nn) {
                let p = item / nn;
                let n = item % nn;
                dinit[bh * comptime!(pd * nn) + item] = F::cast_from(sm[o_ds + p * ns + n]);
            }
        }
    }
    // Per-head sums over time: gather each unit's share.
    sync_cube();
    sm[o_red + l] = head_acc;
    sm[o_red + units + l] = skip_acc;
    sync_cube();
    if l == 0 {
        let mut head_total = 0.0f32;
        let mut skip_total = 0.0f32;
        for q in 0..units {
            head_total += sm[o_red + q];
            skip_total += sm[o_red + units + q];
        }
        head_part[bh] = head_total;
        skip_part[bh] = skip_total;
    }
}

/// Shapes shared by the forward and backward entry points.
struct ScanDims {
    batch: usize,
    seq: usize,
    heads: usize,
    head_dim: usize,
    state: usize,
}

#[allow(clippy::too_many_arguments)]
fn scan_dims<R: Runtime, E: FloatElem>(
    x: &Tensor<R, E>,
    b: &Tensor<R, E>,
    c: &Tensor<R, E>,
    decay: &ScanDecay<'_, R, E>,
    g: &Tensor<R, E>,
    w: &Tensor<R, E>,
    init: Option<&Tensor<R, E>>,
    skip: Option<&Tensor<R, E>>,
) -> Result<ScanDims> {
    x.shape().expect_rank(4)?;
    b.shape().expect_rank(4)?;
    let d = x.dims();
    let (batch, seq, heads, head_dim) = (d[0], d[1], d[2], d[3]);
    let state = b.dims()[3];
    let bad = |what: &str| Error::shape(format!("ssd_scan: {what} disagrees with x {}", x.shape()));
    if b.dims() != [batch, seq, heads, state] {
        return Err(bad("b"));
    }
    if c.dims() != [batch, seq, heads, state] {
        return Err(bad("c"));
    }
    let a: &Tensor<R, E> = match decay {
        ScanDecay::Log(a) => a,
        ScanDecay::Rate { dt, a_head } => {
            if a_head.dims() != [heads] {
                return Err(bad("a_head"));
            }
            dt
        }
    };
    for (name, t) in [("a", a), ("g", g), ("w", w)] {
        if t.dims() != [batch, seq, heads] {
            return Err(bad(name));
        }
    }
    if let Some(init) = init
        && init.dims() != [batch, heads, head_dim, state]
    {
        return Err(bad("initial state"));
    }
    if let Some(skip) = skip
        && skip.dims() != [heads]
    {
        return Err(bad("skip"));
    }
    Ok(ScanDims {
        batch,
        seq,
        heads,
        head_dim,
        state,
    })
}

/// Vector width for the scan's `b` / `c` loads: one the device supports that
/// divides `cols`, the state columns one unit reads.
fn state_vector_width<R: Runtime, E: FloatElem>(client: &ComputeClient<R>, cols: usize) -> usize {
    crate::backend::line_size_for::<R, E>(client, cols).min(4).max(1)
}

/// How the backward kernel lays a problem out; the forward pass needs the same
/// segment length to leave checkpoints the backward can use.
struct BackwardLayout {
    use_planes: bool,
    split_bits: u32,
    vw: usize,
    seg_len: usize,
    segs: usize,
    group: usize,
    groups: usize,
    max_planes: usize,
}

fn backward_layout<R: Runtime, E: FloatElem>(
    client: &ComputeClient<R>,
    s: &ScanDims,
) -> BackwardLayout {
    let hw = &client.properties().hardware;
    let (plane_max, plane_min) = (hw.plane_size_max.max(1) as usize, hw.plane_size_min.max(1) as usize);
    let use_planes = plane_max > 1;
    // Up to four lanes per state row, each holding a quarter of its columns: the
    // state row and a segment of its history in registers is what bounds
    // occupancy, and the extra lanes are cheap to join with a two-step butterfly.
    // The split must fit inside the smallest plane and divide `d_state`.
    let mut split_bits = 0u32;
    while split_bits < 2
        && s.state.is_multiple_of(1 << (split_bits + 1))
        && (1usize << (split_bits + 1)) <= plane_min
    {
        split_bits += 1;
    }
    let vw = state_vector_width::<R, E>(client, s.state >> split_bits);
    // A segment's states live in registers: `seg_len` steps of `n_per` columns.
    let n_per = s.state >> split_bits;
    let seg_len = if chunked_backward(client, s) {
        CHUNK
    } else {
        (HIST_REGS / n_per.max(1)).clamp(1, SEG).min(s.seq.max(1))
    };
    let segs = s.seq.div_ceil(seg_len);
    let group = (s.head_dim << split_bits).div_ceil(plane_max) * plane_max;
    let groups = (256 / group).max(1).min((s.batch * s.heads).max(1));
    let max_planes = group / plane_min.min(plane_max);
    BackwardLayout {
        use_planes,
        split_bits,
        vw,
        seg_len,
        segs,
        group,
        groups,
        max_planes,
    }
}

/// Whether the backward takes the chunked kernel: tile-aligned shapes whose
/// chunk fits in shared memory. `MAMBA3_SCAN_BACKWARD=recurrent` forces the
/// per-step kernel (read once per process, so forward and backward agree on the
/// checkpoint spacing).
fn chunked_backward<R: Runtime>(client: &ComputeClient<R>, s: &ScanDims) -> bool {
    static FORCE_RECURRENT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let recurrent = *FORCE_RECURRENT.get_or_init(|| {
        std::env::var("MAMBA3_SCAN_BACKWARD").is_ok_and(|v| v == "recurrent")
    });
    let shared = client.properties().hardware.max_shared_memory_size;
    !recurrent
        && (CHUNK * s.head_dim).is_multiple_of(CHUNK_UNITS)
        && (CHUNK * s.state).is_multiple_of(CHUNK_UNITS)
        && (s.head_dim * s.state).is_multiple_of(CHUNK_UNITS)
        && s.head_dim.is_multiple_of(4)
        && s.state.is_multiple_of(2)
        && chunked_shared(s.head_dim, s.state) * 4 <= shared.min(48 * 1024)
}

/// The per-segment states [`ssd_scan_saving`] leaves for [`ssd_scan_backward`].
pub struct ScanCheckpoints<R: Runtime> {
    states: Tensor<R, f32>,
}

/// Whether [`ssd_scan`] takes a problem of this shape.
pub fn ssd_scan_supported(head_dim: usize, state: usize) -> bool {
    (1..=MAX_FUSED_STATE).contains(&state) && (1..=MAX_FUSED_HEAD_DIM).contains(&head_dim)
}

/// `y [B, T, H, P]` of the scan (see the module docs) in one launch, plus
/// `skip[h] * x` when a skip gain `[H]` is given.
#[allow(clippy::too_many_arguments)]
pub fn ssd_scan<R: Runtime, E: FloatElem>(
    x: &Tensor<R, E>,
    b: &Tensor<R, E>,
    c: &Tensor<R, E>,
    decay: ScanDecay<'_, R, E>,
    g: &Tensor<R, E>,
    w: &Tensor<R, E>,
    init: Option<&Tensor<R, E>>,
    skip: Option<&Tensor<R, E>>,
) -> Result<Tensor<R, E>> {
    Ok(ssd_scan_saving(x, b, c, decay, g, w, init, skip, false)?.0)
}

/// [`ssd_scan`], also returning the per-segment states the backward pass would
/// otherwise recompute with a full forward sweep of its own, when `save` is set.
#[allow(clippy::too_many_arguments)]
pub fn ssd_scan_saving<R: Runtime, E: FloatElem>(
    x: &Tensor<R, E>,
    b: &Tensor<R, E>,
    c: &Tensor<R, E>,
    decay: ScanDecay<'_, R, E>,
    g: &Tensor<R, E>,
    w: &Tensor<R, E>,
    init: Option<&Tensor<R, E>>,
    skip: Option<&Tensor<R, E>>,
    save: bool,
) -> Result<(Tensor<R, E>, Option<ScanCheckpoints<R>>)> {
    let _op = crate::backend::tally_op_scope("ssd_scan");
    let s = scan_dims(x, b, c, &decay, g, w, init, skip)?;
    if !ssd_scan_supported(s.head_dim, s.state) {
        return Err(Error::Unsupported(format!(
            "ssd_scan: head_dim {} / d_state {} out of range",
            s.head_dim, s.state
        )));
    }
    let y = Tensor::empty(x.shape().clone(), x.device());
    if y.is_empty() {
        return Ok((y, None));
    }
    let placeholder = Tensor::<R, E>::empty(Shape::new(vec![1]), x.device());
    let (a, a_head, rate) = match decay {
        ScanDecay::Log(a) => (a, &placeholder, false),
        ScanDecay::Rate { dt, a_head } => (dt, a_head, true),
    };
    let layout = backward_layout::<R, E>(x.client(), &s);
    let bh = s.batch * s.heads;
    let ckpt = Tensor::<R, f32>::empty(
        Shape::new(vec![if save { bh * layout.segs * s.state * s.head_dim } else { 1 }]),
        x.device(),
    );
    let lanes = bh * s.head_dim;
    let vw = state_vector_width::<R, E>(x.client(), s.state);
    let (count, dim) = launch_1d(x.client(), lanes, s.seq * s.state * 4);
    unsafe {
        ssd_scan_kernel::launch_unchecked::<E, R>(
            x.client(),
            count,
            dim,
            vw,
            x.arg(),
            b.arg(),
            c.arg(),
            a.arg(),
            a_head.arg(),
            g.arg(),
            w.arg(),
            init.unwrap_or(&placeholder).arg(),
            skip.unwrap_or(&placeholder).arg(),
            y.arg(),
            ckpt.arg(),
            s.seq,
            s.heads,
            s.head_dim,
            lanes,
            layout.segs,
            s.state,
            vw,
            init.is_some(),
            rate,
            skip.is_some(),
            layout.seg_len,
            save,
        );
    }
    Ok((y, save.then_some(ScanCheckpoints { states: ckpt })))
}

/// Gradients of [`ssd_scan`].
pub struct ScanGrads<R: Runtime, E: FloatElem> {
    /// `[B, T, H, P]`.
    pub dx: Tensor<R, E>,
    /// `[B, T, H, N]`.
    pub db: Tensor<R, E>,
    /// `[B, T, H, N]`.
    pub dc: Tensor<R, E>,
    /// For `a` ([`ScanDecay::Log`]) or for `dt` ([`ScanDecay::Rate`]), `[B, T, H]`.
    pub d_decay: Tensor<R, E>,
    /// For `a_head` ([`ScanDecay::Rate`] only), `[H]`.
    pub d_a_head: Option<Tensor<R, E>>,
    /// `[B, T, H]`.
    pub dg: Tensor<R, E>,
    /// `[B, T, H]`.
    pub dw: Tensor<R, E>,
    /// For the initial state, when there was one.
    pub d_init: Option<Tensor<R, E>>,
    /// For the skip gain, when there was one, `[H]`.
    pub d_skip: Option<Tensor<R, E>>,
}

/// The adjoint of [`ssd_scan`]: one launch, plus one small fold per per-head
/// parameter.
#[allow(clippy::too_many_arguments)]
pub fn ssd_scan_backward<R: Runtime, E: FloatElem>(
    dy: &Tensor<R, E>,
    x: &Tensor<R, E>,
    b: &Tensor<R, E>,
    c: &Tensor<R, E>,
    decay: ScanDecay<'_, R, E>,
    g: &Tensor<R, E>,
    w: &Tensor<R, E>,
    init: Option<&Tensor<R, E>>,
    skip: Option<&Tensor<R, E>>,
    saved: Option<&ScanCheckpoints<R>>,
) -> Result<ScanGrads<R, E>> {
    let _op = crate::backend::tally_op_scope("ssd_scan_backward");
    let s = scan_dims(x, b, c, &decay, g, w, init, skip)?;
    if dy.shape() != x.shape() {
        return Err(Error::shape(format!(
            "ssd_scan_backward: dy {} vs x {}",
            dy.shape(),
            x.shape()
        )));
    }
    let device = x.device();
    // Two placeholders: a buffer must never be bound to a read slot and a write slot
    // of the same launch.
    let placeholder = Tensor::<R, E>::empty(Shape::new(vec![1]), device);
    let placeholder_out = Tensor::<R, E>::empty(Shape::new(vec![1]), device);
    let (a, a_head, rate) = match decay {
        ScanDecay::Log(a) => (a, &placeholder, false),
        ScanDecay::Rate { dt, a_head } => (dt, a_head, true),
    };
    let mut grads = ScanGrads {
        dx: Tensor::empty(x.shape().clone(), device),
        db: Tensor::empty(b.shape().clone(), device),
        dc: Tensor::empty(c.shape().clone(), device),
        d_decay: Tensor::empty(a.shape().clone(), device),
        d_a_head: None,
        dg: Tensor::empty(g.shape().clone(), device),
        dw: Tensor::empty(w.shape().clone(), device),
        d_init: init.map(|i| Tensor::empty(i.shape().clone(), device)),
        d_skip: None,
    };
    let bh = s.batch * s.heads;
    if grads.dx.is_empty() {
        grads.d_a_head = rate.then(|| Tensor::zeros(vec![s.heads], device));
        grads.d_skip = skip.map(|_| Tensor::zeros(vec![s.heads], device));
        return Ok(grads);
    }
    let head_part = Tensor::<R, f32>::empty(Shape::new(vec![bh]), device);
    let skip_part = Tensor::<R, f32>::empty(Shape::new(vec![bh]), device);

    if chunked_backward(x.client(), &s) {
        let segs = s.seq.div_ceil(CHUNK);
        let fresh;
        let ckpt = match saved {
            Some(saved) if saved.states.len() == bh * segs * s.state * s.head_dim => &saved.states,
            _ => {
                let decay_again = match &decay {
                    ScanDecay::Log(a) => ScanDecay::Log(*a),
                    ScanDecay::Rate { dt, a_head } => ScanDecay::Rate { dt: *dt, a_head: *a_head },
                };
                fresh = ssd_scan_saving(x, b, c, decay_again, g, w, init, skip, true)?
                    .1
                    .map(|saved| saved.states);
                fresh.as_ref().ok_or_else(|| Error::Unsupported("ssd_scan_backward: no checkpoints".into()))?
            }
        };
        let x_cubes = bh.min(u16::MAX as usize);
        let count = CubeCount::Static(x_cubes as u32, bh.div_ceil(x_cubes) as u32, 1);
        crate::backend::count_launch();
        unsafe {
            ssd_scan_backward_chunked_kernel::launch_unchecked::<E, R>(
                x.client(),
                count,
                CubeDim::new_1d(CHUNK_UNITS as u32),
                dy.arg(),
                x.arg(),
                b.arg(),
                c.arg(),
                a.arg(),
                a_head.arg(),
                g.arg(),
                w.arg(),
                skip.unwrap_or(&placeholder).arg(),
                ckpt.arg(),
                grads.dx.arg(),
                grads.db.arg(),
                grads.dc.arg(),
                grads.d_decay.arg(),
                grads.dg.arg(),
                grads.dw.arg(),
                grads.d_init.as_ref().unwrap_or(&placeholder_out).arg(),
                head_part.arg(),
                skip_part.arg(),
                s.seq,
                s.heads,
                segs,
                s.head_dim,
                s.state,
                init.is_some(),
                rate,
                skip.is_some(),
            );
        }
        return finish_grads(grads, &head_part, &skip_part, &s, rate, skip.is_some());
    }

    let BackwardLayout {
        use_planes,
        split_bits,
        vw,
        seg_len,
        segs,
        group,
        groups,
        max_planes,
    } = backward_layout::<R, E>(x.client(), &s);
    let fresh;
    let ckpt = match saved {
        Some(saved) if saved.states.len() == bh * segs * s.state * s.head_dim => &saved.states,
        _ => {
            fresh = Tensor::<R, f32>::empty(Shape::new(vec![bh * segs * s.state * s.head_dim]), device);
            &fresh
        }
    };
    let has_ckpt = saved.is_some_and(|saved| std::ptr::eq(&saved.states, ckpt));
    let cubes = bh.div_ceil(groups);
    let x_cubes = cubes.min(u16::MAX as usize);
    let count = CubeCount::Static(x_cubes as u32, cubes.div_ceil(x_cubes) as u32, 1);
    crate::backend::count_launch();
    unsafe {
        ssd_scan_backward_kernel::launch_unchecked::<E, R>(
            x.client(),
            count,
            CubeDim::new_1d((groups * group) as u32),
            vw,
            dy.arg(),
            x.arg(),
            b.arg(),
            c.arg(),
            a.arg(),
            a_head.arg(),
            g.arg(),
            w.arg(),
            init.unwrap_or(&placeholder).arg(),
            skip.unwrap_or(&placeholder).arg(),
            ckpt.arg(),
            grads.dx.arg(),
            grads.db.arg(),
            grads.dc.arg(),
            grads.d_decay.arg(),
            grads.dg.arg(),
            grads.dw.arg(),
            grads.d_init.as_ref().unwrap_or(&placeholder_out).arg(),
            head_part.arg(),
            skip_part.arg(),
            s.seq,
            s.heads,
            s.head_dim,
            bh,
            segs,
            group,
            groups,
            max_planes,
            s.state,
            vw,
            split_bits,
            seg_len,
            init.is_some(),
            use_planes,
            rate,
            skip.is_some(),
            has_ckpt,
        );
    }
    finish_grads(grads, &head_part, &skip_part, &s, rate, skip.is_some())
}

/// Fold the `[B * H]` per-(batch, head) partials of the per-head parameters over
/// the batch.
fn finish_grads<R: Runtime, E: FloatElem>(
    mut grads: ScanGrads<R, E>,
    head_part: &Tensor<R, f32>,
    skip_part: &Tensor<R, f32>,
    s: &ScanDims,
    rate: bool,
    has_skip: bool,
) -> Result<ScanGrads<R, E>> {
    let fold = |part: &Tensor<R, f32>| -> Result<Tensor<R, E>> {
        let summed = crate::tensor::ops::reduce::sum_dim(
            &part.reshape(Shape::new(vec![s.batch, s.heads]))?,
            0,
        )?;
        crate::tensor::ops::elemwise::cast::<R, f32, E>(&summed).reshape(Shape::new(vec![s.heads]))
    };
    if rate {
        grads.d_a_head = Some(fold(head_part)?);
    }
    if has_skip {
        grads.d_skip = Some(fold(skip_part)?);
    }
    Ok(grads)
}
