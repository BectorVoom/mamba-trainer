//! On-device planner batch assembly (K1): gather inputs and build labels.
//!
//! After one upload per split (the device-resident dataset of K1),
//! a training step moves no data from the host: [`planner_gather_inputs`]
//! collects the `[B]` turns' features and [`planner_labels`] builds every id,
//! keep weight and label tensor the loss needs. Two launches per step, no
//! reads, no atomics. The composed batch constructor stays as the correctness
//! oracle.
//!
//! # Launch shape
//!
//! Both kernels launch through [`crate::backend::launch_1d_spans`] like the
//! entity kernels do: one lane per unit on a GPU, one contiguous span of lanes
//! per worker thread on the CPU runtime. The inputs kernel reads and writes
//! [`Vector`]s as wide as every section allows; the labels kernel is scalar —
//! its outputs interleave too finely for one width to suit them all.

use cubecl::prelude::*;

use crate::backend::{FloatElem, launch_1d_spans, line_size_for};
use crate::error::{Error, Result};
use crate::tensor::base::Tensor;
use crate::tensor::ops::index::IdTensor;
use crate::tensor::shape::Shape;

/// Id sentinel: kernels test for it and never index with it (K invariant 6).
pub const IGNORE: u32 = u32::MAX;

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// The widest vector width the device likes for `E` that divides every one of
/// `parts`, so no vector straddles a row or a feature section.
fn line_dividing<R: Runtime, E: FloatElem>(client: &ComputeClient<R>, parts: &[usize]) -> usize {
    line_size_for::<R, E>(client, parts.iter().fold(0, |g, &p| gcd(g, p)))
}

#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn planner_gather_inputs_kernel<F: Float + CubeElement, N: Size>(
    floats: &Array<Vector<F, N>>,
    turn_ids: &Array<u32>,
    tiles: &mut Array<Vector<F, N>>,
    glob: &mut Array<Vector<F, N>>,
    units: &mut Array<Vector<F, N>>,
    f_vec: usize,
    tn_vec: usize,
    g_vec: usize,
    un_vec: usize,
    lanes: usize,
    span: usize,
) {
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    for pos in start..end {
        let b = pos / f_vec;
        let cv = pos % f_vec;
        let v = floats[turn_ids[b] as usize * f_vec + cv];
        if cv < tn_vec {
            tiles[b * tn_vec + cv] = v;
        } else if cv < tn_vec + g_vec {
            glob[b * g_vec + (cv - tn_vec)] = v;
        } else {
            units[b * un_vec + (cv - tn_vec - g_vec)] = v;
        }
    }
}

/// Gather one batch's features from a device-resident split: `floats` is
/// `[turns, F]` with `F = tiles | glob | units` columns, `turn_ids` is `[B]`.
/// Returns `[B, tiles]`, `[B, glob]`, `[B, units]` in one launch. No adjoint:
/// data are constants.
#[allow(clippy::too_many_arguments)]
pub fn planner_gather_inputs<R: Runtime, E: FloatElem>(
    floats: &Tensor<R, E>,
    turn_ids: &IdTensor<R>,
    tile_cols: usize,
    glob_cols: usize,
    unit_cols: usize,
) -> Result<(Tensor<R, E>, Tensor<R, E>, Tensor<R, E>)> {
    let b = turn_ids.len();
    let f = tile_cols + glob_cols + unit_cols;
    if floats.rank() != 2 || floats.shape().dim(1) != f {
        return Err(Error::shape(format!(
            "planner_gather_inputs needs floats [turns, {f}], got {}",
            floats.shape()
        )));
    }
    let device = floats.device();
    let tiles = Tensor::empty(Shape::new(vec![b, tile_cols]), device);
    let glob = Tensor::empty(Shape::new(vec![b, glob_cols]), device);
    let units = Tensor::empty(Shape::new(vec![b, unit_cols]), device);
    if b == 0 || f == 0 {
        return Ok((tiles, glob, units));
    }
    let line = line_dividing::<R, E>(floats.client(), &[f, tile_cols, glob_cols, unit_cols]);
    let (f_vec, tn_vec, g_vec, un_vec) = (f / line, tile_cols / line, glob_cols / line, unit_cols / line);
    let lanes = b * f_vec;
    let (cube_count, cube_dim, span) = launch_1d_spans(floats.client(), lanes, f_vec);
    unsafe {
        planner_gather_inputs_kernel::launch_unchecked::<E, R>(
            floats.client(),
            cube_count,
            cube_dim,
            line,
            floats.arg(),
            turn_ids.arg(),
            tiles.arg(),
            glob.arg(),
            units.arg(),
            f_vec,
            tn_vec,
            g_vec,
            un_vec,
            lanes,
            span,
        );
    }
    Ok((tiles, glob, units))
}

/// Every label tensor [`planner_labels`] produces.
pub struct PlannerLabels<R: Runtime, E: FloatElem> {
    /// `[B*U]` unit tile ids (`IGNORE` for padding).
    pub unit_ids: IdTensor<R>,
    /// `[B*Q]` target ids (0 where ignored).
    pub target_ids: IdTensor<R>,
    /// `[B*Q]` gather ids for the aux target token (`IGNORE` where ignored).
    pub target_gather: IdTensor<R>,
    /// `[B*Q]` step weight where kept, else 0.
    pub target_w: Tensor<R, E>,
    /// `[B*Q]` op ids (0 where ignored).
    pub op_ids: IdTensor<R>,
    /// `[B*Q]` 1 where the op loss applies.
    pub op_w: Tensor<R, E>,
    /// `[B*Q]` crop ids (0 where ignored).
    pub crop_ids: IdTensor<R>,
    /// `[B*Q]` 1 where the crop loss applies.
    pub crop_w: Tensor<R, E>,
    /// `[B*Q, n_ops]` multi-hot op targets.
    pub opset: Tensor<R, E>,
    /// `[B*Q]` log1p(eta) on step 0 with eta set, else 0.
    pub eta_log: Tensor<R, E>,
    /// `[B*Q]` 1 on step 0 with eta set, else 0.
    pub eta_w: Tensor<R, E>,
}

#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn planner_labels_kernel<F: Float + CubeElement>(
    labels: &Array<u32>,
    turn_ids: &Array<u32>,
    step_w: &Array<F>,
    unit_ids: &mut Array<u32>,
    target_ids: &mut Array<u32>,
    target_gather: &mut Array<u32>,
    target_w: &mut Array<F>,
    op_ids: &mut Array<u32>,
    op_w: &mut Array<F>,
    crop_ids: &mut Array<u32>,
    crop_w: &mut Array<F>,
    opset: &mut Array<F>,
    eta_log: &mut Array<F>,
    eta_w: &mut Array<F>,
    n: usize,
    u: usize,
    k: usize,
    q: usize,
    row: usize,
    n_ops: usize,
    q_lanes: usize,
    lanes: usize,
    span: usize,
    ignore: u32,
) {
    let zero = F::new(0.0_f32);
    let one = F::new(1.0_f32);
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    for pos in start..end {
        if pos < q_lanes {
            // One (b, q) thread: every Q output for its row.
            let b = pos / q;
            let qq = pos % q;
            let j = qq % k;
            let uu = qq / k;
            let base = turn_ids[b] as usize * row;
            let t = labels[base + u + qq];
            if t == ignore {
                target_ids[pos] = 0u32;
                target_gather[pos] = ignore;
                target_w[pos] = zero;
            } else {
                target_ids[pos] = t;
                target_gather[pos] = t;
                target_w[pos] = step_w[j];
            }
            let o = labels[base + u + q + qq];
            let tgt_is_tile = t != ignore && (t as usize) < n;
            if tgt_is_tile && o != ignore {
                op_ids[pos] = o;
                op_w[pos] = one;
            } else {
                op_ids[pos] = 0u32;
                op_w[pos] = zero;
            }
            let c = labels[base + u + 2 * q + qq];
            if c == ignore {
                crop_ids[pos] = 0u32;
                crop_w[pos] = zero;
            } else {
                crop_ids[pos] = c;
                crop_w[pos] = one;
            }
            let mask = labels[base + 2 * u + 3 * q + qq];
            let mut bit = 1u32;
            for i in 0..n_ops {
                if (mask & bit) == 0u32 {
                    opset[pos * n_ops + i] = zero;
                } else {
                    opset[pos * n_ops + i] = one;
                }
                bit *= 2u32;
            }
            let e = labels[base + u + 3 * q + uu];
            if j == 0 && e != ignore {
                eta_log[pos] = (one + F::cast_from(e)).ln();
                eta_w[pos] = one;
            } else {
                eta_log[pos] = zero;
                eta_w[pos] = zero;
            }
        } else {
            // One (b, u) thread: its unit id.
            let i = pos - q_lanes;
            let b = i / u;
            let uu = i % u;
            unit_ids[i] = labels[turn_ids[b] as usize * row + uu];
        }
    }
}

/// Build every id, keep weight and label tensor for `[B]` turns in one launch:
/// `labels` is `[turns, L]` (`upos | tgt | op | crop | eta | opset-mask`),
/// `turn_ids` is `[B]`, `step_w` is `[K]`. Denominators are not computed here:
/// K3 sums the weights on the device as part of the loss.
pub fn planner_labels<R: Runtime, E: FloatElem>(
    labels: &IdTensor<R>,
    turn_ids: &IdTensor<R>,
    step_w: &Tensor<R, E>,
    n: usize,
    u: usize,
    k: usize,
    n_ops: usize,
) -> Result<PlannerLabels<R, E>> {
    let b = turn_ids.len();
    let q = u * k;
    let row = 2 * u + 4 * q;
    if labels.shape().rank() != 2 || labels.shape().dim(1) != row {
        return Err(Error::shape(format!(
            "planner_labels needs labels [turns, {row}], got {}",
            labels.shape()
        )));
    }
    if step_w.len() != k {
        return Err(Error::shape(format!(
            "planner_labels needs step_w [{k}], got {}",
            step_w.shape()
        )));
    }
    let device = labels.device();
    let float_here = |len: usize| Tensor::<R, E>::empty(Shape::new(vec![len]), device);
    let out = PlannerLabels {
        unit_ids: IdTensor::empty(vec![b * u], device),
        target_ids: IdTensor::empty(vec![b * q], device),
        target_gather: IdTensor::empty(vec![b * q], device),
        target_w: float_here(b * q),
        op_ids: IdTensor::empty(vec![b * q], device),
        op_w: float_here(b * q),
        crop_ids: IdTensor::empty(vec![b * q], device),
        crop_w: float_here(b * q),
        opset: Tensor::<R, E>::empty(Shape::new(vec![b * q, n_ops]), device),
        eta_log: float_here(b * q),
        eta_w: float_here(b * q),
    };
    let lanes = b * q + b * u;
    if lanes == 0 {
        return Ok(out);
    }
    // Scalar kernel: the outputs interleave too finely for one vector width.
    let (cube_count, cube_dim, span) = launch_1d_spans(labels.client(), lanes, 1);
    unsafe {
        planner_labels_kernel::launch_unchecked::<E, R>(
            labels.client(),
            cube_count,
            cube_dim,
            labels.arg(),
            turn_ids.arg(),
            step_w.arg(),
            out.unit_ids.arg(),
            out.target_ids.arg(),
            out.target_gather.arg(),
            out.target_w.arg(),
            out.op_ids.arg(),
            out.op_w.arg(),
            out.crop_ids.arg(),
            out.crop_w.arg(),
            out.opset.arg(),
            out.eta_log.arg(),
            out.eta_w.arg(),
            n,
            u,
            k,
            q,
            row,
            n_ops,
            b * q,
            lanes,
            span,
            IGNORE,
        );
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// K2. Token gathers and the query build
// ---------------------------------------------------------------------------

#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn planner_queries_kernel<F: Float + CubeElement, N: Size>(
    um: &Array<Vector<F, N>>,
    t: &Array<Vector<F, N>>,
    step: &Array<Vector<F, N>>,
    unit_ids: &Array<u32>,
    out: &mut Array<Vector<F, N>>,
    q: usize,
    dvec: usize,
    u: usize,
    k: usize,
    n: usize,
    lanes: usize,
    span: usize,
    ignore: u32,
) {
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    for pos in start..end {
        let dv = pos % dvec;
        let qq = (pos / dvec) % q;
        let b = pos / (dvec * q);
        let uu = qq / k;
        let j = qq % k;
        let id = unit_ids[b * u + uu];
        let mut v = um[(b * u + uu) * dvec + dv] + step[j * dvec + dv];
        if id != ignore {
            v += t[(b * n + id as usize) * dvec + dv];
        }
        out[pos] = v;
    }
}

/// Build the query tokens on the device: `q[b, u*K+k, :] = um[b,u,:] +
/// (unit_ids[b*U+u] == IGNORE ? 0 : t[b, id, :]) + step[k, :]` — one launch.
/// `um` is `[B,U,d]`, `t` is `[B,N,d]`, `step` is `[K,d]`, output `[B,U*K,d]`.
pub fn planner_queries_fwd<R: Runtime, E: FloatElem>(
    um: &Tensor<R, E>,
    t: &Tensor<R, E>,
    step: &Tensor<R, E>,
    unit_ids: &IdTensor<R>,
) -> Result<Tensor<R, E>> {
    let (b, u, d) = (um.shape().dim(0), um.shape().dim(1), um.shape().dim(2));
    let (k, n) = (step.shape().dim(0), t.shape().dim(1));
    if um.rank() != 3 || t.dims() != &[b, n, d] || step.dims() != &[k, d] {
        return Err(Error::shape(format!(
            "planner_queries needs um [B,U,d], t [B,N,d], step [K,d]; got {}, {} and {}",
            um.shape(),
            t.shape(),
            step.shape()
        )));
    }
    if unit_ids.len() != b * u {
        return Err(Error::shape(format!(
            "planner_queries needs unit_ids [B*U = {}], got {}",
            b * u,
            unit_ids.shape()
        )));
    }
    let q = u * k;
    let out = Tensor::empty(Shape::new(vec![b, q, d]), um.device());
    if out.len() == 0 {
        return Ok(out);
    }
    let line = line_dividing::<R, E>(um.client(), &[d]);
    let dvec = d / line;
    let lanes = b * q * dvec;
    let (cube_count, cube_dim, span) = launch_1d_spans(um.client(), lanes, dvec);
    unsafe {
        planner_queries_kernel::launch_unchecked::<E, R>(
            um.client(),
            cube_count,
            cube_dim,
            line,
            um.arg(),
            t.arg(),
            step.arg(),
            unit_ids.arg(),
            out.arg(),
            q,
            dvec,
            u,
            k,
            n,
            lanes,
            span,
            IGNORE,
        );
    }
    Ok(out)
}

#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn planner_queries_backward_kernel<F: Float + CubeElement, N: Size>(
    grad: &Array<Vector<F, N>>,
    unit_ids: &Array<u32>,
    d_um: &mut Array<Vector<F, N>>,
    d_t: &mut Array<Vector<F, N>>,
    d_step: &mut Array<Vector<F, N>>,
    q: usize,
    dvec: usize,
    u: usize,
    n: usize,
    k: usize,
    b: usize,
    lanes1: usize,
    lanes2: usize,
    lanes: usize,
    span: usize,
    _ignore: u32,
) {
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    for pos in start..end {
        if pos < lanes1 {
            // d_um[b,u,:] = sum over k of g[b, u*K+k, :].
            let dv = pos % dvec;
            let uu = (pos / dvec) % u;
            let bb = pos / (dvec * u);
            let mut acc = Vector::<F, N>::new(F::new(0.0_f32));
            for j in 0..k {
                acc += grad[(bb * q + uu * k + j) * dvec + dv];
            }
            d_um[pos] = acc;
        } else if pos < lanes1 + lanes2 {
            // d_t[b,n,:] = sum over the units on tile n (and k) of g.
            let p = pos - lanes1;
            let dv = p % dvec;
            let nn = (p / dvec) % n;
            let bb = p / (dvec * n);
            let mut acc = Vector::<F, N>::new(F::new(0.0_f32));
            for uu in 0..u {
                if unit_ids[bb * u + uu] as usize == nn {
                    for j in 0..k {
                        acc += grad[(bb * q + uu * k + j) * dvec + dv];
                    }
                }
            }
            d_t[p] = acc;
        } else {
            // d_step[k,:] = sum over (b, u) of g.
            let p = pos - lanes1 - lanes2;
            let dv = p % dvec;
            let j = p / dvec;
            let mut acc = Vector::<F, N>::new(F::new(0.0_f32));
            for bb in 0..b {
                for uu in 0..u {
                    acc += grad[(bb * q + uu * k + j) * dvec + dv];
                }
            }
            d_step[p] = acc;
        }
    }
}

/// Adjoint of [`planner_queries_fwd`]: three gathers in one launch (one region
/// per output), atomic-free. Returns `(d_um [B,U,d], d_t [B,N,d], d_step [K,d])`.
pub fn planner_queries_backward<R: Runtime, E: FloatElem>(
    grad: &Tensor<R, E>,
    unit_ids: &IdTensor<R>,
    u: usize,
    k: usize,
    n: usize,
) -> Result<(Tensor<R, E>, Tensor<R, E>, Tensor<R, E>)> {
    let (b, q, d) = (grad.shape().dim(0), grad.shape().dim(1), grad.shape().dim(2));
    if grad.rank() != 3 || q != u * k {
        return Err(Error::shape(format!(
            "planner_queries_backward needs grad [B, U*K, d], got {}",
            grad.shape()
        )));
    }
    if unit_ids.len() != b * u {
        return Err(Error::shape(format!(
            "planner_queries_backward needs unit_ids [B*U = {}], got {}",
            b * u,
            unit_ids.shape()
        )));
    }
    let device = grad.device();
    let d_um = Tensor::empty(Shape::new(vec![b, u, d]), device);
    let d_t = Tensor::empty(Shape::new(vec![b, n, d]), device);
    let d_step = Tensor::empty(Shape::new(vec![k, d]), device);
    let line = line_dividing::<R, E>(grad.client(), &[d]);
    let dvec = d / line;
    let (lanes1, lanes2, lanes3) = (b * u * dvec, b * n * dvec, k * dvec);
    let lanes = lanes1 + lanes2 + lanes3;
    if lanes == 0 {
        return Ok((d_um, d_t, d_step));
    }
    let (cube_count, cube_dim, span) = launch_1d_spans(grad.client(), lanes, dvec);
    unsafe {
        planner_queries_backward_kernel::launch_unchecked::<E, R>(
            grad.client(),
            cube_count,
            cube_dim,
            line,
            grad.arg(),
            unit_ids.arg(),
            d_um.arg(),
            d_t.arg(),
            d_step.arg(),
            q,
            dvec,
            u,
            n,
            k,
            b,
            lanes1,
            lanes2,
            lanes,
            span,
            IGNORE,
        );
    }
    Ok((d_um, d_t, d_step))
}

#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn gather_tokens_kernel<F: Float + CubeElement, N: Size>(
    src: &Array<Vector<F, N>>,
    ids: &Array<u32>,
    out: &mut Array<Vector<F, N>>,
    r: usize,
    dvec: usize,
    s: usize,
    lanes: usize,
    span: usize,
    ignore: u32,
) {
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    for pos in start..end {
        let dv = pos % dvec;
        let rr = (pos / dvec) % r;
        let b = pos / (dvec * r);
        let id = ids[b * r + rr];
        if id == ignore {
            out[pos] = Vector::<F, N>::new(F::new(0.0_f32));
        } else {
            out[pos] = src[(b * s + id as usize) * dvec + dv];
        }
    }
}

/// Index-gather rows: `out[b, r, :] = ids[b*R+r] == IGNORE ? 0 :
/// src[b, ids, :]`. `src` is `[B,S,d]`, `ids` is `[B*R]`, output `[B,R,d]` —
/// one launch. Removes the `[B,Q,N+1]` one-hot entirely.
pub fn gather_tokens<R: Runtime, E: FloatElem>(
    src: &Tensor<R, E>,
    ids: &IdTensor<R>,
    r: usize,
) -> Result<Tensor<R, E>> {
    let (b, s, d) = (src.shape().dim(0), src.shape().dim(1), src.shape().dim(2));
    if src.rank() != 3 || ids.len() != b * r {
        return Err(Error::shape(format!(
            "gather_tokens needs src [B,S,d] and ids [B*{r}], got {} and {}",
            src.shape(),
            ids.shape()
        )));
    }
    let out = Tensor::empty(Shape::new(vec![b, r, d]), src.device());
    if out.len() == 0 {
        return Ok(out);
    }
    let line = line_dividing::<R, E>(src.client(), &[d]);
    let dvec = d / line;
    let lanes = b * r * dvec;
    let (cube_count, cube_dim, span) = launch_1d_spans(src.client(), lanes, dvec);
    unsafe {
        gather_tokens_kernel::launch_unchecked::<E, R>(
            src.client(),
            cube_count,
            cube_dim,
            line,
            src.arg(),
            ids.arg(),
            out.arg(),
            r,
            dvec,
            s,
            lanes,
            span,
            IGNORE,
        );
    }
    Ok(out)
}

#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn gather_tokens_backward_kernel<F: Float + CubeElement, N: Size>(
    grad: &Array<Vector<F, N>>,
    ids: &Array<u32>,
    d_src: &mut Array<Vector<F, N>>,
    r: usize,
    dvec: usize,
    s: usize,
    lanes: usize,
    span: usize,
) {
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    for pos in start..end {
        let dv = pos % dvec;
        let ss = (pos / dvec) % s;
        let b = pos / (dvec * s);
        // IGNORE names no row, so it never matches: no explicit test needed.
        let mut acc = Vector::<F, N>::new(F::new(0.0_f32));
        for rr in 0..r {
            if ids[b * r + rr] as usize == ss {
                acc += grad[(b * r + rr) * dvec + dv];
            }
        }
        d_src[pos] = acc;
    }
}

/// Adjoint of [`gather_tokens`]: one thread per `(b, s, dvec)` looping over
/// the `r` rows that may feed it — a pure gather, no atomics.
pub fn gather_tokens_backward<R: Runtime, E: FloatElem>(
    grad: &Tensor<R, E>,
    ids: &IdTensor<R>,
    s: usize,
) -> Result<Tensor<R, E>> {
    let (b, r, d) = (grad.shape().dim(0), grad.shape().dim(1), grad.shape().dim(2));
    if grad.rank() != 3 || ids.len() != b * r {
        return Err(Error::shape(format!(
            "gather_tokens_backward needs grad [B,R,d] and ids [B*R], got {} and {}",
            grad.shape(),
            ids.shape()
        )));
    }
    let d_src = Tensor::empty(Shape::new(vec![b, s, d]), grad.device());
    if d_src.len() == 0 {
        return Ok(d_src);
    }
    let line = line_dividing::<R, E>(grad.client(), &[d]);
    let dvec = d / line;
    let lanes = b * s * dvec;
    let (cube_count, cube_dim, span) = launch_1d_spans(grad.client(), lanes, dvec);
    unsafe {
        gather_tokens_backward_kernel::launch_unchecked::<E, R>(
            grad.client(),
            cube_count,
            cube_dim,
            line,
            grad.arg(),
            ids.arg(),
            d_src.arg(),
            r,
            dvec,
            s,
            lanes,
            span,
        );
    }
    Ok(d_src)
}

// ---------------------------------------------------------------------------
// K3. The fused loss
// ---------------------------------------------------------------------------

/// Label/weight tables the fused loss reads.
pub struct LossTables<'a, R: Runtime, E: FloatElem> {
    /// `[B*Q]` target ids (0 where ignored).
    pub target_ids: &'a IdTensor<R>,
    /// `[B*Q]` step weight where kept, else 0.
    pub target_w: &'a Tensor<R, E>,
    /// `[B*Q]` op ids (0 where ignored).
    pub op_ids: &'a IdTensor<R>,
    /// `[B*Q]` 1 where the op loss applies.
    pub op_w: &'a Tensor<R, E>,
    /// `[B*Q]` crop ids (0 where ignored).
    pub crop_ids: &'a IdTensor<R>,
    /// `[B*Q]` 1 where the crop loss applies.
    pub crop_w: &'a Tensor<R, E>,
    /// `[B*Q, n_ops]` multi-hot op targets.
    pub opset: &'a Tensor<R, E>,
    /// `[B*Q]` log1p(eta) on step 0 with eta set, else 0.
    pub eta_log: &'a Tensor<R, E>,
    /// `[B*Q]` 1 on step 0 with eta set, else 0.
    pub eta_w: &'a Tensor<R, E>,
    /// Op class count (also the opset width).
    pub n_ops: usize,
    /// Crop class count.
    pub n_crops: usize,
    /// Static loss scale (§2.5).
    pub loss_scale: f32,
}

#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn planner_loss_rows_kernel<F: Float + CubeElement>(
    logits: &Array<F>,
    aux: &Array<F>,
    target_ids: &Array<u32>,
    target_w: &Array<F>,
    op_ids: &Array<u32>,
    op_w: &Array<F>,
    crop_ids: &Array<u32>,
    crop_w: &Array<F>,
    opset: &Array<F>,
    eta_log: &Array<F>,
    eta_w: &Array<F>,
    rows_out: &mut Array<F>,
    lse_out: &mut Array<F>,
    n1: usize,
    aw: usize,
    no: usize,
    nc: usize,
    rows: usize,
    span: usize,
) {
    let mut end = (ABSOLUTE_POS + 1) * span;
    if end > rows {
        end = rows;
    }
    for start in ABSOLUTE_POS * span..end {
    // One thread per row, in F like the crate's own cross-entropy kernel
    // (E is f32 wherever the planner trains).
    // Target cross-entropy, max-shifted.
    let mut mx = logits[start * n1];
    for c in 1..n1 {
        mx = mx.max(logits[start * n1 + c]);
    }
    let mut se = F::new(0.0_f32);
    for c in 0..n1 {
        se += F::exp(logits[start * n1 + c] - mx);
    }
    let lse_t = mx + F::ln(se);
    let ce_t = lse_t - logits[start * n1 + target_ids[start] as usize];
    // Op cross-entropy over aux[0..no].
    let mut mo = aux[start * aw];
    for c in 1..no {
        mo = mo.max(aux[start * aw + c]);
    }
    let mut so = F::new(0.0_f32);
    for c in 0..no {
        so += F::exp(aux[start * aw + c] - mo);
    }
    let lse_o = mo + F::ln(so);
    let ce_o = lse_o - aux[start * aw + op_ids[start] as usize];
    // Crop cross-entropy over aux[2*no..2*no+nc].
    let off_c = 2 * no;
    let mut mc = aux[start * aw + off_c];
    for c in 1..nc {
        mc = mc.max(aux[start * aw + off_c + c]);
    }
    let mut sc = F::new(0.0_f32);
    for c in 0..nc {
        sc += F::exp(aux[start * aw + off_c + c] - mc);
    }
    let lse_c = mc + F::ln(sc);
    let ce_c = lse_c - aux[start * aw + off_c + crop_ids[start] as usize];
    // Opset BCE over aux[no..2*no]: softplus(x) - x*y, stable form.
    let mut bce = F::new(0.0_f32);
    for i in 0..no {
        let x = aux[start * aw + no + i];
        let y = opset[start * no + i];
        bce += x.max(F::new(0.0_f32)) + F::ln(F::new(1.0_f32) + F::exp(x.abs() * F::new(-1.0_f32)))
            - x * y;
    }
    // Eta squared error on the last aux column.
    let d = aux[start * aw + aw - 1] - eta_log[start];
    let sq = d * d;
    let tw = target_w[start];
    let ow = op_w[start];
    let cw = crop_w[start];
    let ew = eta_w[start];
    rows_out[start * 10] = tw * ce_t;
    rows_out[start * 10 + 1] = ow * ce_o;
    rows_out[start * 10 + 2] = cw * ce_c;
    rows_out[start * 10 + 3] = ow * bce;
    rows_out[start * 10 + 4] = ew * sq;
    rows_out[start * 10 + 5] = tw;
    rows_out[start * 10 + 6] = ow;
    rows_out[start * 10 + 7] = cw;
    rows_out[start * 10 + 8] = ow;
    rows_out[start * 10 + 9] = ew;
    lse_out[start * 3] = lse_t;
    lse_out[start * 3 + 1] = lse_o;
    lse_out[start * 3 + 2] = lse_c;
    }
}

/// Forward pass 1 of the fused loss: per-row weighted terms plus the saved
/// log-sum-exps. `logits` is `[rows, n1]`, `aux` is `[rows, aw]`. Returns
/// `rows_out [rows, 10]` and `lse_out [rows, 3]`, one launch.
pub fn planner_loss_rows<R: Runtime, E: FloatElem>(
    logits: &Tensor<R, E>,
    aux: &Tensor<R, E>,
    tables: &LossTables<R, E>,
) -> Result<(Tensor<R, E>, Tensor<R, E>)> {
    let rows = logits.shape().dim(0);
    let (n1, aw) = (logits.shape().dim(1), aux.shape().dim(1));
    let (no, nc) = (tables.n_ops, tables.n_crops);
    if logits.rank() != 2 || aux.rank() != 2 || aux.shape().dim(0) != rows {
        return Err(Error::shape(format!(
            "planner_loss_rows needs logits [rows, n1] and aux [rows, aw], got {} and {}",
            logits.shape(),
            aux.shape()
        )));
    }
    if aw != 2 * no + nc + 1 {
        return Err(Error::shape(format!(
            "planner_loss_rows needs aux width 2*{no}+{nc}+1, got {aw}"
        )));
    }
    let device = logits.device();
    let rows_out = Tensor::empty(Shape::new(vec![rows, 10]), device);
    let lse_out = Tensor::empty(Shape::new(vec![rows, 3]), device);
    if rows == 0 {
        return Ok((rows_out, lse_out));
    }
    // Serial per-row work: one lane per row is the whole kernel.
    let (cube_count, cube_dim, span) = launch_1d_spans(logits.client(), rows, n1 + aw);
    unsafe {
        planner_loss_rows_kernel::launch_unchecked::<E, R>(
            logits.client(),
            cube_count,
            cube_dim,
            logits.arg(),
            aux.arg(),
            tables.target_ids.arg(),
            tables.target_w.arg(),
            tables.op_ids.arg(),
            tables.op_w.arg(),
            tables.crop_ids.arg(),
            tables.crop_w.arg(),
            tables.opset.arg(),
            tables.eta_log.arg(),
            tables.eta_w.arg(),
            rows_out.arg(),
            lse_out.arg(),
            n1,
            aw,
            no,
            nc,
            rows,
            span,
        );
    }
    Ok((rows_out, lse_out))
}

#[cube(launch_unchecked)]
fn planner_loss_reduce_kernel<F: Float + CubeElement>(
    rows: &Array<F>,
    loss: &mut Array<F>,
    report: &mut Array<F>,
    bq: usize,
    inv_no: F,
    scale: F,
) {
    if ABSOLUTE_POS == 0 {
    // One thread reduces all rows serially; B*Q is a few thousand.
    let mut s0 = F::new(0.0_f32);
    let mut s1 = F::new(0.0_f32);
    let mut s2 = F::new(0.0_f32);
    let mut s3 = F::new(0.0_f32);
    let mut s4 = F::new(0.0_f32);
    let mut s5 = F::new(0.0_f32);
    let mut s6 = F::new(0.0_f32);
    let mut s7 = F::new(0.0_f32);
    let mut s8 = F::new(0.0_f32);
    let mut s9 = F::new(0.0_f32);
    for r in 0..bq {
        s0 += rows[r * 10];
        s1 += rows[r * 10 + 1];
        s2 += rows[r * 10 + 2];
        s3 += rows[r * 10 + 3];
        s4 += rows[r * 10 + 4];
        s5 += rows[r * 10 + 5];
        s6 += rows[r * 10 + 6];
        s7 += rows[r * 10 + 7];
        s8 += rows[r * 10 + 8];
        s9 += rows[r * 10 + 9];
    }
    let one = F::new(1.0_f32);
    let den0 = s5.max(one);
    let den1 = s6.max(one);
    let den2 = s7.max(one);
    let den3 = s8.max(one);
    let den4 = s9.max(one);
    let u0 = s0 / den0;
    let u1 = s1 / den1;
    let u2 = s3 * inv_no / den3;
    let u3 = s2 / den2;
    let u4 = s4 / den4;
    let w03 = F::new(0.3_f32);
    let w01 = F::new(0.1_f32);
    let total = scale * (u0 + u1 + w03 * u2 + w03 * u3 + w01 * u4);
    loss[0] = total;
    report[0] = total / scale;
    report[1] = u0;
    report[2] = u1;
    report[3] = u2;
    report[4] = u3;
    report[5] = u4;
    report[6] = den0;
    report[7] = den1;
    report[8] = den2;
    report[9] = den3;
    report[10] = den4;
    }
}

/// Forward pass 2 of the fused loss: reduce `rows_out [B*Q, 10]` over rows
/// into the scalar loss `[1]` and the `[11]` report (total, 5 unscaled
/// components, 5 denominators) — one launch. The report is off the tape, for
/// logging; the scalar carries the tape through [`planner_loss_backward`].
pub fn planner_loss_reduce<R: Runtime, E: FloatElem>(
    rows_out: &Tensor<R, E>,
    n_ops: usize,
    loss_scale: f32,
) -> Result<(Tensor<R, E>, Tensor<R, E>)> {
    let bq = rows_out.shape().dim(0);
    if rows_out.rank() != 2 || rows_out.shape().dim(1) != 10 {
        return Err(Error::shape(format!(
            "planner_loss_reduce needs rows_out [B*Q, 10], got {}",
            rows_out.shape()
        )));
    }
    let device = rows_out.device();
    let loss = Tensor::empty(Shape::new(vec![1]), device);
    let report = Tensor::empty(Shape::new(vec![11]), device);
    let (cube_count, cube_dim, _) = launch_1d_spans(rows_out.client(), 1, 1);
    unsafe {
        planner_loss_reduce_kernel::launch_unchecked::<E, R>(
            rows_out.client(),
            cube_count,
            cube_dim,
            rows_out.arg(),
            loss.arg(),
            report.arg(),
            bq,
            E::from_scalar(1.0 / n_ops as f32),
            E::from_scalar(loss_scale),
        );
    }
    Ok((loss, report))
}

#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn planner_loss_backward_kernel<F: Float + CubeElement>(
    upstream: &Array<F>,
    logits: &Array<F>,
    aux: &Array<F>,
    target_ids: &Array<u32>,
    target_w: &Array<F>,
    op_ids: &Array<u32>,
    op_w: &Array<F>,
    crop_ids: &Array<u32>,
    crop_w: &Array<F>,
    opset: &Array<F>,
    eta_log: &Array<F>,
    eta_w: &Array<F>,
    lse: &Array<F>,
    report: &Array<F>,
    d_logits: &mut Array<F>,
    d_aux: &mut Array<F>,
    n1: usize,
    aw: usize,
    no: usize,
    nc: usize,
    bq: usize,
    log_lanes: usize,
    lanes: usize,
    span: usize,
    scale: F,
    inv_no: F,
) {
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    let gu = upstream[0] * scale;
    let den0 = report[6];
    let den1 = report[7];
    let den2 = report[8];
    let den3 = report[9];
    let den4 = report[10];
    let zero = F::new(0.0_f32);
    let one = F::new(1.0_f32);
    let w03 = F::new(0.3_f32);
    let w01 = F::new(0.1_f32);
    let two = F::new(2.0_f32);
    for pos in start..end {
        if pos < log_lanes {
            // Target-logit columns.
            let r = pos / n1;
            let c = pos % n1;
            let w = target_w[r] / den0;
            let p = F::exp(logits[pos] - lse[r * 3]);
            let y = if target_ids[r] as usize == c { one } else { zero };
            d_logits[pos] = gu * w * (p - y);
        } else {
            // Aux columns: op | opset | crop | eta.
            let p = pos - log_lanes;
            let r = p / aw;
            let c = p % aw;
            let x = aux[p];
            let v = if c < no {
                let w = op_w[r] / den1;
                let pr = F::exp(x - lse[r * 3 + 1]);
                let y = if op_ids[r] as usize == c { one } else { zero };
                w * (pr - y)
            } else if c < 2 * no {
                let w = w03 * op_w[r] * inv_no / den3;
                let sig = one / (one + F::exp(x * F::new(-1.0_f32)));
                w * (sig - opset[r * no + (c - no)])
            } else if c < 2 * no + nc {
                let w = w03 * crop_w[r] / den2;
                let pr = F::exp(x - lse[r * 3 + 2]);
                let y = if crop_ids[r] as usize == c - 2 * no {
                    one
                } else {
                    zero
                };
                w * (pr - y)
            } else {
                let w = w01 * eta_w[r] / den4;
                w * two * (x - eta_log[r])
            };
            d_aux[p] = gu * v;
        }
    }
}

/// Adjoint of the fused loss in one launch: every output element depends only
/// on its own row and the saved scalars — a pure gather, no atomics. `logits`
/// is `[B*Q, n1]`, `aux` is `[B*Q, aw]`, `lse` is `[B*Q, 3]`, `report` is the
/// `[11]` forward report (denominators read back on the device).
#[allow(clippy::too_many_arguments)]
pub fn planner_loss_backward<R: Runtime, E: FloatElem>(
    upstream: &Tensor<R, E>,
    logits: &Tensor<R, E>,
    aux: &Tensor<R, E>,
    tables: &LossTables<R, E>,
    lse: &Tensor<R, E>,
    report: &Tensor<R, E>,
) -> Result<(Tensor<R, E>, Tensor<R, E>)> {
    let bq = logits.shape().dim(0);
    let (n1, aw) = (logits.shape().dim(1), aux.shape().dim(1));
    let (no, nc) = (tables.n_ops, tables.n_crops);
    let device = logits.device();
    let d_logits = Tensor::empty(Shape::new(vec![bq, n1]), device);
    let d_aux = Tensor::empty(Shape::new(vec![bq, aw]), device);
    let log_lanes = bq * n1;
    let lanes = log_lanes + bq * aw;
    if lanes == 0 {
        return Ok((d_logits, d_aux));
    }
    let (cube_count, cube_dim, span) = launch_1d_spans(logits.client(), lanes, 1);
    unsafe {
        planner_loss_backward_kernel::launch_unchecked::<E, R>(
            logits.client(),
            cube_count,
            cube_dim,
            upstream.arg(),
            logits.arg(),
            aux.arg(),
            tables.target_ids.arg(),
            tables.target_w.arg(),
            tables.op_ids.arg(),
            tables.op_w.arg(),
            tables.crop_ids.arg(),
            tables.crop_w.arg(),
            tables.opset.arg(),
            tables.eta_log.arg(),
            tables.eta_w.arg(),
            lse.arg(),
            report.arg(),
            d_logits.arg(),
            d_aux.arg(),
            n1,
            aw,
            no,
            nc,
            bq,
            log_lanes,
            lanes,
            span,
            E::from_scalar(tables.loss_scale),
            E::from_scalar(1.0 / tables.n_ops as f32),
        );
    }
    Ok((d_logits, d_aux))
}

// ---------------------------------------------------------------------------
// K4. Grid transpose
// ---------------------------------------------------------------------------

#[cube(launch_unchecked)]
fn grid_transpose_kernel<F: Float + CubeElement, N: Size>(
    input: &Array<Vector<F, N>>,
    output: &mut Array<Vector<F, N>>,
    g: usize,
    dvec: usize,
    lanes: usize,
    span: usize,
) {
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    for pos in start..end {
        let dv = pos % dvec;
        let cell = (pos / dvec) % (g * g);
        let b = pos / (dvec * g * g);
        let y = cell / g;
        let x = cell % g;
        output[(b * g * g + x * g + y) * dvec + dv] = input[pos];
    }
}

/// Transpose `[B, g*g, d]` row-major <-> column-major in one launch. The
/// transpose is its own inverse, so the adjoint is the same kernel applied
/// to the upstream gradient (one launch).
pub fn grid_transpose<R: Runtime, E: FloatElem>(
    input: &Tensor<R, E>,
    g: usize,
) -> Result<Tensor<R, E>> {
    if input.rank() != 3 || input.shape().dim(1) != g * g {
        return Err(Error::shape(format!(
            "grid_transpose needs [B, g*g, d], got {}",
            input.shape()
        )));
    }
    let (b, d) = (input.shape().dim(0), input.shape().dim(2));
    let out = Tensor::empty(Shape::new(vec![b, g * g, d]), input.device());
    if out.len() == 0 {
        return Ok(out);
    }
    let line = line_dividing::<R, E>(input.client(), &[d]);
    let dvec = d / line;
    let lanes = b * g * g * dvec;
    let (cube_count, cube_dim, span) = launch_1d_spans(input.client(), lanes, dvec);
    unsafe {
        grid_transpose_kernel::launch_unchecked::<E, R>(
            input.client(),
            cube_count,
            cube_dim,
            line,
            input.arg(),
            out.arg(),
            g,
            dvec,
            lanes,
            span,
        );
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// K5. Tile embedding join and the NONE key
// ---------------------------------------------------------------------------

#[cube(launch_unchecked)]
fn tile_embed_join_kernel<F: Float + CubeElement, N: Size>(
    x: &Array<Vector<F, N>>,
    pos: &Array<Vector<F, N>>,
    g: &Array<Vector<F, N>>,
    out: &mut Array<Vector<F, N>>,
    n: usize,
    dvec: usize,
    lanes: usize,
    span: usize,
) {
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    for p in start..end {
        let dv = p % dvec;
        let nn = (p / dvec) % n;
        let b = p / (dvec * n);
        out[p] = x[p] + pos[nn * dvec + dv] + g[b * dvec + dv];
    }
}

/// Join tile embeddings in one launch: `t[b,n,:] = x[b,n,:] + pos[n,:] +
/// g[b,:]`. `x` is `[B,N,d]`, `pos` is `[N,d]`, `g` is `[B,d]`.
pub fn tile_embed_join<R: Runtime, E: FloatElem>(
    x: &Tensor<R, E>,
    pos: &Tensor<R, E>,
    g: &Tensor<R, E>,
) -> Result<Tensor<R, E>> {
    let (b, n, d) = (x.shape().dim(0), x.shape().dim(1), x.shape().dim(2));
    if x.rank() != 3 || pos.dims() != &[n, d] || g.dims() != &[b, d] {
        return Err(Error::shape(format!(
            "tile_embed_join needs x [B,N,d], pos [N,d], g [B,d]; got {}, {} and {}",
            x.shape(),
            pos.shape(),
            g.shape()
        )));
    }
    let out = Tensor::empty(Shape::new(vec![b, n, d]), x.device());
    if out.len() == 0 {
        return Ok(out);
    }
    let line = line_dividing::<R, E>(x.client(), &[d]);
    let dvec = d / line;
    let lanes = b * n * dvec;
    let (cube_count, cube_dim, span) = launch_1d_spans(x.client(), lanes, dvec);
    unsafe {
        tile_embed_join_kernel::launch_unchecked::<E, R>(
            x.client(),
            cube_count,
            cube_dim,
            line,
            x.arg(),
            pos.arg(),
            g.arg(),
            out.arg(),
            n,
            dvec,
            lanes,
            span,
        );
    }
    Ok(out)
}

#[cube(launch_unchecked)]
fn tile_embed_join_backward_kernel<F: Float + CubeElement, N: Size>(
    grad: &Array<Vector<F, N>>,
    d_pos: &mut Array<Vector<F, N>>,
    d_g: &mut Array<Vector<F, N>>,
    n: usize,
    dvec: usize,
    b: usize,
    pos_lanes: usize,
    lanes: usize,
    span: usize,
) {
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    for p in start..end {
        if p < pos_lanes {
            // d_pos[nn,:] = sum over b of grad[b,nn,:].
            let dv = p % dvec;
            let nn = p / dvec;
            let mut acc = Vector::<F, N>::new(F::new(0.0_f32));
            for bb in 0..b {
                acc += grad[(bb * n + nn) * dvec + dv];
            }
            d_pos[p] = acc;
        } else {
            // d_g[bb,:] = sum over n of grad[bb,n,:].
            let q = p - pos_lanes;
            let dv = q % dvec;
            let bb = q / dvec;
            let mut acc = Vector::<F, N>::new(F::new(0.0_f32));
            for nn in 0..n {
                acc += grad[(bb * n + nn) * dvec + dv];
            }
            d_g[q] = acc;
        }
    }
}

/// Adjoint of [`tile_embed_join`] in one launch with two gather regions:
/// `d_pos` sums over the batch, `d_g` sums over tiles. `d_x` is the upstream
/// gradient itself (no launch) — handled in the `Var` wrapper.
pub fn tile_embed_join_backward<R: Runtime, E: FloatElem>(
    grad: &Tensor<R, E>,
    n: usize,
) -> Result<(Tensor<R, E>, Tensor<R, E>)> {
    let (b, d) = (grad.shape().dim(0), grad.shape().dim(2));
    if grad.rank() != 3 || grad.shape().dim(1) != n {
        return Err(Error::shape(format!(
            "tile_embed_join_backward needs grad [B,{n},d], got {}",
            grad.shape()
        )));
    }
    let device = grad.device();
    let d_pos = Tensor::empty(Shape::new(vec![n, d]), device);
    let d_g = Tensor::empty(Shape::new(vec![b, d]), device);
    let line = line_dividing::<R, E>(grad.client(), &[d]);
    let dvec = d / line;
    let pos_lanes = n * dvec;
    let lanes = pos_lanes + b * dvec;
    if lanes == 0 {
        return Ok((d_pos, d_g));
    }
    let (cube_count, cube_dim, span) = launch_1d_spans(grad.client(), lanes, dvec);
    unsafe {
        tile_embed_join_backward_kernel::launch_unchecked::<E, R>(
            grad.client(),
            cube_count,
            cube_dim,
            line,
            grad.arg(),
            d_pos.arg(),
            d_g.arg(),
            n,
            dvec,
            b,
            pos_lanes,
            lanes,
            span,
        );
    }
    Ok((d_pos, d_g))
}

#[cube(launch_unchecked)]
fn keys_with_none_kernel<F: Float + CubeElement, N: Size>(
    t2: &Array<Vector<F, N>>,
    none: &Array<Vector<F, N>>,
    out: &mut Array<Vector<F, N>>,
    n: usize,
    dvec: usize,
    lanes: usize,
    span: usize,
) {
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    for p in start..end {
        let dv = p % dvec;
        let nn = (p / dvec) % (n + 1);
        let b = p / (dvec * (n + 1));
        if nn < n {
            out[p] = t2[(b * n + nn) * dvec + dv];
        } else {
            out[p] = none[dv];
        }
    }
}

/// Append the NONE key in one launch: `keys[b,n,:] = n < N ? t2[b,n,:] :
/// none[0,:]`. `t2` is `[B,N,d]`, `none` is `[1,d]`, output `[B,N+1,d]`.
pub fn keys_with_none<R: Runtime, E: FloatElem>(
    t2: &Tensor<R, E>,
    none: &Tensor<R, E>,
) -> Result<Tensor<R, E>> {
    let (b, n, d) = (t2.shape().dim(0), t2.shape().dim(1), t2.shape().dim(2));
    if t2.rank() != 3 || none.dims() != &[1, d] {
        return Err(Error::shape(format!(
            "keys_with_none needs t2 [B,N,d] and none [1,d]; got {} and {}",
            t2.shape(),
            none.shape()
        )));
    }
    let out = Tensor::empty(Shape::new(vec![b, n + 1, d]), t2.device());
    if out.len() == 0 {
        return Ok(out);
    }
    let line = line_dividing::<R, E>(t2.client(), &[d]);
    let dvec = d / line;
    let lanes = b * (n + 1) * dvec;
    let (cube_count, cube_dim, span) = launch_1d_spans(t2.client(), lanes, dvec);
    unsafe {
        keys_with_none_kernel::launch_unchecked::<E, R>(
            t2.client(),
            cube_count,
            cube_dim,
            line,
            t2.arg(),
            none.arg(),
            out.arg(),
            n,
            dvec,
            lanes,
            span,
        );
    }
    Ok(out)
}

#[cube(launch_unchecked)]
fn keys_with_none_backward_kernel<F: Float + CubeElement, N: Size>(
    grad: &Array<Vector<F, N>>,
    d_none: &mut Array<Vector<F, N>>,
    n: usize,
    dvec: usize,
    b: usize,
    copy_lanes: usize,
    lanes: usize,
    span: usize,
) {
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    for p in start..end {
        if p < copy_lanes {
            // d_t2[b,n,:] = grad[b,n,:] for n < N — handled as a copy region
            // is unnecessary: the wrapper slices the gradient (no launch).
            // This region is unreachable by construction; kept for symmetry.
        } else {
            // d_none[:] = sum over b of grad[b,N,:].
            let q = (p - copy_lanes) % dvec;
            let mut acc = Vector::<F, N>::new(F::new(0.0_f32));
            for bb in 0..b {
                acc += grad[(bb * (n + 1) + n) * dvec + q];
            }
            d_none[q] = acc;
        }
    }
}

/// Adjoint of [`keys_with_none`]: `d_t2` is a slice of the upstream gradient
/// (no launch — handled in the `Var` wrapper); `d_none` is one launch
/// gathering the NONE row over the batch.
pub fn keys_with_none_backward<R: Runtime, E: FloatElem>(
    grad: &Tensor<R, E>,
    n: usize,
) -> Result<Tensor<R, E>> {
    let (b, d) = (grad.shape().dim(0), grad.shape().dim(2));
    if grad.rank() != 3 || grad.shape().dim(1) != n + 1 {
        return Err(Error::shape(format!(
            "keys_with_none_backward needs grad [B, N+1, d], got {}",
            grad.shape()
        )));
    }
    let d_none = Tensor::empty(Shape::new(vec![1, d]), grad.device());
    let line = line_dividing::<R, E>(grad.client(), &[d]);
    let dvec = d / line;
    let copy_lanes = b * n * dvec;
    let lanes = copy_lanes + dvec;
    if dvec == 0 {
        return Ok(d_none);
    }
    let (cube_count, cube_dim, span) = launch_1d_spans(grad.client(), lanes, dvec);
    unsafe {
        keys_with_none_backward_kernel::launch_unchecked::<E, R>(
            grad.client(),
            cube_count,
            cube_dim,
            line,
            grad.arg(),
            d_none.arg(),
            n,
            dvec,
            b,
            copy_lanes,
            lanes,
            span,
        );
    }
    Ok(d_none)
}
