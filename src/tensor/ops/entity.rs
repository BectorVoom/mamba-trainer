//! One-launch preparation of an entity set: split, zeroing and presence statistics.
//!
//! The composed path this replaces is, per set, the `features`/`presence` split
//! in [`crate::rl::ObsSpec::split`], the `features * presence` zeroing in
//! [`crate::nn::entity::EntityEncoder::apply`], and the four launches of
//! [`crate::nn::entity::Presence::new`] (`sum_dim`, `clamp_max`, `clamp_min`,
//! `div`). At these sizes the arithmetic is nothing and the launches are the
//! cost, so one thread per `(row, n)` that loops over the set's `N` presence
//! values (at most a few hundred) trades redundant summation for six fewer
//! dispatches.
//!
//! Semantics match the composed path, including non-0/1 presence: `p` is
//! copied verbatim into `legal`, so the masks downstream treat any nonzero `p`
//! as present exactly as before; the mean weight is `p / max(1, sum p)`, `any`
//! is `min(1, sum p)`, and the features are multiplied by `p` (so `0.5` halves
//! them, as `features.mul(presence)` does). The row sum accumulates serially in
//! slot order; [`crate::tensor::ops::reduce::sum_dim`] may fold lanes in another
//! order, so the two can differ in the last bits of the sum, which
//! `tests/rl_entity_parity.rs` bounds at 1e-6.

use cubecl::prelude::*;

use crate::backend::{FloatElem, launch_1d};
use crate::error::{Error, Result};
use crate::tensor::base::Tensor;
use crate::tensor::ops::index::IdTensor;
use crate::tensor::shape::Shape;

/// One thread per `(row, n)`: the row sum over the set's presence column, then
/// this slot's zeroed features, mean weight, legal flag, and — for `n == 0` —
/// the row's `any` flag.
#[cube(launch_unchecked)]
fn entity_prepare_kernel<F: Float + CubeElement>(
    obs: &Array<F>,
    features: &mut Array<F>,
    mean_w: &mut Array<F>,
    legal: &mut Array<F>,
    any: &mut Array<F>,
    off: usize,
    count: usize,
    feats: usize,
    obs_dim: usize,
    lanes: usize,
) {
    if ABSOLUTE_POS < lanes {
        let row = ABSOLUTE_POS / count;
        let n = ABSOLUTE_POS % count;
        let stride = feats + 1;

        // Serial, slot 0 first. Every thread of the row recomputes the same sum,
        // which costs N reads each and saves a launch.
        let mut sum = F::new(0.0_f32);
        for m in 0..count {
            sum += obs[row * obs_dim + off + m * stride + feats];
        }
        let one = F::new(1.0_f32);
        let denom = sum.max(one);

        let p = obs[row * obs_dim + off + n * stride + feats];
        mean_w[ABSOLUTE_POS] = p / denom;
        legal[ABSOLUTE_POS] = p;
        if n == 0 {
            any[row] = sum.min(one);
        }

        let obs_base = row * obs_dim + off + n * stride;
        let out_base = ABSOLUTE_POS * feats;
        for f in 0..feats {
            features[out_base + f] = obs[obs_base + f] * p;
        }
    }
}

/// Adjoint of [`entity_prepare_kernel`] with respect to `obs`.
///
/// Presence gets no gradient, matching the composed path, where `Presence::new`
/// is off the tape by construction: a feature column of this set's band pulls
/// `d_features[row, n, f] * p`, and every other column — the presence column,
/// other sets' columns, globals — is zero. One launch writes the whole
/// `[rows, obs_dim]` gradient, so no separate zero-fill is needed.
#[cube(launch_unchecked)]
fn entity_prepare_backward_kernel<F: Float + CubeElement>(
    grad_features: &Array<F>,
    obs: &Array<F>,
    d_obs: &mut Array<F>,
    off: usize,
    count: usize,
    feats: usize,
    obs_dim: usize,
    lanes: usize,
) {
    if ABSOLUTE_POS < lanes {
        let row = ABSOLUTE_POS / obs_dim;
        let col = ABSOLUTE_POS % obs_dim;
        let width = count * (feats + 1);
        let mut v = F::new(0.0_f32);
        if col >= off && col < off + width {
            let rel = col - off;
            let stride = feats + 1;
            let n = rel / stride;
            let k = rel % stride;
            if k < feats {
                let p = obs[row * obs_dim + off + n * stride + feats];
                v = grad_features[(row * count + n) * feats + k] * p;
            }
        }
        d_obs[ABSOLUTE_POS] = v;
    }
}

/// Split one set's band out of flat `obs` `[rows, obs_dim]`, zero its features
/// by presence, and compute its presence statistics — in one launch.
///
/// `off` is the set's column offset in `obs`, `count` its slot count `N`, and
/// `features` its per-entity feature count `F`. Returns `features [rows, N, F]`
/// (already multiplied by presence), `mean_w [rows, N]`, `legal [rows, N]` (a
/// contiguous copy of the presence column), and `any [rows]`.
#[allow(clippy::type_complexity)] // Four buffers of one type; naming them would not help.
pub fn entity_prepare<R: Runtime, E: FloatElem>(
    obs: &Tensor<R, E>,
    off: usize,
    count: usize,
    features: usize,
) -> Result<(Tensor<R, E>, Tensor<R, E>, Tensor<R, E>, Tensor<R, E>)> {
    if obs.rank() != 2 {
        return Err(Error::shape(format!(
            "entity_prepare needs a flat [rows, obs_dim] observation, got {}",
            obs.shape()
        )));
    }
    let rows = obs.shape().dim(0);
    let obs_dim = obs.shape().dim(1);
    if count == 0 || features == 0 {
        return Err(Error::shape(format!(
            "entity_prepare needs a positive count and feature width, got N={count} F={features}"
        )));
    }
    let width = count * (features + 1);
    if off + width > obs_dim {
        return Err(Error::shape(format!(
            "entity_prepare band [off={off}, width={width}) exceeds obs_dim={obs_dim}"
        )));
    }
    let out_features = Tensor::empty(Shape::new(vec![rows, count, features]), obs.device());
    let mean_w = Tensor::empty(Shape::new(vec![rows, count]), obs.device());
    let legal = Tensor::empty(Shape::new(vec![rows, count]), obs.device());
    let any = Tensor::empty(Shape::new(vec![rows]), obs.device());
    let lanes = rows * count;
    if lanes == 0 {
        return Ok((out_features, mean_w, legal, any));
    }
    let (cube_count, cube_dim) = launch_1d(obs.client(), lanes, count + features);
    unsafe {
        entity_prepare_kernel::launch_unchecked::<E, R>(
            obs.client(),
            cube_count,
            cube_dim,
            obs.arg(),
            out_features.arg(),
            mean_w.arg(),
            legal.arg(),
            any.arg(),
            off,
            count,
            features,
            obs_dim,
            lanes,
        );
    }
    Ok((out_features, mean_w, legal, any))
}

/// Adjoint of [`entity_prepare`]: the `[rows, obs_dim]` gradient of `obs` from
/// the `[rows, N, F]` gradient of its prepared features.
pub fn entity_prepare_backward<R: Runtime, E: FloatElem>(
    grad_features: &Tensor<R, E>,
    obs: &Tensor<R, E>,
    off: usize,
    count: usize,
    features: usize,
) -> Result<Tensor<R, E>> {
    if grad_features.rank() != 3
        || grad_features.shape().dim(1) != count
        || grad_features.shape().dim(2) != features
    {
        return Err(Error::shape(format!(
            "entity_prepare_backward needs grad [rows, {count}, {features}], got {}",
            grad_features.shape()
        )));
    }
    if obs.rank() != 2 {
        return Err(Error::shape(format!(
            "entity_prepare_backward needs obs [rows, obs_dim], got {}",
            obs.shape()
        )));
    }
    let rows = obs.shape().dim(0);
    let obs_dim = obs.shape().dim(1);
    if grad_features.shape().dim(0) != rows {
        return Err(Error::shape(format!(
            "entity_prepare_backward grad has {} rows but obs has {rows}",
            grad_features.shape().dim(0)
        )));
    }
    let width = count * (features + 1);
    if off + width > obs_dim {
        return Err(Error::shape(format!(
            "entity_prepare_backward band [off={off}, width={width}) exceeds obs_dim={obs_dim}"
        )));
    }
    let d_obs = Tensor::empty(Shape::new(vec![rows, obs_dim]), obs.device());
    let lanes = rows * obs_dim;
    if lanes == 0 {
        return Ok(d_obs);
    }
    let (cube_count, cube_dim) = launch_1d(obs.client(), lanes, 1);
    unsafe {
        entity_prepare_backward_kernel::launch_unchecked::<E, R>(
            obs.client(),
            cube_count,
            cube_dim,
            grad_features.arg(),
            obs.arg(),
            d_obs.arg(),
            off,
            count,
            features,
            obs_dim,
            lanes,
        );
    }
    Ok(d_obs)
}

/// One thread per pooled column (plus the globals copy on the first set's
/// launch): mean and max straight into the joined buffer.
///
/// The composed path this replaces is, per set, the mean batched matmul, the
/// `mask_logits` → `max_dim` → `× any` max chain, that set's share of the
/// globals-plus-pools `cat`, and — on the first set's launch — the globals
/// `slice`. Every set writes disjoint columns of the same `joined [rows, W]`
/// buffer, so one launch per set turns "pool, then cat" into a single pass
/// with no intermediate pool tensors.
///
/// All launches must run before `joined` is read, and together they must write
/// every column: the buffer starts uninitialised. Consecutive launches on one
/// device stream are ordered, so issuing them back to back is enough.
///
/// `e` is `[rows, N, d]`, `mean_w`/`legal` are `[rows, N]` and `any` is
/// `[rows]` (all three from [`entity_prepare`]); `obs` is `[rows, obs_dim]`,
/// read only when `copy_globals` is set. `off_mean`/`off_max` are this set's
/// column offsets in `joined`; the lanes cover `rows * (Gc + d)` positions,
/// where `Gc` is `globals` on the copying launch and 0 otherwise.
///
/// # Ties
///
/// The composed `max_dim` shares the gradient equally among tied maxima, while
/// [`entity_pool_backward`] sends it all to the first maximum in slot order.
/// Both are valid subgradients; the fused path does not emulate the sharing,
/// which would cost a second pass. Parity tests use tie-free data.
#[cube]
#[allow(clippy::too_many_arguments)]
fn entity_pool_body<F: Float + CubeElement>(
    e: &Array<F>,
    mean_w: &Array<F>,
    legal: &Array<F>,
    any: &Array<F>,
    joined: &mut Array<F>,
    argmax: &mut Array<u32>,
    row: usize,
    j: usize,
    count: usize,
    width: usize,
    stride_w: usize,
    off_mean: usize,
    off_max: usize,
    #[comptime] has_mean: bool,
    #[comptime] has_max: bool,
) {
    if comptime!(has_mean) {
        let mut acc: f32 = 0.0;
        for n in 0..count {
            acc += f32::cast_from(mean_w[row * count + n])
                * f32::cast_from(e[(row * count + n) * width + j]);
        }
        joined[row * stride_w + off_mean + j] = F::cast_from(acc);
    }
    if comptime!(has_max) {
        // From the first *present* entry, never from a sentinel: no
        // finite stand-in can be outranked by a large embedding, and an
        // empty set keeps index 0 with `any = 0` zeroing the value.
        let mut best: f32 = 0.0;
        let mut idx: u32 = u32::cast_from(0u32);
        let mut seen: u32 = u32::cast_from(0u32);
        for n in 0..count {
            if f32::cast_from(legal[row * count + n]) != 0.0 {
                let v = f32::cast_from(e[(row * count + n) * width + j]);
                if seen == u32::cast_from(0u32) || v > best {
                    best = v;
                    idx = n as u32;
                }
                seen += u32::cast_from(1u32);
            }
        }
        let mut v: f32 = 0.0;
        if f32::cast_from(any[row]) != 0.0 {
            v = best;
        }
        joined[row * stride_w + off_max + j] = F::cast_from(v);
        argmax[row * width + j] = idx;
    }
}

#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn entity_pool_kernel<F: Float + CubeElement>(
    e: &Array<F>,
    mean_w: &Array<F>,
    legal: &Array<F>,
    any: &Array<F>,
    obs: &Array<F>,
    joined: &mut Array<F>,
    argmax: &mut Array<u32>,
    count: usize,
    width: usize,
    stride_w: usize,
    off_mean: usize,
    off_max: usize,
    gc: usize,
    obs_dim: usize,
    lanes: usize,
    #[comptime] has_mean: bool,
    #[comptime] has_max: bool,
    #[comptime] copy_globals: bool,
) {
    if ABSOLUTE_POS < lanes {
        // `gc` is `globals` on the copying launch and 0 otherwise (chosen on
        // the host). The comptime flag keeps the globals copy — and the `obs`
        // read — out of the non-copying specialization entirely; the pool body
        // is shared so both specializations compute identical values.
        if comptime!(copy_globals) {
            let row = ABSOLUTE_POS / (gc + width);
            let c = ABSOLUTE_POS % (gc + width);
            if c < gc {
                joined[row * stride_w + c] = obs[row * obs_dim + c];
            } else {
                entity_pool_body::<F>(
                    e, mean_w, legal, any, joined, argmax, row, c - gc, count, width,
                    stride_w, off_mean, off_max, has_mean, has_max,
                );
            }
        } else {
            let row = ABSOLUTE_POS / width;
            let j = ABSOLUTE_POS % width;
            entity_pool_body::<F>(
                e, mean_w, legal, any, joined, argmax, row, j, count, width, stride_w,
                off_mean, off_max, has_mean, has_max,
            );
        }
    }
}

/// Adjoint of [`entity_pool_kernel`] with respect to `e`: a gather with no
/// atomics.
///
/// One thread per element of `d_e [rows, N, d]` reads the joined gradient at
/// this set's offsets (the joined node owns that buffer, so no split launch is
/// needed):
/// `d_e = g_mean * w + (argmax == n) * any * legal * g_max`. The max term
/// multiplies by `legal` exactly as the composed `mask_logits` rule does, so
/// non-0/1 presence (e.g. 0.5) weights the max gradient the way it does today;
/// a masked slot gets exactly zero — it is never the argmax, and `legal = 0`
/// zeroes it anyway — and an empty set's `any = 0` zeroes the term.
///
/// See the [`entity_pool_kernel`] doc comment for why tied maxima route their
/// whole gradient to the first index here instead of sharing it.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn entity_pool_backward_kernel<F: Float + CubeElement>(
    grad: &Array<F>,
    mean_w: &Array<F>,
    legal: &Array<F>,
    any: &Array<F>,
    argmax: &Array<u32>,
    d_e: &mut Array<F>,
    count: usize,
    width: usize,
    stride_w: usize,
    off_mean: usize,
    off_max: usize,
    lanes: usize,
    #[comptime] has_mean: bool,
    #[comptime] has_max: bool,
) {
    if ABSOLUTE_POS < lanes {
        let j = ABSOLUTE_POS % width;
        let n = (ABSOLUTE_POS / width) % count;
        let row = ABSOLUTE_POS / (width * count);
        let mut v: f32 = 0.0;
        if comptime!(has_mean) {
            v += f32::cast_from(grad[row * stride_w + off_mean + j])
                * f32::cast_from(mean_w[row * count + n]);
        }
        if comptime!(has_max) {
            if argmax[row * width + j] == n as u32
                && f32::cast_from(any[row]) != 0.0
            {
                v += f32::cast_from(grad[row * stride_w + off_max + j])
                    * f32::cast_from(legal[row * count + n]);
            }
        }
        d_e[ABSOLUTE_POS] = F::cast_from(v);
    }
}

/// Pool one set's `[rows, N, d]` embeddings into its columns of the shared
/// `joined [rows, W]` buffer, in one launch.
///
/// `off_mean`/`off_max` are this set's pool offsets in `joined` (unused when
/// the matching `has_*` flag is false), `globals` is `G`, and `copy_globals`
/// must be true on exactly one launch — the first set's — when `G > 0`, which
/// also copies the globals there. `argmax [rows, d]` receives the max indices
/// for the adjoint; it is untouched when `has_max` is false.
#[allow(clippy::too_many_arguments)]
pub fn entity_pool<R: Runtime, E: FloatElem>(
    e: &Tensor<R, E>,
    mean_w: &Tensor<R, E>,
    legal: &Tensor<R, E>,
    any: &Tensor<R, E>,
    obs: &Tensor<R, E>,
    joined: &Tensor<R, E>,
    argmax: &IdTensor<R>,
    off_mean: usize,
    off_max: usize,
    globals: usize,
    has_mean: bool,
    has_max: bool,
    copy_globals: bool,
) -> Result<()> {
    if e.rank() != 3 {
        return Err(Error::shape(format!(
            "entity_pool needs e [rows, N, d], got {}",
            e.shape()
        )));
    }
    let rows = e.shape().dim(0);
    let count = e.shape().dim(1);
    let width = e.shape().dim(2);
    if count == 0 || width == 0 {
        return Err(Error::shape(format!(
            "entity_pool needs a positive count and width, got N={count} d={width}"
        )));
    }
    if mean_w.dims() != &[rows, count] || legal.dims() != &[rows, count] {
        return Err(Error::shape(format!(
            "entity_pool needs mean_w and legal [rows, {count}] for {rows} rows, got {} and {}",
            mean_w.shape(),
            legal.shape()
        )));
    }
    if any.dims() != &[rows] {
        return Err(Error::shape(format!(
            "entity_pool needs any [{rows}], got {}",
            any.shape()
        )));
    }
    if obs.rank() != 2 || obs.shape().dim(0) != rows {
        return Err(Error::shape(format!(
            "entity_pool needs obs [{rows}, obs_dim], got {}",
            obs.shape()
        )));
    }
    if joined.rank() != 2 || joined.shape().dim(0) != rows {
        return Err(Error::shape(format!(
            "entity_pool needs joined [{rows}, W], got {}",
            joined.shape()
        )));
    }
    if argmax.shape().dims() != &[rows, width] {
        return Err(Error::shape(format!(
            "entity_pool needs argmax [{rows}, {width}], got {}",
            argmax.shape()
        )));
    }
    let stride_w = joined.shape().dim(1);
    let obs_dim = obs.shape().dim(1);
    if copy_globals && globals > obs_dim {
        return Err(Error::shape(format!(
            "entity_pool copies {globals} globals from obs_dim={obs_dim}"
        )));
    }
    if has_mean && off_mean + width > stride_w {
        return Err(Error::shape(format!(
            "entity_pool mean band [{off_mean}, {}) exceeds W={stride_w}",
            off_mean + width
        )));
    }
    if has_max && off_max + width > stride_w {
        return Err(Error::shape(format!(
            "entity_pool max band [{off_max}, {}) exceeds W={stride_w}",
            off_max + width
        )));
    }
    let gc = if copy_globals { globals } else { 0 };
    let lanes = rows * (gc + width);
    if lanes == 0 {
        return Ok(());
    }
    let (cube_count, cube_dim) = launch_1d(joined.client(), lanes, count + 1);
    unsafe {
        entity_pool_kernel::launch_unchecked::<E, R>(
            joined.client(),
            cube_count,
            cube_dim,
            e.arg(),
            mean_w.arg(),
            legal.arg(),
            any.arg(),
            obs.arg(),
            joined.arg(),
            argmax.arg(),
            count,
            width,
            stride_w,
            off_mean,
            off_max,
            gc,
            obs_dim,
            lanes,
            has_mean,
            has_max,
            copy_globals,
        );
    }
    Ok(())
}

/// Adjoint of [`entity_pool`]: the `[rows, N, d]` gradient of one set's
/// embeddings from the `[rows, W]` gradient of the joined buffer.
#[allow(clippy::too_many_arguments)]
pub fn entity_pool_backward<R: Runtime, E: FloatElem>(
    grad: &Tensor<R, E>,
    mean_w: &Tensor<R, E>,
    legal: &Tensor<R, E>,
    any: &Tensor<R, E>,
    argmax: &IdTensor<R>,
    count: usize,
    width: usize,
    off_mean: usize,
    off_max: usize,
    has_mean: bool,
    has_max: bool,
) -> Result<Tensor<R, E>> {
    if grad.rank() != 2 {
        return Err(Error::shape(format!(
            "entity_pool_backward needs grad [rows, W], got {}",
            grad.shape()
        )));
    }
    let rows = grad.shape().dim(0);
    let stride_w = grad.shape().dim(1);
    if mean_w.dims() != &[rows, count] || legal.dims() != &[rows, count] {
        return Err(Error::shape(format!(
            "entity_pool_backward needs mean_w and legal [{rows}, {count}], got {} and {}",
            mean_w.shape(),
            legal.shape()
        )));
    }
    if any.dims() != &[rows] {
        return Err(Error::shape(format!(
            "entity_pool_backward needs any [{rows}], got {}",
            any.shape()
        )));
    }
    if argmax.shape().dims() != &[rows, width] {
        return Err(Error::shape(format!(
            "entity_pool_backward needs argmax [{rows}, {width}], got {}",
            argmax.shape()
        )));
    }
    let d_e = Tensor::empty(Shape::new(vec![rows, count, width]), grad.device());
    let lanes = rows * count * width;
    if lanes == 0 {
        return Ok(d_e);
    }
    let (cube_count, cube_dim) = launch_1d(grad.client(), lanes, 1);
    unsafe {
        entity_pool_backward_kernel::launch_unchecked::<E, R>(
            grad.client(),
            cube_count,
            cube_dim,
            grad.arg(),
            mean_w.arg(),
            legal.arg(),
            any.arg(),
            argmax.arg(),
            d_e.arg(),
            count,
            width,
            stride_w,
            off_mean,
            off_max,
            lanes,
            has_mean,
            has_max,
        );
    }
    Ok(d_e)
}

// ---------------------------------------------------------------------------
// K3: pointer_scores — the scorer, the mask and the extras in one launch.
// ---------------------------------------------------------------------------

/// One thread per `(row, c)`, `c < N + K`: the additive scorer, the presence
/// mask and the extras copy in a single launch.
///
/// The composed path this replaces is, for the additive head, the broadcast
/// add of `k = W_e e [rows, N, H]` and `q = W_h h + b [rows, H]`, the ReLU,
/// the `v` matmul, the reshape, `mask_logits` and the extras `cat`. `q`
/// already includes the bias (it rides on `W_h`, as in the composed path);
/// `v` is the `[H, 1]` weight read as a flat `[H]` buffer.
///
/// A masked position writes exactly `F::min_value()`, the same value the
/// `mask_logits` binary op writes, so the downstream softmax sees an exact
/// zero probability either way. Accumulation is in `f32` whatever `E` is, so
/// narrow storage does not change the reduction's error.
///
/// `extra` is read only when `has_extra` is set; otherwise the caller passes a
/// dummy buffer (the branch is compiled out) and `cols == n`, so the copy arm
/// is unreachable.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn pointer_additive_kernel<F: Float + CubeElement>(
    k: &Array<F>,
    q: &Array<F>,
    v: &Array<F>,
    legal: &Array<F>,
    extra: &Array<F>,
    logits: &mut Array<F>,
    n: usize,
    h: usize,
    kx: usize,
    cols: usize,
    lanes: usize,
    #[comptime] has_extra: bool,
) {
    if ABSOLUTE_POS < lanes {
        let row = ABSOLUTE_POS / cols;
        let c = ABSOLUTE_POS % cols;
        if c < n {
            if f32::cast_from(legal[row * n + c]) != 0.0 {
                let mut acc: f32 = 0.0;
                for hh in 0..h {
                    let pre = f32::cast_from(k[(row * n + c) * h + hh])
                        + f32::cast_from(q[row * h + hh]);
                    let mut relu = pre;
                    if relu < 0.0 {
                        relu = 0.0;
                    }
                    acc += f32::cast_from(v[hh]) * relu;
                }
                logits[ABSOLUTE_POS] = F::cast_from(acc);
            } else {
                logits[ABSOLUTE_POS] = F::min_value();
            }
        } else {
            if comptime!(has_extra) {
                logits[ABSOLUTE_POS] = extra[row * kx + (c - n)];
            }
        }
    }
}

/// Score `[rows, N, H]` keys against `[rows, H]` queries with a `[H]` weight,
/// mask by `[rows, N]` presence and append `[rows, K]` extras — in one launch.
///
/// `n` is the entity count `N`; the output is `[rows, N + K]` with `K` from
/// `extra` (0 when `None`). All inputs are contiguous.
pub fn pointer_additive<R: Runtime, E: FloatElem>(
    k: &Tensor<R, E>,
    q: &Tensor<R, E>,
    v: &Tensor<R, E>,
    legal: &Tensor<R, E>,
    extra: Option<&Tensor<R, E>>,
    n: usize,
) -> Result<Tensor<R, E>> {
    if k.rank() != 3 || k.shape().dim(1) != n {
        return Err(Error::shape(format!(
            "pointer_additive needs k [rows, {n}, H], got {}",
            k.shape()
        )));
    }
    let rows = k.shape().dim(0);
    let h = k.shape().dim(2);
    if h == 0 {
        return Err(Error::shape(format!(
            "pointer_additive needs a positive hidden width, got {}",
            k.shape()
        )));
    }
    if q.dims() != &[rows, h] {
        return Err(Error::shape(format!(
            "pointer_additive needs q [{rows}, {h}], got {}",
            q.shape()
        )));
    }
    if v.len() != h {
        return Err(Error::shape(format!(
            "pointer_additive needs v with {h} elements, got {}",
            v.shape()
        )));
    }
    if legal.dims() != &[rows, n] {
        return Err(Error::shape(format!(
            "pointer_additive needs legal [{rows}, {n}], got {}",
            legal.shape()
        )));
    }
    let kx = match extra {
        Some(t) => {
            if t.rank() != 2 || t.shape().dim(0) != rows {
                return Err(Error::shape(format!(
                    "pointer_additive needs extra [{rows}, K], got {}",
                    t.shape()
                )));
            }
            t.shape().dim(1)
        }
        None => 0,
    };
    // A dummy buffer when there is nothing to copy: the copy arm is compiled
    // out of that specialization, so it is never read.
    let extra_buf = extra.unwrap_or(legal);
    let has_extra = extra.is_some();
    let cols = n + kx;
    let logits = Tensor::empty(Shape::new(vec![rows, cols]), k.device());
    let lanes = rows * cols;
    if lanes == 0 {
        return Ok(logits);
    }
    let (cube_count, cube_dim) = launch_1d(k.client(), lanes, h);
    unsafe {
        pointer_additive_kernel::launch_unchecked::<E, R>(
            k.client(),
            cube_count,
            cube_dim,
            k.arg(),
            q.arg(),
            v.arg(),
            legal.arg(),
            extra_buf.arg(),
            logits.arg(),
            n,
            h,
            kx,
            cols,
            lanes,
            has_extra,
        );
    }
    Ok(logits)
}

/// Adjoint of [`pointer_additive`] with respect to `k` (plus `d_extra`): a
/// gather with no atomics.
///
/// One thread per element of `d_k [rows, N, H]`, plus `rows * K` more threads
/// that copy `d_extra[row, kk] = g[row, N + kk]` into their own buffer, all in
/// one lanes range branching on position:
///
/// `d_k = g * legal * v[h] * [k + q > 0]`.
///
/// The pre-activation is recomputed, so nothing `[rows, N, H]` is stored. The
/// `legal` factor is the presence *value*, exactly as the composed
/// `mask_logits` rule multiplies by `legal` (a 0.5 presence halves the
/// gradient); the ReLU gate is strict (`> 0`), exactly as the composed
/// `relu` rule's `gt_scalar` is. A masked slot gets exactly zero.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn pointer_additive_dk_kernel<F: Float + CubeElement>(
    grad: &Array<F>,
    k: &Array<F>,
    q: &Array<F>,
    v: &Array<F>,
    legal: &Array<F>,
    d_k: &mut Array<F>,
    d_extra: &mut Array<F>,
    n: usize,
    h: usize,
    kx: usize,
    cols: usize,
    dk_lanes: usize,
    lanes: usize,
    #[comptime] has_extra: bool,
) {
    if ABSOLUTE_POS < lanes {
        if ABSOLUTE_POS < dk_lanes {
            let hh = ABSOLUTE_POS % h;
            let nn = (ABSOLUTE_POS / h) % n;
            let row = ABSOLUTE_POS / (h * n);
            let pre = f32::cast_from(k[(row * n + nn) * h + hh])
                + f32::cast_from(q[row * h + hh]);
            let mut gate: f32 = 0.0;
            if pre > 0.0 {
                gate = 1.0;
            }
            let val = f32::cast_from(grad[row * cols + nn])
                * f32::cast_from(legal[row * n + nn])
                * f32::cast_from(v[hh])
                * gate;
            d_k[ABSOLUTE_POS] = F::cast_from(val);
        } else {
            if comptime!(has_extra) {
                let epos = ABSOLUTE_POS - dk_lanes;
                let kk = epos % kx;
                let row = epos / kx;
                d_extra[epos] = grad[row * cols + n + kk];
            }
        }
    }
}

/// The `d_k` (and `d_extra`) half of [`pointer_additive`]'s adjoint.
///
/// `grad` is `[rows, N + K]`, `v` a flat `[H]` buffer; returns
/// `d_k [rows, N, H]` and `d_extra [rows, K]` (`K` may be 0, in which case the
/// second buffer is empty and no thread touches it).
#[allow(clippy::too_many_arguments)]
pub fn pointer_additive_backward_dk<R: Runtime, E: FloatElem>(
    grad: &Tensor<R, E>,
    k: &Tensor<R, E>,
    q: &Tensor<R, E>,
    v: &Tensor<R, E>,
    legal: &Tensor<R, E>,
    n: usize,
    kx: usize,
) -> Result<(Tensor<R, E>, Tensor<R, E>)> {
    let rows = k.shape().dim(0);
    let h = k.shape().dim(2);
    if k.dims() != &[rows, n, h] || q.dims() != &[rows, h] || v.len() != h {
        return Err(Error::shape(format!(
            "pointer_additive_backward_dk needs k [{rows}, {n}, {h}], q [{rows}, {h}] and v with {h} elements, got {} and {} and {}",
            k.shape(),
            q.shape(),
            v.shape()
        )));
    }
    if grad.dims() != &[rows, n + kx] || legal.dims() != &[rows, n] {
        return Err(Error::shape(format!(
            "pointer_additive_backward_dk needs grad [{rows}, {}] and legal [{rows}, {n}], got {} and {}",
            n + kx,
            grad.shape(),
            legal.shape()
        )));
    }
    let d_k = Tensor::empty(Shape::new(vec![rows, n, h]), grad.device());
    let d_extra = Tensor::empty(Shape::new(vec![rows, kx]), grad.device());
    let dk_lanes = rows * n * h;
    let lanes = dk_lanes + rows * kx;
    if lanes == 0 {
        return Ok((d_k, d_extra));
    }
    let (cube_count, cube_dim) = launch_1d(grad.client(), lanes, 1);
    unsafe {
        pointer_additive_dk_kernel::launch_unchecked::<E, R>(
            grad.client(),
            cube_count,
            cube_dim,
            grad.arg(),
            k.arg(),
            q.arg(),
            v.arg(),
            legal.arg(),
            d_k.arg(),
            d_extra.arg(),
            n,
            h,
            kx,
            n + kx,
            dk_lanes,
            lanes,
            kx > 0,
        );
    }
    Ok((d_k, d_extra))
}

/// Adjoint of [`pointer_additive`] with respect to `q` and `v`: one thread
/// per `(row, h)` looping over `n` (a gather).
///
/// `d_q[row, h] = Σ_n d_k[row, n, h]`, recomputed from the saved inputs rather
/// than read from the other backward kernel's output, so the two launches are
/// independent; `dv_partial[row, h] = Σ_n g * legal * max(0, k + q)`, whose
/// rows the caller sums into `d_v`. Same `legal`-value and strict-gate
/// conventions as [`pointer_additive_dk_kernel`].
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn pointer_additive_dq_kernel<F: Float + CubeElement>(
    grad: &Array<F>,
    k: &Array<F>,
    q: &Array<F>,
    v: &Array<F>,
    legal: &Array<F>,
    d_q: &mut Array<F>,
    dv_partial: &mut Array<F>,
    n: usize,
    h: usize,
    cols: usize,
    lanes: usize,
) {
    if ABSOLUTE_POS < lanes {
        let hh = ABSOLUTE_POS % h;
        let row = ABSOLUTE_POS / h;
        let vh = f32::cast_from(v[hh]);
        let qh = f32::cast_from(q[row * h + hh]);
        let mut dq: f32 = 0.0;
        let mut dv: f32 = 0.0;
        for nn in 0..n {
            let g = f32::cast_from(grad[row * cols + nn])
                * f32::cast_from(legal[row * n + nn]);
            let pre = f32::cast_from(k[(row * n + nn) * h + hh]) + qh;
            if pre > 0.0 {
                dq += g * vh;
                dv += g * pre;
            }
        }
        d_q[ABSOLUTE_POS] = F::cast_from(dq);
        dv_partial[ABSOLUTE_POS] = F::cast_from(dv);
    }
}

/// The `d_q` (and per-row `d_v`) half of [`pointer_additive`]'s adjoint.
///
/// Returns `d_q [rows, H]` and `dv_partial [rows, H]`; the caller sums the
/// latter over axis 0 with the existing reduction and reshapes it to `[H, 1]`.
pub fn pointer_additive_backward_dq<R: Runtime, E: FloatElem>(
    grad: &Tensor<R, E>,
    k: &Tensor<R, E>,
    q: &Tensor<R, E>,
    v: &Tensor<R, E>,
    legal: &Tensor<R, E>,
    n: usize,
    kx: usize,
) -> Result<(Tensor<R, E>, Tensor<R, E>)> {
    let rows = k.shape().dim(0);
    let h = k.shape().dim(2);
    if k.dims() != &[rows, n, h] || q.dims() != &[rows, h] || v.len() != h {
        return Err(Error::shape(format!(
            "pointer_additive_backward_dq needs k [{rows}, {n}, {h}], q [{rows}, {h}] and v with {h} elements, got {} and {} and {}",
            k.shape(),
            q.shape(),
            v.shape()
        )));
    }
    if grad.dims() != &[rows, n + kx] || legal.dims() != &[rows, n] {
        return Err(Error::shape(format!(
            "pointer_additive_backward_dq needs grad [{rows}, {}] and legal [{rows}, {n}], got {} and {}",
            n + kx,
            grad.shape(),
            legal.shape()
        )));
    }
    let d_q = Tensor::empty(Shape::new(vec![rows, h]), grad.device());
    let dv_partial = Tensor::empty(Shape::new(vec![rows, h]), grad.device());
    let lanes = rows * h;
    if lanes == 0 {
        return Ok((d_q, dv_partial));
    }
    let (cube_count, cube_dim) = launch_1d(grad.client(), lanes, n + 1);
    unsafe {
        pointer_additive_dq_kernel::launch_unchecked::<E, R>(
            grad.client(),
            cube_count,
            cube_dim,
            grad.arg(),
            k.arg(),
            q.arg(),
            v.arg(),
            legal.arg(),
            d_q.arg(),
            dv_partial.arg(),
            n,
            h,
            n + kx,
            lanes,
        );
    }
    Ok((d_q, dv_partial))
}

/// One thread per `(row, c)`, `c < N + K`: the dot scorer, the presence mask
/// and the extras copy in a single launch.
///
/// The composed path this replaces is the batched matmul of `e [rows, N, d]`
/// against `qd = W_q h [rows, d]` (unsqueezed to `[.., d, 1]`, a poor shape
/// for the tiled kernel), then `mask_logits` and the extras `cat`. Masked
/// positions write exactly `F::min_value()`; accumulation is in `f32`.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn pointer_dot_kernel<F: Float + CubeElement>(
    e: &Array<F>,
    qd: &Array<F>,
    legal: &Array<F>,
    extra: &Array<F>,
    logits: &mut Array<F>,
    n: usize,
    d: usize,
    kx: usize,
    cols: usize,
    lanes: usize,
    #[comptime] has_extra: bool,
) {
    if ABSOLUTE_POS < lanes {
        let row = ABSOLUTE_POS / cols;
        let c = ABSOLUTE_POS % cols;
        if c < n {
            if f32::cast_from(legal[row * n + c]) != 0.0 {
                let mut acc: f32 = 0.0;
                for dd in 0..d {
                    acc += f32::cast_from(e[(row * n + c) * d + dd])
                        * f32::cast_from(qd[row * d + dd]);
                }
                logits[ABSOLUTE_POS] = F::cast_from(acc);
            } else {
                logits[ABSOLUTE_POS] = F::min_value();
            }
        } else {
            if comptime!(has_extra) {
                logits[ABSOLUTE_POS] = extra[row * kx + (c - n)];
            }
        }
    }
}

/// Score `[rows, N, d]` entities against `[rows, d]` queries, mask by
/// `[rows, N]` presence and append `[rows, K]` extras — in one launch.
///
/// `n` is the entity count `N`; the output is `[rows, N + K]` with `K` from
/// `extra` (0 when `None`). All inputs are contiguous.
pub fn pointer_dot<R: Runtime, E: FloatElem>(
    e: &Tensor<R, E>,
    qd: &Tensor<R, E>,
    legal: &Tensor<R, E>,
    extra: Option<&Tensor<R, E>>,
    n: usize,
) -> Result<Tensor<R, E>> {
    if e.rank() != 3 || e.shape().dim(1) != n {
        return Err(Error::shape(format!(
            "pointer_dot needs e [rows, {n}, d], got {}",
            e.shape()
        )));
    }
    let rows = e.shape().dim(0);
    let d = e.shape().dim(2);
    if d == 0 {
        return Err(Error::shape(format!(
            "pointer_dot needs a positive entity width, got {}",
            e.shape()
        )));
    }
    if qd.dims() != &[rows, d] {
        return Err(Error::shape(format!(
            "pointer_dot needs qd [{rows}, {d}], got {}",
            qd.shape()
        )));
    }
    if legal.dims() != &[rows, n] {
        return Err(Error::shape(format!(
            "pointer_dot needs legal [{rows}, {n}], got {}",
            legal.shape()
        )));
    }
    let kx = match extra {
        Some(t) => {
            if t.rank() != 2 || t.shape().dim(0) != rows {
                return Err(Error::shape(format!(
                    "pointer_dot needs extra [{rows}, K], got {}",
                    t.shape()
                )));
            }
            t.shape().dim(1)
        }
        None => 0,
    };
    // A dummy buffer when there is nothing to copy: the copy arm is compiled
    // out of that specialization, so it is never read.
    let extra_buf = extra.unwrap_or(legal);
    let has_extra = extra.is_some();
    let cols = n + kx;
    let logits = Tensor::empty(Shape::new(vec![rows, cols]), e.device());
    let lanes = rows * cols;
    if lanes == 0 {
        return Ok(logits);
    }
    let (cube_count, cube_dim) = launch_1d(e.client(), lanes, d);
    unsafe {
        pointer_dot_kernel::launch_unchecked::<E, R>(
            e.client(),
            cube_count,
            cube_dim,
            e.arg(),
            qd.arg(),
            legal.arg(),
            extra_buf.arg(),
            logits.arg(),
            n,
            d,
            kx,
            cols,
            lanes,
            has_extra,
        );
    }
    Ok(logits)
}

/// Adjoint of [`pointer_dot`] with respect to `e` (plus `d_extra`): a gather
/// with no atomics.
///
/// One thread per element of `d_e [rows, N, d]`, plus `rows * K` more threads
/// copying `d_extra[row, kk] = g[row, N + kk]`:
///
/// `d_e = g * legal * qd[row, d]`.
///
/// The `legal` factor is the presence *value*, exactly as the composed
/// `mask_logits` rule multiplies by `legal`.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn pointer_dot_de_kernel<F: Float + CubeElement>(
    grad: &Array<F>,
    qd: &Array<F>,
    legal: &Array<F>,
    d_e: &mut Array<F>,
    d_extra: &mut Array<F>,
    n: usize,
    d: usize,
    kx: usize,
    cols: usize,
    de_lanes: usize,
    lanes: usize,
    #[comptime] has_extra: bool,
) {
    if ABSOLUTE_POS < lanes {
        if ABSOLUTE_POS < de_lanes {
            let dd = ABSOLUTE_POS % d;
            let nn = (ABSOLUTE_POS / d) % n;
            let row = ABSOLUTE_POS / (d * n);
            let val = f32::cast_from(grad[row * cols + nn])
                * f32::cast_from(legal[row * n + nn])
                * f32::cast_from(qd[row * d + dd]);
            d_e[ABSOLUTE_POS] = F::cast_from(val);
        } else {
            if comptime!(has_extra) {
                let epos = ABSOLUTE_POS - de_lanes;
                let kk = epos % kx;
                let row = epos / kx;
                d_extra[epos] = grad[row * cols + n + kk];
            }
        }
    }
}

/// The `d_e` (and `d_extra`) half of [`pointer_dot`]'s adjoint.
///
/// Returns `d_e [rows, N, d]` and `d_extra [rows, K]`.
pub fn pointer_dot_backward_de<R: Runtime, E: FloatElem>(
    grad: &Tensor<R, E>,
    e: &Tensor<R, E>,
    qd: &Tensor<R, E>,
    legal: &Tensor<R, E>,
    n: usize,
    kx: usize,
) -> Result<(Tensor<R, E>, Tensor<R, E>)> {
    let rows = e.shape().dim(0);
    let d = e.shape().dim(2);
    if e.dims() != &[rows, n, d] || qd.dims() != &[rows, d] {
        return Err(Error::shape(format!(
            "pointer_dot_backward_de needs e [{rows}, {n}, {d}] and qd [{rows}, {d}], got {} and {}",
            e.shape(),
            qd.shape()
        )));
    }
    if grad.dims() != &[rows, n + kx] || legal.dims() != &[rows, n] {
        return Err(Error::shape(format!(
            "pointer_dot_backward_de needs grad [{rows}, {}] and legal [{rows}, {n}], got {} and {}",
            n + kx,
            grad.shape(),
            legal.shape()
        )));
    }
    let d_e = Tensor::empty(Shape::new(vec![rows, n, d]), grad.device());
    let d_extra = Tensor::empty(Shape::new(vec![rows, kx]), grad.device());
    let de_lanes = rows * n * d;
    let lanes = de_lanes + rows * kx;
    if lanes == 0 {
        return Ok((d_e, d_extra));
    }
    let (cube_count, cube_dim) = launch_1d(grad.client(), lanes, 1);
    unsafe {
        pointer_dot_de_kernel::launch_unchecked::<E, R>(
            grad.client(),
            cube_count,
            cube_dim,
            grad.arg(),
            qd.arg(),
            legal.arg(),
            d_e.arg(),
            d_extra.arg(),
            n,
            d,
            kx,
            n + kx,
            de_lanes,
            lanes,
            kx > 0,
        );
    }
    Ok((d_e, d_extra))
}

/// Adjoint of [`pointer_dot`] with respect to `qd`: one thread per `(row, d)`
/// looping over `n` (a gather).
///
/// `d_qd[row, d] = Σ_n g[row, n] * legal[row, n] * e[row, n, d]`.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn pointer_dot_dqd_kernel<F: Float + CubeElement>(
    grad: &Array<F>,
    e: &Array<F>,
    legal: &Array<F>,
    d_qd: &mut Array<F>,
    n: usize,
    d: usize,
    cols: usize,
    lanes: usize,
) {
    if ABSOLUTE_POS < lanes {
        let dd = ABSOLUTE_POS % d;
        let row = ABSOLUTE_POS / d;
        let mut acc: f32 = 0.0;
        for nn in 0..n {
            acc += f32::cast_from(grad[row * cols + nn])
                * f32::cast_from(legal[row * n + nn])
                * f32::cast_from(e[(row * n + nn) * d + dd]);
        }
        d_qd[ABSOLUTE_POS] = F::cast_from(acc);
    }
}

/// The `d_qd` half of [`pointer_dot`]'s adjoint.
///
/// Returns `d_qd [rows, d]`.
pub fn pointer_dot_backward_dqd<R: Runtime, E: FloatElem>(
    grad: &Tensor<R, E>,
    e: &Tensor<R, E>,
    legal: &Tensor<R, E>,
    n: usize,
    kx: usize,
) -> Result<Tensor<R, E>> {
    let rows = e.shape().dim(0);
    let d = e.shape().dim(2);
    if e.dims() != &[rows, n, d] {
        return Err(Error::shape(format!(
            "pointer_dot_backward_dqd needs e [{rows}, {n}, {d}], got {}",
            e.shape()
        )));
    }
    if grad.dims() != &[rows, n + kx] || legal.dims() != &[rows, n] {
        return Err(Error::shape(format!(
            "pointer_dot_backward_dqd needs grad [{rows}, {}] and legal [{rows}, {n}], got {} and {}",
            n + kx,
            grad.shape(),
            legal.shape()
        )));
    }
    let d_qd = Tensor::empty(Shape::new(vec![rows, d]), grad.device());
    let lanes = rows * d;
    if lanes == 0 {
        return Ok(d_qd);
    }
    let (cube_count, cube_dim) = launch_1d(grad.client(), lanes, n + 1);
    unsafe {
        pointer_dot_dqd_kernel::launch_unchecked::<E, R>(
            grad.client(),
            cube_count,
            cube_dim,
            grad.arg(),
            e.arg(),
            legal.arg(),
            d_qd.arg(),
            n,
            d,
            n + kx,
            lanes,
        );
    }
    Ok(d_qd)
}

// ---------------------------------------------------------------------------
// K4: bias_relu — the encoder's hidden layers.
// ---------------------------------------------------------------------------

/// One thread per element: `out[i] = max(0, pre[i] + bias[i % h])`.
///
/// The composed path this replaces is, per hidden layer of
/// [`crate::nn::entity::EntityEncoder`], the bias broadcast add inside
/// [`crate::nn::linear::Linear::apply`] and the `relu` that follows it: two
/// launches down to one. The bias is indexed by a plain modulo of the flat
/// position rather than through the broadcast-metadata path, so this kernel
/// uploads no shape/stride buffer at all. Kept scalar on purpose: it is a
/// flat elementwise map, and the launch — not the arithmetic — is the cost.
///
/// Accumulation is in `f32` whatever `E` is, so narrow storage does not
/// change the rounding of the sum.
#[cube(launch_unchecked)]
fn bias_relu_kernel<F: Float + CubeElement>(
    pre: &Array<F>,
    bias: &Array<F>,
    out: &mut Array<F>,
    h: usize,
    lanes: usize,
) {
    if ABSOLUTE_POS < lanes {
        let v = f32::cast_from(pre[ABSOLUTE_POS]) + f32::cast_from(bias[ABSOLUTE_POS % h]);
        let mut r = v;
        if r < 0.0 {
            r = 0.0;
        }
        out[ABSOLUTE_POS] = F::cast_from(r);
    }
}

/// Adjoint of [`bias_relu_kernel`] with respect to `pre`: a flat gather of
/// the saved output.
///
/// One thread per element: `d_pre[i] = g[i] * (y[i] > 0 ? 1 : 0)`. The gate
/// reads the saved *output* `y`, which is exactly the strict `> 0` test the
/// composed `relu` rule applies to its input (`gt_scalar(&x, 0.0)`): `y > 0`
/// iff `pre + bias > 0`, so at `y == 0` the gradient is 0 on both paths.
/// The bias gradient needs no second kernel: it is this same `d_pre` summed
/// over every axis but the last, which the caller does with the existing
/// `reduce_grad_to` — exactly what [`crate::autograd::Var::add`]'s rule does
/// for a broadcast bias, so the gradient is identical.
#[cube(launch_unchecked)]
fn bias_relu_backward_kernel<F: Float + CubeElement>(
    grad: &Array<F>,
    y: &Array<F>,
    d_pre: &mut Array<F>,
    lanes: usize,
) {
    if ABSOLUTE_POS < lanes {
        let mut gate: f32 = 0.0;
        if f32::cast_from(y[ABSOLUTE_POS]) > 0.0 {
            gate = 1.0;
        }
        d_pre[ABSOLUTE_POS] =
            F::cast_from(f32::cast_from(grad[ABSOLUTE_POS]) * gate);
    }
}

fn require_bias_matches_trailing<R: Runtime, E: FloatElem>(
    pre: &Tensor<R, E>,
    bias: &Tensor<R, E>,
) -> Result<usize> {
    if pre.rank() < 1 {
        return Err(Error::shape(format!(
            "bias_relu needs a pre-activation with at least one axis, got {}",
            pre.shape()
        )));
    }
    let h = pre.shape().dim_from_end(0);
    if h == 0 {
        return Err(Error::shape(format!(
            "bias_relu needs a positive trailing axis, got {}",
            pre.shape()
        )));
    }
    if bias.rank() != 1 || bias.len() != h {
        return Err(Error::shape(format!(
            "bias_relu needs a rank-1 bias matching pre's trailing axis of {h}, got {}",
            bias.shape()
        )));
    }
    Ok(h)
}

/// `max(pre + bias, 0)`, where `bias` holds one value per element of `pre`'s
/// trailing axis, broadcast over every leading axis — in one launch.
pub fn bias_relu<R: Runtime, E: FloatElem>(
    pre: &Tensor<R, E>,
    bias: &Tensor<R, E>,
) -> Result<Tensor<R, E>> {
    let h = require_bias_matches_trailing(pre, bias)?;
    let out = Tensor::empty(pre.shape().clone(), pre.device());
    let lanes = out.len();
    if lanes == 0 {
        return Ok(out);
    }
    let (cube_count, cube_dim) = launch_1d(pre.client(), lanes, 1);
    unsafe {
        bias_relu_kernel::launch_unchecked::<E, R>(
            pre.client(),
            cube_count,
            cube_dim,
            pre.arg(),
            bias.arg(),
            out.arg(),
            h,
            lanes,
        );
    }
    Ok(out)
}

/// Adjoint of [`bias_relu`] with respect to `pre`: `g * (y > 0 ? 1 : 0)` from
/// the saved output `y` — in one launch.
pub fn bias_relu_backward<R: Runtime, E: FloatElem>(
    grad: &Tensor<R, E>,
    y: &Tensor<R, E>,
) -> Result<Tensor<R, E>> {
    if grad.shape() != y.shape() {
        return Err(Error::shape(format!(
            "bias_relu_backward needs grad and y with matching shapes, got {} and {}",
            grad.shape(),
            y.shape()
        )));
    }
    let d_pre = Tensor::empty(grad.shape().clone(), grad.device());
    let lanes = d_pre.len();
    if lanes == 0 {
        return Ok(d_pre);
    }
    let (cube_count, cube_dim) = launch_1d(grad.client(), lanes, 1);
    unsafe {
        bias_relu_backward_kernel::launch_unchecked::<E, R>(
            grad.client(),
            cube_count,
            cube_dim,
            grad.arg(),
            y.arg(),
            d_pre.arg(),
            lanes,
        );
    }
    Ok(d_pre)
}
