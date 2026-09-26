//! One-launch preparation of an entity set: split, zeroing and presence statistics.
//!
//! The composed path this replaces is, per set, the `features`/`presence` split
//! in [`crate::rl::ObsSpec::split`], the `features * presence` zeroing in
//! [`crate::nn::entity::EntityEncoder::apply`], and the four launches of
//! [`crate::nn::entity::Presence::new`] (`sum_dim`, `clamp_max`, `clamp_min`,
//! `div`). One launch does all of it: a stats lane per row sums the presence
//! column once, and the remaining lanes copy the zeroed features.
//!
//! Semantics match the composed path, including non-0/1 presence: `p` is
//! copied verbatim into `legal`, so the masks downstream treat any nonzero `p`
//! as present exactly as before; the mean weight is `p / max(1, sum p)`, `any`
//! is `min(1, sum p)`, and the features are multiplied by `p` (so `0.5` halves
//! them, as `features.mul(presence)` does). The row sum accumulates serially in
//! slot order; [`crate::tensor::ops::reduce::sum_dim`] may fold lanes in another
//! order, so the two can differ in the last bits of the sum, which
//! `tests/rl_entity_parity.rs` bounds at 1e-6.
//!
//! # Launch shape
//!
//! Every kernel here launches through [`crate::backend::launch_1d_spans`]: on a
//! GPU one lane per unit, laid out so neighbouring lanes touch neighbouring
//! addresses; on CubeCL's CPU runtime one contiguous span of lanes per worker
//! thread, so no two threads write into the same cache line. Kernels whose
//! inner axis is contiguous (`d`, `H`, the pooled columns) read and write
//! [`Vector`]s as wide as every offset and stride along it allows, which is
//! SIMD on the CPU and vector loads on a GPU. `tests/entity_kernels_wide.rs`
//! checks each kernel at a shape that exercises both.

use cubecl::prelude::*;

use crate::backend::{FloatElem, launch_1d_spans, line_size_for};
use crate::error::{Error, Result};
use crate::tensor::base::Tensor;
use crate::tensor::ops::index::IdTensor;
use crate::tensor::shape::Shape;

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// The widest vector width the device likes for `E` that divides every one of
/// `parts` — the extents, offsets and strides a kernel indexes in vectors — so
/// no vector straddles a row or a band. With `ids`, the width must also suit a
/// `u32` vector, for a kernel that moves argmax indices at the same width.
fn line_dividing<R: Runtime, E: FloatElem>(
    client: &ComputeClient<R>,
    parts: &[usize],
    ids: bool,
) -> usize {
    let line = line_size_for::<R, E>(client, parts.iter().fold(0, |g, &p| gcd(g, p)));
    if !ids {
        return line;
    }
    client
        .io_optimized_vector_sizes(core::mem::size_of::<u32>())
        .find(|width| line.is_multiple_of(*width))
        .unwrap_or(1)
}

/// The row sum over one row's presence column — once, serially, slot 0 first —
/// and the row's mean weights, legal flags and `any` from it.
#[cube]
#[allow(clippy::too_many_arguments)]
fn entity_prepare_stats<F: Float + CubeElement>(
    obs: &Array<F>,
    mean_w: &mut Array<F>,
    legal: &mut Array<F>,
    any: &mut Array<F>,
    base: usize,
    row: usize,
    count: usize,
    feats: usize,
) {
    let stride = feats + 1;
    let mut sum = F::new(0.0_f32);
    for m in 0..count {
        sum += obs[base + m * stride + feats];
    }
    let one = F::new(1.0_f32);
    let denom = sum.max(one);
    for m in 0..count {
        let p = obs[base + m * stride + feats];
        mean_w[row * count + m] = p / denom;
        legal[row * count + m] = p;
    }
    any[row] = sum.min(one);
}

/// Each row of lanes is one *stats* lane — [`entity_prepare_stats`] — followed
/// by the lanes that write `features[row, n, f] = obs[.., n, f] · p[n]`. The
/// row sum is computed once per row, not once per slot.
///
/// How the feature lanes are cut depends on the runtime (`per_slot`, chosen on
/// the host):
///
/// * On a GPU one lane per feature, `lanes = rows · (1 + N·F)`, so the feature
///   stores — nearly all of the traffic — are coalesced across a plane. The
///   stats lane diverges from the feature lanes sharing its plane, but that is
///   one plane per row out of `N·F / plane` of them. A unit's first lane is
///   decoded once and stepped thereafter.
/// * On CubeCL's CPU runtime one lane per slot, `lanes = rows · (1 + N)`, each
///   copying its `F` features in a tight loop; there a unit is a thread
///   walking its own contiguous span of lanes, and coalescing buys nothing.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
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
    span: usize,
    #[comptime] per_slot: bool,
) {
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    let stride = feats + 1;
    if comptime!(per_slot) {
        for pos in start..end {
            let row = pos / (count + 1);
            let c = pos % (count + 1);
            let base = row * obs_dim + off;
            if c == 0 {
                entity_prepare_stats::<F>(obs, mean_w, legal, any, base, row, count, feats);
            } else {
                let n = c - 1;
                let slot = base + n * stride;
                let p = obs[slot + feats];
                let out = (row * count + n) * feats;
                for f in 0..feats {
                    features[out + f] = obs[slot + f] * p;
                }
            }
        }
    } else {
        if start < end {
            let block = count * feats + 1;
            let mut row = start / block;
            let mut c = start % block;
            let mut n = 0usize;
            let mut f = 0usize;
            if c > 0 {
                n = (c - 1) / feats;
                f = (c - 1) % feats;
            }
            for _i in start..end {
                let base = row * obs_dim + off;
                if c == 0 {
                    entity_prepare_stats::<F>(obs, mean_w, legal, any, base, row, count, feats);
                } else {
                    let slot = base + n * stride;
                    features[(row * count + n) * feats + f] = obs[slot + f] * obs[slot + feats];
                    f += 1;
                    if f == feats {
                        f = 0;
                        n += 1;
                    }
                }
                c += 1;
                if c == block {
                    c = 0;
                    row += 1;
                    n = 0;
                    f = 0;
                }
            }
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
///
/// The lanes are cut as in [`entity_prepare_kernel`] (`per_slot` chosen the
/// same way):
///
/// * on a GPU one lane per element of `d_obs`, `lanes = rows · obs_dim`, so
///   the stores coalesce; a unit's first lane is decoded once and stepped;
/// * on the CPU runtime, per row, one lane zeroing every column outside the
///   band and one lane per slot writing its `F` gradients and a zero for its
///   presence column, `lanes = rows · (1 + N)`.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn entity_prepare_backward_kernel<F: Float + CubeElement>(
    grad_features: &Array<F>,
    obs: &Array<F>,
    d_obs: &mut Array<F>,
    off: usize,
    count: usize,
    feats: usize,
    obs_dim: usize,
    lanes: usize,
    span: usize,
    #[comptime] per_slot: bool,
) {
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    let zero = F::new(0.0_f32);
    if comptime!(per_slot) {
        let stride = feats + 1;
        for pos in start..end {
            let row = pos / (count + 1);
            let c = pos % (count + 1);
            let rbase = row * obs_dim;
            if c == 0 {
                for col in 0..off {
                    d_obs[rbase + col] = zero;
                }
                for col in off + count * stride..obs_dim {
                    d_obs[rbase + col] = zero;
                }
            } else {
                let n = c - 1;
                let slot = rbase + off + n * stride;
                let p = obs[slot + feats];
                let g = (row * count + n) * feats;
                for k in 0..feats {
                    d_obs[slot + k] = grad_features[g + k] * p;
                }
                d_obs[slot + feats] = zero;
            }
        }
    } else if start < end {
        let stride = feats + 1;
        let width = count * stride;
        let mut row = start / obs_dim;
        let mut col = start % obs_dim;
        // `(n, k)` is the band position of `col` while it is inside the band,
        // and `(0, 0)` — where the band starts — while it is before it.
        let mut n = 0usize;
        let mut k = 0usize;
        if col > off && col < off + width {
            n = (col - off) / stride;
            k = (col - off) % stride;
        }
        for pos in start..end {
            let mut v = F::new(0.0_f32);
            if col >= off && col < off + width {
                if k < feats {
                    let p = obs[row * obs_dim + off + n * stride + feats];
                    v = grad_features[(row * count + n) * feats + k] * p;
                }
                k += 1;
                if k == stride {
                    k = 0;
                    n += 1;
                }
            }
            d_obs[pos] = v;
            col += 1;
            if col == obs_dim {
                col = 0;
                row += 1;
                n = 0;
                k = 0;
            }
        }
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
    if rows == 0 {
        return Ok((out_features, mean_w, legal, any));
    }
    let per_slot = obs.client().properties().hardware.plane_size_max <= 1;
    let (lanes, work) = if per_slot {
        (rows * (count + 1), 2 * features)
    } else {
        (rows * (count * features + 1), 2)
    };
    let (cube_count, cube_dim, span) = launch_1d_spans(obs.client(), lanes, work);
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
            span,
            per_slot,
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
    if rows * obs_dim == 0 {
        return Ok(d_obs);
    }
    let per_slot = obs.client().properties().hardware.plane_size_max <= 1;
    let (lanes, work) = if per_slot {
        (rows * (count + 1), features + 1)
    } else {
        (rows * obs_dim, 1)
    };
    let (cube_count, cube_dim, span) = launch_1d_spans(obs.client(), lanes, work);
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
            span,
            per_slot,
        );
    }
    Ok(d_obs)
}

/// One lane per vector of pooled columns (plus the globals copy on the first
/// set's launch): mean and max straight into the joined buffer.
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
///
/// # Vectors
///
/// Everything indexed by the embedding column — `e`, `joined`, `argmax` and,
/// on the copying launch, `obs` — is read as `Vector`s of `N` adjacent
/// columns, and a lane pools `N` columns at once: one vector load of `e` per
/// slot, and the mean and the max accumulate from that same load. `N` divides
/// every column offset and stride (the host picks it), so a vector never
/// straddles two bands. Each column still accumulates serially in slot order,
/// exactly as a scalar lane would.
#[cube]
#[allow(clippy::too_many_arguments)]
fn entity_pool_body<F: Float + CubeElement, N: Size>(
    e: &Array<Vector<F, N>>,
    mean_w: &Array<F>,
    legal: &Array<F>,
    any: &Array<F>,
    joined: &mut Array<Vector<F, N>>,
    argmax: &mut Array<Vector<u32, N>>,
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
    let zero = Vector::<f32, N>::new(0.0_f32);
    let mut acc = zero;
    // The max starts from the first *present* entry, never from a sentinel: no
    // finite stand-in can be outranked by a large embedding, and an empty set
    // keeps index 0 with `any = 0` zeroing the value.
    let mut best = zero;
    let mut idx = Vector::<u32, N>::new(0u32);
    let mut seen = false;
    for n in 0..count {
        let x = Vector::<f32, N>::cast_from(e[(row * count + n) * width + j]);
        if comptime!(has_mean) {
            acc += Vector::<f32, N>::new(f32::cast_from(mean_w[row * count + n])) * x;
        }
        if comptime!(has_max) {
            if f32::cast_from(legal[row * count + n]) != 0.0 {
                if seen {
                    let take = x.greater_than(best);
                    best = select_many(take, x, best);
                    idx = select_many(take, Vector::<u32, N>::new(n as u32), idx);
                } else {
                    best = x;
                    idx = Vector::<u32, N>::new(n as u32);
                    seen = true;
                }
            }
        }
    }
    if comptime!(has_mean) {
        joined[row * stride_w + off_mean + j] = Vector::<F, N>::cast_from(acc);
    }
    if comptime!(has_max) {
        let mut v = zero;
        if f32::cast_from(any[row]) != 0.0 {
            v = best;
        }
        joined[row * stride_w + off_max + j] = Vector::<F, N>::cast_from(v);
        argmax[row * width + j] = idx;
    }
}

/// `width`, `stride_w`, `off_mean`, `off_max`, `gc` and `obs_dim` are all in
/// vectors of `N` columns; lanes are `rows · (gc + width)` of them, walked
/// `span` at a time by each unit.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn entity_pool_kernel<F: Float + CubeElement, N: Size>(
    e: &Array<Vector<F, N>>,
    mean_w: &Array<F>,
    legal: &Array<F>,
    any: &Array<F>,
    obs: &Array<Vector<F, N>>,
    joined: &mut Array<Vector<F, N>>,
    argmax: &mut Array<Vector<u32, N>>,
    count: usize,
    width: usize,
    stride_w: usize,
    off_mean: usize,
    off_max: usize,
    gc: usize,
    obs_dim: usize,
    lanes: usize,
    span: usize,
    #[comptime] has_mean: bool,
    #[comptime] has_max: bool,
    #[comptime] copy_globals: bool,
) {
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    // `gc` is `globals` on the copying launch and 0 otherwise (chosen on the
    // host). The comptime flag keeps the globals copy — and the `obs` read —
    // out of the non-copying specialization entirely; the pool body is shared
    // so both specializations compute identical values.
    for pos in start..end {
        let row = pos / (gc + width);
        let c = pos % (gc + width);
        if comptime!(copy_globals) {
            if c < gc {
                joined[row * stride_w + c] = obs[row * obs_dim + c];
            } else {
                entity_pool_body::<F, N>(
                    e,
                    mean_w,
                    legal,
                    any,
                    joined,
                    argmax,
                    row,
                    c - gc,
                    count,
                    width,
                    stride_w,
                    off_mean,
                    off_max,
                    has_mean,
                    has_max,
                );
            }
        } else {
            entity_pool_body::<F, N>(
                e, mean_w, legal, any, joined, argmax, row, c, count, width, stride_w, off_mean,
                off_max, has_mean, has_max,
            );
        }
    }
}

/// Adjoint of [`entity_pool_kernel`] with respect to `e`: a gather with no
/// atomics.
///
/// One lane per vector of `d_e [rows, N, d]` reads the joined gradient at
/// this set's offsets (the joined node owns that buffer, so no split launch is
/// needed):
/// `d_e = g_mean * w + (argmax == n) * any * legal * g_max`. The max term
/// multiplies by `legal` exactly as the composed `mask_logits` rule does, so
/// non-0/1 presence (e.g. 0.5) weights the max gradient the way it does today;
/// a masked slot gets exactly zero — it is never the argmax, and `legal = 0`
/// zeroes it anyway — and an empty set's `any = 0` zeroes the term.
///
/// Column arguments are in vectors of `N`, as in [`entity_pool_kernel`]. See
/// its doc comment for why tied maxima route their whole gradient to the first
/// index here instead of sharing it.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn entity_pool_backward_kernel<F: Float + CubeElement, N: Size>(
    grad: &Array<Vector<F, N>>,
    mean_w: &Array<F>,
    legal: &Array<F>,
    any: &Array<F>,
    argmax: &Array<Vector<u32, N>>,
    d_e: &mut Array<Vector<F, N>>,
    count: usize,
    width: usize,
    stride_w: usize,
    off_mean: usize,
    off_max: usize,
    lanes: usize,
    span: usize,
    #[comptime] has_mean: bool,
    #[comptime] has_max: bool,
) {
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    let zero = Vector::<f32, N>::new(0.0_f32);
    for pos in start..end {
        let j = pos % width;
        let n = (pos / width) % count;
        let row = pos / (width * count);
        let mut v = zero;
        if comptime!(has_mean) {
            v += Vector::<f32, N>::cast_from(grad[row * stride_w + off_mean + j])
                * Vector::<f32, N>::new(f32::cast_from(mean_w[row * count + n]));
        }
        if comptime!(has_max) {
            if f32::cast_from(any[row]) != 0.0 {
                let hit = argmax[row * width + j].equal(Vector::<u32, N>::new(n as u32));
                let term = Vector::<f32, N>::cast_from(grad[row * stride_w + off_max + j])
                    * Vector::<f32, N>::new(f32::cast_from(legal[row * count + n]));
                v += select_many(hit, term, zero);
            }
        }
        d_e[pos] = Vector::<F, N>::cast_from(v);
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
    if rows == 0 || gc + width == 0 {
        return Ok(());
    }
    let line = line_dividing::<R, E>(
        joined.client(),
        &[
            width,
            stride_w,
            if has_mean { off_mean } else { 0 },
            if has_max { off_max } else { 0 },
            gc,
            if copy_globals { obs_dim } else { 0 },
        ],
        has_max,
    );
    let lanes = rows * (gc + width) / line;
    let (cube_count, cube_dim, span) = launch_1d_spans(joined.client(), lanes, line * (count + 1));
    unsafe {
        entity_pool_kernel::launch_unchecked::<E, R>(
            joined.client(),
            cube_count,
            cube_dim,
            line,
            e.arg(),
            mean_w.arg(),
            legal.arg(),
            any.arg(),
            obs.arg(),
            joined.arg(),
            argmax.arg(),
            count,
            width / line,
            stride_w / line,
            off_mean / line,
            off_max / line,
            gc / line,
            obs_dim / line,
            lanes,
            span,
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
    if rows * count * width == 0 {
        return Ok(d_e);
    }
    let line = line_dividing::<R, E>(
        grad.client(),
        &[
            width,
            stride_w,
            if has_mean { off_mean } else { 0 },
            if has_max { off_max } else { 0 },
        ],
        has_max,
    );
    let lanes = rows * count * width / line;
    let (cube_count, cube_dim, span) = launch_1d_spans(grad.client(), lanes, line);
    unsafe {
        entity_pool_backward_kernel::launch_unchecked::<E, R>(
            grad.client(),
            cube_count,
            cube_dim,
            line,
            grad.arg(),
            mean_w.arg(),
            legal.arg(),
            any.arg(),
            argmax.arg(),
            d_e.arg(),
            count,
            width / line,
            stride_w / line,
            off_mean / line,
            off_max / line,
            lanes,
            span,
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
/// The reduction over `H` runs on `Vector`s of `N` hidden units — `N` partial
/// sums folded at the end — which reassociates it relative to a serial loop
/// (exactly serial for `N = 1`); the composed matmul it replaces sums in its
/// own order too. The dot scorer below does the same over `d`.
///
/// `extra` is read only when `has_extra` is set; otherwise the caller passes a
/// dummy buffer (the branch is compiled out) and `cols == n`, so the copy arm
/// is unreachable.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn pointer_additive_kernel<F: Float + CubeElement, N: Size>(
    k: &Array<Vector<F, N>>,
    q: &Array<Vector<F, N>>,
    v: &Array<Vector<F, N>>,
    legal: &Array<F>,
    extra: &Array<F>,
    logits: &mut Array<F>,
    count: usize,
    h: usize,
    kx: usize,
    cols: usize,
    lanes: usize,
    span: usize,
    #[comptime] has_extra: bool,
) {
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    let zero = Vector::<f32, N>::new(0.0_f32);
    for pos in start..end {
        let row = pos / cols;
        let c = pos % cols;
        if c < count {
            if f32::cast_from(legal[row * count + c]) != 0.0 {
                let mut acc = zero;
                for i in 0..h {
                    let pre = Vector::<f32, N>::cast_from(k[(row * count + c) * h + i])
                        + Vector::<f32, N>::cast_from(q[row * h + i]);
                    let relu = select_many(pre.less_than(zero), zero, pre);
                    acc += Vector::<f32, N>::cast_from(v[i]) * relu;
                }
                let mut total = acc[0];
                #[unroll]
                for lane in 1..N::value() {
                    total += acc[lane];
                }
                logits[pos] = F::cast_from(total);
            } else {
                logits[pos] = F::min_value();
            }
        } else {
            if comptime!(has_extra) {
                logits[pos] = extra[row * kx + (c - count)];
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
    let line = line_dividing::<R, E>(k.client(), &[h], false);
    let (cube_count, cube_dim, span) = launch_1d_spans(k.client(), lanes, h);
    unsafe {
        pointer_additive_kernel::launch_unchecked::<E, R>(
            k.client(),
            cube_count,
            cube_dim,
            line,
            k.arg(),
            q.arg(),
            v.arg(),
            legal.arg(),
            extra_buf.arg(),
            logits.arg(),
            n,
            h / line,
            kx,
            cols,
            lanes,
            span,
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
fn pointer_additive_dk_kernel<F: Float + CubeElement, N: Size>(
    grad: &Array<F>,
    k: &Array<Vector<F, N>>,
    q: &Array<Vector<F, N>>,
    v: &Array<Vector<F, N>>,
    legal: &Array<F>,
    d_k: &mut Array<Vector<F, N>>,
    d_extra: &mut Array<F>,
    count: usize,
    h: usize,
    kx: usize,
    cols: usize,
    dk_lanes: usize,
    lanes: usize,
    span: usize,
    #[comptime] has_extra: bool,
) {
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    let zero = Vector::<f32, N>::new(0.0_f32);
    let one = Vector::<f32, N>::new(1.0_f32);
    for pos in start..end {
        if pos < dk_lanes {
            let hh = pos % h;
            let nn = (pos / h) % count;
            let row = pos / (h * count);
            let pre = Vector::<f32, N>::cast_from(k[(row * count + nn) * h + hh])
                + Vector::<f32, N>::cast_from(q[row * h + hh]);
            let gate = select_many(pre.greater_than(zero), one, zero);
            let g = f32::cast_from(grad[row * cols + nn]) * f32::cast_from(legal[row * count + nn]);
            let val = Vector::<f32, N>::new(g) * Vector::<f32, N>::cast_from(v[hh]) * gate;
            d_k[pos] = Vector::<F, N>::cast_from(val);
        } else {
            if comptime!(has_extra) {
                let epos = pos - dk_lanes;
                let kk = epos % kx;
                let row = epos / kx;
                d_extra[epos] = grad[row * cols + count + kk];
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
    let line = line_dividing::<R, E>(grad.client(), &[h], false);
    let dk_lanes = rows * n * h / line;
    let lanes = dk_lanes + rows * kx;
    if lanes == 0 {
        return Ok((d_k, d_extra));
    }
    let (cube_count, cube_dim, span) = launch_1d_spans(grad.client(), lanes, line);
    unsafe {
        pointer_additive_dk_kernel::launch_unchecked::<E, R>(
            grad.client(),
            cube_count,
            cube_dim,
            line,
            grad.arg(),
            k.arg(),
            q.arg(),
            v.arg(),
            legal.arg(),
            d_k.arg(),
            d_extra.arg(),
            n,
            h / line,
            kx,
            n + kx,
            dk_lanes,
            lanes,
            span,
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
fn pointer_additive_dq_kernel<F: Float + CubeElement, N: Size>(
    grad: &Array<F>,
    k: &Array<Vector<F, N>>,
    q: &Array<Vector<F, N>>,
    v: &Array<Vector<F, N>>,
    legal: &Array<F>,
    d_q: &mut Array<Vector<F, N>>,
    dv_partial: &mut Array<Vector<F, N>>,
    count: usize,
    h: usize,
    cols: usize,
    lanes: usize,
    span: usize,
) {
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    let zero = Vector::<f32, N>::new(0.0_f32);
    for pos in start..end {
        let hh = pos % h;
        let row = pos / h;
        let vh = Vector::<f32, N>::cast_from(v[hh]);
        let qh = Vector::<f32, N>::cast_from(q[row * h + hh]);
        let mut dq = zero;
        let mut dv = zero;
        for nn in 0..count {
            let g = Vector::<f32, N>::new(
                f32::cast_from(grad[row * cols + nn]) * f32::cast_from(legal[row * count + nn]),
            );
            let pre = Vector::<f32, N>::cast_from(k[(row * count + nn) * h + hh]) + qh;
            let open = pre.greater_than(zero);
            dq += select_many(open, g * vh, zero);
            dv += select_many(open, g * pre, zero);
        }
        d_q[pos] = Vector::<F, N>::cast_from(dq);
        dv_partial[pos] = Vector::<F, N>::cast_from(dv);
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
    let line = line_dividing::<R, E>(grad.client(), &[h], false);
    let lanes = rows * h / line;
    if lanes == 0 {
        return Ok((d_q, dv_partial));
    }
    let (cube_count, cube_dim, span) = launch_1d_spans(grad.client(), lanes, line * (n + 1));
    unsafe {
        pointer_additive_dq_kernel::launch_unchecked::<E, R>(
            grad.client(),
            cube_count,
            cube_dim,
            line,
            grad.arg(),
            k.arg(),
            q.arg(),
            v.arg(),
            legal.arg(),
            d_q.arg(),
            dv_partial.arg(),
            n,
            h / line,
            n + kx,
            lanes,
            span,
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
fn pointer_dot_kernel<F: Float + CubeElement, N: Size>(
    e: &Array<Vector<F, N>>,
    qd: &Array<Vector<F, N>>,
    legal: &Array<F>,
    extra: &Array<F>,
    logits: &mut Array<F>,
    count: usize,
    d: usize,
    kx: usize,
    cols: usize,
    lanes: usize,
    span: usize,
    #[comptime] has_extra: bool,
) {
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    for pos in start..end {
        let row = pos / cols;
        let c = pos % cols;
        if c < count {
            if f32::cast_from(legal[row * count + c]) != 0.0 {
                let mut acc = Vector::<f32, N>::new(0.0_f32);
                for i in 0..d {
                    acc += Vector::<f32, N>::cast_from(e[(row * count + c) * d + i])
                        * Vector::<f32, N>::cast_from(qd[row * d + i]);
                }
                let mut total = acc[0];
                #[unroll]
                for lane in 1..N::value() {
                    total += acc[lane];
                }
                logits[pos] = F::cast_from(total);
            } else {
                logits[pos] = F::min_value();
            }
        } else {
            if comptime!(has_extra) {
                logits[pos] = extra[row * kx + (c - count)];
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
    let line = line_dividing::<R, E>(e.client(), &[d], false);
    let (cube_count, cube_dim, span) = launch_1d_spans(e.client(), lanes, d);
    unsafe {
        pointer_dot_kernel::launch_unchecked::<E, R>(
            e.client(),
            cube_count,
            cube_dim,
            line,
            e.arg(),
            qd.arg(),
            legal.arg(),
            extra_buf.arg(),
            logits.arg(),
            n,
            d / line,
            kx,
            cols,
            lanes,
            span,
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
fn pointer_dot_de_kernel<F: Float + CubeElement, N: Size>(
    grad: &Array<F>,
    qd: &Array<Vector<F, N>>,
    legal: &Array<F>,
    d_e: &mut Array<Vector<F, N>>,
    d_extra: &mut Array<F>,
    count: usize,
    d: usize,
    kx: usize,
    cols: usize,
    de_lanes: usize,
    lanes: usize,
    span: usize,
    #[comptime] has_extra: bool,
) {
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    for pos in start..end {
        if pos < de_lanes {
            let dd = pos % d;
            let nn = (pos / d) % count;
            let row = pos / (d * count);
            let g = f32::cast_from(grad[row * cols + nn]) * f32::cast_from(legal[row * count + nn]);
            let val = Vector::<f32, N>::new(g) * Vector::<f32, N>::cast_from(qd[row * d + dd]);
            d_e[pos] = Vector::<F, N>::cast_from(val);
        } else {
            if comptime!(has_extra) {
                let epos = pos - de_lanes;
                let kk = epos % kx;
                let row = epos / kx;
                d_extra[epos] = grad[row * cols + count + kk];
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
    let line = line_dividing::<R, E>(grad.client(), &[d], false);
    let de_lanes = rows * n * d / line;
    let lanes = de_lanes + rows * kx;
    if lanes == 0 {
        return Ok((d_e, d_extra));
    }
    let (cube_count, cube_dim, span) = launch_1d_spans(grad.client(), lanes, line);
    unsafe {
        pointer_dot_de_kernel::launch_unchecked::<E, R>(
            grad.client(),
            cube_count,
            cube_dim,
            line,
            grad.arg(),
            qd.arg(),
            legal.arg(),
            d_e.arg(),
            d_extra.arg(),
            n,
            d / line,
            kx,
            n + kx,
            de_lanes,
            lanes,
            span,
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
fn pointer_dot_dqd_kernel<F: Float + CubeElement, N: Size>(
    grad: &Array<F>,
    e: &Array<Vector<F, N>>,
    legal: &Array<F>,
    d_qd: &mut Array<Vector<F, N>>,
    count: usize,
    d: usize,
    cols: usize,
    lanes: usize,
    span: usize,
) {
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    for pos in start..end {
        let dd = pos % d;
        let row = pos / d;
        let mut acc = Vector::<f32, N>::new(0.0_f32);
        for nn in 0..count {
            let g = f32::cast_from(grad[row * cols + nn]) * f32::cast_from(legal[row * count + nn]);
            acc += Vector::<f32, N>::new(g)
                * Vector::<f32, N>::cast_from(e[(row * count + nn) * d + dd]);
        }
        d_qd[pos] = Vector::<F, N>::cast_from(acc);
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
    let line = line_dividing::<R, E>(grad.client(), &[d], false);
    let lanes = rows * d / line;
    if lanes == 0 {
        return Ok(d_qd);
    }
    let (cube_count, cube_dim, span) = launch_1d_spans(grad.client(), lanes, line * (n + 1));
    unsafe {
        pointer_dot_dqd_kernel::launch_unchecked::<E, R>(
            grad.client(),
            cube_count,
            cube_dim,
            line,
            grad.arg(),
            e.arg(),
            legal.arg(),
            d_qd.arg(),
            n,
            d / line,
            n + kx,
            lanes,
            span,
        );
    }
    Ok(d_qd)
}

// ---------------------------------------------------------------------------
// K4: bias_relu — the encoder's hidden layers.
// ---------------------------------------------------------------------------

/// One lane per vector: `out[i] = max(0, pre[i] + bias[i % h])`.
///
/// The composed path this replaces is, per hidden layer of
/// [`crate::nn::entity::EntityEncoder`], the bias broadcast add inside
/// [`crate::nn::linear::Linear::apply`] and the `relu` that follows it: two
/// launches down to one. The bias is indexed by a plain modulo of the flat
/// position rather than through the broadcast-metadata path, so this kernel
/// uploads no shape/stride buffer at all. It reads and writes `Vector`s as
/// wide as `h` allows (the bias index is then `i % (h / N)` in vectors).
///
/// Accumulation is in `f32` whatever `E` is, so narrow storage does not
/// change the rounding of the sum.
#[cube(launch_unchecked)]
fn bias_relu_kernel<F: Float + CubeElement, N: Size>(
    pre: &Array<Vector<F, N>>,
    bias: &Array<Vector<F, N>>,
    out: &mut Array<Vector<F, N>>,
    h: usize,
    lanes: usize,
    span: usize,
) {
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    let zero = Vector::<f32, N>::new(0.0_f32);
    for pos in start..end {
        let v = Vector::<f32, N>::cast_from(pre[pos]) + Vector::<f32, N>::cast_from(bias[pos % h]);
        out[pos] = Vector::<F, N>::cast_from(select_many(v.less_than(zero), zero, v));
    }
}

/// Adjoint of [`bias_relu_kernel`] with respect to `pre`: a flat gather of
/// the saved output.
///
/// One lane per vector: `d_pre[i] = g[i] * (y[i] > 0 ? 1 : 0)`. The gate
/// reads the saved *output* `y`, which is exactly the strict `> 0` test the
/// composed `relu` rule applies to its input (`gt_scalar(&x, 0.0)`): `y > 0`
/// iff `pre + bias > 0`, so at `y == 0` the gradient is 0 on both paths.
/// The bias gradient needs no second kernel: it is this same `d_pre` summed
/// over every axis but the last, which the caller does with the existing
/// `reduce_grad_to` — exactly what [`crate::autograd::Var::add`]'s rule does
/// for a broadcast bias, so the gradient is identical.
#[cube(launch_unchecked)]
fn bias_relu_backward_kernel<F: Float + CubeElement, N: Size>(
    grad: &Array<Vector<F, N>>,
    y: &Array<Vector<F, N>>,
    d_pre: &mut Array<Vector<F, N>>,
    lanes: usize,
    span: usize,
) {
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    let zero = Vector::<f32, N>::new(0.0_f32);
    let one = Vector::<f32, N>::new(1.0_f32);
    for pos in start..end {
        let gate = select_many(
            Vector::<f32, N>::cast_from(y[pos]).greater_than(zero),
            one,
            zero,
        );
        d_pre[pos] = Vector::<F, N>::cast_from(Vector::<f32, N>::cast_from(grad[pos]) * gate);
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
    if out.is_empty() {
        return Ok(out);
    }
    let line = line_dividing::<R, E>(pre.client(), &[h], false);
    let lanes = out.len() / line;
    let (cube_count, cube_dim, span) = launch_1d_spans(pre.client(), lanes, line);
    unsafe {
        bias_relu_kernel::launch_unchecked::<E, R>(
            pre.client(),
            cube_count,
            cube_dim,
            line,
            pre.arg(),
            bias.arg(),
            out.arg(),
            h / line,
            lanes,
            span,
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
    if d_pre.is_empty() {
        return Ok(d_pre);
    }
    let line = line_dividing::<R, E>(grad.client(), &[d_pre.len()], false);
    let lanes = d_pre.len() / line;
    let (cube_count, cube_dim, span) = launch_1d_spans(grad.client(), lanes, line);
    unsafe {
        bias_relu_backward_kernel::launch_unchecked::<E, R>(
            grad.client(),
            cube_count,
            cube_dim,
            line,
            grad.arg(),
            y.arg(),
            d_pre.arg(),
            lanes,
            span,
        );
    }
    Ok(d_pre)
}
