//! Kernels that exist only to collapse a chain of elementwise ops into one launch.
//!
//! Everything here is expressible with [`crate::tensor::ops::elemwise`]. The reason
//! it is not written that way is arithmetic: a launch costs the same whether it
//! touches ten elements or ten thousand, so a twelve-op update over a small
//! parameter is eleven launches of pure overhead. Fusing pays exactly where the
//! tensors are small and the op chain is long, which is the optimizer.

use cubecl::prelude::*;

use crate::backend::{FloatElem, launch_1d, line_size_for};
use crate::error::{Error, Result};
use crate::tensor::base::Tensor;
use crate::tensor::ops::index::IdTensor;
use crate::tensor::shape::{MAX_RANK, Shape};

/// One AdamW update: moments, bias correction, decoupled decay and the step.
///
/// `m` and `v` are updated in place; the new parameter value goes to `out`.
#[cube(launch_unchecked)]
fn adamw_kernel<F: Float + CubeElement, N: Size>(
    param: &Array<Vector<F, N>>,
    grad: &Array<Vector<F, N>>,
    m: &mut Array<Vector<F, N>>,
    v: &mut Array<Vector<F, N>>,
    out: &mut Array<Vector<F, N>>,
    scale: &Array<F>,
    beta1: F,
    beta2: F,
    one_minus_beta1: F,
    one_minus_beta2: F,
    inv_bias1: F,
    inv_bias2: F,
    eps: F,
    lr: F,
    decay: F,
) {
    if ABSOLUTE_POS < out.len() {
        // The gradient scale — micro-batch averaging times the gradient-norm clip —
        // is read from the device rather than passed as a scalar. That is the whole
        // point of it being a buffer: computing the clip factor on the host means
        // reading the gradient norm back, which drains the pipeline in the middle of
        // every step. Here nothing is ever read back and the step stays asynchronous.
        let g = grad[ABSOLUTE_POS] * Vector::<F, N>::new(scale[0]);
        let m_next = Vector::<F, N>::new(beta1) * m[ABSOLUTE_POS]
            + Vector::<F, N>::new(one_minus_beta1) * g;
        let v_next = Vector::<F, N>::new(beta2) * v[ABSOLUTE_POS]
            + Vector::<F, N>::new(one_minus_beta2) * (g * g);
        m[ABSOLUTE_POS] = m_next;
        v[ABSOLUTE_POS] = v_next;

        let m_hat = m_next * Vector::<F, N>::new(inv_bias1);
        let v_hat = v_next * Vector::<F, N>::new(inv_bias2);
        let denom = v_hat.sqrt() + Vector::<F, N>::new(eps);
        let p = param[ABSOLUTE_POS];
        let update = m_hat / denom + Vector::<F, N>::new(decay) * p;
        out[ABSOLUTE_POS] = p - Vector::<F, N>::new(lr) * update;
    }
}

/// Scalars for one [`adamw_step`] call.
#[derive(Debug, Clone, Copy)]
pub struct AdamWStep {
    /// Learning rate for this step.
    pub lr: f32,
    /// First moment decay.
    pub beta1: f32,
    /// Second moment decay.
    pub beta2: f32,
    /// Denominator floor.
    pub eps: f32,
    /// Decoupled weight decay, already resolved to `0.0` where it is skipped.
    pub decay: f32,
    /// `1 - beta1^t`.
    pub bias1: f32,
    /// `1 - beta2^t`.
    pub bias2: f32,
}

/// Apply one AdamW update, returning the new parameter value.
///
/// `m` and `v` are mutated in place: the optimizer owns them and hands them to
/// nobody, which is what makes that safe. The parameter itself is *not* mutated —
/// an autodiff graph retained from the previous step may still be holding it.
///
/// All four tensors must share a shape.
pub fn adamw_step<R: Runtime, E: FloatElem>(
    param: &Tensor<R, E>,
    grad: &Tensor<R, E>,
    m: &Tensor<R, E>,
    v: &Tensor<R, E>,
    scale: &Tensor<R, E>,
    step: AdamWStep,
) -> Tensor<R, E> {
    debug_assert_eq!(param.shape(), grad.shape());
    debug_assert_eq!(param.shape(), m.shape());
    debug_assert_eq!(param.shape(), v.shape());

    let out = Tensor::empty(param.shape().clone(), param.device());
    let n = out.len();
    if n == 0 {
        return out;
    }
    let line = line_size_for::<R, E>(param.client(), n);
    let (count, dim) = launch_1d(param.client(), n / line, line);
    unsafe {
        adamw_kernel::launch_unchecked::<E, R>(
            param.client(),
            count,
            dim,
            line,
            param.arg(),
            grad.arg(),
            m.arg(),
            v.arg(),
            out.arg(),
            scale.arg(),
            E::from_scalar(step.beta1),
            E::from_scalar(step.beta2),
            E::from_scalar(1.0 - step.beta1),
            E::from_scalar(1.0 - step.beta2),
            E::from_scalar(1.0 / step.bias1),
            E::from_scalar(1.0 / step.bias2),
            E::from_scalar(step.eps),
            E::from_scalar(step.lr),
            E::from_scalar(step.decay),
        );
    }
    out
}

// ---------------------------------------------------------------------------
// RMS normalisation
// ---------------------------------------------------------------------------

/// `y = x * rsqrt(mean(x^2) + eps) * w`, one unit per row.
///
/// The reciprocal scale is written out alongside `y` because the backward pass needs
/// it and recomputing it there would mean a second reduction over the same row.
#[cube(launch_unchecked)]
fn rms_norm_kernel<F: Float + CubeElement, N: Size>(
    input: &Array<Vector<F, N>>,
    weight: &Array<Vector<F, N>>,
    output: &mut Array<Vector<F, N>>,
    rscale: &mut Array<F>,
    eps: F,
    inv_dim: F,
    dim_lines: usize,
    rows: usize,
    #[comptime] has_weight: bool,
) {
    if ABSOLUTE_POS < rows {
        let base = ABSOLUTE_POS * dim_lines;

        let head = input[base];
        let mut squares = head * head;
        for i in 1..dim_lines {
            let v = input[base + i];
            squares += v * v;
        }
        let mut total = squares[0];
        #[unroll]
        for lane in 1..N::value() {
            total += squares[lane];
        }

        let scale = F::new(1.0_f32) / (total * inv_dim + eps).sqrt();
        rscale[ABSOLUTE_POS] = scale;
        let scale_v = Vector::<F, N>::new(scale);

        for i in 0..dim_lines {
            let normed = input[base + i] * scale_v;
            if comptime!(has_weight) {
                output[base + i] = normed * weight[i];
            } else {
                output[base + i] = normed;
            }
        }
    }
}

/// Planes per cube for the row-wise kernels.
///
/// Enough that a cube is worth dispatching, few enough that a short-rowed tensor
/// still spreads over many compute units.
const ROW_PLANES: usize = 4;

/// Launch geometry that gives each of `rows` rows one hardware plane, or `None` on a
/// runtime with no planes to give — CubeCL's CPU backend, where a "unit" is a thread
/// and the per-unit kernel is the right shape after all.
///
/// `ABSOLUTE_POS / PLANE_DIM` is the row, which stays correct however the launcher
/// folds the grid, so the cube count can go through the usual helper.
fn plane_per_row<R: Runtime>(
    client: &ComputeClient<R>,
    rows: usize,
) -> Option<(CubeCount, CubeDim)> {
    let plane = client.properties().hardware.plane_size_max as usize;
    if plane <= 1 {
        return None;
    }
    let cube_dim = CubeDim::new_1d((plane * ROW_PLANES) as u32);
    let count = crate::backend::cube_count_for(rows * plane, cube_dim.num_elems());
    crate::backend::count_launch();
    Some((count, cube_dim))
}

/// [`rms_norm_kernel`] with one *plane* per row instead of one unit.
///
/// One unit per row is the wrong shape on a GPU, and expensively so: neighbouring
/// units then read addresses a whole row apart, so a wave's load instruction touches
/// as many cache lines as it has lanes and uses a fraction of each. Measured on an
/// RDNA3.5 iGPU, the per-unit kernel moved 6.3 GB/s against 41 GB/s for a plain
/// elementwise pass over the same tensors.
///
/// Giving each row a plane and striding it by `PLANE_DIM` makes every load contiguous
/// across the lanes that issue it, and the row sum becomes one `plane_sum` rather
/// than a serial loop. Only the lane-zero write of the scale is divergent; the
/// reduction itself is reached by every lane, which is what a plane operation
/// requires.
#[cube(launch_unchecked)]
fn rms_norm_plane_kernel<F: Float + CubeElement, N: Size>(
    input: &Array<Vector<F, N>>,
    weight: &Array<Vector<F, N>>,
    output: &mut Array<Vector<F, N>>,
    rscale: &mut Array<F>,
    eps: F,
    inv_dim: F,
    dim_lines: usize,
    rows: usize,
    #[comptime] has_weight: bool,
) {
    let width = PLANE_DIM as usize;
    let lane = UNIT_POS_PLANE as usize;
    let row = ABSOLUTE_POS / width;
    let live = row < rows;
    let base = select(live, row, 0) * dim_lines;
    let steps = dim_lines.div_ceil(width);

    let mut squares = Vector::<F, N>::new(F::new(0.0_f32));
    for s in 0..steps {
        let i = lane + s * width;
        if i < dim_lines {
            let v = input[base + i];
            squares += v * v;
        }
    }
    let mut total = squares[0];
    #[unroll]
    for l in 1..N::value() {
        total += squares[l];
    }
    let row_total = plane_sum(total);

    let scale = F::new(1.0_f32) / (row_total * inv_dim + eps).sqrt();
    if live && lane == 0 {
        rscale[row] = scale;
    }
    let scale_v = Vector::<F, N>::new(scale);
    for s in 0..steps {
        let i = lane + s * width;
        if i < dim_lines && live {
            let normed = input[base + i] * scale_v;
            if comptime!(has_weight) {
                output[base + i] = normed * weight[i];
            } else {
                output[base + i] = normed;
            }
        }
    }
}

/// [`rms_norm_backward_kernel`] with one plane per row, for the reason above.
#[cube(launch_unchecked)]
fn rms_norm_backward_plane_kernel<F: Float + CubeElement, N: Size>(
    grad: &Array<Vector<F, N>>,
    input: &Array<Vector<F, N>>,
    weight: &Array<Vector<F, N>>,
    rscale: &Array<F>,
    dx: &mut Array<Vector<F, N>>,
    dw_partial: &mut Array<Vector<F, N>>,
    inv_dim: F,
    dim_lines: usize,
    rows: usize,
    #[comptime] has_weight: bool,
) {
    let width = PLANE_DIM as usize;
    let lane = UNIT_POS_PLANE as usize;
    let row = ABSOLUTE_POS / width;
    let live = row < rows;
    let safe_row = select(live, row, 0);
    let base = safe_row * dim_lines;
    let steps = dim_lines.div_ceil(width);
    let scale = rscale[safe_row];

    let mut dots = Vector::<F, N>::new(F::new(0.0_f32));
    for s in 0..steps {
        let i = lane + s * width;
        if i < dim_lines {
            let gw = if comptime!(has_weight) {
                grad[base + i] * weight[i]
            } else {
                grad[base + i]
            };
            dots += gw * input[base + i];
        }
    }
    let mut dot = dots[0];
    #[unroll]
    for l in 1..N::value() {
        dot += dots[l];
    }
    let row_dot = plane_sum(dot);

    let scale_v = Vector::<F, N>::new(scale);
    let pull = Vector::<F, N>::new(scale * scale * scale * row_dot * inv_dim);
    for s in 0..steps {
        let i = lane + s * width;
        if i < dim_lines && live {
            let x = input[base + i];
            let g = grad[base + i];
            let gw = if comptime!(has_weight) { g * weight[i] } else { g };
            dx[base + i] = gw * scale_v - x * pull;
            if comptime!(has_weight) {
                dw_partial[base + i] = g * x * scale_v;
            }
        }
    }
}

/// The adjoint of [`rms_norm_kernel`].
///
/// `dx_j = r * g_j * w_j - (r^3 / dim) * x_j * sum_i(g_i * w_i * x_i)`, and the gain
/// gradient is `sum over rows of g * x * r` — left here as a per-row partial that the
/// caller reduces, because a cross-row sum inside the kernel would need atomics.
#[cube(launch_unchecked)]
fn rms_norm_backward_kernel<F: Float + CubeElement, N: Size>(
    grad: &Array<Vector<F, N>>,
    input: &Array<Vector<F, N>>,
    weight: &Array<Vector<F, N>>,
    rscale: &Array<F>,
    dx: &mut Array<Vector<F, N>>,
    dw_partial: &mut Array<Vector<F, N>>,
    inv_dim: F,
    dim_lines: usize,
    rows: usize,
    #[comptime] has_weight: bool,
) {
    if ABSOLUTE_POS < rows {
        let base = ABSOLUTE_POS * dim_lines;
        let scale = rscale[ABSOLUTE_POS];

        let head = if comptime!(has_weight) {
            grad[base] * weight[0] * input[base]
        } else {
            grad[base] * input[base]
        };
        let mut dots = head;
        for i in 1..dim_lines {
            let gw = if comptime!(has_weight) {
                grad[base + i] * weight[i]
            } else {
                grad[base + i]
            };
            dots += gw * input[base + i];
        }
        let mut dot = dots[0];
        #[unroll]
        for lane in 1..N::value() {
            dot += dots[lane];
        }

        let scale_v = Vector::<F, N>::new(scale);
        let pull = Vector::<F, N>::new(scale * scale * scale * dot * inv_dim);
        for i in 0..dim_lines {
            let x = input[base + i];
            let g = grad[base + i];
            let gw = if comptime!(has_weight) { g * weight[i] } else { g };
            dx[base + i] = gw * scale_v - x * pull;
            if comptime!(has_weight) {
                dw_partial[base + i] = g * x * scale_v;
            }
        }
    }
}

/// Fused RMS normalisation over the trailing axis.
///
/// Returns the normalised tensor and the per-row reciprocal scale, which
/// [`rms_norm_backward`] needs.
pub fn rms_norm<R: Runtime, E: FloatElem>(
    input: &Tensor<R, E>,
    weight: Option<&Tensor<R, E>>,
    eps: f32,
) -> Result<(Tensor<R, E>, Tensor<R, E>)> {
    let dim = input.shape().dim_from_end(0);
    if let Some(w) = weight
        && w.len() != dim
    {
        return Err(Error::shape(format!(
            "RMS norm gain of {} does not match a trailing axis of {dim}",
            w.len()
        )));
    }
    let rows = input.len() / dim.max(1);
    let out = Tensor::empty(input.shape().clone(), input.device());
    let scale = Tensor::empty(Shape::new(vec![rows.max(1)]), input.device());
    if rows == 0 || dim == 0 {
        return Ok((out, scale));
    }

    let line = line_size_for::<R, E>(input.client(), dim);
    let placeholder = Tensor::<R, E>::empty(Shape::new(vec![line]), input.device());
    let gain = weight.unwrap_or(&placeholder);
    if let Some((count, cube_dim)) = plane_per_row::<R>(input.client(), rows) {
        unsafe {
            rms_norm_plane_kernel::launch_unchecked::<E, R>(
                input.client(),
                count,
                cube_dim,
                line,
                input.arg(),
                gain.arg(),
                out.arg(),
                scale.arg(),
                E::from_scalar(eps),
                E::from_scalar(1.0 / dim as f32),
                dim / line,
                rows,
                weight.is_some(),
            );
        }
        return Ok((out, scale));
    }
    let (count, cube_dim) = launch_1d(input.client(), rows, dim);
    unsafe {
        rms_norm_kernel::launch_unchecked::<E, R>(
            input.client(),
            count,
            cube_dim,
            line,
            input.arg(),
            gain.arg(),
            out.arg(),
            scale.arg(),
            E::from_scalar(eps),
            E::from_scalar(1.0 / dim as f32),
            dim / line,
            rows,
            weight.is_some(),
        );
    }
    Ok((out, scale))
}

/// The adjoint of [`rms_norm`]: the input gradient, and the gain gradient when the
/// layer has a gain.
#[allow(clippy::type_complexity)] // Two gradients, one of them optional; naming it would not help.
pub fn rms_norm_backward<R: Runtime, E: FloatElem>(
    grad: &Tensor<R, E>,
    input: &Tensor<R, E>,
    weight: Option<&Tensor<R, E>>,
    scale: &Tensor<R, E>,
) -> Result<(Tensor<R, E>, Option<Tensor<R, E>>)> {
    let dim = input.shape().dim_from_end(0);
    let rows = input.len() / dim.max(1);
    let dx = Tensor::empty(input.shape().clone(), input.device());
    if rows == 0 || dim == 0 {
        return Ok((dx, weight.map(|w| Tensor::zeros(w.shape().clone(), w.device()))));
    }

    let line = line_size_for::<R, E>(input.client(), dim);
    let placeholder = Tensor::<R, E>::empty(Shape::new(vec![line]), input.device());
    let gain = weight.unwrap_or(&placeholder);
    // Only bound, never written, when there is no gain.
    let dw_partial = match weight {
        Some(_) => Tensor::<R, E>::empty(input.shape().clone(), input.device()),
        None => placeholder.clone(),
    };
    if let Some((count, cube_dim)) = plane_per_row::<R>(input.client(), rows) {
        unsafe {
            rms_norm_backward_plane_kernel::launch_unchecked::<E, R>(
                input.client(),
                count,
                cube_dim,
                line,
                grad.arg(),
                input.arg(),
                gain.arg(),
                scale.arg(),
                dx.arg(),
                dw_partial.arg(),
                E::from_scalar(1.0 / dim as f32),
                dim / line,
                rows,
                weight.is_some(),
            );
        }
    } else {
        let (count, cube_dim) = launch_1d(input.client(), rows, dim);
        unsafe {
            rms_norm_backward_kernel::launch_unchecked::<E, R>(
                input.client(),
                count,
                cube_dim,
                line,
                grad.arg(),
                input.arg(),
                gain.arg(),
                scale.arg(),
                dx.arg(),
                dw_partial.arg(),
                E::from_scalar(1.0 / dim as f32),
                dim / line,
                rows,
                weight.is_some(),
            );
        }
    }
    let dw = match weight {
        Some(w) => {
            let flat = dw_partial.reshape(Shape::new(vec![rows, dim]))?;
            Some(
                crate::tensor::ops::reduce::sum_dim(&flat, 0)?
                    .reshape(w.shape().clone())?,
            )
        }
        None => None,
    };
    Ok((dx, dw))
}

// ---------------------------------------------------------------------------
// Half-split rotation (RoPE, and Mamba-3's rotating state frame)
// ---------------------------------------------------------------------------

/// How a `[..., half]` table of angles lines up with a `[..., 2 * half]` tensor.
///
/// Both callers broadcast, and in opposite directions: RoPE shares one table across
/// batch and heads (`[1, 1, seq, half]` against `[b, h, seq, half]`), while the
/// Mamba-3 scan shares one angle across the state rank (`[b, h, 1, half]` against
/// `[b, h, rank, half]`). Rather than pay the general broadcast machinery for either,
/// both collapse to `cos_row = (row / repeat) % rows`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RotationLayout {
    /// Rows of the angle table.
    pub rows: usize,
    /// Consecutive tensor rows that share one table row.
    pub repeat: usize,
}

impl RotationLayout {
    /// Work out the layout, or `None` when the broadcast is not of that shape and
    /// the caller should materialise the table instead.
    ///
    /// A size-1 axis is absorbed into `repeat` while it is trailing, and into `rows`
    /// while it is leading — a leading axis contributes a whole multiple of `rows` to
    /// the row index, so the modulo removes it exactly. A size-1 axis between two
    /// matching axes fits neither and gives `None`.
    pub fn resolve(tensor: &[usize], table: &[usize]) -> Option<Self> {
        if tensor.len() != table.len() {
            return None;
        }
        let lead = tensor.len().saturating_sub(1);
        let mut repeat = 1;
        let mut end = lead;
        while end > 0 && table[end - 1] == 1 {
            repeat *= tensor[end - 1];
            end -= 1;
        }
        let mut rows = 1;
        let mut broadcast_allowed = true;
        for axis in (0..end).rev() {
            if table[axis] == tensor[axis] {
                broadcast_allowed = false;
                rows *= table[axis];
            } else if table[axis] == 1 && broadcast_allowed {
                // A leading broadcast axis: its stride is a multiple of `rows`.
            } else {
                return None;
            }
        }
        Some(Self { rows, repeat })
    }
}

/// `out[i] = x1 * cos - x2 * sin`, `out[half + i] = x1 * sin + x2 * cos`.
#[cube(launch_unchecked)]
fn rotate_halves_kernel<F: Float + CubeElement, N: Size>(
    x: &Array<Vector<F, N>>,
    cos: &Array<Vector<F, N>>,
    sin: &Array<Vector<F, N>>,
    out: &mut Array<Vector<F, N>>,
    half_lines: usize,
    repeat: usize,
    table_rows: usize,
    lanes: usize,
    #[comptime] from_angle: bool,
) {
    if ABSOLUTE_POS < lanes {
        let i = ABSOLUTE_POS % half_lines;
        let row = ABSOLUTE_POS / half_lines;
        let base = row * half_lines * 2;
        let table = ((row / repeat) % table_rows) * half_lines + i;

        let x1 = x[base + i];
        let x2 = x[base + half_lines + i];
        let mut c = cos[table];
        let mut s = sin[table];
        if comptime!(from_angle) {
            // `cos` carries the angle itself; the pair is `(cos phi, -sin phi)`.
            let phi = cos[table];
            c = phi.cos();
            s = phi.sin() * Vector::<F, N>::new(F::new(-1.0_f32));
        }
        out[base + i] = x1 * c - x2 * s;
        out[base + half_lines + i] = x1 * s + x2 * c;
    }
}

/// The adjoint of [`rotate_halves_kernel`].
///
/// The rotation is linear in every argument, so all four gradients fall out of the
/// same two products. `dcos` and `dsin` come out at the tensor's row count; summing
/// them back down to the table's shape is the caller's job, and the autodiff layer
/// already owns that step.
#[cube(launch_unchecked)]
fn rotate_halves_backward_kernel<F: Float + CubeElement, N: Size>(
    grad: &Array<Vector<F, N>>,
    x: &Array<Vector<F, N>>,
    cos: &Array<Vector<F, N>>,
    sin: &Array<Vector<F, N>>,
    dx: &mut Array<Vector<F, N>>,
    dcos: &mut Array<Vector<F, N>>,
    dsin: &mut Array<Vector<F, N>>,
    half_lines: usize,
    repeat: usize,
    table_rows: usize,
    lanes: usize,
    #[comptime] from_angle: bool,
) {
    if ABSOLUTE_POS < lanes {
        let i = ABSOLUTE_POS % half_lines;
        let row = ABSOLUTE_POS / half_lines;
        let base = row * half_lines * 2;
        let table = ((row / repeat) % table_rows) * half_lines + i;

        let g1 = grad[base + i];
        let g2 = grad[base + half_lines + i];
        let x1 = x[base + i];
        let x2 = x[base + half_lines + i];
        let mut c = cos[table];
        let mut s = sin[table];
        if comptime!(from_angle) {
            let phi = cos[table];
            c = phi.cos();
            s = phi.sin() * Vector::<F, N>::new(F::new(-1.0_f32));
        }

        dx[base + i] = g1 * c + g2 * s;
        dx[base + half_lines + i] = g2 * c - g1 * s;
        if comptime!(from_angle) {
            // dphi = dcos * d(cos phi)/dphi + dsin * d(-sin phi)/dphi
            //      = (g1 x1 + g2 x2)(-sin phi) + (g2 x1 - g1 x2)(-cos phi)
            dcos[ABSOLUTE_POS] =
                (g1 * x1 + g2 * x2) * s - (g2 * x1 - g1 * x2) * c;
        } else {
            dcos[ABSOLUTE_POS] = g1 * x1 + g2 * x2;
            dsin[ABSOLUTE_POS] = g2 * x1 - g1 * x2;
        }
    }
}

/// Geometry shared by the rotation forward and backward launches.
fn rotation_launch<R: Runtime, E: FloatElem>(
    x: &Tensor<R, E>,
    layout: RotationLayout,
) -> (usize, usize, usize, CubeCount, CubeDim) {
    let half = x.shape().dim_from_end(0) / 2;
    let rows = x.len() / (2 * half).max(1);
    let line = line_size_for::<R, E>(x.client(), half);
    let lanes = rows * (half / line);
    let (count, dim) = launch_1d(x.client(), lanes, line);
    let _ = layout;
    (line, half / line, lanes, count, dim)
}

/// Rotate the two halves of the trailing axis of `x` by the angles in `cos`/`sin`.
///
/// Written out of primitives this is two slices, four multiplies, an add, a subtract
/// and a concatenation — ten launches, and the Mamba-3 step runs it twice per token.
pub fn rotate_halves<R: Runtime, E: FloatElem>(
    x: &Tensor<R, E>,
    cos: &Tensor<R, E>,
    sin: &Tensor<R, E>,
    layout: RotationLayout,
    from_angle: bool,
) -> Result<Tensor<R, E>> {
    let d = x.shape().dim_from_end(0);
    if !d.is_multiple_of(2) {
        return Err(Error::shape(format!(
            "rotation needs an even trailing dimension, got {d}"
        )));
    }
    let out = Tensor::empty(x.shape().clone(), x.device());
    if x.is_empty() {
        return Ok(out);
    }
    let (line, half_lines, lanes, count, dim) = rotation_launch::<R, E>(x, layout);
    unsafe {
        rotate_halves_kernel::launch_unchecked::<E, R>(
            x.client(),
            count,
            dim,
            line,
            x.arg(),
            cos.arg(),
            sin.arg(),
            out.arg(),
            half_lines,
            layout.repeat,
            layout.rows,
            lanes,
            from_angle,
        );
    }
    Ok(out)
}

/// The adjoint of [`rotate_halves`]: gradients for the tensor and for both tables,
/// the latter still at the tensor's row count.
///
/// In angle mode the whole table gradient lands in the first slot, because there is
/// only one table.
#[allow(clippy::type_complexity)] // Three gradients of the same type; naming them would not help.
pub fn rotate_halves_backward<R: Runtime, E: FloatElem>(
    grad: &Tensor<R, E>,
    x: &Tensor<R, E>,
    cos: &Tensor<R, E>,
    sin: &Tensor<R, E>,
    layout: RotationLayout,
    from_angle: bool,
) -> Result<(Tensor<R, E>, Tensor<R, E>, Tensor<R, E>)> {
    let half = x.shape().dim_from_end(0) / 2;
    let mut table_dims = x.dims().to_vec();
    let last = table_dims.len() - 1;
    table_dims[last] = half;
    let table_shape = Shape::new(table_dims);

    let dx = Tensor::empty(x.shape().clone(), x.device());
    let dcos = Tensor::empty(table_shape.clone(), x.device());
    let dsin = Tensor::empty(table_shape, x.device());
    if x.is_empty() {
        return Ok((dx, dcos, dsin));
    }
    let (line, half_lines, lanes, count, dim) = rotation_launch::<R, E>(x, layout);
    unsafe {
        rotate_halves_backward_kernel::launch_unchecked::<E, R>(
            x.client(),
            count,
            dim,
            line,
            grad.arg(),
            x.arg(),
            cos.arg(),
            sin.arg(),
            dx.arg(),
            dcos.arg(),
            dsin.arg(),
            half_lines,
            layout.repeat,
            layout.rows,
            lanes,
            from_angle,
        );
    }
    Ok((dx, dcos, dsin))
}

// ---------------------------------------------------------------------------
// Depthwise causal convolution
// ---------------------------------------------------------------------------

/// `out[b, s, c] = bias[c] + sum_j weight[j, c] * window[b, s + j, c]`.
///
/// The window is the history followed by the input, but it is never materialised:
/// the kernel picks whichever buffer the position falls in. Composed out of
/// primitives, one decoding step of a four-tap convolution is a concatenation, four
/// weight slices, three zero-padded shifts and seven multiply-adds — about twenty
/// launches to produce one token's worth of output, most of it thrown away.
#[cube(launch_unchecked)]
fn causal_conv1d_kernel<F: Float + CubeElement, N: Size>(
    input: &Array<Vector<F, N>>,
    history: &Array<Vector<F, N>>,
    weight: &Array<Vector<F, N>>,
    bias: &Array<Vector<F, N>>,
    out: &mut Array<Vector<F, N>>,
    channels: usize,
    seq: usize,
    carry: usize,
    taps: usize,
    lanes: usize,
    #[comptime] has_history: bool,
    #[comptime] has_bias: bool,
) {
    if ABSOLUTE_POS < lanes {
        let c = ABSOLUTE_POS % channels;
        let rest = ABSOLUTE_POS / channels;
        let s = rest % seq;
        let batch = rest / seq;

        let mut acc = Vector::<F, N>::new(F::new(0.0_f32));
        for j in 0..taps {
            let u = s + j;
            let mut v = Vector::<F, N>::new(F::new(0.0_f32));
            if u < carry {
                if comptime!(has_history) {
                    v = history[(batch * carry + u) * channels + c];
                }
            } else {
                v = input[(batch * seq + u - carry) * channels + c];
            }
            acc += weight[j * channels + c] * v;
        }
        if comptime!(has_bias) {
            acc += bias[c];
        }
        out[ABSOLUTE_POS] = acc;
    }
}

/// The last `carry` positions of the window, which become the next call's history.
#[cube(launch_unchecked)]
fn causal_conv1d_history_kernel<F: Float + CubeElement, N: Size>(
    input: &Array<Vector<F, N>>,
    history: &Array<Vector<F, N>>,
    out: &mut Array<Vector<F, N>>,
    channels: usize,
    seq: usize,
    carry: usize,
    lanes: usize,
    #[comptime] has_history: bool,
) {
    if ABSOLUTE_POS < lanes {
        let c = ABSOLUTE_POS % channels;
        let rest = ABSOLUTE_POS / channels;
        let i = rest % carry;
        let batch = rest / carry;

        let u = seq + i;
        let mut v = Vector::<F, N>::new(F::new(0.0_f32));
        if u < carry {
            if comptime!(has_history) {
                v = history[(batch * carry + u) * channels + c];
            }
        } else {
            v = input[(batch * seq + u - carry) * channels + c];
        }
        out[ABSOLUTE_POS] = v;
    }
}

/// The adjoint with respect to the window, written straight into the two buffers it
/// came from: `dwin[b, u, c] = sum_j weight[j, c] * grad[b, u - j, c]`.
#[cube(launch_unchecked)]
fn causal_conv1d_input_grad_kernel<F: Float + CubeElement, N: Size>(
    grad: &Array<Vector<F, N>>,
    weight: &Array<Vector<F, N>>,
    d_input: &mut Array<Vector<F, N>>,
    d_history: &mut Array<Vector<F, N>>,
    channels: usize,
    seq: usize,
    carry: usize,
    taps: usize,
    lanes: usize,
    #[comptime] has_history: bool,
) {
    if ABSOLUTE_POS < lanes {
        let c = ABSOLUTE_POS % channels;
        let rest = ABSOLUTE_POS / channels;
        let u = rest % (seq + carry);
        let batch = rest / (seq + carry);

        let mut acc = Vector::<F, N>::new(F::new(0.0_f32));
        for j in 0..taps {
            if u >= j && u - j < seq {
                acc += weight[j * channels + c] * grad[(batch * seq + u - j) * channels + c];
            }
        }
        if u < carry {
            if comptime!(has_history) {
                d_history[(batch * carry + u) * channels + c] = acc;
            }
        } else {
            d_input[(batch * seq + u - carry) * channels + c] = acc;
        }
    }
}

/// The adjoints of the taps and the bias, each unit summing one `(tap, channel)`
/// pair over the whole batch and window.
// The comptime guard has to stay outside the runtime one: collapsing them would
// make the whole test runtime and defeat the specialisation.
#[allow(clippy::collapsible_if)]
#[cube(launch_unchecked)]
fn causal_conv1d_weight_grad_kernel<F: Float + CubeElement, N: Size>(
    grad: &Array<Vector<F, N>>,
    input: &Array<Vector<F, N>>,
    history: &Array<Vector<F, N>>,
    d_weight: &mut Array<Vector<F, N>>,
    d_bias: &mut Array<Vector<F, N>>,
    channels: usize,
    seq: usize,
    carry: usize,
    batches: usize,
    lanes: usize,
    #[comptime] has_history: bool,
    #[comptime] has_bias: bool,
) {
    if ABSOLUTE_POS < lanes {
        let c = ABSOLUTE_POS % channels;
        let j = ABSOLUTE_POS / channels;

        let mut acc = Vector::<F, N>::new(F::new(0.0_f32));
        let mut bias_acc = Vector::<F, N>::new(F::new(0.0_f32));
        for batch in 0..batches {
            for s in 0..seq {
                let g = grad[(batch * seq + s) * channels + c];
                let u = s + j;
                let mut v = Vector::<F, N>::new(F::new(0.0_f32));
                if u < carry {
                    if comptime!(has_history) {
                        v = history[(batch * carry + u) * channels + c];
                    }
                } else {
                    v = input[(batch * seq + u - carry) * channels + c];
                }
                acc += g * v;
                if comptime!(has_bias) {
                    if j == 0 {
                        bias_acc += g;
                    }
                }
            }
        }
        d_weight[ABSOLUTE_POS] = acc;
        if comptime!(has_bias) {
            if j == 0 {
                d_bias[c] = bias_acc;
            }
        }
    }
}

/// Everything the convolution kernels need to agree on.
struct ConvShape {
    batches: usize,
    seq: usize,
    channels: usize,
    carry: usize,
    taps: usize,
    line: usize,
}

fn conv_shape<R: Runtime, E: FloatElem>(
    input: &Tensor<R, E>,
    weight: &Tensor<R, E>,
) -> ConvShape {
    let dims = input.dims();
    let channels = dims[2];
    ConvShape {
        batches: dims[0],
        seq: dims[1],
        channels,
        carry: weight.shape().dim(0) - 1,
        taps: weight.shape().dim(0),
        line: line_size_for::<R, E>(input.client(), channels),
    }
}

/// A depthwise causal convolution over `[batch, seq, channels]`.
///
/// `history` holds the `taps - 1` positions before the window; `None` means they are
/// zero, which is what a fresh sequence wants.
pub fn causal_conv1d<R: Runtime, E: FloatElem>(
    input: &Tensor<R, E>,
    history: Option<&Tensor<R, E>>,
    weight: &Tensor<R, E>,
    bias: Option<&Tensor<R, E>>,
) -> Result<Tensor<R, E>> {
    input.shape().expect_rank(3)?;
    let s = conv_shape::<R, E>(input, weight);
    let out = Tensor::empty(input.shape().clone(), input.device());
    if input.is_empty() {
        return Ok(out);
    }
    let placeholder = Tensor::<R, E>::empty(Shape::new(vec![s.line]), input.device());
    let lanes = s.batches * s.seq * (s.channels / s.line);
    let (count, dim) = launch_1d(input.client(), lanes, s.taps * s.line);
    unsafe {
        causal_conv1d_kernel::launch_unchecked::<E, R>(
            input.client(),
            count,
            dim,
            s.line,
            input.arg(),
            history.unwrap_or(&placeholder).arg(),
            weight.arg(),
            bias.unwrap_or(&placeholder).arg(),
            out.arg(),
            s.channels / s.line,
            s.seq,
            s.carry,
            s.taps,
            lanes,
            history.is_some(),
            bias.is_some(),
        );
    }
    Ok(out)
}

/// The `taps - 1` positions the next call should be given as history.
pub fn causal_conv1d_history<R: Runtime, E: FloatElem>(
    input: &Tensor<R, E>,
    history: Option<&Tensor<R, E>>,
    weight: &Tensor<R, E>,
) -> Result<Tensor<R, E>> {
    let s = conv_shape::<R, E>(input, weight);
    let out = Tensor::empty(
        Shape::new(vec![s.batches, s.carry, s.channels]),
        input.device(),
    );
    if out.is_empty() {
        return Ok(out);
    }
    let placeholder = Tensor::<R, E>::empty(Shape::new(vec![s.line]), input.device());
    let lanes = s.batches * s.carry * (s.channels / s.line);
    let (count, dim) = launch_1d(input.client(), lanes, s.line);
    unsafe {
        causal_conv1d_history_kernel::launch_unchecked::<E, R>(
            input.client(),
            count,
            dim,
            s.line,
            input.arg(),
            history.unwrap_or(&placeholder).arg(),
            out.arg(),
            s.channels / s.line,
            s.seq,
            s.carry,
            lanes,
            history.is_some(),
        );
    }
    Ok(out)
}

/// The adjoint of [`causal_conv1d_history`]: the carried positions are a verbatim
/// copy of the tail of the window, so the gradient is a routing of `grad` back into
/// whichever buffer each position came from, and zero everywhere else.
#[cube(launch_unchecked)]
fn causal_conv1d_history_grad_kernel<F: Float + CubeElement, N: Size>(
    grad: &Array<Vector<F, N>>,
    d_input: &mut Array<Vector<F, N>>,
    d_history: &mut Array<Vector<F, N>>,
    channels: usize,
    seq: usize,
    carry: usize,
    lanes: usize,
    #[comptime] has_history: bool,
) {
    if ABSOLUTE_POS < lanes {
        let c = ABSOLUTE_POS % channels;
        let rest = ABSOLUTE_POS / channels;
        let u = rest % (seq + carry);
        let batch = rest / (seq + carry);

        let mut acc = Vector::<F, N>::new(F::new(0.0_f32));
        if u >= seq {
            acc = grad[(batch * carry + u - seq) * channels + c];
        }
        if u < carry {
            if comptime!(has_history) {
                d_history[(batch * carry + u) * channels + c] = acc;
            }
        } else {
            d_input[(batch * seq + u - carry) * channels + c] = acc;
        }
    }
}

/// The adjoint of [`causal_conv1d_history`].
#[allow(clippy::type_complexity)] // Two gradients, one optional.
pub fn causal_conv1d_history_backward<R: Runtime, E: FloatElem>(
    grad: &Tensor<R, E>,
    input: &Tensor<R, E>,
    history: Option<&Tensor<R, E>>,
    weight: &Tensor<R, E>,
) -> Result<(Tensor<R, E>, Option<Tensor<R, E>>)> {
    let s = conv_shape::<R, E>(input, weight);
    let d_input = Tensor::empty(input.shape().clone(), input.device());
    let d_history = history.map(|h| Tensor::empty(h.shape().clone(), h.device()));
    if input.is_empty() {
        return Ok((d_input, d_history));
    }
    let placeholder = Tensor::<R, E>::empty(Shape::new(vec![s.line]), input.device());
    let channel_lines = s.channels / s.line;
    let lanes = s.batches * (s.seq + s.carry) * channel_lines;
    let (count, dim) = launch_1d(input.client(), lanes, s.line);
    unsafe {
        causal_conv1d_history_grad_kernel::launch_unchecked::<E, R>(
            input.client(),
            count,
            dim,
            s.line,
            grad.arg(),
            d_input.arg(),
            d_history.as_ref().unwrap_or(&placeholder).arg(),
            channel_lines,
            s.seq,
            s.carry,
            lanes,
            history.is_some(),
        );
    }
    Ok((d_input, d_history))
}

/// The adjoint of [`causal_conv1d`].
#[allow(clippy::type_complexity)] // Four gradients, three of them optional.
pub fn causal_conv1d_backward<R: Runtime, E: FloatElem>(
    grad: &Tensor<R, E>,
    input: &Tensor<R, E>,
    history: Option<&Tensor<R, E>>,
    weight: &Tensor<R, E>,
    bias: Option<&Tensor<R, E>>,
) -> Result<(
    Tensor<R, E>,
    Option<Tensor<R, E>>,
    Tensor<R, E>,
    Option<Tensor<R, E>>,
)> {
    let s = conv_shape::<R, E>(input, weight);
    let d_input = Tensor::empty(input.shape().clone(), input.device());
    let d_history = history.map(|h| Tensor::empty(h.shape().clone(), h.device()));
    let d_weight = Tensor::empty(weight.shape().clone(), weight.device());
    let d_bias = bias.map(|b| Tensor::empty(b.shape().clone(), b.device()));
    if input.is_empty() {
        return Ok((d_input, d_history, d_weight, d_bias));
    }
    let placeholder = Tensor::<R, E>::empty(Shape::new(vec![s.line]), input.device());
    let channel_lines = s.channels / s.line;

    let lanes = s.batches * (s.seq + s.carry) * channel_lines;
    let (count, dim) = launch_1d(input.client(), lanes, s.taps * s.line);
    unsafe {
        causal_conv1d_input_grad_kernel::launch_unchecked::<E, R>(
            input.client(),
            count,
            dim,
            s.line,
            grad.arg(),
            weight.arg(),
            d_input.arg(),
            d_history.as_ref().unwrap_or(&placeholder).arg(),
            channel_lines,
            s.seq,
            s.carry,
            s.taps,
            lanes,
            history.is_some(),
        );
    }

    let lanes = s.taps * channel_lines;
    let (count, dim) = launch_1d(input.client(), lanes, s.batches * s.seq * s.line);
    unsafe {
        causal_conv1d_weight_grad_kernel::launch_unchecked::<E, R>(
            input.client(),
            count,
            dim,
            s.line,
            grad.arg(),
            input.arg(),
            history.unwrap_or(&placeholder).arg(),
            d_weight.arg(),
            d_bias.as_ref().unwrap_or(&placeholder).arg(),
            channel_lines,
            s.seq,
            s.carry,
            s.batches,
            lanes,
            history.is_some(),
            bias.is_some(),
        );
    }
    Ok((d_input, d_history, d_weight, d_bias))
}

// ---------------------------------------------------------------------------
// SiLU
// ---------------------------------------------------------------------------

/// `x * sigmoid(x)`, and its derivative `s + x * s * (1 - s)`.
///
/// Two launches become one, and the backward's four become one. `sigmoid` is
/// recomputed in the adjoint rather than saved: one exponential is cheaper than the
/// buffer it would take to carry it.
#[cube(launch_unchecked)]
fn silu_kernel<F: Float + CubeElement, N: Size>(
    input: &Array<Vector<F, N>>,
    output: &mut Array<Vector<F, N>>,
    #[comptime] backward: bool,
    grad: &Array<Vector<F, N>>,
) {
    if ABSOLUTE_POS < output.len() {
        let x = input[ABSOLUTE_POS];
        let one = Vector::<F, N>::new(F::new(1.0_f32));
        let s = one / (one + (x * Vector::<F, N>::new(F::new(-1.0_f32))).exp());
        if comptime!(backward) {
            output[ABSOLUTE_POS] = grad[ABSOLUTE_POS] * (s + x * s * (one - s));
        } else {
            output[ABSOLUTE_POS] = x * s;
        }
    }
}

fn silu_launch<R: Runtime, E: FloatElem>(
    input: &Tensor<R, E>,
    grad: Option<&Tensor<R, E>>,
) -> Tensor<R, E> {
    let out = Tensor::empty(input.shape().clone(), input.device());
    let n = out.len();
    if n == 0 {
        return out;
    }
    let line = line_size_for::<R, E>(input.client(), n);
    let (count, dim) = launch_1d(input.client(), n / line, line);
    unsafe {
        silu_kernel::launch_unchecked::<E, R>(
            input.client(),
            count,
            dim,
            line,
            input.arg(),
            out.arg(),
            grad.is_some(),
            grad.unwrap_or(input).arg(),
        );
    }
    out
}

/// `x * sigmoid(x)`.
pub fn silu<R: Runtime, E: FloatElem>(input: &Tensor<R, E>) -> Tensor<R, E> {
    silu_launch(input, None)
}

/// The adjoint of [`silu`].
pub fn silu_backward<R: Runtime, E: FloatElem>(
    grad: &Tensor<R, E>,
    input: &Tensor<R, E>,
) -> Tensor<R, E> {
    silu_launch(input, Some(grad))
}

// ---------------------------------------------------------------------------
// Softplus
// ---------------------------------------------------------------------------

/// `max(x, 0) + ln(1 + exp(-|x|))`, and its derivative `sigmoid(x)`.
///
/// Seven forward launches (`abs`, `neg`, `exp`, `add_scalar`, `log`, `relu`, `add`)
/// and their adjoints become one launch each. `sigmoid` is recomputed in the
/// adjoint rather than saved from the forward pass, which never needs it.
#[cube(launch_unchecked)]
fn softplus_kernel<F: Float + CubeElement, N: Size>(
    input: &Array<Vector<F, N>>,
    output: &mut Array<Vector<F, N>>,
    #[comptime] backward: bool,
    grad: &Array<Vector<F, N>>,
) {
    if ABSOLUTE_POS < output.len() {
        let x = input[ABSOLUTE_POS];
        let one = Vector::<F, N>::new(F::new(1.0_f32));
        let neg_one = Vector::<F, N>::new(F::new(-1.0_f32));
        if comptime!(backward) {
            let s = one / (one + (x * neg_one).exp());
            output[ABSOLUTE_POS] = grad[ABSOLUTE_POS] * s;
        } else {
            let zero = Vector::<F, N>::new(F::new(0.0_f32));
            let stable = (one + (x.abs() * neg_one).exp()).ln();
            output[ABSOLUTE_POS] = x.max(zero) + stable;
        }
    }
}

fn softplus_launch<R: Runtime, E: FloatElem>(
    input: &Tensor<R, E>,
    grad: Option<&Tensor<R, E>>,
) -> Tensor<R, E> {
    let out = Tensor::empty(input.shape().clone(), input.device());
    let n = out.len();
    if n == 0 {
        return out;
    }
    let line = line_size_for::<R, E>(input.client(), n);
    let (count, dim) = launch_1d(input.client(), n / line, line);
    unsafe {
        softplus_kernel::launch_unchecked::<E, R>(
            input.client(),
            count,
            dim,
            line,
            input.arg(),
            out.arg(),
            grad.is_some(),
            grad.unwrap_or(input).arg(),
        );
    }
    out
}

/// `max(x, 0) + ln(1 + exp(-|x|))`, computed stably in one launch.
pub fn softplus<R: Runtime, E: FloatElem>(input: &Tensor<R, E>) -> Tensor<R, E> {
    softplus_launch(input, None)
}

/// The adjoint of [`softplus`]: `grad * sigmoid(input)`.
pub fn softplus_backward<R: Runtime, E: FloatElem>(
    grad: &Tensor<R, E>,
    input: &Tensor<R, E>,
) -> Tensor<R, E> {
    softplus_launch(input, Some(grad))
}

// ---------------------------------------------------------------------------
// The Mamba-3 state update
// ---------------------------------------------------------------------------

/// `out = x0 * s0 + x1 * s1 + x2 * s2`, where each `x` is a full state tensor and
/// each `s` holds one value per `(batch, head)`.
///
/// This is the trapezoidal recurrence's whole state update. As primitives it is
/// three broadcast multiplies and two adds, on tensors of a few thousand elements —
/// five launches whose arithmetic would fit comfortably in one.
#[cube(launch_unchecked)]
fn ssm_state_update_kernel<F: Float + CubeElement, N: Size>(
    x0: &Array<Vector<F, N>>,
    x1: &Array<Vector<F, N>>,
    x2: &Array<Vector<F, N>>,
    scales: &Array<F>,
    out: &mut Array<Vector<F, N>>,
    inner: usize,
    heads: usize,
) {
    if ABSOLUTE_POS < out.len() {
        let head = ABSOLUTE_POS / inner;
        out[ABSOLUTE_POS] = x0[ABSOLUTE_POS] * Vector::<F, N>::new(scales[head])
            + x1[ABSOLUTE_POS] * Vector::<F, N>::new(scales[heads + head])
            + x2[ABSOLUTE_POS] * Vector::<F, N>::new(scales[2 * heads + head]);
    }
}

/// The state-tensor half of the adjoint: `dx_i = grad * s_i`.
#[cube(launch_unchecked)]
fn ssm_state_update_grad_kernel<F: Float + CubeElement, N: Size>(
    grad: &Array<Vector<F, N>>,
    scales: &Array<F>,
    dx0: &mut Array<Vector<F, N>>,
    dx1: &mut Array<Vector<F, N>>,
    dx2: &mut Array<Vector<F, N>>,
    inner: usize,
    heads: usize,
) {
    if ABSOLUTE_POS < grad.len() {
        let head = ABSOLUTE_POS / inner;
        let g = grad[ABSOLUTE_POS];
        dx0[ABSOLUTE_POS] = g * Vector::<F, N>::new(scales[head]);
        dx1[ABSOLUTE_POS] = g * Vector::<F, N>::new(scales[heads + head]);
        dx2[ABSOLUTE_POS] = g * Vector::<F, N>::new(scales[2 * heads + head]);
    }
}

/// The scale half of the adjoint: `ds_i = sum over the head's elements of grad * x_i`.
#[cube(launch_unchecked)]
fn ssm_state_scale_grad_kernel<F: Float + CubeElement, N: Size>(
    grad: &Array<Vector<F, N>>,
    x0: &Array<Vector<F, N>>,
    x1: &Array<Vector<F, N>>,
    x2: &Array<Vector<F, N>>,
    d_scales: &mut Array<F>,
    inner: usize,
    heads: usize,
) {
    if ABSOLUTE_POS < heads {
        let base = ABSOLUTE_POS * inner;
        let mut a0 = grad[base] * x0[base];
        let mut a1 = grad[base] * x1[base];
        let mut a2 = grad[base] * x2[base];
        for i in 1..inner {
            let g = grad[base + i];
            a0 += g * x0[base + i];
            a1 += g * x1[base + i];
            a2 += g * x2[base + i];
        }
        let mut t0 = a0[0];
        let mut t1 = a1[0];
        let mut t2 = a2[0];
        #[unroll]
        for lane in 1..N::value() {
            t0 += a0[lane];
            t1 += a1[lane];
            t2 += a2[lane];
        }
        d_scales[ABSOLUTE_POS] = t0;
        d_scales[heads + ABSOLUTE_POS] = t1;
        d_scales[2 * heads + ABSOLUTE_POS] = t2;
    }
}

/// `x0 * s0 + x1 * s1 + x2 * s2` over `[batch, heads, ...]` state tensors scaled by
/// `[batch, heads]` coefficients.
pub fn ssm_state_update<R: Runtime, E: FloatElem>(
    x: [&Tensor<R, E>; 3],
    scales: &Tensor<R, E>,
) -> Result<Tensor<R, E>> {
    if x[1].shape() != x[0].shape() || x[2].shape() != x[0].shape() {
        return Err(Error::shape(
            "state update operands must share a shape".to_string(),
        ));
    }
    if !scales.len().is_multiple_of(3) {
        return Err(Error::shape(
            "state update scales must be packed as [3, batch * heads]".to_string(),
        ));
    }
    let heads = scales.len() / 3;
    let out = Tensor::empty(x[0].shape().clone(), x[0].device());
    let n = out.len();
    if n == 0 || heads == 0 {
        return Ok(out);
    }
    let inner = n / heads;
    let line = line_size_for::<R, E>(x[0].client(), inner);
    let (count, dim) = launch_1d(x[0].client(), n / line, line);
    unsafe {
        ssm_state_update_kernel::launch_unchecked::<E, R>(
            x[0].client(),
            count,
            dim,
            line,
            x[0].arg(),
            x[1].arg(),
            x[2].arg(),
            scales.arg(),
            out.arg(),
            inner / line,
            heads,
        );
    }
    Ok(out)
}

/// The adjoint of [`ssm_state_update`]: three tensor gradients and the packed scale
/// gradient.
#[allow(clippy::type_complexity)] // Three gradients of one type plus one of another.
pub fn ssm_state_update_backward<R: Runtime, E: FloatElem>(
    grad: &Tensor<R, E>,
    x: [&Tensor<R, E>; 3],
    scales: &Tensor<R, E>,
) -> Result<([Tensor<R, E>; 3], Tensor<R, E>)> {
    let heads = scales.len() / 3;
    let shape = x[0].shape().clone();
    let dx = [
        Tensor::empty(shape.clone(), grad.device()),
        Tensor::empty(shape.clone(), grad.device()),
        Tensor::empty(shape, grad.device()),
    ];
    let ds = Tensor::empty(scales.shape().clone(), grad.device());
    let n = grad.len();
    if n == 0 || heads == 0 {
        return Ok((dx, ds));
    }
    let inner = n / heads;
    let line = line_size_for::<R, E>(grad.client(), inner);

    let (count, dim) = launch_1d(grad.client(), n / line, line);
    unsafe {
        ssm_state_update_grad_kernel::launch_unchecked::<E, R>(
            grad.client(),
            count,
            dim,
            line,
            grad.arg(),
            scales.arg(),
            dx[0].arg(),
            dx[1].arg(),
            dx[2].arg(),
            inner / line,
            heads,
        );
    }

    let (count, dim) = launch_1d(grad.client(), heads, inner);
    unsafe {
        ssm_state_scale_grad_kernel::launch_unchecked::<E, R>(
            grad.client(),
            count,
            dim,
            line,
            grad.arg(),
            x[0].arg(),
            x[1].arg(),
            x[2].arg(),
            ds.arg(),
            inner / line,
            heads,
        );
    }
    Ok((dx, ds))
}

// ---------------------------------------------------------------------------
// The Mamba-3 rotating frame
// ---------------------------------------------------------------------------

/// `phi = wrap(prev + dt * theta)`, wrapped into `[-pi, pi]`.
///
/// `dt` is one value per `(batch, head)`; `theta` and `prev` carry `d_state / 2`
/// angles each. Composed of primitives this is a broadcast multiply, an add, and a
/// four-op wrap.
#[cube(launch_unchecked)]
fn ssm_angle_kernel<F: Float + CubeElement, N: Size>(
    dt: &Array<F>,
    theta: &Array<Vector<F, N>>,
    prev: &Array<Vector<F, N>>,
    phi: &mut Array<Vector<F, N>>,
    two_pi: F,
    inv_two_pi: F,
    inner: usize,
    lanes: usize,
    #[comptime] has_prev: bool,
) {
    if ABSOLUTE_POS < lanes {
        let head = ABSOLUTE_POS / inner;
        let mut raw = theta[ABSOLUTE_POS] * Vector::<F, N>::new(dt[head]);
        if comptime!(has_prev) {
            raw += prev[ABSOLUTE_POS];
        }
        let turns = (raw * Vector::<F, N>::new(inv_two_pi)).round();
        phi[ABSOLUTE_POS] = raw - turns * Vector::<F, N>::new(two_pi);
    }
}

/// The adjoint of [`ssm_angle_kernel`].
///
/// Wrapping subtracts a locally constant multiple of `2 pi`, so it differentiates to
/// the identity and the gradient passes straight through to `prev`. `dt`'s share is
/// left as a per-element partial for the caller to sum over the angles.
#[cube(launch_unchecked)]
fn ssm_angle_backward_kernel<F: Float + CubeElement, N: Size>(
    grad: &Array<Vector<F, N>>,
    dt: &Array<F>,
    theta: &Array<Vector<F, N>>,
    d_theta: &mut Array<Vector<F, N>>,
    d_dt_partial: &mut Array<Vector<F, N>>,
    inner: usize,
    lanes: usize,
) {
    if ABSOLUTE_POS < lanes {
        let head = ABSOLUTE_POS / inner;
        let g = grad[ABSOLUTE_POS];
        d_theta[ABSOLUTE_POS] = g * Vector::<F, N>::new(dt[head]);
        d_dt_partial[ABSOLUTE_POS] = g * theta[ABSOLUTE_POS];
    }
}

/// Advance the rotating frame by one step: `wrap(prev + dt * theta)`.
///
/// `theta` is `[batch, heads, d_state / 2]` and `dt` is `[batch, heads]`.
pub fn ssm_angle<R: Runtime, E: FloatElem>(
    dt: &Tensor<R, E>,
    theta: &Tensor<R, E>,
    prev: Option<&Tensor<R, E>>,
) -> Result<Tensor<R, E>> {
    let heads = dt.len();
    let out = Tensor::empty(theta.shape().clone(), theta.device());
    let n = out.len();
    if n == 0 || heads == 0 {
        return Ok(out);
    }
    if !n.is_multiple_of(heads) {
        return Err(Error::shape(
            "angle table must divide evenly among the heads".to_string(),
        ));
    }
    let inner = n / heads;
    let line = line_size_for::<R, E>(theta.client(), inner);
    let (count, dim) = launch_1d(theta.client(), n / line, line);
    let two_pi = 2.0 * core::f32::consts::PI;
    unsafe {
        ssm_angle_kernel::launch_unchecked::<E, R>(
            theta.client(),
            count,
            dim,
            line,
            dt.arg(),
            theta.arg(),
            prev.unwrap_or(theta).arg(),
            out.arg(),
            E::from_scalar(two_pi),
            E::from_scalar(1.0 / two_pi),
            inner / line,
            n / line,
            prev.is_some(),
        );
    }
    Ok(out)
}

/// The adjoint of [`ssm_angle`]: gradients for `dt`, `theta` and `prev`.
#[allow(clippy::type_complexity)] // Three gradients, one optional.
pub fn ssm_angle_backward<R: Runtime, E: FloatElem>(
    grad: &Tensor<R, E>,
    dt: &Tensor<R, E>,
    theta: &Tensor<R, E>,
    has_prev: bool,
) -> Result<(Tensor<R, E>, Tensor<R, E>, Option<Tensor<R, E>>)> {
    let heads = dt.len();
    let d_theta = Tensor::empty(theta.shape().clone(), theta.device());
    let d_dt = Tensor::empty(dt.shape().clone(), dt.device());
    let n = grad.len();
    if n == 0 || heads == 0 {
        return Ok((d_dt, d_theta, has_prev.then(|| grad.clone())));
    }
    let inner = n / heads;
    let line = line_size_for::<R, E>(grad.client(), inner);
    let partial = Tensor::empty(theta.shape().clone(), theta.device());
    let (count, dim) = launch_1d(grad.client(), n / line, line);
    unsafe {
        ssm_angle_backward_kernel::launch_unchecked::<E, R>(
            grad.client(),
            count,
            dim,
            line,
            grad.arg(),
            dt.arg(),
            theta.arg(),
            d_theta.arg(),
            partial.arg(),
            inner / line,
            n / line,
        );
    }
    // Sum each head's angles back onto its single `dt`.
    let flat = partial.reshape(Shape::new(vec![heads, inner]))?;
    let d_dt = crate::tensor::ops::reduce::sum_dim(&flat, 1)?.reshape(d_dt.shape().clone())?;
    // Wrapping is the identity in the backward, so `prev` gets the gradient intact.
    Ok((d_dt, d_theta, has_prev.then(|| grad.clone())))
}

// ---------------------------------------------------------------------------
// The Mamba-3 per-head coefficients
// ---------------------------------------------------------------------------

/// `alpha`, `beta` and `g` for one step, packed as `[3, batch * heads]`.
///
/// ```text
/// a     = -exp(a_log[head])
/// alpha = exp(dt * a)
/// beta  = (1 - lambda) * dt * alpha
/// g     = lambda * dt
/// ```
///
/// Packed rather than three tensors because a tape node has one output, and
/// [`ssm_state_update`] reads them packed — so nothing has to slice them apart.
#[cube(launch_unchecked)]
fn ssm_coefficients_kernel<F: Float + CubeElement>(
    a_log: &Array<F>,
    dt: &Array<F>,
    lambda: &Array<F>,
    out: &mut Array<F>,
    heads: usize,
    lanes: usize,
) {
    if ABSOLUTE_POS < lanes {
        let a = F::new(0.0_f32) - a_log[ABSOLUTE_POS % heads].exp();
        let step = dt[ABSOLUTE_POS];
        let lam = lambda[ABSOLUTE_POS];
        let alpha = (step * a).exp();
        out[ABSOLUTE_POS] = alpha;
        out[lanes + ABSOLUTE_POS] = (F::new(1.0_f32) - lam) * step * alpha;
        out[2 * lanes + ABSOLUTE_POS] = lam * step;
    }
}

/// The adjoint of [`ssm_coefficients_kernel`]. `a_log`'s share comes out per
/// `(batch, head)` for the caller to sum over the batch.
#[cube(launch_unchecked)]
fn ssm_coefficients_backward_kernel<F: Float + CubeElement>(
    grad: &Array<F>,
    a_log: &Array<F>,
    dt: &Array<F>,
    lambda: &Array<F>,
    d_dt: &mut Array<F>,
    d_lambda: &mut Array<F>,
    d_a_partial: &mut Array<F>,
    heads: usize,
    lanes: usize,
) {
    if ABSOLUTE_POS < lanes {
        let a = F::new(0.0_f32) - a_log[ABSOLUTE_POS % heads].exp();
        let step = dt[ABSOLUTE_POS];
        let lam = lambda[ABSOLUTE_POS];
        let alpha = (step * a).exp();
        let keep = F::new(1.0_f32) - lam;

        let g_alpha = grad[ABSOLUTE_POS];
        let g_beta = grad[lanes + ABSOLUTE_POS];
        let g_g = grad[2 * lanes + ABSOLUTE_POS];

        d_dt[ABSOLUTE_POS] = g_alpha * a * alpha
            + g_g * lam
            + g_beta * keep * alpha * (F::new(1.0_f32) + step * a);
        d_lambda[ABSOLUTE_POS] = g_g * step - g_beta * step * alpha;
        // d/d a_log = d/da * da/d a_log, and da/d a_log = -exp(a_log) = a.
        d_a_partial[ABSOLUTE_POS] =
            (g_alpha * step * alpha + g_beta * keep * step * step * alpha) * a;
    }
}

/// One step's per-head coefficients, packed as `[3, batch * heads]`.
pub fn ssm_coefficients<R: Runtime, E: FloatElem>(
    a_log: &Tensor<R, E>,
    dt: &Tensor<R, E>,
    lambda: &Tensor<R, E>,
) -> Result<Tensor<R, E>> {
    let lanes = dt.len();
    let heads = a_log.len();
    let out = Tensor::empty(Shape::new(vec![3, lanes.max(1)]), dt.device());
    if lanes == 0 || heads == 0 {
        return Ok(out);
    }
    if lambda.len() != lanes {
        return Err(Error::shape(
            "lambda must have one value per (batch, head)".to_string(),
        ));
    }
    let (count, dim) = launch_1d(dt.client(), lanes, 4);
    unsafe {
        ssm_coefficients_kernel::launch_unchecked::<E, R>(
            dt.client(),
            count,
            dim,
            a_log.arg(),
            dt.arg(),
            lambda.arg(),
            out.arg(),
            heads,
            lanes,
        );
    }
    Ok(out)
}

/// The adjoint of [`ssm_coefficients`].
#[allow(clippy::type_complexity)] // One gradient per input, all the same type.
pub fn ssm_coefficients_backward<R: Runtime, E: FloatElem>(
    grad: &Tensor<R, E>,
    a_log: &Tensor<R, E>,
    dt: &Tensor<R, E>,
    lambda: &Tensor<R, E>,
) -> Result<(Tensor<R, E>, Tensor<R, E>, Tensor<R, E>)> {
    let lanes = dt.len();
    let heads = a_log.len();
    let d_dt = Tensor::empty(dt.shape().clone(), dt.device());
    let d_lambda = Tensor::empty(lambda.shape().clone(), lambda.device());
    let d_a_log = Tensor::empty(a_log.shape().clone(), a_log.device());
    if lanes == 0 || heads == 0 {
        return Ok((d_a_log, d_dt, d_lambda));
    }
    let partial = Tensor::<R, E>::empty(Shape::new(vec![lanes]), dt.device());
    let (count, dim) = launch_1d(dt.client(), lanes, 8);
    unsafe {
        ssm_coefficients_backward_kernel::launch_unchecked::<E, R>(
            dt.client(),
            count,
            dim,
            grad.arg(),
            a_log.arg(),
            dt.arg(),
            lambda.arg(),
            d_dt.arg(),
            d_lambda.arg(),
            partial.arg(),
            heads,
            lanes,
        );
    }
    let batched = partial.reshape(Shape::new(vec![lanes / heads, heads]))?;
    let d_a_log = crate::tensor::ops::reduce::sum_dim(&batched, 0)?
        .reshape(d_a_log.shape().clone())?;
    Ok((d_a_log, d_dt, d_lambda))
}

// ---------------------------------------------------------------------------
// The adjoint of a slice
// ---------------------------------------------------------------------------

/// Place `grad` back into a zeroed frame at `[start, start + len)` along an axis.
///
/// This is what differentiating a slice needs, and composed of primitives it is
/// expensive out of all proportion to the arithmetic: allocate a zero block for the
/// head, allocate another for the tail, then concatenate three pieces — two fills and
/// three copies to move one band of numbers. The training step of a Mamba-3 layer
/// slices eighteen times, so it was paying about fifty launches for it.
#[cube(launch_unchecked)]
fn slice_backward_kernel<F: Float + CubeElement, N: Size>(
    grad: &Array<Vector<F, N>>,
    out: &mut Array<Vector<F, N>>,
    inner: usize,
    axis_len: usize,
    len: usize,
    start: usize,
) {
    if ABSOLUTE_POS < out.len() {
        let i = ABSOLUTE_POS % inner;
        let rest = ABSOLUTE_POS / inner;
        let d = rest % axis_len;
        let outer = rest / axis_len;

        let mut v = Vector::<F, N>::new(F::new(0.0_f32));
        if d >= start && d < start + len {
            v = grad[(outer * len + d - start) * inner + i];
        }
        out[ABSOLUTE_POS] = v;
    }
}

/// The adjoint of [`crate::tensor::ops::movement::slice`].
pub fn slice_backward<R: Runtime, E: FloatElem>(
    grad: &Tensor<R, E>,
    full: &Shape,
    axis: usize,
    start: usize,
) -> Result<Tensor<R, E>> {
    if axis >= full.rank() {
        return Err(Error::shape(format!(
            "axis {axis} out of range for shape {full}"
        )));
    }
    crate::backend::trace_shape!("TRACE slice_backward {full} axis={axis} from {}", grad.shape());
    let out = Tensor::empty(full.clone(), grad.device());
    let n = out.len();
    if n == 0 {
        return Ok(out);
    }
    let inner = full.inner(axis);
    let axis_len = full.dim(axis);
    let len = grad.shape().dim(axis);
    let line = line_size_for::<R, E>(grad.client(), inner);
    let (count, dim) = launch_1d(grad.client(), n / line, line);
    unsafe {
        slice_backward_kernel::launch_unchecked::<E, R>(
            grad.client(),
            count,
            dim,
            line,
            grad.arg(),
            out.arg(),
            inner / line,
            axis_len,
            len,
            start,
        );
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Sum of squares
// ---------------------------------------------------------------------------

/// Sum `input`'s squares into `groups` partials, written at `offset` in `out`.
///
/// Every unit walks a strided share of the buffer and guards the tail, so no padding
/// is needed and the whole reduction is one launch per tensor. The offset lets a
/// caller collect many tensors' partials into one buffer and finish them together —
/// which is what a global gradient norm wants, and what stops it from being four
/// launches and a concatenation per parameter.
#[cube(launch_unchecked)]
fn sum_squares_kernel<F: Float + CubeElement, N: Size>(
    input: &Array<Vector<F, N>>,
    out: &mut Array<F>,
    lines: usize,
    groups: usize,
    offset: usize,
) {
    if ABSOLUTE_POS < groups {
        let mut acc = Vector::<F, N>::new(F::new(0.0_f32));
        let mut i = ABSOLUTE_POS;
        while i < lines {
            let v = input[i];
            acc += v * v;
            i += groups;
        }
        let mut total = acc[0];
        #[unroll]
        for lane in 1..N::value() {
            total += acc[lane];
        }
        out[offset + ABSOLUTE_POS] = total;
    }
}

/// `out[0] = scale * min(1, max_norm / (sqrt(sum_squares[0]) + 1e-6))`.
#[cube(launch_unchecked)]
fn clip_factor_kernel<F: Float + CubeElement>(
    sum_squares: &Array<F>,
    out: &mut Array<F>,
    max_norm: F,
    scale: F,
    #[comptime] clipping: bool,
) {
    if ABSOLUTE_POS < 1 {
        let mut factor = scale;
        if comptime!(clipping) {
            // `sum_squares` is over the *accumulated* gradients, before averaging, so
            // the norm being clipped is the averaged one: `scale` multiplies it here
            // exactly as it multiplies the gradients later.
            let norm = sum_squares[0].sqrt() * scale + F::new(1e-6_f32);
            let clip = max_norm / norm;
            // Bounding the factor from *both* sides is what makes a non-finite norm
            // leave it alone, matching the host version — clipping is not the place
            // to silently zero a diverged step. A `NaN` norm gives a `NaN` factor,
            // and every comparison against it is false; an infinite one gives exactly
            // zero, which the lower bound rejects. Neither needs an `is_finite`, which
            // is just as well: an infinite float *literal* is a shader-creation error
            // in WGSL, and this kernel has to compile on every backend.
            if clip < F::new(1.0_f32) && clip > F::new(0.0_f32) {
                factor *= clip;
            }
        }
        out[0] = factor;
    }
}

/// The gradient scale a step should apply, computed on the device.
///
/// `sum_squares` is the one-element result of reducing every gradient's squares;
/// `scale` is the micro-batch averaging factor. Setting `clipping` false skips the
/// norm entirely and just publishes `scale`, which is what a run with clipping
/// disabled wants — still as a buffer, so the optimizer kernel is identical either
/// way and no host synchronisation is introduced by the choice.
pub fn clip_factor<R: Runtime, E: FloatElem>(
    sum_squares: &Tensor<R, E>,
    max_norm: f32,
    scale: f32,
    clipping: bool,
) -> Tensor<R, E> {
    let out = Tensor::empty(Shape::new(vec![1]), sum_squares.device());
    let (count, dim) = launch_1d(sum_squares.client(), 1, 4);
    unsafe {
        clip_factor_kernel::launch_unchecked::<E, R>(
            sum_squares.client(),
            count,
            dim,
            sum_squares.arg(),
            out.arg(),
            E::from_scalar(max_norm),
            E::from_scalar(scale),
            clipping,
        );
    }
    out
}

/// How many partials [`sum_squares_into`] produces for a tensor of `n` elements.
///
/// Enough to keep a GPU busy, few enough that finishing them costs nothing.
///
/// The first version of this was a flat cap of 256, which is a fine number of
/// *partials* and a terrible number of *lanes*: a 2.4 M-element gradient — an
/// ordinary input projection — then had 256 units walking about 2 400 vectors each,
/// and the global gradient norm cost 5% of a training step for arithmetic that should
/// be free. One lane per few hundred elements keeps a GPU's worth of units busy; the
/// upper bound keeps the partial buffer, and the single reduction that finishes it,
/// negligible.
pub fn sum_squares_groups(n: usize) -> usize {
    n.div_ceil(256).clamp(1, 8192)
}

/// Sum the squares of `input` into `out[offset .. offset + groups]`.
pub fn sum_squares_into<R: Runtime, E: FloatElem>(
    input: &Tensor<R, E>,
    out: &Tensor<R, E>,
    offset: usize,
) -> Result<()> {
    let n = input.len();
    if n == 0 {
        return Ok(());
    }
    let line = line_size_for::<R, E>(input.client(), n);
    let lines = n / line;
    let groups = sum_squares_groups(n).min(lines);
    let (count, dim) = launch_1d(input.client(), groups, lines / groups.max(1) * line);
    unsafe {
        sum_squares_kernel::launch_unchecked::<E, R>(
            input.client(),
            count,
            dim,
            line,
            input.arg(),
            out.arg(),
            lines,
            groups,
            offset,
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// The scan's semiseparable band
// ---------------------------------------------------------------------------

/// `band[t, s] = weight(s) * exp(clamp(acum[t] - acum[s], floor, 0))` for `s <= t`,
/// and zero above the diagonal, where `weight(s)` is `g[s]` on the diagonal and
/// `w[s]` below it.
///
/// This is the whole intra-chunk mixing matrix of the chunked scan apart from
/// `C B^T`, and it is the one place where the scan's arithmetic and its memory
/// traffic disagree by an order of magnitude. Written out of primitives it is a
/// broadcast subtract, a clamp, an exponential, two masked broadcast multiplies and
/// an add — seven passes over a `[rows, chunk, chunk]` tensor, six of which exist
/// only to be consumed by the next one. Here it is one pass that reads three
/// `[rows, chunk]` vectors and writes the band.
///
/// Rows are `(batch, chunk index, head)` flattened, so the band comes out in exactly
/// the layout the following batched matmul wants and no permutation is needed on
/// either side of it.
#[cube(launch_unchecked)]
fn ssd_band_kernel<F: Float + CubeElement>(
    acum: &Array<F>,
    w: &Array<F>,
    g: &Array<F>,
    out: &mut Array<F>,
    chunk: usize,
    floor: F,
) {
    if ABSOLUTE_POS < out.len() {
        let s = ABSOLUTE_POS % chunk;
        let rest = ABSOLUTE_POS / chunk;
        let t = rest % chunk;
        let row = (rest / chunk) * chunk;

        let mut v = F::new(0.0_f32);
        if s <= t {
            // `acum` is a cumulative sum of non-positive steps, so the difference is
            // already at or below zero for `s <= t`; the upper clamp is kept anyway
            // so this matches the composed version bit for bit on padded positions.
            let d = (acum[row + t] - acum[row + s]).max(floor).min(F::new(0.0_f32));
            let scale = select(s == t, g[row + s], w[row + s]);
            v = scale * F::exp(d);
        }
        out[ABSOLUTE_POS] = v;
    }
}

/// The adjoint of [`ssd_band_kernel`]: one unit per `(row, u)`, walking that row and
/// that column of the incoming gradient.
///
/// `d band / d acum[t] = band * inside` and `d band / d acum[s] = -band * inside`,
/// where `inside` is the clamp's own derivative — one strictly inside `(floor, 0)`
/// and zero outside, which is the convention `Var::clamp` uses. So `acum[u]` collects
/// the row sum at `t = u` minus the column sum at `s = u`. The weights are simpler:
/// `w[u]` sees the strictly-lower part of column `u` and `g[u]` sees one element.
#[cube(launch_unchecked)]
fn ssd_band_backward_kernel<F: Float + CubeElement>(
    grad: &Array<F>,
    acum: &Array<F>,
    w: &Array<F>,
    g: &Array<F>,
    d_acum: &mut Array<F>,
    d_w: &mut Array<F>,
    d_g: &mut Array<F>,
    chunk: usize,
    floor: F,
) {
    if ABSOLUTE_POS < d_acum.len() {
        let u = ABSOLUTE_POS % chunk;
        let row = (ABSOLUTE_POS / chunk) * chunk;
        let plane = row * chunk;
        let zero = F::new(0.0_f32);
        let au = acum[row + u];

        // Row `u`: every `s <= u` contributes to `d acum[u]` with a plus sign.
        let mut rows = zero;
        for s in 0..chunk {
            if s <= u {
                let raw = au - acum[row + s];
                let d = raw.max(floor).min(zero);
                let scale = select(s == u, g[row + s], w[row + s]);
                let band = scale * F::exp(d);
                let inside = select(raw > floor && raw < zero, F::new(1.0_f32), zero);
                rows += grad[plane + u * chunk + s] * band * inside;
            }
        }

        // Column `u`: every `t >= u` contributes with a minus sign, and the same
        // sweep is what the two weight gradients need.
        let mut cols = zero;
        let mut dw = zero;
        let mut dg = zero;
        for t in 0..chunk {
            if t >= u {
                let raw = acum[row + t] - au;
                let d = raw.max(floor).min(zero);
                let decay = F::exp(d);
                let scale = select(t == u, g[row + u], w[row + u]);
                let inside = select(raw > floor && raw < zero, F::new(1.0_f32), zero);
                let gr = grad[plane + t * chunk + u];
                cols += gr * scale * decay * inside;
                if t == u {
                    dg += gr * decay;
                } else {
                    dw += gr * decay;
                }
            }
        }

        d_acum[ABSOLUTE_POS] = rows - cols;
        d_w[ABSOLUTE_POS] = dw;
        d_g[ABSOLUTE_POS] = dg;
    }
}

/// Build the scan's `[rows, chunk, chunk]` band from three `[rows, chunk]` vectors.
pub fn ssd_band<R: Runtime, E: FloatElem>(
    acum: &Tensor<R, E>,
    w: &Tensor<R, E>,
    g: &Tensor<R, E>,
    floor: f32,
) -> Result<Tensor<R, E>> {
    acum.shape().expect_rank(2)?;
    if w.shape() != acum.shape() || g.shape() != acum.shape() {
        return Err(Error::shape(format!(
            "ssd_band operands disagree: {} vs {} vs {}",
            acum.shape(),
            w.shape(),
            g.shape()
        )));
    }
    let rows = acum.shape().dim(0);
    let chunk = acum.shape().dim(1);
    let out = Tensor::empty(Shape::new(vec![rows, chunk, chunk]), acum.device());
    if out.is_empty() {
        return Ok(out);
    }
    let (count, dim) = launch_1d(acum.client(), out.len(), 6);
    unsafe {
        ssd_band_kernel::launch_unchecked::<E, R>(
            acum.client(),
            count,
            dim,
            acum.arg(),
            w.arg(),
            g.arg(),
            out.arg(),
            chunk,
            E::from_scalar(floor),
        );
    }
    Ok(out)
}

/// The adjoint of [`ssd_band`]: gradients for `acum`, `w` and `g`.
pub fn ssd_band_backward<R: Runtime, E: FloatElem>(
    grad: &Tensor<R, E>,
    acum: &Tensor<R, E>,
    w: &Tensor<R, E>,
    g: &Tensor<R, E>,
    floor: f32,
) -> Result<(Tensor<R, E>, Tensor<R, E>, Tensor<R, E>)> {
    let chunk = acum.shape().dim(1);
    let d_acum = Tensor::empty(acum.shape().clone(), acum.device());
    let d_w = Tensor::empty(acum.shape().clone(), acum.device());
    let d_g = Tensor::empty(acum.shape().clone(), acum.device());
    if d_acum.is_empty() {
        return Ok((d_acum, d_w, d_g));
    }
    let (count, dim) = launch_1d(acum.client(), d_acum.len(), 12 * chunk);
    unsafe {
        ssd_band_backward_kernel::launch_unchecked::<E, R>(
            acum.client(),
            count,
            dim,
            grad.arg(),
            acum.arg(),
            w.arg(),
            g.arg(),
            d_acum.arg(),
            d_w.arg(),
            d_g.arg(),
            chunk,
            E::from_scalar(floor),
        );
    }
    Ok((d_acum, d_w, d_g))
}

// ---------------------------------------------------------------------------
// The scan's decay pattern
// ---------------------------------------------------------------------------

/// Broadcast metadata for a three-operand kernel:
/// `[rank, out_shape[MAX_RANK], a_strides[MAX_RANK], b_strides[MAX_RANK],
/// m_strides[MAX_RANK]]`, with zero strides standing in for a missing operand.
fn pack_decay_meta(
    out: &Shape,
    a_strides: &[usize],
    b_strides: &[usize],
    m_strides: &[usize],
) -> Vec<u32> {
    let mut meta = Vec::with_capacity(1 + 4 * MAX_RANK);
    meta.push(out.rank() as u32);
    let mut push_padded = |vals: &[usize]| {
        for i in 0..MAX_RANK {
            meta.push(vals.get(i).copied().unwrap_or(0) as u32);
        }
    };
    push_padded(out.dims());
    push_padded(a_strides);
    push_padded(b_strides);
    push_padded(m_strides);
    meta
}

/// The broadcast shape and packed metadata shared by the decay forward and
/// backward launches.
fn decay_meta<R: Runtime, E: FloatElem>(
    a: &Tensor<R, E>,
    b: Option<&Tensor<R, E>>,
    m: Option<&Tensor<R, E>>,
) -> Result<(Shape, Vec<u32>)> {
    let mut out = a.shape().clone();
    if let Some(b) = b {
        out = Shape::broadcast(&out, b.shape())?;
    }
    if let Some(m) = m {
        out = Shape::broadcast(&out, m.shape())?;
    }
    let zeros = vec![0usize; out.rank()];
    let sa = a.shape().broadcast_strides(&out)?;
    let sb = match b {
        Some(b) => b.shape().broadcast_strides(&out)?,
        None => zeros.clone(),
    };
    let sm = match m {
        Some(m) => m.shape().broadcast_strides(&out)?,
        None => zeros,
    };
    let meta = pack_decay_meta(&out, &sa, &sb, &sm);
    Ok((out, meta))
}

/// `exp(clamp(a - b, floor, 0)) * m`, with NumPy broadcasting on all three
/// operands; `b` and `m` are optional.
///
/// This is the chunked scan's decay pattern. Written out of primitives it is a
/// broadcast subtract, a clamp, an exponential and a multiply — four passes plus,
/// in the backward, the clamp adjoint's four launches and three intermediates per
/// site. The scan hits it three times per call outside the band that
/// [`ssd_band`] already fused; here each site is one launch forward and one back.
///
/// Scalar rather than vectorised: one of the sites multiplies by a mask that is
/// broadcast along the trailing axis, which a vector load cannot express, and the
/// tensors involved are chunk-counted — the win here is dispatches, not bandwidth.
#[cube(launch_unchecked)]
fn exp_decay_kernel<F: Float + CubeElement>(
    a: &Array<F>,
    b: &Array<F>,
    m: &Array<F>,
    out: &mut Array<F>,
    meta: &Array<u32>,
    rank: usize,
    floor: F,
    #[comptime] has_b: bool,
    #[comptime] has_m: bool,
) {
    if ABSOLUTE_POS < out.len() {
        let mut rem = ABSOLUTE_POS;
        let mut a_off = 0usize;
        let mut b_off = 0usize;
        let mut m_off = 0usize;
        for i in 0..rank {
            let d = rank - 1 - i;
            let size = meta[1 + d] as usize;
            let coord = rem % size;
            rem /= size;
            a_off += coord * meta[9 + d] as usize;
            b_off += coord * meta[17 + d] as usize;
            m_off += coord * meta[25 + d] as usize;
        }
        let mut diff = a[a_off];
        if comptime!(has_b) {
            diff -= b[b_off];
        }
        let clamped = diff.max(floor).min(F::new(0.0_f32));
        let mut v = F::exp(clamped);
        if comptime!(has_m) {
            v *= m[m_off];
        }
        out[ABSOLUTE_POS] = v;
    }
}

/// The adjoint of [`exp_decay_kernel`], written at the *output* shape; summing
/// each gradient back down to its operand's shape is the autodiff layer's job.
///
/// `d/da = g * m * y * inside` and `d/db` its negation, where `inside` is the
/// clamp's own derivative — one strictly inside `(floor, 0)` and zero outside,
/// the convention `Var::clamp` uses. `d/dm = g * exp(clamp(a - b))`.
#[cube(launch_unchecked)]
fn exp_decay_backward_kernel<F: Float + CubeElement>(
    grad: &Array<F>,
    a: &Array<F>,
    b: &Array<F>,
    m: &Array<F>,
    d_a: &mut Array<F>,
    d_b: &mut Array<F>,
    d_m: &mut Array<F>,
    meta: &Array<u32>,
    rank: usize,
    floor: F,
    #[comptime] has_b: bool,
    #[comptime] has_m: bool,
    #[comptime] want_a: bool,
    #[comptime] want_b: bool,
    #[comptime] want_m: bool,
) {
    if ABSOLUTE_POS < grad.len() {
        let mut rem = ABSOLUTE_POS;
        let mut a_off = 0usize;
        let mut b_off = 0usize;
        let mut m_off = 0usize;
        for i in 0..rank {
            let d = rank - 1 - i;
            let size = meta[1 + d] as usize;
            let coord = rem % size;
            rem /= size;
            a_off += coord * meta[9 + d] as usize;
            b_off += coord * meta[17 + d] as usize;
            m_off += coord * meta[25 + d] as usize;
        }
        let mut diff = a[a_off];
        if comptime!(has_b) {
            diff -= b[b_off];
        }
        let zero = F::new(0.0_f32);
        let e = F::exp(diff.max(floor).min(zero));
        let g = grad[ABSOLUTE_POS];
        if comptime!(want_a || want_b) {
            let inside = select(diff > floor && diff < zero, F::new(1.0_f32), zero);
            let mut pull = g * e * inside;
            if comptime!(has_m) {
                pull *= m[m_off];
            }
            if comptime!(want_a) {
                d_a[ABSOLUTE_POS] = pull;
            }
            if comptime!(want_b) {
                d_b[ABSOLUTE_POS] = zero - pull;
            }
        }
        if comptime!(want_m) {
            d_m[ABSOLUTE_POS] = g * e;
        }
    }
}

/// `exp(clamp(a - b, floor, 0)) * m` in one launch. `b` and `m` are optional.
pub fn exp_decay<R: Runtime, E: FloatElem>(
    a: &Tensor<R, E>,
    b: Option<&Tensor<R, E>>,
    m: Option<&Tensor<R, E>>,
    floor: f32,
) -> Result<Tensor<R, E>> {
    let (out_shape, meta) = decay_meta(a, b, m)?;
    let rank = out_shape.rank();
    let out = Tensor::empty(out_shape, a.device());
    let n = out.len();
    if n == 0 {
        return Ok(out);
    }
    let placeholder = Tensor::<R, E>::empty(Shape::new(vec![1]), a.device());
    let meta_handle = a.client().create_from_slice(u32::as_bytes(&meta));
    let (count, dim) = launch_1d(a.client(), n, rank * 2);
    unsafe {
        exp_decay_kernel::launch_unchecked::<E, R>(
            a.client(),
            count,
            dim,
            a.arg(),
            b.unwrap_or(&placeholder).arg(),
            m.unwrap_or(&placeholder).arg(),
            out.arg(),
            ArrayArg::from_raw_parts(meta_handle, meta.len()),
            rank,
            E::from_scalar(floor),
            b.is_some(),
            m.is_some(),
        );
    }
    Ok(out)
}

/// The adjoint of [`exp_decay`]: gradients at the output's shape for whichever of
/// `a`, `b` and `m` the caller wants, in one launch.
#[allow(clippy::type_complexity)] // Three optional gradients of one type.
pub fn exp_decay_backward<R: Runtime, E: FloatElem>(
    grad: &Tensor<R, E>,
    a: &Tensor<R, E>,
    b: Option<&Tensor<R, E>>,
    m: Option<&Tensor<R, E>>,
    floor: f32,
    want: [bool; 3],
) -> Result<(
    Option<Tensor<R, E>>,
    Option<Tensor<R, E>>,
    Option<Tensor<R, E>>,
)> {
    let (out_shape, meta) = decay_meta(a, b, m)?;
    let rank = out_shape.rank();
    let [want_a, want_b, want_m] = [want[0], want[1] && b.is_some(), want[2] && m.is_some()];
    let alloc = |wanted: bool| wanted.then(|| Tensor::<R, E>::empty(out_shape.clone(), a.device()));
    let (d_a, d_b, d_m) = (alloc(want_a), alloc(want_b), alloc(want_m));
    let n = grad.len();
    if n == 0 || !(want_a || want_b || want_m) {
        return Ok((d_a, d_b, d_m));
    }
    let placeholder = Tensor::<R, E>::empty(Shape::new(vec![1]), a.device());
    let meta_handle = a.client().create_from_slice(u32::as_bytes(&meta));
    let (count, dim) = launch_1d(a.client(), n, rank * 2);
    unsafe {
        exp_decay_backward_kernel::launch_unchecked::<E, R>(
            a.client(),
            count,
            dim,
            grad.arg(),
            a.arg(),
            b.unwrap_or(&placeholder).arg(),
            m.unwrap_or(&placeholder).arg(),
            d_a.as_ref().unwrap_or(&placeholder).arg(),
            d_b.as_ref().unwrap_or(&placeholder).arg(),
            d_m.as_ref().unwrap_or(&placeholder).arg(),
            ArrayArg::from_raw_parts(meta_handle, meta.len()),
            rank,
            E::from_scalar(floor),
            b.is_some(),
            m.is_some(),
            want_a,
            want_b,
            want_m,
        );
    }
    Ok((d_a, d_b, d_m))
}

// ---------------------------------------------------------------------------
// The Mamba-3 trapezoid weights
// ---------------------------------------------------------------------------

/// `g[t] = lambda[t] * dt[t]` and `w[t] = g[t] + (1 - lambda[t+1]) * dt[t+1]`
/// over `[batch, seq, heads]`, the shifted term zero at the final position.
///
/// This is the trapezoidal-weight block of `mamba3_scan`. As primitives it is two
/// multiplies, a scalar subtract, a zero fill, a slice, a concatenation and an add
/// — about seven launches to produce two tensors the size of `dt`; the shift is
/// read directly here instead of materialised.
#[cube(launch_unchecked)]
fn trapezoid_weights_kernel<F: Float + CubeElement>(
    lambda: &Array<F>,
    dt: &Array<F>,
    g: &mut Array<F>,
    w: &mut Array<F>,
    inner: usize,
    seq: usize,
) {
    if ABSOLUTE_POS < g.len() {
        let t = (ABSOLUTE_POS / inner) % seq;
        let gv = lambda[ABSOLUTE_POS] * dt[ABSOLUTE_POS];
        let mut f = F::new(0.0_f32);
        if t + 1 < seq {
            let next = ABSOLUTE_POS + inner;
            f = (F::new(1.0_f32) - lambda[next]) * dt[next];
        }
        g[ABSOLUTE_POS] = gv;
        w[ABSOLUTE_POS] = gv + f;
    }
}

/// The adjoint of [`trapezoid_weights_kernel`].
///
/// Position `t` hears from `g[t]` and `w[t]` through the product `lambda * dt`,
/// and from `w[t-1]` through the shifted term — so each unit reads its own two
/// gradients plus the previous position's `w` gradient and needs no scatter.
// The comptime guard has to stay outside the runtime one: collapsing them would
// make the whole test runtime and defeat the specialisation.
#[allow(clippy::collapsible_if)]
#[cube(launch_unchecked)]
fn trapezoid_weights_backward_kernel<F: Float + CubeElement>(
    grad_g: &Array<F>,
    grad_w: &Array<F>,
    lambda: &Array<F>,
    dt: &Array<F>,
    d_lambda: &mut Array<F>,
    d_dt: &mut Array<F>,
    inner: usize,
    seq: usize,
    #[comptime] has_g: bool,
    #[comptime] has_w: bool,
) {
    if ABSOLUTE_POS < d_lambda.len() {
        let t = (ABSOLUTE_POS / inner) % seq;
        let lam = lambda[ABSOLUTE_POS];
        let step = dt[ABSOLUTE_POS];
        let mut total = F::new(0.0_f32);
        if comptime!(has_g) {
            total += grad_g[ABSOLUTE_POS];
        }
        if comptime!(has_w) {
            total += grad_w[ABSOLUTE_POS];
        }
        let mut dl = total * step;
        let mut dd = total * lam;
        if comptime!(has_w) {
            if t > 0 {
                let prev = grad_w[ABSOLUTE_POS - inner];
                dl -= prev * step;
                dd += prev * (F::new(1.0_f32) - lam);
            }
        }
        d_lambda[ABSOLUTE_POS] = dl;
        d_dt[ABSOLUTE_POS] = dd;
    }
}

/// The scan's trapezoid weights `(g, w)` from `lambda` and `dt`, both
/// `[batch, seq, heads]`, in one launch.
pub fn trapezoid_weights<R: Runtime, E: FloatElem>(
    lambda: &Tensor<R, E>,
    dt: &Tensor<R, E>,
) -> Result<(Tensor<R, E>, Tensor<R, E>)> {
    lambda.shape().expect_rank(3)?;
    if lambda.shape() != dt.shape() {
        return Err(Error::shape(format!(
            "trapezoid weights operands disagree: {} vs {}",
            lambda.shape(),
            dt.shape()
        )));
    }
    let g = Tensor::empty(lambda.shape().clone(), lambda.device());
    let w = Tensor::empty(lambda.shape().clone(), lambda.device());
    let n = g.len();
    if n == 0 {
        return Ok((g, w));
    }
    let (count, dim) = launch_1d(lambda.client(), n, 4);
    unsafe {
        trapezoid_weights_kernel::launch_unchecked::<E, R>(
            lambda.client(),
            count,
            dim,
            lambda.arg(),
            dt.arg(),
            g.arg(),
            w.arg(),
            lambda.shape().inner(1),
            lambda.shape().dim(1),
        );
    }
    Ok((g, w))
}

/// The adjoint of [`trapezoid_weights`]: gradients for `lambda` and `dt` from the
/// two output gradients, either of which may be absent.
pub fn trapezoid_weights_backward<R: Runtime, E: FloatElem>(
    grad_g: Option<&Tensor<R, E>>,
    grad_w: Option<&Tensor<R, E>>,
    lambda: &Tensor<R, E>,
    dt: &Tensor<R, E>,
) -> Result<(Tensor<R, E>, Tensor<R, E>)> {
    let d_lambda = Tensor::empty(lambda.shape().clone(), lambda.device());
    let d_dt = Tensor::empty(dt.shape().clone(), dt.device());
    let n = d_lambda.len();
    if n == 0 {
        return Ok((d_lambda, d_dt));
    }
    let placeholder = Tensor::<R, E>::empty(Shape::new(vec![1]), lambda.device());
    let (count, dim) = launch_1d(lambda.client(), n, 6);
    unsafe {
        trapezoid_weights_backward_kernel::launch_unchecked::<E, R>(
            lambda.client(),
            count,
            dim,
            grad_g.unwrap_or(&placeholder).arg(),
            grad_w.unwrap_or(&placeholder).arg(),
            lambda.arg(),
            dt.arg(),
            d_lambda.arg(),
            d_dt.arg(),
            lambda.shape().inner(1),
            lambda.shape().dim(1),
            grad_g.is_some(),
            grad_w.is_some(),
        );
    }
    Ok((d_lambda, d_dt))
}

// ---------------------------------------------------------------------------
// Cross entropy
// ---------------------------------------------------------------------------

/// The per-row negative log likelihood, one unit per row.
///
/// `loss[row] = lse - (1 - s) * x[target] - s * mean(x)`, where
/// `lse = max + ln(sum(exp(x - max)))`; the smoothing term drops out at `s = 0`.
/// `lse` is written out alongside the loss because the backward pass rebuilds the
/// softmax from it — two `[rows]` vectors instead of a logits-sized activation.
#[cube(launch_unchecked)]
fn cross_entropy_rows_kernel<F: Float + CubeElement>(
    logits: &Array<F>,
    ids: &Array<u32>,
    loss: &mut Array<F>,
    lse_out: &mut Array<F>,
    classes: usize,
    rows: usize,
    smoothing: F,
    inv_classes: F,
    #[comptime] smooth: bool,
) {
    if ABSOLUTE_POS < rows {
        let base = ABSOLUTE_POS * classes;
        let mut mx = logits[base];
        for i in 1..classes {
            mx = mx.max(logits[base + i]);
        }
        let mut sum_exp = F::new(0.0_f32);
        let mut sum_x = F::new(0.0_f32);
        for i in 0..classes {
            let x = logits[base + i];
            sum_exp += F::exp(x - mx);
            if comptime!(smooth) {
                sum_x += x;
            }
        }
        let lse = mx + F::ln(sum_exp);
        lse_out[ABSOLUTE_POS] = lse;
        let picked = logits[base + ids[ABSOLUTE_POS] as usize];
        if comptime!(smooth) {
            loss[ABSOLUTE_POS] =
                lse - (F::new(1.0_f32) - smoothing) * picked - smoothing * sum_x * inv_classes;
        } else {
            loss[ABSOLUTE_POS] = lse - picked;
        }
    }
}

/// [`cross_entropy_rows_kernel`] with one *plane* per row, for the same reason as
/// [`rms_norm_plane_kernel`]: one unit per vocab-sized row is the wrong shape on a
/// GPU. Lanes stride the row, the reductions are `plane_max` / `plane_sum`, and
/// only lane zero's writes are divergent.
#[cube(launch_unchecked)]
fn cross_entropy_rows_plane_kernel<F: Float + CubeElement>(
    logits: &Array<F>,
    ids: &Array<u32>,
    loss: &mut Array<F>,
    lse_out: &mut Array<F>,
    classes: usize,
    rows: usize,
    smoothing: F,
    inv_classes: F,
    neg_huge: F,
    #[comptime] smooth: bool,
) {
    let width = PLANE_DIM as usize;
    let lane = UNIT_POS_PLANE as usize;
    let row = ABSOLUTE_POS / width;
    let live = row < rows;
    let base = select(live, row, 0) * classes;
    let steps = classes.div_ceil(width);

    let mut mx = neg_huge;
    for s in 0..steps {
        let i = lane + s * width;
        if i < classes {
            mx = mx.max(logits[base + i]);
        }
    }
    let row_max = plane_max(mx);

    let mut sum_exp = F::new(0.0_f32);
    let mut sum_x = F::new(0.0_f32);
    for s in 0..steps {
        let i = lane + s * width;
        if i < classes {
            let x = logits[base + i];
            sum_exp += F::exp(x - row_max);
            if comptime!(smooth) {
                sum_x += x;
            }
        }
    }
    let total_exp = plane_sum(sum_exp);
    let total_x = plane_sum(sum_x);

    if live && lane == 0 {
        let lse = row_max + F::ln(total_exp);
        lse_out[row] = lse;
        let picked = logits[base + ids[row] as usize];
        if comptime!(smooth) {
            loss[row] =
                lse - (F::new(1.0_f32) - smoothing) * picked - smoothing * total_x * inv_classes;
        } else {
            loss[row] = lse - picked;
        }
    }
}

/// The adjoint of the fused cross entropy, one unit per logit:
/// `dlogits[row, v] = grad[row] * (softmax(row, v) - (1 - s) * [v == target] - s / classes)`,
/// with the softmax rebuilt from the saved log-sum-exp. Nothing here is the dense
/// one-hot that differentiating `take_along_last` materialises.
#[cube(launch_unchecked)]
fn cross_entropy_rows_backward_kernel<F: Float + CubeElement>(
    grad: &Array<F>,
    logits: &Array<F>,
    ids: &Array<u32>,
    lse: &Array<F>,
    d_logits: &mut Array<F>,
    classes: usize,
    smoothing: F,
    inv_classes: F,
    #[comptime] smooth: bool,
) {
    if ABSOLUTE_POS < d_logits.len() {
        let row = ABSOLUTE_POS / classes;
        let col = ABSOLUTE_POS % classes;
        let p = F::exp(logits[ABSOLUTE_POS] - lse[row]);
        let hit = select(col == ids[row] as usize, F::new(1.0_f32), F::new(0.0_f32));
        let mut d = p - hit;
        if comptime!(smooth) {
            d = p - (F::new(1.0_f32) - smoothing) * hit - smoothing * inv_classes;
        }
        d_logits[ABSOLUTE_POS] = grad[row] * d;
    }
}

/// The per-token cross entropy of `[rows, classes]` logits against `[rows]`
/// targets, with optional label smoothing, in one launch.
///
/// Returns the `[rows]` losses and the `[rows]` log-sum-exp vector
/// [`cross_entropy_rows_backward`] needs.
pub fn cross_entropy_rows<R: Runtime, E: FloatElem>(
    logits: &Tensor<R, E>,
    ids: &IdTensor<R>,
    smoothing: f32,
) -> Result<(Tensor<R, E>, Tensor<R, E>)> {
    logits.shape().expect_rank(2)?;
    let rows = logits.shape().dim(0);
    let classes = logits.shape().dim(1);
    if ids.len() != rows {
        return Err(Error::shape(format!(
            "cross entropy: {rows} logit rows but {} targets",
            ids.len()
        )));
    }
    let loss = Tensor::empty(Shape::new(vec![rows]), logits.device());
    let lse = Tensor::empty(Shape::new(vec![rows]), logits.device());
    if rows == 0 || classes == 0 {
        return Ok((loss, lse));
    }
    let smooth = smoothing > 0.0;
    if let Some((count, cube_dim)) = plane_per_row::<R>(logits.client(), rows) {
        unsafe {
            cross_entropy_rows_plane_kernel::launch_unchecked::<E, R>(
                logits.client(),
                count,
                cube_dim,
                logits.arg(),
                ids.arg(),
                loss.arg(),
                lse.arg(),
                classes,
                rows,
                E::from_scalar(smoothing),
                E::from_scalar(1.0 / classes as f32),
                E::from_scalar(f32::MIN),
                smooth,
            );
        }
        return Ok((loss, lse));
    }
    let (count, dim) = launch_1d(logits.client(), rows, classes);
    unsafe {
        cross_entropy_rows_kernel::launch_unchecked::<E, R>(
            logits.client(),
            count,
            dim,
            logits.arg(),
            ids.arg(),
            loss.arg(),
            lse.arg(),
            classes,
            rows,
            E::from_scalar(smoothing),
            E::from_scalar(1.0 / classes as f32),
            smooth,
        );
    }
    Ok((loss, lse))
}

/// The adjoint of [`cross_entropy_rows`]: the logits gradient, in one launch.
pub fn cross_entropy_rows_backward<R: Runtime, E: FloatElem>(
    grad: &Tensor<R, E>,
    logits: &Tensor<R, E>,
    ids: &IdTensor<R>,
    lse: &Tensor<R, E>,
    smoothing: f32,
) -> Result<Tensor<R, E>> {
    let classes = logits.shape().dim(1);
    let out = Tensor::empty(logits.shape().clone(), logits.device());
    let n = out.len();
    if n == 0 {
        return Ok(out);
    }
    let (count, dim) = launch_1d(logits.client(), n, 4);
    unsafe {
        cross_entropy_rows_backward_kernel::launch_unchecked::<E, R>(
            logits.client(),
            count,
            dim,
            grad.arg(),
            logits.arg(),
            ids.arg(),
            lse.arg(),
            out.arg(),
            classes,
            E::from_scalar(smoothing),
            E::from_scalar(1.0 / classes as f32),
            smoothing > 0.0,
        );
    }
    Ok(out)
}
