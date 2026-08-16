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
use crate::tensor::shape::Shape;

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
    let (count, cube_dim) = launch_1d(input.client(), rows, dim);
    let placeholder = Tensor::<R, E>::empty(Shape::new(vec![line]), input.device());
    let gain = weight.unwrap_or(&placeholder);
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
    let (count, cube_dim) = launch_1d(input.client(), rows, dim);
    let placeholder = Tensor::<R, E>::empty(Shape::new(vec![line]), input.device());
    let gain = weight.unwrap_or(&placeholder);
    // Only bound, never written, when there is no gain.
    let dw_partial = match weight {
        Some(_) => Tensor::<R, E>::empty(input.shape().clone(), input.device()),
        None => placeholder.clone(),
    };
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
