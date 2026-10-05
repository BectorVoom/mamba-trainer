//! The whole of one incremental Mamba-3 mixer step, between the two projections,
//! in three launches.
//!
//! A decoding step composed of the windowed path's pieces is about eighteen
//! launches between `in_proj` and `out_proj` — split, convolution, history,
//! activation, two norms, `dt`, `lambda`, coefficients, angle, two rotations, an
//! outer product, the state update, the readout, the skip and the gate — each
//! over a few thousand elements at most. At that size a launch costs more than
//! its arithmetic, so the step is priced by the launch count. These three
//! kernels compute the same numbers, each over the axis its work is contiguous
//! along:
//!
//! * [`mixer_step_act_kernel`], one lane per vector of the projection's `z | x
//!   | B | C` columns: the convolution, its new history, and every `silu` —
//!   the gate's included — so all the exponentials of the step are vector ones.
//! * [`mixer_step_coef_kernel`], one lane per `(batch, head)`: the recurrence
//!   coefficients, the new angle, and `B` and `C` biased, normalised and
//!   rotated.
//! * [`mixer_step_state_kernel`], one lane per `(batch, head, channel)` row of
//!   the state: the outer product, the state update, the readout, the skip and
//!   the gate.
//!
//! Forward only: there is no adjoint, so the mixer takes this path when nothing
//! is being recorded and keeps the composed one — which these are checked
//! against in `tests/mixer_step.rs` — for everything else.
//!
//! [`mixer_step`] returns a new state for the one it was given, which is what
//! a caller that reads or masks the state between steps needs. A loop that
//! only steps uses [`mixer_step_in_place`] over [`MixerStepBuffers`] instead:
//! there the state kernel is bound by the memory it moves, and that form
//! moves a quarter of it (the hidden state once each way over one buffer, and
//! the previous outer product as its two factors).

use cubecl::prelude::*;

use crate::backend::{FloatElem, launch_1d_spans, line_size_for};
use crate::error::{Error, Result};
use crate::tensor::base::Tensor;
use crate::tensor::ops::fused::plane_segments_per_row;
use crate::tensor::shape::Shape;

/// Array bindings of the widest of the three kernels (the coefficient one).
const MIXER_STEP_BINDINGS: u32 = 13;

/// Whether the fused step may run on `device` (its kernels fit the bindings).
pub fn mixer_step_supported<R: Runtime>(device: &crate::backend::Device<R>) -> bool {
    device.client().properties().hardware.max_bindings >= MIXER_STEP_BINDINGS + 1
}

/// The convolution, its history and every activation.
///
/// `proj` is the fused projection `[batch, width]`, banded `z | x | B | C | dt
/// | lambda | theta`; `act` receives `silu` of its first `d_inner + channels`
/// columns, the `x | B | C` ones convolved first. Every extent is in vectors:
/// the launcher picks a width that divides the row and both bands.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn mixer_step_act_kernel<F: Float + CubeElement, N: Size>(
    proj: &Array<Vector<F, N>>,
    history: &Array<Vector<F, N>>,
    conv_w: &Array<Vector<F, N>>,
    conv_b: &Array<Vector<F, N>>,
    reset: &Array<F>,
    act: &mut Array<Vector<F, N>>,
    history_out: &mut Array<Vector<F, N>>,
    width: usize,
    gate: usize,
    channels: usize,
    carry: usize,
    lanes: usize,
    span: usize,
    #[comptime] has_conv: bool,
    #[comptime] has_conv_bias: bool,
    #[comptime] has_reset: bool,
) {
    let one = Vector::<F, N>::new(F::new(1.0_f32));
    let neg_one = Vector::<F, N>::new(F::new(-1.0_f32));
    let cols = gate + channels;
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    for pos in start..end {
        let col = pos % cols;
        let batch = pos / cols;
        let mut v = proj[batch * width + col];
        if comptime!(has_conv) {
            if col >= gate {
                let ch = col - gate;
                let hist_at = batch * carry * channels + ch;
                let mut past = Vector::<F, N>::new(F::new(0.0_f32));
                for j in 0..carry {
                    past += conv_w[j * channels + ch] * history[hist_at + j * channels];
                }
                // A reset drops every tap that reaches into the finished
                // episode, here and in the history handed to the next step.
                let mut keep = one;
                if comptime!(has_reset) {
                    keep = Vector::<F, N>::new(F::new(1.0_f32) - reset[batch]);
                }
                for i in 0..carry {
                    if i + 1 < carry {
                        history_out[hist_at + i * channels] =
                            history[hist_at + (i + 1) * channels] * keep;
                    } else {
                        history_out[hist_at + i * channels] = v;
                    }
                }
                v = v * conv_w[carry * channels + ch] + past * keep;
                if comptime!(has_conv_bias) {
                    v += conv_b[ch];
                }
            }
        }
        act[pos] = v * (one / (one + (v * neg_one).exp()));
    }
}

/// Everything per `(batch, head)`; see the module documentation.
///
/// `act` is [`mixer_step_act_kernel`]'s output, `[batch, d_inner + channels]`.
/// Outputs: `bc` `[batch * heads, 2, state]` (`B` then `C`, normalised and
/// rotated), `coef` `[4, batch * heads]` (`alpha`, `beta`, `g`, skip) and
/// `angle`.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn mixer_step_coef_kernel<F: Float + CubeElement>(
    proj: &Array<F>,
    act: &Array<F>,
    b_bias: &Array<F>,
    c_bias: &Array<F>,
    gain: &Array<F>,
    dt_bias: &Array<F>,
    a_log: &Array<F>,
    d_skip: &Array<F>,
    reset: &Array<F>,
    prev_angle: &Array<F>,
    bc: &mut Array<F>,
    coef: &mut Array<F>,
    angle: &mut Array<F>,
    width: usize,
    heads: usize,
    head_dim: usize,
    state: usize,
    per_group: usize,
    lanes: usize,
    span: usize,
    theta_off: usize,
    eps: F,
    inv_state: F,
    fixed_lambda: F,
    two_pi: F,
    inv_two_pi: F,
    #[comptime] has_bc_bias: bool,
    #[comptime] has_norm: bool,
    #[comptime] has_gain: bool,
    #[comptime] has_lambda: bool,
    #[comptime] has_theta: bool,
    #[comptime] has_skip: bool,
    #[comptime] has_reset: bool,
) {
    let one = F::new(1.0_f32);
    let zero = F::new(0.0_f32);
    let groups = heads / per_group;
    let d_inner = heads * head_dim;
    let bc_width = groups * state;
    let channels = d_inner + 2 * bc_width;
    let dt_off = d_inner + channels;
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    for lane in start..end {
        let head = lane % heads;
        let batch = lane / heads;
        let group = head / per_group;
        let row = batch * width;

        let mut keep = one;
        if comptime!(has_reset) {
            keep -= reset[batch];
        }

        // dt, lambda and the recurrence coefficients.
        let biased = proj[row + dt_off + head] + dt_bias[head];
        let dt = F::max(biased, zero) + F::ln(one + F::exp(F::abs(biased) * F::new(-1.0_f32)));
        let mut lam = fixed_lambda;
        if comptime!(has_lambda) {
            let raw = proj[row + dt_off + heads + head];
            lam = one / (one + F::exp(raw * F::new(-1.0_f32)));
        }
        let a = zero - F::exp(a_log[head]);
        let alpha = F::exp(dt * a);
        coef[lane] = keep * alpha;
        coef[lanes + lane] = keep * (one - lam) * dt * alpha;
        coef[2 * lanes + lane] = lam * dt;
        if comptime!(has_skip) {
            coef[3 * lanes + lane] = d_skip[head];
        } else {
            coef[3 * lanes + lane] = zero;
        }

        // B then C: biased and normalised over the state.
        let bc_row = lane * 2 * state;
        for which in 0..2usize {
            let out_at = bc_row + which * state;
            let act_at = batch * dt_off + 2 * d_inner + which * bc_width + group * state;
            let mut squares = zero;
            for n in 0..state {
                let mut v = act[act_at + n];
                if comptime!(has_bc_bias) {
                    if which == 0 {
                        v += b_bias[head * state + n];
                    } else {
                        v += c_bias[head * state + n];
                    }
                }
                squares += v * v;
                bc[out_at + n] = v;
            }
            if comptime!(has_norm) {
                let scale = one / F::sqrt(squares * inv_state + eps);
                for n in 0..state {
                    if comptime!(has_gain) {
                        bc[out_at + n] = bc[out_at + n] * scale * gain[n];
                    } else {
                        bc[out_at + n] = bc[out_at + n] * scale;
                    }
                }
            }
        }

        // Advance the rotating frame and map B and C into it.
        if comptime!(has_theta) {
            let half = state / 2;
            for i in 0..half {
                let at = lane * half + i;
                let raw = proj[row + theta_off + head * half + i] * dt + prev_angle[at];
                let phi = raw - F::round(raw * inv_two_pi) * two_pi;
                angle[at] = phi;
                let cos = F::cos(phi);
                let sin = F::sin(phi);
                for which in 0..2usize {
                    let lo_at = bc_row + which * state + i;
                    let lo = bc[lo_at];
                    let hi = bc[lo_at + half];
                    bc[lo_at] = lo * cos + hi * sin;
                    bc[lo_at + half] = hi * cos - lo * sin;
                }
            }
        }
    }
}

/// [`mixer_step_coef_kernel`] with a segment of one plane per `(batch, head)`
/// instead of one unit, for a device with planes.
///
/// The unit-per-head kernel is a few hundred lanes each walking the state
/// three times over, one memory round trip per element: on a GPU it is bound
/// by that latency, not by its arithmetic. Here lane `i` of a head's segment
/// takes element `i` of `B` and of `C` — with a rotation, the pair `(i, i +
/// state / 2)` it mixes — so the walk is one step, and the norm's sum over the
/// state is a butterfly inside the segment (`seg_bits == 0`: a whole plane per
/// head and one `plane_sum`), as in the plane RMS norm.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn mixer_step_coef_plane_kernel<F: Float + CubeElement>(
    proj: &Array<F>,
    act: &Array<F>,
    b_bias: &Array<F>,
    c_bias: &Array<F>,
    gain: &Array<F>,
    dt_bias: &Array<F>,
    a_log: &Array<F>,
    d_skip: &Array<F>,
    reset: &Array<F>,
    prev_angle: &Array<F>,
    bc: &mut Array<F>,
    coef: &mut Array<F>,
    angle: &mut Array<F>,
    width: usize,
    heads: usize,
    head_dim: usize,
    state: usize,
    per_group: usize,
    lanes: usize,
    theta_off: usize,
    eps: F,
    inv_state: F,
    fixed_lambda: F,
    two_pi: F,
    inv_two_pi: F,
    #[comptime] has_bc_bias: bool,
    #[comptime] has_norm: bool,
    #[comptime] has_gain: bool,
    #[comptime] has_lambda: bool,
    #[comptime] has_theta: bool,
    #[comptime] has_skip: bool,
    #[comptime] has_reset: bool,
    #[comptime] seg_bits: u32,
) {
    let one = F::new(1.0_f32);
    let zero = F::new(0.0_f32);
    let mut seg = PLANE_DIM as usize;
    let mut slot = UNIT_POS_PLANE as usize;
    if comptime!(seg_bits > 0) {
        seg = comptime!(1usize << seg_bits);
        slot = UNIT_POS_PLANE as usize % seg;
    }
    let at_lane = ABSOLUTE_POS / seg;
    let live = at_lane < lanes;
    let lane = select(live, at_lane, 0);

    let groups = heads / per_group;
    let d_inner = heads * head_dim;
    let bc_width = groups * state;
    let channels = d_inner + 2 * bc_width;
    let dt_off = d_inner + channels;
    let head = lane % heads;
    let batch = lane / heads;
    let group = head / per_group;
    let row = batch * width;

    let mut keep = one;
    if comptime!(has_reset) {
        keep -= reset[batch];
    }

    // dt, lambda and the recurrence coefficients: every lane of the segment
    // needs `dt`, lane zero stores the coefficients.
    let biased = proj[row + dt_off + head] + dt_bias[head];
    let dt = F::max(biased, zero) + F::ln(one + F::exp(F::abs(biased) * F::new(-1.0_f32)));
    let mut lam = fixed_lambda;
    if comptime!(has_lambda) {
        let raw = proj[row + dt_off + heads + head];
        lam = one / (one + F::exp(raw * F::new(-1.0_f32)));
    }
    let a = zero - F::exp(a_log[head]);
    let alpha = F::exp(dt * a);
    if live && slot == 0 {
        coef[lane] = keep * alpha;
        coef[lanes + lane] = keep * (one - lam) * dt * alpha;
        coef[2 * lanes + lane] = lam * dt;
        if comptime!(has_skip) {
            coef[3 * lanes + lane] = d_skip[head];
        } else {
            coef[3 * lanes + lane] = zero;
        }
    }

    // A lane's elements: `i` of `B` and of `C`, and with a rotation the
    // partner `i + half` of each as well.
    let half = state / 2;
    let mut reach = state;
    if comptime!(has_theta) {
        reach = half;
    }
    let steps = reach.div_ceil(seg);
    let b_in = batch * dt_off + 2 * d_inner + group * state;
    let c_in = b_in + bc_width;
    let b_out = lane * 2 * state;
    let c_out = b_out + state;
    let bias_at = head * state;

    let mut b_squares = zero;
    let mut c_squares = zero;
    if comptime!(has_norm) {
        for s in 0..steps {
            let i = slot + s * seg;
            if i < reach {
                let mut vb = act[b_in + i];
                let mut vc = act[c_in + i];
                if comptime!(has_bc_bias) {
                    vb += b_bias[bias_at + i];
                    vc += c_bias[bias_at + i];
                }
                b_squares += vb * vb;
                c_squares += vc * vc;
                if comptime!(has_theta) {
                    let mut wb = act[b_in + half + i];
                    let mut wc = act[c_in + half + i];
                    if comptime!(has_bc_bias) {
                        wb += b_bias[bias_at + half + i];
                        wc += c_bias[bias_at + half + i];
                    }
                    b_squares += wb * wb;
                    c_squares += wc * wc;
                }
            }
        }
        if comptime!(seg_bits == 0) {
            b_squares = plane_sum(b_squares);
            c_squares = plane_sum(c_squares);
        } else {
            #[unroll]
            for k in 0..seg_bits {
                b_squares += plane_shuffle_xor(b_squares, 1u32 << k);
                c_squares += plane_shuffle_xor(c_squares, 1u32 << k);
            }
        }
    }
    let mut b_scale = one;
    let mut c_scale = one;
    if comptime!(has_norm) {
        b_scale = one / F::sqrt(b_squares * inv_state + eps);
        c_scale = one / F::sqrt(c_squares * inv_state + eps);
    }

    for s in 0..steps {
        let i = slot + s * seg;
        if i < reach && live {
            let mut vb = act[b_in + i];
            let mut vc = act[c_in + i];
            if comptime!(has_bc_bias) {
                vb += b_bias[bias_at + i];
                vc += c_bias[bias_at + i];
            }
            vb = vb * b_scale;
            vc = vc * c_scale;
            if comptime!(has_gain) {
                vb = vb * gain[i];
                vc = vc * gain[i];
            }
            if comptime!(has_theta) {
                let mut wb = act[b_in + half + i];
                let mut wc = act[c_in + half + i];
                if comptime!(has_bc_bias) {
                    wb += b_bias[bias_at + half + i];
                    wc += c_bias[bias_at + half + i];
                }
                wb = wb * b_scale;
                wc = wc * c_scale;
                if comptime!(has_gain) {
                    wb = wb * gain[half + i];
                    wc = wc * gain[half + i];
                }
                // Advance the rotating frame and map the pair into it.
                let at = lane * half + i;
                let raw = proj[row + theta_off + head * half + i] * dt + prev_angle[at];
                let phi = raw - F::round(raw * inv_two_pi) * two_pi;
                angle[at] = phi;
                let cos = F::cos(phi);
                let sin = F::sin(phi);
                bc[b_out + i] = vb * cos + wb * sin;
                bc[b_out + half + i] = wb * cos - vb * sin;
                bc[c_out + i] = vc * cos + wc * sin;
                bc[c_out + half + i] = wc * cos - vc * sin;
            } else {
                bc[b_out + i] = vb;
                bc[c_out + i] = vc;
            }
        }
    }
}

/// Everything per state row `(batch, head, channel)`:
///
/// ```text
/// u  = x B^T
/// h' = alpha h + beta last_u + g u
/// y  = (h' . C + skip x) * silu(z)
/// ```
///
/// `act` holds `silu(z)` then the activated `x` on each of its rows. The state
/// axis is walked in vectors, so `state_lines` is `state / N`.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn mixer_step_state_kernel<F: Float + CubeElement, N: Size>(
    act: &Array<F>,
    bc: &Array<Vector<F, N>>,
    coef: &Array<F>,
    h: &Array<Vector<F, N>>,
    last_u: &Array<Vector<F, N>>,
    h_out: &mut Array<Vector<F, N>>,
    u_out: &mut Array<Vector<F, N>>,
    y: &mut Array<F>,
    act_width: usize,
    d_inner: usize,
    head_dim: usize,
    state_lines: usize,
    lanes: usize,
    rows: usize,
    span: usize,
) {
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > rows {
        end = rows;
    }
    for row in start..end {
        let lane = row / head_dim;
        let act_at = (row / d_inner) * act_width + row % d_inner;

        let x = act[act_at + d_inner];
        let x_v = Vector::<F, N>::new(x);
        let alpha = Vector::<F, N>::new(coef[lane]);
        let beta = Vector::<F, N>::new(coef[lanes + lane]);
        let g = Vector::<F, N>::new(coef[2 * lanes + lane]);

        let b_at = lane * 2 * state_lines;
        let c_at = b_at + state_lines;
        let h_at = row * state_lines;
        let mut acc = Vector::<F, N>::new(F::new(0.0_f32));
        for i in 0..state_lines {
            let u = bc[b_at + i] * x_v;
            let next = h[h_at + i] * alpha + last_u[h_at + i] * beta + u * g;
            h_out[h_at + i] = next;
            u_out[h_at + i] = u;
            acc += next * bc[c_at + i];
        }
        let mut total = acc[0];
        #[unroll]
        for l in 1..N::value() {
            total += acc[l];
        }
        y[row] = (total + coef[3 * lanes + lane] * x) * act[act_at];
    }
}

/// [`mixer_step_state_kernel`] with a segment of one plane per state row
/// instead of one unit, for a device with planes.
///
/// One unit per row walks its own `state` values while its neighbours walk
/// theirs a row away, so every load of a wave touches as many cache lines as
/// it has lanes: on an RDNA3.5 iGPU that made this kernel most of a decoding
/// step. Here lane `l` of a row's segment takes vector `l` of the row, every
/// load and store is contiguous across the wave, and the readout's sum over
/// the state is a butterfly inside the segment (`seg_bits == 0`: a whole plane
/// per row and one `plane_sum`), as in the plane RMS norm.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn mixer_step_state_plane_kernel<F: Float + CubeElement, N: Size>(
    act: &Array<F>,
    bc: &Array<Vector<F, N>>,
    coef: &Array<F>,
    h: &Array<Vector<F, N>>,
    last_u: &Array<Vector<F, N>>,
    h_out: &mut Array<Vector<F, N>>,
    u_out: &mut Array<Vector<F, N>>,
    y: &mut Array<F>,
    act_width: usize,
    d_inner: usize,
    head_dim: usize,
    state_lines: usize,
    lanes: usize,
    rows: usize,
    #[comptime] seg_bits: u32,
) {
    let mut width = PLANE_DIM as usize;
    let mut slot = UNIT_POS_PLANE as usize;
    if comptime!(seg_bits > 0) {
        width = comptime!(1usize << seg_bits);
        slot = UNIT_POS_PLANE as usize % width;
    }
    let row = ABSOLUTE_POS / width;
    let live = row < rows;
    let safe_row = select(live, row, 0);
    let lane = safe_row / head_dim;
    let act_at = (safe_row / d_inner) * act_width + safe_row % d_inner;

    let x = act[act_at + d_inner];
    let x_v = Vector::<F, N>::new(x);
    let alpha = Vector::<F, N>::new(coef[lane]);
    let beta = Vector::<F, N>::new(coef[lanes + lane]);
    let g = Vector::<F, N>::new(coef[2 * lanes + lane]);

    let b_at = lane * 2 * state_lines;
    let c_at = b_at + state_lines;
    let h_at = safe_row * state_lines;
    let steps = state_lines.div_ceil(width);
    let mut acc = Vector::<F, N>::new(F::new(0.0_f32));
    for s in 0..steps {
        let i = slot + s * width;
        if i < state_lines {
            let u = bc[b_at + i] * x_v;
            let next = h[h_at + i] * alpha + last_u[h_at + i] * beta + u * g;
            if live {
                h_out[h_at + i] = next;
                u_out[h_at + i] = u;
            }
            acc += next * bc[c_at + i];
        }
    }
    let mut total = acc[0];
    #[unroll]
    for l in 1..N::value() {
        total += acc[l];
    }
    let mut row_total = total;
    if comptime!(seg_bits == 0) {
        row_total = plane_sum(total);
    } else {
        #[unroll]
        for k in 0..seg_bits {
            row_total += plane_shuffle_xor(row_total, 1u32 << k);
        }
    }
    if live && slot == 0 {
        y[row] = (row_total + coef[3 * lanes + lane] * x) * act[act_at];
    }
}

/// [`mixer_step_state_kernel`] for a state that is only ever stepped, never
/// read back between steps: `h` is updated where it lies, and the previous
/// outer product is read as its two factors — the activated `x` and the `B`
/// the previous step left in its `act` and `bc` buffers — instead of a stored
/// `[rows, state]` tensor.
///
/// The stored form moves four state-sized tensors a step (`h` and `last_u` in,
/// both out); this one moves `h` once each way over one buffer, and the
/// factors are a few rows.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn mixer_step_state_in_place_kernel<F: Float + CubeElement, N: Size>(
    act: &Array<F>,
    bc: &Array<Vector<F, N>>,
    coef: &Array<F>,
    prev_act: &Array<F>,
    prev_bc: &Array<Vector<F, N>>,
    h: &mut Array<Vector<F, N>>,
    y: &mut Array<F>,
    act_width: usize,
    d_inner: usize,
    head_dim: usize,
    state_lines: usize,
    lanes: usize,
    rows: usize,
    span: usize,
) {
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > rows {
        end = rows;
    }
    for row in start..end {
        let lane = row / head_dim;
        let act_at = (row / d_inner) * act_width + row % d_inner;

        let x = act[act_at + d_inner];
        let x_v = Vector::<F, N>::new(x);
        let prev_x_v = Vector::<F, N>::new(prev_act[act_at + d_inner]);
        let alpha = Vector::<F, N>::new(coef[lane]);
        let beta = Vector::<F, N>::new(coef[lanes + lane]);
        let g = Vector::<F, N>::new(coef[2 * lanes + lane]);

        let b_at = lane * 2 * state_lines;
        let c_at = b_at + state_lines;
        let h_at = row * state_lines;
        let mut acc = Vector::<F, N>::new(F::new(0.0_f32));
        for i in 0..state_lines {
            let u = bc[b_at + i] * x_v;
            let last_u = prev_bc[b_at + i] * prev_x_v;
            let next = h[h_at + i] * alpha + last_u * beta + u * g;
            h[h_at + i] = next;
            acc += next * bc[c_at + i];
        }
        let mut total = acc[0];
        #[unroll]
        for l in 1..N::value() {
            total += acc[l];
        }
        y[row] = (total + coef[3 * lanes + lane] * x) * act[act_at];
    }
}

/// [`mixer_step_state_in_place_kernel`] with a segment of one plane per state
/// row, as [`mixer_step_state_plane_kernel`] is to the stored form.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn mixer_step_state_in_place_plane_kernel<F: Float + CubeElement, N: Size>(
    act: &Array<F>,
    bc: &Array<Vector<F, N>>,
    coef: &Array<F>,
    prev_act: &Array<F>,
    prev_bc: &Array<Vector<F, N>>,
    h: &mut Array<Vector<F, N>>,
    y: &mut Array<F>,
    act_width: usize,
    d_inner: usize,
    head_dim: usize,
    state_lines: usize,
    lanes: usize,
    rows: usize,
    #[comptime] seg_bits: u32,
) {
    let mut width = PLANE_DIM as usize;
    let mut slot = UNIT_POS_PLANE as usize;
    if comptime!(seg_bits > 0) {
        width = comptime!(1usize << seg_bits);
        slot = UNIT_POS_PLANE as usize % width;
    }
    let row = ABSOLUTE_POS / width;
    let live = row < rows;
    let safe_row = select(live, row, 0);
    let lane = safe_row / head_dim;
    let act_at = (safe_row / d_inner) * act_width + safe_row % d_inner;

    let x = act[act_at + d_inner];
    let x_v = Vector::<F, N>::new(x);
    let prev_x_v = Vector::<F, N>::new(prev_act[act_at + d_inner]);
    let alpha = Vector::<F, N>::new(coef[lane]);
    let beta = Vector::<F, N>::new(coef[lanes + lane]);
    let g = Vector::<F, N>::new(coef[2 * lanes + lane]);

    let b_at = lane * 2 * state_lines;
    let c_at = b_at + state_lines;
    let h_at = safe_row * state_lines;
    let steps = state_lines.div_ceil(width);
    let mut acc = Vector::<F, N>::new(F::new(0.0_f32));
    for s in 0..steps {
        let i = slot + s * width;
        if i < state_lines {
            let u = bc[b_at + i] * x_v;
            let last_u = prev_bc[b_at + i] * prev_x_v;
            let next = h[h_at + i] * alpha + last_u * beta + u * g;
            if live {
                h[h_at + i] = next;
            }
            acc += next * bc[c_at + i];
        }
    }
    let mut total = acc[0];
    #[unroll]
    for l in 1..N::value() {
        total += acc[l];
    }
    let mut row_total = total;
    if comptime!(seg_bits == 0) {
        row_total = plane_sum(total);
    } else {
        #[unroll]
        for k in 0..seg_bits {
            row_total += plane_shuffle_xor(row_total, 1u32 << k);
        }
    }
    if live && slot == 0 {
        y[row] = (row_total + coef[3 * lanes + lane] * x) * act[act_at];
    }
}

/// The layer shape [`mixer_step`] runs at.
#[derive(Debug, Clone, Copy)]
pub struct MixerStepShape {
    /// Environments (rows of the projection).
    pub batch: usize,
    /// SSM heads.
    pub heads: usize,
    /// Channels per head.
    pub head_dim: usize,
    /// State width per head.
    pub state: usize,
    /// `B`/`C` groups; divides `heads`.
    pub groups: usize,
}

/// The projection of one position and the layer's parameters: what a step
/// reads besides its recurrent state. Every `Option` is a piece of the layer
/// the configuration may leave out.
pub struct MixerStepLayer<'a, R: Runtime, E: FloatElem> {
    /// The fused projection of one position, `[batch, width]` elements, banded
    /// `z | x | B | C | dt | lambda | theta`.
    pub proj: &'a Tensor<R, E>,
    /// Depthwise convolution `(weight, bias)`: `[taps, channels]` and
    /// `[channels]`.
    pub conv: Option<(&'a Tensor<R, E>, Option<&'a Tensor<R, E>>)>,
    /// Per-head `B` and `C` biases, `[heads, state]` each.
    pub bc_bias: Option<(&'a Tensor<R, E>, &'a Tensor<R, E>)>,
    /// The `B`/`C` RMS norm: `(gain, eps)`.
    pub bc_norm: Option<(Option<&'a Tensor<R, E>>, f32)>,
    /// `[heads]`.
    pub dt_bias: &'a Tensor<R, E>,
    /// `[heads]`.
    pub a_log: &'a Tensor<R, E>,
    /// `[heads]`.
    pub d_skip: Option<&'a Tensor<R, E>>,
    /// The trapezoid weight when it is not projected.
    pub fixed_lambda: Option<f32>,
    /// Whether the step is rotational: the projection then carries a `theta`
    /// band and the state an angle.
    pub rotational: bool,
    /// `[batch]`, `1` where the environment was reset before this position.
    pub reset: Option<&'a Tensor<R, E>>,
}

/// The tensors [`mixer_step`] reads.
pub struct MixerStepInputs<'a, R: Runtime, E: FloatElem> {
    /// The projection and the parameters.
    pub layer: MixerStepLayer<'a, R, E>,
    /// The convolution history `[batch, taps - 1, channels]`; present exactly
    /// when the layer has a convolution.
    pub history: Option<&'a Tensor<R, E>>,
    /// The carried rotation angle `[batch, heads, state / 2]`; present exactly
    /// when the layer is rotational.
    pub angle: Option<&'a Tensor<R, E>>,
    /// Hidden state `[batch, heads, head_dim, state]`.
    pub h: &'a Tensor<R, E>,
    /// The previous step's outer product, same shape as `h`.
    pub last_u: &'a Tensor<R, E>,
}

/// What [`mixer_step`] produces.
pub struct MixerStepOutput<R: Runtime, E: FloatElem> {
    /// The gated scan output `[batch, 1, d_inner]`, ready for `out_proj`.
    pub y: Tensor<R, E>,
    /// The new hidden state.
    pub h: Tensor<R, E>,
    /// This step's outer product.
    pub last_u: Tensor<R, E>,
    /// The new rotation angle, when rotational.
    pub angle: Option<Tensor<R, E>>,
    /// The new convolution history, when the layer has a convolution.
    pub history: Option<Tensor<R, E>>,
}

/// The recurrent state and the scratch of [`mixer_step_in_place`], allocated
/// once for a decoding loop.
///
/// It is the state of [`mixer_step`] in the form a loop that only steps
/// wants: `h` is updated where it lies; the previous outer product is kept
/// as its two factors, which are the previous step's `act` and `bc` buffers,
/// so those two, the angle and the convolution history alternate between two
/// buffers each; and no step allocates. Nothing here is meant to be read
/// between steps.
pub struct MixerStepBuffers<R: Runtime, E: FloatElem> {
    h: Tensor<R, E>,
    act: [Tensor<R, E>; 2],
    bc: [Tensor<R, E>; 2],
    coef: Tensor<R, E>,
    y: Tensor<R, E>,
    angle: Option<[Tensor<R, E>; 2]>,
    history: Option<[Tensor<R, E>; 2]>,
    /// Which of each pair the next step writes.
    cur: usize,
}

impl<R: Runtime, E: FloatElem> MixerStepBuffers<R, E> {
    /// The state before the first position: everything a step reads is zero.
    /// `taps` is the convolution's kernel size (`0` without a convolution).
    pub fn zeros(
        shape: MixerStepShape,
        taps: usize,
        rotational: bool,
        device: &crate::backend::Device<R>,
    ) -> Self {
        let MixerStepShape {
            batch,
            heads,
            head_dim,
            state,
            groups,
        } = shape;
        let d_inner = heads * head_dim;
        let channels = d_inner + 2 * groups * state;
        let lanes = batch * heads;
        // The first step writes the first of each pair and reads the second,
        // which is therefore the one that must start at zero.
        let pair = |dims: Vec<usize>| {
            [
                Tensor::empty(dims.clone(), device),
                Tensor::zeros(dims, device),
            ]
        };
        Self {
            h: Tensor::zeros(vec![batch, heads, head_dim, state], device),
            act: pair(vec![batch * (d_inner + channels)]),
            bc: pair(vec![lanes * 2 * state]),
            coef: Tensor::empty(vec![4 * lanes], device),
            y: Tensor::empty(vec![batch, 1, d_inner], device),
            angle: rotational.then(|| pair(vec![batch, heads, state / 2])),
            history: (taps >= 2).then(|| pair(vec![batch, taps - 1, channels])),
            cur: 0,
        }
    }

    /// Elements held on the device.
    pub fn num_elements(&self) -> usize {
        let pair = |p: &[Tensor<R, E>; 2]| p[0].len() + p[1].len();
        self.h.len()
            + pair(&self.act)
            + pair(&self.bc)
            + self.coef.len()
            + self.y.len()
            + self.angle.as_ref().map_or(0, pair)
            + self.history.as_ref().map_or(0, pair)
    }
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

fn expect_len<R: Runtime, E: FloatElem>(what: &str, t: &Tensor<R, E>, want: usize) -> Result<()> {
    if t.len() != want {
        return Err(Error::shape(format!(
            "mixer step: {what} must hold {want} elements, got {}",
            t.shape()
        )));
    }
    Ok(())
}

/// The extents of one step, derived from the shape and the layer.
struct Extents {
    d_inner: usize,
    channels: usize,
    half: usize,
    lanes: usize,
    rows: usize,
    width: usize,
    act_width: usize,
    /// Convolution taps that reach into the past.
    carry: usize,
    has_lambda: bool,
}

/// Check the layer against the shape and derive the step's extents.
fn extents<R: Runtime, E: FloatElem>(
    shape: MixerStepShape,
    layer: &MixerStepLayer<'_, R, E>,
) -> Result<Extents> {
    let MixerStepShape {
        batch,
        heads,
        head_dim,
        state,
        groups,
    } = shape;
    if groups == 0 || !heads.is_multiple_of(groups) {
        return Err(Error::shape(format!(
            "mixer step: {groups} groups do not divide {heads} heads"
        )));
    }
    let d_inner = heads * head_dim;
    let channels = d_inner + 2 * groups * state;
    let half = state / 2;
    let has_lambda = layer.fixed_lambda.is_none();
    let width = d_inner
        + channels
        + heads
        + if has_lambda { heads } else { 0 }
        + if layer.rotational { heads * half } else { 0 };

    expect_len("the projection", layer.proj, batch * width)?;
    expect_len("dt_bias", layer.dt_bias, heads)?;
    expect_len("a_log", layer.a_log, heads)?;
    if let Some(d) = layer.d_skip {
        expect_len("d_skip", d, heads)?;
    }
    if let Some((b, c)) = layer.bc_bias {
        expect_len("b_bias", b, heads * state)?;
        expect_len("c_bias", c, heads * state)?;
    }
    if let Some((Some(gain), _)) = layer.bc_norm {
        expect_len("the B/C norm gain", gain, state)?;
    }
    if layer.rotational && !state.is_multiple_of(2) {
        return Err(Error::shape(format!(
            "mixer step: a rotational state must be even, got {state}"
        )));
    }
    if let Some(reset) = layer.reset {
        expect_len("the reset mask", reset, batch)?;
    }
    let mut carry = 0;
    if let Some((weight, bias)) = layer.conv {
        if weight.len() == 0 || !weight.len().is_multiple_of(channels) {
            return Err(Error::shape(format!(
                "mixer step: convolution weight must be [taps, {channels}], got {}",
                weight.shape()
            )));
        }
        carry = weight.len() / channels - 1;
        if let Some(bias) = bias {
            expect_len("the convolution bias", bias, channels)?;
        }
    }
    Ok(Extents {
        d_inner,
        channels,
        half,
        lanes: batch * heads,
        rows: batch * d_inner,
        width,
        act_width: d_inner + channels,
        carry,
        has_lambda,
    })
}

/// The first two launches of a step: the convolution and the activations into
/// `act` (and the new history into `history.1`), then the coefficients into
/// `coef`, `B` and `C` into `bc` and the new angle into `angle.1`. `history`
/// and `angle` are `(read, written)`.
#[allow(clippy::too_many_arguments)]
fn launch_act_coef<R: Runtime, E: FloatElem>(
    shape: MixerStepShape,
    layer: &MixerStepLayer<'_, R, E>,
    ext: &Extents,
    history: Option<(&Tensor<R, E>, &Tensor<R, E>)>,
    angle: Option<(&Tensor<R, E>, &Tensor<R, E>)>,
    act: &Tensor<R, E>,
    bc: &Tensor<R, E>,
    coef: &Tensor<R, E>,
) {
    let MixerStepShape {
        batch,
        heads,
        head_dim,
        state,
        groups,
    } = shape;
    let Extents {
        d_inner,
        channels,
        lanes,
        width,
        act_width,
        carry,
        has_lambda,
        ..
    } = *ext;
    let device = layer.proj.device();
    let client = layer.proj.client();
    // A binding the configuration leaves out still needs a buffer. Nothing
    // reads one that is only read, so the projection stands in for those; one
    // that is written gets a buffer of its own, allocated only when needed.
    let placeholder = layer.proj;

    // The convolution and every activation. A vector must not straddle a row
    // of the projection or the `z | xBC` boundary, and the channel count is
    // what the history and the weight are indexed by.
    let line = line_size_for::<R, E>(client, gcd(width, gcd(d_inner, channels)));
    {
        let wide = layer.proj;
        let (conv_w, conv_b) = match layer.conv {
            Some((weight, bias)) => (weight, bias),
            None => (wide, None),
        };
        // With a single tap there is no history buffer to read or write.
        let scratch;
        let (history, history_out) = match history {
            Some((history, out)) if carry > 0 => (history, out),
            _ => {
                scratch = Tensor::<R, E>::empty(Shape::new(vec![line]), device);
                (wide, &scratch)
            }
        };
        let act_lanes = batch * act_width / line;
        let (count, dim, span) = launch_1d_spans(client, act_lanes, line * (carry + 4));
        unsafe {
            mixer_step_act_kernel::launch_unchecked::<E, R>(
                client,
                count,
                dim,
                line,
                layer.proj.arg(),
                history.arg(),
                conv_w.arg(),
                conv_b.unwrap_or(wide).arg(),
                layer.reset.unwrap_or(placeholder).arg(),
                act.arg(),
                history_out.arg(),
                width / line,
                d_inner / line,
                channels / line,
                carry,
                act_lanes,
                span,
                layer.conv.is_some(),
                conv_b.is_some(),
                layer.reset.is_some(),
            );
        }
    }

    let (b_bias, c_bias) = layer.bc_bias.unwrap_or((placeholder, placeholder));
    let (gain, eps) = match layer.bc_norm {
        Some((gain, eps)) => (gain, eps),
        None => (None, 0.0),
    };
    let two_pi = 2.0 * core::f32::consts::PI;
    let scratch;
    let (angle, angle_out) = match angle {
        Some(pair) => pair,
        None => {
            scratch = Tensor::<R, E>::empty(Shape::new(vec![1]), device);
            (placeholder, &scratch)
        }
    };
    let theta_off = act_width + heads + if has_lambda { heads } else { 0 };
    // With a rotation a lane takes a pair `(i, i + half)`, so a head's
    // segment is half the state.
    let reach = if layer.rotational { state / 2 } else { state };
    if let Some((count, dim, seg_bits)) = plane_segments_per_row::<R>(client, lanes, reach) {
        unsafe {
            mixer_step_coef_plane_kernel::launch_unchecked::<E, R>(
                client,
                count,
                dim,
                layer.proj.arg(),
                act.arg(),
                b_bias.arg(),
                c_bias.arg(),
                gain.unwrap_or(placeholder).arg(),
                layer.dt_bias.arg(),
                layer.a_log.arg(),
                layer.d_skip.unwrap_or(placeholder).arg(),
                layer.reset.unwrap_or(placeholder).arg(),
                angle.arg(),
                bc.arg(),
                coef.arg(),
                angle_out.arg(),
                width,
                heads,
                head_dim,
                state,
                heads / groups,
                lanes,
                theta_off,
                E::from_scalar(eps),
                E::from_scalar(1.0 / state as f32),
                E::from_scalar(layer.fixed_lambda.unwrap_or(0.0)),
                E::from_scalar(two_pi),
                E::from_scalar(1.0 / two_pi),
                layer.bc_bias.is_some(),
                layer.bc_norm.is_some(),
                gain.is_some(),
                has_lambda,
                layer.rotational,
                layer.d_skip.is_some(),
                layer.reset.is_some(),
                seg_bits,
            );
        }
    } else {
        let (count, dim, span) = launch_1d_spans(client, lanes, 8 * state);
        unsafe {
            mixer_step_coef_kernel::launch_unchecked::<E, R>(
                client,
                count,
                dim,
                layer.proj.arg(),
                act.arg(),
                b_bias.arg(),
                c_bias.arg(),
                gain.unwrap_or(placeholder).arg(),
                layer.dt_bias.arg(),
                layer.a_log.arg(),
                layer.d_skip.unwrap_or(placeholder).arg(),
                layer.reset.unwrap_or(placeholder).arg(),
                angle.arg(),
                bc.arg(),
                coef.arg(),
                angle_out.arg(),
                width,
                heads,
                head_dim,
                state,
                heads / groups,
                lanes,
                span,
                theta_off,
                E::from_scalar(eps),
                E::from_scalar(1.0 / state as f32),
                E::from_scalar(layer.fixed_lambda.unwrap_or(0.0)),
                E::from_scalar(two_pi),
                E::from_scalar(1.0 / two_pi),
                layer.bc_bias.is_some(),
                layer.bc_norm.is_some(),
                gain.is_some(),
                has_lambda,
                layer.rotational,
                layer.d_skip.is_some(),
                layer.reset.is_some(),
            );
        }
    }
}

/// One decoding step of a rank-1 Mamba-3 mixer between its two projections, in
/// three launches. See the module documentation.
pub fn mixer_step<R: Runtime, E: FloatElem>(
    shape: MixerStepShape,
    inputs: MixerStepInputs<'_, R, E>,
) -> Result<MixerStepOutput<R, E>> {
    let MixerStepShape {
        batch,
        heads,
        head_dim,
        state,
        ..
    } = shape;
    let layer = &inputs.layer;
    let ext = extents(shape, layer)?;
    let Extents {
        d_inner,
        channels,
        half,
        lanes,
        rows,
        act_width,
        carry,
        ..
    } = ext;
    let device = layer.proj.device();
    let client = layer.proj.client();

    expect_len("h", inputs.h, rows * state)?;
    expect_len("last_u", inputs.last_u, rows * state)?;
    if layer.rotational != inputs.angle.is_some() {
        return Err(Error::shape(
            "mixer step: the angle is carried exactly when the step is rotational".to_string(),
        ));
    }
    if let Some(angle) = inputs.angle {
        expect_len("the angle", angle, lanes * half)?;
    }
    if layer.conv.is_some() != inputs.history.is_some() {
        return Err(Error::shape(
            "mixer step: the history is carried exactly when the layer has a convolution"
                .to_string(),
        ));
    }
    if let Some(history) = inputs.history {
        expect_len("the convolution history", history, batch * carry * channels)?;
    }

    let state_shape = Shape::new(vec![batch, heads, head_dim, state]);
    let y = Tensor::empty(Shape::new(vec![batch, 1, d_inner]), device);
    let h_out = Tensor::empty(state_shape.clone(), device);
    let u_out = Tensor::empty(state_shape, device);
    let angle_out = inputs
        .angle
        .map(|_| Tensor::empty(Shape::new(vec![batch, heads, half]), device));
    // A convolution of one tap has nothing to carry; its history stays empty.
    let history_out = inputs
        .history
        .map(|_| Tensor::empty(Shape::new(vec![batch, carry, channels]), device));
    if rows == 0 || state == 0 {
        return Ok(MixerStepOutput {
            y,
            h: h_out,
            last_u: u_out,
            angle: angle_out,
            history: history_out,
        });
    }

    let act = Tensor::<R, E>::empty(Shape::new(vec![batch * act_width]), device);
    let bc = Tensor::<R, E>::empty(Shape::new(vec![lanes * 2 * state]), device);
    let coef = Tensor::<R, E>::empty(Shape::new(vec![4 * lanes]), device);
    launch_act_coef(
        shape,
        layer,
        &ext,
        inputs.history.zip(history_out.as_ref()),
        inputs.angle.zip(angle_out.as_ref()),
        &act,
        &bc,
        &coef,
    );

    let line = line_size_for::<R, E>(client, state);
    if let Some((count, dim, seg_bits)) = plane_segments_per_row::<R>(client, rows, state / line) {
        unsafe {
            mixer_step_state_plane_kernel::launch_unchecked::<E, R>(
                client,
                count,
                dim,
                line,
                act.arg(),
                bc.arg(),
                coef.arg(),
                inputs.h.arg(),
                inputs.last_u.arg(),
                h_out.arg(),
                u_out.arg(),
                y.arg(),
                act_width,
                d_inner,
                head_dim,
                state / line,
                lanes,
                rows,
                seg_bits,
            );
        }
        return Ok(MixerStepOutput {
            y,
            h: h_out,
            last_u: u_out,
            angle: angle_out,
            history: history_out,
        });
    }
    let (count, dim, span) = launch_1d_spans(client, rows, 4 * state);
    unsafe {
        mixer_step_state_kernel::launch_unchecked::<E, R>(
            client,
            count,
            dim,
            line,
            act.arg(),
            bc.arg(),
            coef.arg(),
            inputs.h.arg(),
            inputs.last_u.arg(),
            h_out.arg(),
            u_out.arg(),
            y.arg(),
            act_width,
            d_inner,
            head_dim,
            state / line,
            lanes,
            rows,
            span,
        );
    }

    Ok(MixerStepOutput {
        y,
        h: h_out,
        last_u: u_out,
        angle: angle_out,
        history: history_out,
    })
}

/// [`mixer_step`] over [`MixerStepBuffers`]: the same three launches and the
/// same values, with the state updated where it lies and nothing allocated.
///
/// Returns the gated scan output `[batch, 1, d_inner]`, ready for `out_proj`.
/// It is the buffers' own output tensor: the next step overwrites it.
pub fn mixer_step_in_place<R: Runtime, E: FloatElem>(
    shape: MixerStepShape,
    layer: MixerStepLayer<'_, R, E>,
    buffers: &mut MixerStepBuffers<R, E>,
) -> Result<Tensor<R, E>> {
    let MixerStepShape {
        head_dim, state, ..
    } = shape;
    let ext = extents(shape, &layer)?;
    let Extents {
        d_inner,
        channels,
        half,
        lanes,
        rows,
        act_width,
        carry,
        ..
    } = ext;
    let client = layer.proj.client();

    expect_len("h", &buffers.h, rows * state)?;
    expect_len("the output", &buffers.y, rows)?;
    expect_len("the coefficients", &buffers.coef, 4 * lanes)?;
    for i in 0..2 {
        expect_len("the activations", &buffers.act[i], shape.batch * act_width)?;
        expect_len("B and C", &buffers.bc[i], lanes * 2 * state)?;
    }
    if layer.rotational != buffers.angle.is_some() {
        return Err(Error::shape(
            "mixer step: the angle is carried exactly when the step is rotational".to_string(),
        ));
    }
    if let Some(angle) = &buffers.angle {
        expect_len("the angle", &angle[0], lanes * half)?;
    }
    if (carry > 0) != buffers.history.is_some() {
        return Err(Error::shape(
            "mixer step: the history is carried exactly when the convolution has a past tap"
                .to_string(),
        ));
    }
    if let Some(history) = &buffers.history {
        expect_len(
            "the convolution history",
            &history[0],
            shape.batch * carry * channels,
        )?;
    }
    if rows == 0 || state == 0 {
        return Ok(buffers.y.clone());
    }

    let cur = buffers.cur;
    let prev = 1 - cur;
    launch_act_coef(
        shape,
        &layer,
        &ext,
        buffers.history.as_ref().map(|h| (&h[prev], &h[cur])),
        buffers.angle.as_ref().map(|a| (&a[prev], &a[cur])),
        &buffers.act[cur],
        &buffers.bc[cur],
        &buffers.coef,
    );

    let line = line_size_for::<R, E>(client, state);
    if let Some((count, dim, seg_bits)) = plane_segments_per_row::<R>(client, rows, state / line) {
        unsafe {
            mixer_step_state_in_place_plane_kernel::launch_unchecked::<E, R>(
                client,
                count,
                dim,
                line,
                buffers.act[cur].arg(),
                buffers.bc[cur].arg(),
                buffers.coef.arg(),
                buffers.act[prev].arg(),
                buffers.bc[prev].arg(),
                buffers.h.arg(),
                buffers.y.arg(),
                act_width,
                d_inner,
                head_dim,
                state / line,
                lanes,
                rows,
                seg_bits,
            );
        }
    } else {
        let (count, dim, span) = launch_1d_spans(client, rows, 4 * state);
        unsafe {
            mixer_step_state_in_place_kernel::launch_unchecked::<E, R>(
                client,
                count,
                dim,
                line,
                buffers.act[cur].arg(),
                buffers.bc[cur].arg(),
                buffers.coef.arg(),
                buffers.act[prev].arg(),
                buffers.bc[prev].arg(),
                buffers.h.arg(),
                buffers.y.arg(),
                act_width,
                d_inner,
                head_dim,
                state / line,
                lanes,
                rows,
                span,
            );
        }
    }
    buffers.cur = prev;
    Ok(buffers.y.clone())
}
