//! The chunked parallel scan.
//!
//! # From the recurrence to matmuls
//!
//! Mamba-3's exponential-trapezoidal recurrence is
//!
//! ```text
//! h_t = a_t h_{t-1} + b_t B_{t-1} x_{t-1} + g_t B_t x_t,     y_t = C_t^T h_t
//! a_t = exp(dt_t A_t)    b_t = (1-l_t) dt_t a_t    g_t = l_t dt_t
//! ```
//!
//! Unrolling it and grouping by the *input* index `j` (rather than the update
//! index) collapses the two input terms into one:
//!
//! ```text
//! y_t = sum_{j<=t} decay(j->t) * ( g_j + [j<t] * f_j ) * (C_t . B_j) * x_j
//! g_j = l_j dt_j                       (the diagonal weight)
//! f_j = (1 - l_{j+1}) dt_{j+1}         (the extra weight, one step ahead)
//! ```
//!
//! because `b_{j+1} decay(j+1->t) = (1-l_{j+1}) dt_{j+1} decay(j->t)`. So the whole
//! trapezoidal system is the *same* semiseparable structure Mamba-2 uses, with a
//! per-source weight `w_j = g_j + f_j` off the diagonal and `g_j` on it. That is
//! the "2-band mask" of the paper, and it means the second-order rule costs one
//! extra shifted add — not a new kernel.
//!
//! Everything below is therefore the standard chunked SSD algorithm:
//!
//! 1. **intra-chunk** — a masked `(C B^T)` matrix times `X`, all inside one chunk;
//! 2. **chunk summary** — each chunk's contribution to the state at its right edge;
//! 3. **inter-chunk** — a short recurrence over chunk summaries, itself expressed as
//!    a masked matmul;
//! 4. **carry-in** — the incoming state projected onto every position via `C`.
//!
//! Written with differentiable primitives, so the backward pass is generated rather
//! than hand-derived. Cost is `O(T * (N + P) * chunk)` instead of `O(T^2)`.
//!
//! # Carrying state across calls
//!
//! The natural carry is *not* `h_t`: the `b_{t+1} B_t x_t` term has not been applied
//! yet at time `t`. The scan therefore carries the pair `(h, last_u)` where
//! `last_u = B_t x_t^T`, and reconstructs the boundary value as
//! `S = h + (1 - l_0) dt_0 * last_u` when the next window starts. With `f` zero at
//! the final position (there is no `t+1` yet), the scan's own boundary value *is*
//! `h`, so training and incremental decoding agree exactly.

use cubecl::prelude::Runtime;

use crate::autograd::{Var, cat};
use crate::backend::FloatElem;
use crate::error::{Error, Result};
use crate::tensor::Tensor;

/// Log-decays below this are flushed to zero. `exp(-60)` is under `1e-26`, so this
/// only bites on terms that cannot affect the result, and it keeps `exp` away from
/// overflow on the masked upper triangle.
const LOG_DECAY_FLOOR: f32 = -60.0;

/// Shift along `axis` towards *smaller* indices, filling the last slot with zeros.
///
/// This is how `f_j = (1 - l_{j+1}) dt_{j+1}` reads the next token's parameters.
/// Zero-filling the final position is exactly right: `f_{T-1}` would only ever
/// weight an output at `t > T-1`, which this window does not produce.
pub fn shift_left<R: Runtime, E: FloatElem>(x: &Var<R, E>, axis: usize) -> Result<Var<R, E>> {
    let len = x.shape().dim(axis);
    if len == 0 {
        return Ok(x.clone());
    }
    let pad = Var::constant(Tensor::zeros(x.shape().with_dim(axis, 1), x.device()));
    if len == 1 {
        return Ok(pad);
    }
    let tail = x.slice(axis, 1, len - 1)?;
    cat(&[tail, pad], axis)
}

/// Wrap angles into `[-pi, pi]` without disturbing the gradient.
///
/// The subtracted multiple of `2*pi` is detached, so `d/dx wrap(x) = 1` — which is
/// correct, because wrapping is a locally constant shift.
fn wrap_angle<R: Runtime, E: FloatElem>(phi: &Var<R, E>) -> Result<Var<R, E>> {
    phi.wrap_to(2.0 * core::f32::consts::PI)
}

/// Pad `axis` with `n` zeros at the end.
fn pad_end<R: Runtime, E: FloatElem>(
    x: &Var<R, E>,
    axis: usize,
    n: usize,
) -> Result<Var<R, E>> {
    if n == 0 {
        return Ok(x.clone());
    }
    let pad = Var::constant(Tensor::zeros(x.shape().with_dim(axis, n), x.device()));
    cat(&[x.clone(), pad], axis)
}

/// The core structured state space duality scan, one input and one output channel.
///
/// | argument | shape | meaning |
/// |---|---|---|
/// | `x` | `[B, T, H, P]` | per-head input channels |
/// | `b` | `[B, T, H, N]` | state input map |
/// | `c` | `[B, T, H, N]` | state output map |
/// | `a` | `[B, T, H]` | log decay per step, `dt * A` (non-positive) |
/// | `g` | `[B, T, H]` | diagonal input weight |
/// | `w` | `[B, T, H]` | off-diagonal input weight |
/// | `initial_state` | `[B, H, P, N]` | boundary value carried in |
///
/// Returns `y` with shape `[B, T, H, P]` and the boundary state `[B, H, P, N]`.
pub fn ssd_chunked<R: Runtime, E: FloatElem>(
    x: &Var<R, E>,
    b: &Var<R, E>,
    c: &Var<R, E>,
    a: &Var<R, E>,
    g: &Var<R, E>,
    w: &Var<R, E>,
    initial_state: Option<&Var<R, E>>,
    chunk_size: usize,
) -> Result<(Var<R, E>, Var<R, E>)> {
    x.shape().expect_rank(4)?;
    b.shape().expect_rank(4)?;
    let dims = x.dims().to_vec();
    let (batch, seq, heads, head_dim) = (dims[0], dims[1], dims[2], dims[3]);
    let state = b.dims()[3];
    let device = x.device().clone();

    if seq == 0 {
        let zero_state = initial_state.cloned().unwrap_or_else(|| {
            Var::constant(Tensor::zeros(
                vec![batch, heads, head_dim, state],
                &device,
            ))
        });
        return Ok((x.clone(), zero_state));
    }

    let chunk = chunk_size.clamp(1, seq);
    let pad = (chunk - seq % chunk) % chunk;
    let padded = seq + pad;
    let chunks = padded / chunk;

    // Zero padding is inert: `a = 0` leaves the decay at 1 and `w = g = 0` injects
    // nothing, so the state simply passes through the padded tail.
    let x = pad_end(x, 1, pad)?;
    let b = pad_end(&b.clone(), 1, pad)?;
    let c = pad_end(c, 1, pad)?;
    let a = pad_end(a, 1, pad)?;
    let g = pad_end(g, 1, pad)?;
    let w = pad_end(w, 1, pad)?;

    let x = x.reshape(vec![batch, chunks, chunk, heads, head_dim])?;
    let b = b.reshape(vec![batch, chunks, chunk, heads, state])?;
    let c = c.reshape(vec![batch, chunks, chunk, heads, state])?;
    let a = a.reshape(vec![batch, chunks, chunk, heads])?;
    let g = g.reshape(vec![batch, chunks, chunk, heads])?;
    let w = w.reshape(vec![batch, chunks, chunk, heads])?;

    // Cumulative log decay within each chunk: acum[l] = log decay from just before
    // the chunk up to and including position l.
    let acum = a.cumsum(2)?;

    // ---- 1. intra-chunk -------------------------------------------------
    // The mixing matrix is `(C B^T)[t, s] * decay(s -> t) * weight(s)`, where
    // `weight` is `w[s]` below the diagonal, `g[s]` on it and zero above it — the
    // "2-band mask" of the paper. The decay and the weights depend only on `s` and
    // `t`, never on the state, so the whole `[chunk, chunk]` band is a function of
    // three vectors of length `chunk`. `Var::ssd_band` builds it in one launch
    // instead of the seven passes a broadcast subtract, a clamp, an exponential, two
    // masked multiplies and an add would take, and it builds it *head-major*, which
    // is the layout the matmuls on either side of it already use — so the two
    // full-size permutations that used to bracket this step are gone as well.
    let heads_first = |v: &Var<R, E>, last: usize| -> Result<Var<R, E>> {
        v.permute(&[0, 1, 3, 2, 4])?
            .reshape(vec![batch * chunks * heads, chunk, last])
    };
    // The same reordering for the per-position scalars, on tensors `chunk` times
    // smaller than the band they describe.
    let rows = |v: &Var<R, E>| -> Result<Var<R, E>> {
        v.permute(&[0, 1, 3, 2])?
            .reshape(vec![batch * chunks * heads, chunk])
    };

    let band = Var::ssd_band(&rows(&acum)?, &rows(&w)?, &rows(&g)?, LOG_DECAY_FLOOR)?;

    let c_flat = heads_first(&c, state)?;
    let b_flat = heads_first(&b, state)?;
    let x_flat = heads_first(&x, head_dim)?;

    // (C B^T)[t, s] contracted over the state dimension.
    let cb = c_flat.matmul(&b_flat.transpose()?)?;

    let mixing_flat = cb.mul(&band)?;
    let y_diag = mixing_flat
        .matmul(&x_flat)?
        .reshape(vec![batch, chunks, heads, chunk, head_dim])?
        .permute(&[0, 1, 3, 2, 4])?;

    // ---- 2. chunk summaries ---------------------------------------------
    let last_acum = acum.slice(2, chunk - 1, 1)?;
    let decay_to_end = last_acum
        .sub(&acum)?
        .clamp(LOG_DECAY_FLOOR, 0.0)
        .exp();
    let weighted_x = x.mul(&decay_to_end.mul(&w)?.unsqueeze(4)?)?;
    let chunk_state = weighted_x
        .permute(&[0, 1, 3, 4, 2])?
        .reshape(vec![batch * chunks * heads, head_dim, chunk])?
        .matmul(&b_flat)?
        .reshape(vec![batch, chunks, heads, head_dim, state])?;

    // ---- 3. inter-chunk recurrence --------------------------------------
    let chunk_decay = last_acum.reshape(vec![batch, chunks, heads])?;
    let before = chunk_decay.cumsum_exclusive(1)?; // log decay entering chunk k
    let through = chunk_decay.cumsum(1)?; // log decay leaving chunk k
    let strict_chunks = Var::constant(
        Tensor::<R, E>::strict_causal_mask(chunks, &device)
            .reshape(vec![1, chunks, chunks, 1])?,
    );
    let transfer = before
        .unsqueeze(2)?
        .sub(&through.unsqueeze(1)?)?
        .clamp(LOG_DECAY_FLOOR, 0.0)
        .exp()
        .mul(&strict_chunks)?;

    let mut carry_in = transfer
        .permute(&[0, 3, 1, 2])?
        .reshape(vec![batch * heads, chunks, chunks])?
        .matmul(
            &chunk_state
                .permute(&[0, 2, 1, 3, 4])?
                .reshape(vec![batch * heads, chunks, head_dim * state])?,
        )?
        .reshape(vec![batch, heads, chunks, head_dim, state])?
        .permute(&[0, 2, 1, 3, 4])?;

    if let Some(init) = initial_state {
        let scale = before
            .clamp(LOG_DECAY_FLOOR, 0.0)
            .exp()
            .reshape(vec![batch, chunks, heads, 1, 1])?;
        let seeded = init
            .reshape(vec![batch, 1, heads, head_dim, state])?
            .mul(&scale)?;
        carry_in = carry_in.add(&seeded)?;
    }

    // ---- 4. carry-in contribution to every position ---------------------
    let c_decayed = c.mul(&acum.clamp(LOG_DECAY_FLOOR, 0.0).exp().unsqueeze(4)?)?;
    let y_off = heads_first(&c_decayed, state)?
        .matmul(
            &carry_in
                .permute(&[0, 1, 2, 4, 3])?
                .reshape(vec![batch * chunks * heads, state, head_dim])?,
        )?
        .reshape(vec![batch, chunks, heads, chunk, head_dim])?
        .permute(&[0, 1, 3, 2, 4])?;

    let y = y_diag
        .add(&y_off)?
        .reshape(vec![batch, padded, heads, head_dim])?;
    let y = if pad > 0 { y.slice(1, 0, seq)? } else { y };

    // ---- boundary state --------------------------------------------------
    let tail_decay = chunk_decay
        .slice(1, chunks - 1, 1)?
        .clamp(LOG_DECAY_FLOOR, 0.0)
        .exp()
        .reshape(vec![batch, 1, heads, 1, 1])?;
    let final_state = carry_in
        .slice(1, chunks - 1, 1)?
        .mul(&tail_decay)?
        .add(&chunk_state.slice(1, chunks - 1, 1)?)?
        .reshape(vec![batch, heads, head_dim, state])?;

    Ok((y, final_state))
}

/// The state a Mamba-3 layer carries between windows.
pub struct SsmState<R: Runtime, E: FloatElem> {
    /// Hidden state, `[batch, heads, head_dim, d_state]`.
    pub h: Var<R, E>,
    /// The most recent input outer product `sum_j B_j x_j^T`, needed by the
    /// trapezoidal rule's one-step-behind term. Same shape as `h`.
    pub last_u: Var<R, E>,
    /// Running rotation angle `[batch, heads, d_state / 2]` for rotational dynamics.
    pub angle: Option<Var<R, E>>,
}

impl<R: Runtime, E: FloatElem> SsmState<R, E> {
    /// Elements held on the device by this state.
    ///
    /// Fixed by the configuration: it does not depend on how many tokens have been
    /// decoded.
    pub fn num_elements(&self) -> usize {
        self.h.shape().num_elements()
            + self.last_u.shape().num_elements()
            + self
                .angle
                .as_ref()
                .map(|a| a.shape().num_elements())
                .unwrap_or(0)
    }
}

impl<R: Runtime, E: FloatElem> Clone for SsmState<R, E> {
    fn clone(&self) -> Self {
        Self {
            h: self.h.clone(),
            last_u: self.last_u.clone(),
            angle: self.angle.clone(),
        }
    }
}

impl<R: Runtime, E: FloatElem> SsmState<R, E> {
    /// An all-zero state.
    pub fn zeros(
        batch: usize,
        heads: usize,
        head_dim: usize,
        d_state: usize,
        rotational: bool,
        device: &crate::backend::Device<R>,
    ) -> Self {
        let shape = vec![batch, heads, head_dim, d_state];
        Self {
            h: Var::constant(Tensor::zeros(shape.clone(), device)),
            last_u: Var::constant(Tensor::zeros(shape, device)),
            angle: rotational.then(|| {
                Var::constant(Tensor::zeros(vec![batch, heads, d_state / 2], device))
            }),
        }
    }

    /// Detach from the tape, so a cached state does not keep a graph alive.
    pub fn detach(&self) -> Self {
        Self {
            h: self.h.detach(),
            last_u: self.last_u.detach(),
            angle: self.angle.as_ref().map(|a| a.detach()),
        }
    }
}

/// Inputs to a full Mamba-3 scan, after projection and normalisation.
pub struct ScanInputs<'a, R: Runtime, E: FloatElem> {
    /// `[batch, seq, heads, head_dim, rank]`.
    pub x: &'a Var<R, E>,
    /// `[batch, seq, heads, d_state, rank]`.
    pub b: &'a Var<R, E>,
    /// `[batch, seq, heads, d_state, rank]`.
    pub c: &'a Var<R, E>,
    /// Positive time step, `[batch, seq, heads]`.
    pub dt: &'a Var<R, E>,
    /// `log(-A)` per head, `[heads]`.
    pub a_log: &'a Var<R, E>,
    /// Trapezoid mixing coefficient in `[0, 1]`, `[batch, seq, heads]`.
    pub lambda: &'a Var<R, E>,
    /// Angular rate per state pair, `[batch, seq, heads, d_state / 2]`.
    pub theta: Option<&'a Var<R, E>>,
    /// Direct skip gain per head, `[heads]`.
    pub d_skip: Option<&'a Var<R, E>>,
    /// State carried in from the previous window.
    pub state: Option<&'a SsmState<R, E>>,
    /// Chunk length for the parallel scan.
    pub chunk_size: usize,
}

impl<'a, R: Runtime, E: FloatElem> ScanInputs<'a, R, E> {
    /// The required inputs; optional pieces are added with the `with_*` methods.
    pub fn new(
        x: &'a Var<R, E>,
        b: &'a Var<R, E>,
        c: &'a Var<R, E>,
        dt: &'a Var<R, E>,
        a_log: &'a Var<R, E>,
        lambda: &'a Var<R, E>,
    ) -> Self {
        Self {
            x,
            b,
            c,
            dt,
            a_log,
            lambda,
            theta: None,
            d_skip: None,
            state: None,
            chunk_size: 64,
        }
    }

    /// Enable rotational (complex) dynamics with these angular rates.
    pub fn with_theta(mut self, theta: &'a Var<R, E>) -> Self {
        self.theta = Some(theta);
        self
    }

    /// Add the direct `D * x` skip path.
    pub fn with_skip(mut self, d_skip: &'a Var<R, E>) -> Self {
        self.d_skip = Some(d_skip);
        self
    }

    /// Continue from a previous window's state.
    pub fn with_state(mut self, state: &'a SsmState<R, E>) -> Self {
        self.state = Some(state);
        self
    }

    /// Set the scan chunk length.
    pub fn with_chunk_size(mut self, chunk_size: usize) -> Self {
        self.chunk_size = chunk_size;
        self
    }
}

/// Result of a Mamba-3 scan.
pub struct ScanOutput<R: Runtime, E: FloatElem> {
    /// `[batch, seq, heads, head_dim, rank]`.
    pub y: Var<R, E>,
    /// State to carry into the next window.
    pub state: SsmState<R, E>,
}

/// Run the Mamba-3 selective scan.
///
/// Handles the three axes of the architecture at once: the trapezoidal weights,
/// the rotational (complex) state transition, and the MIMO rank decomposition.
pub fn mamba3_scan<R: Runtime, E: FloatElem>(
    inputs: ScanInputs<'_, R, E>,
) -> Result<ScanOutput<R, E>> {
    let x = inputs.x;
    x.shape().expect_rank(5)?;
    let dims = x.dims().to_vec();
    let (batch, seq, heads, head_dim, rank) = (dims[0], dims[1], dims[2], dims[3], dims[4]);
    let d_state = inputs.b.dims()[3];
    let device = x.device().clone();

    // --- decay ---------------------------------------------------------
    // A is parameterised as -exp(a_log) so it stays strictly negative.
    let a_head = inputs
        .a_log
        .exp()
        .neg()
        .reshape(vec![1, 1, heads])?;
    let a = inputs.dt.mul(&a_head)?;

    // --- trapezoidal weights -------------------------------------------
    let g = inputs.lambda.mul(inputs.dt)?;
    let next = inputs.lambda.rsub_scalar(1.0).mul(inputs.dt)?;
    let f = shift_left(&next, 1)?;
    let w = g.add(&f)?;

    // --- rotation -------------------------------------------------------
    let (b_rot, c_rot, end_angle) = match inputs.theta {
        None => (inputs.b.clone(), inputs.c.clone(), None),
        Some(theta) => {
            if !d_state.is_multiple_of(2) {
                return Err(Error::config(
                    "rotational dynamics require an even d_state".to_string(),
                ));
            }
            let step = inputs.dt.unsqueeze(3)?.mul(theta)?;
            let mut phi = step.cumsum(1)?;
            if let Some(state) = inputs.state
                && let Some(prev) = &state.angle
            {
                phi = phi.add(&prev.unsqueeze(1)?)?;
            }
            let phi = wrap_angle(&phi)?;
            // The frame at time t is R_{0:t}, so B and C are mapped by its
            // transpose — a rotation by -phi, which is the convention
            // `rotate_by_angle` takes. The sine and cosine stay inside the kernel.
            let table = phi.unsqueeze(3)?; // [b, t, h, 1, n/2]
            let rotate = |v: &Var<R, E>| -> Result<Var<R, E>> {
                // [b, t, h, n, r] -> [b, t, h, r, n] so the rotation acts on `n`.
                let moved = v.permute(&[0, 1, 2, 4, 3])?;
                let done = moved.rotate_by_angle(&table)?;
                done.permute(&[0, 1, 2, 4, 3])
            };
            let last = phi.slice(1, seq - 1, 1)?.reshape(vec![
                batch,
                heads,
                d_state / 2,
            ])?;
            (rotate(inputs.b)?, rotate(inputs.c)?, Some(last))
        }
    };

    // --- boundary value carried in --------------------------------------
    // S = h + (1 - lambda_0) dt_0 * last_u restores the term the trapezoidal rule
    // still owes from the previous window.
    let carry = match inputs.state {
        None => None,
        Some(state) => {
            let first = inputs
                .lambda
                .slice(1, 0, 1)?
                .rsub_scalar(1.0)
                .mul(&inputs.dt.slice(1, 0, 1)?)?
                .reshape(vec![batch, heads, 1, 1])?;
            Some(state.h.add(&state.last_u.mul(&first)?)?)
        }
    };

    // --- the scan -------------------------------------------------------
    // MIMO decomposes exactly into R^2 SISO systems that share `a`, `g` and `w`:
    // the states add (`h = sum_j h^(j)`) and each output reads the shared state
    // with its own `C^(i)`. This is the reference path; a rank-aware kernel would
    // fold the `R^2` matmuls into one.
    let mut outputs: Vec<Var<R, E>> = Vec::with_capacity(rank);
    let mut total_state: Option<Var<R, E>> = None;

    for i in 0..rank {
        let c_i = c_rot.slice(4, i, 1)?.reshape(vec![
            batch,
            seq,
            heads,
            d_state,
        ])?;
        let mut acc: Option<Var<R, E>> = None;
        for j in 0..rank {
            let b_j = b_rot.slice(4, j, 1)?.reshape(vec![
                batch,
                seq,
                heads,
                d_state,
            ])?;
            let x_j = x.slice(4, j, 1)?.reshape(vec![
                batch,
                seq,
                heads,
                head_dim,
            ])?;
            let (y_ij, state_j) = ssd_chunked(
                &x_j,
                &b_j,
                &c_i,
                &a,
                &g,
                &w,
                carry.as_ref().filter(|_| j == 0),
                inputs.chunk_size,
            )?;
            acc = Some(match acc {
                Some(prev) => prev.add(&y_ij)?,
                None => y_ij,
            });
            if i == 0 {
                total_state = Some(match total_state {
                    Some(prev) => prev.add(&state_j)?,
                    None => state_j,
                });
            }
        }
        outputs.push(acc.expect("rank >= 1").unsqueeze(4)?);
    }

    let mut y = cat(&outputs, 4)?;

    // --- skip path -------------------------------------------------------
    if let Some(d) = inputs.d_skip {
        let gain = d.reshape(vec![1, 1, heads, 1, 1])?;
        y = y.add(&x.mul(&gain)?)?;
    }

    // --- state to carry --------------------------------------------------
    let h = total_state.unwrap_or_else(|| {
        Var::constant(Tensor::zeros(
            vec![batch, heads, head_dim, d_state],
            &device,
        ))
    });
    let last_u = last_outer_product(x, &b_rot, seq)?;

    Ok(ScanOutput {
        y,
        state: SsmState {
            h,
            last_u,
            angle: end_angle,
        },
    })
}

/// `sum_j B_j x_j^T` at the final position, shaped `[batch, heads, head_dim, d_state]`.
fn last_outer_product<R: Runtime, E: FloatElem>(
    x: &Var<R, E>,
    b: &Var<R, E>,
    seq: usize,
) -> Result<Var<R, E>> {
    let dims = x.dims().to_vec();
    let (batch, heads, head_dim, rank) = (dims[0], dims[2], dims[3], dims[4]);
    let d_state = b.dims()[3];
    let x_last = x
        .slice(1, seq - 1, 1)?
        .reshape(vec![batch * heads, head_dim, rank])?;
    let b_last = b
        .slice(1, seq - 1, 1)?
        .reshape(vec![batch * heads, d_state, rank])?;
    x_last
        .matmul(&b_last.transpose()?)?
        .reshape(vec![batch, heads, head_dim, d_state])
}

/// One decoding step of the Mamba-3 recurrence, written directly.
///
/// This is the equation the chunked scan implements, evaluated for a single token.
/// Keeping it separate makes the incremental path cheap (no chunk machinery) and
/// gives the test suite an independent implementation to check the scan against.
///
/// Shapes match [`ScanInputs`] with `seq = 1` and no leading time axis:
/// `x`, `b`, `c`: `[batch, heads, *, rank]`; `dt`, `lambda`: `[batch, heads]`;
/// `theta`: `[batch, heads, d_state / 2]`.
pub fn mamba3_step<R: Runtime, E: FloatElem>(
    x: &Var<R, E>,
    b: &Var<R, E>,
    c: &Var<R, E>,
    dt: &Var<R, E>,
    lambda: &Var<R, E>,
    a_log: &Var<R, E>,
    theta: Option<&Var<R, E>>,
    d_skip: Option<&Var<R, E>>,
    state: &SsmState<R, E>,
) -> Result<(Var<R, E>, SsmState<R, E>)> {
    let dims = x.dims().to_vec();
    let (batch, heads, head_dim, rank) = (dims[0], dims[1], dims[2], dims[3]);
    let d_state = b.dims()[2];

    // alpha, beta and g in one launch, packed as [3, batch * heads].
    let coefficients = Var::ssm_coefficients(a_log, dt, lambda)?;

    // Advance the rotating frame, then map B and C into it.
    let (b_rot, c_rot, angle) = match theta {
        None => (b.clone(), c.clone(), None),
        Some(theta) => {
            // One launch for the angle, then one per rotation: the sine and cosine
            // never become tensors of their own.
            let phi = Var::ssm_angle(dt, theta, state.angle.as_ref())?;
            let table = phi.unsqueeze(2)?;
            let rotate = |v: &Var<R, E>| -> Result<Var<R, E>> {
                let moved = v.permute(&[0, 1, 3, 2])?;
                moved.rotate_by_angle(&table)?.permute(&[0, 1, 3, 2])
            };
            (rotate(b)?, rotate(c)?, Some(phi))
        }
    };

    // u = sum_j B^(j) (x^(j))^T
    let u = x
        .reshape(vec![batch * heads, head_dim, rank])?
        .matmul(
            &b_rot
                .reshape(vec![batch * heads, d_state, rank])?
                .transpose()?,
        )?
        .reshape(vec![batch, heads, head_dim, d_state])?;

    // h <- alpha h + beta last_u + g u, in one launch rather than five.
    let h = Var::ssm_state_update([&state.h, &state.last_u, &u], &coefficients)?;

    // y^(i) = (C^(i))^T h
    let mut y = c_rot
        .reshape(vec![batch * heads, d_state, rank])?
        .transpose()?
        .matmul(&h.reshape(vec![batch * heads, head_dim, d_state])?.transpose()?)?
        .reshape(vec![batch, heads, rank, head_dim])?
        .permute(&[0, 1, 3, 2])?;

    if let Some(d) = d_skip {
        y = y.add(&x.mul(&d.reshape(vec![1, heads, 1, 1])?)?)?;
    }

    Ok((
        y,
        SsmState {
            h,
            last_u: u,
            angle,
        },
    ))
}
