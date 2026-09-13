//! [`MultivariateNormal`]: a Gaussian with a full covariance.
//!
//! Parameterised by the Cholesky factor `L` of the covariance, because that is the
//! form every operation actually wants: the density needs `L⁻¹(x − μ)`, the entropy
//! and the log-determinant need `Σ ln Lᵢᵢ`, and a draw is `μ + Lε`. A caller who has
//! a covariance matrix instead gets it factored once, at construction, by
//! [`MultivariateNormal::from_covariance`] — which is also the only place the
//! matrix's positive-definiteness could be discovered, and the only place it is
//! worth spending an `O(d³)` kernel.
//!
//! # Shape of the work
//!
//! One unit per batch element, walking its own `d × d` triangle. That is the right
//! decomposition while `d` is what it is for control — a joint count, tens at most —
//! because the alternative, one unit per matrix element, needs a barrier between
//! every row of a triangular solve. It is the wrong decomposition for `d` in the
//! hundreds, and the docs say so rather than pretending otherwise.
//!
//! # What is composed rather than written
//!
//! Only the triangular solves and the factorisation are kernels. The draw is
//! `loc + tril @ ε` and the entropy is a sum of logs of a diagonal, both of which
//! [`crate::autograd`] already differentiates — so `rsample` is reparameterised for
//! free, and its gradient flows into `scale_tril` without a line of calculus here.

// See the note in `univariate.rs` on the modules `#[cube]` synthesises.
#![allow(missing_docs, unused_assignments)]

use cubecl::prelude::*;

use crate::autograd::Var;
use crate::autograd::ops::reduce_grad_to;
use crate::backend::{Device, FloatElem, launch_1d};
use crate::error::{Error, Result};
use crate::tensor::base::Tensor;
use crate::tensor::shape::Shape;

use super::special;
use super::univariate;
use super::{Distribution, Support};

/// `L z = δ` by forward substitution, writing `z` into `scratch`.
#[cube]
fn solve_lower<E: Float + CubeElement>(
    tril: &Array<E>,
    scratch: &mut Array<E>,
    mat: usize,
    vec: usize,
    dim: u32,
) {
    let mut i: u32 = 0;
    while i < dim {
        let mut acc = f32::cast_from(scratch[vec + i as usize]);
        let mut j: u32 = 0;
        while j < i {
            acc -= f32::cast_from(tril[mat + (i * dim + j) as usize])
                * f32::cast_from(scratch[vec + j as usize]);
            j += 1u32;
        }
        scratch[vec + i as usize] =
            E::cast_from(acc / f32::cast_from(tril[mat + (i * dim + i) as usize]));
        i += 1u32;
    }
}

/// `Lᵀ w = z` by back substitution, in place over `scratch`.
#[cube]
fn solve_upper<E: Float + CubeElement>(
    tril: &Array<E>,
    scratch: &mut Array<E>,
    mat: usize,
    vec: usize,
    dim: u32,
) {
    let mut back: u32 = 0;
    while back < dim {
        let i = dim - 1u32 - back;
        let mut acc = f32::cast_from(scratch[vec + i as usize]);
        let mut j = i + 1u32;
        while j < dim {
            acc -= f32::cast_from(tril[mat + (j * dim + i) as usize])
                * f32::cast_from(scratch[vec + j as usize]);
            j += 1u32;
        }
        scratch[vec + i as usize] =
            E::cast_from(acc / f32::cast_from(tril[mat + (i * dim + i) as usize]));
        back += 1u32;
    }
}

/// `−½‖L⁻¹(x − μ)‖² − Σ ln Lᵢᵢ − (d/2) ln 2π`.
#[cube(launch_unchecked)]
fn mvn_log_prob_kernel<E: Float + CubeElement>(
    loc: &Array<E>,
    tril: &Array<E>,
    value: &Array<E>,
    scratch: &mut Array<E>,
    out: &mut Array<E>,
    rows: u32,
    dim: u32,
    param_rows: u32,
) {
    if ABSOLUTE_POS < rows as usize {
        let row = ABSOLUTE_POS;
        let prow = row % param_rows as usize;
        let vec = row * dim as usize;
        let pvec = prow * dim as usize;
        let mat = prow * (dim * dim) as usize;

        let mut i: u32 = 0;
        while i < dim {
            scratch[vec + i as usize] = E::cast_from(
                f32::cast_from(value[vec + i as usize]) - f32::cast_from(loc[pvec + i as usize]),
            );
            i += 1u32;
        }
        solve_lower::<E>(tril, scratch, mat, vec, dim);

        let mut quad: f32 = 0.0;
        let mut logdet: f32 = 0.0;
        let mut k: u32 = 0;
        while k < dim {
            let z = f32::cast_from(scratch[vec + k as usize]);
            quad += z * z;
            logdet += f32::ln(f32::abs(f32::cast_from(tril[mat + (k * dim + k) as usize])));
            k += 1u32;
        }
        out[row] =
            E::cast_from(-0.5f32 * quad - logdet - f32::cast_from(dim) * special::HALF_LN_2PI);
    }
}

/// The adjoint of `mvn_log_prob_kernel`, for the value, the location and the
/// factor.
///
/// With `z = L⁻¹(x − μ)` and `w = L⁻ᵀz`, the three derivatives are `−w`, `+w` and
/// `tril(w zᵀ) − diag(1/Lᵢᵢ)`. Both solves are in place over the same scratch row:
/// the back substitution reads `zᵢ` exactly once, immediately before overwriting it.
#[cube(launch_unchecked)]
fn mvn_grad_kernel<E: Float + CubeElement>(
    loc: &Array<E>,
    tril: &Array<E>,
    value: &Array<E>,
    upstream: &Array<E>,
    scratch: &mut Array<E>,
    zbuf: &mut Array<E>,
    grad_value: &mut Array<E>,
    grad_loc: &mut Array<E>,
    grad_tril: &mut Array<E>,
    rows: u32,
    dim: u32,
    param_rows: u32,
    #[comptime] want_value: bool,
    #[comptime] want_loc: bool,
    #[comptime] want_tril: bool,
) {
    if ABSOLUTE_POS < rows as usize {
        let row = ABSOLUTE_POS;
        let prow = row % param_rows as usize;
        let vec = row * dim as usize;
        let pvec = prow * dim as usize;
        let mat = prow * (dim * dim) as usize;
        let g = f32::cast_from(upstream[row]);

        let mut i: u32 = 0;
        while i < dim {
            scratch[vec + i as usize] = E::cast_from(
                f32::cast_from(value[vec + i as usize]) - f32::cast_from(loc[pvec + i as usize]),
            );
            i += 1u32;
        }
        solve_lower::<E>(tril, scratch, mat, vec, dim);
        let mut c: u32 = 0;
        while c < dim {
            zbuf[vec + c as usize] = scratch[vec + c as usize];
            c += 1u32;
        }
        solve_upper::<E>(tril, scratch, mat, vec, dim);

        let mut k: u32 = 0;
        while k < dim {
            let w = f32::cast_from(scratch[vec + k as usize]);
            if comptime!(want_value) {
                grad_value[vec + k as usize] = E::cast_from(-g * w);
            }
            if comptime!(want_loc) {
                grad_loc[vec + k as usize] = E::cast_from(g * w);
            }
            k += 1u32;
        }
        if comptime!(want_tril) {
            let mut r: u32 = 0;
            while r < dim {
                let w = f32::cast_from(scratch[vec + r as usize]);
                let mut col: u32 = 0;
                while col < dim {
                    let mut v: f32 = 0.0;
                    if col <= r {
                        v = w * f32::cast_from(zbuf[vec + col as usize]);
                        if col == r {
                            v -= 1.0f32 / f32::cast_from(tril[mat + (r * dim + r) as usize]);
                        }
                    }
                    grad_tril[row * (dim * dim) as usize + (r * dim + col) as usize] =
                        E::cast_from(g * v);
                    col += 1u32;
                }
                r += 1u32;
            }
        }
    }
}

/// The Cholesky factorisation `A = L Lᵀ`, one matrix per unit.
///
/// The textbook `O(d³)` recurrence. A non-positive pivot means the matrix was not
/// positive definite; rather than failing — which a device kernel cannot usefully do
/// — the pivot is floored at a tiny positive number and the caller is told, in the
/// docs of [`MultivariateNormal::from_covariance`], that the result is then
/// meaningless. Detecting it would cost a read back on every construction.
#[cube(launch_unchecked)]
fn cholesky_kernel<E: Float + CubeElement>(
    cov: &Array<E>,
    out: &mut Array<E>,
    rows: u32,
    dim: u32,
) {
    if ABSOLUTE_POS < rows as usize {
        let mat = ABSOLUTE_POS * (dim * dim) as usize;
        let mut j: u32 = 0;
        while j < dim {
            let mut diag = f32::cast_from(cov[mat + (j * dim + j) as usize]);
            let mut k: u32 = 0;
            while k < j {
                let v = f32::cast_from(out[mat + (j * dim + k) as usize]);
                diag -= v * v;
                k += 1u32;
            }
            let mut pivot = f32::sqrt(diag);
            // Negated rather than `pivot <= tiny`, so that a `NaN` pivot — which is
            // what a badly conditioned matrix produces — is floored too instead of
            // propagating through every later column.
            #[allow(clippy::neg_cmp_op_on_partial_ord)]
            if !(pivot > 1.0e-20f32) {
                pivot = 1.0e-20f32;
            }
            out[mat + (j * dim + j) as usize] = E::cast_from(pivot);

            let mut i = j + 1u32;
            while i < dim {
                let mut acc = f32::cast_from(cov[mat + (i * dim + j) as usize]);
                let mut m: u32 = 0;
                while m < j {
                    acc -= f32::cast_from(out[mat + (i * dim + m) as usize])
                        * f32::cast_from(out[mat + (j * dim + m) as usize]);
                    m += 1u32;
                }
                out[mat + (i * dim + j) as usize] = E::cast_from(acc / pivot);
                i += 1u32;
            }
            // The strict upper triangle is not part of the factor.
            let mut z = j + 1u32;
            while z < dim {
                out[mat + (j * dim + z) as usize] = E::cast_from(0.0f32);
                z += 1u32;
            }
            j += 1u32;
        }
    }
}

/// `KL(N(μ₁, L₁L₁ᵀ) ‖ N(μ₂, L₂L₂ᵀ))`, one batch element per unit.
#[cube(launch_unchecked)]
fn mvn_kl_kernel<E: Float + CubeElement>(
    loc_p: &Array<E>,
    tril_p: &Array<E>,
    loc_q: &Array<E>,
    tril_q: &Array<E>,
    scratch: &mut Array<E>,
    out: &mut Array<E>,
    rows: u32,
    dim: u32,
) {
    if ABSOLUTE_POS < rows as usize {
        let row = ABSOLUTE_POS;
        let vec = row * dim as usize;
        let mat = row * (dim * dim) as usize;

        // `‖L₂⁻¹(μ₁ − μ₂)‖²`.
        let mut i: u32 = 0;
        while i < dim {
            scratch[vec + i as usize] = E::cast_from(
                f32::cast_from(loc_p[vec + i as usize]) - f32::cast_from(loc_q[vec + i as usize]),
            );
            i += 1u32;
        }
        solve_lower::<E>(tril_q, scratch, mat, vec, dim);
        let mut quad: f32 = 0.0;
        let mut k: u32 = 0;
        while k < dim {
            let v = f32::cast_from(scratch[vec + k as usize]);
            quad += v * v;
            k += 1u32;
        }

        // `‖L₂⁻¹L₁‖²_F`, one column of `L₁` at a time so the scratch row is reused.
        let mut trace: f32 = 0.0;
        let mut col: u32 = 0;
        while col < dim {
            let mut r: u32 = 0;
            while r < dim {
                scratch[vec + r as usize] = tril_p[mat + (r * dim + col) as usize];
                r += 1u32;
            }
            solve_lower::<E>(tril_q, scratch, mat, vec, dim);
            let mut s: u32 = 0;
            while s < dim {
                let v = f32::cast_from(scratch[vec + s as usize]);
                trace += v * v;
                s += 1u32;
            }
            col += 1u32;
        }

        let mut logdet: f32 = 0.0;
        let mut d: u32 = 0;
        while d < dim {
            logdet += f32::ln(f32::abs(f32::cast_from(
                tril_q[mat + (d * dim + d) as usize],
            ))) - f32::ln(f32::abs(f32::cast_from(
                tril_p[mat + (d * dim + d) as usize],
            )));
            d += 1u32;
        }
        out[row] = E::cast_from(0.5f32 * (trace + quad - f32::cast_from(dim)) + logdet);
    }
}

/// A Gaussian over `d` dimensions with a full covariance.
pub struct MultivariateNormal<R: Runtime, E: FloatElem = f32> {
    loc: Var<R, E>,
    scale_tril: Var<R, E>,
    batch: Shape,
    dim: usize,
}

impl<R: Runtime, E: FloatElem> Clone for MultivariateNormal<R, E> {
    fn clone(&self) -> Self {
        Self {
            loc: self.loc.clone(),
            scale_tril: self.scale_tril.clone(),
            batch: self.batch.clone(),
            dim: self.dim,
        }
    }
}

impl<R: Runtime, E: FloatElem> core::fmt::Debug for MultivariateNormal<R, E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "MultivariateNormal({} over {})", self.batch, self.dim)
    }
}

impl<R: Runtime, E: FloatElem> MultivariateNormal<R, E> {
    /// From a mean `[..batch, d]` and the lower-triangular Cholesky factor of the
    /// covariance, `[..batch, d, d]`.
    ///
    /// Only the lower triangle and the diagonal are read; whatever is above is
    /// ignored rather than checked.
    pub fn from_scale_tril(
        loc: impl Into<Var<R, E>>,
        scale_tril: impl Into<Var<R, E>>,
    ) -> Result<Self> {
        let loc = loc.into();
        let scale_tril = scale_tril.into();
        if loc.rank() == 0 {
            return Err(Error::shape(
                "a multivariate normal needs a trailing event axis".to_string(),
            ));
        }
        let dim = loc.shape().dim_from_end(0);
        let batch = loc.shape().without(loc.rank() - 1);
        let want: Vec<usize> = batch.dims().iter().copied().chain([dim, dim]).collect();
        if scale_tril.dims() != want.as_slice() {
            return Err(Error::shape(format!(
                "a mean of {} wants a factor of {}, got {}",
                loc.shape(),
                Shape::new(want),
                scale_tril.shape()
            )));
        }
        Ok(Self {
            loc,
            scale_tril,
            batch,
            dim,
        })
    }

    /// From a mean and a covariance matrix, factored here.
    ///
    /// The factorisation floors a non-positive pivot rather than failing, because
    /// noticing would mean reading the device back. A covariance that is not
    /// positive definite therefore produces a distribution whose densities are
    /// nonsense rather than an error — check the matrix, or build one that cannot be
    /// indefinite, such as `A Aᵀ + εI`.
    pub fn from_covariance(loc: impl Into<Var<R, E>>, covariance: &Tensor<R, E>) -> Result<Self> {
        let loc = loc.into();
        let dim = loc.shape().dim_from_end(0);
        let rows = covariance.len() / (dim * dim).max(1);
        let out = Tensor::<R, E>::zeros(covariance.shape().clone(), covariance.device());
        if rows > 0 {
            let (count, dim_launch) = launch_1d(covariance.client(), rows, dim * dim * dim / 3 + 1);
            unsafe {
                cholesky_kernel::launch_unchecked::<E, R>(
                    covariance.client(),
                    count,
                    dim_launch,
                    covariance.arg(),
                    out.arg(),
                    rows as u32,
                    dim as u32,
                );
            }
        }
        Self::from_scale_tril(loc, Var::constant(out))
    }

    /// A Gaussian with a diagonal covariance, given its per-axis standard deviation.
    ///
    /// Equivalent to `Independent(Normal(loc, scale), 1)` and slower than it; the
    /// diagonal case is worth reaching for that instead. Provided so a caller who
    /// needs one type for both can have it.
    pub fn diagonal(loc: impl Into<Var<R, E>>, scale: &Tensor<R, E>) -> Result<Self> {
        let loc = loc.into();
        let dim = loc.shape().dim_from_end(0);
        let device = scale.device();
        let eye = Tensor::<R, E>::eye(dim, device);
        let rows = scale.len() / dim.max(1);
        let spread = scale.reshape(Shape::new(vec![rows, dim, 1]))?;
        let mut dims = loc.shape().dims().to_vec();
        dims.push(dim);
        let tril = crate::tensor::ops::elemwise::mul(
            &crate::tensor::ops::elemwise::expand(&spread, &Shape::new(vec![rows, dim, dim]))?,
            &crate::tensor::ops::elemwise::expand(
                &eye.reshape(Shape::new(vec![1, dim, dim]))?,
                &Shape::new(vec![rows, dim, dim]),
            )?,
        )?;
        Self::from_scale_tril(loc, Var::constant(tril.reshape(Shape::new(dims))?))
    }

    /// The mean, as given.
    pub fn loc(&self) -> &Var<R, E> {
        &self.loc
    }

    /// The Cholesky factor, as given.
    pub fn scale_tril(&self) -> &Var<R, E> {
        &self.scale_tril
    }

    /// The event dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    fn rows(&self) -> usize {
        self.batch.num_elements()
    }

    fn device(&self) -> &Device<R> {
        self.loc.tensor().device()
    }

    /// `n` reparameterised draws in one pass, shaped `[n, ..batch, d]`.
    ///
    /// `μ + Lε` with `ε` standard normal. The noise does not depend on either
    /// parameter, so the whole expression is differentiable in both through
    /// operations [`crate::autograd`] already knows — no adjoint is written here.
    /// All `n·batch` noise values come from one launch, and the products from one
    /// batched matmul, so the cost of `n` draws is the cost of one.
    pub fn rsample_n(&self, n: usize, seed: u64) -> Result<Var<R, E>> {
        let rows = self.rows() * n;
        let device = self.device();
        let zero = Tensor::<R, E>::zeros(Shape::new(vec![1]), device);
        let one = Tensor::<R, E>::ones(Shape::new(vec![1]), device);
        let noise = univariate::sample(
            [&zero, &one, &one],
            Shape::new(vec![rows, self.dim, 1]),
            1,
            0,
            seed,
            super::Kind::Normal,
        )?;
        // The factor is shared across the sample axis, so it is broadcast rather
        // than repeated: `expand` materialises it once for the matmul, which is the
        // only place a `[n·batch, d, d]` operand is needed.
        let tril = self
            .scale_tril
            .reshape(Shape::new(vec![self.rows(), self.dim, self.dim]))?;
        let wide = if n == 1 {
            tril
        } else {
            tril.unsqueeze(0)?
                .expand(Shape::new(vec![n, self.rows(), self.dim, self.dim]))?
                .reshape(Shape::new(vec![rows, self.dim, self.dim]))?
        };
        let mut dims = if n == 1 { vec![] } else { vec![n] };
        dims.extend_from_slice(self.loc.shape().dims());
        let shifted = wide
            .matmul(&Var::constant(noise))?
            .reshape(Shape::new(dims.clone()))?;
        self.loc.add(&shifted)
    }

    /// The diagonal of the factor, `[..batch, d]`.
    fn diag(&self) -> Result<Var<R, E>> {
        let eye = Tensor::<R, E>::eye(self.dim, self.device());
        let mut mask_dims = vec![1usize; self.batch.rank()];
        mask_dims.extend_from_slice(&[self.dim, self.dim]);
        let mask = Var::constant(eye.reshape(Shape::new(mask_dims))?);
        let axis = self.scale_tril.rank() - 1;
        self.scale_tril.mul(&mask)?.sum_dim(axis)?.squeeze(axis)
    }
}

impl<R: Runtime, E: FloatElem> Distribution<R, E> for MultivariateNormal<R, E> {
    fn batch_shape(&self) -> &Shape {
        &self.batch
    }

    fn event_shape(&self) -> Shape {
        Shape::new(vec![self.dim])
    }

    fn support(&self) -> Support {
        Support::Real
    }

    fn has_rsample(&self) -> bool {
        true
    }

    fn sample(&self, seed: u64) -> Result<Tensor<R, E>> {
        Ok(self.rsample(seed)?.into_tensor())
    }

    fn sample_n(&self, n: usize, seed: u64) -> Result<Tensor<R, E>> {
        Ok(self.rsample_n(n, seed)?.into_tensor())
    }

    fn rsample(&self, seed: u64) -> Result<Var<R, E>> {
        self.rsample_n(1, seed)
    }

    fn log_prob(&self, value: &Var<R, E>) -> Result<Var<R, E>> {
        // As for a Dirichlet, the value may prepend a sample axis; it has only to
        // end in the event axis and be a whole number of rows.
        let ends_right = value.rank() >= 1 && value.shape().dim_from_end(0) == self.dim;
        if !ends_right || !value.tensor().len().is_multiple_of(self.loc.tensor().len()) {
            return Err(Error::shape(format!(
                "a multivariate normal over {} cannot score a value of {}",
                self.loc.shape(),
                value.shape()
            )));
        }
        let param_rows = self.rows();
        let rows = value.tensor().len() / self.dim;
        let device = self.device();
        let out = Tensor::<R, E>::empty(value.shape().without(value.rank() - 1), device);
        let scratch = Tensor::<R, E>::empty(value.shape().clone(), device);
        if rows > 0 {
            let (count, dim) = launch_1d(device.client(), rows, self.dim * self.dim);
            unsafe {
                mvn_log_prob_kernel::launch_unchecked::<E, R>(
                    device.client(),
                    count,
                    dim,
                    self.loc.tensor().arg(),
                    self.scale_tril.tensor().arg(),
                    value.tensor().arg(),
                    scratch.arg(),
                    out.arg(),
                    rows as u32,
                    self.dim as u32,
                    param_rows.max(1) as u32,
                );
            }
        }
        let owned = self.clone();
        let seen = value.tensor().clone();
        let value_shape = value.shape().clone();
        Ok(Var::record_with_mask(
            out,
            &[value, &self.loc, &self.scale_tril],
            |want| {
                let (wv, wl, wt) = (want[0], want[1], want[2]);
                Box::new(move |g| {
                    let device = owned.device();
                    let param_rows = owned.rows();
                    let rows = seen.len() / owned.dim;
                    let scratch = Tensor::<R, E>::empty(value_shape.clone(), device);
                    let zbuf = Tensor::<R, E>::empty(value_shape.clone(), device);
                    let tiny = Tensor::<R, E>::empty(Shape::new(vec![1]), device);
                    // Every gradient is the value's shape, and is reduced back onto
                    // its own parameter below — which is what sums a sample axis
                    // away when one is present.
                    let vec_like =
                        |on: bool| on.then(|| Tensor::<R, E>::empty(value_shape.clone(), device));
                    let gv = vec_like(wv);
                    let gl = vec_like(wl);
                    let mut tril_dims = value_shape.dims().to_vec();
                    tril_dims.push(owned.dim);
                    let gt = wt.then(|| Tensor::<R, E>::empty(Shape::new(tril_dims), device));
                    if rows > 0 {
                        let (count, dim) = launch_1d(device.client(), rows, owned.dim * owned.dim);
                        unsafe {
                            mvn_grad_kernel::launch_unchecked::<E, R>(
                                device.client(),
                                count,
                                dim,
                                owned.loc.tensor().arg(),
                                owned.scale_tril.tensor().arg(),
                                seen.arg(),
                                g.arg(),
                                scratch.arg(),
                                zbuf.arg(),
                                gv.as_ref().unwrap_or(&tiny).arg(),
                                gl.as_ref().unwrap_or(&tiny).arg(),
                                gt.as_ref().unwrap_or(&tiny).arg(),
                                rows as u32,
                                owned.dim as u32,
                                param_rows.max(1) as u32,
                                wv,
                                wl,
                                wt,
                            );
                        }
                    }
                    Ok(vec![
                        gv.map(|t| reduce_grad_to(&t, &value_shape)).transpose()?,
                        gl.map(|t| reduce_grad_to(&t, owned.loc.shape()))
                            .transpose()?,
                        gt.map(|t| reduce_grad_to(&t, owned.scale_tril.shape()))
                            .transpose()?,
                    ])
                })
            },
        ))
    }

    fn cdf(&self, _value: &Tensor<R, E>) -> Result<Tensor<R, E>> {
        Err(Error::Unsupported(
            "a multivariate normal has no elementary CDF".to_string(),
        ))
    }

    fn icdf(&self, _q: &Var<R, E>) -> Result<Var<R, E>> {
        Err(Error::Unsupported(
            "a multivariate normal has no elementary quantile".to_string(),
        ))
    }

    fn entropy(&self) -> Result<Var<R, E>> {
        // `Σ ln Lᵢᵢ + (d/2)(1 + ln 2π)`, composed so it differentiates itself.
        let axis = self.diag()?.rank() - 1;
        let logdet = self.diag()?.abs().log().sum_dim(axis)?.squeeze(axis)?;
        let constant = self.dim as f32 * (0.5 + special::HALF_LN_2PI);
        Ok(logdet.add_scalar(constant))
    }

    fn mean(&self) -> Result<Tensor<R, E>> {
        Ok(self.loc.tensor().clone())
    }

    fn variance(&self) -> Result<Tensor<R, E>> {
        // The diagonal of `L Lᵀ`, which is the row-wise sum of squares of `L`.
        let squared =
            crate::tensor::ops::elemwise::mul(self.scale_tril.tensor(), self.scale_tril.tensor())?;
        let axis = squared.rank() - 1;
        crate::tensor::ops::reduce::sum_dim(&squared, axis)?.squeeze(axis)
    }

    fn mode(&self) -> Result<Tensor<R, E>> {
        Ok(self.loc.tensor().clone())
    }
}

/// `KL(p ‖ q)` between two multivariate normals of the same event dimension.
pub fn kl_multivariate_normal<R: Runtime, E: FloatElem>(
    p: &MultivariateNormal<R, E>,
    q: &MultivariateNormal<R, E>,
) -> Result<Tensor<R, E>> {
    if p.dim != q.dim || p.batch != q.batch {
        return Err(Error::shape(format!(
            "cannot compare {p:?} with {q:?}: the shapes differ"
        )));
    }
    let device = p.device();
    let rows = p.rows();
    let out = Tensor::<R, E>::empty(p.batch.clone(), device);
    if rows == 0 {
        return Ok(out);
    }
    let scratch = Tensor::<R, E>::empty(p.loc.shape().clone(), device);
    let (count, dim) = launch_1d(device.client(), rows, p.dim * p.dim * p.dim);
    unsafe {
        mvn_kl_kernel::launch_unchecked::<E, R>(
            device.client(),
            count,
            dim,
            p.loc.tensor().arg(),
            p.scale_tril.tensor().arg(),
            q.loc.tensor().arg(),
            q.scale_tril.tensor().arg(),
            scratch.arg(),
            out.arg(),
            rows as u32,
            p.dim as u32,
        );
    }
    Ok(out)
}
