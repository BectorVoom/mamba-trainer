//! Random tensors.
//!
//! Parameter initialisation is generated on the host with a seeded [`rand`] RNG so
//! that a run is reproducible regardless of backend. Dropout masks, which are
//! needed every step and are large, are generated on device with a counter-based
//! hash so no host round trip is involved.

use cubecl::prelude::*;
use rand::{Rng as _, SeedableRng};
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

use crate::backend::{Device, FloatElem, launch_1d};
use crate::tensor::base::Tensor;
use crate::tensor::shape::Shape;

/// Seeded host RNG used for weight initialisation and data shuffling.
#[derive(Debug)]
pub struct Rng {
    inner: StdRng,
}

impl Rng {
    /// Create an RNG from an explicit seed.
    pub fn seeded(seed: u64) -> Self {
        Self {
            inner: StdRng::seed_from_u64(seed),
        }
    }

    /// Create an RNG from entropy.
    pub fn from_entropy() -> Self {
        Self {
            inner: StdRng::from_rng(&mut rand::rng()),
        }
    }

    /// Draw a `u64`, e.g. to seed a device-side dropout mask.
    pub fn next_u64(&mut self) -> u64 {
        self.inner.random()
    }

    /// Draw a value in `[0, 1)`.
    pub fn next_f32(&mut self) -> f32 {
        self.inner.random()
    }

    /// Draw an index in `[0, n)`.
    pub fn next_index(&mut self, n: usize) -> usize {
        self.inner.random_range(0..n)
    }

    /// Shuffle a slice in place.
    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = self.inner.random_range(0..=i);
            items.swap(i, j);
        }
    }

    /// Normal samples.
    pub fn normal_vec(&mut self, n: usize, mean: f32, std: f32) -> Vec<f32> {
        if std <= 0.0 {
            return vec![mean; n];
        }
        let dist = Normal::new(mean, std).expect("std > 0");
        (0..n).map(|_| dist.sample(&mut self.inner)).collect()
    }

    /// Uniform samples in `[lo, hi)`.
    pub fn uniform_vec(&mut self, n: usize, lo: f32, hi: f32) -> Vec<f32> {
        if hi <= lo {
            return vec![lo; n];
        }
        let dist = Uniform::new(lo, hi).expect("hi > lo");
        (0..n).map(|_| dist.sample(&mut self.inner)).collect()
    }
}

impl Default for Rng {
    fn default() -> Self {
        Self::seeded(0)
    }
}

/// A normally distributed tensor.
pub fn randn<R: Runtime, E: FloatElem>(
    shape: impl Into<Shape>,
    mean: f32,
    std: f32,
    device: &Device<R>,
    rng: &mut Rng,
) -> Tensor<R, E> {
    let shape = shape.into();
    let data = rng.normal_vec(shape.num_elements(), mean, std);
    Tensor::from_f32(&data, shape, device).expect("generated data fills the shape")
}

/// A uniformly distributed tensor over `[lo, hi)`.
pub fn uniform<R: Runtime, E: FloatElem>(
    shape: impl Into<Shape>,
    lo: f32,
    hi: f32,
    device: &Device<R>,
    rng: &mut Rng,
) -> Tensor<R, E> {
    let shape = shape.into();
    let data = rng.uniform_vec(shape.num_elements(), lo, hi);
    Tensor::from_f32(&data, shape, device).expect("generated data fills the shape")
}

pub use hash::{hash_u32, hash_unit};

/// The crate's stateless device-side hash, and the unit draw built on it.
///
/// A module of its own only so that one `missing_docs` allow covers it: `#[cube]`
/// emits a companion module and an expand function beside every function it
/// touches, and neither has anywhere to hang a doc comment. Both items are
/// re-exported, so callers spell them `random::hash_u32` as before.
#[allow(missing_docs)]
pub mod hash {
    use cubecl::prelude::*;

    /// SplitMix-style avalanche on `(index, seed)`.
    ///
    /// Cheap, decorrelated enough for the decisions it makes, and **stateless**: a
    /// unit derives its draw from its own position, so no kernel has to carry a
    /// generator and no two launches share a stream. That is what lets every
    /// randomised kernel in the crate — the dropout mask, the action sampler and
    /// DAgger's coin flip in [`crate::tensor::ops::rl`], the cue draw in
    /// [`crate::rl::env`] — be a pure function of where it runs, and it is what a
    /// [`crate::rl::GameLogic`] should draw with for the same reason. Not a
    /// cryptographic generator, and it does not need to be.
    #[cube]
    pub fn hash_u32(index: u32, seed_lo: u32, seed_hi: u32) -> u32 {
        let mut h = index ^ seed_lo;
        h ^= h >> 16;
        h = h * 0x7feb352du32;
        h ^= h >> 15;
        h = h * 0x846ca68bu32;
        h ^= seed_hi;
        h ^= h >> 16;
        h
    }

    /// A draw in `[0, 1)` from [`hash_u32`].
    ///
    /// Twenty-four bits, which is an `f32`'s mantissa: every value the unit interval
    /// can distinguish at this width, and no value it cannot.
    #[cube]
    pub fn hash_unit<F: Float + CubeElement>(index: u32, seed_lo: u32, seed_hi: u32) -> F {
        F::cast_from(hash_u32(index, seed_lo, seed_hi) >> 8) / F::new(16777216.0_f32)
    }
}

/// Deliberately scalar: the draw is a hash of `ABSOLUTE_POS`, so widening a unit to
/// a vector would hand every lane in it the same coin.
#[cube(launch_unchecked)]
fn bernoulli_kernel<F: Float + CubeElement>(output: &mut Array<F>, seed_lo: u32, seed_hi: u32, keep: F, scale: F) {
    if ABSOLUTE_POS < output.len() {
        let unit = hash_unit::<F>(ABSOLUTE_POS as u32, seed_lo, seed_hi);
        let mut v = F::new(0.0_f32);
        if unit < keep {
            v = scale;
        }
        output[ABSOLUTE_POS] = v;
    }
}

/// A dropout mask: `1/(1-p)` with probability `1-p`, otherwise `0`.
pub fn dropout_mask<R: Runtime, E: FloatElem>(
    shape: impl Into<Shape>,
    p: f32,
    seed: u64,
    device: &Device<R>,
) -> Tensor<R, E> {
    let shape = shape.into();
    let out = Tensor::<R, E>::empty(shape, device);
    let n = out.len();
    if n == 0 {
        return out;
    }
    let keep = 1.0 - p;
    let (count, dim) = launch_1d(out.client(), n, 1);
    unsafe {
        bernoulli_kernel::launch_unchecked::<E, R>(
            out.client(),
            count,
            dim,
            out.arg(),
            seed as u32,
            (seed >> 32) as u32,
            E::from_scalar(keep),
            E::from_scalar(if keep > 0.0 { 1.0 / keep } else { 0.0 }),
        );
    }
    out
}
