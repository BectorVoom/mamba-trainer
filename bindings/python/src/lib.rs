//! Python bindings for the reinforcement learning half of [`mamba3`].
//!
//! A Mamba-3 policy is a good fit for reinforcement learning for one structural
//! reason: acting and learning want opposite shapes, and a state space model is
//! the rare architecture that is cheap at both. Acting wants the smallest possible
//! step — one observation, `B` environments, latency that does not grow with the
//! episode. Learning wants the largest possible batch — a whole `[B, T]`
//! trajectory through one wide parallel pass. A transformer can only do the first
//! by carrying a cache that grows with every action it takes; here the state is a
//! fixed `[layers, envs, heads, head_dim, d_state]` buffer that is overwritten in
//! place, and the same window replays through an `O(T)` scan of `O(log T)` depth.
//!
//! This module is that pair, plus the two loops built on it:
//!
//! ```python
//! import mamba3_rl as m3
//!
//! env = m3.RecallEnv(num_envs=32, symbols=4, horizon=4)
//! policy = m3.Policy(m3.PolicyConfig(env.obs_dim, env.action_dim, d_model=64, n_layers=2))
//! learner = m3.PpoLearner(policy, env, steps=16, learning_rate=1e-3)
//!
//! for stats in learner.run(rounds=80, epochs=4):
//!     print(stats.round, stats.episode_return)
//! ```
//!
//! # What is fixed at build time
//!
//! The backend and the element type. `mamba3` is generic over both, but a Python
//! extension module is a compiled artifact: it exports one set of classes, and
//! those classes name one runtime — whichever feature the wheel was built with —
//! and `f32`. Build for a GPU with
//! `maturin develop --release --no-default-features --features cuda`, and
//! [`backend`] reports what the wheel actually got.
//!
//! # Precision
//!
//! [`set_matmul_precision`] sets the storage precision of matrix products at
//! runtime. The environment variable `MAMBA3_MATMUL_PRECISION` (`"f32"`,
//! `"bf16"` or `"f16"`, case-insensitive) sets the same mode once, at import
//! time, checked against the compiled backend the same way `set_matmul_precision`
//! is — an unset variable touches no device and leaves the `f32` default alone;
//! a set one that is unrecognised, or that this backend cannot compile, fails
//! `import mamba3_rl` itself rather than silently falling back or being ignored.
//! Precision is one process-global value: a later successful call to
//! `set_matmul_precision` overrides whatever the environment set.

#![warn(missing_docs)]
#![allow(clippy::too_many_arguments)]

mod array;
mod config;
mod env;
mod err;
mod learner;
mod policy;
mod session;

use pyo3::prelude::*;

#[cfg(not(any(feature = "cpu", feature = "wgpu", feature = "cuda", feature = "hip")))]
compile_error!(
    "mamba3-rl needs a backend: build with one of the features cpu, wgpu, vulkan, \
     msl, cuda or hip"
);

/// The runtime every object in this module is bound to, resolved from the feature
/// the wheel was built with.
pub(crate) type R = mamba3::backends::Auto;

/// The element type weights, activations and observations are held in.
///
/// `f32` throughout. The crate supports `f16` and `bf16` elements, but mixing
/// element types inside one extension module would mean two of every class, and
/// the knob that actually pays on this workload is
/// [`set_matmul_precision`] — which keeps `f32` master weights and gradients and
/// only rounds what the product kernels read.
pub(crate) type E = f32;

/// The backend this module was compiled against, e.g. `"cpu"` or `"cuda"`.
#[pyfunction]
fn backend() -> &'static str {
    mamba3::backend::Device::<R>::default().name()
}

/// The storage precision matrix products read their operands at.
#[pyfunction]
fn matmul_precision() -> &'static str {
    use mamba3::tensor::ops::matmul::MatmulPrecision;
    match mamba3::tensor::ops::matmul::matmul_precision() {
        MatmulPrecision::F32 => "f32",
        MatmulPrecision::Bf16 => "bf16",
        MatmulPrecision::F16 => "f16",
    }
}

/// Set the storage precision of matrix products: `"f32"`, `"bf16"` or `"f16"`
/// (case-insensitive).
///
/// A semantic knob, and the only one: master weights, gradients and accumulation
/// stay `f32`, and only what the product kernels *read* is rounded — the
/// mixed-precision recipe, needing no loss scaling. It halves the bytes a matmul
/// moves, which is the ceiling these products are at.
///
/// The environment variable `MAMBA3_MATMUL_PRECISION` sets the same mode at
/// import time, checked the same way; an explicit call here that succeeds
/// afterwards simply overrides it, since the mode is one process-global value.
#[pyfunction]
fn set_matmul_precision(precision: &str) -> PyResult<()> {
    use mamba3::tensor::ops::matmul::{parse_matmul_precision, try_set_matmul_precision};
    let parsed = parse_matmul_precision(precision).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "unknown precision {precision:?}; expected 'f32', 'bf16' or 'f16'"
        ))
    })?;
    // Checked, not stored blind: WGSL has no `bf16` type, and a mode it cannot
    // compile aborts inside the shader compiler on a worker thread, once per
    // launch, for the rest of the run.
    try_set_matmul_precision(&mamba3::backend::Device::<R>::default(), parsed)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

/// How many times the host has read a device buffer since the counter was reset.
///
/// The number a collection loop is judged by: it should not grow while a window is
/// collected, however long the window is. It does grow for an environment written
/// in Python, once per step, which is the price of that choice made visible.
#[pyfunction]
fn read_count() -> usize {
    mamba3::backend::read_count()
}

/// Reset the counter [`read_count`] reports.
#[pyfunction]
fn reset_read_count() {
    mamba3::backend::reset_read_count();
}

/// Block until every queued kernel has completed.
///
/// Only needed for timing: every value that crosses back into Python already waits
/// for the work behind it.
#[pyfunction]
fn synchronize() {
    mamba3::backend::Device::<R>::default().synchronize();
}

#[pymodule]
fn _mamba3_rl(module: &Bound<'_, PyModule>) -> PyResult<()> {
    // Checked before anything else registers: an unset `MAMBA3_MATMUL_PRECISION`
    // touches no device and costs nothing, but a *set* one that is unrecognised
    // or unsupported on this backend fails `import mamba3_rl` itself with an
    // actionable message, rather than installing an unsupported mode that later
    // aborts a worker thread the first time a kernel actually reads it.
    mamba3::tensor::ops::matmul::try_set_precision_from_env::<R>()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    module.add_class::<config::PyPolicyConfig>()?;
    module.add_class::<config::PyPpoConfig>()?;
    module.add_class::<config::PyLrSchedule>()?;
    module.add_class::<policy::PyPolicy>()?;
    module.add_class::<policy::PyRollout>()?;
    module.add_class::<env::PyRecallEnv>()?;
    module.add_class::<learner::PyPpoLearner>()?;
    module.add_class::<learner::PyImitationLearner>()?;
    module.add_class::<learner::PyDaggerSchedule>()?;
    module.add_class::<learner::Stats>()?;
    module.add_class::<learner::CloneStats>()?;
    module.add_function(wrap_pyfunction!(learner::evaluate, module)?)?;
    module.add_function(wrap_pyfunction!(backend, module)?)?;
    module.add_function(wrap_pyfunction!(matmul_precision, module)?)?;
    module.add_function(wrap_pyfunction!(set_matmul_precision, module)?)?;
    module.add_function(wrap_pyfunction!(read_count, module)?)?;
    module.add_function(wrap_pyfunction!(reset_read_count, module)?)?;
    module.add_function(wrap_pyfunction!(synchronize, module)?)?;
    Ok(())
}
