//! Python surface for the trained completion-model generation API.
//!
//! Thin marshal over
//! `mamba3::models::ms2::completion_api`: the JSON-in/JSON-out protocol
//! `molecular-completion-generate-v1` lives in Rust, Python only carries the
//! request/response strings. There is no Python-side sampler that could drift.

use std::path::Path;

use mamba3::backend::Device;
use mamba3::models::ms2::completion_api::{CompletionService, GENERATE_PROTOCOL};
use pyo3::prelude::*;

use crate::R;
use crate::err::{IntoPyResult, to_py};

/// A value carried into a section that runs without the interpreter lock.
struct Detached<T>(T);

// SAFETY: as in `ms2.rs`: the closure runs on the calling thread, captures no
// `Bound`/`Py`/`PyRef`, touches no Python object without the lock, and the
// `unsendable` class refuses access from any other thread while the borrow is
// held here.
unsafe impl<T> Send for Detached<T> {}

/// Run `body` with the interpreter lock released, on this thread.
fn detached<T>(py: Python<'_>, body: impl FnOnce() -> T) -> T {
    let body = Detached(body);
    py.detach(move || {
        let body = body;
        Detached((body.0)())
    })
    .0
}

thread_local! {
    /// The device every object of this thread lives on.
    static DEVICE: Device<R> = Device::<R>::default();
}

/// This thread's device.
fn device() -> Device<R> {
    DEVICE.with(Clone::clone)
}

/// Protocol version of the trained-model generation API.
#[pyfunction]
fn molecular_completion_generate_protocol() -> &'static str {
    GENERATE_PROTOCOL
}

/// A loaded trained completion model: `describe` and `generate` are thin
/// marshals over the Rust service.
#[pyclass(module = "mamba3_rl", name = "MolecularCompletionModel", unsendable)]
pub struct PyMolecularCompletionModel {
    inner: CompletionService<R>,
}

#[pymethods]
impl PyMolecularCompletionModel {
    /// Load a checkpoint saved by the fixture example onto this thread's
    /// device. The interpreter lock is released while the weights upload.
    #[new]
    #[pyo3(signature = (checkpoint_path))]
    fn new(py: Python<'_>, checkpoint_path: &str) -> PyResult<Self> {
        let device = device();
        let inner = detached(py, || {
            CompletionService::load(Path::new(checkpoint_path), &device)
        })
        .py()?;
        Ok(Self { inner })
    }

    /// Describe the loaded model as JSON (protocol, versions, config, domain
    /// limits, checkpoint SHA-256, trained steps).
    fn describe(&self) -> String {
        self.inner.describe()
    }

    /// Run one strict `molecular-completion-generate-v1` request and return
    /// the response JSON. The interpreter lock is released for the whole
    /// call. Malformed requests raise `ValueError` with an actionable
    /// message; an `unsupported_input` status is carried in the body, never
    /// as an exception.
    #[pyo3(signature = (request_json))]
    fn generate(&self, py: Python<'_>, request_json: &str) -> PyResult<String> {
        detached(py, || self.inner.generate_json(request_json)).map_err(to_py)
    }
}

/// Register the model class and protocol function on `_mamba3_rl`.
pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyMolecularCompletionModel>()?;
    module.add_function(wrap_pyfunction!(
        molecular_completion_generate_protocol,
        module
    )?)?;
    Ok(())
}
