//! numpy on one side, device tensors on the other.
//!
//! Every conversion here is a copy, and every conversion *out* is also a
//! synchronisation: reading a tensor waits for the kernels that produce it. That
//! is why the learning loops in [`crate::learner`] do their whole round in Rust
//! and hand back a handful of scalars, rather than exposing the trajectory buffer
//! to Python one step at a time. The functions here are for the edges — an
//! observation coming in from an environment, actions going out to it — where a
//! copy of `envs * obs_dim` floats is unavoidable and small.
//!
//! Each conversion in takes a bare object and reports what it wanted, because
//! these are the errors a user meets while wiring an environment up: a `[envs]`
//! array where `[envs, obs_dim]` was promised is the most common mistake there is,
//! and `TypeError: 'ndarray' object is not an instance of 'ndarray'` — which is
//! what a bare extraction produces — helps nobody.

use mamba3::backend::Device;
use mamba3::tensor::Tensor;
use mamba3::tensor::ops::index::IdTensor;
use numpy::{
    AllowTypeChange, PyArray1, PyArray2, PyArrayLike1, PyArrayLike2, PyArrayMethods,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::{E, R};

/// A `[rows, cols]` float array, however it was spelled in Python.
type FloatArray2<'py> = PyArrayLike2<'py, f32, AllowTypeChange>;
/// A `[len]` float array.
type FloatArray1<'py> = PyArrayLike1<'py, f32, AllowTypeChange>;
/// A `[len]` integer array. `i64` is what numpy produces for a Python list of ints
/// and what `argmax` returns, so it is the type to meet callers at.
type IdArray1<'py> = PyArrayLike1<'py, i64, AllowTypeChange>;

/// The values of a read-only array, copying out of a strided view when the array
/// numpy handed over is not contiguous.
fn values<T, D>(array: &numpy::PyReadonlyArray<'_, T, D>) -> Vec<T>
where
    T: numpy::Element + Copy,
    D: numpy::ndarray::Dimension,
{
    match array.as_slice() {
        Ok(slice) => slice.to_vec(),
        Err(_) => array.as_array().iter().copied().collect(),
    }
}

/// What an object is, for an error message: its type, and its shape if it has one.
fn describe(value: &Bound<'_, PyAny>) -> String {
    let kind = value
        .get_type()
        .name()
        .map(|name| name.to_string())
        .unwrap_or_else(|_| "object".to_string());
    match value.getattr("shape").and_then(|shape| shape.str()) {
        Ok(shape) => format!("{kind} of shape {shape}"),
        Err(_) => kind,
    }
}

/// A `[rows, cols]` tensor from whatever Python passed.
pub fn tensor_2d(
    value: &Bound<'_, PyAny>,
    rows: usize,
    cols: usize,
    what: &str,
    device: &Device<R>,
) -> PyResult<Tensor<R, E>> {
    let array: FloatArray2<'_> = value.extract().map_err(|_| {
        PyValueError::new_err(format!(
            "{what} must be a [{rows}, {cols}] array of floats, got {}",
            describe(value)
        ))
    })?;
    if array.shape() != [rows, cols] {
        return Err(PyValueError::new_err(format!(
            "{what} must be [{rows}, {cols}], got {:?}",
            array.shape()
        )));
    }
    Tensor::from_f32(&values(&array), vec![rows, cols], device).map_err(crate::err::to_py)
}

/// A `[rows, cols]` legal-action mask from whatever Python passed — floats, ints
/// or bools — checked on the host against the same contract as
/// `mamba3::rl::validate_action_mask` before it reaches the device: every value
/// `0` or `1`, and at least one legal action per row. Checking here costs no
/// device read, and refuses a bad mask *before* anything is drawn from it.
pub fn action_mask_2d(
    value: &Bound<'_, PyAny>,
    rows: usize,
    cols: usize,
    what: &str,
    device: &Device<R>,
) -> PyResult<Tensor<R, E>> {
    let array: FloatArray2<'_> = value.extract().map_err(|_| {
        PyValueError::new_err(format!(
            "{what} must be a [{rows}, {cols}] array of 0/1 (or bools), got {}",
            describe(value)
        ))
    })?;
    if array.shape() != [rows, cols] {
        return Err(PyValueError::new_err(format!(
            "{what} must be [{rows}, {cols}], got {:?}",
            array.shape()
        )));
    }
    let data = values(&array);
    mamba3::rl::check_action_mask_values(&data, cols).map_err(crate::err::to_py)?;
    Tensor::from_f32(&data, vec![rows, cols], device).map_err(crate::err::to_py)
}

/// A `[len]` tensor from whatever Python passed.
pub fn tensor_1d(
    value: &Bound<'_, PyAny>,
    len: usize,
    what: &str,
    device: &Device<R>,
) -> PyResult<Tensor<R, E>> {
    let array: FloatArray1<'_> = value.extract().map_err(|_| {
        PyValueError::new_err(format!(
            "{what} must be a [{len}] array of floats, got {}",
            describe(value)
        ))
    })?;
    if array.shape() != [len] {
        return Err(PyValueError::new_err(format!(
            "{what} must be [{len}], got {:?}",
            array.shape()
        )));
    }
    Tensor::from_f32(&values(&array), vec![len], device).map_err(crate::err::to_py)
}

/// A `[len]` id tensor, checked against the action space.
///
/// An id outside `0..classes` would index past the end of a logit row on the
/// device, where there is no bounds check to catch it — so it is caught here.
pub fn ids_1d(
    value: &Bound<'_, PyAny>,
    len: usize,
    classes: usize,
    what: &str,
    device: &Device<R>,
) -> PyResult<IdTensor<R>> {
    let array: IdArray1<'_> = value.extract().map_err(|_| {
        PyValueError::new_err(format!(
            "{what} must be a [{len}] array of integers, got {}",
            describe(value)
        ))
    })?;
    if array.shape() != [len] {
        return Err(PyValueError::new_err(format!(
            "{what} must be [{len}], got {:?}",
            array.shape()
        )));
    }
    let raw = values(&array);
    let mut ids = Vec::with_capacity(len);
    for (index, id) in raw.iter().enumerate() {
        if *id < 0 || *id as usize >= classes {
            return Err(PyValueError::new_err(format!(
                "{what}[{index}] is {id}, outside the {classes} actions this \
                 environment declares"
            )));
        }
        ids.push(*id as u32);
    }
    IdTensor::from_slice(&ids, vec![len], device).map_err(crate::err::to_py)
}

/// A tensor as a flat float array. Synchronises.
pub fn to_1d<'py>(py: Python<'py>, tensor: &Tensor<R, E>) -> Bound<'py, PyArray1<f32>> {
    PyArray1::from_vec(py, tensor.to_f32())
}

/// A tensor as a `[rows, cols]` float array. Synchronises.
pub fn to_2d<'py>(
    py: Python<'py>,
    tensor: &Tensor<R, E>,
    rows: usize,
    cols: usize,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    PyArray1::from_vec(py, tensor.to_f32()).reshape((rows, cols))
}

/// Ids as a flat `int64` array, which is what numpy indexing expects. Synchronises.
pub fn ids_to_1d<'py>(py: Python<'py>, ids: &IdTensor<R>) -> Bound<'py, PyArray1<i64>> {
    PyArray1::from_vec(py, ids.to_vec().into_iter().map(i64::from).collect())
}
