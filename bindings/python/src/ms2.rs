//! MS2-to-substructure Python bindings (P6.6, architecture §4.5).
//!
//! The MS2 classes live in the existing extension module (`_mamba3_rl`) and are
//! exposed through the packaged facade `mamba3_ms2`, as `mamba3_graph` is.
//! The rules this file follows are the graph binding's:
//!
//! * **One call per unit of work.** `generate`, `step` and `teacher_eval` each
//!   run their whole loop in Rust, with the interpreter lock released across
//!   it ([`detached`]).
//! * **NumPy in, NumPy out.** Input arrays may be C- or Fortran-ordered or
//!   non-contiguous; they are copied once to contiguous buffers before use,
//!   in their own dtype, and the caller's array is never modified. A dtype of
//!   the wrong kind (floats where integers belong or vice versa) raises
//!   `TypeError` naming the field; a wrong shape or an out-of-range value
//!   raises `ValueError` naming the field.
//! * **Errors use the bindings' existing mapping** (`crate::err::to_py`) with
//!   the Rust message unchanged.
//! * **FP32 only.** Every model is `f32`, the element type of this extension.

use std::collections::HashMap;

use mamba3::backend::Device;
use mamba3::models::ms2::contract::{
    CandidateBatch, ChemistryDomain, GenerationConfig, ModelConfig, SpectrumBatch,
};
use mamba3::models::ms2::pack::PackedCandidateBatch;
use mamba3::models::ms2::experiment::ExperimentSet;
use mamba3::models::ms2::formula::FormulaTable;
use mamba3::models::ms2::formula_head::DeviceFormulaTable;
use mamba3::models::ms2::generate::{GenerationWorkspace, Ms2Model, ResidentCandidates};
use mamba3::models::ms2::targets::RecipeLimits;
use mamba3::models::ms2::train::{Ms2Trainer, TrainConfig};
use mamba3::tensor::ops::random::Rng;
use numpy::{PyArray1, PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::R;
use crate::err::IntoPyResult;

// ---------------------------------------------------------------------------
// Releasing the interpreter lock (as in `graph.rs`)
// ---------------------------------------------------------------------------

/// A value carried into and out of a section that runs without the interpreter
/// lock.
struct Detached<T>(T);

// SAFETY: as in `graph.rs`: the closure runs on the calling thread, captures
// no `Bound`/`Py`/`PyRef`, touches no Python object without the lock, and the
// `unsendable` classes refuse access from any other thread while the borrow
// is held here.
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
    /// The device every MS2 object of this thread lives on.
    static DEVICE: Device<R> = Device::<R>::default();
}

/// This thread's device.
fn device() -> Device<R> {
    DEVICE.with(Clone::clone)
}

// ---------------------------------------------------------------------------
// NumPy intake: any layout, strict kinds
// ---------------------------------------------------------------------------

/// `value` as a NumPy array: lists and scalars go through `numpy.asarray`; a
/// Fortran-ordered, sliced or unaligned array is copied once by
/// `numpy.require`, in logical order and in its own dtype. The caller's array
/// is never modified.
fn as_array<'py>(value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let numpy = value.py().import("numpy")?;
    let array = if value.cast::<numpy::PyUntypedArray>().is_ok() {
        value.clone()
    } else {
        numpy.call_method1("asarray", (value,))?
    };
    let flags = array.getattr("flags")?;
    let viewable = flags.getattr("c_contiguous")?.extract::<bool>()?
        && flags.getattr("aligned")?.extract::<bool>()?;
    if viewable {
        return Ok(array);
    }
    numpy.call_method1("require", (array, value.py().None(), ["C", "A"]))
}

/// Read a C-contiguous, aligned array in logical order.
fn contiguous_values<T>(array: &Bound<'_, PyAny>, key: &str) -> PyResult<Vec<T>>
where
    T: numpy::Element + Copy,
{
    let typed = array.cast::<PyArrayDyn<T>>().map_err(|_| {
        PyValueError::new_err(format!("{key} cannot be read as a contiguous array"))
    })?;
    let guard: PyReadonlyArrayDyn<'_, T> = typed.try_readonly().map_err(|err| {
        PyValueError::new_err(format!("{key} is being written elsewhere: {err}"))
    })?;
    match guard.as_slice() {
        Ok(slice) => Ok(slice.to_vec()),
        Err(_) => Err(PyValueError::new_err(format!(
            "{key} cannot be read as a C-contiguous, aligned array"
        ))),
    }
}

/// Integer values of any width, exactly (every integer dtype fits `i128`).
fn read_int_values(value: &Bound<'_, PyAny>, field: &str) -> PyResult<(Vec<usize>, Vec<i128>)> {
    let array = as_array(value)?;
    macro_rules! attempt {
        ($ty:ty) => {
            if let Ok(typed) = array.cast::<PyArrayDyn<$ty>>() {
                let shape = typed.shape().to_vec();
                let values: Vec<i128> =
                    contiguous_values::<$ty>(&array, field)?
                        .into_iter()
                        .map(|v| v as i128)
                        .collect();
                return Ok((shape, values));
            }
        };
    }
    attempt!(i8);
    attempt!(i16);
    attempt!(i32);
    attempt!(i64);
    attempt!(u8);
    attempt!(u16);
    attempt!(u32);
    attempt!(u64);
    if let Ok(typed) = array.cast::<PyArrayDyn<bool>>() {
        let shape = typed.shape().to_vec();
        let values: Vec<i128> = contiguous_values::<bool>(&array, field)?
            .into_iter()
            .map(|v| i128::from(v))
            .collect();
        return Ok((shape, values));
    }
    Err(PyTypeError::new_err(format!(
        "{field} must be an array of integers, got {}",
        describe(&array)
    )))
}

/// Float values of any width.
fn read_float_values(value: &Bound<'_, PyAny>, field: &str) -> PyResult<(Vec<usize>, Vec<f32>)> {
    let array = as_array(value)?;
    if let Ok(typed) = array.cast::<PyArrayDyn<f32>>() {
        let shape = typed.shape().to_vec();
        return Ok((shape, contiguous_values::<f32>(&array, field)?));
    }
    if let Ok(typed) = array.cast::<PyArrayDyn<f64>>() {
        let shape = typed.shape().to_vec();
        let values: Vec<f32> = contiguous_values::<f64>(&array, field)?
            .into_iter()
            .map(|v| v as f32)
            .collect();
        return Ok((shape, values));
    }
    if let Ok(typed) = array.cast::<PyArrayDyn<half::f16>>() {
        let shape = typed.shape().to_vec();
        let values: Vec<f32> =
            contiguous_values::<half::f16>(&array, field)?
                .into_iter()
                .map(|v| v.to_f32())
                .collect();
        return Ok((shape, values));
    }
    Err(PyTypeError::new_err(format!(
        "{field} must be an array of floats (float16, float32 or float64), got {}",
        describe(&array)
    )))
}

/// What an object is, for an error message.
fn describe(value: &Bound<'_, PyAny>) -> String {
    let dtype = value
        .getattr("dtype")
        .and_then(|d| d.str())
        .map(|d| d.to_string())
        .unwrap_or_else(|_| "object".to_string());
    match value.getattr("shape").and_then(|s| s.str()) {
        Ok(shape) => format!("a {dtype} array of shape {shape}"),
        Err(_) => dtype,
    }
}

fn shape_error(field: &str, want: &str, got: &[usize]) -> PyErr {
    PyValueError::new_err(format!("{field} must be {want}, got shape {got:?}"))
}

macro_rules! cast_int {
    ($name:ident, $ty:ty) => {
        /// Integer values range-checked into `$ty`; out of range is
        /// `ValueError` naming the field.
        fn $name(values: Vec<i128>, field: &str) -> PyResult<Vec<$ty>> {
            let max = i128::from(<$ty>::MAX);
            let min = i128::from(<$ty>::MIN);
            values
                .into_iter()
                .map(|v| {
                    <$ty>::try_from(v).map_err(|_| {
                        PyValueError::new_err(format!(
                            "{field} value {v} is outside {}..={} (range of {})",
                            min,
                            max,
                            stringify!($ty)
                        ))
                    })
                })
                .collect()
        }
    };
}

cast_int!(to_u8, u8);
cast_int!(to_u16, u16);
cast_int!(to_u32, u32);
cast_int!(to_u64, u64);
cast_int!(to_i8, i8);

/// An integer field with an exact shape.
fn int_field<T>(
    value: &Bound<'_, PyAny>,
    field: &str,
    want: &str,
    shape: &[usize],
    cast: impl Fn(Vec<i128>, &str) -> PyResult<Vec<T>>,
) -> PyResult<Vec<T>> {
    let (got, values) = read_int_values(value, field)?;
    if got != shape {
        return Err(shape_error(field, want, &got));
    }
    cast(values, field)
}

/// A float field with an exact shape.
fn float_field(
    value: &Bound<'_, PyAny>,
    field: &str,
    want: &str,
    shape: &[usize],
) -> PyResult<Vec<f32>> {
    let (got, values) = read_float_values(value, field)?;
    if got != shape {
        return Err(shape_error(field, want, &got));
    }
    Ok(values)
}

/// A scalar integer argument.
fn int_scalar<T>(value: &Bound<'_, PyAny>, field: &str) -> PyResult<T>
where
    T: TryFrom<i128>,
{
    let (_, values) = read_int_values(value, field)?;
    if values.len() != 1 {
        return Err(PyValueError::new_err(format!(
            "{field} must be a scalar integer, got shape [{:?}]",
            values.len()
        )));
    }
    T::try_from(values[0]).map_err(|_| {
        PyValueError::new_err(format!("{field} value {} is out of range", values[0]))
    })
}

// ---------------------------------------------------------------------------
// Python <-> JSON for nested structures (no new dependencies)
// ---------------------------------------------------------------------------

/// A Python value as JSON: `None`, bools, ints, floats, strings, lists,
/// tuples and dicts with string keys. Anything else is `TypeError`.
fn py_to_json(value: &Bound<'_, PyAny>) -> PyResult<serde_json::Value> {
    if value.is_none() {
        return Ok(serde_json::Value::Null);
    }
    if let Ok(v) = value.extract::<bool>() {
        return Ok(serde_json::Value::Bool(v));
    }
    if let Ok(v) = value.extract::<i64>() {
        return Ok(serde_json::Value::Number(v.into()));
    }
    if let Ok(v) = value.extract::<f64>() {
        return Ok(serde_json::Value::Number(
            serde_json::Number::from_f64(v).ok_or_else(|| {
                PyValueError::new_err("non-finite floats cannot be represented in JSON")
            })?,
        ));
    }
    if let Ok(v) = value.extract::<String>() {
        return Ok(serde_json::Value::String(v));
    }
    // `bytes` (what a `Vec<u8>` getter hands back) read as a list of ints.
    if let Ok(v) = value.extract::<Vec<u8>>() {
        return Ok(serde_json::Value::Array(
            v.into_iter().map(|b| serde_json::Value::from(u64::from(b))).collect(),
        ));
    }
    if let Ok(seq) = value.cast::<PyList>() {
        return seq.iter().map(|item| py_to_json(&item)).collect::<PyResult<Vec<_>>>().map(
            serde_json::Value::Array,
        );
    }
    if let Ok(seq) = value.cast::<PyTuple>() {
        return seq.iter().map(|item| py_to_json(&item)).collect::<PyResult<Vec<_>>>().map(
            serde_json::Value::Array,
        );
    }
    if let Ok(dict) = value.cast::<PyDict>() {
        let mut map = serde_json::Map::new();
        for (key, item) in dict.iter() {
            let key: String = key.extract().map_err(|_| {
                PyTypeError::new_err("dict keys must be strings for JSON conversion")
            })?;
            map.insert(key, py_to_json(&item)?);
        }
        return Ok(serde_json::Value::Object(map));
    }
    Err(PyTypeError::new_err(format!(
        "cannot convert {} to JSON (expected None, bool, int, float, str, list, tuple or dict)",
        describe(value)
    )))
}

/// JSON as Python objects (dicts, lists, strings, ints, floats, bools, None).
fn json_to_py<'py>(py: Python<'py>, value: &serde_json::Value) -> Bound<'py, PyAny> {
    use pyo3::types::{PyBool, PyFloat, PyInt, PyString};
    match value {
        serde_json::Value::Null => py.None().into_bound(py),
        serde_json::Value::Bool(v) => PyBool::new(py, *v).to_owned().into_any(),
        serde_json::Value::Number(v) => {
            if let Some(v) = v.as_u64() {
                PyInt::new(py, v).into_any()
            } else if let Some(v) = v.as_i64() {
                PyInt::new(py, v).into_any()
            } else {
                PyFloat::new(py, v.as_f64().unwrap_or(f64::NAN)).into_any()
            }
        }
        serde_json::Value::String(v) => PyString::new(py, v).into_any(),
        serde_json::Value::Array(items) => {
            let list = PyList::empty(py);
            for item in items {
                list.append(json_to_py(py, item)).unwrap();
            }
            list.into_any()
        }
        serde_json::Value::Object(map) => {
            let dict = PyDict::new(py);
            for (key, item) in map {
                dict.set_item(key, json_to_py(py, item)).unwrap();
            }
            dict.into_any()
        }
    }
}

/// A nested argument (dict or JSON string) parsed into `T`; `None` gives the
/// default. Errors name the field. The default travels as pre-serialised
/// JSON so this file needs no `serde` trait import (only `serde_json`).
fn nested<T>(
    value: Option<&Bound<'_, PyAny>>,
    field: &str,
    default_json: serde_json::Value,
    parse: impl Fn(serde_json::Value) -> serde_json::Result<T>,
) -> PyResult<T> {
    match value {
        None => parse(default_json)
            .map_err(|e| PyValueError::new_err(format!("{field}: {e}"))),
        Some(v) if v.is_none() => parse(default_json)
            .map_err(|e| PyValueError::new_err(format!("{field}: {e}"))),
        Some(v) => {
            if let Ok(text) = v.extract::<String>() {
                let json: serde_json::Value = serde_json::from_str(&text)
                    .map_err(|e| PyValueError::new_err(format!("{field}: {e}")))?;
                return parse(json).map_err(|e| PyValueError::new_err(format!("{field}: {e}")));
            }
            let json = py_to_json(v)?;
            parse(json).map_err(|e| PyValueError::new_err(format!("{field}: {e}")))
        }
    }
}

/// Parse a JSON document into `T`; failures map through the crate error
/// mapping (`Error::Json` becomes `ValueError`) with the Rust message.
fn parse_json<T>(json: &str, parse: impl Fn(serde_json::Value) -> serde_json::Result<T>) -> PyResult<T> {
    let value: serde_json::Value = serde_json::from_str(json)
        .map_err(|e| crate::err::to_py(mamba3::error::Error::from(e)))?;
    parse(value).map_err(|e| crate::err::to_py(mamba3::error::Error::from(e)))
}

/// Serialise a nested Rust value (already `serde_json`-able at the call
/// site) as a Python dict.
fn nested_getter<'py>(py: Python<'py>, value: serde_json::Value) -> Bound<'py, PyAny> {
    json_to_py(py, &value)
}

fn parse_dtype(value: &Bound<'_, PyAny>, field: &str) -> PyResult<mamba3::backend::DType> {
    let name: String = value.extract().map_err(|_| {
        PyValueError::new_err(format!("{field} must be the string 'f32', got {}", describe(value)))
    })?;
    match name.to_lowercase().as_str() {
        "f32" | "float32" => Ok(mamba3::backend::DType::F32),
        other => Err(PyValueError::new_err(format!(
            "{field} must be 'f32' (the only dtype of this module), got {other:?}"
        ))),
    }
}

fn parse_mode(value: &Bound<'_, PyAny>) -> PyResult<mamba3::models::ms2::contract::GenerationMode> {
    use mamba3::models::ms2::contract::GenerationMode;
    let name: String = value
        .extract()
        .map_err(|_| PyValueError::new_err("mode must be the string 'sampling'"))?;
    match name.to_lowercase().as_str() {
        "sampling" => Ok(GenerationMode::Sampling),
        "beam" => Ok(GenerationMode::Beam),
        other => Err(PyValueError::new_err(format!(
            "mode must be 'sampling' (or 'beam', rejected until P5 builds it), got {other:?}"
        ))),
    }
}

fn mode_to_str(mode: mamba3::models::ms2::contract::GenerationMode) -> &'static str {
    use mamba3::models::ms2::contract::GenerationMode;
    match mode {
        GenerationMode::Sampling => "sampling",
        GenerationMode::Beam => "beam",
    }
}

fn parse_control(value: &Bound<'_, PyAny>) -> PyResult<mamba3::models::ms2::contract::Control> {
    use mamba3::models::ms2::contract::Control;
    let name: String = value
        .extract()
        .map_err(|_| PyValueError::new_err("control must be one of 'none', 'shuffled', 'metadata', 'prior'"))?;
    match name.to_lowercase().as_str() {
        "none" => Ok(Control::None),
        "shuffled" | "shuffledspectrum" | "shuffled_spectrum" => Ok(Control::ShuffledSpectrum),
        "metadata" | "metadataonly" | "metadata_only" => Ok(Control::MetadataOnly),
        "prior" | "structure_prior" | "structureprior" => Ok(Control::StructurePrior),
        other => Err(PyValueError::new_err(format!(
            "control must be one of 'none', 'shuffled', 'metadata', 'prior', got {other:?}"
        ))),
    }
}

fn control_to_str(control: mamba3::models::ms2::contract::Control) -> &'static str {
    use mamba3::models::ms2::contract::Control;
    match control {
        Control::None => "none",
        Control::ShuffledSpectrum => "shuffled",
        Control::MetadataOnly => "metadata",
        Control::StructurePrior => "prior",
    }
}

fn parse_formula_source(
    value: &Bound<'_, PyAny>,
) -> PyResult<mamba3::models::ms2::contract::FormulaSource> {
    use mamba3::models::ms2::contract::FormulaSource;
    let name: String = value
        .extract()
        .map_err(|_| PyValueError::new_err("formula_source must be 'table' or 'enumerate'"))?;
    match name.to_lowercase().as_str() {
        "table" => Ok(FormulaSource::Table),
        "enumerate" => Ok(FormulaSource::Enumerate),
        other => Err(PyValueError::new_err(format!(
            "formula_source must be 'table' (or 'enumerate', rejected until V1 §1.4 is implemented), got {other:?}"
        ))),
    }
}

fn formula_source_to_str(source: mamba3::models::ms2::contract::FormulaSource) -> &'static str {
    use mamba3::models::ms2::contract::FormulaSource;
    match source {
        FormulaSource::Table => "table",
        FormulaSource::Enumerate => "enumerate",
    }
}

fn parse_allocation(
    value: &Bound<'_, PyAny>,
) -> PyResult<mamba3::models::ms2::contract::AllocationMode> {
    use mamba3::models::ms2::contract::AllocationMode;
    let name: String = value.extract().map_err(|_| {
        PyValueError::new_err("allocation must be 'round_robin' or 'proportional'")
    })?;
    match name.to_lowercase().as_str() {
        "roundrobin" | "round_robin" | "round-robin" => Ok(AllocationMode::RoundRobin),
        "proportional" => Ok(AllocationMode::Proportional),
        other => Err(PyValueError::new_err(format!(
            "allocation must be 'round_robin' or 'proportional', got {other:?}"
        ))),
    }
}

fn allocation_to_str(mode: mamba3::models::ms2::contract::AllocationMode) -> &'static str {
    use mamba3::models::ms2::contract::AllocationMode;
    match mode {
        AllocationMode::RoundRobin => "round_robin",
        AllocationMode::Proportional => "proportional",
    }
}

fn parse_identity(
    value: &Bound<'_, PyAny>,
) -> PyResult<mamba3::models::ms2::contract::IdentityMode> {
    use mamba3::models::ms2::contract::IdentityMode;
    let name: String = value
        .extract()
        .map_err(|_| PyValueError::new_err("identity must be 'trace' or 'graph'"))?;
    match name.to_lowercase().as_str() {
        "trace" | "traceonly" | "trace_only" => Ok(IdentityMode::TraceOnly),
        "graph" => Ok(IdentityMode::Graph),
        other => Err(PyValueError::new_err(format!(
            "identity must be 'trace' or 'graph', got {other:?}"
        ))),
    }
}

fn identity_to_str(mode: mamba3::models::ms2::contract::IdentityMode) -> &'static str {
    use mamba3::models::ms2::contract::IdentityMode;
    match mode {
        IdentityMode::TraceOnly => "trace",
        IdentityMode::Graph => "graph",
    }
}

fn parse_gold_conditioning(
    value: &Bound<'_, PyAny>,
) -> PyResult<mamba3::models::ms2::train::GoldFormulaConditioning> {
    use mamba3::models::ms2::train::GoldFormulaConditioning;
    let name: String = value.extract().map_err(|_| {
        PyValueError::new_err("gold_formula_conditioning must be 'composition' or 'row'")
    })?;
    match name.to_lowercase().as_str() {
        "composition" => Ok(GoldFormulaConditioning::Composition),
        "row" | "scoredroworzero" | "scored_row_or_zero" => {
            Ok(GoldFormulaConditioning::ScoredRowOrZero)
        }
        other => Err(PyValueError::new_err(format!(
            "gold_formula_conditioning must be 'composition' or 'row', got {other:?}"
        ))),
    }
}

fn gold_conditioning_to_str(
    mode: mamba3::models::ms2::train::GoldFormulaConditioning,
) -> &'static str {
    use mamba3::models::ms2::train::GoldFormulaConditioning;
    match mode {
        GoldFormulaConditioning::Composition => "composition",
        GoldFormulaConditioning::ScoredRowOrZero => "row",
    }
}

// ---------------------------------------------------------------------------
// SpectrumBatch
// ---------------------------------------------------------------------------

/// A batch of spectra (contract §3.1): `B` spectra, row-major `[B, n_raw]`
/// per-peak fields. Keyword arguments are named exactly as the contract's
/// fields. Arrays may be C- or Fortran-ordered or non-contiguous (copied
/// once); integer fields take any integer width, float fields any float
/// width, and anything else is `TypeError` naming the field.
#[pyclass(module = "mamba3_ms2", name = "SpectrumBatch", unsendable)]
pub struct PySpectrumBatch {
    inner: SpectrumBatch,
}

#[pymethods]
impl PySpectrumBatch {
    #[new]
    #[pyo3(signature = (*, n_raw, spectrum_id, raw_peak_count, peak_count, peak_id,
                        mz_udalton, intensity, mz_uncertainty_udalton,
                        precursor_mz_udalton, precursor_uncertainty_udalton, adduct,
                        polarity, collision_energy_ev, collision_energy_known,
                        energy_count, fragment_tolerance_ppm_tenths,
                        precursor_tolerance_ppm_tenths, instrument_class,
                        schema_version = None, intensity_scale = None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n_raw: &Bound<'_, PyAny>,
        spectrum_id: &Bound<'_, PyAny>,
        raw_peak_count: &Bound<'_, PyAny>,
        peak_count: &Bound<'_, PyAny>,
        peak_id: &Bound<'_, PyAny>,
        mz_udalton: &Bound<'_, PyAny>,
        intensity: &Bound<'_, PyAny>,
        mz_uncertainty_udalton: &Bound<'_, PyAny>,
        precursor_mz_udalton: &Bound<'_, PyAny>,
        precursor_uncertainty_udalton: &Bound<'_, PyAny>,
        adduct: &Bound<'_, PyAny>,
        polarity: &Bound<'_, PyAny>,
        collision_energy_ev: &Bound<'_, PyAny>,
        collision_energy_known: &Bound<'_, PyAny>,
        energy_count: &Bound<'_, PyAny>,
        fragment_tolerance_ppm_tenths: &Bound<'_, PyAny>,
        precursor_tolerance_ppm_tenths: &Bound<'_, PyAny>,
        instrument_class: &Bound<'_, PyAny>,
        schema_version: Option<&Bound<'_, PyAny>>,
        intensity_scale: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let n_raw: u32 = int_scalar(n_raw, "n_raw")?;
        if !matches!(n_raw, 64 | 128 | 256 | 512) {
            return Err(PyValueError::new_err(format!(
                "n_raw {n_raw} is not one of 64, 128, 256, 512"
            )));
        }
        let (shape, spectrum_id) = read_int_values(spectrum_id, "spectrum_id")?;
        if shape.len() != 1 {
            return Err(shape_error("spectrum_id", "[B]", &shape));
        }
        let b = shape[0];
        let n = n_raw as usize;
        let spectrum_id = to_u64(spectrum_id, "spectrum_id")?;
        let one = format!("[{b}]");
        let two = format!("[{b}, {n}]");
        let raw_peak_count = int_field(raw_peak_count, "raw_peak_count", &one, &[b], to_u32)?;
        let peak_count = int_field(peak_count, "peak_count", &one, &[b], to_u32)?;
        let peak_id = int_field(peak_id, "peak_id", &two, &[b, n], to_u32)?;
        let mz_udalton = int_field(mz_udalton, "mz_udalton", &two, &[b, n], to_u32)?;
        let intensity = float_field(intensity, "intensity", &two, &[b, n])?;
        let mz_uncertainty_udalton = int_field(
            mz_uncertainty_udalton,
            "mz_uncertainty_udalton",
            &one,
            &[b],
            to_u32,
        )?;
        let precursor_mz_udalton = int_field(
            precursor_mz_udalton,
            "precursor_mz_udalton",
            &one,
            &[b],
            to_u32,
        )?;
        let precursor_uncertainty_udalton = int_field(
            precursor_uncertainty_udalton,
            "precursor_uncertainty_udalton",
            &one,
            &[b],
            to_u32,
        )?;
        let adduct = int_field(adduct, "adduct", &one, &[b], to_u16)?;
        let polarity = int_field(polarity, "polarity", &one, &[b], to_i8)?;
        let collision_energy_ev = float_field(collision_energy_ev, "collision_energy_ev", &one, &[b])?;
        let collision_energy_known =
            int_field(collision_energy_known, "collision_energy_known", &one, &[b], to_u8)?;
        let energy_count = int_field(energy_count, "energy_count", &one, &[b], to_u8)?;
        let fragment_tolerance_ppm_tenths = int_field(
            fragment_tolerance_ppm_tenths,
            "fragment_tolerance_ppm_tenths",
            &one,
            &[b],
            to_u16,
        )?;
        let precursor_tolerance_ppm_tenths = int_field(
            precursor_tolerance_ppm_tenths,
            "precursor_tolerance_ppm_tenths",
            &one,
            &[b],
            to_u16,
        )?;
        let instrument_class = int_field(instrument_class, "instrument_class", &one, &[b], to_u8)?;
        let schema_version: u32 = match schema_version {
            None => 1,
            Some(v) => int_scalar(v, "schema_version")?,
        };
        let intensity_scale: u8 = match intensity_scale {
            None => 0,
            Some(v) => int_scalar(v, "intensity_scale")?,
        };
        Ok(Self {
            inner: SpectrumBatch {
                schema_version,
                n_raw,
                spectrum_id,
                raw_peak_count,
                peak_count,
                peak_id,
                mz_udalton,
                intensity,
                intensity_scale,
                mz_uncertainty_udalton,
                precursor_mz_udalton,
                precursor_uncertainty_udalton,
                adduct,
                polarity,
                collision_energy_ev,
                collision_energy_known,
                energy_count,
                fragment_tolerance_ppm_tenths,
                precursor_tolerance_ppm_tenths,
                instrument_class,
            },
        })
    }

    /// Spectra per batch.
    #[getter]
    fn batch(&self) -> usize {
        self.inner.len()
    }

    /// Raw peak capacity of the shape bucket.
    #[getter]
    fn n_raw(&self) -> u32 {
        self.inner.n_raw
    }

    /// Validate shapes and per-spectrum metadata (contract §3.1): one
    /// request-status code per spectrum. Malformed batches raise the mapped
    /// `Error::Config` (a `ValueError` with the Rust message).
    fn validate(&self) -> PyResult<Vec<u32>> {
        self.inner.validate().py()
    }

    /// The batch as JSON.
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner)
            .map_err(|e| PyValueError::new_err(format!("SpectrumBatch.to_json: {e}")))
    }

    /// Rebuild from [`to_json`](Self::to_json).
    #[staticmethod]
    fn from_json(json: &str) -> PyResult<Self> {
        Ok(Self {
            inner: parse_json(json, serde_json::from_value)?,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "SpectrumBatch(batch={}, n_raw={})",
            self.inner.len(),
            self.inner.n_raw
        )
    }
}

// ---------------------------------------------------------------------------
// ModelConfig
// ---------------------------------------------------------------------------

/// Model hyperparameters (contract §3.3). Every keyword argument defaults to
/// the documented V0 value (`ModelConfig.v0()`); `encoder`, `decoder` and
/// `formula_table` are dicts (or JSON strings) of the same shape `to_json`
/// produces, and anything omitted keeps the V0 value.
#[pyclass(module = "mamba3_ms2", name = "ModelConfig", unsendable, skip_from_py_object)]
#[derive(Clone)]
pub struct PyModelConfig {
    inner: ModelConfig,
}

#[pymethods]
impl PyModelConfig {
    #[new]
    #[pyo3(signature = (*, schema_version = None, version = None, chemistry = None,
                        n_peaks = None, d_model = None, encoder = None, decoder = None,
                        encoder_blocks = None, decoder_blocks = None,
                        attention_heads = None, fourier_features = None,
                        max_atoms = None, max_ring_closures = None,
                        formula_table = None, assignment = None, energy_scale_ev = None,
                        energy_clip_ev = None, dtype = None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        schema_version: Option<&Bound<'_, PyAny>>,
        version: Option<&Bound<'_, PyAny>>,
        chemistry: Option<&Bound<'_, PyAny>>,
        n_peaks: Option<&Bound<'_, PyAny>>,
        d_model: Option<&Bound<'_, PyAny>>,
        encoder: Option<&Bound<'_, PyAny>>,
        decoder: Option<&Bound<'_, PyAny>>,
        encoder_blocks: Option<&Bound<'_, PyAny>>,
        decoder_blocks: Option<&Bound<'_, PyAny>>,
        attention_heads: Option<&Bound<'_, PyAny>>,
        fourier_features: Option<&Bound<'_, PyAny>>,
        max_atoms: Option<&Bound<'_, PyAny>>,
        max_ring_closures: Option<&Bound<'_, PyAny>>,
        formula_table: Option<&Bound<'_, PyAny>>,
        assignment: Option<&Bound<'_, PyAny>>,
        energy_scale_ev: Option<&Bound<'_, PyAny>>,
        energy_clip_ev: Option<&Bound<'_, PyAny>>,
        dtype: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let v0 = ModelConfig::v0();
        let opt_u32 = |v: Option<&Bound<'_, PyAny>>, field: &str| -> PyResult<u32> {
            match v {
                None => {
                    let json = serde_json::to_value(&v0)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?;
                    Ok(json[field].as_u64().unwrap() as u32)
                }
                Some(b) => int_scalar(b, field),
            }
        };
        let schema_version = opt_u32(schema_version, "schema_version")?;
        let n_peaks = opt_u32(n_peaks, "n_peaks")?;
        let d_model = opt_u32(d_model, "d_model")?;
        let encoder_blocks = opt_u32(encoder_blocks, "encoder_blocks")?;
        let decoder_blocks = opt_u32(decoder_blocks, "decoder_blocks")?;
        let attention_heads = opt_u32(attention_heads, "attention_heads")?;
        let fourier_features = opt_u32(fourier_features, "fourier_features")?;
        let max_atoms = opt_u32(max_atoms, "max_atoms")?;
        let max_ring_closures = opt_u32(max_ring_closures, "max_ring_closures")?;
        let version = match version {
            None => v0.version.clone(),
            Some(v) => v.extract().map_err(|_| {
                PyValueError::new_err(format!("version must be a string, got {}", describe(v)))
            })?,
        };
        let chemistry = match chemistry {
            None => v0.chemistry.clone(),
            Some(v) => v.extract().map_err(|_| {
                PyValueError::new_err(format!("chemistry must be a string, got {}", describe(v)))
            })?,
        };
        let v0_json = serde_json::to_value(&v0)
            .map_err(|e| PyValueError::new_err(format!("ModelConfig: {e}")))?;
        let encoder = nested(
            encoder,
            "encoder",
            v0_json["encoder"].clone(),
            serde_json::from_value,
        )?;
        let decoder = nested(
            decoder,
            "decoder",
            v0_json["decoder"].clone(),
            serde_json::from_value,
        )?;
        let formula_table = nested(
            formula_table,
            "formula_table",
            v0_json["formula_table"].clone(),
            serde_json::from_value,
        )?;
        let energy_scale_ev = match energy_scale_ev {
            None => v0.energy_scale_ev,
            Some(v) => v.extract().map_err(|_| {
                PyValueError::new_err(format!(
                    "energy_scale_ev must be a float, got {}",
                    describe(v)
                ))
            })?,
        };
        let energy_clip_ev = match energy_clip_ev {
            None => v0.energy_clip_ev,
            Some(v) => v.extract().map_err(|_| {
                PyValueError::new_err(format!(
                    "energy_clip_ev must be a float, got {}",
                    describe(v)
                ))
            })?,
        };
        let dtype = match dtype {
            None => v0.dtype,
            Some(v) => parse_dtype(v, "dtype")?,
        };
        let assignment = match assignment {
            None => None,
            Some(b) if b.is_none() => None,
            Some(b) => {
                // Dict with optional `hypotheses`, `work_max`, `labels`
                // (defaults 4, 4096, 64); `True` means defaults.
                if let Ok(flag) = b.extract::<bool>() {
                    if flag {
                        Some(
                            mamba3::models::ms2::contract::AssignmentConfig::default(),
                        )
                    } else {
                        None
                    }
                } else {
                    let v: serde_json::Value = nested(
                        Some(b),
                        "assignment",
                        serde_json::json!({"hypotheses": 4, "work_max": 4096, "labels": 64}),
                        serde_json::from_value,
                    )?;
                    Some(
                        serde_json::from_value(v).map_err(|e| {
                            PyValueError::new_err(format!("assignment: {e}"))
                        })?,
                    )
                }
            }
        };
        Ok(Self {
            inner: ModelConfig {
                schema_version,
                version,
                chemistry,
                n_peaks,
                d_model,
                encoder,
                decoder,
                encoder_blocks,
                decoder_blocks,
                attention_heads,
                fourier_features,
                max_atoms,
                max_ring_closures,
                formula_table,
                formula_artifacts: None,
                assignment,
                formula_features: v0.formula_features,
                energy_scale_ev,
                energy_clip_ev,
                dtype,
            },
        })
    }

    /// The documented V0 configuration.
    #[staticmethod]
    fn v0() -> Self {
        Self {
            inner: ModelConfig::v0(),
        }
    }

    /// Check the schema version, the chemistry version, both SSMs, the shared
    /// width and the structure limits.
    fn validate(&self) -> PyResult<()> {
        self.inner.validate().py()
    }

    /// The config as JSON.
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner)
            .map_err(|e| PyValueError::new_err(format!("ModelConfig.to_json: {e}")))
    }

    /// Rebuild from [`to_json`](Self::to_json).
    #[staticmethod]
    fn from_json(json: &str) -> PyResult<Self> {
        Ok(Self {
            inner: parse_json(json, serde_json::from_value)?,
        })
    }

    #[getter]
    fn schema_version(&self) -> u32 {
        self.inner.schema_version
    }
    #[getter]
    fn version(&self) -> &str {
        &self.inner.version
    }
    #[getter]
    fn chemistry(&self) -> &str {
        &self.inner.chemistry
    }
    #[getter]
    fn n_peaks(&self) -> u32 {
        self.inner.n_peaks
    }
    #[getter]
    fn d_model(&self) -> u32 {
        self.inner.d_model
    }
    #[getter]
    fn encoder<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(nested_getter(
            py,
            serde_json::to_value(&self.inner.encoder)
                .map_err(|e| PyValueError::new_err(format!("encoder: {e}")))?,
        ))
    }
    #[getter]
    fn decoder<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(nested_getter(
            py,
            serde_json::to_value(&self.inner.decoder)
                .map_err(|e| PyValueError::new_err(format!("decoder: {e}")))?,
        ))
    }
    #[getter]
    fn encoder_blocks(&self) -> u32 {
        self.inner.encoder_blocks
    }
    #[getter]
    fn decoder_blocks(&self) -> u32 {
        self.inner.decoder_blocks
    }
    #[getter]
    fn attention_heads(&self) -> u32 {
        self.inner.attention_heads
    }
    #[getter]
    fn fourier_features(&self) -> u32 {
        self.inner.fourier_features
    }
    #[getter]
    fn max_atoms(&self) -> u32 {
        self.inner.max_atoms
    }
    #[getter]
    fn max_ring_closures(&self) -> u32 {
        self.inner.max_ring_closures
    }
    #[getter]
    fn formula_table<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(nested_getter(
            py,
            serde_json::to_value(&self.inner.formula_table)
                .map_err(|e| PyValueError::new_err(format!("formula_table: {e}")))?,
        ))
    }
    #[getter]
    fn assignment<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        match &self.inner.assignment {
            None => Ok(None),
            Some(a) => Ok(Some(nested_getter(
                py,
                serde_json::to_value(a)
                    .map_err(|e| PyValueError::new_err(format!("assignment: {e}")))?,
            ))),
        }
    }
    #[getter]
    fn energy_scale_ev(&self) -> f32 {
        self.inner.energy_scale_ev
    }
    #[getter]
    fn energy_clip_ev(&self) -> f32 {
        self.inner.energy_clip_ev
    }
    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.dtype.name()
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        format!(
            "ModelConfig(version={:?}, d_model={}, max_atoms={}, max_ring_closures={})",
            self.inner.version, self.inner.d_model, self.inner.max_atoms,
            self.inner.max_ring_closures
        )
    }
}

// ---------------------------------------------------------------------------
// GenerationConfig
// ---------------------------------------------------------------------------

/// Generation hyperparameters (contract §3.4). Every keyword argument
/// defaults to the documented value; `mode`, `control` and `formula_source`
/// are strings (`"sampling"`, `"none"`, `"table"`), as are `allocation`
/// (`"round_robin"`, `"proportional"`) and `identity` (`"trace"`, `"graph"`);
/// `returned = 0` means the default `min(10, K)`.
#[pyclass(module = "mamba3_ms2", name = "GenerationConfig", unsendable, skip_from_py_object)]
#[derive(Clone)]
pub struct PyGenerationConfig {
    inner: GenerationConfig,
}

#[pymethods]
impl PyGenerationConfig {
    #[new]
    #[pyo3(signature = (*, schema_version = None, trajectories = None, formulas = None,
                        seed = None, temperature = None, max_steps = None,
                        max_device_bytes = None, formula_rows_visited_max = None,
                        formula_rows_scored_max = None, mode = None,
                        oracle_formula = None, control = None,
                        formula_source = None, formula_window = None,
                        enum_lanes_max = None, enum_lane_visits_max = None,
                        enum_dispatch_visits_max = None, allocation = None,
                        identity = None, identity_work_max = None,
                        returned = None, evidence = None,
                        ion_request_work_max = None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        schema_version: Option<&Bound<'_, PyAny>>,
        trajectories: Option<&Bound<'_, PyAny>>,
        formulas: Option<&Bound<'_, PyAny>>,
        seed: Option<&Bound<'_, PyAny>>,
        temperature: Option<&Bound<'_, PyAny>>,
        max_steps: Option<&Bound<'_, PyAny>>,
        max_device_bytes: Option<&Bound<'_, PyAny>>,
        formula_rows_visited_max: Option<&Bound<'_, PyAny>>,
        formula_rows_scored_max: Option<&Bound<'_, PyAny>>,
        mode: Option<&Bound<'_, PyAny>>,
        oracle_formula: Option<&Bound<'_, PyAny>>,
        control: Option<&Bound<'_, PyAny>>,
        formula_source: Option<&Bound<'_, PyAny>>,
        formula_window: Option<&Bound<'_, PyAny>>,
        enum_lanes_max: Option<&Bound<'_, PyAny>>,
        enum_lane_visits_max: Option<&Bound<'_, PyAny>>,
        enum_dispatch_visits_max: Option<&Bound<'_, PyAny>>,
        allocation: Option<&Bound<'_, PyAny>>,
        identity: Option<&Bound<'_, PyAny>>,
        identity_work_max: Option<&Bound<'_, PyAny>>,
        returned: Option<&Bound<'_, PyAny>>,
        evidence: Option<&Bound<'_, PyAny>>,
        ion_request_work_max: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let d = GenerationConfig::default();
        let u32_or = |v: Option<&Bound<'_, PyAny>>, field: &str, default: u32| -> PyResult<u32> {
            match v {
                None => Ok(default),
                Some(b) => int_scalar(b, field),
            }
        };
        let schema_version = u32_or(schema_version, "schema_version", d.schema_version)?;
        let trajectories = u32_or(trajectories, "trajectories", d.trajectories)?;
        let formulas = u32_or(formulas, "formulas", d.formulas)?;
        let seed: u64 = match seed {
            None => d.seed,
            Some(b) => {
                let (_, values) = read_int_values(b, "seed")?;
                if values.len() != 1 {
                    return Err(PyValueError::new_err("seed must be a scalar integer"));
                }
                u64::try_from(values[0]).map_err(|_| {
                    PyValueError::new_err(format!("seed value {} is out of range", values[0]))
                })?
            }
        };
        let temperature: f32 = match temperature {
            None => d.temperature,
            Some(b) => b.extract().map_err(|_| {
                PyValueError::new_err(format!(
                    "temperature must be a float, got {}",
                    describe(b)
                ))
            })?,
        };
        let max_steps = u32_or(max_steps, "max_steps", d.max_steps)?;
        let max_device_bytes: u64 = match max_device_bytes {
            None => d.max_device_bytes,
            Some(b) => {
                let (_, values) = read_int_values(b, "max_device_bytes")?;
                if values.len() != 1 {
                    return Err(PyValueError::new_err(
                        "max_device_bytes must be a scalar integer",
                    ));
                }
                u64::try_from(values[0]).map_err(|_| {
                    PyValueError::new_err(format!(
                        "max_device_bytes value {} is out of range",
                        values[0]
                    ))
                })?
            }
        };
        let formula_rows_visited_max = u32_or(
            formula_rows_visited_max,
            "formula_rows_visited_max",
            d.formula_rows_visited_max,
        )?;
        let formula_rows_scored_max = u32_or(
            formula_rows_scored_max,
            "formula_rows_scored_max",
            d.formula_rows_scored_max,
        )?;
        let mode = match mode {
            None => d.mode,
            Some(b) => parse_mode(b)?,
        };
        let oracle_formula: bool = match oracle_formula {
            None => d.oracle_formula,
            Some(b) => b.extract().map_err(|_| {
                PyValueError::new_err(format!(
                    "oracle_formula must be a bool, got {}",
                    describe(b)
                ))
            })?,
        };
        let control = match control {
            None => d.control,
            Some(b) => parse_control(b)?,
        };
        let formula_source = match formula_source {
            None => d.formula_source,
            Some(b) => parse_formula_source(b)?,
        };
        let formula_window = u32_or(formula_window, "formula_window", d.formula_window)?;
        let enum_lanes_max = u32_or(enum_lanes_max, "enum_lanes_max", d.enum_lanes_max)?;
        let enum_lane_visits_max = u32_or(
            enum_lane_visits_max,
            "enum_lane_visits_max",
            d.enum_lane_visits_max,
        )?;
        let enum_dispatch_visits_max = u32_or(
            enum_dispatch_visits_max,
            "enum_dispatch_visits_max",
            d.enum_dispatch_visits_max,
        )?;
        let allocation = match allocation {
            None => d.allocation,
            Some(b) => parse_allocation(b)?,
        };
        let identity = match identity {
            None => d.identity,
            Some(b) => parse_identity(b)?,
        };
        let identity_work_max =
            u32_or(identity_work_max, "identity_work_max", d.identity_work_max)?;
        let returned = u32_or(returned, "returned", d.returned)?;
        let evidence: bool = match evidence {
            None => d.evidence,
            Some(b) => b.extract().map_err(|_| {
                PyValueError::new_err(format!("evidence must be a bool, got {}", describe(b)))
            })?,
        };
        let ion_request_work_max = u32_or(
            ion_request_work_max,
            "ion_request_work_max",
            d.ion_request_work_max,
        )?;
        Ok(Self {
            inner: GenerationConfig {
                schema_version,
                trajectories,
                formulas,
                seed,
                temperature,
                max_steps,
                max_device_bytes,
                formula_rows_visited_max,
                formula_rows_scored_max,
                mode,
                oracle_formula,
                control,
                formula_source,
                formula_window,
                enum_lanes_max,
                enum_lane_visits_max,
                enum_dispatch_visits_max,
                allocation,
                identity,
                identity_work_max,
                returned,
                evidence,
                ion_request_work_max,
                formula_evidence_work_max: d.formula_evidence_work_max,
                formula_evidence_dispatch_max: d.formula_evidence_dispatch_max,
            },
        })
    }

    /// Enforce every documented range under these structure limits (default
    /// the V0 `max_atoms = 16`, `max_ring_closures = 4`).
    #[pyo3(signature = (max_atoms = None, max_ring_closures = None))]
    fn validate(
        &self,
        max_atoms: Option<usize>,
        max_ring_closures: Option<usize>,
    ) -> PyResult<()> {
        self.inner
            .validate(max_atoms.unwrap_or(16), max_ring_closures.unwrap_or(4))
            .py()
    }

    /// The config as JSON.
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner)
            .map_err(|e| PyValueError::new_err(format!("GenerationConfig.to_json: {e}")))
    }

    /// Rebuild from [`to_json`](Self::to_json).
    #[staticmethod]
    fn from_json(json: &str) -> PyResult<Self> {
        Ok(Self {
            inner: parse_json(json, serde_json::from_value)?,
        })
    }

    #[getter]
    fn schema_version(&self) -> u32 {
        self.inner.schema_version
    }
    #[getter]
    fn trajectories(&self) -> u32 {
        self.inner.trajectories
    }
    #[getter]
    fn formulas(&self) -> u32 {
        self.inner.formulas
    }
    #[getter]
    fn seed(&self) -> u64 {
        self.inner.seed
    }
    #[getter]
    fn temperature(&self) -> f32 {
        self.inner.temperature
    }
    #[getter]
    fn max_steps(&self) -> u32 {
        self.inner.max_steps
    }
    #[getter]
    fn max_device_bytes(&self) -> u64 {
        self.inner.max_device_bytes
    }
    #[getter]
    fn formula_rows_visited_max(&self) -> u32 {
        self.inner.formula_rows_visited_max
    }
    #[getter]
    fn formula_rows_scored_max(&self) -> u32 {
        self.inner.formula_rows_scored_max
    }
    #[getter]
    fn mode(&self) -> &'static str {
        mode_to_str(self.inner.mode)
    }
    #[getter]
    fn oracle_formula(&self) -> bool {
        self.inner.oracle_formula
    }
    #[getter]
    fn control(&self) -> &'static str {
        control_to_str(self.inner.control)
    }
    #[getter]
    fn formula_source(&self) -> &'static str {
        formula_source_to_str(self.inner.formula_source)
    }
    #[getter]
    fn formula_window(&self) -> u32 {
        self.inner.formula_window
    }
    #[getter]
    fn enum_lanes_max(&self) -> u32 {
        self.inner.enum_lanes_max
    }
    #[getter]
    fn enum_lane_visits_max(&self) -> u32 {
        self.inner.enum_lane_visits_max
    }
    #[getter]
    fn enum_dispatch_visits_max(&self) -> u32 {
        self.inner.enum_dispatch_visits_max
    }
    #[getter]
    fn allocation(&self) -> &'static str {
        allocation_to_str(self.inner.allocation)
    }
    #[getter]
    fn identity(&self) -> &'static str {
        identity_to_str(self.inner.identity)
    }
    #[getter]
    fn identity_work_max(&self) -> u32 {
        self.inner.identity_work_max
    }
    #[getter]
    fn returned(&self) -> u32 {
        self.inner.returned
    }
    #[getter]
    fn evidence(&self) -> bool {
        self.inner.evidence
    }
    #[getter]
    fn ion_request_work_max(&self) -> u32 {
        self.inner.ion_request_work_max
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        format!(
            "GenerationConfig(trajectories={}, formulas={}, seed={}, temperature={}, max_steps={})",
            self.inner.trajectories,
            self.inner.formulas,
            self.inner.seed,
            self.inner.temperature,
            self.inner.max_steps
        )
    }
}

// ---------------------------------------------------------------------------
// ChemistryDomain
// ---------------------------------------------------------------------------

/// The chemistry domain as checkpoint data (contract §3.2). Built with no
/// arguments it is the V0 domain; any field may be overridden with a keyword
/// argument of the same shape `to_json` produces.
#[pyclass(module = "mamba3_ms2", name = "ChemistryDomain", unsendable, skip_from_py_object)]
#[derive(Clone)]
pub struct PyChemistryDomain {
    inner: ChemistryDomain,
}

#[pymethods]
impl PyChemistryDomain {
    #[new]
    #[pyo3(signature = (*, schema_version = None, version = None, mass_scale = None,
                        elements = None, electron_mass = None,
                        electron_residual_nda = None, atom_types = None,
                        bond_orders = None, adducts = None,
                        max_hydrogen_shift = None, grammar = None,
                        traversal = None, recipe = None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        schema_version: Option<&Bound<'_, PyAny>>,
        version: Option<&Bound<'_, PyAny>>,
        mass_scale: Option<&Bound<'_, PyAny>>,
        elements: Option<&Bound<'_, PyAny>>,
        electron_mass: Option<&Bound<'_, PyAny>>,
        electron_residual_nda: Option<&Bound<'_, PyAny>>,
        atom_types: Option<&Bound<'_, PyAny>>,
        bond_orders: Option<&Bound<'_, PyAny>>,
        adducts: Option<&Bound<'_, PyAny>>,
        max_hydrogen_shift: Option<&Bound<'_, PyAny>>,
        grammar: Option<&Bound<'_, PyAny>>,
        traversal: Option<&Bound<'_, PyAny>>,
        recipe: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let v0 = ChemistryDomain::v0();
        let u32_or = |v: Option<&Bound<'_, PyAny>>, field: &str, default: u32| -> PyResult<u32> {
            match v {
                None => Ok(default),
                Some(b) => int_scalar(b, field),
            }
        };
        let u8_or = |v: Option<&Bound<'_, PyAny>>, field: &str, default: u8| -> PyResult<u8> {
            match v {
                None => Ok(default),
                Some(b) => int_scalar(b, field),
            }
        };
        let str_or =
            |v: Option<&Bound<'_, PyAny>>, field: &str, default: String| -> PyResult<String> {
                match v {
                    None => Ok(default),
                    Some(b) => b.extract().map_err(|_| {
                        PyValueError::new_err(format!(
                            "{field} must be a string, got {}",
                            describe(b)
                        ))
                    }),
                }
            };
        Ok(Self {
            inner: ChemistryDomain {
                schema_version: u32_or(schema_version, "schema_version", v0.schema_version)?,
                version: str_or(version, "version", v0.version.clone())?,
                mass_scale: u32_or(mass_scale, "mass_scale", v0.mass_scale)?,
                elements: nested(
                    elements,
                    "elements",
                    serde_json::to_value(&v0.elements).map_err(|e| {
                        PyValueError::new_err(format!("elements: {e}"))
                    })?,
                    serde_json::from_value,
                )?,
                electron_mass: u32_or(electron_mass, "electron_mass", v0.electron_mass)?,
                electron_residual_nda: u32_or(
                    electron_residual_nda,
                    "electron_residual_nda",
                    v0.electron_residual_nda,
                )?,
                atom_types: nested(
                    atom_types,
                    "atom_types",
                    serde_json::to_value(&v0.atom_types).map_err(|e| {
                        PyValueError::new_err(format!("atom_types: {e}"))
                    })?,
                    serde_json::from_value,
                )?,
                bond_orders: nested(
                    bond_orders,
                    "bond_orders",
                    serde_json::to_value(&v0.bond_orders).map_err(|e| {
                        PyValueError::new_err(format!("bond_orders: {e}"))
                    })?,
                    serde_json::from_value,
                )?,
                adducts: nested(
                    adducts,
                    "adducts",
                    serde_json::to_value(&v0.adducts).map_err(|e| {
                        PyValueError::new_err(format!("adducts: {e}"))
                    })?,
                    serde_json::from_value,
                )?,
                max_hydrogen_shift: u8_or(
                    max_hydrogen_shift,
                    "max_hydrogen_shift",
                    v0.max_hydrogen_shift,
                )?,
                grammar: str_or(grammar, "grammar", v0.grammar.clone())?,
                traversal: str_or(traversal, "traversal", v0.traversal.clone())?,
                recipe: str_or(recipe, "recipe", v0.recipe.clone())?,
            },
        })
    }

    /// The V0 domain.
    #[staticmethod]
    fn v0() -> Self {
        Self {
            inner: ChemistryDomain::v0(),
        }
    }

    /// Reject a schema version other than 1, naming both versions.
    fn validate(&self) -> PyResult<()> {
        self.inner.validate().py()
    }

    /// The domain as JSON.
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner)
            .map_err(|e| PyValueError::new_err(format!("ChemistryDomain.to_json: {e}")))
    }

    /// Rebuild from [`to_json`](Self::to_json).
    #[staticmethod]
    fn from_json(json: &str) -> PyResult<Self> {
        Ok(Self {
            inner: parse_json(json, serde_json::from_value)?,
        })
    }

    #[getter]
    fn schema_version(&self) -> u32 {
        self.inner.schema_version
    }
    #[getter]
    fn version(&self) -> &str {
        &self.inner.version
    }
    #[getter]
    fn mass_scale(&self) -> u32 {
        self.inner.mass_scale
    }
    #[getter]
    fn elements<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(nested_getter(
            py,
            serde_json::to_value(&self.inner.elements)
                .map_err(|e| PyValueError::new_err(format!("elements: {e}")))?,
        ))
    }
    #[getter]
    fn electron_mass(&self) -> u32 {
        self.inner.electron_mass
    }
    #[getter]
    fn electron_residual_nda(&self) -> u32 {
        self.inner.electron_residual_nda
    }
    #[getter]
    fn atom_types<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(nested_getter(
            py,
            serde_json::to_value(&self.inner.atom_types)
                .map_err(|e| PyValueError::new_err(format!("atom_types: {e}")))?,
        ))
    }
    #[getter]
    fn bond_orders<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(nested_getter(
            py,
            serde_json::to_value(&self.inner.bond_orders)
                .map_err(|e| PyValueError::new_err(format!("bond_orders: {e}")))?,
        ))
    }
    #[getter]
    fn adducts<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(nested_getter(
            py,
            serde_json::to_value(&self.inner.adducts)
                .map_err(|e| PyValueError::new_err(format!("adducts: {e}")))?,
        ))
    }
    #[getter]
    fn max_hydrogen_shift(&self) -> u8 {
        self.inner.max_hydrogen_shift
    }
    #[getter]
    fn grammar(&self) -> &str {
        &self.inner.grammar
    }
    #[getter]
    fn traversal(&self) -> &str {
        &self.inner.traversal
    }
    #[getter]
    fn recipe(&self) -> &str {
        &self.inner.recipe
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        format!("ChemistryDomain(version={:?})", self.inner.version)
    }
}

// ---------------------------------------------------------------------------
// FormulaTable
// ---------------------------------------------------------------------------

/// The resident V0 formula table (contracts §9): distinct molecular formulas
/// of the in-domain train-subset structures, sorted by integer mass.
#[pyclass(module = "mamba3_ms2", name = "FormulaTable", unsendable)]
pub struct PyFormulaTable {
    inner: FormulaTable,
}

#[pymethods]
impl PyFormulaTable {
    /// Read the `tools/ms2/formula_table.py` JSON format. The argument is a
    /// path when it names an existing file, otherwise the JSON text itself.
    #[staticmethod]
    fn from_json(path_or_text: &str) -> PyResult<Self> {
        let path = std::path::Path::new(path_or_text);
        let text = if path.is_file() {
            std::fs::read_to_string(path)
                .map_err(mamba3::error::Error::from)
                .py()?
        } else {
            path_or_text.to_string()
        };
        Ok(Self {
            inner: FormulaTable::from_json(&text).py()?,
        })
    }

    /// Write the `tools/ms2/formula_table.py` JSON format.
    fn to_json(&self) -> String {
        self.inner.to_json()
    }

    /// Rows in the table.
    #[getter]
    fn rows(&self) -> usize {
        self.inner.len()
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        format!("FormulaTable(rows={})", self.inner.len())
    }
}

// ---------------------------------------------------------------------------
// CandidateBatch
// ---------------------------------------------------------------------------

/// Generated candidates of a batch (contract §3.5): exactly `batch *
/// trajectories` records in `(spectrum, trajectory)` order. Every field is a
/// NumPy array with the contract's dtype and shape.
#[pyclass(module = "mamba3_ms2", name = "CandidateBatch", unsendable)]
pub struct PyCandidateBatch {
    inner: CandidateBatch,
}

fn u64_array<'py>(py: Python<'py>, values: &[u64]) -> Bound<'py, PyArray1<u64>> {
    PyArray1::from_vec(py, values.to_vec())
}

fn u32_array<'py>(py: Python<'py>, values: &[u32]) -> Bound<'py, PyArray1<u32>> {
    PyArray1::from_vec(py, values.to_vec())
}

fn u8_array<'py>(py: Python<'py>, values: &[u8]) -> Bound<'py, PyArray1<u8>> {
    PyArray1::from_vec(py, values.to_vec())
}

fn u16_array<'py>(py: Python<'py>, values: &[u16]) -> Bound<'py, PyArray1<u16>> {
    PyArray1::from_vec(py, values.to_vec())
}

fn f32_array<'py>(py: Python<'py>, values: &[f32]) -> Bound<'py, PyArray1<f32>> {
    PyArray1::from_vec(py, values.to_vec())
}

#[pymethods]
impl PyCandidateBatch {
    /// Rebuild from [`to_json`](Self::to_json).
    #[staticmethod]
    fn from_json(json: &str) -> PyResult<Self> {
        Ok(Self {
            inner: parse_json(json, serde_json::from_value)?,
        })
    }

    /// The batch as JSON.
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner)
            .map_err(|e| PyValueError::new_err(format!("CandidateBatch.to_json: {e}")))
    }

    /// Check lengths and every invariant of contract §§3.5, 4.4 and 8.
    fn validate(&self) -> PyResult<()> {
        self.inner.validate().py()
    }

    /// Indices of the records a caller that wants exact-trace duplicates
    /// removed should keep: finished, valid records without `duplicate_trace`
    /// or `request_failed`.
    fn distinct_traces(&self) -> Vec<usize> {
        self.inner.distinct_traces()
    }

    /// Rank and compact to `returned` slots per spectrum with the raw score
    /// (`formula_log_prob + trace_log_prob`) and trace-only identity: the
    /// host side of the packed readout, for parity checks of
    /// [`generate_packed`](PyMs2Model::generate_packed).
    fn pack(&self, returned: usize) -> PyResult<PyPackedCandidateBatch> {
        let inner = mamba3::models::ms2::pack::pack(
            &self.inner,
            None,
            mamba3::models::ms2::pack::ScoreKind::Raw,
            returned,
        )
        .py()?;
        Ok(PyPackedCandidateBatch { inner })
    }

    #[getter]
    fn schema_version(&self) -> u32 {
        self.inner.schema_version
    }
    #[getter]
    fn batch(&self) -> usize {
        self.inner.batch
    }
    #[getter]
    fn trajectories(&self) -> usize {
        self.inner.trajectories
    }
    #[getter]
    fn max_steps(&self) -> usize {
        self.inner.max_steps
    }
    #[getter]
    fn max_atoms(&self) -> usize {
        self.inner.max_atoms
    }
    #[getter]
    fn max_ring_closures(&self) -> usize {
        self.inner.max_ring_closures
    }

    #[getter]
    fn spectrum_id<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u64>> {
        u64_array(py, &self.inner.spectrum_id)
    }
    #[getter]
    fn trajectory<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        u32_array(py, &self.inner.trajectory)
    }
    #[getter]
    fn actions<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let n = self.inner.batch * self.inner.trajectories;
        let flat = PyArray1::from_vec(py, self.inner.actions.clone());
        Ok(flat.reshape((n, self.inner.max_steps, 4))?.into_any())
    }
    #[getter]
    fn length<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        u32_array(py, &self.inner.length)
    }
    #[getter]
    fn formula_row<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        u32_array(py, &self.inner.formula_row)
    }
    #[getter]
    fn formula_log_prob<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        f32_array(py, &self.inner.formula_log_prob)
    }
    #[getter]
    fn trace_log_prob<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        f32_array(py, &self.inner.trace_log_prob)
    }
    #[getter]
    fn open_valence<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let n = self.inner.batch * self.inner.trajectories;
        let flat = PyArray1::from_vec(py, self.inner.open_valence.clone());
        Ok(flat.reshape((n, self.inner.max_atoms))?.into_any())
    }
    #[getter]
    fn attachment_partition<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u8>> {
        u8_array(py, &self.inner.attachment_partition)
    }
    #[getter]
    fn status<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        u32_array(py, &self.inner.status)
    }
    #[getter]
    fn evidence_status<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u8>> {
        u8_array(py, &self.inner.evidence_status)
    }
    #[getter]
    fn identity_resolution<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u8>> {
        u8_array(py, &self.inner.identity_resolution)
    }
    #[getter]
    fn request_status<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        u32_array(py, &self.inner.request_status)
    }
    #[getter]
    fn rows_visited<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        u32_array(py, &self.inner.rows_visited)
    }
    #[getter]
    fn rows_joined<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        u32_array(py, &self.inner.rows_joined)
    }
    #[getter]
    fn rows_scored<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        u32_array(py, &self.inner.rows_scored)
    }
    #[getter]
    fn formula_support_complete<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u8>> {
        u8_array(py, &self.inner.formula_support_complete)
    }
    #[getter]
    fn formula_mass_retained<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        f32_array(py, &self.inner.formula_mass_retained)
    }
    #[getter]
    fn peaks_kept<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        u32_array(py, &self.inner.peaks_kept)
    }
    #[getter]
    fn intensity_retained<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        f32_array(py, &self.inner.intensity_retained)
    }
    #[getter]
    fn formula_counts<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let n = self.inner.batch * self.inner.trajectories;
        let flat = PyArray1::from_vec(py, self.inner.formula_counts.clone());
        Ok(flat.reshape((n, 10))?.into_any())
    }
    #[getter]
    fn formula_source<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u8>> {
        u8_array(py, &self.inner.formula_source)
    }
    #[getter]
    fn formula_rank<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        u32_array(py, &self.inner.formula_rank)
    }
    #[getter]
    fn evidence_count<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u8>> {
        u8_array(py, &self.inner.evidence_count)
    }
    #[getter]
    fn evidence_peak_id<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let n = self.inner.batch * self.inner.trajectories;
        let flat = PyArray1::from_vec(py, self.inner.evidence_peak_id.clone());
        Ok(flat.reshape((n, 4))?.into_any())
    }
    #[getter]
    fn evidence_hypothesis<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let n = self.inner.batch * self.inner.trajectories;
        let flat = PyArray1::from_vec(py, self.inner.evidence_hypothesis.clone());
        Ok(flat.reshape((n, 4))?.into_any())
    }
    #[getter]
    fn evidence_shift<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let flat = PyArray1::from_vec(py, self.inner.evidence_shift.clone());
        let n = self.inner.batch * self.inner.trajectories;
        Ok(flat.reshape((n, 4))?.into_any())
    }
    #[getter]
    fn evidence_residual<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let n = self.inner.batch * self.inner.trajectories;
        let flat = PyArray1::from_vec(py, self.inner.evidence_residual.clone());
        Ok(flat.reshape((n, 4))?.into_any())
    }
    #[getter]
    fn evidence_log_prob<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let n = self.inner.batch * self.inner.trajectories;
        let flat = PyArray1::from_vec(py, self.inner.evidence_log_prob.clone());
        Ok(flat.reshape((n, 4))?.into_any())
    }

    fn __repr__(&self) -> String {
        format!(
            "CandidateBatch(batch={}, trajectories={}, max_steps={})",
            self.inner.batch, self.inner.trajectories, self.inner.max_steps
        )
    }
}

// ---------------------------------------------------------------------------
// ExperimentSet
// ---------------------------------------------------------------------------

/// A loaded experiment dataset: molecules plus their spectra in file order,
/// with parents and pseudo-labels built under the frozen V0 recipe limits.
#[pyclass(module = "mamba3_ms2", name = "ExperimentSet", unsendable)]
pub struct PyExperimentSet {
    inner: ExperimentSet,
}

#[pymethods]
impl PyExperimentSet {
    /// Load an export file. `table` is accepted for the call shape
    /// (`ExperimentSet.from_export(path, table)`); the labels come from the
    /// export itself and the table binds at trainer/model construction.
    #[staticmethod]
    #[pyo3(signature = (path, table = None))]
    fn from_export(path: &Bound<'_, PyAny>, table: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let path: String = path.extract().map_err(|_| {
            PyValueError::new_err(format!("path must be a string, got {}", describe(path)))
        })?;
        if table.is_some() {
            // Type-checked only: the export carries its own labels.
            table
                .expect("checked above")
                .extract::<PyRef<'_, PyFormulaTable>>()
                .map_err(|_| {
                    PyTypeError::new_err("table must be a FormulaTable or None".to_string())
                })?;
        }
        Ok(Self {
            inner: ExperimentSet::load(std::path::Path::new(&path), &RecipeLimits::V0).py()?,
        })
    }

    /// The first `n` labeled spectra in file order, one per molecule (the
    /// overfit fixture of architecture §7).
    fn take_labeled(&self, n: usize) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.take_labeled(n).py()?,
        })
    }

    /// Spectra in the set.
    #[getter]
    fn spectrum_count(&self) -> usize {
        self.inner.spectra.len()
    }

    /// In-domain spectra with at least one target.
    #[getter]
    fn labeled_count(&self) -> usize {
        self.inner.labeled().len()
    }

    /// Molecule keys in export order.
    #[getter]
    fn molecule_count(&self) -> usize {
        self.inner.molecules.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "ExperimentSet(spectra={}, molecules={})",
            self.inner.spectra.len(),
            self.inner.molecules.len()
        )
    }
}

// ---------------------------------------------------------------------------
// TrainConfig
// ---------------------------------------------------------------------------

/// Training hyperparameters of [`Ms2Trainer`]. Every keyword argument
/// defaults to the documented V0 value; `control` is one of `"none"`,
/// `"shuffled"`, `"metadata"`, `"prior"`, and `gold_formula_conditioning` one
/// of `"composition"`, `"row"`.
#[pyclass(module = "mamba3_ms2", name = "TrainConfig", unsendable, skip_from_py_object)]
#[derive(Clone)]
pub struct PyTrainConfig {
    inner: TrainConfig,
}

#[pymethods]
impl PyTrainConfig {
    #[new]
    #[pyo3(signature = (*, batch = None, slots = None, lr = None, weight_decay = None,
                        formula_weight = None, seed = None, control = None,
                        grad_clip = None, gold_formula_conditioning = None,
                        formula_source = None, formula_window = None,
                        lambda_assign = None, ion_request_work_max = None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        batch: Option<&Bound<'_, PyAny>>,
        slots: Option<&Bound<'_, PyAny>>,
        lr: Option<&Bound<'_, PyAny>>,
        weight_decay: Option<&Bound<'_, PyAny>>,
        formula_weight: Option<&Bound<'_, PyAny>>,
        seed: Option<&Bound<'_, PyAny>>,
        control: Option<&Bound<'_, PyAny>>,
        grad_clip: Option<&Bound<'_, PyAny>>,
        gold_formula_conditioning: Option<&Bound<'_, PyAny>>,
        formula_source: Option<&Bound<'_, PyAny>>,
        formula_window: Option<&Bound<'_, PyAny>>,
        lambda_assign: Option<&Bound<'_, PyAny>>,
        ion_request_work_max: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let d = TrainConfig::default();
        let usize_or =
            |v: Option<&Bound<'_, PyAny>>, field: &str, default: usize| -> PyResult<usize> {
                match v {
                    None => Ok(default),
                    Some(b) => {
                        let (_, values) = read_int_values(b, field)?;
                        if values.len() != 1 {
                            return Err(PyValueError::new_err(format!(
                                "{field} must be a scalar integer"
                            )));
                        }
                        usize::try_from(values[0]).map_err(|_| {
                            PyValueError::new_err(format!(
                                "{field} value {} is out of range",
                                values[0]
                            ))
                        })
                    }
                }
            };
        let f32_or = |v: Option<&Bound<'_, PyAny>>, field: &str, default: f32| -> PyResult<f32> {
            match v {
                None => Ok(default),
                Some(b) => b.extract().map_err(|_| {
                    PyValueError::new_err(format!("{field} must be a float, got {}", describe(b)))
                }),
            }
        };
        let batch = usize_or(batch, "batch", d.batch)?;
        let slots = usize_or(slots, "slots", d.slots)?;
        let lr = f32_or(lr, "lr", d.lr)?;
        let weight_decay = f32_or(weight_decay, "weight_decay", d.weight_decay)?;
        let formula_weight = f32_or(formula_weight, "formula_weight", d.formula_weight)?;
        let seed: u64 = match seed {
            None => d.seed,
            Some(b) => {
                let (_, values) = read_int_values(b, "seed")?;
                if values.len() != 1 {
                    return Err(PyValueError::new_err("seed must be a scalar integer"));
                }
                u64::try_from(values[0]).map_err(|_| {
                    PyValueError::new_err(format!("seed value {} is out of range", values[0]))
                })?
            }
        };
        let control = match control {
            None => d.control,
            Some(b) => parse_control(b)?,
        };
        let grad_clip: Option<f32> = match grad_clip {
            None => d.grad_clip,
            Some(b) if b.is_none() => None,
            Some(b) => Some(b.extract().map_err(|_| {
                PyValueError::new_err(format!(
                    "grad_clip must be a float or None, got {}",
                    describe(b)
                ))
            })?),
        };
        let gold_formula_conditioning = match gold_formula_conditioning {
            None => d.gold_formula_conditioning,
            Some(b) => parse_gold_conditioning(b)?,
        };
        let formula_source = match formula_source {
            None => d.formula_source,
            Some(b) => parse_formula_source(b)?,
        };
        let formula_window = match formula_window {
            None => d.formula_window,
            Some(b) => {
                let (_, values) = read_int_values(b, "formula_window")?;
                if values.len() != 1 {
                    return Err(PyValueError::new_err(
                        "formula_window must be a scalar integer",
                    ));
                }
                u32::try_from(values[0]).map_err(|_| {
                    PyValueError::new_err(format!(
                        "formula_window value {} is out of range",
                        values[0]
                    ))
                })?
            }
        };
        let lambda_assign = f32_or(lambda_assign, "lambda_assign", d.lambda_assign)?;
        let ion_request_work_max = match ion_request_work_max {
            None => d.ion_request_work_max,
            Some(b) => {
                let (_, values) = read_int_values(b, "ion_request_work_max")?;
                if values.len() != 1 {
                    return Err(PyValueError::new_err(
                        "ion_request_work_max must be a scalar integer",
                    ));
                }
                u32::try_from(values[0]).map_err(|_| {
                    PyValueError::new_err(format!(
                        "ion_request_work_max value {} is out of range",
                        values[0]
                    ))
                })?
            }
        };
        Ok(Self {
            inner: TrainConfig {
                batch,
                slots,
                lr,
                weight_decay,
                formula_weight,
                seed,
                control,
                grad_clip,
                gold_formula_conditioning,
                formula_source,
                formula_window,
                enum_lanes_max: d.enum_lanes_max,
                enum_lane_visits_max: d.enum_lane_visits_max,
                enum_dispatch_visits_max: d.enum_dispatch_visits_max,
                enum_fit_name: None,
                enum_fit_sha256: None,
                enum_fit_subset: None,
                lambda_assign,
                ion_request_work_max,
                formula_evidence_work_max: d.formula_evidence_work_max,
                formula_evidence_dispatch_max: d.formula_evidence_dispatch_max,
                precursor_jitter_ppm: d.precursor_jitter_ppm,
                precursor_jitter_variants: d.precursor_jitter_variants,
                nonfinite_guard: false,
                loss_scale: 1.0,
            },
        })
    }

    /// Check the documented ranges.
    fn validate(&self) -> PyResult<()> {
        self.inner.validate().py()
    }

    /// The config as JSON.
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner)
            .map_err(|e| PyValueError::new_err(format!("TrainConfig.to_json: {e}")))
    }

    /// Rebuild from [`to_json`](Self::to_json).
    #[staticmethod]
    fn from_json(json: &str) -> PyResult<Self> {
        Ok(Self {
            inner: parse_json(json, serde_json::from_value)?,
        })
    }

    #[getter]
    fn batch(&self) -> usize {
        self.inner.batch
    }
    #[getter]
    fn slots(&self) -> usize {
        self.inner.slots
    }
    #[getter]
    fn lr(&self) -> f32 {
        self.inner.lr
    }
    #[getter]
    fn weight_decay(&self) -> f32 {
        self.inner.weight_decay
    }
    #[getter]
    fn formula_weight(&self) -> f32 {
        self.inner.formula_weight
    }
    #[getter]
    fn seed(&self) -> u64 {
        self.inner.seed
    }
    #[getter]
    fn control(&self) -> &'static str {
        control_to_str(self.inner.control)
    }
    #[getter]
    fn grad_clip(&self) -> Option<f32> {
        self.inner.grad_clip
    }
    #[getter]
    fn gold_formula_conditioning(&self) -> &'static str {
        gold_conditioning_to_str(self.inner.gold_formula_conditioning)
    }
    #[getter]
    fn lambda_assign(&self) -> f32 {
        self.inner.lambda_assign
    }
    #[getter]
    fn ion_request_work_max(&self) -> u32 {
        self.inner.ion_request_work_max
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        format!(
            "TrainConfig(batch={}, slots={}, lr={}, seed={})",
            self.inner.batch, self.inner.slots, self.inner.lr, self.inner.seed
        )
    }
}

// ---------------------------------------------------------------------------
// Ms2Model
// ---------------------------------------------------------------------------

type Model = Ms2Model<R, f32>;
type ResidentTable = DeviceFormulaTable<R, f32>;

/// The composed MS2 model: spectrum encoder, formula head and graph-action
/// decoder, initialised together from one [`ModelConfig`]. The formula table
/// is uploaded once and reused; the generation workspace keeps one cached
/// bucket per shape.
#[pyclass(module = "mamba3_ms2", name = "Ms2Model", unsendable)]
pub struct PyMs2Model {
    model: Model,
    table: ResidentTable,
    workspace: GenerationWorkspace<R, f32>,
    constants: mamba3::tensor::ops::ms2::Ms2Constants<R>,
}

#[pymethods]
impl PyMs2Model {
    /// Build the encoder, formula head and decoder for `config` and upload
    /// `table` once.
    #[new]
    #[pyo3(signature = (config, table, seed = 0))]
    fn new(
        py: Python<'_>,
        config: &PyModelConfig,
        table: &PyFormulaTable,
        seed: u64,
    ) -> PyResult<Self> {
        let device = device();
        let (model, resident, constants) = detached(py, || {
            let resident = ResidentTable::upload(&table.inner, &device)?;
            let mut rng = Rng::seeded(seed);
            let model = Model::init(&config.inner, &device, &mut rng)?;
            Ok::<_, mamba3::error::Error>((
                model,
                resident,
                mamba3::tensor::ops::ms2::Ms2Constants::new(&device),
            ))
        })
        .py()?;
        Ok(Self {
            model,
            table: resident,
            workspace: GenerationWorkspace::new(),
            constants,
        })
    }

    /// Run the full generation pipeline: upload → peak selection → encoder →
    /// formula window → formula head → top-F → trajectory initialisation →
    /// sampling → validation → one batched read. The interpreter lock is
    /// released for the whole call (no Python object is borrowed across it).
    fn generate(
        &mut self,
        py: Python<'_>,
        batch: &PySpectrumBatch,
        config: &PyGenerationConfig,
    ) -> PyResult<PyCandidateBatch> {
        let inner = detached(py, || {
            self.model.generate(
                &batch.inner,
                &self.table,
                &config.inner,
                &mut self.workspace,
                &self.constants,
            )
        })
        .py()?;
        Ok(PyCandidateBatch { inner })
    }

    /// Run the full packed pipeline and return the ranked, compacted
    /// [`PackedCandidateBatch`](mamba3::models::ms2::pack::PackedCandidateBatch):
    /// `B * R` records in rank order. Exactly one device read. For the same
    /// request and seed the result equals the Rust-side `pack` of `generate`.
    fn generate_packed(
        &mut self,
        py: Python<'_>,
        batch: &PySpectrumBatch,
        config: &PyGenerationConfig,
    ) -> PyResult<PyPackedCandidateBatch> {
        let inner = detached(py, || {
            self.model.generate_packed(
                &batch.inner,
                &self.table,
                &config.inner,
                &mut self.workspace,
                &self.constants,
            )
        })
        .py()?;
        Ok(PyPackedCandidateBatch { inner })
    }

    /// Run the full packed pipeline with no device read and return the
    /// device-resident result. [`read`](PyResidentCandidates::read) performs
    /// exactly one read; [`release`](PyResidentCandidates::release) drops the
    /// leased buffers explicitly.
    fn generate_resident(
        &mut self,
        py: Python<'_>,
        batch: &PySpectrumBatch,
        config: &PyGenerationConfig,
    ) -> PyResult<PyResidentCandidates> {
        let inner = detached(py, || {
            self.model.generate_resident(
                &batch.inner,
                &self.table,
                &config.inner,
                &mut self.workspace,
                &self.constants,
            )
        })
        .py()?;
        Ok(PyResidentCandidates { inner: Some(inner) })
    }

    fn __repr__(&self) -> String {
        format!(
            "Ms2Model(d_model={}, max_atoms={}, table_rows={})",
            self.model.config.d_model,
            self.model.config.max_atoms,
            self.table.rows
        )
    }
}

// ---------------------------------------------------------------------------
// Ms2Trainer
// ---------------------------------------------------------------------------

type Trainer = Ms2Trainer<R, f32>;

/// The MS2 trainer: the composed model with its AdamW optimizer, the resident
/// formula table and the cached buffer buckets. One [`step`](Self::step) is
/// one optimizer step; [`request_report`](Self::request_report) asks the next
/// one to read its losses back (the only device read of the training loop).
#[pyclass(module = "mamba3_ms2", name = "Ms2Trainer", unsendable)]
pub struct PyMs2Trainer {
    inner: Trainer,
}

fn read_indices(value: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
    let (_, values) = read_int_values(value, "indices")?;
    values
        .into_iter()
        .map(|v| {
            usize::try_from(v)
                .map_err(|_| PyValueError::new_err(format!("indices value {v} is out of range")))
        })
        .collect()
}

#[pymethods]
impl PyMs2Trainer {
    /// Build the model, upload the table and prepare the optimizer. The model
    /// config's table reference is stamped with the uploaded table's row
    /// count and SHA-256. `seed` defaults to the training config's seed.
    #[new]
    #[pyo3(signature = (model_config, train_config, table, seed = None))]
    fn new(
        py: Python<'_>,
        model_config: &PyModelConfig,
        train_config: &PyTrainConfig,
        table: &PyFormulaTable,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        let device = device();
        let mut train = train_config.inner.clone();
        if let Some(seed) = seed {
            train.seed = seed;
        }
        let inner = detached(py, || {
            Trainer::new(&model_config.inner, &table.inner, &train, &device)
        })
        .py()?;
        Ok(Self { inner })
    }

    /// Load a checkpoint saved by [`save`](Self::save). `table` must be the
    /// same formula table the checkpoint was saved with.
    #[staticmethod]
    fn load(py: Python<'_>, path: &Bound<'_, PyAny>, table: &PyFormulaTable) -> PyResult<Self> {
        let path: String = path.extract().map_err(|_| {
            PyValueError::new_err(format!("path must be a string, got {}", describe(path)))
        })?;
        let device = device();
        let inner = detached(py, || {
            Trainer::load(std::path::Path::new(&path), &table.inner, &device)
        })
        .py()?;
        Ok(Self { inner })
    }

    /// Save the weights plus the configs and table reference that bind them
    /// (the optimizer state is absent in V0; load rebuilds a fresh AdamW).
    fn save(&self, path: &Bound<'_, PyAny>) -> PyResult<()> {
        let path: String = path.extract().map_err(|_| {
            PyValueError::new_err(format!("path must be a string, got {}", describe(path)))
        })?;
        self.inner.save(std::path::Path::new(&path)).py()
    }

    /// Ask the next [`step`](Self::step) to report its losses (exactly one
    /// batched read).
    fn request_report(&mut self) {
        self.inner.request_report();
    }

    /// One optimizer step over these spectra: forward pass, gradient
    /// computation and optimizer update. Returns the pre-update losses as a
    /// dict (`step`, `loss`, `graph`, `formula`, `assign`, `spectra`,
    /// `formula_present`, `formula_absent`, `gold_not_scored`,
    /// `assign_eligible`, `assign_partial`, `assign_dropped`,
    /// `assignment_label_overflow`) when a report
    /// was requested, else `None`. The interpreter lock is released for the
    /// whole call.
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        set: &PyExperimentSet,
        indices: &Bound<'py, PyAny>,
    ) -> PyResult<Option<Bound<'py, PyDict>>> {
        let indices = read_indices(indices)?;
        let report = detached(py, || self.inner.step(&set.inner, &indices)).py()?;
        match report {
            None => Ok(None),
            Some(rep) => {
                let dict = PyDict::new(py);
                dict.set_item("step", rep.step)?;
                dict.set_item("loss", rep.loss)?;
                dict.set_item("graph", rep.graph)?;
                dict.set_item("formula", rep.formula)?;
                dict.set_item("assign", rep.assign)?;
                dict.set_item("spectra", rep.spectra)?;
                dict.set_item("formula_present", rep.formula_present)?;
                dict.set_item("formula_absent", rep.formula_absent)?;
                dict.set_item("gold_not_scored", rep.gold_not_scored)?;
                dict.set_item("assign_eligible", rep.assign_eligible)?;
                dict.set_item("assign_partial", rep.assign_partial)?;
                dict.set_item("assign_dropped", rep.assign_dropped)?;
                dict.set_item(
                    "assignment_label_overflow",
                    rep.assignment_label_overflow,
                )?;
                Ok(Some(dict))
            }
        }
    }

    /// Teacher-forced evaluation of these spectra: no gradient, one batched
    /// read. Returns a dict of arrays (`nll`, `q`, `scored_tokens`,
    /// `gold_slot`, `gold_log_prob`, `molecules`) with the scalar counts
    /// (`spectra`, `slots`, `donor_same_molecule`,
    /// `donor_no_eligible_peaks`). The interpreter lock is released while the
    /// device work runs.
    fn teacher_eval<'py>(
        &mut self,
        py: Python<'py>,
        set: &PyExperimentSet,
        indices: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let indices = read_indices(indices)?;
        let eval = detached(py, || self.inner.teacher_eval(&set.inner, &indices)).py()?;
        let dict = PyDict::new(py);
        dict.set_item("nll", f32_array(py, &eval.nll))?;
        dict.set_item("q", f32_array(py, &eval.q))?;
        dict.set_item("scored_tokens", u32_array(py, &eval.scored_tokens))?;
        dict.set_item("gold_slot", u32_array(py, &eval.gold_slot))?;
        dict.set_item("gold_log_prob", f32_array(py, &eval.gold_log_prob))?;
        let molecules: Vec<i64> = eval.molecules.into_iter().map(|m| m as i64).collect();
        dict.set_item("molecules", PyArray1::from_vec(py, molecules))?;
        dict.set_item("spectra", eval.spectra)?;
        dict.set_item("slots", eval.slots)?;
        dict.set_item("donor_same_molecule", eval.donor_same_molecule)?;
        dict.set_item("donor_no_eligible_peaks", eval.donor_no_eligible_peaks)?;
        Ok(dict)
    }

    /// Optimizer steps completed so far.
    #[getter]
    fn step_count(&self) -> u64 {
        self.inner.step_count()
    }

    /// SHA-256 of the resident formula table.
    #[getter]
    fn table_sha256(&self) -> &str {
        self.inner.table_sha256()
    }

    /// The training hyperparameters.
    #[getter]
    fn train_config(&self) -> PyTrainConfig {
        PyTrainConfig {
            inner: self.inner.train_config().clone(),
        }
    }

    fn __repr__(&self) -> String {
        format!("Ms2Trainer(steps={})", self.inner.step_count())
    }
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Status-code helpers, for reading `validate` and `generate` outputs without
/// a lookup table.
fn request_names(_bits: u32) -> Vec<String> {
    mamba3::models::ms2::contract::request_status::names(_bits)
        .into_iter()
        .map(str::to_string)
        .collect()
}

fn candidate_names(_bits: u32) -> Vec<String> {
    mamba3::models::ms2::contract::candidate_status::names(_bits)
        .into_iter()
        .map(str::to_string)
        .collect()
}

/// Names of the set request-status bits, in bit order.
#[pyfunction]
fn request_status_names(bits: u32) -> Vec<String> {
    request_names(bits)
}

/// Names of the set candidate-status bits, in bit order.
#[pyfunction]
fn candidate_status_names(bits: u32) -> Vec<String> {
    candidate_names(bits)
}

// ---------------------------------------------------------------------------
// PackedCandidateBatch
// ---------------------------------------------------------------------------

/// Ranked, compacted candidates of a batch (architecture §4.4): `B * R`
/// records in `(spectrum, rank)` order. Every field is a NumPy array with the
/// contract's dtype and shape.
#[pyclass(module = "mamba3_ms2", name = "PackedCandidateBatch", unsendable)]
pub struct PyPackedCandidateBatch {
    inner: PackedCandidateBatch,
}

#[pymethods]
impl PyPackedCandidateBatch {
    /// Rebuild from [`to_json`](Self::to_json).
    #[staticmethod]
    fn from_json(json: &str) -> PyResult<Self> {
        Ok(Self {
            inner: parse_json(json, serde_json::from_value)?,
        })
    }

    /// The batch as JSON.
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner)
            .map_err(|e| PyValueError::new_err(format!("PackedCandidateBatch.to_json: {e}")))
    }

    /// Check every invariant of architecture §4.4.
    fn validate(&self) -> PyResult<()> {
        self.inner.validate().py()
    }

    #[getter]
    fn schema_version(&self) -> u32 {
        self.inner.schema_version
    }
    #[getter]
    fn batch(&self) -> usize {
        self.inner.batch
    }
    #[getter]
    fn returned(&self) -> usize {
        self.inner.returned
    }
    #[getter]
    fn trajectories(&self) -> usize {
        self.inner.trajectories
    }
    #[getter]
    fn max_steps(&self) -> usize {
        self.inner.max_steps
    }
    #[getter]
    fn max_atoms(&self) -> usize {
        self.inner.max_atoms
    }
    #[getter]
    fn max_ring_closures(&self) -> usize {
        self.inner.max_ring_closures
    }
    #[getter]
    fn spectrum_id<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u64>> {
        u64_array(py, &self.inner.spectrum_id)
    }
    #[getter]
    fn trajectory<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        u32_array(py, &self.inner.trajectory)
    }
    #[getter]
    fn actions<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let n = self.inner.batch * self.inner.returned;
        let flat = PyArray1::from_vec(py, self.inner.actions.clone());
        Ok(flat.reshape((n, self.inner.max_steps, 4))?.into_any())
    }
    #[getter]
    fn length<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        u32_array(py, &self.inner.length)
    }
    #[getter]
    fn formula_row<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        u32_array(py, &self.inner.formula_row)
    }
    #[getter]
    fn formula_rank<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        u32_array(py, &self.inner.formula_rank)
    }
    #[getter]
    fn formula_counts<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let n = self.inner.batch * self.inner.returned;
        let flat = PyArray1::from_vec(py, self.inner.formula_counts.clone());
        Ok(flat.reshape((n, 10))?.into_any())
    }
    #[getter]
    fn formula_log_prob<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        f32_array(py, &self.inner.formula_log_prob)
    }
    #[getter]
    fn trace_log_prob<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        f32_array(py, &self.inner.trace_log_prob)
    }
    #[getter]
    fn score<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        f32_array(py, &self.inner.score)
    }
    #[getter]
    fn open_valence<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let n = self.inner.batch * self.inner.returned;
        let flat = PyArray1::from_vec(py, self.inner.open_valence.clone());
        Ok(flat.reshape((n, self.inner.max_atoms))?.into_any())
    }
    #[getter]
    fn status<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        u32_array(py, &self.inner.status)
    }
    #[getter]
    fn evidence_status<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u8>> {
        u8_array(py, &self.inner.evidence_status)
    }
    #[getter]
    fn evidence_count<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u8>> {
        u8_array(py, &self.inner.evidence_count)
    }
    #[getter]
    fn evidence_peak_id<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let n = self.inner.batch * self.inner.returned;
        let flat = PyArray1::from_vec(py, self.inner.evidence_peak_id.clone());
        Ok(flat.reshape((n, 4))?.into_any())
    }
    #[getter]
    fn evidence_hypothesis<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let n = self.inner.batch * self.inner.returned;
        let flat = PyArray1::from_vec(py, self.inner.evidence_hypothesis.clone());
        Ok(flat.reshape((n, 4))?.into_any())
    }
    #[getter]
    fn evidence_shift<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let n = self.inner.batch * self.inner.returned;
        let flat = PyArray1::from_vec(py, self.inner.evidence_shift.clone());
        Ok(flat.reshape((n, 4))?.into_any())
    }
    #[getter]
    fn evidence_residual<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let n = self.inner.batch * self.inner.returned;
        let flat = PyArray1::from_vec(py, self.inner.evidence_residual.clone());
        Ok(flat.reshape((n, 4))?.into_any())
    }
    #[getter]
    fn evidence_log_prob<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let n = self.inner.batch * self.inner.returned;
        let flat = PyArray1::from_vec(py, self.inner.evidence_log_prob.clone());
        Ok(flat.reshape((n, 4))?.into_any())
    }
    #[getter]
    fn identity_resolution<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u8>> {
        u8_array(py, &self.inner.identity_resolution)
    }
    #[getter]
    fn attachment_partition<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u8>> {
        u8_array(py, &self.inner.attachment_partition)
    }
    #[getter]
    fn returned_count<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        u32_array(py, &self.inner.returned_count)
    }
    #[getter]
    fn request_status<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        u32_array(py, &self.inner.request_status)
    }
    #[getter]
    fn rows_visited<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        u32_array(py, &self.inner.rows_visited)
    }
    #[getter]
    fn rows_joined<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        u32_array(py, &self.inner.rows_joined)
    }
    #[getter]
    fn rows_scored<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        u32_array(py, &self.inner.rows_scored)
    }
    #[getter]
    fn formula_support_complete<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u8>> {
        u8_array(py, &self.inner.formula_support_complete)
    }
    #[getter]
    fn formula_mass_retained<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        f32_array(py, &self.inner.formula_mass_retained)
    }
    #[getter]
    fn peaks_kept<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        u32_array(py, &self.inner.peaks_kept)
    }
    #[getter]
    fn intensity_retained<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        f32_array(py, &self.inner.intensity_retained)
    }
    #[getter]
    fn formula_source<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u8>> {
        u8_array(py, &self.inner.formula_source)
    }
}

// ---------------------------------------------------------------------------
// ResidentCandidates
// ---------------------------------------------------------------------------

/// Device-resident packed candidates (architecture §4.4): owns its device
/// buffers with no host copy yet. [`read`](Self::read) performs exactly one
/// device read and returns the [`PackedCandidateBatch`](PyPackedCandidateBatch);
/// [`release`](Self::release) drops the leased buffers explicitly.
#[pyclass(module = "mamba3_ms2", name = "ResidentCandidates", unsendable)]
pub struct PyResidentCandidates {
    inner: Option<ResidentCandidates<R, f32>>,
}

#[pymethods]
impl PyResidentCandidates {
    /// Perform exactly one device read and return the packed batch. `model`
    /// is the model that produced this result (it carries the table and the
    /// config context the host assembly needs).
    fn read(&self, py: Python<'_>, model: &PyMs2Model) -> PyResult<PyPackedCandidateBatch> {
        let inner = self
            .inner
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("ResidentCandidates was released"))?;
        let out = detached(py, || inner.read(&model.model)).py()?;
        Ok(PyPackedCandidateBatch { inner: out })
    }

    /// Drop the leased buffers explicitly (plain object drop frees them the
    /// same way).
    fn release(mut slf: PyRefMut<'_, Self>) {
        slf.inner = None;
    }
}

/// Register the MS2 classes in the extension module.
pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PySpectrumBatch>()?;
    module.add_class::<PyModelConfig>()?;
    module.add_class::<PyGenerationConfig>()?;
    module.add_class::<PyChemistryDomain>()?;
    module.add_class::<PyFormulaTable>()?;
    module.add_class::<PyCandidateBatch>()?;
    module.add_class::<PyPackedCandidateBatch>()?;
    module.add_class::<PyResidentCandidates>()?;
    module.add_class::<PyExperimentSet>()?;
    module.add_class::<PyTrainConfig>()?;
    module.add_class::<PyMs2Model>()?;
    module.add_class::<PyMs2Trainer>()?;
    module.add_function(wrap_pyfunction!(request_status_names, module)?)?;
    module.add_function(wrap_pyfunction!(candidate_status_names, module)?)?;
    // The contract's element order, for callers that build compositions:
    // `[{"index", "symbol", "exact", "mass"}]` in `ELEMENTS` order.
    let elements: Vec<HashMap<String, serde_json::Value>> = mamba3::models::ms2::chem::ELEMENTS
        .iter()
        .enumerate()
        .map(|(i, e)| {
            let mut m = HashMap::new();
            m.insert("index".to_string(), serde_json::Value::from(i as u64));
            m.insert("symbol".to_string(), serde_json::Value::from(e.symbol));
            m.insert("exact".to_string(), serde_json::Value::from(e.exact));
            m.insert("mass".to_string(), serde_json::Value::from(e.mass as u64));
            m
        })
        .collect();
    let elements_json = serde_json::to_value(&elements).unwrap_or(serde_json::Value::Null);
    module.add("ELEMENTS", json_to_py(module.py(), &elements_json))?;
    Ok(())
}
