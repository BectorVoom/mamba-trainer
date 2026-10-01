//! Graph Mamba: the classes behind the `mamba3_graph` import module
//! (GRAPH_MAMBA_PLAN.md §2.6).
//!
//! Same extension library, same device client and same counters as the rest of
//! the bindings; only the import name differs. The rules this file follows:
//!
//! * **One call per unit of work.** `train_epoch`, `evaluate` and `predict` each
//!   run their whole loop in Rust — not to save call overhead (a bound call is
//!   tens of nanoseconds) but so that Rust owns the loop: it can release the
//!   interpreter lock across it, check for interrupts at safe points and batch
//!   the reads.
//! * **The interpreter lock is released whenever a call waits on the device**
//!   or computes on the host without touching Python objects ([`detached`]).
//! * **Interruptible.** `train_epoch` checks for signals between steps and
//!   stops after a completed optimizer step.
//! * **Ingest is one pass per array, makes no device read and uploads by
//!   value.** Arrays are borrowed in their own dtype and read in logical order
//!   ([`Held`]); the core canonicalises straight into the buffers it uploads.
//! * **One read, one pass on the way out.**
//! * **No Python in the loop**: schedules, samplers and metrics are Rust
//!   objects configured from Python.
//! * **A debug build says so.**

use std::ops::ControlFlow;
use std::time::{Duration, Instant};

use mamba3::backend::{DType, Device, FloatElem};
use mamba3::models::graph::{
    BatchMode, Bools, DatasetOptions, EvalOptions, FeatureSpec, FeaturesView, Floats, GraphDataView,
    GraphDataset, GraphMamba, GraphMambaSpec, GraphPool, GraphTaskSpec, GraphTrainConfig,
    GraphTrainer, HostCsr, Ints, LAPLACIAN_MAX_NODES, LabelsView, LocalEncoder, Metric, MpnnKind,
    NodeOrder, PreparedDataset, RWSE_MAX_BALL, RegressionLoss, Split, SplitsView, TokenSampling,
    TokenTail, graph_offsets_of, laplacian_pe_csr, rwse_csr,
};
use mamba3::models::vision::ScanDirection;
use mamba3::nn::Module;
use numpy::{
    PyArray1, PyArray2, PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn, PyUntypedArrayMethods,
};
use pyo3::exceptions::{
    PyFloatingPointError, PyNotImplementedError, PyRuntimeWarning, PyValueError,
};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::R;
use crate::config::PyLrSchedule;
use crate::err::IntoPyResult;

// ---------------------------------------------------------------------------
// Releasing the interpreter lock
// ---------------------------------------------------------------------------

/// A value carried into and out of a section that runs without the interpreter
/// lock.
struct Detached<T>(T);

// SAFETY: `Python::detach` wants a `Send` closure because a free-threaded build
// could in principle move it; it does not — the closure runs on the calling
// thread, and so does everything it borrows. What it borrows here is the Rust
// state of `unsendable` Python objects (they hold `Rc`s), reached through a
// `PyRef` / `PyRefMut` the calling method holds for the whole call:
//
// * nothing crosses a thread, so no `Rc` count is touched from two threads;
// * no `Bound`, `Py` or `PyRef` is captured, and nothing a closure does touches
//   a Python object without the lock: NumPy memory is read before detaching,
//   and the one place that needs Python (the signal check of `train_epoch`)
//   re-attaches for it. The only Python value a closure returns is the `PyErr`
//   a signal handler raised, which is `Send` and whose reference pyo3 releases
//   once the lock is held again;
// * an `unsendable` object refuses access from any other thread (pyo3's thread
//   checker), and the borrow flag refuses re-entry from this one, so while the
//   lock is released nothing else can reach the same state — and every object
//   that shares state with the model (the dataset shares the store) is
//   `unsendable` too.
unsafe impl<T> Send for Detached<T> {}

/// Run `body` with the interpreter lock released, on this thread.
///
/// `body` must not touch Python objects. An error is carried out as a Rust
/// value and converted by the caller once the lock is held again.
fn detached<T>(py: Python<'_>, body: impl FnOnce() -> T) -> T {
    let body = Detached(body);
    py.detach(move || {
        // Capture the wrapper, not the closure inside it.
        let body = body;
        Detached((body.0)())
    })
    .0
}

thread_local! {
    /// The device every graph object of this thread lives on.
    static DEVICE: Device<R> = Device::<R>::default();
}

/// This thread's device. One handle for the datasets and the models, so that
/// they share one identity: per-device caches then hold one copy of each
/// table, and kernels that batch tensors of one device see them as such.
fn device() -> Device<R> {
    DEVICE.with(Clone::clone)
}

// ---------------------------------------------------------------------------
// Borrowing NumPy arrays in their own dtype, in logical order
// ---------------------------------------------------------------------------

/// A NumPy array borrowed in its own dtype, viewed as a slice in logical
/// order.
///
/// A slice is memory order, which is logical order only for a C-contiguous
/// array, and it needs aligned elements. [`as_array`] hands over only arrays
/// that are both — the caller's own, or the copy NumPy makes of a
/// Fortran-ordered, sliced or unaligned one — and this checks it.
struct Held<'py, T: numpy::Element> {
    guard: PyReadonlyArrayDyn<'py, T>,
}

impl<'py, T: numpy::Element + Copy> Held<'py, T> {
    fn new(array: &Bound<'py, PyArrayDyn<T>>, key: &str) -> PyResult<Self> {
        let guard = array.try_readonly().map_err(|err| {
            PyValueError::new_err(format!("{key} is being written elsewhere: {err}"))
        })?;
        if guard.as_slice().is_err() {
            return Err(PyValueError::new_err(format!(
                "{key} cannot be read as a C-contiguous, aligned array"
            )));
        }
        Ok(Self { guard })
    }

    fn slice(&self) -> &[T] {
        self.guard.as_slice().expect("checked when it was borrowed")
    }
}

/// A float array of any width.
enum HeldFloats<'py> {
    F16(Held<'py, half::f16>),
    F32(Held<'py, f32>),
    F64(Held<'py, f64>),
}

impl HeldFloats<'_> {
    fn view(&self) -> Floats<'_> {
        match self {
            HeldFloats::F16(a) => Floats::F16(a.slice()),
            HeldFloats::F32(a) => Floats::F32(a.slice()),
            HeldFloats::F64(a) => Floats::F64(a.slice()),
        }
    }
}

/// An integer array of any width.
enum HeldInts<'py> {
    I8(Held<'py, i8>),
    I16(Held<'py, i16>),
    I32(Held<'py, i32>),
    I64(Held<'py, i64>),
    U8(Held<'py, u8>),
    U16(Held<'py, u16>),
    U32(Held<'py, u32>),
    U64(Held<'py, u64>),
}

impl HeldInts<'_> {
    fn view(&self) -> Ints<'_> {
        match self {
            HeldInts::I8(a) => Ints::I8(a.slice()),
            HeldInts::I16(a) => Ints::I16(a.slice()),
            HeldInts::I32(a) => Ints::I32(a.slice()),
            HeldInts::I64(a) => Ints::I64(a.slice()),
            HeldInts::U8(a) => Ints::U8(a.slice()),
            HeldInts::U16(a) => Ints::U16(a.slice()),
            HeldInts::U32(a) => Ints::U32(a.slice()),
            HeldInts::U64(a) => Ints::U64(a.slice()),
        }
    }

    /// The two rows of a `[2, len]` array.
    fn halves(&self, len: usize) -> (Ints<'_>, Ints<'_>) {
        macro_rules! split {
            ($variant:ident, $a:expr) => {{
                let (first, second) = $a.slice().split_at(len);
                (Ints::$variant(first), Ints::$variant(second))
            }};
        }
        match self {
            HeldInts::I8(a) => split!(I8, a),
            HeldInts::I16(a) => split!(I16, a),
            HeldInts::I32(a) => split!(I32, a),
            HeldInts::I64(a) => split!(I64, a),
            HeldInts::U8(a) => split!(U8, a),
            HeldInts::U16(a) => split!(U16, a),
            HeldInts::U32(a) => split!(U32, a),
            HeldInts::U64(a) => split!(U64, a),
        }
    }
}

/// A mask: booleans or bytes.
enum HeldBools<'py> {
    Bool(Held<'py, bool>),
    U8(Held<'py, u8>),
}

impl HeldBools<'_> {
    fn view(&self) -> Bools<'_> {
        match self {
            HeldBools::Bool(a) => Bools::Bool(a.slice()),
            HeldBools::U8(a) => Bools::U8(a.slice()),
        }
    }
}

/// `value` as a NumPy array a slice can view: C-contiguous and aligned.
///
/// That is the array itself when it already is one. A list goes through
/// `numpy.asarray`; a Fortran-ordered, sliced or unaligned array is copied once
/// by `numpy.require`, in logical order and in its own dtype. The caller's
/// array is never modified.
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

/// Borrow a float array (`float16`, `float32` or `float64`) and its shape.
fn hold_floats<'py>(
    value: &Bound<'py, PyAny>,
    key: &str,
) -> PyResult<(Vec<usize>, HeldFloats<'py>)> {
    let array = as_array(value)?;
    macro_rules! attempt {
        ($ty:ty, $variant:ident) => {
            if let Ok(typed) = array.cast::<PyArrayDyn<$ty>>() {
                return Ok((typed.shape().to_vec(), HeldFloats::$variant(Held::new(typed, key)?)));
            }
        };
    }
    attempt!(f32, F32);
    attempt!(f64, F64);
    attempt!(half::f16, F16);
    Err(PyValueError::new_err(format!(
        "{key} must be an array of floats (float16, float32 or float64), got {}",
        describe(&array)
    )))
}

/// Borrow an integer array of any width and its shape.
fn hold_ints<'py>(value: &Bound<'py, PyAny>, key: &str) -> PyResult<(Vec<usize>, HeldInts<'py>)> {
    let array = as_array(value)?;
    macro_rules! attempt {
        ($ty:ty, $variant:ident) => {
            if let Ok(typed) = array.cast::<PyArrayDyn<$ty>>() {
                return Ok((typed.shape().to_vec(), HeldInts::$variant(Held::new(typed, key)?)));
            }
        };
    }
    attempt!(i64, I64);
    attempt!(i32, I32);
    attempt!(u32, U32);
    attempt!(u8, U8);
    attempt!(i16, I16);
    attempt!(i8, I8);
    attempt!(u16, U16);
    attempt!(u64, U64);
    Err(PyValueError::new_err(format!(
        "{key} must be an array of integers, got {}",
        describe(&array)
    )))
}

/// Borrow a mask (`bool` or `uint8`) and its shape.
fn hold_bools<'py>(
    value: &Bound<'py, PyAny>,
    key: &str,
) -> PyResult<(Vec<usize>, HeldBools<'py>)> {
    let array = as_array(value)?;
    if let Ok(typed) = array.cast::<PyArrayDyn<bool>>() {
        return Ok((typed.shape().to_vec(), HeldBools::Bool(Held::new(typed, key)?)));
    }
    if let Ok(typed) = array.cast::<PyArrayDyn<u8>>() {
        return Ok((typed.shape().to_vec(), HeldBools::U8(Held::new(typed, key)?)));
    }
    Err(PyValueError::new_err(format!(
        "{key} must be a boolean array, got {}",
        describe(&array)
    )))
}

fn shape_error(key: &str, want: &str, got: &[usize]) -> PyErr {
    PyValueError::new_err(format!("{key} must be {want}, got shape {got:?}"))
}

/// A feature table as the spec says it is laid out.
enum HeldFeatures<'py> {
    Float(usize, HeldFloats<'py>),
    Categorical(Vec<usize>, HeldInts<'py>),
}

impl HeldFeatures<'_> {
    fn view(&self) -> FeaturesView<'_> {
        match self {
            HeldFeatures::Float(dim, data) => FeaturesView::Float {
                dim: *dim,
                data: data.view(),
            },
            HeldFeatures::Categorical(vocab, ids) => FeaturesView::Categorical {
                fields: vocab.len(),
                vocab,
                ids: ids.view(),
            },
        }
    }
}

/// Borrow a `[rows, width]` feature array laid out as `spec` says; returns it
/// with its row count.
fn hold_features<'py>(
    value: &Bound<'py, PyAny>,
    spec: &FeatureSpec,
    key: &str,
) -> PyResult<(usize, HeldFeatures<'py>)> {
    match spec {
        FeatureSpec::Float { dim } => {
            let (shape, data) = hold_floats(value, key)?;
            if shape.len() != 2 || shape[1] != *dim {
                return Err(shape_error(key, &format!("[rows, {dim}] floats"), &shape));
            }
            Ok((shape[0], HeldFeatures::Float(*dim, data)))
        }
        FeatureSpec::Categorical { vocab } => {
            let (shape, ids) = hold_ints(value, key)?;
            if shape.len() != 2 || shape[1] != vocab.len() {
                return Err(shape_error(
                    key,
                    &format!("[rows, {}] integer ids", vocab.len()),
                    &shape,
                ));
            }
            Ok((shape[0], HeldFeatures::Categorical(vocab.clone(), ids)))
        }
    }
}

/// The targets as the task says they are laid out.
enum HeldLabels<'py> {
    Node(HeldInts<'py>),
    GraphClass(HeldInts<'py>),
    Graph(usize, HeldFloats<'py>),
    None,
}

/// Every array of a dataset dict, borrowed for the length of the ingest.
struct Ingest<'py> {
    n_nodes: usize,
    n_edges: usize,
    edges: HeldInts<'py>,
    x: HeldFeatures<'py>,
    edge_attr: Option<HeldFeatures<'py>>,
    pe: Option<(usize, HeldFloats<'py>)>,
    y: HeldLabels<'py>,
    graph_ptr: Option<HeldInts<'py>>,
    masks: [Option<HeldBools<'py>>; 3],
    /// What a mask that was not given reads as.
    no_mask: Vec<bool>,
}

const KEYS: [&str; 9] = [
    "edge_index",
    "x",
    "edge_attr",
    "y",
    "graph_ptr",
    "train_mask",
    "val_mask",
    "test_mask",
    "pe",
];

impl<'py> Ingest<'py> {
    /// Borrow the arrays of `arrays` for `spec`. Unknown keys, wrong dtypes and
    /// wrong shapes raise `ValueError` naming the key; nothing is copied unless
    /// an array is not C-contiguous.
    fn new(spec: &GraphMambaSpec, arrays: &Bound<'py, PyDict>) -> PyResult<Self> {
        for (key, _) in arrays.iter() {
            let key: String = key.extract().map_err(|_| {
                PyValueError::new_err("the dataset's keys must be strings".to_string())
            })?;
            if !KEYS.contains(&key.as_str()) {
                return Err(PyValueError::new_err(format!(
                    "unknown key {key:?}; a graph dataset takes {KEYS:?}"
                )));
            }
        }
        let get = |key: &str| -> PyResult<Option<Bound<'py, PyAny>>> {
            Ok(arrays.get_item(key)?.filter(|value| !value.is_none()))
        };
        let need = |key: &str| -> PyResult<Bound<'py, PyAny>> {
            get(key)?.ok_or_else(|| PyValueError::new_err(format!("{key} is required")))
        };

        let (n_nodes, x) = hold_features(&need("x")?, &spec.node_features, "x")?;
        let (shape, edges) = hold_ints(&need("edge_index")?, "edge_index")?;
        if shape.len() != 2 || shape[0] != 2 {
            return Err(shape_error(
                "edge_index",
                "[2, edges] integers (sources, then destinations)",
                &shape,
            ));
        }
        let n_edges = shape[1];

        let edge_attr = match (get("edge_attr")?, &spec.edge_features) {
            (Some(value), Some(edge_spec)) => {
                let (rows, held) = hold_features(&value, edge_spec, "edge_attr")?;
                if rows != n_edges {
                    return Err(PyValueError::new_err(format!(
                        "edge_attr has {rows} rows for {n_edges} edges"
                    )));
                }
                Some(held)
            }
            (None, Some(_)) => {
                return Err(PyValueError::new_err(
                    "edge_attr is required: the spec has edge features".to_string(),
                ));
            }
            // Edge features the spec does not read are not ingested.
            (_, None) => None,
        };

        let pe = match get("pe")? {
            Some(value) => {
                let (shape, data) = hold_floats(&value, "pe")?;
                if shape.len() != 2 || shape[0] != n_nodes {
                    return Err(shape_error("pe", &format!("[{n_nodes}, pe_dim] floats"), &shape));
                }
                Some((shape[1], data))
            }
            None => None,
        };

        let graph_ptr = match get("graph_ptr")? {
            Some(value) => {
                let (shape, held) = hold_ints(&value, "graph_ptr")?;
                if shape.len() != 1 {
                    return Err(shape_error("graph_ptr", "[graphs + 1] integers", &shape));
                }
                Some((shape[0], held))
            }
            None => None,
        };
        let n_graphs = graph_ptr.as_ref().map_or(1, |(len, _)| len.saturating_sub(1));
        let graph_ptr = graph_ptr.map(|(_, held)| held);

        let (y, items) = match (get("y")?, &spec.task) {
            (None, _) => (HeldLabels::None, 0),
            (Some(value), GraphTaskSpec::NodeClass { .. }) => {
                let (shape, ids) = hold_ints(&value, "y")?;
                if shape != [n_nodes] {
                    return Err(shape_error(
                        "y",
                        &format!("[{n_nodes}] integer classes (-1 where unlabelled)"),
                        &shape,
                    ));
                }
                (HeldLabels::Node(ids), n_nodes)
            }
            (Some(value), GraphTaskSpec::GraphClass { .. }) => {
                let (shape, ids) = hold_ints(&value, "y")?;
                if shape != [n_graphs] {
                    return Err(shape_error(
                        "y",
                        &format!("[{n_graphs}] integer classes (-1 where unlabelled)"),
                        &shape,
                    ));
                }
                (HeldLabels::GraphClass(ids), n_graphs)
            }
            (
                Some(value),
                GraphTaskSpec::GraphRegression { targets, .. }
                | GraphTaskSpec::GraphMultiLabel {
                    labels: targets, ..
                },
            ) => {
                let (shape, values) = hold_floats(&value, "y")?;
                let flat = *targets == 1 && shape == [n_graphs];
                if !flat && shape != [n_graphs, *targets] {
                    return Err(shape_error(
                        "y",
                        &format!("[{n_graphs}, {targets}] floats (NaN where missing)"),
                        &shape,
                    ));
                }
                (HeldLabels::Graph(*targets, values), n_graphs)
            }
        };

        let mut masks = [None, None, None];
        for (slot, key) in masks.iter_mut().zip(["train_mask", "val_mask", "test_mask"]) {
            if let Some(value) = get(key)? {
                if matches!(y, HeldLabels::None) {
                    return Err(PyValueError::new_err(format!(
                        "{key} is given without y: a split selects targets"
                    )));
                }
                let (shape, held) = hold_bools(&value, key)?;
                if shape != [items] {
                    return Err(shape_error(key, &format!("[{items}] booleans"), &shape));
                }
                *slot = Some(held);
            }
        }
        Ok(Self {
            n_nodes,
            n_edges,
            edges,
            x,
            edge_attr,
            pe,
            y,
            graph_ptr,
            masks,
            no_mask: vec![false; items],
        })
    }

    /// Mask `index` (train, val, test); all false when it was not given.
    fn mask(&self, index: usize) -> Bools<'_> {
        match &self.masks[index] {
            Some(held) => held.view(),
            None => Bools::Bool(&self.no_mask),
        }
    }

    /// The borrowed arrays as the view the core consumes.
    fn view(&self) -> GraphDataView<'_> {
        let (edge_src, edge_dst) = self.edges.halves(self.n_edges);
        GraphDataView {
            n_nodes: self.n_nodes,
            edge_src,
            edge_dst,
            x: self.x.view(),
            edge_attr: self.edge_attr.as_ref().map(HeldFeatures::view),
            pe: self.pe.as_ref().map(|(dim, data)| (*dim, data.view())),
            y: match &self.y {
                HeldLabels::Node(ids) => LabelsView::Node(ids.view()),
                HeldLabels::GraphClass(ids) => LabelsView::GraphClass(ids.view()),
                HeldLabels::Graph(targets, values) => LabelsView::Graph {
                    targets: *targets,
                    values: values.view(),
                },
                HeldLabels::None => LabelsView::None,
            },
            graph_ptr: self.graph_ptr.as_ref().map(HeldInts::view),
            // With no mask at all, every labelled item trains (the core's
            // default); with some, the ones not given are empty.
            masks: self.masks.iter().any(Option::is_some).then(|| SplitsView {
                train: self.mask(0),
                val: self.mask(1),
                test: self.mask(2),
            }),
        }
    }
}

// ---------------------------------------------------------------------------
// Spec
// ---------------------------------------------------------------------------

/// Categorical node or edge features: one integer id per field, field `f`
/// taking values in `0..vocab[f]`.
#[pyclass(module = "mamba3_graph", name = "Categorical", from_py_object)]
#[derive(Clone)]
pub struct PyCategorical {
    vocab: Vec<usize>,
}

#[pymethods]
impl PyCategorical {
    /// `vocab` is the vocabulary size of each id field.
    #[new]
    fn new(vocab: Vec<usize>) -> Self {
        Self { vocab }
    }

    /// The vocabulary size of each field.
    #[getter]
    fn vocab(&self) -> Vec<usize> {
        self.vocab.clone()
    }

    fn __repr__(&self) -> String {
        format!("Categorical({:?})", self.vocab)
    }
}

/// `int` (that many floats per row) or [`PyCategorical`].
fn feature_spec(value: &Bound<'_, PyAny>, name: &str) -> PyResult<FeatureSpec> {
    if let Ok(categorical) = value.extract::<PyCategorical>() {
        return Ok(FeatureSpec::Categorical {
            vocab: categorical.vocab,
        });
    }
    let dim: usize = value.extract().map_err(|_| {
        PyValueError::new_err(format!(
            "{name} must be an int (floats per row) or a Categorical"
        ))
    })?;
    Ok(FeatureSpec::Float { dim })
}

/// What a model predicts. Build one with `NodeClassification`,
/// `GraphClassification`, `GraphRegression` or `GraphMultiLabel`.
#[pyclass(module = "mamba3_graph", name = "GraphTask", from_py_object)]
#[derive(Clone)]
pub struct PyGraphTask {
    inner: GraphTaskSpec,
}

#[pymethods]
impl PyGraphTask {
    /// Width of the model's output.
    #[getter]
    fn outputs(&self) -> usize {
        self.inner.outputs()
    }

    /// Whether the model predicts per graph rather than per node.
    #[getter]
    fn per_graph(&self) -> bool {
        self.inner.per_graph()
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
}

fn parse_pool(pool: &str) -> PyResult<GraphPool> {
    match pool {
        "mean" => Ok(GraphPool::Mean),
        "sum" => Ok(GraphPool::Sum),
        other => Err(PyValueError::new_err(format!(
            "pool must be 'mean' or 'sum', got {other:?}"
        ))),
    }
}

/// One of `classes` per node.
#[pyfunction(name = "NodeClassification")]
fn node_classification(classes: usize) -> PyGraphTask {
    PyGraphTask {
        inner: GraphTaskSpec::NodeClass { classes },
    }
}

/// One of `classes` per graph, read out of the pooled nodes.
#[pyfunction(name = "GraphClassification")]
#[pyo3(signature = (classes, *, pool = "mean"))]
fn graph_classification(classes: usize, pool: &str) -> PyResult<PyGraphTask> {
    Ok(PyGraphTask {
        inner: GraphTaskSpec::GraphClass {
            classes,
            pool: parse_pool(pool)?,
        },
    })
}

/// `targets` floats per graph; `loss` is `"l1"` or `"mse"`.
#[pyfunction(name = "GraphRegression")]
#[pyo3(signature = (targets, *, pool = "mean", loss = "l1"))]
fn graph_regression(targets: usize, pool: &str, loss: &str) -> PyResult<PyGraphTask> {
    let loss = match loss {
        "l1" => RegressionLoss::L1,
        "mse" => RegressionLoss::Mse,
        other => {
            return Err(PyValueError::new_err(format!(
                "loss must be 'l1' or 'mse', got {other:?}"
            )));
        }
    };
    Ok(PyGraphTask {
        inner: GraphTaskSpec::GraphRegression {
            targets,
            pool: parse_pool(pool)?,
            loss,
        },
    })
}

/// `labels` independent binary labels per graph.
#[pyfunction(name = "GraphMultiLabel")]
#[pyo3(signature = (labels, *, pool = "mean"))]
fn graph_multi_label(labels: usize, pool: &str) -> PyResult<PyGraphTask> {
    Ok(PyGraphTask {
        inner: GraphTaskSpec::GraphMultiLabel {
            labels,
            pool: parse_pool(pool)?,
        },
    })
}

/// A Graph Mamba model, completely: features, the token shape, both stages'
/// depth, the message passing, the node order and the task.
#[pyclass(module = "mamba3_graph", name = "GraphMambaSpec", from_py_object)]
#[derive(Clone)]
pub struct PyGraphMambaSpec {
    pub(crate) inner: GraphMambaSpec,
}

#[pymethods]
impl PyGraphMambaSpec {
    /// Build and validate a spec; a `ValueError` names the argument at fault.
    ///
    /// `node_features` and `edge_features` are an `int` (floats per row) or a
    /// `Categorical`. `max_hops`, `walks` and `repeats` are the paper's `m`,
    /// `M` and `s`; `max_hops = 0` gives node tokens only. `token_layers`
    /// defaults to 1 with walk tokens and 0 without. Heads are per direction.
    #[new]
    #[pyo3(signature = (*, node_features, task, edge_features = None, pe_dim = 0,
                        pe_sign_flip = None, d_model = 64, max_hops = 4, walks = 8, repeats = 4,
                        token_sampling = "step", local = "sgc", token_layers = None,
                        token_tail = "forward", node_layers = 2, mpnn = None, d_state = 8,
                        token_heads = 1, node_heads = 1, node_sequences = 1,
                        bidirectional = true, order = "degree", dropout = 0.0, seed = 0))]
    fn new(
        node_features: &Bound<'_, PyAny>,
        task: PyGraphTask,
        edge_features: Option<&Bound<'_, PyAny>>,
        pe_dim: usize,
        pe_sign_flip: Option<(usize, usize)>,
        d_model: usize,
        max_hops: usize,
        walks: usize,
        repeats: usize,
        token_sampling: &str,
        local: &str,
        token_layers: Option<usize>,
        token_tail: &str,
        node_layers: usize,
        mpnn: Option<&str>,
        d_state: usize,
        token_heads: usize,
        node_heads: usize,
        node_sequences: usize,
        bidirectional: bool,
        order: &str,
        dropout: f32,
        seed: u64,
    ) -> PyResult<Self> {
        let unknown = |name: &str, value: &str, options: &str| {
            PyValueError::new_err(format!("{name} must be {options}, got {value:?}"))
        };
        let mut inner = GraphMambaSpec::new(feature_spec(node_features, "node_features")?, task.inner)
            .with_edge_features(
                edge_features
                    .map(|value| feature_spec(value, "edge_features"))
                    .transpose()?,
            )
            .with_pe_dim(pe_dim)
            .with_pe_sign_flip(pe_sign_flip)
            .with_d_model(d_model)
            .with_tokens(max_hops, walks, repeats)
            .with_token_sampling(match token_sampling {
                "step" => TokenSampling::PerStep,
                "epoch" => TokenSampling::PerEpoch,
                "static" => TokenSampling::Static,
                other => return Err(unknown("token_sampling", other, "'step', 'epoch' or 'static'")),
            })
            .with_local(match local {
                "mean" => LocalEncoder::Mean,
                "sgc" => LocalEncoder::Sgc { hops: 1 },
                other => return Err(unknown("local", other, "'mean' or 'sgc'")),
            })
            .with_token_tail(match token_tail {
                "forward" => TokenTail::Forward,
                "bidirectional" => TokenTail::Bidirectional,
                other => return Err(unknown("token_tail", other, "'forward' or 'bidirectional'")),
            })
            .with_node_layers(node_layers)
            .with_mpnn(match mpnn {
                None | Some("none") => None,
                Some("gine") => Some(MpnnKind::Gine),
                Some("gated_gcn") => Some(MpnnKind::GatedGcn),
                Some(other) => return Err(unknown("mpnn", other, "'gine', 'gated_gcn' or None")),
            })
            .with_d_state(d_state)
            .with_token_heads(token_heads)
            .with_node_heads(node_heads)
            .with_node_sequences(node_sequences)
            .with_direction(if bidirectional {
                ScanDirection::Bidirectional
            } else {
                ScanDirection::Forward
            })
            .with_order(match order {
                "degree" => NodeOrder::Degree { descending: false },
                "degree_desc" => NodeOrder::Degree { descending: true },
                "ppr" => NodeOrder::ppr(),
                "kcore" => NodeOrder::KCore,
                "given" => NodeOrder::Given,
                other => {
                    return Err(unknown(
                        "order",
                        other,
                        "'degree', 'degree_desc', 'ppr', 'kcore' or 'given'",
                    ));
                }
            })
            .with_dropout(dropout)
            .with_seed(seed);
        if let Some(layers) = token_layers {
            inner = inner.with_token_layers(layers);
        }
        inner.validate().py()?;
        Ok(Self { inner })
    }

    /// The spec as JSON.
    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json().py()
    }

    /// Rebuild from `to_json`.
    #[staticmethod]
    fn from_json(json: &str) -> PyResult<Self> {
        Ok(Self {
            inner: GraphMambaSpec::from_json(json).py()?,
        })
    }

    /// Model width.
    #[getter]
    fn d_model(&self) -> usize {
        self.inner.d_model
    }

    /// Tokens per node, `max_hops · repeats + 1`.
    #[getter]
    fn tokens_per_node(&self) -> usize {
        self.inner.tokens.len()
    }

    /// Scalars a model built from this spec holds.
    #[getter]
    fn num_parameters(&self) -> usize {
        self.inner.parameter_count()
    }

    /// The task.
    #[getter]
    fn task(&self) -> PyGraphTask {
        PyGraphTask {
            inner: self.inner.task,
        }
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        format!(
            "GraphMambaSpec(d_model={}, max_hops={}, walks={}, repeats={}, token_layers={}, \
             node_layers={}, mpnn={:?}, task={:?})",
            self.inner.d_model,
            self.inner.tokens.max_hops,
            self.inner.tokens.walks,
            self.inner.tokens.repeats,
            self.inner.token_layers,
            self.inner.node_layers,
            self.inner.mpnn,
            self.inner.task,
        )
    }
}

// ---------------------------------------------------------------------------
// Element types
// ---------------------------------------------------------------------------

fn parse_dtype(dtype: &str) -> PyResult<DType> {
    match dtype.to_lowercase().as_str() {
        "f32" | "float32" => Ok(DType::F32),
        "f16" | "float16" => Ok(DType::F16),
        "bf16" | "bfloat16" => Ok(DType::BF16),
        other => Err(PyValueError::new_err(format!(
            "dtype must be 'f32', 'f16' or 'bf16', got {other:?}"
        ))),
    }
}

/// Refuse an element type the compiled backend cannot store or compute, and
/// the one the optimizer cannot train.
fn check_dtype(device: &Device<R>, dtype: DType) -> PyResult<()> {
    // The mixer's backward pass returns NaN gradients in f16 (bf16 is fine),
    // with or without the graph model around it (`tests/graph_dtype.rs`).
    // Lift this when `BF16_ACTIVATIONS_PLAN.md` has landed and
    // `f16_tracks_f32_for_50_steps` passes.
    if matches!(dtype, DType::F16) {
        return Err(PyNotImplementedError::new_err(
            "dtype=\"f16\" is not available for the graph model: the Mamba-3 mixer's backward \
             pass is not finite in f16 in this build (BF16_ACTIVATIONS_PLAN.md is the work \
             that makes it trainable). Use \"bf16\" or \"f32\".",
        ));
    }
    mamba3::backend::ensure_dtype(device, dtype)
        .map_err(|err| PyNotImplementedError::new_err(err.to_string()))
}

/// One variant per element type the extension is compiled for.
macro_rules! per_dtype {
    ($name:ident, $payload:ident) => {
        enum $name {
            F32($payload<f32>),
            F16($payload<half::f16>),
            Bf16($payload<half::bf16>),
        }

        impl $name {
            fn dtype(&self) -> DType {
                match self {
                    $name::F32(_) => DType::F32,
                    $name::F16(_) => DType::F16,
                    $name::Bf16(_) => DType::BF16,
                }
            }
        }
    };
}

type Dataset<E> = GraphDataset<R, E>;

/// A model with its optimizer.
struct State<E: FloatElem> {
    model: GraphMamba<R, E>,
    trainer: GraphTrainer<R, E>,
}

per_dtype!(AnyDataset, Dataset);
per_dtype!(AnyState, State);

/// Run `$body` with `$d` bound to the dataset, whatever its element type.
macro_rules! each_dataset {
    ($value:expr, $d:ident => $body:expr) => {
        match $value {
            AnyDataset::F32($d) => $body,
            AnyDataset::F16($d) => $body,
            AnyDataset::Bf16($d) => $body,
        }
    };
}

/// Run `$body` with `$s` bound to the model state, whatever its element type.
macro_rules! each_state {
    ($value:expr, $s:ident => $body:expr) => {
        match $value {
            AnyState::F32($s) => $body,
            AnyState::F16($s) => $body,
            AnyState::Bf16($s) => $body,
        }
    };
}

/// Run `$body` with the model state and a dataset of the same element type;
/// a `ValueError` otherwise.
macro_rules! paired {
    ($state:expr, $data:expr, ($s:ident, $d:ident) => $body:expr) => {
        match ($state, $data) {
            (AnyState::F32($s), AnyDataset::F32($d)) => $body,
            (AnyState::F16($s), AnyDataset::F16($d)) => $body,
            (AnyState::Bf16($s), AnyDataset::Bf16($d)) => $body,
            (state, data) => Err(PyValueError::new_err(format!(
                "the model is {} and the dataset is {}: build both with the same dtype",
                state.dtype().name(),
                data.dtype().name()
            ))),
        }
    };
}

// ---------------------------------------------------------------------------
// Dataset
// ---------------------------------------------------------------------------

/// A graph dataset on the device: validated against its spec, put in canonical
/// order and uploaded once.
#[pyclass(module = "mamba3_graph", name = "GraphDataset", unsendable)]
pub struct PyGraphDataset {
    inner: AnyDataset,
    /// The train, validation and test masks as they were given, in the
    /// caller's order: what `predict(split=...)` selects by.
    masks: [Option<Vec<bool>>; 3],
}

/// Canonicalise with the interpreter lock held (NumPy memory is read), upload
/// with it released.
fn build_dataset<E: FloatElem>(
    py: Python<'_>,
    spec: &GraphMambaSpec,
    ingest: &Ingest<'_>,
    options: DatasetOptions,
) -> PyResult<Dataset<E>> {
    let prepared = PreparedDataset::<E>::new(spec, &ingest.view(), options)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    let device = device();
    detached(py, move || prepared.upload(&device)).py()
}

#[pymethods]
impl PyGraphDataset {
    /// Validate `arrays` against `spec` and upload them once.
    ///
    /// `arrays` is a dict: `edge_index` `int[2, E]`, `x` `float[N, F]` (or
    /// `int[N, fields]` for categorical features), and optionally `edge_attr`,
    /// `y`, `graph_ptr` `int[G + 1]`, `train_mask` / `val_mask` / `test_mask`
    /// and `pe` `float[N, pe_dim]`. Floats may be `float16`, `float32` or
    /// `float64` and integers any width; arrays need not be C-contiguous.
    /// Unknown keys, wrong shapes and out-of-range ids raise `ValueError`
    /// naming the key. Nothing is read back from the device.
    #[new]
    #[pyo3(signature = (spec, arrays, *, symmetrize = true, dtype = "f32"))]
    fn new(
        py: Python<'_>,
        spec: &PyGraphMambaSpec,
        arrays: &Bound<'_, PyDict>,
        symmetrize: bool,
        dtype: &str,
    ) -> PyResult<Self> {
        let dtype = parse_dtype(dtype)?;
        check_dtype(&device(), dtype)?;
        let ingest = Ingest::new(&spec.inner, arrays)?;
        let options = DatasetOptions { symmetrize };
        let inner = match dtype {
            DType::F32 => AnyDataset::F32(build_dataset(py, &spec.inner, &ingest, options)?),
            DType::F16 => AnyDataset::F16(build_dataset(py, &spec.inner, &ingest, options)?),
            DType::BF16 => AnyDataset::Bf16(build_dataset(py, &spec.inner, &ingest, options)?),
        };
        let masks = ingest.masks.each_ref().map(|slot| {
            slot.as_ref().map(|held| match held.view() {
                Bools::Bool(flags) => flags.to_vec(),
                Bools::U8(flags) => flags.iter().map(|&flag| flag != 0).collect(),
            })
        });
        Ok(Self { inner, masks })
    }

    /// Nodes over all graphs.
    #[getter]
    fn num_nodes(&self) -> usize {
        each_dataset!(&self.inner, d => d.num_nodes())
    }

    /// Number of graphs.
    #[getter]
    fn num_graphs(&self) -> usize {
        each_dataset!(&self.inner, d => d.num_graphs())
    }

    /// Directed edges after symmetrising and removing duplicates.
    #[getter]
    fn num_edges(&self) -> usize {
        each_dataset!(&self.inner, d => d.store().n_edges())
    }

    /// Bytes the dataset's tables take on the device.
    #[getter]
    fn nbytes(&self) -> usize {
        each_dataset!(&self.inner, d => d.store().bytes())
    }

    /// The element type the tables are stored in.
    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.dtype().name()
    }

    fn __repr__(&self) -> String {
        format!(
            "GraphDataset(nodes={}, graphs={}, edges={}, dtype={:?})",
            self.num_nodes(),
            self.num_graphs(),
            self.num_edges(),
            self.dtype()
        )
    }
}

// ---------------------------------------------------------------------------
// Encodings
// ---------------------------------------------------------------------------

/// The symmetrised CSR of `edge_index`, built with the interpreter lock held.
fn encoding_csr(
    edge_index: &Bound<'_, PyAny>,
    num_nodes: usize,
    graph_ptr: Option<&Bound<'_, PyAny>>,
) -> PyResult<(HostCsr, Vec<usize>)> {
    let (shape, edges) = hold_ints(edge_index, "edge_index")?;
    if shape.len() != 2 || shape[0] != 2 {
        return Err(shape_error("edge_index", "[2, edges] integers", &shape));
    }
    let ptr = graph_ptr
        .map(|value| hold_ints(value, "graph_ptr"))
        .transpose()?;
    if let Some((ptr_shape, _)) = &ptr {
        if ptr_shape.len() != 1 {
            return Err(shape_error("graph_ptr", "[graphs + 1] integers", ptr_shape));
        }
    }
    let (edge_src, edge_dst) = edges.halves(shape[1]);
    let view = GraphDataView {
        n_nodes: num_nodes,
        edge_src,
        edge_dst,
        x: FeaturesView::Float {
            dim: 0,
            data: Floats::F32(&[]),
        },
        edge_attr: None,
        pe: None,
        y: LabelsView::None,
        graph_ptr: ptr.as_ref().map(|(_, held)| held.view()),
        masks: None,
    };
    let csr = HostCsr::from_view(&view, true).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let offsets = graph_offsets_of(&view, &csr);
    Ok((csr, offsets))
}

/// Random-walk structural encoding: `float32[num_nodes, k]`, column `j - 1`
/// the probability that a `j`-step random walk returns to its start.
#[pyfunction]
#[pyo3(signature = (edge_index, num_nodes, k, *, max_ball = RWSE_MAX_BALL))]
fn rwse<'py>(
    py: Python<'py>,
    edge_index: &Bound<'py, PyAny>,
    num_nodes: usize,
    k: usize,
    max_ball: usize,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let (csr, _) = encoding_csr(edge_index, num_nodes, None)?;
    let values = detached(py, move || rwse_csr(&csr, k, max_ball)).py()?;
    PyArray1::from_vec(py, values).reshape((num_nodes, k))
}

/// Laplacian positional encoding: `float32[num_nodes, k]`, the `k`
/// eigenvectors of each graph's normalised Laplacian with the smallest
/// non-zero eigenvalues (zero columns where a graph has fewer).
#[pyfunction]
#[pyo3(signature = (edge_index, num_nodes, k, *, graph_ptr = None,
                    max_nodes = LAPLACIAN_MAX_NODES))]
fn laplacian_pe<'py>(
    py: Python<'py>,
    edge_index: &Bound<'py, PyAny>,
    num_nodes: usize,
    k: usize,
    graph_ptr: Option<&Bound<'py, PyAny>>,
    max_nodes: usize,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let (csr, offsets) = encoding_csr(edge_index, num_nodes, graph_ptr)?;
    let values = detached(py, move || {
        laplacian_pe_csr(&csr, &offsets, k, max_nodes).map(|pe| pe.vectors)
    })
    .py()?;
    PyArray1::from_vec(py, values).reshape((num_nodes, k))
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

/// Whether this extension was compiled without optimisation.
const DEBUG_BUILD: bool = cfg!(debug_assertions);

/// Warn that the extension is a debug build, which trains many times slower.
#[pyfunction(name = "_warn_if_debug_build")]
#[pyo3(signature = (debug = None))]
fn warn_if_debug_build(py: Python<'_>, debug: Option<bool>) -> PyResult<()> {
    if debug.unwrap_or(DEBUG_BUILD) {
        PyErr::warn(
            py,
            &py.get_type::<PyRuntimeWarning>(),
            c"mamba3 was built without optimisation: training will be many times slower. \
              Rebuild with `maturin develop --release`.",
            2,
        )?;
    }
    Ok(())
}

/// How this extension was built: `{"profile", "backend", "version"}`.
#[pyfunction]
fn build_info(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let info = PyDict::new(py);
    info.set_item("profile", if DEBUG_BUILD { "debug" } else { "release" })?;
    info.set_item("backend", device().name())?;
    info.set_item("version", env!("CARGO_PKG_VERSION"))?;
    Ok(info)
}

fn parse_split(split: &str) -> PyResult<Split> {
    match split {
        "train" => Ok(Split::Train),
        "val" => Ok(Split::Val),
        "test" => Ok(Split::Test),
        other => Err(PyValueError::new_err(format!(
            "split must be 'train', 'val' or 'test', got {other:?}"
        ))),
    }
}

fn init_state<E: FloatElem>(
    spec: &GraphMambaSpec,
    config: &GraphTrainConfig,
) -> mamba3::error::Result<State<E>> {
    Ok(State {
        model: GraphMamba::init(spec, &device())?,
        trainer: GraphTrainer::new(config)?,
    })
}

/// What stopped an epoch early.
enum Stopped {
    /// The device or the model refused.
    Failed(mamba3::error::Error),
    /// A signal handler raised (Ctrl-C).
    Interrupted(PyErr),
}

/// Queue every step of one epoch. Runs without the interpreter lock; signals
/// are checked between steps, at most every 50 ms.
fn train_epoch_of<E: FloatElem>(
    state: &mut State<E>,
    dataset: &Dataset<E>,
    epoch: u64,
    batch_rows: Option<usize>,
    parts: Option<usize>,
) -> Result<usize, Stopped> {
    // Explicit arguments choose the mode; otherwise a graph-level task or a
    // dataset of several graphs trains on whole graphs, one large graph on
    // node partitions.
    let whole_graphs = match (batch_rows, parts) {
        (Some(_), Some(_)) => {
            return Err(Stopped::Failed(mamba3::error::Error::config(
                "batch_rows cuts whole-graph batches and parts cuts node partitions: pass one",
            )));
        }
        (Some(_), None) => true,
        (None, Some(_)) => false,
        // A model that cannot run on node partitions at all (GatedGcn, edge
        // features) trains on its one graph whole.
        (None, None) => {
            let spec = state.model.spec();
            spec.task.per_graph()
                || dataset.num_graphs() > 1
                || spec.validate_for(BatchMode::NodeSubset).is_err()
        }
    };
    let plan = if whole_graphs {
        dataset.epoch_graphs(batch_rows, epoch, Split::Train)
    } else {
        dataset.epoch_nodes(parts, epoch, Split::Train)
    }
    .map_err(Stopped::Failed)?;

    let mut last_check = Instant::now();
    let mut interrupt = None;
    let State { model, trainer } = state;
    let outcome = model
        .train_epoch_with(trainer, &plan, |_| {
            if last_check.elapsed() < Duration::from_millis(50) {
                return ControlFlow::Continue(());
            }
            last_check = Instant::now();
            // A pending signal runs its Python handler, so the thread must be
            // attached; off the main thread this does nothing.
            match Python::attach(|py| py.check_signals()) {
                Ok(()) => ControlFlow::Continue(()),
                Err(err) => {
                    interrupt = Some(err);
                    ControlFlow::Break(())
                }
            }
        });
    // The interrupt first: it is what the caller asked for, and a device error
    // behind it will be raised again by the next call.
    match (interrupt, outcome) {
        (Some(err), _) => Err(Stopped::Interrupted(err)),
        (None, Ok(steps)) => Ok(steps),
        (None, Err(err)) => Err(Stopped::Failed(err)),
    }
}

/// Graph Mamba with its optimizer: training, evaluation and prediction, each
/// one call into the device.
#[pyclass(module = "mamba3_graph", name = "GraphMamba", unsendable)]
pub struct PyGraphMamba {
    inner: AnyState,
    spec: GraphMambaSpec,
}

impl PyGraphMamba {
    fn build(
        py: Python<'_>,
        spec: &GraphMambaSpec,
        config: &GraphTrainConfig,
        dtype: DType,
    ) -> PyResult<AnyState> {
        check_dtype(&device(), dtype)?;
        warn_if_debug_build(py, None)?;
        detached(py, || {
            Ok(match dtype {
                DType::F32 => AnyState::F32(init_state(spec, config)?),
                DType::F16 => AnyState::F16(init_state(spec, config)?),
                DType::BF16 => AnyState::Bf16(init_state(spec, config)?),
            })
        })
        .map_err(|err: mamba3::error::Error| PyValueError::new_err(err.to_string()))
    }
}

/// The optimizer settings of the constructor and of `load`.
fn train_config(
    learning_rate: f32,
    weight_decay: f32,
    max_grad_norm: f32,
    lr_schedule: Option<PyLrSchedule>,
    loss_scale: Option<f32>,
) -> GraphTrainConfig {
    GraphTrainConfig {
        learning_rate,
        weight_decay,
        max_grad_norm,
        schedule: lr_schedule.map(|s| s.inner).unwrap_or_default(),
        loss_scale: loss_scale.unwrap_or(1.0),
    }
}

#[pymethods]
impl PyGraphMamba {
    /// Build the model and its AdamW optimizer.
    ///
    /// `loss_scale` multiplies the loss before it is differentiated, which
    /// keeps a 16-bit model's gradients above underflow; reported losses and
    /// gradient norms are divided back. A debug build raises a
    /// `RuntimeWarning`.
    #[new]
    #[pyo3(signature = (spec, *, learning_rate = 1e-3, weight_decay = 0.0, max_grad_norm = 1.0,
                        lr_schedule = None, dtype = "f32", loss_scale = None))]
    fn new(
        py: Python<'_>,
        spec: &PyGraphMambaSpec,
        learning_rate: f32,
        weight_decay: f32,
        max_grad_norm: f32,
        lr_schedule: Option<PyLrSchedule>,
        dtype: &str,
        loss_scale: Option<f32>,
    ) -> PyResult<Self> {
        let config = train_config(
            learning_rate,
            weight_decay,
            max_grad_norm,
            lr_schedule,
            loss_scale,
        );
        Ok(Self {
            inner: Self::build(py, &spec.inner, &config, parse_dtype(dtype)?)?,
            spec: spec.inner.clone(),
        })
    }

    /// Queue every training step of one epoch and return how many were queued.
    ///
    /// Nothing is read back and only the epoch's own table is uploaded.
    /// `batch_rows` is the row budget of whole-graph batches, `parts` the
    /// number of node partitions of one large graph (1 is full batch); with
    /// neither, a graph-level task or a dataset of several graphs trains on
    /// whole graphs and one large graph on node partitions, both sized
    /// automatically. The interpreter lock is released for the whole call, and
    /// Ctrl-C stops it after a completed step, leaving the model usable.
    #[pyo3(signature = (data, epoch, *, batch_rows = None, parts = None))]
    fn train_epoch(
        &mut self,
        py: Python<'_>,
        data: &PyGraphDataset,
        epoch: u64,
        batch_rows: Option<usize>,
        parts: Option<usize>,
    ) -> PyResult<usize> {
        let outcome = paired!(&mut self.inner, &data.inner, (state, dataset) => Ok(detached(
            py,
            || train_epoch_of(state, dataset, epoch, batch_rows, parts),
        )))?;
        match outcome {
            Ok(steps) => Ok(steps),
            Err(Stopped::Interrupted(err)) => Err(err),
            Err(Stopped::Failed(err)) => Err(crate::err::to_py(err)),
        }
    }

    /// The report of every step queued since the last call, oldest first:
    /// dicts of `step`, `loss`, `grad_norm` and `learning_rate`. One device
    /// read for all of them. A non-finite loss raises `FloatingPointError`.
    fn read_losses<'py>(&mut self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let infos = each_state!(&mut self.inner, state => {
            detached(py, || state.trainer.read_losses())
        })
        .py()?;
        let mut out = Vec::with_capacity(infos.len());
        for info in infos {
            if !info.loss.is_finite() || !info.grad_norm.is_finite() {
                return Err(PyFloatingPointError::new_err(format!(
                    "step {} has a non-finite loss ({}) or gradient norm ({}); lower the \
                     learning rate or, for a 16-bit dtype, the loss scale",
                    info.step, info.loss, info.grad_norm
                )));
            }
            let entry = PyDict::new(py);
            entry.set_item("step", info.step)?;
            entry.set_item("loss", info.loss)?;
            entry.set_item("grad_norm", info.grad_norm)?;
            entry.set_item("learning_rate", info.learning_rate)?;
            out.push(entry);
        }
        Ok(out)
    }

    /// A metric over a split: `{metric: value, "count": targets counted}`.
    ///
    /// `metric` is `"accuracy"` or `"f1_macro"` (classification), `"mae"` or
    /// `"mse"` (regression), `"ap"` or `"roc_auc"` (multi-label and two-class
    /// tasks). The model runs over every node or graph; the split only selects
    /// what is counted. One device read.
    #[pyo3(signature = (data, *, split = "val", metric = "accuracy", batch_rows = None,
                        parts = None))]
    fn evaluate<'py>(
        &self,
        py: Python<'py>,
        data: &PyGraphDataset,
        split: &str,
        metric: &str,
        batch_rows: Option<usize>,
        parts: Option<usize>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let split = parse_split(split)?;
        let metric = Metric::parse(metric).py()?;
        let options = EvalOptions { batch_rows, parts };
        let result = paired!(&self.inner, &data.inner, (state, dataset) => Ok(detached(
            py,
            || state.model.evaluate(dataset, split, metric, &options),
        )))?
        .py()?;
        let out = PyDict::new(py);
        out.set_item(metric.name(), result.value)?;
        out.set_item("count", result.count)?;
        Ok(out)
    }

    /// The model's output for every node (node tasks) or graph (graph tasks),
    /// `float32[items, outputs]`, in the order the dataset was given in. One
    /// device read.
    ///
    /// `split` (`"train"`, `"val"` or `"test"`) returns only the rows of the
    /// items in that mask of the dataset, still in the given order. The model
    /// runs over every node either way: a node's output depends on its
    /// neighbours, whatever split they are in.
    #[pyo3(signature = (data, *, split = None, batch_rows = None, parts = None))]
    fn predict<'py>(
        &self,
        py: Python<'py>,
        data: &PyGraphDataset,
        split: Option<&str>,
        batch_rows: Option<usize>,
        parts: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let keep = match split {
            None => None,
            Some(name) => {
                let split = parse_split(name)?;
                if data.masks.iter().all(Option::is_none) {
                    return Err(PyValueError::new_err(format!(
                        "split={name:?}: the dataset was built without train_mask, val_mask \
                         or test_mask"
                    )));
                }
                // A mask that was not given, beside ones that were, is empty.
                Some(data.masks[split.index()].as_deref().unwrap_or(&[]))
            }
        };
        let options = EvalOptions { batch_rows, parts };
        let values = paired!(&self.inner, &data.inner, (state, dataset) => Ok(detached(
            py,
            || state.model.predict(dataset, &options),
        )))?
        .py()?;
        let outputs = self.spec.task.outputs();
        let Some(keep) = keep else {
            // The vector becomes the array's buffer: no further copy.
            return PyArray1::from_vec(py, values)
                .reshape((values_len_rows(&self.spec, data), outputs));
        };
        let selected = keep.iter().filter(|&&flag| flag).count();
        let mut rows = Vec::with_capacity(selected * outputs);
        for (item, _) in keep.iter().enumerate().filter(|(_, flag)| **flag) {
            rows.extend_from_slice(&values[item * outputs..(item + 1) * outputs]);
        }
        PyArray1::from_vec(py, rows).reshape((selected, outputs))
    }

    /// What a step is expected to hold on the device, in bytes: `store`,
    /// `parameters` (with the optimizer's state), `live` (alive when the
    /// backward pass starts), `largest_allocation`, and the `rows` per batch
    /// and allocation `threshold` the estimate is for.
    #[pyo3(signature = (data, *, batch_rows = None, parts = None))]
    fn memory_estimate<'py>(
        &self,
        py: Python<'py>,
        data: &PyGraphDataset,
        batch_rows: Option<usize>,
        parts: Option<usize>,
    ) -> PyResult<Bound<'py, PyDict>> {
        if batch_rows.is_some() && parts.is_some() {
            return Err(PyValueError::new_err(
                "batch_rows cuts whole-graph batches and parts cuts node partitions: pass one",
            ));
        }
        let estimate = paired!(&self.inner, &data.inner, (_state, dataset) => {
            // The estimate is computed from the dataset's spec; it is this
            // model's only when the two were built from the same one.
            if dataset.spec() != &self.spec {
                return Err(PyValueError::new_err(
                    "the dataset was built from a different spec than this model: the \
                     estimate would be for the dataset's",
                ));
            }
            Ok(dataset.memory_estimate(batch_rows, parts))
        })?;
        let out = PyDict::new(py);
        out.set_item("store", estimate.store)?;
        out.set_item("parameters", estimate.parameters)?;
        out.set_item("live", estimate.live)?;
        out.set_item("largest_allocation", estimate.largest_allocation)?;
        out.set_item("rows", estimate.rows)?;
        out.set_item("threshold", estimate.threshold)?;
        Ok(out)
    }

    /// Save the weights and the spec. `step` defaults to the optimizer's.
    #[pyo3(signature = (path, step = None))]
    fn save(&self, py: Python<'_>, path: &str, step: Option<u64>) -> PyResult<()> {
        // Saving reads every weight back from the device: wait without the lock.
        let path = path.to_owned();
        each_state!(&self.inner, state => {
            let step = step.unwrap_or_else(|| state.trainer.step_count());
            detached(py, || state.model.save(&path, step))
        })
        .py()
    }

    /// Rebuild a model from `save`'s file, with a fresh optimizer.
    #[staticmethod]
    #[pyo3(signature = (path, *, dtype = "f32", learning_rate = 1e-3, weight_decay = 0.0,
                        max_grad_norm = 1.0, lr_schedule = None, loss_scale = None))]
    fn load(
        py: Python<'_>,
        path: &str,
        dtype: &str,
        learning_rate: f32,
        weight_decay: f32,
        max_grad_norm: f32,
        lr_schedule: Option<PyLrSchedule>,
        loss_scale: Option<f32>,
    ) -> PyResult<Self> {
        let dtype = parse_dtype(dtype)?;
        let device = device();
        check_dtype(&device, dtype)?;
        warn_if_debug_build(py, None)?;
        let config = train_config(
            learning_rate,
            weight_decay,
            max_grad_norm,
            lr_schedule,
            loss_scale,
        );
        fn restore<E: FloatElem>(
            path: &str,
            config: &GraphTrainConfig,
            device: &Device<R>,
        ) -> mamba3::error::Result<(State<E>, GraphMambaSpec)> {
            let model = GraphMamba::<R, E>::load(path, device)?;
            let spec = model.spec().clone();
            let mut trainer = GraphTrainer::new(config)?;
            trainer.set_step_count(mamba3::train::Checkpoint::load(path)?.step);
            Ok((State { model, trainer }, spec))
        }
        let path = path.to_owned();
        let (inner, spec) = detached(py, || match dtype {
            DType::F32 => restore(&path, &config, &device).map(|(s, spec)| (AnyState::F32(s), spec)),
            DType::F16 => restore(&path, &config, &device).map(|(s, spec)| (AnyState::F16(s), spec)),
            DType::BF16 => {
                restore(&path, &config, &device).map(|(s, spec)| (AnyState::Bf16(s), spec))
            }
        })
        .py()?;
        Ok(Self { inner, spec })
    }

    /// Scalars in the model.
    #[getter]
    fn num_parameters(&self) -> usize {
        each_state!(&self.inner, state => state.model.num_parameters())
    }

    /// Optimizer steps taken so far.
    #[getter]
    fn step(&self) -> u64 {
        each_state!(&self.inner, state => state.trainer.step_count())
    }

    /// The spec the model was built from.
    #[getter]
    fn spec(&self) -> PyGraphMambaSpec {
        PyGraphMambaSpec {
            inner: self.spec.clone(),
        }
    }

    /// The element type of the weights.
    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.dtype().name()
    }

    fn __repr__(&self) -> String {
        format!(
            "GraphMamba(parameters={}, step={}, dtype={:?})",
            self.num_parameters(),
            self.step(),
            self.dtype()
        )
    }
}

/// Rows of a prediction: one per graph for a graph task, one per node
/// otherwise.
fn values_len_rows(spec: &GraphMambaSpec, data: &PyGraphDataset) -> usize {
    if spec.task.per_graph() {
        data.num_graphs()
    } else {
        data.num_nodes()
    }
}

/// Add the graph classes and functions to the extension module.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyCategorical>()?;
    module.add_class::<PyGraphTask>()?;
    module.add_class::<PyGraphMambaSpec>()?;
    module.add_class::<PyGraphDataset>()?;
    module.add_class::<PyGraphMamba>()?;
    module.add_function(wrap_pyfunction!(node_classification, module)?)?;
    module.add_function(wrap_pyfunction!(graph_classification, module)?)?;
    module.add_function(wrap_pyfunction!(graph_regression, module)?)?;
    module.add_function(wrap_pyfunction!(graph_multi_label, module)?)?;
    module.add_function(wrap_pyfunction!(rwse, module)?)?;
    module.add_function(wrap_pyfunction!(laplacian_pe, module)?)?;
    module.add_function(wrap_pyfunction!(build_info, module)?)?;
    module.add_function(wrap_pyfunction!(warn_if_debug_build, module)?)?;
    Ok(())
}
