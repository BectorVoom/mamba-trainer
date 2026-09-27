//! The generic entity model (ENTITY_MODEL_PLAN.md P1).
//!
//! [`PyEntityModelSpec`] is built from [`PyContextSet`]/[`PyQuerySet`]/
//! [`PyHead`] pieces, round-trips through JSON, and names no domain.
//! [`PyEntityDataset`] validates a dict of NumPy arrays against the spec
//! (unknown keys, shapes and id ranges name the key) and uploads it once.
//! [`PyEntityModel`] trains from sample ids (`queue_train_step`), reports
//! per-head losses (`read_losses`), scores (`evaluate`) and decodes
//! (`predict`, greedy or teacher-forced, with an optional Python chooser).

use std::collections::BTreeMap;
use std::rc::Rc;

use mamba3::models::entity::{
    ContextSetSpec, EntityBatch, EntityDataset, EntityModel, EntityModelSpec,
    EntityTask, HeadSpec, HostArrays, LossComponents, QuerySetSpec, SetLayout, StepSelection,
};
use mamba3::tensor::Tensor;
use mamba3::tensor::ops::index::read_all;
use mamba3::train::{AdamW, AdamWConfig, QueuedStep, Trainer, TrainerConfig};
use numpy::{AllowTypeChange, PyArray1, PyArrayLikeDyn, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::{PyFloatingPointError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::config::PyLrSchedule;
use crate::err::IntoPyResult;
use crate::{E, R};

/// Any float array, whatever its rank or dtype.
type FloatArrayDyn<'py> = PyArrayLikeDyn<'py, f32, AllowTypeChange>;
/// Any integer array, whatever its rank or dtype.
type IntArrayDyn<'py> = PyArrayLikeDyn<'py, i64, AllowTypeChange>;

/// `(shape, values)` of a float array, with a message naming `what` if it is
/// not one. float16 arrays are converted to float32.
fn read_floats(value: &Bound<'_, PyAny>, what: &str) -> PyResult<(Vec<usize>, Vec<f32>)> {
    if let Ok(array) = value.extract::<FloatArrayDyn<'_>>() {
        let shape = array.shape().to_vec();
        let data = match array.as_slice() {
            Ok(slice) => slice.to_vec(),
            Err(_) => array.as_array().iter().copied().collect(),
        };
        return Ok((shape, data));
    }
    // float16 fallback.
    if let Ok(array) = value.cast::<numpy::PyArrayDyn<half::f16>>() {
        let shape = array.shape().to_vec();
        let data = array
            .readonly()
            .as_slice()
            .map(|s| s.iter().map(|x| x.to_f32()).collect())
            .unwrap_or_else(|_| {
                array
                    .readonly()
                    .as_array()
                    .iter()
                    .map(|x| x.to_f32())
                    .collect()
            });
        return Ok((shape, data));
    }
    Err(PyValueError::new_err(format!(
        "{what} must be an array of floats"
    )))
}

/// `(shape, values)` of an integer or boolean array, with a message naming
/// `what` if it is not one.
fn read_ints(value: &Bound<'_, PyAny>, what: &str) -> PyResult<(Vec<usize>, Vec<i64>)> {
    if let Ok(array) = value.extract::<IntArrayDyn<'_>>() {
        let shape = array.shape().to_vec();
        let data = match array.as_slice() {
            Ok(slice) => slice.to_vec(),
            Err(_) => array.as_array().iter().copied().collect(),
        };
        return Ok((shape, data));
    }
    if let Ok(array) = value.extract::<PyArrayLikeDyn<'_, bool, AllowTypeChange>>() {
        let shape = array.shape().to_vec();
        let data = array.as_array().iter().map(|&x| i64::from(x)).collect();
        return Ok((shape, data));
    }
    Err(PyValueError::new_err(format!(
        "{what} must be an array of integers or booleans"
    )))
}

/// A row-major grid layout for a context set.
#[pyclass(module = "mamba3_rl", name = "Grid", from_py_object)]
#[derive(Clone)]
pub struct PyGrid {
    inner: (usize, usize, bool),
}

#[pymethods]
impl PyGrid {
    /// A `height × width` grid; `alternate_axes` (default true) scans
    /// column-major on every second encoder layer.
    #[new]
    #[pyo3(signature = (height, width, alternate_axes = true))]
    fn new(height: usize, width: usize, alternate_axes: bool) -> Self {
        Self { inner: (height, width, alternate_axes) }
    }
}

/// One named context set: up to `count` entities with `features` floats each.
#[pyclass(module = "mamba3_rl", name = "ContextSet", from_py_object)]
#[derive(Clone)]
pub struct PyContextSet {
    pub(crate) inner: ContextSetSpec,
}

#[pymethods]
impl PyContextSet {
    /// A context set; `layout` is a [`Grid`] or omitted (sequence order).
    #[new]
    #[pyo3(signature = (name, *, count, features, layout = None, position_embedding = true))]
    fn new(
        name: String,
        count: usize,
        features: usize,
        layout: Option<PyGrid>,
        position_embedding: bool,
    ) -> PyResult<Self> {
        let mut inner = ContextSetSpec::new(name, count, features);
        if let Some(g) = layout {
            inner = inner.with_layout(SetLayout::Grid {
                height: g.inner.0,
                width: g.inner.1,
                alternate_axes: g.inner.2,
            });
        }
        inner = inner.with_position_embedding(position_embedding);
        Ok(Self { inner })
    }
}

/// The query set: up to `count` queries with a plan length `steps`.
#[pyclass(module = "mamba3_rl", name = "QuerySet", from_py_object)]
#[derive(Clone)]
pub struct PyQuerySet {
    pub(crate) inner: QuerySetSpec,
}

#[pymethods]
impl PyQuerySet {
    /// A query set; `anchor` names a context set, `autoregressive_on` names
    /// the pointer head whose previous choices condition later steps.
    #[new]
    #[pyo3(signature = (name, *, count, features, anchor = None, steps = 1, autoregressive_on = None, lags = None))]
    fn new(
        name: String,
        count: usize,
        features: usize,
        anchor: Option<String>,
        steps: usize,
        autoregressive_on: Option<String>,
        lags: Option<usize>,
    ) -> PyResult<Self> {
        let mut inner = QuerySetSpec::new(name, count, features, steps);
        if let Some(a) = anchor {
            inner = inner.with_anchor(a);
        }
        if let Some(p) = autoregressive_on {
            inner = inner.with_autoregressive(p);
        }
        if let Some(l) = lags {
            inner = inner.with_lags(l);
        }
        Ok(Self { inner })
    }
}

/// One prediction head; build with the static constructors.
#[pyclass(module = "mamba3_rl", name = "Head", from_py_object)]
#[derive(Clone)]
pub struct PyHead {
    pub(crate) inner: HeadSpec,
}

fn parse_steps(steps: &str) -> PyResult<StepSelection> {
    match steps {
        "all" => Ok(StepSelection::All),
        "first" => Ok(StepSelection::First),
        other => Err(PyValueError::new_err(format!(
            "steps must be 'all' or 'first', got {other:?}"
        ))),
    }
}

#[pymethods]
impl PyHead {
    /// A pointer head over `set` plus `extra_actions` learned extras.
    #[staticmethod]
    #[pyo3(signature = (name, *, set, extra_actions = 0, condition_on = None, steps = "all", loss_weight = 1.0, step_weights = None))]
    fn pointer(
        name: String,
        set: String,
        extra_actions: usize,
        condition_on: Option<String>,
        steps: &str,
        loss_weight: f32,
        step_weights: Option<Vec<f32>>,
    ) -> PyResult<Self> {
        let mut h = HeadSpec::pointer(name, set, extra_actions);
        h.steps = parse_steps(steps)?;
        h.loss_weight = loss_weight;
        h.step_weights = step_weights;
        h.condition_on = condition_on;
        Ok(Self { inner: h })
    }

    /// A categorical head with `classes` logits.
    #[staticmethod]
    #[pyo3(signature = (name, *, classes, condition_on = None, steps = "all", loss_weight = 1.0, step_weights = None))]
    fn categorical(
        name: String,
        classes: usize,
        condition_on: Option<String>,
        steps: &str,
        loss_weight: f32,
        step_weights: Option<Vec<f32>>,
    ) -> PyResult<Self> {
        let mut h = HeadSpec::categorical(name, classes);
        h.steps = parse_steps(steps)?;
        h.loss_weight = loss_weight;
        h.step_weights = step_weights;
        h.condition_on = condition_on;
        Ok(Self { inner: h })
    }

    /// A multi-label head with `labels` Bernoulli logits.
    #[staticmethod]
    #[pyo3(signature = (name, *, labels, condition_on = None, steps = "all", loss_weight = 1.0, step_weights = None))]
    fn multilabel(
        name: String,
        labels: usize,
        condition_on: Option<String>,
        steps: &str,
        loss_weight: f32,
        step_weights: Option<Vec<f32>>,
    ) -> PyResult<Self> {
        let mut h = HeadSpec::multilabel(name, labels);
        h.steps = parse_steps(steps)?;
        h.loss_weight = loss_weight;
        h.step_weights = step_weights;
        h.condition_on = condition_on;
        Ok(Self { inner: h })
    }

    /// A regression head with `outputs` real numbers.
    #[staticmethod]
    #[pyo3(signature = (name, *, outputs, condition_on = None, steps = "all", loss_weight = 1.0, step_weights = None))]
    fn regression(
        name: String,
        outputs: usize,
        condition_on: Option<String>,
        steps: &str,
        loss_weight: f32,
        step_weights: Option<Vec<f32>>,
    ) -> PyResult<Self> {
        let mut h = HeadSpec::regression(name, outputs);
        h.steps = parse_steps(steps)?;
        h.loss_weight = loss_weight;
        h.step_weights = step_weights;
        h.condition_on = condition_on;
        Ok(Self { inner: h })
    }
}

/// The domain-free entity-model architecture.
#[pyclass(module = "mamba3_rl", name = "EntityModelSpec", from_py_object)]
#[derive(Clone)]
pub struct PyEntityModelSpec {
    pub(crate) inner: EntityModelSpec,
}

#[pymethods]
impl PyEntityModelSpec {
    /// Build the spec; the decoder defaults to step-causal with crew
    /// symmetry when a query set names `autoregressive_on`, else joint.
    /// `decoder` overrides it: `"joint"`, `"step_causal"` (forward-only scan,
    /// so queries are processed in slot order), `"step_causal_symmetric"`
    /// (plus the reversed second scan), or `"query_causal"` (forward-only
    /// scan with previous queries' picks embedded: ordered assignment).
    #[new]
    #[pyo3(signature = (*, globals = 0, context, queries = None, heads, d_model = 128,
                        context_layers = 3, decoder_layers = 3, decoder = None, seed = 0))]
    fn new(
        globals: usize,
        context: Vec<PyContextSet>,
        queries: Option<PyQuerySet>,
        heads: Vec<PyHead>,
        d_model: usize,
        context_layers: usize,
        decoder_layers: usize,
        decoder: Option<String>,
        seed: u64,
    ) -> PyResult<Self> {
        use mamba3::models::entity::DecoderMode;
        let queries = queries.map(|q| q.inner);
        let decoder = match decoder.as_deref() {
            None => EntityModelSpec::default_decoder(&queries),
            Some("joint") => DecoderMode::Joint,
            Some("step_causal") => DecoderMode::StepCausal { crew_symmetric: false },
            Some("step_causal_symmetric") => DecoderMode::StepCausal { crew_symmetric: true },
            Some("query_causal") => DecoderMode::QueryCausal,
            Some(other) => {
                return Err(PyValueError::new_err(format!(
                    "decoder must be 'joint', 'step_causal', 'step_causal_symmetric' or 'query_causal', got {other:?}"
                )));
            }
        };
        let inner = EntityModelSpec {
            globals,
            context: context.into_iter().map(|c| c.inner).collect(),
            queries: queries.clone(),
            heads: heads.into_iter().map(|h| h.inner).collect(),
            d_model,
            context_layers,
            decoder_layers,
            decoder,
            ssm: mamba3::ssm::config::SsmConfig {
                d_model,
                n_heads: 4,
                head_dim: 64,
                d_state: 32,
                n_groups: 1,
                ..Default::default()
            },
            chunk_size: None,
            norm_eps: 1e-5,
            seed,
        };
        inner.validate().py()?;
        Ok(Self { inner })
    }

    /// The spec as JSON.
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Rebuild from [`Self::to_json`].
    #[staticmethod]
    fn from_json(s: &str) -> PyResult<Self> {
        let inner: EntityModelSpec = serde_json::from_str(s)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        inner.validate().py()?;
        Ok(Self { inner })
    }

    fn __repr__(&self) -> String {
        format!(
            "EntityModelSpec(globals={}, context={}, queries={}, heads={}, d_model={})",
            self.inner.globals,
            self.inner.context.len(),
            self.inner.queries.as_ref().map(|q| q.name.as_str()).unwrap_or("-"),
            self.inner.heads.len(),
            self.inner.d_model,
        )
    }
}

/// Read one dict of NumPy arrays into [`HostArrays`], naming the key on any
/// failure. Integer arrays use `-1` for IGNORE; float label arrays use NaN.
/// Label keys route by head kind (class heads read integers, float heads read
/// floats); anchors read integers; presence and legal masks accept either.
fn read_arrays(spec: &EntityModelSpec, arrays: &Bound<'_, PyDict>) -> PyResult<HostArrays> {
    use mamba3::models::entity::HeadKind;
    let mut out = HostArrays::new();
    for (key, value) in arrays.iter() {
        let key: String = key.extract()?;
        if key == "globals" || spec.context.iter().any(|s| s.name == key) {
            let (shape, data) = read_floats(&value, &key)?;
            out.insert_f32(&key, shape, data);
            continue;
        }
        if let Some(q) = &spec.queries {
            if q.name == key {
                let (shape, data) = read_floats(&value, &key)?;
                out.insert_f32(&key, shape, data);
                continue;
            }
            if key == format!("{}.presence", q.name) || key.starts_with("legal.") {
                if let Ok((shape, data)) = read_floats(&value, &key) {
                    out.insert_f32(&key, shape, data);
                    continue;
                }
                let (shape, data) = read_ints(&value, &key)?;
                out.insert_int(&key, shape, data);
                continue;
            }
            if key == format!("{}.anchor", q.name) {
                let (shape, data) = read_ints(&value, &key)?;
                out.insert_int(&key, shape, data);
                continue;
            }
        }
        if spec.context.iter().any(|s| key == format!("{}.presence", s.name)) {
            if let Ok((shape, data)) = read_floats(&value, &key) {
                out.insert_f32(&key, shape, data);
                continue;
            }
            let (shape, data) = read_ints(&value, &key)?;
            out.insert_int(&key, shape, data);
            continue;
        }
        if let Some(head) = key.strip_prefix("label.").and_then(|n| spec.head(n)) {
            match &head.kind {
                HeadKind::Pointer { .. } | HeadKind::Categorical { .. } => {
                    let (shape, data) = read_ints(&value, &key)?;
                    out.insert_int(&key, shape, data);
                }
                HeadKind::MultiLabel { .. } | HeadKind::Regression { .. } => {
                    let (shape, data) = read_floats(&value, &key)?;
                    out.insert_f32(&key, shape, data);
                }
            }
            continue;
        }
        // Unknown keys still load as floats-or-ints so the batch builder —
        // not the reader — reports them.
        if let Ok((shape, data)) = read_floats(&value, &key) {
            out.insert_f32(&key, shape, data);
        } else {
            let (shape, data) = read_ints(&value, &key)?;
            out.insert_int(&key, shape, data);
        }
    }
    Ok(out)
}

/// One dataset split, validated against its spec and uploaded once.
#[pyclass(module = "mamba3_rl", name = "EntityDataset", unsendable)]
pub struct PyEntityDataset {
    inner: EntityDataset<R, E>,
    samples: usize,
}

#[pymethods]
impl PyEntityDataset {
    /// Validate `arrays` (a dict keyed as §1.3) against `spec` and upload it
    /// once. Unknown keys, bad shapes and out-of-range ids raise `ValueError`
    /// naming the key — before anything reaches the device.
    #[new]
    fn new(spec: &PyEntityModelSpec, arrays: &Bound<'_, PyDict>) -> PyResult<Self> {
        let device = mamba3::backend::Device::<R>::default();
        let host = read_arrays(&spec.inner, arrays)?;
        let inner = EntityDataset::from_arrays(&spec.inner, &host, &device)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let samples = inner.samples;
        Ok(Self { inner, samples })
    }

    /// Samples in this split.
    #[getter]
    fn samples(&self) -> usize {
        self.samples
    }
}

/// The domain-free entity-to-plan model.
#[pyclass(module = "mamba3_rl", name = "EntityModel", unsendable)]
pub struct PyEntityModel {
    inner: Rc<EntityModel<R, E>>,
    spec: EntityModelSpec,
    device: mamba3::backend::Device<R>,
    trainer: Trainer<R, E, AdamW<R, E>>,
    /// Queued steps with the per-head losses their own forward computed.
    queued: Vec<(QueuedStep<R, E>, LossComponents<R, E>)>,
    loss_scale: f32,
}

#[pymethods]
impl PyEntityModel {
    /// Build the model and its trainer. `matmul_precision` (`"f32"`/`"f16"`,
    /// `"bf16"` refused) and `loss_scale` behave as for the planner binding.
    #[new]
    #[pyo3(signature = (spec, *, learning_rate = 3e-4, weight_decay = 0.05,
                        max_grad_norm = 1.0, lr_schedule = None,
                        matmul_precision = "f32", loss_scale = 1.0))]
    fn new(
        spec: &PyEntityModelSpec,
        learning_rate: f32,
        weight_decay: f32,
        max_grad_norm: f32,
        lr_schedule: Option<PyLrSchedule>,
        matmul_precision: &str,
        loss_scale: f32,
    ) -> PyResult<Self> {
        use mamba3::tensor::ops::matmul::{MatmulPrecision, try_set_matmul_precision};
        let device = mamba3::backend::Device::<R>::default();
        match matmul_precision.to_lowercase().as_str() {
            "f32" => try_set_matmul_precision(&device, MatmulPrecision::F32).py()?,
            "f16" => try_set_matmul_precision(&device, MatmulPrecision::F16).py()?,
            "bf16" => {
                return Err(PyValueError::new_err(
                    "bf16 is not supported by this binding: only 'f32' and 'f16' are accepted",
                ));
            }
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown matmul_precision {other:?}; expected 'f32' or 'f16'"
                )));
            }
        }
        if !(loss_scale > 0.0) {
            return Err(PyValueError::new_err(format!(
                "loss_scale must be positive, got {loss_scale}"
            )));
        }
        let inner = Rc::new(
            EntityModel::init(&spec.inner, &device).py()?,
        );
        let schedule = lr_schedule.map(|s| s.inner).unwrap_or_default();
        let trainer = Trainer::new(
            TrainerConfig::builder()
                .learning_rate(learning_rate)
                .max_grad_norm(max_grad_norm * loss_scale)
                .schedule(schedule)
                .build()
                .py()?,
            AdamWConfig::builder()
                .learning_rate(learning_rate)
                .eps(1e-8 * loss_scale)
                .weight_decay(weight_decay)
                .build()
                .init::<R, E>(),
        );
        Ok(Self {
            inner,
            spec: spec.inner.clone(),
            device,
            trainer,
            queued: Vec::new(),
            loss_scale,
        })
    }

    /// Queue one training step over `ids` (an int array of sample indices);
    /// nothing is read back.
    fn queue_train_step(&mut self, dataset: &PyEntityDataset, ids: &Bound<'_, PyAny>) -> PyResult<()> {
        // K11: the four parts carry tally labels (`py.ids`, `py.gather`,
        // `py.step`, `py.components`), and `MAMBA3_PROFILE_PY=1` prints each
        // part's wall time so the Python overhead can be attributed.
        let profile = std::env::var("MAMBA3_PROFILE_PY").as_deref() == Ok("1");
        let mut marks = Vec::new();
        let mut mark = |name: &'static str, t: &std::time::Instant| {
            if profile {
                marks.push((name, t.elapsed()));
            }
        };
        let started = std::time::Instant::now();
        let (_, id_data) = {
            let _scope = mamba3::backend::tally_scope("py.ids");
            read_ints(ids, "ids")?
        };
        mark("ids", &started);
        let id_data: Vec<u32> = id_data
            .into_iter()
            .map(|v| {
                u32::try_from(v)
                    .map_err(|_| PyValueError::new_err(format!("ids holds {v}, outside u32 range")))
            })
            .collect::<PyResult<Vec<u32>>>()?;
        let t_gather = std::time::Instant::now();
        let batch = {
            let _scope = mamba3::backend::tally_scope("py.gather");
            EntityBatch::from_ids(&self.spec, &dataset.inner, &id_data)
                .map_err(|e| PyValueError::new_err(e.to_string()))?
        };
        mark("gather", &t_gather);
        let t_step = std::time::Instant::now();
        let task = EntityTask::new(&self.inner).with_loss_scale(self.loss_scale);
        let step = {
            let _scope = mamba3::backend::tally_scope("py.step");
            self.trainer
                .queue_step(&task, std::slice::from_ref(&batch))
                .py()?
        };
        mark("step", &t_step);
        let t_comp = std::time::Instant::now();
        // The step's own forward already computed every head's loss; keep
        // those rather than paying a second forward in `read_losses` (which
        // would also see the weights after this step's update).
        let components = {
            let _scope = mamba3::backend::tally_scope("py.components");
            task.take_components().ok_or_else(|| {
                PyRuntimeError::new_err("entity train step recorded no per-head losses")
            })?
        };
        mark("components", &t_comp);
        if profile {
            let total = started.elapsed();
            let parts: Vec<String> = marks
                .iter()
                .map(|(n, d)| format!("{n}={:.1}ms", d.as_secs_f64() * 1000.0))
                .collect();
            eprintln!(
                "[py] queue_train_step total={:.1}ms {}",
                total.as_secs_f64() * 1000.0,
                parts.join(" ")
            );
        }
        self.queued.push((step, components));
        Ok(())
    }

    /// `{"loss", "grad_norm", "heads": {name: loss}}` for every step queued
    /// since the last read, with the loss scale divided back out. `heads`
    /// are the per-head losses each step's own forward computed. One
    /// synchronisation for the whole backlog. A non-finite loss or gradient
    /// norm raises `FloatingPointError` naming the step.
    fn read_losses<'py>(&mut self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let queued = std::mem::take(&mut self.queued);
        if queued.is_empty() {
            return Ok(Vec::new());
        }
        let (steps, components): (Vec<QueuedStep<R, E>>, Vec<LossComponents<R, E>>) =
            queued.into_iter().unzip();
        let mut tensors: Vec<&Tensor<R, E>> = steps.iter().flat_map(|s| s.scalars()).collect();
        let n_step = tensors.len();
        for c in &components {
            tensors.extend(c.tensors());
        }
        let (_, values) = read_all(&[], &tensors).py()?;
        let step_values: Vec<f32> = values[..n_step].iter().map(|v| v[0]).collect();
        let mut rest = &values[n_step..];
        let mut heads_per_step = Vec::with_capacity(components.len());
        for c in &components {
            let n = c.tensors().len();
            heads_per_step.push(c.resolve(&rest[..n]).py()?);
            rest = &rest[n..];
        }
        let infos = self.trainer.report_steps(&steps, &step_values);
        let mut out = Vec::with_capacity(steps.len());
        for (i, info) in infos.iter().enumerate() {
            let loss = info.loss / self.loss_scale;
            let grad_norm = info.grad_norm / self.loss_scale;
            if !loss.is_finite() || !grad_norm.is_finite() {
                return Err(PyFloatingPointError::new_err(format!(
                    "entity step {} has a non-finite loss ({loss}) or gradient norm \
                     ({grad_norm}); halve the loss scale and restart from the last \
                     epoch checkpoint",
                    info.step,
                )));
            }
            let heads = PyDict::new(py);
            for (name, value) in &heads_per_step[i] {
                heads.set_item(name, value)?;
            }
            let entry = PyDict::new(py);
            entry.set_item("loss", loss)?;
            entry.set_item("grad_norm", grad_norm)?;
            entry.set_item("heads", heads)?;
            out.push(entry);
        }
        Ok(out)
    }

    /// Score `ids` in chunks of `batch`: `{"<head>": {"top1": ..}}` for
    /// pointers and categoricals, `{"bce": ..}` for multilabels,
    /// `{"mse": ..}` for regressions.
    fn evaluate<'py>(
        &self,
        py: Python<'py>,
        dataset: &PyEntityDataset,
        ids: &Bound<'_, PyAny>,
        batch: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        use mamba3::models::entity::EntityMetrics;
        let (_, id_data) = read_ints(ids, "ids")?;
        let ids: Vec<u32> = id_data
            .into_iter()
            .map(|v| {
                u32::try_from(v)
                    .map_err(|_| PyValueError::new_err(format!("ids holds {v}, outside u32 range")))
            })
            .collect::<PyResult<Vec<u32>>>()?;
        let batch = batch.max(1);
        let mut acc: BTreeMap<String, (f64, usize)> = BTreeMap::new();
        for chunk in ids.chunks(batch) {
            let b = EntityBatch::from_ids(&self.spec, &dataset.inner, chunk)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let metrics: EntityMetrics = self.inner.evaluate(&b).py()?;
            for (k, v) in metrics {
                let e = acc.entry(k).or_insert((0.0, 0));
                e.0 += v as f64;
                e.1 += 1;
            }
        }
        // Group dotted keys by head.
        let mut heads: BTreeMap<String, Bound<'_, PyDict>> = BTreeMap::new();
        for (k, (sum, n)) in acc {
            let (head, metric) = k.rsplit_once('.').unwrap_or(("", &k));
            let entry = heads
                .entry(head.to_string())
                .or_insert_with(|| PyDict::new(py));
            entry.set_item(metric, (sum / n.max(1) as f64) as f32)?;
        }
        let out = PyDict::new(py);
        for (head, metrics) in heads {
            out.set_item(head, metrics)?;
        }
        Ok(out)
    }

    /// Greedy decoding (`decode="greedy"`, the default) or conditioning on
    /// the labels (`decode="teacher_forced"`, diagnostics). `data` is an
    /// [`PyEntityDataset`] plus `ids`, or a plain dict of arrays.
    /// `chooser(step, logits)` maps `[B, M, N+E]` logits to `[B, M]` ints
    /// (inference only); the constrained choice feeds the next step.
    #[pyo3(signature = (data, ids = None, *, decode = "greedy", chooser = None))]
    fn predict<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
        ids: Option<&Bound<'py, PyAny>>,
        decode: &str,
        chooser: Option<Py<PyAny>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        use mamba3::models::entity::model::Decode;
        let decode = match decode {
            "greedy" => Decode::Greedy,
            "teacher_forced" => Decode::TeacherForced,
            other => {
                return Err(PyValueError::new_err(format!(
                    "decode must be 'greedy' or 'teacher_forced', got {other:?}"
                )));
            }
        };
        // A dataset plus ids, or a plain dict of arrays.
        let batch;
        if let Ok(dataset) = data.extract::<PyRef<'_, PyEntityDataset>>() {
            let ids = ids.ok_or_else(|| {
                PyValueError::new_err("predict on a dataset needs ids")
            })?;
            let (_, id_data) = read_ints(&ids, "ids")?;
            let id_data: Vec<u32> = id_data
                .into_iter()
                .map(|v| {
                    u32::try_from(v).map_err(|_| {
                        PyValueError::new_err(format!("ids holds {v}, outside u32 range"))
                    })
                })
                .collect::<PyResult<Vec<u32>>>()?;
            batch = EntityBatch::from_ids(&self.spec, &dataset.inner, &id_data)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
        } else if let Ok(dict) = data.cast::<PyDict>() {
            if ids.is_some() {
                return Err(PyValueError::new_err(
                    "predict on array dicts takes no ids",
                ));
            }
            let host = read_arrays(&self.spec, &dict)?;
            batch = EntityBatch::from_host(&self.spec, &host, &self.device)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
        } else {
            return Err(PyValueError::new_err(
                "predict needs an EntityDataset (plus ids) or a dict of arrays",
            ));
        }
        let (cb, cm) = (batch.b, self.spec.queries.as_ref().map(|q| q.count).unwrap_or(0));
        let mut chooser = chooser.map(|f| {
            let f: Py<PyAny> = f;
            move |step: usize, logits: &Tensor<R, E>| -> mamba3::error::Result<mamba3::tensor::ops::index::IdTensor<R>> {
                Python::attach(|py| {
                    let err = |msg: String| mamba3::error::Error::shape(msg);
                    let data = logits.try_to_f32().map_err(|e| err(e.to_string()))?;
                    let dims = logits.shape().dims().to_vec();
                    // [B*M, W] -> [B, M, W]: the chooser sees per-query rows.
                    let (bm, w) = (dims[0], dims[1]);
                    if bm != cb * cm {
                        return Err(err(format!(
                            "chooser logits hold {bm} rows, expected {cb}*{cm}"
                        )));
                    }
                    let array = PyArray1::from_vec(py, data)
                        .reshape((cb, cm, w))
                        .map_err(|e| err(e.to_string()))?;
                    let out = f.call1(py, (step, array)).map_err(|e| err(e.to_string()))?;
                    let out_bound = out.bind(py);
                    let got: Vec<i64> = match out_bound.extract::<IntArrayDyn<'_>>() {
                        Ok(a) => match a.as_slice() {
                            Ok(s) => s.to_vec(),
                            Err(_) => a.as_array().iter().copied().collect(),
                        },
                        Err(_) => {
                            return Err(err("chooser must return [B, M] ints".to_string()));
                        }
                    };
                    if got.len() != cb * cm {
                        return Err(err(format!(
                            "chooser returned {} ids, expected {cb}*{cm}",
                            got.len()
                        )));
                    }
                    let got: Vec<u32> = got
                        .into_iter()
                        .map(|v| {
                            u32::try_from(v).map_err(|_| {
                                err(format!("chooser returned {v}, outside u32 range"))
                            })
                        })
                        .collect::<Result<Vec<u32>, _>>()?;
                    mamba3::tensor::ops::index::IdTensor::from_slice(
                        &got,
                        vec![cb, cm],
                        &mamba3::backend::Device::<R>::default(),
                    )
                })
            }
        });
        let out = self
            .inner
            .predict(&batch, decode, chooser.as_mut().map(|f| f as &mut dyn FnMut(usize, &Tensor<R, E>) -> mamba3::error::Result<mamba3::tensor::ops::index::IdTensor<R>>))
            .py()?;
        let result = PyDict::new(py);
        for (name, logits) in &out.logits {
            let entry = PyDict::new(py);
            let dims = logits.shape().dims().to_vec();
            let flat = logits.try_to_f32().py()?;
            let array: Bound<'py, PyAny> = match dims.as_slice() {
                [b, m, k, w] => PyArray1::from_vec(py, flat)
                    .reshape((*b, *m, *k, *w))?
                    .into_any(),
                [b, m, w] => PyArray1::from_vec(py, flat)
                    .reshape((*b, *m, *w))?
                    .into_any(),
                _ => {
                    return Err(PyValueError::new_err(format!(
                        "head {name} logits have unexpected shape {dims:?}"
                    )));
                }
            };
            entry.set_item("logits", array)?;
            if let Some(ids) = out.choices.get(name) {
                let dd = ids.shape().dims().to_vec();
                match dd.as_slice() {
                    [b, m, k] => {
                        entry.set_item(
                            "choice",
                            PyArray1::from_vec(py, ids.to_vec()).reshape((*b, *m, *k))?,
                        )?;
                    }
                    _ => {
                        return Err(PyValueError::new_err(format!(
                            "head {name} choices have unexpected shape {dd:?}"
                        )));
                    }
                }
            }
            result.set_item(name, entry)?;
        }
        Ok(result)
    }

    /// Write the weights and the spec to a checkpoint.
    #[pyo3(signature = (path, step = 0))]
    fn save(&self, path: &str, step: u64) -> PyResult<()> {
        self.inner.save(path, step).py()
    }

    /// Read back what [`Self::save`] wrote. The trainer is fresh: only the
    /// weights travel, so pass the same `learning_rate` / `loss_scale` to the
    /// constructor when resuming.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let device = mamba3::backend::Device::<R>::default();
        let inner = Rc::new(EntityModel::load(path, &device).py()?);
        let spec = inner.spec().clone();
        let trainer = Trainer::new(
            TrainerConfig::builder().build().py()?,
            AdamWConfig::builder().build().init::<R, E>(),
        );
        Ok(Self {
            inner,
            spec,
            device,
            trainer,
            queued: Vec::new(),
            loss_scale: 1.0,
        })
    }
}

/// Register the entity-model classes.
pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyGrid>()?;
    module.add_class::<PyContextSet>()?;
    module.add_class::<PyQuerySet>()?;
    module.add_class::<PyHead>()?;
    module.add_class::<PyEntityModelSpec>()?;
    module.add_class::<PyEntityDataset>()?;
    module.add_class::<PyEntityModel>()?;
    Ok(())
}
