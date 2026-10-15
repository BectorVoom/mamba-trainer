//! Graph matching from Python: `MatchQueries`, `MatchGroups` and
//! `GraphMatchTrainer` of the `mamba3_graph` import module
//! (`mamba3::models::graph::matching`).
//!
//! A child of the graph bindings, so that it shares their device, their array
//! borrowing and their dataset class. The matcher is `f32` only: a dataset of
//! another dtype is refused.

use super::*;

use mamba3::models::graph::{
    GraphMatchConfig, GraphMatchModel, GraphMatchTrainer, MatchEpoch, MatchGroups, MatchQueries,
};

type Matcher = GraphMatchModel<R, f32>;

/// Integer ids of any dtype as `u32`, refusing negative and too large ones.
fn ids_u32(value: &Bound<'_, PyAny>, key: &str) -> PyResult<(Vec<usize>, Vec<u32>)> {
    let (shape, held) = hold_ints(value, key)?;
    macro_rules! convert {
        ($slice:expr) => {
            $slice
                .iter()
                .map(|&v| {
                    u32::try_from(v).map_err(|_| {
                        PyValueError::new_err(format!("{key} holds {v}, outside 0..2^32"))
                    })
                })
                .collect::<PyResult<Vec<u32>>>()?
        };
    }
    let ids = match held.view() {
        Ints::I8(s) => convert!(s),
        Ints::I16(s) => convert!(s),
        Ints::I32(s) => convert!(s),
        Ints::I64(s) => convert!(s),
        Ints::U8(s) => s.iter().map(|&v| v as u32).collect(),
        Ints::U16(s) => s.iter().map(|&v| v as u32).collect(),
        Ints::U32(s) => s.to_vec(),
        Ints::U64(s) => convert!(s),
    };
    Ok((shape, ids))
}

/// Floats of any width as `f32`.
fn floats_f32(value: &Bound<'_, PyAny>, key: &str) -> PyResult<(Vec<usize>, Vec<f32>)> {
    let (shape, held) = hold_floats(value, key)?;
    let values = match held.view() {
        Floats::F16(s) => s.iter().map(|v| v.to_f32()).collect(),
        Floats::F32(s) => s.to_vec(),
        Floats::F64(s) => s.iter().map(|&v| v as f32).collect(),
    };
    Ok((shape, values))
}

/// The dataset as the matcher's dtype.
fn f32_dataset(data: &PyGraphDataset) -> PyResult<&Dataset<f32>> {
    match &data.inner {
        AnyDataset::F32(dataset) => Ok(dataset),
        other => Err(PyValueError::new_err(format!(
            "graph matching trains in f32 and the dataset is {}: build it with dtype=\"f32\"",
            other.dtype().name()
        ))),
    }
}

// ---------------------------------------------------------------------------
// Queries and groups
// ---------------------------------------------------------------------------

/// Every query vector of a run, uploaded once: ``[queries, k]`` bit ids and
/// weights, ``k <= 128``; a weight of 0 is padding.
#[pyclass(module = "mamba3_graph", name = "MatchQueries", unsendable)]
pub struct PyMatchQueries {
    inner: MatchQueries<R, f32>,
}

#[pymethods]
impl PyMatchQueries {
    /// Validate and upload ``bits`` (integers ``[Q, K]``) and ``weights``
    /// (floats ``[Q, K]``). Every entry with a non-zero weight must name a bit
    /// below ``n_bits`` and have a finite weight; a bit named twice in one
    /// query adds its weights.
    #[new]
    #[pyo3(signature = (bits, weights, n_bits = 4096))]
    fn new(
        py: Python<'_>,
        bits: &Bound<'_, PyAny>,
        weights: &Bound<'_, PyAny>,
        n_bits: usize,
    ) -> PyResult<Self> {
        let (bit_shape, bits) = ids_u32(bits, "bits")?;
        let (weight_shape, weights) = floats_f32(weights, "weights")?;
        if bit_shape.len() != 2 || bit_shape != weight_shape {
            return Err(PyValueError::new_err(format!(
                "bits and weights must both be [queries, k], got shapes {bit_shape:?} and \
                 {weight_shape:?}"
            )));
        }
        let k = bit_shape[1];
        let device = device();
        let inner = detached(py, || {
            MatchQueries::new(&bits, &weights, k, n_bits, &device)
        })
        .py()?;
        Ok(Self { inner })
    }

    /// Number of queries.
    #[getter]
    fn num_queries(&self) -> usize {
        self.inner.len()
    }

    /// Entries per query row.
    #[getter]
    fn k(&self) -> usize {
        self.inner.k()
    }

    /// Width of the bit space.
    #[getter]
    fn n_bits(&self) -> usize {
        self.inner.n_bits()
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "MatchQueries(queries={}, k={}, n_bits={})",
            self.inner.len(),
            self.inner.k(),
            self.inner.n_bits()
        )
    }
}

/// The groups of a run: a query id per group and its member graphs (dataset
/// graph ids), answer first. Repeated members of a group are dropped.
#[pyclass(module = "mamba3_graph", name = "MatchGroups", unsendable)]
pub struct PyMatchGroups {
    inner: MatchGroups,
}

#[pymethods]
impl PyMatchGroups {
    /// ``query`` ``[G]``, ``member_ptr`` ``[G + 1]`` (CSR offsets starting at
    /// 0) and ``members`` ``[M]``, all integers. Every group needs a member.
    #[new]
    #[pyo3(signature = (query, member_ptr, members))]
    fn new(
        query: &Bound<'_, PyAny>,
        member_ptr: &Bound<'_, PyAny>,
        members: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let (_, query) = ids_u32(query, "query")?;
        let (_, ptr) = ids_u32(member_ptr, "member_ptr")?;
        let (_, members) = ids_u32(members, "members")?;
        Ok(Self {
            inner: MatchGroups::new(&query, &ptr, &members).py()?,
        })
    }

    /// Number of groups.
    #[getter]
    fn num_groups(&self) -> usize {
        self.inner.len()
    }

    /// The ids of the groups without a negative (fewer than two distinct
    /// members): refused by ``train_epoch``, accepted by ``score_groups``.
    fn trivial<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        PyArray1::from_vec(py, self.inner.trivial())
    }

    /// The distinct members of group ``g``, answer first.
    fn members<'py>(&self, py: Python<'py>, g: usize) -> PyResult<Bound<'py, PyArray1<u32>>> {
        self.check(g)?;
        Ok(PyArray1::from_slice(py, self.inner.members(g)))
    }

    /// The query id of group ``g``.
    fn query(&self, g: usize) -> PyResult<u32> {
        self.check(g)?;
        Ok(self.inner.query(g))
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "MatchGroups(groups={}, trivial={})",
            self.inner.len(),
            self.inner.trivial().len()
        )
    }
}

impl PyMatchGroups {
    fn check(&self, g: usize) -> PyResult<()> {
        if g >= self.inner.len() {
            return Err(PyValueError::new_err(format!(
                "group {g} of {} groups",
                self.inner.len()
            )));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// The trainer
// ---------------------------------------------------------------------------

/// A graph matcher with its AdamW optimizer: Graph Mamba embeds the member
/// graphs, an encoder embeds the query bits, and a member scores
/// ``scale * cos(query, graph)``.
#[pyclass(module = "mamba3_graph", name = "GraphMatchTrainer", unsendable)]
pub struct PyGraphMatchTrainer {
    model: Matcher,
    trainer: GraphMatchTrainer<R, f32>,
}

fn match_train_config(
    learning_rate: f32,
    weight_decay: f32,
    max_grad_norm: f32,
    lr_schedule: Option<PyLrSchedule>,
) -> GraphTrainConfig {
    GraphTrainConfig {
        learning_rate,
        weight_decay,
        max_grad_norm,
        schedule: lr_schedule.map(|s| s.inner).unwrap_or_default(),
        loss_scale: 1.0,
    }
}

/// An epoch over caller-chosen batches of group ids.
#[allow(clippy::too_many_arguments)]
fn match_epoch(
    model: &Matcher,
    dataset: &Dataset<f32>,
    queries: &PyMatchQueries,
    groups: &PyMatchGroups,
    batches: &[Vec<u32>],
    batch_rows: Option<usize>,
    epoch: u64,
    training: bool,
) -> mamba3::error::Result<MatchEpoch<R, f32>> {
    MatchEpoch::new(
        dataset,
        &queries.inner,
        &groups.inner,
        batches,
        batch_rows,
        epoch,
        training,
        model.config(),
    )
}

#[pymethods]
impl PyGraphMatchTrainer {
    /// Build the matcher and its optimizer. ``spec`` needs a graph-level task
    /// (``GraphRegression(d_out, pool="mean")``); its output width is the
    /// embedding width and its own loss is not used. ``seed`` seeds the
    /// query encoder; the graph model uses the spec's. A debug build raises a
    /// ``RuntimeWarning``.
    #[new]
    #[pyo3(signature = (spec, *, n_bits = 4096, query_hidden = 512, learning_rate = 1e-3,
                        weight_decay = 0.01, max_grad_norm = 1.0, lr_schedule = None, seed = 0))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        spec: &PyGraphMambaSpec,
        n_bits: usize,
        query_hidden: usize,
        learning_rate: f32,
        weight_decay: f32,
        max_grad_norm: f32,
        lr_schedule: Option<PyLrSchedule>,
        seed: u64,
    ) -> PyResult<Self> {
        warn_if_debug_build(py, None)?;
        let config = GraphMatchConfig::new(spec.inner.clone(), n_bits, query_hidden, seed);
        let train = match_train_config(learning_rate, weight_decay, max_grad_norm, lr_schedule);
        let device = device();
        let (model, trainer) = detached(py, || {
            Ok::<_, mamba3::error::Error>((
                GraphMatchModel::init(&config, &device)?,
                GraphMatchTrainer::new(&train)?,
            ))
        })
        .py()?;
        Ok(Self { model, trainer })
    }

    /// Queue one training step per batch (a list of group ids) and return how
    /// many were queued. Each batch's graphs are the distinct members of its
    /// groups; ``batch_rows`` is the row capacity of every graph batch
    /// (default: the largest batch's), and passing the same value every
    /// epoch keeps the shapes few. Groups without a negative are refused.
    /// Nothing is read back; the lock is released for the call, and Ctrl-C
    /// stops it after a completed step.
    #[pyo3(signature = (data, queries, groups, batches, epoch = 0, *, batch_rows = None))]
    #[allow(clippy::too_many_arguments)]
    fn train_epoch(
        &mut self,
        py: Python<'_>,
        data: &PyGraphDataset,
        queries: &PyMatchQueries,
        groups: &PyMatchGroups,
        batches: Vec<Vec<u32>>,
        epoch: u64,
        batch_rows: Option<usize>,
    ) -> PyResult<usize> {
        let dataset = f32_dataset(data)?;
        let Self { model, trainer } = self;
        let outcome = detached(py, || {
            let plan = match_epoch(
                model, dataset, queries, groups, &batches, batch_rows, epoch, true,
            )
            .map_err(Stopped::Failed)?;
            let mut last_check = Instant::now();
            let mut interrupt = None;
            let result = trainer.train_epoch_with(model, &plan, |_| {
                if last_check.elapsed() < Duration::from_millis(50) {
                    return ControlFlow::Continue(());
                }
                last_check = Instant::now();
                match Python::attach(|py| py.check_signals()) {
                    Ok(()) => ControlFlow::Continue(()),
                    Err(err) => {
                        interrupt = Some(err);
                        ControlFlow::Break(())
                    }
                }
            });
            match (interrupt, result) {
                (Some(err), _) => Err(Stopped::Interrupted(err)),
                (None, Ok(steps)) => Ok(steps),
                (None, Err(err)) => Err(Stopped::Failed(err)),
            }
        });
        match outcome {
            Ok(steps) => Ok(steps),
            Err(Stopped::Interrupted(err)) => Err(err),
            Err(Stopped::Failed(err)) => Err(crate::err::to_py(err)),
        }
    }

    /// The loss of every step queued since the last call, oldest first. One
    /// device read. A non-finite loss or gradient norm raises
    /// ``FloatingPointError``.
    fn read_losses(&mut self, py: Python<'_>) -> PyResult<Vec<f32>> {
        let trainer = &mut self.trainer;
        let infos = detached(py, || trainer.read_losses()).py()?;
        let mut out = Vec::with_capacity(infos.len());
        for info in infos {
            if !info.loss.is_finite() || !info.grad_norm.is_finite() {
                return Err(PyFloatingPointError::new_err(format!(
                    "step {} has a non-finite loss ({}) or gradient norm ({}); lower the \
                     learning rate",
                    info.step, info.loss, info.grad_norm
                )));
            }
            out.push(info.loss);
        }
        Ok(out)
    }

    /// Evaluation mode, no gradient: per group, in the order the batches list
    /// them, ``float32`` scores ``scale * cos`` of its distinct members,
    /// answer first. Groups with one member are accepted. Reads every few
    /// batches.
    #[pyo3(signature = (data, queries, groups, batches, *, batch_rows = None))]
    fn score_groups<'py>(
        &self,
        py: Python<'py>,
        data: &PyGraphDataset,
        queries: &PyMatchQueries,
        groups: &PyMatchGroups,
        batches: Vec<Vec<u32>>,
        batch_rows: Option<usize>,
    ) -> PyResult<Vec<Bound<'py, PyArray1<f32>>>> {
        let dataset = f32_dataset(data)?;
        let model = &self.model;
        let scores = detached(py, || {
            let plan = match_epoch(
                model, dataset, queries, groups, &batches, batch_rows, 0, false,
            )?;
            model.score(&plan)
        })
        .py()?;
        Ok(scores
            .into_iter()
            .map(|row| PyArray1::from_vec(py, row))
            .collect())
    }

    /// Evaluation mode: ``float32[n, d_out]`` L2-normalised embeddings of
    /// ``graph_ids`` (default: every graph), in that order. Batches hold at
    /// most ``batch_graphs`` graphs and are read as they finish.
    #[pyo3(signature = (data, graph_ids = None, batch_graphs = 4096))]
    fn embed_graphs<'py>(
        &self,
        py: Python<'py>,
        data: &PyGraphDataset,
        graph_ids: Option<&Bound<'_, PyAny>>,
        batch_graphs: usize,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let dataset = f32_dataset(data)?;
        let ids = graph_ids
            .map(|ids| ids_u32(ids, "graph_ids").map(|(_, ids)| ids))
            .transpose()?;
        let model = &self.model;
        let values = detached(py, || {
            model.embed_graphs(dataset, ids.as_deref(), batch_graphs)
        })
        .py()?;
        let d = model.config().query.d_out;
        let rows = values.len() / d;
        PyArray1::from_vec(py, values).reshape((rows, d))
    }

    /// Evaluation mode: ``float32[queries, d_out]`` L2-normalised embeddings
    /// of every query, ``batch_queries`` at a time.
    #[pyo3(signature = (queries, batch_queries = 1024))]
    fn embed_queries<'py>(
        &self,
        py: Python<'py>,
        queries: &PyMatchQueries,
        batch_queries: usize,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let model = &self.model;
        let values = detached(py, || model.embed_queries(&queries.inner, batch_queries)).py()?;
        let d = model.config().query.d_out;
        PyArray1::from_vec(py, values).reshape((queries.inner.len(), d))
    }

    /// The logit scale ``exp(clamp(log_scale, 0, ln 100))``. One device read.
    #[getter]
    fn scale(&self, py: Python<'_>) -> PyResult<f32> {
        let model = &self.model;
        detached(py, || model.scale()).py()
    }

    /// Optimizer steps taken so far.
    #[getter]
    fn step(&self) -> u64 {
        self.trainer.step_count()
    }

    /// Scalars in the matcher (graph model, query encoder, log scale).
    #[getter]
    fn num_parameters(&self) -> usize {
        self.model.num_parameters()
    }

    /// Width of the embeddings.
    #[getter]
    fn d_out(&self) -> usize {
        self.model.config().query.d_out
    }

    /// Width of the query bit space.
    #[getter]
    fn n_bits(&self) -> usize {
        self.model.config().query.n_bits
    }

    /// The graph model's spec.
    #[getter]
    fn spec(&self) -> PyGraphMambaSpec {
        PyGraphMambaSpec {
            inner: self.model.config().spec.clone(),
        }
    }

    /// Save the complete matcher: the spec, the encoder's config and every
    /// weight. ``step`` defaults to the optimizer's.
    #[pyo3(signature = (path, step = None))]
    fn save(&self, py: Python<'_>, path: &str, step: Option<u64>) -> PyResult<()> {
        let step = step.unwrap_or_else(|| self.trainer.step_count());
        let path = path.to_owned();
        let model = &self.model;
        detached(py, || model.save(&path, step)).py()
    }

    /// Rebuild a matcher from ``save``'s file, with a fresh optimizer whose
    /// step counter resumes at the saved step.
    #[staticmethod]
    #[pyo3(signature = (path, *, learning_rate = 1e-3, weight_decay = 0.01, max_grad_norm = 1.0,
                        lr_schedule = None))]
    fn load(
        py: Python<'_>,
        path: &str,
        learning_rate: f32,
        weight_decay: f32,
        max_grad_norm: f32,
        lr_schedule: Option<PyLrSchedule>,
    ) -> PyResult<Self> {
        warn_if_debug_build(py, None)?;
        let train = match_train_config(learning_rate, weight_decay, max_grad_norm, lr_schedule);
        let device = device();
        let path = path.to_owned();
        let (model, trainer) = detached(py, || {
            let (model, step) = GraphMatchModel::load(&path, &device)?;
            let mut trainer = GraphMatchTrainer::new(&train)?;
            trainer.set_step_count(step);
            Ok::<_, mamba3::error::Error>((model, trainer))
        })
        .py()?;
        Ok(Self { model, trainer })
    }

    fn __repr__(&self) -> String {
        format!(
            "GraphMatchTrainer(parameters={}, d_out={}, n_bits={}, step={})",
            self.num_parameters(),
            self.d_out(),
            self.n_bits(),
            self.step()
        )
    }
}

/// Add the matching classes to the extension module.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyMatchQueries>()?;
    module.add_class::<PyMatchGroups>()?;
    module.add_class::<PyGraphMatchTrainer>()?;
    Ok(())
}
