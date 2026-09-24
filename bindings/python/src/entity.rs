//! Structured observations: entity sets, their encoders, pooling and the
//! pointer head.
//!
//! [`PyObsSpec`] is the reading of a flat observation as `globals` followed by
//! entity sets, each entity `features` values then one presence flag. Its
//! `pack`/`pack_batch`/`unpack` build and read that wire format, so an
//! environment never computes an offset by hand; they call the Rust
//! [`ObsSpec::pack_into`] and [`ObsSpec::unpack`], so the two languages cannot
//! disagree about the layout.
//!
//! ```python
//! spec = m3.ObsSpec(globals=4, sets=[m3.EntitySet("tiles", count=100, features=60)])
//! obs = spec.pack(globals=g, tiles=(features, present))      # float32[obs_dim]
//! batch = spec.pack_batch(globals=G, tiles=(F, P))           # [num_envs, obs_dim]
//! parts = spec.unpack(obs)                                   # {"globals": ..., "tiles": (F, P)}
//! ```

use mamba3::nn::entity::{EntityEncoderConfig, PoolKind, PoolingConfig};
use mamba3::rl::{EntitySet, ObsSpec, PointerHeadConfig, Scoring};
use numpy::{AllowTypeChange, PyArray1, PyArrayLikeDyn, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use crate::err::IntoPyResult;

/// Any float array, whatever its rank or dtype.
type FloatArrayDyn<'py> = PyArrayLikeDyn<'py, f32, AllowTypeChange>;

/// `(shape, values)` of a float array, with a message naming `what` if it is not one.
fn read_array(value: &Bound<'_, PyAny>, what: &str) -> PyResult<(Vec<usize>, Vec<f32>)> {
    let array: FloatArrayDyn<'_> = value.extract().map_err(|_| {
        PyValueError::new_err(format!("{what} must be an array of floats"))
    })?;
    let shape = array.shape().to_vec();
    let data = match array.as_slice() {
        Ok(slice) => slice.to_vec(),
        Err(_) => array.as_array().iter().copied().collect(),
    };
    Ok((shape, data))
}

/// Check a shape, naming what was expected.
fn expect_shape(what: &str, got: &[usize], want: &[usize]) -> PyResult<()> {
    if got == want {
        return Ok(());
    }
    Err(PyValueError::new_err(format!(
        "{what} must be shaped {want:?}, got {got:?}"
    )))
}

/// One kind of entity: `count` slots of `features` values, each followed by a
/// presence flag.
#[pyclass(module = "mamba3_rl", name = "EntitySet", from_py_object)]
#[derive(Clone)]
pub struct PyEntitySet {
    pub(crate) inner: EntitySet,
}

#[pymethods]
impl PyEntitySet {
    #[new]
    #[pyo3(signature = (name, *, count, features))]
    fn new(name: String, count: usize, features: usize) -> Self {
        Self {
            inner: EntitySet::new(name, count, features),
        }
    }

    /// The set's name.
    #[getter]
    fn name(&self) -> String {
        self.inner.name.clone()
    }

    /// Slots in the set.
    #[getter]
    fn count(&self) -> usize {
        self.inner.count
    }

    /// Features per entity, not counting the presence flag.
    #[getter]
    fn features(&self) -> usize {
        self.inner.features
    }

    /// Width on the wire: `count * (features + 1)`.
    #[getter]
    fn width(&self) -> usize {
        self.inner.width()
    }

    fn __repr__(&self) -> String {
        format!(
            "EntitySet({:?}, count={}, features={})",
            self.inner.name, self.inner.count, self.inner.features
        )
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

/// How a policy reads a flat observation: `globals` leading values, then each
/// entity set.
#[pyclass(module = "mamba3_rl", name = "ObsSpec", from_py_object)]
#[derive(Clone)]
pub struct PyObsSpec {
    pub(crate) inner: ObsSpec,
}

#[pymethods]
impl PyObsSpec {
    #[new]
    #[pyo3(signature = (*, globals = 0, sets))]
    fn new(globals: usize, sets: Vec<PyEntitySet>) -> PyResult<Self> {
        let inner = ObsSpec::new(globals, sets.into_iter().map(|s| s.inner).collect());
        inner.validate().py()?;
        Ok(Self { inner })
    }

    /// Values that belong to no entity.
    #[getter]
    fn globals(&self) -> usize {
        self.inner.globals
    }

    /// The entity sets, in wire order.
    #[getter]
    fn sets(&self) -> Vec<PyEntitySet> {
        self.inner
            .sets
            .iter()
            .map(|s| PyEntitySet { inner: s.clone() })
            .collect()
    }

    /// Width of the flat observation.
    #[getter]
    fn obs_dim(&self) -> usize {
        self.inner.obs_dim()
    }

    /// Where each set starts in the flat observation, by name.
    fn offsets(&self) -> Vec<(String, usize)> {
        self.inner
            .sets
            .iter()
            .zip(self.inner.offsets())
            .map(|(s, at)| (s.name.clone(), at))
            .collect()
    }

    /// One flat `float32[obs_dim]` observation.
    ///
    /// Each set is a keyword argument named after it: `(features, presence)`
    /// with `features` shaped `[count, features]` and `presence` `[count]`, or
    /// `features` alone, meaning every slot is present.
    #[pyo3(signature = (globals = None, **sets))]
    fn pack<'py>(
        &self,
        py: Python<'py>,
        globals: Option<&Bound<'py, PyAny>>,
        sets: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let data = self.gather(globals, sets, None)?;
        Ok(PyArray1::from_vec(py, data))
    }

    /// `[num_envs, obs_dim]` observations, one row per environment.
    ///
    /// The same keywords as `pack`, with a leading `num_envs` axis on every
    /// array: `globals` `[num_envs, globals]`, features `[num_envs, count,
    /// features]`, presence `[num_envs, count]`.
    #[pyo3(signature = (globals = None, **sets))]
    fn pack_batch<'py>(
        &self,
        py: Python<'py>,
        globals: Option<&Bound<'py, PyAny>>,
        sets: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let envs = self.batch_size(globals, sets)?;
        let data = self.gather(globals, sets, Some(envs))?;
        Ok(PyArray1::from_vec(py, data)
            .reshape((envs, self.inner.obs_dim()))?
            .into_any())
    }

    /// The inverse of `pack`/`pack_batch`: `{"globals": ..., name: (features,
    /// presence), ...}`, with a leading axis when `obs` is `[num_envs, obs_dim]`.
    fn unpack<'py>(&self, py: Python<'py>, obs: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyDict>> {
        let (shape, data) = read_array(obs, "obs")?;
        let dim = self.inner.obs_dim();
        let (lead, rows): (Vec<usize>, usize) = match shape.as_slice() {
            [d] if *d == dim => (vec![], 1),
            [e, d] if *d == dim => (vec![*e], *e),
            _ => {
                return Err(PyValueError::new_err(format!(
                    "obs must be [{dim}] or [num_envs, {dim}], got {shape:?}"
                )));
            }
        };
        let mut globals = Vec::with_capacity(rows * self.inner.globals);
        let mut sets: Vec<(Vec<f32>, Vec<f32>)> = self
            .inner
            .sets
            .iter()
            .map(|s| {
                (
                    Vec::with_capacity(rows * s.count * s.features),
                    Vec::with_capacity(rows * s.count),
                )
            })
            .collect();
        for row in data.chunks_exact(dim.max(1)).take(rows) {
            let (g, parts) = self.inner.unpack(row).py()?;
            globals.extend(g);
            for ((features, presence), (f, p)) in sets.iter_mut().zip(parts) {
                features.extend(f);
                presence.extend(p);
            }
        }

        let with_lead = |tail: &[usize]| -> Vec<usize> { lead.iter().chain(tail).copied().collect() };
        let out = PyDict::new(py);
        out.set_item(
            "globals",
            PyArray1::from_vec(py, globals).reshape(with_lead(&[self.inner.globals]))?,
        )?;
        for (set, (features, presence)) in self.inner.sets.iter().zip(sets) {
            let features =
                PyArray1::from_vec(py, features).reshape(with_lead(&[set.count, set.features]))?;
            let presence = PyArray1::from_vec(py, presence).reshape(with_lead(&[set.count]))?;
            out.set_item(&set.name, PyTuple::new(py, [features.into_any(), presence.into_any()])?)?;
        }
        Ok(out)
    }

    /// The spec as a plain dictionary, as a checkpoint stores it.
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        json_to_py(py, &serde_json::to_value(&self.inner).map_err(json_err)?)
    }

    /// Rebuild a spec from `to_dict`.
    #[staticmethod]
    fn from_dict(mapping: &Bound<'_, PyAny>) -> PyResult<Self> {
        let inner: ObsSpec = serde_json::from_value(py_to_json(mapping)?)
            .map_err(|e| PyValueError::new_err(format!("not an obs spec: {e}")))?;
        inner.validate().py()?;
        Ok(Self { inner })
    }

    fn __repr__(&self) -> String {
        let sets: Vec<String> = self
            .inner
            .sets
            .iter()
            .map(|s| format!("{}[{}×{}]", s.name, s.count, s.features))
            .collect();
        format!(
            "ObsSpec(globals={}, sets=[{}], obs_dim={})",
            self.inner.globals,
            sets.join(", "),
            self.inner.obs_dim()
        )
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl PyObsSpec {
    /// `num_envs`, read off the first array `pack_batch` was given.
    fn batch_size(
        &self,
        globals: Option<&Bound<'_, PyAny>>,
        sets: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<usize> {
        if let Some(g) = globals {
            let (shape, _) = read_array(g, "globals")?;
            return shape.first().copied().ok_or_else(|| {
                PyValueError::new_err("pack_batch: globals must be [num_envs, globals]")
            });
        }
        let first = self.inner.sets.first().map(|s| s.name.as_str());
        let value = match (sets, first) {
            (Some(sets), Some(name)) => sets.get_item(name)?,
            _ => None,
        };
        let value = value.ok_or_else(|| {
            PyValueError::new_err("pack_batch: no arrays given to read num_envs from")
        })?;
        let features = match value.cast::<PyTuple>() {
            Ok(pair) => pair.get_item(0)?,
            Err(_) => value,
        };
        let (shape, _) = read_array(&features, "features")?;
        shape.first().copied().ok_or_else(|| {
            PyValueError::new_err("pack_batch: features must be [num_envs, count, features]")
        })
    }

    /// Read every array and pack `rows` observations (`None` = one, unbatched).
    fn gather(
        &self,
        globals: Option<&Bound<'_, PyAny>>,
        sets: Option<&Bound<'_, PyDict>>,
        rows: Option<usize>,
    ) -> PyResult<Vec<f32>> {
        let n = rows.unwrap_or(1);
        let lead: Vec<usize> = rows.into_iter().collect();
        let with_lead = |tail: &[usize]| -> Vec<usize> { lead.iter().chain(tail).copied().collect() };
        let spec = &self.inner;

        let globals = match globals {
            Some(g) => {
                let (shape, data) = read_array(g, "globals")?;
                expect_shape("globals", &shape, &with_lead(&[spec.globals]))?;
                data
            }
            None if spec.globals == 0 => Vec::new(),
            None => {
                return Err(PyValueError::new_err(format!(
                    "globals is missing; the spec has {} of them",
                    spec.globals
                )));
            }
        };

        if let Some(sets) = sets {
            for key in sets.keys() {
                let key: String = key.extract()?;
                if spec.set(&key).is_none() {
                    return Err(PyValueError::new_err(format!(
                        "{key:?} is not an entity set of this spec (sets: {:?})",
                        spec.sets.iter().map(|s| s.name.as_str()).collect::<Vec<_>>()
                    )));
                }
            }
        }
        let mut parts = Vec::with_capacity(spec.sets.len());
        for set in &spec.sets {
            let value = sets
                .map(|d| d.get_item(&set.name))
                .transpose()?
                .flatten()
                .ok_or_else(|| {
                    PyValueError::new_err(format!("entity set {:?} is missing", set.name))
                })?;
            let (features, presence) = match value.cast::<PyTuple>() {
                Ok(pair) if pair.len() == 2 => (pair.get_item(0)?, Some(pair.get_item(1)?)),
                Ok(_) => {
                    return Err(PyValueError::new_err(format!(
                        "{} must be (features, presence) or features alone",
                        set.name
                    )));
                }
                Err(_) => (value, None),
            };
            let what = format!("{} features", set.name);
            let (shape, features) = read_array(&features, &what)?;
            expect_shape(&what, &shape, &with_lead(&[set.count, set.features]))?;
            let presence = match presence {
                Some(p) => {
                    let what = format!("{} presence", set.name);
                    let (shape, data) = read_array(&p, &what)?;
                    expect_shape(&what, &shape, &with_lead(&[set.count]))?;
                    data
                }
                None => vec![1.0; n * set.count],
            };
            parts.push((features, presence));
        }

        let dim = spec.obs_dim();
        let mut out = vec![0.0f32; n * dim];
        for (r, row) in out.chunks_exact_mut(dim.max(1)).enumerate().take(n) {
            let sets: Vec<(&[f32], &[f32])> = spec
                .sets
                .iter()
                .zip(&parts)
                .map(|(set, (f, p))| {
                    let fw = set.count * set.features;
                    (&f[r * fw..(r + 1) * fw], &p[r * set.count..(r + 1) * set.count])
                })
                .collect();
            let g = &globals[r * spec.globals..(r + 1) * spec.globals];
            spec.pack_into(g, &sets, row).py()?;
        }
        Ok(out)
    }
}

/// The shared MLP that embeds every entity of one set.
///
/// `hidden` widths each take a ReLU; `slot_embedding=True` adds a learned
/// vector per slot index (off by default: it reintroduces per-slot parameters).
#[pyclass(module = "mamba3_rl", name = "EntityEncoderConfig", from_py_object)]
#[derive(Clone)]
pub struct PyEntityEncoderConfig {
    pub(crate) inner: EntityEncoderConfig,
}

#[pymethods]
impl PyEntityEncoderConfig {
    #[new]
    #[pyo3(signature = (*, hidden = vec![64], d_entity = 64, slot_embedding = false))]
    fn new(hidden: Vec<usize>, d_entity: usize, slot_embedding: bool) -> PyResult<Self> {
        let inner = EntityEncoderConfig::new(hidden, d_entity).with_slot_embedding(slot_embedding);
        inner.validate().py()?;
        Ok(Self { inner })
    }

    /// Hidden widths of the shared MLP.
    #[getter]
    fn hidden(&self) -> Vec<usize> {
        self.inner.hidden.clone()
    }

    /// Width of one entity's embedding.
    #[getter]
    fn d_entity(&self) -> usize {
        self.inner.d_entity
    }

    /// Whether a learned per-slot embedding is added.
    #[getter]
    fn slot_embedding(&self) -> bool {
        self.inner.slot_embedding
    }

    fn __repr__(&self) -> String {
        format!(
            "EntityEncoderConfig(hidden={:?}, d_entity={}, slot_embedding={})",
            self.inner.hidden,
            self.inner.d_entity,
            if self.inner.slot_embedding { "True" } else { "False" }
        )
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

/// How each entity set is summarised for the recurrent backbone: `"mean"`
/// and/or `"max"` over the present entities, concatenated in order.
#[pyclass(module = "mamba3_rl", name = "PoolingConfig", from_py_object)]
#[derive(Clone)]
pub struct PyPoolingConfig {
    pub(crate) inner: PoolingConfig,
}

#[pymethods]
impl PyPoolingConfig {
    #[new]
    #[pyo3(signature = (*, kinds = vec!["mean".to_string(), "max".to_string()]))]
    fn new(kinds: Vec<String>) -> PyResult<Self> {
        let inner = PoolingConfig {
            kinds: kinds
                .iter()
                .map(|k| PoolKind::parse(k))
                .collect::<mamba3::error::Result<Vec<_>>>()
                .py()?,
        };
        inner.validate().py()?;
        Ok(Self { inner })
    }

    /// The pools, in order.
    #[getter]
    fn kinds(&self) -> Vec<&'static str> {
        self.inner.kinds.iter().map(|k| k.name()).collect()
    }

    fn __repr__(&self) -> String {
        format!("PoolingConfig(kinds={:?})", self.kinds())
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

/// An actor that scores the entities of one set: the action id is the entity
/// index, then `extra_actions` flat actions after them.
///
/// `scoring="additive"` is `v · relu(W_h h + b + W_e e_i)`; `"dot"` is
/// `(W_q h) · e_i`, cheaper but measured worse on scorer tasks. Empty slots get
/// probability exactly zero.
#[pyclass(module = "mamba3_rl", name = "PointerHead", from_py_object)]
#[derive(Clone)]
pub struct PyPointerHead {
    pub(crate) inner: PointerHeadConfig,
}

#[pymethods]
impl PyPointerHead {
    #[new]
    #[pyo3(signature = (set, *, hidden = 64, scoring = "additive", extra_actions = 0))]
    fn new(set: String, hidden: usize, scoring: &str, extra_actions: usize) -> PyResult<Self> {
        let scoring = Scoring::parse(scoring).py()?;
        if scoring == Scoring::Additive && hidden == 0 {
            return Err(PyValueError::new_err(
                "hidden must be positive for additive scoring",
            ));
        }
        Ok(Self {
            inner: PointerHeadConfig::new(set)
                .with_hidden(hidden)
                .with_scoring(scoring)
                .with_extra_actions(extra_actions),
        })
    }

    /// The entity set whose entities are the actions.
    #[getter]
    fn set(&self) -> String {
        self.inner.set.clone()
    }

    /// Width of the additive scorer's hidden layer.
    #[getter]
    fn hidden(&self) -> usize {
        self.inner.hidden
    }

    /// `"additive"` or `"dot"`.
    #[getter]
    fn scoring(&self) -> &'static str {
        self.inner.scoring.name()
    }

    /// Flat actions appended after the entity logits.
    #[getter]
    fn extra_actions(&self) -> usize {
        self.inner.extra_actions
    }

    fn __repr__(&self) -> String {
        format!(
            "PointerHead({:?}, hidden={}, scoring={:?}, extra_actions={})",
            self.inner.set,
            self.inner.hidden,
            self.inner.scoring.name(),
            self.inner.extra_actions
        )
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

fn json_err(e: serde_json::Error) -> PyErr {
    PyValueError::new_err(format!("could not serialise: {e}"))
}

/// A JSON value as the Python object `json.loads` would make of it.
pub(crate) fn json_to_py<'py>(
    py: Python<'py>,
    value: &serde_json::Value,
) -> PyResult<Bound<'py, PyAny>> {
    let text = serde_json::to_string(value).map_err(json_err)?;
    py.import("json")?.call_method1("loads", (text,))
}

/// A Python object as JSON, through `json.dumps`.
pub(crate) fn py_to_json(value: &Bound<'_, PyAny>) -> PyResult<serde_json::Value> {
    let text: String = value
        .py()
        .import("json")?
        .call_method1("dumps", (value,))?
        .extract()?;
    serde_json::from_str(&text).map_err(|e| PyValueError::new_err(format!("not JSON: {e}")))
}
