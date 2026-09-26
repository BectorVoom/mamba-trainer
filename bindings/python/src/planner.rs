//! The task planner: next K work visits per unit (TASK_PLANNER_PLAN.md T6).
//!
//! [`PyTaskPlanner`] is the Python handle onto a [`TaskPlanner`]: a host-batch
//! training entry point for tests and the oracle (`queue_train_step_host`), a
//! no-grad `predict`/`predict_aux` pair for evaluation and live inference, and
//! `save`/`load`. The device-resident `PlannerData` path (`queue_train_step`
//! on turn ids) arrives with K1 and plugs into the same `read_losses`.

use std::rc::Rc;

use mamba3::models::planner::{
    HostBatch, PlannerBatch, PlannerTask, TaskPlanner, TaskPlannerConfig,
};
use mamba3::tensor::Tensor;
use mamba3::tensor::ops::index::read_all;
use mamba3::train::{AdamW, AdamWConfig, QueuedStep, Trainer, TrainerConfig};
use numpy::{AllowTypeChange, PyArray1, PyArrayLikeDyn, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::{PyFloatingPointError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::config::PyLrSchedule;
use crate::err::IntoPyResult;
use crate::{E, R};

/// Any float array, whatever its rank or dtype.
type FloatArrayDyn<'py> = PyArrayLikeDyn<'py, f32, AllowTypeChange>;
/// Any integer array, whatever its rank or dtype.
type IntArrayDyn<'py> = PyArrayLikeDyn<'py, i64, AllowTypeChange>;

/// `(shape, values)` of a float array, with a message naming `what` if it is not one.
fn read_floats(value: &Bound<'_, PyAny>, what: &str) -> PyResult<(Vec<usize>, Vec<f32>)> {
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

/// `(shape, values)` of an integer array, with a message naming `what` if it is not one.
fn read_ints(value: &Bound<'_, PyAny>, what: &str) -> PyResult<(Vec<usize>, Vec<i64>)> {
    let array: IntArrayDyn<'_> = value.extract().map_err(|_| {
        PyValueError::new_err(format!("{what} must be an array of integers"))
    })?;
    let shape = array.shape().to_vec();
    let data = match array.as_slice() {
        Ok(slice) => slice.to_vec(),
        Err(_) => array.as_array().iter().copied().collect(),
    };
    Ok((shape, data))
}

/// The planner architecture: all-Mamba-3 task head over a tile grid.
#[pyclass(module = "mamba3_rl", name = "TaskPlannerConfig", from_py_object)]
#[derive(Clone)]
pub struct PyTaskPlannerConfig {
    pub(crate) inner: TaskPlannerConfig,
}

#[pymethods]
impl PyTaskPlannerConfig {
    /// Build the default architecture (`d_model=128`, 3 + 3 layers), overriding
    /// the mixer width, depths, axis alternation, chunk size and seed.
    #[new]
    #[pyo3(signature = (*, d_model = 128, n_tile_layers = 3, n_joint_layers = 3,
                        alternate_axes = true, chunk_size = 32, seed = 0))]
    fn new(
        d_model: usize,
        n_tile_layers: usize,
        n_joint_layers: usize,
        alternate_axes: bool,
        chunk_size: usize,
        seed: u64,
    ) -> PyResult<Self> {
        let mut inner = TaskPlannerConfig::default();
        inner.d_model = d_model;
        inner.n_tile_layers = n_tile_layers;
        inner.n_joint_layers = n_joint_layers;
        inner.alternate_axes = alternate_axes;
        inner.ssm.d_model = d_model;
        inner.ssm.chunk_size = chunk_size;
        inner.seed = seed;
        inner.validate().py()?;
        Ok(Self { inner })
    }

    /// Residual stream width.
    #[getter]
    fn d_model(&self) -> usize {
        self.inner.d_model
    }

    /// Tile-mixer layers.
    #[getter]
    fn n_tile_layers(&self) -> usize {
        self.inner.n_tile_layers
    }

    /// Joint-mixer layers.
    #[getter]
    fn n_joint_layers(&self) -> usize {
        self.inner.n_joint_layers
    }

    /// Tiles per turn (`grid * grid`).
    #[getter]
    fn tiles(&self) -> usize {
        self.inner.tiles()
    }

    /// Query tokens per turn (`max_units * k`).
    #[getter]
    fn queries(&self) -> usize {
        self.inner.queries()
    }

    fn __repr__(&self) -> String {
        format!(
            "TaskPlannerConfig(d_model={}, n_tile_layers={}, n_joint_layers={}, \
             alternate_axes={}, chunk_size={}, seed={})",
            self.inner.d_model,
            self.inner.n_tile_layers,
            self.inner.n_joint_layers,
            self.inner.alternate_axes,
            self.inner.ssm.chunk_size,
            self.inner.seed,
        )
    }
}

/// One host batch in the npz layout, validated before anything reaches the device.
#[allow(clippy::too_many_arguments)]
fn host_batch(
    cfg: &TaskPlannerConfig,
    tiles: &Bound<'_, PyAny>,
    glob: &Bound<'_, PyAny>,
    units: &Bound<'_, PyAny>,
    upos: &Bound<'_, PyAny>,
    tgt: Option<&Bound<'_, PyAny>>,
    op: Option<&Bound<'_, PyAny>>,
    crop: Option<&Bound<'_, PyAny>>,
    opset: Option<&Bound<'_, PyAny>>,
    eta: Option<&Bound<'_, PyAny>>,
) -> PyResult<HostBatch> {
    let (n, g, uu, k) = (cfg.tiles(), cfg.c_glob, cfg.max_units, cfg.k);
    let (ct, cu, no) = (cfg.c_tile, cfg.c_unit, cfg.n_ops);
    let q = cfg.queries();
    let (shape, tiles) = read_floats(tiles, "tiles")?;
    if shape.len() != 3 {
        return Err(PyValueError::new_err(format!(
            "tiles must be [turns, 100, 48], got shape {shape:?}"
        )));
    }
    let b = shape[0];
    if shape != [b, n, ct] {
        return Err(PyValueError::new_err(format!(
            "tiles must be [turns, {n}, {ct}], got {shape:?}"
        )));
    }
    let (shape, glob) = read_floats(glob, "glob")?;
    if shape != [b, g] {
        return Err(PyValueError::new_err(format!(
            "glob must be [turns, {g}], got {shape:?}"
        )));
    }
    let (shape, units) = read_floats(units, "units")?;
    if shape != [b, uu, cu] {
        return Err(PyValueError::new_err(format!(
            "units must be [turns, {uu}, {cu}], got {shape:?}"
        )));
    }
    let (shape, upos) = read_ints(upos, "upos")?;
    if shape != [b, uu] {
        return Err(PyValueError::new_err(format!(
            "upos must be [turns, {uu}], got {shape:?}"
        )));
    }
    // Labels default to all-ignored (inference); training passes them all.
    let ints_or = |v: Option<&Bound<'_, PyAny>>, what: &str, want: &[usize], fill: i64| {
        PyResult::Ok(match v {
            Some(v) => {
                let (shape, data) = read_ints(v, what)?;
                if shape != want {
                    return Err(PyValueError::new_err(format!(
                        "{what} must be shaped {want:?}, got {shape:?}"
                    )));
                }
                data
            }
            None => vec![fill; want.iter().product()],
        })
    };
    let (qq, kk) = (q, k);
    let tgt = ints_or(tgt, "tgt", &[b, uu, kk], -100)?;
    let op = ints_or(op, "op", &[b, uu, kk], -100)?;
    let crop = ints_or(crop, "crop", &[b, uu, kk], -100)?;
    let opset = match opset {
        Some(v) => {
            let (shape, data) = read_ints(v, "opset")?;
            if shape != [b, uu, kk, no] {
                return Err(PyValueError::new_err(format!(
                    "opset must be shaped [{b}, {uu}, {kk}, {no}], got {shape:?}"
                )));
            }
            data.iter()
                .map(|&x| {
                    if x == 0 || x == 1 {
                        Ok(x as u8)
                    } else {
                        Err(PyValueError::new_err(format!(
                            "opset holds {x}, expected only 0 or 1"
                        )))
                    }
                })
                .collect::<PyResult<Vec<u8>>>()?
        }
        None => vec![0u8; b * qq * no],
    };
    let eta = ints_or(eta, "eta", &[b, uu], -1)?;
    let i32s = |xs: Vec<i64>, what: &str| {
        xs.into_iter()
            .map(|x| {
                i32::try_from(x).map_err(|_| {
                    PyValueError::new_err(format!("{what} holds {x}, outside int32 range"))
                })
            })
            .collect::<PyResult<Vec<i32>>>()
    };
    Ok(HostBatch {
        turns: b,
        tiles,
        glob,
        units,
        upos: i32s(upos, "upos")?,
        tgt: i32s(tgt, "tgt")?,
        op: i32s(op, "op")?,
        crop: i32s(crop, "crop")?,
        opset,
        eta: i32s(eta, "eta")?,
    })
}

/// Aux outputs at one target per query, reshaped to unit-major NumPy arrays.
fn aux_dict<'py>(
    py: Python<'py>,
    aux: &[f32],
    b: usize,
    u: usize,
    k: usize,
    no: usize,
    nc: usize,
    aw: usize,
    n_plus_1: usize,
    logits: Vec<f32>,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item(
        "target_logits",
        PyArray1::from_vec(py, logits).reshape((b, u, k, n_plus_1))?,
    )?;
    let q = u * k;
    let mut op = vec![0.0f32; b * q * no];
    let mut opset = vec![0.0f32; b * q * no];
    let mut crop = vec![0.0f32; b * q * nc];
    let mut eta = vec![0.0f32; b * u];
    for bi in 0..b {
        for qi in 0..q {
            let src = (bi * q + qi) * aw;
            op[(bi * q + qi) * no..(bi * q + qi + 1) * no]
                .copy_from_slice(&aux[src..src + no]);
            opset[(bi * q + qi) * no..(bi * q + qi + 1) * no]
                .copy_from_slice(&aux[src + no..src + 2 * no]);
            crop[(bi * q + qi) * nc..(bi * q + qi + 1) * nc]
                .copy_from_slice(&aux[src + 2 * no..src + 2 * no + nc]);
            if qi % k == 0 {
                eta[bi * u + qi / k] = aux[src + aw - 1];
            }
        }
    }
    out.set_item("op", PyArray1::from_vec(py, op).reshape((b, u, k, no))?)?;
    out.set_item(
        "opset",
        PyArray1::from_vec(py, opset).reshape((b, u, k, no))?,
    )?;
    out.set_item(
        "crop",
        PyArray1::from_vec(py, crop).reshape((b, u, k, nc))?,
    )?;
    out.set_item("eta", PyArray1::from_vec(py, eta).reshape((b, u))?)?;
    Ok(out)
}

/// The task planner: imitation-learned work-visit pointer over a tile grid.
#[pyclass(module = "mamba3_rl", name = "TaskPlanner", unsendable)]
pub struct PyTaskPlanner {
    inner: Rc<TaskPlanner<R, E>>,
    device: mamba3::backend::Device<R>,
    trainer: Trainer<R, E, AdamW<R, E>>,
    queued: Vec<(QueuedStep<R, E>, PlannerBatch<R, E>)>,
    loss_scale: f32,
}

#[pymethods]
impl PyTaskPlanner {
    /// Build the model and its trainer.
    ///
    /// `matmul_precision` is `"f32"` (the default) or `"f16"`; anything else —
    /// including `"bf16"`, which cannot be tested on this machine — raises
    /// `ValueError`. The precision is a process-global mode shared with every
    /// other object in the process. `loss_scale` (default `1.0`, `1024.0` with
    /// f16) multiplies the loss; the AdamW `eps` and the gradient-norm clip are
    /// scaled by the same factor, so the update is unchanged but for f16 rounding.
    #[new]
    #[pyo3(signature = (config, *, learning_rate = 3e-4, weight_decay = 0.05,
                        max_grad_norm = 1.0, lr_schedule = None,
                        matmul_precision = "f32", loss_scale = 1.0))]
    fn new(
        config: &PyTaskPlannerConfig,
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
        let inner = Rc::new(config.inner.init::<R, E>(&device).py()?);
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
            device,
            trainer,
            queued: Vec::new(),
            loss_scale,
        })
    }

    /// Queue one host-batch training step; nothing is read back. Prefer the
    /// device-resident `PlannerData` path (K1) for real training: this host
    /// batch exists for tests and the composed oracle.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (tiles, glob, units, upos, tgt, op, crop, opset, eta))]
    fn queue_train_step_host(
        &mut self,
        tiles: &Bound<'_, PyAny>,
        glob: &Bound<'_, PyAny>,
        units: &Bound<'_, PyAny>,
        upos: &Bound<'_, PyAny>,
        tgt: &Bound<'_, PyAny>,
        op: &Bound<'_, PyAny>,
        crop: &Bound<'_, PyAny>,
        opset: &Bound<'_, PyAny>,
        eta: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let cfg = self.inner.config().clone();
        let host = host_batch(
            &cfg,
            tiles,
            glob,
            units,
            upos,
            Some(tgt),
            Some(op),
            Some(crop),
            Some(opset),
            Some(eta),
        )?;
        let batch = PlannerBatch::from_host(&cfg, &host, &self.device).py()?;
        let task = PlannerTask::new(&self.inner).with_loss_scale(self.loss_scale);
        let step = self
            .trainer
            .queue_step(&task, std::slice::from_ref(&batch))
            .py()?;
        self.queued.push((step, batch));
        Ok(())
    }

    /// `(loss, grad_norm, [target, op, opset, crop, eta])` for every step queued
    /// since the last read, with the loss scale divided back out. One
    /// synchronisation for the whole backlog. A non-finite loss or gradient
    /// norm raises `FloatingPointError` naming the step: halve the loss scale
    /// and restart from the last epoch checkpoint.
    fn read_losses(&mut self) -> PyResult<Vec<(f32, f32, Vec<f32>)>> {
        let queued: Vec<(QueuedStep<R, E>, PlannerBatch<R, E>)> =
            std::mem::take(&mut self.queued);
        if queued.is_empty() {
            return Ok(Vec::new());
        }
        // Device work first, then one read for steps and components together.
        let (steps, batches): (Vec<QueuedStep<R, E>>, Vec<PlannerBatch<R, E>>) =
            queued.into_iter().unzip();
        let mut components: Vec<Tensor<R, E>> = Vec::with_capacity(batches.len() * 5);
        for batch in &batches {
            let task = PlannerTask::new(&self.inner).with_loss_scale(self.loss_scale);
            for part in task.component_losses(batch).py()? {
                components.push(part.into_tensor());
            }
        }
        let mut scalars: Vec<&Tensor<R, E>> =
            steps.iter().flat_map(|s| s.scalars()).collect();
        let n_step = scalars.len();
        scalars.extend(&components);
        let (_, values) = read_all(&[], &scalars).py()?;
        let values: Vec<f32> = values.iter().map(|v| v[0]).collect();
        let (step_values, component_values) = values.split_at(n_step);
        let infos = self.trainer.report_steps(&steps, step_values);
        let mut out = Vec::with_capacity(batches.len());
        for (i, info) in infos.iter().enumerate() {
            let loss = info.loss / self.loss_scale;
            let grad_norm = info.grad_norm / self.loss_scale;
            if !loss.is_finite() || !grad_norm.is_finite() {
                return Err(PyFloatingPointError::new_err(format!(
                    "planner step {} has a non-finite loss ({loss}) or gradient norm \
                     ({grad_norm}); halve the loss scale and restart from the last \
                     epoch checkpoint",
                    info.step,
                )));
            }
            out.push((
                loss,
                grad_norm,
                component_values[i * 5..(i + 1) * 5].to_vec(),
            ));
        }
        Ok(out)
    }

    /// No-grad prediction on host arrays: `target_logits [B,U,K,101]` and the
    /// aux heads at the predicted target (`op`/`opset [B,U,K,13]`, `crop
    /// [B,U,K,5]`, `eta [B,U]`).
    fn predict<'a>(
        &self,
        tiles: &Bound<'a, PyAny>,
        glob: &Bound<'_, PyAny>,
        units: &Bound<'_, PyAny>,
        upos: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'a, PyDict>> {
        let py = tiles.py();
        let cfg = self.inner.config().clone();
        let host = host_batch(&cfg, tiles, glob, units, upos, None, None, None, None, None)?;
        let b = host.turns;
        let batch = PlannerBatch::from_host(&cfg, &host, &self.device).py()?;
        let (logits, aux) = self.inner.predict(&batch).py()?;
        let (u, k, no, nc, aw) = (cfg.max_units, cfg.k, cfg.n_ops, cfg.n_crops, cfg.aux_width());
        aux_dict(
            py,
            &aux.try_to_f32().py()?,
            b,
            u,
            k,
            no,
            nc,
            aw,
            cfg.tiles() + 1,
            logits.try_to_f32().py()?,
        )
    }

    /// No-grad aux heads at the given targets (teacher forcing): same dict as
    /// [`Self::predict`] without `target_logits`.
    fn predict_aux<'a>(
        &self,
        tiles: &Bound<'a, PyAny>,
        glob: &Bound<'_, PyAny>,
        units: &Bound<'_, PyAny>,
        upos: &Bound<'_, PyAny>,
        tgt: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'a, PyDict>> {
        let py = tiles.py();
        let cfg = self.inner.config().clone();
        let (n, u, k) = (cfg.tiles(), cfg.max_units, cfg.k);
        let host = host_batch(
            &cfg,
            tiles,
            glob,
            units,
            upos,
            Some(tgt),
            None,
            None,
            None,
            None,
        )?;
        let b = host.turns;
        let batch = PlannerBatch::from_host(&cfg, &host, &self.device).py()?;
        // Teacher-forced target one-hot: NONE (100) is index N, -100 a zero row.
        let mut oh = vec![0.0f32; b * u * k * (n + 1)];
        for (f, &t) in host.tgt.iter().enumerate() {
            if t == -100 {
                continue;
            }
            if t < 0 || t as usize > n {
                return Err(PyValueError::new_err(format!(
                    "tgt holds {t}, expected -100, 0..{n} or 100 (NONE)"
                )));
            }
            oh[f * (n + 1) + t as usize] = 1.0;
        }
        let oh = Tensor::from_f32(&oh, vec![b, u * k, n + 1], &self.device).py()?;
        let aux = self.inner.predict_aux(&batch, &oh).py()?;
        let (no, nc, aw) = (cfg.n_ops, cfg.n_crops, cfg.aux_width());
        let out = aux_dict(
            py,
            &aux.try_to_f32().py()?,
            b,
            u,
            k,
            no,
            nc,
            aw,
            n + 1,
            vec![0.0; b * u * k * (n + 1)],
        )?;
        out.del_item("target_logits")?;
        Ok(out)
    }

    /// Write the weights and the architecture to a checkpoint.
    #[pyo3(signature = (path, step = 0))]
    fn save(&self, path: &str, step: u64) -> PyResult<()> {
        self.inner.save(path, step).py()
    }

    /// Read back what [`Self::save`] wrote.
    ///
    /// The trainer is fresh: only the weights travel, so pass the same
    /// `learning_rate` / `loss_scale` to the constructor when resuming.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let device = mamba3::backend::Device::<R>::default();
        let inner = Rc::new(TaskPlanner::load(path, &device).py()?);
        let trainer = Trainer::new(
            TrainerConfig::builder().build().py()?,
            AdamWConfig::builder().build().init::<R, E>(),
        );
        Ok(Self {
            inner,
            device,
            trainer,
            queued: Vec::new(),
            loss_scale: 1.0,
        })
    }
}
