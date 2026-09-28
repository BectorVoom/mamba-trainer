//! PPO for the entity model (ENTITY_RL_PLAN.md R2).
//!
//! [`PyEntityPolicy`] is an [`EntityActorCritic`](mamba3::models::entity::EntityActorCritic)
//! with its PPO trainer: `act` samples a plan per query in one upload and one
//! device-to-host read, and `update` replays a caller-supplied rollout through
//! several shuffled minibatches and reads the statistics once. The rollout
//! itself (observations plus rewards) lives in NumPy; everything between the
//! upload and the single read stays on the device.

use std::collections::BTreeMap;

use mamba3::backend::Device;
use mamba3::models::entity::{
    EntityActorCritic, EntityBatch, EntityModel, EntityPpoBatch, EntityPpoStats, EntityPpoTask,
    HeadKind, HostArrays, StepSelection, entity_gae,
};
use mamba3::nn::Module;
use mamba3::rl::PpoConfig;
use mamba3::tensor::Tensor;
use mamba3::tensor::ops::index::{IdTensor, gather_rows, read_all};
use mamba3::train::{AdamW, AdamWConfig, Trainer, TrainerConfig};
use numpy::{PyArray1, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::entity_model::{PyEntityModel, PyEntityModelSpec, read_arrays, read_floats, read_ints};
use crate::err::IntoPyResult;
use crate::{E, R};

/// One draw stream step: every `act` and every `update` takes a distinct seed
/// off the constructor seed, so a policy is deterministic given its seed and
/// call order.
const GOLDEN: u64 = 0x9E3779B97F4A7C15;

/// Every pointer head and every categorical head, in spec order: the actions
/// `act` samples and `update` scores. Multilabel and regression heads are not
/// actions.
fn action_heads(
    spec: &mamba3::models::entity::EntityModelSpec,
) -> Vec<(String, StepSelection)> {
    spec.heads
        .iter()
        .filter_map(|h| match &h.kind {
            HeadKind::Pointer { .. } | HeadKind::Categorical { .. } => {
                Some((h.name.clone(), h.steps))
            }
            HeadKind::MultiLabel { .. } | HeadKind::Regression { .. } => None,
        })
        .collect()
}

/// Fisher-Yates with a xorshift64 stream (the same shuffle the Rust bandit
/// test uses), so minibatch order is reproducible.
fn shuffled(n: usize, seed: u64) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..n).collect();
    let mut s = seed.max(1);
    for i in (1..n).rev() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        idx.swap(i, (s % (i as u64 + 1)) as usize);
    }
    idx
}

/// A float32 array of `dims` from row-major host data.
fn reshape_f32<'py>(
    py: Python<'py>,
    data: Vec<f32>,
    dims: &[usize],
) -> PyResult<Bound<'py, PyAny>> {
    let flat = PyArray1::from_vec(py, data);
    match dims {
        [a] => Ok(flat.reshape((*a,))?.into_any()),
        [a, b] => Ok(flat.reshape((*a, *b))?.into_any()),
        [a, b, c] => Ok(flat.reshape((*a, *b, *c))?.into_any()),
        [a, b, c, d] => Ok(flat.reshape((*a, *b, *c, *d))?.into_any()),
        _ => Err(PyValueError::new_err(format!(
            "entity logits have unexpected shape {dims:?}"
        ))),
    }
}

/// An int64 array of `dims` from row-major host data.
fn reshape_int<'py>(
    py: Python<'py>,
    data: Vec<i64>,
    dims: &[usize],
) -> PyResult<Bound<'py, PyAny>> {
    let flat = PyArray1::from_vec(py, data);
    match dims {
        [a] => Ok(flat.reshape((*a,))?.into_any()),
        [a, b] => Ok(flat.reshape((*a, *b))?.into_any()),
        [a, b, c] => Ok(flat.reshape((*a, *b, *c))?.into_any()),
        [a, b, c, d] => Ok(flat.reshape((*a, *b, *c, *d))?.into_any()),
        _ => Err(PyValueError::new_err(format!(
            "entity actions have unexpected shape {dims:?}"
        ))),
    }
}

/// Rows `ids` of every host array, in order: the minibatch's observations and
/// action labels, still on the host. Every entry's leading axis must be the
/// full rollout length `s`.
fn slice_host(full: &HostArrays, ids: &[usize], s: usize) -> PyResult<HostArrays> {
    fn gather<T: Clone>(
        key: &str,
        shape: &[usize],
        data: &[T],
        ids: &[usize],
        s: usize,
    ) -> PyResult<(Vec<usize>, Vec<T>)> {
        if shape.is_empty() || shape[0] != s {
            return Err(PyValueError::new_err(format!(
                "rollout key {key:?} has shape {shape:?}, expected leading axis S={s}"
            )));
        }
        let row: usize = shape[1..].iter().product();
        let mut out = Vec::with_capacity(ids.len() * row);
        for &i in ids {
            out.extend_from_slice(&data[i * row..(i + 1) * row]);
        }
        let mut dims = vec![ids.len()];
        dims.extend_from_slice(&shape[1..]);
        Ok((dims, out))
    }
    let mut out = HostArrays::new();
    for (key, (shape, data)) in &full.f32s {
        let (dims, rows) = gather(key, shape, data, ids, s)?;
        out.insert_f32(key, dims, rows);
    }
    for (key, (shape, data)) in &full.ints {
        let (dims, rows) = gather(key, shape, data, ids, s)?;
        out.insert_int(key, dims, rows);
    }
    Ok(out)
}

/// The actor-critic as a PPO policy: sampling (`act`), scoring (`value`) and
/// the PPO `update` over caller-supplied rollouts.
#[pyclass(module = "mamba3_rl", name = "EntityPolicy", unsendable)]
pub struct PyEntityPolicy {
    ac: EntityActorCritic<R, E>,
    spec: mamba3::models::entity::EntityModelSpec,
    device: Device<R>,
    trainer: Trainer<R, E, AdamW<R, E>>,
    config: PpoConfig,
    temperature: f32,
    seed_base: u64,
    acts: u64,
    updates: u64,
}

#[pymethods]
impl PyEntityPolicy {
    /// Build a fresh actor-critic from `spec` with an AdamW trainer and PPO
    /// hyperparameters. `temperature` tempers sampling (`act`) and scoring
    /// (`update`); `seed` starts the sampling and shuffling streams.
    #[new]
    #[pyo3(signature = (spec, *, learning_rate = 3e-4, weight_decay = 0.0, max_grad_norm = 0.5,
                        gamma = 0.99, lam = 0.95, clip = 0.2, value_coef = 0.5,
                        entropy_coef = 0.01, temperature = 1.0, seed = 0))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        spec: &PyEntityModelSpec,
        learning_rate: f32,
        weight_decay: f32,
        max_grad_norm: f32,
        gamma: f32,
        lam: f32,
        clip: f32,
        value_coef: f32,
        entropy_coef: f32,
        temperature: f32,
        seed: i64,
    ) -> PyResult<Self> {
        let device = Device::<R>::default();
        let ac = EntityActorCritic::init(&spec.inner, &device).py()?;
        Self::with_ac(
            ac,
            learning_rate,
            weight_decay,
            max_grad_norm,
            gamma,
            lam,
            clip,
            value_coef,
            entropy_coef,
            temperature,
            seed,
        )
    }

    /// Start from a behaviour-cloned [`PyEntityModel`]: its weights become the
    /// actor bit for bit (greedy predictions agree), with a fresh value head
    /// seeded off the spec seed. Training hyperparameters as in [`Self::new`].
    #[staticmethod]
    #[pyo3(signature = (model, *, learning_rate = 3e-4, weight_decay = 0.0, max_grad_norm = 0.5,
                        gamma = 0.99, lam = 0.95, clip = 0.2, value_coef = 0.5,
                        entropy_coef = 0.01, temperature = 1.0, seed = 0))]
    #[allow(clippy::too_many_arguments)]
    fn from_model(
        model: &PyEntityModel,
        learning_rate: f32,
        weight_decay: f32,
        max_grad_norm: f32,
        gamma: f32,
        lam: f32,
        clip: f32,
        value_coef: f32,
        entropy_coef: f32,
        temperature: f32,
        seed: i64,
    ) -> PyResult<Self> {
        let src = model.inner();
        let spec = src.spec().clone();
        let device = Device::<R>::default();
        let owned = EntityModel::init(&spec, &device).py()?;
        owned.load_state_dict(&src.state_dict(), true).py()?;
        let ac = EntityActorCritic::from_model(owned).py()?;
        Self::with_ac(
            ac,
            learning_rate,
            weight_decay,
            max_grad_norm,
            gamma,
            lam,
            clip,
            value_coef,
            entropy_coef,
            temperature,
            seed,
        )
    }

    /// Sample a plan for every query: `{"actions": {head: int64 [B, M, K]}`
    /// (`-1` = no action), `"log_prob": float32 [B, M, K]`, `"value": float32
    /// [B]`, `"outputs": {head: float32 logits}}` shaped as
    /// [`PyEntityModel::predict`] returns them. One upload, one
    /// device-to-host read. `greedy=True` takes the argmax instead of
    /// sampling (deterministic, seed-independent).
    #[pyo3(signature = (obs, *, greedy = false))]
    fn act<'py>(
        &mut self,
        py: Python<'py>,
        obs: &Bound<'py, PyDict>,
        greedy: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let host = read_arrays(&self.spec, obs)?;
        // No labels: the fused-loss tables stay empty and the upload reads
        // nothing back, so the read below is the call's only one.
        let batch = EntityBatch::from_host_no_seg(&self.spec, &host, &self.device).py()?;
        let temperature = if greedy { 0.0 } else { self.temperature };
        let seed = self.next_act_seed();
        let acted = self.ac.act(&batch, temperature, seed).py()?;
        let out = acted.read().py()?;
        let (m, k) = match &self.spec.queries {
            Some(q) => (q.count, q.steps),
            None => {
                return Err(PyValueError::new_err(
                    "entity act needs a query set; the spec has none",
                ));
            }
        };
        let b = out.value.len();
        for (name, ids) in &out.actions {
            if ids.len() != b * m * k {
                return Err(PyValueError::new_err(format!(
                    "entity act head {name:?} holds {} ids, expected [B={b}, M={m}, K={k}]",
                    ids.len()
                )));
            }
        }
        let result = PyDict::new(py);
        let actions = PyDict::new(py);
        for (name, ids) in &out.actions {
            actions.set_item(name, reshape_int(py, ids.clone(), &[b, m, k])?)?;
        }
        result.set_item("actions", actions)?;
        result.set_item("log_prob", reshape_f32(py, out.cell_log_prob, &[b, m, k])?)?;
        result.set_item("value", PyArray1::from_vec(py, out.value))?;
        let outputs = PyDict::new(py);
        for (name, (dims, data)) in &out.outputs {
            outputs.set_item(name, reshape_f32(py, data.clone(), dims)?)?;
        }
        result.set_item("outputs", outputs)?;
        Ok(result)
    }

    /// Run PPO over a rollout: `obs` holds `S = T*E` samples in time-major
    /// order (sample `t*E + e`); `actions` maps every action head (each
    /// pointer and categorical head) to int `[S, M, K]` (`-1` = no action;
    /// `First` heads also accept `[S, M]`); `log_prob` is float `[S, M, K]`;
    /// `value` is float `[S]` or `[T, E]`; `reward` and `done` are float
    /// `[T, E]`; `last_value` is float `[E]`. Returns `{"policy_loss",
    /// "value_loss", "entropy", "approx_kl", "clip_fraction", "grad_norm"}`.
    ///
    /// The rollout is uploaded once (auxiliaries) plus one read-free upload
    /// per minibatch (observations and labels, gathered on the host); GAE,
    /// advantage normalisation and minibatch gathering run on the device, and
    /// the statistics are read exactly once.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (obs, actions, log_prob, value, reward, done, last_value, *,
                        epochs = 4, minibatches = 4))]
    fn update<'py>(
        &mut self,
        py: Python<'py>,
        obs: &Bound<'py, PyDict>,
        actions: &Bound<'py, PyDict>,
        log_prob: &Bound<'py, PyAny>,
        value: &Bound<'py, PyAny>,
        reward: &Bound<'py, PyAny>,
        done: &Bound<'py, PyAny>,
        last_value: &Bound<'py, PyAny>,
        epochs: i64,
        minibatches: i64,
    ) -> PyResult<Bound<'py, PyDict>> {
        if epochs < 1 {
            return Err(PyValueError::new_err(format!(
                "epochs must be positive, got {epochs}"
            )));
        }
        if minibatches < 1 {
            return Err(PyValueError::new_err(format!(
                "minibatches must be positive, got {minibatches}"
            )));
        }
        let (m, k) = match &self.spec.queries {
            Some(q) => (q.count, q.steps),
            None => {
                return Err(PyValueError::new_err(
                    "entity update needs a query set; the spec has none",
                ));
            }
        };
        // Time comes from reward/done: both are [T, E].
        let (reward_shape, reward_data) = read_floats(reward, "reward")?;
        let (done_shape, done_data) = read_floats(done, "done")?;
        if reward_shape.len() != 2 || done_shape.len() != 2 {
            return Err(PyValueError::new_err(format!(
                "reward and done must be [T, E], got reward {reward_shape:?} and done {done_shape:?}"
            )));
        }
        if reward_shape != done_shape {
            return Err(PyValueError::new_err(format!(
                "reward has shape {reward_shape:?} but done has shape {done_shape:?}; \
                 both must be [T, E]"
            )));
        }
        let (t, e) = (reward_shape[0], reward_shape[1]);
        if t == 0 || e == 0 {
            return Err(PyValueError::new_err(format!(
                "reward has shape [T, E] = [{t}, {e}] with an empty axis"
            )));
        }
        let s = t * e;
        // Observations: every array's leading axis is S.
        let mut full = read_arrays(&self.spec, obs)?;
        for (key, (shape, _)) in full.f32s.iter() {
            if shape.is_empty() || shape[0] != s {
                return Err(PyValueError::new_err(format!(
                    "obs[{key:?}] has shape {shape:?}, expected leading axis S={s} (T*E)"
                )));
            }
        }
        for (key, (shape, _)) in full.ints.iter() {
            if shape.is_empty() || shape[0] != s {
                return Err(PyValueError::new_err(format!(
                    "obs[{key:?}] has shape {shape:?}, expected leading axis S={s} (T*E)"
                )));
            }
        }
        // Actions: exactly the action heads, as label.<head> arrays.
        let heads = action_heads(&self.spec);
        let mut seen_actions = BTreeMap::new();
        for (key, value) in actions.iter() {
            let key: String = key.extract()?;
            let head = heads.iter().find(|(name, _)| name == &key);
            let Some((_, steps)) = head else {
                if self.spec.head(&key).is_some() {
                    return Err(PyValueError::new_err(format!(
                        "actions[{key:?}] is not an action head: only pointer and \
                         categorical heads act (multilabel and regression heads do not)"
                    )));
                }
                return Err(PyValueError::new_err(format!(
                    "actions holds unknown head {key:?}; action heads are {:?}",
                    heads.iter().map(|(n, _)| n).collect::<Vec<_>>()
                )));
            };
            let (shape, data) = read_ints(&value, &format!("actions[{key:?}]"))?;
            let label: Vec<i64> = match steps {
                StepSelection::All => {
                    if shape != [s, m, k] {
                        return Err(PyValueError::new_err(format!(
                            "actions[{key:?}] has shape {shape:?}, expected [S={s}, M={m}, K={k}]"
                        )));
                    }
                    data
                }
                StepSelection::First => {
                    if shape == [s, m] {
                        data
                    } else if shape == [s, m, k] {
                        data.chunks(m * k)
                            .flat_map(|sample| sample.chunks(k).map(|cell| cell[0]))
                            .collect()
                    } else {
                        return Err(PyValueError::new_err(format!(
                            "actions[{key:?}] has shape {shape:?}, expected [S={s}, M={m}] \
                             or [S={s}, M={m}, K={k}] for a first-step head"
                        )));
                    }
                }
            };
            let dims = match steps {
                StepSelection::All => vec![s, m, k],
                StepSelection::First => vec![s, m],
            };
            full.insert_int(&format!("label.{key}"), dims, label);
            seen_actions.insert(key.clone(), ());
        }
        for (name, _) in &heads {
            if !seen_actions.contains_key(name) {
                return Err(PyValueError::new_err(format!(
                    "update needs actions for head {name:?} (every pointer and \
                     categorical head acts)"
                )));
            }
        }
        // Behaviour statistics.
        let (lp_shape, lp_data) = read_floats(log_prob, "log_prob")?;
        if lp_shape != [s, m, k] {
            return Err(PyValueError::new_err(format!(
                "log_prob has shape {lp_shape:?}, expected [S={s}, M={m}, K={k}]"
            )));
        }
        let (v_shape, v_data) = read_floats(value, "value")?;
        if v_shape != [s] && v_shape != [t, e] {
            return Err(PyValueError::new_err(format!(
                "value has shape {v_shape:?}, expected [S={s}] or [T={t}, E={e}]"
            )));
        }
        let (lv_shape, lv_data) = read_floats(last_value, "last_value")?;
        if lv_shape != [e] {
            return Err(PyValueError::new_err(format!(
                "last_value has shape {lv_shape:?}, expected [E={e}]"
            )));
        }
        let mb = minibatches as usize;
        if s % mb != 0 {
            return Err(PyValueError::new_err(format!(
                "minibatches must divide the {s} rollout samples (T*E), got {minibatches}"
            )));
        }
        // GAE on the device: rewards, values and dones upload, advantages and
        // returns never leave the device until the final statistics read.
        let (full_adv, full_ret) = entity_gae::<R, E>(
            &reward_data,
            &v_data,
            &done_data,
            &lv_data,
            t,
            e,
            self.config.gamma,
            self.config.lambda,
            &self.device,
        )
        .py()?;
        let up = |v: &[f32], shape: Vec<usize>| -> PyResult<Tensor<R, E>> {
            Tensor::from_f32(v, shape, &self.device).py()
        };
        let full_lp = up(&lp_data, vec![s, m, k])?;
        let full_mask = {
            // 1 where any action head acted: -1 everywhere means no action.
            // Built on the host from the validated labels (the device loop's
            // own mask agrees cell for cell; absent queries act no head).
            let mut mask = vec![0.0f32; s * m * k];
            for (name, steps) in &heads {
                let (shape, data) = full
                    .ints
                    .get(&format!("label.{name}"))
                    .expect("every action head has labels");
                match steps {
                    StepSelection::All => {
                        debug_assert_eq!(shape, &vec![s, m, k]);
                        for (cell, &id) in data.iter().enumerate() {
                            if id != -1 {
                                mask[cell] = 1.0;
                            }
                        }
                    }
                    StepSelection::First => {
                        debug_assert_eq!(shape, &vec![s, m]);
                        for (cell, &id) in data.iter().enumerate() {
                            if id != -1 {
                                mask[cell * k] = 1.0;
                            }
                        }
                    }
                }
            }
            up(&mask, vec![s, m, k])?
        };
        let full_val = up(&v_data, vec![s])?;
        // Shuffling streams off the constructor seed; drawn before the task
        // borrows the actor-critic.
        let mut orders = Vec::with_capacity(epochs as usize);
        for _ in 0..epochs {
            orders.push(shuffled(s, self.next_update_seed()));
        }
        let task = EntityPpoTask::new(&self.ac, self.config, self.temperature);
        let per = s / mb;
        let mut queued = Vec::with_capacity(epochs as usize * mb);
        for order in &orders {
            for chunk in order.chunks(per) {
                let mb_host = slice_host(&full, chunk, s)?;
                let batch =
                    EntityBatch::from_host_no_seg(&self.spec, &mb_host, &self.device).py()?;
                let b = chunk.len();
                let ids: Vec<u32> = chunk.iter().map(|&i| i as u32).collect();
                let id_buf = IdTensor::from_slice(&ids, vec![b], &self.device).py()?;
                let gather_3 = |tensor: &Tensor<R, E>| -> PyResult<Tensor<R, E>> {
                    gather_rows(&tensor.reshape(vec![s, m * k]).py()?, &id_buf)
                        .py()?
                        .reshape(vec![b, m, k])
                        .py()
                };
                let gather_1 = |tensor: &Tensor<R, E>| -> PyResult<Tensor<R, E>> {
                    gather_rows(&tensor.reshape(vec![s, 1]).py()?, &id_buf)
                        .py()?
                        .reshape(vec![b])
                        .py()
                };
                let ppo = EntityPpoBatch {
                    batch,
                    old_log_prob: gather_3(&full_lp)?,
                    cell_mask: gather_3(&full_mask)?,
                    advantages: gather_1(&full_adv)?,
                    returns: gather_1(&full_ret)?,
                    old_values: gather_1(&full_val)?,
                };
                queued.push(
                    self.trainer
                        .queue_step(&task, std::slice::from_ref(&ppo))
                        .py()?,
                );
            }
        }
        // The update's only read: every queued step's loss and gradient-norm
        // scale plus the last loss's diagnostics, under one synchronisation.
        let diagnostics = task.stat_tensors().ok_or_else(|| {
            PyValueError::new_err("entity update queued no steps".to_string())
        })?;
        let mut scalars: Vec<&Tensor<R, E>> =
            queued.iter().flat_map(|q| q.scalars()).collect();
        let n_step = scalars.len();
        scalars.extend(diagnostics.each_ref());
        let (_, values) = read_all(&[], &scalars).py()?;
        let step_values: Vec<f32> = values[..n_step].iter().map(|v| v[0]).collect();
        let stat_values: [f32; 5] = values[n_step..]
            .iter()
            .map(|v| v[0])
            .collect::<Vec<_>>()
            .try_into()
            .map_err(|_| {
                PyValueError::new_err("entity update read back an incomplete statistic".to_string())
            })?;
        let infos = self.trainer.report_steps(&queued, &step_values);
        let stats = EntityPpoStats::from_values(stat_values);
        let grad_norm =
            infos.iter().map(|i| i.grad_norm).sum::<f32>() / infos.len().max(1) as f32;
        let out = PyDict::new(py);
        out.set_item("policy_loss", stats.policy_loss)?;
        out.set_item("value_loss", stats.value_loss)?;
        out.set_item("entropy", stats.entropy)?;
        out.set_item("approx_kl", stats.approx_kl)?;
        out.set_item("clip_fraction", stats.clip_fraction)?;
        out.set_item("grad_norm", grad_norm)?;
        Ok(out)
    }

    /// The critic's values for `obs`: float32 `[B]`. One upload, one read.
    fn value<'py>(&self, py: Python<'py>, obs: &Bound<'py, PyDict>) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let host = read_arrays(&self.spec, obs)?;
        let batch = EntityBatch::from_host_no_seg(&self.spec, &host, &self.device).py()?;
        let values = self.ac.values(&batch).py()?;
        let (_, back) = read_all(&[], &[&values]).py()?;
        Ok(PyArray1::from_vec(
            py,
            back.into_iter().next().unwrap_or_default(),
        ))
    }

    /// Write the actor-critic weights and the spec to a checkpoint. Only the
    /// weights travel: the trainer and the sampling streams restart from the
    /// constructor defaults on [`Self::load`].
    #[pyo3(signature = (path, step = 0))]
    fn save(&self, path: &str, step: u64) -> PyResult<()> {
        self.ac.save(path, step).py()
    }

    /// Read back what [`Self::save`] wrote, with a fresh default trainer.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let device = Device::<R>::default();
        let ac = EntityActorCritic::load(path, &device).py()?;
        Self::with_defaults(ac)
    }

    /// The actor's weights as a supervised [`PyEntityModel`] (predict,
    /// evaluate, fine-tune): the same parameters, copied exactly.
    fn to_model(&self) -> PyResult<PyEntityModel> {
        let dict = self.ac.model().state_dict();
        if dict.entries.is_empty() {
            return Err(PyValueError::new_err(
                "entity policy holds no actor weights to export".to_string(),
            ));
        }
        let model = EntityModel::init(&self.spec, &self.device).py()?;
        model.load_state_dict(&dict, true).py()?;
        PyEntityModel::wrap(model)
    }
}

impl PyEntityPolicy {
    /// Validate the hyperparameters, build the trainer and the PPO config.
    #[allow(clippy::too_many_arguments)]
    fn with_ac(
        ac: EntityActorCritic<R, E>,
        learning_rate: f32,
        weight_decay: f32,
        max_grad_norm: f32,
        gamma: f32,
        lam: f32,
        clip: f32,
        value_coef: f32,
        entropy_coef: f32,
        temperature: f32,
        seed: i64,
    ) -> PyResult<Self> {
        if !learning_rate.is_finite() || learning_rate <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "learning_rate must be positive, got {learning_rate}"
            )));
        }
        if !weight_decay.is_finite() || weight_decay < 0.0 {
            return Err(PyValueError::new_err(format!(
                "weight_decay must not be negative, got {weight_decay}"
            )));
        }
        if !max_grad_norm.is_finite() || max_grad_norm < 0.0 {
            return Err(PyValueError::new_err(format!(
                "max_grad_norm must not be negative, got {max_grad_norm}"
            )));
        }
        if !value_coef.is_finite() {
            return Err(PyValueError::new_err(format!(
                "value_coef must be finite, got {value_coef}"
            )));
        }
        if !entropy_coef.is_finite() {
            return Err(PyValueError::new_err(format!(
                "entropy_coef must be finite, got {entropy_coef}"
            )));
        }
        if !temperature.is_finite() || temperature < 0.0 {
            return Err(PyValueError::new_err(format!(
                "temperature must not be negative, got {temperature}"
            )));
        }
        if seed < 0 {
            return Err(PyValueError::new_err(format!(
                "seed must not be negative, got {seed}"
            )));
        }
        let config = PpoConfig {
            gamma,
            lambda: lam,
            clip_coeff: clip,
            value_coeff: value_coef,
            entropy_coeff: entropy_coef,
            clip_value_loss: true,
            normalize_advantages: true,
            reference_coeff: 0.0,
        };
        config.validate().py()?;
        let trainer = Trainer::new(
            TrainerConfig::builder()
                .learning_rate(learning_rate)
                .max_grad_norm(max_grad_norm)
                .build()
                .py()?,
            AdamWConfig::builder()
                .learning_rate(learning_rate)
                .weight_decay(weight_decay)
                .build()
                .init::<R, E>(),
        );
        let seed_base = seed as u64;
        Ok(Self {
            spec: ac.model().spec().clone(),
            device: Device::<R>::default(),
            ac,
            trainer,
            config,
            temperature,
            seed_base,
            acts: 0,
            updates: 0,
        })
    }

    /// Fresh default hyperparameters around loaded weights.
    fn with_defaults(ac: EntityActorCritic<R, E>) -> PyResult<Self> {
        Self::with_ac(ac, 3e-4, 0.0, 0.5, 0.99, 0.95, 0.2, 0.5, 0.01, 1.0, 0)
    }

    /// The next sampling seed, off the constructor seed.
    fn next_act_seed(&mut self) -> u64 {
        let seed = self.seed_base.wrapping_add(self.acts.wrapping_mul(GOLDEN));
        self.acts += 1;
        seed
    }

    /// The next shuffling seed, off the constructor seed.
    fn next_update_seed(&mut self) -> u64 {
        let seed = self
            .seed_base
            .wrapping_add(self.updates.wrapping_mul(GOLDEN));
        self.updates += 1;
        seed
    }
}

/// Register the RL policy class.
pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyEntityPolicy>()?;
    Ok(())
}
