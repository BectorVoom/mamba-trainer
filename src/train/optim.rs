//! Optimizers.
//!
//! Optimizers see a flat `&[Param]` and a [`Grads`] map keyed by [`ParamId`]. They
//! never touch the model tree, which is why the same optimizer works for a full
//! fine-tune, a LoRA-only run (pass only the adapter parameters) and a QAT run.

use std::collections::{BTreeMap, HashMap};

use cubecl::prelude::Runtime;

use crate::autograd::{Grads, ParamId};
use crate::backend::FloatElem;
use crate::error::{Error, Result};
use crate::nn::module::{StateDict, TensorData};
use crate::nn::param::Param;
use crate::tensor::Tensor;
use crate::tensor::ops::fused::{self, AdamWStep, adamw_step};
use crate::tensor::ops::{elemwise, reduce};

/// A saved tensor, checked against the shape it must restore into and copied
/// to `device`. Checking the element count here, rather than trusting
/// [`Tensor::from_f32`] to notice, gives a message that names the entry.
fn staged_tensor<R: Runtime, E: FloatElem>(
    key: &str,
    saved: &TensorData,
    want: &[usize],
    device: &crate::backend::Device<R>,
) -> Result<Tensor<R, E>> {
    if saved.shape != want {
        return Err(Error::StateDict(format!(
            "the optimizer state's {key} is shaped {:?}, but the parameter is {want:?}",
            saved.shape
        )));
    }
    let elements: usize = want.iter().product();
    if saved.data.len() != elements {
        return Err(Error::StateDict(format!(
            "the optimizer state's {key} holds {} values for a {want:?} parameter ({elements})",
            saved.data.len()
        )));
    }
    Tensor::from_f32(&saved.data, saved.shape.clone(), device)
}

/// Under `strict`, refuse any key in `state` that `known` does not claim: an
/// entry for a parameter this model does not have is a checkpoint from a
/// different model, not state to silently ignore.
fn refuse_unknown_keys(state: &StateDict, known: &std::collections::HashSet<String>) -> Result<()> {
    let unknown: Vec<&str> = state
        .entries
        .keys()
        .filter(|k| !known.contains(*k))
        .map(String::as_str)
        .collect();
    if unknown.is_empty() {
        Ok(())
    } else {
        Err(Error::StateDict(format!(
            "the optimizer state has entries for parameters this model does not have: {unknown:?}"
        )))
    }
}

/// Something that turns gradients into parameter updates.
pub trait Optimizer<R: Runtime, E: FloatElem> {
    /// Apply one update to every parameter that has a gradient.
    fn step(&mut self, params: &[Param<R, E>], grads: &Grads<R, E>) -> Result<()> {
        self.step_scaled(params, grads, None)
    }

    /// Apply one update, multiplying every gradient by a **device-side** scale.
    ///
    /// The scale is a one-element tensor rather than an `f32` because of where it
    /// comes from: micro-batch averaging times the global gradient-norm clip. The
    /// clip depends on a reduction over every gradient, so taking it as a number
    /// means reading that reduction back and stalling the step between the backward
    /// pass and the update. Taking it as a buffer keeps the whole step asynchronous,
    /// and costs the optimizer kernel one extra load of one float.
    fn step_scaled(
        &mut self,
        params: &[Param<R, E>],
        grads: &Grads<R, E>,
        scale: Option<&Tensor<R, E>>,
    ) -> Result<()>;

    /// Current learning rate.
    fn learning_rate(&self) -> f32;

    /// Override the learning rate, e.g. from a schedule.
    fn set_learning_rate(&mut self, lr: f32);

    /// Number of updates applied so far.
    fn step_count(&self) -> u64;

    /// Export the per-parameter state this optimizer keeps between steps —
    /// moment estimates, momentum — keyed by each parameter's stable path
    /// rather than its process-local [`ParamId`], which is not stable across
    /// runs and so cannot be what a saved checkpoint is keyed by.
    ///
    /// The step counter is deliberately *not* in here: a [`StateDict`] stores
    /// `f32`s, which cannot hold a count past `2^24` exactly. It travels beside
    /// the tensors as an integer — see [`Optimizer::step_count`] and
    /// [`crate::train::Checkpoint::optimizer_steps`].
    ///
    /// The default is an empty [`StateDict`], honest only for an optimizer
    /// that truly carries none (plain SGD with no momentum). An optimizer
    /// with real state must override this rather than let a caller believe a
    /// save succeeded when it silently kept nothing.
    fn state_dict(&self, params: &[(String, Param<R, E>)]) -> StateDict {
        let _ = params;
        StateDict::default()
    }

    /// Replace this optimizer's state with `state` and its step counter with
    /// `steps`, as saved by [`Optimizer::state_dict`] and
    /// [`Optimizer::step_count`].
    ///
    /// **A replacement, not a merge.** A parameter with no entry in `state`
    /// ends up with no state at all — exactly as it was when the checkpoint was
    /// written (a parameter that never received a gradient has none to save) —
    /// rather than keeping whatever this optimizer held before the call.
    ///
    /// **All or nothing.** Every entry is validated and staged before anything
    /// is replaced, so an error leaves the optimizer exactly as it was.
    ///
    /// `strict` additionally refuses entries for parameters not in `params`, a
    /// half-present set of per-parameter tensors, and a missing `steps`. Without
    /// it, unknown entries are ignored and a missing `steps` restarts the count
    /// at zero — a warm start the caller has asked for explicitly. Shape and
    /// size mismatches are errors either way.
    ///
    /// The default rejects a non-empty `state` under `strict` — restoring
    /// "successfully" into an optimizer that has nowhere to put the state
    /// would silently produce a resume that is not one — and does nothing
    /// otherwise.
    fn load_state_dict(
        &mut self,
        params: &[(String, Param<R, E>)],
        state: &StateDict,
        steps: Option<u64>,
        strict: bool,
    ) -> Result<()> {
        let _ = (params, steps);
        if strict && !state.entries.is_empty() {
            return Err(Error::Unsupported(
                "this optimizer keeps no state of its own and cannot restore \
                 the state a checkpoint carries; pass strict=false for a warm \
                 start that only restores the weights"
                    .to_string(),
            ));
        }
        Ok(())
    }
}

/// Configuration for [`AdamW`].
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AdamWConfig {
    /// Base learning rate.
    pub learning_rate: f32,
    /// First moment decay.
    pub beta1: f32,
    /// Second moment decay.
    pub beta2: f32,
    /// Denominator floor.
    pub eps: f32,
    /// Decoupled weight decay.
    pub weight_decay: f32,
    /// Skip weight decay on parameters with fewer than two dimensions (biases,
    /// gains, `A_log`, `dt_bias`). This is the usual convention and it matters for
    /// SSMs, where decaying `A_log` would quietly change the decay spectrum.
    pub decay_matrices_only: bool,
}

impl Default for AdamWConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3,
            beta1: 0.9,
            beta2: 0.95,
            eps: 1e-8,
            weight_decay: 0.1,
            decay_matrices_only: true,
        }
    }
}

impl AdamWConfig {
    /// Start a builder.
    pub fn builder() -> AdamWConfigBuilder {
        AdamWConfigBuilder::default()
    }

    /// Instantiate the optimizer.
    pub fn init<R: Runtime, E: FloatElem>(&self) -> AdamW<R, E> {
        AdamW {
            config: *self,
            lr: self.learning_rate,
            state: HashMap::new(),
            steps: 0,
        }
    }
}

/// Builder for [`AdamWConfig`].
#[derive(Debug, Clone, Default)]
pub struct AdamWConfigBuilder {
    config: Option<AdamWConfig>,
}

impl AdamWConfigBuilder {
    fn edit(mut self, f: impl FnOnce(&mut AdamWConfig)) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        f(&mut config);
        self.config = Some(config);
        self
    }

    /// Base learning rate.
    pub fn learning_rate(self, lr: f32) -> Self {
        self.edit(|c| c.learning_rate = lr)
    }

    /// Moment decay rates.
    pub fn betas(self, beta1: f32, beta2: f32) -> Self {
        self.edit(|c| {
            c.beta1 = beta1;
            c.beta2 = beta2;
        })
    }

    /// Denominator floor.
    pub fn eps(self, eps: f32) -> Self {
        self.edit(|c| c.eps = eps)
    }

    /// Decoupled weight decay.
    pub fn weight_decay(self, wd: f32) -> Self {
        self.edit(|c| c.weight_decay = wd)
    }

    /// Whether to skip decay on vectors and scalars.
    pub fn decay_matrices_only(self, flag: bool) -> Self {
        self.edit(|c| c.decay_matrices_only = flag)
    }

    /// Build the configuration.
    pub fn build(self) -> AdamWConfig {
        self.config.unwrap_or_default()
    }
}

/// Adam with decoupled weight decay.
pub struct AdamW<R: Runtime, E: FloatElem> {
    config: AdamWConfig,
    lr: f32,
    state: HashMap<ParamId, Moments<R, E>>,
    steps: u64,
}

struct Moments<R: Runtime, E: FloatElem> {
    m: Tensor<R, E>,
    v: Tensor<R, E>,
}

impl<R: Runtime, E: FloatElem> AdamW<R, E> {
    /// Build with default settings.
    pub fn new(learning_rate: f32) -> Self {
        AdamWConfig {
            learning_rate,
            ..Default::default()
        }
        .init()
    }

    /// The hyperparameters this optimizer was built with. The learning rate in
    /// here is the base rate; [`Optimizer::learning_rate`] is the one a schedule
    /// last set.
    pub fn config(&self) -> &AdamWConfig {
        &self.config
    }

    /// Number of parameters with optimizer state.
    pub fn tracked(&self) -> usize {
        self.state.len()
    }

    /// Drop all moment estimates.
    pub fn reset(&mut self) {
        self.state.clear();
        self.steps = 0;
    }
}

impl<R: Runtime, E: FloatElem> Optimizer<R, E> for AdamW<R, E> {
    fn step_scaled(
        &mut self,
        params: &[Param<R, E>],
        grads: &Grads<R, E>,
        scale: Option<&Tensor<R, E>>,
    ) -> Result<()> {
        self.steps += 1;
        let t = self.steps as f32;
        let bias1 = 1.0 - self.config.beta1.powf(t);
        let bias2 = 1.0 - self.config.beta2.powf(t);

        // A scale of exactly one is still a buffer, so the kernel has one code path.
        let unit_scale;
        let scale = match scale {
            Some(s) => s,
            None => {
                let Some((_, any)) = grads.iter().next() else {
                    return Ok(());
                };
                unit_scale = Tensor::full(vec![1], 1.0, any.device());
                &unit_scale
            }
        };

        for param in params {
            if !param.requires_grad() {
                continue;
            }
            let Some(grad) = grads.get(param.id()) else {
                continue;
            };
            let value = param.value();

            let entry = self.state.entry(param.id()).or_insert_with(|| Moments {
                m: Tensor::zeros(value.shape().clone(), value.device()),
                v: Tensor::zeros(value.shape().clone(), value.device()),
            });

            let decay = if self.config.decay_matrices_only && value.rank() < 2 {
                0.0
            } else {
                self.config.weight_decay
            };
            // Moments, bias correction, decay and the step in one launch: these
            // tensors are small and the chain is twelve ops long, so unfused this
            // is almost entirely dispatch overhead.
            let next = adamw_step(
                &value,
                grad,
                &entry.m,
                &entry.v,
                scale,
                AdamWStep {
                    lr: self.lr,
                    beta1: self.config.beta1,
                    beta2: self.config.beta2,
                    eps: self.config.eps,
                    decay,
                    bias1,
                    bias2,
                },
            );
            param.set(next);
        }
        Ok(())
    }

    fn learning_rate(&self) -> f32 {
        self.lr
    }

    fn set_learning_rate(&mut self, lr: f32) {
        self.lr = lr;
    }

    fn step_count(&self) -> u64 {
        self.steps
    }

    fn state_dict(&self, params: &[(String, Param<R, E>)]) -> StateDict {
        let mut entries = BTreeMap::new();
        for (name, param) in params {
            let Some(moments) = self.state.get(&param.id()) else {
                continue;
            };
            entries.insert(
                format!("{name}.m"),
                TensorData {
                    shape: moments.m.shape().dims().to_vec(),
                    data: moments.m.to_f32(),
                },
            );
            entries.insert(
                format!("{name}.v"),
                TensorData {
                    shape: moments.v.shape().dims().to_vec(),
                    data: moments.v.to_f32(),
                },
            );
        }
        StateDict { entries }
    }

    fn load_state_dict(
        &mut self,
        params: &[(String, Param<R, E>)],
        state: &StateDict,
        steps: Option<u64>,
        strict: bool,
    ) -> Result<()> {
        if strict {
            if steps.is_none() {
                return Err(Error::StateDict(
                    "the optimizer state has no step counter; bias correction \
                     could not resume at the right point"
                        .to_string(),
                ));
            }
            let known = params
                .iter()
                .flat_map(|(name, _)| [format!("{name}.m"), format!("{name}.v")])
                .collect();
            refuse_unknown_keys(state, &known)?;
        }
        let mut staged = HashMap::with_capacity(params.len());
        for (name, param) in params {
            let (m_key, v_key) = (format!("{name}.m"), format!("{name}.v"));
            match (state.entries.get(&m_key), state.entries.get(&v_key)) {
                (Some(m), Some(v)) => {
                    let want = param.shape().dims().to_vec();
                    let device = param.value().device().clone();
                    staged.insert(
                        param.id(),
                        Moments {
                            m: staged_tensor(&m_key, m, &want, &device)?,
                            v: staged_tensor(&v_key, v, &want, &device)?,
                        },
                    );
                }
                // Parameters which have never received a gradient have no Adam
                // moments to serialise. This is normal for, for example, the
                // critic in an imitation-only run. A missing pair is therefore
                // an empty state, while a *partial* pair below remains corrupt.
                (None, None) => {}
                _ => {
                    return Err(Error::StateDict(format!(
                        "the optimizer state for {name} has only one of its two moments"
                    )));
                }
            }
        }
        self.state = staged;
        self.steps = steps.unwrap_or(0);
        Ok(())
    }
}

/// Stochastic gradient descent with optional momentum.
pub struct Sgd<R: Runtime, E: FloatElem> {
    lr: f32,
    momentum: f32,
    weight_decay: f32,
    nesterov: bool,
    velocity: HashMap<ParamId, Tensor<R, E>>,
    steps: u64,
}

impl<R: Runtime, E: FloatElem> Sgd<R, E> {
    /// Plain SGD.
    pub fn new(learning_rate: f32) -> Self {
        Self {
            lr: learning_rate,
            momentum: 0.0,
            weight_decay: 0.0,
            nesterov: false,
            velocity: HashMap::new(),
            steps: 0,
        }
    }

    /// Add heavy-ball momentum.
    pub fn with_momentum(mut self, momentum: f32) -> Self {
        self.momentum = momentum;
        self
    }

    /// Add L2 weight decay.
    pub fn with_weight_decay(mut self, wd: f32) -> Self {
        self.weight_decay = wd;
        self
    }

    /// Use Nesterov momentum.
    pub fn with_nesterov(mut self, nesterov: bool) -> Self {
        self.nesterov = nesterov;
        self
    }
}

impl<R: Runtime, E: FloatElem> Optimizer<R, E> for Sgd<R, E> {
    fn step_scaled(
        &mut self,
        params: &[Param<R, E>],
        grads: &Grads<R, E>,
        scale: Option<&Tensor<R, E>>,
    ) -> Result<()> {
        self.steps += 1;
        for param in params {
            if !param.requires_grad() {
                continue;
            }
            let Some(grad) = grads.get(param.id()) else {
                continue;
            };
            let value = param.value();
            // No fused kernel here, so the device-side scale is a broadcast multiply.
            // Still worth it: one extra launch per parameter against a host
            // synchronisation for the whole step.
            let mut d = match scale {
                Some(s) => elemwise::mul(grad, s)?,
                None => grad.clone(),
            };
            if self.weight_decay != 0.0 {
                d = elemwise::add(&d, &elemwise::mul_scalar(&value, self.weight_decay))?;
            }
            if self.momentum != 0.0 {
                let velocity = match self.velocity.remove(&param.id()) {
                    Some(prev) => elemwise::add(&elemwise::mul_scalar(&prev, self.momentum), &d)?,
                    None => d.clone(),
                };
                d = if self.nesterov {
                    elemwise::add(&d, &elemwise::mul_scalar(&velocity, self.momentum))?
                } else {
                    velocity.clone()
                };
                self.velocity.insert(param.id(), velocity);
            }
            param.set(elemwise::sub(&value, &elemwise::mul_scalar(&d, self.lr))?);
        }
        Ok(())
    }

    fn learning_rate(&self) -> f32 {
        self.lr
    }

    fn set_learning_rate(&mut self, lr: f32) {
        self.lr = lr;
    }

    fn step_count(&self) -> u64 {
        self.steps
    }

    fn state_dict(&self, params: &[(String, Param<R, E>)]) -> StateDict {
        let mut entries = BTreeMap::new();
        for (name, param) in params {
            let Some(velocity) = self.velocity.get(&param.id()) else {
                continue;
            };
            entries.insert(
                format!("{name}.velocity"),
                TensorData {
                    shape: velocity.shape().dims().to_vec(),
                    data: velocity.to_f32(),
                },
            );
        }
        StateDict { entries }
    }

    fn load_state_dict(
        &mut self,
        params: &[(String, Param<R, E>)],
        state: &StateDict,
        steps: Option<u64>,
        strict: bool,
    ) -> Result<()> {
        if strict {
            if steps.is_none() {
                return Err(Error::StateDict(
                    "the optimizer state has no step counter".to_string(),
                ));
            }
            let known = params
                .iter()
                .map(|(name, _)| format!("{name}.velocity"))
                .collect();
            refuse_unknown_keys(state, &known)?;
        }
        let mut staged = HashMap::with_capacity(params.len());
        for (name, param) in params {
            let key = format!("{name}.velocity");
            // As for AdamW's moments: a parameter that never received a gradient
            // has no velocity to save, and restores to none.
            if let Some(v) = state.entries.get(&key) {
                let want = param.shape().dims().to_vec();
                let device = param.value().device().clone();
                staged.insert(param.id(), staged_tensor(&key, v, &want, &device)?);
            }
        }
        self.velocity = staged;
        self.steps = steps.unwrap_or(0);
        Ok(())
    }
}

/// The global L2 norm of a gradient set.
///
/// Every partial sum stays on the device until the end: reading each one back
/// individually would put a host synchronisation between every pair of gradients.
pub fn grad_norm<R: Runtime, E: FloatElem>(grads: &Grads<R, E>) -> Result<f32> {
    Ok(match grad_sum_squares(grads)? {
        Some(total) => total.try_to_f32()?[0].sqrt(),
        None => 0.0,
    })
}

/// The sum of every gradient's squares, left on the device as one element.
///
/// This is [`grad_norm`] without the read-back, which is the only part of it that is
/// expensive: a step that reads the norm back to decide a clip factor cannot enqueue
/// its optimizer update until the whole backward pass has finished running.
pub fn grad_sum_squares<R: Runtime, E: FloatElem>(
    grads: &Grads<R, E>,
) -> Result<Option<Tensor<R, E>>> {
    // One launch per gradient, all writing into slices of the same partial buffer,
    // then one reduction over the lot. Squaring, reducing and concatenating each
    // gradient separately cost four launches apiece.
    let sizes: Vec<usize> = grads.iter().map(|(_, g)| g.len()).collect();
    if sizes.is_empty() {
        return Ok(None);
    }
    let mut offsets = Vec::with_capacity(sizes.len());
    let mut total = 0usize;
    for n in &sizes {
        offsets.push(total);
        total += fused::sum_squares_groups(*n).min(*n);
    }
    if total == 0 {
        return Ok(None);
    }
    let device = grads
        .iter()
        .next()
        .expect("at least one gradient")
        .1
        .device()
        .clone();
    let partials = Tensor::<R, E>::zeros(vec![total], &device);
    for ((_, g), offset) in grads.iter().zip(offsets) {
        fused::sum_squares_into(g, &partials, offset)?;
    }
    Ok(Some(reduce::sum_all(&partials)?))
}

/// Rescale gradients so their global norm is at most `max_norm`.
///
/// Returns the norm before clipping, which is worth logging: a training run that
/// clips every step is telling you the learning rate is too high.
///
/// This is the eager form, and it costs a host synchronisation plus a launch per
/// gradient. The trainer uses [`grad_scale`] instead, which computes the same factor
/// on the device and hands it to the optimizer.
pub fn clip_grad_norm<R: Runtime, E: FloatElem>(
    grads: &mut Grads<R, E>,
    max_norm: f32,
) -> Result<f32> {
    let norm = grad_norm(grads)?;
    if norm.is_finite() && norm > max_norm && max_norm > 0.0 {
        grads.scale(max_norm / (norm + 1e-6));
    }
    Ok(norm)
}

/// What [`grad_scale`] produces: both tensors hold exactly one element, and neither
/// has been read back.
pub struct GradScale<R: Runtime, E: FloatElem> {
    /// The factor every gradient is multiplied by — micro-batch averaging times the
    /// gradient-norm clip. Hand it to
    /// [`Optimizer::step_scaled`](Optimizer::step_scaled).
    pub factor: Tensor<R, E>,
    /// The sum of squares the factor was derived from, over the accumulated
    /// gradients before averaging. Reading it gives the norm to report, and is worth
    /// deferring until the rest of the step is enqueued.
    pub sum_squares: Tensor<R, E>,
}

/// The factor a step should multiply its gradients by, plus the reduction it came
/// from, both left on the device.
///
/// Combines the micro-batch averaging factor with the gradient-norm clip. Nothing is
/// read back, which is the point: a step that reads the norm to decide a clip cannot
/// enqueue its update until the backward pass has finished running.
pub fn grad_scale<R: Runtime, E: FloatElem>(
    grads: &Grads<R, E>,
    max_norm: f32,
    average: f32,
) -> Result<Option<GradScale<R, E>>> {
    let Some(sum_squares) = grad_sum_squares(grads)? else {
        return Ok(None);
    };
    let factor = fused::clip_factor(&sum_squares, max_norm, average, max_norm > 0.0);
    Ok(Some(GradScale {
        factor,
        sum_squares,
    }))
}
