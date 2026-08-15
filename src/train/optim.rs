//! Optimizers.
//!
//! Optimizers see a flat `&[Param]` and a [`Grads`] map keyed by [`ParamId`]. They
//! never touch the model tree, which is why the same optimizer works for a full
//! fine-tune, a LoRA-only run (pass only the adapter parameters) and a QAT run.

use std::collections::HashMap;

use cubecl::prelude::Runtime;

use crate::autograd::{Grads, ParamId};
use crate::backend::FloatElem;
use crate::error::Result;
use crate::nn::param::Param;
use crate::tensor::Tensor;
use crate::tensor::ops::{elemwise, reduce};

/// Something that turns gradients into parameter updates.
pub trait Optimizer<R: Runtime, E: FloatElem> {
    /// Apply one update to every parameter that has a gradient.
    fn step(&mut self, params: &[Param<R, E>], grads: &Grads<R, E>) -> Result<()>;

    /// Current learning rate.
    fn learning_rate(&self) -> f32;

    /// Override the learning rate, e.g. from a schedule.
    fn set_learning_rate(&mut self, lr: f32);

    /// Number of updates applied so far.
    fn step_count(&self) -> u64;
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
    fn step(&mut self, params: &[Param<R, E>], grads: &Grads<R, E>) -> Result<()> {
        self.steps += 1;
        let t = self.steps as f32;
        let bias1 = 1.0 - self.config.beta1.powf(t);
        let bias2 = 1.0 - self.config.beta2.powf(t);

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

            // m <- b1 m + (1-b1) g
            entry.m = elemwise::add(
                &elemwise::mul_scalar(&entry.m, self.config.beta1),
                &elemwise::mul_scalar(grad, 1.0 - self.config.beta1),
            )?;
            // v <- b2 v + (1-b2) g^2
            let grad_sq = elemwise::mul(grad, grad)?;
            entry.v = elemwise::add(
                &elemwise::mul_scalar(&entry.v, self.config.beta2),
                &elemwise::mul_scalar(&grad_sq, 1.0 - self.config.beta2),
            )?;

            let m_hat = elemwise::mul_scalar(&entry.m, 1.0 / bias1);
            let v_hat = elemwise::mul_scalar(&entry.v, 1.0 / bias2);
            let denom = elemwise::add_scalar(&elemwise::sqrt(&v_hat), self.config.eps);
            let mut update = elemwise::div(&m_hat, &denom)?;

            let decay = if self.config.decay_matrices_only && value.rank() < 2 {
                0.0
            } else {
                self.config.weight_decay
            };
            if decay != 0.0 {
                update = elemwise::add(&update, &elemwise::mul_scalar(&value, decay))?;
            }

            let next = elemwise::sub(&value, &elemwise::mul_scalar(&update, self.lr))?;
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
    fn step(&mut self, params: &[Param<R, E>], grads: &Grads<R, E>) -> Result<()> {
        self.steps += 1;
        for param in params {
            if !param.requires_grad() {
                continue;
            }
            let Some(grad) = grads.get(param.id()) else {
                continue;
            };
            let value = param.value();
            let mut d = grad.clone();
            if self.weight_decay != 0.0 {
                d = elemwise::add(&d, &elemwise::mul_scalar(&value, self.weight_decay))?;
            }
            if self.momentum != 0.0 {
                let velocity = match self.velocity.remove(&param.id()) {
                    Some(prev) => {
                        elemwise::add(&elemwise::mul_scalar(&prev, self.momentum), &d)?
                    }
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
}

/// The global L2 norm of a gradient set.
pub fn grad_norm<R: Runtime, E: FloatElem>(grads: &Grads<R, E>) -> Result<f32> {
    let mut total = 0.0f32;
    for (_, g) in grads.iter() {
        let sq = elemwise::mul(g, g)?;
        total += reduce::sum_all(&sq)?.to_f32()[0];
    }
    Ok(total.sqrt())
}

/// Rescale gradients so their global norm is at most `max_norm`.
///
/// Returns the norm before clipping, which is worth logging: a training run that
/// clips every step is telling you the learning rate is too high.
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
