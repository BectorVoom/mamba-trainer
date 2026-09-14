//! An exponential moving average of a model's weights, kept on the device.
//!
//! PPO's weights wander: good ones keep appearing and are left behind a few
//! updates later. Averaging them is the standard remedy, and it costs one extra
//! copy of the trainable weights and one launch per trainable parameter per
//! optimizer step — never a host read.
//!
//! # Semantics
//!
//! After optimizer step `t` (1-based, the [`crate::train::Trainer`]'s counter),
//! every parameter the average tracks moves as
//!
//! ```text
//! d_t   = decay                                 (EmaWarmup::None)
//! d_t   = min(decay, (1 + t) / (10 + t))        (EmaWarmup::Tf, TensorFlow's rule)
//! ema_t = ema_{t-1} + (1 - d_t) · (θ_t − ema_{t-1})
//! ema_0 = θ_0                                   (the weights at Ema::new or Ema::reset)
//! ```
//!
//! written in exactly that form and order in the kernel
//! ([`crate::tensor::ops::fused::ema_step`]), so a host twin can reproduce it
//! bit for bit. Two values of `d_t` are not computed at all but defined:
//! `d_t = 0` takes the current weights and `d_t = 1` keeps the average where it
//! is. In floating point `e + 1·(p − e)` is not always `p`, and these are the
//! two endpoints a caller is promised exactly.
//!
//! * **Granularity is the optimizer step**, not the round: a PPO update of
//!   `epochs × minibatches` steps moves the average that many times, and the
//!   half-life is `ln 2 / −ln d` optimizer steps (69 at `0.99`, 693 at `0.999`).
//!   The warm-up counts the trainer's steps too, so an average reset late in a
//!   run is already past it.
//! * **Trainable parameters** (`requires_grad`) are averaged every step, one
//!   that received no gradient included: its weight did not move, and the
//!   average still moves towards it.
//! * **Frozen parameters** are not averaged: the shadow takes the source's
//!   current tensor each step, sharing the buffer. That is safe because
//!   [`crate::nn::Param::set`] replaces a handle and no operation writes into a
//!   parameter's buffer in place; `tests/train_ema.rs` checks it.
//! * **Failure.** The update is queued after the optimizer's and before the
//!   step's reads. A step that fails before the optimizer runs leaves weights and
//!   average untouched; one that fails after it leaves both unspecified, as it
//!   always has the weights.
//! * **Element type.** The shadow has the weights' element type, which must be
//!   32 bits or wider: in 16 bits `(1 − decay)·Δ` underflows to zero for the
//!   decays worth using. The matmul precision knob rounds compute operands, not
//!   weight storage, and does not change this.
//! * **Cost.** One copy of the trainable weights on the device; one launch per
//!   trainable parameter per step (none for frozen parameters, none when
//!   `d_t` is `0` or `1`); no host reads, including at [`Ema::new`] and
//!   [`Ema::reset`], which copy on the device. [`Ema::state_dict`] reads the
//!   shadow back, once per parameter.

use cubecl::prelude::Runtime;

use crate::backend::FloatElem;
use crate::error::{Error, Result};
use crate::nn::module::{Module, StateDict, TensorData};
use crate::nn::param::Param;
use crate::tensor::ops::fused::ema_step;
use crate::tensor::{Shape, Tensor};

/// How the decay ramps up over the first optimizer steps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EmaWarmup {
    /// The configured decay from the first step.
    #[default]
    None,
    /// `min(decay, (1 + t) / (10 + t))`, TensorFlow's `ExponentialMovingAverage`
    /// with `num_updates`: early averages follow the weights closely instead of
    /// being dominated by the initial ones.
    Tf,
}

impl EmaWarmup {
    /// The name the Python bindings and checkpoints use.
    pub fn name(self) -> &'static str {
        match self {
            EmaWarmup::None => "none",
            EmaWarmup::Tf => "tf",
        }
    }

    /// The inverse of [`EmaWarmup::name`].
    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "none" => Ok(EmaWarmup::None),
            "tf" => Ok(EmaWarmup::Tf),
            other => Err(Error::config(format!(
                "unknown EMA warm-up {other:?}; expected 'none' or 'tf'"
            ))),
        }
    }
}

/// Configuration for [`Ema`].
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct EmaConfig {
    /// The weight kept by the average each step, in `[0, 1]`.
    pub decay: f32,
    /// How the decay ramps up; see [`EmaWarmup`].
    pub warmup: EmaWarmup,
}

impl EmaConfig {
    /// A constant `decay`, no warm-up.
    pub fn new(decay: f32) -> Self {
        Self {
            decay,
            warmup: EmaWarmup::None,
        }
    }

    /// Set the warm-up.
    pub fn with_warmup(mut self, warmup: EmaWarmup) -> Self {
        self.warmup = warmup;
        self
    }

    /// Refuse a decay that is not a finite number in `[0, 1]`.
    pub fn validate(&self) -> Result<()> {
        if self.decay.is_finite() && (0.0..=1.0).contains(&self.decay) {
            Ok(())
        } else {
            Err(Error::config(format!(
                "an EMA decay is a finite number in [0, 1], got {}; 0 tracks the weights, \
                 1 keeps the first ones",
                self.decay
            )))
        }
    }

    /// The decay applied after optimizer step `step` (1-based). Pure.
    pub fn decay_at(&self, step: u64) -> f32 {
        match self.warmup {
            EmaWarmup::None => self.decay,
            EmaWarmup::Tf => {
                // In f64, then rounded once: an f32 step count stops being exact
                // at 2^24 and the ratio would round twice.
                let ramp = ((1.0 + step as f64) / (10.0 + step as f64)) as f32;
                self.decay.min(ramp)
            }
        }
    }
}

/// One tracked parameter: where it lives in the model, the weight being
/// averaged, and the average.
struct Tracked<R: Runtime, E: FloatElem> {
    path: String,
    source: Param<R, E>,
    shadow: Param<R, E>,
}

/// An exponential moving average of one model's weights, held in another model
/// of the same architecture — see the [module docs](self).
///
/// The shadow is the caller's to build (for a policy, `config().init()`) and to
/// keep: it is an ordinary model, so it can be evaluated, rolled out and saved.
/// It must not be trained; nothing but [`Ema::update`] should move it.
pub struct Ema<R: Runtime, E: FloatElem> {
    config: EmaConfig,
    updates: u64,
    tracked: Vec<Tracked<R, E>>,
}

impl<R: Runtime, E: FloatElem> Ema<R, E> {
    /// Average `source`'s weights into `shadow`, starting from `source`'s
    /// current weights.
    ///
    /// Refused, with neither model changed: an invalid `config`; an element
    /// type narrower than 32 bits; a `shadow` whose parameter paths or shapes
    /// differ from `source`'s; and a `shadow` that shares a parameter with
    /// `source` (a clone of the same model rather than a second one).
    pub fn new<M: Module<R, E>>(source: &M, shadow: &M, config: EmaConfig) -> Result<Self> {
        config.validate()?;
        if core::mem::size_of::<E>() < 4 {
            return Err(Error::Unsupported(format!(
                "an EMA needs f32 or wider weights, not {}; (1 − decay)·Δ underflows in 16 bits",
                E::DTYPE.name()
            )));
        }
        let sources = source.named_parameters();
        let shadows = shadow.named_parameters();
        if sources.len() != shadows.len() {
            return Err(Error::config(format!(
                "the EMA shadow has {} parameters and its source {}; build the shadow with \
                 the source's configuration",
                shadows.len(),
                sources.len()
            )));
        }
        let mut tracked = Vec::with_capacity(sources.len());
        for ((path, source), (shadow_path, shadow)) in sources.into_iter().zip(shadows) {
            if path != shadow_path {
                return Err(Error::config(format!(
                    "the EMA shadow's parameter `{shadow_path}` is where its source has `{path}`; \
                     build the shadow with the source's configuration"
                )));
            }
            if source.shape() != shadow.shape() {
                return Err(Error::config(format!(
                    "`{path}` is {} in the EMA source and {} in its shadow",
                    source.shape(),
                    shadow.shape()
                )));
            }
            if source.id() == shadow.id() {
                return Err(Error::config(format!(
                    "the EMA shadow shares `{path}` with its source; it must be a separate \
                     model (build it with the source's configuration), or training would move it"
                )));
            }
            tracked.push(Tracked {
                path,
                source,
                shadow,
            });
        }
        let ema = Self {
            config,
            updates: 0,
            tracked,
        };
        ema.seed_from_source();
        Ok(ema)
    }

    /// The configuration.
    pub fn config(&self) -> &EmaConfig {
        &self.config
    }

    /// Replace the configuration, leaving the average and counter as they are —
    /// for a restore that adopts a saved configuration onto a live average.
    pub fn set_config(&mut self, config: EmaConfig) -> Result<()> {
        config.validate()?;
        self.config = config;
        Ok(())
    }

    /// Updates applied since [`Ema::new`], [`Ema::reset`] or the counter a load
    /// restored.
    pub fn updates(&self) -> u64 {
        self.updates
    }

    /// The decay [`Ema::update`] applies after optimizer step `step`. Pure.
    pub fn decay_at(&self, step: u64) -> f32 {
        self.config.decay_at(step)
    }

    /// Number of parameters tracked, frozen ones included.
    pub fn tracked(&self) -> usize {
        self.tracked.len()
    }

    /// Queue one update for optimizer step `step` (1-based). No host read.
    ///
    /// Every tracked parameter's shape is checked before anything is queued, so
    /// a source whose parameter was replaced by one of another shape is refused
    /// with the average unchanged.
    pub fn update(&mut self, step: u64) -> Result<()> {
        for t in &self.tracked {
            if t.source.shape() != t.shadow.shape() {
                return Err(Error::shape(format!(
                    "`{}` is now {} in the EMA source but {} in its shadow",
                    t.path,
                    t.source.shape(),
                    t.shadow.shape()
                )));
            }
        }
        let decay = self.decay_at(step);
        for t in &self.tracked {
            let theta = t.source.value();
            if !t.source.requires_grad() || decay == 0.0 {
                t.shadow.set(theta);
            } else if decay < 1.0 {
                t.shadow
                    .set(ema_step(&t.shadow.value(), &theta, 1.0 - decay)?);
            }
        }
        self.updates += 1;
        Ok(())
    }

    /// Restart the average from the source's current weights, and the counter
    /// from zero. A device copy; no host read.
    pub fn reset(&mut self) -> Result<()> {
        self.seed_from_source();
        self.updates = 0;
        Ok(())
    }

    /// Shadow ← source: a fresh copy of every trainable weight, the source's own
    /// buffer for every frozen one (the rule [`Ema::update`] follows).
    fn seed_from_source(&self) {
        for t in &self.tracked {
            let theta = t.source.value();
            if t.source.requires_grad() {
                t.shadow.set(theta.deep_clone());
            } else {
                t.shadow.set(theta);
            }
        }
    }

    /// The average, keyed by parameter path, read back to the host.
    pub fn state_dict(&self) -> StateDict {
        StateDict {
            entries: self
                .tracked
                .iter()
                .map(|t| {
                    let value = t.shadow.value();
                    (
                        t.path.clone(),
                        TensorData {
                            shape: value.shape().dims().to_vec(),
                            data: value.to_f32(),
                        },
                    )
                })
                .collect(),
        }
    }

    /// Replace the average with `state` and the counter with `updates`, all or
    /// nothing: see [`Ema::stage_state_dict`].
    pub fn load_state_dict(&mut self, state: &StateDict, updates: u64, strict: bool) -> Result<()> {
        self.stage_state_dict(state, updates, strict)?.apply(self);
        Ok(())
    }

    /// Validate `state` and copy it to the device without changing the average.
    ///
    /// `strict` refuses a missing path and an unknown one. Without it, unknown
    /// paths are ignored and a missing one restarts from the source's current
    /// weight — the non-strict warm start [`crate::train::Optimizer::load_state_dict`]
    /// documents. A shape or element-count mismatch is an error either way.
    pub fn stage_state_dict(
        &self,
        state: &StateDict,
        updates: u64,
        strict: bool,
    ) -> Result<StagedEma<R, E>> {
        if strict {
            for t in &self.tracked {
                if !state.entries.contains_key(&t.path) {
                    return Err(Error::StateDict(format!(
                        "the EMA state has no entry for `{}`",
                        t.path
                    )));
                }
            }
            let unknown: Vec<&str> = state
                .entries
                .keys()
                .filter(|k| !self.tracked.iter().any(|t| &t.path == *k))
                .map(String::as_str)
                .collect();
            if !unknown.is_empty() {
                return Err(Error::StateDict(format!(
                    "the EMA state has entries for parameters this model does not have: {unknown:?}"
                )));
            }
        }
        let mut values = Vec::with_capacity(self.tracked.len());
        for t in &self.tracked {
            let want = t.shadow.shape();
            let value = match state.entries.get(&t.path) {
                Some(entry) => {
                    if entry.shape != want.dims() {
                        return Err(Error::StateDict(format!(
                            "the EMA state's `{}` is shaped {:?}, but the parameter is {want}",
                            t.path, entry.shape
                        )));
                    }
                    if entry.data.len() != want.num_elements() {
                        return Err(Error::StateDict(format!(
                            "the EMA state's `{}` holds {} values for a {want} parameter",
                            t.path,
                            entry.data.len()
                        )));
                    }
                    Tensor::from_f32(
                        &entry.data,
                        Shape::new(entry.shape.clone()),
                        t.shadow.value().device(),
                    )?
                }
                None => t.source.value().deep_clone(),
            };
            values.push(value);
        }
        Ok(StagedEma {
            ids: self.tracked.iter().map(|t| t.shadow.id()).collect(),
            values,
            updates,
        })
    }
}

/// An average validated and copied to the device by [`Ema::stage_state_dict`],
/// not yet written into the shadow.
pub struct StagedEma<R: Runtime, E: FloatElem> {
    ids: Vec<crate::autograd::ParamId>,
    values: Vec<Tensor<R, E>>,
    updates: u64,
}

impl<R: Runtime, E: FloatElem> StagedEma<R, E> {
    /// Write the staged average and counter into `ema`. Cannot fail.
    ///
    /// # Panics
    ///
    /// If `ema` is not the average this was staged against: that is a bug in
    /// the caller, not a property of the checkpoint.
    pub fn apply(self, ema: &mut Ema<R, E>) {
        assert!(
            ema.tracked.len() == self.ids.len()
                && ema
                    .tracked
                    .iter()
                    .zip(&self.ids)
                    .all(|(t, id)| t.shadow.id() == *id),
            "a staged EMA state applied to another EMA than the one it was staged for"
        );
        for (t, value) in ema.tracked.iter().zip(self.values) {
            t.shadow.set(value);
        }
        ema.updates = self.updates;
    }
}
