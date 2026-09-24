//! Action heads: a flat linear head, or a pointer that scores entities.
//!
//! A flat head learns one output per action, so "walk to tile 17" and "walk to
//! tile 18" share nothing. When the actions *are* the entities of a set, a
//! [`PointerHead`] scores each entity's embedding against the recurrent state
//! with one set of weights, so the logit of entity `i` depends on what entity `i`
//! is, not on which slot it sits in: permuting the entities permutes the logits.
//!
//! * **additive** (default): `logit_i = v · relu(W_h h + b + W_e e_i)`;
//! * **dot**: `logit_i = (W_q h) · e_i`, cheaper but measured worse on scorer
//!   tasks.
//!
//! Empty slots get the most negative finite logit (a probability of exactly
//! zero) inside the head; a learner's action mask is applied on top as usual.
//! `extra_actions = K` appends `K` flat logits after the `N` entity logits — a
//! "wait" or "pass" action, say — so the action id is the entity index for the
//! first `N` ids and `N + k` for the extras.
//!
//! # Launch budget
//!
//! The additive head is five launches over `[B, T, N, H]` (`W_e e`, the
//! broadcast add, the ReLU, the `v` product, the mask) plus two for `W_h h + b`,
//! which is `[B, T, H]` — the bias rides on `W_h` rather than being its own add.
//! The dot head is three: `W_q h`, one batched product, the mask.

use cubecl::prelude::Runtime;

use crate::autograd::Var;
use crate::backend::{Device, FloatElem};
use crate::error::{Error, Result};
use crate::nn::init::Initializer;
use crate::nn::linear::{Linear, LinearConfig};
use crate::nn::module::{Module, ModuleVisitor};
use crate::tensor::ops::random::Rng;

/// How a [`PointerHead`] scores an entity against the state.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Scoring {
    /// `v · relu(W_h h + b + W_e e_i)`.
    #[default]
    Additive,
    /// `(W_q h) · e_i`.
    Dot,
}

impl Scoring {
    /// The name the configuration uses.
    pub fn name(self) -> &'static str {
        match self {
            Scoring::Additive => "additive",
            Scoring::Dot => "dot",
        }
    }

    /// Parse [`Scoring::name`].
    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "additive" => Ok(Scoring::Additive),
            "dot" => Ok(Scoring::Dot),
            other => Err(Error::config(format!(
                "unknown pointer scoring {other:?}; expected 'additive' or 'dot'"
            ))),
        }
    }
}

/// Which head produces the action logits.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum ActionHeadConfig {
    /// `Linear(d_model → action_dim)`, today's head.
    #[default]
    Flat,
    /// Score the entities of `set`, plus `extra_actions` flat logits after them.
    Pointer(PointerHeadConfig),
}

/// Configuration for a [`PointerHead`].
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PointerHeadConfig {
    /// The entity set whose entities are the actions.
    pub set: String,
    /// Width of the additive scorer's hidden layer; unused by `dot`.
    #[serde(default = "PointerHeadConfig::default_hidden")]
    pub hidden: usize,
    /// How entities are scored.
    #[serde(default)]
    pub scoring: Scoring,
    /// Flat actions appended after the entity logits.
    #[serde(default)]
    pub extra_actions: usize,
}

impl PointerHeadConfig {
    fn default_hidden() -> usize {
        64
    }

    /// A pointer over the entities of `set`, additive, no extra actions.
    pub fn new(set: impl Into<String>) -> Self {
        Self {
            set: set.into(),
            hidden: Self::default_hidden(),
            scoring: Scoring::Additive,
            extra_actions: 0,
        }
    }

    /// Width of the additive scorer's hidden layer.
    pub fn with_hidden(mut self, hidden: usize) -> Self {
        self.hidden = hidden;
        self
    }

    /// How entities are scored.
    pub fn with_scoring(mut self, scoring: Scoring) -> Self {
        self.scoring = scoring;
        self
    }

    /// Flat actions appended after the entity logits.
    pub fn with_extra_actions(mut self, extra: usize) -> Self {
        self.extra_actions = extra;
        self
    }

    /// Instantiate over a state of width `d_model` and embeddings of `d_entity`.
    pub fn init<R: Runtime, E: FloatElem>(
        &self,
        set_index: usize,
        d_model: usize,
        d_entity: usize,
        device: &Device<R>,
        rng: &mut Rng,
    ) -> Result<PointerHead<R, E>> {
        if self.scoring == Scoring::Additive && self.hidden == 0 {
            return Err(Error::config(
                "action_head.hidden must be positive for additive scoring".to_string(),
            ));
        }
        // Small output weights start the policy close to uniform over the
        // entities, as the flat actor's near-zero init does.
        let small = Initializer::Normal {
            mean: 0.0,
            std: 0.01,
        };
        let score = match self.scoring {
            Scoring::Additive => Score::Additive {
                w_h: LinearConfig::new(d_model, self.hidden).init(device, rng),
                w_e: LinearConfig::new(d_entity, self.hidden)
                    .with_bias(false)
                    .init(device, rng),
                v: LinearConfig::new(self.hidden, 1)
                    .with_bias(false)
                    .with_initializer(small)
                    .init(device, rng),
            },
            Scoring::Dot => Score::Dot {
                w_q: LinearConfig::new(d_model, d_entity)
                    .with_bias(false)
                    .with_initializer(small)
                    .init(device, rng),
            },
        };
        let extra = (self.extra_actions > 0).then(|| {
            LinearConfig::new(d_model, self.extra_actions)
                .with_initializer(small)
                .init(device, rng)
        });
        Ok(PointerHead {
            set_index,
            score,
            extra,
        })
    }
}

enum Score<R: Runtime, E: FloatElem> {
    Additive {
        w_h: Linear<R, E>,
        w_e: Linear<R, E>,
        v: Linear<R, E>,
    },
    Dot {
        w_q: Linear<R, E>,
    },
}

/// Scores the entities of one set against the recurrent state.
pub struct PointerHead<R: Runtime, E: FloatElem> {
    set_index: usize,
    score: Score<R, E>,
    extra: Option<Linear<R, E>>,
}

impl<R: Runtime, E: FloatElem> PointerHead<R, E> {
    /// Which set of the observation spec this head points into.
    pub fn set_index(&self) -> usize {
        self.set_index
    }

    /// Logits `[B, T, N (+ K)]` from the state `[B, T, D]`, the set's embeddings
    /// `[B, T, N, d_e]` and its presence `[B, T, N, 1]`.
    pub fn apply(
        &self,
        hidden: &Var<R, E>,
        entities: &Var<R, E>,
        presence: &Var<R, E>,
    ) -> Result<Var<R, E>> {
        hidden.shape().expect_rank(3)?;
        entities.shape().expect_rank(4)?;
        let (b, t, n) = (entities.dims()[0], entities.dims()[1], entities.dims()[2]);
        if crate::nn::entity::fused_entity_enabled() {
            return self.apply_fused(hidden, entities, presence, b, t, n);
        }
        let scores = match &self.score {
            Score::Additive { w_h, w_e, v } => {
                let q = w_h.apply(hidden)?.unsqueeze(2)?;
                let k = w_e.apply(entities)?;
                v.apply(&k.add(&q)?.relu())?
            }
            Score::Dot { w_q } => {
                let q = w_q.apply(hidden)?.unsqueeze(3)?;
                entities.matmul(&q)?
            }
        };
        let legal = presence.tensor().reshape(vec![b, t, n])?;
        let logits = scores.reshape(vec![b, t, n])?.mask_logits(&legal)?;
        match &self.extra {
            Some(extra) => crate::autograd::cat(&[logits, extra.apply(hidden)?], 2),
            None => Ok(logits),
        }
    }

    /// The fused pointer path: the `w_h` / `w_e` / `w_q` and extras projections
    /// stay on the tuned `Linear` kernels; one fused launch replaces the
    /// broadcast add, ReLU, `v` product, reshape, `mask_logits` and extras
    /// `cat` (additive) or the batched `[.., d, 1]` matmul, `mask_logits` and
    /// `cat` (dot).
    ///
    /// Everything is contiguous, so the flat `[rows, ..]` views (`rows = B*T`)
    /// are free reshapes. `legal` is `presence` reshaped flat: the mask never
    /// carried a gradient, so it stays a constant here as in the composed path.
    /// The `v` weight joins the tape as a `Var` off `hidden`, so its gradient
    /// reaches the optimizer.
    fn apply_fused(
        &self,
        hidden: &Var<R, E>,
        entities: &Var<R, E>,
        presence: &Var<R, E>,
        b: usize,
        t: usize,
        n: usize,
    ) -> Result<Var<R, E>> {
        let rows = b * t;
        let legal = presence.tensor().reshape(vec![rows, n])?;
        let kx = self.extra.as_ref().map(|e| e.out_features()).unwrap_or(0);
        let extra_flat = match &self.extra {
            Some(extra) => Some(extra.apply(hidden)?.reshape(vec![rows, kx])?),
            None => None,
        };
        let flat = match &self.score {
            Score::Additive { w_h, w_e, v } => {
                let q = w_h.apply(hidden)?;
                let h = q.dims()[2];
                let q = q.reshape(vec![rows, h])?;
                let k = w_e.apply(entities)?;
                let k = k.reshape(vec![rows, n, h])?;
                Var::pointer_additive(&k, &q, &v.weight().var(hidden), &legal, extra_flat.as_ref(), n)?
            }
            Score::Dot { w_q } => {
                let qd = w_q.apply(hidden)?;
                let d = qd.dims()[2];
                let qd = qd.reshape(vec![rows, d])?;
                let e = entities.reshape(vec![rows, n, d])?;
                Var::pointer_dot(&e, &qd, &legal, extra_flat.as_ref(), n)?
            }
        };
        flat.reshape(vec![b, t, n + kx])
    }
}

impl<R: Runtime, E: FloatElem> Module<R, E> for PointerHead<R, E> {
    fn visit(&self, visitor: &mut ModuleVisitor<'_, R, E>) {
        let pointer = PointerParams(&self.score);
        visitor.child("pointer", &pointer);
        if let Some(extra) = &self.extra {
            visitor.child("extra", extra);
        }
    }
}

/// The scorer's parameters under `pointer.*`.
struct PointerParams<'a, R: Runtime, E: FloatElem>(&'a Score<R, E>);

impl<R: Runtime, E: FloatElem> Module<R, E> for PointerParams<'_, R, E> {
    fn visit(&self, visitor: &mut ModuleVisitor<'_, R, E>) {
        match self.0 {
            Score::Additive { w_h, w_e, v } => {
                visitor.child("w_h", w_h);
                visitor.child("w_e", w_e);
                visitor.child("v", v);
            }
            Score::Dot { w_q } => visitor.child("w_q", w_q),
        }
    }
}

/// The actor: a flat linear head, or a pointer over an entity set.
pub enum ActionHead<R: Runtime, E: FloatElem> {
    /// `Linear(d_model → action_dim)`.
    Flat(Linear<R, E>),
    /// Entity scores, plus optional flat extras.
    Pointer(PointerHead<R, E>),
}
