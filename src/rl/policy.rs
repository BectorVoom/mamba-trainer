//! An actor-critic policy over a Mamba-3 stack.
//!
//! The policy is deliberately thin — an observation encoder, a stack of Mamba-3
//! blocks, and two linear heads — because the interesting part is that the *same*
//! stack serves both of the shapes reinforcement learning needs:
//!
//! * [`Mamba3Policy::step`] advances `B` environments by one observation each,
//!   in `O(1)` per step regardless of how long their episodes have run;
//! * [`Mamba3Policy::forward`] consumes a whole `[B, T]` rollout buffer through
//!   the chunked parallel scan, in `O(T)` with `O(log T)`-depth dependencies.
//!
//! Both honour the same `[B]` / `[B, T]` episode-termination mask, and they agree
//! numerically — which is the property that makes the pair usable, because a
//! policy gradient estimated from the second must be a gradient of what the first
//! actually did.
//!
//! # What is stored for the backward pass
//!
//! Nothing explicitly. The scan is written out of differentiable primitives, so
//! the tape already holds exactly the intermediates its own adjoint needs and
//! releases them when [`crate::autograd::Var::backward`] runs. There is no
//! separate trajectory-of-states buffer to size, fill or keep in sync.
//!
//! # Structured observations
//!
//! By default the encoder is one `Linear(obs_dim → d_model)`, which learns
//! separate weights for every position of the observation. When the observation
//! is a list of like things — tiles, units, cards — set
//! [`Mamba3PolicyConfig::with_obs_spec`] and the encoder becomes one shared
//! [`EntityEncoder`] per entity set, pooled under the presence flags and
//! projected to `d_model`; [`ActionHeadConfig::Pointer`] then lets the actor score
//! the entities themselves. The wire format stays flat (see [`ObsSpec`]), so
//! buffers, collectors, learners and environments are unchanged.

use std::collections::BTreeMap;

use cubecl::prelude::Runtime;

use crate::autograd::Var;
use crate::backend::{Device, FloatElem};
use crate::distributions::Categorical;
use crate::error::{Error, Result};
use crate::models::mamba3::{Mamba3Block, Mamba3BlockConfig, MixerCache};
use crate::nn::entity::{EntityEncoder, EntityEncoderConfig, PoolingConfig, Presence, pool_parts};
use crate::nn::init::Initializer;
use crate::nn::linear::{Linear, LinearConfig};
use crate::nn::module::{Module, ModuleVisitor};
use crate::nn::norm::{RmsNorm, RmsNormConfig};
use crate::ssm::config::SsmConfig;
use crate::tensor::Tensor;
use crate::tensor::ops::random::Rng;

use super::heads::{ActionHead, ActionHeadConfig};
use super::spec::ObsSpec;
use super::state::Mamba3StateBuffer;

/// What the policy produces for every position it is given.
pub struct PolicyOutput<R: Runtime, E: FloatElem> {
    /// Action logits, `[batch, seq, action_dim]`.
    pub logits: Var<R, E>,
    /// State value, `[batch, seq]`.
    pub value: Var<R, E>,
}

impl<R: Runtime, E: FloatElem> PolicyOutput<R, E> {
    /// The action distribution these logits describe.
    ///
    /// The bridge from a policy to [`crate::distributions`]: everything a
    /// policy-gradient loss asks of a policy — the log-probability of what it did,
    /// the entropy of what it might have done, the divergence from what it used to
    /// be — is a method on the returned [`Categorical`], and each one is a single
    /// fused kernel over the logit rows.
    ///
    /// ```no_run
    /// # use mamba3::prelude::*;
    /// # use mamba3::distributions::Distribution;
    /// # fn go<R: cubecl::prelude::Runtime>(out: &mamba3::rl::PolicyOutput<R, f32>,
    /// #        actions: &mamba3::tensor::ops::index::IdTensor<R>) -> Result<()> {
    /// let policy = out.distribution()?;
    /// let log_prob = policy.log_prob_ids(actions)?;
    /// let bonus = policy.entropy()?;
    /// # Ok(()) }
    /// ```
    pub fn distribution(&self) -> Result<Categorical<R, E>> {
        Categorical::from_logits(self.logits.clone())
    }
}

impl<R: Runtime, E: FloatElem> Clone for PolicyOutput<R, E> {
    fn clone(&self) -> Self {
        Self {
            logits: self.logits.clone(),
            value: self.value.clone(),
        }
    }
}

/// Configuration for [`Mamba3Policy`].
///
/// The structured-observation fields (`obs_spec`, `entity_encoders`, `pooling`,
/// `action_head`) all default to the flat policy, which is byte-identical to a
/// configuration that predates them.
#[derive(Debug, Clone, PartialEq)]
pub struct Mamba3PolicyConfig {
    /// Width of one observation vector.
    pub obs_dim: usize,
    /// Number of discrete actions.
    pub action_dim: usize,
    /// Number of Mamba-3 blocks.
    pub n_layers: usize,
    /// The mixer configuration; `d_model` comes from here.
    pub ssm: SsmConfig,
    /// Normalisation epsilon.
    pub norm_eps: f32,
    /// Initialisation seed.
    pub seed: u64,
    /// How to read the flat observation as entity sets; `None` is the flat
    /// encoder.
    pub obs_spec: Option<ObsSpec>,
    /// Encoder per set name; a set without an entry gets
    /// [`EntityEncoderConfig::default`].
    pub entity_encoders: BTreeMap<String, EntityEncoderConfig>,
    /// How each set's embeddings are summarised for the backbone.
    pub pooling: PoolingConfig,
    /// Which head produces the action logits.
    pub action_head: ActionHeadConfig,
}

impl Mamba3PolicyConfig {
    /// A policy sized for on-device rollouts.
    ///
    /// The defaults are the small end of the architecture — one `B`/`C` group per
    /// head, a rotational transition, real state tracking — with `d_model` split
    /// evenly into `n_heads` of `head_dim`.
    pub fn new(obs_dim: usize, action_dim: usize, d_model: usize, n_layers: usize) -> Self {
        let head_dim = 64.min(d_model.max(1));
        let n_heads = (d_model / head_dim).max(1);
        Self {
            obs_dim,
            action_dim,
            n_layers,
            ssm: SsmConfig {
                d_model,
                n_heads,
                head_dim,
                d_state: 16,
                n_groups: n_heads,
                ..SsmConfig::default()
            },
            norm_eps: 1e-5,
            seed: 0,
            obs_spec: None,
            entity_encoders: BTreeMap::new(),
            pooling: PoolingConfig::default(),
            action_head: ActionHeadConfig::Flat,
        }
    }

    /// Edit the mixer configuration.
    pub fn with_ssm(mut self, f: impl FnOnce(&mut SsmConfig)) -> Self {
        f(&mut self.ssm);
        self
    }

    /// Initialisation seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Read the observation as entity sets.
    pub fn with_obs_spec(mut self, spec: ObsSpec) -> Self {
        self.obs_spec = Some(spec);
        self
    }

    /// Configure the encoder of the set called `set`.
    pub fn with_entity_encoder(
        mut self,
        set: impl Into<String>,
        config: EntityEncoderConfig,
    ) -> Self {
        self.entity_encoders.insert(set.into(), config);
        self
    }

    /// How each set's embeddings are summarised.
    pub fn with_pooling(mut self, pooling: PoolingConfig) -> Self {
        self.pooling = pooling;
        self
    }

    /// Which head produces the action logits.
    pub fn with_action_head(mut self, head: ActionHeadConfig) -> Self {
        self.action_head = head;
        self
    }

    /// Whether this policy reads its observation as entity sets.
    pub fn is_structured(&self) -> bool {
        self.obs_spec.is_some()
    }

    /// The encoder configuration of `set`, the default when none was given.
    pub fn entity_encoder(&self, set: &str) -> EntityEncoderConfig {
        self.entity_encoders.get(set).cloned().unwrap_or_default()
    }

    /// Check internal consistency.
    pub fn validate(&self) -> Result<()> {
        if self.obs_dim == 0 || self.action_dim == 0 {
            return Err(Error::config(
                "a policy needs a positive observation and action width".to_string(),
            ));
        }
        if self.n_layers == 0 {
            return Err(Error::config(
                "a policy needs at least one layer".to_string(),
            ));
        }
        self.ssm.validate()?;
        self.validate_structure()
    }

    /// The structured-observation half of [`Mamba3PolicyConfig::validate`].
    fn validate_structure(&self) -> Result<()> {
        let Some(spec) = &self.obs_spec else {
            if let Some(name) = self.entity_encoders.keys().next() {
                return Err(Error::config(format!(
                    "entity_encoders names set {name:?}, but there is no obs_spec to read \
                     entity sets from"
                )));
            }
            if self.pooling != PoolingConfig::default() {
                return Err(Error::config(
                    "pooling is set, but there is no obs_spec to pool entity sets from"
                        .to_string(),
                ));
            }
            if let ActionHeadConfig::Pointer(head) = &self.action_head {
                return Err(Error::config(format!(
                    "action_head points into set {:?}, but there is no obs_spec",
                    head.set
                )));
            }
            return Ok(());
        };
        spec.validate()?;
        if spec.obs_dim() != self.obs_dim {
            return Err(Error::config(format!(
                "obs_dim is {} but obs_spec describes {} (globals {} + entity sets {})",
                self.obs_dim,
                spec.obs_dim(),
                spec.globals,
                spec.obs_dim() - spec.globals
            )));
        }
        for (name, encoder) in &self.entity_encoders {
            if spec.set(name).is_none() {
                return Err(Error::config(format!(
                    "entity_encoders names set {name:?}, which obs_spec does not have \
                     (sets: {:?})",
                    spec.sets.iter().map(|s| s.name.as_str()).collect::<Vec<_>>()
                )));
            }
            encoder
                .validate()
                .map_err(|e| Error::config(format!("entity_encoders[{name:?}]: {e}")))?;
        }
        self.pooling.validate()?;
        if let ActionHeadConfig::Pointer(head) = &self.action_head {
            let set = spec.set(&head.set).ok_or_else(|| {
                Error::config(format!(
                    "action_head points into set {:?}, which obs_spec does not have",
                    head.set
                ))
            })?;
            if set.count + head.extra_actions != self.action_dim {
                return Err(Error::config(format!(
                    "action_head points into {:?} ({} entities) with {} extra actions, \
                     which makes {} actions, but action_dim is {}",
                    head.set,
                    set.count,
                    head.extra_actions,
                    set.count + head.extra_actions,
                    self.action_dim
                )));
            }
            if head.scoring == crate::rl::heads::Scoring::Additive && head.hidden == 0 {
                return Err(Error::config(
                    "action_head.hidden must be positive for additive scoring".to_string(),
                ));
            }
        }
        Ok(())
    }

    /// The entity encoders and the pooled projection, in spec order.
    fn init_entities<R: Runtime, E: FloatElem>(
        &self,
        spec: &ObsSpec,
        device: &Device<R>,
        rng: &mut Rng,
    ) -> Result<EntityStage<R, E>> {
        let encoders = spec
            .sets
            .iter()
            .map(|set| {
                self.entity_encoder(&set.name)
                    .init(set.features, set.count, device, rng)
            })
            .collect::<Result<Vec<_>>>()?;
        let pooled: usize = encoders
            .iter()
            .map(|e: &EntityEncoder<R, E>| e.d_entity() * self.pooling.kinds.len())
            .sum();
        Ok(EntityStage {
            spec: spec.clone(),
            pointer_set: match &self.action_head {
                ActionHeadConfig::Pointer(head) => spec.set_index(&head.set),
                ActionHeadConfig::Flat => None,
            },
            kinds: self.pooling.kinds.clone(),
            encoders,
            proj: LinearConfig::new(spec.globals + pooled, self.ssm.d_model).init(device, rng),
        })
    }

    /// Instantiate on a device.
    pub fn init<R: Runtime, E: FloatElem>(&self, device: &Device<R>) -> Result<Mamba3Policy<R, E>> {
        let mut rng = Rng::seeded(self.seed);
        self.init_with_rng(device, &mut rng)
    }

    /// Instantiate with an explicit RNG.
    pub fn init_with_rng<R: Runtime, E: FloatElem>(
        &self,
        device: &Device<R>,
        rng: &mut Rng,
    ) -> Result<Mamba3Policy<R, E>> {
        self.validate()?;
        let d_model = self.ssm.d_model;
        let normal = Initializer::Normal {
            mean: 0.0,
            std: 0.02,
        };
        let block = Mamba3BlockConfig::new(self.ssm.clone())
            .with_norm_eps(self.norm_eps)
            .mixer(|m| m.with_depth(self.n_layers));
        let blocks = (0..self.n_layers)
            .map(|_| block.init(device, rng))
            .collect::<Result<Vec<_>>>()?;

        // The draw order — blocks, input, norm, actor, critic — is what fixes the
        // flat policy's weights for a seed; keep it.
        let input = match &self.obs_spec {
            None => InputStage::Flat(
                LinearConfig::new(self.obs_dim, d_model)
                    .with_initializer(normal)
                    .init(device, rng),
            ),
            Some(spec) => InputStage::Entities(self.init_entities(spec, device, rng)?),
        };
        let norm = RmsNormConfig::new(d_model)
            .with_eps(self.norm_eps)
            .init(device, rng);
        let actor = match &self.action_head {
            // A near-zero actor head starts the policy close to uniform, which is
            // what keeps the first rollouts exploratory instead of committed.
            ActionHeadConfig::Flat => ActionHead::Flat(
                LinearConfig::new(d_model, self.action_dim)
                    .with_initializer(Initializer::Normal {
                        mean: 0.0,
                        std: 0.01,
                    })
                    .init(device, rng),
            ),
            ActionHeadConfig::Pointer(head) => {
                let index = self
                    .obs_spec
                    .as_ref()
                    .and_then(|spec| spec.set_index(&head.set))
                    .ok_or_else(|| {
                        Error::config(format!("no entity set called {:?}", head.set))
                    })?;
                let d_entity = self.entity_encoder(&head.set).d_entity;
                ActionHead::Pointer(head.init(index, d_model, d_entity, device, rng)?)
            }
        };

        Ok(Mamba3Policy {
            input,
            blocks,
            norm,
            actor,
            critic: LinearConfig::new(d_model, 1)
                .with_initializer(normal)
                .init(device, rng),
            config: self.clone(),
        })
    }
}

/// How the observation reaches the residual stream.
enum InputStage<R: Runtime, E: FloatElem> {
    /// `Linear(obs_dim → d_model)`.
    Flat(Linear<R, E>),
    /// Shared encoders per entity set, pooled and projected.
    Entities(EntityStage<R, E>),
}

/// The structured encoder: `[globals | pools of every set] → Linear → d_model`.
struct EntityStage<R: Runtime, E: FloatElem> {
    spec: ObsSpec,
    /// The set the pointer head scores, whose embeddings are kept for it.
    pointer_set: Option<usize>,
    kinds: Vec<crate::nn::entity::PoolKind>,
    encoders: Vec<EntityEncoder<R, E>>,
    proj: Linear<R, E>,
}

/// What the input stage hands on: the residual stream, and the embeddings and
/// presence of the set a pointer head scores.
struct Encoded<R: Runtime, E: FloatElem> {
    x: Var<R, E>,
    pointed: Option<(Var<R, E>, Var<R, E>)>,
}

impl<R: Runtime, E: FloatElem> InputStage<R, E> {
    fn apply(&self, obs: &Var<R, E>) -> Result<Encoded<R, E>> {
        let stage = match self {
            InputStage::Flat(encoder) => {
                return Ok(Encoded {
                    x: encoder.apply(obs)?,
                    pointed: None,
                });
            }
            InputStage::Entities(stage) => stage,
        };
        if crate::nn::entity::fused_entity_enabled() {
            return stage.apply_fused(obs);
        }
        let split = stage.spec.split(obs)?;
        let mut parts = Vec::with_capacity(1 + stage.encoders.len() * stage.kinds.len());
        parts.extend(split.globals);
        let mut pointed = None;
        for (i, (set, encoder)) in split.sets.into_iter().zip(&stage.encoders).enumerate() {
            let embeddings = encoder.apply(&set.features, &set.presence)?;
            let presence = Presence::new(&set.presence)?;
            parts.extend(pool_parts(&embeddings, &presence, &stage.kinds)?);
            if stage.pointer_set == Some(i) {
                pointed = Some((embeddings, set.presence));
            }
        }
        // One concatenation and one projection for every set and pool together,
        // rather than a projection per part summed afterwards.
        let joined = crate::autograd::cat(&parts, 2)?;
        Ok(Encoded {
            x: stage.proj.apply(&joined)?,
            pointed,
        })
    }
}

impl<R: Runtime, E: FloatElem> EntityStage<R, E> {
    /// The fused input path: one [`Var::entity_prepare`] launch per set instead
    /// of the split, the zeroing multiply and the four presence launches, and
    /// one [`Var::entity_join`] launch per set instead of the mean matmul, the
    /// max chain, the globals slice and the `cat`.
    ///
    /// Everything the join reads comes from the prepare kernels; the joined
    /// buffer is the projection's input directly. The pointer head takes a
    /// constant `legal` view: its mask never carried a gradient (see
    /// [`Var::mask_logits`]), so a traced presence would change nothing but the
    /// launch that produced it.
    fn apply_fused(&self, obs: &Var<R, E>) -> Result<Encoded<R, E>> {
        obs.shape().expect_rank(3)?;
        let dims = obs.dims();
        let (batch, seq, obs_dim) = (dims[0], dims[1], dims[2]);
        if obs_dim != self.spec.obs_dim() {
            return Err(Error::shape(format!(
                "obs_spec describes obs_dim={}, got {}",
                self.spec.obs_dim(),
                obs.shape()
            )));
        }
        let rows = batch * seq;
        // Free reshapes: every tensor here is contiguous.
        let flat = obs.reshape(vec![rows, obs_dim])?;
        let offsets = self.spec.offsets();
        let mut flat_embs = Vec::with_capacity(self.encoders.len());
        let mut metas: Vec<(
            crate::tensor::Tensor<R, E>,
            crate::tensor::Tensor<R, E>,
            crate::tensor::Tensor<R, E>,
        )> = Vec::with_capacity(self.encoders.len());
        let mut pointed = None;
        for (i, (set, encoder)) in self.spec.sets.iter().zip(&self.encoders).enumerate() {
            let (prepared, mean_w, legal, any) =
                Var::entity_prepare(&flat, offsets[i], set.count, set.features)?;
            let features = prepared.reshape(vec![batch, seq, set.count, set.features])?;
            let embeddings = encoder.apply_prepared(&features)?;
            let d = embeddings.dims()[3];
            if self.pointer_set == Some(i) {
                let flags = legal.reshape(vec![batch, seq, set.count, 1])?;
                pointed = Some((embeddings.clone(), Var::constant(flags)));
            }
            flat_embs.push(embeddings.reshape(vec![rows, set.count, d])?);
            metas.push((mean_w, legal, any));
        }
        let inputs: Vec<crate::autograd::ops::EntityPoolInput<'_, R, E>> = flat_embs
            .iter()
            .zip(metas.iter())
            .map(|(e, (mean_w, legal, any))| {
                crate::autograd::ops::EntityPoolInput {
                    embeddings: e,
                    mean_w: mean_w.clone(),
                    legal: legal.clone(),
                    any: any.clone(),
                    kinds: self.kinds.clone(),
                }
            })
            .collect();
        let joined_flat = Var::entity_join(&flat, self.spec.globals, &inputs)?;
        let width = joined_flat.dims()[1];
        let joined = joined_flat.reshape(vec![batch, seq, width])?;
        Ok(Encoded {
            x: self.proj.apply(&joined)?,
            pointed,
        })
    }
}

impl<R: Runtime, E: FloatElem> Module<R, E> for EntityStage<R, E> {
    fn visit(&self, visitor: &mut ModuleVisitor<'_, R, E>) {
        for (set, encoder) in self.spec.sets.iter().zip(&self.encoders) {
            visitor.child(&format!("entity.{}", set.name), encoder);
        }
        visitor.child("pool.proj", &self.proj);
    }
}

/// A recurrent actor-critic policy.
pub struct Mamba3Policy<R: Runtime, E: FloatElem> {
    input: InputStage<R, E>,
    blocks: Vec<Mamba3Block<R, E>>,
    norm: RmsNorm<R, E>,
    actor: ActionHead<R, E>,
    critic: Linear<R, E>,
    config: Mamba3PolicyConfig,
}

impl<R: Runtime, E: FloatElem> core::fmt::Debug for Mamba3Policy<R, E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "Mamba3Policy(obs={}, actions={}, d_model={}, layers={})",
            self.config.obs_dim,
            self.config.action_dim,
            self.config.ssm.d_model,
            self.blocks.len(),
        )
    }
}

impl<R: Runtime, E: FloatElem> Mamba3Policy<R, E> {
    /// The configuration this policy was built from.
    pub fn config(&self) -> &Mamba3PolicyConfig {
        &self.config
    }

    /// The blocks in the stack.
    pub fn blocks(&self) -> &[Mamba3Block<R, E>] {
        &self.blocks
    }

    /// Allocate the rollout state for `envs` environments.
    ///
    /// This is the only allocation an RL loop needs to make: the returned buffer
    /// is written in place from then on. Do it before the loop starts.
    pub fn empty_state(&self, envs: usize, device: &Device<R>) -> Mamba3StateBuffer<R, E> {
        Mamba3StateBuffer::new(
            self.blocks
                .iter()
                .map(|b| b.empty_cache(envs, device))
                .collect(),
            envs,
            device,
        )
    }

    /// Check that an observation window has the shape the encoder expects.
    fn check_obs(&self, obs: &Var<R, E>) -> Result<(usize, usize)> {
        obs.shape().expect_rank(3)?;
        let dims = obs.dims();
        if dims[2] != self.config.obs_dim {
            return Err(Error::shape(format!(
                "Mamba3Policy expects obs_dim={}, got {}",
                self.config.obs_dim,
                obs.shape()
            )));
        }
        Ok((dims[0], dims[1]))
    }

    /// Heads over the stack's output.
    fn heads(
        &self,
        hidden: &Var<R, E>,
        pointed: Option<&(Var<R, E>, Var<R, E>)>,
        batch: usize,
        seq: usize,
    ) -> Result<PolicyOutput<R, E>> {
        let hidden = self.norm.apply(hidden)?;
        let logits = match (&self.actor, pointed) {
            (ActionHead::Flat(actor), _) => actor.apply(&hidden)?,
            (ActionHead::Pointer(head), Some((entities, presence))) => {
                head.apply(&hidden, entities, presence)?
            }
            (ActionHead::Pointer(_), None) => {
                return Err(Error::config(
                    "the pointer head's entity set was not encoded".to_string(),
                ));
            }
        };
        Ok(PolicyOutput {
            logits,
            value: self.critic.apply(&hidden)?.reshape(vec![batch, seq])?,
        })
    }

    /// One rollout step for `B` environments.
    ///
    /// `obs` is `[envs, 1, obs_dim]` and `reset` an optional `[envs]` mask holding
    /// `1` for an environment whose previous episode ended. The state buffer is
    /// advanced in place; nothing leaves the device and no host synchronisation
    /// happens, so the whole step is one queue of kernel launches.
    ///
    /// Runs with the tape disabled — a rollout is data collection, and a cache
    /// that kept a graph alive would grow without bound.
    pub fn step(
        &self,
        obs: &Var<R, E>,
        state: &mut Mamba3StateBuffer<R, E>,
        reset: Option<&Tensor<R, E>>,
    ) -> Result<PolicyOutput<R, E>> {
        let _guard = crate::autograd::no_grad();
        let (batch, seq) = self.check_obs(obs)?;
        if seq != 1 {
            return Err(Error::shape(format!(
                "step() takes a single observation per environment; got {seq} positions. \
                 Use forward() for a window"
            )));
        }
        if batch != state.envs() {
            return Err(Error::shape(format!(
                "observation is for {batch} environments but the state buffer holds {}",
                state.envs()
            )));
        }
        if state.len() != self.blocks.len() {
            return Err(Error::shape(format!(
                "state buffer has {} layers, the policy has {}",
                state.len(),
                self.blocks.len()
            )));
        }

        let Encoded { mut x, pointed } = self.input.apply(obs)?;
        for (i, block) in self.blocks.iter().enumerate() {
            let (out, cache) = block.step_masked(&x, state.layer(i)?, reset)?;
            state.store(i, cache)?;
            x = out;
        }
        self.heads(&x, pointed.as_ref(), batch, 1)
    }

    /// A whole rollout window, through the parallel scan.
    ///
    /// `obs` is `[batch, seq, obs_dim]` and `reset` an optional `[batch, seq]`
    /// mask marking the positions that begin a new episode. `initial` continues
    /// from a state the rollout left behind — pass
    /// [`Mamba3StateBuffer::snapshot`] taken before the rollout began, so the
    /// gradient sees the same history the actor did.
    ///
    /// This is the differentiable path: the result carries a tape, and
    /// [`crate::autograd::Var::backward`] on a loss built from it produces the
    /// reverse scan over the whole window.
    ///
    /// The second return value is the state at the window's right edge, present
    /// only when `initial` was — truncated backpropagation through time hands it
    /// to the next window. It is on the tape, so
    /// [`MixerCache::detach`] it before keeping it, or the graph
    /// stays alive for as long as the state does.
    #[allow(clippy::type_complexity)] // One output plus an optional per-layer state.
    pub fn forward(
        &self,
        obs: &Var<R, E>,
        reset: Option<&Tensor<R, E>>,
        initial: Option<&[MixerCache<R, E>]>,
    ) -> Result<(PolicyOutput<R, E>, Option<Vec<MixerCache<R, E>>>)> {
        let (batch, seq) = self.check_obs(obs)?;
        if let Some(reset) = reset
            && reset.len() != batch * seq
        {
            return Err(Error::shape(format!(
                "reset mask must be [batch, seq] = [{batch}, {seq}], got {}",
                reset.shape()
            )));
        }
        if let Some(initial) = initial
            && initial.len() != self.blocks.len()
        {
            return Err(Error::shape(format!(
                "initial state has {} layers, the policy has {}",
                initial.len(),
                self.blocks.len()
            )));
        }

        let Encoded { mut x, pointed } = self.input.apply(obs)?;
        let mut ends = Vec::with_capacity(self.blocks.len());
        for (i, block) in self.blocks.iter().enumerate() {
            let cache = initial.map(|c| &c[i]);
            let (out, end) = block.apply_with_state_masked(&x, cache, reset)?;
            if let Some(end) = end {
                ends.push(end);
            }
            x = out;
        }
        let ends = (ends.len() == self.blocks.len()).then_some(ends);
        Ok((self.heads(&x, pointed.as_ref(), batch, seq)?, ends))
    }
}

impl<R: Runtime, E: FloatElem> Module<R, E> for Mamba3Policy<R, E> {
    fn visit(&self, visitor: &mut ModuleVisitor<'_, R, E>) {
        match &self.input {
            InputStage::Flat(encoder) => visitor.child("encoder", encoder),
            InputStage::Entities(stage) => stage.visit(visitor),
        }
        for (i, block) in self.blocks.iter().enumerate() {
            visitor.child_at("blocks", i, block);
        }
        visitor.child("norm", &self.norm);
        match &self.actor {
            ActionHead::Flat(actor) => visitor.child("actor", actor),
            ActionHead::Pointer(head) => visitor.child("actor", head),
        }
        visitor.child("critic", &self.critic);
    }
}
