//! Actor-critic PPO over the entity model (ENTITY_RL_PLAN.md R1).
//!
//! The entity model becomes the actor: [`EntityActorCritic::act`] samples a
//! `K`-step plan for every query instead of taking it greedily, and
//! [`entity_ppo_objective`] re-scores stored plans by teacher forcing so PPO
//! can update the actor from rewards an external environment supplies.
//!
//! The read budget is the point of the whole file: [`EntityActorCritic::act`]
//! never calls [`EntityModel::predict`](crate::models::entity::model::EntityModel::predict),
//! which reads the chosen ids back to the host after every step. It runs its
//! own decode loop on device tensors only and hands everything back in one
//! [`read_all`](crate::tensor::ops::index::read_all); the PPO train step
//! likewise reads nothing until the caller reads the statistics.

use std::cell::RefCell;
use std::collections::BTreeMap;

use cubecl::prelude::Runtime;

use crate::autograd::Var;
use crate::backend::{Device, FloatElem};
use crate::distributions::{Categorical, Distribution};
use crate::error::{Error, Result};
use crate::models::entity::batch::{EntityBatch, EntityDataset, HostArrays};
use crate::models::entity::model::{ChoiceIds, EntityModel, fused_entity_model};
use crate::models::entity::spec::{DecoderMode, EntityModelSpec, HeadKind, StepSelection};
use crate::nn::linear::{Linear, LinearConfig};
use crate::nn::module::{Module, ModuleVisitor};
use crate::nn::param::Param;
use crate::tensor::Tensor;
use crate::tensor::ops::entity_model::IGNORE;
use crate::tensor::ops::index::{self, IdTensor};
use crate::tensor::ops::{elemwise, fused, movement, reduce};
use crate::train::checkpoint::Checkpoint;
use crate::train::trainer::TrainStep;

/// Refuse the specs (and paths) the device loop cannot reproduce faithfully.
///
/// Step `j` re-scored by teacher forcing sees the stored choices below `j`,
/// which is what it saw when sampled, only while step `j` cannot see steps
/// `> j` (true of `StepCausal`, and of every mode without
/// `autoregressive_on`) and while one decoder pass covers every query at a
/// step (false of `QueryCausal`, which decodes query-major). The composed
/// path is refused because it funnels every step's ids through the host.
pub fn check_rl_support(spec: &EntityModelSpec) -> Result<()> {
    if !fused_entity_model() {
        return Err(Error::config(
            "entity RL needs the fused on-device path (fused_entity_model()); \
             the composed path reads the chosen ids back to the host after every step"
                .to_string(),
        ));
    }
    if spec.queries.is_none() {
        return Err(Error::config(
            "entity RL needs a query set; the spec has none".to_string(),
        ));
    }
    let autoregressive = spec
        .queries
        .as_ref()
        .and_then(|q| q.autoregressive_on.as_ref())
        .is_some();
    match &spec.decoder {
        DecoderMode::Joint if autoregressive => {
            return Err(Error::config(
                "entity RL refuses DecoderMode::Joint with queries.autoregressive_on: \
                 step j would see later steps' stored choices, which did not exist when \
                 it was sampled, so teacher forcing would not reproduce the rollout"
                    .to_string(),
            ));
        }
        DecoderMode::QueryCausal => {
            return Err(Error::config(
                "entity RL refuses DecoderMode::QueryCausal: the device loop decodes \
                 one step for all queries at a time, which cannot reproduce \
                 query-major decoding"
                    .to_string(),
            ));
        }
        _ => {}
    }
    Ok(())
}

/// The critic: `Linear(d, d) -> GELU -> Linear(d, 1)` over the
/// presence-weighted mean of the encoder output plus the embedded globals.
///
/// It scores the observation alone — never a sampled plan — so one value
/// covers every `(query, step)` cell of the sample in every decoder mode.
pub struct EntityValueHead<R: Runtime, E: FloatElem> {
    lin1: Linear<R, E>,
    lin2: Linear<R, E>,
}

impl<R: Runtime, E: FloatElem> EntityValueHead<R, E> {
    /// Fresh weights on `device` (`d` is the spec's `d_model`).
    pub fn init(
        d: usize,
        device: &Device<R>,
        rng: &mut crate::tensor::ops::random::Rng,
    ) -> Result<Self> {
        if d == 0 {
            return Err(Error::config(
                "entity value head needs d_model > 0".to_string(),
            ));
        }
        Ok(Self {
            lin1: LinearConfig::new(d, d).init(device, rng),
            lin2: LinearConfig::new(d, 1).init(device, rng),
        })
    }

    /// `ctx` is `[B, N_ctx, d]` encoder output, `presence` `[B, N_ctx]` 0/1,
    /// `globals` the embedded `[B, d]` globals (`None` when `G = 0`).
    /// Returns `[B]`. An all-absent context pools to the zero vector (its
    /// count is floored at one) rather than dividing by zero.
    pub fn forward(
        &self,
        ctx: &Var<R, E>,
        presence: &Tensor<R, E>,
        globals: Option<&Var<R, E>>,
    ) -> Result<Var<R, E>> {
        let dims = ctx.shape().dims().to_vec();
        if dims.len() != 3 {
            return Err(Error::shape(format!(
                "entity value head needs ctx [B, N, d], got {}",
                ctx.shape()
            )));
        }
        let (b, n, d) = (dims[0], dims[1], dims[2]);
        if presence.len() != b * n {
            return Err(Error::shape(format!(
                "entity value head needs presence [B, N] for ctx [B={b}, N={n}], got {}",
                presence.shape()
            )));
        }
        let gate = Var::constant(presence.reshape(vec![b, n, 1])?).expand(vec![b, n, d])?;
        let sum = ctx.mul(&gate)?.sum_dim(1)?.reshape(vec![b, d])?;
        let count = Var::constant(presence.clone()).sum_dim(1)?;
        let floor = Var::constant(Tensor::ones(vec![b.max(1), 1], ctx.device()));
        let mean = if b == 0 {
            sum.div(&count.maximum(&floor)?.expand(vec![1, d])?)?
        } else {
            sum.div(&count.maximum(&floor)?.expand(vec![b, d])?)?
        };
        let pooled = match globals {
            Some(g) => mean.add(g)?,
            None => mean,
        };
        self.lin2
            .apply(&self.lin1.apply(&pooled)?.gelu()?)?
            .reshape(vec![b])
    }
}

impl<R: Runtime, E: FloatElem> Module<R, E> for EntityValueHead<R, E> {
    fn visit(&self, visitor: &mut ModuleVisitor<'_, R, E>) {
        visitor.child("lin1", &self.lin1);
        visitor.child("lin2", &self.lin2);
    }
}

/// The actor plus its critic: an [`EntityModel`] and an [`EntityValueHead`].
pub struct EntityActorCritic<R: Runtime, E: FloatElem> {
    model: EntityModel<R, E>,
    value: EntityValueHead<R, E>,
}

impl<R: Runtime, E: FloatElem> EntityActorCritic<R, E> {
    /// Fresh weights from `spec` (the spec's seed drives both halves in order,
    /// so the build is reproducible).
    pub fn init(spec: &EntityModelSpec, device: &Device<R>) -> Result<Self> {
        check_rl_support(spec)?;
        let mut rng = crate::tensor::ops::random::Rng::seeded(spec.seed);
        let model = EntityModel::init_with_rng(spec, device, &mut rng)?;
        let value = EntityValueHead::init(spec.d_model, device, &mut rng)?;
        Ok(Self { model, value })
    }

    /// Keep a behaviour-cloned planner as the actor, with a fresh value head
    /// (seeded off the spec seed so two conversions of the same model agree).
    pub fn from_model(model: EntityModel<R, E>) -> Result<Self> {
        check_rl_support(model.spec())?;
        let d = model.spec().d_model;
        let device = model
            .parameters()
            .first()
            .map(|p| p.value().device().clone())
            .ok_or_else(|| {
                Error::config(
                    "entity actor-critic needs a model with parameters to place the value head"
                        .to_string(),
                )
            })?;
        let mut rng = crate::tensor::ops::random::Rng::seeded(
            model.spec().seed.wrapping_add(0x9E3779B97F4A7C15),
        );
        let value = EntityValueHead::init(d, &device, &mut rng)?;
        Ok(Self { model, value })
    }

    /// The actor.
    pub fn model(&self) -> &EntityModel<R, E> {
        &self.model
    }

    /// The critic.
    pub fn value_head(&self) -> &EntityValueHead<R, E> {
        &self.value
    }

    /// Save both halves plus the spec as one checkpoint.
    pub fn save(&self, path: impl AsRef<std::path::Path>, step: u64) -> Result<()> {
        Checkpoint::capture(self, step)
            .with_metadata(serde_json::to_value(self.model.spec())?)
            .save(path)
    }

    /// Rebuild from a checkpoint's metadata and restore its weights.
    pub fn load(path: impl AsRef<std::path::Path>, device: &Device<R>) -> Result<Self> {
        let ckpt = Checkpoint::load(path)?;
        let spec: EntityModelSpec = serde_json::from_value(ckpt.metadata.clone())?;
        let ac = Self::init(&spec, device)?;
        ckpt.restore(&ac, true)?;
        Ok(ac)
    }
}

impl<R: Runtime, E: FloatElem> Module<R, E> for EntityActorCritic<R, E> {
    fn visit(&self, visitor: &mut ModuleVisitor<'_, R, E>) {
        visitor.child("model", &self.model);
        visitor.child("value", &self.value);
    }
}

/// One action head the decode loop samples: every pointer head and every
/// categorical head, at the steps its [`StepSelection`] covers.
struct ActionHead {
    name: String,
    ptr: Option<usize>,
    classes: usize,
    first_only: bool,
}

fn action_heads<R: Runtime, E: FloatElem>(ac: &EntityActorCritic<R, E>) -> Result<Vec<ActionHead>> {
    let spec = ac.model.spec();
    let mut out = Vec::new();
    for h in &spec.heads {
        let first_only = matches!(h.steps, StepSelection::First);
        match &h.kind {
            HeadKind::Pointer { set, extra_actions } => {
                let n = spec
                    .context
                    .iter()
                    .find(|s| &s.name == set)
                    .map(|s| s.count)
                    .ok_or_else(|| {
                        Error::config(format!(
                            "entity RL head {:?} names no context set {set:?}",
                            h.name
                        ))
                    })?;
                let ptr = ac
                    .model
                    .ptr_heads()
                    .iter()
                    .position(|p| p.name == h.name)
                    .ok_or_else(|| {
                        Error::config(format!("entity RL pointer head {:?} is missing", h.name))
                    })?;
                out.push(ActionHead {
                    name: h.name.clone(),
                    ptr: Some(ptr),
                    classes: n + extra_actions,
                    first_only,
                });
            }
            HeadKind::Categorical { classes } => out.push(ActionHead {
                name: h.name.clone(),
                ptr: None,
                classes: *classes,
                first_only,
            }),
            HeadKind::MultiLabel { .. } | HeadKind::Regression { .. } => {}
        }
    }
    if out.is_empty() {
        return Err(Error::config(
            "entity RL needs at least one pointer or categorical head; the spec has none"
                .to_string(),
        ));
    }
    Ok(out)
}

/// [`EntityActorCritic::evaluate_actions`]: per-cell log-probabilities and
/// entropies `[B, M, K]`, and the critic's values `[B]`, all on the tape.
pub type Scored<R, E> = (Var<R, E>, Var<R, E>, Var<R, E>);

/// Device tensors of one [`EntityActorCritic::act`] call.
pub struct Acted<R: Runtime, E: FloatElem> {
    /// Sampled ids per action head: `[B, M, K]` (`IGNORE` = no action).
    pub actions: BTreeMap<String, IdTensor<R>>,
    /// Summed tempered log-probabilities over the action heads: `[B, M, K]`.
    pub cell_log_prob: Tensor<R, E>,
    /// Summed tempered entropies over the action heads: `[B, M, K]`.
    pub cell_entropy: Tensor<R, E>,
    /// 1 where an action head acted: `[B, M, K]` (built on the device, so the
    /// update's mask never depends on host logic re-deriving it).
    pub cell_mask: Tensor<R, E>,
    /// The critic's estimate: `[B]`.
    pub value: Tensor<R, E>,
    /// Logits of every head: `[B, M, K, width]` (`[B, M, 1, width]` for
    /// `First` heads).
    pub outputs: BTreeMap<String, Tensor<R, E>>,
    b: usize,
}

/// Host copies of [`Acted`], from one [`read_all`](crate::tensor::ops::index::read_all).
pub struct ActedHost {
    /// Sampled ids per action head: `[B*M*K]`, `-1` = no action.
    pub actions: BTreeMap<String, Vec<i64>>,
    /// Per-cell log-probabilities: `[B*M*K]`.
    pub cell_log_prob: Vec<f32>,
    /// Per-cell entropies: `[B*M*K]`.
    pub cell_entropy: Vec<f32>,
    /// Acted-cell mask: `[B*M*K]`, 1 where an action head acted.
    pub cell_mask: Vec<f32>,
    /// The critic's estimates: `[B]`.
    pub value: Vec<f32>,
    /// Logits per head, with shapes.
    pub outputs: BTreeMap<String, (Vec<usize>, Vec<f32>)>,
    b: usize,
}

impl<R: Runtime, E: FloatElem> Acted<R, E> {
    /// Read everything back under one synchronisation.
    pub fn read(&self) -> Result<ActedHost> {
        let ids: Vec<&IdTensor<R>> = self.actions.values().collect();
        let mut floats: Vec<&Tensor<R, E>> = vec![
            &self.cell_log_prob,
            &self.cell_entropy,
            &self.cell_mask,
            &self.value,
        ];
        floats.extend(self.outputs.values());
        let (id_vecs, float_vecs) = index::read_all(&ids, &floats)?;
        let mut actions = BTreeMap::new();
        for (name, v) in self.actions.keys().zip(id_vecs) {
            actions.insert(
                name.clone(),
                v.into_iter()
                    .map(|id| if id == IGNORE { -1 } else { id as i64 })
                    .collect(),
            );
        }
        let mut rest = float_vecs.into_iter();
        let mut take = || {
            rest.next().ok_or_else(|| {
                Error::shape("entity act read back fewer tensors than it queued".to_string())
            })
        };
        let cell_log_prob = take()?;
        let cell_entropy = take()?;
        let cell_mask = take()?;
        let value = take()?;
        let mut outputs = BTreeMap::new();
        for (name, t) in &self.outputs {
            outputs.insert(name.clone(), (t.shape().dims().to_vec(), take()?));
        }
        Ok(ActedHost {
            actions,
            cell_log_prob,
            cell_entropy,
            cell_mask,
            value,
            outputs,
            b: self.b,
        })
    }
}

/// Sample one step's slice of a `[rows, classes]` logit tensor: the ids plus
/// the raw tempered log-probabilities and entropies (`[rows]`). Presence
/// gating happens in the caller's
/// [`accumulate_draw`](crate::tensor::ops::entity_model::accumulate_draw),
/// not here. `T = 0` is greedy on the device, with zero entropy.
/// One step's draw: ids `[rows]`, tempered log-probabilities and entropies `[rows]`.
type Draw<R, E> = (IdTensor<R>, Tensor<R, E>, Tensor<R, E>);

fn sample_step<R: Runtime, E: FloatElem>(
    logits: &Tensor<R, E>,
    temperature: f32,
    seed: u64,
) -> Result<Draw<R, E>> {
    let (ids, log_prob) = crate::tensor::ops::rl::sample_categorical(logits, temperature, seed)?;
    let entropy = if temperature == 0.0 {
        Tensor::zeros(logits.shape().without(logits.rank() - 1), logits.device())
    } else {
        let tempered = elemwise::mul_scalar(logits, 1.0 / temperature);
        Categorical::from_logits(Var::constant(tempered))?.entropy()?.tensor().clone()
    };
    Ok((ids, log_prob, entropy))
}

impl<R: Runtime, E: FloatElem> EntityActorCritic<R, E> {
    /// Sample a plan for every query, fully on the device.
    ///
    /// The encoder stem runs once, then each step decodes from the device
    /// choice tables, draws every action head covering the step with
    /// [`sample_categorical`](crate::tensor::ops::rl::sample_categorical) and
    /// accumulates the tempered per-cell log-probabilities and entropies.
    /// Nothing leaves the device until [`Acted::read`]: in particular absent
    /// queries are set to `IGNORE` and the per-cell masks are built by the
    /// [`mask_absent_ids`](crate::tensor::ops::entity_model::mask_absent_ids)
    /// kernel, not on the host. `temperature = 0` is greedy (device argmax).
    /// Each draw takes a distinct seed off `seed`, so the call is reproducible.
    pub fn act(
        &self,
        batch: &EntityBatch<R, E>,
        temperature: f32,
        seed: u64,
    ) -> Result<Acted<R, E>> {
        let _guard = crate::autograd::no_grad();
        check_rl_support(self.model.spec())?;
        if temperature < 0.0 {
            return Err(Error::config(format!(
                "entity act temperature must not be negative, got {temperature}"
            )));
        }
        let spec = self.model.spec();
        let q = spec.queries.as_ref().ok_or_else(|| {
            Error::config("entity act needs a query set; the spec has none".to_string())
        })?;
        let (b, m, k) = (batch.b, q.count, q.steps);
        let heads = action_heads(self)?;
        let rows = b * m;
        let device = batch
            .ctx_feats
            .first()
            .map(|t| t.device().clone())
            .ok_or_else(|| {
                Error::shape("entity act needs at least one context set".to_string())
            })?;

        // Encoder stem, once, under no_grad.
        let feats: Vec<Var<R, E>> = batch
            .ctx_feats
            .iter()
            .map(|t| Var::constant(t.clone()))
            .collect();
        let cpresence: Vec<Var<R, E>> = batch
            .ctx_presence
            .iter()
            .map(|t| Var::constant(t.clone()))
            .collect();
        let globals = batch.globals.as_ref().map(|g| Var::constant(g.clone()));
        let ctx = self.model.encode(&feats, &cpresence, globals.as_ref())?;
        let qf = Var::constant(
            batch
                .q_feats
                .as_ref()
                .ok_or_else(|| Error::config("entity act needs query features".to_string()))?
                .clone(),
        );
        let qp = Var::constant(
            batch
                .q_presence
                .as_ref()
                .ok_or_else(|| Error::config("entity act needs query presence".to_string()))?
                .clone(),
        );
        let u = self.model.query_base(&qf, &qp)?;
        let g = self.model.global_embed(globals.as_ref())?;
        let anchor_tok = if q.anchor.is_some() {
            match (&batch.anchor_dev, &batch.anchor_ids) {
                (Some(dev), _) => Some(self.model.anchor_tokens_ids(dev, b, m, &ctx)?),
                // Host ids upload the same values without a read; the decode
                // itself still never leaves the device.
                (None, Some(ids)) => Some(self.model.anchor_tokens(ids, b, m, &ctx)?),
                (None, None) => {
                    return Err(Error::shape(
                        "entity act needs anchor ids (queries.anchor is set)".to_string(),
                    ));
                }
            }
        } else {
            None
        };
        // The critic scores the same stem's encoder output.
        let ctx_presence = movement::cat(&batch.ctx_presence, 1)?;
        let value = self
            .value
            .forward(&ctx, &ctx_presence, g.as_ref())?
            .tensor()
            .clone();

        // Per-head `[rows, K]` id tables (IGNORE until sampled) and the
        // `[rows, K]` float accumulators.
        let mut tables: BTreeMap<String, IdTensor<R>> = BTreeMap::new();
        for a in &heads {
            tables.insert(
                a.name.clone(),
                IdTensor::from_slice(&vec![IGNORE; rows * k], vec![rows, k], &device)?,
            );
        }
        let cell_lp = Tensor::zeros(vec![rows, k], &device);
        let cell_ent = Tensor::zeros(vec![rows, k], &device);
        // Per-head `[B, M, 1, W]` logit slices, in step order, for `outputs`.
        let mut slices: BTreeMap<String, Vec<Tensor<R, E>>> = BTreeMap::new();
        for h in &spec.heads {
            slices.insert(h.name.clone(), Vec::new());
        }
        let qpres = batch
            .q_presence
            .as_ref()
            .ok_or_else(|| Error::config("entity act needs query presence".to_string()))?;
        let presence = qpres.reshape(vec![rows])?;

        // One seed stream for the call: draw (step, head) counts off it so
        // every draw has a distinct seed and the call stays reproducible.
        let mut seed_ctr = seed;
        let mut next_seed = || {
            seed_ctr = seed_ctr.wrapping_add(0x9E3779B97F4A7C15);
            seed_ctr
        };
        // Each pointer head's presence / legal mask, built once: it depends
        // only on the batch, never on the decoded step.
        let ptr_masks = self.model.pointer_masks_for_act(batch, b, m, k, &device)?;
        // Without an autoregressive plan head every pass decodes the same
        // queries, so one pass serves all steps; otherwise step j decodes
        // from the choices below j, like greedy predict.
        let passes: Vec<Option<usize>> = match self.model.plan_ptr() {
            Some(_) => (0..k).map(Some).collect(),
            None => vec![None],
        };
        for pass in &passes {
            // Device choice tables in the shape the heads take them.
            let mut dev_map: BTreeMap<String, IdTensor<R>> = BTreeMap::new();
            for (name, t) in &tables {
                dev_map.insert(name.clone(), t.reshape(vec![b, m, k])?);
            }
            let plan_flat = match &q.autoregressive_on {
                Some(name) => match dev_map.get(name) {
                    Some(t) => t.reshape(vec![rows * k])?,
                    None => {
                        return Err(Error::shape(format!(
                            "entity act needs the plan head {name:?} among the action heads"
                        )));
                    }
                },
                None => IdTensor::from_slice(&vec![IGNORE; rows * k], vec![rows * k], &device)?,
            };
            let prev = self.model.prev_tokens_ids(&plan_flat, b, m, k, &ctx)?;
            let queries =
                self.model
                    .build_queries(&u, anchor_tok.as_ref(), g.as_ref(), &prev, None)?;
            let dec = self.model.decode(&ctx, &queries)?;
            // Computed once per pass and shared: the pointer sampling below
            // and `head_logits_with` both read these, so neither recomputes
            // the query / key projections, the matmuls or the masks.
            let ptr_logits = self.model.pointer_logits_with_masks(&dec, &ptr_masks)?;
            // Pointer heads first: the conditioned heads below read the
            // freshly drawn choices at the same step.
            let steps_here: Vec<usize> = match pass {
                Some(j) => vec![*j],
                None => (0..k).collect(),
            };
            for a in heads.iter().filter(|a| a.ptr.is_some()) {
                let full = ptr_logits.get(&a.name).ok_or_else(|| {
                    Error::config(format!("entity act is missing pointer logits for {:?}", a.name))
                })?;
                let w = full.shape().dim(3);
                if w != a.classes {
                    return Err(Error::shape(format!(
                        "entity act head {:?} has width {w}, expected {}",
                        a.name, a.classes
                    )));
                }
                for &j in &steps_here {
                    if a.first_only && j > 0 {
                        continue;
                    }
                    let flat = full.slice(2, j, 1)?.tensor().reshape(vec![rows, w])?;
                    let (ids, lp, ent) = sample_step(&flat, temperature, next_seed())?;
                    let table = tables.get(&a.name).ok_or_else(|| {
                        Error::config(format!("entity act lost the table for {:?}", a.name))
                    })?;
                    crate::tensor::ops::rl::write_step_ids(table, &ids, j)?;
                    crate::tensor::ops::entity_model::accumulate_draw(
                        &cell_lp, &cell_ent, &lp, &ent, &presence, j,
                    )?;
                    // The conditioned heads of this step read this choice.
                    dev_map.insert(a.name.clone(), tables[&a.name].reshape(vec![b, m, k])?);
                }
            }
            // Conditioned / unconditioned head logits over the pointer logits
            // above: no pointer recompute, no device argmax.
            let out = self.model.head_logits_with(
                &dec,
                batch,
                ChoiceIds::Device(&dev_map),
                ptr_logits.clone(),
            )?;
            for a in heads.iter().filter(|a| a.ptr.is_none()) {
                let full = out.get(&a.name).ok_or_else(|| {
                    Error::config(format!("entity act is missing logits for {:?}", a.name))
                })?;
                let w = full.shape().dim(3);
                if w != a.classes {
                    return Err(Error::shape(format!(
                        "entity act head {:?} has width {w}, expected {}",
                        a.name, a.classes
                    )));
                }
                for &j in &steps_here {
                    if a.first_only && j > 0 {
                        continue;
                    }
                    let jj = if matches!(
                        spec.head(&a.name).map(|h| h.steps),
                        Some(StepSelection::First)
                    ) {
                        0
                    } else {
                        j
                    };
                    let flat = full.slice(2, jj, 1)?.tensor().reshape(vec![rows, w])?;
                    let (ids, lp, ent) = sample_step(&flat, temperature, next_seed())?;
                    let table = tables.get(&a.name).ok_or_else(|| {
                        Error::config(format!("entity act lost the table for {:?}", a.name))
                    })?;
                    crate::tensor::ops::rl::write_step_ids(table, &ids, j)?;
                    crate::tensor::ops::entity_model::accumulate_draw(
                        &cell_lp, &cell_ent, &lp, &ent, &presence, j,
                    )?;
                }
            }
            // Step slices of every head's logits for `outputs`.
            for h in &spec.heads {
                let full = out.get(&h.name).ok_or_else(|| {
                    Error::config(format!("entity act is missing logits for {:?}", h.name))
                })?;
                let entry = slices.get_mut(&h.name).ok_or_else(|| {
                    Error::config(format!("entity act lost the slices for {:?}", h.name))
                })?;
                match h.steps {
                    StepSelection::All => {
                        for &j in &steps_here {
                            entry.push(full.slice(2, j, 1)?.tensor().clone());
                        }
                    }
                    StepSelection::First => {
                        if steps_here.contains(&0) {
                            entry.push(full.tensor().clone());
                        }
                    }
                }
            }
        }
        // Absent queries take no action: IGNORE plus a zero acted flag, on
        // the device. The flat presence broadcasts `[rows]` to `[rows*K]`.
        let presence_flat =
            elemwise::expand(&presence.reshape(vec![rows, 1])?, &vec![rows, k].into())?
                .reshape(vec![rows * k])?;
        let mut actions: BTreeMap<String, IdTensor<R>> = BTreeMap::new();
        let mut mask_sum: Option<Tensor<R, E>> = None;
        for a in &heads {
            let flat = tables[&a.name].reshape(vec![rows * k])?;
            let (masked, acted) =
                crate::tensor::ops::entity_model::mask_absent_ids(&flat, &presence_flat)?;
            actions.insert(a.name.clone(), masked.reshape(vec![b, m, k])?);
            let acted = acted.reshape(vec![b, m, k])?;
            mask_sum = Some(match mask_sum {
                Some(s) => elemwise::add(&s, &acted)?,
                None => acted,
            });
        }
        let cell_mask = elemwise::clamp(
            &mask_sum.ok_or_else(|| Error::config("entity act sampled no heads".to_string()))?,
            0.0,
            1.0,
        );
        let mut outputs = BTreeMap::new();
        for (name, parts) in &slices {
            outputs.insert(
                name.clone(),
                if parts.len() == 1 {
                    parts[0].clone()
                } else {
                    movement::cat(parts, 2)?
                },
            );
        }
        Ok(Acted {
            actions,
            cell_log_prob: cell_lp.reshape(vec![b, m, k])?,
            cell_entropy: cell_ent.reshape(vec![b, m, k])?,
            cell_mask,
            value,
            outputs,
            b,
        })
    }

    /// Re-score stored actions under the current weights, by teacher forcing.
    ///
    /// One training forward with the actions as `label.<head>`: for a decoder
    /// in which step `j` cannot see steps `> j` this reproduces the sampling
    /// logits exactly, so the PPO ratio starts at 1. Per-cell log-probability
    /// is the sum over action heads of `log softmax(logits / T)` at the
    /// stored id (`T = 0` scores untempered); cells whose label is absent
    /// contribute 0, as do their entropies. The value comes from the value
    /// head on the same forward's encoder output. The batch must carry
    /// `label.<head>` for the action heads (see [`actions_to_arrays`]).
    pub fn evaluate_actions(
        &self,
        batch: &EntityBatch<R, E>,
        temperature: f32,
    ) -> Result<Scored<R, E>> {
        check_rl_support(self.model.spec())?;
        if temperature < 0.0 {
            return Err(Error::config(format!(
                "entity scoring temperature must not be negative, got {temperature}"
            )));
        }
        let spec = self.model.spec();
        let q = spec.queries.as_ref().ok_or_else(|| {
            Error::config("entity scoring needs a query set; the spec has none".to_string())
        })?;
        let (b, m, k) = (batch.b, q.count, q.steps);
        let inv = if temperature == 0.0 { 1.0 } else { 1.0 / temperature };
        let (dec, _, enc_ctx, g) = self.model.train_decode_with_encoder(batch, false)?;
        let out = self.model.heads(&dec, batch, ChoiceIds::Device(&batch.choice_dev))?;
        let qp = batch
            .q_presence
            .as_ref()
            .ok_or_else(|| Error::config("entity scoring needs query presence".to_string()))?;
        let device = dec.h.device().clone();
        let mut lp_sum = Var::constant(Tensor::zeros(vec![b, m, k], &device));
        let mut ent_sum = Var::constant(Tensor::zeros(vec![b, m, k], &device));
        for h in &spec.heads {
            match &h.kind {
                HeadKind::Pointer { .. } | HeadKind::Categorical { .. } => {}
                HeadKind::MultiLabel { .. } | HeadKind::Regression { .. } => continue,
            }
            let logits = out.logits.get(&h.name).ok_or_else(|| {
                Error::shape(format!("entity scoring is missing logits for {:?}", h.name))
            })?;
            let dims = logits.shape().dims().to_vec();
            let (kj, w) = (dims[2], dims[3]);
            let labels = batch
                .labels
                .get(&h.name)
                .and_then(|l| l.as_ref())
                .ok_or_else(|| Error::shape(format!("entity scoring needs label.{}", h.name)))?;
            let (safe_ids, keep) = match labels {
                crate::models::entity::batch::HeadLabels::Class { ids, keep, .. } => (ids, keep),
                _ => {
                    return Err(Error::shape(format!(
                        "entity scoring needs class label.{} for an action head",
                        h.name
                    )));
                }
            };
            // Label ids already map absent labels to 0 (a valid row), so the
            // gather stays in bounds; the keep-weighted mask below zeroes
            // those cells back out. Binarising keeps step-weighted labels
            // acted: a step weight of exactly 0 reads as no action, as in the
            // supervised loss, which skips it too. ANDing the query presence
            // keeps a labelled-but-absent cell (a caller-side contract breach)
            // from silently counting as acted.
            let pres_flat =
                elemwise::expand(&qp.reshape(vec![b * m, 1])?, &vec![b * m, kj.max(1)].into())?
                    .reshape(vec![b * m * kj])?;
            let acted_t = elemwise::mul(&elemwise::gt_scalar(keep, 0.0), &pres_flat)?;
            let tempered = logits.mul_scalar(inv).reshape(vec![b * m * kj, w])?;
            let dist = Categorical::from_logits(tempered)?;
            let score = dist.log_prob_ids(safe_ids)?.reshape(vec![b, m, kj])?;
            let ent = dist.entropy()?.reshape(vec![b, m, kj])?;
            let acted = Var::constant(acted_t).reshape(vec![b, m, kj])?;
            let (score, ent, acted) = match h.steps {
                StepSelection::All => (score, ent, acted),
                StepSelection::First => {
                    if k > 1 {
                        let z = Var::constant(Tensor::zeros(vec![b, m, k - 1], &device));
                        (
                            crate::autograd::cat(&[score, z.clone()], 2)?,
                            crate::autograd::cat(&[ent, z.clone()], 2)?,
                            crate::autograd::cat(&[acted, z], 2)?,
                        )
                    } else {
                        (score, ent, acted)
                    }
                }
            };
            lp_sum = lp_sum.add(&score.mul(&acted)?)?;
            ent_sum = ent_sum.add(&ent.mul(&acted)?)?;
        }
        let ctx_presence = movement::cat(&batch.ctx_presence, 1)?;
        let value = self.value.forward(&enc_ctx, &ctx_presence, g.as_ref())?;
        Ok((lp_sum, ent_sum, value))
    }

    /// The critic's values for a batch, on the device: the encoder stem once
    /// plus the value head, without the decoder. No grad, no reads — for the
    /// Python `EntityPolicy.value`, which only needs numbers for one batch.
    pub fn values(&self, batch: &EntityBatch<R, E>) -> Result<Tensor<R, E>> {
        let _guard = crate::autograd::no_grad();
        check_rl_support(self.model.spec())?;
        let feats: Vec<Var<R, E>> = batch
            .ctx_feats
            .iter()
            .map(|t| Var::constant(t.clone()))
            .collect();
        let cpresence: Vec<Var<R, E>> = batch
            .ctx_presence
            .iter()
            .map(|t| Var::constant(t.clone()))
            .collect();
        let globals = batch.globals.as_ref().map(|g| Var::constant(g.clone()));
        let ctx = self.model.encode(&feats, &cpresence, globals.as_ref())?;
        let g = self.model.global_embed(globals.as_ref())?;
        let ctx_presence = movement::cat(&batch.ctx_presence, 1)?;
        Ok(self
            .value
            .forward(&ctx, &ctx_presence, g.as_ref())?
            .tensor()
            .clone())
    }
}

/// The `label.<head>` int arrays for stored actions, so an update batch is
/// built with the ordinary [`EntityBatch::from_host`] path. Cells without an
/// action stay `-1`. Heads the spec does not know, or actions it does not
/// hold, are skipped.
pub fn actions_to_arrays(acted: &ActedHost, spec: &EntityModelSpec) -> HostArrays {
    let mut a = HostArrays::new();
    let q = match &spec.queries {
        Some(q) => q,
        None => return a,
    };
    let (m, k) = (q.count, q.steps);
    for h in &spec.heads {
        match &h.kind {
            HeadKind::Pointer { .. } | HeadKind::Categorical { .. } => {}
            HeadKind::MultiLabel { .. } | HeadKind::Regression { .. } => continue,
        }
        let ids = match acted.actions.get(&h.name) {
            Some(v) => v,
            None => continue,
        };
        match h.steps {
            StepSelection::All => {
                a.insert_int(&format!("label.{}", h.name), vec![acted.b, m, k], ids.clone());
            }
            StepSelection::First => {
                let mut step0 = Vec::with_capacity(acted.b * m);
                for cell in 0..acted.b * m {
                    step0.push(ids.get(cell * k).copied().unwrap_or(-1));
                }
                a.insert_int(&format!("label.{}", h.name), vec![acted.b, m], step0);
            }
        }
    }
    a
}

/// One rollout slice of experience for the PPO update: the update batch
/// (with `label.<head>` = the stored actions), the behaviour policy's
/// per-cell log-probabilities, the acted-cell mask, and the per-sample
/// advantage, return and value estimates.
pub struct EntityPpoBatch<R: Runtime, E: FloatElem> {
    /// Update batch, with the stored actions as labels.
    pub batch: EntityBatch<R, E>,
    /// Behaviour log-probabilities: `[B, M, K]`.
    pub old_log_prob: Tensor<R, E>,
    /// 1 where an action head acted: `[B, M, K]`.
    pub cell_mask: Tensor<R, E>,
    /// Advantage estimates: `[B]`.
    pub advantages: Tensor<R, E>,
    /// λ-returns, the critic's target: `[B]`.
    pub returns: Tensor<R, E>,
    /// Critic estimates made at collection: `[B]`.
    pub old_values: Tensor<R, E>,
}

impl<R: Runtime, E: FloatElem> EntityPpoBatch<R, E> {
    /// Gather a minibatch on the device: `data` is the rollout split uploaded
    /// once (with the actions as labels), the `full_*` tensors are the
    /// rollout's `[S, ..]` auxiliaries, and `ids` names the samples. Uploading
    /// the small id buffer is the only host traffic; nothing is read back.
    #[allow(clippy::too_many_arguments)]
    pub fn from_rollout(
        spec: &EntityModelSpec,
        data: &EntityDataset<R, E>,
        full_log_prob: &Tensor<R, E>,
        full_mask: &Tensor<R, E>,
        full_advantages: &Tensor<R, E>,
        full_returns: &Tensor<R, E>,
        full_values: &Tensor<R, E>,
        ids: &[u32],
    ) -> Result<Self> {
        let b = ids.len();
        if b == 0 {
            return Err(Error::shape(
                "entity PPO minibatch needs at least one sample id".to_string(),
            ));
        }
        let batch = EntityBatch::from_ids(spec, data, ids)?;
        let device = batch
            .ctx_feats
            .first()
            .map(|t| t.device().clone())
            .ok_or_else(|| {
                Error::shape("entity PPO minibatch needs at least one context set".to_string())
            })?;
        let id_buf = IdTensor::from_slice(ids, vec![b], &device)?;
        let q = spec.queries.as_ref().ok_or_else(|| {
            Error::config("entity PPO needs a query set; the spec has none".to_string())
        })?;
        let (m, k) = (q.count, q.steps);
        let gather_3 = |t: &Tensor<R, E>, name: &str| -> Result<Tensor<R, E>> {
            if t.shape().dims() != [data.samples, m, k] {
                return Err(Error::shape(format!(
                    "entity PPO {name} is {}, expected [S={}, M={m}, K={k}]",
                    t.shape(),
                    data.samples
                )));
            }
            crate::tensor::ops::index::gather_rows(&t.reshape(vec![data.samples, m * k])?, &id_buf)?
                .reshape(vec![b, m, k])
        };
        let gather_1 = |t: &Tensor<R, E>, name: &str| -> Result<Tensor<R, E>> {
            if t.len() != data.samples {
                return Err(Error::shape(format!(
                    "entity PPO {name} holds {} elements, expected S={}",
                    t.len(),
                    data.samples
                )));
            }
            crate::tensor::ops::index::gather_rows(&t.reshape(vec![data.samples, 1])?, &id_buf)?
                .reshape(vec![b])
        };
        Ok(Self {
            batch,
            old_log_prob: gather_3(full_log_prob, "log_prob")?,
            cell_mask: gather_3(full_mask, "cell_mask")?,
            advantages: gather_1(full_advantages, "advantages")?,
            returns: gather_1(full_returns, "returns")?,
            old_values: gather_1(full_values, "old_values")?,
        })
    }
}

/// Mean of `x` over the cells `mask` keeps (the weighted mean, so excluding
/// cells changes which terms contribute, not their scale). A mask that keeps
/// nothing reads the numerator's zero over a floor of one.
fn masked_mean<R: Runtime, E: FloatElem>(x: &Var<R, E>, mask: &Var<R, E>) -> Result<Var<R, E>> {
    let kept = mask.sum()?;
    let floor = Var::constant(Tensor::full(kept.shape().clone(), 1.0, kept.device()));
    x.mul(mask)?.sum()?.div(&kept.maximum(&floor)?)
}

/// The PPO terms for one update batch: the per-cell surrogate (advantages
/// broadcast to cells, the masked mean over acted cells — per-token PPO, as
/// in LLM fine-tuning, since a joint ratio over `M * K` cells would leave the
/// trust region on every update), the per-sample clipped value loss, and the
/// masked entropy mean, with the KL and clip diagnostics over acted cells.
/// Reuses [`PpoLoss`](crate::rl::PpoLoss); the reference KL stays zero (R1 has
/// no reference anchor). Nothing is read back.
pub fn entity_ppo_objective<R: Runtime, E: FloatElem>(
    ac: &EntityActorCritic<R, E>,
    batch: &EntityPpoBatch<R, E>,
    config: &crate::rl::PpoConfig,
    temperature: f32,
) -> Result<crate::rl::PpoLoss<R, E>> {
    config.validate()?;
    let (cell_lp, cell_ent, value) = ac.evaluate_actions(&batch.batch, temperature)?;
    let dims = cell_lp.shape().dims().to_vec();
    if dims.len() != 3 {
        return Err(Error::shape(format!(
            "entity PPO needs cell log-probs [B, M, K], got {}",
            cell_lp.shape()
        )));
    }
    let (b, m, k) = (dims[0], dims[1], dims[2]);
    let cells = b * m * k;
    for (name, t) in [
        ("old_log_prob", &batch.old_log_prob),
        ("cell_mask", &batch.cell_mask),
    ] {
        if t.shape().dims() != [b, m, k] {
            return Err(Error::shape(format!(
                "entity PPO {name} is {}, expected [B={b}, M={m}, K={k}]",
                t.shape()
            )));
        }
    }
    for (name, t) in [
        ("advantages", &batch.advantages),
        ("returns", &batch.returns),
        ("old_values", &batch.old_values),
    ] {
        if t.len() != b {
            return Err(Error::shape(format!(
                "entity PPO {name} holds {} elements, expected B={b}",
                t.len()
            )));
        }
    }
    let chosen = cell_lp.reshape(vec![cells])?;
    let old = batch.old_log_prob.reshape(vec![cells])?;
    let adv = if config.normalize_advantages {
        crate::tensor::ops::rl::normalize(&batch.advantages, 1e-8)?
    } else {
        batch.advantages.clone()
    };
    let adv =
        elemwise::expand(&adv.reshape(vec![b, 1, 1])?, &vec![b, m, k].into())?.reshape(vec![cells])?;
    let mask = Var::constant(batch.cell_mask.reshape(vec![cells])?);
    let (surrogate, ratio) = chosen.ppo_surrogate(&old, &adv, config.clip_coeff)?;
    let policy = masked_mean(&surrogate, &mask)?.neg();
    let squared = value.ppo_value_loss(
        &batch.returns,
        &batch.old_values,
        config.clip_coeff,
        config.clip_value_loss,
    )?;
    let value_loss = squared.mean()?.mul_scalar(0.5);
    let entropy_mean = masked_mean(&cell_ent.reshape(vec![cells])?, &mask)?;
    let total = policy
        .add(&value_loss.mul_scalar(config.value_coeff))?
        .sub(&entropy_mean.mul_scalar(config.entropy_coeff))?;
    let (kl_terms, clipped_flags) =
        fused::ppo_diagnostics(chosen.tensor(), &old, &ratio, config.clip_coeff)?;
    let m = batch.cell_mask.reshape(vec![cells])?;
    let kept = elemwise::clamp(&reduce::sum_all(&m)?, 1.0, f32::MAX);
    let approx_kl =
        elemwise::div(&reduce::sum_all(&elemwise::mul(&kl_terms, &m)?)?, &kept)?.reshape(vec![1])?;
    let clip_fraction =
        elemwise::div(&reduce::sum_all(&elemwise::mul(&clipped_flags, &m)?)?, &kept)?
            .reshape(vec![1])?;
    Ok(crate::rl::PpoLoss {
        total,
        policy,
        value: value_loss,
        entropy: entropy_mean,
        approx_kl,
        clip_fraction,
        reference_kl: Tensor::zeros(vec![1], chosen.tensor().device()),
    })
}

/// Time-major GAE for entity rollouts: `rewards`, `values` and `dones` are
/// `[T*E]` time-major (sample `t*E + e`), `last_value` is `[E]`. Returns
/// `(advantages, returns)` in the same time-major sample order. The
/// transposition runs on the device: [`generalized_advantage`](crate::tensor::ops::rl::generalized_advantage)
/// walks `[envs, steps]`, so the rollout is uploaded transposed and reshaped
/// back transposed.
#[allow(clippy::too_many_arguments)]
pub fn entity_gae<R: Runtime, E: FloatElem>(
    rewards: &[f32],
    values: &[f32],
    dones: &[f32],
    last_value: &[f32],
    t: usize,
    e: usize,
    gamma: f32,
    lambda: f32,
    device: &Device<R>,
) -> Result<(Tensor<R, E>, Tensor<R, E>)> {
    if rewards.len() != t * e || values.len() != t * e || dones.len() != t * e {
        return Err(Error::shape(format!(
            "entity GAE needs [T={t}*E={e}] rewards, values and dones; got {}, {} and {}",
            rewards.len(),
            values.len(),
            dones.len()
        )));
    }
    if last_value.len() != e {
        return Err(Error::shape(format!(
            "entity GAE needs [E={e}] last values, got {}",
            last_value.len()
        )));
    }
    // Transpose on the host (no device involved yet): [T, E] to [E, T].
    let transpose_host = |v: &[f32]| {
        let mut out = vec![0.0f32; t * e];
        for tt in 0..t {
            for ee in 0..e {
                out[ee * t + tt] = v[tt * e + ee];
            }
        }
        out
    };
    let rewards_t = Tensor::from_f32(&transpose_host(rewards), vec![e, t], device)?;
    let values_t = Tensor::from_f32(&transpose_host(values), vec![e, t], device)?;
    let dones_t = Tensor::from_f32(&transpose_host(dones), vec![e, t], device)?;
    let bootstrap = Tensor::from_f32(last_value, vec![e], device)?;
    let adv = crate::tensor::ops::rl::generalized_advantage(
        &rewards_t, &values_t, &dones_t, &bootstrap, gamma, lambda,
    )?;
    Ok((
        movement::transpose(&adv.advantages)?.reshape(vec![t * e])?,
        movement::transpose(&adv.returns)?.reshape(vec![t * e])?,
    ))
}

/// Diagnostics of the most recent [`EntityPpoTask::loss`], on the host.
#[derive(Debug, Clone, Copy, Default)]
pub struct EntityPpoStats {
    /// Clipped surrogate objective, negated so lower is better.
    pub policy_loss: f32,
    /// Value loss.
    pub value_loss: f32,
    /// Mean policy entropy over acted cells.
    pub entropy: f32,
    /// Approximate `KL(π_old || π_θ)` over acted cells.
    pub approx_kl: f32,
    /// Fraction of acted cells whose ratio was clipped.
    pub clip_fraction: f32,
}

impl EntityPpoStats {
    /// The values of [`EntityPpoTask::stat_tensors`], read back in that order.
    pub fn from_values(values: [f32; 5]) -> Self {
        let [policy_loss, value_loss, entropy, approx_kl, clip_fraction] = values;
        Self {
            policy_loss,
            value_loss,
            entropy,
            approx_kl,
            clip_fraction,
        }
    }
}

/// A [`TrainStep`] that optimises an [`EntityActorCritic`] with PPO.
///
/// One `loss` call scores one minibatch against its stored behaviour
/// log-probabilities. Several epochs over shuffled minibatches are several
/// [`Trainer::queue_step`](crate::train::trainer::Trainer::queue_step) calls,
/// whose statistics the caller reads once, together, like
/// [`PpoTask`](crate::rl::PpoTask).
pub struct EntityPpoTask<'a, R: Runtime, E: FloatElem> {
    ac: &'a EntityActorCritic<R, E>,
    params: Vec<Param<R, E>>,
    config: crate::rl::PpoConfig,
    temperature: f32,
    last: RefCell<Option<[Tensor<R, E>; 5]>>,
}

impl<'a, R: Runtime, E: FloatElem> EntityPpoTask<'a, R, E> {
    /// Train every parameter of actor and critic at `temperature`.
    pub fn new(ac: &'a EntityActorCritic<R, E>, config: crate::rl::PpoConfig, temperature: f32) -> Self {
        Self {
            params: ac.parameters(),
            ac,
            config,
            temperature,
            last: RefCell::new(None),
        }
    }

    /// The actor-critic being optimised.
    pub fn ac(&self) -> &'a EntityActorCritic<R, E> {
        self.ac
    }

    /// The configuration.
    pub fn config(&self) -> &crate::rl::PpoConfig {
        &self.config
    }

    /// Diagnostics of the most recent loss, or `None` before the first.
    ///
    /// Kept as device tensors by [`EntityPpoTask::loss`]; reading them is a
    /// synchronisation, so a loop that never asks never stalls.
    pub fn stats(&self) -> Option<EntityPpoStats> {
        let tensors = self.stat_tensors()?;
        let ([], values) =
            index::read_together([], tensors.each_ref()).unwrap_or_else(|err| panic!("{err}"));
        Some(EntityPpoStats::from_values(values.map(|v| v[0])))
    }

    /// The `[1]` device tensors [`EntityPpoStats`] reads, in
    /// [`EntityPpoStats`] field order, or `None` before the first loss — for
    /// a caller that has other values to read and wants them all under the
    /// same synchronisation.
    pub fn stat_tensors(&self) -> Option<[Tensor<R, E>; 5]> {
        self.last.borrow().clone()
    }
}

impl<R: Runtime, E: FloatElem> TrainStep<R, E> for EntityPpoTask<'_, R, E> {
    type Batch = EntityPpoBatch<R, E>;

    fn parameters(&self) -> Vec<Param<R, E>> {
        self.params.clone()
    }

    fn loss(&self, batch: &Self::Batch) -> Result<Var<R, E>> {
        let loss = entity_ppo_objective(self.ac, batch, &self.config, self.temperature)?;
        // Kept as tensors, not read: turning them into numbers is what
        // `stats` (or one shared read over many queued steps) does.
        *self.last.borrow_mut() = Some([
            loss.policy.tensor().clone(),
            loss.value.tensor().clone(),
            loss.entropy.tensor().clone(),
            loss.approx_kl.clone(),
            loss.clip_fraction.clone(),
        ]);
        Ok(loss.total)
    }
}
