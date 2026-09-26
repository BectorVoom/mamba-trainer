//! Domain-free entity-to-plan model (ENTITY_MODEL_PLAN.md G2–G5, composed path).
//!
//! [`EntityModel`] is the domain-free set-to-plan model: a context of
//! named entity sets plus globals is encoded with bidirectional mixer layers,
//! one query set is expanded to plan tokens (teacher-forced in training,
//! decoded step by step at inference), a step-causal or joint decoder mixes
//! them, and named heads predict pointers, classes, label sets and numbers.
//! Nothing here names a domain.

use std::collections::BTreeMap;

use cubecl::prelude::Runtime;

use crate::autograd::Var;
use crate::backend::{Device, FloatElem};
use crate::error::{Error, Result};
use crate::models::entity::blocks::{BiBlock, DecoderLayer, Permutation};
use crate::models::entity::spec::{DecoderMode, EntityModelSpec, HeadKind, StepSelection};
use crate::nn::linear::{Linear, LinearConfig};
use crate::nn::module::{Module, ModuleVisitor};
use crate::nn::norm::{RmsNorm, RmsNormConfig};
use crate::nn::param::Param;
use crate::tensor::Tensor;
use crate::tensor::ops::index::IdTensor;
use crate::tensor::ops::random::Rng;

/// Fused on-device entity-model path (K0): `0` off, `1` on, `-1` not yet read
/// from the environment.
static FUSED_ENTITY_MODEL: core::sync::atomic::AtomicI8 = core::sync::atomic::AtomicI8::new(-1);

/// Whether the fused on-device entity-model path is enabled.
///
/// On by default, and `MAMBA3_FUSED_ENTITY_MODEL=0` turns it off. Until the K
/// kernels land, every call site takes the composed path regardless.
pub(crate) fn fused_entity_model_enabled() -> bool {
    use core::sync::atomic::Ordering;
    match FUSED_ENTITY_MODEL.load(Ordering::Relaxed) {
        -1 => {
            let on = std::env::var("MAMBA3_FUSED_ENTITY_MODEL").as_deref() != Ok("0");
            FUSED_ENTITY_MODEL.store(on as i8, Ordering::Relaxed);
            on
        }
        flag => flag == 1,
    }
}

/// Choose whether the entity model uses its fused on-device kernels or the
/// composed oracle path.
pub fn set_fused_entity_model(on: bool) {
    FUSED_ENTITY_MODEL.store(on as i8, core::sync::atomic::Ordering::Relaxed);
}

/// Whether the fused on-device entity-model path is currently enabled.
pub fn fused_entity_model() -> bool {
    fused_entity_model_enabled()
}

/// Encoder for one context set: `MLP_s` (the per-slot and per-set embeddings
/// live in single tables on [`EntityModel`]: `pos`/`typ` with `pos_row` /
/// `set_of` index buffers).
pub struct CtxEncoder<R: Runtime, E: FloatElem> {
    /// Set name (data key).
    pub name: String,
    /// Slots in the set.
    pub count: usize,
    /// Features per entity.
    pub features: usize,
    /// `MLP_s`: features → d → d.
    pub in1: Linear<R, E>,
    /// `MLP_s`: second layer.
    pub in2: Linear<R, E>,
}

/// Encoder for the query set: `MLP_q` plus one projection per lag.
pub struct QueryEncoder<R: Runtime, E: FloatElem> {
    /// Query-set name (data key).
    pub name: String,
    /// Query slots.
    pub count: usize,
    /// Plan length `K`.
    pub steps: usize,
    /// `MLP_q`: features → d → d.
    pub in1: Linear<R, E>,
    /// `MLP_q`: second layer.
    pub in2: Linear<R, E>,
    /// `P_lag`: one `Linear(d, d)` per lag in `1..=lags`.
    pub lags: Vec<Linear<R, E>>,
    /// `P_query`: projects summed previous-query picks (QueryCausal mode).
    pub pq: Option<Linear<R, E>>,
}

/// Runtime parameters of one pointer head.
pub struct PtrHead<R: Runtime, E: FloatElem> {
    /// Head name.
    pub name: String,
    /// Context-set index the pointer selects over.
    pub set: usize,
    /// Learned extra actions appended to the entities.
    pub extra: usize,
    /// Query projection.
    pub q_proj: Linear<R, E>,
    /// Key projection.
    pub k_proj: Linear<R, E>,
    /// Learned `[extra, d]` extra-action embeddings (`None` when `extra` = 0).
    pub extra_emb: Option<Param<R, E>>,
}

/// How one head reads the decoder state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeadInput {
    /// From the query state alone (unconditioned shared Linear).
    Unconditioned,
    /// From `concat(query state, chosen-entity token)` (conditioned Linear),
    /// where the token comes from pointer head `usize` (index into
    /// [`EntityModel::ptrs`]).
    Conditioned(usize),
}

/// One head's runtime layout: which shared Linear feeds it and which columns.
pub struct HeadRun {
    /// Head name.
    pub name: String,
    /// Pointer-head index for [`HeadKind::Pointer`], else `None`.
    pub ptr: Option<usize>,
    /// How the head reads the decoder state.
    pub input: HeadInput,
    /// Which steps carry outputs.
    pub steps: StepSelection,
    /// Multiplier on this head's loss.
    pub loss_weight: f32,
    /// Per-step loss weights (length = plan steps) or `None` for all 1.0.
    pub step_weights: Option<Vec<f32>>,
    /// Column range in the shared Linear's output.
    pub out_range: (usize, usize),
}

/// Output of [`EntityModel::decode`].
pub struct DecoderOut<R: Runtime, E: FloatElem> {
    /// `[B, N_ctx, d]` encoded context tokens.
    pub ctx: Var<R, E>,
    /// `[B, M, K, d]` query states.
    pub h: Var<R, E>,
}

/// Logits of every head plus the decoded pointer choices.
pub struct HeadOutputs<R: Runtime, E: FloatElem> {
    /// Logits per head: `[B, M, K, width]` (`[B, M, 1, width]` for `First`).
    pub logits: BTreeMap<String, Var<R, E>>,
    /// Decoded pointer choice per head: `[B, M, K]` ids (`u32::MAX` = IGNORE).
    pub choices: BTreeMap<String, IdTensor<R>>,
}

/// Shared head Linear outputs before per-head slicing: pointer logits plus
/// the conditioned (one per condition source, usually one) and
/// unconditioned shared outputs over `[R, ·]` (`R = B*M*K`).
pub struct CoreLogits<R: Runtime, E: FloatElem> {
    /// Pointer logits per head: `[B, M, K, width]`.
    pub ptr: BTreeMap<String, Var<R, E>>,
    /// `(condition pointer, [R, Wc])` per condition source.
    pub cond: Vec<(usize, Var<R, E>)>,
    /// `[R, Wu]` (`None` without unconditioned heads).
    pub uncond: Option<Var<R, E>>,
    /// Rows per head table (`B*M*K`).
    pub rows: usize,
}

/// Inference stem outputs: `(context, base queries, anchor tokens, globals)`.
pub type StemVars<R, E> = (Var<R, E>, Var<R, E>, Option<Var<R, E>>, Option<Var<R, E>>);


/// Which choices condition the heads: host ids (composed path, `IGNORE`
/// allowed) or device ids (fused path, no host upload).
pub enum ChoiceIds<'a, R: Runtime> {
    /// Host `[B*M*K]` ids per pointer head.
    Host(&'a BTreeMap<String, Vec<u32>>),
    /// Device `[B, M, K]` ids per pointer head.
    Device(&'a BTreeMap<String, IdTensor<R>>),
}

/// The domain-free entity-to-plan model.
pub struct EntityModel<R: Runtime, E: FloatElem> {
    spec: EntityModelSpec,
    ctx: Vec<CtxEncoder<R, E>>,
    glob: Option<(Linear<R, E>, Linear<R, E>)>,
    queries: Option<QueryEncoder<R, E>>,
    /// `[K, d]` step embeddings (when queries are present).
    step_emb: Option<Param<R, E>>,
    /// `[1, d]` token for "no previous choice" (IGNORE / step 0).
    none_prev: Param<R, E>,
    /// `[T, d]` per-set type embeddings (one row per context set).
    typ: Param<R, E>,
    /// `[P, d]` concatenated position tables (`None` when no set embeds).
    pos: Option<Param<R, E>>,
    /// Host pos-table start row per set (`None` = no embedding).
    pos_starts: Vec<Option<usize>>,
    /// Device copies of [`EntityModel::pos_row`] / [`EntityModel::set_of`]
    /// (uploaded once at init; K5).
    pos_row_dev: IdTensor<R>,
    set_of_dev: IdTensor<R>,
    ctx_blocks: Vec<BiBlock<R, E>>,
    /// Whole-context permutation per encoder layer (grid transposes).
    ctx_perms: Vec<Permutation>,
    /// Device copies of [`EntityModel::ctx_perms`] (`None` = identity; K4).
    ctx_perms_dev: Vec<Option<(IdTensor<R>, IdTensor<R>)>>,
    decoder: Vec<DecoderLayer<R, E>>,
    /// Device copy of the decoder's within-step reversal (K4).
    rev_perm_dev: Option<(IdTensor<R>, IdTensor<R>)>,
    norm: RmsNorm<R, E>,
    ptrs: Vec<PtrHead<R, E>>,
    heads: Vec<HeadRun>,
    /// Shared `Linear(2d, ·)` over conditioned heads (`None` when empty).
    cond_linear: Option<Linear<R, E>>,
    /// Shared `Linear(d, ·)` over unconditioned heads (`None` when empty).
    uncond_linear: Option<Linear<R, E>>,
}

fn normal_param<R: Runtime, E: FloatElem>(
    rows: usize,
    d: usize,
    device: &Device<R>,
    rng: &mut Rng,
) -> Result<Param<R, E>> {
    Ok(Param::new(Tensor::from_f32(
        &rng.normal_vec(rows * d, 0.0, 0.02),
        vec![rows, d],
        device,
    )?))
}

impl<R: Runtime, E: FloatElem> EntityModel<R, E> {
    /// The spec this model was built from.
    pub fn spec(&self) -> &EntityModelSpec {
        &self.spec
    }

    /// Instantiate with the configured seed.
    pub fn init(spec: &EntityModelSpec, device: &Device<R>) -> Result<Self> {
        let mut rng = Rng::seeded(spec.seed);
        Self::init_with_rng(spec, device, &mut rng)
    }

    /// Instantiate with an explicit RNG.
    pub fn init_with_rng(
        spec: &EntityModelSpec,
        device: &Device<R>,
        rng: &mut Rng,
    ) -> Result<Self> {
        spec.validate()?;
        let d = spec.d_model;
        let mut ssm = spec.ssm.clone();
        ssm.d_model = d;
        let depth = spec.context_layers + spec.decoder_layers;

        // Context encoders (MLPs); pos/typ live in single tables below.
        let mut ctx = Vec::with_capacity(spec.context.len());
        for s in &spec.context {
            ctx.push(CtxEncoder {
                name: s.name.clone(),
                count: s.count,
                features: s.features,
                in1: LinearConfig::new(s.features, d).init(device, rng),
                in2: LinearConfig::new(d, d).init(device, rng),
            });
        }
        // Per-set type table [T, d] and concatenated position table [P, d].
        let typ = normal_param(spec.context.len(), d, device, rng)?;
        let n_ctx: usize = spec.n_ctx();
        let mut pos_row = vec![crate::tensor::ops::entity_model::IGNORE; n_ctx];
        let mut set_of = vec![0u32; n_ctx];
        let mut pos_starts: Vec<Option<usize>> = Vec::with_capacity(spec.context.len());
        {
            let mut at = 0;
            let mut p = 0usize;
            for (i, s) in spec.context.iter().enumerate() {
                if s.position_embedding {
                    pos_starts.push(Some(p));
                    for j in 0..s.count {
                        pos_row[at + j] = (p + j) as u32;
                    }
                    p += s.count;
                } else {
                    pos_starts.push(None);
                }
                for j in 0..s.count {
                    set_of[at + j] = i as u32;
                }
                at += s.count;
            }
        }
        let pos = {
            let total: usize = spec
                .context
                .iter()
                .filter(|s| s.position_embedding)
                .map(|s| s.count)
                .sum();
            if total > 0 {
                Some(normal_param(total, d, device, rng)?)
            } else {
                None
            }
        };
        let pos_row_dev = IdTensor::from_slice(&pos_row, vec![n_ctx], device)?;
        let set_of_dev = IdTensor::from_slice(&set_of, vec![n_ctx], device)?;
        // Globals MLP (skipped when G = 0).
        let glob = if spec.globals > 0 {
            Some((
                LinearConfig::new(spec.globals, d).init(device, rng),
                LinearConfig::new(d, d).init(device, rng),
            ))
        } else {
            None
        };
        // Query encoder.
        let mut queries = None;
        let mut step_emb = None;
        if let Some(q) = &spec.queries {
            if let Some(name) = &q.anchor
                && spec.set_offset(name).is_none()
            {
                return Err(crate::error::Error::config(format!(
                    "entity model queries.anchor {name:?} names no context set"
                )));
            }
            let mut lags = Vec::with_capacity(q.lags);
            // Lags are ignored without autoregressive_on (§1.2): build no
            // dead projections then (every parameter must see a gradient).
            if q.autoregressive_on.is_some() {
                for _ in 0..q.lags {
                    lags.push(LinearConfig::new(d, d).init(device, rng));
                }
            }
            queries = Some(QueryEncoder {
                name: q.name.clone(),
                count: q.count,
                steps: q.steps,
                in1: LinearConfig::new(q.features, d).init(device, rng),
                in2: LinearConfig::new(d, d).init(device, rng),
                lags,
                pq: if matches!(spec.decoder, DecoderMode::QueryCausal) {
                    Some(LinearConfig::new(d, d).init(device, rng))
                } else {
                    None
                },
            });
            step_emb = Some(normal_param(q.steps, d, device, rng)?);
        }
        let none_prev = normal_param(1, d, device, rng)?;

        // Context mixer layers + per-layer whole-context permutations: on odd
        // layers the token range of every Grid { alternate_axes: true } set is
        // transposed (row-major <-> column-major).
        let mut ctx_blocks = Vec::with_capacity(spec.context_layers);
        for _ in 0..spec.context_layers {
            ctx_blocks.push(BiBlock::new(d, &ssm, spec.norm_eps, depth, device, rng)?);
        }
        let n_ctx: usize = spec.n_ctx();
        let mut ctx_perms = Vec::with_capacity(spec.context_layers);
        for l in 0..spec.context_layers {
            let mut fwd: Vec<u32> = (0..n_ctx as u32).collect();
            if l % 2 == 1 {
                let mut at = 0;
                for s in &spec.context {
                    if let crate::models::entity::spec::SetLayout::Grid {
                        height,
                        width,
                        alternate_axes: true,
                    } = &s.layout
                    {
                        for r in 0..*height {
                            for c in 0..*width {
                                fwd[at + c * height + r] = (at + r * width + c) as u32;
                            }
                        }
                    }
                    at += s.count;
                }
            }
            ctx_perms.push(Permutation::from_fwd(fwd));
        }
        let ctx_perms_dev: Vec<Option<(IdTensor<R>, IdTensor<R>)>> = ctx_perms
            .iter()
            .map(|p| {
                if p.is_identity() {
                    Ok(None)
                } else {
                    Ok(Some((
                        IdTensor::from_slice(&p.fwd, vec![n_ctx], device)?,
                        IdTensor::from_slice(&p.inv, vec![n_ctx], device)?,
                    )))
                }
            })
            .collect::<Result<Vec<_>>>()?;

        // Decoder layers.
        let mut decoder = Vec::with_capacity(spec.decoder_layers);
        match &spec.decoder {
            DecoderMode::Joint => {
                for _ in 0..spec.decoder_layers {
                    decoder.push(DecoderLayer::joint(
                        d,
                        &ssm,
                        spec.norm_eps,
                        depth,
                        device,
                        rng,
                    )?);
                }
            }
            DecoderMode::StepCausal { crew_symmetric } => {
                let m = spec.queries.as_ref().map(|q| q.count).unwrap_or(0);
                let k = spec.queries.as_ref().map(|q| q.steps).unwrap_or(0);
                let perm = if *crew_symmetric {
                    Some(Permutation::reverse_blocks(n_ctx, m, k, n_ctx + m * k))
                } else {
                    None
                };
                for _ in 0..spec.decoder_layers {
                    decoder.push(DecoderLayer::step_causal(
                        d,
                        &ssm,
                        spec.norm_eps,
                        depth,
                        *crew_symmetric,
                        perm.clone(),
                        device,
                        rng,
                    )?);
                }
            }
            DecoderMode::QueryCausal => {
                // Forward-only layers, no reversal: query m sees queries < m.
                for _ in 0..spec.decoder_layers {
                    decoder.push(DecoderLayer::step_causal(
                        d, &ssm, spec.norm_eps, depth, false, None, device, rng,
                    )?);
                }
            }
        }
        let rev_perm_dev = match &spec.decoder {
            DecoderMode::StepCausal {
                crew_symmetric: true,
            } => {
                let n = spec.n_ctx();
                let m = spec.queries.as_ref().map(|q| q.count).unwrap_or(0);
                let k = spec.queries.as_ref().map(|q| q.steps).unwrap_or(0);
                let p = Permutation::reverse_blocks(n, m, k, n + m * k);
                Some((
                    IdTensor::from_slice(&p.fwd, vec![n + m * k], device)?,
                    IdTensor::from_slice(&p.inv, vec![n + m * k], device)?,
                ))
            }
            _ => None,
        };
        let norm = RmsNormConfig::new(d)
            .with_eps(spec.norm_eps)
            .init(device, rng);

        // Pointer heads: resolve set indices, build projections + extras.
        let ptr_index = |name: &str| -> Option<usize> {
            spec.heads
                .iter()
                .filter(|h| matches!(h.kind, HeadKind::Pointer { .. }))
                .position(|h| h.name == name)
        };
        let mut ptrs = Vec::new();
        for h in spec.heads.iter().filter(|h| h.is_pointer()) {
            let (set_name, extra) = match &h.kind {
                HeadKind::Pointer { set, extra_actions } => (set, *extra_actions),
                _ => unreachable!(),
            };
            let set = spec
                .context
                .iter()
                .position(|s| &s.name == set_name)
                .ok_or_else(|| {
                    crate::error::Error::config(format!(
                        "entity model head {:?} names no context set {set_name:?}",
                        h.name
                    ))
                })?;
            ptrs.push(PtrHead {
                name: h.name.clone(),
                set,
                extra,
                q_proj: LinearConfig::new(d, d).init(device, rng),
                k_proj: LinearConfig::new(d, d).init(device, rng),
                extra_emb: if extra > 0 {
                    Some(normal_param(extra, d, device, rng)?)
                } else {
                    None
                },
            });
        }
        // Head layouts over the two shared Linears.
        let mut heads = Vec::with_capacity(spec.heads.len());
        let (mut cond_width, mut uncond_width) = (0usize, 0usize);
        for h in &spec.heads {
            let width = h.width(spec).ok_or_else(|| {
                crate::error::Error::config(format!(
                    "entity model head {:?} has no width (unknown set)",
                    h.name
                ))
            })?;
            let (ptr, input) = match &h.kind {
                HeadKind::Pointer { .. } => {
                    let ptr = ptr_index(&h.name).ok_or_else(|| {
                        crate::error::Error::config(format!(
                            "entity model pointer head {:?} missing",
                            h.name
                        ))
                    })?;
                    (Some(ptr), HeadInput::Unconditioned)
                }
                _ => match &h.condition_on {
                    Some(cond) => {
                        let ptr = ptr_index(cond).ok_or_else(|| {
                            crate::error::Error::config(format!(
                                "entity model head {:?}.condition_on {cond:?} is not a pointer",
                                h.name
                            ))
                        })?;
                        (None, HeadInput::Conditioned(ptr))
                    }
                    None => (None, HeadInput::Unconditioned),
                },
            };
            // Pointer heads cannot carry condition_on (rejected by validate),
            // so `input` from the match above stands.
            let out_range = match input {
                HeadInput::Conditioned(_) => {
                    let r = (cond_width, cond_width + width);
                    cond_width += width;
                    r
                }
                // Pointer heads read their own matmul logits, never the
                // shared outputs: they consume no columns.
                HeadInput::Unconditioned if ptr.is_some() => (0, 0),
                HeadInput::Unconditioned => {
                    let r = (uncond_width, uncond_width + width);
                    uncond_width += width;
                    r
                }
            };
            heads.push(HeadRun {
                name: h.name.clone(),
                ptr,
                input,
                steps: h.steps,
                loss_weight: h.loss_weight,
                step_weights: h.step_weights.clone(),
                out_range,
            });
        }
        let cond_linear = if cond_width > 0 {
            Some(LinearConfig::new(2 * d, cond_width).init(device, rng))
        } else {
            None
        };
        let uncond_linear = if uncond_width > 0 {
            Some(LinearConfig::new(d, uncond_width).init(device, rng))
        } else {
            None
        };

        Ok(Self {
            spec: spec.clone(),
            ctx,
            glob,
            queries,
            step_emb,
            none_prev,
            typ,
            pos,
            pos_starts,
            pos_row_dev,
            set_of_dev,
            ctx_blocks,
            ctx_perms,
            ctx_perms_dev,
            decoder,
            rev_perm_dev,
            norm,
            ptrs,
            heads,
            cond_linear,
            uncond_linear,
        })
    }

    /// Run the context encoder (§2.1): one `[B, N_s, F_s]` feature Var and one
    /// `[B, N_s]` presence Var per set (in spec order), plus globals `[B, G]`
    /// (`None` when `G = 0`). Returns `[B, N_ctx, d]`.
    pub fn encode(
        &self,
        feats: &[Var<R, E>],
        presence: &[Var<R, E>],
        globals: Option<&Var<R, E>>,
    ) -> Result<Var<R, E>> {
        let d = self.spec.d_model;
        if feats.len() != self.ctx.len() || presence.len() != self.ctx.len() {
            return Err(Error::shape(format!(
                "entity encode needs {} context sets, got {} feature and {} presence inputs",
                self.ctx.len(),
                feats.len(),
                presence.len()
            )));
        }
        let b = feats[0].shape().dim(0);
        let fused = fused_entity_model_enabled();
        let mut parts = Vec::with_capacity(self.ctx.len());
        for (si, ((enc, f), p)) in self.ctx.iter().zip(feats).zip(presence).enumerate() {
            let mut x = self.ctx_in(&enc.in1, &enc.in2, f)?;
            // Presence gating: absent entities contribute exactly 0 here.
            let gate = p
                .reshape(vec![b, enc.count, 1])?
                .expand(vec![b, enc.count, d])?;
            x = x.mul(&gate)?;
            if !fused {
                if let Some(start) = self.pos_starts[si] {
                    let pos = self.pos.as_ref().unwrap();
                    let pv = pos
                        .var(&x)
                        .slice(0, start, enc.count)?
                        .reshape(vec![1, enc.count, d])?
                        .expand(vec![b, enc.count, d])?;
                    x = x.add(&pv)?;
                }
                let tv = self
                    .typ
                    .var(&x)
                    .slice(0, si, 1)?
                    .reshape(vec![1, 1, d])?
                    .expand(vec![b, enc.count, d])?;
                x = x.add(&tv)?;
            }
            parts.push(x);
        }
        let mut c = crate::autograd::cat(&parts, 1)?;
        // Globals: broadcast add (composed) or the K5 join below.
        let mut g_var: Option<Var<R, E>> = None;
        if let Some((g1, g2)) = &self.glob {
            let g = globals.ok_or_else(|| {
                Error::shape("entity encode needs globals (spec.globals > 0)".to_string())
            })?;
            let g = g2.apply(&g1.apply(g)?.gelu()?)?;
            if !fused {
                let n_ctx = self.spec.n_ctx();
                c = c.add(&g.reshape(vec![b, 1, d])?.expand(vec![b, n_ctx, d])?)?;
            } else {
                g_var = Some(g);
            }
        }
        if fused {
            let pos_var = match &self.pos {
                Some(p) => p.var(&c),
                None => Var::constant(Tensor::from_f32(&vec![0.0f32; d], vec![1, d], c.device())?),
            };
            c = Var::broadcast_join(
                &c,
                &pos_var,
                &self.pos_row_dev,
                &self.typ.var(&c),
                &self.set_of_dev,
                g_var.as_ref(),
            )?;
        }
        for (l, block) in self.ctx_blocks.iter().enumerate() {
            match &self.ctx_perms_dev[l] {
                Some((fwd, inv)) if fused => {
                    c = Var::permute_tokens(&c, fwd, inv)?;
                    c = block.apply(&c)?;
                    c = Var::permute_tokens(&c, fwd, inv)?;
                }
                _ => {
                    let perm = &self.ctx_perms[l];
                    if !perm.is_identity() {
                        c = perm.apply(&c)?;
                    }
                    c = block.apply(&c)?;
                    if !perm.is_identity() {
                        c = perm.apply(&c)?;
                    }
                }
            }
        }
        Ok(c)
    }

    /// `MLP(x) = Linear2(GELU(Linear1(x)))`.
    fn ctx_in(&self, in1: &Linear<R, E>, in2: &Linear<R, E>, x: &Var<R, E>) -> Result<Var<R, E>> {
        in2.apply(&in1.apply(x)?.gelu()?)
    }

    /// Base query states (§2.2): `MLP_q` gated by presence. Returns `[B, M, d]`.
    pub fn query_base(&self, feats: &Var<R, E>, presence: &Var<R, E>) -> Result<Var<R, E>> {
        let q = self
            .queries
            .as_ref()
            .ok_or_else(|| Error::config("entity model has no query set".to_string()))?;
        let d = self.spec.d_model;
        let b = feats.shape().dim(0);
        let u = self.ctx_in(&q.in1, &q.in2, feats)?;
        let gate = presence
            .reshape(vec![b, q.count, 1])?
            .expand(vec![b, q.count, d])?;
        u.mul(&gate)
    }

    /// Embedded globals `[B, d]` (`None` when `G = 0`).
    pub fn global_embed(&self, globals: Option<&Var<R, E>>) -> Result<Option<Var<R, E>>> {
        match &self.glob {
            Some((g1, g2)) => {
                let g = globals.ok_or_else(|| {
                    Error::shape("entity model needs globals (spec.globals > 0)".to_string())
                })?;
                Ok(Some(g2.apply(&g1.apply(g)?.gelu()?)?))
            }
            None => Ok(None),
        }
    }

    /// Index of the plan head (the pointer named by `autoregressive_on`).
    pub fn plan_ptr(&self) -> Option<usize> {
        let name = self.spec.queries.as_ref()?.autoregressive_on.as_ref()?;
        self.ptrs.iter().position(|p| &p.name == name)
    }

    /// Token table for pointer head `ptr`: `[ctx set tokens ; extra
    /// embeddings ; none_prev]` as `[B, N + E + 1, d]`.
    fn token_table(&self, ptr: usize, ctx: &Var<R, E>) -> Result<Var<R, E>> {
        let p = &self.ptrs[ptr];
        let b = ctx.shape().dim(0);
        let d = self.spec.d_model;
        let (at, n) = self.set_span(p.set);
        let mut table = ctx.slice(1, at, n)?;
        if let Some(extra) = &p.extra_emb {
            let e = extra
                .var(ctx)
                .reshape(vec![1, p.extra, d])?
                .expand(vec![b, p.extra, d])?;
            table = crate::autograd::cat(&[table, e], 1)?;
        }
        let none = self
            .none_prev
            .var(ctx)
            .reshape(vec![1, 1, d])?
            .expand(vec![b, 1, d])?;
        crate::autograd::cat(&[table, none], 1)
    }

    /// `(offset, count)` of a context set in the concatenated sequence.
    fn set_span(&self, set: usize) -> (usize, usize) {
        let mut at = 0;
        for (i, e) in self.ctx.iter().enumerate() {
            if i == set {
                return (at, e.count);
            }
            at += e.count;
        }
        (at, 0)
    }

    /// Tokens of the chosen entities from device ids: `ids` is `[B*M*K]`
    /// (`IGNORE` = no choice → `none_prev`). Returns `[B, M, K, d]`.
    /// Unlike [`EntityModel::choice_tokens`] this needs no host upload.
    pub fn choice_tokens_ids(
        &self,
        ptr: usize,
        ids: &IdTensor<R>,
        b: usize,
        m: usize,
        k: usize,
        ctx: &Var<R, E>,
    ) -> Result<Var<R, E>> {
        let table = self.token_table(ptr, ctx)?;
        let s = table.shape().dim(1);
        if ids.len() != b * m * k {
            return Err(Error::shape(format!(
                "choice ids hold {} entries, expected {b}*{m}*{k}",
                ids.len()
            )));
        }
        Var::gather_choice(&table, ids, m * k, s - 1)?.reshape(vec![
            b,
            m,
            k,
            self.spec.d_model,
        ])
    }

    /// Anchor tokens from device ids: `ids` is `[B*M]` global context
    /// indices (`IGNORE` = no anchor → the zero vector). Returns `[B, M, d]`.
    pub fn anchor_tokens_ids(
        &self,
        ids: &IdTensor<R>,
        b: usize,
        m: usize,
        ctx: &Var<R, E>,
    ) -> Result<Var<R, E>> {
        if ids.len() != b * m {
            return Err(Error::shape(format!(
                "anchor ids hold {} entries, expected {b}*{m}",
                ids.len()
            )));
        }
        Var::gather_tokens(ctx, ids, m)
    }

    /// Previous-step tokens from device plan ids: `ids` is the plan head's
    /// `[B*M*K]` choices; output `lags` tensors of `[B, M, K, d]`.
    /// Empty when the spec has no `autoregressive_on`.
    pub fn prev_tokens_ids(
        &self,
        ids: &IdTensor<R>,
        b: usize,
        m: usize,
        k: usize,
        ctx: &Var<R, E>,
    ) -> Result<Vec<Var<R, E>>> {
        let q = match &self.queries {
            Some(q) => q,
            None => return Ok(Vec::new()),
        };
        let Some(plan) = self.plan_ptr() else {
            return Ok(Vec::new());
        };
        if ids.len() != b * m * k {
            return Err(Error::shape(format!(
                "plan ids hold {} entries, expected {b}*{m}*{k}",
                ids.len()
            )));
        }
        let table = self.token_table(plan, ctx)?;
        let s = table.shape().dim(1);
        let mut out = Vec::with_capacity(q.lags.len());
        for (l, _) in q.lags.iter().enumerate() {
            out.push(Var::prev_choice(&table, ids, m, k, l + 1, s - 1)?);
        }
        Ok(out)
    }

    /// Previous-query pick sums for QueryCausal mode (§2.2): `toks` holds
    /// `[B, M, K, d]` choice tokens (all queries); output `[B, M, K, d]`
    /// where query `m` carries the sum of queries `< m`.
    pub fn query_prev_sum(&self, toks: &Var<R, E>) -> Result<Var<R, E>> {
        let q = match &self.queries {
            Some(q) => q,
            None => {
                return Err(Error::config("entity model has no query set".to_string()));
            }
        };
        let d = self.spec.d_model;
        let dims = toks.shape().dims().to_vec();
        let (b, m, k) = (dims[0], q.count, q.steps);
        let mut acc = Var::constant(Tensor::from_f32(
            &vec![0.0f32; b * k * d],
            vec![b, 1, k, d],
            toks.device(),
        )?);
        let mut parts = Vec::with_capacity(m);
        for mi in 0..m {
            parts.push(acc.clone());
            acc = acc.add(&toks.slice(1, mi, 1)?)?;
        }
        crate::autograd::cat(&parts, 1)
    }

    /// Tokens of the chosen entities for pointer head `ptr` (§2.2 `tok(i)`):
    /// `host_ids` is `[B*M*K]` (`IGNORE` = no choice → `none_prev`).
    /// Returns `[B, M, K, d]`.
    pub fn choice_tokens(
        &self,
        ptr: usize,
        host_ids: &[u32],
        b: usize,
        m: usize,
        k: usize,
        ctx: &Var<R, E>,
    ) -> Result<Var<R, E>> {
        use crate::tensor::ops::entity_model::IGNORE;
        if host_ids.len() != b * m * k {
            return Err(Error::shape(format!(
                "choice ids hold {} entries, expected {b}*{m}*{k}",
                host_ids.len()
            )));
        }
        let table = self.token_table(ptr, ctx)?;
        let s = table.shape().dim(1);
        for &id in host_ids {
            if id != IGNORE && id as usize >= s {
                return Err(Error::shape(format!(
                    "choice id {id} is outside 0..{}",
                    s - 1
                )));
            }
        }
        let mapped: Vec<u32> = host_ids
            .iter()
            .map(|&id| if id == IGNORE { (s - 1) as u32 } else { id })
            .collect();
        let ids = IdTensor::from_slice(&mapped, vec![b * m * k], ctx.device())?;
        Var::gather_tokens(&table, &ids, m * k)?.reshape(vec![b, m, k, self.spec.d_model])
    }

    /// Anchor tokens: `host_ids` is `[B*M]` global context indices
    /// (`IGNORE` = no anchor → the zero vector). Returns `[B, M, d]`.
    pub fn anchor_tokens(
        &self,
        host_ids: &[u32],
        b: usize,
        m: usize,
        ctx: &Var<R, E>,
    ) -> Result<Var<R, E>> {
        if host_ids.len() != b * m {
            return Err(Error::shape(format!(
                "anchor ids hold {} entries, expected {b}*{m}",
                host_ids.len()
            )));
        }
        let ids = IdTensor::from_slice(host_ids, vec![b * m], ctx.device())?;
        Var::gather_tokens(ctx, &ids, m)
    }

    /// Previous-step tokens for every lag (§2.2): `host_ids` is the plan
    /// head's `[B*M*K]` choices; output `lags` tensors of `[B, M, K, d]`
    /// where lag `l` carries the choice of step `j - l` (`none_prev` where
    /// `j < l`). Empty when the spec has no `autoregressive_on`.
    pub fn prev_tokens(
        &self,
        host_ids: &[u32],
        b: usize,
        m: usize,
        k: usize,
        ctx: &Var<R, E>,
    ) -> Result<Vec<Var<R, E>>> {
        use crate::tensor::ops::entity_model::IGNORE;
        let q = match &self.queries {
            Some(q) => q,
            None => return Ok(Vec::new()),
        };
        let Some(plan) = self.plan_ptr() else {
            return Ok(Vec::new());
        };
        let mut out = Vec::with_capacity(q.lags.len());
        for (l, _) in q.lags.iter().enumerate() {
            let lag = l + 1;
            let mut shifted = vec![IGNORE; b * m * k];
            for bi in 0..b {
                for mi in 0..m {
                    for j in lag..k {
                        shifted[(bi * m + mi) * k + j] = host_ids[(bi * m + mi) * k + (j - lag)];
                    }
                }
            }
            out.push(self.choice_tokens(plan, &shifted, b, m, k, ctx)?);
        }
        Ok(out)
    }

    /// Build the decoder query tokens (§2.2): `u` is `[B,M,d]`, with the
    /// optional anchor token and globals plus `step_emb` and the projected
    /// previous-step tokens. Returns `[B, M*K, d]` in the decoder's token
    /// order (query-major for `Joint`, step-major for `StepCausal`).
    /// The fused path (K2) assembles the same tokens in one launch.
    pub fn build_queries(
        &self,
        u: &Var<R, E>,
        anchor_tok: Option<&Var<R, E>>,
        g: Option<&Var<R, E>>,
        prev_toks: &[Var<R, E>],
        qprev_sum: Option<&Var<R, E>>,
    ) -> Result<Var<R, E>> {
        let q = self.queries.as_ref().ok_or_else(|| {
            Error::config("entity model has no query set; decode needs queries".to_string())
        })?;
        let d = self.spec.d_model;
        let (b, m, k) = (u.shape().dim(0), q.count, q.steps);
        let mut base = u.clone();
        if let Some(a) = anchor_tok {
            base = base.add(a)?;
        }
        if let Some(g) = g {
            base = base.add(&g.reshape(vec![b, 1, d])?)?;
        }
        let step_emb = self.step_emb.as_ref().ok_or_else(|| {
            Error::config("entity model queries have no step embeddings".to_string())
        })?;
        let step = step_emb.var(u);
        // Summed lag projections [B, M, K, d] (zeros without autoregression).
        let mut extra: Option<Var<R, E>> = None;
        for (l, pt) in prev_toks.iter().enumerate() {
            let proj = q.lags[l]
                .apply(&pt.reshape(vec![b * m * k, d])?)?
                .reshape(vec![b, m, k, d])?;
            extra = Some(match extra {
                Some(e) => e.add(&proj)?,
                None => proj,
            });
        }
        let extra = match extra {
            Some(e) => e,
            None => Var::constant(Tensor::from_f32(
                &vec![0.0f32; b * m * k * d],
                vec![b, m, k, d],
                u.device(),
            )?),
        };
        // QueryCausal cross-query picks, projected once (P_query).
        let extra = match (&q.pq, qprev_sum) {
            (Some(pq), Some(qs)) => {
                let proj = pq
                    .apply(&qs.reshape(vec![b * m * k, d])?)?
                    .reshape(vec![b, m, k, d])?;
                extra.add(&proj)?
            }
            _ => extra,
        };
        if fused_entity_model_enabled() {
            let step_major = matches!(self.spec.decoder, DecoderMode::StepCausal { .. });
            return Var::assemble_queries(&base, &step, &extra, step_major);
        }
        let mut steps_out = Vec::with_capacity(k);
        for j in 0..k {
            let mut qj = base.clone();
            let se = step
                .slice(0, j, 1)?
                .reshape(vec![1, 1, d])?
                .expand(vec![b, m, d])?;
            qj = qj.add(&se)?;
            let ej = extra.slice(2, j, 1)?.reshape(vec![b, m, d])?;
            qj = qj.add(&ej)?;
            steps_out.push(qj.reshape(vec![b, m, 1, d])?);
        }
        let stacked = crate::autograd::cat(&steps_out, 2)?;
        match &self.spec.decoder {
            DecoderMode::Joint | DecoderMode::QueryCausal => stacked.reshape(vec![b, m * k, d]),
            DecoderMode::StepCausal { .. } => {
                stacked.permute(&[0, 2, 1, 3])?.reshape(vec![b, k * m, d])
            }
        }
    }

    /// Run the decoder (§2.3): `ctx` is `[B, N_ctx, d]`, `queries` holds
    /// `M*K` tokens in the decoder's order. Returns the encoded context and
    /// the query states as `[B, M, K, d]`.
    pub fn decode(&self, ctx: &Var<R, E>, queries: &Var<R, E>) -> Result<DecoderOut<R, E>> {
        let q = self.queries.as_ref().ok_or_else(|| {
            Error::config("entity model has no query set; decode needs queries".to_string())
        })?;
        let d = self.spec.d_model;
        let (b, n, m, k) = (ctx.shape().dim(0), self.spec.n_ctx(), q.count, q.steps);
        let mut s = crate::autograd::cat(&[ctx.clone(), queries.clone()], 1)?;
        let rev_dev = self.rev_perm_dev.as_ref().map(|(f, i)| (f, i));
        for layer in &self.decoder {
            s = layer.apply(&s, rev_dev)?;
        }
        s = self.norm.apply(&s)?;
        let out_ctx = s.slice(1, 0, n)?;
        let h_flat = s.slice(1, n, m * k)?;
        let h = match &self.spec.decoder {
            DecoderMode::Joint | DecoderMode::QueryCausal => h_flat.reshape(vec![b, m, k, d])?,
            DecoderMode::StepCausal { .. } => {
                h_flat.reshape(vec![b, k, m, d])?.permute(&[0, 2, 1, 3])?
            }
        };
        Ok(DecoderOut { ctx: out_ctx, h })
    }
}

impl<R: Runtime, E: FloatElem> Module<R, E> for EntityModel<R, E> {
    fn visit(&self, visitor: &mut ModuleVisitor<'_, R, E>) {
        for (i, e) in self.ctx.iter().enumerate() {
            visitor.child(&format!("ctx_in1_{i}"), &e.in1);
            visitor.child(&format!("ctx_in2_{i}"), &e.in2);
        }
        visitor.param("ctx_typ", &self.typ);
        if let Some(p) = &self.pos {
            visitor.param("ctx_pos", p);
        }
        if let Some((g1, g2)) = &self.glob {
            visitor.child("glob_in1", g1);
            visitor.child("glob_in2", g2);
        }
        if let Some(q) = &self.queries {
            visitor.child("query_in1", &q.in1);
            visitor.child("query_in2", &q.in2);
            for (l, p) in q.lags.iter().enumerate() {
                visitor.child(&format!("lag_{l}"), p);
            }
            if let Some(pq) = &q.pq {
                visitor.child("pq", pq);
            }
        }
        if let Some(s) = &self.step_emb {
            visitor.param("step_emb", s);
        }
        visitor.param("none_prev", &self.none_prev);
        for (i, block) in self.ctx_blocks.iter().enumerate() {
            visitor.child_at("ctx_blocks", i, block);
        }
        for (i, layer) in self.decoder.iter().enumerate() {
            visitor.child_at("decoder", i, layer);
        }
        visitor.child("norm", &self.norm);
        for (i, p) in self.ptrs.iter().enumerate() {
            visitor.child(&format!("ptr_q_{i}"), &p.q_proj);
            visitor.child(&format!("ptr_k_{i}"), &p.k_proj);
            if let Some(e) = &p.extra_emb {
                visitor.param(&format!("ptr_extra_{i}"), e);
            }
        }
        if let Some(l) = &self.cond_linear {
            visitor.child("cond_heads", l);
        }
        if let Some(l) = &self.uncond_linear {
            visitor.child("uncond_heads", l);
        }
    }
}

impl<R: Runtime, E: FloatElem> EntityModel<R, E> {
    /// Head layouts in spec order (loss weights, column ranges).
    pub fn head_runs(&self) -> &[HeadRun] {
        &self.heads
    }

    /// Pointer-head runtime parameters in spec order.
    pub fn ptr_heads(&self) -> &[PtrHead<R, E>] {
        &self.ptrs
    }

    /// Keys for pointer head `ptr`: `[ctx set tokens ; extra embeddings]` as
    /// `[B, N + E, d]`.
    fn pointer_keys(&self, ptr: usize, ctx: &Var<R, E>) -> Result<Var<R, E>> {
        let p = &self.ptrs[ptr];
        let b = ctx.shape().dim(0);
        let d = self.spec.d_model;
        let (at, n) = self.set_span(p.set);
        let mut keys = ctx.slice(1, at, n)?;
        if let Some(extra) = &p.extra_emb {
            let e = extra
                .var(ctx)
                .reshape(vec![1, p.extra, d])?
                .expand(vec![b, p.extra, d])?;
            keys = crate::autograd::cat(&[keys, e], 1)?;
        }
        Ok(keys)
    }

    /// `-1e4` mask over `[B, M, K, N + E]` for pointer head `ptr`: absent
    /// entities and `legal` = 0 are masked (a constant added, no host read).
    fn pointer_mask(
        &self,
        ptr: usize,
        batch: &crate::models::entity::batch::EntityBatch<R, E>,
        b: usize,
        m: usize,
        k: usize,
        like: &Var<R, E>,
    ) -> Result<Var<R, E>> {
        let p = &self.ptrs[ptr];
        let n = self.ctx[p.set].count;
        let device = like.device();
        let presence = &batch.ctx_presence[p.set];
        let ones = Tensor::from_f32(&vec![1.0f32; b * n], vec![b, n], device)?;
        let ent = Var::constant(ones)
            .sub(&Var::constant(presence.clone()))?
            .mul_scalar(-1e4)
            .reshape(vec![b, 1, 1, n])?
            .expand(vec![b, m, k, n])?;
        let ent = if p.extra > 0 {
            let zeros = Tensor::from_f32(
                &vec![0.0f32; b * m * k * p.extra],
                vec![b, m, k, p.extra],
                device,
            )?;
            crate::autograd::cat(&[ent, Var::constant(zeros)], 3)?
        } else {
            ent
        };
        match batch.legal.get(&p.name) {
            Some(leg) => {
                let dims = leg.shape().dims().to_vec();
                let leg = Var::constant(leg.clone());
                let ones_l: usize = dims.iter().product();
                let ones_t = Tensor::from_f32(&vec![1.0f32; ones_l], dims.clone(), device)?;
                let leg = Var::constant(ones_t).sub(&leg)?.mul_scalar(-1e4);
                let leg = if dims.len() == 2 {
                    leg.reshape(vec![b, 1, 1, dims[1]])?
                        .expand(vec![b, m, k, dims[1]])?
                } else {
                    leg.reshape(vec![b, m, 1, dims[2]])?
                        .expand(vec![b, m, k, dims[2]])?
                };
                ent.add(&leg)
            }
            None => Ok(ent),
        }
    }

    /// Pointer-head logits (§2.4): one `matmul_nt` each over
    /// `[context tokens ; extra embeddings]`, with the `-1e4` presence/legal
    /// mask. Returns `(name → [B, M, K, width])`.
    pub fn pointer_logits(
        &self,
        dec: &DecoderOut<R, E>,
        batch: &crate::models::entity::batch::EntityBatch<R, E>,
    ) -> Result<BTreeMap<String, Var<R, E>>> {
        let q = self
            .spec
            .queries
            .as_ref()
            .ok_or_else(|| Error::config("entity heads need a query set".to_string()))?;
        let d = self.spec.d_model;
        let (b, m, k) = (batch.b, q.count, q.steps);
        let scale = 1.0 / (d as f32).sqrt();
        let mut logits = BTreeMap::new();
        for (pi, p) in self.ptrs.iter().enumerate() {
            let keys = self.pointer_keys(pi, &dec.ctx)?;
            let width = keys.shape().dim(1);
            let hbk = dec.h.reshape(vec![b, m * k, d])?;
            let mut logit = p
                .q_proj
                .apply(&hbk)?
                .matmul_nt(&p.k_proj.apply(&keys)?)?
                .mul_scalar(scale)
                .reshape(vec![b, m, k, width])?;
            logit = logit.add(&self.pointer_mask(pi, batch, b, m, k, &dec.h)?)?;
            logits.insert(p.name.clone(), logit);
        }
        Ok(logits)
    }

    /// Shared head Linear outputs before per-head slicing.
    pub fn core_logits(
        &self,
        dec: &DecoderOut<R, E>,
        batch: &crate::models::entity::batch::EntityBatch<R, E>,
        choice_ids: ChoiceIds<'_, R>,
    ) -> Result<CoreLogits<R, E>> {
        let q = self
            .spec
            .queries
            .as_ref()
            .ok_or_else(|| Error::config("entity heads need a query set".to_string()))?;
        let d = self.spec.d_model;
        let (b, m, k) = (batch.b, q.count, q.steps);
        let rows = b * m * k;
        let ptr = self.pointer_logits(dec, batch)?;
        // Conditioned inputs, grouped by condition source: one shared-Linear
        // apply per group (usually one: the plan head).
        let mut cond: Vec<(usize, Var<R, E>)> = Vec::new();
        if let Some(lin) = &self.cond_linear {
            let mut sources: Vec<usize> = Vec::new();
            for run in &self.heads {
                if let HeadInput::Conditioned(p) = run.input
                    && !sources.contains(&p)
                {
                    sources.push(p);
                }
            }
            for p in sources {
                let cond_name = &self.ptrs[p].name;
                let tok = match &choice_ids {
                    ChoiceIds::Host(map) => {
                        let host = map.get(cond_name).ok_or_else(|| {
                            Error::shape(format!(
                                "entity heads need choice ids for conditioning head {cond_name:?}"
                            ))
                        })?;
                        self.choice_tokens(p, host, b, m, k, &dec.ctx)?
                    }
                    ChoiceIds::Device(map) => {
                        let dev = map.get(cond_name).ok_or_else(|| {
                            Error::shape(format!(
                                "entity heads need choice ids for conditioning head {cond_name:?}"
                            ))
                        })?;
                        let flat = dev.reshape(vec![b * m * k])?;
                        self.choice_tokens_ids(p, &flat, b, m, k, &dec.ctx)?
                    }
                };
                let cat = crate::autograd::cat(&[dec.h.clone(), tok], 3)?;
                cond.push((p, lin.apply(&cat.reshape(vec![rows, 2 * d])?)?));
            }
        }
        let uncond = match &self.uncond_linear {
            Some(lin) => Some(lin.apply(&dec.h.reshape(vec![rows, d])?)?),
            None => None,
        };
        Ok(CoreLogits {
            ptr,
            cond,
            uncond,
            rows,
        })
    }

    /// Head logits for a decoded batch (§2.4): `choice_ids` carries the ids
    /// of every pointer head that conditions another head (host or device).
    /// Returns logits shaped `[B, M, K, width]` (`[B, M, 1, width]` for
    /// `First` heads) plus device argmax choices for the pointer heads.
    pub fn heads(
        &self,
        dec: &DecoderOut<R, E>,
        batch: &crate::models::entity::batch::EntityBatch<R, E>,
        choice_ids: ChoiceIds<'_, R>,
    ) -> Result<HeadOutputs<R, E>> {
        let q = self
            .spec
            .queries
            .as_ref()
            .ok_or_else(|| Error::config("entity heads need a query set".to_string()))?;
        let (b, m, k) = (batch.b, q.count, q.steps);
        let core = self.core_logits(dec, batch, choice_ids)?;
        let mut logits = core.ptr;
        let mut choices = BTreeMap::new();

        // Device argmax choices for the pointer heads.
        for p in &self.ptrs {
            let l = logits.get(&p.name).unwrap();
            let width = l.shape().dim(3);
            let flat = l.reshape(vec![b * m * k, width])?;
            let ids =
                crate::tensor::ops::reduce::argmax(flat.tensor(), 1)?.reshape(vec![b, m, k])?;
            choices.insert(p.name.clone(), ids);
        }

        // Slice the shared outputs per head.
        for run in &self.heads {
            if run.ptr.is_some() {
                continue; // pointer logits already above.
            }
            let (start, end) = run.out_range;
            let len = end - start;
            let input = match run.input {
                HeadInput::Conditioned(p) => core
                    .cond
                    .iter()
                    .find(|(q, _)| *q == p)
                    .map(|(_, v)| v)
                    .ok_or_else(|| {
                        Error::config(format!(
                            "entity model has no conditioned output for head {:?}",
                            run.name
                        ))
                    })?,
                HeadInput::Unconditioned => core.uncond.as_ref().ok_or_else(|| {
                    Error::config("entity model has no unconditioned-heads Linear".to_string())
                })?,
            };
            let out = match run.steps {
                StepSelection::All => input.slice(1, start, len)?.reshape(vec![b, m, k, len])?,
                StepSelection::First => {
                    let all: Var<R, E> = input.slice(1, start, len)?;
                    let per_step = all.reshape(vec![b, m, k, len])?;
                    per_step.slice(2, 0, 1)?
                }
            };
            logits.insert(run.name.clone(), out);
        }
        Ok(HeadOutputs { logits, choices })
    }

    /// Training decode stem (teacher-forced): encode → queries → decode.
    /// Computes the QueryCausal query-prev sums from the plan head's labels
    /// when the mode needs them.
    pub fn train_decode(
        &self,
        batch: &crate::models::entity::batch::EntityBatch<R, E>,
    ) -> Result<(DecoderOut<R, E>, bool)> {
        let want_qprev = matches!(self.spec.decoder, DecoderMode::QueryCausal);
        self.train_decode_with(batch, want_qprev)
    }
    /// Teacher-forced choice tokens of the plan head (`[B, M, K, d]`), host
    /// or device ids by path. Shared by step lags, query sums and heads.
    pub fn plan_choice_tokens(
        &self,
        batch: &crate::models::entity::batch::EntityBatch<R, E>,
        ctx: &Var<R, E>,
        b: usize,
        m: usize,
        k: usize,
    ) -> Result<Var<R, E>> {
        let fused = fused_entity_model_enabled();
        let qspec = self.spec.queries.as_ref().unwrap();
        let plan = self.plan_ptr().ok_or_else(|| {
            Error::config("query-causal decoding needs a plan head".to_string())
        })?;
        let plan_name = qspec.autoregressive_on.as_ref().unwrap();
        if fused {
            let dev = batch.choice_dev.get(plan_name).ok_or_else(|| {
                Error::shape(format!(
                    "entity forward needs choice ids for plan head {plan_name:?}"
                ))
            })?;
            let flat = dev.reshape(vec![b * m * k])?;
            self.choice_tokens_ids(plan, &flat, b, m, k, ctx)
        } else {
            if batch.resident {
                return Err(Error::shape(
                    "entity forward on the composed path needs host ids; resident batches only run fused (use from_host batches for the oracle)".to_string(),
                ));
            }
            let host = batch.choice_ids.get(plan_name).ok_or_else(|| {
                Error::shape(format!(
                    "entity forward needs choice ids for plan head {plan_name:?}"
                ))
            })?;
            self.choice_tokens(plan, host, b, m, k, ctx)
        }
    }

    /// Training decode stem with explicit query-prev wiring.
    pub fn train_decode_with(
        &self,
        batch: &crate::models::entity::batch::EntityBatch<R, E>,
        want_qprev: bool,
    ) -> Result<(DecoderOut<R, E>, bool)> {
        let fused = fused_entity_model_enabled();
        if !fused && batch.resident {
            return Err(Error::shape(
                "entity forward on the composed path needs host ids; resident batches only run fused (use from_host batches for the oracle)".to_string(),
            ));
        }
        let feats: Vec<Var<R, E>> = batch
            .ctx_feats
            .iter()
            .map(|t| Var::traced(t.clone()))
            .collect();
        let presence: Vec<Var<R, E>> = batch
            .ctx_presence
            .iter()
            .map(|t| Var::constant(t.clone()))
            .collect();
        let globals = batch.globals.as_ref().map(|g| Var::traced(g.clone()));
        let ctx = self.encode(&feats, &presence, globals.as_ref())?;
        let qf = Var::traced(
            batch
                .q_feats
                .as_ref()
                .ok_or_else(|| Error::config("entity forward needs query features".to_string()))?
                .clone(),
        );
        let qp = Var::constant(
            batch
                .q_presence
                .as_ref()
                .ok_or_else(|| Error::config("entity forward needs query presence".to_string()))?
                .clone(),
        );
        let u = self.query_base(&qf, &qp)?;
        let g = self.global_embed(globals.as_ref())?;
        let qspec = self.spec.queries.as_ref().unwrap();
        let (b, m, k) = (batch.b, qspec.count, qspec.steps);
        let anchor_tok = match (&batch.anchor_ids, &batch.anchor_dev) {
            (_, Some(dev)) if fused => Some(self.anchor_tokens_ids(dev, b, m, &ctx)?),
            (Some(ids), _) => Some(self.anchor_tokens(ids, b, m, &ctx)?),
            (None, None) => None,
            (None, Some(_)) => {
                return Err(Error::shape(
                    "entity forward needs host anchor ids on the composed path".to_string(),
                ));
            }
        };
        // Teacher forcing: the plan head's labels condition later steps
        // (IGNORE → none_prev).
        let prev = match self.plan_ptr() {
            Some(_) => {
                let plan_name = qspec.autoregressive_on.as_ref().unwrap();
                if fused {
                    let dev = batch.choice_dev.get(plan_name).ok_or_else(|| {
                        Error::shape(format!(
                            "entity forward needs choice ids for plan head {plan_name:?}"
                        ))
                    })?;
                    self.prev_tokens_ids(dev, b, m, k, &ctx)?
                } else {
                    let host = batch.choice_ids.get(plan_name).ok_or_else(|| {
                        Error::shape(format!(
                            "entity forward needs choice ids for plan head {plan_name:?}"
                        ))
                    })?;
                    self.prev_tokens(host, b, m, k, &ctx)?
                }
            }
            None => Vec::new(),
        };
        // QueryCausal cross-query picks (teacher-forced plan labels).
        let qprev_sum = if want_qprev {
            let toks = self.plan_choice_tokens(batch, &ctx, b, m, k)?;
            Some(self.query_prev_sum(&toks)?)
        } else {
            None
        };
        let queries = self.build_queries(
            &u,
            anchor_tok.as_ref(),
            g.as_ref(),
            &prev,
            qprev_sum.as_ref(),
        )?;
        let dec = self.decode(&ctx, &queries)?;
        Ok((dec, fused))
    }

    /// Training forward: encode → queries (teacher-forced) → decode → heads.
    /// The fused path (K2) gathers choice and anchor tokens from device ids
    /// (no host upload); the composed path maps host ids.
    pub fn forward_train(
        &self,
        batch: &crate::models::entity::batch::EntityBatch<R, E>,
    ) -> Result<(DecoderOut<R, E>, HeadOutputs<R, E>)> {
        let (dec, fused) = self.train_decode(batch)?;
        let out = if fused {
            self.heads(&dec, batch, ChoiceIds::Device(&batch.choice_dev))?
        } else {
            self.heads(&dec, batch, ChoiceIds::Host(&batch.choice_ids))?
        };
        Ok((dec, out))
    }

}

/// How [`EntityModel::predict`] fills the choices that condition later steps
/// and conditioned heads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decode {
    /// Decode step by step with the model's own choices (default argmax on
    /// the device, or `chooser`).
    Greedy,
    /// Condition on the batch's labels (diagnostics; needs `label.<plan>`).
    TeacherForced,
}

/// No-grad prediction output: host-readable logits plus device choices.
pub struct Prediction<R: Runtime, E: FloatElem> {
    /// Logits per head: `[B, M, K, width]` host tensors (`[B, M, 1, width]`
    /// for `First` heads).
    pub logits: BTreeMap<String, Tensor<R, E>>,
    /// Decoded ids per pointer head: `[B, M, K]`.
    pub choices: BTreeMap<String, IdTensor<R>>,
    /// Phantom element type.
    pub _elem: std::marker::PhantomData<E>,
}

/// Per-head evaluation metrics, keyed `<head>.top1` / `<head>.top3`
/// (pointer), `<head>.top1` (categorical), `<head>.bce` (multilabel),
/// `<head>.mse` (regression). Computed from a single greedy [`Decode`]
/// pass; kept = labelled (query, step) rows, as in training.
pub type EntityMetrics = BTreeMap<String, f32>;

impl<R: Runtime, E: FloatElem> EntityModel<R, E> {
    /// Shared inference stem: encode, base queries, anchors, globals.
    fn infer_stem(
        &self,
        batch: &crate::models::entity::batch::EntityBatch<R, E>,
    ) -> Result<StemVars<R, E>> {
        let feats: Vec<Var<R, E>> = batch
            .ctx_feats
            .iter()
            .map(|t| Var::constant(t.clone()))
            .collect();
        let presence: Vec<Var<R, E>> = batch
            .ctx_presence
            .iter()
            .map(|t| Var::constant(t.clone()))
            .collect();
        let globals = batch.globals.as_ref().map(|g| Var::constant(g.clone()));
        let ctx = self.encode(&feats, &presence, globals.as_ref())?;
        let qf = Var::constant(
            batch
                .q_feats
                .as_ref()
                .ok_or_else(|| Error::config("entity predict needs query features".to_string()))?
                .clone(),
        );
        let qp = Var::constant(
            batch
                .q_presence
                .as_ref()
                .ok_or_else(|| Error::config("entity predict needs query presence".to_string()))?
                .clone(),
        );
        let u = self.query_base(&qf, &qp)?;
        let g = self.global_embed(globals.as_ref())?;
        let anchor_tok = match &batch.anchor_ids {
            Some(ids) => {
                let q = self.spec.queries.as_ref().unwrap();
                Some(self.anchor_tokens(ids, batch.b, q.count, &ctx)?)
            }
            None => None,
        };
        Ok((ctx, u, anchor_tok, g))
    }

    /// No-grad prediction (§2.4 greedy decoding): the encoder runs once; for
    /// `j in 0..K` the decoder runs with `choice[<j]` filled and step `j`'s
    /// pointer choices are stored (`K` decoder passes). Without an
    /// autoregressive plan head this is a single pass. `chooser`, when given,
    /// maps `(step, logits [B, M, N+E])` to `[B, M]` ids (one device read
    /// per step, inference only); the constrained choice feeds the next step.
    pub fn predict(
        &self,
        batch: &crate::models::entity::batch::EntityBatch<R, E>,
        decode: Decode,
        chooser: Option<&mut dyn FnMut(usize, &Tensor<R, E>) -> Result<IdTensor<R>>>,
    ) -> Result<Prediction<R, E>> {
        use crate::tensor::ops::entity_model::IGNORE;
        let _guard = crate::autograd::no_grad();
        let q = self
            .spec
            .queries
            .as_ref()
            .ok_or_else(|| Error::config("entity predict needs a query set".to_string()))?;
        let (b, m, k) = (batch.b, q.count, q.steps);
        let (ctx, u, anchor_tok, g) = self.infer_stem(batch)?;

        // Host choice table per pointer head, filled as decoding proceeds.
        let mut host_choices: BTreeMap<String, Vec<u32>> = BTreeMap::new();
        for p in &self.ptrs {
            host_choices.insert(p.name.clone(), vec![IGNORE; b * m * k]);
        }
        if decode == Decode::TeacherForced {
            if batch.resident {
                // One read: resident batches carry no host id maps.
                for (name, ids) in &batch.choice_dev {
                    host_choices.insert(name.clone(), ids.to_vec());
                }
            } else {
                for (name, ids) in &batch.choice_ids {
                    host_choices.insert(name.clone(), ids.clone());
                }
            }
        }
        let plan_name: Option<String> = self
            .spec
            .queries
            .as_ref()
            .and_then(|q| q.autoregressive_on.clone());

        // Per-head accumulated step slices (constants under no_grad).
        let mut step_logits: BTreeMap<String, Vec<Var<R, E>>> = BTreeMap::new();
        let is_qc = matches!(
            self.spec.decoder,
            crate::models::entity::spec::DecoderMode::QueryCausal
        );
        // Passes as (query, step): TeacherForced and plan-free modes run a
        // single pass; StepCausal runs K step passes; QueryCausal runs M*K
        // query-major passes (each query sees previous queries' decoded
        // picks, each step its own decoded previous steps).
        let passes: Vec<(Option<usize>, usize)> = if decode == Decode::TeacherForced {
            vec![(None, 0)]
        } else if is_qc {
            (0..m)
                .flat_map(|mm| (0..k).map(move |j| (Some(mm), j)))
                .collect()
        } else {
            match &plan_name {
                Some(_) => (0..k).map(|j| (None, j)).collect(),
                None => vec![(None, 0)],
            }
        };
        let mut chooser = chooser;
        for &(qm, j) in &passes {
            let prev = match &plan_name {
                Some(name) => {
                    let host = host_choices.get(name).unwrap();
                    self.prev_tokens(host, b, m, k, &ctx)?
                }
                None => Vec::new(),
            };
            // QueryCausal cross-query picks from choices decoded so far
            // (future queries read IGNORE → none).
            let qprev_sum = if is_qc {
                let plan = self.plan_ptr().ok_or_else(|| {
                    Error::config("query-causal predict needs a plan head".to_string())
                })?;
                let host = host_choices.get(&self.ptrs[plan].name).unwrap();
                Some(self.query_prev_sum(&self.choice_tokens(plan, host, b, m, k, &ctx)?)?)
            } else {
                None
            };
            let queries = self.build_queries(
                &u,
                anchor_tok.as_ref(),
                g.as_ref(),
                &prev,
                qprev_sum.as_ref(),
            )?;
            let dec = self.decode(&ctx, &queries)?;
            // Decode pointer choices first (all pointer heads), so the
            // conditioned heads below see the decoded choice. Only the cells
            // this pass is responsible for are filled.
            let ptr_logits = self.pointer_logits(&dec, batch)?;
            let fill: Vec<(usize, usize)> = if decode == Decode::TeacherForced {
                // Single pass with every label known: choose at every step
                // from this pass's logits (argmax, or the chooser).
                (0..m)
                    .flat_map(|mm| (0..k).map(move |jj| (mm, jj)))
                    .collect()
            } else {
                match (qm, &plan_name, is_qc) {
                    (Some(mm), _, _) => vec![(mm, j)],
                    (None, Some(_), false) => (0..m).map(|mm| (mm, j)).collect(),
                    (None, _, _) => (0..m)
                        .flat_map(|mm| (0..k).map(move |jj| (mm, jj)))
                        .collect(),
                }
            };
            for p in &self.ptrs {
                // Group fill cells by step to slice each step's logits once.
                let mut by_step: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
                for (mm, jj) in &fill {
                    by_step.entry(*jj).or_default().push(*mm);
                }
                for (jj, mis) in &by_step {
                    let lj = ptr_logits.get(&p.name).unwrap().slice(2, *jj, 1)?;
                    let dims = lj.shape().dims().to_vec();
                    let w = dims[3];
                    let flat = lj.reshape(vec![b * m, w])?;
                    let ids: IdTensor<R> = match &mut chooser {
                        Some(f) if is_qc => {
                            // One query at a time: [B, 1, W] in, [B, 1] out;
                            // the step index passed is query-major.
                            let qm1 = mis[0];
                            let qslice = flat
                                .slice(0, qm1 * b, b)?
                                .reshape(vec![b, 1, w])?
                                .into_tensor();
                            let got = f(qm1 * k + j, &qslice)?;
                            if got.shape().dims() != [b] && got.shape().dims() != [b, 1] {
                                return Err(Error::shape(format!(
                                    "chooser returned {}, expected [B, 1]",
                                    got.shape()
                                )));
                            }
                            got.reshape(vec![b])?
                        }
                        Some(f) => {
                            let t = flat.into_tensor();
                            let got = f(*jj, &t)?;
                            if got.shape().dims() != [b * m] && got.shape().dims() != [b, m] {
                                return Err(Error::shape(format!(
                                    "chooser returned {}, expected [B, M]",
                                    got.shape()
                                )));
                            }
                            got.reshape(vec![b * m])?
                        }
                        _ => crate::tensor::ops::reduce::argmax(flat.tensor(), 1)?,
                    };
                    let vec_ids = ids.to_vec();
                    let entry = host_choices.get_mut(&p.name).unwrap();
                    if is_qc {
                        let qm1 = mis[0];
                        for bi in 0..b {
                            entry[(bi * m + qm1) * k + *jj] = vec_ids[bi];
                        }
                    } else {
                        for mi in 0..b * m {
                            entry[mi * k + *jj] = vec_ids[mi];
                        }
                    }
                }
            }
            // Now the full heads with this pass's choices known; keep only
            // this pass's cells (First heads: only step 0).
            let out = self.heads(&dec, batch, ChoiceIds::Host(&host_choices))?;
            if is_qc {
                for run in &self.heads {
                    let full = out.logits.get(&run.name).unwrap();
                    for (mm, jj) in &fill {
                        let keep = match run.steps {
                            crate::models::entity::spec::StepSelection::First => *jj == 0,
                            crate::models::entity::spec::StepSelection::All => true,
                        };
                        if !keep {
                            continue;
                        }
                        // [B, 1, 1, W] cells; assembly groups them per query.
                        let t = full
                            .slice(1, *mm, 1)?
                            .slice(2, *jj, 1)?
                            .reshape(vec![b, 1, 1, full.shape().dim(3)])?;
                        step_logits.entry(run.name.clone()).or_default().push(t);
                    }
                }
            } else {
                for run in &self.heads {
                    let full = out.logits.get(&run.name).unwrap();
                    if plan_name.is_none() || decode == Decode::TeacherForced {
                        // Single pass: keep every step (or step 0 for First).
                        match run.steps {
                            crate::models::entity::spec::StepSelection::First => {
                                step_logits.entry(run.name.clone()).or_default().push(full.slice(2, 0, 1)?);
                            }
                            crate::models::entity::spec::StepSelection::All => {
                                for jj in 0..k {
                                    step_logits.entry(run.name.clone()).or_default().push(full.slice(2, jj, 1)?);
                                }
                            }
                        }
                        continue;
                    }
                    let take = match run.steps {
                        crate::models::entity::spec::StepSelection::First => {
                            if j == 0 {
                                Some(full.slice(2, 0, 1)?)
                            } else {
                                None
                            }
                        }
                        crate::models::entity::spec::StepSelection::All => {
                            Some(full.slice(2, j, 1)?)
                        }
                    };
                    if let Some(t) = take {
                        step_logits.entry(run.name.clone()).or_default().push(t);
                    }
                }
            }
        }
        // Assemble full logits and device choices.
        let mut logits = BTreeMap::new();
        if is_qc {
            // Cells arrive query-major ([B,1,1,W] per (query, step)).
            for (name, cells) in &step_logits {
                let mut per_q = Vec::with_capacity(m);
                for mm in 0..m {
                    let qcells: Vec<Var<R, E>> = cells[mm * k..(mm + 1) * k].to_vec();
                    per_q.push(crate::autograd::cat(&qcells, 2)?);
                }
                logits.insert(name.clone(), crate::autograd::cat(&per_q, 1)?.into_tensor());
            }
        } else {
            for (name, steps) in &step_logits {
                // Cells arrive as [B, M, 1, W] step slices... reshape to
                // [B, M, 1, W] is already the pushed shape; cat over steps.
                let refs: Vec<Var<R, E>> = steps.to_vec();
                logits.insert(name.clone(), crate::autograd::cat(&refs, 2)?.into_tensor());
            }
        }
        let mut choices = BTreeMap::new();
        for (name, ids) in &host_choices {
            // Non-plan pointer heads in TeacherForced mode with no labels
            // stay IGNORE; still report them.
            choices.insert(
                name.clone(),
                IdTensor::from_slice(ids, vec![b, m, k], batch.ctx_feats[0].device())?,
            );
        }
        Ok(Prediction {
            logits,
            choices,
            _elem: std::marker::PhantomData,
        })
    }

    /// No-grad per-head metrics from one greedy pass (see [`EntityMetrics`]).
    pub fn evaluate(
        &self,
        batch: &crate::models::entity::batch::EntityBatch<R, E>,
    ) -> Result<EntityMetrics> {
        let pred = self.predict(batch, Decode::Greedy, None)?;
        let mut metrics = BTreeMap::new();
        for run in &self.heads {
            let spec_head = self.spec.head(&run.name).unwrap();
            match &spec_head.kind {
                crate::models::entity::spec::HeadKind::Pointer { .. }
                | crate::models::entity::spec::HeadKind::Categorical { .. } => {
                    let is_ptr = matches!(
                        spec_head.kind,
                        crate::models::entity::spec::HeadKind::Pointer { .. }
                    );
                    let host = batch.choice_ids.get(&run.name);
                    // Categorical heads have no host ids; read the label ids.
                    let (true_ids, rows): (Vec<u32>, usize) = match host {
                        Some(h) => (h.clone(), h.len()),
                        None => match batch.labels.get(&run.name).and_then(|l| l.as_ref()) {
                            Some(crate::models::entity::batch::HeadLabels::Class {
                                ids, ..
                            }) => (ids.to_vec(), ids.len()),
                            _ => {
                                return Err(Error::shape(format!(
                                    "entity evaluate needs label.{}",
                                    run.name
                                )));
                            }
                        },
                    };
                    let pred_logits = pred.logits.get(&run.name).unwrap().to_f32();
                    let w = pred_logits.len() / rows.max(1);
                    let pred_ids: Vec<u32> = match pred.choices.get(&run.name) {
                        Some(t) => t.to_vec(),
                        None => {
                            // Categorical heads: argmax the host logits.
                            (0..rows)
                                .map(|r| {
                                    let row = &pred_logits[r * w..(r + 1) * w];
                                    row.iter()
                                        .enumerate()
                                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                                        .unwrap()
                                        .0 as u32
                                })
                                .collect()
                        }
                    };
                    let logits = pred_logits;
                    let mut top1 = (0usize, 0usize);
                    let mut top3 = (0usize, 0usize);
                    for r in 0..rows {
                        let t = true_ids[r];
                        if t == crate::tensor::ops::entity_model::IGNORE {
                            continue;
                        }
                        top1.1 += 1;
                        if pred_ids[r] == t {
                            top1.0 += 1;
                        }
                        if is_ptr {
                            let row = &logits[r * w..(r + 1) * w];
                            let mut idx: Vec<usize> = (0..w).collect();
                            idx.sort_by(|&a, &b| row[b].partial_cmp(&row[a]).unwrap());
                            top3.1 += 1;
                            if idx[..3.min(w)].contains(&(t as usize)) {
                                top3.0 += 1;
                            }
                        }
                    }
                    let frac =
                        |(a, n): (usize, usize)| if n == 0 { 1.0 } else { a as f32 / n as f32 };
                    metrics.insert(format!("{}.top1", run.name), frac(top1));
                    if is_ptr {
                        metrics.insert(format!("{}.top3", run.name), frac(top3));
                    }
                }
                crate::models::entity::spec::HeadKind::MultiLabel { labels: l } => {
                    let (targets, keep) = match batch.labels.get(&run.name).and_then(|x| x.as_ref())
                    {
                        Some(crate::models::entity::batch::HeadLabels::Multi {
                            targets,
                            keep,
                            ..
                        }) => (targets.to_f32(), keep.to_f32()),
                        _ => {
                            return Err(Error::shape(format!(
                                "entity evaluate needs label.{}",
                                run.name
                            )));
                        }
                    };
                    let logits = pred.logits.get(&run.name).unwrap().to_f32();
                    let rows = keep.len();
                    let mut acc = 0.0;
                    let mut n = 0usize;
                    for r in 0..rows {
                        if keep[r] == 0.0 {
                            continue;
                        }
                        for c in 0..*l {
                            let x = logits[r * l + c] as f64;
                            let y = targets[r * l + c] as f64;
                            acc += x.max(0.0) + (-x.abs()).exp().ln_1p() - x * y;
                            n += 1;
                        }
                    }
                    metrics.insert(
                        format!("{}.bce", run.name),
                        if n == 0 { 0.0 } else { (acc / n as f64) as f32 },
                    );
                }
                crate::models::entity::spec::HeadKind::Regression { outputs: o } => {
                    let (targets, keep) = match batch.labels.get(&run.name).and_then(|x| x.as_ref())
                    {
                        Some(crate::models::entity::batch::HeadLabels::Reg {
                            targets,
                            keep,
                            ..
                        }) => (targets.to_f32(), keep.to_f32()),
                        _ => {
                            return Err(Error::shape(format!(
                                "entity evaluate needs label.{}",
                                run.name
                            )));
                        }
                    };
                    let logits = pred.logits.get(&run.name).unwrap().to_f32();
                    let rows = keep.len();
                    let mut acc = 0.0;
                    let mut n = 0usize;
                    for r in 0..rows {
                        if keep[r] == 0.0 {
                            continue;
                        }
                        for c in 0..*o {
                            let d = logits[r * o + c] - targets[r * o + c];
                            acc += d * d;
                            n += 1;
                        }
                    }
                    metrics.insert(
                        format!("{}.mse", run.name),
                        if n == 0 { 0.0 } else { acc / n as f32 },
                    );
                }
            }
        }
        Ok(metrics)
    }

    /// Save weights plus the spec (as metadata) to `path`.
    pub fn save(&self, path: impl AsRef<std::path::Path>, step: u64) -> Result<()> {
        crate::train::Checkpoint::capture(self, step)
            .with_metadata(serde_json::to_value(&self.spec)?)
            .save(path)
    }

    /// Rebuild from a checkpoint's metadata and restore its weights.
    pub fn load(path: impl AsRef<std::path::Path>, device: &Device<R>) -> Result<Self> {
        let ckpt = crate::train::Checkpoint::load(path)?;
        let spec: EntityModelSpec = serde_json::from_value(ckpt.metadata.clone())?;
        let model = EntityModel::init(&spec, device)?;
        ckpt.restore(&model, true)?;
        Ok(model)
    }
}
