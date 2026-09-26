//! Generic imitation-learning task over [`EntityBatch`]es (G4).
//!
//! Per-head loss: the mean over kept (query, step) rows of CE (pointer,
//! categorical), BCE averaged over labels (multilabel) or squared error
//! averaged over outputs (regression), rows weighted by the head's step
//! weights; the total is `loss_scale × Σ_h loss_weight_h × L_h`. Kept =
//! query present and label not IGNORE / NaN (presence enters through the
//! labels: absent queries must carry IGNORE labels).
//!
//! The fused path (K3) runs one segment-table-driven row kernel over the
//! shared head outputs for all all-steps heads, plus composed losses for
//! `First` heads; with several condition sources it falls back to the
//! composed path.

use std::collections::BTreeMap;

use cubecl::prelude::Runtime;

use crate::autograd::Var;
use crate::backend::FloatElem;
use crate::error::{Error, Result};
use crate::models::entity::batch::{EntityBatch, HeadLabels};
use crate::models::entity::model::{ChoiceIds, EntityModel};
use crate::models::entity::{EntityModelSpec, HeadKind};
use crate::nn::module::Module;
use crate::nn::param::Param;
use crate::tensor::Tensor;

/// Imitation-learning task over [`EntityBatch`]es.
pub struct EntityTask<'a, R: Runtime, E: FloatElem> {
    model: &'a EntityModel<R, E>,
    params: Vec<Param<R, E>>,
    loss_scale: f32,
}

/// Which logit sides a fused loss addresses: (cond, uncond, ptr).
fn seg_sides(spec: &EntityModelSpec, heads: &[String]) -> (bool, bool, bool) {
    let mut out = (false, false, false);
    for name in heads {
        let h = spec.head(name).unwrap();
        match &h.kind {
            HeadKind::Pointer { .. } => out.2 = true,
            _ if h.condition_on.is_some() => out.0 = true,
            _ => out.1 = true,
        }
    }
    out
}

/// Unscaled, unweighted loss of one head from flat `[rows, width]` logits.
fn flat_head_loss<R: Runtime, E: FloatElem>(
    flat: &Var<R, E>,
    labels: &HeadLabels<R, E>,
) -> Result<Var<R, E>> {
    let dims = flat.shape().dims().to_vec();
    let rows = dims[0];
    debug_assert_eq!(flat.shape().dims().len(), 2);
    Ok(match labels {
        HeadLabels::Class { ids, keep, div } => flat
            .cross_entropy_rows(ids, 0.0)?
            .mul(&Var::constant(keep.clone()))?
            .sum()?
            .mul_scalar(1.0 / div),
        HeadLabels::Multi { targets, keep, div } => {
            let l = targets.shape().dim(1);
            let bce = flat
                .softplus()?
                .sub(&flat.mul(&Var::constant(targets.clone()))?)?;
            bce.sum_dim(1)?
                .reshape(vec![rows])?
                .mul(&Var::constant(keep.clone()))?
                .sum()?
                .mul_scalar(1.0 / (div * l as f32))
        }
        HeadLabels::Reg { targets, keep, div } => {
            let o = targets.shape().dim(1);
            let diff = flat.sub(&Var::constant(targets.clone()))?;
            diff.mul(&diff)?
                .sum_dim(1)?
                .reshape(vec![rows])?
                .mul(&Var::constant(keep.clone()))?
                .sum()?
                .mul_scalar(1.0 / (div * o as f32))
        }
    })
}

impl<'a, R: Runtime, E: FloatElem> EntityTask<'a, R, E> {
    /// Train every parameter of `model` with `loss_scale` 1.
    pub fn new(model: &'a EntityModel<R, E>) -> Self {
        Self {
            params: model.parameters(),
            model,
            loss_scale: 1.0,
        }
    }

    /// Static loss scale: multiplies the loss; scale `eps` and the trainer
    /// clip by the same factor so the update is unchanged.
    pub fn with_loss_scale(mut self, scale: f32) -> Self {
        self.loss_scale = scale;
        self
    }

    /// The unscaled component losses by head name (before `loss_weight` and
    /// the loss scale).
    pub fn component_losses(&self, b: &EntityBatch<R, E>) -> Result<BTreeMap<String, Var<R, E>>> {
        let (_, out) = self.model.forward_train(b)?;
        let mut losses = BTreeMap::new();
        for run in self.model.head_runs() {
            let logits = out.logits.get(&run.name).ok_or_else(|| {
                Error::config(format!("entity loss is missing head {:?}", run.name))
            })?;
            let dims = logits.shape().dims().to_vec();
            let rows = dims[0] * dims[1] * dims[2];
            let width = dims[3];
            let labels = b.labels.get(&run.name).ok_or_else(|| {
                Error::shape(format!(
                    "entity loss needs label.{} for head {:?}",
                    run.name, run.name
                ))
            })?;
            let labels = labels.as_ref().ok_or_else(|| {
                Error::shape(format!(
                    "entity loss needs label.{} for head {:?}",
                    run.name, run.name
                ))
            })?;
            let flat = logits.reshape(vec![rows, width])?;
            losses.insert(run.name.clone(), flat_head_loss(&flat, labels)?);
        }
        Ok(losses)
    }

    /// Fused loss (K3): one segment-table row kernel over the shared outputs
    /// for all all-steps heads, composed losses for `First` heads. Falls back
    /// to the composed path without all-steps heads or with several
    /// condition sources. Returns the scalar loss (on the tape).
    pub fn loss_fused(&self, b: &EntityBatch<R, E>) -> Result<Var<R, E>> {
        let seg = match &b.seg {
            Some(s) if !s.heads.is_empty() => s,
            _ => return self.loss_composed(b),
        };
        let (dec, fused) = self.model.train_decode(b)?;
        let choices = if fused {
            ChoiceIds::Device(&b.choice_dev)
        } else {
            ChoiceIds::Host(&b.choice_ids)
        };
        let core = self.model.core_logits(&dec, b, choices)?;
        if core.cond.len() > 1 {
            // Several condition sources: no single shared table; composed.
            return self.loss_composed(b);
        }
        let q = self.model.spec().queries.as_ref().unwrap();
        let (bb, m, k) = (b.b, q.count, q.steps);
        let rows = bb * m * k;
        // Packed pointer logits in seg order (spec pointer order).
        let mut ptr_parts = Vec::new();
        for name in &seg.heads {
            if let Some(l) = core.ptr.get(name) {
                let w = l.shape().dim(3);
                ptr_parts.push(l.reshape(vec![rows, w])?);
            }
        }
        let ptr_packed = if ptr_parts.is_empty() {
            Var::constant(Tensor::from_f32(
                &vec![0.0f32; rows.max(1)],
                vec![rows.max(1), 1],
                dec.h.device(),
            )?)
        } else {
            crate::autograd::cat(&ptr_parts, 1)?
        };
        let cond = match core.cond.first() {
            Some((_, v)) => v.clone(),
            None => Var::constant(Tensor::from_f32(
                &vec![0.0f32; rows.max(1)],
                vec![rows.max(1), 1],
                dec.h.device(),
            )?),
        };
        let uncond = match &core.uncond {
            Some(v) => v.clone(),
            None => Var::constant(Tensor::from_f32(
                &vec![0.0f32; rows.max(1)],
                vec![rows.max(1), 1],
                dec.h.device(),
            )?),
        };
        let (uc, uu, up) = seg_sides(self.model.spec(), &seg.heads);
        let (seg_total, _report) = Var::segmented_loss(
            &cond,
            &uncond,
            &ptr_packed,
            &seg.class_ids,
            &seg.keep,
            &seg.ft,
            &seg.seg_dev,
            &seg.inv_width,
            &seg.coef,
            uc,
            uu,
            up,
        )?;
        let mut total = seg_total.mul_scalar(self.loss_scale);
        // First heads stay composed (step-0 slices of the shared outputs).
        for run in self.model.head_runs() {
            use crate::models::entity::StepSelection;
            if !matches!(run.steps, StepSelection::First) {
                continue;
            }
            let (start, end) = run.out_range;
            let len = end - start;
            let shared = match run.input {
                crate::models::entity::model::HeadInput::Conditioned(p) => core
                    .cond
                    .iter()
                    .find(|(q, _)| *q == p)
                    .map(|(_, v)| v)
                    .ok_or_else(|| {
                        Error::config(format!(
                            "entity fused loss has no conditioned output for {:?}",
                            run.name
                        ))
                    })?,
                crate::models::entity::model::HeadInput::Unconditioned => {
                    core.uncond.as_ref().ok_or_else(|| {
                        Error::config("entity fused loss has no unconditioned output".to_string())
                    })?
                }
            };
            let flat = shared
                .slice(1, start, len)?
                .reshape(vec![bb, m, k, len])?
                .slice(2, 0, 1)?
                .reshape(vec![bb * m, len])?;
            let labels = b
                .labels
                .get(&run.name)
                .and_then(|l| l.as_ref())
                .ok_or_else(|| {
                    Error::shape(format!("entity fused loss needs label.{}", run.name))
                })?;
            total = total.add(
                &flat_head_loss(&flat, labels)?.mul_scalar(run.loss_weight * self.loss_scale),
            )?;
        }
        Ok(total)
    }

    /// Composed loss: `loss_scale × Σ_h loss_weight_h × L_h`.
    pub fn loss_composed(&self, b: &EntityBatch<R, E>) -> Result<Var<R, E>> {
        let comps = self.component_losses(b)?;
        let mut total: Option<Var<R, E>> = None;
        for run in self.model.head_runs() {
            let l = comps.get(&run.name).ok_or_else(|| {
                Error::config(format!("entity loss is missing head {:?}", run.name))
            })?;
            let w = l.mul_scalar(run.loss_weight);
            total = Some(match total {
                Some(t) => t.add(&w)?,
                None => w,
            });
        }
        total
            .ok_or_else(|| Error::config("entity loss has no heads".to_string()))
            .map(|v| v.mul_scalar(self.loss_scale))
    }

    /// Forward report of the fused loss (K3): per-head unscaled components
    /// `[H]` (off the tape, for logging). `None` without all-steps heads.
    pub fn fused_report(&self, b: &EntityBatch<R, E>) -> Result<Option<Tensor<R, E>>> {
        let seg = match &b.seg {
            Some(s) if !s.heads.is_empty() => s,
            _ => return Ok(None),
        };
        let (dec, fused) = self.model.train_decode(b)?;
        let choices = if fused {
            ChoiceIds::Device(&b.choice_dev)
        } else {
            ChoiceIds::Host(&b.choice_ids)
        };
        let core = self.model.core_logits(&dec, b, choices)?;
        if core.cond.len() > 1 {
            return Ok(None);
        }
        let q = self.model.spec().queries.as_ref().unwrap();
        let (bb, m, k) = (b.b, q.count, q.steps);
        let rows = bb * m * k;
        let mut ptr_parts = Vec::new();
        for name in &seg.heads {
            if let Some(l) = core.ptr.get(name) {
                let w = l.shape().dim(3);
                ptr_parts.push(l.reshape(vec![rows, w])?);
            }
        }
        let ptr_packed = if ptr_parts.is_empty() {
            Var::constant(Tensor::from_f32(
                &vec![0.0f32; rows.max(1)],
                vec![rows.max(1), 1],
                dec.h.device(),
            )?)
        } else {
            crate::autograd::cat(&ptr_parts, 1)?
        };
        let cond = match core.cond.first() {
            Some((_, v)) => v.clone(),
            None => Var::constant(Tensor::from_f32(
                &vec![0.0f32; rows.max(1)],
                vec![rows.max(1), 1],
                dec.h.device(),
            )?),
        };
        let uncond = match &core.uncond {
            Some(v) => v.clone(),
            None => Var::constant(Tensor::from_f32(
                &vec![0.0f32; rows.max(1)],
                vec![rows.max(1), 1],
                dec.h.device(),
            )?),
        };
        let (uc, uu, up) = seg_sides(self.model.spec(), &seg.heads);
        let (_, report) = Var::segmented_loss(
            &cond,
            &uncond,
            &ptr_packed,
            &seg.class_ids,
            &seg.keep,
            &seg.ft,
            &seg.seg_dev,
            &seg.inv_width,
            &seg.coef,
            uc,
            uu,
            up,
        )?;
        Ok(Some(report))
    }
}

impl<R: Runtime, E: FloatElem> crate::train::trainer::TrainStep<R, E> for EntityTask<'_, R, E> {
    type Batch = EntityBatch<R, E>;

    fn parameters(&self) -> Vec<Param<R, E>> {
        self.params.clone()
    }

    fn loss(&self, b: &Self::Batch) -> Result<Var<R, E>> {
        if crate::models::entity::model::fused_entity_model_enabled() {
            return self.loss_fused(b);
        }
        self.loss_composed(b)
    }
}
