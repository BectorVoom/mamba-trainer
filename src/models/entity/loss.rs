//! Generic imitation-learning task over [`EntityBatch`]es (G4).
//!
//! Per-head loss: the mean over kept (query, step) rows of CE (pointer,
//! categorical), BCE averaged over labels (multilabel) or squared error
//! averaged over outputs (regression), rows weighted by the head's step
//! weights; the total is `loss_scale × Σ_h loss_weight_h × L_h`. Kept =
//! query present and label not IGNORE / NaN (presence enters through the
//! labels: absent queries must carry IGNORE labels).

use std::collections::BTreeMap;

use cubecl::prelude::Runtime;

use crate::autograd::Var;
use crate::backend::FloatElem;
use crate::error::{Error, Result};
use crate::models::entity::batch::{EntityBatch, HeadLabels};
use crate::models::entity::model::EntityModel;
use crate::nn::module::Module;
use crate::nn::param::Param;

/// Imitation-learning task over [`EntityBatch`]es.
pub struct EntityTask<'a, R: Runtime, E: FloatElem> {
    model: &'a EntityModel<R, E>,
    params: Vec<Param<R, E>>,
    loss_scale: f32,
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
            let loss = match labels {
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
            };
            losses.insert(run.name.clone(), loss);
        }
        Ok(losses)
    }
}

impl<R: Runtime, E: FloatElem> crate::train::trainer::TrainStep<R, E> for EntityTask<'_, R, E> {
    type Batch = EntityBatch<R, E>;

    fn parameters(&self) -> Vec<Param<R, E>> {
        self.params.clone()
    }

    fn loss(&self, b: &Self::Batch) -> Result<Var<R, E>> {
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
}
