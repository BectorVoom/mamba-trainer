//! Losses of the graph tasks and the training task (GRAPH_MAMBA_PLAN.md §2.4).
//!
//! Masking comes **before** any indexing or arithmetic on a missing target:
//! the target kernels write class 0 in place of a missing class and 0 in place
//! of a missing value, and return the mask that says which were real — label
//! present, split matching, row or graph not absent. Every mean is a
//! [`Var::masked_mean`], whose denominator is the count of real targets.

use std::cell::RefCell;

use cubecl::prelude::Runtime;

use crate::autograd::Var;
use crate::backend::FloatElem;
use crate::error::{Error, Result};
use crate::models::graph::batch::GraphBatch;
use crate::models::graph::model::GraphMamba;
use crate::models::graph::spec::{GraphTaskSpec, RegressionLoss};
use crate::models::graph::store::TargetStore;
use crate::nn::module::Module;
use crate::nn::param::Param;
use crate::tensor::Tensor;
use crate::tensor::ops::graph::{safe_class_targets, safe_float_targets};
use crate::tensor::ops::index::IdTensor;
use crate::train::TrainStep;

/// A batch's targets with the missing ones made safe, and the mask of the
/// real ones.
pub enum SafeTargets<R: Runtime, E: FloatElem> {
    /// One class per row (node tasks) or per graph slot.
    Class {
        /// Classes, 0 where masked.
        ids: IdTensor<R>,
        /// 1 where the class is real.
        mask: Tensor<R, E>,
    },
    /// `[graphs, targets]` floats per graph slot.
    Float {
        /// Values, 0 where masked.
        values: Tensor<R, E>,
        /// 1 where the value is real.
        mask: Tensor<R, E>,
    },
}

impl<R: Runtime, E: FloatElem> SafeTargets<R, E> {
    /// The mask of the real targets.
    pub fn mask(&self) -> &Tensor<R, E> {
        match self {
            SafeTargets::Class { mask, .. } | SafeTargets::Float { mask, .. } => mask,
        }
    }
}

/// The targets of `batch` for the splits in `flags`, masked before use. One
/// launch.
pub fn safe_targets<R: Runtime, E: FloatElem>(
    task: &GraphTaskSpec,
    batch: &GraphBatch<R, E>,
    flags: u32,
) -> Result<SafeTargets<R, E>> {
    let mismatch = || {
        Error::config(format!(
            "the dataset's targets do not fit the model's task {task:?}"
        ))
    };
    if let GraphTaskSpec::NodeClass { classes } | GraphTaskSpec::GraphClass { classes, .. } = task {
        let bound = batch.store.class_bound();
        if bound > *classes {
            return Err(Error::config(format!(
                "y: the dataset holds class {} and the model has {classes} classes; the dataset \
                 was built for another spec",
                bound - 1
            )));
        }
    }
    match (task, batch.store.targets()) {
        (_, TargetStore::None) => Err(Error::config(
            "the dataset has no targets: it can be predicted on, not trained or evaluated on"
                .to_string(),
        )),
        (GraphTaskSpec::NodeClass { .. }, TargetStore::Node { y, split }) => {
            let (ids, mask) = safe_class_targets(y, split, &batch.layout, false, flags)?;
            Ok(SafeTargets::Class { ids, mask })
        }
        (GraphTaskSpec::GraphClass { .. }, TargetStore::GraphClass { y, split }) => {
            let (ids, mask) = safe_class_targets(y, split, &batch.layout, true, flags)?;
            Ok(SafeTargets::Class { ids, mask })
        }
        (
            GraphTaskSpec::GraphRegression { targets: want, .. }
            | GraphTaskSpec::GraphMultiLabel { labels: want, .. },
            TargetStore::Graph {
                targets,
                y,
                present,
                split,
            },
        ) if want == targets => {
            let (values, mask) = safe_float_targets(y, present, split, &batch.layout, flags)?;
            Ok(SafeTargets::Float { values, mask })
        }
        _ => Err(mismatch()),
    }
}

/// The per-target loss terms of `output` against `targets`, before the masked
/// mean: `[rows]` cross-entropies, or `[graphs, targets]` errors.
fn loss_terms<R: Runtime, E: FloatElem>(
    task: &GraphTaskSpec,
    output: &Var<R, E>,
    targets: &SafeTargets<R, E>,
) -> Result<Var<R, E>> {
    match (task, targets) {
        (
            GraphTaskSpec::NodeClass { .. } | GraphTaskSpec::GraphClass { .. },
            SafeTargets::Class { ids, .. },
        ) => output.cross_entropy_rows(ids, 0.0),
        (GraphTaskSpec::GraphMultiLabel { .. }, SafeTargets::Float { values, .. }) => {
            // Binary cross-entropy from logits: softplus(x) − x·y.
            output
                .softplus()?
                .sub(&output.mul(&Var::constant(values.clone()))?)
        }
        (GraphTaskSpec::GraphRegression { loss, .. }, SafeTargets::Float { values, .. }) => {
            let error = output.sub(&Var::constant(values.clone()))?;
            match loss {
                RegressionLoss::L1 => Ok(error.abs()),
                RegressionLoss::Mse => error.mul(&error),
            }
        }
        _ => Err(Error::config(format!(
            "the targets do not fit the task {task:?}"
        ))),
    }
}

/// The unscaled loss of a batch: the mean of the task's per-target loss over
/// the real targets of the batch's split.
///
/// A batch in which no target of the split is real gives a zero loss with a
/// zero gradient (the denominator is `max(count, 1)`).
pub fn graph_loss<R: Runtime, E: FloatElem>(
    task: &GraphTaskSpec,
    output: &Var<R, E>,
    batch: &GraphBatch<R, E>,
) -> Result<Var<R, E>> {
    let split = batch.split.ok_or_else(|| {
        Error::config("a batch built for prediction has no split to take a loss over".to_string())
    })?;
    let targets = safe_targets(task, batch, split.flag())?;
    loss_terms(task, output, &targets)?.masked_mean(targets.mask())
}

/// The training task over [`GraphBatch`]es.
///
/// With a loss scale (16-bit dtypes), the loss handed to the trainer — and so
/// every gradient — is multiplied by it; the unscaled loss of each step is
/// kept on the device and handed out by [`GraphTask::take_losses`].
pub struct GraphTask<'a, R: Runtime, E: FloatElem> {
    model: &'a GraphMamba<R, E>,
    params: Vec<Param<R, E>>,
    loss_scale: f32,
    recorded: RefCell<Vec<Tensor<R, E>>>,
}

impl<'a, R: Runtime, E: FloatElem> GraphTask<'a, R, E> {
    /// A task over `model`'s parameters.
    pub fn new(model: &'a GraphMamba<R, E>) -> Self {
        Self {
            model,
            params: model.parameters(),
            loss_scale: 1.0,
            recorded: RefCell::new(Vec::new()),
        }
    }

    /// Multiply the loss by `scale` before it is differentiated.
    pub fn with_loss_scale(mut self, scale: f32) -> Self {
        self.loss_scale = scale;
        self
    }

    /// The loss scale.
    pub fn loss_scale(&self) -> f32 {
        self.loss_scale
    }

    /// The `[1]` unscaled losses built since the last call, oldest first.
    pub fn take_losses(&self) -> Vec<Tensor<R, E>> {
        core::mem::take(&mut *self.recorded.borrow_mut())
    }
}

impl<R: Runtime, E: FloatElem> TrainStep<R, E> for GraphTask<'_, R, E> {
    type Batch = GraphBatch<R, E>;

    fn parameters(&self) -> Vec<Param<R, E>> {
        self.params.clone()
    }

    fn loss(&self, batch: &Self::Batch) -> Result<Var<R, E>> {
        let output = self.model.forward(batch)?;
        let loss = graph_loss(&self.model.spec().task, &output, batch)?;
        self.recorded.borrow_mut().push(loss.tensor().clone());
        if self.loss_scale == 1.0 {
            Ok(loss)
        } else {
            Ok(loss.mul_scalar(self.loss_scale))
        }
    }

    fn set_training(&self, training: bool) {
        self.model.set_training(training);
    }
}
