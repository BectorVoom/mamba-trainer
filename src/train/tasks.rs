//! Ready-made [`TrainStep`] implementations.
//!
//! These are thin: each one holds a borrowed model, a cached parameter list and a
//! loss. Writing a new task — distillation, contrastive pretraining, a regression
//! head — means writing about twenty lines here, not touching the trainer.

use std::cell::Cell;

use cubecl::prelude::Runtime;

use crate::autograd::Var;
use crate::backend::FloatElem;
use crate::error::Result;
use crate::models::lm::Mamba3Lm;
use crate::models::vision::VisionMamba3;
use crate::nn::module::Module;
use crate::nn::param::Param;
use crate::tensor::Tensor;
use crate::tensor::ops::index::IdTensor;
use crate::train::loss::{CrossEntropyConfig, cross_entropy_with};
use crate::train::trainer::TrainStep;

/// Input ids and the ids they should predict.
pub struct LmBatch<R: Runtime> {
    /// `[batch, seq]` input ids.
    pub inputs: IdTensor<R>,
    /// `[batch, seq]` next-token targets.
    pub targets: IdTensor<R>,
}

impl<R: Runtime> Clone for LmBatch<R> {
    fn clone(&self) -> Self {
        Self {
            inputs: self.inputs.clone(),
            targets: self.targets.clone(),
        }
    }
}

/// Next-token prediction.
pub struct LmTask<'a, R: Runtime, E: FloatElem> {
    model: &'a Mamba3Lm<R, E>,
    params: Vec<Param<R, E>>,
    training: Cell<bool>,
    loss_config: CrossEntropyConfig,
}

impl<'a, R: Runtime, E: FloatElem> LmTask<'a, R, E> {
    /// Train every parameter of `model`.
    pub fn new(model: &'a Mamba3Lm<R, E>) -> Self {
        Self {
            params: model.parameters(),
            model,
            training: Cell::new(true),
            loss_config: CrossEntropyConfig::default(),
        }
    }

    /// Train only the parameters whose path contains one of `patterns`.
    ///
    /// This is how a LoRA run is expressed: `only(&["lora"])` after freezing the
    /// base weights.
    pub fn only(mut self, patterns: &[&str]) -> Self {
        self.params = self
            .model
            .named_parameters()
            .into_iter()
            .filter(|(name, _)| patterns.iter().any(|p| name.contains(p)))
            .map(|(_, p)| p)
            .collect();
        self
    }

    /// Configure the cross-entropy loss.
    pub fn with_loss(mut self, config: CrossEntropyConfig) -> Self {
        self.loss_config = config;
        self
    }

    /// The parameters this task will update.
    pub fn trainable(&self) -> &[Param<R, E>] {
        &self.params
    }
}

impl<R: Runtime, E: FloatElem> TrainStep<R, E> for LmTask<'_, R, E> {
    type Batch = LmBatch<R>;

    fn parameters(&self) -> Vec<Param<R, E>> {
        self.params.clone()
    }

    fn loss(&self, batch: &Self::Batch) -> Result<Var<R, E>> {
        let training = self.training.get();
        if !training {
            let _guard = crate::autograd::no_grad();
            let logits = self.model.forward(&batch.inputs, false)?;
            return cross_entropy_with(&logits, &batch.targets, self.loss_config);
        }
        let logits = self.model.forward(&batch.inputs, true)?;
        cross_entropy_with(&logits, &batch.targets, self.loss_config)
    }

    fn set_training(&self, training: bool) {
        self.training.set(training);
        self.model.set_training(training);
    }
}

/// Images and their class labels.
pub struct ImageBatch<R: Runtime, E: FloatElem> {
    /// `[batch, channels, height, width]` pixels.
    pub images: Tensor<R, E>,
    /// `[batch]` class ids.
    pub labels: IdTensor<R>,
}

impl<R: Runtime, E: FloatElem> Clone for ImageBatch<R, E> {
    fn clone(&self) -> Self {
        Self {
            images: self.images.clone(),
            labels: self.labels.clone(),
        }
    }
}

/// Single-label image classification.
pub struct ClassificationTask<'a, R: Runtime, E: FloatElem> {
    model: &'a VisionMamba3<R, E>,
    params: Vec<Param<R, E>>,
    training: Cell<bool>,
    loss_config: CrossEntropyConfig,
}

impl<'a, R: Runtime, E: FloatElem> ClassificationTask<'a, R, E> {
    /// Train every parameter of `model`.
    pub fn new(model: &'a VisionMamba3<R, E>) -> Self {
        Self {
            params: model.parameters(),
            model,
            training: Cell::new(true),
            loss_config: CrossEntropyConfig::default(),
        }
    }

    /// Train only parameters matching one of `patterns`.
    pub fn only(mut self, patterns: &[&str]) -> Self {
        self.params = self
            .model
            .named_parameters()
            .into_iter()
            .filter(|(name, _)| patterns.iter().any(|p| name.contains(p)))
            .map(|(_, p)| p)
            .collect();
        self
    }

    /// Configure the loss, e.g. to add label smoothing.
    pub fn with_loss(mut self, config: CrossEntropyConfig) -> Self {
        self.loss_config = config;
        self
    }
}

impl<R: Runtime, E: FloatElem> TrainStep<R, E> for ClassificationTask<'_, R, E> {
    type Batch = ImageBatch<R, E>;

    fn parameters(&self) -> Vec<Param<R, E>> {
        self.params.clone()
    }

    fn loss(&self, batch: &Self::Batch) -> Result<Var<R, E>> {
        if !self.training.get() {
            let _guard = crate::autograd::no_grad();
            let images = Var::constant(batch.images.clone());
            let logits = self.model.forward(&images)?;
            return cross_entropy_with(&logits, &batch.labels, self.loss_config);
        }
        let images = Var::traced(batch.images.clone());
        let logits = self.model.forward(&images)?;
        cross_entropy_with(&logits, &batch.labels, self.loss_config)
    }

    fn set_training(&self, training: bool) {
        self.training.set(training);
        self.model.set_training(training);
    }
}
