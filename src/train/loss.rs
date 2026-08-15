//! Loss functions.

use cubecl::prelude::Runtime;

use crate::autograd::Var;
use crate::backend::FloatElem;
use crate::error::{Error, Result};
use crate::tensor::Shape;
use crate::tensor::ops::index::IdTensor;

/// Options for [`cross_entropy_with`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CrossEntropyConfig {
    /// Mass moved from the target class to a uniform distribution.
    pub label_smoothing: f32,
    /// Target id whose positions are excluded from the loss.
    pub ignore_index: Option<u32>,
}

impl Default for CrossEntropyConfig {
    fn default() -> Self {
        Self {
            label_smoothing: 0.0,
            ignore_index: None,
        }
    }
}

impl CrossEntropyConfig {
    /// Start from defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set label smoothing.
    pub fn with_label_smoothing(mut self, smoothing: f32) -> Self {
        self.label_smoothing = smoothing;
        self
    }

    /// Exclude positions whose target equals `id` (padding).
    pub fn with_ignore_index(mut self, id: u32) -> Self {
        self.ignore_index = Some(id);
        self
    }
}

/// Mean token-level cross entropy.
///
/// `logits` is `[..., classes]` and `targets` has the same leading shape.
pub fn cross_entropy<R: Runtime, E: FloatElem>(
    logits: &Var<R, E>,
    targets: &IdTensor<R>,
) -> Result<Var<R, E>> {
    cross_entropy_with(logits, targets, CrossEntropyConfig::default())
}

/// Cross entropy with label smoothing and an optional ignore index.
pub fn cross_entropy_with<R: Runtime, E: FloatElem>(
    logits: &Var<R, E>,
    targets: &IdTensor<R>,
    config: CrossEntropyConfig,
) -> Result<Var<R, E>> {
    let last = logits.rank() - 1;
    let classes = logits.shape().dim(last);
    let positions: usize = logits.shape().dims()[..last].iter().product();
    if positions != targets.len() {
        return Err(Error::shape(format!(
            "cross entropy: {} logit rows but {} targets",
            positions,
            targets.len()
        )));
    }

    let flat_logits = logits.reshape(Shape::new(vec![positions, classes]))?;
    let flat_targets = targets.reshape(Shape::new(vec![positions]))?;

    let log_probs = flat_logits.log_softmax(1)?;
    let picked = log_probs.take_along_last(&flat_targets)?;

    let mut per_token = picked.neg();
    if config.label_smoothing > 0.0 {
        // (1-e) * NLL(target) + e * mean over classes of NLL
        let uniform = log_probs.mean_dim(1)?.squeeze(1)?.neg();
        per_token = per_token
            .mul_scalar(1.0 - config.label_smoothing)
            .add(&uniform.mul_scalar(config.label_smoothing))?;
    }

    match config.ignore_index {
        None => per_token.mean(),
        Some(ignore) => {
            // Build a 0/1 keep mask on the host: targets are already there in the
            // common case, and this keeps the masking exact rather than approximate.
            let host = flat_targets.to_vec();
            let keep: Vec<f32> = host
                .iter()
                .map(|id| if *id == ignore { 0.0 } else { 1.0 })
                .collect();
            let count: f32 = keep.iter().sum();
            if count == 0.0 {
                return Err(Error::config("every target was the ignore index"));
            }
            let mask = Var::constant(crate::tensor::Tensor::from_f32(
                &keep,
                vec![positions],
                logits.device(),
            )?);
            Ok(per_token.mul(&mask)?.sum()?.mul_scalar(1.0 / count))
        }
    }
}

/// Mean squared error between two same-shaped values.
pub fn mse<R: Runtime, E: FloatElem>(
    prediction: &Var<R, E>,
    target: &Var<R, E>,
) -> Result<Var<R, E>> {
    let diff = prediction.sub(target)?;
    diff.mul(&diff)?.mean()
}

/// Mean absolute error between two same-shaped values.
pub fn mae<R: Runtime, E: FloatElem>(
    prediction: &Var<R, E>,
    target: &Var<R, E>,
) -> Result<Var<R, E>> {
    prediction.sub(target)?.abs().mean()
}

/// Perplexity from a mean cross-entropy value.
pub fn perplexity(loss: f32) -> f32 {
    loss.exp()
}

/// Fraction of positions whose argmax matches the target.
pub fn accuracy<R: Runtime, E: FloatElem>(
    logits: &Var<R, E>,
    targets: &IdTensor<R>,
) -> Result<f32> {
    let last = logits.rank() - 1;
    let predicted = crate::tensor::ops::reduce::argmax(logits.tensor(), last)?.to_vec();
    let expected = targets.to_vec();
    if predicted.len() != expected.len() {
        return Err(Error::shape("accuracy: shape mismatch".to_string()));
    }
    let hits = predicted
        .iter()
        .zip(expected.iter())
        .filter(|(a, b)| a == b)
        .count();
    Ok(hits as f32 / predicted.len().max(1) as f32)
}
