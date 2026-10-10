//! Loss functions.

use cubecl::prelude::Runtime;

use crate::autograd::Var;
use crate::backend::FloatElem;
use crate::error::{Error, Result};
use crate::tensor::Shape;
use crate::tensor::Tensor;
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

    let per_token = flat_logits.cross_entropy_rows(&flat_targets, config.label_smoothing)?;

    mean_over_kept(per_token, &flat_targets, config)
}

/// Mean of `per_token` (`[positions]`) over the positions whose target is
/// not the ignore index (all of them when there is none).
fn mean_over_kept<R: Runtime, E: FloatElem>(
    per_token: Var<R, E>,
    flat_targets: &IdTensor<R>,
    config: CrossEntropyConfig,
) -> Result<Var<R, E>> {
    match config.ignore_index {
        None => per_token.mean(),
        Some(ignore) => {
            // Build a 0/1 keep mask on the host: targets are already there in the
            // common case, and this keeps the masking exact rather than approximate.
            let positions = flat_targets.len();
            let host = flat_targets.try_to_vec()?;
            let keep: Vec<f32> = host
                .iter()
                .map(|id| if *id == ignore { 0.0 } else { 1.0 })
                .collect();
            let count: f32 = keep.iter().sum();
            if count == 0.0 {
                return Err(Error::config("every target was the ignore index"));
            }
            let mask = Var::constant(Tensor::from_f32(
                &keep,
                vec![positions],
                per_token.device(),
            )?);
            Ok(per_token.mul(&mask)?.sum()?.mul_scalar(1.0 / count))
        }
    }
}

/// [`cross_entropy_with`] of `x @ weight + bias` without the logits:
/// `x` is `[..., d]`, `weight` `[d, classes]`, `bias` `[classes]`, and
/// `targets` has `x`'s leading shape. Rows are processed `chunk_rows` at
/// a time; the forward keeps one `[positions]` log-sum-exp vector and
/// the backward recomputes each chunk's logits, so the largest
/// transient is `chunk_rows * classes` instead of `positions * classes`.
/// Same value and gradients as the composed form (one more product per
/// chunk in the backward).
pub fn linear_cross_entropy_with<R: Runtime, E: FloatElem>(
    x: &Var<R, E>,
    weight: &Var<R, E>,
    bias: Option<&Var<R, E>>,
    targets: &IdTensor<R>,
    config: CrossEntropyConfig,
    chunk_rows: usize,
) -> Result<Var<R, E>> {
    use crate::tensor::ops::elemwise::{add, add_assign_};
    use crate::tensor::ops::fused;
    use crate::tensor::ops::index::slice_ids;
    use crate::tensor::ops::matmul::{matmul, matmul_nt, matmul_tn};
    use crate::tensor::ops::movement::{cat, slice};
    use crate::tensor::ops::reduce::sum_dim;

    if x.rank() == 0 {
        return Err(Error::shape(format!(
            "linear cross entropy: x has shape {} but needs a trailing dim",
            x.shape(),
        )));
    }
    let d = x.shape().dim_from_end(0);
    if d == 0 {
        return Err(Error::shape(format!(
            "linear cross entropy: x has trailing dim 0 (shape {})",
            x.shape(),
        )));
    }
    let positions = x.shape().num_elements() / d;
    if weight.rank() != 2 || weight.shape().dim(0) != d {
        return Err(Error::shape(format!(
            "linear cross entropy: x has trailing dim {d} (shape {}) but weight is {}",
            x.shape(),
            weight.shape(),
        )));
    }
    let classes = weight.shape().dim(1);
    if let Some(b) = bias {
        if b.rank() != 1 || b.shape().dim(0) != classes {
            return Err(Error::shape(format!(
                "linear cross entropy: weight has {classes} classes (shape {}) but bias is {}",
                weight.shape(),
                b.shape(),
            )));
        }
    }
    if targets.len() != positions {
        return Err(Error::shape(format!(
            "linear cross entropy: {positions} rows (x shape {}) but {} targets",
            x.shape(),
            targets.len(),
        )));
    }
    if positions == 0 {
        return Err(Error::shape(format!(
            "linear cross entropy: no rows (x shape {})",
            x.shape(),
        )));
    }

    let x2 = x.reshape(Shape::new(vec![positions, d]))?;
    let flat_targets = targets.reshape(Shape::new(vec![positions]))?;
    let chunk = chunk_rows.max(1);
    let smoothing = config.label_smoothing;

    let x_t = x2.tensor().clone();
    let w_t = weight.tensor().clone();
    let b_t = bias.map(|b| b.tensor().clone());
    let device = x.device().clone();

    let mut losses = Vec::new();
    let mut lses = Vec::new();
    for start in (0..positions).step_by(chunk) {
        let len = chunk.min(positions - start);
        let xc = slice(&x_t, 0, start, len)?;
        let mut logits = matmul(&xc, &w_t)?;
        if let Some(b) = b_t.as_ref() {
            logits = add(&logits, b)?;
        }
        let ids_c = slice_ids(&flat_targets, start, len)?;
        let (loss_c, lse_c) = fused::cross_entropy_rows(&logits, &ids_c, smoothing)?;
        losses.push(loss_c);
        lses.push(lse_c);
    }
    let per_token_tensor = cat(&losses, 0)?;
    let lse = cat(&lses, 0)?;

    let mut parents: Vec<&Var<R, E>> = vec![&x2, weight];
    if let Some(b) = bias {
        parents.push(b);
    }
    let has_bias = bias.is_some();
    let per_token_var = Var::record(per_token_tensor, &parents, || {
        let x_t = x_t.clone();
        let w_t = w_t.clone();
        let b_t = b_t.clone();
        let flat_targets = flat_targets.clone();
        let lse = lse.clone();
        let device = device.clone();
        Box::new(move |g: &Tensor<R, E>| {
            let mut dx_parts = Vec::new();
            let dw = Tensor::<R, E>::zeros(Shape::new(vec![d, classes]), &device);
            let db = Tensor::<R, E>::zeros(Shape::new(vec![1, classes]), &device);
            for start in (0..positions).step_by(chunk) {
                let len = chunk.min(positions - start);
                let xc = slice(&x_t, 0, start, len)?;
                let mut logits = matmul(&xc, &w_t)?;
                if let Some(b) = b_t.as_ref() {
                    logits = add(&logits, b)?;
                }
                let ids_c = slice_ids(&flat_targets, start, len)?;
                let g_c = slice(g, 0, start, len)?;
                let lse_c = slice(&lse, 0, start, len)?;
                let dl =
                    fused::cross_entropy_rows_backward(&g_c, &logits, &ids_c, &lse_c, smoothing)?;
                dx_parts.push(matmul_nt(&dl, &w_t)?);
                add_assign_(&dw, &matmul_tn(&xc, &dl)?);
                if has_bias {
                    add_assign_(&db, &sum_dim(&dl, 0)?);
                }
            }
            let dx = cat(&dx_parts, 0)?;
            let mut grads = vec![Some(dx), Some(dw)];
            if has_bias {
                grads.push(Some(db.reshape(Shape::new(vec![classes]))?));
            }
            Ok(grads)
        })
    });

    mean_over_kept(per_token_var, &flat_targets, config)
}

/// The per-token loss one primitive at a time: log-softmax, a gather, and the
/// label-smoothing mix. This is what `cross_entropy_with` computed before the
/// fused kernel and is kept as the oracle the fused path is tested against.
pub fn cross_entropy_per_token_composed<R: Runtime, E: FloatElem>(
    logits: &Var<R, E>,
    targets: &IdTensor<R>,
    label_smoothing: f32,
) -> Result<Var<R, E>> {
    let log_probs = logits.log_softmax(1)?;
    let picked = log_probs.take_along_last(&targets.reshape(Shape::new(vec![targets.len()]))?)?;

    let mut per_token = picked.neg();
    if label_smoothing > 0.0 {
        // (1-e) * NLL(target) + e * mean over classes of NLL
        let uniform = log_probs.mean_dim(1)?.squeeze(1)?.neg();
        per_token = per_token
            .mul_scalar(1.0 - label_smoothing)
            .add(&uniform.mul_scalar(label_smoothing))?;
    }
    Ok(per_token)
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
    let predicted = crate::tensor::ops::reduce::argmax(logits.tensor(), last)?.try_to_vec()?;
    let expected = targets.try_to_vec()?;
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
