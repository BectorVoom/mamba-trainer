//! Inference over an evaluation split, without gradient tracking.

use mamba3::autograd::Var;
use mamba3::autograd::grad_mode::with_no_grad;
use mamba3::backend::{Device, FloatElem};
use mamba3::cubecl::prelude::Runtime;
use mamba3::error::Result;
use mamba3::tensor::Tensor;
use serde::{Deserialize, Serialize};

use crate::batch::{BatchSource, Split};
use crate::model::DecisionModel;

/// Model outputs for every sample of a split, in the source's order.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SplitPredictions {
    /// Raw action scores `[n][3]` (logits / scaled similarities).
    pub scores: Vec<[f32; 3]>,
    /// Predicted vol-scaled return, when the model has the head.
    pub ret: Option<Vec<f32>>,
    /// Stop-loss probability, when the model has the head.
    pub risk: Option<Vec<f32>>,
    /// The labels, copied from the batches.
    pub labels: Vec<u32>,
}

impl SplitPredictions {
    /// Softmax of the scores at temperature 1.
    pub fn probabilities(&self) -> Vec<[f32; 3]> {
        self.scores.iter().map(|s| softmax3(*s, 1.0)).collect()
    }

    /// Mean negative log-likelihood of the labels under [`Self::probabilities`].
    pub fn nll(&self) -> f64 {
        if self.labels.is_empty() {
            return f64::NAN;
        }
        let total: f64 = self
            .scores
            .iter()
            .zip(&self.labels)
            .map(|(s, &l)| -(softmax3(*s, 1.0)[l as usize].max(1e-12) as f64).ln())
            .sum();
        total / self.labels.len() as f64
    }
}

/// Numerically stable softmax of three scores divided by `temperature`.
pub fn softmax3(s: [f32; 3], temperature: f32) -> [f32; 3] {
    let t = temperature as f64;
    let m = s.iter().fold(f64::NEG_INFINITY, |m, &v| m.max(v as f64 / t));
    let e: Vec<f64> = s.iter().map(|&v| (v as f64 / t - m).exp()).collect();
    let z: f64 = e.iter().sum();
    [(e[0] / z) as f32, (e[1] / z) as f32, (e[2] / z) as f32]
}

/// Run the model over every sample of `split`.
pub fn predict_split<R: Runtime, E: FloatElem>(
    model: &DecisionModel<R, E>,
    source: &dyn BatchSource,
    split: Split,
    batch_size: usize,
    device: &Device<R>,
) -> Result<SplitPredictions> {
    let n = source.eval_len(split);
    let mut out = SplitPredictions::default();
    let mut rets = Vec::new();
    let mut risks = Vec::new();
    let (mut has_ret, mut has_risk) = (false, false);
    with_no_grad(|| -> Result<()> {
        for index in 0..n.div_ceil(batch_size.max(1)) {
            let batch = source.eval_batch(split, index, batch_size);
            if batch.size == 0 {
                continue;
            }
            let x = Var::constant(Tensor::<R, E>::from_f32(
                &batch.inputs,
                vec![batch.size, batch.window, batch.features],
                device,
            )?);
            let o = model.forward(&x)?;
            let scores = o.scores.try_to_f32()?;
            out.scores.extend(scores.chunks_exact(3).map(|c| [c[0], c[1], c[2]]));
            if let Some(r) = o.ret {
                has_ret = true;
                rets.extend(r.try_to_f32()?);
            }
            if let Some(k) = o.risk {
                has_risk = true;
                risks.extend(k.try_to_f32()?.into_iter().map(|v| 1.0 / (1.0 + (-v).exp())));
            }
            out.labels.extend(batch.labels);
        }
        Ok(())
    })?;
    out.ret = has_ret.then_some(rets);
    out.risk = has_risk.then_some(risks);
    Ok(out)
}
