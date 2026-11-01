//! Training objectives.
//!
//! * **Decision loss** — InfoNCE over the candidate actions with (possibly several)
//!   positives: `-log sum_{a in P} softmax(score)_a`. With one positive it is
//!   exactly cross-entropy, which is the point the plan makes: with three fixed
//!   actions InfoNCE *is* a 3-way classifier. Used by every head.
//! * **Supervised contrastive** (Khosla et al. 2020) over the batch's state
//!   embeddings: states with the same label are positives of each other.
//! * **Auxiliary**: squared error on the clipped vol-scaled return; logistic loss
//!   on the stop-loss flag.
//!
//! `L = w_dec * L_dec + w_sup * L_supcon + w_ret * L_ret + w_risk * L_risk`,
//! each a weighted mean over the batch.

use mamba3::autograd::Var;
use mamba3::backend::{Device, FloatElem};
use mamba3::cubecl::prelude::Runtime;
use mamba3::error::Result;
use mamba3::tensor::Tensor;
use serde::{Deserialize, Serialize};

use crate::batch::Batch;
use crate::model::Outputs;

/// Loss weights.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LossConfig {
    /// Decision (InfoNCE / cross-entropy) weight.
    pub decision: f32,
    /// Supervised-contrastive weight (contrastive head only).
    pub supcon: f32,
    /// Supervised-contrastive temperature.
    pub supcon_temperature: f32,
    /// Return-regression weight.
    pub ret: f32,
    /// Stop-loss weight.
    pub risk: f32,
    /// Vol-scaled return targets are clipped to `[-clip, clip]`.
    pub ret_clip: f32,
}

impl Default for LossConfig {
    fn default() -> Self {
        Self {
            decision: 1.0,
            supcon: 0.0,
            supcon_temperature: 0.1,
            ret: 0.0,
            risk: 0.0,
            ret_clip: 5.0,
        }
    }
}

/// Each term of the last loss, for logging.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct LossTerms {
    /// Decision loss.
    pub decision: f32,
    /// Supervised contrastive loss.
    pub supcon: f32,
    /// Return loss.
    pub ret: f32,
    /// Risk loss.
    pub risk: f32,
}

fn constant<R: Runtime, E: FloatElem>(data: &[f32], shape: Vec<usize>, device: &Device<R>) -> Result<Var<R, E>> {
    Ok(Var::constant(Tensor::from_f32(data, shape, device)?))
}

/// `sum(v * w) / sum(w)` for `[B]`-shaped `v` (any trailing size-1 axes).
fn weighted_mean<R: Runtime, E: FloatElem>(v: &Var<R, E>, w: &[f32], device: &Device<R>) -> Result<Var<R, E>> {
    let total: f32 = w.iter().sum();
    let wv = constant(w, v.dims().to_vec(), device)?;
    Ok(v.mul(&wv)?.sum()?.mul_scalar(1.0 / total.max(1e-12)))
}

/// The multi-positive decision loss per row, `[B, 1]`.
pub fn decision_loss_rows<R: Runtime, E: FloatElem>(
    scores: &Var<R, E>,
    positives: &[f32],
    device: &Device<R>,
) -> Result<Var<R, E>> {
    let dims = scores.dims().to_vec();
    let mask = constant(positives, dims, device)?;
    let p = scores.softmax(1)?;
    Ok(p.mul(&mask)?.sum_dim(1)?.add_scalar(1e-12).log().neg())
}

/// Supervised contrastive loss over unit-norm `z [B, E]` with class `labels`.
/// Rows without another same-label row in the batch are left out.
pub fn supcon_loss<R: Runtime, E: FloatElem>(
    z: &Var<R, E>,
    labels: &[u32],
    weights: &[f32],
    temperature: f32,
    device: &Device<R>,
) -> Result<Option<Var<R, E>>> {
    let b = labels.len();
    let mut pos = vec![0.0f32; b * b];
    let mut row_w = vec![0.0f32; b];
    let mut diag = vec![0.0f32; b * b];
    for i in 0..b {
        diag[i * b + i] = -1e4;
        let mut count = 0;
        for j in 0..b {
            if i != j && labels[i] == labels[j] {
                pos[i * b + j] = 1.0;
                count += 1;
            }
        }
        if count > 0 {
            for j in 0..b {
                pos[i * b + j] /= count as f32;
            }
            row_w[i] = weights[i];
        }
    }
    if row_w.iter().all(|&w| w == 0.0) {
        return Ok(None);
    }
    let sim = z.matmul_nt(z)?.mul_scalar(1.0 / temperature);
    let sim = sim.add(&constant(&diag, vec![b, b], device)?)?;
    let lp = sim.log_softmax(1)?;
    let per_row = lp.mul(&constant(&pos, vec![b, b], device)?)?.sum_dim(1)?.neg();
    Ok(Some(weighted_mean(&per_row, &row_w, device)?))
}

/// The total loss of a batch and its terms (read back only when asked).
pub fn total_loss<R: Runtime, E: FloatElem>(
    out: &Outputs<R, E>,
    batch: &Batch,
    cfg: &LossConfig,
    device: &Device<R>,
) -> Result<Var<R, E>> {
    let dec = weighted_mean(&decision_loss_rows(&out.scores, &batch.positives, device)?, &batch.weights, device)?;
    let mut total = dec.mul_scalar(cfg.decision);
    if cfg.supcon > 0.0
        && let Some(z) = &out.state_embedding
        && let Some(sc) = supcon_loss(z, &batch.labels, &batch.weights, cfg.supcon_temperature, device)?
    {
        total = total.add(&sc.mul_scalar(cfg.supcon))?;
    }
    if cfg.ret > 0.0
        && let Some(r) = &out.ret
    {
        let clip = cfg.ret_clip;
        let target: Vec<f32> = batch.ret_target.iter().map(|v| v.clamp(-clip, clip)).collect();
        let diff = r.sub(&constant(&target, vec![batch.size], device)?)?;
        total = total.add(&weighted_mean(&diff.mul(&diff)?, &batch.weights, device)?.mul_scalar(cfg.ret))?;
    }
    if cfg.risk > 0.0
        && let Some(k) = &out.risk
    {
        // softplus(k) - y k, the logistic loss with logit k.
        let y = constant(&batch.stop_target, vec![batch.size], device)?;
        let bce = k.softplus()?.sub(&k.mul(&y)?)?;
        total = total.add(&weighted_mean(&bce, &batch.weights, device)?.mul_scalar(cfg.risk))?;
    }
    Ok(total)
}

