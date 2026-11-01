//! Training one model on one fold: AdamW, warmup + cosine schedule, gradient
//! clipping, NaN/Inf-guarded steps, early stopping on the validation split's
//! log loss with the best weights restored, and a JSON-lines log.

use std::time::Instant;

use mamba3::autograd::Var;
use mamba3::backend::{Device, FloatElem};
use mamba3::cubecl::prelude::Runtime;
use mamba3::error::{Error, Result};
use mamba3::nn::{Module, Param};
use mamba3::tensor::Tensor;
use mamba3::train::optim::AdamWConfig;
use mamba3::train::sched::LrSchedule;
use mamba3::train::trainer::{TrainStep, Trainer, TrainerConfig};
use serde::{Deserialize, Serialize};

use crate::batch::{Batch, BatchSource, Split};
use crate::losses::{LossConfig, total_loss};
use crate::model::DecisionModel;
use crate::predict::predict_split;

/// Optimisation settings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainConfig {
    /// Most optimizer steps.
    pub max_steps: u64,
    /// Samples per step.
    pub batch_size: usize,
    /// Peak learning rate.
    pub learning_rate: f32,
    /// Decoupled weight decay (matrices only).
    pub weight_decay: f32,
    /// Linear warmup steps.
    pub warmup_steps: u64,
    /// Global gradient-norm clip.
    pub grad_clip: f32,
    /// Evaluate on the validation split every this many steps.
    pub eval_every: u64,
    /// Stop after this many evaluations without improvement.
    pub patience: u32,
    /// Batch size for evaluation.
    pub eval_batch: usize,
    /// Loss weights.
    pub loss: LossConfig,
    /// Give up after this many consecutive non-finite steps.
    pub max_bad_steps: u32,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            max_steps: 3000,
            batch_size: 128,
            learning_rate: 1e-3,
            weight_decay: 0.05,
            warmup_steps: 100,
            grad_clip: 1.0,
            eval_every: 200,
            patience: 4,
            eval_batch: 512,
            loss: LossConfig::default(),
            max_bad_steps: 20,
        }
    }
}

/// One evaluation during training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalPoint {
    /// Optimizer step.
    pub step: u64,
    /// Mean training loss since the previous evaluation.
    pub train_loss: f32,
    /// Validation log loss (softmax of the scores at temperature 1).
    pub val_nll: f64,
    /// Learning rate at this step.
    pub learning_rate: f32,
    /// Mean gradient norm since the previous evaluation.
    pub grad_norm: f32,
    /// Device bytes held by live tensors at this evaluation.
    #[serde(default)]
    pub live_device_bytes: u64,
}

/// What a fit did.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FitReport {
    /// Step whose weights were kept.
    pub best_step: u64,
    /// Validation log loss of the kept weights.
    pub best_val_nll: f64,
    /// Steps taken.
    pub steps: u64,
    /// Steps skipped because the loss or gradients were not finite.
    pub skipped_steps: u32,
    /// Evaluation history.
    pub history: Vec<EvalPoint>,
    /// Wall-clock seconds.
    pub seconds: f64,
    /// Training samples per second.
    pub samples_per_second: f64,
    /// High-water mark of device bytes held by live tensors during the fit (an
    /// upper bound of the tensor-owned peak; the allocator's own pages are extra).
    #[serde(default)]
    pub peak_device_bytes: u64,
}

struct Task<'a, R: Runtime, E: FloatElem> {
    model: &'a DecisionModel<R, E>,
    loss: &'a LossConfig,
    device: &'a Device<R>,
}

impl<R: Runtime, E: FloatElem> TrainStep<R, E> for Task<'_, R, E> {
    type Batch = Batch;

    fn parameters(&self) -> Vec<Param<R, E>> {
        self.model.parameters()
    }

    fn loss(&self, batch: &Batch) -> Result<Var<R, E>> {
        let x = Var::constant(Tensor::<R, E>::from_f32(
            &batch.inputs,
            vec![batch.size, batch.window, batch.features],
            self.device,
        )?);
        let out = self.model.forward(&x)?;
        total_loss(&out, batch, self.loss, self.device)
    }
}

/// Train `model` on `source`'s training split; keep the weights with the best
/// validation log loss. `log` receives one JSON object per evaluation.
pub fn fit<R: Runtime, E: FloatElem>(
    model: &DecisionModel<R, E>,
    source: &mut dyn BatchSource,
    cfg: &TrainConfig,
    device: &Device<R>,
    log: &mut dyn FnMut(&EvalPoint),
) -> Result<FitReport> {
    let start = Instant::now();
    mamba3::backend::reset_live_high_water();
    let opt = AdamWConfig {
        learning_rate: cfg.learning_rate,
        weight_decay: cfg.weight_decay,
        ..Default::default()
    }
    .init::<R, E>();
    let trainer_cfg = TrainerConfig::builder()
        .learning_rate(cfg.learning_rate)
        .max_grad_norm(cfg.grad_clip)
        .schedule(LrSchedule::CosineWithWarmup {
            warmup_steps: cfg.warmup_steps,
            total_steps: cfg.max_steps,
            min_ratio: 0.05,
        })
        .build()?;
    let mut trainer = Trainer::new(trainer_cfg, opt);
    let task = Task {
        model,
        loss: &cfg.loss,
        device,
    };

    let mut best: Option<Best> = None;
    let mut history = Vec::new();
    let mut since_best = 0u32;
    let mut bad_run = 0u32;
    let mut skipped = 0u32;
    let (mut loss_acc, mut norm_acc, mut acc_n) = (0.0f64, 0.0f64, 0u32);
    let mut lr = cfg.learning_rate;
    let mut steps = 0u64;
    let mut samples = 0usize;

    while steps < cfg.max_steps {
        let batch = source.train_batch(cfg.batch_size);
        batch.validate().map_err(Error::Shape)?;
        samples += batch.size;
        match trainer.step_guarded(&task, std::slice::from_ref(&batch))? {
            Some(info) => {
                bad_run = 0;
                steps = info.step;
                lr = info.learning_rate;
                loss_acc += info.loss as f64;
                norm_acc += info.grad_norm as f64;
                acc_n += 1;
            }
            None => {
                skipped += 1;
                bad_run += 1;
                if bad_run >= cfg.max_bad_steps {
                    return Err(Error::Shape(format!(
                        "{bad_run} consecutive non-finite training steps at step {steps}"
                    )));
                }
                continue;
            }
        }
        if steps % cfg.eval_every == 0 || steps == cfg.max_steps {
            let point = eval_point(model, &*source, cfg, device, steps, (loss_acc, norm_acc, acc_n), lr)?;
            let improved = keep_if_better(model, &point, &mut best);
            log(&point);
            history.push(point);
            (loss_acc, norm_acc, acc_n) = (0.0, 0.0, 0);
            if improved {
                since_best = 0;
            } else {
                since_best += 1;
                if since_best >= cfg.patience {
                    break;
                }
            }
        }
    }
    if acc_n > 0 || history.is_empty() {
        let point = eval_point(model, &*source, cfg, device, steps, (loss_acc, norm_acc, acc_n), lr)?;
        keep_if_better(model, &point, &mut best);
        log(&point);
        history.push(point);
    }
    let (best_val_nll, best_step, state) = best.ok_or_else(|| {
        Error::Shape("no finite validation loss: is the validation split empty?".into())
    })?;
    model.load_state_dict(&state, true)?;
    model.invalidate();
    let seconds = start.elapsed().as_secs_f64();
    Ok(FitReport {
        best_step,
        best_val_nll,
        steps,
        skipped_steps: skipped,
        history,
        seconds,
        samples_per_second: samples as f64 / seconds.max(1e-9),
        peak_device_bytes: mamba3::backend::live_high_water_bytes(),
    })
}

/// The best weights so far: validation log loss, step, weights.
type Best = (f64, u64, mamba3::nn::StateDict);

fn eval_point<R: Runtime, E: FloatElem>(
    model: &DecisionModel<R, E>,
    source: &dyn BatchSource,
    cfg: &TrainConfig,
    device: &Device<R>,
    step: u64,
    (loss_acc, norm_acc, acc_n): (f64, f64, u32),
    lr: f32,
) -> Result<EvalPoint> {
    model.invalidate();
    let preds = predict_split(model, source, Split::Val, cfg.eval_batch, device)?;
    Ok(EvalPoint {
        step,
        train_loss: (loss_acc / acc_n.max(1) as f64) as f32,
        val_nll: preds.nll(),
        learning_rate: lr,
        grad_norm: (norm_acc / acc_n.max(1) as f64) as f32,
        live_device_bytes: mamba3::backend::live_bytes(),
    })
}

fn keep_if_better<R: Runtime, E: FloatElem>(
    model: &DecisionModel<R, E>,
    point: &EvalPoint,
    best: &mut Option<Best>,
) -> bool {
    let better = point.val_nll.is_finite() && best.as_ref().is_none_or(|(b, _, _)| point.val_nll < *b);
    if better {
        *best = Some((point.val_nll, point.step, model.state_dict()));
    }
    better
}
