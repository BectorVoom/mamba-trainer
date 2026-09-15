//! The training loop.
//!
//! The loop knows nothing about models. It drives a [`TrainStep`], which is the
//! one thing a task must supply: the parameters to update and a scalar loss for a
//! batch. Language modelling, image classification, LoRA fine-tuning and QAT are
//! all the same loop with a different `TrainStep`.

use cubecl::prelude::Runtime;

use crate::autograd::Var;
use crate::backend::FloatElem;
use crate::error::{Error, Result};
use crate::nn::param::Param;
use crate::tensor::Tensor;
use crate::train::ema::Ema;
use crate::train::optim::Optimizer;
use crate::train::sched::LrSchedule;

/// A task the trainer can optimise.
pub trait TrainStep<R: Runtime, E: FloatElem> {
    /// One unit of work.
    type Batch;

    /// The parameters this task updates. Frozen parameters may be included; the
    /// optimizer skips them.
    fn parameters(&self) -> Vec<Param<R, E>>;

    /// Scalar loss for a batch, recorded on the tape.
    fn loss(&self, batch: &Self::Batch) -> Result<Var<R, E>>;

    /// Switch dropout and observers between train and eval behaviour.
    fn set_training(&self, _training: bool) {}
}

/// What happened during one optimizer step.
#[derive(Debug, Clone, Copy)]
pub struct StepInfo {
    /// 1-based optimizer step.
    pub step: u64,
    /// Mean loss across the accumulated micro-batches.
    pub loss: f32,
    /// Learning rate actually applied.
    pub learning_rate: f32,
    /// Global gradient norm before clipping.
    pub grad_norm: f32,
}

/// Summary of a run.
#[derive(Debug, Clone, Default)]
pub struct TrainReport {
    /// Optimizer steps taken.
    pub steps: u64,
    /// Loss at each step.
    pub losses: Vec<f32>,
    /// Loss of the final step.
    pub final_loss: f32,
}

/// Configuration for [`Trainer`].
#[derive(Debug, Clone)]
pub struct TrainerConfig {
    /// Stop after this many optimizer steps. `None` runs until the data ends.
    pub max_steps: Option<u64>,
    /// Micro-batches accumulated into one optimizer step.
    pub grad_accumulation: u32,
    /// Global gradient-norm clip. `0` disables clipping.
    pub max_grad_norm: f32,
    /// Base learning rate handed to the schedule.
    pub learning_rate: f32,
    /// Schedule applied per optimizer step.
    pub schedule: LrSchedule,
    /// Emit a step callback every `log_every` steps.
    pub log_every: u64,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            max_steps: None,
            grad_accumulation: 1,
            max_grad_norm: 1.0,
            learning_rate: 1e-3,
            schedule: LrSchedule::Constant,
            log_every: 1,
        }
    }
}

impl TrainerConfig {
    /// Start a builder.
    pub fn builder() -> TrainerConfigBuilder {
        TrainerConfigBuilder::default()
    }
}

/// Builder for [`TrainerConfig`].
#[derive(Debug, Clone, Default)]
pub struct TrainerConfigBuilder {
    config: Option<TrainerConfig>,
}

impl TrainerConfigBuilder {
    fn edit(mut self, f: impl FnOnce(&mut TrainerConfig)) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        f(&mut config);
        self.config = Some(config);
        self
    }

    /// Stop after this many optimizer steps.
    pub fn max_steps(self, steps: u64) -> Self {
        self.edit(|c| c.max_steps = Some(steps))
    }

    /// Micro-batches per optimizer step.
    pub fn grad_accumulation(self, n: u32) -> Self {
        self.edit(|c| c.grad_accumulation = n.max(1))
    }

    /// Global gradient-norm clip.
    pub fn max_grad_norm(self, norm: f32) -> Self {
        self.edit(|c| c.max_grad_norm = norm)
    }

    /// Base learning rate.
    pub fn learning_rate(self, lr: f32) -> Self {
        self.edit(|c| c.learning_rate = lr)
    }

    /// Learning-rate schedule.
    pub fn schedule(self, schedule: LrSchedule) -> Self {
        self.edit(|c| c.schedule = schedule)
    }

    /// Callback frequency.
    pub fn log_every(self, n: u64) -> Self {
        self.edit(|c| c.log_every = n.max(1))
    }

    /// Validate and build.
    pub fn build(self) -> Result<TrainerConfig> {
        let config = self.config.unwrap_or_default();
        if config.learning_rate <= 0.0 {
            return Err(Error::config("learning rate must be positive"));
        }
        config.schedule.validate()?;
        Ok(config)
    }
}

/// What [`Trainer::on_step`] calls.
type StepCallback = Box<dyn FnMut(&StepInfo)>;

/// An optimizer step on the device queue whose report has not been read yet —
/// see [`Trainer::queue_step`].
pub struct QueuedStep<R: Runtime, E: FloatElem> {
    step: u64,
    learning_rate: f32,
    average: f32,
    losses: Vec<Tensor<R, E>>,
    sum_squares: Option<Tensor<R, E>>,
}

impl<R: Runtime, E: FloatElem> QueuedStep<R, E> {
    /// The 1-based optimizer step this is.
    pub fn step(&self) -> u64 {
        self.step
    }

    /// The `[1]` device scalars this step's [`StepInfo`] is computed from: each
    /// micro-batch's loss, then the gradients' sum of squares if there were any.
    pub fn scalars(&self) -> impl Iterator<Item = &Tensor<R, E>> {
        self.losses.iter().chain(self.sum_squares.as_ref())
    }

    fn scalar_count(&self) -> usize {
        self.losses.len() + usize::from(self.sum_squares.is_some())
    }

    /// The report, from the first value of each of [`QueuedStep::scalars`].
    fn info(&self, values: &[f32]) -> StepInfo {
        let (losses, sum_squares) = values.split_at(self.losses.len());
        StepInfo {
            step: self.step,
            loss: losses.iter().map(|loss| loss * self.average).sum(),
            learning_rate: self.learning_rate,
            // The reported norm is the one *before* clipping, as it always was: the
            // sum of squares this came from was reduced before the factor was applied.
            grad_norm: sum_squares
                .first()
                .map_or(0.0, |s| (s * self.average * self.average).sqrt()),
        }
    }
}

/// Drives optimization.
pub struct Trainer<R: Runtime, E: FloatElem, O: Optimizer<R, E>> {
    config: TrainerConfig,
    optimizer: O,
    step: u64,
    on_step: Option<StepCallback>,
    ema: Option<Ema<R, E>>,
    _marker: core::marker::PhantomData<(R, E)>,
}

impl<R: Runtime, E: FloatElem, O: Optimizer<R, E>> Trainer<R, E, O> {
    /// Build a trainer around an optimizer.
    pub fn new(config: TrainerConfig, optimizer: O) -> Self {
        Self {
            config,
            optimizer,
            step: 0,
            on_step: None,
            ema: None,
            _marker: core::marker::PhantomData,
        }
    }

    /// Register a callback invoked every `log_every` steps.
    pub fn on_step(mut self, callback: impl FnMut(&StepInfo) + 'static) -> Self {
        self.on_step = Some(Box::new(callback));
        self
    }

    /// Keep an exponential moving average of the weights, updated after every
    /// optimizer step — see [`Ema`]. Replaces any average already attached.
    ///
    /// Attaching one changes nothing about training: the losses, learning
    /// rates, gradient norms, weights and optimizer state are those of the same
    /// run without it, to the bit.
    pub fn with_ema(mut self, ema: Ema<R, E>) -> Self {
        self.ema = Some(ema);
        self
    }

    /// The moving average, if one is attached.
    pub fn ema(&self) -> Option<&Ema<R, E>> {
        self.ema.as_ref()
    }

    /// Mutable access to the moving average, to reset or restore it.
    pub fn ema_mut(&mut self) -> Option<&mut Ema<R, E>> {
        self.ema.as_mut()
    }

    /// Detach the moving average, e.g. to move it onto a trainer rebuilt from a
    /// checkpoint.
    pub fn take_ema(&mut self) -> Option<Ema<R, E>> {
        self.ema.take()
    }

    /// The optimizer.
    pub fn optimizer(&self) -> &O {
        &self.optimizer
    }

    /// Mutable access to the optimizer.
    pub fn optimizer_mut(&mut self) -> &mut O {
        &mut self.optimizer
    }

    /// Optimizer steps taken so far.
    pub fn step_count(&self) -> u64 {
        self.step
    }

    /// Set the step counter directly, e.g. after restoring a checkpoint so the
    /// schedule ([`TrainerConfig::schedule`]) resumes at the position it left
    /// off at rather than restarting at zero. Does not touch the optimizer's
    /// own counter — restore that separately through
    /// [`Optimizer::load_state_dict`], normally with the same value.
    pub fn set_step_count(&mut self, step: u64) {
        self.step = step;
    }

    /// Run one optimizer step over `micro_batches` accumulated micro-batches.
    ///
    /// With an [`Ema`] attached, its update is queued right after the
    /// optimizer's, before the step's reads. A step that fails before the
    /// optimizer runs (no batches, a refused loss) leaves the weights and the
    /// average untouched; one that fails after it leaves both unspecified.
    pub fn step<T: TrainStep<R, E>>(
        &mut self,
        task: &T,
        micro_batches: &[T::Batch],
    ) -> Result<StepInfo> {
        let queued = self.queue_step(task, micro_batches)?;
        Ok(self.read_steps(std::slice::from_ref(&queued))?[0])
    }

    /// [`Trainer::step`] up to the point where it reads: the whole step is on the
    /// device queue — the optimizer update and any [`Ema`] update included — and
    /// the step counter has advanced, but the loss and gradient norm are still
    /// device scalars.
    ///
    /// Each read is a fixed wait for the device (~1.4 ms on wgpu, whatever its
    /// size), so a loop taking several steps between looks at the numbers — PPO's
    /// epochs and minibatches — queues them all and hands them to
    /// [`Trainer::read_steps`] together, or reads [`QueuedStep::scalars`] with its
    /// own values and calls [`Trainer::report_steps`]. The `on_step` callback runs
    /// then, not here.
    pub fn queue_step<T: TrainStep<R, E>>(
        &mut self,
        task: &T,
        micro_batches: &[T::Batch],
    ) -> Result<QueuedStep<R, E>> {
        if micro_batches.is_empty() {
            return Err(Error::config("an optimizer step needs at least one batch"));
        }
        task.set_training(true);

        // Nothing in this method reads a device value until the whole step is on the
        // queue. That is deliberate and it is worth more than it looks: reading the
        // loss between the forward pass and the backward pass, as this used to, drains
        // the pipeline in the middle of every step, and reading the gradient norm to
        // decide a clip factor drains it again before the update. On a busy machine
        // those two stalls were the difference between the median step and the best
        // one. Both values are still reported — they are just read at the end, by
        // which point the work that produces them is already running.
        let average = 1.0 / micro_batches.len() as f32;
        let mut accumulated: Option<crate::autograd::Grads<R, E>> = None;
        let mut losses: Vec<Tensor<R, E>> = Vec::with_capacity(micro_batches.len());

        for batch in micro_batches {
            let loss = task.loss(batch)?;
            losses.push(loss.tensor().clone());
            let grads = loss.backward()?;
            accumulated = Some(match accumulated {
                Some(mut acc) => {
                    acc.merge(grads)?;
                    acc
                }
                None => grads,
            });
        }

        // Micro-batch averaging is folded into the same factor as the clip, so the
        // per-gradient rescale that used to apply it disappears too.
        let grads = accumulated.expect("at least one micro-batch");
        let scaling = crate::train::optim::grad_scale(&grads, self.config.max_grad_norm, average)?;

        // Before the update, not after: an optimizer step applied to gradients
        // whose kernels never ran would overwrite the weights with garbage that a
        // later error could no longer undo.
        if let Some(loss) = losses.first() {
            crate::backend::check_launches(loss.device())?;
        }

        self.step += 1;
        let lr = self
            .config
            .schedule
            .at(self.config.learning_rate, self.step);
        self.optimizer.set_learning_rate(lr);
        self.optimizer.step_scaled(
            &task.parameters(),
            &grads,
            scaling.as_ref().map(|s| &s.factor),
        )?;
        if let Some(ema) = &mut self.ema {
            ema.update(self.step)?;
        }

        Ok(QueuedStep {
            step: self.step,
            learning_rate: lr,
            average,
            losses,
            sum_squares: scaling.map(|s| s.sum_squares),
        })
    }

    /// Read the reports of steps [`Trainer::queue_step`] queued, oldest first,
    /// under one synchronisation, running the `on_step` callback for each.
    ///
    /// The read checks first that every kernel queued so far actually ran, so a
    /// step whose kernels failed to launch is an error rather than a report
    /// computed from zeros.
    pub fn read_steps(&mut self, queued: &[QueuedStep<R, E>]) -> Result<Vec<StepInfo>> {
        let scalars: Vec<&Tensor<R, E>> = queued.iter().flat_map(QueuedStep::scalars).collect();
        let (_, values) = crate::tensor::ops::index::read_all(&[], &scalars)?;
        let values: Vec<f32> = values.iter().map(|v| v[0]).collect();
        Ok(self.report_steps(queued, &values))
    }

    /// The reports of `queued` from values the caller read itself: the first
    /// element of every tensor of each step's [`QueuedStep::scalars`], all steps
    /// concatenated in order. Runs the `on_step` callback for each step.
    ///
    /// # Panics
    ///
    /// If `values` is not exactly that many numbers.
    pub fn report_steps(&mut self, queued: &[QueuedStep<R, E>], values: &[f32]) -> Vec<StepInfo> {
        let wanted: usize = queued.iter().map(QueuedStep::scalar_count).sum();
        assert_eq!(
            values.len(),
            wanted,
            "report_steps needs one value per queued step scalar"
        );
        let mut rest = values;
        queued
            .iter()
            .map(|q| {
                let (mine, others) = rest.split_at(q.scalar_count());
                rest = others;
                let info = q.info(mine);
                if info.step.is_multiple_of(self.config.log_every)
                    && let Some(cb) = &mut self.on_step
                {
                    cb(&info);
                }
                info
            })
            .collect()
    }

    /// Consume batches until they run out or `max_steps` is reached.
    pub fn fit<T: TrainStep<R, E>>(
        &mut self,
        task: &T,
        batches: impl IntoIterator<Item = T::Batch>,
    ) -> Result<TrainReport> {
        let accumulation = self.config.grad_accumulation as usize;
        let mut report = TrainReport::default();
        let mut pending: Vec<T::Batch> = Vec::with_capacity(accumulation);

        for batch in batches {
            pending.push(batch);
            if pending.len() < accumulation {
                continue;
            }
            let info = self.step(task, &pending)?;
            pending.clear();
            report.steps = info.step;
            report.final_loss = info.loss;
            report.losses.push(info.loss);
            if let Some(max) = self.config.max_steps
                && self.step >= max
            {
                return Ok(report);
            }
        }
        // A trailing partial accumulation window still deserves a step.
        if !pending.is_empty() {
            let info = self.step(task, &pending)?;
            report.steps = info.step;
            report.final_loss = info.loss;
            report.losses.push(info.loss);
        }
        Ok(report)
    }

    /// Mean loss over batches, with no gradient tracking.
    pub fn evaluate<T: TrainStep<R, E>>(
        &self,
        task: &T,
        batches: impl IntoIterator<Item = T::Batch>,
    ) -> Result<f32> {
        // Leave the task in eval mode; `step` turns training back on when it runs.
        task.set_training(false);
        let mut total = 0.0f32;
        let mut count = 0usize;
        for batch in batches {
            total += task.loss(&batch)?.try_to_f32()?[0];
            count += 1;
        }
        Ok(if count == 0 {
            0.0
        } else {
            total / count as f32
        })
    }
}
