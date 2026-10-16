//! Data-parallel training over several devices.
//!
//! Every device holds a full replica of the task — its own model, its own
//! optimizer — and takes its own share of each step's batches. After the
//! backward passes the replicas' gradients are summed ([`all_reduce_grads`]),
//! so every replica applies the same update to the same weights and they stay
//! identical, bit for bit, without the weights ever crossing devices again.
//!
//! One host thread drives every replica. That works because nothing between a
//! step's first launch and its gradient read waits for a device: the forward and
//! backward passes of replica 0 are queued, then replica 1's while device 0 is
//! still running, and so on, so the devices compute at the same time. A task
//! whose passes read from the device (an id read in an embedding backward, a
//! loss read for a decision) makes the host wait there, and the replicas then
//! run one after another — build ids with
//! [`IdTensor::from_host`](crate::tensor::ops::index::IdTensor::from_host) so
//! the embedding backward needs no read. `MAMBA3_DP_TIMING=1` prints each
//! step's phases to stderr; a `queue` phase as long as the device's step is
//! such a read.
//!
//! The gradients cross devices through the host, which every runtime supports:
//! each replica's are read once as raw bytes, those bytes are uploaded to every
//! other device unconverted, and each device adds the terms in replica order.
//! A step over `n` replicas, each given the micro-batches `shards[i]`, is the
//! step a single [`Trainer`] takes over all of them concatenated: same
//! averaging, same clip, same update — to the bit for two replicas, where each
//! sum is the one add the single trainer's accumulation makes.
//!
//! On two Tesla T4s (PCIe, no NVLink) with a 31.7M-parameter model in `f32`,
//! the exchange costs about 125 ms a step, so two devices train 1.34x the
//! tokens per second of one at a 0.28 s step and 1.75x at a 1.36 s step.
//!
//! Replicas are built by the caller, one per device, from the same
//! configuration: parameters are matched by position in
//! [`TrainStep::parameters`], not by name or id.
//!
//! [`Trainer`]: crate::train::Trainer

use cubecl::prelude::Runtime;

use crate::autograd::Grads;
use crate::backend::FloatElem;
use crate::error::{Error, Result};
use crate::nn::param::Param;
use crate::tensor::Tensor;
use crate::train::ema::Ema;
use crate::train::optim::Optimizer;
use crate::train::trainer::{
    StepInfo, TrainReport, TrainStep, TrainerConfig, backward_passes, clip_scale,
};

/// Phase timer for one data-parallel step, printed to stderr when
/// `MAMBA3_DP_TIMING` is set: where a step's wall clock goes between queueing
/// each replica's passes (`queueN`), reading its gradients (`readN`, which
/// waits for its device), uploading and summing them on each device (`sumN`)
/// and queueing the updates. Costs one `Instant::now` per phase when on, nothing when off.
struct Marks {
    start: std::time::Instant,
    marks: Vec<(String, std::time::Instant)>,
}

impl Marks {
    fn new() -> Option<Self> {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        ON.get_or_init(|| std::env::var_os("MAMBA3_DP_TIMING").is_some())
            .then(|| Self {
                start: std::time::Instant::now(),
                marks: Vec::new(),
            })
    }

    fn mark(marks: &mut Option<Self>, label: impl FnOnce() -> String) {
        if let Some(m) = marks {
            m.marks.push((label(), std::time::Instant::now()));
        }
    }

    fn print(self) {
        let mut last = self.start;
        let phases: Vec<String> = self
            .marks
            .iter()
            .map(|(label, at)| {
                let phase = format!("{label} {:.1}", (*at - last).as_secs_f64() * 1e3);
                last = *at;
                phase
            })
            .collect();
        eprintln!(
            "dp step {:.1} ms: {}",
            (last - self.start).as_secs_f64() * 1e3,
            phases.join(", ")
        );
    }
}

/// What [`DataParallel::on_step`] calls.
type StepCallback = Box<dyn FnMut(&StepInfo)>;

/// Check that every replica lists the same parameters: same count, and at each
/// position the same shape and the same frozen flag.
fn check_replicas<R: Runtime, E: FloatElem>(replicas: &[Vec<Param<R, E>>]) -> Result<()> {
    let Some(first) = replicas.first() else {
        return Err(Error::config("data parallelism needs at least one replica"));
    };
    for (r, params) in replicas.iter().enumerate().skip(1) {
        if params.len() != first.len() {
            return Err(Error::config(format!(
                "replica {r} has {} parameters and replica 0 has {}; build every replica \
                 from the same configuration",
                params.len(),
                first.len()
            )));
        }
        for (j, (a, b)) in first.iter().zip(params).enumerate() {
            if a.shape() != b.shape() {
                return Err(Error::config(format!(
                    "parameter {j} is {} on replica 0 and {} on replica {r}",
                    a.shape(),
                    b.shape()
                )));
            }
            if a.requires_grad() != b.requires_grad() {
                return Err(Error::config(format!(
                    "parameter {j} is frozen on one of replicas 0 and {r} but not the other"
                )));
            }
            if a.id() == b.id() {
                return Err(Error::config(format!(
                    "replicas 0 and {r} share parameter {j}; every replica must be its own \
                     model on its own device"
                )));
            }
        }
    }
    Ok(())
}

/// Copy replica 0's parameter values onto every other replica, so the replicas
/// start from the same weights however they were initialised. One read on
/// replica 0's device and one upload per parameter per other replica.
pub fn broadcast_parameters<R: Runtime, E: FloatElem>(replicas: &[Vec<Param<R, E>>]) -> Result<()> {
    check_replicas(replicas)?;
    if replicas.len() == 1 {
        return Ok(());
    }
    let values: Vec<Tensor<R, E>> = replicas[0].iter().map(Param::value).collect();
    let refs: Vec<&Tensor<R, E>> = values.iter().collect();
    let (_, host) = crate::tensor::ops::index::read_all(&[], &refs)?;
    for params in &replicas[1..] {
        for (param, data) in params.iter().zip(&host) {
            let device = param.value().device().clone();
            param.set(Tensor::from_f32(data, param.shape(), &device)?);
        }
    }
    Ok(())
}

/// Sum the replicas' gradients so every replica holds the total.
///
/// `grads[r]` are replica `r`'s gradients of the parameters `replicas[r]`; the
/// gradient of position `j` is summed over every replica that has one (a replica
/// without one contributes nothing) and written back into every replica's
/// [`Grads`], on that replica's device. Gradients of parameters a replica does
/// not list are dropped: no optimizer would apply them, and left in they would
/// enter each replica's gradient norm differently.
///
/// Each replica's gradients are read once, as raw bytes, and those bytes are
/// uploaded to every other device unconverted; each device then adds the terms
/// in replica order. An elementwise add is exact IEEE arithmetic, so every
/// replica computes the same sums to the bit. One replica is left exactly as
/// it is, with no read.
pub fn all_reduce_grads<R: Runtime, E: FloatElem>(
    replicas: &[Vec<Param<R, E>>],
    grads: &mut [Grads<R, E>],
) -> Result<()> {
    all_reduce_grads_marked(replicas, grads, &mut None)
}

fn all_reduce_grads_marked<R: Runtime, E: FloatElem>(
    replicas: &[Vec<Param<R, E>>],
    grads: &mut [Grads<R, E>],
    marks: &mut Option<Marks>,
) -> Result<()> {
    check_replicas(replicas)?;
    if grads.len() != replicas.len() {
        return Err(Error::config(format!(
            "{} gradient sets for {} replicas",
            grads.len(),
            replicas.len()
        )));
    }
    let n = replicas.len();
    if n == 1 {
        return Ok(());
    }
    let positions = replicas[0].len();

    // local[r][j]: replica r's own gradient of position j, if it has one.
    let mut local: Vec<Vec<Option<Tensor<R, E>>>> = Vec::with_capacity(n);
    for (params, replica_grads) in replicas.iter().zip(grads.iter_mut()) {
        let mut mine = Vec::with_capacity(positions);
        for (j, param) in params.iter().enumerate() {
            let grad = replica_grads.take(param.id());
            if let Some(g) = &grad
                && g.len() != param.numel()
            {
                return Err(Error::shape(format!(
                    "the gradient of parameter {j} holds {} values for a {} parameter",
                    g.len(),
                    param.shape()
                )));
            }
            mine.push(grad);
        }
        local.push(mine);
    }

    // One read per replica, of the gradients it has, as raw bytes. Every read
    // first checks that the replica's kernels ran.
    let mut bytes: Vec<Vec<Option<cubecl::bytes::Bytes>>> = Vec::with_capacity(n);
    for (r, mine) in local.iter().enumerate() {
        let present: Vec<&Tensor<R, E>> = mine.iter().flatten().filter(|g| !g.is_empty()).collect();
        let mut read = match present.first() {
            Some(first) => {
                crate::backend::check_launches(first.device())?;
                crate::backend::read_handles(
                    first.device(),
                    present.iter().map(|g| g.handle.clone()).collect(),
                )
            }
            None => Vec::new(),
        }
        .into_iter();
        bytes.push(
            mine.iter()
                .map(|g| match g {
                    Some(g) if !g.is_empty() => read.next(),
                    _ => None,
                })
                .collect(),
        );
        Marks::mark(marks, || format!("read{r}"));
    }

    // Every device sums the same terms in the same (replica) order.
    for (r, (params, replica_grads)) in replicas.iter().zip(grads.iter_mut()).enumerate() {
        let mut reduced = Grads::default();
        for (j, param) in params.iter().enumerate() {
            let device = param.value().device().clone();
            let mut sum: Option<Tensor<R, E>> = None;
            for s in 0..n {
                let term = if s == r {
                    local[r][j].clone()
                } else {
                    // The last device to receive these bytes takes them; the
                    // others upload a copy.
                    let last = if s == n - 1 { n - 2 } else { n - 1 };
                    let b = if r == last {
                        bytes[s][j].take()
                    } else {
                        bytes[s][j].clone()
                    };
                    match (&local[s][j], b) {
                        (Some(g), Some(b)) => {
                            Some(Tensor::from_bytes(b, g.shape().clone(), &device)?)
                        }
                        (Some(g), None) => Some(Tensor::zeros(g.shape().clone(), &device)),
                        (None, _) => None,
                    }
                };
                sum = match (sum, term) {
                    (Some(acc), Some(t)) => Some(crate::tensor::ops::elemwise::add(&acc, &t)?),
                    (acc, None) => acc,
                    (None, t) => t,
                };
            }
            if let Some(sum) = sum {
                reduced.accumulate(param.id(), sum)?;
            }
        }
        *replica_grads = reduced;
        Marks::mark(marks, || format!("sum{r}"));
    }
    Ok(())
}

/// An optimizer step of every replica on the device queues whose report has not
/// been read yet — see [`DataParallel::queue_step`].
pub struct ParallelStep<R: Runtime, E: FloatElem> {
    step: u64,
    learning_rate: f32,
    average: f32,
    norm_scale: f32,
    /// Each replica's micro-batch losses, on its own device.
    losses: Vec<Vec<Tensor<R, E>>>,
    /// Replica 0's sum of squares; every replica's is the same.
    sum_squares: Option<Tensor<R, E>>,
}

impl<R: Runtime, E: FloatElem> ParallelStep<R, E> {
    /// The 1-based optimizer step this is.
    pub fn step(&self) -> u64 {
        self.step
    }

    /// The `[1]` scalars replica `r` reads for this step's report.
    fn scalars(&self, r: usize) -> impl Iterator<Item = &Tensor<R, E>> {
        self.losses[r]
            .iter()
            .chain(self.sum_squares.as_ref().filter(|_| r == 0))
    }

    /// The report from the values of each replica's [`ParallelStep::scalars`].
    fn info(&self, values: &[Vec<f32>]) -> StepInfo {
        let losses: f32 = values
            .iter()
            .zip(&self.losses)
            .flat_map(|(v, l)| &v[..l.len()])
            .map(|loss| loss * self.average)
            .sum();
        let grad_norm = self.sum_squares.as_ref().map_or(0.0, |_| {
            (values[0][self.losses[0].len()] * self.norm_scale * self.norm_scale).sqrt()
        });
        StepInfo {
            step: self.step,
            loss: losses,
            learning_rate: self.learning_rate,
            grad_norm,
        }
    }
}

/// Drives data-parallel optimization: [`Trainer`](crate::train::Trainer) over
/// several replicas, one per device.
pub struct DataParallel<R: Runtime, E: FloatElem, O: Optimizer<R, E>> {
    config: TrainerConfig,
    optimizers: Vec<O>,
    step: u64,
    on_step: Option<StepCallback>,
    ema: Option<Ema<R, E>>,
}

impl<R: Runtime, E: FloatElem, O: Optimizer<R, E>> DataParallel<R, E, O> {
    /// One optimizer per replica, in replica order, all built with the same
    /// settings (each updates its own replica's weights and keeps its own state,
    /// which stays identical to the others' because their gradients are).
    pub fn new(config: TrainerConfig, optimizers: Vec<O>) -> Result<Self> {
        if optimizers.is_empty() {
            return Err(Error::config("data parallelism needs at least one replica"));
        }
        Ok(Self {
            config,
            optimizers,
            step: 0,
            on_step: None,
            ema: None,
        })
    }

    /// Number of replicas.
    pub fn replicas(&self) -> usize {
        self.optimizers.len()
    }

    /// Register a callback invoked every `log_every` steps.
    pub fn on_step(mut self, callback: impl FnMut(&StepInfo) + 'static) -> Self {
        self.on_step = Some(Box::new(callback));
        self
    }

    /// Keep an exponential moving average of replica 0's weights (build it over
    /// replica 0's model), updated after every optimizer step.
    pub fn with_ema(mut self, ema: Ema<R, E>) -> Self {
        self.ema = Some(ema);
        self
    }

    /// The moving average, if one is attached.
    pub fn ema(&self) -> Option<&Ema<R, E>> {
        self.ema.as_ref()
    }

    /// Mutable access to the moving average.
    pub fn ema_mut(&mut self) -> Option<&mut Ema<R, E>> {
        self.ema.as_mut()
    }

    /// Replica `r`'s optimizer. Save checkpoints from replica 0's.
    pub fn optimizer(&self, r: usize) -> &O {
        &self.optimizers[r]
    }

    /// Every replica's optimizer, e.g. to load the same state into each.
    pub fn optimizers_mut(&mut self) -> &mut [O] {
        &mut self.optimizers
    }

    /// Optimizer steps taken so far.
    pub fn step_count(&self) -> u64 {
        self.step
    }

    /// Set the step counter, e.g. after restoring a checkpoint (see
    /// [`Trainer::set_step_count`](crate::train::Trainer::set_step_count)).
    pub fn set_step_count(&mut self, step: u64) {
        self.step = step;
    }

    /// Copy replica 0's weights onto the others; see [`broadcast_parameters`].
    /// Call once before training unless every replica was built (or loaded)
    /// with the same weights.
    pub fn broadcast<T: TrainStep<R, E>>(&self, tasks: &[T]) -> Result<()> {
        self.check_tasks(tasks)?;
        broadcast_parameters(&tasks.iter().map(TrainStep::parameters).collect::<Vec<_>>())
    }

    fn check_tasks<T>(&self, tasks: &[T]) -> Result<()> {
        if tasks.len() != self.replicas() {
            return Err(Error::config(format!(
                "{} tasks for {} replicas",
                tasks.len(),
                self.replicas()
            )));
        }
        Ok(())
    }

    /// One optimizer step: replica `r` runs the micro-batches `shards[r]`, which
    /// must live on its device. Every shard needs at least one micro-batch; they
    /// need not have the same number. The reported loss is the mean over every
    /// micro-batch of every replica.
    pub fn step<T: TrainStep<R, E>, S: AsRef<[T::Batch]>>(
        &mut self,
        tasks: &[T],
        shards: &[S],
    ) -> Result<StepInfo> {
        let queued = self.queue_step(tasks, shards)?;
        Ok(self.read_steps(std::slice::from_ref(&queued))?[0])
    }

    /// [`DataParallel::step`] up to the point where it reads (see
    /// [`Trainer::queue_step`](crate::train::Trainer::queue_step)). The gradient
    /// sum reads every replica's gradients, so this waits for the backward
    /// passes; the update itself is left on the queues.
    pub fn queue_step<T: TrainStep<R, E>, S: AsRef<[T::Batch]>>(
        &mut self,
        tasks: &[T],
        shards: &[S],
    ) -> Result<ParallelStep<R, E>> {
        self.check_tasks(tasks)?;
        if shards.len() != tasks.len() {
            return Err(Error::config(format!(
                "{} shards for {} replicas",
                shards.len(),
                tasks.len()
            )));
        }
        if let Some(r) = shards.iter().position(|s| s.as_ref().is_empty()) {
            return Err(Error::config(format!(
                "replica {r}'s shard is empty; every replica needs at least one batch"
            )));
        }
        let params: Vec<Vec<Param<R, E>>> = tasks.iter().map(TrainStep::parameters).collect();
        check_replicas(&params)?;

        let micro_batches: usize = shards.iter().map(|s| s.as_ref().len()).sum();
        let average = 1.0 / micro_batches as f32;

        // Every replica's passes go on its queue before anything is read, so the
        // devices run them at the same time.
        let mut marks = Marks::new();
        let mut grads = Vec::with_capacity(tasks.len());
        let mut losses = Vec::with_capacity(tasks.len());
        for (r, (task, shard)) in tasks.iter().zip(shards).enumerate() {
            let (g, l) = backward_passes(task, shard.as_ref(), self.config.loss_scale)?;
            grads.push(g);
            losses.push(l);
            Marks::mark(&mut marks, || format!("queue{r}"));
        }
        all_reduce_grads_marked(&params, &mut grads, &mut marks)?;

        let scales = grads
            .iter()
            .map(|g| clip_scale(&self.config, g, average))
            .collect::<Result<Vec<_>>>()?;
        // Before any update: an optimizer step applied to gradients whose kernels
        // never ran would overwrite the weights with garbage, and one replica
        // updated without the others would leave them different models.
        for replica_losses in &losses {
            if let Some(loss) = replica_losses.first() {
                crate::backend::check_launches(loss.device())?;
            }
        }

        self.step += 1;
        let lr = self
            .config
            .schedule
            .at(self.config.learning_rate, self.step);
        let mut report: Option<(Option<Tensor<R, E>>, f32)> = None;
        let replicas = params.iter().zip(&grads).zip(scales);
        for (r, ((replica_params, replica_grads), (scaling, norm_scale))) in replicas.enumerate() {
            let _opt_scope = crate::backend::tally_scope("optimizer");
            let optimizer = &mut self.optimizers[r];
            optimizer.set_learning_rate(lr);
            optimizer.step_scaled(
                replica_params,
                replica_grads,
                scaling.as_ref().map(|s| &s.factor),
            )?;
            if r == 0 {
                report = Some((scaling.map(|s| s.sum_squares), norm_scale));
            }
        }
        if let Some(ema) = &mut self.ema {
            ema.update(self.step)?;
        }
        Marks::mark(&mut marks, || "update".into());
        if let Some(m) = marks {
            m.print();
        }
        let (sum_squares, norm_scale) = report.expect("at least one replica");
        Ok(ParallelStep {
            step: self.step,
            learning_rate: lr,
            average,
            norm_scale,
            losses,
            sum_squares,
        })
    }

    /// Read the reports of queued steps, oldest first, with one read per
    /// replica, running the `on_step` callback for each.
    pub fn read_steps(&mut self, queued: &[ParallelStep<R, E>]) -> Result<Vec<StepInfo>> {
        let replicas = self.replicas();
        if let Some(q) = queued.iter().find(|q| q.losses.len() != replicas) {
            return Err(Error::config(format!(
                "step {} was queued over {} replicas, this trainer has {replicas}",
                q.step,
                q.losses.len()
            )));
        }
        // values[r] = replica r's scalars of every step, concatenated.
        let mut values = Vec::with_capacity(replicas);
        for r in 0..replicas {
            let scalars: Vec<&Tensor<R, E>> = queued.iter().flat_map(|q| q.scalars(r)).collect();
            let (_, read) = crate::tensor::ops::index::read_all(&[], &scalars)?;
            values.push(read.into_iter().map(|v| v[0]).collect::<Vec<f32>>());
        }
        let mut offsets = vec![0usize; replicas];
        let mut infos = Vec::with_capacity(queued.len());
        for q in queued {
            let mine: Vec<Vec<f32>> = (0..replicas)
                .map(|r| {
                    let n = q.scalars(r).count();
                    let slice = values[r][offsets[r]..offsets[r] + n].to_vec();
                    offsets[r] += n;
                    slice
                })
                .collect();
            let info = q.info(&mine);
            if info.step.is_multiple_of(self.config.log_every)
                && let Some(cb) = &mut self.on_step
            {
                cb(&info);
            }
            infos.push(info);
        }
        Ok(infos)
    }

    /// Consume rounds of batches — one batch per replica each, in replica order
    /// — until they run out or `max_steps` is reached. `grad_accumulation`
    /// rounds make one optimizer step.
    pub fn fit<T: TrainStep<R, E>>(
        &mut self,
        tasks: &[T],
        rounds: impl IntoIterator<Item = Vec<T::Batch>>,
    ) -> Result<TrainReport> {
        let accumulation = self.config.grad_accumulation as usize;
        let mut report = TrainReport::default();
        let mut pending: Vec<Vec<T::Batch>> = (0..self.replicas()).map(|_| Vec::new()).collect();
        let mut rounds_pending = 0usize;
        let mut take_step =
            |this: &mut Self, pending: &mut Vec<Vec<T::Batch>>| -> Result<StepInfo> {
                let info = this.step(tasks, pending)?;
                pending.iter_mut().for_each(Vec::clear);
                report.steps = info.step;
                report.final_loss = info.loss;
                report.losses.push(info.loss);
                Ok(info)
            };

        for round in rounds {
            if round.len() != self.replicas() {
                return Err(Error::config(format!(
                    "a round of {} batches for {} replicas",
                    round.len(),
                    self.replicas()
                )));
            }
            for (shard, batch) in pending.iter_mut().zip(round) {
                shard.push(batch);
            }
            rounds_pending += 1;
            if rounds_pending < accumulation {
                continue;
            }
            rounds_pending = 0;
            take_step(self, &mut pending)?;
            if let Some(max) = self.config.max_steps
                && self.step >= max
            {
                return Ok(report);
            }
        }
        // A trailing partial accumulation window still deserves a step.
        if rounds_pending > 0 {
            take_step(self, &mut pending)?;
        }
        Ok(report)
    }
}
