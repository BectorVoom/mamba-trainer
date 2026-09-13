//! Collecting from many environments at once, on worker threads.
//!
//! [`Collector`] drives one [`VecEnv`] from the thread that owns
//! the policy. That is the right shape when the environment is itself a batched
//! kernel, because then the environment *is* already parallel. It is the wrong
//! shape as soon as the environment is real work on the host — a physics step, a
//! game tick, a simulator behind a socket — because then the device sits idle for
//! the whole of it, once per step, forever.
//!
//! [`ParallelEnvs`] fixes that by putting each environment on its own thread and
//! joining them at a barrier every step. It is itself a [`VecEnv`], so it drops
//! into the existing collector with nothing else changed, and
//! [`MultiSyncCollector`] is the two bundled together.
//!
//! # How this differs from TorchRL's `MultiSyncDataCollector`
//!
//! Same contract — `W` workers, each with its own environment, a synchronous
//! barrier every batch, and one concatenated result — but the work is split down a
//! different seam, for two reasons that are specific to this crate.
//!
//! **The workers hold environments, not policies.** TorchRL gives each process a
//! replica of the policy because Python cannot run one in parallel and because its
//! environments are usually Python objects. Here the policy is evaluated once, on
//! the owning thread, over the concatenated observations of every worker. That is
//! not a compromise: a rollout step is ~140 kernel dispatches whatever the batch
//! width, so `W` replicas of the policy would multiply the dispatch count by `W`
//! and shrink each launch — precisely the wrong trade on a device where launch
//! overhead dominates a small model. One wide forward pass beats `W` narrow ones.
//!
//! **There is no `update_policy_weights_()`, because there is nothing to update.**
//! Worker replicas are what make weight synchronisation necessary in the first
//! place. Here the single policy on the owner thread is the one that acts, so a
//! step taken by the optimizer is visible to the very next rollout step. The whole
//! class of bug where workers quietly act on stale weights cannot arise.
//!
//! The mechanism is threads rather than processes, which Rust can do and Python
//! cannot: there is no interpreter lock to escape, so the address space is shared,
//! observations are never serialised, and a worker hands back a device tensor by
//! moving a handle.
//!
//! # What it does not do
//!
//! Workers step in lockstep: the barrier means one slow environment holds up the
//! batch, and the device is idle while the environments run and vice versa.
//! Overlapping those two phases is what TorchRL's *asynchronous* collector is for,
//! and it is not implemented here — it needs a second trajectory buffer and changes
//! which policy the data is on, which is a decision for a caller rather than a
//! default.

use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread::JoinHandle;

use cubecl::prelude::Runtime;

use crate::backend::{Device, FloatElem};
use crate::error::{Error, Result};
use crate::tensor::Tensor;
use crate::tensor::ops::index::{IdTensor, cat_ids, slice_ids};
use crate::tensor::ops::movement;

use super::buffer::TrajectoryBuffer;
use super::collect::{CollectReport, Collector};
use super::env::{EnvStep, VecEnv};
use super::imitation::ImitationBatch;
use super::policy::Mamba3Policy;
use super::ppo::{PpoBatch, PpoConfig};

/// What the owner thread asks a worker to do.
enum Command<R: Runtime> {
    /// Start the episode over and report the first observation.
    Reset,
    /// Apply this worker's slice of the batched actions.
    Step(IdTensor<R>),
    /// Finish and let the thread end.
    Shutdown,
}

/// What a worker sends back.
struct Reply<R: Runtime, E: FloatElem> {
    observation: Tensor<R, E>,
    /// Absent after a reset, which earns nothing and ends nothing.
    reward: Option<Tensor<R, E>>,
    done: Option<Tensor<R, E>>,
    /// The expert's action for `observation`, if the environment can say.
    expert: Option<IdTensor<R>>,
}

struct Worker<R: Runtime, E: FloatElem> {
    commands: Sender<Command<R>>,
    replies: Receiver<Result<Reply<R, E>>>,
    thread: Option<JoinHandle<()>>,
    envs: usize,
    offset: usize,
}

/// Several [`VecEnv`]s driven together, one thread each, presented as one.
///
/// The environments keep their own state and their own random draws; what they
/// share is the step boundary. Row `offset .. offset + envs` of every tensor this
/// produces belongs to one worker, in the order they were given.
pub struct ParallelEnvs<R: Runtime, E: FloatElem> {
    workers: Vec<Worker<R, E>>,
    total: usize,
    obs_dim: usize,
    action_dim: usize,
    /// The concatenated expert labels for the most recent observation, cached
    /// because [`VecEnv::expert_actions`] cannot ask the workers again — by then
    /// their environments have moved on.
    expert: Option<IdTensor<R>>,
    device: Device<R>,
}

impl<R: Runtime, E: FloatElem> ParallelEnvs<R, E> {
    /// Put each environment on a thread of its own.
    ///
    /// Every environment must agree on the observation width and the action space;
    /// they may differ in how many environments each one batches, which is how a
    /// heterogeneous pool of machines or simulators is expressed.
    pub fn new<V>(envs: Vec<V>, device: &Device<R>) -> Result<Self>
    where
        V: VecEnv<R, E> + Send + 'static,
    {
        if envs.is_empty() {
            return Err(Error::config(
                "a parallel collector needs at least one environment".to_string(),
            ));
        }
        let obs_dim = envs[0].obs_dim();
        let action_dim = envs[0].action_dim();
        for (i, env) in envs.iter().enumerate() {
            if env.obs_dim() != obs_dim || env.action_dim() != action_dim {
                return Err(Error::shape(format!(
                    "worker {i} has observations of width {} over {} actions, \
                     but worker 0 has {obs_dim} over {action_dim}; \
                     the batch they are concatenated into has to be rectangular",
                    env.obs_dim(),
                    env.action_dim(),
                )));
            }
            if env.envs() == 0 {
                return Err(Error::config(format!("worker {i} drives no environments")));
            }
        }

        let mut workers = Vec::with_capacity(envs.len());
        let mut offset = 0;
        for (index, mut env) in envs.into_iter().enumerate() {
            let count = env.envs();
            let (commands, command_rx) = channel::<Command<R>>();
            let (reply_tx, replies) = channel::<Result<Reply<R, E>>>();
            let thread = std::thread::Builder::new()
                .name(format!("mamba3-env-{index}"))
                .spawn(move || {
                    // The worker owns its environment outright for the whole of its
                    // life, so nothing is shared and nothing needs a lock.
                    while let Ok(command) = command_rx.recv() {
                        let reply = match command {
                            Command::Shutdown => break,
                            Command::Reset => env.reset().map(|observation| Reply {
                                observation,
                                reward: None,
                                done: None,
                                expert: env.expert_actions(),
                            }),
                            Command::Step(actions) => env.step(&actions).map(|step| Reply {
                                observation: step.observation,
                                reward: Some(step.reward),
                                done: Some(step.done),
                                expert: env.expert_actions(),
                            }),
                        };
                        // A closed channel means the owner is gone; stop quietly
                        // rather than panicking a detached thread.
                        if reply_tx.send(reply).is_err() {
                            break;
                        }
                    }
                })
                .map_err(|e| {
                    Error::config(format!("could not start an environment thread: {e}"))
                })?;

            workers.push(Worker {
                commands,
                replies,
                thread: Some(thread),
                envs: count,
                offset,
            });
            offset += count;
        }

        Ok(Self {
            workers,
            total: offset,
            obs_dim,
            action_dim,
            expert: None,
            device: device.clone(),
        })
    }

    /// Number of worker threads.
    pub fn workers(&self) -> usize {
        self.workers.len()
    }

    /// How many environments each worker drives, in order.
    pub fn widths(&self) -> Vec<usize> {
        self.workers.iter().map(|w| w.envs).collect()
    }

    /// The device the joined batches are built on.
    pub fn device(&self) -> &Device<R> {
        &self.device
    }

    /// Send every worker a command, then wait for every answer.
    ///
    /// The fan-out is a loop of sends that do not block, so by the time the first
    /// answer is waited on all of the workers are already running. That is the
    /// whole of the parallelism, and it is why the sends and the receives are two
    /// loops rather than one.
    fn exchange(
        &mut self,
        mut command: impl FnMut(&Worker<R, E>) -> Result<Command<R>>,
    ) -> Result<Vec<Reply<R, E>>> {
        for (index, worker) in self.workers.iter().enumerate() {
            let message = command(worker)?;
            worker.commands.send(message).map_err(|_| {
                Error::config(format!("environment worker {index} stopped unexpectedly"))
            })?;
        }
        let mut replies = Vec::with_capacity(self.workers.len());
        for (index, worker) in self.workers.iter().enumerate() {
            let reply = worker.replies.recv().map_err(|_| {
                Error::config(format!(
                    "environment worker {index} stopped before answering; \
                     it panicked or its environment returned an error it could not send"
                ))
            })?;
            replies.push(reply?);
        }
        Ok(replies)
    }

    /// Join the workers' answers into one batch, and cache the expert labels.
    fn join(&mut self, replies: Vec<Reply<R, E>>) -> Result<Joined<R, E>> {
        // Every worker either knows its expert or none of them do; a half-labelled
        // batch would silently train on whatever the uninitialised rows held.
        self.expert = if replies.iter().all(|r| r.expert.is_some()) {
            let parts: Vec<IdTensor<R>> = replies
                .iter()
                .map(|r| r.expert.clone().expect("checked just above"))
                .collect();
            Some(if parts.len() == 1 {
                parts.into_iter().next().expect("one part")
            } else {
                cat_ids(&parts)?
            })
        } else {
            None
        };

        if replies.len() == 1 {
            // One worker is the common case in a test and the degenerate case in
            // production; joining a single part would be a copy for nothing.
            let only = replies.into_iter().next().expect("one reply");
            return Ok(Joined {
                observation: only.observation,
                reward: only.reward,
                done: only.done,
            });
        }

        // A field is joined only if every worker supplied it. A reset supplies no
        // reward and no termination, and a half-filled batch would be worse than
        // none at all.
        fn gather<R: Runtime, E: FloatElem>(
            replies: &[Reply<R, E>],
            pick: impl Fn(&Reply<R, E>) -> Option<&Tensor<R, E>>,
        ) -> Result<Option<Tensor<R, E>>> {
            let parts: Option<Vec<Tensor<R, E>>> =
                replies.iter().map(|r| pick(r).cloned()).collect();
            match parts {
                None => Ok(None),
                Some(parts) => Ok(Some(movement::cat(&parts, 0)?)),
            }
        }

        Ok(Joined {
            observation: gather(&replies, |r| Some(&r.observation))?
                .expect("every reply carries an observation"),
            reward: gather(&replies, |r| r.reward.as_ref())?,
            done: gather(&replies, |r| r.done.as_ref())?,
        })
    }
}

/// The workers' replies, concatenated into one batch.
struct Joined<R: Runtime, E: FloatElem> {
    observation: Tensor<R, E>,
    reward: Option<Tensor<R, E>>,
    done: Option<Tensor<R, E>>,
}

impl<R: Runtime, E: FloatElem> VecEnv<R, E> for ParallelEnvs<R, E> {
    fn envs(&self) -> usize {
        self.total
    }

    fn obs_dim(&self) -> usize {
        self.obs_dim
    }

    fn action_dim(&self) -> usize {
        self.action_dim
    }

    fn reset(&mut self) -> Result<Tensor<R, E>> {
        let replies = self.exchange(|_| Ok(Command::Reset))?;
        Ok(self.join(replies)?.observation)
    }

    fn step(&mut self, actions: &IdTensor<R>) -> Result<EnvStep<R, E>> {
        if actions.len() != self.total {
            return Err(Error::shape(format!(
                "the worker pool drives {} environments but was given {} actions",
                self.total,
                actions.len()
            )));
        }
        let single = self.workers.len() == 1;
        let replies = self.exchange(|worker| {
            let slice = if single {
                actions.clone()
            } else {
                slice_ids(actions, worker.offset, worker.envs)?
            };
            Ok(Command::Step(slice))
        })?;
        let joined = self.join(replies)?;
        Ok(EnvStep {
            observation: joined.observation,
            reward: joined.reward.ok_or_else(|| {
                Error::config("a worker answered a step without a reward".to_string())
            })?,
            done: joined.done.ok_or_else(|| {
                Error::config("a worker answered a step without a termination flag".to_string())
            })?,
        })
    }

    fn expert_actions(&self) -> Option<IdTensor<R>> {
        self.expert.clone()
    }
}

impl<R: Runtime, E: FloatElem> Drop for ParallelEnvs<R, E> {
    fn drop(&mut self) {
        // Ask first, then wait. A worker blocked on `recv` wakes on the message; one
        // whose channel is already closed has stopped on its own.
        for worker in &self.workers {
            let _ = worker.commands.send(Command::Shutdown);
        }
        for worker in &mut self.workers {
            if let Some(thread) = worker.thread.take() {
                let _ = thread.join();
            }
        }
    }
}

impl<R: Runtime, E: FloatElem> core::fmt::Debug for ParallelEnvs<R, E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "ParallelEnvs(workers={}, envs={}, widths={:?})",
            self.workers.len(),
            self.total,
            self.widths(),
        )
    }
}

/// A [`Collector`] over a pool of environments running on their own threads.
///
/// The analogue of TorchRL's `MultiSyncDataCollector`, with the split described in
/// the [module documentation](self): the workers hold environments and the policy
/// stays on one thread, so there are no weights to synchronise and the batch stays
/// whole.
///
/// ```no_run
/// use mamba3::prelude::*;
/// use mamba3::rl::{MultiSyncCollector, PpoTask, RecallEnv};
///
/// type R = mamba3::backends::Auto;
/// # fn main() -> mamba3::error::Result<()> {
/// let device = Device::<R>::default();
///
/// // Four workers of sixteen environments each: a batch of sixty-four, stepped
/// // on four threads and evaluated by the policy in one pass.
/// let envs: Vec<RecallEnv<R, f32>> = (0..4)
///     .map(|w| RecallEnv::new(16, 4, 8, w, &device))
///     .collect::<mamba3::error::Result<_>>()?;
///
/// let policy = Mamba3PolicyConfig::new(6, 4, 128, 2).init::<R, f32>(&device)?;
/// let config = PpoConfig::default();
/// let mut collector = MultiSyncCollector::new(&policy, envs, 128, &device)?;
///
/// let report = collector.collect()?;
/// let batch = collector.ppo_batch(&report, &config)?;
/// # let _ = batch;
/// # Ok(())
/// # }
/// ```
pub struct MultiSyncCollector<'a, R: Runtime, E: FloatElem> {
    collector: Collector<'a, R, E>,
    envs: ParallelEnvs<R, E>,
}

impl<'a, R: Runtime, E: FloatElem> MultiSyncCollector<'a, R, E> {
    /// Build a collector over `envs`, one thread each, with windows of `steps`.
    pub fn new<V>(
        policy: &'a Mamba3Policy<R, E>,
        envs: Vec<V>,
        steps: usize,
        device: &Device<R>,
    ) -> Result<Self>
    where
        V: VecEnv<R, E> + Send + 'static,
    {
        let envs = ParallelEnvs::new(envs, device)?;
        let collector = Collector::new(policy, envs.envs(), steps, envs.obs_dim(), device)?;
        Ok(Self { collector, envs })
    }

    /// Sampling temperature; `0` acts greedily.
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.collector = self.collector.with_temperature(temperature);
        self
    }

    /// Seed for the action draws.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.collector = self.collector.with_seed(seed);
        self
    }

    /// Also record what each worker's expert would have done.
    pub fn recording_expert_labels(mut self) -> Self {
        self.collector = self.collector.recording_expert_labels();
        self
    }

    /// Number of worker threads.
    pub fn workers(&self) -> usize {
        self.envs.workers()
    }

    /// Total environments across every worker.
    pub fn envs(&self) -> usize {
        self.envs.envs()
    }

    /// How many environments each worker drives, in order.
    pub fn widths(&self) -> Vec<usize> {
        self.envs.widths()
    }

    /// The buffer the last window was collected into.
    ///
    /// Rows are worker-major: the first worker's environments come first, in the
    /// order the workers were given.
    pub fn buffer(&self) -> &TrajectoryBuffer<R, E> {
        self.collector.buffer()
    }

    /// The underlying single-threaded collector.
    pub fn collector(&self) -> &Collector<'a, R, E> {
        &self.collector
    }

    /// Forget everything: zero the recurrent state and restart every environment.
    pub fn reset(&mut self) {
        self.collector.reset();
    }

    /// Collect one window, every worker stepping in parallel.
    pub fn collect(&mut self) -> Result<CollectReport<R, E>> {
        self.collector.collect(&mut self.envs)
    }

    /// Collect one window on a mixture of each worker's expert and the policy.
    ///
    /// See [`Collector::collect_with_expert`] for why the result is an imitation
    /// window and not a PPO one.
    pub fn collect_with_expert(&mut self, beta: f32) -> Result<CollectReport<R, E>> {
        self.collector.collect_with_expert(&mut self.envs, beta)
    }

    /// The collected window as a PPO batch.
    pub fn ppo_batch(
        &self,
        report: &CollectReport<R, E>,
        config: &PpoConfig,
    ) -> Result<PpoBatch<R, E>> {
        self.collector.ppo_batch(report, config)
    }

    /// The collected window as an imitation batch.
    pub fn imitation_batch(&self) -> Result<ImitationBatch<R, E>> {
        self.collector.imitation_batch()
    }

    /// Mean reward per completed episode in the last window, and how many
    /// completed. See [`Collector::episode_return`].
    pub fn episode_return(&self) -> Result<(Tensor<R, E>, Tensor<R, E>)> {
        self.collector.episode_return()
    }
}

impl<R: Runtime, E: FloatElem> core::fmt::Debug for MultiSyncCollector<'_, R, E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "MultiSyncCollector({:?})", self.envs)
    }
}
