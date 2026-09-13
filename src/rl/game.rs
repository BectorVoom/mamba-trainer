//! User game logic as device code, and the world that hosts it.
//!
//! [`super::env::VecEnv`] fixes the *interface* an environment speaks — device
//! tensors in, device tensors out — but it says nothing about where the transition
//! runs. An implementation is free to be a host simulator that uploads a batch of
//! observations per step, and many are. That costs an upload and a download per
//! step, and worse, it puts a host round trip between the policy's launches and the
//! next step's, so the device drains its queue once per step however fast the
//! kernels are.
//!
//! [`GameLogic`] closes that gap by asking for the transition itself, as a
//! `#[cube]` function. The crate then owns the kernel it goes in, which buys two
//! things a `VecEnv` cannot:
//!
//! * the transition is a device function, so [`GameWorld`] is a `VecEnv` that never
//!   touches the host — the same property [`super::env::RecallEnv`] has, now
//!   available to a user's own game;
//! * because the crate compiles the kernel, it can compile the transition into the
//!   *same* kernel as the action draw and the trajectory write. That is
//!   [`super::fused`], and it is the reason this trait exists rather than a
//!   `VecEnv` impl being good enough.
//!
//! # Writing one
//!
//! A game is two device functions over three buffers it owns. The buffers are flat
//! and the game indexes its own rows:
//!
//! | buffer | shape | what it is |
//! |---|---|---|
//! | `ints` | `[envs, int_words]` | integer state — counters, positions on a grid, an RNG stream |
//! | `floats` | `[envs, float_words]` | continuous state — velocities, angles |
//! | `obs` | `[envs, obs_dim]` | what the policy sees next; the game writes it |
//!
//! ```
//! use mamba3::cubecl::prelude::*;
//! use mamba3::rl::{GameLogic, GameSpec, Outcome};
//! use mamba3::tensor::ops::random::hash_u32;
//!
//! /// Say the number you were shown. One step, one reward, no memory needed.
//! pub struct Echo;
//!
//! #[cube]
//! impl<F: Float + CubeElement> GameLogic<F> for Echo {
//!     fn reset(
//!         env: u32,
//!         ints: &mut Array<u32>,
//!         _floats: &mut Array<F>,
//!         obs: &mut Array<F>,
//!         seed_lo: u32,
//!         seed_hi: u32,
//!         #[comptime] spec: GameSpec,
//!     ) {
//!         let cue = hash_u32(env, seed_lo, seed_hi) % spec.action_dim as u32;
//!         ints[env as usize * spec.int_words] = cue;
//!         let base = env as usize * spec.obs_dim;
//!         for i in 0..spec.obs_dim {
//!             obs[base + i] = select(i == cue as usize, F::new(1.0), F::new(0.0));
//!         }
//!     }
//!
//!     fn transition(
//!         env: u32,
//!         action: u32,
//!         ints: &mut Array<u32>,
//!         floats: &mut Array<F>,
//!         obs: &mut Array<F>,
//!         seed_lo: u32,
//!         seed_hi: u32,
//!         #[comptime] spec: GameSpec,
//!     ) -> Outcome<F> {
//!         let cue = ints[env as usize * spec.int_words];
//!         let reward = select(action == cue, F::new(1.0), F::new(0.0));
//!         // Every episode is one step long, so the transition is a reset.
//!         Echo::reset(env, ints, floats, obs, seed_lo, seed_hi, spec);
//!         Outcome::<F> { reward, done: F::new(1.0) }
//!     }
//!
//!     // Every action is always legal; see `GameSpec::with_action_mask`.
//!     fn legal(
//!         _env: u32,
//!         action: u32,
//!         _ints: &Array<u32>,
//!         _floats: &Array<F>,
//!         #[comptime] spec: GameSpec,
//!     ) -> bool {
//!         action < spec.action_dim as u32
//!     }
//! }
//! ```
//!
//! # Determinism
//!
//! Both functions are handed a per-step seed rather than carrying a generator,
//! for the reason [`crate::tensor::ops::random::hash_u32`] explains: a unit that
//! derives its draw from its own position and a seed needs no state, so nothing
//! serialises the environments against each other. Use `env` as the index so two
//! environments do not draw the same value on the same step.

use core::marker::PhantomData;

use cubecl::prelude::*;

use crate::backend::{Device, FloatElem, launch_1d};
use crate::error::{Error, Result};
use crate::tensor::Tensor;
use crate::tensor::ops::index::IdTensor;

use super::env::{EnvStep, VecEnv};

/// The fixed dimensions of a game.
///
/// Comptime, so a game's own loops over `obs_dim` or `action_dim` have their bounds
/// available when the kernel is generated rather than as runtime scalars. Changing
/// one compiles a new kernel, which is what you want: these are the dimensions the
/// generated code is specialised on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GameSpec {
    /// Width of one observation.
    pub obs_dim: usize,
    /// Number of discrete actions.
    pub action_dim: usize,
    /// `u32`s of state per environment.
    pub int_words: usize,
    /// Floats of state per environment.
    pub float_words: usize,
    /// Whether the game restricts its actions through [`GameLogic::legal`].
    /// Comptime like the rest: an unmasked game compiles kernels that never
    /// call it, and collects exactly what it did before masks existed.
    pub masked: bool,
}

impl GameSpec {
    /// A game with `int_words` integers and no floats of state per environment.
    pub fn new(obs_dim: usize, action_dim: usize, int_words: usize) -> Self {
        Self {
            obs_dim,
            action_dim,
            int_words,
            float_words: 0,
            masked: false,
        }
    }

    /// Restrict each step's legal actions to those [`GameLogic::legal`] allows.
    /// Both rollout paths honour it identically: [`GameWorld`]'s
    /// [`VecEnv::action_mask`] and the fused step compute the same mask from the
    /// same state before the draw.
    pub fn with_action_mask(mut self) -> Self {
        self.masked = true;
        self
    }

    /// Give each environment `float_words` floats of state as well.
    pub fn with_float_words(mut self, float_words: usize) -> Self {
        self.float_words = float_words;
        self
    }

    /// Check internal consistency.
    pub fn validate(&self) -> Result<()> {
        if self.obs_dim == 0 {
            return Err(Error::config(
                "a game must show the policy something: obs_dim is 0".to_string(),
            ));
        }
        if self.action_dim < 2 {
            return Err(Error::config(format!(
                "a game needs at least two actions to have a policy at all, got {}",
                self.action_dim
            )));
        }
        Ok(())
    }
}

pub use outcome::Outcome;

/// [`Outcome`] and the expand type `#[derive(CubeType)]` generates beside it.
///
/// A module of its own so that one allow covers the generated half, which has
/// nowhere to hang a doc comment; [`Outcome`] itself is re-exported and documented.
#[allow(missing_docs)]
pub mod outcome {
    use cubecl::prelude::*;

    /// What one transition earned and whether it ended the episode.
    ///
    /// `done` is a float rather than a bool because that is the form every consumer
    /// wants it in — the advantage estimator multiplies by `1 - done`, the
    /// recurrence masks by it — and converting once here is cheaper than converting
    /// at each.
    #[derive(CubeType)]
    pub struct Outcome<F: Float> {
        /// Reward for the action just taken.
        pub reward: F,
        /// `1` if the action ended the episode, `0` otherwise.
        pub done: F,
    }
}

/// A game's transition function, as device code.
///
/// Implemented on a marker type — the implementation is all associated functions,
/// so nothing is instantiated and the type exists only to name the game at the type
/// level. That is what lets a kernel be generic over it and specialise completely.
///
/// # Auto-reset
///
/// [`GameLogic::transition`] must leave `obs` holding the observation the policy
/// will act on *next*, which where it reports `done` means the first observation of
/// a fresh episode. Rollouts are rectangular — every environment produces exactly
/// one transition per step — and an environment that stopped producing observations
/// at the end of an episode would break that. See [`super::env::EnvStep`], which
/// states the same convention for a `VecEnv`.
#[allow(missing_docs)] // `#[cube]` adds an expand function per method.
#[cube]
pub trait GameLogic<F: Float + CubeElement>: Send + Sync + 'static {
    /// Start environment `env`: initialise its state and write its first
    /// observation.
    fn reset(
        env: u32,
        ints: &mut Array<u32>,
        floats: &mut Array<F>,
        obs: &mut Array<F>,
        seed_lo: u32,
        seed_hi: u32,
        #[comptime] spec: GameSpec,
    );

    /// Apply `action` in environment `env`.
    ///
    /// Advances the state, writes the next observation into `obs`, and returns what
    /// the action earned. Only row `env` of any buffer may be touched: every
    /// environment is a separate unit and there is no synchronisation between them.
    fn transition(
        env: u32,
        action: u32,
        ints: &mut Array<u32>,
        floats: &mut Array<F>,
        obs: &mut Array<F>,
        seed_lo: u32,
        seed_hi: u32,
        #[comptime] spec: GameSpec,
    ) -> Outcome<F>;

    /// Whether `action` is legal for environment `env` in its current state —
    /// the state the next action is drawn from, before
    /// [`GameLogic::transition`] runs.
    ///
    /// Only consulted when the spec asks for it ([`GameSpec::with_action_mask`]);
    /// a game that never restricts its actions returns `true`. Every environment
    /// must leave at least one action legal; a row with none is refused when the
    /// window becomes a batch. (Required rather than defaulted: `#[cube]` traits
    /// do not carry default bodies into their generated expansions.)
    fn legal(
        env: u32,
        action: u32,
        ints: &Array<u32>,
        floats: &Array<F>,
        #[comptime] spec: GameSpec,
    ) -> bool;
}

// ---------------------------------------------------------------------------
// The unfused path: a GameLogic is a VecEnv
// ---------------------------------------------------------------------------

#[cube(launch_unchecked)]
fn game_reset_kernel<F: Float + CubeElement, G: GameLogic<F>>(
    ints: &mut Array<u32>,
    floats: &mut Array<F>,
    obs: &mut Array<F>,
    envs: usize,
    seed_lo: u32,
    seed_hi: u32,
    #[comptime] spec: GameSpec,
) {
    if ABSOLUTE_POS < envs {
        G::reset(
            ABSOLUTE_POS as u32,
            ints,
            floats,
            obs,
            seed_lo,
            seed_hi,
            spec,
        );
    }
}

#[cube(launch_unchecked)]
fn game_step_kernel<F: Float + CubeElement, G: GameLogic<F>>(
    actions: &Array<u32>,
    ints: &mut Array<u32>,
    floats: &mut Array<F>,
    obs: &mut Array<F>,
    reward: &mut Array<F>,
    done: &mut Array<F>,
    envs: usize,
    seed_lo: u32,
    seed_hi: u32,
    #[comptime] spec: GameSpec,
) {
    if ABSOLUTE_POS < envs {
        let out = G::transition(
            ABSOLUTE_POS as u32,
            actions[ABSOLUTE_POS],
            ints,
            floats,
            obs,
            seed_lo,
            seed_hi,
            spec,
        );
        reward[ABSOLUTE_POS] = out.reward;
        done[ABSOLUTE_POS] = out.done;
    }
}

#[cube(launch_unchecked)]
fn game_mask_kernel<F: Float + CubeElement, G: GameLogic<F>>(
    ints: &Array<u32>,
    floats: &Array<F>,
    mask: &mut Array<F>,
    envs: usize,
    #[comptime] spec: GameSpec,
) {
    if ABSOLUTE_POS < envs {
        let base = ABSOLUTE_POS * spec.action_dim;
        for action in 0..spec.action_dim {
            let legal = G::legal(ABSOLUTE_POS as u32, action as u32, ints, floats, spec);
            mask[base + action] = select(legal, F::new(1.0_f32), F::new(0.0_f32));
        }
    }
}

/// `envs` copies of a [`GameLogic`], and the device state they run on.
///
/// This is the ordinary, unfused way to run a game: a [`VecEnv`] like any other,
/// usable with [`super::Collector`] and with everything built on it. It costs two
/// kernels per step — the transition and nothing else — and it is the reference the
/// fused path in [`super::fused`] is tested against.
///
/// # Footprint
///
/// `envs * (int_words + float_words + obs_dim)` elements of state, fixed at
/// construction. A step's outputs — the next observation and the `[envs]` reward
/// and termination pair — are fresh tensors, which the pooled allocator hands back
/// unchanged every step, so the steady-state footprint is flat. The fused path does
/// not allocate them at all.
pub struct GameWorld<R: Runtime, E: FloatElem, G: GameLogic<E>> {
    ints: IdTensor<R>,
    floats: Tensor<R, E>,
    obs: Tensor<R, E>,
    spec: GameSpec,
    envs: usize,
    seed: u64,
    ticks: u64,
    device: Device<R>,
    _game: PhantomData<fn() -> G>,
}

impl<R: Runtime, E: FloatElem, G: GameLogic<E>> GameWorld<R, E, G> {
    /// Allocate `envs` environments of the game `G`.
    pub fn new(envs: usize, spec: GameSpec, seed: u64, device: &Device<R>) -> Result<Self> {
        spec.validate()?;
        if envs == 0 {
            return Err(Error::config(
                "a world needs at least one environment".to_string(),
            ));
        }
        // A game with no state of one kind still gets a one-element buffer for it.
        // Zero-length bindings are not portable across the runtimes this crate
        // targets, and one element is cheaper than a second kernel per arity.
        let width = |words: usize| envs * words.max(1);
        Ok(Self {
            ints: IdTensor::empty(vec![width(spec.int_words)], device),
            floats: Tensor::zeros(vec![width(spec.float_words)], device),
            obs: Tensor::zeros(vec![envs, spec.obs_dim], device),
            spec,
            envs,
            seed,
            ticks: 0,
            device: device.clone(),
            _game: PhantomData,
        })
    }

    /// The game's dimensions.
    pub fn spec(&self) -> GameSpec {
        self.spec
    }

    /// How many environments run in parallel.
    ///
    /// The same number [`VecEnv::envs`] reports; inherent so that callers who hold a
    /// world concretely need not import the trait for it.
    pub fn envs(&self) -> usize {
        self.envs
    }

    /// `[envs, obs_dim]` observation the policy is to act on next.
    ///
    /// Valid until the next transition. The fused path overwrites this buffer in
    /// place — see [`super::fused`] — and [`VecEnv::step`] replaces it, so ask
    /// again each step rather than caching the handle.
    pub fn observation(&self) -> &Tensor<R, E> {
        &self.obs
    }

    /// `[envs, int_words]` integer state, as the game left it.
    pub fn ints(&self) -> &IdTensor<R> {
        &self.ints
    }

    /// `[envs, float_words]` continuous state, as the game left it.
    pub fn floats(&self) -> &Tensor<R, E> {
        &self.floats
    }

    /// Elements of device state, fixed for the life of the world.
    pub fn num_elements(&self) -> usize {
        self.ints.len() + self.floats.len() + self.obs.len()
    }

    /// The seed for the next transition, and the advance of the schedule.
    ///
    /// Every step draws one, so the schedule is a pure function of the base seed
    /// and how many steps have been taken — which is what lets the fused loop in
    /// [`super::fused`] reproduce an unfused run exactly.
    pub(crate) fn next_seed(&mut self) -> u64 {
        self.ticks = self.ticks.wrapping_add(1);
        self.seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(self.ticks.wrapping_mul(1442695040888963407))
    }

    /// Queue the transition kernel: the next observation into `obs`, and what the
    /// action earned into `reward` and `done`.
    fn launch_step(
        &mut self,
        actions: &IdTensor<R>,
        obs: &Tensor<R, E>,
        reward: &Tensor<R, E>,
        done: &Tensor<R, E>,
    ) {
        let seed = self.next_seed();
        let (count, dim) = launch_1d(self.device.client(), self.envs, self.spec.obs_dim);
        unsafe {
            game_step_kernel::launch_unchecked::<E, G, R>(
                self.device.client(),
                count,
                dim,
                actions.arg(),
                self.ints.arg(),
                self.floats.arg(),
                obs.arg(),
                reward.arg(),
                done.arg(),
                self.envs,
                seed as u32,
                (seed >> 32) as u32,
                self.spec,
            );
        }
    }
}

impl<R: Runtime, E: FloatElem, G: GameLogic<E>> VecEnv<R, E> for GameWorld<R, E, G> {
    fn envs(&self) -> usize {
        self.envs
    }

    fn obs_dim(&self) -> usize {
        self.spec.obs_dim
    }

    fn action_dim(&self) -> usize {
        self.spec.action_dim
    }

    fn reset(&mut self) -> Result<Tensor<R, E>> {
        let seed = self.next_seed();
        let (count, dim) = launch_1d(self.device.client(), self.envs, self.spec.obs_dim);
        unsafe {
            game_reset_kernel::launch_unchecked::<E, G, R>(
                self.device.client(),
                count,
                dim,
                self.ints.arg(),
                self.floats.arg(),
                self.obs.arg(),
                self.envs,
                seed as u32,
                (seed >> 32) as u32,
                self.spec,
            );
        }
        Ok(self.obs.clone())
    }

    fn step(&mut self, actions: &IdTensor<R>) -> Result<EnvStep<R, E>> {
        if actions.len() != self.envs {
            return Err(Error::shape(format!(
                "this world drives {} environments but was given {} actions",
                self.envs,
                actions.len()
            )));
        }
        let reward = Tensor::<R, E>::empty(vec![self.envs], &self.device);
        let done = Tensor::<R, E>::empty(vec![self.envs], &self.device);
        // The *next* observation goes to a fresh buffer rather than over the one
        // just acted on, because a `VecEnv` is stepped before its caller records the
        // observation the action was chosen from — [`super::Collector`] needs the
        // reward this call returns before it can write the column — and overwriting
        // in place would hand it the wrong one. The transition writes every element
        // of the row from state, never reading what was there, so there is nothing
        // to carry over. Nothing is allocated in steady state: the pooled allocator
        // hands the previous step's buffer straight back once the caller drops it.
        let next = Tensor::<R, E>::empty(vec![self.envs, self.spec.obs_dim], &self.device);
        self.launch_step(actions, &next, &reward, &done);
        self.obs = next.clone();
        Ok(EnvStep {
            observation: next,
            reward,
            done,
        })
    }

    fn action_mask(&self) -> Result<Option<Tensor<R, E>>> {
        if !self.spec.masked {
            return Ok(None);
        }
        let mask = Tensor::<R, E>::empty(vec![self.envs, self.spec.action_dim], &self.device);
        let (count, dim) = launch_1d(self.device.client(), self.envs, self.spec.action_dim);
        unsafe {
            game_mask_kernel::launch_unchecked::<E, G, R>(
                self.device.client(),
                count,
                dim,
                self.ints.arg(),
                self.floats.arg(),
                mask.arg(),
                self.envs,
                self.spec,
            );
        }
        Ok(Some(mask))
    }
}

impl<R: Runtime, E: FloatElem, G: GameLogic<E>> core::fmt::Debug for GameWorld<R, E, G> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "GameWorld<{}>(envs={}, obs_dim={}, actions={})",
            core::any::type_name::<G>(),
            self.envs,
            self.spec.obs_dim,
            self.spec.action_dim,
        )
    }
}
