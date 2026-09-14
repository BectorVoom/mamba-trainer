//! Device games compiled into the extension, reachable by name.
//!
//! A [`mamba3::rl::GameLogic`] is device code: its transition is a `#[cube]` function inlined
//! into the rollout kernel, so it cannot be written in Python, and a Python caller
//! cannot hand one to a learner. What it can do is ask for one that was compiled
//! in — `mamba3_rl.game("recall", num_envs=64)` — and a learner given the result
//! collects through [`mamba3::rl::Collector::collect_fused`]: one kernel per step
//! doing the action draw, the trajectory writes and the transition, instead of a
//! launch per operation and an environment step between them.
//!
//! # Adding a game
//!
//! 1. Implement [`mamba3::rl::GameLogic`] for a unit struct in Rust (`reset`, `transition`,
//!    `legal`) — see `mamba3::rl::games::Recall` — and a function returning its
//!    [`GameSpec`].
//! 2. Add a variant to [`World`] and an arm to each `match` on it below: the
//!    compiler lists every one that is missing.
//! 3. Add its name to [`GAMES`] and parse its parameters in [`build`].
//! 4. Rebuild the wheel, then run the fused-against-host parity and footprint tests
//!    in `tests/test_game.py` for it.

use mamba3::backend::Device;
use mamba3::error::Result;
use mamba3::rl::{
    CollectReport, Collector, EnvStep, GameSpec, GameWorld, Recall, VecEnv, recall_spec,
};
use mamba3::tensor::Tensor;
use numpy::{PyArray1, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::array;
use crate::err::IntoPyResult;
use crate::{E, R};

/// The compiled-in games, by the name `game()` takes.
const GAMES: [&str; 1] = ["recall"];

/// One world of each compiled-in game. Every learner-facing operation dispatches
/// through it, so adding a game is a variant and the arms the compiler asks for.
pub enum World {
    Recall(GameWorld<R, E, Recall>),
}

impl World {
    /// The world as an ordinary environment, for the host path and for saving.
    pub fn as_env(&mut self) -> &mut dyn VecEnv<R, E> {
        match self {
            World::Recall(world) => world,
        }
    }

    fn env(&self) -> &dyn VecEnv<R, E> {
        match self {
            World::Recall(world) => world,
        }
    }

    fn spec(&self) -> GameSpec {
        match self {
            World::Recall(world) => world.spec(),
        }
    }

    /// One window through the fused rollout.
    pub fn collect_fused(
        &mut self,
        collector: &mut Collector<'static, R, E>,
    ) -> Result<CollectReport<R, E>> {
        match self {
            World::Recall(world) => collector.collect_fused(world),
        }
    }
}

/// Build a world by name, checking every parameter the game has an opinion on.
fn build(
    name: &str,
    num_envs: usize,
    symbols: usize,
    horizon: usize,
    seed: u64,
    masked: bool,
    device: &Device<R>,
) -> PyResult<World> {
    if num_envs == 0 {
        return Err(PyValueError::new_err(
            "a game needs at least one environment",
        ));
    }
    match name {
        "recall" => {
            if symbols < 2 {
                return Err(PyValueError::new_err(format!(
                    "recall needs at least two symbols, got {symbols}"
                )));
            }
            let compiled = mamba3::rl::RECALL_HORIZON as usize;
            if horizon != compiled {
                return Err(PyValueError::new_err(format!(
                    "the compiled recall game's horizon is {compiled}, not {horizon}: a \
                     device game's constants are compiled into its kernel. Use \
                     RecallEnv(horizon={horizon}) for another horizon on the host path"
                )));
            }
            let spec = recall_spec(symbols);
            let spec = if masked {
                spec.with_action_mask()
            } else {
                spec
            };
            Ok(World::Recall(
                GameWorld::new(num_envs, spec, seed, device).py()?,
            ))
        }
        other => Err(PyValueError::new_err(format!(
            "no game named {other:?} is compiled into this extension; available: {}",
            GAMES.join(", ")
        ))),
    }
}

/// A device game compiled into this extension.
///
/// Built with [`game`]. Learners that are given one collect through the fused
/// rollout (`learner.collection_path == "fused"`). It is also an ordinary
/// environment — `reset`, `step` and `action_mask` work from Python for
/// inspection, at the price of a device read each.
#[pyclass(module = "mamba3_rl", name = "Game", unsendable)]
pub struct PyGame {
    pub(crate) world: World,
    name: &'static str,
    symbols: usize,
    horizon: usize,
    seed: u64,
    device: Device<R>,
}

impl PyGame {
    pub fn envs(&self) -> usize {
        self.world.env().envs()
    }

    pub fn observation_width(&self) -> usize {
        self.world.env().obs_dim()
    }

    pub fn actions(&self) -> usize {
        self.world.env().action_dim()
    }

    pub fn masked(&self) -> bool {
        self.world.spec().masked
    }
}

#[pymethods]
impl PyGame {
    /// The name `game()` built this from.
    #[getter]
    fn name(&self) -> &'static str {
        self.name
    }

    /// How many environments run in parallel.
    #[getter]
    fn num_envs(&self) -> usize {
        self.world.env().envs()
    }

    /// Width of one observation.
    #[getter]
    fn obs_dim(&self) -> usize {
        self.world.env().obs_dim()
    }

    /// Number of discrete actions.
    #[getter]
    fn action_dim(&self) -> usize {
        self.world.env().action_dim()
    }

    /// Whether the game restricts its legal actions.
    #[getter(masked)]
    fn is_masked(&self) -> bool {
        self.masked()
    }

    /// Start every environment and return the first observation.
    fn reset<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let (envs, obs_dim) = (self.num_envs(), self.obs_dim());
        let obs = self.world.as_env().reset().py()?;
        array::to_2d(py, &obs, envs, obs_dim)
    }

    /// Apply one action per environment: `(observation, reward, done)`.
    #[allow(clippy::type_complexity)]
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        actions: &Bound<'py, PyAny>,
    ) -> PyResult<(
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray1<f32>>,
        Bound<'py, PyArray1<f32>>,
    )> {
        let (envs, obs_dim, action_dim) = (self.num_envs(), self.obs_dim(), self.action_dim());
        let ids = array::ids_1d(actions, envs, action_dim, "actions", &self.device)?;
        let EnvStep {
            observation,
            reward,
            done,
        } = self.world.as_env().step(&ids).py()?;
        Ok((
            array::to_2d(py, &observation, envs, obs_dim)?,
            array::to_1d(py, &reward)?,
            array::to_1d(py, &done)?,
        ))
    }

    /// The legal actions on the current observation, or `None` for an unmasked game.
    fn action_mask<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray2<f32>>>> {
        let (envs, action_dim) = (self.num_envs(), self.action_dim());
        self.world
            .env()
            .action_mask()
            .py()?
            .map(|mask: Tensor<R, E>| array::to_2d(py, &mask, envs, action_dim))
            .transpose()
    }

    fn __repr__(&self) -> String {
        format!(
            "Game({:?}, num_envs={}, symbols={}, horizon={}, seed={}, masked={})",
            self.name,
            self.num_envs(),
            self.symbols,
            self.horizon,
            self.seed,
            self.masked()
        )
    }
}

/// A compiled-in device game, by name.
///
/// `"recall"` is the device twin of `RecallEnv`: `symbols` actions, the cue shown
/// at the first step, the reward for naming it at the last. Its horizon is compiled
/// into the kernel (8); `masked=True` makes the symbol after the cue illegal on
/// every step. Unknown names and parameters the game cannot honour raise
/// `ValueError`.
#[pyfunction]
#[pyo3(signature = (name, num_envs, *, symbols = 4, horizon = 8, seed = 0, masked = false))]
pub fn game(
    name: &str,
    num_envs: usize,
    symbols: usize,
    horizon: usize,
    seed: u64,
    masked: bool,
) -> PyResult<PyGame> {
    let device = Device::<R>::default();
    let world = build(name, num_envs, symbols, horizon, seed, masked, &device)?;
    let name = GAMES
        .iter()
        .copied()
        .find(|n| *n == name)
        .expect("build accepted a known name");
    Ok(PyGame {
        world,
        name,
        symbols,
        horizon,
        seed,
        device,
    })
}
