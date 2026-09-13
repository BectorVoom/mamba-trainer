//! A fused rollout is the rollout it replaced.
//!
//! The fused step in `mamba3::rl::fused` rewrites eight kernels as one, and the
//! only thing that makes that a fusion rather than a second implementation is that
//! the window it produces is the window the eight would have produced. So the
//! central test here runs both over the same policy, the same game and the same
//! seeds and compares every byte of the trajectory buffer: the observations, the
//! actions, their log-probabilities, the critic's estimates, the rewards and the
//! termination flags.
//!
//! The game is written the way a user would write one — a `#[cube]` implementation
//! of [`GameLogic`] in a crate that depends on this one — so the same file is the
//! worked example of the feature and the proof that it is correct.

#![cfg(feature = "backend")]

use mamba3::cubecl::prelude::*;
use mamba3::prelude::*;
use mamba3::rl::{
    Collector, GameLogic, GameSpec, GameWorld, Mamba3Policy, Mamba3PolicyConfig, Outcome, VecEnv,
};
use mamba3::tensor::ops::random::hash_u32;

type R = mamba3::backends::Auto;

/// Symbols to choose between, which is also the action space.
const SYMBOLS: usize = 4;
/// Steps in an episode. The cue is on screen at `0` and the reward is at `3`.
const HORIZON: u32 = 4;

// ---------------------------------------------------------------------------
// A game, written as a user would write one
// ---------------------------------------------------------------------------

/// See a symbol once, name it three steps later.
///
/// The same task as [`mamba3::rl::RecallEnv`], written against [`GameLogic`]
/// instead of against [`VecEnv`]: a policy with no memory cannot beat `1/SYMBOLS`,
/// because the only thing that distinguishes the step it is rewarded on from every
/// other step is what it saw at the start of the episode.
///
/// Two `u32`s of state per environment — the cue and the clock — and no floats.
pub struct Recall;

/// Write what environment `env` can see: the cue, but only while the clock is zero.
#[cube]
fn show<F: Float + CubeElement>(
    obs: &mut Array<F>,
    env: u32,
    cue: u32,
    clock: u32,
    #[comptime] spec: GameSpec,
) {
    let base = env as usize * spec.obs_dim;
    let showing = select(clock == 0u32, F::new(1.0_f32), F::new(0.0_f32));
    for i in 0..spec.action_dim {
        let hit = select(i == cue as usize, F::new(1.0_f32), F::new(0.0_f32));
        obs[base + i] = showing * hit;
    }
    // Two channels the agent may always read: how far into the episode it is, and
    // whether the cue is on screen. Neither says what the cue was.
    obs[base + spec.action_dim] = F::cast_from(clock) * F::new(0.25_f32);
    obs[base + spec.action_dim + 1] = showing;
}

#[cube]
impl<F: Float + CubeElement> GameLogic<F> for Recall {
    fn reset(
        env: u32,
        ints: &mut Array<u32>,
        _floats: &mut Array<F>,
        obs: &mut Array<F>,
        seed_lo: u32,
        seed_hi: u32,
        #[comptime] spec: GameSpec,
    ) {
        let cue = hash_u32(env, seed_lo, seed_hi) % spec.action_dim as u32;
        let slot = env as usize * spec.int_words;
        ints[slot] = cue;
        ints[slot + 1] = 0u32;
        show::<F>(obs, env, cue, 0u32, spec);
    }

    fn transition(
        env: u32,
        action: u32,
        ints: &mut Array<u32>,
        _floats: &mut Array<F>,
        obs: &mut Array<F>,
        seed_lo: u32,
        seed_hi: u32,
        #[comptime] spec: GameSpec,
    ) -> Outcome<F> {
        let slot = env as usize * spec.int_words;
        let cue = ints[slot];
        let clock = ints[slot + 1];
        let terminal = clock + 1u32 == HORIZON;

        // Sparse, and at the end: naming the cue early earns nothing, so the only
        // way to score is to still know it when the episode closes.
        let correct = select(action == cue, F::new(1.0_f32), F::new(0.0_f32));
        let reward = select(terminal, correct, F::new(0.0_f32));

        // Auto-reset, which is what keeps the rollout rectangular.
        let fresh = hash_u32(env, seed_lo, seed_hi) % spec.action_dim as u32;
        let next_cue = select(terminal, fresh, cue);
        let next_clock = select(terminal, 0u32, clock + 1u32);
        ints[slot] = next_cue;
        ints[slot + 1] = next_clock;
        show::<F>(obs, env, next_cue, next_clock, spec);

        Outcome::<F> {
            reward,
            done: select(terminal, F::new(1.0_f32), F::new(0.0_f32)),
        }
    }
}

fn spec() -> GameSpec {
    GameSpec::new(SYMBOLS + 2, SYMBOLS, 2)
}

fn world(envs: usize, seed: u64, device: &Device<R>) -> GameWorld<R, f32, Recall> {
    GameWorld::new(envs, spec(), seed, device).expect("the spec is valid")
}

fn policy(device: &Device<R>) -> Mamba3Policy<R, f32> {
    Mamba3PolicyConfig::new(SYMBOLS + 2, SYMBOLS, 16, 2)
        .with_seed(7)
        .with_ssm(|s| {
            s.n_heads = 2;
            s.head_dim = 8;
            s.n_groups = 2;
            s.d_state = 4;
            s.chunk_size = 4;
            s.conv_kernel = Some(3);
        })
        .init::<R, f32>(device)
        .expect("the configuration is valid")
}

fn assert_identical(actual: &[f32], expected: &[f32], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            a, e,
            "{what}: element {i} is {a} fused and {e} unfused. The fused step is \
             supposed to be the same arithmetic in the same order, so any difference \
             here is a difference in what the policy is being trained on."
        );
    }
}

// ---------------------------------------------------------------------------

#[test]
fn a_fused_window_is_the_window_the_unfused_loop_would_have_collected() {
    let device = Device::<R>::default();
    let (envs, steps) = (8usize, 12usize);
    let policy = policy(&device);

    // Two of everything, seeded identically: the same policy weights, the same
    // cues, the same action draws. The only difference between the two runs is how
    // many kernels each step is spread over.
    let mut plain_world = world(envs, 5, &device);
    let mut fused_world = world(envs, 5, &device);
    let collector = || {
        Collector::new(&policy, envs, steps, SYMBOLS + 2, &device)
            .expect("the collector is well formed")
            .with_seed(9)
    };
    let mut plain = collector();
    let mut fused = collector();

    // Three windows, not one: the interesting state — the recurrent buffer, the
    // carried termination flag, the environment's clocks — is what a *second*
    // window starts from, and a fusion that dropped any of it would still match on
    // the first.
    for window in 0..3 {
        let plain_report = plain.collect(&mut plain_world).expect("unfused window");
        let fused_report = fused
            .collect_fused(&mut fused_world)
            .expect("fused window");

        assert_eq!(
            plain.buffer().actions().to_vec(),
            fused.buffer().actions().to_vec(),
            "window {window}: the two rollouts took different actions"
        );
        for (what, a, b) in [
            (
                "observations",
                plain.buffer().observations(),
                fused.buffer().observations(),
            ),
            (
                "log_probs",
                plain.buffer().log_probs(),
                fused.buffer().log_probs(),
            ),
            ("values", plain.buffer().values(), fused.buffer().values()),
            ("rewards", plain.buffer().rewards(), fused.buffer().rewards()),
            ("dones", plain.buffer().dones(), fused.buffer().dones()),
        ] {
            assert_identical(&b.to_f32(), &a.to_f32(), &format!("window {window} {what}"));
        }
        assert_identical(
            &fused_report.bootstrap.to_f32(),
            &plain_report.bootstrap.to_f32(),
            &format!("window {window} bootstrap"),
        );
        assert_eq!(fused_report.steps, plain_report.steps);

        // The reset mask is derived from the recorded terminations *and* the flag
        // carried in from the previous window, so it is the one place a fused loop
        // could agree on every column and still disagree about where episodes begin.
        assert_identical(
            &fused.buffer().reset_mask().expect("mask").to_f32(),
            &plain.buffer().reset_mask().expect("mask").to_f32(),
            &format!("window {window} reset mask"),
        );
    }
}

#[test]
fn a_fused_window_learns_the_same_batch() {
    use mamba3::rl::PpoConfig;

    let device = Device::<R>::default();
    let (envs, steps) = (8usize, 12usize);
    let policy = policy(&device);
    let config = PpoConfig::default();

    let mut plain_world = world(envs, 11, &device);
    let mut fused_world = world(envs, 11, &device);
    let mut plain = Collector::new(&policy, envs, steps, SYMBOLS + 2, &device)
        .unwrap()
        .with_seed(3);
    let mut fused = Collector::new(&policy, envs, steps, SYMBOLS + 2, &device)
        .unwrap()
        .with_seed(3);

    let plain_report = plain.collect(&mut plain_world).unwrap();
    let fused_report = fused.collect_fused(&mut fused_world).unwrap();
    let plain_batch = plain.ppo_batch(&plain_report, &config).unwrap();
    let fused_batch = fused.ppo_batch(&fused_report, &config).unwrap();

    // The batch is what the optimizer actually sees, and it is two estimators
    // downstream of the buffer: the advantages come from a backwards recurrence over
    // the rewards and terminations, then get centred and rescaled across the whole
    // window. Both of those spread a single wrong element over every other one.
    assert_identical(
        &fused_batch.advantages.to_f32(),
        &plain_batch.advantages.to_f32(),
        "advantages",
    );
    assert_identical(
        &fused_batch.returns.to_f32(),
        &plain_batch.returns.to_f32(),
        "returns",
    );
    assert_identical(
        &fused.episode_return().unwrap().to_f32(),
        &plain.episode_return().unwrap().to_f32(),
        "episode return",
    );
}

#[test]
fn a_game_that_disagrees_with_the_policy_about_the_action_space_is_refused() {
    let device = Device::<R>::default();
    let envs = 4usize;
    // The policy chooses between `SYMBOLS` actions; tell the world there are more.
    let wrong = GameSpec::new(SYMBOLS + 2, SYMBOLS + 1, 2);
    let mut world: GameWorld<R, f32, Recall> =
        GameWorld::new(envs, wrong, 1, &device).expect("the spec is self-consistent");
    let policy = policy(&device);
    let mut collector = Collector::new(&policy, envs, 4, SYMBOLS + 2, &device).unwrap();

    let err = collector
        .collect_fused(&mut world)
        .expect_err("a policy and a game that disagree cannot be rolled out together");
    let message = err.to_string();
    assert!(
        message.contains(&format!("{SYMBOLS} actions")),
        "the error should name both action spaces, got: {message}"
    );
}

#[test]
fn a_world_is_an_ordinary_environment_too() {
    let device = Device::<R>::default();
    let mut world = world(6, 2, &device);
    assert_eq!(world.envs(), 6);
    assert_eq!(VecEnv::<R, f32>::obs_dim(&world), SYMBOLS + 2);
    assert_eq!(VecEnv::<R, f32>::action_dim(&world), SYMBOLS);

    let obs = world.reset().expect("reset");
    assert_eq!(obs.shape().dims(), &[6, SYMBOLS + 2]);
    // Exactly one symbol channel is lit at the start of an episode, and the
    // cue-present flag with it.
    let data = obs.to_f32();
    for env in 0..6 {
        let row = &data[env * (SYMBOLS + 2)..(env + 1) * (SYMBOLS + 2)];
        let lit: f32 = row[..SYMBOLS].iter().sum();
        assert_eq!(lit, 1.0, "environment {env} was shown {lit} cues");
        assert_eq!(row[SYMBOLS], 0.0, "the clock should start at zero");
        assert_eq!(row[SYMBOLS + 1], 1.0, "the cue should be on screen");
    }
}
