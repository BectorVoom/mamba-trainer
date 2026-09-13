//! What fusing the rollout step actually buys, as a number.
//!
//! `tests/rl_fused.rs` shows the fused loop collects the same window. This file
//! shows it collects it for less: the eight launches that follow each policy step
//! become one, so a window of `T` steps issues `7 * T` fewer dispatches. That is
//! the entire claim of the feature, and it is exact rather than approximate —
//! nothing here is a measurement of time, which would say more about this machine
//! than about the code.
//!
//! Alone in its binary, like its two siblings: the launch and read counters are
//! process-wide and any test running beside this one would contribute to them.

#![cfg(feature = "backend")]

use mamba3::backend::{
    Device, launch_count, read_count, reserved_bytes, reset_launch_count, reset_read_count,
};
use mamba3::cubecl::prelude::*;
use mamba3::autograd::Var;
use mamba3::rl::{
    Collector, GameLogic, GameSpec, GameWorld, Mamba3PolicyConfig, Outcome, PpoConfig,
    RolloutEngine,
};
use mamba3::tensor::Tensor;
use mamba3::tensor::ops::random::hash_u32;

type R = mamba3::backends::Auto;

const SYMBOLS: usize = 4;
const OBS_DIM: usize = SYMBOLS + 2;
const HORIZON: u32 = 4;

/// The same recall game `tests/rl_fused.rs` builds, restated because integration
/// test binaries do not share code and the alternative — a shared module — would
/// pull this file's counters into that one's process.
pub struct Recall;

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
        let base = env as usize * spec.obs_dim;
        for i in 0..spec.action_dim {
            obs[base + i] = select(i == cue as usize, F::new(1.0_f32), F::new(0.0_f32));
        }
        obs[base + spec.action_dim] = F::new(0.0_f32);
        obs[base + spec.action_dim + 1] = F::new(1.0_f32);
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
        let correct = select(action == cue, F::new(1.0_f32), F::new(0.0_f32));

        let fresh = hash_u32(env, seed_lo, seed_hi) % spec.action_dim as u32;
        let next_cue = select(terminal, fresh, cue);
        let next_clock = select(terminal, 0u32, clock + 1u32);
        ints[slot] = next_cue;
        ints[slot + 1] = next_clock;

        let showing = select(next_clock == 0u32, F::new(1.0_f32), F::new(0.0_f32));
        let base = env as usize * spec.obs_dim;
        for i in 0..spec.action_dim {
            let hit = select(i == next_cue as usize, F::new(1.0_f32), F::new(0.0_f32));
            obs[base + i] = showing * hit;
        }
        obs[base + spec.action_dim] = F::cast_from(next_clock) * F::new(0.25_f32);
        obs[base + spec.action_dim + 1] = showing;

        Outcome::<F> {
            reward: select(terminal, correct, F::new(0.0_f32)),
            done: select(terminal, F::new(1.0_f32), F::new(0.0_f32)),
        }
    }

    // Every action is always legal: this game does not restrict its actions.
    fn legal(
        _env: u32,
        action: u32,
        _ints: &Array<u32>,
        _floats: &Array<F>,
        #[comptime] spec: GameSpec,
    ) -> bool {
        action < spec.action_dim as u32
    }
}

/// Launches the tail of one unfused step costs: the action draw, the transition,
/// and the six writes that record the step into the trajectory buffer.
const UNFUSED_TAIL: usize = 8;

#[test]
fn fusing_the_rollout_collapses_its_tail_into_one_launch() {
    let device = Device::<R>::default();
    let (envs, steps) = (8usize, 8usize);
    let spec = GameSpec::new(OBS_DIM, SYMBOLS, 2);
    let policy = Mamba3PolicyConfig::new(OBS_DIM, SYMBOLS, 16, 2)
        .with_seed(7)
        .with_ssm(|s| {
            s.n_heads = 2;
            s.head_dim = 8;
            s.n_groups = 2;
            s.d_state = 4;
            s.chunk_size = 4;
            s.conv_kernel = Some(3);
        })
        .init::<R, f32>(&device)
        .unwrap();
    let config = PpoConfig::default();

    let mut plain_world: GameWorld<R, f32, Recall> =
        GameWorld::new(envs, spec, 5, &device).unwrap();
    let mut fused_world: GameWorld<R, f32, Recall> =
        GameWorld::new(envs, spec, 5, &device).unwrap();
    let mut plain = Collector::new(&policy, envs, steps, OBS_DIM, &device)
        .unwrap()
        .with_seed(9);
    let mut fused = Collector::new(&policy, envs, steps, OBS_DIM, &device)
        .unwrap()
        .with_seed(9);

    // Warm up: the first windows compile kernels and grow the allocator's pools to
    // the size one window needs. What matters is that they then stop growing.
    for _ in 0..4 {
        let report = plain.collect(&mut plain_world).unwrap();
        let _ = plain.ppo_batch(&report, &config).unwrap();
        let report = fused.collect_fused(&mut fused_world).unwrap();
        let _ = fused.ppo_batch(&report, &config).unwrap();
    }

    const WINDOWS: usize = 8;
    let footprint = reserved_bytes(&device);
    let buffer_bytes = fused.buffer().bytes();
    let world_elements = fused_world.num_elements();

    reset_read_count();
    reset_launch_count();
    for _ in 0..WINDOWS {
        let report = plain.collect(&mut plain_world).unwrap();
        let _ = plain.ppo_batch(&report, &config).unwrap();
    }
    let unfused_per_window = launch_count() / WINDOWS;

    reset_launch_count();
    for _ in 0..WINDOWS {
        let report = fused.collect_fused(&mut fused_world).unwrap();
        let _ = fused.ppo_batch(&report, &config).unwrap();
    }
    let fused_per_window = launch_count() / WINDOWS;

    // What one policy step costs on its own, so the saving can be read as a share
    // of a step rather than as a bare number. This is the part the fusion does not
    // touch, and it is the part that grows with the model — the tail it does touch
    // is the same eight launches whatever the policy is, which is why the fusion
    // matters most for the small policies reinforcement learning tends to use.
    let mut bare = RolloutEngine::new(&policy, envs, &device);
    let obs = Var::constant(Tensor::<R, f32>::zeros(vec![envs, 1, OBS_DIM], &device));
    for _ in 0..4 {
        bare.step(&obs, None).unwrap();
    }
    reset_launch_count();
    for _ in 0..8 {
        bare.step(&obs, None).unwrap();
    }
    let policy_step = launch_count() / 8;

    println!(
        "launches per {steps}-step window over {envs} environments: \
         {unfused_per_window} unfused, {fused_per_window} fused \
         ({} fewer, {:.0}% of the original).\n\
         one policy step is {policy_step} launches, so the rollout tail went from \
         {UNFUSED_TAIL} launches a step ({:.0}% of one) to 1 ({:.0}%).",
        unfused_per_window - fused_per_window,
        100.0 * fused_per_window as f64 / unfused_per_window as f64,
        100.0 * UNFUSED_TAIL as f64 / (policy_step + UNFUSED_TAIL) as f64,
        100.0 / (policy_step + 1) as f64,
    );

    assert_eq!(
        unfused_per_window - fused_per_window,
        (UNFUSED_TAIL - 1) * steps,
        "a fused step should replace exactly {UNFUSED_TAIL} launches with one, \
         {steps} times a window"
    );

    // The whole point of doing the work on the device is that the host is never
    // asked anything, and a fused step must not have quietly reintroduced a read.
    assert_eq!(
        read_count(),
        0,
        "a collection loop read back to the host; neither path is supposed to"
    );

    // The fused step writes the world's observation, the trajectory column and the
    // carried termination flag all in place, so a window allocates nothing that the
    // pooled allocator does not already hold.
    assert_eq!(
        buffer_bytes,
        fused.buffer().bytes(),
        "the trajectory buffer changed size mid-run"
    );
    assert_eq!(
        world_elements,
        fused_world.num_elements(),
        "the world's state changed size mid-run"
    );
    if let (Some(before), Some(after)) = (footprint, reserved_bytes(&device)) {
        assert_eq!(
            before,
            after,
            "{WINDOWS} windows of each kind reserved {} more bytes on the device; \
             at that rate a 10,000-window run would reserve {} more",
            after - before,
            (after - before) * 10_000 / (2 * WINDOWS) as u64,
        );
    }

    // Every window costs the same, so a long run costs its length times one. This is
    // the property that makes the saving worth stating per window at all.
    reset_launch_count();
    for _ in 0..WINDOWS {
        let report = fused.collect_fused(&mut fused_world).unwrap();
        let _ = fused.ppo_batch(&report, &config).unwrap();
    }
    assert_eq!(
        launch_count() / WINDOWS,
        fused_per_window,
        "the fused per-window launch count drifted from {fused_per_window}"
    );
}
