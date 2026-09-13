//! Built-in device games.
//!
//! [`Recall`] lives here for two reasons: [`super::GameLogic`] gets a genuine
//! worked example inside the crate itself (`tests/rl_fused.rs` keeps its own
//! copy too, written the way an external user would write one, since that is
//! the point of that test), and the Python bindings' `game()` factory has
//! something real compiled in to hand back — see A6 in the trainer's
//! `FIX_PLAN.md` for the rest of that story, including why this is a *named
//! registry of games compiled into the extension* and not a way to turn an
//! arbitrary Python callback into device code.

use cubecl::prelude::*;

use crate::tensor::ops::random::hash_u32;

use super::game::{GameLogic, GameSpec, Outcome};

/// Episode length for [`Recall`].
///
/// Fixed rather than a runtime parameter: a [`GameLogic`] transition is
/// `#[comptime]`-specialised on [`GameSpec`] alone, which has no horizon field
/// of its own — and adding one there for the sake of this one game would be
/// the wrong place to put it, since not every game has an episode length in
/// the first place. Matches [`super::RecallEnv`]'s and the Python bindings'
/// own `RecallEnv(horizon=8)` default.
pub const RECALL_HORIZON: u32 = 8;

/// See a symbol once, name it [`RECALL_HORIZON`] steps later — the
/// device-native twin of [`super::RecallEnv`]. A policy with no memory cannot
/// beat `1 / symbols`, because nothing distinguishes the rewarded step from
/// any other except what the episode's first observation showed.
///
/// Two `u32`s of per-environment state (the cue and the clock), no floats.
///
/// Unmasked by default, matching [`super::RecallEnv`]. Under
/// [`GameSpec::with_action_mask`] the symbol after the cue (`(cue + 1) %
/// symbols`) is illegal on every step: the right answer always stays legal, so
/// the task is unchanged in what it rewards, while the fused and unfused paths
/// have a real, state-dependent mask to agree on.
pub struct Recall;

#[cube]
fn write_observation<F: Float + CubeElement>(
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
    // Two channels always visible: how far into the episode this is, and
    // whether the cue is on screen. Neither says what the cue was.
    obs[base + spec.action_dim] = F::cast_from(clock) * F::new(1.0_f32 / RECALL_HORIZON as f32);
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
        write_observation::<F>(obs, env, cue, 0u32, spec);
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
        let terminal = clock + 1u32 == RECALL_HORIZON;

        // Sparse, and at the end: naming the cue early earns nothing, so the
        // only way to score is to still know it when the episode closes.
        let correct = select(action == cue, F::new(1.0_f32), F::new(0.0_f32));
        let reward = select(terminal, correct, F::new(0.0_f32));

        // Auto-reset, which is what keeps the rollout rectangular.
        let fresh = hash_u32(env, seed_lo, seed_hi) % spec.action_dim as u32;
        let next_cue = select(terminal, fresh, cue);
        let next_clock = select(terminal, 0u32, clock + 1u32);
        ints[slot] = next_cue;
        ints[slot + 1] = next_clock;
        write_observation::<F>(obs, env, next_cue, next_clock, spec);

        Outcome::<F> {
            reward,
            done: select(terminal, F::new(1.0_f32), F::new(0.0_f32)),
        }
    }

    fn legal(
        env: u32,
        action: u32,
        ints: &Array<u32>,
        _floats: &Array<F>,
        #[comptime] spec: GameSpec,
    ) -> bool {
        let cue = ints[env as usize * spec.int_words];
        action != (cue + 1u32) % spec.action_dim as u32
    }
}

/// [`GameSpec`] for [`Recall`] over `symbols` actions: one observation channel
/// per symbol plus the clock and cue-present flag, `symbols` actions, and two
/// `u32`s of state (the cue and the clock).
pub fn recall_spec(symbols: usize) -> GameSpec {
    GameSpec::new(symbols + 2, symbols, 2)
}
