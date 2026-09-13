//! PPO on a game you wrote yourself, with the environment inside the rollout kernel.
//!
//! `train_rl.rs` learns [`mamba3::rl::RecallEnv`], which ships with the crate. This
//! one learns a game defined in *this file* — a `#[cube]` implementation of
//! [`GameLogic`] — which is the point: the transition is device code the crate can
//! compile into the middle of its own rollout kernel, so the action draw, the
//! environment step and the six writes that record the step become one launch
//! instead of eight. Nothing about the learning changes; `tests/rl_fused.rs` checks
//! that the fused window is byte-for-byte the window the unfused loop collects.
//!
//! The game is Catch. A ball falls one row a step down a `WIDTH x HEIGHT` grid, a
//! paddle on the bottom row moves left, right or not at all, and the only reward in
//! an episode is `1` for being under the ball when it lands. It is fully observable
//! — unlike the recall task, nothing here has to be remembered — so what it
//! demonstrates is the *plumbing*, not the recurrence: a game with integer state, a
//! rendered observation, an auto-reset, and PPO closing on it.
//!
//! ```text
//! cargo run --release --example train_rl_fused
//! ```

use mamba3::cubecl::prelude::*;
use mamba3::prelude::*;
use mamba3::rl::{GameLogic, GameSpec, GameWorld, Mamba3Policy, Outcome, PpoTask};
use mamba3::tensor::ops::random::hash_u32;
use mamba3::train::{AdamWConfig, Trainer, TrainerConfig};

type R = mamba3::backends::Auto;

/// Columns of the grid, and the number of places the paddle can be.
const WIDTH: u32 = 5;
/// Rows the ball falls through. The episode is this many steps long.
const HEIGHT: u32 = 5;
/// Left, stay, right.
const ACTIONS: usize = 3;
/// One-hot paddle, one-hot ball column, and how far the ball has fallen.
const OBS_DIM: usize = WIDTH as usize * 2 + 1;

const ENVS: usize = 32;
/// Four whole episodes per environment per window, so every window holds several
/// episode boundaries and the reset masking is exercised continuously.
const WINDOW: usize = HEIGHT as usize * 4;
const ROUNDS: usize = 120;
const EPOCHS: usize = 4;

// ---------------------------------------------------------------------------
// The game
// ---------------------------------------------------------------------------

/// Catch the falling ball with the paddle.
///
/// Three `u32`s of state per environment: the ball's column, the ball's row, and
/// the paddle's column.
pub struct Catch;

/// Where environment `env`'s three state words start.
#[cube]
fn slot(env: u32, #[comptime] spec: GameSpec) -> usize {
    env as usize * spec.int_words
}

/// Draw the grid as the policy sees it.
#[cube]
fn render<F: Float + CubeElement>(
    obs: &mut Array<F>,
    env: u32,
    ball_col: u32,
    ball_row: u32,
    paddle: u32,
    #[comptime] spec: GameSpec,
) {
    let base = env as usize * spec.obs_dim;
    let width = comptime!(WIDTH as usize);
    for i in 0..width {
        obs[base + i] = select(i == paddle as usize, F::new(1.0_f32), F::new(0.0_f32));
        obs[base + width + i] = select(i == ball_col as usize, F::new(1.0_f32), F::new(0.0_f32));
    }
    // How far the ball has fallen, scaled into `[0, 1)`. The agent needs it to know
    // how many moves it has left.
    obs[base + 2 * width] = F::cast_from(ball_row) / F::new(HEIGHT as f32);
}

/// A fresh ball in a random column, with the paddle back in the middle.
#[cube]
fn serve<F: Float + CubeElement>(
    env: u32,
    ints: &mut Array<u32>,
    obs: &mut Array<F>,
    seed_lo: u32,
    seed_hi: u32,
    #[comptime] spec: GameSpec,
) {
    let ball_col = hash_u32(env, seed_lo, seed_hi) % WIDTH;
    let paddle = comptime!(WIDTH / 2);
    let at = slot(env, spec);
    ints[at] = ball_col;
    ints[at + 1] = 0u32;
    ints[at + 2] = paddle;
    render::<F>(obs, env, ball_col, 0u32, paddle, spec);
}

#[cube]
impl<F: Float + CubeElement> GameLogic<F> for Catch {
    fn reset(
        env: u32,
        ints: &mut Array<u32>,
        _floats: &mut Array<F>,
        obs: &mut Array<F>,
        seed_lo: u32,
        seed_hi: u32,
        #[comptime] spec: GameSpec,
    ) {
        serve::<F>(env, ints, obs, seed_lo, seed_hi, spec);
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
        let at = slot(env, spec);
        let ball_col = ints[at];
        let ball_row = ints[at + 1];
        let paddle = ints[at + 2];

        // Action 0 is left, 1 is stay, 2 is right. Written as two guarded moves
        // rather than as `paddle + action - 1` because these are unsigned: at column
        // zero that expression wraps rather than clamping, and the paddle teleports.
        let mut moved = paddle;
        if action == 0u32 && paddle > 0u32 {
            moved = paddle - 1u32;
        }
        if action == 2u32 && paddle + 1u32 < WIDTH {
            moved = paddle + 1u32;
        }

        let next_row = ball_row + 1u32;
        let landed = next_row + 1u32 == HEIGHT;
        let caught = select(moved == ball_col, F::new(1.0_f32), F::new(0.0_f32));
        let reward = select(landed, caught, F::new(0.0_f32));

        if landed {
            // Auto-reset: the next observation is the first of a new episode, which
            // is what keeps a fixed-length rollout rectangular.
            serve::<F>(env, ints, obs, seed_lo, seed_hi, spec);
        } else {
            ints[at + 1] = next_row;
            ints[at + 2] = moved;
            render::<F>(obs, env, ball_col, next_row, moved, spec);
        }

        Outcome::<F> {
            reward,
            done: select(landed, F::new(1.0_f32), F::new(0.0_f32)),
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

// ---------------------------------------------------------------------------

fn policy(device: &Device<R>) -> Result<Mamba3Policy<R, f32>> {
    mamba3::rl::Mamba3PolicyConfig::new(OBS_DIM, ACTIONS, 64, 2)
        .with_seed(7)
        .with_ssm(|s| {
            s.n_heads = 4;
            s.head_dim = 16;
            s.n_groups = 4;
            s.d_state = 8;
            s.chunk_size = 8;
            s.conv_kernel = Some(4);
        })
        .init::<R, f32>(device)
}

fn main() -> Result<()> {
    let device = Device::<R>::default();
    let policy = policy(&device)?;
    let spec = GameSpec::new(OBS_DIM, ACTIONS, 3);
    let mut world: GameWorld<R, f32, Catch> = GameWorld::new(ENVS, spec, 17, &device)?;

    let config = PpoConfig::default().with_coefficients(0.5, 0.02);
    let task = PpoTask::new(&policy, config);
    let mut trainer = Trainer::new(
        TrainerConfig::builder()
            .learning_rate(3e-4)
            .max_grad_norm(0.5)
            .build()?,
        AdamWConfig::builder()
            .learning_rate(3e-4)
            .build()
            .init::<R, f32>(),
    );
    let mut collector =
        mamba3::rl::Collector::new(&policy, ENVS, WINDOW, OBS_DIM, &device)?.with_seed(11);

    println!("Catch on a {WIDTH}x{HEIGHT} grid: {ENVS} environments, windows of {WINDOW} steps");
    println!("the transition is a #[cube] fn in this file, fused into the rollout step\n");
    println!("{:>6}  {:>14}", "round", "return/episode");

    for round in 0..ROUNDS {
        // The whole window — policy, action draw, game, trajectory write — is one
        // queue of launches. Nothing is read back until the line below asks.
        let report = collector.collect_fused(&mut world)?;
        let batch = collector.ppo_batch(&report, &config)?;
        for _ in 0..EPOCHS {
            trainer.step(&task, std::slice::from_ref(&batch))?;
        }

        if round % 10 == 0 || round + 1 == ROUNDS {
            // The one synchronisation point, and it is between updates rather than
            // inside one.
            let (mean, _count) = collector.episode_return()?;
            let earned = mean.to_f32()[0];
            println!("{round:>6}  {earned:>14.3}");
        }
    }

    println!(
        "\na paddle that never moves catches the ball {:.2} of the time; \
         a perfect one catches all of them",
        1.0 / WIDTH as f32
    );
    Ok(())
}
