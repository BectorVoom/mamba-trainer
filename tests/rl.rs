//! The reinforcement learning engines.
//!
//! Two engines compute the same function at different shapes: `mamba3_step` one
//! observation at a time, and the chunked scan a whole `[B, T]` trajectory at
//! once. Everything here exists to hold them to that, especially across episode
//! boundaries — where a reset has to make the past invisible through *both* of the
//! paths a Mamba-3 layer reaches backwards through, the recurrence and the short
//! causal convolution.

#![cfg(feature = "backend")]

use mamba3::autograd::Var;
use mamba3::backend::Device;
use mamba3::backends::Auto;
use mamba3::models::mamba3::MixerCache;
use mamba3::nn::module::Module;
use mamba3::rl::{Mamba3Policy, Mamba3PolicyConfig, RolloutEngine};
use mamba3::ssm::config::{Discretization, StateDynamics};
use mamba3::tensor::Tensor;

type R = Auto;
type V = Var<R, f32>;

fn dev() -> Device<R> {
    Device::<R>::default()
}

/// Deterministic pseudo-random values in `[-1, 1)`.
fn noise(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        })
        .collect()
}

fn assert_close(actual: &[f32], expected: &[f32], eps: f32, what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length mismatch");
    let mut worst = 0.0f32;
    let mut at = 0;
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        let err = (a - e).abs() / (1.0 + e.abs());
        if err > worst {
            worst = err;
            at = i;
        }
    }
    assert!(
        worst <= eps,
        "{what}: worst relative error {worst} at index {at} (got {}, want {})",
        actual[at],
        expected[at]
    );
}

/// The shape of every case below. Small enough to run on the CPU runtime in the
/// time a test should take, wide enough that a broken broadcast shows up.
struct Case {
    envs: usize,
    steps: usize,
    obs_dim: usize,
    actions: usize,
    layers: usize,
}

impl Case {
    fn small() -> Self {
        Self {
            envs: 3,
            steps: 9,
            obs_dim: 6,
            actions: 4,
            layers: 2,
        }
    }

    fn policy(&self, rotational: bool) -> Mamba3Policy<R, f32> {
        Mamba3PolicyConfig::new(self.obs_dim, self.actions, 16, self.layers)
            .with_seed(7)
            .with_ssm(|s| {
                s.n_heads = 2;
                s.head_dim = 8;
                s.n_groups = 2;
                s.d_state = 4;
                s.chunk_size = 4;
                s.conv_kernel = Some(3);
                s.dynamics = if rotational {
                    StateDynamics::Rotational
                } else {
                    StateDynamics::Real
                };
                s.discretization = Discretization::LearnedTrapezoid;
            })
            .init::<R, f32>(&dev())
            .expect("policy config is valid")
    }

    /// `[envs, steps, obs_dim]` of noise.
    fn observations(&self, seed: u64) -> V {
        V::constant(
            Tensor::from_f32(
                &noise(self.envs * self.steps * self.obs_dim, seed),
                vec![self.envs, self.steps, self.obs_dim],
                &dev(),
            )
            .unwrap(),
        )
    }

    /// One position of an observation window, as the rollout step wants it.
    fn observation_at(&self, obs: &V, step: usize) -> V {
        obs.slice(1, step, 1)
            .unwrap()
            .reshape(vec![self.envs, 1, self.obs_dim])
            .unwrap()
    }
}

/// A `[envs, steps]` reset mask from a per-environment schedule of reset steps.
fn reset_mask(envs: usize, steps: usize, schedule: &[&[usize]]) -> Vec<f32> {
    let mut mask = vec![0.0f32; envs * steps];
    for (env, positions) in schedule.iter().enumerate() {
        for &p in *positions {
            mask[env * steps + p] = 1.0;
        }
    }
    mask
}

/// Roll `case.steps` observations through the step engine, one at a time.
///
/// `reset` is the flattened `[envs, steps]` mask; the column for step `t` is what
/// the engine is told before consuming observation `t`.
fn roll(case: &Case, policy: &Mamba3Policy<R, f32>, obs: &V, reset: Option<&[f32]>) -> Vec<f32> {
    let mut engine = RolloutEngine::new(policy, case.envs, &dev());
    let mut logits = Vec::with_capacity(case.steps);
    for step in 0..case.steps {
        let flags = reset.map(|m| {
            let column: Vec<f32> = (0..case.envs).map(|e| m[e * case.steps + step]).collect();
            Tensor::from_f32(&column, vec![case.envs], &dev()).unwrap()
        });
        let out = engine
            .step(&case.observation_at(obs, step), flags.as_ref())
            .unwrap();
        logits.push(out.logits);
    }
    mamba3::autograd::cat(&logits, 1).unwrap().to_f32()
}

/// One scan over the whole window.
fn scan(case: &Case, policy: &Mamba3Policy<R, f32>, obs: &V, reset: Option<&[f32]>) -> Vec<f32> {
    let mask = reset.map(|m| Tensor::from_f32(m, vec![case.envs, case.steps], &dev()).unwrap());
    policy
        .forward(obs, mask.as_ref(), None)
        .unwrap()
        .0
        .logits
        .to_f32()
}

// ---------------------------------------------------------------------------
// Acceptance criterion 1: the two engines agree
// ---------------------------------------------------------------------------

#[test]
fn rollout_matches_the_parallel_scan() {
    for rotational in [false, true] {
        let case = Case::small();
        let policy = case.policy(rotational);
        let obs = case.observations(1);
        assert_close(
            &roll(&case, &policy, &obs, None),
            &scan(&case, &policy, &obs, None),
            1e-4,
            &format!("rollout vs scan (rotational={rotational})"),
        );
    }
}

#[test]
fn rollout_matches_the_parallel_scan_across_episode_boundaries() {
    for rotational in [false, true] {
        let case = Case::small();
        let policy = case.policy(rotational);
        let obs = case.observations(2);
        // Environment 0 restarts twice, environment 1 once mid-chunk, environment
        // 2 never — so the mask exercises a boundary inside a chunk, on a chunk
        // edge, and an untouched row, all in one launch.
        let mask = reset_mask(case.envs, case.steps, &[&[0, 5], &[4], &[]]);
        assert_close(
            &roll(&case, &policy, &obs, Some(&mask)),
            &scan(&case, &policy, &obs, Some(&mask)),
            1e-4,
            &format!("rollout vs scan with resets (rotational={rotational})"),
        );
    }
}

#[test]
fn a_reset_is_indistinguishable_from_a_fresh_episode() {
    let case = Case::small();
    let policy = case.policy(true);
    let obs = case.observations(3);
    let boundary = 4;

    // Every environment restarts at the same step, so the tail of this rollout
    // must equal a rollout that only ever saw the tail.
    let mask = reset_mask(case.envs, case.steps, &vec![&[boundary][..]; case.envs]);
    let with_reset = roll(&case, &policy, &obs, Some(&mask));

    let tail_len = case.steps - boundary;
    let tail = Case {
        steps: tail_len,
        ..Case::small()
    };
    let tail_obs = obs.slice(1, boundary, tail_len).unwrap();
    let fresh = roll(&tail, &policy, &tail_obs, None);

    // Compare the post-boundary positions of the long rollout with the whole of
    // the short one.
    let width = case.actions;
    let mut tail_of_long = Vec::with_capacity(fresh.len());
    for env in 0..case.envs {
        for step in boundary..case.steps {
            let base = (env * case.steps + step) * width;
            tail_of_long.extend_from_slice(&with_reset[base..base + width]);
        }
    }
    assert_close(
        &tail_of_long,
        &fresh,
        1e-5,
        "after a reset vs a fresh start",
    );
}

#[test]
fn a_reset_erases_the_state_it_is_given() {
    // Two rollouts that differ only before the boundary must agree after it.
    let case = Case::small();
    let policy = case.policy(true);
    let boundary = 5;
    let mask = reset_mask(case.envs, case.steps, &vec![&[boundary][..]; case.envs]);

    let a = case.observations(11);
    let b = case.observations(12);
    // Same tail, different heads.
    let mixed = mamba3::autograd::cat(
        &[
            b.slice(1, 0, boundary).unwrap(),
            a.slice(1, boundary, case.steps - boundary).unwrap(),
        ],
        1,
    )
    .unwrap();

    let from_a = roll(&case, &policy, &a, Some(&mask));
    let from_mixed = roll(&case, &policy, &mixed, Some(&mask));

    let width = case.actions;
    let tail = |v: &[f32]| {
        let mut out = Vec::new();
        for env in 0..case.envs {
            for step in boundary..case.steps {
                let base = (env * case.steps + step) * width;
                out.extend_from_slice(&v[base..base + width]);
            }
        }
        out
    };
    assert_close(
        &tail(&from_mixed),
        &tail(&from_a),
        1e-6,
        "a different past leaking through a reset",
    );
}

#[test]
fn the_fused_reset_matches_an_explicit_state_clear() {
    // `RolloutEngine::step` folds the mask into the step's own kernels;
    // `Mamba3StateBuffer::reset_env_states` clears the buffers directly. The two
    // are different amounts of work and must be the same numbers.
    let case = Case::small();
    let policy = case.policy(true);
    let obs = case.observations(4);
    let mask = reset_mask(case.envs, case.steps, &[&[2, 6], &[3], &[7]]);

    let fused = roll(&case, &policy, &obs, Some(&mask));

    let mut engine = RolloutEngine::new(&policy, case.envs, &dev());
    let mut logits = Vec::with_capacity(case.steps);
    for step in 0..case.steps {
        let column: Vec<f32> = (0..case.envs)
            .map(|e| mask[e * case.steps + step])
            .collect();
        let flags = Tensor::from_f32(&column, vec![case.envs], &dev()).unwrap();
        engine.state_mut().reset_env_states(&flags).unwrap();
        let out = engine.step(&case.observation_at(&obs, step), None).unwrap();
        logits.push(out.logits);
    }
    let explicit = mamba3::autograd::cat(&logits, 1).unwrap().to_f32();

    assert_close(&explicit, &fused, 1e-6, "explicit clear vs fused reset");
}

// ---------------------------------------------------------------------------
// Acceptance criterion 2: the loop is flat
//
// The measurement of that lives in `tests/rl_footprint.rs`, alone in its own
// binary: it reads process-wide launch and read counters, which any test running
// beside it would perturb.
// ---------------------------------------------------------------------------

#[test]
fn the_state_buffer_is_sized_by_the_configuration_alone() {
    let case = Case::small();
    let policy = case.policy(true);
    let engine = RolloutEngine::new(&policy, case.envs, &dev());
    let cfg = policy.config().ssm.clone();

    // Two state tensors per layer plus the convolution history.
    let ssm = 2 * case.envs * cfg.n_heads * cfg.head_dim * cfg.d_state;
    let angle = case.envs * cfg.n_heads * cfg.d_state / 2;
    let conv = case.envs * (cfg.conv_kernel.unwrap() - 1) * (cfg.d_inner() + 2 * cfg.bc_width());
    assert_eq!(
        engine.state().num_elements(),
        case.layers * (ssm + angle + conv),
    );
}

// ---------------------------------------------------------------------------
// The gradient
// ---------------------------------------------------------------------------

/// How much of a pre-boundary gradient counts as none.
///
/// The sharpest statement of what a reset means is that an observation before the
/// boundary cannot change any output after it, so its gradient from a
/// post-boundary loss should be zero. Two things keep it merely negligible rather
/// than literally zero, and both are deliberate:
///
/// * the decay is floored at `exp(-60)`, about `9e-27`, rather than set to zero,
///   which is what keeps the masked upper triangle away from `exp` overflow;
/// * with a rotational transition the running angle is shared across the
///   boundary. It cancels exactly — only the rotation *relative* to the position
///   reading it reaches the output — but the two branches of that cancellation
///   are differentiated separately, so their sum leaves a float residue around
///   `1e-14` on values of order one.
///
/// Both are more than ten orders of magnitude below anything the model computes,
/// so the bound below is a real test of the cut and not of the arithmetic.
const RESET_GRADIENT_FLOOR: f32 = 1e-9;

#[test]
fn no_gradient_crosses_a_reset() {
    let case = Case::small();
    let policy = case.policy(true);
    let boundary = 4;
    let mask = Tensor::from_f32(
        &reset_mask(case.envs, case.steps, &vec![&[boundary][..]; case.envs]),
        vec![case.envs, case.steps],
        &dev(),
    )
    .unwrap();

    let obs = V::traced(
        Tensor::from_f32(
            &noise(case.envs * case.steps * case.obs_dim, 5),
            vec![case.envs, case.steps, case.obs_dim],
            &dev(),
        )
        .unwrap(),
    );

    let (out, _) = policy.forward(&obs, Some(&mask), None).unwrap();
    // A loss that only sees positions at and after the boundary.
    let loss = out
        .logits
        .slice(1, boundary, case.steps - boundary)
        .unwrap()
        .sum()
        .unwrap();
    let grads = loss.backward_retain().unwrap();
    let d_obs = grads
        .node(obs.node().unwrap())
        .expect("the observation is on the tape")
        .to_f32();

    for env in 0..case.envs {
        for step in 0..boundary {
            let base = (env * case.steps + step) * case.obs_dim;
            for (i, g) in d_obs[base..base + case.obs_dim].iter().enumerate() {
                assert!(
                    g.abs() <= RESET_GRADIENT_FLOOR,
                    "gradient {g} reached observation {i} of step {step} \
                     (env {env}), which a reset at {boundary} should have hidden"
                );
            }
        }
    }

    // The sanity half: gradients after the boundary are large, so the bound above
    // is a statement about the cut and not about a pass that produced nothing.
    let after = d_obs
        .iter()
        .skip(boundary * case.obs_dim)
        .fold(0.0f32, |m, g| m.max(g.abs()));
    assert!(
        after > 1e-3,
        "no gradient reached the post-reset window either (max {after})"
    );
}

#[test]
fn a_trajectory_pass_is_differentiable_and_reaches_every_parameter() {
    let case = Case::small();
    let policy = case.policy(true);
    let obs = case.observations(9);
    let mask = Tensor::from_f32(
        &reset_mask(case.envs, case.steps, &[&[3], &[6], &[]]),
        vec![case.envs, case.steps],
        &dev(),
    )
    .unwrap();

    let (out, _) = policy.forward(&obs, Some(&mask), None).unwrap();
    // A stand-in for an actor-critic objective: both heads contribute.
    let loss = out
        .logits
        .sum()
        .unwrap()
        .add(&out.value.sum().unwrap())
        .unwrap();
    let grads = loss.backward().unwrap();

    let params = policy.named_parameters();
    assert!(!params.is_empty());
    for (name, param) in &params {
        let g = grads
            .get(param.id())
            .unwrap_or_else(|| panic!("no gradient for {name}"));
        assert!(
            g.to_f32().iter().all(|v| v.is_finite()),
            "non-finite gradient for {name}"
        );
    }
}

// ---------------------------------------------------------------------------
// Continuing a rollout
// ---------------------------------------------------------------------------

#[test]
fn a_split_window_matches_one_pass() {
    // Truncated backpropagation hands one window's end state to the next. With a
    // reset straddling the split, both halves have to agree with a single pass.
    let case = Case::small();
    let policy = case.policy(true);
    let obs = case.observations(13);
    let split = 5;
    let mask_data = reset_mask(case.envs, case.steps, &[&[2], &[6], &[4, 7]]);
    let mask = |from: usize, len: usize| {
        let mut out = Vec::with_capacity(case.envs * len);
        for env in 0..case.envs {
            out.extend_from_slice(&mask_data[env * case.steps + from..][..len]);
        }
        Tensor::from_f32(&out, vec![case.envs, len], &dev()).unwrap()
    };

    let whole = policy
        .forward(&obs, Some(&mask(0, case.steps)), None)
        .unwrap()
        .0
        .logits
        .to_f32();

    let zeros: Vec<MixerCache<R, f32>> = policy.empty_state(case.envs, &dev()).snapshot();
    let (first, carry) = policy
        .forward(
            &obs.slice(1, 0, split).unwrap(),
            Some(&mask(0, split)),
            Some(&zeros),
        )
        .unwrap();
    let carry: Vec<MixerCache<R, f32>> = carry
        .expect("an initial state asks for a final one")
        .iter()
        .map(|c| c.detach())
        .collect();
    let (second, _) = policy
        .forward(
            &obs.slice(1, split, case.steps - split).unwrap(),
            Some(&mask(split, case.steps - split)),
            Some(&carry),
        )
        .unwrap();

    let joined = mamba3::autograd::cat(&[first.logits, second.logits], 1)
        .unwrap()
        .to_f32();
    assert_close(&joined, &whole, 1e-4, "two windows vs one");
}

// ---------------------------------------------------------------------------
// The fused PPO terms against the forms they replace
// ---------------------------------------------------------------------------
//
// `Var::ppo_surrogate` and `Var::ppo_value_loss` each collapse a chain of
// elementwise ops into one launch, and each carries a hand-written adjoint. The
// oracle is the chain itself, transcribed below out of the same primitives the
// objective used before it was fused, so these tests fail if the fused kernel and
// the composed form ever disagree — in value or in gradient.
//
// The inputs are deliberately nasty: ratios well outside the trust region in both
// directions, advantages of both signs, and values on both sides of the clip, so
// every branch of both kernels is taken and the tie conventions (`minimum` gives a
// tie to its left operand, `maximum` to its right) are actually exercised.

const CLIP: f32 = 0.2;

/// `log_probs`, `old`, `advantages` covering every branch of the surrogate.
fn surrogate_inputs(dev: &Device<R>) -> (Vec<f32>, Tensor<R, f32>, Tensor<R, f32>) {
    // `chosen - old` spans well past ln(1 ± 0.2) either way, so the ratio lands
    // inside, above and below the trust region.
    let chosen: Vec<f32> = (0..64).map(|i| -1.0 + i as f32 * 0.04).collect();
    let old: Vec<f32> = (0..64)
        .map(|i| -1.0 + (i as f32 * 0.037).sin() * 0.5)
        .collect();
    let advantages: Vec<f32> = (0..64)
        .map(|i| if i % 3 == 0 { -1.0 } else { 1.0 } * (0.1 + i as f32 * 0.05))
        .collect();
    (
        chosen,
        Tensor::from_f32(&old, vec![64], dev).unwrap(),
        Tensor::from_f32(&advantages, vec![64], dev).unwrap(),
    )
}

#[test]
fn fused_ppo_surrogate_matches_the_composed_form() {
    let dev = dev();
    let (chosen, old, advantages) = surrogate_inputs(&dev);
    let seed = Tensor::from_f32(&chosen, vec![64], &dev).unwrap();

    // The composed form: a subtraction, an exp, two products, a clamp and a minimum.
    let composed_input = V::traced(seed.clone());
    let old_v = V::constant(old.clone());
    let adv_v = V::constant(advantages.clone());
    let ratio = composed_input.sub(&old_v).unwrap().exp();
    let composed = ratio
        .mul(&adv_v)
        .unwrap()
        .minimum(&ratio.clamp(1.0 - CLIP, 1.0 + CLIP).mul(&adv_v).unwrap())
        .unwrap();

    let fused_input = V::traced(seed);
    let (fused, fused_ratio) = fused_input.ppo_surrogate(&old, &advantages, CLIP).unwrap();

    assert_close(
        &fused.tensor().to_f32(),
        &composed.tensor().to_f32(),
        1e-6,
        "surrogate value",
    );
    assert_close(
        &fused_ratio.to_f32(),
        &ratio.tensor().to_f32(),
        1e-6,
        "surrogate ratio",
    );

    // Gradients, under a non-uniform upstream so that no position is masked by a
    // constant factor and a per-position error cannot cancel in the sum.
    let weights: Vec<f32> = (0..64).map(|i| 0.5 + (i % 7) as f32 * 0.3).collect();
    let w = V::constant(Tensor::from_f32(&weights, vec![64], &dev).unwrap());
    let composed_grads = composed
        .mul(&w)
        .unwrap()
        .sum()
        .unwrap()
        .backward_retain()
        .unwrap();
    let fused_grads = fused
        .mul(&w)
        .unwrap()
        .sum()
        .unwrap()
        .backward_retain()
        .unwrap();
    assert_close(
        &fused_grads
            .node(fused_input.node().unwrap())
            .unwrap()
            .to_f32(),
        &composed_grads
            .node(composed_input.node().unwrap())
            .unwrap()
            .to_f32(),
        1e-5,
        "surrogate gradient",
    );
}

#[test]
fn fused_ppo_value_loss_matches_the_composed_form() {
    let dev = dev();
    // Predictions on both sides of the clip around the recorded estimate, and
    // returns that make each branch the larger error in turn.
    let predicted: Vec<f32> = (0..64).map(|i| -1.5 + i as f32 * 0.05).collect();
    let old: Vec<f32> = (0..64).map(|i| (i as f32 * 0.05).cos()).collect();
    let returns: Vec<f32> = (0..64).map(|i| (i as f32 * 0.11).sin() * 1.5).collect();
    let seed = Tensor::from_f32(&predicted, vec![64], &dev).unwrap();
    let old_t = Tensor::from_f32(&old, vec![64], &dev).unwrap();
    let returns_t = Tensor::from_f32(&returns, vec![64], &dev).unwrap();

    let weights: Vec<f32> = (0..64).map(|i| 0.5 + (i % 5) as f32 * 0.25).collect();
    let w = V::constant(Tensor::from_f32(&weights, vec![64], &dev).unwrap());

    for clip in [false, true] {
        let composed_input = V::traced(seed.clone());
        let returns_v = V::constant(returns_t.clone());
        let error = composed_input.sub(&returns_v).unwrap();
        let squared = error.mul(&error).unwrap();
        let composed = if clip {
            let old_v = V::constant(old_t.clone());
            let bounded = old_v
                .add(&composed_input.sub(&old_v).unwrap().clamp(-CLIP, CLIP))
                .unwrap();
            let clipped_error = bounded.sub(&returns_v).unwrap();
            squared
                .maximum(&clipped_error.mul(&clipped_error).unwrap())
                .unwrap()
        } else {
            squared
        };

        let fused_input = V::traced(seed.clone());
        let fused = fused_input
            .ppo_value_loss(&returns_t, &old_t, CLIP, clip)
            .unwrap();

        assert_close(
            &fused.tensor().to_f32(),
            &composed.tensor().to_f32(),
            1e-6,
            &format!("value loss (clip={clip})"),
        );

        let composed_grads = composed
            .mul(&w)
            .unwrap()
            .sum()
            .unwrap()
            .backward_retain()
            .unwrap();
        let fused_grads = fused
            .mul(&w)
            .unwrap()
            .sum()
            .unwrap()
            .backward_retain()
            .unwrap();
        assert_close(
            &fused_grads
                .node(fused_input.node().unwrap())
                .unwrap()
                .to_f32(),
            &composed_grads
                .node(composed_input.node().unwrap())
                .unwrap()
                .to_f32(),
            1e-5,
            &format!("value loss gradient (clip={clip})"),
        );
    }
}

#[test]
fn fused_ppo_diagnostics_match_the_composed_form() {
    use mamba3::tensor::ops::{elemwise, fused};

    let dev = dev();
    let (chosen, old, _) = surrogate_inputs(&dev);
    let chosen_t = Tensor::from_f32(&chosen, vec![64], &dev).unwrap();
    let log_ratio = elemwise::sub(&chosen_t, &old).unwrap();
    let ratio = elemwise::exp(&log_ratio);

    let (kl, clipped) = fused::ppo_diagnostics(&chosen_t, &old, &ratio, CLIP).unwrap();

    let want_kl = elemwise::sub(&elemwise::add_scalar(&ratio, -1.0), &log_ratio).unwrap();
    let departure = elemwise::abs(&elemwise::add_scalar(&ratio, -1.0));
    let want_clipped = elemwise::gt_scalar(&departure, CLIP);

    assert_close(&kl.to_f32(), &want_kl.to_f32(), 1e-6, "approx kl terms");
    assert_close(&clipped.to_f32(), &want_clipped.to_f32(), 0.0, "clip flags");
    // The estimator is non-negative by construction; a negative one would mean the
    // fused form had lost the `- log r` term.
    assert!(
        kl.to_f32().iter().all(|v| *v >= -1e-6),
        "the KL estimator must be non-negative for every ratio"
    );
}
