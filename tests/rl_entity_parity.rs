//! Fused against composed on the structured path: logits, values and gradients.
//!
//! Alone in its binary with a single test function on purpose: the switch is
//! process-global, so two comparisons running on neighbouring threads could
//! interleave modes. One function runs every mode back to back, and later
//! tasks (K1–K4) plug kernels in behind the switch with this as the judge.
//!
//! Both modes run the same composed code until those tasks land, so today this
//! passes trivially — the harness is what matters, not the result.

#![cfg(feature = "backend")]

use mamba3::autograd::Var;
use mamba3::backend::Device;
use mamba3::backends::Auto;
use mamba3::nn::entity::{EntityEncoderConfig, set_fused_entity};
use mamba3::nn::module::Module;
use mamba3::rl::{
    ActionHeadConfig, EntitySet, Mamba3Policy, Mamba3PolicyConfig, ObsSpec, PointerHeadConfig,
    RolloutEngine, Scoring,
};
use mamba3::ssm::config::{Discretization, StateDynamics};
use mamba3::tensor::Tensor;

type R = Auto;
type V = Var<R, f32>;

fn dev() -> Device<R> {
    Device::<R>::default()
}

/// Deterministic pseudo-random values in `[-1, 1)`: continuous, so tie-free.
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

fn constant(data: &[f32], shape: Vec<usize>) -> V {
    V::constant(Tensor::from_f32(data, shape, &dev()).unwrap())
}

/// `|actual - expected| <= tol * (1 + |expected|)`, element by element.
fn assert_relative(actual: &[f32], expected: &[f32], tol: f32, what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= tol * (1.0 + e.abs()),
            "{what}: index {i} got {a}, want {e}"
        );
    }
}

/// Logits agree: masked positions are exactly `f32::MIN` on both sides, and
/// everything else agrees relatively.
fn assert_logits_close(actual: &[f32], expected: &[f32], tol: f32, what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        if *e == f32::MIN {
            assert_eq!(*a, f32::MIN, "{what}: index {i} lost its mask");
        } else {
            assert!(
                (a - e).abs() <= tol * (1.0 + e.abs()),
                "{what}: index {i} got {a}, want {e}"
            );
        }
    }
}

const N: usize = 4;
const F: usize = 3;

struct Structured {
    spec: ObsSpec,
    actions: usize,
}

impl Structured {
    fn new(extra: usize) -> Self {
        Self {
            spec: ObsSpec::new(
                2,
                vec![EntitySet::new("tiles", N, F), EntitySet::new("units", 2, 2)],
            ),
            actions: N + extra,
        }
    }

    fn config(&self, scoring: Scoring, extra: usize) -> Mamba3PolicyConfig {
        Mamba3PolicyConfig::new(self.spec.obs_dim(), self.actions, 16, 2)
            .with_seed(7)
            .with_ssm(|s| {
                s.n_heads = 2;
                s.head_dim = 8;
                s.n_groups = 2;
                s.d_state = 4;
                s.chunk_size = 4;
                s.conv_kernel = Some(3);
                s.dynamics = StateDynamics::Rotational;
                s.discretization = Discretization::LearnedTrapezoid;
            })
            .with_obs_spec(self.spec.clone())
            .with_entity_encoder("tiles", EntityEncoderConfig::new(vec![8], 6))
            .with_entity_encoder(
                "units",
                EntityEncoderConfig::new(vec![], 4).with_slot_embedding(true),
            )
            .with_action_head(ActionHeadConfig::Pointer(
                PointerHeadConfig::new("tiles")
                    .with_hidden(8)
                    .with_scoring(scoring)
                    .with_extra_actions(extra),
            ))
    }

    fn policy(&self, scoring: Scoring, extra: usize) -> Mamba3Policy<R, f32> {
        self.config(scoring, extra).init::<R, f32>(&dev()).unwrap()
    }

    /// `[envs, steps, obs_dim]` with random features and some empty slots.
    /// One tiles presence is `0.5`: non-0/1 presence keeps today's semantics —
    /// nonzero counts as present for masks, and the value weights the mean —
    /// so the harness pins it before any kernel can reinterpret it.
    fn observations(&self, envs: usize, steps: usize, seed: u64) -> Vec<f32> {
        let values = noise(envs * steps * self.spec.obs_dim(), seed);
        let mut out = Vec::with_capacity(values.len());
        for (r, row) in values.chunks_exact(self.spec.obs_dim()).enumerate() {
            let (g, sets) = self.spec.unpack(row).unwrap();
            let sets: Vec<(Vec<f32>, Vec<f32>)> = sets
                .into_iter()
                .enumerate()
                .map(|(k, (f, p))| {
                    // Slot `(r + k) % count` is empty; the rest are present.
                    let empty = (r + k) % p.len();
                    let mut p: Vec<f32> = (0..p.len()).map(|i| (i != empty) as u8 as f32).collect();
                    if r == 0 && k == 0 {
                        p[1] = 0.5;
                    }
                    (f, p)
                })
                .collect();
            let refs: Vec<(&[f32], &[f32])> = sets.iter().map(|(f, p)| (&f[..], &p[..])).collect();
            out.extend(self.spec.pack(&g, &refs).unwrap());
        }
        out
    }
}

struct Snapshot {
    logits: Vec<f32>,
    values: Vec<f32>,
    grads: Vec<(String, Vec<f32>)>,
}

/// A forward window, its loss, and every parameter's gradient.
fn run_window(policy: &Mamba3Policy<R, f32>, obs: &V, reset: &Tensor<R, f32>) -> Snapshot {
    let (out, _) = policy.forward(obs, Some(reset), None).unwrap();
    let logits = out.logits.to_f32();
    let values = out.value.to_f32();
    // Masked logits are f32::MIN: a log-softmax keeps the loss finite where a
    // plain sum of logits would not.
    let loss = out
        .logits
        .log_softmax(2)
        .unwrap()
        .mean()
        .unwrap()
        .add(&out.value.sum().unwrap())
        .unwrap();
    let grads = loss.backward().unwrap();
    let mut named = Vec::new();
    for (name, param) in policy.named_parameters() {
        let g = grads
            .get(param.id())
            .unwrap_or_else(|| panic!("no gradient for {name}"))
            .to_f32();
        named.push((name, g));
    }
    Snapshot {
        logits,
        values,
        grads: named,
    }
}

#[test]
fn fused_and_composed_entity_paths_agree() {
    for scoring in [Scoring::Additive, Scoring::Dot] {
        let case = Structured::new(1);
        let policy = case.policy(scoring, 1);
        let (envs, steps, dim) = (2, 5, case.spec.obs_dim());
        let raw = case.observations(envs, steps, 6);
        let obs = constant(&raw, vec![envs, steps, dim]);
        let mut mask = vec![0.0f32; envs * steps];
        mask[2] = 1.0;
        mask[steps + 3] = 1.0;
        let reset = Tensor::from_f32(&mask, vec![envs, steps], &dev()).unwrap();

        set_fused_entity(true);
        let fused = run_window(&policy, &obs, &reset);
        let mut fused_engine = RolloutEngine::new(&policy, envs, &dev());
        let fused_step = fused_engine
            .step(
                &obs.slice(1, 0, 1)
                    .unwrap()
                    .reshape(vec![envs, 1, dim])
                    .unwrap(),
                Some(&Tensor::from_f32(&vec![0.0f32; envs], vec![envs], &dev()).unwrap()),
            )
            .unwrap();

        set_fused_entity(false);
        let composed = run_window(&policy, &obs, &reset);
        let mut composed_engine = RolloutEngine::new(&policy, envs, &dev());
        let composed_step = composed_engine
            .step(
                &obs.slice(1, 0, 1)
                    .unwrap()
                    .reshape(vec![envs, 1, dim])
                    .unwrap(),
                Some(&Tensor::from_f32(&vec![0.0f32; envs], vec![envs], &dev()).unwrap()),
            )
            .unwrap();

        let tag = format!("{scoring:?} forward");
        assert_logits_close(
            &fused.logits,
            &composed.logits,
            1e-6,
            &format!("{tag} logits"),
        );
        assert_relative(
            &fused.values,
            &composed.values,
            1e-6,
            &format!("{tag} values").as_str(),
        );
        assert_eq!(
            fused.grads.len(),
            composed.grads.len(),
            "{tag}: parameter count"
        );
        for ((name, f), (other, c)) in fused.grads.iter().zip(&composed.grads) {
            assert_eq!(name, other, "{tag}: parameter order");
            assert_relative(c, f, 1e-5, &format!("{tag} grad {name}").as_str());
        }

        let tag = format!("{scoring:?} rollout step");
        assert_logits_close(
            &fused_step.logits.to_f32(),
            &composed_step.logits.to_f32(),
            1e-6,
            &format!("{tag} logits"),
        );
        assert_relative(
            &fused_step.value.to_f32(),
            &composed_step.value.to_f32(),
            1e-6,
            &format!("{tag} values").as_str(),
        );
    }
    // Leave the process the way it was found: fused on.
    set_fused_entity(true);
}
