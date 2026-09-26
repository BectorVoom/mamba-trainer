//! Structured observations: the observation spec, the shared entity encoder,
//! masked pooling, the pointer head, and a policy built from them.
//!
//! The properties that make the structure worth having are symmetries, so most of
//! what is here checks one: permuting the entities of a set leaves the pooled
//! summary unchanged and permutes the pointer's logits the same way, and what an
//! environment leaves in an empty slot never reaches the output.

#![cfg(feature = "backend")]

use mamba3::autograd::Var;
use mamba3::backend::Device;
use mamba3::backends::Auto;
use mamba3::nn::entity::{
    EntityEncoderConfig, PoolKind, PoolingConfig, Presence, masked_pool, pool_parts,
};
use mamba3::nn::module::Module;
use mamba3::rl::{
    ActionHeadConfig, BehaviourCloningTask, EntitySet, ImitationBatch, Mamba3Policy,
    Mamba3PolicyConfig, ObsSpec, PointerHeadConfig, RolloutEngine, Scoring,
};
use mamba3::ssm::config::{Discretization, StateDynamics};
use mamba3::tensor::ops::index::IdTensor;
use mamba3::tensor::ops::random::Rng;
use mamba3::tensor::{Shape, Tensor};
use mamba3::train::{AdamWConfig, Trainer, TrainerConfig};

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

fn constant(data: &[f32], shape: Vec<usize>) -> V {
    V::constant(Tensor::from_f32(data, shape, &dev()).unwrap())
}

fn assert_close(actual: &[f32], expected: &[f32], eps: f32, what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= eps * (1.0 + e.abs()),
            "{what}: index {i} got {a}, want {e}"
        );
    }
}

/// Central-difference check of the gradient of scalar `f` at `data`.
fn check_grad<F>(name: &str, data: &[f32], shape: impl Into<Shape> + Clone, f: F)
where
    F: Fn(&V) -> V,
{
    let shape = shape.into();
    let x = V::traced(Tensor::from_f32(data, shape.clone(), &dev()).unwrap());
    let grads = f(&x).backward_retain().unwrap();
    let analytic = grads
        .node(x.node().unwrap())
        .unwrap_or_else(|| panic!("{name}: no gradient reached the input"))
        .to_f32();
    let eps = 1e-3f32;
    for i in 0..data.len() {
        let at = |delta: f32| {
            let mut moved = data.to_vec();
            moved[i] += delta;
            f(&V::constant(
                Tensor::from_f32(&moved, shape.clone(), &dev()).unwrap(),
            ))
            .to_f32()[0]
        };
        let numeric = (at(eps) - at(-eps)) / (2.0 * eps);
        assert!(
            (analytic[i] - numeric).abs() < 2e-2 * (1.0 + numeric.abs()),
            "{name}: grad[{i}] analytic={} numeric={numeric}",
            analytic[i]
        );
    }
}

/// `[B, T, N, 1]` presence with the given absent slots per `(b, t)` row.
fn presence(rows: usize, n: usize, absent: &[(usize, usize)]) -> Vec<f32> {
    let mut p = vec![1.0f32; rows * n];
    for &(row, slot) in absent {
        p[row * n + slot] = 0.0;
    }
    p
}

/// Permute the entity axis of `[rows, N, width]` data.
fn permute_rows(data: &[f32], rows: usize, n: usize, width: usize, perm: &[usize]) -> Vec<f32> {
    let mut out = vec![0.0; data.len()];
    for r in 0..rows {
        for (dst, &src) in perm.iter().enumerate() {
            let from = (r * n + src) * width;
            let to = (r * n + dst) * width;
            out[to..to + width].copy_from_slice(&data[from..from + width]);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// E1: the observation spec
// ---------------------------------------------------------------------------

fn two_set_spec() -> ObsSpec {
    ObsSpec::new(
        2,
        vec![EntitySet::new("tiles", 3, 2), EntitySet::new("units", 2, 1)],
    )
}

#[test]
fn obs_dim_counts_globals_features_and_presence() {
    let spec = two_set_spec();
    assert_eq!(spec.obs_dim(), 2 + 3 * 3 + 2 * 2);
    assert_eq!(spec.offsets(), vec![2, 11]);
}

#[test]
fn pack_then_unpack_is_the_identity() {
    let spec = two_set_spec();
    let globals = [0.5, -1.0];
    let tiles = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tiles_p = [1.0, 0.0, 1.0];
    let units = [7.0, 8.0];
    let units_p = [0.0, 1.0];
    let flat = spec
        .pack(&globals, &[(&tiles, &tiles_p), (&units, &units_p)])
        .unwrap();
    assert_eq!(
        flat,
        vec![
            0.5, -1.0, 1.0, 2.0, 1.0, 3.0, 4.0, 0.0, 5.0, 6.0, 1.0, 7.0, 0.0, 8.0, 1.0
        ]
    );
    let (g, sets) = spec.unpack(&flat).unwrap();
    assert_eq!(g, globals);
    assert_eq!(sets[0], (tiles.to_vec(), tiles_p.to_vec()));
    assert_eq!(sets[1], (units.to_vec(), units_p.to_vec()));
    assert!(spec.pack(&globals, &[(&tiles, &tiles_p)]).is_err());
    assert!(
        spec.pack(&[0.0], &[(&tiles, &tiles_p), (&units, &units_p)])
            .is_err()
    );
}

#[test]
fn split_on_the_device_matches_the_packed_parts() {
    let spec = two_set_spec();
    let (b, t) = (2, 3);
    let raw = noise(b * t * spec.obs_dim(), 5);
    let obs = constant(&raw, vec![b, t, spec.obs_dim()]);
    let split = spec.split(&obs).unwrap();

    let mut want_globals = Vec::new();
    let mut want = vec![(Vec::new(), Vec::new()); 2];
    for row in raw.chunks_exact(spec.obs_dim()) {
        let (g, sets) = spec.unpack(row).unwrap();
        want_globals.extend(g);
        for (w, (f, p)) in want.iter_mut().zip(sets) {
            w.0.extend(f);
            w.1.extend(p);
        }
    }
    assert_eq!(split.globals.unwrap().to_f32(), want_globals);
    for ((parts, (f, p)), set) in split.sets.iter().zip(&want).zip(&spec.sets) {
        assert_eq!(parts.features.dims(), &[b, t, set.count, set.features]);
        assert_eq!(parts.presence.dims(), &[b, t, set.count, 1]);
        assert_eq!(&parts.features.to_f32(), f);
        assert_eq!(&parts.presence.to_f32(), p);
    }
}

#[test]
fn a_spec_without_globals_or_with_one_set_splits_without_losing_the_trace() {
    let spec = ObsSpec::new(0, vec![EntitySet::new("a", 2, 2)]);
    let obs = V::traced(Tensor::from_f32(&noise(6, 1), vec![1, 1, 6], &dev()).unwrap());
    let split = spec.split(&obs).unwrap();
    assert!(split.globals.is_none());
    let grads = split.sets[0]
        .features
        .sum()
        .unwrap()
        .backward_retain()
        .unwrap();
    let g = grads.node(obs.node().unwrap()).unwrap().to_f32();
    assert_eq!(g, vec![1.0, 1.0, 0.0, 1.0, 1.0, 0.0]);
}

#[test]
fn validation_names_the_offending_field() {
    let err = |spec: ObsSpec| spec.validate().unwrap_err().to_string();
    assert!(err(ObsSpec::new(3, vec![])).contains("obs_spec.sets"));
    assert!(err(ObsSpec::new(0, vec![EntitySet::new("a", 0, 2)])).contains("sets[0].count"));
    assert!(err(ObsSpec::new(0, vec![EntitySet::new("a", 2, 0)])).contains("sets[0].features"));
    assert!(err(ObsSpec::new(0, vec![EntitySet::new("", 2, 2)])).contains("sets[0].name"));
    assert!(err(ObsSpec::new(0, vec![EntitySet::new("a.b", 2, 2)])).contains("'.'"));
    assert!(
        err(ObsSpec::new(
            0,
            vec![EntitySet::new("a", 1, 1), EntitySet::new("a", 1, 1)]
        ))
        .contains("used twice")
    );
}

#[test]
fn the_policy_config_refuses_inconsistent_structure() {
    let spec = two_set_spec();
    let base = || Mamba3PolicyConfig::new(spec.obs_dim(), 3, 16, 1);
    let err = |c: Mamba3PolicyConfig| c.validate().unwrap_err().to_string();

    assert!(
        err(Mamba3PolicyConfig::new(spec.obs_dim() + 1, 3, 16, 1).with_obs_spec(spec.clone()))
            .contains("obs_dim")
    );
    assert!(
        err(base()
            .with_obs_spec(spec.clone())
            .with_entity_encoder("cards", EntityEncoderConfig::default()))
        .contains("\"cards\"")
    );
    assert!(
        err(base()
            .with_obs_spec(spec.clone())
            .with_action_head(ActionHeadConfig::Pointer(PointerHeadConfig::new("units"))))
        .contains("action_dim is 3")
    );
    assert!(
        err(base()
            .with_obs_spec(spec.clone())
            .with_action_head(ActionHeadConfig::Pointer(PointerHeadConfig::new("cards"))))
        .contains("\"cards\"")
    );
    assert!(
        err(base().with_entity_encoder("tiles", EntityEncoderConfig::default()))
            .contains("no obs_spec")
    );
    assert!(
        err(base().with_action_head(ActionHeadConfig::Pointer(PointerHeadConfig::new("tiles"))))
            .contains("no obs_spec")
    );
    // A pointer into three tiles is three actions; with one extra it is four.
    base()
        .with_obs_spec(spec.clone())
        .with_action_head(ActionHeadConfig::Pointer(PointerHeadConfig::new("tiles")))
        .validate()
        .unwrap();
    Mamba3PolicyConfig::new(spec.obs_dim(), 4, 16, 1)
        .with_obs_spec(spec)
        .with_action_head(ActionHeadConfig::Pointer(
            PointerHeadConfig::new("tiles").with_extra_actions(1),
        ))
        .validate()
        .unwrap();
}

// ---------------------------------------------------------------------------
// E2: the entity encoder and masked pooling
// ---------------------------------------------------------------------------

const N: usize = 4;
const F: usize = 3;
const D: usize = 5;

fn encoder(slot: bool) -> mamba3::nn::entity::EntityEncoder<R, f32> {
    EntityEncoderConfig::new(vec![6], D)
        .with_slot_embedding(slot)
        .init(F, N, &dev(), &mut Rng::seeded(3))
        .unwrap()
}

#[test]
fn encoder_and_pooling_gradients_match_finite_differences() {
    let rows = 2;
    let p = presence(rows, N, &[(0, 1), (1, 3)]);
    let pv = constant(&p, vec![1, rows, N, 1]);
    let enc = encoder(false);
    let weights = noise(rows * 2 * D, 17);
    let w = constant(&weights, vec![1, rows, 2 * D]);
    check_grad(
        "encoder+mean+max",
        &noise(rows * N * F, 9),
        vec![1, rows, N, F],
        |x| {
            let e = enc.apply(x, &pv).unwrap();
            masked_pool(&e, &pv, &[PoolKind::Mean, PoolKind::Max])
                .unwrap()
                .mul(&w)
                .unwrap()
                .sum()
                .unwrap()
        },
    );
    // Through the embeddings directly, so the pools' own adjoints are checked
    // without an encoder smoothing them.
    check_grad("pools", &noise(rows * N * D, 4), vec![1, rows, N, D], |e| {
        masked_pool(e, &pv, &[PoolKind::Mean, PoolKind::Max])
            .unwrap()
            .mul(&w)
            .unwrap()
            .sum()
            .unwrap()
    });
}

#[test]
fn pooling_is_permutation_invariant() {
    let rows = 3;
    let enc = encoder(false);
    let features = noise(rows * N * F, 2);
    let p = presence(rows, N, &[(0, 0), (2, 2), (2, 3)]);
    let pooled = |f: &[f32], p: &[f32]| {
        let pv = constant(p, vec![rows, 1, N, 1]);
        let e = enc.apply(&constant(f, vec![rows, 1, N, F]), &pv).unwrap();
        masked_pool(&e, &pv, &[PoolKind::Mean, PoolKind::Max])
            .unwrap()
            .to_f32()
    };
    let perm = [2, 0, 3, 1];
    assert_close(
        &pooled(
            &permute_rows(&features, rows, N, F, &perm),
            &permute_rows(&p, rows, N, 1, &perm),
        ),
        &pooled(&features, &p),
        1e-6,
        "pooled after a permutation",
    );
}

#[test]
fn what_an_empty_slot_holds_never_reaches_the_output() {
    let rows = 2;
    let enc = encoder(false);
    let p = presence(rows, N, &[(0, 1), (1, 0), (1, 2)]);
    let pv = constant(&p, vec![rows, 1, N, 1]);
    let pooled = |f: &[f32]| {
        let e = enc.apply(&constant(f, vec![rows, 1, N, F]), &pv).unwrap();
        masked_pool(&e, &pv, &[PoolKind::Mean, PoolKind::Max])
            .unwrap()
            .to_f32()
    };
    let features = noise(rows * N * F, 8);
    let mut garbage = features.clone();
    for (row, slot) in [(0, 1), (1, 0), (1, 2)] {
        for k in 0..F {
            garbage[(row * N + slot) * F + k] = 1e6 * (k as f32 + 1.0);
        }
    }
    assert_eq!(pooled(&garbage), pooled(&features));
}

#[test]
fn an_empty_set_pools_to_zero_and_a_slot_embedding_is_per_slot() {
    let enc = encoder(true);
    let pv = constant(&[0.0; N], vec![1, 1, N, 1]);
    let e = enc
        .apply(&constant(&noise(N * F, 1), vec![1, 1, N, F]), &pv)
        .unwrap();
    let parts = pool_parts(
        &e,
        &Presence::new(&pv).unwrap(),
        &[PoolKind::Mean, PoolKind::Max],
    )
    .unwrap();
    for part in parts {
        assert!(
            part.to_f32().iter().all(|v| *v == 0.0),
            "{:?}",
            part.to_f32()
        );
    }
    let names: Vec<String> = enc.named_parameters().into_iter().map(|(n, _)| n).collect();
    assert_eq!(
        names,
        [
            "mlp.0.weight",
            "mlp.0.bias",
            "mlp.1.weight",
            "mlp.1.bias",
            "slot"
        ]
    );
}

// ---------------------------------------------------------------------------
// E3 and E4: the pointer head, inside a policy
// ---------------------------------------------------------------------------

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
                    let p = (0..p.len()).map(|i| (i != empty) as u8 as f32).collect();
                    (f, p)
                })
                .collect();
            let refs: Vec<(&[f32], &[f32])> = sets.iter().map(|(f, p)| (&f[..], &p[..])).collect();
            out.extend(self.spec.pack(&g, &refs).unwrap());
        }
        out
    }
}

#[test]
fn parameter_paths_follow_the_structure_and_the_flat_policy_keeps_its_own() {
    let case = Structured::new(1);
    let names: Vec<String> = case
        .policy(Scoring::Additive, 1)
        .named_parameters()
        .into_iter()
        .map(|(n, _)| n)
        .collect();
    for want in [
        "entity.tiles.mlp.0.weight",
        "entity.tiles.mlp.1.bias",
        "entity.units.mlp.0.weight",
        "entity.units.slot",
        "pool.proj.weight",
        "actor.pointer.w_h.weight",
        "actor.pointer.w_h.bias",
        "actor.pointer.w_e.weight",
        "actor.pointer.v.weight",
        "actor.extra.weight",
        "critic.weight",
        "blocks.1.mixer.in_proj.weight",
    ] {
        assert!(
            names.iter().any(|n| n == want),
            "missing {want} in {names:?}"
        );
    }
    assert!(!names.iter().any(|n| n.starts_with("encoder.")));

    let flat: Vec<String> = Mamba3PolicyConfig::new(6, 3, 16, 1)
        .init::<R, f32>(&dev())
        .unwrap()
        .named_parameters()
        .into_iter()
        .map(|(n, _)| n)
        .collect();
    assert!(flat.iter().any(|n| n == "encoder.weight"));
    assert!(flat.iter().any(|n| n == "actor.weight"));
    assert!(
        !flat
            .iter()
            .any(|n| n.starts_with("entity.") || n.starts_with("pool."))
    );
}

#[test]
fn the_pointer_is_permutation_equivariant_and_gives_empty_slots_nothing() {
    for scoring in [Scoring::Additive, Scoring::Dot] {
        let case = Structured::new(1);
        let policy = case.policy(scoring, 1);
        let (envs, dim) = (3, case.spec.obs_dim());
        let obs = case.observations(envs, 1, 21);
        let logits = |o: &[f32]| {
            policy
                .forward(&constant(o, vec![envs, 1, dim]), None, None)
                .unwrap()
                .0
                .logits
                .to_f32()
        };
        let base = logits(&obs);

        // Permute the tiles of every environment, flags included.
        let perm = [3, 1, 0, 2];
        let tiles_at = case.spec.offsets()[0];
        let mut permuted = obs.clone();
        for row in permuted.chunks_exact_mut(dim) {
            let band = &mut row[tiles_at..tiles_at + N * (F + 1)];
            let moved = permute_rows(band, 1, N, F + 1, &perm);
            band.copy_from_slice(&moved);
        }
        let moved = logits(&permuted);
        for env in 0..envs {
            let a = &base[env * case.actions..][..case.actions];
            let b = &moved[env * case.actions..][..case.actions];
            for (dst, &src) in perm.iter().enumerate() {
                assert!(
                    (b[dst] - a[src]).abs() <= 1e-5 * (1.0 + a[src].abs()),
                    "{scoring:?}: env {env} slot {dst} got {}, want {}",
                    b[dst],
                    a[src]
                );
            }
            // The extra action does not move with the tiles.
            assert!((b[N] - a[N]).abs() <= 1e-5 * (1.0 + a[N].abs()));
            // The empty tile slot (see `observations`) is masked to exactly zero
            // probability.
            let empty = env % N;
            assert_eq!(a[empty], f32::MIN, "{scoring:?}: empty slot not masked");
            let max = a.iter().cloned().fold(f32::MIN, f32::max);
            assert_eq!((a[empty] - max).exp(), 0.0);
        }
    }
}

#[test]
fn a_structured_rollout_matches_the_parallel_scan_across_resets() {
    let case = Structured::new(0);
    let policy = case.policy(Scoring::Additive, 0);
    let (envs, steps, dim) = (3, 7, case.spec.obs_dim());
    let raw = case.observations(envs, steps, 4);
    let obs = constant(&raw, vec![envs, steps, dim]);
    let mut mask = vec![0.0f32; envs * steps];
    mask[2] = 1.0; // env 0, step 2
    mask[steps + 5] = 1.0; // env 1, step 5

    let scan = policy
        .forward(
            &obs,
            Some(&Tensor::from_f32(&mask, vec![envs, steps], &dev()).unwrap()),
            None,
        )
        .unwrap()
        .0
        .logits
        .to_f32();

    let mut engine = RolloutEngine::new(&policy, envs, &dev());
    let mut rolled = Vec::new();
    for t in 0..steps {
        let flags: Vec<f32> = (0..envs).map(|e| mask[e * steps + t]).collect();
        let flags = Tensor::from_f32(&flags, vec![envs], &dev()).unwrap();
        let out = engine
            .step(
                &obs.slice(1, t, 1)
                    .unwrap()
                    .reshape(vec![envs, 1, dim])
                    .unwrap(),
                Some(&flags),
            )
            .unwrap();
        rolled.push(out.logits);
    }
    let rolled = mamba3::autograd::cat(&rolled, 1).unwrap().to_f32();
    // Masked slots are f32::MIN on both sides; compare everything else.
    assert_eq!(rolled.len(), scan.len());
    for (i, (a, b)) in rolled.iter().zip(&scan).enumerate() {
        if *b == f32::MIN {
            assert_eq!(*a, f32::MIN, "index {i}");
        } else {
            assert!(
                (a - b).abs() <= 1e-4 * (1.0 + b.abs()),
                "index {i}: {a} vs {b}"
            );
        }
    }
}

#[test]
fn a_structured_trajectory_pass_reaches_every_parameter() {
    for scoring in [Scoring::Additive, Scoring::Dot] {
        let case = Structured::new(1);
        let policy = case.policy(scoring, 1);
        let (envs, steps, dim) = (2, 5, case.spec.obs_dim());
        let obs = constant(&case.observations(envs, steps, 6), vec![envs, steps, dim]);
        let (out, _) = policy.forward(&obs, None, None).unwrap();
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
        for (name, param) in policy.named_parameters() {
            let g = grads
                .get(param.id())
                .unwrap_or_else(|| panic!("{scoring:?}: no gradient for {name}"));
            let g = g.to_f32();
            assert!(
                g.iter().all(|v| v.is_finite()),
                "{scoring:?}: {name} not finite"
            );
            assert!(
                g.iter().any(|v| *v != 0.0),
                "{scoring:?}: {name} is all zero"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// The reason for all of it: "pick the entity with the largest feature"
// ---------------------------------------------------------------------------

const PICK_N: usize = 100;
const PICK_F: usize = 4;

/// Observations of `envs × steps` sets of `PICK_N` entities, and the index of
/// the present entity whose feature 3 is largest.
fn pick_batch(spec: &ObsSpec, envs: usize, steps: usize, seed: u64) -> (Vec<f32>, Vec<u32>) {
    let rows = envs * steps;
    let features = noise(rows * PICK_N * PICK_F, seed);
    let coin = noise(rows * PICK_N, seed ^ 0x5eed);
    let mut obs = Vec::with_capacity(rows * spec.obs_dim());
    let mut labels = Vec::with_capacity(rows);
    for r in 0..rows {
        let f = &features[r * PICK_N * PICK_F..][..PICK_N * PICK_F];
        // About a quarter of the slots are empty, never all of them.
        let p: Vec<f32> = (0..PICK_N)
            .map(|i| (i == 0 || coin[r * PICK_N + i] > -0.5) as u8 as f32)
            .collect();
        let best = (0..PICK_N)
            .filter(|&i| p[i] == 1.0)
            .max_by(|&a, &b| f[a * PICK_F + 3].total_cmp(&f[b * PICK_F + 3]))
            .unwrap();
        obs.extend(spec.pack(&[], &[(f, &p)]).unwrap());
        labels.push(best as u32);
    }
    (obs, labels)
}

fn pick_agreement(config: Mamba3PolicyConfig, rounds: usize) -> f32 {
    let spec = ObsSpec::new(0, vec![EntitySet::new("items", PICK_N, PICK_F)]);
    let policy = config.init::<R, f32>(&dev()).unwrap();
    let task = BehaviourCloningTask::new(&policy);
    let lr = 3e-3;
    let mut trainer = Trainer::new(
        TrainerConfig::builder()
            .learning_rate(lr)
            .max_grad_norm(1.0)
            .build()
            .unwrap(),
        AdamWConfig::builder()
            .learning_rate(lr)
            .build()
            .init::<R, f32>(),
    );
    let (envs, steps) = (32, 2);
    let batch = |seed: u64| {
        let (obs, labels) = pick_batch(&spec, envs, steps, seed);
        ImitationBatch {
            observations: Tensor::from_f32(&obs, vec![envs, steps, spec.obs_dim()], &dev())
                .unwrap(),
            expert_actions: IdTensor::from_slice(&labels, vec![envs, steps], &dev()).unwrap(),
            reset: None,
            initial: None,
            mask: None,
            action_mask: None,
        }
    };
    let held_out = batch(999_999);
    let mut agreement = task.agreement(&held_out).unwrap();
    for round in 0..rounds {
        trainer.step(&task, &[batch(round as u64)]).unwrap();
        if round % 25 == 24 {
            agreement = task.agreement(&held_out).unwrap();
            if agreement >= 0.99 {
                break;
            }
        }
    }
    agreement
}

fn pick_config(obs_dim: usize) -> Mamba3PolicyConfig {
    Mamba3PolicyConfig::new(obs_dim, PICK_N, 32, 1)
        .with_seed(1)
        .with_ssm(|s| {
            s.n_heads = 2;
            s.head_dim = 16;
            s.n_groups = 2;
            s.d_state = 8;
            s.chunk_size = 8;
        })
}

#[test]
fn a_pointer_learns_to_pick_the_largest_entity_where_a_flat_head_cannot() {
    let spec = ObsSpec::new(0, vec![EntitySet::new("items", PICK_N, PICK_F)]);
    let pointer = pick_agreement(
        pick_config(spec.obs_dim())
            .with_obs_spec(spec.clone())
            .with_entity_encoder("items", EntityEncoderConfig::new(vec![32], 32))
            .with_pooling(PoolingConfig {
                kinds: vec![PoolKind::Mean, PoolKind::Max],
            })
            .with_action_head(ActionHeadConfig::Pointer(
                PointerHeadConfig::new("items").with_hidden(32),
            )),
        200,
    );
    assert!(pointer >= 0.99, "the pointer reached only {pointer:.3}");

    let flat = pick_agreement(pick_config(spec.obs_dim()), 200);
    assert!(
        flat < 0.5,
        "the flat head reached {flat:.3}, so this task no longer shows the difference"
    );
}
