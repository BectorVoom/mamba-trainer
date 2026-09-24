//! K1+K2: the fused `entity_prepare` / `entity_pool` kernels against the composed
//! paths they replace.
//!
//! A separate binary from `rl_entity` on purpose: the fused switch is
//! process-global, so a test that flips it here must not run beside tests that
//! depend on it there. These tests call the kernel directly and never touch the
//! switch.

#![cfg(feature = "backend")]

use mamba3::autograd::Var;
use mamba3::autograd::ops::EntityPoolInput;
use mamba3::backend::Device;
use mamba3::backends::Auto;
use mamba3::nn::entity::{PoolKind, Presence, pool_parts};
use mamba3::rl::{EntitySet, ObsSpec};
use mamba3::tensor::{Shape, Tensor};

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

fn assert_relative(actual: &[f32], expected: &[f32], tol: f32, what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= tol * (1.0 + e.abs()),
            "{what}: index {i} got {a}, want {e}"
        );
    }
}

fn two_set_spec() -> ObsSpec {
    ObsSpec::new(
        2,
        vec![EntitySet::new("a", 3, 2), EntitySet::new("b", 2, 1)],
    )
}

/// `[rows, obs_dim]` observations: row 1 is all-absent, row 2 carries a 0.5
/// presence in set `a` slot 1, the rest are present with random features.
fn observations(spec: &ObsSpec, rows: usize, seed: u64) -> Vec<f32> {
    let offsets = spec.offsets();
    let mut out = Vec::with_capacity(rows * spec.obs_dim());
    for r in 0..rows {
        let globals = noise(spec.globals, seed + r as u64 * 131);
        let fa = noise(spec.sets[0].count * spec.sets[0].features, seed + 1000 + r as u64);
        let fb = noise(spec.sets[1].count * spec.sets[1].features, seed + 2000 + r as u64);
        let (mut pa, mut pb): (Vec<f32>, Vec<f32>) = (
            vec![1.0; spec.sets[0].count],
            vec![1.0; spec.sets[1].count],
        );
        if r == 1 {
            pa.fill(0.0);
            pb.fill(0.0);
        }
        if r == 2 {
            pa[1] = 0.5;
        }
        // Keep the offsets used so a mis-packed row cannot hide a mis-read one.
        assert_eq!(offsets, spec.offsets());
        out.extend(spec.pack(&globals, &[(&fa, &pa), (&fb, &pb)]).unwrap());
    }
    out
}

#[test]
fn prepare_matches_the_composed_chain() {
    let spec = two_set_spec();
    let rows = 4usize;
    let raw = observations(&spec, rows, 11);
    let device = dev();
    let obs = Tensor::<R, f32>::from_f32(&raw, vec![rows, spec.obs_dim()], &device).unwrap();
    let offsets = spec.offsets();

    // The composed reference runs through the public path: rank-3 split, the
    // broadcast multiply, and `Presence::new`.
    let obs3 = V::constant(obs.reshape(vec![1, rows, spec.obs_dim()]).unwrap());
    let split = spec.split(&obs3).unwrap();

    for (k, set) in spec.sets.iter().enumerate() {
        let (f, mean_w, legal, any) =
            mamba3::tensor::ops::entity::entity_prepare(&obs, offsets[k], set.count, set.features)
                .unwrap();
        assert_eq!(f.dims(), &[rows, set.count, set.features]);
        assert_eq!(mean_w.dims(), &[rows, set.count]);
        assert_eq!(legal.dims(), &[rows, set.count]);
        assert_eq!(any.dims(), &[rows]);

        let zeroed = split.sets[k]
            .features
            .mul(&split.sets[k].presence)
            .unwrap()
            .to_f32();
        assert_relative(&f.to_f32(), &zeroed, 1e-6, &format!("set {} features", set.name));

        let presence = Presence::new(&split.sets[k].presence).unwrap();
        let want_legal = presence.flags.reshape(vec![rows, set.count]).unwrap().to_f32();
        assert_relative(&legal.to_f32(), &want_legal, 1e-6, &format!("set {} legal", set.name));
        let want_mean = presence
            .mean_weights
            .reshape(vec![rows, set.count])
            .unwrap()
            .to_f32();
        assert_relative(
            &mean_w.to_f32(),
            &want_mean,
            1e-6,
            &format!("set {} mean_w", set.name),
        );
        let want_any = presence.any.reshape(vec![rows]).unwrap().to_f32();
        assert_relative(&any.to_f32(), &want_any, 1e-6, &format!("set {} any", set.name));
    }

    // The all-absent row pools to nothing: mean weights all 0, any 0.
    for (k, set) in spec.sets.iter().enumerate() {
        let (_, mean_w, _, any) =
            mamba3::tensor::ops::entity::entity_prepare(&obs, offsets[k], set.count, set.features)
                .unwrap();
        let mw = mean_w.to_f32();
        for n in 0..set.count {
            assert_eq!(mw[1 * set.count + n], 0.0, "set {} slot {n}", set.name);
        }
        if k == 0 {
            assert_eq!(any.to_f32()[1], 0.0);
        }
    }
}

#[test]
fn prepare_gradient_matches_finite_differences() {
    // One set with globals, so the offsets and the untouched columns are both
    // exercised. The adjoint treats presence as a constant — zero gradient at
    // the presence column, globals and other sets' columns — so central
    // differences apply at the set's feature columns, and the rest must be
    // exactly zero.
    let spec = ObsSpec::new(1, vec![EntitySet::new("a", 2, 2)]);
    let rows = 3usize;
    let full = ObsSpec::new(1, vec![EntitySet::new("a", 2, 2), EntitySet::new("b", 1, 1)]);
    let raw = observations(&full, rows, 23);
    // Re-pack down to the single-set spec: keep globals and set `a` only.
    let mut obs = Vec::with_capacity(rows * spec.obs_dim());
    for row in raw.chunks_exact(full.obs_dim()) {
        let (g, sets) = full.unpack(row).unwrap();
        obs.extend(spec.pack(&g, &[(&sets[0].0, &sets[0].1)]).unwrap());
    }
    let off = spec.offsets()[0];
    let (count, feats) = (2usize, 2usize);
    let w = noise(rows * count * feats, 7);
    let w_tensor = Tensor::<R, f32>::from_f32(&w, vec![rows, count, feats], &dev()).unwrap();
    let f = |x: &V| {
        let (feat, _, _, _) = V::entity_prepare(x, off, count, feats).unwrap();
        feat.mul(&V::constant(w_tensor.clone()))
            .unwrap()
            .sum()
            .unwrap()
    };

    let x = V::traced(Tensor::from_f32(&obs, vec![rows, spec.obs_dim()], &dev()).unwrap());
    let grads = f(&x).backward_retain().unwrap();
    let analytic = grads.node(x.node().unwrap()).unwrap().to_f32();

    // Feature columns of this set's band: central differences.
    let stride = feats + 1;
    let mut feature_cols = Vec::new();
    for n in 0..count {
        for k in 0..feats {
            feature_cols.push(off + n * stride + k);
        }
    }
    let shape = Shape::new(vec![rows, spec.obs_dim()]);
    let eps = 1e-3f32;
    for r in 0..rows {
        for c in 0..spec.obs_dim() {
            let i = r * spec.obs_dim() + c;
            if feature_cols.contains(&c) {
                let at = |delta: f32| {
                    let mut moved = obs.clone();
                    moved[i] += delta;
                    f(&V::constant(
                        Tensor::from_f32(&moved, shape.clone(), &dev()).unwrap(),
                    ))
                    .to_f32()[0]
                };
                let numeric = (at(eps) - at(-eps)) / (2.0 * eps);
                assert!(
                    (analytic[i] - numeric).abs() < 2e-2 * (1.0 + numeric.abs()),
                    "grad[{i}] analytic={} numeric={numeric}",
                    analytic[i]
                );
            } else {
                // Presence is a constant by construction: exactly zero.
                assert_eq!(analytic[i], 0.0, "grad[{i}] should be exactly 0");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// K2: entity_pool / entity_join
// ---------------------------------------------------------------------------

/// Two sets with different pool shapes: `a` (`N=3`) pools max only, `b`
/// (`N=1`) pools mean then max. Different embedding widths catch offset bugs.
fn join_spec() -> ObsSpec {
    ObsSpec::new(
        2,
        vec![EntitySet::new("a", 3, 2), EntitySet::new("b", 1, 1)],
    )
}

fn join_kinds() -> (Vec<PoolKind>, Vec<PoolKind>) {
    (vec![PoolKind::Max], vec![PoolKind::Mean, PoolKind::Max])
}

fn prepared_constants(
    obs: &Tensor<R, f32>,
    spec: &ObsSpec,
) -> Vec<(Tensor<R, f32>, Tensor<R, f32>, Tensor<R, f32>)> {
    let offsets = spec.offsets();
    spec.sets
        .iter()
        .enumerate()
        .map(|(k, set)| {
            let (_, mean_w, legal, any) =
                mamba3::tensor::ops::entity::entity_prepare(obs, offsets[k], set.count, set.features)
                    .unwrap();
            (mean_w, legal, any)
        })
        .collect()
}

fn presence_flat(
    mean_w: &Tensor<R, f32>,
    legal: &Tensor<R, f32>,
    any: &Tensor<R, f32>,
) -> Presence<R, f32> {
    let rows = legal.dims()[0];
    let count = legal.dims()[1];
    Presence {
        flags: legal.reshape(vec![rows, count, 1]).unwrap(),
        mean_weights: mean_w.reshape(vec![rows, 1, count]).unwrap(),
        any: any.reshape(vec![rows, 1, 1]).unwrap(),
    }
}

/// `[rows, obs_dim]` observations with 0/1 presence only: row 1 is all-absent,
/// every other slot is present. Finite differences check the fused adjoint
/// against the fused forward, whose max value does not depend on the presence
/// magnitude — so the gradient there must be binary too, and 0.5 data would
/// conflate the check with the composed `mask_logits` 0.5-weighting the fused
/// path deliberately preserves (see the backward kernel docs). The forward
/// and adjoint-equality tests above cover 0.5 against the composed path.
fn observations_binary(spec: &ObsSpec, rows: usize, seed: u64) -> Vec<f32> {
    let mut out = Vec::with_capacity(rows * spec.obs_dim());
    for r in 0..rows {
        let globals = noise(spec.globals, seed + r as u64 * 131);
        let mut sets = Vec::with_capacity(spec.sets.len());
        for (k, set) in spec.sets.iter().enumerate() {
            let f = noise(set.count * set.features, seed + 1000 + 2000 * k as u64 + r as u64);
            let mut p = vec![1.0f32; set.count];
            if r == 1 {
                p.fill(0.0);
            }
            sets.push((f, p));
        }
        let refs: Vec<(&[f32], &[f32])> = sets.iter().map(|(f, p)| (&f[..], &p[..])).collect();
        out.extend(spec.pack(&globals, &refs).unwrap());
    }
    out
}

#[test]
fn pool_join_matches_composed_pools_and_globals() {
    let spec = join_spec();
    let (kinds_a, kinds_b) = join_kinds();
    let (da, db) = (3usize, 2usize);
    let rows = 4usize;
    let raw = observations(&spec, rows, 11);
    let device = dev();
    let obs = Tensor::<R, f32>::from_f32(&raw, vec![rows, spec.obs_dim()], &device).unwrap();
    let consts = prepared_constants(&obs, &spec);

    // Tie-free random embeddings; set `b` has N=1.
    let ea_data = noise(rows * spec.sets[0].count * da, 101);
    let eb_data = noise(rows * spec.sets[1].count * db, 202);
    let ea = Tensor::<R, f32>::from_f32(&ea_data, vec![rows, spec.sets[0].count, da], &device).unwrap();
    let eb = Tensor::<R, f32>::from_f32(&eb_data, vec![rows, spec.sets[1].count, db], &device).unwrap();

    let obs_var = V::constant(obs.clone());
    let ea_var = V::constant(ea.clone());
    let eb_var = V::constant(eb.clone());
    let inputs = [
        EntityPoolInput {
            embeddings: &ea_var,
            mean_w: consts[0].0.clone(),
            legal: consts[0].1.clone(),
            any: consts[0].2.clone(),
            kinds: kinds_a.clone(),
        },
        EntityPoolInput {
            embeddings: &eb_var,
            mean_w: consts[1].0.clone(),
            legal: consts[1].1.clone(),
            any: consts[1].2.clone(),
            kinds: kinds_b.clone(),
        },
    ];
    let joined = V::entity_join(&obs_var, spec.globals, &inputs)
        .unwrap()
        .to_f32();
    let width = spec.globals + kinds_a.len() * da + kinds_b.len() * db;
    assert_eq!(joined.len(), rows * width);

    // Composed reference: globals slice plus pool_parts per set, same order.
    let globals = obs_var.slice(1, 0, spec.globals).unwrap();
    let pa = presence_flat(&consts[0].0, &consts[0].1, &consts[0].2);
    let pb = presence_flat(&consts[1].0, &consts[1].1, &consts[1].2);
    let mut parts = vec![globals];
    parts.extend(pool_parts(&ea_var, &pa, &kinds_a).unwrap());
    parts.extend(pool_parts(&eb_var, &pb, &kinds_b).unwrap());
    let expected = mamba3::autograd::cat(&parts, 1).unwrap().to_f32();
    assert_relative(&joined, &expected, 1e-6, "join forward");

    // The all-absent row (row 1) pools to zero past the globals.
    for c in spec.globals..width {
        assert_eq!(joined[1 * width + c], 0.0, "absent row col {c}");
    }
    // Globals are a verbatim copy of obs's first columns.
    let obs_raw = obs.to_f32();
    let obs_dim = spec.obs_dim();
    for r in 0..rows {
        for c in 0..spec.globals {
            assert_eq!(joined[r * width + c], obs_raw[r * obs_dim + c]);
        }
    }
}

#[test]
fn pool_join_gradient_wrt_embeddings_matches_finite_differences() {
    let spec = join_spec();
    let (kinds_a, kinds_b) = join_kinds();
    let (da, db) = (3usize, 2usize);
    let rows = 3usize;
    let raw = observations_binary(&spec, rows, 31);
    let device = dev();
    let obs = Tensor::<R, f32>::from_f32(&raw, vec![rows, spec.obs_dim()], &device).unwrap();
    let consts = prepared_constants(&obs, &spec);
    let width = spec.globals + kinds_a.len() * da + kinds_b.len() * db;
    let w_data = noise(rows * width, 77);
    let w = Tensor::<R, f32>::from_f32(&w_data, vec![rows, width], &device).unwrap();

    let ea_data = noise(rows * spec.sets[0].count * da, 103);
    let eb_data = noise(rows * spec.sets[1].count * db, 204);
    let obs_var = V::constant(obs);
    let eb_var = V::constant(
        Tensor::<R, f32>::from_f32(&eb_data, vec![rows, spec.sets[1].count, db], &device).unwrap(),
    );
    let f = |e: &V| {
        let inputs = [
            EntityPoolInput {
                embeddings: e,
                mean_w: consts[0].0.clone(),
                legal: consts[0].1.clone(),
                any: consts[0].2.clone(),
                kinds: kinds_a.clone(),
            },
            EntityPoolInput {
                embeddings: &eb_var,
                mean_w: consts[1].0.clone(),
                legal: consts[1].1.clone(),
                any: consts[1].2.clone(),
                kinds: kinds_b.clone(),
            },
        ];
        let joined = V::entity_join(&obs_var, spec.globals, &inputs).unwrap();
        joined.mul(&V::constant(w.clone())).unwrap().sum().unwrap()
    };

    let shape = Shape::new(vec![rows, spec.sets[0].count, da]);
    let x = V::traced(
        Tensor::from_f32(&ea_data, shape.clone(), &device).unwrap(),
    );
    let grads = f(&x).backward_retain().unwrap();
    let analytic = grads.node(x.node().unwrap()).unwrap().to_f32();
    let eps = 1e-3f32;
    for i in 0..ea_data.len() {
        let at = |delta: f32| {
            let mut moved = ea_data.clone();
            moved[i] += delta;
            f(&V::constant(
                Tensor::from_f32(&moved, shape.clone(), &device).unwrap(),
            ))
            .to_f32()[0]
        };
        let numeric = (at(eps) - at(-eps)) / (2.0 * eps);
        assert!(
            (analytic[i] - numeric).abs() < 2e-2 * (1.0 + numeric.abs()),
            "e grad[{i}] analytic={} numeric={numeric}",
            analytic[i]
        );
    }
}

#[test]
fn pool_join_gradient_wrt_obs_matches_finite_differences() {
    // The prepare constants stay detached, so a traced obs only reaches the
    // loss through the globals copy: globals columns carry `w`, the rest is 0.
    let spec = join_spec();
    let (kinds_a, kinds_b) = join_kinds();
    let (da, db) = (3usize, 2usize);
    let rows = 3usize;
    let raw = observations(&spec, rows, 41);
    let device = dev();
    let obs = Tensor::<R, f32>::from_f32(&raw, vec![rows, spec.obs_dim()], &device).unwrap();
    let consts = prepared_constants(&obs, &spec);
    let width = spec.globals + kinds_a.len() * da + kinds_b.len() * db;
    let w_data = noise(rows * width, 79);
    let w = Tensor::<R, f32>::from_f32(&w_data, vec![rows, width], &device).unwrap();

    let ea = Tensor::<R, f32>::from_f32(
        &noise(rows * spec.sets[0].count * da, 107),
        vec![rows, spec.sets[0].count, da],
        &device,
    )
    .unwrap();
    let eb = Tensor::<R, f32>::from_f32(
        &noise(rows * spec.sets[1].count * db, 208),
        vec![rows, spec.sets[1].count, db],
        &device,
    )
    .unwrap();
    let ea_var = V::constant(ea);
    let eb_var = V::constant(eb);
    let f = |o: &V| {
        let inputs = [
            EntityPoolInput {
                embeddings: &ea_var,
                mean_w: consts[0].0.clone(),
                legal: consts[0].1.clone(),
                any: consts[0].2.clone(),
                kinds: kinds_a.clone(),
            },
            EntityPoolInput {
                embeddings: &eb_var,
                mean_w: consts[1].0.clone(),
                legal: consts[1].1.clone(),
                any: consts[1].2.clone(),
                kinds: kinds_b.clone(),
            },
        ];
        let joined = V::entity_join(o, spec.globals, &inputs).unwrap();
        joined.mul(&V::constant(w.clone())).unwrap().sum().unwrap()
    };

    let shape = Shape::new(vec![rows, spec.obs_dim()]);
    let x = V::traced(Tensor::from_f32(&raw, shape.clone(), &device).unwrap());
    let grads = f(&x).backward_retain().unwrap();
    let analytic = grads.node(x.node().unwrap()).unwrap().to_f32();
    let eps = 1e-3f32;
    for i in 0..raw.len() {
        let at = |delta: f32| {
            let mut moved = raw.clone();
            moved[i] += delta;
            f(&V::constant(
                Tensor::from_f32(&moved, shape.clone(), &device).unwrap(),
            ))
            .to_f32()[0]
        };
        let numeric = (at(eps) - at(-eps)) / (2.0 * eps);
        assert!(
            (analytic[i] - numeric).abs() < 2e-2 * (1.0 + numeric.abs()),
            "obs grad[{i}] analytic={} numeric={numeric}",
            analytic[i]
        );
    }
}

#[test]
fn pool_join_adjoint_matches_composed_on_tie_free_data() {
    let spec = join_spec();
    let (kinds_a, kinds_b) = join_kinds();
    let (da, db) = (3usize, 2usize);
    let rows = 4usize;
    let raw = observations(&spec, rows, 53);
    let device = dev();
    let obs = Tensor::<R, f32>::from_f32(&raw, vec![rows, spec.obs_dim()], &device).unwrap();
    let consts = prepared_constants(&obs, &spec);
    let width = spec.globals + kinds_a.len() * da + kinds_b.len() * db;
    let w_data = noise(rows * width, 83);
    let w = Tensor::<R, f32>::from_f32(&w_data, vec![rows, width], &device).unwrap();
    let ea_data = noise(rows * spec.sets[0].count * da, 109);
    let eb_data = noise(rows * spec.sets[1].count * db, 210);

    // Fused gradients.
    let obs_var = V::constant(obs.clone());
    let ea_traced = V::traced(
        Tensor::from_f32(&ea_data, vec![rows, spec.sets[0].count, da], &device).unwrap(),
    );
    let eb_traced = V::traced(
        Tensor::from_f32(&eb_data, vec![rows, spec.sets[1].count, db], &device).unwrap(),
    );
    let fused_inputs = [
        EntityPoolInput {
            embeddings: &ea_traced,
            mean_w: consts[0].0.clone(),
            legal: consts[0].1.clone(),
            any: consts[0].2.clone(),
            kinds: kinds_a.clone(),
        },
        EntityPoolInput {
            embeddings: &eb_traced,
            mean_w: consts[1].0.clone(),
            legal: consts[1].1.clone(),
            any: consts[1].2.clone(),
            kinds: kinds_b.clone(),
        },
    ];
    let fused_loss = V::entity_join(&obs_var, spec.globals, &fused_inputs)
        .unwrap()
        .mul(&V::constant(w.clone()))
        .unwrap()
        .sum()
        .unwrap();
    let fused_grads = fused_loss.backward_retain().unwrap();
    let fused_ea = fused_grads.node(ea_traced.node().unwrap()).unwrap().to_f32();
    let fused_eb = fused_grads.node(eb_traced.node().unwrap()).unwrap().to_f32();

    // Composed gradients on the same data.
    let obs_c = V::constant(obs);
    let ea_c = V::traced(
        Tensor::from_f32(&ea_data, vec![rows, spec.sets[0].count, da], &device).unwrap(),
    );
    let eb_c = V::traced(
        Tensor::from_f32(&eb_data, vec![rows, spec.sets[1].count, db], &device).unwrap(),
    );
    let pa = presence_flat(&consts[0].0, &consts[0].1, &consts[0].2);
    let pb = presence_flat(&consts[1].0, &consts[1].1, &consts[1].2);
    let mut parts = vec![obs_c.slice(1, 0, spec.globals).unwrap()];
    parts.extend(pool_parts(&ea_c, &pa, &kinds_a).unwrap());
    parts.extend(pool_parts(&eb_c, &pb, &kinds_b).unwrap());
    let composed_loss = mamba3::autograd::cat(&parts, 1)
        .unwrap()
        .mul(&V::constant(w))
        .unwrap()
        .sum()
        .unwrap();
    let composed_grads = composed_loss.backward_retain().unwrap();
    let composed_ea = composed_grads.node(ea_c.node().unwrap()).unwrap().to_f32();
    let composed_eb = composed_grads.node(eb_c.node().unwrap()).unwrap().to_f32();

    assert_relative(&fused_ea, &composed_ea, 1e-5, "join adjoint e_a");
    assert_relative(&fused_eb, &composed_eb, 1e-5, "join adjoint e_b");
}

// ---------------------------------------------------------------------------
// K3: pointer_additive / pointer_dot
// ---------------------------------------------------------------------------

const PN: usize = 4;
const PH: usize = 5;
const PK: usize = 2;
const PD: usize = 4;
const PROWS: usize = 3;

/// `[rows, N]` presence: row 1 is all-absent, row 2 carries a 0.5 in slot 1,
/// the rest are present.
fn pointer_legal() -> Vec<f32> {
    let mut out = vec![1.0f32; PROWS * PN];
    for n in 0..PN {
        out[1 * PN + n] = 0.0;
    }
    out[2 * PN + 1] = 0.5;
    out
}

/// `[rows, N]` presence with 0/1 flags only: row 1 is all-absent, the rest
/// are present.
///
/// Finite differences check the fused adjoint against the fused forward,
/// whose masked value does not depend on the presence magnitude — so the true
/// derivative there is binary, while the adjoint deliberately weights by the
/// `legal` value exactly as the composed `mask_logits` rule does (see K2's
/// `observations_binary`). The 0.5 case is covered against the composed path
/// by the forward and adjoint-equality tests.
fn pointer_legal_binary() -> Vec<f32> {
    let mut out = vec![1.0f32; PROWS * PN];
    for n in 0..PN {
        out[1 * PN + n] = 0.0;
    }
    out
}

/// The composed additive reference: broadcast add, ReLU, `v` matmul, mask, extras cat.
fn composed_additive(
    k: &V,
    q: &V,
    v: &V,
    legal: &Tensor<R, f32>,
    extra: &V,
) -> V {
    let pre = k
        .add(&q.unsqueeze(1).unwrap())
        .unwrap()
        .relu();
    let scores = pre
        .matmul(v)
        .unwrap()
        .reshape(vec![PROWS, PN])
        .unwrap();
    let masked = scores.mask_logits(legal).unwrap();
    mamba3::autograd::cat(&[masked, extra.clone()], 1).unwrap()
}

/// The composed dot reference: batched `[.., d, 1]` matmul, mask, extras cat.
fn composed_dot(e: &V, qd: &V, legal: &Tensor<R, f32>, extra: &V) -> V {
    let scores = e
        .matmul(&qd.unsqueeze(2).unwrap())
        .unwrap()
        .reshape(vec![PROWS, PN])
        .unwrap();
    let masked = scores.mask_logits(legal).unwrap();
    mamba3::autograd::cat(&[masked, extra.clone()], 1).unwrap()
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

#[test]
fn pointer_additive_matches_composed() {
    let device = dev();
    // Tie-free random data: continuous, so no exact-zero pre-activation.
    let k_data = noise(PROWS * PN * PH, 1001);
    let q_data = noise(PROWS * PH, 1002);
    let v_data = noise(PH, 1003);
    let extra_data = noise(PROWS * PK, 1004);
    let legal_data = pointer_legal();
    let legal = Tensor::<R, f32>::from_f32(&legal_data, vec![PROWS, PN], &device).unwrap();

    for has_extra in [true, false] {
        let kx = if has_extra { PK } else { 0 };
        let k = V::constant(
            Tensor::from_f32(&k_data, vec![PROWS, PN, PH], &device).unwrap(),
        );
        let q = V::constant(Tensor::from_f32(&q_data, vec![PROWS, PH], &device).unwrap());
        let v = V::constant(Tensor::from_f32(&v_data, vec![PH, 1], &device).unwrap());
        let extra = V::constant(
            Tensor::from_f32(&extra_data, vec![PROWS, PK], &device).unwrap(),
        );
        let extra_ref = if has_extra {
            Some(&extra)
        } else {
            None
        };
        let fused = V::pointer_additive(&k, &q, &v, &legal, extra_ref, PN)
            .unwrap()
            .to_f32();
        assert_eq!(fused.len(), PROWS * (PN + kx));

        let composed = composed_additive(&k, &q, &v, &legal, &extra).to_f32();
        let want = if has_extra {
            composed
        } else {
            composed
                .chunks_exact(PN + PK)
                .flat_map(|row| row[..PN].to_vec())
                .collect()
        };
        assert_logits_close(&fused, &want, 1e-6, "additive forward");

        // The all-absent row (row 1) is masked everywhere.
        for n in 0..PN {
            assert_eq!(fused[1 * (PN + kx) + n], f32::MIN, "absent row slot {n}");
        }
        // Extras ride through verbatim.
        if has_extra {
            for r in 0..PROWS {
                for kk in 0..PK {
                    assert_eq!(
                        fused[r * (PN + PK) + PN + kk],
                        extra_data[r * PK + kk],
                        "extra [{r}, {kk}]"
                    );
                }
            }
        }
    }
}

#[test]
fn pointer_dot_matches_composed() {
    let device = dev();
    let e_data = noise(PROWS * PN * PD, 2001);
    let qd_data = noise(PROWS * PD, 2002);
    let extra_data = noise(PROWS * PK, 2003);
    let legal_data = pointer_legal();
    let legal = Tensor::<R, f32>::from_f32(&legal_data, vec![PROWS, PN], &device).unwrap();

    for has_extra in [true, false] {
        let kx = if has_extra { PK } else { 0 };
        let e = V::constant(
            Tensor::from_f32(&e_data, vec![PROWS, PN, PD], &device).unwrap(),
        );
        let qd = V::constant(Tensor::from_f32(&qd_data, vec![PROWS, PD], &device).unwrap());
        let extra = V::constant(
            Tensor::from_f32(&extra_data, vec![PROWS, PK], &device).unwrap(),
        );
        let extra_ref = if has_extra {
            Some(&extra)
        } else {
            None
        };
        let fused = V::pointer_dot(&e, &qd, &legal, extra_ref, PN)
            .unwrap()
            .to_f32();
        assert_eq!(fused.len(), PROWS * (PN + kx));

        let composed = composed_dot(&e, &qd, &legal, &extra).to_f32();
        let want = if has_extra {
            composed
        } else {
            composed
                .chunks_exact(PN + PK)
                .flat_map(|row| row[..PN].to_vec())
                .collect()
        };
        assert_logits_close(&fused, &want, 1e-6, "dot forward");

        for n in 0..PN {
            assert_eq!(fused[1 * (PN + kx) + n], f32::MIN, "absent row slot {n}");
        }
        if has_extra {
            for r in 0..PROWS {
                for kk in 0..PK {
                    assert_eq!(
                        fused[r * (PN + PK) + PN + kk],
                        extra_data[r * PK + kk],
                        "extra [{r}, {kk}]"
                    );
                }
            }
        }
    }
}

/// Central differences of `sum(w * logits)` with `w` zeroed at masked entity
/// positions, so `f32::MIN` never enters the numeric difference.
fn check_grad_masked(
    inputs: &[Vec<f32>],
    shapes: &[Vec<usize>],
    make_loss: &dyn Fn(Vec<V>) -> V,
    what: &[&str],
) {
    let device = dev();
    let traced: Vec<V> = inputs
        .iter()
        .zip(shapes)
        .map(|(data, shape)| {
            V::traced(Tensor::from_f32(data, shape.clone(), &device).unwrap())
        })
        .collect();
    let loss = make_loss(traced.clone());
    let grads = loss.backward_retain().unwrap();
    let analytic: Vec<Vec<f32>> = traced
        .iter()
        .map(|t| grads.node(t.node().unwrap()).unwrap().to_f32())
        .collect();

    let eps = 1e-3f32;
    for (k, data) in inputs.iter().enumerate() {
        for i in 0..data.len() {
            let at = |delta: f32| {
                let moved: Vec<Vec<f32>> = inputs
                    .iter()
                    .enumerate()
                    .map(|(j, d)| {
                        let mut m = d.clone();
                        if j == k {
                            m[i] += delta;
                        }
                        m
                    })
                    .collect();
                let vars: Vec<V> = moved
                    .iter()
                    .zip(shapes)
                    .map(|(d, s)| {
                        V::constant(Tensor::from_f32(d, s.clone(), &device).unwrap())
                    })
                    .collect();
                make_loss(vars).to_f32()[0]
            };
            let numeric = (at(eps) - at(-eps)) / (2.0 * eps);
            assert!(
                (analytic[k][i] - numeric).abs() < 2e-2 * (1.0 + numeric.abs()),
                "{} grad[{i}] analytic={} numeric={numeric}",
                what[k],
                analytic[k][i]
            );
        }
    }
}

#[test]
fn pointer_additive_gradients_match_finite_differences() {
    let device = dev();
    let k_data = noise(PROWS * PN * PH, 3001);
    let q_data = noise(PROWS * PH, 3002);
    let v_data = noise(PH, 3003);
    let extra_data = noise(PROWS * PK, 3004);
    let legal_data = pointer_legal_binary();
    // Zero weight at masked entity positions; extras keep theirs.
    let mut w_data = noise(PROWS * (PN + PK), 3005);
    for r in 0..PROWS {
        for n in 0..PN {
            if legal_data[r * PN + n] == 0.0 {
                w_data[r * (PN + PK) + n] = 0.0;
            }
        }
    }
    let legal = Tensor::<R, f32>::from_f32(&legal_data, vec![PROWS, PN], &device).unwrap();
    let w = Tensor::<R, f32>::from_f32(&w_data, vec![PROWS, PN + PK], &device).unwrap();
    let make_loss = |vars: Vec<V>| {
        V::pointer_additive(&vars[0], &vars[1], &vars[2], &legal, Some(&vars[3]), PN)
            .unwrap()
            .mul(&V::constant(w.clone()))
            .unwrap()
            .sum()
            .unwrap()
    };
    check_grad_masked(
        &[k_data, q_data, v_data, extra_data],
        &[
            vec![PROWS, PN, PH],
            vec![PROWS, PH],
            vec![PH, 1],
            vec![PROWS, PK],
        ],
        &make_loss,
        &["k", "q", "v", "extra"],
    );
}

#[test]
fn pointer_dot_gradients_match_finite_differences() {
    let device = dev();
    let e_data = noise(PROWS * PN * PD, 4001);
    let qd_data = noise(PROWS * PD, 4002);
    let extra_data = noise(PROWS * PK, 4003);
    let legal_data = pointer_legal_binary();
    let mut w_data = noise(PROWS * (PN + PK), 4004);
    for r in 0..PROWS {
        for n in 0..PN {
            if legal_data[r * PN + n] == 0.0 {
                w_data[r * (PN + PK) + n] = 0.0;
            }
        }
    }
    let legal = Tensor::<R, f32>::from_f32(&legal_data, vec![PROWS, PN], &device).unwrap();
    let w = Tensor::<R, f32>::from_f32(&w_data, vec![PROWS, PN + PK], &device).unwrap();
    let make_loss = |vars: Vec<V>| {
        V::pointer_dot(&vars[0], &vars[1], &legal, Some(&vars[2]), PN)
            .unwrap()
            .mul(&V::constant(w.clone()))
            .unwrap()
            .sum()
            .unwrap()
    };
    check_grad_masked(
        &[e_data, qd_data, extra_data],
        &[
            vec![PROWS, PN, PD],
            vec![PROWS, PD],
            vec![PROWS, PK],
        ],
        &make_loss,
        &["e", "qd", "extra"],
    );
}

#[test]
fn pointer_adjoints_match_composed_on_tie_free_data() {
    let device = dev();
    let k_data = noise(PROWS * PN * PH, 5001);
    let q_data = noise(PROWS * PH, 5002);
    let v_data = noise(PH, 5003);
    let extra_data = noise(PROWS * PK, 5004);
    let e_data = noise(PROWS * PN * PD, 5005);
    let qd_data = noise(PROWS * PD, 5006);
    let legal_data = pointer_legal();
    let legal = Tensor::<R, f32>::from_f32(&legal_data, vec![PROWS, PN], &device).unwrap();
    // Zero weight at masked positions: the comparison is about gradients, and
    // this keeps `f32::MIN` (and any `-inf` sum of it) out of the loss itself.
    let mut w_data = noise(PROWS * (PN + PK), 5007);
    for r in 0..PROWS {
        for n in 0..PN {
            if legal_data[r * PN + n] == 0.0 {
                w_data[r * (PN + PK) + n] = 0.0;
            }
        }
    }
    let w = Tensor::<R, f32>::from_f32(&w_data, vec![PROWS, PN + PK], &device).unwrap();

    // Additive, fused.
    let k_t = V::traced(Tensor::from_f32(&k_data, vec![PROWS, PN, PH], &device).unwrap());
    let q_t = V::traced(Tensor::from_f32(&q_data, vec![PROWS, PH], &device).unwrap());
    let v_t = V::traced(Tensor::from_f32(&v_data, vec![PH, 1], &device).unwrap());
    let x_t = V::traced(Tensor::from_f32(&extra_data, vec![PROWS, PK], &device).unwrap());
    let fused_loss = V::pointer_additive(&k_t, &q_t, &v_t, &legal, Some(&x_t), PN)
        .unwrap()
        .mul(&V::constant(w.clone()))
        .unwrap()
        .sum()
        .unwrap();
    let fused_grads = fused_loss.backward_retain().unwrap();
    let fused = [
        fused_grads.node(k_t.node().unwrap()).unwrap().to_f32(),
        fused_grads.node(q_t.node().unwrap()).unwrap().to_f32(),
        fused_grads.node(v_t.node().unwrap()).unwrap().to_f32(),
        fused_grads.node(x_t.node().unwrap()).unwrap().to_f32(),
    ];

    // Additive, composed, on the same data.
    let k_c = V::traced(Tensor::from_f32(&k_data, vec![PROWS, PN, PH], &device).unwrap());
    let q_c = V::traced(Tensor::from_f32(&q_data, vec![PROWS, PH], &device).unwrap());
    let v_c = V::traced(Tensor::from_f32(&v_data, vec![PH, 1], &device).unwrap());
    let x_c = V::traced(Tensor::from_f32(&extra_data, vec![PROWS, PK], &device).unwrap());
    let composed_loss = composed_additive(&k_c, &q_c, &v_c, &legal, &x_c)
        .mul(&V::constant(w.clone()))
        .unwrap()
        .sum()
        .unwrap();
    let composed_grads = composed_loss.backward_retain().unwrap();
    let composed = [
        composed_grads.node(k_c.node().unwrap()).unwrap().to_f32(),
        composed_grads.node(q_c.node().unwrap()).unwrap().to_f32(),
        composed_grads.node(v_c.node().unwrap()).unwrap().to_f32(),
        composed_grads.node(x_c.node().unwrap()).unwrap().to_f32(),
    ];
    for (i, name) in ["d_k", "d_q", "d_v", "d_extra"].iter().enumerate() {
        assert_relative(&fused[i], &composed[i], 1e-5, &format!("additive {name}"));
    }

    // Dot, fused.
    let e_t = V::traced(Tensor::from_f32(&e_data, vec![PROWS, PN, PD], &device).unwrap());
    let qd_t = V::traced(Tensor::from_f32(&qd_data, vec![PROWS, PD], &device).unwrap());
    let xe_t = V::traced(Tensor::from_f32(&extra_data, vec![PROWS, PK], &device).unwrap());
    let fused_dot = V::pointer_dot(&e_t, &qd_t, &legal, Some(&xe_t), PN)
        .unwrap()
        .mul(&V::constant(w.clone()))
        .unwrap()
        .sum()
        .unwrap();
    let fused_dot_grads = fused_dot.backward_retain().unwrap();
    let fused_d = [
        fused_dot_grads.node(e_t.node().unwrap()).unwrap().to_f32(),
        fused_dot_grads
            .node(qd_t.node().unwrap())
            .unwrap()
            .to_f32(),
        fused_dot_grads
            .node(xe_t.node().unwrap())
            .unwrap()
            .to_f32(),
    ];

    // Dot, composed, on the same data.
    let e_c = V::traced(Tensor::from_f32(&e_data, vec![PROWS, PN, PD], &device).unwrap());
    let qd_c = V::traced(Tensor::from_f32(&qd_data, vec![PROWS, PD], &device).unwrap());
    let xe_c = V::traced(Tensor::from_f32(&extra_data, vec![PROWS, PK], &device).unwrap());
    let composed_dot_loss = composed_dot(&e_c, &qd_c, &legal, &xe_c)
        .mul(&V::constant(w))
        .unwrap()
        .sum()
        .unwrap();
    let composed_dot_grads = composed_dot_loss.backward_retain().unwrap();
    let composed_d = [
        composed_dot_grads
            .node(e_c.node().unwrap())
            .unwrap()
            .to_f32(),
        composed_dot_grads
            .node(qd_c.node().unwrap())
            .unwrap()
            .to_f32(),
        composed_dot_grads
            .node(xe_c.node().unwrap())
            .unwrap()
            .to_f32(),
    ];
    for (i, name) in ["d_e", "d_qd", "d_extra"].iter().enumerate() {
        assert_relative(&fused_d[i], &composed_d[i], 1e-5, &format!("dot {name}"));
    }
}

// ---------------------------------------------------------------------------
// K4: bias_relu — a hidden layer's bias add and ReLU in one launch.
// ---------------------------------------------------------------------------

const BH: usize = 6;
const BLEAD: usize = 4;
const BCOL: usize = 3;

/// Random `[BLEAD, BCOL, BH]` pre-activations and `[BH]` biases, with exact
/// zeros in `pre + bias` at the listed flat positions (bias slot `i % BH`).
fn bias_relu_data(zero_rows: &[usize], seed: u64) -> (Vec<f32>, Vec<f32>) {
    let mut pre = noise(BLEAD * BCOL * BH, seed);
    let bias = noise(BH, seed + 1);
    for &r in zero_rows {
        for k in 0..BH {
            let idx = (r * BCOL) * BH + k;
            pre[idx] = -bias[k];
        }
    }
    (pre, bias)
}

#[test]
fn bias_relu_matches_composed_including_exact_zeros() {
    let device = dev();
    let (pre_data, bias_data) = bias_relu_data(&[0, 2], 6001);
    let pre =
        V::constant(Tensor::from_f32(&pre_data, vec![BLEAD, BCOL, BH], &device).unwrap());
    let bias = V::constant(Tensor::from_f32(&bias_data, vec![BH], &device).unwrap());

    let fused = pre.bias_relu(&bias).unwrap().to_f32();
    let composed = pre.bias_relu_composed(&bias).unwrap().to_f32();
    assert_relative(&fused, &composed, 1e-6, "bias_relu forward");

    // The forced positions hold exact zeros on both sides: `max(0, 0)`.
    for &r in &[0usize, 2usize] {
        for k in 0..BH {
            let idx = (r * BCOL) * BH + k;
            assert_eq!(fused[idx], 0.0, "fused zero at flat {idx}");
            assert_eq!(composed[idx], 0.0, "composed zero at flat {idx}");
        }
    }
}

/// Central differences of `sum(w * y)` against the analytic gradients, for
/// both `pre` and `bias`.
fn check_grad_pair(
    pre_data: &[f32],
    bias_data: &[f32],
    make_loss: &dyn Fn(V, V) -> V,
    what: &str,
) {
    let device = dev();
    let pre_t = V::traced(
        Tensor::from_f32(pre_data, vec![BLEAD, BCOL, BH], &device).unwrap(),
    );
    let bias_t = V::traced(Tensor::from_f32(bias_data, vec![BH], &device).unwrap());
    let loss = make_loss(pre_t.clone(), bias_t.clone());
    let grads = loss.backward_retain().unwrap();
    let analytic = [
        grads.node(pre_t.node().unwrap()).unwrap().to_f32(),
        grads.node(bias_t.node().unwrap()).unwrap().to_f32(),
    ];

    let eps = 1e-3f32;
    let inputs: [&[f32]; 2] = [pre_data, bias_data];
    for (k, data) in inputs.iter().enumerate() {
        for i in 0..data.len() {
            let at = |delta: f32| {
                let mut moved_pre = pre_data.to_vec();
                let mut moved_bias = bias_data.to_vec();
                if k == 0 {
                    moved_pre[i] += delta;
                } else {
                    moved_bias[i] += delta;
                }
                let p = V::constant(
                    Tensor::from_f32(&moved_pre, vec![BLEAD, BCOL, BH], &device).unwrap(),
                );
                let b =
                    V::constant(Tensor::from_f32(&moved_bias, vec![BH], &device).unwrap());
                make_loss(p, b).to_f32()[0]
            };
            let numeric = (at(eps) - at(-eps)) / (2.0 * eps);
            assert!(
                (analytic[k][i] - numeric).abs() < 2e-2 * (1.0 + numeric.abs()),
                "{what} grad[{k}][{i}] analytic={} numeric={numeric}",
                analytic[k][i]
            );
        }
    }
}

#[test]
fn bias_relu_gradients_match_finite_differences() {
    let device = dev();
    // Data that stays ±0.15 away from the kink, so central differences with
    // eps 1e-3 never straddle it.
    let mut pre = noise(BLEAD * BCOL * BH, 6002);
    let bias = noise(BH, 6003);
    for i in 0..pre.len() {
        let s = pre[i] + bias[i % BH];
        if s.abs() < 0.15 {
            pre[i] += 0.3 * if s >= 0.0 { 1.0 } else { -1.0 };
        }
    }
    for i in 0..pre.len() {
        assert!(
            (pre[i] + bias[i % BH]).abs() >= 0.14,
            "kink margin lost at flat {i}"
        );
    }
    let w_data = noise(BLEAD * BCOL * BH, 6004);
    let w = Tensor::<R, f32>::from_f32(&w_data, vec![BLEAD, BCOL, BH], &device).unwrap();
    let make_loss = |p: V, b: V| {
        p.bias_relu(&b)
            .unwrap()
            .mul(&V::constant(w.clone()))
            .unwrap()
            .sum()
            .unwrap()
    };
    check_grad_pair(&pre, &bias, &make_loss, "bias_relu");
}

#[test]
fn bias_relu_adjoint_matches_composed_with_exact_zeros() {
    let device = dev();
    // Exact zeros included, so the `y == 0` convention is checked: the fused
    // gate on the saved output and the composed `relu` gate on the
    // pre-activation must both give exactly zero there.
    let (pre_data, bias_data) = bias_relu_data(&[0, 2], 6005);
    let w_data = noise(BLEAD * BCOL * BH, 6006);
    let w = Tensor::<R, f32>::from_f32(&w_data, vec![BLEAD, BCOL, BH], &device).unwrap();

    let pre_t = V::traced(
        Tensor::from_f32(&pre_data, vec![BLEAD, BCOL, BH], &device).unwrap(),
    );
    let bias_t = V::traced(Tensor::from_f32(&bias_data, vec![BH], &device).unwrap());
    let fused_loss = pre_t
        .bias_relu(&bias_t)
        .unwrap()
        .mul(&V::constant(w.clone()))
        .unwrap()
        .sum()
        .unwrap();
    let fused_grads = fused_loss.backward_retain().unwrap();
    let fused = [
        fused_grads.node(pre_t.node().unwrap()).unwrap().to_f32(),
        fused_grads
            .node(bias_t.node().unwrap())
            .unwrap()
            .to_f32(),
    ];

    let pre_c = V::traced(
        Tensor::from_f32(&pre_data, vec![BLEAD, BCOL, BH], &device).unwrap(),
    );
    let bias_c = V::traced(Tensor::from_f32(&bias_data, vec![BH], &device).unwrap());
    let composed_loss = pre_c
        .bias_relu_composed(&bias_c)
        .unwrap()
        .mul(&V::constant(w))
        .unwrap()
        .sum()
        .unwrap();
    let composed_grads = composed_loss.backward_retain().unwrap();
    let composed = [
        composed_grads.node(pre_c.node().unwrap()).unwrap().to_f32(),
        composed_grads
            .node(bias_c.node().unwrap())
            .unwrap()
            .to_f32(),
    ];
    assert_relative(&fused[0], &composed[0], 1e-5, "bias_relu d_pre");
    assert_relative(&fused[1], &composed[1], 1e-5, "bias_relu d_bias");
}
