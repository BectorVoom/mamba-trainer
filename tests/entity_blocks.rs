//! G2 tests (ENTITY_MODEL_PLAN.md): `Permutation` parity with
//! `transpose_grid`, inverse identity, and the two-set encoder.

#![cfg(feature = "backend")]

use mamba3::autograd::Var;
use mamba3::backend::Device;
use mamba3::backends::Auto;
use mamba3::models::entity::blocks::{Permutation, transpose_grid};
use mamba3::models::entity::model::EntityModel;
use mamba3::models::entity::{ContextSetSpec, EntityModelSpec, HeadSpec, QuerySetSpec, SetLayout};
use mamba3::tensor::Tensor;

type R = Auto;

fn dev() -> Device<R> {
    Device::<R>::default()
}

fn frand(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.max(1);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 11) as f32) / (u64::MAX >> 11) as f32 * 2.0 - 1.0
        })
        .collect()
}

fn two_set_spec() -> EntityModelSpec {
    EntityModelSpec {
        globals: 4,
        context: vec![
            ContextSetSpec::new("cells", 16, 6).with_layout(SetLayout::Grid {
                height: 4,
                width: 4,
                alternate_axes: true,
            }),
            ContextSetSpec::new("items", 5, 3),
        ],
        queries: Some(QuerySetSpec::new("agents", 2, 5, 2)),
        heads: vec![HeadSpec::categorical("kind", 3)],
        d_model: 8,
        context_layers: 2,
        decoder_layers: 1,
        decoder: mamba3::models::entity::DecoderMode::Joint,
        ssm: mamba3::ssm::config::SsmConfig {
            d_model: 8,
            n_heads: 2,
            head_dim: 4,
            d_state: 4,
            n_groups: 2,
            chunk_size: 4,
            ..Default::default()
        },
        chunk_size: None,
        norm_eps: 1e-5,
        seed: 0,
    }
}

#[test]
fn grid_permutation_equals_transpose_grid() {
    let device = dev();
    let data = frand(2 * 100 * 8, 7);
    let x = Var::constant(Tensor::<R, f32>::from_f32(&data, vec![2, 100, 8], &device).unwrap());
    let perm = Permutation::grid_transpose(0, 10, 10, 100);
    let got = perm.apply(&x).unwrap().tensor().to_f32();
    let want = transpose_grid(&x, 10).unwrap().tensor().to_f32();
    assert_eq!(got.len(), want.len());
    for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
        assert!((a - b).abs() < 1e-5, "element {i}: {a} != {b}");
    }
}

#[test]
fn inverse_apply_is_identity() {
    let device = dev();
    let data = frand(2 * 21 * 4, 9);
    let x = Var::constant(Tensor::<R, f32>::from_f32(&data, vec![2, 21, 4], &device).unwrap());
    // Offset grid transpose (leaves the first 5 tokens alone) + block reversal.
    let p = Permutation::grid_transpose(5, 4, 4, 21);
    let q = Permutation::reverse_blocks(0, 3, 7, 21);
    let y = q.apply(&p.apply(&x).unwrap()).unwrap();
    let back = p.inverse(&q.inverse(&y).unwrap()).unwrap();
    assert_eq!(back.tensor().to_f32(), x.tensor().to_f32());
}

#[test]
fn encoder_shape_finite_two_sets() {
    let device = dev();
    let spec = two_set_spec();
    spec.validate().unwrap();
    let model = EntityModel::<R, f32>::init(&spec, &device).unwrap();
    let b = 2;
    let cells = Var::constant(
        Tensor::<R, f32>::from_f32(&frand(b * 16 * 6, 11), vec![b, 16, 6], &device).unwrap(),
    );
    let items = Var::constant(
        Tensor::<R, f32>::from_f32(&frand(b * 5 * 3, 12), vec![b, 5, 3], &device).unwrap(),
    );
    let cells_p = Var::constant(
        Tensor::<R, f32>::from_f32(&vec![1.0; b * 16], vec![b, 16], &device).unwrap(),
    );
    let items_p =
        Var::constant(Tensor::<R, f32>::from_f32(&vec![1.0; b * 5], vec![b, 5], &device).unwrap());
    let glob =
        Var::constant(Tensor::<R, f32>::from_f32(&frand(b * 4, 13), vec![b, 4], &device).unwrap());
    let c = model
        .encode(&[cells, items], &[cells_p, items_p], Some(&glob))
        .unwrap();
    assert_eq!(c.shape().dims(), &[b, 21, 8]);
    assert!(c.tensor().to_f32().iter().all(|v| v.is_finite()));
}

#[test]
fn presence_gates_absent_entities() {
    let device = dev();
    let spec = two_set_spec();
    let model = EntityModel::<R, f32>::init(&spec, &device).unwrap();
    let b = 1;
    let feats_a = frand(16 * 6, 21);
    let mut feats_b = feats_a.clone();
    feats_b[3 * 6] += 5.0; // entity 3 differs, but it is absent in both runs.
    let run = |feats: &[f32]| {
        let cells =
            Var::constant(Tensor::<R, f32>::from_f32(feats, vec![b, 16, 6], &device).unwrap());
        let items = Var::constant(
            Tensor::<R, f32>::from_f32(&frand(b * 5 * 3, 22), vec![b, 5, 3], &device).unwrap(),
        );
        let mut presence = vec![1.0f32; 16];
        presence[3] = 0.0;
        let cells_p =
            Var::constant(Tensor::<R, f32>::from_f32(&presence, vec![b, 16], &device).unwrap());
        let items_p = Var::constant(
            Tensor::<R, f32>::from_f32(&vec![1.0; b * 5], vec![b, 5], &device).unwrap(),
        );
        let glob = Var::constant(
            Tensor::<R, f32>::from_f32(&vec![0.0; b * 4], vec![b, 4], &device).unwrap(),
        );
        model
            .encode(&[cells, items], &[cells_p, items_p], Some(&glob))
            .unwrap()
            .tensor()
            .to_f32()
    };
    // Absent entity 3: its features cannot move the output.
    assert_eq!(run(&feats_a), run(&feats_b));
    // Sanity: a present entity's features do move the output.
    let mut feats_c = feats_a.clone();
    feats_c[4 * 6] += 5.0;
    assert_ne!(run(&feats_a), run(&feats_c));
}
