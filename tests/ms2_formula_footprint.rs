//! V0-A footprint: warmed `formula_window` + `formula_top` + `score` + `loss`
//! performs no runtime reads, and the launch count per call is constant.

#![cfg(feature = "backend")]

use mamba3::autograd::Var;
use mamba3::backend::{
    Device, check_launches, launch_count, reset_launch_count, reset_read_count,
    reset_transfer_counters, runtime_read_count,
};
use mamba3::backends::Auto;
use mamba3::models::ms2::chem::composition_error_nda;
use mamba3::models::ms2::contract::ModelConfig;
use mamba3::models::ms2::formula::FormulaTable;
use mamba3::models::ms2::formula_head::{DeviceFormulaTable, FormulaHead};
use mamba3::tensor::Tensor;
use mamba3::tensor::ops::index::IdTensor;
use mamba3::tensor::ops::ms2;
use mamba3::tensor::ops::random::Rng;

type R = Auto;
type E = f32;

#[test]
fn ms2_formula_footprint() {
    let device = Device::<R>::default();
    let table = FormulaTable::from_compositions(
        [
            [0u16, 2, 0, 1, 0, 0, 0, 0, 0, 0],
            [1, 4, 0, 0, 0, 0, 0, 0, 0, 0],
            [2, 6, 0, 1, 0, 0, 0, 0, 0, 0],
            [6, 6, 0, 0, 0, 0, 0, 0, 0, 0],
            [6, 12, 0, 6, 0, 0, 0, 0, 0, 0],
            [8, 10, 4, 2, 0, 0, 0, 0, 0, 0],
        ]
        .into_iter(),
    )
    .unwrap();
    let (batch, m, f, d) = (2usize, 8usize, 2usize, 8usize);
    let mut search_host = Vec::with_capacity(table.len() * 2);
    for row in 0..table.len() {
        search_host.push(table.mass(row));
        search_host.push((composition_error_nda(table.composition(row)).div_ceil(1000)) as u32);
    }
    let search = IdTensor::from_slice(&search_host, vec![table.len(), 2], &device).unwrap();
    let hit0 = table.mass(2) + 1_007_825 - 549;
    let hit1 = table.mass(4) + 1_007_825 - 549;
    let meta_host: Vec<u32> = vec![
        3, hit0, 50, 1, 100, 1000, 0, 0, //
        3, hit1, 50, 1, 100, 200, 0, 0,
    ];
    let meta = IdTensor::from_slice(&meta_host, vec![batch, 8], &device).unwrap();
    let uploaded = DeviceFormulaTable::<R, E>::upload(&table, &device).unwrap();
    let mut model = ModelConfig::v0();
    model.d_model = d as u32;
    let mut rng = Rng::seeded(13);
    let head = FormulaHead::<R, E>::init(&model, &device, &mut rng).unwrap();
    let mut rng = Rng::seeded(29);
    let pool_host = rng.uniform_vec(batch * d, -0.5, 0.5);
    let pool =
        Var::constant(Tensor::<R, E>::from_f32(&pool_host, vec![batch, d], &device).unwrap());
    let gold = IdTensor::from_slice(&[0u32, 1], vec![batch], &device).unwrap();
    let call = || {
        let buffers = ms2::FormulaBuffers::<R, E>::new(batch, m, f, &device);
        ms2::formula_window(
            &search,
            &meta,
            table.max_error(),
            u32::MAX,
            u32::MAX,
            &buffers,
        )
        .unwrap();
        let out = head.score(&uploaded, &buffers, &pool).unwrap();
        ms2::formula_top(out.log_prob.tensor(), &buffers.window, &buffers).unwrap();
        let _ = head.loss(&out, &gold).unwrap();
        check_launches(&device).unwrap();
    };
    call();
    call();
    // Warmed production: no reads, constant launches per call.
    reset_launch_count();
    reset_read_count();
    reset_transfer_counters();
    let before = launch_count();
    call();
    let first = launch_count() - before;
    let first_reads = runtime_read_count();
    reset_launch_count();
    reset_read_count();
    reset_transfer_counters();
    let before = launch_count();
    call();
    let second = launch_count() - before;
    println!("formula launches per call: {second} (previous {first})");
    assert_eq!(second, first, "launch count stable");
    assert_eq!(runtime_read_count(), 0, "warmed call does no reads");
    assert_eq!(first_reads, 0, "previous warmed call did no reads");
}
