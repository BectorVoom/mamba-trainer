//! P2.2 V0-shape generation reconciliation (split from `ms2_footprint.rs`).
//!
//! `reserved_bytes` is the pool high-water mark, so the V0 shape measured
//! after the V1-candidate pool has been built would see no growth. This is
//! the only test in its binary: the V0 shape is the first allocation of its
//! process, which makes the reserved-increase measurement valid on every
//! backend. Band 0.5..=8.0, unchanged.

#![cfg(feature = "backend")]

use mamba3::backend::{Device, reserved_bytes};
use mamba3::backends::Auto;
use mamba3::models::ms2::chem::{Composition, composition_mass};
use mamba3::models::ms2::contract::{
    GenerationConfig, ModelConfig, SPECTRUM_SCHEMA_VERSION, SpectrumBatch,
};
use mamba3::models::ms2::formula::FormulaTable;
use mamba3::models::ms2::formula_head::DeviceFormulaTable;
use mamba3::models::ms2::generate::{GenerationWorkspace, Ms2Model};
use mamba3::models::ms2::workspace::Ms2MemoryEstimate;
use mamba3::tensor::ops::ms2::Ms2Constants;
use mamba3::tensor::ops::random::Rng;

type R = Auto;
type E = f32;

fn parent_comps(b: usize) -> Vec<Composition> {
    let base = vec![
        [6, 6, 0, 0, 0, 0, 0, 0, 0, 0],
        [3, 7, 1, 2, 0, 0, 0, 0, 0, 0],
        [6, 12, 0, 6, 0, 0, 0, 0, 0, 0],
        [10, 8, 0, 0, 0, 0, 0, 0, 0, 0],
    ];
    (0..b).map(|i| base[i % base.len()]).collect()
}

fn precursor_of(comp: &Composition) -> u32 {
    composition_mass(comp).unwrap() + 1_007_825 - 549
}

fn spectra_batch(comps: &[Composition], n_raw: usize, seed: u64) -> SpectrumBatch {
    let b = comps.len();
    let precursors: Vec<u32> = comps.iter().map(precursor_of).collect();
    let mut rng = Rng::seeded(seed);
    let mut peak_id = vec![u32::MAX; b * n_raw];
    let mut mz = vec![0u32; b * n_raw];
    let mut intensity = vec![0.0f32; b * n_raw];
    let mut peak_count = vec![0u32; b];
    let mut raw_peak_count = vec![0u32; b];
    for bi in 0..b {
        let count = (n_raw / 2) as u32;
        peak_count[bi] = count;
        raw_peak_count[bi] = count;
        for i in 0..count as usize {
            peak_id[bi * n_raw + i] = i as u32;
            let f = rng.uniform_vec(1, 60_000_000.0, (precursors[bi] - 5_000_000) as f32)[0] as u32;
            mz[bi * n_raw + i] = f.max(50_000_001);
            intensity[bi * n_raw + i] = 0.5 + 2.0 * rng.uniform_vec(1, 0.0, 1.0)[0];
        }
    }
    SpectrumBatch {
        schema_version: SPECTRUM_SCHEMA_VERSION,
        n_raw: n_raw as u32,
        spectrum_id: (0..b as u64).map(|i| 1000 + i).collect(),
        raw_peak_count,
        peak_count,
        peak_id,
        mz_udalton: mz,
        intensity,
        intensity_scale: 0,
        mz_uncertainty_udalton: vec![50; b],
        precursor_mz_udalton: precursors,
        precursor_uncertainty_udalton: vec![50; b],
        adduct: vec![1; b],
        polarity: vec![1; b],
        collision_energy_ev: vec![30.0; b],
        collision_energy_known: vec![1; b],
        energy_count: vec![1; b],
        fragment_tolerance_ppm_tenths: vec![0; b],
        precursor_tolerance_ppm_tenths: vec![0; b],
        instrument_class: vec![0; b],
    }
}

#[test]
fn generation_estimate_reconciles_with_reserved_bytes_v0() {
    let device = Device::<R>::default();
    let Some(baseline) = reserved_bytes(&device) else {
        println!("generation footprint: reserved bytes unavailable");
        println!("unavailable");
        return;
    };
    let (b, k, t) = (8usize, 8u32, 22u32);
    let model_config = ModelConfig::v0();
    let comps = parent_comps(b);
    let host_table = FormulaTable::from_compositions(comps.clone().into_iter()).unwrap();
    let table = DeviceFormulaTable::<R, E>::upload(&host_table, &device).unwrap();
    let mut config = model_config.clone();
    config.formula_table.rows = table.rows as u32;
    config.formula_table.sha256 = table.sha256.clone();
    let mut rng = Rng::seeded(5);
    let model = Ms2Model::<R, E>::init(&config, &device, &mut rng).unwrap();
    let constants = Ms2Constants::new(&device);
    let batch = spectra_batch(&comps, 64, 21);
    let gen_config = GenerationConfig {
        trajectories: k,
        max_steps: t,
        ..GenerationConfig::default()
    };
    let estimate = Ms2MemoryEstimate::generation(
        &model_config,
        table.rows as u64,
        b as u64,
        k as u64,
        64,
        gen_config.max_steps as u64,
        gen_config.formula_window as u64,
        gen_config.formulas as u64,
    )
    .unwrap()
    .total()
    .unwrap();
    let mut workspace = GenerationWorkspace::new();
    for _ in 0..2 {
        let out = model
            .generate(&batch, &table, &gen_config, &mut workspace, &constants)
            .unwrap();
        out.validate().unwrap();
    }
    let Some(end) = reserved_bytes(&device) else {
        println!("generation footprint: reserved bytes unavailable");
        println!("unavailable");
        return;
    };
    let measured = end.saturating_sub(baseline);
    let ratio = measured as f64 / estimate as f64;
    println!("generation estimate total (b={b} k={k} t={t}): {estimate} bytes");
    println!("generation measured reserved increase (b={b} k={k} t={t}): {measured} bytes");
    println!("generation ratio measured/estimate (b={b} k={k} t={t}): {ratio:.3}");
    assert!(
        (0.5..=8.0).contains(&ratio),
        "generation reserved/estimate ratio {ratio:.3} outside 0.5..=8.0 \
         (estimate {estimate}, measured {measured}): the pool rounds and holds \
         matmul/autotune scratch, so reserved is a high-water mark, not the live sum"
    );
}
