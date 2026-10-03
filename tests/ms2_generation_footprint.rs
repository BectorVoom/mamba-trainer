//! V0-C footprint: one warmed generation call performs exactly one runtime
//! read, consecutive warmed calls launch the same kernels, and reserved bytes
//! stay flat over 200 repeated calls and an alternating-bucket sequence.

#![cfg(feature = "backend")]

use mamba3::backend::{
    check_launches, launch_count, launch_tally_detailed, reserved_bytes, reset_launch_count,
    reset_transfer_counters, runtime_read_count, start_launch_tally, stop_launch_tally,
};
use mamba3::backends::Auto;
use mamba3::models::ms2::chem::Composition;
use mamba3::models::ms2::contract::{Control, GenerationConfig, GenerationMode, SCHEMA_VERSION};
use mamba3::models::ms2::contract::{ModelConfig, SpectrumBatch};
use mamba3::models::ms2::formula::FormulaTable;
use mamba3::models::ms2::formula_head::DeviceFormulaTable;
use mamba3::models::ms2::generate::{GenerationWorkspace, Ms2Model};
use mamba3::tensor::ops::ms2::Ms2Constants;
use mamba3::tensor::ops::random::Rng;

type R = Auto;

fn tiny_config() -> ModelConfig {
    let mut m = ModelConfig::v0();
    m.d_model = 16;
    m.n_peaks = 16;
    m.encoder_blocks = 1;
    m.decoder_blocks = 1;
    m.attention_heads = 2;
    m.encoder.d_model = 16;
    m.encoder.n_heads = 2;
    m.encoder.head_dim = 8;
    m.encoder.d_state = 8;
    m.encoder.n_groups = 2;
    m.decoder.d_model = 16;
    m.decoder.n_heads = 2;
    m.decoder.head_dim = 8;
    m.decoder.d_state = 8;
    m.decoder.n_groups = 2;
    m
}

fn tiny_generation() -> GenerationConfig {
    GenerationConfig {
        schema_version: SCHEMA_VERSION,
        trajectories: 4,
        formulas: 2,
        seed: 7,
        temperature: 1.0,
        max_steps: 22,
        max_device_bytes: 2 * 1024 * 1024 * 1024,
        formula_rows_visited_max: u32::MAX,
        formula_rows_scored_max: 4096,
        mode: GenerationMode::Sampling,
        oracle_formula: false,
        control: Control::None,
    }
}

fn make_spectra(
    spectrum_ids: &[u64],
    precursors: &[u32],
    n_raw: usize,
    peak_counts: &[u32],
    seed: u64,
) -> SpectrumBatch {
    let b = spectrum_ids.len();
    let mut rng = Rng::seeded(seed);
    let mut peak_id = vec![u32::MAX; b * n_raw];
    let mut mz = vec![0u32; b * n_raw];
    let mut intensity = vec![0.0f32; b * n_raw];
    let mut peak_count = vec![0u32; b];
    let mut raw_peak_count = vec![0u32; b];
    for (bi, &count) in peak_counts.iter().enumerate() {
        let count = count as usize;
        peak_count[bi] = count as u32;
        raw_peak_count[bi] = count as u32;
        let precursor = precursors[bi];
        for i in 0..count {
            peak_id[bi * n_raw + i] = i as u32;
            let f = rng.uniform_vec(1, 60_000_000.0, (precursor - 5_000_000) as f32)[0] as u32;
            mz[bi * n_raw + i] = f.max(50_000_001);
            let u = rng.uniform_vec(1, 0.0, 1.0)[0];
            intensity[bi * n_raw + i] = 0.5 + 2.0 * u;
        }
    }
    SpectrumBatch {
        schema_version: SCHEMA_VERSION,
        n_raw: n_raw as u32,
        spectrum_id: spectrum_ids.to_vec(),
        raw_peak_count,
        peak_count,
        peak_id,
        mz_udalton: mz,
        intensity,
        intensity_scale: 0,
        mz_uncertainty_udalton: vec![50; b],
        precursor_mz_udalton: precursors.to_vec(),
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
fn warmed_generate_reads_once_launches_constantly_and_holds_memory() {
    let device = mamba3::backend::Device::<R>::default();
    let mut cfg = tiny_config();
    let comps: Vec<Composition> = vec![
        [2, 6, 0, 1, 0, 0, 0, 0, 0, 0],
        [6, 6, 0, 0, 0, 0, 0, 0, 0, 0],
        [3, 7, 1, 2, 0, 0, 0, 0, 0, 0],
    ];
    let host_table = FormulaTable::from_compositions(comps.clone()).unwrap();
    let table = DeviceFormulaTable::<R, f32>::upload(&host_table, &device).unwrap();
    cfg.formula_table.rows = table.rows as u32;
    cfg.formula_table.sha256 = table.sha256.clone();
    let mut rng = Rng::seeded(5);
    let model = Ms2Model::<R, f32>::init(&cfg, &device, &mut rng).unwrap();
    let constants = Ms2Constants::new(&device);
    let mut ws = GenerationWorkspace::new();
    let gcfg = tiny_generation();
    let precursors: Vec<u32> = comps[0..2]
        .iter()
        .map(|c| mamba3::models::ms2::chem::composition_mass(c).unwrap() + 1_007_825 - 549)
        .collect();
    let batch_a = make_spectra(&[501, 502], &precursors, 64, &[10, 12], 31);
    let precursors_b: Vec<u32> = comps[0..3]
        .iter()
        .map(|c| mamba3::models::ms2::chem::composition_mass(c).unwrap() + 1_007_825 - 549)
        .collect();
    let batch_b = make_spectra(&[601, 602, 603], &precursors_b, 64, &[10, 12, 9], 33);
    // Two warm-up calls on bucket A (tuning and compilation settle here).
    for _ in 0..2 {
        let out = model
            .generate(&batch_a, &table, &gcfg, &mut ws, &constants)
            .unwrap();
        check_launches(&device).unwrap();
        out.validate().unwrap();
    }
    // The measured third call, with the launch tally running.
    start_launch_tally();
    reset_launch_count();
    reset_transfer_counters();
    let l_before = launch_count();
    let r_before = runtime_read_count();
    let mem_before = reserved_bytes(&device);
    let out = model
        .generate(&batch_a, &table, &gcfg, &mut ws, &constants)
        .unwrap();
    check_launches(&device).unwrap();
    out.validate().unwrap();
    let l_after = launch_count();
    let r_after = runtime_read_count();
    let mem_after = reserved_bytes(&device);
    let l_total = l_after - l_before;
    let r_total = r_after - r_before;
    // A fourth call on the same shapes: identical cost.
    reset_launch_count();
    reset_transfer_counters();
    let l_before2 = launch_count();
    let r_before2 = runtime_read_count();
    let out2 = model
        .generate(&batch_a, &table, &gcfg, &mut ws, &constants)
        .unwrap();
    check_launches(&device).unwrap();
    out2.validate().unwrap();
    let l_after2 = launch_count();
    let r_after2 = runtime_read_count();
    let mem_after2 = reserved_bytes(&device);
    stop_launch_tally();
    println!(
        "warmed generate: L_total {l_total}, reads +{r_total} then +{}",
        r_after2 - r_before2
    );
    assert_eq!(
        r_total, 1,
        "third warmed call performs exactly one runtime read"
    );
    assert_eq!(
        r_after2 - r_before2,
        1,
        "fourth warmed call performs exactly one runtime read"
    );
    assert_eq!(
        l_after2 - l_before2,
        l_total,
        "consecutive warmed calls launch equally"
    );
    // Per-stage split from the tally (labels are `ms2.preprocess`,
    // `ms2.encoder`, `ms2.search`, `ms2.step`, `ms2.finalize`).
    let mut stage: std::collections::HashMap<String, usize> = Default::default();
    for row in launch_tally_detailed() {
        *stage.entry(row.label).or_insert(0) += row.count;
    }
    let steps = gcfg.max_steps as usize - 1;
    let l_pre = stage.get("ms2.preprocess").copied().unwrap_or(0) / 2;
    let l_enc = stage.get("ms2.encoder").copied().unwrap_or(0) / 2;
    let l_search = stage.get("ms2.search").copied().unwrap_or(0) / 2;
    let l_step = stage.get("ms2.step").copied().unwrap_or(0) / 2 / steps;
    let l_fin = stage.get("ms2.finalize").copied().unwrap_or(0) / 2;
    let l_other = stage.get("-").copied().unwrap_or(0) / 2;
    let tally_total: usize = stage.values().sum();
    println!(
        "per call: preprocess {l_pre}, encoder {l_enc}, search {l_search}, per step {l_step} x {steps}, finalize {l_fin}, unscored {l_other} (tally total {})",
        tally_total / 2
    );
    if let (Some(a), Some(b), Some(c)) = (mem_before, mem_after, mem_after2) {
        assert_eq!(a, b, "reserved bytes unchanged by the third call");
        assert_eq!(b, c, "reserved bytes unchanged by the fourth call");
        println!("reserved bytes: {a}");
    }
    // 200 repeated calls with constant reserved bytes.
    let base_mem = reserved_bytes(&device);
    for _ in 0..200 {
        let out = model
            .generate(&batch_a, &table, &gcfg, &mut ws, &constants)
            .unwrap();
        out.validate().unwrap();
    }
    check_launches(&device).unwrap();
    let end_mem = reserved_bytes(&device);
    if let (Some(a), Some(b)) = (base_mem, end_mem) {
        assert_eq!(a, b, "reserved bytes constant over 200 calls");
        println!("reserved bytes over 200 calls: {a}");
    }
    // An alternating-bucket sequence (two batch sizes) without growth after
    // warm-up: warm bucket B, then alternate.
    for _ in 0..2 {
        let out = model
            .generate(&batch_b, &table, &gcfg, &mut ws, &constants)
            .unwrap();
        out.validate().unwrap();
    }
    check_launches(&device).unwrap();
    let alt_base = reserved_bytes(&device);
    for _ in 0..20 {
        for batch in [&batch_a, &batch_b] {
            let out = model
                .generate(batch, &table, &gcfg, &mut ws, &constants)
                .unwrap();
            out.validate().unwrap();
        }
    }
    check_launches(&device).unwrap();
    let alt_end = reserved_bytes(&device);
    if let (Some(a), Some(b)) = (alt_base, alt_end) {
        assert_eq!(
            a, b,
            "reserved bytes constant over the alternating sequence"
        );
        println!("reserved bytes alternating: {a}");
    }
}
