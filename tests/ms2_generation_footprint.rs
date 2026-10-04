//! V0-C footprint: one warmed generation call performs exactly one runtime
//! read, consecutive warmed calls launch the same kernels, and reserved bytes
//! stay flat over 200 repeated calls and an alternating-bucket sequence.

#![cfg(feature = "backend")]

use mamba3::backend::{
    allocation_calls, check_launches, download_bytes, launch_count, launch_tally_detailed,
    reserved_bytes, reset_launch_count, reset_transfer_counters, runtime_read_count,
    start_launch_tally, stop_launch_tally,
};
use mamba3::backends::Auto;
use mamba3::models::ms2::chem::Composition;
use mamba3::models::ms2::contract::{
    Control, FormulaSource, GenerationConfig, GenerationMode, SCHEMA_VERSION,
    SPECTRUM_SCHEMA_VERSION,
};
use mamba3::models::ms2::contract::{ModelConfig, SpectrumBatch};
use mamba3::models::ms2::formula::FormulaTable;
use mamba3::models::ms2::formula_head::DeviceFormulaTable;
use mamba3::models::ms2::generate::{GenerationWorkspace, Ms2Model};
use mamba3::models::ms2::workspace::Ms2MemoryEstimate;
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

/// Tiny V1-shape model for the footprint tests: `(A, R_max, T) = (32, 8, 42)`
/// with 4 decoder blocks and the decoder inner width set apart (4 heads x 8
/// channels = 32 against the encoder 2 x 8 = 16); `d`, attention heads and
/// the peak cap stay tiny/V0 so the CPU cost stays reasonable.
fn v1_tiny_footprint_config() -> ModelConfig {
    let mut m = tiny_config();
    m.max_atoms = 32;
    m.max_ring_closures = 8;
    m.decoder_blocks = 4;
    m.decoder.n_heads = 4;
    m.decoder.head_dim = 8;
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
        formula_source: FormulaSource::Table,
        formula_window: 32,
        enum_lanes_max: 262_144,
        enum_lane_visits_max: 65_536,
        enum_dispatch_visits_max: 4_000_000,
        allocation: mamba3::models::ms2::contract::AllocationMode::RoundRobin,
        identity: mamba3::models::ms2::contract::IdentityMode::TraceOnly,
        identity_work_max: 4096,
        returned: 0,
        evidence: false,
        ion_request_work_max: 268435456,
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
        schema_version: SPECTRUM_SCHEMA_VERSION,
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

/// File-level serialisation for counter-reading tests: the process-wide
/// launch/read/transfer counters (and the shared device allocator) are
/// perturbed by any test running beside these, so every test holds this
/// mutex. Poison-tolerant: a panicking holder still releases the lock.
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn serial() -> std::sync::MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}
#[test]
fn warmed_generate_reads_once_launches_constantly_and_holds_memory() {
    let _serial = serial();
    // V1 §3.1 shape pins: one read per warmed `generate` at both the V0
    // shape and the V1 capacities (32, 8, 42) with small B and K. Only the
    // V0 shape pins launch numbers; both assert the single read, constant
    // launches and constant memory.
    for (mkm, gcfg_t, n_repeat, is_v0) in [
        (tiny_config as fn() -> ModelConfig, 22u32, 200usize, true),
        (
            v1_tiny_footprint_config as fn() -> ModelConfig,
            42u32,
            50usize,
            false,
        ),
    ] {
        let device = mamba3::backend::Device::<R>::default();
        let mut cfg = mkm();
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
        let mut gcfg = tiny_generation();
        gcfg.max_steps = gcfg_t;
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
        // B1-fix2 pinned V0 launch counts (CPU): a warmed call at this exact
        // tiny config launches `l_total` kernels, `l_search` of them in the
        // search stage. Any kernel added to or removed from the V0 path fails
        // here rather than drifting silently. I2 adds exactly one search
        // launch (`ms2_allocate`), so the pins moved 2377 -> 2378 and
        // 32 -> 33 on CPU (2252 -> 2253 on wgpu, same delta). The fused
        // sampler step (P8/O4) then moved the totals to 956 on CPU and 894 on
        // wgpu.
        println!("PINNED warmed generate total launches: {l_total}");
        println!("PINNED warmed generate search launches per call: {l_search}");
        // CPU-runtime numbers; wgpu (including the `vulkan`/`msl` builds, which
        // set the `wgpu` feature too) picks other kernels for the same ops, so it
        // pins its own measured numbers; any other backend prints and skips.
        if is_v0 {
            if device.name() == "cpu" {
                assert_eq!(
                    l_total, 956,
                    "warmed generate total launches pinned on CPU"
                );
                assert_eq!(
                    l_search, 33,
                    "warmed generate search-stage launches per call pinned on CPU"
                );
            } else if device.name() == "wgpu" {
                assert_eq!(
                    l_total, 894,
                    "warmed generate total launches pinned on wgpu"
                );
                assert_eq!(
                    l_search, 33,
                    "warmed generate search-stage launches per call pinned on wgpu"
                );
            } else {
                println!(
                    "unpinned backend {}: warmed generate total {l_total}, search {l_search}",
                    device.name()
                );
            }
        } else {
            println!(
                "V1 shape: warmed generate total {l_total}, search {l_search} (backend {}, no numeric pin)",
                device.name()
            );
        }
        if let (Some(a), Some(b), Some(c)) = (mem_before, mem_after, mem_after2) {
            assert_eq!(a, b, "reserved bytes unchanged by the third call");
            assert_eq!(b, c, "reserved bytes unchanged by the fourth call");
            println!("reserved bytes: {a}");
        }
        // 200 repeated calls with constant reserved bytes.
        let base_mem = reserved_bytes(&device);
        for _ in 0..n_repeat {
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
}

fn gen_setup(
    max_atoms: u32,
    seed: u64,
) -> (
    mamba3::backend::Device<R>,
    DeviceFormulaTable<R, f32>,
    Ms2Model<R, f32>,
    Ms2Constants<R>,
    Vec<Composition>,
) {
    let device = mamba3::backend::Device::<R>::default();
    let mut cfg = tiny_config();
    cfg.max_atoms = max_atoms;
    let comps: Vec<Composition> = vec![
        [2, 6, 0, 1, 0, 0, 0, 0, 0, 0],
        [6, 6, 0, 0, 0, 0, 0, 0, 0, 0],
        [3, 7, 1, 2, 0, 0, 0, 0, 0, 0],
    ];
    let host_table = FormulaTable::from_compositions(comps.clone()).unwrap();
    let table = DeviceFormulaTable::<R, f32>::upload(&host_table, &device).unwrap();
    cfg.formula_table.rows = table.rows as u32;
    cfg.formula_table.sha256 = table.sha256.clone();
    let mut rng = Rng::seeded(seed);
    let model = Ms2Model::<R, f32>::init(&cfg, &device, &mut rng).unwrap();
    let constants = Ms2Constants::new(&device);
    (device, table, model, constants, comps)
}

fn precursors_for(comps: &[Composition], b: usize) -> Vec<u32> {
    comps[..b]
        .iter()
        .map(|c| mamba3::models::ms2::chem::composition_mass(c).unwrap() + 1_007_825 - 549)
        .collect()
}

#[test]
fn readout_bytes_match_estimate_exactly_for_several_shapes() {
    let _serial = serial();
    // B1-fix2 finding 2: the estimate's `readout` item equals the
    // `download_bytes` delta of a real warmed `generate` call exactly, for
    // several (B, K, F, T, A) including K = 1, F = 8. A warmed call performs
    // exactly one runtime read (the packed readout), so the delta is the
    // read's byte size; equality proves the estimate prices the actual read
    // layout (`generation_readout_counts`, shared by both sides).
    for (b, k, f, t, a, seed) in [
        (2usize, 4u32, 2u32, 22u32, 8u32, 111u64),
        (1usize, 1u32, 8u32, 12u32, 6u32, 112u64),
        (3usize, 2u32, 4u32, 18u32, 10u32, 113u64),
        // V1 §3.1: the 32-atom, 42-step shape reads once with exact bytes too.
        (2usize, 2u32, 2u32, 42u32, 32u32, 114u64),
    ] {
        let (device, table, model, constants, comps) = gen_setup(a, seed);
        let mut gcfg = tiny_generation();
        gcfg.trajectories = k;
        gcfg.formulas = f;
        gcfg.max_steps = t;
        let mut ws = GenerationWorkspace::new();
        let precursors = precursors_for(&comps, b);
        let ids: Vec<u64> = (0..b as u64).map(|i| 900 + i).collect();
        let counts = vec![10u32; b];
        let batch = make_spectra(&ids, &precursors, 64, &counts, seed);
        for _ in 0..2 {
            let out = model
                .generate(&batch, &table, &gcfg, &mut ws, &constants)
                .unwrap();
            mamba3::backend::check_launches(&device).unwrap();
            out.validate().unwrap();
        }
        let est = Ms2MemoryEstimate::generation(
            &model.config,
            table.rows as u64,
            b as u64,
            k as u64,
            batch.n_raw as u64,
            t as u64,
            gcfg.formula_window as u64,
            f as u64,
        )
        .unwrap();
        let want_readout = est.get("readout").unwrap();
        reset_transfer_counters();
        let d0 = download_bytes();
        let r0 = runtime_read_count();
        let out = model
            .generate(&batch, &table, &gcfg, &mut ws, &constants)
            .unwrap();
        mamba3::backend::check_launches(&device).unwrap();
        out.validate().unwrap();
        let reads = runtime_read_count() - r0;
        let bytes = download_bytes() - d0;
        println!(
            "B={b} K={k} F={f} T={t} A={a}: reads {reads}, download {bytes}, estimate readout {want_readout}"
        );
        assert_eq!(
            reads, 1,
            "B={b} K={k} F={f} T={t} A={a}: warmed call reads once"
        );
        assert_eq!(
            bytes, want_readout,
            "B={b} K={k} F={f} T={t} A={a}: download bytes equal the readout estimate"
        );
    }
}

#[test]
fn refused_m2048_first_call_leaves_counters_unchanged() {
    let _serial = serial();
    // B1-fix2 finding 6: the FIRST refused call leaves the allocation and
    // launch counters unchanged. This binary owns those counters while it
    // runs (single-threaded footprint run): no retry loop, one refused
    // call, exact assertions.
    let (_device, table, model, constants, comps) = gen_setup(8, 121);
    let precursors = precursors_for(&comps, 2);
    let batch = make_spectra(&[901, 902], &precursors, 64, &[10, 12], 122);
    let mut cfg32 = tiny_generation();
    cfg32.formula_window = 32;
    let est32 = Ms2MemoryEstimate::generation(
        &model.config,
        table.rows as u64,
        batch.len() as u64,
        cfg32.trajectories as u64,
        batch.n_raw as u64,
        cfg32.max_steps as u64,
        cfg32.formula_window as u64,
        cfg32.formulas as u64,
    )
    .unwrap()
    .total()
    .unwrap();
    let mut cfg = tiny_generation();
    cfg.formula_window = 2048;
    let est2048 = Ms2MemoryEstimate::generation(
        &model.config,
        table.rows as u64,
        batch.len() as u64,
        cfg.trajectories as u64,
        batch.n_raw as u64,
        cfg.max_steps as u64,
        cfg.formula_window as u64,
        cfg.formulas as u64,
    )
    .unwrap()
    .total()
    .unwrap();
    assert!(
        est2048 > est32,
        "M=2048 estimate {est2048} exceeds M=32 estimate {est32}"
    );
    cfg.max_device_bytes = est32 + (est2048 - est32) / 2;
    let mut ws = GenerationWorkspace::new();
    let a0 = allocation_calls();
    let l0 = launch_count();
    let err = model
        .generate(&batch, &table, &cfg, &mut ws, &constants)
        .unwrap_err();
    assert!(
        matches!(err, mamba3::error::Error::Config(_)),
        "refused M=2048 call is Config, got {err:?}"
    );
    assert!(err.to_string().contains("exceeds limit"), "{err}");
    assert_eq!(
        allocation_calls(),
        a0,
        "the first refused call allocates nothing"
    );
    assert_eq!(
        launch_count(),
        l0,
        "the first refused call launches nothing"
    );
    assert!(
        ws.bucket_keys().is_empty(),
        "a refused M=2048 call creates no workspace bucket"
    );
    println!("refused M=2048 first call: allocations and launches unchanged");
}
