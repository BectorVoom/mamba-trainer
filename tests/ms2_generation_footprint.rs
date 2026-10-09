//! V0-C footprint: one warmed generation call performs exactly one runtime
//! read, consecutive warmed calls launch the same kernels, and reserved bytes
//! stay flat over 200 repeated calls and an alternating-bucket sequence.

#![cfg(feature = "backend")]

use mamba3::backend::{
    ScratchArena, allocation_calls, check_launches, download_bytes, launch_count,
    launch_tally_detailed, live_bytes, live_high_water_bytes, reserved_bytes, reset_launch_count,
    reset_live_high_water, reset_transfer_counters, runtime_read_count, start_launch_tally,
    stop_launch_tally, with_scratch,
};
use mamba3::backends::Auto;
use mamba3::models::mamba3::set_fused_step;
use mamba3::models::ms2::workspace::{Ms2MemoryEstimate, carry_bytes};
use mamba3::models::ms2::chem::Composition;
use mamba3::models::ms2::contract::{
    Control, FormulaSource, GenerationConfig, GenerationMode, SCHEMA_VERSION,
    SPECTRUM_SCHEMA_VERSION,
};
use mamba3::models::ms2::contract::{ModelConfig, SpectrumBatch};
use mamba3::models::ms2::formula::FormulaTable;
use mamba3::models::ms2::formula_head::DeviceFormulaTable;
use mamba3::models::ms2::generate::{
    GenerateStage, GenerationWorkspace, Ms2Model, set_scratch_arena,
};
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
        formula_evidence_work_max: 2048,
        formula_evidence_dispatch_max: 268435456,
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
        // wgpu, and the fused mixer step with the stored pointer scores and
        // the paired carry freeze moved the CPU total to 675. Carries stepped
        // in place (no freeze launches, one more fill at decoder init) moved
        // it to 634.
        println!("PINNED warmed generate total launches: {l_total}");
        println!("PINNED warmed generate search launches per call: {l_search}");
        // CPU-runtime numbers; wgpu (including the `vulkan`/`msl` builds, which
        // set the `wgpu` feature too) picks other kernels for the same ops, so it
        // pins its own measured numbers; any other backend prints and skips.
        if is_v0 {
            if device.name() == "cpu" {
                assert_eq!(
                    l_total, 634,
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

/// Like [`gen_setup`] on the 4-layer V1 probe
/// ([`v1_tiny_footprint_config`]): `(A, T) = (32, 42)` with 4 decoder blocks.
fn gen_setup_v1(
    seed: u64,
) -> (
    mamba3::backend::Device<R>,
    DeviceFormulaTable<R, f32>,
    Ms2Model<R, f32>,
    Ms2Constants<R>,
    Vec<Composition>,
) {
    let device = mamba3::backend::Device::<R>::default();
    let mut cfg = v1_tiny_footprint_config();
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
fn refused_m2048_first_call_leaves_counters_unchanged() {    let _serial = serial();
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

// ---------------------------------------------------------------------------
// Task T3: no device-buffer allocation in the warmed decode loop.
// ---------------------------------------------------------------------------

use mamba3::models::ms2::contract::{AssignmentConfig, IdentityMode};
use mamba3::models::ms2::formula_enum::{EnumDomain, RatioBounds};

/// Which generation entry point a T3 loop-allocation probe drives.
#[derive(Debug, Clone, Copy)]
enum LoopCall {
    Generate,
    Packed,
    Resident,
}

/// Whole-call allocation calls and launches of one warmed UNHOOKED call
/// (task F10 item A1: hooked calls run without the arena, so the
/// arena-engaged measurement below cannot use hooks to delimit the loop).
fn measure_whole_allocs(
    model: &Ms2Model<R, f32>,
    batch: &SpectrumBatch,
    table: &DeviceFormulaTable<R, f32>,
    gcfg: &GenerationConfig,
    ws: &mut GenerationWorkspace<R, f32>,
    constants: &Ms2Constants<R>,
    which: LoopCall,
) -> (usize, usize) {
    reset_transfer_counters();
    reset_launch_count();
    match which {
        LoopCall::Generate => {
            let out = model.generate(batch, table, gcfg, ws, constants).unwrap();
            check_launches(table.table.device()).unwrap();
            out.validate().unwrap();
        }
        LoopCall::Packed => {
            let out = model.generate_packed(batch, table, gcfg, ws, constants).unwrap();
            check_launches(table.table.device()).unwrap();
            out.validate().unwrap();
        }
        LoopCall::Resident => {
            let resident = model.generate_resident(batch, table, gcfg, ws, constants).unwrap();
            check_launches(table.table.device()).unwrap();
            let out = resident.read(model).unwrap();
            out.validate().unwrap();
            resident.release_into(ws);
        }
    }
    (allocation_calls(), launch_count())
}

/// Allocation calls of one warmed HOOKED call in total and inside its
/// decode loop (the hook delimits the loop; hooked calls never engage the
/// arena, so this measurer is the scratch-OFF probe).
fn measure_loop_allocs(
    model: &Ms2Model<R, f32>,
    batch: &SpectrumBatch,
    table: &DeviceFormulaTable<R, f32>,
    gcfg: &GenerationConfig,
    ws: &mut GenerationWorkspace<R, f32>,
    constants: &Ms2Constants<R>,
    which: LoopCall,
) -> (usize, usize) {
    reset_transfer_counters();
    let mut start = 0usize;
    let mut last_step = 0usize;
    let mut in_loop = false;
    {
        let mut hook = |stage: GenerateStage| {
            match stage {
                GenerateStage::AfterDecoderInit => {
                    start = allocation_calls();
                    in_loop = true;
                }
                GenerateStage::AfterDecodeStep(_) => {
                    if in_loop {
                        last_step = allocation_calls();
                    }
                }
                GenerateStage::AfterValidate => {
                    in_loop = false;
                }
                _ => {}
            }
        };
        let hook_ref = Some(&mut hook as &mut dyn FnMut(GenerateStage));
        match which {
            LoopCall::Generate => {
                let out = model
                    .generate_with_hook(batch, table, gcfg, ws, constants, hook_ref)
                    .unwrap();
                check_launches(table.table.device()).unwrap();
                out.validate().unwrap();
            }
            LoopCall::Packed => {
                let out = model
                    .generate_packed_with_hook(batch, table, gcfg, ws, constants, hook_ref)
                    .unwrap();
                check_launches(table.table.device()).unwrap();
                out.validate().unwrap();
            }
            LoopCall::Resident => {
                let resident = model
                    .generate_resident_with_hook(batch, table, gcfg, ws, constants, hook_ref)
                    .unwrap();
                check_launches(table.table.device()).unwrap();
                let out = resident.read(model).unwrap();
                out.validate().unwrap();
                resident.release_into(ws);
            }
        }
    }
    let whole = allocation_calls();
    (whole, last_step - start)
}

/// Build the T3 probe model: the tiny footprint config with the assignment
/// head (evidence-capable) and the enumeration artifacts uploaded, so the
/// same model serves the table and the enumerating source.
fn scratch_probe_setup(
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
    cfg.assignment = Some(AssignmentConfig::default());
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
    let mut model = Ms2Model::<R, f32>::init(&cfg, &device, &mut rng).unwrap();
    let domain = EnumDomain::from_compositions(comps.clone(), 0).unwrap();
    let bounds = RatioBounds::fit(comps.clone(), 0).unwrap();
    model.upload_enum_artifacts(&domain, &bounds, &device).unwrap();
    let constants = Ms2Constants::new(&device);
    (device, table, model, constants, comps)
}

fn scratch_probe_batch(comps: &[Composition], b: usize, seed: u64) -> SpectrumBatch {
    let precursors: Vec<u32> = (0..b)
        .map(|i| {
            mamba3::models::ms2::chem::composition_mass(&comps[i % comps.len()]).unwrap()
                + 1_007_825
                - 549
        })
        .collect();
    let ids: Vec<u64> = (0..b as u64).map(|i| 700 + i).collect();
    make_spectra(&ids, &precursors, 64, &vec![10u32; b], seed)
}

#[test]
fn warmed_decode_loop_allocates_zero() {
    let _serial = serial();
    // Task T3 acceptance, re-homed for task F10 item A1: after two warm-up
    // calls, the decode loop of one `generate` call allocates nothing with
    // the arena on — for `generate`, `generate_packed` and
    // `generate_resident`, with the table and the enumerating source.
    // Hooked calls run WITHOUT the arena (a hook may flip toggles
    // mid-call), so the hook-delimited loop probe below is the scratch-OFF
    // run: it proves the probe measures something (its loop allocates).
    // The arena-ON run is unhooked, and must come out exactly the OFF
    // whole-call count minus the OFF in-loop count — i.e. the loop holds
    // still. The launch count is identical either way (the arena changes no
    // launch).
    let (device, table, model, constants, comps) = scratch_probe_setup(8, 501);
    for source in [FormulaSource::Table, FormulaSource::Enumerate] {
        for which in [LoopCall::Generate, LoopCall::Packed, LoopCall::Resident] {
            let mut gcfg = tiny_generation();
            gcfg.trajectories = 4;
            gcfg.formula_source = source;
            let batch = scratch_probe_batch(&comps, 2, 502);
            // OFF (arena off, hooked): whole-call and in-loop allocs plus
            // launches. The loop must allocate (the probe measures).
            set_scratch_arena(false);
            let mut ws_off = GenerationWorkspace::new();
            for _ in 0..2 {
                measure_loop_allocs(&model, &batch, &table, &gcfg, &mut ws_off, &constants, which);
            }
            check_launches(&device).unwrap();
            reset_launch_count();
            let l0 = launch_count();
            let (whole_off, inside_off) =
                measure_loop_allocs(&model, &batch, &table, &gcfg, &mut ws_off, &constants, which);
            check_launches(&device).unwrap();
            let launches_off = launch_count() - l0;
            assert!(
                inside_off > 0,
                "the OFF probe must allocate in the loop (else it measures nothing)"
            );
            println!(
                "scratch off source {:?} {:?}: whole-call allocs {whole_off}, in-loop allocs {inside_off}, launches {launches_off}",
                source, which,
            );
            // ON (arena on, unhooked): the whole call must cost exactly the
            // OFF whole-call count minus the OFF in-loop count.
            set_scratch_arena(true);
            let mut ws_on = GenerationWorkspace::new();
            for _ in 0..2 {
                measure_whole_allocs(&model, &batch, &table, &gcfg, &mut ws_on, &constants, which);
            }
            check_launches(&device).unwrap();
            let (whole_on, launches_on) =
                measure_whole_allocs(&model, &batch, &table, &gcfg, &mut ws_on, &constants, which);
            check_launches(&device).unwrap();
            println!(
                "scratch on  source {:?} {:?}: whole-call allocs {whole_on}, launches {launches_on}",
                source, which,
            );
            assert_eq!(
                whole_on,
                whole_off - inside_off,
                "warmed {which:?} ({source:?}): the arena must save exactly the loop's {inside_off} allocs"
            );
            assert_eq!(
                launches_on, launches_off,
                "launch count moves with the scratch switch for {which:?} ({source:?})"
            );
        }
    }
    set_scratch_arena(true);
}

#[test]
fn decode_scratch_estimate_matches_arena_within_ten_percent() {
    let _serial = serial();
    // Task T3: `decode_scratch` (shape-derived) reconciles with the arena's
    // steady-state retention within 10%.
    set_scratch_arena(true);
    let (device, table, model, constants, comps) = scratch_probe_setup(8, 601);
    // (B, K, atoms, decoder_blocks, d_model, n_heads, head_dim, d_state):
    // the first three reuse the probe model, the last builds the V1-tiny
    // shape (Ld = 4) to pin the layer scaling of the estimate.
    for (b, k) in [(2usize, 4u32), (1usize, 8u32), (3usize, 2u32)] {
        let mut gcfg = tiny_generation();
        gcfg.trajectories = k;
        let batch = scratch_probe_batch(&comps, b, 602);
        let mut ws = GenerationWorkspace::new();
        for _ in 0..3 {
            let out = model.generate(&batch, &table, &gcfg, &mut ws, &constants).unwrap();
            check_launches(&device).unwrap();
            out.validate().unwrap();
        }
        let stats = ws.scratch_stats().expect("a bucket is cached");
        let est = Ms2MemoryEstimate::generation(
            &model.config,
            table.rows as u64,
            b as u64,
            k as u64,
            batch.n_raw as u64,
            gcfg.max_steps as u64,
            gcfg.formula_window as u64,
            gcfg.formulas as u64,
        )
        .unwrap();
        let want = est.get("decode_scratch").expect("the item exists");
        println!(
            "B={b} K={k}: arena holds {} bytes in {} buffers, estimate decode_scratch {want}",
            stats.bytes_held, stats.buffers,
        );
        if let Some(sizes) = ws.scratch_sizes() {
            println!("B={b} K={k}: arena sizes (bytes x count): {sizes:?}");
        }
        let ratio = stats.bytes_held as f64 / want as f64;
        assert!(
            (0.9..=1.1).contains(&ratio),
            "B={b} K={k}: arena bytes {} against estimate {want} (ratio {ratio:.3}) outside 10%",
            stats.bytes_held,
        );
    }
    // The V1-tiny shape (32 atoms, 4 decoder blocks): pins the layer
    // scaling with an independent geometry.
    {
        let device = mamba3::backend::Device::<R>::default();
        let mut cfg = v1_tiny_footprint_config();
        cfg.assignment = Some(AssignmentConfig::default());
        let comps: Vec<Composition> = vec![
            [2, 6, 0, 1, 0, 0, 0, 0, 0, 0],
            [6, 6, 0, 0, 0, 0, 0, 0, 0, 0],
            [3, 7, 1, 2, 0, 0, 0, 0, 0, 0],
        ];
        let host_table = FormulaTable::from_compositions(comps.clone()).unwrap();
        let table = DeviceFormulaTable::<R, f32>::upload(&host_table, &device).unwrap();
        cfg.formula_table.rows = table.rows as u32;
        cfg.formula_table.sha256 = table.sha256.clone();
        let mut rng = Rng::seeded(603);
        let model = Ms2Model::<R, f32>::init(&cfg, &device, &mut rng).unwrap();
        let constants = Ms2Constants::new(&device);
        let mut gcfg = tiny_generation();
        gcfg.max_steps = 42;
        gcfg.trajectories = 3;
        let batch = scratch_probe_batch(&comps, 2, 604);
        let mut ws = GenerationWorkspace::new();
        for _ in 0..3 {
            let out = model.generate(&batch, &table, &gcfg, &mut ws, &constants).unwrap();
            check_launches(&device).unwrap();
            out.validate().unwrap();
        }
        let stats = ws.scratch_stats().expect("a bucket is cached");
        let est = Ms2MemoryEstimate::generation(
            &model.config,
            table.rows as u64,
            2,
            3,
            batch.n_raw as u64,
            gcfg.max_steps as u64,
            gcfg.formula_window as u64,
            gcfg.formulas as u64,
        )
        .unwrap();
        let want = est.get("decode_scratch").expect("the item exists");
        println!(
            "V1-tiny B=2 K=3: arena holds {} bytes in {} buffers, estimate decode_scratch {want}",
            stats.bytes_held, stats.buffers,
        );
        if let Some(sizes) = ws.scratch_sizes() {
            println!("V1-tiny B=2 K=3: arena sizes (bytes x count): {sizes:?}");
        }
        let ratio = stats.bytes_held as f64 / want as f64;
        assert!(
            (0.9..=1.1).contains(&ratio),
            "V1-tiny: arena bytes {} against estimate {want} (ratio {ratio:.3}) outside 10%",
            stats.bytes_held,
        );
    }
    // Adequacy independent of the cap: the bucket arena above is capped at
    // the estimate itself, so a too-small estimate would truncate the
    // retention to the cap and pass vacuously. Drive the decode loop once
    // more inside an UNCAPPED arena (the inner bucket scopes are refused
    // while it is active, so every step transient lands in the outer
    // arena) and compare what it retains with the estimate.
    {
        let mut gcfg = tiny_generation();
        gcfg.trajectories = 4;
        let batch = scratch_probe_batch(&comps, 2, 606);
        let mut ws = GenerationWorkspace::new();
        for _ in 0..2 {
            let out = model.generate(&batch, &table, &gcfg, &mut ws, &constants).unwrap();
            check_launches(&device).unwrap();
            out.validate().unwrap();
        }
        let pre = model.generate_preflight(&batch, &table, &gcfg).unwrap();
        let spectra = model.generate_preprocess(&batch, &gcfg, &device).unwrap();
        let encoded = model
            .generate_encode_ws(&mut ws, &spectra, gcfg.control, &pre, &device)
            .unwrap();
        model
            .generate_search_ws(
                &mut ws,
                &spectra,
                &batch,
                &encoded.pool,
                &table,
                pre.spectra_n,
                pre.trajectories,
                pre.formulas,
                false,
                gcfg.formula_rows_visited_max,
                gcfg.formula_rows_scored_max,
                &gcfg,
                &pre,
                &device,
            )
            .unwrap();
        let (mut state, bonds, traj) = model
            .generate_decoder_init_ws(&mut ws, &encoded, &pre, &device)
            .unwrap();
        assert!(
            state.carries_in_place(),
            "the uncapped probe runs the production in-place path"
        );
        let seed_lo = (gcfg.seed & 0xFFFF_FFFF) as u32;
        let seed_hi = (gcfg.seed >> 32) as u32;
        let uncapped = ScratchArena::new(usize::MAX);
        with_scratch(&uncapped, || {
            for step in 1..pre.steps {
                model
                    .generate_decode_step_ws(
                        &mut ws,
                        &encoded,
                        &traj,
                        &mut state,
                        &bonds,
                        &constants.atom_table,
                        step,
                        seed_lo,
                        seed_hi,
                        gcfg.temperature,
                        pre.trajectories,
                        &pre,
                        &device,
                    )
                    .unwrap();
            }
        });
        check_launches(&device).unwrap();
        let held = uncapped.stats().bytes_held;
        let est = Ms2MemoryEstimate::generation(
            &model.config,
            table.rows as u64,
            2,
            4,
            batch.n_raw as u64,
            gcfg.max_steps as u64,
            gcfg.formula_window as u64,
            gcfg.formulas as u64,
        )
        .unwrap();
        let want = est.get("decode_scratch").expect("the item exists");
        println!("uncapped B=2 K=4: decode loop retains {held} bytes, estimate decode_scratch {want}");
        let ratio = held as f64 / want as f64;
        assert!(
            (0.9..=1.1).contains(&ratio),
            "uncapped: decode loop retains {held} against estimate {want} (ratio {ratio:.3}) outside 10%",
        );
    }
    set_scratch_arena(true);
}

#[test]
fn scratch_on_equals_off() {
    let _serial = serial();
    // Task T3: `generate` with the arena ON equals OFF — every integer
    // field bit-for-bit, float fields bit-for-bit on the cpu runtime
    // (within 1e-6 on a GPU backend) — for B.K in {1.8, 8.8, 8.32}, both
    // formula sources, evidence on, graph identity.
    let (device, table, model, constants, comps) = scratch_probe_setup(8, 701);
    let gpu = device.name() != "cpu";
    for (b, k) in [(1usize, 8u32), (8usize, 8u32), (8usize, 32u32)] {
        for source in [FormulaSource::Table, FormulaSource::Enumerate] {
            let mut gcfg = tiny_generation();
            gcfg.trajectories = k;
            gcfg.formula_source = source;
            gcfg.evidence = true;
            gcfg.identity = IdentityMode::Graph;
            let batch = scratch_probe_batch(&comps, b, 702);
            let mut outs = Vec::new();
            let mut served = Vec::new();
            for on in [false, true] {
                set_scratch_arena(on);
                let mut ws = GenerationWorkspace::new();
                // Two warm-ups settle kernel autotune identically; the
                // third call is the compared one.
                for _ in 0..2 {
                    let out = model
                        .generate(&batch, &table, &gcfg, &mut ws, &constants)
                        .unwrap();
                    check_launches(&device).unwrap();
                    out.validate().unwrap();
                }
                let out = model
                    .generate(&batch, &table, &gcfg, &mut ws, &constants)
                    .unwrap();
                check_launches(&device).unwrap();
                out.validate().unwrap();
                served.push(ws.scratch_stats().map(|s| s.served).unwrap_or(0));
                outs.push(out);
            }
            assert_eq!(outs.len(), 2);
            assert_eq!(served.len(), 2);
            assert!(
                served[1] > 0,
                "B={b} K={k} {source:?}: the ON call must observe recycling (served {})",
                served[1]
            );
            let (off, on) = (&outs[0], &outs[1]);
            assert_eq!(on.actions, off.actions, "B={b} K={k} {source:?}: actions");
            assert_eq!(on.length, off.length, "B={b} K={k} {source:?}: length");
            assert_eq!(on.status, off.status, "B={b} K={k} {source:?}: status");
            assert_eq!(on.formula_row, off.formula_row, "B={b} K={k} {source:?}: formula_row");
            assert_eq!(
                on.open_valence, off.open_valence,
                "B={b} K={k} {source:?}: open_valence"
            );
            assert_eq!(
                on.request_status, off.request_status,
                "B={b} K={k} {source:?}: request_status"
            );
            assert_eq!(
                on.identity_resolution, off.identity_resolution,
                "B={b} K={k} {source:?}: identity_resolution"
            );
            let tol = if gpu { 1e-6 } else { 0.0 };
            for (name, a, c) in [
                ("trace_log_prob", &on.trace_log_prob, &off.trace_log_prob),
                ("formula_log_prob", &on.formula_log_prob, &off.formula_log_prob),
                (
                    "formula_mass_retained",
                    &on.formula_mass_retained,
                    &off.formula_mass_retained,
                ),
                (
                    "intensity_retained",
                    &on.intensity_retained,
                    &off.intensity_retained,
                ),
            ] {
                assert_eq!(a.len(), c.len(), "B={b} K={k} {source:?}: {name} length");
                for (i, (x, y)) in a.iter().zip(c.iter()).enumerate() {
                    assert!(
                        (x - y).abs() <= tol * (1.0 + x.abs().max(y.abs())),
                        "B={b} K={k} {source:?}: {name}[{i}] on {x} against off {y}"
                    );
                }
            }
            println!("B={b} K={k} {source:?}: scratch ON equals OFF");
        }
    }
    set_scratch_arena(true);
}

#[test]
fn scratch_alternating_buckets_stay_correct_and_flat() {
    let _serial = serial();
    // Task T3: alternating buckets (B = 1 and B = 8 interleaved, 20 calls)
    // stay correct, and 200 repeated calls leave reserved bytes flat.
    set_scratch_arena(true);
    let (device, table, model, constants, comps) = scratch_probe_setup(8, 801);
    let mut gcfg = tiny_generation();
    gcfg.trajectories = 8;
    let batch_a = scratch_probe_batch(&comps, 1, 802);
    let batch_b = scratch_probe_batch(&comps, 8, 803);
    let mut ws = GenerationWorkspace::new();
    // Reference outputs with the arena off.
    set_scratch_arena(false);
    let mut refs = Vec::new();
    for batch in [&batch_a, &batch_b] {
        let out = model.generate(batch, &table, &gcfg, &mut ws, &constants).unwrap();
        check_launches(&device).unwrap();
        out.validate().unwrap();
        refs.push((out.actions.clone(), out.length.clone(), out.status.clone()));
    }
    // Interleaved with the arena on: identical trajectories.
    set_scratch_arena(true);
    for _ in 0..20 {
        for (batch, (actions, length, status)) in [&batch_a, &batch_b].iter().zip(refs.iter()) {
            let out = model.generate(batch, &table, &gcfg, &mut ws, &constants).unwrap();
            check_launches(&device).unwrap();
            out.validate().unwrap();
            assert_eq!(&out.actions, actions, "alternating buckets: actions");
            assert_eq!(&out.length, length, "alternating buckets: length");
            assert_eq!(&out.status, status, "alternating buckets: status");
        }
    }
    // 200 repeated calls leave reserved bytes flat.
    let base_mem = mamba3::backend::reserved_bytes(&device);
    for _ in 0..200 {
        let out = model.generate(&batch_a, &table, &gcfg, &mut ws, &constants).unwrap();
        out.validate().unwrap();
    }
    check_launches(&device).unwrap();
    let end_mem = mamba3::backend::reserved_bytes(&device);
    if let (Some(a), Some(b)) = (base_mem, end_mem) {
        assert_eq!(a, b, "reserved bytes constant over 200 calls with scratch on");
        println!("reserved bytes over 200 scratch calls: {a}");
    }
    assert!(
        ws.scratch_stats().map(|s| s.served).unwrap_or(0) > 0,
        "the interleaved loop must observe recycling (else this test cannot tell it is on)"
    );
    set_scratch_arena(true);
}

/// `set_fused_step(false)` with restoration on drop: the "in-place
/// unsupported" mode below is process-global, so it is always restored,
/// even on panic.
struct FusedStepOnRestore;

impl Drop for FusedStepOnRestore {
    fn drop(&mut self) {
        set_fused_step(true);
    }
}

/// Which decode execution mode a carry-measurement case runs.
#[derive(Debug, Clone, Copy)]
enum CarryMode {
    /// Production: fused step, nothing captured.
    InPlace,
    /// `composed_step = true`.
    Composed,
    /// `composed_step = true` with carry capture on.
    ComposedCapture,
    /// Carry capture on.
    Capture,
    /// In-place unsupported (`MAMBA3_FUSED_STEP=0` equivalent).
    Unsupported,
}

impl CarryMode {
    /// The workspace flags of this mode.
    fn flags(self) -> (bool, bool) {
        match self {
            CarryMode::InPlace | CarryMode::Unsupported => (false, false),
            CarryMode::Composed => (true, false),
            CarryMode::ComposedCapture => (true, true),
            CarryMode::Capture => (false, true),
        }
    }
}

#[test]
fn decode_carry_estimate_covers_live_carries_per_mode() {
    let _serial = serial();
    // T3F finding 2 (task F10 item A2): the carry estimate follows the
    // execution mode. For each mode the live carry bytes across one decode
    // step are at most the estimate's carry items (`decoder_carries` +
    // `decode_functional_step`), and on the functional paths the live
    // bytes exceed the one-bank number — so the test proves the selection
    // matters. Every staged call carries its own latched mode (task F10
    // item A3): the preflight is mode-explicit, and the latched
    // `decode_in_place` is asserted to agree with the built state, so
    // generator and estimate cannot drift.
    set_scratch_arena(false);
    // The 4-layer probe (`v1_tiny_footprint_config`): carries dominate the
    // step's live bytes here (heads and mixer transients are a fraction of
    // one bank), so the allocation high-water mark below is the step's
    // carry peak to within a small, printed non-carry remainder.
    let (device, table, model, constants, comps) = gen_setup_v1(901);
    assert!(
        model.config.decoder.conv_kernel.is_none(),
        "the tiny probe has no convolution (conv_history 0 below)"
    );
    let b = 2usize;
    let k = 4u32;
    let mut gcfg = tiny_generation();
    gcfg.trajectories = k;
    gcfg.max_steps = 42;
    let precursors = precursors_for(&comps, b);
    let ids: Vec<u64> = (0..b as u64).map(|i| 950 + i).collect();
    let batch = make_spectra(&ids, &precursors, 64, &vec![10u32; b], 902);
    let elem = 4u64;
    let dec = &model.config.decoder;
    let one_bank = carry_bytes(
        b as u64,
        k as u64,
        u64::from(model.config.decoder_blocks),
        dec.n_heads as u64,
        dec.head_dim as u64,
        dec.d_state as u64,
        0,
        elem,
    )
    .unwrap();
    for mode in [
        CarryMode::InPlace,
        CarryMode::Composed,
        CarryMode::ComposedCapture,
        CarryMode::Capture,
        CarryMode::Unsupported,
    ] {
        let (composed, capture) = mode.flags();
        let mut ws = GenerationWorkspace::new();
        ws.composed_step = composed;
        ws.capture_carry_trace = capture;
        let _restore = match mode {
            CarryMode::Unsupported => {
                set_fused_step(false);
                Some(FusedStepOnRestore)
            }
            _ => None,
        };
        let pre = model
            .generate_preflight_with_decode_mode(&batch, &table, &gcfg, composed, capture)
            .unwrap();
        let spectra = model.generate_preprocess(&batch, &gcfg, &device).unwrap();
        let encoded = model
            .generate_encode_ws(&mut ws, &spectra, gcfg.control, &pre, &device)
            .unwrap();
        model
            .generate_search_ws(
                &mut ws,
                &spectra,
                &batch,
                &encoded.pool,
                &table,
                pre.spectra_n,
                pre.trajectories,
                pre.formulas,
                false,
                gcfg.formula_rows_visited_max,
                gcfg.formula_rows_scored_max,
                &gcfg,
                &pre,
                &device,
            )
            .unwrap();
        let (mut state, bonds, traj) = model
            .generate_decoder_init_ws(&mut ws, &encoded, &pre, &device)
            .unwrap();
        let in_place = state.carries_in_place();
        assert_eq!(
            pre.decode_in_place, in_place,
            "{mode:?}: the latched preflight mode agrees with the built state"
        );
        assert_eq!(
            model.decoder.steps_carries_in_place(&device, composed, capture),
            in_place,
            "{mode:?}: the shared predicate agrees with the built state"
        );
        assert_eq!(
            in_place,
            matches!(mode, CarryMode::InPlace),
            "{mode:?}: only the production mode steps in place"
        );
        let est = Ms2MemoryEstimate::generation_for_decode_mode(
            &model.config,
            in_place,
            table.rows as u64,
            b as u64,
            k as u64,
            batch.n_raw as u64,
            gcfg.max_steps as u64,
            gcfg.formula_window as u64,
            gcfg.formulas as u64,
        )
        .unwrap();
        let carry_items =
            est.get("decoder_carries").unwrap() + est.get("decode_functional_step").unwrap();
        let seed_lo = (gcfg.seed & 0xFFFF_FFFF) as u32;
        let seed_hi = (gcfg.seed >> 32) as u32;
        let run_step = |ws: &mut GenerationWorkspace<R, f32>, state| {
            model
                .generate_decode_step_ws(
                    ws,
                    &encoded,
                    &traj,
                    state,
                    &bonds,
                    &constants.atom_table,
                    1,
                    seed_lo,
                    seed_hi,
                    gcfg.temperature,
                    pre.trajectories,
                    &pre,
                    &device,
                )
                .unwrap();
        };
        let live_bytes: u64 = if in_place {
            run_step(&mut ws, &mut state);
            check_launches(&device).unwrap();
            // The live in-place recurrent tensors after the step: `h` and
            // the just-written `act`/`bc` factors (whose outer product is
            // the `last_u` the functional step returns), with the angle
            // when the layer carries one.
            let carries = state.in_place_carries().expect("carries are in place");
            carries
                .iter()
                .map(|c| {
                    (c.h.len()
                        + c.act.len()
                        + c.bc.len()
                        + c.angle.as_ref().map(|a| a.len()).unwrap_or(0)
                        + c.history.as_ref().map(|h| h.len()).unwrap_or(0))
                        as u64
                        * elem
                })
                .sum()
        } else {
            // Hold the old bank alive across the step: afterwards the old
            // and the new (frozen) banks are live together, which is the
            // peak the one-bank number misses. The freeze runs in place
            // (task F10 item A2), so this post-step sum IS the step's carry
            // peak — and the allocation high-water mark below proves no
            // hidden carry-sized temporary joins it during the step.
            let old = state.caches.clone();
            let old_elems: usize = old.iter().map(|c| c.num_elements()).sum();
            let base = live_bytes();
            reset_live_high_water();
            run_step(&mut ws, &mut state);
            check_launches(&device).unwrap();
            // The step's transient high-water above the pre-step baseline:
            // the new bank, the heads, the mixer/sampling temporaries, and
            // the step's internal old-bank clones (which stand in, in the
            // counting, for the originals the new bank replaces). It bounds
            // the step's live peak from above — every live value counts —
            // and must still fit the carry items: that is the during-step
            // proof that the freeze adds no hidden temporary.
            let step_delta = live_high_water_bytes().saturating_sub(base);
            let new_elems: usize = state.caches.iter().map(|c| c.num_elements()).sum();
            let old_new = (old_elems + new_elems) as u64 * elem;
            println!(
                "mode {mode:?}: step live delta {step_delta}, old+new {old_new}"
            );
            assert!(
                step_delta >= new_elems as u64 * elem,
                "{mode:?}: live high-water {step_delta} misses the new bank (the counter misses step allocations)"
            );
            assert!(
                step_delta <= carry_items,
                "{mode:?}: during-step live peak {step_delta} exceeds the estimate carry items {carry_items}"
            );
            old_new
        };
        println!(
            "mode {mode:?}: live carry bytes {live_bytes}, estimate carry items {carry_items} (one bank {one_bank})"
        );
        assert!(
            live_bytes <= carry_items,
            "{mode:?}: live carry bytes {live_bytes} exceed the estimate carry items {carry_items}"
        );
        if in_place {
            assert_eq!(est.get("decoder_carries"), Some(one_bank));
            assert_eq!(est.get("decode_functional_step"), Some(0));
        } else {
            assert_eq!(est.get("decoder_carries"), Some(2 * one_bank));
            assert_eq!(est.get("decode_functional_step"), Some(one_bank));
            assert!(
                live_bytes > one_bank,
                "{mode:?}: live carry bytes {live_bytes} do not exceed the one-bank number {one_bank} (the selection would not matter)"
            );
        }
    }
    set_scratch_arena(true);
}

#[test]
fn hooked_call_skips_arena_and_matches_scratch_off() {
    let _serial = serial();
    // Task F10 item A1: a `generate_with_hook` call with a hook installed
    // runs WITHOUT the arena (plain allocations, exactly as with
    // `MAMBA3_MS2_SCRATCH=0`), so a hook that flips toggles mid-call cannot
    // change the running call's allocation behaviour — and its results are
    // bit-identical to the scratch-off reference.
    let (device, table, model, constants, comps) = gen_setup(8, 911);
    let b = 1usize;
    let mut gcfg = tiny_generation();
    gcfg.trajectories = 2;
    gcfg.max_steps = 14;
    let precursors = precursors_for(&comps, b);
    let batch = make_spectra(&[960], &precursors, 64, &vec![10u32; b], 912);
    // Scratch-off reference: two calls (warmup, measured) with the arena
    // globally off.
    set_scratch_arena(false);
    let mut ws_ref = GenerationWorkspace::new();
    let ref_first = model
        .generate(&batch, &table, &gcfg, &mut ws_ref, &constants)
        .unwrap();
    check_launches(&device).unwrap();
    reset_transfer_counters();
    let ref_out = model
        .generate(&batch, &table, &gcfg, &mut ws_ref, &constants)
        .unwrap();
    check_launches(&device).unwrap();
    let ref_allocs = allocation_calls();
    drop(ref_first);
    // Hooked call with the arena globally on: same in-loop allocations as
    // scratch-off, bit-identical results.
    set_scratch_arena(true);
    let mut ws_hook = GenerationWorkspace::new();
    let mut stages = 0usize;
    let mut hook = |_: GenerateStage| {
        stages += 1;
    };
    model
        .generate_with_hook(&batch, &table, &gcfg, &mut ws_hook, &constants, Some(&mut hook))
        .unwrap();
    check_launches(&device).unwrap();
    reset_transfer_counters();
    let hook_out = model
        .generate_with_hook(&batch, &table, &gcfg, &mut ws_hook, &constants, Some(&mut hook))
        .unwrap();
    check_launches(&device).unwrap();
    let hook_allocs = allocation_calls();
    assert!(stages > 0, "the hook observed the call");
    assert_eq!(
        hook_allocs, ref_allocs,
        "a hooked call performs the same number of allocations as scratch-off (the arena is not used)"
    );
    assert_eq!(hook_out.actions, ref_out.actions, "hooked actions bit-identical");
    assert_eq!(hook_out.length, ref_out.length, "hooked lengths bit-identical");
    assert_eq!(hook_out.status, ref_out.status, "hooked status bit-identical");
    assert_eq!(
        hook_out.trace_log_prob, ref_out.trace_log_prob,
        "hooked trace log-probs bit-identical"
    );
    // The unhooked call with the arena on allocates strictly less: the
    // arena serves the loop's transients.
    let mut ws_plain = GenerationWorkspace::new();
    model
        .generate(&batch, &table, &gcfg, &mut ws_plain, &constants)
        .unwrap();
    check_launches(&device).unwrap();
    reset_transfer_counters();
    model
        .generate(&batch, &table, &gcfg, &mut ws_plain, &constants)
        .unwrap();
    check_launches(&device).unwrap();
    let plain_allocs = allocation_calls();
    println!("allocs: scratch-off {ref_allocs}, hooked {hook_allocs}, arena {plain_allocs}");
    assert!(
        plain_allocs < hook_allocs,
        "the unhooked arena call allocates less than the hooked call ({plain_allocs} vs {hook_allocs})"
    );
    set_scratch_arena(true);
}

#[test]
fn scratch_external_refs_zero_after_each_entry_point() {
    let _serial = serial();
    // Task F10 item A1: after each of the three entry points returns, its
    // arena reports zero externally referenced buffers — no scratch-backed
    // tensor escaped the owning call.
    set_scratch_arena(true);
    let (device, table, model, constants, comps) = gen_setup(8, 913);
    let b = 1usize;
    let mut gcfg = tiny_generation();
    gcfg.trajectories = 2;
    gcfg.max_steps = 14;
    let precursors = precursors_for(&comps, b);
    let batch = make_spectra(&[970], &precursors, 64, &vec![10u32; b], 914);
    let mut ws = GenerationWorkspace::new();
    for _ in 0..2 {
        model.generate(&batch, &table, &gcfg, &mut ws, &constants).unwrap();
    }
    check_launches(&device).unwrap();
    assert_eq!(
        ws.scratch_external_refs(),
        0,
        "generate leaves no externally referenced scratch buffer"
    );
    for _ in 0..2 {
        model.generate_packed(&batch, &table, &gcfg, &mut ws, &constants).unwrap();
    }
    check_launches(&device).unwrap();
    assert_eq!(
        ws.scratch_external_refs(),
        0,
        "generate_packed leaves no externally referenced scratch buffer"
    );
    for _ in 0..2 {
        model
            .generate_resident(&batch, &table, &gcfg, &mut ws, &constants)
            .unwrap()
            .release_into(&mut ws);
    }
    check_launches(&device).unwrap();
    assert_eq!(
        ws.scratch_external_refs(),
        0,
        "generate_resident leaves no externally referenced scratch buffer"
    );
    // The leased result itself: its bucket travels with the lease, and it
    // is clean too.
    let resident = model
        .generate_resident(&batch, &table, &gcfg, &mut ws, &constants)
        .unwrap();
    assert_eq!(
        resident.scratch_external_refs(),
        0,
        "the leased resident result holds no externally referenced scratch buffer"
    );
    resident.release_into(&mut ws);
    set_scratch_arena(true);
}

#[test]
fn latched_mode_survives_mid_call_toggle() {
    let _serial = serial();
    // Task F10 item A3: preflight, decoder init and every step of one
    // generate call use ONE latched mode. An `AfterEncoder` hook that calls
    // `set_fused_step(false)` cannot change the running call: it keeps the
    // preflighted in-place mode (same launches and bit-identical results as
    // the unhooked call — the carries are provably still stepped in place,
    // since any other mode launches differently). The NEXT call uses the
    // functional mode and its preflight estimate.
    let _restore = FusedStepOnRestore;
    set_fused_step(true);
    let (device, table, model, constants, comps) = gen_setup(8, 915);
    let b = 1usize;
    let mut gcfg = tiny_generation();
    gcfg.trajectories = 2;
    gcfg.max_steps = 14;
    let precursors = precursors_for(&comps, b);
    let batch = make_spectra(&[980], &precursors, 64, &vec![10u32; b], 916);
    // Unhooked in-place reference (warmed).
    let mut ws_ref = GenerationWorkspace::new();
    for _ in 0..2 {
        model.generate(&batch, &table, &gcfg, &mut ws_ref, &constants).unwrap();
    }
    check_launches(&device).unwrap();
    reset_launch_count();
    let ref_out = model
        .generate(&batch, &table, &gcfg, &mut ws_ref, &constants)
        .unwrap();
    check_launches(&device).unwrap();
    let ref_launches = launch_count();
    // Hooked call flipping the toggle at AfterEncoder.
    let mut ws_hook = GenerationWorkspace::new();
    for _ in 0..2 {
        let mut warm_hook = |_: GenerateStage| {};
        model
            .generate_with_hook(
                &batch,
                &table,
                &gcfg,
                &mut ws_hook,
                &constants,
                Some(&mut warm_hook),
            )
            .unwrap();
    }
    check_launches(&device).unwrap();
    reset_launch_count();
    let mut hook = |stage: GenerateStage| {
        if stage == GenerateStage::AfterEncoder {
            set_fused_step(false);
        }
    };
    let hook_out = model
        .generate_with_hook(&batch, &table, &gcfg, &mut ws_hook, &constants, Some(&mut hook))
        .unwrap();
    check_launches(&device).unwrap();
    let hook_launches = launch_count();
    assert_eq!(
        hook_launches, ref_launches,
        "the running call keeps its preflighted in-place mode despite the mid-call toggle"
    );
    assert_eq!(hook_out.actions, ref_out.actions, "toggled actions bit-identical");
    assert_eq!(
        hook_out.trace_log_prob, ref_out.trace_log_prob,
        "toggled trace log-probs bit-identical"
    );
    // The toggle is still off: the next call's preflight prices the
    // functional mode, and the call runs it.
    let pre_next = model.generate_preflight(&batch, &table, &gcfg).unwrap();
    assert!(
        !pre_next.decode_in_place,
        "the next call latches the functional mode"
    );
    let mut ws_next = GenerationWorkspace::new();
    reset_launch_count();
    let next_out = model
        .generate(&batch, &table, &gcfg, &mut ws_next, &constants)
        .unwrap();
    check_launches(&device).unwrap();
    next_out.validate().unwrap();
    let next_launches = launch_count();
    assert_ne!(
        next_launches, ref_launches,
        "the next call runs the functional mode (launches {next_launches} vs in-place {ref_launches})"
    );
}
