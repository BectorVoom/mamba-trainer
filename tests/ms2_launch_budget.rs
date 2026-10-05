//! Launch-budget reconciliation (A1 finding 8): with decoder initialisation
//! measured in its own hook window, `L_call = L_preprocess + L_encoder +
//! L_search + L_init + (T-1)*L_step + L_finalize` equals the measured warmed
//! total exactly, for two shapes.
//!
//! This is the only test in its binary on purpose. It reads the
//! process-wide launch counter, and any test running beside it would add to
//! it — a false failure that says nothing about the budget. Cargo gives each
//! integration test file its own process, so keeping the counter reader
//! alone is what makes the numbers mean what they say.

#![cfg(feature = "backend")]

use mamba3::backend::{
    Device, launch_count, read_count, reset_launch_count, reset_read_count,
    reset_transfer_counters, upload_bytes,
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
use mamba3::models::ms2::generate::{GenerateStage, GenerationWorkspace, Ms2Model};
use mamba3::tensor::ops::ms2::Ms2Constants;
use mamba3::tensor::ops::random::Rng;

type R = Auto;

fn dev() -> Device<R> {
    Device::<R>::default()
}

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
fn launch_budget_reconciles_exactly_for_two_shapes() {
    let _serial = serial();
    // Two bucket shapes (B = 1 and B = 2, same tiny model): every hook
    // boundary fires in pipeline order, each decode step costs the
    // independently measured T+1-minus-T slope, the readout interval holds
    // exactly the single batched read, and the stage launches sum to the
    // warmed-call total with difference 0.
    let device = dev();
    let comps: Vec<Composition> = vec![
        [6, 6, 0, 0, 0, 0, 0, 0, 0, 0],
        [3, 7, 1, 2, 0, 0, 0, 0, 0, 0],
    ];
    let host_table = FormulaTable::from_compositions(comps.clone()).unwrap();
    let mut cfg = tiny_config();
    let table = DeviceFormulaTable::<R, f32>::upload(&host_table, &device).unwrap();
    cfg.formula_table.rows = table.rows as u32;
    cfg.formula_table.sha256 = table.sha256.clone();
    let mut rng = Rng::seeded(5);
    let model = Ms2Model::<R, f32>::init(&cfg, &device, &mut rng).unwrap();
    let constants = Ms2Constants::new(&device);
    let mut ws = GenerationWorkspace::new();
    let gcfg = tiny_generation();
    let decode_steps = (gcfg.max_steps as usize - 1).max(1);
    let precursors: Vec<u32> = comps
        .iter()
        .map(|c| mamba3::models::ms2::chem::composition_mass(c).unwrap() + 1_007_825 - 549)
        .collect();
    // The complete ordered stage sequence of one hooked call: preprocess,
    // encoder, search, decoder init, one boundary per decode step, validate,
    // readout. Any reordered, added or dropped boundary fails here.
    let mut expected: Vec<GenerateStage> = vec![
        GenerateStage::AfterPreprocess,
        GenerateStage::AfterEncoder,
        GenerateStage::AfterSearch,
        GenerateStage::AfterDecoderInit,
    ];
    for step in 1..=decode_steps {
        expected.push(GenerateStage::AfterDecodeStep(step));
    }
    expected.push(GenerateStage::AfterValidate);
    expected.push(GenerateStage::AfterReadout);

    for (b, seed) in [(1usize, 21u64), (2usize, 22u64)] {
        let ids: Vec<u64> = (0..b as u64).map(|i| 800 + i).collect();
        let counts = vec![10u32; b];
        let batch = make_spectra(&ids, &precursors[..b], 64, &counts, seed);
        for _ in 0..2 {
            model
                .generate(&batch, &table, &gcfg, &mut ws, &constants)
                .unwrap();
        }
        device.synchronize();
        // Warmed total: drain, reset, one call, drain, read.
        device.synchronize();
        reset_launch_count();
        reset_read_count();
        reset_transfer_counters();
        let plain = model
            .generate(&batch, &table, &gcfg, &mut ws, &constants)
            .unwrap();
        device.synchronize();
        let total = launch_count();
        // Instrumented call: the stage at each hook boundary, in order, with
        // launch and host-read snapshots (launch counts are host-side, so no
        // sync is needed for exactness).
        device.synchronize();
        reset_launch_count();
        reset_read_count();
        let mut stages: Vec<GenerateStage> = Vec::new();
        let mut launches: Vec<usize> = Vec::new();
        let mut reads: Vec<usize> = Vec::new();
        let mut uploads: Vec<u64> = Vec::new();
        {
            let mut hook = |stage: GenerateStage| {
                stages.push(stage);
                launches.push(launch_count());
                reads.push(read_count());
                uploads.push(upload_bytes());
            };
            let hooked = model
                .generate_with_hook(&batch, &table, &gcfg, &mut ws, &constants, Some(&mut hook))
                .unwrap();
            assert_eq!(hooked, plain, "b={b}: the hooked call is production");
        }
        let end_launches = launch_count();
        let end_reads = read_count();
        assert_eq!(
            stages, expected,
            "b={b}: the complete ordered stage sequence fires"
        );
        let stage_launches = |i: usize, prev: usize| launches[i] - prev;
        let l_pre = stage_launches(0, 0);
        let l_enc = stage_launches(1, launches[0]);
        let l_search = stage_launches(2, launches[1]);
        let l_init = stage_launches(3, launches[2]);
        let mut l_steps: Vec<usize> = Vec::with_capacity(decode_steps);
        for i in 0..decode_steps {
            l_steps.push(stage_launches(4 + i, launches[3 + i]));
        }
        let l_finalize = stage_launches(4 + decode_steps, launches[3 + decode_steps]);
        let l_readout_launches = stage_launches(5 + decode_steps, launches[4 + decode_steps]);
        let l_readout_reads = reads[5 + decode_steps] - reads[4 + decode_steps];
        // The readout interval (AfterValidate to AfterReadout) holds exactly
        // the single batched `read_all`: 1 read, 0 launches. The final
        // snapshot closes the window, so nothing after the readout hook can
        // hide there.
        assert_eq!(
            l_readout_launches, 0,
            "b={b}: the readout interval launches nothing"
        );
        assert_eq!(
            l_readout_reads, 1,
            "b={b}: the readout interval performs exactly the single batched read"
        );
        assert_eq!(
            end_launches - launches[5 + decode_steps],
            0,
            "b={b}: nothing launches after the readout hook"
        );
        assert_eq!(
            end_reads - reads[5 + decode_steps],
            0,
            "b={b}: nothing reads after the readout hook"
        );
        // Every stage that must do work does some: a zero here means a stage
        // went missing or silent. Preprocess is upload-only on a warmed call
        // (the request upload launches no kernel), so it pins uploads rather
        // than launches.
        assert!(
            uploads[0] > 0,
            "b={b}: preprocess uploads {} bytes, want > 0",
            uploads[0]
        );
        for (name, count) in [
            ("encoder", l_enc),
            ("search", l_search),
            ("decoder_init", l_init),
            ("finalize", l_finalize),
        ] {
            assert!(count > 0, "b={b}: stage {name} launches {count}, want > 0");
        }
        // The independently measured T+1-minus-T slope: two warmups plus one
        // measured warmed call at T+1 steps (a new bucket), minus the warmed
        // T-step total. Every per-step delta must equal it — not just their
        // sum — so no step can hide work in another.
        let mut gcfg_long = gcfg.clone();
        gcfg_long.max_steps += 1;
        for _ in 0..2 {
            model
                .generate(&batch, &table, &gcfg_long, &mut ws, &constants)
                .unwrap();
        }
        device.synchronize();
        reset_launch_count();
        model
            .generate(&batch, &table, &gcfg_long, &mut ws, &constants)
            .unwrap();
        device.synchronize();
        let long_total = launch_count();
        let l_step = long_total - total;
        assert!(
            l_step > 0,
            "b={b}: the per-step slope is {l_step}, want > 0"
        );
        for (i, got) in l_steps.iter().enumerate() {
            assert_eq!(
                *got,
                l_step,
                "b={b}: decode step {} launches {got}, want the T+1-minus-T slope {l_step}",
                i + 1,
            );
        }
        // The hooked call launches exactly what the plain warmed call does:
        // the hook itself adds no work (the pre-existing independent check).
        assert_eq!(
            end_launches, total,
            "b={b}: plain versus hooked totals agree"
        );
        // Pinned per-stage numbers for this exact tiny config on the CPU
        // runtime (both shapes): preprocess 0 (upload-only), encoder 153,
        // search 33 (32 V0 launches plus the one `ms2_allocate` launch of
        // I2), decoder init 27, every decode step 20, finalize 1,
        // readout 0 launches / 1 read. Moving launches across a stage
        // boundary — e.g. charging the 13 decoder-init launches to search —
        // fails here rather than drifting silently. (The fused step of P8/O4
        // moved these from 7 and 104 to 13 and 36: six more launches build
        // the per-call tables once, and the step itself is one kernel per
        // stage. The pointer products computed once per call instead of once
        // per step, the fused mixer step and the paired carry freeze then
        // moved them to 26 and 22, and carries stepped in place — nothing
        // to freeze, one more buffer zeroed at init — to 27 and 20.) On any
        // other backend the same
        // structural assertions above still run, but the numeric pins are
        // backend-specific: print one line per stage,
        // `STAGE-LAUNCHES <backend> <shape> <stage> <n>`, and skip the numeric
        // equality (the supervisor pins the wgpu numbers from that output,
        // the way the footprint pins are keyed).
        let backend = device.name();
        let shape = format!("b{b}");
        // Printed on every backend, before the pins: a changed count is read
        // off this line instead of one failed assertion at a time.
        println!(
            "STAGE-LAUNCHES {backend} {shape} preprocess {l_pre} encoder {l_enc} search {l_search} decoder_init {l_init} step {l_step} finalize {l_finalize}"
        );
        if backend == "cpu" {
            assert_eq!(
                l_pre, 0,
                "b={b}: preprocess launches nothing on a warmed call"
            );
            assert_eq!(l_enc, 153, "b={b}: encoder launches 153");
            assert_eq!(l_search, 33, "b={b}: search launches 33");
            assert_eq!(l_init, 27, "b={b}: decoder init launches 27");
            assert_eq!(l_finalize, 1, "b={b}: finalize launches 1");
            assert_eq!(l_step, 20, "b={b}: the T+1-minus-T slope is 20 per step");
        } else if backend == "wgpu" {
            assert_eq!(
                l_pre, 0,
                "b={b}: preprocess launches nothing on a warmed call"
            );
            assert_eq!(l_enc, 91, "b={b}: encoder launches 91");
            assert_eq!(l_search, 33, "b={b}: search launches 33");
            assert_eq!(l_init, 13, "b={b}: decoder init launches 13");
            assert_eq!(l_finalize, 1, "b={b}: finalize launches 1");
            assert_eq!(l_step, 36, "b={b}: the T+1-minus-T slope is 36 per step");
        } else {
            for (stage, n) in [
                ("preprocess", l_pre),
                ("encoder", l_enc),
                ("search", l_search),
                ("decoder_init", l_init),
                ("step", l_step),
                ("finalize", l_finalize),
            ] {
                println!("STAGE-LAUNCHES {backend} {shape} {stage} {n}");
            }
        }
        let l_call = l_pre + l_enc + l_search + l_init + l_step * decode_steps + l_finalize;
        assert_eq!(
            l_call, total,
            "b={b}: L_call reconciles against the warmed total exactly"
        );
        println!(
            "b={b}: total {total} = pre {l_pre} + enc {l_enc} + search {l_search} + init {l_init} + {decode_steps}*step {l_step} + finalize {l_finalize}"
        );
    }
}

#[test]
fn memory_limit_refuses_only_the_larger_configuration() {
    let _serial = serial();
    // Finding 4 self-check: `--max-device-bytes` reaches every generation
    // configuration the driver builds, and every configuration is preflighted
    // with the same estimate before any allocation. A limit between the
    // estimate of the base configuration and that of a larger one (one extra
    // decode step, as the driver's slope configuration) admits the base and
    // refuses only the larger one. Pure host-side: touches no counter, so it
    // shares this binary with the launch reader above.
    use mamba3::models::ms2::workspace::Ms2MemoryEstimate;
    let comps: Vec<Composition> = vec![[6, 6, 0, 0, 0, 0, 0, 0, 0, 0]];
    let host_table = FormulaTable::from_compositions(comps.clone()).unwrap();
    let base = Ms2MemoryEstimate::generation(
        &ModelConfig::v0(),
        host_table.len() as u64,
        1,
        4,
        64,
        22,
        32,
        4,
    )
    .expect("base estimate builds")
    .total()
    .expect("base total builds");
    let larger = Ms2MemoryEstimate::generation(
        &ModelConfig::v0(),
        host_table.len() as u64,
        1,
        4,
        64,
        23,
        32,
        4,
    )
    .expect("larger estimate builds")
    .total()
    .expect("larger total builds");
    assert!(
        larger > base,
        "one extra decode step estimates larger: {larger} vs {base}"
    );
    let limit = base;
    Ms2MemoryEstimate::generation(
        &ModelConfig::v0(),
        host_table.len() as u64,
        1,
        4,
        64,
        22,
        32,
        4,
    )
    .expect("base estimate builds")
    .check_limit(limit)
    .expect("the base configuration fits a limit equal to its estimate");
    assert!(
        Ms2MemoryEstimate::generation(
            &ModelConfig::v0(),
            host_table.len() as u64,
            1,
            4,
            64,
            23,
            32,
            4
        )
        .expect("larger estimate builds")
        .check_limit(limit)
        .is_err(),
        "the larger configuration is refused by the same limit"
    );
    println!("memory limit {limit}: base {base} fits, larger {larger} refused");
}
