//! I2 integration: allocation, graph identity, ranking/packing and the
//! readout modes in the `Ms2Model` generation path.
//!
//! This is the only test in its binary on purpose. It reads the
//! process-wide launch/read counters, and any test running beside it would
//! add to them. Every test holds the file mutex, so the binary is
//! self-serialised whether cargo runs it threaded or not.

#![cfg(feature = "backend")]

use mamba3::backend::{
    Device, allocation_calls, check_launches, launch_count, reserved_bytes, reset_launch_count,
    reset_read_count, reset_transfer_counters, runtime_read_count,
};
use mamba3::backends::Auto;
use mamba3::models::ms2::chem::{Composition, ELEMENTS, HYDROGEN, composition_mass};
use mamba3::models::ms2::contract::{
    AllocationMode, AssignmentConfig, CandidateBatch, Control, FormulaSource, GenerationConfig,
    GenerationMode, IdentityMode, ModelConfig, NO_FORMULA, SCHEMA_VERSION, SPECTRUM_SCHEMA_VERSION,
    SpectrumBatch, candidate_status,
};
use mamba3::models::ms2::formula::FormulaTable;
use mamba3::models::ms2::formula_head::DeviceFormulaTable;
use mamba3::models::ms2::generate::{GenerationWorkspace, Ms2Model};
use mamba3::models::ms2::identity::{DUPLICATE_GRAPH, IDENTITY_UNRESOLVED, identity_batch};
use mamba3::models::ms2::pack::{ScoreKind, pack};
use mamba3::tensor::ops::index::IdTensor;
use mamba3::tensor::ops::ms2::{self, Ms2Constants};
use mamba3::tensor::ops::random::Rng;

type R = Auto;

fn dev() -> Device<R> {
    Device::<R>::default()
}

/// File-level serialisation for counter-reading tests.
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());
fn serial() -> std::sync::MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
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

fn tiny_generation(
    seed: u64,
    allocation: AllocationMode,
    identity: IdentityMode,
    returned: u32,
) -> GenerationConfig {
    GenerationConfig {
        schema_version: SCHEMA_VERSION,
        trajectories: 4,
        formulas: 2,
        seed,
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
        allocation,
        identity,
        identity_work_max: 4096,
        returned,
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

struct Fixture {
    model: Ms2Model<R, f32>,
    table: DeviceFormulaTable<R, f32>,
    constants: Ms2Constants<R>,
    batch: SpectrumBatch,
}

fn fixture(seed: u64) -> Fixture {
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
    let precursors: Vec<u32> = comps
        .iter()
        .map(|c| composition_mass(c).unwrap() + 1_007_825 - 549)
        .collect();
    let batch = make_spectra(&[801, 802], &precursors, 64, &[10, 12], seed);
    Fixture {
        model,
        table,
        constants,
        batch,
    }
}

/// The identity bits of a read-back batch: status bits 7 and 8, for the host
/// `pack` comparison.
fn identity_bits_of(batch: &CandidateBatch) -> Vec<u32> {
    batch
        .status
        .iter()
        .map(|s| s & (DUPLICATE_GRAPH | IDENTITY_UNRESOLVED))
        .collect()
}

#[test]
fn packed_equals_host_pack_of_generate() {
    let _serial = serial();
    // `generate_packed` equals `pack(&generate(..), identity bits, Raw, R)`
    // for the same request and seed, on several seeds and both identity
    // modes.
    let device = dev();
    for seed in [7u64, 21, 99] {
        for identity in [IdentityMode::TraceOnly, IdentityMode::Graph] {
            let f = fixture(seed);
            let gcfg = tiny_generation(seed, AllocationMode::RoundRobin, identity, 0);
            let r = gcfg.effective_returned() as usize;
            let mut ws = GenerationWorkspace::new();
            for _ in 0..2 {
                f.model
                    .generate(&f.batch, &f.table, &gcfg, &mut ws, &f.constants)
                    .unwrap();
                f.model
                    .generate_packed(&f.batch, &f.table, &gcfg, &mut ws, &f.constants)
                    .unwrap();
            }
            check_launches(&device).unwrap();
            let unpacked = f
                .model
                .generate(&f.batch, &f.table, &gcfg, &mut ws, &f.constants)
                .unwrap();
            unpacked.validate().unwrap();
            let packed = f
                .model
                .generate_packed(&f.batch, &f.table, &gcfg, &mut ws, &f.constants)
                .unwrap();
            packed.validate().unwrap();
            let bits = identity_bits_of(&unpacked);
            let want = pack(
                &unpacked,
                if identity == IdentityMode::Graph {
                    Some(&bits)
                } else {
                    None
                },
                ScoreKind::Raw,
                r,
            )
            .unwrap();
            assert_eq!(
                packed, want,
                "seed {seed} identity {identity:?}: generate_packed differs from host pack"
            );
            println!("seed {seed} identity {identity:?}: packed matches host pack (R={r})");
        }
    }
}

#[test]
fn proportional_generates_valid_deterministic_batches() {
    let _serial = serial();
    // `Proportional` end to end: the batch validates, every formula record
    // carries a rank below its spectrum's `rows_scored` with matching
    // counts, and the same seed reproduces the batch. (The kernel's
    // largest-remainder arithmetic against the twin is pinned by
    // `tests/ms2_allocate.rs`.)
    let device = dev();
    let f = fixture(31);
    let gcfg = tiny_generation(31, AllocationMode::Proportional, IdentityMode::TraceOnly, 0);
    let mut ws = GenerationWorkspace::new();
    for _ in 0..2 {
        f.model
            .generate(&f.batch, &f.table, &gcfg, &mut ws, &f.constants)
            .unwrap();
    }
    check_launches(&device).unwrap();
    let a = f
        .model
        .generate(&f.batch, &f.table, &gcfg, &mut ws, &f.constants)
        .unwrap();
    a.validate().unwrap();
    let b = f
        .model
        .generate(&f.batch, &f.table, &gcfg, &mut ws, &f.constants)
        .unwrap();
    assert_eq!(a, b, "proportional: same seed reproduces the batch");
    for r in 0..a.batch * a.trajectories {
        let failed = a.status[r] & candidate_status::REQUEST_FAILED != 0;
        if failed {
            continue;
        }
        assert!(
            a.formula_rank[r] != NO_FORMULA,
            "proportional: record {r} of a live spectrum has a formula"
        );
        let b = r / a.trajectories;
        assert!(
            a.formula_rank[r] < a.rows_scored[b],
            "proportional: record {r} rank {} below rows_scored {}",
            a.formula_rank[r],
            a.rows_scored[b]
        );
    }
    // The packed path agrees with the host pack under Proportional too.
    let r = gcfg.effective_returned() as usize;
    let packed = f
        .model
        .generate_packed(&f.batch, &f.table, &gcfg, &mut ws, &f.constants)
        .unwrap();
    packed.validate().unwrap();
    let want = pack(&a, None, ScoreKind::Raw, r).unwrap();
    assert_eq!(packed, want, "proportional: packed matches host pack");
}

#[test]
fn identity_flags_same_graph_by_different_traces() {
    let _serial = serial();
    // Two trajectories of one spectrum build propane by different traces
    // (linear vs. branched addition order): the later one is flagged
    // `duplicate_graph` and both records remain. The device bits equal
    // `identity_batch` on the read-back batch.
    let device = dev();
    // Atom types: 3 = C H2 v4, 4 = C H3 v4.
    let steps = 6usize;
    let atoms = 4usize;
    let record = steps * 4 + atoms + 4;
    let state_width = 3 * atoms + 16;
    // Trace A (linear): root CH3, CH2 on 0, CH3 on 1.
    let trace_a: Vec<[u32; 4]> = vec![
        [1, 0, 0, 0],
        [2, 4, 0, 0],
        [2, 3, 1, 0],
        [2, 4, 1, 1],
        [4, 0, 0, 0],
        [0, 0, 0, 0],
    ];
    // Trace B (branched order, same graph): root CH2, CH3 on 0, CH3 on 0.
    let trace_b: Vec<[u32; 4]> = vec![
        [1, 0, 0, 0],
        [2, 3, 0, 0],
        [2, 4, 1, 0],
        [2, 4, 1, 0],
        [4, 0, 0, 0],
        [0, 0, 0, 0],
    ];
    assert_ne!(trace_a, trace_b, "the traces differ");
    let rows = 2usize;
    let mut actions_h = vec![0u32; rows * record];
    for (r, trace) in [trace_a, trace_b].iter().enumerate() {
        let abase = r * record;
        for (s, tok) in trace.iter().enumerate() {
            for c in 0..4 {
                actions_h[abase + s * 4 + c] = tok[c];
            }
        }
        let len_off = abase + steps * 4 + atoms;
        actions_h[len_off] = 5;
        // Finished with a generous formula budget (validation checks the
        // composition against `traj_meta`).
        actions_h[len_off + 1] = candidate_status::FINISHED;
        actions_h[len_off + 3] = 7;
    }
    let mut traj_h = vec![0u32; rows * 14];
    for r in 0..rows {
        traj_h[r * 14 + 3] = 1;
        traj_h[r * 14 + 4] = 3; // C
        traj_h[r * 14 + 5] = 8; // H
    }
    let constants = Ms2Constants::new(&device);
    let mut actions =
        IdTensor::from_slice(&actions_h, vec![rows, record], &device).unwrap();
    let traj_meta = IdTensor::from_slice(&traj_h, vec![rows, 14], &device).unwrap();
    let mut scratch = IdTensor::empty(vec![rows, state_width], &device);
    ms2::validate_trajectories(
        &mut actions,
        &traj_meta,
        &mut scratch,
        &constants.atom_table,
        1,
        rows,
        steps,
        atoms,
        4,
    )
    .unwrap();
    check_launches(&device).unwrap();
    let validated = actions.try_to_vec().unwrap();
    for r in 0..rows {
        let st = validated[r * record + steps * 4 + atoms + 1];
        assert_eq!(
            st & candidate_status::INVALID_FINAL,
            0,
            "record {r} validates (status {st})"
        );
        assert_eq!(
            st & candidate_status::DUPLICATE_TRACE,
            0,
            "record {r} is not a trace duplicate"
        );
    }
    // Graph identity with the production mask.
    let mut graph_hash = IdTensor::empty(vec![rows], &device);
    let g_stride = mamba3::models::ms2::identity::graph_scratch_len(atoms as u32, 4);
    let mut graph_scratch = IdTensor::empty(vec![rows, g_stride], &device);
    let mut identity = IdTensor::empty(vec![rows, 2], &device);
    let s_stride = mamba3::models::ms2::identity::identity_stack_len(atoms as u32);
    let mut stack = IdTensor::empty(vec![rows, s_stride], &device);
    mamba3::tensor::ops::ms2_identity::graph_hash(
        &actions,
        &mut graph_hash,
        &mut graph_scratch,
        steps,
        atoms as u32,
        4,
        u32::MAX,
    )
    .unwrap();
    mamba3::tensor::ops::ms2_identity::graph_identity(
        &actions,
        &graph_hash,
        &graph_scratch,
        &mut identity,
        &mut stack,
        steps,
        atoms as u32,
        (atoms as u32 - 1) + 4,
        rows,
        4096,
    )
    .unwrap();
    check_launches(&device).unwrap();
    let id = identity.try_to_vec().unwrap();
    assert_eq!(id[0], 0, "the first record carries no identity bits");
    assert_eq!(id[1], 1, "the first record resolves exactly");
    assert_eq!(
        id[2], DUPLICATE_GRAPH,
        "the later same-graph record is flagged duplicate_graph"
    );
    assert_eq!(id[3], 1, "the duplicate resolves exactly");
    // The host twin agrees on a read-back batch with the same actions.
    let n = rows;
    let batch = CandidateBatch {
        schema_version: SCHEMA_VERSION,
        batch: 1,
        trajectories: rows,
        max_steps: steps,
        max_atoms: atoms,
        max_ring_closures: 4,
        spectrum_id: vec![4242; n],
        trajectory: vec![0, 1],
        actions: {
            let mut flat = vec![0u32; n * steps * 4];
            for r in 0..n {
                for s in 0..steps {
                    for c in 0..4 {
                        flat[(r * steps + s) * 4 + c] = actions_h[(r * record + s * 4) + c];
                    }
                }
            }
            flat
        },
        length: vec![5; n],
        formula_row: vec![7; n],
        formula_log_prob: vec![-0.5; n],
        trace_log_prob: vec![-1.0; n],
        open_valence: vec![0; n * atoms],
        attachment_partition: vec![0; n],
        status: vec![candidate_status::FINISHED; n],
        evidence_status: vec![0; n],
        evidence_count: vec![0; n],
        evidence_peak_id: vec![0; (n) * 4],
        evidence_hypothesis: vec![0; (n) * 4],
        evidence_shift: vec![0; (n) * 4],
        evidence_residual: vec![0; (n) * 4],
        evidence_log_prob: vec![0.0; (n) * 4],
        identity_resolution: vec![0; n],
        request_status: vec![0],
        rows_visited: vec![4],
        rows_joined: vec![4],
        rows_scored: vec![4],
        formula_support_complete: vec![1],
        formula_mass_retained: vec![0.5],
        peaks_kept: vec![10],
        intensity_retained: vec![0.5],
        formula_counts: vec![3, 8, 0, 0, 0, 0, 0, 0, 0, 0, 3, 8, 0, 0, 0, 0, 0, 0, 0, 0],
        formula_source: vec![0],
        formula_rank: vec![0, 1],
    };
    batch.validate().unwrap();
    let twin = identity_batch(&batch, u32::MAX, 4096);
    assert_eq!(
        twin.status_bits,
        vec![0, DUPLICATE_GRAPH],
        "the twin flags the same later record"
    );
    assert_eq!(
        vec![id[0], id[2]],
        twin.status_bits,
        "device identity equals identity_batch on the read-back batch"
    );
    assert_eq!(twin.resolution, vec![1, 1]);
}

#[test]
fn packed_reinflates_to_valid_candidate_batch() {
    let _serial = serial();
    // `to_candidate_batch` (driver support for precision/coverage at R):
    // the re-inflated batch validates with `trajectories = R`, filled slots
    // round-trip every field, and unfilled slots are empty records.
    let f = fixture(61);
    let gcfg = tiny_generation(61, AllocationMode::RoundRobin, IdentityMode::TraceOnly, 0);
    let r = gcfg.effective_returned() as usize;
    let mut ws = GenerationWorkspace::new();
    for _ in 0..2 {
        f.model
            .generate(&f.batch, &f.table, &gcfg, &mut ws, &f.constants)
            .unwrap();
    }
    let packed = f
        .model
        .generate_packed(&f.batch, &f.table, &gcfg, &mut ws, &f.constants)
        .unwrap();
    packed.validate().unwrap();
    let re = packed.to_candidate_batch().unwrap();
    re.validate().unwrap();
    assert_eq!(re.trajectories, r);
    assert_eq!(re.batch, packed.batch);
    assert_eq!(re.max_steps, packed.max_steps);
    assert_eq!(re.max_atoms, packed.max_atoms);
    let n = packed.batch * r;
    for s in 0..n {
        let filled = packed.trajectory[s] != u32::MAX;
        if filled {
            assert_eq!(re.actions[s * re.max_steps * 4..(s + 1) * re.max_steps * 4], packed.actions[s * packed.max_steps * 4..(s + 1) * packed.max_steps * 4]);
            assert_eq!(re.length[s], packed.length[s]);
            assert_eq!(re.status[s], packed.status[s]);
            assert_eq!(re.formula_row[s], packed.formula_row[s]);
            assert_eq!(re.formula_rank[s], packed.formula_rank[s]);
        } else {
            assert_eq!(re.length[s], 0, "unfilled slot {s} is empty");
            assert_eq!(re.status[s], 0, "unfilled slot {s} carries no bits");
        }
    }
    println!(
        "re-inflated {n} records at R={r} (filled {})",
        packed.trajectory.iter().filter(|t| **t != u32::MAX).count()
    );
}

#[test]
fn read_counts_per_mode() {
    let _serial = serial();
    // Read counts: generate 1, packed 1, resident 0 + deferred 1.
    let device = dev();
    let f = fixture(41);
    let gcfg = tiny_generation(41, AllocationMode::RoundRobin, IdentityMode::TraceOnly, 0);
    let mut ws = GenerationWorkspace::new();
    for _ in 0..2 {
        f.model
            .generate(&f.batch, &f.table, &gcfg, &mut ws, &f.constants)
            .unwrap();
        f.model
            .generate_packed(&f.batch, &f.table, &gcfg, &mut ws, &f.constants)
            .unwrap();
        f.model
            .generate_resident(&f.batch, &f.table, &gcfg, &mut ws, &f.constants)
            .unwrap()
            .release_into(&mut ws);
    }
    device.synchronize();
    reset_launch_count();
    reset_transfer_counters();
    f.model
        .generate(&f.batch, &f.table, &gcfg, &mut ws, &f.constants)
        .unwrap();
    device.synchronize();
    assert_eq!(runtime_read_count(), 1, "generate performs one read");
    reset_transfer_counters();
    f.model
        .generate_packed(&f.batch, &f.table, &gcfg, &mut ws, &f.constants)
        .unwrap();
    device.synchronize();
    assert_eq!(runtime_read_count(), 1, "generate_packed performs one read");
    reset_transfer_counters();
    let resident = f
        .model
        .generate_resident(&f.batch, &f.table, &gcfg, &mut ws, &f.constants)
        .unwrap();
    device.synchronize();
    assert_eq!(runtime_read_count(), 0, "generate_resident performs no read");
    let packed = resident.read(&f.model).unwrap();
    device.synchronize();
    assert_eq!(runtime_read_count(), 1, "the deferred read performs one read");
    packed.validate().unwrap();
    resident.release_into(&mut ws);
}

#[test]
fn resident_survives_second_generate() {
    let _serial = serial();
    // A resident result survives a second `generate` on the same workspace
    // unchanged: the later call allocates another bucket rather than
    // overwriting the leased one.
    let device = dev();
    let f = fixture(43);
    let gcfg = tiny_generation(43, AllocationMode::RoundRobin, IdentityMode::Graph, 0);
    let mut ws = GenerationWorkspace::new();
    for _ in 0..2 {
        f.model
            .generate(&f.batch, &f.table, &gcfg, &mut ws, &f.constants)
            .unwrap();
    }
    let resident = f
        .model
        .generate_resident(&f.batch, &f.table, &gcfg, &mut ws, &f.constants)
        .unwrap();
    let before = resident.read(&f.model).unwrap();
    before.validate().unwrap();
    // A second generate on the same workspace (plus a packed one for good
    // measure) must not disturb the leased buffers.
    f.model
        .generate(&f.batch, &f.table, &gcfg, &mut ws, &f.constants)
        .unwrap();
    f.model
        .generate_packed(&f.batch, &f.table, &gcfg, &mut ws, &f.constants)
        .unwrap();
    check_launches(&device).unwrap();
    let after = resident.read(&f.model).unwrap();
    assert_eq!(before, after, "the resident result is unchanged");
    resident.release_into(&mut ws);
}

#[test]
fn launches_constant_per_mode() {
    let _serial = serial();
    // Launches per call are constant per mode (warmed).
    let device = dev();
    let f = fixture(47);
    let gcfg = tiny_generation(47, AllocationMode::RoundRobin, IdentityMode::TraceOnly, 0);
    let mut ws = GenerationWorkspace::new();
    for _ in 0..2 {
        f.model
            .generate(&f.batch, &f.table, &gcfg, &mut ws, &f.constants)
            .unwrap();
        f.model
            .generate_packed(&f.batch, &f.table, &gcfg, &mut ws, &f.constants)
            .unwrap();
        f.model
            .generate_resident(&f.batch, &f.table, &gcfg, &mut ws, &f.constants)
            .unwrap()
            .release_into(&mut ws);
    }
    device.synchronize();
    reset_launch_count();
    f.model
        .generate(&f.batch, &f.table, &gcfg, &mut ws, &f.constants)
        .unwrap();
    device.synchronize();
    let l_gen = launch_count();
    reset_launch_count();
    f.model
        .generate(&f.batch, &f.table, &gcfg, &mut ws, &f.constants)
        .unwrap();
    device.synchronize();
    assert_eq!(launch_count(), l_gen, "generate launches are constant");
    reset_launch_count();
    f.model
        .generate_packed(&f.batch, &f.table, &gcfg, &mut ws, &f.constants)
        .unwrap();
    device.synchronize();
    let l_packed = launch_count();
    reset_launch_count();
    f.model
        .generate_packed(&f.batch, &f.table, &gcfg, &mut ws, &f.constants)
        .unwrap();
    device.synchronize();
    assert_eq!(launch_count(), l_packed, "packed launches are constant");
    reset_launch_count();
    f.model
        .generate_resident(&f.batch, &f.table, &gcfg, &mut ws, &f.constants)
        .unwrap()
        .release_into(&mut ws);
    device.synchronize();
    let l_resident = launch_count();
    reset_launch_count();
    f.model
        .generate_resident(&f.batch, &f.table, &gcfg, &mut ws, &f.constants)
        .unwrap()
        .release_into(&mut ws);
    device.synchronize();
    assert_eq!(launch_count(), l_resident, "resident launches are constant");
    println!("launches per warmed call: generate {l_gen}, packed {l_packed}, resident {l_resident}");
    // The packed/resident paths add exactly the six pack-stage launches
    // (scores, slot translation, rank, record_pack, record_pack_f, pack).
    assert_eq!(
        l_packed,
        l_gen + 6,
        "packed adds the six pack-stage launches"
    );
    assert_eq!(
        l_resident,
        l_gen + 6,
        "resident adds the six pack-stage launches"
    );
}

#[test]
fn reserved_flat_over_repeated_calls() {
    let _serial = serial();
    // Reserved bytes stay flat over 50 repeated calls of each mode.
    let device = dev();
    let f = fixture(53);
    let gcfg = tiny_generation(53, AllocationMode::RoundRobin, IdentityMode::TraceOnly, 0);
    let mut ws = GenerationWorkspace::new();
    for _ in 0..2 {
        f.model
            .generate(&f.batch, &f.table, &gcfg, &mut ws, &f.constants)
            .unwrap();
    }
    check_launches(&device).unwrap();
    let Some(base) = reserved_bytes(&device) else {
        println!("reserved bytes unavailable here: stability skipped");
        return;
    };
    for _ in 0..50 {
        f.model
            .generate(&f.batch, &f.table, &gcfg, &mut ws, &f.constants)
            .unwrap();
    }
    check_launches(&device).unwrap();
    let after_gen = reserved_bytes(&device).expect("reserved bytes reported");
    assert_eq!(base, after_gen, "reserved bytes flat over 50 generate calls");
    for _ in 0..50 {
        f.model
            .generate_packed(&f.batch, &f.table, &gcfg, &mut ws, &f.constants)
            .unwrap();
    }
    check_launches(&device).unwrap();
    let after_packed = reserved_bytes(&device).expect("reserved bytes reported");
    assert_eq!(
        base, after_packed,
        "reserved bytes flat over 50 packed calls"
    );
    for _ in 0..50 {
        f.model
            .generate_resident(&f.batch, &f.table, &gcfg, &mut ws, &f.constants)
            .unwrap()
            .release_into(&mut ws);
    }
    check_launches(&device).unwrap();
    let after_resident = reserved_bytes(&device).expect("reserved bytes reported");
    assert_eq!(
        base, after_resident,
        "reserved bytes flat over 50 resident calls"
    );
    println!("reserved bytes flat at {base} over 50 calls of each mode");
}

#[test]
fn request_bounds_refused_before_dispatch() {
    let _serial = serial();
    // `returned > K` and an excessive identity request bound are refused
    // before any upload, allocation or launch.
    let device = dev();
    let f = fixture(59);
    let mut ws = GenerationWorkspace::new();
    let mut gcfg = tiny_generation(59, AllocationMode::RoundRobin, IdentityMode::Graph, 0);
    gcfg.returned = 5;
    device.synchronize();
    let l0 = launch_count();
    let a0 = allocation_calls();
    let err = f
        .model
        .generate(&f.batch, &f.table, &gcfg, &mut ws, &f.constants)
        .expect_err("returned > K must be refused");
    assert!(
        err.to_string().contains("returned"),
        "the refusal names returned: {err}"
    );
    device.synchronize();
    assert_eq!(launch_count(), l0, "the refused call launches nothing");
    assert_eq!(allocation_calls(), a0, "the refused call allocates nothing");
    // Identity request bound: B=2, K=64, work_max=u32::MAX overflows the
    // 2^28 request budget.
    let mut gcfg = tiny_generation(59, AllocationMode::RoundRobin, IdentityMode::Graph, 0);
    gcfg.trajectories = 64;
    gcfg.identity_work_max = u32::MAX;
    let l0 = launch_count();
    let a0 = allocation_calls();
    let err = f
        .model
        .generate(&f.batch, &f.table, &gcfg, &mut ws, &f.constants)
        .expect_err("an excessive identity bound must be refused");
    assert!(
        err.to_string().contains("identity_request_work_max"),
        "the refusal names the request bound: {err}"
    );
    device.synchronize();
    assert_eq!(launch_count(), l0, "the refused call launches nothing");
    assert_eq!(allocation_calls(), a0, "the refused call allocates nothing");
}

// ---------------------------------------------------------------------------
// I3b: packed and resident evidence (assignment head, `evidence = true`)
// ---------------------------------------------------------------------------

/// Generation config with fragment-ion evidence on (`J = 4` default head).
fn evidence_generation(seed: u64, returned: u32) -> GenerationConfig {
    let mut g = tiny_generation(seed, AllocationMode::RoundRobin, IdentityMode::TraceOnly, returned);
    g.evidence = true;
    g
}

struct EvidenceFixture {
    model: Ms2Model<R, f32>,
    table: DeviceFormulaTable<R, f32>,
    constants: Ms2Constants<R>,
    batch: SpectrumBatch,
}

/// Single-heavy-atom parents with fragment-mass peaks and non-identity peak
/// ids.
///
/// Parents `[I1]` and `[C1, I1]` (127.9 and 139.9 Da, in range): every
/// finished candidate is a single heavy atom (no second heavy atom fits the
/// budget, so STOP is forced after the root), and its heavy vector always
/// matches a kept hypothesis — evidence flows deterministically on any
/// backend. Peaks sit at exact `(heavy, h)` ion masses
/// (`mz = mass + h * m_H - 549`, residual 0), so each peak keeps one
/// hypothesis. `peak_id = 2 * i` with `raw_peak_count = 20` is strictly
/// increasing but never the identity, so the kept-position → original-id
/// mapping is exercised at readout.
fn evidence_fixture(seed: u64) -> EvidenceFixture {
    let _seed = seed;
    let device = dev();
    // ELEMENTS order: C, H, N, O, F, P, S, Cl, Br, I.
    let comps: Vec<Composition> = vec![
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    ];
    let host_table = FormulaTable::from_compositions(comps.clone()).unwrap();
    let mut cfg = tiny_config();
    cfg.assignment = Some(AssignmentConfig::default());
    let table = DeviceFormulaTable::<R, f32>::upload(&host_table, &device).unwrap();
    cfg.formula_table.rows = table.rows as u32;
    cfg.formula_table.sha256 = table.sha256.clone();
    let mut rng = Rng::seeded(5);
    let model = Ms2Model::<R, f32>::init(&cfg, &device, &mut rng).unwrap();
    let constants = Ms2Constants::new(&device);
    let precursors: Vec<u32> = comps
        .iter()
        .map(|c| composition_mass(c).unwrap() + 1_007_825 - 549)
        .collect();
    let mh = ELEMENTS[HYDROGEN].mass;
    let m_c = ELEMENTS[0].mass;
    let m_i = ELEMENTS[9].mass;
    let mz_i = |h: u32| m_i + h * mh - 549;
    let mz_c = |h: u32| m_c + h * mh - 549;
    // Spectrum 0 (parent I1, hypotheses h = 0..=3), spectrum 1 (parent C1I1:
    // iodine h = 0..=3 plus carbon h = 0..=3).
    let peak_masses = vec![
        (0..4).map(mz_i).collect::<Vec<u32>>(),
        (0..4)
            .map(mz_i)
            .chain((0..4).map(mz_c))
            .collect::<Vec<u32>>(),
    ];
    let b = 2usize;
    let n_raw = 64usize;
    let mut peak_id = vec![u32::MAX; b * n_raw];
    let mut mz = vec![0u32; b * n_raw];
    let mut intensity = vec![0.0f32; b * n_raw];
    let mut peak_count = vec![0u32; b];
    let mut raw_peak_count = vec![20u32; b];
    for (bi, masses) in peak_masses.iter().enumerate() {
        peak_count[bi] = masses.len() as u32;
        raw_peak_count[bi] = 20;
        for (i, &m) in masses.iter().enumerate() {
            peak_id[bi * n_raw + i] = 2 * i as u32;
            mz[bi * n_raw + i] = m;
            intensity[bi * n_raw + i] = 10.0 - i as f32;
        }
    }
    let batch = SpectrumBatch {
        schema_version: SPECTRUM_SCHEMA_VERSION,
        n_raw: n_raw as u32,
        spectrum_id: vec![901, 902],
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
    };
    batch.validate().unwrap();
    EvidenceFixture { model, table, constants, batch }
}

#[test]
fn packed_evidence_equals_host_pack_of_generate() {
    let _serial = serial();
    // `generate_packed == pack(generate)` for EVERY field including the
    // evidence details, with `evidence = true`, on spectra whose `peak_id`
    // is not the identity; the resident read equals the packed read.
    let device = dev();
    for seed in [7u64, 21] {
        let f = evidence_fixture(seed);
        let gcfg = evidence_generation(seed, 2);
        let r = 2usize;
        let mut ws = GenerationWorkspace::new();
        for _ in 0..2 {
            f.model
                .generate(&f.batch, &f.table, &gcfg, &mut ws, &f.constants)
                .unwrap();
            f.model
                .generate_packed(&f.batch, &f.table, &gcfg, &mut ws, &f.constants)
                .unwrap();
        }
        check_launches(&device).unwrap();
        let unpacked = f
            .model
            .generate(&f.batch, &f.table, &gcfg, &mut ws, &f.constants)
            .unwrap();
        unpacked.validate().unwrap();
        let packed = f
            .model
            .generate_packed(&f.batch, &f.table, &gcfg, &mut ws, &f.constants)
            .unwrap();
        packed.validate().unwrap();
        // Evidence really flows: single-carbon candidates always match.
        let ev_total: u32 = unpacked.evidence_count.iter().map(|&c| u32::from(c)).sum();
        assert!(ev_total > 0, "seed {seed}: evidence flows end to end ({ev_total} records)");
        let packed_total: u32 = packed.evidence_count.iter().map(|&c| u32::from(c)).sum();
        assert!(packed_total > 0, "seed {seed}: packed evidence is non-zero");
        // The kept-position → original-id mapping ran: every carried id is
        // even (`peak_id = 2 * i`), which raw positions are not.
        assert!(
            packed.evidence_peak_id.iter().all(|&id| id % 2 == 0),
            "seed {seed}: packed peak ids are original ids, not positions"
        );
        assert!(
            packed.evidence_peak_id.iter().any(|&id| id != 0),
            "seed {seed}: a non-zero original id is carried"
        );
        let want = pack(&unpacked, None, ScoreKind::Raw, r).unwrap();
        assert_eq!(
            packed, want,
            "seed {seed}: generate_packed differs from host pack (every field incl. evidence)"
        );
        println!("seed {seed}: packed matches host pack with {ev_total} evidence records");
        let resident = f
            .model
            .generate_resident(&f.batch, &f.table, &gcfg, &mut ws, &f.constants)
            .unwrap();
        let read = resident.read(&f.model).unwrap();
        read.validate().unwrap();
        assert_eq!(read, packed, "seed {seed}: the resident read equals the packed read");
        resident.release_into(&mut ws);
    }
}

#[test]
fn evidence_read_counts_per_mode() {
    let _serial = serial();
    // Read counts with `evidence = true`: generate 1, packed 1, resident 0
    // + deferred 1 (the packed evidence buffers join the same batched read).
    let device = dev();
    let f = evidence_fixture(33);
    let gcfg = evidence_generation(33, 2);
    let mut ws = GenerationWorkspace::new();
    for _ in 0..2 {
        f.model
            .generate(&f.batch, &f.table, &gcfg, &mut ws, &f.constants)
            .unwrap();
        f.model
            .generate_packed(&f.batch, &f.table, &gcfg, &mut ws, &f.constants)
            .unwrap();
        f.model
            .generate_resident(&f.batch, &f.table, &gcfg, &mut ws, &f.constants)
            .unwrap()
            .release_into(&mut ws);
    }
    device.synchronize();
    reset_transfer_counters();
    f.model
        .generate(&f.batch, &f.table, &gcfg, &mut ws, &f.constants)
        .unwrap();
    device.synchronize();
    assert_eq!(runtime_read_count(), 1, "generate with evidence performs one read");
    reset_transfer_counters();
    f.model
        .generate_packed(&f.batch, &f.table, &gcfg, &mut ws, &f.constants)
        .unwrap();
    device.synchronize();
    assert_eq!(runtime_read_count(), 1, "generate_packed with evidence performs one read");
    reset_transfer_counters();
    let resident = f
        .model
        .generate_resident(&f.batch, &f.table, &gcfg, &mut ws, &f.constants)
        .unwrap();
    device.synchronize();
    assert_eq!(runtime_read_count(), 0, "generate_resident performs no read");
    let packed = resident.read(&f.model).unwrap();
    device.synchronize();
    assert_eq!(runtime_read_count(), 1, "the deferred read performs one read");
    packed.validate().unwrap();
    resident.release_into(&mut ws);
}

#[test]
fn evidence_launch_delta() {
    let _serial = serial();
    // Launch counts with evidence OFF vs ON (warmed): OFF adds nothing past
    // the six pack-stage launches; ON adds the ion, evidence and
    // evidence-pack launches.
    let device = dev();
    let f = evidence_fixture(55);
    let mut ws = GenerationWorkspace::new();
    let off = tiny_generation(55, AllocationMode::RoundRobin, IdentityMode::TraceOnly, 2);
    for _ in 0..2 {
        f.model
            .generate(&f.batch, &f.table, &off, &mut ws, &f.constants)
            .unwrap();
        f.model
            .generate_packed(&f.batch, &f.table, &off, &mut ws, &f.constants)
            .unwrap();
    }
    device.synchronize();
    reset_launch_count();
    f.model
        .generate(&f.batch, &f.table, &off, &mut ws, &f.constants)
        .unwrap();
    device.synchronize();
    let gen_off = launch_count();
    reset_launch_count();
    f.model
        .generate_packed(&f.batch, &f.table, &off, &mut ws, &f.constants)
        .unwrap();
    device.synchronize();
    let packed_off = launch_count();
    assert_eq!(packed_off, gen_off + 6, "OFF: packed adds the six pack launches");
    let on = evidence_generation(55, 2);
    for _ in 0..2 {
        f.model
            .generate(&f.batch, &f.table, &on, &mut ws, &f.constants)
            .unwrap();
        f.model
            .generate_packed(&f.batch, &f.table, &on, &mut ws, &f.constants)
            .unwrap();
        f.model
            .generate_resident(&f.batch, &f.table, &on, &mut ws, &f.constants)
            .unwrap()
            .release_into(&mut ws);
    }
    device.synchronize();
    reset_launch_count();
    f.model
        .generate(&f.batch, &f.table, &on, &mut ws, &f.constants)
        .unwrap();
    device.synchronize();
    let gen_on = launch_count();
    reset_launch_count();
    f.model
        .generate_packed(&f.batch, &f.table, &on, &mut ws, &f.constants)
        .unwrap();
    device.synchronize();
    let packed_on = launch_count();
    reset_launch_count();
    f.model
        .generate_resident(&f.batch, &f.table, &on, &mut ws, &f.constants)
        .unwrap()
        .release_into(&mut ws);
    device.synchronize();
    let resident_on = launch_count();
    println!("OFF launches: generate {gen_off}, packed {packed_off}");
    println!("ON launches: generate {gen_on}, packed {packed_on}, resident {resident_on}");
    assert_eq!(packed_on, resident_on, "ON: packed and resident launch equally");
    assert!(
        gen_on > gen_off,
        "ON: generate launches more ({gen_on} > {gen_off})"
    );
    assert!(
        packed_on > packed_off,
        "ON: packed launches more ({packed_on} > {packed_off})"
    );
}

#[test]
fn evidence_off_after_on_returns_zero_evidence() {
    // Finding R1-C1 (rest): evidence ON then OFF on the same workspace
    // bucket: `generate(evidence = true)` followed by
    // `generate_packed(evidence = false)` and by `generate(evidence =
    // false)` must return the all-zero evidence of the OFF mode (no stale
    // status). Buckets are keyed by the evidence flag and the OFF readout
    // ignores evidence buffers entirely.
    let _serial = serial();
    let device = dev();
    let f = evidence_fixture(7);
    let on = evidence_generation(7, 2);
    let mut ws = GenerationWorkspace::new();
    for _ in 0..2 {
        f.model.generate(&f.batch, &f.table, &on, &mut ws, &f.constants).unwrap();
    }
    check_launches(&device).unwrap();
    let unpacked_on = f.model.generate(&f.batch, &f.table, &on, &mut ws, &f.constants).unwrap();
    unpacked_on.validate().unwrap();
    let ev_total: u32 = unpacked_on.evidence_count.iter().map(|&c| u32::from(c)).sum();
    assert!(ev_total > 0, "the ON call really flows evidence ({ev_total} records)");
    let mut off = on.clone();
    off.evidence = false;
    let packed_off = f
        .model
        .generate_packed(&f.batch, &f.table, &off, &mut ws, &f.constants)
        .unwrap();
    packed_off.validate().unwrap();
    for (name, v) in [
        ("evidence_status", packed_off.evidence_status.iter().map(|&c| u32::from(c)).collect::<Vec<_>>()),
        ("evidence_count", packed_off.evidence_count.iter().map(|&c| u32::from(c)).collect::<Vec<_>>()),
    ] {
        assert!(v.iter().all(|&c| c == 0), "packed OFF {name} is all zero");
    }
    assert!(packed_off.evidence_peak_id.iter().all(|&c| c == 0));
    assert!(packed_off.evidence_hypothesis.iter().all(|&c| c == 0));
    assert!(packed_off.evidence_shift.iter().all(|&c| c == 0));
    assert!(packed_off.evidence_residual.iter().all(|&c| c == 0));
    assert!(packed_off.evidence_log_prob.iter().all(|&c| c == 0.0));
    let unpacked_off = f
        .model
        .generate(&f.batch, &f.table, &off, &mut ws, &f.constants)
        .unwrap();
    unpacked_off.validate().unwrap();
    assert!(unpacked_off.evidence_status.iter().all(|&c| c == 0));
    assert!(unpacked_off.evidence_count.iter().all(|&c| c == 0));
    assert!(unpacked_off.evidence_peak_id.iter().all(|&c| c == 0));
    assert!(unpacked_off.evidence_hypothesis.iter().all(|&c| c == 0));
    assert!(unpacked_off.evidence_shift.iter().all(|&c| c == 0));
    assert!(unpacked_off.evidence_residual.iter().all(|&c| c == 0));
    assert!(unpacked_off.evidence_log_prob.iter().all(|&c| c == 0.0));
}

#[test]
fn packed_validation_accepts_enumeration_counters() {
    // Finding R1-C2: `PackedCandidateBatch::validate` uses the SAME
    // source-specific counter rule as `CandidateBatch::validate` (one shared
    // function): with the enumerating source `joined` may exceed `visited`.
    // The reviewer's case (visited 1, joined 2, scored 2) validates through
    // `generate` and through host `pack` into packed validation. A real
    // table-source batch supplies the finished trajectory; only the
    // per-spectrum enumeration counters are installed.
    let _serial = serial();
    let f = fixture(7);
    let gcfg = tiny_generation(7, AllocationMode::RoundRobin, IdentityMode::TraceOnly, 0);
    let mut ws = GenerationWorkspace::new();
    let out = f.model.generate(&f.batch, &f.table, &gcfg, &mut ws, &f.constants).unwrap();
    out.validate().unwrap();
    assert_eq!(out.formula_source, vec![0, 0], "table source to start");
    let finished_real = out
        .status
        .iter()
        .enumerate()
        .filter(|(r, s)| {
            *s & candidate_status::FINISHED != 0 && out.formula_rank[*r] != NO_FORMULA
        })
        .count();
    assert!(finished_real > 0, "the batch has finished trajectories with formulas");
    let mut mutated = out.clone();
    mutated.formula_source = vec![1, 1];
    mutated.formula_row = vec![NO_FORMULA; mutated.formula_row.len()];
    mutated.rows_visited = vec![1, 1];
    mutated.rows_joined = vec![2, 2];
    mutated.rows_scored = vec![2, 2];
    mutated.formula_support_complete = vec![1, 1];
    // Every real rank is 0 or 1 (F = 2 scored slots), hence below
    // rows_scored = 2; rank-MAX records carry zero counts by construction.
    for r in mutated.formula_rank.iter() {
        assert!(*r == 0 || *r == 1 || *r == NO_FORMULA, "rank {r} fits scored 2");
    }
    mutated.validate().expect("unpacked validation accepts visited 1, joined 2, scored 2");
    let packed = pack(&mutated, None, ScoreKind::Raw, 4).unwrap();
    packed.validate().expect("packed validation accepts visited 1, joined 2, scored 2");
    let filled: u32 = packed.returned_count.iter().sum();
    assert!(filled > 0, "the packed case is non-trivial ({filled} filled slots)");
}

#[test]
fn dtype_gate_rejects_element_type_mismatch_before_any_work() {
    // Finding R1-D1: `Ms2Model::init`, the trainer constructor and the
    // generation/training preflights require `E::DTYPE == config.dtype`
    // (else `Error::Config`, before any allocation, upload or launch) and
    // apply the policy to `E::DTYPE`.
    use half::bf16;
    use mamba3::backend::{DType, upload_bytes};
    use mamba3::error::Error;
    use mamba3::models::ms2::train::{Ms2Trainer, TrainConfig};
    let _serial = serial();
    let device = dev();
    let comps: Vec<Composition> = vec![[6, 6, 0, 0, 0, 0, 0, 0, 0, 0]];
    let host_table = FormulaTable::from_compositions(comps.clone()).unwrap();
    let counters_unchanged = || {
        (
            allocation_calls(),
            upload_bytes(),
            launch_count(),
        )
    };
    // `E = f32` with `config.dtype = BF16`, and `E = bf16` with
    // `config.dtype = F32`: init refuses with `Error::Config`.
    for (make_f32, dtype_name) in [(true, "BF16"), (false, "F32")] {
        let mut cfg = tiny_config();
        cfg.dtype = if make_f32 { DType::BF16 } else { DType::F32 };
        reset_transfer_counters();
        reset_launch_count();
        let base = counters_unchanged();
        let err = if make_f32 {
            let mut rng = Rng::seeded(5);
            match Ms2Model::<R, f32>::init(&cfg, &device, &mut rng) {
                Ok(_) => panic!("init must refuse the E::DTYPE != config.dtype mismatch"),
                Err(err) => err,
            }
        } else {
            let mut rng = Rng::seeded(5);
            match Ms2Model::<R, bf16>::init(&cfg, &device, &mut rng) {
                Ok(_) => panic!("init must refuse the E::DTYPE != config.dtype mismatch"),
                Err(err) => err,
            }
        };
        assert!(
            matches!(err, Error::Config(_)),
            "init mismatch (E vs {dtype_name}) is Error::Config: {err}"
        );
        assert!(
            err.to_string().contains("E::DTYPE"),
            "the mismatch names E::DTYPE == config.dtype: {err}"
        );
        assert_eq!(counters_unchanged(), base, "no allocation, upload or launch");
    }
    // The trainer constructor refuses the same mismatch.
    {
        let mut cfg = tiny_config();
        cfg.dtype = DType::BF16;
        reset_transfer_counters();
        reset_launch_count();
        let base = counters_unchanged();
        let err = match Ms2Trainer::<R, f32>::new(&cfg, &host_table, &TrainConfig::default(), &device) {
            Ok(_) => panic!("trainer construction must refuse the mismatch"),
            Err(err) => err,
        };
        assert!(matches!(err, Error::Config(_)), "trainer mismatch is Error::Config: {err}");
        assert_eq!(counters_unchanged(), base, "no allocation, upload or launch");
    }
    // The generation preflight refuses a model whose config drifted from its
    // element type (the field is public, so drift is reachable).
    {
        let f = fixture(11);
        let gcfg = tiny_generation(11, AllocationMode::RoundRobin, IdentityMode::TraceOnly, 0);
        let mut ws = GenerationWorkspace::new();
        let mut model = f.model;
        model.config.dtype = DType::BF16;
        reset_transfer_counters();
        reset_launch_count();
        let base = counters_unchanged();
        let err = model
            .generate(&f.batch, &f.table, &gcfg, &mut ws, &f.constants)
            .unwrap_err();
        assert!(
            matches!(err, Error::Config(_)),
            "generate preflight mismatch is Error::Config: {err}"
        );
        assert_eq!(counters_unchanged(), base, "no allocation, upload or launch");
    }
    // The training preflight (`step` via the forward prefix) refuses too.
    {
        use mamba3::models::ms2::dataset::ExportSpectrum;
        use mamba3::models::ms2::experiment::{ExperimentSet, ExperimentSpectrum, SpectrumDomain};
        use mamba3::models::ms2::graph::MolGraph;
        let mut cfg = tiny_config();
        let table = DeviceFormulaTable::<R, f32>::upload(&host_table, &device).unwrap();
        cfg.formula_table.rows = table.rows as u32;
        cfg.formula_table.sha256 = table.sha256.clone();
        let mut trainer =
            Ms2Trainer::<R, f32>::new(&cfg, &host_table, &TrainConfig::default(), &device)
                .unwrap();
        let set = ExperimentSet {
            name: "dtype-gate".to_string(),
            source_sha256: "synthetic".to_string(),
            molecules: vec!["mol0".to_string()],
            spectra: vec![ExperimentSpectrum {
                molecule: 0,
                spectrum: ExportSpectrum {
                    row: 0,
                    spectrum_id: 7001,
                    adduct: 1,
                    polarity: 1,
                    precursor_mz_udalton: composition_mass(&comps[0]).unwrap() + 1_007_825 - 549,
                    precursor_uncertainty_udalton: 50,
                    raw_peak_count: 4,
                    peak_id: vec![0, 1, 2, 3],
                    mz_udalton: vec![60_000_000, 70_000_000, 80_000_000, 90_000_000],
                    intensity: vec![1.0, 1.0, 1.0, 1.0],
                    mz_uncertainty_udalton: 50,
                    collision_energy_ev: 30.0,
                    collision_energy_known: 1,
                    energy_count: 1,
                    instrument_class: 0,
                },
                parent: MolGraph::new(Vec::new(), Vec::new()).expect("empty graph builds"),
                parent_composition: comps[0],
                labels: None,
                domain: SpectrumDomain::InDomainUnlabeled,
            }],
        };
        trainer.model.config.dtype = DType::BF16;
        reset_transfer_counters();
        reset_launch_count();
        let base = counters_unchanged();
        let err = trainer.step(&set, &[0]).unwrap_err();
        assert!(
            matches!(err, Error::Config(_)),
            "training preflight mismatch is Error::Config: {err}"
        );
        assert_eq!(counters_unchanged(), base, "no allocation, upload or launch");
    }
}
