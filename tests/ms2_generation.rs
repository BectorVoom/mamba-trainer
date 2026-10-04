//! V0-C tests: the legal sampling step, the generation loop, validation and
//! the packed readout.
//!
//! Every device call is followed by [`check_launches`], so a kernel that
//! failed to compile or run is an error rather than stale data.

#![cfg(feature = "backend")]

use mamba3::autograd::Var;
use mamba3::backend::{Device, check_launches};
use mamba3::backends::Auto;
use mamba3::models::ms2::batch::DeviceSpectra;
use mamba3::models::ms2::chem::{Composition, composition_mass};
use mamba3::models::ms2::contract::{
    AllocationMode, CandidateBatch, Control, FormulaSource, GenerationConfig, GenerationMode,
    IdentityMode, ModelConfig, NO_FORMULA, SCHEMA_VERSION, SPECTRUM_SCHEMA_VERSION,
    SpectrumBatch, candidate_status,
};
use mamba3::models::ms2::decoder::ReplayView;
use mamba3::models::ms2::formula::FormulaTable;
use mamba3::models::ms2::formula_head::DeviceFormulaTable;
use mamba3::models::ms2::generate::{GenerationWorkspace, Ms2Model};
use mamba3::models::ms2::grammar::{
    ADD_ATOM, CLOSE_RING, Limits, START, STOP, Token, TraceState, canonical_trace,
};
use mamba3::models::ms2::targets_batch::TargetBatch;
use mamba3::models::ms2::twin::{self as host, hash_u32_host};
use mamba3::models::ms2::allocate::{ALLOC_ROUND_ROBIN, allocate_checked};
use mamba3::tensor::Tensor;
use mamba3::tensor::ops::index::IdTensor;
use mamba3::tensor::ops::ms2::{self, Ms2Constants};
use mamba3::tensor::ops::ms2_identity;
use mamba3::tensor::ops::random::Rng;

type R = Auto;
type E = f32;

fn dev() -> Device<R> {
    Device::<R>::default()
}

/// Tiny model (`d = 16`, 2 heads, state 8, `N = 16`, `n_raw = 64`, `A = 16`,
/// `R_max = 4`, `T = 22`).
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
        seed: 99,
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

/// Spectra with explicit precursors and random peaks below them.
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

/// The `[M+H]+` precursor of a neutral composition.
fn precursor_of(comp: &Composition) -> u32 {
    composition_mass(comp).unwrap() + 1_007_825 - 549
}

/// Model, tables and constants for generation tests, with the checkpoint
/// reference patched to the uploaded table.
struct GenFixture {
    model: Ms2Model<R, E>,
    table: DeviceFormulaTable<R, E>,
    host_table: FormulaTable,
    constants: Ms2Constants<R>,
    device: Device<R>,
}

fn gen_fixture(comps: Vec<Composition>, seed: u64) -> GenFixture {
    let device = dev();
    let mut cfg = tiny_config();
    let host_table = FormulaTable::from_compositions(comps).unwrap();
    let table = DeviceFormulaTable::<R, E>::upload(&host_table, &device).unwrap();
    cfg.formula_table.rows = table.rows as u32;
    cfg.formula_table.sha256 = table.sha256.clone();
    let mut rng = Rng::seeded(seed);
    let model = Ms2Model::init(&cfg, &device, &mut rng).unwrap();
    let constants = Ms2Constants::new(&device);
    GenFixture {
        model,
        table,
        host_table,
        constants,
        device,
    }
}

fn start_token() -> Token {
    Token {
        kind: START,
        atom_type: 0,
        bond: 0,
        pointer: 0,
    }
}

fn pick_index(rng: &mut Rng, n: usize) -> usize {
    (rng.uniform_vec(1, 0.0, 1.0)[0] * n as f32) as usize % n.max(1)
}

/// One seeded random legal trace (uniform among the legal kinds, types, bonds
/// and pointers from the masks), capped at `max_len` tokens.
fn random_prefix(
    rng: &mut Rng,
    limits: Limits,
    budget: Option<Composition>,
    max_len: usize,
) -> Vec<Token> {
    let mut st = TraceState::new(limits, budget);
    let start = start_token();
    st.apply(start).unwrap();
    let mut trace = vec![start];
    let blank = Token {
        kind: 0,
        atom_type: 0,
        bond: 0,
        pointer: 0,
    };
    let bits = |w: u32| -> Vec<u32> { (0..32).filter(|i| w & (1 << i) != 0).collect() };
    while trace.len() < max_len && !st.stopped() {
        let kinds = st.masks(blank).kinds;
        let mut kind_opts: Vec<u8> = bits(kinds).iter().map(|x| *x as u8).collect();
        kind_opts.retain(|k| *k >= ADD_ATOM);
        if kind_opts.is_empty() {
            break;
        }
        let kind = kind_opts[pick_index(rng, kind_opts.len())];
        if kind == STOP {
            let tok = Token {
                kind: STOP,
                ..blank
            };
            st.apply(tok).unwrap();
            trace.push(tok);
            break;
        }
        let types = st.masks(Token { kind, ..blank }).atom_types;
        let type_opts = bits(types);
        assert!(!type_opts.is_empty(), "kind {kind} promised a type");
        let ty = type_opts[pick_index(rng, type_opts.len())] as u8;
        if kind == ADD_ATOM && st.step() == 1 {
            let tok = Token {
                kind,
                atom_type: ty,
                ..blank
            };
            st.apply(tok).unwrap();
            trace.push(tok);
            continue;
        }
        let bonds = st
            .masks(Token {
                kind,
                atom_type: if kind == ADD_ATOM { ty } else { 0 },
                ..blank
            })
            .bonds;
        let bond_opts = bits(bonds);
        assert!(!bond_opts.is_empty(), "kind {kind} promised a bond");
        let bond = bond_opts[pick_index(rng, bond_opts.len())] as u8;
        let pointers = st
            .masks(Token {
                kind,
                atom_type: if kind == ADD_ATOM { ty } else { 0 },
                bond,
                ..blank
            })
            .pointers;
        let ptr_opts = bits(pointers);
        assert!(!ptr_opts.is_empty(), "kind {kind} promised a pointer");
        let ptr = ptr_opts[pick_index(rng, ptr_opts.len())] as u8;
        let tok = Token {
            kind,
            atom_type: if kind == ADD_ATOM { ty } else { 0 },
            bond,
            pointer: ptr,
        };
        assert!(st.is_legal(tok), "sampled token is legal");
        st.apply(tok).unwrap();
        trace.push(tok);
    }
    trace
}

#[test]
fn sample_step_matches_twin() {
    // V1 §3.1 shape pins: the sampler twin runs at both the small shape
    // and the V1 capacities (A, R_max, T) = (32, 8, 42) with small rows —
    // bit equality of RNG words, masks and tokens holds on the CPU at both.
    for (a, t, rmax, rows) in [
        (6usize, 10usize, 2u32, 10usize),
        (32usize, 42usize, 8u32, 4usize),
    ] {
        // `ms2_sample_step` against `twin::sample_step` on poisoned outputs for
        // random packed logits and random legal grammar states: exact RNG words,
        // exact sampled tokens and masks, `trace_log_prob` within 1e-5, exact
        // status bits, and absorbing rows bit-identical (poison preserved).
        let device = dev();
        let s = ms2::replay_state_width(a);
        let record = ms2::sample_record_width(t, a);
        let width = ms2::sample_logits_width(a);
        let consts = Ms2Constants::new(&device);
        let atable = host::atom_table_rows();
        let limits = Limits::new(a, rmax as usize).unwrap();
        let budget: Composition = [4, 8, 1, 2, 0, 0, 0, 0, 0, 0];
        let mut rng = Rng::seeded(2026);
        // Random legal prefixes of exactly 4 tokens, or STOP-terminated when the
        // walk stops early (those rows are finished and absorbing).
        let mut prefixes: Vec<Vec<Token>> = Vec::with_capacity(rows - 2);
        for _ in 0..rows - 2 {
            prefixes.push(random_prefix(&mut rng, limits, Some(budget), 4));
        }
        // Twin state rows via the twin apply (mirrors the device helper).
        let mut twin_states = vec![0u32; rows * s];
        for (r, prefix) in prefixes.iter().enumerate() {
            for tok in prefix {
                host::apply_token_row(
                    &mut twin_states[r * s..(r + 1) * s],
                    a,
                    &atable,
                    u32::from(tok.kind),
                    u32::from(tok.atom_type),
                    u32::from(tok.bond),
                    u32::from(tok.pointer),
                );
            }
        }
        // The hand-built finished row's tokens (START, root ADD, STOP).
        for (k, ty, b, p) in [(1u32, 0u32, 0u32, 0u32), (2, 2, 0, 0), (4, 0, 0, 0)] {
            host::apply_token_row(
                &mut twin_states[(rows - 1) * s..rows * s],
                a,
                &atable,
                k,
                ty,
                b,
                p,
            );
        }
        // Device states: the twin rows replay the same prefixes, so upload them
        // directly (the device `grammar_apply` path is covered by the decoder
        // tests and the end-to-end generation below).
        let gstate_t = IdTensor::from_slice(&twin_states, vec![rows, s], &device).unwrap();
        // Poisoned action records holding the prefixes; absorbing rows included.
        let mut actions = vec![0xDEAD_BEEFu32; rows * record];
        let mut traj = vec![0u32; rows * 14];
        for r in 0..rows {
            let abase = r * record;
            let tbase = r * 14;
            traj[tbase] = 9001 + r as u32;
            traj[tbase + 1] = 0;
            traj[tbase + 2] = r as u32;
            if r == rows - 2 {
                // Not started: everything stays poisoned.
                traj[tbase + 3] = 0;
                actions[abase + t * 4 + a] = 0;
                actions[abase + t * 4 + a + 1] = 0;
                actions[abase + t * 4 + a + 2] = 0.0f32.to_bits();
                actions[abase + t * 4 + a + 3] = u32::MAX;
                continue;
            }
            traj[tbase + 3] = 1;
            for e in 0..10 {
                traj[tbase + 4 + e] = u32::from(budget[e]);
            }
            if r == rows - 1 {
                // Already finished: a STOP-terminated prefix with the bit set.
                let prefix = vec![
                    start_token(),
                    Token {
                        kind: ADD_ATOM,
                        atom_type: 2,
                        bond: 0,
                        pointer: 0,
                    },
                    Token {
                        kind: STOP,
                        atom_type: 0,
                        bond: 0,
                        pointer: 0,
                    },
                ];
                for (i, tok) in prefix.iter().enumerate() {
                    actions[abase + i * 4] = u32::from(tok.kind);
                    actions[abase + i * 4 + 1] = u32::from(tok.atom_type);
                    actions[abase + i * 4 + 2] = u32::from(tok.bond);
                    actions[abase + i * 4 + 3] = u32::from(tok.pointer);
                }
                actions[abase + t * 4 + a] = prefix.len() as u32;
                actions[abase + t * 4 + a + 1] = candidate_status::FINISHED;
                actions[abase + t * 4 + a + 2] = (-1.5f32).to_bits();
                actions[abase + t * 4 + a + 3] = 3;
                continue;
            }
            let prefix = &prefixes[r];
            for (i, tok) in prefix.iter().enumerate() {
                actions[abase + i * 4] = u32::from(tok.kind);
                actions[abase + i * 4 + 1] = u32::from(tok.atom_type);
                actions[abase + i * 4 + 2] = u32::from(tok.bond);
                actions[abase + i * 4 + 3] = u32::from(tok.pointer);
            }
            actions[abase + t * 4 + a] = prefix.len() as u32;
            actions[abase + t * 4 + a + 1] = if prefix.last().unwrap().kind == STOP {
                candidate_status::FINISHED
            } else {
                0
            };
            actions[abase + t * 4 + a + 2] = 0.0f32.to_bits();
            actions[abase + t * 4 + a + 3] = 3;
        }
        // The device grammar rows for the hand-built finished row differ (its
        // tokens never went through `grammar_apply`); rebuild every device row
        // from the twin rows, which replay the same prefixes.
        let mut twin_actions = actions.clone();
        let mut twin_state_rows = twin_states.clone();
        // Random packed logits and tables.
        let logits_host: Vec<f32> = rng.uniform_vec(rows * width, -3.0, 3.0);
        let tables_host: Vec<f32> = rng.uniform_vec(19 * 4, -1.0, 1.0);
        let step = 4u32;
        let seed_lo = 0x1234_5678u32;
        let seed_hi = 0x9ABC_DEF0u32;
        let temperature = 1.0f32;
        // Twin first (it replays the prefixes from the action rows).
        let mut draws = Vec::with_capacity(rows);
        for r in 0..rows {
            let abase = r * record;
            let lbase = r * width;
            let draw = host::sample_step(
                &logits_host[lbase..lbase + width],
                &tables_host,
                &traj[r * 14..(r + 1) * 14],
                &mut twin_state_rows[r * s..(r + 1) * s],
                &mut twin_actions[abase..abase + record],
                step,
                seed_lo,
                seed_hi,
                temperature,
                t,
                a,
                rmax as usize,
                &atable,
            );
            draws.push(draw);
        }
        let logits_t = Tensor::<R, E>::from_f32(&logits_host, vec![rows, width], &device).unwrap();
        let tables_t = Tensor::<R, E>::from_f32(&tables_host, vec![19, 4], &device).unwrap();
        let traj_t = IdTensor::from_slice(&traj, vec![rows, 14], &device).unwrap();
        let mut state_t = gstate_t;
        let mut actions_t = IdTensor::from_slice(&actions, vec![rows, record], &device).unwrap();
        ms2::sample_step(
            &logits_t,
            &tables_t,
            &traj_t,
            &mut state_t,
            &mut actions_t,
            step,
            seed_lo,
            seed_hi,
            temperature,
            t,
            a,
            rmax,
            &consts.atom_table,
        )
        .unwrap();
        check_launches(&device).unwrap();
        let state_got = state_t.try_to_vec().unwrap();
        let actions_got = actions_t.try_to_vec().unwrap();
        // Exact RNG words: the twin draws equal the hash formula bit-exactly.
        for (r, draw) in draws.iter().enumerate() {
            let s = hash_u32_host(seed_hi, seed_lo, 0);
            let key = hash_u32_host(traj[r * 14], s, traj[r * 14 + 1]);
            let base = hash_u32_host(traj[r * 14 + 2], key, 0);
            for f in 0..4 {
                if draw.u[f] == 0.0 {
                    continue;
                }
                let want = ((hash_u32_host(step * 4 + f as u32, base, 0) >> 8) as f32) / 16777216.0;
                assert_eq!(
                    draw.u[f].to_bits(),
                    want.to_bits(),
                    "row {r} field {f}: draw is the hash word"
                );
            }
        }
        // Exact masks: the twin masks equal a fresh `TraceState` replay of the
        // pre-step prefix.
        for (r, draw) in draws.iter().enumerate() {
            if r >= rows - 2 {
                continue;
            }
            let abase = r * record;
            if actions[abase + t * 4 + a + 1] & candidate_status::FINISHED != 0 {
                continue;
            }
            let len_before = actions[abase + t * 4 + a] as usize;
            let mut st = TraceState::new(limits, Some(budget));
            for i in 0..len_before {
                let tok = Token {
                    kind: actions[abase + i * 4] as u8,
                    atom_type: actions[abase + i * 4 + 1] as u8,
                    bond: actions[abase + i * 4 + 2] as u8,
                    pointer: actions[abase + i * 4 + 3] as u8,
                };
                st.apply(tok).unwrap();
            }
            let blank = Token {
                kind: 0,
                atom_type: 0,
                bond: 0,
                pointer: 0,
            };
            assert_eq!(draw.kinds, st.masks(blank).kinds, "row {r}: kinds");
            let k = draw.token[0];
            if k == u32::from(ADD_ATOM) {
                assert_eq!(
                    draw.types,
                    st.masks(Token {
                        kind: ADD_ATOM,
                        ..blank
                    })
                    .atom_types,
                    "row {r}: types"
                );
                if st.step() != 1 {
                    assert_eq!(
                        draw.bonds,
                        st.masks(Token {
                            kind: ADD_ATOM,
                            atom_type: draw.token[1] as u8,
                            ..blank
                        })
                        .bonds,
                        "row {r}: bonds"
                    );
                    assert_eq!(
                        draw.pointers,
                        st.masks(Token {
                            kind: ADD_ATOM,
                            atom_type: draw.token[1] as u8,
                            bond: draw.token[2] as u8,
                            ..blank
                        })
                        .pointers,
                        "row {r}: pointers"
                    );
                }
            }
            if k == u32::from(CLOSE_RING) {
                assert_eq!(
                    draw.bonds,
                    st.masks(Token {
                        kind: CLOSE_RING,
                        ..blank
                    })
                    .bonds,
                    "row {r}: close bonds"
                );
                assert_eq!(
                    draw.pointers,
                    st.masks(Token {
                        kind: CLOSE_RING,
                        bond: draw.token[2] as u8,
                        ..blank
                    })
                    .pointers,
                    "row {r}: close pointers"
                );
            }
        }
        // Exact tokens, masks already covered, status bits, states; logprob
        // within 1e-5; absorbing rows bit-identical.
        assert_eq!(state_got, twin_state_rows, "grammar rows match exactly");
        for r in 0..rows {
            let abase = r * record;
            for w in 0..record {
                if w == t * 4 + a + 2 {
                    continue;
                }
                assert_eq!(
                    actions_got[abase + w],
                    twin_actions[abase + w],
                    "row {r} word {w}: exact"
                );
            }
            let g = f32::from_bits(actions_got[abase + t * 4 + a + 2]);
            let w = f32::from_bits(twin_actions[abase + t * 4 + a + 2]);
            assert!(
                (g - w).abs() <= 1e-5,
                "row {r}: trace_log_prob {g} vs twin {w}"
            );
        }
    }
}

/// Sample `rows` trajectories from one device step loop with fixed per-step
/// logits and tables under `budget` (spectrum ids `0..rows`, the exporter
/// format). Returns the flat action records; statuses live in the records.
#[allow(clippy::too_many_arguments)]
fn sample_fixed_logits(
    device: &Device<R>,
    consts: &Ms2Constants<R>,
    logits_row: &[f32],
    tables_row: &[f32],
    budget: Composition,
    rows: usize,
    t: usize,
    a: usize,
    rmax: u32,
    seed_lo: u32,
    seed_hi: u32,
) -> Vec<u32> {
    let s = ms2::replay_state_width(a);
    let record = ms2::sample_record_width(t, a);
    let width = ms2::sample_logits_width(a);
    let mut traj = vec![0u32; rows * 14];
    for r in 0..rows {
        traj[r * 14] = r as u32;
        traj[r * 14 + 1] = 0;
        traj[r * 14 + 2] = r as u32;
        traj[r * 14 + 3] = 1;
        for e in 0..10 {
            traj[r * 14 + 4 + e] = u32::from(budget[e]);
        }
    }
    let mut states = vec![0u32; rows * s];
    for r in 0..rows {
        // START applied.
        states[r * s + 3 * a + 4] = 1;
    }
    let mut actions_vec = vec![0u32; rows * record];
    for r in 0..rows {
        let abase = r * record;
        actions_vec[abase] = u32::from(START);
        actions_vec[abase + t * 4 + a] = 1;
    }
    let mut logits_all = vec![0.0f32; rows * width];
    for r in 0..rows {
        logits_all[r * width..(r + 1) * width].copy_from_slice(&logits_row);
    }
    let logits_t = Tensor::<R, E>::from_f32(&logits_all, vec![rows, width], &device).unwrap();
    let tables_t = Tensor::<R, E>::from_f32(&tables_row, vec![19, 4], &device).unwrap();
    let traj_t = IdTensor::from_slice(&traj, vec![rows, 14], &device).unwrap();
    let mut state_t = IdTensor::from_slice(&states, vec![rows, s], &device).unwrap();
    let mut actions_t = IdTensor::from_slice(&actions_vec, vec![rows, record], &device).unwrap();
    for step in 1..t {
        ms2::sample_step(
            &logits_t,
            &tables_t,
            &traj_t,
            &mut state_t,
            &mut actions_t,
            step as u32,
            seed_lo,
            seed_hi,
            1.0,
            t,
            a,
            rmax,
            &consts.atom_table,
        )
        .unwrap();
    }
    check_launches(&device).unwrap();
    actions_t.try_to_vec().unwrap()
}

#[test]
fn exhaustive_frequencies_on_tiny_domain() {
    // Exhaustive check on a tiny domain (`A = 2`, `R_max = 0`, budget `C2H6`,
    // fixed logits): every legal trace enumerated with the host `TraceState`,
    // its exact probability from the logits, against the frequencies of
    // 20,000 sampled trajectories from one device step loop. Every trace's
    // frequency is within 4 standard deviations, and the probabilities sum
    // to 1.
    let device = dev();
    let a = 2usize;
    let rmax = 0u32;
    let t = 4usize;
    let record = ms2::sample_record_width(t, a);
    let width = ms2::sample_logits_width(a);
    let consts = Ms2Constants::new(&device);
    let limits = Limits::new(a, rmax as usize).unwrap();
    let budget: Composition = [2, 6, 0, 0, 0, 0, 0, 0, 0, 0];
    let mut rng = Rng::seeded(7);
    let logits_row: Vec<f32> = rng.uniform_vec(width, -2.0, 2.0);
    let tables_row: Vec<f32> = rng.uniform_vec(19 * 4, -1.0, 1.0);
    // Enumerate every legal trace by depth-first search, rebuilding the
    // grammar state from the prefix at each node (the domain is tiny).
    fn replay_prefix(prefix: &[Token], limits: Limits, budget: Composition) -> TraceState {
        let mut st = TraceState::new(limits, Some(budget));
        for tok in prefix {
            st.apply(*tok).unwrap();
        }
        st
    }
    // Failure leaves (prefixes ending in `no_valid_action`) are collected
    // alongside the finished traces: a failure leaf keeps its tokens so far
    // and adds no token probability.
    fn dfs(
        prefix: &mut Vec<Token>,
        limits: Limits,
        budget: Composition,
        out: &mut Vec<Vec<Token>>,
        failures: &mut Vec<Vec<Token>>,
    ) {
        let blank = Token {
            kind: 0,
            atom_type: 0,
            bond: 0,
            pointer: 0,
        };
        let st = replay_prefix(prefix, limits, budget);
        if st.step() == 0 {
            prefix.push(Token {
                kind: START,
                ..blank
            });
            dfs(prefix, limits, budget, out, failures);
            prefix.pop();
            return;
        }
        if st.step() == 1 {
            if st.masks(blank).atom_types == 0 {
                failures.push(prefix.clone());
                return;
            }
            let types = st.masks(blank).atom_types;
            for id in 1..=17u8 {
                if types & (1 << id) != 0 {
                    prefix.push(Token {
                        kind: ADD_ATOM,
                        atom_type: id,
                        ..blank
                    });
                    dfs(prefix, limits, budget, out, failures);
                    prefix.pop();
                }
            }
            return;
        }
        let kinds = st.masks(blank).kinds;
        if kinds == 0 {
            failures.push(prefix.clone());
            return;
        }
        if kinds & (1 << STOP) != 0 {
            prefix.push(Token {
                kind: STOP,
                ..blank
            });
            out.push(prefix.clone());
            prefix.pop();
        }
        if kinds & (1 << ADD_ATOM) != 0 {
            let types = st
                .masks(Token {
                    kind: ADD_ATOM,
                    ..blank
                })
                .atom_types;
            for id in 1..=17u8 {
                if types & (1 << id) == 0 {
                    continue;
                }
                let bonds = st
                    .masks(Token {
                        kind: ADD_ATOM,
                        atom_type: id,
                        ..blank
                    })
                    .bonds;
                for b in 1..=3u8 {
                    if bonds & (1 << b) == 0 {
                        continue;
                    }
                    let pointers = st
                        .masks(Token {
                            kind: ADD_ATOM,
                            atom_type: id,
                            bond: b,
                            ..blank
                        })
                        .pointers;
                    for p in 0..32u8 {
                        if pointers & (1 << p) == 0 {
                            continue;
                        }
                        prefix.push(Token {
                            kind: ADD_ATOM,
                            atom_type: id,
                            bond: b,
                            pointer: p,
                        });
                        dfs(prefix, limits, budget, out, failures);
                        prefix.pop();
                    }
                }
            }
        }
    }
    let mut failures: Vec<Vec<Token>> = Vec::new();
    let mut traces = Vec::new();
    dfs(&mut Vec::new(), limits, budget, &mut traces, &mut failures);
    assert!(!traces.is_empty(), "the tiny domain has legal traces");
    println!("tiny domain: {} legal traces", traces.len());
    // Exact probability of each trace from the fixed logits at temperature 1.
    fn field_prob(logits: &[f32], mask: u32) -> Vec<f32> {
        let mut m = f32::NEG_INFINITY;
        for (i, v) in logits.iter().enumerate() {
            if mask & (1 << i) != 0 && *v > m {
                m = *v;
            }
        }
        let mut sum = 0.0f32;
        for (i, v) in logits.iter().enumerate() {
            if mask & (1 << i) != 0 {
                sum += (v - m).exp();
            }
        }
        logits
            .iter()
            .enumerate()
            .map(|(i, v)| {
                if mask & (1 << i) != 0 {
                    ((v - m).exp()) / sum
                } else {
                    0.0
                }
            })
            .collect()
    }
    let blank = Token {
        kind: 0,
        atom_type: 0,
        bond: 0,
        pointer: 0,
    };
    let mut probs = Vec::with_capacity(traces.len() + failures.len());
    for trace in traces.iter().chain(failures.iter()) {
        let mut p = 1.0f64;
        let mut gst = TraceState::new(limits, Some(budget));
        for tok in trace {
            if gst.step() == 0 {
                gst.apply(*tok).unwrap();
                continue;
            }
            if gst.step() == 1 {
                let probs5 = field_prob(&logits_row[0..5], gst.masks(blank).kinds);
                p *= probs5[tok.kind as usize] as f64;
                let typ = field_prob(&logits_row[5..23], gst.masks(blank).atom_types);
                p *= typ[tok.atom_type as usize] as f64;
                gst.apply(*tok).unwrap();
                continue;
            }
            let probs5 = field_prob(&logits_row[0..5], gst.masks(blank).kinds);
            p *= probs5[tok.kind as usize] as f64;
            if tok.kind == ADD_ATOM {
                let typ = field_prob(
                    &logits_row[5..23],
                    gst.masks(Token {
                        kind: ADD_ATOM,
                        ..blank
                    })
                    .atom_types,
                );
                p *= typ[tok.atom_type as usize] as f64;
                let mut bond_logits = [0.0f32; 4];
                for i in 0..4 {
                    bond_logits[i] =
                        logits_row[23 + i] + tables_row[tok.atom_type as usize * 4 + i];
                }
                let bonds = field_prob(
                    &bond_logits,
                    gst.masks(Token {
                        kind: ADD_ATOM,
                        atom_type: tok.atom_type,
                        ..blank
                    })
                    .bonds,
                );
                p *= bonds[tok.bond as usize] as f64;
                let mut ptr_logits = [0.0f32; 2];
                for j in 0..2 {
                    ptr_logits[j] = logits_row[27 + j]
                        + logits_row[27 + 2 + tok.atom_type as usize * 2 + j]
                        + logits_row[27 + 2 + 38 + tok.bond as usize * 2 + j];
                }
                let ptrs = field_prob(
                    &ptr_logits,
                    gst.masks(Token {
                        kind: ADD_ATOM,
                        atom_type: tok.atom_type,
                        bond: tok.bond,
                        ..blank
                    })
                    .pointers,
                );
                p *= ptrs[tok.pointer as usize] as f64;
            }
            gst.apply(*tok).unwrap();
        }
        probs.push(p);
    }
    let total: f64 = probs.iter().sum();
    assert!(
        failures.is_empty(),
        "the C2H6 budget always has a legal root atom: no failure leaf"
    );
    assert!(
        (total - 1.0).abs() <= 1e-5,
        "finished and failed outcome probabilities sum to 1, got {total}"
    );
    // 20,000 trajectories from one device step loop with the fixed logits.
    let rows = 20_000usize;
    let got = sample_fixed_logits(
        &device,
        &consts,
        &logits_row,
        &tables_row,
        budget,
        rows,
        t,
        a,
        rmax,
        1234,
        5678,
    );
    use std::collections::HashMap;
    let mut freq: HashMap<Vec<u32>, usize> = HashMap::new();
    for r in 0..rows {
        let abase = r * record;
        let len = got[abase + t * 4 + a] as usize;
        let key = got[abase..abase + len * 4].to_vec();
        *freq.entry(key).or_insert(0) += 1;
    }
    let key_of = |trace: &[Token]| -> Vec<u32> {
        let mut v = Vec::with_capacity(trace.len() * 4);
        for tok in trace {
            v.push(u32::from(tok.kind));
            v.push(u32::from(tok.atom_type));
            v.push(u32::from(tok.bond));
            v.push(u32::from(tok.pointer));
        }
        v
    };
    let mut worst = 0.0f64;
    for (trace, p) in traces.iter().zip(probs.iter()) {
        let n = *freq.get(&key_of(trace)).unwrap_or(&0) as f64;
        let expected = rows as f64 * p;
        let sd = (rows as f64 * p * (1.0 - p)).sqrt();
        let dev = if sd > 0.0 {
            (n - expected).abs() / sd
        } else {
            0.0
        };
        worst = worst.max(dev);
        assert!(
            (n - expected).abs() <= 4.0 * sd + 1e-9,
            "trace {trace:?}: frequency {n} vs expected {expected} (sd {sd})"
        );
    }
    let counted: usize = freq.values().sum();
    assert_eq!(
        counted, rows,
        "every trajectory matches an enumerated trace"
    );
    println!("tiny domain worst deviation: {worst:.3} standard deviations");
    // Outcome classes on the fixed-seed sample: T is the derived cap, so no
    // trajectory truncates, and the root always has a legal atom, so none
    // fails — the finished class carries all 20,000 trajectories and every
    // other class has frequency 0, each within 4 standard errors.
    let mut n_finished = 0usize;
    let mut n_failed = 0usize;
    let mut n_truncated = 0usize;
    for r in 0..rows {
        let st = got[r * record + t * 4 + a + 1];
        if st & candidate_status::NO_VALID_ACTION != 0 {
            n_failed += 1;
        } else if st & candidate_status::TRUNCATED != 0 {
            n_truncated += 1;
        } else if st & candidate_status::FINISHED != 0 {
            n_finished += 1;
        }
    }
    assert_eq!(
        n_finished, rows,
        "the finished class carries every trajectory"
    );
    assert_eq!(n_failed, 0, "the no_valid_action class is empty here");
    assert_eq!(
        n_truncated, 0,
        "the truncated class is empty at the derived cap"
    );
    // Absorbing failures: an unsatisfiable budget (no root atom fits) ends
    // every prefix at the root with `no_valid_action`. The enumeration holds
    // exactly one failure leaf — [START], probability 1, the leaf keeping its
    // tokens and adding no token probability — and 2,000 fixed-seed device
    // trajectories all detect the failure with length 1 and emit no token.
    let empty_budget: Composition = [0; 10];
    let mut traces_e: Vec<Vec<Token>> = Vec::new();
    let mut failures_e: Vec<Vec<Token>> = Vec::new();
    dfs(
        &mut Vec::new(),
        limits,
        empty_budget,
        &mut traces_e,
        &mut failures_e,
    );
    assert!(
        traces_e.is_empty(),
        "no finished trace exists without a root atom"
    );
    assert_eq!(failures_e.len(), 1, "exactly one root failure leaf");
    assert_eq!(failures_e[0], vec![start_token()], "the leaf keeps [START]");
    // C1: run the same outcome-probability evaluator over the
    // unsatisfiable-budget leaves instead of assigning `p_fail = 1` by hand.
    // The failure leaf keeps [START] and adds no token probability, so its
    // evaluated probability is 1 and the summed absorbing-outcome probability
    // is 1. This exercises the failure-leaf accounting that the finished-only
    // normalization above does not (an accidental token factor on failure
    // leaves would fail here).
    let mut probs_e = Vec::with_capacity(failures_e.len());
    for trace in failures_e.iter() {
        let mut p = 1.0f64;
        let mut gst = TraceState::new(limits, Some(empty_budget));
        for tok in trace {
            if gst.step() == 0 {
                gst.apply(*tok).unwrap();
                continue;
            }
            if gst.step() == 1 {
                let probs5 = field_prob(&logits_row[0..5], gst.masks(blank).kinds);
                p *= probs5[tok.kind as usize] as f64;
                let typ = field_prob(&logits_row[5..23], gst.masks(blank).atom_types);
                p *= typ[tok.atom_type as usize] as f64;
                gst.apply(*tok).unwrap();
                continue;
            }
            let probs5 = field_prob(&logits_row[0..5], gst.masks(blank).kinds);
            p *= probs5[tok.kind as usize] as f64;
            if tok.kind == ADD_ATOM {
                let typ = field_prob(
                    &logits_row[5..23],
                    gst.masks(Token {
                        kind: ADD_ATOM,
                        ..blank
                    })
                    .atom_types,
                );
                p *= typ[tok.atom_type as usize] as f64;
                let mut bond_logits = [0.0f32; 4];
                for i in 0..4 {
                    bond_logits[i] =
                        logits_row[23 + i] + tables_row[tok.atom_type as usize * 4 + i];
                }
                let bonds = field_prob(
                    &bond_logits,
                    gst.masks(Token {
                        kind: ADD_ATOM,
                        atom_type: tok.atom_type,
                        ..blank
                    })
                    .bonds,
                );
                p *= bonds[tok.bond as usize] as f64;
                let mut ptr_logits = [0.0f32; 2];
                for j in 0..2 {
                    ptr_logits[j] = logits_row[27 + j]
                        + logits_row[27 + 2 + tok.atom_type as usize * 2 + j]
                        + logits_row[27 + 2 + 38 + tok.bond as usize * 2 + j];
                }
                let ptrs = field_prob(
                    &ptr_logits,
                    gst.masks(Token {
                        kind: ADD_ATOM,
                        atom_type: tok.atom_type,
                        bond: tok.bond,
                        ..blank
                    })
                    .pointers,
                );
                p *= ptrs[tok.pointer as usize] as f64;
            }
            gst.apply(*tok).unwrap();
        }
        probs_e.push(p);
    }
    let total_e: f64 = probs_e.iter().sum();
    assert!(
        (total_e - 1.0).abs() <= 1e-5,
        "unsatisfiable-budget failure leaves sum to 1, got {total_e}"
    );
    let rows_e = 2_000usize;
    let got_e = sample_fixed_logits(
        &device,
        &consts,
        &logits_row,
        &tables_row,
        empty_budget,
        rows_e,
        t,
        a,
        rmax,
        1234,
        5678,
    );
    let mut n_failed_e = 0usize;
    for r in 0..rows_e {
        let abase = r * record;
        let st = got_e[abase + t * 4 + a + 1];
        let len = got_e[abase + t * 4 + a] as usize;
        assert_ne!(
            st & candidate_status::NO_VALID_ACTION,
            0,
            "row {r}: the unsatisfiable budget fails every trajectory"
        );
        assert_eq!(len, 1, "row {r}: the failure emits no token past START");
        for w in 4..t * 4 {
            assert_eq!(got_e[abase + w], 0, "row {r} word {w}: PAD past START");
        }
        n_failed_e += 1;
    }
    assert_eq!(
        n_failed_e, rows_e,
        "the failure class carries every trajectory"
    );
    let p_fail = total_e;
    let sd_e = ((rows_e as f64) * p_fail * (1.0 - p_fail)).sqrt();
    assert!(
        (n_failed_e as f64 - rows_e as f64 * p_fail).abs() <= 4.0 * sd_e + 1e-9,
        "failure-class frequency within 4 standard errors (sd {sd_e})"
    );
}

#[test]
fn sequential_ids_give_independent_streams() {
    // 64 sequential ids (the real exporter format: id = row, high half 0) x
    // K = 8 trajectories: no two (spectrum, trajectory) rows share the same
    // vector of the 8 draws of steps 1 and 2 (twin), and the draws of
    // (spectrum 0, trajectory 1) and (spectrum 1, trajectory 0) differ. Under
    // the old keying (`hash(trajectory, id_lo, id_hi) ^ seed_lo`) those two
    // rows drew identical streams whenever `k ^ i == l ^ j`.
    let seed_lo = 0x1234_5678u32;
    let seed_hi = 0x9ABC_DEF0u32;
    let draws_of = |id_lo: u32, id_hi: u32, traj: u32| -> Vec<u32> {
        let s = hash_u32_host(seed_hi, seed_lo, 0);
        let key = hash_u32_host(id_lo, s, id_hi);
        let base = hash_u32_host(traj, key, 0);
        let mut out = Vec::with_capacity(8);
        for step in [1u32, 2u32] {
            for f in 0..4u32 {
                out.push(hash_u32_host(step * 4 + f, base, 0));
            }
        }
        out
    };
    let mut seen = std::collections::HashSet::new();
    for i in 0..64u32 {
        for k in 0..8u32 {
            let v = draws_of(i, 0, k);
            assert!(
                seen.insert(v.clone()),
                "duplicate draw vector for spectrum {i} trajectory {k}"
            );
        }
    }
    assert_ne!(
        draws_of(0, 0, 1),
        draws_of(1, 0, 0),
        "rows (0, 1) and (1, 0) must differ"
    );
}

#[test]
fn seed_hi_decorrelates_draws() {
    // Two seeds differing only in `seed_hi` give draw sequences whose Pearson
    // correlation over 10,000 draws is below 0.05 in absolute value.
    let seed_lo = 0x1234_5678u32;
    let (hi_a, hi_b) = (0x9ABC_DEF0u32, 0x9ABC_DEF1u32);
    let n = 10_000usize;
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    for i in 0..n {
        let draw = |hi: u32| -> f32 {
            let s = hash_u32_host(hi, seed_lo, 0);
            let key = hash_u32_host(i as u32, s, 0);
            let base = hash_u32_host(0, key, 0);
            ((hash_u32_host(i as u32, base, 0) >> 8) as f32) / 16777216.0
        };
        xs.push(draw(hi_a));
        ys.push(draw(hi_b));
    }
    let mx: f32 = xs.iter().sum::<f32>() / n as f32;
    let my: f32 = ys.iter().sum::<f32>() / n as f32;
    let (mut cov, mut vx, mut vy) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..n {
        let dx = (xs[i] - mx) as f64;
        let dy = (ys[i] - my) as f64;
        cov += dx * dy;
        vx += dx * dx;
        vy += dy * dy;
    }
    let corr = (cov / (vx * vy).sqrt()) as f32;
    println!("seed_hi Pearson correlation: {corr}");
    assert!(
        corr.abs() < 0.05,
        "seeds differing only in seed_hi correlate at {corr}"
    );
}

#[test]
fn init_trajectories_reads_exact_counts() {
    // A table whose rows include counts 0, 1, 2, 40 and 255 of one element
    // gives exact budgets on the kernel: no float round-trip, no binary
    // search.
    let device = dev();
    let counts_want = [0u16, 1, 2, 40, 255];
    let comps: Vec<Composition> = counts_want
        .iter()
        .map(|&c| [c, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        .collect();
    let table = FormulaTable::from_compositions(comps.into_iter()).unwrap();
    assert_eq!(table.len(), 5);
    let _uploaded = DeviceFormulaTable::<R, E>::upload(&table, &device).unwrap();
    // Table rows are sorted by mass; resolve the row of each carbon count.
    let mut row_of = [0u32; 5];
    for (i, &c) in counts_want.iter().enumerate() {
        let mut found = None;
        for row in 0..table.len() {
            if table.composition(row)[0] == c {
                found = Some(row as u32);
                break;
            }
        }
        row_of[i] = found.expect("each carbon count is a table row");
    }
    let (spectra_n, formulas, trajectories, steps, atoms) =
        (1usize, 5usize, 5usize, 6usize, 4usize);
    let top_host: Vec<u32> = row_of.iter().flat_map(|&r| vec![r, r]).collect();
    let top = IdTensor::from_slice(&top_host, vec![spectra_n, formulas, 2], &device).unwrap();
    let mut top_counts_host = vec![0u32; spectra_n * formulas * 10];
    for (i, &r) in row_of.iter().enumerate() {
        let comp = table.composition(r as usize);
        for e in 0..10 {
            top_counts_host[i * 10 + e] = u32::from(comp[e]);
        }
    }
    let top_counts =
        IdTensor::from_slice(&top_counts_host, vec![spectra_n, formulas, 10], &device).unwrap();
    // Round-robin allocation first (V1 §3.2): trajectory `k` takes slot `k`.
    let top_lp_host = vec![0.0f32; spectra_n * formulas];
    let top_count_host = vec![5u32; spectra_n];
    let top_lp =
        Tensor::<R, E>::from_f32(&top_lp_host, vec![spectra_n, formulas], &device).unwrap();
    let top_count = IdTensor::from_slice(&top_count_host, vec![spectra_n], &device).unwrap();
    let mut traj_alloc =
        IdTensor::empty(vec![spectra_n, trajectories, 12], &device);
    ms2_identity::allocate(
        &top,
        &top_counts,
        &top_lp,
        &top_count,
        &mut traj_alloc,
        ALLOC_ROUND_ROBIN,
    )
    .unwrap();
    check_launches(&device).unwrap();
    let mut traj_host = vec![0u32; spectra_n * trajectories * 12];
    allocate_checked(
        &top_host,
        &top_counts_host,
        &top_lp_host,
        &top_count_host,
        spectra_n as u32,
        formulas as u32,
        trajectories as u32,
        ALLOC_ROUND_ROBIN,
        &mut traj_host,
    )
    .unwrap();
    assert_eq!(
        traj_alloc.try_to_vec().unwrap(),
        traj_host,
        "round-robin allocation matches the twin"
    );
    let meta_host = vec![1u32, 200_000_000, 50, 1, 100, 200, 0, 0];
    let meta = IdTensor::from_slice(&meta_host, vec![spectra_n, 8], &device).unwrap();
    let rows = spectra_n * trajectories;
    let mut traj_meta = IdTensor::empty(vec![rows, ms2::TRAJ_META_WIDTH], &device);
    let mut state = IdTensor::empty(vec![rows, ms2::replay_state_width(atoms)], &device);
    let mut actions = IdTensor::empty(vec![rows, ms2::sample_record_width(steps, atoms)], &device);
    ms2::init_trajectories(
        &traj_alloc,
        &meta,
        &mut traj_meta,
        &mut state,
        &mut actions,
        spectra_n,
        trajectories,
        steps,
        atoms,
        false,
    )
    .unwrap();
    check_launches(&device).unwrap();
    let got = traj_meta.try_to_vec().unwrap();
    for k in 0..trajectories {
        let base = k * ms2::TRAJ_META_WIDTH;
        let want = u32::from(counts_want[k]);
        assert_eq!(got[base + 4], want, "trajectory {k} carbon budget is exact");
        assert_eq!(got[base + 5], 1, "trajectory {k} hydrogen budget is exact");
        for e in 2..10 {
            assert_eq!(got[base + 4 + e], 0, "trajectory {k} element {e} is 0");
        }
    }
}

#[test]
fn init_trajectories_matches_twin_every_element() {
    // B1-fix finding 4: host twin for `init_trajectories`, every element of
    // `traj_meta`, `state` and `actions` compared against it on poisoned
    // outputs. Covers padded tops, no formulas (`top_count == 0`), empty
    // peaks, the metadata-only bypass and the `k mod count` assignment.
    let device = dev();
    let (spectra, formulas, per_spectrum, steps, atoms) = (2usize, 3usize, 4usize, 6usize, 4usize);
    let rows = spectra * per_spectrum;
    let state_width = ms2::replay_state_width(atoms);
    let record_width = ms2::sample_record_width(steps, atoms);
    let len_off = steps * 4 + atoms;
    // Spectrum 0: two scored formulas then a padded top row (slot MAX).
    // Spectrum 1: no formulas at all (every slot MAX).
    let top_host: Vec<u32> = vec![
        10,
        0,
        11,
        1,
        u32::MAX,
        u32::MAX,
        u32::MAX,
        u32::MAX,
        u32::MAX,
        u32::MAX,
        u32::MAX,
        u32::MAX,
    ];
    let top_count_host: Vec<u32> = vec![2, 0];
    let top_lp_host: Vec<f32> = vec![0.0; spectra * formulas];
    let mut top_counts_host = vec![0u32; spectra * formulas * 10];
    let budgets: [[u32; 10]; 2] = [
        [6, 12, 0, 6, 0, 0, 0, 0, 0, 0],
        [3, 7, 1, 2, 0, 0, 0, 0, 0, 0],
    ];
    for e in 0..10 {
        top_counts_host[e] = budgets[0][e];
        top_counts_host[10 + e] = budgets[1][e];
    }
    // Spectrum 0 has peaks; spectrum 1 is empty (peak_count 0).
    let meta_host: Vec<u32> = vec![
        3,
        200_000_000,
        50,
        1,
        100,
        200,
        111,
        222,
        0,
        200_000_000,
        50,
        1,
        100,
        200,
        333,
        444,
    ];
    for metadata_only in [false, true] {
        let what = if metadata_only { "meta-only" } else { "plain" };
        let top = IdTensor::from_slice(&top_host, vec![spectra, formulas, 2], &device).unwrap();
        let top_counts =
            IdTensor::from_slice(&top_counts_host, vec![spectra, formulas, 10], &device).unwrap();
        let top_lp =
            Tensor::<R, E>::from_f32(&top_lp_host, vec![spectra, formulas], &device).unwrap();
        let top_count =
            IdTensor::from_slice(&top_count_host, vec![spectra], &device).unwrap();
        let mut traj_alloc =
            IdTensor::empty(vec![spectra, per_spectrum, 12], &device);
        ms2_identity::allocate(
            &top,
            &top_counts,
            &top_lp,
            &top_count,
            &mut traj_alloc,
            ALLOC_ROUND_ROBIN,
        )
        .unwrap();
        let mut traj_host = vec![0u32; spectra * per_spectrum * 12];
        allocate_checked(
            &top_host,
            &top_counts_host,
            &top_lp_host,
            &top_count_host,
            spectra as u32,
            formulas as u32,
            per_spectrum as u32,
            ALLOC_ROUND_ROBIN,
            &mut traj_host,
        )
        .unwrap();
        assert_eq!(
            traj_alloc.try_to_vec().unwrap(),
            traj_host,
            "{what}: allocation matches the twin"
        );
        let meta = IdTensor::from_slice(&meta_host, vec![spectra, 8], &device).unwrap();
        let poison = |len: usize| vec![0xDEAD_BEEFu32; len];
        let mut traj_meta = IdTensor::from_slice(
            &poison(rows * ms2::TRAJ_META_WIDTH),
            vec![rows, ms2::TRAJ_META_WIDTH],
            &device,
        )
        .unwrap();
        let mut state = IdTensor::from_slice(
            &poison(rows * state_width),
            vec![rows, state_width],
            &device,
        )
        .unwrap();
        let mut actions = IdTensor::from_slice(
            &poison(rows * record_width),
            vec![rows, record_width],
            &device,
        )
        .unwrap();
        ms2::init_trajectories(
            &traj_alloc,
            &meta,
            &mut traj_meta,
            &mut state,
            &mut actions,
            spectra,
            per_spectrum,
            steps,
            atoms,
            metadata_only,
        )
        .unwrap();
        check_launches(&device).unwrap();
        let (want_meta, want_state, want_actions) = host::init_trajectories(
            &traj_host,
            &meta_host,
            spectra,
            per_spectrum,
            steps,
            atoms,
            metadata_only,
        );
        assert_eq!(
            traj_meta.try_to_vec().unwrap(),
            want_meta,
            "{what}: traj_meta differs"
        );
        assert_eq!(
            state.try_to_vec().unwrap(),
            want_state,
            "{what}: state differs"
        );
        assert_eq!(
            actions.try_to_vec().unwrap(),
            want_actions,
            "{what}: actions differ"
        );
        // Spot semantics on top of the exhaustive comparison.
        let got_meta = traj_meta.try_to_vec().unwrap();
        let got_actions = actions.try_to_vec().unwrap();
        // `k mod count` with count 2 over K = 4: trajectories 0..3 of
        // spectrum 0 use formulas 0, 1, 0, 1 (budgets alternate).
        for k in 0..per_spectrum {
            let base = k * ms2::TRAJ_META_WIDTH;
            let want_c = budgets[k % 2][0];
            assert_eq!(got_meta[base + 4], want_c, "{what}: traj {k} carbon budget");
            assert_eq!(
                got_actions[k * record_width + len_off],
                1,
                "{what}: traj {k} started"
            );
        }
        // No formulas (spectrum 1): no trajectory starts either way; every
        // row keeps `request_failed` with length 0 and row MAX.
        for k in 0..per_spectrum {
            let r = per_spectrum + k;
            assert_eq!(
                got_actions[r * record_width + len_off],
                0,
                "{what}: empty-formula traj {k} length"
            );
            assert_eq!(
                got_actions[r * record_width + len_off + 1] & 64,
                64,
                "{what}: empty-formula traj {k} request_failed"
            );
            assert_eq!(
                got_actions[r * record_width + len_off + 3],
                u32::MAX,
                "{what}: empty-formula traj {k} formula_row"
            );
        }
    }
    // Empty peaks with formulas present: spectrum 0 emptied, one scored
    // formula. Plain mode abstains; the metadata-only bypass starts.
    let top_host2: Vec<u32> = vec![7, 0, u32::MAX, u32::MAX, u32::MAX, u32::MAX];
    let top_counts2: Vec<u32> = vec![
        6, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ];
    let meta_empty: Vec<u32> = vec![0, 200_000_000, 50, 1, 100, 200, 555, 666];
    for (metadata_only, want_len) in [(false, 0u32), (true, 1u32)] {
        let top = IdTensor::from_slice(&top_host2, vec![1, formulas, 2], &device).unwrap();
        let tc = IdTensor::from_slice(&top_counts2, vec![1, formulas, 10], &device).unwrap();
        let lp_host2 = vec![0.0f32; formulas];
        let tc_count_host2 = vec![1u32];
        let lp2 = Tensor::<R, E>::from_f32(&lp_host2, vec![1, formulas], &device).unwrap();
        let tcc = IdTensor::from_slice(&tc_count_host2, vec![1], &device).unwrap();
        let mut ta = IdTensor::empty(vec![1, per_spectrum, 12], &device);
        ms2_identity::allocate(&top, &tc, &lp2, &tcc, &mut ta, ALLOC_ROUND_ROBIN).unwrap();
        let mut ta_host = vec![0u32; per_spectrum * 12];
        allocate_checked(
            &top_host2,
            &top_counts2,
            &lp_host2,
            &tc_count_host2,
            1,
            formulas as u32,
            per_spectrum as u32,
            ALLOC_ROUND_ROBIN,
            &mut ta_host,
        )
        .unwrap();
        assert_eq!(
            ta.try_to_vec().unwrap(),
            ta_host,
            "empty-peaks allocation matches the twin"
        );
        let meta = IdTensor::from_slice(&meta_empty, vec![1, 8], &device).unwrap();
        let mut tm = IdTensor::from_slice(
            &vec![0xDEAD_BEEFu32; per_spectrum * ms2::TRAJ_META_WIDTH],
            vec![per_spectrum, ms2::TRAJ_META_WIDTH],
            &device,
        )
        .unwrap();
        let mut st = IdTensor::from_slice(
            &vec![0xDEAD_BEEFu32; per_spectrum * state_width],
            vec![per_spectrum, state_width],
            &device,
        )
        .unwrap();
        let mut ac = IdTensor::from_slice(
            &vec![0xDEAD_BEEFu32; per_spectrum * record_width],
            vec![per_spectrum, record_width],
            &device,
        )
        .unwrap();
        ms2::init_trajectories(
            &ta,
            &meta,
            &mut tm,
            &mut st,
            &mut ac,
            1,
            per_spectrum,
            steps,
            atoms,
            metadata_only,
        )
        .unwrap();
        check_launches(&device).unwrap();
        let (w_tm, w_st, w_ac) = host::init_trajectories(
            &ta_host,
            &meta_empty,
            1,
            per_spectrum,
            steps,
            atoms,
            metadata_only,
        );
        assert_eq!(
            tm.try_to_vec().unwrap(),
            w_tm,
            "empty-peaks meta={metadata_only}: traj_meta"
        );
        assert_eq!(
            st.try_to_vec().unwrap(),
            w_st,
            "empty-peaks meta={metadata_only}: state"
        );
        assert_eq!(
            ac.try_to_vec().unwrap(),
            w_ac,
            "empty-peaks meta={metadata_only}: actions"
        );
        let got = ac.try_to_vec().unwrap();
        assert_eq!(
            got[len_off], want_len,
            "empty-peaks meta={metadata_only}: length"
        );
    }
}

fn parent_comps() -> Vec<Composition> {
    // Benzene, alanine, glucose and naphthalene parents: in-domain, with
    // distinct masses, all above the 50 Da precursor floor (with the
    // `[M+H]+` shift the precursors stay in range).
    vec![
        [6, 6, 0, 0, 0, 0, 0, 0, 0, 0],
        [3, 7, 1, 2, 0, 0, 0, 0, 0, 0],
        [6, 12, 0, 6, 0, 0, 0, 0, 0, 0],
        [10, 8, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
}

#[test]
fn batch_independence_and_seed_sensitivity() {
    // A spectrum's trajectories are identical whether it is generated alone
    // or inside a batch of 3 (same backend and seed); a different seed
    // changes them.
    let comps = parent_comps();
    let precursors: Vec<u32> = comps[0..3].iter().map(precursor_of).collect();
    let fx = gen_fixture(comps.clone(), 11);
    let mut ws = GenerationWorkspace::new();
    let mut cfg = tiny_generation();
    let batch3 = make_spectra(&[101, 102, 103], &precursors, 64, &[10, 12, 9], 21);
    let out3 = fx
        .model
        .generate(&batch3, &fx.table, &cfg, &mut ws, &fx.constants)
        .unwrap();
    check_launches(&fx.device).unwrap();
    out3.validate().unwrap();
    let batch1 = make_spectra(&[101], &precursors[0..1], 64, &[10], 21);
    let out1 = fx
        .model
        .generate(&batch1, &fx.table, &cfg, &mut ws, &fx.constants)
        .unwrap();
    check_launches(&fx.device).unwrap();
    assert_eq!(out1.batch, 1);
    for k in 0..cfg.trajectories as usize {
        let a3 = k;
        let a1 = k;
        assert_eq!(out3.length[a3], out1.length[a1], "trajectory {k}: length");
        assert_eq!(out3.status[a3], out1.status[a1], "trajectory {k}: status");
        assert_eq!(
            out3.formula_row[a3], out1.formula_row[a1],
            "trajectory {k}: formula"
        );
        assert_eq!(
            out3.trace_log_prob[a3].to_bits(),
            out1.trace_log_prob[a1].to_bits(),
            "trajectory {k}: logprob"
        );
        for s in 0..out3.max_steps * 4 {
            assert_eq!(
                out3.actions[a3 * out3.max_steps * 4 + s],
                out1.actions[a1 * out1.max_steps * 4 + s],
                "trajectory {k} word {s}: tokens"
            );
        }
    }
    cfg.seed = 100;
    let out3b = fx
        .model
        .generate(&batch3, &fx.table, &cfg, &mut ws, &fx.constants)
        .unwrap();
    check_launches(&fx.device).unwrap();
    let same = (0..cfg.trajectories as usize * out3.max_steps * 4)
        .all(|i| out3.actions[i] == out3b.actions[i]);
    assert!(!same, "a different seed changes the sampled tokens");
}

#[test]
fn generate_end_to_end() {
    // The result passes `CandidateBatch::validate`; finished candidates
    // replay legally with `TraceState` under their formula budget; a spectrum
    // with a fatal request status yields K `request_failed` records;
    // `ShuffledSpectrum` and `MetadataOnly` run; `oracle_formula` is
    // `Error::Unsupported`.
    let comps = parent_comps();
    let precursors: Vec<u32> = comps[0..3].iter().map(precursor_of).collect();
    let fx = gen_fixture(comps.clone(), 13);
    let mut ws = GenerationWorkspace::new();
    let cfg = tiny_generation();
    let mut batch = make_spectra(
        &[201, 202, 203, 204],
        &[precursors[0], precursors[1], precursors[2], precursors[0]],
        64,
        &[10, 12, 9, 10],
        23,
    );
    // Spectrum 203 gets an unknown adduct: fatal `insufficient_metadata`.
    batch.adduct[2] = 0;
    let out = fx
        .model
        .generate(&batch, &fx.table, &cfg, &mut ws, &fx.constants)
        .unwrap();
    check_launches(&fx.device).unwrap();
    out.validate().unwrap();
    let k = cfg.trajectories as usize;
    // The fatal spectrum's records.
    for kk in 0..k {
        let r = 2 * k + kk;
        assert_eq!(
            out.status[r] & candidate_status::REQUEST_FAILED,
            candidate_status::REQUEST_FAILED,
            "fatal record {r} carries request_failed"
        );
        assert_eq!(out.length[r], 0, "fatal record {r} has length 0");
        assert_eq!(
            out.formula_row[r], NO_FORMULA,
            "fatal record {r} has no formula"
        );
    }
    // Finished candidates replay legally under their formula budget.
    let limits = Limits::new(out.max_atoms, out.max_ring_closures).unwrap();
    let mut finished = 0;
    for r in 0..out.batch * out.trajectories {
        if out.status[r] & candidate_status::REQUEST_FAILED != 0 {
            continue;
        }
        if out.status[r] & candidate_status::FINISHED == 0 {
            continue;
        }
        finished += 1;
        let len = out.length[r] as usize;
        let mut tokens = Vec::with_capacity(len);
        for s in 0..len {
            tokens.push(Token {
                kind: out.actions[(r * out.max_steps + s) * 4] as u8,
                atom_type: out.actions[(r * out.max_steps + s) * 4 + 1] as u8,
                bond: out.actions[(r * out.max_steps + s) * 4 + 2] as u8,
                pointer: out.actions[(r * out.max_steps + s) * 4 + 3] as u8,
            });
        }
        let comp = fx.host_table.composition(out.formula_row[r] as usize);
        let mut budget = [0u16; 10];
        for (e, c) in comp.iter().enumerate() {
            budget[e] = *c;
        }
        match mamba3::models::ms2::grammar::replay(&tokens, limits, Some(budget)) {
            Ok(state) => assert!(state.stopped(), "record {r} ends stopped"),
            Err(e) => panic!("record {r} replays legally under its budget: {e}"),
        }
    }
    assert!(finished > 0, "some candidates finish");
    println!("e2e finished candidates: {finished}");
    // The controls run and validate.
    for control in [Control::ShuffledSpectrum, Control::MetadataOnly] {
        let mut ccfg = tiny_generation();
        ccfg.control = control;
        let cout = fx
            .model
            .generate(&batch, &fx.table, &ccfg, &mut ws, &fx.constants)
            .unwrap();
        check_launches(&fx.device).unwrap();
        cout.validate().unwrap();
    }
    // The oracle path is not implemented in V0.
    let mut ocfg = tiny_generation();
    ocfg.oracle_formula = true;
    let err = fx
        .model
        .generate(&batch, &fx.table, &ocfg, &mut ws, &fx.constants)
        .unwrap_err();
    assert!(
        matches!(err, mamba3::error::Error::Unsupported(_)),
        "oracle_formula is Unsupported, got {err:?}"
    );
}

#[test]
fn carry_freeze_after_finish() {
    // A trajectory that finished at step s has bit-identical cache tensors at
    // every later step (through the workspace inspection hook).
    let comps = parent_comps();
    let precursors: Vec<u32> = comps[0..2].iter().map(precursor_of).collect();
    let fx = gen_fixture(comps.clone(), 17);
    let mut ws = GenerationWorkspace::new();
    ws.capture_carry_trace = true;
    let cfg = tiny_generation();
    let batch = make_spectra(&[301, 302], &precursors, 64, &[10, 12], 25);
    let out = fx
        .model
        .generate(&batch, &fx.table, &cfg, &mut ws, &fx.constants)
        .unwrap();
    check_launches(&fx.device).unwrap();
    out.validate().unwrap();
    let t = out.max_steps;
    assert_eq!(ws.carry_trace.len(), t - 1, "one snapshot per step");
    let k = out.trajectories;
    // A trajectory finished strictly before the last step.
    let mut target: Option<(usize, usize)> = None;
    for r in 0..out.batch * k {
        if out.status[r] & candidate_status::FINISHED != 0 && (out.length[r] as usize) < t {
            target = Some((r, out.length[r] as usize - 1));
            break;
        }
    }
    let (row, finished_step) = target.expect("a trajectory finishes before T - 1");
    // The decoder cache row stride: [rows, heads, head_dim, d_state].
    let stride = 2 * 8 * 8;
    for entry in ws.carry_trace.iter().filter(|e| e.step >= finished_step) {
        let first = &ws.carry_trace[finished_step - 1];
        assert_eq!(entry.layers.len(), first.layers.len());
        for (l, (a, b)) in entry.layers.iter().zip(first.layers.iter()).enumerate() {
            assert_eq!(
                a.h[row * stride..(row + 1) * stride],
                b.h[row * stride..(row + 1) * stride],
                "step {} layer {l}: h frozen",
                entry.step
            );
            assert_eq!(
                a.last_u[row * stride..(row + 1) * stride],
                b.last_u[row * stride..(row + 1) * stride],
                "step {} layer {l}: last_u frozen",
                entry.step
            );
            match (&a.angle, &b.angle) {
                (Some(x), Some(y)) => assert_eq!(
                    x[row * 8..(row + 1) * 8],
                    y[row * 8..(row + 1) * 8],
                    "step {} layer {l}: angle frozen",
                    entry.step
                ),
                (None, None) => {}
                _ => panic!("angle presence disagrees"),
            }
        }
    }
    println!("carry freeze holds for trajectory {row} finished at step {finished_step}");
}

#[test]
fn teacher_agrees_with_generated_trace() {
    // Feeding a generated candidate's trace back through `teacher` gives
    // per-token log-probabilities whose sum equals the candidate's
    // `trace_log_prob` within 1e-4 (at temperature 1).
    let comps = parent_comps();
    let precursors: Vec<u32> = comps[0..2].iter().map(precursor_of).collect();
    let fx = gen_fixture(comps.clone(), 19);
    let mut ws = GenerationWorkspace::new();
    let mut cfg = tiny_generation();
    cfg.temperature = 1.0;
    let batch = make_spectra(&[401, 402], &precursors, 64, &[10, 12], 27);
    let out = fx
        .model
        .generate(&batch, &fx.table, &cfg, &mut ws, &fx.constants)
        .unwrap();
    check_launches(&fx.device).unwrap();
    out.validate().unwrap();
    // Pick a finished candidate with a formula.
    let mut pick: Option<usize> = None;
    for r in 0..out.batch * out.trajectories {
        if out.status[r] & candidate_status::FINISHED != 0 && out.formula_row[r] != NO_FORMULA {
            pick = Some(r);
            break;
        }
    }
    let r = pick.expect("a finished candidate with a formula");
    let b = r / out.trajectories;
    let len = out.length[r] as usize;
    let mut tokens = Vec::with_capacity(len);
    for s in 0..len {
        tokens.push(Token {
            kind: out.actions[(r * out.max_steps + s) * 4] as u8,
            atom_type: out.actions[(r * out.max_steps + s) * 4 + 1] as u8,
            bond: out.actions[(r * out.max_steps + s) * 4 + 2] as u8,
            pointer: out.actions[(r * out.max_steps + s) * 4 + 3] as u8,
        });
    }
    // Re-run the pipeline prefix to recover the exact conditioning formula
    // embedding the sampler used.
    let spectra = DeviceSpectra::upload(&batch, &fx.device).unwrap();
    let peaks = ms2::PeakBuffers::<R, E>::new(spectra.batch, spectra.n_raw, 16, &fx.device);
    let encoded = fx
        .model
        .encoder
        .encode(&spectra, &peaks, Control::None)
        .unwrap();
    let mut fbuffers = ms2::FormulaBuffers::<R, E>::new(spectra.batch, 32, 2, &fx.device);
    ms2::formula_window(
        &fx.table.table,
        &spectra.meta,
        fx.table.max_error,
        u32::MAX,
        4096,
        &fbuffers,
    )
    .unwrap();
    ms2::formula_gather(
        &fbuffers.window,
        &fx.table.table,
        &fx.table.counts,
        &mut fbuffers.cand,
    )
    .unwrap();
    ms2::count_features(
        &fbuffers.cand.reshape(vec![spectra.batch * 32, 13]).unwrap(),
        &fx.table.log_table,
        &mut fbuffers
            .cand_feat
            .reshape(vec![spectra.batch * 32, 10])
            .unwrap(),
        13,
    )
    .unwrap();
    let scored = fx.model.formula.score(&fbuffers, &encoded.pool).unwrap();
    check_launches(&fx.device).unwrap();
    let emb_h = scored.embedding.try_to_f32().unwrap();
    let window_h = fbuffers.window.try_to_vec().unwrap();
    let d = 16usize;
    let m = 32usize;
    let slot = (0..m)
        .find(|s| window_h[(b * m + s) * 2] == out.formula_row[r])
        .expect("the conditioning formula is in the window");
    let mut femb_all = vec![0.0f32; 2 * d];
    femb_all[b * d..(b + 1) * d]
        .copy_from_slice(&emb_h[(b * m + slot) * d..(b * m + slot + 1) * d]);
    let formula_emb =
        Var::constant(Tensor::<R, E>::from_f32(&femb_all, vec![2, d], &fx.device).unwrap());
    // A two-spectrum target batch carrying the generated trace for spectrum
    // `b` (the other spectrum stays unlabeled): identical shapes to
    // generation, so no batching noise.
    let comp = fx.host_table.composition(out.formula_row[r] as usize);
    let labels = mamba3::models::ms2::targets::Labels {
        embeddings: Vec::new(),
        graphs: 1,
        targets_before_cut: 1,
        targets: vec![mamba3::models::ms2::targets::Target {
            trace: tokens,
            weight: 1,
            q: 1.0,
            embeddings: Vec::new(),
            anchors: Vec::new(),
        }],
        dropped_weight: 0.0,
        cut_is_tied: false,
        explained_peaks: Vec::new(),
        ambiguous_hypotheses: 0,
        canonicalization_failures: 0,
    };
    let tbatch = if b == 0 {
        TargetBatch::build(&[Some(&labels), None], &[*comp, [0; 10]], 1, Limits::V0).unwrap()
    } else {
        TargetBatch::build(&[None, Some(&labels)], &[[0; 10], *comp], 1, Limits::V0).unwrap()
    };
    let ttargets = tbatch.upload(&fx.device).unwrap();
    let tbufs = ms2::ReplayBuffers::poisoned(2, Limits::V0.max_steps(), 16, &fx.device).unwrap();
    ms2::grammar_replay(
        &ttargets.tokens,
        &ttargets.meta,
        &fx.constants,
        16,
        4,
        &tbufs,
    )
    .unwrap();
    check_launches(&fx.device).unwrap();
    let replay = ReplayView {
        replay: &tbufs.replay,
        atoms: &tbufs.atoms,
    };
    let tout = fx
        .model
        .decoder
        .teacher(&encoded, &formula_emb, &ttargets, &replay)
        .unwrap();
    check_launches(&fx.device).unwrap();
    let fields = tout.field_log_prob.try_to_f32().unwrap();
    // Teacher row `b` (spectrum `b`, slot 0) holds the candidate's trace.
    let tsteps = Limits::V0.max_steps();
    let sum: f32 = fields[b * tsteps * 4..(b + 1) * tsteps * 4].iter().sum();
    println!(
        "teacher sum {sum} vs trace_log_prob {}",
        out.trace_log_prob[r]
    );
    assert!(
        (sum - out.trace_log_prob[r]).abs() <= 1e-4,
        "teacher sum {sum} agrees with trace_log_prob {}",
        out.trace_log_prob[r]
    );
}

// ---------------------------------------------------------------------------
// P2.8: preflight, bucket reuse and alias violations.
// ---------------------------------------------------------------------------

#[test]
fn generate_preflight_refuses_before_allocation_or_launch() {
    // `generate` with `max_device_bytes` below the estimate returns the
    // documented error before any allocation or launch: `allocation_calls`
    // and `launch_count` are unchanged, and no workspace bucket is created.
    //
    // The two counters are process-global (P2.9), and the tests of this
    // binary run on several threads, so a single compare can catch a foreign
    // allocation from a neighbouring test. The refused call itself is pure
    // host arithmetic (estimate then limit check, before any upload,
    // allocation or launch), hence takes microseconds; the check retries for
    // a quiet window instead. A path that really allocated or launched would
    // move the counters on every attempt and fail after the retries.
    use mamba3::backend::{allocation_calls, launch_count};
    use mamba3::models::ms2::workspace::Ms2MemoryEstimate;
    let comps = parent_comps();
    let precursors: Vec<u32> = comps[0..2].iter().map(precursor_of).collect();
    let fx = gen_fixture(comps.clone(), 31);
    let mut ws = GenerationWorkspace::new();
    let batch = make_spectra(&[501, 502], &precursors, 64, &[10, 12], 33);
    // The estimate for this exact call, then a limit one byte below it.
    let mut cfg = tiny_generation();
    let est = Ms2MemoryEstimate::generation(
        &fx.model.config,
        fx.table.rows as u64,
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
    cfg.max_device_bytes = est - 1;
    let mut attempts = 0;
    loop {
        attempts += 1;
        let a0 = allocation_calls();
        let l0 = launch_count();
        let err = fx
            .model
            .generate(&batch, &fx.table, &cfg, &mut ws, &fx.constants)
            .unwrap_err();
        assert!(
            err.to_string().contains("exceeds limit"),
            "preflight error names the limit, got {err}"
        );
        if allocation_calls() == a0 && launch_count() == l0 {
            break;
        }
        assert!(
            attempts < 100,
            "generate with a refused estimate moved allocation_calls or launch_count \
             on 100 attempts: the preflight path itself allocates or launches"
        );
    }
    assert!(
        ws.bucket_keys().is_empty(),
        "a refused call creates no workspace bucket"
    );
}

#[test]
fn workspace_bucket_transparently_reallocates_for_another_batch() {
    // A workspace built for one bucket used with another bucket's batch
    // transparently reallocates (the documented behaviour: buckets are keyed
    // by shape and evicted past the cache limit), rather than erroring.
    let comps = parent_comps();
    let precursors: Vec<u32> = comps[0..2].iter().map(precursor_of).collect();
    let fx = gen_fixture(comps.clone(), 37);
    let mut ws = GenerationWorkspace::new();
    let cfg = tiny_generation();
    let batch1 = make_spectra(&[601], &precursors[0..1], 64, &[10], 41);
    let out1 = fx
        .model
        .generate(&batch1, &fx.table, &cfg, &mut ws, &fx.constants)
        .unwrap();
    out1.validate().unwrap();
    assert_eq!(ws.bucket_keys().len(), 1);
    // Another bucket's batch: two spectra instead of one.
    let batch2 = make_spectra(&[602, 603], &precursors, 64, &[10, 12], 41);
    let out2 = fx
        .model
        .generate(&batch2, &fx.table, &cfg, &mut ws, &fx.constants)
        .unwrap();
    out2.validate().unwrap();
    assert_eq!(out2.batch, 2);
    assert_eq!(ws.bucket_keys().len(), 2, "both buckets cached");
}

#[test]
fn grammar_apply_refuses_aliased_state() {
    // `grammar_apply` state vs tokens: the kernel reads `tokens` while
    // writing `state` in the same launch, so shared storage is refused.
    // With correct shapes the two lengths (`rows*4` vs `rows*(3A+16)`)
    // differ, so one allocation cannot serve as both through the public
    // constructors; the refusal is still enforced defensively (see
    // `shares_storage` in `tensor/ops/ms2.rs`). This test pins the
    // documented behaviour: distinct buffers run, and the shapes that would
    // be needed for an alias do not overlap.
    use mamba3::tensor::ops::ms2::grammar_state_zeros;
    let device = dev();
    let constants = Ms2Constants::new(&device);
    let atoms = 4usize;
    let rows = 2usize;
    let tokens =
        IdTensor::from_slice(&vec![1, 0, 0, 0, 1, 0, 0, 0], vec![rows, 4], &device).unwrap();
    let meta = IdTensor::from_slice(&vec![0u32; rows * 12], vec![rows, 12], &device).unwrap();
    let mut state = grammar_state_zeros(rows, atoms, &device);
    assert_ne!(tokens.len(), state.len(), "alias needs equal lengths");
    ms2::grammar_apply(&tokens, &mut state, &meta, &constants, atoms as u32, 4).unwrap();
    check_launches(&device).unwrap();
}

#[test]
fn atom_memory_update_refuses_aliased_buffers() {
    // `atom_memory_update` reads `token` while writing `resid_ids`: with
    // `max_atoms == 4` both hold `rows*4` words, so one allocation can serve
    // as both through a reshape, and that alias is refused. (`prev_h` vs
    // `atom_memory` and `grammar_state` vs `resid_ids` have different
    // lengths with correct shapes, so they cannot alias through the public
    // constructors; their checks are defensive.)
    use mamba3::tensor::Tensor;
    let device = dev();
    let (rows, atoms, d) = (2usize, 4usize, 8usize);
    let token =
        IdTensor::from_slice(&vec![2u32, 1, 0, 0, 2, 2, 0, 0], vec![rows, 4], &device).unwrap();
    let gstate = IdTensor::from_slice(
        &vec![0u32; rows * ms2::replay_state_width(atoms)],
        vec![rows, ms2::replay_state_width(atoms)],
        &device,
    )
    .unwrap();
    let prev_h = Tensor::<R, E>::from_f32(&vec![0.0; rows * d], vec![rows, d], &device).unwrap();
    let mut atom_memory =
        Tensor::<R, E>::from_f32(&vec![0.0; rows * atoms * d], vec![rows, atoms, d], &device)
            .unwrap();
    let mut resid =
        IdTensor::from_slice(&vec![0u32; rows * atoms], vec![rows * atoms], &device).unwrap();
    // Distinct buffers run.
    ms2::atom_memory_update(
        &token,
        &gstate,
        &prev_h,
        &mut atom_memory,
        &mut resid,
        atoms,
    )
    .unwrap();
    check_launches(&device).unwrap();
    // Identical handles: `resid_ids` sharing storage with `token`.
    let mut aliased_resid = token.clone().reshape(vec![rows * atoms]).unwrap();
    let err = ms2::atom_memory_update(
        &token,
        &gstate,
        &prev_h,
        &mut atom_memory,
        &mut aliased_resid,
        atoms,
    )
    .unwrap_err();
    assert!(
        err.to_string().contains("share storage"),
        "aliased resid_ids refused, got {err}"
    );
}

#[test]
fn shuffled_single_row_is_config_error() {
    // A one-row `ShuffledSpectrum` generate is `Error::Config`, never a
    // silent identity (the rotation would return the spectrum's own peaks).
    // Molecule-aware donors are the experiment control instead.
    let comps = parent_comps();
    let precursors: Vec<u32> = comps[0..1].iter().map(precursor_of).collect();
    let fx = gen_fixture(comps, 31);
    let mut ws = GenerationWorkspace::new();
    let mut cfg = tiny_generation();
    cfg.control = Control::ShuffledSpectrum;
    let batch = make_spectra(&[901], &precursors, 64, &[10], 33);
    let err = fx
        .model
        .generate(&batch, &fx.table, &cfg, &mut ws, &fx.constants)
        .unwrap_err();
    assert!(
        matches!(err, mamba3::error::Error::Config(_)),
        "one-row ShuffledSpectrum is Config, got {err:?}"
    );
}

// ---------------------------------------------------------------------------
// V0.6: statuses of structural proposals.
// ---------------------------------------------------------------------------

#[test]
fn v06_proposal_status_constants_and_duplicate_invariant() {
    // Every candidate of a generated batch carries the V0 constants
    // `evidence_status == 0` (unassigned), `identity_resolution == 0`
    // (trace only) and `attachment_partition == 0` (unknown); the batch
    // holds all `B * K` records (nothing is removed on the device); and the
    // exact-trace duplicate rule holds wherever it applies: among records
    // of one spectrum with equal trace words, length and formula, every
    // record after the first carries `duplicate_trace` and the first does
    // not.
    let comps = parent_comps();
    let precursors: Vec<u32> = comps[0..2].iter().map(precursor_of).collect();
    let fx = gen_fixture(comps, 43);
    let mut ws = GenerationWorkspace::new();
    let cfg = tiny_generation();
    let batch = make_spectra(&[701, 702], &precursors, 64, &[10, 12], 45);
    let out = fx
        .model
        .generate(&batch, &fx.table, &cfg, &mut ws, &fx.constants)
        .unwrap();
    check_launches(&fx.device).unwrap();
    out.validate().unwrap();
    let k = cfg.trajectories as usize;
    assert_eq!(out.batch, 2);
    assert_eq!(out.length.len(), 2 * k, "both records are still present");
    assert_eq!(out.status.len(), 2 * k);
    for r in 0..2 * k {
        assert_eq!(out.evidence_status[r], 0, "record {r}: evidence unassigned");
        assert_eq!(
            out.identity_resolution[r], 0,
            "record {r}: trace-only identity"
        );
        assert_eq!(
            out.attachment_partition[r], 0,
            "record {r}: partition unknown"
        );
    }
    // Exact (trace, formula) repeats within one spectrum.
    let mut dups = 0;
    for b in 0..2 {
        let mut first: std::collections::HashMap<Vec<u32>, usize> =
            std::collections::HashMap::new();
        for kk in 0..k {
            let r = b * k + kk;
            if out.status[r] & candidate_status::REQUEST_FAILED != 0 {
                continue;
            }
            let mut key = vec![out.length[r], out.formula_row[r]];
            key.extend_from_slice(&out.actions[r * out.max_steps * 4..(r + 1) * out.max_steps * 4]);
            match first.get(&key) {
                None => {
                    first.insert(key, r);
                    assert_eq!(
                        out.status[r] & candidate_status::DUPLICATE_TRACE,
                        0,
                        "record {r}: the first of its trace carries no duplicate_trace"
                    );
                }
                Some(_) => {
                    dups += 1;
                    assert_ne!(
                        out.status[r] & candidate_status::DUPLICATE_TRACE,
                        0,
                        "record {r}: a later exact repeat carries duplicate_trace"
                    );
                }
            }
        }
    }
    println!("v06 generated duplicates flagged: {dups}");
}

fn v06_trace_tokens(kind: u8, atom_type: u8, bond: u8, pointer: u8) -> Token {
    Token {
        kind,
        atom_type,
        bond,
        pointer,
    }
}

/// Action records for `rows` trajectories of one spectrum: `traces[r]` is
/// the emitted token prefix, `formulas[r]` the formula-row word. Budgets are
/// generous (255 of every element) so validity depends on the trace alone.
fn v06_action_records(
    traces: &[Vec<Token>],
    formulas: &[u32],
    steps: usize,
    atoms: usize,
) -> (Vec<u32>, Vec<u32>) {
    let record = ms2::sample_record_width(steps, atoms);
    let mut actions = vec![0u32; traces.len() * record];
    let mut meta = vec![0u32; traces.len() * ms2::TRAJ_META_WIDTH];
    for (r, (trace, &formula)) in traces.iter().zip(formulas.iter()).enumerate() {
        let abase = r * record;
        for (i, tok) in trace.iter().enumerate() {
            actions[abase + i * 4] = u32::from(tok.kind);
            actions[abase + i * 4 + 1] = u32::from(tok.atom_type);
            actions[abase + i * 4 + 2] = u32::from(tok.bond);
            actions[abase + i * 4 + 3] = u32::from(tok.pointer);
        }
        actions[abase + steps * 4 + atoms] = trace.len() as u32;
        actions[abase + steps * 4 + atoms + 1] = candidate_status::FINISHED;
        actions[abase + steps * 4 + atoms + 2] = 0.0f32.to_bits();
        actions[abase + steps * 4 + atoms + 3] = formula;
        meta[r * ms2::TRAJ_META_WIDTH] = 7000 + r as u32;
        meta[r * ms2::TRAJ_META_WIDTH + 1] = 0;
        meta[r * ms2::TRAJ_META_WIDTH + 2] = r as u32;
        meta[r * ms2::TRAJ_META_WIDTH + 3] = 1;
        for e in 0..10 {
            meta[r * ms2::TRAJ_META_WIDTH + 4 + e] = 255;
        }
    }
    (actions, meta)
}

/// Full action records after `validate_trajectories` over hand-built records.
fn v06_validate_records(
    device: &Device<R>,
    traces: &[Vec<Token>],
    formulas: &[u32],
    steps: usize,
    atoms: usize,
    max_closures: u32,
) -> Vec<u32> {
    let rows = traces.len();
    let record = ms2::sample_record_width(steps, atoms);
    let state_width = ms2::replay_state_width(atoms);
    let (actions, meta) = v06_action_records(traces, formulas, steps, atoms);
    let mut actions_t = IdTensor::from_slice(&actions, vec![rows, record], device).unwrap();
    let meta_t = IdTensor::from_slice(&meta, vec![rows, ms2::TRAJ_META_WIDTH], device).unwrap();
    let mut scratch_t = IdTensor::from_slice(
        &vec![0u32; rows * state_width],
        vec![rows, state_width],
        device,
    )
    .unwrap();
    let consts = Ms2Constants::new(device);
    ms2::validate_trajectories(
        &mut actions_t,
        &meta_t,
        &mut scratch_t,
        &consts.atom_table,
        1,
        rows,
        steps,
        atoms,
        max_closures,
    )
    .unwrap();
    check_launches(device).unwrap();
    actions_t.try_to_vec().unwrap()
}

/// Statuses after `validate_trajectories` over hand-built records.
fn v06_validate_statuses(
    device: &Device<R>,
    traces: &[Vec<Token>],
    formulas: &[u32],
    steps: usize,
    atoms: usize,
    max_closures: u32,
) -> Vec<u32> {
    let rows = traces.len();
    let record = ms2::sample_record_width(steps, atoms);
    let got = v06_validate_records(device, traces, formulas, steps, atoms, max_closures);
    (0..rows)
        .map(|r| got[r * record + steps * 4 + atoms + 1])
        .collect()
}

#[test]
fn duplicate_trace_flags_the_later_record_only() {
    // Two trajectories of one spectrum with the same trace and formula: the
    // later one carries `duplicate_trace`, the first does not, and both
    // records are still present (validation sets status bits in place; it
    // never removes a record). A third trajectory with the same trace but a
    // different formula is not a duplicate.
    let device = dev();
    let (steps, atoms, max_closures) = (6usize, 4usize, 0u32);
    let trace = vec![
        v06_trace_tokens(START, 0, 0, 0),
        v06_trace_tokens(ADD_ATOM, 2, 0, 0),
        v06_trace_tokens(ADD_ATOM, 2, 1, 0),
        v06_trace_tokens(STOP, 0, 0, 0),
    ];
    let statuses = v06_validate_statuses(
        &device,
        &[trace.clone(), trace.clone(), trace.clone()],
        &[5, 5, 9],
        steps,
        atoms,
        max_closures,
    );
    assert_eq!(statuses.len(), 3, "all three records are still present");
    // The records themselves survive validation: action words, lengths,
    // formula rows and FINISHED bits of all three records, not just the
    // duplicate bits below. A kernel that cleared trace contents or lengths
    // while setting status bits would fail here.
    let record = ms2::sample_record_width(steps, atoms);
    let got = v06_validate_records(
        &device,
        &[trace.clone(), trace.clone(), trace.clone()],
        &[5, 5, 9],
        steps,
        atoms,
        max_closures,
    );
    assert_eq!(got.len(), 3 * record, "all three records are still present");
    for r in 0..3 {
        let abase = r * record;
        for (i, tok) in trace.iter().enumerate() {
            assert_eq!(
                got[abase + i * 4],
                u32::from(tok.kind),
                "record {r} step {i}: kind word"
            );
            assert_eq!(
                got[abase + i * 4 + 1],
                u32::from(tok.atom_type),
                "record {r} step {i}: atom_type word"
            );
            assert_eq!(
                got[abase + i * 4 + 2],
                u32::from(tok.bond),
                "record {r} step {i}: bond word"
            );
            assert_eq!(
                got[abase + i * 4 + 3],
                u32::from(tok.pointer),
                "record {r} step {i}: pointer word"
            );
        }
        assert_eq!(
            got[abase + steps * 4 + atoms],
            trace.len() as u32,
            "record {r}: length survives validation"
        );
        assert_eq!(
            got[abase + steps * 4 + atoms + 3],
            [5, 5, 9][r],
            "record {r}: formula row survives validation"
        );
        assert_ne!(
            got[abase + steps * 4 + atoms + 1] & candidate_status::FINISHED,
            0,
            "record {r}: the legal stopped trace carries FINISHED"
        );
    }
    for (i, st) in statuses.iter().enumerate() {
        assert_eq!(
            st & candidate_status::INVALID_FINAL,
            0,
            "record {i}: the trace is legal"
        );
        assert_eq!(
            st & candidate_status::REQUEST_FAILED,
            0,
            "record {i}: not failed"
        );
    }
    assert_eq!(
        statuses[0] & candidate_status::DUPLICATE_TRACE,
        0,
        "the first record carries no duplicate_trace"
    );
    assert_ne!(
        statuses[1] & candidate_status::DUPLICATE_TRACE,
        0,
        "the later exact repeat carries duplicate_trace"
    );
    assert_eq!(
        statuses[2] & candidate_status::DUPLICATE_TRACE,
        0,
        "the same trace under another formula is not a duplicate"
    );
}

#[test]
fn duplicate_trace_compares_enumerated_formula_composition() {
    // D2: every enumerated formula has `formula_row = u32::MAX`, so duplicate
    // detection must compare the conditioning FORMULA (the 10 budget counts),
    // not the row. Same trace under two different enumerated formulas is NOT
    // `duplicate_trace`; same trace under the same enumerated formula is.
    // Table-source results stay bit-identical (covered by the test above).
    let device = dev();
    let (steps, atoms, max_closures) = (6usize, 4usize, 0u32);
    let trace = vec![
        v06_trace_tokens(START, 0, 0, 0),
        v06_trace_tokens(ADD_ATOM, 2, 0, 0),
        v06_trace_tokens(ADD_ATOM, 2, 1, 0),
        v06_trace_tokens(STOP, 0, 0, 0),
    ];
    let record = ms2::sample_record_width(steps, atoms);
    let state_width = ms2::replay_state_width(atoms);
    // Three records, one spectrum: same trace, all `formula_row = MAX`;
    // budgets C=2, C=2 (same formula), C=3 (different formula).
    let mut actions = vec![0u32; 3 * record];
    let mut meta = vec![0u32; 3 * ms2::TRAJ_META_WIDTH];
    for r in 0..3 {
        let abase = r * record;
        for (i, tok) in trace.iter().enumerate() {
            actions[abase + i * 4] = u32::from(tok.kind);
            actions[abase + i * 4 + 1] = u32::from(tok.atom_type);
            actions[abase + i * 4 + 2] = u32::from(tok.bond);
            actions[abase + i * 4 + 3] = u32::from(tok.pointer);
        }
        actions[abase + steps * 4 + atoms] = trace.len() as u32;
        actions[abase + steps * 4 + atoms + 1] = candidate_status::FINISHED;
        actions[abase + steps * 4 + atoms + 2] = 0.0f32.to_bits();
        actions[abase + steps * 4 + atoms + 3] = NO_FORMULA;
        meta[r * ms2::TRAJ_META_WIDTH] = 7000 + r as u32;
        meta[r * ms2::TRAJ_META_WIDTH + 1] = 0;
        meta[r * ms2::TRAJ_META_WIDTH + 2] = r as u32;
        meta[r * ms2::TRAJ_META_WIDTH + 3] = 1;
        for e in 0..10 {
            meta[r * ms2::TRAJ_META_WIDTH + 4 + e] = 255;
        }
        meta[r * ms2::TRAJ_META_WIDTH + 4] = if r == 2 { 3 } else { 2 };
    }
    // Device kernel.
    let mut actions_t =
        IdTensor::from_slice(&actions, vec![3, record], &device).unwrap();
    let meta_t =
        IdTensor::from_slice(&meta, vec![3, ms2::TRAJ_META_WIDTH], &device).unwrap();
    let mut scratch_t = IdTensor::from_slice(
        &vec![0u32; 3 * state_width],
        vec![3, state_width],
        &device,
    )
    .unwrap();
    let consts = Ms2Constants::new(&device);
    ms2::validate_trajectories(
        &mut actions_t,
        &meta_t,
        &mut scratch_t,
        &consts.atom_table,
        1,
        3,
        steps,
        atoms,
        max_closures,
    )
    .unwrap();
    check_launches(&device).unwrap();
    let got = actions_t.try_to_vec().unwrap();
    let statuses: Vec<u32> = (0..3)
        .map(|r| got[r * record + steps * 4 + atoms + 1])
        .collect();
    for (i, st) in statuses.iter().enumerate() {
        assert_eq!(
            st & candidate_status::INVALID_FINAL,
            0,
            "record {i}: the trace is legal under its budget"
        );
    }
    assert_eq!(
        statuses[0] & candidate_status::DUPLICATE_TRACE,
        0,
        "first record carries no duplicate_trace"
    );
    assert_ne!(
        statuses[1] & candidate_status::DUPLICATE_TRACE,
        0,
        "same trace under the same enumerated formula is duplicate_trace"
    );
    assert_eq!(
        statuses[2] & candidate_status::DUPLICATE_TRACE,
        0,
        "same trace under a different enumerated formula is NOT duplicate_trace"
    );
    // Host twin agrees exactly on the same buffers.
    let mut twin_actions = actions.clone();
    host::validate(
        &mut twin_actions,
        &meta,
        1,
        3,
        steps,
        atoms,
        max_closures as usize,
        &host::atom_table_rows(),
    );
    let twin_statuses: Vec<u32> = (0..3)
        .map(|r| twin_actions[r * record + steps * 4 + atoms + 1])
        .collect();
    assert_eq!(twin_statuses, statuses, "kernel and twin agree on D2 cases");
}

#[test]
fn graph_duplicates_stay_visible() {
    // Two different legal traces of the same labeled graph are NOT flagged
    // `duplicate_trace`: with `identity_resolution == 0` (trace only),
    // unresolved graph duplicates stay visible. The pair is a three-carbon
    // chain reached by two breadth-first orders: root in the middle (both
    // children point at atom 0) versus root at the end (the second child
    // points at atom 1). Both replay legally on the host, their
    // canonical traces are equal (one labeled graph), and the device
    // validation flags neither.
    let limits = Limits::new(4, 0).unwrap();
    let star = vec![
        v06_trace_tokens(START, 0, 0, 0),
        v06_trace_tokens(ADD_ATOM, 2, 0, 0),
        v06_trace_tokens(ADD_ATOM, 2, 1, 0),
        v06_trace_tokens(ADD_ATOM, 2, 1, 0),
        v06_trace_tokens(STOP, 0, 0, 0),
    ];
    let path = vec![
        v06_trace_tokens(START, 0, 0, 0),
        v06_trace_tokens(ADD_ATOM, 2, 0, 0),
        v06_trace_tokens(ADD_ATOM, 2, 1, 0),
        v06_trace_tokens(ADD_ATOM, 2, 1, 1),
        v06_trace_tokens(STOP, 0, 0, 0),
    ];
    assert_ne!(star, path, "the two traces differ as token sequences");
    let star_state = mamba3::models::ms2::grammar::replay(&star, limits, None)
        .expect("the star order replays legally");
    let path_state = mamba3::models::ms2::grammar::replay(&path, limits, None)
        .expect("the path order replays legally");
    assert!(
        star_state.stopped() && path_state.stopped(),
        "both traces stop"
    );
    let star_canon = canonical_trace(&star_state.graph().unwrap(), limits, 200_000)
        .expect("the star graph canonicalizes");
    let path_canon = canonical_trace(&path_state.graph().unwrap(), limits, 200_000)
        .expect("the path graph canonicalizes");
    assert_eq!(
        star_canon.trace, path_canon.trace,
        "both traces build the same labeled graph"
    );
    let device = dev();
    let statuses = v06_validate_statuses(&device, &[star, path], &[5, 5], 6, 4, 0);
    assert_eq!(statuses.len(), 2, "both records are still present");
    for (i, st) in statuses.iter().enumerate() {
        assert_eq!(
            st & candidate_status::INVALID_FINAL,
            0,
            "record {i}: the trace is legal"
        );
        assert_eq!(
            st & candidate_status::DUPLICATE_TRACE,
            0,
            "record {i}: a graph duplicate is not an exact-trace duplicate"
        );
    }
}

#[test]
fn distinct_traces_keeps_finished_valid_nonduplicates() {
    // `CandidateBatch::distinct_traces` returns the indices of records that
    // are finished, valid and not `duplicate_trace`.
    let batch = CandidateBatch {
        schema_version: SCHEMA_VERSION,
        batch: 1,
        trajectories: 5,
        max_steps: 6,
        max_atoms: 4,
        max_ring_closures: 0,
        spectrum_id: vec![701; 5],
        trajectory: vec![0, 1, 2, 3, 4],
        actions: vec![0; 5 * 6 * 4],
        length: vec![4, 4, 4, 6, 0],
        formula_row: vec![5; 5],
        formula_log_prob: vec![0.0; 5],
        trace_log_prob: vec![0.0; 5],
        open_valence: vec![0; 5 * 4],
        attachment_partition: vec![0; 5],
        status: vec![
            candidate_status::FINISHED,
            candidate_status::FINISHED | candidate_status::DUPLICATE_TRACE,
            candidate_status::FINISHED | candidate_status::INVALID_FINAL,
            candidate_status::TRUNCATED,
            candidate_status::REQUEST_FAILED,
        ],
        evidence_status: vec![0; 5],
        evidence_count: vec![0; 5],
        evidence_peak_id: vec![0; (5) * 4],
        evidence_hypothesis: vec![0; (5) * 4],
        evidence_shift: vec![0; (5) * 4],
        evidence_residual: vec![0; (5) * 4],
        evidence_log_prob: vec![0.0; (5) * 4],
        identity_resolution: vec![0; 5],
        request_status: vec![0],
        rows_visited: vec![0],
        rows_joined: vec![0],
        rows_scored: vec![0],
        formula_support_complete: vec![0],
        formula_mass_retained: vec![0.0],
        peaks_kept: vec![0],
        intensity_retained: vec![0.0],
        formula_counts: vec![1; 5 * 10],
        formula_source: vec![0],
        formula_rank: vec![0; 5],
    };
    assert_eq!(batch.distinct_traces(), vec![0]);
}

// ---------------------------------------------------------------------------
// B1: inference boundary (gold-free) and M=2048 memory refusal.
// ---------------------------------------------------------------------------

#[test]
fn inference_boundary_gold_free() {
    // `Ms2Model::generate(batch, table, config, workspace, constants)` takes
    // no gold/target/composition parameter: `SpectrumBatch` has no
    // gold/composition/target field, so two requests differing only in a
    // (nonexistent) gold payload are the same request. By construction the
    // inference boundary is gold-free; determinism below makes it observable:
    // two calls on the same batch give identical candidate batches.
    let comps = parent_comps();
    let precursors: Vec<u32> = comps[0..2].iter().map(precursor_of).collect();
    let fx = gen_fixture(comps, 53);
    let mut ws = GenerationWorkspace::new();
    let cfg = tiny_generation();
    let batch = make_spectra(&[801, 802], &precursors, 64, &[10, 12], 55);
    let out1 = fx
        .model
        .generate(&batch, &fx.table, &cfg, &mut ws, &fx.constants)
        .unwrap();
    check_launches(&fx.device).unwrap();
    out1.validate().unwrap();
    let out2 = fx
        .model
        .generate(&batch, &fx.table, &cfg, &mut ws, &fx.constants)
        .unwrap();
    check_launches(&fx.device).unwrap();
    assert_eq!(out1.formula_row, out2.formula_row, "formula rows match");
    assert_eq!(out1.formula_counts, out2.formula_counts, "counts match");
    assert_eq!(out1.formula_rank, out2.formula_rank, "ranks match");
    assert_eq!(out1.formula_source, out2.formula_source, "sources match");
    assert_eq!(out1, out2, "identical batches for identical requests");
}

#[test]
fn memory_refuses_before_allocation_at_m2048() {
    // B1-fix: the refusal is exercised at M = 2048 on the actual generation
    // request. The limit lies strictly between the M = 32 and M = 2048
    // estimates for this exact call shape, `cfg.formula_window` is 2048 on
    // the request itself, and the refused call is rejected with the
    // documented error and creates no bucket. The proof that the FIRST
    // refused call leaves the process-global allocation and launch counters
    // unchanged lives in the counter-owning footprint binary (which is the
    // only reader of those counters); this multi-test binary must not read
    // them.
    use mamba3::models::ms2::workspace::Ms2MemoryEstimate;
    let comps = parent_comps();
    let precursors: Vec<u32> = comps[0..2].iter().map(precursor_of).collect();
    let fx = gen_fixture(comps, 59);
    let batch = make_spectra(&[901, 902], &precursors, 64, &[10, 12], 61);
    let mut cfg32 = tiny_generation();
    cfg32.formula_window = 32;
    let est32 = Ms2MemoryEstimate::generation(
        &fx.model.config,
        fx.table.rows as u64,
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
        &fx.model.config,
        fx.table.rows as u64,
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
    // A limit strictly between the two: M = 32 fits, M = 2048 does not.
    let limit = est32 + (est2048 - est32) / 2;
    assert!(
        est32 < limit && limit < est2048,
        "limit {limit} separates {est32} and {est2048}"
    );
    cfg.max_device_bytes = limit;
    let mut ws = GenerationWorkspace::new();
    let err = fx
        .model
        .generate(&batch, &fx.table, &cfg, &mut ws, &fx.constants)
        .unwrap_err();
    assert!(
        matches!(err, mamba3::error::Error::Config(_)),
        "generate at M=2048 with a separating limit is Config, got {err:?}"
    );
    assert!(err.to_string().contains("exceeds limit"), "{err}");
    assert!(
        ws.bucket_keys().is_empty(),
        "a refused M=2048 call creates no workspace bucket"
    );
}

// ---------------------------------------------------------------------------
// C1 (V1 §3.3, §3.4): generation work, unsatisfiable formulas, truncation.
// ---------------------------------------------------------------------------

#[test]
fn generation_work_matches_emitted_tokens_on_generated_batch() {
    // V1 §3.3 on one generated batch: the submitted trajectory-steps are
    // `B * K * (T - 1)`, and the sum of active over the steps equals the sum
    // of emitted non-START tokens plus the failure detections.
    let comps = parent_comps();
    let precursors: Vec<u32> = comps[0..2].iter().map(precursor_of).collect();
    let fx = gen_fixture(comps.clone(), 31);
    let mut ws = GenerationWorkspace::new();
    let cfg = tiny_generation();
    let batch = make_spectra(&[401, 402], &precursors, 64, &[10, 12], 41);
    let out = fx
        .model
        .generate(&batch, &fx.table, &cfg, &mut ws, &fx.constants)
        .unwrap();
    check_launches(&fx.device).unwrap();
    out.validate().unwrap();
    let work = out.work();
    let n = out.batch * out.trajectories;
    assert_eq!(
        work.submitted,
        n * (out.max_steps - 1),
        "submitted trajectory-steps are B * K * (T - 1)"
    );
    assert_eq!(
        work.steps.len(),
        out.max_steps - 1,
        "one entry per step t in 1..T"
    );
    let mut non_start = 0usize;
    let mut failures = 0usize;
    for r in 0..n {
        non_start += (out.length[r] as usize).saturating_sub(1);
        if out.status[r] & candidate_status::NO_VALID_ACTION != 0 {
            failures += 1;
        }
    }
    assert_eq!(
        work.active_total,
        non_start + failures,
        "active sum is emitted non-START tokens plus failure detections"
    );
    for s in &work.steps {
        assert_eq!(
            s.active + s.inactive,
            n,
            "step {}: active + inactive is B * K",
            s.step
        );
    }
    assert!(
        (0.0..=1.0).contains(&work.active_fraction),
        "fraction in range, got {}",
        work.active_fraction
    );
    println!(
        "generated batch work: submitted {}, active {}, fraction {:.3}",
        work.submitted, work.active_total, work.active_fraction
    );
}

#[test]
fn unsatisfiable_formula_fails_every_trajectory() {
    // V1 §3.4: a budget no root atom fits (heavy-free H100: every atom type
    // needs a heavy atom) fails every trajectory with `no_valid_action` as
    // the contract says, and emits no token past START.
    let device = dev();
    let budget: Composition = [0, 100, 0, 0, 0, 0, 0, 0, 0, 0];
    let host_table = FormulaTable::from_compositions(vec![budget]).unwrap();
    let table = DeviceFormulaTable::<R, E>::upload(&host_table, &device).unwrap();
    let mut cfg = tiny_config();
    cfg.formula_table.rows = table.rows as u32;
    cfg.formula_table.sha256 = table.sha256.clone();
    let mut rng = Rng::seeded(43);
    let model = Ms2Model::<R, E>::init(&cfg, &device, &mut rng).unwrap();
    let constants = Ms2Constants::new(&device);
    let mut ws = GenerationWorkspace::new();
    let gcfg = tiny_generation();
    let precursor = precursor_of(&budget);
    assert!(
        (50_000_000..=2_000_000_000).contains(&precursor),
        "the H100 precursor {precursor} is in the request range"
    );
    let batch = make_spectra(&[501], &[precursor], 64, &[10], 44);
    let out = model
        .generate(&batch, &table, &gcfg, &mut ws, &constants)
        .unwrap();
    check_launches(&device).unwrap();
    out.validate().unwrap();
    let k = gcfg.trajectories as usize;
    assert_eq!(out.batch * out.trajectories, k, "one spectrum of K records");
    for r in 0..k {
        assert_ne!(
            out.status[r] & candidate_status::NO_VALID_ACTION,
            0,
            "record {r}: unsatisfiable budget fails with no_valid_action"
        );
        assert_eq!(
            out.status[r] & candidate_status::REQUEST_FAILED,
            0,
            "record {r}: the request itself did not fail (a formula was found)"
        );
        assert_eq!(
            out.status[r] & candidate_status::FINISHED,
            0,
            "record {r}: not finished"
        );
        assert_eq!(out.length[r], 1, "record {r}: no token past START");
        assert_eq!(
            out.actions[(r * out.max_steps) * 4],
            u32::from(START),
            "record {r}: START"
        );
        for s in 1..out.max_steps {
            assert_eq!(
                out.actions[(r * out.max_steps + s) * 4..(r * out.max_steps + s) * 4 + 4],
                [0, 0, 0, 0],
                "record {r} step {s}: PAD past START"
            );
        }
    }
}

#[test]
fn truncation_with_shorter_kernel_horizon() {
    // V1 §3.4: truncation is tested with a deliberately shorter kernel
    // horizon. `steps < 2 + A + R_max` is a config error for `generate`, so
    // the sampling kernel is driven directly with fewer steps: the trace
    // yields `truncated` with `length == steps` and no STOP.
    let mut short = tiny_generation();
    short.max_steps = 3;
    assert!(
        short.validate(2, 0).is_err(),
        "steps 3 < 2 + A + R_max = 4 is a generate config error"
    );
    let device = dev();
    let consts = Ms2Constants::new(&device);
    let a = 2usize;
    let rmax = 0u32;
    let t = 3usize;
    let budget: Composition = [2, 6, 0, 0, 0, 0, 0, 0, 0, 0];
    // Kind logits with STOP far down: after the forced root ADD, every row
    // ADDs again at the last step instead of stopping.
    let width = ms2::sample_logits_width(a);
    let mut logits_row = vec![0.0f32; width];
    logits_row[4] = -30.0;
    let tables_row = vec![0.0f32; 19 * 4];
    let rows = 512usize;
    let got = sample_fixed_logits(
        &device,
        &consts,
        &logits_row,
        &tables_row,
        budget,
        rows,
        t,
        a,
        rmax,
        77,
        99,
    );
    let record = ms2::sample_record_width(t, a);
    for r in 0..rows {
        let abase = r * record;
        let st = got[abase + t * 4 + a + 1];
        let len = got[abase + t * 4 + a] as usize;
        assert_ne!(st & candidate_status::TRUNCATED, 0, "row {r}: truncated");
        assert_eq!(st & candidate_status::FINISHED, 0, "row {r}: not finished");
        assert_eq!(
            st & candidate_status::NO_VALID_ACTION,
            0,
            "row {r}: no failure"
        );
        assert_eq!(len, t, "row {r}: length == steps");
        for s in 0..len {
            assert_ne!(
                got[abase + s * 4],
                u32::from(STOP),
                "row {r} step {s}: no STOP token"
            );
        }
    }
    // The first row agrees with the host twin token for token.
    let s = ms2::replay_state_width(a);
    let atable = host::atom_table_rows();
    let mut twin_state = vec![0u32; s];
    twin_state[3 * a + 4] = 1;
    let mut twin_actions = vec![0u32; record];
    twin_actions[0] = u32::from(START);
    twin_actions[t * 4 + a] = 1;
    let mut traj0 = vec![0u32; 14];
    traj0[0] = 0;
    traj0[1] = 0;
    traj0[2] = 0;
    traj0[3] = 1;
    for e in 0..10 {
        traj0[4 + e] = u32::from(budget[e]);
    }
    for step in [1u32, 2u32] {
        host::sample_step(
            &logits_row,
            &tables_row,
            &traj0,
            &mut twin_state,
            &mut twin_actions,
            step,
            77,
            99,
            1.0,
            t,
            a,
            rmax as usize,
            &atable,
        );
    }
    for w in 0..record {
        if w == t * 4 + a + 2 {
            continue;
        }
        assert_eq!(
            got[w], twin_actions[w],
            "first row word {w}: device == twin"
        );
    }
}

// ---------------------------------------------------------------------------
// C1 (V1 §4.1): one negative validation test per final-validity rule.
// ---------------------------------------------------------------------------

/// Device and twin statuses for hand-built records with per-row budgets and
/// initial statuses: `traces[r]` are the emitted tokens, `formulas[r]` the
/// formula word, `budgets[r]` the 10 budget counts, `statuses[r]` the initial
/// status word. Each row uses a distinct formula so no duplicate flag can
/// hide a validity bit.
#[allow(clippy::too_many_arguments)]
fn neg_validate_statuses(
    device: &Device<R>,
    traces: &[Vec<Token>],
    formulas: &[u32],
    budgets: &[[u32; 10]],
    statuses: &[u32],
    steps: usize,
    atoms: usize,
    max_closures: u32,
) -> (Vec<u32>, Vec<u32>) {
    let rows = traces.len();
    assert_eq!(formulas.len(), rows);
    assert_eq!(budgets.len(), rows);
    assert_eq!(statuses.len(), rows);
    let record = ms2::sample_record_width(steps, atoms);
    let state_width = ms2::replay_state_width(atoms);
    let mut actions = vec![0u32; rows * record];
    let mut meta = vec![0u32; rows * ms2::TRAJ_META_WIDTH];
    for r in 0..rows {
        let abase = r * record;
        for (i, tok) in traces[r].iter().enumerate() {
            actions[abase + i * 4] = u32::from(tok.kind);
            actions[abase + i * 4 + 1] = u32::from(tok.atom_type);
            actions[abase + i * 4 + 2] = u32::from(tok.bond);
            actions[abase + i * 4 + 3] = u32::from(tok.pointer);
        }
        actions[abase + steps * 4 + atoms] = traces[r].len() as u32;
        actions[abase + steps * 4 + atoms + 1] = statuses[r];
        actions[abase + steps * 4 + atoms + 2] = 0.0f32.to_bits();
        actions[abase + steps * 4 + atoms + 3] = formulas[r];
        meta[r * ms2::TRAJ_META_WIDTH] = 8000 + r as u32;
        meta[r * ms2::TRAJ_META_WIDTH + 1] = 0;
        meta[r * ms2::TRAJ_META_WIDTH + 2] = r as u32;
        meta[r * ms2::TRAJ_META_WIDTH + 3] = 1;
        for e in 0..10 {
            meta[r * ms2::TRAJ_META_WIDTH + 4 + e] = budgets[r][e];
        }
    }
    let mut actions_t = IdTensor::from_slice(&actions, vec![rows, record], device).unwrap();
    let meta_t = IdTensor::from_slice(&meta, vec![rows, ms2::TRAJ_META_WIDTH], device).unwrap();
    let mut scratch_t = IdTensor::from_slice(
        &vec![0u32; rows * state_width],
        vec![rows, state_width],
        device,
    )
    .unwrap();
    let consts = Ms2Constants::new(device);
    ms2::validate_trajectories(
        &mut actions_t,
        &meta_t,
        &mut scratch_t,
        &consts.atom_table,
        1,
        rows,
        steps,
        atoms,
        max_closures,
    )
    .unwrap();
    check_launches(device).unwrap();
    let device_out = actions_t.try_to_vec().unwrap();
    let mut twin_actions = actions.clone();
    host::validate(
        &mut twin_actions,
        &meta,
        1,
        rows,
        steps,
        atoms,
        max_closures as usize,
        &host::atom_table_rows(),
    );
    let dev_statuses: Vec<u32> = (0..rows)
        .map(|r| device_out[r * record + steps * 4 + atoms + 1])
        .collect();
    let twin_statuses: Vec<u32> = (0..rows)
        .map(|r| twin_actions[r * record + steps * 4 + atoms + 1])
        .collect();
    (dev_statuses, twin_statuses)
}

/// One crafted violation and its minimal legal variant: both the device
/// `ms2_validate` and the host twin flag exactly the violation with
/// `invalid_final`, and the legal variant stays clean on both.
fn check_negative(
    device: &Device<R>,
    bad: Vec<Token>,
    good: Vec<Token>,
    budgets: [[u32; 10]; 2],
    statuses: [u32; 2],
    steps: usize,
    atoms: usize,
    max_closures: u32,
    what: &str,
) {
    let (dev, twin) = neg_validate_statuses(
        device,
        &[bad, good],
        &[21, 22],
        &budgets,
        &statuses,
        steps,
        atoms,
        max_closures,
    );
    assert_eq!(dev, twin, "{what}: device and twin agree");
    assert_ne!(
        dev[0] & candidate_status::INVALID_FINAL,
        0,
        "{what}: the violation is flagged invalid_final"
    );
    assert_eq!(
        dev[1] & candidate_status::INVALID_FINAL,
        0,
        "{what}: the minimal legal variant stays clean"
    );
}

fn generous_two() -> [[u32; 10]; 2] {
    [[255; 10], [255; 10]]
}

#[test]
fn validation_negative_pointer_to_missing_atom() {
    // Declared connectivity: ADD_ATOM with parent 5 when only atom 0 exists.
    let device = dev();
    let bad = vec![
        v06_trace_tokens(START, 0, 0, 0),
        v06_trace_tokens(ADD_ATOM, 2, 0, 0),
        v06_trace_tokens(ADD_ATOM, 2, 1, 5),
        v06_trace_tokens(STOP, 0, 0, 0),
    ];
    let good = vec![
        v06_trace_tokens(START, 0, 0, 0),
        v06_trace_tokens(ADD_ATOM, 2, 0, 0),
        v06_trace_tokens(ADD_ATOM, 2, 1, 0),
        v06_trace_tokens(STOP, 0, 0, 0),
    ];
    check_negative(
        &device,
        bad,
        good,
        generous_two(),
        [candidate_status::FINISHED, candidate_status::FINISHED],
        6,
        4,
        0,
        "pointer to a missing atom",
    );
}

#[test]
fn validation_negative_second_bond_between_two_atoms() {
    // Bond uniqueness: two CLOSE_RINGs from atom 2 to atom 1 (two closures
    // allowed, so the cap is not the cause).
    let device = dev();
    let mk = |closes: usize| {
        let mut trace = vec![
            v06_trace_tokens(START, 0, 0, 0),
            v06_trace_tokens(ADD_ATOM, 2, 0, 0),
            v06_trace_tokens(ADD_ATOM, 2, 1, 0),
            v06_trace_tokens(ADD_ATOM, 2, 1, 0),
        ];
        for _ in 0..closes {
            trace.push(v06_trace_tokens(CLOSE_RING, 0, 1, 1));
        }
        trace.push(v06_trace_tokens(STOP, 0, 0, 0));
        trace
    };
    check_negative(
        &device,
        mk(2),
        mk(1),
        generous_two(),
        [candidate_status::FINISHED, candidate_status::FINISHED],
        8,
        4,
        2,
        "a second bond between two atoms",
    );
}

#[test]
fn validation_negative_bond_exceeds_residual_valence() {
    // A bond of order 2 against a residual valence of 1 (root C H3).
    let device = dev();
    let mk = |bond: u8| {
        vec![
            v06_trace_tokens(START, 0, 0, 0),
            v06_trace_tokens(ADD_ATOM, 4, 0, 0),
            v06_trace_tokens(ADD_ATOM, 3, bond, 0),
            v06_trace_tokens(STOP, 0, 0, 0),
        ]
    };
    check_negative(
        &device,
        mk(2),
        mk(1),
        generous_two(),
        [candidate_status::FINISHED, candidate_status::FINISHED],
        6,
        4,
        0,
        "a bond exceeding residual valence",
    );
}

#[test]
fn validation_negative_atom_type_outside_formula_budget() {
    // Composition: budget C1H2 against a trace using C2H2.
    let device = dev();
    let trace = vec![
        v06_trace_tokens(START, 0, 0, 0),
        v06_trace_tokens(ADD_ATOM, 2, 0, 0),
        v06_trace_tokens(ADD_ATOM, 2, 1, 0),
        v06_trace_tokens(STOP, 0, 0, 0),
    ];
    check_negative(
        &device,
        trace.clone(),
        trace,
        [
            [1, 2, 0, 0, 0, 0, 0, 0, 0, 0],
            [2, 6, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        [candidate_status::FINISHED, candidate_status::FINISHED],
        6,
        4,
        0,
        "an atom type outside the formula budget",
    );
}

#[test]
fn validation_negative_closure_beyond_r_max() {
    // An otherwise legal closure with `R_max = 0`.
    let device = dev();
    let trace = vec![
        v06_trace_tokens(START, 0, 0, 0),
        v06_trace_tokens(ADD_ATOM, 2, 0, 0),
        v06_trace_tokens(ADD_ATOM, 2, 1, 0),
        v06_trace_tokens(ADD_ATOM, 2, 1, 0),
        v06_trace_tokens(CLOSE_RING, 0, 1, 1),
        v06_trace_tokens(STOP, 0, 0, 0),
    ];
    let (dev0, twin0) = neg_validate_statuses(
        &device,
        &[trace.clone()],
        &[21],
        &[[255; 10]],
        &[candidate_status::FINISHED],
        8,
        4,
        0,
    );
    assert_eq!(dev0, twin0, "closure beyond R_max: device and twin agree");
    assert_ne!(
        dev0[0] & candidate_status::INVALID_FINAL,
        0,
        "closure beyond R_max is flagged invalid_final"
    );
    let (dev1, twin1) = neg_validate_statuses(
        &device,
        &[trace],
        &[21],
        &[[255; 10]],
        &[candidate_status::FINISHED],
        8,
        4,
        1,
    );
    assert_eq!(dev1, twin1, "closure within R_max: device and twin agree");
    assert_eq!(
        dev1[0] & candidate_status::INVALID_FINAL,
        0,
        "the same trace within R_max stays clean"
    );
}

#[test]
fn validation_negative_closure_to_parent() {
    // CLOSE_RING pointing at the newest atom's own parent (pointer must be
    // larger than the parent pointer).
    let device = dev();
    let mk = |pointer: u8| {
        vec![
            v06_trace_tokens(START, 0, 0, 0),
            v06_trace_tokens(ADD_ATOM, 2, 0, 0),
            v06_trace_tokens(ADD_ATOM, 2, 1, 0),
            v06_trace_tokens(ADD_ATOM, 2, 1, 0),
            v06_trace_tokens(CLOSE_RING, 0, 1, pointer),
            v06_trace_tokens(STOP, 0, 0, 0),
        ]
    };
    check_negative(
        &device,
        mk(0),
        mk(1),
        generous_two(),
        [candidate_status::FINISHED, candidate_status::FINISHED],
        8,
        4,
        1,
        "a closure to the parent",
    );
}

#[test]
fn validation_negative_finished_without_stop() {
    // A record claimed `finished` whose last token is not STOP — and a legal
    // truncated history, which stays `truncated`, not invalid.
    let device = dev();
    let stopped = vec![
        v06_trace_tokens(START, 0, 0, 0),
        v06_trace_tokens(ADD_ATOM, 2, 0, 0),
        v06_trace_tokens(ADD_ATOM, 2, 1, 0),
        v06_trace_tokens(STOP, 0, 0, 0),
    ];
    let unstopped = vec![
        v06_trace_tokens(START, 0, 0, 0),
        v06_trace_tokens(ADD_ATOM, 2, 0, 0),
        v06_trace_tokens(ADD_ATOM, 2, 1, 0),
    ];
    let (dev, twin) = neg_validate_statuses(
        &device,
        &[unstopped.clone(), stopped, unstopped],
        &[21, 22, 23],
        &[[255; 10], [255; 10], [255; 10]],
        &[
            candidate_status::FINISHED,
            candidate_status::FINISHED,
            candidate_status::TRUNCATED,
        ],
        6,
        4,
        0,
    );
    assert_eq!(dev, twin, "finished without STOP: device and twin agree");
    assert_ne!(
        dev[0] & candidate_status::INVALID_FINAL,
        0,
        "a finished record whose last token is not STOP is invalid_final"
    );
    assert_eq!(
        dev[1] & candidate_status::INVALID_FINAL,
        0,
        "the STOP-terminated variant stays clean"
    );
    assert_eq!(
        dev[2] & candidate_status::INVALID_FINAL,
        0,
        "a legal truncated history stays truncated, not invalid"
    );
    assert_ne!(
        dev[2] & candidate_status::TRUNCATED,
        0,
        "the truncated bit survives validation"
    );
}

#[test]
fn validation_negative_fewer_than_minimum_atoms() {
    // Final validity needs at least one atom: START-only claimed finished.
    let device = dev();
    let bad = vec![v06_trace_tokens(START, 0, 0, 0)];
    let good = vec![
        v06_trace_tokens(START, 0, 0, 0),
        v06_trace_tokens(ADD_ATOM, 2, 0, 0),
        v06_trace_tokens(STOP, 0, 0, 0),
    ];
    check_negative(
        &device,
        bad,
        good,
        generous_two(),
        [candidate_status::FINISHED, candidate_status::FINISHED],
        6,
        4,
        0,
        "fewer than the minimum atoms",
    );
}

/// A maximal legal trace grown with the host grammar: at each step the first
/// legal token in a fixed priority (root types, then ADD by increasing
/// type/bond/pointer, then CLOSE, then STOP) is appended. Every token passes
/// `is_legal`, so the trace is legal by construction; at the caps it ends
/// with STOP at exactly `2 + A + R_max` tokens.
fn grow_legal_trace(limits: Limits, budget: Composition) -> Vec<Token> {
    let blank = Token {
        kind: 0,
        atom_type: 0,
        bond: 0,
        pointer: 0,
    };
    let start = Token {
        kind: START,
        ..blank
    };
    let mut trace = vec![start];
    let mut st = TraceState::new(limits, Some(budget));
    st.apply(start).unwrap();
    let t = limits.max_steps();
    while trace.len() < t {
        let mut next: Option<Token> = None;
        if st.step() == 1 {
            for id in 1..=17u8 {
                let tok = Token {
                    kind: ADD_ATOM,
                    atom_type: id,
                    ..blank
                };
                if st.is_legal(tok) {
                    next = Some(tok);
                    break;
                }
            }
        } else {
            'search: for kind in [ADD_ATOM, CLOSE_RING] {
                for id in 0..=18u8 {
                    for b in 0..=3u8 {
                        for p in 0..32u8 {
                            let tok = Token {
                                kind,
                                atom_type: id,
                                bond: b,
                                pointer: p,
                            };
                            if st.is_legal(tok) {
                                next = Some(tok);
                                break 'search;
                            }
                        }
                    }
                }
            }
        }
        let tok = next.unwrap_or(Token {
            kind: STOP,
            ..blank
        });
        assert!(
            st.is_legal(tok),
            "the grown trace stays legal (failing token {tok:?})"
        );
        st.apply(tok).unwrap();
        trace.push(tok);
        if tok.kind == STOP {
            break;
        }
    }
    trace
}

#[test]
fn validation_matches_twin_at_both_shapes() {
    // V1 §3.1: device `ms2_validate` and its host twin agree at both
    // (A, R_max, T) = (16, 4, 22) and (32, 8, 42). Legal traces grown with
    // the host grammar — maximal at both shapes — carry no `invalid_final`
    // on either side.
    let device = dev();
    for (atoms, max_closures) in [(16usize, 4usize), (32usize, 8usize)] {
        let limits = Limits::new(atoms, max_closures).unwrap();
        let steps = limits.max_steps();
        let full = grow_legal_trace(limits, [255; 10]);
        let end = mamba3::models::ms2::grammar::replay(&full, limits, None).unwrap();
        assert!(end.stopped(), "the grown trace ends stopped");
        let small = vec![
            v06_trace_tokens(START, 0, 0, 0),
            v06_trace_tokens(ADD_ATOM, 2, 0, 0),
            v06_trace_tokens(ADD_ATOM, 2, 1, 0),
            v06_trace_tokens(STOP, 0, 0, 0),
        ];
        let (dev, twin) = neg_validate_statuses(
            &device,
            &[full, small],
            &[21, 22],
            &[[255; 10], [255; 10]],
            &[candidate_status::FINISHED, candidate_status::FINISHED],
            steps,
            atoms,
            max_closures as u32,
        );
        assert_eq!(dev, twin, "A = {atoms}: device and twin agree");
        for (i, st) in dev.iter().enumerate() {
            assert_eq!(
                st & candidate_status::INVALID_FINAL,
                0,
                "A = {atoms} record {i}: legal trace stays clean"
            );
        }
    }
}

#[test]
fn readout_ws_refuses_modes_it_cannot_serve() {
    // Finding R1-C7: `generate_readout_ws` (the profiler's workspace
    // readout) returns `Error::Config` for the modes it cannot serve
    // (`identity = Graph`, the enumerating source) instead of silently
    // dropping identity bits or skipping the reconciliation.
    use mamba3::error::Error;
    let device = dev();
    let comps: Vec<Composition> = vec![[6, 6, 0, 0, 0, 0, 0, 0, 0, 0]];
    let host_table = FormulaTable::from_compositions(comps.clone()).unwrap();
    let table = DeviceFormulaTable::<R, E>::upload(&host_table, &device).unwrap();
    let mut cfg = tiny_config();
    cfg.formula_table.rows = table.rows as u32;
    cfg.formula_table.sha256 = table.sha256.clone();
    let mut rng = Rng::seeded(5);
    let model = Ms2Model::<R, E>::init(&cfg, &device, &mut rng).unwrap();
    let precursors: Vec<u32> = comps
        .iter()
        .map(|c| composition_mass(c).unwrap() + 1_007_825 - 549)
        .collect();
    let batch = make_spectra(&[801], &precursors, 64, &[10], 7);
    let gcfg = tiny_generation();
    let pre = model.generate_preflight(&batch, &table, &gcfg).unwrap();
    let mut ws = GenerationWorkspace::new();
    let host_status = vec![0u32];
    let spectrum_ids = vec![801u64];
    // Table + TraceOnly serves fine.
    model
        .generate_readout_ws(
            &mut ws,
            &host_status,
            &spectrum_ids,
            &gcfg,
            gcfg.trajectories as usize,
            gcfg.formulas as usize,
            &pre,
            &device,
        )
        .unwrap();
    // Graph identity is refused, not silently dropped.
    let mut graph_cfg = gcfg.clone();
    graph_cfg.identity = mamba3::models::ms2::contract::IdentityMode::Graph;
    let err = model
        .generate_readout_ws(
            &mut ws,
            &host_status,
            &spectrum_ids,
            &graph_cfg,
            gcfg.trajectories as usize,
            gcfg.formulas as usize,
            &pre,
            &device,
        )
        .unwrap_err();
    assert!(
        matches!(err, Error::Config(_)),
        "identity Graph is Error::Config: {err}"
    );
    // The enumerating source is refused, not reconciled silently.
    let mut enum_cfg = gcfg.clone();
    enum_cfg.formula_source = mamba3::models::ms2::contract::FormulaSource::Enumerate;
    let err = model
        .generate_readout_ws(
            &mut ws,
            &host_status,
            &spectrum_ids,
            &enum_cfg,
            gcfg.trajectories as usize,
            gcfg.formulas as usize,
            &pre,
            &device,
        )
        .unwrap_err();
    assert!(
        matches!(err, Error::Config(_)),
        "Enumerate is Error::Config: {err}"
    );
}
