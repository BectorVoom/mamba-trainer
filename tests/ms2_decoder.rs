//! V0-B tests: device grammar replay, target batches, the graph-action
//! decoder (teacher forcing) and the graph loss.
//!
//! Every device call is followed by [`check_launches`], so a kernel that
//! failed to compile or run is an error rather than stale data.

#![cfg(feature = "backend")]

use mamba3::autograd::Var;
use mamba3::backend::{Device, check_launches, runtime_read_count};
use mamba3::backends::Auto;
use mamba3::models::ms2::batch::DeviceSpectra;
use mamba3::models::ms2::chem::{Composition, composition_mass, element_index};
use mamba3::models::ms2::contract::{
    Control, ModelConfig, SCHEMA_VERSION, SPECTRUM_SCHEMA_VERSION, SpectrumBatch,
};
use mamba3::models::ms2::decoder::{Ms2Decoder, ReplayView, graph_loss};
use mamba3::models::ms2::encoder::Ms2Encoder;
use mamba3::models::ms2::formula::{FormulaTable, WindowQuery};
use mamba3::models::ms2::formula_head::{DeviceFormulaTable, FormulaHead, gold_slots_host};
use mamba3::models::ms2::grammar::{
    ADD_ATOM, CLOSE_RING, LegalMasks, Limits, START, STOP, Token, TraceState,
};
use mamba3::models::ms2::targets::{Labels, Target};
use mamba3::models::ms2::targets_batch::TargetBatch;
use mamba3::nn::Module;
use mamba3::tensor::Tensor;
use mamba3::tensor::ops::index::IdTensor;
use mamba3::tensor::ops::ms2::{self, Ms2Constants, ReplayBuffers};
use mamba3::tensor::ops::random::Rng;
use mamba3::train::optim::{AdamWConfig, Optimizer};

type R = Auto;
type E = f32;

fn dev() -> Device<R> {
    Device::<R>::default()
}

/// Process-global counters (launch tallies, `runtime_read_count`) are read
/// by tests in this binary: every device-touching test holds this lock for
/// its whole body (as in `tests/ms2_fused_step.rs`), so `cargo test`
/// without `--test-threads 1` stays green.
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn serial() -> std::sync::MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

fn assert_close(actual: &[f32], expected: &[f32], tol: f32, what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        let ok = if e.abs() <= 1.0 {
            (a - e).abs() <= tol
        } else {
            (a - e).abs() <= tol * e.abs()
        };
        assert!(ok, "{what}: index {i} got {a}, want {e}");
    }
}

fn fixture() -> serde_json::Value {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/ms2/chemistry_v0.json");
    serde_json::from_str(&std::fs::read_to_string(path).expect("fixture readable"))
        .expect("fixture parses")
}

fn token_of(t: &serde_json::Value) -> Token {
    let a = t.as_array().expect("token array");
    Token {
        kind: a[0].as_u64().expect("kind") as u8,
        atom_type: a[1].as_u64().expect("atom_type") as u8,
        bond: a[2].as_u64().expect("bond") as u8,
        pointer: a[3].as_u64().expect("pointer") as u8,
    }
}

fn trace_of(t: &serde_json::Value) -> Vec<Token> {
    t.as_array()
        .expect("trace array")
        .iter()
        .map(token_of)
        .collect()
}

fn composition_of(formula: &serde_json::Value) -> Composition {
    let mut c: Composition = [0; 10];
    for (symbol, count) in formula.as_object().expect("formula object") {
        let e = element_index(symbol).expect("known element");
        c[e] = count.as_u64().expect("count") as u16;
    }
    c
}

fn start_token() -> Token {
    Token {
        kind: START,
        atom_type: 0,
        bond: 0,
        pointer: 0,
    }
}

/// A single-target [`Labels`] with weight 1 on `trace`.
fn single_target(trace: Vec<Token>) -> Labels {
    Labels {
        embeddings: Vec::new(),
        graphs: 1,
        targets_before_cut: 1,
        targets: vec![Target {
            trace,
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
    }
}

/// Tiny model config: `d = 16`, 2 attention heads, SSM state 8, `N = 16`.
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

/// Minimal valid spectrum batch with random peaks (the `ms2_encoder.rs`
/// shape: strictly increasing peak ids, precursor in range, adduct 1).
fn make_batch(
    spectrum_ids: &[u64],
    n_raw: usize,
    peak_counts: &[u32],
    precursor_base: u32,
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
        let precursor = precursor_base + bi as u32 * 10_000_000;
        for i in 0..count {
            peak_id[bi * n_raw + i] = i as u32;
            let f = rng.uniform_vec(1, 60_000_000.0, (precursor - 5_000_000) as f32)[0] as u32;
            mz[bi * n_raw + i] = f.max(50_000_001);
            let u = rng.uniform_vec(1, 0.0, 1.0)[0];
            intensity[bi * n_raw + i] = 0.5 + 2.0 * u;
        }
    }
    let mut precursor = vec![0u32; b];
    for (bi, _) in spectrum_ids.iter().enumerate() {
        precursor[bi] = precursor_base + bi as u32 * 10_000_000;
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
        precursor_mz_udalton: precursor,
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

/// Host replay of one trace: per-step masks and residuals before each token,
/// the atom-add steps and the first illegal step (`u32::MAX` when legal).
struct HostReplay {
    masks: Vec<LegalMasks>,
    resids: Vec<Vec<u8>>,
    add_steps: Vec<u32>,
    first_illegal: u32,
}

fn host_replay(
    trace: &[Token],
    limits: Limits,
    budget: Option<Composition>,
    a: usize,
) -> HostReplay {
    let mut st = TraceState::new(limits, budget);
    let mut masks = Vec::with_capacity(trace.len());
    let mut resids = Vec::with_capacity(trace.len());
    let mut add_steps = vec![u32::MAX; a];
    let mut first_illegal = u32::MAX;
    for (t, tok) in trace.iter().enumerate() {
        masks.push(st.masks(*tok));
        let mut r = vec![0u8; a];
        for (j, v) in st.residual_valence().iter().enumerate() {
            r[j] = *v;
        }
        resids.push(r);
        if st.is_legal(*tok) {
            let before = st.atoms();
            st.apply(*tok).unwrap();
            if tok.kind == ADD_ATOM && before < a {
                add_steps[before] = t as u32;
            }
        } else {
            first_illegal = t as u32;
            break;
        }
    }
    HostReplay {
        masks,
        resids,
        add_steps,
        first_illegal,
    }
}

/// One seeded random legal trace: uniform choices among the legal kinds,
/// types, bonds and pointers from the masks, STOP with probability 1/4 past
/// the root (forced at the length cap).
fn random_trace(rng: &mut Rng, limits: Limits, budget: Option<Composition>) -> Vec<Token> {
    let mut st = TraceState::new(limits, budget);
    let mut trace = vec![start_token()];
    st.apply(start_token()).unwrap();
    let pick =
        |rng: &mut Rng, n: usize| (rng.uniform_vec(1, 0.0, 1.0)[0] * n as f32) as usize % n.max(1);
    let set_bits = |word: u32| -> Vec<u32> { (0..32).filter(|i| word & (1 << i) != 0).collect() };
    loop {
        let step = trace.len();
        if step >= limits.max_steps() - 1 {
            break;
        }
        // STOP as soon as it is offered and chosen (past the root it always
        // is, once an atom exists).
        let kinds = set_bits(
            st.masks(Token {
                kind: 0,
                atom_type: 0,
                bond: 0,
                pointer: 0,
            })
            .kinds,
        );
        if kinds.is_empty() {
            break;
        }
        let stop_offered = kinds.contains(&u32::from(STOP));
        let kind = if stop_offered && (step >= limits.max_steps() - 1 || pick(rng, 4) == 0) {
            STOP
        } else {
            let mut non_stop: Vec<u32> = kinds
                .into_iter()
                .filter(|&k| k != u32::from(STOP))
                .collect();
            if non_stop.is_empty() {
                STOP
            } else {
                non_stop.sort();
                non_stop[pick(rng, non_stop.len())] as u8
            }
        };
        let taken = |kind: u8| Token {
            kind,
            atom_type: 0,
            bond: 0,
            pointer: 0,
        };
        let token = if kind == ADD_ATOM && step == 1 {
            let types = set_bits(st.masks(taken(kind)).atom_types);
            if types.is_empty() {
                break;
            }
            Token {
                kind,
                atom_type: types[pick(rng, types.len())] as u8,
                bond: 0,
                pointer: 0,
            }
        } else if kind == ADD_ATOM {
            let types = set_bits(st.masks(taken(kind)).atom_types);
            if types.is_empty() {
                break;
            }
            let ty = types[pick(rng, types.len())] as u8;
            let probe = Token {
                kind,
                atom_type: ty,
                bond: 0,
                pointer: 0,
            };
            let bonds = set_bits(st.masks(probe).bonds);
            if bonds.is_empty() {
                break;
            }
            let b = bonds[pick(rng, bonds.len())] as u8;
            let probe = Token {
                kind,
                atom_type: ty,
                bond: b,
                pointer: 0,
            };
            let ptrs = set_bits(st.masks(probe).pointers);
            if ptrs.is_empty() {
                break;
            }
            Token {
                kind,
                atom_type: ty,
                bond: b,
                pointer: ptrs[pick(rng, ptrs.len())] as u8,
            }
        } else if kind == CLOSE_RING {
            let bonds = set_bits(st.masks(taken(kind)).bonds);
            if bonds.is_empty() {
                break;
            }
            let b = bonds[pick(rng, bonds.len())] as u8;
            let probe = Token {
                kind,
                atom_type: 0,
                bond: b,
                pointer: 0,
            };
            let ptrs = set_bits(st.masks(probe).pointers);
            if ptrs.is_empty() {
                break;
            }
            Token {
                kind,
                atom_type: 0,
                bond: b,
                pointer: ptrs[pick(rng, ptrs.len())] as u8,
            }
        } else {
            taken(kind)
        };
        assert!(st.is_legal(token), "the walk only picks legal tokens");
        st.apply(token).unwrap();
        trace.push(token);
        if kind == STOP {
            break;
        }
    }
    trace
}

#[test]
fn replay_matches_trace_state() {
    let _serial = serial();
    // V1 §3.1 shape pins: replay masks run at both (A, R_max, T) =
    // (16, 4, 22) and (32, 8, 42) — synthetic traces from the host grammar
    // (up to 32 atoms and 8 closures at the larger shape) plus the fixture
    // traces — against `TraceState::masks`. Fewer random traces at the
    // larger shape keep the CPU cost reasonable.
    for (a_loop, r_loop, seed_loop, n_rand) in [
        (16usize, 4usize, 20261003u64, 200usize),
        (32usize, 8usize, 20261004u64, 40usize),
    ] {
        let limits = Limits::new(a_loop, r_loop).unwrap();
        let (a, t) = (a_loop, limits.max_steps());
        let device = dev();
        let constants = Ms2Constants::new(&device);
        let f = fixture();
        // Rows: every fixture whole trace and every invalid trace, each with and
        // without the parent budget, plus seeded random legal traces.
        let mut traces: Vec<Vec<Token>> = Vec::new();
        let mut budgets: Vec<Option<Composition>> = Vec::new();
        let mut names: Vec<String> = Vec::new();
        for m in f["molecules"].as_array().unwrap() {
            let Some(wt) = m.get("whole_trace") else {
                continue;
            };
            let trace = trace_of(&wt["trace"]);
            let budget = composition_of(&m["formula"]);
            traces.push(trace.clone());
            budgets.push(None);
            names.push(format!("{}-nobudget", m["name"].as_str().unwrap()));
            traces.push(trace);
            budgets.push(Some(budget));
            names.push(format!("{}-budget", m["name"].as_str().unwrap()));
        }
        for inv in f["invalid_traces"].as_array().unwrap() {
            let trace = trace_of(&inv["trace"]);
            traces.push(trace.clone());
            budgets.push(None);
            names.push(format!("{}-nobudget", inv["name"].as_str().unwrap()));
            traces.push(trace);
            budgets.push(Some(composition_of(&f["molecules"][2]["formula"])));
            names.push(format!("{}-budget", inv["name"].as_str().unwrap()));
        }
        let mut rng = Rng::seeded(seed_loop);
        for i in 0..n_rand {
            let budget = if i % 2 == 0 {
                None
            } else {
                Some(composition_of(&f["molecules"][9]["formula"]))
            };
            traces.push(random_trace(&mut rng, limits, budget));
            budgets.push(budget);
            names.push(format!("random-{i}"));
        }
        if a_loop == 32 {
            // Deterministic large traces at the V1 shape, so the masks are
            // pinned up to the caps rather than wherever the random walk
            // lands: a 32-atom chain (type C H0, single bonds) and a maximal
            // trace grown with the host grammar (first legal token in
            // ADD-then-CLOSE-then-STOP priority, hence legal by
            // construction).
            let mut chain = vec![start_token()];
            let mut cst = TraceState::new(limits, None);
            cst.apply(start_token()).unwrap();
            while cst.atoms() < a_loop {
                let n = cst.atoms();
                // The root carries no bond or pointer (unused fields are 0);
                // later atoms extend the chain with single bonds.
                let (bond, pointer) = if n == 0 {
                    (0, 0)
                } else {
                    (1, n.saturating_sub(1) as u8)
                };
                let tok = Token {
                    kind: ADD_ATOM,
                    atom_type: 1,
                    bond,
                    pointer,
                };
                assert!(cst.is_legal(tok), "the chain stays legal at atom {n}");
                cst.apply(tok).unwrap();
                chain.push(tok);
            }
            let stop = Token {
                kind: STOP,
                atom_type: 0,
                bond: 0,
                pointer: 0,
            };
            assert!(cst.is_legal(stop), "the chain stops");
            chain.push(stop);
            assert_eq!(cst.atoms(), 32, "the chain reaches 32 atoms");
            traces.push(chain);
            budgets.push(None);
            names.push("chain-32-nobudget".to_string());
            let blank = Token {
                kind: 0,
                atom_type: 0,
                bond: 0,
                pointer: 0,
            };
            // Two maximal traces grown with the host grammar (first legal
            // token in kind priority, hence legal by construction): one
            // ADD-first, one CLOSE-first, so the masks are pinned at both
            // the atom and the closure caps. A third CLOSE-first walk over
            // sulfur hubs (type 15, valence 6) reaches the 8-closure cap.
            for (grow_name, kinds, type_first) in [
                ("grown-add-first-nobudget", [ADD_ATOM, CLOSE_RING], 1u8),
                ("grown-close-first-nobudget", [CLOSE_RING, ADD_ATOM], 1u8),
                ("grown-close-sulfur-nobudget", [CLOSE_RING, ADD_ATOM], 15u8),
            ] {
                let mut grown = vec![start_token()];
                let mut gst = TraceState::new(limits, None);
                gst.apply(start_token()).unwrap();
                while grown.len() < t {
                    let mut next = None;
                    if gst.step() == 1 {
                        for id in [type_first].into_iter().chain(1..=17u8) {
                            let tok = Token {
                                kind: ADD_ATOM,
                                atom_type: id,
                                ..blank
                            };
                            if gst.is_legal(tok) {
                                next = Some(tok);
                                break;
                            }
                        }
                    } else {
                        'grow: for kind in kinds {
                            for id in [type_first].into_iter().chain(0..=18u8) {
                                for b in 0..=3u8 {
                                    for p in 0..32u8 {
                                        let tok = Token {
                                            kind,
                                            atom_type: id,
                                            bond: b,
                                            pointer: p,
                                        };
                                        if gst.is_legal(tok) {
                                            next = Some(tok);
                                            break 'grow;
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
                    assert!(gst.is_legal(tok), "the grown trace stays legal");
                    gst.apply(tok).unwrap();
                    grown.push(tok);
                    if tok.kind == STOP {
                        break;
                    }
                }
                println!(
                    "{grow_name}: {} tokens, {} atoms, {} closures",
                    grown.len(),
                    gst.atoms(),
                    grown.iter().filter(|tok| tok.kind == CLOSE_RING).count(),
                );
                if grow_name == "grown-close-sulfur-nobudget" {
                    assert_eq!(gst.atoms(), 32, "the sulfur walk reaches 32 atoms");
                    assert_eq!(
                        grown.iter().filter(|tok| tok.kind == CLOSE_RING).count(),
                        8,
                        "the sulfur walk reaches 8 closures"
                    );
                    assert_eq!(grown.len(), t, "the sulfur walk fills T = 42");
                }
                traces.push(grown);
                budgets.push(None);
                names.push(grow_name.to_string());
            }
        }
        let rows = traces.len();
        let mut tokens = vec![0u32; rows * t * 4];
        let mut meta = vec![0u32; rows * 12];
        for (r, (trace, budget)) in traces.iter().zip(budgets.iter()).enumerate() {
            assert!(
                trace.len() <= t,
                "row {} ({}): length {} fits T = {t}",
                r,
                names[r],
                trace.len()
            );
            for (s, tok) in trace.iter().enumerate() {
                tokens[(r * t + s) * 4] = u32::from(tok.kind);
                tokens[(r * t + s) * 4 + 1] = u32::from(tok.atom_type);
                tokens[(r * t + s) * 4 + 2] = u32::from(tok.bond);
                tokens[(r * t + s) * 4 + 3] = u32::from(tok.pointer);
            }
            meta[r * 12] = trace.len() as u32;
            if let Some(comp) = budget {
                meta[r * 12 + 1] = 1;
                for (e, count) in comp.iter().enumerate() {
                    meta[r * 12 + 2 + e] = u32::from(*count);
                }
            }
        }
        let tokens_t = IdTensor::from_slice(&tokens, vec![rows, t, 4], &device).unwrap();
        let meta_t = IdTensor::from_slice(&meta, vec![rows, 12], &device).unwrap();
        let out = ReplayBuffers::poisoned(rows, t, a, &device).unwrap();
        ms2::grammar_replay(
            &tokens_t,
            &meta_t,
            &constants,
            a_loop as u32,
            r_loop as u32,
            &out,
        )
        .unwrap();
        check_launches(&device).unwrap();
        let replay = out.replay.try_to_vec().unwrap();
        let atoms = out.atoms.try_to_vec().unwrap();
        for (r, ((trace, budget), name)) in traces
            .iter()
            .zip(budgets.iter())
            .zip(names.iter())
            .enumerate()
        {
            let host = host_replay(trace, limits, *budget, a);
            let length = trace.len();
            let illegal = host.first_illegal;
            for s in 0..t {
                let base = (r * t + s) * (4 + a);
                if s >= length || (illegal != u32::MAX && s > illegal as usize) {
                    for w in 0..4 + a {
                        assert_eq!(
                            replay[base + w],
                            0,
                            "row {r} ({name}) step {s}: post-trace word {w} is zero"
                        );
                    }
                    continue;
                }
                let m = &host.masks[s];
                assert_eq!(replay[base], m.kinds, "row {r} ({name}) step {s}: kinds");
                assert_eq!(
                    replay[base + 1],
                    m.atom_types,
                    "row {r} ({name}) step {s}: types"
                );
                assert_eq!(
                    replay[base + 2],
                    m.bonds,
                    "row {r} ({name}) step {s}: bonds"
                );
                assert_eq!(
                    replay[base + 3],
                    m.pointers,
                    "row {r} ({name}) step {s}: pointers"
                );
                for j in 0..a {
                    assert_eq!(
                        replay[base + 4 + j],
                        u32::from(host.resids[s][j]),
                        "row {r} ({name}) step {s}: residual {j}"
                    );
                }
            }
            for j in 0..a {
                assert_eq!(
                    atoms[r * (a + 1) + j],
                    host.add_steps[j],
                    "row {r} ({name}): atom {j} step"
                );
            }
            assert_eq!(
                atoms[r * (a + 1) + a],
                illegal,
                "row {r} ({name}): first illegal"
            );
        }
    }
}

#[test]
fn replay_root_budget_cases() {
    let _serial = serial();
    // The fixture's root-budget cases: the root masks under tight budgets.
    let device = dev();
    let constants = Ms2Constants::new(&device);
    let f = fixture();
    let limits = Limits::new(16, 4).unwrap();
    let (a, t) = (16usize, limits.max_steps());
    let cases = f["root_budget_cases"].as_array().unwrap();
    let rows = cases.len();
    let mut tokens = vec![0u32; rows * t * 4];
    let mut meta = vec![0u32; rows * 12];
    for (r, case) in cases.iter().enumerate() {
        // START then a root candidate; the masks at step 1 carry the budget.
        tokens[(r * t) * 4] = u32::from(START);
        tokens[(r * t + 1) * 4] = u32::from(ADD_ATOM);
        tokens[(r * t + 1) * 4 + 1] = 2;
        meta[r * 12] = 2;
        meta[r * 12 + 1] = 1;
        for (symbol, count) in case["budget"].as_object().unwrap() {
            let e = element_index(symbol).expect("known element");
            meta[r * 12 + 2 + e] = count.as_u64().unwrap() as u32;
        }
    }
    let tokens_t = IdTensor::from_slice(&tokens, vec![rows, t, 4], &device).unwrap();
    let meta_t = IdTensor::from_slice(&meta, vec![rows, 12], &device).unwrap();
    let out = ReplayBuffers::poisoned(rows, t, a, &device).unwrap();
    ms2::grammar_replay(&tokens_t, &meta_t, &constants, 16, 4, &out).unwrap();
    check_launches(&device).unwrap();
    let replay = out.replay.try_to_vec().unwrap();
    for (r, case) in cases.iter().enumerate() {
        let base = (r * t + 1) * (4 + a);
        assert_eq!(
            replay[base],
            case["kinds"].as_u64().unwrap() as u32,
            "row {r}: root kinds match the fixture"
        );
        assert_eq!(
            replay[base + 1],
            case["atom_types"].as_u64().unwrap() as u32,
            "row {r}: root types match the fixture"
        );
    }
}

#[test]
fn replay_edge_cases() {
    let _serial = serial();
    let device = dev();
    let constants = Ms2Constants::new(&device);
    let f = fixture();
    let limits = Limits::new(16, 4).unwrap();
    let (a, t) = (16usize, limits.max_steps());
    let pyrene = f["molecules"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["name"] == "pyrene")
        .unwrap();
    let whole = trace_of(&pyrene["whole_trace"]["trace"]);
    assert_eq!(whole.len(), 22, "pyrene's whole_trace is a 22-token trace");
    // Rows: pyrene (16 atoms: the atom cap, 22 tokens), the same trace with a
    // closure onto the parent, with a repeated closure pointer (illegal), and
    // with a non-zero unused field (illegal).
    let mut close_on_parent = whole.clone();
    // Reroute pyrene's first CLOSE_RING at its own pointer onto the parent of
    // the newest atom instead: still legal (residual valence allows it).
    let close_idx = whole.iter().position(|tok| tok.kind == CLOSE_RING).unwrap();
    close_on_parent[close_idx].pointer = 0;
    let mut repeat_close = whole.clone();
    repeat_close[close_idx + 1] = repeat_close[close_idx];
    let mut bad_unused = whole.clone();
    bad_unused[2].pointer = 7;
    let rows_spec: Vec<(&str, Vec<Token>)> = vec![
        ("pyrene", whole.clone()),
        ("close-on-parent", close_on_parent),
        ("repeated-close", repeat_close),
        ("nonzero-unused", bad_unused),
    ];
    // A closure-cap row: four closures then one more CLOSE (illegal at 4).
    let mut cap627 = vec![start_token()];
    cap627.push(Token {
        kind: ADD_ATOM,
        atom_type: 2,
        bond: 0,
        pointer: 0,
    });
    cap627.push(Token {
        kind: ADD_ATOM,
        atom_type: 2,
        bond: 2,
        pointer: 0,
    });
    cap627.push(Token {
        kind: ADD_ATOM,
        atom_type: 2,
        bond: 2,
        pointer: 0,
    });
    cap627.push(Token {
        kind: ADD_ATOM,
        atom_type: 2,
        bond: 2,
        pointer: 1,
    });
    cap627.push(Token {
        kind: CLOSE_RING,
        atom_type: 0,
        bond: 1,
        pointer: 0,
    });
    cap627.push(Token {
        kind: ADD_ATOM,
        atom_type: 2,
        bond: 2,
        pointer: 2,
    });
    cap627.push(Token {
        kind: CLOSE_RING,
        atom_type: 0,
        bond: 1,
        pointer: 0,
    });
    cap627.push(Token {
        kind: CLOSE_RING,
        atom_type: 0,
        bond: 1,
        pointer: 1,
    });
    cap627.push(Token {
        kind: ADD_ATOM,
        atom_type: 2,
        bond: 2,
        pointer: 3,
    });
    cap627.push(Token {
        kind: CLOSE_RING,
        atom_type: 0,
        bond: 1,
        pointer: 0,
    });
    cap627.push(Token {
        kind: CLOSE_RING,
        atom_type: 0,
        bond: 1,
        pointer: 1,
    });
    cap627.push(Token {
        kind: CLOSE_RING,
        atom_type: 0,
        bond: 1,
        pointer: 2,
    });
    let mut rows_spec = rows_spec;
    rows_spec.push(("closure-cap", cap627));
    let rows = rows_spec.len();
    let mut tokens = vec![0u32; rows * t * 4];
    let mut meta = vec![0u32; rows * 12];
    for (r, (_, trace)) in rows_spec.iter().enumerate() {
        assert!(trace.len() <= t, "edge row {r} fits T");
        for (s, tok) in trace.iter().enumerate() {
            tokens[(r * t + s) * 4] = u32::from(tok.kind);
            tokens[(r * t + s) * 4 + 1] = u32::from(tok.atom_type);
            tokens[(r * t + s) * 4 + 2] = u32::from(tok.bond);
            tokens[(r * t + s) * 4 + 3] = u32::from(tok.pointer);
        }
        meta[r * 12] = trace.len() as u32;
    }
    let tokens_t = IdTensor::from_slice(&tokens, vec![rows, t, 4], &device).unwrap();
    let meta_t = IdTensor::from_slice(&meta, vec![rows, 12], &device).unwrap();
    let out = ReplayBuffers::poisoned(rows, t, a, &device).unwrap();
    ms2::grammar_replay(&tokens_t, &meta_t, &constants, 16, 4, &out).unwrap();
    check_launches(&device).unwrap();
    let replay = out.replay.try_to_vec().unwrap();
    let atoms = out.atoms.try_to_vec().unwrap();
    for (r, (name, trace)) in rows_spec.iter().enumerate() {
        let host = host_replay(trace, limits, None, a);
        assert_eq!(
            atoms[r * (a + 1) + a],
            host.first_illegal,
            "{name}: first illegal step"
        );
        for j in 0..a {
            assert_eq!(
                atoms[r * (a + 1) + j],
                host.add_steps[j],
                "{name}: atom {j} step"
            );
        }
        for s in 0..t {
            let base = (r * t + s) * (4 + a);
            let illegal = host.first_illegal;
            if s >= trace.len() || (illegal != u32::MAX && s > illegal as usize) {
                assert!(
                    replay[base..base + 4 + a].iter().all(|&w| w == 0),
                    "{name} step {s}: zeros after the end"
                );
            } else {
                let m = &host.masks[s];
                assert_eq!(
                    [
                        replay[base],
                        replay[base + 1],
                        replay[base + 2],
                        replay[base + 3]
                    ],
                    [m.kinds, m.atom_types, m.bonds, m.pointers],
                    "{name} step {s}: masks"
                );
            }
        }
    }
    // Spot checks: pyrene is legal and fills all 16 atoms; the repeated
    // closure and the non-zero unused field are illegal.
    assert_eq!(atoms[0 * (a + 1) + a], u32::MAX, "pyrene replays legally");
    assert!(
        atoms[0 * (a + 1)..0 * (a + 1) + a]
            .iter()
            .all(|&s| s != u32::MAX),
        "pyrene adds 16 atoms"
    );
    assert_ne!(
        atoms[2 * (a + 1) + a],
        u32::MAX,
        "the repeated closure is illegal"
    );
    assert_ne!(
        atoms[3 * (a + 1) + a],
        u32::MAX,
        "the non-zero unused field is illegal"
    );
}

#[test]
fn grammar_replay_rejects_bad_limits_and_widths() {
    let _serial = serial();
    // `grammar_replay` validates before any subtraction or launch: `1 <=
    // max_atoms <= 32`, `max_closures <= 32` (the host `TraceState` bounds),
    // the replay width `4 + max_atoms` (at least 4), and the state/atoms
    // widths. Each bad input is `Error::Shape`/`Error::Config`, never an
    // underflow or a launch.
    let device = dev();
    let constants = Ms2Constants::new(&device);
    let rows = 1usize;
    let t = 4usize;
    let tokens = IdTensor::from_slice(
        &vec![1u32, 0, 0, 0, 2, 2, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0],
        vec![rows, t, 4],
        &device,
    )
    .unwrap();
    let meta = IdTensor::from_slice(
        &vec![3u32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        vec![rows, 12],
        &device,
    )
    .unwrap();
    // A = 0: below the host lower bound.
    let out = ReplayBuffers::new(rows, t, 4, &device);
    assert!(ms2::grammar_replay(&tokens, &meta, &constants, 0, 4, &out).is_err());
    // A = 33: above the host upper bound.
    assert!(ms2::grammar_replay(&tokens, &meta, &constants, 33, 4, &out).is_err());
    // max_closures above the host bound.
    assert!(ms2::grammar_replay(&tokens, &meta, &constants, 4, 33, &out).is_err());
    // A width-3 replay buffer: narrower than the 4 mask words, so `width - 4`
    // would underflow without the guard.
    let narrow_replay =
        IdTensor::from_slice(&vec![0u32; rows * t * 3], vec![rows, t, 3], &device).unwrap();
    let narrow = ReplayBuffers {
        state: IdTensor::empty(vec![rows, ms2::replay_state_width(4)], &device),
        replay: narrow_replay,
        atoms: IdTensor::empty(vec![rows, 5], &device),
    };
    assert!(ms2::grammar_replay(&tokens, &meta, &constants, 4, 4, &narrow).is_err());
    check_launches(&device).unwrap();
}

#[test]
fn target_batch_build() {
    let _serial = serial();
    let f = fixture();
    let limits = Limits::V0;
    let t = limits.max_steps();
    let ethanol = &f["molecules"][0];
    let benzene = &f["molecules"][2];
    let eth_trace = trace_of(&ethanol["whole_trace"]["trace"]);
    let ben_trace = trace_of(&benzene["whole_trace"]["trace"]);
    let eth_parent = composition_of(&ethanol["formula"]);
    let ben_parent = composition_of(&benzene["formula"]);
    // One budget covering both traces: element-wise maximum.
    let mut both_parent = eth_parent;
    for (a, b) in both_parent.iter_mut().zip(ben_parent.iter()) {
        *a = (*a).max(*b);
    }
    // A two-target spectrum plus an unlabeled one.
    let lab0 = Labels {
        embeddings: Vec::new(),
        graphs: 2,
        targets_before_cut: 2,
        targets: vec![
            Target {
                trace: eth_trace.clone(),
                weight: 3,
                q: 0.75,
                embeddings: vec![],
                anchors: vec![],
            },
            Target {
                trace: ben_trace.clone(),
                weight: 1,
                q: 0.25,
                embeddings: vec![],
                anchors: vec![],
            },
        ],
        dropped_weight: 0.0,
        cut_is_tied: false,
        explained_peaks: Vec::new(),
        ambiguous_hypotheses: 0,
        canonicalization_failures: 0,
    };
    let batch =
        TargetBatch::build(&[Some(&lab0), None], &[both_parent, ben_parent], 4, limits).unwrap();
    assert_eq!(batch.spectra, 2);
    assert_eq!(batch.slots, 4);
    assert_eq!(batch.max_steps, t);
    // q keeps its frozen normalisation; empty slots have q 0 and length 0.
    assert_close(&batch.q[0..4], &[0.75, 0.25, 0.0, 0.0], 0.0, "q row 0");
    assert_close(&batch.q[4..8], &[0.0, 0.0, 0.0, 0.0], 0.0, "q row 1");
    assert_eq!(&batch.labeled, &[1, 0]);
    assert_eq!(batch.meta[0], eth_trace.len() as u32);
    assert_eq!(batch.meta[12], ben_trace.len() as u32);
    assert_eq!(batch.meta[2 * 12], 0, "empty slot has length 0");
    assert_eq!(batch.meta[4 * 12], 0, "unlabeled spectrum has length 0");
    // Budget is the parent composition.
    for e in 0..10 {
        assert_eq!(
            batch.meta[2 + e],
            u32::from(both_parent[e]),
            "budget element {e}"
        );
    }
    // Hand check on the two-atom ethanol trace: START excluded, STOP
    // included, unused fields 0. Ethanol: START, root, ADD, ADD?, STOP...
    let length = eth_trace.len();
    for i in 0..t {
        let base = (0 * t + i) * 4;
        let pos = i + 1;
        if pos < length {
            let kind = eth_trace[pos].kind;
            assert_eq!(batch.use_mask[base], 1.0, "position {i}: kind used");
            assert_eq!(
                batch.use_mask[base + 1],
                f32::from(kind == ADD_ATOM),
                "position {i}: type"
            );
            let bond_used = (kind == ADD_ATOM && pos > 1) || kind == CLOSE_RING;
            assert_eq!(
                batch.use_mask[base + 2],
                f32::from(bond_used),
                "position {i}: bond"
            );
            assert_eq!(
                batch.use_mask[base + 3],
                f32::from(bond_used),
                "position {i}: pointer"
            );
        } else {
            assert_eq!(
                &batch.use_mask[base..base + 4],
                &[0.0; 4],
                "position {i}: unscored"
            );
        }
    }
    // The STOP position contributes its kind.
    let stop_pos = length - 1;
    let stop_base = (0 * t + (stop_pos - 1)) * 4;
    assert_eq!(eth_trace[stop_pos].kind, STOP);
    assert_eq!(batch.use_mask[stop_base], 1.0, "STOP kind is scored");
    assert_eq!(&batch.use_mask[stop_base + 1..stop_base + 4], &[0.0; 3]);
    // Errors: more targets than slots, an illegal trace, a missing STOP, an
    // over-long trace.
    let many = Labels {
        targets: vec![
            Target {
                trace: eth_trace.clone(),
                weight: 1,
                q: 1.0,
                embeddings: vec![],
                anchors: vec![]
            };
            5
        ],
        graphs: 5,
        targets_before_cut: 5,
        embeddings: vec![],
        dropped_weight: 0.0,
        cut_is_tied: false,
        explained_peaks: vec![],
        ambiguous_hypotheses: 0,
        canonicalization_failures: 0,
    };
    assert!(TargetBatch::build(&[Some(&many)], &[eth_parent], 4, limits).is_err());
    // Contract §7.2 / §9: at most 16 targets per spectrum regardless of
    // `slots`, every q finite and > 0, and the kept q sums to 1 within 1e-5.
    let sixteen = Labels {
        targets: vec![
            Target {
                trace: eth_trace.clone(),
                weight: 1,
                q: 1.0 / 16.0,
                embeddings: vec![],
                anchors: vec![]
            };
            16
        ],
        graphs: 16,
        targets_before_cut: 16,
        embeddings: vec![],
        dropped_weight: 0.0,
        cut_is_tied: false,
        explained_peaks: vec![],
        ambiguous_hypotheses: 0,
        canonicalization_failures: 0,
    };
    assert!(TargetBatch::build(&[Some(&sixteen)], &[eth_parent], 16, limits).is_ok());
    let seventeen = Labels {
        targets: vec![
            Target {
                trace: eth_trace.clone(),
                weight: 1,
                q: 1.0 / 17.0,
                embeddings: vec![],
                anchors: vec![]
            };
            17
        ],
        graphs: 17,
        targets_before_cut: 17,
        embeddings: vec![],
        dropped_weight: 0.0,
        cut_is_tied: false,
        explained_peaks: vec![],
        ambiguous_hypotheses: 0,
        canonicalization_failures: 0,
    };
    let err = match TargetBatch::build(&[Some(&seventeen)], &[eth_parent], 32, limits) {
        Err(e) => e,
        Ok(_) => panic!("17 targets past the 16 cap must fail even with 32 slots"),
    };
    assert!(
        err.to_string().contains("spectrum 0") && err.to_string().contains("17"),
        "the 16-cap error names the spectrum and the count: {err}"
    );
    for bad_q in [f64::NAN, f64::INFINITY, 0.0, -0.25] {
        let mut traces = vec![
            Target {
                trace: eth_trace.clone(),
                weight: 3,
                q: 0.75,
                embeddings: vec![],
                anchors: vec![],
            },
            Target {
                trace: ben_trace.clone(),
                weight: 1,
                q: 0.25,
                embeddings: vec![],
                anchors: vec![],
            },
        ];
        traces[0].q = bad_q;
        let lab = Labels {
            targets: traces,
            graphs: 2,
            targets_before_cut: 2,
            embeddings: vec![],
            dropped_weight: 0.0,
            cut_is_tied: false,
            explained_peaks: vec![],
            ambiguous_hypotheses: 0,
            canonicalization_failures: 0,
        };
        let err = match TargetBatch::build(&[Some(&lab)], &[both_parent], 4, limits) {
            Err(e) => e,
            Ok(_) => panic!("a non-finite or non-positive q must fail"),
        };
        assert!(
            err.to_string().contains("spectrum 0") && err.to_string().contains(&format!("{bad_q}")),
            "the q error names the spectrum and the value: {err}"
        );
    }
    let unnormalised = Labels {
        targets: vec![
            Target {
                trace: eth_trace.clone(),
                weight: 3,
                q: 0.6,
                embeddings: vec![],
                anchors: vec![],
            },
            Target {
                trace: ben_trace.clone(),
                weight: 1,
                q: 0.25,
                embeddings: vec![],
                anchors: vec![],
            },
        ],
        graphs: 2,
        targets_before_cut: 2,
        embeddings: vec![],
        dropped_weight: 0.0,
        cut_is_tied: false,
        explained_peaks: vec![],
        ambiguous_hypotheses: 0,
        canonicalization_failures: 0,
    };
    let err = match TargetBatch::build(&[Some(&unnormalised)], &[both_parent], 4, limits) {
        Err(e) => e,
        Ok(_) => panic!("q summing to 0.85 must fail the 1e-5 normalisation"),
    };
    assert!(
        err.to_string().contains("spectrum 0") && err.to_string().contains("0.85"),
        "the normalisation error names the spectrum and the sum: {err}"
    );
    let mut illegal = eth_trace.clone();
    illegal[1].atom_type = 0;
    let lab_illegal = single_target(illegal);
    assert!(TargetBatch::build(&[Some(&lab_illegal)], &[eth_parent], 4, limits).is_err());
    let mut no_stop = eth_trace.clone();
    no_stop.pop();
    assert!(
        TargetBatch::build(&[Some(&single_target(no_stop))], &[eth_parent], 4, limits).is_err()
    );
    let long: Vec<Token> = eth_trace.iter().cycle().take(t + 1).copied().collect();
    assert!(TargetBatch::build(&[Some(&single_target(long))], &[eth_parent], 4, limits).is_err());
    // Upload: 4 uploads, shapes as documented.
    let device = dev();
    let up = batch.upload::<R, E>(&device).unwrap();
    assert_eq!(up.tokens.shape().dims(), &[2 * 4, t, 4]);
    assert_eq!(up.meta.shape().dims(), &[2 * 4, 12]);
    assert_eq!(up.q.shape().dims(), &[2 * 4]);
    assert_eq!(up.use_mask.shape().dims(), &[2 * 4, t, 4]);
    check_launches(&device).unwrap();
}

/// Encode + replay + teacher on the tiny model, returning everything a test
/// needs without reading the device until the caller does.
struct TinySetup {
    decoder: Ms2Decoder<R, E>,
    encoded: mamba3::models::ms2::encoder::EncoderOutput<R, E>,
    formula_emb: Var<R, E>,
    targets: mamba3::models::ms2::targets_batch::DeviceTargets<R, E>,
    batch: TargetBatch,
    buffers: ReplayBuffers<R>,
    spectra: usize,
    slots: usize,
}

/// Tiny V1-shape model: `(A, R_max) = (32, 8)` with 4 decoder blocks and the
/// decoder inner width set apart (4 heads x 8 channels = 32 against the
/// encoder's 2 x 8 = 16, at the same `d_model`); `d`, attention heads and
/// the peak cap stay tiny/V0 so the CPU cost stays reasonable.
fn tiny_v1_config() -> ModelConfig {
    let mut m = tiny_config();
    m.max_atoms = 32;
    m.max_ring_closures = 8;
    m.decoder_blocks = 4;
    m.decoder.n_heads = 4;
    m.decoder.head_dim = 8;
    m
}

fn tiny_setup_with(
    model: ModelConfig,
    labels: &[Option<&Labels>],
    parents: &[Composition],
    slots: usize,
    spectra_batch: &SpectrumBatch,
    limits: Limits,
    atoms: usize,
) -> TinySetup {
    let device = dev();
    let mut rng = Rng::seeded(7);
    let encoder = Ms2Encoder::init(&model, &device, &mut rng).unwrap();
    let decoder = Ms2Decoder::init(&model, &device, &mut rng).unwrap();
    let spectra = DeviceSpectra::upload(spectra_batch, &device).unwrap();
    let peaks = ms2::PeakBuffers::<R, E>::new(spectra.batch, spectra.n_raw, 16, &device);
    let encoded = encoder.encode(&spectra, &peaks, Control::None).unwrap();
    check_launches(&device).unwrap();
    // Formula embedding: the pooled vector, broadcast per spectrum.
    let formula_emb = encoded.pool.clone();
    let batch = TargetBatch::build(labels, parents, slots, limits).unwrap();
    let t = limits.max_steps();
    let rows = batch.spectra * slots;
    let targets = batch.upload(&device).unwrap();
    let constants = Ms2Constants::new(&device);
    let buffers = ReplayBuffers::poisoned(rows, t, atoms, &device).unwrap();
    ms2::grammar_replay(
        &targets.tokens,
        &targets.meta,
        &constants,
        atoms as u32,
        limits.max_closures() as u32,
        &buffers,
    )
    .unwrap();
    check_launches(&device).unwrap();
    let spectra_n = batch.spectra;
    TinySetup {
        decoder,
        encoded,
        formula_emb,
        targets,
        batch,
        buffers,
        spectra: spectra_n,
        slots,
    }
}

fn tiny_setup(
    labels: &[Option<&Labels>],
    parents: &[Composition],
    slots: usize,
    spectra_batch: &SpectrumBatch,
) -> TinySetup {
    tiny_setup_with(
        tiny_config(),
        labels,
        parents,
        slots,
        spectra_batch,
        Limits::V0,
        16,
    )
}

#[test]
fn teacher_finite_masked_and_causal() {
    let _serial = serial();
    let device = dev();
    let f = fixture();
    let eth = &f["molecules"][0];
    let ben = &f["molecules"][2];
    let eth_parent = composition_of(&eth["formula"]);
    let ben_parent = composition_of(&ben["formula"]);
    let lab0 = single_target(trace_of(&eth["whole_trace"]["trace"]));
    let lab1 = single_target(trace_of(&ben["whole_trace"]["trace"]));
    let spectra_batch = make_batch(&[11, 12], 64, &[9, 12], 200_000_000, 3);
    let setup = tiny_setup(
        &[Some(&lab0), Some(&lab1)],
        &[eth_parent, ben_parent],
        4,
        &spectra_batch,
    );
    let replay = ReplayView {
        replay: &setup.buffers.replay,
        atoms: &setup.buffers.atoms,
    };
    let out = setup
        .decoder
        .teacher(&setup.encoded, &setup.formula_emb, &setup.targets, &replay)
        .unwrap();
    check_launches(&device).unwrap();
    let nll = out.nll.try_to_f32().unwrap();
    assert_eq!(nll.len(), 8);
    for (i, v) in nll.iter().enumerate() {
        assert!(v.is_finite(), "slot {i} nll is finite: {v}");
        assert!(*v >= 0.0, "slot {i} nll is non-negative: {v}");
    }
    // The three empty slots per spectrum have nll exactly 0.
    for &slot in &[1u32, 2, 3, 5, 6, 7] {
        assert_eq!(
            nll[slot as usize].to_bits(),
            0.0f32.to_bits(),
            "empty slot {slot} nll is exactly 0"
        );
    }
    // ... and contribute exactly 0 gradient to every parameter, even with
    // arbitrary (out-of-range) tokens written into one of them.
    let grads_of = |tokens: &[u32]| {
        let rows = setup.batch.spectra * setup.slots;
        let t = Limits::V0.max_steps();
        let tok = IdTensor::from_slice(tokens, vec![rows, t, 4], &device).unwrap();
        let dt = mamba3::models::ms2::targets_batch::DeviceTargets {
            tokens: tok,
            meta: setup.targets.meta.clone(),
            q: setup.targets.q.clone(),
            use_mask: setup.targets.use_mask.clone(),
        };
        let o = setup
            .decoder
            .teacher(&setup.encoded, &setup.formula_emb, &dt, &replay)
            .unwrap();
        let loss = graph_loss(&o, &setup.targets.q, setup.spectra).unwrap();
        loss.backward_retain().unwrap()
    };
    let clean = grads_of(&setup.batch.tokens);
    let mut dirty = setup.batch.tokens.clone();
    for k in 0..(Limits::V0.max_steps() * 4) {
        dirty[(3 * Limits::V0.max_steps() * 4) + k] = 200 + k as u32;
    }
    let dirty_grads = grads_of(&dirty);
    for (name, param) in setup.decoder.named_parameters() {
        let a = clean.get(param.id()).unwrap().to_f32();
        let b = dirty_grads.get(param.id()).unwrap().to_f32();
        assert_eq!(a.len(), b.len(), "{name}: gradient length");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "{name}[{i}]: clean {x} vs dirty {y}"
            );
        }
    }
    // Causality: changing token t + 1 leaves h and every earlier output
    // unchanged. Change the atom type of token 2 of spectrum 0's target.
    let h_before = setup
        .decoder
        .hidden(&setup.encoded, &setup.formula_emb, &setup.targets)
        .unwrap()
        .try_to_f32()
        .unwrap();
    let fields_before = out.field_log_prob.try_to_f32().unwrap();
    let mut changed = setup.batch.tokens.clone();
    let t = Limits::V0.max_steps();
    let orig_ty = changed[(0 * t + 2) * 4 + 1];
    changed[(0 * t + 2) * 4 + 1] = if orig_ty == 3 { 5 } else { 3 };
    let tok = IdTensor::from_slice(&changed, vec![8, t, 4], &device).unwrap();
    let dt = mamba3::models::ms2::targets_batch::DeviceTargets {
        tokens: tok,
        meta: setup.targets.meta.clone(),
        q: setup.targets.q.clone(),
        use_mask: setup.targets.use_mask.clone(),
    };
    let h_after = setup
        .decoder
        .hidden(&setup.encoded, &setup.formula_emb, &dt)
        .unwrap()
        .try_to_f32()
        .unwrap();
    let d = 16usize;
    // Rows of spectrum 0: positions <= 1 (predicting tokens <= 2... token 2
    // changed, so positions 0..=1 predict tokens 1..=2 — position 1 predicts
    // the changed token, but h at position 1 is built from token 1 only).
    for row in 0..8 {
        for i in 0..2 {
            let (a, b) = (
                &h_before[(row * t + i) * d..(row * t + i + 1) * d],
                &h_after[(row * t + i) * d..(row * t + i + 1) * d],
            );
            assert_eq!(a, b, "row {row} position {i}: h unchanged");
        }
    }
    // Positions past the change may move.
    let moved = (0..8).any(|row| {
        h_before[(row * t + 3) * d..(row * t + 4) * d]
            != h_after[(row * t + 3) * d..(row * t + 4) * d]
    });
    assert!(moved, "a later position does move");
    // The conditional heads of positions <= t hold their conditioning fixed:
    // field rows before the changed token are unchanged.
    let out2 = setup
        .decoder
        .teacher(&setup.encoded, &setup.formula_emb, &dt, &replay)
        .unwrap();
    let fields_after = out2.field_log_prob.try_to_f32().unwrap();
    for row in 0..8 {
        assert_eq!(
            &fields_before[row * t * 4..(row * t + 1) * 4],
            &fields_after[row * t * 4..(row * t + 1) * 4],
            "row {row} position 0 fields unchanged"
        );
    }
    check_launches(&device).unwrap();
}

#[test]
fn stepped_parity_with_teacher() {
    let _serial = serial();
    // For a batch of real fixture targets, run start_state + step_logits over
    // the prefix position by position and compare the stepped head outputs
    // with the parallel pass's masked log-probabilities at every scored
    // position, within 1e-4.
    let device = dev();
    let f = fixture();
    let eth = &f["molecules"][0];
    let ben = &f["molecules"][2];
    let eth_parent = composition_of(&eth["formula"]);
    let ben_parent = composition_of(&ben["formula"]);
    let lab0 = single_target(trace_of(&eth["whole_trace"]["trace"]));
    let lab1 = single_target(trace_of(&ben["whole_trace"]["trace"]));
    let spectra_batch = make_batch(&[21, 22], 64, &[9, 12], 200_000_000, 5);
    // V1 §3.1 shape pins: teacher-forced versus stepped logits run at both
    // (A, R_max, T) = (16, 4, 22) with 2 decoder blocks and (32, 8, 42)
    // with 4: the stepped caches advance every carry, compared within 1e-4.
    for (model, limits, a, r_shape) in [
        (tiny_config(), Limits::V0, 16usize, 4u32),
        (tiny_v1_config(), Limits::new(32, 8).unwrap(), 32usize, 8u32),
    ] {
        let setup = tiny_setup_with(
            model,
            &[Some(&lab0), Some(&lab1)],
            &[eth_parent, ben_parent],
            2,
            &spectra_batch,
            limits,
            a,
        );
        let t = limits.max_steps();
        let replay = ReplayView {
            replay: &setup.buffers.replay,
            atoms: &setup.buffers.atoms,
        };
        let out = setup
            .decoder
            .teacher(&setup.encoded, &setup.formula_emb, &setup.targets, &replay)
            .unwrap();
        check_launches(&device).unwrap();
        let teacher_fields = out.field_log_prob.try_to_f32().unwrap();
        let replay_h = setup.buffers.replay.try_to_vec().unwrap();
        // Bond table for the stepped bond correction, read once here.
        let bond_by_type = setup
            .decoder
            .named_parameters()
            .into_iter()
            .find(|(n, _)| n == "bond_by_type")
            .expect("bond_by_type")
            .1
            .value()
            .to_f32();
        // Step each spectrum separately (rows = G, rows_per_spectrum = G).
        for b in 0..2 {
            // The encoded memory is per spectrum: slice it out.
            let mem = setup
                .encoded
                .memory
                .slice(0, b, 1)
                .unwrap()
                .reshape(vec![1, 17, 16])
                .unwrap();
            let msk = mamba3::tensor::ops::movement::slice(&setup.encoded.memory_mask, 0, b, 1)
                .unwrap()
                .reshape(vec![1, 17])
                .unwrap();
            let enc = mamba3::models::ms2::encoder::EncoderOutput {
                x: setup.encoded.x.slice(0, b, 1).unwrap(),
                valid: mamba3::tensor::ops::movement::slice(&setup.encoded.valid, 0, b, 1)
                    .unwrap()
                    .reshape(vec![1, 16])
                    .unwrap(),
                memory: mem,
                memory_mask: msk,
                pool: setup.encoded.pool.slice(0, b, 1).unwrap(),
                context: setup.encoded.context.slice(0, b, 1).unwrap(),
            };
            let mut state = setup.decoder.start_state(&enc, 2, &device).unwrap();
            // Grammar rows after each token, maintained with the same apply
            // helper the sampler uses; the budget is the spectrum's parent.
            let parent = if b == 0 { eth_parent } else { ben_parent };
            let mut meta_host = vec![0u32; 2 * 12];
            for g in 0..2 {
                meta_host[g * 12 + 1] = 1;
                for e in 0..10 {
                    meta_host[g * 12 + 2 + e] = u32::from(parent[e]);
                }
            }
            let meta_t = IdTensor::from_slice(&meta_host, vec![2, 12], &device).unwrap();
            let apply_consts = Ms2Constants::new(&device);
            let mut gstate = ms2::grammar_state_zeros(2, a, &device);
            let mut stepped: Vec<mamba3::models::ms2::decoder::StepHeads<R, E>> = Vec::new();
            for pos in 0..t - 1 {
                let mut tok = vec![0u32; 2 * 4];
                for g in 0..2 {
                    let row = b * 2 + g;
                    for c in 0..4 {
                        tok[g * 4 + c] = setup.batch.tokens[(row * t + pos) * 4 + c];
                    }
                }
                let token_t = IdTensor::from_slice(&tok, vec![2, 4], &device).unwrap();
                ms2::grammar_apply(
                    &token_t,
                    &mut gstate,
                    &meta_t,
                    &apply_consts,
                    a as u32,
                    r_shape,
                )
                .unwrap();
                let mut femb = vec![0.0f32; 2 * 16];
                let full = setup.formula_emb.try_to_f32().unwrap();
                for g in 0..2 {
                    femb[g * 16..(g + 1) * 16].copy_from_slice(&full[b * 16..(b + 1) * 16]);
                }
                let femb_t =
                    Var::constant(Tensor::<R, E>::from_f32(&femb, vec![2, 16], &device).unwrap());
                stepped.push(
                    setup
                        .decoder
                        .step_logits(&enc, &femb_t, &token_t, pos, &gstate, &mut state, 2)
                        .unwrap(),
                );
                check_launches(&device).unwrap();
            }
            // Compare at every scored position of the labeled slot (slot 0).
            let row = b * 2;
            let length = setup.batch.meta[row * 12] as usize;
            for i in 0..t - 1 {
                let pos = i + 1;
                if pos >= length {
                    continue;
                }
                let heads = &stepped[i];
                let kind = heads.kind.try_to_f32().unwrap();
                let atype = heads.atom_type.try_to_f32().unwrap();
                let bond_base = heads.bond_base.try_to_f32().unwrap();
                let pbase = heads.pointer_base.try_to_f32().unwrap();
                let ptype = heads.pointer_by_type.try_to_f32().unwrap();
                let pbond = heads.pointer_by_bond.try_to_f32().unwrap();
                // Target conditioning fields from the batch tokens.
                let tk = setup.batch.tokens[(row * t + pos) * 4];
                let ty = setup.batch.tokens[(row * t + pos) * 4 + 1];
                let bd = setup.batch.tokens[(row * t + pos) * 4 + 2];
                let pt = setup.batch.tokens[(row * t + pos) * 4 + 3];
                let c_id = if tk == 2 {
                    ty as usize
                } else if tk == 3 {
                    18
                } else {
                    0
                };
                let rbase = (row * t + pos) * (4 + a);
                let mask_word = |f: usize| replay_h[rbase + f];
                let host_log_softmax = |logits: &[f32], bits: u32| -> Vec<f32> {
                    let mut lse = f32::NEG_INFINITY;
                    for (j, v) in logits.iter().enumerate() {
                        if bits & (1 << j) != 0 {
                            lse = lse.max(*v);
                        }
                    }
                    let mut sum = 0.0f32;
                    for (j, v) in logits.iter().enumerate() {
                        if bits & (1 << j) != 0 {
                            sum += (v - lse).exp();
                        }
                    }
                    let l = lse + sum.ln();
                    logits.iter().map(|v| v - l).collect()
                };
                // Only used fields carry the teacher's gathered value; unused
                // fields are exactly 0 on both sides by construction.
                let used = [
                    true,
                    tk == u32::from(ADD_ATOM),
                    (tk == u32::from(ADD_ATOM) && pos > 1) || tk == u32::from(CLOSE_RING),
                    (tk == u32::from(ADD_ATOM) && pos > 1) || tk == u32::from(CLOSE_RING),
                ];
                let check_field = |logits: &[f32], field: usize, id: u32, what: &str| {
                    if !used[field] {
                        let got = teacher_fields[(row * t + i) * 4 + field];
                        assert_eq!(
                            got.to_bits(),
                            0.0f32.to_bits(),
                            "spectrum {b} position {i} {what}: unused is 0"
                        );
                        return;
                    }
                    let lp = host_log_softmax(logits, mask_word(field));
                    let got = teacher_fields[(row * t + i) * 4 + field];
                    let want = lp[id as usize];
                    assert!(
                        (got - want).abs() <= 1e-4,
                        "spectrum {b} position {i} {what}: teacher {got} vs stepped {want}"
                    );
                };
                check_field(&kind[0..5], 0, tk, "kind");
                check_field(&atype[0..18], 1, ty, "type");
                let mut bond_logits = bond_base[0..4].to_vec();
                for j in 0..4 {
                    bond_logits[j] += bond_by_type[c_id * 4 + j];
                }
                check_field(&bond_logits, 2, bd, "bond");
                // Pointer: base + by_type[c] + by_bond[b].
                let b_id = if tk == 2 && pos > 1 {
                    bd as usize
                } else if tk == 3 {
                    bd as usize
                } else {
                    0
                };
                let mut ptr_logits = pbase[0..a].to_vec();
                for j in 0..a {
                    ptr_logits[j] += ptype[c_id * a + j] + pbond[b_id * a + j];
                }
                check_field(&ptr_logits, 3, pt, "pointer");
            }
        }
    }
}

/// Permute the spectra of a [`SpectrumBatch`] (peak blocks, metadata and
/// provenance together).
fn permute_spectra(batch: &SpectrumBatch, perm: &[usize]) -> SpectrumBatch {
    let n_raw = batch.n_raw as usize;
    let b = batch.len();
    let mut out = batch.clone();
    for (dst, &src) in perm.iter().enumerate() {
        out.spectrum_id[dst] = batch.spectrum_id[src];
        out.raw_peak_count[dst] = batch.raw_peak_count[src];
        out.peak_count[dst] = batch.peak_count[src];
        out.peak_id[dst * n_raw..(dst + 1) * n_raw]
            .copy_from_slice(&batch.peak_id[src * n_raw..(src + 1) * n_raw]);
        out.mz_udalton[dst * n_raw..(dst + 1) * n_raw]
            .copy_from_slice(&batch.mz_udalton[src * n_raw..(src + 1) * n_raw]);
        out.intensity[dst * n_raw..(dst + 1) * n_raw]
            .copy_from_slice(&batch.intensity[src * n_raw..(src + 1) * n_raw]);
        out.mz_uncertainty_udalton[dst] = batch.mz_uncertainty_udalton[src];
        out.precursor_mz_udalton[dst] = batch.precursor_mz_udalton[src];
        out.precursor_uncertainty_udalton[dst] = batch.precursor_uncertainty_udalton[src];
        out.adduct[dst] = batch.adduct[src];
        out.polarity[dst] = batch.polarity[src];
        out.collision_energy_ev[dst] = batch.collision_energy_ev[src];
        out.collision_energy_known[dst] = batch.collision_energy_known[src];
        out.energy_count[dst] = batch.energy_count[src];
        out.fragment_tolerance_ppm_tenths[dst] = batch.fragment_tolerance_ppm_tenths[src];
        out.precursor_tolerance_ppm_tenths[dst] = batch.precursor_tolerance_ppm_tenths[src];
        out.instrument_class[dst] = batch.instrument_class[src];
    }
    let _ = b;
    out
}

#[test]
fn batch_independence_and_graph_loss() {
    let _serial = serial();
    // A spectrum alone versus inside a batch of 3, and a permuted batch, give
    // the same nll (within 1e-5). Non-uniform q plus an unlabeled spectrum:
    // graph_loss equals (1/B) sum q nll recomputed on the host, with B
    // counting the unlabeled spectrum.
    let device = dev();
    let f = fixture();
    let names = ["ethanol", "benzene", "naphthalene"];
    let mols: Vec<serde_json::Value> = names
        .iter()
        .map(|n| {
            f["molecules"]
                .as_array()
                .unwrap()
                .iter()
                .find(|m| m["name"] == *n)
                .unwrap()
                .clone()
        })
        .collect();
    let parents: Vec<Composition> = mols.iter().map(|m| composition_of(&m["formula"])).collect();
    let labs: Vec<Labels> = mols
        .iter()
        .map(|m| single_target(trace_of(&m["whole_trace"]["trace"])))
        .collect();
    let spectra_batch = make_batch(&[31, 32, 33], 64, &[9, 12, 10], 200_000_000, 11);
    let refs: Vec<Option<&Labels>> = labs.iter().map(Some).collect();
    let full = tiny_setup(&refs, &parents, 2, &spectra_batch);
    let replay_full = ReplayView {
        replay: &full.buffers.replay,
        atoms: &full.buffers.atoms,
    };
    let out_full = full
        .decoder
        .teacher(
            &full.encoded,
            &full.formula_emb,
            &full.targets,
            &replay_full,
        )
        .unwrap();
    check_launches(&device).unwrap();
    let nll_full = out_full.nll.try_to_f32().unwrap();
    // Alone: spectrum 0 with identical peaks (same seed, first draws).
    let alone_batch = make_batch(&[31], 64, &[9], 200_000_000, 11);
    let alone = tiny_setup(&[Some(&labs[0])], &parents[0..1], 2, &alone_batch);
    let replay_alone = ReplayView {
        replay: &alone.buffers.replay,
        atoms: &alone.buffers.atoms,
    };
    let out_alone = alone
        .decoder
        .teacher(
            &alone.encoded,
            &alone.formula_emb,
            &alone.targets,
            &replay_alone,
        )
        .unwrap();
    check_launches(&device).unwrap();
    let nll_alone = out_alone.nll.try_to_f32().unwrap();
    assert_close(&nll_alone, &nll_full[0..2], 1e-5, "alone vs in-batch");
    // Permuted batch: same peaks per spectrum (host permutation), same seed
    // (identical weights), reversed order.
    let perm_spectra = permute_spectra(&spectra_batch, &[2, 1, 0]);
    let perm_parents = vec![parents[2], parents[1], parents[0]];
    let perm_refs: Vec<Option<&Labels>> = vec![Some(&labs[2]), Some(&labs[1]), Some(&labs[0])];
    let perm = tiny_setup(&perm_refs, &perm_parents, 2, &perm_spectra);
    let replay_perm = ReplayView {
        replay: &perm.buffers.replay,
        atoms: &perm.buffers.atoms,
    };
    let out_perm = perm
        .decoder
        .teacher(
            &perm.encoded,
            &perm.formula_emb,
            &perm.targets,
            &replay_perm,
        )
        .unwrap();
    check_launches(&device).unwrap();
    let nll_perm = out_perm.nll.try_to_f32().unwrap();
    assert_close(
        &nll_perm[0..2],
        &nll_full[4..6],
        1e-5,
        "permuted spectrum 2",
    );
    assert_close(
        &nll_perm[2..4],
        &nll_full[2..4],
        1e-5,
        "permuted spectrum 1",
    );
    assert_close(
        &nll_perm[4..6],
        &nll_full[0..2],
        1e-5,
        "permuted spectrum 0",
    );
    // graph_loss with non-uniform q and an unlabeled spectrum.
    let refs_u: Vec<Option<&Labels>> = vec![Some(&labs[0]), None, Some(&labs[2])];
    let batch_u = make_batch(&[41, 42, 43], 64, &[9, 7, 10], 200_000_000, 13);
    let setup_u = tiny_setup(&refs_u, &parents, 2, &batch_u);
    let mut q = setup_u.targets.q.try_to_f32().unwrap();
    assert_eq!(q.len(), 6);
    q[0] = 0.3;
    let q_t = Tensor::<R, E>::from_f32(&q, vec![6], &device).unwrap();
    let replay_u = ReplayView {
        replay: &setup_u.buffers.replay,
        atoms: &setup_u.buffers.atoms,
    };
    let out_u = setup_u
        .decoder
        .teacher(
            &setup_u.encoded,
            &setup_u.formula_emb,
            &setup_u.targets,
            &replay_u,
        )
        .unwrap();
    check_launches(&device).unwrap();
    let loss = graph_loss(&out_u, &q_t, 3).unwrap();
    check_launches(&device).unwrap();
    let nll_u = out_u.nll.try_to_f32().unwrap();
    let mut want = 0.0f32;
    for (qq, nn) in q.iter().zip(nll_u.iter()) {
        want += qq * nn;
    }
    want /= 3.0;
    let got = loss.try_to_f32().unwrap()[0];
    assert!(
        (got - want).abs() <= 1e-5 * want.abs().max(1.0),
        "graph_loss (1/B) sum q nll: got {got} want {want}"
    );
    // The unlabeled spectrum contributes nothing but counts in B.
    assert_eq!(setup_u.batch.labeled, &[1, 0, 1]);
    assert_close(&nll_u[2..4], &[0.0, 0.0], 0.0, "unlabeled slots are 0");
}

#[test]
fn field_distributions_normalise_over_legal_sets() {
    let _serial = serial();
    // For one target, the exponentiated log-softmax rows sum to 1 over each
    // field's legal set, within 1e-5. Every row is a distribution:
    // unscored positions and position `T - 1` carry the index-0-only
    // distribution (log-probability 0 at index 0, the masked value
    // elsewhere).
    let device = dev();
    let f = fixture();
    let ben = &f["molecules"][2];
    let ben_parent = composition_of(&ben["formula"]);
    let lab = single_target(trace_of(&ben["whole_trace"]["trace"]));
    let spectra_batch = make_batch(&[51], 64, &[12], 200_000_000, 17);
    let setup = tiny_setup(&[Some(&lab)], &[ben_parent], 2, &spectra_batch);
    let t = Limits::V0.max_steps();
    let a = 16usize;
    let replay = ReplayView {
        replay: &setup.buffers.replay,
        atoms: &setup.buffers.atoms,
    };
    let dists = setup
        .decoder
        .field_distributions(&setup.encoded, &setup.formula_emb, &setup.targets, &replay)
        .unwrap();
    check_launches(&device).unwrap();
    let rows = [
        &dists.kind_log_prob,
        &dists.type_log_prob,
        &dists.bond_log_prob,
        &dists.pointer_log_prob,
    ];
    let widths = [5usize, 18, 4, a];
    let host = rows
        .iter()
        .map(|v| v.try_to_f32().unwrap())
        .collect::<Vec<_>>();
    let replay_h = setup.buffers.replay.try_to_vec().unwrap();
    let length = setup.batch.meta[0] as usize;
    for i in 0..t {
        let scored = i + 1 < length && i + 1 < t;
        if scored {
            let pos = i + 1;
            let tk = setup.batch.tokens[pos * 4];
            let used = [
                true,
                tk == u32::from(ADD_ATOM),
                (tk == u32::from(ADD_ATOM) && pos > 1) || tk == u32::from(CLOSE_RING),
                (tk == u32::from(ADD_ATOM) && pos > 1) || tk == u32::from(CLOSE_RING),
            ];
            for (fidx, w) in widths.iter().enumerate() {
                if !used[fidx] {
                    // An unused field at a scored position is index-0-only.
                    assert_eq!(
                        host[fidx][i * w],
                        0.0,
                        "unused field {fidx} position {i}: index 0 is 0"
                    );
                    let mut total = 0.0f32;
                    for j in 0..*w {
                        total += host[fidx][i * w + j].exp();
                    }
                    assert!(
                        (total - 1.0).abs() <= 1e-5,
                        "unused field {fidx} position {i}: mass {total}"
                    );
                    continue;
                }
                let bits = replay_h[pos * (4 + a) + fidx];
                assert_ne!(
                    bits, 0,
                    "used field {fidx} at position {i} has a non-empty legal set"
                );
                let mut total = 0.0f32;
                for j in 0..*w {
                    if bits & (1 << j) != 0 {
                        total += host[fidx][(i * w) + j].exp();
                    }
                }
                assert!(
                    (total - 1.0).abs() <= 1e-5,
                    "field {fidx} position {i}: legal mass {total}"
                );
            }
        } else {
            // Position `T - 1` and unscored positions: index-0-only.
            for (fidx, w) in widths.iter().enumerate() {
                assert_eq!(
                    host[fidx][i * w],
                    0.0,
                    "unscored field {fidx} position {i}: index 0 is 0"
                );
                let mut total = 0.0f32;
                for j in 0..*w {
                    total += host[fidx][i * w + j].exp();
                }
                assert!(
                    (total - 1.0).abs() <= 1e-5,
                    "unscored field {fidx} position {i}: mass {total}"
                );
            }
        }
    }
}

#[test]
fn graph_loss_gradient_matches_finite_differences() {
    let _serial = serial();
    // Central finite differences of graph_loss with respect to three entries
    // of each decoder weight family and of the encoder's peak_in weight.
    // The benzene trace exercises every family with real (non-singleton)
    // choices; the ethanol trace cannot: all of its bond and pointer legal
    // sets are singletons, so those log-probabilities are identically 0 and
    // their gradients are exactly 0.
    let device = dev();
    let f = fixture();
    let ben = &f["molecules"][2];
    let ben_parent = composition_of(&ben["formula"]);
    let lab = single_target(trace_of(&ben["whole_trace"]["trace"]));
    // Small limits keep the pass cheap: T = 12, A = 8 (benzene has 6 atoms
    // and 1 ring closure).
    let (a, rmax) = (8usize, 2usize);
    let limits = Limits::new(a, rmax).unwrap();
    let t = limits.max_steps();
    assert_eq!((t, a), (12, 8));
    let mut model = tiny_config();
    model.max_atoms = a as u32;
    let spectra_batch = make_batch(&[61, 62], 64, &[9, 10], 200_000_000, 19);
    let parents = vec![ben_parent, ben_parent];
    let refs: Vec<Option<&Labels>> = vec![Some(&lab), Some(&lab)];
    let batch = TargetBatch::build(&refs, &parents, 2, limits).unwrap();
    let rows = batch.spectra * 2;
    let mut rng = Rng::seeded(23);
    let encoder = Ms2Encoder::init(&model, &device, &mut rng).unwrap();
    let decoder = Ms2Decoder::init(&model, &device, &mut rng).unwrap();
    let spectra = DeviceSpectra::upload(&spectra_batch, &device).unwrap();
    let peaks = ms2::PeakBuffers::<R, E>::new(spectra.batch, spectra.n_raw, 16, &device);
    let encoded = encoder.encode(&spectra, &peaks, Control::None).unwrap();
    check_launches(&device).unwrap();
    let targets = batch.upload(&device).unwrap();
    let constants = Ms2Constants::new(&device);
    let buffers = ReplayBuffers::poisoned(rows, t, a, &device).unwrap();
    ms2::grammar_replay(
        &targets.tokens,
        &targets.meta,
        &constants,
        a as u32,
        rmax as u32,
        &buffers,
    )
    .unwrap();
    check_launches(&device).unwrap();
    let replay = ReplayView {
        replay: &buffers.replay,
        atoms: &buffers.atoms,
    };
    let loss_of = || {
        // Re-encode every time: an encoder perturbation only reaches the
        // loss through a fresh forward pass.
        let enc = encoder.encode(&spectra, &peaks, Control::None).unwrap();
        let out = decoder.teacher(&enc, &enc.pool, &targets, &replay).unwrap();
        graph_loss(&out, &targets.q, batch.spectra)
            .unwrap()
            .try_to_f32()
            .unwrap()[0]
    };
    let out = decoder
        .teacher(&encoded, &encoded.pool, &targets, &replay)
        .unwrap();
    check_launches(&device).unwrap();
    let loss = graph_loss(&out, &targets.q, batch.spectra).unwrap();
    assert!(loss.try_to_f32().unwrap()[0].is_finite());
    let grads = loss.backward_retain().unwrap();
    // Entries per family come from rows the benzene trace actually
    // exercises: fixed indices like 0/1/mid would land on rows the trace
    // never uses (e.g. `e_ptr_type` rows of absent atom types), where
    // numerical and analytic gradients are both 0 and the check is vacuous.
    // The benzene trace holds START, root and non-root ADD_ATOMs with real
    // bond and pointer choices, CLOSE_RINGs and STOP, so every family is
    // exercised through those tokens (CLOSE-row 18 of `bond_by_type` and
    // `e_ptr_type` is selected by the ring closures).
    let trace = lab.targets[0].trace.clone();
    let length = trace.len();
    let mut used_kinds: Vec<usize> = trace.iter().map(|t| t.kind as usize).collect();
    used_kinds.sort_unstable();
    used_kinds.dedup();
    let mut used_types: Vec<usize> = trace
        .iter()
        .filter(|t| t.kind == ADD_ATOM)
        .map(|t| t.atom_type as usize)
        .collect();
    used_types.sort_unstable();
    used_types.dedup();
    let mut used_bonds: Vec<usize> = trace
        .iter()
        .enumerate()
        .filter(|(i, t)| (t.kind == ADD_ATOM && *i > 1) || t.kind == CLOSE_RING)
        .map(|(_, t)| t.bond as usize)
        .collect();
    used_bonds.sort_unstable();
    used_bonds.dedup();
    let mut used_ptrs: Vec<usize> = trace
        .iter()
        .enumerate()
        .filter(|(i, t)| (t.kind == ADD_ATOM && *i > 1) || t.kind == CLOSE_RING)
        .map(|(_, t)| t.pointer as usize)
        .collect();
    used_ptrs.sort_unstable();
    used_ptrs.dedup();
    let used_steps: Vec<usize> = (0..length.saturating_sub(1)).collect();
    let mut used_c: Vec<usize> = used_types.clone();
    if trace.iter().any(|t| t.kind == CLOSE_RING) {
        used_c.push(18);
    }
    used_c.sort_unstable();
    used_c.dedup();
    // Residual valences the pointer keys actually see: the replay rows at
    // scored positions, clamped like the kernel clamps them.
    let replay_h = buffers.replay.try_to_vec().unwrap();
    let mut used_resid: Vec<usize> = Vec::new();
    for i in 0..t.saturating_sub(1) {
        let pos = i + 1;
        if pos >= length {
            continue;
        }
        for j in 0..a {
            let v = replay_h[i * (4 + a) + 4 + j] as usize;
            used_resid.push(v.min(7));
        }
    }
    used_resid.sort_unstable();
    used_resid.dedup();
    let mut names: Vec<(String, mamba3::nn::param::Param<R, E>)> = decoder.named_parameters();
    let peak_in = encoder
        .named_parameters()
        .into_iter()
        .find(|(n, _)| n == "peak_in.weight")
        .expect("peak_in.weight")
        .1;
    names.push(("encoder.peak_in.weight".to_string(), peak_in));
    let mut vacuous: Vec<String> = Vec::new();
    // All exercised flat indices of a `[rows, cols]` table for these rows.
    let table_all = |rows: &[usize], cols: usize| -> Vec<usize> {
        rows.iter()
            .flat_map(|&r| (0..cols).map(move |c| r * cols + c))
            .collect()
    };
    // Top-3 by |analytic gradient| among the exercised candidates: the
    // trace guarantees these entries carry gradient, and the ranking picks
    // the most sensitive ones so the check is not vacuous.
    let top3 = |candidates: Vec<usize>, analytic: &[f32]| -> Vec<usize> {
        let mut order = candidates;
        order.sort_by(|&a, &b| {
            analytic[b]
                .abs()
                .partial_cmp(&analytic[a].abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        order.into_iter().take(3).collect()
    };
    for (name, param) in &names {
        let shape = param.shape().dims().to_vec();
        let base = param.value().to_f32();
        let analytic = grads.get(param.id()).unwrap().to_f32();
        assert_eq!(analytic.len(), base.len(), "{name}: gradient length");
        let d1 = shape.get(1).copied().unwrap_or(0);
        let d0 = shape.first().copied().unwrap_or(0);
        let all: Vec<usize> = if name.contains("kind_emb") {
            table_all(&used_kinds, d1)
        } else if name.contains("type_emb") {
            table_all(&used_types, shape[1])
        } else if name.contains("bond_emb") {
            let mut rows = vec![0];
            rows.extend(used_bonds.iter().copied());
            rows.sort_unstable();
            rows.dedup();
            table_all(&rows, d1)
        } else if name.contains("ptr_emb") {
            let mut rows = vec![0];
            rows.extend(used_ptrs.iter().copied());
            rows.sort_unstable();
            rows.dedup();
            table_all(&rows, d1)
        } else if name.contains("step_emb") {
            table_all(&used_steps, d1)
        } else if name.contains("kind_head") {
            if shape.len() == 1 {
                used_kinds.clone()
            } else {
                let cols = used_kinds.clone();
                (0..d0)
                    .flat_map(|r| cols.iter().map(move |&c| r * d1 + c))
                    .collect()
            }
        } else if name.contains("type_head") {
            if shape.len() == 1 {
                used_types.clone()
            } else {
                let cols = used_types.clone();
                (0..d0)
                    .flat_map(|r| cols.iter().map(move |&c| r * d1 + c))
                    .collect()
            }
        } else if name.contains("bond_head") {
            if shape.len() == 1 {
                used_bonds.clone()
            } else {
                let cols = used_bonds.clone();
                (0..d0)
                    .flat_map(|r| cols.iter().map(move |&c| r * d1 + c))
                    .collect()
            }
        } else if name.contains("bond_by_type") {
            used_c
                .iter()
                .flat_map(|&r| (0..d1).map(move |c| r * d1 + c))
                .collect()
        } else if name.contains("e_ptr_type") {
            table_all(&used_c, d1)
        } else if name.contains("e_ptr_bond") {
            table_all(&used_bonds, d1)
        } else if name.contains("e_residual") {
            table_all(&used_resid, d1)
        } else {
            // Shared mixer, norm and attention weights act on every
            // position, so every entry is exercised in principle; rank by
            // |analytic gradient| to avoid near-flat directions of this
            // tiny trace. A broken backward would still have to match the
            // independent finite difference below.
            (0..base.len()).collect()
        };
        let idxs = top3(
            all.into_iter().filter(|&i| i < base.len()).collect(),
            &analytic,
        );
        assert!(
            !idxs.is_empty(),
            "{name}: no exercised entry to sample (trace length {length})"
        );
        // Families whose single-entry effect stays below 1e-4 on this trace
        // are named here and skip only the non-vacuity assertion; their
        // analytic/numeric agreement is still checked above. All are SSM
        // state-dynamics params of the mixer: one entry's 1e-2 shift barely
        // moves the 12-step loss (`dt_bias` sits exactly at the boundary and
        // flaps run to run, the rest measured at most ~1e-4).
        // `mixer.a_log`, `mixer.bc_norm.weight`, `mixer.dt_bias`, `mixer.d`,
        // `mixer.b_bias`, `mixer.c_bias`.
        let mut exercised = name.contains("mixer.a_log")
            || name.contains("mixer.bc_norm.weight")
            || name.contains("mixer.dt_bias")
            || name.contains("mixer.d")
            || name.contains("mixer.b_bias")
            || name.contains("mixer.c_bias");
        for &idx in idxs.iter().take(3) {
            let mut up = base.clone();
            up[idx] += 1e-2;
            param.set(Tensor::<R, E>::from_f32(&up, shape.clone(), &device).unwrap());
            let fu = loss_of();
            let mut down = base.clone();
            down[idx] -= 1e-2;
            param.set(Tensor::<R, E>::from_f32(&down, shape.clone(), &device).unwrap());
            let fd = loss_of();
            param.set(Tensor::<R, E>::from_f32(&base, shape.clone(), &device).unwrap());
            let numeric = (fu - fd) / 2e-2;
            let analytic_v = analytic[idx];
            if numeric.abs() > 1e-4 {
                exercised = true;
            }
            assert!(
                (analytic_v - numeric).abs() <= 2e-2 * numeric.abs() + 1e-3,
                "{name}[{idx}]: analytic={analytic_v} numeric={numeric}"
            );
        }
        if !exercised {
            vacuous.push(name.clone());
        }
    }
    assert!(
        vacuous.is_empty(),
        "families with no sampled entry above 1e-4: {vacuous:?}"
    );
    check_launches(&device).unwrap();
}

/// Net hydrogen shift of contract §4.3, `m_H - m_e`.
const H_NET: u32 = 1_007_825 - 549;

#[test]
fn overfit_smoke() {
    let _serial = serial();
    // A smoke test only: 4 fixture spectra, the full model (encoder +
    // formula head + decoder), AdamW lr 3e-3, 150 steps. graph_loss falls
    // below 25% of its first value; the loss is read only every 50 steps.
    let device = dev();
    let f = fixture();
    let names = ["ethanol", "acetic acid", "cyclopropane", "acetonitrile"];
    let mols: Vec<serde_json::Value> = names
        .iter()
        .map(|n| {
            f["molecules"]
                .as_array()
                .unwrap()
                .iter()
                .find(|m| m["name"] == *n)
                .unwrap()
                .clone()
        })
        .collect();
    let parents: Vec<Composition> = mols.iter().map(|m| composition_of(&m["formula"])).collect();
    let labs: Vec<Labels> = mols
        .iter()
        .map(|m| single_target(trace_of(&m["whole_trace"]["trace"])))
        .collect();
    // Precursors from the true parent masses, so every gold formula joins.
    let n_raw = 64usize;
    let b = 4usize;
    let mut peak_id = vec![u32::MAX; b * n_raw];
    let mut mz = vec![0u32; b * n_raw];
    let mut intensity = vec![0.0f32; b * n_raw];
    let mut peak_count = vec![0u32; b];
    let mut raw_peak_count = vec![0u32; b];
    let mut precursor = vec![0u32; b];
    let mut rng = Rng::seeded(31);
    for (bi, parent) in parents.iter().enumerate() {
        let mass = composition_mass(parent).unwrap();
        precursor[bi] = mass + H_NET;
        let n = 20usize;
        peak_count[bi] = n as u32;
        raw_peak_count[bi] = n as u32;
        for i in 0..n {
            peak_id[bi * n_raw + i] = i as u32;
            let f = rng.uniform_vec(1, 60_000_000.0, (precursor[bi] - 5_000_000) as f32)[0] as u32;
            mz[bi * n_raw + i] = f.max(50_000_001);
            intensity[bi * n_raw + i] = 0.5 + 2.0 * rng.uniform_vec(1, 0.0, 1.0)[0];
        }
    }
    let spectra_batch = SpectrumBatch {
        schema_version: SPECTRUM_SCHEMA_VERSION,
        n_raw: n_raw as u32,
        spectrum_id: vec![101, 102, 103, 104],
        raw_peak_count,
        peak_count,
        peak_id,
        mz_udalton: mz,
        intensity,
        intensity_scale: 0,
        mz_uncertainty_udalton: vec![50; b],
        precursor_mz_udalton: precursor.clone(),
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
    // Formula table from the four parents (plus two decoys for company).
    let mut comps = parents.clone();
    comps.push([6, 6, 0, 0, 0, 0, 0, 0, 0, 0]);
    comps.push([2, 6, 0, 1, 0, 0, 0, 0, 0, 0]);
    let table = FormulaTable::from_compositions(comps.into_iter()).unwrap();
    let uploaded = DeviceFormulaTable::<R, E>::upload(&table, &device).unwrap();
    let model = tiny_config();
    let mut rng = Rng::seeded(41);
    let encoder = Ms2Encoder::init(&model, &device, &mut rng).unwrap();
    let formula_head = FormulaHead::<R, E>::init(&model, &device, &mut rng).unwrap();
    let decoder = Ms2Decoder::init(&model, &device, &mut rng).unwrap();
    let spectra = DeviceSpectra::upload(&spectra_batch, &device).unwrap();
    let peaks = ms2::PeakBuffers::<R, E>::new(b, n_raw, 16, &device);
    let m_window = 32usize;
    let mut buffers = ms2::FormulaBuffers::<R, E>::new(b, m_window, 4, &device);
    let mut search = vec![0u32; table.len() * 2];
    for row in 0..table.len() {
        search[row * 2] = table.mass(row);
        search[row * 2 + 1] =
            (mamba3::models::ms2::chem::composition_error_nda(table.composition(row))
                .div_ceil(1000)) as u32;
    }
    let search_t = IdTensor::from_slice(&search, vec![table.len(), 2], &device).unwrap();
    let refs: Vec<Option<&Labels>> = labs.iter().map(Some).collect();
    let targets_batch = TargetBatch::build(&refs, &parents, 4, Limits::V0).unwrap();
    let targets = targets_batch.upload(&device).unwrap();
    let constants = Ms2Constants::new(&device);
    let t = Limits::V0.max_steps();
    let rows = b * 4;
    let replay_buffers = ReplayBuffers::poisoned(rows, t, 16, &device).unwrap();
    ms2::grammar_replay(
        &targets.tokens,
        &targets.meta,
        &constants,
        16,
        4,
        &replay_buffers,
    )
    .unwrap();
    check_launches(&device).unwrap();
    // Gold slots from the host reference (no device read involved).
    let queries: Vec<WindowQuery> = precursor
        .iter()
        .map(|&p| WindowQuery {
            precursor_mz: p,
            adduct: 1,
            ppm_tenths: 200,
            precursor_uncertainty: 50,
            rows_visited_max: u32::MAX,
            rows_scored_max: 4096,
        })
        .collect();
    let mut gold_rows = vec![u32::MAX; b];
    for (bi, parent) in parents.iter().enumerate() {
        for row in 0..table.len() {
            if table.composition(row) == parent {
                gold_rows[bi] = row as u32;
                break;
            }
        }
        assert_ne!(gold_rows[bi], u32::MAX, "parent {bi} is in the table");
    }
    let gold_slots = gold_slots_host(&table, &queries, &gold_rows, m_window);
    assert!(
        gold_slots.iter().all(|&s| s != u32::MAX),
        "every gold is scored: {gold_slots:?}"
    );
    let gold_t = IdTensor::from_slice(&gold_slots, vec![b], &device).unwrap();
    let mut opt = AdamWConfig::builder()
        .learning_rate(3e-3)
        .build()
        .init::<R, E>();
    let mut params = encoder.named_parameters();
    params.extend(formula_head.named_parameters());
    params.extend(decoder.named_parameters());
    let only_values: Vec<mamba3::nn::param::Param<R, E>> =
        params.iter().map(|(_, p)| p.clone()).collect();
    let replay = ReplayView {
        replay: &replay_buffers.replay,
        atoms: &replay_buffers.atoms,
    };
    let mut curve = Vec::new();
    for step in 0..=150 {
        let encoded = encoder.encode(&spectra, &peaks, Control::None).unwrap();
        ms2::formula_window(
            &search_t,
            &spectra.meta,
            table.max_error(),
            u32::MAX,
            4096,
            &buffers,
        )
        .unwrap();
        ms2::formula_gather(
            &buffers.window,
            &uploaded.table,
            &uploaded.counts,
            &mut buffers.cand,
        )
        .unwrap();
        ms2::count_features(
            &buffers.cand.reshape(vec![b * m_window, 13]).unwrap(),
            &uploaded.log_table,
            &mut buffers.cand_feat.reshape(vec![b * m_window, 10]).unwrap(),
            13,
        )
        .unwrap();
        let scored = formula_head.score(&buffers, &encoded.pool).unwrap();
        let formula_loss = formula_head.loss(&scored, &gold_t).unwrap();
        // Condition on the gold row's embedding (oracle formula).
        let gold_ids = IdTensor::from_slice(&gold_slots, vec![b], &device).unwrap();
        let e_gold = Var::gather_tokens(&scored.embedding, &gold_ids, 1)
            .unwrap()
            .reshape(vec![b, 16])
            .unwrap();
        let tout = decoder
            .teacher(&encoded, &e_gold, &targets, &replay)
            .unwrap();
        let gloss = graph_loss(&tout, &targets.q, b).unwrap();
        let total = gloss.add(&formula_loss.mul_scalar(0.2)).unwrap();
        if step % 50 == 0 {
            let v = total.try_to_f32().unwrap()[0];
            curve.push((step, v));
            println!("overfit step {step}: loss {v}");
        }
        let grads = total.backward_retain().unwrap();
        opt.step(&only_values, &grads).unwrap();
    }
    check_launches(&device).unwrap();
    assert_eq!(curve.len(), 4);
    println!("overfit curve: {curve:?}");
    assert!(
        curve[3].1 < 0.25 * curve[0].1,
        "loss falls below 25%: {curve:?}"
    );
}

#[test]
fn atom_memory_update_matches_twin_on_poison() {
    let _serial = serial();
    // The device atom-memory write against its host twin on poisoned buffers:
    // ADD_ATOM rows copy the previous output into `count - 1` (the count
    // read from the post-token grammar row), other kinds leave the memory
    // untouched, and every residual id is `min(residual, 7)`. Row 3 applies an
    // ADD onto a stopped row (illegal: the row keeps its count and gains the
    // error marker, and the kernel still copies into `count - 1`).
    use mamba3::models::ms2::grammar::{Limits, TraceState};
    use mamba3::models::ms2::twin as host;
    let device = dev();
    let a = 4usize;
    let d = 8usize;
    let rows = 5usize;
    let s = ms2::replay_state_width(a);
    let atable = host::atom_table_rows();
    assert_eq!(atable.len(), 54);
    fn tok(kind: u32, ty: u32, b: u32, p: u32) -> [u32; 4] {
        [kind, ty, b, p]
    }
    // Budget-free legal prefixes (checked below with `TraceState`).
    let seqs: Vec<Vec<[u32; 4]>> = vec![
        vec![tok(1, 0, 0, 0), tok(2, 2, 0, 0)],
        vec![tok(1, 0, 0, 0), tok(2, 3, 0, 0), tok(2, 2, 1, 0)],
        vec![
            tok(1, 0, 0, 0),
            tok(2, 2, 0, 0),
            tok(2, 3, 1, 0),
            tok(2, 3, 1, 0),
            tok(3, 0, 1, 1),
        ],
        vec![tok(1, 0, 0, 0), tok(2, 2, 0, 0), tok(4, 0, 0, 0)],
        vec![tok(1, 0, 0, 0)],
    ];
    let current = [
        tok(2, 5, 1, 0),
        tok(2, 5, 1, 0),
        tok(4, 0, 0, 0),
        tok(2, 5, 1, 0),
        tok(1, 0, 0, 0),
    ];
    let limits = Limits::new(a, 4).unwrap();
    // Twin post-token rows, built exactly like `grammar_apply`: apply when
    // legal, else keep the row and set the stopped word to 2.
    let mut twin_states = vec![0u32; rows * s];
    for (r, seq) in seqs.iter().enumerate() {
        let mut st = TraceState::new(limits, None);
        for t in seq {
            let token = mamba3::models::ms2::grammar::Token {
                kind: t[0] as u8,
                atom_type: t[1] as u8,
                bond: t[2] as u8,
                pointer: t[3] as u8,
            };
            assert!(st.is_legal(token), "row {r} prefix stays legal");
            host::apply_token_row(
                &mut twin_states[r * s..(r + 1) * s],
                a,
                &atable,
                t[0],
                t[1],
                t[2],
                t[3],
            );
            st.apply(token).unwrap();
        }
        let c = current[r];
        let token = mamba3::models::ms2::grammar::Token {
            kind: c[0] as u8,
            atom_type: c[1] as u8,
            bond: c[2] as u8,
            pointer: c[3] as u8,
        };
        if st.is_legal(token) {
            host::apply_token_row(
                &mut twin_states[r * s..(r + 1) * s],
                a,
                &atable,
                c[0],
                c[1],
                c[2],
                c[3],
            );
        } else {
            twin_states[r * s + 3 * a + 5] = 2;
        }
    }
    let mut rng = Rng::seeded(41);
    let prev_h: Vec<f32> = rng.uniform_vec(rows * d, -2.0, 2.0);
    // Twin atom update from the post-token rows.
    let mut twin_mem = vec![f32::NAN; rows * a * d];
    let mut twin_resid = vec![0u32; rows * a];
    for r in 0..rows {
        if current[r][0] == 2 {
            let n = twin_states[r * s + 3 * a] as usize;
            assert!((1..=a).contains(&n), "row {r} count {n} names a slot");
            for j in 0..d {
                twin_mem[(r * a + n - 1) * d + j] = prev_h[r * d + j];
            }
        }
        for j in 0..a {
            twin_resid[r * a + j] = twin_states[r * s + a + j].min(7);
        }
    }
    let mut flat_tok = vec![0u32; rows * 4];
    for (r, c) in current.iter().enumerate() {
        flat_tok[r * 4..r * 4 + 4].copy_from_slice(c);
    }
    let token_t = IdTensor::from_slice(&flat_tok, vec![rows, 4], &device).unwrap();
    let gstate_t = IdTensor::from_slice(&twin_states, vec![rows, s], &device).unwrap();
    let prev_t = Tensor::<R, E>::from_f32(&prev_h, vec![rows, d], &device).unwrap();
    let mut mem_t =
        Tensor::<R, E>::from_f32(&vec![f32::NAN; rows * a * d], vec![rows, a, d], &device).unwrap();
    let mut resid_t =
        IdTensor::from_slice(&vec![0xDEAD_BEEFu32; rows * a], vec![rows * a], &device).unwrap();
    ms2::atom_memory_update(&token_t, &gstate_t, &prev_t, &mut mem_t, &mut resid_t, a).unwrap();
    check_launches(&device).unwrap();
    let mem_got = mem_t.try_to_f32().unwrap();
    let resid_got = resid_t.try_to_vec().unwrap();
    assert_eq!(mem_got.len(), twin_mem.len());
    for (i, (g, w)) in mem_got.iter().zip(twin_mem.iter()).enumerate() {
        if w.is_nan() {
            assert!(g.is_nan(), "slot {i}: poison survives");
        } else {
            assert_eq!(g.to_bits(), w.to_bits(), "slot {i}: copied row matches");
        }
    }
    assert_eq!(resid_got, twin_resid, "clamped residual ids match");
}

#[test]
fn step_logits_no_read_and_row_independent_launches() {
    let _serial = serial();
    // `step_logits` performs no device read and its launch count does not
    // depend on the rows or the token values: two consecutive positions at
    // rows 4 and rows 32 report equal launch deltas and zero read deltas.
    let device = dev();
    let model = tiny_config();
    let mut rng = Rng::seeded(43);
    let encoder = Ms2Encoder::init(&model, &device, &mut rng).unwrap();
    let decoder = Ms2Decoder::init(&model, &device, &mut rng).unwrap();
    let consts = Ms2Constants::new(&device);
    let spectra_batch = make_batch(&[51], 64, &[12], 200_000_000, 44);
    let spectra = DeviceSpectra::upload(&spectra_batch, &device).unwrap();
    let peaks = ms2::PeakBuffers::<R, E>::new(1, spectra.n_raw, 16, &device);
    let encoded = encoder.encode(&spectra, &peaks, Control::None).unwrap();
    check_launches(&device).unwrap();
    // Honest construction: B = 1 spectrum, rows_per_spectrum = rows. The
    // memory stays [1, 17, d]; only the token batch, grammar rows and formula
    // rows grow. Keys/values are per spectrum, so shapes are shared.
    //
    // Launches are attributed to per-probe tally labels: the launch counter
    // is process-global and other tests run concurrently, so a global delta
    // would see their kernels. The tally labels are thread-local, which
    // isolates this thread's launches. Reads have no labels, so the read
    // probe repeats and takes the minimum: a concurrent test can only add
    // reads, never remove this thread's.
    mamba3::backend::start_launch_tally();
    let tally_count = |label: &str| -> usize {
        mamba3::backend::launch_tally_detailed()
            .iter()
            .filter(|row| row.label == label)
            .map(|row| row.count)
            .sum()
    };
    let mut launch_deltas = Vec::new();
    let mut read_deltas = Vec::new();
    for _iter in 0..4 {
        for rows in [4usize, 32usize] {
            let enc = mamba3::models::ms2::encoder::EncoderOutput {
                x: encoded.x.slice(0, 0, 1).unwrap(),
                valid: mamba3::tensor::ops::movement::slice(&encoded.valid, 0, 0, 1)
                    .unwrap()
                    .reshape(vec![1, 16])
                    .unwrap(),
                memory: encoded.memory.slice(0, 0, 1).unwrap(),
                memory_mask: mamba3::tensor::ops::movement::slice(&encoded.memory_mask, 0, 0, 1)
                    .unwrap()
                    .reshape(vec![1, 17])
                    .unwrap(),
                pool: encoded.pool.slice(0, 0, 1).unwrap(),
                context: encoded.context.slice(0, 0, 1).unwrap(),
            };
            let mut state = decoder.start_state(&enc, rows, &device).unwrap();
            let mut gstate = ms2::grammar_state_zeros(rows, 16, &device);
            let mut meta_host = vec![0u32; rows * 12];
            for r in 0..rows {
                meta_host[r * 12 + 1] = 1;
                meta_host[r * 12 + 2] = 4;
                meta_host[r * 12 + 3] = 8;
            }
            let meta_t = IdTensor::from_slice(&meta_host, vec![rows, 12], &device).unwrap();
            // Position 0 (START) then position 1 (root ADD of different types per
            // row: token values differ across rows on purpose).
            let mut tok0 = vec![0u32; rows * 4];
            let mut tok1 = vec![0u32; rows * 4];
            for r in 0..rows {
                tok0[r * 4] = 1;
                tok1[r * 4] = 2;
                tok1[r * 4 + 1] = 1 + (r % 4) as u32;
            }
            let token0 = IdTensor::from_slice(&tok0, vec![rows, 4], &device).unwrap();
            let token1 = IdTensor::from_slice(&tok1, vec![rows, 4], &device).unwrap();
            let femb = Var::constant(
                Tensor::<R, E>::from_f32(&vec![0.1; rows * 16], vec![rows, 16], &device).unwrap(),
            );
            // Warm-up: same shapes, so matmul tuning settles before measuring.
            ms2::grammar_apply(&token0, &mut gstate, &meta_t, &consts, 16, 4).unwrap();
            decoder
                .step_logits(&enc, &femb, &token0, 0, &gstate, &mut state, rows)
                .unwrap();
            ms2::grammar_apply(&token1, &mut gstate, &meta_t, &consts, 16, 4).unwrap();
            decoder
                .step_logits(&enc, &femb, &token1, 1, &gstate, &mut state, rows)
                .unwrap();
            check_launches(&device).unwrap();
            // Two consecutive measured positions with different token values.
            let mut tok2 = vec![0u32; rows * 4];
            let mut tok3 = vec![0u32; rows * 4];
            for r in 0..rows {
                tok2[r * 4] = 4;
                tok3[r * 4] = 2;
                tok3[r * 4 + 1] = 7;
                tok3[r * 4 + 2] = 2;
                tok3[r * 4 + 3] = 0;
            }
            let token2 = IdTensor::from_slice(&tok2, vec![rows, 4], &device).unwrap();
            let token3 = IdTensor::from_slice(&tok3, vec![rows, 4], &device).unwrap();
            // Two consecutive measured positions, each under its own label.
            let t0 = tally_count("v0c.probe");
            let r0 = runtime_read_count();
            {
                let _scope = mamba3::backend::tally_scope("v0c.probe");
                ms2::grammar_apply(&token2, &mut gstate, &meta_t, &consts, 16, 4).unwrap();
                decoder
                    .step_logits(&enc, &femb, &token2, 2, &gstate, &mut state, rows)
                    .unwrap();
                check_launches(&device).unwrap();
            }
            let t1 = tally_count("v0c.probe");
            let r1 = runtime_read_count();
            {
                let _scope = mamba3::backend::tally_scope("v0c.probe");
                ms2::grammar_apply(&token3, &mut gstate, &meta_t, &consts, 16, 4).unwrap();
                decoder
                    .step_logits(&enc, &femb, &token3, 3, &gstate, &mut state, rows)
                    .unwrap();
                check_launches(&device).unwrap();
            }
            let t2 = tally_count("v0c.probe");
            let r2 = runtime_read_count();
            launch_deltas.push((rows, t1 - t0, t2 - t1));
            read_deltas.push((rows, r1 - r0, r2 - r1));
        }
    }
    mamba3::backend::stop_launch_tally();
    println!("step_logits per-probe launches (rows, pos2, pos3): {launch_deltas:?}");
    println!("step_logits per-probe read deltas (rows, pos2, pos3): {read_deltas:?}");
    let first = launch_deltas[0].1;
    for (rows, a, b) in &launch_deltas {
        assert_eq!(*a, *b, "rows {rows}: consecutive positions launch equally");
        // Not exact equality: the matmul router picks its strategy by shape,
        // and on wgpu the route chosen at 32 rows needs one more contiguous
        // copy than at 4 rows (measured: 67 versus 68). A per-row loop would
        // add launches in proportion to the 28 extra rows, which this bound
        // still catches.
        assert!(
            a.abs_diff(first) <= 2,
            "rows {rows}: launch count {a} is not independent of rows (rows 4: {first})"
        );
    }
    // Reads have no tally labels, so a concurrent test's reporting read can
    // land inside a probe. A concurrent read only adds: the minimum across
    // all probes of one shape is zero exactly when this thread reads
    // nothing itself.
    for rows in [4usize, 32usize] {
        let min_read = read_deltas
            .iter()
            .filter(|(r, _, _)| *r == rows)
            .flat_map(|(_, a, b)| [*a, *b])
            .min()
            .unwrap();
        assert_eq!(min_read, 0, "rows {rows}: a probe performs no runtime read");
    }
}
