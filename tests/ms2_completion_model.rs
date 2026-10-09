//! MC3a tests: completion-conditioned model (substructure-set encoder, exact
//! teacher forcing, trainer).
//!
//! Hand-built molecules only (atom type ids from
//! [`chem::ATOM_TYPES`](mamba3::models::ms2::chem::ATOM_TYPES)): ethanol,
//! dimethyl ether, propan-1-ol, propan-2-ol, methoxyethane, ethylamine,
//! dimethylamine, cyclopropane and a kekulized benzene ring with a methyl.
//! Every device call is followed by [`check_launches`]. Tolerances are 1e-5
//! absolute for values in [-1, 1] and 1e-5 relative otherwise, unless a test
//! states its own.

#![cfg(feature = "backend")]

use std::collections::BTreeMap;

use mamba3::backend::{Device, check_launches, runtime_read_count};
use mamba3::backends::Auto;
use mamba3::models::ms2::chem::Composition;
use mamba3::models::ms2::completion::contains_pattern;
use mamba3::models::ms2::completion_data::{CompletionExample, CompletionSet, ExtractionConfig};
use mamba3::models::ms2::completion_model::{
    COMPLETION_CHECKPOINT_FORMAT, COMPLETION_MODEL_VERSION, CompletionModel, CompletionModelConfig,
    CompletionTrainConfig, CompletionTrainer, MAX_PATTERNS, PATTERN_SLOTS, PatternBatch,
};
use mamba3::models::ms2::contain::Containment;
use mamba3::models::ms2::grammar::{
    ADD_ATOM, CANONICAL_WORK_LIMIT, CLOSE_RING, COMPLETION_GRAMMAR_VERSION, Limits, STOP, Token,
    canonical_trace, replay_exact,
};
use mamba3::models::ms2::graph::MolGraph;
use mamba3::models::ms2::targets_batch::TargetBatch;
use mamba3::nn::Module;
use mamba3::tensor::Tensor;
use mamba3::tensor::ops::index::read_all;
use mamba3::tensor::ops::ms2::{self, Ms2Constants, ReplayBuffers};
use mamba3::tensor::ops::random::Rng;

type R = Auto;
type E = f32;

/// File-local serial lock: counter-sensitive tests need exact process-global
/// launch/read counters, so the whole binary runs serially.
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn serial() -> std::sync::MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

fn dev() -> Device<R> {
    Device::<R>::default()
}

fn limits() -> Limits {
    Limits::new(16, 4).unwrap()
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

/// Ethanol `[C(H3), C(H2), O(H1)]`.
fn ethanol() -> MolGraph {
    MolGraph::new(vec![4, 3, 9], vec![(0, 1, 1), (1, 2, 1)]).unwrap()
}

/// Dimethyl ether `[C(H3), O(H0), C(H3)]`, a C2H6O isomer of ethanol.
fn dimethyl_ether() -> MolGraph {
    MolGraph::new(vec![4, 8, 4], vec![(0, 1, 1), (1, 2, 1)]).unwrap()
}

/// Propan-1-ol: a C(H3)-C(H2)-C(H2)-O(H1) chain.
fn propan_1_ol() -> MolGraph {
    MolGraph::new(vec![4, 3, 3, 9], vec![(0, 1, 1), (1, 2, 1), (2, 3, 1)]).unwrap()
}

/// Propan-2-ol: central C(H1) with two methyls and one O(H1).
fn propan_2_ol() -> MolGraph {
    MolGraph::new(vec![4, 2, 4, 9], vec![(0, 1, 1), (1, 2, 1), (1, 3, 1)]).unwrap()
}

/// Methoxyethane: C(H3)-O(H0)-C(H2)-C(H3), a C3H8O isomer.
fn methoxyethane() -> MolGraph {
    MolGraph::new(vec![4, 8, 3, 4], vec![(0, 1, 1), (1, 2, 1), (2, 3, 1)]).unwrap()
}

/// Ethylamine: C(H3)-C(H2)-N(H2).
fn ethylamine() -> MolGraph {
    MolGraph::new(vec![4, 3, 7], vec![(0, 1, 1), (1, 2, 1)]).unwrap()
}

/// Dimethylamine: C(H3)-N(H1)-C(H3), a C2H7N isomer of ethylamine (the
/// two-bond nitrogen carries one hydrogen, type 6).
fn dimethylamine() -> MolGraph {
    MolGraph::new(vec![4, 6, 4], vec![(0, 1, 1), (1, 2, 1)]).unwrap()
}

/// Cyclopropane: three `C(H2)` (id 3) in a single-bond ring.
fn cyclopropane() -> MolGraph {
    MolGraph::new(vec![3, 3, 3], vec![(0, 1, 1), (1, 2, 1), (2, 0, 1)]).unwrap()
}

/// Kekulized benzene ring with a methyl: six ring atoms with alternating
/// single/double bonds (`C(H1)` id 2, ipso `C(H0)` id 1) plus a `C(H3)`
/// methyl (id 4) on atom 0.
fn methylbenzene() -> MolGraph {
    MolGraph::new(
        vec![1, 2, 2, 2, 2, 2, 4],
        vec![
            (0, 1, 1),
            (1, 2, 2),
            (2, 3, 1),
            (3, 4, 2),
            (4, 5, 1),
            (5, 0, 2),
            (0, 6, 1),
        ],
    )
    .unwrap()
}

/// The nine overfit molecules in a fixed order.
fn nine_molecules() -> Vec<(&'static str, MolGraph)> {
    vec![
        ("ethanol", ethanol()),
        ("dimethyl ether", dimethyl_ether()),
        ("propan-1-ol", propan_1_ol()),
        ("propan-2-ol", propan_2_ol()),
        ("methoxyethane", methoxyethane()),
        ("ethylamine", ethylamine()),
        ("dimethylamine", dimethylamine()),
        ("cyclopropane", cyclopropane()),
        ("methylbenzene", methylbenzene()),
    ]
}

/// Canonical trace and composition of `graph`, asserting the trace replays
/// under the exact-completion grammar to a stopped, complete state.
fn trace_and_composition(graph: &MolGraph) -> (Vec<Token>, Composition) {
    let canonical = canonical_trace(graph, limits(), CANONICAL_WORK_LIMIT).unwrap();
    let composition = graph.composition();
    let end = replay_exact(&canonical.trace, limits(), composition).unwrap();
    assert!(
        end.stopped() && end.is_complete(),
        "canonical trace replays exact to a complete molecule"
    );
    (canonical.trace, composition)
}

/// A [`CompletionSet`] over the nine hand-built molecules.
fn nine_set() -> CompletionSet {
    let mut examples = Vec::new();
    for (i, (key, graph)) in nine_molecules().into_iter().enumerate() {
        let (trace, composition) = trace_and_composition(&graph);
        examples.push(CompletionExample {
            key: key.to_string(),
            identity_group: i as u64,
            source_index: i,
            target: graph,
            composition,
            trace,
            skeleton_trace: Vec::new(),
        });
    }
    CompletionSet {
        limits: limits(),
        examples,
        skipped: BTreeMap::new(),
        max_expansions: CANONICAL_WORK_LIMIT,
    }
}

fn train_config() -> CompletionTrainConfig {
    let extraction = ExtractionConfig::default();
    CompletionTrainConfig {
        lr: 3e-3,
        weight_decay: 0.0,
        grad_clip: None,
        seed: 1,
        extraction_seed: 11,
        pattern_source:
            mamba3::models::ms2::completion_data::PatternSource::RandomPatches(extraction.clone()),
        extraction,
        fingerprint_mode: None,
        fingerprint_threshold: 0.1,
    }
}

fn new_trainer(device: &Device<R>) -> CompletionTrainer<R, E> {
    CompletionTrainer::new(&CompletionModelConfig::small(), &train_config(), device).unwrap()
}

#[test]
fn config_versions_and_ranges() {
    let _lock = serial();
    // The shipped configs name the frozen versions and validate; the mapped
    // `ModelConfig` validates too.
    for config in [
        CompletionModelConfig::small(),
        CompletionModelConfig::base(),
    ] {
        assert_eq!(config.version, COMPLETION_MODEL_VERSION);
        config.validate().unwrap();
    }
    let mut bad = CompletionModelConfig::small();
    bad.version = "bogus".to_string();
    assert!(bad.validate().is_err());
    let mut bad = CompletionModelConfig::small();
    bad.grammar = "bogus".to_string();
    assert!(bad.validate().is_err());
    // A checkpoint trained under the v1 exact-completion grammar is refused
    // at load: the error names both the stored version and the current one.
    let mut stale = CompletionModelConfig::small();
    stale.grammar = "completion-exact-v1".to_string();
    let err = stale
        .validate()
        .err()
        .expect("a v1 grammar config is refused");
    assert!(
        err.to_string().contains("completion-exact-v1"),
        "the error names the stored v1 version: {err}"
    );
    assert!(
        err.to_string().contains(COMPLETION_GRAMMAR_VERSION),
        "the error names the current {COMPLETION_GRAMMAR_VERSION}: {err}"
    );
    let mut bad = CompletionModelConfig::small();
    bad.max_atoms = 33;
    assert!(bad.validate().is_err());
    let mut bad = CompletionModelConfig::small();
    bad.max_ring_closures = 9;
    assert!(bad.validate().is_err());
    let mut bad = CompletionModelConfig::small();
    bad.message_rounds = 0;
    assert!(bad.validate().is_err());
    let mut bad = CompletionModelConfig::small();
    bad.attention_heads = 3;
    assert!(bad.validate().is_err());
    let mut bad = train_config();
    bad.lr = -1.0;
    assert!(bad.validate().is_err());
}

#[test]
fn pattern_batch_layout() {
    let _lock = serial();
    // Query 0 holds two patterns (a C(H2)-O(H1) bond and a lone C(H3));
    // query 1 holds none.
    let p0 = MolGraph::new(vec![3, 9], vec![(0, 1, 1)]).unwrap();
    let p1 = MolGraph::new(vec![4], vec![]).unwrap();
    let q0 = vec![p0, p1];
    let q1: Vec<MolGraph> = vec![];
    let comp0 = ethanol().composition();
    let comp1 = dimethyl_ether().composition();
    let batch = PatternBatch::build(&[q0.as_slice(), q1.as_slice()], &[comp0, comp1]).unwrap();
    assert_eq!(batch.queries, 2);
    let p = PATTERN_SLOTS;
    // Slots: query 0 takes [3, 9, 4], then padding; query 1 is all padding.
    assert_eq!(&batch.types[0..3], &[3, 9, 4]);
    assert!(batch.types[3..p].iter().all(|&t| t == 0));
    assert!(batch.types[p..].iter().all(|&t| t == 0));
    // Open values: C(H2) with one bond has residual 1 -> 2; O(H1) with one
    // bond has residual 0 -> 1; lone C(H3) has residual 1 -> 2.
    assert_eq!(&batch.open[0..3], &[2, 1, 2]);
    assert!(batch.open[3..].iter().all(|&o| o == 0));
    assert_eq!(&batch.valid[0..3], &[1.0, 1.0, 1.0]);
    assert!(batch.valid[3..].iter().all(|&v| v == 0.0));
    // Adjacency: symmetric, block-diagonal, per order. Only slots 0-1 are
    // bonded (order 1); everything else is 0.
    let at =
        |b: usize, o: usize, i: usize, j: usize| batch.adjacency[(b * 3 + o) * p * p + i * p + j];
    assert_eq!(at(0, 0, 0, 1), 1.0);
    assert_eq!(at(0, 0, 1, 0), 1.0);
    let ones: f32 = batch.adjacency.iter().sum();
    assert_eq!(ones, 2.0, "exactly the two symmetric entries are set");
    // Features: ln(1 + count).
    for e in 0..10 {
        assert_eq!(batch.features[e], (1.0 + f32::from(comp0[e])).ln());
        assert_eq!(batch.features[10 + e], (1.0 + f32::from(comp1[e])).ln());
    }
    // The three `Error::Config` cases.
    let many: Vec<MolGraph> = (0..MAX_PATTERNS + 1)
        .map(|_| MolGraph::new(vec![4], vec![]).unwrap())
        .collect();
    let err = PatternBatch::build(&[many.as_slice()], &[comp0]).unwrap_err();
    assert!(
        err.to_string().contains("patterns"),
        "too many patterns names the count: {err}"
    );
    let chain = || {
        MolGraph::new(
            vec![4, 3, 3, 3, 3, 3, 3, 4],
            vec![
                (0, 1, 1),
                (1, 2, 1),
                (2, 3, 1),
                (3, 4, 1),
                (4, 5, 1),
                (5, 6, 1),
                (6, 7, 1),
            ],
        )
        .unwrap()
    };
    // Enough eight-atom chains to overrun the slots, whatever the width is.
    let big: Vec<MolGraph> = (0..PATTERN_SLOTS / 8 + 1).map(|_| chain()).collect();
    let err = PatternBatch::build(&[big.as_slice()], &[comp0]).unwrap_err();
    assert!(
        err.to_string().contains("pattern atoms"),
        "too many atoms names the total: {err}"
    );
    let empty = MolGraph::new(vec![], vec![]).unwrap();
    let with_empty = vec![empty];
    let err = PatternBatch::build(&[with_empty.as_slice()], &[comp0]).unwrap_err();
    assert!(
        err.to_string().contains("empty"),
        "an empty pattern is rejected: {err}"
    );
}

#[test]
fn build_exact_layout_and_rejection() {
    let _lock = serial();
    // One slot per query, q = 1, labeled = 1, meta flag 2.
    let (trace, comp) = trace_and_composition(&ethanol());
    let batch = TargetBatch::build_exact(&[trace.as_slice()], &[comp], limits()).unwrap();
    assert_eq!((batch.spectra, batch.slots), (1, 1));
    assert_eq!(batch.max_steps, limits().max_steps());
    assert_eq!(batch.q, vec![1.0]);
    assert_eq!(batch.labeled, vec![1]);
    assert_eq!(batch.meta[0], trace.len() as u32);
    assert_eq!(batch.meta[1], 2, "meta flag 2 is exact completion");
    for e in 0..10 {
        assert_eq!(batch.meta[2 + e], u32::from(comp[e]));
    }
    // Token rows match the trace.
    for (t, token) in trace.iter().enumerate() {
        assert_eq!(
            &batch.tokens[t * 4..t * 4 + 4],
            &[
                u32::from(token.kind),
                u32::from(token.atom_type),
                u32::from(token.bond),
                u32::from(token.pointer)
            ]
        );
    }
    // A trace that stops one atom short is rejected: START, root, one child,
    // STOP under the two-heavy-atom composition.
    let short = vec![
        Token {
            kind: 1,
            atom_type: 0,
            bond: 0,
            pointer: 0,
        },
        Token {
            kind: ADD_ATOM,
            atom_type: 4,
            bond: 0,
            pointer: 0,
        },
        Token {
            kind: ADD_ATOM,
            atom_type: 3,
            bond: 1,
            pointer: 0,
        },
        Token {
            kind: STOP,
            atom_type: 0,
            bond: 0,
            pointer: 0,
        },
    ];
    let err = TargetBatch::build_exact(&[short.as_slice()], &[comp], limits());
    let err = match err {
        Err(e) => e,
        Ok(_) => panic!("the short trace must be rejected"),
    };
    assert!(
        err.to_string().contains("query 0"),
        "the short trace names its query: {err}"
    );
    // An over-long trace and a length/count mismatch are rejected too.
    let long: Vec<Token> = trace
        .iter()
        .cycle()
        .take(limits().max_steps() + 1)
        .copied()
        .collect();
    assert!(TargetBatch::build_exact(&[long.as_slice()], &[comp], limits()).is_err());
    assert!(
        TargetBatch::build_exact(&[trace.as_slice(), trace.as_slice()], &[comp], limits()).is_err()
    );
}

#[test]
fn teacher_rejects_non_exact_targets() {
    let _lock = serial();
    let device = dev();
    let mut rng = Rng::seeded(10);
    let model: CompletionModel<R, E> =
        CompletionModel::init(&CompletionModelConfig::small(), &device, &mut rng).unwrap();
    let (trace, comp) = trace_and_composition(&ethanol());
    let p0 = MolGraph::new(vec![3, 9], vec![(0, 1, 1)]).unwrap();
    let pats = vec![p0];
    let batch = PatternBatch::build(&[pats.as_slice()], &[comp]).unwrap();
    let constants = Ms2Constants::new(&device);
    let config_err = |err: mamba3::error::Error| {
        assert!(
            matches!(err, mamba3::error::Error::Config(_)),
            "non-exact targets are Error::Config: {err}"
        );
    };
    let probe = |targets: &TargetBatch| {
        model
            .teacher(&batch, targets, &constants, &device)
            .map(|_| ())
    };
    // A flag-1 (`TargetBatch::build`) batch is rejected.
    let mut flag1 = TargetBatch::build_exact(&[trace.as_slice()], &[comp], limits()).unwrap();
    flag1.meta[1] = 1;
    config_err(probe(&flag1).expect_err("flag 1 is rejected"));
    // A wrong slot count is rejected.
    let mut slots = TargetBatch::build_exact(&[trace.as_slice()], &[comp], limits()).unwrap();
    slots.slots = 2;
    config_err(probe(&slots).expect_err("two slots are rejected"));
    // A wrong horizon is rejected.
    let mut horizon = TargetBatch::build_exact(&[trace.as_slice()], &[comp], limits()).unwrap();
    horizon.max_steps += 1;
    config_err(probe(&horizon).expect_err("a wrong T is rejected"));
    check_launches(&device).unwrap();
}

#[test]
fn pattern_batch_validate_rejects() {
    let _lock = serial();
    let p0 = MolGraph::new(vec![4, 3], vec![(0, 1, 1)]).unwrap();
    let pats = vec![p0];
    let comp = ethanol().composition();
    let fresh = || PatternBatch::build(&[pats.as_slice()], &[comp]).unwrap();
    fresh().validate().unwrap();
    let config_err = |err: mamba3::error::Error, what: &str| {
        assert!(
            matches!(err, mamba3::error::Error::Config(_)),
            "{what} is Error::Config: {err}"
        );
    };
    let mut bad = fresh();
    bad.types.pop();
    config_err(bad.validate().expect_err("lengths"), "short types");
    let mut bad = fresh();
    bad.valid[0] = 0.5;
    config_err(bad.validate().expect_err("valid"), "valid 0.5");
    let mut bad = fresh();
    bad.types[0] = 18;
    config_err(bad.validate().expect_err("types"), "types 18");
    let mut bad = fresh();
    bad.open[0] = 9;
    config_err(bad.validate().expect_err("open"), "open 9");
    let mut bad = fresh();
    bad.adjacency[0] = f32::NAN;
    config_err(bad.validate().expect_err("adjacency"), "NaN adjacency");
    let mut bad = fresh();
    bad.features[0] = f32::INFINITY;
    config_err(bad.validate().expect_err("features"), "infinite features");
    // Adjacency touching an invalid slot (flat P + 2 is order 0, slots 1,2;
    // slot 2 is padding here).
    let mut bad = fresh();
    bad.adjacency[PATTERN_SLOTS + 2] = 1.0;
    let err = bad.validate().expect_err("padding adjacency");
    assert!(
        err.to_string().contains("invalid slot"),
        "padding adjacency names the rule: {err}"
    );
    config_err(err, "padding adjacency");
    // Asymmetric adjacency between two valid slots (the build wrote a
    // symmetric 1.0 there; 2.0 breaks the mirror).
    let mut bad = fresh();
    bad.adjacency[1] = 2.0;
    let err = bad.validate().expect_err("asymmetry");
    assert!(
        err.to_string().contains("symmetry"),
        "asymmetry names the rule: {err}"
    );
    config_err(err, "asymmetry");
    // A non-zero diagonal on a valid slot.
    let mut bad = fresh();
    bad.adjacency[PATTERN_SLOTS + 1] = 1.0;
    let err = bad.validate().expect_err("diagonal");
    assert!(
        err.to_string().contains("diagonal"),
        "a diagonal entry names the rule: {err}"
    );
    config_err(err, "diagonal");
}

#[test]
fn training_with_grad_clip() {
    let _lock = serial();
    let device = dev();
    let set = nine_set();
    let indices: Vec<usize> = (0..set.examples.len()).collect();
    let clipped = CompletionTrainConfig {
        grad_clip: Some(0.5),
        ..train_config()
    };
    let mut trainer: CompletionTrainer<R, E> =
        CompletionTrainer::new(&CompletionModelConfig::small(), &clipped, &device).unwrap();
    let mut losses = Vec::new();
    for _ in 0..20 {
        trainer.request_report();
        losses.push(trainer.step(&set, &indices, 0).unwrap().unwrap());
    }
    check_launches(&device).unwrap();
    println!("clipped losses: {losses:?}");
    assert!(losses.iter().all(|loss| loss.is_finite()));
    assert!(
        losses[19] < losses[0],
        "the final loss is below the first: {losses:?}"
    );
}

#[test]
fn encoder_is_invariant_to_pattern_and_atom_order() {
    let _lock = serial();
    let device = dev();
    let mut rng = Rng::seeded(5);
    let model: CompletionModel<R, E> =
        CompletionModel::init(&CompletionModelConfig::small(), &device, &mut rng).unwrap();
    // One query with three patterns.
    let p0 = MolGraph::new(vec![4, 3], vec![(0, 1, 1)]).unwrap();
    let p1 = MolGraph::new(vec![9], vec![]).unwrap();
    let p2 = MolGraph::new(vec![3, 9], vec![(0, 1, 1)]).unwrap();
    let (trace, comp) = trace_and_composition(&ethanol());
    let pats = vec![p0, p1, p2];
    // Permuted: reversed pattern order, atoms swapped inside each pattern.
    let q0 = pats[2].permuted(&[1, 0]).unwrap();
    let q1 = pats[1].permuted(&[0]).unwrap();
    let q2 = pats[0].permuted(&[1, 0]).unwrap();
    let permuted = vec![q0, q1, q2];
    let batch_a = PatternBatch::build(&[pats.as_slice()], &[comp]).unwrap();
    let batch_b = PatternBatch::build(&[permuted.as_slice()], &[comp]).unwrap();
    let out_a = model.encode(&batch_a, &device).unwrap();
    let out_b = model.encode(&batch_b, &device).unwrap();
    check_launches(&device).unwrap();
    let pool_a = out_a.pool.try_to_f32().unwrap();
    let pool_b = out_b.pool.try_to_f32().unwrap();
    assert_close(&pool_a, &pool_b, 1e-5, "pool");
    // The multiset of valid rows of `x`: sort rows lexicographically first.
    let sorted_rows = |x: &[f32], valid: &[f32]| {
        let d = 64usize;
        let mut rows: Vec<Vec<f32>> = Vec::new();
        for (slot, &v) in valid.iter().enumerate() {
            if v != 0.0 {
                rows.push(x[slot * d..(slot + 1) * d].to_vec());
            }
        }
        rows.sort_by(|a, b| a.partial_cmp(b).expect("finite encoder rows"));
        rows.concat()
    };
    let x_a = out_a.x.try_to_f32().unwrap();
    let x_b = out_b.x.try_to_f32().unwrap();
    assert_close(
        &sorted_rows(&x_a, &batch_a.valid),
        &sorted_rows(&x_b, &batch_b.valid),
        1e-4,
        "valid x rows as a multiset",
    );
    // The teacher NLL of the query is unchanged.
    let targets = TargetBatch::build_exact(&[trace.as_slice()], &[comp], limits()).unwrap();
    let constants = Ms2Constants::new(&device);
    let (tout_a, _) = model
        .teacher(&batch_a, &targets, &constants, &device)
        .unwrap();
    let (tout_b, _) = model
        .teacher(&batch_b, &targets, &constants, &device)
        .unwrap();
    check_launches(&device).unwrap();
    let nll_a = tout_a.nll.try_to_f32().unwrap();
    let nll_b = tout_b.nll.try_to_f32().unwrap();
    assert_close(&nll_a, &nll_b, 1e-4, "teacher NLL");
}

#[test]
fn padding_is_exact_zero_and_inert() {
    let _lock = serial();
    let device = dev();
    let mut rng = Rng::seeded(6);
    let model: CompletionModel<R, E> =
        CompletionModel::init(&CompletionModelConfig::small(), &device, &mut rng).unwrap();
    let p0 = MolGraph::new(vec![4, 3], vec![(0, 1, 1)]).unwrap();
    let (trace, comp) = trace_and_composition(&ethanol());
    let pats = vec![p0];
    let batch = PatternBatch::build(&[pats.as_slice()], &[comp]).unwrap();
    let out = model.encode(&batch, &device).unwrap();
    check_launches(&device).unwrap();
    // Padded rows of `x` are exactly 0.
    let x = out.x.try_to_f32().unwrap();
    let d = 64usize;
    for slot in 2..PATTERN_SLOTS {
        for v in &x[slot * d..(slot + 1) * d] {
            assert_eq!(
                v.to_bits(),
                0.0f32.to_bits(),
                "padded row {slot} is exactly zero"
            );
        }
    }
    let pool = out.pool.try_to_f32().unwrap();
    let memory = out.memory.try_to_f32().unwrap();
    // Garbage in the padded `types`/`open` host entries, validity untouched:
    // in-range values stay inert by selection.
    let mut garbage = PatternBatch::build(&[pats.as_slice()], &[comp]).unwrap();
    for slot in 2..PATTERN_SLOTS {
        garbage.types[slot] = 17;
        garbage.open[slot] = 8;
    }
    garbage.validate().unwrap();
    let dirty = model.encode(&garbage, &device).unwrap();
    check_launches(&device).unwrap();
    assert_eq!(
        dirty.pool.try_to_f32().unwrap(),
        pool,
        "pool is bit-identical under padded types/open garbage"
    );
    assert_eq!(
        dirty.memory.try_to_f32().unwrap(),
        memory,
        "memory is bit-identical under padded types/open garbage"
    );
    // The teacher loss is bit-identical too.
    let targets = TargetBatch::build_exact(&[trace.as_slice()], &[comp], limits()).unwrap();
    let constants = Ms2Constants::new(&device);
    let loss_clean = model
        .teacher(&batch, &targets, &constants, &device)
        .unwrap()
        .1
        .try_to_f32()
        .unwrap();
    let loss_dirty = model
        .teacher(&garbage, &targets, &constants, &device)
        .unwrap()
        .1
        .try_to_f32()
        .unwrap();
    check_launches(&device).unwrap();
    assert_eq!(
        loss_clean, loss_dirty,
        "teacher loss is bit-identical under padded types/open garbage"
    );
    // Garbage or NaN adjacency touching a padded slot is rejected instead.
    let mut bad_adj = PatternBatch::build(&[pats.as_slice()], &[comp]).unwrap();
    bad_adj.adjacency[PATTERN_SLOTS + 2] = 1.0;
    assert!(bad_adj.validate().is_err());
    assert!(model.encode(&bad_adj, &device).is_err());
    let mut nan_adj = PatternBatch::build(&[pats.as_slice()], &[comp]).unwrap();
    nan_adj.adjacency[(2 * PATTERN_SLOTS + 2) * PATTERN_SLOTS + 3] = f32::NAN;
    let err = nan_adj.validate().expect_err("NaN adjacency is rejected");
    assert!(
        matches!(err, mamba3::error::Error::Config(_)),
        "NaN adjacency is Error::Config: {err}"
    );
    assert!(model.encode(&nan_adj, &device).is_err());
}

#[test]
fn zero_patterns_is_formula_only() {
    let _lock = serial();
    let device = dev();
    let mut rng = Rng::seeded(7);
    let model: CompletionModel<R, E> =
        CompletionModel::init(&CompletionModelConfig::small(), &device, &mut rng).unwrap();
    let (trace, comp) = trace_and_composition(&ethanol());
    let no_pats: Vec<MolGraph> = vec![];
    let empty = PatternBatch::build(&[no_pats.as_slice()], &[comp]).unwrap();
    let out = model.encode(&empty, &device).unwrap();
    check_launches(&device).unwrap();
    // `pool == g` exactly, and the memory mask is `[1, 0, ...]`.
    let pool = out.pool.try_to_f32().unwrap();
    let g = out.context.try_to_f32().unwrap();
    assert_eq!(pool, g, "pool equals the composition vector exactly");
    let mask = out.memory_mask.to_f32();
    assert_eq!(mask.len(), 1 + PATTERN_SLOTS);
    assert_eq!(mask[0], 1.0);
    assert!(mask[1..].iter().all(|&v| v == 0.0));
    // The NLL is finite, and a mixed batch gives each query its lone NLL.
    let mut trainer = new_trainer(&device);
    let p0 = MolGraph::new(vec![4, 3], vec![(0, 1, 1)]).unwrap();
    let pats = vec![p0];
    let nll_empty = trainer
        .teacher_eval_with(&[no_pats.as_slice()], &[trace.as_slice()], &[comp])
        .unwrap();
    let nll_full = trainer
        .teacher_eval_with(&[pats.as_slice()], &[trace.as_slice()], &[comp])
        .unwrap();
    assert!(nll_empty[0].is_finite());
    assert!(nll_full[0].is_finite());
    let nll_mixed = trainer
        .teacher_eval_with(
            &[pats.as_slice(), no_pats.as_slice()],
            &[trace.as_slice(), trace.as_slice()],
            &[comp, comp],
        )
        .unwrap();
    check_launches(&device).unwrap();
    assert_close(
        &nll_mixed,
        &[nll_full[0], nll_empty[0]],
        1e-4,
        "batch independence",
    );
}

#[test]
fn exact_masks_in_teacher_forcing() {
    let _lock = serial();
    let device = dev();
    let mut rng = Rng::seeded(8);
    let model: CompletionModel<R, E> =
        CompletionModel::init(&CompletionModelConfig::small(), &device, &mut rng).unwrap();
    // Cyclopropane's canonical trace holds a root ADD, a child ADD, a
    // CLOSE_RING and the final STOP. Output position `i` predicts token
    // `i + 1`, so token `pos` is scored at `use_mask` row `pos - 1`.
    let (trace, comp) = trace_and_composition(&cyclopropane());
    let root = trace
        .iter()
        .position(|t| t.kind == ADD_ATOM)
        .expect("a root ADD");
    assert_eq!(root, 1, "the root ADD is token 1");
    let child = trace
        .iter()
        .enumerate()
        .skip(root + 1)
        .find(|(_, t)| t.kind == ADD_ATOM)
        .map(|(i, _)| i)
        .expect("a child ADD");
    let close = trace
        .iter()
        .position(|t| t.kind == CLOSE_RING)
        .expect("a CLOSE_RING");
    let stop = trace.len() - 1;
    assert_eq!(trace[stop].kind, STOP);
    let p0 = MolGraph::new(vec![3, 9], vec![(0, 1, 1)]).unwrap();
    let pats = vec![p0];
    let batch = PatternBatch::build(&[pats.as_slice()], &[comp]).unwrap();
    let targets = TargetBatch::build_exact(&[trace.as_slice()], &[comp], limits()).unwrap();
    // The host `use_mask` rows: root ADD, child ADD, CLOSE_RING, STOP, and a
    // padding position past the trace end.
    let row = |pos: usize| {
        let base = (pos - 1) * 4;
        targets.use_mask[base..base + 4].to_vec()
    };
    assert_eq!(row(root), vec![1.0, 1.0, 0.0, 0.0], "root ADD row");
    assert_eq!(row(child), vec![1.0, 1.0, 1.0, 1.0], "child ADD row");
    assert_eq!(row(close), vec![1.0, 0.0, 1.0, 1.0], "CLOSE_RING row");
    assert_eq!(row(stop), vec![1.0, 0.0, 0.0, 0.0], "STOP row");
    let pad_base = trace.len() * 4;
    assert_eq!(
        targets.use_mask[pad_base..pad_base + 4].to_vec(),
        vec![0.0, 0.0, 0.0, 0.0],
        "a padding row is all zero"
    );
    let constants = Ms2Constants::new(&device);
    let out = model
        .teacher(&batch, &targets, &constants, &device)
        .unwrap()
        .0;
    check_launches(&device).unwrap();
    // At the STOP position the kind field log-probability is exactly 0:
    // STOP is the only legal kind on a complete molecule.
    let fields = out.field_log_prob.try_to_f32().unwrap();
    assert_eq!(
        fields[(stop - 1) * 4].to_bits(),
        0.0f32.to_bits(),
        "STOP kind log-probability is exactly 0"
    );
    // The replay's first-illegal-step column is `u32::MAX` for every row.
    let uploaded = targets.upload::<R, E>(&device).unwrap();
    let buffers = ReplayBuffers::poisoned(1, targets.max_steps, 16, &device).unwrap();
    ms2::grammar_replay(
        &uploaded.tokens,
        &uploaded.meta,
        &constants,
        16,
        4,
        &buffers,
    )
    .unwrap();
    check_launches(&device).unwrap();
    let atoms = buffers.atoms.try_to_vec().unwrap();
    assert_eq!(
        atoms[16],
        u32::MAX,
        "the first-illegal-step column is u32::MAX"
    );
}

#[test]
fn gradients_reach_every_parameter() {
    let _lock = serial();
    let device = dev();
    let mut rng = Rng::seeded(9);
    let model: CompletionModel<R, E> =
        CompletionModel::init(&CompletionModelConfig::small(), &device, &mut rng).unwrap();
    let (eth_trace, eth_comp) = trace_and_composition(&ethanol());
    let (ether_trace, ether_comp) = trace_and_composition(&dimethyl_ether());
    let p_eth = MolGraph::new(vec![3, 9], vec![(0, 1, 1)]).unwrap();
    let p_ether = MolGraph::new(vec![4, 8, 4], vec![(0, 1, 1), (1, 2, 1)]).unwrap();
    let pats_eth = vec![p_eth];
    let pats_ether = vec![p_ether];
    let batch = PatternBatch::build(
        &[pats_eth.as_slice(), pats_ether.as_slice()],
        &[eth_comp, ether_comp],
    )
    .unwrap();
    let targets = TargetBatch::build_exact(
        &[eth_trace.as_slice(), ether_trace.as_slice()],
        &[eth_comp, ether_comp],
        limits(),
    )
    .unwrap();
    let constants = Ms2Constants::new(&device);
    let loss_of = || {
        model
            .teacher(&batch, &targets, &constants, &device)
            .unwrap()
            .1
            .try_to_f32()
            .unwrap()[0]
    };
    let (_, loss) = model
        .teacher(&batch, &targets, &constants, &device)
        .unwrap();
    check_launches(&device).unwrap();
    assert!(loss.try_to_f32().unwrap()[0].is_finite());
    let grads = loss.backward_retain().unwrap();
    let named = model.named_parameters();
    assert!(!named.is_empty());
    // One batched read for every parameter gradient: other tests in this
    // binary run concurrently and read the device, so per-parameter reads
    // here would pollute their counters (see
    // `training_step_performs_no_device_read`).
    let grad_tensors: Vec<Tensor<R, E>> = named
        .iter()
        .map(|(name, param)| {
            grads
                .get(param.id())
                .unwrap_or_else(|| panic!("{name} has no gradient"))
                .clone()
        })
        .collect();
    let grad_refs: Vec<&Tensor<R, E>> = grad_tensors.iter().collect();
    let (_, grad_floats) = read_all(&[], &grad_refs).unwrap();
    let grad_of = |name: &str| {
        let (index, _) = named
            .iter()
            .enumerate()
            .find(|(_, (n, _))| n == name)
            .expect(name);
        &grad_floats[index]
    };
    let norm_of = |name: &str| {
        let values = grad_of(name);
        assert!(
            values.iter().all(|v| v.is_finite()),
            "{name} has a non-finite gradient"
        );
        values.iter().map(|v| v * v).sum::<f32>().sqrt()
    };
    for (name, _) in &named {
        assert!(
            grad_of(name).iter().all(|v| v.is_finite()),
            "{name} has a non-finite gradient"
        );
    }
    assert!(norm_of("encoder.type_emb") > 0.0, "type_emb moves");
    assert!(norm_of("encoder.open_emb") > 0.0, "open_emb moves");
    assert!(norm_of("encoder.row_in.weight") > 0.0, "row_in moves");
    assert!(norm_of("encoder.memory_in.weight") > 0.0, "memory_in moves");
    let bond_name = (0..3)
        .flat_map(|r| (0..3).map(move |o| format!("encoder.round.{r}.bond.{o}.weight")))
        .find(|n| norm_of(n) > 0.0)
        .expect("at least one bond weight moves");
    // Finite differences on three entries of `type_emb` (used rows only) and
    // three of a bond weight, in the style of
    // `graph_loss_gradient_matches_finite_differences` (same step and
    // tolerance).
    let check = |name: &str, candidates: Vec<usize>| {
        let (_, param) = named.iter().find(|(n, _)| n == name).expect(name);
        let shape = param.shape().dims().to_vec();
        let base = param.value().to_f32();
        let analytic = grad_of(name);
        let mut order = candidates;
        order.retain(|&i| i < base.len());
        order.sort_by(|&a, &b| {
            analytic[b]
                .abs()
                .partial_cmp(&analytic[a].abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let idxs: Vec<usize> = order.into_iter().take(3).collect();
        assert_eq!(idxs.len(), 3, "{name}: three entries to sample");
        for &idx in &idxs {
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
            assert!(
                numeric.abs() > 1e-4,
                "{name}[{idx}]: the sampled entry carries gradient (numeric={numeric})"
            );
            assert!(
                (analytic_v - numeric).abs() <= 2e-2 * numeric.abs() + 1e-3,
                "{name}[{idx}]: analytic={analytic_v} numeric={numeric}"
            );
        }
    };
    let d = 64usize;
    let type_all: Vec<usize> = [3usize, 4, 8, 9]
        .iter()
        .flat_map(|&r| (0..d).map(move |c| r * d + c))
        .collect();
    check("encoder.type_emb", type_all);
    let (_, bond_param) = named.iter().find(|(n, _)| n == &bond_name).unwrap();
    let all: Vec<usize> = (0..bond_param.numel()).collect();
    check(&bond_name, all);
    check_launches(&device).unwrap();
}

#[test]
fn overfit_reduces_nll() {
    let _lock = serial();
    // Nine molecules, extracted patterns at a fixed draw, batch 9: the mean
    // NLL falls below 30% of its initial value within 200 steps.
    let device = dev();
    let set = nine_set();
    let indices: Vec<usize> = (0..set.examples.len()).collect();
    let mut trainer = new_trainer(&device);
    trainer.request_report();
    let initial = trainer.step(&set, &indices, 0).unwrap().unwrap();
    println!("overfit step 0: loss {initial}");
    check_launches(&device).unwrap();
    let mut curve = vec![(0, initial)];
    let mut done = false;
    for step in 1..=200 {
        if step % 10 == 0 {
            trainer.request_report();
        }
        let report = trainer.step(&set, &indices, 0).unwrap();
        if let Some(loss) = report {
            curve.push((step, loss));
            println!("overfit step {step}: loss {loss}");
            if loss < 0.3 * initial {
                done = true;
                break;
            }
        }
    }
    check_launches(&device).unwrap();
    println!("overfit curve: {curve:?}");
    let final_loss = curve.last().copied().unwrap().1;
    assert!(
        done && final_loss < 0.3 * initial,
        "loss falls below 30%: {curve:?}"
    );
}

#[test]
fn patterns_condition_the_decoder() {
    let _lock = serial();
    // Isomer pairs with fixed discriminating patterns, written out here. The
    // compositions within a pair are identical, so only the patterns tell
    // the isomers apart.
    let device = dev();
    let names = [
        "ethanol",
        "dimethyl ether",
        "propan-1-ol",
        "propan-2-ol",
        "methoxyethane",
        "ethylamine",
        "dimethylamine",
    ];
    let graphs = [
        ethanol(),
        dimethyl_ether(),
        propan_1_ol(),
        propan_2_ol(),
        methoxyethane(),
        ethylamine(),
        dimethylamine(),
    ];
    // Fixed discriminating patterns: ethanol sees its C(H2)-O(H1) bond,
    // dimethyl ether its C-O-C spine, propan-1-ol its terminal C-C-O(H),
    // propan-2-ol its three-atom C-C(-O) centre, methoxyethane its C-O-C,
    // and each amine its whole heavy-atom spine.
    let fixed: Vec<Vec<MolGraph>> = vec![
        vec![MolGraph::new(vec![3, 9], vec![(0, 1, 1)]).unwrap()],
        vec![MolGraph::new(vec![4, 8, 4], vec![(0, 1, 1), (1, 2, 1)]).unwrap()],
        vec![MolGraph::new(vec![3, 3, 9], vec![(0, 1, 1), (1, 2, 1)]).unwrap()],
        vec![MolGraph::new(vec![2, 4, 9], vec![(0, 1, 1), (0, 2, 1)]).unwrap()],
        vec![MolGraph::new(vec![4, 8, 3], vec![(0, 1, 1), (1, 2, 1)]).unwrap()],
        vec![MolGraph::new(vec![4, 3, 7], vec![(0, 1, 1), (1, 2, 1)]).unwrap()],
        vec![MolGraph::new(vec![4, 6, 4], vec![(0, 1, 1), (1, 2, 1)]).unwrap()],
    ];
    // Each fixed pattern is contained in its molecule (sanity: the test
    // conditions on true substructures).
    for ((name, graph), pats) in names.iter().zip(graphs.iter()).zip(fixed.iter()) {
        for pat in pats {
            assert_eq!(
                contains_pattern(graph, pat, 10_000),
                Containment::Contained,
                "the fixed pattern is contained in {name}"
            );
        }
    }
    let traced: Vec<(Vec<Token>, Composition)> = graphs.iter().map(trace_and_composition).collect();
    let trace_refs: Vec<&[Token]> = traced.iter().map(|(t, _)| t.as_slice()).collect();
    let comps: Vec<Composition> = traced.iter().map(|(_, c)| *c).collect();
    let pat_refs: Vec<&[MolGraph]> = fixed.iter().map(|v| v.as_slice()).collect();
    let mut trainer = new_trainer(&device);
    // Isomer groups: the C2 alcohol/ether pair, the C3 triple and the amine pair.
    let groups: Vec<Vec<usize>> = vec![vec![0, 1], vec![2, 3, 4], vec![5, 6]];
    // One batched eval for every (query, pattern-set) pair: a single read
    // (other tests in this binary run concurrently; see
    // `training_step_performs_no_device_read`).
    let batched_nll = |trainer: &mut CompletionTrainer<R, E>| {
        let mut pats = Vec::with_capacity(names.len() * names.len());
        let mut trs = Vec::with_capacity(names.len() * names.len());
        let mut cps = Vec::with_capacity(names.len() * names.len());
        for i in 0..names.len() {
            for j in 0..names.len() {
                pats.push(fixed[j].as_slice());
                trs.push(trace_refs[i]);
                cps.push(comps[i]);
            }
        }
        trainer.teacher_eval_with(&pats, &trs, &cps).unwrap()
    };
    let satisfied = |trainer: &mut CompletionTrainer<R, E>| {
        let nlls = batched_nll(trainer);
        let at = |query: usize, pat: usize| nlls[query * names.len() + pat];
        groups.iter().all(|group| {
            group
                .iter()
                .all(|&i| group.iter().all(|&j| i == j || at(i, i) < at(i, j)))
        })
    };
    for step in 1..=300 {
        trainer.step_with(&pat_refs, &trace_refs, &comps).unwrap();
        if step % 10 == 0 && satisfied(&mut trainer) {
            println!("conditioning satisfied after {step} steps");
            break;
        }
    }
    check_launches(&device).unwrap();
    // The own-versus-other NLL table, from one more batched eval.
    let table = batched_nll(&mut trainer);
    let at = |query: usize, pat: usize| table[query * names.len() + pat];
    println!("own-versus-other NLLs (row: query, col: patterns):");
    for (i, name) in names.iter().enumerate() {
        let row: Vec<f32> = (0..names.len()).map(|j| at(i, j)).collect();
        println!("{name}: {row:?}");
    }
    for group in &groups {
        for &i in group {
            for &j in group {
                if i == j {
                    continue;
                }
                let own = at(i, i);
                let other = at(i, j);
                assert!(
                    own < other,
                    "{}: NLL with own patterns ({own}) is not below NLL with {} patterns ({other})",
                    names[i],
                    names[j]
                );
            }
        }
    }
}

#[test]
fn checkpoint_round_trip() {
    let _lock = serial();
    let device = dev();
    let set = nine_set();
    let indices: Vec<usize> = (0..set.examples.len()).collect();
    let mut trainer = new_trainer(&device);
    for _ in 0..5 {
        trainer.step(&set, &indices, 0).unwrap();
    }
    assert_eq!(trainer.step_count(), 5);
    let before = trainer.teacher_eval(&set, &indices, 0).unwrap();
    let path = std::env::temp_dir().join(format!("mc3a_{}.json", std::process::id()));
    trainer.save(&path).unwrap();
    let header: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    assert_eq!(header["format"], COMPLETION_CHECKPOINT_FORMAT);
    let mut loaded = CompletionTrainer::<R, E>::load(&path, &device).unwrap();
    check_launches(&device).unwrap();
    let after = loaded.teacher_eval(&set, &indices, 0).unwrap();
    assert_eq!(trainer.step_count(), loaded.step_count());
    assert_eq!(before, after, "reloaded NLLs are bit-identical");
    // A header with another format is rejected.
    let tampered_path =
        std::env::temp_dir().join(format!("mc3a_tampered_{}.json", std::process::id()));
    let text = std::fs::read_to_string(&path).unwrap().replacen(
        COMPLETION_CHECKPOINT_FORMAT,
        "bogus-format",
        1,
    );
    std::fs::write(&tampered_path, text).unwrap();
    let err = CompletionTrainer::<R, E>::load(&tampered_path, &device);
    let err = match err {
        Err(e) => e,
        Ok(_) => panic!("a foreign format must be rejected"),
    };
    assert!(
        err.to_string().contains("format"),
        "a foreign format is rejected: {err}"
    );
}

#[test]
fn save_without_parent_dir_fails_and_leaves_good_checkpoint() {
    let _lock = serial();
    // A destination whose parent does not exist cannot host the sibling
    // temporary file: the save errors and a good checkpoint saved earlier at
    // another path is byte-identical afterwards.
    let device = dev();
    let trainer = new_trainer(&device);
    let dir = std::env::temp_dir();
    let good = dir.join(format!("mc11_good_{}.json", std::process::id()));
    trainer.save(&good).unwrap();
    let before = std::fs::read(&good).unwrap();
    let bad = dir.join(format!(
        "mc11_missing_{}/nested.ckpt",
        std::process::id()
    ));
    assert!(!bad.exists());
    let err = trainer.save(&bad).expect_err("a missing parent dir must fail");
    assert!(!err.to_string().is_empty(), "the error says why: {err}");
    assert!(!bad.exists(), "no partial destination appears");
    let after = std::fs::read(&good).unwrap();
    assert_eq!(before, after, "the good checkpoint is untouched");
    std::fs::remove_file(&good).unwrap();
}

#[test]
fn successful_save_leaves_no_temporary_file() {
    let _lock = serial();
    let device = dev();
    let trainer = new_trainer(&device);
    let dir = std::env::temp_dir().join(format!("mc11_clean_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let dest = dir.join("ckpt.json");
    trainer.save(&dest).unwrap();
    let mut names: Vec<String> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    assert_eq!(names, vec!["ckpt.json"], "no temporary sibling remains");
    std::fs::remove_file(&dest).unwrap();
    std::fs::remove_dir(&dir).unwrap();
}

#[test]
fn save_rename_failure_leaves_destination_untouched() {
    let _lock = serial();
    // Pointing the save at an existing non-empty directory makes the final
    // rename fail after the full bytes were written to the sibling: the
    // error surfaces, the directory keeps its sentinel, and no temporary
    // sibling is left behind.
    let device = dev();
    let trainer = new_trainer(&device);
    let parent = std::env::temp_dir().join(format!("mc11_rename_{}", std::process::id()));
    std::fs::create_dir_all(&parent).unwrap();
    let dest = parent.join("dest_dir");
    std::fs::create_dir_all(&dest).unwrap();
    let sentinel = dest.join("sentinel.txt");
    std::fs::write(&sentinel, "sentinel").unwrap();
    let err = trainer
        .save(&dest)
        .expect_err("rename onto a non-empty directory must fail");
    assert!(!err.to_string().is_empty(), "the error says why: {err}");
    assert!(dest.is_dir(), "the destination is still a directory");
    assert_eq!(
        std::fs::read_to_string(&sentinel).unwrap(),
        "sentinel",
        "the destination keeps its contents"
    );
    for entry in std::fs::read_dir(&parent).unwrap() {
        let name = entry.unwrap().file_name().to_string_lossy().into_owned();
        assert!(
            !name.starts_with(".dest_dir.tmp-"),
            "no temporary sibling remains: {name}"
        );
    }
    std::fs::remove_file(&sentinel).unwrap();
    std::fs::remove_dir(&dest).unwrap();
    std::fs::remove_dir(&parent).unwrap();
}

#[test]
fn training_step_performs_no_device_read() {
    let _lock = serial();
    // Without `request_report` a step reads nothing; with it, exactly one
    // read more. Reads have no tally labels, so each probe repeats and takes
    // the minimum: a concurrent test can only add reads, never remove this
    // thread's (see `step_logits_no_read_and_row_independent_launches`).
    // The file-local lock above serialises every test of this binary, yet one
    // run still saw transient extra reads (+1..3 over ~15 consecutive reps,
    // then clean) inflate a whole window: each window is therefore measured
    // up to three times and passes on the first clean one. The assertion per
    // window is unchanged — a step that systematically reads fails every
    // window — so no systematic read can hide behind the retry.
    let device = dev();
    let set = nine_set();
    let indices: Vec<usize> = (0..set.examples.len()).collect();
    let mut trainer = new_trainer(&device);
    for _ in 0..4 {
        trainer.step(&set, &indices, 0).unwrap();
    }
    trainer.request_report();
    trainer.step(&set, &indices, 0).unwrap();
    trainer.request_report();
    trainer.step(&set, &indices, 0).unwrap();
    check_launches(&device).unwrap();
    // Up to three windows of ten plain reps; a window passes when its
    // minimum is 0 (a step without `request_report` reads nothing).
    let mut plain_windows: Vec<Vec<usize>> = Vec::new();
    let mut plain_ok = false;
    for _ in 0..3 {
        let mut window = Vec::new();
        for _ in 0..10 {
            let r0 = runtime_read_count();
            trainer.step(&set, &indices, 0).unwrap();
            window.push(runtime_read_count() - r0);
        }
        if window.iter().min() == Some(&0) {
            plain_ok = true;
        }
        plain_windows.push(window);
        if plain_ok {
            break;
        }
    }
    // Up to three windows of ten reported reps; a window passes when its
    // minimum is 1 (a step with `request_report` reads exactly once).
    let mut reported_windows: Vec<Vec<usize>> = Vec::new();
    let mut reported_ok = false;
    for _ in 0..3 {
        let mut window = Vec::new();
        for _ in 0..10 {
            trainer.request_report();
            let r0 = runtime_read_count();
            let loss = trainer.step(&set, &indices, 0).unwrap();
            window.push(runtime_read_count() - r0);
            assert!(loss.unwrap().is_finite());
        }
        if window.iter().min() == Some(&1) {
            reported_ok = true;
        }
        reported_windows.push(window);
        if reported_ok {
            break;
        }
    }
    check_launches(&device).unwrap();
    println!("plain step read deltas: {plain_windows:?}");
    println!("reported step read deltas: {reported_windows:?}");
    assert!(
        plain_ok,
        "a step without request_report reads nothing: {plain_windows:?}"
    );
    assert!(
        reported_ok,
        "a step with request_report reads exactly once: {reported_windows:?}"
    );
}

// ---------------------------------------------------------------------------
// MC12: formula artifacts bound to the checkpoint.
// ---------------------------------------------------------------------------

use mamba3::models::ms2::completion_model::FormulaArtifacts;

#[test]
fn checkpoint_with_and_without_artifacts_round_trips() {
    let _lock = serial();
    let device = dev();
    let set = nine_set();
    let indices: Vec<usize> = (0..set.examples.len()).collect();
    let mut trainer = new_trainer(&device);
    for _ in 0..3 {
        trainer.step(&set, &indices, 0).unwrap();
    }
    // Without artifacts.
    let path_none = std::env::temp_dir().join(format!("mc12_none_{}.json", std::process::id()));
    trainer.save(&path_none).unwrap();
    let loaded_none = CompletionTrainer::<R, E>::load(&path_none, &device).unwrap();
    assert!(loaded_none.formula_artifacts().is_none());
    assert_eq!(trainer.step_count(), loaded_none.step_count());
    // With artifacts fitted on the nine molecules.
    let comps: Vec<mamba3::models::ms2::Composition> =
        set.examples.iter().map(|e| e.composition).collect();
    let artifacts = FormulaArtifacts::fit(&comps, 16, 0, 0, "test:nine".to_string()).unwrap();
    trainer.set_formula_artifacts(artifacts.clone());
    let path_some = std::env::temp_dir().join(format!("mc12_some_{}.json", std::process::id()));
    trainer.save(&path_some).unwrap();
    let loaded_some = CompletionTrainer::<R, E>::load(&path_some, &device).unwrap();
    assert_eq!(loaded_some.formula_artifacts(), Some(&artifacts));
    // Header carries the block with hashes.
    let text = std::fs::read_to_string(&path_some).unwrap();
    let header: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert!(header.get("formula_artifacts").is_some());
    assert_eq!(
        header["formula_artifacts"]["fit"]["molecules"],
        serde_json::json!(9)
    );
}

#[test]
fn old_checkpoint_without_block_still_loads() {
    let _lock = serial();
    // A header without `formula_artifacts` (the pre-task shape) loads with
    // `None`: craft one by stripping the block from a no-artifact save.
    let device = dev();
    let trainer = new_trainer(&device);
    let path = std::env::temp_dir().join(format!("mc12_old_{}.json", std::process::id()));
    trainer.save(&path).unwrap();
    let text = std::fs::read_to_string(&path).unwrap();
    let mut header: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert!(header.get("formula_artifacts").is_none());
    // Explicitly ensure absence still parses (serde default).
    if let Some(obj) = header.as_object_mut() {
        obj.remove("formula_artifacts");
    }
    let stripped = std::env::temp_dir().join(format!("mc12_old_stripped_{}.json", std::process::id()));
    std::fs::write(&stripped, serde_json::to_string_pretty(&header).unwrap()).unwrap();
    let loaded = CompletionTrainer::<R, E>::load(&stripped, &device).unwrap();
    assert!(loaded.formula_artifacts().is_none());
}

#[test]
fn attach_leaves_weights_byte_identical() {
    let _lock = serial();
    // Through the library function the driver uses (`fit_and_attach`).
    let device = dev();
    let set = nine_set();
    let indices: Vec<usize> = (0..set.examples.len()).collect();
    let mut trainer = new_trainer(&device);
    for _ in 0..3 {
        trainer.step(&set, &indices, 0).unwrap();
    }
    let path_before =
        std::env::temp_dir().join(format!("mc12_attach_before_{}.json", std::process::id()));
    trainer.save(&path_before).unwrap();
    let before_text = std::fs::read_to_string(&path_before).unwrap();
    let before_json: serde_json::Value = serde_json::from_str(&before_text).unwrap();
    let comps: Vec<mamba3::models::ms2::Composition> =
        set.examples.iter().map(|e| e.composition).collect();
    mamba3::models::ms2::completion_experiment::fit_and_attach(
        &mut trainer,
        &comps,
        16,
        "test:attach".to_string(),
    )
    .unwrap();
    let path_after =
        std::env::temp_dir().join(format!("mc12_attach_after_{}.json", std::process::id()));
    trainer.save(&path_after).unwrap();
    let after_text = std::fs::read_to_string(&path_after).unwrap();
    let after_json: serde_json::Value = serde_json::from_str(&after_text).unwrap();
    assert_eq!(
        before_json.get("weights"),
        after_json.get("weights"),
        "weights byte-identical after attach"
    );
    assert!(after_json.get("formula_artifacts").is_some());
}

// ---------------------------------------------------------------------------
// MC15: functional-group pattern source.
// ---------------------------------------------------------------------------

#[test]
fn tiny_checkpoint_without_pattern_source_loads_as_random() {
    let _lock = serial();
    let device = dev();
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/ms2/completion_tiny.ckpt");
    let text = std::fs::read_to_string(&path).unwrap();
    let mut header: serde_json::Value = serde_json::from_str(&text).unwrap();
    // Recover the pre-task header shape by stripping the block the current
    // writer emits (the committed fixture itself now carries it): an old
    // checkpoint without `pattern_source` must still load, as RandomPatches.
    header["train_config"]
        .as_object_mut()
        .expect("train_config is an object")
        .remove("pattern_source");
    assert!(
        header["train_config"].get("pattern_source").is_none(),
        "the stripped header has no pattern_source"
    );
    let stripped = std::env::temp_dir().join(format!("mc15_tiny_stripped_{}.json", std::process::id()));
    std::fs::write(&stripped, serde_json::to_string_pretty(&header).unwrap()).unwrap();
    let trainer = CompletionTrainer::<R, E>::load(&stripped, &device).unwrap();
    // Access through a save round-trip header.
    let tmp = std::env::temp_dir().join(format!("mc15_tiny_{}.json", std::process::id()));
    trainer.save(&tmp).unwrap();
    let saved: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&tmp).unwrap()).unwrap();
    assert!(saved["train_config"].get("pattern_source").is_some());
    assert!(
        saved["train_config"]["pattern_source"].get("random_patches").is_some(),
        "old checkpoint loads as RandomPatches"
    );
    check_launches(&device).unwrap();
}

#[test]
fn functional_groups_config_round_trips() {
    let _lock = serial();
    let device = dev();
    let fg = mamba3::models::ms2::completion_data::FunctionalGroupConfig::default();
    let mut cfg = train_config();
    cfg.pattern_source =
        mamba3::models::ms2::completion_data::PatternSource::FunctionalGroups(fg);
    let mut trainer: CompletionTrainer<R, E> =
        CompletionTrainer::new(&CompletionModelConfig::small(), &cfg, &device).unwrap();
    let set = nine_set();
    let indices: Vec<usize> = (0..set.examples.len()).collect();
    trainer.step(&set, &indices, 0).unwrap();
    let path = std::env::temp_dir().join(format!("mc15_fg_{}.json", std::process::id()));
    trainer.save(&path).unwrap();
    let loaded = CompletionTrainer::<R, E>::load(&path, &device).unwrap();
    assert_eq!(
        loaded
            .model()
            .config
            .max_atoms,
        CompletionModelConfig::small().max_atoms
    );
    let text = std::fs::read_to_string(&path).unwrap();
    let header: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(
        header["train_config"]["pattern_source"]["functional_groups"]["max_groups"],
        serde_json::json!(8)
    );
    check_launches(&device).unwrap();
}

#[test]
fn functional_group_training_step_runs_and_reduces_loss() {
    let _lock = serial();
    let device = dev();
    let set = nine_set();
    let indices: Vec<usize> = (0..set.examples.len()).collect();
    let mut cfg = train_config();
    cfg.pattern_source = mamba3::models::ms2::completion_data::PatternSource::FunctionalGroups(
        mamba3::models::ms2::completion_data::FunctionalGroupConfig::default(),
    );
    let mut trainer: CompletionTrainer<R, E> =
        CompletionTrainer::new(&CompletionModelConfig::small(), &cfg, &device).unwrap();
    trainer.request_report();
    let initial = trainer.step(&set, &indices, 0).unwrap().unwrap();
    assert!(initial.is_finite());
    let mut loss = initial;
    for _ in 0..20 {
        trainer.request_report();
        loss = trainer.step(&set, &indices, 0).unwrap().unwrap();
    }
    check_launches(&device).unwrap();
    assert!(loss.is_finite(), "final loss finite");
    assert!(loss < initial, "loss reduces: {initial} -> {loss}");
}

// ---------------------------------------------------------------------------
// MC12b: training composition counts in the formula artifacts.
// ---------------------------------------------------------------------------

#[test]
fn artifacts_carry_sorted_training_composition_counts() {
    let _lock = serial();
    let set = nine_set();
    let comps: Vec<mamba3::models::ms2::Composition> =
        set.examples.iter().map(|e| e.composition).collect();
    let artifacts =
        FormulaArtifacts::fit(&comps, 16, 0, 0, "test:nine".to_string()).unwrap();
    // One entry per distinct training composition, sorted by formula text.
    let texts: Vec<&str> = artifacts
        .composition_counts
        .iter()
        .map(|entry| entry.formula.as_str())
        .collect();
    assert_eq!(texts, vec!["C2H6O", "C2H7N", "C3H6", "C3H8O", "C7H8"]);
    let counts: Vec<u64> = artifacts
        .composition_counts
        .iter()
        .map(|entry| entry.count)
        .collect();
    assert_eq!(counts, vec![2, 2, 1, 3, 1]);
    assert_eq!(
        counts.iter().sum::<u64>(),
        9,
        "counts sum to the training molecules"
    );
    // Binary-search lookup: seen formulas hit, unseen ones count 0.
    assert_eq!(artifacts.composition_count("C2H6O"), 2);
    assert_eq!(artifacts.composition_count("C3H8O"), 3);
    assert_eq!(artifacts.composition_count("C6H12O6"), 0);
    // Round trip through save/load keeps the counts.
    let device = dev();
    let mut trainer = new_trainer(&device);
    trainer.set_formula_artifacts(artifacts);
    let path = std::env::temp_dir().join(format!("mc12b_counts_{}.json", std::process::id()));
    trainer.save(&path).unwrap();
    let loaded = CompletionTrainer::<R, E>::load(&path, &device).unwrap();
    let back = loaded.formula_artifacts().expect("counts survive");
    assert_eq!(back.composition_count("C2H6O"), 2);
    assert_eq!(back.composition_counts.len(), 5);
}

#[test]
fn artifact_text_matches_formula_text() {
    let _lock = serial();
    // The artifact renderer must match completion_formula::formula_text
    // exactly (the two cannot share code: the formula module depends on the
    // model module).
    let set = nine_set();
    let comps: Vec<mamba3::models::ms2::Composition> =
        set.examples.iter().map(|e| e.composition).collect();
    let artifacts =
        FormulaArtifacts::fit(&comps, 16, 0, 0, "test:text".to_string()).unwrap();
    for entry in &artifacts.composition_counts {
        let comp = comps
            .iter()
            .find(|c| {
                mamba3::models::ms2::completion_formula::formula_text(c) == entry.formula
            })
            .expect("every artifact text renders from a training composition");
        assert_eq!(
            mamba3::models::ms2::completion_formula::formula_text(comp),
            entry.formula
        );
    }
    // Edge shapes render identically on both sides.
    let edges: Vec<mamba3::models::ms2::Composition> = vec![
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [12, 26, 0, 0, 0, 0, 0, 0, 0, 0],
    ];
    let edge_artifacts =
        FormulaArtifacts::fit(&edges, 32, 0, 0, "test:edges".to_string()).unwrap();
    for entry in &edge_artifacts.composition_counts {
        let comp = edges
            .iter()
            .find(|c| {
                mamba3::models::ms2::completion_formula::formula_text(c) == entry.formula
            })
            .expect("edge text renders");
        assert_eq!(
            mamba3::models::ms2::completion_formula::formula_text(comp),
            entry.formula
        );
    }
}

#[test]
fn tiny_fixture_checkpoint_stays_under_300kb() {
    let _lock = serial();
    // The committed tiny fixture (with its composition counts) must stay
    // under 300 KB.
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/ms2/completion_tiny.ckpt");
    let bytes = std::fs::read(&path).unwrap();
    assert!(
        bytes.len() < 300 * 1024,
        "tiny fixture is {} bytes, past 300 KB",
        bytes.len()
    );
    let header: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let counts = header["formula_artifacts"]["composition_counts"]
        .as_array()
        .expect("fixture carries composition counts");
    assert!(!counts.is_empty(), "the counts list is populated");
}

#[test]
fn resumed_learning_rate_controls_updates_and_saved_config() {
    let _lock = serial();
    let device = dev();
    let set = nine_set();
    let indices = [0, 1];
    let path = std::env::temp_dir().join(format!("mc_lr_{}.json", std::process::id()));
    let mut initial = new_trainer(&device);
    initial.step(&set, &indices, 0).unwrap();
    initial.save(&path).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    let mut header: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let lr = 1e-4f32;
    header["train_config"]["lr"] = serde_json::json!(lr);
    let mut expected = CompletionTrainer::<R, E>::load_bytes(
        &serde_json::to_vec(&header).unwrap(), &device,
    ).unwrap();
    let mut actual = CompletionTrainer::<R, E>::load_bytes(&bytes, &device).unwrap();
    actual.set_learning_rate(lr).unwrap();
    assert_eq!(actual.step_count(), initial.step_count());
    for invalid in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        assert!(actual.set_learning_rate(invalid).is_err());
        assert_eq!(actual.train_config().lr, lr);
    }
    actual.step(&set, &indices, 1).unwrap();
    expected.step(&set, &indices, 1).unwrap();
    initial.step(&set, &indices, 1).unwrap();
    let actual_nll = actual.teacher_eval(&set, &indices, 0).unwrap();
    let expected_nll = expected.teacher_eval(&set, &indices, 0).unwrap();
    let original_nll = initial.teacher_eval(&set, &indices, 0).unwrap();
    check_launches(&device).unwrap();
    assert_close(&actual_nll, &expected_nll, 1e-5, "explicit LR applies to optimizer");
    assert!(actual_nll.iter().zip(&original_nll).any(|(a, b)| (a-b).abs() > 1e-4));
    actual.save(&path).unwrap();
    let loaded = CompletionTrainer::<R, E>::load(&path, &device).unwrap();
    assert_eq!(loaded.train_config().lr, lr);
    assert_eq!(loaded.step_count(), actual.step_count());
    std::fs::remove_file(path).unwrap();
}
