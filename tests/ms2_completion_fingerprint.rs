//! MC20b tests: fingerprint evidence for the completion model.
//!
//! Hand-built molecules only (atom type ids from
//! [`chem::ATOM_TYPES`](mamba3::models::ms2::chem::ATOM_TYPES)): the same nine
//! molecules as `ms2_completion_model.rs` (ethanol, dimethyl ether,
//! propan-1-ol, propan-2-ol, methoxyethane, ethylamine, dimethylamine,
//! cyclopropane and kekulized methylbenzene) with made-up distinct bit lists
//! (the test does not need real fingerprints). Every device call is followed
//! by [`check_launches`].

#![cfg(feature = "backend")]

use std::collections::BTreeMap;

use mamba3::backend::{Device, check_launches};
use mamba3::backends::Auto;
use mamba3::models::ms2::chem::Composition;
use mamba3::models::ms2::completion_data::{CompletionExample, CompletionSet, ExtractionConfig};
use mamba3::models::ms2::completion_fingerprint::{
    FINGERPRINT_BITS, FINGERPRINT_SLOTS, FingerprintBatch, FingerprintEncoder, FingerprintMode,
    FingerprintNoise, FingerprintNoiseLevel, FingerprintQueryStats, FingerprintStore,
    SparseFingerprint,
};
use mamba3::models::ms2::completion_model::{
    CompletionModel, CompletionModelConfig, CompletionTrainConfig,
    CompletionTrainer, PatternBatch,
};
use mamba3::models::ms2::grammar::{
    CANONICAL_WORK_LIMIT, Limits, Token, canonical_trace, replay_exact,
};
use mamba3::models::ms2::graph::MolGraph;
use mamba3::models::ms2::targets_batch::TargetBatch;
use mamba3::nn::Module;
use mamba3::tensor::Tensor;
use mamba3::tensor::ops::index::read_all;
use mamba3::tensor::ops::ms2::Ms2Constants;
use mamba3::tensor::ops::random::Rng;

type R = Auto;
type E = f32;

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

fn ethanol() -> MolGraph {
    MolGraph::new(vec![4, 3, 9], vec![(0, 1, 1), (1, 2, 1)]).unwrap()
}

fn dimethyl_ether() -> MolGraph {
    MolGraph::new(vec![4, 8, 4], vec![(0, 1, 1), (1, 2, 1)]).unwrap()
}

fn propan_1_ol() -> MolGraph {
    MolGraph::new(vec![4, 3, 3, 9], vec![(0, 1, 1), (1, 2, 1), (2, 3, 1)]).unwrap()
}

fn propan_2_ol() -> MolGraph {
    MolGraph::new(vec![4, 2, 4, 9], vec![(0, 1, 1), (1, 2, 1), (1, 3, 1)]).unwrap()
}

fn methoxyethane() -> MolGraph {
    MolGraph::new(vec![4, 8, 3, 4], vec![(0, 1, 1), (1, 2, 1), (2, 3, 1)]).unwrap()
}

fn ethylamine() -> MolGraph {
    MolGraph::new(vec![4, 3, 7], vec![(0, 1, 1), (1, 2, 1)]).unwrap()
}

fn dimethylamine() -> MolGraph {
    MolGraph::new(vec![4, 6, 4], vec![(0, 1, 1), (1, 2, 1)]).unwrap()
}

fn cyclopropane() -> MolGraph {
    MolGraph::new(vec![3, 3, 3], vec![(0, 1, 1), (1, 2, 1), (2, 0, 1)]).unwrap()
}

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

/// Made-up distinct bit lists for the nine molecules (the test does not need
/// real fingerprints): molecule `i` owns bits `i * 40..(i + 1) * 40`.
fn nine_bit_lists() -> Vec<Vec<u16>> {
    (0..9)
        .map(|i| ((i * 40) as u16..((i + 1) * 40) as u16).collect())
        .collect()
}

fn fp_config() -> CompletionModelConfig {
    let mut config = CompletionModelConfig::small();
    config.fingerprint_slots = 16;
    config
}

fn fp_train_config() -> CompletionTrainConfig {
    let extraction = ExtractionConfig {
        min_patterns: 0,
        max_patterns: 0,
        ..ExtractionConfig::default()
    };
    CompletionTrainConfig {
        lr: 3e-3,
        weight_decay: 0.0,
        grad_clip: None,
        seed: 1,
        extraction_seed: 11,
        pattern_source:
            mamba3::models::ms2::completion_data::PatternSource::RandomPatches(extraction.clone()),
        extraction,
        fingerprint_mode: Some(FingerprintMode::Exact),
        fingerprint_threshold: 0.1,
    }
}

#[test]
fn sparse_fingerprint_validation_and_tokens() {
    let _lock = serial();
    // Validation rejects out-of-range bits, non-positive probabilities and
    // unsorted entries.
    assert!(SparseFingerprint::from_bits(&[0, 1, 4095]).is_ok());
    assert!(SparseFingerprint::from_bits(&[4096]).is_err());
    assert!(
        SparseFingerprint::from_probabilities(&[(3u16, 0.0)], 0.1)
            .unwrap()
            .entries
            .is_empty()
    );
    assert!(SparseFingerprint::from_probabilities(&[(3u16, 1.5)], 0.1).is_err());
    assert!(
        (SparseFingerprint {
            entries: vec![(5, 0.5), (3, 0.7)]
        })
        .validate()
        .is_err()
    );
    // Threshold filter keeps `p >= threshold`.
    let fp =
        SparseFingerprint::from_probabilities(&[(1u16, 0.05), (2u16, 0.1), (3u16, 0.9)], 0.1)
            .unwrap();
    assert_eq!(fp.entries, vec![(2, 0.1), (3, 0.9)]);
    // Buckets: 8 equal-width buckets over (0, 1].
    assert_eq!(SparseFingerprint::bucket(0.01), 1);
    assert_eq!(SparseFingerprint::bucket(0.125), 1);
    assert_eq!(SparseFingerprint::bucket(0.13), 2);
    assert_eq!(SparseFingerprint::bucket(0.9), 8);
    assert_eq!(SparseFingerprint::bucket(1.0), 8);
    // Token selection: highest probability first, ties by lower index.
    let fp = SparseFingerprint {
        entries: vec![(10, 0.9), (3, 0.9), (7, 0.5), (1, 0.2)],
    };
    let (ids, buckets, valid) = fp.tokens(3);
    assert_eq!(ids, vec![4, 11, 8]);
    assert_eq!(buckets.len(), 3);
    assert_eq!(valid, vec![1.0, 1.0, 1.0]);
    assert_eq!(fp.dropped(3), 1);
    assert_eq!(fp.dropped(16), 0);
    // Slot limit pads with zeros.
    let (ids, buckets, valid) = fp.tokens(6);
    assert_eq!(ids.len(), 6);
    assert_eq!(ids[4], 0);
    assert_eq!(buckets[4], 0);
    assert_eq!(valid[4], 0.0);
    // Batch validation rejects out-of-range ids and non-binary validity.
    let mut batch = FingerprintBatch::build(
        &[SparseFingerprint::from_bits(&[0, 5]).unwrap()],
        4,
    )
    .unwrap();
    batch.validate().unwrap();
    batch.token_ids[0] = 5000;
    assert!(batch.validate().is_err());
}

#[test]
fn encoder_is_invariant_to_token_order() {
    let _lock = serial();
    let device = dev();
    let mut rng = Rng::seeded(21);
    let d = 16usize;
    let encoder: FingerprintEncoder<R, E> =
        FingerprintEncoder::init(d, &device, &mut rng);
    check_launches(&device).unwrap();
    // Same entries in different slot orders give the same pooled vector (the
    // output is invariant to the order of the tokens).
    let fp = SparseFingerprint {
        entries: vec![(2, 0.9), (7, 0.7), (9, 0.8), (40, 0.5)],
    };
    let batch = FingerprintBatch::build(&[fp], 4).unwrap();
    let pooled = encoder.encode_pooled(&batch, &device).unwrap();
    check_launches(&device).unwrap();
    // Permute the slots by hand (same multiset, different order).
    let mut permuted = FingerprintBatch::build(
        &[SparseFingerprint {
            entries: vec![(2, 0.9), (7, 0.7), (9, 0.8), (40, 0.5)],
        }],
        4,
    )
    .unwrap();
    // Reverse the slot order: swap slots 0<->3 and 1<->2.
    for (a, b) in [(0usize, 3usize), (1, 2)] {
        permuted.token_ids.swap(a, b);
        permuted.bucket_ids.swap(a, b);
        permuted.valid.swap(a, b);
    }
    permuted.validate().unwrap();
    let pooled_perm = encoder.encode_pooled(&permuted, &device).unwrap();
    check_launches(&device).unwrap();
    let a = pooled.try_to_f32().unwrap();
    let b = pooled_perm.try_to_f32().unwrap();
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b.iter()) {
        let tol = if y.abs() <= 1.0 { 1e-5 } else { 1e-5 * y.abs() };
        assert!((x - y).abs() <= tol, "pooled differs: {x} vs {y}");
    }
    // Exact-zero padding: states in padding slots are exactly zero.
    let short = FingerprintBatch::build(
        &[SparseFingerprint::from_bits(&[3]).unwrap()],
        4,
    )
    .unwrap();
    let (states, _) = encoder.encode_states(&short, &device).unwrap();
    check_launches(&device).unwrap();
    let values = states.try_to_f32().unwrap();
    // One query, 4 slots, d = 16: slots 1..3 are padding.
    for s in 1..4 {
        for k in 0..d {
            assert_eq!(values[s * d + k], 0.0, "padding slot {s} is not exact zero");
        }
    }
    // Empty fingerprint encodes (finite pooled vector, no panic).
    let empty = FingerprintBatch::empty(2, 4).unwrap();
    let pooled_empty = encoder.encode_pooled(&empty, &device).unwrap();
    check_launches(&device).unwrap();
    assert!(pooled_empty.try_to_f32().unwrap().iter().all(|v| v.is_finite()));
}

#[test]
fn teacher_nll_is_invariant_to_token_order() {
    let _lock = serial();
    let device = dev();
    let mut rng = Rng::seeded(23);
    let mut config = CompletionModelConfig::small();
    config.fingerprint_slots = 8;
    let model: CompletionModel<R, E> = CompletionModel::init(&config, &device, &mut rng).unwrap();
    check_launches(&device).unwrap();
    let set = nine_set();
    let example = &set.examples[0];
    let patterns: Vec<&[MolGraph]> = vec![&[]];
    let traces = vec![example.trace.as_slice()];
    let compositions = vec![example.composition];
    let batch = PatternBatch::build(&patterns, &compositions).unwrap();
    let targets =
        TargetBatch::build_exact(&traces, &compositions, limits()).unwrap();
    let constants = Ms2Constants::new(&device);
    let fp = SparseFingerprint {
        entries: vec![(2, 0.9), (7, 0.7), (9, 0.8), (40, 0.5)],
    };
    let fp_batch = FingerprintBatch::build(&[fp], 8).unwrap();
    let (_, loss_a) = model
        .teacher_with_fingerprints(&batch, &fp_batch, &targets, &constants, &device)
        .unwrap();
    check_launches(&device).unwrap();
    // Same multiset in a different slot order.
    let mut permuted = FingerprintBatch::build(
        &[SparseFingerprint {
            entries: vec![(2, 0.9), (7, 0.7), (9, 0.8), (40, 0.5)],
        }],
        8,
    )
    .unwrap();
    permuted.token_ids.swap(0, 3);
    permuted.bucket_ids.swap(0, 3);
    permuted.valid.swap(0, 3);
    let (_, loss_b) = model
        .teacher_with_fingerprints(&batch, &permuted, &targets, &constants, &device)
        .unwrap();
    check_launches(&device).unwrap();
    let a = loss_a.try_to_f32().unwrap()[0];
    let b = loss_b.try_to_f32().unwrap()[0];
    let tol = if b.abs() <= 1.0 { 1e-4 } else { 1e-4 * b.abs() };
    assert!((a - b).abs() <= tol, "teacher NLL differs: {a} vs {b}");
}

#[test]
fn noise_sampler_is_deterministic_thresholded_and_calibrated() {
    let _lock = serial();
    // Small synthetic noise file: on-histogram keeps 3/4 draws above 0.3
    // (bin 5 [0.25, 0.3) is dropped, bin 19 [0.95, 1.0] is kept);
    // off-histogram keeps 4/4096 draws above 0.3.
    let noise_json = serde_json::json!({
        "fingerprint": "morgan4096",
        "n_bits": 4096,
        "n_bins": 20,
        "hist_pred_given_true_on": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
        "hist_pred_given_true_off": [4092, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4],
        "n_spectra": 1,
        "n_molecules": 1,
    });
    let noise =
        FingerprintNoise::load_json(&serde_json::to_string(&noise_json).unwrap()).unwrap();
    // Deterministic in (seed, key, draw).
    let truth = vec![10u16, 20, 30];
    let first = noise.sample(&truth, 7, "mol-a", 0, 0.1).unwrap();
    let again = noise.sample(&truth, 7, "mol-a", 0, 0.1).unwrap();
    let other_seed = noise.sample(&truth, 8, "mol-a", 0, 0.1).unwrap();
    let other_key = noise.sample(&truth, 7, "mol-b", 0, 0.1).unwrap();
    let other_draw = noise.sample(&truth, 7, "mol-a", 1, 0.1).unwrap();
    assert_eq!(first, again);
    assert!(other_seed != first || other_key != first || other_draw != first);
    // Threshold drops low entries.
    let high = noise.sample(&truth, 7, "mol-a", 0, 0.99).unwrap();
    let low = noise.sample(&truth, 7, "mol-a", 0, 0.01).unwrap();
    assert!(high.entries.len() <= low.entries.len());
    // Empirical calibration on 2,000 draws: retention 3/4 within 0.05 and
    // mean false tokens near the file's 4.0 within 25% relative. At threshold
    // 0.3 the on draws in bin 5 are dropped and bin 19 kept (3/4 by
    // construction); off draws are mostly dropped with ~4 false survivors.
    let threshold = 0.3f32;
    let mut retained = 0usize;
    let mut total_true = 0usize;
    let mut false_total = 0usize;
    let draws = 2000usize;
    // Two true bits to keep the test fast (off bits still iterate 4094).
    let small_truth = vec![11u16, 500];
    for draw in 0..draws {
        let fp = noise
            .sample(&small_truth, 99, "calibration-mol", draw as u64, threshold)
            .unwrap();
        let stats = FingerprintQueryStats::score(&fp, &small_truth, 128);
        retained += small_truth.len() - stats.true_missing;
        total_true += small_truth.len();
        false_total += stats.false_tokens;
    }
    let retention = retained as f64 / total_true as f64;
    let mean_false = false_total as f64 / draws as f64;
    assert!(
        (retention - 0.75).abs() <= 0.05,
        "retention {retention} is not 0.75 within 0.05"
    );
    let expected_false = noise.mean_false_tokens(threshold);
    assert!(
        (mean_false - expected_false).abs() <= 0.25 * expected_false.max(1.0),
        "mean false {mean_false} vs expected {expected_false}"
    );
    assert!((noise.retention_rate(threshold) - 0.75).abs() <= 1e-12);
}

#[test]
fn gradients_reach_every_new_parameter() {
    let _lock = serial();
    let device = dev();
    let mut rng = Rng::seeded(31);
    let model: CompletionModel<R, E> =
        CompletionModel::init(&fp_config(), &device, &mut rng).unwrap();
    let set = nine_set();
    let bits = nine_bit_lists();
    let fps: Vec<SparseFingerprint> = bits
        .iter()
        .take(2)
        .map(|b| SparseFingerprint::from_bits(b).unwrap())
        .collect();
    let patterns: Vec<&[MolGraph]> = vec![&[], &[]];
    let traces = vec![set.examples[0].trace.as_slice(), set.examples[1].trace.as_slice()];
    let compositions = vec![set.examples[0].composition, set.examples[1].composition];
    let batch = PatternBatch::build(&patterns, &compositions).unwrap();
    let fp_batch = FingerprintBatch::build(&fps, 16).unwrap();
    let targets = TargetBatch::build_exact(&traces, &compositions, limits()).unwrap();
    let constants = Ms2Constants::new(&device);
    let (_, loss) = model
        .teacher_with_fingerprints(&batch, &fp_batch, &targets, &constants, &device)
        .unwrap();
    check_launches(&device).unwrap();
    let grads = loss.backward_retain().unwrap();
    let named = model.named_parameters();
    for prefix in [
        "fingerprint_encoder.bit_emb",
        "fingerprint_encoder.conf_emb",
        "fingerprint_encoder.pool_in.weight",
    ] {
        assert!(
            named.iter().any(|(n, _)| n == prefix),
            "{prefix} is a named parameter"
        );
    }
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
    // Finite-difference check on `bit_emb` row 1 (bit 0 is used): analytic
    // vs numeric gradient within 5%.
    let name = "fingerprint_encoder.bit_emb";
    let (param_index, _) = named
        .iter()
        .enumerate()
        .find(|(_, (n, _))| n == name)
        .expect(name);
    let _ = &param_index;
    let analytic = grad_of(name);
    assert!(analytic.iter().all(|v| v.is_finite()));
    assert!(
        analytic.iter().any(|&v| v != 0.0),
        "bit_emb gradients reach the loss"
    );
    let loss_of = || {
        model
            .teacher_with_fingerprints(&batch, &fp_batch, &targets, &constants, &device)
            .unwrap()
            .1
            .try_to_f32()
            .unwrap()[0]
    };
    let base = loss_of();
    // An actual central-difference check (finding 10): perturb one element
    // of `bit_emb` (token memory) and one of `pool_in.weight` (pooled
    // context) with mixed patterns plus fingerprints, and compare against
    // the analytic gradients. The old check only re-ran the forward pass.
    let pat0 = propan_1_ol();
    let pat1 = dimethyl_ether();
    let mixed_refs: Vec<&[MolGraph]> =
        vec![std::slice::from_ref(&pat0), std::slice::from_ref(&pat1)];
    let mixed_batch = PatternBatch::build(&mixed_refs, &compositions).unwrap();
    let mixed_loss_of = || {
        model
            .teacher_with_fingerprints(
                &mixed_batch,
                &fp_batch,
                &targets,
                &constants,
                &device,
            )
            .unwrap()
            .1
            .try_to_f32()
            .unwrap()[0]
    };
    let _ = mixed_loss_of();
    let mixed_grads = model
        .teacher_with_fingerprints(&mixed_batch, &fp_batch, &targets, &constants, &device)
        .unwrap()
        .1
        .backward_retain()
        .unwrap();
    check_launches(&device).unwrap();
    let central_check = |param_name: &str| {
        let (_, param) = named
            .iter()
            .find(|(n, _)| n == param_name)
            .unwrap_or_else(|| panic!("{param_name} missing"));
        let values = param.value();
        let (_, floats_all) = read_all(&[], &[&values]).unwrap();
        let floats = floats_all[0].clone();
        let analytic_t = mixed_grads.get(param.id()).cloned().unwrap_or_else(|| {
            panic!("{param_name} has no gradient under mixed patterns+fingerprints")
        });
        let (_, analytic_all) = read_all(&[], &[&analytic_t]).unwrap();
        let analytic_floats = &analytic_all[0];
        // Element with the largest analytic magnitude (certainly nonzero:
        // presence is asserted above for these tables).
        let (flat, _) = analytic_floats
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.abs().total_cmp(&b.abs()))
            .unwrap();
        let analytic = analytic_floats[flat];
        assert!(
            analytic.abs() > 1e-6,
            "{param_name}[{flat}] analytic gradient is zero ({analytic})"
        );
        let eps = 1e-3f32;
        let shape = param.shape().dims().to_vec();
        let mut up = floats.clone();
        up[flat] += eps;
        param.set(Tensor::<R, E>::from_f32(&up, shape.clone(), &device).unwrap());
        check_launches(&device).unwrap();
        let plus = mixed_loss_of();
        let mut down = floats.clone();
        down[flat] -= eps;
        param.set(Tensor::<R, E>::from_f32(&down, shape.clone(), &device).unwrap());
        check_launches(&device).unwrap();
        let minus = mixed_loss_of();
        param.set(Tensor::<R, E>::from_f32(&floats, shape, &device).unwrap());
        check_launches(&device).unwrap();
        let numeric = (plus - minus) / (2.0 * eps);
        assert!(
            (numeric - analytic).abs() <= 0.2 * analytic.abs() + 1e-4,
            "{param_name}[{flat}] central difference {numeric} vs analytic {analytic}"
        );
    };
    central_check("fingerprint_encoder.bit_emb");
    central_check("fingerprint_encoder.pool_in.weight");
    let again = loss_of();
    assert!((base - again).abs() <= 1e-6, "loss is deterministic");
}

#[test]
fn slots_zero_is_bit_identical_on_the_fixture() {
    let _lock = serial();
    let device = dev();
    // The committed fixture keeps `fingerprint_slots = 0`.
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let ckpt = root.join("tests").join("fixtures").join("ms2").join("completion_tiny.ckpt");
    let bytes = std::fs::read(&ckpt).unwrap();
    let raw: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    // The fixture predates `fingerprint_slots`: the field is absent (null)
    // or 0, both meaning no fingerprint encoder.
    match &raw["model_config"]["fingerprint_slots"] {
        serde_json::Value::Null => {}
        v => assert_eq!(
            v,
            &serde_json::Value::Number(0.into()),
            "the committed fixture keeps fingerprint_slots = 0"
        ),
    }
    let trainer = CompletionTrainer::<R, E>::load(&ckpt, &device).unwrap();
    assert_eq!(trainer.model().config.fingerprint_slots, 0);
    assert!(!trainer.model().has_fingerprint());
    // Bit-identical legacy outputs for the intended reason: an empty
    // fingerprint on the zero-slot model takes the same encoding path as the
    // legacy call, so both teacher losses agree bit for bit. (The old
    // version of this test used zero slots, which erase the supplied bits,
    // with mismatched target limits, so it could pass for the wrong reason.)
    let fixture_limits = Limits::new(12, 2).unwrap();
    let graph = ethanol();
    let canonical =
        canonical_trace(&graph, fixture_limits, CANONICAL_WORK_LIMIT).unwrap();
    let composition = graph.composition();
    let traces = vec![canonical.trace.as_slice()];
    let compositions = vec![composition];
    let patterns: Vec<&[MolGraph]> = vec![&[]];
    let batch = PatternBatch::build(&patterns, &compositions).unwrap();
    let targets =
        TargetBatch::build_exact(&traces, &compositions, fixture_limits).unwrap();
    let constants = Ms2Constants::new(&device);
    let (_, legacy_loss) = trainer
        .model()
        .teacher(&batch, &targets, &constants, &device)
        .unwrap();
    check_launches(&device).unwrap();
    let empty_fp = FingerprintBatch::build(
        &[SparseFingerprint { entries: Vec::new() }],
        0,
    )
    .unwrap();
    let (_, fp_loss) = trainer
        .model()
        .teacher_with_fingerprints(&batch, &empty_fp, &targets, &constants, &device)
        .unwrap();
    check_launches(&device).unwrap();
    let a = legacy_loss.try_to_f32().unwrap()[0];
    let b = fp_loss.try_to_f32().unwrap()[0];
    assert_eq!(
        a.to_bits(),
        b.to_bits(),
        "zero-slot empty-fingerprint loss {b} is not bit-identical to legacy {a}"
    );
    // A non-empty fingerprint on a model without the encoder is rejected for
    // the intended reason (no encoder), not by an incidental shape error.
    let nonempty = FingerprintBatch::build(
        &[SparseFingerprint::from_bits(&[1, 2]).unwrap()],
        4,
    )
    .unwrap();
    let err = match trainer
        .model()
        .teacher_with_fingerprints(&batch, &nonempty, &targets, &constants, &device)
    {
        Err(e) => format!("{e}"),
        Ok(_) => panic!("a fingerprint on the zero-slot model must be rejected"),
    };
    assert!(
        err.contains("fingerprint"),
        "rejection names the fingerprint ({err})"
    );
}

#[test]
fn fingerprint_overfits_nine_molecules_and_separates_isomers() {
    let _lock = serial();
    let device = dev();
    let set = nine_set();
    let bits = nine_bit_lists();
    let fps: Vec<SparseFingerprint> = bits
        .iter()
        .map(|b| SparseFingerprint::from_bits(b).unwrap())
        .collect();
    let indices: Vec<usize> = (0..set.examples.len()).collect();
    let mut trainer =
        CompletionTrainer::<R, E>::new(&fp_config(), &fp_train_config(), &device).unwrap();
    // Initial NLL with fingerprints.
    let owned_empty: Vec<Vec<MolGraph>> = indices.iter().map(|_| Vec::new()).collect();
    let refs: Vec<&[MolGraph]> = owned_empty.iter().map(Vec::as_slice).collect();
    let traces: Vec<&[Token]> = set.examples.iter().map(|e| e.trace.as_slice()).collect();
    let compositions: Vec<Composition> =
        set.examples.iter().map(|e| e.composition).collect();
    let initial = trainer
        .teacher_eval_with_fingerprints(&refs, &fps, &traces, &compositions)
        .unwrap();
    check_launches(&device).unwrap();
    let initial_mean: f64 =
        initial.iter().map(|&v| f64::from(v)).sum::<f64>() / initial.len() as f64;
    for step in 0..300 {
        if step % 20 == 0 {
            trainer.request_report();
        }
        let loss = trainer
            .step_with_fingerprints(&refs, &fps, &traces, &compositions)
            .unwrap();
        if step % 20 == 0 {
            check_launches(&device).unwrap();
            let _ = loss;
        }
        if step == 299 {
            check_launches(&device).unwrap();
        }
        // Early stop once below 30%: further training saturates every NLL
        // to exactly 0 and the own-vs-other comparison below ties.
        if (step + 1) % 20 == 0 {
            let cur = trainer
                .teacher_eval_with_fingerprints(&refs, &fps, &traces, &compositions)
                .unwrap();
            check_launches(&device).unwrap();
            let mean: f64 = cur.iter().map(|&v| f64::from(v)).sum::<f64>() / cur.len() as f64;
            if mean < 0.30 * initial_mean {
                break;
            }
        }
    }
    let final_nlls = trainer
        .teacher_eval_with_fingerprints(&refs, &fps, &traces, &compositions)
        .unwrap();
    check_launches(&device).unwrap();
    let final_mean: f64 =
        final_nlls.iter().map(|&v| f64::from(v)).sum::<f64>() / final_nlls.len() as f64;
    println!("fingerprint overfit: initial {initial_mean:.3} final {final_mean:.3}");
    assert!(
        final_mean < 0.30 * initial_mean,
        "trained NLL {final_mean:.3} is not below 30% of {initial_mean:.3}"
    );
    // Own-vs-other: each molecule's NLL given its own bits is lower than
    // given another molecule's bits, including the C2H6O isomers (ethanol 0
    // vs dimethyl ether 1). Early stopping above keeps NLLs out of the
    // saturated tie regime, so a strict comparison holds.
    let mut nll_with = |mol: usize, fp_idx: usize| -> f32 {
        let single_refs: Vec<&[MolGraph]> = vec![&[]];
        let single_traces = vec![set.examples[mol].trace.as_slice()];
        let single_comps = vec![set.examples[mol].composition];
        trainer
            .teacher_eval_with_fingerprints(
                &single_refs,
                &[fps[fp_idx].clone()],
                &single_traces,
                &single_comps,
            )
            .unwrap()[0]
    };
    for mol in 0..set.examples.len() {
        let other = (mol + 4) % set.examples.len();
        let own = nll_with(mol, mol);
        let foreign = nll_with(mol, other);
        assert!(
            own < foreign,
            "molecule {mol} NLL with own bits {own} is not below other {other} bits {foreign}"
        );
    }
    // Isomer spot checks (same formula, different bits).
    assert!(nll_with(0, 0) < nll_with(0, 1), "ethanol prefers its own bits");
    assert!(nll_with(1, 1) < nll_with(1, 0), "dimethyl ether prefers its own bits");
}

#[test]
fn fingerprint_store_loads_by_index_and_noise_rates() {
    let _lock = serial();
    // Bits file with `bits_by_molecule` aligned with the molecule order.
    let doc = serde_json::json!({
        "fingerprint": "morgan4096",
        "n_molecules": 2,
        "bits": {"A|1": [1, 2], "B|2": [3]},
        "bits_by_molecule": [[1, 2], [3]],
    });
    let store =
        FingerprintStore::load_json(&serde_json::to_string(&doc).unwrap()).unwrap();
    assert_eq!(store.len(), 2);
    store.assert_molecule_count(2).unwrap();
    assert!(store.assert_molecule_count(3).is_err());
    assert_eq!(store.get_by_index(0).unwrap(), &[1u16, 2]);
    assert_eq!(store.get_by_index(1).unwrap(), &[3u16]);
    assert!(store.get_by_index(2).is_err());
    assert_eq!(FINGERPRINT_BITS, 4096);
    assert_eq!(FINGERPRINT_SLOTS, 128);
}

#[test]
fn empty_pooled_vector_is_exact_zero_with_nonzero_bias() {
    // Finding 8 (review's completion_fingerprint.rs:473 case): the masked
    // mean is zero for an empty fingerprint, but `pool_in` has a trainable
    // bias, so the projected pool was the bias, not zero. The bias starts at
    // zero, so initialization-only checks miss this: drive every bias off
    // zero first, then assert exact zeros.
    let _lock = serial();
    let device = dev();
    let mut rng = Rng::seeded(41);
    let d = 16usize;
    let encoder: FingerprintEncoder<R, E> = FingerprintEncoder::init(d, &device, &mut rng);
    check_launches(&device).unwrap();
    let named = encoder.named_parameters();
    // Every bias off zero (pool_in to 1.0, the rest to 0.5).
    for (name, param) in named.iter() {
        if name.ends_with(".bias") {
            let shape = param.shape().dims().to_vec();
            let numel: usize = shape.iter().product();
            let fill = if name == "pool_in.bias" { 1.0f32 } else { 0.5f32 };
            param.set(Tensor::<R, E>::from_f32(&vec![fill; numel], shape, &device).unwrap());
        }
    }
    check_launches(&device).unwrap();
    // Empty queries pool to exact zero despite the nonzero projection bias.
    let empty = FingerprintBatch::empty(2, 4).unwrap();
    let pooled = encoder.encode_pooled(&empty, &device).unwrap();
    check_launches(&device).unwrap();
    for (k, v) in pooled.try_to_f32().unwrap().iter().enumerate() {
        assert_eq!(*v, 0.0, "empty pooled element {k} is not exact zero");
    }
    // A non-empty query still pools to a finite nonzero vector.
    let full =
        FingerprintBatch::build(&[SparseFingerprint::from_bits(&[3]).unwrap()], 4).unwrap();
    let pooled_full = encoder.encode_pooled(&full, &device).unwrap();
    check_launches(&device).unwrap();
    let vals = pooled_full.try_to_f32().unwrap();
    assert_eq!(vals.len(), d);
    assert!(vals.iter().all(|v| v.is_finite()));
    assert!(vals.iter().any(|&v| v != 0.0));
    // Padding token states stay exact zero with trained (nonzero) round
    // biases: selection after normalization restores them.
    let (states, _) = encoder.encode_states(&full, &device).unwrap();
    check_launches(&device).unwrap();
    let values = states.try_to_f32().unwrap();
    for s in 1..4 {
        for k in 0..d {
            assert_eq!(
                values[s * d + k],
                0.0,
                "padding slot {s} is not exact zero after trained biases"
            );
        }
    }
}

#[test]
fn noise_threshold_inside_bin_integrates_partial_mass() {
    // Finding 9 (review's 0.125 example): half of bin [0.10, 0.15) survives
    // sampling, but `ceil(threshold * 20)` dropped the whole intersected bin
    // from the reported retention.
    let _lock = serial();
    let mut on = vec![0u64; 20];
    let mut off = vec![0u64; 20];
    on[2] = 4;
    off[2] = 4092;
    let noise_json = serde_json::json!({
        "fingerprint": "morgan4096",
        "n_bits": 4096,
        "n_bins": 20,
        "hist_pred_given_true_on": on,
        "hist_pred_given_true_off": off,
        "n_spectra": 1,
        "n_molecules": 1,
    });
    let noise =
        FingerprintNoise::load_json(&serde_json::to_string(&noise_json).unwrap()).unwrap();
    // Half the bin survives: retention and off survival are 0.5, not 0.
    assert!((noise.retention_rate(0.125) - 0.5).abs() <= 1e-12);
    assert!(
        (noise.off_survival_at_level(0.125, FingerprintNoiseLevel::Spectrum) - 0.5).abs()
            <= 1e-12
    );
    // Query-specific expectation is (4096 - t) times the off survival, not
    // the corpus average.
    let query = noise.expected_false_tokens(46, 0.125, FingerprintNoiseLevel::Spectrum);
    assert!((query - (4096 - 46) as f64 * 0.5).abs() <= 1e-9);
    assert!((noise.mean_false_tokens(0.125) - 4092.0 * 0.5).abs() <= 1e-9);
    // The sampler agrees: empirical retention over draws is ~0.5 (score
    // with ample slots so token selection cannot drop sampled true bits:
    // the off histogram in the same bin yields ~2k false survivors).
    let truth = vec![5u16, 9];
    let draws = 500usize;
    let mut retained = 0usize;
    let mut total = 0usize;
    for draw in 0..draws {
        let fp = noise
            .sample(&truth, 5, "partial-bin-mol", draw as u64, 0.125)
            .unwrap();
        let stats = FingerprintQueryStats::score(&fp, &truth, 5000);
        retained += truth.len() - stats.true_missing;
        total += truth.len();
    }
    let empirical = retained as f64 / total as f64;
    assert!(
        (empirical - 0.5).abs() <= 0.05,
        "empirical retention {empirical} is not ~0.5 at threshold 0.125"
    );
}

#[test]
fn noise_level_selects_molecule_histograms() {
    // Finding 5: the v2 noise file carries both histogram sets; the sampler
    // takes the set by name.
    let _lock = serial();
    let mut on = vec![0u64; 20];
    let mut off = vec![0u64; 20];
    on[19] = 4;
    off[0] = 4096;
    let mut on_mol = vec![0u64; 20];
    let mut off_mol = vec![0u64; 20];
    on_mol[10] = 2;
    off_mol[10] = 2048;
    let noise_json = serde_json::json!({
        "fingerprint": "morgan4096",
        "n_bits": 4096,
        "n_bins": 20,
        "hist_pred_given_true_on": on,
        "hist_pred_given_true_off": off,
        "hist_pred_given_true_on_molecule": on_mol,
        "hist_pred_given_true_off_molecule": off_mol,
        "n_spectra": 2,
        "n_molecules": 1,
    });
    let noise =
        FingerprintNoise::load_json(&serde_json::to_string(&noise_json).unwrap()).unwrap();
    assert!(noise.has_molecule_histograms);
    // The two levels report different retention at 0.5: spectrum keeps the
    // bin-19 on mass (1.0), molecule keeps nothing above 0.5... bin 10 is
    // [0.50, 0.55): threshold 0.5 keeps all of it (1.0). Use 0.6 instead:
    // spectrum 1.0, molecule 0.0.
    assert!((noise.retention_rate_at_level(0.6, FingerprintNoiseLevel::Spectrum) - 1.0).abs() <= 1e-12);
    assert!((noise.retention_rate_at_level(0.6, FingerprintNoiseLevel::Molecule) - 0.0).abs() <= 1e-12);
    // Sampling is deterministic per level.
    let truth = vec![1u16, 2];
    let a = noise
        .sample_at_level(&truth, 3, "mol", 0, 0.1, FingerprintNoiseLevel::Molecule)
        .unwrap();
    let b = noise
        .sample_at_level(&truth, 3, "mol", 0, 0.1, FingerprintNoiseLevel::Molecule)
        .unwrap();
    assert_eq!(a, b);
    // Pre-v2 files without the molecule sets fall back to the spectrum ones
    // (flagged, so runs can refuse `--fp-noise-level molecule`).
    let legacy_json = serde_json::json!({
        "fingerprint": "morgan4096",
        "n_bits": 4096,
        "n_bins": 20,
        "hist_pred_given_true_on": on,
        "hist_pred_given_true_off": off,
        "n_spectra": 2,
        "n_molecules": 1,
    });
    let legacy =
        FingerprintNoise::load_json(&serde_json::to_string(&legacy_json).unwrap()).unwrap();
    assert!(!legacy.has_molecule_histograms);
    assert!(
        (legacy.retention_rate_at_level(0.6, FingerprintNoiseLevel::Molecule)
            - legacy.retention_rate_at_level(0.6, FingerprintNoiseLevel::Spectrum))
        .abs()
            <= 1e-12
    );
}

#[test]
fn store_rejects_reordered_sidecar() {
    // Review answer 1: the count check cannot detect a reordered same-length
    // sidecar, so v2 bits files bind the export order with keys_by_molecule.
    let _lock = serial();
    let doc = serde_json::json!({
        "fingerprint": "morgan4096",
        "n_molecules": 2,
        "bits": {"A|1": [1, 2], "B|2": [3]},
        "bits_by_molecule": [[1, 2], [3]],
        "keys_by_molecule": ["A|1", "B|2"],
    });
    let store =
        FingerprintStore::load_json(&serde_json::to_string(&doc).unwrap()).unwrap();
    assert!(store.has_keys());
    store
        .assert_keys_match(&["A|1".to_string(), "B|2".to_string()])
        .unwrap();
    assert!(
        store
            .assert_keys_match(&["B|2".to_string(), "A|1".to_string()])
            .is_err(),
        "a reordered same-length sidecar is an error"
    );
    // Pre-v2 files without the list skip the check (backwards compatible).
    let legacy = serde_json::json!({
        "fingerprint": "morgan4096",
        "n_molecules": 2,
        "bits": {"A|1": [1, 2], "B|2": [3]},
        "bits_by_molecule": [[1, 2], [3]],
    });
    let store =
        FingerprintStore::load_json(&serde_json::to_string(&legacy).unwrap()).unwrap();
    assert!(!store.has_keys());
    store
        .assert_keys_match(&["B|2".to_string(), "A|1".to_string()])
        .unwrap();
}
