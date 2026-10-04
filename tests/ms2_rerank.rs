//! K5 tests for the reranker model, its trainer and its training examples.
//!
//! The device-read test ([`trainer_reports_only_at_boundaries`]) shares the
//! process-global transfer counters with this binary, so every test in this
//! file holds `SERIAL`: the read assertion is exact, never racy.

#![cfg(feature = "backend")]

use std::sync::Mutex;

use mamba3::backend::{Device, reset_transfer_counters, runtime_read_count};
use mamba3::backends::Auto;
use mamba3::error::Error;
use mamba3::models::ms2::contain::Containment;
use mamba3::models::ms2::contract::{CandidateBatch, SCHEMA_VERSION, candidate_status};
use mamba3::models::ms2::identity::DUPLICATE_GRAPH;
use mamba3::models::ms2::rerank::{
    ExcludedCounts, N_FEATURES, RERANKER_VERSION, RerankTrainer, Reranker, eligible_examples,
};
use mamba3::nn::module::Module;
use mamba3::tensor::Tensor;
use mamba3::tensor::ops::random::Rng;

type R = Auto;
type E = f32;

/// Serializes this binary's tests: the read-counter assertion is exact only
/// when no other test reads concurrently.
static SERIAL: Mutex<()> = Mutex::new(());

/// Tiny deterministic generator (SplitMix64): no new dependencies.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: u32) -> u32 {
        (self.next() % u64::from(n.max(1))) as u32
    }

    fn f32(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * ((self.next() >> 11) as f32 / 9007199254740992.0)
    }
}

fn dev() -> Device<R> {
    Device::<R>::default()
}

fn upload(data: &[f32], shape: Vec<usize>, device: &Device<R>) -> Tensor<R, E> {
    Tensor::<R, f32>::from_f32(data, shape, device).unwrap()
}

/// Numerically stable BCE with logits, in `f64`, independent of the model.
fn bce_f64(x: f64, z: f64) -> f64 {
    x.max(0.0) - x * z + (1.0 + (-x.abs()).exp()).ln()
}

fn weighted_loss_f64(logits: &[f64], labels: &[f64], weights: &[f64]) -> f64 {
    let mut num = 0.0;
    let mut den = 0.0;
    for ((&x, &z), &w) in logits.iter().zip(labels.iter()).zip(weights.iter()) {
        num += w * bce_f64(x, z);
        den += w;
    }
    num / den.max(1.0)
}

#[test]
fn loss_matches_independent_f64() {
    let _guard = SERIAL.lock().unwrap();
    let device = dev();
    let mut rng = Rng::seeded(11);
    let model = Reranker::<R, E>::init(&device, &mut rng);
    let rows = 16usize;
    let mut lcg = Lcg(0x1001);
    let features: Vec<f32> = (0..rows * N_FEATURES).map(|_| lcg.f32(-3.0, 3.0)).collect();
    let labels: Vec<f32> = (0..rows).map(|_| lcg.below(2) as f32).collect();
    // Weights with zeros mixed in (but a positive sum).
    let mut weights: Vec<f32> = (0..rows).map(|i| if i % 3 == 0 { 0.0 } else { 1.0 }).collect();
    weights[0] = 1.0;
    let f_t = upload(&features, vec![rows, N_FEATURES], &device);
    let l_t = upload(&labels, vec![rows], &device);
    let w_t = upload(&weights, vec![rows], &device);
    let logits = model.logits(&f_t).unwrap().try_to_f32().unwrap();
    assert_eq!(logits.len(), rows);
    let expected = weighted_loss_f64(
        &logits.iter().map(|&v| f64::from(v)).collect::<Vec<_>>(),
        &labels.iter().map(|&v| f64::from(v)).collect::<Vec<_>>(),
        &weights.iter().map(|&v| f64::from(v)).collect::<Vec<_>>(),
    );
    let got = model.loss(&f_t, &l_t, &w_t).unwrap().try_to_f32().unwrap()[0];
    assert!(
        (f64::from(got) - expected).abs() < 1e-5,
        "loss {got} differs from the f64 reference {expected}"
    );
}

#[test]
fn loss_gradient_finite_difference() {
    let _guard = SERIAL.lock().unwrap();
    let device = dev();
    let mut rng = Rng::seeded(23);
    let model = Reranker::<R, E>::init(&device, &mut rng);
    let rows = 4usize;
    let mut lcg = Lcg(0x2002);
    let features: Vec<f32> = (0..rows * N_FEATURES).map(|_| lcg.f32(-2.0, 2.0)).collect();
    let labels: Vec<f32> = vec![1.0, 0.0, 1.0, 0.0];
    let weights: Vec<f32> = vec![1.0; rows];
    let f_t = upload(&features, vec![rows, N_FEATURES], &device);
    let l_t = upload(&labels, vec![rows], &device);
    let w_t = upload(&weights, vec![rows], &device);
    let loss_of = |m: &Reranker<R, E>| m.loss(&f_t, &l_t, &w_t).unwrap().try_to_f32().unwrap()[0];
    let grads = model.loss(&f_t, &l_t, &w_t).unwrap().backward().unwrap();
    let eps = 1e-3f32;
    for (name, param) in model.named_parameters() {
        let shape = param.shape().dims().to_vec();
        let base = param.value().to_f32();
        let analytic = grads
            .get(param.id())
            .unwrap_or_else(|| panic!("no gradient for {name}"))
            .to_f32();
        assert_eq!(analytic.len(), base.len(), "gradient length for {name}");
        for i in 0..base.len() {
            let mut plus = base.clone();
            plus[i] += eps;
            param.set(Tensor::<R, E>::from_f32(&plus, shape.clone(), &device).unwrap());
            let hi = loss_of(&model);
            let mut minus = base.clone();
            minus[i] -= eps;
            param.set(Tensor::<R, E>::from_f32(&minus, shape.clone(), &device).unwrap());
            let lo = loss_of(&model);
            param.set(Tensor::<R, E>::from_f32(&base, shape.clone(), &device).unwrap());
            let numeric = (hi - lo) / (2.0 * eps);
            assert!(
                (numeric - analytic[i]).abs() < 1e-3,
                "param {name}[{i}]: analytic {} vs numeric {numeric}",
                analytic[i]
            );
        }
    }
}

#[test]
fn learns_separable_task() {
    let _guard = SERIAL.lock().unwrap();
    let device = dev();
    let mut rng = Rng::seeded(37);
    let model = Reranker::<R, E>::init(&device, &mut rng);
    // Label = 1 when 1.5 x0 − 2.0 x1 + 0.5 > 0, with 5% label noise.
    let rows = 200usize;
    let mut lcg = Lcg(0x3003);
    let mut features = vec![0.0f32; rows * N_FEATURES];
    let mut labels = vec![0.0f32; rows];
    for r in 0..rows {
        let x0 = lcg.f32(-2.0, 2.0);
        let x1 = lcg.f32(-2.0, 2.0);
        features[r * N_FEATURES] = x0;
        features[r * N_FEATURES + 1] = x1;
        let mut y = u32::from(1.5 * x0 - 2.0 * x1 + 0.5 > 0.0);
        if lcg.below(100) < 5 {
            y = 1 - y;
        }
        labels[r] = y as f32;
    }
    let weights = vec![1.0f32; rows];
    let f_t = upload(&features, vec![rows, N_FEATURES], &device);
    let l_t = upload(&labels, vec![rows], &device);
    let w_t = upload(&weights, vec![rows], &device);
    let initial = model.loss(&f_t, &l_t, &w_t).unwrap().try_to_f32().unwrap()[0];
    let mut trainer = RerankTrainer::new(0.1, usize::MAX);
    let steps = 400;
    for _ in 0..steps {
        assert!(trainer.step(&model, &f_t, &l_t, &w_t, rows).unwrap().is_none());
    }
    let final_loss = model.loss(&f_t, &l_t, &w_t).unwrap().try_to_f32().unwrap()[0];
    assert!(
        final_loss < 0.5 * initial,
        "loss did not halve: {initial} -> {final_loss}"
    );
    let logits = model.logits(&f_t).unwrap().try_to_f32().unwrap();
    let correct = logits
        .iter()
        .zip(labels.iter())
        .filter(|(x, y)| (**x > 0.0) == (**y == 1.0))
        .count();
    let accuracy = correct as f64 / rows as f64;
    assert!(accuracy > 0.9, "accuracy {accuracy} is not above 0.9");
}

#[test]
fn zero_weights_zero_loss_no_change() {
    let _guard = SERIAL.lock().unwrap();
    let device = dev();
    let mut rng = Rng::seeded(41);
    let model = Reranker::<R, E>::init(&device, &mut rng);
    let rows = 8usize;
    let mut lcg = Lcg(0x4004);
    let features: Vec<f32> = (0..rows * N_FEATURES).map(|_| lcg.f32(-3.0, 3.0)).collect();
    let labels: Vec<f32> = (0..rows).map(|_| lcg.below(2) as f32).collect();
    let weights = vec![0.0f32; rows];
    let f_t = upload(&features, vec![rows, N_FEATURES], &device);
    let l_t = upload(&labels, vec![rows], &device);
    let w_t = upload(&weights, vec![rows], &device);
    let loss = model.loss(&f_t, &l_t, &w_t).unwrap().try_to_f32().unwrap()[0];
    assert_eq!(loss, 0.0, "zero weights must give zero loss");
    let before = model.state_dict();
    let mut trainer = RerankTrainer::new(0.1, usize::MAX);
    for _ in 0..5 {
        trainer.step(&model, &f_t, &l_t, &w_t, 0).unwrap();
    }
    let after = model.state_dict();
    assert_eq!(before.entries.keys().collect::<Vec<_>>(), after.entries.keys().collect::<Vec<_>>());
    for (name, entry) in &before.entries {
        assert_eq!(entry.data, after.entries[name].data, "param {name} changed on zero weights");
    }
}

#[test]
fn trainer_reports_only_at_boundaries() {
    let _guard = SERIAL.lock().unwrap();
    let device = dev();
    let mut rng = Rng::seeded(53);
    let model = Reranker::<R, E>::init(&device, &mut rng);
    let rows = 8usize;
    let mut lcg = Lcg(0x5005);
    let features: Vec<f32> = (0..rows * N_FEATURES).map(|_| lcg.f32(-2.0, 2.0)).collect();
    let labels: Vec<f32> = (0..rows).map(|_| lcg.below(2) as f32).collect();
    let weights = vec![1.0f32; rows];
    let f_t = upload(&features, vec![rows, N_FEATURES], &device);
    let l_t = upload(&labels, vec![rows], &device);
    let w_t = upload(&weights, vec![rows], &device);
    let mut trainer = RerankTrainer::new(0.05, 7);
    // Warmup absorbs any cold-start reads; none of these steps report.
    for _ in 0..3 {
        assert!(trainer.step(&model, &f_t, &l_t, &w_t, rows).unwrap().is_none());
    }
    reset_transfer_counters();
    let reads_before = runtime_read_count();
    // Steps 4, 5, 6 report nothing; step 7 hits the boundary.
    for step in [4, 5, 6] {
        let out = trainer.step(&model, &f_t, &l_t, &w_t, rows).unwrap();
        assert!(out.is_none(), "step {step} reported off-boundary");
    }
    assert_eq!(
        runtime_read_count() - reads_before,
        0,
        "non-report steps must not read the device"
    );
    let reported = trainer.step(&model, &f_t, &l_t, &w_t, rows).unwrap();
    assert!(reported.is_some(), "the boundary step must report the loss");
    assert!(reported.unwrap().is_finite(), "the reported loss must be finite");
}

#[test]
fn save_load_roundtrip_bit_identical() {
    let _guard = SERIAL.lock().unwrap();
    let device = dev();
    let mut rng = Rng::seeded(67);
    let model = Reranker::<R, E>::init(&device, &mut rng);
    let rows = 8usize;
    let mut lcg = Lcg(0x6006);
    let features: Vec<f32> = (0..rows * N_FEATURES).map(|_| lcg.f32(-3.0, 3.0)).collect();
    let labels: Vec<f32> = (0..rows).map(|_| lcg.below(2) as f32).collect();
    let weights = vec![1.0f32; rows];
    let f_t = upload(&features, vec![rows, N_FEATURES], &device);
    let l_t = upload(&labels, vec![rows], &device);
    let w_t = upload(&weights, vec![rows], &device);
    // Move off the initialization so the round trip covers trained weights.
    let mut trainer = RerankTrainer::new(0.05, usize::MAX);
    for _ in 0..10 {
        trainer.step(&model, &f_t, &l_t, &w_t, rows).unwrap();
    }
    let gen_config = serde_json::json!({
        "domain": "test-v0",
        "trajectories": 8,
        "formulas": 4,
        "dedup": "trace+graph",
        "seed": 1,
    });
    let path = std::env::temp_dir().join(format!("ms2_rerank_k5_{}.json", std::process::id()));
    model.save(&path, &gen_config).unwrap();
    let (restored, stored_config) = Reranker::<R, E>::load(&path, &device).unwrap();
    assert_eq!(stored_config, gen_config, "the stored configuration must round-trip");
    let before = model.logits(&f_t).unwrap().try_to_f32().unwrap();
    let after = restored.logits(&f_t).unwrap().try_to_f32().unwrap();
    assert_eq!(before.len(), after.len());
    for (i, (a, b)) in before.iter().zip(after.iter()).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "logit {i} changed over save/load");
    }
    // A tampered version is refused.
    let text = std::fs::read_to_string(&path).unwrap();
    let tampered = text.replace(RERANKER_VERSION, "ms2-reranker-v0");
    let bad_path = std::env::temp_dir().join(format!("ms2_rerank_k5_bad_{}.json", std::process::id()));
    std::fs::write(&bad_path, tampered).unwrap();
    assert!(matches!(
        Reranker::<R, E>::load(&bad_path, &device),
        Err(Error::Config(_))
    ));
}

/// A batch with hand-set statuses: no grammar replay is needed since
/// [`eligible_examples`] reads statuses and containment only.
fn hand_batch(statuses: &[u32]) -> CandidateBatch {
    let batch = 1usize;
    let trajectories = statuses.len();
    let (t, a) = (2usize, 2usize);
    let n = batch * trajectories;
    CandidateBatch {
        schema_version: SCHEMA_VERSION,
        batch,
        trajectories,
        max_steps: t,
        max_atoms: a,
        max_ring_closures: 1,
        spectrum_id: vec![7; n],
        trajectory: (0..n).map(|r| (r % trajectories) as u32).collect(),
        actions: vec![0; n * t * 4],
        length: vec![0; n],
        formula_row: vec![u32::MAX; n],
        formula_log_prob: vec![0.0; n],
        trace_log_prob: vec![0.0; n],
        open_valence: vec![0; n * a],
        attachment_partition: vec![0; n],
        status: statuses.to_vec(),
        evidence_status: vec![0; n],
        evidence_count: vec![0; n],
        evidence_peak_id: vec![0; n * 4],
        evidence_hypothesis: vec![0; n * 4],
        evidence_shift: vec![0; n * 4],
        evidence_residual: vec![0; n * 4],
        evidence_log_prob: vec![0.0; n * 4],
        identity_resolution: vec![0; n],
        request_status: vec![0; batch],
        rows_visited: vec![0; batch],
        rows_joined: vec![0; batch],
        rows_scored: vec![0; batch],
        formula_support_complete: vec![0; batch],
        formula_mass_retained: vec![0.0; batch],
        peaks_kept: vec![0; batch],
        intensity_retained: vec![0.0; batch],
        formula_counts: vec![0; n * 10],
        formula_source: vec![0; batch],
        formula_rank: vec![u32::MAX; n],
    }
}

#[test]
fn eligible_examples_counts() {
    let _guard = SERIAL.lock().unwrap();
    use candidate_status as cs;
    let statuses = vec![
        cs::FINISHED,                        // 0: contained -> (0, 1.0)
        cs::FINISHED,                        // 1: not contained -> (1, 0.0)
        0,                                   // 2: unfinished
        cs::FINISHED | cs::INVALID_FINAL,    // 3: invalid
        cs::FINISHED | cs::DUPLICATE_TRACE,  // 4: duplicate
        cs::FINISHED | DUPLICATE_GRAPH,      // 5: duplicate (graph)
        cs::FINISHED,                        // 6: work limit -> excluded
        cs::FINISHED | cs::TRUNCATED,        // 7: invalid (truncated)
    ];
    let batch = hand_batch(&statuses);
    let containment = vec![
        Containment::Contained,
        Containment::NotContained,
        Containment::Contained,
        Containment::Contained,
        Containment::Contained,
        Containment::NotContained,
        Containment::WorkLimit,
        Containment::Contained,
    ];
    let (indices, labels, excluded) = eligible_examples(&batch, &containment).unwrap();
    assert_eq!(indices, vec![0, 1], "only finished/valid/non-duplicate/resolved rows");
    assert_eq!(labels, vec![1.0, 0.0]);
    assert_eq!(
        excluded,
        ExcludedCounts { not_finished: 1, invalid: 2, duplicate: 2, work_limit: 1 },
        "every exclusion reason counted"
    );
    // Length mismatches are refused, not truncated.
    assert!(matches!(
        eligible_examples(&batch, &containment[..4]),
        Err(Error::Shape(_))
    ));
}

#[test]
fn bce_zero_logit_gradients_are_half() {
    let _guard = SERIAL.lock().unwrap();
    let device = dev();
    // Zero every parameter so the logits are exactly 0 whatever the
    // features hold (an untrained head with zero bias sits exactly here).
    // With one unit-weight row the loss is ln 2 and dL/dlogit must be
    // sigmoid(0) − y: +0.5 for y = 0, −0.5 for y = 1, visible on the output
    // bias; the old maximum/abs composition gave 0 and −1 instead.
    for (y, want) in [(0.0f32, 0.5f32), (1.0f32, -0.5f32)] {
        let mut rng = Rng::seeded(99);
        let model = Reranker::<R, E>::init(&device, &mut rng);
        for (_, param) in model.named_parameters() {
            let shape = param.shape().dims().to_vec();
            param.set(Tensor::<R, E>::zeros(shape, &device));
        }
        let f_t = upload(&[0.0f32; N_FEATURES], vec![1, N_FEATURES], &device);
        let l_t = upload(&[y], vec![1], &device);
        let w_t = upload(&[1.0f32], vec![1], &device);
        let loss = model.loss(&f_t, &l_t, &w_t).unwrap();
        let val = loss.try_to_f32().unwrap()[0];
        assert!((val - std::f32::consts::LN_2).abs() < 1e-6, "y={y}: loss {val}, want ln 2");
        let grads = loss.backward().unwrap();
        let (_, bias) = model
            .named_parameters()
            .into_iter()
            .find(|(n, _)| n.ends_with("out.bias"))
            .expect("out.bias");
        let g = grads.get(bias.id()).expect("gradient for out.bias").to_f32();
        assert_eq!(g.len(), 1);
        assert!(
            (g[0] - want).abs() < 1e-6,
            "y={y}: dL/dlogit is {}, want {want}",
            g[0]
        );
    }
}

#[test]
fn bce_matches_finite_differences_and_extreme_logits_finite() {
    use mamba3::autograd::Var;
    use mamba3::models::ms2::rerank::bce_with_logits;
    let _guard = SERIAL.lock().unwrap();
    let device = dev();
    // Finite differences away from zero, through traced leaves.
    for (xs, zs) in [
        (vec![-2.0f32, -0.5, 0.7, 3.0], vec![0.0f32, 1.0, 0.0, 1.0]),
        (vec![-80.0f32, 80.0], vec![1.0f32, 0.0]),
    ] {
        let n = xs.len();
        let x = Var::traced(upload(&xs, vec![n], &device));
        let z = Var::constant(upload(&zs, vec![n], &device));
        let loss = bce_with_logits(&x, &z).unwrap().sum().unwrap();
        let val = loss.try_to_f32().unwrap()[0];
        assert!(val.is_finite(), "BCE at {xs:?} is not finite: {val}");
        let grads = loss.backward_retain().unwrap();
        let g = grads
            .node(x.node().expect("traced leaf has a node"))
            .expect("logit gradient")
            .to_f32();
        for (i, (&xi, &zi)) in xs.iter().zip(zs.iter()).enumerate() {
            let sig = 1.0 / (1.0 + (-f64::from(xi)).exp());
            let want = (sig - f64::from(zi)) as f32;
            assert!(
                (g[i] - want).abs() < 2e-3,
                "x={xi} y={zi}: analytic {} vs sigmoid-based {want}",
                g[i]
            );
        }
    }
}

#[test]
fn zero_weight_steps_preserve_params_and_moments() {
    let _guard = SERIAL.lock().unwrap();
    let device = dev();
    let mut rng = Rng::seeded(43);
    let model = Reranker::<R, E>::init(&device, &mut rng);
    let rows = 8usize;
    let mut lcg = Lcg(0x4005);
    let features: Vec<f32> = (0..rows * N_FEATURES).map(|_| lcg.f32(-3.0, 3.0)).collect();
    let labels: Vec<f32> = (0..rows).map(|_| lcg.below(2) as f32).collect();
    let weights = vec![1.0f32; rows];
    let zero_w = vec![0.0f32; rows];
    let f_t = upload(&features, vec![rows, N_FEATURES], &device);
    let l_t = upload(&labels, vec![rows], &device);
    let w_t = upload(&weights, vec![rows], &device);
    let z_t = upload(&zero_w, vec![rows], &device);
    let mut trainer = RerankTrainer::new(0.1, usize::MAX);
    // One positive-weight step so Adam holds nonzero moments.
    trainer.step(&model, &f_t, &l_t, &w_t, rows).unwrap();
    let params_before = model.state_dict();
    let moments_before = trainer.optimizer_state_dict(&model);
    let steps_before = trainer.steps();
    let opt_steps_before = trainer.optimizer_steps();
    for _ in 0..5 {
        assert!(trainer.step(&model, &f_t, &l_t, &z_t, 0).unwrap().is_none());
    }
    let params_after = model.state_dict();
    let moments_after = trainer.optimizer_state_dict(&model);
    assert_eq!(trainer.steps(), steps_before, "zero-weight steps advanced the step counter");
    assert_eq!(
        trainer.optimizer_steps(),
        opt_steps_before,
        "zero-weight steps advanced Adam's clock"
    );
    assert_eq!(
        params_before.entries.keys().collect::<Vec<_>>(),
        params_after.entries.keys().collect::<Vec<_>>()
    );
    for (name, entry) in &params_before.entries {
        assert_eq!(
            entry.data, params_after.entries[name].data,
            "param {name} changed on zero-weight steps"
        );
    }
    assert_eq!(
        moments_before.entries.keys().collect::<Vec<_>>(),
        moments_after.entries.keys().collect::<Vec<_>>()
    );
    for (name, entry) in &moments_before.entries {
        assert_eq!(
            entry.data, moments_after.entries[name].data,
            "optimizer moment {name} changed on zero-weight steps"
        );
    }
}
