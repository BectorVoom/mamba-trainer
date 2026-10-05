//! Driver-logic smoke test for task RK: the reranker learns a planted signal,
//! Platt scaling lowers ECE, and a zero-weight step changes nothing.
//!
//! Synthetic data only (no export files): 200 training examples and 200 fresh
//! examples in 40 groups of 5, with the label planted in feature 2 and pure
//! noise in the raw-score column (feature 0). The launch/read counter
//! assertion lives in exactly one test ([`report_boundaries_read_once`]);
//! every test in this binary holds `SERIAL` so that assertion stays exact.

#![cfg(feature = "backend")]

use std::sync::Mutex;

use mamba3::backend::{Device, reset_transfer_counters, runtime_read_count};
use mamba3::backends::Auto;
use mamba3::models::ms2::calibration::{
    Binning, apply_platt, expected_calibration_error, fit_platt,
};
use mamba3::models::ms2::rerank::{N_FEATURES, RerankTrainer, Reranker};
use mamba3::models::ms2::rerank_eval::roc_auc;
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

    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / 9007199254740992.0
    }
}

fn dev() -> Device<R> {
    Device::<R>::default()
}

fn upload(data: &[f32], shape: Vec<usize>, device: &Device<R>) -> Tensor<R, E> {
    Tensor::<R, f32>::from_f32(data, shape, device).unwrap()
}

fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// One synthetic draw: 200 examples in 40 groups of 5. Feature 2 carries the
/// label (`2 * y` plus uniform noise); feature 0 (the raw-score column) is
/// pure noise; the rest is small noise. With `flip` above zero a fraction of
/// the labels is flipped, which is what makes an overconfident model
/// miscalibrated on fresh draws.
fn draw(seed: u64, flip: f64) -> (Vec<f32>, Vec<f32>) {
    let mut rng = Lcg(seed);
    let rows = 200usize;
    let mut features = vec![0.0f32; rows * N_FEATURES];
    let mut labels = vec![0.0f32; rows];
    for r in 0..rows {
        let truth = if r % 2 == 0 { 1.0f32 } else { 0.0f32 };
        let mut label = truth;
        if rng.unit() < flip {
            label = 1.0 - label;
        }
        labels[r] = label;
        features[r * N_FEATURES] = (rng.unit() as f32 - 0.5) * 4.0;
        features[r * N_FEATURES + 1] = (rng.unit() as f32 - 0.5) * 0.5;
        // The signal tracks the truth, never the flipped label: with
        // `flip > 0` the features stay predictive but the labels are noisy.
        features[r * N_FEATURES + 2] = truth * 2.0 + (rng.unit() as f32 - 0.5);
        for f in 3..N_FEATURES {
            features[r * N_FEATURES + f] = (rng.unit() as f32 - 0.5) * 0.5;
        }
    }
    (features, labels)
}

fn to_f64(v: &[f32]) -> Vec<f64> {
    v.iter().map(|&x| f64::from(x)).collect()
}

#[test]
fn pipeline_learns_signal_and_calibrates() {
    let _guard = SERIAL.lock().unwrap();
    let device = dev();
    // Clean training labels make the head overconfident (extreme logits);
    // the calibration and fresh draws carry 25% label flips, so the extreme
    // probabilities are miscalibrated there and Platt scaling has to temper
    // them — the same split roles as the driver (rank/calibration/report).
    let (train_f, train_l) = draw(0x51ab, 0.0);
    let (cal_f, cal_l) = draw(0xc411, 0.25);
    let (fresh_f, fresh_l) = draw(0xf9e5, 0.25);
    let rows = 200usize;
    let weights = vec![1.0f32; rows];
    let f_t = upload(&train_f, vec![rows, N_FEATURES], &device);
    let l_t = upload(&train_l, vec![rows], &device);
    let w_t = upload(&weights, vec![rows], &device);

    let mut rng = Rng::seeded(7);
    let model = Reranker::<R, E>::init(&device, &mut rng);
    // A few hundred steps with no intermediate reads (`report_every` past
    // the step budget, so the loss is never downloaded here).
    let mut trainer = RerankTrainer::new(0.1, usize::MAX);
    for _ in 0..300 {
        assert!(trainer.step(&model, &f_t, &l_t, &w_t, rows).unwrap().is_none());
    }

    // Fresh examples: the trained logit beats the raw-score column.
    let fresh_t = upload(&fresh_f, vec![rows, N_FEATURES], &device);
    let fresh_logits = model.logits(&fresh_t).unwrap().try_to_f32().unwrap();
    let raw_col: Vec<f32> = (0..rows).map(|r| fresh_f[r * N_FEATURES]).collect();
    let logit_auc = roc_auc(&to_f64(&fresh_logits), &to_f64(&fresh_l)).unwrap();
    let raw_auc = roc_auc(&to_f64(&raw_col), &to_f64(&fresh_l)).unwrap();
    assert!(
        logit_auc > raw_auc + 0.2,
        "trained AUC {logit_auc} must beat the raw column {raw_auc}"
    );

    // Platt fitted on the calibration logits lowers ECE on fresh examples
    // below the uncalibrated (sigmoid) ECE.
    let cal_t = upload(&cal_f, vec![rows, N_FEATURES], &device);
    let cal_logits = model.logits(&cal_t).unwrap().try_to_f32().unwrap();
    let params = fit_platt(&to_f64(&cal_logits), &to_f64(&cal_l)).unwrap();
    assert!(params.converged);
    let cal = apply_platt(&params, &to_f64(&fresh_logits));
    let uncal: Vec<f64> = fresh_logits.iter().map(|&x| sigmoid(f64::from(x))).collect();
    let fresh_y = to_f64(&fresh_l);
    let ece_cal =
        expected_calibration_error(&cal, &fresh_y, 10, Binning::EqualWidth).unwrap();
    let ece_uncal =
        expected_calibration_error(&uncal, &fresh_y, 10, Binning::EqualWidth).unwrap();
    assert!(
        ece_uncal > 0.05,
        "the overconfident model must be visibly miscalibrated, got {ece_uncal}"
    );
    assert!(
        ece_cal < ece_uncal,
        "calibrated ECE {ece_cal} must beat uncalibrated {ece_uncal}"
    );

    // A zero-weight padded last step (no eligible example) leaves the
    // parameters bit-identical.
    let before = model.state_dict();
    let zero_w = upload(&vec![0.0f32; rows], vec![rows], &device);
    assert!(trainer.step(&model, &f_t, &l_t, &zero_w, 0).unwrap().is_none());
    let after = model.state_dict();
    assert_eq!(
        before.entries.keys().collect::<Vec<_>>(),
        after.entries.keys().collect::<Vec<_>>()
    );
    for (name, entry) in &before.entries {
        assert_eq!(
            entry.data, after.entries[name].data,
            "param {name} changed on a zero-weight step"
        );
    }
}

/// The only test in this binary that asserts transfer counters: loss reports
/// happen exactly at the boundary (`report_every` 7: steps 1–6 silent with no
/// read, step 7 reports with one read).
#[test]
fn report_boundaries_read_once() {
    let _guard = SERIAL.lock().unwrap();
    let device = dev();
    let (train_f, train_l) = draw(0x1234, 0.0);
    let rows = 200usize;
    let weights = vec![1.0f32; rows];
    let f_t = upload(&train_f, vec![rows, N_FEATURES], &device);
    let l_t = upload(&train_l, vec![rows], &device);
    let w_t = upload(&weights, vec![rows], &device);
    let mut rng = Rng::seeded(99);
    let model = Reranker::<R, E>::init(&device, &mut rng);
    let mut trainer = RerankTrainer::new(0.05, 7);
    for _ in 0..3 {
        assert!(trainer.step(&model, &f_t, &l_t, &w_t, rows).unwrap().is_none());
    }
    reset_transfer_counters();
    let reads_before = runtime_read_count();
    for step in [4, 5, 6] {
        assert!(
            trainer.step(&model, &f_t, &l_t, &w_t, rows).unwrap().is_none(),
            "step {step} reported off-boundary"
        );
    }
    assert_eq!(
        runtime_read_count() - reads_before,
        0,
        "non-report steps must not read the device"
    );
    let reported = trainer.step(&model, &f_t, &l_t, &w_t, rows).unwrap();
    assert!(reported.is_some(), "the boundary step must report the loss");
    assert!(reported.unwrap().is_finite());
    assert_eq!(
        runtime_read_count() - reads_before,
        1,
        "the boundary step reads the loss exactly once"
    );
}
