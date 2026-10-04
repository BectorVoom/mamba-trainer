//! K5 tests for Platt scaling, reliability metrics and the calibration
//! artifact.
//!
//! Pure host `f64`: no device, no counters.

use mamba3::error::Error;
use mamba3::models::ms2::calibration::{
    Binning, CalibrationArtifact, ConfigKeys, apply_platt, brier_by_stratum, brier_score,
    ece_by_stratum, expected_calibration_error, fit_platt, reliability_by_stratum,
    reliability_table, stratum_of,
};

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

    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }
}

fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

fn keys() -> ConfigKeys {
    ConfigKeys {
        domain: "test-domain-v1".to_string(),
        k: "8".to_string(),
        f: "4".to_string(),
        precision: "f32".to_string(),
        ranking: "reranker-v1".to_string(),
        search_policy: "table-m32".to_string(),
    }
}

#[test]
fn platt_recovers_known_parameters() {
    let (a0, b0) = (1.5, -0.4);
    let n = 4000usize;
    let mut rng = Lcg(0xc411);
    let mut logits = Vec::with_capacity(n);
    let mut labels = Vec::with_capacity(n);
    for _ in 0..n {
        let f = rng.range(-4.0, 4.0);
        logits.push(f);
        labels.push(if rng.unit() < sigmoid(a0 * f + b0) { 1.0 } else { 0.0 });
    }
    let params = fit_platt(&logits, &labels).unwrap();
    assert!(params.converged, "recovery fit must report convergence");
    assert!(params.a.is_finite() && params.b.is_finite());
    assert!(
        (params.a - a0).abs() < 0.2,
        "a = {} but the truth is {a0}",
        params.a
    );
    assert!(
        (params.b - b0).abs() < 0.15,
        "b = {} but the truth is {b0}",
        params.b
    );
    // Deterministic: a second fit agrees bit-for-bit.
    let again = fit_platt(&logits, &labels).unwrap();
    assert_eq!(params, again);
}

#[test]
fn platt_imbalanced_high_leverage_fixture() {
    // Review Part B1: 17 negatives at logit 0, positives at logits 1 and 100.
    // Undamped Newton diverges here (a ≈ 54250, b ≈ −7778786, all fitted
    // probabilities 0); damped Newton with objective-based backtracking gives
    // about a = 0.0344, b = −2.303, probabilities 0.0909, 0.0937, 0.7566.
    let mut logits = vec![0.0; 17];
    logits.push(1.0);
    logits.push(100.0);
    let mut labels = vec![0.0; 17];
    labels.push(1.0);
    labels.push(1.0);
    let params = fit_platt(&logits, &labels).unwrap();
    assert!(params.converged, "the reviewer fixture must converge");
    assert!(
        (params.a - 0.0344).abs() < 2e-3,
        "a = {} but the damped reference is 0.0344",
        params.a
    );
    assert!(
        (params.b + 2.303).abs() < 5e-3,
        "b = {} but the damped reference is −2.303",
        params.b
    );
    let probs = apply_platt(&params, &[0.0, 1.0, 100.0]);
    for (got, want) in probs.iter().zip([0.0909, 0.0937, 0.7566]) {
        assert!((got - want).abs() < 2e-3, "probability {got}, want {want}");
    }
    // The smoothed negative log-likelihood is ≈ 6.04, not ≈ 14.5M.
    let (n_pos, n_neg) = (2.0, 17.0);
    let (t_pos, t_neg) = ((n_pos + 1.0) / (n_pos + 2.0), 1.0 / (n_neg + 2.0));
    let mut nll = 0.0;
    for (&f, &y) in logits.iter().zip(labels.iter()) {
        let t = if y == 1.0 { t_pos } else { t_neg };
        let p = if f == 0.0 {
            probs[0]
        } else if f == 1.0 {
            probs[1]
        } else {
            probs[2]
        }
        .clamp(1e-300, 1.0 - 1e-15);
        nll += -(t * p.ln() + (1.0 - t) * (1.0 - p).ln());
    }
    assert!(nll < 7.0, "smoothed NLL {nll} is far above the damped reference 6.04");
    // A second imbalanced, high-leverage shape: 40 negatives at −1, positives
    // at 0 and 50. Must converge with finite, ordered probabilities.
    let mut logits2 = vec![-1.0; 40];
    logits2.push(0.0);
    logits2.push(50.0);
    let mut labels2 = vec![0.0; 40];
    labels2.push(1.0);
    labels2.push(1.0);
    let params2 = fit_platt(&logits2, &labels2).unwrap();
    assert!(params2.converged);
    assert!(params2.a.is_finite() && params2.b.is_finite());
    assert!(params2.a > 0.0, "the slope must stay positive, got {}", params2.a);
    let probs2 = apply_platt(&params2, &[-1.0, 0.0, 50.0]);
    assert!(probs2.iter().all(|p| p.is_finite() && (0.0..=1.0).contains(p)));
    assert!(probs2[0] < probs2[1] && probs2[1] < probs2[2], "order, got {probs2:?}");
}

#[test]
fn platt_overflow_resistant_standardisation() {
    // Review Part B5: logits around 1e200 and 1e308. A non-finite fit is
    // rejected with an error; a returned fit is always finite.
    for logits in [
        vec![1e308, 1e308],
        vec![-1e308, 1e308],
        vec![1e200, 1e200 + 1e190, -1e200],
        vec![0.0, 1e308],
    ] {
        let n = logits.len();
        let labels: Vec<f64> = (0..n).map(|i| f64::from((i % 2) as u8)).collect();
        if let Ok(params) = fit_platt(&logits, &labels) {
            assert!(params.converged);
            assert!(
                params.a.is_finite() && params.b.is_finite(),
                "non-finite fit returned for {logits:?}: ({}, {})",
                params.a,
                params.b
            );
            let probs = apply_platt(&params, &logits);
            assert!(probs.iter().all(|p| p.is_finite()));
        }
    }
    // Identical extreme logits: zero variance, still a finite converged fit.
    let params = fit_platt(&[1e308, 1e308], &[0.0, 1.0]).unwrap();
    assert!(params.converged);
    assert!(params.a.is_finite() && params.b.is_finite());
}

#[test]
fn ece_perfect_vs_miscalibrated() {
    // Perfectly calibrated: three pure bins hold exactly their probability.
    let mut probs = Vec::new();
    let mut labels = Vec::new();
    for _ in 0..90 {
        probs.push(0.1);
        labels.push(0.0);
    }
    for _ in 0..10 {
        probs.push(0.1);
        labels.push(1.0);
    }
    for _ in 0..50 {
        probs.push(0.5);
        labels.push(0.0);
    }
    for _ in 0..50 {
        probs.push(0.5);
        labels.push(1.0);
    }
    for _ in 0..90 {
        probs.push(0.9);
        labels.push(1.0);
    }
    for _ in 0..10 {
        probs.push(0.9);
        labels.push(0.0);
    }
    // Bins at 0.1/0.5/0.9 are pure: |freq − conf| is 0, 0, 0.
    let ece = expected_calibration_error(&probs, &labels, 10, Binning::EqualWidth).unwrap();
    assert!(ece < 1e-12, "a calibrated set must have ECE near 0, got {ece}");
    // Hand-computed miscalibration: bin [0, 0.5) holds (0, 0) with labels
    // (0, 1): |0.5 − 0| * 2/4 = 0.25; bin [0.5, 1] holds (1, 1) with labels
    // (0, 1): |0.5 − 1| * 2/4 = 0.25. Total 0.5.
    let probs = vec![0.0, 0.0, 1.0, 1.0];
    let labels = vec![0.0, 1.0, 0.0, 1.0];
    let ece = expected_calibration_error(&probs, &labels, 2, Binning::EqualWidth).unwrap();
    assert!((ece - 0.5).abs() < 1e-12, "hand-computed ECE is 0.5, got {ece}");
    let table = reliability_table(&probs, &labels, 2, Binning::EqualWidth).unwrap();
    assert_eq!(table.len(), 2);
    assert_eq!(table[0].count, 2);
    assert_eq!(table[0].mean_conf, 0.0);
    assert_eq!(table[0].freq, 0.5);
    assert_eq!(table[1].count, 2);
    assert_eq!(table[1].mean_conf, 1.0);
    assert_eq!(table[1].freq, 0.5);
    // Equal-mass on a sorted-separable set: groups [0.1, 0.2] (both
    // negative) and [0.8, 0.9] (both positive): 0.15 * 0.5 + 0.15 * 0.5.
    let probs = vec![0.1, 0.2, 0.8, 0.9];
    let labels = vec![0.0, 0.0, 1.0, 1.0];
    let ece = expected_calibration_error(&probs, &labels, 2, Binning::EqualMass).unwrap();
    assert!((ece - 0.15).abs() < 1e-12, "equal-mass ECE is 0.15, got {ece}");
    let table = reliability_table(&probs, &labels, 2, Binning::EqualMass).unwrap();
    assert_eq!(table.len(), 2);
    assert_eq!((table[0].lo, table[0].hi), (0.1, 0.2));
    assert_eq!((table[1].lo, table[1].hi), (0.8, 0.9));
}

#[test]
fn brier_hand_example() {
    // (0 + 0.25 + 0) / 3.
    let probs = vec![0.0, 0.5, 1.0];
    let labels = vec![0.0, 1.0, 1.0];
    let brier = brier_score(&probs, &labels).unwrap();
    assert!((brier - 1.0 / 12.0).abs() < 1e-12, "Brier is 1/12, got {brier}");
}

#[test]
fn degenerate_inputs_stay_finite() {
    // All one class.
    let logits: Vec<f64> = (0..20).map(|i| i as f64 * 0.5).collect();
    let labels = vec![1.0; 20];
    let params = fit_platt(&logits, &labels).unwrap();
    assert!(params.a.is_finite() && params.b.is_finite(), "all-positive fit");
    let labels = vec![0.0; 20];
    let params = fit_platt(&logits, &labels).unwrap();
    assert!(params.a.is_finite() && params.b.is_finite(), "all-negative fit");
    // Empty: the neutral map.
    let params = fit_platt(&[], &[]).unwrap();
    assert_eq!((params.a, params.b), (0.0, 0.0));
    assert_eq!(apply_platt(&params, &[3.0, -7.0]), vec![0.5, 0.5]);
    // A single point.
    let params = fit_platt(&[1.0], &[1.0]).unwrap();
    assert!(params.a.is_finite() && params.b.is_finite(), "single-point fit");
    // Extreme logits with mixed labels.
    let logits = vec![-1000.0, -500.0, 500.0, 1000.0];
    let labels = vec![0.0, 0.0, 1.0, 1.0];
    let params = fit_platt(&logits, &labels).unwrap();
    assert!(params.a.is_finite() && params.b.is_finite(), "extreme-logit fit");
    let probs = apply_platt(&params, &logits);
    assert!(probs.iter().all(|p| p.is_finite()), "extreme-logit apply");
    assert!(probs.iter().all(|&p| (0.0..=1.0).contains(&p)), "probabilities in range");
    assert!(probs[0] < probs[3], "extreme-logit apply preserves order");
    // Length mismatches and bad values are refused, not NaN.
    assert!(matches!(fit_platt(&[0.0], &[]), Err(Error::Config(_))));
    assert!(matches!(fit_platt(&[f64::NAN], &[1.0]), Err(Error::Config(_))));
    assert!(matches!(fit_platt(&[0.0], &[0.5]), Err(Error::Config(_))));
}

#[test]
fn strata_cover_contract_sizes() {
    assert_eq!(stratum_of(2), None);
    assert_eq!(stratum_of(3), Some(0));
    assert_eq!(stratum_of(5), Some(0));
    assert_eq!(stratum_of(6), Some(1));
    assert_eq!(stratum_of(9), Some(1));
    assert_eq!(stratum_of(10), Some(2));
    assert_eq!(stratum_of(16), Some(2));
    assert_eq!(stratum_of(17), None);
    // Only the 3–5 stratum is populated; the others are empty (ECE 0.0,
    // Brier 0.0, empty tables), and out-of-range atoms are excluded.
    let probs = vec![0.2, 0.8, 0.4, 0.9];
    let labels = vec![0.0, 1.0, 0.0, 1.0];
    let atoms = vec![4, 5, 2, 17];
    let ece = ece_by_stratum(&probs, &labels, &atoms, 2, Binning::EqualWidth).unwrap();
    assert_eq!(ece[1], 0.0);
    assert_eq!(ece[2], 0.0);
    // Stratum 0 holds (0.2, 0) and (0.8, 1): bin [0, 0.5) has |0 − 0.2| at
    // weight 1/2 and bin [0.5, 1] has |1 − 0.8| at weight 1/2: ECE 0.2.
    assert!((ece[0] - 0.2).abs() < 1e-12, "stratum ECE, got {}", ece[0]);
    let brier = brier_by_stratum(&probs, &labels, &atoms).unwrap();
    // ((0.2)^2 + (0.2)^2) / 2 = 0.04.
    assert!((brier[0] - 0.04).abs() < 1e-12, "stratum Brier, got {}", brier[0]);
    assert_eq!(brier[1], 0.0);
    assert_eq!(brier[2], 0.0);
    let tables = reliability_by_stratum(&probs, &labels, &atoms, 2, Binning::EqualWidth).unwrap();
    assert_eq!(tables[0].len(), 2);
    assert!(tables[1].is_empty());
    assert!(tables[2].is_empty());
    assert!(matches!(
        ece_by_stratum(&probs, &labels, &[4, 5], 2, Binning::EqualWidth),
        Err(Error::Config(_))
    ));
}

#[test]
fn artifact_roundtrip_and_refusal() {
    let params = mamba3::models::ms2::calibration::PlattParams { a: 1.25, b: -0.75, converged: true };
    let config = keys();
    let artifact = CalibrationArtifact::new(&params, "calibration", 500, 123, config.clone());
    // Serde round trip.
    let text = serde_json::to_string(&artifact).unwrap();
    let back: CalibrationArtifact = serde_json::from_str(&text).unwrap();
    assert_eq!(artifact, back);
    assert_eq!(back.fit_split, "calibration");
    assert_eq!((back.a, back.b), (1.25, -0.75));
    // Matching keys apply.
    artifact.validate_matches(&config).unwrap();
    let probs = artifact.apply(&config, &[0.0, 2.0]).unwrap();
    assert_eq!(probs, apply_platt(&params, &[0.0, 2.0]));
    // Every key refuses on its own.
    let mut variants = Vec::new();
    for i in 0..6 {
        let mut bad = config.clone();
        match i {
            0 => bad.domain = "other".to_string(),
            1 => bad.k = "16".to_string(),
            2 => bad.f = "8".to_string(),
            3 => bad.precision = "f16".to_string(),
            4 => bad.ranking = "raw".to_string(),
            _ => bad.search_policy = "enumerate".to_string(),
        }
        variants.push(bad);
    }
    for bad in &variants {
        assert!(matches!(artifact.validate_matches(bad), Err(Error::Config(_))), "key {bad:?}");
        assert!(matches!(artifact.apply(bad, &[0.0]), Err(Error::Config(_))));
    }
}
