//! Calibration: never worse than identity on the fit set, fixes over/under-confidence,
//! serde round trip.

use clm_eval::calibration::{CalibrationMethod, Calibrator};
use clm_eval::metrics::{log_loss, softmax, softmax_rows};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const METHODS: [CalibrationMethod; 3] = [
    CalibrationMethod::Temperature,
    CalibrationMethod::Vector,
    CalibrationMethod::Isotonic,
];

fn normal(rng: &mut StdRng) -> f64 {
    // Box–Muller
    let u1: f64 = rng.random::<f64>().max(1e-300);
    let u2: f64 = rng.random();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Labels drawn from softmax(true logits); returns (true logits, labels).
fn draw(rng: &mut StdRng, n: usize, bias: [f64; 3]) -> (Vec<[f64; 3]>, Vec<u8>) {
    let mut z = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for _ in 0..n {
        let zt: [f64; 3] = std::array::from_fn(|k| normal(rng) + bias[k]);
        let p = softmax(&zt.map(|v| v as f32));
        let u: f64 = rng.random();
        let label = if u < p[0] {
            0
        } else if u < p[0] + p[1] {
            1
        } else {
            2
        };
        z.push(zt);
        y.push(label);
    }
    (z, y)
}

fn scaled(z: &[[f64; 3]], s: f64) -> Vec<[f32; 3]> {
    z.iter().map(|r| r.map(|v| (v * s) as f32)).collect()
}

#[test]
fn never_worse_than_identity_on_fit_set() {
    let mut rng = StdRng::seed_from_u64(7);
    for case in 0..40 {
        let n = rng.random_range(1..300);
        // random logits, labels unrelated or related depending on the case
        let (z, y) = draw(&mut rng, n, [0.0; 3]);
        let scale = [0.1, 1.0, 4.0, 20.0][case % 4];
        let logits = if case % 3 == 0 {
            // pure noise: logits independent of the labels
            (0..n)
                .map(|_| std::array::from_fn(|_| (normal(&mut rng) * scale) as f32))
                .collect()
        } else {
            scaled(&z, scale)
        };
        let id = log_loss(&softmax_rows(&logits), &y);
        for m in METHODS {
            let c = Calibrator::fit(m, &logits, &y);
            let ll = log_loss(&c.apply(&logits), &y);
            assert!(ll <= id, "case {case} {m:?}: {ll} > identity {id}");
        }
    }
}

#[test]
fn over_confident_model_is_improved() {
    let mut rng = StdRng::seed_from_u64(11);
    let (z_fit, y_fit) = draw(&mut rng, 4000, [0.0; 3]);
    let (z_test, y_test) = draw(&mut rng, 4000, [0.0; 3]);
    let fit = scaled(&z_fit, 4.0); // 4x over-confident
    let test = scaled(&z_test, 4.0);
    let id_fit = log_loss(&softmax_rows(&fit), &y_fit);
    let id_test = log_loss(&softmax_rows(&test), &y_test);
    let truth_test = log_loss(&softmax_rows(&scaled(&z_test, 1.0)), &y_test);

    let c = Calibrator::fit(CalibrationMethod::Temperature, &fit, &y_fit);
    let Calibrator::Temperature { t } = c else {
        panic!("{c:?}")
    };
    assert!((t - 4.0).abs() < 0.3, "T = {t}");
    for m in METHODS {
        let c = Calibrator::fit(m, &fit, &y_fit);
        let lf = log_loss(&c.apply(&fit), &y_fit);
        let lt = log_loss(&c.apply(&test), &y_test);
        assert!(lf < id_fit - 0.1, "{m:?}: fit {lf} vs {id_fit}");
        assert!(lt < id_test - 0.1, "{m:?}: held-out {lt} vs {id_test}");
        assert!(
            lt < truth_test + 0.05,
            "{m:?}: held-out {lt} vs oracle {truth_test}"
        );
    }
}

#[test]
fn under_confident_model_gets_t_below_one() {
    let mut rng = StdRng::seed_from_u64(12);
    let (z, y) = draw(&mut rng, 4000, [0.0; 3]);
    let fit = scaled(&z, 0.4);
    let Calibrator::Temperature { t } = Calibrator::fit(CalibrationMethod::Temperature, &fit, &y)
    else {
        panic!()
    };
    assert!((t - 0.4).abs() < 0.05, "T = {t}");
}

#[test]
fn vector_scaling_recovers_class_bias() {
    let mut rng = StdRng::seed_from_u64(13);
    // the true model has a class bias the logits lack
    let (z, y) = draw(&mut rng, 6000, [1.0, 0.0, -1.0]);
    let logits: Vec<[f32; 3]> = z
        .iter()
        .map(|r| [(r[0] - 1.0) as f32, r[1] as f32, (r[2] + 1.0) as f32])
        .collect();
    let c = Calibrator::fit(CalibrationMethod::Vector, &logits, &y);
    let Calibrator::Vector { a, b } = c else {
        panic!("{c:?}")
    };
    for ak in a {
        assert!((ak - 1.0).abs() < 0.15, "a = {a:?}");
    }
    // b is identified up to a shift: compare differences
    assert!(((b[0] - b[1]) - 1.0).abs() < 0.2, "b = {b:?}");
    assert!(((b[1] - b[2]) - 1.0).abs() < 0.2, "b = {b:?}");
}

#[test]
fn isotonic_rows_sum_to_one_and_are_monotone_per_class() {
    let mut rng = StdRng::seed_from_u64(14);
    let (z, y) = draw(&mut rng, 1000, [0.0; 3]);
    let logits = scaled(&z, 3.0);
    let c = Calibrator::fit(CalibrationMethod::Isotonic, &logits, &y);
    let Calibrator::Isotonic { classes } = &c else {
        panic!("{c:?}")
    };
    for cl in classes {
        assert!(cl.values.windows(2).all(|w| w[0] <= w[1]));
        assert!(cl.xs.windows(2).all(|w| w[0] < w[1]));
    }
    for r in c.apply(&logits) {
        let s: f32 = r.iter().sum();
        assert!((s - 1.0).abs() < 1e-5);
        assert!(r.iter().all(|&p| p > 0.0));
    }
}

#[test]
fn empty_fit_set_gives_identity() {
    assert_eq!(
        Calibrator::fit(CalibrationMethod::Temperature, &[], &[]),
        Calibrator::Temperature { t: 1.0 }
    );
    assert_eq!(
        Calibrator::fit(CalibrationMethod::Vector, &[], &[]),
        Calibrator::Identity
    );
    assert_eq!(
        Calibrator::fit(CalibrationMethod::Isotonic, &[], &[]),
        Calibrator::Identity
    );
    let z = [[1.0f32, 2.0, 3.0]];
    assert_eq!(Calibrator::Identity.apply(&z), softmax_rows(&z));
}

#[test]
fn all_zero_logits_keep_t_one() {
    let logits = vec![[0.0f32; 3]; 50];
    let y: Vec<u8> = (0..50).map(|i| (i % 3) as u8).collect();
    assert_eq!(
        Calibrator::fit(CalibrationMethod::Temperature, &logits, &y),
        Calibrator::Temperature { t: 1.0 }
    );
}

#[test]
fn serde_round_trip_preserves_outputs() {
    let mut rng = StdRng::seed_from_u64(15);
    let (z, y) = draw(&mut rng, 500, [0.3, 0.0, -0.2]);
    let logits = scaled(&z, 2.0);
    for m in METHODS {
        let c = Calibrator::fit(m, &logits, &y);
        let back: Calibrator = serde_json::from_str(&serde_json::to_string(&c).unwrap()).unwrap();
        assert_eq!(back, c, "{m:?}");
        assert_eq!(back.apply(&logits), c.apply(&logits), "{m:?}");
    }
}

#[test]
#[should_panic(expected = "non-finite logits at row 1")]
fn non_finite_logits_panic() {
    Calibrator::fit(
        CalibrationMethod::Temperature,
        &[[0.0, 1.0, 2.0], [f32::NAN, 0.0, 0.0]],
        &[0, 1],
    );
}

#[test]
#[should_panic(expected = "non-finite logits")]
fn infinite_logits_panic() {
    Calibrator::fit(
        CalibrationMethod::Vector,
        &[[f32::INFINITY, 0.0, 0.0]],
        &[0],
    );
}
