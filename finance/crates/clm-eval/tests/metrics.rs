//! Metrics against hand-computed values.

use clm_eval::metrics::*;

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

#[test]
fn average_ranks_with_ties() {
    assert_eq!(
        average_ranks(&[3.0, 1.0, 3.0, 2.0, 3.0]),
        vec![4.0, 1.0, 4.0, 2.0, 4.0]
    );
    assert_eq!(average_ranks(&[]), Vec::<f64>::new());
}

#[test]
fn auc_with_ties_hand_computed() {
    // positives {0.4, 0.8}, negatives {0.1, 0.35, 0.4}:
    // 0.4 beats 0.1, 0.35 and ties 0.4 (1/2) -> 2.5; 0.8 beats all -> 3; 5.5 / 6.
    let s = [0.1, 0.4, 0.35, 0.8, 0.4];
    let p = [false, true, false, true, false];
    assert!(close(roc_auc(&s, &p).unwrap(), 5.5 / 6.0, 1e-15));
    assert_eq!(roc_auc(&s, &[true; 5]), None);
    // all tied -> 0.5
    assert_eq!(roc_auc(&[0.3; 4], &[true, false, true, false]), Some(0.5));
}

#[test]
fn auc_ovr_macro_skips_absent_class() {
    let probs = [
        [0.7, 0.2, 0.1],
        [0.2, 0.7, 0.1],
        [0.6, 0.3, 0.1],
        [0.1, 0.8, 0.1],
    ];
    let labels = [0u8, 1, 0, 1];
    let (per, m) = roc_auc_ovr(&probs, &labels);
    assert_eq!(per[0], Some(1.0));
    assert_eq!(per[1], Some(1.0));
    assert_eq!(per[2], None); // HOLD never occurs
    assert_eq!(m, Some(1.0));
}

#[test]
fn accuracy_and_balanced_accuracy() {
    let probs = [
        [0.8, 0.1, 0.1],
        [0.6, 0.3, 0.1],
        [0.2, 0.7, 0.1],
        [0.1, 0.8, 0.1],
    ];
    let labels = [0u8, 0, 0, 1];
    assert!(close(accuracy(&probs, &labels), 0.75, 1e-15));
    // recall BUY 2/3, SELL 1/1; HOLD absent
    assert!(close(
        balanced_accuracy(&probs, &labels),
        (2.0 / 3.0 + 1.0) / 2.0,
        1e-15
    ));
}

#[test]
fn log_loss_and_brier_sanity() {
    let ll = log_loss(&[[1.0, 0.0, 0.0]], &[1]);
    assert!(close(ll, -(1e-12f64).ln(), 1e-9));
    assert!(close(log_loss(&[[0.5, 0.25, 0.25]], &[0]), 2f64.ln(), 1e-7));
    assert_eq!(log_loss(&[[1.0, 0.0, 0.0]], &[0]), 0.0);
    let u = 1.0f32 / 3.0;
    assert!(close(log_loss(&[[u, u, u]], &[2]), 3f64.ln(), 1e-6));

    assert_eq!(brier(&[[1.0, 0.0, 0.0]], &[0]), 0.0);
    assert_eq!(brier(&[[0.0, 1.0, 0.0]], &[0]), 2.0);
    assert!(close(brier(&[[u, u, u]], &[1]), 6.0 / 9.0, 1e-7));
    // mean over rows
    assert!(close(
        brier(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], &[0, 0]),
        1.0,
        1e-15
    ));
}

#[test]
fn ece_toy_hand_computed() {
    let probs = [
        [0.9, 0.05, 0.05], // conf .9, correct  -> bin 13
        [0.9, 0.05, 0.05], // conf .9, wrong    -> bin 13
        [0.5, 0.3, 0.2],   // conf .5, correct  -> bin 7
        [0.2, 0.7, 0.1],   // conf .7, wrong    -> bin 10
    ];
    let labels = [0u8, 1, 0, 2];
    let (ece, bins) = ece_top_label(&probs, &labels);
    // 2/4 * |0.5 - 0.9| + 1/4 * |1 - 0.5| + 1/4 * |0 - 0.7| = 0.2 + 0.125 + 0.175
    assert!(close(ece, 0.5, 1e-6), "{ece}");
    assert_eq!(bins.len(), 15);
    assert_eq!(bins[13].count, 2);
    assert!(close(bins[13].accuracy, 0.5, 0.0));
    assert!(close(bins[13].mean_confidence, 0.9, 1e-6));
    assert_eq!(bins[7].count, 1);
    assert_eq!(bins[10].count, 1);
    assert_eq!(bins.iter().map(|b| b.count).sum::<usize>(), 4);
    assert!(close(bins[0].lo, 0.0, 0.0) && close(bins[14].hi, 1.0, 1e-15));
    // perfectly confident and correct -> 0
    let (e0, b0) = ece_top_label(&[[1.0, 0.0, 0.0]], &[0]);
    assert_eq!(e0, 0.0);
    assert_eq!(b0[14].count, 1);
}

#[test]
fn classification_metrics_bundle_serializes() {
    let probs = [[0.6, 0.3, 0.1], [0.2, 0.5, 0.3], [0.1, 0.2, 0.7]];
    let labels = [0u8, 1, 2];
    let m = classification_metrics(&probs, &labels);
    assert_eq!(m.n, 3);
    assert_eq!(m.accuracy, 1.0);
    let back: ClassificationMetrics =
        serde_json::from_str(&serde_json::to_string(&m).unwrap()).unwrap();
    assert_eq!(back, m);
    let from_logits = classification_metrics_from_logits(&[[0.0, 0.0, 0.0]], &[1]);
    assert!(close(from_logits.log_loss, 3f64.ln(), 1e-6));
}

#[test]
fn spearman_with_ties_hand_computed() {
    // ranks x [1, 2.5, 2.5, 4], y [1, 3, 2, 4] -> r = 4.5 / sqrt(4.5 * 5)
    let r = spearman(&[1.0, 2.0, 2.0, 3.0], &[1.0, 3.0, 2.0, 4.0]).unwrap();
    assert!(close(r, 4.5 / (22.5f64).sqrt(), 1e-15));
    assert_eq!(spearman(&[1.0, 1.0, 1.0], &[1.0, 2.0, 3.0]), None);
}

#[test]
fn daily_rank_ic_hand_computed() {
    let mut day = Vec::new();
    let mut score = Vec::new();
    let mut ret = Vec::new();
    let mut push = |d: u32, s: &[f64], r: &[f64]| {
        for i in 0..s.len() {
            day.push(d);
            score.push(s[i]);
            ret.push(r[i]);
        }
    };
    push(0, &[1., 2., 3., 4., 5.], &[0.1, 0.2, 0.3, 0.4, 0.5]); // IC 1
    push(1, &[1., 2., 3., 4., 5.], &[2., 1., 4., 3., 5.]); // IC 1 - 6*4/120 = 0.8
    push(2, &[1., 2., 3., 4.], &[1., 2., 3., 4.]); // 4 securities: skipped
    push(3, &[1., 1., 1., 1., 1.], &[1., 2., 3., 4., 5.]); // constant score: skipped
    // a NaN pair on day 1 is ignored without dropping the day
    push(1, &[f64::NAN], &[0.0]);
    let ic = rank_ic(&day, &score, &ret);
    assert_eq!(ic.n_days, 2);
    assert_eq!(ic.n_skipped, 2);
    assert!(close(ic.mean, 0.9, 1e-12));
    let sd = (2.0f64 * 0.01).sqrt(); // sample sd of {1, 0.8}
    assert!(close(ic.sd, sd, 1e-12));
    assert!(close(ic.t_stat, 0.9 / sd * 2f64.sqrt(), 1e-9));
    assert_eq!(ic.daily, vec![(0, 1.0), (1, 0.8)]);
}

#[test]
fn regression_metrics_skip_non_finite() {
    let m = regression_metrics(&[1.0, 2.0, f64::NAN], &[2.0, 2.0, 5.0]);
    assert_eq!(m.n, 2);
    assert!(close(m.rmse, 0.5f64.sqrt(), 1e-15));
    assert!(close(m.mae, 0.5, 1e-15));
    assert_eq!(regression_metrics(&[], &[]).n, 0);
}

// ------------------------------------------------- invalid probabilities are rejected

#[test]
fn check_prob_row_cases() {
    assert!(check_prob_row(&[0.2, 0.3, 0.5]).is_ok());
    assert!(check_prob_row(&[1.0, 0.0, 0.0]).is_ok());
    assert!(check_prob_row(&[0.33334, 0.33333, 0.33333]).is_ok()); // within 1e-4
    for bad in [
        [f32::NAN, 0.0, 0.0],
        [f32::INFINITY, 0.0, 0.0],
        [f32::NEG_INFINITY, 1.0, 0.0],
        [1.2, -0.2, 0.0],
        [0.5, 0.3, 0.1],   // sums to 0.9
        [0.5, 0.5, 0.001], // sums to 1.001
    ] {
        assert!(check_prob_row(&bad).is_err(), "{bad:?}");
    }
}

#[test]
#[should_panic(expected = "invalid probabilities at row 1")]
fn nan_row_panics_in_accuracy() {
    // [NaN, 0, 0] used to argmax to BUY and score accuracy 1.0
    accuracy(&[[1.0, 0.0, 0.0], [f32::NAN, 0.0, 0.0]], &[0, 0]);
}

#[test]
#[should_panic(expected = "not finite")]
fn pos_inf_panics_in_brier() {
    brier(&[[f32::INFINITY, 0.0, 0.0]], &[0]);
}

#[test]
#[should_panic(expected = "not finite")]
fn neg_inf_panics_in_log_loss() {
    log_loss(&[[f32::NEG_INFINITY, 0.5, 0.5]], &[1]);
}

#[test]
#[should_panic(expected = "invalid probabilities")]
fn nan_panics_in_ece() {
    ece_top_label(&[[0.5, f32::NAN, 0.5]], &[0]);
}

#[test]
#[should_panic(expected = "invalid probabilities")]
fn nan_panics_in_auc() {
    roc_auc_ovr(&[[0.5, 0.5, 0.0], [0.2, 0.3, f32::NAN]], &[0, 1]);
}

#[test]
#[should_panic(expected = "not 1")]
fn unnormalised_row_panics_in_bundle() {
    classification_metrics(&[[0.6, 0.6, 0.6]], &[0]);
}

#[test]
#[should_panic(expected = "outside [0, 1]")]
fn negative_probability_panics() {
    balanced_accuracy(&[[1.5, -0.5, 0.0]], &[0]);
}

#[test]
#[should_panic(expected = "label out of range")]
fn bad_label_panics() {
    accuracy(&[[1.0, 0.0, 0.0]], &[3]);
}
