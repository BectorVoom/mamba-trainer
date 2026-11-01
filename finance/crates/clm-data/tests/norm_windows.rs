//! Normaliser (C4) and windows (C2).

use clm_data::{
    Action, FeatureTensor, LabelConfig, Labels, Normalizer, SampleIndex, gather_last,
    gather_windows, valid_samples,
};

/// 2 securities x `nd` days x 2 features; value `(s, t, f) = 100*s + 10*t + f`.
fn tensor(nd: usize) -> FeatureTensor {
    let mut x = FeatureTensor::new_invalid(2, nd, vec!["a".into(), "b".into()]);
    for s in 0..2 {
        for t in 0..nd {
            let i = x.row_index(s, t);
            x.valid[i] = true;
            for f in 0..2 {
                x.row_mut(s, t)[f] = (100 * s + 10 * t + f) as f32;
            }
        }
    }
    x
}

fn labels_all(nd: usize) -> Labels {
    let n = 2 * nd;
    Labels {
        config: LabelConfig::default(),
        num_securities: 2,
        num_days: nd,
        class: vec![Some(Action::Hold); n],
        fwd_ret: vec![0.0; n],
        sigma_h: vec![0.01; n],
        theta: vec![0.001; n],
        vol_ret: vec![0.0; n],
        mae: vec![0.0; n],
        stop: vec![false; n],
    }
}

#[test]
fn normalizer_hand_values_and_invalid_rows_ignored() {
    let mut x = tensor(6);
    x.invalidate(1, 2);
    let n = Normalizer::fit(&x, 1..4).unwrap();
    // Feature a over rows (0,1..=3) and (1,1),(1,3): 10,20,30,110,130.
    let v = [10.0, 20.0, 30.0, 110.0, 130.0f64];
    let mean = v.iter().sum::<f64>() / 5.0;
    let sd = (v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / 5.0).sqrt();
    assert_eq!(n.rows, 5);
    assert!((n.mean[0] - mean).abs() < 1e-12 && (n.sd[0] - sd).abs() < 1e-12);
    assert!((n.mean[1] - (mean + 1.0)).abs() < 1e-12);
    let y = n.apply(&x);
    assert!(!y.is_valid(1, 2) && y.row(1, 2).iter().all(|v| v.is_nan()));
    assert!((y.get(0, 1, 0) as f64 - (10.0 - mean) / sd).abs() < 1e-6);
}

#[test]
fn rows_outside_fit_range_do_not_matter() {
    let x = tensor(10);
    let a = Normalizer::fit(&x, 2..5).unwrap();
    let mut y = x.clone();
    for s in 0..2 {
        for t in (0..2).chain(5..10) {
            y.row_mut(s, t).fill(1e6 * (t as f32 + 1.0));
        }
    }
    y.invalidate(0, 9);
    let b = Normalizer::fit(&y, 2..5).unwrap();
    assert_eq!(a, b);
    // Bitwise: same f64 statistics.
    assert!(
        a.mean
            .iter()
            .zip(&b.mean)
            .all(|(p, q)| p.to_bits() == q.to_bits())
    );
}

#[test]
fn clamp_floor_and_serde() {
    let mut x = tensor(4);
    // Constant feature b over the fit range -> sd floored at 1e-6.
    for s in 0..2 {
        for t in 0..4 {
            x.row_mut(s, t)[1] = 5.0;
        }
    }
    let n = Normalizer::fit(&x, 0..4).unwrap();
    assert_eq!(n.sd[1], 1e-6);
    assert_eq!(n.transform(1, 5.0), 0.0);
    assert_eq!(n.transform(1, 6.0), 8.0);
    assert_eq!(n.transform(1, 4.0), -8.0);
    assert_eq!(n.transform(0, 1e30), 8.0);
    assert_eq!(n.transform(0, -1e30), -8.0);
    let y = n.apply(&x);
    assert!(y.values.iter().all(|v| (-8.0..=8.0).contains(v)));
    let back: Normalizer = serde_json::from_str(&serde_json::to_string(&n).unwrap()).unwrap();
    assert_eq!(back, n);
    assert!(Normalizer::fit(&x, 10..20).is_err());
}

#[test]
fn valid_samples_edges() {
    let nd = 8;
    let mut x = tensor(nd);
    let mut l = labels_all(nd);
    x.invalidate(0, 3);
    l.class[nd + 6] = None; // security 1, day 6
    let got = valid_samples(&x, &l, 3);
    let want: Vec<SampleIndex> = [
        (0, 2),
        (0, 6),
        (0, 7),
        (1, 2),
        (1, 3),
        (1, 4),
        (1, 5),
        (1, 7),
    ]
    .iter()
    .map(|&(s, t)| SampleIndex::new(s, t))
    .collect();
    assert_eq!(got, want);
    // W = 1: every valid row with a label.
    assert_eq!(valid_samples(&x, &l, 1).len(), 2 * nd - 2);
    // W longer than the calendar: nothing.
    assert!(valid_samples(&x, &l, nd + 1).is_empty());
    assert_eq!(valid_samples(&x, &l, nd), vec![SampleIndex::new(1, 7)]);
}

#[test]
fn gather_layout() {
    let x = tensor(8);
    let smp = [SampleIndex::new(1, 4), SampleIndex::new(0, 2)];
    let w = gather_windows(&x, &smp, 3);
    assert_eq!(w.len(), 2 * 3 * 2);
    // [b][w][f]: b=0 -> security 1 days 2,3,4; b=1 -> security 0 days 0,1,2.
    let want: Vec<f32> = [(1, 2), (1, 3), (1, 4), (0, 0), (0, 1), (0, 2)]
        .iter()
        .flat_map(|&(s, t)| [(100 * s + 10 * t) as f32, (100 * s + 10 * t + 1) as f32])
        .collect();
    assert_eq!(w, want);
    assert_eq!(gather_last(&x, &smp), vec![140.0, 141.0, 20.0, 21.0]);
}

#[test]
#[should_panic(expected = "window starts before day 0")]
fn gather_rejects_short_history() {
    gather_windows(&tensor(4), &[SampleIndex::new(0, 1)], 3);
}

#[test]
#[should_panic(expected = "invalid row")]
fn gather_rejects_invalid_row() {
    let mut x = tensor(4);
    x.invalidate(0, 1);
    gather_windows(&x, &[SampleIndex::new(0, 3)], 3);
}
