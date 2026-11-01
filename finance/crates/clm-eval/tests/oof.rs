//! Out-of-fold Parquet and fold-metadata JSON round trips.

use clm_eval::folds::{FoldConfig, Scheme, make_folds};
use clm_eval::oof::*;

fn tmp(name: &str) -> std::path::PathBuf {
    let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("clm-eval-oof");
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

fn records() -> Vec<OofRecord> {
    (0..37)
        .map(|i| OofRecord {
            fold: (i / 10) as u32,
            test_year: 2010 + (i / 10) as i32,
            s: (i % 4) as u32,
            t: 1000 + i as u32,
            date: format!("2010-01-{:02}", 1 + i % 28),
            symbol: ["AAPL", "MSFT", "XOM", "JPM"][i % 4].to_string(),
            probs: [0.2, 0.3, 0.5],
            logits: [i as f32 * 0.1, -1.0, f32::MIN_POSITIVE],
            calibrated: [0.25, 0.25, 0.5],
            pred_ret: (i % 3 != 0).then_some(i as f32 * 0.01),
            label: (i % 5 != 0).then_some((i % 3) as u8),
            fwd_ret: -0.003 * i as f32,
        })
        .collect()
}

#[test]
fn parquet_round_trip() {
    let path = tmp("oof.parquet");
    let recs = records();
    write_oof_parquet(&path, &recs).unwrap();
    let back = read_oof_parquet(&path).unwrap();
    assert_eq!(back, recs);
    // empty file round-trips too
    let p2 = tmp("empty.parquet");
    write_oof_parquet(&p2, &[]).unwrap();
    assert!(read_oof_parquet(&p2).unwrap().is_empty());
}

#[test]
fn schema_has_expected_columns() {
    let s = oof_schema();
    let names: Vec<&str> = s.fields().iter().map(|f| f.name().as_str()).collect();
    assert_eq!(
        names[..6],
        ["fold", "test_year", "s", "t", "date", "symbol"]
    );
    assert!(
        names.contains(&"cal_hold") && names.contains(&"pred_ret") && names.contains(&"fwd_ret")
    );
    assert!(s.field_with_name("label").unwrap().is_nullable());
}

#[test]
fn fold_metadata_json_round_trip() {
    let years: Vec<i32> = (0..60).map(|t| 2000 + t / 10).collect();
    let dates: Vec<String> = (0..60).map(|t| format!("d{t:02}")).collect();
    let cfg = FoldConfig {
        scheme: Scheme::Rolling { train_days: 20 },
        first_test_year: 2003,
        last_test_year: 2004,
        val_days: 3,
        cal_days: 2,
        horizon: 2,
        embargo: 1,
        min_train_days: 5,
    };
    let folds = make_folds(&years, 60, &cfg);
    let meta: Vec<FoldMetadata> = folds
        .iter()
        .enumerate()
        .map(|(i, f)| FoldMetadata::new(i, f, &cfg, &dates, 17, "bars-v1", "abc123"))
        .collect();
    assert_eq!(meta[0].test.first_date.as_deref(), Some("d30"));
    assert_eq!(meta[0].test.last_date.as_deref(), Some("d39"));
    assert_eq!(meta[0].cal.first_date.as_deref(), Some("d28"));
    assert_eq!(meta[0].software_version, SOFTWARE_VERSION);
    let path = tmp("meta.json");
    write_json(&path, &meta).unwrap();
    let back: Vec<FoldMetadata> = read_json(&path).unwrap();
    assert_eq!(back, meta);
}

// ----------------------------------------------------------- malformed files are rejected

use std::sync::Arc;

use arrow::array::{ArrayRef, Float32Array, UInt8Array, UInt32Array};
use arrow::datatypes::{Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;

/// A valid batch with column `name` replaced, written with every column nullable (as
/// a file from another tool may be).
fn write_tampered(file: &str, name: &str, col: ArrayRef) -> std::path::PathBuf {
    let good = oof_to_batch(&records()[..3]).unwrap();
    let schema = Arc::new(Schema::new(
        good.schema()
            .fields()
            .iter()
            .map(|f| {
                let dt = if f.name() == name {
                    col.data_type().clone()
                } else {
                    f.data_type().clone()
                };
                Field::new(f.name(), dt, true)
            })
            .collect::<Vec<_>>(),
    ));
    let cols: Vec<ArrayRef> = schema
        .fields()
        .iter()
        .map(|f| {
            if f.name() == name {
                col.clone()
            } else {
                good.column_by_name(f.name()).unwrap().clone()
            }
        })
        .collect();
    let batch = RecordBatch::try_new(schema.clone(), cols).unwrap();
    let path = tmp(file);
    let mut w = ArrowWriter::try_new(std::fs::File::create(&path).unwrap(), schema, None).unwrap();
    w.write(&batch).unwrap();
    w.close().unwrap();
    path
}

fn assert_invalid(path: &std::path::Path, row: usize, needle: &str) {
    match read_oof_parquet(path) {
        Err(OofError::Invalid { row: r, reason }) => {
            assert_eq!(r, row, "{reason}");
            assert!(reason.contains(needle), "{reason}");
        }
        other => panic!("expected Invalid for {needle}, got {other:?}"),
    }
}

fn f32s(v: [Option<f32>; 3]) -> ArrayRef {
    Arc::new(Float32Array::from(v.to_vec()))
}

#[test]
fn nullable_schema_with_valid_data_reads() {
    // the tampering helper itself does not break a valid file
    let p = write_tampered(
        "ok.parquet",
        "fwd_ret",
        f32s([Some(0.1), Some(f32::NAN), None].map(|x| x.or(Some(0.0)))),
    );
    let back = read_oof_parquet(&p).unwrap();
    assert!(back[1].fwd_ret.is_nan()); // NaN fwd_ret (unlabelled day) is allowed
}

#[test]
fn label_out_of_range_is_rejected() {
    let p = write_tampered(
        "label.parquet",
        "label",
        Arc::new(UInt8Array::from(vec![Some(1), None, Some(255)])),
    );
    assert_invalid(&p, 2, "label 255");
}

#[test]
fn non_finite_probabilities_and_logits_are_rejected() {
    let p = write_tampered(
        "pnan.parquet",
        "p_buy",
        f32s([Some(0.2), Some(f32::NAN), Some(0.2)]),
    );
    assert_invalid(&p, 1, "probs");
    let p = write_tampered(
        "linf.parquet",
        "logit_sell",
        f32s([Some(0.0), Some(0.0), Some(f32::NEG_INFINITY)]),
    );
    assert_invalid(&p, 2, "logit");
    let p = write_tampered(
        "cinf.parquet",
        "cal_hold",
        f32s([Some(f32::INFINITY), Some(0.5), Some(0.5)]),
    );
    assert_invalid(&p, 0, "calibrated");
}

#[test]
fn rows_not_summing_to_one_are_rejected() {
    // p = [0.2, 0.3, 0.5] -> replace p_hold so the row sums to 0.9
    let p = write_tampered(
        "psum.parquet",
        "p_hold",
        f32s([Some(0.5), Some(0.4), Some(0.5)]),
    );
    assert_invalid(&p, 1, "probs");
    let p = write_tampered(
        "cneg.parquet",
        "cal_buy",
        f32s([Some(0.25), Some(0.25), Some(-0.25)]),
    );
    assert_invalid(&p, 2, "calibrated");
}

#[test]
fn nulls_in_non_nullable_columns_are_rejected() {
    let p = write_tampered(
        "pnull.parquet",
        "p_sell",
        f32s([Some(0.3), None, Some(0.3)]),
    );
    assert_invalid(&p, 1, "p_sell");
    let p = write_tampered(
        "fnull.parquet",
        "fold",
        Arc::new(UInt32Array::from(vec![None, Some(0), Some(0)])),
    );
    assert_invalid(&p, 0, "fold");
    let p = write_tampered(
        "rnull.parquet",
        "fwd_ret",
        f32s([Some(0.1), Some(0.1), None]),
    );
    assert_invalid(&p, 2, "fwd_ret");
}

#[test]
fn infinite_fwd_ret_and_pred_ret_are_rejected() {
    let p = write_tampered(
        "finf.parquet",
        "fwd_ret",
        f32s([Some(0.1), Some(f32::INFINITY), Some(0.1)]),
    );
    assert_invalid(&p, 1, "fwd_ret");
    let p = write_tampered(
        "prinf.parquet",
        "pred_ret",
        f32s([None, Some(f32::NAN), None]),
    );
    assert_invalid(&p, 1, "pred_ret");
}

#[test]
fn wrong_column_type_is_a_schema_error() {
    let p = write_tampered(
        "type.parquet",
        "label",
        Arc::new(UInt32Array::from(vec![0u32, 1, 2])),
    );
    assert!(matches!(read_oof_parquet(&p), Err(OofError::Schema(_))));
}

#[test]
fn writing_invalid_records_fails_without_a_file() {
    let path = tmp("never.parquet");
    let _ = std::fs::remove_file(&path);
    let mut recs = records();
    recs[5].label = Some(3);
    assert!(matches!(
        write_oof_parquet(&path, &recs),
        Err(OofError::Invalid { row: 5, .. })
    ));
    recs[5].label = None;
    recs[6].probs = [f32::NAN, 0.5, 0.5];
    assert!(matches!(
        write_oof_parquet(&path, &recs),
        Err(OofError::Invalid { row: 6, .. })
    ));
    assert!(!path.exists());
    // NaN fwd_ret with a null label is a valid unlabelled test day
    recs[6].probs = [0.2, 0.3, 0.5];
    recs[7].label = None;
    recs[7].fwd_ret = f32::NAN;
    write_oof_parquet(&path, &recs).unwrap();
    assert!(read_oof_parquet(&path).unwrap()[7].fwd_ret.is_nan());
}
