//! Parquet round trips: bars, sample table, predictions.

use std::fs::File;
use std::path::PathBuf;

use arrow::array::{
    Array, Float32Array, Float64Array, Int8Array, Int32Array, StringArray, UInt32Array,
};
use arrow::datatypes::DataType;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use clm_data::synthetic::generate;
use clm_data::{
    LabelConfig, Normalizer, Panel, Predictions, SyntheticConfig, compute_features, compute_labels,
    gather_last, load_parquet, read_predictions, valid_samples, write_parquet, write_predictions,
    write_samples,
};

fn tmp(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("clm-data-export");
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

#[test]
fn bars_round_trip_and_panel_from_files() {
    let sp = generate(&SyntheticConfig {
        missing_prob: 0.02,
        ..SyntheticConfig::default()
    });
    let mut back = Vec::new();
    for b in &sp.bars {
        let path = tmp(&format!("{}.parquet", b.symbol));
        write_parquet(b, &path).unwrap();
        let r = load_parquet(&path).unwrap();
        assert_eq!(&r, b);
        back.push(r);
    }
    let (p, report) =
        Panel::from_bars(&back[0], &back[1..3], &back[3..], &[0, 0, 0, 0, 1, 1, 1, 1]).unwrap();
    assert_eq!(p, sp.panel);
    assert_eq!(report.total_dropped_bad(), 0);
}

#[test]
fn rejects_unsorted_dates() {
    let sp = generate(&SyntheticConfig::default());
    let mut b = sp.bars[3].clone();
    b.date.swap(5, 6);
    assert!(write_parquet(&b, tmp("bad.parquet")).is_err());
    b.date[6] = b.date[5]; // duplicate
    assert!(b.validate().is_err());
}

#[test]
fn sample_table_round_trip() {
    let sp = generate(&SyntheticConfig {
        num_days: 500,
        ..SyntheticConfig::default()
    });
    let p = &sp.panel;
    let f = compute_features(p);
    let cfg = LabelConfig::default();
    let l = compute_labels(p, &cfg).unwrap();
    let norm = Normalizer::fit(&f, 0..400).unwrap();
    let nf = norm.apply(&f);
    let samples = valid_samples(&f, &l, 16);
    assert!(!samples.is_empty());
    let path = tmp("samples.parquet");
    write_samples(&path, p, &nf, &l, &samples).unwrap();

    let builder = ParquetRecordBatchReaderBuilder::try_new(File::open(&path).unwrap()).unwrap();
    let meta = builder.schema().metadata().clone();
    assert_eq!(meta["horizon"], "5");
    let names: Vec<String> = serde_json::from_str(&meta["feature_names"]).unwrap();
    assert_eq!(names, f.names);
    let batches: Vec<_> = builder.build().unwrap().map(|b| b.unwrap()).collect();
    let batch = arrow::compute::concat_batches(&batches[0].schema(), &batches).unwrap();
    assert_eq!(batch.num_rows(), samples.len());
    assert_eq!(batch.num_columns(), 4 + 20 + 4);
    let get = |n: &str| batch.column_by_name(n).unwrap().clone();
    assert_eq!(get("label").data_type(), &DataType::Int8);
    let s = get("s");
    let s = s.as_any().downcast_ref::<UInt32Array>().unwrap();
    let t = get("t");
    let t = t.as_any().downcast_ref::<UInt32Array>().unwrap();
    let date = get("date");
    let date = date.as_any().downcast_ref::<Int32Array>().unwrap();
    let sym = get("symbol");
    let sym = sym.as_any().downcast_ref::<StringArray>().unwrap();
    let label = get("label");
    let label = label.as_any().downcast_ref::<Int8Array>().unwrap();
    let fwd = get("fwd_ret");
    let fwd = fwd.as_any().downcast_ref::<Float64Array>().unwrap();
    let stop = get("stop");
    let stop = stop.as_any().downcast_ref::<Int8Array>().unwrap();
    let last = gather_last(&nf, &samples);
    for (i, smp) in samples.iter().enumerate() {
        assert_eq!((s.value(i), t.value(i)), (smp.s, smp.t));
        assert_eq!(date.value(i), p.dates[smp.day()]);
        assert_eq!(sym.value(i), p.stocks[smp.security()].symbol);
        let c = l.class_of(smp.security(), smp.day()).unwrap();
        assert_eq!(label.value(i), c.id() as i8);
        assert_eq!(
            fwd.value(i).to_bits(),
            l.fwd_ret_of(smp.security(), smp.day()).to_bits()
        );
        assert_eq!(stop.value(i), l.stop_of(smp.security(), smp.day()) as i8);
        for k in 0..20 {
            let col = get(&format!("f{k}"));
            let v = col
                .as_any()
                .downcast_ref::<Float32Array>()
                .unwrap()
                .value(i);
            assert_eq!(v.to_bits(), last[i * 20 + k].to_bits());
        }
        if i > 50 {
            break; // the per-feature column lookups make this quadratic-ish; 50 rows suffice
        }
    }
}

#[test]
fn predictions_round_trip_and_validation() {
    let preds = Predictions {
        s: vec![0, 1, 2],
        t: vec![300, 300, 301],
        probs: vec![[0.2, 0.3, 0.5], [1.0, 0.0, 0.0], [0.1, 0.1, 0.8]],
        pred_ret: Some(vec![0.01, -0.02, 0.0]),
    };
    let path = tmp("preds.parquet");
    write_predictions(&path, &preds).unwrap();
    assert_eq!(read_predictions(&path).unwrap(), preds);
    let no_ret = Predictions {
        pred_ret: None,
        ..preds.clone()
    };
    write_predictions(&path, &no_ret).unwrap();
    assert_eq!(read_predictions(&path).unwrap(), no_ret);

    let bad = Predictions {
        probs: vec![[0.2, 0.3, 0.6], [1.0, 0.0, 0.0], [0.1, 0.1, 0.8]],
        ..no_ret.clone()
    };
    write_predictions(&path, &bad).unwrap();
    assert!(read_predictions(&path).is_err());
    let dup = Predictions {
        t: vec![300, 300, 300],
        s: vec![0, 0, 1],
        ..no_ret.clone()
    };
    write_predictions(&path, &dup).unwrap();
    assert!(read_predictions(&path).is_err());
}

#[test]
fn predictions_accept_int64_ids_from_pandas() {
    use arrow::array::{ArrayRef, Int64Array, RecordBatch};
    use std::sync::Arc;
    let f = |v: Vec<f64>| Arc::new(Float64Array::from(v)) as ArrayRef;
    let batch = RecordBatch::try_from_iter([
        ("s", Arc::new(Int64Array::from(vec![3i64, 4])) as ArrayRef),
        ("t", Arc::new(Int64Array::from(vec![10i64, 11])) as ArrayRef),
        ("p_buy", f(vec![0.5, 0.25])),
        ("p_sell", f(vec![0.25, 0.25])),
        ("p_hold", f(vec![0.25, 0.5])),
    ])
    .unwrap();
    let path = tmp("preds_i64.parquet");
    let file = File::create(&path).unwrap();
    let mut w = parquet::arrow::ArrowWriter::try_new(file, batch.schema(), None).unwrap();
    w.write(&batch).unwrap();
    w.close().unwrap();
    let p = read_predictions(&path).unwrap();
    assert_eq!(p.s, vec![3, 4]);
    assert_eq!(p.t, vec![10, 11]);
    assert!(p.pred_ret.is_none());
    assert!(!batch.column(0).is_empty());
}
