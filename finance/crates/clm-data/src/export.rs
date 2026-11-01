//! Parquet exchange with external baselines (LightGBM in Python, PLAN AD8).
//!
//! * [`write_samples`]: one row per sample — `s` u32, `t` u32, `date` i32, `symbol` utf8,
//!   `f0..f{F-1}` f32 (the **normalised** last feature row; the caller passes the
//!   normalisation fitted on the fold), `label` i8 (class id, -1 none), `fwd_ret` f64,
//!   `vol_ret` f64, `stop` i8 (0/1, -1 none). Schema metadata: `feature_names` (JSON
//!   list), `horizon`, `kappa`, `margin_mult`, `stop_frac`.
//! * [`read_predictions`]: `s`, `t` (any integer type, cast to u32), `p_buy`, `p_sell`,
//!   `p_hold` f64 and optional `pred_ret` f64.

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, Float32Array, Float64Array, Int8Array, Int32Array, RecordBatch, StringArray,
    UInt32Array,
};
use arrow::compute::cast;
use arrow::datatypes::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;

use crate::bars::{Panel, column, f64_column};
use crate::error::{DataError, Result};
use crate::features::FeatureTensor;
use crate::labels::Labels;
use crate::types::{NUM_ACTIONS, SampleIndex};

/// Writes one record batch as a Snappy-compressed Parquet file.
pub(crate) fn write_batch(path: &Path, batch: &RecordBatch) -> Result<()> {
    let file = File::create(path).map_err(|e| DataError::io(path, e))?;
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .build();
    let mut writer = ArrowWriter::try_new(file, batch.schema(), Some(props))?;
    writer.write(batch)?;
    writer.close()?;
    Ok(())
}

/// Writes the sample table for external baselines (see the module docs). `normalized`
/// is the feature tensor after the fold's [`crate::Normalizer::apply`].
pub fn write_samples(
    path: impl AsRef<Path>,
    panel: &Panel,
    normalized: &FeatureTensor,
    labels: &Labels,
    samples: &[SampleIndex],
) -> Result<()> {
    let nf = normalized.num_features;
    let mut fields = vec![
        Field::new("s", DataType::UInt32, false),
        Field::new("t", DataType::UInt32, false),
        Field::new("date", DataType::Int32, false),
        Field::new("symbol", DataType::Utf8, false),
    ];
    let mut cols: Vec<ArrayRef> = vec![
        Arc::new(UInt32Array::from_iter_values(samples.iter().map(|x| x.s))),
        Arc::new(UInt32Array::from_iter_values(samples.iter().map(|x| x.t))),
        Arc::new(Int32Array::from_iter_values(
            samples.iter().map(|x| panel.dates[x.day()]),
        )),
        Arc::new(StringArray::from_iter_values(
            samples
                .iter()
                .map(|x| panel.stocks[x.security()].symbol.as_str()),
        )),
    ];
    for f in 0..nf {
        fields.push(Field::new(format!("f{f}"), DataType::Float32, false));
        cols.push(Arc::new(Float32Array::from_iter_values(
            samples
                .iter()
                .map(|x| normalized.get(x.security(), x.day(), f)),
        )));
    }
    let (s_of, t_of) = (|x: &SampleIndex| x.security(), |x: &SampleIndex| x.day());
    fields.push(Field::new("label", DataType::Int8, false));
    cols.push(Arc::new(Int8Array::from_iter_values(samples.iter().map(
        |x| {
            labels
                .class_of(s_of(x), t_of(x))
                .map_or(-1, |a| a.id() as i8)
        },
    ))));
    fields.push(Field::new("fwd_ret", DataType::Float64, false));
    cols.push(Arc::new(Float64Array::from_iter_values(
        samples.iter().map(|x| labels.fwd_ret_of(s_of(x), t_of(x))),
    )));
    fields.push(Field::new("vol_ret", DataType::Float64, false));
    cols.push(Arc::new(Float64Array::from_iter_values(
        samples.iter().map(|x| labels.vol_ret_of(s_of(x), t_of(x))),
    )));
    fields.push(Field::new("stop", DataType::Int8, false));
    cols.push(Arc::new(Int8Array::from_iter_values(samples.iter().map(
        |x| {
            let (s, t) = (s_of(x), t_of(x));
            if labels.exists(s, t) {
                labels.stop_of(s, t) as i8
            } else {
                -1
            }
        },
    ))));
    let c = &labels.config;
    let meta: HashMap<String, String> = [
        ("feature_names", serde_json::to_string(&normalized.names)?),
        ("horizon", c.horizon.to_string()),
        ("kappa", c.kappa.to_string()),
        ("margin_mult", c.margin_mult.to_string()),
        ("stop_frac", c.stop_frac.to_string()),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v))
    .collect();
    let schema = Arc::new(Schema::new(fields).with_metadata(meta));
    let batch = RecordBatch::try_new(schema, cols)?;
    write_batch(path.as_ref(), &batch)
}

/// Class probabilities (and optionally a predicted return) per sample from an external
/// model. `probs[i]` is in class-id order `[BUY, SELL, HOLD]`.
#[derive(Debug, Clone, PartialEq)]
pub struct Predictions {
    /// Security per row.
    pub s: Vec<u32>,
    /// Decision day per row.
    pub t: Vec<u32>,
    /// `[p_buy, p_sell, p_hold]` per row.
    pub probs: Vec<[f64; NUM_ACTIONS]>,
    /// Predicted return per row, if the file has `pred_ret`.
    pub pred_ret: Option<Vec<f64>>,
}

impl Predictions {
    /// Number of rows.
    pub fn len(&self) -> usize {
        self.s.len()
    }

    /// True if there are no rows.
    pub fn is_empty(&self) -> bool {
        self.s.is_empty()
    }

    /// The sample of row `i`.
    pub fn sample(&self, i: usize) -> SampleIndex {
        SampleIndex {
            s: self.s[i],
            t: self.t[i],
        }
    }

    /// Row lookup by sample.
    pub fn index(&self) -> HashMap<SampleIndex, usize> {
        (0..self.len()).map(|i| (self.sample(i), i)).collect()
    }
}

fn u32_column(batch: &RecordBatch, name: &str) -> Result<Vec<u32>> {
    let col = cast(&column(batch, name)?, &DataType::UInt32)?;
    let arr = col
        .as_any()
        .downcast_ref::<UInt32Array>()
        .ok_or_else(|| DataError::Schema(format!("{name}: not u32")))?;
    if arr.null_count() > 0 {
        return Err(DataError::Schema(format!(
            "{name}: nulls or values outside u32"
        )));
    }
    Ok(arr.values().to_vec())
}

/// Reads a predictions file (see the module docs). Errors on missing columns, nulls,
/// non-finite or negative probabilities, rows whose probabilities do not sum to 1
/// (tolerance 1e-4), and duplicate `(s, t)`.
pub fn read_predictions(path: impl AsRef<Path>) -> Result<Predictions> {
    let path = path.as_ref();
    let file = File::open(path).map_err(|e| DataError::io(path, e))?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;
    let mut out = Predictions {
        s: Vec::new(),
        t: Vec::new(),
        probs: Vec::new(),
        pred_ret: None,
    };
    let mut pred: Vec<f64> = Vec::new();
    let mut has_pred = None;
    for batch in reader {
        let batch = batch?;
        out.s.extend(u32_column(&batch, "s")?);
        out.t.extend(u32_column(&batch, "t")?);
        let (b, s, h) = (
            f64_column(&batch, "p_buy")?,
            f64_column(&batch, "p_sell")?,
            f64_column(&batch, "p_hold")?,
        );
        out.probs.extend((0..b.len()).map(|i| [b[i], s[i], h[i]]));
        let this = batch.column_by_name("pred_ret").is_some();
        if *has_pred.get_or_insert(this) != this {
            return Err(DataError::Schema("pred_ret in some batches only".into()));
        }
        if this {
            pred.extend(f64_column(&batch, "pred_ret")?);
        }
    }
    if has_pred == Some(true) {
        out.pred_ret = Some(pred);
    }
    for (i, p) in out.probs.iter().enumerate() {
        let ok = p.iter().all(|x| x.is_finite() && *x >= 0.0)
            && (p.iter().sum::<f64>() - 1.0).abs() <= 1e-4;
        if !ok {
            return Err(DataError::Invalid(format!(
                "row {i}: bad probabilities {p:?}"
            )));
        }
    }
    if out.index().len() != out.len() {
        return Err(DataError::Invalid("duplicate (s, t) rows".into()));
    }
    Ok(out)
}

/// Writes predictions in the layout [`read_predictions`] reads.
pub fn write_predictions(path: impl AsRef<Path>, preds: &Predictions) -> Result<()> {
    let p = |k: usize| -> ArrayRef {
        Arc::new(Float64Array::from_iter_values(
            preds.probs.iter().map(|r| r[k]),
        ))
    };
    let mut cols: Vec<(&str, ArrayRef)> = vec![
        ("s", Arc::new(UInt32Array::from(preds.s.clone()))),
        ("t", Arc::new(UInt32Array::from(preds.t.clone()))),
        ("p_buy", p(0)),
        ("p_sell", p(1)),
        ("p_hold", p(2)),
    ];
    if let Some(r) = &preds.pred_ret {
        cols.push(("pred_ret", Arc::new(Float64Array::from(r.clone()))));
    }
    let batch = RecordBatch::try_from_iter(cols)?;
    write_batch(path.as_ref(), &batch)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn u32_cast_rejects_negative() {
        let batch = RecordBatch::try_from_iter([(
            "s",
            Arc::new(arrow::array::Int64Array::from(vec![1i64, -1])) as ArrayRef,
        )])
        .unwrap();
        assert!(u32_column(&batch, "s").is_err());
        let ok = RecordBatch::try_from_iter([(
            "s",
            Arc::new(arrow::array::Int64Array::from(vec![1i64, 7])) as ArrayRef,
        )])
        .unwrap();
        assert_eq!(u32_column(&ok, "s").unwrap(), vec![1, 7]);
    }

    #[test]
    fn nullable_float_column_reads_nan() {
        let batch = RecordBatch::try_from_iter([(
            "x",
            Arc::new(Float64Array::from(vec![Some(1.0), None])) as ArrayRef,
        )])
        .unwrap();
        let v = f64_column(&batch, "x").unwrap();
        assert_eq!(v[0], 1.0);
        assert!(v[1].is_nan());
        assert!(batch.column(0).is_null(1));
    }
}
