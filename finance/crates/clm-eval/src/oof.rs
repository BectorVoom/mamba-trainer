//! Out-of-fold prediction records (Parquet) and per-fold metadata (JSON).
//!
//! Parquet layout (one row per test sample, flat columns so Python reads it directly):
//!
//! | column | type |
//! |---|---|
//! | `fold` | UInt32 |
//! | `test_year` | Int32 |
//! | `s`, `t` | UInt32 |
//! | `date`, `symbol` | Utf8 |
//! | `p_buy`, `p_sell`, `p_hold` | Float32 |
//! | `logit_buy`, `logit_sell`, `logit_hold` | Float32 |
//! | `cal_buy`, `cal_sell`, `cal_hold` | Float32 |
//! | `pred_ret` | Float32, nullable |
//! | `label` | UInt8, nullable |
//! | `fwd_ret` | Float32 |

use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, Float32Array, Int32Array, StringArray, UInt8Array, UInt32Array,
};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use thiserror::Error;

use clm_data::types::NUM_ACTIONS;

use crate::folds::{Fold, FoldConfig, Segment};
use crate::metrics::check_prob_row;

/// I/O errors of OOF records and metadata.
#[derive(Debug, Error)]
pub enum OofError {
    /// File system error.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    /// Parquet error.
    #[error("parquet: {0}")]
    Parquet(#[from] parquet::errors::ParquetError),
    /// Arrow error.
    #[error("arrow: {0}")]
    Arrow(#[from] arrow::error::ArrowError),
    /// JSON error.
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    /// The file does not have the expected columns.
    #[error("schema: {0}")]
    Schema(String),
    /// A row violates the record invariants ([`OofRecord::validate`]).
    #[error("row {row}: {reason}")]
    Invalid {
        /// Row index in the batch / record slice.
        row: usize,
        /// What is wrong.
        reason: String,
    },
}

/// One out-of-fold prediction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OofRecord {
    /// Fold index.
    pub fold: u32,
    /// Test year of the fold.
    pub test_year: i32,
    /// Security index.
    pub s: u32,
    /// Decision day.
    pub t: u32,
    /// Calendar date of `t` (ISO `YYYY-MM-DD`).
    pub date: String,
    /// Ticker.
    pub symbol: String,
    /// Uncalibrated probabilities `[BUY, SELL, HOLD]`.
    pub probs: [f32; 3],
    /// Logits.
    pub logits: [f32; 3],
    /// Calibrated probabilities.
    pub calibrated: [f32; 3],
    /// Predicted (vol-scaled) return, if the model has a return head.
    pub pred_ret: Option<f32>,
    /// Class label, if it exists.
    pub label: Option<u8>,
    /// Realised forward log return `r_h(t)`.
    pub fwd_ret: f32,
}

impl OofRecord {
    /// Check the record invariants:
    ///
    /// * `label` is `None` or a class id `< 3`;
    /// * `probs` and `calibrated` are finite, in `[0, 1]` and sum to 1 within
    ///   [`crate::metrics::ROW_SUM_TOL`];
    /// * `logits` are finite; `pred_ret`, if present, is finite;
    /// * `fwd_ret` is finite **or NaN** (unlabelled test days carry NaN), never ±inf.
    pub fn validate(&self) -> Result<(), String> {
        if let Some(y) = self.label
            && (y as usize) >= NUM_ACTIONS
        {
            return Err(format!("label {y} is not a class id (0..3)"));
        }
        check_prob_row(&self.probs).map_err(|e| format!("probs: {e}"))?;
        check_prob_row(&self.calibrated).map_err(|e| format!("calibrated: {e}"))?;
        if let Some(k) = self.logits.iter().position(|v| !v.is_finite()) {
            return Err(format!("logit {k} is not finite ({})", self.logits[k]));
        }
        if let Some(p) = self.pred_ret
            && !p.is_finite()
        {
            return Err(format!("pred_ret is not finite ({p})"));
        }
        if self.fwd_ret.is_infinite() {
            return Err(format!("fwd_ret is infinite ({})", self.fwd_ret));
        }
        Ok(())
    }
}

fn validate_all(records: &[OofRecord]) -> Result<(), OofError> {
    for (row, r) in records.iter().enumerate() {
        r.validate()
            .map_err(|reason| OofError::Invalid { row, reason })?;
    }
    Ok(())
}

/// Columns that may hold nulls; every other column must not.
const NULLABLE: [&str; 2] = ["pred_ret", "label"];

const TRIPLES: [(&str, &str, &str); 3] = [
    ("p_buy", "p_sell", "p_hold"),
    ("logit_buy", "logit_sell", "logit_hold"),
    ("cal_buy", "cal_sell", "cal_hold"),
];

/// The Arrow schema of OOF files.
pub fn oof_schema() -> SchemaRef {
    let mut f = vec![
        Field::new("fold", DataType::UInt32, false),
        Field::new("test_year", DataType::Int32, false),
        Field::new("s", DataType::UInt32, false),
        Field::new("t", DataType::UInt32, false),
        Field::new("date", DataType::Utf8, false),
        Field::new("symbol", DataType::Utf8, false),
    ];
    for (a, b, c) in TRIPLES {
        for n in [a, b, c] {
            f.push(Field::new(n, DataType::Float32, false));
        }
    }
    f.push(Field::new("pred_ret", DataType::Float32, true));
    f.push(Field::new("label", DataType::UInt8, true));
    f.push(Field::new("fwd_ret", DataType::Float32, false));
    Arc::new(Schema::new(f))
}

/// Convert records to one Arrow record batch (validating every record first).
pub fn oof_to_batch(records: &[OofRecord]) -> Result<RecordBatch, OofError> {
    validate_all(records)?;
    let mut cols: Vec<ArrayRef> = vec![
        Arc::new(UInt32Array::from_iter_values(
            records.iter().map(|r| r.fold),
        )),
        Arc::new(Int32Array::from_iter_values(
            records.iter().map(|r| r.test_year),
        )),
        Arc::new(UInt32Array::from_iter_values(records.iter().map(|r| r.s))),
        Arc::new(UInt32Array::from_iter_values(records.iter().map(|r| r.t))),
        Arc::new(StringArray::from_iter_values(
            records.iter().map(|r| r.date.as_str()),
        )),
        Arc::new(StringArray::from_iter_values(
            records.iter().map(|r| r.symbol.as_str()),
        )),
    ];
    let pick: [fn(&OofRecord) -> [f32; 3]; 3] = [|r| r.probs, |r| r.logits, |r| r.calibrated];
    for get in pick {
        for k in 0..3 {
            cols.push(Arc::new(Float32Array::from_iter_values(
                records.iter().map(|r| get(r)[k]),
            )));
        }
    }
    cols.push(Arc::new(Float32Array::from(
        records.iter().map(|r| r.pred_ret).collect::<Vec<_>>(),
    )));
    cols.push(Arc::new(UInt8Array::from(
        records.iter().map(|r| r.label).collect::<Vec<_>>(),
    )));
    cols.push(Arc::new(Float32Array::from_iter_values(
        records.iter().map(|r| r.fwd_ret),
    )));
    Ok(RecordBatch::try_new(oof_schema(), cols)?)
}

fn column<'a, T: 'static>(batch: &'a RecordBatch, name: &str) -> Result<&'a T, OofError> {
    batch
        .column_by_name(name)
        .ok_or_else(|| OofError::Schema(format!("missing column {name}")))?
        .as_any()
        .downcast_ref::<T>()
        .ok_or_else(|| OofError::Schema(format!("column {name} has the wrong type")))
}

/// Convert an Arrow record batch (OOF schema) back to records. Fails on wrong column
/// types, nulls outside `pred_ret`/`label` (files written elsewhere may declare every
/// column nullable), and any row breaking [`OofRecord::validate`].
pub fn oof_from_batch(batch: &RecordBatch) -> Result<Vec<OofRecord>, OofError> {
    for field in oof_schema().fields() {
        let name = field.name().as_str();
        if NULLABLE.contains(&name) {
            continue;
        }
        let col = batch
            .column_by_name(name)
            .ok_or_else(|| OofError::Schema(format!("missing column {name}")))?;
        if col.null_count() > 0 {
            let row = (0..col.len()).find(|&i| col.is_null(i)).unwrap_or(0);
            return Err(OofError::Invalid {
                row,
                reason: format!("null in non-nullable column {name}"),
            });
        }
    }
    let fold = column::<UInt32Array>(batch, "fold")?;
    let year = column::<Int32Array>(batch, "test_year")?;
    let s = column::<UInt32Array>(batch, "s")?;
    let t = column::<UInt32Array>(batch, "t")?;
    let date = column::<StringArray>(batch, "date")?;
    let sym = column::<StringArray>(batch, "symbol")?;
    let mut trip: Vec<[&Float32Array; 3]> = Vec::new();
    for (a, b, c) in TRIPLES {
        trip.push([
            column::<Float32Array>(batch, a)?,
            column::<Float32Array>(batch, b)?,
            column::<Float32Array>(batch, c)?,
        ]);
    }
    let pred = column::<Float32Array>(batch, "pred_ret")?;
    let label = column::<UInt8Array>(batch, "label")?;
    let fwd = column::<Float32Array>(batch, "fwd_ret")?;
    let get3 = |c: &[&Float32Array; 3], i: usize| [c[0].value(i), c[1].value(i), c[2].value(i)];
    let records = (0..batch.num_rows())
        .map(|i| OofRecord {
            fold: fold.value(i),
            test_year: year.value(i),
            s: s.value(i),
            t: t.value(i),
            date: date.value(i).to_string(),
            symbol: sym.value(i).to_string(),
            probs: get3(&trip[0], i),
            logits: get3(&trip[1], i),
            calibrated: get3(&trip[2], i),
            pred_ret: pred.is_valid(i).then(|| pred.value(i)),
            label: label.is_valid(i).then(|| label.value(i)),
            fwd_ret: fwd.value(i),
        })
        .collect::<Vec<_>>();
    validate_all(&records)?;
    Ok(records)
}

/// Write records to a Parquet file (Snappy); fails without writing on an invalid
/// record.
pub fn write_oof_parquet(path: impl AsRef<Path>, records: &[OofRecord]) -> Result<(), OofError> {
    let batch = oof_to_batch(records)?;
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .build();
    let mut w = ArrowWriter::try_new(File::create(path)?, oof_schema(), Some(props))?;
    w.write(&batch)?;
    w.close()?;
    Ok(())
}

/// Read records from a Parquet file written by [`write_oof_parquet`] (or any file
/// with the same column names and types).
pub fn read_oof_parquet(path: impl AsRef<Path>) -> Result<Vec<OofRecord>, OofError> {
    let reader = ParquetRecordBatchReaderBuilder::try_new(File::open(path)?)?.build()?;
    let mut out = Vec::new();
    for batch in reader {
        out.extend(oof_from_batch(&batch?)?);
    }
    Ok(out)
}

/// A segment's day range and the dates of its first and last day.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SegmentDates {
    /// First day index.
    pub start: usize,
    /// One past the last day index.
    pub end: usize,
    /// Date of `start` (`None` if the segment is empty).
    pub first_date: Option<String>,
    /// Date of `end - 1` (`None` if the segment is empty).
    pub last_date: Option<String>,
}

impl SegmentDates {
    /// Dates of a segment from the calendar `dates` (missing entries give `None`).
    pub fn new(seg: Segment, dates: &[String]) -> Self {
        let (first_date, last_date) = if seg.is_empty() {
            (None, None)
        } else {
            (
                dates.get(seg.start).cloned(),
                dates.get(seg.end - 1).cloned(),
            )
        };
        Self {
            start: seg.start,
            end: seg.end,
            first_date,
            last_date,
        }
    }
}

/// Everything needed to reproduce one fold of a run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FoldMetadata {
    /// Fold index.
    pub fold: usize,
    /// Test year.
    pub test_year: i32,
    /// Calendar length.
    pub n_days: usize,
    /// Training segment.
    pub train: SegmentDates,
    /// Validation segment.
    pub val: SegmentDates,
    /// Calibration segment.
    pub cal: SegmentDates,
    /// Test segment.
    pub test: SegmentDates,
    /// Kept decision days of `[train, val, cal, test]`.
    pub keep: [SegmentDates; 4],
    /// The fold configuration.
    pub config: FoldConfig,
    /// Random seed of the run.
    pub seed: u64,
    /// Version string of the dataset.
    pub dataset_version: String,
    /// Version of this software (`clm-eval <crate version>`).
    pub software_version: String,
    /// Git commit given by the caller.
    pub git_commit: String,
}

/// The software version recorded in metadata.
pub const SOFTWARE_VERSION: &str = concat!("clm-eval ", env!("CARGO_PKG_VERSION"));

impl FoldMetadata {
    /// Metadata of `fold` (index `index`) with segment dates from `dates[t]`.
    pub fn new(
        index: usize,
        fold: &Fold,
        config: &FoldConfig,
        dates: &[String],
        seed: u64,
        dataset_version: &str,
        git_commit: &str,
    ) -> Self {
        Self {
            fold: index,
            test_year: fold.test_year,
            n_days: fold.n_days,
            train: SegmentDates::new(fold.train, dates),
            val: SegmentDates::new(fold.val, dates),
            cal: SegmentDates::new(fold.cal, dates),
            test: SegmentDates::new(fold.test, dates),
            keep: fold.keep.map(|k| SegmentDates::new(k, dates)),
            config: *config,
            seed,
            dataset_version: dataset_version.to_string(),
            software_version: SOFTWARE_VERSION.to_string(),
            git_commit: git_commit.to_string(),
        }
    }
}

/// Write any serializable value as pretty JSON.
pub fn write_json<T: Serialize>(path: impl AsRef<Path>, value: &T) -> Result<(), OofError> {
    let f = std::io::BufWriter::new(File::create(path)?);
    serde_json::to_writer_pretty(f, value)?;
    Ok(())
}

/// Read a JSON value written by [`write_json`].
pub fn read_json<T: DeserializeOwned>(path: impl AsRef<Path>) -> Result<T, OofError> {
    let f = std::io::BufReader::new(File::open(path)?);
    Ok(serde_json::from_reader(f)?)
}
