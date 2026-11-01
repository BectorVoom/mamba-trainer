//! The walk-forward CV runner.
//!
//! Per fold: normalise on training days, purge, run the leakage guard. Per model:
//! train every hyperparameter candidate on `train`, early-stop and select on the
//! `val` log loss; fit the calibrator and choose the decision threshold on `cal`;
//! predict every input-valid `test` day. Nothing fitted ever sees `test`.
//! Outputs go to `<out>/fold_<year>/<model>/` and resume: a model whose
//! `model.json` exists (written last) is skipped.

use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use clm_data::{Action, SampleIndex, read_predictions};
use clm_eval::backtest::{backtest_metrics, run_backtest};
use clm_eval::calibration::{CalibrationMethod, Calibrator};
use clm_eval::folds::{Fold, Segment, SegmentKind, make_folds, render_folds_line};
use clm_eval::oof::{FoldMetadata, OofRecord, read_json, write_json, write_oof_parquet};
use clm_eval::purging::check_folds_disjoint;
use clm_model::{
    DecisionModel, FitReport, PanelSource, Split, SplitPredictions, fit, predict_split, softmax3,
};
use mamba3::backend::Device;
use mamba3::backends::Auto;
use mamba3::train::checkpoint::Checkpoint;
use serde::{Deserialize, Serialize};

use crate::config::anyhow_like::{Result, err, wrap};
use crate::config::{ExperimentConfig, GridPoint, ModelSpec};
use crate::data::{Dataset, FoldData, prepare_fold};

type R = Auto;

/// What one model did on one fold.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FoldModelResult {
    /// Model name.
    pub model: String,
    /// Test year.
    pub test_year: i32,
    /// Candidates tried and their validation log loss.
    pub candidates: Vec<(GridPoint, f64)>,
    /// Index of the kept candidate.
    pub chosen: usize,
    /// Seed of the kept candidate.
    pub seed: u64,
    /// Training report of the kept candidate (internal models).
    pub fit: Option<FitReport>,
    /// Calibrator fitted on `cal` (applied to the test scores).
    pub calibrator: Calibrator,
    /// Decision threshold chosen on `cal`.
    pub tau: f64,
    /// Calibration-segment Sharpe per threshold candidate.
    pub tau_sharpe: Vec<(f64, f64)>,
    /// Calibration-segment log loss before and after calibration.
    pub cal_nll: (f64, f64),
    /// Number of parameters (internal models).
    pub num_params: usize,
    /// Seconds spent on this model and fold.
    pub seconds: f64,
}

/// The decision rule feeding the C5 policy: trade on the calibrated edge
/// `p_buy - p_sell`. Above `tau` be long (BUY), below `-tau` be flat (SELL),
/// otherwise keep the position (HOLD). A large HOLD probability shrinks the edge
/// rather than vetoing the trade: with HOLD as the most frequent class, a veto
/// left every model holding whatever it bought first, i.e. buy and hold.
pub fn decide(p: [f32; 3], tau: f64) -> Action {
    let edge = p[0] as f64 - p[1] as f64;
    if edge > tau {
        Action::Buy
    } else if edge < -tau {
        Action::Sell
    } else {
        Action::Hold
    }
}

fn git_commit() -> String {
    let rev = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".into());
    let dirty = std::process::Command::new("git")
        .args(["status", "--porcelain", "--untracked-files=no"])
        .output()
        .ok()
        .is_some_and(|o| !o.stdout.is_empty());
    if dirty { format!("{rev}+dirty") } else { rev }
}

/// Run every fold and model of `cfg` into `out`.
pub fn run(cfg: &ExperimentConfig, out: &Path, only: &[String], years: &[i32]) -> Result<()> {
    std::fs::create_dir_all(out).map_err(wrap("creating output dir"))?;
    let device = Device::<R>::default();
    let ds = Dataset::load(cfg)?;
    let n = ds.dates.len();
    let fold_cfg = cfg.fold_config();
    let folds = make_folds(&ds.years, n, &fold_cfg);
    check_folds_disjoint(&folds).map_err(|e| err(format!("folds overlap: {e}")))?;
    let commit = git_commit();
    let signature = cfg.signature(&ds.version);
    // A run directory belongs to one signature: never resume with other config,
    // data or code (the report would mix them).
    if let Ok(prev) = read_json::<serde_json::Value>(out.join("run.json")) {
        let old = prev.get("signature").and_then(|v| v.as_str()).unwrap_or("");
        if old != signature {
            return Err(err(format!(
                "{} holds a run with signature {old:?}, this one is {signature:?} (config, data or source changed); use a new --out",
                out.display()
            )));
        }
    }
    let run_meta = serde_json::json!({
        "signature": signature,
        "source_hash": env!("CLM_SOURCE_HASH"),
        "config": cfg,
        "dataset_version": ds.version,
        "git_commit": commit,
        "software_version": env!("CARGO_PKG_VERSION"),
        "device": format!("{:?}", device.inner()),
        "folds": folds.iter().map(render_folds_line).collect::<Vec<_>>(),
        "class_counts": ds.labels.class_counts(),
    });
    write_json(out.join("run.json"), &run_meta).map_err(wrap("run.json"))?;
    eprintln!("{} folds; dataset {}; commit {}", folds.len(), ds.version, commit);

    for (k, fold) in folds.iter().enumerate() {
        if !years.is_empty() && !years.contains(&fold.test_year) {
            continue;
        }
        let fdir = out.join(format!("fold_{}", fold.test_year));
        std::fs::create_dir_all(&fdir).map_err(wrap("fold dir"))?;
        let fd = prepare_fold(&ds, fold, cfg)?;
        let meta = FoldMetadata::new(k, fold, &fold_cfg, &ds.dates, cfg.seed, &ds.version, &commit);
        write_json(fdir.join("fold.json"), &meta).map_err(wrap("fold.json"))?;
        write_json(fdir.join("normalizer.json"), &fd.normalizer).map_err(wrap("normalizer.json"))?;
        let sizes: HashMap<String, usize> = fd.splits.iter().map(|(k, v)| (format!("{k:?}"), v.len())).collect();
        write_json(fdir.join("splits.json"), &sizes).map_err(wrap("splits.json"))?;
        eprintln!("fold {} ({}): {:?}", k, fold.test_year, sizes);
        for spec in &cfg.models {
            if !only.is_empty() && !only.contains(&spec.name) {
                continue;
            }
            let mdir = fdir.join(&spec.name);
            if mdir.join("model.json").exists() {
                eprintln!("  {}: done, skipping", spec.name);
                continue;
            }
            std::fs::create_dir_all(&mdir).map_err(wrap("model dir"))?;
            let started = Instant::now();
            let (cal_preds, test_preds, mut result) = if spec.is_internal() {
                train_model(cfg, spec, &ds, fold, k, &fd, &mdir, &device)?
            } else {
                external_model(spec, fold, &fd, &crate::data::fold_manifest(&ds, fold, &fd))?
            };
            finish_model(cfg, &ds, fold, k, &fd, &cal_preds, &test_preds, &mut result, &mdir)?;
            result.seconds = started.elapsed().as_secs_f64();
            write_json(mdir.join("model.json"), &result).map_err(wrap("model.json"))?;
            eprintln!(
                "  {}: chosen {:?} val_nll {:.4} cal_nll {:.4}->{:.4} tau {} ({:.0}s)",
                spec.name,
                result.candidates[result.chosen].0,
                result.candidates[result.chosen].1,
                result.cal_nll.0,
                result.cal_nll.1,
                result.tau,
                result.seconds
            );
        }
    }
    Ok(())
}

/// Predictions of one split, with the samples they belong to.
pub struct Preds {
    /// Samples in prediction order.
    pub samples: Vec<SampleIndex>,
    /// Raw scores (logits).
    pub scores: Vec<[f32; 3]>,
    /// Predicted vol-scaled return.
    pub ret: Option<Vec<f32>>,
}

#[allow(clippy::too_many_arguments)]
fn train_model(
    cfg: &ExperimentConfig,
    spec: &ModelSpec,
    ds: &Dataset,
    fold: &Fold,
    k: usize,
    fd: &FoldData,
    mdir: &Path,
    device: &Device<R>,
) -> Result<(Preds, Preds, FoldModelResult)> {
    let nf = ds.features.num_features;
    let mut best: Option<(f64, usize, u64, DecisionModel<R, f32>, FitReport)> = None;
    let mut candidates = Vec::new();
    let mut log = std::fs::File::create(mdir.join("train_log.jsonl")).map_err(wrap("train log"))?;
    for (gi, g) in spec.candidates().iter().enumerate() {
        // The same seed for every model of a fold and candidate index: differences
        // between models are not differences between seeds.
        let seed = cfg
            .seed
            .wrapping_add(1000 * k as u64)
            .wrapping_add(gi as u64)
            .wrapping_add(1_000_003 * spec.seed_offset);
        let model = DecisionModel::<R, f32>::new(&spec.model_config(cfg, nf, g, seed), device)
            .map_err(wrap("building model"))?;
        let mut src = PanelSource::new(&fd.normalized, &ds.labels, fd.splits.clone(), spec.source_config(cfg, seed));
        let report = fit(&model, &mut src, &spec.train_config(cfg, g), device, &mut |p| {
            let line = serde_json::json!({"candidate": gi, "point": p});
            let _ = writeln!(log, "{line}");
        })
        .map_err(|e| err(format!("{} fold {} candidate {gi}: {e}", spec.name, fold.test_year)))?;
        candidates.push((g.clone(), report.best_val_nll));
        if best.as_ref().is_none_or(|b| report.best_val_nll < b.0) {
            best = Some((report.best_val_nll, gi, seed, model, report));
        }
    }
    let (_, chosen, seed, model, report) = best.expect("at least one candidate");
    Checkpoint::capture(&model, report.best_step)
        .with_metadata(serde_json::json!({"model": spec.name, "test_year": fold.test_year, "seed": seed, "config": model.config()}))
        .save(mdir.join("model.ckpt"))
        .map_err(wrap("checkpoint"))?;
    let src = PanelSource::new(&fd.normalized, &ds.labels, fd.splits.clone(), spec.source_config(cfg, seed));
    let eb = cfg.train.eval_batch;
    let to_preds = |split: Split, p: SplitPredictions| Preds {
        samples: src.samples(split).to_vec(),
        scores: p.scores,
        ret: p.ret,
    };
    let cal = to_preds(Split::Cal, predict_split(&model, &src, Split::Cal, eb, device).map_err(wrap("predict cal"))?);
    let test = to_preds(Split::Test, predict_split(&model, &src, Split::Test, eb, device).map_err(wrap("predict test"))?);
    let num_params = model.num_params();
    Ok((
        cal,
        test,
        FoldModelResult {
            model: spec.name.clone(),
            test_year: fold.test_year,
            candidates,
            chosen,
            seed,
            fit: Some(report),
            calibrator: Calibrator::Identity,
            tau: 0.0,
            tau_sharpe: vec![],
            cal_nll: (f64::NAN, f64::NAN),
            num_params,
            seconds: 0.0,
        },
    ))
}

fn external_model(
    spec: &ModelSpec,
    fold: &Fold,
    fd: &FoldData,
    manifest: &serde_json::Value,
) -> Result<(Preds, Preds, FoldModelResult)> {
    let dir: PathBuf = spec
        .external_dir
        .clone()
        .expect("validated")
        .join(format!("fold_{}", fold.test_year));
    // Provenance: the external program must echo this fold's manifest exactly.
    let meta: serde_json::Value = read_json(dir.join("meta.json")).map_err(wrap("external meta.json"))?;
    if meta.get("manifest") != Some(manifest) {
        return Err(err(format!(
            "{}: {} was not produced from this run's data/fold/splits (manifest mismatch)",
            spec.name,
            dir.display()
        )));
    }
    let load = |split: Split, file: &str| -> Result<Preds> {
        let p = read_predictions(dir.join(file)).map_err(wrap(file))?;
        let by_key: HashMap<(u32, u32), usize> = p.s.iter().zip(&p.t).enumerate().map(|(i, (&s, &t))| ((s, t), i)).collect();
        let samples = fd.splits[&split].clone();
        if p.s.len() != samples.len() {
            return Err(err(format!(
                "{}: {file} has {} rows, the split has {} samples",
                spec.name,
                p.s.len(),
                samples.len()
            )));
        }
        let mut scores = Vec::with_capacity(samples.len());
        let mut ret = p.pred_ret.as_ref().map(|_| Vec::with_capacity(samples.len()));
        for smp in &samples {
            let i = *by_key.get(&(smp.s, smp.t)).ok_or_else(|| {
                err(format!("{}: {file} lacks sample s={} t={}", spec.name, smp.s, smp.t))
            })?;
            scores.push(p.probs[i].map(|v| (v.max(1e-9)).ln() as f32));
            if let (Some(r), Some(src)) = (ret.as_mut(), p.pred_ret.as_ref()) {
                r.push(src[i] as f32);
            }
        }
        Ok(Preds { samples, scores, ret })
    };
    let cal = load(Split::Cal, "pred_cal.parquet")?;
    let test = load(Split::Test, "pred_test.parquet")?;
    let candidates: Vec<(GridPoint, f64)> = meta
        .get("candidate_val_nll")
        .and_then(|v| v.as_array())
        .map(|a| a.iter().map(|x| (GridPoint::default(), x.as_f64().unwrap_or(f64::NAN))).collect())
        .ok_or_else(|| err(format!("{}: meta.json lacks candidate_val_nll", spec.name)))?;
    let chosen = meta.get("candidate").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    Ok((
        cal,
        test,
        FoldModelResult {
            model: spec.name.clone(),
            test_year: fold.test_year,
            candidates,
            chosen,
            seed: 0,
            fit: None,
            calibrator: Calibrator::Identity,
            tau: 0.0,
            tau_sharpe: vec![],
            cal_nll: (f64::NAN, f64::NAN),
            num_params: 0,
            seconds: 0.0,
        },
    ))
}

/// Calibrate and choose the threshold on `cal`, write the out-of-fold records.
#[allow(clippy::too_many_arguments)]
fn finish_model(
    cfg: &ExperimentConfig,
    ds: &Dataset,
    fold: &Fold,
    k: usize,
    fd: &FoldData,
    cal: &Preds,
    test: &Preds,
    result: &mut FoldModelResult,
    mdir: &Path,
) -> Result<()> {
    let cal_labels: Vec<u8> = cal
        .samples
        .iter()
        .map(|s| ds.labels.class_of(s.security(), s.day()).expect("cal samples are labelled").id() as u8)
        .collect();
    let calibrator = Calibrator::fit(CalibrationMethod::Temperature, &cal.scores, &cal_labels);
    let raw_p: Vec<[f32; 3]> = cal.scores.iter().map(|s| softmax3(*s, 1.0)).collect();
    let cal_p = calibrator.apply(&cal.scores);
    result.cal_nll = (
        clm_eval::metrics::log_loss(&raw_p, &cal_labels),
        clm_eval::metrics::log_loss(&cal_p, &cal_labels),
    );
    // Threshold: the backtest on the kept part of `cal` only, so no price it reads
    // lies in the test segment (kept: t + 1 + h + e < test start).
    let keep = fold.kept(SegmentKind::Cal);
    let seg = Segment::new(fold.cal.start, keep.end);
    let opens = ds.opens();
    let ns = ds.panel.stocks.len();
    let mut tau_sharpe = Vec::new();
    for &tau in &cfg.tau_grid {
        let mut decisions = vec![vec![None; seg.len()]; ns];
        for (smp, p) in cal.samples.iter().zip(&cal_p) {
            if seg.contains(smp.day()) {
                decisions[smp.security()][smp.day() - seg.start] = Some(decide(*p, tau));
            }
        }
        let sharpe = if seg.is_empty() {
            0.0
        } else {
            let bt = run_backtest(&opens, seg, ds.dates.len(), &decisions, cfg.kappa()).map_err(wrap("cal backtest"))?;
            backtest_metrics(&bt.daily).sharpe
        };
        tau_sharpe.push((tau, sharpe));
    }
    let tau = tau_sharpe
        .iter()
        .fold((f64::NEG_INFINITY, 0.0), |best, &(t, s)| if s > best.0 { (s, t) } else { best })
        .1;
    result.calibrator = calibrator.clone();
    result.tau = tau;
    result.tau_sharpe = tau_sharpe;

    let record = |smp: &SampleIndex, scores: [f32; 3], calibrated: [f32; 3], ret: Option<f32>| {
        let (s, t) = (smp.security(), smp.day());
        OofRecord {
            fold: k as u32,
            test_year: fold.test_year,
            s: smp.s,
            t: smp.t,
            date: ds.dates[t].clone(),
            symbol: ds.panel.stocks[s].symbol.clone(),
            probs: softmax3(scores, 1.0),
            logits: scores,
            calibrated,
            pred_ret: ret,
            label: ds.labels.class_of(s, t).map(|a| a.id() as u8),
            fwd_ret: if ds.labels.exists(s, t) { ds.labels.fwd_ret_of(s, t) as f32 } else { f32::NAN },
        }
    };
    let test_cal = calibrator.apply(&test.scores);
    let recs: Vec<OofRecord> = test
        .samples
        .iter()
        .enumerate()
        .map(|(i, smp)| record(smp, test.scores[i], test_cal[i], test.ret.as_ref().map(|r| r[i])))
        .collect();
    write_oof_parquet(mdir.join("oof_test.parquet"), &recs).map_err(wrap("oof_test"))?;
    let recs: Vec<OofRecord> = cal
        .samples
        .iter()
        .enumerate()
        .map(|(i, smp)| record(smp, cal.scores[i], cal_p[i], cal.ret.as_ref().map(|r| r[i])))
        .collect();
    write_oof_parquet(mdir.join("oof_cal.parquet"), &recs).map_err(wrap("oof_cal"))?;
    let _ = fd;
    Ok(())
}
