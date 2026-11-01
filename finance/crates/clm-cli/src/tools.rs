//! `export-lgbm`, `infer` and `bench`.

use std::path::Path;
use std::time::Instant;

use clm_data::write_samples;
use clm_eval::calibration::Calibrator;
use clm_eval::folds::make_folds;
use clm_eval::oof::{read_json, write_json};
use clm_model::{DecisionModel, ModelConfig, Split, TrainConfig, input_valid, softmax3};
use mamba3::autograd::Var;
use mamba3::autograd::grad_mode::with_no_grad;
use mamba3::backend::Device;
use mamba3::backends::Auto;
use mamba3::ssm::selective::{SelectiveDims, SelectiveInputs, selective_scan_reference};
use mamba3::tensor::Tensor;
use mamba3::tensor::ops::random::Rng;
use mamba3::tensor::ops::selective_scan::{SelectiveScanMode, selective_scan_backward, selective_scan_saving};
use mamba3::train::checkpoint::Checkpoint;
use serde::Serialize;

use crate::config::ExperimentConfig;
use crate::config::anyhow_like::{Result, err, wrap};
use crate::cv::{FoldModelResult, decide};
use crate::data::{Dataset, prepare_fold};

type R = Auto;

/// Write every fold's samples (last-row features, normalised per fold) for an
/// external baseline: `<out>/fold_<year>/{train,val,cal,test}.parquet`.
pub fn export_lgbm(cfg: &ExperimentConfig, out: &Path) -> Result<()> {
    let ds = Dataset::load(cfg)?;
    let folds = make_folds(&ds.years, ds.dates.len(), &cfg.fold_config());
    for fold in &folds {
        let fd = prepare_fold(&ds, fold, cfg)?;
        let dir = out.join(format!("fold_{}", fold.test_year));
        std::fs::create_dir_all(&dir).map_err(wrap("export dir"))?;
        for (name, split) in [("train", Split::Train), ("val", Split::Val), ("cal", Split::Cal), ("test", Split::Test)] {
            write_samples(dir.join(format!("{name}.parquet")), &ds.panel, &fd.normalized, &ds.labels, &fd.splits[&split])
                .map_err(wrap(name))?;
        }
        write_json(dir.join("manifest.json"), &crate::data::fold_manifest(&ds, fold, &fd)).map_err(wrap("manifest"))?;
        eprintln!("fold {}: exported", fold.test_year);
    }
    write_json(out.join("export.json"), &serde_json::json!({"dataset_version": ds.version, "config": cfg}))
        .map_err(wrap("export.json"))?;
    Ok(())
}

/// Decisions for every stock on `date` with the model trained for the fold whose
/// test year contains it (or the last fold).
pub fn infer(cfg: &ExperimentConfig, run: &Path, model_name: &str, date: &str) -> Result<()> {
    let ds = Dataset::load(cfg)?;
    let t = ds
        .dates
        .iter()
        .position(|d| d == date)
        .ok_or_else(|| err(format!("{date} is not a trading day in the data")))?;
    let folds = make_folds(&ds.years, ds.dates.len(), &cfg.fold_config());
    let fold = folds
        .iter()
        .find(|f| f.test.contains(t))
        .or(folds.last())
        .ok_or_else(|| err("no folds"))?;
    if t < fold.test.start {
        return Err(err(format!("{date} precedes the first test segment; its fold's model may have seen it")));
    }
    let dir = run.join(format!("fold_{}", fold.test_year)).join(model_name);
    let result: FoldModelResult = read_json(dir.join("model.json")).map_err(wrap("model.json"))?;
    let ckpt = Checkpoint::load(dir.join("model.ckpt")).map_err(wrap("checkpoint"))?;
    let mcfg: ModelConfig = serde_json::from_value(ckpt.metadata.get("config").cloned().ok_or_else(|| err("checkpoint has no config"))?)
        .map_err(wrap("model config"))?;
    let device = Device::<R>::default();
    let model = DecisionModel::<R, f32>::new(&mcfg, &device).map_err(wrap("model"))?;
    ckpt.restore(&model, true).map_err(wrap("restore"))?;
    let normalizer: clm_data::Normalizer = read_json(run.join(format!("fold_{}", fold.test_year)).join("normalizer.json"))
        .map_err(wrap("normalizer"))?;
    let normalized = normalizer.apply(&ds.features);
    let samples: Vec<clm_data::SampleIndex> = (0..ds.panel.stocks.len())
        .filter(|&s| input_valid(&ds.features, s, t, cfg.window))
        .map(|s| clm_data::SampleIndex::new(s, t))
        .collect();
    if samples.is_empty() {
        return Err(err("no stock has a valid input window on that day"));
    }
    let x = clm_data::gather_windows(&normalized, &samples, cfg.window);
    let scores = with_no_grad(|| -> Result<Vec<f32>> {
        let xv = Var::constant(
            Tensor::<R, f32>::from_f32(&x, vec![samples.len(), cfg.window, ds.features.num_features], &device)
                .map_err(wrap("input"))?,
        );
        model.forward(&xv).map_err(wrap("forward"))?.scores.try_to_f32().map_err(wrap("read"))
    })?;
    let cal: &Calibrator = &result.calibrator;
    println!("decision after the close of {date}, executed at the next open (model {model_name}, fold {})", fold.test_year);
    println!("{:<6} {:>7} {:>7} {:>7}  {:<5}", "symbol", "p_buy", "p_sell", "p_hold", "action");
    for (i, smp) in samples.iter().enumerate() {
        let s = [scores[3 * i], scores[3 * i + 1], scores[3 * i + 2]];
        let p = cal.apply_row(&s).map(|v| v as f32);
        println!(
            "{:<6} {:>7.3} {:>7.3} {:>7.3}  {:<5}",
            ds.panel.stocks[smp.security()].symbol,
            p[0],
            p[1],
            p[2],
            decide(p, result.tau).name()
        );
        let _ = softmax3;
    }
    Ok(())
}

#[derive(Serialize)]
struct BenchRow {
    seq: usize,
    mode: String,
    forward_ms: f64,
    backward_ms: f64,
    checkpoint_mb: f64,
    max_rel_err: f64,
}

/// Selective-scan kernel benchmarks over sequence lengths.
pub fn bench(out: &Path, batch: usize, dim: usize, state: usize, reps: usize, ab_flush: bool) -> Result<()> {
    let device = Device::<R>::default();
    let mut rows = Vec::new();
    for seq in [64usize, 128, 256, 512, 1024] {
        let mut rng = Rng::seeded(seq as u64);
        let act = batch * seq * dim;
        let bc = batch * seq * state;
        let x = rng.normal_vec(act, 0.0, 1.0);
        let dt = rng.uniform_vec(act, 0.001, 0.1);
        let a = rng.uniform_vec(dim * state, -8.0, -0.5);
        let bm = rng.normal_vec(bc, 0.0, 1.0);
        let cm = rng.normal_vec(bc, 0.0, 1.0);
        let skip = rng.normal_vec(dim, 0.0, 1.0);
        let t = |v: &[f32], s: Vec<usize>| Tensor::<R, f32>::from_f32(v, s, &device).map_err(wrap("tensor"));
        let (xt, dtt, at, bt, ct, st) = (
            t(&x, vec![batch, seq, dim])?,
            t(&dt, vec![batch, seq, dim])?,
            t(&a, vec![dim, state])?,
            t(&bm, vec![batch, seq, state])?,
            t(&cm, vec![batch, seq, state])?,
            t(&skip, vec![dim])?,
        );
        let dy = t(&rng.normal_vec(act, 0.0, 1.0), vec![batch, seq, dim])?;
        // Reference on the first batch row only (the f64 loop is slow).
        let row = |v: &[f32], n: usize| v[..n].iter().map(|&x| x as f64).collect::<Vec<f64>>();
        let want = selective_scan_reference(
            SelectiveDims { batch: 1, seq, dim, state },
            &SelectiveInputs {
                x: &row(&x, seq * dim),
                dt: &row(&dt, seq * dim),
                a: &row(&a, dim * state),
                bm: &row(&bm, seq * state),
                cm: &row(&cm, seq * state),
                skip: Some(&row(&skip, dim)),
            },
        );
        let scale = want.iter().fold(1e-12f64, |m, v| m.max(v.abs()));
        type Variant = (&'static str, SelectiveScanMode, Option<usize>, Option<usize>);
        let variants: Vec<Variant> = if ab_flush {
            // Interleaved A/B of segment length and reduction interval, same process.
            (0..3)
                .flat_map(|_| {
                    [
                        ("seg8f8", SelectiveScanMode::Sequential, Some(8), Some(8)),
                        ("seg4f4", SelectiveScanMode::Sequential, Some(4), Some(4)),
                        ("seg4f2", SelectiveScanMode::Sequential, Some(4), Some(2)),
                        ("seg2f2", SelectiveScanMode::Sequential, Some(2), Some(2)),
                    ]
                })
                .collect()
        } else {
            vec![
                ("sequential", SelectiveScanMode::Sequential, None, None),
                ("chunked64", SelectiveScanMode::Chunked(64), None, None),
            ]
        };
        for (name, mode, seg_override, flush) in variants {
            mamba3::tensor::ops::selective_scan::set_segment_len(seg_override);
            mamba3::tensor::ops::selective_scan::set_backward_flush(flush);
            // Warm up (kernel compilation), then time.
            let (y, ck) = selective_scan_saving(&xt, &dtt, &at, &bt, &ct, Some(&st), mode, true).map_err(wrap("scan"))?;
            let _ = selective_scan_backward(&dy, &xt, &dtt, &at, &bt, &ct, Some(&st), ck.as_ref()).map_err(wrap("bwd"))?;
            device.synchronize();
            let got = y.to_f32();
            let err = got[..seq * dim].iter().zip(&want).map(|(g, w)| (*g as f64 - w).abs()).fold(0.0, f64::max) / scale;
            let start = Instant::now();
            let mut saved = None;
            for _ in 0..reps {
                saved = selective_scan_saving(&xt, &dtt, &at, &bt, &ct, Some(&st), mode, true).map_err(wrap("scan"))?.1;
            }
            device.synchronize();
            let fwd = start.elapsed().as_secs_f64() * 1e3 / reps as f64;
            let start = Instant::now();
            for _ in 0..reps {
                let _ = selective_scan_backward(&dy, &xt, &dtt, &at, &bt, &ct, Some(&st), saved.as_ref()).map_err(wrap("bwd"))?;
            }
            device.synchronize();
            let bwd = start.elapsed().as_secs_f64() * 1e3 / reps as f64;
            let seg = mamba3::tensor::ops::selective_scan::segment_len(device.client(), state);
            let ckpt_mb = (batch * seq.div_ceil(seg) * state * dim * 4) as f64 / 1e6;
            eprintln!("T={seq:5} {name:10} fwd {fwd:8.3} ms  bwd {bwd:8.3} ms  ckpt {ckpt_mb:6.1} MB  rel err {err:.2e}");
            mamba3::tensor::ops::selective_scan::set_backward_flush(None);
            mamba3::tensor::ops::selective_scan::set_segment_len(None);
            rows.push(BenchRow {
                seq,
                mode: name.into(),
                forward_ms: fwd,
                backward_ms: bwd,
                checkpoint_mb: ckpt_mb,
                max_rel_err: err,
            });
        }
    }
    std::fs::create_dir_all(out).map_err(wrap("bench dir"))?;
    write_json(
        out.join("scan_bench.json"),
        &serde_json::json!({"device": format!("{:?}", device.inner()), "batch": batch, "dim": dim, "state": state, "rows": rows}),
    )
    .map_err(wrap("bench json"))?;
    Ok(())
}

/// Training throughput and inference latency of whole models at the study's sizes
/// (synthetic inputs: speed does not depend on the values).
pub fn bench_model(cfg: &ExperimentConfig, out: &Path, steps: u64) -> Result<()> {
    use clm_model::{Batch, BatchSource};
    struct Fixed(Batch);
    impl BatchSource for Fixed {
        fn train_batch(&mut self, _: usize) -> Batch {
            self.0.clone()
        }
        fn eval_len(&self, _: Split) -> usize {
            self.0.size
        }
        fn eval_batch(&self, _: Split, index: usize, _: usize) -> Batch {
            if index == 0 { self.0.clone() } else { Batch::default() }
        }
    }
    let device = Device::<R>::default();
    let (w, f) = (cfg.window, clm_data::NUM_FEATURES);
    let make = |b: usize, seed: u64| {
        let mut rng = Rng::seeded(seed);
        let labels: Vec<u32> = (0..b).map(|i| (i % 3) as u32).collect();
        Batch {
            size: b,
            window: w,
            features: f,
            inputs: rng.normal_vec(b * w * f, 0.0, 1.0),
            positives: labels.iter().flat_map(|&l| (0..3).map(move |k| if k == l { 1.0 } else { 0.0 })).collect(),
            labels,
            weights: vec![1.0; b],
            ret_target: vec![0.0; b],
            stop_target: vec![0.0; b],
        }
    };
    let mut rows = Vec::new();
    for spec in cfg.models.iter().filter(|m| m.is_internal()) {
        let g = spec.candidates()[0].clone();
        let mcfg = spec.model_config(cfg, f, &g, 1);
        let model = DecisionModel::<R, f32>::new(&mcfg, &device).map_err(wrap("model"))?;
        let mut tc = spec.train_config(cfg, &g);
        tc.max_steps = steps;
        tc.warmup_steps = 1;
        tc.eval_every = steps;
        tc.patience = 1;
        tc.eval_batch = 64;
        let batch = make(tc.batch_size, 3);
        // Warm-up run (kernel compilation, autotuning), then the timed one.
        let _ = clm_model::fit(&model, &mut Fixed(batch.clone()), &TrainConfig { max_steps: 5.min(steps), ..tc.clone() }, &device, &mut |_| {});
        let report = clm_model::fit(&model, &mut Fixed(batch), &tc, &device, &mut |_| {}).map_err(wrap("fit"))?;
        let mut latency = Vec::new();
        for b in [1usize, 56, 1024] {
            let x = make(b, 4).inputs;
            let run = || -> Result<()> {
                with_no_grad(|| -> Result<()> {
                    let xv = Var::constant(Tensor::<R, f32>::from_f32(&x, vec![b, w, f], &device).map_err(wrap("x"))?);
                    model.forward(&xv).map_err(wrap("forward"))?.scores.try_to_f32().map_err(wrap("read"))?;
                    Ok(())
                })
            };
            for _ in 0..3 {
                run()?;
            }
            let reps = 20;
            let start = Instant::now();
            for _ in 0..reps {
                run()?;
            }
            latency.push((b, start.elapsed().as_secs_f64() * 1e3 / reps as f64));
        }
        eprintln!(
            "{:<18} params {:>8}  train {:>8.0} samples/s ({:.1} ms/step at batch {})  infer ms: b1 {:.2}  b56 {:.2}  b1024 {:.2}",
            spec.name,
            model.num_params(),
            report.samples_per_second,
            1e3 * tc.batch_size as f64 / report.samples_per_second,
            tc.batch_size,
            latency[0].1,
            latency[1].1,
            latency[2].1
        );
        rows.push(serde_json::json!({
            "model": spec.name, "params": model.num_params(), "batch": tc.batch_size,
            "train_samples_per_second": report.samples_per_second, "steps": report.steps,
            "infer_ms": latency.iter().map(|(b, ms)| serde_json::json!({"batch": b, "ms": ms})).collect::<Vec<_>>(),
        }));
    }
    std::fs::create_dir_all(out).map_err(wrap("bench dir"))?;
    write_json(out.join("model_bench.json"), &serde_json::json!({"device": format!("{:?}", device.inner()), "window": w, "rows": rows}))
        .map_err(wrap("model bench json"))?;
    Ok(())
}
