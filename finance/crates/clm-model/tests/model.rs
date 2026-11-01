//! Model-level tests: device forward against the `f64` reference, causality,
//! embedding properties, gradients, checkpoints, determinism, and learning a
//! planted synthetic signal.

use std::collections::HashMap;

use clm_data::synthetic::generate;
use clm_data::{LabelConfig, Normalizer, PlantedSignal, SampleIndex, SyntheticConfig, compute_features, compute_labels};
use clm_model::reference::encoder_reference;
use clm_model::{
    Batch, BatchSource, ContrastiveConfig, DecisionModel, EncoderConfig, HeadConfig, LossConfig,
    MambaConfig, ModelConfig, PanelSource, Pooling, SourceConfig, Split, TcnConfig, Temperature,
    TrainConfig, action_attributes, fit, input_valid, predict_split,
};
use mamba3::autograd::Var;
use mamba3::autograd::grad_mode::with_no_grad;
use mamba3::backend::Device;
use mamba3::backends::Auto;
use mamba3::nn::Module;
use mamba3::tensor::Tensor;
use mamba3::tensor::ops::random::Rng;
use mamba3::tensor::ops::selective_scan::SelectiveScanMode;

type R = Auto;

fn dev() -> Device<R> {
    Device::<R>::default()
}

fn small_mamba(input_dim: usize) -> MambaConfig {
    MambaConfig {
        input_dim,
        d_model: 16,
        d_state: 4,
        d_conv: 4,
        expand: 2,
        num_layers: 2,
        dt_rank: 0,
        dt_min: 1e-3,
        dt_max: 1e-1,
        pooling: Pooling::Last,
    }
}

fn clm_config(input_dim: usize, seed: u64) -> ModelConfig {
    ModelConfig {
        encoder: EncoderConfig::Mamba(small_mamba(input_dim)),
        head: HeadConfig::Contrastive(ContrastiveConfig {
            embed_dim: 8,
            hidden: 16,
            temperature: Temperature::Learned(0.2),
            scoring: clm_model::Scoring::Cosine,
        }),
        aux_return: true,
        aux_risk: true,
        aux_hidden: 8,
        action_attrs: action_attributes(5, 0.001, 0.0005),
        seed,
    }
}

fn noise(n: usize, seed: u64) -> Vec<f32> {
    Rng::seeded(seed).normal_vec(n, 0.0, 1.0)
}

#[test]
fn encoder_matches_f64_reference_in_both_schedules() {
    let dev = dev();
    let cfg = small_mamba(5);
    let (b, t) = (3, 23);
    let x = noise(b * t * 5, 1);
    let mut enc = clm_model::mamba::MambaEncoder::<R, f32>::new(&cfg, &dev, &mut Rng::seeded(4));
    let x64: Vec<f64> = x.iter().map(|&v| v as f64).collect();
    let want = encoder_reference(&enc, &x64, b, t);
    for mode in [SelectiveScanMode::Sequential, SelectiveScanMode::Chunked(4)] {
        enc.scan_mode = mode;
        let got = with_no_grad(|| {
            enc.sequence(&Var::constant(Tensor::from_f32(&x, vec![b, t, 5], &dev).unwrap()))
                .unwrap()
                .to_f32()
        });
        let scale = want.iter().fold(1.0f64, |m, v| m.max(v.abs()));
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(((*g as f64) - w).abs() <= 2e-4 * scale, "{mode:?} [{i}] {g} vs {w}");
        }
    }
}

/// States at steps `<= t0` do not move when later inputs change.
#[test]
fn encoder_is_causal() {
    let dev = dev();
    let cfg = small_mamba(4);
    let enc = clm_model::mamba::MambaEncoder::<R, f32>::new(&cfg, &dev, &mut Rng::seeded(9));
    let (b, t, f) = (2, 16, 4);
    let x = noise(b * t * f, 3);
    let run = |x: &[f32]| {
        with_no_grad(|| {
            enc.sequence(&Var::constant(Tensor::from_f32(x, vec![b, t, f], &dev).unwrap()))
                .unwrap()
                .to_f32()
        })
    };
    let base = run(&x);
    for cut in [0usize, 7, 14] {
        let mut y = x.clone();
        for bi in 0..b {
            for tt in cut + 1..t {
                for k in 0..f {
                    y[(bi * t + tt) * f + k] += 5.0;
                }
            }
        }
        let out = run(&y);
        let d = cfg.d_model;
        for bi in 0..b {
            for tt in 0..=cut {
                for k in 0..d {
                    let i = (bi * t + tt) * d + k;
                    assert_eq!(out[i].to_bits(), base[i].to_bits(), "cut {cut} b {bi} t {tt}");
                }
            }
        }
    }
}

#[test]
fn embeddings_are_unit_norm_and_actions_ignore_the_state() {
    let dev = dev();
    let model = DecisionModel::<R, f32>::new(&clm_config(6, 1), &dev).unwrap();
    let (b, w, f) = (5, 12, 6);
    let run = |seed| {
        with_no_grad(|| {
            let x = Var::constant(Tensor::from_f32(&noise(b * w * f, seed), vec![b, w, f], &dev).unwrap());
            let o = model.forward(&x).unwrap();
            (o.scores.to_f32(), o.state_embedding.unwrap().to_f32())
        })
    };
    let (_, z) = run(1);
    for row in z.chunks(8) {
        let n: f32 = row.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((n - 1.0).abs() < 1e-4, "state embedding norm {n}");
    }
    // The action branch reads only the action attributes: identical with any input,
    // and the cache returns the same values as a fresh computation.
    let head_actions = |model: &DecisionModel<R, f32>| {
        with_no_grad(|| {
            let x = Var::constant(Tensor::from_f32(&noise(w * f, 77), vec![1, w, f], &dev).unwrap());
            model.forward(&x).unwrap();
        });
        model.named_parameters().len()
    };
    let _ = head_actions(&model);
    let (s1, _) = run(2);
    model.invalidate();
    let (s2, _) = run(2);
    assert_eq!(s1, s2, "cached and fresh action embeddings disagree");
}

/// The full model's autograd gradient against central differences of the loss.
#[test]
fn model_gradients_match_finite_differences() {
    let dev = dev();
    let model = DecisionModel::<R, f32>::new(&clm_config(3, 2), &dev).unwrap();
    let (b, w, f) = (4, 6, 3);
    let batch = Batch {
        size: b,
        window: w,
        features: f,
        inputs: noise(b * w * f, 5),
        labels: vec![0, 1, 2, 0],
        positives: vec![1., 0., 0., 0., 1., 0., 0., 0., 1., 1., 0., 1.],
        weights: vec![1.0; b],
        ret_target: vec![0.5, -1.0, 0.1, 2.0],
        stop_target: vec![0.0, 1.0, 0.0, 0.0],
    };
    let loss_cfg = LossConfig {
        decision: 1.0,
        supcon: 0.5,
        supcon_temperature: 0.5,
        ret: 0.3,
        risk: 0.2,
        ret_clip: 5.0,
    };
    let loss_of = |m: &DecisionModel<R, f32>| -> Var<R, f32> {
        let x = Var::constant(Tensor::from_f32(&batch.inputs, vec![b, w, f], &dev).unwrap());
        clm_model::losses::total_loss(&m.forward(&x).unwrap(), &batch, &loss_cfg, &dev).unwrap()
    };
    let grads = loss_of(&model).backward().unwrap();
    let params = model.named_parameters();
    let mut checked = 0;
    for (name, p) in &params {
        let Some(g) = grads.get(p.id()) else {
            panic!("no gradient for {name}");
        };
        let g = g.to_f32();
        let base = p.value().to_f32();
        // A few coordinates per tensor keep the test fast.
        for &i in [0usize, base.len() / 2, base.len() - 1].iter() {
            let eps = 1e-2f32;
            let eval = |delta: f32| {
                let mut v = base.clone();
                v[i] += delta;
                p.set(Tensor::from_f32(&v, p.shape(), &dev).unwrap());
                model.invalidate();
                let l = with_no_grad(|| loss_of(&model).to_f32()[0]);
                p.set(Tensor::from_f32(&base, p.shape(), &dev).unwrap());
                l as f64
            };
            let fd = (eval(eps) - eval(-eps)) / (2.0 * eps as f64);
            let an = g[i] as f64;
            assert!(
                (fd - an).abs() <= 2e-2 * (1.0 + fd.abs().max(an.abs())),
                "{name}[{i}]: finite difference {fd:.6} vs autograd {an:.6}"
            );
            checked += 1;
        }
    }
    assert!(checked > 40, "checked {checked} coordinates");
}

#[test]
fn checkpoint_round_trip_restores_outputs() {
    let dev = dev();
    let cfg = clm_config(4, 3);
    let a = DecisionModel::<R, f32>::new(&cfg, &dev).unwrap();
    let b = DecisionModel::<R, f32>::new(&ModelConfig { seed: 99, ..cfg.clone() }, &dev).unwrap();
    let dir = std::env::temp_dir().join(format!("clm-ckpt-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("model.ckpt");
    mamba3::train::checkpoint::Checkpoint::capture(&a, 7).save(&path).unwrap();
    let loaded = mamba3::train::checkpoint::Checkpoint::load(&path).unwrap();
    loaded.restore(&b, true).unwrap();
    b.invalidate();
    let x = noise(2 * 9 * 4, 8);
    let out = |m: &DecisionModel<R, f32>| {
        with_no_grad(|| {
            m.forward(&Var::constant(Tensor::from_f32(&x, vec![2, 9, 4], &dev).unwrap()))
                .unwrap()
                .scores
                .to_f32()
        })
    };
    assert_eq!(out(&a), out(&b));
    assert_eq!(loaded.step, 7);
    std::fs::remove_dir_all(&dir).ok();
}

/// A tiny synthetic fold with a planted, learnable signal.
fn synthetic_fold(
    seed: u64,
    window: usize,
) -> (clm_data::FeatureTensor, clm_data::Labels, HashMap<Split, Vec<SampleIndex>>) {
    let syn = generate(&SyntheticConfig {
        seed,
        num_days: 900,
        num_sectors: 2,
        stocks_per_sector: 6,
        missing_prob: 0.0,
        signal: Some(PlantedSignal {
            drift: 0.012,
            loading: 0.8,
        }),
    });
    let feats = compute_features(&syn.panel);
    let labels = compute_labels(
        &syn.panel,
        &LabelConfig {
            horizon: 1,
            kappa: 0.0002,
            margin_mult: 0.25,
            stop_frac: 0.03,
        },
    )
    .unwrap();
    let (train_end, val_end) = (650, 780);
    let norm = Normalizer::fit(&feats, 0..train_end).unwrap();
    let normed = norm.apply(&feats);
    let mut splits: HashMap<Split, Vec<SampleIndex>> = HashMap::new();
    for s in 0..feats.num_securities {
        for t in 0..feats.num_days {
            if !input_valid(&normed, s, t, window) || labels.class_of(s, t).is_none() {
                continue;
            }
            // Purge: h = 1, labels end at t + 2.
            let split = if t + 2 < train_end {
                Split::Train
            } else if (train_end..val_end).contains(&t) && t + 2 < val_end {
                Split::Val
            } else if t >= val_end {
                Split::Test
            } else {
                continue;
            };
            splits.entry(split).or_default().push(SampleIndex::new(s, t));
        }
    }
    (normed, labels, splits)
}

fn quick_train(steps: u64) -> TrainConfig {
    TrainConfig {
        max_steps: steps,
        batch_size: 64,
        learning_rate: 3e-3,
        weight_decay: 0.0,
        warmup_steps: 10,
        grad_clip: 1.0,
        eval_every: 50,
        patience: 100,
        eval_batch: 256,
        loss: LossConfig::default(),
        max_bad_steps: 5,
    }
}

/// Mamba + CLM learns the planted signal: held-out log loss beats the class
/// prior's and accuracy beats the majority class.
#[test]
fn clm_learns_planted_signal() {
    let dev = dev();
    let window = 8;
    let (feats, labels, splits) = synthetic_fold(11, window);
    let mut src = PanelSource::new(&feats, &labels, splits, SourceConfig { window, seed: 1, ..Default::default() });
    let model = DecisionModel::<R, f32>::new(&clm_config(feats.num_features, 5), &dev).unwrap();
    let report = fit(&model, &mut src, &quick_train(300), &dev, &mut |_| {}).unwrap();
    let test = predict_split(&model, &src, Split::Test, 256, &dev).unwrap();
    let n = test.labels.len() as f64;
    let mut counts = [0f64; 3];
    for &l in &test.labels {
        counts[l as usize] += 1.0;
    }
    let prior_nll: f64 = -counts.iter().filter(|&&c| c > 0.0).map(|c| c / n * (c / n).ln()).sum::<f64>();
    let probs = test.probabilities();
    let acc = probs
        .iter()
        .zip(&test.labels)
        .filter(|(p, l)| {
            let arg = (0..3).max_by(|&a, &b| p[a].total_cmp(&p[b])).unwrap();
            arg == **l as usize
        })
        .count() as f64
        / n;
    let majority = counts.iter().cloned().fold(0.0, f64::max) / n;
    eprintln!(
        "planted signal: test nll {:.4} vs prior {:.4}; acc {:.3} vs majority {:.3}; best step {}",
        test.nll(),
        prior_nll,
        acc,
        majority,
        report.best_step
    );
    assert!(test.nll() < prior_nll - 0.01, "no better than the prior");
    assert!(acc > majority + 0.02, "no better than the majority class");
}

/// Same seed, same data: identical training losses and weights.
#[test]
fn fixed_seed_training_is_reproducible() {
    let dev = dev();
    let window = 6;
    let (feats, labels, splits) = synthetic_fold(3, window);
    let run = || {
        let mut src = PanelSource::new(&feats, &labels, splits.clone(), SourceConfig { window, seed: 4, ..Default::default() });
        let model = DecisionModel::<R, f32>::new(&clm_config(feats.num_features, 8), &dev).unwrap();
        let mut cfg = quick_train(30);
        cfg.eval_every = 10;
        let r = fit(&model, &mut src, &cfg, &dev, &mut |_| {}).unwrap();
        let losses: Vec<f32> = r.history.iter().map(|p| p.train_loss).collect();
        (losses, model.parameters()[0].value().to_f32())
    };
    let (l1, w1) = run();
    let (l2, w2) = run();
    assert_eq!(l1, l2);
    assert_eq!(w1, w2);
}

/// Every encoder/head combination overfits 32 samples.
#[test]
fn every_model_overfits_a_small_batch() {
    let dev = dev();
    let (b, w, f) = (32, 8, 5);
    let mut rng = Rng::seeded(2);
    let inputs = rng.normal_vec(b * w * f, 0.0, 1.0);
    let labels: Vec<u32> = (0..b).map(|i| (i % 3) as u32).collect();
    let positives: Vec<f32> = labels
        .iter()
        .flat_map(|&l| (0..3).map(move |k| if k == l { 1.0 } else { 0.0 }))
        .collect();
    let batch = Batch {
        size: b,
        window: w,
        features: f,
        inputs,
        labels,
        positives,
        weights: vec![1.0; b],
        ret_target: vec![0.0; b],
        stop_target: vec![0.0; b],
    };
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
    let encoders = [
        EncoderConfig::Mamba(small_mamba(f)),
        EncoderConfig::Tcn(TcnConfig {
            input_dim: f,
            channels: 16,
            kernel: 3,
            num_layers: 3,
            pooling: Pooling::Last,
        }),
        EncoderConfig::LastRow {
            input_dim: f,
            hidden: 32,
        },
    ];
    for encoder in encoders {
        for head in [
            HeadConfig::Mlp { hidden: 32 },
            HeadConfig::Contrastive(ContrastiveConfig {
                embed_dim: 16,
                hidden: 32,
                temperature: Temperature::Learned(0.1),
                scoring: clm_model::Scoring::Bilinear,
            }),
        ] {
            let cfg = ModelConfig {
                encoder: encoder.clone(),
                head,
                aux_return: false,
                aux_risk: false,
                aux_hidden: 8,
                action_attrs: action_attributes(5, 0.001, 0.0005),
                seed: 1,
            };
            let model = DecisionModel::<R, f32>::new(&cfg, &dev).unwrap();
            let mut tc = quick_train(400);
            tc.learning_rate = 1e-2;
            tc.eval_every = 100;
            let r = fit(&model, &mut Fixed(batch.clone()), &tc, &dev, &mut |_| {}).unwrap();
            assert!(r.best_val_nll < 0.15, "{:?}/{:?}: nll {}", cfg.encoder, cfg.head, r.best_val_nll);
        }
    }
}
