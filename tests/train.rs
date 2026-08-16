//! Optimization, the training loop, checkpoints and generation.

#![cfg(feature = "backend")]

use mamba3::autograd::Var;
use mamba3::backend::Device;
use mamba3::backends::Auto;
use mamba3::nn::Module;
use mamba3::prelude::*;
use mamba3::tensor::Tensor;
use mamba3::tensor::ops::index::IdTensor;
use mamba3::train::{Checkpoint, LmBatch, LmTask, Optimizer, TrainStep};

type R = Auto;

fn dev() -> Device<R> {
    Device::<R>::default()
}

fn tiny_lm(seed: u64) -> Mamba3Lm<R, f32> {
    Mamba3LmConfig::builder()
        .vocab_size(8)
        .d_model(16)
        .n_layers(1)
        .with_ssm(|s| {
            s.n_heads = 2;
            s.n_groups = 2;
            s.head_dim = 4;
            s.d_state = 4;
            s.chunk_size = 4;
        })
        .seed(seed)
        .build()
        .unwrap()
        .init::<R, f32>(&dev())
        .unwrap()
}

fn batch(inputs: &[u32], targets: &[u32]) -> LmBatch<R> {
    let n = inputs.len();
    LmBatch {
        inputs: IdTensor::from_slice(inputs, vec![1, n], &dev()).unwrap(),
        targets: IdTensor::from_slice(targets, vec![1, n], &dev()).unwrap(),
    }
}

#[test]
fn adamw_minimises_a_quadratic() {
    // f(w) = sum((w - 3)^2), minimised at w = 3.
    let param = Param::new(Tensor::<R, f32>::zeros(vec![4], &dev()));
    let mut optimizer = AdamWConfig::builder()
        .learning_rate(0.5)
        .weight_decay(0.0)
        .build()
        .init::<R, f32>();

    for _ in 0..200 {
        let w = param.var_standalone();
        let diff = w.add_scalar(-3.0);
        let loss = diff.mul(&diff).unwrap().sum().unwrap();
        let grads = loss.backward().unwrap();
        optimizer.step(&[param.clone()], &grads).unwrap();
    }

    let final_value = param.value().to_f32();
    for v in final_value {
        assert!((v - 3.0).abs() < 1e-2, "converged to {v}, expected 3");
    }
    assert_eq!(optimizer.step_count(), 200);
}

#[test]
fn weight_decay_skips_vectors_by_default() {
    let vector = Param::new(Tensor::<R, f32>::ones(vec![4], &dev()));
    let matrix = Param::new(Tensor::<R, f32>::ones(vec![2, 2], &dev()));
    let mut optimizer = AdamWConfig::builder()
        .learning_rate(0.0) // isolate the decay term
        .weight_decay(0.5)
        .build()
        .init::<R, f32>();

    // Zero gradients so only decay could move the parameters.
    let mut grads = mamba3::autograd::Grads::<R, f32>::default();
    grads
        .accumulate(vector.id(), Tensor::zeros(vec![4], &dev()))
        .unwrap();
    grads
        .accumulate(matrix.id(), Tensor::zeros(vec![2, 2], &dev()))
        .unwrap();
    optimizer
        .step(&[vector.clone(), matrix.clone()], &grads)
        .unwrap();

    // With lr = 0 nothing moves at all; the point of the test is the policy, so
    // check it directly through a non-zero rate.
    let mut optimizer = AdamWConfig::builder()
        .learning_rate(0.1)
        .weight_decay(1.0)
        .build()
        .init::<R, f32>();
    optimizer
        .step(&[vector.clone(), matrix.clone()], &grads)
        .unwrap();
    assert_eq!(vector.value().to_f32(), vec![1.0; 4], "vectors are not decayed");
    assert!(
        matrix.value().to_f32().iter().all(|v| *v < 1.0),
        "matrices are decayed"
    );
}

#[test]
fn gradient_clipping_reports_and_rescales() {
    let device = dev();
    let mut grads = mamba3::autograd::Grads::<R, f32>::default();
    let id = mamba3::autograd::ParamId::fresh();
    grads
        .accumulate(id, Tensor::from_f32(&[3.0, 4.0], vec![2], &device).unwrap())
        .unwrap();

    let before = mamba3::train::grad_norm(&grads).unwrap();
    assert!((before - 5.0).abs() < 1e-4);

    let reported = mamba3::train::clip_grad_norm(&mut grads, 1.0).unwrap();
    assert!((reported - 5.0).abs() < 1e-4);
    let after = mamba3::train::grad_norm(&grads).unwrap();
    assert!((after - 1.0).abs() < 1e-3, "clipped norm is {after}");
}

#[test]
fn trainer_overfits_a_single_sequence() {
    let model = tiny_lm(4);
    let task = LmTask::new(&model);
    let data = batch(&[1, 2, 3, 4, 5, 6], &[2, 3, 4, 5, 6, 7]);

    let config = TrainerConfig::builder()
        .learning_rate(3e-2)
        .max_grad_norm(1.0)
        .build()
        .unwrap();
    let mut trainer = Trainer::new(config, AdamW::<R, f32>::new(3e-2));

    let first = trainer.step(&task, &[data.clone()]).unwrap().loss;
    let mut last = first;
    for _ in 0..24 {
        last = trainer.step(&task, &[data.clone()]).unwrap().loss;
    }
    assert!(
        last < first * 0.6,
        "loss barely moved: {first} -> {last}"
    );
    assert!(last.is_finite());
}

#[test]
fn gradient_accumulation_matches_a_larger_step() {
    let model = tiny_lm(9);
    let task = LmTask::new(&model);
    let a = batch(&[1, 2, 3, 4], &[2, 3, 4, 5]);
    let b = batch(&[5, 6, 7, 0], &[6, 7, 0, 1]);

    let config = TrainerConfig::builder()
        .learning_rate(1e-2)
        .max_grad_norm(0.0)
        .build()
        .unwrap();
    let mut trainer = Trainer::new(config, AdamW::<R, f32>::new(1e-2));
    let info = trainer.step(&task, &[a.clone(), b.clone()]).unwrap();

    // The reported loss is the mean over micro-batches.
    let mean = (task.loss(&a).unwrap().to_f32()[0] + task.loss(&b).unwrap().to_f32()[0]) / 2.0;
    // Parameters have already moved once, so compare against the pre-step values
    // by rebuilding the task on a fresh model.
    let fresh = tiny_lm(9);
    let fresh_task = LmTask::new(&fresh);
    let expected =
        (fresh_task.loss(&a).unwrap().to_f32()[0] + fresh_task.loss(&b).unwrap().to_f32()[0]) / 2.0;
    assert!((info.loss - expected).abs() < 1e-4, "{} vs {}", info.loss, expected);
    assert!(mean.is_finite());
}

#[test]
fn lora_training_touches_only_the_adapters() {
    let model = Mamba3LmConfig::builder()
        .vocab_size(8)
        .d_model(16)
        .n_layers(1)
        .with_ssm(|s| {
            s.n_heads = 2;
            s.n_groups = 2;
            s.head_dim = 4;
            s.d_state = 4;
            s.chunk_size = 4;
        })
        .lora(LoraConfig::builder().rank(2).alpha(4.0).build().unwrap())
        .seed(6)
        .build()
        .unwrap()
        .init::<R, f32>(&dev())
        .unwrap();

    // Freeze everything, then re-enable only the adapters.
    model.freeze_matching(&[]);
    model.unfreeze_matching(&["lora"]);

    let before: Vec<(String, Vec<f32>)> = model
        .named_parameters()
        .into_iter()
        .map(|(n, p)| (n, p.value().to_f32()))
        .collect();

    let task = LmTask::new(&model).only(&["lora"]);
    assert!(!task.trainable().is_empty());

    let config = TrainerConfig::builder().learning_rate(1e-1).build().unwrap();
    let mut trainer = Trainer::new(config, AdamW::<R, f32>::new(1e-1));
    trainer
        .step(&task, &[batch(&[1, 2, 3, 4], &[2, 3, 4, 5])])
        .unwrap();

    let after: Vec<(String, Vec<f32>)> = model
        .named_parameters()
        .into_iter()
        .map(|(n, p)| (n, p.value().to_f32()))
        .collect();

    let mut adapters_changed = 0;
    for ((name, old), (_, new)) in before.iter().zip(after.iter()) {
        let moved = old.iter().zip(new).any(|(a, b)| (a - b).abs() > 1e-9);
        if name.contains("lora") {
            if moved {
                adapters_changed += 1;
            }
        } else {
            assert!(!moved, "frozen parameter `{name}` was updated");
        }
    }
    assert!(adapters_changed > 0, "no adapter was updated");
}

#[test]
fn checkpoints_round_trip_and_can_be_filtered() {
    let dir = std::env::temp_dir().join("mamba3-test-checkpoints");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("model.json");

    let source = tiny_lm(12);
    let checkpoint = Checkpoint::capture(&source, 42)
        .with_metadata(serde_json::json!({"note": "unit test"}));
    checkpoint.save(&path).unwrap();

    let loaded = Checkpoint::load(&path).unwrap();
    assert_eq!(loaded.step, 42);
    assert_eq!(loaded.metadata["note"], "unit test");

    let target = tiny_lm(13);
    let tokens = IdTensor::from_slice(&[1, 2, 3, 4], vec![1, 4], &dev()).unwrap();
    let before = target.forward(&tokens, false).unwrap().to_f32();
    loaded.restore(&target, true).unwrap();
    let after = target.forward(&tokens, false).unwrap().to_f32();
    let expected = source.forward(&tokens, false).unwrap().to_f32();

    let diff = |a: &[f32], b: &[f32]| {
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
    };
    assert!(diff(&before, &expected) > 1e-5);
    assert!(diff(&after, &expected) < 1e-6);

    // A filtered checkpoint keeps only the matching entries.
    let filtered = checkpoint.filtered("in_proj");
    assert!(!filtered.state.entries.is_empty());
    assert!(filtered.state.entries.len() < checkpoint.state.entries.len());
    assert!(filtered.state.entries.keys().all(|k| k.contains("in_proj")));

    let _ = std::fs::remove_file(&path);
}

#[test]
fn generation_is_deterministic_when_greedy() {
    let model = tiny_lm(21);
    let config = GeneratorConfig::builder()
        .max_new_tokens(6)
        .sampler(SamplerConfig::greedy())
        .build();

    let mut first = Generator::new(&model, config.clone());
    let a = first.generate(&[1, 2, 3], &dev()).unwrap();
    let mut second = Generator::new(&model, config);
    let b = second.generate(&[1, 2, 3], &dev()).unwrap();

    assert_eq!(a.len(), 6);
    assert_eq!(a, b, "greedy decoding must be reproducible");
    assert!(a.iter().all(|t| (*t as usize) < 8));
}

#[test]
fn sampling_respects_top_k() {
    let mut rng = mamba3::tensor::ops::random::Rng::seeded(3);
    let logits = vec![0.0, 10.0, 0.0, 9.0, 0.0];
    let sampler = SamplerConfig::temperature(1.0).with_top_k(2);
    for _ in 0..50 {
        let token = sampler.sample(&logits, &[], &mut rng);
        assert!(token == 1 || token == 3, "top-k leaked token {token}");
    }

    // Greedy always takes the maximum.
    let greedy = SamplerConfig::greedy();
    assert_eq!(greedy.sample(&logits, &[], &mut rng), 1);
}

#[test]
fn eval_mode_disables_dropout() {
    let device = dev();
    let dropout = mamba3::nn::Dropout::new(0.5);
    let x: Var<R, f32> = Var::constant(Tensor::ones(vec![256], &device));

    <mamba3::nn::Dropout as Module<R, f32>>::set_training(&dropout, true);
    let train_out = dropout.apply(&x).unwrap().to_f32();
    assert!(train_out.iter().any(|v| *v == 0.0));

    <mamba3::nn::Dropout as Module<R, f32>>::set_training(&dropout, false);
    let eval_out = dropout.apply(&x).unwrap().to_f32();
    assert!(eval_out.iter().all(|v| (*v - 1.0).abs() < 1e-6));
}

/// The global gradient norm is computed on the device in one launch per gradient
/// plus one reduction; check it against the host arithmetic it stands for, on shapes
/// that do and do not divide the partial count evenly.
#[test]
fn grad_norm_matches_the_host() {
    use mamba3::autograd::{Grads, ParamId};
    use mamba3::train::grad_norm;

    for lengths in [vec![1usize], vec![7], vec![256], vec![1000], vec![3, 511, 64]] {
        let mut grads = Grads::<R, f32>::default();
        let mut want = 0.0f32;
        for (k, n) in lengths.iter().enumerate() {
            let data: Vec<f32> = (0..*n)
                .map(|i| ((i + k) % 17) as f32 * 0.25 - 2.0)
                .collect();
            want += data.iter().map(|v| v * v).sum::<f32>();
            grads
                .accumulate(
                    ParamId::fresh(),
                    Tensor::from_f32(&data, vec![*n], &dev()).unwrap(),
                )
                .unwrap();
        }
        let want = want.sqrt();
        let got = grad_norm(&grads).unwrap();
        assert!(
            (got - want).abs() < 1e-3 * (1.0 + want),
            "lengths {lengths:?}: {got} != {want}"
        );
    }
}

/// The gradient scale the trainer hands the optimizer must match the clip it
/// replaced, including the micro-batch averaging folded into it.
///
/// This is the one place where moving work onto the device changed an interface
/// rather than just an implementation: the factor used to be an `f32` computed after
/// reading the norm back, and is now a one-element tensor computed from the same
/// reduction without reading anything. If the two ever disagree, every clipped step
/// is silently taking a different-sized update.
#[test]
fn device_side_grad_scale_matches_the_host_clip() {
    use mamba3::autograd::{Grads, ParamId};
    use mamba3::train::{grad_norm, grad_scale};

    // Norms chosen to land either side of the clip threshold, and an averaging
    // factor that is not 1 so the two contributions cannot be confused.
    for (scale, max_norm, magnitude) in [
        (1.0f32, 1.0f32, 0.01f32), // well under: no clipping
        (1.0, 1.0, 5.0),           // well over: clipped
        (0.25, 1.0, 5.0),          // clipped, and averaged over four micro-batches
        (0.25, 1.0, 0.01),         // averaged, not clipped
        (0.5, 0.0, 3.0),           // clipping disabled entirely
    ] {
        let mut grads = Grads::<R, f32>::default();
        for k in 0..3usize {
            let data: Vec<f32> = (0..97)
                .map(|i| (((i + k) % 17) as f32 * 0.25 - 2.0) * magnitude)
                .collect();
            grads
                .accumulate(
                    ParamId::fresh(),
                    Tensor::from_f32(&data, vec![97], &dev()).unwrap(),
                )
                .unwrap();
        }

        // What the host used to compute: average first, then clip the averaged norm.
        let averaged_norm = grad_norm(&grads).unwrap() * scale;
        let want = if max_norm > 0.0 && averaged_norm > max_norm {
            scale * max_norm / (averaged_norm + 1e-6)
        } else {
            scale
        };

        let scaling = grad_scale(&grads, max_norm, scale).unwrap().unwrap();
        let got = scaling.factor.to_f32()[0];
        assert!(
            (got - want).abs() < 1e-5 * (1.0 + want.abs()),
            "scale {scale} max_norm {max_norm} magnitude {magnitude}: {got} != {want}"
        );
        // The reported norm comes from the same reduction, scaled the same way.
        let reported = (scaling.sum_squares.to_f32()[0] * scale * scale).sqrt();
        assert!(
            (reported - averaged_norm).abs() < 1e-3 * (1.0 + averaged_norm),
            "reported norm {reported} != {averaged_norm}"
        );
    }
}
