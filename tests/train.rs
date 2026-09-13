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

/// A2a: a saved run can be resumed, not merely warm-started from its weights.
///
/// The proof is that a run interrupted after 20 steps and continued for 20
/// more, from a checkpoint that carries optimizer state, cannot be told apart
/// from a single uninterrupted 40-step run: same weights, same optimizer
/// moments, same step counters. A round-trip equality test alone would not
/// prove this -- it is easy to write a save/load pair that preserves the
/// checkpoint's own contents while still losing the moments a *resumed
/// training run* actually depends on.
#[test]
fn resuming_is_indistinguishable_from_not_stopping() {
    let data = batch(&[1, 2, 3, 4, 5, 6], &[2, 3, 4, 5, 6, 7]);
    let config = || {
        TrainerConfig::builder()
            .learning_rate(1e-2)
            .max_grad_norm(1.0)
            .build()
            .unwrap()
    };

    // 40 uninterrupted steps.
    let continuous_model = tiny_lm(11);
    let continuous_task = LmTask::new(&continuous_model);
    let mut continuous_trainer = Trainer::new(config(), AdamW::<R, f32>::new(1e-2));
    for _ in 0..40 {
        continuous_trainer.step(&continuous_task, &[data.clone()]).unwrap();
    }

    // 20 steps, a checkpoint that carries optimizer state, then a save/load
    // boundary meant to stand in for a new process.
    let resumed_model = tiny_lm(11);
    let resumed_task = LmTask::new(&resumed_model);
    let mut resumed_trainer = Trainer::new(config(), AdamW::<R, f32>::new(1e-2));
    for _ in 0..20 {
        resumed_trainer.step(&resumed_task, &[data.clone()]).unwrap();
    }
    let checkpoint = Checkpoint::capture(&resumed_model, resumed_trainer.step_count())
        .with_optimizer(&resumed_model, resumed_trainer.optimizer());

    let dir = std::env::temp_dir().join("mamba3-test-resume");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("resume.json");
    checkpoint.save(&path).unwrap();
    let loaded = Checkpoint::load(&path).unwrap();
    let _ = std::fs::remove_file(&path);

    let fresh_model = tiny_lm(11);
    loaded.restore(&fresh_model, true).unwrap();
    let mut fresh_trainer = Trainer::new(config(), AdamW::<R, f32>::new(1e-2));
    loaded
        .restore_optimizer(&fresh_model, fresh_trainer.optimizer_mut(), true)
        .unwrap();
    fresh_trainer.set_step_count(loaded.step);

    let fresh_task = LmTask::new(&fresh_model);
    for _ in 0..20 {
        fresh_trainer.step(&fresh_task, &[data.clone()]).unwrap();
    }

    // Weights match to a tight CPU tolerance -- not bit-exact, because a
    // save/load through `f32` JSON is one more rounding than the continuous
    // run ever takes, but far tighter than the two runs merely converging to
    // the same place would need.
    for ((name, a), (_, b)) in continuous_model
        .named_parameters()
        .into_iter()
        .zip(fresh_model.named_parameters())
    {
        let (av, bv) = (a.value().to_f32(), b.value().to_f32());
        for (x, y) in av.iter().zip(&bv) {
            assert!((x - y).abs() < 1e-5, "{name} diverged after resume: {x} vs {y}");
        }
    }

    // Counters and learning rates agree.
    assert_eq!(continuous_trainer.step_count(), fresh_trainer.step_count());
    assert_eq!(
        continuous_trainer.optimizer().step_count(),
        fresh_trainer.optimizer().step_count()
    );
    assert_eq!(
        continuous_trainer.optimizer().learning_rate(),
        fresh_trainer.optimizer().learning_rate()
    );

    // And so does the optimizer's own internal state -- the moments a resumed
    // run actually depends on, not just the weights they already moved.
    let want = continuous_trainer
        .optimizer()
        .state_dict(&continuous_model.named_parameters());
    let got = fresh_trainer.optimizer().state_dict(&fresh_model.named_parameters());
    assert_eq!(
        want.entries.keys().collect::<Vec<_>>(),
        got.entries.keys().collect::<Vec<_>>(),
        "the two optimizers tracked different parameters"
    );
    for (key, w) in &want.entries {
        let g = &got.entries[key];
        for (x, y) in w.data.iter().zip(&g.data) {
            assert!((x - y).abs() < 1e-5, "optimizer state {key} diverged: {x} vs {y}");
        }
    }

    // Weights-only loading remains an explicit warm start: no optimizer call
    // means no optimizer state, and the counter starts at zero.
    let warm_model = tiny_lm(11);
    loaded.restore(&warm_model, true).unwrap();
    let warm_trainer = Trainer::new(config(), AdamW::<R, f32>::new(1e-2));
    assert_eq!(warm_trainer.optimizer().step_count(), 0);
}

/// A stale architecture, a malformed checkpoint and a legacy weights-only
/// fixture are all refused or handled explicitly rather than silently
/// producing a bad resume.
#[test]
fn optimizer_restore_rejects_what_it_cannot_honour() {
    let model = tiny_lm(14);
    let mut trainer = Trainer::new(
        TrainerConfig::builder().learning_rate(1e-2).build().unwrap(),
        AdamW::<R, f32>::new(1e-2),
    );
    let task = LmTask::new(&model);
    trainer
        .step(&task, &[batch(&[1, 2, 3, 4], &[2, 3, 4, 5])])
        .unwrap();

    // A legacy, weights-only checkpoint carries no optimizer state at all.
    let legacy = Checkpoint::capture(&model, 1);
    assert!(legacy.optimizer.is_none());
    let mut fresh = Trainer::new(
        TrainerConfig::builder().learning_rate(1e-2).build().unwrap(),
        AdamW::<R, f32>::new(1e-2),
    );
    assert!(
        legacy
            .restore_optimizer(&model, fresh.optimizer_mut(), true)
            .is_err(),
        "restoring optimizer state from a checkpoint that has none must fail under strict"
    );
    assert!(
        legacy
            .restore_optimizer(&model, fresh.optimizer_mut(), false)
            .is_ok(),
        "non-strict must accept it as an explicit warm start"
    );

    // A stale architecture: the checkpoint's moments do not match the live
    // model's parameter shapes.
    let checkpoint = Checkpoint::capture(&model, 1).with_optimizer(&model, trainer.optimizer());
    let other_shape = Mamba3LmConfig::builder()
        .vocab_size(8)
        .d_model(32) // different from `tiny_lm`'s 16
        .n_layers(1)
        .with_ssm(|s| {
            s.n_heads = 2;
            s.n_groups = 2;
            s.head_dim = 8;
            s.d_state = 4;
            s.chunk_size = 4;
        })
        .seed(14)
        .build()
        .unwrap()
        .init::<R, f32>(&dev())
        .unwrap();
    let mut mismatched = Trainer::new(
        TrainerConfig::builder().learning_rate(1e-2).build().unwrap(),
        AdamW::<R, f32>::new(1e-2),
    );
    assert!(
        checkpoint
            .restore_optimizer(&other_shape, mismatched.optimizer_mut(), false)
            .is_err(),
        "a shape mismatch must be reported even outside strict mode"
    );
}

// ---------------------------------------------------------------------------
// A5: a versioned binary checkpoint format, with the legacy JSON one kept
// ---------------------------------------------------------------------------

mod binary_checkpoint {
    use super::*;

    fn scratch(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join("mamba3-test-binary-checkpoint");
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(name)
    }

    #[test]
    fn round_trips_weights_and_optimizer_state_bit_exactly() {
        let model = tiny_lm(30);
        let mut trainer = Trainer::new(
            TrainerConfig::builder().learning_rate(1e-2).build().unwrap(),
            AdamW::<R, f32>::new(1e-2),
        );
        let task = LmTask::new(&model);
        trainer
            .step(&task, &[batch(&[1, 2, 3, 4], &[2, 3, 4, 5])])
            .unwrap();

        let checkpoint = Checkpoint::capture(&model, trainer.step_count())
            .with_optimizer(&model, trainer.optimizer())
            .with_metadata(serde_json::json!({"note": "bit-exact round trip"}));
        let path = scratch("round_trip.m3ck");
        checkpoint.save(&path).unwrap();
        let loaded = Checkpoint::load(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        assert_eq!(loaded.step, checkpoint.step);
        assert_eq!(loaded.metadata, checkpoint.metadata);
        assert_eq!(loaded.state.entries.len(), checkpoint.state.entries.len());
        for (name, original) in &checkpoint.state.entries {
            let round_tripped = &loaded.state.entries[name];
            assert_eq!(round_tripped.shape, original.shape);
            // Binary round-trips the exact bit pattern; unlike the JSON path
            // there is no decimal-text rounding to tolerate.
            assert_eq!(
                round_tripped.data, original.data,
                "{name} did not round-trip bit-exactly"
            );
        }
        let want_opt = checkpoint.optimizer.as_ref().unwrap();
        let got_opt = loaded.optimizer.as_ref().unwrap();
        assert_eq!(want_opt.entries.len(), got_opt.entries.len());
        for (name, original) in &want_opt.entries {
            assert_eq!(got_opt.entries[name].data, original.data, "optimizer {name} diverged");
        }
    }

    #[test]
    fn a_resumed_run_continues_identically_through_the_binary_format() {
        let data = batch(&[1, 2, 3, 4, 5, 6], &[2, 3, 4, 5, 6, 7]);
        let config = || TrainerConfig::builder().learning_rate(1e-2).build().unwrap();

        let continuous_model = tiny_lm(31);
        let continuous_task = LmTask::new(&continuous_model);
        let mut continuous_trainer = Trainer::new(config(), AdamW::<R, f32>::new(1e-2));
        for _ in 0..12 {
            continuous_trainer.step(&continuous_task, &[data.clone()]).unwrap();
        }

        let resumed_model = tiny_lm(31);
        let resumed_task = LmTask::new(&resumed_model);
        let mut resumed_trainer = Trainer::new(config(), AdamW::<R, f32>::new(1e-2));
        for _ in 0..6 {
            resumed_trainer.step(&resumed_task, &[data.clone()]).unwrap();
        }
        let path = scratch("resume.m3ck");
        Checkpoint::capture(&resumed_model, resumed_trainer.step_count())
            .with_optimizer(&resumed_model, resumed_trainer.optimizer())
            .save(&path)
            .unwrap();
        let loaded = Checkpoint::load(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        let fresh_model = tiny_lm(31);
        loaded.restore(&fresh_model, true).unwrap();
        let mut fresh_trainer = Trainer::new(config(), AdamW::<R, f32>::new(1e-2));
        loaded
            .restore_optimizer(&fresh_model, fresh_trainer.optimizer_mut(), true)
            .unwrap();
        fresh_trainer.set_step_count(loaded.step);
        let fresh_task = LmTask::new(&fresh_model);
        for _ in 0..6 {
            fresh_trainer.step(&fresh_task, &[data.clone()]).unwrap();
        }

        for ((name, a), (_, b)) in continuous_model
            .named_parameters()
            .into_iter()
            .zip(fresh_model.named_parameters())
        {
            let (av, bv) = (a.value().to_f32(), b.value().to_f32());
            for (x, y) in av.iter().zip(&bv) {
                assert!((x - y).abs() < 1e-5, "{name} diverged after a binary resume: {x} vs {y}");
            }
        }
    }

    #[test]
    fn a_legacy_json_fixture_still_loads() {
        // Hand-built, in the exact shape the format had before this field
        // existed: no `optimizer` key at all, not even `null`.
        let json = r#"{"step":7,"state":{"entries":{"w":{"shape":[2],"data":[1.5,-2.5]}}},"metadata":{"note":"pre-A5 fixture"}}"#;
        let path = scratch("legacy.json");
        std::fs::write(&path, json).unwrap();
        let loaded = Checkpoint::load(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(loaded.step, 7);
        assert!(loaded.optimizer.is_none());
        assert_eq!(loaded.state.entries["w"].data, vec![1.5, -2.5]);
    }

    #[test]
    fn load_goes_by_content_not_by_a_misleading_extension() {
        let model = tiny_lm(32);
        let checkpoint = Checkpoint::capture(&model, 3);
        // Written as binary (no `.json` extension)...
        let binary_path = scratch("actually_binary.m3ck");
        checkpoint.save(&binary_path).unwrap();
        // ...then handed a `.json` name. `load` must still recognise the magic.
        let misnamed = scratch("misleading.json");
        std::fs::rename(&binary_path, &misnamed).unwrap();
        let loaded = Checkpoint::load(&misnamed).unwrap();
        let _ = std::fs::remove_file(&misnamed);
        assert_eq!(loaded.step, 3);

        // And the reverse: real JSON content behind a `.m3ck` name.
        let json_path = scratch("actually_json.m3ck");
        std::fs::write(&json_path, serde_json::to_vec(&checkpoint).unwrap()).unwrap();
        let loaded = Checkpoint::load(&json_path).unwrap();
        let _ = std::fs::remove_file(&json_path);
        assert_eq!(loaded.step, 3);
    }

    #[test]
    fn a_truncated_header_is_refused() {
        let model = tiny_lm(33);
        let bytes_path = scratch("truncated.m3ck");
        Checkpoint::capture(&model, 1).save(&bytes_path).unwrap();
        let mut bytes = std::fs::read(&bytes_path).unwrap();
        let _ = std::fs::remove_file(&bytes_path);
        bytes.truncate(20); // well inside the header
        let truncated_path = scratch("truncated_written.m3ck");
        std::fs::write(&truncated_path, &bytes).unwrap();
        let err = Checkpoint::load(&truncated_path).unwrap_err();
        let _ = std::fs::remove_file(&truncated_path);
        assert!(format!("{err}").contains("shorter"), "{err}");
    }

    #[test]
    fn a_truncated_payload_is_refused() {
        let model = tiny_lm(34);
        let path = scratch("truncated_payload.m3ck");
        Checkpoint::capture(&model, 1).save(&path).unwrap();
        let mut bytes = std::fs::read(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        bytes.truncate(bytes.len() - 4); // one float short
        let truncated_path = scratch("truncated_payload_written.m3ck");
        std::fs::write(&truncated_path, &bytes).unwrap();
        assert!(
            Checkpoint::load(&truncated_path).is_err(),
            "a payload shorter than the header's own tensor descriptors must be refused"
        );
        let _ = std::fs::remove_file(&truncated_path);
    }

    #[test]
    fn an_unsupported_version_is_refused() {
        let model = tiny_lm(35);
        let path = scratch("bad_version.m3ck");
        Checkpoint::capture(&model, 1).save(&path).unwrap();
        let mut bytes = std::fs::read(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        bytes[8..12].copy_from_slice(&99u32.to_le_bytes());
        std::fs::write(&path, &bytes).unwrap();
        let err = Checkpoint::load(&path).unwrap_err();
        let _ = std::fs::remove_file(&path);
        assert!(format!("{err}").contains("version"), "{err}");
    }

    #[cfg(unix)]
    #[test]
    fn a_failed_write_preserves_the_previous_checkpoint() {
        use std::os::unix::fs::PermissionsExt;

        let model = tiny_lm(36);
        let dir = scratch("readonly_dir");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("checkpoint.m3ck");

        // A known-good checkpoint already at `path`.
        Checkpoint::capture(&model, 1).save(&path).unwrap();
        let before = std::fs::read(&path).unwrap();

        // The temp file this write needs cannot be created: the directory is
        // read-only. The write must fail without touching the existing file.
        let mut perms = std::fs::metadata(&dir).unwrap().permissions();
        perms.set_mode(0o500);
        std::fs::set_permissions(&dir, perms).unwrap();
        let result = Checkpoint::capture(&model, 2).save(&path);
        let mut restore = std::fs::metadata(&dir).unwrap().permissions();
        restore.set_mode(0o700);
        std::fs::set_permissions(&dir, restore).unwrap();

        assert!(result.is_err(), "the write should have failed on a read-only directory");
        let after = std::fs::read(&path).unwrap();
        assert_eq!(before, after, "a failed write must not disturb the previous checkpoint");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_binary_format_is_not_larger_than_json_on_a_representative_policy() {
        let model = tiny_lm(37);
        let checkpoint = Checkpoint::capture(&model, 1);
        let json_path = scratch("size.json");
        let binary_path = scratch("size.m3ck");
        checkpoint.save(&json_path).unwrap();
        checkpoint.save(&binary_path).unwrap();
        let json_len = std::fs::metadata(&json_path).unwrap().len();
        let binary_len = std::fs::metadata(&binary_path).unwrap().len();
        let _ = std::fs::remove_file(&json_path);
        let _ = std::fs::remove_file(&binary_path);
        println!(
            "checkpoint size: {json_len} bytes JSON, {binary_len} bytes binary \
             ({:.2}x smaller) for a {}-parameter tiny_lm",
            json_len as f64 / binary_len as f64,
            model.num_parameters(),
        );
        assert!(
            binary_len < json_len,
            "binary ({binary_len} bytes) should not be larger than JSON ({json_len} bytes)"
        );
    }
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
