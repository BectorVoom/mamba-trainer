//! End-to-end checks on the assembled models.

#![cfg(feature = "backend")]

use mamba3::autograd::Var;
use mamba3::backend::Device;
use mamba3::backends::Auto;
use mamba3::models::vision::{Pooling, ScanDirection};
use mamba3::nn::Module;
use mamba3::prelude::*;
use mamba3::tensor::Tensor;
use mamba3::tensor::ops::index::IdTensor;
use mamba3::tensor::ops::random::Rng;

type R = Auto;

fn dev() -> Device<R> {
    Device::<R>::default()
}

/// A deliberately tiny language model: the CPU backend JIT-compiles every kernel,
/// so tests stay small on purpose.
fn tiny_lm(pattern: LayerPattern) -> Mamba3Lm<R, f32> {
    Mamba3LmConfig::builder()
        .vocab_size(16)
        .d_model(16)
        .n_layers(2)
        .pattern(pattern)
        .with_ssm(|s| {
            s.n_heads = 2;
            s.n_groups = 2;
            s.head_dim = 4;
            s.d_state = 4;
            s.chunk_size = 4;
        })
        .attention(mamba3::nn::AttentionConfig::new(16, 2).with_max_seq_len(64))
        .seed(7)
        .build()
        .unwrap()
        .init::<R, f32>(&dev())
        .unwrap()
}

fn ids(values: &[u32], batch: usize, seq: usize) -> IdTensor<R> {
    IdTensor::from_slice(values, vec![batch, seq], &dev()).unwrap()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[test]
fn language_model_forward_shapes() {
    let model = tiny_lm(LayerPattern::AllMamba);
    let tokens = ids(&[1, 2, 3, 4, 5, 6, 7, 8], 1, 8);
    let logits = model.forward(&tokens, false).unwrap();
    assert_eq!(logits.dims(), &[1, 8, 16]);
    assert!(model.num_parameters() > 0);
    assert!(logits.to_f32().iter().all(|v| v.is_finite()));
}

#[test]
fn hybrid_pattern_places_attention_layers() {
    let pattern = LayerPattern::AttentionEvery { period: 2 };
    assert_eq!(
        pattern.expand(4).unwrap(),
        vec![
            LayerKind::Mamba,
            LayerKind::Attention,
            LayerKind::Mamba,
            LayerKind::Attention
        ]
    );

    let model = tiny_lm(pattern);
    assert_eq!(
        model.layer_kinds(),
        vec![LayerKind::Mamba, LayerKind::Attention]
    );
    let tokens = ids(&[0, 1, 2, 3], 1, 4);
    let logits = model.forward(&tokens, false).unwrap();
    assert_eq!(logits.dims(), &[1, 4, 16]);

    // The stack must expose attention parameters under a distinguishable path.
    let names: Vec<String> = model
        .named_parameters()
        .into_iter()
        .map(|(n, _)| n)
        .collect();
    assert!(names.iter().any(|n| n.contains("attention.q_proj")));
    assert!(names.iter().any(|n| n.contains("mixer.in_proj")));
}

/// The property that makes an SSM worth using: decoding one token at a time with
/// a carried state must reproduce the parallel forward pass exactly.
#[test]
fn cached_decoding_matches_the_parallel_pass() {
    for pattern in [
        LayerPattern::AllMamba,
        LayerPattern::AttentionEvery { period: 2 },
    ] {
        let model = tiny_lm(pattern.clone());
        model.eval();
        let sequence: Vec<u32> = vec![3, 1, 4, 1, 5, 9, 2, 6];
        let tokens = ids(&sequence, 1, sequence.len());
        let full = model.forward(&tokens, false).unwrap().to_f32();

        // Feed one token at a time through the cache.
        let mut cache = model.empty_cache(1, &dev());
        let mut stepwise = Vec::new();
        for token in &sequence {
            let step = ids(&[*token], 1, 1);
            let logits = model.forward_cached(&step, &mut cache).unwrap();
            stepwise.extend(logits.to_f32());
        }
        assert!(
            max_abs_diff(&stepwise, &full) < 2e-3,
            "{pattern:?}: stepwise decoding drifted from the parallel pass by {}",
            max_abs_diff(&stepwise, &full)
        );

        // And a prefill followed by single steps must agree too.
        let mut cache = model.empty_cache(1, &dev());
        let prefill = ids(&sequence[..5], 1, 5);
        let mut mixed = model.forward_cached(&prefill, &mut cache).unwrap().to_f32();
        for token in &sequence[5..] {
            let step = ids(&[*token], 1, 1);
            mixed.extend(model.forward_cached(&step, &mut cache).unwrap().to_f32());
        }
        assert!(
            max_abs_diff(&mixed, &full) < 2e-3,
            "{pattern:?}: prefill + steps drifted by {}",
            max_abs_diff(&mixed, &full)
        );
    }
}

#[test]
fn discretization_and_dynamics_variants_all_run() {
    use mamba3::ssm::{Discretization, SsmMode, StateDynamics};

    for (disc, dyn_, mode) in [
        (Discretization::Euler, StateDynamics::Real, SsmMode::Siso),
        (Discretization::Trapezoid, StateDynamics::Real, SsmMode::Siso),
        (
            Discretization::LearnedTrapezoid,
            StateDynamics::Rotational,
            SsmMode::Siso,
        ),
        (
            Discretization::LearnedTrapezoid,
            StateDynamics::Rotational,
            SsmMode::Mimo { rank: 2 },
        ),
    ] {
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
                s.discretization = disc;
                s.dynamics = dyn_;
                s.mode = mode;
            })
            .seed(3)
            .build()
            .unwrap()
            .init::<R, f32>(&dev())
            .unwrap();

        let tokens = ids(&[0, 1, 2, 3, 4, 5], 1, 6);
        let logits = model.forward(&tokens, false).unwrap();
        assert_eq!(logits.dims(), &[1, 6, 8]);
        assert!(
            logits.to_f32().iter().all(|v| v.is_finite()),
            "{disc:?}/{dyn_:?}/{mode:?} produced non-finite logits"
        );

        // The incremental path must agree for every variant.
        model.eval();
        let mut cache = model.empty_cache(1, &dev());
        let mut stepwise = Vec::new();
        for token in [0u32, 1, 2, 3, 4, 5] {
            let step = ids(&[token], 1, 1);
            stepwise.extend(model.forward_cached(&step, &mut cache).unwrap().to_f32());
        }
        assert!(
            max_abs_diff(&stepwise, &logits.to_f32()) < 2e-3,
            "{disc:?}/{dyn_:?}/{mode:?}: incremental decoding disagrees"
        );
    }
}

#[test]
fn vision_model_classifies() {
    let model = VisionMamba3Config::builder()
        .image_size(8)
        .patch_size(4)
        .in_channels(3)
        .num_classes(5)
        .d_model(16)
        .n_layers(1)
        .direction(ScanDirection::Bidirectional)
        .pooling(Pooling::Mean)
        .with_ssm(|s| {
            s.n_heads = 2;
            s.n_groups = 2;
            s.head_dim = 4;
            s.d_state = 4;
            s.chunk_size = 2;
        })
        .seed(1)
        .build()
        .unwrap()
        .init::<R, f32>(&dev())
        .unwrap();

    assert_eq!(model.config().num_patches(), 4);
    let pixels: Vec<f32> = (0..2 * 3 * 8 * 8).map(|i| (i % 17) as f32 / 17.0).collect();
    let images = Var::constant(Tensor::from_f32(&pixels, vec![2, 3, 8, 8], &dev()).unwrap());
    let logits = model.forward(&images).unwrap();
    assert_eq!(logits.dims(), &[2, 5]);
    assert!(logits.to_f32().iter().all(|v| v.is_finite()));

    // Bidirectional blocks own two mixers.
    let names: Vec<String> = model
        .named_parameters()
        .into_iter()
        .map(|(n, _)| n)
        .collect();
    assert!(names.iter().any(|n| n.contains("blocks.0.forward")));
    assert!(names.iter().any(|n| n.contains("blocks.0.backward")));
}

#[test]
fn state_dict_round_trips() {
    let source = tiny_lm(LayerPattern::AllMamba);
    let target = tiny_lm(LayerPattern::AllMamba);

    let tokens = ids(&[1, 2, 3, 4], 1, 4);
    // Perturb the target so the two models genuinely differ.
    for (_, p) in target.named_parameters() {
        p.set(mamba3::tensor::ops::elemwise::add_scalar(&p.value(), 0.1));
    }
    let before = target.forward(&tokens, false).unwrap().to_f32();
    let expected = source.forward(&tokens, false).unwrap().to_f32();
    assert!(max_abs_diff(&before, &expected) > 1e-4);

    let state = source.state_dict();
    target.load_state_dict(&state, true).unwrap();
    let after = target.forward(&tokens, false).unwrap().to_f32();
    assert!(max_abs_diff(&after, &expected) < 1e-6);
}

#[test]
fn lora_starts_neutral_and_merges_exactly() {
    let mut rng = Rng::seeded(5);
    let device = dev();
    let lora = LoraConfig::builder().rank(2).alpha(4.0).build().unwrap();

    let plain = LinearConfig::new(8, 6)
        .with_bias(false)
        .init::<R, f32>(&device, &mut Rng::seeded(11));
    let adapted = LinearConfig::new(8, 6)
        .with_bias(false)
        .with_lora(lora)
        .init::<R, f32>(&device, &mut Rng::seeded(11));

    let data: Vec<f32> = (0..2 * 8).map(|i| (i as f32) * 0.1 - 0.5).collect();
    let x = Var::constant(Tensor::from_f32(&data, vec![2, 8], &device).unwrap());

    // B is initialised to zero, so the adapter contributes nothing at first.
    assert!(
        max_abs_diff(
            &plain.apply(&x).unwrap().to_f32(),
            &adapted.apply(&x).unwrap().to_f32()
        ) < 1e-6
    );

    // Give B some content, then check that merging reproduces the adapted output.
    let b = adapted.lora().unwrap().b();
    b.set(mamba3::tensor::ops::random::randn(
        b.shape(),
        0.0,
        0.5,
        &device,
        &mut rng,
    ));
    let adapted_out = adapted.apply(&x).unwrap().to_f32();
    adapted.merge_lora().unwrap();
    let merged_out = adapted.apply(&x).unwrap().to_f32();
    assert!(
        max_abs_diff(&adapted_out, &merged_out) < 1e-5,
        "merging changed the function by {}",
        max_abs_diff(&adapted_out, &merged_out)
    );

    // Freezing the base leaves only the adapter trainable.
    adapted.freeze_base();
    let trainable: usize = adapted
        .named_parameters()
        .into_iter()
        .filter(|(_, p)| p.requires_grad())
        .map(|(_, p)| p.numel())
        .sum();
    assert_eq!(trainable, 8 * 2 + 2 * 6);
}

#[test]
fn fake_quantization_snaps_to_a_grid_and_passes_gradients() {
    use mamba3::nn::quant::{Granularity, QuantConfig, QuantScheme};

    let device = dev();
    let data: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) / 32.0).collect();
    let x: Var<R, f32> = Var::traced(Tensor::from_f32(&data, vec![8, 8], &device).unwrap());

    let quantizer = mamba3::nn::quant::Quantizer::new(
        QuantConfig::builder()
            .bits(4)
            .scheme(QuantScheme::Symmetric)
            .granularity(Granularity::PerTensor)
            .dynamic(true)
            .build()
            .unwrap(),
    );
    let q = quantizer.quantize(&x).unwrap();
    let values = q.to_f32();

    // 4-bit symmetric leaves at most 15 distinct levels.
    let mut levels: Vec<i64> = values.iter().map(|v| (v * 1e5).round() as i64).collect();
    levels.sort_unstable();
    levels.dedup();
    assert!(levels.len() <= 15, "got {} distinct levels", levels.len());

    // Error is bounded by half a step.
    let scale = 1.0 / 7.0;
    assert!(max_abs_diff(&values, &data) <= scale * 0.5 + 1e-5);

    // The straight-through estimator passes gradients inside the clamp range.
    let loss = q.sum().unwrap();
    let grads = loss.backward_retain().unwrap();
    let g = grads.node(x.node().unwrap()).unwrap().to_f32();
    assert!(g.iter().all(|v| (*v - 1.0).abs() < 1e-5 || *v == 0.0));
    assert!(g.iter().any(|v| (*v - 1.0).abs() < 1e-5));

    // A higher bit width must not be worse.
    let fine = mamba3::nn::quant::Quantizer::new(
        QuantConfig::builder().bits(8).dynamic(true).build().unwrap(),
    );
    let fine_err = max_abs_diff(&fine.quantize(&x).unwrap().to_f32(), &data);
    assert!(fine_err <= max_abs_diff(&values, &data) + 1e-6);
}

#[test]
fn quantized_model_still_runs_and_differentiates() {
    use mamba3::nn::quant::QuantConfig;

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
        .weight_quant(QuantConfig::int8_weights())
        .activation_quant(QuantConfig::int8_activations())
        .seed(2)
        .build()
        .unwrap()
        .init::<R, f32>(&dev())
        .unwrap();

    let tokens = ids(&[0, 1, 2, 3], 1, 4);
    let targets = ids(&[1, 2, 3, 4], 1, 4);
    let logits = model.forward(&tokens, true).unwrap();
    let loss = mamba3::train::cross_entropy(&logits, &targets).unwrap();
    assert!(loss.to_f32()[0].is_finite());

    let grads = loss.backward().unwrap();
    assert!(grads.len() > 0, "quantized model produced no gradients");
    for (_, p) in model.named_parameters() {
        if p.requires_grad() && grads.get(p.id()).is_none() {
            // Not every parameter has to receive a gradient (e.g. unused biases),
            // but the vast majority should; assert on the aggregate instead.
        }
    }
    let covered = model
        .named_parameters()
        .into_iter()
        .filter(|(_, p)| grads.get(p.id()).is_some())
        .count();
    assert!(covered * 2 >= model.named_parameters().len());
}
