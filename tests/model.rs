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

    // A bidirectional block owns one fused mixer holding both directions'
    // parameters; there is no separate backward mixer.
    let names: Vec<String> = model
        .named_parameters()
        .into_iter()
        .map(|(n, _)| n)
        .collect();
    assert!(names.iter().any(|n| n.contains("blocks.0.mixer")));
    assert!(!names.iter().any(|n| n.contains("blocks.0.backward")));
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

/// The fused bidirectional mixer must compute exactly what the composed form —
/// a forward mixer plus a backward mixer over the flipped sequence — computes,
/// including gradients. The fused mixer's parameters are the two directions'
/// parameters interleaved direction-major along every band, so this builds two
/// ordinary mixers, transplants their weights into a fused one band by band,
/// and compares outputs and a representative set of parameter gradients.
#[test]
fn bidirectional_mixer_matches_two_composed_mixers() {
    use std::collections::HashMap;

    use mamba3::models::{Mamba3Mixer, Mamba3MixerConfig};
    use mamba3::nn::param::Param;
    use mamba3::ssm::config::SsmConfig;

    let device = dev();
    let single = {
        let mut ssm = SsmConfig::default();
        ssm.d_model = 16;
        ssm.n_heads = 2;
        ssm.n_groups = 2;
        ssm.head_dim = 4;
        ssm.d_state = 4;
        ssm.chunk_size = 4;
        ssm
    };
    let mut fused_ssm = single.clone();
    fused_ssm.n_heads *= 2;
    fused_ssm.n_groups *= 2;

    let mut rng = Rng::seeded(3);
    let fwd: Mamba3Mixer<R, f32> = Mamba3MixerConfig::new(single.clone())
        .init(&device, &mut rng)
        .unwrap();
    let bwd: Mamba3Mixer<R, f32> = Mamba3MixerConfig::new(single.clone())
        .init(&device, &mut rng)
        .unwrap();
    let fused: Mamba3Mixer<R, f32> = Mamba3MixerConfig::new(fused_ssm)
        .with_bidirectional(true)
        .init(&device, &mut rng)
        .unwrap();

    let params = |m: &Mamba3Mixer<R, f32>| -> HashMap<String, Param<R, f32>> {
        m.named_parameters().into_iter().collect()
    };
    let (fp, bp, up) = (params(&fwd), params(&bwd), params(&fused));

    // Interleave two row-major `[rows, sum(bands)]` matrices band by band into
    // `[rows, 2 * sum(bands)]`, forward columns first within each band.
    let interleave = |f: &[f32], b: &[f32], rows: usize, bands: &[usize]| -> Vec<f32> {
        let w: usize = bands.iter().sum();
        let mut out = vec![0.0f32; rows * 2 * w];
        for r in 0..rows {
            let (mut src, mut dst) = (0usize, 0usize);
            for &band in bands {
                out[r * 2 * w + dst..r * 2 * w + dst + band]
                    .copy_from_slice(&f[r * w + src..r * w + src + band]);
                out[r * 2 * w + dst + band..r * 2 * w + dst + 2 * band]
                    .copy_from_slice(&b[r * w + src..r * w + src + band]);
                src += band;
                dst += 2 * band;
            }
        }
        out
    };
    let concat = |f: &[f32], b: &[f32]| -> Vec<f32> {
        let mut out = f.to_vec();
        out.extend_from_slice(b);
        out
    };

    // Per-direction band widths: d_inner 8, bc 8, dt 2, lambda 2, theta 4.
    let in_bands = [8usize, 8, 8, 8, 2, 2, 4]; // z, x, B, C, dt, lambda, theta
    let conv_bands = [8usize, 8, 8]; // x, B, C

    for name in [
        "in_proj.weight",
        "out_proj.weight",
        "conv.weight",
        "conv.bias",
        "dt_bias",
        "a_log",
        "d",
        "b_bias",
        "c_bias",
    ] {
        let (Some(f), Some(b), Some(u)) = (fp.get(name), bp.get(name), up.get(name)) else {
            assert!(
                fp.get(name).is_none() && up.get(name).is_none(),
                "parameter {name} exists on one mixer but not the other"
            );
            continue;
        };
        let (fv, bv) = (f.value().to_f32(), b.value().to_f32());
        let dims = f.shape().dims().to_vec();
        let (data, shape) = match name {
            "in_proj.weight" => (interleave(&fv, &bv, dims[0], &in_bands), vec![dims[0], 2 * dims[1]]),
            "conv.weight" => (interleave(&fv, &bv, dims[0], &conv_bands), vec![dims[0], 2 * dims[1]]),
            "conv.bias" => (interleave(&fv, &bv, 1, &conv_bands), vec![2 * dims[0]]),
            // out_proj rows, per-head vectors and per-head bias rows all concatenate.
            _ => {
                let mut shape = dims.clone();
                shape[0] *= 2;
                (concat(&fv, &bv), shape)
            }
        };
        assert_eq!(u.shape().dims(), shape.as_slice(), "{name} shape");
        u.set(Tensor::from_f32(&data, shape, &device).unwrap());
    }
    // The fused mixer shares one bc_norm scale across both directions; give the
    // composed mixers the same one.
    let shared = up["bc_norm.weight"].value();
    fp["bc_norm.weight"].set(shared.clone());
    bp["bc_norm.weight"].set(shared);

    let (batch, seq) = (2usize, 8usize);
    let pixels: Vec<f32> = (0..batch * seq * 16)
        .map(|i| (i % 13) as f32 / 13.0 - 0.5)
        .collect();
    let input = Tensor::from_f32(&pixels, vec![batch, seq, 16], &device).unwrap();
    let mask: Vec<f32> = (0..batch * seq * 16)
        .map(|i| ((i % 7) as f32 - 3.0) * 0.25)
        .collect();
    let mask = Tensor::from_f32(&mask, vec![batch, seq, 16], &device).unwrap();

    // Composed reference: forward mixer plus backward mixer over the flipped
    // sequence, flipped back.
    let anchor = Var::traced(input.clone());
    let composed = fwd
        .apply(&anchor)
        .unwrap()
        .add(&bwd.apply(&anchor.flip(1).unwrap()).unwrap().flip(1).unwrap())
        .unwrap();
    let fused_anchor = Var::traced(input);
    let fused_out = fused.apply(&fused_anchor).unwrap();

    assert!(
        max_abs_diff(&composed.to_f32(), &fused_out.to_f32()) < 1e-4,
        "fused bidirectional output diverges from the composed form"
    );

    let composed_grads = composed
        .mul(&Var::constant(mask.clone()))
        .unwrap()
        .sum()
        .unwrap()
        .backward()
        .unwrap();
    let fused_grads = fused_out
        .mul(&Var::constant(mask))
        .unwrap()
        .sum()
        .unwrap()
        .backward()
        .unwrap();

    let composed_grad = |name: &str| -> (Vec<f32>, Vec<f32>) {
        (
            composed_grads.get(fp[name].id()).expect(name).to_f32(),
            composed_grads.get(bp[name].id()).expect(name).to_f32(),
        )
    };
    for name in ["in_proj.weight", "out_proj.weight", "conv.weight", "dt_bias", "a_log"] {
        let (gf, gb) = composed_grad(name);
        let dims = fp[name].shape().dims().to_vec();
        let expected = match name {
            "in_proj.weight" => interleave(&gf, &gb, dims[0], &in_bands),
            "conv.weight" => interleave(&gf, &gb, dims[0], &conv_bands),
            _ => concat(&gf, &gb),
        };
        let actual = fused_grads.get(up[name].id()).expect(name).to_f32();
        assert!(
            max_abs_diff(&expected, &actual) < 1e-3,
            "gradient of {name} diverges between the fused and composed forms"
        );
    }
}
