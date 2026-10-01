//! GM4: the ragged bidirectional scan and the last-position forward block.
//!
//! Every test holds [`LOCK`]: one of them pins launch counts and another reads
//! the process-wide matmul log.

#![cfg(feature = "backend")]

use mamba3::autograd::Var;
use mamba3::backend::{Device, launch_count, reset_launch_count};
use mamba3::backends::Auto;
use mamba3::models::entity::blocks::{BiBlock, ForwardBlock};
use mamba3::models::mamba3::Mamba3MixerConfig;
use mamba3::nn::lora::LoraConfig;
use mamba3::nn::module::Module;
use mamba3::nn::quant::QuantConfig;
use mamba3::ssm::config::SsmConfig;
use mamba3::tensor::ops::matmul::{start_matmul_log, take_matmul_log};
use mamba3::tensor::ops::movement::{RaggedLengths, reverse_bands, reverse_bands_ragged};
use mamba3::tensor::ops::random::Rng;
use mamba3::tensor::Tensor;

type R = Auto;

static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn lock() -> std::sync::MutexGuard<'static, ()> {
    LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn dev() -> Device<R> {
    Device::<R>::default()
}

fn frand(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.max(1);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 11) as f32) / (u64::MAX >> 11) as f32 * 2.0 - 1.0
        })
        .collect()
}

fn assert_close(actual: &[f32], expected: &[f32], tol: f32, what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= tol * (1.0 + e.abs()),
            "{what}: index {i} got {a}, want {e}"
        );
    }
}

/// One head per direction, as the graph model uses it.
fn ssm(d_model: usize) -> SsmConfig {
    SsmConfig {
        d_model,
        n_heads: 1,
        head_dim: d_model,
        d_state: 8,
        n_groups: 1,
        chunk_size: 4,
        ..Default::default()
    }
}

#[test]
fn ragged_reversal_matches_a_host_reference() {
    let _guard = lock();
    let device = dev();
    let (rows, seq, inner) = (3usize, 6usize, 8usize);
    let host = frand(rows * seq * inner, 3);
    let x = Tensor::<R, f32>::from_f32(&host, vec![rows, seq, inner], &device).unwrap();
    let bands = [(2usize, 4usize), (6, 8)];
    let lengths = [6u32, 4, 1];
    let ragged = RaggedLengths::<R>::from_host(&lengths, seq, &device).unwrap();
    assert!(!ragged.all_full());
    let got = reverse_bands_ragged(&x, 1, &bands, &ragged).unwrap();
    let mut want = host.clone();
    for b in 0..rows {
        let len = lengths[b] as usize;
        for d in 0..len {
            for &(start, end) in &bands {
                for i in start..end {
                    want[(b * seq + d) * inner + i] = host[(b * seq + (len - 1 - d)) * inner + i];
                }
            }
        }
    }
    assert_eq!(got.to_f32(), want);
    // An involution.
    let back = reverse_bands_ragged(&got, 1, &bands, &ragged).unwrap();
    assert_eq!(back.to_f32(), host);

    // Full rows are the plain reversal.
    let full = RaggedLengths::<R>::from_host(&[6, 6, 6], seq, &device).unwrap();
    assert!(full.all_full());
    assert_eq!(
        reverse_bands_ragged(&x, 1, &bands, &full).unwrap().to_f32(),
        reverse_bands(&x, 1, &bands).unwrap().to_f32()
    );
    // The same lengths without the flag go through the ragged kernel and agree.
    let unflagged = RaggedLengths::new(full.ids().clone(), seq, false).unwrap();
    assert_eq!(
        reverse_bands_ragged(&x, 1, &bands, &unflagged).unwrap().to_f32(),
        reverse_bands(&x, 1, &bands).unwrap().to_f32()
    );

    // Through the tape: its own adjoint.
    let leaf = Var::traced(x.clone());
    let weights = Tensor::<R, f32>::from_f32(&frand(rows * seq * inner, 9), vec![rows, seq, inner], &device)
        .unwrap();
    let loss = leaf
        .reverse_bands_ragged(1, &bands, &ragged)
        .unwrap()
        .mul(&Var::constant(weights.clone()))
        .unwrap()
        .sum()
        .unwrap();
    let grads = loss.backward_retain().unwrap();
    assert_eq!(
        grads.node(leaf.node().unwrap()).unwrap().to_f32(),
        reverse_bands_ragged(&weights, 1, &bands, &ragged).unwrap().to_f32()
    );

    assert!(RaggedLengths::<R>::from_host(&[7, 1, 1], seq, &device).is_err());
    let two = RaggedLengths::<R>::from_host(&[6, 4], seq, &device).unwrap();
    assert!(reverse_bands_ragged(&x, 1, &bands, &two).is_err());
    let short = RaggedLengths::<R>::from_host(&[5, 4, 1], 5, &device).unwrap();
    assert!(reverse_bands_ragged(&x, 1, &bands, &short).is_err());
    assert!(reverse_bands_ragged(&x, 0, &bands, &ragged).is_err());
}

/// Gradients of `loss` for every parameter of `module`, by name.
fn parameter_grads<M: Module<R, f32>>(module: &M, loss: &Var<R, f32>) -> Vec<(String, Vec<f32>)> {
    let grads = loss.backward_retain().unwrap();
    module
        .named_parameters()
        .into_iter()
        .map(|(name, param)| {
            let grad = grads
                .get(param.id())
                .unwrap_or_else(|| panic!("no gradient for {name}"))
                .to_f32();
            (name, grad)
        })
        .collect()
}

#[test]
fn padding_cannot_reach_a_real_position() {
    let _guard = lock();
    let device = dev();
    let d = 16usize;
    let (short, max) = (5usize, 9usize);
    let block =
        BiBlock::<R, f32>::new(d, &ssm(d), 1e-5, 1, &device, &mut Rng::seeded(7)).unwrap();
    let alone_host = frand(short * d, 11);
    let other_host = frand(max * d, 12);
    let weights_host = frand(short * d, 13);

    // The short sequence alone.
    let alone = Var::traced(
        Tensor::<R, f32>::from_f32(&alone_host, vec![1, short, d], &device).unwrap(),
    );
    let weights = Tensor::<R, f32>::from_f32(&weights_host, vec![1, short, d], &device).unwrap();
    let out_alone = block.apply(&alone).unwrap();
    let loss_alone = out_alone
        .mul(&Var::constant(weights))
        .unwrap()
        .sum()
        .unwrap();
    let want = out_alone.to_f32();
    let want_grads = parameter_grads(&block, &loss_alone);
    let want_input = loss_alone
        .backward_retain()
        .unwrap()
        .node(alone.node().unwrap())
        .unwrap()
        .to_f32();

    // The same sequence padded with loud values, next to a longer row.
    let mut padded_host = Vec::with_capacity(2 * max * d);
    padded_host.extend_from_slice(&alone_host);
    padded_host.extend(frand((max - short) * d, 14).iter().map(|v| v * 1.0e3));
    padded_host.extend_from_slice(&other_host);
    let mut mask_host = vec![0.0f32; 2 * max * d];
    mask_host[..short * d].copy_from_slice(&weights_host);
    let padded = Var::traced(
        Tensor::<R, f32>::from_f32(&padded_host, vec![2, max, d], &device).unwrap(),
    );
    let lengths = RaggedLengths::<R>::from_host(&[short as u32, max as u32], max, &device).unwrap();
    let out = block.apply_ragged(&padded, &lengths).unwrap();
    let got = out.to_f32();
    assert_close(&got[..short * d], &want, 1e-5, "real positions of the padded row");

    // The full row equals itself alone.
    let other = Var::constant(
        Tensor::<R, f32>::from_f32(&other_host, vec![1, max, d], &device).unwrap(),
    );
    assert_close(
        &got[max * d..],
        &block.apply(&other).unwrap().to_f32(),
        1e-5,
        "the full row",
    );

    // The plain path lets the padding in: its error on the real positions is
    // far above the ragged path's rounding.
    let worst_of = |got: &[f32]| {
        got[..short * d]
            .iter()
            .zip(&want)
            .map(|(a, e)| (a - e).abs())
            .fold(0.0f32, f32::max)
    };
    let ragged_error = worst_of(&got);
    let leak = worst_of(&block.apply(&padded).unwrap().to_f32());
    assert!(
        leak > 20.0 * ragged_error.max(1e-7),
        "the plain reversal should leak padding (leak {leak}, ragged error {ragged_error})"
    );

    // Gradients through the real positions agree, for the input and for every
    // parameter.
    let mask = Tensor::<R, f32>::from_f32(&mask_host, vec![2, max, d], &device).unwrap();
    let loss = out.mul(&Var::constant(mask)).unwrap().sum().unwrap();
    let grads = parameter_grads(&block, &loss);
    for ((name, got), (_, want)) in grads.iter().zip(&want_grads) {
        assert_close(got, want, 2e-4, &format!("gradient of {name}"));
    }
    let input_grad = loss
        .backward_retain()
        .unwrap()
        .node(padded.node().unwrap())
        .unwrap()
        .to_f32();
    assert_close(&input_grad[..short * d], &want_input, 2e-4, "gradient of the real rows");
    assert!(
        input_grad[short * d..].iter().all(|&g| g == 0.0),
        "nothing flows into the padding or the other row"
    );
}

#[test]
fn full_lengths_are_the_plain_path() {
    let _guard = lock();
    let device = dev();
    let d = 16usize;
    let block =
        BiBlock::<R, f32>::new(d, &ssm(d), 1e-5, 1, &device, &mut Rng::seeded(3)).unwrap();
    let x = Var::constant(Tensor::<R, f32>::from_f32(&frand(2 * 9 * d, 5), vec![2, 9, d], &device).unwrap());
    let full = RaggedLengths::<R>::from_host(&[9, 9], 9, &device).unwrap();
    // Warm both paths so neither count includes a first-use upload.
    block.apply(&x).unwrap();
    block.apply_ragged(&x, &full).unwrap();
    reset_launch_count();
    let plain = block.apply(&x).unwrap().to_f32();
    let plain_launches = launch_count();
    reset_launch_count();
    let ragged = block.apply_ragged(&x, &full).unwrap().to_f32();
    assert_eq!(launch_count(), plain_launches, "same launches");
    assert_eq!(ragged, plain, "bit for bit");

    // A forward block ignores the lengths: padding is already last.
    let forward =
        ForwardBlock::<R, f32>::new(d, &ssm(d), 1e-5, 1, &device, &mut Rng::seeded(3)).unwrap();
    let ragged_lengths = RaggedLengths::<R>::from_host(&[4, 9], 9, &device).unwrap();
    assert_eq!(
        forward.apply_ragged(&x, &ragged_lengths).unwrap().to_f32(),
        forward.apply(&x).unwrap().to_f32()
    );
    // Lengths that do not describe the batch are refused.
    let wrong = RaggedLengths::<R>::from_host(&[4, 9, 9], 9, &device).unwrap();
    assert!(block.apply_ragged(&x, &wrong).is_err());
    let wrong = RaggedLengths::<R>::from_host(&[4, 8], 8, &device).unwrap();
    assert!(block.apply_ragged(&x, &wrong).is_err());
}

#[test]
fn activation_quantisation_is_refused_on_a_padded_batch() {
    let _guard = lock();
    let device = dev();
    let d = 16usize;
    let mut cfg = ssm(d);
    cfg.n_heads = 2;
    cfg.n_groups = 2;
    let mixer = Mamba3MixerConfig::new(cfg).with_activation_quant(QuantConfig::int8_activations());
    let block =
        BiBlock::<R, f32>::from_mixer_config(&mixer, 1e-5, &device, &mut Rng::seeded(1)).unwrap();
    let x = Var::constant(Tensor::<R, f32>::from_f32(&frand(2 * 6 * d, 5), vec![2, 6, d], &device).unwrap());
    let lengths = RaggedLengths::<R>::from_host(&[3, 6], 6, &device).unwrap();
    let err = block.apply_ragged(&x, &lengths).unwrap_err().to_string();
    assert!(err.contains("activation quantisation"), "{err}");
}

/// A forward block for each projection variant `apply_last` must handle.
fn forward_variants(d: usize, device: &Device<R>) -> Vec<(&'static str, ForwardBlock<R, f32>)> {
    let base = SsmConfig {
        d_model: d,
        n_heads: 2,
        head_dim: 8,
        d_state: 8,
        n_groups: 2,
        chunk_size: 4,
        ..Default::default()
    };
    let gated = SsmConfig {
        post_gate_norm: true,
        ..base.clone()
    };
    let build = |mixer: Mamba3MixerConfig| {
        ForwardBlock::<R, f32>::from_mixer_config(&mixer, 1e-5, device, &mut Rng::seeded(21))
            .unwrap()
    };
    let lora = build(
        Mamba3MixerConfig::new(base.clone())
            .with_lora(LoraConfig::builder().rank(4).build().unwrap()),
    );
    // A fresh adapter's `B` is zero, which would hide its gradients.
    for (name, param) in lora.named_parameters() {
        if name.ends_with("lora_b") {
            let shape = param.shape();
            param.set(
                Tensor::from_f32(
                    &frand(shape.num_elements(), 77)
                        .iter()
                        .map(|v| v * 0.1)
                        .collect::<Vec<_>>(),
                    shape,
                    device,
                )
                .unwrap(),
            );
        }
    }
    vec![
        ("plain", build(Mamba3MixerConfig::new(base.clone()))),
        ("post_gate_norm", build(Mamba3MixerConfig::new(gated))),
        (
            "weight quantisation",
            build(Mamba3MixerConfig::new(base).with_weight_quant(QuantConfig::int8_weights())),
        ),
        ("lora", lora),
    ]
}

#[test]
fn apply_last_is_the_last_row_of_apply() {
    let _guard = lock();
    let device = dev();
    let (b, t, d) = (3usize, 7usize, 12usize);
    let d_inner = 16usize;
    let x_host = frand(b * t * d, 31);
    let weights = Tensor::<R, f32>::from_f32(&frand(b * d, 32), vec![b, d], &device).unwrap();
    for (name, block) in forward_variants(d, &device) {
        let width = block.mixer().config().in_proj_width();

        let x = Var::traced(Tensor::<R, f32>::from_f32(&x_host, vec![b, t, d], &device).unwrap());
        let full = block
            .apply(&x)
            .unwrap()
            .slice(1, t - 1, 1)
            .unwrap()
            .reshape(vec![b, d])
            .unwrap();
        let loss_full = full.mul(&Var::constant(weights.clone())).unwrap().sum().unwrap();
        let want = full.to_f32();
        let want_grads = parameter_grads(&block, &loss_full);
        let want_input = loss_full
            .backward_retain()
            .unwrap()
            .node(x.node().unwrap())
            .unwrap()
            .to_f32();

        let x = Var::traced(Tensor::<R, f32>::from_f32(&x_host, vec![b, t, d], &device).unwrap());
        start_matmul_log();
        let last = block.apply_last(&x).unwrap();
        let products = take_matmul_log();
        assert_eq!(last.dims(), &[b, d], "{name}");
        assert_close(&last.to_f32(), &want, 1e-5, &format!("{name}: last row"));

        let loss = last.mul(&Var::constant(weights.clone())).unwrap().sum().unwrap();
        let grads = parameter_grads(&block, &loss);
        assert_eq!(grads.len(), want_grads.len());
        for ((param, got), (_, want)) in grads.iter().zip(&want_grads) {
            assert_close(got, want, 2e-4, &format!("{name}: gradient of {param}"));
            assert!(
                got.iter().any(|&g| g != 0.0),
                "{name}: the gradient of {param} is all zero"
            );
        }
        let input_grad = loss
            .backward_retain()
            .unwrap()
            .node(x.node().unwrap())
            .unwrap()
            .to_f32();
        assert_close(&input_grad, &want_input, 2e-4, &format!("{name}: input gradient"));

        // The products of the forward pass: rows are `batch · m`.
        let rows = |p: &mamba3::tensor::ops::matmul::MatmulShape| p.batch * p.m;
        let out_proj: Vec<_> = products
            .iter()
            .filter(|p| p.k == d_inner && p.n == d)
            .collect();
        assert!(!out_proj.is_empty(), "{name}: no output projection in {products:?}");
        assert!(
            out_proj.iter().all(|p| rows(p) == b),
            "{name}: the output projection ran on {out_proj:?}, not on {b} rows"
        );
        let all_rows: Vec<_> = products
            .iter()
            .filter(|p| p.k == d && rows(p) == b * t && p.n > 2 * d_inner)
            .collect();
        assert_eq!(all_rows.len(), 1, "{name}: all-rows projections in {products:?}");
        if name == "lora" {
            assert_eq!(all_rows[0].n, width, "a LoRA projection is not split");
        } else {
            assert_eq!(all_rows[0].n, width - d_inner, "{name}: the gate band is not projected for every row");
            assert!(
                products
                    .iter()
                    .any(|p| p.k == d && rows(p) == b && p.n == d_inner),
                "{name}: no last-row gate projection in {products:?}"
            );
        }
    }
}

#[test]
fn apply_last_with_activation_quantisation_is_apply_then_slice() {
    let _guard = lock();
    let device = dev();
    let (b, t, d) = (3usize, 7usize, 12usize);
    let cfg = SsmConfig {
        d_model: d,
        n_heads: 2,
        head_dim: 8,
        d_state: 8,
        n_groups: 2,
        chunk_size: 4,
        ..Default::default()
    };
    let mixer = Mamba3MixerConfig::new(cfg).with_activation_quant(QuantConfig::int8_activations());
    let block =
        ForwardBlock::<R, f32>::from_mixer_config(&mixer, 1e-5, &device, &mut Rng::seeded(4)).unwrap();
    let x = Var::constant(Tensor::<R, f32>::from_f32(&frand(b * t * d, 6), vec![b, t, d], &device).unwrap());
    // One training pass so the observers hold a range, then freeze them.
    block.apply(&x).unwrap();
    block.set_training(false);
    let want = block
        .apply(&x)
        .unwrap()
        .slice(1, t - 1, 1)
        .unwrap()
        .reshape(vec![b, d])
        .unwrap()
        .to_f32();
    assert_eq!(block.apply_last(&x).unwrap().to_f32(), want);
}

#[test]
fn apply_last_refuses_what_it_cannot_do() {
    let _guard = lock();
    let device = dev();
    let d = 16usize;
    let forward =
        ForwardBlock::<R, f32>::new(d, &ssm(d), 1e-5, 1, &device, &mut Rng::seeded(3)).unwrap();
    let x = Var::constant(Tensor::<R, f32>::from_f32(&frand(2 * 5 * d, 5), vec![2, 5, d], &device).unwrap());
    // A single position is its own last row.
    let one = x.slice(1, 0, 1).unwrap();
    assert_close(
        &forward.apply_last(&one).unwrap().to_f32(),
        &forward.apply(&one).unwrap().to_f32(),
        1e-5,
        "a single position",
    );
    let bi = BiBlock::<R, f32>::new(d, &ssm(d), 1e-5, 1, &device, &mut Rng::seeded(3)).unwrap();
    let err = bi.mixer().apply_last(&x).unwrap_err().to_string();
    assert!(err.contains("bidirectional"), "{err}");
    let wrong = Var::constant(Tensor::<R, f32>::zeros(vec![2, 5, d + 1], &device));
    assert!(forward.mixer().apply_last(&wrong).is_err());
}
