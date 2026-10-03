//! P2-A tests: MS2 device capabilities, memory estimates and parameter counts.

#![cfg(feature = "backend")]

use mamba3::backend::{DType, Device};
use mamba3::backends::Auto;
use mamba3::error::Error;
use mamba3::models::mamba3::{Mamba3Block, Mamba3BlockConfig};
use mamba3::models::ms2::contract::ModelConfig;
use mamba3::models::ms2::workspace::{
    MS2_MAX_KERNEL_ARRAYS, Ms2Capabilities, Ms2MemoryEstimate, block_parameter_count, carry_bytes,
    parameter_count,
};
use mamba3::nn::Module;
use mamba3::tensor::ops::random::Rng;

type R = Auto;

fn device() -> Device<R> {
    Device::<R>::default()
}

#[test]
fn carry_bytes_matches_the_design_figure() {
    // Architecture §1: at most 6 array bindings per MS2 kernel.
    assert_eq!(MS2_MAX_KERNEL_ARRAYS, 6);
    // Architecture §6.2: B = 8, K = 32, Ld = 4, h = 8, p = 32, s = 64 gives
    // 135,266,304 bytes (129 MiB) per bank and twice that for two banks.
    let one_bank = carry_bytes(8, 32, 4, 8, 32, 64, 0, 4).unwrap();
    assert_eq!(one_bank, 135_266_304);
    assert_eq!(one_bank / (1024 * 1024), 129);
    assert_eq!(2 * one_bank, 270_532_608);
    assert_eq!(2 * one_bank / (1024 * 1024), 258);
}

#[test]
fn v0_carries_match_contracts_section_9() {
    // Contracts §9: 2,105,344 elements per bank for the V0 shapes.
    let one_bank = carry_bytes(8, 8, 2, 4, 64, 32, 0, 4).unwrap();
    assert_eq!(one_bank, 8_421_376);
    assert_eq!(one_bank / 4, 2_105_344);
}

#[test]
fn v0_generation_estimate_contents_and_total() {
    let model = ModelConfig::v0();
    let est = Ms2MemoryEstimate::generation(&model, 37_859, 8, 8, 512, 22).unwrap();
    for name in [
        "weights",
        "formula_table",
        "raw_peaks",
        "peak_selection",
        "encoder_activations",
        "spectrum_memory",
        "decoder_carries",
        "graph_state",
        "actions",
        "atom_memory",
        "head_scratch",
        "readout",
    ] {
        assert!(est.get(name).is_some(), "generation item {name} is present");
    }
    assert_eq!(est.get("decoder_carries"), Some(2 * 8_421_376));
    assert_eq!(est.get("formula_table"), Some(37_859 * 48));
    let sum: u64 = est.items.iter().map(|(_, bytes)| *bytes).sum();
    assert_eq!(est.total().unwrap(), sum);
    assert!(est.check_limit(2 * 1024 * 1024 * 1024).is_ok());
    println!(
        "V0 generation estimate total: {} bytes",
        est.total().unwrap()
    );
    for (name, bytes) in &est.items {
        println!("  {name}: {bytes}");
    }
}

#[test]
fn check_limit_names_total_limit_and_largest_item() {
    let est = Ms2MemoryEstimate::generation(&ModelConfig::v0(), 37_859, 8, 8, 512, 22).unwrap();
    let total = est.total().unwrap();
    let (largest, _) = est.items.iter().max_by_key(|(_, b)| *b).unwrap();
    let err = est.check_limit(1).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains(&total.to_string()), "{msg}");
    assert!(msg.contains("1"), "{msg}");
    assert!(msg.contains(largest), "{msg}");
}

#[test]
fn generation_overflow_is_a_config_error() {
    let err = Ms2MemoryEstimate::generation(&ModelConfig::v0(), 37_859, u64::MAX / 2, 8, 512, 22)
        .unwrap_err();
    assert!(matches!(err, Error::Config(_)), "{err}");
    assert!(err.to_string().contains("overflow"), "{err}");
}

#[test]
fn v0_training_estimate_has_training_items_only() {
    let model = ModelConfig::v0();
    let est = Ms2MemoryEstimate::training(&model, 37_859, 8, 16, 512, 22).unwrap();
    for name in [
        "weights",
        "formula_table",
        "raw_peaks",
        "peak_selection",
        "encoder_activations",
        "spectrum_memory",
        "targets",
        "decoder_activations",
        "attention_scores",
        "gradients",
        "optimizer_moments",
        "activation_gradients",
        "atom_memory",
        "head_scratch",
        // P2.2 retained teacher-path tensors, each forward-plus-gradient
        // (see `Ms2MemoryEstimate` docs for the per-item shapes).
        "decoder_mixer_retained",
        "decoder_attention_retained",
        "decoder_head_positions_retained",
        "decoder_embed_retained",
        "encoder_scan_retained",
    ] {
        assert!(est.get(name).is_some(), "training item {name} is present");
    }
    for name in ["decoder_carries", "graph_state", "actions", "readout"] {
        assert!(est.get(name).is_none(), "generation item {name} is absent");
    }
    let sum: u64 = est.items.iter().map(|(_, b)| *b).sum();
    assert_eq!(est.total().unwrap(), sum);
    println!("V0 training estimate total: {} bytes", est.total().unwrap());
    for (name, bytes) in &est.items {
        println!("  {name}: {bytes}");
    }
}

#[test]
fn block_parameter_count_matches_a_real_block() {
    let device = device();
    let ssm = ModelConfig::v0().encoder;
    let mut rng = Rng::seeded(11);
    let block: Mamba3Block<R, f32> = Mamba3BlockConfig::new(ssm.clone())
        .init(&device, &mut rng)
        .unwrap();
    // The visitor walk behind `num_parameters` sees every parameter the block
    // owns; the formula must count exactly those elements.
    let actual = block.num_parameters() as u64;
    assert_eq!(block_parameter_count(&ssm).unwrap(), actual);
    println!(
        "parameter_count(&ModelConfig::v0()) = {}",
        parameter_count(&ModelConfig::v0()).unwrap()
    );
}

#[test]
fn capabilities_probe_and_check() {
    let device = device();
    let caps = Ms2Capabilities::probe(&device);
    assert!(caps.f32_supported);
    assert!(caps.check(&ModelConfig::v0()).is_ok());
    let small = Ms2Capabilities {
        max_bindings: 6,
        ..caps.clone()
    };
    let err = small.check(&ModelConfig::v0()).unwrap_err();
    assert!(err.to_string().contains("max_bindings"), "{err}");
    let no_bf16 = Ms2Capabilities {
        bf16_supported: false,
        ..caps.clone()
    };
    let mut bf16 = ModelConfig::v0();
    bf16.dtype = DType::BF16;
    let err = no_bf16.check(&bf16).unwrap_err();
    assert!(matches!(err, Error::Unsupported(_)), "{err}");
    println!("probed capabilities: {caps:?}");
}

#[test]
fn capabilities_timing_probe_returns_a_definite_value() {
    // P2.4: `timing` is determined from what `client.profile` actually
    // returns on the device (probed once with a trivial launch), never from
    // the backend name. On the current backend the probe must return a
    // definite value (`DeviceTimestamps` or `SystemTime`), not
    // `Unavailable`; `check` still refuses an unsupported dtype and a
    // too-small binding count.
    use mamba3::models::ms2::workspace::TimingMethod;
    let device = device();
    let caps = Ms2Capabilities::probe(&device);
    assert!(
        matches!(
            caps.timing,
            TimingMethod::DeviceTimestamps | TimingMethod::SystemTime
        ),
        "timing probe on {} returned {:?}, want a definite value",
        caps.backend,
        caps.timing
    );
    println!(
        "backend {} timing: {:?} (reports_reserved_bytes={})",
        caps.backend, caps.timing, caps.reports_reserved_bytes
    );
    // `check` still refuses an unsupported dtype and a too-small binding
    // count alongside the new fields.
    let mut f16 = ModelConfig::v0();
    f16.dtype = DType::F16;
    let no_f16 = Ms2Capabilities {
        f16_supported: false,
        ..caps.clone()
    };
    assert!(matches!(
        no_f16.check(&f16).unwrap_err(),
        Error::Unsupported(_)
    ));
    let small = Ms2Capabilities {
        max_bindings: 6,
        ..caps.clone()
    };
    let err = small.check(&ModelConfig::v0()).unwrap_err();
    assert!(err.to_string().contains("max_bindings"), "{err}");
}

#[test]
fn cache_gather_per_step_matches_shapes() {
    // P2.2: the `cache_gather` item is the per-step bytes read for the K/V
    // caches and the atom memory, computed from the shapes: K/V is
    // `B * K * (1 + N) * d * elem * 2 * Ld`, atom memory is
    // `B * K * A * d * elem. V0 sampling does no beam gather, so no
    // trajectory state is moved between slots.
    use mamba3::models::ms2::workspace::cache_gather_per_step;
    let (b, k, n, d, ld, a, elem) = (8u64, 8, 128, 128, 2, 16, 4);
    let kv = b * k * (1 + n) * d * elem * (2 * ld);
    let atom = b * k * a * d * elem;
    assert_eq!(
        cache_gather_per_step(b, k, n, d, ld, a, elem).unwrap(),
        kv + atom
    );
    // The generation estimate carries it as its own named item.
    let est = Ms2MemoryEstimate::generation(&ModelConfig::v0(), 37_859, 8, 8, 512, 22).unwrap();
    assert_eq!(est.get("cache_gather"), Some(kv + atom));
    // Overflow is an error, not a wrapped size.
    assert!(cache_gather_per_step(u64::MAX / 2, 8, 128, 128, 2, 16, 4).is_err());
}
