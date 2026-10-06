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
    let est = Ms2MemoryEstimate::generation(&model, 37_859, 8, 8, 512, 22, 32, 4).unwrap();
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
        "decode_scratch",
        "readout",
        "window",
        "counters",
        "cand",
        "cand_feat",
        "formula_head",
        "formula_scores",
        "formula_mask",
        "formula_top",
        "formula_top_log_prob",
        "formula_top_counts",
        "top_count",
    ] {
        assert!(est.get(name).is_some(), "generation item {name} is present");
    }
    // P5.9 (task T3): one bank — the production loop steps the carries in
    // place, and `tests/ms2_fused_step.rs` pins the in-place step against
    // the functional step after every step, bit-equal on the cpu runtime.
    // T3F: the functional paths charge two banks plus the freeze's
    // replacement tensors (see `v0_functional_carry_estimate`); the
    // production form prices the in-place path, with a zero
    // `decode_functional_step` item so the order is stable across modes.
    assert_eq!(est.get("decoder_carries"), Some(8_421_376));
    assert_eq!(est.get("decode_functional_step"), Some(0));
    assert_eq!(est.get("formula_table"), Some(37_859 * 88 + 1024 * 4));
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
fn v0_functional_carry_estimate() {
    // T3F finding 2: off the in-place path the old caches stay live while
    // the new caches are constructed (two banks in `decoder_carries`),
    // and the composed freeze builds replacement tensors (one bank in
    // `decode_functional_step`). At the V0 shapes one bank is 8,421,376
    // bytes (see `v0_carries_match_contracts_section_9`).
    let model = ModelConfig::v0();
    let one_bank = 8_421_376u64;
    let inplace =
        Ms2MemoryEstimate::generation_for_decode_mode(&model, true, 37_859, 8, 8, 512, 22, 32, 4)
            .unwrap();
    assert_eq!(inplace.get("decoder_carries"), Some(one_bank));
    assert_eq!(inplace.get("decode_functional_step"), Some(0));
    let functional =
        Ms2MemoryEstimate::generation_for_decode_mode(&model, false, 37_859, 8, 8, 512, 22, 32, 4)
            .unwrap();
    assert_eq!(functional.get("decoder_carries"), Some(2 * one_bank));
    assert_eq!(functional.get("decode_functional_step"), Some(one_bank));
    assert_eq!(
        functional.total().unwrap() - inplace.total().unwrap(),
        2 * one_bank,
        "the functional path costs exactly two further banks"
    );
    // The production constructor is the in-place form.
    let production = Ms2MemoryEstimate::generation(&model, 37_859, 8, 8, 512, 22, 32, 4).unwrap();
    assert_eq!(production.items, inplace.items);
    println!(
        "V0 carry items: in-place decoder_carries {one_bank} + functional_step 0; functional {} + {}",
        2 * one_bank,
        one_bank,
    );
}

#[test]
fn check_limit_names_total_limit_and_largest_item() {
    let est = Ms2MemoryEstimate::generation(&ModelConfig::v0(), 37_859, 8, 8, 512, 22, 32, 4).unwrap();
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
    let err = Ms2MemoryEstimate::generation(&ModelConfig::v0(), 37_859, u64::MAX / 2, 8, 512, 22, 32, 4)
        .unwrap_err();
    assert!(matches!(err, Error::Config(_)), "{err}");
    assert!(err.to_string().contains("overflow"), "{err}");
}

#[test]
fn v0_training_estimate_has_training_items_only() {
    let model = ModelConfig::v0();
    let est = Ms2MemoryEstimate::training(&model, 37_859, 8, 16, 512, 22, 32).unwrap();
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
        // V1 §1.3 complete formula workspace, gold path and packed readout.
        "window",
        "counters",
        "cand",
        "cand_feat",
        "formula_head",
        "formula_scores",
        "formula_mask",
        "formula_top",
        "formula_top_log_prob",
        "formula_top_counts",
        "top_count",
        "gold_counts",
        "gold_feat",
        "gold_formula_head",
        "gold_slot",
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
    // The allowlist is independent of hardware capability: on the CPU
    // backend bf16 validates even when the probe reports no support, and a
    // non-CPU backend refuses it with "not validated for this backend".
    let mut bf16 = ModelConfig::v0();
    bf16.dtype = DType::BF16;
    let no_bf16 = Ms2Capabilities {
        bf16_supported: false,
        ..caps.clone()
    };
    if caps.backend == "cpu" {
        no_bf16.check(&bf16).expect("bf16 validates on the CPU backend");
        caps.check(&bf16).expect("bf16 validates on the CPU backend");
    } else {
        let err = no_bf16.check(&bf16).unwrap_err();
        assert!(matches!(err, Error::Unsupported(_)), "{err}");
        assert!(
            err.to_string().contains("not validated for this backend"),
            "{err}"
        );
    }
    let other = Ms2Capabilities {
        backend: "cuda".to_string(),
        ..caps.clone()
    };
    let err = other.check(&bf16).unwrap_err();
    assert!(matches!(err, Error::Unsupported(_)), "{err}");
    assert!(
        err.to_string().contains("not validated for this backend"),
        "the bf16 refusal states it is not validated for this backend: {err}"
    );
    // f16 is refused even where the device reports support: it is not
    // validated for the MS2 model on any backend.
    let mut f16 = ModelConfig::v0();
    f16.dtype = DType::F16;
    let err = caps.check(&f16).unwrap_err();
    assert!(matches!(err, Error::Unsupported(_)), "{err}");
    assert!(
        err.to_string().contains("not validated"),
        "the f16 refusal states it is not validated: {err}"
    );
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
    let est = Ms2MemoryEstimate::generation(&ModelConfig::v0(), 37_859, 8, 8, 512, 22, 32, 4).unwrap();
    assert_eq!(est.get("cache_gather"), Some(kv + atom));
    // Overflow is an error, not a wrapped size.
    assert!(cache_gather_per_step(u64::MAX / 2, 8, 128, 128, 2, 16, 4).is_err());
}

#[test]
fn formula_items_scale_exactly_with_m() {
    // V1 §1.3/B1-fix: every M-dependent item has the exact documented byte
    // size at M = 32/128/512/2048 (B = 8, d = 128, F32). Generation and
    // training share the workspace shapes; training doubles the head
    // activations/scores and adds the gold path (here only the M-dependent
    // gold items are pinned; M-independent `gold_counts`/`gold_slot` are
    // covered by presence above).
    let model = ModelConfig::v0();
    let (b, d, elem) = (8u64, 128u64, 4u64);
    for m in [32u64, 128, 512, 2048] {
        let gen_est = Ms2MemoryEstimate::generation(&model, 37_859, b, 8, 512, 22, m, 4).unwrap();
        assert_eq!(gen_est.get("window"), Some(b * m * 2 * 4), "M={m} window");
        assert_eq!(gen_est.get("counters"), Some(b * 5 * 4), "M={m} counters");
        assert_eq!(gen_est.get("cand"), Some(b * m * 13 * 4), "M={m} cand");
        assert_eq!(
            gen_est.get("cand_feat"),
            Some(b * m * 10 * elem),
            "M={m} cand_feat"
        );
        assert_eq!(
            gen_est.get("formula_head"),
            Some(3 * b * m * d * elem),
            "M={m} formula_head"
        );
        assert_eq!(
            gen_est.get("formula_scores"),
            Some(b * m * elem),
            "M={m} formula_scores"
        );
        assert_eq!(
            gen_est.get("formula_mask"),
            Some(b * m * elem),
            "M={m} formula_mask"
        );
        assert_eq!(
            gen_est.get("formula_top_log_prob"),
            Some(b * 4 * elem),
            "M={m} top_lp"
        );
        assert_eq!(gen_est.get("top_count"), Some(b * 4), "M={m} top_count");
        // The packed readout is exactly the single batched read: the seven id
        // buffers (`actions`, `top = B*F*2`, `top_count = B`,
        // `counters = B*5`, `summary = B*2`, `traj_alloc = B*K*12`,
        // `identity = B*K*2`) at 4 bytes plus the two float buffers
        // (`top_log_prob = B*F`, `stats = B*3`) at `elem` bytes — the shared
        // `generation_readout_counts` layout, so this pins the actual read
        // rather than the estimator's expression.
        let readout = gen_est.get("readout").unwrap();
        let (want_ids, want_floats) =
            Ms2MemoryEstimate::generation_readout_counts(b, 8, 4, 22, model.max_atoms as u64)
                .unwrap();
        let mut want = 0u64;
        for len in want_ids {
            want += len * 4;
        }
        for len in want_floats {
            want += len * elem;
        }
        assert_eq!(readout, want, "M={m} readout is the packed read");
        let actions = gen_est.get("actions").unwrap();
        assert_eq!(
            readout,
            actions
                + (b * 4 * 2 + b + b * 5 + b * 2 + b * 8 * 12 + b * 8 * 2) * 4
                + (b * 4 + b * 3) * elem,
            "M={m} readout packs top/top_count/counters/summary/traj_alloc/identity/top_log_prob/stats"
        );
        let train_est = Ms2MemoryEstimate::training(&model, 37_859, b, 16, 512, 22, m).unwrap();
        assert_eq!(train_est.get("window"), Some(b * m * 2 * 4), "M={m} train window");
        assert_eq!(train_est.get("counters"), Some(b * 5 * 4), "M={m} train counters");
        assert_eq!(train_est.get("cand"), Some(b * m * 13 * 4), "M={m} train cand");
        assert_eq!(
            train_est.get("cand_feat"),
            Some(b * m * 10 * elem),
            "M={m} train cand_feat"
        );
        assert_eq!(
            train_est.get("formula_head"),
            Some(6 * b * m * d * elem),
            "M={m} train formula_head"
        );
        assert_eq!(
            train_est.get("formula_scores"),
            Some(2 * b * m * elem),
            "M={m} train formula_scores"
        );
        assert_eq!(
            train_est.get("formula_mask"),
            Some(b * m * elem),
            "M={m} train formula_mask"
        );
        assert_eq!(
            train_est.get("gold_feat"),
            Some(2 * b * 10 * elem),
            "M={m} train gold_feat"
        );
        assert_eq!(
            train_est.get("gold_formula_head"),
            Some(2 * 3 * b * d * elem),
            "M={m} train gold_formula_head"
        );
        println!(
            "M={m}: gen window {} cand {} head {}; train window {} head {}",
            gen_est.get("window").unwrap(),
            gen_est.get("cand").unwrap(),
            gen_est.get("formula_head").unwrap(),
            train_est.get("window").unwrap(),
            train_est.get("formula_head").unwrap(),
        );
    }
}

#[test]
fn formula_top_chunk_scratch_accounted() {
    // Task F10 item B3: the chunk scratch (`B * ceil(M/64) * (elem + 4)`
    // bytes) is estimate item `formula_top_chunk_scratch` at EVERY `M` —
    // the `FormulaBuffers::chunk_score` / `chunk_slot` workspace buffers are
    // allocated once per bucket for every `M` (including below 256), so the
    // estimate covers the allocation in both the default routing and the
    // forced-on configuration. `ModelConfig::v0()` is f32 (`elem = 4`).
    let model = ModelConfig::v0();
    let elem = 4u64;
    for (m, b) in [(32u64, 8u64), (255, 8), (256, 8), (512, 8), (2048, 2)] {
        let est = Ms2MemoryEstimate::generation(&model, 100, b, 4, 64, 22, m, 4).unwrap();
        let want = b * m.div_ceil(64) * (elem + 4);
        assert_eq!(
            est.get("formula_top_chunk_scratch"),
            Some(want),
            "M={m} B={b}: chunk scratch"
        );
        println!("M={m} B={b}: formula_top_chunk_scratch={want}");
    }
}
