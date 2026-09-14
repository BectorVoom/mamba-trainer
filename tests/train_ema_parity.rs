//! T1 P5: the moving average agrees between Rust and Python.
//!
//! One fixed scenario — what `mamba3_rl.PpoLearner(policy, RecallEnv(...),
//! ema=EmaConfig(0.9))` does, spelled out in Rust — and its result recorded per
//! backend in `tests/golden/ema_parity.json`. The Python test
//! `bindings/python/tests/test_ema.py::test_rust_python_parity` runs the same
//! scenario through the bindings and compares with the same record.
//!
//! Regenerate a backend's record (after a change that is meant to move the
//! numbers) with
//!
//! ```text
//! MAMBA3_WRITE_EMA_PARITY=1 cargo test --release --no-default-features \
//!     --features cpu|wgpu --test train_ema_parity
//! ```
//!
//! Alone in its binary because it pins the matmul kernel, which is process-wide:
//! on a GPU the default picks kernels by timing, per process, and they agree only
//! to a few ulp, so this process and the Python one both pin `block_tiled`.

#![cfg(feature = "backend")]

use mamba3::backend::Device;
use mamba3::backends::Auto;
use mamba3::rl::{Collector, Mamba3PolicyConfig, PpoConfig, PpoTask, RecallEnv, VecEnv};
use mamba3::ssm::config::{Discretization, StateDynamics};
use mamba3::tensor::ops::matmul::{parse_matmul_kernel, set_default_kernel};
use mamba3::train::{AdamWConfig, Ema, EmaConfig, LrSchedule, Trainer, TrainerConfig};
use serde_json::{Value, json};

type R = Auto;

fn golden_path() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden/ema_parity.json")
}

/// The scenario, as the Python test reads it back.
fn scenario() -> Value {
    json!({
        "envs": 8, "symbols": 4, "horizon": 4, "env_seed": 1,
        "d_model": 32, "n_layers": 2, "n_heads": 2, "head_dim": 16, "d_state": 8,
        "chunk_size": 4, "policy_seed": 7,
        "steps": 8, "learning_rate": 1e-3, "decay": 0.9, "rounds": 2, "epochs": 2,
        "matmul_kernel_off_cpu": "block_tiled"
    })
}

#[test]
fn rust_python_parity() {
    let device = Device::<R>::default();
    if device.name() != "cpu" {
        set_default_kernel(parse_matmul_kernel("block_tiled").expect("a kernel name"));
    }
    let s = scenario();
    let n = |key: &str| s[key].as_u64().expect("an integer") as usize;

    let mut env = RecallEnv::<R, f32>::new(
        n("envs"),
        n("symbols"),
        n("horizon"),
        n("env_seed") as u64,
        &device,
    )
    .unwrap();
    // `PolicyConfig(obs_dim, action_dim, d_model, n_layers, n_heads=, head_dim=,
    // d_state=, chunk_size=, seed=)`, as the bindings build it.
    let mut config =
        Mamba3PolicyConfig::new(env.obs_dim(), env.action_dim(), n("d_model"), n("n_layers"));
    config.norm_eps = 1e-5;
    config.seed = n("policy_seed") as u64;
    config.ssm.head_dim = n("head_dim");
    config.ssm.n_heads = n("n_heads");
    config.ssm.n_groups = n("n_heads");
    config.ssm.d_state = n("d_state");
    config.ssm.chunk_size = n("chunk_size");
    config.ssm.discretization = Discretization::LearnedTrapezoid;
    config.ssm.dynamics = StateDynamics::Rotational;
    config.validate().unwrap();
    let policy = config.init::<R, f32>(&device).unwrap();
    let shadow = config.init::<R, f32>(&device).unwrap();

    // `PpoLearner`'s defaults: max_grad_norm 0.5, betas (0.9, 0.999), eps 1e-8,
    // no weight decay, a constant schedule, temperature 1, seed 0.
    let learning_rate = s["learning_rate"].as_f64().unwrap() as f32;
    let mut trainer = Trainer::new(
        TrainerConfig::builder()
            .learning_rate(learning_rate)
            .max_grad_norm(0.5)
            .schedule(LrSchedule::Constant)
            .build()
            .unwrap(),
        AdamWConfig::builder()
            .learning_rate(learning_rate)
            .betas(0.9, 0.999)
            .eps(1e-8)
            .weight_decay(0.0)
            .build()
            .init::<R, f32>(),
    )
    .with_ema(
        Ema::new(
            &policy,
            &shadow,
            EmaConfig::new(s["decay"].as_f64().unwrap() as f32),
        )
        .unwrap(),
    );
    let mut collector = Collector::new(&policy, n("envs"), n("steps"), env.obs_dim(), &device)
        .unwrap()
        .with_temperature(1.0)
        .with_seed(0);
    let ppo = PpoConfig::default();
    for _ in 0..n("rounds") {
        let report = collector.collect(&mut env).unwrap();
        let batch = collector.ppo_batch(&report, &ppo).unwrap();
        let task = PpoTask::new(&policy, ppo);
        for _ in 0..n("epochs") {
            trainer.step(&task, std::slice::from_ref(&batch)).unwrap();
        }
    }

    let ema = trainer.ema().unwrap();
    let average = ema.state_dict();
    // The first eight values of the average, in path order across entries.
    let first: Vec<f32> = average
        .entries
        .values()
        .flat_map(|t| t.data.iter().copied())
        .take(8)
        .collect();
    let record = json!({
        "ema_updates": ema.updates(),
        "ema_fingerprint": average.fingerprint(),
        "policy_fingerprint": mamba3::nn::Module::state_dict(&policy).fingerprint(),
        "first_values_bits": first.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "first_values": first,
    });

    let path = golden_path();
    let mut golden: Value = std::fs::read(&path)
        .ok()
        .map(|bytes| serde_json::from_slice(&bytes).expect("the parity record is JSON"))
        .unwrap_or_else(|| json!({"scenario": scenario(), "backends": {}}));
    let backend = device.name();
    if std::env::var_os("MAMBA3_WRITE_EMA_PARITY").is_some() {
        golden["scenario"] = scenario();
        golden["backends"][backend] = record;
        let text = serde_json::to_string_pretty(&golden).unwrap() + "\n";
        std::fs::write(&path, text).unwrap();
        println!("wrote the {backend} parity record to {path:?}");
        return;
    }
    assert_eq!(
        golden["scenario"],
        scenario(),
        "the recorded scenario is not this test's; regenerate the record"
    );
    let Some(want) = golden["backends"].get(backend) else {
        println!("skipped: rust_python_parity has no record for {backend} (see this file's docs)");
        return;
    };
    // The readable values are for people; the bits are the claim.
    let mut want = want.clone();
    let mut got = record;
    want.as_object_mut().unwrap().remove("first_values");
    got.as_object_mut().unwrap().remove("first_values");
    assert_eq!(
        got, want,
        "{backend}: the EMA scenario no longer matches its record"
    );
}
