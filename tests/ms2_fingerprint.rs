//! K10 tests: whole-parent fingerprint sidecar labels, head, loss and metrics.
//!
//! Standalone: the head reads the pooled spectrum vector `pool [B, d]` and
//! predicts the whole parent's fingerprint as auxiliary supervision. It is
//! never a target for an individual fragment or candidate.

#![cfg(feature = "backend")]

use mamba3::autograd::Var;
use mamba3::backend::{Device, check_launches};
use mamba3::backends::Auto;
use mamba3::models::ms2::dataset::ExportFile;
use mamba3::models::ms2::fingerprint::{
    FINGERPRINT_BITS, FINGERPRINT_VERSION, FingerprintHead, FingerprintLabels,
    bit_accuracy, cosine_per_spectrum, tanimoto,
};
use mamba3::nn::Module;
use mamba3::tensor::Tensor;
use mamba3::tensor::ops::random::Rng;
use mamba3::train::optim::{AdamWConfig, Optimizer};

type R = Auto;
type E = f32;

fn dev() -> Device<R> {
    Device::<R>::default()
}

/// Stable binary cross-entropy with logits, in `f64`.
fn stable_bce_f64(x: f64, z: f64) -> f64 {
    x.max(0.0) - x * z + (1.0 + (-x.abs()).exp()).ln()
}

/// Independent `f64` implementation of
/// [`FingerprintHead::loss`](mamba3::models::ms2::fingerprint::FingerprintHead::loss).
///
/// The rule: per-spectrum mean over the 1,024 bits, then the weighted mean
/// over spectra `sum_b w_b * l_b / max(sum_b w_b, eps)` with `eps` tiny
/// ([`FINGERPRINT_WEIGHT_EPS`](mamba3::models::ms2::fingerprint::FINGERPRINT_WEIGHT_EPS));
/// exactly 0 when the weight sum is 0.
fn reference_loss(logits: &[f32], targets: &[f32], weights: &[f32], batch: usize) -> f64 {
    let mut num = 0.0;
    let mut den = 0.0;
    for b in 0..batch {
        let mut mean = 0.0;
        for i in 0..FINGERPRINT_BITS {
            mean += stable_bce_f64(
                f64::from(logits[b * FINGERPRINT_BITS + i]),
                f64::from(targets[b * FINGERPRINT_BITS + i]),
            );
        }
        mean /= FINGERPRINT_BITS as f64;
        num += f64::from(weights[b]) * mean;
        den += f64::from(weights[b]);
    }
    if den == 0.0 {
        return 0.0;
    }
    num / den.max(1e-12)
}

fn spectrum_json(row: u64) -> serde_json::Value {
    serde_json::json!({
        "row": row, "spectrum_id": row, "adduct": 1, "polarity": 1,
        "precursor_mz_udalton": 200_000_000u32,
        "precursor_uncertainty_udalton": 50u32,
        "raw_peak_count": 1u32, "peak_id": [0u32],
        "mz_udalton": [100_000_000u32], "intensity": [1.0],
        "mz_uncertainty_udalton": 50u32,
        "collision_energy_ev": 30.0, "collision_energy_known": 1u8,
        "energy_count": 1u8, "instrument_class": 0u8
    })
}

fn export_json(keys: &[&str]) -> String {
    let molecules: Vec<serde_json::Value> = keys
        .iter()
        .enumerate()
        .map(|(i, k)| {
            serde_json::json!({
                "key": k, "identity_group": i as u64, "fold_identity": 2u32,
                "atoms": [2u8], "bonds": [], "spectra": [spectrum_json(i as u64)]
            })
        })
        .collect();
    serde_json::json!({
        "schema_version": 1u32, "chemistry": "ms2-chem-v0.1",
        "rdkit": "test", "source": "synthetic", "seed": 1u64, "n_raw": 64u32,
        "spectra_per_molecule": 1u32, "skipped_spectra": {},
        "subset": "train", "molecules": molecules
    })
    .to_string()
}

fn sidecar_json(keys: &[&str]) -> String {
    let molecules: Vec<serde_json::Value> = keys
        .iter()
        .enumerate()
        .map(|(i, k)| {
            serde_json::json!({
                "key": k, "smiles": "CCO",
                "bits": [0u32, (i as u32 * 7 + 3) % 1023 + 1, 1023u32]
            })
        })
        .collect();
    serde_json::json!({
        "schema_version": 1u32, "fingerprint": FINGERPRINT_VERSION,
        "bits": FINGERPRINT_BITS, "radius": 2u32,
        "generator": "rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)",
        "rdkit": "test", "source_export": "synth.json", "source_sha256": "00",
        "molecules": molecules
    })
    .to_string()
}

#[test]
fn constants() {
    assert_eq!(FINGERPRINT_BITS, 1024);
    assert_eq!(FINGERPRINT_VERSION, "morgan-r2-1024");
}

#[test]
fn loss_matches_f64_reference() {
    let device = dev();
    let mut rng = Rng::seeded(11);
    let d = 4usize;
    let batch = 3usize;
    let head = FingerprintHead::<R, E>::init(d, &device, &mut rng).unwrap();
    let mut data_rng = Rng::seeded(12);
    let pool_host = data_rng.uniform_vec(batch * d, -1.0, 1.0);
    let draws = data_rng.uniform_vec(batch * FINGERPRINT_BITS, 0.0, 1.0);
    let targets_host: Vec<f32> = draws
        .iter()
        .map(|&v| if v < 0.3 { 1.0 } else { 0.0 })
        .collect();
    let weights_host = vec![1.0f32, 0.0, 2.0];
    let pool_t = Tensor::<R, f32>::from_f32(&pool_host, vec![batch, d], &device).unwrap();
    let targets_t =
        Tensor::<R, f32>::from_f32(&targets_host, vec![batch, FINGERPRINT_BITS], &device)
            .unwrap();
    let weights_t = Tensor::<R, f32>::from_f32(&weights_host, vec![batch], &device).unwrap();
    let logits = head.logits(&Var::constant(pool_t)).unwrap();
    let loss = head.loss(&logits, &targets_t, &weights_t).unwrap();
    check_launches(&device).unwrap();
    let got = loss.try_to_f32().unwrap()[0];
    let logits_host = logits.try_to_f32().unwrap();
    let want = reference_loss(&logits_host, &targets_host, &weights_host, batch);
    assert!(
        (f64::from(got) - want).abs() <= 1e-5,
        "loss {got} differs from the f64 reference {want}"
    );
    check_launches(&device).unwrap();
}

#[test]
fn gradients_match_finite_differences() {
    let device = dev();
    let mut rng = Rng::seeded(21);
    let d = 4usize;
    let batch = 2usize;
    let head = FingerprintHead::<R, E>::init(d, &device, &mut rng).unwrap();
    let mut data_rng = Rng::seeded(22);
    let pool_host = data_rng.uniform_vec(batch * d, -1.0, 1.0);
    let pool_t = Tensor::<R, f32>::from_f32(&pool_host, vec![batch, d], &device).unwrap();
    let targets_host: Vec<f32> = (0..batch * FINGERPRINT_BITS)
        .map(|i| if i % 3 == 0 { 1.0 } else { 0.0 })
        .collect();
    let weights_host = vec![1.0f32, 0.5];
    let targets_t =
        Tensor::<R, f32>::from_f32(&targets_host, vec![batch, FINGERPRINT_BITS], &device)
            .unwrap();
    let weights_t = Tensor::<R, f32>::from_f32(&weights_host, vec![batch], &device).unwrap();
    let loss_of = |head: &FingerprintHead<R, f32>| -> f32 {
        let logits = head.logits(&Var::constant(pool_t.clone())).unwrap();
        head.loss(&logits, &targets_t, &weights_t)
            .unwrap()
            .try_to_f32()
            .unwrap()[0]
    };
    let logits = head.logits(&Var::constant(pool_t.clone())).unwrap();
    let loss = head.loss(&logits, &targets_t, &weights_t).unwrap();
    check_launches(&device).unwrap();
    let grads = loss.backward().unwrap();
    for (name, param) in head.named_parameters() {
        let analytic = grads
            .get(param.id())
            .unwrap_or_else(|| panic!("no gradient for {name}"))
            .to_f32();
        let shape = param.value().shape().dims().to_vec();
        let base = param.value().to_f32();
        for &idx in &[0usize, 1, 2] {
            assert!(idx < base.len(), "{name} too small");
            let mut up = base.clone();
            up[idx] += 1e-2;
            param.set(Tensor::<R, f32>::from_f32(&up, shape.clone(), &device).unwrap());
            let fu = loss_of(&head);
            let mut down = base.clone();
            down[idx] -= 1e-2;
            param.set(Tensor::<R, f32>::from_f32(&down, shape.clone(), &device).unwrap());
            let fd = loss_of(&head);
            param.set(Tensor::<R, f32>::from_f32(&base, shape.clone(), &device).unwrap());
            let numeric = (fu - fd) / 2e-2;
            let a = analytic[idx];
            assert!(
                (a - numeric).abs() <= 3e-2 * numeric.abs() + 2e-3,
                "{name}[{idx}]: analytic={a} numeric={numeric}"
            );
        }
    }
    check_launches(&device).unwrap();
}

#[test]
fn zero_weights_give_zero_loss_and_no_gradient() {
    let device = dev();
    let mut rng = Rng::seeded(31);
    let d = 4usize;
    let batch = 2usize;
    let head = FingerprintHead::<R, E>::init(d, &device, &mut rng).unwrap();
    let mut data_rng = Rng::seeded(32);
    let pool_host = data_rng.uniform_vec(batch * d, -1.0, 1.0);
    let targets_host: Vec<f32> = (0..batch * FINGERPRINT_BITS)
        .map(|i| if i % 5 == 0 { 1.0 } else { 0.0 })
        .collect();
    let pool_t = Tensor::<R, f32>::from_f32(&pool_host, vec![batch, d], &device).unwrap();
    let targets_t =
        Tensor::<R, f32>::from_f32(&targets_host, vec![batch, FINGERPRINT_BITS], &device)
            .unwrap();
    let weights_t = Tensor::<R, f32>::zeros(vec![batch], &device);
    let logits = head.logits(&Var::constant(pool_t)).unwrap();
    let loss = head.loss(&logits, &targets_t, &weights_t).unwrap();
    check_launches(&device).unwrap();
    assert_eq!(loss.try_to_f32().unwrap()[0], 0.0);
    let grads = loss.backward().unwrap();
    for (name, param) in head.named_parameters() {
        match grads.get(param.id()) {
            None => {}
            Some(g) => assert!(
                g.to_f32().iter().all(|&v| v == 0.0),
                "{name} has a nonzero gradient under zero weights"
            ),
        }
    }
    check_launches(&device).unwrap();
}

#[test]
fn learnable_task_reaches_high_accuracy() {
    let device = dev();
    let mut rng = Rng::seeded(41);
    let d = 8usize;
    let batch = 64usize;
    let head = FingerprintHead::<R, E>::init(d, &device, &mut rng).unwrap();
    // 16 bits linearly determined by the pool rows; the rest are constant 0.
    // Every spectrum of a molecule would share its whole-parent row; here
    // each row is its own molecule.
    let mut data_rng = Rng::seeded(42);
    let pool_host = data_rng.uniform_vec(batch * d, -1.0, 1.0);
    let mut targets_host = vec![0.0f32; batch * FINGERPRINT_BITS];
    for b in 0..batch {
        for j in 0..16 {
            if pool_host[b * d + j % d] > 0.0 {
                targets_host[b * FINGERPRINT_BITS + j] = 1.0;
            }
        }
    }
    let pool_t = Tensor::<R, f32>::from_f32(&pool_host, vec![batch, d], &device).unwrap();
    let pool_var = Var::constant(pool_t);
    let targets_t =
        Tensor::<R, f32>::from_f32(&targets_host, vec![batch, FINGERPRINT_BITS], &device)
            .unwrap();
    let weights_t = Tensor::<R, f32>::ones(vec![batch], &device);
    let mut optim = AdamWConfig::builder()
        .learning_rate(0.05)
        .weight_decay(0.0)
        .build()
        .init();
    let mut last = f32::INFINITY;
    for _ in 0..2000 {
        let logits = head.logits(&pool_var).unwrap();
        let loss = head.loss(&logits, &targets_t, &weights_t).unwrap();
        let grads = loss.backward().unwrap();
        optim.step(&head.parameters(), &grads).unwrap();
        last = loss.try_to_f32().unwrap()[0];
    }
    check_launches(&device).unwrap();
    assert!(last.is_finite(), "final loss is not finite: {last}");
    let logits = head.logits(&pool_var).unwrap();
    let got = logits.try_to_f32().unwrap();
    let mut correct = 0u64;
    for b in 0..batch {
        for j in 0..16 {
            let pred = got[b * FINGERPRINT_BITS + j] >= 0.0;
            let label = targets_host[b * FINGERPRINT_BITS + j] >= 0.5;
            if pred == label {
                correct += 1;
            }
        }
    }
    let accuracy = correct as f64 / (batch * 16) as f64;
    assert!(
        accuracy > 0.95,
        "bit accuracy on the 16 determined bits is {accuracy} (loss {last})"
    );
    check_launches(&device).unwrap();
}

#[test]
fn matches_export_accepts_right_sidecar() {
    let keys = ["KA", "KB", "KC"];
    let export = ExportFile::from_json(&export_json(&keys)).unwrap();
    let labels = FingerprintLabels::from_json(&sidecar_json(&keys)).unwrap();
    assert_eq!(labels.len(), 3);
    assert!(!labels.is_empty());
    assert_eq!(labels.keys, vec!["KA", "KB", "KC"]);
    labels.matches_export(&export, None).unwrap();
}

#[test]
fn matches_export_rejects_reordered_or_truncated_sidecar() {
    let keys = ["KA", "KB", "KC"];
    let export = ExportFile::from_json(&export_json(&keys)).unwrap();
    let reordered = FingerprintLabels::from_json(&sidecar_json(&["KB", "KA", "KC"])).unwrap();
    assert!(
        reordered.matches_export(&export, None).is_err(),
        "a reordered sidecar must not match"
    );
    let truncated = FingerprintLabels::from_json(&sidecar_json(&["KA", "KB"])).unwrap();
    assert!(
        truncated.matches_export(&export, None).is_err(),
        "a truncated sidecar must not match"
    );
    let extended = FingerprintLabels::from_json(&sidecar_json(&["KA", "KB", "KC", "KD"])).unwrap();
    assert!(
        extended.matches_export(&export, None).is_err(),
        "a longer sidecar must not match"
    );
}

#[test]
fn from_json_reads_a_file_path() {
    let keys = ["KA", "KB"];
    let text = sidecar_json(&keys);
    let path = std::env::temp_dir().join(format!("ms2_fp_sidecar_{}.json", std::process::id()));
    std::fs::write(&path, &text).unwrap();
    let from_path = FingerprintLabels::from_json(path.to_str().expect("temp path is utf-8"));
    std::fs::remove_file(&path).ok();
    let labels = from_path.unwrap();
    assert_eq!(labels.keys, vec!["KA", "KB"]);
    assert_eq!(labels.version, FINGERPRINT_VERSION);
}

#[test]
fn dense_hand_example() {
    let keys = ["A", "B"];
    let labels = FingerprintLabels::from_json(&sidecar_json(&keys)).unwrap();
    // Sidecar rows: A has bits {0, 4, 1023} (i=0: 0*7+3+1 = 4), B has {0, 11, 1023}.
    assert_eq!(labels.bits[0], vec![0, 4, 1023]);
    assert_eq!(labels.bits[1], vec![0, 11, 1023]);
    let got = labels.dense(&[1, 0, 1]);
    assert_eq!(got.len(), 3 * FINGERPRINT_BITS);
    let row_sum = |r: usize| -> f32 { got[r * FINGERPRINT_BITS..(r + 1) * FINGERPRINT_BITS].iter().sum() };
    assert_eq!(row_sum(0), 3.0);
    assert_eq!(row_sum(1), 3.0);
    assert_eq!(row_sum(2), 3.0);
    assert_eq!(got[FINGERPRINT_BITS], 1.0);
    assert_eq!(got[FINGERPRINT_BITS + 1], 0.0);
    assert_eq!(got[FINGERPRINT_BITS + 4], 1.0);
    assert_eq!(got[FINGERPRINT_BITS + 11], 0.0);
    assert_eq!(got[FINGERPRINT_BITS + 1023], 1.0);
    assert_eq!(got[11], 1.0);
    assert_eq!(got[4], 0.0);
    assert!(labels.dense(&[]).is_empty());
}

#[test]
fn metrics_hand_examples() {
    // Bit accuracy: one all-correct row and one all-wrong row.
    let mut logits = vec![-10.0f32; 2 * FINGERPRINT_BITS];
    for v in logits.iter_mut().skip(FINGERPRINT_BITS) {
        *v = 10.0;
    }
    let targets = vec![0.0f32; 2 * FINGERPRINT_BITS];
    assert!((bit_accuracy(&logits, &targets, 2) - 0.5).abs() < 1e-12);

    // Tanimoto: both empty is defined as 1 ...
    let logits_empty = vec![-10.0f32; FINGERPRINT_BITS];
    let targets_empty = vec![0.0f32; FINGERPRINT_BITS];
    assert!((tanimoto(&logits_empty, &targets_empty, 1) - 1.0).abs() < 1e-12);
    // ... a nonempty prediction against an empty label is 0 ...
    let logits_full = vec![10.0f32; FINGERPRINT_BITS];
    assert!((tanimoto(&logits_full, &targets_empty, 1) - 0.0).abs() < 1e-12);
    // ... and {0, 2} predicted against {0, 1} labeled is 1/3.
    let mut logits_part = vec![-10.0f32; FINGERPRINT_BITS];
    logits_part[0] = 10.0;
    logits_part[2] = 10.0;
    let mut targets_part = vec![0.0f32; FINGERPRINT_BITS];
    targets_part[0] = 1.0;
    targets_part[1] = 1.0;
    assert!((tanimoto(&logits_part, &targets_part, 1) - 1.0 / 3.0).abs() < 1e-12);

    // Cosine: both all-zero is 1 ...
    let neg_inf = vec![f32::NEG_INFINITY; FINGERPRINT_BITS];
    assert_eq!(cosine_per_spectrum(&neg_inf, &targets_empty, 1), vec![1.0]);
    // ... orthogonal one-hots are 0 ...
    let mut logits_e1 = vec![f32::NEG_INFINITY; FINGERPRINT_BITS];
    logits_e1[1] = f32::INFINITY;
    let mut targets_e0 = vec![0.0f32; FINGERPRINT_BITS];
    targets_e0[0] = 1.0;
    assert_eq!(cosine_per_spectrum(&logits_e1, &targets_e0, 1), vec![0.0]);
    // ... an aligned one-hot is 1 ...
    let mut targets_e1 = vec![0.0f32; FINGERPRINT_BITS];
    targets_e1[1] = 1.0;
    assert_eq!(cosine_per_spectrum(&logits_e1, &targets_e1, 1), vec![1.0]);
    // ... and (0.5, 0, ...) against (1, 0, ...) is exactly 1.
    let mut logits_half = vec![f32::NEG_INFINITY; FINGERPRINT_BITS];
    logits_half[0] = 0.0;
    assert_eq!(cosine_per_spectrum(&logits_half, &targets_e0, 1), vec![1.0]);
}

fn zero_head(d: usize, device: &Device<R>) -> FingerprintHead<R, E> {
    let mut rng = Rng::seeded(99);
    let head = FingerprintHead::<R, E>::init(d, device, &mut rng).unwrap();
    for (_, param) in head.named_parameters() {
        let shape = param.shape().dims().to_vec();
        param.set(Tensor::<R, f32>::from_f32(&vec![0.0; shape.iter().product()], shape, device).unwrap());
    }
    head
}

#[test]
fn zero_logit_gradients_are_half_over_bits() {
    let device = dev();
    // Zero every parameter so the bit logits are exactly 0 (an untrained
    // head with zero bias sits exactly here). The shared BCE derivative is
    // sigmoid(0) − y = ±0.5 per bit, divided by the documented normalisation
    // (1024 bits, unit weight sum): every fc2 bias gradient must be
    // +0.5/1024 for all-zero targets and −0.5/1024 for all-one targets. The
    // old maximum/abs composition gave 0 instead of +0.5/1024, so negative
    // bits got no initial learning signal.
    for (target, want) in [(0.0f32, 0.5f32 / FINGERPRINT_BITS as f32), (1.0f32, -0.5f32 / FINGERPRINT_BITS as f32)] {
        let head = zero_head(4, &device);
        let pool_t = Tensor::<R, f32>::zeros(vec![1, 4], &device);
        let targets_t =
            Tensor::<R, f32>::from_f32(&vec![target; FINGERPRINT_BITS], vec![1, FINGERPRINT_BITS], &device)
                .unwrap();
        let weights_t = Tensor::<R, f32>::ones(vec![1], &device);
        let logits = head.logits(&Var::constant(pool_t)).unwrap();
        assert!(logits.try_to_f32().unwrap().iter().all(|&v| v == 0.0));
        let loss = head.loss(&logits, &targets_t, &weights_t).unwrap();
        let val = loss.try_to_f32().unwrap()[0];
        assert!(
            (val - std::f32::consts::LN_2).abs() < 1e-5,
            "target {target}: loss {val}, want ln 2"
        );
        let grads = loss.backward().unwrap();
        let (_, bias) = head
            .named_parameters()
            .into_iter()
            .find(|(n, _)| n.ends_with("fc2.bias"))
            .expect("fc2.bias");
        let g = grads.get(bias.id()).expect("gradient for fc2.bias").to_f32();
        assert_eq!(g.len(), FINGERPRINT_BITS);
        for (i, &v) in g.iter().enumerate() {
            assert!(
                (v - want).abs() < 1e-6,
                "target {target}: bit {i} gradient {v}, want {want}"
            );
        }
    }
}

#[test]
fn fractional_weight_sums_below_one_keep_full_weight() {
    // The rule under test (stated next to the number): per-spectrum mean over
    // the 1,024 bits, then the weighted mean over spectra
    // `sum_b w_b * l_b / max(sum_b w_b, eps)` with `eps` tiny. One spectrum,
    // weight 0.5, zero logits, all-zero targets gives ln 2 ≈ 0.693147 (the
    // old `max(weight_sum, 1)` rule and its test reference gave half that,
    // 0.346574).
    let device = dev();
    let head = zero_head(4, &device);
    let pool_t = Tensor::<R, f32>::zeros(vec![1, 4], &device);
    let targets_t = Tensor::<R, f32>::zeros(vec![1, FINGERPRINT_BITS], &device);
    let weights_t = Tensor::<R, f32>::from_f32(&[0.5], vec![1], &device).unwrap();
    let logits = head.logits(&Var::constant(pool_t)).unwrap();
    let loss = head.loss(&logits, &targets_t, &weights_t).unwrap();
    let got = loss.try_to_f32().unwrap()[0];
    assert!(
        (got - std::f32::consts::LN_2).abs() < 1e-5,
        "loss {got} under the weighted-mean rule, want ln 2 ({})",
        std::f32::consts::LN_2
    );
}

fn export_json_with_smiles(keys: &[&str], smiles: &[&str]) -> String {
    let molecules: Vec<serde_json::Value> = keys
        .iter()
        .zip(smiles.iter())
        .enumerate()
        .map(|(i, (k, s))| {
            serde_json::json!({
                "key": k, "smiles": s, "identity_group": i as u64, "fold_identity": 2u32,
                "atoms": [2u8], "bonds": [], "spectra": [spectrum_json(i as u64)]
            })
        })
        .collect();
    serde_json::json!({
        "schema_version": 1u32, "chemistry": "ms2-chem-v0.1",
        "rdkit": "test", "source": "synthetic", "seed": 1u64, "n_raw": 64u32,
        "spectra_per_molecule": 1u32, "skipped_spectra": {},
        "subset": "train", "molecules": molecules
    })
    .to_string()
}

fn sidecar_json_with_smiles_and_hash(keys: &[&str], smiles: &[&str], hash: &str) -> String {
    let molecules: Vec<serde_json::Value> = keys
        .iter()
        .zip(smiles.iter())
        .enumerate()
        .map(|(i, (k, s))| {
            serde_json::json!({
                "key": k, "smiles": s,
                "bits": [0u32, (i as u32 * 7 + 3) % 1023 + 1, 1023u32]
            })
        })
        .collect();
    serde_json::json!({
        "schema_version": 1u32, "fingerprint": FINGERPRINT_VERSION,
        "bits": FINGERPRINT_BITS, "radius": 2u32,
        "generator": "rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)",
        "rdkit": "test", "source_export": "synth.json", "source_sha256": hash,
        "molecules": molecules
    })
    .to_string()
}

#[test]
fn matches_export_checks_parent_smiles_and_source_hash() {
    use mamba3::models::ms2::fingerprint::sha256_hex;
    let keys = ["MOL0001", "MOL0002"];
    let export_text = export_json_with_smiles(&keys, &["CCO", "CCC"]);
    let export = ExportFile::from_json(&export_text).unwrap();
    let hash = sha256_hex(export_text.as_bytes());
    // The right sidecar passes with the export bytes.
    let good = FingerprintLabels::from_json(&sidecar_json_with_smiles_and_hash(
        &keys,
        &["CCO", "CCC"],
        &hash,
    ))
    .unwrap();
    good.matches_export(&export, Some(export_text.as_bytes())).unwrap();
    // Key-only matching still works without the bytes.
    good.matches_export(&export, None).unwrap();
    // Same keys but a wrong parent (CC(=O)O for CCO) fails.
    let wrong_parent = FingerprintLabels::from_json(&sidecar_json_with_smiles_and_hash(
        &keys,
        &["CC(=O)O", "CCC"],
        &hash,
    ))
    .unwrap();
    assert!(
        wrong_parent
            .matches_export(&export, Some(export_text.as_bytes()))
            .is_err(),
        "same-key/wrong-parent sidecar must not match"
    );
    // A stale source hash fails even with right keys and parents.
    let stale = FingerprintLabels::from_json(&sidecar_json_with_smiles_and_hash(
        &keys,
        &["CCO", "CCC"],
        "00",
    ))
    .unwrap();
    assert!(
        stale
            .matches_export(&export, Some(export_text.as_bytes()))
            .is_err(),
        "stale-hash sidecar must not match"
    );
}
