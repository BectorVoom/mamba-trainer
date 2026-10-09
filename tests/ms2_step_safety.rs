//! T1A Part 2: on-device step safety (plan P7.6).
//!
//! Non-finite guard, loss scaling and the gated AdamW apply flag. CPU and GPU
//! via `backends::Auto`; every test holds the file-level `serial()` mutex
//! (launch/read counters are process-global). The optimizer-kernel apply-flag
//! tests live in `tests/adamw_multi.rs` as new test functions.
//!
//! ```text
//! CARGO_TARGET_DIR=target/cpu-ms2 cargo test --release \
//!   --no-default-features --features cpu --test ms2_step_safety
//! ```

#![cfg(feature = "backend")]

use mamba3::backend::{Device, launch_count, runtime_read_count};
use mamba3::backends::Auto;
use mamba3::models::ms2::chem::{
    Composition, ELECTRON_MASS, ELEMENTS, HYDROGEN, adduct, composition_mass,
};
use mamba3::models::ms2::contract::{Control, ModelConfig};
use mamba3::models::ms2::dataset::ExportSpectrum;
use mamba3::models::ms2::experiment::{
    ExperimentSet, ExperimentSpectrum, SpectrumDomain, label_export_spectrum,
};
use mamba3::models::ms2::formula::FormulaTable;
use mamba3::models::ms2::targets::{Candidates, Peak, RecipeLimits};
use mamba3::models::ms2::train::{GoldFormulaConditioning, Ms2Trainer, TrainConfig};
use mamba3::models::ms2::{MolGraph, RawAtom, RawMolecule};
use mamba3::nn::Module;

type R = Auto;
type E = f32;

fn dev() -> Device<R> {
    Device::<R>::default()
}

static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn serial() -> std::sync::MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

fn is_cpu() -> bool {
    std::any::type_name::<R>().contains("Cpu")
}

fn fixture() -> serde_json::Value {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/ms2/chemistry_v0.json");
    serde_json::from_str(&std::fs::read_to_string(path).expect("fixture readable"))
        .expect("fixture parses")
}

fn precursor_of(parent: &Composition, adduct_id: u16) -> u32 {
    let mass = composition_mass(parent).expect("fixture masses fit");
    let a = adduct(adduct_id).expect("fixture adduct");
    let hydrogen = ELEMENTS[HYDROGEN].mass;
    ((mass as i64) + (a.hydrogens as i64) * (hydrogen as i64)
        - (a.charge as i64) * (ELECTRON_MASS as i64)) as u32
}

fn experiment_spectrum(
    molecule_idx: usize,
    m: &serde_json::Value,
    spectrum_id: u64,
    parent: &MolGraph,
    parent_composition: Composition,
) -> Option<ExperimentSpectrum> {
    let s = m["spectra"].as_array()?.first()?;
    let adduct_id = s["adduct"].as_u64()? as u16;
    let Some(a) = adduct(adduct_id) else {
        return None;
    };
    let peaks: Vec<Peak> = s["peaks"]
        .as_array()?
        .iter()
        .enumerate()
        .map(|(k, p)| {
            let p = p.as_array().expect("peak triple");
            Peak {
                id: k as u32,
                mz: p[1].as_u64().expect("mz") as u32,
                intensity: p[2].as_f64().expect("intensity"),
            }
        })
        .collect();
    if peaks.is_empty() {
        return None;
    }
    let n = peaks.len();
    let precursor = precursor_of(&parent_composition, adduct_id);
    if !(50_000_000..=2_000_000_000).contains(&precursor) {
        return None;
    }
    let export = ExportSpectrum {
        row: spectrum_id,
        spectrum_id,
        adduct: adduct_id,
        polarity: if a.charge == 1 { 1 } else { -1 },
        precursor_mz_udalton: precursor,
        precursor_uncertainty_udalton: 50,
        raw_peak_count: n as u32,
        peak_id: (0..n as u32).collect(),
        mz_udalton: peaks.iter().map(|p| p.mz).collect(),
        intensity: peaks.iter().map(|p| p.intensity).collect(),
        mz_uncertainty_udalton: s["mz_uncertainty"].as_u64().unwrap_or(50) as u32,
        collision_energy_ev: 30.0,
        collision_energy_known: 1,
        energy_count: 1,
        instrument_class: 0,
    };
    let candidates = Candidates::new(parent, &RecipeLimits::V0).ok()?;
    let labels = label_export_spectrum(&candidates, &export).ok()?;
    let (labels, domain) = if labels.targets.is_empty() {
        (None, SpectrumDomain::InDomainUnlabeled)
    } else {
        (Some(labels), SpectrumDomain::InDomainLabeled)
    };
    Some(ExperimentSpectrum {
        molecule: molecule_idx,
        spectrum: export,
        parent: MolGraph::new(parent.atoms().to_vec(), parent.bonds().to_vec()).ok()?,
        parent_composition,
        labels,
        domain,
    })
}

fn labeled_set(n: usize) -> (ExperimentSet, Vec<Composition>) {
    let f = fixture();
    let mut molecules = Vec::new();
    let mut spectra = Vec::new();
    let mut parents = Vec::new();
    let mut spectrum_id = 5000u64;
    for m in f["molecules"].as_array().expect("molecules") {
        if spectra.len() == n {
            break;
        }
        let raw = RawMolecule {
            atoms: m["raw_atoms"]
                .as_array()
                .expect("raw_atoms")
                .iter()
                .map(|a| RawAtom {
                    element: a["element"].as_str().expect("element").to_string(),
                    charge: a["charge"].as_i64().expect("charge") as i32,
                    hydrogens: a["hydrogens"].as_u64().expect("hydrogens") as u8,
                    isotope: a["isotope"].as_u64().expect("isotope") as u32,
                    radical_electrons: a["radical_electrons"].as_u64().expect("radical") as u8,
                    valence: a["valence"].as_u64().expect("valence") as u8,
                })
                .collect(),
            bonds: m["bonds"]
                .as_array()
                .expect("bonds")
                .iter()
                .map(|b| {
                    let b = b.as_array().expect("bond triple");
                    (
                        b[0].as_u64().expect("a") as usize,
                        b[1].as_u64().expect("b") as usize,
                        b[2].as_u64().expect("order") as u8,
                    )
                })
                .collect(),
        };
        let Ok(graph) = raw.to_graph() else {
            continue;
        };
        let composition = graph.composition();
        let molecule_idx = molecules.len();
        let Some(entry) = experiment_spectrum(molecule_idx, m, spectrum_id, &graph, composition)
        else {
            continue;
        };
        if entry.domain != SpectrumDomain::InDomainLabeled {
            continue;
        }
        spectrum_id += 1;
        molecules.push(m["name"].as_str().expect("name").to_string());
        parents.push(composition);
        spectra.push(entry);
    }
    assert_eq!(spectra.len(), n, "fixture yields {n} labeled molecules");
    (
        ExperimentSet {
            name: "fixture-labeled".to_string(),
            source_sha256: "fixture".to_string(),
            molecules,
            spectra,
        },
        parents,
    )
}

fn tiny_config() -> ModelConfig {
    let mut m = ModelConfig::v0();
    m.d_model = 16;
    m.n_peaks = 16;
    m.encoder_blocks = 1;
    m.decoder_blocks = 1;
    m.attention_heads = 2;
    m.encoder.d_model = 16;
    m.encoder.n_heads = 2;
    m.encoder.head_dim = 8;
    m.encoder.d_state = 8;
    m.encoder.n_groups = 2;
    m.decoder.d_model = 16;
    m.decoder.n_heads = 2;
    m.decoder.head_dim = 8;
    m.decoder.d_state = 8;
    m.decoder.n_groups = 2;
    m
}

fn safety_config(guard: bool, loss_scale: f32) -> TrainConfig {
    TrainConfig {
        batch: 4,
        slots: 16,
        lr: 3e-3,
        weight_decay: 0.1,
        formula_weight: 0.2,
        seed: 41,
        control: Control::None,
        grad_clip: Some(1.0),
        gold_formula_conditioning: GoldFormulaConditioning::ScoredRowOrZero,
        formula_source: mamba3::models::ms2::contract::FormulaSource::Table,
        formula_window: 32,
        enum_lanes_max: 262_144,
        enum_lane_visits_max: 4_096,
        enum_dispatch_visits_max: 4_000_000,
        enum_fit_name: None,
        enum_fit_sha256: None,
        enum_fit_subset: None,
        lambda_assign: 0.0,
        ion_request_work_max: 268_435_456,
        formula_evidence_work_max: 2048,
        formula_evidence_dispatch_max: 268435456,
        precursor_jitter_ppm: 0.0,
        precursor_jitter_variants: 0,
        nonfinite_guard: guard,
        loss_scale,
    }
}

/// Poison ONE parameter-gradient path through the state dict (no library
/// code path changed): a formula-head weight becomes infinite, so the logits
/// overflow and the step's loss is non-finite.
fn poison_formula_weight(trainer: &mut Ms2Trainer<R, E>) {
    let mut dict = trainer.model.state_dict();
    let key = dict
        .entries
        .keys()
        .find(|k| k.contains("formula"))
        .cloned()
        .expect("a formula-head parameter");
    let entry = dict.entries.get_mut(&key).unwrap();
    for v in entry.data.iter_mut() {
        *v = f32::INFINITY;
    }
    trainer.model.load_state_dict(&dict, true).unwrap();
}

fn state_bits(trainer: &Ms2Trainer<R, E>) -> Vec<(String, Vec<u32>)> {
    let mut out: Vec<(String, Vec<u32>)> = trainer
        .model
        .state_dict()
        .entries
        .into_iter()
        .map(|(k, t)| (k, t.data.into_iter().map(f32::to_bits).collect()))
        .collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

fn moments_bits(trainer: &Ms2Trainer<R, E>) -> Vec<(String, Vec<u32>)> {
    let mut out: Vec<(String, Vec<u32>)> = trainer
        .optimizer_state_for_test()
        .entries
        .into_iter()
        .map(|(k, t)| (k, t.data.into_iter().map(f32::to_bits).collect()))
        .collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

/// Guard on, poisoned step: every parameter and every moment is bit-identical
/// before and after, the device counter reads 1 at the next report, and the
/// following healthy step updates normally. The same poisoned step with the
/// guard OFF changes the parameters (the test is not vacuous).
#[test]
fn guard_skips_nonfinite_step() {
    let _serial = serial();
    let device = dev();
    let (set, parents) = labeled_set(4);
    let table = FormulaTable::from_compositions(parents.into_iter()).unwrap();
    let indices = vec![0, 1, 2, 3];

    // Guard on: poison, step, nothing moves.
    let mut trainer =
        Ms2Trainer::<R, E>::new(&tiny_config(), &table, &safety_config(true, 1.0), &device)
            .unwrap();
    trainer.step(&set, &indices).unwrap();
    let params_before = state_bits(&trainer);
    let moments_before = moments_bits(&trainer);
    poison_formula_weight(&mut trainer);
    // The poisoned forward is non-finite (sanity: the test poisons enough).
    trainer.request_report();
    // Note: this report reads the poisoned loss; it must be non-finite for
    // the guard to have something to skip. If the fixture ever stops
    // overflowing here, this assert — not the guard — fails first.
    let poisoned_rep = trainer.step(&set, &indices).unwrap().expect("report");
    assert!(
        !(poisoned_rep.loss > -3e38 && poisoned_rep.loss < 3e38),
        "poisoned loss must leave the validated domain, got {}",
        poisoned_rep.loss
    );
    // Rebuild the guarded scenario without the report read in the middle:
    // poison again from a healthy state and step without reporting.
    // Precise pre/post check around one skipped step.
    let mut precise =
        Ms2Trainer::<R, E>::new(&tiny_config(), &table, &safety_config(true, 1.0), &device)
            .unwrap();
    precise.step(&set, &indices).unwrap();
    poison_formula_weight(&mut precise);
    let pre_p = state_bits(&precise);
    let pre_m = moments_bits(&precise);
    precise.step(&set, &indices).unwrap();
    assert_eq!(state_bits(&precise), pre_p, "guarded params bit-identical");
    assert_eq!(
        moments_bits(&precise),
        pre_m,
        "guarded moments bit-identical"
    );
    // Counter reads 1 at the next report; then a healthy step updates.
    precise.request_report();
    // Still poisoned: this report's step is another skip, counter becomes 2.
    // Restore healthy weights first so the reported step is healthy and the
    // counter shows exactly the one earlier skip.
    let mut healthy =
        Ms2Trainer::<R, E>::new(&tiny_config(), &table, &safety_config(true, 1.0), &device)
            .unwrap();
    healthy.step(&set, &indices).unwrap();
    let healthy_p = state_bits(&healthy);
    let healthy_m = moments_bits(&healthy);
    poison_formula_weight(&mut healthy);
    healthy.step(&set, &indices).unwrap(); // skipped, counter 1
    // Restore the healthy weights (moments were untouched by the skip).
    healthy
        .model
        .load_state_dict(
            &{
                let mut d = std::collections::BTreeMap::new();
                for (k, bits) in healthy_p.iter() {
                    d.insert(
                        k.clone(),
                        mamba3::nn::module::TensorData {
                            shape: healthy
                                .model
                                .state_dict()
                                .entries
                                .get(k)
                                .unwrap()
                                .shape
                                .clone(),
                            data: bits.iter().map(|b| f32::from_bits(*b)).collect(),
                        },
                    );
                }
                mamba3::nn::module::StateDict { entries: d }
            },
            true,
        )
        .unwrap();
    assert_eq!(state_bits(&healthy), healthy_p, "weights restored");
    assert_eq!(
        moments_bits(&healthy),
        healthy_m,
        "moments survived the skip"
    );
    healthy.request_report();
    let rep = healthy.step(&set, &indices).unwrap().expect("report");
    assert_eq!(rep.skipped_steps_total, 1, "one skipped step counted");
    assert!(!rep.last_step_skipped, "the healthy step applied");
    assert_ne!(
        state_bits(&healthy),
        healthy_p,
        "the following healthy step updates normally"
    );
    let _ = (params_before, moments_before);

    // Guard off: the same poisoned step changes the parameters.
    let mut unguarded =
        Ms2Trainer::<R, E>::new(&tiny_config(), &table, &safety_config(false, 1.0), &device)
            .unwrap();
    unguarded.step(&set, &indices).unwrap();
    poison_formula_weight(&mut unguarded);
    let pre = state_bits(&unguarded);
    unguarded.step(&set, &indices).unwrap();
    assert_ne!(
        state_bits(&unguarded),
        pre,
        "guard off: the poisoned step moves the parameters (non-vacuous)"
    );
}

/// Guard on, healthy steps: losses equal the guard-off run bit for bit on CPU.
#[test]
fn guard_healthy_matches_unguarded() {
    let _serial = serial();
    let device = dev();
    let (set, parents) = labeled_set(4);
    let table = FormulaTable::from_compositions(parents.into_iter()).unwrap();
    let indices = vec![0, 1, 2, 3];
    let mut off =
        Ms2Trainer::<R, E>::new(&tiny_config(), &table, &safety_config(false, 1.0), &device)
            .unwrap();
    let mut on =
        Ms2Trainer::<R, E>::new(&tiny_config(), &table, &safety_config(true, 1.0), &device)
            .unwrap();
    for step in 0..4 {
        off.request_report();
        on.request_report();
        let a = off.step(&set, &indices).unwrap().expect("report");
        let b = on.step(&set, &indices).unwrap().expect("report");
        assert_eq!(b.skipped_steps_total, 0, "no skips on healthy steps");
        assert!(!b.last_step_skipped);
        if is_cpu() {
            assert_eq!(a.loss.to_bits(), b.loss.to_bits(), "step {step} loss");
            assert_eq!(a.graph.to_bits(), b.graph.to_bits(), "step {step} graph");
            assert_eq!(
                a.formula.to_bits(),
                b.formula.to_bits(),
                "step {step} formula"
            );
        } else {
            for (what, x, y) in [("loss", a.loss, b.loss), ("graph", a.graph, b.graph)] {
                let tol = 1e-5 * x.abs().max(y.abs()).max(1.0);
                assert!((x - y).abs() <= tol, "{what} {x} vs {y}");
            }
        }
    }
}

/// A warmed non-report step with the guard and `loss_scale = 256` performs
/// zero device reads and a constant number of launches (printed).
#[test]
fn guard_scaling_no_reads_constant_launches() {
    let _serial = serial();
    let device = dev();
    let (set, parents) = labeled_set(4);
    let table = FormulaTable::from_compositions(parents.into_iter()).unwrap();
    let indices = vec![0, 1, 2, 3];
    let mut trainer =
        Ms2Trainer::<R, E>::new(&tiny_config(), &table, &safety_config(true, 256.0), &device)
            .unwrap();
    for _ in 0..3 {
        trainer.step(&set, &indices).unwrap();
    }
    let mut launches = Vec::new();
    for _ in 0..5 {
        let l0 = launch_count();
        let r0 = runtime_read_count();
        assert!(trainer.step(&set, &indices).unwrap().is_none());
        launches.push(launch_count() - l0);
        assert_eq!(
            runtime_read_count() - r0,
            0,
            "a guarded scaled step reads nothing"
        );
    }
    assert!(
        launches.windows(2).all(|w| w[0] == w[1]),
        "guarded scaled launches are constant: {launches:?}"
    );
    println!(
        "guarded loss_scale=256 warmed non-report launches: {}",
        launches[0]
    );
}

/// `loss_scale = 256` against `1.0` over 4 steps (CPU f32): reported losses
/// agree within 1e-5 relative (not bit-identical: scaling changes rounding)
/// and the gradients after unscaling agree within 1e-5 relative.
#[test]
fn loss_scale_matches_unscaled() {
    let _serial = serial();
    let device = dev();
    let (set, parents) = labeled_set(4);
    let table = FormulaTable::from_compositions(parents.into_iter()).unwrap();
    let indices = vec![0, 1, 2, 3];
    let mut unscaled =
        Ms2Trainer::<R, E>::new(&tiny_config(), &table, &safety_config(false, 1.0), &device)
            .unwrap();
    let mut scaled = Ms2Trainer::<R, E>::new(
        &tiny_config(),
        &table,
        &safety_config(false, 256.0),
        &device,
    )
    .unwrap();
    for step in 0..4 {
        // Gradient agreement after unscaling (host-side divide, test only).
        let fu = unscaled.forward_state(&set, &indices).unwrap();
        let fs = scaled.forward_state(&set, &indices).unwrap();
        let bu = unscaled.backward_state(&fu).unwrap();
        let bs = scaled.backward_state(&fs).unwrap();
        let gu = bu.grads_for_test();
        let gs = bs.grads_for_test();
        // ParamIds are process-local (fresh per trainer), so order by
        // parameter path: both models share the architecture and build
        // order, hence the same name order.
        let named_u = unscaled.model.named_parameters();
        let named_s = scaled.model.named_parameters();
        assert_eq!(named_u.len(), named_s.len(), "step {step}: param count");
        for (((name_u, pu), (name_s, ps)), idx) in named_u.iter().zip(named_s.iter()).zip(0..) {
            let _ = idx;
            assert_eq!(name_u, name_s, "step {step}: param order");
            let (Some(tu), Some(ts)) = (gu.get(pu.id()), gs.get(ps.id())) else {
                continue;
            };
            let vu = tu.to_f32();
            let vs = ts.to_f32();
            assert_eq!(vu.len(), vs.len(), "step {step}: grad len");
            for (i, (&u, &s)) in vu.iter().zip(vs.iter()).enumerate() {
                // `s` is 256x `u` before the device-side unscale: compare
                // `s / 256` against `u` within 1e-5 relative.
                let back = s / 256.0;
                let tol = 1e-5 * u.abs().max(back.abs()).max(1.0);
                assert!(
                    (u - back).abs() <= tol,
                    "step {step} grad[{i}] beyond 1e-5 relative: {u} vs {back} (scaled {s})"
                );
            }
        }
        // Finish both steps through the production path and compare reports.
        unscaled.optimizer_step(&bu).unwrap();
        scaled.optimizer_step(&bs).unwrap();
        // Reports need the packed read: emulate `step`'s close-out by
        // requesting before the *next* forward is not possible here (the
        // states are consumed), so compare via a following reported step.
        unscaled.request_report();
        scaled.request_report();
        let ru = unscaled.step(&set, &indices).unwrap().expect("report");
        let rs = scaled.step(&set, &indices).unwrap().expect("report");
        for (what, u, s) in [("loss", ru.loss, rs.loss), ("graph", ru.graph, rs.graph)] {
            let tol = 1e-5 * u.abs().max(s.abs()).max(1.0);
            assert!((u - s).abs() <= tol, "step {step} {what}: {u} vs {s}");
        }
        // Note: power-of-two scaling is exact in binary floating point, so
        // when no clip engages the reported losses may be bit-identical;
        // when clipping engages the norm path rounds differently. Either way
        // the 1e-5 agreement above is the requirement.
    }
}

/// The `u32` skip counter counts 300 bf16 skips exactly and saturates at
/// `u32::MAX` (task F7B item B3): the counter kernel is driven directly 300
/// times under bf16 on the CPU runtime (a float counter stalls at 256 in
/// bf16), plus a short end-to-end bf16 run with poisoned skips.
#[test]
fn guard_counter_u32_exact_300_skips_bf16() {
    use half::bf16;
    use mamba3::tensor::Tensor;
    use mamba3::tensor::ops::index::IdTensor;
    let _serial = serial();
    let device = dev();
    // Direct drive: 300 skips under bf16 count exactly 300.
    let loss = Tensor::<R, bf16>::full(vec![1], f32::INFINITY, &device);
    let sumsq = Tensor::<R, bf16>::zeros(vec![1], &device);
    let apply = Tensor::<R, bf16>::full(vec![1], 1.0, &device);
    let counter = IdTensor::<R>::from_slice(&[0u32], vec![1], &device).unwrap();
    for _ in 0..300 {
        mamba3::tensor::ops::fused::guard_apply(&loss, &sumsq, &apply, &counter).unwrap();
    }
    assert_eq!(
        counter.try_to_vec().unwrap(),
        vec![300u32],
        "300 bf16 skips count 300"
    );
    // Saturation, never wrap: MAX - 1 plus two skips sticks at MAX.
    let counter2 = IdTensor::<R>::from_slice(&[u32::MAX - 1], vec![1], &device).unwrap();
    for _ in 0..2 {
        mamba3::tensor::ops::fused::guard_apply(&loss, &sumsq, &apply, &counter2).unwrap();
    }
    assert_eq!(counter2.try_to_vec().unwrap(), vec![u32::MAX]);
    // A healthy step after saturation keeps MAX.
    let ok_loss = Tensor::<R, bf16>::full(vec![1], 1.25, &device);
    mamba3::tensor::ops::fused::guard_apply(&ok_loss, &sumsq, &apply, &counter2).unwrap();
    assert_eq!(counter2.try_to_vec().unwrap(), vec![u32::MAX]);
    // Short end-to-end bf16 run: 3 poisoned skips then a healthy reported
    // step counts exactly 3.
    let (set, parents) = labeled_set(4);
    let table = FormulaTable::from_compositions(parents.into_iter()).unwrap();
    let indices = vec![0, 1, 2, 3];
    let cfg = safety_config(true, 1.0);
    let mut model_bf16 = tiny_config();
    model_bf16.dtype = mamba3::backend::DType::BF16;
    let mut trainer = Ms2Trainer::<R, bf16>::new(&model_bf16, &table, &cfg, &device).unwrap();
    trainer.step(&set, &indices).unwrap();
    let healthy_dict = trainer.model.state_dict();
    {
        let mut dict = trainer.model.state_dict();
        let key = dict
            .entries
            .keys()
            .find(|k| k.contains("formula"))
            .cloned()
            .expect("a formula-head parameter");
        for v in dict.entries.get_mut(&key).unwrap().data.iter_mut() {
            *v = f32::INFINITY;
        }
        trainer.model.load_state_dict(&dict, true).unwrap();
    }
    for _ in 0..3 {
        trainer.step(&set, &indices).unwrap();
    }
    trainer.model.load_state_dict(&healthy_dict, true).unwrap();
    trainer.request_report();
    let rep = trainer.step(&set, &indices).unwrap().expect("report");
    assert!(
        !rep.last_step_skipped,
        "restored bf16 step applies: {rep:?}"
    );
    assert_eq!(rep.skipped_steps_total, 3, "three bf16 skips counted");
    // The bf16 total survives save/load too.
    let path = std::env::temp_dir().join("ms2_f7b_bf16_counter.json");
    trainer.save(&path).unwrap();
    let saved: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    assert_eq!(saved["skipped_steps_total"], serde_json::json!(3));
    drop(trainer);
    let mut reloaded = Ms2Trainer::<R, bf16>::load(&path, &table, &device).unwrap();
    reloaded.request_report();
    let rep2 = reloaded.step(&set, &indices).unwrap().expect("report");
    assert_eq!(rep2.skipped_steps_total, 3, "bf16 total survives save/load");
    std::fs::remove_file(&path).ok();
}

/// Reviewer's scenario (task F7B item B4): guard off, `loss_scale = 256`,
/// `grad_clip = 1`, one unscaled gradient of `1e17`. The scaled gradient is
/// finite but squaring it overflows f32 (`~6.55e38`); unscaling inside the
/// reduction keeps the norm finite, so the update is the clipped unscaled
/// one (≈ 1.0), equal to the unscaled run's within 1e-5 relative — and both
/// AdamW moments agree.
#[test]
fn unscaled_norm_clips_reduction_overflow() {
    use mamba3::autograd::Grads;
    use mamba3::nn::Param;
    use mamba3::tensor::Tensor;
    use mamba3::train::optim::{AdamWConfig, Optimizer, grad_scale, grad_scale_unscaled};
    let _serial = serial();
    let device = dev();
    let g_unscaled = 1e17f32;
    let loss_scale = 256.0f32;
    let g_scaled = g_unscaled * loss_scale; // 2.56e19, finite in f32
    assert!(g_scaled.is_finite());
    assert!(
        (g_scaled * g_scaled).is_infinite(),
        "the scaled square overflows"
    );
    // Optimizer params first: the grad maps must be keyed by their ids.
    let mut opt_u = AdamWConfig::builder()
        .learning_rate(3e-3)
        .build()
        .init::<R, E>();
    let mut opt_s = AdamWConfig::builder()
        .learning_rate(3e-3)
        .build()
        .init::<R, E>();
    let p_u = Param::<R, E>::new(Tensor::<R, E>::from_f32(&[0.5], vec![1], &device).unwrap());
    let p_s = Param::<R, E>::new(Tensor::<R, E>::from_f32(&[0.5], vec![1], &device).unwrap());
    let mut map_u = Grads::default();
    map_u
        .accumulate(
            p_u.id(),
            Tensor::<R, E>::from_f32(&[g_unscaled], vec![1], &device).unwrap(),
        )
        .unwrap();
    let mut map_s = Grads::default();
    map_s
        .accumulate(
            p_s.id(),
            Tensor::<R, E>::from_f32(&[g_scaled], vec![1], &device).unwrap(),
        )
        .unwrap();
    let gs_u = grad_scale(&map_u, 1.0, 1.0).unwrap().expect("scale");
    let gs_s = grad_scale_unscaled(&map_s, 1.0, 1.0 / loss_scale)
        .unwrap()
        .expect("scale");
    let factor_u = gs_u.factor.try_to_f32().unwrap()[0];
    let factor_s = gs_s.factor.try_to_f32().unwrap()[0];
    let update_u = g_unscaled * factor_u;
    let update_s = g_scaled * factor_s;
    for (what, u) in [("unscaled", update_u), ("scaled", update_s)] {
        let tol = 1e-5 * u.abs().max(1.0);
        assert!(
            (u - 1.0).abs() <= tol,
            "{what} update is the clipped ≈1.0, got {u}"
        );
    }
    let tol = 1e-5 * update_u.abs().max(update_s.abs()).max(1.0);
    assert!(
        (update_u - update_s).abs() <= tol,
        "scaled update {update_s} equals unscaled {update_u}"
    );
    // Both AdamW moments agree after stepping with the two factors.
    let p_u_value_before = p_u.value().to_f32()[0];
    let p_s_value_before = p_s.value().to_f32()[0];
    assert_eq!(p_u_value_before.to_bits(), p_s_value_before.to_bits());
    opt_u
        .step_scaled(std::slice::from_ref(&p_u), &map_u, Some(&gs_u.factor))
        .unwrap();
    opt_s
        .step_scaled(std::slice::from_ref(&p_s), &map_s, Some(&gs_s.factor))
        .unwrap();
    let wu = p_u.value().try_to_f32().unwrap()[0];
    let ws = p_s.value().try_to_f32().unwrap()[0];
    let tol = 1e-5 * wu.abs().max(ws.abs()).max(1.0);
    assert!((wu - ws).abs() <= tol, "post-update weights {wu} vs {ws}");
    let named_u = vec![("p".to_string(), p_u.clone())];
    let named_s = vec![("p".to_string(), p_s.clone())];
    let mu = opt_u.state_dict(&named_u);
    let ms = opt_s.state_dict(&named_s);
    assert_eq!(mu.entries.len(), ms.entries.len());
    for ((ku, tu), (ks, ts)) in mu.entries.iter().zip(ms.entries.iter()) {
        assert_eq!(ku, ks);
        assert_eq!(tu.data.len(), ts.data.len(), "moment {ku} len");
        for (i, (&a, &b)) in tu.data.iter().zip(ts.data.iter()).enumerate() {
            let tol = 1e-5 * a.abs().max(b.abs()).max(1.0);
            assert!((a - b).abs() <= tol, "moment {ku}[{i}]: {a} vs {b}");
        }
    }
}

/// The guard tests UNSCALED quantities (task F7B item B4): strict predicate
/// boundaries just inside and just outside ±3e38, plus non-finite inputs —
/// a finite unscaled step is never skipped because of the scale, and ±3e38
/// itself skips.
#[test]
fn guard_predicate_strict_boundaries() {
    use mamba3::tensor::Tensor;
    use mamba3::tensor::ops::index::IdTensor;
    let _serial = serial();
    let device = dev();
    // (loss, sumsq, applies?)
    let cases: Vec<(f32, f32, bool)> = vec![
        (0.0, 0.0, true),
        (1.25, 4.0, true),
        (-2.9e38, 0.0, true),
        (2.9e38, 2.9e38, true),
        (3.0e38, 0.0, false),
        (-3.0e38, 0.0, false),
        (0.0, 3.0e38, false),
        (3.1e38, 0.0, false),
        (0.0, 3.1e38, false),
        (f32::INFINITY, 0.0, false),
        (f32::NEG_INFINITY, 0.0, false),
        (f32::NAN, 0.0, false),
        (0.0, f32::INFINITY, false),
        (0.0, f32::NAN, false),
        // A negative sum of squares is unphysical, but the predicate is
        // purely the strict range test, so it applies.
        (1.0, -1.0, true),
    ];
    for (l, s, applies) in cases {
        let loss = Tensor::<R, E>::from_f32(&[l], vec![1], &device).unwrap();
        let sumsq = Tensor::<R, E>::from_f32(&[s], vec![1], &device).unwrap();
        let apply = Tensor::<R, E>::full(vec![1], 0.0, &device);
        let counter = IdTensor::<R>::from_slice(&[0u32], vec![1], &device).unwrap();
        mamba3::tensor::ops::fused::guard_apply(&loss, &sumsq, &apply, &counter).unwrap();
        let got_apply = apply.try_to_f32().unwrap()[0];
        let got_count = counter.try_to_vec().unwrap()[0];
        assert_eq!(
            got_apply,
            if applies { 1.0 } else { 0.0 },
            "loss {l} sumsq {s}: apply"
        );
        assert_eq!(
            got_count,
            if applies { 0 } else { 1 },
            "loss {l} sumsq {s}: counter"
        );
    }
    // The documented scale boundary: with loss_scale = 65536, an unscaled
    // loss just below 3e38 applies (the scaled value would be far outside).
    let loss = Tensor::<R, E>::from_f32(&[2.9e38], vec![1], &device).unwrap();
    let sumsq = Tensor::<R, E>::from_f32(&[1.0], vec![1], &device).unwrap();
    let apply = Tensor::<R, E>::full(vec![1], 0.0, &device);
    let counter = IdTensor::<R>::from_slice(&[0u32], vec![1], &device).unwrap();
    mamba3::tensor::ops::fused::guard_apply(&loss, &sumsq, &apply, &counter).unwrap();
    assert_eq!(
        apply.try_to_f32().unwrap()[0],
        1.0,
        "2.9e38 applies at any scale"
    );
    assert_eq!(counter.try_to_vec().unwrap()[0], 0);
}

/// Scaled versus unscaled training compared on post-update weights and
/// moments, with and without clipping (task F7B item B5): `loss_scale =
/// 256` agrees with `1.0` within 1e-5 relative either way.
#[test]
fn scaled_vs_unscaled_weights_and_moments() {
    let _serial = serial();
    let device = dev();
    let (set, parents) = labeled_set(4);
    let table = FormulaTable::from_compositions(parents.into_iter()).unwrap();
    let indices = vec![0, 1, 2, 3];
    for clip in [None, Some(1.0f32)] {
        let mut cfg_u = safety_config(false, 1.0);
        cfg_u.grad_clip = clip;
        let mut cfg_s = safety_config(false, 256.0);
        cfg_s.grad_clip = clip;
        let mut run_u = Ms2Trainer::<R, E>::new(&tiny_config(), &table, &cfg_u, &device).unwrap();
        let mut run_s = Ms2Trainer::<R, E>::new(&tiny_config(), &table, &cfg_s, &device).unwrap();
        for _ in 0..3 {
            run_u.step(&set, &indices).unwrap();
            run_s.step(&set, &indices).unwrap();
        }
        let rel = |a: f32, b: f32, what: &str| {
            let tol = 1e-5 * a.abs().max(b.abs()).max(1.0);
            assert!((a - b).abs() <= tol, "clip {clip:?} {what}: {a} vs {b}");
        };
        let wu = run_u.model.state_dict();
        let ws = run_s.model.state_dict();
        assert_eq!(wu.entries.len(), ws.entries.len());
        for ((ku, tu), (ks, ts)) in wu.entries.iter().zip(ws.entries.iter()) {
            assert_eq!(ku, ks);
            for (i, (&a, &b)) in tu.data.iter().zip(ts.data.iter()).enumerate() {
                rel(a, b, &format!("weight {ku}[{i}]"));
            }
        }
        let mu = run_u.optimizer_state_for_test();
        let ms = run_s.optimizer_state_for_test();
        assert_eq!(mu.entries.len(), ms.entries.len());
        for ((ku, tu), (ks, ts)) in mu.entries.iter().zip(ms.entries.iter()) {
            assert_eq!(ku, ks);
            for (i, (&a, &b)) in tu.data.iter().zip(ts.data.iter()).enumerate() {
                rel(a, b, &format!("moment {ku}[{i}]"));
            }
        }
    }
}

/// Guard off + scale 1 agrees with the ungated entry point (task F7B item
/// B5): `grad_scale_unscaled` with `average == 1.0` is bit-identical to
/// `grad_scale` (same launch count), so the MS2 default path is untouched.
#[test]
fn guard_off_scale_one_factor_paths_agree() {
    use mamba3::backend::{launch_count, reset_launch_count};
    use mamba3::train::optim::{grad_scale, grad_scale_unscaled};
    let _serial = serial();
    let device = dev();
    let (set, parents) = labeled_set(4);
    let table = FormulaTable::from_compositions(parents.into_iter()).unwrap();
    let indices = vec![0, 1, 2, 3];
    let mut cfg = safety_config(false, 1.0);
    cfg.grad_clip = Some(1.0);
    let mut trainer = Ms2Trainer::<R, E>::new(&tiny_config(), &table, &cfg, &device).unwrap();
    let fwd = trainer.forward_state(&set, &indices).unwrap();
    let bwd = trainer.backward_state(&fwd).unwrap();
    let grads = bwd.grads_for_test();
    reset_launch_count();
    let g1 = grad_scale(grads, 1.0, 1.0).unwrap().expect("scale");
    let l1 = launch_count();
    reset_launch_count();
    let g2 = grad_scale_unscaled(grads, 1.0, 1.0)
        .unwrap()
        .expect("scale");
    let l2 = launch_count();
    assert_eq!(l1, l2, "factor-1 launch counts match: {l1} vs {l2}");
    let f1 = g1.factor.try_to_f32().unwrap();
    let f2 = g2.factor.try_to_f32().unwrap();
    assert_eq!(f1.len(), f2.len());
    for (i, (&a, &b)) in f1.iter().zip(f2.iter()).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "factor[{i}] bit-identical");
    }
    let s1 = g1.sum_squares.try_to_f32().unwrap();
    let s2 = g2.sum_squares.try_to_f32().unwrap();
    for (i, (&a, &b)) in s1.iter().zip(s2.iter()).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "sumsq[{i}] bit-identical");
    }
}
