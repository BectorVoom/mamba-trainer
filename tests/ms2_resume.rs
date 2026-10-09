//! T1A Part 1: schema-2 checkpoints with exact resume (plan P7.8).
//!
//! CPU and GPU via `backends::Auto`: on the CPU runtime resumed losses and
//! weights compare bit-identical; on a GPU backend within 1e-5 relative (the
//! tolerance is stated at each comparison). Every device-touching test holds
//! the file-level `serial()` mutex: launch/read counters are process-global.
//!
//! ```text
//! CARGO_TARGET_DIR=target/cpu-ms2 cargo test --release \
//!   --no-default-features --features cpu --test ms2_resume
//! ```

#![cfg(feature = "backend")]

#[path = "common/mod.rs"]
mod common;

use mamba3::backend::Device;
use mamba3::backends::Auto;
use mamba3::models::ms2::chem::{
    CHEMISTRY_VERSION, Composition, ELECTRON_MASS, ELEMENTS, HYDROGEN, adduct, composition_mass,
};
use mamba3::models::ms2::contract::{Control, GenerationConfig, ModelConfig};
use mamba3::models::ms2::dataset::ExportSpectrum;
use mamba3::models::ms2::experiment::{
    ExperimentSet, ExperimentSpectrum, SpectrumDomain, label_export_spectrum,
};
use mamba3::models::ms2::formula::FormulaTable;
use mamba3::models::ms2::grammar::Limits;
use mamba3::models::ms2::targets::{Candidates, Peak, RecipeLimits};
use mamba3::models::ms2::train::{
    DataCursor, ExportProvenance, GoldFormulaConditioning, Ms2Trainer, TrainConfig, TrainProvenance,
};
use mamba3::models::ms2::{MolGraph, RawAtom, RawMolecule};
use mamba3::nn::Module;

type R = Auto;
type E = f32;

fn dev() -> Device<R> {
    Device::<R>::default()
}

/// Process-global counters are read by tests in this binary: every
/// device-touching test holds this lock for its whole body.
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn serial() -> std::sync::MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

/// Whether the test backend is the CPU runtime (bit-equal floats) rather
/// than a GPU one (1e-5 relative).
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

/// Resume-test config: weight decay > 0 and a gradient clip, so both the
/// AdamW decay path and the device-side clip scale are exercised.
fn resume_config() -> TrainConfig {
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
        nonfinite_guard: false,
        loss_scale: 1.0,
    }
}

fn test_provenance() -> TrainProvenance {
    TrainProvenance {
        main: ExportProvenance {
            export_name: "train.json".to_string(),
            export_sha256: "aa".repeat(32),
            molecules: 4,
            molecule_keys_sha256: "bb".repeat(32),
        },
        fitted_on: vec![ExportProvenance {
            export_name: "fit.json".to_string(),
            export_sha256: "cc".repeat(32),
            molecules: 3,
            molecule_keys_sha256: "dd".repeat(32),
        }],
        previous_exposure: Vec::new(),
    }
}

fn assert_losses_close(a: &[f32], b: &[f32], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: loss count");
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        if is_cpu() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "{what} loss[{i}] differs on cpu: {x} vs {y}"
            );
        } else {
            // GPU tolerance, stated: 1e-5 relative.
            let tol = 1e-5 * x.abs().max(y.abs()).max(1.0);
            assert!(
                (x - y).abs() <= tol,
                "{what} loss[{i}] differs on gpu beyond 1e-5 relative: {x} vs {y}"
            );
        }
    }
}

fn assert_weights_close(
    a: &mamba3::nn::module::StateDict,
    b: &mamba3::nn::module::StateDict,
    what: &str,
) {
    assert_eq!(a.entries.len(), b.entries.len(), "{what}: entry count");
    for ((name_a, ta), (name_b, tb)) in a.entries.iter().zip(b.entries.iter()) {
        assert_eq!(name_a, name_b, "{what}: entry order");
        assert_eq!(ta.shape, tb.shape, "{what}: {name_a} shape");
        assert_eq!(ta.data.len(), tb.data.len(), "{what}: {name_a} len");
        for (i, (&x, &y)) in ta.data.iter().zip(tb.data.iter()).enumerate() {
            if is_cpu() {
                assert_eq!(
                    x.to_bits(),
                    y.to_bits(),
                    "{what} {name_a}[{i}] differs on cpu: {x} vs {y}"
                );
            } else {
                // GPU tolerance, stated: 1e-5 relative.
                let tol = 1e-5 * x.abs().max(y.abs()).max(1.0);
                assert!(
                    (x - y).abs() <= tol,
                    "{what} {name_a}[{i}] differs on gpu beyond 1e-5 relative: {x} vs {y}"
                );
            }
        }
    }
}

/// Train 6 steps uninterrupted against 3 + save/load + 3 with the restored
/// cursor: the last 3 reported losses and every final weight must match
/// (bit-identical on CPU, 1e-5 relative on GPU, tolerances stated above).
/// Weight decay > 0 and a gradient clip are on, so both paths are exercised.
#[test]
fn exact_resume() {
    let _serial = serial();
    let device = dev();
    let (set, parents) = labeled_set(4);
    let table = FormulaTable::from_compositions(parents.into_iter()).unwrap();
    let indices = vec![0, 1, 2, 3];
    let path = std::env::temp_dir().join("ms2_resume_exact.json");

    // Uninterrupted run: 6 reported steps.
    let mut ref_trainer =
        Ms2Trainer::<R, E>::new(&tiny_config(), &table, &resume_config(), &device).unwrap();
    let mut ref_losses = Vec::new();
    for _ in 0..6 {
        ref_trainer.request_report();
        let rep = ref_trainer.step(&set, &indices).unwrap().expect("report");
        ref_losses.push(rep.loss);
    }
    let ref_weights = ref_trainer.model.state_dict();
    let ref_opt = ref_trainer.optimizer_state_for_test();

    // Interrupted run: 3 steps, save with cursor + provenance, drop, load, 3 more.
    let mut first =
        Ms2Trainer::<R, E>::new(&tiny_config(), &table, &resume_config(), &device).unwrap();
    for _ in 0..3 {
        first.request_report();
        first.step(&set, &indices).unwrap().expect("report");
    }
    first.set_data_cursor(DataCursor {
        seed: 41,
        epoch: 1,
        position: 2,
    });
    first.set_train_provenance(test_provenance());
    first.save(&path).unwrap();
    drop(first);
    let mut second = Ms2Trainer::<R, E>::load(&path, &table, &device).unwrap();
    assert!(
        second.optimizer_restored(),
        "schema-2 load restores the optimizer"
    );
    // The stored cursor continues the SAME epoch order (driver-side); the
    // trainer-level run below uses the same batches, which is what the driver
    // feeds after restoring the cursor.
    assert_eq!(
        second.data_cursor(),
        Some(&DataCursor {
            seed: 41,
            epoch: 1,
            position: 2
        }),
        "data cursor round-trips"
    );
    assert_eq!(
        second.train_provenance(),
        Some(&test_provenance()),
        "provenance round-trips"
    );
    let mut got_losses = Vec::new();
    for _ in 0..3 {
        second.request_report();
        let rep = second.step(&set, &indices).unwrap().expect("report");
        got_losses.push(rep.loss);
    }
    let got_weights = second.model.state_dict();
    let got_opt = second.optimizer_state_for_test();

    assert_losses_close(&ref_losses[3..], &got_losses, "resumed");
    assert_weights_close(&ref_weights, &got_weights, "resumed weights");
    assert_weights_close(&ref_opt, &got_opt, "resumed moments");
    assert_eq!(second.step_count(), ref_trainer.step_count());
    std::fs::remove_file(&path).ok();
}

/// A schema-1 checkpoint loads with a fresh optimizer
/// (`optimizer_restored() == false`, provenance `None`) and trains.
#[test]
fn schema_1_loads_fresh_and_trains() {
    let _serial = serial();
    let device = dev();
    let (set, parents) = labeled_set(2);
    let table = FormulaTable::from_compositions(parents.into_iter()).unwrap();
    let indices = vec![0, 1];
    let path = std::env::temp_dir().join("ms2_resume_schema1.json");
    let mut trainer =
        Ms2Trainer::<R, E>::new(&tiny_config(), &table, &resume_config(), &device).unwrap();
    trainer.request_report();
    trainer.step(&set, &indices).unwrap().expect("report");
    trainer.save(&path).unwrap();
    // Rewrite as the old struct shape: schema 1 with none of the schema-2 keys.
    let text = std::fs::read_to_string(&path).unwrap();
    let mut v: serde_json::Value = serde_json::from_str(&text).unwrap();
    v["schema_version"] = serde_json::json!(1);
    for key in [
        "optimizer_state",
        "optimizer_steps",
        "dtype",
        "chemistry_version",
        "recipe_version",
        "grammar_version",
        "traversal_version",
        "spectrum_schema_version",
        "crate_version",
        "data_cursor",
        "train_provenance",
    ] {
        v.as_object_mut().unwrap().remove(key);
    }
    std::fs::write(&path, serde_json::to_string_pretty(&v).unwrap()).unwrap();
    let mut loaded = Ms2Trainer::<R, E>::load(&path, &table, &device).unwrap();
    assert!(
        !loaded.optimizer_restored(),
        "schema-1 load rebuilds a fresh optimizer"
    );
    assert_eq!(
        loaded.train_provenance(),
        None,
        "schema 1 has no provenance"
    );
    assert_eq!(loaded.data_cursor(), None, "schema 1 has no cursor");
    loaded.request_report();
    let rep = loaded.step(&set, &indices).unwrap().expect("report");
    assert!(rep.loss.is_finite(), "schema-1 load trains: {rep:?}");
    std::fs::remove_file(&path).ok();
}

/// Each refusal names its field: dtype, chemistry version, recipe version and
/// unknown schema are all `Error::Config` naming the field. The mismatching
/// files are made by editing the checkpoint JSON in the test.
#[test]
fn refusals_name_the_field() {
    let _serial = serial();
    let device = dev();
    let (set, parents) = labeled_set(2);
    let table = FormulaTable::from_compositions(parents.into_iter()).unwrap();
    let indices = vec![0, 1];
    let path = std::env::temp_dir().join("ms2_resume_refuse.json");
    let mut trainer =
        Ms2Trainer::<R, E>::new(&tiny_config(), &table, &resume_config(), &device).unwrap();
    trainer.step(&set, &indices).unwrap();
    trainer.save(&path).unwrap();
    let base = std::fs::read_to_string(&path).unwrap();
    let edit = |f: &dyn Fn(&mut serde_json::Value)| {
        let mut v: serde_json::Value = serde_json::from_str(&base).unwrap();
        f(&mut v);
        std::fs::write(&path, serde_json::to_string_pretty(&v).unwrap()).unwrap();
        Ms2Trainer::<R, E>::load(&path, &table, &device)
            .err()
            .expect("refused")
    };
    let err = edit(&|v| v["dtype"] = serde_json::json!("f16"));
    assert!(matches!(err, mamba3::error::Error::Config(_)), "{err}");
    assert!(err.to_string().contains("dtype"), "{err}");
    let err = edit(&|v| v["chemistry_version"] = serde_json::json!("ms2-chem-v9.9"));
    assert!(matches!(err, mamba3::error::Error::Config(_)), "{err}");
    assert!(err.to_string().contains("chemistry_version"), "{err}");
    let err = edit(&|v| v["recipe_version"] = serde_json::json!("q-cut-v9"));
    assert!(matches!(err, mamba3::error::Error::Config(_)), "{err}");
    assert!(err.to_string().contains("recipe_version"), "{err}");
    let err = edit(&|v| v["schema_version"] = serde_json::json!(99));
    assert!(matches!(err, mamba3::error::Error::Config(_)), "{err}");
    assert!(err.to_string().contains("schema_version"), "{err}");
    std::fs::remove_file(&path).ok();
}

/// Training provenance round-trips through save/load (leakage-guard fields).
#[test]
fn provenance_round_trip() {
    let _serial = serial();
    let device = dev();
    let (set, parents) = labeled_set(2);
    let table = FormulaTable::from_compositions(parents.into_iter()).unwrap();
    let indices = vec![0, 1];
    let path = std::env::temp_dir().join("ms2_resume_prov.json");
    let mut trainer =
        Ms2Trainer::<R, E>::new(&tiny_config(), &table, &resume_config(), &device).unwrap();
    trainer.step(&set, &indices).unwrap();
    trainer.set_train_provenance(test_provenance());
    trainer.set_data_cursor(DataCursor {
        seed: 7,
        epoch: 3,
        position: 1,
    });
    trainer.save(&path).unwrap();
    let loaded = Ms2Trainer::<R, E>::load(&path, &table, &device).unwrap();
    assert_eq!(loaded.train_provenance(), Some(&test_provenance()));
    assert_eq!(
        loaded.data_cursor(),
        Some(&DataCursor {
            seed: 7,
            epoch: 3,
            position: 1
        })
    );
    std::fs::remove_file(&path).ok();
}

/// Enumeration artifacts survive the round trip.
#[test]
fn enum_artifacts_survive() {
    let _serial = serial();
    use mamba3::models::ms2::formula_enum::{EnumDomain, RatioBounds};
    let device = dev();
    let (set, parents) = labeled_set(4);
    let table = FormulaTable::from_compositions(parents.clone().into_iter()).unwrap();
    let indices = vec![0, 1, 2, 3];
    let mut cfg = resume_config();
    cfg.formula_source = mamba3::models::ms2::contract::FormulaSource::Enumerate;
    let mut trainer = Ms2Trainer::<R, E>::new(&tiny_config(), &table, &cfg, &device).unwrap();
    let domain = EnumDomain::from_compositions(parents.clone(), 0).unwrap();
    let bounds = RatioBounds::fit(parents, 0).unwrap();
    trainer.upload_enum_artifacts(&domain, &bounds).unwrap();
    trainer.request_report();
    let before = trainer.step(&set, &indices).unwrap().expect("report");
    let path = std::env::temp_dir().join("ms2_resume_enum.json");
    trainer.save(&path).unwrap();
    let mut loaded = Ms2Trainer::<R, E>::load(&path, &table, &device).unwrap();
    assert!(
        loaded.model.enum_artifacts.is_some(),
        "enum artifacts survive"
    );
    loaded.request_report();
    // A fresh trainer at the same weights would need the same artifacts to
    // run; the loaded one runs and reports finite losses.
    let after = loaded.teacher_eval(&set, &indices).unwrap();
    assert!(after.nll.iter().all(|v| v.is_finite()));
    assert!(before.loss.is_finite());
    std::fs::remove_file(&path).ok();
}

/// The assignment head survives the round trip (parameters travel with the
/// model and keep working after load).
#[test]
fn assignment_head_survives() {
    let _serial = serial();
    use mamba3::models::ms2::contract::AssignmentConfig;
    let device = dev();
    let (set, parents) = labeled_set(2);
    let table = FormulaTable::from_compositions(parents.into_iter()).unwrap();
    let indices = vec![0, 1];
    let mut model = tiny_config();
    model.assignment = Some(AssignmentConfig {
        hypotheses: 4,
        work_max: 4096,
        labels: 64,
    });
    let mut cfg = resume_config();
    cfg.lambda_assign = 0.1;
    let mut trainer = Ms2Trainer::<R, E>::new(&model, &table, &cfg, &device).unwrap();
    trainer.request_report();
    let before = trainer.step(&set, &indices).unwrap().expect("report");
    let keys_before: Vec<String> = {
        let mut ks: Vec<String> = trainer
            .model
            .state_dict()
            .entries
            .keys()
            .filter(|k| k.contains("assignment"))
            .cloned()
            .collect();
        ks.sort();
        ks
    };
    assert!(!keys_before.is_empty(), "assignment params exist");
    let path = std::env::temp_dir().join("ms2_resume_assign.json");
    trainer.save(&path).unwrap();
    let mut loaded = Ms2Trainer::<R, E>::load(&path, &table, &device).unwrap();
    let keys_after: Vec<String> = {
        let mut ks: Vec<String> = loaded
            .model
            .state_dict()
            .entries
            .keys()
            .filter(|k| k.contains("assignment"))
            .cloned()
            .collect();
        ks.sort();
        ks
    };
    assert_eq!(keys_before, keys_after, "assignment params survive");
    loaded.request_report();
    let after = loaded.step(&set, &indices).unwrap().expect("report");
    assert!(after.assign.is_finite(), "assignment loss works: {after:?}");
    assert!(
        before.assign.is_finite(),
        "baseline assignment loss: {before:?}"
    );
    std::fs::remove_file(&path).ok();
}

/// The evidence branch survives the round trip when the model has it.
#[test]
fn evidence_branch_survives() {
    let _serial = serial();
    let device = dev();
    let (set, parents) = labeled_set(2);
    let table = FormulaTable::from_compositions(parents.into_iter()).unwrap();
    let indices = vec![0, 1];
    let mut model = tiny_config();
    model.formula_features = mamba3::models::ms2::contract::FormulaFeatures::Evidence;
    let mut trainer = Ms2Trainer::<R, E>::new(&model, &table, &resume_config(), &device).unwrap();
    let keys_before: Vec<String> = {
        let mut ks: Vec<String> = trainer
            .model
            .state_dict()
            .entries
            .keys()
            .filter(|k| k.contains("evidence"))
            .cloned()
            .collect();
        ks.sort();
        ks
    };
    let path = std::env::temp_dir().join("ms2_resume_evidence.json");
    trainer.step(&set, &indices).unwrap();
    trainer.save(&path).unwrap();
    let mut loaded = Ms2Trainer::<R, E>::load(&path, &table, &device).unwrap();
    let keys_after: Vec<String> = {
        let mut ks: Vec<String> = loaded
            .model
            .state_dict()
            .entries
            .keys()
            .filter(|k| k.contains("evidence"))
            .cloned()
            .collect();
        ks.sort();
        ks
    };
    if keys_before.is_empty() {
        // The model has no separate evidence branch at this width: the round
        // trip still works and evaluates.
        loaded.request_report();
        let rep = loaded.step(&set, &indices).unwrap().expect("report");
        assert!(rep.loss.is_finite());
    } else {
        assert_eq!(keys_before, keys_after, "evidence params survive");
        let a = trainer.teacher_eval(&set, &indices).unwrap();
        let b = loaded.teacher_eval(&set, &indices).unwrap();
        assert_eq!(a.nll.len(), b.nll.len());
        for (i, (&x, &y)) in a.nll.iter().zip(b.nll.iter()).enumerate() {
            assert_eq!(x.to_bits(), y.to_bits(), "nll[{i}] {x} vs {y}");
        }
    }
    std::fs::remove_file(&path).ok();
}

/// `TrainConfig::loss_scale` validates: non-powers-of-two and out-of-range
/// values are `Error::Config`.
#[test]
fn loss_scale_validates() {
    let _serial = serial();
    for bad in [0.5, 3.0, 100.0, 131072.0, f32::NAN, f32::INFINITY] {
        let mut cfg = resume_config();
        cfg.loss_scale = bad;
        let err = cfg.validate().unwrap_err();
        assert!(
            matches!(err, mamba3::error::Error::Config(_)),
            "{bad}: {err}"
        );
        assert!(err.to_string().contains("loss_scale"), "{bad}: {err}");
    }
    for good in [1.0, 2.0, 256.0, 65536.0] {
        let mut cfg = resume_config();
        cfg.loss_scale = good;
        cfg.validate().unwrap();
    }
    let _ = GenerationConfig::default();
    let _ = CHEMISTRY_VERSION;
}

/// Training provenance with a non-empty `previous_exposure` (task F7B item
/// B1) round-trips through save/load, and an older checkpoint JSON without
/// the field loads it as empty.
#[test]
fn provenance_previous_exposure_round_trip() {
    let _serial = serial();
    let device = dev();
    let (set, parents) = labeled_set(2);
    let table = FormulaTable::from_compositions(parents.into_iter()).unwrap();
    let indices = vec![0, 1];
    let path = std::env::temp_dir().join("ms2_resume_prev_exp.json");
    let mut trainer =
        Ms2Trainer::<R, E>::new(&tiny_config(), &table, &resume_config(), &device).unwrap();
    trainer.step(&set, &indices).unwrap();
    let mut prov = test_provenance();
    prov.previous_exposure = vec![
        ExportProvenance {
            export_name: "older.json".to_string(),
            export_sha256: "ee".repeat(32),
            molecules: 2,
            molecule_keys_sha256: "ff".repeat(32),
        },
        mamba3::models::ms2::train::schema1_unrecorded_provenance(),
    ];
    trainer.set_train_provenance(prov.clone());
    trainer.save(&path).unwrap();
    let loaded = Ms2Trainer::<R, E>::load(&path, &table, &device).unwrap();
    assert_eq!(loaded.train_provenance(), Some(&prov));
    assert_eq!(
        loaded.train_provenance().unwrap().previous_exposure.len(),
        2,
        "previous exposure survives"
    );
    // An older checkpoint JSON without `previous_exposure` loads it as empty.
    let text = std::fs::read_to_string(&path).unwrap();
    let mut v: serde_json::Value = serde_json::from_str(&text).unwrap();
    v["train_provenance"]["previous_exposure"] = serde_json::Value::Null;
    // `None` is not `[]`: remove the key instead to mimic the old shape.
    v["train_provenance"]
        .as_object_mut()
        .unwrap()
        .remove("previous_exposure");
    std::fs::write(&path, serde_json::to_string_pretty(&v).unwrap()).unwrap();
    let old = Ms2Trainer::<R, E>::load(&path, &table, &device).unwrap();
    assert_eq!(
        old.train_provenance().unwrap().previous_exposure,
        Vec::new(),
        "missing previous_exposure loads as empty"
    );
    std::fs::remove_file(&path).ok();
}

/// Refusals for the remaining versioned fields and malformed states (task
/// F7B item B5): grammar, traversal and spectrum-schema versions, a missing
/// required field and a malformed optimizer state are all `Error::Config`
/// naming the field.
#[test]
fn refusals_name_the_field_f7b() {
    let _serial = serial();
    let device = dev();
    let (set, parents) = labeled_set(2);
    let table = FormulaTable::from_compositions(parents.into_iter()).unwrap();
    let indices = vec![0, 1];
    let path = std::env::temp_dir().join("ms2_resume_refuse_f7b.json");
    let mut trainer =
        Ms2Trainer::<R, E>::new(&tiny_config(), &table, &resume_config(), &device).unwrap();
    trainer.step(&set, &indices).unwrap();
    trainer.save(&path).unwrap();
    let base = std::fs::read_to_string(&path).unwrap();
    let edit = |f: &dyn Fn(&mut serde_json::Value)| {
        let mut v: serde_json::Value = serde_json::from_str(&base).unwrap();
        f(&mut v);
        std::fs::write(&path, serde_json::to_string_pretty(&v).unwrap()).unwrap();
        Ms2Trainer::<R, E>::load(&path, &table, &device)
            .err()
            .expect("refused")
    };
    let err = edit(&|v| v["grammar_version"] = serde_json::json!("grammar-bfs-v9"));
    assert!(matches!(err, mamba3::error::Error::Config(_)), "{err}");
    assert!(err.to_string().contains("grammar_version"), "{err}");
    let err = edit(&|v| v["traversal_version"] = serde_json::json!("nope"));
    assert!(matches!(err, mamba3::error::Error::Config(_)), "{err}");
    assert!(err.to_string().contains("traversal_version"), "{err}");
    let err = edit(&|v| v["spectrum_schema_version"] = serde_json::json!(999));
    assert!(matches!(err, mamba3::error::Error::Config(_)), "{err}");
    assert!(err.to_string().contains("spectrum_schema_version"), "{err}");
    // A missing required field names the field.
    let err = edit(&|v| {
        v.as_object_mut().unwrap().remove("optimizer_state");
    });
    assert!(matches!(err, mamba3::error::Error::Config(_)), "{err}");
    assert!(err.to_string().contains("optimizer_state"), "{err}");
    let err = edit(&|v| {
        v.as_object_mut().unwrap().remove("dtype");
    });
    assert!(matches!(err, mamba3::error::Error::Config(_)), "{err}");
    assert!(err.to_string().contains("dtype"), "{err}");
    // A malformed optimizer state (unknown entry under strict load) is
    // refused naming the optimizer state.
    let err = edit(&|v| {
        v["optimizer_state"]["entries"]["zzz_bogus"] = serde_json::json!({
            "shape": [1],
            "data": [0.0],
        });
    });
    assert!(matches!(err, mamba3::error::Error::Config(_)), "{err}");
    assert!(err.to_string().contains("optimizer_state"), "{err}");
    std::fs::remove_file(&path).ok();
}

/// Poison one formula-head weight to +inf so the step's loss is non-finite
/// (same technique as `tests/ms2_step_safety.rs`).
fn poison_formula_weight_here(trainer: &mut Ms2Trainer<R, E>) {
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

/// Skip, save, load, then a healthy step: the resumed total equals the
/// uninterrupted run's, and the post-resume weights and losses are
/// bit-identical on CPU (task F7B items B3 and B5: resume after a skip;
/// the checkpoint carries the counter).
#[test]
fn resume_after_skip_counts_and_matches() {
    let _serial = serial();
    let device = dev();
    let (set, parents) = labeled_set(4);
    let table = FormulaTable::from_compositions(parents.into_iter()).unwrap();
    let indices = vec![0, 1, 2, 3];
    let mut cfg = resume_config();
    cfg.nonfinite_guard = true;
    let path = std::env::temp_dir().join("ms2_resume_skip.json");

    // Uninterrupted run: healthy, skip, restore, healthy — all reported.
    let mut run_a = Ms2Trainer::<R, E>::new(&tiny_config(), &table, &cfg, &device).unwrap();
    run_a.request_report();
    let a0 = run_a.step(&set, &indices).unwrap().expect("report");
    assert_eq!(a0.skipped_steps_total, 0);
    let healthy_dict = run_a.model.state_dict();
    poison_formula_weight_here(&mut run_a);
    run_a.request_report();
    let a1 = run_a.step(&set, &indices).unwrap().expect("report");
    assert!(a1.last_step_skipped, "poisoned step skips");
    assert_eq!(a1.skipped_steps_total, 1);
    run_a.model.load_state_dict(&healthy_dict, true).unwrap();
    run_a.request_report();
    let a2 = run_a.step(&set, &indices).unwrap().expect("report");
    assert!(!a2.last_step_skipped);
    assert_eq!(a2.skipped_steps_total, 1, "total survives a healthy step");
    let a_weights = run_a.model.state_dict();

    // Interrupted run: healthy, skip, SAVE, load, restore, healthy.
    let mut run_b = Ms2Trainer::<R, E>::new(&tiny_config(), &table, &cfg, &device).unwrap();
    run_b.request_report();
    let b0 = run_b.step(&set, &indices).unwrap().expect("report");
    let healthy_b = run_b.model.state_dict();
    poison_formula_weight_here(&mut run_b);
    run_b.request_report();
    let b1 = run_b.step(&set, &indices).unwrap().expect("report");
    assert!(b1.last_step_skipped);
    // Restore BEFORE saving: non-finite weights have no JSON form (a
    // pre-existing checkpoint limitation, orthogonal to the counter), while
    // the device counter keeps its total across the restore.
    run_b.model.load_state_dict(&healthy_b, true).unwrap();
    run_b.save(&path).unwrap();
    // The checkpoint names the counter total.
    let saved: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    assert_eq!(saved["skipped_steps_total"], serde_json::json!(1));
    drop(run_b);
    let mut run_c = Ms2Trainer::<R, E>::load(&path, &table, &device).unwrap();
    run_c.request_report();
    let c2 = run_c.step(&set, &indices).unwrap().expect("report");
    assert!(!c2.last_step_skipped);
    assert_eq!(
        c2.skipped_steps_total, a2.skipped_steps_total,
        "resumed total equals the uninterrupted total"
    );
    if is_cpu() {
        assert_eq!(b0.loss.to_bits(), a0.loss.to_bits());
        assert_eq!(b1.loss.to_bits(), a1.loss.to_bits());
        assert_eq!(
            c2.loss.to_bits(),
            a2.loss.to_bits(),
            "post-resume loss bit-identical"
        );
        let c_weights = run_c.model.state_dict();
        assert_eq!(a_weights.entries.len(), c_weights.entries.len());
        for ((ka, ta), (kc, tc)) in a_weights.entries.iter().zip(c_weights.entries.iter()) {
            assert_eq!(ka, kc);
            assert_eq!(ta.shape, tc.shape);
            assert_eq!(ta.data, tc.data, "post-resume weight {ka} bit-identical");
        }
        assert_eq!(
            run_c.optimizer_state_for_test().entries.len(),
            run_a.optimizer_state_for_test().entries.len()
        );
        for ((ka, ta), (kc, tc)) in run_a
            .optimizer_state_for_test()
            .entries
            .iter()
            .zip(run_c.optimizer_state_for_test().entries.iter())
        {
            assert_eq!(ka, kc);
            assert_eq!(ta.data, tc.data, "post-resume moment {ka} bit-identical");
        }
    } else {
        let tol = 1e-5 * a2.loss.abs().max(c2.loss.abs()).max(1.0);
        assert!((a2.loss - c2.loss).abs() <= tol);
    }
    std::fs::remove_file(&path).ok();
}

/// Exact resume across two epoch boundaries with the real shuffle, jitter
/// on, clip and decay on (task F7B item B5): the interrupted run's reported
/// losses, weights and moments are bit-identical to the uninterrupted run
/// on CPU.
#[test]
fn exact_resume_two_epochs_real_shuffle_jitter() {
    let _serial = serial();
    let device = dev();
    let (set, _) = labeled_set(6);
    let parents: Vec<Composition> = set.spectra.iter().map(|s| s.parent_composition).collect();
    let table = FormulaTable::from_compositions(parents.into_iter()).unwrap();
    let indices: Vec<usize> = (0..6).collect();
    let mut cfg = resume_config();
    cfg.batch = 2;
    cfg.seed = 41;
    cfg.precursor_jitter_ppm = 2.0;
    cfg.precursor_jitter_variants = 2;
    let path = std::env::temp_dir().join("ms2_resume_epochs.json");

    // Driver-shaped epoch loop with the real shuffle.
    fn run_steps(
        trainer: &mut Ms2Trainer<R, E>,
        set: &ExperimentSet,
        indices: &[usize],
        batch: usize,
        seed: u64,
        epoch0: u64,
        skip_in_epoch: usize,
        n: usize,
    ) -> (Vec<f32>, u64, usize) {
        let mut losses = Vec::new();
        let mut epoch = epoch0;
        let mut done = 0usize;
        let mut first = true;
        while done < n {
            let batches = set.batches(indices, batch, seed.wrapping_add(epoch));
            let skip = if first { skip_in_epoch } else { 0 };
            first = false;
            for chunk in batches.into_iter().skip(skip) {
                if done >= n {
                    break;
                }
                trainer.request_report();
                let rep = trainer.step(set, &chunk).unwrap().expect("report");
                losses.push(rep.loss);
                done += 1;
            }
            epoch += 1;
        }
        (losses, epoch - 1, done)
    }

    // Uninterrupted: 7 steps = epoch 0 (3) + epoch 1 (3) + epoch 2 (1).
    let mut run_ref = Ms2Trainer::<R, E>::new(&tiny_config(), &table, &cfg, &device).unwrap();
    let (ref_losses, _, _) = run_steps(&mut run_ref, &set, &indices, 2, 41, 0, 0, 7);
    let ref_weights = run_ref.model.state_dict();
    let ref_moments = run_ref.optimizer_state_for_test();

    // Interrupted after 4 steps (epoch 1, position 1): save, load, 3 more.
    let mut run_a = Ms2Trainer::<R, E>::new(&tiny_config(), &table, &cfg, &device).unwrap();
    let (losses_a, _, _) = run_steps(&mut run_a, &set, &indices, 2, 41, 0, 0, 4);
    run_a.set_data_cursor(DataCursor {
        seed: 41,
        epoch: 1,
        position: 1,
    });
    run_a.save(&path).unwrap();
    drop(run_a);
    let mut run_b = Ms2Trainer::<R, E>::load(&path, &table, &device).unwrap();
    assert!(run_b.optimizer_restored());
    let cursor = run_b.data_cursor().cloned().expect("cursor");
    assert_eq!(
        cursor,
        DataCursor {
            seed: 41,
            epoch: 1,
            position: 1
        }
    );
    let (losses_b, _, _) = run_steps(
        &mut run_b,
        &set,
        &indices,
        2,
        41,
        cursor.epoch,
        cursor.position,
        3,
    );
    let got_weights = run_b.model.state_dict();
    let got_moments = run_b.optimizer_state_for_test();

    assert_eq!(losses_a.len(), 4);
    assert_eq!(losses_b.len(), 3);
    let mut got_losses = losses_a;
    got_losses.extend(losses_b);
    assert_losses_close(&ref_losses, &got_losses, "two-epoch resume");
    if is_cpu() {
        for (i, (&x, &y)) in ref_losses.iter().zip(got_losses.iter()).enumerate() {
            assert_eq!(x.to_bits(), y.to_bits(), "loss[{i}] bit-identical");
        }
    }
    assert_weights_close(&ref_weights, &got_weights, "two-epoch weights");
    assert_weights_close(&ref_moments, &got_moments, "two-epoch moments");
    std::fs::remove_file(&path).ok();
}

/// Driver fixture for the F7B subprocess tests (task F7B items B1/B2, Part
/// C): `train_a.json` (first 4 molecules) and `train_b.json` (first 3) cut
/// from `data/ms2/overfit_train.json`, mirroring `e5f_k_driver_report`.
/// Returns `(train_a, train_b, table)` as strings, or `None` (skip) when
/// the data files are absent.
fn driver_fixture(dir: &std::path::Path) -> Option<(String, String, String)> {
    let manifest = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let src = manifest.join("data/ms2/overfit_train.json");
    if !src.exists() {
        println!("F7B-SKIP: data/ms2/overfit_train.json absent");
        return None;
    }
    let table = manifest.join("data/ms2/formula_table_msgym_v0.json");
    if !table.exists() {
        println!("F7B-SKIP: data/ms2/formula_table_msgym_v0.json absent");
        return None;
    }
    let raw = std::fs::read_to_string(&src).expect("export readable");
    let mut va: serde_json::Value = serde_json::from_str(&raw).expect("export parses");
    va["molecules"]
        .as_array_mut()
        .expect("molecules")
        .truncate(4);
    let train_a = dir.join("train_a.json");
    std::fs::write(&train_a, serde_json::to_string(&va).unwrap()).unwrap();
    let mut vb: serde_json::Value = serde_json::from_str(&raw).expect("export parses");
    vb["molecules"]
        .as_array_mut()
        .expect("molecules")
        .truncate(3);
    let train_b = dir.join("train_b.json");
    std::fs::write(&train_b, serde_json::to_string(&vb).unwrap()).unwrap();
    let s = |p: &std::path::Path| p.to_string_lossy().into_owned();
    Some((s(&train_a), s(&train_b), s(&table)))
}

fn driver_output(bin: &std::path::Path, args: &[String]) -> std::process::Output {
    std::process::Command::new(bin)
        .args(args)
        .output()
        .expect("example runs")
}

fn driver_arg(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| s.to_string()).collect()
}

/// `--resume` with a different export exits 2 naming the mismatch, while
/// the same export resumes through the driver (task F7B item B1; the
/// subprocess binary is resolved by the shared Part C helper).
#[test]
fn driver_refuses_resume_on_different_export() {
    let _serial = serial();
    let dir = std::env::temp_dir().join("ms2_f7b_resume_refuse");
    std::fs::create_dir_all(&dir).expect("temp dir");
    let Some((train_a, train_b, table)) = driver_fixture(&dir) else {
        return;
    };
    let bin = common::resolve_example_bin("ms2_experiment");
    let s = |p: &std::path::Path| p.to_string_lossy().into_owned();
    let ckpt = s(&dir.join("ckpt.json"));
    // Fresh run on A with an explicit seed (also pins B2's input).
    let out = driver_output(
        &bin,
        &driver_arg(&[
            "--train",
            &train_a,
            "--overfit",
            "3",
            "--table",
            &table,
            "--name",
            "f7b-a",
            "--steps",
            "1",
            "--batch",
            "2",
            "--seed",
            "41",
            "--save",
            &ckpt,
            "--out",
            &s(&dir.join("rep_a.json")),
        ]),
    );
    assert!(
        out.status.success(),
        "fresh run failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    // Resume on B: refusal with exit code 2 naming the mismatch.
    let out = driver_output(
        &bin,
        &driver_arg(&[
            "--train",
            &train_b,
            "--overfit",
            "1",
            "--table",
            &table,
            "--name",
            "f7b-b",
            "--steps",
            "2",
            "--batch",
            "2",
            "--load",
            &ckpt,
            "--resume",
            "--out",
            &s(&dir.join("rep_b.json")),
        ]),
    );
    assert!(
        !out.status.success(),
        "resume on a different export must fail"
    );
    assert_eq!(out.status.code(), Some(2), "resume mismatch exits 2");
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(stderr.contains("mismatch"), "names the mismatch: {stderr}");
    // Control: resume on the SAME export succeeds through the driver.
    let out = driver_output(
        &bin,
        &driver_arg(&[
            "--train",
            &train_a,
            "--overfit",
            "3",
            "--table",
            &table,
            "--name",
            "f7b-a2",
            "--steps",
            "1",
            "--batch",
            "2",
            "--load",
            &ckpt,
            "--resume",
            "--out",
            &s(&dir.join("rep_a2.json")),
        ]),
    );
    assert!(
        out.status.success(),
        "same-export resume failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// A non-resume `--load` that trains further records the new export as
/// `main` and appends the previous `main` to `previous_exposure`; a
/// schema-1 continuation marks earlier exposure as unrecorded (task F7B
/// item B1).
#[test]
fn driver_non_resume_load_chains_previous_exposure() {
    let _serial = serial();
    let dir = std::env::temp_dir().join("ms2_f7b_chain");
    std::fs::create_dir_all(&dir).expect("temp dir");
    let Some((train_a, train_b, table)) = driver_fixture(&dir) else {
        return;
    };
    let bin = common::resolve_example_bin("ms2_experiment");
    let s = |p: &std::path::Path| p.to_string_lossy().into_owned();
    let ckpt1 = s(&dir.join("ckpt1.json"));
    let out = driver_output(
        &bin,
        &driver_arg(&[
            "--train",
            &train_a,
            "--overfit",
            "3",
            "--table",
            &table,
            "--name",
            "f7b-c1",
            "--steps",
            "1",
            "--batch",
            "2",
            "--seed",
            "41",
            "--save",
            &ckpt1,
            "--out",
            &s(&dir.join("rep_c1.json")),
        ]),
    );
    assert!(
        out.status.success(),
        "fresh run failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let ck1: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&ckpt1).unwrap()).unwrap();
    let main1_sha = ck1["train_provenance"]["main"]["export_sha256"]
        .as_str()
        .expect("ckpt1 main sha")
        .to_string();
    // Continue on B without --resume: new main, old main in previous_exposure.
    let ckpt2 = s(&dir.join("ckpt2.json"));
    let out = driver_output(
        &bin,
        &driver_arg(&[
            "--train",
            &train_b,
            "--overfit",
            "1",
            "--table",
            &table,
            "--name",
            "f7b-c2",
            "--steps",
            "1",
            "--batch",
            "2",
            "--load",
            &ckpt1,
            "--save",
            &ckpt2,
            "--out",
            &s(&dir.join("rep_c2.json")),
        ]),
    );
    assert!(
        out.status.success(),
        "continued run failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let ck2: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&ckpt2).unwrap()).unwrap();
    let main2_sha = ck2["train_provenance"]["main"]["export_sha256"]
        .as_str()
        .expect("ckpt2 main sha");
    assert_ne!(main2_sha, main1_sha, "continued run records the new export");
    let prev = ck2["train_provenance"]["previous_exposure"]
        .as_array()
        .expect("previous_exposure array");
    assert_eq!(prev.len(), 1, "previous main appended: {ck2}");
    assert_eq!(prev[0]["export_sha256"].as_str().unwrap(), main1_sha);
    // Schema-1 continuation: current export recorded, earlier exposure marked.
    let mut v1: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&ckpt1).unwrap()).unwrap();
    v1["schema_version"] = serde_json::json!(1);
    for key in [
        "optimizer_state",
        "optimizer_steps",
        "dtype",
        "chemistry_version",
        "recipe_version",
        "grammar_version",
        "traversal_version",
        "spectrum_schema_version",
        "crate_version",
        "data_cursor",
        "train_provenance",
        "skipped_steps_total",
    ] {
        v1.as_object_mut().unwrap().remove(key);
    }
    let ckpt1s1 = s(&dir.join("ckpt1s1.json"));
    std::fs::write(&ckpt1s1, serde_json::to_string_pretty(&v1).unwrap()).unwrap();
    let ckpt3 = s(&dir.join("ckpt3.json"));
    let out = driver_output(
        &bin,
        &driver_arg(&[
            "--train",
            &train_b,
            "--overfit",
            "1",
            "--table",
            &table,
            "--name",
            "f7b-c3",
            "--steps",
            "1",
            "--batch",
            "2",
            "--load",
            &ckpt1s1,
            "--save",
            &ckpt3,
            "--out",
            &s(&dir.join("rep_c3.json")),
        ]),
    );
    assert!(
        out.status.success(),
        "schema-1 continuation failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let ck3: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&ckpt3).unwrap()).unwrap();
    let prev3 = ck3["train_provenance"]["previous_exposure"]
        .as_array()
        .expect("previous_exposure array");
    assert_eq!(prev3.len(), 1);
    assert_eq!(
        prev3[0]["export_name"].as_str().unwrap(),
        "unrecorded (schema-1 checkpoint)"
    );
}

/// A `--load` without `--seed` shuffles with the checkpoint seed and stores
/// it in the cursor, so a later `--resume` succeeds; an explicit
/// `--seed` that differs is refused (task F7B item B2).
#[test]
fn driver_load_uses_checkpoint_seed() {
    let _serial = serial();
    let dir = std::env::temp_dir().join("ms2_f7b_seed");
    std::fs::create_dir_all(&dir).expect("temp dir");
    let Some((train_a, _train_b, table)) = driver_fixture(&dir) else {
        return;
    };
    let bin = common::resolve_example_bin("ms2_experiment");
    let s = |p: &std::path::Path| p.to_string_lossy().into_owned();
    let ckpt1 = s(&dir.join("ckpt1.json"));
    let out = driver_output(
        &bin,
        &driver_arg(&[
            "--train",
            &train_a,
            "--overfit",
            "3",
            "--table",
            &table,
            "--name",
            "f7b-s1",
            "--steps",
            "1",
            "--batch",
            "2",
            "--seed",
            "41",
            "--save",
            &ckpt1,
            "--out",
            &s(&dir.join("rep_s1.json")),
        ]),
    );
    assert!(
        out.status.success(),
        "fresh run failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    // Load WITHOUT --seed, train one more step, save.
    let ckpt2 = s(&dir.join("ckpt2.json"));
    let out = driver_output(
        &bin,
        &driver_arg(&[
            "--train",
            &train_a,
            "--overfit",
            "3",
            "--table",
            &table,
            "--name",
            "f7b-s2",
            "--steps",
            "1",
            "--batch",
            "2",
            "--load",
            &ckpt1,
            "--save",
            &ckpt2,
            "--out",
            &s(&dir.join("rep_s2.json")),
        ]),
    );
    assert!(
        out.status.success(),
        "load without --seed failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let ck2: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&ckpt2).unwrap()).unwrap();
    assert_eq!(
        ck2["train_config"]["seed"].as_u64().unwrap(),
        41,
        "checkpoint seed preserved"
    );
    assert_eq!(
        ck2["data_cursor"]["seed"].as_u64().unwrap(),
        41,
        "stored cursor seed equals train_config.seed"
    );
    // ... so --resume now succeeds (before the fix the cursor stored the CLI
    // default 1 and this refused).
    let out = driver_output(
        &bin,
        &driver_arg(&[
            "--train",
            &train_a,
            "--overfit",
            "3",
            "--table",
            &table,
            "--name",
            "f7b-s3",
            "--steps",
            "2",
            "--batch",
            "2",
            "--load",
            &ckpt2,
            "--resume",
            "--out",
            &s(&dir.join("rep_s3.json")),
        ]),
    );
    assert!(
        out.status.success(),
        "--resume after seed-preserving load failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    // An explicit mismatching --seed is refused.
    let out = driver_output(
        &bin,
        &driver_arg(&[
            "--train",
            &train_a,
            "--overfit",
            "3",
            "--table",
            &table,
            "--name",
            "f7b-s4",
            "--steps",
            "1",
            "--batch",
            "2",
            "--seed",
            "7",
            "--load",
            &ckpt1,
            "--out",
            &s(&dir.join("rep_s4.json")),
        ]),
    );
    assert!(!out.status.success(), "mismatching --seed must fail");
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(stderr.contains("--seed"), "names --seed: {stderr}");
}
