//! V0-D2 tests: the training driver, its read budget, evaluation under every
//! control, checkpoint round-trips and the structure-prior invariance.
//!
//! Fixture-derived data only: the in-memory [`ExperimentSet`] is built from
//! `tests/fixtures/ms2/chemistry_v0.json` molecules with synthetic spectra
//! from the fixture's peak lists. No CASMI export is read here.

#![cfg(feature = "backend")]

use mamba3::backend::{Device, read_count, runtime_read_count};
use mamba3::backends::Auto;
use mamba3::models::ms2::batch::DeviceSpectra;
use mamba3::models::ms2::chem::{
    CHEMISTRY_VERSION, Composition, ELECTRON_MASS, ELEMENTS, HYDROGEN, adduct, composition_mass,
};
use mamba3::models::ms2::contract::{
    AssignmentConfig, Control, GenerationConfig, ModelConfig, SpectrumBatch,
};
use mamba3::models::ms2::dataset::ExportSpectrum;
use mamba3::models::ms2::encoder::Ms2Encoder;
use mamba3::models::ms2::experiment::{
    ExperimentSet, ExperimentSpectrum, SpectrumDomain, label_export_spectrum, spectrum_batch_for,
};
use mamba3::models::ms2::formula::FormulaTable;
use mamba3::models::ms2::grammar::Limits;
use mamba3::models::ms2::ion::ion_labels;
use mamba3::models::ms2::targets::{Candidates, Peak, RecipeLimits};
use mamba3::models::ms2::train::{GoldFormulaConditioning, Ms2Trainer, TrainConfig};
use mamba3::models::ms2::{MolGraph, RawAtom, RawMolecule};
use mamba3::tensor::ops::ms2;
use mamba3::tensor::ops::random::Rng;

type R = Auto;
type E = f32;

fn dev() -> Device<R> {
    Device::<R>::default()
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

/// One in-memory experiment spectrum from a fixture molecule and its first
/// fixture spectrum's peak list, with the precursor set from the true parent
/// mass so the gold formula joins the window.
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

/// An [`ExperimentSet`] of the first `n` labeled fixture molecules in fixture
/// order (precursor in range, at least one recipe target), with their parent
/// compositions for the table.
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
    assert_eq!(
        spectra.len(),
        n,
        "fixture yields {n} labeled in-range molecules"
    );
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

/// Tiny model config (`d = 16`, matching the decoder-test smoke setup).
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

fn train_config(control: Control) -> TrainConfig {
    TrainConfig {
        batch: 4,
        slots: 16,
        lr: 3e-3,
        weight_decay: 0.1,
        formula_weight: 0.2,
        seed: 41,
        control,
        grad_clip: None,
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
    }
}

fn tiny_generation(k: u32, control: Control) -> GenerationConfig {
    GenerationConfig {
        trajectories: k,
        formulas: 2,
        seed: 7,
        max_steps: Limits::V0.max_steps() as u32,
        control,
        ..GenerationConfig::default()
    }
}

#[test]
fn overfit_lowes_reported_loss() {
    // 60 trainer steps on 4 fixture spectra lower the reported loss below 50%
    // of its first report.
    let device = dev();
    let (set, parents) = labeled_set(4);
    let table = FormulaTable::from_compositions(parents.into_iter()).unwrap();
    let mut trainer = Ms2Trainer::<R, E>::new(
        &tiny_config(),
        &table,
        &train_config(Control::None),
        &device,
    )
    .unwrap();
    let indices = vec![0, 1, 2, 3];
    trainer.request_report();
    let first = trainer.step(&set, &indices).unwrap().expect("report");
    assert_eq!(first.spectra, 4);
    assert_eq!(first.formula_present, 4, "every gold joins: {first:?}");
    assert_eq!(first.formula_absent, 0);
    for _ in 1..59 {
        assert!(trainer.step(&set, &indices).unwrap().is_none());
    }
    trainer.request_report();
    let last = trainer.step(&set, &indices).unwrap().expect("report");
    println!("overfit: first {first:?}, last {last:?}");
    assert!(
        last.loss < 0.5 * first.loss,
        "loss {first:?} -> {last:?} falls below 50%"
    );
}

/// `TargetBatch::compact` keeps every occupied slot, with its spectrum, and
/// nothing else.
#[test]
fn compact_targets_keep_every_occupied_slot() {
    use mamba3::models::ms2::experiment::target_batch_for;
    let (set, _) = labeled_set(4);
    let indices = vec![0, 1, 2, 3];
    let padded = target_batch_for(&set, &indices, 16, Limits::V0).unwrap();
    let (compact, owner) = padded.compact(4, 8);
    assert_eq!(compact.slots, 4);
    assert_eq!(compact.spectra % 8, 0, "virtual spectra come in buckets");
    assert_eq!(owner.len(), compact.spectra);
    assert_eq!(compact.q.len(), compact.spectra * 4);
    let t4 = padded.max_steps * 4;
    // Every occupied padded slot appears once, under its own spectrum, in
    // slot order, with its row intact.
    let mut seen = vec![0usize; padded.spectra];
    for row in 0..compact.spectra * 4 {
        if compact.meta[row * 12] == 0 {
            assert_eq!(compact.q[row], 0.0, "an empty slot has no weight");
            continue;
        }
        let b = owner[row / 4] as usize;
        let from = b * 16 + seen[b];
        seen[b] += 1;
        assert_eq!(compact.q[row], padded.q[from]);
        assert_eq!(compact.meta[row * 12..(row + 1) * 12], padded.meta[from * 12..(from + 1) * 12]);
        assert_eq!(compact.tokens[row * t4..(row + 1) * t4], padded.tokens[from * t4..(from + 1) * t4]);
        assert_eq!(
            compact.use_mask[row * t4..(row + 1) * t4],
            padded.use_mask[from * t4..(from + 1) * t4]
        );
    }
    for b in 0..padded.spectra {
        let occupied = (0..16).filter(|g| padded.meta[(b * 16 + g) * 12] != 0).count();
        assert_eq!(seen[b], occupied, "spectrum {b} keeps its occupied slots");
    }
    let total: f32 = compact.q.iter().sum();
    let want: f32 = padded.q.iter().sum();
    assert!((total - want).abs() < 1e-6);
}

/// `TargetBatch::pack` lays every occupied trace, whole and once, into both
/// of its layouts: scan rows of any spectrum, attention rows of one spectrum
/// each, with maps between the two that invert each other.
#[test]
fn packed_targets_hold_every_trace_once() {
    use mamba3::models::ms2::experiment::target_batch_for;
    let (set, _) = labeled_set(4);
    let indices = vec![0, 1, 2, 3];
    let padded = target_batch_for(&set, &indices, 16, Limits::V0).unwrap();
    let t = padded.max_steps;
    let packed = padded.pack(2 * t, 4, 8, 32);
    let (scan, attn) = (&packed.scan, &packed.attn);
    assert_eq!((scan.row_len, attn.row_len), (2 * t, 2 * t));
    assert_eq!(scan.rows % 4, 0);
    assert_eq!(attn.rows % 8, 0);
    assert_eq!(packed.traces.spectra % 32, 0);
    assert_eq!(packed.traces.slots, 1);
    // The traces are the occupied padded slots, in slot order.
    let occupied: Vec<usize> = (0..padded.spectra * 16)
        .filter(|&row| padded.meta[row * 12] != 0)
        .collect();
    let live = (0..packed.traces.spectra)
        .filter(|&r| packed.traces.meta[r * 12] != 0)
        .count();
    assert_eq!(live, occupied.len());
    let mut cells_used = 0usize;
    for (trace, &from) in occupied.iter().enumerate() {
        let n = padded.meta[from * 12] as usize;
        let spectrum = (from / 16) as u32;
        assert_eq!(packed.traces.q[trace], padded.q[from]);
        assert_eq!(
            packed.traces.tokens[trace * t * 4..(trace + 1) * t * 4],
            padded.tokens[from * t * 4..(from + 1) * t * 4]
        );
        let first = scan.unpack[trace * t] as usize;
        assert_eq!(
            scan.reset[first],
            if first % scan.row_len == 0 { 0.0 } else { 1.0 },
            "a reset where a trace begins after another"
        );
        for i in 0..t {
            let cell = scan.unpack[trace * t + i];
            if i >= n {
                assert_eq!(cell, u32::MAX, "nothing past the trace's end");
                continue;
            }
            let cell = cell as usize;
            assert_eq!(cell, first + i, "a trace is contiguous in the scan layout");
            assert_eq!(cell / scan.row_len, first / scan.row_len, "and stays in one row");
            assert_eq!(scan.pack[cell] as usize, trace * t + i, "the maps invert");
            assert_eq!(packed.steps[cell] as usize, i, "the step is the trace's own");
            assert_eq!(packed.cell_owner[cell], spectrum, "the cell knows its spectrum");
            assert_eq!(
                packed.tokens[cell * 4..cell * 4 + 4],
                padded.tokens[(from * t + i) * 4..(from * t + i) * 4 + 4]
            );
            if i > 0 {
                assert_eq!(scan.reset[cell], 0.0);
            }
            // The same position in the attention layout: a row of its own
            // spectrum, and the two cells name each other.
            let attn_cell = packed.to_scan[cell] as usize;
            assert_eq!(attn_cell, attn.unpack[trace * t + i] as usize);
            assert_eq!(packed.to_attn[attn_cell] as usize, cell);
            assert_eq!(attn.row_group[attn_cell / attn.row_len], spectrum);
            cells_used += 1;
        }
    }
    for layout in [&scan.pack, &attn.pack, &packed.to_scan, &packed.to_attn] {
        let mapped = layout.iter().filter(|&&c| c != u32::MAX).count();
        assert_eq!(mapped, cells_used, "no cell belongs to two traces");
    }
    assert!(
        scan.cells() <= attn.cells() && attn.cells() < padded.spectra * 16 * t,
        "the packed rows hold fewer positions than the padded batch"
    );
}

/// A training step reports the same losses whichever layout its teacher
/// pass takes — padded, occupied slots, ragged — before and after updates.
#[test]
fn teacher_pass_layouts_agree() {
    use mamba3::models::ms2::experiment::target_batch_for;
    use mamba3::models::ms2::train::{TeacherPass, set_teacher_pass};
    let device = dev();
    let (set, parents) = labeled_set(4);
    let table = FormulaTable::from_compositions(parents.into_iter()).unwrap();
    let indices = vec![0, 1, 2, 3];
    // The fixture must exercise the compact paths: fewer rows than padded,
    // and at least one packed row holding more than one trace.
    let padded = target_batch_for(&set, &indices, 16, Limits::V0).unwrap();
    let (compact, _) = padded.compact(4, 8);
    assert!(compact.spectra * compact.slots < padded.spectra * padded.slots);
    let packed = padded.pack(2 * padded.max_steps, 4, 8, 32);
    assert!(
        packed.scan.reset.iter().any(|&r| r == 1.0),
        "the fixture packs two traces into one row"
    );
    let two_spectra_in_a_row = (0..packed.scan.rows).any(|row| {
        let cells = &packed.cell_owner[row * packed.scan.row_len..(row + 1) * packed.scan.row_len];
        let mut owners: Vec<u32> = cells.iter().copied().filter(|&o| o != u32::MAX).collect();
        owners.dedup();
        owners.len() > 1
    });
    assert!(two_spectra_in_a_row, "the fixture mixes spectra in a scan row");
    let mut reports = Vec::new();
    for pass in [TeacherPass::Padded, TeacherPass::Slots, TeacherPass::Ragged] {
        set_teacher_pass(pass);
        let mut trainer = Ms2Trainer::<R, E>::new(
            &tiny_config(),
            &table,
            &train_config(Control::None),
            &device,
        )
        .unwrap();
        let mut run = Vec::new();
        for _ in 0..4 {
            trainer.request_report();
            run.push(trainer.step(&set, &indices).unwrap().expect("report"));
        }
        reports.push((pass, run));
    }
    set_teacher_pass(TeacherPass::Ragged);
    let (_, reference) = &reports[0];
    for (pass, run) in &reports[1..] {
        for (step, (p, c)) in reference.iter().zip(run).enumerate() {
            // The first step compares one forward pass; later ones have gone
            // through optimizer updates, which amplify rounding.
            let tol = if step == 0 { 1e-5 } else { 2e-3 };
            for (what, a, b) in [
                ("loss", p.loss, c.loss),
                ("graph", p.graph, c.graph),
                ("formula", p.formula, c.formula),
            ] {
                assert!(
                    (a - b).abs() <= tol * (1.0 + a.abs()),
                    "step {step} {what}: padded {a} vs {pass:?} {b}"
                );
            }
            assert_eq!(p.spectra, c.spectra);
            assert_eq!(p.formula_present, c.formula_present);
        }
    }
}

#[test]
fn step_read_budget() {
    // A warmed training step without a report performs no device read; with a
    // report exactly one (by both the step-sync counter and the total
    // runtime counter). Minima over repetitions: a concurrent test thread can
    // only add reads, never remove this thread's.
    let device = dev();
    let (set, parents) = labeled_set(4);
    let table = FormulaTable::from_compositions(parents.into_iter()).unwrap();
    let mut trainer = Ms2Trainer::<R, E>::new(
        &tiny_config(),
        &table,
        &train_config(Control::None),
        &device,
    )
    .unwrap();
    let indices = vec![0, 1, 2, 3];
    for _ in 0..3 {
        trainer.step(&set, &indices).unwrap();
    }
    let mut min_plain = (usize::MAX, usize::MAX);
    for _ in 0..5 {
        let r0 = runtime_read_count();
        let c0 = read_count();
        assert!(trainer.step(&set, &indices).unwrap().is_none());
        min_plain.0 = min_plain.0.min(runtime_read_count() - r0);
        min_plain.1 = min_plain.1.min(read_count() - c0);
    }
    assert_eq!(min_plain, (0, 0), "a plain step reads nothing");
    let mut min_report = (usize::MAX, usize::MAX);
    for _ in 0..5 {
        trainer.request_report();
        let r0 = runtime_read_count();
        let c0 = read_count();
        let report = trainer.step(&set, &indices).unwrap().expect("report");
        assert!(report.loss.is_finite());
        min_report.0 = min_report.0.min(runtime_read_count() - r0);
        min_report.1 = min_report.1.min(read_count() - c0);
    }
    assert_eq!(min_report, (1, 1), "a report step reads exactly once");
}

#[test]
fn evals_run_under_every_control() {
    // `teacher_eval` and `generate_eval` run under every control and return
    // finite values / valid evaluations with formula recall set.
    let device = dev();
    let (set, parents) = labeled_set(2);
    let table = FormulaTable::from_compositions(parents.into_iter()).unwrap();
    let indices = vec![0, 1];
    for control in [
        Control::None,
        Control::ShuffledSpectrum,
        Control::MetadataOnly,
        Control::StructurePrior,
    ] {
        let mut trainer =
            Ms2Trainer::<R, E>::new(&tiny_config(), &table, &train_config(control), &device)
                .unwrap();
        let eval = trainer.teacher_eval(&set, &indices).unwrap();
        assert_eq!(eval.spectra, 2);
        assert_eq!(eval.slots, 16);
        assert_eq!(eval.nll.len(), 2 * 16);
        assert!(eval.nll.iter().all(|v| v.is_finite()), "{control:?} nll");
        assert_eq!(eval.gold_log_prob.len(), 2);
        assert!(
            eval.gold_slot.iter().all(|&s| s != u32::MAX),
            "{control:?} gold scored"
        );
        let gen_config = tiny_generation(2, control);
        let evals = trainer.generate_eval(&set, &indices, &gen_config).unwrap();
        assert_eq!(evals.len(), 2, "{control:?} spectra");
        for e in &evals {
            assert_eq!(e.candidates.len(), 2, "{control:?} trajectories");
            assert!(e.q_found.is_finite(), "{control:?} q_found");
            assert!(e.formula_recall.is_some(), "{control:?} recall set");
        }
    }
}

#[test]
fn donor_peaks_keep_each_models_own_blinding() {
    // The `--diagnose` peak sensitivity compares one model on two inputs.
    // A model that never sees peaks must give bit-identical teacher NLL with
    // donor peaks; an earlier version encoded every donor batch with
    // `Control::None`, which switched the blinded models' peak path on.
    let device = dev();
    let (set, parents) = labeled_set(2);
    let table = FormulaTable::from_compositions(parents.into_iter()).unwrap();
    let indices = vec![0, 1];
    for control in [
        Control::MetadataOnly,
        Control::StructurePrior,
        Control::None,
    ] {
        let mut trainer =
            Ms2Trainer::<R, E>::new(&tiny_config(), &table, &train_config(control), &device)
                .unwrap();
        let own = trainer.teacher_field_eval(&set, &indices, false).unwrap();
        let donor = trainer.teacher_field_eval(&set, &indices, true).unwrap();
        assert_eq!(donor.donor_same_molecule, 0, "{control:?} donors");
        if control == Control::None {
            // Positive control: a model that reads peaks must see the swap.
            assert_ne!(own.nll, donor.nll, "{control:?} sees donor peaks");
        } else {
            assert_eq!(own.nll, donor.nll, "{control:?} ignores peaks");
            assert_eq!(
                own.field_log_prob, donor.field_log_prob,
                "{control:?} fields"
            );
        }
    }
}

#[test]
fn save_load_round_trip() {
    // `save` then `load` gives bit-identical teacher NLL.
    let device = dev();
    let (set, parents) = labeled_set(2);
    let table = FormulaTable::from_compositions(parents.into_iter()).unwrap();
    let indices = vec![0, 1];
    let mut trainer = Ms2Trainer::<R, E>::new(
        &tiny_config(),
        &table,
        &train_config(Control::None),
        &device,
    )
    .unwrap();
    for _ in 0..2 {
        trainer.step(&set, &indices).unwrap();
    }
    let path = std::env::temp_dir().join("ms2_v0d2_trainer_roundtrip.json");
    trainer.save(&path).unwrap();
    let before = trainer.teacher_eval(&set, &indices).unwrap();
    let mut loaded = Ms2Trainer::<R, E>::load(&path, &table, &device).unwrap();
    let after = loaded.teacher_eval(&set, &indices).unwrap();
    assert_eq!(before.nll.len(), after.nll.len());
    for (i, (a, b)) in before.nll.iter().zip(after.nll.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "nll[{i}] differs after save/load: {a} vs {b}"
        );
    }
    assert_eq!(before.gold_log_prob, after.gold_log_prob);
    assert_eq!(loaded.step_count(), trainer.step_count());
    std::fs::remove_file(&path).ok();
}

#[test]
fn structure_prior_invariance() {
    // Under `StructurePrior`, changing a spectrum's peaks, collision energy,
    // instrument, precursor (within the formula window) or other
    // adduct-independent metadata leaves the encoder memory and pool
    // bit-identical; under `MetadataOnly` changing the energy changes them.
    let device = dev();
    let (set, _) = labeled_set(2);
    let config = tiny_config();
    let mut rng = Rng::seeded(5);
    let encoder = Ms2Encoder::<R, E>::init(&config, &device, &mut rng).unwrap();
    let base = spectrum_batch_for(&set, &[0, 1], 64).unwrap();
    let run = |batch: &SpectrumBatch, control: Control| {
        let spectra = DeviceSpectra::<R, E>::upload(batch, &device).unwrap();
        let peaks = ms2::PeakBuffers::<R, E>::new(batch.len(), 64, 16, &device);
        let out = encoder.encode(&spectra, &peaks, control).unwrap();
        (
            out.memory.try_to_f32().unwrap(),
            out.pool.try_to_f32().unwrap(),
        )
    };
    let (mem_base, pool_base) = run(&base, Control::StructurePrior);
    let mut variant = base.clone();
    // Peaks: scale the valid intensities and shift the valid m/z of spectrum 0.
    for i in 0..base.peak_count[0] as usize {
        variant.intensity[i] = base.intensity[i] * 1.7 + 0.01;
        variant.mz_udalton[i] = base.mz_udalton[i] + 500;
    }
    // Energy, instrument, precursor (a 1 mDa shift stays in the window) and
    // other metadata of spectrum 0.
    variant.collision_energy_ev[0] = 200.0;
    variant.energy_count[0] = 3;
    variant.instrument_class[0] = 1;
    variant.precursor_mz_udalton[0] += 1000;
    variant.fragment_tolerance_ppm_tenths[0] = 500;
    let (mem_var, pool_var) = run(&variant, Control::StructurePrior);
    assert_eq!(mem_base.len(), mem_var.len());
    for (i, (a, b)) in mem_base.iter().zip(&mem_var).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "memory[{i}] differs under StructurePrior: {a} vs {b}"
        );
    }
    for (i, (a, b)) in pool_base.iter().zip(&pool_var).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "pool[{i}] differs under StructurePrior: {a} vs {b}"
        );
    }
    // Positive control: under `MetadataOnly` the energy change moves the
    // memory and pool.
    let (_, pool_meta_base) = run(&base, Control::MetadataOnly);
    let (_, pool_meta_var) = run(&variant, Control::MetadataOnly);
    let moved = pool_meta_base
        .iter()
        .zip(&pool_meta_var)
        .any(|(a, b)| a.to_bits() != b.to_bits());
    assert!(moved, "MetadataOnly pool ignores the energy change");
}

/// The chemistry version the fixture was built with matches the crate's, so
/// the in-memory set above is genuinely in-domain.
#[test]
fn fixture_chemistry_matches() {
    let f = fixture();
    assert_eq!(
        f["chemistry"].as_str().expect("chemistry"),
        CHEMISTRY_VERSION
    );
}

#[test]
fn donor_map_never_same_molecule_and_deterministic() {
    // Every donor is a different molecule; the same seed gives the same map.
    let (set, _) = labeled_set(4);
    let a = set.donor_map(123).unwrap();
    let b = set.donor_map(123).unwrap();
    assert_eq!(a, b, "donor_map is deterministic");
    assert_eq!(a.len(), set.spectra.len());
    for (i, &d) in a.iter().enumerate() {
        assert_ne!(
            set.spectra[i].molecule, set.spectra[d].molecule,
            "spectrum {i} donor {d} is the same molecule"
        );
    }
    // A single-molecule set has no donor.
    let (one, _) = labeled_set(1);
    let single = ExperimentSet {
        name: "single".to_string(),
        source_sha256: "x".to_string(),
        molecules: vec![one.molecules[0].clone()],
        spectra: vec![{
            let s = &one.spectra[0];
            ExperimentSpectrum {
                molecule: 0,
                spectrum: s.spectrum.clone(),
                parent: MolGraph::new(s.parent.atoms().to_vec(), s.parent.bonds().to_vec())
                    .unwrap(),
                parent_composition: s.parent_composition,
                labels: s.labels.clone(),
                domain: s.domain.clone(),
            }
        }],
    };
    assert!(
        single.donor_map(1).is_err(),
        "one molecule has no different-molecule donor"
    );
}

#[test]
fn donor_batch_carries_donor_peaks_and_recipient_metadata() {
    // Row b carries the donor's peak fields and the recipient's identity,
    // metadata, precursor and targets.
    use mamba3::models::ms2::experiment::{
        donor_stats, spectrum_batch_with_donors, target_batch_for,
    };
    use mamba3::models::ms2::grammar::Limits;
    let (set, _) = labeled_set(4);
    let map = set.donor_map(7).unwrap();
    let indices = vec![0, 1, 2, 3];
    let donors: Vec<usize> = indices.iter().map(|&i| map[i]).collect();
    let batch = spectrum_batch_with_donors(&set, &indices, &donors, 64).unwrap();
    let n_raw = 64usize;
    for (b, (&ri, &di)) in indices.iter().zip(donors.iter()).enumerate() {
        let r = &set.spectra[ri].spectrum;
        let d = &set.spectra[di].spectrum;
        // Identity, precursor and metadata stay the recipient's.
        assert_eq!(batch.spectrum_id[b], r.spectrum_id);
        assert_eq!(batch.precursor_mz_udalton[b], r.precursor_mz_udalton);
        assert_eq!(batch.adduct[b], r.adduct);
        assert_eq!(batch.polarity[b], r.polarity);
        // Peak fields come from the donor.
        assert_eq!(batch.peak_count[b], d.peak_id.len() as u32);
        assert_eq!(batch.raw_peak_count[b], d.raw_peak_count);
        assert_eq!(batch.mz_uncertainty_udalton[b], d.mz_uncertainty_udalton);
        for k in 0..d.peak_id.len() {
            assert_eq!(batch.peak_id[b * n_raw + k], d.peak_id[k]);
            assert_eq!(batch.mz_udalton[b * n_raw + k], d.mz_udalton[k]);
            assert_eq!(
                batch.intensity[b * n_raw + k],
                d.intensity[k] as f32,
                "row {b} peak {k} intensity"
            );
        }
    }
    // Targets stay the recipient's: the recipient target batch builds with
    // the recipients' labels (all labeled here) alongside the donor batch.
    let got = target_batch_for(&set, &indices, 16, Limits::V0).unwrap();
    assert_eq!(got.labeled, vec![1; 4], "recipients are labeled");
    assert_eq!(got.spectra, batch.len(), "targets match the donor batch");
    // Donor diagnostics: same-molecule is 0 by construction.
    let (same, _) = donor_stats(&set, &indices, &donors).unwrap();
    assert_eq!(same, 0, "donor_map never reuses the molecule");
}

#[test]
fn shuffled_single_final_chunk_uses_donor_path() {
    // D5: a one-spectrum final chunk under ShuffledSpectrum works through the
    // trainer molecule-aware donor path (generate_eval), unlike the old extra
    // probe that passed the original batch to model.generate directly (which
    // refuses single-row ShuffledSpectrum). Exhaustion comes from these same
    // request statuses, not from an extra probe.
    let device = dev();
    let (set, parents) = labeled_set(5);
    let table = FormulaTable::from_compositions(parents.into_iter()).unwrap();
    let mut trainer = Ms2Trainer::<R, E>::new(
        &tiny_config(),
        &table,
        &train_config(Control::ShuffledSpectrum),
        &device,
    )
    .unwrap();
    let indices: Vec<usize> = (0..5).collect();
    let gen_config = tiny_generation(2, Control::ShuffledSpectrum);
    // Chunks of 2 leave a one-spectrum final chunk (2,2,1); each must work.
    let mut all_statuses = Vec::new();
    for chunk in indices.chunks(2) {
        let (evals, _work, statuses) = trainer
            .generate_eval_with_work(&set, chunk, &gen_config)
            .unwrap();
        assert_eq!(evals.len(), chunk.len());
        assert_eq!(statuses.len(), chunk.len());
        all_statuses.extend(statuses);
    }
    assert_eq!(all_statuses.len(), 5);
    // Exhaustion statistics come from these statuses (here: table source, so
    // none exhausted, but the path works for single-row chunks).
    let exhausted = all_statuses
        .iter()
        .filter(|&&rs| rs & mamba3::models::ms2::contract::request_status::FORMULA_SEARCH_EXHAUSTED != 0)
        .count();
    assert_eq!(exhausted, 0);
}

#[test]
fn enum_fit_refusals_are_config_errors() {
    // D6: a fitting file whose subset is not train/fit, or that shares any
    // molecule key with validation, is Error::Config.
    use mamba3::models::ms2::experiment::check_enum_fit;
    // Wrong subset.
    let err = check_enum_fit("validation", "fit.json", &["molA".to_string()], &[]).unwrap_err();
    assert!(matches!(err, mamba3::error::Error::Config(_)), "{err}");
    assert!(err.to_string().contains("subset"), "{err}");
    // Shared molecule key.
    let err = check_enum_fit(
        "train",
        "fit.json",
        &["molA".to_string(), "molB".to_string()],
        &["molB".to_string(), "molC".to_string()],
    )
    .unwrap_err();
    assert!(matches!(err, mamba3::error::Error::Config(_)), "{err}");
    assert!(err.to_string().contains("shares"), "{err}");
    // Train/fit subsets with disjoint keys pass.
    assert!(check_enum_fit("train", "fit.json", &["molA".to_string()], &["molB".to_string()]).is_ok());
    assert!(check_enum_fit("fit", "fit.json", &["molA".to_string()], &[]).is_ok());
}

#[test]
fn assignment_label_overflow_matches_host_count() {
    // `LossReport::assignment_label_overflow` equals the host count on a
    // fixture with more than `L` labels: `L = 1` with labeled fixture
    // spectra, counted with the same upload mapping as the trainer.
    let device = dev();
    let (set, parents) = labeled_set(4);
    let table = FormulaTable::from_compositions(parents.into_iter()).unwrap();
    let mut model = tiny_config();
    model.assignment = Some(AssignmentConfig {
        hypotheses: 4,
        work_max: 4096,
        labels: 1,
    });
    let mut cfg = train_config(Control::None);
    cfg.lambda_assign = 0.1;
    let mut trainer =
        Ms2Trainer::<R, E>::new(&model, &table, &cfg, &device).unwrap();
    let indices = vec![0, 1, 2, 3];
    trainer.request_report();
    let rep = trainer.step(&set, &indices).unwrap().expect("report");
    // Independent host count with the trainer's upload mapping
    // (`Control::None`: own peaks via `spectrum_batch_for`).
    let mut longest = 0usize;
    for &i in &indices {
        longest = longest.max(set.spectra[i].spectrum.peak_id.len());
    }
    let n_raw = [64usize, 128, 256, 512]
        .into_iter()
        .find(|&b| longest <= b)
        .expect("fixture fits");
    let batch = spectrum_batch_for(&set, &indices, n_raw as u32).unwrap();
    let mut want = 0usize;
    for (bi, &idx) in indices.iter().enumerate() {
        let Some(lbls) = set.spectra[idx].labels.as_ref() else {
            continue;
        };
        let adduct_id = batch.adduct[bi];
        let base = bi * n_raw;
        let count = (batch.peak_count[bi] as usize).min(n_raw);
        let raw_of = |pid: u32| -> Option<u32> {
            for k in 0..count {
                if batch.peak_id[base + k] == pid {
                    return Some(k as u32);
                }
            }
            None
        };
        want += ion_labels(lbls, adduct_id, raw_of, 1).overflow;
    }
    assert!(want > 0, "the fixture overflows L = 1");
    assert_eq!(
        rep.assignment_label_overflow, want,
        "LossReport overflow equals the host count"
    );
    println!("assignment_label_overflow {want} on 4 spectra at L = 1");
}
