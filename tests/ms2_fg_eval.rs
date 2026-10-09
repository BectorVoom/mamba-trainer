//! FG1 device smoke test: a freshly initialised model's candidates go through
//! the functional-group evaluation without error, with graphs rebuilt under
//! their own conditioning formulas.

#![cfg(feature = "backend")]

use mamba3::backend::Device;
use mamba3::backends::Auto;
use mamba3::models::ms2::chem::{
    Composition, ELECTRON_MASS, ELEMENTS, HYDROGEN, adduct, composition_mass,
};
use mamba3::models::ms2::contract::{Control, GenerationConfig, ModelConfig, candidate_status};
use mamba3::models::ms2::experiment::{
    ExperimentSet, ExperimentSpectrum, SpectrumDomain, label_export_spectrum,
};
use mamba3::models::ms2::formula::FormulaTable;
use mamba3::models::ms2::functional_groups::functional_groups_v4;
use mamba3::models::ms2::functional_groups_eval::{
    eval_records, evaluate_fg, label_union, recipe_fragments_union, spectrum_datum,
};
use mamba3::models::ms2::grammar::{Limits, replay};
use mamba3::models::ms2::targets::{Candidates, Peak, RecipeLimits};
use mamba3::models::ms2::train::{GoldFormulaConditioning, Ms2Trainer, TrainConfig};
use mamba3::models::ms2::{MolGraph, RawAtom, RawMolecule};

type R = Auto;
type E = f32;

static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

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
    let a = adduct(adduct_id)?;
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
    let export = mamba3::models::ms2::dataset::ExportSpectrum {
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
        let Ok(graph) = raw.to_graph() else { continue };
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

#[test]
fn fg_eval_smoke_on_fresh_model() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let device = Device::<R>::default();
    let (set, parents) = labeled_set(3);
    let table = FormulaTable::from_compositions(parents.into_iter()).unwrap();
    let train_config = TrainConfig {
        batch: 4,
        slots: 16,
        lr: 3e-3,
        weight_decay: 0.1,
        formula_weight: 0.2,
        seed: 41,
        control: Control::None,
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
        formula_evidence_work_max: 2048,
        formula_evidence_dispatch_max: 268435456,
        precursor_jitter_ppm: 0.0,
        precursor_jitter_variants: 0,
        nonfinite_guard: false,
        loss_scale: 1.0,
    };
    let trainer = Ms2Trainer::<R, E>::new(&tiny_config(), &table, &train_config, &device).unwrap();
    let indices = vec![0, 1, 2];
    let gen_config = GenerationConfig {
        trajectories: 4,
        formulas: 2,
        seed: 7,
        max_steps: Limits::V0.max_steps() as u32,
        ..GenerationConfig::default()
    };
    let mut batch = trainer
        .generate_candidates(&set, &indices, &gen_config)
        .unwrap();
    let rows = eval_records(&batch).unwrap();
    assert_eq!(rows.len(), 3);
    let limits = Limits::new(batch.max_atoms, batch.max_ring_closures).unwrap();
    let mut data = Vec::new();
    for (pos, &idx) in indices.iter().enumerate() {
        let entry = &set.spectra[idx];
        let lu = entry.labels.as_ref().map(label_union).unwrap_or(0);
        let (ou, _) = recipe_fragments_union(&entry.parent);
        data.push(spectrum_datum(
            &entry.parent,
            entry.molecule,
            &rows[pos],
            limits,
            lu,
            ou,
        ));
    }
    // The whole pipeline runs without error on fresh-model candidates.
    let reps = evaluate_fg(&data, None, &[1, 2], 10, 0);
    assert_eq!(reps.len(), 2);

    // Own-formula rebuild: recondition a replayable candidate on a formula
    // different from the parent's; it is still evaluated, under its own
    // budget rather than the parent composition.
    let mut fixed = false;
    for (pos, &idx) in indices.iter().enumerate() {
        if fixed {
            break;
        }
        let k = batch.trajectories;
        for kk in 0..k {
            let r = pos * k + kk;
            if batch.status[r] & candidate_status::FINISHED == 0 {
                continue;
            }
            let len = batch.length[r] as usize;
            let base = r * batch.max_steps * 4;
            let mut tokens = Vec::with_capacity(len);
            for step in 0..len {
                let f = &batch.actions[base + step * 4..base + step * 4 + 4];
                tokens.push(mamba3::models::ms2::grammar::Token {
                    kind: f[0] as u8,
                    atom_type: f[1] as u8,
                    bond: f[2] as u8,
                    pointer: f[3] as u8,
                });
            }
            let Ok(state) = replay(&tokens, limits, None) else {
                continue;
            };
            if !state.stopped() || state.atoms() == 0 {
                continue;
            }
            let graph = state.graph().unwrap();
            let mut comp = graph.composition();
            // A formula guaranteed different from the parent's that still
            // budgets the trace: the graph's own composition plus margin.
            for e in 0..10 {
                comp[e] = comp[e].saturating_add(1000);
            }
            assert_ne!(comp, set.spectra[idx].parent_composition);
            for e in 0..10 {
                batch.formula_counts[r * 10 + e] = comp[e];
            }
            let rows = eval_records(&batch).unwrap();
            assert!(rows[pos][kk].eligible(), "record stays eligible");
            let entry = &set.spectra[idx];
            let datum = spectrum_datum(&entry.parent, entry.molecule, &rows[pos], limits, 0, 0);
            let found = datum
                .candidates
                .iter()
                .find(|c| c.mask() == functional_groups_v4(&graph).mask());
            assert!(
                found.is_some(),
                "the reconditioned candidate is evaluated under its own formula"
            );
            fixed = true;
            break;
        }
    }
    assert!(
        fixed,
        "fresh-model batch holds a replayable finished candidate"
    );
}
