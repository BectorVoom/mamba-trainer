//! Committed fixture generator: trains the tiny completion model on nine
//! hand-built molecules and writes the checkpoint plus the shared
//! request/response fixtures.
//!
//! Usage:
//! ```text
//! cargo run --release --no-default-features --features cpu \
//!   --example molecular_completion_fixture
//! ```
//!
//! Trains
//! [`CompletionModelConfig::tiny`](mamba3::models::ms2::completion_model::CompletionModelConfig::tiny)
//! on ethanol, dimethyl ether, propan-1-ol, propan-2-ol, methoxyethane,
//! ethylamine, dimethylamine, cyclopropane and kekulized methylbenzene with
//! fixed seeds for a fixed number of steps on the CPU backend, saves
//! `tests/fixtures/ms2/completion_tiny.ckpt` and writes
//! `tests/fixtures/ms2/completion_tiny_requests.json` (eight requests) and
//! `tests/fixtures/ms2/completion_tiny_responses.json` (the `generate_json`
//! responses). The fixture is defined on CPU: the example refuses to run on a
//! non-CPU backend. Running it twice produces byte-identical files.

use std::collections::BTreeMap;
use std::path::PathBuf;

use mamba3::backend::Device;
use mamba3::backends::Auto;
use mamba3::models::ms2::completion_api::CompletionService;
use mamba3::models::ms2::completion_data::{CompletionExample, CompletionSet, ExtractionConfig, PatternSource};
use mamba3::models::ms2::completion_model::{CompletionModelConfig, CompletionTrainConfig};
use mamba3::models::ms2::grammar::{CANONICAL_WORK_LIMIT, Limits, canonical_trace, replay_exact};
use mamba3::models::ms2::graph::MolGraph;

type R = Auto;
type E = f32;

/// Fixed optimizer steps of the fixture.
const STEPS: usize = 200;

/// Ethanol `[C(H3), C(H2), O(H1)]`.
fn ethanol() -> MolGraph {
    MolGraph::new(vec![4, 3, 9], vec![(0, 1, 1), (1, 2, 1)]).unwrap()
}

/// Dimethyl ether `[C(H3), O(H0), C(H3)]`.
fn dimethyl_ether() -> MolGraph {
    MolGraph::new(vec![4, 8, 4], vec![(0, 1, 1), (1, 2, 1)]).unwrap()
}

/// Propan-1-ol.
fn propan_1_ol() -> MolGraph {
    MolGraph::new(vec![4, 3, 3, 9], vec![(0, 1, 1), (1, 2, 1), (2, 3, 1)]).unwrap()
}

/// Propan-2-ol.
fn propan_2_ol() -> MolGraph {
    MolGraph::new(vec![4, 2, 4, 9], vec![(0, 1, 1), (1, 2, 1), (1, 3, 1)]).unwrap()
}

/// Methoxyethane.
fn methoxyethane() -> MolGraph {
    MolGraph::new(vec![4, 8, 3, 4], vec![(0, 1, 1), (1, 2, 1), (2, 3, 1)]).unwrap()
}

/// Ethylamine.
fn ethylamine() -> MolGraph {
    MolGraph::new(vec![4, 3, 7], vec![(0, 1, 1), (1, 2, 1)]).unwrap()
}

/// Dimethylamine.
fn dimethylamine() -> MolGraph {
    MolGraph::new(vec![4, 6, 4], vec![(0, 1, 1), (1, 2, 1)]).unwrap()
}

/// Cyclopropane.
fn cyclopropane() -> MolGraph {
    MolGraph::new(vec![3, 3, 3], vec![(0, 1, 1), (1, 2, 1), (2, 0, 1)]).unwrap()
}

/// Kekulized methylbenzene.
fn methylbenzene() -> MolGraph {
    MolGraph::new(
        vec![1, 2, 2, 2, 2, 2, 4],
        vec![
            (0, 1, 1),
            (1, 2, 2),
            (2, 3, 1),
            (3, 4, 2),
            (4, 5, 1),
            (5, 0, 2),
            (0, 6, 1),
        ],
    )
    .unwrap()
}

fn main() -> mamba3::error::Result<()> {
    let device = Device::<R>::default();
    if device.name() != "cpu" {
        eprintln!(
            "molecular_completion_fixture: the fixture is defined on CPU, refusing backend '{}'",
            device.name()
        );
        std::process::exit(2);
    }
    let limits = Limits::new(12, 2).unwrap();
    let molecules: Vec<(&str, MolGraph)> = vec![
        ("ethanol", ethanol()),
        ("dimethyl ether", dimethyl_ether()),
        ("propan-1-ol", propan_1_ol()),
        ("propan-2-ol", propan_2_ol()),
        ("methoxyethane", methoxyethane()),
        ("ethylamine", ethylamine()),
        ("dimethylamine", dimethylamine()),
        ("cyclopropane", cyclopropane()),
        ("methylbenzene", methylbenzene()),
    ];
    let mut examples = Vec::new();
    for (i, (key, graph)) in molecules.into_iter().enumerate() {
        let canonical = canonical_trace(&graph, limits, CANONICAL_WORK_LIMIT).unwrap();
        let composition = graph.composition();
        let end = replay_exact(&canonical.trace, limits, composition).unwrap();
        assert!(end.stopped() && end.is_complete());
        examples.push(CompletionExample {
            key: key.to_string(),
            identity_group: i as u64,
            source_index: i,
            target: graph,
            composition,
            trace: canonical.trace,
            skeleton_trace: Vec::new(),
        });
    }
    let set = CompletionSet {
        limits,
        examples,
        skipped: BTreeMap::new(),
        max_expansions: CANONICAL_WORK_LIMIT,
    };
    let train_config = CompletionTrainConfig {
        lr: 3e-3,
        weight_decay: 0.0,
        grad_clip: None,
        seed: 1,
        extraction: ExtractionConfig::default(),
        extraction_seed: 11,
        pattern_source: PatternSource::RandomPatches(ExtractionConfig::default()),
        fingerprint_mode: None,
        fingerprint_threshold: 0.1,
    };
    let mut trainer = mamba3::models::ms2::completion_model::CompletionTrainer::<R, E>::new(
        &CompletionModelConfig::tiny(),
        &train_config,
        &device,
    )?;
    let indices: Vec<usize> = (0..set.examples.len()).collect();
    for step in 0..STEPS {
        if step % 20 == 0 {
            trainer.request_report();
        }
        if let Some(loss) = trainer.step(&set, &indices, 0)? {
            println!("fixture step {step}: loss {loss}");
        }
    }
    // Formula artifacts bound to the checkpoint, fitted on the 9 training
    // molecules (margin 0, quantile margin 0, heavy_max clamped to 12).
    {
        let compositions: Vec<mamba3::models::ms2::chem::Composition> =
            set.examples.iter().map(|e| e.composition).collect();
        let artifacts = mamba3::models::ms2::completion_model::FormulaArtifacts::fit(
            &compositions,
            12,
            0,
            0,
            "fixture:9-training-molecules".to_string(),
        )?;
        trainer.set_formula_artifacts(artifacts);
    }
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let dir = root.join("tests").join("fixtures").join("ms2");
    std::fs::create_dir_all(&dir)?;
    let ckpt = dir.join("completion_tiny.ckpt");
    trainer.save(&ckpt)?;

    let requests = serde_json::json!([
        {
            "protocol": "molecular-completion-generate-v1",
            "id": "tiny-ethanol-co",
            "provenance": "synthetic fixture",
            "mass_role": "target_molecule",
            "composition": {"C": 2, "H": 6, "O": 1},
            "substructures": [
                {"atoms": [3, 9], "bonds": [[0, 1, 1]],
                 "parent_hydrogen_semantics": "v0_parent_hydrogen_counts",
                 "certainty": "confirmed", "provenance": "synthetic: C(H2)-O(H1)"}
            ],
            "generation": {"trajectories": 64, "temperature": 1.0, "seed": 1, "returned": 25}
        },
        {
            "protocol": "molecular-completion-generate-v1",
            "id": "tiny-c3h8o-coc",
            "provenance": "synthetic fixture",
            "mass_role": "target_molecule",
            "composition": {"C": 3, "H": 8, "O": 1},
            "substructures": [
                {"atoms": [4, 8, 3], "bonds": [[0, 1, 1], [1, 2, 1]],
                 "parent_hydrogen_semantics": "v0_parent_hydrogen_counts",
                 "certainty": "confirmed", "provenance": "synthetic: C-O-C"}
            ],
            "generation": {"trajectories": 64, "temperature": 1.0, "seed": 1, "returned": 25}
        },
        {
            "protocol": "molecular-completion-generate-v1",
            "id": "tiny-c2h7n-empty",
            "provenance": "synthetic fixture",
            "mass_role": "target_molecule",
            "composition": {"C": 2, "H": 7, "N": 1},
            "substructures": [],
            "generation": {"trajectories": 64, "temperature": 1.0, "seed": 1, "returned": 25}
        },
        {
            "protocol": "molecular-completion-generate-v1",
            "id": "tiny-cyclopropane-cc",
            "provenance": "synthetic fixture",
            "mass_role": "target_molecule",
            "composition": {"C": 3, "H": 6},
            "substructures": [
                {"atoms": [3, 3], "bonds": [[0, 1, 1]],
                 "parent_hydrogen_semantics": "v0_parent_hydrogen_counts",
                 "certainty": "confirmed", "provenance": "synthetic: C(H2)-C(H2)"}
            ],
            "generation": {"trajectories": 64, "temperature": 1.0, "seed": 1, "returned": 25}
        },
        {
            "protocol": "molecular-completion-generate-v1",
            "id": "tiny-butanol-c4h10o",
            "provenance": "synthetic fixture",
            "mass_role": "target_molecule",
            "composition": {"C": 4, "H": 10, "O": 1},
            "substructures": [
                {"atoms": [4, 2, 3, 9], "bonds": [[0, 1, 1], [1, 2, 1], [1, 3, 1]],
                 "parent_hydrogen_semantics": "v0_parent_hydrogen_counts",
                 "certainty": "confirmed", "provenance": "synthetic: C(H3)-C(H1)(-O(H1))-C(H2)"}
            ],
            "generation": {"trajectories": 64, "temperature": 1.0, "seed": 1, "returned": 25}
        },
        {
            "protocol": "molecular-completion-generate-v1",
            "id": "tiny-ethanol-mass-neutral",
            "provenance": "synthetic fixture",
            "mass_role": "target_molecule",
            "target_mass": {"units": "microdalton", "value": 46041865, "ppm_tenths": 100, "uncertainty_uda": 50, "source": "synthetic"},
            "neutralization": "already_neutral",
            "formula_search": {"hypotheses": 8, "nodes_visited_max": 2000000},
            "substructures": [
                {"atoms": [3, 9], "bonds": [[0, 1, 1]],
                 "parent_hydrogen_semantics": "v0_parent_hydrogen_counts",
                 "certainty": "confirmed", "provenance": "synthetic: C(H2)-O(H1)"}
            ],
            "generation": {"trajectories": 64, "temperature": 1.0, "seed": 1, "returned": 25}
        },
        {
            "protocol": "molecular-completion-generate-v1",
            "id": "tiny-ethanol-mass-protonated",
            "provenance": "synthetic fixture",
            "mass_role": "target_molecule",
            "target_mass": {"units": "microdalton", "value": 47049141, "ppm_tenths": 100, "uncertainty_uda": 50, "source": "synthetic"},
            "neutralization": {"precursor_ion": {"adduct": "[M+H]+"}},
            "formula_search": {"hypotheses": 8, "nodes_visited_max": 2000000},
            "substructures": [
                {"atoms": [3, 9], "bonds": [[0, 1, 1]],
                 "parent_hydrogen_semantics": "v0_parent_hydrogen_counts",
                 "certainty": "confirmed", "provenance": "synthetic: C(H2)-O(H1)"}
            ],
            "generation": {"trajectories": 64, "temperature": 1.0, "seed": 1, "returned": 25}
        },
        {
            "protocol": "molecular-completion-generate-v1",
            "id": "tiny-ethanol-fg-complete",
            "provenance": "synthetic fixture",
            "mass_role": "target_molecule",
            "composition": {"C": 2, "H": 6, "O": 1},
            "substructures": [
                {"atoms": [9], "bonds": [],
                 "parent_hydrogen_semantics": "v0_parent_hydrogen_counts",
                 "certainty": "confirmed", "provenance": "synthetic: O(H1) hydroxyl oxygen"}
            ],
            "substructure_semantics": "complete_functional_groups",
            "generation": {"trajectories": 64, "temperature": 1.0, "seed": 1, "returned": 25}
        }
    ]);
    let req_path = dir.join("completion_tiny_requests.json");
    std::fs::write(&req_path, serde_json::to_string_pretty(&requests)?)?;
    let service = CompletionService::load(&ckpt, &device)?;
    let mut responses = Vec::new();
    for req in requests.as_array().unwrap() {
        let text = serde_json::to_string(req)?;
        let out = service.generate_json(&text)?;
        let value: serde_json::Value = serde_json::from_str(&out)?;
        responses.push(value);
    }
    let resp_path = dir.join("completion_tiny_responses.json");
    std::fs::write(
        &resp_path,
        serde_json::to_string_pretty(&serde_json::Value::Array(responses))?,
    )?;
    println!(
        "wrote {} {} {}",
        ckpt.display(),
        req_path.display(),
        resp_path.display()
    );
    Ok(())
}
