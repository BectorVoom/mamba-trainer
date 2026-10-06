//! MC4 tests: the molecular-completion experiment driver.
//!
//! Hand-built molecules only (atom type ids from
//! [`chem::ATOM_TYPES`](mamba3::models::ms2::chem::ATOM_TYPES)): ethanol,
//! dimethyl ether, propan-1-ol, propan-2-ol and a 20-atom chain that is
//! out of domain for the small preset. Exports are written to temp dirs as
//! JSON; the driver is exercised through the library entry point
//! [`run`](mamba3::models::ms2::completion_experiment::run), never a
//! subprocess. Reports and predictions contain no keys and no SMILES.
//!
//! Every test holds the file-local serial lock: the binary shares one
//! process-global device.

#![cfg(feature = "backend")]

use std::collections::HashMap;
use std::path::PathBuf;

use mamba3::backend::Device;
use mamba3::backends::Auto;
use mamba3::models::ms2::chem::CHEMISTRY_VERSION;
use mamba3::models::ms2::completion::contains_pattern;
use mamba3::models::ms2::completion::stable_hash;
use mamba3::models::ms2::completion_data::{CompletionSet, ExtractionConfig};
use mamba3::models::ms2::completion_experiment::{
    ExperimentArgs, ExperimentArm, eval_extraction_seed, eval_generation_config, run,
};
use mamba3::models::ms2::completion_formula::{FormulaAllocation, FormulaPruning};
use mamba3::models::ms2::completion_model::{CompletionRequest, CompletionTrainer};
use mamba3::models::ms2::contain::Containment;
use mamba3::models::ms2::dataset::{ExportFile, ExportMolecule};
use mamba3::models::ms2::grammar::{CANONICAL_WORK_LIMIT, Limits};
use mamba3::models::ms2::graph::MolGraph;
use mamba3::tensor::ops::ms2::Ms2Constants;

type R = Auto;

/// File-local serial lock: the tests share one device.
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn serial() -> std::sync::MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

fn dev() -> Device<R> {
    Device::<R>::default()
}

/// Ethanol `[C(H3), C(H2), O(H1)]`.
fn ethanol() -> MolGraph {
    MolGraph::new(vec![4, 3, 9], vec![(0, 1, 1), (1, 2, 1)]).unwrap()
}

/// Dimethyl ether `[C(H3), O(H0), C(H3)]`.
fn dimethyl_ether() -> MolGraph {
    MolGraph::new(vec![4, 8, 4], vec![(0, 1, 1), (1, 2, 1)]).unwrap()
}

/// Propan-1-ol: a `C(H3)-C(H2)-C(H2)-O(H1)` chain.
fn propan_1_ol() -> MolGraph {
    MolGraph::new(vec![4, 3, 3, 9], vec![(0, 1, 1), (1, 2, 1), (2, 3, 1)]).unwrap()
}

/// Propan-2-ol: central `C(H1)` with two methyls and one `O(H1)`.
fn propan_2_ol() -> MolGraph {
    MolGraph::new(vec![4, 2, 4, 9], vec![(0, 1, 1), (1, 2, 1), (1, 3, 1)]).unwrap()
}

/// A 20-atom chain: out of domain for the small preset (16 atoms).
fn long_chain() -> MolGraph {
    let mut atoms = vec![4u8];
    atoms.extend(std::iter::repeat_n(3u8, 18));
    atoms.push(4);
    let bonds: Vec<(usize, usize, u8)> = (0..19).map(|i| (i, i + 1, 1)).collect();
    MolGraph::new(atoms, bonds).unwrap()
}

fn export_molecule(key: &str, graph: &MolGraph, identity_group: u64) -> ExportMolecule {
    ExportMolecule {
        key: key.to_string(),
        identity_group,
        fold_identity: 0,
        atoms: graph.atoms().to_vec(),
        bonds: graph.bonds().to_vec(),
        spectra: Vec::new(),
    }
}

/// Write one export file of hand-built molecules to `dir`.
fn write_export(
    dir: &std::path::Path,
    name: &str,
    subset: &str,
    molecules: Vec<ExportMolecule>,
) -> PathBuf {
    let file = ExportFile {
        schema_version: 1,
        chemistry: CHEMISTRY_VERSION.to_string(),
        rdkit: "test".to_string(),
        source: "test".to_string(),
        seed: 0,
        n_raw: 0,
        spectra_per_molecule: 0,
        skipped_spectra: Default::default(),
        subset: subset.to_string(),
        molecules,
    };
    let path = dir.join(name);
    std::fs::write(&path, serde_json::to_string(&file).unwrap()).unwrap();
    path
}

/// Train on three small molecules; validate on an overlapping ethanol, a
/// fresh propan-2-ol and the out-of-domain chain.
fn write_pair(dir: &std::path::Path) -> (PathBuf, PathBuf) {
    let train = write_export(
        dir,
        "train.json",
        "train",
        vec![
            export_molecule("m-ethanol", &ethanol(), 1),
            export_molecule("m-ether", &dimethyl_ether(), 2),
            export_molecule("m-propanol", &propan_1_ol(), 3),
        ],
    );
    let validation = write_export(
        dir,
        "validation.json",
        "validation",
        vec![
            export_molecule("m-ethanol", &ethanol(), 4),
            export_molecule("m-propan2ol", &propan_2_ol(), 5),
            export_molecule("m-bigchain", &long_chain(), 6),
        ],
    );
    (train, validation)
}

fn temp_dir(test: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("mc4_{test}_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Base [`ExperimentArgs`]: small model, 30 steps, K = 8.
fn base_args(
    test: &str,
    train: PathBuf,
    validation: PathBuf,
    arm: ExperimentArm,
) -> ExperimentArgs {
    ExperimentArgs {
        train,
        validation,
        out: temp_dir(test).join(format!("{}_{}", arm.as_str(), test)),
        name: format!("mc4-{test}-{}", arm.as_str()),
        arm,
        model: "small".to_string(),
        max_atoms: None,
        max_closures: None,
        steps: 30,
        batch: 4,
        lr: 3e-3,
        weight_decay: 0.0,
        grad_clip: None,
        seed: 1,
        report_every: 10,
        eval_every: 0,
        eval_subset: 16,
        trajectories: 8,
        temperature: 1.0,
        returned: 25,
        gen_batch: 2,
        gen_seed: 2,
        extraction_seed: 3,
        limit_train: None,
        limit_validation: None,
        subgroups: None,
        load: None,
        save: None,
        eval_only: false,
        bootstrap: 50,
        progress: false,
        formula_source: mamba3::models::ms2::completion_experiment::FormulaSource::Oracle,
        mass_ppm_tenths: 50,
        mass_uncertainty_uda: 50,
        formula_hypotheses: 8,
        formula_pruning: FormulaPruning::TrainFit,
        formula_allocation: FormulaAllocation::Equal,
        attach_formula_artifacts: false,
        patterns: mamba3::models::ms2::completion_experiment::PatternArg::Random,
        fg_keep_percent: 100,
        fg_aromatic_rings: false,
        substructure_semantics: mamba3::models::ms2::completion_model::SubstructureSemantics::Contained,
        evidence: mamba3::models::ms2::completion_fingerprint::Evidence::Patterns,
        fp_train: None,
        fp_validation: None,
        fp_noise: None,
        fp_train_mode: mamba3::models::ms2::completion_fingerprint::FingerprintMode::Exact,
        fp_eval_mode: mamba3::models::ms2::completion_fingerprint::FingerprintEvalMode::Exact,
        fp_noise_level: mamba3::models::ms2::completion_fingerprint::FingerprintNoiseLevel::Spectrum,
        fp_threshold: 0.1,
        fp_slots: 128,
        exclude_identity_groups: None,
        dump_candidates: None,
    }
}

/// Run the driver, returning the error text (panics on success).
fn run_err(args: &ExperimentArgs, device: &Device<R>) -> String {
    match run(args, device) {
        Err(e) => format!("{e}"),
        Ok(_) => panic!("expected an error"),
    }
}

fn report_json(out: &std::path::Path) -> serde_json::Value {
    let text = std::fs::read_to_string(out.join("report.json")).unwrap();
    serde_json::from_str(&text).unwrap()
}

fn prediction_lines(out: &std::path::Path) -> Vec<serde_json::Value> {
    let text = std::fs::read_to_string(out.join("predictions.jsonl")).unwrap();
    text.lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

#[test]
fn smoke_run_writes_consistent_outputs() {
    let _guard = serial();
    let device = dev();
    let dir = temp_dir("smoke");
    let (train, validation) = write_pair(&dir);
    let args = base_args("smoke", train, validation, ExperimentArm::Full);
    run(&args, &device).unwrap();

    let report = report_json(&args.out);
    let predictions = prediction_lines(&args.out);
    let accounting = &report["accounting"];
    assert_eq!(accounting["validation_read"], 3);
    assert_eq!(accounting["validation_kept"], 2);
    assert_eq!(accounting["validation_skipped"]["too_many_atoms"], 1);
    assert_eq!(accounting["train_read"], 3);
    assert_eq!(accounting["train_kept"], 3);
    // The overlapping ethanol is flagged, not removed.
    assert_eq!(accounting["overlap_strict"], 1);

    assert_eq!(predictions.len(), 3, "one line per molecule read");
    let mut by_hash: HashMap<u64, serde_json::Value> = HashMap::new();
    for line in &predictions {
        let hash = line["key_hash"].as_u64().unwrap();
        by_hash.insert(hash, line.clone());
    }
    for key in ["m-ethanol", "m-propan2ol", "m-bigchain"] {
        let hash = stable_hash(&[key]);
        assert!(by_hash.contains_key(&hash), "a line for {key}");
    }
    let big = &by_hash[&stable_hash(&["m-bigchain"])];
    assert_eq!(big["eligible"], false);
    assert_eq!(big["status"], "out_of_domain:too_many_atoms");
    assert!(big["rank"].is_null());
    for key in ["m-ethanol", "m-propan2ol"] {
        let line = &by_hash[&stable_hash(&[key])];
        assert_eq!(line["eligible"], true);
        assert_eq!(line["status"], "eligible");
        assert!(line["pattern_count"].as_u64().unwrap() > 0);
    }

    assert_eq!(report["metrics_all"]["queries"], 3);
    assert_eq!(report["metrics_eligible"]["queries"], 2);
    // Skipped molecules are misses: the all-read hits equal the eligible
    // hits exactly.
    assert_eq!(
        report["metrics_all"]["top25"]["hits"],
        report["metrics_eligible"]["top25"]["hits"]
    );
    assert!(!report["curve"].as_array().unwrap().is_empty());
    assert_eq!(
        report["scope"],
        format!(
            "{}; substructure_semantics=contained; evidence=patterns",
            mamba3::models::ms2::completion_experiment::EXPERIMENT_SCOPE
        )
        .as_str()
    );
    // MC6: the progress flag exists (off here) and the dead-end diagnostics
    // count every dead-end trajectory exactly once.
    assert_eq!(report["args"]["progress"], false);
    let dead = &report["diagnostics"]["dead_ends"];
    let reason_sum = dead["hydrogen_bound"].as_u64().unwrap()
        + dead["no_open_site"].as_u64().unwrap()
        + dead["open_valence_without_atoms"].as_u64().unwrap()
        + dead["valence_bound"].as_u64().unwrap()
        + dead["unexplained"].as_u64().unwrap();
    assert_eq!(
        dead["total"].as_u64().unwrap(),
        reason_sum,
        "reason counts sum to the dead-end total"
    );
    let mut dead_trajectories = 0u64;
    for line in &predictions {
        if line["eligible"].as_bool().unwrap() {
            dead_trajectories += line["outcome"]["dead_end"].as_u64().unwrap();
        }
    }
    assert_eq!(
        dead["total"].as_u64().unwrap(),
        dead_trajectories,
        "diagnostics count every dead-end trajectory"
    );
}

#[test]
fn arms_share_queries_and_filter() {
    let _guard = serial();
    let device = dev();
    let dir = temp_dir("arms");
    let (train, validation) = write_pair(&dir);
    let mut full = base_args(
        "arms",
        train.clone(),
        validation.clone(),
        ExperimentArm::Full,
    );
    full.save = Some(full.out.join("model.ckpt"));
    let mut formula = base_args("arms", train, validation, ExperimentArm::FormulaOnly);
    formula.save = Some(formula.out.join("model.ckpt"));
    run(&full, &device).unwrap();
    run(&formula, &device).unwrap();

    // Identical evaluation patterns per query in both predictions files.
    let pattern_stats = |out: &std::path::Path| {
        let mut map = HashMap::new();
        for line in prediction_lines(out) {
            if line["eligible"].as_bool().unwrap() {
                map.insert(
                    line["key_hash"].as_u64().unwrap(),
                    (
                        line["pattern_count"].as_u64().unwrap(),
                        line["pattern_atoms"].as_u64().unwrap(),
                    ),
                );
            }
        }
        map
    };
    assert_eq!(pattern_stats(&full.out), pattern_stats(&formula.out));

    // Every candidate of both arms passed the same containment filter:
    // re-run generation from each saved checkpoint and re-check.
    for (args, condition) in [(&full, true), (&formula, false)] {
        let trainer =
            CompletionTrainer::<R, f32>::load(args.save.as_ref().unwrap(), &device).unwrap();
        let file = ExportFile::load(&args.validation).unwrap();
        let set =
            CompletionSet::from_export(&file, Limits::new(16, 4).unwrap(), CANONICAL_WORK_LIMIT)
                .unwrap();
        let config = ExtractionConfig::default();
        let seed = eval_extraction_seed(args.extraction_seed);
        let constants = Ms2Constants::new(&device);
        let gen_config = eval_generation_config(
            args.trajectories,
            args.temperature,
            args.gen_seed,
            args.returned,
            condition,
        );
        for example in &set.examples {
            let patterns = example.patterns(&config, seed, 0).unwrap();
            let graphs: Vec<MolGraph> = patterns
                .iter()
                .map(|p| MolGraph::new(p.graph.atoms().to_vec(), p.graph.bonds().to_vec()).unwrap())
                .collect();
            let requests = [CompletionRequest {
                id: stable_hash(&[example.key.as_str()]),
                composition: example.composition,
                patterns: graphs.as_slice(),
                acceptance_patterns: None,
                fingerprint: None,
            }];
            let outcomes = trainer
                .model()
                .generate(&requests, &gen_config, &constants, &device)
                .unwrap();
            assert_eq!(outcomes.len(), 1);
            for candidate in &outcomes[0].candidates {
                for pattern in &graphs {
                    assert_eq!(
                        contains_pattern(&candidate.graph, pattern, 100_000),
                        Containment::Contained,
                        "every candidate contains every pattern"
                    );
                }
            }
        }
    }
}

#[test]
fn eval_only_reproduces() {
    let _guard = serial();
    let device = dev();
    let dir = temp_dir("repro");
    let (train, validation) = write_pair(&dir);
    let mut first = base_args(
        "repro",
        train.clone(),
        validation.clone(),
        ExperimentArm::Full,
    );
    first.save = Some(first.out.join("model.ckpt"));
    run(&first, &device).unwrap();

    // --eval-only --load reproduces metrics and predictions exactly.
    let mut second = base_args(
        "repro",
        train.clone(),
        validation.clone(),
        ExperimentArm::Full,
    );
    second.out = temp_dir("repro").join("eval_only");
    second.steps = 0;
    second.eval_only = true;
    second.load = first.save.clone();
    run(&second, &device).unwrap();
    for key in ["metrics_all", "metrics_eligible", "diagnostics"] {
        assert_eq!(
            report_json(&first.out)[key],
            report_json(&second.out)[key],
            "{key} reproduces under --eval-only --load"
        );
    }
    assert_eq!(
        std::fs::read(second.out.join("predictions.jsonl")).unwrap(),
        std::fs::read(first.out.join("predictions.jsonl")).unwrap(),
        "predictions reproduce under --eval-only --load"
    );

    // A second identical full run is bit-identical in metrics.
    let mut third = base_args("repro", train, validation, ExperimentArm::Full);
    third.out = temp_dir("repro").join("rerun");
    run(&third, &device).unwrap();
    for key in ["metrics_all", "metrics_eligible", "diagnostics"] {
        assert_eq!(
            report_json(&first.out)[key],
            report_json(&third.out)[key],
            "{key} is deterministic across runs"
        );
    }
}

#[test]
fn subgroups_are_reported() {
    let _guard = serial();
    let device = dev();
    let dir = temp_dir("subgroups");
    let (train, validation) = write_pair(&dir);
    let subgroups_path = dir.join("groups.json");
    std::fs::write(
        &subgroups_path,
        serde_json::json!({
            "group_a": ["m-ethanol", "m-propan2ol", "UNKNOWN_KEY"],
            "group_b": ["m-bigchain"],
            "_meta": ["m-ethanol"],
        })
        .to_string(),
    )
    .unwrap();
    let mut args = base_args("subgroups", train, validation, ExperimentArm::Full);
    args.subgroups = Some(subgroups_path);
    run(&args, &device).unwrap();

    let report = report_json(&args.out);
    let subgroups = report["metrics_subgroups"].as_object().unwrap();
    assert!(subgroups.contains_key("group_a"));
    assert!(subgroups.contains_key("group_b"));
    assert!(
        !subgroups.contains_key("_meta"),
        "_-prefixed labels are ignored"
    );
    // Denominators intersect the read set; the unknown key is only counted.
    assert_eq!(report["accounting"]["subgroup_denominators"]["group_a"], 2);
    assert_eq!(report["accounting"]["subgroup_denominators"]["group_b"], 1);
    assert_eq!(report["accounting"]["subgroup_unknown_keys"], 1);
    assert_eq!(subgroups["group_a"]["queries"], 2);
    assert_eq!(subgroups["group_b"]["queries"], 1);
    assert_eq!(subgroups["group_b"]["top25"]["hits"], 0);
}

#[test]
fn formula_only_ignores_extraction_seed() {
    let _guard = serial();
    let device = dev();
    let dir = temp_dir("seed_invariance");
    let (train, validation) = write_pair(&dir);
    // Two `formula_only` runs differing only in `extraction_seed`: the model
    // never sees a pattern, so training curves, validation NLLs and sampled
    // traces are identical while acceptance counts may differ. A `full` run's
    // validation NLL does change with the seed (regression test for the
    // formula-only validation-loss fix: `eval_nlls` passes empty patterns for
    // `FormulaOnly`).
    let mut first = base_args(
        "seed_a",
        train.clone(),
        validation.clone(),
        ExperimentArm::FormulaOnly,
    );
    first.eval_every = 10;
    first.extraction_seed = 3;
    let mut second = base_args(
        "seed_b",
        train.clone(),
        validation.clone(),
        ExperimentArm::FormulaOnly,
    );
    second.eval_every = 10;
    second.extraction_seed = 4;
    let first_report = run(&first, &device).unwrap();
    let second_report = run(&second, &device).unwrap();
    let curve_key = |curve: &[mamba3::models::ms2::completion_experiment::CurvePoint]| {
        curve
            .iter()
            .map(|p| {
                (
                    p.step,
                    p.loss.map(|v| v.to_bits()),
                    p.eval_nll_per_example.map(f64::to_bits),
                    p.eval_nll_per_token.map(f64::to_bits),
                )
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(
        curve_key(&first_report.curve),
        curve_key(&second_report.curve),
        "formula_only training curves (loss and eval NLLs) ignore the extraction seed"
    );
    assert_eq!(
        first_report.best_eval_nll_per_example.map(f64::to_bits),
        second_report.best_eval_nll_per_example.map(f64::to_bits),
        "formula_only validation NLLs ignore the extraction seed"
    );
    assert_eq!(
        first_report.outcomes.len(),
        second_report.outcomes.len(),
        "same evaluated queries"
    );
    for (q, (a, b)) in first_report
        .outcomes
        .iter()
        .zip(second_report.outcomes.iter())
        .enumerate()
    {
        assert_eq!(a.sampled.len(), b.sampled.len(), "query {q}: sampled count");
        for (i, (sa, sb)) in a.sampled.iter().zip(b.sampled.iter()).enumerate() {
            assert_eq!(sa.trajectory, sb.trajectory, "query {q} row {i}: index");
            assert_eq!(sa.trace, sb.trace, "query {q} row {i}: trace");
            assert_eq!(
                sa.log_prob.to_bits(),
                sb.log_prob.to_bits(),
                "query {q} row {i}: log-probability"
            );
            assert_eq!(sa.status, sb.status, "query {q} row {i}: status");
        }
    }
    // Acceptance counts may differ (patterns differ, the filter does not):
    // no assertion on candidates here by design.
    let mut full_a = base_args(
        "seed_a",
        train.clone(),
        validation.clone(),
        ExperimentArm::Full,
    );
    full_a.eval_every = 10;
    full_a.extraction_seed = 3;
    let mut full_b = base_args("seed_b", train, validation, ExperimentArm::Full);
    full_b.eval_every = 10;
    full_b.extraction_seed = 4;
    let full_a_report = run(&full_a, &device).unwrap();
    let full_b_report = run(&full_b, &device).unwrap();
    let (Some(nll_a), Some(nll_b)) = (
        full_a_report.best_eval_nll_per_example,
        full_b_report.best_eval_nll_per_example,
    ) else {
        panic!("full runs report a validation NLL");
    };
    assert!(
        (nll_a - nll_b).abs() > 1e-12,
        "a full run's validation NLL changes with the extraction seed: {nll_a} vs {nll_b}"
    );
}

#[test]
fn mass_smoke_reports_stages_and_true_joined() {
    let _guard = serial();
    let device = dev();
    let dir = temp_dir("mass_smoke");
    let (train, validation) = write_pair(&dir);
    let mut args = base_args("mass_smoke", train, validation, ExperimentArm::Full);
    args.formula_source = mamba3::models::ms2::completion_experiment::FormulaSource::Mass;
    args.trajectories = 8;
    args.formula_hypotheses = 8;
    args.mass_ppm_tenths = 50;
    args.mass_uncertainty_uda = 50;
    run(&args, &device).unwrap();
    let report = report_json(&args.out);
    let fs = &report["formula_search"];
    for key in [
        "fraction_joined",
        "fraction_after_domain",
        "fraction_after_substructures",
        "fraction_after_completability",
        "fraction_selected",
        "fraction_sampled",
    ] {
        let v = fs[key].as_f64().unwrap();
        assert!((0.0..=1.0).contains(&v), "{key} in [0, 1]: {v}");
    }
    let vals: Vec<f64> = [
        "fraction_joined",
        "fraction_after_domain",
        "fraction_after_substructures",
        "fraction_after_completability",
        "fraction_selected",
        "fraction_sampled",
    ]
    .iter()
    .map(|k| fs[k].as_f64().unwrap())
    .collect();
    for w in vals.windows(2) {
        assert!(w[0] + 1e-12 >= w[1], "stage fractions monotone: {vals:?}");
    }
    // Exact synthetic masses: the true formula joins for every eligible query.
    assert_eq!(fs["fraction_joined"], 1.0);
    let predictions = prediction_lines(&args.out);
    for line in predictions.iter().filter(|l| l["eligible"].as_bool().unwrap()) {
        assert!(line["formula_joined"].as_u64().is_some());
        assert!(line["true_formula_stage"].as_str().is_some());
    }
    assert!(report["scope"].as_str().unwrap().contains("synthetic"));
}

#[test]
fn oracle_run_carries_no_mass_fields() {
    let _guard = serial();
    let device = dev();
    let dir = temp_dir("oracle_clean");
    let (train, validation) = write_pair(&dir);
    let args = base_args("oracle_clean", train, validation, ExperimentArm::Full);
    run(&args, &device).unwrap();
    let report = report_json(&args.out);
    assert!(report.get("formula_search").is_none());
    for line in prediction_lines(&args.out) {
        assert!(line.get("formula_joined").is_none());
        assert!(line.get("true_formula_stage").is_none());
        assert!(line.get("true_excluded_by_train_fit").is_none());
        assert!(line.get("true_absent_from_search").is_none());
    }
}

#[test]
fn functional_groups_smoke_run_writes_new_fields() {
    let _guard = serial();
    let device = dev();
    let dir = temp_dir("fg_smoke");
    let (train, validation) = write_pair(&dir);
    let mut args = base_args("fg_smoke", train, validation, ExperimentArm::Full);
    args.patterns = mamba3::models::ms2::completion_experiment::PatternArg::FunctionalGroups;
    args.steps = 5;
    run(&args, &device).unwrap();
    let report = report_json(&args.out);
    assert_eq!(report["versions"]["pattern_source"], "functional_groups");
    assert_eq!(
        report["versions"]["functional_groups"],
        mamba3::models::ms2::functional_groups::FUNCTIONAL_GROUPS_VERSION
    );
    assert_eq!(
        report["versions"]["aromaticity"],
        mamba3::models::ms2::functional_groups::AROMATICITY_VERSION
    );
    assert!(
        report["scope"]
            .as_str()
            .unwrap()
            .contains("functional groups (Ertl)")
    );
    let eval = &report["eval_patterns"];
    assert_eq!(eval["pattern_source"], "functional_groups");
    assert!(eval.get("fraction_no_group").is_some());
    assert!(eval.get("fraction_truncated").is_some());
    assert!(eval.get("fraction_dropped_oversized").is_some());
    assert!(eval.get("top_groups").is_some());
    // Counts are consistent: eligible lines carry groups_found/truncated.
    let mut found_sum = 0u64;
    for line in prediction_lines(&args.out) {
        if line["eligible"].as_bool().unwrap() {
            assert!(line.get("groups_found").is_some());
            assert!(line.get("truncated").is_some());
            found_sum += line["groups_found"].as_u64().unwrap();
        }
    }
    assert!(found_sum > 0, "the smoke molecules carry functional groups");
    // Mean groups matches the prediction lines.
    let eligible = report["accounting"]["validation_kept"].as_u64().unwrap() as f64;
    let mean = eval["mean_patterns"].as_f64().unwrap();
    assert!((mean * eligible - found_sum as f64).abs() < 1e-9);
}

#[test]
fn random_run_is_deterministic_in_metrics_and_predictions() {
    let _guard = serial();
    let device = dev();
    let dir = temp_dir("random_determinism");
    let (train, validation) = write_pair(&dir);
    let mut first = base_args("random_a", train.clone(), validation.clone(), ExperimentArm::Full);
    first.steps = 5;
    let mut second = base_args("random_b", train, validation, ExperimentArm::Full);
    second.steps = 5;
    second.out = temp_dir("random_determinism").join("rerun");
    second.name = "mc4-random_b-full".to_string();
    // Both default to random patterns.
    assert_eq!(
        first.patterns,
        mamba3::models::ms2::completion_experiment::PatternArg::Random
    );
    run(&first, &device).unwrap();
    run(&second, &device).unwrap();
    for key in ["metrics_all", "metrics_eligible", "diagnostics"] {
        assert_eq!(
            report_json(&first.out)[key],
            report_json(&second.out)[key],
            "{key} is bit-identical across random runs"
        );
    }
    let strip = |lines: Vec<serde_json::Value>| {
        lines
            .into_iter()
            .map(|mut l| {
                // New audit fields may differ in presence; the pinned
                // determinism is the pattern counts and scores.
                l.as_object_mut().unwrap().remove("groups_found");
                l.as_object_mut().unwrap().remove("truncated");
                l
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(
        strip(prediction_lines(&first.out)),
        strip(prediction_lines(&second.out)),
        "predictions are bit-identical modulo the new audit fields"
    );
}

#[test]
fn mass_chemical_only_pruning_reports_recovery_and_exclusions() {
    let _guard = serial();
    let device = dev();
    let dir = temp_dir("mass_chem_only");
    let (train, validation) = write_pair(&dir);
    let mut args = base_args("mass_chem_only", train, validation, ExperimentArm::Full);
    args.formula_source = mamba3::models::ms2::completion_experiment::FormulaSource::Mass;
    args.trajectories = 8;
    args.formula_hypotheses = 8;
    args.mass_ppm_tenths = 50;
    args.mass_uncertainty_uda = 50;
    args.formula_pruning = FormulaPruning::ChemicalOnly;
    run(&args, &device).unwrap();
    let report = report_json(&args.out);
    // The pruning is echoed; the same recovery numbers are reported.
    let fs = &report["formula_search"];
    assert_eq!(fs["pruning"], "chemical_only");
    assert_eq!(fs["allocation"], "equal");
    assert!(report["metrics_eligible"]["queries"].as_u64().unwrap() >= 1);
    assert!(report["metrics_eligible"]["top25"].is_object());
    // Per eligible query the train-fit exclusion is checked both directly
    // and by absence from the actual search, with an aggregate fraction.
    let mut eligible = 0usize;
    for line in prediction_lines(&args.out) {
        if !line["eligible"].as_bool().unwrap() {
            continue;
        }
        eligible += 1;
        assert!(
            line["true_excluded_by_train_fit"].is_boolean(),
            "direct exclusion flag: {line}"
        );
        assert!(
            line["true_absent_from_search"].is_boolean(),
            "absence flag: {line}"
        );
    }
    assert!(eligible >= 1);
    for key in [
        "fraction_true_excluded_by_train_fit",
        "fraction_true_absent_from_search",
    ] {
        let v = fs[key].as_f64().unwrap();
        assert!((0.0..=1.0).contains(&v), "{key} in [0, 1]: {v}");
    }
    // Counts agree with their fractions over the eligible denominator.
    let n = report["accounting"]["validation_kept"].as_f64().unwrap();
    assert_eq!(
        fs["true_excluded_by_train_fit_queries"].as_u64().unwrap() as f64 / n,
        fs["fraction_true_excluded_by_train_fit"].as_f64().unwrap()
    );
    assert_eq!(
        fs["true_absent_from_search_queries"].as_u64().unwrap() as f64 / n,
        fs["fraction_true_absent_from_search"].as_f64().unwrap()
    );
}

#[test]
fn mass_train_frequency_allocation_reports_recovery() {
    let _guard = serial();
    let device = dev();
    let dir = temp_dir("mass_train_freq");
    let (train, validation) = write_pair(&dir);
    let mut args = base_args("mass_train_freq", train, validation, ExperimentArm::Full);
    args.formula_source = mamba3::models::ms2::completion_experiment::FormulaSource::Mass;
    args.trajectories = 8;
    args.formula_hypotheses = 8;
    args.mass_ppm_tenths = 50;
    args.mass_uncertainty_uda = 50;
    args.formula_allocation = FormulaAllocation::TrainFrequency;
    run(&args, &device).unwrap();
    let report = report_json(&args.out);
    let fs = &report["formula_search"];
    assert_eq!(fs["pruning"], "train_fit");
    assert_eq!(fs["allocation"], "train_frequency");
    assert!(report["metrics_eligible"]["queries"].as_u64().unwrap() >= 1);
    assert!(report["metrics_eligible"]["top25"].is_object());
    // Exclusion diagnostics are reported under every mass-arm pruning.
    assert!(fs["fraction_true_excluded_by_train_fit"].as_f64().is_some());
}

#[test]
fn complete_semantics_smoke_run_reports_splits_and_pass_fractions() {
    let _guard = serial();
    let device = dev();
    let dir = temp_dir("complete_smoke");
    let (train, validation) = write_pair(&dir);
    let mut args = base_args("complete_smoke", train, validation, ExperimentArm::Full);
    args.patterns = mamba3::models::ms2::completion_experiment::PatternArg::FunctionalGroups;
    args.substructure_semantics =
        mamba3::models::ms2::completion_model::SubstructureSemantics::CompleteFunctionalGroups;
    args.steps = 5;
    let live = run(&args, &device).unwrap();
    let report = report_json(&args.out);
    // The scope names the semantics.
    assert!(
        report["scope"]
            .as_str()
            .unwrap()
            .contains("substructure_semantics=complete_functional_groups"),
        "scope names the semantics"
    );
    assert_eq!(
        report["args"]["substructure_semantics"],
        "complete_functional_groups"
    );
    // The generation config echoes the semantics.
    assert_eq!(
        report["generation"]["config"]["substructure_semantics"],
        "complete_functional_groups"
    );
    // Predictions were written for every molecule read.
    let predictions = prediction_lines(&args.out);
    assert_eq!(predictions.len(), 3, "one line per molecule read");
    assert!(
        predictions.iter().any(|l| l["eligible"].as_bool().unwrap()),
        "some query was evaluated"
    );
    let diag = &report["diagnostics"];
    // The two new rejection counts are present (u64).
    assert!(diag["rejected_extra_groups"].as_u64().is_some());
    assert!(diag["rejected_missing_groups"].as_u64().is_some());
    // The three pass fractions are present and monotone: complete implies
    // disjoint implies contained, so complete <= disjoint <= contained.
    let contained = diag["pass_contained_fraction"].as_f64().unwrap();
    let disjoint = diag["pass_disjoint_fraction"].as_f64().unwrap();
    let complete = diag["pass_complete_fraction"].as_f64().unwrap();
    for (name, v) in [("contained", contained), ("disjoint", disjoint), ("complete", complete)] {
        assert!((0.0..=1.0).contains(&v), "{name} is a fraction: {v}");
    }
    assert!(
        complete <= disjoint + 1e-12 && disjoint <= contained + 1e-12,
        "monotone pass fractions: complete {complete} <= disjoint {disjoint} <= contained {contained}"
    );
    // The extra/missing split sits inside rejected containment: with the
    // probabilities of this smoke run the accounting identity holds per
    // query in the recorded outcomes (checked by construction in
    // `generate`; here the totals must at least be consistent).
    let eval = &report["eval_patterns"];
    assert!(eval["full_list_acceptance_queries"].as_u64().is_some());
    // The pooled pass fractions equal the pooled outcome counters: every
    // replayed graph counts, including trajectories the active rule
    // rejects (outcomes are memory-only, never written to disk).
    let finished: u32 = live.outcomes.iter().map(|o| o.finished).sum();
    let pass = [
        live.outcomes.iter().map(|o| o.pass_contained).sum::<u32>(),
        live.outcomes.iter().map(|o| o.pass_disjoint).sum::<u32>(),
        live.outcomes.iter().map(|o| o.pass_complete).sum::<u32>(),
    ];
    if finished > 0 {
        for (key, count) in [
            ("pass_contained_fraction", pass[0]),
            ("pass_disjoint_fraction", pass[1]),
            ("pass_complete_fraction", pass[2]),
        ] {
            let fraction = diag[key].as_f64().unwrap();
            assert!(
                (fraction - f64::from(count) / f64::from(finished)).abs() < 1e-12,
                "{key} pools every replayed pass: {fraction} vs {}",
                f64::from(count) / f64::from(finished)
            );
        }
    }
    // Feasible queries serialize no infeasible reason; the field is
    // present only when a query is infeasible (a zero-trajectory miss).
    for line in &predictions {
        assert!(line.get("infeasible_reason").is_none(), "{line}");
    }
}

/// Write a bits sidecar with `bits_by_molecule` aligned with `molecules`.
fn write_bits(dir: &std::path::Path, name: &str, molecules: usize) -> PathBuf {
    let by_molecule: Vec<Vec<u16>> = (0..molecules)
        .map(|i| {
            let mut v = vec![
                (i * 13 % 4000) as u16,
                (i * 29 + 7 % 4000) as u16,
                (i * 7 + 3) as u16,
            ];
            v.sort_unstable();
            v.dedup();
            v
        })
        .collect();
    let mut bits = std::collections::BTreeMap::new();
    for (i, list) in by_molecule.iter().enumerate() {
        bits.insert(format!("m{i}"), list.clone());
    }
    let doc = serde_json::json!({
        "fingerprint": "morgan4096",
        "n_molecules": molecules,
        "bits": bits,
        "bits_by_molecule": by_molecule,
    });
    let path = dir.join(name);
    std::fs::write(&path, serde_json::to_string(&doc).unwrap()).unwrap();
    path
}

fn write_noise(dir: &std::path::Path) -> PathBuf {
    let mut on = vec![0u64; 20];
    let mut off = vec![0u64; 20];
    on[5] = 1;
    on[19] = 3;
    off[0] = 4092;
    off[19] = 4;
    // Molecule-averaged histograms (v2 shape): the averaged row keeps the
    // bin-19 on mass and a share of the off mass.
    let mut on_mol = vec![0u64; 20];
    let mut off_mol = vec![0u64; 20];
    on_mol[19] = 3;
    off_mol[0] = 4090;
    off_mol[19] = 2;
    let doc = serde_json::json!({
        "fingerprint": "morgan4096",
        "n_bits": 4096,
        "n_bins": 20,
        "hist_pred_given_true_on": on,
        "hist_pred_given_true_off": off,
        "hist_pred_given_true_on_molecule": on_mol,
        "hist_pred_given_true_off_molecule": off_mol,
        "n_spectra": 1,
        "n_molecules": 1,
    });
    let path = dir.join("noise.json");
    std::fs::write(&path, serde_json::to_string(&doc).unwrap()).unwrap();
    path
}

fn fp_base_args(
    test: &str,
    train: PathBuf,
    validation: PathBuf,
    fp_train: PathBuf,
    fp_validation: PathBuf,
    fp_noise: PathBuf,
    eval_mode: &str,
) -> ExperimentArgs {
    let mut args = base_args(test, train, validation, ExperimentArm::Full);
    args.steps = 4;
    args.eval_subset = 2;
    args.trajectories = 4;
    args.gen_batch = 2;
    args.evidence = mamba3::models::ms2::completion_fingerprint::Evidence::Fingerprint;
    args.fp_train = Some(fp_train);
    args.fp_validation = Some(fp_validation);
    args.fp_noise = Some(fp_noise);
    args.fp_train_mode = mamba3::models::ms2::completion_fingerprint::FingerprintMode::Exact;
    args.fp_eval_mode = match eval_mode {
        "exact" => mamba3::models::ms2::completion_fingerprint::FingerprintEvalMode::Exact,
        "mist_like" => mamba3::models::ms2::completion_fingerprint::FingerprintEvalMode::MistLike,
        _ => panic!("unknown eval mode"),
    };
    args.fp_threshold = 0.1;
    args.fp_slots = 8;
    args
}

#[test]
fn fingerprint_evidence_smoke_runs_in_each_eval_mode() {
    let _guard = serial();
    let device = dev();
    for eval_mode in ["exact", "mist_like"] {
        let dir = temp_dir(&format!("fp_{eval_mode}"));
        let (train, validation) = write_pair(&dir);
        let fp_train = write_bits(&dir, "fp_train.json", 3);
        let fp_validation = write_bits(&dir, "fp_validation.json", 3);
        let fp_noise = write_noise(&dir);
        let mut args = fp_base_args(
            &format!("fp_{eval_mode}"),
            train,
            validation,
            fp_train,
            fp_validation,
            fp_noise,
            eval_mode,
        );
        // The dump file must live outside the repository.
        let dump = dir.join("candidates.jsonl");
        args.dump_candidates = Some(dump.clone());
        run(&args, &device).unwrap();
        let report = report_json(&args.out);
        assert!(
            report["scope"].as_str().unwrap().contains("evidence=fingerprint"),
            "scope names the evidence"
        );
        assert!(
            report["diagnostics"]["fp_tokens_mean"].as_f64().unwrap() >= 0.0,
            "diagnostics carry tokens per query"
        );
        assert!(
            report["diagnostics"]["fp_entries_dropped_mean"].is_number(),
            "diagnostics carry dropped entries"
        );
        if eval_mode == "mist_like" {
            assert!(
                report["diagnostics"]["fp_true_missing_mean"].is_number(),
                "mist_like reports true missing"
            );
            assert!(
                report["diagnostics"]["fp_false_mean"].is_number(),
                "mist_like reports false tokens"
            );
        }
        // One parseable dump line per evaluated query.
        let text = std::fs::read_to_string(&dump).unwrap();
        let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
        assert_eq!(lines.len(), 2, "one line per eligible query");
        for line in lines {
            let v: serde_json::Value = serde_json::from_str(line).unwrap();
            assert!(v["target"]["atoms"].is_array());
            assert!(v["composition"].is_array());
            assert!(v["candidates"].is_array());
        }
    }
}

#[test]
fn fingerprint_exclusion_removes_groups_and_rejects_remainder() {
    let _guard = serial();
    let device = dev();
    let dir = temp_dir("fp_exclude");
    let (train, validation) = write_pair(&dir);
    let fp_train = write_bits(&dir, "fp_train.json", 3);
    let fp_validation = write_bits(&dir, "fp_validation.json", 3);
    let fp_noise = write_noise(&dir);
    // Exclude group 1 (m-ethanol in train): one molecule removed.
    let exclude = dir.join("exclude.json");
    std::fs::write(
        &exclude,
        serde_json::to_string(&serde_json::json!({"panel_identity_groups": [1]})).unwrap(),
    )
    .unwrap();
    let mut args = fp_base_args(
        "fp_exclude",
        train.clone(),
        validation.clone(),
        fp_train.clone(),
        fp_validation.clone(),
        fp_noise.clone(),
        "exact",
    );
    args.exclude_identity_groups = Some(exclude);
    run(&args, &device).unwrap();
    let report = report_json(&args.out);
    assert_eq!(
        report["accounting"]["train_excluded_identity_groups"], 1,
        "one training molecule removed"
    );
    // Excluding nothing keeps everything.
    let mut args = fp_base_args(
        "fp_exclude_none",
        train.clone(),
        validation.clone(),
        fp_train.clone(),
        fp_validation.clone(),
        fp_noise.clone(),
        "exact",
    );
    args.out = temp_dir("fp_exclude_none").join("full_none");
    run(&args, &device).unwrap();
    let report = report_json(&args.out);
    assert_eq!(report["accounting"]["train_excluded_identity_groups"], 0);
    // Excluding an evaluation group that remains in training errors: exclude
    // group 5 is not in train (validation-only), so this passes; excluding
    // group 4 (validation ethanol) while train still holds ethanol's trace
    // is fine by groups (4 not in train). A remainder error needs an eval
    // group present in train: craft it by excluding group 2 (train-only),
    // which leaves eval groups untouched, then verify the passing path.
    // The error path (eval group left in train) is exercised by excluding
    // nothing while an eval group collides: use a dedicated pair below.
    let dir2 = temp_dir("fp_exclude_err");
    let train2 = write_export(
        &dir2,
        "train.json",
        "train",
        vec![export_molecule("m-ethanol", &ethanol(), 9)],
    );
    let validation2 = write_export(
        &dir2,
        "validation.json",
        "validation",
        vec![export_molecule("m-ethanol2", &ethanol(), 9)],
    );
    let fp_train2 = write_bits(&dir2, "fp_train.json", 1);
    let fp_validation2 = write_bits(&dir2, "fp_validation.json", 1);
    let fp_noise2 = write_noise(&dir2);
    let exclude2 = dir2.join("exclude.json");
    std::fs::write(
        &exclude2,
        serde_json::to_string(&serde_json::json!({"panel_identity_groups": [7]})).unwrap(),
    )
    .unwrap();
    let mut args = fp_base_args(
        "fp_exclude_err",
        train2,
        validation2,
        fp_train2,
        fp_validation2,
        fp_noise2,
        "exact",
    );
    args.exclude_identity_groups = Some(exclude2);
    assert!(
        run(&args, &device).is_err(),
        "an evaluation identity group remaining in training is an error"
    );
    // The dump path inside the repository is refused.
    let mut args = fp_base_args(
        "fp_dump_refused",
        train,
        validation,
        fp_train,
        fp_validation,
        fp_noise,
        "exact",
    );
    args.out = temp_dir("fp_dump_refused").join("full");
    args.dump_candidates = Some(std::path::PathBuf::from("src/fp_dump.jsonl"));
    assert!(run(&args, &device).is_err(), "dump inside the repo is refused");
}

/// Write a panel file (molecule-export schema with `fp_pred_mean`/`fp_true`)
/// for explicit molecules, predictions and truths.
fn write_panel(
    dir: &std::path::Path,
    name: &str,
    molecules: Vec<ExportMolecule>,
    preds: Vec<Vec<(u16, f32)>>,
    trues: Vec<Vec<u16>>,
) -> PathBuf {
    assert_eq!(molecules.len(), preds.len());
    assert_eq!(molecules.len(), trues.len());
    let mols: Vec<serde_json::Value> = molecules
        .iter()
        .enumerate()
        .map(|(i, m)| {
            let pairs: Vec<serde_json::Value> = preds[i]
                .iter()
                .map(|(b, p)| serde_json::json!([b, p]))
                .collect();
            serde_json::json!({
                "key": m.key.clone(),
                "identity_group": m.identity_group,
                "fold_identity": m.fold_identity,
                "atoms": m.atoms.clone(),
                "bonds": m.bonds.clone(),
                "fp_true": trues[i],
                "fp_pred_mean": pairs,
                "spectra": 1,
                "formula": "C2H6O",
            })
        })
        .collect();
    let doc = serde_json::json!({
        "schema_version": 1,
        "chemistry": CHEMISTRY_VERSION,
        "rdkit": "test",
        "source": "test",
        "fingerprint": "morgan4096",
        "definition": "test",
        "max_heavy": 32,
        "panel_identity_groups": molecules.iter().map(|m| m.identity_group).collect::<Vec<_>>(),
        "skipped_panel_molecules": {},
        "molecules": mols,
    });
    let path = dir.join(name);
    std::fs::write(&path, serde_json::to_string(&doc).unwrap()).unwrap();
    path
}

/// Train a tiny exact-fingerprint checkpoint for eval-only tests.
fn train_fp_checkpoint(dir: &std::path::Path, tag: &str) -> PathBuf {
    let device = dev();
    let (train, validation) = write_pair(dir);
    let fp_train = write_bits(dir, &format!("{tag}_fp_train.json"), 3);
    let fp_validation = write_bits(dir, &format!("{tag}_fp_validation.json"), 3);
    let fp_noise = write_noise(dir);
    let mut args = fp_base_args(
        tag,
        train.clone(),
        validation,
        fp_train,
        fp_validation,
        fp_noise,
        "exact",
    );
    let ckpt = dir.join(format!("{tag}.ckpt"));
    args.save = Some(ckpt.clone());
    run(&args, &device).unwrap();
    ckpt
}

/// Eval-only predicted args over a panel file with a pre-chosen checkpoint.
fn predicted_eval_args(
    test: &str,
    train: PathBuf,
    panel: PathBuf,
    ckpt: PathBuf,
    fp_noise: PathBuf,
) -> ExperimentArgs {
    let mut args = base_args(test, train, panel, ExperimentArm::Full);
    args.steps = 0;
    args.eval_only = true;
    args.load = Some(ckpt);
    args.evidence = mamba3::models::ms2::completion_fingerprint::Evidence::Fingerprint;
    args.fp_train_mode = mamba3::models::ms2::completion_fingerprint::FingerprintMode::Exact;
    args.fp_eval_mode =
        mamba3::models::ms2::completion_fingerprint::FingerprintEvalMode::Predicted;
    args.fp_noise = Some(fp_noise);
    args.fp_threshold = 0.1;
    args.fp_slots = 8;
    args.eval_subset = 16;
    args.trajectories = 4;
    args.gen_batch = 2;
    args
}

#[test]
fn duplicate_keys_validate_by_source_index() {
    // Finding 1 (the review's molecules_v1_validation.json case: 68
    // duplicated keys): rows sharing one key validate by source_index, never
    // by key. The old driver rejected the export outright.
    let _guard = serial();
    let device = dev();
    let dir = temp_dir("fp_dupkeys");
    let train = write_export(
        &dir,
        "train.json",
        "train",
        vec![
            export_molecule("t-ethanol", &ethanol(), 1),
            export_molecule("t-ether", &dimethyl_ether(), 2),
            export_molecule("t-propanol", &propan_1_ol(), 3),
        ],
    );
    // A skipped out-of-domain row first (positional alignment must survive
    // it), then two kept rows sharing one key AND one identity group with
    // different graphs (same-key same-group, different bits).
    let validation = write_export(
        &dir,
        "validation.json",
        "validation",
        vec![
            export_molecule("v-big", &long_chain(), 10),
            export_molecule("m-dup", &ethanol(), 11),
            export_molecule("m-dup", &propan_2_ol(), 11),
        ],
    );
    let fp_train = write_bits(&dir, "fp_train.json", 3);
    let fp_validation = write_bits(&dir, "fp_validation.json", 3);
    let fp_noise = write_noise(&dir);
    let mut args = fp_base_args(
        "fp_dupkeys",
        train,
        validation,
        fp_train,
        fp_validation,
        fp_noise,
        "exact",
    );
    let dump = dir.join("candidates.jsonl");
    args.dump_candidates = Some(dump.clone());
    run(&args, &device).unwrap();
    let predictions = prediction_lines(&args.out);
    assert_eq!(predictions.len(), 3, "one line per molecule read");
    let kept: Vec<_> = predictions
        .iter()
        .filter(|l| l["eligible"].as_bool().unwrap())
        .collect();
    assert_eq!(kept.len(), 2);
    // Same key hash (same key) but distinct source positions.
    assert_eq!(kept[0]["key_hash"], kept[1]["key_hash"]);
    assert_ne!(kept[0]["source_index"], kept[1]["source_index"]);
    // The candidate dump carries distinct source positions.
    let text = std::fs::read_to_string(&dump).unwrap();
    let mut positions = std::collections::HashSet::new();
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let v: serde_json::Value = serde_json::from_str(line).unwrap();
        positions.insert(v["source_index"].as_u64().unwrap());
    }
    assert_eq!(positions.len(), 2, "dump lines carry source positions");
}

#[test]
fn predicted_eval_requires_eval_only_with_checkpoint() {
    // Finding 2: a run evaluating on predicted-panel fingerprints must be
    // --eval-only with a checkpoint chosen beforehand; otherwise the driver
    // exits with an error that says why (panel scores must not select
    // `.best`).
    let _guard = serial();
    let device = dev();
    let dir = temp_dir("fp_pred_gate");
    let (train, _) = write_pair(&dir);
    let fp_train = write_bits(&dir, "fp_train.json", 3);
    let fp_validation = write_bits(&dir, "fp_validation.json", 3);
    let fp_noise = write_noise(&dir);
    let panel = write_panel(
        &dir,
        "panel.json",
        vec![
            export_molecule("m-ethanol", &ethanol(), 4),
            export_molecule("m-propan2ol", &propan_2_ol(), 5),
        ],
        vec![vec![(7u16, 0.9f32)], vec![(9u16, 0.8f32)]],
        vec![vec![7u16], vec![9u16]],
    );
    // A training run on predicted fingerprints is refused.
    let mut args = fp_base_args(
        "fp_pred_gate",
        train.clone(),
        panel.clone(),
        fp_train.clone(),
        fp_validation.clone(),
        fp_noise.clone(),
        "exact",
    );
    args.fp_eval_mode =
        mamba3::models::ms2::completion_fingerprint::FingerprintEvalMode::Predicted;
    let err = run_err(&args, &device);
    assert!(
        err.contains("--eval-only"),
        "the error says why: {err}"
    );
    // --eval-only without a checkpoint is refused too.
    let mut args = predicted_eval_args("fp_pred_gate_noload", train, panel, PathBuf::from("no.ckpt"), fp_noise);
    args.load = None;
    let err = run_err(&args, &device);
    assert!(err.contains("--load"), "the error names --load: {err}");
}

#[test]
fn predicted_eval_only_runs_and_ignores_true_bits() {
    // Predicted evaluation works eval-only, and true-bit mutation changes
    // nothing: predicted evidence never substitutes true bits into the
    // fingerprint tokens (review answer 2).
    let _guard = serial();
    let device = dev();
    let dir = temp_dir("fp_pred_truebits");
    let (train, _) = write_pair(&dir);
    let ckpt = train_fp_checkpoint(&dir, "predbase");
    let fp_noise = write_noise(&dir);
    let mols = || {
        vec![
            export_molecule("m-ethanol", &ethanol(), 4),
            export_molecule("m-propan2ol", &propan_2_ol(), 5),
        ]
    };
    let preds = vec![vec![(7u16, 0.9f32)], vec![(9u16, 0.8f32)]];
    let panel_a = write_panel(&dir, "panel_a.json", mols(), preds.clone(), vec![vec![7u16], vec![9u16]]);
    // Same predictions, mutated truths.
    let panel_b = write_panel(&dir, "panel_b.json", mols(), preds, vec![vec![100u16], vec![200u16]]);
    let args_a = predicted_eval_args("fp_pred_a", train.clone(), panel_a, ckpt.clone(), fp_noise.clone());
    run(&args_a, &device).unwrap();
    let mut args_b = predicted_eval_args("fp_pred_b", train, panel_b, ckpt, fp_noise);
    args_b.out = temp_dir("fp_pred_truebits").join("eval_b");
    run(&args_b, &device).unwrap();
    assert_eq!(
        std::fs::read(args_a.out.join("predictions.jsonl")).unwrap(),
        std::fs::read(args_b.out.join("predictions.jsonl")).unwrap(),
        "mutating fp_true changes no prediction"
    );
}

#[test]
fn mass_arm_uses_fingerprint_evidence() {
    // Finding 3: the driver forwarded `None` to run_mass_completion, so the
    // mass path silently dropped the fingerprint. Two mist_like mass runs
    // with different thresholds must condition differently (without the fix
    // both ignore fingerprints and agree exactly).
    let _guard = serial();
    let device = dev();
    let mut bodies = Vec::new();
    for threshold in [0.1f32, 0.9f32] {
        let dir = temp_dir(&format!("fp_mass_{threshold}"));
        let (train, validation) = write_pair(&dir);
        let fp_train = write_bits(&dir, "fp_train.json", 3);
        let fp_validation = write_bits(&dir, "fp_validation.json", 3);
        let fp_noise = write_noise(&dir);
        let mut args = fp_base_args(
            &format!("fp_mass_{threshold}"),
            train,
            validation,
            fp_train,
            fp_validation,
            fp_noise,
            "mist_like",
        );
        args.fp_train_mode =
            mamba3::models::ms2::completion_fingerprint::FingerprintMode::MistLike;
        args.formula_source =
            mamba3::models::ms2::completion_experiment::FormulaSource::Mass;
        args.fp_threshold = threshold;
        run(&args, &device).unwrap();
        bodies.push(std::fs::read(args.out.join("predictions.jsonl")).unwrap());
    }
    assert_ne!(
        bodies[0], bodies[1],
        "threshold changes mass-path fingerprint conditioning"
    );
}

#[test]
fn resume_conflicts_rejected_and_effective_config_reported() {
    // Finding 6: requesting mist_like while loading an exact checkpoint
    // continued with exact fingerprints while the report echoed mist_like.
    let _guard = serial();
    let device = dev();
    let dir = temp_dir("fp_resume");
    let (train, validation) = write_pair(&dir);
    let ckpt = train_fp_checkpoint(&dir, "resumebase");
    let fp_train = write_bits(&dir, "fp_train.json", 3);
    let fp_validation = write_bits(&dir, "fp_validation.json", 3);
    let fp_noise = write_noise(&dir);
    // A conflicting resume is an error naming the conflict.
    let mut args = fp_base_args(
        "fp_resume_conflict",
        train.clone(),
        validation.clone(),
        fp_train.clone(),
        fp_validation.clone(),
        fp_noise.clone(),
        "exact",
    );
    args.load = Some(ckpt.clone());
    args.out = temp_dir("fp_resume").join("conflict");
    args.fp_train_mode =
        mamba3::models::ms2::completion_fingerprint::FingerprintMode::MistLike;
    let err = run_err(&args, &device);
    assert!(err.contains("fp-train-mode"), "names the conflict: {err}");
    // A matching resume trains on.
    let mut args = fp_base_args(
        "fp_resume_ok",
        train.clone(),
        validation.clone(),
        fp_train,
        fp_validation,
        fp_noise.clone(),
        "exact",
    );
    args.load = Some(ckpt.clone());
    args.out = temp_dir("fp_resume").join("ok");
    run(&args, &device).unwrap();
    // An eval-only resume may request anything but reports the checkpoint's
    // own (effective) training configuration.
    let mut args = predicted_eval_args(
        "fp_resume_eval",
        train,
        validation,
        ckpt,
        fp_noise,
    );
    args.fp_eval_mode =
        mamba3::models::ms2::completion_fingerprint::FingerprintEvalMode::Exact;
    args.fp_train_mode =
        mamba3::models::ms2::completion_fingerprint::FingerprintMode::MistLike;
    // Exact eval needs the validation bits sidecar (the train checkpoint's
    // own 3-molecule validation file).
    args.fp_validation = Some(write_bits(&dir, "fp_validation_eval.json", 3));
    args.out = temp_dir("fp_resume").join("eval");
    run(&args, &device).unwrap();
    let report = report_json(&args.out);
    assert_eq!(
        report["train_config"]["fingerprint_mode"], "exact",
        "the report carries the effective (checkpoint) training mode"
    );
}

#[test]
fn malformed_panel_fingerprints_rejected() {
    // Finding 7 (review's 65536-becomes-bit-0 case and unchecked lengths):
    // malformed panel fingerprints are loud errors, never panics or silent
    // bit-identity changes.
    let _guard = serial();
    let device = dev();
    let dir = temp_dir("fp_malformed");
    let (train, _) = write_pair(&dir);
    let ckpt = train_fp_checkpoint(&dir, "malbase");
    let fp_noise = write_noise(&dir);
    let good_mols = || {
        vec![
            export_molecule("m-ethanol", &ethanol(), 4),
            export_molecule("m-propan2ol", &propan_2_ol(), 5),
        ]
    };
    let good_panel = write_panel(
        &dir,
        "panel_good.json",
        good_mols(),
        vec![vec![(7u16, 0.9f32)], vec![(9u16, 0.8f32)]],
        vec![vec![7u16], vec![9u16]],
    );
    let mutate = |tag: &str, f: &dyn Fn(&mut serde_json::Value)| -> PathBuf {
        let text = std::fs::read_to_string(&good_panel).unwrap();
        let mut doc: serde_json::Value = serde_json::from_str(&text).unwrap();
        f(&mut doc);
        let path = dir.join(format!("panel_{tag}.json"));
        std::fs::write(&path, serde_json::to_string(&doc).unwrap()).unwrap();
        path
    };
    let cases: Vec<(&str, PathBuf)> = vec![
        ("pair3", mutate("pair3", &|d| {
            d["molecules"][0]["fp_pred_mean"] = serde_json::json!([[7, 0.9, 0.1]]);
        })),
        ("bit65536", mutate("bit65536", &|d| {
            d["molecules"][0]["fp_pred_mean"] = serde_json::json!([[65536, 0.9]]);
        })),
        ("true99999", mutate("true99999", &|d| {
            d["molecules"][0]["fp_true"] = serde_json::json!([99999]);
        })),
        ("atom256", mutate("atom256", &|d| {
            d["molecules"][0]["atoms"] = serde_json::json!([256, 3, 9]);
        })),
        ("badname", mutate("badname", &|d| {
            d["fingerprint"] = serde_json::json!("ecfp4");
        })),
        ("badschema", mutate("badschema", &|d| {
            d["schema_version"] = serde_json::json!(2);
        })),
    ];
    for (tag, panel) in cases {
        let args = predicted_eval_args(
            &format!("fp_malformed_{tag}"),
            train.clone(),
            panel,
            ckpt.clone(),
            fp_noise.clone(),
        );
        assert!(
            run(&args, &device).is_err(),
            "malformed panel case {tag} is a loud error"
        );
    }
}

#[test]
#[cfg(unix)]
fn dump_symlink_escape_refused() {
    // Finding 11: a dump path through a symlink into the repository is
    // refused (the old check only compared lexical paths).
    let _guard = serial();
    let device = dev();
    let dir = temp_dir("fp_symlink");
    let (train, validation) = write_pair(&dir);
    let mut args = base_args("fp_symlink", train, validation, ExperimentArm::Full);
    args.steps = 0;
    let target = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src");
    let link = dir.join("link_into_repo");
    std::os::unix::fs::symlink(&target, &link).unwrap();
    args.dump_candidates = Some(link.join("evil.jsonl"));
    assert!(
        run(&args, &device).is_err(),
        "a dump through a symlink into the repo is refused"
    );
}

#[test]
fn molecule_noise_level_recorded() {
    // Finding 5: the sampler takes the histogram set by name; the report
    // records it. A v2 noise file is required for the molecule set.
    let _guard = serial();
    let device = dev();
    let dir = temp_dir("fp_noiselevel");
    let (train, validation) = write_pair(&dir);
    let fp_train = write_bits(&dir, "fp_train.json", 3);
    let fp_validation = write_bits(&dir, "fp_validation.json", 3);
    let fp_noise = write_noise(&dir);
    let mut args = fp_base_args(
        "fp_noiselevel",
        train,
        validation,
        fp_train,
        fp_validation,
        fp_noise,
        "mist_like",
    );
    args.fp_train_mode =
        mamba3::models::ms2::completion_fingerprint::FingerprintMode::MistLike;
    args.fp_noise_level =
        mamba3::models::ms2::completion_fingerprint::FingerprintNoiseLevel::Molecule;
    run(&args, &device).unwrap();
    let report = report_json(&args.out);
    assert_eq!(report["diagnostics"]["fp_noise_level"], "molecule");
    assert!(
        report["scope"].as_str().unwrap().contains("fp_noise_level=molecule"),
        "the scope records the noise level"
    );
    // A pre-v2 noise file without molecule histograms refuses the level.
    let legacy = dir.join("noise_legacy.json");
    let text = std::fs::read_to_string(dir.join("noise.json")).unwrap();
    let mut doc: serde_json::Value = serde_json::from_str(&text).unwrap();
    doc.as_object_mut().unwrap().remove("hist_pred_given_true_on_molecule");
    doc.as_object_mut().unwrap().remove("hist_pred_given_true_off_molecule");
    std::fs::write(&legacy, serde_json::to_string(&doc).unwrap()).unwrap();
    let (train2, validation2) = write_pair(&dir);
    let mut args = fp_base_args(
        "fp_noiselevel_legacy",
        train2,
        validation2,
        write_bits(&dir, "fp_train_b.json", 3),
        write_bits(&dir, "fp_validation_b.json", 3),
        legacy,
        "mist_like",
    );
    args.fp_train_mode =
        mamba3::models::ms2::completion_fingerprint::FingerprintMode::MistLike;
    args.fp_noise_level =
        mamba3::models::ms2::completion_fingerprint::FingerprintNoiseLevel::Molecule;
    args.out = temp_dir("fp_noiselevel").join("legacy");
    let err = run_err(&args, &device);
    assert!(err.contains("v2"), "legacy file refuses molecule level: {err}");
}

#[test]
fn initial_best_checkpoint_saved() {
    // Review answer 7: the initial best score was recorded without saving
    // its checkpoint, so generation used final weights despite reporting the
    // earlier best step. The initial best now saves `.best` too.
    let _guard = serial();
    let device = dev();
    let dir = temp_dir("fp_initbest");
    let (train, validation) = write_pair(&dir);
    let mut args = base_args("fp_initbest", train, validation, ExperimentArm::Full);
    args.steps = 11;
    args.eval_every = 10;
    let ckpt = dir.join("model.ckpt");
    args.save = Some(ckpt.clone());
    run(&args, &device).unwrap();
    let mut best = ckpt.as_os_str().to_owned();
    best.push(".best");
    assert!(
        std::path::Path::new(&best).exists(),
        "the initial best has a saved checkpoint"
    );
    let report = report_json(&args.out);
    assert!(
        report["checkpoint_evaluated"]
            .as_str()
            .unwrap()
            .contains("best validation NLL"),
        "generation uses the best checkpoint"
    );
}
