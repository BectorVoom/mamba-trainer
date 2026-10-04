//! V0 experiment driver: train one control with a fixed step budget and report
//! teacher NLL plus generation metrics with bootstrap intervals.
//!
//! Usage:
//! ```text
//! cargo run --release --no-default-features --features cpu --example ms2_experiment -- \
//!   --train <export.json> [--validation <export.json>] --table <formula_table_v0.json> \
//!   --name <run name> [--overfit 128] [--control none|shuffled|metadata|prior] \
//!   [--steps 3000] [--batch 16] [--lr 3e-4] [--seed 1] [--report-every 50] \
//!   [--eval-every 0] [--k 8] [--bootstrap 1000] [--save <path>] [--load <path>] \
//!   [--eval-only] [--diagnose] --out bench/results/ms2/<name>.json
//! ```
//!
//! `--overfit N` trains on `take_labeled(N)` and evaluates on the same
//! spectra (architecture §7 overfit fixture); otherwise the driver trains on
//! the labeled spectra of `--train` and evaluates on every spectrum of
//! `--validation`. Epochs shuffle deterministically by seed; the step budget
//! is the same for every control, which is what makes the controls
//! comparable. Evaluation runs under the same control the model trains with.
//!
//! `--diagnose` with `--load <ckpt>` runs no training: for the loaded model,
//! on the validation set and on a train subset of the same size (the first N
//! labeled train spectra, N = the validation spectrum count), it reports
//! teacher NLL per token overall and split by field (kind with STOP
//! separated, atom type, bond, pointer), the same NLLs with donor peaks
//! (peak sensitivity, paired donor − own), and an independent NLL
//! recomputation. All aggregates only; `--out` JSON plus a printed table.
//!
//! The export files sample in-domain molecules only, so the metrics are
//! reported as conditional on in-domain molecules (with the export's
//! allow-listed scalar provenance copied into the report by
//! `export_provenance`), never as full-dataset numbers. The report is
//! aggregates only: no per-molecule or per-spectrum field (SMILES, spectrum
//! ids, peaks) is written.

use std::path::PathBuf;
use std::time::Instant;

use mamba3::backend::{Device, launch_count, memory_snapshot, runtime_read_count};
use mamba3::backends::Auto;
use mamba3::models::ms2::contract::{Control, GenerationConfig, ModelConfig};
use mamba3::models::ms2::dataset::percentile;
use mamba3::models::ms2::experiment::{ExperimentSet, SpectrumDomain, export_provenance};
use mamba3::models::ms2::formula::FormulaTable;
use mamba3::models::ms2::metrics::{
    SpectrumEval, field_per_spectrum, paired_interval, summarize, teacher_field_split,
    teacher_nll_per_token,
};
use mamba3::models::ms2::targets::RecipeLimits;
use mamba3::models::ms2::train::{GoldFormulaConditioning, Ms2Trainer, TrainConfig};

type R = Auto;
type E = f32;

fn usage() -> ! {
    eprintln!(
        "usage: ms2_experiment --train <export.json> [--validation <export.json>] \
         --table <table.json> --name <run> [--overfit N] [--control none|shuffled|metadata|prior] \
         [--steps 3000] [--batch 16] [--lr 3e-4] [--seed 1] [--report-every 50] [--eval-every 0] \
         [--k 8] [--bootstrap 1000] [--save <path>] [--load <path>] [--eval-only] [--diagnose] \
         [--formula-window 32|128|512|2048] [--gold-conditioning composition|row] [--formula-source table|enumerate] [--enum-fit <export.json>] [--enum-lane-visits <n>] [--enum-dispatch-visits <n>] [--allocation round-robin|proportional] [--identity trace|graph] [--returned R] [--assign] [--evidence] --out <report.json>"
    );
    std::process::exit(2);
}

fn fail(msg: String) -> ! {
    eprintln!("ms2_experiment: {msg}");
    std::process::exit(1);
}

/// p50/p95 of a sample by linear interpolation (empty samples give zeros).
fn p50p95(mut values: Vec<f64>) -> (f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0);
    }
    values.sort_by(|a, b| a.total_cmp(b));
    (percentile(&values, 50.0), percentile(&values, 95.0))
}

/// Counts of an experiment set by per-spectrum domain.
fn domain_counts(set: &ExperimentSet) -> serde_json::Value {
    let mut labeled = 0u64;
    let mut unlabeled = 0u64;
    let mut out_of_domain = 0u64;
    for s in &set.spectra {
        match s.domain {
            SpectrumDomain::InDomainLabeled => labeled += 1,
            SpectrumDomain::InDomainUnlabeled => unlabeled += 1,
            SpectrumDomain::OutOfDomain(_) => out_of_domain += 1,
        }
    }
    serde_json::json!({
        "spectra": set.spectra.len(),
        "molecules": set.molecules.len(),
        "labeled": labeled,
        "unlabeled": unlabeled,
        "out_of_domain": out_of_domain,
    })
}

fn main() {
    let started = Instant::now();
    let argv: Vec<String> = std::env::args().collect();
    let mut train: Option<PathBuf> = None;
    let mut validation: Option<PathBuf> = None;
    let mut table: Option<PathBuf> = None;
    let mut name: Option<String> = None;
    let mut overfit: Option<usize> = None;
    let mut control = Control::None;
    let mut steps = 3000usize;
    let mut batch = 16usize;
    let mut lr = 3e-4f32;
    let mut seed = 1u64;
    let mut report_every = 50usize;
    let mut eval_every = 0usize;
    let mut k = 8u32;
    let mut bootstrap = 1000usize;
    let mut save: Option<PathBuf> = None;
    let mut load: Option<PathBuf> = None;
    let mut eval_only = false;
    let mut diagnose = false;
    let mut out: Option<PathBuf> = None;
    let mut formula_window = 32u32;
    let mut allocation = mamba3::models::ms2::contract::AllocationMode::RoundRobin;
    let mut identity = mamba3::models::ms2::contract::IdentityMode::TraceOnly;
    let mut returned = 0u32;
    let mut gold_conditioning = mamba3::models::ms2::train::GoldFormulaConditioning::Composition;
    let mut formula_source = mamba3::models::ms2::contract::FormulaSource::Table;
    let mut enum_fit: Option<PathBuf> = None;
    let mut enum_lane_visits: u32 = 4_096;
    let mut enum_dispatch_visits: u32 = 4_000_000;
    let mut assign_flag = false;
    let mut evidence_flag = false;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut next = || args.next().unwrap_or_else(|| usage());
        match arg.as_str() {
            "--train" => train = Some(PathBuf::from(next())),
            "--validation" => validation = Some(PathBuf::from(next())),
            "--table" => table = Some(PathBuf::from(next())),
            "--name" => name = Some(next()),
            "--overfit" => {
                overfit = Some(next().parse().unwrap_or_else(|_| usage()));
            }
            "--control" => {
                control = match next().as_str() {
                    "none" => Control::None,
                    "shuffled" => Control::ShuffledSpectrum,
                    "metadata" => Control::MetadataOnly,
                    "prior" => Control::StructurePrior,
                    _ => usage(),
                };
            }
            "--steps" => steps = next().parse().unwrap_or_else(|_| usage()),
            "--batch" => batch = next().parse().unwrap_or_else(|_| usage()),
            "--lr" => lr = next().parse().unwrap_or_else(|_| usage()),
            "--seed" => seed = next().parse().unwrap_or_else(|_| usage()),
            "--report-every" => report_every = next().parse().unwrap_or_else(|_| usage()),
            "--eval-every" => eval_every = next().parse().unwrap_or_else(|_| usage()),
            "--k" => k = next().parse().unwrap_or_else(|_| usage()),
            "--bootstrap" => bootstrap = next().parse().unwrap_or_else(|_| usage()),
            "--save" => save = Some(PathBuf::from(next())),
            "--load" => load = Some(PathBuf::from(next())),
            "--eval-only" => eval_only = true,
            "--diagnose" => diagnose = true,
            "--formula-window" => {
                formula_window = next().parse().unwrap_or_else(|_| usage());
                if !matches!(formula_window, 32 | 128 | 512 | 2048) {
                    usage();
                }
            }
            "--gold-conditioning" => {
                gold_conditioning = match next().as_str() {
                    "composition" => {
                        mamba3::models::ms2::train::GoldFormulaConditioning::Composition
                    }
                    "row" => mamba3::models::ms2::train::GoldFormulaConditioning::ScoredRowOrZero,
                    _ => usage(),
                };
            }
            "--formula-source" => {
                formula_source = match next().as_str() {
                    "table" => mamba3::models::ms2::contract::FormulaSource::Table,
                    "enumerate" => mamba3::models::ms2::contract::FormulaSource::Enumerate,
                    _ => usage(),
                };
            }
            "--enum-fit" => enum_fit = Some(PathBuf::from(next())),
            "--enum-lane-visits" => enum_lane_visits = next().parse().unwrap_or_else(|_| usage()),
            "--enum-dispatch-visits" => enum_dispatch_visits = next().parse().unwrap_or_else(|_| usage()),
            "--allocation" => {
                allocation = match next().as_str() {
                    "round-robin" => {
                        mamba3::models::ms2::contract::AllocationMode::RoundRobin
                    }
                    "proportional" => {
                        mamba3::models::ms2::contract::AllocationMode::Proportional
                    }
                    _ => usage(),
                };
            }
            "--identity" => {
                identity = match next().as_str() {
                    "trace" => mamba3::models::ms2::contract::IdentityMode::TraceOnly,
                    "graph" => mamba3::models::ms2::contract::IdentityMode::Graph,
                    _ => usage(),
                };
            }
            "--returned" => returned = next().parse().unwrap_or_else(|_| usage()),
            "--assign" => assign_flag = true,
            "--evidence" => evidence_flag = true,
            "--out" => out = Some(PathBuf::from(next())),
            _ => usage(),
        }
    }
    let (Some(train_path), Some(table_path), Some(name), Some(out)) = (train, table, name, out)
    else {
        usage()
    };
    if overfit.is_none() && validation.is_none() {
        fail("either --overfit N or --validation <export.json> is required".to_string());
    }
    if report_every == 0 {
        fail("--report-every must be at least 1".to_string());
    }
    if enum_lane_visits == 0 {
        fail("--enum-lane-visits must be non-zero".to_string());
    }
    if enum_dispatch_visits == 0 {
        fail("--enum-dispatch-visits must be non-zero".to_string());
    }

    let device = Device::<R>::default();
    let table_text = std::fs::read_to_string(&table_path)
        .unwrap_or_else(|e| fail(format!("cannot read {}: {e}", table_path.display())));
    let table = FormulaTable::from_json(&table_text)
        .unwrap_or_else(|e| fail(format!("cannot parse {}: {e}", table_path.display())));
    // Gold-membership probe for the absent-from-table count below.
    let mut table_rows = std::collections::BTreeSet::new();
    for row in 0..table.len() {
        table_rows.insert(*table.composition(row));
    }

    let train_file = std::fs::read_to_string(&train_path)
        .unwrap_or_else(|e| fail(format!("cannot read {}: {e}", train_path.display())));
    let train_json: serde_json::Value = serde_json::from_str(&train_file)
        .unwrap_or_else(|e| fail(format!("cannot parse {}: {e}", train_path.display())));
    let train_full = ExperimentSet::load(&train_path, &RecipeLimits::V0)
        .unwrap_or_else(|e| fail(format!("cannot load {}: {e}", train_path.display())));
    // Enumerating source (V1 §1.4): fit EnumDomain and RatioBounds (margin 0)
    // on the --enum-fit export's molecules ONLY (default: the --train export),
    // before any validation molecule is loaded. D6: the fitting export must be
    // a training subset (train/fit) and share no molecule key with validation.
    let mut enum_domain: Option<mamba3::models::ms2::formula_enum::EnumDomain> = None;
    let mut enum_bounds: Option<mamba3::models::ms2::formula_enum::RatioBounds> = None;
    let mut enum_fit_provenance: Option<(String, String, String)> = None;
    let mut enum_fit_molecules: Vec<String> = Vec::new();
    if matches!(
        formula_source,
        mamba3::models::ms2::contract::FormulaSource::Enumerate
    ) && load.is_none() {
        let fit_path = enum_fit.clone().unwrap_or_else(|| train_path.clone());
        // Subset check from the export file (train/fit only).
        let fit_text = std::fs::read_to_string(&fit_path)
            .unwrap_or_else(|e| fail(format!("cannot read {}: {e}", fit_path.display())));
        let fit_export = mamba3::models::ms2::dataset::ExportFile::from_json(&fit_text)
            .unwrap_or_else(|e| fail(format!("cannot parse --enum-fit {}: {e}", fit_path.display())));
        // D6 subset check via the library (Error::Config); empty eval keys here
        // (overlap is checked after validation loads).
        if let Err(e) = mamba3::models::ms2::experiment::check_enum_fit(
            &fit_export.subset,
            &fit_path.display().to_string(),
            &[],
            &[],
        ) {
            fail(format!("{e}"));
        }
        let fit_set = if fit_path == train_path {
            None
        } else {
            Some(
                ExperimentSet::load(&fit_path, &RecipeLimits::V0).unwrap_or_else(|e| {
                    fail(format!("cannot load --enum-fit {}: {e}", fit_path.display()))
                }),
            )
        };
        let fit_ref = fit_set.as_ref().unwrap_or(&train_full);
        // Provenance for the report and the checkpoint (via TrainConfig).
        let fit_name = fit_path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| fit_path.display().to_string());
        let fit_sha = fit_ref.source_sha256.clone();
        let fit_subset = fit_export.subset.clone();
        enum_fit_provenance = Some((fit_name, fit_sha, fit_subset));
        enum_fit_molecules = fit_ref.molecules.clone();
        let mut comps = Vec::with_capacity(fit_ref.spectra.len());
        for s in &fit_ref.spectra {
            comps.push(s.parent_composition);
        }
        let domain =
            mamba3::models::ms2::formula_enum::EnumDomain::from_compositions(comps.clone(), 0)
                .unwrap_or_else(|e| fail(format!("cannot fit EnumDomain: {e}")));
        let bounds = mamba3::models::ms2::formula_enum::RatioBounds::fit(comps, 0)
            .unwrap_or_else(|e| fail(format!("cannot fit RatioBounds: {e}")));
        enum_domain = Some(domain);
        enum_bounds = Some(bounds);
    }
    // Owned sets for the two roles; each branch below borrows one of these or
    // `train_full`, all of which outlive the run.
    let mut overfit_set: Option<ExperimentSet> = None;
    let mut eval_set_owned: Option<ExperimentSet> = None;
    let mut eval_json = serde_json::Value::Null;
    if let Some(n) = overfit {
        overfit_set = Some(
            train_full
                .take_labeled(n)
                .unwrap_or_else(|e| fail(format!("--overfit {n}: {e}"))),
        );
    } else {
        let validation_path = validation.clone().expect("--validation checked above");
        let raw = std::fs::read_to_string(&validation_path)
            .unwrap_or_else(|e| fail(format!("cannot read {}: {e}", validation_path.display())));
        eval_json = serde_json::from_str(&raw)
            .unwrap_or_else(|e| fail(format!("cannot parse {}: {e}", validation_path.display())));
        eval_set_owned = Some(
            ExperimentSet::load(&validation_path, &RecipeLimits::V0).unwrap_or_else(|e| {
                fail(format!("cannot load {}: {e}", validation_path.display()))
            }),
        );
    }
    let (train_set, train_indices, eval_set, eval_indices): (
        &ExperimentSet,
        Vec<usize>,
        &ExperimentSet,
        Vec<usize>,
    ) = match (&overfit_set, &eval_set_owned) {
        (Some(over), _) => {
            let idx: Vec<usize> = (0..over.spectra.len()).collect();
            (over, idx.clone(), over, idx)
        }
        (None, Some(eval)) => {
            let idx: Vec<usize> = (0..eval.spectra.len()).collect();
            (&train_full, train_full.labeled(), eval, idx)
        }
        (None, None) => unreachable!("one branch always fills"),
    };
    // D6: the fitting export must share no molecule key with validation.
    if matches!(
        formula_source,
        mamba3::models::ms2::contract::FormulaSource::Enumerate
    ) && load.is_none()
        && !enum_fit_molecules.is_empty()
        && eval_set_owned.is_some()
    {
        if let Err(e) = mamba3::models::ms2::experiment::check_enum_fit(
            "train",
            "--enum-fit",
            &enum_fit_molecules,
            &eval_set.molecules,
        ) {
            // check_enum_fit with subset train passes subset; overlap failure
            // names the shared key.
            fail(format!("{e}"));
        }
    }

    let mut train_config = TrainConfig {
        batch,
        slots: 16,
        lr,
        weight_decay: 0.1,
        formula_weight: 0.2,
        seed,
        control,
        grad_clip: None,
        gold_formula_conditioning: gold_conditioning,
        formula_source,
        formula_window,
        enum_lanes_max: 262_144,
        enum_lane_visits_max: enum_lane_visits,
        enum_dispatch_visits_max: enum_dispatch_visits,
        enum_fit_name: enum_fit_provenance.as_ref().map(|p| p.0.clone()),
        enum_fit_sha256: enum_fit_provenance.as_ref().map(|p| p.1.clone()),
        enum_fit_subset: enum_fit_provenance.as_ref().map(|p| p.2.clone()),
        lambda_assign: if assign_flag { 0.1 } else { 0.0 },
        ion_request_work_max: 268_435_456,
    };
    let mut model_config = ModelConfig::v0();
    if assign_flag {
        model_config.assignment = Some(mamba3::models::ms2::contract::AssignmentConfig::default());
    }
    let mut trainer = match &load {
        Some(path) => {
            let t = Ms2Trainer::<R, E>::load(path, &table, &device)
                .unwrap_or_else(|e| fail(format!("cannot load {}: {e}", path.display())));
            if t.train_config().control != control {
                fail(format!(
                    "checkpoint control {:?} does not match --control {control:?}",
                    t.train_config().control
                ));
            }
            if !eval_only {
                eprintln!(
                    "ms2_experiment: continuing from {} with the checkpoint's train config \
                     (CLI --lr/--batch/--seed are ignored except the step budget)",
                    path.display()
                );
            }
            t
        }
        None => Ms2Trainer::<R, E>::new(&model_config, &table, &train_config, &device)
            .unwrap_or_else(|e| fail(format!("cannot build trainer: {e}"))),
    };
    // Enumerating source: upload the fitted artifacts (fresh run) or restore
    // and check them (loaded run via the checkpoint JSON).
    if matches!(
        formula_source,
        mamba3::models::ms2::contract::FormulaSource::Enumerate
    ) && load.is_none()
    {
        let (Some(domain), Some(bounds)) = (enum_domain, enum_bounds) else {
            fail("enumerate source needs fitted EnumDomain/RatioBounds".to_string())
        };
        trainer
            .upload_enum_artifacts(&domain, &bounds)
            .unwrap_or_else(|e| fail(format!("cannot upload enum artifacts: {e}")));
    }
    if matches!(
        formula_source,
        mamba3::models::ms2::contract::FormulaSource::Enumerate
    ) && load.is_some()
    {
        // `Ms2Trainer::load` already restored and checked the artifacts from
        // the checkpoint JSON; ensure the CLI source matches the checkpoint.
        if trainer.train_config().formula_source
            != mamba3::models::ms2::contract::FormulaSource::Enumerate
        {
            fail("checkpoint formula_source is not Enumerate but --formula-source enumerate was given".to_string());
        }
    }
    let effective_train = trainer.train_config().clone();
    let effective_batch = effective_train.batch;

    if diagnose {
        run_diagnose(
            &mut trainer,
            train_set,
            &train_indices,
            eval_set,
            &eval_indices,
            &train_path,
            validation.clone(),
            &table_path,
            &name,
            &argv,
            bootstrap,
            seed,
            effective_batch,
            &out,
        );
        return;
    }

    // One evaluation round: teacher NLL per token over the eval spectra plus
    // generation metrics at K, all under the training control. Teacher batches
    // read once each; generation batches read once per call. Under
    // `ShuffledSpectrum` the peaks are molecule-aware donors (section 1);
    // every evaluation reports `donor_same_molecule` (0) and
    // `donor_no_eligible_peaks`.
    let evaluate = |trainer: &mut Ms2Trainer<R, E>| -> serde_json::Value {
        let gen_batch = effective_batch.min(8).max(1);
        let mut nll_all = Vec::new();
        let mut q_all = Vec::new();
        let mut tok_all = Vec::new();
        let mut mol_all = Vec::new();
        let mut donor_same = 0usize;
        let mut donor_no_eligible = 0usize;
        let mut gold_slots_all: Vec<u32> = Vec::new();
        for chunk in eval_indices.chunks(effective_batch.max(1)) {
            let eval = trainer
                .teacher_eval(eval_set, chunk)
                .unwrap_or_else(|e| fail(format!("teacher_eval: {e}")));
            donor_same += eval.donor_same_molecule;
            donor_no_eligible += eval.donor_no_eligible_peaks;
            nll_all.extend(eval.nll);
            q_all.extend(eval.q);
            tok_all.extend(eval.scored_tokens);
            mol_all.extend(eval.molecules);
            gold_slots_all.extend(eval.gold_slot);
        }
        let gold_not_scored_eval = gold_slots_all.iter().filter(|&&s| s == u32::MAX).count();
        let gold_total_eval = gold_slots_all.len();
        let spectra_n = eval_indices.len();
        let (point, (lo, hi)) = teacher_nll_per_token(
            &nll_all,
            &q_all,
            &tok_all,
            spectra_n,
            effective_train.slots,
            &mol_all,
        );
        let gen_config = GenerationConfig {
            trajectories: k,
            formulas: 4,
            seed,
            control: effective_train.control,
            formula_source: effective_train.formula_source,
            formula_window: effective_train.formula_window,
            enum_lanes_max: effective_train.enum_lanes_max,
            enum_lane_visits_max: effective_train.enum_lane_visits_max,
            enum_dispatch_visits_max: effective_train.enum_dispatch_visits_max,
            allocation,
            identity,
            identity_work_max: 4096,
            returned,
            evidence: evidence_flag,
            ion_request_work_max: 268435456,
            ..GenerationConfig::default()
        };
        let packed_r = gen_config.effective_returned() as usize;
        let mut spectrum_evals: Vec<SpectrumEval> = Vec::with_capacity(spectra_n);
        let mut packed_evals: Vec<SpectrumEval> =
            Vec::with_capacity(spectra_n);
        let mut gen_seconds = Vec::new();
        let mut gen_launches = Vec::new();
        let mut gen_reads = Vec::new();
        // Packed top-R evaluation (V1 §4.4) alongside the trajectory
        // evaluation: duplicate-graph and unresolved rates plus precision /
        // coverage at R.
        let mut dup_weighted = 0.0f64;
        let mut unres_weighted = 0.0f64;
        let mut packed_weight = 0usize;
        // V1 §3.3 dispatch work, summed over the generation chunks below.
        let mut work_submitted = 0usize;
        let mut work_active = 0usize;
        // D5: exhaustion from the request statuses of this same donor-path
        // evaluation (works for one-spectrum final chunks under
        // ShuffledSpectrum via donor_map, unlike the old extra probe with
        // in-batch rotation).
        let mut exhausted_eval = 0usize;
        let mut exhausted_total = 0usize;
        for chunk in eval_indices.chunks(gen_batch) {
            let l0 = launch_count();
            let r0 = runtime_read_count();
            let t0 = Instant::now();
            let (mut evals, work, request_status) = trainer
                .generate_eval_with_work(eval_set, chunk, &gen_config)
                .unwrap_or_else(|e| fail(format!("generate_eval: {e}")));
            work_submitted += work.submitted;
            work_active += work.active_total;
            gen_seconds.push(t0.elapsed().as_secs_f64());
            gen_launches.push((launch_count() - l0) as f64);
            gen_reads.push((runtime_read_count() - r0) as f64);
            for rs in request_status {
                exhausted_total += 1;
                if rs & mamba3::models::ms2::contract::request_status::FORMULA_SEARCH_EXHAUSTED != 0 {
                    exhausted_eval += 1;
                }
            }
            spectrum_evals.append(&mut evals);
            // The packed top-R companion of the same chunk (same request,
            // same seed): precision/coverage at R plus the identity rates.
            let (mut packed_chunk, _, dup_rate, unres_rate, _, _) = trainer
                .generate_eval_packed(eval_set, chunk, &gen_config)
                .unwrap_or_else(|e| fail(format!("generate_eval_packed: {e}")));
            dup_weighted += dup_rate * chunk.len() as f64;
            unres_weighted += unres_rate * chunk.len() as f64;
            packed_weight += chunk.len();
            packed_evals.append(&mut packed_chunk);
        }
        let exhausted_rate_eval = if exhausted_total == 0 { 0.0 } else { exhausted_eval as f64 / exhausted_total as f64 };
        let summary = summarize(&spectrum_evals, k as usize, bootstrap, seed);
        let packed_summary = summarize(&packed_evals, packed_r, bootstrap, seed);
        let duplicate_graph_rate = if packed_weight == 0 { 0.0 } else { dup_weighted / packed_weight as f64 };
        let identity_unresolved_rate = if packed_weight == 0 { 0.0 } else { unres_weighted / packed_weight as f64 };
        let packed_recall_hits = packed_evals
            .iter()
            .filter(|e| e.formula_recall == Some(true))
            .count();
        let packed_recall_total = packed_evals
            .iter()
            .filter(|e| e.formula_recall.is_some())
            .count();
        let (g50, g95) = p50p95(gen_seconds);
        let (l50, l95) = p50p95(gen_launches);
        let absent = eval_indices
            .iter()
            .filter(|&&i| !table_rows.contains(&eval_set.spectra[i].parent_composition))
            .count();
        // Assignment validation metrics (pseudo-label, oracle formula) when
        // `--assign`: `L_assign` on the validation spectra under the true
        // parent plus top-1 and the eligible/partial/dropped counts.
        let (assign_nll_oracle_formula, assign_top1_pseudo_label, assign_eligible, assign_partial, assign_dropped) =
            if assign_flag {
                let mut num = 0.0f64;
                let mut den = 0usize;
                let mut part = 0usize;
                let mut drop = 0usize;
                let mut top_num = 0usize;
                let mut top_den = 0usize;
                for chunk in eval_indices.chunks(effective_batch.max(1)) {
                    match trainer.assign_eval(eval_set, chunk) {
                        Ok((loss, elig, par, dro, top1)) => {
                            num += loss as f64 * elig as f64;
                            den += elig;
                            part += par;
                            drop += dro;
                            if let Some(t1) = top1 {
                                top_num += (t1 * elig as f64).round() as usize;
                                top_den += elig;
                            }
                        }
                        Err(e) => fail(format!("assign_eval: {e}")),
                    }
                }
                let nll = if den == 0 { serde_json::Value::Null } else { serde_json::json!(num / den as f64) };
                let top1 = if top_den == 0 { serde_json::Value::Null } else { serde_json::json!(top_num as f64 / top_den as f64) };
                (nll, top1, den, part, drop)
            } else {
                (serde_json::Value::Null, serde_json::Value::Null, 0, 0, 0)
            };
        // Generated-candidate evidence metrics when `--evidence`: fractions
        // with `evidence_status` 0/1/2 (base, ignoring bit 7) and the mean
        // evidence count.
        let (ev_frac_0, ev_frac_1, ev_frac_2, ev_mean_count) = if evidence_flag {
            let mut c0 = 0usize;
            let mut c1 = 0usize;
            let mut c2 = 0usize;
            let mut tot = 0usize;
            let mut cnt_sum = 0usize;
            for chunk in eval_indices.chunks(gen_batch) {
                let cand = trainer
                    .generate_candidates(eval_set, chunk, &gen_config)
                    .unwrap_or_else(|e| fail(format!("generate_candidates: {e}")));
                for r in 0..cand.batch * cand.trajectories {
                    if cand.status[r] & mamba3::models::ms2::contract::candidate_status::REQUEST_FAILED != 0 {
                        continue;
                    }
                    tot += 1;
                    let base = cand.evidence_status[r] & 0x7F;
                    if base == 0 {
                        c0 += 1;
                    } else if base == 1 {
                        c1 += 1;
                    } else if base == 2 {
                        c2 += 1;
                    }
                    cnt_sum += cand.evidence_count[r] as usize;
                }
            }
            let f0 = if tot == 0 { 0.0 } else { c0 as f64 / tot as f64 };
            let f1 = if tot == 0 { 0.0 } else { c1 as f64 / tot as f64 };
            let f2 = if tot == 0 { 0.0 } else { c2 as f64 / tot as f64 };
            let mean = if tot == 0 { 0.0 } else { cnt_sum as f64 / tot as f64 };
            (
                serde_json::json!(f0),
                serde_json::json!(f1),
                serde_json::json!(f2),
                serde_json::json!(mean),
            )
        } else {
            (
                serde_json::Value::Null,
                serde_json::Value::Null,
                serde_json::Value::Null,
                serde_json::Value::Null,
            )
        };
        serde_json::json!({
            "teacher_nll_per_token": {"point": point, "lo": lo, "hi": hi},
            "donor_same_molecule": donor_same,
            "donor_no_eligible_peaks": donor_no_eligible,
            "metrics": summary,
            "packed_metrics_at_r": packed_summary,
            "packed_r": packed_r,
            "allocation": match allocation {
                mamba3::models::ms2::contract::AllocationMode::RoundRobin => "round_robin",
                mamba3::models::ms2::contract::AllocationMode::Proportional => "proportional",
            },
            "identity": match identity {
                mamba3::models::ms2::contract::IdentityMode::TraceOnly => "trace",
                mamba3::models::ms2::contract::IdentityMode::Graph => "graph",
            },
            "duplicate_graph_rate": duplicate_graph_rate,
            "identity_unresolved_rate": identity_unresolved_rate,
            "packed_formula_recall_hits": packed_recall_hits,
            "packed_formula_recall_total": packed_recall_total,
            "packed_formula_recall_rate": if packed_recall_total == 0 { serde_json::Value::Null } else { serde_json::json!(packed_recall_hits as f64 / packed_recall_total as f64) },
            "gold_formula_absent_from_table": absent,
            "n_spectra": summary.n_spectra,
            "n_molecules": summary.n_molecules,
            "generate_seconds_per_call": {"p50": g50, "p95": g95},
            "launches_per_generate_call": {"p50": l50, "p95": l95},
            "reads_per_generate_call": gen_reads,
            "generation_work": {
                "submitted_trajectory_steps": work_submitted,
                "active_trajectory_steps": work_active,
                "active_fraction": if work_submitted == 0 { 0.0 } else { work_active as f64 / work_submitted as f64 },
            },
            "exhausted": exhausted_eval,
            "exhausted_total": exhausted_total,
            "exhausted_rate": exhausted_rate_eval,
            "gold_not_scored": gold_not_scored_eval,
            "gold_total": gold_total_eval,
            "gold_not_scored_rate": if gold_total_eval == 0 { serde_json::Value::Null } else { serde_json::json!(gold_not_scored_eval as f64 / gold_total_eval as f64) },
            "assign_nll_oracle_formula": assign_nll_oracle_formula,
            "assign_top1_pseudo_label": assign_top1_pseudo_label,
            "assign_eligible_pseudo_label": assign_eligible,
            "assign_partial_pseudo_label": assign_partial,
            "assign_dropped_pseudo_label": assign_dropped,
            "evidence_status_frac_0": ev_frac_0,
            "evidence_status_frac_1": ev_frac_1,
            "evidence_status_frac_2": ev_frac_2,
            "evidence_mean_count": ev_mean_count,
        })
    };

    // Initial evaluation before any step.
    let initial_eval = evaluate(&mut trainer);

    let mut loss_curve = Vec::new();
    let mut step_seconds = Vec::new();
    let mut step_launches = Vec::new();
    let mut step_reads = Vec::new();
    let mut evaluations = Vec::new();
    let mut done = 0usize;
    let mut epoch = 0u64;
    if !eval_only {
        while done < steps {
            for chunk in
                train_set.batches(&train_indices, effective_batch, seed.wrapping_add(epoch))
            {
                if done >= steps {
                    break;
                }
                if done % report_every == 0 {
                    trainer.request_report();
                }
                let l0 = launch_count();
                let r0 = runtime_read_count();
                let t0 = Instant::now();
                let report = trainer
                    .step(train_set, &chunk)
                    .unwrap_or_else(|e| fail(format!("step {done}: {e}")));
                // No device sync here: the step only enqueues. The single
                // batched loss read inside `step` on report steps is the only
                // sync of the training loop.
                step_seconds.push(t0.elapsed().as_secs_f64());
                step_launches.push((launch_count() - l0) as f64);
                step_reads.push((runtime_read_count() - r0) as f64);
                if let Some(rep) = report {
                    loss_curve.push(serde_json::json!({
                        "step": rep.step,
                        "loss": rep.loss,
                        "graph": rep.graph,
                        "formula": rep.formula,
                        "assign_nll_oracle_formula": rep.assign,
                        "spectra": rep.spectra,
                        "formula_present": rep.formula_present,
                        "formula_absent": rep.formula_absent,
                        "gold_not_scored": rep.gold_not_scored,
                        "assign_eligible_pseudo_label": rep.assign_eligible,
                        "assign_partial_pseudo_label": rep.assign_partial,
                        "assign_dropped_pseudo_label": rep.assign_dropped,
                        "assignment_label_overflow": rep.assignment_label_overflow,
                    }));
                }
                done += 1;
                if eval_every > 0 && done % eval_every == 0 && done < steps {
                    let mut round = evaluate(&mut trainer);
                    round["step"] = serde_json::json!(done);
                    evaluations.push(round);
                }
            }
            epoch += 1;
        }
    }
    if let Some(path) = &save {
        trainer
            .save(path)
            .unwrap_or_else(|e| fail(format!("cannot save {}: {e}", path.display())));
    }
    let final_eval = evaluate(&mut trainer);
    evaluations.push({
        let mut round = final_eval.clone();
        round["step"] = serde_json::json!(done);
        round
    });

    let (s50, s95) = p50p95(step_seconds);
    let (sl50, sl95) = p50p95(step_launches);
    let reserved_bytes = memory_snapshot(&device).map(|snapshot| snapshot.bytes_reserved);
    // Enumerating source (V1 §1.4): resident artifact identity plus the
    // per-evaluation exhausted and gold-miss rates.
    let enum_info = if matches!(
        effective_train.formula_source,
        mamba3::models::ms2::contract::FormulaSource::Enumerate
    ) {
        let (p_rows, domain_version, domain_sha, bounds_version, bounds_sha) =
            match trainer.model.enum_artifacts.as_ref() {
                Some(a) => (
                    a.p,
                    a.domain_version.clone(),
                    a.domain_sha256.clone(),
                    a.bounds_version.clone(),
                    a.bounds_sha256.clone(),
                ),
                None => (0, String::new(), String::new(), String::new(), String::new()),
            };
        // D5: exhaustion from the request statuses of the evaluation already
        // run above (donor-path, works for one-spectrum final chunks), not
        // from an extra generate probe. D9: gold_not_scored from the
        // evaluation gold slots (or null when unavailable), never a default 0.
        let exhausted_rate = final_eval
            .get("exhausted_rate")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let gold_not_scored_rate = final_eval
            .get("gold_not_scored_rate")
            .cloned()
            .unwrap_or(serde_json::Value::Null);
        serde_json::json!({
            "formula_source": "enumerate",
            "formula_window": effective_train.formula_window,
            "enum_p": p_rows,
            "enum_domain_version": domain_version,
            "enum_domain_sha256": domain_sha,
            "enum_bounds_version": bounds_version,
            "enum_bounds_sha256": bounds_sha,
            "enum_lanes_max": effective_train.enum_lanes_max,
            "enum_lane_visits_max": effective_train.enum_lane_visits_max,
            "enum_dispatch_visits_max": effective_train.enum_dispatch_visits_max,
            "enum_fit_file": effective_train.enum_fit_name.clone().or_else(|| enum_fit_provenance.as_ref().map(|p| p.0.clone())).unwrap_or_else(|| train_path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()),
            "enum_fit_sha256": effective_train.enum_fit_sha256.clone().or_else(|| enum_fit_provenance.as_ref().map(|p| p.1.clone())).unwrap_or(train_set.source_sha256.clone()),
            "enum_fit_subset": effective_train.enum_fit_subset.clone().or_else(|| enum_fit_provenance.as_ref().map(|p| p.2.clone())).unwrap_or("train".to_string()),
            "exhausted_rate": exhausted_rate,
            "gold_not_scored_rate": gold_not_scored_rate,
        })
    } else {
        serde_json::json!({
            "formula_source": "table",
            "formula_window": effective_train.formula_window,
        })
    };
    let train_file_name = train_path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| train_path.display().to_string());
    let report = serde_json::json!({
        "schema_version": 1,
        "name": name,
        "command_line": argv,
        "provenance": {
            "crate_name": env!("CARGO_PKG_NAME"),
            "crate_version": env!("CARGO_PKG_VERSION"),
            "backend": std::any::type_name::<R>(),
            "train_file": train_file_name,
            "train_source_sha256": train_set.source_sha256,
            "validation_file": validation.clone().map(|p| p.file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| p.display().to_string())),
            "validation_source_sha256": if overfit.is_some() {
                serde_json::json!(train_set.source_sha256)
            } else {
                serde_json::json!(eval_set.source_sha256)
            },
            "table_file": table_path.file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| table_path.display().to_string()),
            "table_rows": table.len(),
            "table_sha256": trainer.table_sha256(),
            "model_config": trainer.model.config,
            "train_config": effective_train,
            "seed": seed,
            "formula_window": formula_window,
            "formula_source_info": enum_info,
            "gold_conditioning": match gold_conditioning {
                GoldFormulaConditioning::Composition => "composition",
                GoldFormulaConditioning::ScoredRowOrZero => "row",
            },
        },
        "datasets": {
            "train_source": domain_counts(&train_full),
            "train_used": train_indices.len(),
            "eval": domain_counts(eval_set),
            "overfit": overfit,
        },
        "export_provenance": {
            // Aggregates only: `export_provenance` keeps the allow-listed
            // scalar header fields plus file name and hash, never the
            // per-molecule `molecules` array (SMILES, spectrum ids, peaks).
            "train": export_provenance(
                &train_json,
                &train_file_name,
                &train_set.source_sha256,
            ),
            "validation": if overfit.is_some() {
                export_provenance(&train_json, &train_file_name, &train_set.source_sha256)
            } else {
                export_provenance(
                    &eval_json,
                    &validation
                        .as_ref()
                        .and_then(|p| p.file_name())
                        .map(|n| n.to_string_lossy().into_owned())
                        .unwrap_or_default(),
                    &eval_set.source_sha256,
                )
            },
            "note": "The exports sample in-domain molecules; the metrics below are conditional on in-domain molecules, never full-dataset numbers. Request-level rejections live in skipped_spectra.",
        },
        "loss_curve": loss_curve,
        "initial_teacher_nll_per_token": initial_eval["teacher_nll_per_token"],
        "final_teacher_nll_per_token": final_eval["teacher_nll_per_token"],
        "final_metrics": final_eval["metrics"],
        "gold_formula_absent_from_table": final_eval["gold_formula_absent_from_table"],
        "evaluations": evaluations,
        "timing": {
            "step_seconds": {"p50": s50, "p95": s95, "n": done},
            "step_sync": "The only device sync in the training loop is the single batched loss read (read_all of [L, L_graph, L_formula, scored_gold_count]) on report steps; per-step times are enqueue times with no sync.",
            "launches_per_training_step": {"p50": sl50, "p95": sl95},
            "reads_per_training_step": step_reads,
            "generate_seconds_per_call": final_eval["generate_seconds_per_call"],
            "launches_per_generate_call": final_eval["launches_per_generate_call"],
            "reads_per_generate_call": final_eval["reads_per_generate_call"],
            "reserved_bytes": reserved_bytes,
            "total_seconds": started.elapsed().as_secs_f64(),
        },
    });
    if let Some(parent) = out.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)
            .unwrap_or_else(|e| fail(format!("cannot create {}: {e}", parent.display())));
    }
    std::fs::write(&out, serde_json::to_string_pretty(&report).unwrap())
        .unwrap_or_else(|e| fail(format!("cannot write {}: {e}", out.display())));

    // One-screen summary.
    let first_loss = loss_curve.first();
    let last_loss = loss_curve.last();
    let nll = |eval: &serde_json::Value| {
        format!(
            "{:.4} [{:.4}, {:.4}]",
            eval["point"].as_f64().unwrap_or(f64::NAN),
            eval["lo"].as_f64().unwrap_or(f64::NAN),
            eval["hi"].as_f64().unwrap_or(f64::NAN)
        )
    };
    println!("run             {name}");
    println!(
        "control         {:?}  steps {done}  batch {effective_batch}  k {k}",
        effective_train.control
    );
    println!(
        "loss            {} -> {}",
        first_loss
            .map(|r| format!("{:.4}", r["loss"].as_f64().unwrap_or(f64::NAN)))
            .as_deref()
            .unwrap_or("n/a"),
        last_loss
            .map(|r| format!("{:.4}", r["loss"].as_f64().unwrap_or(f64::NAN)))
            .as_deref()
            .unwrap_or("n/a"),
    );
    println!(
        "teacher NLL/tok {} -> {}",
        nll(&initial_eval["teacher_nll_per_token"]),
        nll(&final_eval["teacher_nll_per_token"])
    );
    println!(
        "precision       {:.4} [{:.4}, {:.4}]",
        final_eval["metrics"]["precision"]["overall"]["point"]
            .as_f64()
            .unwrap_or(f64::NAN),
        final_eval["metrics"]["precision"]["overall"]["lo"]
            .as_f64()
            .unwrap_or(f64::NAN),
        final_eval["metrics"]["precision"]["overall"]["hi"]
            .as_f64()
            .unwrap_or(f64::NAN),
    );
    println!(
        "coverage (in-domain) {:.4} [{:.4}, {:.4}]",
        final_eval["metrics"]["coverage_conditional"]["overall"]["point"]
            .as_f64()
            .unwrap_or(f64::NAN),
        final_eval["metrics"]["coverage_conditional"]["overall"]["lo"]
            .as_f64()
            .unwrap_or(f64::NAN),
        final_eval["metrics"]["coverage_conditional"]["overall"]["hi"]
            .as_f64()
            .unwrap_or(f64::NAN),
    );
    println!(
        "timing          step {s50:.4}s p50 / {s95:.4}s p95; gen {:.4}s p50/call; launches {sl50:.0} p50/step",
        final_eval["generate_seconds_per_call"]["p50"]
            .as_f64()
            .unwrap_or(f64::NAN),
    );
    println!(
        "packed R={}   precision {:.4} / coverage {:.4} (at R); dup-graph {:.4}; unresolved {:.4}",
        final_eval["packed_r"].as_u64().unwrap_or(0),
        final_eval["packed_metrics_at_r"]["precision"]["overall"]["point"]
            .as_f64()
            .unwrap_or(f64::NAN),
        final_eval["packed_metrics_at_r"]["coverage_conditional"]["overall"]["point"]
            .as_f64()
            .unwrap_or(f64::NAN),
        final_eval["duplicate_graph_rate"].as_f64().unwrap_or(f64::NAN),
        final_eval["identity_unresolved_rate"].as_f64().unwrap_or(f64::NAN),
    );
    println!("out             {}", out.display());
}

/// `--diagnose` of V0-G §2: field-split teacher NLL, peak sensitivity and an
/// independent NLL recomputation, on validation and on a train subset.
///
/// For the loaded model, on the validation set and on the first N labeled
/// train spectra (N = the validation spectrum count), all under the model's
/// own control: teacher NLL per token overall and split by field (kind with
/// STOP separated, atom type, bond, pointer) from `field_log_prob` with the
/// host `use_mask` and target tokens; the same NLLs with donor peaks
/// (section 1 donors, everything else fixed) and the paired per-molecule
/// donor − own differences; and an independent host recomputation of the
/// overall NLL that must match within 1e-6 relative. Aggregates only.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_lines)]
fn run_diagnose(
    trainer: &mut Ms2Trainer<R, E>,
    train_set: &ExperimentSet,
    train_indices: &[usize],
    eval_set: &ExperimentSet,
    eval_indices: &[usize],
    train_path: &std::path::PathBuf,
    validation: Option<std::path::PathBuf>,
    table_path: &std::path::PathBuf,
    name: &str,
    argv: &[String],
    bootstrap: usize,
    seed: u64,
    batch: usize,
    out: &std::path::PathBuf,
) {
    let n = eval_indices.len();
    if n == 0 {
        fail("diagnose: validation set is empty".to_string());
    }
    if train_indices.len() < n {
        fail(format!(
            "diagnose: need {n} labeled train spectra for the train subset, have {}",
            train_indices.len()
        ));
    }
    let train_subset = &train_indices[0..n];
    let val = diagnose_one_set(
        trainer,
        eval_set,
        eval_indices,
        bootstrap,
        seed,
        batch.max(1),
    );
    let tr = diagnose_one_set(
        trainer,
        train_set,
        train_subset,
        bootstrap,
        seed,
        batch.max(1),
    );
    let effective = trainer.train_config().clone();
    let report = serde_json::json!({
        "schema_version": 1,
        "mode": "diagnose",
        "name": name,
        "command_line": argv,
        "control": effective.control,
        "bootstrap": bootstrap,
        "seed": seed,
        "n_spectra": n,
        "provenance": {
            "train_file": train_path.file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| train_path.display().to_string()),
            "validation_file": validation.as_ref().and_then(|p| p.file_name())
                .map(|s| s.to_string_lossy().into_owned()),
            "table_file": table_path.file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| table_path.display().to_string()),
            "table_sha256": trainer.table_sha256(),
            "model_config": trainer.model.config,
            "train_config": effective,
        },
        "validation": val.json,
        "train_subset": tr.json,
        "note": "Aggregates only: no per-spectrum arrays. own = spectrum's own peaks; donor = section-1 donor peaks with everything else fixed. sensitivity = paired per-molecule donor - own.",
    });
    if let Some(parent) = out.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)
            .unwrap_or_else(|e| fail(format!("cannot create {}: {e}", parent.display())));
    }
    std::fs::write(out, serde_json::to_string_pretty(&report).unwrap())
        .unwrap_or_else(|e| fail(format!("cannot write {}: {e}", out.display())));
    println!(
        "diagnose        {name}  control {:?}  n {n}",
        effective.control
    );
    print_diagnose_table("validation", &val);
    print_diagnose_table("train_subset", &tr);
    println!("out             {}", out.display());
}

/// Accumulated host detail for one (set, input) pair.
struct DiagnoseAccum {
    /// JSON for the report.
    json: serde_json::Value,
}

/// Diagnose one spectrum set: own vs donor field splits, sensitivity and the
/// independent NLL check. Returns the JSON plus per-spectrum rows for the
/// table.
fn diagnose_one_set(
    trainer: &mut Ms2Trainer<R, E>,
    set: &ExperimentSet,
    indices: &[usize],
    bootstrap: usize,
    seed: u64,
    batch: usize,
) -> DiagnoseAccum {
    let own = collect_field(trainer, set, indices, false, batch);
    let donor = collect_field(trainer, set, indices, true, batch);
    assert_eq!(own.nll.len(), donor.nll.len(), "own/donor lengths");
    let slots = own.slots;
    let t = own.max_steps;
    let spectra = indices.len();
    let split_own = teacher_field_split(
        &own.nll,
        &own.q,
        &own.scored,
        &own.field,
        &own.use_mask,
        &own.tokens,
        spectra,
        slots,
        t,
        &own.molecules,
        bootstrap,
        seed,
    );
    let split_donor = teacher_field_split(
        &donor.nll,
        &donor.q,
        &donor.scored,
        &donor.field,
        &donor.use_mask,
        &donor.tokens,
        spectra,
        slots,
        t,
        &donor.molecules,
        bootstrap,
        seed,
    );
    // Independent recomputation of the overall NLL per token from the
    // read-back nll/q/scored counts: a second host loop that shares no helper
    // with `teacher_nll_per_token` / `teacher_field_split`.
    let recomputed_own = recompute_nll(
        &own.nll,
        &own.q,
        &own.scored,
        spectra,
        slots,
        &own.molecules,
    );
    let recomputed_donor = recompute_nll(
        &donor.nll,
        &donor.q,
        &donor.scored,
        spectra,
        slots,
        &donor.molecules,
    );
    for (label, primary, recomputed) in [
        ("own", split_own.overall.point, recomputed_own),
        ("donor", split_donor.overall.point, recomputed_donor),
    ] {
        let denom = primary.abs().max(recomputed.abs()).max(1e-12);
        let rel = (primary - recomputed).abs() / denom;
        if rel > 1e-6 {
            fail(format!(
                "diagnose: {label} NLL recomputation differs: primary {primary} vs recomputed {recomputed} (rel {rel})"
            ));
        }
    }
    // Paired per-molecule donor − own differences per bucket from
    // per-spectrum rows, paired by molecule id.
    let own_rows = field_per_spectrum(
        &own.nll,
        &own.q,
        &own.scored,
        &own.field,
        &own.use_mask,
        &own.tokens,
        spectra,
        slots,
        t,
    );
    let donor_rows = field_per_spectrum(
        &donor.nll,
        &donor.q,
        &donor.scored,
        &donor.field,
        &donor.use_mask,
        &donor.tokens,
        spectra,
        slots,
        t,
    );
    let bucket_names = [
        "overall",
        "kind_stop",
        "kind_other",
        "atom_type",
        "bond",
        "pointer",
    ];
    let mut sens_map = serde_json::Map::new();
    for (k, bname) in bucket_names.iter().enumerate() {
        let diffs = paired_diffs(&own_rows, &donor_rows, &own.molecules, k);
        let iv = paired_interval(&diffs, bootstrap, seed);
        sens_map.insert(
            (*bname).to_string(),
            serde_json::json!({"point": iv.point, "lo": iv.lo, "hi": iv.hi, "n_molecules": diffs.len()}),
        );
    }
    let n_molecules = {
        let mut seen = std::collections::BTreeSet::new();
        for &i in indices {
            seen.insert(set.spectra[i].molecule);
        }
        seen.len()
    };
    let json = serde_json::json!({
        "n_spectra": spectra,
        "n_molecules": n_molecules,
        "own": {
            "field_split": split_own,
            "nll_primary": split_own.overall.point,
            "nll_recomputed": recomputed_own,
            "donor_same_molecule": own.donor_same,
            "donor_no_eligible_peaks": own.donor_no,
        },
        "donor": {
            "field_split": split_donor,
            "nll_primary": split_donor.overall.point,
            "nll_recomputed": recomputed_donor,
            "donor_same_molecule": donor.donor_same,
            "donor_no_eligible_peaks": donor.donor_no,
        },
        "sensitivity_donor_minus_own": sens_map,
    });
    // The donor inputs must never reuse the same molecule.
    if donor.donor_same != 0 {
        fail(format!(
            "diagnose: donor_same_molecule = {} (must be 0)",
            donor.donor_same
        ));
    }
    DiagnoseAccum { json }
}

/// Host buffers of one (set, input) collection.
struct FieldBuffers {
    /// Per-target nll `[B*G]`.
    nll: Vec<f32>,
    /// Per-target q `[B*G]`.
    q: Vec<f32>,
    /// Scored tokens `[B*G]`.
    scored: Vec<u32>,
    /// Field log-probs `[B*G*T*4]`.
    field: Vec<f32>,
    /// Use mask `[B*G*T*4]`.
    use_mask: Vec<f32>,
    /// Target tokens `[B*G*T*4]`.
    tokens: Vec<u32>,
    /// Molecule per spectrum `[B]`.
    molecules: Vec<usize>,
    /// Spectra, slots, trace length.
    slots: usize,
    /// Trace length.
    max_steps: usize,
    /// Donor diagnostics summed over chunks.
    donor_same: usize,
    /// Donor diagnostics summed over chunks.
    donor_no: usize,
}

/// Collect `teacher_field_eval` over chunks into concatenated host buffers.
fn collect_field(
    trainer: &mut Ms2Trainer<R, E>,
    set: &ExperimentSet,
    indices: &[usize],
    use_donors: bool,
    batch: usize,
) -> FieldBuffers {
    let mut nll = Vec::new();
    let mut q = Vec::new();
    let mut scored = Vec::new();
    let mut field = Vec::new();
    let mut use_mask = Vec::new();
    let mut tokens = Vec::new();
    let mut molecules = Vec::new();
    let mut slots = 0usize;
    let mut max_steps = 0usize;
    let mut donor_same = 0usize;
    let mut donor_no = 0usize;
    for chunk in indices.chunks(batch) {
        let e = trainer
            .teacher_field_eval(set, chunk, use_donors)
            .unwrap_or_else(|e| fail(format!("teacher_field_eval: {e}")));
        slots = e.slots;
        max_steps = e.max_steps;
        donor_same += e.donor_same_molecule;
        donor_no += e.donor_no_eligible_peaks;
        nll.extend(e.nll);
        q.extend(e.q);
        scored.extend(e.scored_tokens);
        field.extend(e.field_log_prob);
        use_mask.extend(e.use_mask);
        tokens.extend(e.tokens);
        molecules.extend(e.molecules);
    }
    FieldBuffers {
        nll,
        q,
        scored,
        field,
        use_mask,
        tokens,
        molecules,
        slots,
        max_steps,
        donor_same,
        donor_no,
    }
}

/// Second, independent host loop for the overall teacher NLL per token.
///
/// Per spectrum `sum q*nll / sum q*tokens`, then per-molecule means, then the
/// mean over molecules. Written without any metrics helper, so a metric bug
/// shows as a mismatch against the primary value.
fn recompute_nll(
    nll: &[f32],
    q: &[f32],
    scored_tokens: &[u32],
    spectra: usize,
    slots: usize,
    molecules: &[usize],
) -> f64 {
    let mut spec_vals: Vec<Option<(usize, f64)>> = Vec::with_capacity(spectra);
    for b in 0..spectra {
        let mut top = 0.0f64;
        let mut bot = 0.0f64;
        for g in 0..slots {
            let r = b * slots + g;
            top += f64::from(q[r]) * f64::from(nll[r]);
            bot += f64::from(q[r]) * f64::from(scored_tokens[r]);
        }
        if bot > 0.0 {
            spec_vals.push(Some((molecules[b], top / bot)));
        } else {
            spec_vals.push(None);
        }
    }
    let mut groups: std::collections::BTreeMap<usize, (f64, usize)> =
        std::collections::BTreeMap::new();
    for opt in &spec_vals {
        if let Some((m, v)) = opt {
            let e = groups.entry(*m).or_insert((0.0, 0));
            e.0 += *v;
            e.1 += 1;
        }
    }
    if groups.is_empty() {
        return 0.0;
    }
    let mut total = 0.0f64;
    let mut count = 0usize;
    for (_, (s, c)) in &groups {
        total += s / *c as f64;
        count += 1;
    }
    total / count as f64
}

/// Paired per-molecule diffs (donor − own) for bucket `k` from per-spectrum
/// rows, paired by molecule id. Molecules undefined in either input are
/// skipped (own/donor share targets, so defined-ness agrees).
fn paired_diffs(
    own_rows: &[[f64; 6]],
    donor_rows: &[[f64; 6]],
    molecules: &[usize],
    k: usize,
) -> Vec<f64> {
    let mut own_by: std::collections::BTreeMap<usize, Vec<f64>> = std::collections::BTreeMap::new();
    let mut donor_by: std::collections::BTreeMap<usize, Vec<f64>> =
        std::collections::BTreeMap::new();
    for (b, m) in molecules.iter().enumerate() {
        let o = own_rows[b][k];
        let d = donor_rows[b][k];
        if !o.is_nan() && !d.is_nan() {
            own_by.entry(*m).or_default().push(o);
            donor_by.entry(*m).or_default().push(d);
        }
    }
    let mut diffs = Vec::new();
    for (m, ov) in &own_by {
        if let Some(dv) = donor_by.get(m) {
            let mo: f64 = ov.iter().sum::<f64>() / ov.len() as f64;
            let md: f64 = dv.iter().sum::<f64>() / dv.len() as f64;
            diffs.push(md - mo);
        }
    }
    diffs
}

/// Print one diagnose set as a table.
fn print_diagnose_table(label: &str, acc: &DiagnoseAccum) {
    let get = |v: &serde_json::Value, key: &str| -> (f64, f64, f64) {
        (
            v[key]["point"].as_f64().unwrap_or(f64::NAN),
            v[key]["lo"].as_f64().unwrap_or(f64::NAN),
            v[key]["hi"].as_f64().unwrap_or(f64::NAN),
        )
    };
    let own = &acc.json["own"]["field_split"];
    let donor = &acc.json["donor"]["field_split"];
    let sens = &acc.json["sensitivity_donor_minus_own"];
    println!("-- {label} (n_spectra {})", acc.json["n_spectra"]);
    println!(
        "   donor_same_molecule own/donor {}/{}  no_eligible own/donor {}/{}",
        acc.json["own"]["donor_same_molecule"],
        acc.json["donor"]["donor_same_molecule"],
        acc.json["own"]["donor_no_eligible_peaks"],
        acc.json["donor"]["donor_no_eligible_peaks"],
    );
    println!(
        "   overall NLL own {:.4} recomputed {:.4} | donor {:.4} recomputed {:.4}",
        acc.json["own"]["nll_primary"].as_f64().unwrap_or(f64::NAN),
        acc.json["own"]["nll_recomputed"]
            .as_f64()
            .unwrap_or(f64::NAN),
        acc.json["donor"]["nll_primary"]
            .as_f64()
            .unwrap_or(f64::NAN),
        acc.json["donor"]["nll_recomputed"]
            .as_f64()
            .unwrap_or(f64::NAN),
    );
    for bname in [
        "overall",
        "kind_stop",
        "kind_other",
        "atom_type",
        "bond",
        "pointer",
    ] {
        let (op, olo, ohi) = get(own, bname);
        let (dp, dlo, dhi) = get(donor, bname);
        let (sp, slo, shi) = get(sens, bname);
        println!(
            "   {bname:10} own {op:.4} [{olo:.4},{ohi:.4}] donor {dp:.4} [{dlo:.4},{dhi:.4}] diff {sp:.4} [{slo:.4},{shi:.4}]"
        );
    }
}
