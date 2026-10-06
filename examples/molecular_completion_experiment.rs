//! Molecular-completion experiment driver (MC4): train one arm with a fixed
//! step budget and report top-25 exact-identity recovery with bootstrap
//! intervals, saved checkpoints and per-query predictions.
//!
//! Usage:
//! ```text
//! cargo run --release --no-default-features --features cpu \
//!   --example molecular_completion_experiment -- \
//!   --train <export.json> --validation <export.json> --out <dir> --name <run> \
//!   [--arm full|formula_only|untrained] [--model small|base] \
//!   [--max-atoms 32] [--max-closures 6] \
//!   [--steps N] [--batch B] [--lr F] [--weight-decay F] [--grad-clip F] [--seed S] \
//!   [--report-every N] [--eval-every N] [--eval-subset N] \
//!   [--trajectories K] [--temperature F] [--returned 25] [--gen-batch Q] \
//!   [--gen-seed S] [--extraction-seed S] \
//!   [--limit-train N] [--limit-validation N] [--subgroups <json>] \
//!   [--load <ckpt>] [--save <ckpt>] [--eval-only] [--bootstrap N] \
//!   [--progress|--no-progress]
//! ```
//!
//! Functional-group acceptance:
//! ```text
//!   [--patterns random|functional_groups] [--fg-keep-percent N] [--fg-aromatic-rings] \
//!   [--substructure-semantics contained|disjoint|complete]
//! ```
//!
//! `--substructure-semantics` (default `contained`) picks the host
//! acceptance rule: `contained` (every pattern somewhere, sharing allowed),
//! `disjoint` (distinct occurrences on disjoint atoms) or `complete` (the
//! patterns are the molecule's full functional-group list). With
//! `--patterns functional_groups` and `complete`, evaluation queries carry
//! the untruncated full group list for acceptance while the model sees the
//! seeded subset that fits the encoder.
//!
//! The query is a held-out molecule's exact composition plus substructures
//! cut from that molecule with its own hydrogen counts (synthetic,
//! parent-relative, oracle-formula setting). The report states this scope;
//! the driver does not measure mass-derived formulas, real fragment
//! evidence, calibration or physical validity. The validation export is the
//! model-selection set (best teacher NLL picks the checkpoint); no untouched
//! test set is used.
//!
//! Outputs in `--out`: `report.json`, `predictions.jsonl` (64-bit key
//! hashes only: no keys, no SMILES) and the checkpoint(s). The driver
//! refuses to write inside the repository's `docs/` or `src/`.

use std::path::PathBuf;

use mamba3::backend::Device;
use mamba3::backends::Auto;
use mamba3::models::ms2::completion_eval::HitRate;
use mamba3::models::ms2::completion_experiment::{
    ExperimentArgs, ExperimentArm, ExperimentReport, FormulaSource, PatternArg,
};
use mamba3::models::ms2::completion_fingerprint::{Evidence, FingerprintEvalMode, FingerprintMode};
use mamba3::models::ms2::completion_formula::{FormulaAllocation, FormulaPruning};
use mamba3::models::ms2::completion_model::SubstructureSemantics;

type R = Auto;

fn usage() -> ! {
    eprintln!(
        "usage: molecular_completion_experiment --train <export.json> --validation <export.json> \
         --out <dir> --name <run> [--arm full|formula_only|untrained] [--model small|base] \
         [--max-atoms N] [--max-closures N] [--steps N] [--batch B] [--lr F] [--weight-decay F] \
         [--grad-clip F] [--seed S] [--report-every N] [--eval-every N] [--eval-subset N] \
         [--trajectories K] [--temperature F] [--returned 25] [--gen-batch Q] [--gen-seed S] \
         [--extraction-seed S] [--limit-train N] [--limit-validation N] [--subgroups <json>] \
         [--load <ckpt>] [--save <ckpt>] [--eval-only] [--bootstrap N] [--progress|--no-progress] \
         [--formula-source oracle|mass] [--mass-ppm-tenths N] [--mass-uncertainty-uda N] \
         [--formula-hypotheses N] [--formula-pruning train_fit|chemical_only] \
         [--formula-allocation equal|train_frequency] [--attach-formula-artifacts] \
         [--patterns random|functional_groups] [--fg-keep-percent N] [--fg-aromatic-rings] \
         [--substructure-semantics contained|disjoint|complete] \
         [--evidence patterns|fingerprint|both] [--fp-train <bits.json> --fp-validation <bits.json> \
         --fp-noise <noise.json> --fp-train-mode exact|mist_like --fp-eval-mode exact|mist_like|predicted \
          --fp-noise-level spectrum|molecule \
         --fp-threshold F --fp-slots N] [--exclude-identity-groups <panel.json>] \
         [--dump-candidates <out.jsonl>]"
    );
    std::process::exit(2);
}

fn fail(msg: String) -> ! {
    eprintln!("molecular_completion_experiment: {msg}");
    std::process::exit(1);
}

/// `hits/queries rate [lo, hi]` for one hit rate.
fn hit(rate: &HitRate) -> String {
    format!(
        "{}/{} {:.4} [{:.4}, {:.4}]",
        rate.hits, rate.queries, rate.rate, rate.lo, rate.hi
    )
}

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    let mut train: Option<PathBuf> = None;
    let mut validation: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut name: Option<String> = None;
    let mut arm = ExperimentArm::Full;
    let mut model = "base".to_string();
    let mut max_atoms: Option<usize> = None;
    let mut max_closures: Option<usize> = None;
    let mut steps = 1000usize;
    let mut batch = 16usize;
    let mut lr = 3e-4f32;
    let mut weight_decay = 0.01f32;
    let mut grad_clip: Option<f32> = None;
    let mut seed = 1u64;
    let mut report_every = 50usize;
    let mut eval_every = 0usize;
    let mut eval_subset = 256usize;
    let mut trajectories = 64u32;
    let mut temperature = 1.0f32;
    let mut returned = 25u32;
    let mut gen_batch = 8usize;
    let mut gen_seed = 0u64;
    let mut extraction_seed = 1u64;
    let mut limit_train: Option<usize> = None;
    let mut limit_validation: Option<usize> = None;
    let mut subgroups: Option<PathBuf> = None;
    let mut load: Option<PathBuf> = None;
    let mut save: Option<PathBuf> = None;
    let mut eval_only = false;
    let mut bootstrap = 1000usize;
    let mut progress = true;
    let mut formula_source = FormulaSource::Oracle;
    let mut mass_ppm_tenths = 50u32;
    let mut mass_uncertainty_uda = 50u32;
    let mut formula_hypotheses = 8u32;
    let mut formula_pruning = FormulaPruning::TrainFit;
    let mut formula_allocation = FormulaAllocation::Equal;
    let mut attach_formula_artifacts = false;
    let mut patterns = PatternArg::Random;
    let mut fg_keep_percent = 100u8;
    let mut fg_aromatic_rings = false;
    let mut substructure_semantics = SubstructureSemantics::Contained;
    let mut evidence = Evidence::Patterns;
    let mut fp_train: Option<PathBuf> = None;
    let mut fp_validation: Option<PathBuf> = None;
    let mut fp_noise: Option<PathBuf> = None;
    let mut fp_train_mode = FingerprintMode::Exact;
    let mut fp_eval_mode = FingerprintEvalMode::Exact;
    let mut fp_noise_level = mamba3::models::ms2::completion_fingerprint::FingerprintNoiseLevel::Spectrum;
    let mut fp_threshold = 0.1f32;
    let mut fp_slots = 128u32;
    let mut exclude_identity_groups: Option<PathBuf> = None;
    let mut dump_candidates: Option<PathBuf> = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut next = || args.next().unwrap_or_else(|| usage());
        match arg.as_str() {
            "--train" => train = Some(PathBuf::from(next())),
            "--validation" => validation = Some(PathBuf::from(next())),
            "--out" => out = Some(PathBuf::from(next())),
            "--name" => name = Some(next()),
            "--arm" => {
                arm = ExperimentArm::parse(&next()).unwrap_or_else(|_| usage());
            }
            "--model" => {
                let value = next();
                if value != "small" && value != "base" {
                    usage();
                }
                model = value;
            }
            "--max-atoms" => max_atoms = Some(next().parse().unwrap_or_else(|_| usage())),
            "--max-closures" => max_closures = Some(next().parse().unwrap_or_else(|_| usage())),
            "--steps" => steps = next().parse().unwrap_or_else(|_| usage()),
            "--batch" => batch = next().parse().unwrap_or_else(|_| usage()),
            "--lr" => lr = next().parse().unwrap_or_else(|_| usage()),
            "--weight-decay" => weight_decay = next().parse().unwrap_or_else(|_| usage()),
            "--grad-clip" => grad_clip = Some(next().parse().unwrap_or_else(|_| usage())),
            "--seed" => seed = next().parse().unwrap_or_else(|_| usage()),
            "--report-every" => report_every = next().parse().unwrap_or_else(|_| usage()),
            "--eval-every" => eval_every = next().parse().unwrap_or_else(|_| usage()),
            "--eval-subset" => eval_subset = next().parse().unwrap_or_else(|_| usage()),
            "--trajectories" => trajectories = next().parse().unwrap_or_else(|_| usage()),
            "--temperature" => temperature = next().parse().unwrap_or_else(|_| usage()),
            "--returned" => returned = next().parse().unwrap_or_else(|_| usage()),
            "--gen-batch" => gen_batch = next().parse().unwrap_or_else(|_| usage()),
            "--gen-seed" => gen_seed = next().parse().unwrap_or_else(|_| usage()),
            "--extraction-seed" => extraction_seed = next().parse().unwrap_or_else(|_| usage()),
            "--limit-train" => limit_train = Some(next().parse().unwrap_or_else(|_| usage())),
            "--limit-validation" => {
                limit_validation = Some(next().parse().unwrap_or_else(|_| usage()));
            }
            "--subgroups" => subgroups = Some(PathBuf::from(next())),
            "--load" => load = Some(PathBuf::from(next())),
            "--save" => save = Some(PathBuf::from(next())),
            "--eval-only" => eval_only = true,
            "--bootstrap" => bootstrap = next().parse().unwrap_or_else(|_| usage()),
            "--progress" => progress = true,
            "--no-progress" => progress = false,
            "--formula-source" => {
                formula_source = FormulaSource::parse(&next()).unwrap_or_else(|_| usage());
            }
            "--mass-ppm-tenths" => {
                mass_ppm_tenths = next().parse().unwrap_or_else(|_| usage());
            }
            "--mass-uncertainty-uda" => {
                mass_uncertainty_uda = next().parse().unwrap_or_else(|_| usage());
            }
            "--formula-hypotheses" => {
                formula_hypotheses = next().parse().unwrap_or_else(|_| usage());
            }
            "--formula-pruning" => {
                let value = next();
                formula_pruning = FormulaPruning::parse(&value).unwrap_or_else(|| usage());
            }
            "--formula-allocation" => {
                let value = next();
                formula_allocation = FormulaAllocation::parse(&value).unwrap_or_else(|| usage());
            }
            "--attach-formula-artifacts" => attach_formula_artifacts = true,
            "--patterns" => {
                patterns = PatternArg::parse(&next()).unwrap_or_else(|_| usage());
            }
            "--fg-keep-percent" => {
                fg_keep_percent = next().parse().unwrap_or_else(|_| usage());
            }
            "--fg-aromatic-rings" => fg_aromatic_rings = true,
            "--substructure-semantics" => {
                let value = next();
                substructure_semantics = match value.as_str() {
                    "contained" => SubstructureSemantics::Contained,
                    "disjoint" => SubstructureSemantics::DisjointOccurrences,
                    "complete" => SubstructureSemantics::CompleteFunctionalGroups,
                    _ => usage(),
                };
            }
            "--evidence" => {
                let value = next();
                evidence = Evidence::parse(&value).unwrap_or_else(|| usage());
            }
            "--fp-train" => fp_train = Some(PathBuf::from(next())),
            "--fp-validation" => fp_validation = Some(PathBuf::from(next())),
            "--fp-noise" => fp_noise = Some(PathBuf::from(next())),
            "--fp-train-mode" => {
                let value = next();
                fp_train_mode = FingerprintMode::parse(&value).unwrap_or_else(|| usage());
            }
            "--fp-eval-mode" => {
                let value = next();
                fp_eval_mode = FingerprintEvalMode::parse(&value).unwrap_or_else(|| usage());
            }
            "--fp-noise-level" => {
                let value = next();
                fp_noise_level =
                    mamba3::models::ms2::completion_fingerprint::FingerprintNoiseLevel::parse(
                        &value,
                    )
                    .unwrap_or_else(|| usage());
            }
            "--fp-threshold" => fp_threshold = next().parse().unwrap_or_else(|_| usage()),
            "--fp-slots" => fp_slots = next().parse().unwrap_or_else(|_| usage()),
            "--exclude-identity-groups" => {
                exclude_identity_groups = Some(PathBuf::from(next()));
            }
            "--dump-candidates" => dump_candidates = Some(PathBuf::from(next())),
            _ => usage(),
        }
    }
    let (Some(train), Some(validation), Some(out), Some(name)) = (train, validation, out, name)
    else {
        usage()
    };
    let cli = ExperimentArgs {
        train,
        validation,
        out,
        name,
        arm,
        model,
        max_atoms,
        max_closures,
        steps,
        batch,
        lr,
        weight_decay,
        grad_clip,
        seed,
        report_every,
        eval_every,
        eval_subset,
        trajectories,
        temperature,
        returned,
        gen_batch,
        gen_seed,
        extraction_seed,
        limit_train,
        limit_validation,
        subgroups,
        load,
        save,
        eval_only,
        bootstrap,
        progress,
        formula_source,
        mass_ppm_tenths,
        mass_uncertainty_uda,
        formula_hypotheses,
        formula_pruning,
        formula_allocation,
        attach_formula_artifacts,
        patterns,
        fg_keep_percent,
        fg_aromatic_rings,
        substructure_semantics,
        evidence,
        fp_train,
        fp_validation,
        fp_noise,
        fp_train_mode,
        fp_eval_mode,
        fp_noise_level,
        fp_threshold,
        fp_slots,
        exclude_identity_groups,
        dump_candidates,
    };
    // Attach-only mode: load, fit artifacts from --train, save, assert
    // weights byte-identical, then exit without training or evaluation.
    if cli.attach_formula_artifacts {
        let load_path = cli.load.clone().unwrap_or_else(|| usage());
        let save_path = cli.save.clone().unwrap_or_else(|| usage());
        if cli.steps != 0 {
            // The mode is defined with no training; steps must be zero to
            // avoid accidental retraining.
            eprintln!("molecular_completion_experiment: --attach-formula-artifacts needs --steps 0 (no training)");
            std::process::exit(1);
        }
        let device = Device::<R>::default();
        let mut trainer: mamba3::models::ms2::completion_model::CompletionTrainer<R, f32> =
            mamba3::models::ms2::completion_model::CompletionTrainer::load(&load_path, &device)
                .unwrap_or_else(|e| fail(format!("{e}")));
        // Read the train export and fit on its kept compositions under the
        // loaded model's domain.
        let train_bytes = std::fs::read(&cli.train).unwrap_or_else(|e| fail(format!("{e}")));
        let train_text =
            String::from_utf8(train_bytes).unwrap_or_else(|e| fail(format!("{e}")));
        let train_file: mamba3::models::ms2::dataset::ExportFile =
            mamba3::models::ms2::dataset::ExportFile::from_json(&train_text)
                .unwrap_or_else(|e| fail(format!("{e}")));
        let max_atoms_cfg = trainer.model().config.max_atoms;
        let max_closures_cfg = trainer.model().config.max_ring_closures;
        let limits = mamba3::models::ms2::grammar::Limits::new(
            max_atoms_cfg as usize,
            max_closures_cfg as usize,
        )
        .unwrap_or_else(|e| fail(format!("{e}")));
        let set = mamba3::models::ms2::completion_data::CompletionSet::from_export(
            &train_file,
            limits,
            mamba3::models::ms2::grammar::CANONICAL_WORK_LIMIT,
        )
        .unwrap_or_else(|e| fail(format!("{e}")));
        let keep = cli.limit_train.map_or(set.examples.len(), |n| {
            n.min(set.examples.len())
        });
        let compositions: Vec<mamba3::models::ms2::chem::Composition> = set.examples[..keep]
            .iter()
            .map(|e| e.composition)
            .collect();
        let train_name = cli
            .train
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| cli.train.display().to_string());
        mamba3::models::ms2::completion_experiment::fit_and_attach(
            &mut trainer,
            &compositions,
            max_atoms_cfg,
            format!("completion_experiment:train:{train_name}"),
        )
        .unwrap_or_else(|e| fail(format!("{e}")));
        trainer
            .save(&save_path)
            .unwrap_or_else(|e| fail(format!("{e}")));
        // Weights must be byte-identical: compare the `weights` fields.
        let before_text =
            std::fs::read_to_string(&load_path).unwrap_or_else(|e| fail(format!("{e}")));
        let after_text =
            std::fs::read_to_string(&save_path).unwrap_or_else(|e| fail(format!("{e}")));
        let before_json: serde_json::Value =
            serde_json::from_str(&before_text).unwrap_or_else(|e| fail(format!("{e}")));
        let after_json: serde_json::Value =
            serde_json::from_str(&after_text).unwrap_or_else(|e| fail(format!("{e}")));
        if before_json.get("weights") != after_json.get("weights") {
            fail("attach changed the weights (must be byte-identical)".to_string());
        }
        println!("attached formula artifacts ({} molecules) to {}", compositions.len(), save_path.display());
        return;
    }
    let device = Device::<R>::default();
    let report: ExperimentReport =
        match mamba3::models::ms2::completion_experiment::run(&cli, &device) {
            Ok(report) => report,
            Err(e) => fail(format!("{e}")),
        };
    let _ = argv;

    // Compact table of the metrics and the accounting.
    let a = &report.accounting;
    println!(
        "run             {}  arm {}  model {}",
        report.name, report.arm, report.model_preset
    );
    println!("scope           {}", report.scope);
    println!(
        "validation    read {}  kept {}  skipped {}  limit_excluded {}",
        a.validation_read,
        a.validation_kept,
        a.validation_skipped.values().sum::<u64>(),
        a.limit_excluded
    );
    println!(
        "train         read {}  kept {}  skipped {}  limit_excluded {}",
        a.train_read,
        a.train_kept,
        a.train_skipped.values().sum::<u64>(),
        a.train_limit_excluded
    );
    println!(
        "overlap       strict {}  skeleton {}",
        a.overlap_strict, a.overlap_skeleton
    );
    println!(
        "all-read      top1 {}  top10 {}  top25 {}  skel25 {}",
        hit(&report.metrics_all.top1),
        hit(&report.metrics_all.top10),
        hit(&report.metrics_all.top25),
        hit(&report.metrics_all.skeleton_top25),
    );
    println!(
        "eligible      top1 {}  top10 {}  top25 {}  skel25 {}",
        hit(&report.metrics_eligible.top1),
        hit(&report.metrics_eligible.top10),
        hit(&report.metrics_eligible.top25),
        hit(&report.metrics_eligible.skeleton_top25),
    );
    for (label, metrics) in &report.metrics_subgroups {
        println!("subgroup {label:<12} top25 {}", hit(&metrics.top25));
    }
    println!(
        "in-train      true top25 {}  false top25 {}",
        hit(&report.metrics_identity_in_train_true.top25),
        hit(&report.metrics_identity_in_train_false.top25),
    );
    println!(
        "diagnostics   distinct {:.3}  zero-cand {}  fin {:.3}  dead {:.3}  trunc {:.3}  rej-replay {:.3}  rej-cont {:.3}  unres {:.3}",
        report.diagnostics.mean_distinct,
        report.diagnostics.zero_candidate_queries,
        report.diagnostics.finished_fraction,
        report.diagnostics.dead_end_fraction,
        report.diagnostics.truncated_fraction,
        report.diagnostics.rejected_replay_fraction,
        report.diagnostics.rejected_containment_fraction,
        report.diagnostics.containment_unresolved_fraction,
    );
    println!(
        "timing        train {:.1}s ({:.3}s/100 steps)  gen {:.1}s ({:.1} traj/s)",
        report.train_seconds,
        report.seconds_per_100_steps,
        report.generation.seconds,
        report.generation.trajectories_per_second,
    );
    let dead = &report.diagnostics.dead_ends;
    let cont = &report.diagnostics.rejected_containment_detail;
    println!(
        "dead-ends     total {}  hydrogen {}  no-site {}  open-valence {}  valence {}  unexplained {}  mean-step {:.2}  median-step {:.2}  after-doom {:.2}  root<={} {:.3}",
        dead.total,
        dead.hydrogen_bound,
        dead.no_open_site,
        dead.open_valence_without_atoms,
        dead.valence_bound,
        dead.unexplained,
        dead.mean_dead_end_step,
        dead.median_dead_end_step,
        dead.mean_steps_after_doomed,
        2,
        dead.fraction_doomed_at_root,
    );
    println!(
        "containment   rejected {}  miss-all {} ({:.3})  miss-some {}  type-shortfall {}  type-sufficient {}  queries-all-shortfall {:.3}  finished-cover {:.3}",
        cont.total,
        cont.missing_all,
        cont.fraction_missing_all,
        cont.missing_some,
        cont.type_shortfall,
        cont.type_sufficient,
        cont.fraction_queries_all_shortfall,
        cont.finished_type_cover_fraction,
    );
    println!(
        "semantics     scope {}  extra {}  missing {}  pass-cont {:.3}  pass-disjoint {:.3}  pass-complete {:.3}  full-list-acceptance {}",
        report.scope,
        report.diagnostics.rejected_extra_groups,
        report.diagnostics.rejected_missing_groups,
        report.diagnostics.pass_contained_fraction,
        report.diagnostics.pass_disjoint_fraction,
        report.diagnostics.pass_complete_fraction,
        report.eval_patterns.full_list_acceptance_queries,
    );
    println!("checkpoint    {}", report.checkpoint_evaluated);
    println!("out             {}", cli.out.display());
}
