//! Functional-group evaluation driver `ms2-fg-v4`.
//!
//! Loads a frozen trainer checkpoint exactly as `ms2_experiment --load
//! --eval-only` does (formula source, window, enumeration artifacts, formula
//! features and every other setting from the checkpoint; the checkpoint's
//! control must equal `--control`), generates candidates for every validation
//! spectrum, and scores the predicted graphs at the functional-group level
//! against the true parent's groups.
//!
//! Usage:
//! ```text
//! cargo run --release --no-default-features --features cpu --example ms2_fg_eval -- \
//!   --load <trainer checkpoint> --table <table.json> --train <export> \
//!   --validation <export> [--control none|shuffled] [--k 8] [--batch 16] \
//!   [--bootstrap 1000] [--seed 1] [--allocation round-robin|proportional] \
//!   [--identity trace|graph] [--enum-dispatch-visits N] [--limit-spectra N] \
//!   [--donor-peaks] --out <report.json>
//! ```
//!
//! `--donor-peaks` (an evaluation-only input ablation, documented as such)
//! evaluates the SAME loaded checkpoint — which must be trained with control
//! none — a second time with each validation spectrum's peaks replaced by a
//! donor spectrum's peaks from another molecule. Donor assembly reuses the
//! trainer's molecule-aware donors from [`ExperimentSet::donor_map`] under
//! the trainer seed (the same path evaluation under the shuffled control
//! takes, via a per-request shuffled control that the loader accepts without
//! retraining); precursor and metadata stay the spectrum's own. The report
//! then holds both rows (`model` and `model_donor_peaks`) and, for every set
//! metric at each `k`, a paired bootstrap interval over molecules of the
//! difference own − donor (same spectra, same resamples).

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

use mamba3::backend::{Device, runtime_read_count};
use mamba3::backends::Auto;
use mamba3::models::ms2::contract::{AllocationMode, Control, GenerationConfig, IdentityMode};
use mamba3::models::ms2::experiment::ExperimentSet;
use mamba3::models::ms2::functional_groups::{FG_NAMES, FG_VERSION, functional_groups_v4};
use mamba3::models::ms2::functional_groups_eval::{
    FgSpectrumDatum, MetricPoint, candidate_size_dist, choose_prior, closing_fragment_not_found,
    eval_records, evaluate_fg, evaluate_sets, formula_aware_set, label_union,
    recipe_fragments_union, spectrum_datum,
};
use mamba3::models::ms2::targets::RecipeLimits;
use mamba3::models::ms2::train::Ms2Trainer;
use mamba3::models::ms2::{CHEMISTRY_VERSION, experiment};

type R = Auto;
type E = f32;

fn usage() -> ! {
    eprintln!(
        "usage: ms2_fg_eval --load <ckpt> --table <table.json> --train <export> \
         --validation <export> [--control none|shuffled] [--k 8] [--batch 16] \
         [--bootstrap 1000] [--seed 1] [--allocation round-robin|proportional] \
         [--identity trace|graph] [--enum-dispatch-visits N] [--limit-spectra N] \
         [--donor-peaks] --out <report.json>"
    );
    std::process::exit(2);
}

fn fail(msg: String) -> ! {
    eprintln!("ms2_fg_eval: {msg}");
    std::process::exit(1);
}

fn main() {
    let started = Instant::now();
    let mut load: Option<PathBuf> = None;
    let mut table: Option<PathBuf> = None;
    let mut train: Option<PathBuf> = None;
    let mut validation: Option<PathBuf> = None;
    let mut control = Control::None;
    let mut control_given = false;
    let mut k = 8usize;
    let mut batch = 16usize;
    let mut bootstrap = 1000usize;
    let mut seed = 1u64;
    let mut allocation = AllocationMode::RoundRobin;
    let mut identity = IdentityMode::Graph;
    let mut enum_dispatch_visits: Option<u32> = None;
    let mut limit_spectra: Option<usize> = None;
    let mut donor_peaks = false;
    let mut out: Option<PathBuf> = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut next = || args.next().unwrap_or_else(|| usage());
        match arg.as_str() {
            "--load" => load = Some(PathBuf::from(next())),
            "--table" => table = Some(PathBuf::from(next())),
            "--train" => train = Some(PathBuf::from(next())),
            "--validation" => validation = Some(PathBuf::from(next())),
            "--control" => {
                control_given = true;
                control = match next().as_str() {
                    "none" => Control::None,
                    "shuffled" => Control::ShuffledSpectrum,
                    _ => usage(),
                };
            }
            "--k" => k = next().parse().unwrap_or_else(|_| usage()),
            "--batch" => batch = next().parse().unwrap_or_else(|_| usage()),
            "--bootstrap" => bootstrap = next().parse().unwrap_or_else(|_| usage()),
            "--seed" => seed = next().parse().unwrap_or_else(|_| usage()),
            "--allocation" => {
                allocation = match next().as_str() {
                    "round-robin" => AllocationMode::RoundRobin,
                    "proportional" => AllocationMode::Proportional,
                    _ => usage(),
                };
            }
            "--identity" => {
                identity = match next().as_str() {
                    "trace" => IdentityMode::TraceOnly,
                    "graph" => IdentityMode::Graph,
                    _ => usage(),
                };
            }
            "--enum-dispatch-visits" => {
                enum_dispatch_visits = Some(next().parse().unwrap_or_else(|_| usage()));
            }
            "--limit-spectra" => limit_spectra = Some(next().parse().unwrap_or_else(|_| usage())),
            "--donor-peaks" => donor_peaks = true,
            "--out" => out = Some(PathBuf::from(next())),
            _ => usage(),
        }
    }
    let (Some(load), Some(table_path), Some(train_path), Some(validation_path), Some(out)) =
        (load, table, train, validation, out)
    else {
        usage()
    };
    if !(1..=64).contains(&k) {
        fail("--k must be in 1..=64".to_string());
    }
    if batch == 0 {
        fail("--batch must be non-zero".to_string());
    }
    if bootstrap == 0 {
        fail("--bootstrap must be non-zero".to_string());
    }
    let _ = control_given;

    let device = Device::<R>::default();
    let table_text = std::fs::read_to_string(&table_path)
        .unwrap_or_else(|e| fail(format!("cannot read {}: {e}", table_path.display())));
    let table = mamba3::models::ms2::formula::FormulaTable::from_json(&table_text)
        .unwrap_or_else(|e| fail(format!("cannot parse {}: {e}", table_path.display())));
    let train_set = ExperimentSet::load(&train_path, &RecipeLimits::V0)
        .unwrap_or_else(|e| fail(format!("cannot load {}: {e}", train_path.display())));
    let val_set = ExperimentSet::load(&validation_path, &RecipeLimits::V0)
        .unwrap_or_else(|e| fail(format!("cannot load {}: {e}", validation_path.display())));

    let trainer = Ms2Trainer::<R, E>::load(&load, &table, &device)
        .unwrap_or_else(|e| fail(format!("cannot load {}: {e}", load.display())));
    if trainer.train_config().control != control {
        fail(format!(
            "checkpoint control {:?} does not match --control {control:?}",
            trainer.train_config().control
        ));
    }
    let train_cfg = trainer.train_config().clone();
    let mut gen_config = GenerationConfig {
        trajectories: k as u32,
        formulas: 4,
        seed,
        control: Control::None,
        formula_source: train_cfg.formula_source,
        formula_window: train_cfg.formula_window,
        enum_lanes_max: train_cfg.enum_lanes_max,
        enum_lane_visits_max: train_cfg.enum_lane_visits_max,
        enum_dispatch_visits_max: enum_dispatch_visits
            .unwrap_or(train_cfg.enum_dispatch_visits_max),
        allocation,
        identity,
        identity_work_max: 4096,
        returned: 0,
        evidence: false,
        ion_request_work_max: 268_435_456,
        formula_evidence_work_max: train_cfg.formula_evidence_work_max,
        formula_evidence_dispatch_max: train_cfg.formula_evidence_dispatch_max,
        ..GenerationConfig::default()
    };
    let model_atoms = trainer.model.config.max_atoms as usize;
    let model_closures = trainer.model.config.max_ring_closures as usize;
    gen_config.max_steps = (2 + model_atoms + model_closures) as u32;
    if let Err(e) = gen_config.validate(model_atoms, model_closures) {
        fail(format!("generation config rejected: {e}"));
    }

    // All validation spectra (or the first N): generation reads once per
    // `generate` call; the first call warms up, the rest are asserted.
    let mut val_indices: Vec<usize> = (0..val_set.spectra.len()).collect();
    if let Some(n) = limit_spectra {
        val_indices.truncate(n);
    }
    let gen_batch = batch.min(8).max(1);
    let mut data: Vec<FgSpectrumDatum> = Vec::with_capacity(val_indices.len());
    let mut oracle_limited = 0usize;
    let mut warmed = false;
    let mut reads: Vec<u64> = Vec::new();
    for chunk in val_indices.chunks(gen_batch) {
        let r0 = runtime_read_count();
        let cand = trainer
            .generate_candidates(&val_set, chunk, &gen_config)
            .unwrap_or_else(|e| fail(format!("generate_candidates: {e}")));
        let delta = runtime_read_count() - r0;
        if warmed {
            reads.push(delta as u64);
            if delta != 1 {
                fail(format!(
                    "generate call read {delta} time(s), expected exactly 1 per call after warm-up"
                ));
            }
        } else {
            warmed = true;
        }
        let rows = eval_records(&cand).unwrap_or_else(|e| fail(format!("eval_records: {e}")));
        let limits =
            mamba3::models::ms2::grammar::Limits::new(cand.max_atoms, cand.max_ring_closures)
                .unwrap_or_else(|e| fail(format!("batch limits rejected: {e}")));
        for (pos, &idx) in chunk.iter().enumerate() {
            let entry = &val_set.spectra[idx];
            let lu = entry.labels.as_ref().map(label_union).unwrap_or(0);
            let (ou, limited) = recipe_fragments_union(&entry.parent);
            if limited {
                oracle_limited += 1;
            }
            data.push(spectrum_datum(
                &entry.parent,
                entry.molecule,
                &rows[pos],
                limits,
                lu,
                ou,
            ));
        }
    }

    // Donor-peaks ablation (evaluation-only input ablation): the SAME loaded
    // checkpoint — which must be trained with control none — generates again
    // with each validation spectrum's peaks replaced by a donor spectrum's
    // from another molecule. Donor assembly is the trainer's molecule-aware
    // path (`donor_map` under the trainer seed, the same one evaluation
    // under the shuffled control takes): a per-request shuffled control that
    // needs no retraining. Precursor and metadata stay the spectrum's own.
    let mut donor_data: Option<Vec<FgSpectrumDatum>> = None;
    if donor_peaks {
        if train_cfg.control != Control::None {
            fail(format!(
                "--donor-peaks needs a checkpoint trained with control none, got {:?}",
                train_cfg.control
            ));
        }
        let mut donor_cfg = gen_config.clone();
        donor_cfg.control = Control::ShuffledSpectrum;
        if let Err(e) = donor_cfg.validate(model_atoms, model_closures) {
            fail(format!("donor generation config rejected: {e}"));
        }
        let mut dd: Vec<FgSpectrumDatum> = Vec::with_capacity(val_indices.len());
        for chunk in val_indices.chunks(gen_batch) {
            let cand = trainer
                .generate_candidates(&val_set, chunk, &donor_cfg)
                .unwrap_or_else(|e| fail(format!("donor generate_candidates: {e}")));
            let rows =
                eval_records(&cand).unwrap_or_else(|e| fail(format!("donor eval_records: {e}")));
            let limits =
                mamba3::models::ms2::grammar::Limits::new(cand.max_atoms, cand.max_ring_closures)
                    .unwrap_or_else(|e| fail(format!("donor batch limits rejected: {e}")));
            for (pos, &idx) in chunk.iter().enumerate() {
                let entry = &val_set.spectra[idx];
                let lu = entry.labels.as_ref().map(label_union).unwrap_or(0);
                let (ou, _) = recipe_fragments_union(&entry.parent);
                dd.push(spectrum_datum(
                    &entry.parent,
                    entry.molecule,
                    &rows[pos],
                    limits,
                    lu,
                    ou,
                ));
            }
        }
        donor_data = Some(dd);
    }

    // Prior baseline on train molecules (one parent set per molecule).
    let mut train_mol: BTreeMap<usize, u32> = BTreeMap::new();
    for s in &train_set.spectra {
        train_mol
            .entry(s.molecule)
            .or_insert_with(|| functional_groups_v4(&s.parent).mask());
    }
    let train_masks: Vec<u32> = train_mol.values().copied().collect();
    let prior = choose_prior(&train_masks);

    let mut ks: Vec<usize> = [1, 4, 8].into_iter().filter(|x| *x <= k).collect();
    if !ks.contains(&k) {
        ks.push(k);
        ks.sort_unstable();
    }
    let eval_report = evaluate_fg(&data, donor_data.as_deref(), &ks, bootstrap, seed);

    // Reference rows by the same set code (full / specific / heteroatom).
    let molecules: Vec<usize> = data.iter().map(|d| d.molecule).collect();
    let truths: Vec<u32> = data.iter().map(|d| d.parent_mask).collect();
    let prior_pred = prior.set.iter().fold(0u32, |m, id| m | (1u32 << (id - 1)));
    let ref_rows = |preds: Vec<u32>| {
        use mamba3::models::ms2::functional_groups::{FULL_MASK, HETEROATOM_MASK, SPECIFIC_MASK};
        let full = evaluate_sets(&preds, &truths, &molecules, bootstrap, seed);
        let mask = |vocab: u32| -> (Vec<u32>, Vec<u32>) {
            (
                preds.iter().map(|p| p & vocab).collect(),
                truths.iter().map(|t| t & vocab).collect(),
            )
        };
        let (sp, st) = mask(SPECIFIC_MASK);
        let specific = evaluate_sets(&sp, &st, &molecules, bootstrap, seed);
        let (hp, ht) = mask(HETEROATOM_MASK);
        let heteroatom = evaluate_sets(&hp, &ht, &molecules, bootstrap, seed);
        let _ = FULL_MASK;
        (full, specific, heteroatom)
    };
    let (prior_full, prior_spec, prior_het) = ref_rows(vec![prior_pred; data.len()]);
    let formula_preds: Vec<u32> = data
        .iter()
        .map(|d| {
            formula_aware_set(&prior.set, d.top_formula)
                .iter()
                .fold(0u32, |m, id| m | (1u32 << (id - 1)))
        })
        .collect();
    let (formula_full, formula_spec, formula_het) = ref_rows(formula_preds);
    let label_preds: Vec<u32> = data.iter().map(|d| d.label_union).collect();
    let (label_full, label_spec, label_het) = ref_rows(label_preds);
    let recipe_preds: Vec<u32> = data.iter().map(|d| d.oracle_union).collect();
    let (recipe_full, recipe_spec, recipe_het) = ref_rows(recipe_preds);

    // The determined rule itself does not cap recall for the model's fragment
    // family (any group instance fits with its closing neighbours in a
    // 16-atom fragment); verify on the evaluated validation parents and
    // report the count instead of assuming zero.
    let closing_fragment_not_found: usize = {
        let parents: Vec<mamba3::models::ms2::MolGraph> = val_indices
            .iter()
            .map(|&idx| {
                let e = &val_set.spectra[idx];
                mamba3::models::ms2::MolGraph::new(
                    e.parent.atoms().to_vec(),
                    e.parent.bonds().to_vec(),
                )
                .unwrap()
            })
            .collect();
        closing_fragment_not_found(&parents)
    };
    let sizes = candidate_size_dist(&data);
    let vocabulary: Vec<serde_json::Value> = FG_NAMES
        .iter()
        .enumerate()
        .map(|(i, name)| serde_json::json!({"id": i + 1, "name": name}))
        .collect();

    let ckpt_bytes = std::fs::read(&load)
        .unwrap_or_else(|e| fail(format!("cannot hash {}: {e}", load.display())));
    let report = serde_json::json!({
        "versions": {"functional_groups": FG_VERSION, "chemistry": CHEMISTRY_VERSION},
        "checkpoint": {
            "file": load.display().to_string(),
            "sha256": experiment::sha256_hex(&ckpt_bytes),
            "steps": trainer.step_count(),
            "control": format!("{:?}", train_cfg.control),
            "formula_source": format!("{:?}", train_cfg.formula_source),
            "formula_window": train_cfg.formula_window,
        },
        "exports": {
            "train": {"file": train_path.display().to_string(), "source_sha256": train_set.source_sha256,
                      "spectra": train_set.spectra.len(), "molecules": train_set.molecules.len(),
                      "train_molecules_used": train_masks.len()},
            "validation": {"file": validation_path.display().to_string(), "source_sha256": val_set.source_sha256,
                           "spectra": val_set.spectra.len(), "molecules": val_set.molecules.len(),
                           "spectra_evaluated": data.len()},
        },
        "generation": {
            "k": k, "batch": batch, "gen_batch": gen_batch, "seed": seed,
            "allocation": format!("{allocation:?}"), "identity": format!("{identity:?}"),
            "control": format!("{control:?}"),
            "enum_dispatch_visits_max": gen_config.enum_dispatch_visits_max,
            "reads_per_generate_call": reads,
        },
        "split_sizes": {"train_molecules": train_masks.len(), "validation_spectra": data.len()},
        "vocabulary": {"version": FG_VERSION, "names": FG_NAMES, "entries": vocabulary,
            "specific": "every type except carbonyl (id 1)",
            "heteroatom": "every type except carbonyl, alkene, alkyne, arene_ring (ids 1, 14, 15, 24)"},
        "candidate_sizes": sizes,
        "closing_fragment_not_found": closing_fragment_not_found,
        "model": eval_report,
        "model_donor_peaks": if donor_peaks {
            serde_json::json!({"note": "same checkpoint with donor peaks (evaluation-only input ablation: precursor and metadata stay the spectrum's own); paired own-donor intervals under model[k].donor.diff_*",
                "rows": eval_report.iter().map(|r| serde_json::json!({"k": r.k, "donor": r.donor})).collect::<Vec<_>>()})
        } else {
            serde_json::json!(null)
        },
        "donor_peaks": donor_peaks,
        "references": {
            "prior": {"tau": prior.tau, "set": prior.set, "full": prior_full, "specific": prior_spec, "heteroatom": prior_het},
            "candidate_formula_prior": {"description": "plain prior set minus types whose needed elements are absent from the spectrum's best-formula_log_prob eligible record (top-ranked formula hypothesis among allocated trajectories; unallocated hypotheses are not exposed)", "full": formula_full, "specific": formula_spec, "heteroatom": formula_het},
            "label_ceiling": {"full": label_full, "specific": label_spec, "heteroatom": label_het},
            "recipe_fragments": {"description": "union over the label recipe's candidate fragments (at most two bond cuts, 3 to 16 atoms): coverage of the recipe, not a ceiling for the model", "full": recipe_full, "specific": recipe_spec, "heteroatom": recipe_het,
                       "limited_spectra": oracle_limited},
        },
        "caveats": [
            "overlapping structural motifs, not a chemist's unique classification",
            "kekule-invariant by the delocalised-bond rule, NOT tautomer-invariant (2-pyridone and 2-hydroxypyridine differ)",
            "instance precision checks type presence in the parent, not location or multiplicity",
            "replay failures occupy top-k slots as empty predictions",
            "recall mixes model coverage with the conservative determined rule",
            "the reference rows that use the parent structure (label_ceiling, recipe_fragments) are not baselines",
        ],
        "evaluation": "functional groups determined inside each predicted graph (ms2-fg-v4), compared as type sets with the true parent's groups; parent structure used only as the label",
        "seconds": started.elapsed().as_secs_f64(),
    });
    std::fs::write(
        &out,
        serde_json::to_string_pretty(&report).unwrap_or_else(|e| fail(format!("serialize: {e}"))),
    )
    .unwrap_or_else(|e| fail(format!("cannot write {}: {e}", out.display())));

    // Console tables: for the model and each reference row, micro P/R/F1 AND
    // macro F1 AND the empty-P fraction, for the three subsets at k = 8 (or
    // the last k), plus the per-type table for the model next to the prior.
    // `None` prints as n/a.
    let fmt = |v: &MetricPoint| -> String {
        match v.point {
            Some(x) => format!("{x:.3}"),
            None => "n/a".to_string(),
        }
    };
    let row = |name: &str, s: &mamba3::models::ms2::functional_groups_eval::SetMetrics| {
        println!(
            "{name:22} P={:>5} R={:>5} F1={:>5} macroF1={:>5} emptyP={:>5}",
            fmt(&s.micro_precision),
            fmt(&s.micro_recall),
            fmt(&s.micro_f1),
            fmt(&s.macro_f1),
            fmt(&s.empty_p),
        );
    };
    println!("functional groups ({FG_VERSION}) on {} spectra", data.len());
    println!(
        "candidate sizes: n={} f1={:?} f2={:?} f3-5={:?} f6-9={:?} f10-16={:?} closing_fragment_not_found={}",
        sizes.n,
        sizes.frac_1,
        sizes.frac_2,
        sizes.frac_3_5,
        sizes.frac_6_9,
        sizes.frac_10_16,
        closing_fragment_not_found
    );
    let k8 = eval_report
        .iter()
        .find(|r| r.k == 8)
        .or(eval_report.last())
        .unwrap();
    for subset in ["full", "specific", "heteroatom"] {
        let (mset, pset, fset, lset, rset) = match subset {
            "full" => (
                &k8.full.set,
                &prior_full,
                &formula_full,
                &label_full,
                &recipe_full,
            ),
            "specific" => (
                &k8.specific.set,
                &prior_spec,
                &formula_spec,
                &label_spec,
                &recipe_spec,
            ),
            _ => (
                &k8.heteroatom.set,
                &prior_het,
                &formula_het,
                &label_het,
                &recipe_het,
            ),
        };
        println!("k=8 {subset}: model vs references (micro P/R/F1, macro F1, emptyP):");
        row("  model", mset);
        row("  prior", pset);
        row("  cand_formula_prior", fset);
        row("  label_ceiling", lset);
        row("  recipe_fragments", rset);
        println!("k=8 {subset} model min_atoms_3:");
        let m3 = match subset {
            "full" => &k8.full.set_min_atoms_3,
            "specific" => &k8.specific.set_min_atoms_3,
            _ => &k8.heteroatom.set_min_atoms_3,
        };
        row("  model_min3", m3);
        // Donor-peaks ablation row plus paired own − donor intervals.
        if let Some(donor) = &k8.donor {
            let (dset, diff) = match subset {
                "full" => (&donor.full.set, &donor.diff_full),
                "specific" => (&donor.specific.set, &donor.diff_specific),
                _ => (&donor.heteroatom.set, &donor.diff_heteroatom),
            };
            row("  model_donor_peaks", dset);
            let fmt_d = |v: &mamba3::models::ms2::functional_groups_eval::DiffPoint| -> String {
                match (v.point, v.lo, v.hi) {
                    (Some(p), Some(lo), Some(hi)) => format!("{p:+.3} [{lo:+.3},{hi:+.3}]"),
                    (Some(p), _, _) => format!("{p:+.3}"),
                    _ => "n/a".to_string(),
                }
            };
            println!(
                "  own-donor paired: dP={} dR={} dF1={} dmacroF1={} demptyP={}",
                fmt_d(&diff.micro_precision),
                fmt_d(&diff.micro_recall),
                fmt_d(&diff.micro_f1),
                fmt_d(&diff.macro_f1),
                fmt_d(&diff.empty_p),
            );
        }
    }
    {
        println!(
            "per-type table at k={} (full): model vs prior (intervals: molecule bootstrap; dR: paired own-donor recall diff)",
            k8.k
        );
        println!(
            "{:<26} {:>5} {:>5} {:>5} {:>13} {:>13} {:>16} | {:>5} {:>5} {:>5} {:>6} {:>6}",
            "type",
            "m_true",
            "m_pred",
            "m_tp",
            "m_prec",
            "m_rec",
            "m_dR",
            "p_pred",
            "p_tp",
            "p_tr",
            "p_prec",
            "p_rec"
        );
        let pft = &prior_full.per_type;
        let fp = |v: &Option<f64>| match v {
            Some(x) => format!("{x:.3}"),
            None => "n/a".to_string(),
        };
        let fpair = |lo: &Option<f64>, hi: &Option<f64>| match (lo, hi) {
            (Some(l), Some(h)) => format!("[{l:.2},{h:.2}]"),
            _ => "n/a".to_string(),
        };
        let diff_rows: Option<&Vec<mamba3::models::ms2::functional_groups_eval::DiffPoint>> =
            k8.donor.as_ref().map(|d| &d.diff_full.per_type_recall_diff);
        for t in &k8.full.set.per_type {
            let p = &pft[t.id - 1];
            let dr = match diff_rows {
                Some(rows) => {
                    let dd = &rows[t.id - 1];
                    match (dd.point, dd.lo, dd.hi) {
                        (Some(x), Some(l), Some(h)) => format!("{x:+.2}[{l:.2},{h:.2}]"),
                        _ => "n/a".to_string(),
                    }
                }
                None => "n/a".to_string(),
            };
            println!(
                "{:<26} {:>5} {:>5} {:>5} {:>13} {:>13} {:>16} | {:>5} {:>5} {:>5} {:>6} {:>6}",
                t.name,
                t.true_count,
                t.predicted,
                t.tp,
                format!(
                    "{} {}",
                    fp(&t.precision),
                    fpair(&t.precision_lo, &t.precision_hi)
                ),
                format!("{} {}", fp(&t.recall), fpair(&t.recall_lo, &t.recall_hi)),
                dr,
                p.predicted,
                p.tp,
                p.true_count,
                fp(&p.precision),
                fp(&p.recall)
            );
        }
    }
}
