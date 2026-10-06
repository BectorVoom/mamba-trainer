//! Supervised molecular-completion data coverage report.
//!
//! Usage:
//! ```text
//! cargo run --release --no-default-features --features cpu \
//!   --example molecular_completion_data_report -- \
//!   --train <export.json> --validation <export.json> \
//!   [--out report.json] [--work-limit N]
//! ```
//!
//! For each limit pair (16, 4), (24, 4), (24, 6), (32, 6) this loads the
//! train and validation [`CompletionSet`](mamba3::models::ms2::completion_data::CompletionSet)
//! and reports aggregates only (the export data is CC BY-NC: no keys, no
//! SMILES, no per-molecule rows): molecules read and kept, skips by reason,
//! trace-length min/median/max, canonicalization expansions median / 99th
//! percentile / max with the count above
//! [`CANONICAL_WORK_LIMIT`](mamba3::models::ms2::grammar::CANONICAL_WORK_LIMIT),
//! load wall time, the validation-in-train overlap (strict and skeleton),
//! skeleton canonicalization failures, and extraction statistics for
//! [`ExtractionConfig::default()`](mamba3::models::ms2::completion_data::ExtractionConfig)
//! with seed 1, draw 0 over the validation set. The run exits non-zero when
//! any extracted pattern is not contained in its target.

use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use mamba3::models::ms2::completion::contains_pattern;
use mamba3::models::ms2::completion_data::{
    COMPLETION_DATA_VERSION, CompletionSet, ExtractionConfig,
};
use mamba3::models::ms2::contain::Containment;
use mamba3::models::ms2::dataset::ExportFile;
use mamba3::models::ms2::grammar::{CANONICAL_WORK_LIMIT, Limits, canonical_trace};

/// Default canonicalization budget: large enough that expansions above
/// [`CANONICAL_WORK_LIMIT`] are measurable instead of censored.
const DEFAULT_WORK_LIMIT: usize = 5_000_000;

/// Containment budget for the extraction audit.
const CONTAIN_NODES: usize = 100_000;

fn usage() -> ! {
    eprintln!(
        "usage: molecular_completion_data_report --train <export.json> --validation <export.json> [--out report.json] [--work-limit N]"
    );
    std::process::exit(2)
}

/// Median of a sorted non-empty slice as `f64`.
fn median(sorted: &[usize]) -> f64 {
    assert!(!sorted.is_empty(), "median needs values");
    let n = sorted.len();
    if n % 2 == 1 {
        sorted[n / 2] as f64
    } else {
        (sorted[n / 2 - 1] as f64 + sorted[n / 2] as f64) / 2.0
    }
}

/// Nearest-rank 99th percentile of a sorted non-empty slice.
fn percentile99(sorted: &[usize]) -> usize {
    assert!(!sorted.is_empty(), "percentile needs values");
    let rank = (99 * sorted.len()).div_ceil(100);
    sorted[rank.saturating_sub(1).min(sorted.len() - 1)]
}

fn main() -> mamba3::error::Result<()> {
    let mut train: Option<String> = None;
    let mut validation: Option<String> = None;
    let mut out: Option<String> = None;
    let mut work_limit = DEFAULT_WORK_LIMIT;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--train" => train = Some(args.next().unwrap_or_else(|| usage())),
            "--validation" => validation = Some(args.next().unwrap_or_else(|| usage())),
            "--out" => out = Some(args.next().unwrap_or_else(|| usage())),
            "--work-limit" => {
                work_limit = args
                    .next()
                    .unwrap_or_else(|| usage())
                    .parse()
                    .unwrap_or_else(|_| {
                        eprintln!("--work-limit must be a non-negative integer");
                        usage()
                    });
            }
            other => {
                eprintln!("unknown arg: {other}");
                usage();
            }
        }
    }
    let (Some(train), Some(validation)) = (train, validation) else {
        usage();
    };
    let train_file = ExportFile::load(PathBuf::from(&train).as_path())?;
    let validation_file = ExportFile::load(PathBuf::from(&validation).as_path())?;
    let pairs = [(16usize, 4usize), (24, 4), (24, 6), (32, 6)];
    let mut limit_rows = Vec::with_capacity(pairs.len());
    let mut containment_ok = true;
    println!(
        "limits\tset\tread\tkept\tskipped\tlen_min\tlen_med\tlen_max\texp_med\texp_p99\texp_max\texp_over{s}\tskel_fail\tload_s\tover_strict\tover_skel\tpat_n\tpat_atoms\tcover\tcontained",
        s = CANONICAL_WORK_LIMIT
    );
    for (max_atoms, max_closures) in pairs {
        let limits = Limits::new(max_atoms, max_closures)?;
        let started_train = Instant::now();
        let train_set = CompletionSet::from_export(&train_file, limits, work_limit)?;
        let train_secs = started_train.elapsed().as_secs_f64();
        let started_validation = Instant::now();
        let validation_set = CompletionSet::from_export(&validation_file, limits, work_limit)?;
        let validation_secs = started_validation.elapsed().as_secs_f64();
        let train_stats = set_stats(&train_set, limits, work_limit, train_secs);
        let validation_stats = set_stats(&validation_set, limits, work_limit, validation_secs);
        // Validation examples found in the train set.
        let (overlap_strict, overlap_skeleton) = validation_set.overlap(&train_set);
        // Extraction audit over the validation set (default config, seed 1,
        // draw 0): every pattern must embed in its target.
        let config = ExtractionConfig::default();
        let mut pattern_total = 0usize;
        let mut pattern_atoms = 0usize;
        let mut covered_sum = 0.0f64;
        let mut contained_examples = 0usize;
        for example in &validation_set.examples {
            let patterns = example.patterns(&config, 1, 0)?;
            pattern_total += patterns.len();
            let mut covered = vec![false; example.target.atoms().len()];
            let mut all_contained = true;
            for p in &patterns {
                pattern_atoms += p.graph.atoms().len();
                for a in &p.parent_atoms {
                    covered[*a] = true;
                }
                if contains_pattern(&example.target, &p.graph, CONTAIN_NODES)
                    != Containment::Contained
                {
                    all_contained = false;
                }
            }
            if all_contained {
                contained_examples += 1;
            }
            let hit = covered.iter().filter(|&&c| c).count();
            covered_sum += hit as f64 / example.target.atoms().len().max(1) as f64;
        }
        let n_validation = validation_set.examples.len().max(1);
        let extraction = serde_json::json!({
            "mean_patterns": pattern_total as f64 / n_validation as f64,
            "mean_pattern_atoms": if pattern_total == 0 { 0.0 } else { pattern_atoms as f64 / pattern_total as f64 },
            "mean_covered_fraction": covered_sum / n_validation as f64,
            "all_contained_fraction": contained_examples as f64 / n_validation as f64,
        });
        if contained_examples != validation_set.examples.len() {
            containment_ok = false;
        }
        println!(
            "({max_atoms},{max_closures})\ttrain\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.2}\t{}\t{}\t-\t-\t-\t-",
            train_stats["read"],
            train_stats["kept"],
            train_stats["skipped"],
            train_stats["trace_len_min"],
            train_stats["trace_len_median"],
            train_stats["trace_len_max"],
            train_stats["expansions_median"],
            train_stats["expansions_p99"],
            train_stats["expansions_max"],
            train_stats["expansions_above_canonical_work_limit"],
            train_stats["skeleton_failures"],
            train_secs,
            overlap_strict,
            overlap_skeleton,
        );
        println!(
            "({max_atoms},{max_closures})\tvalidation\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.2}\t{}\t{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}",
            validation_stats["read"],
            validation_stats["kept"],
            validation_stats["skipped"],
            validation_stats["trace_len_min"],
            validation_stats["trace_len_median"],
            validation_stats["trace_len_max"],
            validation_stats["expansions_median"],
            validation_stats["expansions_p99"],
            validation_stats["expansions_max"],
            validation_stats["expansions_above_canonical_work_limit"],
            validation_stats["skeleton_failures"],
            validation_secs,
            overlap_strict,
            overlap_skeleton,
            extraction["mean_patterns"],
            extraction["mean_pattern_atoms"],
            extraction["mean_covered_fraction"],
            extraction["all_contained_fraction"],
        );
        limit_rows.push(serde_json::json!({
            "max_atoms": max_atoms,
            "max_closures": max_closures,
            "train": train_stats,
            "validation": validation_stats,
            "overlap_validation_in_train": {
                "strict": overlap_strict,
                "skeleton": overlap_skeleton,
            },
            "extraction_validation_default_seed1_draw0": extraction,
        }));
    }
    if let Some(out) = out {
        let doc = serde_json::json!({
            "protocol": COMPLETION_DATA_VERSION,
            "train": train,
            "validation": validation,
            "work_limit": work_limit,
            "canonical_work_limit": CANONICAL_WORK_LIMIT,
            "limits": limit_rows,
        });
        fs::write(&out, serde_json::to_string_pretty(&doc)?)?;
        println!("wrote {out}");
    }
    if !containment_ok {
        eprintln!("FAIL: some extracted pattern is not contained in its target");
        std::process::exit(1);
    }
    Ok(())
}

/// Aggregate statistics of one built set (no molecule rows: aggregates only).
fn set_stats(
    set: &CompletionSet,
    limits: Limits,
    work_limit: usize,
    load_secs: f64,
) -> serde_json::Value {
    let mut trace_lens: Vec<usize> = set.examples.iter().map(|e| e.trace.len()).collect();
    trace_lens.sort_unstable();
    // Canonicalization work behind the kept examples: re-run the canonical
    // trace of each kept target under the same budget. Kept targets
    // canonicalized within budget during the load, so this succeeds.
    let mut expansions: Vec<usize> = Vec::with_capacity(set.examples.len());
    for example in &set.examples {
        let canonical = canonical_trace(&example.target, limits, work_limit)
            .expect("kept targets canonicalize within the report budget");
        expansions.push(canonical.expansions);
    }
    expansions.sort_unstable();
    let skeleton_failures = set
        .examples
        .iter()
        .filter(|e| e.skeleton_trace.is_empty())
        .count();
    let read: usize = set.examples.len() + set.skipped.values().sum::<u64>() as usize;
    serde_json::json!({
        "read": read,
        "kept": set.examples.len(),
        "skipped": set.skipped,
        "trace_len_min": trace_lens.first().copied().unwrap_or(0),
        "trace_len_median": if trace_lens.is_empty() { 0.0 } else { median(&trace_lens) },
        "trace_len_max": trace_lens.last().copied().unwrap_or(0),
        "expansions_median": if expansions.is_empty() { 0.0 } else { median(&expansions) },
        "expansions_p99": if expansions.is_empty() { 0 } else { percentile99(&expansions) },
        "expansions_max": expansions.last().copied().unwrap_or(0),
        "expansions_above_canonical_work_limit": expansions.iter().filter(|&&e| e > CANONICAL_WORK_LIMIT).count(),
        "skeleton_failures": skeleton_failures,
        "load_seconds": load_secs,
    })
}
