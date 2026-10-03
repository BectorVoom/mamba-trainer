//! Rust label report over an `export_casmi.py` file: the same per-spectrum
//! rows and aggregates as `tools/ms2/label_report.py`, with canonical-trace
//! graph identity.
//!
//! Usage: `cargo run --release --no-default-features --features cpu
//! --example ms2_label_report -- --input <export.json> --out <report.json>`.
//!
//! The example needs no device and constructs none.

use std::collections::{BTreeSet, HashMap};
use std::path::PathBuf;
use std::time::Instant;

use mamba3::models::ms2::dataset::{ExportFile, percentile};
use mamba3::models::ms2::experiment::{LABEL_PPM_TENTHS, label_export_spectrum};
use mamba3::models::ms2::grammar::{Limits, replay};
use mamba3::models::ms2::targets::{Candidates, RecipeLimits};

/// Summary of one key across spectra: mean and percentiles.
fn summarize(values: &[f64]) -> serde_json::Value {
    if values.is_empty() {
        return serde_json::json!({"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0});
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    serde_json::json!({
        "mean": mean,
        "p50": percentile(&sorted, 50.0),
        "p95": percentile(&sorted, 95.0),
        "max": sorted[sorted.len() - 1],
    })
}

fn usage() -> ! {
    eprintln!("usage: ms2_label_report --input <export.json> --out <report.json>");
    std::process::exit(2);
}

fn main() {
    let started = Instant::now();
    let mut input: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--input" => input = args.next().map(PathBuf::from),
            "--out" => out = args.next().map(PathBuf::from),
            _ => usage(),
        }
    }
    let Some(input) = input else { usage() };
    let Some(out) = out else { usage() };

    let file = ExportFile::load(&input).unwrap_or_else(|e| {
        eprintln!("ms2_label_report: cannot read {}: {e}", input.display());
        std::process::exit(1);
    });
    let name = input
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| input.display().to_string());

    let mut rows: Vec<serde_json::Value> = Vec::new();
    let mut canonicalization_failures = 0usize;
    let mut max_expansions = 0usize;
    // P1.9: every training target must replay legally under the grammar with the
    // parent's composition as its budget, and end stopped within the step limit.
    let mut targets_replayed = 0usize;
    let mut replay_failures = 0usize;
    let mut longest_trace = 0usize;
    for mol in &file.molecules {
        let graph = mol.graph().unwrap_or_else(|e| {
            eprintln!("ms2_label_report: molecule {}: {e}", mol.key);
            std::process::exit(1);
        });
        // One preparation per molecule: canonicalization is spectrum-independent.
        let candidates = Candidates::new(&graph, &RecipeLimits::V0).unwrap_or_else(|e| {
            eprintln!("ms2_label_report: molecule {}: {e}", mol.key);
            std::process::exit(1);
        });
        canonicalization_failures += candidates.canonicalization_failures();
        max_expansions = max_expansions.max(candidates.max_expansions());
        for s in &mol.spectra {
            let peaks = s.peaks().unwrap_or_else(|e| {
                eprintln!("ms2_label_report: row {}: {e}", s.row);
                std::process::exit(1);
            });
            let labels = label_export_spectrum(&candidates, s).unwrap_or_else(|e| {
                eprintln!("ms2_label_report: row {}: {e}", s.row);
                std::process::exit(1);
            });
            for target in &labels.targets {
                targets_replayed += 1;
                longest_trace = longest_trace.max(target.trace.len());
                let legal = replay(&target.trace, Limits::V0, Some(graph.composition()))
                    .map(|state| state.stopped())
                    .unwrap_or(false);
                if !legal || target.trace.len() > Limits::V0.max_steps() {
                    replay_failures += 1;
                }
            }
            let explained: BTreeSet<u32> = labels.explained_peaks.iter().copied().collect();
            let by_id: HashMap<u32, f64> = peaks.iter().map(|p| (p.id, p.intensity)).collect();
            let total: f64 = peaks.iter().map(|p| p.intensity).sum();
            let explained_intensity = if total > 0.0 {
                explained.iter().map(|id| by_id[id]).sum::<f64>() / total
            } else {
                0.0
            };
            // One object per kept target, sorted by embeddings, exactly as
            // `tools/ms2/label_report.py` writes `targets_detail`.
            let mut detail: Vec<(Vec<Vec<usize>>, serde_json::Value)> = labels
                .targets
                .iter()
                .map(|t| {
                    let mut embeddings: Vec<Vec<usize>> = t
                        .embeddings
                        .iter()
                        .map(|&i| {
                            let mut atoms = labels.embeddings[i].atoms.clone();
                            atoms.sort();
                            atoms
                        })
                        .collect();
                    embeddings.sort();
                    let anchors: Vec<serde_json::Value> = t
                        .anchors
                        .iter()
                        .map(|(id, shift)| serde_json::json!([id, shift]))
                        .collect();
                    let value = serde_json::json!({
                        "embeddings": embeddings,
                        "weight": t.weight,
                        "q": t.q,
                        "anchors": anchors,
                    });
                    (embeddings, value)
                })
                .collect();
            detail.sort_by(|a, b| a.0.cmp(&b.0));
            let detail: Vec<serde_json::Value> = detail.into_iter().map(|(_, v)| v).collect();
            rows.push(serde_json::json!({
                "row": s.row,
                "peaks": peaks.len(),
                "graphs": candidates.graphs(),
                "embeddings": candidates.embeddings().len(),
                "targets_before_cut": labels.targets_before_cut,
                "targets": labels.targets.len(),
                "explained_peaks": labels.explained_peaks.len(),
                "explained_intensity": explained_intensity,
                "dropped_weight": labels.dropped_weight,
                "ambiguous_hypotheses": labels.ambiguous_hypotheses,
                "q_sorted": labels.targets.iter().map(|t| t.q).collect::<Vec<_>>(),
                "anchors": labels.targets.iter().map(|t| t.anchors.len()).sum::<usize>(),
                "explained_peak_ids": labels.explained_peaks,
                "cut_is_tied": labels.cut_is_tied,
                "targets_detail": detail,
            }));
        }
    }
    rows.sort_by_key(|r| r["row"].as_u64().unwrap_or(u64::MAX));

    let get = |key: &str| -> Vec<f64> {
        rows.iter()
            .map(|r| r[key].as_f64().unwrap_or(0.0))
            .collect()
    };
    let spectra = rows.len();
    let labeled = rows
        .iter()
        .filter(|r| r["targets"].as_u64().unwrap_or(0) > 0)
        .count();
    let peaks_sum: u64 = rows.iter().map(|r| r["peaks"].as_u64().unwrap_or(0)).sum();
    let explained_sum: u64 = rows
        .iter()
        .map(|r| r["explained_peaks"].as_u64().unwrap_or(0))
        .sum();
    let ambiguous_sum: u64 = rows
        .iter()
        .map(|r| r["ambiguous_hypotheses"].as_u64().unwrap_or(0))
        .sum();
    let embeddings_sum: u64 = rows
        .iter()
        .map(|r| r["embeddings"].as_u64().unwrap_or(0))
        .sum();
    let graphs_sum: u64 = rows.iter().map(|r| r["graphs"].as_u64().unwrap_or(0)).sum();
    let aggregate = serde_json::json!({
        "spectra": spectra,
        "labeled_fraction": if spectra > 0 { labeled as f64 / spectra as f64 } else { 0.0 },
        "peaks": peaks_sum,
        "explained_peaks": explained_sum,
        "explained_intensity": summarize(&get("explained_intensity")),
        "targets": summarize(&get("targets")),
        "targets_before_cut": summarize(&get("targets_before_cut")),
        "dropped_weight": summarize(&get("dropped_weight")),
        "embeddings": summarize(&get("embeddings")),
        "graphs": summarize(&get("graphs")),
        "ambiguous_hypotheses": ambiguous_sum,
        "embeddings_per_graph": embeddings_sum as f64 / graphs_sum.max(1) as f64,
        "canonicalization_failures": canonicalization_failures,
        "max_expansions": max_expansions,
        "targets_replayed": targets_replayed,
        "replay_failures": replay_failures,
        "longest_trace": longest_trace,
        "seconds": started.elapsed().as_secs_f64(),
    });
    let report = serde_json::json!({
        "schema_version": 1,
        "implementation": "rust mamba3::models::ms2 (canonical-trace identity)",
        "input": name,
        "recipe": "q-cut-v1",
        "ppm_tenths": LABEL_PPM_TENTHS,
        "aggregate": aggregate,
        "spectra": rows,
    });
    if let Some(parent) = out.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent).unwrap_or_else(|e| {
            eprintln!("ms2_label_report: cannot create {}: {e}", parent.display());
            std::process::exit(1);
        });
    }
    std::fs::write(&out, serde_json::to_string_pretty(&report).unwrap()).unwrap_or_else(|e| {
        eprintln!("ms2_label_report: cannot write {}: {e}", out.display());
        std::process::exit(1);
    });
    println!("{}", serde_json::to_string_pretty(&aggregate).unwrap());
}
