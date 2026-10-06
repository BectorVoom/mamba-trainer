//! Bounded molecular-completion ambiguity driver.
//!
//! Usage:
//! ```text
//! cargo run --release --no-default-features --features cpu --example ms2_completion -- \
//!   --fixtures experiments/molecular_completion/20261004_completion_ambiguity/fixtures.json \
//!   --out experiments/molecular_completion/20261004_completion_ambiguity/completion_report.json
//! ```
//!
//! Reads the provenance fixtures, runs the bounded audit per fixture, checks
//! the pinned expectations, and writes one JSON report plus a printed table.

use std::fs;
use std::path::PathBuf;

use mamba3::models::ms2::completion::{self, CompletionBudgets, load_fixture_set, run};

fn usage() -> ! {
    eprintln!(
        "usage: ms2_completion --fixtures <fixtures.json> --out <report.json> [--seed N] (default: fixtures dir)"
    );
    std::process::exit(2)
}

fn main() -> mamba3::error::Result<()> {
    let mut fixtures = None;
    let mut out = None;
    let mut seed = 1u64;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--fixtures" => fixtures = Some(args.next().unwrap_or_else(|| usage())),
            "--out" => out = Some(args.next().unwrap_or_else(|| usage())),
            "--seed" => {
                seed = args
                    .next()
                    .unwrap_or_else(|| usage())
                    .parse()
                    .unwrap_or_else(|_| {
                        eprintln!("--seed must be a non-negative integer");
                        usage()
                    });
            }
            other => {
                eprintln!("unknown arg: {other}");
                usage();
            }
        }
    }
    let (Some(fixtures), Some(out)) = (fixtures, out) else {
        usage();
    };
    let text = fs::read_to_string(&fixtures)?;
    let fixtures_list = load_fixture_set(&text)?;
    let budgets = CompletionBudgets::default();
    let mut rows = Vec::new();
    let mut failures = Vec::new();
    println!("name\tstatus\tmass_status\tformulas\tgraphs\trecovery\treasons\telapsed_ms");
    for f in &fixtures_list {
        let report = run(&f.query, &budgets, seed)?;
        let expected_match = report.status == f.expected.status
            && report.mass_status == f.expected.mass_status
            && report.unique_graphs == f.expected.unique_graphs
            && report.accepted_formulas == f.expected.accepted_formulas
            && report.ambiguous_formulas == f.expected.ambiguous_formulas
            && report.certifies_zero == f.expected.certifies_zero
            && f.expected
                .termination_reasons
                .iter()
                .all(|r| report.termination_reasons.contains(r))
            && match (f.expected.recovery, report.recovery) {
                (Some(a), Some(b)) => a == b,
                (None, _) => true,
                _ => false,
            };
        if !expected_match {
            failures.push(format!(
                "{}: expected {:?} got status={} mass={} graphs={} formulas={:?}",
                f.name,
                f.expected.status,
                report.status,
                report.mass_status,
                report.unique_graphs,
                report.accepted_formulas.len()
            ));
        }
        println!(
            "{}\t{}\t{}\t{}\t{}\t{:?}\t{:?}\t{}",
            f.name,
            report.status,
            report.mass_status,
            report.accepted_formulas.len(),
            report.unique_graphs,
            report.recovery,
            report.termination_reasons,
            report.elapsed_ms
        );
        rows.push(serde_json::json!({"name": f.name, "report": report}));
    }
    let doc = serde_json::json!({
        "protocol": completion::COMPLETION_VERSION,
        "fixtures": PathBuf::from(&fixtures).display().to_string(),
        "seed": seed,
        "budgets": serde_json::json!({
            "formula_visits": budgets.formula_visits,
            "graph_extensions": budgets.graph_extensions,
            "embedding_nodes": budgets.embedding_nodes,
            "canonical_expansions": budgets.canonical_expansions,
            "retained_graphs": budgets.retained_graphs,
            "memory_bytes": budgets.memory_bytes,
            "watchdog_s": budgets.watchdog.as_secs(),
        }),
        "rows": rows,
        "failures": failures,
    });
    fs::write(&out, serde_json::to_string_pretty(&doc)?)?;
    if !failures.is_empty() {
        for f in &failures {
            eprintln!("FAIL {f}");
        }
        std::process::exit(1);
    }
    println!("all {} fixtures matched expectations", fixtures_list.len());
    Ok(())
}
