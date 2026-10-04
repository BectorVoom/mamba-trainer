//! Host formula-search comparison report: indexed table versus bounded
//! closed-form-hydrogen enumeration, with and without train-fit pruning
//! (P4.9, host reference of P4.1/P4.3).
//!
//! Usage: `cargo run --release --no-default-features --features cpu
//! --example ms2_formula_report -- --train <export.json>
//! --validation <export.json> --table <formula_table.json>
//! [--ppm-tenths 200] [--nodes-max N] [--capacity M] [--ratio-margin Q]
//! [--ratio-train <export.json>] [--formula-window M] [--lane-visits-max N]
//! --out <report.json>`.
//!
//! The domain and the ratio bounds are derived from the fit export's
//! molecules only (`--ratio-train`, defaulting to `--train`): in-domain
//! graphs (those that build) contribute their compositions with margin 0 for
//! the domain and with `--ratio-margin` for the bounds. For every spectrum of
//! the validation export (and separately of the train export) the report
//! compares [`FormulaTable::window`] with [`enumerate`] (exact) and with
//! [`enumerate`] under the ratio bounds on gold-formula recall at each stage,
//! candidate-set sizes, work counters, exhaustion and absent rates, bytes and
//! per-spectrum wall-clock latency (single thread, `std::time::Instant`,
//! release build). The JSON holds aggregates only: no SMILES, no spectrum
//! ids, no peaks.
//!
//! The example needs no device and constructs none.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::time::Instant;

use mamba3::models::ms2::dataset::{ExportFile, percentile};
use mamba3::models::ms2::formula::{FormulaTable, WindowQuery};
use mamba3::models::ms2::formula_enum::{
    DeviceEnumLimits, EnumDomain, EnumLimits, EnumQuery, GoldStages, RatioBounds, enumerate,
    enumerate_device_order, gold_stages, validate_device_artifacts,
};
use mamba3::models::ms2::{Composition, composition_mass, parent_mass, tolerance};

/// Default precursor tolerance in tenths of a ppm (contracts §6).
const DEFAULT_PPM_TENTHS: u32 = 200;

/// Default scored-candidate capacity per spectrum of the device-order
/// twin (spec §1.4 `M`).
const DEFAULT_FORMULA_WINDOW: u32 = 2048;

/// Default per-lane visit budget of the device-order twin
/// (spec §1.4 `enum_lane_visits_max`).
const DEFAULT_LANE_VISITS_MAX: u32 = 65_536;

/// Ratio recall/size stages in application order, with their JSON keys:
/// the (i) bucketed caps, the (iii) rare stages, the six (ii) ratios in
/// `RATIO_FEATURES` order, the (iv) DBE bucket.
const RATIO_STAGES: [&str; 9] = [
    "after_ratio_cap",
    "after_ratio_rare",
    "after_ratio_hc",
    "after_ratio_nc",
    "after_ratio_oc",
    "after_ratio_hal",
    "after_ratio_s",
    "after_ratio_p",
    "after_ratio_dbe",
];

fn usage() -> ! {
    eprintln!(
        "usage: ms2_formula_report --train <export.json> --validation <export.json> \
         --table <formula_table.json> [--ppm-tenths N] [--nodes-max N] [--capacity M] \
         [--ratio-margin Q] [--ratio-train <export.json>] [--formula-window M] \
         [--lane-visits-max N] --out <report.json>"
    );
    std::process::exit(2);
}

fn parse_u64(text: &str, flag: &str) -> u64 {
    text.parse::<u64>().unwrap_or_else(|_| {
        eprintln!("ms2_formula_report: {flag} is not a u64 integer: {text:?}");
        std::process::exit(2);
    })
}

/// Checked `u64` to `u32` conversion: a value outside the `u32` range is a
/// usage error, never a silent truncation.
fn parse_u32(text: &str, flag: &str) -> u32 {
    let value = parse_u64(text, flag);
    match u32::try_from(value) {
        Ok(v) => v,
        Err(_) => {
            eprintln!("ms2_formula_report: {flag} {value} does not fit u32");
            std::process::exit(2);
        }
    }
}

/// Checked `u64` to `usize` conversion, likewise never truncating.
fn parse_usize(text: &str, flag: &str) -> usize {
    let value = parse_u64(text, flag);
    match usize::try_from(value) {
        Ok(v) => v,
        Err(_) => {
            eprintln!("ms2_formula_report: {flag} {value} does not fit usize");
            std::process::exit(2);
        }
    }
}

/// Checked `u64` to `u16` conversion, likewise never truncating.
fn parse_u16(text: &str, flag: &str) -> u16 {
    let value = parse_u64(text, flag);
    match u16::try_from(value) {
        Ok(v) => v,
        Err(_) => {
            eprintln!("ms2_formula_report: {flag} {value} does not fit u16");
            std::process::exit(2);
        }
    }
}

/// Accumulators for one method on one split (denominator: every spectrum).
#[derive(Default)]
struct MethodStats {
    /// Spectra evaluated.
    spectra: usize,
    /// Gold recall at each stage.
    window: usize,
    accepted: usize,
    accepted_or_ambiguous: usize,
    after_h_max: usize,
    after_parity: usize,
    after_dbe: usize,
    scored: usize,
    /// Ratio-only cumulative gold recall after each [`RATIO_STAGES`] stage.
    ratio_recall: [usize; 9],
    /// Joined rows per spectrum.
    joined: Vec<f64>,
    /// Ratio-only candidate sizes per spectrum: compositions reaching the
    /// verdict stage (past the DFS ratio caps), passing the verdict, past
    /// the exact filters, past each [`RATIO_STAGES`] stage, and scored.
    size_evaluated: Vec<f64>,
    size_verdict: Vec<f64>,
    size_exact: Vec<f64>,
    size_stages: Vec<[f64; 9]>,
    size_scored: Vec<f64>,
    /// Ratio-only DFS pruned branches per spectrum.
    pruned_cap: Vec<f64>,
    pruned_rare: Vec<f64>,
    /// Work per spectrum (one series per counter).
    work_primary: Vec<f64>,
    work_secondary: Vec<f64>,
    /// Spectra reporting exhaustion / absence.
    exhausted: usize,
    absent: usize,
    /// Per-spectrum wall-clock seconds.
    latency: Vec<f64>,
}

impl MethodStats {
    fn recall(&self, count: usize) -> f64 {
        if self.spectra == 0 {
            0.0
        } else {
            count as f64 / self.spectra as f64
        }
    }

    fn rate(&self, count: usize) -> f64 {
        self.recall(count)
    }
}

/// Accumulators for the device-order twin (`enum+ratio(device order)`) on
/// one split (denominator: every spectrum). Aggregates only.
#[derive(Default)]
struct DeviceStats {
    /// Spectra evaluated.
    spectra: usize,
    /// Gold recall in the scored prefix.
    scored: usize,
    /// Joined candidates per spectrum.
    joined: Vec<f64>,
    /// Lane visits per spectrum.
    visited: Vec<f64>,
    /// Rare-table lanes (`P`); fixed by the fit, recorded per spectrum.
    lanes: usize,
    /// Spectra reporting exhaustion / absence.
    exhausted: usize,
    absent: usize,
    /// Exhausted spectra whose gold is still in the scored prefix (hits).
    exhausted_hit: usize,
    /// Exhausted spectra whose gold is not in the scored prefix (misses).
    exhausted_miss: usize,
    /// Per-spectrum wall-clock seconds.
    latency: Vec<f64>,
}

impl DeviceStats {
    fn recall(&self, count: usize) -> f64 {
        if self.spectra == 0 {
            0.0
        } else {
            count as f64 / self.spectra as f64
        }
    }

    fn rate(&self, count: usize) -> f64 {
        self.recall(count)
    }
}

/// p50/p95/max summary of per-spectrum values.
fn summarize(values: &[f64]) -> serde_json::Value {
    if values.is_empty() {
        return serde_json::json!({"p50": 0.0, "p95": 0.0, "max": 0.0});
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    serde_json::json!({
        "p50": percentile(&sorted, 50.0),
        "p95": percentile(&sorted, 95.0),
        "max": sorted[sorted.len() - 1],
    })
}

fn main() {
    let mut train: Option<PathBuf> = None;
    let mut validation: Option<PathBuf> = None;
    let mut table_path: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut ratio_train: Option<PathBuf> = None;
    let mut ppm_tenths = DEFAULT_PPM_TENTHS;
    let mut nodes_max = u64::MAX;
    let mut capacity = usize::MAX;
    let mut ratio_margin: u16 = 0;
    let mut formula_window = DEFAULT_FORMULA_WINDOW;
    let mut lane_visits_max = DEFAULT_LANE_VISITS_MAX;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--train" => train = args.next().map(PathBuf::from),
            "--validation" => validation = args.next().map(PathBuf::from),
            "--table" => table_path = args.next().map(PathBuf::from),
            "--out" => out = args.next().map(PathBuf::from),
            "--ratio-train" => ratio_train = args.next().map(PathBuf::from),
            "--ppm-tenths" => {
                ppm_tenths = args.next().map(|v| parse_u32(&v, "--ppm-tenths")).unwrap_or_else(|| usage());
                if ppm_tenths > 1000 {
                    eprintln!("ms2_formula_report: --ppm-tenths {ppm_tenths} exceeds 1000");
                    std::process::exit(2);
                }
            }
            "--nodes-max" => {
                nodes_max = args.next().map(|v| parse_u64(&v, "--nodes-max")).unwrap_or_else(|| usage());
            }
            "--capacity" => {
                capacity = args.next().map(|v| parse_usize(&v, "--capacity")).unwrap_or_else(|| usage());
            }
            "--ratio-margin" => {
                ratio_margin = args.next().map(|v| parse_u16(&v, "--ratio-margin")).unwrap_or_else(|| usage());
            }
            "--formula-window" => {
                formula_window = args.next().map(|v| parse_u32(&v, "--formula-window")).unwrap_or_else(|| usage());
            }
            "--lane-visits-max" => {
                lane_visits_max = args.next().map(|v| parse_u32(&v, "--lane-visits-max")).unwrap_or_else(|| usage());
            }
            _ => usage(),
        }
    }
    let (Some(train), Some(validation), Some(table_path), Some(out)) =
        (train, validation, table_path, out)
    else {
        usage()
    };

    // The fit export (`--ratio-train`, defaulting to `--train`) is loaded
    // FIRST: the domain and the ratio bounds derive from its molecules only.
    let fit_path = ratio_train.unwrap_or_else(|| train.clone());
    let fit_file = ExportFile::load(&fit_path).unwrap_or_else(|e| {
        eprintln!("ms2_formula_report: cannot read {}: {e}", fit_path.display());
        std::process::exit(1);
    });
    // The domain comes from the fit export's molecules only: in-domain
    // graphs (those that build) contribute their compositions with margin 0,
    // so the caps are exactly the observed maxima.
    let mut fit_comps: Vec<Composition> = Vec::new();
    let mut fit_skipped_molecules: usize = 0;
    for mol in &fit_file.molecules {
        match mol.graph() {
            Ok(graph) => fit_comps.push(graph.composition()),
            Err(_) => fit_skipped_molecules += 1,
        }
    }
    let domain = EnumDomain::from_compositions(fit_comps.iter().copied(), 0).unwrap_or_else(|e| {
        eprintln!("ms2_formula_report: cannot derive the domain: {e}");
        std::process::exit(1);
    });
    let bounds = RatioBounds::fit(fit_comps.iter().copied(), ratio_margin).unwrap_or_else(|e| {
        eprintln!("ms2_formula_report: cannot fit the ratio bounds: {e}");
        std::process::exit(1);
    });
    // The fit is frozen here. The report files load only after this point,
    // so no validation (or report-train) molecule is read before the fit.
    let fit_frozen = true;

    let train_file = if fit_path == train {
        fit_file
    } else {
        ExportFile::load(&train).unwrap_or_else(|e| {
            eprintln!("ms2_formula_report: cannot read {}: {e}", train.display());
            std::process::exit(1);
        })
    };
    assert!(
        fit_frozen,
        "the domain and ratio bounds are fitted before any report molecule is read"
    );
    let validation_file = ExportFile::load(&validation).unwrap_or_else(|e| {
        eprintln!("ms2_formula_report: cannot read {}: {e}", validation.display());
        std::process::exit(1);
    });
    assert!(
        fit_frozen,
        "the domain and ratio bounds are fitted before any validation molecule is read"
    );

    let table_text = std::fs::read_to_string(&table_path).unwrap_or_else(|e| {
        eprintln!("ms2_formula_report: cannot read {}: {e}", table_path.display());
        std::process::exit(1);
    });
    let table = FormulaTable::from_json(&table_text).unwrap_or_else(|e| {
        eprintln!("ms2_formula_report: cannot parse {}: {e}", table_path.display());
        std::process::exit(1);
    });

    // Table lookup: composition to row index for the scored-stage check.
    let mut table_rows: HashMap<Composition, usize> = HashMap::new();
    let mut table_set: HashSet<Composition> = HashSet::new();
    for row in 0..table.len() {
        let comp = *table.composition(row);
        table_rows.entry(comp).or_insert(row);
        table_set.insert(comp);
    }

    let cap_u32 = u32::try_from(capacity).unwrap_or(u32::MAX);
    let limits = EnumLimits {
        nodes_visited_max: nodes_max,
        capacity,
        scored_max: capacity,
        filter_h_max: true,
        filter_parity: true,
        filter_dbe: true,
        ratio: None,
    };
    let limits_ratio = EnumLimits {
        ratio: Some(bounds.clone()),
        ..limits.clone()
    };
    // The device-order twin runs the same bounds in lane order with the
    // per-lane visit budget and the scored capacity `M`.
    let device_limits = DeviceEnumLimits {
        lane_visits_max,
        scored_cap: formula_window,
    };
    if let Err(e) = validate_device_artifacts(&domain, &bounds) {
        eprintln!("ms2_formula_report: device artifacts invalid: {e}");
        std::process::exit(1);
    }

    let mut train_table = MethodStats::default();
    let mut train_enum = MethodStats::default();
    let mut train_ratio = MethodStats::default();
    let mut train_device = DeviceStats::default();
    let train_unavailable = run_split(
        &train_file,
        &table,
        &table_rows,
        &table_set,
        &domain,
        &bounds,
        ppm_tenths,
        cap_u32,
        &limits,
        &limits_ratio,
        &device_limits,
        &mut train_table,
        &mut train_enum,
        &mut train_ratio,
        &mut train_device,
    );
    let mut validation_table = MethodStats::default();
    let mut validation_enum = MethodStats::default();
    let mut validation_ratio = MethodStats::default();
    let mut validation_device = DeviceStats::default();
    let validation_unavailable = run_split(
        &validation_file,
        &table,
        &table_rows,
        &table_set,
        &domain,
        &bounds,
        ppm_tenths,
        cap_u32,
        &limits,
        &limits_ratio,
        &device_limits,
        &mut validation_table,
        &mut validation_enum,
        &mut validation_ratio,
        &mut validation_device,
    );

    // Validation molecules outside the fit-derived domain. Out-of-domain
    // gold is a miss and stays in the denominator of the recall rates above.
    let mut out_of_domain_molecules: usize = 0;
    let mut gold_unavailable_molecules: usize = 0;
    for mol in &validation_file.molecules {
        match mol.graph() {
            Ok(graph) => {
                if !domain.contains(&graph.composition()) {
                    out_of_domain_molecules += 1;
                }
            }
            Err(_) => gold_unavailable_molecules += 1,
        }
    }
    let validation_molecules = validation_file.molecules.len();
    let out_of_domain_fraction = if validation_molecules == 0 {
        0.0
    } else {
        out_of_domain_molecules as f64 / validation_molecules as f64
    };

    let file_name = |p: &PathBuf| {
        p.file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| p.display().to_string())
    };
    // Query context: the stored precursor precision drives the window width,
    // hence the candidate-set sizes. Aggregates only.
    let query_json = |file: &ExportFile| {
        let mut unc: Vec<f64> = Vec::new();
        let mut pmz: Vec<f64> = Vec::new();
        for mol in &file.molecules {
            for s in &mol.spectra {
                unc.push(s.precursor_uncertainty_udalton as f64);
                pmz.push(s.precursor_mz_udalton as f64);
            }
        }
        unc.sort_by(|a, b| a.total_cmp(b));
        pmz.sort_by(|a, b| a.total_cmp(b));
        serde_json::json!({
            "unit": "integer micro-dalton (1e-6 Da)",
            "precursor_uncertainty": summarize(&unc),
            "precursor_mz": summarize(&pmz),
        })
    };
    let unavailable_rate = |count: usize, spectra: usize| {
        if spectra == 0 {
            0.0
        } else {
            count as f64 / spectra as f64
        }
    };
    let report = serde_json::json!({
        "schema_version": 2,
        "implementation": "rust mamba3::models::ms2 (formula table versus closed-form-hydrogen enumeration, exact and ratio-pruned)",
        "timing": "per-spectrum wall clock via std::time::Instant, single thread, release build",
        "train_file": file_name(&train),
        "validation_file": file_name(&validation),
        "table_file": file_name(&table_path),
        "chemistry": mamba3::models::ms2::chem::CHEMISTRY_VERSION,
        "ppm_tenths": ppm_tenths,
        "nodes_max": nodes_max,
        "capacity": capacity,
        "ratio_margin": ratio_margin,
        "ratio_train_file": file_name(&fit_path),
        "formula_window": formula_window,
        "lane_visits_max": lane_visits_max,
        "domain": serde_json::from_str::<serde_json::Value>(&domain.to_json()).unwrap_or(serde_json::Value::Null),
        "domain_bytes": domain.bytes(),
        "domain_derived_from_molecules": fit_comps.len(),
        "domain_skipped_molecules": fit_skipped_molecules,
        "ratio_bounds": serde_json::from_str::<serde_json::Value>(&bounds.to_json()).unwrap_or(serde_json::Value::Null),
        "ratio_bounds_bytes": bounds.bytes(),
        "ratio_stages": RATIO_STAGES,
        "table_rows": table.len(),
        "table_bytes": table.bytes(),
        "out_of_domain": {
            "validation_molecules": validation_molecules,
            "out_of_domain": out_of_domain_molecules,
            "gold_unavailable": gold_unavailable_molecules,
            "fraction": out_of_domain_fraction,
        },
        "train": {
            "spectra": train_table.spectra,
            "exact_mass_unavailable": train_unavailable,
            "exact_mass_unavailable_rate": unavailable_rate(train_unavailable, train_table.spectra),
            "query": query_json(&train_file),
            "table": method_json(&train_table, true),
            "enumeration": method_json(&train_enum, false),
            "enumeration_ratio": method_json_ratio(&train_ratio),
            "enumeration_ratio_device_order": method_json_device(&train_device),
        },
        "validation": {
            "spectra": validation_table.spectra,
            "exact_mass_unavailable": validation_unavailable,
            "exact_mass_unavailable_rate": unavailable_rate(validation_unavailable, validation_table.spectra),
            "query": query_json(&validation_file),
            "table": method_json(&validation_table, true),
            "enumeration": method_json(&validation_enum, false),
            "enumeration_ratio": method_json_ratio(&validation_ratio),
            "enumeration_ratio_device_order": method_json_device(&validation_device),
        },
    });
    if let Some(parent) = out.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent).unwrap_or_else(|e| {
            eprintln!("ms2_formula_report: cannot create {}: {e}", parent.display());
            std::process::exit(1);
        });
    }
    std::fs::write(&out, serde_json::to_string_pretty(&report).unwrap()).unwrap_or_else(|e| {
        eprintln!("ms2_formula_report: cannot write {}: {e}", out.display());
        std::process::exit(1);
    });
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
}

/// Evaluate one export file with all three methods, accumulating per-split
/// stats. Returns the spectra whose exact mass is unavailable (unknown
/// precursor precision or invalid parent mass): the search evaluates nothing
/// for them under any method, so they contribute to no recall stage while
/// staying in every denominator.
#[allow(clippy::too_many_arguments)]
fn run_split(
    file: &ExportFile,
    table: &FormulaTable,
    table_rows: &HashMap<Composition, usize>,
    table_set: &HashSet<Composition>,
    domain: &EnumDomain,
    bounds: &RatioBounds,
    ppm_tenths: u32,
    cap_u32: u32,
    limits: &EnumLimits,
    limits_ratio: &EnumLimits,
    device_limits: &DeviceEnumLimits,
    table_stats: &mut MethodStats,
    enum_stats: &mut MethodStats,
    ratio_stats: &mut MethodStats,
    device_stats: &mut DeviceStats,
) -> usize {
    let table_width = |tol: u32, bound: u32| -> u64 {
        u64::from(tol) + u64::from(bound) + u64::from(table.max_error())
    };
    let enum_width = |tol: u32, bound: u32| -> u64 {
        u64::from(tol) + u64::from(bound) + u64::from(domain.max_error())
    };
    let mut unavailable: usize = 0;
    for mol in &file.molecules {
        // Gold is the molecule's own composition from its atoms: element
        // counts plus the parent hydrogens of its atom types. A molecule
        // whose graph does not build has no gold and is a miss everywhere.
        let gold: Option<Composition> = mol.graph().ok().map(|g| g.composition());
        for s in &mol.spectra {
            let window = WindowQuery {
                precursor_mz: s.precursor_mz_udalton,
                adduct: s.adduct,
                ppm_tenths,
                precursor_uncertainty: s.precursor_uncertainty_udalton,
                rows_visited_max: u32::MAX,
                rows_scored_max: cap_u32,
            };
            let query = EnumQuery::from(&window);

            // Method-independent gold facts from the shared library function:
            // with unknown precision or an invalid parent mass the gold
            // contributes to no stage (but stays in the denominator).
            let parent = parent_mass(window.precursor_mz, window.adduct).ok();
            let gold_mass = gold.as_ref().and_then(|c| composition_mass(c).ok());
            let tol = tolerance(window.precursor_mz, window.ppm_tenths);
            let bound = window.precursor_uncertainty.saturating_add(1);
            let table_in_domain = gold.as_ref().is_some_and(|c| table_set.contains(c));
            let table_stages = gold_stages(
                parent,
                gold.as_ref(),
                gold_mass,
                tol,
                window.precursor_uncertainty,
                table_width(tol, bound),
                table_in_domain,
            );
            let enum_in_domain = gold.as_ref().is_some_and(|c| domain.contains(c));
            let enum_stages = gold_stages(
                parent,
                gold.as_ref(),
                gold_mass,
                tol,
                window.precursor_uncertainty,
                enum_width(tol, bound),
                enum_in_domain,
            );
            if table_stages.exact_mass_unavailable {
                unavailable += 1;
            }

            // Indexed table.
            let started = Instant::now();
            let table_found = table.window(&window);
            let table_latency = started.elapsed().as_secs_f64();
            let table_scored = match gold {
                Some(c) => table_rows
                    .get(&c)
                    .is_some_and(|row| table_found.joined.contains(row)),
                None => false,
            };
            push_table(
                table_stats,
                &table_stages,
                table_scored,
                table_found.rows_joined as f64,
                table_found.rows_visited as f64,
                table_found.exhausted,
                table_found.absent,
                table_latency,
            );

            // Closed-form-hydrogen enumeration.
            let started = Instant::now();
            let enum_found = enumerate(domain, &query, limits)
                .expect("ms2_formula_report: enumeration inputs are validated");
            let enum_latency = started.elapsed().as_secs_f64();
            let enum_scored = match gold {
                Some(c) => enum_found.scored_contains(&c),
                None => false,
            };
            push_enum(
                enum_stats,
                &enum_stages,
                enum_scored,
                enum_found.rows_joined as f64,
                enum_found.nodes_visited as f64,
                enum_found.hydrogen_checks as f64,
                enum_found.exhausted,
                enum_found.absent,
                enum_latency,
            );

            // Ratio-pruned enumeration. The window, verdict and exact-filter
            // stages match the exact enumeration (same domain and window);
            // the ratio stages then apply in `RATIO_STAGES` order.
            let started = Instant::now();
            let ratio_found = enumerate(domain, &query, limits_ratio)
                .expect("ms2_formula_report: enumeration inputs are validated");
            let ratio_latency = started.elapsed().as_secs_f64();
            let ratio_scored = match gold {
                Some(c) => ratio_found.scored_contains(&c),
                None => false,
            };
            let mut stage_ok = enum_stages.after_dbe;
            let mut ratio_recall = [false; 9];
            if let Some(c) = gold {
                let checks = [
                    bounds.passes_cap(&c),
                    bounds.passes_rare_max(&c) && bounds.passes_rare_min(&c),
                    bounds.passes_ratio(0, &c),
                    bounds.passes_ratio(1, &c),
                    bounds.passes_ratio(2, &c),
                    bounds.passes_ratio(3, &c),
                    bounds.passes_ratio(4, &c),
                    bounds.passes_ratio(5, &c),
                    bounds.passes_ratio_dbe(&c),
                ];
                for (i, ok) in checks.iter().enumerate() {
                    stage_ok = stage_ok && *ok;
                    ratio_recall[i] = stage_ok;
                }
            }
            // Per-stage candidate sizes, reconstructed from the counted
            // rejects (every check ends in exactly one bucket, so each
            // difference is the survivor count past that stage).
            let to_f = |n: u64| n as f64;
            let verdict_n = ratio_found
                .hydrogen_checks
                .saturating_sub(ratio_found.rejected_mass);
            let exact_n = verdict_n
                .saturating_sub(ratio_found.rejected_h_max)
                .saturating_sub(ratio_found.rejected_parity)
                .saturating_sub(ratio_found.rejected_dbe);
            let cap_n = exact_n.saturating_sub(ratio_found.rejected_ratio_cap);
            let rare_n = cap_n.saturating_sub(ratio_found.rejected_rare);
            let hc_n = rare_n.saturating_sub(ratio_found.rejected_ratio_hc);
            let nc_n = hc_n.saturating_sub(ratio_found.rejected_ratio_nc);
            let oc_n = nc_n.saturating_sub(ratio_found.rejected_ratio_oc);
            let hal_n = oc_n.saturating_sub(ratio_found.rejected_ratio_hal);
            let s_n = hal_n.saturating_sub(ratio_found.rejected_ratio_s);
            let p_n = s_n.saturating_sub(ratio_found.rejected_ratio_p);
            let dbe_n = p_n.saturating_sub(ratio_found.rejected_ratio_dbe);
            assert_eq!(
                dbe_n, ratio_found.rows_joined,
                "ratio stage sizes reconcile to the joined rows"
            );
            push_ratio(
                ratio_stats,
                &enum_stages,
                ratio_recall,
                ratio_scored,
                [
                    to_f(cap_n),
                    to_f(rare_n),
                    to_f(hc_n),
                    to_f(nc_n),
                    to_f(oc_n),
                    to_f(hal_n),
                    to_f(s_n),
                    to_f(p_n),
                    to_f(dbe_n),
                ],
                to_f(ratio_found.hydrogen_checks),
                to_f(verdict_n),
                to_f(exact_n),
                to_f(ratio_found.rows_scored),
                to_f(ratio_found.pruned_ratio_cap),
                to_f(ratio_found.pruned_rare),
                ratio_found.rows_joined as f64,
                ratio_found.nodes_visited as f64,
                ratio_found.hydrogen_checks as f64,
                ratio_found.exhausted,
                ratio_found.absent,
                ratio_latency,
            );

            // Device-order twin of the ratio-pruned enumeration: the same
            // domain and bounds in lane order (rare row, then C, N, O, H)
            // with the per-lane visit budget and the scored capacity `M`.
            // Gold recall is measured in the scored prefix. The artifacts
            // were validated once in `main`; a per-spectrum failure is
            // therefore unreachable.
            let started = Instant::now();
            let device_found = enumerate_device_order(domain, bounds, &query, device_limits)
                .unwrap_or_else(|e| {
                    eprintln!("ms2_formula_report: device-order enumeration failed: {e}");
                    std::process::exit(1);
                });
            let device_latency = started.elapsed().as_secs_f64();
            let device_scored = match gold {
                Some(c) => device_found.scored_contains(&c),
                None => false,
            };
            push_device(
                device_stats,
                device_scored,
                device_found.joined as f64,
                device_found.visited as f64,
                device_found.lanes.len(),
                device_found.exhausted,
                device_found.absent,
                device_latency,
            );
        }
    }
    unavailable
}

/// Record one spectrum for the indexed table.
#[allow(clippy::too_many_arguments)]
fn push_table(
    stats: &mut MethodStats,
    stages: &GoldStages,
    scored: bool,
    joined: f64,
    visited: f64,
    exhausted: bool,
    absent: bool,
    latency: f64,
) {
    stats.spectra += 1;
    if stages.window {
        stats.window += 1;
    }
    if stages.accepted {
        stats.accepted += 1;
    }
    if stages.accepted_or_ambiguous {
        stats.accepted_or_ambiguous += 1;
    }
    if stages.after_h_max {
        stats.after_h_max += 1;
    }
    if stages.after_parity {
        stats.after_parity += 1;
    }
    if stages.after_dbe {
        stats.after_dbe += 1;
    }
    if scored {
        stats.scored += 1;
    }
    stats.joined.push(joined);
    stats.work_primary.push(visited);
    if exhausted {
        stats.exhausted += 1;
    }
    if absent {
        stats.absent += 1;
    }
    stats.latency.push(latency);
}

/// Record one spectrum for the enumeration.
#[allow(clippy::too_many_arguments)]
fn push_enum(
    stats: &mut MethodStats,
    stages: &GoldStages,
    scored: bool,
    joined: f64,
    nodes: f64,
    checks: f64,
    exhausted: bool,
    absent: bool,
    latency: f64,
) {
    push_table(
        stats, stages, scored, joined, nodes, exhausted, absent, latency,
    );
    stats.work_secondary.push(checks);
}

/// Record one spectrum for the ratio-pruned enumeration: the base stages as
/// for the exact enumeration, then the cumulative ratio-stage recall and the
/// per-stage candidate sizes.
#[allow(clippy::too_many_arguments)]
fn push_ratio(
    stats: &mut MethodStats,
    stages: &GoldStages,
    ratio_recall: [bool; 9],
    scored: bool,
    size_stages: [f64; 9],
    evaluated: f64,
    verdict: f64,
    exact: f64,
    scored_size: f64,
    pruned_cap: f64,
    pruned_rare: f64,
    joined: f64,
    nodes: f64,
    checks: f64,
    exhausted: bool,
    absent: bool,
    latency: f64,
) {
    push_enum(
        stats, stages, scored, joined, nodes, checks, exhausted, absent, latency,
    );
    for (i, ok) in ratio_recall.iter().enumerate() {
        if *ok {
            stats.ratio_recall[i] += 1;
        }
    }
    stats.size_evaluated.push(evaluated);
    stats.size_verdict.push(verdict);
    stats.size_exact.push(exact);
    stats.size_stages.push(size_stages);
    stats.size_scored.push(scored_size);
    stats.pruned_cap.push(pruned_cap);
    stats.pruned_rare.push(pruned_rare);
}

/// Record one spectrum for the device-order twin: gold recall in the
/// scored prefix, candidate counts, visits, lanes `P`, latency. Exhausted
/// spectra can still count as hits (gold membership in the retained bounded
/// output); the hit/miss split of exhausted spectra is counted separately.
#[allow(clippy::too_many_arguments)]
fn push_device(
    stats: &mut DeviceStats,
    scored: bool,
    joined: f64,
    visited: f64,
    lanes: usize,
    exhausted: bool,
    absent: bool,
    latency: f64,
) {
    stats.spectra += 1;
    if scored {
        stats.scored += 1;
    }
    stats.joined.push(joined);
    stats.visited.push(visited);
    stats.lanes = lanes;
    if exhausted {
        stats.exhausted += 1;
        if scored {
            stats.exhausted_hit += 1;
        } else {
            stats.exhausted_miss += 1;
        }
    }
    if absent {
        stats.absent += 1;
    }
    stats.latency.push(latency);
}

/// Aggregates-only JSON for one method on one split.
fn method_json(stats: &MethodStats, is_table: bool) -> serde_json::Value {
    let work = if is_table {
        serde_json::json!({"rows_visited": summarize(&stats.work_primary)})
    } else {
        serde_json::json!({
            "nodes_visited": summarize(&stats.work_primary),
            "hydrogen_checks": summarize(&stats.work_secondary),
        })
    };
    serde_json::json!({
        "spectra": stats.spectra,
        "recall": {
            "window": stats.recall(stats.window),
            "accepted": stats.recall(stats.accepted),
            "accepted_or_ambiguous": stats.recall(stats.accepted_or_ambiguous),
            "after_h_max": stats.recall(stats.after_h_max),
            "after_parity": stats.recall(stats.after_parity),
            "after_dbe": stats.recall(stats.after_dbe),
            "scored": stats.recall(stats.scored),
        },
        "joined": summarize(&stats.joined),
        "work": work,
        "exhausted_rate": stats.rate(stats.exhausted),
        "absent_rate": stats.rate(stats.absent),
        "latency_s": {
            "p50": percentile(&sorted(&stats.latency), 50.0),
            "p95": percentile(&sorted(&stats.latency), 95.0),
        },
    })
}

/// Aggregates-only JSON for the ratio-pruned enumeration on one split:
/// recall after every stage, candidate-set sizes after every stage, work
/// (including pruned DFS branches), exhaustion/absence rates and latency.
fn method_json_ratio(stats: &MethodStats) -> serde_json::Value {
    let mut recall = serde_json::Map::new();
    for (key, value) in [
        ("window", stats.recall(stats.window)),
        ("accepted", stats.recall(stats.accepted)),
        (
            "accepted_or_ambiguous",
            stats.recall(stats.accepted_or_ambiguous),
        ),
        ("after_h_max", stats.recall(stats.after_h_max)),
        ("after_parity", stats.recall(stats.after_parity)),
        ("after_dbe", stats.recall(stats.after_dbe)),
    ] {
        recall.insert(key.to_string(), serde_json::json!(value));
    }
    for (i, key) in RATIO_STAGES.iter().enumerate() {
        recall.insert(key.to_string(), serde_json::json!(stats.recall(stats.ratio_recall[i])));
    }
    recall.insert(
        "scored".to_string(),
        serde_json::json!(stats.recall(stats.scored)),
    );
    let mut sizes = serde_json::Map::new();
    sizes.insert(
        "evaluated".to_string(),
        summarize(&stats.size_evaluated),
    );
    sizes.insert("verdict_passing".to_string(), summarize(&stats.size_verdict));
    sizes.insert("after_exact".to_string(), summarize(&stats.size_exact));
    for (i, key) in RATIO_STAGES.iter().enumerate() {
        let column: Vec<f64> = stats.size_stages.iter().map(|row| row[i]).collect();
        sizes.insert(key.to_string(), summarize(&column));
    }
    sizes.insert("scored".to_string(), summarize(&stats.size_scored));
    serde_json::json!({
        "spectra": stats.spectra,
        "recall": serde_json::Value::Object(recall),
        "sizes": serde_json::Value::Object(sizes),
        "joined": summarize(&stats.joined),
        "work": {
            "nodes_visited": summarize(&stats.work_primary),
            "hydrogen_checks": summarize(&stats.work_secondary),
            "pruned_ratio_cap": summarize(&stats.pruned_cap),
            "pruned_rare": summarize(&stats.pruned_rare),
        },
        "exhausted_rate": stats.rate(stats.exhausted),
        "absent_rate": stats.rate(stats.absent),
        "latency_s": {
            "p50": percentile(&sorted(&stats.latency), 50.0),
            "p95": percentile(&sorted(&stats.latency), 95.0),
        },
    })
}

/// Sorted copy for [`percentile`].
fn sorted(values: &[f64]) -> Vec<f64> {
    let mut out = values.to_vec();
    out.sort_by(|a, b| a.total_cmp(b));
    out
}

/// Aggregates-only JSON for the device-order twin on one split: gold recall
/// in the scored prefix, candidate counts, visits, lanes `P`, exhausted and
/// absent rates, the exhausted hit/miss split, latency.
fn method_json_device(stats: &DeviceStats) -> serde_json::Value {
    serde_json::json!({
        "spectra": stats.spectra,
        "scored_prefix_note": "under lane exhaustion the scored output is a prefix of the VISITED lane results in lane order, not necessarily of the unrestricted candidate stream",
        "exhausted_hit_note": "exhausted spectra can still count as hits: scored recall is gold membership in the retained bounded output",
        "recall": {
            "scored": stats.recall(stats.scored),
        },
        "joined": summarize(&stats.joined),
        "work": {
            "visits": summarize(&stats.visited),
        },
        "lanes": stats.lanes,
        "exhausted_rate": stats.rate(stats.exhausted),
        "exhausted_hit": stats.exhausted_hit,
        "exhausted_miss": stats.exhausted_miss,
        "absent_rate": stats.rate(stats.absent),
        "latency_s": {
            "p50": percentile(&sorted(&stats.latency), 50.0),
            "p95": percentile(&sorted(&stats.latency), 95.0),
        },
    })
}
