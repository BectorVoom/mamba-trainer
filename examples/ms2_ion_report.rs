//! Host ion-assignment report over an `export_casmi.py` file (H4.3).
//!
//! Usage: `cargo run --release --no-default-features --features cpu
//! --example ms2_ion_report -- --input <export.json> [--j 4]
//! [--work-max 4096] [--limit-spectra N] --out <report.json>`.
//!
//! Oracle-conditioned (true parent formula) diagnostic: for every inspected
//! spectrum, under the TRUE parent formula, over the device-filtered kept
//! peaks (the host twin of peak selection, N = 128): hypotheses per peak,
//! assignment support statuses, anchored-peak label retention at `J` and the
//! explained intensity the kept labels carry. Aggregates only: no SMILES, no
//! ids, no peaks.
//!
//! Denominators cover every inspected spectrum: labeled, unlabeled,
//! out-of-domain and unknown-precision counts with labeled coverage over
//! inspected. `--limit-spectra` limits INSPECTED spectra. Label-upload
//! losses (the `LABEL_CAP` truncation) are reported separately from
//! hypothesis-cap/search losses: capped retention statistics carry
//! explicitly conditional names, uncapped statistics name the full
//! deduplicated label set.
//!
//! The example needs no device and constructs none.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::time::Instant;

use mamba3::models::ms2::chem::adduct;
use mamba3::models::ms2::contract::{PRECURSOR_MAX, PRECURSOR_MIN};
use mamba3::models::ms2::dataset::{ExportFile, ExportSpectrum, percentile};
use mamba3::models::ms2::experiment::{LABEL_PPM_TENTHS, label_export_spectrum};
use mamba3::models::ms2::ion::{
    ION_CAPACITY_EXCEEDED, ION_SEARCH_EXHAUSTED, ION_UNAVAILABLE, IonLimits, ion_assign,
    ion_labels,
};
use mamba3::models::ms2::targets::RecipeLimits;
use mamba3::models::ms2::targets::{Candidates, Peak};
use mamba3::models::ms2::twin;

/// Device peak capacity of the report (contracts §3.3 `n_peaks`).
const N_KEEP: usize = 128;
/// Uploaded label capacity of spec §2.3.
const LABEL_CAP: usize = 64;

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
    eprintln!(
        "usage: ms2_ion_report --input <export.json> [--j 4] [--work-max 4096] [--limit-spectra N] --out <report.json>"
    );
    std::process::exit(2);
}

fn parse_flag(args: &mut impl Iterator<Item = String>, name: &str) -> u32 {
    args.next().unwrap_or_else(|| usage()).parse().unwrap_or_else(|_| {
        eprintln!("ms2_ion_report: invalid value for {name}");
        std::process::exit(2);
    })
}

/// Request-domain gate of contracts §4.6 (adduct, polarity, precursor range).
///
/// Spectra outside the gate never train and count as out-of-domain in the
/// report denominator; structure failures are handled at the molecule level.
fn request_out_of_domain(s: &ExportSpectrum) -> bool {
    if s.adduct == 0 {
        return true;
    }
    let Some(a) = adduct(s.adduct) else {
        return true;
    };
    if s.polarity != 1 && s.polarity != -1 {
        return true;
    }
    if i32::from(s.polarity) != a.charge {
        return true;
    }
    if !(PRECURSOR_MIN..=PRECURSOR_MAX).contains(&s.precursor_mz_udalton) {
        return true;
    }
    false
}

fn main() {
    let started = Instant::now();
    let mut input: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut j: u32 = 4;
    let mut work_max: u32 = 4096;
    let mut limit_spectra: Option<usize> = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--input" => input = args.next().map(PathBuf::from),
            "--out" => out = args.next().map(PathBuf::from),
            "--j" => j = parse_flag(&mut args, "--j"),
            "--work-max" => work_max = parse_flag(&mut args, "--work-max"),
            "--limit-spectra" => limit_spectra = Some(parse_flag(&mut args, "--limit-spectra") as usize),
            _ => usage(),
        }
    }
    let Some(input) = input else { usage() };
    let Some(out) = out else { usage() };
    let limits = IonLimits {
        work_max,
        kept: j,
    };

    let file = ExportFile::load(&input).unwrap_or_else(|e| {
        eprintln!("ms2_ion_report: cannot read {}: {e}", input.display());
        std::process::exit(1);
    });
    let name = input
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| input.display().to_string());

    // Full spectrum denominator over inspected spectra (contracts §7.1):
    // labeled, unlabeled, out-of-domain and unknown-precision counts.
    // Unknown-precision spectra carry no decisions by design and form their
    // own partition: inspected = labeled + unlabeled + out_of_domain +
    // unknown_precision. Peak and anchored statistics below are conditional
    // on labeled spectra (peak keys say so); capped retention statistics
    // are conditional on uploaded labels.
    let mut spectra_inspected = 0u64;
    let mut spectra_labeled = 0u64;
    let mut spectra_unlabeled = 0u64;
    let mut spectra_out_of_domain = 0u64;
    let mut spectra_unknown_precision = 0u64;
    // Pooled peak counters over labeled spectra only.
    let mut peaks_total = 0u64;
    let mut peaks_with_hyp = 0u64;
    let mut peaks_exhausted = 0u64;
    let mut peaks_capacity = 0u64;
    let mut peaks_unavailable = 0u64;
    let mut accepted_all: Vec<f64> = Vec::new();
    // Pooled anchored-peak counters and intensity sums.
    let mut anchored_total = 0u64;
    let mut anchored_full = 0u64;
    let mut anchored_partial = 0u64;
    let mut anchored_dropped = 0u64;
    let mut anchored_total_uncapped = 0u64;
    let mut anchored_slots_uncapped = 0u64;
    let mut anchored_full_uncapped = 0u64;
    let mut anchored_partial_uncapped = 0u64;
    let mut anchored_dropped_uncapped = 0u64;
    let mut anchored_per_spectrum: Vec<f64> = Vec::new();
    let mut anchored_per_spectrum_uncapped: Vec<f64> = Vec::new();
    let mut labels_uncapped = 0u64;
    let mut labels_uploaded = 0u64;
    let mut upload_peaks_affected = 0u64;
    let mut overflow_labels = 0u64;
    let mut overflow_spectra = 0u64;
    let mut explained_intensity_sum = 0.0f64;
    let mut kept_intensity_sum = 0.0f64;

    for mol in &file.molecules {
        // Structure domain per molecule: a failing graph or preparation
        // marks every spectrum of the molecule out of domain (counted
        // per spectrum below under the inspected limit).
        let prepared = mol.graph().ok().and_then(|g| {
            let comp = g.composition();
            Candidates::new(&g, &RecipeLimits::V0)
                .ok()
                .map(|c| (comp, c))
        });
        for s in &mol.spectra {
            // --limit-spectra limits INSPECTED spectra.
            if limit_spectra.is_some_and(|n| spectra_inspected as usize >= n) {
                break;
            }
            spectra_inspected += 1;
            let Some((parent, candidates)) = prepared.as_ref() else {
                spectra_out_of_domain += 1;
                continue;
            };
            if request_out_of_domain(s) {
                spectra_out_of_domain += 1;
                continue;
            }
            if s.mz_uncertainty_udalton == u32::MAX {
                spectra_unknown_precision += 1;
                continue;
            }
            let labels = label_export_spectrum(candidates, s).unwrap_or_else(|e| {
                eprintln!("ms2_ion_report: row {}: {e}", s.row);
                std::process::exit(1);
            });
            if labels.targets.is_empty() {
                spectra_unlabeled += 1;
                continue;
            }
            spectra_labeled += 1;
            // Device-filtered kept peaks through the host twin of peak
            // selection, so these are the peaks the device keeps (N = 128).
            let n_raw = s.mz_udalton.len();
            let intensity: Vec<f32> = s.intensity.iter().map(|&v| v as f32).collect();
            let mut meta = vec![0u32; 8];
            meta[0] = n_raw as u32;
            meta[1] = s.precursor_mz_udalton;
            let sel = twin::peak_select(&s.mz_udalton, &intensity, &meta, 1, n_raw, N_KEEP, 0);
            // Labels keyed by raw batch index (what kept[.., 0] holds).
            let raw_of: HashMap<u32, u32> = s
                .peak_id
                .iter()
                .enumerate()
                .map(|(k, &id)| (id, k as u32))
                .collect();
            // Uncapped deduplicated label set (diagnostic denominator) next to
            // the capped uploaded set: upload losses are the difference.
            let uncapped =
                ion_labels(&labels, s.adduct, |id| raw_of.get(&id).copied(), usize::MAX);
            let label_set =
                ion_labels(&labels, s.adduct, |id| raw_of.get(&id).copied(), LABEL_CAP);
            let lost = uncapped.labels.len().saturating_sub(label_set.labels.len());
            debug_assert_eq!(lost, label_set.overflow);
            labels_uncapped += uncapped.labels.len() as u64;
            labels_uploaded += label_set.labels.len() as u64;
            overflow_labels += lost as u64;
            if lost > 0 {
                overflow_spectra += 1;
            }
            // Peaks touched by the upload truncation: distinct raw indices
            // among the dropped suffix labels.
            let dropped_raw: HashSet<u32> = uncapped.labels[label_set.labels.len()..]
                .iter()
                .map(|l| l.raw_index)
                .collect();
            let by_raw: HashMap<u32, Vec<&mamba3::models::ms2::Composition>> = {
                let mut map: HashMap<u32, Vec<&mamba3::models::ms2::Composition>> =
                    HashMap::new();
                for l in &label_set.labels {
                    map.entry(l.raw_index).or_default().push(&l.counts);
                }
                map
            };
            let by_raw_uncapped: HashMap<u32, Vec<&mamba3::models::ms2::Composition>> = {
                let mut map: HashMap<u32, Vec<&mamba3::models::ms2::Composition>> =
                    HashMap::new();
                for l in &uncapped.labels {
                    map.entry(l.raw_index).or_default().push(&l.counts);
                }
                map
            };
            anchored_total_uncapped += by_raw_uncapped.len() as u64;
            upload_peaks_affected += dropped_raw.len() as u64;
            // Filtered-peak intensities for the explained-intensity share.
            let filtered: Vec<Peak> = s.peaks().unwrap_or_else(|e| {
                eprintln!("ms2_ion_report: row {}: {e}", s.row);
                std::process::exit(1);
            });
            let intensity_of: HashMap<u32, f64> =
                filtered.iter().map(|p| (p.id, p.intensity)).collect();
            let mut anchored_here = 0u64;
            let mut anchored_here_uncapped = 0u64;
            for p in 0..N_KEEP {
                let raw = sel.kept[p * 3];
                if raw == u32::MAX {
                    continue;
                }
                let mz = sel.kept[p * 3 + 1];
                let assign = ion_assign(
                    parent,
                    s.adduct,
                    mz,
                    s.mz_uncertainty_udalton,
                    LABEL_PPM_TENTHS,
                    &limits,
                )
                .unwrap_or_else(|e| {
                    eprintln!("ms2_ion_report: row {}: {e}", s.row);
                    std::process::exit(1);
                });
                peaks_total += 1;
                accepted_all.push(assign.accepted as f64);
                if assign.accepted > 0 {
                    peaks_with_hyp += 1;
                }
                if assign.status & ION_SEARCH_EXHAUSTED != 0 {
                    peaks_exhausted += 1;
                }
                if assign.status & ION_CAPACITY_EXCEEDED != 0 {
                    peaks_capacity += 1;
                }
                if assign.status & ION_UNAVAILABLE != 0 {
                    peaks_unavailable += 1;
                }
                // Anchored peaks: raw indices carrying at least one label.
                // The capped (`by_raw`) view is conditional on uploaded
                // labels; the uncapped view mixes upload losses with
                // hypothesis-cap/search losses (disentangled above).
                let capped_want = by_raw.get(&raw);
                let uncapped_want = by_raw_uncapped.get(&raw);
                if capped_want.is_some() || uncapped_want.is_some() {
                    let kept_here: Vec<&mamba3::models::ms2::Composition> =
                        assign.kept.iter().map(|h| &h.counts).collect();
                    if let Some(want) = capped_want {
                        anchored_here += 1;
                        anchored_total += 1;
                        let kept_count = want.iter().filter(|c| kept_here.contains(c)).count();
                        if kept_count == want.len() {
                            anchored_full += 1;
                        } else if kept_count == 0 {
                            anchored_dropped += 1;
                        } else {
                            anchored_partial += 1;
                        }
                        // Intensity carried by kept labels (state 1: some label kept).
                        if kept_count > 0
                            && let Some(&id) = s.peak_id.get(raw as usize)
                        {
                            kept_intensity_sum += intensity_of.get(&id).copied().unwrap_or(0.0);
                        }
                    }
                    if let Some(want_u) = uncapped_want {
                        anchored_here_uncapped += 1;
                        anchored_slots_uncapped += 1;
                        let kept_u = want_u.iter().filter(|c| kept_here.contains(c)).count();
                        if kept_u == want_u.len() {
                            anchored_full_uncapped += 1;
                        } else if kept_u == 0 {
                            anchored_dropped_uncapped += 1;
                        } else {
                            anchored_partial_uncapped += 1;
                        }
                    }
                }
            }
            anchored_per_spectrum.push(anchored_here as f64);
            anchored_per_spectrum_uncapped.push(anchored_here_uncapped as f64);
            for id in &labels.explained_peaks {
                explained_intensity_sum += intensity_of.get(id).copied().unwrap_or(0.0);
            }
        }
        if limit_spectra.is_some_and(|n| spectra_inspected as usize >= n) {
            break;
        }
    }

    let frac = |n: u64, d: u64| {
        if d > 0 {
            n as f64 / d as f64
        } else {
            0.0
        }
    };
    let aggregate = serde_json::json!({
        "oracle_conditioned": true,
        "conditioning": "oracle-conditioned (true parent formula)",
        "spectra_inspected": spectra_inspected,
        "spectra_labeled": spectra_labeled,
        "spectra_unlabeled": spectra_unlabeled,
        "spectra_out_of_domain": spectra_out_of_domain,
        "spectra_unknown_precision": spectra_unknown_precision,
        "labeled_coverage_over_inspected": frac(spectra_labeled, spectra_inspected),
        "limit_spectra": limit_spectra.map_or(serde_json::Value::Null, serde_json::Value::from),
        "limit_applies_to": "inspected_spectra",
        "peaks_kept_given_labeled": peaks_total,
        "accepted_per_peak_given_labeled": summarize(&accepted_all),
        "peaks_with_hypothesis_fraction_given_labeled": frac(peaks_with_hyp, peaks_total),
        "exhausted_rate_given_labeled": frac(peaks_exhausted, peaks_total),
        "capacity_exceeded_rate_given_labeled": frac(peaks_capacity, peaks_total),
        "unavailable_rate_given_labeled": frac(peaks_unavailable, peaks_total),
        "labels_uncapped": labels_uncapped,
        "labels_uploaded": labels_uploaded,
        "label_upload_lost_labels": labels_uncapped.saturating_sub(labels_uploaded),
        "label_upload_lost_label_rate": frac(labels_uncapped.saturating_sub(labels_uploaded), labels_uncapped),
        "anchored_peaks_uncapped": anchored_total_uncapped,
        "upload_peaks_affected": upload_peaks_affected,
        "upload_peaks_affected_rate": frac(upload_peaks_affected, anchored_total_uncapped),
        "anchored_per_spectrum_uncapped": summarize(&anchored_per_spectrum_uncapped),
        "anchored_slots_uncapped": anchored_slots_uncapped,
        "anchored_fully_kept_fraction_uncapped": frac(anchored_full_uncapped, anchored_slots_uncapped),
        "anchored_partially_kept_fraction_uncapped": frac(anchored_partial_uncapped, anchored_slots_uncapped),
        "anchored_dropped_fraction_uncapped": frac(anchored_dropped_uncapped, anchored_slots_uncapped),
        "anchored_per_spectrum_given_uploaded_labels": summarize(&anchored_per_spectrum),
        "anchored_peaks_given_uploaded_labels": anchored_total,
        "anchored_fully_kept_fraction_given_uploaded_labels": frac(anchored_full, anchored_total),
        "anchored_partially_kept_fraction_given_uploaded_labels": frac(anchored_partial, anchored_total),
        "anchored_dropped_fraction_given_uploaded_labels": frac(anchored_dropped, anchored_total),
        "label_overflow_labels": overflow_labels,
        "label_overflow_spectra": overflow_spectra,
        "label_overflow_spectra_fraction": frac(overflow_spectra, spectra_labeled),
        "kept_explained_intensity_fraction_given_uploaded_labels": if explained_intensity_sum > 0.0 {
            kept_intensity_sum / explained_intensity_sum
        } else {
            0.0
        },
        "j": j,
        "work_max": work_max,
        "seconds": started.elapsed().as_secs_f64(),
    });
    let report = serde_json::json!({
        "schema_version": 1,
        "implementation": "rust mamba3::models::ms2::ion, oracle-conditioned (true parent formula), device-filtered peaks",
        "input": name,
        "recipe": "q-cut-v1",
        "ppm_tenths": LABEL_PPM_TENTHS,
        "aggregate": aggregate,
    });
    if let Some(parent) = out.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent).unwrap_or_else(|e| {
            eprintln!("ms2_ion_report: cannot create {}: {e}", parent.display());
            std::process::exit(1);
        });
    }
    std::fs::write(&out, serde_json::to_string_pretty(&report).unwrap()).unwrap_or_else(|e| {
        eprintln!("ms2_ion_report: cannot write {}: {e}", out.display());
        std::process::exit(1);
    });
    println!("{}", serde_json::to_string_pretty(&aggregate).unwrap());
}
