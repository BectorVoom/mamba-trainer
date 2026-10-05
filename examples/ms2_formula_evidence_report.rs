//! Host-only peak-evidence formula ranking report with precursor jitter (FE3).
//!
//! Usage: `cargo run --release --no-default-features --features cpu
//! --example ms2_formula_evidence_report -- --fit <train export> --train
//! <train export> --validation <export> [--window 2048] [--limit-train N]
//! [--limit-validation N] [--seed 1] [--precursor-jitter-ppm 0,1,2,5]
//! --out <report.json>`.
//!
//! For each `sigma` in `--precursor-jitter-ppm`, every spectrum's precursor
//! m/z (train sample and validation) is replaced by `mz * (1 + e * 1e-6)`,
//! `e ~ Normal(0, sigma)` truncated to `|e| <= 3 sigma`, seeded per
//! `(seed, split, spectrum index, sigma)` via
//! `formula_evidence::jitter_precursor_mz` (pure, hence thread-order
//! independent). The enumeration window, candidates, residuals and kept
//! peaks are recomputed from the jittered value; the gold formula stays.
//! `sigma = 0` is the unmodified data.
//!
//! Rankers per sigma: `residual_only` (rule), `prior_only`, plus five
//! trained softmax rankers on STANDARDISED features (FE3): `prior_only` and
//! `prior_plus_residual` (prior + `|r|/1ppm`, `(|r|/1ppm)^2`,
//! `ln(1+|r|/0.1ppm)`, `|r|/tolerance` with `tolerance = 20 ppm`),
//! `prior_plus_residual_plus_evidence` (those plus `expl_count_frac`,
//! `expl_intensity`, `expl_minus_max`, `expl_rank_frac`),
//! `prior_plus_evidence` (prior + the four evidence features, no residual),
//! `prior_nonlinear` (prior plus all pairwise products of the 12 prior
//! features, standardised after forming products — a peak-free stand-in for
//! a non-linear composition prior) and
//! `prior_nonlinear_plus_residual_plus_evidence` (the same plus residual and
//! evidence features). Each trained ranker trains and evaluates at the same
//! sigma; the three evidence rankers additionally train on shuffled peaks
//! and evaluate on shuffled peaks.
//!
//! FE3 optimisation (the FE2 defect: raw features span `|r|/1ppm` up to 20,
//! its square up to 400, log-counts 1–4 and fractions in `[0,1]`, so fixed-
//! step Adam on raw features diverged or stalled and larger models ended with
//! worse train NLL than their subsets): every feature is standardised with
//! the mean and standard deviation over the TRAIN candidates of the run (per
//! sigma, per shuffle variant; zero-variance columns are centred with scale
//! 1; non-finite raw values map to 0), the same affine map is applied at
//! evaluation, the map is stored in the report, and weights are reported in
//! standardised units. Optimisation is full-batch Adam with a decaying step
//! (`lr_init / (1 + lr_decay * epoch)`) until the relative decrease of the
//! train objective over 20 epochs is below `1e-5` (capped; epochs used and
//! convergence are reported). L2 stays `1e-4` on standardised weights.
//!
//! Nestedness is enforced: for every sigma (real and shuffled train sets),
//! `NLL(prior_plus_residual) <= NLL(prior_only) + 1e-3`,
//! `NLL(prior_plus_residual_plus_evidence) <=
//! min(NLL(prior_plus_residual), NLL(prior_plus_evidence)) + 1e-3`,
//! `NLL(prior_plus_evidence) <= NLL(prior_only) + 1e-3`, plus the nonlinear
//! analogues. Violations are printed, recorded under `nestedness_violations`
//! and exit non-zero AFTER writing the report.
//!
//! Evidence cache: fragment evidence does not use the precursor except
//! through window membership (kept peaks are recomputed from the jittered
//! precursor, but the `precursor + 2 Da` cut moves by ppm, so the kept set is
//! near-identical). Per data group (train, validation, shuffled train,
//! shuffled validation) the `sigma = 0` run fills a
//! `(spectrum index, composition) -> (expl_count, expl_intensity)` cache;
//! `sigma > 0` reuses it and only builds fresh evidence for compositions
//! entering the window under jitter. The cache is read-only during the
//! threaded run, so results are thread-order independent.
//!
//! The example needs no device and constructs none. Parallelism is `std`
//! threads only (`rayon` is not a dependency).

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use mamba3::error::Result;
use mamba3::models::ms2::Composition;
use mamba3::models::ms2::dataset::ExportFile;
use mamba3::models::ms2::formula_enum::{
    DeviceEnumLimits, EnumDomain, EnumQuery, RatioBounds, dbe_twice, enumerate_device_order,
    validate_device_artifacts, HEAVY_ELEMENTS,
};
use mamba3::models::ms2::formula_evidence_ref::{
    EVIDENCE4_DIMS, FEATURE_NAMES, ION_PPM_TENTHS, KeptPeak, PRECURSOR_PPM_TENTHS,
    PRIOR_DIMS, PRIOR_PLUS_EVIDENCE4_DIMS, PRIOR_PLUS_RESIDUAL_DIMS,
    PRIOR_PLUS_RESIDUAL_PLUS_EVIDENCE_DIMS, ConvergedTrainConfig, NonlinearKind, RankRule,
    SpectrumEvidence, Standardizer, apply_sharp_features, bootstrap_recall, build_evidence_index,
    check_nestedness, derangement, fit_standardizer, fit_standardizer_nonlinear,
    jitter_precursor_mz, kept_peaks, nonlinear_init_from_prior, nonlinear_plus_feature_names,
    nonlinear_feature_names, precursor_tol_ppm, project_init, recall_at, recall_at_conditional,
    residual_features, standardized_nll, standardized_nll_nonlinear, spectrum_parent_mass,
    train_nonlinear_converged_init, train_softmax_converged, train_softmax_converged_init,
    CandidateEvidence, SpectrumInput, spectrum_evidence,
};

/// Default scored-candidate window `M` (spec §1.4).
const DEFAULT_WINDOW: u32 = 2048;

/// Default per-lane visit budget of the device-order twin.
const DEFAULT_LANE_VISITS_MAX: u32 = 65_536;

/// Default train sample size.
const DEFAULT_LIMIT_TRAIN: usize = 3000;

/// Bootstrap resamples per recall statistic.
const BOOTSTRAP_RESAMPLES: usize = 1000;

/// Default softmax training cap (FE3 converged training; `--epochs` overrides).
const DEFAULT_MAX_EPOCHS: usize = 600;

/// Full-batch training: one Adam step per epoch over all eligible spectra.
const BATCH_SPECTRA: usize = usize::MAX;

/// Initial Adam step size on standardised features (decaying per epoch).
const ADAM_LR_INIT: f64 = 0.5;

/// Per-epoch decay rate: epoch `e` uses `lr_init / (1 + decay * e)`.
const ADAM_LR_DECAY: f64 = 0.5;

/// Nestedness tolerance on train NLL.
const NESTEDNESS_TOL: f64 = 1e-3;

/// L2 penalty of the ranking models (on standardised weights).
const L2: f64 = 1e-4;

/// Recall cutoffs (FE3 adds 64/128/256: how good a cheap first stage is).
const KS: [u32; 6] = [1, 4, 16, 64, 128, 256];

/// Ranker names in evaluation order (FE3).
const RANKERS: [&str; 7] = [
    "residual_only",
    "prior_only",
    "prior_plus_residual",
    "prior_plus_residual_plus_evidence",
    "prior_plus_evidence",
    "prior_nonlinear",
    "prior_nonlinear_plus_residual_plus_evidence",
];

/// Bootstrap ranker offsets. `residual_only` (0) and `prior_only` (3) reuse
/// the first report's offsets so `sigma = 0` reproduces it exactly.
const RANKER_OFFSETS: [u64; 7] = [0, 3, 6, 7, 8, 9, 10];

/// Split tags for the jitter seed: train = 0, validation = 1. Shuffled
/// spectra inherit the jittered precursor of their recipient index.
const SPLIT_TRAIN: u64 = 0;
const SPLIT_VALIDATION: u64 = 1;

fn usage() -> ! {
    eprintln!(
        "usage: ms2_formula_evidence_report --fit <export.json> --train <export.json> \
         --validation <export.json> [--window M] [--limit-train N] [--limit-validation N] \
         [--ratio-margin Q] [--lane-visits-max N] [--epochs E] [--seed S] \
         [--precursor-jitter-ppm 0,1,2,5] --out <report.json>"
    );
    std::process::exit(2);
}

fn parse_u64(text: &str, flag: &str) -> u64 {
    text.parse::<u64>().unwrap_or_else(|_| {
        eprintln!("ms2_formula_evidence_report: {flag} is not a u64 integer: {text:?}");
        std::process::exit(2);
    })
}

fn parse_u32(text: &str, flag: &str) -> u32 {
    let value = parse_u64(text, flag);
    u32::try_from(value).unwrap_or_else(|_| {
        eprintln!("ms2_formula_evidence_report: {flag} {value} does not fit u32");
        std::process::exit(2);
    })
}

fn parse_usize(text: &str, flag: &str) -> usize {
    let value = parse_u64(text, flag);
    usize::try_from(value).unwrap_or_else(|_| {
        eprintln!("ms2_formula_evidence_report: {flag} {value} does not fit usize");
        std::process::exit(2);
    })
}

fn parse_jitter_list(text: &str) -> Vec<f64> {
    let mut out = Vec::new();
    for part in text.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let v: f64 = part.parse().unwrap_or_else(|_| {
            eprintln!("ms2_formula_evidence_report: --precursor-jitter-ppm is not a float list: {text:?}");
            std::process::exit(2);
        });
        if !v.is_finite() || v < 0.0 {
            eprintln!(
                "ms2_formula_evidence_report: --precursor-jitter-ppm values must be finite and >= 0: {text:?}"
            );
            std::process::exit(2);
        }
        out.push(v);
    }
    if out.is_empty() {
        eprintln!("ms2_formula_evidence_report: --precursor-jitter-ppm is empty: {text:?}");
        std::process::exit(2);
    }
    out
}

/// One labeled spectrum's working data (peak lists owned for shuffling).
#[derive(Clone)]
struct WorkSpectrum {
    gold: Composition,
    precursor_mz: u32,
    precursor_uncertainty: u32,
    adduct: u16,
    mz_uncertainty: u32,
    raw_peak_count: u32,
    peak_id: Vec<u32>,
    mz_udalton: Vec<u32>,
    intensity: Vec<f64>,
}

/// Per-spectrum feature output with timing and distribution inputs.
struct WorkOutput {
    evidence: SpectrumEvidence,
    kept_count: usize,
    feature_secs: f64,
    gold_frac: f64,
    gold_intensity: f64,
}

/// Cached fragment evidence of one window candidate at `sigma = 0`:
/// `(expl_count, expl_intensity share)`. The count fraction is recomputed
/// per sigma from the jittered kept-peak count.
type EvidenceCache = HashMap<(usize, Composition), (u32, f64)>;

/// Collect labeled spectra (molecules whose graph builds) in file order.
fn labeled_spectra(file: &ExportFile) -> Vec<WorkSpectrum> {
    let mut out = Vec::new();
    for mol in &file.molecules {
        let Ok(graph) = mol.graph() else { continue };
        let gold = graph.composition();
        for s in &mol.spectra {
            out.push(WorkSpectrum {
                gold,
                precursor_mz: s.precursor_mz_udalton,
                precursor_uncertainty: s.precursor_uncertainty_udalton,
                adduct: s.adduct,
                mz_uncertainty: s.mz_uncertainty_udalton,
                raw_peak_count: s.raw_peak_count,
                peak_id: s.peak_id.clone(),
                mz_udalton: s.mz_udalton.clone(),
                intensity: s.intensity.clone(),
            });
        }
    }
    out
}

/// Seeded random subsample of at most `limit` spectra (Fisher–Yates).
fn subsample(mut spectra: Vec<WorkSpectrum>, limit: usize, seed: u64) -> Vec<WorkSpectrum> {
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rand::Rng;
    if spectra.len() <= limit {
        return spectra;
    }
    let mut rng = StdRng::seed_from_u64(seed);
    for i in (1..spectra.len()).rev() {
        let j = rng.random_range(0..=i);
        spectra.swap(i, j);
    }
    spectra.truncate(limit);
    spectra
}

/// Jitter every spectrum's precursor m/z at `sigma` ppm (pure per index).
fn jitter_spectra(
    spectra: &[WorkSpectrum],
    sigma: f64,
    seed: u64,
    split_tag: u64,
) -> Vec<WorkSpectrum> {
    if sigma <= 0.0 {
        return spectra.to_vec();
    }
    spectra
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let mut j = s.clone();
            j.precursor_mz =
                jitter_precursor_mz(s.precursor_mz, sigma, seed, split_tag, i as u64);
            j
        })
        .collect()
}

/// Shuffle peak lists along a derangement: spectrum `i` gets donor
/// `perm[i]`'s stored peaks, m/z uncertainty and raw count; precursor,
/// adduct and gold stay.
fn shuffle_peaks(spectra: &[WorkSpectrum], perm: &[usize]) -> Vec<WorkSpectrum> {
    spectra
        .iter()
        .enumerate()
        .map(|(i, recipient)| {
            let donor = &spectra[perm[i]];
            WorkSpectrum {
                gold: recipient.gold,
                precursor_mz: recipient.precursor_mz,
                precursor_uncertainty: recipient.precursor_uncertainty,
                adduct: recipient.adduct,
                mz_uncertainty: donor.mz_uncertainty,
                raw_peak_count: donor.raw_peak_count,
                peak_id: donor.peak_id.clone(),
                mz_udalton: donor.mz_udalton.clone(),
                intensity: donor.intensity.clone(),
            }
        })
        .collect()
}

/// Fresh fragment evidence of one candidate on jittered peaks.
fn fresh_evidence(
    candidate: &Composition,
    adduct: u16,
    mz_uncertainty: u32,
    peaks: &[KeptPeak],
    total_intensity: f64,
    parent: Option<u32>,
) -> Result<(u32, f64)> {
    if parent.is_none() {
        return Ok((0, 0.0));
    }
    if let Some(index) =
        build_evidence_index(candidate, adduct, mz_uncertainty, ION_PPM_TENTHS)?
    {
        let mut count: u32 = 0;
        let mut explained = 0.0f64;
        for peak in peaks {
            if index.explains(peak.mz)? {
                count = count.saturating_add(1);
                explained += peak.intensity;
            }
        }
        let share = if total_intensity > 0.0 {
            explained / total_intensity
        } else {
            0.0
        };
        Ok((count, share))
    } else {
        Ok((0, 0.0))
    }
}

/// Feature computation of one spectrum with an optional evidence cache.
///
/// Without a cache this is exactly [`work_one`]: kept peaks, device-order
/// window, per-candidate evidence, gold explanation fractions. With a cache
/// (`sigma > 0`), candidates present at `sigma = 0` reuse their
/// `(expl_count, expl_intensity)`; only new window members build fresh
/// evidence. Residuals, the window, kept peaks and the sharper evidence
/// features always come from the (possibly jittered) spectrum.
fn work_one_cached(
    spectrum: &WorkSpectrum,
    spec_idx: usize,
    domain: &EnumDomain,
    bounds: &RatioBounds,
    dev_limits: &DeviceEnumLimits,
    cache: Option<&EvidenceCache>,
) -> Result<WorkOutput> {
    let started = Instant::now();
    let peaks: Vec<KeptPeak> = kept_peaks(
        &spectrum.peak_id,
        &spectrum.mz_udalton,
        &spectrum.intensity,
        spectrum.precursor_mz,
    )?;
    let query = EnumQuery {
        precursor_mz: spectrum.precursor_mz,
        adduct: spectrum.adduct,
        ppm_tenths: PRECURSOR_PPM_TENTHS,
        precursor_uncertainty: spectrum.precursor_uncertainty,
    };
    let found = enumerate_device_order(domain, bounds, &query, dev_limits)?;
    let parent = spectrum_parent_mass(spectrum.precursor_mz, spectrum.adduct);
    let kept_f = peaks.len() as f64;
    let total_intensity: f64 = peaks.iter().map(|p| p.intensity).sum();
    let tol = precursor_tol_ppm();
    let mut candidates = Vec::with_capacity(found.compositions.len());
    for (c, mass) in found.compositions.iter().zip(found.masses.iter()) {
        let (expl_count, expl_intensity) = match cache.and_then(|cc| cc.get(&(spec_idx, *c))) {
            Some(&(count, share)) => (count, share),
            None => fresh_evidence(
                c,
                spectrum.adduct,
                spectrum.mz_uncertainty,
                &peaks,
                total_intensity,
                parent,
            )?,
        };
        let residual_ppm = match parent {
            Some(p) if p > 0 => (mass.abs_diff(p) as f64) * 1e6 / f64::from(p),
            _ => f64::INFINITY,
        };
        let mut heavy_atoms: u32 = 0;
        for e in HEAVY_ELEMENTS {
            heavy_atoms = heavy_atoms.saturating_add(u32::from(c[e]));
        }
        let dbe_twice = dbe_twice(c).unwrap_or(0);
        let [resid_o1, resid_o1_sq, resid_ln1p, resid_otol] = residual_features(residual_ppm, tol);
        candidates.push(CandidateEvidence {
            counts: *c,
            mass: *mass,
            expl_count,
            expl_count_frac: if kept_f > 0.0 {
                f64::from(expl_count) / kept_f
            } else {
                0.0
            },
            expl_intensity,
            residual_ppm,
            heavy_atoms,
            dbe_twice,
            resid_o1,
            resid_o1_sq,
            resid_ln1p,
            resid_otol,
            expl_minus_max: 0.0,
            expl_rank_frac: 1.0,
        });
    }
    let gold = Some(&spectrum.gold).and_then(|g| found.compositions.iter().position(|c| c == g));
    let mut evidence = SpectrumEvidence {
        candidates,
        gold,
        has_window: !found.compositions.is_empty(),
    };
    apply_sharp_features(&mut evidence);
    let (gold_frac, gold_intensity) = gold_evidence(spectrum, &peaks, parent)?;
    Ok(WorkOutput {
        evidence,
        kept_count: peaks.len(),
        feature_secs: started.elapsed().as_secs_f64(),
        gold_frac,
        gold_intensity,
    })
}

/// Feature computation of one spectrum: kept peaks, device-order window,
/// per-candidate evidence, gold explanation fractions.
fn work_one(
    spectrum: &WorkSpectrum,
    domain: &EnumDomain,
    bounds: &RatioBounds,
    dev_limits: &DeviceEnumLimits,
) -> Result<WorkOutput> {
    work_one_cached(spectrum, usize::MAX, domain, bounds, dev_limits, None)
}

/// `expl_count_frac` / `expl_intensity` of the gold composition itself.
fn gold_evidence(
    spectrum: &WorkSpectrum,
    peaks: &[KeptPeak],
    parent: Option<u32>,
) -> Result<(f64, f64)> {
    let Some(_) = parent else {
        return Ok((0.0, 0.0));
    };
    let Some(index) =
        build_evidence_index(&spectrum.gold, spectrum.adduct, spectrum.mz_uncertainty, ION_PPM_TENTHS)?
    else {
        return Ok((0.0, 0.0));
    };
    let mut count = 0u32;
    let mut explained = 0.0f64;
    let mut total = 0.0f64;
    for peak in peaks {
        total += peak.intensity;
        if index.explains(peak.mz)? {
            count += 1;
            explained += peak.intensity;
        }
    }
    let frac = if peaks.is_empty() {
        0.0
    } else {
        f64::from(count) / peaks.len() as f64
    };
    let share = if total > 0.0 { explained / total } else { 0.0 };
    Ok((frac, share))
}

/// Feature computation over spectra on `std` threads (order-preserving),
/// with an optional read-only evidence cache.
fn work_all_cached(
    spectra: &[WorkSpectrum],
    domain: &Arc<EnumDomain>,
    bounds: &Arc<RatioBounds>,
    dev_limits: &DeviceEnumLimits,
    label: &str,
    cache: Option<&EvidenceCache>,
) -> Vec<WorkOutput> {
    if spectra.is_empty() {
        return Vec::new();
    }
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(spectra.len())
        .max(1);
    let chunk = spectra.len().div_ceil(threads);
    let mut outputs: Vec<WorkOutput> = Vec::with_capacity(spectra.len());
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for (chunk_i, piece) in spectra.chunks(chunk).enumerate() {
            let (domain, bounds) = (Arc::clone(domain), Arc::clone(bounds));
            let dev_limits = *dev_limits;
            let base = chunk_i * chunk;
            handles.push(scope.spawn(move || {
                let mut out = Vec::with_capacity(piece.len());
                for (j, spectrum) in piece.iter().enumerate() {
                    match work_one_cached(spectrum, base + j, &domain, &bounds, &dev_limits, cache)
                    {
                        Ok(done) => out.push(done),
                        Err(e) => {
                            eprintln!("ms2_formula_evidence_report ({label}): {e}");
                            std::process::exit(1);
                        }
                    }
                }
                out
            }));
        }
        for handle in handles {
            match handle.join() {
                Ok(mut piece) => outputs.append(&mut piece),
                Err(_) => {
                    eprintln!("ms2_formula_evidence_report ({label}): worker panicked");
                    std::process::exit(1);
                }
            }
        }
    });
    outputs
}

/// Ranks of the gold formula under all seven rankers for evaluated spectra.
struct RankSet {
    ranks: Vec<[Option<u32>; 7]>,
}

impl RankSet {
    fn column(&self, ranker: usize) -> Vec<Option<u32>> {
        self.ranks.iter().map(|row| row[ranker]).collect()
    }
}

/// Bundled standardised linear model for evaluation.
struct StdLinear<'a> {
    weights: &'a [f64],
    dims: &'a [usize],
    scaler: &'a Standardizer,
}

/// Bundled standardised nonlinear model for evaluation.
struct StdNonlinear<'a> {
    weights: &'a [f64],
    scaler: &'a Standardizer,
    kind: NonlinearKind,
}

/// Evaluate the seven rankers (residual_only is a rule; the rest are
/// standardised linear / nonlinear models with their train scalers).
#[allow(clippy::too_many_arguments)]
fn evaluate(
    outputs: &[WorkOutput],
    prior: &StdLinear,
    pr: &StdLinear,
    pre: &StdLinear,
    pe: &StdLinear,
    nl: &StdNonlinear,
    nlpre: &StdNonlinear,
) -> RankSet {
    let mut ranks = Vec::with_capacity(outputs.len());
    for output in outputs {
        let ev = &output.evidence;
        ranks.push([
            RankRule::ResidualOnly.gold_rank(ev),
            RankRule::LinearStd(prior.weights, prior.dims, prior.scaler).gold_rank(ev),
            RankRule::LinearStd(pr.weights, pr.dims, pr.scaler).gold_rank(ev),
            RankRule::LinearStd(pre.weights, pre.dims, pre.scaler).gold_rank(ev),
            RankRule::LinearStd(pe.weights, pe.dims, pe.scaler).gold_rank(ev),
            RankRule::NonlinearStd(nl.weights, nl.scaler, nl.kind).gold_rank(ev),
            RankRule::NonlinearStd(nlpre.weights, nlpre.scaler, nlpre.kind).gold_rank(ev),
        ]);
    }
    RankSet { ranks }
}

fn percentile_sorted(values: &[f64], p: f64) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    mamba3::models::ms2::dataset::percentile(&sorted, p)
}

fn summarize(values: &[f64]) -> serde_json::Value {
    if values.is_empty() {
        return serde_json::json!({"count": 0, "p10": 0.0, "p50": 0.0, "p90": 0.0});
    }
    serde_json::json!({
        "count": values.len(),
        "p10": percentile_sorted(values, 10.0),
        "p50": percentile_sorted(values, 50.0),
        "p90": percentile_sorted(values, 90.0),
    })
}

/// JSON for one ranker on one evaluation: full and conditional recall@k with
/// bootstrap intervals.
fn ranker_json(ranks: &[Option<u32>], seed: u64) -> serde_json::Value {
    let mut full = serde_json::Map::new();
    let mut cond = serde_json::Map::new();
    for (i, k) in KS.iter().enumerate() {
        let (v, lo, hi) = bootstrap_recall(ranks, *k, BOOTSTRAP_RESAMPLES, seed + i as u64);
        full.insert(
            format!("recall_at_{k}"),
            serde_json::json!({"value": v, "ci95_lo": lo, "ci95_hi": hi}),
        );
        cond.insert(
            format!("recall_at_{k}"),
            serde_json::json!(recall_at_conditional(ranks, *k)),
        );
    }
    let in_window = ranks.iter().filter(|r| r.is_some()).count();
    serde_json::json!({
        "spectra": ranks.len(),
        "gold_in_window": in_window,
        "gold_in_window_rate": recall_at(&ranks.iter().map(|r| r.map(|_| 1)).collect::<Vec<_>>(), 1),
        "full": full,
        "conditional_on_window": cond,
    })
}

/// Gold residual distribution of evaluated spectra (gold in window only).
fn gold_residual_json(outputs: &[WorkOutput]) -> serde_json::Value {
    let mut residuals: Vec<f64> = Vec::new();
    for output in outputs {
        if let Some(g) = output.evidence.gold {
            residuals.push(output.evidence.candidates[g].residual_ppm);
        }
    }
    let frac_below = |tol: f64| {
        if residuals.is_empty() {
            0.0
        } else {
            residuals.iter().filter(|&&r| r < tol).count() as f64 / residuals.len() as f64
        }
    };
    let mut summary = summarize(&residuals);
    if let serde_json::Value::Object(ref mut map) = summary {
        map.insert("frac_below_0p1_ppm".to_string(), serde_json::json!(frac_below(0.1)));
        map.insert("frac_below_1_ppm".to_string(), serde_json::json!(frac_below(1.0)));
    }
    summary
}

/// Window candidate-count distribution of evaluated spectra.
fn window_count_json(outputs: &[WorkOutput]) -> serde_json::Value {
    let counts: Vec<f64> = outputs
        .iter()
        .map(|o| o.evidence.candidates.len() as f64)
        .collect();
    serde_json::json!({
        "count": counts.len(),
        "p50": percentile_sorted(&counts, 50.0),
        "p95": percentile_sorted(&counts, 95.0),
    })
}

fn weights_json(weights: &[f64], dims: &[usize]) -> serde_json::Value {
    let mut map = serde_json::Map::new();
    for (w, &d) in weights.iter().zip(dims.iter()) {
        map.insert(FEATURE_NAMES[d].to_string(), serde_json::json!(*w));
    }
    serde_json::Value::Object(map)
}

fn weights_json_named(weights: &[f64], names: &[String]) -> serde_json::Value {
    let mut map = serde_json::Map::new();
    for (w, n) in weights.iter().zip(names.iter()) {
        map.insert(n.clone(), serde_json::json!(*w));
    }
    serde_json::Value::Object(map)
}

/// Standardisation map of one ranker: per-feature train mean and scale.
fn standardization_json(names: &[String], scaler: &Standardizer) -> serde_json::Value {
    let mut means = serde_json::Map::new();
    let mut scales = serde_json::Map::new();
    for (i, n) in names.iter().enumerate() {
        means.insert(n.clone(), serde_json::json!(scaler.means[i]));
        scales.insert(n.clone(), serde_json::json!(scaler.scales[i]));
    }
    serde_json::json!({"means": means, "scales": scales})
}

fn build_cache(outputs: &[WorkOutput]) -> EvidenceCache {
    let mut cache = EvidenceCache::new();
    for (i, output) in outputs.iter().enumerate() {
        for c in &output.evidence.candidates {
            cache.insert((i, c.counts), (c.expl_count, c.expl_intensity));
        }
    }
    cache
}

fn main() {
    let mut fit: Option<PathBuf> = None;
    let mut train: Option<PathBuf> = None;
    let mut validation: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut window = DEFAULT_WINDOW;
    let mut limit_train: Option<usize> = None;
    let mut limit_validation: Option<usize> = None;
    let mut ratio_margin: u16 = 0;
    let mut lane_visits_max = DEFAULT_LANE_VISITS_MAX;
    let mut epochs = DEFAULT_MAX_EPOCHS;
    let mut seed: u64 = 1;
    let mut jitter_ppms: Vec<f64> = vec![0.0, 1.0, 2.0, 5.0];
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--fit" => fit = args.next().map(PathBuf::from),
            "--train" => train = args.next().map(PathBuf::from),
            "--validation" => validation = args.next().map(PathBuf::from),
            "--out" => out = args.next().map(PathBuf::from),
            "--window" => {
                window = args.next().map(|v| parse_u32(&v, "--window")).unwrap_or_else(|| usage());
            }
            "--limit-train" => {
                limit_train =
                    Some(args.next().map(|v| parse_usize(&v, "--limit-train")).unwrap_or_else(|| usage()));
            }
            "--limit-validation" => {
                limit_validation =
                    Some(args.next().map(|v| parse_usize(&v, "--limit-validation")).unwrap_or_else(|| usage()));
            }
            "--ratio-margin" => {
                let value = args.next().map(|v| parse_u64(&v, "--ratio-margin")).unwrap_or_else(|| usage());
                ratio_margin = u16::try_from(value).unwrap_or_else(|_| {
                    eprintln!("ms2_formula_evidence_report: --ratio-margin {value} does not fit u16");
                    std::process::exit(2);
                });
            }
            "--lane-visits-max" => {
                lane_visits_max =
                    args.next().map(|v| parse_u32(&v, "--lane-visits-max")).unwrap_or_else(|| usage());
            }
            "--epochs" => {
                epochs = args.next().map(|v| parse_usize(&v, "--epochs")).unwrap_or_else(|| usage());
            }
            "--seed" => {
                seed = args.next().map(|v| parse_u64(&v, "--seed")).unwrap_or_else(|| usage());
            }
            "--precursor-jitter-ppm" => {
                let text = args.next().unwrap_or_else(|| usage());
                jitter_ppms = parse_jitter_list(&text);
            }
            _ => usage(),
        }
    }
    let (Some(fit), Some(train), Some(validation), Some(out)) = (fit, train, validation, out) else {
        usage()
    };
    let limit_train = limit_train.unwrap_or(DEFAULT_LIMIT_TRAIN);
    let dev_limits = DeviceEnumLimits {
        lane_visits_max,
        scored_cap: window,
    };

    // Fit the domain and ratio bounds on --fit only, before reading any
    // report molecule.
    let fit_file = ExportFile::load(&fit).unwrap_or_else(|e| {
        eprintln!("ms2_formula_evidence_report: cannot read {}: {e}", fit.display());
        std::process::exit(1);
    });
    let mut fit_comps: Vec<Composition> = Vec::new();
    let mut fit_skipped = 0usize;
    for mol in &fit_file.molecules {
        match mol.graph() {
            Ok(graph) => fit_comps.push(graph.composition()),
            Err(_) => fit_skipped += 1,
        }
    }
    let domain = Arc::new(EnumDomain::from_compositions(fit_comps.iter().copied(), 0).unwrap_or_else(|e| {
        eprintln!("ms2_formula_evidence_report: cannot derive the domain: {e}");
        std::process::exit(1);
    }));
    let bounds = Arc::new(RatioBounds::fit(fit_comps.iter().copied(), ratio_margin).unwrap_or_else(|e| {
        eprintln!("ms2_formula_evidence_report: cannot fit the ratio bounds: {e}");
        std::process::exit(1);
    }));
    if let Err(e) = validate_device_artifacts(&domain, &bounds) {
        eprintln!("ms2_formula_evidence_report: device artifacts invalid: {e}");
        std::process::exit(1);
    }

    let train_file = ExportFile::load(&train).unwrap_or_else(|e| {
        eprintln!("ms2_formula_evidence_report: cannot read {}: {e}", train.display());
        std::process::exit(1);
    });
    let validation_file = ExportFile::load(&validation).unwrap_or_else(|e| {
        eprintln!("ms2_formula_evidence_report: cannot read {}: {e}", validation.display());
        std::process::exit(1);
    });
    let train_labeled = labeled_spectra(&train_file);
    let train_sample = subsample(train_labeled, limit_train, seed);
    let mut validation_labeled = labeled_spectra(&validation_file);
    if let Some(n) = limit_validation {
        validation_labeled.truncate(n);
    }

    let converged_cfg = |max_epochs: usize, seed: u64| ConvergedTrainConfig {
        max_epochs,
        batch_spectra: BATCH_SPECTRA,
        lr_init: ADAM_LR_INIT,
        lr_decay: ADAM_LR_DECAY,
        l2: L2,
        seed,
    };
    let prior_dims: Vec<usize> = PRIOR_DIMS.to_vec();
    let pr_dims: Vec<usize> = PRIOR_PLUS_RESIDUAL_DIMS.to_vec();
    let pre_dims: Vec<usize> = PRIOR_PLUS_RESIDUAL_PLUS_EVIDENCE_DIMS.to_vec();
    let pe_dims: Vec<usize> = PRIOR_PLUS_EVIDENCE4_DIMS.to_vec();
    let nl_kind = NonlinearKind::PriorOnly;
    let nlpre_kind = NonlinearKind::PlusResEv;
    let nl_names: Vec<String> = nonlinear_feature_names();
    let nlpre_names: Vec<String> = nonlinear_plus_feature_names();

    // Shuffled-peak controls: seeded derangements without fixed points, same
    // permutation for every sigma (peak shuffling is jitter-independent).
    let val_perm = derangement(validation_labeled.len(), seed).unwrap_or_else(|e| {
        eprintln!("ms2_formula_evidence_report: validation shuffle failed: {e}");
        std::process::exit(1);
    });
    let train_perm = derangement(train_sample.len(), seed.wrapping_add(1)).unwrap_or_else(|e| {
        eprintln!("ms2_formula_evidence_report: train shuffle failed: {e}");
        std::process::exit(1);
    });

    // Per-sigma evidence caches, filled by the sigma = 0 run.
    let mut train_cache: Option<EvidenceCache> = None;
    let mut val_cache: Option<EvidenceCache> = None;
    let mut train_shuf_cache: Option<EvidenceCache> = None;
    let mut val_shuf_cache: Option<EvidenceCache> = None;

    let mut per_sigma_json = Vec::new();
    let mut all_nestedness_violations: Vec<String> = Vec::new();
    // recall@k per (ranker, sigma) for the text tables.
    let mut table_r4: Vec<Vec<f64>> = vec![Vec::new(); RANKERS.len()];
    let mut table_r1: Vec<Vec<f64>> = vec![Vec::new(); RANKERS.len()];
    let mut table_r16: Vec<Vec<f64>> = vec![Vec::new(); RANKERS.len()];
    let mut table_r128: Vec<Vec<f64>> = vec![Vec::new(); RANKERS.len()];
    // Shuffled-trained-shuffled-eval values for the three evidence rankers.
    let mut table_r4_shuf: Vec<Vec<Option<f64>>> = vec![Vec::new(); RANKERS.len()];
    let mut table_r1_shuf: Vec<Vec<Option<f64>>> = vec![Vec::new(); RANKERS.len()];
    let mut table_r16_shuf: Vec<Vec<Option<f64>>> = vec![Vec::new(); RANKERS.len()];
    let mut table_r128_shuf: Vec<Vec<Option<f64>>> = vec![Vec::new(); RANKERS.len()];
    // Train/validation NLL tables for the final summary.
    let mut table_train_nll: Vec<Vec<f64>> = vec![Vec::new(); RANKERS.len()];
    let mut table_val_nll: Vec<Vec<f64>> = vec![Vec::new(); RANKERS.len()];
    let mut table_epochs: Vec<Vec<String>> = vec![Vec::new(); RANKERS.len()];

    for &sigma in &jitter_ppms {
        let train_j = jitter_spectra(&train_sample, sigma, seed, SPLIT_TRAIN);
        let val_j = jitter_spectra(&validation_labeled, sigma, seed, SPLIT_VALIDATION);
        let train_shuf = shuffle_peaks(&train_j, &train_perm);
        let val_shuf = shuffle_peaks(&val_j, &val_perm);

        let train_out = work_all_cached(
            &train_j,
            &domain,
            &bounds,
            &dev_limits,
            "train",
            train_cache.as_ref(),
        );
        if train_cache.is_none() {
            train_cache = Some(build_cache(&train_out));
        }
        let train_ev: Vec<SpectrumEvidence> =
            train_out.iter().map(|o| o.evidence.clone()).collect();
        let val_out = work_all_cached(
            &val_j,
            &domain,
            &bounds,
            &dev_limits,
            "validation",
            val_cache.as_ref(),
        );
        if val_cache.is_none() {
            val_cache = Some(build_cache(&val_out));
        }

        // FE3: standardise on the TRAIN candidates of this run (per sigma,
        // per shuffle variant), then optimise to convergence on standardised
        // features. Shared base features reuse identical statistics across
        // rankers because each scaler is fitted on the same train
        // candidates, so superset models can replicate subsets (nestedness).
        let scaler_prior = fit_standardizer(&train_ev, &prior_dims);
        let scaler_pr = fit_standardizer(&train_ev, &pr_dims);
        let scaler_pre = fit_standardizer(&train_ev, &pre_dims);
        let scaler_pe = fit_standardizer(&train_ev, &pe_dims);
        let scaler_nl = fit_standardizer_nonlinear(&train_ev, nl_kind);
        let scaler_nlpre = fit_standardizer_nonlinear(&train_ev, nlpre_kind);

        let train_one = |dims: &[usize],
                         scaler: &Standardizer,
                         tag: &str,
                         off: u64,
                         init: Option<&[f64]>| {
            let cfg = converged_cfg(epochs, seed.wrapping_add(off));
            train_softmax_converged_init(&train_ev, dims, scaler, &cfg, init).unwrap_or_else(|e| {
                eprintln!("ms2_formula_evidence_report: {tag} training failed: {e}");
                std::process::exit(1);
            })
        };
        // Warm starts from the prior solution: shared dimensions copy their
        // weights (identical standardisation statistics on the same train
        // candidates), new dimensions start at zero — so every superset
        // reproduces the prior loss exactly at initialisation and nesting
        // holds from the first step.
        let r_prior = train_one(&prior_dims, &scaler_prior, "prior_only", 100, None);
        let init_pr = project_init(&prior_dims, &r_prior.weights, &pr_dims);
        let init_pre = project_init(&prior_dims, &r_prior.weights, &pre_dims);
        let init_pe = project_init(&prior_dims, &r_prior.weights, &pe_dims);
        let init_nl = nonlinear_init_from_prior(&prior_dims, &r_prior.weights, nl_kind);
        let init_nlpre = nonlinear_init_from_prior(&prior_dims, &r_prior.weights, nlpre_kind);
        let r_pr = train_one(&pr_dims, &scaler_pr, "prior_plus_residual", 101, Some(&init_pr));
        let r_pre = train_one(
            &pre_dims,
            &scaler_pre,
            "prior_plus_residual_plus_evidence",
            102,
            Some(&init_pre),
        );
        let r_pe = train_one(&pe_dims, &scaler_pe, "prior_plus_evidence", 103, Some(&init_pe));
        let r_nl = {
            let cfg = converged_cfg(epochs, seed.wrapping_add(104));
            train_nonlinear_converged_init(&train_ev, nl_kind, &scaler_nl, &cfg, Some(&init_nl))
                .unwrap_or_else(|e| {
                    eprintln!("ms2_formula_evidence_report: prior_nonlinear training failed: {e}");
                    std::process::exit(1);
                })
        };
        let r_nlpre = {
            let cfg = converged_cfg(epochs, seed.wrapping_add(105));
            train_nonlinear_converged_init(
                &train_ev,
                nlpre_kind,
                &scaler_nlpre,
                &cfg,
                Some(&init_nlpre),
            )
            .unwrap_or_else(|e| {
                eprintln!("ms2_formula_evidence_report: prior_nonlinear_plus training failed: {e}");
                std::process::exit(1);
            })
        };

        let train_shuf_out = work_all_cached(
            &train_shuf,
            &domain,
            &bounds,
            &dev_limits,
            "shuffled train",
            train_shuf_cache.as_ref(),
        );
        if train_shuf_cache.is_none() {
            train_shuf_cache = Some(build_cache(&train_shuf_out));
        }
        let train_shuf_ev: Vec<SpectrumEvidence> =
            train_shuf_out.iter().map(|o| o.evidence.clone()).collect();
        let scaler_pre_shuf = fit_standardizer(&train_shuf_ev, &pre_dims);
        let scaler_pe_shuf = fit_standardizer(&train_shuf_ev, &pe_dims);
        let scaler_nlpre_shuf = fit_standardizer_nonlinear(&train_shuf_ev, nlpre_kind);
        let r_pre_shuf = {
            let cfg = converged_cfg(epochs, seed.wrapping_add(202));
            // Prior features are peak-independent, so the real prior solution
            // is a valid nested warm start on shuffled train too.
            let init = project_init(&prior_dims, &r_prior.weights, &pre_dims);
            train_softmax_converged_init(&train_shuf_ev, &pre_dims, &scaler_pre_shuf, &cfg, Some(&init))
                .unwrap_or_else(|e| {
                    eprintln!("ms2_formula_evidence_report: shuffled evidence training failed: {e}");
                    std::process::exit(1);
                })
        };
        let r_pe_shuf = {
            let cfg = converged_cfg(epochs, seed.wrapping_add(203));
            let init = project_init(&prior_dims, &r_prior.weights, &pe_dims);
            train_softmax_converged_init(&train_shuf_ev, &pe_dims, &scaler_pe_shuf, &cfg, Some(&init))
                .unwrap_or_else(|e| {
                    eprintln!("ms2_formula_evidence_report: shuffled evidence training failed: {e}");
                    std::process::exit(1);
                })
        };
        let r_nlpre_shuf = {
            let cfg = converged_cfg(epochs, seed.wrapping_add(205));
            let init = nonlinear_init_from_prior(&prior_dims, &r_prior.weights, nlpre_kind);
            train_nonlinear_converged_init(&train_shuf_ev, nlpre_kind, &scaler_nlpre_shuf, &cfg, Some(&init))
                .unwrap_or_else(|e| {
                    eprintln!("ms2_formula_evidence_report: shuffled nonlinear training failed: {e}");
                    std::process::exit(1);
                })
        };

        let val_shuf_out = work_all_cached(
            &val_shuf,
            &domain,
            &bounds,
            &dev_limits,
            "shuffled validation",
            val_shuf_cache.as_ref(),
        );
        if val_shuf_cache.is_none() {
            val_shuf_cache = Some(build_cache(&val_shuf_out));
        }

        let val_ev: Vec<SpectrumEvidence> =
            val_out.iter().map(|o| o.evidence.clone()).collect();
        let val_shuf_ev: Vec<SpectrumEvidence> =
            val_shuf_out.iter().map(|o| o.evidence.clone()).collect();
        // Validation NLLs use the TRAIN scaler of each ranker (same affine
        // map at evaluation), with the same L2 term as training.
        let nll = |w: &[f64], dims: &[usize], scaler: &Standardizer, data: &[SpectrumEvidence]| {
            standardized_nll(data, dims, scaler, w, L2).unwrap_or(f64::NAN)
        };
        let nll_nl =
            |w: &[f64], kind: NonlinearKind, scaler: &Standardizer, data: &[SpectrumEvidence]| {
                standardized_nll_nonlinear(data, kind, scaler, w, L2).unwrap_or(f64::NAN)
            };
        let train_nll_real = std::collections::HashMap::from([
            ("prior_only".to_string(), r_prior.train_nll),
            ("prior_plus_residual".to_string(), r_pr.train_nll),
            (
                "prior_plus_residual_plus_evidence".to_string(),
                r_pre.train_nll,
            ),
            ("prior_plus_evidence".to_string(), r_pe.train_nll),
            ("prior_nonlinear".to_string(), r_nl.train_nll),
            (
                "prior_nonlinear_plus_residual_plus_evidence".to_string(),
                r_nlpre.train_nll,
            ),
        ]);
        let train_nll_shuf = std::collections::HashMap::from([
            (
                "prior_plus_residual_plus_evidence".to_string(),
                r_pre_shuf.train_nll,
            ),
            ("prior_plus_evidence".to_string(), r_pe_shuf.train_nll),
            (
                "prior_nonlinear_plus_residual_plus_evidence".to_string(),
                r_nlpre_shuf.train_nll,
            ),
        ]);
        // Enforced nestedness check on train NLLs (real and shuffled).
        let mut sigma_violations: Vec<String> = Vec::new();
        for v in check_nestedness(&train_nll_real, NESTEDNESS_TOL) {
            sigma_violations.push(format!("sigma={sigma} real-train: {v}"));
        }
        for v in check_nestedness(&train_nll_shuf, NESTEDNESS_TOL) {
            sigma_violations.push(format!("sigma={sigma} shuffled-train: {v}"));
        }
        for v in &sigma_violations {
            eprintln!("ms2_formula_evidence_report: {v}");
            all_nestedness_violations.push(v.clone());
        }

        let eval_real = evaluate(
            &val_out,
            &StdLinear { weights: &r_prior.weights, dims: &prior_dims, scaler: &scaler_prior },
            &StdLinear { weights: &r_pr.weights, dims: &pr_dims, scaler: &scaler_pr },
            &StdLinear { weights: &r_pre.weights, dims: &pre_dims, scaler: &scaler_pre },
            &StdLinear { weights: &r_pe.weights, dims: &pe_dims, scaler: &scaler_pe },
            &StdNonlinear { weights: &r_nl.weights, scaler: &scaler_nl, kind: nl_kind },
            &StdNonlinear { weights: &r_nlpre.weights, scaler: &scaler_nlpre, kind: nlpre_kind },
        );
        // Shuffled-train models evaluate on shuffled validation with their
        // shuffled-train scalers; peak-free rankers reuse real models.
        let eval_shuf_shuf = evaluate(
            &val_shuf_out,
            &StdLinear { weights: &r_prior.weights, dims: &prior_dims, scaler: &scaler_prior },
            &StdLinear { weights: &r_pr.weights, dims: &pr_dims, scaler: &scaler_pr },
            &StdLinear {
                weights: &r_pre_shuf.weights,
                dims: &pre_dims,
                scaler: &scaler_pre_shuf,
            },
            &StdLinear {
                weights: &r_pe_shuf.weights,
                dims: &pe_dims,
                scaler: &scaler_pe_shuf,
            },
            &StdNonlinear { weights: &r_nl.weights, scaler: &scaler_nl, kind: nl_kind },
            &StdNonlinear {
                weights: &r_nlpre_shuf.weights,
                scaler: &scaler_nlpre_shuf,
                kind: nlpre_kind,
            },
        );

        let has_shuf = |r: usize| r == 3 || r == 4 || r == 6;
        let mut rankers_json = serde_json::Map::new();
        for (r, name) in RANKERS.iter().enumerate() {
            let base = seed
                .wrapping_add(1_000_000)
                .wrapping_add(RANKER_OFFSETS[r] * 100);
            let real = ranker_json(&eval_real.column(r), base);
            // Full-dataset recall@k for the text tables.
            let get = |v: &serde_json::Value, k: u32| {
                v["full"][format!("recall_at_{k}")]["value"]
                    .as_f64()
                    .unwrap_or(f64::NAN)
            };
            table_r1[r].push(get(&real, 1));
            table_r4[r].push(get(&real, 4));
            table_r16[r].push(get(&real, 16));
            table_r128[r].push(get(&real, 128));
            let entry = if has_shuf(r) {
                let shuf = ranker_json(&eval_shuf_shuf.column(r), base + 2);
                table_r1_shuf[r].push(shuf["full"]["recall_at_1"]["value"].as_f64());
                table_r4_shuf[r].push(shuf["full"]["recall_at_4"]["value"].as_f64());
                table_r16_shuf[r].push(shuf["full"]["recall_at_16"]["value"].as_f64());
                table_r128_shuf[r].push(shuf["full"]["recall_at_128"]["value"].as_f64());
                serde_json::json!({
                    "real_train_real_eval": real,
                    "shuffled_train_shuffled_eval": shuf,
                })
            } else {
                table_r1_shuf[r].push(None);
                table_r4_shuf[r].push(None);
                table_r16_shuf[r].push(None);
                table_r128_shuf[r].push(None);
                serde_json::json!({
                    "real_train_real_eval": real,
                })
            };
            rankers_json.insert(name.to_string(), entry);
        }
        // Train/validation NLL tables (real-train-real-eval; shuffled in JSON).
        let val_nll_vals = [
            nll(&r_prior.weights, &prior_dims, &scaler_prior, &val_ev),
            nll(&r_pr.weights, &pr_dims, &scaler_pr, &val_ev),
            nll(&r_pre.weights, &pre_dims, &scaler_pre, &val_ev),
            nll(&r_pe.weights, &pe_dims, &scaler_pe, &val_ev),
            nll_nl(&r_nl.weights, nl_kind, &scaler_nl, &val_ev),
            nll_nl(&r_nlpre.weights, nlpre_kind, &scaler_nlpre, &val_ev),
        ];
        // Column 0 (residual_only) has no NLL.
        table_train_nll[0].push(f64::NAN);
        table_val_nll[0].push(f64::NAN);
        table_epochs[0].push("n/a".to_string());
        let train_vals = [
            r_prior.train_nll,
            r_pr.train_nll,
            r_pre.train_nll,
            r_pe.train_nll,
            r_nl.train_nll,
            r_nlpre.train_nll,
        ];
        let epoch_strs = [
            format!("{}/{}", r_prior.epochs_used, r_prior.converged),
            format!("{}/{}", r_pr.epochs_used, r_pr.converged),
            format!("{}/{}", r_pre.epochs_used, r_pre.converged),
            format!("{}/{}", r_pe.epochs_used, r_pe.converged),
            format!("{}/{}", r_nl.epochs_used, r_nl.converged),
            format!("{}/{}", r_nlpre.epochs_used, r_nlpre.converged),
        ];
        for k in 0..6 {
            table_train_nll[k + 1].push(train_vals[k]);
            table_val_nll[k + 1].push(val_nll_vals[k]);
            table_epochs[k + 1].push(epoch_strs[k].clone());
        }
        let prior_names: Vec<String> =
            prior_dims.iter().map(|&d| FEATURE_NAMES[d].to_string()).collect();
        let pr_names: Vec<String> =
            pr_dims.iter().map(|&d| FEATURE_NAMES[d].to_string()).collect();
        let pre_names: Vec<String> =
            pre_dims.iter().map(|&d| FEATURE_NAMES[d].to_string()).collect();
        let pe_names: Vec<String> =
            pe_dims.iter().map(|&d| FEATURE_NAMES[d].to_string()).collect();
        per_sigma_json.push(serde_json::json!({
            "sigma_ppm": sigma,
            "spectra": {
                "train_sampled": train_out.len(),
                "validation_used": val_out.len(),
            },
            "gold_residual_ppm": gold_residual_json(&val_out),
            "candidates_in_window": window_count_json(&val_out),
            "rankers": rankers_json,
            "weights": {
                "prior_only": {
                    "features": prior_names,
                    "units": "standardised (zero-mean unit-variance over train candidates)",
                    "real_train": weights_json(&r_prior.weights, &prior_dims),
                },
                "prior_plus_residual": {
                    "features": pr_names,
                    "units": "standardised (zero-mean unit-variance over train candidates)",
                    "real_train": weights_json(&r_pr.weights, &pr_dims),
                },
                "prior_plus_residual_plus_evidence": {
                    "features": pre_names,
                    "units": "standardised (zero-mean unit-variance over train candidates)",
                    "real_train": weights_json(&r_pre.weights, &pre_dims),
                    "shuffled_train": weights_json(&r_pre_shuf.weights, &pre_dims),
                },
                "prior_plus_evidence": {
                    "features": pe_names,
                    "units": "standardised (zero-mean unit-variance over train candidates)",
                    "real_train": weights_json(&r_pe.weights, &pe_dims),
                    "shuffled_train": weights_json(&r_pe_shuf.weights, &pe_dims),
                },
                "prior_nonlinear": {
                    "features": nl_names,
                    "units": "standardised after forming pairwise products",
                    "real_train": weights_json_named(&r_nl.weights, &nl_names),
                },
                "prior_nonlinear_plus_residual_plus_evidence": {
                    "features": nlpre_names,
                    "units": "standardised after forming pairwise products",
                    "real_train": weights_json_named(&r_nlpre.weights, &nlpre_names),
                    "shuffled_train": weights_json_named(&r_nlpre_shuf.weights, &nlpre_names),
                },
            },
            "standardization": {
                "fitted_on": "train candidates of this sigma and shuffle variant (mean/std per feature; zero-variance scale 1; non-finite raw maps to 0)",
                "prior_only": { "real_train": standardization_json(&prior_names, &scaler_prior) },
                "prior_plus_residual": { "real_train": standardization_json(&pr_names, &scaler_pr) },
                "prior_plus_residual_plus_evidence": {
                    "real_train": standardization_json(&pre_names, &scaler_pre),
                    "shuffled_train": standardization_json(&pre_names, &scaler_pre_shuf),
                },
                "prior_plus_evidence": {
                    "real_train": standardization_json(&pe_names, &scaler_pe),
                    "shuffled_train": standardization_json(&pe_names, &scaler_pe_shuf),
                },
                "prior_nonlinear": { "real_train": standardization_json(&nl_names, &scaler_nl) },
                "prior_nonlinear_plus_residual_plus_evidence": {
                    "real_train": standardization_json(&nlpre_names, &scaler_nlpre),
                    "shuffled_train": standardization_json(&nlpre_names, &scaler_nlpre_shuf),
                },
            },
            "train_nll": {
                "prior_only": r_prior.train_nll,
                "prior_plus_residual": r_pr.train_nll,
                "prior_plus_residual_plus_evidence": r_pre.train_nll,
                "prior_plus_evidence": r_pe.train_nll,
                "prior_nonlinear": r_nl.train_nll,
                "prior_nonlinear_plus_residual_plus_evidence": r_nlpre.train_nll,
            },
            "train_nll_shuffled": {
                "prior_plus_residual_plus_evidence": r_pre_shuf.train_nll,
                "prior_plus_evidence": r_pe_shuf.train_nll,
                "prior_nonlinear_plus_residual_plus_evidence": r_nlpre_shuf.train_nll,
            },
            "validation_nll": {
                "prior_only": val_nll_vals[0],
                "prior_plus_residual": val_nll_vals[1],
                "prior_plus_residual_plus_evidence": val_nll_vals[2],
                "prior_plus_evidence": val_nll_vals[3],
                "prior_nonlinear": val_nll_vals[4],
                "prior_nonlinear_plus_residual_plus_evidence": val_nll_vals[5],
            },
            "validation_nll_shuffled": {
                "prior_plus_residual_plus_evidence": nll(&r_pre_shuf.weights, &pre_dims, &scaler_pre_shuf, &val_shuf_ev),
                "prior_plus_evidence": nll(&r_pe_shuf.weights, &pe_dims, &scaler_pe_shuf, &val_shuf_ev),
                "prior_nonlinear_plus_residual_plus_evidence": nll_nl(&r_nlpre_shuf.weights, nlpre_kind, &scaler_nlpre_shuf, &val_shuf_ev),
            },
            "training": {
                "prior_only": {"real_train": {"epochs_used": r_prior.epochs_used, "converged": r_prior.converged}},
                "prior_plus_residual": {"real_train": {"epochs_used": r_pr.epochs_used, "converged": r_pr.converged}},
                "prior_plus_residual_plus_evidence": {
                    "real_train": {"epochs_used": r_pre.epochs_used, "converged": r_pre.converged},
                    "shuffled_train": {"epochs_used": r_pre_shuf.epochs_used, "converged": r_pre_shuf.converged},
                },
                "prior_plus_evidence": {
                    "real_train": {"epochs_used": r_pe.epochs_used, "converged": r_pe.converged},
                    "shuffled_train": {"epochs_used": r_pe_shuf.epochs_used, "converged": r_pe_shuf.converged},
                },
                "prior_nonlinear": {"real_train": {"epochs_used": r_nl.epochs_used, "converged": r_nl.converged}},
                "prior_nonlinear_plus_residual_plus_evidence": {
                    "real_train": {"epochs_used": r_nlpre.epochs_used, "converged": r_nlpre.converged},
                    "shuffled_train": {"epochs_used": r_nlpre_shuf.epochs_used, "converged": r_nlpre_shuf.converged},
                },
            },
            "nestedness_violations": sigma_violations,
        }));
    }

    let file_name = |p: &PathBuf| {
        p.file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| p.display().to_string())
    };
    let report = serde_json::json!({
        "schema_version": 3,
        "implementation": "rust mamba3::models::ms2::formula_evidence_ref (host-only peak-evidence ranking, FE3 standardised converged training)",
        "timing": "per-spectrum wall clock via std::time::Instant, std threads, release build",
        "fit_file": file_name(&fit),
        "train_file": file_name(&train),
        "validation_file": file_name(&validation),
        "chemistry": mamba3::models::ms2::chem::CHEMISTRY_VERSION,
        "seed": seed,
        "window": window,
        "lane_visits_max": lane_visits_max,
        "precursor_ppm_tenths": PRECURSOR_PPM_TENTHS,
        "ion_ppm_tenths": ION_PPM_TENTHS,
        "precursor_tol_ppm": precursor_tol_ppm(),
        "ratio_margin": ratio_margin,
        "limit_train": limit_train,
        "limit_validation": limit_validation,
        "max_epochs": epochs,
        "convergence": "full-batch Adam with decaying step lr_init/(1+lr_decay*epoch) until relative train-objective decrease over 20 epochs < 1e-5",
        "warm_start": "every superset model starts from the prior_only solution on shared dimensions (identical standardisation statistics on the same train candidates), zeros elsewhere — nesting holds at initialisation",
        "batch_spectra": "full-batch (one Adam step per epoch)",
        "adam_lr_init": ADAM_LR_INIT,
        "adam_lr_decay": ADAM_LR_DECAY,
        "l2": L2,
        "nestedness_tol": NESTEDNESS_TOL,
        "standardization": "mean/std over TRAIN candidates per sigma and shuffle variant; zero-variance scale 1; non-finite raw maps to 0; same map at evaluation; weights in standardised units",
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "peak_selection": "device twin N=128; precursor-region peaks above precursor+2Da excluded by the contract filter (recomputed from the jittered precursor); nothing else excluded",
        "shuffle": "seeded derangement without fixed points (validation seed, train seed+1); donor peak_id/mz/intensity/raw count/mz uncertainty; precursor/adduct/gold kept; same permutation for every sigma",
        "jitter": "mz * (1 + e * 1e-6), e ~ Normal(0, sigma) truncated to |e| <= 3 sigma, seeded per (seed, split, spectrum index, sigma); split tags train=0 validation=1; shuffled spectra inherit the recipient jittered precursor",
        "evidence_cache": "per group (train, validation, shuffled train, shuffled validation) the sigma=0 run fills (spectrum index, composition) -> (expl_count, expl_intensity); sigma>0 reuses it and builds fresh evidence only for new window members; residuals, window, kept peaks and sharp features always recomputed",
        "precursor_jitter_ppms": jitter_ppms,
        "fit": {
            "molecules": fit_file.molecules.len(),
            "compositions_used": fit_comps.len(),
            "skipped_molecules": fit_skipped,
            "domain_bytes": domain.bytes(),
            "bounds_bytes": bounds.bytes(),
        },
        "per_sigma": per_sigma_json,
        "nestedness_violations": all_nestedness_violations,
    });
    if let Some(parent) = out.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent).unwrap_or_else(|e| {
            eprintln!("ms2_formula_evidence_report: cannot create {}: {e}", parent.display());
            std::process::exit(1);
        });
    }
    std::fs::write(&out, serde_json::to_string_pretty(&report).unwrap()).unwrap_or_else(|e| {
        eprintln!("ms2_formula_evidence_report: cannot write {}: {e}", out.display());
        std::process::exit(1);
    });

    // Compact text tables: rows = rankers, columns = sigma.
    let header: Vec<String> =
        jitter_ppms.iter().map(|s| format!("sigma={s}")).collect();
    let print_table = |title: &str, table: &[Vec<f64>], shuf: &[Vec<Option<f64>>]| {
        println!("{title} (shuffled-trained-shuffled-eval in parentheses for evidence rankers):");
        println!("ranker                              | {}", header.join(" | "));
        for (r, name) in RANKERS.iter().enumerate() {
            let mut cells = Vec::new();
            for (j, _) in jitter_ppms.iter().enumerate() {
                if r == 3 || r == 4 || r == 6 {
                    cells.push(format!(
                        "{:.4} ({:.4})",
                        table[r][j],
                        shuf[r][j].unwrap_or(f64::NAN)
                    ));
                } else {
                    cells.push(format!("{:.4}", table[r][j]));
                }
            }
            println!("{name:37} | {}", cells.join(" | "));
        }
    };
    print_table("recall@1", &table_r1, &table_r1_shuf);
    print_table("recall@4", &table_r4, &table_r4_shuf);
    print_table("recall@16", &table_r16, &table_r16_shuf);
    print_table("recall@128", &table_r128, &table_r128_shuf);
    println!("train NLL per ranker and sigma (mean NLL plus L2 on standardised weights):");
    println!("ranker                              | {}", header.join(" | "));
    for (r, name) in RANKERS.iter().enumerate() {
        let cells: Vec<String> = table_train_nll[r].iter().map(|v| format!("{v:.4}")).collect();
        println!("{name:37} | {}", cells.join(" | "));
    }
    println!("validation NLL per ranker and sigma (train scaler, same L2 term):");
    println!("ranker                              | {}", header.join(" | "));
    for (r, name) in RANKERS.iter().enumerate() {
        let cells: Vec<String> = table_val_nll[r].iter().map(|v| format!("{v:.4}")).collect();
        println!("{name:37} | {}", cells.join(" | "));
    }
    println!("training epochs_used/converged per ranker and sigma:");
    println!("ranker                              | {}", header.join(" | "));
    for (r, name) in RANKERS.iter().enumerate() {
        println!("{name:37} | {}", table_epochs[r].join(" | "));
    }

    // Keep the old single-sigma summary line shape for sigma = 0 so the
    // reproduction check against the first report is a direct comparison.
    if let Some(zero) = jitter_ppms.iter().position(|&s| s <= 0.0) {
        println!(
            "sigma=0 check: residual_only recall@4={:.4} prior_only recall@4={:.4} (first report: 0.7073 / 0.2255)",
            table_r4[0][zero], table_r4[1][zero]
        );
    }

    if !all_nestedness_violations.is_empty() {
        eprintln!(
            "ms2_formula_evidence_report: {} nestedness violation(s):",
            all_nestedness_violations.len()
        );
        for v in &all_nestedness_violations {
            eprintln!("ms2_formula_evidence_report: {v}");
        }
        std::process::exit(1);
    }
}
