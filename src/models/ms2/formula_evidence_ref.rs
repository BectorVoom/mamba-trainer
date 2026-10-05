//! Peak-explanation evidence for enumerated parent formulas (plan P4.1/P4.3).
//!
//! Host-only experiment code: no tensors, no kernels, no neural code. It
//! measures how much ranking signal sits in simple, exactly computable
//! peak-explanation features of a candidate parent formula.
//!
//! * [`kept_peaks`] selects the `N = 128` kept peaks through the device peak
//!   selection twin, so these are the peaks the device sees. Peaks above
//!   `precursor + 2 Da` are excluded there, exactly as contracts §2 filter 1
//!   requires; nothing else is excluded.
//! * [`EvidenceIndex`] tests, per candidate composition, whether a kept peak
//!   has at least one accepted sub-composition hypothesis under the
//!   [`ion_assign`](super::ion::ion_assign) rule. It enumerates the heavy
//!   sub-vectors of the candidate once, sorts them by mass, and tests every
//!   peak by binary search plus at most a few hydrogen verdicts. The verdict
//!   arithmetic (target mass, tolerance, window, hydrogen interval, §5
//!   decision) is the lane's arithmetic step for step; an ambiguous verdict
//!   explains nothing, exactly as `accepted` counts only accepts.
//! * [`spectrum_evidence`] computes every candidate's features:
//!   `expl_count`, `expl_intensity`, `expl_count_frac`, `residual_ppm`,
//!   `heavy_atoms`, `dbe` and `ln(1 + count)` for the 10 elements.
//! * [`train_softmax`] fits a multinomial logistic ranking model (softmax over
//!   a spectrum's candidates) with mini-batch Adam in `f64`, seeded.
//! * [`Standardizer`] (FE3) standardises every feature to zero mean and unit
//!   variance over the train candidates of the run; [`train_softmax_converged`]
//!   optimises the same softmax objective on standardised features with
//!   full-batch Adam and a decaying step until the relative train-objective
//!   decrease over 20 epochs is below `1e-5`; [`check_nestedness`] enforces the
//!   subset-nesting inequalities on train NLLs.
//! * [`nonlinear_raw_vector`] (FE3) forms the prior features plus all pairwise
//!   products of the 12 prior features (standardised after forming products).
//! * [`rank_order`], [`recall_at`] and [`bootstrap_recall`] evaluate the
//!   rankers; [`derangement`] builds the seeded fixed-point-free peak
//!   permutation of the shuffled-peak control.

use std::collections::HashMap;
use std::sync::OnceLock;

use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

use crate::error::{Error, Result};

use super::chem::{
    Composition, ELECTRON_MASS, ELECTRON_RESIDUAL_NDA, ELEMENTS, HYDROGEN, composition_mass,
    decide, parent_mass, tolerance,
};
use super::dataset::percentile;
use super::formula_enum::HEAVY_ELEMENTS;
use super::ion::{IonLimits, ion_assign};
use super::twin;

/// Kept peaks per spectrum in the device peak selection (contracts §3.3).
pub const N_KEEP: usize = 128;

/// Fragment tolerance of the ion rule in tenths of a ppm (contracts §6).
pub const ION_PPM_TENTHS: u32 = 100;

/// Precursor tolerance of the formula window in tenths of a ppm (§6).
pub const PRECURSOR_PPM_TENTHS: u32 = 200;

/// Feature names of [`feature_vector`]: `ln(1 + count)` in [`ELEMENTS`] order,
/// then heavy-atom total, DBE, explained-count fraction, explained intensity
/// share and precursor residual in ppm, followed by the FE2 residual
/// transforms (`|r| / 1 ppm`, `(|r| / 1 ppm)^2`, `ln(1 + |r| / 0.1 ppm)`,
/// `|r| / tolerance` with `tolerance = 20 ppm`, the precursor window
/// tolerance) and the sharper evidence features (`expl_intensity` minus the
/// spectrum maximum, 0 at the maximum; the fraction of the spectrum's
/// candidates explaining at least as much intensity).
///
/// The first fifteen positions are exactly the pre-FE2 features in the old
/// order, so `sigma = 0` values are bit-identical to the first report.
pub const FEATURE_NAMES: [&str; 21] = [
    "ln1p_C",
    "ln1p_H",
    "ln1p_N",
    "ln1p_O",
    "ln1p_F",
    "ln1p_P",
    "ln1p_S",
    "ln1p_Cl",
    "ln1p_Br",
    "ln1p_I",
    "heavy_atoms",
    "dbe",
    "expl_count_frac",
    "expl_intensity",
    "residual_ppm",
    "abs_resid_over_1ppm",
    "abs_resid_sq_over_1ppm",
    "ln1p_abs_resid_over_01ppm",
    "abs_resid_over_tol",
    "expl_minus_max",
    "expl_rank_frac",
];

/// Total feature count.
pub const N_FEATURES: usize = 21;

/// Indices of the stand-in prior features (what the neural head already sees).
pub const PRIOR_DIMS: [usize; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];

/// Indices of the three peak-evidence features.
pub const EVIDENCE_DIMS: [usize; 3] = [12, 13, 14];

/// All fifteen feature indices (prior plus evidence).
pub const FULL_DIMS: [usize; 15] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14];

/// Indices of the four residual transform features (FE2).
pub const RESIDUAL_DIMS: [usize; 4] = [15, 16, 17, 18];

/// Indices of the four evidence features without any residual (FE2): the two
/// base evidence features plus the two sharper ones.
pub const EVIDENCE4_DIMS: [usize; 4] = [12, 13, 19, 20];

/// Prior plus the four residual transforms (FE2, 16 dims).
pub const PRIOR_PLUS_RESIDUAL_DIMS: [usize; 16] =
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 16, 17, 18];

/// Prior plus the four evidence features, no residual (FE2, 16 dims).
pub const PRIOR_PLUS_EVIDENCE4_DIMS: [usize; 16] =
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 19, 20];

/// Prior plus residual transforms plus evidence (FE2, 20 dims).
pub const PRIOR_PLUS_RESIDUAL_PLUS_EVIDENCE_DIMS: [usize; 20] =
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20];

/// Number of prior features (FE3 nonlinear base).
pub const N_PRIOR_FE3: usize = 12;

/// Pairwise products of the 12 prior features with `i <= j` (FE3): 78.
pub const N_PAIR_FE3: usize = 78;

/// Prior features plus all pairwise products (FE3, 90 dims).
pub const N_NONLINEAR_FE3: usize = 90;

/// Prior-nonlinear plus residual transforms plus evidence features (FE3):
/// 90 nonlinear plus 4 residual plus 4 evidence (`expl_count_frac`,
/// `expl_intensity`, `expl_minus_max`, `expl_rank_frac`), 98 dims.
pub const N_NONLINEAR_PLUS_FE3: usize = 98;

/// Names of the 90 FE3 nonlinear features: the 12 prior names, then
/// `a*b` for `i <= j` over the prior names in order.
pub fn nonlinear_feature_names() -> Vec<String> {
    let mut out = Vec::with_capacity(N_NONLINEAR_FE3);
    for &d in PRIOR_DIMS.iter() {
        out.push(FEATURE_NAMES[d].to_string());
    }
    for i in 0..N_PRIOR_FE3 {
        for j in i..N_PRIOR_FE3 {
            out.push(format!(
                "{}*{}",
                FEATURE_NAMES[PRIOR_DIMS[i]], FEATURE_NAMES[PRIOR_DIMS[j]]
            ));
        }
    }
    out
}

/// Names of the 98 FE3 nonlinear-plus features: the 90 nonlinear names,
/// then the 4 residual names, then the 4 evidence names.
pub fn nonlinear_plus_feature_names() -> Vec<String> {
    let mut out = nonlinear_feature_names();
    for &d in RESIDUAL_DIMS.iter() {
        out.push(FEATURE_NAMES[d].to_string());
    }
    for &d in EVIDENCE4_DIMS.iter() {
        out.push(FEATURE_NAMES[d].to_string());
    }
    out
}

/// Nominal precursor window tolerance in ppm (`PRECURSOR_PPM_TENTHS / 10`).
/// The per-spectrum integer tolerance `tolerance(precursor, 200)` floored to
/// `u32` converts back to this nominal value up to the `u32` floor and the
/// parent/precursor mass scaling, so the normalized feature
/// `|r| / tolerance` equals `abs_diff / tol_udalton` up to that floor.
pub fn precursor_tol_ppm() -> f64 {
    f64::from(PRECURSOR_PPM_TENTHS) / 10.0
}

/// `ln(1 + n)` for every `u16` count, shared by training and evaluation so
/// both read identical values.
fn ln1p_table() -> &'static [f64] {
    static TABLE: OnceLock<Vec<f64>> = OnceLock::new();
    TABLE.get_or_init(|| {
        (0..=u16::MAX)
            .map(|n| (1.0 + f64::from(n)).ln())
            .collect()
    })
}

/// One kept peak: its integer m/z and its stored linear intensity.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KeptPeak {
    /// m/z in integer units of 10⁻⁶ Da.
    pub mz: u32,
    /// Stored linear intensity.
    pub intensity: f64,
}

/// The `N = 128` kept peaks of an export spectrum through the host twin of
/// device peak selection, in m/z order with padding removed.
///
/// This is the peak-filtering twin of `examples/ms2_ion_report.rs`: the
/// contract §2 filter (`0 < mz <= precursor + 2 Da`, relative intensity
/// `>= 1e-3`, top 128 by intensity with ties by peak position, then m/z
/// order). Precursor-region peaks above `precursor + 2 Da` are excluded by
/// that filter, exactly as the contract excludes them; no further exclusion
/// is applied here.
pub fn kept_peaks(
    peak_id: &[u32],
    mz_udalton: &[u32],
    intensity: &[f64],
    precursor_mz_udalton: u32,
) -> Result<Vec<KeptPeak>> {
    if peak_id.len() != mz_udalton.len() || peak_id.len() != intensity.len() {
        return Err(Error::config(format!(
            "kept_peaks: peak list length mismatch ({} / {} / {})",
            peak_id.len(),
            mz_udalton.len(),
            intensity.len()
        )));
    }
    let n_raw = mz_udalton.len();
    let intensity_f32: Vec<f32> = intensity.iter().map(|&v| v as f32).collect();
    let mut meta = vec![0u32; 8];
    meta[0] = n_raw as u32;
    meta[1] = precursor_mz_udalton;
    let sel = twin::peak_select(
        mz_udalton,
        &intensity_f32,
        &meta,
        1,
        n_raw,
        N_KEEP,
        0,
    );
    let mut out = Vec::new();
    for p in 0..N_KEEP {
        let raw = sel.kept[p * 3] as usize;
        if sel.kept[p * 3] == u32::MAX {
            continue;
        }
        if raw >= mz_udalton.len() {
            return Err(Error::config(format!(
                "kept_peaks: twin raw index {raw} outside {} stored peaks",
                mz_udalton.len()
            )));
        }
        out.push(KeptPeak {
            mz: mz_udalton[raw],
            intensity: intensity[raw],
        });
    }
    Ok(out)
}

/// Fast per-candidate peak-explanation test, exactly equivalent to
/// `ion_assign(parent, adduct, peak, U, ppm, large_limits).accepted > 0`.
///
/// The heavy sub-vectors of `parent` (every non-empty heavy count vector
/// below the parent, carbon least significant in enumeration order) are
/// enumerated once with their integer mass and rounding-residual sum, then
/// sorted by mass. Each peak is tested by binary search: vectors above the
/// window end break the scan, and every remaining vector contributes the
/// closed-form hydrogen interval of the lane, each with its own §5 verdict.
///
/// Equivalence with the lane, case by case (adducts 1 and 2, the supported
/// domain):
/// * padding peaks (`mz == 0`), empty parents and `U == u32::MAX` are never
///   searched in either implementation;
/// * the target `t = mz ± 549` (electron mass, signed by adduct), the
///   tolerance at the observed m/z, `half = tol + U + E_ion` with
///   `E_ion = ceil((parent residuals + 3 * H residual + electron residual) /
///   1000)`, and the `half > m_H` scope gate are identical; a gated peak is
///   unexplained in both;
/// * below the gate the window spans at most three hydrogen counts, so the
///   lane's `min(width, 3)` truncation never engages and the full
///   `[h_lo, h_hi]` interval here matches the lane's hypotheses one for one;
/// * every partial sum fits `u32` whenever the total mass is in the window
///   (all terms are non-negative), so the lane's guarded accumulation and
///   the `u64` accumulation here agree, and [`decide`] agrees with the
///   lane's overflow-free verdict rule term for term;
/// * `accepted` counts every visited accept whatever `kept` is, so any large
///   `work_max` (no exhaustion) with any `kept` agrees with this
///   exhaustive scan.
///
/// An ambiguous verdict explains nothing here, exactly as `accepted` counts
/// only accepts there.
///
/// Differences outside the equivalence scope, all documented: an unknown
/// adduct id is [`Error::Unsupported`] in `ion_assign` but builds no index
/// here ([`build_evidence_index`] returns `Ok(None)`, every peak unexplained,
/// matching the lane's `ION_UNAVAILABLE` row); a target mass outside `u32`
/// is an error there but an unexplained peak here; `ppm_tenths > 1000` is an
/// error in both.
pub struct EvidenceIndex {
    /// Signed adduct charge (`+1` or `-1`).
    charge: i32,
    /// Largest ion hydrogen count: `parent[H] + (1 for `[M+H]+`, else 0) + 2`.
    h_cap: u32,
    /// Upper bound of every hypothesis error: `ceil(parent nda / 1000)`.
    e_ion: u32,
    /// Spectrum m/z uncertainty `U`.
    uncertainty: u32,
    /// Fragment tolerance in tenths of a ppm.
    ppm_tenths: u32,
    /// `(heavy mass, heavy residual sum)` per non-empty heavy sub-vector,
    /// sorted by mass.
    heavy: Vec<(u32, u32)>,
    /// Hydrogen integer mass.
    m_h: u32,
    /// Hydrogen rounding residual.
    h_res: u32,
}

impl EvidenceIndex {
    /// Whether the peak has at least one accepted sub-composition hypothesis.
    pub fn explains(&self, peak_mz: u32) -> Result<bool> {
        if peak_mz == 0 {
            return Ok(false);
        }
        let t = if self.charge > 0 {
            peak_mz.checked_add(ELECTRON_MASS)
        } else {
            peak_mz.checked_sub(ELECTRON_MASS)
        };
        let Some(t) = t else {
            return Ok(false);
        };
        let tol = tolerance(peak_mz, self.ppm_tenths);
        let half = tol
            .saturating_add(self.uncertainty)
            .saturating_add(self.e_ion);
        if half > self.m_h {
            return Ok(false);
        }
        let lo = t.saturating_sub(half);
        let hi = t.saturating_add(half);
        let end = self.heavy.partition_point(|&(m, _)| m <= hi);
        for &(m, r) in &self.heavy[..end] {
            // Closed-form hydrogen interval of the lane: masses below the
            // window need `ceil((lo - m) / m_H)` hydrogens to reach it.
            let h_lo = if m < lo {
                let gap = u64::from(lo - m);
                let unit = u64::from(self.m_h);
                (gap.div_ceil(unit)) as u32
            } else {
                0
            };
            let span = hi - m;
            let h_hi = (span / self.m_h).min(self.h_cap);
            if h_lo > h_hi {
                continue;
            }
            for h in h_lo..=h_hi {
                // `m + h * m_H <= hi <= u32::MAX` by the `h_hi` bound, so the
                // `u64` product narrows exactly.
                let cand = m + (h * self.m_h);
                let res = u64::from(r)
                    + u64::from(h) * u64::from(self.h_res)
                    + u64::from(ELECTRON_RESIDUAL_NDA);
                let arith = res.div_ceil(1000).min(u64::from(u32::MAX)) as u32;
                let bound = arith.saturating_add(self.uncertainty);
                if decide(t, cand, bound, tol) == super::chem::Verdict::Accept {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }
}

/// Build the fast peak-explanation test of one candidate parent composition.
///
/// Returns `Ok(None)` when no peak can be searched under the lane semantics:
/// unknown adduct (the lane's `ION_UNAVAILABLE` row; `ion_assign` reports
/// this as an error instead), an empty parent, or `U == u32::MAX`. Errors on
/// `ppm_tenths > 1000`, exactly as `ion_assign` does.
pub fn build_evidence_index(
    parent: &Composition,
    adduct_id: u16,
    mz_uncertainty: u32,
    ppm_tenths: u32,
) -> Result<Option<EvidenceIndex>> {
    if ppm_tenths > 1000 {
        return Err(Error::config(format!(
            "build_evidence_index: ppm_tenths {ppm_tenths} exceeds the 1000 proof bound"
        )));
    }
    let Some(a) = super::chem::adduct(adduct_id) else {
        return Ok(None);
    };
    if a.charge != 1 && a.charge != -1 {
        return Ok(None);
    }
    if parent.iter().all(|&n| n == 0) {
        return Ok(None);
    }
    if mz_uncertainty == u32::MAX {
        return Ok(None);
    }
    // Mixed-radix odometer over the 9 heavy elements in ELEMENTS order with
    // the parent counts as caps; the all-zero vector never joins (the lane
    // visits indices from 1 upward). Masses and residuals accumulate in
    // `u64`: every sub-vector mass is below the parent mass, which fits
    // `u32` for every scored candidate and every real gold composition.
    let caps: [u64; 9] = HEAVY_ELEMENTS.map(|e| u64::from(parent[e]));
    let masses: [u64; 9] = HEAVY_ELEMENTS.map(|e| u64::from(ELEMENTS[e].mass));
    let resids: [u64; 9] = HEAVY_ELEMENTS.map(|e| u64::from(ELEMENTS[e].residual_nda));
    let total: u128 = caps.iter().map(|&c| u128::from(c + 1)).product();
    if total > u128::from(u32::MAX) {
        return Err(Error::config(format!(
            "build_evidence_index: {total} heavy sub-vectors exceed the u32 visit range"
        )));
    }
    let mut heavy: Vec<(u32, u32)> = Vec::new();
    let mut digits: [u64; 9] = [0; 9];
    for _ in 1..total {
        // Increment the odometer (position 0 least significant).
        let mut pos = 0;
        loop {
            digits[pos] += 1;
            if digits[pos] <= caps[pos] {
                break;
            }
            digits[pos] = 0;
            pos += 1;
        }
        let mut mass: u64 = 0;
        let mut res: u64 = 0;
        for i in 0..9 {
            mass += digits[i] * masses[i];
            res += digits[i] * resids[i];
        }
        let (Ok(m), Ok(r)) = (u32::try_from(mass), u32::try_from(res)) else {
            return Err(Error::config(
                "build_evidence_index: heavy sub-vector mass exceeds u32".to_string(),
            ));
        };
        heavy.push((m, r));
    }
    heavy.sort();
    // `E_ion` upper bound of the lane: parent residuals plus at most
    // `parent[H] + 3` hydrogens (`max(h_a, 0) <= 1` for the V0 adducts, plus
    // 2) plus the electron residual. Checked: real counts keep this far
    // inside `u64`.
    let mut nda: u64 = 0;
    for (e, n) in parent.iter().enumerate() {
        nda = nda
            .checked_add(u64::from(*n) * u64::from(ELEMENTS[e].residual_nda))
            .ok_or_else(|| Error::config("build_evidence_index: parent residual sum overflows u64"))?;
    }
    nda = nda
        .checked_add(3 * u64::from(ELEMENTS[HYDROGEN].residual_nda) + u64::from(ELECTRON_RESIDUAL_NDA))
        .ok_or_else(|| Error::config("build_evidence_index: ion residual sum overflows u64"))?;
    let e_ion = nda.div_ceil(1000).min(u64::from(u32::MAX)) as u32;
    let h_cap = u64::from(parent[HYDROGEN])
        + if a.charge > 0 { 1 } else { 0 }
        + 2;
    let h_cap = h_cap.min(u64::from(u32::MAX)) as u32;
    Ok(Some(EvidenceIndex {
        charge: a.charge,
        h_cap,
        e_ion,
        uncertainty: mz_uncertainty,
        ppm_tenths,
        heavy,
        m_h: ELEMENTS[HYDROGEN].mass,
        h_res: ELEMENTS[HYDROGEN].residual_nda,
    }))
}

/// Evidence features of one candidate composition on one spectrum.
#[derive(Clone, Debug, PartialEq)]
pub struct CandidateEvidence {
    /// Element counts in [`ELEMENTS`] order.
    pub counts: Composition,
    /// Integer mass (0 when the mass overflows `u32`).
    pub mass: u32,
    /// Kept peaks with at least one accepted sub-composition.
    pub expl_count: u32,
    /// `expl_count / kept peaks` (0 without kept peaks).
    pub expl_count_frac: f64,
    /// Fraction of kept intensity on explained peaks (0 without intensity).
    pub expl_intensity: f64,
    /// `|candidate mass − precursor neutral mass|` in ppm (infinity when the
    /// precursor neutral mass is missing or zero).
    pub residual_ppm: f64,
    /// Heavy-atom total.
    pub heavy_atoms: u32,
    /// Twice the double-bond equivalent under maximum valences.
    pub dbe_twice: i64,
    /// `|r| / 1 ppm` (infinity exactly when `residual_ppm` is infinite).
    pub resid_o1: f64,
    /// `(|r| / 1 ppm)^2` (infinity exactly when `residual_ppm` is infinite).
    pub resid_o1_sq: f64,
    /// `ln(1 + |r| / 0.1 ppm)` (infinity exactly when infinite).
    pub resid_ln1p: f64,
    /// `|r| / tolerance` with `tolerance = 20 ppm` (infinity when infinite).
    pub resid_otol: f64,
    /// `expl_intensity` minus the spectrum maximum over its candidates (0
    /// for candidates at the maximum; 0 without candidates).
    pub expl_minus_max: f64,
    /// Fraction of the spectrum's candidates with `expl_intensity` at least
    /// this candidate's (in `[0, 1]`; 1 for a single candidate; 0 without
    /// candidates, unreachable in practice).
    pub expl_rank_frac: f64,
}

/// The four residual transform features of `|r|` in ppm.
///
/// `tol_ppm` is the precursor window tolerance in ppm (20 for this
/// experiment). Infinite residuals map to infinities, never NaN.
pub fn residual_features(residual_ppm: f64, tol_ppm: f64) -> [f64; 4] {
    let a = residual_ppm.abs() / 1.0;
    let b = a * a;
    let c = (1.0 + residual_ppm.abs() / 0.1).ln();
    let d = residual_ppm.abs() / tol_ppm;
    [a, b, c, d]
}

/// Sharper evidence features of one spectrum from its candidates'
/// `expl_intensity` values: `(minus_max, rank_frac)` per candidate.
///
/// `minus_max[i] = expl[i] - max(expl)` (0 at the maximum); `rank_frac[i]`
/// is the fraction of candidates with `expl[j] >= expl[i]` (ties count, so
/// tied maxima share `tied / n`; a single candidate yields `(0, 1)`).
/// Empty input yields empty outputs.
pub fn sharp_evidence_features(expl: &[f64]) -> (Vec<f64>, Vec<f64>) {
    if expl.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let max = expl.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let n = expl.len() as f64;
    let mut minus = Vec::with_capacity(expl.len());
    let mut rank = Vec::with_capacity(expl.len());
    for &v in expl {
        minus.push(v - max);
        let ge = expl.iter().filter(|&&u| u >= v).count();
        rank.push(ge as f64 / n);
    }
    (minus, rank)
}

/// Fill the sharper evidence features of `ev` from its candidates'
/// `expl_intensity` values (second pass of [`spectrum_evidence`]).
pub fn apply_sharp_features(ev: &mut SpectrumEvidence) {
    let expl: Vec<f64> = ev.candidates.iter().map(|c| c.expl_intensity).collect();
    let (minus, rank) = sharp_evidence_features(&expl);
    for (c, (m, r)) in ev.candidates.iter_mut().zip(minus.into_iter().zip(rank)) {
        c.expl_minus_max = m;
        c.expl_rank_frac = r;
    }
}

/// Seeded jitter precursor m/z (FE2).
///
/// `sigma_ppm <= 0` returns the input unchanged. Otherwise `e ~ Normal(0,
/// sigma)` via Box–Muller on a [`StdRng`] seeded by
/// [`jitter_seed`] from `(seed, split_tag, spectrum_index, sigma)`, resampled
/// until `|e| <= 3 sigma` (truncation), and the output is
/// `round(mz * (1 + e * 1e-6))` clamped to `[1, u32::MAX]` (`0` maps to `0`).
/// Pure in its inputs, hence independent of thread scheduling.
pub fn jitter_precursor_mz(
    precursor_mz: u32,
    sigma_ppm: f64,
    seed: u64,
    split_tag: u64,
    spectrum_index: u64,
) -> u32 {
    if sigma_ppm <= 0.0 {
        return precursor_mz;
    }
    debug_assert!(sigma_ppm.is_finite());
    let rng_seed = jitter_seed(seed, split_tag, spectrum_index, sigma_ppm.to_bits());
    let mut rng = StdRng::seed_from_u64(rng_seed);
    let bound = 3.0 * sigma_ppm;
    loop {
        let mut u1: f64 = rng.random_range(0.0..1.0);
        while u1 <= 0.0 || !(u1 < 1.0) {
            u1 = rng.random_range(0.0..1.0);
        }
        let u2: f64 = rng.random_range(0.0..1.0);
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        let e = z * sigma_ppm;
        if e.abs() <= bound {
            if precursor_mz == 0 {
                return 0;
            }
            let jittered = f64::from(precursor_mz) * (1.0 + e * 1e-6);
            let rounded = jittered.round();
            if rounded < 1.0 {
                return 1;
            }
            if rounded > f64::from(u32::MAX) {
                return u32::MAX;
            }
            return rounded as u32;
        }
    }
}

/// Mix `(seed, split_tag, spectrum_index, sigma_bits)` into one RNG seed.
///
/// Split tags are caller-chosen (`0` train, `1` validation, `2` shuffled
/// train, `3` shuffled validation in the report example); the wrapping
/// multiplies are fixed constants, so equal inputs give equal seeds.
pub fn jitter_seed(seed: u64, split_tag: u64, spectrum_index: u64, sigma_bits: u64) -> u64 {
    const A: u64 = 0x9E3779B97F4A7C15;
    const B: u64 = 0xBF58476D1CE4E5B9;
    const C: u64 = 0x94D049BB133111EB;
    seed.wrapping_mul(A)
        .wrapping_add(split_tag.wrapping_mul(B))
        .wrapping_add(spectrum_index.wrapping_mul(C))
        .wrapping_add(sigma_bits.wrapping_mul(0x2545F4914F6CDD1D))
}

/// The full feature vector of a candidate in [`FEATURE_NAMES`] order (21).
pub fn feature_vector(c: &CandidateEvidence) -> [f64; N_FEATURES] {
    let table = ln1p_table();
    let mut out = [0.0f64; N_FEATURES];
    for e in 0..10 {
        out[e] = table[c.counts[e] as usize];
    }
    out[10] = f64::from(c.heavy_atoms);
    out[11] = c.dbe_twice as f64 / 2.0;
    out[12] = c.expl_count_frac;
    out[13] = c.expl_intensity;
    out[14] = c.residual_ppm;
    out[15] = c.resid_o1;
    out[16] = c.resid_o1_sq;
    out[17] = c.resid_ln1p;
    out[18] = c.resid_otol;
    out[19] = c.expl_minus_max;
    out[20] = c.expl_rank_frac;
    out
}

/// Raw FE3 nonlinear feature vector of a candidate (90): the 12 prior
/// features, then all pairwise products `x_i * x_j` with `i <= j` over those
/// 12 in order. Products are formed before standardisation; the caller
/// standardises the 90-vector with a [`Standardizer`] fitted on train.
/// Non-finite raw values (possible only through infinite residuals, which do
/// not enter the prior features) propagate as non-finite here and are mapped
/// to zero by [`Standardizer::transform`].
pub fn nonlinear_raw_vector(c: &CandidateEvidence) -> Vec<f64> {
    let base = feature_vector(c);
    let mut prior = [0.0f64; 12];
    for (k, &d) in PRIOR_DIMS.iter().enumerate() {
        prior[k] = base[d];
    }
    let mut out = Vec::with_capacity(N_NONLINEAR_FE3);
    out.extend_from_slice(&prior);
    for i in 0..N_PRIOR_FE3 {
        for j in i..N_PRIOR_FE3 {
            out.push(prior[i] * prior[j]);
        }
    }
    out
}

/// Raw FE3 nonlinear-plus feature vector of a candidate (98): the 90
/// nonlinear features, then the 4 residual transforms (dims 15–18), then the
/// 4 evidence features (`EVIDENCE4_DIMS` order).
pub fn nonlinear_plus_raw_vector(c: &CandidateEvidence) -> Vec<f64> {
    let base = feature_vector(c);
    let mut out = nonlinear_raw_vector(c);
    for &d in RESIDUAL_DIMS.iter() {
        out.push(base[d]);
    }
    for &d in EVIDENCE4_DIMS.iter() {
        out.push(base[d]);
    }
    out
}

/// Which raw extractor a standardised nonlinear ranker uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NonlinearKind {
    /// 90-dim prior plus pairwise products.
    PriorOnly,
    /// 98-dim prior-nonlinear plus residual plus evidence.
    PlusResEv,
}

impl NonlinearKind {
    /// Raw (unstandardised) feature vector of a candidate.
    pub fn raw_vector(&self, c: &CandidateEvidence) -> Vec<f64> {
        match self {
            NonlinearKind::PriorOnly => nonlinear_raw_vector(c),
            NonlinearKind::PlusResEv => nonlinear_plus_raw_vector(c),
        }
    }

    /// Feature count.
    pub fn len(&self) -> usize {
        match self {
            NonlinearKind::PriorOnly => N_NONLINEAR_FE3,
            NonlinearKind::PlusResEv => N_NONLINEAR_PLUS_FE3,
        }
    }

    /// Feature names in order.
    pub fn names(&self) -> Vec<String> {
        match self {
            NonlinearKind::PriorOnly => nonlinear_feature_names(),
            NonlinearKind::PlusResEv => nonlinear_plus_feature_names(),
        }
    }
}

/// Inputs of [`spectrum_evidence`] for one spectrum.
pub struct SpectrumInput<'a> {
    /// Window candidates in device order with their integer masses.
    pub candidates: &'a [Composition],
    /// Integer mass per candidate (same length as `candidates`).
    pub masses: &'a [u32],
    /// Kept peaks of the spectrum (real or shuffled donor peaks).
    pub peaks: &'a [KeptPeak],
    /// Adduct id of the spectrum (kept under shuffling).
    pub adduct_id: u16,
    /// m/z uncertainty of the (possibly donor) peaks.
    pub mz_uncertainty: u32,
    /// Fragment tolerance in tenths of a ppm.
    pub ion_ppm_tenths: u32,
    /// Neutral precursor mass (kept under shuffling; `None` searches nothing).
    pub parent_mass: Option<u32>,
}

/// Evidence of one spectrum: per-candidate features plus provenance.
#[derive(Clone, Debug)]
pub struct SpectrumEvidence {
    /// Per-candidate features in window order.
    pub candidates: Vec<CandidateEvidence>,
    /// Index of the gold formula in `candidates`, if it is in the window.
    pub gold: Option<usize>,
    /// Whether the window holds any candidate.
    pub has_window: bool,
}

/// Evidence features of every window candidate of one spectrum.
///
/// `gold` is the spectrum's own composition (kept under shuffling); when it
/// is a window member its index is recorded, otherwise the spectrum still
/// counts as a miss in every full-dataset recall denominator.
pub fn spectrum_evidence(
    input: &SpectrumInput,
    gold: Option<&Composition>,
) -> Result<SpectrumEvidence> {
    if input.candidates.len() != input.masses.len() {
        return Err(Error::config(format!(
            "spectrum_evidence: {} candidates but {} masses",
            input.candidates.len(),
            input.masses.len()
        )));
    }
    let kept = input.peaks.len() as f64;
    let total_intensity: f64 = input.peaks.iter().map(|p| p.intensity).sum();
    let mut candidates = Vec::with_capacity(input.candidates.len());
    for (c, mass) in input.candidates.iter().zip(input.masses.iter()) {
        let (expl_count, expl_intensity) = if input.parent_mass.is_none() {
            (0, 0.0)
        } else if let Some(index) = build_evidence_index(
            c,
            input.adduct_id,
            input.mz_uncertainty,
            input.ion_ppm_tenths,
        )? {
            let mut count: u32 = 0;
            let mut explained_intensity = 0.0;
            for peak in input.peaks {
                if index.explains(peak.mz)? {
                    count = count.saturating_add(1);
                    explained_intensity += peak.intensity;
                }
            }
            let share = if total_intensity > 0.0 {
                explained_intensity / total_intensity
            } else {
                0.0
            };
            (count, share)
        } else {
            (0, 0.0)
        };
        let residual_ppm = match input.parent_mass {
            Some(parent) if parent > 0 => {
                (mass.abs_diff(parent) as f64) * 1e6 / f64::from(parent)
            }
            _ => f64::INFINITY,
        };
        let mut heavy_atoms: u32 = 0;
        for e in HEAVY_ELEMENTS {
            heavy_atoms = heavy_atoms.saturating_add(u32::from(c[e]));
        }
        let dbe_twice =
            super::formula_enum::dbe_twice(c).unwrap_or(0);
        let [resid_o1, resid_o1_sq, resid_ln1p, resid_otol] =
            residual_features(residual_ppm, precursor_tol_ppm());
        candidates.push(CandidateEvidence {
            counts: *c,
            mass: *mass,
            expl_count,
            expl_count_frac: if kept > 0.0 {
                f64::from(expl_count) / kept
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
            // Second pass below: spectrum-relative sharper features.
            expl_minus_max: 0.0,
            expl_rank_frac: if input.candidates.is_empty() {
                0.0
            } else {
                1.0
            },
        });
    }
    let gold = gold.and_then(|g| input.candidates.iter().position(|c| c == g));
    let mut out = SpectrumEvidence {
        candidates,
        gold,
        has_window: !input.candidates.is_empty(),
    };
    apply_sharp_features(&mut out);
    Ok(out)
}

/// A candidate ranking rule.
pub enum RankRule<'a> {
    /// By precursor residual in ppm, ascending; ties by mass, counts, index.
    ResidualOnly,
    /// By explained intensity, descending; ties by residual, then index.
    ExplainedIntensity,
    /// By explained count, descending; ties by residual, then index.
    ExplainedCount,
    /// By a linear score over `dims` with `weights`, descending; ties by
    /// residual, then index.
    Linear(&'a [f64], &'a [usize]),
    /// By a linear score over standardised features: `sum_j weights[j] *
    /// (raw[dims[j]] - means[j]) / scales[j]`, descending (FE3).
    LinearStd(&'a [f64], &'a [usize], &'a Standardizer),
    /// By a linear score over a standardised FE3 nonlinear raw vector,
    /// descending (FE3).
    NonlinearStd(&'a [f64], &'a Standardizer, NonlinearKind),
}

impl RankRule<'_> {
    /// 1-based rank of the gold formula, or `None` when it is not in the window.
    pub fn gold_rank(&self, ev: &SpectrumEvidence) -> Option<u32> {
        let gold = ev.gold?;
        rank_order(ev, self)
            .iter()
            .position(|&i| i == gold)
            .map(|pos| pos as u32 + 1)
    }
}

/// Window indices in best-first order under the rule (total, deterministic).
pub fn rank_order(ev: &SpectrumEvidence, rule: &RankRule) -> Vec<usize> {
    let mut order: Vec<usize> = (0..ev.candidates.len()).collect();
    order.sort_by(|&a, &b| {
        let (ca, cb) = (&ev.candidates[a], &ev.candidates[b]);
        match rule {
            RankRule::ResidualOnly => ca
                .residual_ppm
                .total_cmp(&cb.residual_ppm)
                .then(ca.mass.cmp(&cb.mass))
                .then(ca.counts.cmp(&cb.counts))
                .then(a.cmp(&b)),
            RankRule::ExplainedIntensity => cb
                .expl_intensity
                .total_cmp(&ca.expl_intensity)
                .then(ca.residual_ppm.total_cmp(&cb.residual_ppm))
                .then(a.cmp(&b)),
            RankRule::ExplainedCount => cb
                .expl_count
                .cmp(&ca.expl_count)
                .then(ca.residual_ppm.total_cmp(&cb.residual_ppm))
                .then(a.cmp(&b)),
            RankRule::Linear(weights, dims) => {
                let fa = feature_vector(ca);
                let fb = feature_vector(cb);
                let sa: f64 = dims.iter().zip(weights.iter()).map(|(&d, &w)| w * fa[d]).sum();
                let sb: f64 = dims.iter().zip(weights.iter()).map(|(&d, &w)| w * fb[d]).sum();
                sb.total_cmp(&sa)
                    .then(ca.residual_ppm.total_cmp(&cb.residual_ppm))
                    .then(a.cmp(&b))
            }
            RankRule::LinearStd(weights, dims, scaler) => {
                let fa = feature_vector(ca);
                let fb = feature_vector(cb);
                let sa: f64 = dims
                    .iter()
                    .zip(weights.iter())
                    .enumerate()
                    .map(|(j, (&d, &w))| w * scaler.transform_value(fa[d], j))
                    .sum();
                let sb: f64 = dims
                    .iter()
                    .zip(weights.iter())
                    .enumerate()
                    .map(|(j, (&d, &w))| w * scaler.transform_value(fb[d], j))
                    .sum();
                sb.total_cmp(&sa)
                    .then(ca.residual_ppm.total_cmp(&cb.residual_ppm))
                    .then(a.cmp(&b))
            }
            RankRule::NonlinearStd(weights, scaler, kind) => {
                let fa = scaler.transform(&kind.raw_vector(ca));
                let fb = scaler.transform(&kind.raw_vector(cb));
                let sa: f64 = weights.iter().zip(fa.iter()).map(|(&w, &x)| w * x).sum();
                let sb: f64 = weights.iter().zip(fb.iter()).map(|(&w, &x)| w * x).sum();
                sb.total_cmp(&sa)
                    .then(ca.residual_ppm.total_cmp(&cb.residual_ppm))
                    .then(a.cmp(&b))
            }
        }
    });
    order
}

/// Recall@k over spectra: `None` ranks (gold outside the window) are misses.
pub fn recall_at(ranks: &[Option<u32>], k: u32) -> f64 {
    if ranks.is_empty() {
        return 0.0;
    }
    let hits = ranks
        .iter()
        .filter(|r| r.is_some_and(|v| v <= k))
        .count();
    hits as f64 / ranks.len() as f64
}

/// Recall@k conditional on the gold formula being in the window.
pub fn recall_at_conditional(ranks: &[Option<u32>], k: u32) -> f64 {
    let in_window: Vec<&Option<u32>> = ranks.iter().filter(|r| r.is_some()).collect();
    if in_window.is_empty() {
        return 0.0;
    }
    let hits = in_window
        .iter()
        .filter(|r| r.is_some_and(|v| v <= k))
        .count();
    hits as f64 / in_window.len() as f64
}

/// Recall@k with a 95% bootstrap interval over spectra.
///
/// Returns `(recall, lower, upper)` from `resamples` resamples with
/// replacement, seeded. Empty input yields zeros.
pub fn bootstrap_recall(
    ranks: &[Option<u32>],
    k: u32,
    resamples: usize,
    seed: u64,
) -> (f64, f64, f64) {
    let value = recall_at(ranks, k);
    if ranks.is_empty() || resamples == 0 {
        return (value, value, value);
    }
    let mut rng = StdRng::seed_from_u64(seed);
    let mut draws = Vec::with_capacity(resamples);
    for _ in 0..resamples {
        let mut hits = 0usize;
        for _ in 0..ranks.len() {
            let i = rng.random_range(0..ranks.len());
            if ranks[i].is_some_and(|v| v <= k) {
                hits += 1;
            }
        }
        draws.push(hits as f64 / ranks.len() as f64);
    }
    draws.sort_by(|a, b| a.total_cmp(b));
    (value, percentile(&draws, 2.5), percentile(&draws, 97.5))
}

/// Seeded permutation of `n` positions without fixed points.
///
/// Fisher–Yates with [`StdRng`] on the seed, then one deterministic repair
/// pass: a single fixed point swaps with its successor (both become
/// non-fixed and nothing else moves), while two or more rotate their values
/// (every rotated position receives another fixed position's index, hence a
/// different value). Errors on `n < 2`, where no derangement exists.
pub fn derangement(n: usize, seed: u64) -> Result<Vec<usize>> {
    if n < 2 {
        return Err(Error::config(format!(
            "derangement: no fixed-point-free permutation of {n} positions exists"
        )));
    }
    let mut rng = StdRng::seed_from_u64(seed);
    let mut perm: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = rng.random_range(0..=i);
        perm.swap(i, j);
    }
    let fixed: Vec<usize> = perm
        .iter()
        .enumerate()
        .filter(|&(i, &v)| i == v)
        .map(|(i, _)| i)
        .collect();
    if fixed.len() == 1 {
        perm.swap(fixed[0], (fixed[0] + 1) % n);
    } else if fixed.len() >= 2 {
        let values: Vec<usize> = fixed.iter().map(|&i| perm[i]).collect();
        for (j, &i) in fixed.iter().enumerate() {
            perm[i] = values[(j + 1) % values.len()];
        }
    }
    debug_assert!(perm.iter().enumerate().all(|(i, &v)| i != v));
    Ok(perm)
}

/// Mini-batch Adam configuration of [`train_softmax`].
#[derive(Clone, Copy, Debug)]
pub struct TrainConfig {
    /// Passes over the train spectra.
    pub epochs: usize,
    /// Spectra per gradient step.
    pub batch_spectra: usize,
    /// Adam step size.
    pub lr: f64,
    /// L2 penalty (`0.5 * l2 * ||w||²` on the loss).
    pub l2: f64,
    /// Seed of the per-epoch spectrum order.
    pub seed: u64,
}

/// Mean softmax NLL over gold-in-window spectra plus the L2 term, with its
/// gradient in `weights` (full batch over `data`).
///
/// The loss of one spectrum is `-log p(gold)` under the softmax over its
/// candidates; spectra without gold in the window contribute nothing.
/// Checked: empty `dims` is an error; a weights/dims length mismatch is an
/// error.
pub fn softmax_loss_grad(
    data: &[SpectrumEvidence],
    dims: &[usize],
    weights: &[f64],
    l2: f64,
) -> Result<(f64, Vec<f64>)> {
    if dims.is_empty() {
        return Err(Error::config(
            "softmax_loss_grad: no feature dimensions selected".to_string(),
        ));
    }
    if weights.len() != dims.len() {
        return Err(Error::config(format!(
            "softmax_loss_grad: {} weights but {} dimensions",
            weights.len(),
            dims.len()
        )));
    }
    if dims.iter().any(|&d| d >= N_FEATURES) {
        return Err(Error::config(format!(
            "softmax_loss_grad: dimension >= {N_FEATURES}"
        )));
    }
    let mut loss = 0.0f64;
    let mut grad = vec![0.0f64; dims.len()];
    let mut spectra = 0usize;
    for ev in data {
        let Some(gold) = ev.gold else { continue };
        spectra += 1;
        let feats: Vec<[f64; N_FEATURES]> = ev.candidates.iter().map(feature_vector).collect();
        let scores: Vec<f64> = feats
            .iter()
            .map(|f| {
                dims.iter()
                    .zip(weights.iter())
                    .map(|(&d, &w)| w * f[d])
                    .sum()
            })
            .collect();
        let max = scores.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let mut denom = 0.0f64;
        for s in &scores {
            denom += (s - max).exp();
        }
        let log_denom = max + denom.ln();
        loss += log_denom - scores[gold];
        for (i, f) in feats.iter().enumerate() {
            let p = (scores[i] - max).exp() / denom;
            let target = if i == gold { 1.0 } else { 0.0 };
            for (j, &d) in dims.iter().enumerate() {
                grad[j] += (p - target) * f[d];
            }
        }
    }
    if spectra > 0 {
        let n = spectra as f64;
        loss /= n;
        for g in grad.iter_mut() {
            *g /= n;
        }
    }
    for (w, g) in weights.iter().zip(grad.iter_mut()) {
        loss += 0.5 * l2 * w * w;
        *g += l2 * w;
    }
    Ok((loss, grad))
}

/// Multinomial logistic ranking model: softmax over a spectrum's candidates,
/// linear in the selected feature dimensions, trained by mini-batch Adam.
///
/// Weights start at zero; each epoch shuffles the gold-in-window train
/// spectra with the seeded RNG and steps over spectrum batches. Returns the
/// weight vector over `dims`. Deterministic for fixed data, config and seed
/// (single-threaded).
pub fn train_softmax(
    data: &[SpectrumEvidence],
    dims: &[usize],
    cfg: &TrainConfig,
) -> Result<Vec<f64>> {
    if dims.is_empty() {
        return Err(Error::config(
            "train_softmax: no feature dimensions selected".to_string(),
        ));
    }
    if cfg.batch_spectra == 0 {
        return Err(Error::config(
            "train_softmax: batch_spectra is 0".to_string(),
        ));
    }
    if cfg.lr <= 0.0 || !cfg.lr.is_finite() {
        return Err(Error::config(format!(
            "train_softmax: learning rate {} is not a positive finite value",
            cfg.lr
        )));
    }
    let mut eligible: Vec<usize> = data
        .iter()
        .enumerate()
        .filter(|(_, ev)| ev.gold.is_some())
        .map(|(i, _)| i)
        .collect();
    let mut weights = vec![0.0f64; dims.len()];
    if eligible.is_empty() {
        return Ok(weights);
    }
    // Adam state over the weight vector.
    let beta1 = 0.9f64;
    let beta2 = 0.999f64;
    let eps = 1e-8f64;
    let mut m = vec![0.0f64; dims.len()];
    let mut v = vec![0.0f64; dims.len()];
    let mut step = 0u64;
    let mut rng = StdRng::seed_from_u64(cfg.seed);
    for _ in 0..cfg.epochs {
        // Seeded Fisher–Yates over the eligible spectra.
        for i in (1..eligible.len()).rev() {
            let j = rng.random_range(0..=i);
            eligible.swap(i, j);
        }
        for batch in eligible.chunks(cfg.batch_spectra) {
            let subset: Vec<SpectrumEvidence> =
                batch.iter().map(|&i| data[i].clone()).collect();
            let (_, grad) = softmax_loss_grad(&subset, dims, &weights, cfg.l2)?;
            step += 1;
            let t = step as f64;
            for j in 0..dims.len() {
                m[j] = beta1 * m[j] + (1.0 - beta1) * grad[j];
                v[j] = beta2 * v[j] + (1.0 - beta2) * grad[j] * grad[j];
                let m_hat = m[j] / (1.0 - beta1.powf(t));
                let v_hat = v[j] / (1.0 - beta2.powf(t));
                weights[j] -= cfg.lr * m_hat / (v_hat.sqrt() + eps);
            }
        }
    }
    Ok(weights)
}

/// Affine feature standardisation of one ranker (FE3).
///
/// `means[j]` / `scales[j]` describe raw column `j` (for linear rankers,
/// raw column `j` is `feature_vector[dims[j]]`; for nonlinear rankers it is
/// entry `j` of the [`NonlinearKind`] raw vector). Standardised values are
/// `(raw - mean) / scale`. A zero-variance column is left centred with scale
/// 1, i.e. standardised values are `raw - mean` (all zero when every train
/// value is finite and equal). Non-finite raw values (infinite residuals)
/// standardise to `0.0` (the train mean), so missing-precursor candidates are
/// neutral rather than poisonous. The map is fitted on the TRAIN candidates
/// of the run (per sigma, per shuffle variant) and the same affine map is
/// applied at evaluation; reported weights are in these standardised units.
#[derive(Clone, Debug, PartialEq)]
pub struct Standardizer {
    /// Per-column train means.
    pub means: Vec<f64>,
    /// Per-column train scales (std, or 1.0 for zero-variance columns).
    pub scales: Vec<f64>,
}

impl Standardizer {
    /// Fit on raw rows (one row per train candidate).
    pub fn fit(rows: &[Vec<f64>], dim: usize) -> Self {
        let mut means = vec![0.0f64; dim];
        let mut scales = vec![1.0f64; dim];
        if rows.is_empty() || dim == 0 {
            return Self { means, scales };
        }
        let mut counts = vec![0usize; dim];
        for row in rows {
            for j in 0..dim.min(row.len()) {
                let v = row[j];
                if v.is_finite() {
                    means[j] += v;
                    counts[j] += 1;
                }
            }
        }
        for j in 0..dim {
            if counts[j] > 0 {
                means[j] /= counts[j] as f64;
            } else {
                means[j] = 0.0;
            }
        }
        let mut sum_sq = vec![0.0f64; dim];
        for row in rows {
            for j in 0..dim.min(row.len()) {
                let v = row[j];
                if v.is_finite() {
                    let d = v - means[j];
                    sum_sq[j] += d * d;
                }
            }
        }
        for j in 0..dim {
            if counts[j] > 0 {
                let var = sum_sq[j] / counts[j] as f64;
                let std = if var > 0.0 { var.sqrt() } else { 0.0 };
                scales[j] = if std > 0.0 && std.is_finite() { std } else { 1.0 };
            } else {
                scales[j] = 1.0;
            }
        }
        Self { means, scales }
    }

    /// Standardise one raw value of column `j`.
    pub fn transform_value(&self, raw: f64, j: usize) -> f64 {
        if !raw.is_finite() {
            return 0.0;
        }
        (raw - self.means[j]) / self.scales[j]
    }

    /// Standardise one raw row.
    pub fn transform(&self, raw: &[f64]) -> Vec<f64> {
        let n = self.means.len().min(raw.len());
        let mut out = Vec::with_capacity(self.means.len());
        for j in 0..n {
            out.push(self.transform_value(raw[j], j));
        }
        for j in n..self.means.len() {
            out.push((-self.means[j]) / self.scales[j]);
        }
        out
    }

    /// Invert standardisation of one value of column `j`.
    pub fn invert_value(&self, std: f64, j: usize) -> f64 {
        std * self.scales[j] + self.means[j]
    }
}

/// Fit a [`Standardizer`] for linear `dims` over all TRAIN candidates of
/// `data` (every candidate of every spectrum, gold or not).
pub fn fit_standardizer(data: &[SpectrumEvidence], dims: &[usize]) -> Standardizer {
    let mut rows: Vec<Vec<f64>> = Vec::new();
    for ev in data {
        for c in &ev.candidates {
            let f = feature_vector(c);
            rows.push(dims.iter().map(|&d| f[d]).collect());
        }
    }
    Standardizer::fit(&rows, dims.len())
}

/// Fit a [`Standardizer`] for an FE3 nonlinear extractor over all TRAIN
/// candidates of `data`.
pub fn fit_standardizer_nonlinear(data: &[SpectrumEvidence], kind: NonlinearKind) -> Standardizer {
    let mut rows: Vec<Vec<f64>> = Vec::new();
    for ev in data {
        for c in &ev.candidates {
            rows.push(kind.raw_vector(c));
        }
    }
    Standardizer::fit(&rows, kind.len())
}

/// One spectrum's standardised feature matrix for the converged optimiser.
#[derive(Clone, Debug)]
struct SpecMat {
    feats: Vec<Vec<f64>>,
    gold: Option<usize>,
}

fn build_linear_matrix(
    data: &[SpectrumEvidence],
    dims: &[usize],
    scaler: &Standardizer,
) -> Vec<SpecMat> {
    data.iter()
        .map(|ev| {
            let feats = ev
                .candidates
                .iter()
                .map(|c| {
                    let f = feature_vector(c);
                    (0..dims.len())
                        .map(|j| scaler.transform_value(f[dims[j]], j))
                        .collect()
                })
                .collect();
            SpecMat { feats, gold: ev.gold }
        })
        .collect()
}

fn build_nonlinear_matrix(
    data: &[SpectrumEvidence],
    kind: NonlinearKind,
    scaler: &Standardizer,
) -> Vec<SpecMat> {
    data.iter()
        .map(|ev| {
            let feats = ev
                .candidates
                .iter()
                .map(|c| scaler.transform(&kind.raw_vector(c)))
                .collect();
            SpecMat { feats, gold: ev.gold }
        })
        .collect()
}

fn spec_loss_grad(spec: &SpecMat, weights: &[f64], out_grad: &mut [f64]) -> Option<f64> {
    let gold = spec.gold?;
    if gold >= spec.feats.len() {
        return None;
    }
    let scores: Vec<f64> = spec
        .feats
        .iter()
        .map(|f| weights.iter().zip(f.iter()).map(|(&w, &x)| w * x).sum())
        .collect();
    let max = scores.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let mut denom = 0.0f64;
    for s in &scores {
        denom += (s - max).exp();
    }
    let log_denom = max + denom.ln();
    let loss = log_denom - scores[gold];
    for (i, f) in spec.feats.iter().enumerate() {
        let p = (scores[i] - max).exp() / denom;
        let target = if i == gold { 1.0 } else { 0.0 };
        let err = p - target;
        for (j, &x) in f.iter().enumerate() {
            out_grad[j] += err * x;
        }
    }
    Some(loss)
}

fn full_loss_grad_mats(mats: &[SpecMat], weights: &[f64], l2: f64) -> (f64, Vec<f64>) {
    // Fixed thread count (up to 8) with contiguous chunks and ordered
    // reduction, so results are deterministic across runs and machines.
    let dim = weights.len();
    let threads = (mats.len().min(8)).max(1);
    let (loss_sum, grad_sum, spectra) = if threads <= 1 || mats.len() < 64 {
        let mut grad = vec![0.0f64; dim];
        let mut loss = 0.0f64;
        let mut spectra = 0usize;
        for spec in mats {
            if let Some(l) = spec_loss_grad(spec, weights, &mut grad) {
                loss += l;
                spectra += 1;
            }
        }
        (loss, grad, spectra)
    } else {
        let chunk = mats.len().div_ceil(threads);
        let mut partials: Vec<(f64, Vec<f64>, usize)> = Vec::new();
        std::thread::scope(|scope| {
            let mut handles = Vec::new();
            for piece in mats.chunks(chunk) {
                handles.push(scope.spawn(move || {
                    let mut grad = vec![0.0f64; dim];
                    let mut loss = 0.0f64;
                    let mut spectra = 0usize;
                    for spec in piece {
                        if let Some(l) = spec_loss_grad(spec, weights, &mut grad) {
                            loss += l;
                            spectra += 1;
                        }
                    }
                    (loss, grad, spectra)
                }));
            }
            for handle in handles {
                partials.push(handle.join().expect("loss worker runs"));
            }
        });
        let mut loss = 0.0f64;
        let mut grad = vec![0.0f64; dim];
        let mut spectra = 0usize;
        for (l, g, n) in partials {
            loss += l;
            for (a, b) in grad.iter_mut().zip(g.iter()) {
                *a += *b;
            }
            spectra += n;
        }
        (loss, grad, spectra)
    };
    let mut loss = loss_sum;
    let mut grad = grad_sum;
    if spectra > 0 {
        let n = spectra as f64;
        loss /= n;
        for g in grad.iter_mut() {
            *g /= n;
        }
    }
    for (w, g) in weights.iter().zip(grad.iter_mut()) {
        loss += 0.5 * l2 * w * w;
        *g += l2 * w;
    }
    (loss, grad)
}

/// Converged-training configuration (FE3): full-batch or large-batch Adam
/// with a decaying step.
#[derive(Clone, Copy, Debug)]
pub struct ConvergedTrainConfig {
    /// Hard cap on passes over the train spectra.
    pub max_epochs: usize,
    /// Spectra per gradient step; values at or above the eligible count give
    /// full-batch training (one Adam step per epoch).
    pub batch_spectra: usize,
    /// Initial Adam step size; epoch `e` uses `lr_init / (1 + lr_decay * e)`.
    pub lr_init: f64,
    /// Per-epoch decay rate of the step size.
    pub lr_decay: f64,
    /// L2 penalty (`0.5 * l2 * ||w||²` on standardised weights).
    pub l2: f64,
    /// Seed of the per-epoch spectrum order (large-batch only; full-batch is
    /// order-independent).
    pub seed: u64,
}

/// Outcome of [`train_softmax_converged`] / [`train_nonlinear_converged`].
#[derive(Clone, Debug)]
pub struct ConvergedTrainResult {
    /// Weights in standardised units.
    pub weights: Vec<f64>,
    /// Passes actually run (`<= max_epochs`).
    pub epochs_used: usize,
    /// Whether the 20-epoch relative-decrease criterion (`< 1e-5`) was met.
    pub converged: bool,
    /// Final train objective (mean NLL plus the L2 term).
    pub train_nll: f64,
}

fn adam_converged(
    mats: &[SpecMat],
    dim: usize,
    cfg: &ConvergedTrainConfig,
    init: Option<&[f64]>,
) -> ConvergedTrainResult {
    let mut weights = match init {
        Some(w0) => w0.to_vec(),
        None => vec![0.0f64; dim],
    };
    let eligible: Vec<usize> = mats
        .iter()
        .enumerate()
        .filter(|(_, s)| s.gold.is_some())
        .map(|(i, _)| i)
        .collect();
    if eligible.is_empty() || dim == 0 || cfg.max_epochs == 0 {
        let (loss, _) = full_loss_grad_mats(mats, &weights, cfg.l2);
        return ConvergedTrainResult {
            weights,
            epochs_used: 0,
            converged: true,
            train_nll: loss,
        };
    }
    let full_batch = cfg.batch_spectra >= eligible.len();
    let beta1 = 0.9f64;
    let beta2 = 0.999f64;
    let eps = 1e-8f64;
    let mut m = vec![0.0f64; dim];
    let mut v = vec![0.0f64; dim];
    let mut step: u64 = 0;
    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let mut order = eligible.clone();
    let (mut prev_loss, _) = full_loss_grad_mats(mats, &weights, cfg.l2);
    let mut last_block_loss = prev_loss;
    let mut converged = false;
    let mut epochs_run = 0usize;
    for epoch in 0..cfg.max_epochs {
        let lr = cfg.lr_init / (1.0 + cfg.lr_decay * epoch as f64);
        if full_batch {
            let (_, grad) = full_loss_grad_mats(mats, &weights, cfg.l2);
            step += 1;
            let t = step as f64;
            for j in 0..dim {
                m[j] = beta1 * m[j] + (1.0 - beta1) * grad[j];
                v[j] = beta2 * v[j] + (1.0 - beta2) * grad[j] * grad[j];
                let m_hat = m[j] / (1.0 - beta1.powf(t));
                let v_hat = v[j] / (1.0 - beta2.powf(t));
                weights[j] -= lr * m_hat / (v_hat.sqrt() + eps);
            }
        } else {
            for i in (1..order.len()).rev() {
                let j = rng.random_range(0..=i);
                order.swap(i, j);
            }
            let batch = cfg.batch_spectra.max(1);
            for chunk in order.chunks(batch) {
                let subset: Vec<SpecMat> = chunk.iter().map(|&i| mats[i].clone()).collect();
                let (_, grad) = full_loss_grad_mats(&subset, &weights, cfg.l2);
                step += 1;
                let t = step as f64;
                for j in 0..dim {
                    m[j] = beta1 * m[j] + (1.0 - beta1) * grad[j];
                    v[j] = beta2 * v[j] + (1.0 - beta2) * grad[j] * grad[j];
                    let m_hat = m[j] / (1.0 - beta1.powf(t));
                    let v_hat = v[j] / (1.0 - beta2.powf(t));
                    weights[j] -= lr * m_hat / (v_hat.sqrt() + eps);
                }
            }
        }
        epochs_run = epoch + 1;
        if epochs_run % 20 == 0 {
            let (cur, _) = full_loss_grad_mats(mats, &weights, cfg.l2);
            let denom = last_block_loss.abs().max(1e-12);
            if cur <= last_block_loss && (last_block_loss - cur) / denom < 1e-5 {
                prev_loss = cur;
                converged = true;
                break;
            }
            last_block_loss = cur;
            prev_loss = cur;
        }
    }
    if !converged && epochs_run % 20 != 0 && epochs_run >= 20 {
        // Final partial block also counts when the cap is not a multiple of
        // 20: compare against the value 20 epochs earlier is unavailable, so
        // just record the final loss without claiming convergence.
        let (cur, _) = full_loss_grad_mats(mats, &weights, cfg.l2);
        prev_loss = cur;
    } else if epochs_run % 20 != 0 {
        let (cur, _) = full_loss_grad_mats(mats, &weights, cfg.l2);
        prev_loss = cur;
    }
    ConvergedTrainResult {
        weights,
        epochs_used: epochs_run,
        converged,
        train_nll: prev_loss,
    }
}

/// Train a linear ranker on standardised features to convergence (FE3).
///
/// `scaler` must be fitted on the same train `data` (see
/// [`fit_standardizer`]); weights are in standardised units with `l2` on
/// those units. Weights start at zero; see
/// [`train_softmax_converged_init`] for a warm start.
pub fn train_softmax_converged(
    data: &[SpectrumEvidence],
    dims: &[usize],
    scaler: &Standardizer,
    cfg: &ConvergedTrainConfig,
) -> Result<ConvergedTrainResult> {
    train_softmax_converged_init(data, dims, scaler, cfg, None)
}

/// Train a linear ranker on standardised features to convergence (FE3),
/// starting from `init` when present (warm start, e.g. subset weights).
///
/// A `None` start is the zero vector. `init` must match `dims` in length.
/// Starting a superset model from its subset's weights (zeros elsewhere)
/// reproduces the subset loss exactly, so nesting holds at initialisation.
pub fn train_softmax_converged_init(
    data: &[SpectrumEvidence],
    dims: &[usize],
    scaler: &Standardizer,
    cfg: &ConvergedTrainConfig,
    init: Option<&[f64]>,
) -> Result<ConvergedTrainResult> {
    if dims.is_empty() {
        return Err(Error::config(
            "train_softmax_converged: no feature dimensions selected".to_string(),
        ));
    }
    if scaler.means.len() != dims.len() || scaler.scales.len() != dims.len() {
        return Err(Error::config(format!(
            "train_softmax_converged: scaler dim {} but {} dims",
            scaler.means.len(),
            dims.len()
        )));
    }
    if cfg.lr_init <= 0.0 || !cfg.lr_init.is_finite() {
        return Err(Error::config(format!(
            "train_softmax_converged: lr_init {} is not positive finite",
            cfg.lr_init
        )));
    }
    if let Some(w0) = init {
        if w0.len() != dims.len() {
            return Err(Error::config(format!(
                "train_softmax_converged_init: init dim {} but {} dims",
                w0.len(),
                dims.len()
            )));
        }
    }
    Ok(adam_converged(
        &build_linear_matrix(data, dims, scaler),
        dims.len(),
        cfg,
        init,
    ))
}

/// Train an FE3 nonlinear ranker on standardised product features (FE3),
/// starting from zero (see [`train_nonlinear_converged_init`] to warm-start).
pub fn train_nonlinear_converged(
    data: &[SpectrumEvidence],
    kind: NonlinearKind,
    scaler: &Standardizer,
    cfg: &ConvergedTrainConfig,
) -> Result<ConvergedTrainResult> {
    train_nonlinear_converged_init(data, kind, scaler, cfg, None)
}

/// Train an FE3 nonlinear ranker from `init` when present (warm start).
pub fn train_nonlinear_converged_init(
    data: &[SpectrumEvidence],
    kind: NonlinearKind,
    scaler: &Standardizer,
    cfg: &ConvergedTrainConfig,
    init: Option<&[f64]>,
) -> Result<ConvergedTrainResult> {
    if scaler.means.len() != kind.len() {
        return Err(Error::config(format!(
            "train_nonlinear_converged: scaler dim {} but kind len {}",
            scaler.means.len(),
            kind.len()
        )));
    }
    if let Some(w0) = init {
        if w0.len() != kind.len() {
            return Err(Error::config(format!(
                "train_nonlinear_converged_init: init dim {} but kind len {}",
                w0.len(),
                kind.len()
            )));
        }
    }
    Ok(adam_converged(&build_nonlinear_matrix(data, kind, scaler), kind.len(), cfg, init))
}

/// Warm-start weights for `to_dims` from a model over `from_dims`: shared
/// dimensions copy their weights, new dimensions start at zero.
pub fn project_init(from_dims: &[usize], from_w: &[f64], to_dims: &[usize]) -> Vec<f64> {
    to_dims
        .iter()
        .map(|&d| {
            from_dims
                .iter()
                .position(|&e| e == d)
                .map(|j| from_w[j])
                .unwrap_or(0.0)
        })
        .collect()
}

/// Warm-start weights for an FE3 nonlinear ranker from the linear prior
/// model: the leading 12 entries are the same prior features in the same
/// order (identical standardisation statistics on the same train data), so
/// they copy `prior_w` while every product starts at zero.
pub fn nonlinear_init_from_prior(prior_dims: &[usize], prior_w: &[f64], kind: NonlinearKind) -> Vec<f64> {
    let mut out = vec![0.0f64; kind.len()];
    for (k, &d) in PRIOR_DIMS.iter().enumerate() {
        if let Some(j) = prior_dims.iter().position(|&e| e == d) {
            out[k] = prior_w[j];
        }
    }
    out
}

/// Mean softmax NLL (plus `l2` on standardised weights) of a linear ranker on
/// `data` with the train `scaler` (train or validation data alike).
pub fn standardized_nll(
    data: &[SpectrumEvidence],
    dims: &[usize],
    scaler: &Standardizer,
    weights: &[f64],
    l2: f64,
) -> Result<f64> {
    if dims.len() != weights.len() {
        return Err(Error::config(format!(
            "standardized_nll: {} weights but {} dims",
            weights.len(),
            dims.len()
        )));
    }
    Ok(full_loss_grad_mats(&build_linear_matrix(data, dims, scaler), weights, l2).0)
}

/// Mean softmax NLL (plus `l2`) of a nonlinear ranker with the train scaler.
pub fn standardized_nll_nonlinear(
    data: &[SpectrumEvidence],
    kind: NonlinearKind,
    scaler: &Standardizer,
    weights: &[f64],
    l2: f64,
) -> Result<f64> {
    if weights.len() != kind.len() {
        return Err(Error::config(format!(
            "standardized_nll_nonlinear: {} weights but kind len {}",
            weights.len(),
            kind.len()
        )));
    }
    Ok(full_loss_grad_mats(&build_nonlinear_matrix(data, kind, scaler), weights, l2).0)
}

/// Enforced nestedness check on train NLLs (FE3).
///
/// Expected keys: `prior_only`, `prior_plus_residual`,
/// `prior_plus_residual_plus_evidence`, `prior_plus_evidence`,
/// `prior_nonlinear`, `prior_nonlinear_plus_residual_plus_evidence`. Missing
/// keys are skipped (shuffled variants only hold the evidence rankers they
/// train). Every present superset must satisfy
/// `NLL(sup) <= NLL(sub) + tol`, and the two full models must satisfy
/// `NLL(full) <= min(components) + tol`. Returns one message per violation
/// (empty when nested).
pub fn check_nestedness(train_nll: &HashMap<String, f64>, tol: f64) -> Vec<String> {
    let mut out = Vec::new();
    let get = |k: &str| train_nll.get(k).copied();
    let check = |sup: &str, sub: &str, sup_v: Option<f64>, sub_v: Option<f64>, out: &mut Vec<String>| {
        if let (Some(a), Some(b)) = (sup_v, sub_v) {
            if a > b + tol {
                out.push(format!(
                    "nestedness violation: train NLL({sup}) = {a:.6} > NLL({sub}) = {b:.6} + {tol}"
                ));
            }
        }
    };
    let (prior, pr, pre, pe, nl, nlpre) = (
        get("prior_only"),
        get("prior_plus_residual"),
        get("prior_plus_residual_plus_evidence"),
        get("prior_plus_evidence"),
        get("prior_nonlinear"),
        get("prior_nonlinear_plus_residual_plus_evidence"),
    );
    check("prior_plus_residual", "prior_only", pr, prior, &mut out);
    check("prior_plus_evidence", "prior_only", pe, prior, &mut out);
    check("prior_nonlinear", "prior_only", nl, prior, &mut out);
    if let Some(full) = pre {
        let mut best: Option<f64> = None;
        let mut label = String::new();
        for (name, v) in [("prior_plus_residual", pr), ("prior_plus_evidence", pe)] {
            if let Some(x) = v {
                if best.is_none_or(|b: f64| x < b) {
                    best = Some(x);
                    label = name.to_string();
                }
            }
        }
        if let Some(b) = best {
            if full > b + tol {
                out.push(format!(
                    "nestedness violation: train NLL(prior_plus_residual_plus_evidence) = {full:.6} > min NLL({label}) = {b:.6} + {tol}"
                ));
            }
        }
    }
    if let Some(full) = nlpre {
        let mut best: Option<f64> = None;
        let mut label = String::new();
        for (name, v) in [
            ("prior_only", prior),
            ("prior_plus_residual", pr),
            ("prior_plus_evidence", pe),
            ("prior_plus_residual_plus_evidence", pre),
            ("prior_nonlinear", nl),
        ] {
            if let Some(x) = v {
                if best.is_none_or(|b: f64| x < b) {
                    best = Some(x);
                    label = name.to_string();
                }
            }
        }
        if let Some(b) = best {
            if full > b + tol {
                out.push(format!(
                    "nestedness violation: train NLL(prior_nonlinear_plus_residual_plus_evidence) = {full:.6} > min NLL({label}) = {b:.6} + {tol}"
                ));
            }
        }
    }
    out
}

/// Reference [`ion_assign`] call behind the equivalence claim, with limits
/// large enough to visit every sub-vector of a small parent.
pub fn reference_ion_assign(
    parent: &Composition,
    adduct_id: u16,
    peak_mz: u32,
    mz_uncertainty: u32,
    ppm_tenths: u32,
) -> Result<super::ion::IonAssignment> {
    let limits = IonLimits {
        work_max: u32::MAX,
        kept: 64,
    };
    ion_assign(parent, adduct_id, peak_mz, mz_uncertainty, ppm_tenths, &limits)
}

/// Neutral precursor mass of an export spectrum, or `None` when the adduct is
/// unknown or the mass leaves `u32` (then no formula search runs).
pub fn spectrum_parent_mass(precursor_mz: u32, adduct_id: u16) -> Option<u32> {
    parent_mass(precursor_mz, adduct_id).ok()
}

/// Integer mass of a composition, or `None` on overflow.
pub fn candidate_mass(c: &Composition) -> Option<u32> {
    composition_mass(c).ok()
}
