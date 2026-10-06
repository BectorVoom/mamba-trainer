//! Request schemas and validation (contract §§3 and 8).
//!
//! Host-side, `serde`-serializable, row-major batches. Pure host Rust: no
//! kernels, no tensors, no new dependencies.

use crate::backend::DType;
use crate::error::{Error, Result};
use crate::ssm::SsmConfig;
use serde::{Deserialize, Serialize};

use super::chem::{CHEMISTRY_VERSION, adduct};
use super::grammar::{GRAMMAR_VERSION, Limits, STOP, TRAVERSAL_VERSION, Token, replay};
use super::{chem, targets};

/// Schema version every batch starts with; readers reject anything else.
///
/// V1 (§1.2): `GenerationConfig`, `CandidateBatch` and `ModelConfig` move to
/// version 2. A version-1 document still loads (see each `validate`); any
/// other version is `Error::Config`. `SpectrumBatch` and `ChemistryDomain`
/// stay at version 1.
pub const SCHEMA_VERSION: u32 = 2;
/// Previous schema version, still accepted by the three V1 schemas with the
/// version-1 defaults below.
pub const SCHEMA_VERSION_V1: u32 = 1;
/// Schema version of the unchanged V0 schemas (`SpectrumBatch`,
/// `ChemistryDomain`).
pub const SPECTRUM_SCHEMA_VERSION: u32 = 1;

/// Uncertainty sentinel: precision unknown, exact-mass decisions disabled.
pub const UNKNOWN_UNCERTAINTY: u32 = u32::MAX;

/// Padding marker for [`SpectrumBatch::peak_id`].
pub const NO_PEAK: u32 = u32::MAX;

/// Missing formula hypothesis in [`CandidateBatch::formula_row`].
pub const NO_FORMULA: u32 = u32::MAX;

/// Largest value any single model dimension may take (`ModelConfig::validate`).
pub const MAX_MODEL_DIMENSION: usize = 65_536;

/// Request status bits of contract §8 (fatal: 0–15, warnings: 16–31).
pub mod request_status {
    /// No usable peak (`peak_count` 0 or every valid intensity zero).
    pub const EMPTY_SPECTRUM: u32 = 1 << 0;
    /// A valid intensity or a known energy is NaN, infinite or negative.
    pub const NONFINITE_INPUT: u32 = 1 << 1;
    /// A valid peak has a negative intensity.
    pub const NEGATIVE_INTENSITY: u32 = 1 << 2;
    /// A valid peak has m/z 0.
    pub const INVALID_PEAK: u32 = 1 << 3;
    /// Polarity is not `+1` or `-1`.
    pub const INVALID_POLARITY: u32 = 1 << 4;
    /// Polarity sign differs from the (known) adduct's charge.
    pub const POLARITY_ADDUCT_CONFLICT: u32 = 1 << 5;
    /// Adduct id is 0 (unknown).
    pub const INSUFFICIENT_METADATA: u32 = 1 << 6;
    /// Adduct id is non-zero but not in the chemistry domain.
    pub const UNSUPPORTED_ADDUCT: u32 = 1 << 7;
    /// Precursor m/z outside 50–2000 Da.
    pub const PRECURSOR_OUT_OF_RANGE: u32 = 1 << 8;
    /// `peak_count` exceeds the batch's `n_raw` capacity.
    pub const OVER_CAPACITY: u32 = 1 << 9;
    /// No table row in the precursor window.
    pub const FORMULA_ABSENT: u32 = 1 << 10;
    /// An integer mass sum left the `u32` range.
    pub const MASS_OVERFLOW: u32 = 1 << 11;
    /// The caller held more raw peaks than `peak_count` reports.
    pub const RAW_TRUNCATED: u32 = 1 << 16;
    /// A peak or precursor uncertainty is unknown.
    pub const EXACT_MASS_UNAVAILABLE: u32 = 1 << 17;
    /// A formula-work limit cut the search short.
    pub const FORMULA_SEARCH_EXHAUSTED: u32 = 1 << 18;
    /// The device cap `N` dropped peaks.
    pub const PEAKS_TRUNCATED: u32 = 1 << 19;
    /// Bits 0–15: the spectrum yields no candidate.
    pub const FATAL_MASK: u32 = 0x0000_FFFF;

    /// Names of the set bits, in bit order; undefined bits are skipped.
    pub fn names(bits: u32) -> Vec<&'static str> {
        const TABLE: [(u32, &str); 16] = [
            (EMPTY_SPECTRUM, "empty_spectrum"),
            (NONFINITE_INPUT, "nonfinite_input"),
            (NEGATIVE_INTENSITY, "negative_intensity"),
            (INVALID_PEAK, "invalid_peak"),
            (INVALID_POLARITY, "invalid_polarity"),
            (POLARITY_ADDUCT_CONFLICT, "polarity_adduct_conflict"),
            (INSUFFICIENT_METADATA, "insufficient_metadata"),
            (UNSUPPORTED_ADDUCT, "unsupported_adduct"),
            (PRECURSOR_OUT_OF_RANGE, "precursor_out_of_range"),
            (OVER_CAPACITY, "over_capacity"),
            (FORMULA_ABSENT, "formula_absent"),
            (MASS_OVERFLOW, "mass_overflow"),
            (RAW_TRUNCATED, "raw_truncated"),
            (EXACT_MASS_UNAVAILABLE, "exact_mass_unavailable"),
            (FORMULA_SEARCH_EXHAUSTED, "formula_search_exhausted"),
            (PEAKS_TRUNCATED, "peaks_truncated"),
        ];
        TABLE
            .iter()
            .filter_map(|(bit, name)| (bits & bit != 0).then_some(*name))
            .collect()
    }
}

/// Candidate status bits of contract §8, one set per trajectory.
pub mod candidate_status {
    /// The trace ended with STOP.
    pub const FINISHED: u32 = 1 << 0;
    /// `max_steps` reached without STOP.
    pub const TRUNCATED: u32 = 1 << 1;
    /// No legal action existed mid-trace.
    pub const NO_VALID_ACTION: u32 = 1 << 2;
    /// The final graph fails the validity rules.
    pub const INVALID_FINAL: u32 = 1 << 3;
    /// A later trajectory with the same trace and formula.
    pub const DUPLICATE_TRACE: u32 = 1 << 4;
    /// The conditioning formula is the oracle one.
    pub const FORMULA_SOURCE_ORACLE: u32 = 1 << 5;
    /// The request failed; the record carries no trace.
    pub const REQUEST_FAILED: u32 = 1 << 6;
    /// An exact comparison proved equality with an earlier trajectory of the
    /// same spectrum (V1 §4.2; the candidate stays, ranked out).
    pub const DUPLICATE_GRAPH: u32 = 1 << 7;
    /// Some comparison of this (the later) trajectory ran out of budget (V1
    /// §4.2); not a duplicate flag, and the candidate stays eligible.
    pub const IDENTITY_UNRESOLVED: u32 = 1 << 8;

    /// Names of the set bits, in bit order; undefined bits are skipped.
    pub fn names(bits: u32) -> Vec<&'static str> {
        const TABLE: [(u32, &str); 9] = [
            (FINISHED, "finished"),
            (TRUNCATED, "truncated"),
            (NO_VALID_ACTION, "no_valid_action"),
            (INVALID_FINAL, "invalid_final"),
            (DUPLICATE_TRACE, "duplicate_trace"),
            (FORMULA_SOURCE_ORACLE, "formula_source_oracle"),
            (REQUEST_FAILED, "request_failed"),
            (DUPLICATE_GRAPH, "duplicate_graph"),
            (IDENTITY_UNRESOLVED, "identity_unresolved"),
        ];
        TABLE
            .iter()
            .filter_map(|(bit, name)| (bits & bit != 0).then_some(*name))
            .collect()
    }
}

/// Lowest precursor m/z of the V0 request domain, in integer units (50 Da).
pub const PRECURSOR_MIN: u32 = 50_000_000;
/// Highest precursor m/z of the V0 request domain, in integer units (2000 Da).
pub const PRECURSOR_MAX: u32 = 2_000_000_000;

/// A batch of spectra (contract §3.1): `B` spectra, row-major `[B, n_raw]`
/// per-peak fields. Padding slots (index `>= peak_count`) are never read.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct SpectrumBatch {
    /// Schema version; only [`SCHEMA_VERSION`] is accepted.
    pub schema_version: u32,
    /// Raw peak capacity of the shape bucket: 64, 128, 256 or 512.
    pub n_raw: u32,
    /// Stable identity per spectrum: keys the RNG and provenance.
    pub spectrum_id: Vec<u64>,
    /// Peaks the caller held before any host pre-selection.
    pub raw_peak_count: Vec<u32>,
    /// Peaks supplied in this batch (`0` means empty, `> n_raw` over capacity).
    pub peak_count: Vec<u32>,
    /// Caller-side peak index, strictly increasing over the valid peaks.
    pub peak_id: Vec<u32>,
    /// m/z in integer units; padding is 0, a valid 0 is `invalid_peak`.
    pub mz_udalton: Vec<u32>,
    /// Non-negative finite intensities; padding is 0 and never read.
    pub intensity: Vec<f32>,
    /// `0` linear (default), `1` square root of the relative intensity.
    pub intensity_scale: u8,
    /// Half-width of the stored m/z rounding interval (`u32::MAX` unknown).
    pub mz_uncertainty_udalton: Vec<u32>,
    /// Precursor m/z in integer units.
    pub precursor_mz_udalton: Vec<u32>,
    /// Precursor uncertainty (`u32::MAX` skips formula search).
    pub precursor_uncertainty_udalton: Vec<u32>,
    /// Adduct id in the chemistry domain (`0` unknown).
    pub adduct: Vec<u16>,
    /// `+1` or `-1`.
    pub polarity: Vec<i8>,
    /// Mean collision energy in eV (read only when known).
    pub collision_energy_ev: Vec<f32>,
    /// `1` when the eV value is a measurement.
    pub collision_energy_known: Vec<u8>,
    /// Collision energies behind the spectrum (`8` means 8 or more).
    pub energy_count: Vec<u8>,
    /// Fragment tolerance, tenths of a ppm (`0` means the default 100).
    pub fragment_tolerance_ppm_tenths: Vec<u16>,
    /// Precursor tolerance, tenths of a ppm (`0` means the default 200).
    pub precursor_tolerance_ppm_tenths: Vec<u16>,
    /// `0` unknown, `1` timstof, `2` orbitrap, `3` qtof, `4` other.
    pub instrument_class: Vec<u8>,
}

impl SpectrumBatch {
    /// Spectra per batch (`spectrum_id.len()`).
    pub fn len(&self) -> usize {
        self.spectrum_id.len()
    }

    /// Whether the batch holds no spectrum.
    pub fn is_empty(&self) -> bool {
        self.spectrum_id.is_empty()
    }

    /// Validate shapes and per-spectrum metadata (contract §3.1).
    ///
    /// Malformed batches are [`Error::Config`] naming the offending field;
    /// otherwise one request-status bit set per spectrum, with every
    /// applicable bit set. Over-capacity spectra are checked over their first
    /// `n_raw` peaks only, and padding slots are never read.
    pub fn validate(&self) -> Result<Vec<u32>> {
        if self.schema_version != SPECTRUM_SCHEMA_VERSION {
            return Err(Error::config(format!(
                "SpectrumBatch::validate: unknown schema_version {} (expected {SPECTRUM_SCHEMA_VERSION})",
                self.schema_version
            )));
        }
        if !matches!(self.n_raw, 64 | 128 | 256 | 512) {
            return Err(Error::config(format!(
                "SpectrumBatch::validate: n_raw {} is not one of 64, 128, 256, 512",
                self.n_raw
            )));
        }
        let n = self.len();
        let n_raw = self.n_raw as usize;
        let per_spectrum: [(&str, usize); 13] = [
            ("raw_peak_count", self.raw_peak_count.len()),
            ("peak_count", self.peak_count.len()),
            ("mz_uncertainty_udalton", self.mz_uncertainty_udalton.len()),
            ("precursor_mz_udalton", self.precursor_mz_udalton.len()),
            (
                "precursor_uncertainty_udalton",
                self.precursor_uncertainty_udalton.len(),
            ),
            ("adduct", self.adduct.len()),
            ("polarity", self.polarity.len()),
            ("collision_energy_ev", self.collision_energy_ev.len()),
            ("collision_energy_known", self.collision_energy_known.len()),
            ("energy_count", self.energy_count.len()),
            (
                "fragment_tolerance_ppm_tenths",
                self.fragment_tolerance_ppm_tenths.len(),
            ),
            (
                "precursor_tolerance_ppm_tenths",
                self.precursor_tolerance_ppm_tenths.len(),
            ),
            ("instrument_class", self.instrument_class.len()),
        ];
        for (field, len) in per_spectrum {
            if len != n {
                return Err(Error::config(format!(
                    "SpectrumBatch::validate: field {field} has length {len} for {n} spectra"
                )));
            }
        }
        let per_peak: [(&str, usize); 3] = [
            ("peak_id", self.peak_id.len()),
            ("mz_udalton", self.mz_udalton.len()),
            ("intensity", self.intensity.len()),
        ];
        for (field, len) in per_peak {
            if len != n * n_raw {
                return Err(Error::config(format!(
                    "SpectrumBatch::validate: field {field} has length {len} for {n} spectra of n_raw {n_raw}"
                )));
            }
        }
        let mut seen = std::collections::HashSet::new();
        for id in &self.spectrum_id {
            if !seen.insert(id) {
                return Err(Error::config(format!(
                    "SpectrumBatch::validate: duplicate spectrum_id {id}"
                )));
            }
        }
        if self.intensity_scale > 1 {
            return Err(Error::config(format!(
                "SpectrumBatch::validate: intensity_scale {} is not 0 or 1",
                self.intensity_scale
            )));
        }
        for (b, known) in self.collision_energy_known.iter().enumerate() {
            if *known > 1 {
                return Err(Error::config(format!(
                    "SpectrumBatch::validate: spectrum {b} collision_energy_known {known} is not 0 or 1"
                )));
            }
        }
        for (b, count) in self.energy_count.iter().enumerate() {
            if *count > 8 {
                return Err(Error::config(format!(
                    "SpectrumBatch::validate: spectrum {b} energy_count {count} exceeds 8"
                )));
            }
        }
        for (b, tol) in self.fragment_tolerance_ppm_tenths.iter().enumerate() {
            if *tol > 1000 {
                return Err(Error::config(format!(
                    "SpectrumBatch::validate: spectrum {b} fragment_tolerance_ppm_tenths {tol} exceeds 1000"
                )));
            }
        }
        for (b, tol) in self.precursor_tolerance_ppm_tenths.iter().enumerate() {
            if *tol > 1000 {
                return Err(Error::config(format!(
                    "SpectrumBatch::validate: spectrum {b} precursor_tolerance_ppm_tenths {tol} exceeds 1000"
                )));
            }
        }
        for (b, class) in self.instrument_class.iter().enumerate() {
            if *class > 4 {
                return Err(Error::config(format!(
                    "SpectrumBatch::validate: spectrum {b} instrument_class {class} exceeds 4"
                )));
            }
        }
        let mut statuses = Vec::with_capacity(n);
        for b in 0..n {
            let base = b * n_raw;
            let count = self.peak_count[b] as usize;
            // Over capacity the first `n_raw` peaks are the valid ones.
            let valid = count.min(n_raw);
            for k in 1..valid {
                if self.peak_id[base + k - 1] >= self.peak_id[base + k] {
                    return Err(Error::config(format!(
                        "SpectrumBatch::validate: spectrum {b} peak_id is not strictly increasing \
                         ({} then {})",
                        self.peak_id[base + k - 1],
                        self.peak_id[base + k]
                    )));
                }
            }
            let mut bits = 0u32;
            if count > n_raw {
                bits |= request_status::OVER_CAPACITY;
            }
            let mut all_zero = true;
            for k in 0..valid {
                let v = self.intensity[base + k];
                // The device treats a transformed intensity at or above FINITE_MAX as
                // non-finite (a fast-math-safe range test), so the host rejects those
                // values here rather than letting them vanish silently on the device.
                let limit = if self.intensity_scale == 1 {
                    crate::tensor::ops::ms2::FINITE_MAX.sqrt()
                } else {
                    crate::tensor::ops::ms2::FINITE_MAX
                };
                if v.is_nan() || v.is_infinite() || v >= limit {
                    bits |= request_status::NONFINITE_INPUT;
                }
                if v < 0.0 {
                    bits |= request_status::NEGATIVE_INTENSITY;
                }
                if v != 0.0 {
                    all_zero = false;
                }
                if self.mz_udalton[base + k] == 0 {
                    bits |= request_status::INVALID_PEAK;
                }
            }
            if count == 0 || (valid > 0 && all_zero) {
                bits |= request_status::EMPTY_SPECTRUM;
            }
            let polarity = self.polarity[b];
            if polarity != 1 && polarity != -1 {
                bits |= request_status::INVALID_POLARITY;
            } else if let Some(a) = adduct(self.adduct[b])
                && i32::from(polarity) != a.charge
            {
                bits |= request_status::POLARITY_ADDUCT_CONFLICT;
            }
            if self.adduct[b] == 0 {
                bits |= request_status::INSUFFICIENT_METADATA;
            } else if adduct(self.adduct[b]).is_none() {
                bits |= request_status::UNSUPPORTED_ADDUCT;
            }
            let precursor = self.precursor_mz_udalton[b];
            if !(PRECURSOR_MIN..=PRECURSOR_MAX).contains(&precursor) {
                bits |= request_status::PRECURSOR_OUT_OF_RANGE;
            }
            if self.collision_energy_known[b] == 1 {
                let ev = self.collision_energy_ev[b];
                if ev.is_nan() || ev.is_infinite() || ev < 0.0 {
                    bits |= request_status::NONFINITE_INPUT;
                }
            }
            if self.raw_peak_count[b] > self.peak_count[b] {
                bits |= request_status::RAW_TRUNCATED;
            }
            if self.raw_peak_count[b] < self.peak_count[b] {
                return Err(Error::config(format!(
                    "SpectrumBatch::validate: spectrum {b} raw_peak_count {} \
                     is below peak_count {}",
                    self.raw_peak_count[b], self.peak_count[b]
                )));
            }
            for k in 0..valid {
                if self.peak_id[base + k] >= self.raw_peak_count[b] {
                    return Err(Error::config(format!(
                        "SpectrumBatch::validate: spectrum {b} peak {k} peak_id {} \
                         is at or above raw_peak_count {}",
                        self.peak_id[base + k],
                        self.raw_peak_count[b]
                    )));
                }
            }
            if self.mz_uncertainty_udalton[b] == UNKNOWN_UNCERTAINTY
                || self.precursor_uncertainty_udalton[b] == UNKNOWN_UNCERTAINTY
            {
                bits |= request_status::EXACT_MASS_UNAVAILABLE;
            }
            statuses.push(bits);
        }
        Ok(statuses)
    }

    /// Fragment tolerance in tenths of a ppm (stored 0 means the default 100).
    pub fn fragment_tolerance(&self, b: usize) -> u32 {
        match self.fragment_tolerance_ppm_tenths[b] {
            0 => 100,
            t => u32::from(t),
        }
    }

    /// Largest fragment tolerance over the batch's uploaded peaks (task E4F
    /// item 2, `tol_max`): the maximum over spectra of the tolerance at the
    /// spectrum's largest uploaded peak m/z under that spectrum's ppm, via
    /// [`chem::tolerance`]. Host data only (no device read); padding slots
    /// and zero m/z values are skipped. Returns 0 when the batch holds no
    /// valid peak.
    pub fn max_fragment_tolerance(&self) -> u32 {
        let n_raw = self.n_raw as usize;
        let mut tol_max: u32 = 0;
        for b in 0..self.len() {
            let ppm = self.fragment_tolerance(b);
            let count = (self.peak_count[b] as usize).min(n_raw);
            let mut mz_max: u32 = 0;
            for k in 0..count {
                let mz = self.mz_udalton[b * n_raw + k];
                if mz > mz_max {
                    mz_max = mz;
                }
            }
            if mz_max == 0 {
                continue;
            }
            let tol = chem::tolerance(mz_max, ppm);
            if tol > tol_max {
                tol_max = tol;
            }
        }
        tol_max
    }

    /// Precursor tolerance in tenths of a ppm (stored 0 means the default 200).
    pub fn precursor_tolerance(&self, b: usize) -> u32 {
        match self.precursor_tolerance_ppm_tenths[b] {
            0 => 200,
            t => u32::from(t),
        }
    }
}

/// How candidates are generated (contract §3.4).
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
pub enum GenerationMode {
    /// Ancestral sampling over the legal support.
    Sampling,
    /// Beam search (P5; rejected until then).
    Beam,
}

/// Evaluation controls run through the same generation path (contract §10).
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
pub enum Control {
    /// Real requests.
    None,
    /// Each spectrum gets another spectrum's peaks (metadata and targets kept).
    ShuffledSpectrum,
    /// The peak memory is replaced by the single metadata token.
    MetadataOnly,
    /// The structure prior: like [`Control::MetadataOnly`], and the encoder
    /// additionally sees the unknown row of every metadata embedding and a
    /// zeroed energy and precursor feature. Only the encoder is blinded: the
    /// request still carries its real adduct and precursor, so request
    /// validation, the formula search and the grammar budget are unchanged.
    #[serde(rename = "structure_prior")]
    StructurePrior,
}

/// Formula candidate source (V1 §1.2): where the scored compositions come
/// from. `Table` is the V0 resident table; `Enumerate` is §1.4.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
pub enum FormulaSource {
    /// Resident formula table (the only V1-B1 source).
    Table,
    /// Bounded enumeration (§1.4).
    Enumerate,
}

/// What the formula head ranks with (architecture §1.6): the feature layout
/// of the scored candidates. `Counts` is today's 10 `ln(1 + count)` features;
/// `Evidence` is the 16-feature layout (counts, precursor residual and
/// explained-peak features) scored through the additive evidence branch.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum FormulaFeatures {
    /// Today's 10 features (the default: every earlier document and
    /// checkpoint loads as this).
    #[default]
    Counts,
    /// The 16-feature evidence layout (architecture §1.6).
    Evidence,
}

fn default_formula_source() -> FormulaSource {
    FormulaSource::Table
}

fn default_formula_window() -> u32 {
    32
}

fn default_enum_lanes_max() -> u32 {
    262_144
}

fn default_enum_lane_visits_max() -> u32 {
    4_096
}

/// Default worst-case visits per count/fill launch (`enum_dispatch_visits_max`).
///
/// Task T6B (plan P4.9, P8.2 / O1): 16,000,000, from
/// `bench/results/ms2/p4_enum_dispatch_bench_wgpu_radeon860m.json` — worst-case
/// launch about 0.1 s, a third of the launches, less than half the stage time
/// on real data. The bound changes no result (the lane takes the absolute lane
/// index), only the launch count.
fn default_enum_dispatch_visits_max() -> u32 {
    16_000_000
}

fn default_allocation() -> AllocationMode {
    AllocationMode::RoundRobin
}

fn default_identity() -> IdentityMode {
    IdentityMode::TraceOnly
}

fn default_identity_work_max() -> u32 {
    4096
}

/// Default `returned`: 0 means "the default", resolved by
/// [`GenerationConfig::effective_returned`] to `min(10, K)`.
fn default_returned() -> u32 {
    0
}

fn default_evidence() -> bool {
    false
}

fn default_ion_request_work_max() -> u32 {
    268_435_456
}

/// Default per-lane visit budget of the evidence walk (architecture §1.6,
/// `W = 2,048`).
fn default_formula_evidence_work_max() -> u32 {
    2_048
}

/// Default worst-case hydrogen trials covered by one evidence dispatch
/// launch (architecture §1.6, `2^33` hydrogen trials).
///
/// Measured by the supervisor on wgpu (Radeon 860M) with
/// `examples/bench_ms2_evidence.rs` at B = 16, M = 2048 (32,768 lanes): a
/// launch's time is set by its longest lane (about 0.1 s for a worst-case
/// lane), not by how many lanes it holds, so splitting only multiplies the
/// cost. At `2^28` real spectra took 25 launches (28.9 ms per call) and the
/// adversarial case 586 launches (63.9 s per call); at `2^33` real spectra
/// run in one launch (5.6 ms per call) while a worst-case launch stays near
/// 0.12 s, far below the driver's job timeout. Larger bounds were faster on
/// the CPU runtime too, so both runtimes share this default.
fn default_formula_evidence_dispatch_max() -> u64 {
    8_589_934_592
}

/// Fragment-ion assignment configuration (architecture §2).
///
/// `None` (or a version-1 document) means assignment disabled: exactly today's
/// behaviour and results. `Some` enables the assignment head with `J`
/// hypotheses per peak (`1..=8`), a per-peak visit budget `work_max` and a
/// label capacity `labels` (`L`).
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct AssignmentConfig {
    /// Hypotheses kept per peak (`J`, `1..=8`).
    #[serde(default = "default_assignment_hypotheses")]
    pub hypotheses: u32,
    /// Per-peak visit budget (`ion_work_max`).
    #[serde(default = "default_assignment_work_max")]
    pub work_max: u32,
    /// Label capacity (`L`).
    #[serde(default = "default_assignment_labels")]
    pub labels: u32,
}

fn default_assignment_hypotheses() -> u32 {
    4
}

fn default_assignment_work_max() -> u32 {
    4096
}

fn default_assignment_labels() -> u32 {
    64
}

impl Default for AssignmentConfig {
    fn default() -> Self {
        Self {
            hypotheses: default_assignment_hypotheses(),
            work_max: default_assignment_work_max(),
            labels: default_assignment_labels(),
        }
    }
}

impl AssignmentConfig {
    /// Check `1 <= hypotheses <= 8`, `work_max != 0` and `labels != 0`.
    pub fn validate(&self) -> Result<()> {
        if !(1..=8).contains(&self.hypotheses) {
            return Err(Error::config(format!(
                "AssignmentConfig::validate: hypotheses {} is not in 1..=8",
                self.hypotheses
            )));
        }
        if self.work_max == 0 {
            return Err(Error::config(
                "AssignmentConfig::validate: work_max 0 is not non-zero".to_string(),
            ));
        }
        if self.labels == 0 {
            return Err(Error::config(
                "AssignmentConfig::validate: labels 0 is not non-zero".to_string(),
            ));
        }
        Ok(())
    }
}

/// Evidence capacity `E` of architecture §2.4: at most 4 records per candidate.
pub const EVIDENCE_CAP: usize = 4;

/// How the `K` trajectories share the retained formulas (V1 §3.2).
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
pub enum AllocationMode {
    /// Trajectory `k` uses formula `k mod top_count` (the V0 rule).
    RoundRobin,
    /// Every retained formula first receives one trajectory, the rest shared
    /// proportionally to the renormalised retained probabilities.
    Proportional,
}

/// Graph identity resolution (V1 §4.2).
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
pub enum IdentityMode {
    /// Trace equality only: no identity kernel is launched and
    /// `identity_resolution` stays 0.
    TraceOnly,
    /// `graph_hash` then `graph_identity` run after validation; the bits are
    /// ORed into the candidates' `status` and `identity_resolution` is 1 or 2.
    Graph,
}

/// Generation hyperparameters (contract §3.4).
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct GenerationConfig {
    /// Schema version; 1 or 2 are accepted (see [`GenerationConfig::validate`]).
    pub schema_version: u32,
    /// Total trajectories per spectrum across formula hypotheses.
    pub trajectories: u32,
    /// Formula hypotheses per spectrum.
    pub formulas: u32,
    /// RNG seed.
    pub seed: u64,
    /// Softmax temperature.
    pub temperature: f32,
    /// Maximum trace steps.
    pub max_steps: u32,
    /// Device-memory budget refused before allocation, never trimmed silently.
    pub max_device_bytes: u64,
    /// Formula rows visited before the search reports exhausted.
    pub formula_rows_visited_max: u32,
    /// Formula rows scored before the search reports exhausted.
    pub formula_rows_scored_max: u32,
    /// Sampling or beam search.
    pub mode: GenerationMode,
    /// Diagnostic only: condition on the true formula.
    pub oracle_formula: bool,
    /// Evaluation control.
    pub control: Control,
    /// Formula candidate source (V1 §1.2). A version-1 document takes `Table`.
    #[serde(default = "default_formula_source")]
    pub formula_source: FormulaSource,
    /// Scored-candidate capacity per spectrum (V1 §1.2, `M`): one of 32,
    /// 128, 512, 2048. A version-1 document takes 32.
    #[serde(default = "default_formula_window")]
    pub formula_window: u32,
    /// Submitted lanes (`B * P`) refused before any launch when above this
    /// (V1 §1.4, `Enumerate` only). A version-1 document takes 262,144.
    #[serde(default = "default_enum_lanes_max")]
    pub enum_lanes_max: u32,
    /// Per-lane visit budget of the enumerating source (V1 §1.4,
    /// `Enumerate` only; `formula_rows_visited_max` is the table source's
    /// limit and is not used here). A version-1 document takes 4,096.
    #[serde(default = "default_enum_lane_visits_max")]
    pub enum_lane_visits_max: u32,
    /// Worst-case visits covered by one count or fill launch (V1 §1.4,
    /// `Enumerate` only): one launch covers at most
    /// `max(1, enum_dispatch_visits_max / enum_lane_visits_max)` lanes, so
    /// its worst case is about `enum_dispatch_visits_max` visits. Must be
    /// non-zero. Default 16,000,000 (task T6B, from
    /// `bench/results/ms2/p4_enum_dispatch_bench_wgpu_radeon860m.json`).
    /// A version-1 document takes 4,000,000.
    #[serde(default = "default_enum_dispatch_visits_max")]
    pub enum_dispatch_visits_max: u32,
    /// How the `K` trajectories share the retained formulas (V1 §3.2).
    /// A version-1 document takes `RoundRobin`.
    #[serde(default = "default_allocation")]
    pub allocation: AllocationMode,
    /// Graph identity resolution (V1 §4.2). A version-1 document takes
    /// `TraceOnly` (no identity kernel is launched).
    #[serde(default = "default_identity")]
    pub identity: IdentityMode,
    /// Per-pair exact-comparison budget of `graph_identity` (V1 §4.2):
    /// at most `identity_work_max` assignments per pair. Must be non-zero.
    /// A version-1 document takes 4096.
    #[serde(default = "default_identity_work_max")]
    pub identity_work_max: u32,
    /// Packed slots per spectrum (`R`, V1 §4.4): `1 <= R <= K`. `0` means
    /// the default `min(10, K)` (see [`GenerationConfig::effective_returned`]);
    /// a version-1 document takes the default.
    #[serde(default = "default_returned")]
    pub returned: u32,
    /// Emit fragment-ion evidence (architecture §2.4). Default false; requires
    /// `ModelConfig::assignment`, else `Error::Config` at preflight. A
    /// version-1 document takes false.
    #[serde(default = "default_evidence")]
    pub evidence: bool,
    /// Request-level ion work bound of §2.1: `B * F * N * ion_work_max` above
    /// this is refused before dispatch. Default `2^28`. Must be non-zero.
    #[serde(default = "default_ion_request_work_max")]
    pub ion_request_work_max: u32,
    /// Per-lane visit budget of the evidence walk (architecture §1.6, `W`):
    /// at most this many sub-composition visits per `(b, m)` lane. Default
    /// 2,048. Must be non-zero. A version-1 document takes 2,048.
    #[serde(default = "default_formula_evidence_work_max")]
    pub formula_evidence_work_max: u32,
    /// Worst-case hydrogen trials covered by one evidence dispatch launch
    /// (architecture §1.6): the `B * M` lanes run in contiguous chunks of
    /// `max(1, dispatch_max / (work_max * P * trials_bound))` lanes, with
    /// `trials_bound` from the host-known `h_cap_max` / `tol_max` (task E4F
    /// item 2). Default `2^33` (8,589,934,592 — measured faster than `2^28`
    /// on both wgpu and CPU; see the default function's doc comment). Must
    /// be non-zero. A version-1 document takes `2^28` (still accepted) or
    /// the default.
    #[serde(default = "default_formula_evidence_dispatch_max")]
    pub formula_evidence_dispatch_max: u64,
}

impl Default for GenerationConfig {
    /// The documented defaults; the visited cap is `u32::MAX` (no limit).
    /// V1 defaults keep V0 behaviour: `formula_source = Table`, `M = 32`.
    fn default() -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            trajectories: 8,
            formulas: 4,
            seed: 0,
            temperature: 1.0,
            max_steps: 22,
            max_device_bytes: 2 * 1024 * 1024 * 1024,
            formula_rows_visited_max: u32::MAX,
            formula_rows_scored_max: 4096,
            mode: GenerationMode::Sampling,
            oracle_formula: false,
            control: Control::None,
            formula_source: FormulaSource::Table,
            formula_window: 32,
            enum_lanes_max: default_enum_lanes_max(),
            enum_lane_visits_max: default_enum_lane_visits_max(),
            enum_dispatch_visits_max: default_enum_dispatch_visits_max(),
            allocation: AllocationMode::RoundRobin,
            identity: IdentityMode::TraceOnly,
            identity_work_max: default_identity_work_max(),
            returned: default_returned(),
            evidence: default_evidence(),
            ion_request_work_max: default_ion_request_work_max(),
            formula_evidence_work_max: default_formula_evidence_work_max(),
            formula_evidence_dispatch_max: default_formula_evidence_dispatch_max(),
        }
    }
}

impl GenerationConfig {
    /// Enforce every documented range under these structure limits.
    ///
    /// Schema versions 1 and 2 are accepted; anything else is
    /// [`Error::Config`] naming both versions. A version-1 document takes
    /// the version-1 values (`formula_source = Table`, `formula_window = 32`,
    /// `enum_lanes_max = 262144`, `enum_lane_visits_max = 4096`,
    /// `enum_dispatch_visits_max = 4000000` (version-1 value; the task-T6B
    /// default 16000000 is also accepted since the bound changes no result,
    /// only launches), `allocation = RoundRobin`,
    /// `identity = TraceOnly`, `identity_work_max = 4096`, `returned = 0`/the
    /// default):
    /// a version-1 config with other values is `Error::Config`.
    /// `formula_window` must be one of 32, 128, 512, 2048 (`Error::Config`
    /// otherwise). `enum_lanes_max`, `enum_lane_visits_max` and
    /// `enum_dispatch_visits_max` must be non-zero (`Error::Config`
    /// otherwise); `formula_rows_visited_max` is
    /// the table source's limit and is not used by `Enumerate`. `usize`
    /// limits that do
    /// not fit `u32`, or whose sum with the 2-token framing overflows, are
    /// [`Error::Config`] (never a truncation). `Beam` is
    /// [`Error::Unsupported`] until P5 builds it.
    ///
    /// `identity_work_max` must be non-zero. `returned` resolves through
    /// [`GenerationConfig::effective_returned`] (`0` means `min(10, K)`) and
    /// the resolved `R` must satisfy `1 <= R <= K` (`Error::Config`
    /// otherwise).
    pub fn validate(&self, max_atoms: usize, max_ring_closures: usize) -> Result<()> {
        if self.schema_version != SCHEMA_VERSION && self.schema_version != SCHEMA_VERSION_V1 {
            return Err(Error::config(format!(
                "GenerationConfig::validate: unknown schema_version {} (expected {SCHEMA_VERSION_V1} or {SCHEMA_VERSION})",
                self.schema_version
            )));
        }
        if self.schema_version == SCHEMA_VERSION_V1
            && (self.formula_source != FormulaSource::Table
                || self.formula_window != 32
                || self.enum_lanes_max != 262_144
                || self.enum_lane_visits_max != 4_096
                || (self.enum_dispatch_visits_max != 4_000_000
                    && self.enum_dispatch_visits_max != 16_000_000)
                || self.allocation != AllocationMode::RoundRobin
                || self.identity != IdentityMode::TraceOnly
                || self.identity_work_max != 4_096
                || self.returned != 0
                || self.evidence
                || self.ion_request_work_max != default_ion_request_work_max()
                || self.formula_evidence_work_max != default_formula_evidence_work_max()
                || (self.formula_evidence_dispatch_max != 268_435_456
                    && self.formula_evidence_dispatch_max
                        != default_formula_evidence_dispatch_max()))
        {
            return Err(Error::config(format!(
                "GenerationConfig::validate: version-1 config must take formula_source Table, formula_window 32, enum_lanes_max 262144, enum_lane_visits_max 4096, enum_dispatch_visits_max 4000000 or 16000000 (the task-T6B default, accepted since the bound changes no result), allocation RoundRobin, identity TraceOnly, identity_work_max 4096, returned 0 (the default), evidence false, ion_request_work_max {} and formula_evidence_work_max {} and formula_evidence_dispatch_max 268435456 (the version-1 value) or {} (the task-F10 default, accepted since the bound changes no result) (got {:?} and {} and {} and {} and {} and {:?} and {:?} and {} and {} and {} and {} and {} and {})",
                default_ion_request_work_max(),
                default_formula_evidence_work_max(),
                default_formula_evidence_dispatch_max(),
                self.formula_source, self.formula_window, self.enum_lanes_max, self.enum_lane_visits_max, self.enum_dispatch_visits_max, self.allocation, self.identity, self.identity_work_max, self.returned, self.evidence, self.ion_request_work_max, self.formula_evidence_work_max, self.formula_evidence_dispatch_max
            )));
        }
        if !matches!(self.formula_window, 32 | 128 | 512 | 2048) {
            return Err(Error::config(format!(
                "GenerationConfig::validate: formula_window {} is not one of 32, 128, 512, 2048",
                self.formula_window
            )));
        }
        if self.enum_lanes_max == 0 {
            return Err(Error::config(format!(
                "GenerationConfig::validate: enum_lanes_max {} is not non-zero",
                self.enum_lanes_max
            )));
        }
        if self.enum_lane_visits_max == 0 {
            return Err(Error::config(format!(
                "GenerationConfig::validate: enum_lane_visits_max {} is not non-zero",
                self.enum_lane_visits_max
            )));
        }
        if self.enum_dispatch_visits_max == 0 {
            return Err(Error::config(format!(
                "GenerationConfig::validate: enum_dispatch_visits_max {} is not non-zero",
                self.enum_dispatch_visits_max
            )));
        }
        if self.identity_work_max == 0 {
            return Err(Error::config(
                "GenerationConfig::validate: identity_work_max 0 is not non-zero (the per-pair exact-comparison budget of V1 §4.2)".to_string(),
            ));
        }
        let returned = self.effective_returned();
        if !(1..=64).contains(&self.trajectories) {
            return Err(Error::config(format!(
                "GenerationConfig::validate: trajectories {} is not in 1..=64",
                self.trajectories
            )));
        }
        if returned == 0 || returned > self.trajectories {
            return Err(Error::config(format!(
                "GenerationConfig::validate: returned {} (resolved {}) is not in 1..=trajectories {}",
                self.returned, returned, self.trajectories
            )));
        }
        if !(1..=8).contains(&self.formulas) {
            return Err(Error::config(format!(
                "GenerationConfig::validate: formulas {} is not in 1..=8",
                self.formulas
            )));
        }
        if !(self.temperature > 0.0 && self.temperature <= 4.0) {
            return Err(Error::config(format!(
                "GenerationConfig::validate: temperature {} is not in (0, 4]",
                self.temperature
            )));
        }
        let atoms: u32 = max_atoms.try_into().map_err(|_| {
            Error::config(format!(
                "GenerationConfig::validate: max_atoms {max_atoms} does not fit u32"
            ))
        })?;
        let closures: u32 = max_ring_closures.try_into().map_err(|_| {
            Error::config(format!(
                "GenerationConfig::validate: max_ring_closures {max_ring_closures} does not fit u32"
            ))
        })?;
        let min_steps = 2u32
            .checked_add(atoms)
            .and_then(|v| v.checked_add(closures))
            .ok_or_else(|| {
                Error::config(format!(
                    "GenerationConfig::validate: 2 + {max_atoms} + {max_ring_closures} \
                     overflows u32"
                ))
            })?;
        if self.max_steps < min_steps || self.max_steps > 64 {
            return Err(Error::config(format!(
                "GenerationConfig::validate: max_steps {} is not in {min_steps}..=64",
                self.max_steps
            )));
        }
        if matches!(self.mode, GenerationMode::Beam) {
            return Err(Error::Unsupported(
                "GenerationConfig::validate: mode Beam is not implemented (P5)".to_string(),
            ));
        }
        if self.ion_request_work_max == 0 {
            return Err(Error::config(
                "GenerationConfig::validate: ion_request_work_max 0 is not non-zero".to_string(),
            ));
        }
        if self.formula_evidence_work_max == 0 {
            return Err(Error::config(
                "GenerationConfig::validate: formula_evidence_work_max 0 is not non-zero".to_string(),
            ));
        }
        if self.formula_evidence_dispatch_max == 0 {
            return Err(Error::config(
                "GenerationConfig::validate: formula_evidence_dispatch_max 0 is not non-zero".to_string(),
            ));
        }
        // The identity request bound of V1 §4.2 is checked in
        // `generate_preflight` (it needs the batch size); the per-pair budget
        // above is configuration only.
        Ok(())
    }

    /// Packed slots per spectrum (`R`, V1 §4.4): `self.returned` when
    /// non-zero, else the default `min(10, K)`.
    pub fn effective_returned(&self) -> u32 {
        if self.returned == 0 {
            10.min(self.trajectories)
        } else {
            self.returned
        }
    }
}

/// Shared source-specific search-counter rule (contracts §9), used by
/// [`CandidateBatch::validate`] and
/// [`PackedCandidateBatch`](super::pack::PackedCandidateBatch::validate)
/// alike so legal enumeration output validates in every mode.
///
/// `scored <= joined` always; table source requires `joined <= visited` (one
/// join per visited row); enumeration source allows `joined <= 4 * visited`
/// (one visited heavy vector can join up to 4 hydrogen counts — e.g.
/// `visited = 1, joined = 2, scored = 2` is legal). Saturated
/// (`u32::MAX - 1` or `u32::MAX`) counters are lower bounds that must carry
/// `formula_search_exhausted` instead of satisfying the bound. `context`
/// names the caller in errors.
///
/// Completeness rides with the counters (contracts §9: `complete` is 1
/// exactly when the search completed and every joined row was scored): a
/// `complete = 1` claim is rejected when the request carries
/// `formula_search_exhausted` or any counter is saturated, on either source.
pub fn validate_search_counters(
    context: &str,
    spectrum: usize,
    is_enum: bool,
    visited: u32,
    joined: u32,
    scored: u32,
    req_status: u32,
    complete: u8,
) -> Result<()> {
    if scored > joined {
        return Err(Error::config(format!(
            "{context}: spectrum {spectrum} counters break the range \
             rows_scored {scored} <= rows_joined {joined}"
        )));
    }
    if is_enum {
        const SAT: u32 = u32::MAX - 1;
        let saturated =
            visited == SAT || visited == u32::MAX || joined == SAT || joined == u32::MAX;
        if saturated {
            if req_status & request_status::FORMULA_SEARCH_EXHAUSTED == 0 {
                return Err(Error::config(format!(
                    "{context}: spectrum {spectrum} has saturated enumeration counters \
                     (visited {visited}, joined {joined}) without formula_search_exhausted"
                )));
            }
        } else {
            let bound = (visited as u64) * 4;
            if (joined as u64) > bound {
                return Err(Error::config(format!(
                    "{context}: spectrum {spectrum} enumeration counters break the range \
                     rows_joined {joined} <= 4 * rows_visited {visited}"
                )));
            }
        }
    } else if joined > visited {
        return Err(Error::config(format!(
            "{context}: spectrum {spectrum} counters break the range \
             rows_joined {joined} <= rows_visited {visited} (table source)"
        )));
    }
    // Completeness with search status (contracts §9, finding N3): `complete`
    // is 1 exactly for a completed search, so an exhausted search — an
    // explicit EXHAUSTED bit or any saturated (lower-bound) counter — must
    // carry `complete = 0`.
    if complete == 1 {
        const SAT: u32 = u32::MAX - 1;
        let saturated = visited == SAT
            || visited == u32::MAX
            || joined == SAT
            || joined == u32::MAX
            || scored == SAT
            || scored == u32::MAX;
        if req_status & request_status::FORMULA_SEARCH_EXHAUSTED != 0 {
            return Err(Error::config(format!(
                "{context}: spectrum {spectrum} claims formula_support_complete with \
                 formula_search_exhausted (visited {visited}, joined {joined}, scored {scored})"
            )));
        }
        if saturated {
            return Err(Error::config(format!(
                "{context}: spectrum {spectrum} claims formula_support_complete with \
                 saturated counters (visited {visited}, joined {joined}, scored {scored})"
            )));
        }
    }
    Ok(())
}

/// Generated candidates of a batch (contract §3.5): exactly `batch *
/// trajectories` records in `(spectrum, trajectory)` order, read in one
/// batched read. V0 does no compaction: a failed request keeps its records.
///
/// V1 §1.2 adds `formula_counts`, `formula_source` and `formula_rank`. A
/// version-1 document has the three fields absent (empty on load); `validate`
/// accepts empty only for version 1.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct CandidateBatch {
    /// Schema version; 1 or 2 are accepted (see [`CandidateBatch::validate`]).
    pub schema_version: u32,
    /// Spectra per batch.
    pub batch: usize,
    /// Trajectories per spectrum.
    pub trajectories: usize,
    /// Maximum trace steps.
    pub max_steps: usize,
    /// Maximum atoms per candidate (open-valence width).
    pub max_atoms: usize,
    /// Maximum ring closures per candidate (grammar limit).
    pub max_ring_closures: usize,
    /// Provenance per record.
    pub spectrum_id: Vec<u64>,
    /// `0..trajectories` per record.
    pub trajectory: Vec<u32>,
    /// `(kind, atom_type, bond_order, pointer)` per step; steps at or after
    /// `length` are PAD (all zero).
    pub actions: Vec<u32>,
    /// Tokens emitted: START, the actions, and STOP only when finished.
    pub length: Vec<u32>,
    /// Conditioning formula row; [`NO_FORMULA`] when there is none (and
    /// always [`NO_FORMULA`] for an enumerated formula, V1 §1.2).
    pub formula_row: Vec<u32>,
    /// `log p(formula | spectrum)`; no NaN is ever emitted.
    pub formula_log_prob: Vec<f32>,
    /// Summed action log-probabilities up to `length`.
    pub trace_log_prob: Vec<f32>,
    /// Residual valence per atom; meaningful only when finished.
    pub open_valence: Vec<u8>,
    /// `0` unknown, the only V0 value.
    pub attachment_partition: Vec<u8>,
    /// Candidate bits of [`candidate_status`].
    pub status: Vec<u32>,
    /// `0` unassigned, the only V0 value.
    pub evidence_status: Vec<u8>,
    /// `0` trace only (`identity = TraceOnly`, V1 §4.2), `1` exact, `2`
    /// unresolved.
    pub identity_resolution: Vec<u8>,
    /// Request bits of [`request_status`], per spectrum.
    pub request_status: Vec<u32>,
    /// Table rows compared against the mass window, per spectrum.
    pub rows_visited: Vec<u32>,
    /// Rows inside the window, per spectrum.
    pub rows_joined: Vec<u32>,
    /// Rows given a neural score, per spectrum.
    pub rows_scored: Vec<u32>,
    /// `1` when every joined row was scored (contract §9 step 6), else `0`.
    pub formula_support_complete: Vec<u8>,
    /// Probability mass of the retained formulas within the scored window.
    pub formula_mass_retained: Vec<f32>,
    /// Peaks kept after device selection, per spectrum.
    pub peaks_kept: Vec<u32>,
    /// Fraction of filtered intensity the kept peaks hold, per spectrum.
    pub intensity_retained: Vec<f32>,
    /// Composition of the conditioning formula in `ELEMENTS` order
    /// (`u16 [B*K, 10]`); all `0` when there is none. Absent (empty) in
    /// version-1 documents.
    #[serde(default)]
    pub formula_counts: Vec<u16>,
    /// Formula source per spectrum (`u8 [B]`): `0` table, `1` enumeration.
    /// Absent (empty) in version-1 documents.
    #[serde(default)]
    pub formula_source: Vec<u8>,
    /// Rank of the formula in the scored support of its spectrum (its window
    /// slot); `u32::MAX` when there is none. Absent (empty) in version-1
    /// documents.
    #[serde(default)]
    pub formula_rank: Vec<u32>,
    /// Evidence record count per trajectory (`u8 [B*K]`, at most
    /// [`EVIDENCE_CAP`]). Zero when evidence is disabled (the V0 constant).
    #[serde(default)]
    pub evidence_count: Vec<u8>,
    /// Original `peak_id` per evidence record (`u32 [B*K, E]`, `E =
    /// EVIDENCE_CAP`); zero beyond `evidence_count`.
    #[serde(default)]
    pub evidence_peak_id: Vec<u32>,
    /// Hypothesis index among the kept `J` per record (`u8 [B*K, E]`).
    #[serde(default)]
    pub evidence_hypothesis: Vec<u8>,
    /// Hydrogen shift `s` per record (`i8 [B*K, E]`).
    #[serde(default)]
    pub evidence_shift: Vec<i8>,
    /// Signed residual in integer mass units per record (`i32 [B*K, E]`).
    #[serde(default)]
    pub evidence_residual: Vec<i32>,
    /// Assignment log-probability per record (`f32 [B*K, E]`).
    #[serde(default)]
    pub evidence_log_prob: Vec<f32>,
}

/// Per-step dispatch work of one generation call (V1 §3.3).
///
/// See [`CandidateBatch::work`]: step `t` runs over `1..max_steps`, and
/// `active + inactive` is the record count `batch * trajectories` at every
/// step.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StepWork {
    /// The sampling step (`1..max_steps`).
    pub step: usize,
    /// Invocations that ran at this step.
    pub active: usize,
    /// The rest: finished, failed earlier, or never started.
    pub inactive: usize,
}

/// Dispatch work of one generation call, derived on the host from the one
/// final read (V1 §3.3). No device counter is added; fixed dispatch and zero
/// per-step reads are V0 properties the footprint tests keep.
#[derive(Clone, Debug, PartialEq)]
pub struct GenerationWork {
    /// One entry per step `t` in `1..max_steps`.
    pub steps: Vec<StepWork>,
    /// Submitted trajectory-steps: `batch * trajectories * (max_steps - 1)`.
    pub submitted: usize,
    /// Sum of `active` over the steps.
    pub active_total: usize,
    /// `active_total / submitted` (`0` when `submitted` is `0`).
    pub active_fraction: f32,
}

impl CandidateBatch {
    /// The all-PAD batch of a failed request: length 0, `request_failed`
    /// status, `formula_row = NO_FORMULA` on every record.
    pub fn empty(
        spectrum_ids: &[u64],
        trajectories: usize,
        max_steps: usize,
        max_atoms: usize,
        max_ring_closures: usize,
    ) -> Self {
        let n = spectrum_ids.len() * trajectories;
        Self {
            schema_version: SCHEMA_VERSION,
            batch: spectrum_ids.len(),
            trajectories,
            max_steps,
            max_atoms,
            max_ring_closures,
            spectrum_id: spectrum_ids
                .iter()
                .flat_map(|&id| std::iter::repeat_n(id, trajectories))
                .collect(),
            trajectory: (0..n).map(|r| (r % trajectories.max(1)) as u32).collect(),
            actions: vec![0; n * max_steps * 4],
            length: vec![0; n],
            formula_row: vec![NO_FORMULA; n],
            formula_log_prob: vec![0.0; n],
            trace_log_prob: vec![0.0; n],
            open_valence: vec![0; n * max_atoms],
            attachment_partition: vec![0; n],
            status: vec![candidate_status::REQUEST_FAILED; n],
            evidence_status: vec![0; n],
            identity_resolution: vec![0; n],
            request_status: vec![0; spectrum_ids.len()],
            rows_visited: vec![0; spectrum_ids.len()],
            rows_joined: vec![0; spectrum_ids.len()],
            rows_scored: vec![0; spectrum_ids.len()],
            formula_support_complete: vec![0; spectrum_ids.len()],
            formula_mass_retained: vec![0.0; spectrum_ids.len()],
            peaks_kept: vec![0; spectrum_ids.len()],
            intensity_retained: vec![0.0; spectrum_ids.len()],
            formula_counts: vec![0; n * 10],
            formula_source: vec![0; spectrum_ids.len()],
            formula_rank: vec![NO_FORMULA; n],
            evidence_count: vec![0; n],
            evidence_peak_id: vec![0; n * EVIDENCE_CAP],
            evidence_hypothesis: vec![0; n * EVIDENCE_CAP],
            evidence_shift: vec![0; n * EVIDENCE_CAP],
            evidence_residual: vec![0; n * EVIDENCE_CAP],
            evidence_log_prob: vec![0.0; n * EVIDENCE_CAP],
        }
    }

    /// Indices of the records a caller that wants exact-trace duplicates
    /// removed should keep (V0.6): records that are finished, valid and not
    /// `duplicate_trace` — `FINISHED` set, `INVALID_FINAL` and
    /// `REQUEST_FAILED` unset, `DUPLICATE_TRACE` unset. Unresolved graph
    /// duplicates stay visible by contract (`identity_resolution == 0` means
    /// trace only), so they are kept: only exact (trace, formula) repeats
    /// are dropped, and only the later ones.
    pub fn distinct_traces(&self) -> Vec<usize> {
        (0..self.batch * self.trajectories)
            .filter(|&r| {
                let st = self.status[r];
                st & candidate_status::FINISHED != 0
                    && st & candidate_status::INVALID_FINAL == 0
                    && st & candidate_status::REQUEST_FAILED == 0
                    && st & candidate_status::DUPLICATE_TRACE == 0
            })
            .collect()
    }

    /// Per-step dispatch work from the one final read (V1 §3.3, host only,
    /// no device change): for each step `t` in `1..max_steps`, the active
    /// invocations are the started trajectories with `length > t` plus the
    /// ones that failed at this step (`no_valid_action` with `length == t`:
    /// the invocation that detected the failure ran and emitted no token);
    /// the rest (finished, failed earlier, or never started with
    /// `length == 0`) are inactive. `submitted` is the fixed dispatch
    /// `batch * trajectories * (max_steps - 1)` trajectory-steps.
    pub fn work(&self) -> GenerationWork {
        let n = self.batch * self.trajectories;
        let mut steps = Vec::new();
        let mut active_total = 0usize;
        if self.max_steps >= 1 {
            for t in 1..self.max_steps {
                let t_u32 = t as u32;
                let mut active = 0usize;
                for r in 0..n {
                    let len = self.length[r];
                    let st = self.status[r];
                    if len > t_u32 || (st & candidate_status::NO_VALID_ACTION != 0 && len == t_u32)
                    {
                        active += 1;
                    }
                }
                active_total += active;
                steps.push(StepWork {
                    step: t,
                    active,
                    inactive: n - active,
                });
            }
        }
        let submitted = n * self.max_steps.saturating_sub(1);
        let active_fraction = if submitted == 0 {
            0.0
        } else {
            active_total as f32 / submitted as f32
        };
        GenerationWork {
            steps,
            submitted,
            active_total,
            active_fraction,
        }
    }

    /// Check lengths and every invariant of contract §§3.5, 4.4 and 8:
    ///
    /// 1. record order: record `r` has `trajectory == r % K`, the K records
    ///    of a spectrum share one `spectrum_id`, and different spectra have
    ///    different ids;
    /// 2. every emitted token field fits `u8` and the emitted prefix
    ///    (`length` tokens) replays legally with [`replay`] under
    ///    [`Limits::new`]`(max_atoms, max_ring_closures)` and no budget; a
    ///    non-zero unused field is therefore an error;
    /// 3. `finished` exactly when the replayed prefix is stopped; a
    ///    `finished` record's `open_valence[..atoms]` equals the replayed
    ///    residual valences and the rest is 0;
    /// 4. `request_failed` implies `length == 0`; a spectrum whose
    ///    `request_status` has a fatal bit has `request_failed` on all K
    ///    records, and a record with `request_failed` belongs to such a
    ///    spectrum or to one with `rows_scored == 0` (no scored formula,
    ///    contracts §9 abstention);
    /// 5. `attachment_partition` and `evidence_status` are 0 (the only V0
    ///    values); `identity_resolution` is 0 (trace only), 1 (exact) or 2
    ///    (unresolved) — V1 §4.2;
    /// 6. `formula_source_oracle` implies `formula_log_prob == 0`; every
    ///    log-probability is finite and `<= 1e-4`;
    /// 7. `intensity_retained` and `formula_mass_retained` are in
    ///    `[0, 1 + 1e-4]`; `formula_support_complete` is 0 or 1 and, when 1,
    ///    `rows_scored == rows_joined`; `rows_scored <= rows_joined` always;
    ///    table source requires `rows_joined <= rows_visited`, enumeration
    ///    requires `rows_joined <= 4 * rows_visited` (at most 4 hydrogen
    ///    counts per visited heavy vector), with saturated (`u32::MAX - 1`)
    ///    counters accepted as lower bounds carrying
    ///    `formula_search_exhausted` (contracts §9).
    /// 8. V1 §1.2: for version 2, `formula_counts` has `n * 10` entries,
    ///    `formula_source` has `batch` entries of 0 or 1, `formula_rank` has
    ///    `n` entries; `formula_row == u32::MAX` whenever the spectrum's
    ///    `formula_source == 1`; the 10 counts are all zero exactly when
    ///    `formula_rank == u32::MAX` (no formula), and then `formula_row` is
    ///    `u32::MAX` as well; a real `formula_rank` is below that spectrum's
    ///    `rows_scored`, and a table-source record with a formula has a real
    ///    `formula_row` (not `u32::MAX`). No trajectory starts without a
    ///    formula, so a finished record is never formula-less. For version 1
    ///    the three fields must be empty
    ///    (absent); they cannot be reconstructed without the table.
    ///
    /// Steps at or after `length` must still be PAD (all zero); `finished`
    /// and `truncated` stay exclusive, and `truncated` still implies
    /// `length == max_steps` with no STOP token.
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != SCHEMA_VERSION && self.schema_version != SCHEMA_VERSION_V1 {
            return Err(Error::config(format!(
                "CandidateBatch::validate: unknown schema_version {} (expected {SCHEMA_VERSION_V1} or {SCHEMA_VERSION})",
                self.schema_version
            )));
        }
        if self.trajectories == 0 {
            return Err(Error::config(
                "CandidateBatch::validate: trajectories is 0: record order needs at least one trajectory per spectrum"
                    .to_string(),
            ));
        }
        let n = self.batch.checked_mul(self.trajectories).ok_or_else(|| {
            Error::config(format!(
                "CandidateBatch::validate: batch {} times trajectories {} overflows usize",
                self.batch, self.trajectories
            ))
        })?;
        let per_record: [(&str, usize); 10] = [
            ("spectrum_id", self.spectrum_id.len()),
            ("trajectory", self.trajectory.len()),
            ("length", self.length.len()),
            ("formula_row", self.formula_row.len()),
            ("formula_log_prob", self.formula_log_prob.len()),
            ("trace_log_prob", self.trace_log_prob.len()),
            ("attachment_partition", self.attachment_partition.len()),
            ("status", self.status.len()),
            ("evidence_status", self.evidence_status.len()),
            ("identity_resolution", self.identity_resolution.len()),
        ];
        // Ten fields share the record count.
        for (field, len) in per_record {
            if len != n {
                return Err(Error::config(format!(
                    "CandidateBatch::validate: field {field} has length {len} for {n} records"
                )));
            }
        }
        // Evidence count: version 2 carries `n` entries; version 1 predates
        // evidence and carries none (treated as all-zero).
        if self.schema_version == SCHEMA_VERSION_V1 {
            if !self.evidence_count.is_empty() {
                return Err(Error::config(format!(
                    "CandidateBatch::validate: field evidence_count has length {} for version-1 batch (expected empty)",
                    self.evidence_count.len()
                )));
            }
        } else if self.evidence_count.len() != n {
            return Err(Error::config(format!(
                "CandidateBatch::validate: field evidence_count has length {} for {n} records",
                self.evidence_count.len()
            )));
        }
        for (field, len) in [
            ("evidence_peak_id", self.evidence_peak_id.len()),
            ("evidence_hypothesis", self.evidence_hypothesis.len()),
            ("evidence_shift", self.evidence_shift.len()),
            ("evidence_residual", self.evidence_residual.len()),
            ("evidence_log_prob", self.evidence_log_prob.len()),
        ] {
            // Version-1 documents predate evidence: empty is accepted there;
            // version 2 always carries `n * E` entries (zero-padded).
            if self.schema_version == SCHEMA_VERSION_V1 {
                if len != 0 {
                    return Err(Error::config(format!(
                        "CandidateBatch::validate: field {field} has length {len} for version-1 batch (expected empty)"
                    )));
                }
            } else if len != n * EVIDENCE_CAP {
                return Err(Error::config(format!(
                    "CandidateBatch::validate: field {field} has length {len} for {n} records of {EVIDENCE_CAP} evidence slots"
                )));
            }
        }
        if self.actions.len() != n * self.max_steps * 4 {
            return Err(Error::config(format!(
                "CandidateBatch::validate: field actions has length {} for {n} records of {} steps",
                self.actions.len(),
                self.max_steps
            )));
        }
        if self.open_valence.len() != n * self.max_atoms {
            return Err(Error::config(format!(
                "CandidateBatch::validate: field open_valence has length {} for {n} records of {} atoms",
                self.open_valence.len(),
                self.max_atoms
            )));
        }
        let per_spectrum: [(&str, usize); 8] = [
            ("request_status", self.request_status.len()),
            ("rows_visited", self.rows_visited.len()),
            ("rows_joined", self.rows_joined.len()),
            ("rows_scored", self.rows_scored.len()),
            (
                "formula_support_complete",
                self.formula_support_complete.len(),
            ),
            ("formula_mass_retained", self.formula_mass_retained.len()),
            ("peaks_kept", self.peaks_kept.len()),
            ("intensity_retained", self.intensity_retained.len()),
        ];
        for (field, len) in per_spectrum {
            if len != self.batch {
                return Err(Error::config(format!(
                    "CandidateBatch::validate: field {field} has length {len} for {} spectra",
                    self.batch
                )));
            }
        }
        // Rule 8: V1 §1.2 lengths, ranges and the enumeration / empty rules.
        if self.schema_version == SCHEMA_VERSION_V1 {
            if !self.formula_counts.is_empty()
                || !self.formula_source.is_empty()
                || !self.formula_rank.is_empty()
            {
                return Err(Error::config(format!(
                    "CandidateBatch::validate: version-1 batch must have empty formula_counts/source/rank (got {}/{}/{})",
                    self.formula_counts.len(),
                    self.formula_source.len(),
                    self.formula_rank.len()
                )));
            }
        } else {
            if self.formula_counts.len() != n * 10 {
                return Err(Error::config(format!(
                    "CandidateBatch::validate: field formula_counts has length {} for {n} records of 10 counts",
                    self.formula_counts.len()
                )));
            }
            if self.formula_source.len() != self.batch {
                return Err(Error::config(format!(
                    "CandidateBatch::validate: field formula_source has length {} for {} spectra",
                    self.formula_source.len(),
                    self.batch
                )));
            }
            if self.formula_rank.len() != n {
                return Err(Error::config(format!(
                    "CandidateBatch::validate: field formula_rank has length {} for {n} records",
                    self.formula_rank.len()
                )));
            }
            for b in 0..self.batch {
                if self.formula_source[b] > 1 {
                    return Err(Error::config(format!(
                        "CandidateBatch::validate: spectrum {b} formula_source {} is not 0 (table) or 1 (enumeration)",
                        self.formula_source[b]
                    )));
                }
            }
            for r in 0..n {
                let b = if self.trajectories == 0 {
                    0
                } else {
                    r / self.trajectories
                };
                let src = if b < self.formula_source.len() {
                    self.formula_source[b]
                } else {
                    0
                };
                if src == 1 && self.formula_row[r] != NO_FORMULA {
                    return Err(Error::config(format!(
                        "CandidateBatch::validate: record {r} has formula_source 1 (enumeration) but formula_row {} (expected u32::MAX)",
                        self.formula_row[r]
                    )));
                }
                let all_zero = self.formula_counts[r * 10..r * 10 + 10]
                    .iter()
                    .all(|&c| c == 0);
                let rank_none = self.formula_rank[r] == NO_FORMULA;
                if all_zero != rank_none {
                    return Err(Error::config(format!(
                        "CandidateBatch::validate: record {r} counts all-zero {all_zero} disagrees with formula_rank {} (all zero exactly when rank is u32::MAX)",
                        self.formula_rank[r]
                    )));
                }
                if rank_none && self.formula_row[r] != NO_FORMULA {
                    return Err(Error::config(format!(
                        "CandidateBatch::validate: record {r} has no formula (rank u32::MAX) but formula_row {} (expected u32::MAX)",
                        self.formula_row[r]
                    )));
                }
                // Provenance: a real rank is a slot in the spectrum's scored
                // support, so it must lie below that spectrum's
                // `rows_scored`; a table-source hypothesis with a formula
                // must name a real table row (enumeration already requires
                // `u32::MAX` above).
                if !rank_none {
                    let scored = self.rows_scored[b];
                    if self.formula_rank[r] >= scored {
                        return Err(Error::config(format!(
                            "CandidateBatch::validate: record {r} formula_rank {} is not below spectrum {b} rows_scored {scored}",
                            self.formula_rank[r]
                        )));
                    }
                    if src == 0 && self.formula_row[r] == NO_FORMULA {
                        return Err(Error::config(format!(
                            "CandidateBatch::validate: record {r} has table source with formula_rank {} but formula_row u32::MAX (expected a real table row)",
                            self.formula_rank[r]
                        )));
                    }
                }
                // No trajectory starts without a formula, so a finished
                // record is never formula-less: a finished graph with zero
                // counts and MAX row/rank is corrupt, not formula-less.
                if rank_none && self.status[r] & candidate_status::FINISHED != 0 {
                    return Err(Error::config(format!(
                        "CandidateBatch::validate: record {r} is finished but has no formula provenance (rank u32::MAX with zero counts); every finished record needs a real formula rank below rows_scored {} with non-zero counts",
                        self.rows_scored[b]
                    )));
                }
            }
        }
        let limits = Limits::new(self.max_atoms, self.max_ring_closures).map_err(|e| {
            Error::config(format!(
                "CandidateBatch::validate: replay limits from max_atoms {} and max_ring_closures {} rejected: {e}",
                self.max_atoms, self.max_ring_closures
            ))
        })?;
        // Rule 1: (spectrum, trajectory) order and provenance.
        for r in 0..n {
            if self.trajectory[r] != (r % self.trajectories) as u32 {
                return Err(Error::config(format!(
                    "CandidateBatch::validate: record {r} trajectory {} breaks \
                     (spectrum, trajectory) order (expected {})",
                    self.trajectory[r],
                    r % self.trajectories
                )));
            }
        }
        for b in 0..self.batch {
            let first = self.spectrum_id[b * self.trajectories];
            for k in 1..self.trajectories {
                if self.spectrum_id[b * self.trajectories + k] != first {
                    return Err(Error::config(format!(
                        "CandidateBatch::validate: spectrum {b} records do not share one \
                         spectrum_id in (spectrum, trajectory) order"
                    )));
                }
            }
        }
        for a in 0..self.batch {
            for b in (a + 1)..self.batch {
                if self.spectrum_id[a * self.trajectories]
                    == self.spectrum_id[b * self.trajectories]
                {
                    return Err(Error::config(format!(
                        "CandidateBatch::validate: spectra {a} and {b} share spectrum_id {} \
                         against (spectrum, trajectory) order",
                        self.spectrum_id[a * self.trajectories]
                    )));
                }
            }
        }
        for r in 0..n {
            let len = self.length[r] as usize;
            if len > self.max_steps {
                return Err(Error::config(format!(
                    "CandidateBatch::validate: record {r} length {len} exceeds max_steps {}",
                    self.max_steps
                )));
            }
            let base = r * self.max_steps * 4;
            for step in len..self.max_steps {
                if self.actions[base + step * 4..base + step * 4 + 4] != [0, 0, 0, 0] {
                    return Err(Error::config(format!(
                        "CandidateBatch::validate: record {r} step {step} past length {len} is not PAD"
                    )));
                }
            }
            let request_failed = self.status[r] & candidate_status::REQUEST_FAILED != 0;
            // Rule 4: a failed request carries no trace.
            if request_failed && len != 0 {
                return Err(Error::config(format!(
                    "CandidateBatch::validate: record {r} has request_failed with length {len} (expected 0)"
                )));
            }
            // Rule 2: the emitted prefix replays legally; every field fits
            // `u8` first, since the grammar tokens are bytes.
            let mut tokens = Vec::with_capacity(len);
            for step in 0..len {
                let fields = &self.actions[base + step * 4..base + step * 4 + 4];
                let mut bytes = [0u8; 4];
                for (f, v) in fields.iter().enumerate() {
                    if *v > u32::from(u8::MAX) {
                        return Err(Error::config(format!(
                            "CandidateBatch::validate: record {r} step {step} field {f} value {v} \
                             does not fit u8 for the grammar replay"
                        )));
                    }
                    bytes[f] = *v as u8;
                }
                tokens.push(Token {
                    kind: bytes[0],
                    atom_type: bytes[1],
                    bond: bytes[2],
                    pointer: bytes[3],
                });
            }
            let state = replay(&tokens, limits, None).map_err(|e| {
                Error::config(format!(
                    "CandidateBatch::validate: record {r} emitted prefix fails the grammar replay: {e}"
                ))
            })?;
            // Rule 3: `finished` exactly when the replayed prefix is stopped.
            let finished = self.status[r] & candidate_status::FINISHED != 0;
            if finished != state.stopped() {
                return Err(Error::config(format!(
                    "CandidateBatch::validate: record {r} finished bit {finished} disagrees \
                     with the replayed prefix (stopped: {})",
                    state.stopped()
                )));
            }
            if finished {
                let atoms = state.atoms();
                let residual = state.residual_valence();
                let open = &self.open_valence[r * self.max_atoms..(r + 1) * self.max_atoms];
                if open[..atoms] != residual[..] || open[atoms..].iter().any(|&v| v != 0) {
                    return Err(Error::config(format!(
                        "CandidateBatch::validate: record {r} open valence {open:?} does not match \
                         the replayed residual valences {residual:?} with zeros after"
                    )));
                }
            }
            let truncated = self.status[r] & candidate_status::TRUNCATED != 0;
            if finished && truncated {
                return Err(Error::config(format!(
                    "CandidateBatch::validate: record {r} is both finished and truncated"
                )));
            }
            let stop = u32::from(STOP);
            let has_stop = (0..len).any(|step| self.actions[base + step * 4] == stop);
            if truncated {
                if len != self.max_steps {
                    return Err(Error::config(format!(
                        "CandidateBatch::validate: record {r} is truncated but length {len} is not max_steps {}",
                        self.max_steps
                    )));
                }
                if has_stop {
                    return Err(Error::config(format!(
                        "CandidateBatch::validate: record {r} is truncated but holds a STOP token"
                    )));
                }
            }
            // Rule 5: attachment stays 0; evidence_status is 0, 1, 2 with
            // optional bit 7 (incomplete support); identity_resolution 0..=2.
            if self.attachment_partition[r] != 0 {
                return Err(Error::config(format!(
                    "CandidateBatch::validate: record {r} attachment_partition {} is not the V0 constant 0",
                    self.attachment_partition[r]
                )));
            }
            {
                let ev = self.evidence_status[r];
                let base = ev & 0x7F;
                if base != 0 && base != 1 && base != 2 {
                    return Err(Error::config(format!(
                        "CandidateBatch::validate: record {r} evidence_status {ev} has base {base} outside 0, 1, 2 (bit 7 is incomplete support)"
                    )));
                }
                if ev & 0x7F != ev & 0xFF && (ev & !(0x7F | 0x80) != 0) {
                    return Err(Error::config(format!(
                        "CandidateBatch::validate: record {r} evidence_status {ev} carries reserved bits (only bit 7 beyond 0, 1, 2)"
                    )));
                }
                // Evidence pseudo-label invariant: status 0 carries no
                // records; nonzero status carries 1..=E records with zero
                // padding beyond the count.
                let count = if (r as usize) < self.evidence_count.len() {
                    self.evidence_count[r] as usize
                } else {
                    0
                };
                if count > EVIDENCE_CAP {
                    return Err(Error::config(format!(
                        "CandidateBatch::validate: record {r} evidence_count {count} exceeds {EVIDENCE_CAP}"
                    )));
                }
                if base == 0 && count != 0 {
                    return Err(Error::config(format!(
                        "CandidateBatch::validate: record {r} evidence_status {ev} is unassigned but evidence_count {count} is not 0"
                    )));
                }
                if base != 0 && count == 0 {
                    return Err(Error::config(format!(
                        "CandidateBatch::validate: record {r} evidence_status {ev} claims evidence but evidence_count is 0"
                    )));
                }
                for q in 0..EVIDENCE_CAP {
                    let pid = self.evidence_peak_id.get(r * EVIDENCE_CAP + q).copied().unwrap_or(0);
                    let hyp = self.evidence_hypothesis.get(r * EVIDENCE_CAP + q).copied().unwrap_or(0);
                    let lp = self.evidence_log_prob.get(r * EVIDENCE_CAP + q).copied().unwrap_or(0.0);
                    if q >= count {
                        let sh = self.evidence_shift.get(r * EVIDENCE_CAP + q).copied().unwrap_or(0);
                        let rs = self.evidence_residual.get(r * EVIDENCE_CAP + q).copied().unwrap_or(0);
                        if pid != 0 || hyp != 0 || sh != 0 || rs != 0 || lp != 0.0 {
                            return Err(Error::config(format!(
                                "CandidateBatch::validate: record {r} evidence slot {q} beyond count {count} is not zero-padded"
                            )));
                        }
                    } else {
                        if !(lp.is_finite() && lp <= 1e-4) {
                            return Err(Error::config(format!(
                                "CandidateBatch::validate: record {r} evidence slot {q} log_prob {lp} is not a log-probability (finite and <= 1e-4)"
                            )));
                        }
                        let _ = pid;
                        let _ = hyp;
                    }
                }
            }
            if self.identity_resolution[r] > 2 {
                return Err(Error::config(format!(
                    "CandidateBatch::validate: record {r} identity_resolution {} is not in 0..=2 (0 trace only, 1 exact, 2 unresolved)",
                    self.identity_resolution[r]
                )));
            }
            // Rule 6: oracle formulas carry no log-probability; every
            // log-probability is finite and at most 1e-4 above zero.
            let oracle = self.status[r] & candidate_status::FORMULA_SOURCE_ORACLE != 0;
            if oracle && self.formula_log_prob[r] != 0.0 {
                return Err(Error::config(format!(
                    "CandidateBatch::validate: record {r} has formula_source_oracle with \
                     formula_log_prob {} (expected 0 for the oracle)",
                    self.formula_log_prob[r]
                )));
            }
            for (field, value) in [
                ("formula_log_prob", self.formula_log_prob[r]),
                ("trace_log_prob", self.trace_log_prob[r]),
            ] {
                if !(value.is_finite() && value <= 1e-4) {
                    return Err(Error::config(format!(
                        "CandidateBatch::validate: record {r} {field} {value} is outside the \
                         log-probability range (finite and <= 1e-4)"
                    )));
                }
            }
        }
        for b in 0..self.batch {
            // Rule 4: fatal spectra fail on every trajectory, and failed
            // records belong to a fatal spectrum or to a spectrum with no
            // scored formula (`rows_scored == 0`, contracts §9 abstention for
            // exhausted/overflow cases whose request status alone is not
            // fatal).
            let fatal = self.request_status[b] & request_status::FATAL_MASK != 0;
            let no_scored = self.rows_scored[b] == 0;
            for k in 0..self.trajectories {
                let r = b * self.trajectories + k;
                let failed = self.status[r] & candidate_status::REQUEST_FAILED != 0;
                if fatal && !failed {
                    return Err(Error::config(format!(
                        "CandidateBatch::validate: spectrum {b} has fatal request status {} \
                         but record {r} lacks request_failed",
                        self.request_status[b]
                    )));
                }
                if failed && !(fatal || no_scored) {
                    return Err(Error::config(format!(
                        "CandidateBatch::validate: record {r} has request_failed but spectrum {b} \
                         request status {} is not fatal and rows_scored {} is not 0",
                        self.request_status[b], self.rows_scored[b]
                    )));
                }
            }
            // Rule 7: retained fractions, support completeness, counters.
            for (field, value) in [
                ("intensity_retained", self.intensity_retained[b]),
                ("formula_mass_retained", self.formula_mass_retained[b]),
            ] {
                if !(0.0..=1.0 + 1e-4).contains(&value) {
                    return Err(Error::config(format!(
                        "CandidateBatch::validate: spectrum {b} {field} {value} is outside the \
                         retained-fraction range [0, 1 + 1e-4]"
                    )));
                }
            }
            if self.formula_support_complete[b] > 1 {
                return Err(Error::config(format!(
                    "CandidateBatch::validate: spectrum {b} formula_support_complete {} \
                     is not 0 or 1 (range 0..=1)",
                    self.formula_support_complete[b]
                )));
            }
            if self.formula_support_complete[b] == 1 && self.rows_scored[b] != self.rows_joined[b] {
                return Err(Error::config(format!(
                    "CandidateBatch::validate: spectrum {b} formula_support_complete is 1 but \
                     rows_scored {} != rows_joined {} (range: scored must equal joined)",
                    self.rows_scored[b], self.rows_joined[b]
                )));
            }
            // Rule 7 counters, source-specific (contracts §9): the shared
            // [`validate_search_counters`] rule, so packed validation accepts
            // exactly what this validation accepts.
            let is_enum = self.schema_version == SCHEMA_VERSION
                && self.formula_source.len() == self.batch
                && self.formula_source[b] == 1;
            validate_search_counters(
                "CandidateBatch::validate",
                b,
                is_enum,
                self.rows_visited[b],
                self.rows_joined[b],
                self.rows_scored[b],
                self.request_status[b],
                self.formula_support_complete[b],
            )?;
        }
        Ok(())
    }
}

/// Reference to the formula table a checkpoint was built with.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct FormulaTableRef {
    /// Table version string.
    pub version: String,
    /// Rows in the table.
    pub rows: u32,
    /// SHA-256 of the table JSON.
    pub sha256: String,
}

/// Reference to the enumeration artifacts a checkpoint was built with
/// (V1 §1.4): the enum domain and ratio bounds version plus SHA-256.
/// `None` means a table-only config (a version-1 config has none).
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct FormulaArtifactsRef {
    /// Enum domain version string.
    pub domain_version: String,
    /// SHA-256 of the enum domain JSON.
    pub domain_sha256: String,
    /// Ratio bounds version string.
    pub bounds_version: String,
    /// SHA-256 of the ratio bounds JSON.
    pub bounds_sha256: String,
}

/// Model hyperparameters (contract §3.3).
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct ModelConfig {
    /// Schema version; 1 or 2 are accepted. V1 §1.2 moves the config to
    /// version 2; a version-1 document names a formula table only (no
    /// enumeration artifacts, no assignment head).
    pub schema_version: u32,
    /// Config version string.
    pub version: String,
    /// Domain version the weights were trained on.
    pub chemistry: String,
    /// Device peak slots.
    pub n_peaks: u32,
    /// Residual stream width.
    pub d_model: u32,
    /// Peak-mixing SSM.
    pub encoder: SsmConfig,
    /// Trace-decoding SSM.
    pub decoder: SsmConfig,
    /// Encoder blocks.
    pub encoder_blocks: u32,
    /// Decoder blocks.
    pub decoder_blocks: u32,
    /// Cross-attention heads.
    pub attention_heads: u32,
    /// Fourier frequencies per scalar.
    pub fourier_features: u32,
    /// Maximum atoms per candidate.
    pub max_atoms: u32,
    /// Maximum ring closures per candidate.
    pub max_ring_closures: u32,
    /// Formula table the checkpoint binds to.
    pub formula_table: FormulaTableRef,
    /// Enumeration artifacts the checkpoint binds to (V1 §1.4, optional);
    /// a version-1 or table-only config has none.
    #[serde(default)]
    pub formula_artifacts: Option<FormulaArtifactsRef>,
    /// Fragment-ion assignment head (architecture §2). `None` (or a
    /// version-1 document) means assignment disabled: exactly today's
    /// behaviour and results.
    #[serde(default)]
    pub assignment: Option<AssignmentConfig>,
    /// What the formula head ranks with (architecture §1.6): `Counts`
    /// (default: today's 10 features) or `Evidence` (the 16-feature layout
    /// with the additive evidence branch). Absent in a version-1 document,
    /// which takes `Counts`.
    #[serde(default)]
    pub formula_features: FormulaFeatures,
    /// Energy feature scale: the feature is `min(ce, clip) / scale`.
    pub energy_scale_ev: f32,
    /// Energy feature clip in eV.
    pub energy_clip_ev: f32,
    /// Compute dtype.
    pub dtype: DType,
}

impl ModelConfig {
    /// The documented V0 values; both SSMs are [`SsmConfig::default`] with
    /// `d_model` 128, `n_heads` 4, `head_dim` 64, `d_state` 32, `n_groups` 4,
    /// SISO, rotational, learned trapezoid and no convolution.
    pub fn v0() -> Self {
        let ssm = SsmConfig {
            d_model: 128,
            n_heads: 4,
            head_dim: 64,
            d_state: 32,
            n_groups: 4,
            conv_kernel: None,
            ..SsmConfig::default()
        };
        Self {
            schema_version: SCHEMA_VERSION,
            version: "ms2-model-v0".to_string(),
            chemistry: CHEMISTRY_VERSION.to_string(),
            n_peaks: 128,
            d_model: 128,
            encoder: ssm.clone(),
            decoder: ssm,
            encoder_blocks: 2,
            decoder_blocks: 2,
            attention_heads: 4,
            fourier_features: 16,
            max_atoms: 16,
            max_ring_closures: 4,
            formula_table: FormulaTableRef {
                version: "ms2-formula-v0".to_string(),
                // Measured row count of the train-only table (§9); the
                // checkpoint's SHA-256 binds the exact rows on load.
                rows: 37_859,
                sha256: String::new(),
            },
            formula_artifacts: None,
            assignment: None,
            formula_features: FormulaFeatures::Counts,
            energy_scale_ev: 100.0,
            energy_clip_ev: 400.0,
            dtype: DType::F32,
        }
    }

    /// The design's V1 candidate shape restricted to what exists (V1 §3.1):
    /// `max_atoms = 32`, `max_ring_closures = 8`, 4 decoder blocks, the
    /// decoder `d_model`/`d_inner` set apart from the encoder's (decoder 8
    /// heads × 16 channels = 128 inner against the encoder's 4 × 64 = 256,
    /// at the same `d_model` 128); encoder blocks, attention heads, the
    /// peak cap and the formula table as V0.
    ///
    /// For tests only: it is not a default, and `T = 42` belongs to the
    /// [`GenerationConfig`] (`max_steps = 2 + A + R_max`), not here.
    pub fn v1_candidate() -> Self {
        let mut m = Self::v0();
        m.max_atoms = 32;
        m.max_ring_closures = 8;
        m.decoder_blocks = 4;
        m.decoder.n_heads = 8;
        m.decoder.head_dim = 16;
        m
    }

    /// Check the schema version, the chemistry version, both SSMs, the
    /// shared width and the structure limits: the schema version must be 1 or 2,
    /// chemistry must equal [`CHEMISTRY_VERSION`], both
    /// [`SsmConfig::validate`] must pass, `encoder.d_model ==
    /// decoder.d_model == d_model`, `1 <= max_atoms <= 32` (V1 §3.1),
    /// `max_ring_closures <= 8` (V1 §3.1) and `1 <= decoder_blocks <= 4`
    /// (V1 §3.1). The decoder's `n_heads * head_dim` (`d_inner`) is
    /// deliberately unconstrained relative to the encoder's: the two SSMs
    /// only share `d_model`, so a decoder inner width set apart from the
    /// encoder's validates.
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != SCHEMA_VERSION && self.schema_version != SCHEMA_VERSION_V1 {
            return Err(Error::config(format!(
                "ModelConfig::validate: unknown schema_version {} (expected {SCHEMA_VERSION_V1} or {SCHEMA_VERSION})",
                self.schema_version
            )));
        }
        if self.schema_version == SCHEMA_VERSION_V1 && self.formula_artifacts.is_some() {
            return Err(Error::config(
                "ModelConfig::validate: version-1 config must have no formula_artifacts (table only)".to_string(),
            ));
        }
        if self.schema_version == SCHEMA_VERSION_V1 && self.assignment.is_some() {
            return Err(Error::config(
                "ModelConfig::validate: version-1 config must have no assignment (assignment disabled)".to_string(),
            ));
        }
        if let Some(assign) = &self.assignment {
            assign.validate()?;
        }
        if self.chemistry != CHEMISTRY_VERSION {
            return Err(Error::config(format!(
                "ModelConfig::validate: chemistry {:?} does not match {CHEMISTRY_VERSION:?}",
                self.chemistry
            )));
        }
        self.encoder.validate()?;
        self.decoder.validate()?;
        if self.encoder.d_model != self.d_model as usize
            || self.decoder.d_model != self.d_model as usize
        {
            return Err(Error::config(format!(
                "ModelConfig::validate: encoder d_model {} and decoder d_model {} must equal d_model {}",
                self.encoder.d_model, self.decoder.d_model, self.d_model
            )));
        }
        if !(1..=32).contains(&self.max_atoms) {
            return Err(Error::config(format!(
                "ModelConfig::validate: max_atoms {} is not in 1..=32",
                self.max_atoms
            )));
        }
        if self.max_ring_closures > 8 {
            return Err(Error::config(format!(
                "ModelConfig::validate: max_ring_closures {} exceeds 8 (V1 §3.1)",
                self.max_ring_closures
            )));
        }
        if !(1..=4).contains(&self.decoder_blocks) {
            return Err(Error::config(format!(
                "ModelConfig::validate: decoder_blocks {} is not in 1..=4 (V1 §3.1)",
                self.decoder_blocks
            )));
        }
        // Bounded sizes keep every derived width (in_proj_width, d_inner, ...) and
        // every memory-estimate product far inside u64, whatever the platform.
        for (name, ssm) in [("encoder", &self.encoder), ("decoder", &self.decoder)] {
            for (field, value) in [
                ("d_model", ssm.d_model),
                ("n_heads", ssm.n_heads),
                ("head_dim", ssm.head_dim),
                ("d_state", ssm.d_state),
                ("n_groups", ssm.n_groups),
            ] {
                if value > MAX_MODEL_DIMENSION {
                    return Err(Error::config(format!(
                        "ModelConfig::validate: {name}.{field} {value} exceeds {MAX_MODEL_DIMENSION}"
                    )));
                }
            }
        }
        if self.n_peaks as usize > MAX_MODEL_DIMENSION
            || self.attention_heads as usize > MAX_MODEL_DIMENSION
        {
            return Err(Error::config(format!(
                "ModelConfig::validate: n_peaks {} or attention_heads {} exceeds {MAX_MODEL_DIMENSION}",
                self.n_peaks, self.attention_heads
            )));
        }
        Ok(())
    }
}

/// One domain element for checkpoint comparison.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct DomainElement {
    /// Chemical symbol.
    pub symbol: String,
    /// Defining exact mass as a decimal string.
    pub exact: String,
    /// Exact mass in integer units.
    pub mass: u32,
    /// Rounding residual in nano-dalton, rounded up.
    pub residual_nda: u32,
}

/// One atom type for checkpoint comparison.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct DomainAtomType {
    /// Type id, 1–17.
    pub id: u8,
    /// Index into the domain elements.
    pub element: usize,
    /// Parent hydrogen count.
    pub hydrogens: u8,
    /// Valence.
    pub valence: u8,
}

/// One adduct for checkpoint comparison.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct DomainAdduct {
    /// Adduct id.
    pub id: u16,
    /// Adduct name.
    pub name: String,
    /// Hydrogens added to the neutral composition.
    pub hydrogens: i32,
    /// Signed charge.
    pub charge: i32,
}

/// The chemistry domain as checkpoint data (contract §3.2).
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct ChemistryDomain {
    /// Schema version; only [`SCHEMA_VERSION`] is accepted.
    pub schema_version: u32,
    /// Domain version string.
    pub version: String,
    /// Integer mass units per dalton.
    pub mass_scale: u32,
    /// The ten V0 elements in id order.
    pub elements: Vec<DomainElement>,
    /// Electron mass in integer units.
    pub electron_mass: u32,
    /// Electron rounding residual in nano-dalton.
    pub electron_residual_nda: u32,
    /// The 17 V0 atom types in id order.
    pub atom_types: Vec<DomainAtomType>,
    /// Kekulized bond orders.
    pub bond_orders: Vec<u8>,
    /// The V0 adducts.
    pub adducts: Vec<DomainAdduct>,
    /// Largest supported hydrogen shift.
    pub max_hydrogen_shift: u8,
    /// Grammar version string.
    pub grammar: String,
    /// Traversal version string.
    pub traversal: String,
    /// Pseudo-label recipe version string.
    pub recipe: String,
}

impl ChemistryDomain {
    /// The V0 domain, built from the `chem` constants so a checkpoint can be
    /// compared on load.
    pub fn v0() -> Self {
        Self {
            schema_version: SPECTRUM_SCHEMA_VERSION,
            version: CHEMISTRY_VERSION.to_string(),
            mass_scale: chem::MASS_SCALE,
            elements: chem::ELEMENTS
                .iter()
                .map(|e| DomainElement {
                    symbol: e.symbol.to_string(),
                    exact: e.exact.to_string(),
                    mass: e.mass,
                    residual_nda: e.residual_nda,
                })
                .collect(),
            electron_mass: chem::ELECTRON_MASS,
            electron_residual_nda: chem::ELECTRON_RESIDUAL_NDA,
            atom_types: chem::ATOM_TYPES
                .iter()
                .map(|t| DomainAtomType {
                    id: t.id,
                    element: t.element,
                    hydrogens: t.hydrogens,
                    valence: t.valence,
                })
                .collect(),
            bond_orders: vec![1, 2, 3],
            adducts: chem::ADDUCTS
                .iter()
                .map(|a| DomainAdduct {
                    id: a.id,
                    name: a.name.to_string(),
                    hydrogens: a.hydrogens,
                    charge: a.charge,
                })
                .collect(),
            max_hydrogen_shift: 2,
            grammar: GRAMMAR_VERSION.to_string(),
            traversal: TRAVERSAL_VERSION.to_string(),
            recipe: targets::RECIPE_VERSION.to_string(),
        }
    }

    /// Reject a schema version other than [`SPECTRUM_SCHEMA_VERSION`], naming both
    /// versions.
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != SPECTRUM_SCHEMA_VERSION {
            return Err(Error::config(format!(
                "ChemistryDomain::validate: unknown schema_version {} (expected {SPECTRUM_SCHEMA_VERSION})",
                self.schema_version
            )));
        }
        Ok(())
    }

    /// Fail on the first field that differs, naming it.
    pub fn check_compatible(&self, other: &Self) -> Result<()> {
        if self.version != other.version {
            return Err(Error::config(format!(
                "ChemistryDomain::check_compatible: field 'version' differs ({:?} vs {:?})",
                self.version, other.version
            )));
        }
        if self.mass_scale != other.mass_scale {
            return Err(Error::config(format!(
                "ChemistryDomain::check_compatible: field 'mass_scale' differs ({} vs {})",
                self.mass_scale, other.mass_scale
            )));
        }
        if self.elements != other.elements {
            return Err(Error::config(
                "ChemistryDomain::check_compatible: field 'elements' differs".to_string(),
            ));
        }
        if self.electron_mass != other.electron_mass {
            return Err(Error::config(format!(
                "ChemistryDomain::check_compatible: field 'electron_mass' differs ({} vs {})",
                self.electron_mass, other.electron_mass
            )));
        }
        if self.electron_residual_nda != other.electron_residual_nda {
            return Err(Error::config(format!(
                "ChemistryDomain::check_compatible: field 'electron_residual_nda' differs ({} vs {})",
                self.electron_residual_nda, other.electron_residual_nda
            )));
        }
        if self.atom_types != other.atom_types {
            return Err(Error::config(
                "ChemistryDomain::check_compatible: field 'atom_types' differs".to_string(),
            ));
        }
        if self.bond_orders != other.bond_orders {
            return Err(Error::config(
                "ChemistryDomain::check_compatible: field 'bond_orders' differs".to_string(),
            ));
        }
        if self.adducts != other.adducts {
            return Err(Error::config(
                "ChemistryDomain::check_compatible: field 'adducts' differs".to_string(),
            ));
        }
        if self.max_hydrogen_shift != other.max_hydrogen_shift {
            return Err(Error::config(format!(
                "ChemistryDomain::check_compatible: field 'max_hydrogen_shift' differs ({} vs {})",
                self.max_hydrogen_shift, other.max_hydrogen_shift
            )));
        }
        if self.grammar != other.grammar {
            return Err(Error::config(format!(
                "ChemistryDomain::check_compatible: field 'grammar' differs ({:?} vs {:?})",
                self.grammar, other.grammar
            )));
        }
        if self.traversal != other.traversal {
            return Err(Error::config(format!(
                "ChemistryDomain::check_compatible: field 'traversal' differs ({:?} vs {:?})",
                self.traversal, other.traversal
            )));
        }
        if self.recipe != other.recipe {
            return Err(Error::config(format!(
                "ChemistryDomain::check_compatible: field 'recipe' differs ({:?} vs {:?})",
                self.recipe, other.recipe
            )));
        }
        Ok(())
    }
}
