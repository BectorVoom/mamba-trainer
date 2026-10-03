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
pub const SCHEMA_VERSION: u32 = 1;

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

    /// Names of the set bits, in bit order; undefined bits are skipped.
    pub fn names(bits: u32) -> Vec<&'static str> {
        const TABLE: [(u32, &str); 7] = [
            (FINISHED, "finished"),
            (TRUNCATED, "truncated"),
            (NO_VALID_ACTION, "no_valid_action"),
            (INVALID_FINAL, "invalid_final"),
            (DUPLICATE_TRACE, "duplicate_trace"),
            (FORMULA_SOURCE_ORACLE, "formula_source_oracle"),
            (REQUEST_FAILED, "request_failed"),
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
        if self.schema_version != SCHEMA_VERSION {
            return Err(Error::config(format!(
                "SpectrumBatch::validate: unknown schema_version {} (expected {SCHEMA_VERSION})",
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

/// Generation hyperparameters (contract §3.4).
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct GenerationConfig {
    /// Schema version; only [`SCHEMA_VERSION`] is accepted.
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
}

impl Default for GenerationConfig {
    /// The documented defaults; the visited cap is `u32::MAX` (no limit).
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
        }
    }
}

impl GenerationConfig {
    /// Enforce every documented range under these structure limits.
    ///
    /// The schema version must be [`SCHEMA_VERSION`]; `usize` limits that do
    /// not fit `u32`, or whose sum with the 2-token framing overflows, are
    /// [`Error::Config`] (never a truncation). `Beam` is
    /// [`Error::Unsupported`] until P5 builds it.
    pub fn validate(&self, max_atoms: usize, max_ring_closures: usize) -> Result<()> {
        if self.schema_version != SCHEMA_VERSION {
            return Err(Error::config(format!(
                "GenerationConfig::validate: unknown schema_version {} (expected {SCHEMA_VERSION})",
                self.schema_version
            )));
        }
        if !(1..=64).contains(&self.trajectories) {
            return Err(Error::config(format!(
                "GenerationConfig::validate: trajectories {} is not in 1..=64",
                self.trajectories
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
        Ok(())
    }
}

/// Generated candidates of a batch (contract §3.5): exactly `batch *
/// trajectories` records in `(spectrum, trajectory)` order, read in one
/// batched read. V0 does no compaction: a failed request keeps its records.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct CandidateBatch {
    /// Schema version; only [`SCHEMA_VERSION`] is accepted.
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
    /// Conditioning formula row; [`NO_FORMULA`] when there is none.
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
    /// `0` trace only, the only V0 value.
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
    ///    spectrum;
    /// 5. `attachment_partition`, `evidence_status`, `identity_resolution`
    ///    are 0 (the only V0 values);
    /// 6. `formula_source_oracle` implies `formula_log_prob == 0`; every
    ///    log-probability is finite and `<= 1e-4`;
    /// 7. `intensity_retained` and `formula_mass_retained` are in
    ///    `[0, 1 + 1e-4]`; `formula_support_complete` is 0 or 1 and, when 1,
    ///    `rows_scored == rows_joined`; `rows_scored <= rows_joined <=
    ///    rows_visited`.
    ///
    /// Steps at or after `length` must still be PAD (all zero); `finished`
    /// and `truncated` stay exclusive, and `truncated` still implies
    /// `length == max_steps` with no STOP token.
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != SCHEMA_VERSION {
            return Err(Error::config(format!(
                "CandidateBatch::validate: unknown schema_version {} (expected {SCHEMA_VERSION})",
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
            // Rule 5: the only V0 values are 0.
            for (field, value) in [
                ("attachment_partition", self.attachment_partition[r]),
                ("evidence_status", self.evidence_status[r]),
                ("identity_resolution", self.identity_resolution[r]),
            ] {
                if value != 0 {
                    return Err(Error::config(format!(
                        "CandidateBatch::validate: record {r} {field} {value} is not the V0 constant 0"
                    )));
                }
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
            // records belong to fatal spectra.
            let fatal = self.request_status[b] & request_status::FATAL_MASK != 0;
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
                if failed && !fatal {
                    return Err(Error::config(format!(
                        "CandidateBatch::validate: record {r} has request_failed but spectrum {b} \
                         request status {} is not fatal",
                        self.request_status[b]
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
            if self.rows_scored[b] > self.rows_joined[b]
                || self.rows_joined[b] > self.rows_visited[b]
            {
                return Err(Error::config(format!(
                    "CandidateBatch::validate: spectrum {b} counters break the range \
                     rows_scored {} <= rows_joined {} <= rows_visited {}",
                    self.rows_scored[b], self.rows_joined[b], self.rows_visited[b]
                )));
            }
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

/// Model hyperparameters (contract §3.3).
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct ModelConfig {
    /// Schema version; only [`SCHEMA_VERSION`] is accepted.
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
            energy_scale_ev: 100.0,
            energy_clip_ev: 400.0,
            dtype: DType::F32,
        }
    }

    /// Check the schema version, the chemistry version, both SSMs, the
    /// shared width and the atom limit: the schema version must equal
    /// [`SCHEMA_VERSION`], chemistry must equal [`CHEMISTRY_VERSION`], both
    /// [`SsmConfig::validate`] must pass, `encoder.d_model ==
    /// decoder.d_model == d_model`, and `1 <= max_atoms <= 32`.
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != SCHEMA_VERSION {
            return Err(Error::config(format!(
                "ModelConfig::validate: unknown schema_version {} (expected {SCHEMA_VERSION})",
                self.schema_version
            )));
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
            schema_version: SCHEMA_VERSION,
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

    /// Reject a schema version other than [`SCHEMA_VERSION`], naming both
    /// versions.
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != SCHEMA_VERSION {
            return Err(Error::config(format!(
                "ChemistryDomain::validate: unknown schema_version {} (expected {SCHEMA_VERSION})",
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
