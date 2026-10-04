//! Dataset loader for the files `tools/ms2/export_casmi.py` writes.
//!
//! Pure host Rust: serde types with the export file's JSON field names, the
//! [`ExportSpectrum::peaks`] filter over the stored peak lists, the
//! [`spectrum_batch`] adapter into a contract [`SpectrumBatch`], and the
//! [`percentile`] helper the label report aggregates with.

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

use super::chem::CHEMISTRY_VERSION;
use super::contract::{NO_PEAK, SPECTRUM_SCHEMA_VERSION, SpectrumBatch};
use super::graph::MolGraph;
use super::targets::{Peak, filter_peaks};

/// Schema version the export files (and this reader) use.
pub const EXPORT_SCHEMA_VERSION: u32 = 1;

/// One spectrum of the export file: the integer [`SpectrumBatch`] fields.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct ExportSpectrum {
    /// Source row in the competition data.
    pub row: u64,
    /// Stable spectrum identity.
    pub spectrum_id: u64,
    /// Adduct id in the chemistry domain.
    pub adduct: u16,
    /// `+1` or `-1`.
    pub polarity: i8,
    /// Precursor m/z in integer units.
    pub precursor_mz_udalton: u32,
    /// Precursor uncertainty in integer units.
    pub precursor_uncertainty_udalton: u32,
    /// Peaks the caller held before the `n_raw` pre-selection.
    pub raw_peak_count: u32,
    /// Caller-side peak index per stored peak.
    pub peak_id: Vec<u32>,
    /// m/z in integer units per stored peak.
    pub mz_udalton: Vec<u32>,
    /// Stored intensities per peak.
    pub intensity: Vec<f64>,
    /// Half-width of the stored m/z rounding interval.
    pub mz_uncertainty_udalton: u32,
    /// Mean collision energy in eV (`0` when unknown).
    pub collision_energy_ev: f32,
    /// `1` when the eV value is a measurement.
    pub collision_energy_known: u8,
    /// Collision energies behind the spectrum (`8` means 8 or more).
    pub energy_count: u8,
    /// `0` unknown, `1` timstof, `2` orbitrap, `3` qtof, `4` other.
    pub instrument_class: u8,
}

/// One molecule of the export file: the parent graph and its spectra.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct ExportMolecule {
    /// Stable molecule key.
    pub key: String,
    /// Identity group for the frozen splits.
    pub identity_group: u64,
    /// Identity fold for the frozen splits.
    pub fold_identity: u32,
    /// Atom type id per atom.
    pub atoms: Vec<u8>,
    /// Bonds as `(a, b, order)` triples.
    pub bonds: Vec<(usize, usize, u8)>,
    /// The molecule's spectra.
    pub spectra: Vec<ExportSpectrum>,
}

/// The export file of `tools/ms2/export_casmi.py`.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct ExportFile {
    /// Schema version; only [`EXPORT_SCHEMA_VERSION`] is accepted.
    pub schema_version: u32,
    /// Chemistry domain version the graphs were built with.
    pub chemistry: String,
    /// RDKit version the export ran under.
    pub rdkit: String,
    /// Source data description.
    pub source: String,
    /// Sampling seed.
    pub seed: u64,
    /// Raw peak capacity the peak lists were pre-selected to.
    pub n_raw: u32,
    /// Spectra kept per molecule.
    pub spectra_per_molecule: u32,
    /// Skipped spectra by reason.
    pub skipped_spectra: BTreeMap<String, u64>,
    /// Subset name (`train` or `validation`).
    pub subset: String,
    /// The exported molecules.
    pub molecules: Vec<ExportMolecule>,
}

impl ExportFile {
    /// Parse and validate an export file.
    ///
    /// Rejects a `schema_version` other than 1 and a `chemistry` other than
    /// [`CHEMISTRY_VERSION`] (naming both versions), a spectrum whose
    /// `peak_id`, `mz_udalton` and `intensity` arrays differ in length, and a
    /// spectrum holding more peaks than `n_raw`.
    pub fn from_json(text: &str) -> Result<Self> {
        let file: Self = serde_json::from_str(text)?;
        if file.schema_version != EXPORT_SCHEMA_VERSION {
            return Err(Error::config(format!(
                "ExportFile::from_json: schema_version {} does not match {EXPORT_SCHEMA_VERSION}",
                file.schema_version
            )));
        }
        if file.chemistry != CHEMISTRY_VERSION {
            return Err(Error::config(format!(
                "ExportFile::from_json: chemistry {:?} does not match {CHEMISTRY_VERSION:?}",
                file.chemistry
            )));
        }
        for mol in &file.molecules {
            for s in &mol.spectra {
                if s.peak_id.len() != s.mz_udalton.len() || s.peak_id.len() != s.intensity.len() {
                    return Err(Error::config(format!(
                        "ExportFile::from_json: spectrum row {} arrays differ in length \
                         (peak_id {}, mz_udalton {}, intensity {})",
                        s.row,
                        s.peak_id.len(),
                        s.mz_udalton.len(),
                        s.intensity.len()
                    )));
                }
                if s.peak_id.len() as u32 > file.n_raw {
                    return Err(Error::config(format!(
                        "ExportFile::from_json: spectrum row {} holds {} peaks, \
                         more than n_raw {}",
                        s.row,
                        s.peak_id.len(),
                        file.n_raw
                    )));
                }
            }
        }
        Ok(file)
    }

    /// Read an export file from disk.
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)?;
        Self::from_json(&text)
    }
}

impl ExportMolecule {
    /// Validate the stored atom types and bonds into the labeled graph.
    pub fn graph(&self) -> Result<MolGraph> {
        MolGraph::new(self.atoms.clone(), self.bonds.clone())
    }
}

impl ExportSpectrum {
    /// The contract §2 filter over the stored peak lists, without the cap.
    ///
    /// Errors ([`Error::Config`]) when the peak lists differ in length or an
    /// intensity is invalid; see [`filter_peaks`].
    pub fn peaks(&self) -> Result<Vec<Peak>> {
        filter_peaks(
            &self.peak_id,
            &self.mz_udalton,
            &self.intensity,
            self.precursor_mz_udalton,
        )
    }
}

/// Build a contract §3.1 [`SpectrumBatch`] from export spectra.
///
/// Peak lists are padded to `n_raw` (`peak_id` [`NO_PEAK`], m/z 0, intensity
/// 0); `intensity_scale` and both tolerances are 0 (the defaults). A spectrum
/// with more peaks than `n_raw` is an error, as are arrays that differ in
/// length. The result passes [`SpectrumBatch::validate`].
pub fn spectrum_batch(items: &[&ExportSpectrum], n_raw: u32) -> Result<SpectrumBatch> {
    let n = items.len();
    let width = n_raw as usize;
    let mut batch = SpectrumBatch {
        schema_version: SPECTRUM_SCHEMA_VERSION,
        n_raw,
        spectrum_id: Vec::with_capacity(n),
        raw_peak_count: Vec::with_capacity(n),
        peak_count: Vec::with_capacity(n),
        peak_id: vec![NO_PEAK; n * width],
        mz_udalton: vec![0; n * width],
        intensity: vec![0.0; n * width],
        intensity_scale: 0,
        mz_uncertainty_udalton: Vec::with_capacity(n),
        precursor_mz_udalton: Vec::with_capacity(n),
        precursor_uncertainty_udalton: Vec::with_capacity(n),
        adduct: Vec::with_capacity(n),
        polarity: Vec::with_capacity(n),
        collision_energy_ev: Vec::with_capacity(n),
        collision_energy_known: Vec::with_capacity(n),
        energy_count: Vec::with_capacity(n),
        fragment_tolerance_ppm_tenths: vec![0; n],
        precursor_tolerance_ppm_tenths: vec![0; n],
        instrument_class: Vec::with_capacity(n),
    };
    for (b, s) in items.iter().enumerate() {
        if s.peak_id.len() != s.mz_udalton.len() || s.peak_id.len() != s.intensity.len() {
            return Err(Error::config(format!(
                "spectrum_batch: spectrum row {} arrays differ in length \
                 (peak_id {}, mz_udalton {}, intensity {})",
                s.row,
                s.peak_id.len(),
                s.mz_udalton.len(),
                s.intensity.len()
            )));
        }
        if s.peak_id.len() > width {
            return Err(Error::config(format!(
                "spectrum_batch: spectrum row {} holds {} peaks, more than n_raw {n_raw}",
                s.row,
                s.peak_id.len()
            )));
        }
        batch.spectrum_id.push(s.spectrum_id);
        batch.raw_peak_count.push(s.raw_peak_count);
        batch.peak_count.push(s.peak_id.len() as u32);
        batch.mz_uncertainty_udalton.push(s.mz_uncertainty_udalton);
        batch.precursor_mz_udalton.push(s.precursor_mz_udalton);
        batch
            .precursor_uncertainty_udalton
            .push(s.precursor_uncertainty_udalton);
        batch.adduct.push(s.adduct);
        batch.polarity.push(s.polarity);
        batch.collision_energy_ev.push(s.collision_energy_ev);
        batch.collision_energy_known.push(s.collision_energy_known);
        batch.energy_count.push(s.energy_count);
        batch.instrument_class.push(s.instrument_class);
        let base = b * width;
        for (k, ((id, mz), it)) in s
            .peak_id
            .iter()
            .zip(s.mz_udalton.iter())
            .zip(s.intensity.iter())
            .enumerate()
        {
            batch.peak_id[base + k] = *id;
            batch.mz_udalton[base + k] = *mz;
            batch.intensity[base + k] = *it as f32;
        }
    }
    batch.validate()?;
    Ok(batch)
}

/// The `p`th percentile of a sorted slice, by numpy's default linear
/// interpolation: rank `(n − 1) * p / 100`, linearly between neighbours.
///
/// Returns 0 for an empty slice; a single value returns itself.
pub fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let rank = (sorted.len() - 1) as f64 * p / 100.0;
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let frac = rank - lo as f64;
        sorted[lo] + (sorted[hi] - sorted[lo]) * frac
    }
}
