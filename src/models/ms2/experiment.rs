//! Experiment datasets for the V0 experiments (architecture §7).
//!
//! [`ExperimentSet`] loads an `export_casmi.py` file into per-spectrum
//! parents, peaks and pseudo-labels, with an explicit per-spectrum domain.
//! The per-spectrum label step is [`label_export_spectrum`], which is also
//! what `examples/ms2_label_report.rs` calls, so the report and the training
//! path share one recipe implementation.

use std::path::Path;

use crate::error::{Error, Result};

use super::chem::{Composition, adduct};
use super::contract::{PRECURSOR_MAX, PRECURSOR_MIN, SpectrumBatch};
use super::dataset::{ExportSpectrum, spectrum_batch};
use super::formula_evidence_ref::jitter_precursor_mz;
use super::grammar::Limits;
use super::graph::MolGraph;
use super::targets::{Candidates, Labels, RecipeLimits};
use super::targets_batch::TargetBatch;

/// Tolerance of the frozen recipe, in tenths of a ppm (contract §6).
pub const LABEL_PPM_TENTHS: u32 = 100;

/// Build the pseudo-labels of one export spectrum against its molecule's
/// cached [`Candidates`], exactly as the label report does: the contract §2
/// filter over the stored peaks, then the recipe match at
/// [`LABEL_PPM_TENTHS`] with the spectrum's own adduct and uncertainty.
pub fn label_export_spectrum(candidates: &Candidates, spectrum: &ExportSpectrum) -> Result<Labels> {
    let peaks = spectrum.peaks()?;
    candidates.label(
        &peaks,
        spectrum.adduct,
        LABEL_PPM_TENTHS,
        spectrum.mz_uncertainty_udalton,
    )
}

/// Per-spectrum domain (contracts §4.6 and §7.1).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SpectrumDomain {
    /// In the V0 request and structure domain, with at least one target.
    InDomainLabeled,
    /// In domain, but with no target (no graph loss; still a coverage miss).
    InDomainUnlabeled,
    /// Outside the V0 domain; the string names the first reason in the
    /// contract order (adduct, polarity, structure, precursor range).
    OutOfDomain(String),
}

/// One spectrum of an [`ExperimentSet`]: the export row plus its parent and
/// labels.
#[derive(Clone, Debug)]
pub struct ExperimentSpectrum {
    /// Index into [`ExperimentSet::molecules`].
    pub molecule: usize,
    /// The export row (peaks, precursor, adduct, metadata).
    pub spectrum: ExportSpectrum,
    /// The parent graph (empty when the molecule itself is out of domain;
    /// out-of-domain spectra never train).
    pub parent: MolGraph,
    /// Element composition of [`ExperimentSpectrum::parent`].
    pub parent_composition: Composition,
    /// The kept targets (`None` when out of domain or with no target).
    pub labels: Option<Labels>,
    /// The domain of this spectrum.
    pub domain: SpectrumDomain,
}

/// A loaded experiment dataset: molecules plus their spectra in file order.
#[derive(Clone, Debug)]
pub struct ExperimentSet {
    /// File name of the source export file.
    pub name: String,
    /// SHA-256 (hex) of the source file bytes.
    pub source_sha256: String,
    /// Molecule keys in export order.
    pub molecules: Vec<String>,
    /// Spectra in file order (molecule order, spectrum order within).
    pub spectra: Vec<ExperimentSpectrum>,
}

/// SHA-256 (hex) of the sorted, newline-joined molecule keys of `molecules`
/// (plan P7.8 training provenance: the molecule-keys hash stored beside the
/// export's byte hash and count).
pub fn molecule_keys_sha256(molecules: &[String]) -> String {
    let mut sorted: Vec<&str> = molecules.iter().map(String::as_str).collect();
    sorted.sort_unstable();
    let joined = sorted.join("\n");
    sha256_hex(joined.as_bytes())
}

impl ExperimentSet {
    /// Molecule-aware donors for the `ShuffledSpectrum` control (contracts
    /// §10: another spectrum's peaks, own metadata and targets).
    ///
    /// For every spectrum `i` of the set, the donor is a spectrum index of a
    /// *different* molecule drawn uniformly (seeded) from the whole set:
    /// rejection sampling over `SplitMix64(seed)` in spectrum order, so the
    /// same seed gives the same donors on any platform. Deterministic.
    ///
    /// Errors ([`Error::Config`]) when the set holds fewer than two
    /// molecules: with one molecule there is no different-molecule donor.
    pub fn donor_map(&self, seed: u64) -> Result<Vec<usize>> {
        if self.molecules.len() < 2 {
            return Err(Error::config(format!(
                "ExperimentSet::donor_map: {} molecule(s), needs at least 2 for a different-molecule donor",
                self.molecules.len()
            )));
        }
        let n = self.spectra.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        let mut rng = SplitMix64::new(seed);
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let mol = self.spectra[i].molecule;
            // Rejection sampling from the uniform index: each accepted draw
            // is uniform over the eligible (different-molecule) spectra.
            // The loop terminates because at least one other molecule has at
            // least one spectrum.
            loop {
                let j = (rng.next() % (n as u64)) as usize;
                if self.spectra[j].molecule != mol {
                    out.push(j);
                    break;
                }
            }
        }
        Ok(out)
    }
    /// Load an export file, building parents and pseudo-labels.
    ///
    /// One [`Candidates`] preparation runs per molecule and is reused for
    /// every spectrum of that molecule; each spectrum is labeled with
    /// [`label_export_spectrum`] under `limits`. Out-of-domain spectra get
    /// `labels = None` and stay in the set: they count as misses in the
    /// full-dataset denominator (contracts §7.1), never as training targets.
    pub fn load(path: &Path, limits: &RecipeLimits) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        let source_sha256 = sha256_hex(&bytes);
        let text = String::from_utf8(bytes).map_err(|e| {
            Error::config(format!(
                "ExperimentSet::load: {} is not UTF-8: {e}",
                path.display()
            ))
        })?;
        let file = super::dataset::ExportFile::from_json(&text)?;
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.display().to_string());
        let mut molecules: Vec<String> = Vec::with_capacity(file.molecules.len());
        let mut spectra: Vec<ExperimentSpectrum> = Vec::new();
        for mol in &file.molecules {
            let mol_idx = molecules.len();
            molecules.push(mol.key.clone());
            // Structure domain first: the graph, then the cached candidates.
            // Either failing marks every spectrum of the molecule out of
            // domain with a `structure:` reason; the stored parent is then
            // the empty graph, which never trains.
            let graph = mol.graph();
            // A candidates failure is a structure problem (e.g. a
            // disconnected export graph, whose canonicalization is
            // undefined): keep it out of domain rather than failing the
            // whole load.
            let candidates: Option<Candidates> = match &graph {
                Ok(g) => Candidates::new(g, limits).ok(),
                Err(_) => None,
            };
            let structure_reason: Option<String> = match &graph {
                Ok(_) if candidates.is_some() => None,
                Ok(_) => Some("structure:candidates".to_string()),
                Err(e) => Some(format!("structure:{e}")),
            };
            for s in &mol.spectra {
                let domain: SpectrumDomain;
                let labels: Option<Labels>;
                let (parent, parent_composition) = match &graph {
                    Ok(g) => (
                        MolGraph::new(g.atoms().to_vec(), g.bonds().to_vec()).map_err(|e| {
                            Error::config(format!(
                                "ExperimentSet::load: molecule {} rebuild: {e}",
                                mol.key
                            ))
                        })?,
                        g.composition(),
                    ),
                    Err(_) => (
                        MolGraph::new(Vec::new(), Vec::new()).expect("empty graph builds"),
                        [0; 10],
                    ),
                };
                if let Some(reason) = request_reason(s) {
                    domain = SpectrumDomain::OutOfDomain(reason);
                    labels = None;
                } else if let Some(reason) = structure_reason.clone() {
                    domain = SpectrumDomain::OutOfDomain(reason);
                    labels = None;
                } else if precursor_reason(s).is_some() {
                    domain =
                        SpectrumDomain::OutOfDomain(precursor_reason(s).expect("checked above"));
                    labels = None;
                } else {
                    let cands = candidates.as_ref().expect("in-domain has candidates");
                    let built = label_export_spectrum(cands, s)?;
                    if built.targets.is_empty() {
                        domain = SpectrumDomain::InDomainUnlabeled;
                        labels = None;
                    } else {
                        domain = SpectrumDomain::InDomainLabeled;
                        labels = Some(built);
                    }
                }
                spectra.push(ExperimentSpectrum {
                    molecule: mol_idx,
                    spectrum: s.clone(),
                    parent,
                    parent_composition,
                    labels,
                    domain,
                });
            }
        }
        Ok(Self {
            name,
            source_sha256,
            molecules,
            spectra,
        })
    }

    /// Indices of the labeled spectra, in file order.
    pub fn labeled(&self) -> Vec<usize> {
        self.spectra
            .iter()
            .enumerate()
            .filter(|(_, s)| s.domain == SpectrumDomain::InDomainLabeled)
            .map(|(i, _)| i)
            .collect()
    }

    /// The first `n` labeled spectra in file order, one per molecule.
    ///
    /// This is the overfit fixture of architecture §7: 128 labeled spectra of
    /// 128 molecules. Molecule indices are remapped to the new molecule list;
    /// the name records the source and `n`.
    pub fn take_labeled(&self, n: usize) -> Result<Self> {
        let mut picked: Vec<usize> = Vec::new();
        let mut seen_mol: Vec<bool> = vec![false; self.molecules.len()];
        for i in 0..self.spectra.len() {
            if self.spectra[i].domain != SpectrumDomain::InDomainLabeled {
                continue;
            }
            let m = self.spectra[i].molecule;
            if seen_mol[m] {
                continue;
            }
            seen_mol[m] = true;
            picked.push(i);
            if picked.len() == n {
                break;
            }
        }
        if picked.len() < n {
            return Err(Error::config(format!(
                "ExperimentSet::take_labeled: only {} labeled molecules, asked for {n}",
                picked.len()
            )));
        }
        let mut molecules: Vec<String> = Vec::with_capacity(n);
        let mut index_of: Vec<Option<usize>> = vec![None; self.molecules.len()];
        for &i in &picked {
            let m = self.spectra[i].molecule;
            if index_of[m].is_none() {
                index_of[m] = Some(molecules.len());
                molecules.push(self.molecules[m].clone());
            }
        }
        let mut spectra: Vec<ExperimentSpectrum> = Vec::with_capacity(n);
        for &i in &picked {
            let s = &self.spectra[i];
            let new_mol = index_of[s.molecule].expect("picked molecules are indexed");
            let parent = MolGraph::new(s.parent.atoms().to_vec(), s.parent.bonds().to_vec())
                .map_err(|e| Error::config(format!("take_labeled rebuild: {e}")))?;
            spectra.push(ExperimentSpectrum {
                molecule: new_mol,
                spectrum: s.spectrum.clone(),
                parent,
                parent_composition: s.parent_composition,
                labels: s.labels.clone(),
                domain: s.domain.clone(),
            });
        }
        Ok(Self {
            name: format!("{}:take_labeled({n})", self.name),
            source_sha256: self.source_sha256.clone(),
            molecules,
            spectra,
        })
    }

    /// Partition `indices` into batches of `batch` spectra.
    ///
    /// The indices are shuffled deterministically by `epoch_seed`
    /// (SplitMix64 Fisher-Yates, so the same seed gives the same order on
    /// any platform); the last partial batch is kept.
    pub fn batches(&self, indices: &[usize], batch: usize, epoch_seed: u64) -> Vec<Vec<usize>> {
        if indices.is_empty() || batch == 0 {
            return Vec::new();
        }
        let mut order: Vec<usize> = indices.to_vec();
        let mut rng = SplitMix64::new(epoch_seed);
        for i in (1..order.len()).rev() {
            let j = (rng.next() % (i as u64 + 1)) as usize;
            order.swap(i, j);
        }
        order.chunks(batch).map(|c| c.to_vec()).collect()
    }
}

/// Allow-listed provenance of one export file for the experiment report.
///
/// The report must stay aggregates-only: export files carry per-molecule
/// rows (`molecules` with SMILES, spectrum ids and peaks), which are
/// CC BY-NC data that must not be stored in the repository. This copies
/// only scalar provenance fields (`schema_version`, `chemistry`, `rdkit`,
/// `source`, `seed`, `n_raw`, `spectra_per_molecule`, `spectrum_sampling`,
/// `skipped_spectra`, `subset`) plus the file name and its `source_sha256`;
/// anything else in `raw` — in particular `molecules` and any per-spectrum
/// key (`spectra`, `spectrum_id`, `smiles`, `mz_udalton`) — is dropped.
/// Missing allow-list keys become `null` so old and new exports share the
/// shape. See `examples/ms2_experiment.rs`, the only writer.
pub fn export_provenance(
    raw: &serde_json::Value,
    file_name: &str,
    source_sha256: &str,
) -> serde_json::Value {
    const ALLOW: [&str; 10] = [
        "schema_version",
        "chemistry",
        "rdkit",
        "source",
        "seed",
        "n_raw",
        "spectra_per_molecule",
        "spectrum_sampling",
        "skipped_spectra",
        "subset",
    ];
    let mut out = serde_json::Map::with_capacity(ALLOW.len() + 2);
    out.insert(
        "file".to_string(),
        serde_json::Value::String(file_name.to_string()),
    );
    out.insert(
        "source_sha256".to_string(),
        serde_json::Value::String(source_sha256.to_string()),
    );
    for key in ALLOW {
        out.insert(
            key.to_string(),
            raw.get(key).cloned().unwrap_or(serde_json::Value::Null),
        );
    }
    serde_json::Value::Object(out)
}

/// Build a contract [`SpectrumBatch`] over the chosen spectra.
///
/// Spectrum ids are the export's `spectrum_id`, so generation provenance
/// keys back to the source rows.
pub fn spectrum_batch_for(
    set: &ExperimentSet,
    indices: &[usize],
    n_raw: u32,
) -> Result<SpectrumBatch> {
    let mut items: Vec<&ExportSpectrum> = Vec::with_capacity(indices.len());
    for &i in indices {
        let s = set.spectra.get(i).ok_or_else(|| {
            Error::config(format!(
                "spectrum_batch_for: spectrum index {i} outside {} spectra",
                set.spectra.len()
            ))
        })?;
        items.push(&s.spectrum);
    }
    spectrum_batch(&items, n_raw)
}

/// Build a host [`TargetBatch`] over the chosen spectra.
///
/// Labels come from the set (`None` for out-of-domain or unlabeled spectra,
/// which contribute empty slots); every training budget is the true parent
/// composition (architecture §4.4).
pub fn target_batch_for(
    set: &ExperimentSet,
    indices: &[usize],
    slots: usize,
    limits: Limits,
) -> Result<TargetBatch> {
    let mut label_refs: Vec<Option<&Labels>> = Vec::with_capacity(indices.len());
    let mut parents: Vec<Composition> = Vec::with_capacity(indices.len());
    for &i in indices {
        let s = set.spectra.get(i).ok_or_else(|| {
            Error::config(format!(
                "target_batch_for: spectrum index {i} outside {} spectra",
                set.spectra.len()
            ))
        })?;
        label_refs.push(s.labels.as_ref());
        parents.push(s.parent_composition);
    }
    TargetBatch::build(&label_refs, &parents, slots, limits)
}

/// Build a contract [`SpectrumBatch`] with molecule-aware donor peaks
/// (contracts §10: another spectrum's peaks, own metadata and targets).
///
/// `indices` and `donors` run in parallel: row `b` carries the peak fields
/// (`peak_count`, `raw_peak_count`, `peak_id`, `mz_udalton`, `intensity`,
/// `mz_uncertainty_udalton`) of donor spectrum `donors[b]` and every other
/// field (identity, adduct, polarity, precursor, energies, tolerances,
/// instrument) of recipient `indices[b]`. Targets stay the recipient's: use
/// [`target_batch_for`] with `indices` as usual.
///
/// Donor peaks above the recipient's precursor + 2 Da are dropped by the
/// existing peak filter as for any request (the device eligibility
/// `0 < mz <= precursor + 2_000_000`); when that leaves a recipient with no
/// eligible peak the row stays empty (peak lists as donated, selection finds
/// nothing) and counts in [`donor_stats`] — this never falls back to the
/// recipient's own peaks.
pub fn spectrum_batch_with_donors(
    set: &ExperimentSet,
    indices: &[usize],
    donors: &[usize],
    n_raw: u32,
) -> Result<SpectrumBatch> {
    use super::contract::{NO_PEAK, SPECTRUM_SCHEMA_VERSION};
    if indices.len() != donors.len() {
        return Err(Error::config(format!(
            "spectrum_batch_with_donors: {} indices for {} donors (must match)",
            indices.len(),
            donors.len()
        )));
    }
    let n = indices.len();
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
    for (b, (&ri, &di)) in indices.iter().zip(donors.iter()).enumerate() {
        let recipient = set.spectra.get(ri).ok_or_else(|| {
            Error::config(format!(
                "spectrum_batch_with_donors: recipient index {ri} outside {} spectra",
                set.spectra.len()
            ))
        })?;
        let donor = set.spectra.get(di).ok_or_else(|| {
            Error::config(format!(
                "spectrum_batch_with_donors: donor index {di} outside {} spectra",
                set.spectra.len()
            ))
        })?;
        let r = &recipient.spectrum;
        let d = &donor.spectrum;
        if d.peak_id.len() != d.mz_udalton.len() || d.peak_id.len() != d.intensity.len() {
            return Err(Error::config(format!(
                "spectrum_batch_with_donors: donor row {di} arrays differ in length \
                 (peak_id {}, mz_udalton {}, intensity {})",
                d.peak_id.len(),
                d.mz_udalton.len(),
                d.intensity.len()
            )));
        }
        if d.peak_id.len() > width {
            return Err(Error::config(format!(
                "spectrum_batch_with_donors: donor row {di} holds {} peaks, more than n_raw {n_raw}",
                d.peak_id.len()
            )));
        }
        batch.spectrum_id.push(r.spectrum_id);
        batch.raw_peak_count.push(d.raw_peak_count);
        batch.peak_count.push(d.peak_id.len() as u32);
        batch.mz_uncertainty_udalton.push(d.mz_uncertainty_udalton);
        batch.precursor_mz_udalton.push(r.precursor_mz_udalton);
        batch
            .precursor_uncertainty_udalton
            .push(r.precursor_uncertainty_udalton);
        batch.adduct.push(r.adduct);
        batch.polarity.push(r.polarity);
        batch.collision_energy_ev.push(r.collision_energy_ev);
        batch.collision_energy_known.push(r.collision_energy_known);
        batch.energy_count.push(r.energy_count);
        batch.instrument_class.push(r.instrument_class);
        let base = b * width;
        for (k, ((id, mz), it)) in d
            .peak_id
            .iter()
            .zip(d.mz_udalton.iter())
            .zip(d.intensity.iter())
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

/// Apply precursor jitter to an assembled batch in place (architecture
/// §1.6).
///
/// Each row's precursor m/z is replaced by
/// [`jitter_precursor_mz`](super::formula_evidence_ref::jitter_precursor_mz)
/// with `spectrum_index = indices[b]` (the spectrum's index in its
/// `ExperimentSet`, not its position in the batch), so the draw for a
/// spectrum does not depend on batch composition. Keying: training batches
/// use `split_tag = 1 + step` (seed, step, spectrum), evaluation batches a
/// fixed `split_tag = 0` (seed, spectrum). `sigma_ppm <= 0` leaves every
/// byte of the batch unchanged. The gold composition and the targets are
/// not touched.
pub fn apply_precursor_jitter(
    batch: &mut SpectrumBatch,
    indices: &[usize],
    sigma_ppm: f32,
    seed: u64,
    split_tag: u64,
) {
    if sigma_ppm <= 0.0 {
        return;
    }
    debug_assert!(sigma_ppm.is_finite());
    for (b, &idx) in indices.iter().enumerate() {
        if b >= batch.precursor_mz_udalton.len() {
            break;
        }
        let mz = batch.precursor_mz_udalton[b];
        batch.precursor_mz_udalton[b] =
            jitter_precursor_mz(mz, f64::from(sigma_ppm), seed, split_tag, idx as u64);
    }
}

/// Build a contract [`SpectrumBatch`] over the chosen spectra with
/// precursor jitter (architecture §1.6).
///
/// As [`spectrum_batch_for`], then [`apply_precursor_jitter`] with
/// `spectrum_index = indices[b]`, so evaluation draws do not depend on
/// batch composition. Training uses `split_tag = 1 + step`, evaluation a
/// fixed `split_tag = 0`; see [`apply_precursor_jitter`] for the keying.
/// `sigma_ppm = 0` returns the unjittered batch byte-for-byte.
pub fn spectrum_batch_for_with_jitter(
    set: &ExperimentSet,
    indices: &[usize],
    n_raw: u32,
    sigma_ppm: f32,
    seed: u64,
    split_tag: u64,
) -> Result<SpectrumBatch> {
    let mut batch = spectrum_batch_for(set, indices, n_raw)?;
    apply_precursor_jitter(&mut batch, indices, sigma_ppm, seed, split_tag);
    Ok(batch)
}

/// Build a contract [`SpectrumBatch`] with molecule-aware donor peaks and
/// precursor jitter (architecture §1.6).
///
/// As [`spectrum_batch_with_donors`], then [`apply_precursor_jitter`] keyed
/// by the recipient indices (the precursor is the recipient's: donor peaks
/// never move the precursor, so the evidence peaks follow the donor's peaks
/// while the residual stays the recipient's). See
/// [`apply_precursor_jitter`] for the keying. `sigma_ppm = 0` returns the
/// unjittered batch byte-for-byte.
pub fn spectrum_batch_with_donors_with_jitter(
    set: &ExperimentSet,
    indices: &[usize],
    donors: &[usize],
    n_raw: u32,
    sigma_ppm: f32,
    seed: u64,
    split_tag: u64,
) -> Result<SpectrumBatch> {
    let mut batch = spectrum_batch_with_donors(set, indices, donors, n_raw)?;
    apply_precursor_jitter(&mut batch, indices, sigma_ppm, seed, split_tag);
    Ok(batch)
}

/// Fixed jitter-draw pool of one batch (architecture §1.6, task F8 item 7):
/// the shared production pool-construction function the driver and the cache
/// precompute pass use. Yields `variants` batches LAZILY (task F9 item A3):
/// one variant's batch is alive at a time — each [`Iterator::next`] clones
/// the base batch once, applies [`apply_precursor_jitter`] at `split_tag = 1
/// + v` for `v in 0..variants`, and hands ownership to the caller, so a
/// `for variant in ...` driver loop drops each variant before the next is
/// materialised (temporary memory never grows with `V`). `sigma_ppm <= 0`
/// still clones without touching a byte; `variants == 0` yields nothing
/// (callers then use the batch itself with fresh per-step draws).
///
/// The yielded variant `v` is exactly the draw training step `s` selects via
/// [`jitter_variant_index`] (`split_tag = 1 + v`).
pub fn jitter_variants_of_batch(
    batch: &SpectrumBatch,
    indices: &[usize],
    sigma_ppm: f32,
    seed: u64,
    variants: u32,
) -> JitterVariants {
    JitterVariants {
        base: batch.clone(),
        indices: indices.to_vec(),
        sigma_ppm,
        seed,
        next_variant: 0,
        total: variants,
    }
}

/// Lazy fixed jitter-draw pool yielded by [`jitter_variants_of_batch`]
/// (task F9 item A3): owns one base-batch clone plus the draw parameters and
/// materialises one variant per [`Iterator::next`], so at most one variant's
/// batch is alive at a time beyond the shared base.
pub struct JitterVariants {
    /// The unjittered batch every variant clones.
    base: SpectrumBatch,
    /// Spectrum indices keying the jitter draws.
    indices: Vec<usize>,
    /// Jitter width in ppm.
    sigma_ppm: f32,
    /// Seed keying the jitter draws.
    seed: u64,
    /// Next variant draw to yield.
    next_variant: u32,
    /// Total variant draws to yield.
    total: u32,
}

impl Iterator for JitterVariants {
    type Item = SpectrumBatch;

    /// Materialise the next variant draw (one base clone plus the jitter
    /// application), or `None` once all `total` draws are yielded.
    fn next(&mut self) -> Option<SpectrumBatch> {
        if self.next_variant >= self.total {
            return None;
        }
        let v = self.next_variant;
        self.next_variant += 1;
        let mut jittered = self.base.clone();
        apply_precursor_jitter(
            &mut jittered,
            &self.indices,
            self.sigma_ppm,
            self.seed,
            1 + u64::from(v),
        );
        Some(jittered)
    }

    /// Exact remaining draws, so `len()` never materialises a variant.
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.total.saturating_sub(self.next_variant) as usize;
        (n, Some(n))
    }
}

impl ExactSizeIterator for JitterVariants {
    /// Remaining variant draws (no variant materialised).
    fn len(&self) -> usize {
        self.total.saturating_sub(self.next_variant) as usize
    }
}

/// A copy of `set` with every spectrum's precursor m/z jittered for
/// evaluation (architecture §1.6).
///
/// Each spectrum `i` gets
/// [`jitter_precursor_mz`](super::formula_evidence_ref::jitter_precursor_mz)
/// with `split_tag = 0` and `spectrum_index = i`, so the draw is fixed per
/// `(seed, spectrum)` and independent of batch composition. The peaks, the
/// gold composition and the targets are not touched. `sigma_ppm = 0`
/// returns an identical set.
pub fn jittered_set_for_eval(set: &ExperimentSet, sigma_ppm: f32, seed: u64) -> ExperimentSet {
    if sigma_ppm <= 0.0 {
        return set.clone();
    }
    let mut out = set.clone();
    for (i, s) in out.spectra.iter_mut().enumerate() {
        let mz = s.spectrum.precursor_mz_udalton;
        s.spectrum.precursor_mz_udalton =
            jitter_precursor_mz(mz, f64::from(sigma_ppm), seed, 0, i as u64);
    }
    out
}

/// Which fixed jitter draw of [`TrainConfig::precursor_jitter_variants`](super::train::TrainConfig::precursor_jitter_variants)
/// training step `step` uses for spectrum `spectrum_index`.
///
/// With `variants == 0` there is no fixed pool (a fresh draw per
/// `(seed, step, spectrum)`); callers must not call this then. Otherwise the
/// draw is `hash(seed, step, spectrum_index) mod variants`, a deterministic
/// function of its inputs (wrapping multiplies with fixed constants), so
/// step `s` always reuses draw `v` and the driver can cache all `variants`
/// draws of every training spectrum up front. With `V > 0` the jitter is
/// drawn from a fixed pool of `V` draws per spectrum, not a fresh draw per
/// step: document this wherever the flag is surfaced.
pub fn jitter_variant_index(
    seed: u64,
    step: u64,
    spectrum_index: u64,
    variants: u32,
) -> u32 {
    debug_assert!(variants > 0);
    const A: u64 = 0x9E3779B97F4A7C15;
    const B: u64 = 0xBF58476D1CE4E5B9;
    const C: u64 = 0x94D049BB133111EB;
    let h = seed
        .wrapping_mul(A)
        .wrapping_add(step.wrapping_mul(B))
        .wrapping_add(spectrum_index.wrapping_mul(C));
    // Top bits of a multiplicative hash mix best; `variants` can be any
    // `u32` (including non-powers of two).
    ((h >> 32) % u64::from(variants.max(1))) as u32
}

/// Donor diagnostics for one batch: `(donor_same_molecule,
/// donor_no_eligible_peaks)`.
///
/// `donor_same_molecule` counts rows whose donor is the same molecule as the
/// recipient (must be 0 for [`ExperimentSet::donor_map`] donors).
/// `donor_no_eligible_peaks` counts rows where no donated peak passes the
/// mass clause of the existing peak filter (`0 < mz <= recipient precursor +
/// 2 Da`): the device selection then finds nothing, exactly as for any
/// request with no eligible peak. Such rows stay empty; they never fall back
/// to the recipient's own peaks.
pub fn donor_stats(
    set: &ExperimentSet,
    indices: &[usize],
    donors: &[usize],
) -> Result<(usize, usize)> {
    if indices.len() != donors.len() {
        return Err(Error::config(format!(
            "donor_stats: {} indices for {} donors (must match)",
            indices.len(),
            donors.len()
        )));
    }
    let mut same = 0usize;
    let mut no_eligible = 0usize;
    for (&ri, &di) in indices.iter().zip(donors.iter()) {
        let recipient = set.spectra.get(ri).ok_or_else(|| {
            Error::config(format!(
                "donor_stats: recipient index {ri} outside {} spectra",
                set.spectra.len()
            ))
        })?;
        let donor = set.spectra.get(di).ok_or_else(|| {
            Error::config(format!(
                "donor_stats: donor index {di} outside {} spectra",
                set.spectra.len()
            ))
        })?;
        if recipient.molecule == donor.molecule {
            same += 1;
        }
        let bound = recipient
            .spectrum
            .precursor_mz_udalton
            .saturating_add(2_000_000);
        let any = donor
            .spectrum
            .mz_udalton
            .iter()
            .any(|&mz| mz != 0 && mz <= bound);
        if !any {
            no_eligible += 1;
        }
    }
    Ok((same, no_eligible))
}

/// First request-domain reason in the contract §4.6 order (adduct, polarity),
/// or `None` when the adduct and polarity pass.
fn request_reason(s: &ExportSpectrum) -> Option<String> {
    if s.adduct == 0 {
        return Some("insufficient_metadata".to_string());
    }
    let Some(a) = adduct(s.adduct) else {
        return Some("unsupported_adduct".to_string());
    };
    if s.polarity != 1 && s.polarity != -1 {
        return Some("invalid_polarity".to_string());
    }
    if i32::from(s.polarity) != a.charge {
        return Some("polarity_adduct_conflict".to_string());
    }
    None
}

/// Precursor-range reason, checked after the structure domain per §4.6.
fn precursor_reason(s: &ExportSpectrum) -> Option<String> {
    if !(PRECURSOR_MIN..=PRECURSOR_MAX).contains(&s.precursor_mz_udalton) {
        return Some("precursor_out_of_range".to_string());
    }
    None
}

/// Deterministic 64-bit generator for [`ExperimentSet::batches`]: SplitMix64.
struct SplitMix64 {
    /// Current state.
    state: u64,
}

impl SplitMix64 {
    /// Seed the generator.
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Next `u64` (Steele et al. SplitMix64, fixed constants).
    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// SHA-256 of `bytes`, as lowercase hex (FIPS 180-4, no dependencies: the
/// crate has no hash crate and `Cargo.toml` is frozen for this task).
pub fn sha256_hex(bytes: &[u8]) -> String {
    // Initial hash values (first 32 bits of the fractional parts of the
    // square roots of the first 8 primes).
    let mut h: [u32; 8] = [
        0x6a09_e667,
        0xbb67_ae85,
        0x3c6e_f372,
        0xa54f_f53a,
        0x510e_527f,
        0x9b05_688c,
        0x1f83_d9ab,
        0x5be0_cd19,
    ];
    // Round constants (first 32 bits of the fractional parts of the cube
    // roots of the first 64 primes).
    const K: [u32; 64] = [
        0x428a_2f98,
        0x7137_4491,
        0xb5c0_fbcf,
        0xe9b5_dba5,
        0x3956_c25b,
        0x59f1_11f1,
        0x923f_82a4,
        0xab1c_5ed5,
        0xd807_aa98,
        0x1283_5b01,
        0x2431_85be,
        0x550c_7dc3,
        0x72be_5d74,
        0x80de_b1fe,
        0x9bdc_06a7,
        0xc19b_f174,
        0xe49b_69c1,
        0xefbe_4786,
        0x0fc1_9dc6,
        0x240c_a1cc,
        0x2de9_2c6f,
        0x4a74_84aa,
        0x5cb0_a9dc,
        0x76f9_88da,
        0x983e_5152,
        0xa831_c66d,
        0xb003_27c8,
        0xbf59_7fc7,
        0xc6e0_0bf3,
        0xd5a7_9147,
        0x06ca_6351,
        0x1429_2967,
        0x27b7_0a85,
        0x2e1b_2138,
        0x4d2c_6dfc,
        0x5338_0d13,
        0x650a_7354,
        0x766a_0abb,
        0x81c2_c92e,
        0x9272_2c85,
        0xa2bf_e8a1,
        0xa81a_664b,
        0xc24b_8b70,
        0xc76c_51a3,
        0xd192_e819,
        0xd699_0624,
        0xf40e_3585,
        0x106a_a070,
        0x19a4_c116,
        0x1e37_6c08,
        0x2748_774c,
        0x34b0_bcb5,
        0x391c_0cb3,
        0x4ed8_aa4a,
        0x5b9c_ca4f,
        0x682e_6ff3,
        0x748f_82ee,
        0x78a5_636f,
        0x84c8_7814,
        0x8cc7_0208,
        0x90be_fffa,
        0xa450_6ceb,
        0xbef9_a3f7,
        0xc671_78f2,
    ];
    // Padding: a 1 bit, zeros, then the bit length as a 64-bit big-endian
    // integer; the message is processed in 64-byte blocks.
    let mut msg = bytes.to_vec();
    let bit_len = (bytes.len() as u64).wrapping_mul(8);
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());
    for block in msg.chunks_exact(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                block[4 * i],
                block[4 * i + 1],
                block[4 * i + 2],
                block[4 * i + 3],
            ]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh) =
            (h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }
    let mut out = String::with_capacity(64);
    for v in h {
        out.push_str(&format!("{v:08x}"));
    }
    out
}

/// D6: refuse an `--enum-fit` export whose subset is not a training subset
/// (`train` or `fit`) or that shares any molecule key with validation.
/// Returns `Error::Config` naming the reason; the driver maps it to its
/// failure exit.
pub fn check_enum_fit(
    fit_subset: &str,
    fit_name: &str,
    fit_molecules: &[String],
    eval_molecules: &[String],
) -> crate::error::Result<()> {
    if fit_subset != "train" && fit_subset != "fit" {
        return Err(crate::error::Error::config(format!(
            "--enum-fit {fit_name} has subset '{fit_subset}' (expected a training subset 'train' or 'fit')"
        )));
    }
    if !eval_molecules.is_empty() {
        use std::collections::HashSet;
        let fit_keys: HashSet<&str> = fit_molecules.iter().map(|s| s.as_str()).collect();
        for key in eval_molecules {
            if fit_keys.contains(key.as_str()) {
                return Err(crate::error::Error::config(format!(
                    "--enum-fit {fit_name} shares molecule key '{key}' with validation: fitting must be train-only"
                )));
            }
        }
    }
    Ok(())
}
