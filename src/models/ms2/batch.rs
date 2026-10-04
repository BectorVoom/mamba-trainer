//! Device batches for the MS2 encoder (architecture §2).
//!
//! [`DeviceSpectra::upload`] validates the host batch first and copies its
//! five tensors to the device with no read. A spectrum with a fatal host
//! status is uploaded with `peak_count = 0`, so the selection kernels never
//! see a non-finite or negative intensity of a valid peak.

use cubecl::prelude::Runtime;

use crate::backend::{Device, FloatElem};
use crate::error::Result;
use crate::tensor::Tensor;
use crate::tensor::ops::index::IdTensor;

use super::contract::{SpectrumBatch, request_status};

/// Device copy of a [`SpectrumBatch`]: the raw peaks plus the packed integer
/// metadata and the float energy pair the encoder reads.
pub struct DeviceSpectra<R: Runtime, E: FloatElem> {
    /// Spectra per batch.
    pub batch: usize,
    /// Raw peak capacity of the shape bucket.
    pub n_raw: usize,
    /// Intensity scale of the upload (0 linear, 1 square root).
    pub intensity_scale: u32,
    /// Host copy, for provenance and the RNG.
    pub spectrum_id: Vec<u64>,
    /// [`SpectrumBatch::validate`] result, one bit set per spectrum.
    pub host_status: Vec<u32>,
    /// Host copy of the uploaded peak counts (0 for a spectrum with a fatal
    /// host status): an upper bound on the peaks the device keeps of each
    /// spectrum, known without a read.
    pub peak_count: Vec<u32>,
    /// `[B, n_raw]` m/z in micro-dalton units.
    pub mz: IdTensor<R>,
    /// `[B, n_raw]` untransformed intensities.
    pub intensity: Tensor<R, E>,
    /// `[B, 8]` per-spectrum metadata: `peak_count` (0 for a spectrum with a
    /// fatal host status), `precursor`, `precursor_uncertainty`, `adduct`,
    /// `fragment_tolerance`, `precursor_tolerance`, `id_lo`, `id_hi`.
    pub meta: IdTensor<R>,
    /// `[B, 4]`: adduct row (0 unknown, 1, 2), polarity row (0 = −1, 1 = +1),
    /// energy count (0..=8), energy known flag (0/1).
    pub meta_ids: IdTensor<R>,
    /// `[B, 2]`: collision energy in eV (0 when unknown), known flag.
    pub energy: Tensor<R, E>,
}

impl<R: Runtime, E: FloatElem> DeviceSpectra<R, E> {
    /// Validate the host batch (the error propagates) and upload its five
    /// tensors. Exactly 5 uploads, no device read.
    pub fn upload(batch: &SpectrumBatch, device: &Device<R>) -> Result<Self> {
        let statuses = batch.validate()?;
        let n = batch.len();
        let n_raw = batch.n_raw as usize;
        let mut meta = vec![0u32; n * 8];
        let mut peak_counts = vec![0u32; n];
        let mut meta_ids = vec![0u32; n * 4];
        let mut energy = vec![0.0f32; n * 2];
        for b in 0..n {
            let fatal = statuses[b] & request_status::FATAL_MASK != 0;
            let peak_count = if fatal { 0 } else { batch.peak_count[b] };
            meta[b * 8] = peak_count;
            peak_counts[b] = peak_count;
            meta[b * 8 + 1] = batch.precursor_mz_udalton[b];
            meta[b * 8 + 2] = batch.precursor_uncertainty_udalton[b];
            meta[b * 8 + 3] = u32::from(batch.adduct[b]);
            meta[b * 8 + 4] = batch.fragment_tolerance(b);
            meta[b * 8 + 5] = batch.precursor_tolerance(b);
            let id = batch.spectrum_id[b];
            meta[b * 8 + 6] = (id & 0xFFFF_FFFF) as u32;
            meta[b * 8 + 7] = (id >> 32) as u32;
            let adduct = batch.adduct[b];
            meta_ids[b * 4] = if adduct <= 2 { u32::from(adduct) } else { 0 };
            meta_ids[b * 4 + 1] = if batch.polarity[b] == 1 { 1 } else { 0 };
            meta_ids[b * 4 + 2] = u32::from(batch.energy_count[b]);
            meta_ids[b * 4 + 3] = u32::from(batch.collision_energy_known[b]);
            let known = batch.collision_energy_known[b] == 1;
            energy[b * 2] = if known {
                batch.collision_energy_ev[b]
            } else {
                0.0
            };
            energy[b * 2 + 1] = if known { 1.0 } else { 0.0 };
        }
        // Five uploads, in this order: mz, intensity, meta, meta_ids, energy.
        let mz = IdTensor::from_slice(&batch.mz_udalton, vec![n, n_raw], device)?;
        let intensity = Tensor::<R, E>::from_f32(&batch.intensity, vec![n, n_raw], device)?;
        let meta_t = IdTensor::from_slice(&meta, vec![n, 8], device)?;
        let meta_ids_t = IdTensor::from_slice(&meta_ids, vec![n, 4], device)?;
        let energy_t = Tensor::<R, E>::from_f32(&energy, vec![n, 2], device)?;
        Ok(Self {
            batch: n,
            n_raw,
            intensity_scale: u32::from(batch.intensity_scale),
            spectrum_id: batch.spectrum_id.clone(),
            host_status: statuses,
            peak_count: peak_counts,
            mz,
            intensity,
            meta: meta_t,
            meta_ids: meta_ids_t,
            energy: energy_t,
        })
    }
}

/// The `ShuffledSpectrum` control as an in-batch rotation: spectrum `b` gets
/// the peaks (`peak_count`, `raw_peak_count`, `peak_id`, `mz_udalton`,
/// `intensity`, `mz_uncertainty_udalton`) of spectrum `(b + 1) % B` and keeps
/// everything else, including its identity and metadata.
///
/// Why experiments must not use this directly: the rotation pairs spectra
/// without checking their molecules, so with molecule-ordered batches a
/// spectrum is usually evaluated on its sibling's peaks (sibling labels
/// overlap own labels by 0.56), and a one-row batch gets its own peaks back.
/// Experiments must use molecule-aware donors instead
/// ([`crate::models::ms2::experiment::ExperimentSet::donor_map`] plus
/// [`crate::models::ms2::experiment::spectrum_batch_with_donors`]), which draw
/// a different molecule's spectrum for every row. `generate` under
/// `ShuffledSpectrum` rejects batches of fewer than 2 spectra
/// ([`crate::error::Error::Config`]) rather than silently returning the
/// identity here.
pub fn rotate_peaks(batch: &SpectrumBatch) -> SpectrumBatch {
    let n = batch.len();
    if n == 0 {
        return batch.clone();
    }
    let n_raw = batch.n_raw as usize;
    let mut out = batch.clone();
    for b in 0..n {
        let src = (b + 1) % n;
        out.peak_count[b] = batch.peak_count[src];
        out.raw_peak_count[b] = batch.raw_peak_count[src];
        out.mz_uncertainty_udalton[b] = batch.mz_uncertainty_udalton[src];
        out.peak_id[b * n_raw..(b + 1) * n_raw]
            .copy_from_slice(&batch.peak_id[src * n_raw..(src + 1) * n_raw]);
        out.mz_udalton[b * n_raw..(b + 1) * n_raw]
            .copy_from_slice(&batch.mz_udalton[src * n_raw..(src + 1) * n_raw]);
        out.intensity[b * n_raw..(b + 1) * n_raw]
            .copy_from_slice(&batch.intensity[src * n_raw..(src + 1) * n_raw]);
    }
    out
}
