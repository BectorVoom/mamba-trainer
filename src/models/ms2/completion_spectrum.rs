//! Spectral evidence for the completion model: fragment peaks, the adduct
//! and the neutral mass.
//!
//! This is the third conditioning input of
//! [`CompletionModel`](super::completion_model::CompletionModel), next to the
//! substructure set and the fingerprint. One [`SpectrumEvidence`] holds a
//! measured MS/MS peak list, the precursor m/z it was isolated at, the adduct
//! and the neutral monoisotopic mass. The host side ([`SpectrumBatch`]) keeps
//! the most intense peaks and turns them into the 71 peak features and the
//! 34 metadata features of the spectrum encoder contract (the tested host
//! twins [`twin::peak_features`] and [`twin::meta_features`], with the
//! neutral mass in the precursor slot of the metadata row); the device side
//! ([`SpectrumEncoder`], a [`Module`]) embeds the peaks as a set, mixes them
//! and pools them with the adduct and the mass.
//!
//! What this input is and is not: it is conditioning evidence. It does not
//! constrain legality. The mass constraint on a generated molecule is the
//! exact composition the grammar is run with (a formula hypothesis of the
//! neutral mass); the adduct ties that neutral mass to the precursor m/z
//! through [`neutral_mass_of`]. Agreement of a candidate with the peaks or
//! with the fingerprint is checked after generation, never assumed.
//!
//! All masses are integer micro-dalton (`u32`), as everywhere in
//! [`chem`](super::chem).

use cubecl::prelude::Runtime;
use serde::{Deserialize, Serialize};

use crate::autograd::Var;
use crate::backend::{Device, FloatElem};
use crate::error::{Error, Result};
use crate::nn::init::Initializer;
use crate::nn::linear::{Linear, LinearConfig};
use crate::nn::module::{Module, ModuleVisitor};
use crate::nn::norm::{RmsNorm, RmsNormConfig};
use crate::nn::param::Param;
use crate::tensor::Tensor;
use crate::tensor::ops::index::IdTensor;
use crate::tensor::ops::ms2::{META_FEATURES, META_WIDTH, PEAK_FEATURES};
use crate::tensor::ops::random::Rng;

use super::completion_fingerprint::pooled_mean;
use super::twin;

/// Peak slots per query (the default of
/// [`CompletionModelConfig::base_spectrum`](super::completion_model::CompletionModelConfig::base_spectrum)).
pub const SPECTRUM_SLOTS: usize = 64;
/// Set-mixing rounds of [`SpectrumEncoder`].
pub const SPECTRUM_ROUNDS: usize = 2;

/// One adduct of the completion domain: a singly charged ion whose m/z is the
/// neutral mass plus a fixed shift.
pub struct CompletionAdduct {
    /// Adduct id (0 means unknown and has no entry). Ids 1 and 2 are the V0
    /// adducts of [`chem::ADDUCTS`](super::chem::ADDUCTS).
    pub id: u16,
    /// Adduct name, e.g. `[M+Na]+`.
    pub name: &'static str,
    /// `m/z - neutral mass` in micro-dalton (the added species minus the
    /// electron for a cation, plus it for an anion), rounded once.
    pub shift_uda: i64,
}

/// The adducts a [`SpectrumEvidence`] may name, in id order.
///
/// Shifts from the exact masses of [`chem::ELEMENTS`](super::chem::ELEMENTS)
/// and the electron (`0.000548579909`): `H = 1.00782503223`,
/// `Na = 22.98976928`, `N + 4 H = 18.03437413335`, `K = 38.96370649`.
pub const COMPLETION_ADDUCTS: [CompletionAdduct; 5] = [
    CompletionAdduct {
        id: 1,
        name: "[M+H]+",
        shift_uda: 1_007_276,
    },
    CompletionAdduct {
        id: 2,
        name: "[M-H]-",
        shift_uda: -1_007_276,
    },
    CompletionAdduct {
        id: 3,
        name: "[M+Na]+",
        shift_uda: 22_989_221,
    },
    CompletionAdduct {
        id: 4,
        name: "[M+NH4]+",
        shift_uda: 18_033_826,
    },
    CompletionAdduct {
        id: 5,
        name: "[M+K]+",
        shift_uda: 38_963_158,
    },
];

/// Rows of the adduct embedding table: row 0 (unknown) plus one per adduct.
pub const SPECTRUM_ADDUCT_ROWS: usize = COMPLETION_ADDUCTS.len() + 1;

/// Rounding bound in micro-dalton of one [`neutral_mass_of`] /
/// [`precursor_mz_of`] conversion (the shift is rounded once).
pub const ADDUCT_CONVERSION_ERROR_UDA: u32 = 1;

/// The adduct with this id, or `None` for 0 and unknown ids.
pub fn completion_adduct(id: u16) -> Option<&'static CompletionAdduct> {
    COMPLETION_ADDUCTS.iter().find(|a| a.id == id)
}

/// The adduct with this name, or `None`.
pub fn completion_adduct_by_name(name: &str) -> Option<&'static CompletionAdduct> {
    COMPLETION_ADDUCTS.iter().find(|a| a.name == name)
}

/// Neutral monoisotopic mass of a precursor m/z under `adduct`
/// (`m/z - shift`), or `None` when the adduct is unknown or the result leaves
/// `1..=u32::MAX`. Exact up to [`ADDUCT_CONVERSION_ERROR_UDA`].
pub fn neutral_mass_of(precursor_mz: u32, adduct: u16) -> Option<u32> {
    let shift = completion_adduct(adduct)?.shift_uda;
    let neutral = i64::from(precursor_mz) - shift;
    if neutral < 1 {
        return None;
    }
    u32::try_from(neutral).ok()
}

/// Precursor m/z of a neutral mass under `adduct` (`neutral + shift`), or
/// `None` when the adduct is unknown or the result leaves `1..=u32::MAX`.
pub fn precursor_mz_of(neutral_mass: u32, adduct: u16) -> Option<u32> {
    let shift = completion_adduct(adduct)?.shift_uda;
    let mz = i64::from(neutral_mass) + shift;
    if mz < 1 {
        return None;
    }
    u32::try_from(mz).ok()
}

/// Spectral evidence of one query.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SpectrumEvidence {
    /// `(m/z in micro-dalton, intensity)` fragment peaks in any order.
    /// Intensities are non-negative and finite on any scale; the encoder
    /// sees them relative to the strongest kept peak.
    pub peaks: Vec<(u32, f32)>,
    /// Precursor m/z in micro-dalton (`> 0`).
    pub precursor_mz: u32,
    /// Adduct id (one of [`COMPLETION_ADDUCTS`], or 0 for unknown).
    pub adduct: u16,
    /// Neutral monoisotopic mass in micro-dalton (`> 0`).
    pub neutral_mass: u32,
}

impl SpectrumEvidence {
    /// Check the fields: a positive precursor m/z and neutral mass, a known
    /// adduct id or 0, every peak m/z positive and every intensity finite
    /// and non-negative. Anything else is [`Error::Config`].
    pub fn validate(&self) -> Result<()> {
        if self.precursor_mz == 0 {
            return Err(Error::config(
                "SpectrumEvidence::validate: precursor_mz is 0 (needs a positive m/z)".to_string(),
            ));
        }
        if self.neutral_mass == 0 {
            return Err(Error::config(
                "SpectrumEvidence::validate: neutral_mass is 0 (needs a positive mass)".to_string(),
            ));
        }
        if self.adduct != 0 && completion_adduct(self.adduct).is_none() {
            return Err(Error::config(format!(
                "SpectrumEvidence::validate: adduct id {} is not one of the {} completion adducts (0 means unknown)",
                self.adduct,
                COMPLETION_ADDUCTS.len()
            )));
        }
        for (i, &(mz, intensity)) in self.peaks.iter().enumerate() {
            if mz == 0 {
                return Err(Error::config(format!(
                    "SpectrumEvidence::validate: peak {i} has m/z 0"
                )));
            }
            if !(intensity.is_finite() && intensity >= 0.0) {
                return Err(Error::config(format!(
                    "SpectrumEvidence::validate: peak {i} intensity {intensity} is not finite and non-negative"
                )));
            }
        }
        Ok(())
    }

    /// Peak selection for `slots` slots: the at most `slots` peaks with the
    /// highest positive intensity (ties by lower m/z, then input order), as
    /// `(m/z, intensity / strongest kept intensity)` in ascending m/z.
    /// Zero-intensity peaks are never kept. See
    /// [`dropped`](Self::dropped) for the slot-limit loss.
    pub fn selected(&self, slots: usize) -> Vec<(u32, f32)> {
        let mut order: Vec<usize> = (0..self.peaks.len())
            .filter(|&i| self.peaks[i].1 > 0.0)
            .collect();
        order.sort_by(|&a, &b| {
            self.peaks[b]
                .1
                .total_cmp(&self.peaks[a].1)
                .then_with(|| self.peaks[a].0.cmp(&self.peaks[b].0))
                .then_with(|| a.cmp(&b))
        });
        order.truncate(slots);
        let top = order.first().map(|&i| self.peaks[i].1).unwrap_or(1.0);
        order.sort_by(|&a, &b| {
            self.peaks[a]
                .0
                .cmp(&self.peaks[b].0)
                .then_with(|| a.cmp(&b))
        });
        order
            .into_iter()
            .map(|i| (self.peaks[i].0, self.peaks[i].1 / top))
            .collect()
    }

    /// Positive-intensity peaks dropped by the slot limit at `slots`.
    pub fn dropped(&self, slots: usize) -> usize {
        let positive = self.peaks.iter().filter(|p| p.1 > 0.0).count();
        positive.saturating_sub(slots)
    }
}

/// Host-side spectrum batch: the queries' selected peaks as contract peak
/// features, plus the mass features and the adduct id.
///
/// A query without evidence (`None`) has `present` 0, no valid slot, adduct
/// id 0 and zero features, and contributes exact zeros to the model.
#[derive(Clone, Debug)]
pub struct SpectrumBatch {
    /// Queries per batch.
    pub queries: usize,
    /// Peak slots per query.
    pub slots: usize,
    /// `[B*S*71]` peak features ([`twin::peak_features`]; zeros in padding).
    pub features: Vec<f32>,
    /// `[B*S]` slot validity, 1/0.
    pub valid: Vec<f32>,
    /// `[B*34]` mass features ([`twin::meta_features`] with the neutral mass
    /// in the precursor slot and no collision energy; zeros when absent).
    pub meta: Vec<f32>,
    /// `[B]` adduct ids (0 unknown or absent).
    pub adduct_ids: Vec<u32>,
    /// `[B]` 1 when the query carries evidence, else 0.
    pub present: Vec<f32>,
}

impl SpectrumBatch {
    /// Build a batch from one optional evidence per query with `slots` peak
    /// slots each. Every present evidence is validated. Empty input is
    /// [`Error::Config`].
    pub fn build(evidence: &[Option<&SpectrumEvidence>], slots: usize) -> Result<Self> {
        if evidence.is_empty() {
            return Err(Error::config(
                "SpectrumBatch::build: needs at least one query".to_string(),
            ));
        }
        let b = evidence.len();
        // The selection arrays of the encoder contract: `kept` is (raw index,
        // m/z, reverse index), `kept_f` is (relative intensity, valid flag),
        // `meta[.., 1]` the precursor m/z. Invalid slots carry `u32::MAX`.
        let mut kept = vec![u32::MAX; b * slots * 3];
        let mut kept_f = vec![0.0f32; b * slots * 2];
        let mut peak_meta = vec![0u32; b * META_WIDTH];
        let mut mass_meta = vec![0u32; b * META_WIDTH];
        let mut valid = vec![0.0f32; b * slots];
        let mut adduct_ids = vec![0u32; b];
        let mut present = vec![0.0f32; b];
        for (q, item) in evidence.iter().enumerate() {
            let Some(item) = item else { continue };
            item.validate()?;
            present[q] = 1.0;
            adduct_ids[q] = u32::from(item.adduct);
            peak_meta[q * META_WIDTH + 1] = item.precursor_mz;
            mass_meta[q * META_WIDTH + 1] = item.neutral_mass;
            let selected = item.selected(slots);
            let count = selected.len();
            for (s, &(mz, relative)) in selected.iter().enumerate() {
                let slot = (q * slots + s) * 3;
                kept[slot] = s as u32;
                kept[slot + 1] = mz;
                kept[slot + 2] = (count - 1 - s) as u32;
                kept_f[(q * slots + s) * 2] = relative;
                kept_f[(q * slots + s) * 2 + 1] = 1.0;
                valid[q * slots + s] = 1.0;
            }
        }
        let features = if slots == 0 {
            Vec::new()
        } else {
            twin::peak_features(&kept, &kept_f, &peak_meta, b, slots)
        };
        let energy = vec![0.0f32; b * 2];
        let mut meta = twin::meta_features(&mass_meta, &energy, b);
        for q in 0..b {
            if present[q] == 0.0 {
                meta[q * META_FEATURES..(q + 1) * META_FEATURES].fill(0.0);
            }
        }
        Ok(Self {
            queries: b,
            slots,
            features,
            valid,
            meta,
            adduct_ids,
            present,
        })
    }

    /// A batch of `queries` queries without evidence.
    pub fn empty(queries: usize, slots: usize) -> Result<Self> {
        let none: Vec<Option<&SpectrumEvidence>> = vec![None; queries];
        Self::build(&none, slots)
    }

    /// Check the host arrays: lengths fit `queries` queries of `slots`
    /// slots; `valid` and `present` are exactly 0.0 or 1.0; adduct ids are
    /// below [`SPECTRUM_ADDUCT_ROWS`]; every feature is finite; a query with
    /// `present` 0 has no valid slot and adduct id 0. Anything else is
    /// [`Error::Config`].
    pub fn validate(&self) -> Result<()> {
        let (b, s) = (self.queries, self.slots);
        if self.features.len() != b * s * PEAK_FEATURES
            || self.valid.len() != b * s
            || self.meta.len() != b * META_FEATURES
            || self.adduct_ids.len() != b
            || self.present.len() != b
        {
            return Err(Error::config(format!(
                "SpectrumBatch::validate: lengths {} {} {} {} {} do not fit {b} queries of {s} slots",
                self.features.len(),
                self.valid.len(),
                self.meta.len(),
                self.adduct_ids.len(),
                self.present.len()
            )));
        }
        if let Some(i) = self.features.iter().position(|v| !v.is_finite()) {
            return Err(Error::config(format!(
                "SpectrumBatch::validate: features[{i}] is not finite"
            )));
        }
        if let Some(i) = self.meta.iter().position(|v| !v.is_finite()) {
            return Err(Error::config(format!(
                "SpectrumBatch::validate: meta[{i}] is not finite"
            )));
        }
        for q in 0..b {
            let p = self.present[q];
            if p != 0.0 && p != 1.0 {
                return Err(Error::config(format!(
                    "SpectrumBatch::validate: present[{q}] is {p} (needs exactly 0.0 or 1.0)"
                )));
            }
            if self.adduct_ids[q] >= SPECTRUM_ADDUCT_ROWS as u32 {
                return Err(Error::config(format!(
                    "SpectrumBatch::validate: adduct_ids[{q}] is {} (needs below {SPECTRUM_ADDUCT_ROWS})",
                    self.adduct_ids[q]
                )));
            }
            for slot in 0..s {
                let v = self.valid[q * s + slot];
                if v != 0.0 && v != 1.0 {
                    return Err(Error::config(format!(
                        "SpectrumBatch::validate: valid[{}] is {v} (needs exactly 0.0 or 1.0)",
                        q * s + slot
                    )));
                }
                if p == 0.0 && v != 0.0 {
                    return Err(Error::config(format!(
                        "SpectrumBatch::validate: query {q} has a valid peak slot but no evidence (present 0)"
                    )));
                }
            }
            if p == 0.0 && self.adduct_ids[q] != 0 {
                return Err(Error::config(format!(
                    "SpectrumBatch::validate: query {q} has adduct id {} but no evidence (present 0)",
                    self.adduct_ids[q]
                )));
            }
        }
        Ok(())
    }
}

/// One set-mixing round of [`SpectrumEncoder`].
struct SpectrumRound<R: Runtime, E: FloatElem> {
    /// `Linear(d, d)` self term.
    own: Linear<R, E>,
    /// `Linear(d, d)` pooled-mean term.
    mean: Linear<R, E>,
    /// `Linear(d, d)` output projection of the residual update.
    out: Linear<R, E>,
    /// Per-round normalisation.
    norm: RmsNorm<R, E>,
}

impl<R: Runtime, E: FloatElem> Module<R, E> for SpectrumRound<R, E> {
    fn visit(&self, visitor: &mut ModuleVisitor<'_, R, E>) {
        visitor.child("own", &self.own);
        visitor.child("mean", &self.mean);
        visitor.child("out", &self.out);
        visitor.child("norm", &self.norm);
    }
}

/// Permutation-invariant encoder of the peak set, the adduct and the mass.
///
/// Peak features go through `Linear(71, d)`, SiLU, `Linear(d, d)` and
/// [`SPECTRUM_ROUNDS`] set-mixing rounds `h = h + out(silu(own(h) +
/// mean(masked mean of h)))`, each followed by RmsNorm and selection to exact
/// zeros in padding. The m/z and neutral-loss Fourier features carry the
/// position, so there is no slot embedding and the output does not depend on
/// the order of the peaks. The pooled vector is `pool_in(masked mean of h)`
/// (exactly zero without a valid peak) plus `meta_out(silu(meta_in(mass
/// features) + adduct embedding))`, the whole selected to exact zero for a
/// query without evidence.
pub struct SpectrumEncoder<R: Runtime, E: FloatElem> {
    /// `Linear(71, d)` peak feature input.
    peak_in: Linear<R, E>,
    /// `Linear(d, d)` peak feature output.
    peak_out: Linear<R, E>,
    /// Set-mixing rounds (always [`SPECTRUM_ROUNDS`]).
    rounds: Vec<SpectrumRound<R, E>>,
    /// `[6, d]` adduct table (row 0 is unknown).
    adduct_emb: Param<R, E>,
    /// `Linear(34, d)` mass feature input.
    meta_in: Linear<R, E>,
    /// `Linear(d, d)` mass/adduct output.
    meta_out: Linear<R, E>,
    /// `Linear(d, d)` projection of the pooled peak vector.
    pool_in: Linear<R, E>,
    /// Residual width.
    d_model: usize,
}

impl<R: Runtime, E: FloatElem> Module<R, E> for SpectrumEncoder<R, E> {
    fn visit(&self, visitor: &mut ModuleVisitor<'_, R, E>) {
        visitor.child("peak_in", &self.peak_in);
        visitor.child("peak_out", &self.peak_out);
        for (r, round) in self.rounds.iter().enumerate() {
            visitor.child_at("round", r, round);
        }
        visitor.param("adduct_emb", &self.adduct_emb);
        visitor.child("meta_in", &self.meta_in);
        visitor.child("meta_out", &self.meta_out);
        visitor.child("pool_in", &self.pool_in);
    }
}

/// Device output of [`SpectrumEncoder::encode`].
pub struct SpectrumStates<R: Runtime, E: FloatElem> {
    /// `[B, S, d]` peak states (exact zeros in padding).
    pub states: Var<R, E>,
    /// `[B, S]` slot validity.
    pub valid: Tensor<R, E>,
    /// `[B, d]` pooled peak, adduct and mass vector (exact zero rows without
    /// evidence).
    pub pooled: Var<R, E>,
}

impl<R: Runtime, E: FloatElem> SpectrumEncoder<R, E> {
    /// Build the encoder for width `d`.
    pub fn init(d_model: usize, device: &Device<R>, rng: &mut Rng) -> Self {
        let mut rounds = Vec::with_capacity(SPECTRUM_ROUNDS);
        for _ in 0..SPECTRUM_ROUNDS {
            rounds.push(SpectrumRound {
                own: LinearConfig::new(d_model, d_model).init(device, rng),
                mean: LinearConfig::new(d_model, d_model).init(device, rng),
                out: LinearConfig::new(d_model, d_model).init(device, rng),
                norm: RmsNormConfig::new(d_model).init(device, rng),
            });
        }
        let peak_in = LinearConfig::new(PEAK_FEATURES, d_model).init(device, rng);
        let peak_out = LinearConfig::new(d_model, d_model).init(device, rng);
        let adduct_emb = Param::new(
            Initializer::Normal {
                mean: 0.0,
                std: 0.02,
            }
            .init(vec![SPECTRUM_ADDUCT_ROWS, d_model], device, rng),
        );
        Self {
            peak_in,
            peak_out,
            rounds,
            adduct_emb,
            meta_in: LinearConfig::new(META_FEATURES, d_model).init(device, rng),
            meta_out: LinearConfig::new(d_model, d_model).init(device, rng),
            pool_in: LinearConfig::new(d_model, d_model).init(device, rng),
            d_model,
        }
    }

    /// Encode `batch`: 5 uploads, no device read. See the type docs for the
    /// computation.
    pub fn encode(
        &self,
        batch: &SpectrumBatch,
        device: &Device<R>,
    ) -> Result<SpectrumStates<R, E>> {
        batch.validate()?;
        let (b, s, d) = (batch.queries, batch.slots, self.d_model);
        if b == 0 {
            return Ok(SpectrumStates {
                states: Var::constant(Tensor::<R, E>::zeros(vec![0, s, d], device)),
                valid: Tensor::<R, E>::zeros(vec![0, s], device),
                pooled: Var::constant(Tensor::<R, E>::zeros(vec![0, d], device)),
            });
        }
        let present = Tensor::<R, E>::from_f32(&batch.present, vec![b, 1], device)?;
        let present_mask = Var::constant(present).expand(vec![b, d])?;
        let meta = Tensor::<R, E>::from_f32(&batch.meta, vec![b, META_FEATURES], device)?;
        let adduct_ids = IdTensor::from_slice(&batch.adduct_ids, vec![b], device)?;
        // Mass and adduct: `meta_in(mass) + adduct`, with the mass term first
        // (the first parent keeps the accumulator tape, as in the pattern and
        // fingerprint encoders).
        let adduct =
            Var::ms2_lookup(&self.adduct_emb.var_standalone(), &adduct_ids)?.reshape(vec![b, d])?;
        let mass = self.meta_out.apply(
            &self
                .meta_in
                .apply(&Var::constant(meta))?
                .add(&adduct)?
                .silu()?,
        )?;
        if s == 0 {
            return Ok(SpectrumStates {
                states: Var::constant(Tensor::<R, E>::zeros(vec![b, 0, d], device)),
                valid: Tensor::<R, E>::zeros(vec![b, 0], device),
                pooled: mass.mul(&present_mask)?,
            });
        }
        let valid = Tensor::<R, E>::from_f32(&batch.valid, vec![b, s], device)?;
        let features =
            Tensor::<R, E>::from_f32(&batch.features, vec![b, s, PEAK_FEATURES], device)?;
        let mut h = self
            .peak_out
            .apply(&self.peak_in.apply(&Var::constant(features))?.silu()?)?
            .ms2_select_valid(&valid)?;
        for round in &self.rounds {
            let own_h = round.own.apply(&h)?;
            let mean = pooled_mean(&h, &valid, device)?;
            let broadcast = round
                .mean
                .apply(&mean)?
                .unsqueeze(1)?
                .expand(vec![b, s, d])?;
            let update = round.out.apply(&own_h.add(&broadcast)?.silu()?)?;
            h = h.add(&update)?;
            h = round.norm.apply(&h)?.ms2_select_valid(&valid)?;
        }
        // Row mask of the pooled peaks: `pool_in` has a bias, so a query
        // without a valid peak is selected back to exact zero.
        let len_t = crate::tensor::ops::reduce::sum_dim(&valid, 1)?.reshape(vec![b, 1])?;
        let len_var = Var::constant(len_t);
        let one_var = Var::constant(Tensor::<R, E>::ones(vec![b, 1], device));
        let den = len_var.maximum(&one_var)?;
        let peak_mask = len_var.div(&den)?.expand(vec![b, d])?;
        let peaks = self
            .pool_in
            .apply(&pooled_mean(&h, &valid, device)?)?
            .mul(&peak_mask)?;
        let pooled = mass.add(&peaks)?.mul(&present_mask)?;
        Ok(SpectrumStates {
            states: h,
            valid,
            pooled,
        })
    }
}
