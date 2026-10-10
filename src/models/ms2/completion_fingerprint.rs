//! Fingerprint evidence for the completion model (`morgan4096`, 4096 bits).
//!
//! The Rust crate cannot compute this fingerprint itself, so it is always an
//! input: true on-bit lists exported by
//! `tools/ms2/export_fingerprints_mist.py` (`bits` subcommand) for training,
//! predicted probabilities at inference. This module holds the host side
//! (sparse fingerprints, token selection, the bits/noise file readers and
//! the MIST-like sampler) and the device side ([`FingerprintEncoder`], a
//! [`Module`]).
//!
//! Token selection: at most `slots` entries with the highest probability
//! (ties by lower bit index) become token ids (`1 + bit`, 0 is padding),
//! confidence bucket ids (8 equal-width probability buckets, `1..=8`, 0 is
//! padding) and validity. [`SparseFingerprint::tokens`] documents the bucket
//! map; [`SparseFingerprint::dropped`] records how many entries the slot
//! limit removed.
//!
//! Noise sampler numbers (from `fp_mist_noise.json`): the sampler uses
//! `hist_pred_given_true_on` (20 bins over `[0, 1]`, bin `i` covering
//! `[i/20, (i+1)/20)`, last bin including 1.0), `hist_pred_given_true_off`
//! (same bins), `n_spectra` (746) and `n_bins` (20). The v2 noise file
//! additionally carries `hist_pred_given_true_on_molecule` /
//! `hist_pred_given_true_off_molecule`, which pool the per-molecule averaged
//! probabilities (one row per molecule, the averaged-panel deployment
//! setting); the run selects the set with `--fp-noise-level
//! spectrum|molecule` (default `spectrum`). Every true-on bit draws one
//! probability from the on-histogram (bin proportional to its count,
//! uniform within the bin); every true-off bit (each of the `4096 -
//! true.len()` bits) independently draws one probability from the
//! off-histogram the same way. Entries below the token threshold are
//! dropped. Sampling keeps the surviving fraction of a bin the threshold
//! cuts through (uniform within the bin), so the expected true-bit
//! retention rate integrates the partial bin instead of dropping it whole;
//! see [`FingerprintNoise::retention_rate`]. The corpus-average false-token
//! rate is the kept off-mass per spectrum; the expectation for one query
//! with `t` true bits is `(4096 - t)` times the off survival probability
//! (see [`FingerprintNoise::expected_false_tokens`]). The RNG is the crate's
//! [`SplitMix64`](super::completion_data::SplitMix64), mixed from seed,
//! molecule key and draw exactly like the pattern extraction
//! ([`mix_fp_seed`]).
//!
//! Per-bit channel ([`FingerprintChannel`], file from
//! `tools/ms2/fit_fingerprint_channel.py`): the pooled histograms above say
//! how often a predictor is right, not which bits it is right about, and a
//! decoder trained on them did not transfer to real predictions. The channel
//! keeps one outcome distribution per bit, truth value and latent quality
//! class, fitted on real predictions for molecules the predictor never
//! trained on, and is what lets structure-only molecules train on
//! predicted-looking fingerprints.

use std::collections::{HashMap, HashSet};
use std::path::Path;

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
use crate::tensor::ops::random::Rng;

use super::completion::stable_hash;
use super::completion_data::SplitMix64;

/// Fingerprint width: MIST `morgan4096`.
pub const FINGERPRINT_BITS: usize = 4096;
/// Fingerprint token slots per query (the request layer's limit).
pub const FINGERPRINT_SLOTS: usize = 128;
/// Token id rows: padding row 0 plus `1 + bit`.
pub const FINGERPRINT_TOKEN_ROWS: usize = FINGERPRINT_BITS + 1;
/// Confidence bucket rows: padding row 0 plus buckets `1..=8`.
pub const FINGERPRINT_BUCKET_ROWS: usize = 9;
/// Confidence buckets over `[0, 1]`.
pub const FINGERPRINT_BUCKETS: usize = 8;
/// Set-mixing rounds of [`FingerprintEncoder`].
pub const FINGERPRINT_ROUNDS: usize = 2;

/// Which numbers of `fp_mist_noise.json` the sampler uses: `n_bins` must be
/// 20.
pub const FINGERPRINT_NOISE_BINS: usize = 20;

/// Which histogram set of the noise file a run samples.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FingerprintNoiseLevel {
    /// Per-spectrum histograms: every spectrum's prediction is one row.
    #[default]
    Spectrum,
    /// Per-molecule-averaged histograms: one averaged row per molecule (the
    /// averaged-panel deployment setting).
    Molecule,
}

impl FingerprintNoiseLevel {
    /// The CLI spelling.
    pub fn as_str(self) -> &'static str {
        match self {
            FingerprintNoiseLevel::Spectrum => "spectrum",
            FingerprintNoiseLevel::Molecule => "molecule",
        }
    }

    /// Parse a CLI spelling; `None` for anything else.
    pub fn parse(text: &str) -> Option<Self> {
        match text {
            "spectrum" => Some(FingerprintNoiseLevel::Spectrum),
            "molecule" => Some(FingerprintNoiseLevel::Molecule),
            _ => None,
        }
    }
}

/// Sparse fingerprint: sorted `(bit, probability)` entries.
///
/// `bit < 4096`, probability in `(0, 1]`, sorted by bit index with no
/// duplicates. Built by [`from_bits`](SparseFingerprint::from_bits)
/// (probability 1) or
/// [`from_probabilities`](SparseFingerprint::from_probabilities) (threshold
/// filter), both validated.
#[derive(Clone, Debug, PartialEq)]
pub struct SparseFingerprint {
    /// Sorted `(bit index, probability)` entries.
    pub entries: Vec<(u16, f32)>,
}

impl SparseFingerprint {
    /// Check `entries`: sorted by index, no duplicates, every index below
    /// [`FINGERPRINT_BITS`], every probability in `(0, 1]` and finite.
    /// Anything else is [`Error::Config`].
    pub fn validate(&self) -> Result<()> {
        let mut prev: Option<u16> = None;
        for (i, &(bit, prob)) in self.entries.iter().enumerate() {
            if usize::from(bit) >= FINGERPRINT_BITS {
                return Err(Error::config(format!(
                    "SparseFingerprint::validate: entry {i} bit {bit} is past {FINGERPRINT_BITS}"
                )));
            }
            if !(prob.is_finite() && prob > 0.0 && prob <= 1.0) {
                return Err(Error::config(format!(
                    "SparseFingerprint::validate: entry {i} bit {bit} probability {prob} is not in (0, 1]"
                )));
            }
            if let Some(p) = prev
                && bit <= p
            {
                return Err(Error::config(format!(
                    "SparseFingerprint::validate: entry {i} bit {bit} is not strictly after {p} (needs sorted unique indices)"
                )));
            }
            prev = Some(bit);
        }
        Ok(())
    }

    /// Fingerprint with every bit at probability 1 (exact training evidence).
    /// `bits` may repeat or arrive unsorted; duplicates collapse. Out-of-range
    /// indices are [`Error::Config`].
    pub fn from_bits(bits: &[u16]) -> Result<Self> {
        let mut sorted = bits.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        for &bit in &sorted {
            if usize::from(bit) >= FINGERPRINT_BITS {
                return Err(Error::config(format!(
                    "SparseFingerprint::from_bits: bit {bit} is past {FINGERPRINT_BITS}"
                )));
            }
        }
        Ok(Self {
            entries: sorted.into_iter().map(|bit| (bit, 1.0)).collect(),
        })
    }

    /// Fingerprint from `(bit, probability)` pairs, keeping entries with
    /// `probability >= threshold`. `threshold` must be in `(0, 1]`; entries
    /// with non-finite or non-positive probabilities are dropped with the
    /// below-threshold ones (never an error); out-of-range indices and
    /// probabilities above 1 are [`Error::Config`]. Output sorted by index.
    pub fn from_probabilities(pairs: &[(u16, f32)], threshold: f32) -> Result<Self> {
        if !(threshold.is_finite() && threshold > 0.0 && threshold <= 1.0) {
            return Err(Error::config(format!(
                "SparseFingerprint::from_probabilities: threshold {threshold} is not in (0, 1]"
            )));
        }
        let mut kept: Vec<(u16, f32)> = Vec::new();
        for &(bit, prob) in pairs {
            if usize::from(bit) >= FINGERPRINT_BITS {
                return Err(Error::config(format!(
                    "SparseFingerprint::from_probabilities: bit {bit} is past {FINGERPRINT_BITS}"
                )));
            }
            if prob > 1.0 || (prob.is_finite() && prob <= 0.0) {
                if prob > 1.0 {
                    return Err(Error::config(format!(
                        "SparseFingerprint::from_probabilities: bit {bit} probability {prob} exceeds 1"
                    )));
                }
                continue;
            }
            if !prob.is_finite() || prob < threshold {
                continue;
            }
            kept.push((bit, prob));
        }
        kept.sort_by_key(|&(bit, _)| bit);
        kept.dedup_by_key(|&mut (bit, _)| bit);
        let out = Self { entries: kept };
        out.validate()?;
        Ok(out)
    }

    /// Confidence bucket of probability `p` in `(0, 1]`: `ceil(p * 8)`
    /// clamped to `1..=8` (so `(0, 1/8]` maps to 1 and exactly 1.0 maps to
    /// 8). Non-finite or non-positive input gives 0 (padding; never produced
    /// for validated entries).
    pub fn bucket(prob: f32) -> u32 {
        if !(prob.is_finite() && prob > 0.0) {
            return 0;
        }
        let bucket = (f32::from(FINGERPRINT_BUCKETS as u8) * prob).ceil() as u32;
        bucket.clamp(1, FINGERPRINT_BUCKETS as u32)
    }

    /// Token selection for `slots` slots: the at most `slots` entries with
    /// the highest probability (ties by lower bit index) as token ids
    /// (`1 + bit`, 0 is padding), confidence bucket ids (`1..=8`, 0 is
    /// padding) and validity (1/0). Output length is exactly `slots`
    /// (padded with zeros when entries run out). Order of the selected
    /// tokens is probability-descending, index-ascending; the encoder is
    /// order-invariant so this order carries no signal. See
    /// [`dropped`](SparseFingerprint::dropped) for the slot-limit loss.
    pub fn tokens(&self, slots: usize) -> (Vec<u32>, Vec<u32>, Vec<f32>) {
        let mut order: Vec<usize> = (0..self.entries.len()).collect();
        order.sort_by(|&a, &b| {
            self.entries[b]
                .1
                .total_cmp(&self.entries[a].1)
                .then_with(|| self.entries[a].0.cmp(&self.entries[b].0))
        });
        let take = order.len().min(slots);
        let mut ids = vec![0u32; slots];
        let mut buckets = vec![0u32; slots];
        let mut valid = vec![0.0f32; slots];
        for (s, &e) in order.iter().take(take).enumerate() {
            let (bit, prob) = self.entries[e];
            ids[s] = u32::from(bit) + 1;
            buckets[s] = Self::bucket(prob);
            valid[s] = 1.0;
        }
        (ids, buckets, valid)
    }

    /// Entries dropped by the slot limit at `slots`: `entries.len() -
    /// min(entries.len(), slots)`.
    pub fn dropped(&self, slots: usize) -> usize {
        self.entries
            .len()
            .saturating_sub(slots.min(self.entries.len()))
    }
}

/// Host-side fingerprint batch: the queries' token selections laid into
/// `slots` slots each.
///
/// Padding slots hold token id 0, bucket id 0 and validity 0. Only these
/// three arrays reach the device.
#[derive(Clone, Debug)]
pub struct FingerprintBatch {
    /// Queries per batch.
    pub queries: usize,
    /// Slots per query.
    pub slots: usize,
    /// `[B*S]` token ids (`0` padding, else `1 + bit`).
    pub token_ids: Vec<u32>,
    /// `[B*S]` confidence bucket ids (`0` padding, else `1..=8`).
    pub bucket_ids: Vec<u32>,
    /// `[B*S]` slot validity, 1/0.
    pub valid: Vec<f32>,
}

impl FingerprintBatch {
    /// Build a batch from one fingerprint per query with `slots` slots each:
    /// [`tokens`](SparseFingerprint::tokens) per query. Empty input is
    /// [`Error::Config`].
    pub fn build(fingerprints: &[SparseFingerprint], slots: usize) -> Result<Self> {
        if fingerprints.is_empty() {
            return Err(Error::config(
                "FingerprintBatch::build: needs at least one query".to_string(),
            ));
        }
        let queries = fingerprints.len();
        let mut token_ids = vec![0u32; queries * slots];
        let mut bucket_ids = vec![0u32; queries * slots];
        let mut valid = vec![0.0f32; queries * slots];
        for (b, fp) in fingerprints.iter().enumerate() {
            fp.validate()?;
            let (ids, buckets, mask) = fp.tokens(slots);
            for s in 0..slots {
                token_ids[b * slots + s] = ids[s];
                bucket_ids[b * slots + s] = buckets[s];
                valid[b * slots + s] = mask[s];
            }
        }
        Ok(Self {
            queries,
            slots,
            token_ids,
            bucket_ids,
            valid,
        })
    }

    /// An empty (all-padding) batch of `queries` queries with `slots` slots.
    pub fn empty(queries: usize, slots: usize) -> Result<Self> {
        if queries == 0 {
            return Err(Error::config(
                "FingerprintBatch::empty: needs at least one query".to_string(),
            ));
        }
        Ok(Self {
            queries,
            slots,
            token_ids: vec![0u32; queries * slots],
            bucket_ids: vec![0u32; queries * slots],
            valid: vec![0.0f32; queries * slots],
        })
    }

    /// Check the host arrays: lengths fit `queries` queries of `slots` slots;
    /// every `valid` entry exactly 0.0 or 1.0; every token id below 4097 with
    /// 0 exactly where invalid (and nonzero where valid); every bucket id
    /// below 9 with 0 exactly where invalid and `1..=8` where valid.
    /// Anything else is [`Error::Config`].
    pub fn validate(&self) -> Result<()> {
        let (b, s) = (self.queries, self.slots);
        if self.token_ids.len() != b * s {
            return Err(Error::config(format!(
                "FingerprintBatch::validate: token_ids holds {} entries for {b} queries of {s} slots",
                self.token_ids.len()
            )));
        }
        if self.bucket_ids.len() != b * s {
            return Err(Error::config(format!(
                "FingerprintBatch::validate: bucket_ids holds {} entries for {b} queries of {s} slots",
                self.bucket_ids.len()
            )));
        }
        if self.valid.len() != b * s {
            return Err(Error::config(format!(
                "FingerprintBatch::validate: valid holds {} entries for {b} queries of {s} slots",
                self.valid.len()
            )));
        }
        for (i, &v) in self.valid.iter().enumerate() {
            if v != 0.0 && v != 1.0 {
                return Err(Error::config(format!(
                    "FingerprintBatch::validate: valid[{i}] is {v} (needs exactly 0.0 or 1.0)"
                )));
            }
            let id = self.token_ids[i];
            let bucket = self.bucket_ids[i];
            if id >= FINGERPRINT_TOKEN_ROWS as u32 {
                return Err(Error::config(format!(
                    "FingerprintBatch::validate: token_ids[{i}] is {id} (needs below {})",
                    FINGERPRINT_TOKEN_ROWS
                )));
            }
            if bucket >= FINGERPRINT_BUCKET_ROWS as u32 {
                return Err(Error::config(format!(
                    "FingerprintBatch::validate: bucket_ids[{i}] is {bucket} (needs below {})",
                    FINGERPRINT_BUCKET_ROWS
                )));
            }
            if v == 0.0 {
                if id != 0 {
                    return Err(Error::config(format!(
                        "FingerprintBatch::validate: token_ids[{i}] is {id} in a padding slot (needs 0)"
                    )));
                }
                if bucket != 0 {
                    return Err(Error::config(format!(
                        "FingerprintBatch::validate: bucket_ids[{i}] is {bucket} in a padding slot (needs 0)"
                    )));
                }
            } else {
                if id == 0 {
                    return Err(Error::config(format!(
                        "FingerprintBatch::validate: token_ids[{i}] is 0 in a valid slot (needs 1 + bit)"
                    )));
                }
                if !(1..=FINGERPRINT_BUCKETS as u32).contains(&bucket) {
                    return Err(Error::config(format!(
                        "FingerprintBatch::validate: bucket_ids[{i}] is {bucket} in a valid slot (needs 1..=8)"
                    )));
                }
            }
        }
        Ok(())
    }

    /// Upload the three arrays: 3 uploads, no launch, no read.
    pub(crate) fn upload<R: Runtime, E: FloatElem>(
        &self,
        device: &Device<R>,
    ) -> Result<DeviceFingerprints<R, E>> {
        let (b, s) = (self.queries, self.slots);
        if self.token_ids.len() != b * s
            || self.bucket_ids.len() != b * s
            || self.valid.len() != b * s
        {
            return Err(Error::shape(format!(
                "FingerprintBatch::upload: lengths {} {} {} do not fit {b} queries of {s} slots",
                self.token_ids.len(),
                self.bucket_ids.len(),
                self.valid.len()
            )));
        }
        Ok(DeviceFingerprints {
            token_ids: IdTensor::from_host(self.token_ids.clone(), vec![b, s], device)?,
            bucket_ids: IdTensor::from_host(self.bucket_ids.clone(), vec![b, s], device)?,
            valid: Tensor::<R, E>::from_f32(&self.valid, vec![b, s], device)?,
        })
    }
}

/// Device copy of a [`FingerprintBatch`]: exactly the three host arrays.
pub(crate) struct DeviceFingerprints<R: Runtime, E: FloatElem> {
    /// `[B, S]` token ids.
    pub(crate) token_ids: IdTensor<R>,
    /// `[B, S]` confidence bucket ids.
    pub(crate) bucket_ids: IdTensor<R>,
    /// `[B, S]` slot validity.
    pub(crate) valid: Tensor<R, E>,
}

/// One set-mixing round of [`FingerprintEncoder`].
struct FingerprintRound<R: Runtime, E: FloatElem> {
    /// `Linear(d, d)` self term.
    own: Linear<R, E>,
    /// `Linear(d, d)` pooled-mean term.
    mean: Linear<R, E>,
    /// `Linear(d, d)` output projection of the residual update.
    out: Linear<R, E>,
    /// Per-round normalisation.
    norm: RmsNorm<R, E>,
}

impl<R: Runtime, E: FloatElem> Module<R, E> for FingerprintRound<R, E> {
    fn visit(&self, visitor: &mut ModuleVisitor<'_, R, E>) {
        visitor.child("own", &self.own);
        visitor.child("mean", &self.mean);
        visitor.child("out", &self.out);
        visitor.child("norm", &self.norm);
    }
}

/// Permutation-invariant encoder of the fingerprint token set.
///
/// Token embeddings (`bit_emb` plus `conf_emb`) are refined by two shared
/// set-mixing rounds `h = h + out(silu(own(h) + mean(masked mean of h)))`,
/// each followed by RmsNorm and selection to exact zeros in padding, so
/// in-range padded ids cannot reach any output bit. There is no position
/// embedding: the output is invariant to the order of the tokens. The pooled
/// vector is the masked mean of the token states (exactly zero when no slot
/// is valid, as in the substructure encoder).
pub struct FingerprintEncoder<R: Runtime, E: FloatElem> {
    /// `[4097, d]` bit table (row 0 is padding, rows `1 + bit`).
    bit_emb: Param<R, E>,
    /// `[9, d]` confidence-bucket table (row 0 is padding, rows `1..=8`).
    conf_emb: Param<R, E>,
    /// Set-mixing rounds (always [`FINGERPRINT_ROUNDS`]).
    rounds: Vec<FingerprintRound<R, E>>,
    /// `Linear(d, d)` projection of the pooled fingerprint vector into the
    /// composition row network.
    pool_in: Linear<R, E>,
    /// Residual width.
    d_model: usize,
}

impl<R: Runtime, E: FloatElem> Module<R, E> for FingerprintEncoder<R, E> {
    fn visit(&self, visitor: &mut ModuleVisitor<'_, R, E>) {
        visitor.param("bit_emb", &self.bit_emb);
        visitor.param("conf_emb", &self.conf_emb);
        for (r, round) in self.rounds.iter().enumerate() {
            visitor.child_at("round", r, round);
        }
        visitor.child("pool_in", &self.pool_in);
    }
}

impl<R: Runtime, E: FloatElem> FingerprintEncoder<R, E> {
    /// Build the encoder for width `d`.
    pub fn init(d_model: usize, device: &Device<R>, rng: &mut Rng) -> Self {
        let bit_emb = Param::new(
            Initializer::Normal {
                mean: 0.0,
                std: 0.02,
            }
            .init(vec![FINGERPRINT_TOKEN_ROWS, d_model], device, rng),
        );
        let conf_emb = Param::new(
            Initializer::Normal {
                mean: 0.0,
                std: 0.02,
            }
            .init(vec![FINGERPRINT_BUCKET_ROWS, d_model], device, rng),
        );
        let mut rounds = Vec::with_capacity(FINGERPRINT_ROUNDS);
        for _ in 0..FINGERPRINT_ROUNDS {
            rounds.push(FingerprintRound {
                own: LinearConfig::new(d_model, d_model).init(device, rng),
                mean: LinearConfig::new(d_model, d_model).init(device, rng),
                out: LinearConfig::new(d_model, d_model).init(device, rng),
                norm: RmsNormConfig::new(d_model).init(device, rng),
            });
        }
        Self {
            bit_emb,
            conf_emb,
            rounds,
            pool_in: LinearConfig::new(d_model, d_model).init(device, rng),
            d_model,
        }
    }

    /// Encode `batch` into token states `[B, S, d]`: `h = lookup(bit) +
    /// lookup(bucket)`, selected to exact zeros in padding, then the two
    /// mixing rounds. No device read.
    pub fn encode_states(
        &self,
        batch: &FingerprintBatch,
        device: &Device<R>,
    ) -> Result<(Var<R, E>, Tensor<R, E>)> {
        batch.validate()?;
        let uploaded = batch.upload(device)?;
        let (b, s, d) = (batch.queries, batch.slots, self.d_model);
        if b == 0 || s == 0 {
            let h = Var::constant(Tensor::<R, E>::zeros(vec![b, s, d], device));
            let valid = Tensor::<R, E>::zeros(vec![b, s], device);
            return Ok((h, valid));
        }
        let flat = b * s;
        let bit_ids = uploaded.token_ids.reshape(vec![flat])?;
        let conf_ids = uploaded.bucket_ids.reshape(vec![flat])?;
        let mut h = Var::ms2_lookup(&self.bit_emb.var_standalone(), &bit_ids)?
            .reshape(vec![b, s, d])?
            .add(
                &Var::ms2_lookup(&self.conf_emb.var_standalone(), &conf_ids)?
                    .reshape(vec![b, s, d])?,
            )?;
        h = h.ms2_select_valid(&uploaded.valid)?;
        for round in &self.rounds {
            let own_h = round.own.apply(&h)?;
            // Masked mean exactly as the substructure encoder does it, so the
            // mean part is exactly zero when no slot is valid.
            let mean = pooled_mean(&h, &uploaded.valid, device)?;
            let broadcast = round
                .mean
                .apply(&mean)?
                .unsqueeze(1)?
                .expand(vec![b, s, d])?;
            let update = round.out.apply(&own_h.add(&broadcast)?.silu()?)?;
            h = h.add(&update)?;
            h = round.norm.apply(&h)?.ms2_select_valid(&uploaded.valid)?;
        }
        Ok((h, uploaded.valid))
    }

    /// Pooled fingerprint vector of `batch`: the masked mean of the token
    /// states through [`pool_in`](FingerprintEncoder::pool_in) (`Linear(d,
    /// d)`), exactly zero rows when no slot is valid. No device read.
    pub fn encode_pooled(&self, batch: &FingerprintBatch, device: &Device<R>) -> Result<Var<R, E>> {
        let (h, valid) = self.encode_states(batch, device)?;
        self.encode_pooled_from_states(&h, &valid, device)
    }

    /// Pooled vector from precomputed states (see
    /// [`encode_pooled`](Self::encode_pooled)): the masked mean of `h` over
    /// `valid` through `pool_in`, with rows of queries with no valid slot
    /// selected back to exact zero (`pool_in` has a bias, so the projection
    /// of the zero mean would not be zero without this). No device read.
    pub(crate) fn encode_pooled_from_states(
        &self,
        h: &Var<R, E>,
        valid: &Tensor<R, E>,
        device: &Device<R>,
    ) -> Result<Var<R, E>> {
        let mean = pooled_mean(h, valid, device)?;
        let pooled = self.pool_in.apply(&mean)?;
        // Row mask: 1 where the query holds a valid slot, exact 0 where it
        // holds none (`len / max(len, 1)` is exactly 0 or 1). This keeps the
        // documented exact-zero empty pool whatever the learned bias is, and
        // stays on the composition tape (elementwise ops on the pooled var).
        let b = valid.shape().dim(0);
        let len_t = crate::tensor::ops::reduce::sum_dim(valid, 1)?.reshape(vec![b, 1])?;
        let len_var = Var::constant(len_t);
        let one_var = Var::constant(Tensor::<R, E>::ones(vec![b, 1], device));
        let den = len_var.maximum(&one_var)?;
        let mask = len_var.div(&den)?.expand(vec![b, self.d_model])?;
        pooled.mul(&mask)
    }
}

/// Masked mean of `h` (`[B, S, d]`) over `valid` (`[B, S]`): `sum / max(len,
/// 1)`, squeezed to `[B, d]` (exactly zero rows when no slot is valid).
pub(crate) fn pooled_mean<R: Runtime, E: FloatElem>(
    h: &Var<R, E>,
    valid: &Tensor<R, E>,
    device: &Device<R>,
) -> Result<Var<R, E>> {
    let x_sum = h.sum_dim(1)?;
    let b = valid.shape().dim(0);
    let len_t = crate::tensor::ops::reduce::sum_dim(valid, 1)?.reshape(vec![b, 1, 1])?;
    let len_var = Var::constant(len_t);
    let one_var = Var::constant(Tensor::<R, E>::ones(vec![b, 1, 1], device));
    let den = len_var.maximum(&one_var)?;
    let mean = x_sum.div(&den)?.squeeze(1)?;
    Ok(mean)
}

/// How the fingerprint evidence of a training query is produced.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FingerprintMode {
    /// Exact true bits at probability 1.
    Exact,
    /// MIST-like synthetic predictions from [`FingerprintNoise`].
    #[default]
    MistLike,
}

impl FingerprintMode {
    /// The CLI / config spelling.
    pub fn as_str(self) -> &'static str {
        match self {
            FingerprintMode::Exact => "exact",
            FingerprintMode::MistLike => "mist_like",
        }
    }

    /// Parse a spelling; `None` for anything else.
    pub fn parse(text: &str) -> Option<Self> {
        match text {
            "exact" => Some(FingerprintMode::Exact),
            "mist_like" => Some(FingerprintMode::MistLike),
            _ => None,
        }
    }
}

/// True on-bit lists by molecule index, from the `bits` subcommand.
///
/// Loads `fp_morgan4096_*.json`: `bits_by_molecule[i]` is the sorted on-bit
/// list of export molecule `i`. Keyed `bits` are ignored (they lose
/// tautomer variants sharing one key). [`load`](FingerprintStore::load)
/// requires `bits_by_molecule`; [`assert_molecule_count`] checks the file's
/// molecule count equals the export's. V2 bits files also carry
/// `keys_by_molecule` (the export's `<key>|<identity_group>` per row, in
/// order); [`assert_keys_match`](FingerprintStore::assert_keys_match) binds
/// the sidecar to the export order, so a reordered same-length sidecar is a
/// loud error instead of silent misattribution (pre-v2 files without the
/// list skip the check).
#[derive(Clone, Debug)]
pub struct FingerprintStore {
    /// Sorted on-bit lists aligned with the export molecule order.
    bits_by_molecule: Vec<Vec<u16>>,
    /// Export entry keys in order, when the file carries them.
    keys_by_molecule: Option<Vec<String>>,
}

impl FingerprintStore {
    /// Load a bits file. Requires `fingerprint == "morgan4096"` and a
    /// `bits_by_molecule` list; every list must be sorted with entries below
    /// 4096. Anything else is [`Error::Config`].
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        let text = std::str::from_utf8(&bytes).map_err(|e| {
            Error::config(format!(
                "FingerprintStore::load: {} is not valid UTF-8: {e}",
                path.display()
            ))
        })?;
        Self::load_json(text)
    }

    /// Load from already-read JSON text (see [`load`](FingerprintStore::load)).
    pub fn load_json(text: &str) -> Result<Self> {
        let raw: serde_json::Value = serde_json::from_str(text)?;
        let name = raw
            .get("fingerprint")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if name != "morgan4096" {
            return Err(Error::config(format!(
                "FingerprintStore::load: fingerprint {name:?} is not \"morgan4096\""
            )));
        }
        let lists = raw.get("bits_by_molecule").ok_or_else(|| {
            Error::config(
                "FingerprintStore::load: missing bits_by_molecule (regenerate the bits file with tools/ms2/export_fingerprints_mist.py)".to_string(),
            )
        })?;
        let arrays = lists.as_array().ok_or_else(|| {
            Error::config("FingerprintStore::load: bits_by_molecule is not a list".to_string())
        })?;
        let mut bits_by_molecule = Vec::with_capacity(arrays.len());
        for (i, entry) in arrays.iter().enumerate() {
            let list = entry.as_array().ok_or_else(|| {
                Error::config(format!(
                    "FingerprintStore::load: bits_by_molecule[{i}] is not a list"
                ))
            })?;
            let mut bits = Vec::with_capacity(list.len());
            for v in list {
                let n = v.as_u64().ok_or_else(|| {
                    Error::config(format!(
                        "FingerprintStore::load: bits_by_molecule[{i}] holds a non-integer"
                    ))
                })?;
                if n >= FINGERPRINT_BITS as u64 {
                    return Err(Error::config(format!(
                        "FingerprintStore::load: bits_by_molecule[{i}] bit {n} is past {FINGERPRINT_BITS}"
                    )));
                }
                bits.push(n as u16);
            }
            if bits.windows(2).any(|w| w[0] >= w[1]) && !bits.is_empty() {
                let mut sorted = bits.clone();
                sorted.sort_unstable();
                sorted.dedup();
                if sorted != bits {
                    return Err(Error::config(format!(
                        "FingerprintStore::load: bits_by_molecule[{i}] is not sorted unique"
                    )));
                }
            }
            bits_by_molecule.push(bits);
        }
        let keys_by_molecule =
            match raw.get("keys_by_molecule") {
                None => None,
                Some(list) => {
                    let array = list.as_array().ok_or_else(|| {
                        Error::config(
                            "FingerprintStore::load: keys_by_molecule is not a list".to_string(),
                        )
                    })?;
                    let mut keys = Vec::with_capacity(array.len());
                    for (i, v) in array.iter().enumerate() {
                        keys.push(v.as_str().ok_or_else(|| {
                        Error::config(format!(
                            "FingerprintStore::load: keys_by_molecule[{i}] is not a string"
                        ))
                    })?.to_string());
                    }
                    Some(keys)
                }
            };
        if let Some(keys) = &keys_by_molecule
            && keys.len() != bits_by_molecule.len()
        {
            return Err(Error::config(format!(
                "FingerprintStore::load: keys_by_molecule holds {} keys for {} molecules",
                keys.len(),
                bits_by_molecule.len()
            )));
        }
        Ok(Self {
            bits_by_molecule,
            keys_by_molecule,
        })
    }

    /// Molecules in the file.
    pub fn len(&self) -> usize {
        self.bits_by_molecule.len()
    }

    /// Whether the file holds no molecule.
    pub fn is_empty(&self) -> bool {
        self.bits_by_molecule.is_empty()
    }

    /// True on-bit list of export molecule `index` ([`Error::Config`] when
    /// out of range).
    pub fn get_by_index(&self, index: usize) -> Result<&[u16]> {
        self.bits_by_molecule
            .get(index)
            .map(Vec::as_slice)
            .ok_or_else(|| {
                Error::config(format!(
                    "FingerprintStore::get_by_index: molecule index {index} outside {} molecules",
                    self.bits_by_molecule.len()
                ))
            })
    }

    /// Assert the file's molecule count equals the export's (`expected`).
    pub fn assert_molecule_count(&self, expected: usize) -> Result<()> {
        if self.bits_by_molecule.len() != expected {
            return Err(Error::config(format!(
                "FingerprintStore::assert_molecule_count: bits file holds {} molecules for an export of {expected}",
                self.bits_by_molecule.len()
            )));
        }
        Ok(())
    }

    /// Assert the sidecar's ordered entry keys equal the export's (see the
    /// struct docs): a reordered same-length sidecar is an error. Files
    /// without `keys_by_molecule` skip the check.
    pub fn assert_keys_match(&self, expected: &[String]) -> Result<()> {
        let Some(keys) = &self.keys_by_molecule else {
            return Ok(());
        };
        if keys.len() != expected.len() {
            return Err(Error::config(format!(
                "FingerprintStore::assert_keys_match: bits file holds {} keys for an export of {}",
                keys.len(),
                expected.len()
            )));
        }
        for (i, (got, want)) in keys.iter().zip(expected.iter()).enumerate() {
            if got != want {
                return Err(Error::config(format!(
                    "FingerprintStore::assert_keys_match: bits entry {i} is for {got:?} but the export holds {want:?} there (reordered sidecar?)"
                )));
            }
        }
        Ok(())
    }

    /// Whether the file carries ordered entry keys.
    pub fn has_keys(&self) -> bool {
        self.keys_by_molecule.is_some()
    }
}

/// MIST prediction noise: the `fp_mist_noise.json` histograms behind
/// MIST-like synthetic fingerprints.
///
/// Uses exactly: `hist_pred_given_true_on` (20 counts over `[0, 1]`),
/// `hist_pred_given_true_off` (20 counts), `n_spectra` (the per-spectrum
/// false-token divisor) and `n_bins` (20); the v2 file additionally carries
/// `hist_pred_given_true_on_molecule` / `hist_pred_given_true_off_molecule`
/// (same bins over the per-molecule averaged rows), selected with
/// [`FingerprintNoiseLevel::Molecule`]. Pre-v2 files without the molecule
/// histograms load with those sets cloned from the spectrum ones
/// ([`has_molecule_histograms`](FingerprintNoise::has_molecule_histograms)
/// is then false). See the module docs.
#[derive(Clone, Debug)]
pub struct FingerprintNoise {
    /// Counts of predicted probabilities over the 20 bins for true-on bits.
    pub hist_on: [u64; FINGERPRINT_NOISE_BINS],
    /// Counts for true-off bits.
    pub hist_off: [u64; FINGERPRINT_NOISE_BINS],
    /// Counts over the per-molecule averaged rows for true-on bits.
    pub hist_on_molecule: [u64; FINGERPRINT_NOISE_BINS],
    /// Counts over the per-molecule averaged rows for true-off bits.
    pub hist_off_molecule: [u64; FINGERPRINT_NOISE_BINS],
    /// Whether the file carried its own molecule histograms (false for the
    /// pre-v2 clone fallback).
    pub has_molecule_histograms: bool,
    /// Spectra behind the histograms (the false-token divisor).
    pub n_spectra: u64,
    /// Molecules behind the histograms (provenance only).
    pub n_molecules: u64,
}

impl FingerprintNoise {
    /// Load `fp_mist_noise.json`: requires `fingerprint == "morgan4096"`,
    /// `n_bins == 20`, two 20-entry histograms and a positive `n_spectra`.
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        let text = std::str::from_utf8(&bytes).map_err(|e| {
            Error::config(format!(
                "FingerprintNoise::load: {} is not valid UTF-8: {e}",
                path.display()
            ))
        })?;
        Self::load_json(text)
    }

    /// Load from already-read JSON text (see [`load`](FingerprintNoise::load)).
    pub fn load_json(text: &str) -> Result<Self> {
        let raw: serde_json::Value = serde_json::from_str(text)?;
        let name = raw
            .get("fingerprint")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if name != "morgan4096" {
            return Err(Error::config(format!(
                "FingerprintNoise::load: fingerprint {name:?} is not \"morgan4096\""
            )));
        }
        let bins = raw
            .get("n_bins")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| Error::config("FingerprintNoise::load: missing n_bins".to_string()))?;
        if bins as usize != FINGERPRINT_NOISE_BINS {
            return Err(Error::config(format!(
                "FingerprintNoise::load: n_bins {bins} is not {}",
                FINGERPRINT_NOISE_BINS
            )));
        }
        let read_hist = |key: &str| -> Result<[u64; FINGERPRINT_NOISE_BINS]> {
            let list = raw
                .get(key)
                .ok_or_else(|| Error::config(format!("FingerprintNoise::load: missing {key}")))?;
            let array = list.as_array().ok_or_else(|| {
                Error::config(format!("FingerprintNoise::load: {key} is not a list"))
            })?;
            if array.len() != FINGERPRINT_NOISE_BINS {
                return Err(Error::config(format!(
                    "FingerprintNoise::load: {key} holds {} bins (needs {})",
                    array.len(),
                    FINGERPRINT_NOISE_BINS
                )));
            }
            let mut out = [0u64; FINGERPRINT_NOISE_BINS];
            for (i, v) in array.iter().enumerate() {
                out[i] = v.as_u64().ok_or_else(|| {
                    Error::config(format!(
                        "FingerprintNoise::load: {key}[{i}] is not a non-negative integer"
                    ))
                })?;
            }
            Ok(out)
        };
        let hist_on = read_hist("hist_pred_given_true_on")?;
        let hist_off = read_hist("hist_pred_given_true_off")?;
        // Molecule-averaged histograms (v2 files); pre-v2 files fall back to
        // the spectrum sets (flagged, so runs can refuse `--fp-noise-level
        // molecule` instead of silently sampling the wrong distribution).
        let read_hist_opt = |key: &str| -> Result<Option<[u64; FINGERPRINT_NOISE_BINS]>> {
            let Some(list) = raw.get(key) else {
                return Ok(None);
            };
            let array = list.as_array().ok_or_else(|| {
                Error::config(format!("FingerprintNoise::load: {key} is not a list"))
            })?;
            if array.len() != FINGERPRINT_NOISE_BINS {
                return Err(Error::config(format!(
                    "FingerprintNoise::load: {key} holds {} bins (needs {})",
                    array.len(),
                    FINGERPRINT_NOISE_BINS
                )));
            }
            let mut out = [0u64; FINGERPRINT_NOISE_BINS];
            for (i, v) in array.iter().enumerate() {
                out[i] = v.as_u64().ok_or_else(|| {
                    Error::config(format!(
                        "FingerprintNoise::load: {key}[{i}] is not a non-negative integer"
                    ))
                })?;
            }
            Ok(Some(out))
        };
        let hist_on_molecule_opt = read_hist_opt("hist_pred_given_true_on_molecule")?;
        let hist_off_molecule_opt = read_hist_opt("hist_pred_given_true_off_molecule")?;
        let has_molecule_histograms =
            hist_on_molecule_opt.is_some() && hist_off_molecule_opt.is_some();
        let hist_on_molecule = hist_on_molecule_opt.unwrap_or(hist_on);
        let hist_off_molecule = hist_off_molecule_opt.unwrap_or(hist_off);
        let n_spectra = raw
            .get("n_spectra")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| {
                Error::config("FingerprintNoise::load: missing n_spectra".to_string())
            })?;
        if n_spectra == 0 {
            return Err(Error::config(
                "FingerprintNoise::load: n_spectra is 0".to_string(),
            ));
        }
        let n_molecules = raw.get("n_molecules").and_then(|v| v.as_u64()).unwrap_or(0);
        if hist_on.iter().sum::<u64>() == 0 || hist_off.iter().sum::<u64>() == 0 {
            return Err(Error::config(
                "FingerprintNoise::load: an empty histogram cannot be sampled".to_string(),
            ));
        }
        Ok(Self {
            hist_on,
            hist_off,
            hist_on_molecule,
            hist_off_molecule,
            has_molecule_histograms,
            n_spectra,
            n_molecules,
        })
    }

    /// Histograms behind `level`.
    fn hists_at(
        &self,
        level: FingerprintNoiseLevel,
    ) -> (
        &[u64; FINGERPRINT_NOISE_BINS],
        &[u64; FINGERPRINT_NOISE_BINS],
    ) {
        match level {
            FingerprintNoiseLevel::Spectrum => (&self.hist_on, &self.hist_off),
            FingerprintNoiseLevel::Molecule => (&self.hist_on_molecule, &self.hist_off_molecule),
        }
    }

    /// Surviving fraction of bin `bin` at `threshold` under the `p >=
    /// threshold` convention with bin width `1/20`: sampling draws uniform
    /// probabilities within the bin and keeps `p >= threshold`, so a bin the
    /// threshold cuts through keeps its surviving fraction (at 0.125, half
    /// of bin `[0.10, 0.15)` survives) instead of being dropped whole.
    fn bin_survival(bin: usize, threshold: f32) -> f64 {
        let width = 1.0f64 / FINGERPRINT_NOISE_BINS as f64;
        let lo = bin as f64 * width;
        let hi = lo + width;
        let t = threshold as f64;
        if t <= lo {
            1.0
        } else if t >= hi {
            0.0
        } else {
            (hi - t) / width
        }
    }

    /// Expected surviving mass of `hist` at `threshold`: `sum(hist[bin] *
    /// survival(bin, threshold))`.
    fn kept_mass(hist: &[u64; FINGERPRINT_NOISE_BINS], threshold: f32) -> f64 {
        hist.iter()
            .enumerate()
            .map(|(bin, &count)| count as f64 * Self::bin_survival(bin, threshold))
            .sum()
    }

    /// Expected true-bit retention rate at `threshold` and `level`:
    /// surviving on-mass over total on-mass (the intersected bin contributes
    /// its surviving fraction).
    pub fn retention_rate_at_level(&self, threshold: f32, level: FingerprintNoiseLevel) -> f64 {
        let (on, _) = self.hists_at(level);
        let total: u64 = on.iter().sum();
        if total == 0 {
            return 0.0;
        }
        Self::kept_mass(on, threshold) / total as f64
    }

    /// Expected true-bit retention rate at `threshold` (the spectrum set;
    /// see [`retention_rate_at_level`](Self::retention_rate_at_level)).
    pub fn retention_rate(&self, threshold: f32) -> f64 {
        self.retention_rate_at_level(threshold, FingerprintNoiseLevel::Spectrum)
    }

    /// Off survival probability at `threshold` and `level`: surviving
    /// off-mass over total off-mass (one true-off bit's keep probability).
    pub fn off_survival_at_level(&self, threshold: f32, level: FingerprintNoiseLevel) -> f64 {
        let (_, off) = self.hists_at(level);
        let total: u64 = off.iter().sum();
        if total == 0 {
            return 0.0;
        }
        Self::kept_mass(off, threshold) / total as f64
    }

    /// Expected false tokens per spectrum at `threshold` (the spectrum set):
    /// surviving off-mass over `n_spectra` (the corpus average; one query
    /// with `t` true bits expects
    /// [`expected_false_tokens`](Self::expected_false_tokens) instead).
    pub fn mean_false_tokens(&self, threshold: f32) -> f64 {
        self.mean_false_tokens_at_level(threshold, FingerprintNoiseLevel::Spectrum)
    }

    /// Corpus-average false tokens at `threshold` and `level`: surviving
    /// off-mass over `n_spectra`.
    pub fn mean_false_tokens_at_level(&self, threshold: f32, level: FingerprintNoiseLevel) -> f64 {
        let (_, off) = self.hists_at(level);
        Self::kept_mass(off, threshold) / self.n_spectra as f64
    }

    /// Expected false tokens of one query with `true_count` true bits at
    /// `threshold` and `level`: `(4096 - true_count)` times the off survival
    /// probability (not the corpus average, which folds in the corpus mean
    /// true count).
    pub fn expected_false_tokens(
        &self,
        true_count: usize,
        threshold: f32,
        level: FingerprintNoiseLevel,
    ) -> f64 {
        (FINGERPRINT_BITS - true_count.min(FINGERPRINT_BITS)) as f64
            * self.off_survival_at_level(threshold, level)
    }

    /// Sample one MIST-like fingerprint from `true_bits` (the spectrum
    /// histogram set; see [`sample_at_level`](Self::sample_at_level)).
    ///
    /// Every true-on bit draws one probability from `hist_pred_given_true_on`
    /// (bin proportional to its count, uniform within the bin); every
    /// true-off bit (each of the `4096 - true.len()` bits) independently
    /// draws one probability from `hist_pred_given_true_off` the same way.
    /// Entries with `p < threshold` are dropped; the rest (sorted by index)
    /// is the fingerprint. Deterministic in (`seed`, `key`, `draw`) through
    /// [`mix_fp_seed`]: the same arguments always give the same fingerprint.
    /// `threshold` must be in `(0, 1]`.
    pub fn sample(
        &self,
        true_bits: &[u16],
        seed: u64,
        key: &str,
        draw: u64,
        threshold: f32,
    ) -> Result<SparseFingerprint> {
        self.sample_at_level(
            true_bits,
            seed,
            key,
            draw,
            threshold,
            FingerprintNoiseLevel::Spectrum,
        )
    }

    /// Sample one MIST-like fingerprint from `true_bits` with the histogram
    /// set behind `level`.
    ///
    /// Same draws as [`sample`](Self::sample) but from the level's
    /// histograms (see [`hists_at`](Self::hists_at)). `threshold` must be in
    /// `(0, 1]`.
    pub fn sample_at_level(
        &self,
        true_bits: &[u16],
        seed: u64,
        key: &str,
        draw: u64,
        threshold: f32,
        level: FingerprintNoiseLevel,
    ) -> Result<SparseFingerprint> {
        if !(threshold.is_finite() && threshold > 0.0 && threshold <= 1.0) {
            return Err(Error::config(format!(
                "FingerprintNoise::sample: threshold {threshold} is not in (0, 1]"
            )));
        }
        for &bit in true_bits {
            if usize::from(bit) >= FINGERPRINT_BITS {
                return Err(Error::config(format!(
                    "FingerprintNoise::sample: true bit {bit} is past {FINGERPRINT_BITS}"
                )));
            }
        }
        let (hist_on, hist_off) = self.hists_at(level);
        let mut rng = SplitMix64::new(mix_fp_seed(seed, key, draw));
        let mut sorted_true = true_bits.to_vec();
        sorted_true.sort_unstable();
        sorted_true.dedup();
        let mut pairs: Vec<(u16, f32)> = Vec::new();
        for &bit in &sorted_true {
            let prob = draw_prob(&mut rng, hist_on);
            if prob >= threshold {
                pairs.push((bit, prob));
            }
        }
        let in_true: HashSet<u16> = sorted_true.into_iter().collect();
        for bit in 0..FINGERPRINT_BITS as u32 {
            let bit = bit as u16;
            if in_true.contains(&bit) {
                continue;
            }
            let prob = draw_prob(&mut rng, hist_off);
            if prob >= threshold {
                pairs.push((bit, prob));
            }
        }
        pairs.sort_by_key(|&(bit, _)| bit);
        let out = SparseFingerprint { entries: pairs };
        out.validate()?;
        Ok(out)
    }
}

/// Fold `seed`, a stable hash of `key` and `draw` into one generator seed:
/// each value goes through a full SplitMix64 round (mirroring
/// `completion_data::mix_seed`, which is private to that module), so nearby
/// seeds still spread.
pub(crate) fn mix_fp_seed(seed: u64, key: &str, draw: u64) -> u64 {
    let mut first = SplitMix64::new(seed);
    let a = first.next();
    let mut second = SplitMix64::new(a.wrapping_add(stable_hash(&[key])));
    let b = second.next();
    let mut third = SplitMix64::new(b.wrapping_add(draw));
    third.next()
}

/// Draw one probability from a 20-bin histogram: bin proportional to its
/// count (rejection-free via [`below`](SplitMix64::below)), then uniform
/// within `[i/20, (i+1)/20)` (clamped to `[0, 1]`).
fn draw_prob(rng: &mut SplitMix64, hist: &[u64; FINGERPRINT_NOISE_BINS]) -> f32 {
    let total: u64 = hist.iter().sum();
    debug_assert!(total > 0);
    let mut roll = rng.below(total);
    let mut bin = 0usize;
    for (i, &count) in hist.iter().enumerate() {
        if roll < count {
            bin = i;
            break;
        }
        roll -= count;
    }
    let width = 1.0f64 / FINGERPRINT_NOISE_BINS as f64;
    let lo = bin as f64 * width;
    let u = rng.next() as f64 / u64::MAX as f64;
    let mut prob = lo + u * width;
    if prob < 0.0 {
        prob = 0.0;
    }
    if prob > 1.0 {
        prob = 1.0;
    }
    // Avoid exact 0 (outside the `(0, 1]` fingerprint domain): redraws are
    // unnecessary, the smallest positive f32 is indistinguishable here.
    if prob <= 0.0 {
        prob = f32::MIN_POSITIVE as f64;
    }
    prob as f32
}

/// Format tag of a [`FingerprintChannel`] file.
pub const FINGERPRINT_CHANNEL_FORMAT: &str = "fingerprint_channel_v1";
/// Outcomes per channel row: "no token" followed by the eight buckets.
pub const FINGERPRINT_CHANNEL_OUTCOMES: usize = FINGERPRINT_BUCKETS + 1;

/// Per-bit error channel of a fingerprint predictor, fitted on real
/// predictions by `tools/ms2/fit_fingerprint_channel.py`.
///
/// [`FingerprintNoise`] draws every bit from one pooled histogram, so it
/// reproduces how often the predictor is right but not which bits it is
/// right about. The channel keeps one distribution per bit, truth value and
/// latent quality class: `P(outcome | truth, bit, class)`, where outcome 0
/// is "no token" (the prediction is below the token threshold) and outcomes
/// `1..=8` are the confidence buckets of [`SparseFingerprint::bucket`]. A
/// sample first draws one class for the whole molecule (weights `weights`),
/// which is what makes a poorly predicted molecule poor across its bits,
/// then draws every bit independently from that class's rows.
///
/// A sampled token's probability is drawn uniformly inside its bucket and
/// never below the file's threshold, so it lands in the bucket the row named
/// and survives [`SparseFingerprint::from_probabilities`] at that threshold.
#[derive(Clone, Debug)]
pub struct FingerprintChannel {
    /// Latent quality classes.
    pub classes: usize,
    /// Token threshold the channel was fitted at.
    pub threshold: f32,
    /// Class weights, summing to 1.
    pub weights: Vec<f64>,
    /// `[class][bit][outcome]` cumulative probabilities for true-off bits.
    off: Vec<f32>,
    /// `[class][bit][outcome]` cumulative probabilities for true-on bits.
    on: Vec<f32>,
}

impl FingerprintChannel {
    /// Load a `fingerprint_channel_v1` file (see [`load_json`](Self::load_json)).
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        let text = std::str::from_utf8(&bytes).map_err(|e| {
            Error::config(format!(
                "FingerprintChannel::load: {} is not valid UTF-8: {e}",
                path.display()
            ))
        })?;
        Self::load_json(text)
    }

    /// Load from JSON text: requires `format == "fingerprint_channel_v1"`,
    /// `fingerprint == "morgan4096"`, `buckets == 8`, a threshold in
    /// `(0, 1/8]` (so a token of bucket 1 can sit at or above it), positive
    /// weights summing to 1 and `on` / `off` tables of `classes x 4096 x 9`
    /// non-negative rows that each sum to 1 within `1e-3`. Anything else is
    /// [`Error::Config`].
    pub fn load_json(text: &str) -> Result<Self> {
        let raw: serde_json::Value = serde_json::from_str(text)?;
        let bad = |what: String| Error::config(format!("FingerprintChannel::load: {what}"));
        let format = raw.get("format").and_then(|v| v.as_str()).unwrap_or("");
        if format != FINGERPRINT_CHANNEL_FORMAT {
            return Err(bad(format!(
                "format {format:?} is not {FINGERPRINT_CHANNEL_FORMAT:?}"
            )));
        }
        let name = raw
            .get("fingerprint")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if name != "morgan4096" {
            return Err(bad(format!("fingerprint {name:?} is not \"morgan4096\"")));
        }
        let buckets = raw.get("buckets").and_then(|v| v.as_u64()).unwrap_or(0);
        if buckets as usize != FINGERPRINT_BUCKETS {
            return Err(bad(format!(
                "buckets {buckets} is not {FINGERPRINT_BUCKETS}"
            )));
        }
        let threshold = raw
            .get("threshold")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| bad("missing threshold".to_string()))? as f32;
        if !(threshold.is_finite()
            && threshold > 0.0
            && threshold <= 1.0 / FINGERPRINT_BUCKETS as f32)
        {
            return Err(bad(format!(
                "threshold {threshold} is not in (0, 1/{FINGERPRINT_BUCKETS}]"
            )));
        }
        let weights: Vec<f64> = raw
            .get("weights")
            .and_then(|v| v.as_array())
            .ok_or_else(|| bad("missing weights".to_string()))?
            .iter()
            .map(|v| v.as_f64().unwrap_or(f64::NAN))
            .collect();
        let classes = weights.len();
        let declared = raw.get("classes").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        if classes == 0 || classes != declared {
            return Err(bad(format!(
                "{classes} weights for {declared} declared classes"
            )));
        }
        let total: f64 = weights.iter().sum();
        if weights.iter().any(|w| !(w.is_finite() && *w > 0.0)) || (total - 1.0).abs() > 1e-3 {
            return Err(bad(format!(
                "weights must be positive and sum to 1 (sum {total})"
            )));
        }
        let table = |field: &str| -> Result<Vec<f32>> {
            let per_class = raw
                .get(field)
                .and_then(|v| v.as_array())
                .ok_or_else(|| bad(format!("missing {field} table")))?;
            if per_class.len() != classes {
                return Err(bad(format!(
                    "{field} holds {} classes for {classes} weights",
                    per_class.len()
                )));
            }
            let mut out =
                Vec::with_capacity(classes * FINGERPRINT_BITS * FINGERPRINT_CHANNEL_OUTCOMES);
            for (k, class) in per_class.iter().enumerate() {
                let rows = class
                    .as_array()
                    .filter(|rows| rows.len() == FINGERPRINT_BITS)
                    .ok_or_else(|| {
                        bad(format!("{field}[{k}] is not {FINGERPRINT_BITS} rows"))
                    })?;
                for (bit, row) in rows.iter().enumerate() {
                    let row = row
                        .as_array()
                        .filter(|row| row.len() == FINGERPRINT_CHANNEL_OUTCOMES)
                        .ok_or_else(|| {
                            bad(format!(
                                "{field}[{k}][{bit}] is not {FINGERPRINT_CHANNEL_OUTCOMES} outcomes"
                            ))
                        })?;
                    let mut sum = 0.0f64;
                    let start = out.len();
                    for value in row {
                        let p = value.as_f64().unwrap_or(f64::NAN);
                        if !(p.is_finite() && p >= 0.0) {
                            return Err(bad(format!(
                                "{field}[{k}][{bit}] holds {value}, not a probability"
                            )));
                        }
                        sum += p;
                        out.push(sum as f32);
                    }
                    if (sum - 1.0).abs() > 1e-3 {
                        return Err(bad(format!("{field}[{k}][{bit}] sums to {sum}")));
                    }
                    // Normalised cumulative row: the last entry is exactly 1.
                    for value in &mut out[start..] {
                        *value = (f64::from(*value) / sum) as f32;
                    }
                    out[start + FINGERPRINT_CHANNEL_OUTCOMES - 1] = 1.0;
                }
            }
            Ok(out)
        };
        let off = table("off")?;
        let on = table("on")?;
        Ok(Self {
            classes,
            threshold,
            weights: weights.iter().map(|w| w / total).collect(),
            off,
            on,
        })
    }

    /// `P(outcome | truth, bit, class)`: outcome 0 is "no token", `1..=8`
    /// the bucket. Out-of-range arguments give 0.
    pub fn probability(&self, class: usize, bit: usize, on: bool, outcome: usize) -> f32 {
        if class >= self.classes
            || bit >= FINGERPRINT_BITS
            || outcome >= FINGERPRINT_CHANNEL_OUTCOMES
        {
            return 0.0;
        }
        let row = self.row(class, bit, on);
        if outcome == 0 {
            row[0]
        } else {
            row[outcome] - row[outcome - 1]
        }
    }

    fn row(&self, class: usize, bit: usize, on: bool) -> &[f32] {
        let table = if on { &self.on } else { &self.off };
        let start = (class * FINGERPRINT_BITS + bit) * FINGERPRINT_CHANNEL_OUTCOMES;
        &table[start..start + FINGERPRINT_CHANNEL_OUTCOMES]
    }

    /// Expected tokens and expected true tokens of one sample from
    /// `true_bits`, averaged over the classes: the channel's own precision
    /// (`true / tokens`) and recall (`true / true_bits.len()`) for that
    /// molecule, for reporting.
    pub fn expected_tokens(&self, true_bits: &[u16]) -> (f64, f64) {
        let mut is_on = vec![false; FINGERPRINT_BITS];
        for &bit in true_bits {
            if usize::from(bit) < FINGERPRINT_BITS {
                is_on[usize::from(bit)] = true;
            }
        }
        let (mut tokens, mut hits) = (0.0f64, 0.0f64);
        for (class, &weight) in self.weights.iter().enumerate() {
            for (bit, &on) in is_on.iter().enumerate() {
                let kept = 1.0 - f64::from(self.row(class, bit, on)[0]);
                tokens += weight * kept;
                if on {
                    hits += weight * kept;
                }
            }
        }
        (tokens, hits)
    }

    /// Sample one predicted-looking fingerprint from `true_bits`.
    ///
    /// One class for the molecule, then one outcome per bit of the
    /// fingerprint (all 4096, on and off). Deterministic in (`seed`, `key`,
    /// `draw`) through [`mix_fp_seed`]. Out-of-range true bits are
    /// [`Error::Config`].
    pub fn sample(
        &self,
        true_bits: &[u16],
        seed: u64,
        key: &str,
        draw: u64,
    ) -> Result<SparseFingerprint> {
        let mut is_on = vec![false; FINGERPRINT_BITS];
        for &bit in true_bits {
            if usize::from(bit) >= FINGERPRINT_BITS {
                return Err(Error::config(format!(
                    "FingerprintChannel::sample: true bit {bit} is past {FINGERPRINT_BITS}"
                )));
            }
            is_on[usize::from(bit)] = true;
        }
        let mut rng = SplitMix64::new(mix_fp_seed(seed, key, draw));
        let mut unit = move || (rng.next() >> 11) as f64 / (1u64 << 53) as f64;
        let roll = unit();
        let mut class = self.classes - 1;
        let mut acc = 0.0f64;
        for (k, &weight) in self.weights.iter().enumerate() {
            acc += weight;
            if roll < acc {
                class = k;
                break;
            }
        }
        let width = 1.0f64 / FINGERPRINT_BUCKETS as f64;
        let floor = f64::from(self.threshold);
        let mut entries: Vec<(u16, f32)> = Vec::new();
        for (bit, &on) in is_on.iter().enumerate() {
            let row = self.row(class, bit, on);
            let roll = unit() as f32;
            if roll < row[0] {
                continue;
            }
            let bucket = row
                .iter()
                .position(|&cumulative| roll < cumulative)
                .unwrap_or(FINGERPRINT_BUCKETS)
                .max(1);
            // Uniform inside the bucket `((b-1)/8, b/8]`, at or above the
            // token threshold (bucket 1 starts at the threshold).
            let lo = (((bucket - 1) as f64) * width).max(floor);
            let hi = bucket as f64 * width;
            let mut prob = (hi - unit() * (hi - lo)) as f32;
            if SparseFingerprint::bucket(prob) != bucket as u32 || prob < self.threshold {
                prob = hi as f32;
            }
            entries.push((bit as u16, prob));
        }
        let out = SparseFingerprint { entries };
        out.validate()?;
        Ok(out)
    }
}

/// Which evidence a run trains and generates with.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Evidence {
    /// Functional-group / random patterns only (today's behaviour).
    #[default]
    Patterns,
    /// Fingerprint only (no patterns).
    Fingerprint,
    /// Functional-group patterns and the fingerprint together.
    Both,
}

impl Evidence {
    /// The CLI spelling.
    pub fn as_str(self) -> &'static str {
        match self {
            Evidence::Patterns => "patterns",
            Evidence::Fingerprint => "fingerprint",
            Evidence::Both => "both",
        }
    }

    /// Parse a CLI spelling; `None` for anything else.
    pub fn parse(text: &str) -> Option<Self> {
        match text {
            "patterns" => Some(Evidence::Patterns),
            "fingerprint" => Some(Evidence::Fingerprint),
            "both" => Some(Evidence::Both),
            _ => None,
        }
    }

    /// Whether the model sees patterns.
    pub fn uses_patterns(self) -> bool {
        matches!(self, Evidence::Patterns | Evidence::Both)
    }

    /// Whether the model sees the fingerprint.
    pub fn uses_fingerprint(self) -> bool {
        matches!(self, Evidence::Fingerprint | Evidence::Both)
    }
}

/// Which fingerprint a training/evaluation query carries.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FingerprintEvalMode {
    /// Exact true bits at probability 1.
    #[default]
    Exact,
    /// MIST-like synthetic predictions from [`FingerprintNoise`].
    MistLike,
    /// Real predicted probabilities (the panel file's `fp_pred_mean`).
    Predicted,
}

impl FingerprintEvalMode {
    /// The CLI spelling.
    pub fn as_str(self) -> &'static str {
        match self {
            FingerprintEvalMode::Exact => "exact",
            FingerprintEvalMode::MistLike => "mist_like",
            FingerprintEvalMode::Predicted => "predicted",
        }
    }

    /// Parse a CLI spelling; `None` for anything else.
    pub fn parse(text: &str) -> Option<Self> {
        match text {
            "exact" => Some(FingerprintEvalMode::Exact),
            "mist_like" => Some(FingerprintEvalMode::MistLike),
            "predicted" => Some(FingerprintEvalMode::Predicted),
            _ => None,
        }
    }
}

/// Fingerprint evidence of one query for diagnostics: token counts and
/// true/false confusion against the true bits.
#[derive(Clone, Debug, Default)]
pub struct FingerprintQueryStats {
    /// Tokens used (selected entries).
    pub tokens_used: usize,
    /// Entries dropped by the slot limit.
    pub entries_dropped: usize,
    /// True bits missing from the tokens.
    pub true_missing: usize,
    /// False tokens (selected bits outside the truth).
    pub false_tokens: usize,
}

impl FingerprintQueryStats {
    /// Score `fingerprint` (already thresholded) with `slots` slots against
    /// `true_bits`: selected tokens are the top-`slots` entries, as in
    /// [`tokens`](SparseFingerprint::tokens).
    pub fn score(fingerprint: &SparseFingerprint, true_bits: &[u16], slots: usize) -> Self {
        let mut order: Vec<usize> = (0..fingerprint.entries.len()).collect();
        order.sort_by(|&a, &b| {
            fingerprint.entries[b]
                .1
                .total_cmp(&fingerprint.entries[a].1)
                .then_with(|| fingerprint.entries[a].0.cmp(&fingerprint.entries[b].0))
        });
        let take = order.len().min(slots);
        let selected: HashSet<u16> = order
            .iter()
            .take(take)
            .map(|&e| fingerprint.entries[e].0)
            .collect();
        let truth: HashSet<u16> = true_bits.iter().copied().collect();
        Self {
            tokens_used: take,
            entries_dropped: fingerprint.entries.len().saturating_sub(take),
            true_missing: truth.iter().filter(|b| !selected.contains(b)).count(),
            false_tokens: selected.iter().filter(|b| !truth.contains(b)).count(),
        }
    }
}

/// Mean of a slice (`0.0` when empty).
pub fn mean_of(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

/// Build the [`HashMap`] behind `--exclude-identity-groups <panel json>`:
/// identity group to molecule key, from a panel file in the molecule-export
/// schema.
pub fn panel_identity_groups(text: &str) -> Result<HashMap<u64, String>> {
    let raw: serde_json::Value = serde_json::from_str(text)?;
    let molecules = raw.get("molecules").ok_or_else(|| {
        Error::config("panel_identity_groups: panel file has no molecules".to_string())
    })?;
    let list = molecules.as_array().ok_or_else(|| {
        Error::config("panel_identity_groups: panel molecules is not a list".to_string())
    })?;
    let mut out = HashMap::new();
    for (i, mol) in list.iter().enumerate() {
        let group = mol
            .get("identity_group")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| {
                Error::config(format!(
                    "panel_identity_groups: molecule {i} has no identity_group"
                ))
            })?;
        let key = mol
            .get("key")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        out.insert(group, key);
    }
    Ok(out)
}
