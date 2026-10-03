//! Bidirectional Mamba-3 spectrum encoder (architecture §4.1).
//!
//! [`Ms2Encoder::encode`] runs peak selection, peak features and metadata
//! features into the given buffers and its own, then the network with no
//! device read. All SSM blocks are internally unidirectional
//! [`Mamba3Block`]s; the backward direction is a second block run over the
//! per-spectrum `reverse` ids with [`Var::gather_tokens`].

use cubecl::prelude::Runtime;

use crate::autograd::{Var, cat};
use crate::backend::{Device, FloatElem};
use crate::error::Result;
use crate::models::mamba3::{Mamba3Block, Mamba3BlockConfig};
use crate::nn::linear::{Linear, LinearConfig};
use crate::nn::module::{Module, ModuleVisitor};
use crate::nn::norm::{RmsNorm, RmsNormConfig};
use crate::nn::param::Param;
use crate::tensor::Tensor;
use crate::tensor::ops::index::{IdTensor, slice_ids_along};
use crate::tensor::ops::ms2::{
    META_FEATURES, Ms2Constants, PEAK_FEATURES, kept_column, meta_features, peak_features,
    peak_select,
};
use crate::tensor::ops::random::Rng;

use super::batch::DeviceSpectra;
use super::contract::{Control, ModelConfig};
use crate::nn::init::Initializer;

/// One bidirectional encoder layer: a forward and a backward
/// [`Mamba3Block`] over the same width.
struct EncoderBlockPair<R: Runtime, E: FloatElem> {
    forward: Mamba3Block<R, E>,
    backward: Mamba3Block<R, E>,
}

impl<R: Runtime, E: FloatElem> Module<R, E> for EncoderBlockPair<R, E> {
    fn visit(&self, visitor: &mut ModuleVisitor<'_, R, E>) {
        visitor.child("forward", &self.forward);
        visitor.child("backward", &self.backward);
    }
}

/// The MS2 spectrum encoder of architecture §4.1.
pub struct Ms2Encoder<R: Runtime, E: FloatElem> {
    peak_in: Linear<R, E>,
    peak_out: Linear<R, E>,
    adduct: Param<R, E>,
    polarity: Param<R, E>,
    energy_count: Param<R, E>,
    energy_known: Param<R, E>,
    meta_in: Linear<R, E>,
    condition: Linear<R, E>,
    blocks: Vec<EncoderBlockPair<R, E>>,
    norm: RmsNorm<R, E>,
    memory_in: Linear<R, E>,
    /// Resident Fourier wavelengths, uploaded once and reused by every
    /// feature call, so a warmed encode performs no upload.
    constants: Ms2Constants<R>,
    d_model: usize,
}

/// Output of [`Ms2Encoder::encode`].
pub struct EncoderOutput<R: Runtime, E: FloatElem> {
    /// `[B, N, d]` encoder states, exact zeros in padding.
    pub x: Var<R, E>,
    /// `[B, N]` 1/0 peak validity.
    pub valid: Tensor<R, E>,
    /// `[B, 1 + N, d]` spectrum memory: `[Linear(g); x]`.
    pub memory: Var<R, E>,
    /// `[B, 1 + N]` memory mask: `[1; valid]` (peaks zero under
    /// [`Control::MetadataOnly`] and [`Control::StructurePrior`]).
    pub memory_mask: Tensor<R, E>,
    /// `[B, d]` pooled vector: masked mean of `x` plus `g` (mean part zero
    /// under [`Control::MetadataOnly`] and [`Control::StructurePrior`]).
    pub pool: Var<R, E>,
    /// `[B, d]` metadata context `g`.
    pub context: Var<R, E>,
}

impl<R: Runtime, E: FloatElem> Ms2Encoder<R, E> {
    /// Build the encoder for `model` on `device`.
    pub fn init(model: &ModelConfig, device: &Device<R>, rng: &mut Rng) -> Result<Self> {
        let d = model.d_model as usize;
        let mk_table = |rows: usize, device: &Device<R>, rng: &mut Rng| {
            Param::new(
                Initializer::Normal {
                    mean: 0.0,
                    std: 0.02,
                }
                .init(vec![rows, d], device, rng),
            )
        };
        let adduct = mk_table(3, device, rng);
        let polarity = mk_table(2, device, rng);
        let energy_count = mk_table(9, device, rng);
        let energy_known = mk_table(2, device, rng);
        let peak_in = LinearConfig::new(PEAK_FEATURES, d).init(device, rng);
        let peak_out = LinearConfig::new(d, d).init(device, rng);
        let meta_in = LinearConfig::new(META_FEATURES, d).init(device, rng);
        let condition = LinearConfig::new(d, d).with_bias(false).init(device, rng);
        let norm = RmsNormConfig::new(d).init(device, rng);
        let memory_in = LinearConfig::new(d, d).init(device, rng);
        let mut blocks = Vec::with_capacity(model.encoder_blocks as usize);
        for _ in 0..model.encoder_blocks {
            let forward = Mamba3BlockConfig::new(model.encoder.clone()).init(device, rng)?;
            let backward = Mamba3BlockConfig::new(model.encoder.clone()).init(device, rng)?;
            blocks.push(EncoderBlockPair { forward, backward });
        }
        Ok(Self {
            peak_in,
            peak_out,
            adduct,
            polarity,
            energy_count,
            energy_known,
            meta_in,
            condition,
            blocks,
            norm,
            memory_in,
            constants: Ms2Constants::new(device),
            d_model: d,
        })
    }

    /// The resident Fourier wavelengths shared by the feature kernels.
    ///
    /// Test support: lets the footprint test attribute uploads.
    pub fn constants(&self) -> &Ms2Constants<R> {
        &self.constants
    }

    /// The `l`-th bidirectional block (forward, backward).
    ///
    /// Test and inspection support: the parity test steps these blocks
    /// explicitly against [`Mamba3Block::apply`].
    pub fn block(&self, l: usize) -> (&Mamba3Block<R, E>, &Mamba3Block<R, E>) {
        (&self.blocks[l].forward, &self.blocks[l].backward)
    }

    /// The block-0 input: peak embedding plus metadata conditioning,
    /// selected to exact zeros in padding.
    ///
    /// Test and inspection support: the bidirectional-correctness test
    /// builds its stepped reference from this tensor and [`Ms2Encoder::block`].
    pub fn embed(
        &self,
        spectra: &DeviceSpectra<R, E>,
        peaks: &crate::tensor::ops::ms2::PeakBuffers<R, E>,
        _control: Control,
    ) -> Result<Var<R, E>> {
        let (x0, _, _, _) = self.run_pre(spectra, peaks, _control)?;
        Ok(x0)
    }

    /// Peak selection, features and the peak/metadata embedding, up to the
    /// block-0 input. Returns `(x0, valid, g, reverse_ids)`.
    ///
    /// Under [`Control::StructurePrior`] the metadata path is blinded inside
    /// the encoder only: every table lookup uses the unknown row (adduct 0,
    /// energy count 0, energy-known 0; V0 has no unknown polarity row, so the
    /// constant row 0 stands in and no per-spectrum polarity survives) and the
    /// 34 energy/precursor features are exact zeros rather than computed. The
    /// [`DeviceSpectra`] itself is untouched, so validation, the formula
    /// search and the grammar budget still see the real request.
    #[allow(clippy::type_complexity)]
    fn run_pre(
        &self,
        spectra: &DeviceSpectra<R, E>,
        peaks: &crate::tensor::ops::ms2::PeakBuffers<R, E>,
        control: Control,
    ) -> Result<(Var<R, E>, Tensor<R, E>, Var<R, E>, IdTensor<R>)> {
        let batch = spectra.batch;
        let n_keep = peaks.kept.shape().dim(1);
        let device = spectra.mz.device().clone();
        peak_select(
            &spectra.mz,
            &spectra.intensity,
            &spectra.meta,
            spectra.intensity_scale,
            peaks,
        )?;
        let features = Tensor::<R, E>::empty(vec![batch, n_keep, PEAK_FEATURES], &device);
        peak_features(
            &peaks.kept,
            &peaks.kept_f,
            &spectra.meta,
            &self.constants,
            &features,
        )?;
        // The 34 metadata features are a pure function of the collision energy
        // and the precursor, so under the structure prior they are exact zeros
        // (no launch, no information) rather than computed values.
        let meta_feat = if control == Control::StructurePrior {
            Tensor::<R, E>::zeros(vec![batch, META_FEATURES], &device)
        } else {
            let out = Tensor::<R, E>::empty(vec![batch, META_FEATURES], &device);
            meta_features(&spectra.meta, &spectra.energy, &self.constants, &out)?;
            out
        };
        // `valid` is the second column of `kept_f` as `[B, N]`.
        let valid = if batch == 0 || n_keep == 0 {
            Tensor::<R, E>::zeros(vec![batch, n_keep], &device)
        } else {
            crate::tensor::ops::movement::slice(&peaks.kept_f, 2, 1, 1)?
                .reshape(vec![batch, n_keep])?
        };
        let reverse_ids = kept_column(&peaks.kept, 2)?;
        // Peak path: Linear(71 -> d), SiLU, Linear(d -> d), then select.
        let features_var = Var::constant(features);
        let h = self
            .peak_out
            .apply(&self.peak_in.apply(&features_var)?.silu()?)?;
        let h = h.ms2_select_valid(&valid)?;
        // Metadata context: four table lookups plus Linear(34 -> d). Under the
        // structure prior every lookup reads the unknown row (constant zeros
        // are uploaded instead of the spectrum's rows), so `g` carries no
        // per-spectrum metadata.
        let col = |c: usize| -> Result<IdTensor<R>> {
            if batch == 0 {
                return IdTensor::from_slice(&[], vec![0], &device);
            }
            if control == Control::StructurePrior {
                return IdTensor::from_slice(&vec![0u32; batch], vec![batch], &device);
            }
            slice_ids_along(&spectra.meta_ids, 1, c, 1)?.reshape(vec![batch])
        };
        let adduct_ids = col(0)?;
        let polarity_ids = col(1)?;
        let count_ids = col(2)?;
        let known_ids = col(3)?;
        let adduct_var = Var::ms2_lookup(&self.adduct.var_standalone(), &adduct_ids)?;
        let polarity_var = Var::ms2_lookup(&self.polarity.var_standalone(), &polarity_ids)?;
        let count_var = Var::ms2_lookup(&self.energy_count.var_standalone(), &count_ids)?;
        let known_var = Var::ms2_lookup(&self.energy_known.var_standalone(), &known_ids)?;
        let meta_proj = self.meta_in.apply(&Var::constant(meta_feat))?;
        let g = adduct_var
            .add(&polarity_var)?
            .add(&count_var)?
            .add(&known_var)?
            .add(&meta_proj)?;
        // Conditioning broadcast over peaks.
        let d = self.d_model;
        let conditioned = self.condition.apply(&g)?;
        let broadcast = if batch == 0 || n_keep == 0 {
            Var::constant(Tensor::<R, E>::zeros(vec![batch, n_keep, d], &device))
        } else {
            conditioned.unsqueeze(1)?.expand(vec![batch, n_keep, d])?
        };
        let x0 = h.add(&broadcast)?.ms2_select_valid(&valid)?;
        Ok((x0, valid, g, reverse_ids))
    }

    /// Encode `spectra` (with scratch `peaks`) into spectrum memory, pool
    /// and context. Runs peak selection, peak features and metadata features,
    /// then the network, with no device read. [`Control::ShuffledSpectrum`]
    /// is applied by the caller with `rotate_peaks` before upload, so it is
    /// treated like [`Control::None`]; [`Control::MetadataOnly`] zeroes the
    /// peak part of the memory mask and the pool. [`Control::StructurePrior`]
    /// does that too, and additionally zeroes the peak states themselves
    /// (the mask alone would leave peak-dependent values in memory) while
    /// [`run_pre`] blinds the metadata path, so the memory and pool carry no
    /// per-spectrum peak or metadata information at all.
    pub fn encode(
        &self,
        spectra: &DeviceSpectra<R, E>,
        peaks: &crate::tensor::ops::ms2::PeakBuffers<R, E>,
        control: Control,
    ) -> Result<EncoderOutput<R, E>> {
        let batch = spectra.batch;
        let n_keep = peaks.kept.shape().dim(1);
        let d = self.d_model;
        let device = spectra.mz.device().clone();
        if batch == 0 {
            let x = Var::constant(Tensor::<R, E>::zeros(vec![0, n_keep, d], &device));
            let valid = Tensor::<R, E>::zeros(vec![0, n_keep], &device);
            let memory = Var::constant(Tensor::<R, E>::zeros(vec![0, 1 + n_keep, d], &device));
            let memory_mask = Tensor::<R, E>::zeros(vec![0, 1 + n_keep], &device);
            let pool = Var::constant(Tensor::<R, E>::zeros(vec![0, d], &device));
            let context = Var::constant(Tensor::<R, E>::zeros(vec![0, d], &device));
            return Ok(EncoderOutput {
                x,
                valid,
                memory,
                memory_mask,
                pool,
                context,
            });
        }
        let (mut x, valid, g, reverse_ids) = self.run_pre(spectra, peaks, control)?;
        for pair in &self.blocks {
            let f = pair.forward.apply(&x)?;
            let gathered = Var::gather_tokens(&x, &reverse_ids, n_keep)?;
            let bwd = pair.backward.apply(&gathered)?;
            let r = Var::gather_tokens(&bwd, &reverse_ids, n_keep)?;
            x = f.add(&r)?.sub(&x)?.ms2_select_valid(&valid)?;
        }
        x = self.norm.apply(&x)?.ms2_select_valid(&valid)?;
        // The structure prior discards the peak states: the memory mask below
        // zeroes attention to the peaks, but the values would still differ per
        // spectrum, so exact zeros replace them (and carry no gradient, which
        // is the point of the blinded encoder).
        if control == Control::StructurePrior {
            x = Var::constant(Tensor::<R, E>::zeros(vec![batch, n_keep, d], &device));
        }
        // Validity for the memory and the pool: zeroed under MetadataOnly and
        // the structure prior.
        let valid_pool = match control {
            Control::MetadataOnly | Control::StructurePrior => {
                Tensor::<R, E>::zeros(vec![batch, n_keep], &device)
            }
            _ => valid.clone(),
        };
        // Memory `[Linear(g); x]` and mask `[1; valid]`.
        let mem0 = self.memory_in.apply(&g)?.unsqueeze(1)?;
        let memory = cat(&[mem0, x.clone()], 1)?;
        let ones = Tensor::<R, E>::ones(vec![batch, 1], &device);
        let memory_mask = crate::tensor::ops::movement::cat(&[ones, valid_pool.clone()], 1)?;
        // Pool: masked mean of `x` plus `g`; the mean part is zero under
        // MetadataOnly and the structure prior because there are no valid peaks.
        let pool = match control {
            Control::MetadataOnly | Control::StructurePrior => g.clone(),
            _ => {
                let x_sum = x.sum_dim(1)?;
                let len_t = crate::tensor::ops::reduce::sum_dim(&valid_pool, 1)?
                    .reshape(vec![batch, 1, 1])?;
                let len_var = Var::constant(len_t);
                let one_var = Var::constant(Tensor::<R, E>::ones(vec![batch, 1, 1], &device));
                let den = len_var.maximum(&one_var)?;
                let mean = x_sum.div(&den)?.squeeze(1)?;
                mean.add(&g)?
            }
        };
        Ok(EncoderOutput {
            x,
            valid,
            memory,
            memory_mask,
            pool,
            context: g,
        })
    }
}

impl<R: Runtime, E: FloatElem> Module<R, E> for Ms2Encoder<R, E> {
    fn visit(&self, visitor: &mut ModuleVisitor<'_, R, E>) {
        visitor.child("peak_in", &self.peak_in);
        visitor.child("peak_out", &self.peak_out);
        visitor.param("adduct", &self.adduct);
        visitor.param("polarity", &self.polarity);
        visitor.param("energy_count", &self.energy_count);
        visitor.param("energy_known", &self.energy_known);
        visitor.child("meta_in", &self.meta_in);
        visitor.child("condition", &self.condition);
        for (i, b) in self.blocks.iter().enumerate() {
            visitor.child_at("blocks", i, b);
        }
        visitor.child("norm", &self.norm);
        visitor.child("memory_in", &self.memory_in);
    }
}
