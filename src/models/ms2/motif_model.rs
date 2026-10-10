//! Motif decoder with the completion model's conditioning encoders.
//!
//! [`motif`](super::motif) defines the motif alphabet and its stack machine;
//! the first decoder built on it (`examples/ms2_motif_decoder.rs --arch
//! prefix`) fed the formula and the fingerprint to a plain sequence model as
//! a token prefix, so everything it knew of them had to survive in the
//! recurrent state. [`MotifModel`] is the same alphabet behind the
//! conditioning path the atom-level
//! [`CompletionModel`](super::completion_model::CompletionModel) uses:
//!
//! * the fingerprint goes through the same
//!   [`FingerprintEncoder`] (a permutation-invariant set encoder) and the
//!   fragment peaks, adduct and neutral mass through the same
//!   [`SpectrumEncoder`];
//! * their token states form a memory, headed by one row made from the
//!   formula embedding plus both pooled vectors;
//! * every decoder layer is a [`Mamba3Block`] followed by residual
//!   cross-attention into that memory, and the pooled row is also added to
//!   every input position.
//!
//! What it does not have, because the alphabet does not need them: the
//! atom-level decoder's factored heads, pointer head and atom memory (an
//! attachment names its atom by a token), its substructure-pattern encoder
//! and its per-step progress features.

use cubecl::prelude::Runtime;

use crate::autograd::{Var, cat};
use crate::backend::{Device, FloatElem};
use crate::error::{Error, Result};
use crate::models::mamba3::{Mamba3Block, Mamba3BlockConfig, MixerCache};
use crate::nn::init::Initializer;
use crate::nn::linear::{Linear, LinearConfig};
use crate::nn::module::{Module, ModuleVisitor};
use crate::nn::norm::{RmsNorm, RmsNormConfig};
use crate::nn::param::Param;
use crate::ssm::{SsmConfig, SsmState};
use crate::tensor::ops::index::{IdTensor, check_gather_ids, gather_rows};
use crate::tensor::ops::random::Rng;
use crate::tensor::{Shape, Tensor};

use super::completion_fingerprint::{FingerprintBatch, FingerprintEncoder};
use super::completion_spectrum::{SpectrumBatch, SpectrumEncoder};
use super::motif::Formula;

/// Scale of the formula counts before the formula network.
const FORMULA_SCALE: f32 = 1.0 / 32.0;

/// Shape of a [`MotifModel`].
#[derive(Clone, Debug)]
pub struct MotifModelConfig {
    /// Output ids ([`Layout::n_out`](super::motif::Layout::n_out)).
    pub n_out: usize,
    /// Residual width.
    pub d_model: usize,
    /// Decoder layers.
    pub layers: usize,
    /// Cross-attention heads (must divide `d_model`).
    pub attention_heads: usize,
    /// State size of the Mamba blocks.
    pub d_state: usize,
    /// Longest target sequence (rows of the position table).
    pub max_tokens: usize,
    /// Fingerprint token slots of the memory.
    pub fingerprint_slots: usize,
    /// Peak slots of the memory; 0 builds no spectrum encoder.
    pub spectrum_slots: usize,
    /// Initialisation seed.
    pub seed: u64,
}

impl MotifModelConfig {
    /// Width of the head: `n_out` rounded up to a multiple of 64 so its product uses the block matmul kernels.
    pub fn n_out_padded(&self) -> usize {
        self.n_out.div_ceil(64) * 64
    }
}

/// One decoder layer: the block, then cross-attention into the memory.
struct MotifLayer<R: Runtime, E: FloatElem> {
    mixer: Mamba3Block<R, E>,
    norm: RmsNorm<R, E>,
    q: Linear<R, E>,
    k: Linear<R, E>,
    v: Linear<R, E>,
    o: Linear<R, E>,
}

impl<R: Runtime, E: FloatElem> Module<R, E> for MotifLayer<R, E> {
    fn visit(&self, visitor: &mut ModuleVisitor<'_, R, E>) {
        visitor.child("mixer", &self.mixer);
        visitor.child("norm", &self.norm);
        visitor.child("q", &self.q);
        visitor.child("k", &self.k);
        visitor.child("v", &self.v);
        visitor.child("o", &self.o);
    }
}

/// The encoded conditioning of a batch of queries.
pub struct MotifConditioning<R: Runtime, E: FloatElem> {
    /// `[B, d]`: formula embedding plus the pooled fingerprint and spectrum.
    pub context: Var<R, E>,
    /// `[B, 1 + slots, d]`: the context row, then fingerprint token states,
    /// then peak states.
    pub memory: Var<R, E>,
    /// `[B, 1 + slots]` validity of the memory rows.
    pub mask: Tensor<R, E>,
}

/// Decoding state of one query's beam: every row shares the query's memory.
pub struct MotifStepState<R: Runtime, E: FloatElem> {
    caches: Vec<MixerCache<R, E>>,
    /// Per layer, unsplit keys `[1, m, d]` and values `[1, m, d]`.
    keys: Vec<Tensor<R, E>>,
    values: Vec<Tensor<R, E>>,
    mask: Vec<f32>,
    mask_dev: Tensor<R, E>,
    context: Var<R, E>,
    position: usize,
    rows: usize,
}

impl<R: Runtime, E: FloatElem> MotifStepState<R, E> {
    /// Live rows.
    pub fn rows(&self) -> usize {
        self.rows
    }
}

/// The motif decoder with conditioning encoders (see the module docs).
pub struct MotifModel<R: Runtime, E: FloatElem> {
    /// `[n_out + 1, d]`: one row per output id, then the start row.
    token_emb: Param<R, E>,
    /// `[max_tokens, d]` position table.
    pos_emb: Param<R, E>,
    formula_in: Linear<R, E>,
    formula_out: Linear<R, E>,
    fingerprint: FingerprintEncoder<R, E>,
    spectrum: Option<SpectrumEncoder<R, E>>,
    memory_in: Linear<R, E>,
    layers: Vec<MotifLayer<R, E>>,
    norm_f: RmsNorm<R, E>,
    head: Linear<R, E>,
    config: MotifModelConfig,
}

impl<R: Runtime, E: FloatElem> Module<R, E> for MotifModel<R, E> {
    fn visit(&self, visitor: &mut ModuleVisitor<'_, R, E>) {
        visitor.param("token_emb", &self.token_emb);
        visitor.param("pos_emb", &self.pos_emb);
        visitor.child("formula_in", &self.formula_in);
        visitor.child("formula_out", &self.formula_out);
        visitor.child("fingerprint", &self.fingerprint);
        if let Some(spectrum) = &self.spectrum {
            visitor.child("spectrum", spectrum);
        }
        visitor.child("memory_in", &self.memory_in);
        for (l, layer) in self.layers.iter().enumerate() {
            visitor.child_at("layer", l, layer);
        }
        visitor.child("norm_f", &self.norm_f);
        visitor.child("head", &self.head);
    }
}

impl<R: Runtime, E: FloatElem> MotifModel<R, E> {
    /// Build the model. [`Error::Config`] for a zero size or an attention
    /// head count that does not divide `d_model`.
    pub fn init(config: &MotifModelConfig, device: &Device<R>) -> Result<Self> {
        let d = config.d_model;
        if d == 0
            || config.n_out == 0
            || config.layers == 0
            || config.max_tokens == 0
            || config.attention_heads == 0
            || !d.is_multiple_of(config.attention_heads)
        {
            return Err(Error::config(format!("MotifModel::init: bad shape {config:?}")));
        }
        let mut rng = Rng::seeded(config.seed);
        let table = |rows: usize, rng: &mut Rng| -> Param<R, E> {
            Param::new(
                Initializer::Normal {
                    mean: 0.0,
                    std: 0.02,
                }
                .init(vec![rows, d], device, rng),
            )
        };
        let token_emb = table(config.n_out + 1, &mut rng);
        let pos_emb = table(config.max_tokens, &mut rng);
        // The atom-level decoder's block: four 64-wide heads at width 128.
        let heads = (2 * d / 64).max(1);
        let ssm = SsmConfig {
            d_model: d,
            n_heads: heads,
            head_dim: 64,
            d_state: config.d_state,
            n_groups: heads,
            chunk_size: 64,
            ..SsmConfig::default()
        };
        let mut layers = Vec::with_capacity(config.layers);
        for _ in 0..config.layers {
            layers.push(MotifLayer {
                mixer: Mamba3BlockConfig::new(ssm.clone()).init(device, &mut rng)?,
                norm: RmsNormConfig::new(d).init(device, &mut rng),
                q: LinearConfig::new(d, d).with_bias(false).init(device, &mut rng),
                k: LinearConfig::new(d, d).with_bias(false).init(device, &mut rng),
                v: LinearConfig::new(d, d).with_bias(false).init(device, &mut rng),
                o: LinearConfig::new(d, d).with_bias(false).init(device, &mut rng),
            });
        }
        Ok(Self {
            token_emb,
            pos_emb,
            formula_in: LinearConfig::new(10, d).init(device, &mut rng),
            formula_out: LinearConfig::new(d, d).init(device, &mut rng),
            fingerprint: FingerprintEncoder::init(d, device, &mut rng),
            spectrum: (config.spectrum_slots > 0)
                .then(|| SpectrumEncoder::init(d, device, &mut rng)),
            memory_in: LinearConfig::new(d, d).init(device, &mut rng),
            layers,
            norm_f: RmsNormConfig::new(d).init(device, &mut rng),
            head: LinearConfig::new(d, config.n_out_padded()).init(device, &mut rng),
            config: config.clone(),
        })
    }

    /// The configuration the model was built with.
    pub fn config(&self) -> &MotifModelConfig {
        &self.config
    }

    /// The id of the start row: the input at position 0.
    pub fn start_token(&self) -> u32 {
        self.config.n_out as u32
    }

    /// Encode the conditioning of `formulas.len()` queries. `fingerprints`
    /// must hold that many queries of `fingerprint_slots` slots; `spectra`
    /// likewise for a model with a spectrum encoder (`None` encodes no
    /// evidence, which contributes exact zeros) and is refused by a model
    /// without one when it carries evidence. No device read.
    pub fn encode(
        &self,
        formulas: &[Formula],
        fingerprints: &FingerprintBatch,
        spectra: Option<&SpectrumBatch>,
        device: &Device<R>,
    ) -> Result<MotifConditioning<R, E>> {
        let b = formulas.len();
        let d = self.config.d_model;
        if b == 0
            || fingerprints.queries != b
            || fingerprints.slots != self.config.fingerprint_slots
        {
            return Err(Error::config(format!(
                "MotifModel::encode: {b} formulas, fingerprint batch of {} queries and {} slots (model has {})",
                fingerprints.queries, fingerprints.slots, self.config.fingerprint_slots
            )));
        }
        let counts: Vec<f32> = formulas
            .iter()
            .flat_map(|formula| formula.iter().map(|&c| f32::from(c) * FORMULA_SCALE))
            .collect();
        let counts = Var::constant(Tensor::<R, E>::from_f32(&counts, vec![b, 10], device)?);
        let mut g = self
            .formula_out
            .apply(&self.formula_in.apply(&counts)?.silu()?)?;
        let (h_fp, valid_fp) = self.fingerprint.encode_states(fingerprints, device)?;
        // `g + pooled`, the accumulator first, as the completion model does.
        g = g.add(
            &self
                .fingerprint
                .encode_pooled_from_states(&h_fp, &valid_fp, device)?,
        )?;
        let mut memory_parts = vec![h_fp];
        let mut mask_parts = vec![valid_fp];
        match (&self.spectrum, spectra) {
            (Some(encoder), spectra) => {
                let empty;
                let batch = match spectra {
                    Some(batch) => batch,
                    None => {
                        empty = SpectrumBatch::empty(b, self.config.spectrum_slots)?;
                        &empty
                    }
                };
                if batch.queries != b || batch.slots != self.config.spectrum_slots {
                    return Err(Error::config(format!(
                        "MotifModel::encode: spectrum batch of {} queries and {} slots for {b} queries and {} slots",
                        batch.queries, batch.slots, self.config.spectrum_slots
                    )));
                }
                let encoded = encoder.encode(batch, device)?;
                g = g.add(&encoded.pooled)?;
                memory_parts.push(encoded.states);
                mask_parts.push(encoded.valid);
            }
            (None, Some(batch)) if batch.present.iter().any(|&v| v != 0.0) => {
                return Err(Error::config(
                    "MotifModel::encode: spectral evidence for a model without a spectrum encoder"
                        .to_string(),
                ));
            }
            (None, _) => {}
        }
        let mut memory_all = vec![self.memory_in.apply(&g)?.unsqueeze(1)?];
        memory_all.extend(memory_parts);
        let mut mask_all = vec![Tensor::<R, E>::ones(vec![b, 1], device)];
        mask_all.extend(mask_parts);
        let memory = cat(&memory_all, 1)?;
        debug_assert_eq!(memory.dims()[2], d);
        Ok(MotifConditioning {
            context: g,
            memory,
            mask: crate::tensor::ops::movement::cat(&mask_all, 1)?,
        })
    }

    /// Keys `[b, h, hd, m]` and values `[b, h, m, hd]` of one layer.
    fn keys_values(
        &self,
        layer: &MotifLayer<R, E>,
        memory: &Var<R, E>,
    ) -> Result<(Var<R, E>, Var<R, E>)> {
        let (b, m) = (memory.dims()[0], memory.dims()[1]);
        let h = self.config.attention_heads;
        let hd = self.config.d_model / h;
        Ok((
            layer
                .k
                .apply(memory)?
                .reshape(vec![b, m, h, hd])?
                .permute(&[0, 2, 3, 1])?,
            layer
                .v
                .apply(memory)?
                .reshape(vec![b, m, h, hd])?
                .permute(&[0, 2, 1, 3])?,
        ))
    }

    /// Cross-attention of one layer: queries `[b, qt, d]` over the split
    /// keys and values, masked rows taking no weight.
    fn attend(
        &self,
        layer: &MotifLayer<R, E>,
        q_in: &Var<R, E>,
        kh: &Var<R, E>,
        vh: &Var<R, E>,
        mask: &Tensor<R, E>,
    ) -> Result<Var<R, E>> {
        let (b, qt) = (q_in.dims()[0], q_in.dims()[1]);
        let d = self.config.d_model;
        let h = self.config.attention_heads;
        let hd = d / h;
        let q = layer.q.apply(&layer.norm.apply(q_in)?)?;
        let qh = q.reshape(vec![b, qt, h, hd])?.permute(&[0, 2, 1, 3])?;
        let weights = qh
            .matmul(kh)?
            .masked_softmax(mask, 1.0 / (hd as f32).sqrt())?;
        let ctx = weights
            .matmul(vh)?
            .permute(&[0, 2, 1, 3])?
            .reshape(vec![b, qt, d])?;
        layer.o.apply(&ctx)
    }

    /// The final hidden states `[B, T, d]` that [`logits`](Self::logits)
    /// projects with the head: the same checks and layers, without the
    /// head product.
    pub fn final_hidden(
        &self,
        inputs: &IdTensor<R>,
        conditioning: &MotifConditioning<R, E>,
    ) -> Result<Var<R, E>> {
        let dims = inputs.shape().dims().to_vec();
        if dims.len() != 2 || dims[1] == 0 || dims[1] > self.config.max_tokens {
            return Err(Error::shape(format!(
                "MotifModel::logits: inputs {} for at most {} positions",
                inputs.shape(),
                self.config.max_tokens
            )));
        }
        let (b, t) = (dims[0], dims[1]);
        let d = self.config.d_model;
        if conditioning.context.dims()[0] != b {
            return Err(Error::shape(format!(
                "MotifModel::logits: {b} rows for {} conditioned queries",
                conditioning.context.dims()[0]
            )));
        }
        let mut x = Var::ms2_lookup(
            &self.token_emb.var_standalone(),
            &inputs.reshape(vec![b * t])?,
        )?
        .reshape(vec![b, t, d])?
        .add(
            &self
                .pos_emb
                .var_standalone()
                .slice(0, 0, t)?
                .reshape(vec![1, t, d])?,
        )?
        .add(&conditioning.context.reshape(vec![b, 1, d])?)?;
        for layer in &self.layers {
            x = layer.mixer.apply(&x)?;
            let (kh, vh) = self.keys_values(layer, &conditioning.memory)?;
            let ctx = self.attend(layer, &x, &kh, &vh, &conditioning.mask)?;
            x = x.add(&ctx)?;
        }
        self.norm_f.apply(&x)
    }

    /// The output head (`[d, n_out_padded]` weight and `[n_out_padded]` bias).
    pub fn head(&self) -> &Linear<R, E> {
        &self.head
    }

    /// Teacher-forced logits `[B, T, n_out_padded()]`: position `i` reads `inputs[i]`
    /// (the start token, then the target shifted by one) and predicts target
    /// `i`. `inputs` is `[B, T]` with `T <= max_tokens`. Ids at or past
    /// `n_out` are padding.
    pub fn logits(
        &self,
        inputs: &IdTensor<R>,
        conditioning: &MotifConditioning<R, E>,
    ) -> Result<Var<R, E>> {
        self.head.apply(&self.final_hidden(inputs, conditioning)?)
    }

    /// Start decoding one query (`conditioning` must hold exactly one).
    pub fn start(
        &self,
        conditioning: &MotifConditioning<R, E>,
        device: &Device<R>,
    ) -> Result<MotifStepState<R, E>> {
        if conditioning.context.dims()[0] != 1 {
            return Err(Error::shape(format!(
                "MotifModel::start: needs one query, got {}",
                conditioning.context.dims()[0]
            )));
        }
        let mut keys = Vec::with_capacity(self.layers.len());
        let mut values = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            keys.push(layer.k.apply(&conditioning.memory)?.tensor().clone());
            values.push(layer.v.apply(&conditioning.memory)?.tensor().clone());
        }
        Ok(MotifStepState {
            caches: self
                .layers
                .iter()
                .map(|layer| layer.mixer.mixer().empty_cache(1, device))
                .collect(),
            keys,
            values,
            mask: conditioning.mask.to_f32(),
            mask_dev: conditioning.mask.clone(),
            context: conditioning.context.detach(),
            position: 0,
            rows: 1,
        })
    }

    /// Reorder the rows of `state`: row `i` continues row `parents[i]`.
    /// A parent at or past the live rows is [`Error::Shape`].
    pub fn gather(&self, state: &mut MotifStepState<R, E>, parents: &IdTensor<R>) -> Result<()> {
        check_gather_ids(parents, state.rows)?;
        self.gather_uploaded(state, parents)
    }

    /// [`gather`](Self::gather) from host ids: `parents` is validated on the
    /// host (every parent below the live rows) and uploaded once, with no
    /// device read. A parent at or past the live rows is [`Error::Shape`].
    pub fn gather_host(&self, state: &mut MotifStepState<R, E>, parents: &[u32], device: &Device<R>) -> Result<()> {
        if let Some(&bad) = parents.iter().find(|&&p| p as usize >= state.rows) {
            return Err(Error::shape(format!(
                "MotifModel::gather_host: parent {bad} is not below the {} live rows",
                state.rows
            )));
        }
        let ids = IdTensor::from_slice(parents, vec![parents.len()], device)?;
        self.gather_uploaded(state, &ids)
    }

    fn gather_uploaded(&self, state: &mut MotifStepState<R, E>, parents: &IdTensor<R>) -> Result<()> {
        let rows = state.rows;
        let take = |value: &Var<R, E>| -> Result<Var<R, E>> {
            let shape = value.shape().clone();
            let width = shape.num_elements() / rows;
            let gathered = gather_rows(&value.tensor().reshape(vec![rows, width])?, parents)?;
            let mut dims = shape.dims().to_vec();
            dims[0] = parents.len();
            Ok(Var::constant(gathered.reshape(Shape::new(dims))?))
        };
        for cache in &mut state.caches {
            *cache = MixerCache {
                ssm: SsmState {
                    h: take(&cache.ssm.h)?,
                    last_u: take(&cache.ssm.last_u)?,
                    angle: match &cache.ssm.angle {
                        Some(angle) => Some(take(angle)?),
                        None => None,
                    },
                },
                conv: match &cache.conv {
                    Some(conv) => Some(take(conv)?),
                    None => None,
                },
            };
        }
        state.rows = parents.len();
        Ok(())
    }

    /// One decoding step: feed `tokens` (`[rows]`, the start token first)
    /// and return the next logits `[rows, n_out_padded()]`. Ids at or past
    /// `n_out` are padding. Matches
    /// [`logits`](Self::logits) position by position.
    pub fn step(
        &self,
        state: &mut MotifStepState<R, E>,
        tokens: &IdTensor<R>,
        device: &Device<R>,
    ) -> Result<Var<R, E>> {
        let _no_grad = crate::autograd::no_grad();
        let rows = state.rows;
        if tokens.len() != rows {
            return Err(Error::shape(format!(
                "MotifModel::step: {} tokens for {rows} rows",
                tokens.len()
            )));
        }
        if state.position >= self.config.max_tokens {
            return Err(Error::shape(format!(
                "MotifModel::step: position {} is past the {} the model holds",
                state.position, self.config.max_tokens
            )));
        }
        let d = self.config.d_model;
        let m = state.mask.len();
        let position = self
            .pos_emb
            .var_standalone()
            .slice(0, state.position, 1)?
            .reshape(vec![1, d])?
            .expand(vec![rows, d])?;
        let mut x = Var::ms2_lookup(&self.token_emb.var_standalone(), &tokens.reshape(vec![rows])?)?
            .add(&position)?
            .add(&state.context.expand(vec![rows, d])?)?
            .unsqueeze(1)?;
        let h = self.config.attention_heads;
        for (l, layer) in self.layers.iter().enumerate() {
            let (y, cache) = layer.mixer.step(&x, &state.caches[l])?;
            state.caches[l] = cache;
            let q = layer.q.apply(&layer.norm.apply(&y)?)?.tensor().reshape(vec![rows, d])?;
            let mut w = Tensor::<R, E>::empty(vec![rows, h, m], device);
            crate::tensor::ops::ms2::attn_weights(&q, &state.keys[l], &state.mask_dev, &mut w, h, rows)?;
            let ctx = crate::tensor::ops::ms2::attn_context(&w, &state.values[l], rows)?;
            let back = layer.o.apply(&Var::constant(ctx.reshape(vec![rows, 1, d])?))?;
            x = y.add(&back)?;
        }
        state.position += 1;
        self.head
            .apply(&self.norm_f.apply(&x)?)?
            .reshape(vec![rows, self.config.n_out_padded()])
    }
}
