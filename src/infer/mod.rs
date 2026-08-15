//! Inference: recurrent caches, sampling, and generation.
//!
//! The point of a state space model is that decoding costs the same at position
//! 100,000 as at position 1. That property only survives if the recurrent state is
//! carried explicitly, which is what [`StateCache`] does — one entry per layer,
//! holding the SSM state (and convolution history) for Mamba layers and a
//! key/value cache for any attention layers in a hybrid.

use cubecl::prelude::Runtime;

use crate::backend::{Device, FloatElem};
use crate::error::{Error, Result};
use crate::models::hybrid::LayerCache;
use crate::models::lm::Mamba3Lm;
use crate::nn::module::Module;
use crate::tensor::ops::index::IdTensor;
use crate::tensor::ops::random::Rng;

/// Per-layer decoding state for a whole model.
pub struct StateCache<R: Runtime, E: FloatElem> {
    layers: Vec<LayerCache<R, E>>,
    position: usize,
}

impl<R: Runtime, E: FloatElem> StateCache<R, E> {
    /// A zeroed cache for `batch` sequences.
    pub fn new(model: &Mamba3Lm<R, E>, batch: usize, device: &Device<R>) -> Self {
        Self {
            layers: model.empty_cache(batch, device),
            position: 0,
        }
    }

    /// Number of tokens consumed so far.
    pub fn position(&self) -> usize {
        self.position
    }

    /// Mutable access to the per-layer entries.
    pub fn layers_mut(&mut self) -> &mut [LayerCache<R, E>] {
        &mut self.layers
    }

    /// Advance the position counter.
    fn advance(&mut self, tokens: usize) {
        self.position += tokens;
    }
}

/// How the next token is chosen.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SamplerConfig {
    /// Softmax temperature. `0` means greedy.
    pub temperature: f32,
    /// Keep only the `k` most likely tokens. `0` disables.
    pub top_k: usize,
    /// Keep the smallest set whose probability mass reaches `p`. `1` disables.
    pub top_p: f32,
    /// Divide the logit of already-generated tokens by this factor.
    pub repetition_penalty: f32,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
        }
    }
}

impl SamplerConfig {
    /// Always take the most likely token.
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            ..Default::default()
        }
    }

    /// Temperature sampling.
    pub fn temperature(t: f32) -> Self {
        Self {
            temperature: t,
            ..Default::default()
        }
    }

    /// Restrict to the top `k` logits.
    pub fn with_top_k(mut self, k: usize) -> Self {
        self.top_k = k;
        self
    }

    /// Restrict to the nucleus of mass `p`.
    pub fn with_top_p(mut self, p: f32) -> Self {
        self.top_p = p;
        self
    }

    /// Penalise tokens already in the context.
    pub fn with_repetition_penalty(mut self, penalty: f32) -> Self {
        self.repetition_penalty = penalty;
        self
    }

    /// Pick one token id from a row of logits.
    pub fn sample(&self, logits: &[f32], history: &[u32], rng: &mut Rng) -> usize {
        let mut scores = logits.to_vec();

        if self.repetition_penalty != 1.0 {
            for &token in history {
                let idx = token as usize;
                if idx < scores.len() {
                    // Penalise both directions so negative logits move the right way.
                    scores[idx] = if scores[idx] > 0.0 {
                        scores[idx] / self.repetition_penalty
                    } else {
                        scores[idx] * self.repetition_penalty
                    };
                }
            }
        }

        if self.temperature <= 0.0 {
            return scores
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(core::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);
        }

        for s in &mut scores {
            *s /= self.temperature;
        }

        let mut order: Vec<usize> = (0..scores.len()).collect();
        order.sort_by(|a, b| {
            scores[*b]
                .partial_cmp(&scores[*a])
                .unwrap_or(core::cmp::Ordering::Equal)
        });

        if self.top_k > 0 && self.top_k < order.len() {
            order.truncate(self.top_k);
        }

        let max = order.iter().map(|i| scores[*i]).fold(f32::MIN, f32::max);
        let mut probs: Vec<f32> = order.iter().map(|i| (scores[*i] - max).exp()).collect();
        let total: f32 = probs.iter().sum();
        for p in &mut probs {
            *p /= total;
        }

        if self.top_p < 1.0 {
            let mut cumulative = 0.0;
            let mut keep = probs.len();
            for (i, p) in probs.iter().enumerate() {
                cumulative += p;
                if cumulative >= self.top_p {
                    keep = i + 1;
                    break;
                }
            }
            order.truncate(keep);
            probs.truncate(keep);
            let total: f32 = probs.iter().sum();
            for p in &mut probs {
                *p /= total;
            }
        }

        let mut draw = rng.next_f32();
        for (i, p) in probs.iter().enumerate() {
            draw -= p;
            if draw <= 0.0 {
                return order[i];
            }
        }
        *order.last().unwrap_or(&0)
    }
}

/// Configuration for [`Generator`].
#[derive(Debug, Clone)]
pub struct GeneratorConfig {
    /// Maximum number of tokens to produce.
    pub max_new_tokens: usize,
    /// Sampling strategy.
    pub sampler: SamplerConfig,
    /// Stop as soon as this token is produced.
    pub eos_token: Option<u32>,
    /// Sampling seed.
    pub seed: u64,
}

impl Default for GeneratorConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 64,
            sampler: SamplerConfig::default(),
            eos_token: None,
            seed: 0,
        }
    }
}

impl GeneratorConfig {
    /// Start a builder.
    pub fn builder() -> GeneratorConfigBuilder {
        GeneratorConfigBuilder::default()
    }
}

/// Builder for [`GeneratorConfig`].
#[derive(Debug, Clone, Default)]
pub struct GeneratorConfigBuilder {
    config: Option<GeneratorConfig>,
}

impl GeneratorConfigBuilder {
    fn edit(mut self, f: impl FnOnce(&mut GeneratorConfig)) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        f(&mut config);
        self.config = Some(config);
        self
    }

    /// Token budget.
    pub fn max_new_tokens(self, n: usize) -> Self {
        self.edit(|c| c.max_new_tokens = n)
    }

    /// Sampling strategy.
    pub fn sampler(self, sampler: SamplerConfig) -> Self {
        self.edit(|c| c.sampler = sampler)
    }

    /// Stop token.
    pub fn eos_token(self, token: u32) -> Self {
        self.edit(|c| c.eos_token = Some(token))
    }

    /// Sampling seed.
    pub fn seed(self, seed: u64) -> Self {
        self.edit(|c| c.seed = seed)
    }

    /// Build the configuration.
    pub fn build(self) -> GeneratorConfig {
        self.config.unwrap_or_default()
    }
}

/// Autoregressive decoding with a recurrent cache.
pub struct Generator<'a, R: Runtime, E: FloatElem> {
    model: &'a Mamba3Lm<R, E>,
    config: GeneratorConfig,
    rng: Rng,
}

impl<'a, R: Runtime, E: FloatElem> Generator<'a, R, E> {
    /// Build a generator for a model.
    pub fn new(model: &'a Mamba3Lm<R, E>, config: GeneratorConfig) -> Self {
        let rng = Rng::seeded(config.seed);
        Self {
            model,
            config,
            rng,
        }
    }

    /// Continue a single prompt.
    ///
    /// The prompt is consumed in one chunked pass (the parallel scan), then each
    /// new token costs one recurrent step regardless of how long the context is.
    pub fn generate(&mut self, prompt: &[u32], device: &Device<R>) -> Result<Vec<u32>> {
        if prompt.is_empty() {
            return Err(Error::config("generation needs a non-empty prompt"));
        }
        self.model.eval();
        let _guard = crate::autograd::no_grad();

        let mut cache = StateCache::new(self.model, 1, device);
        let mut history: Vec<u32> = prompt.to_vec();

        // Prefill.
        let ids = IdTensor::from_slice(prompt, vec![1, prompt.len()], device)?;
        let logits = self
            .model
            .forward_cached(&ids, cache.layers_mut())?;
        cache.advance(prompt.len());

        let vocab = logits.shape().dim_from_end(0);
        let mut last_row = tail_row(&logits.to_f32(), vocab);

        let mut generated = Vec::with_capacity(self.config.max_new_tokens);
        for _ in 0..self.config.max_new_tokens {
            let next = self.config.sampler.sample(&last_row, &history, &mut self.rng) as u32;
            generated.push(next);
            history.push(next);
            if Some(next) == self.config.eos_token {
                break;
            }
            let step_ids = IdTensor::from_slice(&[next], vec![1, 1], device)?;
            let logits = self
                .model
                .forward_cached(&step_ids, cache.layers_mut())?;
            cache.advance(1);
            last_row = tail_row(&logits.to_f32(), vocab);
        }
        Ok(generated)
    }

    /// The sampler in use.
    pub fn sampler(&self) -> &SamplerConfig {
        &self.config.sampler
    }
}

/// The final `vocab` values of a flattened `[batch, seq, vocab]` logit tensor.
fn tail_row(flat: &[f32], vocab: usize) -> Vec<f32> {
    flat[flat.len() - vocab..].to_vec()
}
