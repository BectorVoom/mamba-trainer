//! Multi-head attention with grouped-query support and rotary positions.
//!
//! This exists so Mamba-3 can be mixed with attention: see
//! [`crate::models::hybrid`]. It keeps the same `Module`/builder conventions as the
//! rest of the crate, so a hybrid stack inherits LoRA and QAT from its projections
//! without any extra plumbing.

use cubecl::prelude::Runtime;

use crate::autograd::{Var, cat};
use crate::backend::{Device, FloatElem};
use crate::error::{Error, Result};
use crate::nn::dropout::Dropout;
use crate::nn::linear::{Linear, LinearConfig};
use crate::nn::lora::LoraConfig;
use crate::nn::module::{Module, ModuleVisitor};
use crate::nn::quant::QuantConfig;
use crate::nn::rope::RotaryEmbedding;
use crate::tensor::Tensor;
use crate::tensor::ops::random::Rng;

/// Per-layer key/value cache for incremental decoding.
pub struct AttentionCache<R: Runtime, E: FloatElem> {
    keys: Option<Tensor<R, E>>,
    values: Option<Tensor<R, E>>,
}

impl<R: Runtime, E: FloatElem> Default for AttentionCache<R, E> {
    fn default() -> Self {
        Self {
            keys: None,
            values: None,
        }
    }
}

impl<R: Runtime, E: FloatElem> AttentionCache<R, E> {
    /// An empty cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of positions already cached.
    pub fn len(&self) -> usize {
        self.keys.as_ref().map(|k| k.shape().dim(2)).unwrap_or(0)
    }

    /// Whether nothing has been cached yet.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Forget everything.
    pub fn reset(&mut self) {
        self.keys = None;
        self.values = None;
    }
}

/// Configuration for [`MultiHeadAttention`].
#[derive(Debug, Clone)]
pub struct AttentionConfig {
    d_model: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: Option<usize>,
    bias: bool,
    causal: bool,
    dropout: f32,
    rope: bool,
    rope_base: f32,
    max_seq_len: usize,
    lora: Option<LoraConfig>,
    weight_quant: Option<QuantConfig>,
}

impl AttentionConfig {
    /// Standard multi-head self-attention.
    pub fn new(d_model: usize, n_heads: usize) -> Self {
        Self {
            d_model,
            n_heads,
            n_kv_heads: n_heads,
            head_dim: None,
            bias: false,
            causal: true,
            dropout: 0.0,
            rope: true,
            rope_base: 10000.0,
            max_seq_len: 4096,
            lora: None,
            weight_quant: None,
        }
    }

    /// Retarget the configuration at a different residual width, keeping every
    /// other setting. Used by stacks that own `d_model`.
    pub fn with_d_model(mut self, d_model: usize) -> Self {
        self.d_model = d_model;
        self
    }

    /// Residual width.
    pub fn d_model_value(&self) -> usize {
        self.d_model
    }

    /// Number of query heads.
    pub fn n_heads_value(&self) -> usize {
        self.n_heads
    }

    /// Longest sequence the rotary tables cover.
    pub fn max_seq_len_value(&self) -> usize {
        self.max_seq_len
    }

    /// Use grouped-query attention with `n_kv_heads` key/value heads.
    pub fn with_kv_heads(mut self, n_kv_heads: usize) -> Self {
        self.n_kv_heads = n_kv_heads;
        self
    }

    /// Override the per-head width (defaults to `d_model / n_heads`).
    pub fn with_head_dim(mut self, head_dim: usize) -> Self {
        self.head_dim = Some(head_dim);
        self
    }

    /// Enable projection biases.
    pub fn with_bias(mut self, bias: bool) -> Self {
        self.bias = bias;
        self
    }

    /// Apply a causal mask. Turn this off for vision encoders.
    pub fn with_causal(mut self, causal: bool) -> Self {
        self.causal = causal;
        self
    }

    /// Dropout on the attention probabilities.
    pub fn with_dropout(mut self, p: f32) -> Self {
        self.dropout = p;
        self
    }

    /// Enable or disable rotary positions.
    pub fn with_rope(mut self, rope: bool) -> Self {
        self.rope = rope;
        self
    }

    /// Rotary frequency base.
    pub fn with_rope_base(mut self, base: f32) -> Self {
        self.rope_base = base;
        self
    }

    /// Longest sequence the rotary tables and causal masks must cover.
    pub fn with_max_seq_len(mut self, len: usize) -> Self {
        self.max_seq_len = len;
        self
    }

    /// Attach LoRA adapters to every projection.
    pub fn with_lora(mut self, lora: LoraConfig) -> Self {
        self.lora = Some(lora);
        self
    }

    /// Fake-quantize every projection weight.
    pub fn with_weight_quant(mut self, config: QuantConfig) -> Self {
        self.weight_quant = Some(config);
        self
    }

    fn linear(&self, d_in: usize, d_out: usize) -> LinearConfig {
        let mut cfg = LinearConfig::new(d_in, d_out).with_bias(self.bias);
        if let Some(lora) = self.lora {
            cfg = cfg.with_lora(lora);
        }
        if let Some(q) = self.weight_quant {
            cfg = cfg.with_weight_quant(q);
        }
        cfg
    }

    /// Instantiate on a device.
    pub fn init<R: Runtime, E: FloatElem>(
        &self,
        device: &Device<R>,
        rng: &mut Rng,
    ) -> Result<MultiHeadAttention<R, E>> {
        let head_dim = self.head_dim.unwrap_or(self.d_model / self.n_heads);
        if self.n_heads % self.n_kv_heads != 0 {
            return Err(Error::config(format!(
                "n_heads ({}) must be a multiple of n_kv_heads ({})",
                self.n_heads, self.n_kv_heads
            )));
        }
        let q_dim = self.n_heads * head_dim;
        let kv_dim = self.n_kv_heads * head_dim;
        Ok(MultiHeadAttention {
            q_proj: self.linear(self.d_model, q_dim).init(device, rng),
            k_proj: self.linear(self.d_model, kv_dim).init(device, rng),
            v_proj: self.linear(self.d_model, kv_dim).init(device, rng),
            o_proj: self.linear(q_dim, self.d_model).init(device, rng),
            rope: self
                .rope
                .then(|| {
                    RotaryEmbedding::new(head_dim, self.max_seq_len, self.rope_base, device)
                })
                .transpose()?,
            dropout: (self.dropout > 0.0).then(|| Dropout::new(self.dropout)),
            n_heads: self.n_heads,
            n_kv_heads: self.n_kv_heads,
            head_dim,
            causal: self.causal,
        })
    }
}

/// Multi-head attention.
pub struct MultiHeadAttention<R: Runtime, E: FloatElem> {
    q_proj: Linear<R, E>,
    k_proj: Linear<R, E>,
    v_proj: Linear<R, E>,
    o_proj: Linear<R, E>,
    rope: Option<RotaryEmbedding<R, E>>,
    dropout: Option<Dropout>,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    causal: bool,
}

impl<R: Runtime, E: FloatElem> core::fmt::Debug for MultiHeadAttention<R, E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "MultiHeadAttention(heads={}, kv_heads={}, head_dim={}, causal={})",
            self.n_heads, self.n_kv_heads, self.head_dim, self.causal
        )
    }
}

impl<R: Runtime, E: FloatElem> MultiHeadAttention<R, E> {
    /// Number of query heads.
    pub fn n_heads(&self) -> usize {
        self.n_heads
    }

    /// Per-head width.
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Attend over `[batch, seq, d_model]`.
    pub fn apply(&self, input: &Var<R, E>) -> Result<Var<R, E>> {
        self.apply_cached(input, None)
    }

    /// Attend, optionally extending and reading a key/value cache.
    pub fn apply_cached(
        &self,
        input: &Var<R, E>,
        mut cache: Option<&mut AttentionCache<R, E>>,
    ) -> Result<Var<R, E>> {
        input.shape().expect_rank(3)?;
        let dims = input.dims().to_vec();
        let (b, t) = (dims[0], dims[1]);
        let past = cache.as_ref().map(|c| c.len()).unwrap_or(0);

        // [b, t, h, d] -> [b, h, t, d]
        let to_heads = |x: &Var<R, E>, heads: usize| -> Result<Var<R, E>> {
            x.reshape(vec![b, t, heads, self.head_dim])?
                .permute(&[0, 2, 1, 3])
        };

        let mut q = to_heads(&self.q_proj.apply(input)?, self.n_heads)?;
        let mut k = to_heads(&self.k_proj.apply(input)?, self.n_kv_heads)?;
        let v = to_heads(&self.v_proj.apply(input)?, self.n_kv_heads)?;

        if let Some(rope) = &self.rope {
            q = rope.apply(&q, past)?;
            k = rope.apply(&k, past)?;
        }

        // Extend the cache, then attend over the whole history.
        let (k, v) = match cache.as_deref_mut() {
            Some(c) => {
                let k_full = match &c.keys {
                    Some(prev) => cat(
                        &[Var::constant(prev.clone()), k.clone()],
                        2,
                    )?,
                    None => k.clone(),
                };
                let v_full = match &c.values {
                    Some(prev) => cat(
                        &[Var::constant(prev.clone()), v.clone()],
                        2,
                    )?,
                    None => v.clone(),
                };
                c.keys = Some(k_full.tensor().clone());
                c.values = Some(v_full.tensor().clone());
                (k_full, v_full)
            }
            None => (k, v),
        };

        let source = k.shape().dim(2);
        let (k, v) = if self.n_kv_heads == self.n_heads {
            (k, v)
        } else {
            let repeat = self.n_heads / self.n_kv_heads;
            let grow = |x: &Var<R, E>| -> Result<Var<R, E>> {
                x.reshape(vec![b, self.n_kv_heads, 1, source, self.head_dim])?
                    .expand(vec![b, self.n_kv_heads, repeat, source, self.head_dim])?
                    .reshape(vec![b, self.n_heads, source, self.head_dim])
            };
            (grow(&k)?, grow(&v)?)
        };

        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let scores = q.matmul(&k.transpose()?)?.mul_scalar(scale);

        let scores = if self.causal {
            let mask = causal_bias::<R, E>(t, source, past, input.device());
            scores.add(&Var::constant(mask))?
        } else {
            scores
        };

        let mut weights = scores.softmax(3)?;
        if let Some(d) = &self.dropout {
            weights = d.apply(&weights)?;
        }

        let context = weights
            .matmul(&v)?
            .permute(&[0, 2, 1, 3])?
            .reshape(vec![b, t, self.n_heads * self.head_dim])?;
        self.o_proj.apply(&context)
    }
}

/// An additive mask: `0` where attention is allowed, a large negative value where
/// it is not. Query `i` sits at absolute position `past + i`.
fn causal_bias<R: Runtime, E: FloatElem>(
    queries: usize,
    keys: usize,
    past: usize,
    device: &Device<R>,
) -> Tensor<R, E> {
    let mut data = vec![0.0f32; queries * keys];
    for i in 0..queries {
        for j in 0..keys {
            if j > past + i {
                data[i * keys + j] = -1.0e9;
            }
        }
    }
    Tensor::from_f32(&data, vec![1, 1, queries, keys], device)
        .expect("mask shape is consistent")
}

impl<R: Runtime, E: FloatElem> Module<R, E> for MultiHeadAttention<R, E> {
    fn visit(&self, visitor: &mut ModuleVisitor<'_, R, E>) {
        visitor.child("q_proj", &self.q_proj);
        visitor.child("k_proj", &self.k_proj);
        visitor.child("v_proj", &self.v_proj);
        visitor.child("o_proj", &self.o_proj);
        if let Some(d) = &self.dropout {
            visitor.child("dropout", d);
        }
    }
}
