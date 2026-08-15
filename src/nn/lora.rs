//! Low-rank adaptation.
//!
//! A LoRA adapter adds `x @ A @ B * (alpha / r)` to a frozen base projection.
//! Because [`Param`] is a shared handle, attaching an adapter to an existing
//! [`Linear`](crate::nn::Linear) neither copies nor rebuilds the base weight — and
//! [`LoraLinear::merge_into`] folds the update back so inference pays nothing.

use cubecl::prelude::Runtime;

use crate::autograd::Var;
use crate::backend::{Device, FloatElem};
use crate::error::{Error, Result};
use crate::nn::dropout::Dropout;
use crate::nn::init::Initializer;
use crate::nn::module::{Module, ModuleVisitor};
use crate::nn::param::Param;
use crate::tensor::Tensor;
use crate::tensor::ops::random::Rng;
use crate::tensor::ops::{elemwise, matmul as mm};

/// Adapter hyper-parameters.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct LoraConfig {
    /// Bottleneck width `r`.
    pub rank: usize,
    /// Scaling numerator; the update is multiplied by `alpha / rank`.
    pub alpha: f32,
    /// Dropout applied to the adapter input.
    pub dropout: f32,
    /// Initialiser for the down projection `A`. `B` always starts at zero so the
    /// adapter is a no-op at step 0.
    pub init: Initializer,
}

impl Default for LoraConfig {
    fn default() -> Self {
        Self {
            rank: 8,
            alpha: 16.0,
            dropout: 0.0,
            init: Initializer::KaimingUniform { gain: 1.0 },
        }
    }
}

impl LoraConfig {
    /// Start a builder.
    pub fn builder() -> LoraConfigBuilder {
        LoraConfigBuilder::default()
    }

    /// The multiplier applied to `A @ B`.
    pub fn scaling(&self) -> f32 {
        if self.rank == 0 {
            0.0
        } else {
            self.alpha / self.rank as f32
        }
    }

    /// Instantiate the adapter for an `[in_features, out_features]` projection.
    pub fn init<R: Runtime, E: FloatElem>(
        &self,
        in_features: usize,
        out_features: usize,
        device: &Device<R>,
        rng: &mut Rng,
    ) -> LoraLinear<R, E> {
        let a = Param::new(self.init.init(vec![in_features, self.rank], device, rng));
        let b = Param::new(Tensor::zeros(vec![self.rank, out_features], device));
        LoraLinear {
            a,
            b,
            scaling: self.scaling(),
            dropout: (self.dropout > 0.0).then(|| Dropout::new(self.dropout)),
        }
    }
}

/// Builder for [`LoraConfig`].
#[derive(Debug, Clone, Default)]
pub struct LoraConfigBuilder {
    rank: Option<usize>,
    alpha: Option<f32>,
    dropout: Option<f32>,
    init: Option<Initializer>,
}

impl LoraConfigBuilder {
    /// Bottleneck width.
    pub fn rank(mut self, rank: usize) -> Self {
        self.rank = Some(rank);
        self
    }

    /// Scaling numerator.
    pub fn alpha(mut self, alpha: f32) -> Self {
        self.alpha = Some(alpha);
        self
    }

    /// Dropout on the adapter input.
    pub fn dropout(mut self, p: f32) -> Self {
        self.dropout = Some(p);
        self
    }

    /// Initialiser for `A`.
    pub fn init(mut self, init: Initializer) -> Self {
        self.init = Some(init);
        self
    }

    /// Validate and build.
    pub fn build(self) -> Result<LoraConfig> {
        let defaults = LoraConfig::default();
        let rank = self.rank.unwrap_or(defaults.rank);
        if rank == 0 {
            return Err(Error::config("LoRA rank must be positive"));
        }
        let dropout = self.dropout.unwrap_or(defaults.dropout);
        if !(0.0..1.0).contains(&dropout) {
            return Err(Error::config(format!(
                "LoRA dropout must be in [0, 1), got {dropout}"
            )));
        }
        Ok(LoraConfig {
            rank,
            alpha: self.alpha.unwrap_or(defaults.alpha),
            dropout,
            init: self.init.unwrap_or(defaults.init),
        })
    }
}

/// A rank-`r` update to a linear projection.
pub struct LoraLinear<R: Runtime, E: FloatElem> {
    a: Param<R, E>,
    b: Param<R, E>,
    scaling: f32,
    dropout: Option<Dropout>,
}

impl<R: Runtime, E: FloatElem> core::fmt::Debug for LoraLinear<R, E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "LoraLinear(rank={}, scaling={})",
            self.a.shape().dim(1),
            self.scaling
        )
    }
}

impl<R: Runtime, E: FloatElem> LoraLinear<R, E> {
    /// The bottleneck width.
    pub fn rank(&self) -> usize {
        self.a.shape().dim(1)
    }

    /// The down projection `A`.
    pub fn a(&self) -> &Param<R, E> {
        &self.a
    }

    /// The up projection `B`.
    pub fn b(&self) -> &Param<R, E> {
        &self.b
    }

    /// The additive contribution for an input of shape `[..., in_features]`.
    pub fn delta(&self, input: &Var<R, E>) -> Result<Var<R, E>> {
        let x = match &self.dropout {
            Some(d) => d.apply(input)?,
            None => input.clone(),
        };
        let low = x.matmul(&self.a.var(input))?;
        Ok(low.matmul(&self.b.var(input))?.mul_scalar(self.scaling))
    }

    /// `A @ B * scaling`, shaped like the base weight `[in_features, out_features]`.
    pub fn delta_weight(&self) -> Result<Tensor<R, E>> {
        let product = mm::matmul(&self.a.value(), &self.b.value())?;
        Ok(elemwise::mul_scalar(&product, self.scaling))
    }

    /// Fold the adapter into a base weight and reset `B` to zero, so repeated calls
    /// are idempotent.
    pub fn merge_into(&self, base: &Param<R, E>) -> Result<()> {
        let merged = elemwise::add(&base.value(), &self.delta_weight()?)?;
        base.set(merged);
        self.b
            .set(Tensor::zeros(self.b.shape(), &self.b.value().device().clone()));
        Ok(())
    }
}

impl<R: Runtime, E: FloatElem> Module<R, E> for LoraLinear<R, E> {
    fn visit(&self, visitor: &mut ModuleVisitor<'_, R, E>) {
        visitor.param("lora_a", &self.a);
        visitor.param("lora_b", &self.b);
        if let Some(d) = &self.dropout {
            visitor.child("dropout", d);
        }
    }
}
