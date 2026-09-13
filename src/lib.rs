//! # mamba3
//!
//! Mamba-3 state space models in Rust, on [CubeCL](https://github.com/tracel-ai/cubecl).
//!
//! The crate is built as five stacked layers; each one is useful on its own and
//! each one is generic over the CubeCL runtime `R` and the float element `E`.
//!
//! | layer | module | what it owns |
//! |---|---|---|
//! | 1 | [`tensor`] | contiguous device buffers + raw kernels |
//! | 2 | [`autograd`] | tape-based reverse-mode differentiation |
//! | 3 | [`nn`] | parameters, modules, initializers, LoRA, quantization |
//! | 4 | [`ssm`] / [`models`] | the Mamba-3 mixer and the model zoo |
//! | 5 | [`train`] / [`infer`] / [`rl`] | optimizers, schedules, trainer, KV/SSM caches, rollout engines |
//! | 5 | [`distributions`] | probability distributions, sampled and scored on device |
//!
//! ## Quick start
//!
//! ```no_run
//! use mamba3::prelude::*;
//!
//! type R = mamba3::backends::Auto;
//!
//! let device = Device::<R>::default();
//! let config = Mamba3LmConfig::builder()
//!     .vocab_size(1000)
//!     .d_model(128)
//!     .n_layers(4)
//!     .build()
//!     .unwrap();
//! let model = config.init::<R, f32>(&device);
//! ```
//!
//! ## Extension points
//!
//! * **Vision** — [`models::vision`] reuses the same mixer with a patch embedding
//!   and bidirectional scanning.
//! * **Hybrids** — [`models::hybrid`] interleaves Mamba-3 mixers with attention
//!   blocks through a declarative [`models::hybrid::LayerPattern`].
//! * **QAT** — [`nn::quant`] provides observers and fake-quant that plug into any
//!   [`nn::Linear`] via its builder.
//! * **LoRA** — [`nn::lora`] adapts the same `Linear`, with merge/unmerge.
//! * **Inference** — [`infer`] holds the recurrent state cache and samplers.
//! * **Reinforcement learning** — [`rl`] pairs an `O(1)` rollout engine over `B`
//!   environments with the `O(T)` scan that trains on what it collected, both
//!   cutting the recurrence at episode boundaries so the two agree. On top of that
//!   pair sit PPO ([`rl::ppo`]) and imitation learning ([`rl::imitation`]), whose
//!   advantage estimator, action sampler and trajectory writes are all device
//!   kernels ([`tensor::ops::rl`]) — so a collection loop never reads back.
//! * **Distributions** — [`distributions`] is `torch.distributions` as CubeCL
//!   kernels: thirty-odd families that sample, score, differentiate and diverge
//!   without leaving the device, each operation one fused launch. It is what a
//!   policy is made of, discrete or continuous.

#![warn(missing_docs)]
#![allow(clippy::too_many_arguments)]

/// The CubeCL version this crate is built against.
///
/// Re-exported so that a downstream crate writing `#[cube]` game logic for
/// [`rl::GameLogic`] compiles it against the same version the kernels here were
/// generated from, rather than whichever one its own `Cargo.toml` resolves to.
pub use cubecl;

pub mod autograd;
pub mod backend;
pub mod distributions;
pub mod error;
pub mod infer;
pub mod models;
pub mod nn;
pub mod rl;
pub mod ssm;
pub mod tensor;
pub mod train;

/// Concrete runtime aliases for the backends enabled by feature flags.
pub mod backends {
    /// Multi-threaded CPU runtime (MLIR/LLVM JIT).
    #[cfg(feature = "cpu")]
    pub type Cpu = cubecl::cpu::CpuRuntime;

    /// WebGPU / Vulkan / Metal / DX12 runtime.
    #[cfg(feature = "wgpu")]
    pub type Wgpu = cubecl::wgpu::WgpuRuntime;

    /// NVIDIA CUDA runtime.
    #[cfg(feature = "cuda")]
    pub type Cuda = cubecl::cuda::CudaRuntime;

    /// AMD ROCm/HIP runtime.
    #[cfg(feature = "hip")]
    pub type Hip = cubecl::hip::HipRuntime;

    /// The runtime a program gets when it does not name one.
    ///
    /// Resolved from the enabled feature flags, most specialised first: CUDA, HIP,
    /// wgpu, CPU. Every test and example in the repository binds to this alias
    /// rather than to a concrete runtime, which is what lets
    /// `cargo test --no-default-features --features hip` run the whole suite on a
    /// GPU without editing a line of it.
    #[cfg(feature = "cuda")]
    pub type Auto = Cuda;
    /// See [`Auto`].
    #[cfg(all(feature = "hip", not(feature = "cuda")))]
    pub type Auto = Hip;
    /// See [`Auto`].
    #[cfg(all(feature = "wgpu", not(feature = "cuda"), not(feature = "hip")))]
    pub type Auto = Wgpu;
    /// See [`Auto`].
    #[cfg(all(
        feature = "cpu",
        not(feature = "cuda"),
        not(feature = "hip"),
        not(feature = "wgpu")
    ))]
    pub type Auto = Cpu;
}

/// The imports most users want.
pub mod prelude {
    pub use crate::autograd::{Grads, Var};
    pub use crate::backend::{DType, Device, FloatElem};
    pub use crate::error::{Error, Result};
    pub use crate::infer::{Generator, GeneratorConfig, SamplerConfig, StateCache};
    pub use crate::models::hybrid::{HybridConfig, LayerKind, LayerPattern};
    pub use crate::models::lm::{Mamba3Lm, Mamba3LmConfig};
    pub use crate::models::mamba3::{Mamba3Block, Mamba3BlockConfig};
    pub use crate::models::vision::{VisionMamba3, VisionMamba3Config};
    pub use crate::nn::{
        Initializer, Linear, LinearConfig, Module, ModuleVisitor, Param, ParamId, RmsNorm,
        RmsNormConfig,
    };
    pub use crate::nn::lora::{LoraConfig, LoraLinear};
    pub use crate::rl::{
        BehaviourCloningTask, CollectReport, Collector, FusedStep, GameLogic, GameSpec, GameWorld,
        ImitationBatch, Mamba3Policy, Mamba3PolicyConfig, Mamba3StateBuffer,
        MultiSyncCollector, Outcome, ParallelEnvs, PpoBatch, PpoConfig, PpoTask,
        RolloutEngine, TrajectoryBuffer, VecEnv,
    };
    pub use crate::nn::quant::{QuantConfig, QuantScheme, Quantizer};
    pub use crate::ssm::config::{Discretization, SsmConfig, StateDynamics, SsmMode};
    pub use crate::distributions::{
        Categorical, Dirichlet, Distribution, Independent, MultivariateNormal, Univariate,
        kl_divergence,
    };
    pub use crate::tensor::{Shape, Tensor};
    pub use crate::train::{
        AdamW, AdamWConfig, LrSchedule, Optimizer, Trainer, TrainerConfig, cross_entropy,
    };
}
