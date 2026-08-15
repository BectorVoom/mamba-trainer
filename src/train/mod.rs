//! Training: optimizers, schedules, losses, checkpoints and the loop that ties
//! them together.
//!
//! The loop is deliberately model-agnostic. It drives a [`TrainStep`], which
//! supplies parameters and a scalar loss; swapping a language model for a vision
//! model, or a full fine-tune for a LoRA run, changes only which `TrainStep` is
//! passed in.

pub mod checkpoint;
pub mod loss;
pub mod optim;
pub mod sched;
pub mod tasks;
pub mod trainer;

pub use checkpoint::Checkpoint;
pub use loss::{
    CrossEntropyConfig, accuracy, cross_entropy, cross_entropy_with, mae, mse, perplexity,
};
pub use optim::{AdamW, AdamWConfig, Optimizer, Sgd, clip_grad_norm, grad_norm};
pub use sched::LrSchedule;
pub use tasks::{ClassificationTask, ImageBatch, LmBatch, LmTask};
pub use trainer::{StepInfo, TrainReport, TrainStep, Trainer, TrainerConfig};
