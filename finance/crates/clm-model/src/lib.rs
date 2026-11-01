//! Mamba-1 encoder, contrastive state-action decision model (CLM), baselines,
//! losses and the per-fold trainer for Mamba-CLM.
//!
//! * [`mamba`]: the Mamba-1 encoder on the CubeCL selective-scan kernels.
//! * [`tcn`]: the convolutional baseline encoder.
//! * [`heads`]: contrastive state/action head, MLP and auxiliary heads.
//! * [`model`]: [`DecisionModel`], one config per compared model.
//! * [`losses`]: InfoNCE (multi-positive), supervised contrastive, auxiliary.
//! * [`train`]: [`fit`] with early stopping on validation log loss.
//! * [`predict`]: inference over a split.
//! * [`reference`]: an `f64` CPU reference of the Mamba block, for equivalence tests.

pub mod batch;
pub mod heads;
pub mod losses;
pub mod mamba;
pub mod model;
pub mod predict;
pub mod reference;
pub mod source;
pub mod tcn;
pub mod train;

pub use batch::{Batch, BatchSource, Split};
pub use heads::{ContrastiveConfig, Scoring, Temperature, action_attributes};
pub use losses::LossConfig;
pub use mamba::{MambaConfig, Pooling};
pub use model::{DecisionModel, EncoderConfig, HeadConfig, ModelConfig, Outputs};
pub use predict::{SplitPredictions, predict_split, softmax3};
pub use source::{PanelSource, SourceConfig, input_valid};
pub use tcn::TcnConfig;
pub use train::{EvalPoint, FitReport, TrainConfig, fit};
