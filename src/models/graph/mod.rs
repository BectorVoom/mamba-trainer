//! Graph Mamba Networks on Mamba-3 (GRAPH_MAMBA_PLAN.md).
//!
//! *Graph Mamba: Towards Learning on Graphs with State Space Models* (Behrouz
//! and Hashemi, KDD 2024) as a model family of this crate. A graph model in two
//! stages, both made of the crate's bidirectional Mamba-3 block:
//!
//! 1. **Per-node stage.** Each node gets a short sequence of *subgraph tokens*:
//!    samples of its 1-hop, 2-hop, … `m`-hop neighbourhood drawn with random
//!    walks, each encoded to one vector by a local encoder. A Mamba scans that
//!    sequence from the farthest neighbourhood inwards; the last position,
//!    which is the node itself, is the node's new encoding.
//! 2. **Node stage.** The node encodings of one graph, ordered by degree, form
//!    one long sequence. Bidirectional Mamba scans it, optionally summed with a
//!    message-passing layer over the real edges.
//!
//! The maximum walk length `m` switches between the two tokenisations:
//! `m >= 1` uses both stages, `m = 0` skips the first and is "GPS with the
//! Transformer replaced by bidirectional Mamba".
//!
//! # How it runs
//!
//! A dataset is uploaded once. After that, sampling the walks, building the
//! tokens, assembling the batch and computing the loss are device kernels
//! ([`crate::tensor::ops::graph`]): a training step uploads nothing and reads
//! nothing.
//!
//! # Deviations from the paper
//!
//! | Paper | Here |
//! |---|---|
//! | Two Mamba-1 blocks, summed, then `W_out` | one fused bidirectional Mamba-3 mixer: the same function class in one mixer's launches |
//! | Mamba-1 selective SSM | the Mamba-3 SSM (trapezoidal discretisation, rotational state) |
//! | `LayerNorm`, BatchNorm inside GatedGCN | `RmsNorm`, as everywhere else in the crate |
//! | local encoder = GatedGCN or walk features on the induced subgraph | its linear members, `Mean` and `Sgc`, with token statistics: they let tokens be built on the device from constants |
//! | walks sampled once, shuffled within a walk length | walks resampled on the device every step (equivalent in distribution) |
//! | full batch on one large graph | jittered stratified node partitions; one part is full batch |
//! | every stage-1 layer bidirectional | the last stage-1 layer is forward-only by default: only its last position is read |
//!
//! Out of scope: link prediction, class-weighted losses, a nonlinear local
//! encoder, message passing on directed graphs, datasets that do not fit on
//! the device, BatchNorm, virtual nodes and dataset downloaders.

pub mod batch;
pub mod data;
pub mod encoding;
pub mod layers;
pub mod loss;
pub mod metrics;
pub mod model;
pub mod spec;
pub mod store;
pub mod tokenize;

pub use batch::{
    DatasetOptions, Epoch, EpochOptions, GraphBatch, GraphDataset, MAX_BYTES_ENV, MemoryEstimate,
    PreparedDataset, TENSOR_MAX_ENV, max_bytes, tensor_threshold,
};
pub use data::{
    Bools, CanonicalGraph, CanonicalLabels, CanonicalizeOptions, EdgeFeatures, FeatureTable,
    Features, FeaturesView, Floats, GraphData, GraphDataView, HostCsr, Ints, Labels, LabelsView,
    NodeFeatures, NodeOrder, SPLIT_TEST, SPLIT_TRAIN, SPLIT_VAL, Split, Splits, SplitsView,
    canonicalize,
};
pub use encoding::{
    LAPLACIAN_MAX_NODES, LaplacianPe, RWSE_MAX_BALL, graph_offsets_of, laplacian_pe,
    laplacian_pe_csr, laplacian_pe_with, rwse, rwse_csr, rwse_with,
};
pub use layers::{GatedGcn, Gine, LocalEncoderModule};
pub use loss::{GraphTask, SafeTargets, graph_loss, safe_targets};
pub use model::{
    EvalOptions, Evaluation, GraphMamba, GraphTrainConfig, GraphTrainer, Metric,
};
pub use spec::{
    BatchMode, FeatureSpec, GraphMambaSpec, GraphPool, GraphTaskSpec, LocalEncoder, MAX_MULTI_HOT,
    MpnnKind, RegressionLoss, TokenSampling, TokenTail, WalkTokens,
};
pub use store::{FeatureStore, GraphStore, TargetStore};
pub use tokenize::{HostTokens, hash_u32_host, tokens_host};
