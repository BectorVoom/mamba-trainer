//! The graph model's specification (GRAPH_MAMBA_PLAN.md §2.4).
//!
//! A [`GraphMambaSpec`] says everything about a model and about how its data is
//! laid out — feature widths, the token shape, the two stages' depth, the
//! message passing, the node order, the task — and nothing about a particular
//! dataset. It is plain data: it serialises to JSON and compares by value.

use crate::error::{Error, Result};
use crate::models::graph::data::NodeOrder;
use crate::models::vision::ScanDirection;
use crate::ssm::config::SsmConfig;
use crate::tensor::ops::graph::WalkShape;

/// The widest multi-hot a categorical feature set may expand to.
pub const MAX_MULTI_HOT: usize = 1024;

/// An error that names the spec field it is about.
fn field(name: &str, msg: impl core::fmt::Display) -> Error {
    Error::config(format!("{name}: {msg}"))
}

/// What a node or an edge carries.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum FeatureSpec {
    /// `dim` floats.
    Float {
        /// Floats per row.
        dim: usize,
    },
    /// One id per field; field `f` takes values in `0..vocab[f]`. The model
    /// reads them as one multi-hot of width `Σ vocab`.
    Categorical {
        /// Vocabulary size of each field.
        vocab: Vec<usize>,
    },
}

impl FeatureSpec {
    /// Width of the model input: `dim`, or the multi-hot width.
    pub fn width(&self) -> usize {
        match self {
            FeatureSpec::Float { dim } => *dim,
            FeatureSpec::Categorical { vocab } => vocab.iter().sum(),
        }
    }

    fn validate(&self, name: &str) -> Result<()> {
        match self {
            FeatureSpec::Float { dim: 0 } => Err(field(name, "needs at least one float per row")),
            FeatureSpec::Float { .. } => Ok(()),
            FeatureSpec::Categorical { vocab } => {
                if vocab.is_empty() || vocab.contains(&0) {
                    return Err(field(
                        name,
                        "needs at least one id field, each with a non-empty vocabulary",
                    ));
                }
                if self.width() > MAX_MULTI_HOT {
                    return Err(field(
                        name,
                        format!(
                            "the multi-hot width {} is above the limit of {MAX_MULTI_HOT}",
                            self.width()
                        ),
                    ));
                }
                Ok(())
            }
        }
    }
}

/// The shape of a node's token sequence: `max_hops` walk lengths, `repeats`
/// tokens of each, `walks` random walks per token.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct WalkTokens {
    /// Maximum walk length `m`; 0 turns the per-node stage off.
    pub max_hops: usize,
    /// Walks per token `M`.
    pub walks: usize,
    /// Tokens per walk length `s`.
    pub repeats: usize,
}

impl WalkTokens {
    /// Tokens per node, `L = m·s + 1`.
    pub const fn len(&self) -> usize {
        self.max_hops * self.repeats + 1
    }

    /// Never: a sequence always holds the node itself.
    pub const fn is_empty(&self) -> bool {
        false
    }

    /// Node slots per token, `C = 1 + M·m`.
    pub const fn cap(&self) -> usize {
        1 + self.walks * self.max_hops
    }
}

/// When a node's tokens are resampled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum TokenSampling {
    /// Every training step (the default): what the paper's shuffle within a
    /// walk length asks for, with more samples.
    PerStep,
    /// Once per epoch.
    PerEpoch,
    /// Once: the paper's sample-before-training. Evaluation always uses this.
    Static,
}

/// How a subgraph token becomes one vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum LocalEncoder {
    /// The mean of the token's nodes.
    Mean,
    /// One step of normalised propagation inside the token: its nodes weighted
    /// by their degree in the induced subgraph. Only `hops = 1` is built.
    Sgc {
        /// Propagation steps.
        hops: usize,
    },
    /// A message-passing network on the induced subgraph, as in the paper.
    /// Reserved: not built.
    Mpnn {
        /// Message-passing layers.
        layers: usize,
    },
}

/// The last layer of the per-node stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum TokenTail {
    /// Forward-only, finishing only the last position (the default): there the
    /// backward direction has seen nothing but that one token.
    Forward,
    /// Bidirectional, as the paper writes it, at about twice the cost.
    Bidirectional,
}

/// The message-passing layer summed with the node-stage mixer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum MpnnKind {
    /// GIN with edge features (`ε = 0`).
    Gine,
    /// Residual gated graph convolution, with edge states carried across
    /// layers. Whole-graph batches only.
    GatedGcn,
}

/// How a graph's node rows become one vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum GraphPool {
    /// The mean over the graph's nodes.
    Mean,
    /// Their sum.
    Sum,
}

/// The per-target loss of a regression task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum RegressionLoss {
    /// Absolute error (the default).
    L1,
    /// Squared error.
    Mse,
}

/// What the model predicts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum GraphTaskSpec {
    /// One of `classes` per node.
    NodeClass {
        /// Number of classes.
        classes: usize,
    },
    /// One of `classes` per graph.
    GraphClass {
        /// Number of classes.
        classes: usize,
        /// The readout.
        pool: GraphPool,
    },
    /// `targets` floats per graph.
    GraphRegression {
        /// Number of targets.
        targets: usize,
        /// The readout.
        pool: GraphPool,
        /// The per-target loss.
        loss: RegressionLoss,
    },
    /// `labels` independent binary labels per graph.
    GraphMultiLabel {
        /// Number of labels.
        labels: usize,
        /// The readout.
        pool: GraphPool,
    },
}

impl GraphTaskSpec {
    /// Width of the head's output.
    pub fn outputs(&self) -> usize {
        match *self {
            GraphTaskSpec::NodeClass { classes } | GraphTaskSpec::GraphClass { classes, .. } => {
                classes
            }
            GraphTaskSpec::GraphRegression { targets, .. } => targets,
            GraphTaskSpec::GraphMultiLabel { labels, .. } => labels,
        }
    }

    /// The readout of a graph task; `None` for a node task.
    pub fn pool(&self) -> Option<GraphPool> {
        match *self {
            GraphTaskSpec::NodeClass { .. } => None,
            GraphTaskSpec::GraphClass { pool, .. }
            | GraphTaskSpec::GraphRegression { pool, .. }
            | GraphTaskSpec::GraphMultiLabel { pool, .. } => Some(pool),
        }
    }

    /// Whether the model predicts per graph.
    pub fn per_graph(&self) -> bool {
        self.pool().is_some()
    }
}

/// How a batch is cut from a dataset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum BatchMode {
    /// Whole graphs, many per batch.
    Graphs,
    /// A stratified subset of the nodes of one large graph.
    NodeSubset,
}

/// A Graph Mamba model, completely.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct GraphMambaSpec {
    /// Node features.
    pub node_features: FeatureSpec,
    /// Edge features; they need message passing to be read.
    pub edge_features: Option<FeatureSpec>,
    /// Width of the positional / structural encoding; 0 for none.
    pub pe_dim: usize,
    /// Encoding columns `start..end` whose sign flips per graph while training
    /// (Laplacian eigenvectors are defined up to sign).
    pub pe_sign_flip: Option<(usize, usize)>,
    /// Model width.
    pub d_model: usize,
    /// The token shape; `max_hops = 0` gives node tokens only.
    pub tokens: WalkTokens,
    /// When tokens are resampled while training.
    pub token_sampling: TokenSampling,
    /// The local encoder of a token.
    pub local: LocalEncoder,
    /// Layers of the per-node stage; at least 1 exactly when `max_hops >= 1`.
    pub token_layers: usize,
    /// The last layer of the per-node stage.
    pub token_tail: TokenTail,
    /// Layers of the node stage, at least 1.
    pub node_layers: usize,
    /// Message passing summed with the node-stage mixer.
    pub mpnn: Option<MpnnKind>,
    /// Scan direction of the node stage; `Forward` is an ablation.
    pub direction: ScanDirection,
    /// A single graph's nodes as this many interleaved sequences.
    pub node_sequences: usize,
    /// The node order within each graph.
    pub order: NodeOrder,
    /// What the model predicts.
    pub task: GraphTaskSpec,
    /// The per-node stage's mixer.
    pub token_ssm: SsmConfig,
    /// The node stage's mixer.
    pub node_ssm: SsmConfig,
    /// A batch's row capacity is a multiple of this.
    pub row_quantum: usize,
    /// Dropout in the feed-forward blocks and the head.
    pub dropout: f32,
    /// Norm epsilon.
    pub norm_eps: f32,
    /// Seed of the initialisation, the epochs and the tokens.
    pub seed: u64,
}

/// The lean mixer both stages default to: one head per direction as wide as
/// the model, state size 8.
fn lean_ssm(d_model: usize, heads: usize, d_state: usize) -> SsmConfig {
    SsmConfig {
        d_model,
        n_heads: heads,
        head_dim: d_model,
        d_state,
        n_groups: heads,
        ..SsmConfig::default()
    }
}

/// Scalars of one Mamba-3 mixer built from `ssm`, with heads and groups
/// doubled when bidirectional.
fn mixer_parameters(ssm: &SsmConfig, bidirectional: bool) -> usize {
    let mut cfg = ssm.clone();
    if bidirectional {
        cfg.n_heads *= 2;
        cfg.n_groups *= 2;
    }
    let (d, inner, heads) = (cfg.d_model, cfg.d_inner(), cfg.n_heads);
    let bias = cfg.bias as usize;
    // The two projections.
    d * cfg.in_proj_width() + bias * cfg.in_proj_width() + inner * d + bias * d
        // The depthwise convolution: `kernel` taps and a bias per channel.
        + cfg.conv_kernel.map_or(0, |k| (k + 1) * (inner + 2 * cfg.bc_width()))
        // dt_bias and a_log, then the skip, the B/C biases and their norm.
        + 2 * heads
        + cfg.skip_connection as usize * heads
        + cfg.bc_bias as usize * 2 * heads * cfg.d_state
        + cfg.bc_norm as usize * cfg.d_state
        + cfg.post_gate_norm as usize * inner
}

impl GraphMambaSpec {
    /// A spec with the defaults: width 64, tokens of up to 4 hops (8 walks,
    /// 4 repeats) through one forward token layer, two bidirectional node
    /// layers, no message passing, ascending-degree order.
    pub fn new(node_features: FeatureSpec, task: GraphTaskSpec) -> Self {
        let d_model = 64;
        Self {
            node_features,
            edge_features: None,
            pe_dim: 0,
            pe_sign_flip: None,
            d_model,
            tokens: WalkTokens {
                max_hops: 4,
                walks: 8,
                repeats: 4,
            },
            token_sampling: TokenSampling::PerStep,
            local: LocalEncoder::Sgc { hops: 1 },
            token_layers: 1,
            token_tail: TokenTail::Forward,
            node_layers: 2,
            mpnn: None,
            direction: ScanDirection::Bidirectional,
            node_sequences: 1,
            order: NodeOrder::default(),
            task,
            token_ssm: lean_ssm(d_model, 1, 8),
            // Padded lengths are multiples of 32, so chunks of 32 never pad.
            node_ssm: SsmConfig {
                chunk_size: 32,
                ..lean_ssm(d_model, 1, 8)
            },
            row_quantum: 256,
            dropout: 0.0,
            norm_eps: 1e-5,
            seed: 0,
        }
    }

    /// Set the model width; both mixers follow (their head width is the model
    /// width).
    pub fn with_d_model(mut self, d_model: usize) -> Self {
        self.d_model = d_model;
        for ssm in [&mut self.token_ssm, &mut self.node_ssm] {
            ssm.d_model = d_model;
            ssm.head_dim = d_model;
        }
        self
    }

    /// Set the token shape. `max_hops = 0` turns the per-node stage off and
    /// takes its layers with it; a positive `max_hops` gives it one layer if it
    /// had none.
    pub fn with_tokens(mut self, max_hops: usize, walks: usize, repeats: usize) -> Self {
        self.tokens = WalkTokens {
            max_hops,
            walks,
            repeats,
        };
        if max_hops == 0 {
            self.token_layers = 0;
        } else if self.token_layers == 0 {
            self.token_layers = 1;
        }
        self
    }

    /// Set when tokens are resampled.
    pub fn with_token_sampling(mut self, sampling: TokenSampling) -> Self {
        self.token_sampling = sampling;
        self
    }

    /// Set the local encoder.
    pub fn with_local(mut self, local: LocalEncoder) -> Self {
        self.local = local;
        self
    }

    /// Set the number of per-node layers.
    pub fn with_token_layers(mut self, layers: usize) -> Self {
        self.token_layers = layers;
        self
    }

    /// Set the last per-node layer's direction.
    pub fn with_token_tail(mut self, tail: TokenTail) -> Self {
        self.token_tail = tail;
        self
    }

    /// Set the number of node-stage layers.
    pub fn with_node_layers(mut self, layers: usize) -> Self {
        self.node_layers = layers;
        self
    }

    /// Set the message passing.
    pub fn with_mpnn(mut self, mpnn: Option<MpnnKind>) -> Self {
        self.mpnn = mpnn;
        self
    }

    /// Set the edge features.
    pub fn with_edge_features(mut self, edge_features: Option<FeatureSpec>) -> Self {
        self.edge_features = edge_features;
        self
    }

    /// Set the width of the positional / structural encoding.
    pub fn with_pe_dim(mut self, pe_dim: usize) -> Self {
        self.pe_dim = pe_dim;
        self
    }

    /// Set the encoding columns whose sign flips per graph while training.
    pub fn with_pe_sign_flip(mut self, columns: Option<(usize, usize)>) -> Self {
        self.pe_sign_flip = columns;
        self
    }

    /// Set the node stage's scan direction.
    pub fn with_direction(mut self, direction: ScanDirection) -> Self {
        self.direction = direction;
        self
    }

    /// Set the number of interleaved node sequences of a single graph.
    pub fn with_node_sequences(mut self, sequences: usize) -> Self {
        self.node_sequences = sequences;
        self
    }

    /// Set the node order.
    pub fn with_order(mut self, order: NodeOrder) -> Self {
        self.order = order;
        self
    }

    /// Set the state size of both mixers.
    pub fn with_d_state(mut self, d_state: usize) -> Self {
        self.token_ssm.d_state = d_state;
        self.node_ssm.d_state = d_state;
        self
    }

    /// Set the per-node mixer's heads (per direction); groups follow.
    pub fn with_token_heads(mut self, heads: usize) -> Self {
        self.token_ssm.n_heads = heads;
        self.token_ssm.n_groups = heads;
        self
    }

    /// Set the node mixer's heads (per direction); groups follow.
    pub fn with_node_heads(mut self, heads: usize) -> Self {
        self.node_ssm.n_heads = heads;
        self.node_ssm.n_groups = heads;
        self
    }

    /// Set the dropout probability.
    pub fn with_dropout(mut self, dropout: f32) -> Self {
        self.dropout = dropout;
        self
    }

    /// Set the row quantum of a batch's capacity.
    pub fn with_row_quantum(mut self, quantum: usize) -> Self {
        self.row_quantum = quantum;
        self
    }

    /// Set the seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// The chunk length the per-node stage scans its `L` tokens in: the
    /// token mixer's `chunk_size` is an upper bound, and the sequence is cut
    /// into the fewest equal chunks under it, so a composed scan pads at most
    /// a position per chunk (65 tokens at 64 scan as two chunks of 33, not as
    /// 128 positions).
    pub fn token_chunk(&self) -> usize {
        let len = self.tokens.len();
        let chunks = len.div_ceil(self.token_ssm.chunk_size.max(1));
        len.div_ceil(chunks)
    }

    /// Width of a node's (or a token's) input: features next to the encoding.
    pub fn input_width(&self) -> usize {
        self.node_features.width() + self.pe_dim
    }

    /// Whether the per-node stage runs.
    pub fn has_token_stage(&self) -> bool {
        self.tokens.max_hops >= 1
    }

    /// The token shape as the token kernel takes it; `None` without the
    /// per-node stage.
    pub fn walk_shape(&self) -> Option<WalkShape> {
        self.has_token_stage().then_some(WalkShape {
            hops: self.tokens.max_hops,
            walks: self.tokens.walks,
            repeats: self.tokens.repeats,
            sgc: matches!(self.local, LocalEncoder::Sgc { .. }),
        })
    }

    /// Whether a dataset for this spec needs the reverse-edge tables.
    pub fn needs_reverse_index(&self) -> bool {
        self.edge_features.is_some() || self.mpnn == Some(MpnnKind::GatedGcn)
    }

    /// Check the spec; every error names the field it is about.
    pub fn validate(&self) -> Result<()> {
        self.node_features.validate("node_features")?;
        if let Some(edge) = &self.edge_features {
            edge.validate("edge_features")?;
            if self.mpnn.is_none() {
                return Err(field(
                    "edge_features",
                    "only message passing reads edge features: set mpnn or drop them",
                ));
            }
        }
        if self.d_model == 0 {
            return Err(field("d_model", "must be positive"));
        }
        if let Some((start, end)) = self.pe_sign_flip
            && (start >= end || end > self.pe_dim)
        {
            return Err(field(
                "pe_sign_flip",
                format!(
                    "columns {start}..{end} are not a range inside the {} encoding columns",
                    self.pe_dim
                ),
            ));
        }

        let tokens = &self.tokens;
        if tokens.max_hops == 0 {
            if self.token_layers > 0 {
                return Err(field(
                    "token_layers",
                    format!(
                        "{} token layers with max_hops = 0: without walk tokens there is no \
                         per-node stage",
                        self.token_layers
                    ),
                ));
            }
        } else {
            if self.token_layers == 0 {
                return Err(field(
                    "token_layers",
                    "walk tokens (max_hops >= 1) need at least one token layer",
                ));
            }
            if tokens.walks == 0 {
                return Err(field("tokens.walks", "must be at least 1 when max_hops >= 1"));
            }
            if tokens.repeats == 0 {
                return Err(field(
                    "tokens.repeats",
                    "must be at least 1 when max_hops >= 1",
                ));
            }
            if tokens.cap() > 128 {
                return Err(field(
                    "tokens",
                    format!(
                        "1 + walks·max_hops = {} node slots per token, above the limit of 128",
                        tokens.cap()
                    ),
                ));
            }
        }
        match self.local {
            LocalEncoder::Mean | LocalEncoder::Sgc { hops: 1 } => {}
            LocalEncoder::Sgc { hops } => {
                return Err(field(
                    "local",
                    format!("Sgc is built for hops = 1 only, got {hops}"),
                ));
            }
            LocalEncoder::Mpnn { .. } => {
                return Err(field(
                    "local",
                    "the message-passing local encoder is not built; use Mean or Sgc",
                ));
            }
        }
        if self.node_layers == 0 {
            return Err(field("node_layers", "the node stage needs at least one layer"));
        }
        if self.node_sequences == 0 {
            return Err(field("node_sequences", "must be at least 1"));
        }
        if self.task.outputs() == 0 {
            return Err(field("task", "needs at least one class, target or label"));
        }
        if matches!(
            self.task,
            GraphTaskSpec::NodeClass { classes: 1 } | GraphTaskSpec::GraphClass { classes: 1, .. }
        ) {
            return Err(field("task", "a classification needs at least two classes"));
        }
        for (name, ssm) in [("token_ssm", &self.token_ssm), ("node_ssm", &self.node_ssm)] {
            ssm.validate().map_err(|err| field(name, err))?;
            if ssm.d_model != self.d_model {
                return Err(field(
                    name,
                    format!(
                        "its d_model {} differs from the model's {}",
                        ssm.d_model, self.d_model
                    ),
                ));
            }
        }
        // A bidirectional mixer refuses a post-gate norm: it would normalise
        // across both directions at once.
        let token_bidirectional = self.token_layers > 1
            || (self.token_layers == 1 && self.token_tail == TokenTail::Bidirectional);
        if self.token_ssm.post_gate_norm && token_bidirectional {
            return Err(field(
                "token_ssm",
                "post_gate_norm cannot be used with a bidirectional token layer",
            ));
        }
        if self.node_ssm.post_gate_norm && self.direction == ScanDirection::Bidirectional {
            return Err(field(
                "node_ssm",
                "post_gate_norm cannot be used with a bidirectional node stage",
            ));
        }
        if self.row_quantum == 0 {
            return Err(field("row_quantum", "must be positive"));
        }
        if !(0.0..1.0).contains(&self.dropout) {
            return Err(field(
                "dropout",
                format!("must be in [0, 1), got {}", self.dropout),
            ));
        }
        if self.norm_eps.is_nan() || self.norm_eps <= 0.0 {
            return Err(field("norm_eps", "must be positive"));
        }
        Ok(())
    }

    /// [`GraphMambaSpec::validate`], plus what depends on how batches are cut.
    pub fn validate_for(&self, mode: BatchMode) -> Result<()> {
        self.validate()?;
        match mode {
            BatchMode::Graphs => {
                if self.node_sequences > 1 {
                    return Err(field(
                        "node_sequences",
                        "interleaved node sequences cut one large graph; whole-graph batches \
                         already hold one sequence per graph",
                    ));
                }
            }
            BatchMode::NodeSubset => {
                if self.mpnn == Some(MpnnKind::GatedGcn) {
                    return Err(field(
                        "mpnn",
                        "GatedGcn carries per-edge states and runs on whole-graph batches only; \
                         use Gine for node partitions",
                    ));
                }
                if self.edge_features.is_some() {
                    return Err(field(
                        "edge_features",
                        "edge features need per-edge tensors, which node partitions do not have; \
                         train on whole-graph batches or drop them",
                    ));
                }
                if self.task.per_graph() {
                    return Err(field(
                        "task",
                        "a graph-level task needs whole-graph batches, not node partitions",
                    ));
                }
            }
        }
        Ok(())
    }

    /// Number of scalars the model built from this spec holds.
    ///
    /// Worked out from the spec alone, so that a dataset can say what a step
    /// will take before any model exists; `tests/graph_model.rs` holds it
    /// equal to the built model's own count.
    pub fn parameter_count(&self) -> usize {
        let d = self.d_model;
        let linear = |d_in: usize, d_out: usize| d_in * d_out + d_out;
        let edge_width = self.edge_features.as_ref().map(FeatureSpec::width);
        // The input embedding, shared by tokens and nodes.
        let mut count = linear(self.input_width(), d);
        if self.has_token_stage() {
            // The projection of the token statistics.
            count += linear(3, d);
            for layer in 0..self.token_layers {
                let tail = layer + 1 == self.token_layers;
                let bidirectional = !tail || self.token_tail == TokenTail::Bidirectional;
                count += d + mixer_parameters(&self.token_ssm, bidirectional);
            }
        }
        if self.mpnn == Some(MpnnKind::GatedGcn) {
            // The initial edge states: a projection of the edge inputs, or one
            // learned vector.
            count += edge_width.map_or(d, |w| linear(w, d));
        }
        let bidirectional = self.direction == ScanDirection::Bidirectional;
        for layer in 0..self.node_layers {
            count += d + mixer_parameters(&self.node_ssm, bidirectional);
            count += match self.mpnn {
                None => 0,
                Some(MpnnKind::Gine) => {
                    2 * linear(d, d) + edge_width.map_or(0, |w| linear(w, d))
                }
                Some(MpnnKind::GatedGcn) => {
                    5 * linear(d, d) + d + if layer + 1 < self.node_layers { d } else { 0 }
                }
            };
            // The feed-forward block: norm, d -> 2d -> d.
            count += d + linear(d, 2 * d) + linear(2 * d, d);
        }
        // The head: norm, then one projection per node or a two-layer readout
        // per graph.
        count += d;
        count += if self.task.per_graph() {
            linear(d, d) + linear(d, self.task.outputs())
        } else {
            linear(d, self.task.outputs())
        };
        count
    }

    /// The spec as JSON.
    pub fn to_json(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// A spec from JSON, validated.
    pub fn from_json(json: &str) -> Result<Self> {
        let spec: Self = serde_json::from_str(json)?;
        spec.validate()?;
        Ok(spec)
    }
}
