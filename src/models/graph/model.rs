//! The Graph Mamba model (GRAPH_MAMBA_PLAN.md §2.4).
//!
//! ```text
//! rows                    = the batch descriptor                       (1 launch, the batch's)
//! --- per-node stage (max_hops >= 1) ---
//! tokens                  = walk_tokens(store, rows, seed, counter)     (1 launch, constants)
//! tf                      = token_features(tokens, store.x ‖ pe)        (1 launch, constants)
//! tok                     = gelu(embed(tf) + stats_proj(stats))         [rows · L, d]
//! Φ                       = bidirectional blocks over [rows, L, d]
//! y                       = the tail block's last position              [rows, d]
//! --- node stage ---
//! h0                      = embed(node_inputs(store.x ‖ pe, rows))      only with message
//!                                                                       passing or max_hops = 0
//! h                       = y, or h0 when max_hops = 0
//! per layer:  g = unpad(mixer(norm(pad(h))))   over each graph's own length
//!             l = Ψ(h0) in the first layer, Ψ(h) after                   (message passing)
//!             h = h + g + l
//!             h = h + ffn(norm(h))
//! out                     = head(norm(h))                node tasks, per row
//!                           head(pool(norm(h)))          graph tasks, per graph
//! ```
//!
//! Every input of the per-node stage is a constant the device computed from the
//! store, so that stage has no input gradient and a step uploads and reads
//! nothing.

use std::cell::Cell;
use std::ops::ControlFlow;

use cubecl::prelude::Runtime;

use crate::autograd::{Var, no_grad};
use crate::backend::{Device, FloatElem, tally_scope};
use crate::error::{Error, Result};
use crate::models::entity::blocks::{BiBlock, ForwardBlock};
use crate::models::graph::batch::{Epoch, EpochOptions, GraphBatch, GraphDataset};
use crate::models::graph::data::Split;
use crate::models::graph::layers::{GatedGcn, Gine, LocalEncoderModule};
use crate::models::graph::loss::{GraphTask, SafeTargets, safe_targets};
use crate::models::graph::metrics;
use crate::models::graph::spec::{
    BatchMode, FeatureSpec, GraphMambaSpec, GraphPool, GraphTaskSpec, MpnnKind, RegressionLoss,
    TokenTail,
};
use crate::models::vision::ScanDirection;
use crate::nn::dropout::Dropout;
use crate::nn::linear::{Linear, LinearConfig};
use crate::nn::mlp::{Activation, Mlp, MlpConfig};
use crate::nn::module::{Module, ModuleVisitor};
use crate::nn::norm::{RmsNorm, RmsNormConfig};
use crate::nn::param::Param;
use crate::tensor::ops::graph::{
    SignFlip, confusion, edge_inputs, node_inputs, token_features_rows, walk_tokens,
};
use crate::tensor::ops::index::{IdTensor, read_all};
use crate::tensor::ops::movement::RaggedLengths;
use crate::tensor::ops::random::Rng;
use crate::tensor::ops::{IGNORE, elemwise, reduce};
use crate::tensor::Tensor;
use crate::train::{
    AdamW, AdamWConfig, LrSchedule, Optimizer, QueuedStep, StepInfo, Trainer, TrainerConfig,
};

/// One layer of the per-node stage.
enum TokenLayer<R: Runtime, E: FloatElem> {
    /// A bidirectional block over the whole token sequence.
    Bi(BiBlock<R, E>),
    /// The forward-only tail, of which only the last position is finished.
    Forward(ForwardBlock<R, E>),
}

impl<R: Runtime, E: FloatElem> Module<R, E> for TokenLayer<R, E> {
    fn visit(&self, visitor: &mut ModuleVisitor<'_, R, E>) {
        match self {
            TokenLayer::Bi(block) => block.visit(visitor),
            TokenLayer::Forward(block) => block.visit(visitor),
        }
    }
}

/// The mixer of a node layer.
enum NodeMixer<R: Runtime, E: FloatElem> {
    Bi(BiBlock<R, E>),
    Forward(ForwardBlock<R, E>),
}

impl<R: Runtime, E: FloatElem> NodeMixer<R, E> {
    /// `mixer(norm(x))` over a padded batch.
    fn branch_ragged(&self, x: &Var<R, E>, lengths: &RaggedLengths<R>) -> Result<Var<R, E>> {
        match self {
            NodeMixer::Bi(block) => block.branch_ragged(x, lengths),
            NodeMixer::Forward(block) => block.branch_ragged(x, lengths),
        }
    }
}

impl<R: Runtime, E: FloatElem> Module<R, E> for NodeMixer<R, E> {
    fn visit(&self, visitor: &mut ModuleVisitor<'_, R, E>) {
        match self {
            NodeMixer::Bi(block) => block.visit(visitor),
            NodeMixer::Forward(block) => block.visit(visitor),
        }
    }
}

/// The message passing of a node layer.
enum Mpnn<R: Runtime, E: FloatElem> {
    Gine(Gine<R, E>),
    Gated(GatedGcn<R, E>),
}

impl<R: Runtime, E: FloatElem> Module<R, E> for Mpnn<R, E> {
    fn visit(&self, visitor: &mut ModuleVisitor<'_, R, E>) {
        match self {
            Mpnn::Gine(layer) => layer.visit(visitor),
            Mpnn::Gated(layer) => layer.visit(visitor),
        }
    }
}

/// `x + MLP(norm(x))`'s branch: norm, then `d → 2d → d` with GELU.
struct Ffn<R: Runtime, E: FloatElem> {
    norm: RmsNorm<R, E>,
    mlp: Mlp<R, E>,
}

impl<R: Runtime, E: FloatElem> Module<R, E> for Ffn<R, E> {
    fn visit(&self, visitor: &mut ModuleVisitor<'_, R, E>) {
        visitor.child("norm", &self.norm);
        visitor.child("mlp", &self.mlp);
    }
}

/// One layer of the node stage.
struct NodeLayer<R: Runtime, E: FloatElem> {
    mixer: NodeMixer<R, E>,
    mpnn: Option<Mpnn<R, E>>,
    ffn: Ffn<R, E>,
}

impl<R: Runtime, E: FloatElem> Module<R, E> for NodeLayer<R, E> {
    fn visit(&self, visitor: &mut ModuleVisitor<'_, R, E>) {
        visitor.child("mixer", &self.mixer);
        visitor.child_opt("mpnn", &self.mpnn);
        visitor.child("ffn", &self.ffn);
    }
}

/// The initial edge states of GatedGCN.
enum EdgeEmbed<R: Runtime, E: FloatElem> {
    /// A projection of the edge inputs.
    Linear(Linear<R, E>),
    /// One learned vector, for a dataset without edge features.
    Vector(Param<R, E>),
}

impl<R: Runtime, E: FloatElem> Module<R, E> for EdgeEmbed<R, E> {
    fn visit(&self, visitor: &mut ModuleVisitor<'_, R, E>) {
        match self {
            EdgeEmbed::Linear(linear) => linear.visit(visitor),
            EdgeEmbed::Vector(vector) => visitor.param("vector", vector),
        }
    }
}

/// The readout: a norm, then one projection per row or a two-layer readout of
/// each graph's pooled rows.
struct Head<R: Runtime, E: FloatElem> {
    norm: RmsNorm<R, E>,
    hidden: Option<Linear<R, E>>,
    dropout: Option<Dropout>,
    out: Linear<R, E>,
}

impl<R: Runtime, E: FloatElem> Module<R, E> for Head<R, E> {
    fn visit(&self, visitor: &mut ModuleVisitor<'_, R, E>) {
        visitor.child("norm", &self.norm);
        visitor.child_opt("hidden", &self.hidden);
        if let Some(dropout) = &self.dropout {
            visitor.child("dropout", dropout);
        }
        visitor.child("out", &self.out);
    }
}

/// Graph Mamba: random-walk subgraph tokens scanned per node, then the nodes
/// of each graph scanned in degree order, with optional message passing.
pub struct GraphMamba<R: Runtime, E: FloatElem> {
    spec: GraphMambaSpec,
    embed: Linear<R, E>,
    local: Option<LocalEncoderModule<R, E>>,
    token: Vec<TokenLayer<R, E>>,
    edge: Option<EdgeEmbed<R, E>>,
    node: Vec<NodeLayer<R, E>>,
    head: Head<R, E>,
    training: Cell<bool>,
}

impl GraphMambaSpec {
    /// Instantiate the model on a device; see [`GraphMamba::init`].
    pub fn init<R: Runtime, E: FloatElem>(&self, device: &Device<R>) -> Result<GraphMamba<R, E>> {
        GraphMamba::init(self, device)
    }
}

/// The seed of one dropout layer's mask stream: the spec's seed and the
/// layer's index, mixed.
fn dropout_seed(seed: u64, layer: u64) -> u64 {
    let mut z = seed ^ layer.wrapping_mul(0x9E3779B97F4A7C15) ^ 0xD809_D809_D809_D809;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

impl<R: Runtime, E: FloatElem> GraphMamba<R, E> {
    /// Instantiate with the spec's seed.
    pub fn init(spec: &GraphMambaSpec, device: &Device<R>) -> Result<Self> {
        spec.validate()?;
        crate::backend::ensure_dtype(device, E::DTYPE)?;
        let mut rng = Rng::seeded(spec.seed);
        let rng = &mut rng;
        let d = spec.d_model;
        let eps = spec.norm_eps;
        let depth = spec.token_layers + spec.node_layers;
        let norm = |rng: &mut Rng| RmsNormConfig::new(d).with_eps(eps).init(device, rng);

        // The token sequence has one length, so its scan gets a chunk that
        // fits it.
        let token_ssm = crate::ssm::config::SsmConfig {
            chunk_size: spec.token_chunk(),
            ..spec.token_ssm.clone()
        };

        let embed = LinearConfig::new(spec.input_width(), d).init(device, rng);
        let local = spec
            .has_token_stage()
            .then(|| LocalEncoderModule::new(d, device, rng));
        let mut token = Vec::with_capacity(spec.token_layers);
        for layer in 0..spec.token_layers {
            let tail = layer + 1 == spec.token_layers;
            token.push(if tail && spec.token_tail == TokenTail::Forward {
                TokenLayer::Forward(ForwardBlock::new(d, &token_ssm, eps, depth, device, rng)?)
            } else {
                TokenLayer::Bi(BiBlock::new(d, &token_ssm, eps, depth, device, rng)?)
            });
        }

        let edge_width = spec.edge_features.as_ref().map(FeatureSpec::width);
        let edge = (spec.mpnn == Some(MpnnKind::GatedGcn)).then(|| match edge_width {
            Some(width) => EdgeEmbed::Linear(LinearConfig::new(width, d).init(device, rng)),
            None => EdgeEmbed::Vector(Param::new(crate::tensor::ops::random::randn(
                vec![d],
                0.0,
                0.02,
                device,
                rng,
            ))),
        });
        let mut node = Vec::with_capacity(spec.node_layers);
        for layer in 0..spec.node_layers {
            let mixer = match spec.direction {
                ScanDirection::Bidirectional => {
                    NodeMixer::Bi(BiBlock::new(d, &spec.node_ssm, eps, depth, device, rng)?)
                }
                ScanDirection::Forward => NodeMixer::Forward(ForwardBlock::new(
                    d,
                    &spec.node_ssm,
                    eps,
                    depth,
                    device,
                    rng,
                )?),
            };
            let mpnn = spec.mpnn.map(|kind| match kind {
                MpnnKind::Gine => Mpnn::Gine(Gine::new(d, edge_width, device, rng)),
                // The last layer's edge update would be read by no one.
                MpnnKind::GatedGcn => Mpnn::Gated(GatedGcn::new(
                    d,
                    layer + 1 < spec.node_layers,
                    eps,
                    device,
                    rng,
                )),
            });
            let ffn = Ffn {
                norm: norm(rng),
                mlp: MlpConfig::new(d, 2 * d)
                    .with_gated(false)
                    .with_activation(Activation::Gelu)
                    .with_bias(true)
                    .with_dropout(spec.dropout)
                    // Its own stream: layers of one shape would otherwise
                    // drop the same positions at every step.
                    .with_dropout_seed(dropout_seed(spec.seed, layer as u64))
                    .init(device, rng),
            };
            node.push(NodeLayer { mixer, mpnn, ffn });
        }

        let outputs = spec.task.outputs();
        let head = if spec.task.per_graph() {
            Head {
                norm: norm(rng),
                hidden: Some(LinearConfig::new(d, d).init(device, rng)),
                dropout: (spec.dropout > 0.0).then(|| {
                    Dropout::new(spec.dropout).with_seed(dropout_seed(spec.seed, u64::MAX))
                }),
                out: LinearConfig::new(d, outputs).init(device, rng),
            }
        } else {
            Head {
                norm: norm(rng),
                hidden: None,
                dropout: None,
                out: LinearConfig::new(d, outputs).init(device, rng),
            }
        };
        Ok(Self {
            spec: spec.clone(),
            embed,
            local,
            token,
            edge,
            node,
            head,
            training: Cell::new(true),
        })
    }

    /// The spec the model was built from.
    pub fn spec(&self) -> &GraphMambaSpec {
        &self.spec
    }

    /// Whether the model is in training mode (tokens resampled, signs
    /// flipped, dropout on).
    pub fn is_training(&self) -> bool {
        self.training.get()
    }

    /// Refuse a batch the model cannot run on, naming why.
    fn check_batch(&self, batch: &GraphBatch<R, E>) -> Result<()> {
        let spec = &self.spec;
        let store = &batch.store;
        if store.x().width() + store.pe_dim() != spec.input_width() {
            return Err(Error::config(format!(
                "the dataset's node inputs are {} wide (features {} + encoding {}), the model's \
                 {}: the dataset was built for another spec",
                store.x().width() + store.pe_dim(),
                store.x().width(),
                store.pe_dim(),
                spec.input_width()
            )));
        }
        let needs_edges = spec.needs_reverse_index();
        if needs_edges && batch.layout.edges().is_none() {
            return Err(Error::config(
                "edge features and GatedGcn need a whole-graph batch laid out with its edges; \
                 build the dataset from the model's spec and use epoch_graphs"
                    .to_string(),
            ));
        }
        if spec.edge_features.is_some() && store.edge_x().is_none() {
            return Err(Error::config(
                "the model has edge features and the dataset has none".to_string(),
            ));
        }
        if spec.mpnn.is_some() && !store.symmetric() {
            return Err(Error::config(
                "message passing needs a dataset built with symmetrize = true".to_string(),
            ));
        }
        if spec.task.per_graph() && batch.mode != BatchMode::Graphs {
            return Err(Error::config(
                "a graph-level task needs whole-graph batches".to_string(),
            ));
        }
        Ok(())
    }

    /// The per-node stage: each row's token sequence scanned to one vector,
    /// `[rows, d]`.
    fn token_stage(
        &self,
        batch: &GraphBatch<R, E>,
        flip: Option<SignFlip>,
    ) -> Result<Option<Var<R, E>>> {
        let (Some(shape), Some(local)) = (self.spec.walk_shape(), &self.local) else {
            return Ok(None);
        };
        let spec = &self.spec;
        let store = &batch.store;
        let (rows, d, len) = (batch.rows, spec.d_model, shape.len());

        let (tokens, features) = {
            let _scope = tally_scope("graph.data");
            let counter = batch.token_counter(spec.token_sampling, self.training.get());
            let tokens = walk_tokens(
                store.adjacency(),
                &batch.layout,
                shape,
                batch.token_seed,
                counter,
            )?;
            // Features wider than the allocation threshold come in row chunks.
            let per_row = len * spec.input_width() * core::mem::size_of::<E>();
            let features = batch
                .row_chunks(per_row)
                .into_iter()
                .map(|(first, count)| {
                    token_features_rows(
                        &store.x().source(),
                        store.pe(),
                        &tokens,
                        &batch.layout,
                        flip,
                        first,
                        count,
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            (tokens, features)
        };
        let encoded = {
            let _scope = tally_scope("graph.local");
            local.apply_chunks(&self.embed, &features, tokens.stats())?
        };

        let _scope = tally_scope("graph.token");
        let mut sequence = encoded.reshape(vec![rows, len, d])?;
        let mut last = None;
        for (index, layer) in self.token.iter().enumerate() {
            let tail = index + 1 == self.token.len();
            match layer {
                TokenLayer::Forward(block) => last = Some(block.apply_last(&sequence)?),
                TokenLayer::Bi(block) => {
                    sequence = block.apply(&sequence)?;
                    if tail {
                        // The paper-literal tail: the node is the last token.
                        last = Some(sequence.slice(1, len - 1, 1)?.reshape(vec![rows, d])?);
                    }
                }
            }
        }
        Ok(last)
    }

    /// One node layer's mixer branch over the rows `h` `[rows, d]`: each graph
    /// (or each interleaved sequence of a single graph) is one row of the scan.
    fn mixer_branch(
        &self,
        layer: &NodeLayer<R, E>,
        h: &Var<R, E>,
        batch: &GraphBatch<R, E>,
    ) -> Result<Var<R, E>> {
        let (rows, d) = (batch.rows, self.spec.d_model);
        match batch.mode {
            BatchMode::Graphs => layer
                .mixer
                .branch_ragged(
                    &h.pad_ragged(&batch.layout, batch.lengths.max())?,
                    &batch.lengths,
                )?
                .unpad_ragged(&batch.layout),
            BatchMode::NodeSubset => {
                let sequences = batch.lengths.rows();
                if sequences == 1 {
                    // The sequence is the batch: the layouts are identical.
                    return layer
                        .mixer
                        .branch_ragged(&h.reshape(vec![1, rows, d])?, &batch.lengths)?
                        .reshape(vec![rows, d]);
                }
                // Row `r` is position `r / K` of sequence `r mod K`.
                layer
                    .mixer
                    .branch_ragged(
                        &h.reshape(vec![rows / sequences, sequences, d])?
                            .permute(&[1, 0, 2])?,
                        &batch.lengths,
                    )?
                    .permute(&[1, 0, 2])?
                    .reshape(vec![rows, d])
            }
        }
    }

    /// The model's output for a batch: `[rows, classes]` for a node task,
    /// `[graphs, outputs]` for a graph task. Absent rows and graph slots hold
    /// values no one should read; the loss and the metrics mask them.
    pub fn forward(&self, batch: &GraphBatch<R, E>) -> Result<Var<R, E>> {
        self.check_batch(batch)?;
        let spec = &self.spec;
        let store = &batch.store;
        let layout = &batch.layout;
        let adjacency = store.adjacency();
        let d = spec.d_model;
        let flip = match spec.pe_sign_flip {
            Some((start, end)) if self.training.get() => Some(SignFlip {
                start,
                end,
                seed: batch.sign_seed,
            }),
            _ => None,
        };

        let encoded = self.token_stage(batch, flip)?;

        // The embedded input features: the node tokens when there is no
        // per-node stage, and what the first message-passing layer reads.
        let h0 = if encoded.is_none() || spec.mpnn.is_some() {
            let inputs = {
                let _scope = tally_scope("graph.data");
                node_inputs(&store.x().source(), store.pe(), layout, flip)?
            };
            let _scope = tally_scope("graph.embed");
            Some(self.embed.apply(&Var::constant(inputs))?)
        } else {
            None
        };
        let mut h = match (&encoded, &h0) {
            (Some(y), _) => y.clone(),
            (None, Some(h0)) => h0.clone(),
            (None, None) => unreachable!("the node inputs are embedded without a token stage"),
        };

        // Edge inputs and GatedGCN's initial edge states.
        let edge_in = match store.edge_x() {
            Some(table) if spec.edge_features.is_some() => {
                let _scope = tally_scope("graph.data");
                Some(Var::constant(edge_inputs(
                    &table.source(),
                    layout,
                    store.n_edges(),
                )?))
            }
            _ => None,
        };
        let mut edge_state = match (&self.edge, &edge_in) {
            (Some(EdgeEmbed::Linear(linear)), Some(input)) => Some(linear.apply(input)?),
            (Some(EdgeEmbed::Vector(vector)), _) => Some(
                vector
                    .var(&h)
                    .reshape(vec![1, d])?
                    .expand(vec![batch.edges, d])?,
            ),
            (Some(EdgeEmbed::Linear(_)), None) => {
                return Err(Error::config(
                    "the model embeds edge features and the dataset has none".to_string(),
                ));
            }
            (None, _) => None,
        };

        for (index, layer) in self.node.iter().enumerate() {
            let branch = {
                let _scope = tally_scope("graph.node");
                self.mixer_branch(layer, &h, batch)?
            };
            let message = {
                let _scope = tally_scope("graph.mpnn");
                // The first layer's message passing reads the input features
                // (the paper's Ψ(G, X ‖ P)); later layers stack as GPS does.
                let source = match (&h0, index) {
                    (Some(h0), 0) => h0,
                    _ => &h,
                };
                match &layer.mpnn {
                    None => None,
                    Some(Mpnn::Gine(gine)) => {
                        Some(gine.apply(source, edge_in.as_ref(), adjacency, layout)?)
                    }
                    Some(Mpnn::Gated(gated)) => {
                        let state = edge_state
                            .as_ref()
                            .expect("GatedGcn has its initial edge states");
                        let (message, updated) = gated.apply(source, state, adjacency, layout)?;
                        if updated.is_some() {
                            edge_state = updated;
                        }
                        Some(message)
                    }
                }
            };
            let _scope = tally_scope("graph.node");
            h = h.add(&branch)?;
            if let Some(message) = message {
                h = h.add(&message)?;
            }
            h = h.add(&layer.ffn.mlp.apply(&layer.ffn.norm.apply(&h)?)?)?;
        }

        let _scope = tally_scope("graph.head");
        let normed = self.head.norm.apply(&h)?;
        match (spec.task.pool(), &self.head.hidden) {
            (Some(pool), Some(hidden)) => {
                let pooled = normed.segment_pool(layout, pool == GraphPool::Mean)?;
                let mut hidden = hidden.apply(&pooled)?.gelu()?;
                if let Some(dropout) = &self.head.dropout {
                    hidden = dropout.apply(&hidden)?;
                }
                self.head.out.apply(&hidden)
            }
            _ => self.head.out.apply(&normed),
        }
    }

    /// Save the weights with the spec as metadata.
    pub fn save(&self, path: impl AsRef<std::path::Path>, step: u64) -> Result<()> {
        crate::train::Checkpoint::capture(self, step)
            .with_metadata(serde_json::to_value(&self.spec)?)
            .save(path)
    }

    /// Rebuild from a checkpoint's metadata and restore its weights.
    pub fn load(path: impl AsRef<std::path::Path>, device: &Device<R>) -> Result<Self> {
        let ckpt = crate::train::Checkpoint::load(path)?;
        let spec: GraphMambaSpec = serde_json::from_value(ckpt.metadata.clone())?;
        let model = GraphMamba::init(&spec, device)?;
        ckpt.restore(&model, true)?;
        Ok(model)
    }

    /// The epoch an evaluation or a prediction runs over. Never shuffled.
    ///
    /// Whole-graph batches for a graph task or a dataset of several graphs.
    /// One graph is cut into node parts — unless only a row budget was given,
    /// or the model cannot run on node parts at all (GatedGcn, edge features):
    /// such a model was trained on the whole graph, and is evaluated on it.
    fn eval_epoch(
        &self,
        dataset: &GraphDataset<R, E>,
        split: Option<Split>,
        options: &EvalOptions,
    ) -> Result<Epoch<R, E>> {
        let whole_graphs = self.spec.task.per_graph()
            || dataset.num_graphs() > 1
            || (options.parts.is_none()
                && (options.batch_rows.is_some()
                    || self.spec.validate_for(BatchMode::NodeSubset).is_err()));
        if whole_graphs {
            dataset.epoch_graphs_with(
                options.batch_rows,
                0,
                split,
                EpochOptions {
                    shuffle: false,
                    bucket: true,
                },
            )
        } else {
            dataset.epoch_nodes(options.parts, 0, split)
        }
    }

    /// Run `body` in evaluation mode and without a tape, restoring the mode.
    ///
    /// `no_grad` alone would leave dropout on: the mode is switched too, which
    /// also fixes the tokens and turns the sign flips off.
    fn evaluating<T>(&self, body: impl FnOnce() -> Result<T>) -> Result<T> {
        let was_training = self.training.get();
        self.set_training(false);
        let result = {
            let _guard = no_grad();
            body()
        };
        self.set_training(was_training);
        result
    }

    /// The model's output for every node (node tasks) or every graph (graph
    /// tasks) of the dataset, row-major `[items, outputs]`, in the caller's
    /// original order. One device read.
    pub fn predict(&self, dataset: &GraphDataset<R, E>, options: &EvalOptions) -> Result<Vec<f32>> {
        let epoch = self.eval_epoch(dataset, None, options)?;
        let outputs = self.spec.task.outputs();
        let per_graph = self.spec.task.per_graph();
        let store = dataset.store();
        self.evaluating(|| {
            let mut values = Vec::with_capacity(epoch.len());
            let mut row_ids = Vec::with_capacity(epoch.len());
            for index in 0..epoch.len() {
                let batch = epoch.batch(index)?;
                values.push(self.forward(&batch)?.into_tensor());
                row_ids.push(batch.layout.gid().clone());
            }
            // Node parts are chosen on the device, so their row ids come back
            // with the outputs; whole-graph rows are known to the host.
            let read_ids = !per_graph && epoch.mode() == BatchMode::NodeSubset;
            let ids: Vec<&IdTensor<R>> = if read_ids {
                row_ids.iter().collect()
            } else {
                Vec::new()
            };
            let (ids, values) = read_all(&ids, &values.iter().collect::<Vec<_>>())?;

            let items = if per_graph {
                store.n_graphs()
            } else {
                store.n_nodes()
            };
            let mut out = vec![0.0f32; items * outputs];
            let mut place = |item: usize, row: &[f32]| {
                out[item * outputs..(item + 1) * outputs].copy_from_slice(row);
            };
            let perm = store.perm();
            let ptr = store.graph_ptr();
            for (index, batch_values) in values.iter().enumerate() {
                let row = |r: usize| &batch_values[r * outputs..(r + 1) * outputs];
                if per_graph {
                    for (slot, &graph) in epoch.batch_graphs(index).iter().enumerate() {
                        place(graph as usize, row(slot));
                    }
                } else if read_ids {
                    for (r, &node) in ids[index].iter().enumerate() {
                        if node != IGNORE {
                            place(perm[node as usize] as usize, row(r));
                        }
                    }
                } else {
                    let mut r = 0;
                    for &graph in epoch.batch_graphs(index) {
                        for node in ptr[graph as usize]..ptr[graph as usize + 1] {
                            place(perm[node as usize] as usize, row(r));
                            r += 1;
                        }
                    }
                }
            }
            Ok(out)
        })
    }

    /// A metric over a split. The model runs over every node (every part,
    /// every graph) and the split only selects what the metric counts:
    /// filtering the context by split would change what a node sees.
    ///
    /// One device read: the confusion matrices of a countable metric, the
    /// means and counts of an error, or the scores with their labels and masks
    /// of a rank metric.
    pub fn evaluate(
        &self,
        dataset: &GraphDataset<R, E>,
        split: Split,
        metric: Metric,
        options: &EvalOptions,
    ) -> Result<Evaluation> {
        let task = self.spec.task;
        metric.check(&task)?;
        let epoch = self.eval_epoch(dataset, Some(split), options)?;
        self.evaluating(|| {
            let mut ids: Vec<IdTensor<R>> = Vec::new();
            let mut floats: Vec<Tensor<R, E>> = Vec::new();
            let mut scalars: Vec<Tensor<R, f32>> = Vec::new();
            for index in 0..epoch.len() {
                let batch = epoch.batch(index)?;
                let output = self.forward(&batch)?.into_tensor();
                let targets = safe_targets(&task, &batch, split.flag())?;
                match (metric, &targets) {
                    (Metric::Accuracy | Metric::F1Macro, SafeTargets::Class { ids: y, mask }) => {
                        let predicted = reduce::argmax(&output, 1)?;
                        ids.push(confusion(&predicted, y, mask, task.outputs())?);
                    }
                    (Metric::Mae | Metric::Mse, SafeTargets::Float { values, mask }) => {
                        let error = elemwise::sub(&output, values)?;
                        let error = if metric == Metric::Mae {
                            elemwise::abs(&error)
                        } else {
                            elemwise::mul(&error, &error)?
                        };
                        let (mean, count) = crate::tensor::ops::graph::masked_mean(&error, mask)?;
                        scalars.push(elemwise::cast::<R, E, f32>(&mean));
                        scalars.push(count);
                    }
                    (
                        Metric::AveragePrecision | Metric::RocAuc,
                        SafeTargets::Float { values, mask },
                    ) => {
                        floats.extend([output, values.clone(), mask.clone()]);
                    }
                    (
                        Metric::AveragePrecision | Metric::RocAuc,
                        SafeTargets::Class { ids: y, mask },
                    ) => {
                        ids.push(y.clone());
                        floats.extend([output, mask.clone()]);
                    }
                    _ => unreachable!("Metric::check matched the metric to the task"),
                }
            }

            let outputs = task.outputs();
            match metric {
                Metric::Accuracy | Metric::F1Macro => {
                    let (counts, _) = read_all::<R, E>(&ids.iter().collect::<Vec<_>>(), &[])?;
                    let mut total = vec![0u64; outputs * outputs];
                    for batch in &counts {
                        for (sum, &count) in total.iter_mut().zip(batch) {
                            *sum += count as u64;
                        }
                    }
                    let value = if metric == Metric::Accuracy {
                        metrics::accuracy(&total, outputs)
                    } else {
                        metrics::f1_macro(&total, outputs)
                    };
                    Ok(Evaluation {
                        value,
                        count: total.iter().sum::<u64>() as usize,
                    })
                }
                Metric::Mae | Metric::Mse => {
                    let (_, values) =
                        read_all::<R, f32>(&[], &scalars.iter().collect::<Vec<_>>())?;
                    // Per batch: the masked mean, then [1 / max(count, 1), count].
                    let parts: Vec<(f32, f32)> =
                        values.chunks(2).map(|pair| (pair[0][0], pair[1][1])).collect();
                    Ok(Evaluation {
                        value: metrics::weighted_mean(&parts),
                        count: parts.iter().map(|p| p.1 as usize).sum(),
                    })
                }
                Metric::AveragePrecision | Metric::RocAuc => {
                    let (labels, values) = read_all(
                        &ids.iter().collect::<Vec<_>>(),
                        &floats.iter().collect::<Vec<_>>(),
                    )?;
                    let (mut scores, mut truth, mut mask) = (Vec::new(), Vec::new(), Vec::new());
                    let columns = if labels.is_empty() {
                        // Multi-label: one score column per label.
                        for batch in values.chunks(3) {
                            scores.extend_from_slice(&batch[0]);
                            truth.extend_from_slice(&batch[1]);
                            mask.extend_from_slice(&batch[2]);
                        }
                        outputs
                    } else {
                        // Two classes: the score is the margin of class 1.
                        for (batch, classes) in values.chunks(2).zip(&labels) {
                            scores.extend(batch[0].chunks(2).map(|logits| logits[1] - logits[0]));
                            truth.extend(classes.iter().map(|&c| c as f32));
                            mask.extend_from_slice(&batch[1]);
                        }
                        1
                    };
                    let value = if metric == Metric::AveragePrecision {
                        metrics::average_precision(&scores, &truth, &mask, columns)
                    } else {
                        metrics::roc_auc(&scores, &truth, &mask, columns)
                    };
                    Ok(Evaluation {
                        value: value.ok_or_else(|| {
                            Error::config(format!(
                                "the {} split has no label column with both classes, so {} is \
                                 undefined",
                                split.name(),
                                metric.name()
                            ))
                        })?,
                        count: mask.iter().filter(|&&m| m != 0.0).count(),
                    })
                }
            }
        })
    }

    /// Queue every step of `epoch` on the device: no read, and no upload beyond
    /// the epoch's own table. `control` is called after each queued step with
    /// the optimizer's step count; on [`ControlFlow::Break`] the queue is
    /// drained and the call returns, leaving the model after a completed step.
    /// Returns the number of steps queued; their reports are read with
    /// [`GraphTrainer::read_losses`].
    pub fn train_epoch_with(
        &self,
        trainer: &mut GraphTrainer<R, E>,
        epoch: &Epoch<R, E>,
        mut control: impl FnMut(u64) -> ControlFlow<()>,
    ) -> Result<usize> {
        if epoch.split().is_none() {
            return Err(Error::config(
                "an epoch built for prediction has no split to train on".to_string(),
            ));
        }
        let task = GraphTask::new(self).with_loss_scale(trainer.loss_scale);
        let mut queued = 0;
        for index in 0..epoch.len() {
            let batch = epoch.batch(index)?;
            let step = trainer
                .trainer
                .queue_step(&task, core::slice::from_ref(&batch))?;
            let loss = task
                .take_losses()
                .pop()
                .expect("a queued step built one loss");
            trainer.pending.push((step, loss));
            queued += 1;
            if control(trainer.trainer.step_count()).is_break() {
                epoch.store().device().try_synchronize()?;
                break;
            }
        }
        Ok(queued)
    }

    /// [`GraphMamba::train_epoch_with`] without interruption.
    pub fn train_epoch(
        &self,
        trainer: &mut GraphTrainer<R, E>,
        epoch: &Epoch<R, E>,
    ) -> Result<usize> {
        self.train_epoch_with(trainer, epoch, |_| ControlFlow::Continue(()))
    }
}

impl<R: Runtime, E: FloatElem> Module<R, E> for GraphMamba<R, E> {
    fn on_mode_change(&self, training: bool) {
        self.training.set(training);
    }

    fn visit(&self, visitor: &mut ModuleVisitor<'_, R, E>) {
        visitor.child("embed", &self.embed);
        visitor.child_opt("local", &self.local);
        for (index, layer) in self.token.iter().enumerate() {
            visitor.child_at("token", index, layer);
        }
        if let Some(edge) = &self.edge {
            visitor.child("edge", edge);
        }
        for (index, layer) in self.node.iter().enumerate() {
            visitor.child_at("node", index, layer);
        }
        visitor.child("head", &self.head);
    }
}

/// How an evaluation or a prediction is batched.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EvalOptions {
    /// Row budget of whole-graph batches; `None` is automatic.
    pub batch_rows: Option<usize>,
    /// Node parts of a single large graph; `None` is automatic, 1 full batch.
    pub parts: Option<usize>,
}

/// A metric [`GraphMamba::evaluate`] computes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Metric {
    /// Fraction of correct classes.
    Accuracy,
    /// Unweighted mean of the per-class F1 scores.
    F1Macro,
    /// Mean absolute error over the real targets.
    Mae,
    /// Mean squared error over the real targets.
    Mse,
    /// Average precision, averaged over label columns.
    AveragePrecision,
    /// Area under the ROC curve, averaged over label columns.
    RocAuc,
}

impl Metric {
    /// The name used in errors and in the Python API.
    pub const fn name(self) -> &'static str {
        match self {
            Metric::Accuracy => "accuracy",
            Metric::F1Macro => "f1_macro",
            Metric::Mae => "mae",
            Metric::Mse => "mse",
            Metric::AveragePrecision => "ap",
            Metric::RocAuc => "roc_auc",
        }
    }

    /// The metric of a name, as [`Metric::name`] spells it.
    pub fn parse(name: &str) -> Result<Self> {
        [
            Metric::Accuracy,
            Metric::F1Macro,
            Metric::Mae,
            Metric::Mse,
            Metric::AveragePrecision,
            Metric::RocAuc,
        ]
        .into_iter()
        .find(|metric| metric.name() == name)
        .ok_or_else(|| {
            Error::config(format!(
                "metric: unknown metric {name:?}; expected accuracy, f1_macro, mae, mse, ap or \
                 roc_auc"
            ))
        })
    }

    /// Whether the metric is defined for `task`.
    fn check(self, task: &GraphTaskSpec) -> Result<()> {
        let fits = match (self, task) {
            (
                Metric::Accuracy | Metric::F1Macro,
                GraphTaskSpec::NodeClass { .. } | GraphTaskSpec::GraphClass { .. },
            ) => true,
            (Metric::Mae | Metric::Mse, GraphTaskSpec::GraphRegression { .. }) => true,
            (
                Metric::AveragePrecision | Metric::RocAuc,
                GraphTaskSpec::GraphMultiLabel { .. }
                | GraphTaskSpec::NodeClass { classes: 2 }
                | GraphTaskSpec::GraphClass { classes: 2, .. },
            ) => true,
            _ => false,
        };
        if fits {
            Ok(())
        } else {
            Err(Error::config(format!(
                "metric: {} is not defined for the task {task:?} (accuracy and f1_macro are for \
                 classification, mae and mse for regression, ap and roc_auc for multi-label and \
                 two-class tasks)",
                self.name()
            )))
        }
    }

    /// The loss a regression task trains with, as a metric.
    pub const fn of_loss(loss: RegressionLoss) -> Self {
        match loss {
            RegressionLoss::L1 => Metric::Mae,
            RegressionLoss::Mse => Metric::Mse,
        }
    }
}

/// The result of [`GraphMamba::evaluate`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Evaluation {
    /// The metric's value.
    pub value: f32,
    /// Targets the metric counted.
    pub count: usize,
}

/// How a [`GraphTrainer`] optimises.
#[derive(Debug, Clone)]
pub struct GraphTrainConfig {
    /// Base learning rate.
    pub learning_rate: f32,
    /// Decoupled weight decay (matrices only).
    pub weight_decay: f32,
    /// Global gradient-norm clip; 0 disables it.
    pub max_grad_norm: f32,
    /// Learning-rate schedule per optimizer step.
    pub schedule: LrSchedule,
    /// Factor the loss is multiplied by before it is differentiated: keeps a
    /// 16-bit model's gradients above underflow. The clip and AdamW's epsilon
    /// scale with it, and reports are divided by it.
    pub loss_scale: f32,
}

impl Default for GraphTrainConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3,
            weight_decay: 0.0,
            max_grad_norm: 1.0,
            schedule: LrSchedule::Constant,
            loss_scale: 1.0,
        }
    }
}

/// The optimizer side of a training run: AdamW, the schedule, and the steps
/// queued on the device whose reports have not been read yet.
pub struct GraphTrainer<R: Runtime, E: FloatElem> {
    trainer: Trainer<R, E, AdamW<R, E>>,
    loss_scale: f32,
    pending: Vec<(QueuedStep<R, E>, Tensor<R, E>)>,
}

impl<R: Runtime, E: FloatElem> GraphTrainer<R, E> {
    /// Build the optimizer.
    pub fn new(config: &GraphTrainConfig) -> Result<Self> {
        if config.loss_scale.is_nan() || config.loss_scale <= 0.0 {
            return Err(Error::config(format!(
                "loss_scale must be positive, got {}",
                config.loss_scale
            )));
        }
        let scale = config.loss_scale;
        let trainer = Trainer::new(
            TrainerConfig::builder()
                .learning_rate(config.learning_rate)
                .max_grad_norm(config.max_grad_norm * scale)
                .schedule(config.schedule.clone())
                .build()?,
            AdamWConfig {
                learning_rate: config.learning_rate,
                weight_decay: config.weight_decay,
                eps: 1e-8 * scale,
                ..AdamWConfig::default()
            }
            .init::<R, E>(),
        );
        Ok(Self {
            trainer,
            loss_scale: scale,
            pending: Vec::new(),
        })
    }

    /// Optimizer steps taken so far.
    pub fn step_count(&self) -> u64 {
        self.trainer.step_count()
    }

    /// Set the step counter, e.g. after restoring a checkpoint.
    pub fn set_step_count(&mut self, step: u64) {
        self.trainer.set_step_count(step);
    }

    /// Steps queued whose reports have not been read.
    pub fn pending(&self) -> usize {
        self.pending.len()
    }

    /// The trainer underneath.
    pub fn trainer(&self) -> &Trainer<R, E, AdamW<R, E>> {
        &self.trainer
    }

    /// The reports of every queued step, oldest first, under one device read.
    /// Losses are the unscaled ones; gradient norms are divided by the loss
    /// scale.
    pub fn read_losses(&mut self) -> Result<Vec<StepInfo>> {
        let pending = core::mem::take(&mut self.pending);
        if pending.is_empty() {
            return Ok(Vec::new());
        }
        let (steps, losses): (Vec<_>, Vec<_>) = pending.into_iter().unzip();
        let scalars: Vec<&Tensor<R, E>> = steps
            .iter()
            .flat_map(QueuedStep::scalars)
            .chain(losses.iter())
            .collect();
        let (_, values) = read_all(&[], &scalars)?;
        let values: Vec<f32> = values.iter().map(|v| v[0]).collect();
        let (reported, unscaled) = values.split_at(values.len() - losses.len());
        let mut infos = self.trainer.report_steps(&steps, reported);
        for (info, &loss) in infos.iter_mut().zip(unscaled) {
            info.loss = loss;
            info.grad_norm /= self.loss_scale;
        }
        Ok(infos)
    }
}

/// The optimizer of a [`GraphTrainer`], for checkpointing its state.
impl<R: Runtime, E: FloatElem> GraphTrainer<R, E> {
    /// The AdamW optimizer.
    pub fn optimizer(&self) -> &AdamW<R, E> {
        self.trainer.optimizer()
    }

    /// Mutable access to the AdamW optimizer.
    pub fn optimizer_mut(&mut self) -> &mut AdamW<R, E> {
        self.trainer.optimizer_mut()
    }

    /// The learning rate last applied.
    pub fn learning_rate(&self) -> f32 {
        self.trainer.optimizer().learning_rate()
    }
}
