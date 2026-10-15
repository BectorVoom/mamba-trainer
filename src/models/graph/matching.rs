//! Graph matching: ranking each group's member graphs against the group's
//! query vector.
//!
//! A training **group** has one query — a sparse vector of up to
//! [`MAX_QUERY_BITS`] weighted bits over `n_bits` bits — and several member
//! graphs of a [`GraphDataset`]: the answer first, then decoys. The score of a
//! member is
//!
//! ```text
//! score(g, m) = exp(clamp(log_scale, 0, ln 100)) · cos(q_g, z_m)
//! q_g         = QueryEncoder(bag of the query's bits)
//! z_m         = GraphMamba(graph m)            the per-graph output, used as an embedding
//! ```
//!
//! and the loss is the softmax cross entropy of the answer among the group's
//! members, averaged over the real groups of a batch.
//!
//! # How a step runs
//!
//! ```text
//! graphs = union of the batch's groups' members      (deduplicated, first-seen order)
//! z      = l2norm(GraphMamba(graphs))                [slots, d_out]
//! bag    = match_bag(group table, query table)       [groups, n_bits]   1 launch, constant
//! q      = l2norm(QueryEncoder(bag))                 [groups, d_out]
//! logits = scale · q zᵀ                              [groups, slots]
//! legal  = match_layout(group table)                 each group's own member slots
//! loss   = masked_mean(cross_entropy(mask_logits(logits, legal), answers), real groups)
//! ```
//!
//! A graph that two groups of one batch share is encoded once; the product's
//! adjoint sums both groups' contributions to it. There are no in-batch
//! negatives: a group's softmax runs over its own members only.
//!
//! The query table ([`MatchQueries`]) is uploaded once. An epoch
//! ([`MatchEpoch`]) uploads one group table, the graph epoch's slot table and
//! nothing else; a step only launches kernels over those tables and reads
//! nothing.
//!
//! The model is trained in `f32`.

use std::collections::HashMap;
use std::ops::ControlFlow;

use cubecl::prelude::Runtime;

use crate::autograd::{Var, no_grad};
use crate::backend::{Device, FloatElem};
use crate::error::{Error, Result};
use crate::models::graph::batch::{Epoch, GraphBatch, GraphDataset, MAX_BYTES_ENV, max_bytes};
use crate::models::graph::model::{GraphMamba, GraphTrainConfig};
use crate::models::graph::spec::GraphMambaSpec;
use crate::nn::linear::{Linear, LinearConfig};
use crate::nn::module::{Module, ModuleVisitor};
use crate::nn::norm::{RmsNorm, RmsNormConfig};
use crate::nn::param::Param;
use crate::tensor::Tensor;
use crate::tensor::ops::IGNORE;
use crate::tensor::ops::graph_match::{GROUP_STRIDE, match_bag, match_layout};
use crate::tensor::ops::index::{IdTensor, read_all};
use crate::tensor::ops::random::Rng;
use crate::train::{
    AdamW, AdamWConfig, Optimizer, QueuedStep, StepInfo, TrainStep, Trainer, TrainerConfig,
};

/// Most bits one query may name.
pub const MAX_QUERY_BITS: usize = 128;

/// The initial logit scale, `ln(1 / 0.07)`: a temperature of 0.07.
pub const LOG_SCALE_INIT: f32 = 2.659_260_1;

/// The largest logit scale is `exp(LOG_SCALE_MAX) = 100`.
pub const LOG_SCALE_MAX: f32 = 4.605_170_3;

/// The smallest logit scale is `exp(LOG_SCALE_MIN) = 1`.
pub const LOG_SCALE_MIN: f32 = 0.0;

/// Added to a squared norm before the reciprocal square root of the L2
/// normalisation: a zero vector normalises to zero, not to NaN.
pub const NORM_EPS: f32 = 1e-12;

/// Bytes of logits the scoring path keeps on the device before it reads them.
const SCORE_READ_BYTES: usize = 64 << 20;

fn round_up(value: usize, multiple: usize) -> usize {
    value.div_ceil(multiple) * multiple
}

// ---------------------------------------------------------------------------
// Queries and groups
// ---------------------------------------------------------------------------

/// Every query of a run on the device: `[queries, k]` bit ids and weights, with
/// the real entries of each query first and their count. Uploaded once.
pub struct MatchQueries<R: Runtime, E: FloatElem> {
    bits: IdTensor<R>,
    weights: Tensor<R, E>,
    counts: IdTensor<R>,
    queries: usize,
    k: usize,
    n_bits: usize,
}

impl<R: Runtime, E: FloatElem> Clone for MatchQueries<R, E> {
    fn clone(&self) -> Self {
        Self {
            bits: self.bits.clone(),
            weights: self.weights.clone(),
            counts: self.counts.clone(),
            queries: self.queries,
            k: self.k,
            n_bits: self.n_bits,
        }
    }
}

impl<R: Runtime, E: FloatElem> MatchQueries<R, E> {
    /// Validate and upload `[queries, k]` row-major `bits` and `weights`.
    ///
    /// A weight of 0 is padding, whatever its bit; every other entry must name
    /// a bit below `n_bits` with a finite weight. A bit named twice in one
    /// query adds its weights. `k` is at most [`MAX_QUERY_BITS`].
    pub fn new(
        bits: &[u32],
        weights: &[f32],
        k: usize,
        n_bits: usize,
        device: &Device<R>,
    ) -> Result<Self> {
        if k == 0 || k > MAX_QUERY_BITS {
            return Err(Error::config(format!(
                "a query names between 1 and {MAX_QUERY_BITS} bits, got k = {k}"
            )));
        }
        if n_bits == 0 || n_bits > u32::MAX as usize {
            return Err(Error::config(format!(
                "n_bits must be positive, got {n_bits}"
            )));
        }
        if bits.len() != weights.len() || !bits.len().is_multiple_of(k) || bits.is_empty() {
            return Err(Error::config(format!(
                "bits and weights must both be [queries, {k}] with at least one query, got {} \
                 and {} values",
                bits.len(),
                weights.len()
            )));
        }
        let queries = bits.len() / k;
        // Real entries first: the kernel walks only `counts[q]` of them.
        let mut packed_bits = Vec::with_capacity(bits.len());
        let mut packed_weights = Vec::with_capacity(bits.len());
        let mut counts = Vec::with_capacity(queries);
        for q in 0..queries {
            let row = q * k..(q + 1) * k;
            let mut count = 0u32;
            for (&bit, &weight) in bits[row.clone()].iter().zip(&weights[row]) {
                if weight == 0.0 {
                    continue;
                }
                if !weight.is_finite() {
                    return Err(Error::config(format!(
                        "weights: query {q} has the non-finite weight {weight}"
                    )));
                }
                if bit as usize >= n_bits {
                    return Err(Error::config(format!(
                        "bits: query {q} names bit {bit}, outside the {n_bits} bits"
                    )));
                }
                packed_bits.push(bit);
                packed_weights.push(E::from_scalar(weight));
                count += 1;
            }
            let pad = k - count as usize;
            packed_bits.extend(std::iter::repeat_n(0u32, pad));
            packed_weights.extend(std::iter::repeat_n(E::from_scalar(0.0), pad));
            counts.push(count);
        }
        Ok(Self {
            bits: IdTensor::from_vec(packed_bits, vec![queries * k], device)?,
            weights: Tensor::from_vec(packed_weights, vec![queries * k], device)?,
            counts: IdTensor::from_vec(counts, vec![queries], device)?,
            queries,
            k,
            n_bits,
        })
    }

    /// Number of queries.
    pub fn len(&self) -> usize {
        self.queries
    }

    /// Whether there is no query (never true for a built table).
    pub fn is_empty(&self) -> bool {
        self.queries == 0
    }

    /// Entries per query row.
    pub fn k(&self) -> usize {
        self.k
    }

    /// Width of the bit space.
    pub fn n_bits(&self) -> usize {
        self.n_bits
    }

    /// Bytes the table takes on the device.
    pub fn bytes(&self) -> usize {
        self.queries * self.k * (4 + core::mem::size_of::<E>()) + self.queries * 4
    }

    /// The dense `[rows, n_bits]` bag of the queries `index` names at
    /// `index_at + r * stride`. One launch.
    fn bag(
        &self,
        index: &IdTensor<R>,
        index_at: usize,
        stride: usize,
        rows: usize,
    ) -> Result<Tensor<R, E>> {
        match_bag(
            index,
            index_at,
            stride,
            rows,
            &self.bits,
            &self.weights,
            &self.counts,
            self.n_bits,
        )
    }
}

/// The groups of a run, on the host: per group a query id and its member
/// graphs, answer first. Repeated members of a group are dropped (first
/// occurrence kept), so the answer stays first.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatchGroups {
    query: Vec<u32>,
    ptr: Vec<u32>,
    members: Vec<u32>,
}

impl MatchGroups {
    /// Groups from a query id per group and CSR member lists
    /// (`member_ptr` is `[groups + 1]`, starting at 0). Every group needs at
    /// least one member. Ids are checked against a dataset and a query table
    /// when an epoch is built.
    pub fn new(query: &[u32], member_ptr: &[u32], members: &[u32]) -> Result<Self> {
        if query.is_empty() {
            return Err(Error::config(
                "query: at least one group is needed".to_string(),
            ));
        }
        if member_ptr.len() != query.len() + 1 {
            return Err(Error::config(format!(
                "member_ptr must be [groups + 1] = [{}], got {} values",
                query.len() + 1,
                member_ptr.len()
            )));
        }
        if member_ptr[0] != 0 || member_ptr[query.len()] as usize != members.len() {
            return Err(Error::config(format!(
                "member_ptr must start at 0 and end at the {} members, got {} .. {}",
                members.len(),
                member_ptr[0],
                member_ptr[query.len()]
            )));
        }
        let mut ptr = Vec::with_capacity(member_ptr.len());
        let mut kept = Vec::with_capacity(members.len());
        ptr.push(0u32);
        for (g, window) in member_ptr.windows(2).enumerate() {
            if window[1] <= window[0] {
                return Err(Error::config(format!(
                    "member_ptr: group {g} has no member (member_ptr must increase)"
                )));
            }
            let start = kept.len();
            for &m in &members[window[0] as usize..window[1] as usize] {
                if !kept[start..].contains(&m) {
                    kept.push(m);
                }
            }
            ptr.push(kept.len() as u32);
        }
        Ok(Self {
            query: query.to_vec(),
            ptr,
            members: kept,
        })
    }

    /// Number of groups.
    pub fn len(&self) -> usize {
        self.query.len()
    }

    /// Whether there is no group (never true for built groups).
    pub fn is_empty(&self) -> bool {
        self.query.is_empty()
    }

    /// The query id of group `g`.
    pub fn query(&self, g: usize) -> u32 {
        self.query[g]
    }

    /// The distinct members of group `g`, answer first.
    pub fn members(&self, g: usize) -> &[u32] {
        &self.members[self.ptr[g] as usize..self.ptr[g + 1] as usize]
    }

    /// The groups without a negative: fewer than two distinct members.
    pub fn trivial(&self) -> Vec<u32> {
        (0..self.len())
            .filter(|&g| self.members(g).len() < 2)
            .map(|g| g as u32)
            .collect()
    }
}

// ---------------------------------------------------------------------------
// The query encoder and the model
// ---------------------------------------------------------------------------

/// The shape of a [`QueryEncoder`].
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct QueryEncoderConfig {
    /// Width of the bit space.
    pub n_bits: usize,
    /// Width of the bag projection and the hidden layer.
    pub hidden: usize,
    /// Width of the embedding; the graph model's output width.
    pub d_out: usize,
    /// Epsilon of the norm after the bag projection.
    pub norm_eps: f32,
    /// Seed of the initialisation.
    pub seed: u64,
}

impl QueryEncoderConfig {
    /// An encoder of `n_bits` bits into `d_out` through `hidden`.
    pub fn new(n_bits: usize, hidden: usize, d_out: usize) -> Self {
        Self {
            n_bits,
            hidden,
            d_out,
            norm_eps: 1e-5,
            seed: 0,
        }
    }

    /// Set the seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Scalars in the encoder.
    pub fn parameter_count(&self) -> usize {
        let (n, h, d) = (self.n_bits, self.hidden, self.d_out);
        n * h + h + (h * h + h) + (h * d + d)
    }

    fn validate(&self) -> Result<()> {
        if self.n_bits == 0 || self.hidden == 0 || self.d_out == 0 {
            return Err(Error::config(format!(
                "the query encoder needs positive widths, got n_bits = {}, hidden = {}, d_out \
                 = {}",
                self.n_bits, self.hidden, self.d_out
            )));
        }
        Ok(())
    }
}

/// The query encoder: a bias-free projection of the dense bag of weighted bits
/// (an embedding bag, computed as a product so that its adjoint is a product
/// too), `RmsNorm`, then `Linear → GELU → Linear`.
pub struct QueryEncoder<R: Runtime, E: FloatElem> {
    config: QueryEncoderConfig,
    embed: Linear<R, E>,
    norm: RmsNorm<R, E>,
    up: Linear<R, E>,
    down: Linear<R, E>,
}

impl<R: Runtime, E: FloatElem> QueryEncoder<R, E> {
    /// Instantiate with the config's seed.
    pub fn init(config: &QueryEncoderConfig, device: &Device<R>) -> Result<Self> {
        config.validate()?;
        let mut rng = Rng::seeded(config.seed ^ 0x9E37_79B9_7F4A_7C15);
        let rng = &mut rng;
        let h = config.hidden;
        Ok(Self {
            config: config.clone(),
            embed: LinearConfig::new(config.n_bits, h)
                .with_bias(false)
                .init(device, rng),
            norm: RmsNormConfig::new(h)
                .with_eps(config.norm_eps)
                .init(device, rng),
            up: LinearConfig::new(h, h).init(device, rng),
            down: LinearConfig::new(h, config.d_out).init(device, rng),
        })
    }

    /// The config.
    pub fn config(&self) -> &QueryEncoderConfig {
        &self.config
    }

    /// The unnormalised embedding `[rows, d_out]` of a `[rows, n_bits]` bag.
    pub fn apply(&self, bag: &Var<R, E>) -> Result<Var<R, E>> {
        let hidden = self.norm.apply(&self.embed.apply(bag)?)?;
        self.down.apply(&self.up.apply(&hidden)?.gelu()?)
    }
}

impl<R: Runtime, E: FloatElem> Module<R, E> for QueryEncoder<R, E> {
    fn visit(&self, visitor: &mut ModuleVisitor<'_, R, E>) {
        visitor.child("embed", &self.embed);
        visitor.child("norm", &self.norm);
        visitor.child("up", &self.up);
        visitor.child("down", &self.down);
    }
}

/// Everything a [`GraphMatchModel`] is built from; the checkpoint's metadata.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct GraphMatchConfig {
    /// The graph model: any graph-level task, whose output width is the
    /// embedding width. Its loss is not used.
    pub spec: GraphMambaSpec,
    /// The query encoder; `d_out` equals the spec's output width.
    pub query: QueryEncoderConfig,
}

impl GraphMatchConfig {
    /// A config with a query encoder of `n_bits` bits through `hidden`.
    pub fn new(spec: GraphMambaSpec, n_bits: usize, hidden: usize, seed: u64) -> Self {
        let d_out = spec.task.outputs();
        Self {
            spec,
            query: QueryEncoderConfig::new(n_bits, hidden, d_out).with_seed(seed),
        }
    }

    /// Refuse a config the model cannot be built from.
    pub fn validate(&self) -> Result<()> {
        self.spec.validate()?;
        if !self.spec.task.per_graph() {
            return Err(Error::config(format!(
                "graph matching embeds whole graphs: the spec needs a graph-level task, got {:?}",
                self.spec.task
            )));
        }
        if self.query.d_out != self.spec.task.outputs() {
            return Err(Error::config(format!(
                "the query encoder's d_out ({}) must equal the graph model's output width ({})",
                self.query.d_out,
                self.spec.task.outputs()
            )));
        }
        self.query.validate()
    }

    /// Bytes a matching step adds on the device to what the graph dataset's
    /// [`GraphDataset::memory_estimate`] counts, for `groups` groups over
    /// `slots` graph slots: the encoder's weights, gradients and AdamW
    /// moments, the log scale, the bag, the encoder's activations, the graph
    /// embeddings' normalisation and the `[groups, slots]` logits chain. An
    /// estimate, as the graph model's own is.
    pub fn step_bytes(&self, groups: usize, slots: usize) -> usize {
        let e = 4;
        let q = &self.query;
        let params = q.parameter_count() + 1;
        // Weights, two moments and a gradient.
        let parameters = 4 * params * e;
        let bag = groups * q.n_bits * e;
        // Projection, norm, hidden, GELU chain (four buffers), output and its
        // normalisation, each kept for the backward pass.
        let encoder = groups * (8 * q.hidden + 4 * q.d_out) * e;
        let graphs = 4 * slots * q.d_out * e;
        // Cosines, scaled, masked, legal mask, the cross entropy's adjoint.
        let logits = 6 * groups * slots * e;
        parameters + bag + encoder + graphs + logits
    }
}

/// The matcher: a Graph Mamba embedding the member graphs, a query encoder
/// and a learned logit scale.
pub struct GraphMatchModel<R: Runtime, E: FloatElem> {
    config: GraphMatchConfig,
    graph: GraphMamba<R, E>,
    query: QueryEncoder<R, E>,
    log_scale: Param<R, E>,
}

/// `v / sqrt(Σ v² + ε)` per row of `[rows, d]`.
fn l2_normalise<R: Runtime, E: FloatElem>(v: &Var<R, E>) -> Result<Var<R, E>> {
    let inverse = v.mul(v)?.sum_dim(1)?.add_scalar(NORM_EPS).rsqrt();
    v.mul(&inverse)
}

impl<R: Runtime, E: FloatElem> GraphMatchModel<R, E> {
    /// Instantiate: the graph model with the spec's seed, the encoder with its
    /// own, the log scale at [`LOG_SCALE_INIT`].
    pub fn init(config: &GraphMatchConfig, device: &Device<R>) -> Result<Self> {
        config.validate()?;
        if E::DTYPE != crate::backend::DType::F32 {
            return Err(Error::config(
                "graph matching is trained in f32 in this build".to_string(),
            ));
        }
        Ok(Self {
            config: config.clone(),
            graph: GraphMamba::init(&config.spec, device)?,
            query: QueryEncoder::init(&config.query, device)?,
            log_scale: Param::new(Tensor::full(vec![1], LOG_SCALE_INIT, device)),
        })
    }

    /// The config the model was built from.
    pub fn config(&self) -> &GraphMatchConfig {
        &self.config
    }

    /// The graph model.
    pub fn graph(&self) -> &GraphMamba<R, E> {
        &self.graph
    }

    /// The query encoder.
    pub fn query_encoder(&self) -> &QueryEncoder<R, E> {
        &self.query
    }

    /// The log scale parameter (unclamped).
    pub fn log_scale(&self) -> &Param<R, E> {
        &self.log_scale
    }

    /// The logit scale `exp(clamp(log_scale))`. One device read.
    pub fn scale(&self) -> Result<f32> {
        let raw = self.log_scale.value().try_to_f32()?[0];
        Ok(raw.clamp(LOG_SCALE_MIN, LOG_SCALE_MAX).exp())
    }

    /// The L2-normalised embedding `[slots, d_out]` of every graph slot of a
    /// whole-graph batch; empty slots give zero vectors.
    pub fn graph_embeddings(&self, batch: &GraphBatch<R, E>) -> Result<Var<R, E>> {
        l2_normalise(&self.graph.forward(batch)?)
    }

    /// The L2-normalised embedding `[rows, d_out]` of a `[rows, n_bits]` bag.
    pub fn query_embeddings(&self, bag: &Tensor<R, E>) -> Result<Var<R, E>> {
        l2_normalise(&self.query.apply(&Var::constant(bag.clone()))?)
    }

    /// `scale · q zᵀ`, `[groups, slots]`, before masking.
    pub fn logits(&self, batch: &MatchBatch<R, E>) -> Result<Var<R, E>> {
        let z = self.graph_embeddings(&batch.graph)?;
        let q = self.query_embeddings(&batch.bag)?;
        let cosine = q.matmul_nt(&z)?;
        let scale = self
            .log_scale
            .var(&cosine)
            .clamp(LOG_SCALE_MIN, LOG_SCALE_MAX)
            .exp();
        cosine.mul(&scale)
    }

    /// The `[1]` loss of a batch: the cross entropy of each real group's answer
    /// among its own members, averaged over the real groups.
    pub fn loss(&self, batch: &MatchBatch<R, E>) -> Result<Var<R, E>> {
        self.logits(batch)?
            .mask_logits(&batch.legal)?
            .cross_entropy_rows(&batch.targets, 0.0)?
            .masked_mean(&batch.valid)
    }

    /// Run `body` in evaluation mode (fixed tokens, no sign flips, no dropout)
    /// and without a tape, restoring the mode.
    pub fn evaluating<T>(&self, body: impl FnOnce() -> Result<T>) -> Result<T> {
        let was_training = self.graph.is_training();
        self.set_training(false);
        let result = {
            let _guard = no_grad();
            body()
        };
        self.set_training(was_training);
        result
    }

    /// The scores of every group of `epoch`, batch by batch, in the order the
    /// batches list them: per group, `scale · cos` of each distinct member,
    /// answer first. Evaluation mode; reads every few batches.
    pub fn score(&self, epoch: &MatchEpoch<R, E>) -> Result<Vec<Vec<f32>>> {
        self.evaluating(|| {
            let mut out = Vec::new();
            let mut pending: Vec<(usize, Tensor<R, E>)> = Vec::new();
            let mut held = 0usize;
            let flush =
                |pending: &mut Vec<(usize, Tensor<R, E>)>, out: &mut Vec<Vec<f32>>| -> Result<()> {
                    if pending.is_empty() {
                        return Ok(());
                    }
                    let tensors: Vec<&Tensor<R, E>> = pending.iter().map(|(_, t)| t).collect();
                    let (_, values) = read_all(&[], &tensors)?;
                    let slots = epoch.slots();
                    for ((index, _), logits) in pending.iter().zip(&values) {
                        for (row, members) in epoch.group_slots(*index).iter().enumerate() {
                            out.push(
                                members
                                    .iter()
                                    .map(|&s| logits[row * slots + s as usize])
                                    .collect(),
                            );
                        }
                    }
                    pending.clear();
                    Ok(())
                };
            for index in 0..epoch.len() {
                let batch = epoch.batch(index)?;
                let logits = self.logits(&batch)?.into_tensor();
                held += logits.len() * core::mem::size_of::<E>();
                pending.push((index, logits));
                if held >= SCORE_READ_BYTES {
                    flush(&mut pending, &mut out)?;
                    held = 0;
                }
            }
            flush(&mut pending, &mut out)?;
            Ok(out)
        })
    }

    /// The L2-normalised embeddings `[ids, d_out]` (row-major) of the graphs
    /// `graph_ids` (`None`: every graph), in that order, in evaluation mode.
    /// Batches hold at most `batch_graphs` graphs and are read as they finish,
    /// so the device never holds more than one batch's outputs.
    pub fn embed_graphs(
        &self,
        dataset: &GraphDataset<R, E>,
        graph_ids: Option<&[u32]>,
        batch_graphs: usize,
    ) -> Result<Vec<f32>> {
        let n_graphs = dataset.num_graphs();
        let all: Vec<u32>;
        let ids = match graph_ids {
            Some(ids) => ids,
            None => {
                all = (0..n_graphs as u32).collect();
                &all
            }
        };
        if batch_graphs == 0 {
            return Err(Error::config("batch_graphs must be positive".to_string()));
        }
        if let Some(&bad) = ids.iter().find(|&&g| g as usize >= n_graphs) {
            return Err(Error::config(format!(
                "graph_ids: graph {bad} of a dataset with {n_graphs} graphs"
            )));
        }
        let d = self.config.query.d_out;
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let store = dataset.store();
        let size = |g: u32| store.graph_len(g as usize);
        let total: usize = ids.iter().map(|&g| size(g)).sum();
        let largest = ids.iter().map(|&g| size(g)).max().unwrap_or(1);
        let budget = dataset
            .auto_rows()
            .min(round_up(total, self.config.spec.row_quantum))
            .max(largest);
        let mut batches: Vec<Vec<u32>> = Vec::new();
        let mut current: Vec<u32> = Vec::new();
        let mut used = 0usize;
        for &g in ids {
            if !current.is_empty() && (current.len() == batch_graphs || used + size(g) > budget) {
                batches.push(core::mem::take(&mut current));
                used = 0;
            }
            used += size(g);
            current.push(g);
        }
        batches.push(current);
        let lengths: Vec<usize> = batches.iter().map(Vec::len).collect();
        let epoch = dataset.epoch_from_batches(batches, budget, 0, None)?;
        self.evaluating(|| {
            let mut out = Vec::with_capacity(ids.len() * d);
            for (index, &len) in lengths.iter().enumerate() {
                let batch = epoch.batch(index)?;
                let z = self.graph_embeddings(&batch)?.into_tensor();
                let values = z.try_to_f32()?;
                out.extend_from_slice(&values[..len * d]);
            }
            Ok(out)
        })
    }

    /// The L2-normalised embeddings `[queries, d_out]` (row-major) of every
    /// query, `batch_queries` at a time, in evaluation mode.
    pub fn embed_queries(
        &self,
        queries: &MatchQueries<R, E>,
        batch_queries: usize,
    ) -> Result<Vec<f32>> {
        self.check_queries(queries)?;
        if batch_queries == 0 {
            return Err(Error::config("batch_queries must be positive".to_string()));
        }
        let d = self.config.query.d_out;
        let device = queries.counts.device().clone();
        self.evaluating(|| {
            let mut out = Vec::with_capacity(queries.len() * d);
            for first in (0..queries.len()).step_by(batch_queries) {
                let rows = batch_queries.min(queries.len() - first);
                let ids: Vec<u32> = (first as u32..(first + rows) as u32).collect();
                let index = IdTensor::from_vec(ids, vec![rows], &device)?;
                let bag = queries.bag(&index, 0, 1, rows)?;
                let q = self.query_embeddings(&bag)?.into_tensor();
                out.extend_from_slice(&q.try_to_f32()?);
            }
            Ok(out)
        })
    }

    /// Refuse a query table of another bit width.
    fn check_queries(&self, queries: &MatchQueries<R, E>) -> Result<()> {
        if queries.n_bits() != self.config.query.n_bits {
            return Err(Error::config(format!(
                "the queries have {} bits and the encoder {}",
                queries.n_bits(),
                self.config.query.n_bits
            )));
        }
        Ok(())
    }

    /// Save the weights with the config as metadata.
    pub fn save(&self, path: impl AsRef<std::path::Path>, step: u64) -> Result<()> {
        crate::train::Checkpoint::capture(self, step)
            .with_metadata(serde_json::json!({
                "kind": "graph_match",
                "config": serde_json::to_value(&self.config)?,
            }))
            .save(path)
    }

    /// Rebuild from a checkpoint of [`GraphMatchModel::save`]; returns the
    /// model and the step it was saved at.
    pub fn load(path: impl AsRef<std::path::Path>, device: &Device<R>) -> Result<(Self, u64)> {
        let ckpt = crate::train::Checkpoint::load(path)?;
        if ckpt.metadata.get("kind").and_then(|k| k.as_str()) != Some("graph_match") {
            return Err(Error::config(
                "the checkpoint is not a graph matcher's (saved by GraphMatchModel::save)"
                    .to_string(),
            ));
        }
        let config: GraphMatchConfig = serde_json::from_value(ckpt.metadata["config"].clone())?;
        let model = Self::init(&config, device)?;
        ckpt.restore(&model, true)?;
        Ok((model, ckpt.step))
    }
}

impl<R: Runtime, E: FloatElem> Module<R, E> for GraphMatchModel<R, E> {
    fn visit(&self, visitor: &mut ModuleVisitor<'_, R, E>) {
        // `child` switches the graph model's mode (tokens, sign flips,
        // dropout) on a mode change.
        visitor.child("graph", &self.graph);
        visitor.child("query", &self.query);
        visitor.param("log_scale", &self.log_scale);
    }
}

// ---------------------------------------------------------------------------
// Epochs and batches
// ---------------------------------------------------------------------------

/// One pass over caller-chosen batches of groups: the graph epoch of their
/// members and the group table, both on the device.
///
/// The group table holds, per batch, `groups` rows of `(query, first, end)` —
/// a padding row has query [`IGNORE`] — followed by every row's member slots.
pub struct MatchEpoch<R: Runtime, E: FloatElem> {
    graphs: Epoch<R, E>,
    queries: MatchQueries<R, E>,
    table: IdTensor<R>,
    groups: usize,
    batch_groups: Vec<Vec<u32>>,
    group_slots: Vec<Vec<Vec<u32>>>,
}

/// One step's inputs: the graph batch, the bag of the groups' queries and the
/// groups' layout.
pub struct MatchBatch<R: Runtime, E: FloatElem> {
    /// The member graphs, one slot each.
    pub graph: GraphBatch<R, E>,
    /// `[groups, n_bits]` bags of the groups' queries; zero rows for padding.
    pub bag: Tensor<R, E>,
    /// `[groups, slots]`: 1 at each group's member slots (slot 0 for padding).
    pub legal: Tensor<R, E>,
    /// `[groups]` answer slots (0 for padding).
    pub targets: IdTensor<R>,
    /// `[groups]`: 1 for a real group.
    pub valid: Tensor<R, E>,
}

impl<R: Runtime, E: FloatElem> MatchEpoch<R, E> {
    /// Lay out `batches` — each a list of group ids — over `dataset`.
    ///
    /// Each batch's graphs are the distinct members of its groups in
    /// first-seen order. `batch_rows` is the row capacity of every graph batch
    /// (`None`: the largest batch's rows); passing the same value every epoch
    /// keeps the shapes of a run few. `training` refuses groups without a
    /// negative, whose loss would be zero. Two uploads: the graph epoch's slot
    /// table and the group table.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        dataset: &GraphDataset<R, E>,
        queries: &MatchQueries<R, E>,
        groups: &MatchGroups,
        batches: &[Vec<u32>],
        batch_rows: Option<usize>,
        epoch: u64,
        training: bool,
        config: &GraphMatchConfig,
    ) -> Result<Self> {
        if batches.is_empty() || batches.iter().any(Vec::is_empty) {
            return Err(Error::config(
                "batches: at least one batch, and every batch at least one group".to_string(),
            ));
        }
        if queries.n_bits() != config.query.n_bits {
            return Err(Error::config(format!(
                "the queries have {} bits and the encoder {}",
                queries.n_bits(),
                config.query.n_bits
            )));
        }
        let n_graphs = dataset.num_graphs() as u32;
        for (b, batch) in batches.iter().enumerate() {
            for &g in batch {
                if g as usize >= groups.len() {
                    return Err(Error::config(format!(
                        "batches[{b}] names group {g} of {} groups",
                        groups.len()
                    )));
                }
                let q = groups.query(g as usize);
                if q as usize >= queries.len() {
                    return Err(Error::config(format!(
                        "group {g} names query {q} of {} queries",
                        queries.len()
                    )));
                }
                if let Some(&m) = groups.members(g as usize).iter().find(|&&m| m >= n_graphs) {
                    return Err(Error::config(format!(
                        "group {g} names graph {m} of a dataset with {n_graphs} graphs"
                    )));
                }
            }
        }
        if training {
            let trivial: Vec<u32> = batches
                .iter()
                .flatten()
                .copied()
                .filter(|&g| groups.members(g as usize).len() < 2)
                .collect();
            if let Some(first) = trivial.first() {
                return Err(Error::config(format!(
                    "{} group(s) of the batches have no negative (fewer than 2 distinct \
                     members), e.g. group {first}: their loss is zero; leave them out of \
                     training batches (MatchGroups::trivial lists them)",
                    trivial.len()
                )));
            }
        }

        let store = dataset.store();
        let group_cap = round_up(batches.iter().map(Vec::len).max().unwrap_or(1), 8);
        let mut graph_batches = Vec::with_capacity(batches.len());
        let mut group_slots = Vec::with_capacity(batches.len());
        let mut most_rows = 0usize;
        for batch in batches {
            let mut slot_of: HashMap<u32, u32> = HashMap::new();
            let mut graphs: Vec<u32> = Vec::new();
            let mut slots_per_group = Vec::with_capacity(batch.len());
            for &g in batch {
                let slots: Vec<u32> = groups
                    .members(g as usize)
                    .iter()
                    .map(|&m| {
                        *slot_of.entry(m).or_insert_with(|| {
                            graphs.push(m);
                            (graphs.len() - 1) as u32
                        })
                    })
                    .collect();
                slots_per_group.push(slots);
            }
            most_rows = most_rows.max(graphs.iter().map(|&m| store.graph_len(m as usize)).sum());
            graph_batches.push(graphs);
            group_slots.push(slots_per_group);
        }
        let rows = batch_rows.unwrap_or(most_rows);

        // The group rows of every batch, then every row's member slots.
        let head = batches.len() * group_cap * GROUP_STRIDE;
        let members: usize = group_slots.iter().flatten().map(Vec::len).sum();
        if head + members > u32::MAX as usize {
            return Err(Error::config(
                "the epoch's group table does not fit 32-bit positions; use fewer batches"
                    .to_string(),
            ));
        }
        let mut table = Vec::with_capacity(head + members);
        let mut tail = Vec::with_capacity(members);
        for (batch, slots) in batches.iter().zip(&group_slots) {
            for (&g, members) in batch.iter().zip(slots) {
                let first = head + tail.len();
                tail.extend_from_slice(members);
                table.extend([
                    groups.query(g as usize),
                    first as u32,
                    (head + tail.len()) as u32,
                ]);
            }
            for _ in batch.len()..group_cap {
                table.extend([IGNORE, 0, 0]);
            }
        }
        table.extend_from_slice(&tail);

        let graphs = dataset.epoch_from_batches(graph_batches, rows, epoch, None)?;
        let estimate = dataset.memory_estimate(Some(graphs.rows()), None);
        let extra = config.step_bytes(group_cap, graphs.graphs()) + queries.bytes();
        if estimate.total() + extra > max_bytes() {
            return Err(Error::config(format!(
                "a matching step of {} rows and {group_cap} groups needs about {} bytes on the \
                 device (graph step {}, matcher {extra}), above the limit of {} \
                 ({MAX_BYTES_ENV}); use smaller batches or raise the limit",
                graphs.rows(),
                estimate.total() + extra,
                estimate.total(),
                max_bytes()
            )));
        }
        let len = table.len();
        Ok(Self {
            table: IdTensor::from_vec(table, vec![len], store.device())?,
            graphs,
            queries: queries.clone(),
            groups: group_cap,
            batch_groups: batches.to_vec(),
            group_slots,
        })
    }

    /// Number of batches.
    pub fn len(&self) -> usize {
        self.batch_groups.len()
    }

    /// Whether the epoch has no batch (never true for a built epoch).
    pub fn is_empty(&self) -> bool {
        self.batch_groups.is_empty()
    }

    /// Group capacity of every batch: the most groups of a batch, rounded up
    /// to a multiple of 8.
    pub fn groups(&self) -> usize {
        self.groups
    }

    /// Graph slot capacity of every batch.
    pub fn slots(&self) -> usize {
        self.graphs.graphs()
    }

    /// Row capacity of every graph batch.
    pub fn rows(&self) -> usize {
        self.graphs.rows()
    }

    /// The graph epoch underneath.
    pub fn graph_epoch(&self) -> &Epoch<R, E> {
        &self.graphs
    }

    /// The group ids of batch `index`, in row order.
    pub fn batch_groups(&self, index: usize) -> &[u32] {
        &self.batch_groups[index]
    }

    /// The dataset graph ids of batch `index`, in slot order.
    pub fn batch_graphs(&self, index: usize) -> &[u32] {
        self.graphs.batch_graphs(index)
    }

    /// Per group of batch `index`, the slots of its distinct members, answer
    /// first.
    pub fn group_slots(&self, index: usize) -> &[Vec<u32>] {
        &self.group_slots[index]
    }

    /// Batch `index`: three launches (the graph batch, the bag, the layout),
    /// no upload, no read.
    pub fn batch(&self, index: usize) -> Result<MatchBatch<R, E>> {
        let graph = self.graphs.batch(index)?;
        let group_at = index * self.groups * GROUP_STRIDE;
        let bag = self
            .queries
            .bag(&self.table, group_at, GROUP_STRIDE, self.groups)?;
        let (legal, targets, valid) =
            match_layout(&self.table, group_at, self.groups, self.graphs.graphs())?;
        Ok(MatchBatch {
            graph,
            bag,
            legal,
            targets,
            valid,
        })
    }
}

// ---------------------------------------------------------------------------
// Training
// ---------------------------------------------------------------------------

/// The training task over [`MatchBatch`]es: every parameter of the graph
/// model, the query encoder and the log scale.
pub struct MatchTask<'a, R: Runtime, E: FloatElem> {
    model: &'a GraphMatchModel<R, E>,
    params: Vec<Param<R, E>>,
}

impl<'a, R: Runtime, E: FloatElem> MatchTask<'a, R, E> {
    /// A task over `model`'s parameters.
    pub fn new(model: &'a GraphMatchModel<R, E>) -> Self {
        Self {
            model,
            params: model.parameters(),
        }
    }
}

impl<R: Runtime, E: FloatElem> TrainStep<R, E> for MatchTask<'_, R, E> {
    type Batch = MatchBatch<R, E>;

    fn parameters(&self) -> Vec<Param<R, E>> {
        self.params.clone()
    }

    fn loss(&self, batch: &Self::Batch) -> Result<Var<R, E>> {
        self.model.loss(batch)
    }

    fn set_training(&self, training: bool) {
        self.model.set_training(training);
    }
}

/// The optimizer side of a matching run: AdamW with clipping and a schedule,
/// and the steps queued whose reports have not been read.
pub struct GraphMatchTrainer<R: Runtime, E: FloatElem> {
    trainer: Trainer<R, E, AdamW<R, E>>,
    pending: Vec<QueuedStep<R, E>>,
}

impl<R: Runtime, E: FloatElem> GraphMatchTrainer<R, E> {
    /// Build the optimizer. The matcher trains in `f32`: a loss scale other
    /// than 1 is refused.
    pub fn new(config: &GraphTrainConfig) -> Result<Self> {
        if config.loss_scale != 1.0 {
            return Err(Error::config(format!(
                "graph matching trains in f32 and takes no loss scale, got {}",
                config.loss_scale
            )));
        }
        let trainer = Trainer::new(
            TrainerConfig::builder()
                .learning_rate(config.learning_rate)
                .max_grad_norm(config.max_grad_norm)
                .schedule(config.schedule.clone())
                .build()?,
            AdamWConfig {
                learning_rate: config.learning_rate,
                weight_decay: config.weight_decay,
                ..AdamWConfig::default()
            }
            .init::<R, E>(),
        );
        Ok(Self {
            trainer,
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

    /// The learning rate last applied.
    pub fn learning_rate(&self) -> f32 {
        self.trainer.optimizer().learning_rate()
    }

    /// Queue one step per batch of `epoch`: no read, no upload. `control` is
    /// called after each queued step with the step count; on
    /// [`ControlFlow::Break`] the queue is drained and the call returns after
    /// a completed step. Returns the number of steps queued.
    pub fn train_epoch_with(
        &mut self,
        model: &GraphMatchModel<R, E>,
        epoch: &MatchEpoch<R, E>,
        mut control: impl FnMut(u64) -> ControlFlow<()>,
    ) -> Result<usize> {
        let task = MatchTask::new(model);
        let mut queued = 0;
        for index in 0..epoch.len() {
            let batch = epoch.batch(index)?;
            let step = self
                .trainer
                .queue_step(&task, core::slice::from_ref(&batch))?;
            self.pending.push(step);
            queued += 1;
            if control(self.trainer.step_count()).is_break() {
                epoch.graphs.store().device().try_synchronize()?;
                break;
            }
        }
        Ok(queued)
    }

    /// [`GraphMatchTrainer::train_epoch_with`] without interruption.
    pub fn train_epoch(
        &mut self,
        model: &GraphMatchModel<R, E>,
        epoch: &MatchEpoch<R, E>,
    ) -> Result<usize> {
        self.train_epoch_with(model, epoch, |_| ControlFlow::Continue(()))
    }

    /// The reports of every queued step, oldest first, under one device read.
    pub fn read_losses(&mut self) -> Result<Vec<StepInfo>> {
        let pending = core::mem::take(&mut self.pending);
        if pending.is_empty() {
            return Ok(Vec::new());
        }
        self.trainer.read_steps(&pending)
    }
}
