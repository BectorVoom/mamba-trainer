//! The graph model's own layers (GRAPH_MAMBA_PLAN.md §2.4): the local encoder
//! of a token and the two message-passing layers.
//!
//! Message passing runs directly on the store's CSR through the kernels of
//! [`crate::tensor::ops::graph`]; both layers return their **branch** only —
//! the residual sum belongs to the model.

use cubecl::prelude::Runtime;

use crate::autograd::Var;
use crate::backend::{DType, Device, FloatElem};
use crate::error::{Error, Result};
use crate::nn::linear::{Linear, LinearConfig};
use crate::nn::module::{Module, ModuleVisitor};
use crate::nn::norm::{RmsNorm, RmsNormConfig};
use crate::tensor::Tensor;
use crate::tensor::ops::elemwise;
use crate::tensor::ops::graph::{Adjacency, BatchRows};
use crate::tensor::ops::random::Rng;

/// The local encoder of a subgraph token, on the device path.
///
/// A token's input is the weighted sum of its nodes' input features (weights
/// by the `Mean` or `Sgc` rule, written by the token kernel) and three
/// statistics of the token. The encoding is
///
/// ```text
/// tok = gelu( embed(features) + stats_proj(stats) )
/// ```
///
/// with `embed` the model's one input embedding, shared with the node stage.
/// Because that embedding is affine and a token's weights sum to one, embedding
/// the aggregated features equals aggregating the embedded nodes — so the
/// token inputs are constants and nothing here has an input gradient.
pub struct LocalEncoderModule<R: Runtime, E: FloatElem> {
    stats_proj: Linear<R, E>,
}

impl<R: Runtime, E: FloatElem> LocalEncoderModule<R, E> {
    /// Build the encoder's own parameters: the projection of the statistics.
    pub fn new(d_model: usize, device: &Device<R>, rng: &mut Rng) -> Self {
        Self {
            stats_proj: LinearConfig::new(3, d_model).init(device, rng),
        }
    }

    /// Encode tokens: `features` is `[tokens, F + pe_dim]` and `stats` is
    /// `[tokens, 3]`; the result is `[tokens, d]`.
    pub fn apply(
        &self,
        embed: &Linear<R, E>,
        features: &Tensor<R, E>,
        stats: &Tensor<R, f32>,
    ) -> Result<Var<R, E>> {
        self.apply_chunks(embed, core::slice::from_ref(features), stats)
    }

    /// [`LocalEncoderModule::apply`] with the token features in consecutive
    /// row chunks, each embedded on its own and the results joined: how a
    /// batch whose features would be one oversized allocation is encoded. The
    /// value is that of the unchunked form; every chunk stays alive for the
    /// embedding's weight gradient.
    pub fn apply_chunks(
        &self,
        embed: &Linear<R, E>,
        features: &[Tensor<R, E>],
        stats: &Tensor<R, f32>,
    ) -> Result<Var<R, E>> {
        let embedded = features
            .iter()
            .map(|chunk| embed.apply(&Var::constant(chunk.clone())))
            .collect::<Result<Vec<_>>>()?;
        let embedded = match embedded.len() {
            0 => {
                return Err(Error::shape(
                    "the local encoder needs at least one chunk of token features".to_string(),
                ));
            }
            1 => embedded.into_iter().next().expect("one chunk"),
            _ => crate::autograd::ops::cat(&embedded, 0)?,
        };
        let stats = Var::constant(stats_as::<R, E>(stats));
        embedded.add(&self.stats_proj.apply(&stats)?)?.gelu()
    }
}

/// The `f32` token statistics in the model's element type.
fn stats_as<R: Runtime, E: FloatElem>(stats: &Tensor<R, f32>) -> Tensor<R, E> {
    if E::DTYPE == DType::F32 {
        // The same buffer under the same element type: no launch.
        Tensor::from_handle(
            stats.handle.clone(),
            stats.shape().clone(),
            stats.device().clone(),
        )
    } else {
        elemwise::cast::<R, f32, E>(stats)
    }
}

impl<R: Runtime, E: FloatElem> Module<R, E> for LocalEncoderModule<R, E> {
    fn visit(&self, visitor: &mut ModuleVisitor<'_, R, E>) {
        visitor.child("stats_proj", &self.stats_proj);
    }
}

/// GIN with edge features, `ε = 0`:
///
/// ```text
/// a[r]   = u[r] + Σ_{e into r} relu(u[src(e)] + ee[e])      ee = Linear(edge input)
/// branch = Linear(relu(Linear(a)))
/// ```
///
/// The `u[r]` term is GIN's self term, not the layer's residual.
pub struct Gine<R: Runtime, E: FloatElem> {
    edge: Option<Linear<R, E>>,
    lin1: Linear<R, E>,
    lin2: Linear<R, E>,
}

impl<R: Runtime, E: FloatElem> Gine<R, E> {
    /// A layer of width `d_model`; `edge_width` is the width of the edge
    /// inputs, `None` without edge features.
    pub fn new(
        d_model: usize,
        edge_width: Option<usize>,
        device: &Device<R>,
        rng: &mut Rng,
    ) -> Self {
        Self {
            edge: edge_width.map(|w| LinearConfig::new(w, d_model).init(device, rng)),
            lin1: LinearConfig::new(d_model, d_model).init(device, rng),
            lin2: LinearConfig::new(d_model, d_model).init(device, rng),
        }
    }

    /// The branch for node values `u` `[rows, d]`; `edge_in` is the
    /// `[edges, edge_width]` edge inputs of the batch when the layer has edge
    /// features.
    pub fn apply(
        &self,
        u: &Var<R, E>,
        edge_in: Option<&Var<R, E>>,
        adjacency: &Adjacency<R>,
        rows: &BatchRows<R>,
    ) -> Result<Var<R, E>> {
        let ee = match (&self.edge, edge_in) {
            (Some(edge), Some(input)) => Some(edge.apply(input)?),
            (None, None) => None,
            (Some(_), None) => {
                return Err(Error::config(
                    "this Gine layer was built with edge features and got none".to_string(),
                ));
            }
            (None, Some(_)) => {
                return Err(Error::config(
                    "this Gine layer was built without edge features and got some".to_string(),
                ));
            }
        };
        let aggregated = Var::gine_aggregate(u, ee.as_ref(), adjacency, rows)?;
        self.lin2.apply(&self.lin1.apply(&aggregated)?.relu())
    }
}

impl<R: Runtime, E: FloatElem> Module<R, E> for Gine<R, E> {
    fn visit(&self, visitor: &mut ModuleVisitor<'_, R, E>) {
        visitor.child_opt("edge", &self.edge);
        visitor.child("lin1", &self.lin1);
        visitor.child("lin2", &self.lin2);
    }
}

/// Residual gated graph convolution, with `RmsNorm` in place of BatchNorm:
///
/// ```text
/// ê_ij   = C e_ij + D u_i + E u_j                     (edge j → i)
/// z_i    = Σ_{j → i} σ(ê_ij) ⊙ B u_j / (Σ_{j → i} σ(ê_ij) + 1e-6)
/// branch = relu(RmsNorm(A u_i + z_i))
/// e_ij'  = e_ij + relu(RmsNorm(ê_ij))                 unless this is the last layer
/// ```
///
/// Edge states `e` `[edges, d]` are carried from layer to layer; the last node
/// layer does not compute the update, since nothing would read it and its norm
/// would be a parameter without a gradient.
pub struct GatedGcn<R: Runtime, E: FloatElem> {
    a: Linear<R, E>,
    b: Linear<R, E>,
    c: Linear<R, E>,
    d: Linear<R, E>,
    e: Linear<R, E>,
    norm_node: RmsNorm<R, E>,
    norm_edge: Option<RmsNorm<R, E>>,
}

impl<R: Runtime, E: FloatElem> GatedGcn<R, E> {
    /// A layer of width `d_model`; `update_edges` is false for the last layer.
    pub fn new(
        d_model: usize,
        update_edges: bool,
        norm_eps: f32,
        device: &Device<R>,
        rng: &mut Rng,
    ) -> Self {
        let mut linear = || LinearConfig::new(d_model, d_model).init(device, rng);
        let (a, b, c, d, e) = (linear(), linear(), linear(), linear(), linear());
        let mut norm = || RmsNormConfig::new(d_model).with_eps(norm_eps).init(device, rng);
        Self {
            a,
            b,
            c,
            d,
            e,
            norm_node: norm(),
            norm_edge: update_edges.then(norm),
        }
    }

    /// Whether the layer updates the edge states.
    pub fn updates_edges(&self) -> bool {
        self.norm_edge.is_some()
    }

    /// The branch for node values `u` `[rows, d]` and edge states `e`
    /// `[edges, d]`, and the updated edge states when the layer computes them.
    pub fn apply(
        &self,
        u: &Var<R, E>,
        e: &Var<R, E>,
        adjacency: &Adjacency<R>,
        rows: &BatchRows<R>,
    ) -> Result<(Var<R, E>, Option<Var<R, E>>)> {
        let ehat = Var::gated_edge(
            &self.c.apply(e)?,
            &self.d.apply(u)?,
            &self.e.apply(u)?,
            adjacency,
            rows,
        )?;
        let z = Var::gated_node(&ehat, &self.b.apply(u)?, adjacency, rows)?;
        let branch = self.norm_node.apply(&self.a.apply(u)?.add(&z)?)?.relu();
        let updated = match &self.norm_edge {
            Some(norm) => Some(e.add(&norm.apply(&ehat)?.relu())?),
            None => None,
        };
        Ok((branch, updated))
    }
}

impl<R: Runtime, E: FloatElem> Module<R, E> for GatedGcn<R, E> {
    fn visit(&self, visitor: &mut ModuleVisitor<'_, R, E>) {
        visitor.child("a", &self.a);
        visitor.child("b", &self.b);
        visitor.child("c", &self.c);
        visitor.child("d", &self.d);
        visitor.child("e", &self.e);
        visitor.child("norm_node", &self.norm_node);
        visitor.child_opt("norm_edge", &self.norm_edge);
    }
}
