//! The device dataset (GRAPH_MAMBA_PLAN.md §2.2).
//!
//! A [`GraphStore`] is a [`CanonicalGraph`] on the device: every table uploaded
//! once, by value. After that a training step moves nothing from the host —
//! the kernels of [`crate::tensor::ops::graph`] read these tables in place.
//!
//! The host keeps only what shapes and launch geometry need: the graph and
//! edge offsets, the number of labelled items per split, and the permutation
//! back to the caller's node numbering.

use cubecl::prelude::Runtime;

use crate::backend::{Device, FloatElem};
use crate::error::{Error, Result};
use crate::models::graph::data::{CanonicalGraph, CanonicalLabels, FeatureTable, Split};
use crate::tensor::Tensor;
use crate::tensor::ops::graph::{Adjacency, FeatureSource};
use crate::tensor::ops::index::IdTensor;

/// A feature table on the device.
pub enum FeatureStore<R: Runtime, E: FloatElem> {
    /// `[rows, dim]` floats.
    Float(Tensor<R, E>),
    /// `[rows, fields]` ids and the multi-hot column offset of each field.
    Categorical {
        /// `[rows · fields]` ids.
        ids: IdTensor<R>,
        /// `[fields + 1]` prefix sums of the vocabulary sizes.
        field_offset: IdTensor<R>,
        /// The multi-hot width: the sum of the vocabulary sizes.
        vocab: usize,
    },
}

impl<R: Runtime, E: FloatElem> FeatureStore<R, E> {
    fn upload(table: FeatureTable<E>, rows: usize, device: &Device<R>) -> Result<Self> {
        Ok(match table {
            FeatureTable::Float { dim, data } => {
                FeatureStore::Float(Tensor::from_vec(data, vec![rows, dim], device)?)
            }
            FeatureTable::Categorical { field_offset, ids } => {
                let vocab = *field_offset.last().expect("field offsets are never empty") as usize;
                let (fields, len) = (field_offset.len(), ids.len());
                FeatureStore::Categorical {
                    ids: IdTensor::from_vec(ids, vec![len], device)?,
                    field_offset: IdTensor::from_vec(field_offset, vec![fields], device)?,
                    vocab,
                }
            }
        })
    }

    /// The table as the kernels read it.
    pub fn source(&self) -> FeatureSource<'_, R, E> {
        match self {
            FeatureStore::Float(x) => FeatureSource::Float(x),
            FeatureStore::Categorical {
                ids,
                field_offset,
                vocab,
            } => FeatureSource::Categorical {
                ids,
                field_offset,
                vocab: *vocab,
            },
        }
    }

    /// Width of the model input the table expands to.
    pub fn width(&self) -> usize {
        match self {
            FeatureStore::Float(x) => x.dims()[1],
            FeatureStore::Categorical { vocab, .. } => *vocab,
        }
    }

    /// Whether the table holds ids.
    pub fn is_categorical(&self) -> bool {
        matches!(self, FeatureStore::Categorical { .. })
    }
}

/// The targets of a dataset on the device.
pub enum TargetStore<R: Runtime, E: FloatElem> {
    /// One class per node.
    Node {
        /// `[n_nodes]` classes, `IGNORE` where unlabelled.
        y: IdTensor<R>,
        /// `[n_nodes]` split flags.
        split: IdTensor<R>,
    },
    /// One class per graph.
    GraphClass {
        /// `[n_graphs]` classes, `IGNORE` where unlabelled.
        y: IdTensor<R>,
        /// `[n_graphs]` split flags.
        split: IdTensor<R>,
    },
    /// Float targets per graph.
    Graph {
        /// Targets per graph.
        targets: usize,
        /// `[n_graphs, targets]` values, zero where missing.
        y: Tensor<R, E>,
        /// `[n_graphs, targets]`: one where the target is present.
        present: Tensor<R, E>,
        /// `[n_graphs]` split flags.
        split: IdTensor<R>,
    },
    /// No targets.
    None,
}

/// A dataset on the device.
pub struct GraphStore<R: Runtime, E: FloatElem> {
    device: Device<R>,
    adjacency: Adjacency<R>,
    x: FeatureStore<R, E>,
    pe: Option<Tensor<R, E>>,
    edge_x: Option<FeatureStore<R, E>>,
    targets: TargetStore<R, E>,
    graph_ptr: Vec<u32>,
    edge_ptr: Vec<u32>,
    perm: Vec<u32>,
    labelled: [usize; 3],
    class_bound: usize,
    symmetric: bool,
    bytes: usize,
}

impl<R: Runtime, E: FloatElem> GraphStore<R, E> {
    /// Bytes the tables of `canon` take on the device.
    pub fn estimate_bytes(canon: &CanonicalGraph<E>) -> usize {
        let elem = core::mem::size_of::<E>();
        let table = |t: &FeatureTable<E>| match t {
            FeatureTable::Float { data, .. } => data.len() * elem,
            FeatureTable::Categorical { field_offset, ids } => 4 * (ids.len() + field_offset.len()),
        };
        let edges = canon.n_edges().max(1);
        4 * (canon.n_nodes + 1)
            + 4 * edges
            + canon.adj_rev.as_ref().map_or(0, |_| 4 * edges)
            + canon.adj_dst.as_ref().map_or(0, |_| 4 * edges)
            + table(&canon.x)
            + canon.pe.as_ref().map_or(0, |(_, data)| data.len() * elem)
            + canon.edge_x.as_ref().map_or(0, table)
            + match &canon.y {
                CanonicalLabels::Node(ids) | CanonicalLabels::GraphClass(ids) => 4 * ids.len(),
                CanonicalLabels::Graph { values, .. } => 2 * values.len() * elem,
                CanonicalLabels::None => 0,
            }
            + 4 * canon.split.len()
    }

    /// Upload every table of `canon`, by value: the one upload of a dataset.
    pub fn upload(canon: CanonicalGraph<E>, device: &Device<R>) -> Result<Self> {
        crate::backend::ensure_dtype(device, E::DTYPE)?;
        let bytes = Self::estimate_bytes(&canon);
        let (n, g, e) = (canon.n_nodes, canon.n_graphs, canon.n_edges());
        let CanonicalGraph {
            graph_ptr,
            edge_ptr,
            perm,
            adj_off,
            adj_col,
            adj_rev,
            adj_dst,
            x,
            pe,
            edge_x,
            y,
            split,
            labelled,
            symmetric,
            ..
        } = canon;
        let class_bound = match &y {
            CanonicalLabels::Node(ids) | CanonicalLabels::GraphClass(ids) => ids
                .iter()
                .filter(|&&class| class != crate::tensor::ops::IGNORE)
                .max()
                .map_or(0, |&class| class as usize + 1),
            _ => 0,
        };
        let adjacency = Adjacency::upload(adj_off, adj_col, adj_rev, adj_dst, device)?;
        let x = FeatureStore::upload(x, n, device)?;
        let pe = match pe {
            Some((dim, data)) if dim > 0 => Some(Tensor::from_vec(data, vec![n, dim], device)?),
            _ => None,
        };
        let edge_x = match edge_x {
            // An edgeless dataset has no edge rows to gather.
            Some(table) if e > 0 => Some(FeatureStore::upload(table, e, device)?),
            _ => None,
        };
        let targets = match y {
            CanonicalLabels::Node(ids) => TargetStore::Node {
                y: IdTensor::from_vec(ids, vec![n], device)?,
                split: IdTensor::from_vec(split, vec![n], device)?,
            },
            CanonicalLabels::GraphClass(ids) => TargetStore::GraphClass {
                y: IdTensor::from_vec(ids, vec![g], device)?,
                split: IdTensor::from_vec(split, vec![g], device)?,
            },
            CanonicalLabels::Graph { targets, values } => {
                // The device never holds a NaN: a missing target is a zero and
                // a zero in the mask.
                let mut present = Vec::with_capacity(values.len());
                let mut clean = Vec::with_capacity(values.len());
                for v in values {
                    let missing = v.to_scalar().is_nan();
                    present.push(E::from_scalar(if missing { 0.0 } else { 1.0 }));
                    clean.push(if missing { E::from_scalar(0.0) } else { v });
                }
                TargetStore::Graph {
                    targets,
                    y: Tensor::from_vec(clean, vec![g, targets], device)?,
                    present: Tensor::from_vec(present, vec![g, targets], device)?,
                    split: IdTensor::from_vec(split, vec![g], device)?,
                }
            }
            CanonicalLabels::None => TargetStore::None,
        };
        Ok(Self {
            device: device.clone(),
            adjacency,
            x,
            pe,
            edge_x,
            targets,
            graph_ptr,
            edge_ptr,
            perm,
            labelled,
            class_bound,
            symmetric,
            bytes,
        })
    }

    /// The device the tables live on.
    pub fn device(&self) -> &Device<R> {
        &self.device
    }

    /// Nodes over all graphs.
    pub fn n_nodes(&self) -> usize {
        self.adjacency.n_nodes()
    }

    /// Number of graphs.
    pub fn n_graphs(&self) -> usize {
        self.graph_ptr.len() - 1
    }

    /// Number of directed edges.
    pub fn n_edges(&self) -> usize {
        self.adjacency.n_edges()
    }

    /// The edges as CSR.
    pub fn adjacency(&self) -> &Adjacency<R> {
        &self.adjacency
    }

    /// Node features.
    pub fn x(&self) -> &FeatureStore<R, E> {
        &self.x
    }

    /// `[n_nodes, pe_dim]` positional / structural encoding.
    pub fn pe(&self) -> Option<&Tensor<R, E>> {
        self.pe.as_ref()
    }

    /// Width of the encoding, 0 without one.
    pub fn pe_dim(&self) -> usize {
        self.pe.as_ref().map_or(0, |pe| pe.dims()[1])
    }

    /// Edge features, one row per directed edge.
    pub fn edge_x(&self) -> Option<&FeatureStore<R, E>> {
        self.edge_x.as_ref()
    }

    /// The targets.
    pub fn targets(&self) -> &TargetStore<R, E> {
        &self.targets
    }

    /// `[n_graphs + 1]` node offsets (host).
    pub fn graph_ptr(&self) -> &[u32] {
        &self.graph_ptr
    }

    /// `[n_graphs + 1]` edge offsets (host).
    pub fn edge_ptr(&self) -> &[u32] {
        &self.edge_ptr
    }

    /// `[n_nodes]` canonical → original node id (host).
    pub fn perm(&self) -> &[u32] {
        &self.perm
    }

    /// One more than the largest class among the labels; 0 without class
    /// labels. A model with fewer classes cannot be trained or evaluated on
    /// this dataset: its loss would index past its logits.
    pub fn class_bound(&self) -> usize {
        self.class_bound
    }

    /// Labelled items of a split, counted when the dataset was built.
    pub fn labelled(&self, split: Split) -> usize {
        self.labelled[split.index()]
    }

    /// Whether the reverse of every edge was added.
    pub fn symmetric(&self) -> bool {
        self.symmetric
    }

    /// Bytes the tables take on the device.
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// Nodes of graph `g`.
    pub fn graph_len(&self, g: usize) -> usize {
        (self.graph_ptr[g + 1] - self.graph_ptr[g]) as usize
    }

    /// Check that the store was built with the reverse-edge tables.
    pub fn require_reverse_index(&self, what: &str) -> Result<()> {
        if self.adjacency.rev().is_none() {
            return Err(Error::config(format!(
                "{what} needs the reverse-edge index; build the dataset with a spec that has \
                 edge features or GatedGCN"
            )));
        }
        Ok(())
    }
}
