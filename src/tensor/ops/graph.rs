//! Graph data-path kernels (GRAPH_MAMBA_PLAN.md §2.2).
//!
//! A graph dataset is uploaded once. Everything a training step then needs —
//! which rows a batch holds, the random-walk tokens of every row, their
//! features, the padded per-graph layout, pooling, the targets — is computed by
//! the kernels here. Every one is a **gather**: one unit owns an output element
//! and loops over its sources. None scatters, none uses atomics, none needs a
//! transposed index.
//!
//! # Safety by construction
//!
//! The kernels launch unchecked, so every index they form has to be proved on
//! the host. Rather than re-proving it per launch, the proof is carried by the
//! argument types, none of which can be built from outside with unchecked
//! contents:
//!
//! * [`Adjacency`] validates the CSR when it is uploaded.
//! * [`EpochTable`] is built from the dataset's own graph offsets, so every
//!   slot it describes lies inside the dataset; [`SlotTable`] is one batch of it.
//! * [`BatchRows`] is only produced by [`batch_rows_graphs`] /
//!   [`batch_rows_subset`], whose ids are `< n_nodes` or [`IGNORE`].
//! * [`Tokens`] is only produced by [`walk_tokens`], whose node ids come from
//!   the adjacency.
//!
//! Each launcher then only has to check that its operands belong together
//! (row counts and table sizes), which its `SAFETY` comment names.
//!
//! # A kernel trap on the CPU runtime
//!
//! cubecl-cpu 0.10 cannot compile a loop whose **carried variable starts as a
//! plain copy of a kernel scalar argument**:
//!
//! ```text
//! let mut hi = graphs;            // `graphs` is a scalar argument
//! while lo < hi { … hi = mid; … } // -> the launch is dropped
//! ```
//!
//! The argument reaches the loop header as a block argument, and the MLIR
//! lowering loads it in whichever block it visited last ("operation with block
//! successors must terminate its parent block" / "operand does not dominate
//! this use"). The failure is on the device thread: the launch is dropped, the
//! output keeps what it held, and a read returns that without an error. A
//! literal, a loaded value or anything computed (`graphs + offset`) as the
//! first value is fine, and so is an argument that is only *read* in the loop.
//! [`slot_of`] is written around it.
//!
//! A build does not verify a kernel: launch every one in a test, on every
//! backend, and compare its output.
//!
//! # Launch shape
//!
//! All kernels launch through [`crate::backend::launch_1d_spans`]: one lane per
//! unit on a GPU, one contiguous span of lanes per worker thread on the CPU
//! runtime. Float kernels put the vector index innermost, so writes are
//! unit-stride and a lane reads its source row contiguously; they accumulate in
//! `f32` registers whatever the element type and cast once on store. Integer
//! kernels are scalar.

use cubecl::prelude::*;

use crate::backend::{Device, FloatElem, launch_1d_spans};
use crate::error::{Error, Result};
use crate::tensor::base::Tensor;
use crate::tensor::ops::entity_model::{IGNORE, line_dividing};
use crate::tensor::ops::index::IdTensor;
use crate::tensor::ops::random::hash_u32;
use crate::tensor::shape::Shape;

// ---------------------------------------------------------------------------
// The validated argument types
// ---------------------------------------------------------------------------

/// A graph's edges on the device as compressed sparse rows: row `u` lists,
/// ascending, the sources of the edges into `u`.
pub struct Adjacency<R: Runtime> {
    off: IdTensor<R>,
    col: IdTensor<R>,
    rev: Option<IdTensor<R>>,
    dst: Option<IdTensor<R>>,
    n_nodes: usize,
    n_edges: usize,
    symmetric: bool,
}

impl<R: Runtime> Clone for Adjacency<R> {
    /// Cheap: the tensors are aliased, not copied.
    fn clone(&self) -> Self {
        Self {
            off: self.off.clone(),
            col: self.col.clone(),
            rev: self.rev.clone(),
            dst: self.dst.clone(),
            n_nodes: self.n_nodes,
            n_edges: self.n_edges,
            symmetric: self.symmetric,
        }
    }
}

impl<R: Runtime> Adjacency<R> {
    /// Validate a CSR and upload it by value.
    ///
    /// `off` is `[n + 1]`, non-decreasing from 0 to `col.len()`; every row of
    /// `col` is strictly ascending and `< n`. `rev`, when given, maps each edge
    /// `s → d` to the edge `d → s`, and `dst` names each edge's row.
    pub fn upload(
        off: Vec<u32>,
        mut col: Vec<u32>,
        rev: Option<Vec<u32>>,
        dst: Option<Vec<u32>>,
        device: &Device<R>,
    ) -> Result<Self> {
        if off.len() < 2 || off[0] != 0 {
            return Err(Error::shape(
                "adjacency offsets must be [n + 1] with n >= 1 and start at 0".to_string(),
            ));
        }
        let n = off.len() - 1;
        let e = col.len();
        if off[n] as usize != e || off.windows(2).any(|w| w[1] < w[0]) {
            return Err(Error::shape(format!(
                "adjacency offsets must be non-decreasing and end at the edge count {e}"
            )));
        }
        if n >= IGNORE as usize || e >= IGNORE as usize {
            return Err(Error::shape(
                "adjacency does not fit u32 ids".to_string(),
            ));
        }
        for u in 0..n {
            let row = &col[off[u] as usize..off[u + 1] as usize];
            if row.iter().any(|&c| c as usize >= n) || row.windows(2).any(|w| w[1] <= w[0]) {
                return Err(Error::shape(format!(
                    "adjacency row {u} must be strictly ascending node ids below {n}"
                )));
            }
        }
        if let Some(dst) = &dst {
            let ok = dst.len() == e
                && (0..n).all(|u| {
                    dst[off[u] as usize..off[u + 1] as usize]
                        .iter()
                        .all(|&d| d as usize == u)
                });
            if !ok {
                return Err(Error::shape(
                    "adjacency destinations must name each edge's row".to_string(),
                ));
            }
        }
        if let Some(rev) = &rev {
            // Edge `i` runs `col[i] → row(i)`; its reverse is the edge in row
            // `col[i]` whose source is `row(i)`. An involution alone would let
            // through an index that pairs unrelated edges, which the adjoints
            // of the message-passing kernels would follow out of their slot.
            let mut ok = rev.len() == e;
            for u in 0..n {
                if !ok {
                    break;
                }
                for i in off[u] as usize..off[u + 1] as usize {
                    let (r, s) = (rev[i] as usize, col[i] as usize);
                    ok = r < e
                        && rev[r] as usize == i
                        && col[r] as usize == u
                        && (off[s] as usize..off[s + 1] as usize).contains(&r);
                    if !ok {
                        break;
                    }
                }
            }
            if !ok {
                return Err(Error::shape(
                    "the reverse-edge index must map every edge s → d to the edge d → s"
                        .to_string(),
                ));
            }
        }
        // A validated reverse index proves the graph symmetric; without one,
        // look every edge's reverse up in its (sorted) row.
        let symmetric = rev.is_some()
            || (0..n).all(|u| {
                col[off[u] as usize..off[u + 1] as usize].iter().all(|&s| {
                    col[off[s as usize] as usize..off[s as usize + 1] as usize]
                        .binary_search(&(u as u32))
                        .is_ok()
                })
            });
        // A buffer is never empty: an edgeless graph keeps one entry no row reaches.
        let pad = |mut v: Vec<u32>| {
            if v.is_empty() {
                v.push(IGNORE);
            }
            v
        };
        if col.is_empty() {
            col.push(IGNORE);
        }
        let len = col.len();
        Ok(Self {
            off: IdTensor::from_vec(off, vec![n + 1], device)?,
            col: IdTensor::from_vec(col, vec![len], device)?,
            rev: rev
                .map(|v| IdTensor::from_vec(pad(v), vec![len], device))
                .transpose()?,
            dst: dst
                .map(|v| IdTensor::from_vec(pad(v), vec![len], device))
                .transpose()?,
            n_nodes: n,
            n_edges: e,
            symmetric,
        })
    }

    /// Whether every edge `s → d` has its reverse `d → s`: what the adjoints of
    /// message passing rely on, since they read the graph as its own
    /// transpose.
    pub fn symmetric(&self) -> bool {
        self.symmetric
    }

    /// Number of nodes.
    pub fn n_nodes(&self) -> usize {
        self.n_nodes
    }

    /// Number of directed edges.
    pub fn n_edges(&self) -> usize {
        self.n_edges
    }

    /// `[n + 1]` row offsets.
    pub fn off(&self) -> &IdTensor<R> {
        &self.off
    }

    /// `[max(edges, 1)]` sources.
    pub fn col(&self) -> &IdTensor<R> {
        &self.col
    }

    /// Position of each edge's reverse, when the store built it.
    pub fn rev(&self) -> Option<&IdTensor<R>> {
        self.rev.as_ref()
    }

    /// Row of each edge, when the store built it.
    pub fn dst(&self) -> Option<&IdTensor<R>> {
        self.dst.as_ref()
    }

    /// Bytes held on the device.
    pub fn bytes(&self) -> usize {
        4 * (self.off.len()
            + self.col.len()
            + self.rev.as_ref().map_or(0, IdTensor::len)
            + self.dst.as_ref().map_or(0, IdTensor::len))
    }
}

/// Every batch of one epoch of whole-graph batches, as one device table.
///
/// Batch `i` occupies `stride = 5·B + 2` ids at `i · stride`, `B` being the
/// epoch's graph capacity:
///
/// | ids | content |
/// |---|---|
/// | `B` | dataset graph id of each slot, [`IGNORE`] where the slot is empty |
/// | `B` | dataset node id of the slot's first node |
/// | `B + 1` | row offset of each slot in the batch |
/// | `B` | dataset edge id of the slot's first edge |
/// | `B + 1` | edge-row offset of each slot in the batch |
///
/// Empty slots come last and have length zero.
pub struct EpochTable<R: Runtime> {
    table: IdTensor<R>,
    graphs: usize,
    rows_used: Vec<usize>,
    edges_used: Vec<usize>,
    n_graphs: usize,
    n_nodes: usize,
    n_edges: usize,
}

impl<R: Runtime> EpochTable<R> {
    /// Ids one batch occupies at a graph capacity of `graphs`.
    pub const fn stride(graphs: usize) -> usize {
        5 * graphs + 2
    }

    /// Describe `batches` — each a list of dataset graph ids — from the
    /// dataset's `[G + 1]` node and edge offsets, and upload the table once.
    pub fn build(
        batches: &[Vec<u32>],
        graph_ptr: &[u32],
        edge_ptr: &[u32],
        graphs: usize,
        device: &Device<R>,
    ) -> Result<Self> {
        if graph_ptr.len() != edge_ptr.len() || graph_ptr.len() < 2 {
            return Err(Error::shape(
                "an epoch table needs matching [G + 1] node and edge offsets".to_string(),
            ));
        }
        for (name, ptr) in [("node", graph_ptr), ("edge", edge_ptr)] {
            if ptr[0] != 0 || ptr.windows(2).any(|w| w[1] < w[0]) {
                return Err(Error::shape(format!(
                    "an epoch table's {name} offsets must start at 0 and never decrease"
                )));
            }
        }
        if batches.is_empty() || graphs == 0 {
            return Err(Error::config(
                "an epoch needs at least one batch and one graph slot".to_string(),
            ));
        }
        let n_graphs = graph_ptr.len() - 1;
        let stride = Self::stride(graphs);
        let mut table = Vec::with_capacity(batches.len() * stride);
        let mut rows_used = Vec::with_capacity(batches.len());
        let mut edges_used = Vec::with_capacity(batches.len());
        for (index, batch) in batches.iter().enumerate() {
            if batch.len() > graphs {
                return Err(Error::config(format!(
                    "batch {index} holds {} graphs, above the capacity {graphs}",
                    batch.len()
                )));
            }
            if let Some(&bad) = batch.iter().find(|&&g| g as usize >= n_graphs) {
                return Err(Error::config(format!(
                    "batch {index} names graph {bad} of {n_graphs}"
                )));
            }
            let empty = graphs - batch.len();
            let pad = |table: &mut Vec<u32>, value: u32| {
                table.extend(std::iter::repeat_n(value, empty));
            };
            table.extend_from_slice(batch);
            pad(&mut table, IGNORE);
            table.extend(batch.iter().map(|&g| graph_ptr[g as usize]));
            pad(&mut table, 0);
            let mut rows = 0u32;
            table.push(0);
            for &g in batch {
                rows += graph_ptr[g as usize + 1] - graph_ptr[g as usize];
                table.push(rows);
            }
            pad(&mut table, rows);
            table.extend(batch.iter().map(|&g| edge_ptr[g as usize]));
            pad(&mut table, 0);
            let mut edges = 0u32;
            table.push(0);
            for &g in batch {
                edges += edge_ptr[g as usize + 1] - edge_ptr[g as usize];
                table.push(edges);
            }
            pad(&mut table, edges);
            rows_used.push(rows as usize);
            edges_used.push(edges as usize);
        }
        let len = table.len();
        Ok(Self {
            // The one upload of an epoch.
            table: IdTensor::from_vec(table, vec![len], device)?,
            graphs,
            rows_used,
            edges_used,
            n_graphs,
            n_nodes: graph_ptr[n_graphs] as usize,
            n_edges: edge_ptr[n_graphs] as usize,
        })
    }

    /// Number of batches.
    pub fn len(&self) -> usize {
        self.rows_used.len()
    }

    /// Whether the epoch has no batch.
    pub fn is_empty(&self) -> bool {
        self.rows_used.is_empty()
    }

    /// Graph capacity of every batch.
    pub fn graphs(&self) -> usize {
        self.graphs
    }

    /// Rows batch `index` really holds.
    pub fn rows_used(&self, index: usize) -> usize {
        self.rows_used[index]
    }

    /// Edges batch `index` really holds.
    pub fn edges_used(&self, index: usize) -> usize {
        self.edges_used[index]
    }

    /// The slots of batch `index`.
    pub fn slots(&self, index: usize) -> Result<SlotTable<R>> {
        if index >= self.len() {
            return Err(Error::shape(format!(
                "batch {index} of an epoch with {} batches",
                self.len()
            )));
        }
        Ok(SlotTable {
            table: self.table.clone(),
            base: index * Self::stride(self.graphs),
            graphs: self.graphs,
            rows_used: self.rows_used[index],
            edges_used: self.edges_used[index],
            n_graphs: self.n_graphs,
            n_nodes: self.n_nodes,
            n_edges: self.n_edges,
        })
    }
}

/// One batch's slots inside an [`EpochTable`].
pub struct SlotTable<R: Runtime> {
    table: IdTensor<R>,
    base: usize,
    graphs: usize,
    rows_used: usize,
    edges_used: usize,
    n_graphs: usize,
    n_nodes: usize,
    n_edges: usize,
}

impl<R: Runtime> Clone for SlotTable<R> {
    fn clone(&self) -> Self {
        Self {
            table: self.table.clone(),
            base: self.base,
            graphs: self.graphs,
            rows_used: self.rows_used,
            edges_used: self.edges_used,
            n_graphs: self.n_graphs,
            n_nodes: self.n_nodes,
            n_edges: self.n_edges,
        }
    }
}

impl<R: Runtime> SlotTable<R> {
    /// Graph capacity `B`.
    pub fn graphs(&self) -> usize {
        self.graphs
    }

    /// Rows the batch really holds.
    pub fn rows_used(&self) -> usize {
        self.rows_used
    }

    /// Edges the batch really holds.
    pub fn edges_used(&self) -> usize {
        self.edges_used
    }

    fn graph_id_at(&self) -> usize {
        self.base
    }

    fn graph_start_at(&self) -> usize {
        self.base + self.graphs
    }

    fn node_off_at(&self) -> usize {
        self.base + 2 * self.graphs
    }

    fn edge_start_at(&self) -> usize {
        self.base + 3 * self.graphs + 1
    }

    fn edge_off_at(&self) -> usize {
        self.base + 4 * self.graphs + 1
    }
}

/// The edges of a whole-graph batch in row space: for each edge row its dataset
/// edge id and the batch rows of its two ends, [`IGNORE`] where absent.
pub struct BatchEdges<R: Runtime> {
    eid: IdTensor<R>,
    src: IdTensor<R>,
    dst: IdTensor<R>,
    edges: usize,
}

impl<R: Runtime> Clone for BatchEdges<R> {
    fn clone(&self) -> Self {
        Self {
            eid: self.eid.clone(),
            src: self.src.clone(),
            dst: self.dst.clone(),
            edges: self.edges,
        }
    }
}

impl<R: Runtime> BatchEdges<R> {
    /// Edge capacity `Eb`.
    pub fn edges(&self) -> usize {
        self.edges
    }

    /// `[Eb]` dataset edge ids.
    pub fn eid(&self) -> &IdTensor<R> {
        &self.eid
    }

    /// `[Eb]` batch row of each edge's source.
    pub fn src(&self) -> &IdTensor<R> {
        &self.src
    }

    /// `[Eb]` batch row of each edge's destination.
    pub fn dst(&self) -> &IdTensor<R> {
        &self.dst
    }
}

/// The rows of a batch: which dataset node each row is, and which group — a
/// graph slot, or one of the interleaved sequences of a single graph — it
/// belongs to.
pub struct BatchRows<R: Runtime> {
    gid: IdTensor<R>,
    row_graph: IdTensor<R>,
    row_of: Option<IdTensor<R>>,
    lengths: IdTensor<R>,
    rows: usize,
    groups: usize,
    n_nodes: usize,
    slots: Option<SlotTable<R>>,
    edges: Option<BatchEdges<R>>,
}

impl<R: Runtime> Clone for BatchRows<R> {
    /// Cheap: the tensors are aliased, not copied.
    fn clone(&self) -> Self {
        Self {
            gid: self.gid.clone(),
            row_graph: self.row_graph.clone(),
            row_of: self.row_of.clone(),
            lengths: self.lengths.clone(),
            rows: self.rows,
            groups: self.groups,
            n_nodes: self.n_nodes,
            slots: self.slots.clone(),
            edges: self.edges.clone(),
        }
    }
}

impl<R: Runtime> BatchRows<R> {
    /// Row capacity `Nb`.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of length entries: the graph capacity of a whole-graph batch, or
    /// the number of interleaved sequences of a node-subset batch.
    pub fn groups(&self) -> usize {
        self.groups
    }

    /// Nodes of the dataset the rows index.
    pub fn n_nodes(&self) -> usize {
        self.n_nodes
    }

    /// `[Nb]` dataset node id of each row, [`IGNORE`] where absent.
    pub fn gid(&self) -> &IdTensor<R> {
        &self.gid
    }

    /// `[Nb]` batch slot of each row, [`IGNORE`] where absent.
    pub fn row_graph(&self) -> &IdTensor<R> {
        &self.row_graph
    }

    /// Node-subset batches: `[n_nodes]` node id → row, [`IGNORE`] outside.
    pub fn row_of(&self) -> Option<&IdTensor<R>> {
        self.row_of.as_ref()
    }

    /// `[groups]` true lengths.
    pub fn lengths(&self) -> &IdTensor<R> {
        &self.lengths
    }

    /// Whole-graph batches: the slot table the rows were laid out from.
    pub fn slots(&self) -> Option<&SlotTable<R>> {
        self.slots.as_ref()
    }

    /// Whole-graph batches laid out with their edges.
    pub fn edges(&self) -> Option<&BatchEdges<R>> {
        self.edges.as_ref()
    }

    /// Whether every edge row holds an edge: `false` when the edge capacity is
    /// above the batch's edges, so that trailing edge rows are absent.
    pub fn edges_full(&self) -> bool {
        match (&self.slots, &self.edges) {
            (Some(slots), Some(edges)) => slots.edges_used == edges.edges,
            _ => true,
        }
    }
}

// ---------------------------------------------------------------------------
// G1. Batch rows
// ---------------------------------------------------------------------------

/// The slot that holds position `pos` of a batch: the last whose start is not
/// past it, by descending the bits of the slot index over the `[graphs + 1]`
/// offsets at `off_at`; `graphs` when `pos` is at or past the last offset.
///
/// `top` is the largest power of two not above `graphs` and `bits` the number
/// of bits down to one. Offsets are non-decreasing, so an empty slot (same
/// start as its successor) is never the answer. A fixed number of steps, and
/// the only carried variable starts from a literal: see the module's kernel
/// trap.
#[cube]
fn slot_of(
    table: &Array<u32>,
    off_at: usize,
    graphs: usize,
    top: usize,
    bits: usize,
    pos: usize,
) -> usize {
    let mut slot = 0usize;
    for bit in 0..bits {
        let next = slot + (top >> bit);
        if next <= graphs {
            if table[off_at + next] as usize <= pos {
                slot = next;
            }
        }
    }
    slot
}

/// One unit per row, slot and (when asked) edge row.
///
/// A row finds its slot by binary search of its index in the slots' row
/// offsets; an edge row does the same in the edge offsets.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn batch_rows_graphs_kernel(
    table: &Array<u32>,
    adj_col: &Array<u32>,
    adj_dst: &Array<u32>,
    gid: &mut Array<u32>,
    row_graph: &mut Array<u32>,
    lengths: &mut Array<u32>,
    eid: &mut Array<u32>,
    edge_src: &mut Array<u32>,
    edge_dst: &mut Array<u32>,
    graph_start_at: usize,
    node_off_at: usize,
    edge_start_at: usize,
    edge_off_at: usize,
    graphs: usize,
    top: usize,
    bits: usize,
    rows: usize,
    edges: usize,
    lanes: usize,
    span: usize,
    ignore: u32,
    #[comptime] with_edges: bool,
) {
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    for pos in start..end {
        if pos < graphs {
            lengths[pos] = table[node_off_at + pos + 1] - table[node_off_at + pos];
        }
        if pos < rows {
            // `graphs` for a row past the last slot's end.
            let g = slot_of(table, node_off_at, graphs, top, bits, pos);
            let mut node = ignore;
            let mut slot = ignore;
            if g < graphs {
                node = table[graph_start_at + g] + (pos as u32 - table[node_off_at + g]);
                slot = g as u32;
            }
            gid[pos] = node;
            row_graph[pos] = slot;
        }
        if comptime!(with_edges) {
            if pos < edges {
                let g = slot_of(table, edge_off_at, graphs, top, bits, pos);
                let mut edge = ignore;
                let mut src = ignore;
                let mut dst = ignore;
                if g < graphs {
                    edge = table[edge_start_at + g] + (pos as u32 - table[edge_off_at + g]);
                    // Dataset node id -> batch row, within this slot.
                    let shift = table[node_off_at + g] - table[graph_start_at + g];
                    src = adj_col[edge as usize] + shift;
                    dst = adj_dst[edge as usize] + shift;
                }
                eid[pos] = edge;
                edge_src[pos] = src;
                edge_dst[pos] = dst;
            }
        }
    }
}

/// G1, whole-graph batches: lay the slots of `slots` out as `rows` rows.
///
/// Row `r` of slot `g` is dataset node `graph_start[g] + r − node_off[g]`.
/// `edges = Some(Eb)` also lays the batch's edges out as `Eb` edge rows, each
/// with its dataset edge id and the batch rows of its ends; that needs the
/// adjacency's destination table. One launch, no upload, no read.
pub fn batch_rows_graphs<R: Runtime>(
    adjacency: &Adjacency<R>,
    slots: &SlotTable<R>,
    rows: usize,
    edges: Option<usize>,
) -> Result<BatchRows<R>> {
    let _op = crate::backend::tally_op_scope("batch_rows_graphs");
    if slots.n_nodes != adjacency.n_nodes || slots.n_edges != adjacency.n_edges {
        return Err(Error::shape(
            "the epoch table was built for a different dataset".to_string(),
        ));
    }
    if rows == 0 || slots.rows_used > rows {
        return Err(Error::shape(format!(
            "a batch of {} rows does not fit a row capacity of {rows}",
            slots.rows_used
        )));
    }
    if let Some(cap) = edges {
        if cap == 0 || slots.edges_used > cap {
            return Err(Error::shape(format!(
                "a batch of {} edges does not fit an edge capacity of {cap}",
                slots.edges_used
            )));
        }
        if adjacency.dst.is_none() {
            return Err(Error::config(
                "edge rows need the store's edge-destination table (built with edge features \
                 or GatedGCN)"
                    .to_string(),
            ));
        }
    }
    let device = adjacency.off.device();
    let graphs = slots.graphs;
    let gid = IdTensor::empty(vec![rows], device);
    let row_graph = IdTensor::empty(vec![rows], device);
    let lengths = IdTensor::empty(vec![graphs], device);
    let edge_cap = edges.unwrap_or(1);
    let eid = IdTensor::empty(vec![edge_cap], device);
    let edge_src = IdTensor::empty(vec![edge_cap], device);
    let edge_dst = IdTensor::empty(vec![edge_cap], device);
    // Bound but never read without edge rows: a buffer is not bound twice.
    let no_dst;
    let adj_dst = match &adjacency.dst {
        Some(dst) => dst,
        None => {
            no_dst = IdTensor::empty(vec![1], device);
            &no_dst
        }
    };
    let lanes = rows.max(graphs).max(edges.unwrap_or(0));
    // Bits of the slot index: `graphs >= 1`, so at least one.
    let bits = (usize::BITS - graphs.leading_zeros()) as usize;
    let client = device.client();
    let (cube_count, cube_dim, span) = launch_1d_spans(client, lanes, 4 * bits);
    // SAFETY: the table is an `EpochTable` built from this dataset's offsets
    // (checked above by node and edge count), so slot offsets are within
    // `rows` / `edges` (checked above) and node and edge ids are within the
    // adjacency; `adj_dst` is present whenever edge rows are asked for.
    unsafe {
        batch_rows_graphs_kernel::launch_unchecked::<R>(
            client,
            cube_count,
            cube_dim,
            slots.table.arg(),
            adjacency.col.arg(),
            adj_dst.arg(),
            gid.arg(),
            row_graph.arg(),
            lengths.arg(),
            eid.arg(),
            edge_src.arg(),
            edge_dst.arg(),
            slots.graph_start_at(),
            slots.node_off_at(),
            slots.edge_start_at(),
            slots.edge_off_at(),
            graphs,
            1usize << (bits - 1),
            bits,
            rows,
            edges.unwrap_or(0),
            lanes,
            span,
            IGNORE,
            edges.is_some(),
        );
    }
    Ok(BatchRows {
        gid,
        row_graph,
        row_of: None,
        lengths,
        rows,
        groups: graphs,
        n_nodes: adjacency.n_nodes,
        slots: Some(slots.clone()),
        edges: edges.map(|cap| BatchEdges {
            eid,
            src: edge_src,
            dst: edge_dst,
            edges: cap,
        }),
    })
}

/// The node part `part` of an epoch takes from block `block`.
///
/// The hash is reduced before the part is added: adding to a full-range `u32`
/// could wrap.
#[cube]
fn subset_pick(block: u32, parts: u32, part: u32, seed_lo: u32, seed_hi: u32) -> u32 {
    block * parts + (hash_u32(block, seed_lo, seed_hi) % parts + part) % parts
}

/// One unit per node (for `row_of`), row and sequence.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn batch_rows_subset_kernel(
    gid: &mut Array<u32>,
    row_graph: &mut Array<u32>,
    row_of: &mut Array<u32>,
    lengths: &mut Array<u32>,
    n_nodes: u32,
    parts: u32,
    part: u32,
    blocks: u32,
    seed_lo: u32,
    seed_hi: u32,
    rows: usize,
    seqs: usize,
    lanes: usize,
    span: usize,
    ignore: u32,
) {
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    for pos in start..end {
        if pos < rows {
            let mut node = ignore;
            let mut slot = ignore;
            if (pos as u32) < blocks {
                let pick = subset_pick(pos as u32, parts, part, seed_lo, seed_hi);
                if pick < n_nodes {
                    node = pick;
                    slot = 0u32;
                }
            }
            gid[pos] = node;
            row_graph[pos] = slot;
        }
        if (pos as u32) < n_nodes {
            let block = pos as u32 / parts;
            let mut row = ignore;
            if subset_pick(block, parts, part, seed_lo, seed_hi) == pos as u32 {
                row = block;
            }
            row_of[pos] = row;
        }
        if pos < seqs {
            // Only the last block can pick a node past the end, so the real
            // rows are a prefix; sequence `k` holds rows `k, k + seqs, …`.
            let mut real = blocks;
            if subset_pick(blocks - 1, parts, part, seed_lo, seed_hi) >= n_nodes {
                real = blocks - 1;
            }
            lengths[pos] = (real + seqs as u32 - 1 - pos as u32) / seqs as u32;
        }
    }
}

/// G1, node-subset batches: part `part` of `parts` of one large graph.
///
/// The canonical order is cut into blocks of `parts` consecutive nodes and the
/// batch takes one node of every block, chosen by a hash of the block and the
/// epoch seed and rotated by `part` — so the `parts` batches of an epoch
/// partition the nodes exactly, and every batch is in canonical order and
/// stratified along it. Rows are `⌈n / parts⌉` rounded up to a multiple of
/// `sequences`; a row whose node would lie past the end is absent, and so are
/// the rounding rows. `lengths` holds the true length of each of the
/// `sequences` interleaved sequences (row `r` is in sequence `r mod sequences`).
/// One launch, no upload, no read.
pub fn batch_rows_subset<R: Runtime>(
    adjacency: &Adjacency<R>,
    parts: usize,
    part: usize,
    seed: (u32, u32),
    sequences: usize,
) -> Result<BatchRows<R>> {
    let _op = crate::backend::tally_op_scope("batch_rows_subset");
    let n = adjacency.n_nodes;
    if parts == 0 || parts > n || part >= parts {
        return Err(Error::config(format!(
            "part {part} of {parts} parts of a graph with {n} nodes: parts must be in 1..={n} \
             and the part below it"
        )));
    }
    if sequences == 0 {
        return Err(Error::config(
            "a batch needs at least one node sequence".to_string(),
        ));
    }
    let blocks = n.div_ceil(parts);
    let rows = blocks.div_ceil(sequences) * sequences;
    let device = adjacency.off.device();
    let gid = IdTensor::empty(vec![rows], device);
    let row_graph = IdTensor::empty(vec![rows], device);
    let row_of = IdTensor::empty(vec![n], device);
    let lengths = IdTensor::empty(vec![sequences], device);
    let lanes = n.max(rows).max(sequences);
    let client = device.client();
    let (cube_count, cube_dim, span) = launch_1d_spans(client, lanes, 8);
    // SAFETY: every write is guarded by its own extent (`rows`, `n_nodes`,
    // `seqs`), which are the lengths of the buffers allocated above, and
    // nothing is read. `blocks >= 1` because the adjacency has a node.
    unsafe {
        batch_rows_subset_kernel::launch_unchecked::<R>(
            client,
            cube_count,
            cube_dim,
            gid.arg(),
            row_graph.arg(),
            row_of.arg(),
            lengths.arg(),
            n as u32,
            parts as u32,
            part as u32,
            blocks as u32,
            seed.0,
            seed.1,
            rows,
            sequences,
            lanes,
            span,
            IGNORE,
        );
    }
    Ok(BatchRows {
        gid,
        row_graph,
        row_of: Some(row_of),
        lengths,
        rows,
        groups: sequences,
        n_nodes: n,
        slots: None,
        edges: None,
    })
}

// ---------------------------------------------------------------------------
// G2. Walk tokens
// ---------------------------------------------------------------------------

/// The shape of a node's token sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WalkShape {
    /// Maximum walk length `m` (`>= 1`).
    pub hops: usize,
    /// Walks per token `M`.
    pub walks: usize,
    /// Tokens per walk length `s`.
    pub repeats: usize,
    /// Weight a token's nodes by their degree inside the token (`Sgc`) instead
    /// of uniformly (`Mean`).
    pub sgc: bool,
}

impl WalkShape {
    /// Tokens per node, `L = m·s + 1`: the last is the node itself.
    pub const fn len(&self) -> usize {
        self.hops * self.repeats + 1
    }

    /// Never: a sequence always holds the node itself.
    pub const fn is_empty(&self) -> bool {
        false
    }

    /// Node slots per token, `C = 1 + M·m`.
    pub const fn cap(&self) -> usize {
        1 + self.walks * self.hops
    }

    /// Check the shape: at least one hop, walk and repeat, and `C <= 128`.
    pub fn validate(&self) -> Result<()> {
        if self.hops == 0 || self.walks == 0 || self.repeats == 0 {
            return Err(Error::config(
                "walk tokens need max_hops, walks and repeats of at least 1".to_string(),
            ));
        }
        if self.cap() > 128 {
            return Err(Error::config(format!(
                "tokens: 1 + walks·max_hops = {} node slots per token, above the limit of 128",
                self.cap()
            )));
        }
        Ok(())
    }
}

/// The random-walk tokens of a batch's rows.
pub struct Tokens<R: Runtime> {
    node: IdTensor<R>,
    w: Tensor<R, f32>,
    stats: Tensor<R, f32>,
    rows: usize,
    shape: WalkShape,
    n_nodes: usize,
}

impl<R: Runtime> Tokens<R> {
    /// Rows the tokens belong to.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// The token shape.
    pub fn shape(&self) -> WalkShape {
        self.shape
    }

    /// `[rows · L · C]` dataset node ids, visited order, [`IGNORE`] padded.
    pub fn node(&self) -> &IdTensor<R> {
        &self.node
    }

    /// `[rows · L · C]` weights; a token's sum to 1.
    pub fn w(&self) -> &Tensor<R, f32> {
        &self.w
    }

    /// `[rows · L, 3]`: `ln(1 + |T|)`, `ln(1 + |E_T|)`, walk length over `m`.
    pub fn stats(&self) -> &Tensor<R, f32> {
        &self.stats
    }
}

/// One unit per token `(row, position)`; scalar, because the walk is a hash per
/// lane.
///
/// Position `t` holds repetition `t mod s` of walk length `m − t / s`; the last
/// position is the node itself. The unit writes the nodes its walks visit into
/// its own `C` output slots in visit order, skipping a node already there (the
/// output is the scratch space), then the weights and the statistics.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn walk_tokens_kernel(
    adj_off: &Array<u32>,
    adj_col: &Array<u32>,
    gid: &Array<u32>,
    tok_node: &mut Array<u32>,
    tok_w: &mut Array<f32>,
    tok_stats: &mut Array<f32>,
    seed_lo: u32,
    seed_hi: u32,
    counter: u32,
    hops: u32,
    walks: u32,
    repeats: u32,
    lanes: usize,
    span: usize,
    ignore: u32,
    #[comptime] sgc: bool,
) {
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    let len = (hops * repeats + 1) as usize;
    let cap = (1 + walks * hops) as usize;
    for pos in start..end {
        let t = (pos % len) as u32;
        let v = gid[pos / len];
        let base = pos * cap;
        // Walk length of this position: `m` first, the node itself (0) last.
        let mut hop = 0u32;
        if t < hops * repeats {
            hop = hops - t / repeats;
        }
        let mut count = 0usize;
        if v != ignore {
            tok_node[base] = v;
            count = 1;
            if hop > 0 {
                // Every component passes through the avalanche: folding one
                // into a seed word by XOR would only flip low bits.
                let k0 = hash_u32(v, seed_lo, seed_hi);
                let k1 = hash_u32(k0 + counter * 0x9E3779B1u32, 0x85EBCA6Bu32, 0xC2B2AE35u32);
                for k in 0..walks {
                    let k2 = hash_u32(
                        k1 + (t * walks + k) * 0x27D4EB2Fu32,
                        0x165667B1u32,
                        0x9E3779B9u32,
                    );
                    let mut u = v;
                    for q in 0..hop {
                        let a = adj_off[u as usize];
                        let deg = adj_off[u as usize + 1] - a;
                        // A node without neighbours stays put.
                        if deg > 0 {
                            let draw =
                                hash_u32(k2 + q * 0x85EBCA77u32, 0xC2B2AE3Du32, 0x27D4EB2Fu32);
                            u = adj_col[(a + draw % deg) as usize];
                            let mut seen = false;
                            for j in 0..count {
                                if tok_node[base + j] == u {
                                    seen = true;
                                }
                            }
                            if !seen {
                                tok_node[base + count] = u;
                                count += 1;
                            }
                        }
                    }
                }
            }
        }
        for j in count..cap {
            tok_node[base + j] = ignore;
            tok_w[base + j] = 0.0f32;
        }

        // Twice the edges of the induced subgraph: the sum of in-token degrees.
        let mut twice_edges = 0u32;
        if comptime!(sgc) {
            // Column sums of `A_T + I`, normalised; `A_T` has no self-loops.
            for i in 0..count {
                let u = tok_node[base + i];
                let a = adj_off[u as usize] as usize;
                let b = adj_off[u as usize + 1] as usize;
                let mut inside = 0u32;
                for j in 0..count {
                    if j != i {
                        let target = tok_node[base + j];
                        let mut lo = a;
                        let mut hi = b;
                        while lo < hi {
                            let mid = (lo + hi) / 2;
                            if adj_col[mid] < target {
                                lo = mid + 1;
                            } else {
                                hi = mid;
                            }
                        }
                        if lo < b {
                            if adj_col[lo] == target {
                                inside += 1;
                            }
                        }
                    }
                }
                tok_w[base + i] = f32::cast_from(1u32 + inside);
                twice_edges += inside;
            }
            let total = f32::cast_from(count as u32 + twice_edges);
            for i in 0..count {
                tok_w[base + i] = tok_w[base + i] / total;
            }
        } else {
            for i in 0..count {
                tok_w[base + i] = 1.0f32 / f32::cast_from(count as u32);
            }
        }

        let mut size = 0.0f32;
        let mut inner = 0.0f32;
        let mut reach = 0.0f32;
        if count > 0 {
            size = f32::ln(1.0f32 + f32::cast_from(count as u32));
            inner = f32::ln(1.0f32 + 0.5f32 * f32::cast_from(twice_edges));
            reach = f32::cast_from(hop) / f32::cast_from(hops);
        }
        tok_stats[pos * 3] = size;
        tok_stats[pos * 3 + 1] = inner;
        tok_stats[pos * 3 + 2] = reach;
    }
}

/// G2: sample every row's random-walk tokens.
///
/// For each row and each of its `L = m·s + 1` positions, `M` walks of the
/// position's length start at the row's node and the token is the set of nodes
/// they visit. The randomness is counter-based: the tokens of a node depend on
/// `(seed, counter, node)` only — not on the batch the node is in — so an
/// evaluation with a fixed counter is reproducible and a training step with a
/// fresh one resamples. Absent rows get empty tokens. One launch.
pub fn walk_tokens<R: Runtime>(
    adjacency: &Adjacency<R>,
    rows: &BatchRows<R>,
    shape: WalkShape,
    seed: (u32, u32),
    counter: u32,
) -> Result<Tokens<R>> {
    let _op = crate::backend::tally_op_scope("walk_tokens");
    shape.validate()?;
    if rows.n_nodes != adjacency.n_nodes {
        return Err(Error::shape(
            "the batch rows index a different dataset than the adjacency".to_string(),
        ));
    }
    let device = adjacency.off.device();
    let (len, cap) = (shape.len(), shape.cap());
    let lanes = rows.rows * len;
    let node = IdTensor::empty(vec![lanes * cap], device);
    let w = Tensor::<R, f32>::empty(vec![lanes * cap], device);
    let stats = Tensor::<R, f32>::empty(vec![lanes, 3], device);
    let client = device.client();
    let work = shape.walks * shape.hops * cap / 2 + if shape.sgc { cap * cap * 4 } else { cap };
    let (cube_count, cube_dim, span) = launch_1d_spans(client, lanes, work);
    // SAFETY: `gid` ids are below the adjacency's node count or `IGNORE`
    // (`BatchRows` is only built by G1; the node counts were compared above), a
    // walk only moves along validated `adj_col` entries, and a unit writes at
    // most `1 + walks·hops = cap` slots, all inside its own `cap`.
    unsafe {
        walk_tokens_kernel::launch_unchecked::<R>(
            client,
            cube_count,
            cube_dim,
            adjacency.off.arg(),
            adjacency.col.arg(),
            rows.gid.arg(),
            node.arg(),
            w.arg(),
            stats.arg(),
            seed.0,
            seed.1,
            counter,
            shape.hops as u32,
            shape.walks as u32,
            shape.repeats as u32,
            lanes,
            span,
            IGNORE,
            shape.sgc,
        );
    }
    Ok(Tokens {
        node,
        w,
        stats,
        rows: rows.rows,
        shape,
        n_nodes: adjacency.n_nodes,
    })
}

#[cube(launch_unchecked)]
fn hash_ids_kernel(ids: &Array<u32>, out: &mut Array<u32>, seed_lo: u32, seed_hi: u32) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = hash_u32(ids[ABSOLUTE_POS], seed_lo, seed_hi);
    }
}

/// [`hash_u32`] of every id, on the device: what the host twin of the hash is
/// checked against.
pub fn hash_ids<R: Runtime>(ids: &IdTensor<R>, seed: (u32, u32)) -> IdTensor<R> {
    let out = IdTensor::empty(ids.shape().clone(), ids.device());
    if out.is_empty() {
        return out;
    }
    let client = ids.device().client();
    let (cube_count, cube_dim) = crate::backend::launch_1d(client, out.len(), 8);
    unsafe {
        hash_ids_kernel::launch_unchecked::<R>(
            client,
            cube_count,
            cube_dim,
            ids.arg(),
            out.arg(),
            seed.0,
            seed.1,
        );
    }
    out
}

// ---------------------------------------------------------------------------
// G3 / G4. Features of tokens and of rows
// ---------------------------------------------------------------------------

/// Per-graph sign flips of a range of positional-encoding columns.
///
/// An eigenvector is defined up to its sign, so a Laplacian encoding is
/// trained with the sign of each column flipped at random per graph; the
/// stored table never changes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SignFlip {
    /// First encoding column that flips.
    pub start: usize,
    /// One past the last encoding column that flips.
    pub end: usize,
    /// Seed of the flips: one sign per `(seed, batch slot, column)`.
    pub seed: (u32, u32),
}

/// The sign of encoding column `col` for the rows of batch slot `slot`.
#[cube]
fn pe_sign(slot: u32, col: u32, seed_lo: u32, seed_hi: u32) -> f32 {
    let mut sign = 1.0f32;
    if hash_u32(slot * 0x9E3779B1u32 + col, seed_lo, seed_hi) & 1u32 == 1u32 {
        sign = -1.0f32;
    }
    sign
}

/// One lane per `(item, vector)` of the output `[items, F + P]`.
///
/// With `tokens`, an item is a token and its value is the weighted sum of its
/// nodes' rows of `x ‖ pe`; without, an item is a row and its value is its own
/// node's row. Items with no node give zeros.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn float_features_kernel<F: Float + CubeElement, N: Size>(
    x: &Array<Vector<F, N>>,
    pe: &Array<Vector<F, N>>,
    node: &Array<u32>,
    weight: &Array<f32>,
    row_graph: &Array<u32>,
    out: &mut Array<Vector<F, N>>,
    f_vec: usize,
    p_vec: usize,
    cap: usize,
    per_row: usize,
    item_at: usize,
    flip_start: usize,
    flip_end: usize,
    seed_lo: u32,
    seed_hi: u32,
    lanes: usize,
    span: usize,
    ignore: u32,
    #[comptime] tokens: bool,
    #[comptime] has_pe: bool,
    #[comptime] flip: bool,
) {
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    let width = f_vec + p_vec;
    for pos in start..end {
        let item = item_at + pos / width;
        let cv = pos % width;
        let mut acc = Vector::<f32, N>::new(0.0_f32);
        if comptime!(tokens) {
            if cv < f_vec {
                for c in 0..cap {
                    let id = node[item * cap + c];
                    // Slots are filled front to back.
                    if id == ignore {
                        break;
                    }
                    acc += Vector::<f32, N>::new(weight[item * cap + c])
                        * Vector::<f32, N>::cast_from(x[id as usize * f_vec + cv]);
                }
            } else {
                if comptime!(has_pe) {
                    for c in 0..cap {
                        let id = node[item * cap + c];
                        if id == ignore {
                            break;
                        }
                        acc += Vector::<f32, N>::new(weight[item * cap + c])
                            * Vector::<f32, N>::cast_from(pe[id as usize * p_vec + (cv - f_vec)]);
                    }
                }
            }
        } else {
            let id = node[item];
            if id != ignore {
                if cv < f_vec {
                    acc = Vector::<f32, N>::cast_from(x[id as usize * f_vec + cv]);
                } else {
                    if comptime!(has_pe) {
                        acc = Vector::<f32, N>::cast_from(pe[id as usize * p_vec + (cv - f_vec)]);
                    }
                }
            }
        }
        if comptime!(flip) {
            // Only launched at a vector width of one, so a lane is a column.
            if cv >= f_vec + flip_start {
                if cv < f_vec + flip_end {
                    let slot = row_graph[item / per_row];
                    if slot != ignore {
                        acc = acc
                            * Vector::<f32, N>::new(pe_sign(
                                slot,
                                (cv - f_vec) as u32,
                                seed_lo,
                                seed_hi,
                            ));
                    }
                }
            }
        }
        out[pos] = Vector::<F, N>::cast_from(acc);
    }
}

/// The field of a multi-hot column: the last field whose offset is not past it.
#[cube]
fn field_of(field_offset: &Array<u32>, fields: usize, col: usize) -> usize {
    let mut field = 0usize;
    for j in 1..fields {
        if field_offset[j] as usize <= col {
            field = j;
        }
    }
    field
}

/// One lane per `(item, column)` of the output `[items, V + P]`: the weighted
/// multi-hot counts of the item's nodes in the first `V` columns, their
/// encodings in the rest.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn id_features_kernel<F: Float + CubeElement>(
    feat_ids: &Array<u32>,
    field_offset: &Array<u32>,
    pe: &Array<F>,
    node: &Array<u32>,
    weight: &Array<f32>,
    row_graph: &Array<u32>,
    out: &mut Array<F>,
    fields: usize,
    vocab: usize,
    pe_dim: usize,
    cap: usize,
    per_row: usize,
    item_at: usize,
    flip_start: usize,
    flip_end: usize,
    seed_lo: u32,
    seed_hi: u32,
    lanes: usize,
    span: usize,
    ignore: u32,
    #[comptime] tokens: bool,
    #[comptime] has_pe: bool,
    #[comptime] flip: bool,
) {
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    let width = vocab + pe_dim;
    for pos in start..end {
        let item = item_at + pos / width;
        let col = pos % width;
        let mut acc = 0.0f32;
        if col < vocab {
            let field = field_of(field_offset, fields, col);
            let want = col as u32 - field_offset[field];
            if comptime!(tokens) {
                for c in 0..cap {
                    let id = node[item * cap + c];
                    if id == ignore {
                        break;
                    }
                    if feat_ids[id as usize * fields + field] == want {
                        acc += weight[item * cap + c];
                    }
                }
            } else {
                let id = node[item];
                if id != ignore {
                    if feat_ids[id as usize * fields + field] == want {
                        acc = 1.0f32;
                    }
                }
            }
        } else {
            if comptime!(has_pe) {
                if comptime!(tokens) {
                    for c in 0..cap {
                        let id = node[item * cap + c];
                        if id == ignore {
                            break;
                        }
                        acc += weight[item * cap + c]
                            * f32::cast_from(pe[id as usize * pe_dim + (col - vocab)]);
                    }
                } else {
                    let id = node[item];
                    if id != ignore {
                        acc = f32::cast_from(pe[id as usize * pe_dim + (col - vocab)]);
                    }
                }
                if comptime!(flip) {
                    if col >= vocab + flip_start {
                        if col < vocab + flip_end {
                            let slot = row_graph[item / per_row];
                            if slot != ignore {
                                acc = acc * pe_sign(slot, (col - vocab) as u32, seed_lo, seed_hi);
                            }
                        }
                    }
                }
            }
        }
        out[pos] = F::cast_from(acc);
    }
}

/// A feature table of the store, as the kernels read it.
pub enum FeatureSource<'a, R: Runtime, E: FloatElem> {
    /// `[n, F]` floats.
    Float(&'a Tensor<R, E>),
    /// `[n, fields]` ids with their `[fields + 1]` multi-hot column offsets;
    /// `vocab` is the multi-hot width.
    Categorical {
        /// `[n · fields]` ids.
        ids: &'a IdTensor<R>,
        /// `[fields + 1]` prefix sums of the vocabulary sizes.
        field_offset: &'a IdTensor<R>,
        /// The multi-hot width.
        vocab: usize,
    },
}

impl<R: Runtime, E: FloatElem> FeatureSource<'_, R, E> {
    /// Width of the model input the table expands to.
    pub fn width(&self) -> usize {
        match self {
            FeatureSource::Float(x) => x.dims()[1],
            FeatureSource::Categorical { vocab, .. } => *vocab,
        }
    }

    /// Rows of the table.
    fn rows(&self) -> usize {
        match self {
            FeatureSource::Float(x) => x.dims()[0],
            FeatureSource::Categorical {
                ids, field_offset, ..
            } => ids.len() / (field_offset.len() - 1).max(1),
        }
    }

    fn check(&self, source_rows: usize) -> Result<()> {
        let ok = match self {
            FeatureSource::Float(x) => x.rank() == 2 && x.dims()[1] > 0,
            FeatureSource::Categorical {
                ids,
                field_offset,
                vocab,
            } => {
                field_offset.len() >= 2
                    && *vocab > 0
                    && ids.len().is_multiple_of(field_offset.len() - 1)
            }
        };
        if !ok || self.rows() != source_rows {
            return Err(Error::shape(format!(
                "a feature table must hold one row per source row ({source_rows})"
            )));
        }
        Ok(())
    }
}

/// What an item of a feature gather is made of.
enum Items<'a, R: Runtime> {
    /// `[items · cap]` node slots and their weights; `per_row` tokens per row.
    Tokens {
        node: &'a IdTensor<R>,
        weight: &'a Tensor<R, f32>,
        cap: usize,
        per_row: usize,
    },
    /// `[items]` ids, one source row each.
    Rows(&'a IdTensor<R>),
}

/// The shared launcher of G3 and G4.
fn gather_features<R: Runtime, E: FloatElem>(
    features: &FeatureSource<'_, R, E>,
    pe: Option<&Tensor<R, E>>,
    items: Items<'_, R>,
    item_at: usize,
    count: usize,
    source_rows: usize,
    row_graph: &IdTensor<R>,
    flip: Option<SignFlip>,
) -> Result<Tensor<R, E>> {
    features.check(source_rows)?;
    let pe_dim = match pe {
        Some(pe) => {
            if pe.rank() != 2 || pe.dims()[0] != source_rows || pe.dims()[1] == 0 {
                return Err(Error::shape(format!(
                    "the encoding table must be [{source_rows}, pe_dim], got {}",
                    pe.shape()
                )));
            }
            pe.dims()[1]
        }
        None => 0,
    };
    let flip = flip.filter(|f| f.start < f.end);
    if let Some(f) = &flip
        && f.end > pe_dim
    {
        return Err(Error::shape(format!(
            "sign flips cover encoding columns {}..{} of {pe_dim}",
            f.start, f.end
        )));
    }
    let width = features.width() + pe_dim;
    let device = row_graph.device();
    let out = Tensor::<R, E>::empty(Shape::new(vec![count, width]), device);
    if out.is_empty() {
        return Ok(out);
    }
    let (tokens, node, weight_dummy, cap, per_row);
    let weight = match items {
        Items::Tokens {
            node: n,
            weight,
            cap: c,
            per_row: p,
        } => {
            (tokens, node, cap, per_row) = (true, n, c, p);
            weight
        }
        Items::Rows(ids) => {
            (tokens, node, cap, per_row) = (false, ids, 1, 1);
            // Never read: the row form takes a weight of one.
            weight_dummy = Tensor::<R, f32>::empty(vec![1], device);
            &weight_dummy
        }
    };
    let (flip_start, flip_end, seed) = match &flip {
        Some(f) => (f.start, f.end, f.seed),
        None => (0, 0, (0, 0)),
    };
    let client = device.client();
    // Bound but never read without an encoding: a buffer is not bound twice.
    let pe_dummy;
    let pe_arg = match pe {
        Some(pe) => pe,
        None => {
            pe_dummy = Tensor::<R, E>::empty(vec![16], device);
            &pe_dummy
        }
    };
    match features {
        FeatureSource::Float(x) => {
            let f = x.dims()[1];
            // A flipped column has its own sign, so a lane must be one column.
            let line = if flip.is_some() {
                1
            } else if pe_dim > 0 {
                line_dividing::<R, E>(client, &[f, pe_dim])
            } else {
                line_dividing::<R, E>(client, &[f])
            };
            let lanes = count * (width / line);
            let (cube_count, cube_dim, span) = launch_1d_spans(client, lanes, cap * line);
            // SAFETY: node ids are below `source_rows` or `IGNORE` (they come
            // from `BatchRows` / `Tokens` / `BatchEdges`, whose dataset size the
            // caller compared with `source_rows`), both tables hold
            // `source_rows` rows (checked above), `row_graph` holds one slot per
            // row and `item / per_row` is a row, and the vector width divides
            // both table widths.
            unsafe {
                float_features_kernel::launch_unchecked::<E, R>(
                    client,
                    cube_count,
                    cube_dim,
                    line,
                    x.arg(),
                    pe_arg.arg(),
                    node.arg(),
                    weight.arg(),
                    row_graph.arg(),
                    out.arg(),
                    f / line,
                    pe_dim / line,
                    cap,
                    per_row,
                    item_at,
                    flip_start,
                    flip_end,
                    seed.0,
                    seed.1,
                    lanes,
                    span,
                    IGNORE,
                    tokens,
                    pe.is_some(),
                    flip.is_some(),
                );
            }
        }
        FeatureSource::Categorical {
            ids,
            field_offset,
            vocab,
        } => {
            let fields = field_offset.len() - 1;
            let lanes = count * width;
            let (cube_count, cube_dim, span) = launch_1d_spans(client, lanes, cap + fields);
            // SAFETY: as above; feature ids are only compared, never indexed
            // with, and `field_of` stays below `fields`.
            unsafe {
                id_features_kernel::launch_unchecked::<E, R>(
                    client,
                    cube_count,
                    cube_dim,
                    ids.arg(),
                    field_offset.arg(),
                    pe_arg.arg(),
                    node.arg(),
                    weight.arg(),
                    row_graph.arg(),
                    out.arg(),
                    fields,
                    *vocab,
                    pe_dim,
                    cap,
                    per_row,
                    item_at,
                    flip_start,
                    flip_end,
                    seed.0,
                    seed.1,
                    lanes,
                    span,
                    IGNORE,
                    tokens,
                    pe.is_some(),
                    flip.is_some(),
                );
            }
        }
    }
    Ok(out)
}

/// G3: the input of every token, `[rows · L, F + pe_dim]` (or `[rows · L,
/// V + pe_dim]` for categorical features): the weighted sum of the token's
/// nodes' features — weighted multi-hot counts for ids — next to the weighted
/// sum of their encodings.
///
/// The weights of a token sum to one, so an affine embedding applied to this is
/// the weighted mean of the embedded nodes: the token path needs no gradient
/// and no adjoint. Accumulated in `f32`. One launch.
pub fn token_features<R: Runtime, E: FloatElem>(
    features: &FeatureSource<'_, R, E>,
    pe: Option<&Tensor<R, E>>,
    tokens: &Tokens<R>,
    rows: &BatchRows<R>,
    flip: Option<SignFlip>,
) -> Result<Tensor<R, E>> {
    token_features_rows(features, pe, tokens, rows, flip, 0, rows.rows)
}

/// [`token_features`] for the rows `first_row..first_row + n_rows` only,
/// `[n_rows · L, width]`: how a batch whose token features would be one
/// oversized allocation is produced in chunks. One launch.
pub fn token_features_rows<R: Runtime, E: FloatElem>(
    features: &FeatureSource<'_, R, E>,
    pe: Option<&Tensor<R, E>>,
    tokens: &Tokens<R>,
    rows: &BatchRows<R>,
    flip: Option<SignFlip>,
    first_row: usize,
    n_rows: usize,
) -> Result<Tensor<R, E>> {
    let _op = crate::backend::tally_op_scope("token_features_rows");
    if tokens.rows != rows.rows || tokens.n_nodes != rows.n_nodes {
        return Err(Error::shape(
            "the tokens were sampled for a different batch".to_string(),
        ));
    }
    if first_row + n_rows > rows.rows {
        return Err(Error::shape(format!(
            "rows {first_row}..{} are outside a batch of {} rows",
            first_row + n_rows,
            rows.rows
        )));
    }
    let shape = tokens.shape;
    gather_features(
        features,
        pe,
        Items::Tokens {
            node: &tokens.node,
            weight: &tokens.w,
            cap: shape.cap(),
            per_row: shape.len(),
        },
        first_row * shape.len(),
        n_rows * shape.len(),
        tokens.n_nodes,
        &rows.row_graph,
        flip,
    )
}

/// G4: the input of every row, `[rows, F + pe_dim]` (or `[rows, V + pe_dim]`
/// multi-hot for categorical features); absent rows give zeros. One launch.
pub fn node_inputs<R: Runtime, E: FloatElem>(
    features: &FeatureSource<'_, R, E>,
    pe: Option<&Tensor<R, E>>,
    rows: &BatchRows<R>,
    flip: Option<SignFlip>,
) -> Result<Tensor<R, E>> {
    let _op = crate::backend::tally_op_scope("node_inputs");
    gather_features(
        features,
        pe,
        Items::Rows(&rows.gid),
        0,
        rows.rows,
        rows.n_nodes,
        &rows.row_graph,
        flip,
    )
}

/// G4 for edges: the input of every edge row, `[edges, Fe]` (or multi-hot);
/// absent edge rows give zeros. One launch.
pub fn edge_inputs<R: Runtime, E: FloatElem>(
    features: &FeatureSource<'_, R, E>,
    rows: &BatchRows<R>,
    n_edges: usize,
) -> Result<Tensor<R, E>> {
    let _op = crate::backend::tally_op_scope("edge_inputs");
    let edges = rows.edges.as_ref().ok_or_else(|| {
        Error::config("this batch was laid out without its edges".to_string())
    })?;
    let slots = rows.slots.as_ref().expect("edge rows come with slots");
    if slots.n_edges != n_edges {
        return Err(Error::shape(
            "the edge feature table belongs to a different dataset".to_string(),
        ));
    }
    gather_features(
        features,
        None,
        Items::Rows(&edges.eid),
        0,
        edges.edges,
        n_edges,
        &rows.row_graph,
        None,
    )
}

// ---------------------------------------------------------------------------
// G4. Targets
// ---------------------------------------------------------------------------

/// One unit per item: its class, or 0 with a zero mask.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn safe_class_targets_kernel<F: Float + CubeElement>(
    y: &Array<u32>,
    split: &Array<u32>,
    index: &Array<u32>,
    ids: &mut Array<u32>,
    mask: &mut Array<F>,
    index_at: usize,
    flag: u32,
    lanes: usize,
    span: usize,
    ignore: u32,
) {
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    for pos in start..end {
        let item = index[index_at + pos];
        let mut class = 0u32;
        let mut keep = F::new(0.0_f32);
        if item != ignore {
            let label = y[item as usize];
            if label != ignore {
                if split[item as usize] & flag != 0u32 {
                    class = label;
                    keep = F::new(1.0_f32);
                }
            }
        }
        ids[pos] = class;
        mask[pos] = keep;
    }
}

/// G4: the class of every row for a node task — or of every slot for a graph
/// task — with 0 in place of a missing one, and the mask that says which were
/// real: the label is present **and** the item's split has one of `flags`
/// **and** the row or slot is not absent.
///
/// `y` and `split` are the store's `[n_nodes]` (`[n_graphs]`) tables. Masking
/// happens here, before any loss indexes with a class.
pub fn safe_class_targets<R: Runtime, E: FloatElem>(
    y: &IdTensor<R>,
    split: &IdTensor<R>,
    rows: &BatchRows<R>,
    per_graph: bool,
    flags: u32,
) -> Result<(IdTensor<R>, Tensor<R, E>)> {
    let _op = crate::backend::tally_op_scope("safe_class_targets");
    let (index, index_at, count, items) = if per_graph {
        let slots = rows.slots.as_ref().ok_or_else(|| {
            Error::config("graph targets need a batch of whole graphs".to_string())
        })?;
        // One label per graph of the dataset the epoch was laid out from: the
        // kernel indexes them by the dataset graph ids in the slot table.
        (
            &slots.table,
            slots.graph_id_at(),
            slots.graphs,
            slots.n_graphs,
        )
    } else {
        (&rows.gid, 0, rows.rows, rows.n_nodes)
    };
    if y.len() != items || split.len() != items {
        return Err(Error::shape(format!(
            "class targets need [{items}] labels and split flags, got {} and {}",
            y.len(),
            split.len()
        )));
    }
    let device = y.device();
    let ids = IdTensor::empty(vec![count], device);
    let mask = Tensor::<R, E>::empty(vec![count], device);
    let client = device.client();
    let (cube_count, cube_dim, span) = launch_1d_spans(client, count, 4);
    // SAFETY: `index` holds ids below `items` or `IGNORE` — row ids from G1
    // (`items == n_nodes`, checked) or the epoch table's graph ids, which are
    // below the dataset's graph count; a table of another length is refused
    // above for node targets, and for graph targets the store pairs the table
    // with its own dataset.
    unsafe {
        safe_class_targets_kernel::launch_unchecked::<E, R>(
            client,
            cube_count,
            cube_dim,
            y.arg(),
            split.arg(),
            index.arg(),
            ids.arg(),
            mask.arg(),
            index_at,
            flags,
            count,
            span,
            IGNORE,
        );
    }
    Ok((ids, mask))
}

/// One unit per `(slot, target)`.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn safe_float_targets_kernel<F: Float + CubeElement>(
    y: &Array<F>,
    present: &Array<F>,
    split: &Array<u32>,
    index: &Array<u32>,
    targets: &mut Array<F>,
    mask: &mut Array<F>,
    index_at: usize,
    width: usize,
    flag: u32,
    lanes: usize,
    span: usize,
    ignore: u32,
) {
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    for pos in start..end {
        let item = index[index_at + pos / width];
        let mut value = F::new(0.0_f32);
        let mut keep = F::new(0.0_f32);
        if item != ignore {
            if split[item as usize] & flag != 0u32 {
                value = y[item as usize * width + pos % width];
                keep = present[item as usize * width + pos % width];
            }
        }
        targets[pos] = value;
        mask[pos] = keep;
    }
}

/// G4: the float targets of every slot, `[graphs, targets]`, with 0 in place
/// of a missing one, and the mask of the real ones.
///
/// `y` is the store's `[n_graphs, targets]` table with zeros where a target is
/// missing and `present` its `[n_graphs, targets]` mask — the device never
/// holds a `NaN`, whose comparison a shader compiler is free to fold away.
pub fn safe_float_targets<R: Runtime, E: FloatElem>(
    y: &Tensor<R, E>,
    present: &Tensor<R, E>,
    split: &IdTensor<R>,
    rows: &BatchRows<R>,
    flags: u32,
) -> Result<(Tensor<R, E>, Tensor<R, E>)> {
    let _op = crate::backend::tally_op_scope("safe_float_targets");
    let slots = rows
        .slots
        .as_ref()
        .ok_or_else(|| Error::config("graph targets need a batch of whole graphs".to_string()))?;
    if y.rank() != 2
        || y.shape() != present.shape()
        || y.dims()[0] != split.len()
        || y.dims()[0] != slots.n_graphs
    {
        return Err(Error::shape(format!(
            "float targets need [{0}, targets] values and mask and [{0}] split flags (one row \
             per graph of the dataset), got {1}, {2} and {3}",
            slots.n_graphs,
            y.shape(),
            present.shape(),
            split.shape()
        )));
    }
    let width = y.dims()[1];
    let device = y.device();
    let shape = Shape::new(vec![slots.graphs, width]);
    let targets = Tensor::<R, E>::empty(shape.clone(), device);
    let mask = Tensor::<R, E>::empty(shape, device);
    let lanes = slots.graphs * width;
    if lanes == 0 {
        return Ok((targets, mask));
    }
    let client = device.client();
    let (cube_count, cube_dim, span) = launch_1d_spans(client, lanes, 4);
    // SAFETY: the epoch table's graph ids are below the dataset's graph count,
    // which the store pairs with these tables; shapes checked above.
    unsafe {
        safe_float_targets_kernel::launch_unchecked::<E, R>(
            client,
            cube_count,
            cube_dim,
            y.arg(),
            present.arg(),
            split.arg(),
            slots.table.arg(),
            targets.arg(),
            mask.arg(),
            slots.graph_id_at(),
            width,
            flags,
            lanes,
            span,
            IGNORE,
        );
    }
    Ok((targets, mask))
}

// ---------------------------------------------------------------------------
// G5. Ragged padding
// ---------------------------------------------------------------------------

/// One lane per `(slot, position, vector)` of the padded output.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn pad_ragged_kernel<F: Float + CubeElement, N: Size>(
    input: &Array<Vector<F, N>>,
    table: &Array<u32>,
    out: &mut Array<Vector<F, N>>,
    node_off_at: usize,
    nmax: usize,
    dvec: usize,
    lanes: usize,
    span: usize,
) {
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    for pos in start..end {
        let dv = pos % dvec;
        let p = (pos / dvec) % nmax;
        let g = pos / (dvec * nmax);
        let first = table[node_off_at + g] as usize;
        let len = table[node_off_at + g + 1] as usize - first;
        if p < len {
            out[pos] = input[(first + p) * dvec + dv];
        } else {
            out[pos] = Vector::<F, N>::new(F::new(0.0_f32));
        }
    }
}

/// G5: `[rows, d] → [graphs, nmax, d]`, each slot's rows followed by zeros.
/// The adjoint of [`unpad_ragged`]. One launch.
pub fn pad_ragged<R: Runtime, E: FloatElem>(
    input: &Tensor<R, E>,
    rows: &BatchRows<R>,
    nmax: usize,
) -> Result<Tensor<R, E>> {
    let _op = crate::backend::tally_op_scope("pad_ragged");
    let slots = ragged_slots(input, rows)?;
    let d = input.dims()[1];
    let out = Tensor::<R, E>::empty(Shape::new(vec![slots.graphs, nmax, d]), input.device());
    if out.is_empty() {
        return Ok(out);
    }
    let client = input.client();
    let line = line_dividing::<R, E>(client, &[d]);
    let dvec = d / line;
    let lanes = slots.graphs * nmax * dvec;
    let (cube_count, cube_dim, span) = launch_1d_spans(client, lanes, line);
    // SAFETY: a slot's rows lie inside `rows.rows` (the batch was laid out from
    // this table) and `input` has that many rows (`ragged_slots`); a position
    // past a slot's length reads nothing.
    unsafe {
        pad_ragged_kernel::launch_unchecked::<E, R>(
            client,
            cube_count,
            cube_dim,
            line,
            input.arg(),
            slots.table.arg(),
            out.arg(),
            slots.node_off_at(),
            nmax,
            dvec,
            lanes,
            span,
        );
    }
    Ok(out)
}

/// One lane per `(row, vector)` of the unpadded output.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn unpad_ragged_kernel<F: Float + CubeElement, N: Size>(
    input: &Array<Vector<F, N>>,
    table: &Array<u32>,
    row_graph: &Array<u32>,
    out: &mut Array<Vector<F, N>>,
    node_off_at: usize,
    nmax: usize,
    dvec: usize,
    lanes: usize,
    span: usize,
    ignore: u32,
) {
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    for pos in start..end {
        let dv = pos % dvec;
        let row = pos / dvec;
        let g = row_graph[row];
        let mut value = Vector::<F, N>::new(F::new(0.0_f32));
        if g != ignore {
            let p = row - table[node_off_at + g as usize] as usize;
            // A graph longer than `nmax` has no padded position to read.
            if p < nmax {
                value = input[(g as usize * nmax + p) * dvec + dv];
            }
        }
        out[pos] = value;
    }
}

/// G5: `[graphs, nmax, d] → [rows, d]`, each row from its slot's position;
/// absent rows give zeros. The adjoint of [`pad_ragged`]. One launch.
pub fn unpad_ragged<R: Runtime, E: FloatElem>(
    input: &Tensor<R, E>,
    rows: &BatchRows<R>,
) -> Result<Tensor<R, E>> {
    let _op = crate::backend::tally_op_scope("unpad_ragged");
    let slots = rows
        .slots
        .as_ref()
        .ok_or_else(|| Error::config("padding needs a batch of whole graphs".to_string()))?;
    if input.rank() != 3 || input.dims()[0] != slots.graphs {
        return Err(Error::shape(format!(
            "unpad_ragged needs [{}, nmax, d], got {}",
            slots.graphs,
            input.shape()
        )));
    }
    let (nmax, d) = (input.dims()[1], input.dims()[2]);
    let out = Tensor::<R, E>::empty(Shape::new(vec![rows.rows, d]), input.device());
    if out.is_empty() {
        return Ok(out);
    }
    let client = input.client();
    let line = line_dividing::<R, E>(client, &[d]);
    let dvec = d / line;
    let lanes = rows.rows * dvec;
    let (cube_count, cube_dim, span) = launch_1d_spans(client, lanes, line);
    // SAFETY: `row_graph` holds slots below `graphs` or `IGNORE` (G1), a row's
    // position is its distance from its slot's first row, and a position at or
    // past `nmax` is not read.
    unsafe {
        unpad_ragged_kernel::launch_unchecked::<E, R>(
            client,
            cube_count,
            cube_dim,
            line,
            input.arg(),
            slots.table.arg(),
            rows.row_graph.arg(),
            out.arg(),
            slots.node_off_at(),
            nmax,
            dvec,
            lanes,
            span,
            IGNORE,
        );
    }
    Ok(out)
}

/// The slot table of a whole-graph batch, after checking `input` is `[rows, d]`.
fn ragged_slots<'a, R: Runtime, E: FloatElem>(
    input: &Tensor<R, E>,
    rows: &'a BatchRows<R>,
) -> Result<&'a SlotTable<R>> {
    let slots = rows
        .slots
        .as_ref()
        .ok_or_else(|| Error::config("this needs a batch of whole graphs".to_string()))?;
    if input.rank() != 2 || input.dims()[0] != rows.rows {
        return Err(Error::shape(format!(
            "expected [{}, d] rows, got {}",
            rows.rows,
            input.shape()
        )));
    }
    Ok(slots)
}

// ---------------------------------------------------------------------------
// G6. Segment pooling
// ---------------------------------------------------------------------------

/// One lane per `(slot, vector)`: the sum, or the mean, of the slot's rows.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn segment_pool_kernel<F: Float + CubeElement, N: Size>(
    input: &Array<Vector<F, N>>,
    table: &Array<u32>,
    out: &mut Array<Vector<F, N>>,
    node_off_at: usize,
    dvec: usize,
    lanes: usize,
    span: usize,
    #[comptime] mean: bool,
) {
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    for pos in start..end {
        let dv = pos % dvec;
        let g = pos / dvec;
        let first = table[node_off_at + g] as usize;
        let last = table[node_off_at + g + 1] as usize;
        let mut acc = Vector::<f32, N>::new(0.0_f32);
        for row in first..last {
            acc += Vector::<f32, N>::cast_from(input[row * dvec + dv]);
        }
        if comptime!(mean) {
            if last > first {
                acc = acc / Vector::<f32, N>::new(f32::cast_from((last - first) as u32));
            }
        }
        out[pos] = Vector::<F, N>::cast_from(acc);
    }
}

/// G6: `[rows, d] → [graphs, d]`, the sum or the mean of each slot's rows; an
/// empty slot gives zeros. Accumulated in `f32`. One launch.
pub fn segment_pool<R: Runtime, E: FloatElem>(
    input: &Tensor<R, E>,
    rows: &BatchRows<R>,
    mean: bool,
) -> Result<Tensor<R, E>> {
    let _op = crate::backend::tally_op_scope("segment_pool");
    let slots = ragged_slots(input, rows)?;
    let d = input.dims()[1];
    let out = Tensor::<R, E>::empty(Shape::new(vec![slots.graphs, d]), input.device());
    if out.is_empty() {
        return Ok(out);
    }
    let client = input.client();
    let line = line_dividing::<R, E>(client, &[d]);
    let dvec = d / line;
    let lanes = slots.graphs * dvec;
    let per_slot = slots.rows_used.div_ceil(slots.graphs).max(1);
    let (cube_count, cube_dim, span) = launch_1d_spans(client, lanes, per_slot * line);
    // SAFETY: as `pad_ragged`.
    unsafe {
        segment_pool_kernel::launch_unchecked::<E, R>(
            client,
            cube_count,
            cube_dim,
            line,
            input.arg(),
            slots.table.arg(),
            out.arg(),
            slots.node_off_at(),
            dvec,
            lanes,
            span,
            mean,
        );
    }
    Ok(out)
}

/// One lane per `(row, vector)`: its slot's value, over the slot's length for
/// the mean.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn segment_broadcast_kernel<F: Float + CubeElement, N: Size>(
    input: &Array<Vector<F, N>>,
    table: &Array<u32>,
    row_graph: &Array<u32>,
    out: &mut Array<Vector<F, N>>,
    node_off_at: usize,
    dvec: usize,
    lanes: usize,
    span: usize,
    ignore: u32,
    #[comptime] mean: bool,
) {
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    for pos in start..end {
        let dv = pos % dvec;
        let g = row_graph[pos / dvec];
        let mut value = Vector::<f32, N>::new(0.0_f32);
        if g != ignore {
            value = Vector::<f32, N>::cast_from(input[g as usize * dvec + dv]);
            if comptime!(mean) {
                let len = table[node_off_at + g as usize + 1] - table[node_off_at + g as usize];
                value = value / Vector::<f32, N>::new(f32::cast_from(len));
            }
        }
        out[pos] = Vector::<F, N>::cast_from(value);
    }
}

/// G6's adjoint: `[graphs, d] → [rows, d]`, each row its slot's value (over the
/// slot's length for the mean); absent rows give zeros. One launch.
pub fn segment_broadcast<R: Runtime, E: FloatElem>(
    input: &Tensor<R, E>,
    rows: &BatchRows<R>,
    mean: bool,
) -> Result<Tensor<R, E>> {
    let _op = crate::backend::tally_op_scope("segment_broadcast");
    let slots = rows
        .slots
        .as_ref()
        .ok_or_else(|| Error::config("pooling needs a batch of whole graphs".to_string()))?;
    if input.rank() != 2 || input.dims()[0] != slots.graphs {
        return Err(Error::shape(format!(
            "segment_broadcast needs [{}, d], got {}",
            slots.graphs,
            input.shape()
        )));
    }
    let d = input.dims()[1];
    let out = Tensor::<R, E>::empty(Shape::new(vec![rows.rows, d]), input.device());
    if out.is_empty() {
        return Ok(out);
    }
    let client = input.client();
    let line = line_dividing::<R, E>(client, &[d]);
    let dvec = d / line;
    let lanes = rows.rows * dvec;
    let (cube_count, cube_dim, span) = launch_1d_spans(client, lanes, line);
    // SAFETY: `row_graph` holds slots below `graphs` or `IGNORE` (G1), and a
    // row that names a slot makes that slot's length at least one.
    unsafe {
        segment_broadcast_kernel::launch_unchecked::<E, R>(
            client,
            cube_count,
            cube_dim,
            line,
            input.arg(),
            slots.table.arg(),
            rows.row_graph.arg(),
            out.arg(),
            slots.node_off_at(),
            dvec,
            lanes,
            span,
            IGNORE,
            mean,
        );
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// G8. Confusion counts
// ---------------------------------------------------------------------------

/// One unit per cell `(true class, predicted class)`, counting the kept rows
/// that fall in it.
#[cube(launch_unchecked)]
fn confusion_kernel<F: Float + CubeElement>(
    predicted: &Array<u32>,
    target: &Array<u32>,
    mask: &Array<F>,
    out: &mut Array<u32>,
    classes: usize,
    rows: usize,
) {
    if ABSOLUTE_POS < classes * classes {
        let want_true = (ABSOLUTE_POS / classes) as u32;
        let want_pred = (ABSOLUTE_POS % classes) as u32;
        let mut count = 0u32;
        for row in 0..rows {
            if mask[row] != F::new(0.0_f32) {
                if target[row] == want_true {
                    if predicted[row] == want_pred {
                        count += 1;
                    }
                }
            }
        }
        out[ABSOLUTE_POS] = count;
    }
}

/// G8: `[classes, classes]` counts, row = true class, column = predicted
/// class, over the rows `mask` keeps. Accuracy and macro-F1 are read off the
/// sum of these over a split's batches. One launch.
pub fn confusion<R: Runtime, E: FloatElem>(
    predicted: &IdTensor<R>,
    target: &IdTensor<R>,
    mask: &Tensor<R, E>,
    classes: usize,
) -> Result<IdTensor<R>> {
    let _op = crate::backend::tally_op_scope("confusion");
    let rows = predicted.len();
    if target.len() != rows || mask.len() != rows || classes == 0 {
        return Err(Error::shape(format!(
            "confusion needs [{rows}] predictions, targets and mask and at least one class"
        )));
    }
    let out = IdTensor::empty(vec![classes, classes], predicted.device());
    let client = predicted.device().client();
    let (cube_count, cube_dim) = crate::backend::launch_1d(client, classes * classes, rows);
    // SAFETY: the three inputs hold `rows` elements (checked above) and are
    // only compared; the output holds `classes²` cells.
    unsafe {
        confusion_kernel::launch_unchecked::<E, R>(
            client,
            cube_count,
            cube_dim,
            predicted.arg(),
            target.arg(),
            mask.arg(),
            out.arg(),
            classes,
            rows,
        );
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Masked mean
// ---------------------------------------------------------------------------

/// A single unit: the masked sum over the count of kept elements, both
/// accumulated in `f32`.
#[cube(launch_unchecked)]
fn masked_mean_kernel<F: Float + CubeElement>(
    input: &Array<F>,
    mask: &Array<F>,
    out: &mut Array<F>,
    inv_count: &mut Array<f32>,
    len: usize,
) {
    if ABSOLUTE_POS == 0 {
        let mut sum = 0.0f32;
        let mut count = 0.0f32;
        for i in 0..len {
            let m = f32::cast_from(mask[i]);
            sum += f32::cast_from(input[i]) * m;
            count += m;
        }
        let kept = count;
        if count < 1.0f32 {
            count = 1.0f32;
        }
        out[0] = F::cast_from(sum / count);
        inv_count[0] = 1.0f32 / count;
        inv_count[1] = kept;
    }
}

/// `Σ input·mask / max(Σ mask, 1)` as a `[1]` tensor, and a `[2]` `f32` tensor
/// holding the reciprocal of the denominator and the count `Σ mask` itself.
/// Both stay in `f32` whatever the element type: the adjoint reads the first
/// (a count above a 16-bit float's range must not round to infinity there),
/// and a metric summed over batches reads the second. One launch.
pub fn masked_mean<R: Runtime, E: FloatElem>(
    input: &Tensor<R, E>,
    mask: &Tensor<R, E>,
) -> Result<(Tensor<R, E>, Tensor<R, f32>)> {
    let _op = crate::backend::tally_op_scope("masked_mean");
    if input.len() != mask.len() || input.is_empty() {
        return Err(Error::shape(format!(
            "masked_mean needs a mask the size of its non-empty input, got {} and {}",
            input.shape(),
            mask.shape()
        )));
    }
    let device = input.device();
    let out = Tensor::<R, E>::empty(vec![1], device);
    let inv_count = Tensor::<R, f32>::empty(vec![2], device);
    let client = device.client();
    crate::backend::count_launch();
    // SAFETY: both inputs hold `len` elements (checked above).
    unsafe {
        masked_mean_kernel::launch_unchecked::<E, R>(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            input.arg(),
            mask.arg(),
            out.arg(),
            inv_count.arg(),
            input.len(),
        );
    }
    Ok((out, inv_count))
}

/// One lane per vector of the mask: `grad · mask · inv_count`.
#[cube(launch_unchecked)]
fn masked_mean_backward_kernel<F: Float + CubeElement, N: Size>(
    grad: &Array<F>,
    mask: &Array<Vector<F, N>>,
    inv_count: &Array<f32>,
    out: &mut Array<Vector<F, N>>,
    lanes: usize,
    span: usize,
) {
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    let scale = Vector::<f32, N>::new(f32::cast_from(grad[0]) * inv_count[0]);
    for pos in start..end {
        out[pos] = Vector::<F, N>::cast_from(Vector::<f32, N>::cast_from(mask[pos]) * scale);
    }
}

/// Adjoint of [`masked_mean`] with respect to its input.
pub fn masked_mean_backward<R: Runtime, E: FloatElem>(
    grad: &Tensor<R, E>,
    mask: &Tensor<R, E>,
    inv_count: &Tensor<R, f32>,
) -> Result<Tensor<R, E>> {
    let _op = crate::backend::tally_op_scope("masked_mean_backward");
    if grad.len() != 1 || inv_count.len() != 2 {
        return Err(Error::shape(
            "masked_mean_backward needs a [1] gradient and masked_mean's [2] denominator"
                .to_string(),
        ));
    }
    let out = Tensor::<R, E>::empty(mask.shape().clone(), mask.device());
    let client = mask.client();
    let line = crate::backend::line_size_for::<R, E>(client, mask.len());
    let lanes = mask.len() / line;
    let (cube_count, cube_dim, span) = launch_1d_spans(client, lanes, line);
    // SAFETY: the output has the mask's length, which the vector width divides.
    unsafe {
        masked_mean_backward_kernel::launch_unchecked::<E, R>(
            client,
            cube_count,
            cube_dim,
            line,
            grad.arg(),
            mask.arg(),
            inv_count.arg(),
            out.arg(),
            lanes,
            span,
        );
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// G7. Message passing
// ---------------------------------------------------------------------------
//
// Both layers run on the store's CSR: row `u` of the adjacency lists the edges
// into `u`, and the graph is symmetric, so the edges *out of* `u` are the
// reverses of those — which is what makes every adjoint a gather over the same
// rows. A neighbour `c` of a row becomes a batch row by the batch descriptor:
// `c − graph_start + node_off` in a whole-graph batch (one wrapping shift per
// slot; an edge id shifts the same way to its edge row), `row_of[c]` in a
// node-subset batch, where a neighbour outside the batch is skipped — in the
// forward pass and in the adjoints alike.

/// What a node-lane message-passing kernel needs to turn a neighbour into a
/// batch row: the slot table of a whole-graph batch, or `row_of` of a
/// node-subset batch, with a stand-in for whichever is not used (a buffer is
/// never bound twice).
struct RowContext<R: Runtime> {
    table: IdTensor<R>,
    row_of: IdTensor<R>,
    graph_start_at: usize,
    node_off_at: usize,
    edge_start_at: usize,
    edge_off_at: usize,
    subset: bool,
}

fn row_context<R: Runtime>(
    adjacency: &Adjacency<R>,
    rows: &BatchRows<R>,
) -> Result<RowContext<R>> {
    // Node and edge counts both: the kernels shift dataset edge ids into the
    // batch's edge rows, which is only in range for the adjacency the epoch's
    // edge offsets were taken from.
    let other_edges = rows
        .slots
        .as_ref()
        .is_some_and(|slots| slots.n_edges != adjacency.n_edges);
    if rows.n_nodes != adjacency.n_nodes || other_edges {
        return Err(Error::shape(
            "the batch rows index a different dataset than the adjacency".to_string(),
        ));
    }
    let device = adjacency.off.device();
    Ok(match (&rows.slots, &rows.row_of) {
        (Some(slots), _) => RowContext {
            table: slots.table.clone(),
            row_of: IdTensor::empty(vec![1], device),
            graph_start_at: slots.graph_start_at(),
            node_off_at: slots.node_off_at(),
            edge_start_at: slots.edge_start_at(),
            edge_off_at: slots.edge_off_at(),
            subset: false,
        },
        (None, Some(row_of)) => RowContext {
            table: IdTensor::empty(vec![1], device),
            row_of: row_of.clone(),
            graph_start_at: 0,
            node_off_at: 0,
            edge_start_at: 0,
            edge_off_at: 0,
            subset: true,
        },
        (None, None) => unreachable!("a batch is laid out from slots or from a node subset"),
    })
}

/// `[rows, d]` node values of a batch, with the vector layout of its lanes.
fn node_lanes<R: Runtime, E: FloatElem>(
    value: &Tensor<R, E>,
    rows: &BatchRows<R>,
    what: &str,
) -> Result<(usize, usize)> {
    if value.rank() != 2 || value.dims()[0] != rows.rows || value.dims()[1] == 0 {
        return Err(Error::shape(format!(
            "{what} must be [{}, d], got {}",
            rows.rows,
            value.shape()
        )));
    }
    let d = value.dims()[1];
    let line = line_dividing::<R, E>(value.client(), &[d]);
    Ok((line, d / line))
}

/// The edge rows of a whole-graph batch, after checking `value` is
/// `[edges, d]` with the node values' width.
fn edge_rows<'a, R: Runtime, E: FloatElem>(
    value: &Tensor<R, E>,
    rows: &'a BatchRows<R>,
    d: usize,
    what: &str,
) -> Result<&'a BatchEdges<R>> {
    let edges = rows.edges.as_ref().ok_or_else(|| {
        Error::config(format!(
            "{what} needs a whole-graph batch laid out with its edges"
        ))
    })?;
    if value.rank() != 2 || value.dims()[0] != edges.edges || value.dims()[1] != d {
        return Err(Error::shape(format!(
            "{what} must be [{}, {d}], got {}",
            edges.edges,
            value.shape()
        )));
    }
    Ok(edges)
}

/// One lane per `(row, vector)`: the row's own value plus the rectified
/// messages of its in-neighbours.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn gine_aggregate_kernel<F: Float + CubeElement, N: Size>(
    u: &Array<Vector<F, N>>,
    ee: &Array<Vector<F, N>>,
    gid: &Array<u32>,
    row_graph: &Array<u32>,
    adj_off: &Array<u32>,
    adj_col: &Array<u32>,
    table: &Array<u32>,
    row_of: &Array<u32>,
    out: &mut Array<Vector<F, N>>,
    graph_start_at: usize,
    node_off_at: usize,
    edge_start_at: usize,
    edge_off_at: usize,
    dvec: usize,
    lanes: usize,
    span: usize,
    ignore: u32,
    #[comptime] subset: bool,
    #[comptime] has_edge: bool,
) {
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    let zero = Vector::<f32, N>::new(0.0_f32);
    for pos in start..end {
        let dv = pos % dvec;
        let row = pos / dvec;
        let node = gid[row];
        let mut acc = Vector::<f32, N>::cast_from(u[pos]);
        if node != ignore {
            let first = adj_off[node as usize] as usize;
            let last = adj_off[node as usize + 1] as usize;
            if comptime!(subset) {
                for e in first..last {
                    let other = row_of[adj_col[e] as usize];
                    if other != ignore {
                        let msg = Vector::<f32, N>::cast_from(u[other as usize * dvec + dv]);
                        acc += select_many(msg.less_than(zero), zero, msg);
                    }
                }
            } else {
                let g = row_graph[row] as usize;
                let node_shift = table[node_off_at + g] - table[graph_start_at + g];
                let edge_shift = table[edge_off_at + g] - table[edge_start_at + g];
                for e in first..last {
                    let other = (adj_col[e] + node_shift) as usize;
                    let mut msg = Vector::<f32, N>::cast_from(u[other * dvec + dv]);
                    if comptime!(has_edge) {
                        let edge = (e as u32 + edge_shift) as usize;
                        msg += Vector::<f32, N>::cast_from(ee[edge * dvec + dv]);
                    }
                    acc += select_many(msg.less_than(zero), zero, msg);
                }
            }
        }
        out[pos] = Vector::<F, N>::cast_from(acc);
    }
}

/// G7, GINE aggregation (`ε = 0`): `a[r] = u[r] + Σ_{e into r} relu(u[src(e)] +
/// ee[e])`, with `ee` the `[edges, d]` edge embeddings of a whole-graph batch
/// laid out with its edges, or `None`. No `[edges, d]` message tensor is
/// formed. An absent row keeps its own value. One launch.
pub fn gine_aggregate<R: Runtime, E: FloatElem>(
    u: &Tensor<R, E>,
    ee: Option<&Tensor<R, E>>,
    adjacency: &Adjacency<R>,
    rows: &BatchRows<R>,
) -> Result<Tensor<R, E>> {
    let _op = crate::backend::tally_op_scope("gine_aggregate");
    let (line, dvec) = node_lanes(u, rows, "gine_aggregate: u")?;
    let ctx = row_context(adjacency, rows)?;
    if let Some(ee) = ee {
        edge_rows(ee, rows, u.dims()[1], "gine_aggregate: ee")?;
    }
    let out = Tensor::<R, E>::empty(u.shape().clone(), u.device());
    let client = u.client();
    let lanes = rows.rows * dvec;
    let degree = adjacency.n_edges / adjacency.n_nodes.max(1) + 1;
    let (cube_count, cube_dim, span) = launch_1d_spans(client, lanes, degree * line);
    let no_edges;
    let ee_arg = match ee {
        Some(ee) => ee,
        None => {
            no_edges = Tensor::<R, E>::empty(vec![16], u.device());
            &no_edges
        }
    };
    // SAFETY: row ids and slots come from G1 over this adjacency (node counts
    // compared in `row_context`), so neighbours shift to rows inside the batch
    // and edges to edge rows inside `ee` (`edge_rows` checked its length
    // against the batch's edge capacity); `row_of` holds rows below `rows` or
    // `IGNORE`.
    unsafe {
        gine_aggregate_kernel::launch_unchecked::<E, R>(
            client,
            cube_count,
            cube_dim,
            line,
            u.arg(),
            ee_arg.arg(),
            rows.gid.arg(),
            rows.row_graph.arg(),
            adjacency.off.arg(),
            adjacency.col.arg(),
            ctx.table.arg(),
            ctx.row_of.arg(),
            out.arg(),
            ctx.graph_start_at,
            ctx.node_off_at,
            ctx.edge_start_at,
            ctx.edge_off_at,
            dvec,
            lanes,
            span,
            IGNORE,
            ctx.subset,
            ee.is_some(),
        );
    }
    Ok(out)
}

/// One lane per `(row, vector)`: the adjoint of [`gine_aggregate_kernel`] with
/// respect to `u`, gathered over the row's in-edges, whose reverses are the
/// edges the row sent a message along.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn gine_du_kernel<F: Float + CubeElement, N: Size>(
    grad: &Array<Vector<F, N>>,
    u: &Array<Vector<F, N>>,
    ee: &Array<Vector<F, N>>,
    gid: &Array<u32>,
    row_graph: &Array<u32>,
    adj_off: &Array<u32>,
    adj_col: &Array<u32>,
    adj_rev: &Array<u32>,
    table: &Array<u32>,
    row_of: &Array<u32>,
    out: &mut Array<Vector<F, N>>,
    graph_start_at: usize,
    node_off_at: usize,
    edge_start_at: usize,
    edge_off_at: usize,
    dvec: usize,
    lanes: usize,
    span: usize,
    ignore: u32,
    #[comptime] subset: bool,
    #[comptime] has_edge: bool,
) {
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    let zero = Vector::<f32, N>::new(0.0_f32);
    for pos in start..end {
        let dv = pos % dvec;
        let row = pos / dvec;
        let node = gid[row];
        let own = Vector::<f32, N>::cast_from(u[pos]);
        let mut sent = zero;
        if node != ignore {
            let first = adj_off[node as usize] as usize;
            let last = adj_off[node as usize + 1] as usize;
            if comptime!(subset) {
                for e in first..last {
                    let other = row_of[adj_col[e] as usize];
                    if other != ignore {
                        sent += Vector::<f32, N>::cast_from(grad[other as usize * dvec + dv]);
                    }
                }
                sent = select_many(own.greater_than(zero), sent, zero);
            } else {
                let g = row_graph[row] as usize;
                let node_shift = table[node_off_at + g] - table[graph_start_at + g];
                let edge_shift = table[edge_off_at + g] - table[edge_start_at + g];
                if comptime!(has_edge) {
                    for e in first..last {
                        let other = (adj_col[e] + node_shift) as usize;
                        let back = (adj_rev[e] + edge_shift) as usize;
                        let pre = own + Vector::<f32, N>::cast_from(ee[back * dvec + dv]);
                        let got = Vector::<f32, N>::cast_from(grad[other * dvec + dv]);
                        sent += select_many(pre.greater_than(zero), got, zero);
                    }
                } else {
                    for e in first..last {
                        let other = (adj_col[e] + node_shift) as usize;
                        sent += Vector::<f32, N>::cast_from(grad[other * dvec + dv]);
                    }
                    sent = select_many(own.greater_than(zero), sent, zero);
                }
            }
        }
        out[pos] = Vector::<F, N>::cast_from(Vector::<f32, N>::cast_from(grad[pos]) + sent);
    }
}

/// Adjoint of [`gine_aggregate`] with respect to `u`:
/// `du[r] = g[r] + Σ_{e into r} g[src(e)] ⊙ [u[r] + ee[rev(e)] > 0]`. Without
/// edge embeddings the test is `[u[r] > 0]` for every edge and the reverse-edge
/// index is not read. One launch.
pub fn gine_aggregate_du<R: Runtime, E: FloatElem>(
    grad: &Tensor<R, E>,
    u: &Tensor<R, E>,
    ee: Option<&Tensor<R, E>>,
    adjacency: &Adjacency<R>,
    rows: &BatchRows<R>,
) -> Result<Tensor<R, E>> {
    let _op = crate::backend::tally_op_scope("gine_aggregate_du");
    if !adjacency.symmetric {
        return Err(Error::config(
            "the adjoint of message passing reads the graph as its own transpose: it needs a \
             symmetric adjacency (symmetrize = true)"
                .to_string(),
        ));
    }
    let (line, dvec) = node_lanes(u, rows, "gine_aggregate_du: u")?;
    if grad.shape() != u.shape() {
        return Err(Error::shape(format!(
            "gine_aggregate_du: the gradient {} does not match u {}",
            grad.shape(),
            u.shape()
        )));
    }
    let ctx = row_context(adjacency, rows)?;
    let device = u.device();
    let (no_edges, no_rev);
    let (ee_arg, rev_arg) = match ee {
        Some(ee) => {
            edge_rows(ee, rows, u.dims()[1], "gine_aggregate_du: ee")?;
            let rev = adjacency.rev.as_ref().ok_or_else(|| {
                Error::config(
                    "edge features in message passing need the store's reverse-edge index"
                        .to_string(),
                )
            })?;
            (ee, rev)
        }
        None => {
            no_edges = Tensor::<R, E>::empty(vec![16], device);
            no_rev = IdTensor::empty(vec![1], device);
            (&no_edges, &no_rev)
        }
    };
    let out = Tensor::<R, E>::empty(u.shape().clone(), device);
    let client = u.client();
    let lanes = rows.rows * dvec;
    let degree = adjacency.n_edges / adjacency.n_nodes.max(1) + 1;
    let (cube_count, cube_dim, span) = launch_1d_spans(client, lanes, degree * line);
    // SAFETY: as `gine_aggregate`; the reverse of an edge lies in the same
    // graph, so it shifts to an edge row of the same slot.
    unsafe {
        gine_du_kernel::launch_unchecked::<E, R>(
            client,
            cube_count,
            cube_dim,
            line,
            grad.arg(),
            u.arg(),
            ee_arg.arg(),
            rows.gid.arg(),
            rows.row_graph.arg(),
            adjacency.off.arg(),
            adjacency.col.arg(),
            rev_arg.arg(),
            ctx.table.arg(),
            ctx.row_of.arg(),
            out.arg(),
            ctx.graph_start_at,
            ctx.node_off_at,
            ctx.edge_start_at,
            ctx.edge_off_at,
            dvec,
            lanes,
            span,
            IGNORE,
            ctx.subset,
            ee.is_some(),
        );
    }
    Ok(out)
}

/// What an edge-lane kernel computes.
///
/// One kernel serves the four per-edge maps of the two layers: each reads the
/// rows of its edge's two ends and writes one `[edges, d]` value, and they
/// differ in two or three lines of arithmetic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
enum EdgeMap {
    /// `g[dst] ⊙ [u[src] + ee > 0]`: GINE's adjoint with respect to `ee`.
    /// `a = g`, `b = u`, `e = ee`.
    GineDee = 0,
    /// `ce + du[dst] + eu[src]`: GatedGCN's edge pre-activation.
    /// `a = du`, `b = eu`, `e = ce`.
    GatedEdge = 1,
    /// `g[dst] ⊙ (b[src] − z[dst]) / D[dst] ⊙ σ(ê)(1 − σ(ê))`: the adjoint of
    /// the gated aggregation with respect to `ê`. `a = g`, `b = b`, `e = ê`,
    /// `c = z`, `d = D`.
    GatedDehat = 2,
    /// `e` itself on the rows that hold an edge, zero on the absent ones: the
    /// adjoint of [`gated_edge`] with respect to `C e`.
    MaskEdge = 3,
}

#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn edge_map_kernel<F: Float + CubeElement, N: Size>(
    a: &Array<Vector<F, N>>,
    b: &Array<Vector<F, N>>,
    c: &Array<Vector<F, N>>,
    d: &Array<Vector<F, N>>,
    e: &Array<Vector<F, N>>,
    edge_src: &Array<u32>,
    edge_dst: &Array<u32>,
    out: &mut Array<Vector<F, N>>,
    dvec: usize,
    lanes: usize,
    span: usize,
    ignore: u32,
    #[comptime] map: u32,
) {
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    let zero = Vector::<f32, N>::new(0.0_f32);
    let one = Vector::<f32, N>::new(1.0_f32);
    for pos in start..end {
        let dv = pos % dvec;
        let edge = pos / dvec;
        let src = edge_src[edge];
        let mut value = zero;
        if src != ignore {
            let s = src as usize * dvec + dv;
            let t = edge_dst[edge] as usize * dvec + dv;
            let on_edge = Vector::<f32, N>::cast_from(e[pos]);
            if comptime!(map == 0) {
                let pre = Vector::<f32, N>::cast_from(b[s]) + on_edge;
                value = select_many(
                    pre.greater_than(zero),
                    Vector::<f32, N>::cast_from(a[t]),
                    zero,
                );
            } else if comptime!(map == 1) {
                value = on_edge
                    + Vector::<f32, N>::cast_from(a[t])
                    + Vector::<f32, N>::cast_from(b[s]);
            } else if comptime!(map == 2) {
                let gate = one / (one + (zero - on_edge).exp());
                value = Vector::<f32, N>::cast_from(a[t])
                    * (Vector::<f32, N>::cast_from(b[s]) - Vector::<f32, N>::cast_from(c[t]))
                    / Vector::<f32, N>::cast_from(d[t])
                    * gate
                    * (one - gate);
            } else {
                // Exact in every element type: 16-bit floats round-trip
                // through f32.
                value = on_edge;
            }
        }
        out[pos] = Vector::<F, N>::cast_from(value);
    }
}

/// The shared launcher of the edge-lane maps: `a`, `b` (and `c`, `d` for the
/// gated adjoint) are `[rows, d]`, `e` is `[edges, d]`.
fn edge_map<R: Runtime, E: FloatElem>(
    map: EdgeMap,
    a: &Tensor<R, E>,
    b: &Tensor<R, E>,
    cd: Option<(&Tensor<R, E>, &Tensor<R, E>)>,
    e: &Tensor<R, E>,
    rows: &BatchRows<R>,
    what: &str,
) -> Result<Tensor<R, E>> {
    let (line, dvec) = node_lanes(a, rows, what)?;
    let width = a.dims()[1];
    for node_value in [Some(b), cd.map(|p| p.0), cd.map(|p| p.1)]
        .into_iter()
        .flatten()
    {
        if node_value.shape() != a.shape() {
            return Err(Error::shape(format!(
                "{what}: node values {} and {} differ",
                a.shape(),
                node_value.shape()
            )));
        }
    }
    let edges = edge_rows(e, rows, width, what)?;
    let out = Tensor::<R, E>::empty(e.shape().clone(), e.device());
    let client = e.client();
    let lanes = edges.edges * dvec;
    let (cube_count, cube_dim, span) = launch_1d_spans(client, lanes, 2 * line);
    let (c, d) = cd.unwrap_or((a, b));
    // Never read by the maps that take no `c` and `d`; bound to buffers of
    // their own so nothing is bound twice.
    let (c_stand_in, d_stand_in);
    let (c_arg, d_arg) = if cd.is_some() {
        (c, d)
    } else {
        c_stand_in = Tensor::<R, E>::empty(vec![16], e.device());
        d_stand_in = Tensor::<R, E>::empty(vec![16], e.device());
        (&c_stand_in, &d_stand_in)
    };
    // SAFETY: edge ends are batch rows below `rows` or `IGNORE` (G1), the node
    // values hold `rows` rows and the edge value `edges` rows (checked above).
    unsafe {
        edge_map_kernel::launch_unchecked::<E, R>(
            client,
            cube_count,
            cube_dim,
            line,
            a.arg(),
            b.arg(),
            c_arg.arg(),
            d_arg.arg(),
            e.arg(),
            edges.src.arg(),
            edges.dst.arg(),
            out.arg(),
            dvec,
            lanes,
            span,
            IGNORE,
            map as u32,
        );
    }
    Ok(out)
}

/// `value` `[edges, d]` with its absent edge rows zeroed: the adjoint of
/// [`gated_edge`] with respect to `C e`, whose forward writes zero there
/// whatever `C e` holds. One launch; a batch whose edge rows are all present
/// ([`BatchRows::edges_full`]) needs none.
pub fn mask_edge_rows<R: Runtime, E: FloatElem>(
    value: &Tensor<R, E>,
    rows: &BatchRows<R>,
) -> Result<Tensor<R, E>> {
    let _op = crate::backend::tally_op_scope("mask_edge_rows");
    if value.rank() != 2 || value.dims()[1] == 0 {
        return Err(Error::shape(format!(
            "mask_edge_rows must be [edges, d], got {}",
            value.shape()
        )));
    }
    let width = value.dims()[1];
    let edges = edge_rows(value, rows, width, "mask_edge_rows")?;
    let line = line_dividing::<R, E>(value.client(), &[width]);
    let dvec = width / line;
    let out = Tensor::<R, E>::empty(value.shape().clone(), value.device());
    let client = value.client();
    let lanes = edges.edges * dvec;
    let (cube_count, cube_dim, span) = launch_1d_spans(client, lanes, line);
    // The node operands of the other maps, never read by this one; buffers of
    // their own so nothing is bound twice.
    let stand_ins: [Tensor<R, E>; 4] =
        core::array::from_fn(|_| Tensor::<R, E>::empty(vec![16], value.device()));
    // SAFETY: the edge value holds `edges` rows (checked above), and this map
    // reads nothing else but the edge ends, which hold `edges` ids.
    unsafe {
        edge_map_kernel::launch_unchecked::<E, R>(
            client,
            cube_count,
            cube_dim,
            line,
            stand_ins[0].arg(),
            stand_ins[1].arg(),
            stand_ins[2].arg(),
            stand_ins[3].arg(),
            value.arg(),
            edges.src.arg(),
            edges.dst.arg(),
            out.arg(),
            dvec,
            lanes,
            span,
            IGNORE,
            EdgeMap::MaskEdge as u32,
        );
    }
    Ok(out)
}

/// Adjoint of [`gine_aggregate`] with respect to the edge embeddings:
/// `dee[e] = g[dst(e)] ⊙ [u[src(e)] + ee[e] > 0]`; absent edge rows give zeros.
/// One launch.
pub fn gine_aggregate_dee<R: Runtime, E: FloatElem>(
    grad: &Tensor<R, E>,
    u: &Tensor<R, E>,
    ee: &Tensor<R, E>,
    rows: &BatchRows<R>,
) -> Result<Tensor<R, E>> {
    let _op = crate::backend::tally_op_scope("gine_aggregate_dee");
    edge_map(EdgeMap::GineDee, grad, u, None, ee, rows, "gine_aggregate_dee")
}

/// G7, GatedGCN edge pre-activation: `ê[e] = ce[e] + du[dst(e)] + eu[src(e)]`
/// for the three products `C e`, `D u` and `E u`; absent edge rows give zeros.
/// One launch.
pub fn gated_edge<R: Runtime, E: FloatElem>(
    ce: &Tensor<R, E>,
    du: &Tensor<R, E>,
    eu: &Tensor<R, E>,
    rows: &BatchRows<R>,
) -> Result<Tensor<R, E>> {
    let _op = crate::backend::tally_op_scope("gated_edge");
    edge_map(EdgeMap::GatedEdge, du, eu, None, ce, rows, "gated_edge")
}

/// What a node-lane kernel over the edge rows of a whole-graph batch computes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
enum NodeMap {
    /// Two sums of an edge value `x`: over the row's in-edges, and over their
    /// reverses (the row's out-edges). The adjoint of [`gated_edge`] with
    /// respect to `D u` and `E u`. `x = dê`.
    EdgeSums = 0,
    /// `z = Σ σ(ê) ⊙ b[src] / D` with `D = Σ σ(ê) + 1e-6` over the row's
    /// in-edges; writes `z` and `D`. `x = ê`, `p = b`.
    GatedNode = 1,
    /// `Σ_{out-edges} g[dst] ⊙ σ(ê) / D[dst]`: the adjoint of the gated
    /// aggregation with respect to `b`. `x = ê`, `p = g`, `q = D`.
    GatedDb = 2,
}

#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn node_map_kernel<F: Float + CubeElement, N: Size>(
    x: &Array<Vector<F, N>>,
    p: &Array<Vector<F, N>>,
    q: &Array<Vector<F, N>>,
    gid: &Array<u32>,
    row_graph: &Array<u32>,
    adj_off: &Array<u32>,
    adj_col: &Array<u32>,
    adj_rev: &Array<u32>,
    table: &Array<u32>,
    out: &mut Array<Vector<F, N>>,
    out2: &mut Array<Vector<F, N>>,
    graph_start_at: usize,
    node_off_at: usize,
    edge_start_at: usize,
    edge_off_at: usize,
    dvec: usize,
    lanes: usize,
    span: usize,
    ignore: u32,
    #[comptime] map: u32,
) {
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    let zero = Vector::<f32, N>::new(0.0_f32);
    let one = Vector::<f32, N>::new(1.0_f32);
    for pos in start..end {
        let dv = pos % dvec;
        let row = pos / dvec;
        let node = gid[row];
        let mut first_sum = zero;
        let mut second_sum = zero;
        if comptime!(map == 1) {
            second_sum = Vector::<f32, N>::new(1.0e-6_f32);
        }
        if node != ignore {
            let first = adj_off[node as usize] as usize;
            let last = adj_off[node as usize + 1] as usize;
            let g = row_graph[row] as usize;
            let node_shift = table[node_off_at + g] - table[graph_start_at + g];
            let edge_shift = table[edge_off_at + g] - table[edge_start_at + g];
            for e in first..last {
                let edge = (e as u32 + edge_shift) as usize * dvec + dv;
                if comptime!(map == 0) {
                    let back = (adj_rev[e] + edge_shift) as usize * dvec + dv;
                    first_sum += Vector::<f32, N>::cast_from(x[edge]);
                    second_sum += Vector::<f32, N>::cast_from(x[back]);
                } else if comptime!(map == 1) {
                    let other = (adj_col[e] + node_shift) as usize * dvec + dv;
                    let gate = one / (one + (zero - Vector::<f32, N>::cast_from(x[edge])).exp());
                    first_sum += gate * Vector::<f32, N>::cast_from(p[other]);
                    second_sum += gate;
                } else {
                    let other = (adj_col[e] + node_shift) as usize * dvec + dv;
                    let back = (adj_rev[e] + edge_shift) as usize * dvec + dv;
                    let gate = one / (one + (zero - Vector::<f32, N>::cast_from(x[back])).exp());
                    first_sum += Vector::<f32, N>::cast_from(p[other]) * gate
                        / Vector::<f32, N>::cast_from(q[other]);
                }
            }
        }
        if comptime!(map == 1) {
            out[pos] = Vector::<F, N>::cast_from(first_sum / second_sum);
            out2[pos] = Vector::<F, N>::cast_from(second_sum);
        } else {
            out[pos] = Vector::<F, N>::cast_from(first_sum);
            if comptime!(map == 0) {
                out2[pos] = Vector::<F, N>::cast_from(second_sum);
            }
        }
    }
}

/// The shared launcher of the node-lane maps. `x` is `[edges, d]`; `p` and `q`
/// are `[rows, d]` where the map takes them. Returns the map's one or two
/// `[rows, d]` outputs.
#[allow(clippy::type_complexity)]
fn node_map<R: Runtime, E: FloatElem>(
    map: NodeMap,
    x: &Tensor<R, E>,
    pq: (Option<&Tensor<R, E>>, Option<&Tensor<R, E>>),
    adjacency: &Adjacency<R>,
    rows: &BatchRows<R>,
    what: &str,
) -> Result<(Tensor<R, E>, Option<Tensor<R, E>>)> {
    let ctx = row_context(adjacency, rows)?;
    if ctx.subset {
        return Err(Error::config(format!(
            "{what} needs a whole-graph batch laid out with its edges"
        )));
    }
    if x.rank() != 2 {
        return Err(Error::shape(format!(
            "{what}: the edge value must be [edges, d], got {}",
            x.shape()
        )));
    }
    let d = x.dims()[1];
    edge_rows(x, rows, d, what)?;
    let device = x.device();
    let shape = Shape::new(vec![rows.rows, d]);
    for node_value in [pq.0, pq.1].into_iter().flatten() {
        if node_value.shape() != &shape {
            return Err(Error::shape(format!(
                "{what}: node values must be {shape}, got {}",
                node_value.shape()
            )));
        }
    }
    let needs_rev = map != NodeMap::GatedNode;
    let no_rev;
    let rev = match (&adjacency.rev, needs_rev) {
        (Some(rev), true) => rev,
        (None, true) => {
            return Err(Error::config(format!(
                "{what} needs the store's reverse-edge index"
            )));
        }
        (_, false) => {
            no_rev = IdTensor::empty(vec![1], device);
            &no_rev
        }
    };
    let client = x.client();
    let line = line_dividing::<R, E>(client, &[d]);
    let dvec = d / line;
    let lanes = rows.rows * dvec;
    let degree = adjacency.n_edges / adjacency.n_nodes.max(1) + 1;
    let (cube_count, cube_dim, span) = launch_1d_spans(client, lanes, degree * line);
    let out = Tensor::<R, E>::empty(shape.clone(), device);
    let two = map != NodeMap::GatedDb;
    let out2 = if two {
        Tensor::<R, E>::empty(shape, device)
    } else {
        Tensor::<R, E>::empty(vec![16], device)
    };
    let (p_stand_in, q_stand_in);
    let p_arg = match pq.0 {
        Some(p) => p,
        None => {
            p_stand_in = Tensor::<R, E>::empty(vec![16], device);
            &p_stand_in
        }
    };
    let q_arg = match pq.1 {
        Some(q) => q,
        None => {
            q_stand_in = Tensor::<R, E>::empty(vec![16], device);
            &q_stand_in
        }
    };
    // SAFETY: as `gine_aggregate`: rows, slots and edge rows come from G1 over
    // this adjacency, the edge value holds the batch's edge capacity and the
    // node values `rows` rows (checked above), and an edge's reverse lies in
    // the same slot.
    unsafe {
        node_map_kernel::launch_unchecked::<E, R>(
            client,
            cube_count,
            cube_dim,
            line,
            x.arg(),
            p_arg.arg(),
            q_arg.arg(),
            rows.gid.arg(),
            rows.row_graph.arg(),
            adjacency.off.arg(),
            adjacency.col.arg(),
            rev.arg(),
            ctx.table.arg(),
            out.arg(),
            out2.arg(),
            ctx.graph_start_at,
            ctx.node_off_at,
            ctx.edge_start_at,
            ctx.edge_off_at,
            dvec,
            lanes,
            span,
            IGNORE,
            map as u32,
        );
    }
    Ok((out, two.then_some(out2)))
}

/// Adjoint of [`gated_edge`] with respect to its two node operands:
/// `d(D u)[i] = Σ_{e into i} dê[e]` and `d(E u)[j] = Σ_{e out of j} dê[e]`,
/// the second through the reverse-edge index. (Its adjoint with respect to
/// `C e` is `dê` itself.) One launch.
pub fn gated_edge_backward<R: Runtime, E: FloatElem>(
    dehat: &Tensor<R, E>,
    adjacency: &Adjacency<R>,
    rows: &BatchRows<R>,
) -> Result<(Tensor<R, E>, Tensor<R, E>)> {
    let _op = crate::backend::tally_op_scope("gated_edge_backward");
    let (d_du, d_eu) = node_map(
        NodeMap::EdgeSums,
        dehat,
        (None, None),
        adjacency,
        rows,
        "gated_edge_backward",
    )?;
    Ok((d_du, d_eu.expect("the edge sums are two outputs")))
}

/// G7, GatedGCN aggregation: `z[i] = Σ_{j → i} σ(ê_ij) ⊙ b[j] / D[i]` with
/// `D[i] = Σ_{j → i} σ(ê_ij) + 1e-6`, numerator and denominator accumulated in
/// registers. Returns `z` and `D`, both `[rows, d]`. One launch.
pub fn gated_node<R: Runtime, E: FloatElem>(
    ehat: &Tensor<R, E>,
    b: &Tensor<R, E>,
    adjacency: &Adjacency<R>,
    rows: &BatchRows<R>,
) -> Result<(Tensor<R, E>, Tensor<R, E>)> {
    let _op = crate::backend::tally_op_scope("gated_node");
    let (z, den) = node_map(
        NodeMap::GatedNode,
        ehat,
        (Some(b), None),
        adjacency,
        rows,
        "gated_node",
    )?;
    Ok((z, den.expect("the gated aggregation writes its denominator")))
}

/// Adjoint of [`gated_node`] with respect to `ê`:
/// `dê_ij = g[i] ⊙ (b[j] − z[i]) / D[i] ⊙ σ(ê_ij)(1 − σ(ê_ij))`. One launch.
pub fn gated_node_dehat<R: Runtime, E: FloatElem>(
    grad: &Tensor<R, E>,
    b: &Tensor<R, E>,
    z: &Tensor<R, E>,
    den: &Tensor<R, E>,
    ehat: &Tensor<R, E>,
    rows: &BatchRows<R>,
) -> Result<Tensor<R, E>> {
    let _op = crate::backend::tally_op_scope("gated_node_dehat");
    edge_map(
        EdgeMap::GatedDehat,
        grad,
        b,
        Some((z, den)),
        ehat,
        rows,
        "gated_node_dehat",
    )
}

/// Adjoint of [`gated_node`] with respect to `b`:
/// `db[j] = Σ_{i: j → i} g[i] ⊙ σ(ê_ij) / D[i]`, over `j`'s out-edges through
/// the reverse-edge index. One launch.
pub fn gated_node_db<R: Runtime, E: FloatElem>(
    grad: &Tensor<R, E>,
    den: &Tensor<R, E>,
    ehat: &Tensor<R, E>,
    adjacency: &Adjacency<R>,
    rows: &BatchRows<R>,
) -> Result<Tensor<R, E>> {
    let _op = crate::backend::tally_op_scope("gated_node_db");
    Ok(node_map(
        NodeMap::GatedDb,
        ehat,
        (Some(grad), Some(den)),
        adjacency,
        rows,
        "gated_node_db",
    )?
    .0)
}
