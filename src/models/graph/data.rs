//! Host graph data and its canonical form (GRAPH_MAMBA_PLAN.md §2.3, GM1).
//!
//! Two shapes of the same thing:
//!
//! * [`GraphData`] owns plain vectors and is what a Rust caller builds by hand.
//! * [`GraphDataView`] borrows typed slices of whatever width the caller has
//!   them in (`f64` features, `i64` edges, …) and is what everything below
//!   consumes. A `GraphData` is a view over its own vectors.
//!
//! [`canonicalize`] turns a view into a [`CanonicalGraph`]: nodes of every graph
//! sorted by the ordering key, the edges as one CSR with sorted rows, every
//! table written **once**, already in canonical order and in the element type
//! the device store wants. After this, storage order is sequence order — the
//! node stage of the model never permutes on the device — and the tables go to
//! the device by value.
//!
//! All of it runs once, on the host, when a dataset is built.

use crate::backend::FloatElem;
use crate::error::{Error, Result};
use crate::tensor::ops::IGNORE;

/// Split flag of a training item, as stored in the device split tables.
pub const SPLIT_TRAIN: u32 = 1;
/// Split flag of a validation item.
pub const SPLIT_VAL: u32 = 2;
/// Split flag of a test item.
pub const SPLIT_TEST: u32 = 4;

/// One of the three splits of a dataset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Split {
    /// Items the loss is computed on.
    Train,
    /// Items held out for model selection.
    Val,
    /// Items held out for the final number.
    Test,
}

impl Split {
    /// The split's bit in the device split tables.
    pub const fn flag(self) -> u32 {
        match self {
            Split::Train => SPLIT_TRAIN,
            Split::Val => SPLIT_VAL,
            Split::Test => SPLIT_TEST,
        }
    }

    /// The split's index into a `[train, val, test]` array.
    pub const fn index(self) -> usize {
        match self {
            Split::Train => 0,
            Split::Val => 1,
            Split::Test => 2,
        }
    }

    /// The name used in errors and in the Python API.
    pub const fn name(self) -> &'static str {
        match self {
            Split::Train => "train",
            Split::Val => "val",
            Split::Test => "test",
        }
    }
}

/// An error that names the input field it is about.
fn field_err(field: &str, msg: impl core::fmt::Display) -> Error {
    Error::config(format!("{field}: {msg}"))
}

// ---------------------------------------------------------------------------
// Typed borrowed slices
// ---------------------------------------------------------------------------

/// A host integer type an id array may arrive in.
pub trait IdSource: Copy + core::fmt::Display {
    /// The value as a `u32`, or `None` when it is negative or does not fit.
    fn id(self) -> Option<u32>;
    /// Whether the value is below zero.
    fn negative(self) -> bool;
}

macro_rules! id_source {
    (signed: $($t:ty),*) => {$(
        impl IdSource for $t {
            #[inline]
            fn id(self) -> Option<u32> {
                u32::try_from(self).ok()
            }
            #[inline]
            fn negative(self) -> bool {
                self < 0
            }
        }
    )*};
    (unsigned: $($t:ty),*) => {$(
        impl IdSource for $t {
            #[inline]
            fn id(self) -> Option<u32> {
                u32::try_from(self).ok()
            }
            #[inline]
            fn negative(self) -> bool {
                false
            }
        }
    )*};
}
id_source!(signed: i8, i16, i32, i64);
id_source!(unsigned: u8, u16, u32, u64);

/// A host float type a feature array may arrive in.
pub trait FloatSource: Copy {
    /// The value as `f32`.
    fn to_f32(self) -> f32;
}

impl FloatSource for f32 {
    #[inline]
    fn to_f32(self) -> f32 {
        self
    }
}

impl FloatSource for f64 {
    #[inline]
    fn to_f32(self) -> f32 {
        self as f32
    }
}

impl FloatSource for half::f16 {
    #[inline]
    fn to_f32(self) -> f32 {
        half::f16::to_f32(self)
    }
}

/// A borrowed integer array of any width.
#[derive(Debug, Clone, Copy)]
pub enum Ints<'a> {
    /// `int8`.
    I8(&'a [i8]),
    /// `int16`.
    I16(&'a [i16]),
    /// `int32`.
    I32(&'a [i32]),
    /// `int64`.
    I64(&'a [i64]),
    /// `uint8`.
    U8(&'a [u8]),
    /// `uint16`.
    U16(&'a [u16]),
    /// `uint32`.
    U32(&'a [u32]),
    /// `uint64`.
    U64(&'a [u64]),
}

/// Run `$body` with `$s` bound to the typed slice inside an [`Ints`].
macro_rules! each_int {
    ($value:expr, $s:ident => $body:expr) => {
        match $value {
            Ints::I8($s) => $body,
            Ints::I16($s) => $body,
            Ints::I32($s) => $body,
            Ints::I64($s) => $body,
            Ints::U8($s) => $body,
            Ints::U16($s) => $body,
            Ints::U32($s) => $body,
            Ints::U64($s) => $body,
        }
    };
}

impl Ints<'_> {
    /// Number of integers.
    pub fn len(&self) -> usize {
        each_int!(self, s => s.len())
    }

    /// Whether the array is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A borrowed float array of any width.
#[derive(Debug, Clone, Copy)]
pub enum Floats<'a> {
    /// `float16`.
    F16(&'a [half::f16]),
    /// `float32`.
    F32(&'a [f32]),
    /// `float64`.
    F64(&'a [f64]),
}

/// Run `$body` with `$s` bound to the typed slice inside a [`Floats`].
macro_rules! each_float {
    ($value:expr, $s:ident => $body:expr) => {
        match $value {
            Floats::F16($s) => $body,
            Floats::F32($s) => $body,
            Floats::F64($s) => $body,
        }
    };
}

impl Floats<'_> {
    /// Number of floats.
    pub fn len(&self) -> usize {
        each_float!(self, s => s.len())
    }

    /// Whether the array is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A borrowed mask.
#[derive(Debug, Clone, Copy)]
pub enum Bools<'a> {
    /// `bool`.
    Bool(&'a [bool]),
    /// Bytes: non-zero is `true`.
    U8(&'a [u8]),
}

impl Bools<'_> {
    /// Number of flags.
    pub fn len(&self) -> usize {
        match self {
            Bools::Bool(s) => s.len(),
            Bools::U8(s) => s.len(),
        }
    }

    /// Whether the mask is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    fn get(&self, i: usize) -> bool {
        match self {
            Bools::Bool(s) => s[i],
            Bools::U8(s) => s[i] != 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Owned data
// ---------------------------------------------------------------------------

/// Features of a node or an edge.
#[derive(Debug, Clone, PartialEq)]
pub enum Features {
    /// `dim` floats per row, row-major.
    Float {
        /// Floats per row.
        dim: usize,
        /// `[rows, dim]`.
        data: Vec<f32>,
    },
    /// `fields` ids per row, row-major; field `f` takes values in `0..vocab[f]`.
    Categorical {
        /// Id fields per row.
        fields: usize,
        /// Vocabulary size of each field.
        vocab: Vec<usize>,
        /// `[rows, fields]`.
        ids: Vec<u32>,
    },
}

/// Node features.
pub type NodeFeatures = Features;
/// Edge features.
pub type EdgeFeatures = Features;

/// Targets of a dataset.
#[derive(Debug, Clone, PartialEq)]
pub enum Labels {
    /// One class per node; a negative value marks an unlabelled node.
    Node(Vec<i64>),
    /// `targets` floats per graph, row-major; `NaN` marks a missing target.
    Graph {
        /// Targets per graph.
        targets: usize,
        /// `[graphs, targets]`.
        values: Vec<f32>,
    },
    /// One class per graph; [`IGNORE`] marks an unlabelled graph.
    GraphClass(Vec<u32>),
    /// No targets (prediction only).
    None,
}

/// Train / validation / test masks: one flag per node for node targets, one per
/// graph for graph targets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Splits {
    /// Training items.
    pub train: Vec<bool>,
    /// Validation items.
    pub val: Vec<bool>,
    /// Test items.
    pub test: Vec<bool>,
}

/// One or many graphs on the host, as plain vectors.
#[derive(Debug, Clone, PartialEq)]
pub struct GraphData {
    /// Nodes over all graphs.
    pub n_nodes: usize,
    /// Source of each directed edge.
    pub edge_src: Vec<u32>,
    /// Destination of each directed edge.
    pub edge_dst: Vec<u32>,
    /// Node features.
    pub x: NodeFeatures,
    /// Optional edge features, one row per directed edge.
    pub edge_attr: Option<EdgeFeatures>,
    /// Optional positional / structural encoding: its width and `[n_nodes, width]`.
    pub pe: Option<(usize, Vec<f32>)>,
    /// Targets.
    pub y: Labels,
    /// `[n_graphs + 1]` node offsets; a single graph is `[0, n_nodes]`.
    pub graph_ptr: Vec<u32>,
    /// Train / validation / test masks.
    pub masks: Option<Splits>,
}

impl GraphData {
    /// A single graph with the given edges and node features and nothing else.
    pub fn new(n_nodes: usize, edge_src: Vec<u32>, edge_dst: Vec<u32>, x: NodeFeatures) -> Self {
        Self {
            n_nodes,
            edge_src,
            edge_dst,
            x,
            edge_attr: None,
            pe: None,
            y: Labels::None,
            graph_ptr: vec![0, n_nodes as u32],
            masks: None,
        }
    }

    /// Borrow as the form [`canonicalize`] and the encodings consume.
    pub fn view(&self) -> GraphDataView<'_> {
        GraphDataView {
            n_nodes: self.n_nodes,
            edge_src: Ints::U32(&self.edge_src),
            edge_dst: Ints::U32(&self.edge_dst),
            x: self.x.view(),
            edge_attr: self.edge_attr.as_ref().map(Features::view),
            pe: self
                .pe
                .as_ref()
                .map(|(dim, data)| (*dim, Floats::F32(data))),
            y: match &self.y {
                Labels::Node(ids) => LabelsView::Node(Ints::I64(ids)),
                Labels::Graph { targets, values } => LabelsView::Graph {
                    targets: *targets,
                    values: Floats::F32(values),
                },
                Labels::GraphClass(ids) => LabelsView::GraphClass(Ints::U32(ids)),
                Labels::None => LabelsView::None,
            },
            graph_ptr: Some(Ints::U32(&self.graph_ptr)),
            masks: self.masks.as_ref().map(|m| SplitsView {
                train: Bools::Bool(&m.train),
                val: Bools::Bool(&m.val),
                test: Bools::Bool(&m.test),
            }),
        }
    }

    /// Check every field against the others, with an error that names the field.
    pub fn validate(&self) -> Result<()> {
        canonicalize::<f32>(
            &self.view(),
            &CanonicalizeOptions {
                order: NodeOrder::Given,
                symmetrize: false,
                reverse_index: false,
            },
        )
        .map(|_| ())
    }

    /// Add the reverse of every edge and remove duplicates.
    ///
    /// Of duplicate edges the first keeps its features; a reverse edge that was
    /// not in the input takes the features of the edge it mirrors. Edges come
    /// out sorted by destination, then source, so applying this twice changes
    /// nothing.
    pub fn symmetrize(&mut self) -> Result<()> {
        let rows = match &self.edge_attr {
            Some(Features::Float { dim, data }) => Some(data.len() / (*dim).max(1)),
            Some(Features::Categorical { fields, ids, .. }) => Some(ids.len() / (*fields).max(1)),
            None => None,
        };
        if rows.is_some_and(|rows| rows != self.edge_src.len()) {
            return Err(field_err(
                "edge_attr",
                format!(
                    "{} rows for {} edges",
                    rows.unwrap_or(0),
                    self.edge_src.len()
                ),
            ));
        }
        let csr = HostCsr::from_view(&self.view(), true)?;
        let nnz = csr.col.len();
        let mut dst = Vec::with_capacity(nnz);
        for u in 0..self.n_nodes {
            dst.extend(std::iter::repeat_n(u as u32, csr.degree(u)));
        }
        self.edge_attr = match self.edge_attr.take() {
            Some(Features::Float { dim, data }) => Some(Features::Float {
                dim,
                data: csr
                    .src_edge
                    .iter()
                    .flat_map(|&e| data[e as usize * dim..(e as usize + 1) * dim].iter().copied())
                    .collect(),
            }),
            Some(Features::Categorical { fields, vocab, ids }) => Some(Features::Categorical {
                fields,
                vocab,
                ids: csr
                    .src_edge
                    .iter()
                    .flat_map(|&e| {
                        ids[e as usize * fields..(e as usize + 1) * fields]
                            .iter()
                            .copied()
                    })
                    .collect(),
            }),
            None => None,
        };
        self.edge_src = csr.col;
        self.edge_dst = dst;
        Ok(())
    }

    /// Sort the nodes of every graph by `order` and relabel everything.
    ///
    /// Returns the relabelled data and the permutation `new → original`. The
    /// edges are taken as given (call [`GraphData::symmetrize`] first for an
    /// undirected graph); duplicates are removed. This is the owned convenience
    /// form of [`canonicalize`], which a dataset uses directly.
    pub fn canonicalize(&self, order: NodeOrder) -> Result<(GraphData, Vec<u32>)> {
        let canon = canonicalize::<f32>(
            &self.view(),
            &CanonicalizeOptions {
                order,
                symmetrize: false,
                reverse_index: false,
            },
        )?;
        let n = canon.n_nodes;
        let mut edge_dst = Vec::with_capacity(canon.adj_col.len());
        for u in 0..n {
            let deg = (canon.adj_off[u + 1] - canon.adj_off[u]) as usize;
            edge_dst.extend(std::iter::repeat_n(u as u32, deg));
        }
        let features = |table: FeatureTable<f32>, vocab: Option<Vec<usize>>| match table {
            FeatureTable::Float { dim, data } => Features::Float { dim, data },
            FeatureTable::Categorical { field_offset, ids } => Features::Categorical {
                fields: field_offset.len() - 1,
                vocab: vocab.unwrap_or_default(),
                ids,
            },
        };
        let vocab_of = |f: &Features| match f {
            Features::Categorical { vocab, .. } => Some(vocab.clone()),
            Features::Float { .. } => None,
        };
        let split_mask = |flag: u32| canon.split.iter().map(|s| s & flag != 0).collect();
        let data = GraphData {
            n_nodes: n,
            edge_src: canon.adj_col,
            edge_dst,
            x: features(canon.x, vocab_of(&self.x)),
            edge_attr: canon
                .edge_x
                .map(|t| features(t, self.edge_attr.as_ref().and_then(vocab_of))),
            pe: canon.pe,
            y: match canon.y {
                CanonicalLabels::Node(ids) => Labels::Node(
                    ids.into_iter()
                        .map(|c| if c == IGNORE { -1 } else { c as i64 })
                        .collect(),
                ),
                CanonicalLabels::GraphClass(ids) => Labels::GraphClass(ids),
                CanonicalLabels::Graph { targets, values } => Labels::Graph { targets, values },
                CanonicalLabels::None => Labels::None,
            },
            graph_ptr: canon.graph_ptr,
            masks: self.masks.as_ref().map(|_| Splits {
                train: split_mask(SPLIT_TRAIN),
                val: split_mask(SPLIT_VAL),
                test: split_mask(SPLIT_TEST),
            }),
        };
        Ok((data, canon.perm))
    }
}

impl Features {
    /// Borrow as a [`FeaturesView`].
    pub fn view(&self) -> FeaturesView<'_> {
        match self {
            Features::Float { dim, data } => FeaturesView::Float {
                dim: *dim,
                data: Floats::F32(data),
            },
            Features::Categorical { fields, vocab, ids } => FeaturesView::Categorical {
                fields: *fields,
                vocab,
                ids: Ints::U32(ids),
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Borrowed data
// ---------------------------------------------------------------------------

/// Borrowed [`Features`].
#[derive(Debug, Clone, Copy)]
pub enum FeaturesView<'a> {
    /// `dim` floats per row.
    Float {
        /// Floats per row.
        dim: usize,
        /// `[rows, dim]`, row-major.
        data: Floats<'a>,
    },
    /// `fields` ids per row.
    Categorical {
        /// Id fields per row.
        fields: usize,
        /// Vocabulary size of each field.
        vocab: &'a [usize],
        /// `[rows, fields]`, row-major.
        ids: Ints<'a>,
    },
}

/// Borrowed [`Labels`].
#[derive(Debug, Clone, Copy)]
pub enum LabelsView<'a> {
    /// One class per node; negative marks an unlabelled node.
    Node(Ints<'a>),
    /// `targets` floats per graph; `NaN` marks a missing target.
    Graph {
        /// Targets per graph.
        targets: usize,
        /// `[graphs, targets]`, row-major.
        values: Floats<'a>,
    },
    /// One class per graph; negative or [`IGNORE`] marks an unlabelled graph.
    GraphClass(Ints<'a>),
    /// No targets.
    None,
}

/// Borrowed [`Splits`].
#[derive(Debug, Clone, Copy)]
pub struct SplitsView<'a> {
    /// Training items.
    pub train: Bools<'a>,
    /// Validation items.
    pub val: Bools<'a>,
    /// Test items.
    pub test: Bools<'a>,
}

/// One or many graphs as typed borrowed slices: the form [`canonicalize`]
/// consumes, so that data arriving as `f64` or `i64` is read once, where it
/// lies, and converted on the way into the canonical tables.
#[derive(Debug, Clone, Copy)]
pub struct GraphDataView<'a> {
    /// Nodes over all graphs.
    pub n_nodes: usize,
    /// Source of each directed edge.
    pub edge_src: Ints<'a>,
    /// Destination of each directed edge.
    pub edge_dst: Ints<'a>,
    /// Node features.
    pub x: FeaturesView<'a>,
    /// Optional edge features, one row per directed edge.
    pub edge_attr: Option<FeaturesView<'a>>,
    /// Optional positional / structural encoding: its width and `[n_nodes, width]`.
    pub pe: Option<(usize, Floats<'a>)>,
    /// Targets.
    pub y: LabelsView<'a>,
    /// `[n_graphs + 1]` node offsets; `None` is a single graph.
    pub graph_ptr: Option<Ints<'a>>,
    /// Train / validation / test masks.
    pub masks: Option<SplitsView<'a>>,
}

// ---------------------------------------------------------------------------
// The host CSR
// ---------------------------------------------------------------------------

/// A graph's edges as compressed sparse rows on the host, in the caller's node
/// numbering: row `u` lists, ascending and without duplicates, the sources of
/// the edges into `u`. Of a symmetrised graph that is simply `u`'s neighbours.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostCsr {
    /// `[n + 1]` row offsets.
    pub off: Vec<u32>,
    /// `[nnz]` sources.
    pub col: Vec<u32>,
    /// `[nnz]` index of the input edge each entry came from (the edge whose
    /// features it carries).
    pub src_edge: Vec<u32>,
}

impl HostCsr {
    /// Build from validated `u32` edge lists.
    ///
    /// With `symmetrize`, the reverse of every edge is added. A self-loop is
    /// kept once: its reverse is itself. Of duplicates the first in input order
    /// is kept, and an edge that was given wins over a reverse that was added.
    pub fn build(n: usize, src: &[u32], dst: &[u32], symmetrize: bool) -> Result<Self> {
        debug_assert_eq!(src.len(), dst.len());
        let mut start = vec![0usize; n + 1];
        for (&s, &d) in src.iter().zip(dst) {
            start[d as usize + 1] += 1;
            if symmetrize && s != d {
                start[s as usize + 1] += 1;
            }
        }
        for u in 0..n {
            start[u + 1] += start[u];
        }
        let total = start[n];
        if total >= u32::MAX as usize {
            return Err(field_err(
                "edge_index",
                format!("{total} directed edges do not fit a u32 index"),
            ));
        }

        // One key per entry, ordered the way duplicates are resolved: by source,
        // then given-before-added, then input position.
        const ADDED: u64 = 1 << 32;
        let mut keys = vec![0u64; total];
        let mut fill = start.clone();
        for (i, (&s, &d)) in src.iter().zip(dst).enumerate() {
            keys[fill[d as usize]] = ((s as u64) << 33) | i as u64;
            fill[d as usize] += 1;
            if symmetrize && s != d {
                keys[fill[s as usize]] = ((d as u64) << 33) | ADDED | i as u64;
                fill[s as usize] += 1;
            }
        }
        drop(fill);

        let mut off = Vec::with_capacity(n + 1);
        let mut col = Vec::with_capacity(total);
        let mut src_edge = Vec::with_capacity(total);
        off.push(0u32);
        for u in 0..n {
            let row = &mut keys[start[u]..start[u + 1]];
            row.sort_unstable();
            let mut last = u64::MAX;
            for &key in row.iter() {
                let c = key >> 33;
                if c != last {
                    col.push(c as u32);
                    src_edge.push(key as u32);
                    last = c;
                }
            }
            off.push(col.len() as u32);
        }
        Ok(Self { off, col, src_edge })
    }

    /// Build from a view, validating the edges (and `graph_ptr`) on the way.
    pub fn from_view(view: &GraphDataView<'_>, symmetrize: bool) -> Result<Self> {
        let graph_ptr = graph_ptr_of(view)?;
        let (src, dst) = edges_of(view, &graph_ptr)?;
        Self::build(view.n_nodes, &src, &dst, symmetrize)
    }

    /// Number of nodes.
    pub fn n_nodes(&self) -> usize {
        self.off.len() - 1
    }

    /// Row `u`: the sources of the edges into `u`, ascending.
    #[inline]
    pub fn row(&self, u: usize) -> &[u32] {
        &self.col[self.off[u] as usize..self.off[u + 1] as usize]
    }

    /// Length of row `u`.
    #[inline]
    pub fn degree(&self, u: usize) -> usize {
        (self.off[u + 1] - self.off[u]) as usize
    }
}

/// `graph_ptr` as validated `u32` offsets (`[0, n_nodes]` when absent).
fn graph_ptr_of(view: &GraphDataView<'_>) -> Result<Vec<u32>> {
    let n = view.n_nodes;
    if n == 0 {
        return Err(field_err("n_nodes", "a dataset needs at least one node"));
    }
    // The CSR builder packs a node id beside an edge index in one 64-bit sort
    // key, with 31 bits for the node.
    if n > 1 << 31 {
        return Err(field_err(
            "n_nodes",
            format!("{n} nodes: at most 2^31 are supported"),
        ));
    }
    let Some(ptr) = view.graph_ptr else {
        return Ok(vec![0, n as u32]);
    };
    fn convert<T: IdSource>(src: &[T]) -> Result<Vec<u32>> {
        src.iter()
            .enumerate()
            .map(|(i, v)| {
                v.id().ok_or_else(|| {
                    field_err(
                        "graph_ptr",
                        format!("offset {v} at position {i} is not a valid node offset"),
                    )
                })
            })
            .collect()
    }
    let ptr = each_int!(ptr, s => convert(s))?;
    if ptr.len() < 2 {
        return Err(field_err(
            "graph_ptr",
            "needs at least two offsets, [0, n_nodes] for a single graph",
        ));
    }
    if ptr[0] != 0 {
        return Err(field_err(
            "graph_ptr",
            format!("must start at 0, starts at {}", ptr[0]),
        ));
    }
    if let Some(i) = ptr.windows(2).position(|w| w[1] < w[0]) {
        return Err(field_err(
            "graph_ptr",
            format!(
                "must be non-decreasing, but offset {} at position {} follows {}",
                ptr[i + 1],
                i + 1,
                ptr[i]
            ),
        ));
    }
    if *ptr.last().expect("non-empty") as usize != n {
        return Err(field_err(
            "graph_ptr",
            format!(
                "must end at n_nodes = {n}, ends at {}",
                ptr.last().expect("non-empty")
            ),
        ));
    }
    Ok(ptr)
}

/// The edges as validated `u32` pairs: in range, and within one graph.
fn edges_of(view: &GraphDataView<'_>, graph_ptr: &[u32]) -> Result<(Vec<u32>, Vec<u32>)> {
    let n = view.n_nodes;
    if view.edge_src.len() != view.edge_dst.len() {
        return Err(field_err(
            "edge_index",
            format!(
                "{} sources but {} destinations",
                view.edge_src.len(),
                view.edge_dst.len()
            ),
        ));
    }
    fn convert<T: IdSource>(src: &[T], n: usize) -> Result<Vec<u32>> {
        let mut out = Vec::with_capacity(src.len());
        for (i, v) in src.iter().enumerate() {
            match v.id() {
                Some(id) if (id as usize) < n => out.push(id),
                _ if v.negative() => {
                    return Err(field_err(
                        "edge_index",
                        format!("negative node id {v} at position {i}"),
                    ));
                }
                _ => {
                    return Err(field_err(
                        "edge_index",
                        format!("node id {v} at position {i} is outside 0..{n}"),
                    ));
                }
            }
        }
        Ok(out)
    }
    let src = each_int!(view.edge_src, s => convert(s, n))?;
    let dst = each_int!(view.edge_dst, s => convert(s, n))?;

    if graph_ptr.len() > 2 {
        let mut graph_of = vec![0u32; n];
        for g in 0..graph_ptr.len() - 1 {
            graph_of[graph_ptr[g] as usize..graph_ptr[g + 1] as usize].fill(g as u32);
        }
        for (i, (&s, &d)) in src.iter().zip(&dst).enumerate() {
            let (gs, gd) = (graph_of[s as usize], graph_of[d as usize]);
            if gs != gd {
                return Err(field_err(
                    "edge_index",
                    format!("edge {i} ({s} -> {d}) joins graph {gs} and graph {gd}"),
                ));
            }
        }
    }
    Ok((src, dst))
}

// ---------------------------------------------------------------------------
// Node orders
// ---------------------------------------------------------------------------

/// The key the nodes of each graph are sorted by. Ties keep the original order.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum NodeOrder {
    /// By degree; ascending puts the important nodes late, where a causal scan
    /// has seen the most.
    Degree {
        /// Largest degree first.
        descending: bool,
    },
    /// By personalised PageRank with uniform teleport (power iteration),
    /// ascending.
    Ppr {
        /// Teleport probability.
        alpha: f32,
        /// Power iterations.
        iters: usize,
    },
    /// By core number, ascending.
    KCore,
    /// The order the nodes were given in.
    Given,
}

impl Default for NodeOrder {
    fn default() -> Self {
        NodeOrder::Degree { descending: false }
    }
}

impl NodeOrder {
    /// PageRank order with the usual constants: teleport 0.15, 50 iterations.
    pub const fn ppr() -> Self {
        NodeOrder::Ppr {
            alpha: 0.15,
            iters: 50,
        }
    }
}

/// Core number of every node (Batagelj–Zaversnik bucket algorithm, linear in
/// nodes plus edges). Self-loops do not count towards a degree.
pub fn core_numbers(csr: &HostCsr) -> Vec<u32> {
    let n = csr.n_nodes();
    let mut deg: Vec<u32> = (0..n)
        .map(|u| csr.row(u).iter().filter(|&&c| c as usize != u).count() as u32)
        .collect();
    let max_deg = deg.iter().copied().max().unwrap_or(0) as usize;

    // Vertices bucketed by degree: `bin[d]` is where degree `d` starts in `vert`.
    let mut bin = vec![0usize; max_deg + 2];
    for &d in &deg {
        bin[d as usize] += 1;
    }
    let mut start = 0;
    for slot in bin.iter_mut() {
        let count = *slot;
        *slot = start;
        start += count;
    }
    let mut pos = vec![0usize; n];
    let mut vert = vec![0usize; n];
    for v in 0..n {
        pos[v] = bin[deg[v] as usize];
        vert[pos[v]] = v;
        bin[deg[v] as usize] += 1;
    }
    for d in (1..=max_deg + 1).rev() {
        bin[d] = bin[d - 1];
    }
    bin[0] = 0;

    for i in 0..n {
        let v = vert[i];
        for &u in csr.row(v) {
            let u = u as usize;
            if u == v || deg[u] <= deg[v] {
                continue;
            }
            // Move `u` to the front of its bucket and shrink the bucket by one,
            // which is what lowering its degree by one means.
            let du = deg[u] as usize;
            let (pu, pw) = (pos[u], bin[du]);
            let w = vert[pw];
            if u != w {
                pos[u] = pw;
                vert[pu] = w;
                pos[w] = pu;
                vert[pw] = u;
            }
            bin[du] += 1;
            deg[u] -= 1;
        }
    }
    deg
}

/// PageRank with uniform teleport, per graph, by power iteration.
pub fn pagerank(csr: &HostCsr, graph_ptr: &[u32], alpha: f32, iters: usize) -> Vec<f64> {
    let n = csr.n_nodes();
    let alpha = alpha as f64;
    // Row `u` lists the sources of the edges into `u`; a source spreads its
    // mass over the edges *out* of it.
    let mut out_deg = vec![0u32; n];
    for &c in &csr.col {
        out_deg[c as usize] += 1;
    }
    let mut rank = vec![0.0f64; n];
    let mut next = vec![0.0f64; n];
    for g in 0..graph_ptr.len() - 1 {
        let (a, b) = (graph_ptr[g] as usize, graph_ptr[g + 1] as usize);
        if a == b {
            continue;
        }
        let size = (b - a) as f64;
        rank[a..b].fill(1.0 / size);
        for _ in 0..iters {
            let dangling: f64 = (a..b).filter(|&u| out_deg[u] == 0).map(|u| rank[u]).sum();
            for u in a..b {
                let pulled: f64 = csr
                    .row(u)
                    .iter()
                    .map(|&c| rank[c as usize] / out_deg[c as usize] as f64)
                    .sum();
                next[u] = alpha / size + (1.0 - alpha) * (pulled + dangling / size);
            }
            rank[a..b].copy_from_slice(&next[a..b]);
        }
    }
    rank
}

/// The permutation `new → original` that sorts every graph's nodes by `order`.
fn node_permutation(csr: &HostCsr, graph_ptr: &[u32], order: NodeOrder) -> Vec<u32> {
    let n = csr.n_nodes();
    let mut perm: Vec<u32> = (0..n as u32).collect();
    let keys: Vec<f64> = match order {
        NodeOrder::Given => return perm,
        NodeOrder::Degree { descending } => (0..n)
            .map(|u| {
                let d = csr.degree(u) as f64;
                if descending { -d } else { d }
            })
            .collect(),
        NodeOrder::Ppr { alpha, iters } => pagerank(csr, graph_ptr, alpha, iters),
        NodeOrder::KCore => core_numbers(csr).into_iter().map(f64::from).collect(),
    };
    for g in 0..graph_ptr.len() - 1 {
        // Stable, so ties keep the original order.
        perm[graph_ptr[g] as usize..graph_ptr[g + 1] as usize]
            .sort_by(|&a, &b| keys[a as usize].total_cmp(&keys[b as usize]));
    }
    perm
}

// ---------------------------------------------------------------------------
// Canonical tables
// ---------------------------------------------------------------------------

/// A feature table in canonical row order, ready for the device.
#[derive(Debug, Clone, PartialEq)]
pub enum FeatureTable<E> {
    /// `[rows, dim]` floats in the store's element type.
    Float {
        /// Floats per row.
        dim: usize,
        /// `[rows, dim]`.
        data: Vec<E>,
    },
    /// `[rows, fields]` ids; field `f` owns columns
    /// `field_offset[f]..field_offset[f + 1]` of the multi-hot it expands to.
    Categorical {
        /// `[fields + 1]` prefix sums of the vocabulary sizes.
        field_offset: Vec<u32>,
        /// `[rows, fields]`.
        ids: Vec<u32>,
    },
}

impl<E> FeatureTable<E> {
    /// Width of the model input this table expands to: `dim`, or the multi-hot
    /// width (the sum of the vocabulary sizes).
    pub fn width(&self) -> usize {
        match self {
            FeatureTable::Float { dim, .. } => *dim,
            FeatureTable::Categorical { field_offset, .. } => {
                *field_offset.last().expect("field_offset is never empty") as usize
            }
        }
    }

    /// Whether the table holds ids.
    pub fn is_categorical(&self) -> bool {
        matches!(self, FeatureTable::Categorical { .. })
    }
}

/// Targets in canonical order.
#[derive(Debug, Clone, PartialEq)]
pub enum CanonicalLabels<E> {
    /// `[n_nodes]` classes, [`IGNORE`] where unlabelled.
    Node(Vec<u32>),
    /// `[n_graphs]` classes, [`IGNORE`] where unlabelled.
    GraphClass(Vec<u32>),
    /// `[n_graphs, targets]`, `NaN` where missing.
    Graph {
        /// Targets per graph.
        targets: usize,
        /// `[n_graphs, targets]`.
        values: Vec<E>,
    },
    /// No targets.
    None,
}

/// How [`canonicalize`] builds its tables.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CanonicalizeOptions {
    /// The node order within each graph.
    pub order: NodeOrder,
    /// Add the reverse of every edge.
    pub symmetrize: bool,
    /// Also build `adj_rev` and `adj_dst`; needs a symmetric graph.
    pub reverse_index: bool,
}

impl Default for CanonicalizeOptions {
    fn default() -> Self {
        Self {
            order: NodeOrder::default(),
            symmetrize: true,
            reverse_index: false,
        }
    }
}

/// A dataset in canonical order: what the device store uploads, table by table.
///
/// Nodes of a graph are contiguous and sorted by the ordering key, the edges of
/// a graph are contiguous, and every id is dataset-wide.
#[derive(Debug, Clone, PartialEq)]
pub struct CanonicalGraph<E> {
    /// Nodes over all graphs.
    pub n_nodes: usize,
    /// Number of graphs.
    pub n_graphs: usize,
    /// `[n_graphs + 1]` node offsets.
    pub graph_ptr: Vec<u32>,
    /// `[n_graphs + 1]` edge offsets.
    pub edge_ptr: Vec<u32>,
    /// `[n_nodes]` canonical → original node id.
    pub perm: Vec<u32>,
    /// `[n_nodes + 1]` CSR row offsets.
    pub adj_off: Vec<u32>,
    /// `[n_edges]` source of each edge; row `u` holds the edges into `u`,
    /// ascending.
    pub adj_col: Vec<u32>,
    /// `[n_edges]` position of each edge's reverse, when asked for.
    pub adj_rev: Option<Vec<u32>>,
    /// `[n_edges]` destination of each edge (its row), when asked for.
    pub adj_dst: Option<Vec<u32>>,
    /// Node features.
    pub x: FeatureTable<E>,
    /// Positional / structural encoding: its width and `[n_nodes, width]`.
    pub pe: Option<(usize, Vec<E>)>,
    /// Edge features, one row per CSR entry.
    pub edge_x: Option<FeatureTable<E>>,
    /// Targets.
    pub y: CanonicalLabels<E>,
    /// Split flags ([`SPLIT_TRAIN`] | [`SPLIT_VAL`] | [`SPLIT_TEST`]): one per
    /// node for node targets, one per graph for graph targets, empty without
    /// targets.
    pub split: Vec<u32>,
    /// Labelled items in the train, validation and test split.
    pub labelled: [usize; 3],
    /// Whether the reverse of every edge was added.
    pub symmetric: bool,
}

impl<E> CanonicalGraph<E> {
    /// Number of directed edges.
    pub fn n_edges(&self) -> usize {
        self.adj_col.len()
    }

    /// Nodes of graph `g`.
    pub fn graph_len(&self, g: usize) -> usize {
        (self.graph_ptr[g + 1] - self.graph_ptr[g]) as usize
    }
}

/// Convert `count` rows of `dim` floats, picked by `rows`, into `E`.
fn gather_floats<E: FloatElem, T: FloatSource>(
    src: &[T],
    rows: impl Iterator<Item = usize>,
    count: usize,
    dim: usize,
    field: &str,
    allow_nan: bool,
) -> Result<Vec<E>> {
    let mut out = Vec::with_capacity(count * dim);
    let mut bad = false;
    for row in rows {
        for v in &src[row * dim..(row + 1) * dim] {
            let e = E::from_scalar(v.to_f32());
            let back = e.to_scalar();
            bad |= !(back.is_finite() || (allow_nan && back.is_nan()));
            out.push(e);
        }
    }
    if bad {
        // Found on the slow path, so the pass above stays branch-free.
        let at = out
            .iter()
            .position(|e| {
                let v = e.to_scalar();
                !(v.is_finite() || (allow_nan && v.is_nan()))
            })
            .expect("a bad value was seen");
        return Err(field_err(
            field,
            format!(
                "value at row {}, column {} is not finite in {} (canonical order)",
                at / dim.max(1),
                at % dim.max(1),
                E::DTYPE.name()
            ),
        ));
    }
    Ok(out)
}

/// Convert `count` rows of ids, picked by `rows`, checking each against its
/// field's vocabulary.
fn gather_ids<T: IdSource>(
    src: &[T],
    rows: impl Iterator<Item = usize>,
    count: usize,
    vocab: &[usize],
    field: &str,
) -> Result<Vec<u32>> {
    let fields = vocab.len();
    let mut out = Vec::with_capacity(count * fields);
    for row in rows {
        for (f, v) in src[row * fields..(row + 1) * fields].iter().enumerate() {
            match v.id() {
                Some(id) if (id as usize) < vocab[f] => out.push(id),
                _ => {
                    return Err(field_err(
                        field,
                        format!(
                            "id {v} in row {row}, field {f} is outside 0..{}",
                            vocab[f]
                        ),
                    ));
                }
            }
        }
    }
    Ok(out)
}

/// One feature table: `count` rows picked by `rows`.
fn feature_table<E: FloatElem>(
    features: &FeaturesView<'_>,
    rows: impl Iterator<Item = usize>,
    source_rows: usize,
    count: usize,
    field: &str,
) -> Result<FeatureTable<E>> {
    match features {
        FeaturesView::Float { dim, data } => {
            if data.len() != source_rows * dim {
                return Err(field_err(
                    field,
                    format!(
                        "expected {source_rows} rows of {dim} floats ({} values), got {}",
                        source_rows * dim,
                        data.len()
                    ),
                ));
            }
            let data = each_float!(data, s => gather_floats::<E, _>(s, rows, count, *dim, field, false))?;
            Ok(FeatureTable::Float { dim: *dim, data })
        }
        FeaturesView::Categorical { fields, vocab, ids } => {
            if vocab.len() != *fields {
                return Err(field_err(
                    field,
                    format!("{fields} id fields but {} vocabulary sizes", vocab.len()),
                ));
            }
            if ids.len() != source_rows * fields {
                return Err(field_err(
                    field,
                    format!(
                        "expected {source_rows} rows of {fields} ids ({} values), got {}",
                        source_rows * fields,
                        ids.len()
                    ),
                ));
            }
            let mut field_offset = Vec::with_capacity(fields + 1);
            let mut total = 0usize;
            field_offset.push(0u32);
            for &v in vocab.iter() {
                total += v;
                if total >= IGNORE as usize {
                    return Err(field_err(field, "vocabulary sizes overflow a u32"));
                }
                field_offset.push(total as u32);
            }
            let ids = each_int!(ids, s => gather_ids(s, rows, count, vocab, field))?;
            Ok(FeatureTable::Categorical { field_offset, ids })
        }
    }
}

/// Class labels: negative (and, for unsigned input, [`IGNORE`]) is unlabelled.
fn class_labels<T: IdSource>(
    src: &[T],
    rows: impl Iterator<Item = usize>,
    count: usize,
) -> Result<Vec<u32>> {
    let mut out = Vec::with_capacity(count);
    for row in rows {
        let v = src[row];
        out.push(match v.id() {
            Some(id) => id,
            None if v.negative() => IGNORE,
            None => {
                return Err(field_err(
                    "y",
                    format!("class {v} at position {row} does not fit a u32"),
                ));
            }
        });
    }
    Ok(out)
}

/// Sort the nodes of every graph by the ordering key, relabel everything, and
/// write each table once, in canonical order and in element type `E`.
///
/// One pass over the edges converts, checks and counts them; one more fills the
/// CSR. Every other array is read once, through the permutation, straight into
/// the buffer that will be uploaded. Errors name the field they are about.
pub fn canonicalize<E: FloatElem>(
    view: &GraphDataView<'_>,
    options: &CanonicalizeOptions,
) -> Result<CanonicalGraph<E>> {
    let n = view.n_nodes;
    let graph_ptr = graph_ptr_of(view)?;
    let n_graphs = graph_ptr.len() - 1;
    let (src, dst) = edges_of(view, &graph_ptr)?;
    let input_edges = src.len();
    let csr = HostCsr::build(n, &src, &dst, options.symmetrize)?;
    drop((src, dst));

    let perm = if options.order == NodeOrder::KCore && !options.symmetrize {
        // Core numbers are defined on the undirected graph.
        let (src, dst) = edges_of(view, &graph_ptr)?;
        let undirected = HostCsr::build(n, &src, &dst, true)?;
        node_permutation(&undirected, &graph_ptr, options.order)
    } else {
        node_permutation(&csr, &graph_ptr, options.order)
    };
    let identity = perm.iter().enumerate().all(|(i, &p)| i == p as usize);

    // The CSR again, in canonical numbering. Relabelling reorders a row's
    // sources, so each row is sorted once more.
    let (adj_off, adj_col, src_edge) = if identity {
        (csr.off, csr.col, csr.src_edge)
    } else {
        let mut inv = vec![0u32; n];
        for (new, &old) in perm.iter().enumerate() {
            inv[old as usize] = new as u32;
        }
        let nnz = csr.col.len();
        let mut off = Vec::with_capacity(n + 1);
        let mut col = Vec::with_capacity(nnz);
        let mut edge = Vec::with_capacity(nnz);
        let mut row: Vec<(u32, u32)> = Vec::new();
        off.push(0u32);
        for &old in &perm {
            let (a, b) = (
                csr.off[old as usize] as usize,
                csr.off[old as usize + 1] as usize,
            );
            row.clear();
            row.extend(
                csr.col[a..b]
                    .iter()
                    .zip(&csr.src_edge[a..b])
                    .map(|(&c, &e)| (inv[c as usize], e)),
            );
            row.sort_unstable();
            for &(c, e) in &row {
                col.push(c);
                edge.push(e);
            }
            off.push(col.len() as u32);
        }
        (off, col, edge)
    };
    let n_edges = adj_col.len();
    let edge_ptr: Vec<u32> = graph_ptr.iter().map(|&p| adj_off[p as usize]).collect();

    let (adj_rev, adj_dst) = if options.reverse_index {
        let mut rev = Vec::with_capacity(n_edges);
        let mut dst = Vec::with_capacity(n_edges);
        for u in 0..n {
            for e in adj_off[u] as usize..adj_off[u + 1] as usize {
                let c = adj_col[e] as usize;
                let row = &adj_col[adj_off[c] as usize..adj_off[c + 1] as usize];
                let Ok(at) = row.binary_search(&(u as u32)) else {
                    return Err(field_err(
                        "edge_index",
                        format!(
                            "the edge {} -> {} has no reverse; message passing and edge \
                             features need a symmetric graph (symmetrize = true)",
                            perm[c], perm[u]
                        ),
                    ));
                };
                rev.push(adj_off[c] + at as u32);
                dst.push(u as u32);
            }
        }
        (Some(rev), Some(dst))
    } else {
        (None, None)
    };

    let node_rows = || perm.iter().map(|&p| p as usize);
    let x = feature_table::<E>(&view.x, node_rows(), n, n, "x")?;
    let pe = match &view.pe {
        Some((dim, data)) => {
            if data.len() != n * dim {
                return Err(field_err(
                    "pe",
                    format!(
                        "expected {n} rows of {dim} floats ({} values), got {}",
                        n * dim,
                        data.len()
                    ),
                ));
            }
            let table =
                each_float!(data, s => gather_floats::<E, _>(s, node_rows(), n, *dim, "pe", false))?;
            Some((*dim, table))
        }
        None => None,
    };
    let edge_x = match &view.edge_attr {
        Some(features) => Some(feature_table::<E>(
            features,
            src_edge.iter().map(|&e| e as usize),
            input_edges,
            n_edges,
            "edge_attr",
        )?),
        None => None,
    };
    drop(src_edge);

    let (y, items, permuted) = match &view.y {
        LabelsView::Node(ids) => {
            if ids.len() != n {
                return Err(field_err(
                    "y",
                    format!("expected one class per node ({n}), got {}", ids.len()),
                ));
            }
            let ids = each_int!(ids, s => class_labels(s, node_rows(), n))?;
            (CanonicalLabels::Node(ids), n, true)
        }
        LabelsView::GraphClass(ids) => {
            if ids.len() != n_graphs {
                return Err(field_err(
                    "y",
                    format!(
                        "expected one class per graph ({n_graphs}), got {}",
                        ids.len()
                    ),
                ));
            }
            let ids = each_int!(ids, s => class_labels(s, 0..n_graphs, n_graphs))?;
            (CanonicalLabels::GraphClass(ids), n_graphs, false)
        }
        LabelsView::Graph { targets, values } => {
            if *targets == 0 || values.len() != n_graphs * targets {
                return Err(field_err(
                    "y",
                    format!(
                        "expected {n_graphs} rows of {targets} targets ({} values), got {}",
                        n_graphs * targets,
                        values.len()
                    ),
                ));
            }
            let table = each_float!(values, s => {
                gather_floats::<E, _>(s, 0..n_graphs, n_graphs, *targets, "y", true)
            })?;
            (
                CanonicalLabels::Graph {
                    targets: *targets,
                    values: table,
                },
                n_graphs,
                false,
            )
        }
        LabelsView::None => (CanonicalLabels::None, 0, false),
    };

    if view.masks.is_some() && matches!(view.y, LabelsView::None) {
        return Err(field_err(
            "train_mask",
            "masks select targets, and the data has no y",
        ));
    }
    let split: Vec<u32> = match (&view.masks, items) {
        (_, 0) => Vec::new(),
        // Without masks every item trains.
        (None, items) => vec![SPLIT_TRAIN; items],
        (Some(masks), items) => {
            for (name, mask) in [
                ("train_mask", &masks.train),
                ("val_mask", &masks.val),
                ("test_mask", &masks.test),
            ] {
                if mask.len() != items {
                    return Err(field_err(
                        name,
                        format!("expected {items} flags, got {}", mask.len()),
                    ));
                }
            }
            let flags = |i: usize| {
                (masks.train.get(i) as u32) * SPLIT_TRAIN
                    | (masks.val.get(i) as u32) * SPLIT_VAL
                    | (masks.test.get(i) as u32) * SPLIT_TEST
            };
            if permuted {
                perm.iter().map(|&p| flags(p as usize)).collect()
            } else {
                (0..items).map(flags).collect()
            }
        }
    };

    let mut labelled = [0usize; 3];
    for (i, &flags) in split.iter().enumerate() {
        let present = match &y {
            CanonicalLabels::Node(ids) | CanonicalLabels::GraphClass(ids) => ids[i] != IGNORE,
            CanonicalLabels::Graph { targets, values } => values[i * targets..(i + 1) * targets]
                .iter()
                .any(|v| !v.to_scalar().is_nan()),
            CanonicalLabels::None => false,
        };
        if present {
            for split in [Split::Train, Split::Val, Split::Test] {
                labelled[split.index()] += (flags & split.flag() != 0) as usize;
            }
        }
    }

    Ok(CanonicalGraph {
        n_nodes: n,
        n_graphs,
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
        symmetric: options.symmetrize,
    })
}
