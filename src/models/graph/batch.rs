//! Datasets, epochs and batches (GRAPH_MAMBA_PLAN.md §2.5).
//!
//! A [`GraphDataset`] is a spec and the device store built for it: canonical
//! order, validation and the one upload happen in [`GraphDataset::new`]. An
//! [`Epoch`] is a plan for one pass over a split, and a [`GraphBatch`] is a
//! **descriptor** — shapes worked out on the host, contents written by one
//! kernel:
//!
//! * [`GraphDataset::epoch_graphs`] cuts many small graphs into batches filled
//!   to a row budget. The epoch's slot table is its one upload.
//! * [`GraphDataset::epoch_nodes`] cuts one large graph into stratified node
//!   parts. It uploads nothing: a batch is `(parts, part, epoch seed)`.
//!
//! Batches carry **capacities**, not sizes: the row capacity is the same for
//! every batch of a run, the edge capacity the same within an epoch, the graph
//! capacity a multiple of 8 and the padded length a multiple of 32. Rows,
//! edges and graphs beyond what a batch holds are absent, and every kernel
//! that crosses a row boundary skips them. That keeps the shapes of a run —
//! and so the matmul tuner's keys — few.

use std::rc::Rc;

use cubecl::prelude::Runtime;

use crate::backend::{Device, FloatElem};
use crate::error::{Error, Result};
use crate::models::graph::data::{
    CanonicalLabels, CanonicalizeOptions, FeaturesView, GraphData, GraphDataView, LabelsView,
    Split, canonicalize,
};
use crate::models::graph::spec::{
    BatchMode, FeatureSpec, GraphMambaSpec, GraphTaskSpec, MpnnKind, TokenSampling, TokenTail,
};
use crate::models::graph::store::GraphStore;
use crate::models::vision::ScanDirection;
use crate::tensor::ops::IGNORE;
use crate::tensor::ops::graph::{BatchRows, EpochTable, batch_rows_graphs, batch_rows_subset};
use crate::tensor::ops::movement::RaggedLengths;
use crate::tensor::ops::random::Rng;

/// Environment variable bounding the device bytes a dataset and a step may
/// take; 2 GiB when unset.
pub const MAX_BYTES_ENV: &str = "MAMBA3_GRAPH_MAX_BYTES";

/// Environment variable overriding the single-allocation threshold the
/// automatic batch size keeps under.
pub const TENSOR_MAX_ENV: &str = "MAMBA3_GRAPH_TENSOR_MAX_BYTES";

/// Most graphs a size bucket holds.
const BUCKET: usize = 256;

/// Batches' worth of graphs a size bucket holds, when that is fewer.
const BUCKET_BATCHES: usize = 4;

fn env_bytes(name: &str) -> Option<usize> {
    std::env::var(name).ok().and_then(|v| v.trim().parse().ok())
}

/// The bound on a dataset's store plus a step's memory.
pub fn max_bytes() -> usize {
    env_bytes(MAX_BYTES_ENV).unwrap_or(2 << 30)
}

/// The size a single allocation is kept under by the automatic batch size.
///
/// CubeCL's default pool sorts allocations into classes and reserves a page
/// per class; its largest sliced class takes allocations up to an eighth of the
/// device's largest page, and anything above that is given a page of the
/// largest size. That an eighth is the boundary is read off the pool's
/// construction, not exposed by it — an assumption, which
/// [`TENSOR_MAX_ENV`] overrides.
pub fn tensor_threshold<R: Runtime>(device: &Device<R>) -> usize {
    env_bytes(TENSOR_MAX_ENV).unwrap_or_else(|| {
        (device.client().properties().memory.max_page_size / 8).min(usize::MAX as u64) as usize
    })
}

fn round_up(value: usize, multiple: usize) -> usize {
    value.div_ceil(multiple) * multiple
}

/// SplitMix64: one well-mixed word from two.
/// Step counters reserved per epoch: more than any epoch has batches.
const STEP_STRIDE: u64 = 1 << 20;

fn mix(a: u64, b: u64) -> u64 {
    let mut z = a ^ b.wrapping_mul(0x9E3779B97F4A7C15);
    z = z.wrapping_add(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

fn halves(word: u64) -> (u32, u32) {
    (word as u32, (word >> 32) as u32)
}

/// How a dataset is built.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DatasetOptions {
    /// Add the reverse of every edge. Message passing needs it.
    pub symmetrize: bool,
}

impl Default for DatasetOptions {
    fn default() -> Self {
        Self { symmetrize: true }
    }
}

/// How an epoch's whole-graph batches are ordered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EpochOptions {
    /// Shuffle the graphs (per epoch). Off, the order is by size or as stored.
    pub shuffle: bool,
    /// Fill batches from size buckets, which keeps padding short.
    pub bucket: bool,
}

impl EpochOptions {
    /// The default of a split: buckets always, shuffling for training only.
    pub fn for_split(split: Option<Split>) -> Self {
        Self {
            shuffle: split == Some(Split::Train),
            bucket: true,
        }
    }
}

/// What a step is expected to hold on the device, before the first step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryEstimate {
    /// The dataset's tables.
    pub store: usize,
    /// Parameters and the optimizer's two moments.
    pub parameters: usize,
    /// Bytes alive when a step's backward pass starts.
    pub live: usize,
    /// The largest single allocation of a step.
    pub largest_allocation: usize,
    /// Row capacity of the batches the estimate is for.
    pub rows: usize,
    /// The allocation size the automatic batch size keeps under.
    pub threshold: usize,
}

impl MemoryEstimate {
    /// Store, parameters and live bytes together. The pool's reservation is
    /// not estimated: it has measured at three to nine times the live bytes.
    pub fn total(&self) -> usize {
        self.store + self.parameters + self.live
    }
}

/// A dataset on the device, with the spec it was built for.
pub struct GraphDataset<R: Runtime, E: FloatElem> {
    spec: GraphMambaSpec,
    store: Rc<GraphStore<R, E>>,
    /// Split flags per graph for graph-level targets; empty otherwise. The one
    /// label-side table the host keeps: an epoch selects its graphs with it.
    graph_flags: Vec<u32>,
}

/// A dataset validated and put in canonical order on the host, ready for its
/// one upload.
///
/// The two halves of [`GraphDataset::from_view`], apart: [`PreparedDataset::new`]
/// reads the caller's arrays and touches no device, and
/// [`PreparedDataset::upload`] touches the device and none of the caller's
/// arrays. A binding that must hold a lock while foreign memory is read — the
/// Python bindings and the interpreter lock — can release it for the second.
pub struct PreparedDataset<E: FloatElem> {
    spec: GraphMambaSpec,
    canon: crate::models::graph::data::CanonicalGraph<E>,
    graph_flags: Vec<u32>,
}

impl<E: FloatElem> PreparedDataset<E> {
    /// Check `view` against `spec` and write its tables in canonical order.
    /// Every error names the field it is about.
    pub fn new(
        spec: &GraphMambaSpec,
        view: &GraphDataView<'_>,
        options: DatasetOptions,
    ) -> Result<Self> {
        spec.validate()?;
        check_features(&spec.node_features, &view.x, "x")?;
        let mut view = *view;
        match (&spec.edge_features, &view.edge_attr) {
            (Some(edge), Some(attr)) => check_features(edge, attr, "edge_attr")?,
            (Some(_), None) => {
                return Err(Error::config(
                    "edge_attr: the spec has edge features and the data has none".to_string(),
                ));
            }
            // Features the model does not read are not uploaded.
            (None, _) => view.edge_attr = None,
        }
        if view.pe.as_ref().map_or(0, |(dim, _)| *dim) != spec.pe_dim {
            return Err(Error::config(format!(
                "pe: the spec has pe_dim = {} and the data has {} encoding columns",
                spec.pe_dim,
                view.pe.as_ref().map_or(0, |(dim, _)| *dim)
            )));
        }
        let task_ok = match (&spec.task, &view.y) {
            (_, LabelsView::None) => true,
            (GraphTaskSpec::NodeClass { .. }, LabelsView::Node(_)) => true,
            (GraphTaskSpec::GraphClass { .. }, LabelsView::GraphClass(_)) => true,
            (
                GraphTaskSpec::GraphRegression { targets: n, .. }
                | GraphTaskSpec::GraphMultiLabel { labels: n, .. },
                LabelsView::Graph { targets, .. },
            ) => n == targets,
            _ => false,
        };
        if !task_ok {
            return Err(Error::config(format!(
                "y: the targets do not match the spec's task {:?}",
                spec.task
            )));
        }
        if spec.mpnn.is_some() && !options.symmetrize {
            return Err(Error::config(
                "symmetrize: message passing runs on the symmetrised graph only".to_string(),
            ));
        }

        let canon = canonicalize::<E>(
            &view,
            &CanonicalizeOptions {
                order: spec.order,
                symmetrize: options.symmetrize,
                reverse_index: spec.needs_reverse_index(),
            },
        )?;
        match (&spec.task, &canon.y) {
            (
                GraphTaskSpec::NodeClass { classes } | GraphTaskSpec::GraphClass { classes, .. },
                CanonicalLabels::Node(ids) | CanonicalLabels::GraphClass(ids),
            ) => {
                if let Some(bad) = ids.iter().find(|&&c| c != IGNORE && c as usize >= *classes) {
                    return Err(Error::config(format!(
                        "y: class {bad} is outside the task's {classes} classes"
                    )));
                }
            }
            (GraphTaskSpec::GraphMultiLabel { .. }, CanonicalLabels::Graph { values, .. }) => {
                let binary = values.iter().all(|v| {
                    let v = v.to_scalar();
                    v.is_nan() || v == 0.0 || v == 1.0
                });
                if !binary {
                    return Err(Error::config(
                        "y: multi-label targets must be 0, 1 or NaN (missing)".to_string(),
                    ));
                }
            }
            _ => {}
        }
        let graph_flags = match &canon.y {
            CanonicalLabels::GraphClass(_) | CanonicalLabels::Graph { .. } => canon.split.clone(),
            _ => Vec::new(),
        };
        Ok(Self {
            spec: spec.clone(),
            canon,
            graph_flags,
        })
    }

    /// Nodes over all graphs.
    pub fn num_nodes(&self) -> usize {
        self.canon.n_nodes
    }

    /// Number of graphs.
    pub fn num_graphs(&self) -> usize {
        self.canon.n_graphs
    }

    /// Upload every table, by value: the one upload of a dataset. Refused when
    /// the tables would not fit [`max_bytes`].
    pub fn upload<R: Runtime>(self, device: &Device<R>) -> Result<GraphDataset<R, E>> {
        crate::backend::ensure_dtype(device, E::DTYPE)?;
        let bytes = GraphStore::<R, E>::estimate_bytes(&self.canon);
        if bytes > max_bytes() {
            return Err(Error::config(format!(
                "the dataset takes {bytes} bytes on the device, above the limit of {} \
                 ({MAX_BYTES_ENV}); use a subset of it, a 16-bit dtype, or raise the limit",
                max_bytes()
            )));
        }
        Ok(GraphDataset {
            spec: self.spec,
            store: Rc::new(GraphStore::upload(self.canon, device)?),
            graph_flags: self.graph_flags,
        })
    }
}

/// A feature table's spec, named for errors.
fn check_features(spec: &FeatureSpec, view: &FeaturesView<'_>, field: &str) -> Result<()> {
    let matches = match (spec, view) {
        (FeatureSpec::Float { dim }, FeaturesView::Float { dim: got, .. }) => dim == got,
        (FeatureSpec::Categorical { vocab }, FeaturesView::Categorical { vocab: got, .. }) => {
            vocab.as_slice() == *got
        }
        _ => false,
    };
    if matches {
        Ok(())
    } else {
        Err(Error::config(format!(
            "{field}: the data does not match the spec's {spec:?}"
        )))
    }
}

impl<R: Runtime, E: FloatElem> GraphDataset<R, E> {
    /// Canonicalise, validate and upload `data` for `spec`: the one upload of
    /// a dataset. The graph is symmetrised.
    pub fn new(spec: &GraphMambaSpec, data: GraphData, device: &Device<R>) -> Result<Self> {
        Self::from_view(spec, &data.view(), DatasetOptions::default(), device)
    }

    /// [`GraphDataset::new`] from borrowed, typed slices, read where they lie.
    pub fn from_view(
        spec: &GraphMambaSpec,
        view: &GraphDataView<'_>,
        options: DatasetOptions,
        device: &Device<R>,
    ) -> Result<Self> {
        PreparedDataset::<E>::new(spec, view, options)?.upload(device)
    }

    /// The spec the dataset was built for.
    pub fn spec(&self) -> &GraphMambaSpec {
        &self.spec
    }

    /// The device tables.
    pub fn store(&self) -> &Rc<GraphStore<R, E>> {
        &self.store
    }

    /// Nodes over all graphs.
    pub fn num_nodes(&self) -> usize {
        self.store.n_nodes()
    }

    /// Number of graphs.
    pub fn num_graphs(&self) -> usize {
        self.store.n_graphs()
    }

    /// Whether the dataset has targets.
    pub fn has_targets(&self) -> bool {
        !matches!(
            self.store.targets(),
            crate::models::graph::store::TargetStore::None
        )
    }

    /// Refuse a split the dataset has no labelled item in.
    fn check_split(&self, split: Option<Split>) -> Result<()> {
        let Some(split) = split else {
            return Ok(());
        };
        if !self.has_targets() {
            return Err(Error::config(format!(
                "the dataset has no targets, so it has no {} split",
                split.name()
            )));
        }
        if self.store.labelled(split) == 0 {
            return Err(Error::config(format!(
                "the {} split has no labelled item",
                split.name()
            )));
        }
        Ok(())
    }

    /// The largest row budget whose largest single allocation stays under the
    /// device's allocation threshold and whose step fits the memory bound; a
    /// multiple of the spec's row quantum.
    pub fn auto_rows(&self) -> usize {
        let elem = core::mem::size_of::<E>();
        let threshold = tensor_threshold(self.store.device());
        let by_allocation = threshold as f64 / self.per_row_peak();
        let fixed = self.store.bytes() + 3 * self.spec.parameter_count() * elem;
        let by_memory = max_bytes().saturating_sub(fixed) as f64
            / live_bytes_per_row(&self.spec, elem, self.avg_degree());
        let quantum = self.spec.row_quantum;
        (by_allocation.min(by_memory) as usize / quantum).max(1) * quantum
    }

    /// Bytes per row of the largest row-proportional allocation of a step.
    fn per_row_peak(&self) -> f64 {
        allocations(&self.spec, core::mem::size_of::<E>(), self.avg_degree())
            .into_iter()
            .map(|(_, bytes)| bytes)
            .fold(1.0, f64::max)
    }

    /// Directed edges per node.
    fn avg_degree(&self) -> f64 {
        self.store.n_edges() as f64 / self.store.n_nodes().max(1) as f64
    }

    /// One epoch of whole-graph batches over `split`, each filled to
    /// `batch_rows` rows (`None`: [`GraphDataset::auto_rows`]). Training
    /// epochs are shuffled per `epoch`; all are cut from size buckets.
    /// `split = None` is every graph, for prediction.
    pub fn epoch_graphs(
        &self,
        batch_rows: Option<usize>,
        epoch: u64,
        split: impl Into<Option<Split>>,
    ) -> Result<Epoch<R, E>> {
        let split = split.into();
        self.epoch_graphs_with(batch_rows, epoch, split, EpochOptions::for_split(split))
    }

    /// [`GraphDataset::epoch_graphs`] with the ordering spelled out.
    pub fn epoch_graphs_with(
        &self,
        batch_rows: Option<usize>,
        epoch: u64,
        split: Option<Split>,
        options: EpochOptions,
    ) -> Result<Epoch<R, E>> {
        self.spec.validate_for(BatchMode::Graphs)?;
        self.check_split(split)?;
        let store = &self.store;
        let ptr = store.graph_ptr();
        let size = |g: u32| (ptr[g as usize + 1] - ptr[g as usize]) as usize;

        // Graph targets select their graphs by split; node targets keep every
        // graph, since a node's context is its whole graph whatever its split.
        let mut graphs: Vec<u32> = (0..store.n_graphs() as u32)
            .filter(|&g| match split {
                Some(split) if !self.graph_flags.is_empty() => {
                    self.graph_flags[g as usize] & split.flag() != 0
                }
                _ => true,
            })
            .collect();
        if graphs.is_empty() {
            return Err(Error::config(format!(
                "no graph in the {} split",
                split.map_or("selected", Split::name)
            )));
        }
        let selected_rows: usize = graphs.iter().map(|&g| size(g)).sum();
        let quantum = self.spec.row_quantum;
        let budget = match batch_rows {
            Some(0) => return Err(Error::config("batch_rows must be positive".to_string())),
            Some(rows) => rows,
            // No point in a budget above what the split holds.
            None => self.auto_rows().min(round_up(selected_rows, quantum)),
        };
        if let Some(&g) = graphs.iter().find(|&&g| size(g) > budget) {
            return Err(Error::config(format!(
                "graph {g} has {} nodes, above the row budget of {budget}; raise batch_rows",
                size(g)
            )));
        }
        // Order: size buckets (sorted, cut into runs, shuffled inside a run and
        // run against run) keep the padded length of a batch near its graphs'.
        // A run is a few batches' worth of graphs, so that a small dataset
        // still has several of them.
        let mut rng = Rng::seeded(mix(self.spec.seed, epoch.wrapping_add(0x51ED)));
        if options.bucket {
            graphs.sort_by_key(|&g| (size(g), g));
            if options.shuffle {
                let per_batch = (budget * graphs.len()).div_ceil(selected_rows.max(1));
                let run = (BUCKET_BATCHES * per_batch).clamp(8, BUCKET);
                let mut runs: Vec<Vec<u32>> = graphs.chunks(run).map(<[u32]>::to_vec).collect();
                for run in &mut runs {
                    rng.shuffle(run);
                }
                rng.shuffle(&mut runs);
                graphs = runs.into_iter().flatten().collect();
            }
        } else if options.shuffle {
            rng.shuffle(&mut graphs);
        }

        let mut batches: Vec<Vec<u32>> = Vec::new();
        let mut current: Vec<u32> = Vec::new();
        let mut used = 0usize;
        for g in graphs {
            if used + size(g) > budget && !current.is_empty() {
                batches.push(core::mem::take(&mut current));
                used = 0;
            }
            used += size(g);
            current.push(g);
        }
        batches.push(current);

        self.epoch_from_batches(batches, budget, epoch, split)
    }

    /// An epoch of whole-graph batches chosen by the caller: each of `batches`
    /// lists the dataset graph ids of one batch, in slot order. Every batch
    /// must fit `batch_rows` rows. This is what
    /// [`GraphDataset::epoch_graphs`] builds its own batches with; use it for
    /// a sampler of your own.
    pub fn epoch_from_batches(
        &self,
        batches: Vec<Vec<u32>>,
        batch_rows: usize,
        epoch: u64,
        split: Option<Split>,
    ) -> Result<Epoch<R, E>> {
        self.spec.validate_for(BatchMode::Graphs)?;
        self.check_split(split)?;
        let store = &self.store;
        let ptr = store.graph_ptr();
        let n_graphs = store.n_graphs();
        if batches.is_empty() || batches.iter().any(Vec::is_empty) {
            return Err(Error::config(
                "an epoch needs at least one batch, and every batch at least one graph"
                    .to_string(),
            ));
        }
        if let Some(&bad) = batches.iter().flatten().find(|&&g| g as usize >= n_graphs) {
            return Err(Error::config(format!(
                "a batch names graph {bad} of a dataset with {n_graphs} graphs"
            )));
        }
        let size = |g: u32| (ptr[g as usize + 1] - ptr[g as usize]) as usize;
        let rows = round_up(batch_rows.max(1), self.spec.row_quantum);
        if let Some(index) = batches
            .iter()
            .position(|batch| batch.iter().map(|&g| size(g)).sum::<usize>() > rows)
        {
            return Err(Error::config(format!(
                "batch {index} holds more than the {rows} rows of a batch"
            )));
        }

        let edge_ptr = store.edge_ptr();
        let graph_cap = round_up(batches.iter().map(Vec::len).max().unwrap_or(1), 8);
        let edges = self.spec.needs_reverse_index().then(|| {
            let most = batches
                .iter()
                .map(|batch| {
                    batch
                        .iter()
                        .map(|&g| (edge_ptr[g as usize + 1] - edge_ptr[g as usize]) as usize)
                        .sum::<usize>()
                })
                .max()
                .unwrap_or(0);
            round_up(most.max(1), 4096)
        });
        let plans = batches
            .iter()
            .map(|batch| {
                let longest = batch.iter().map(|&g| size(g)).max().unwrap_or(1);
                let nmax = round_up(longest.max(1), 32);
                BatchPlan {
                    nmax,
                    all_full: batch.len() == graph_cap && batch.iter().all(|&g| size(g) == nmax),
                }
            })
            .collect();
        self.check_memory(rows)?;
        // The one upload of the epoch.
        let table = EpochTable::build(&batches, ptr, edge_ptr, graph_cap, store.device())?;
        Ok(Epoch {
            store: store.clone(),
            mode: BatchMode::Graphs,
            table: Some(Rc::new(table)),
            batches,
            plans,
            rows,
            edges,
            graphs: graph_cap,
            parts: 0,
            sequences: 1,
            epoch,
            split,
            seed: self.spec.seed,
            threshold: tensor_threshold(store.device()),
        })
    }

    /// The number of parts whose batches stay under the automatic row budget.
    pub fn auto_parts(&self) -> usize {
        self.store.n_nodes().div_ceil(self.auto_rows()).max(1)
    }

    /// One epoch of node partitions of a single large graph: `parts` batches
    /// (`None`: [`GraphDataset::auto_parts`]; 1 is full batch) that partition
    /// the nodes exactly, each in canonical order and stratified along it.
    /// Uploads nothing. `split = None` is for prediction.
    pub fn epoch_nodes(
        &self,
        parts: Option<usize>,
        epoch: u64,
        split: impl Into<Option<Split>>,
    ) -> Result<Epoch<R, E>> {
        let split = split.into();
        self.spec.validate_for(BatchMode::NodeSubset)?;
        self.check_split(split)?;
        let store = &self.store;
        if store.n_graphs() != 1 {
            return Err(Error::config(format!(
                "node partitions cut one large graph, and this dataset has {} graphs; use \
                 epoch_graphs",
                store.n_graphs()
            )));
        }
        let n = store.n_nodes();
        let parts = match parts {
            Some(0) => return Err(Error::config("parts must be positive".to_string())),
            Some(parts) if parts > n => {
                return Err(Error::config(format!(
                    "parts: {parts} parts of a graph with {n} nodes"
                )));
            }
            Some(parts) => parts,
            None => self.auto_parts(),
        };
        let sequences = self.spec.node_sequences;
        let rows = round_up(n.div_ceil(parts), sequences);
        self.check_memory(rows)?;
        Ok(Epoch {
            store: store.clone(),
            mode: BatchMode::NodeSubset,
            table: None,
            batches: Vec::new(),
            plans: Vec::new(),
            rows,
            edges: None,
            graphs: 1,
            parts,
            sequences,
            epoch,
            split,
            seed: self.spec.seed,
            threshold: tensor_threshold(store.device()),
        })
    }

    /// What a step is expected to hold on the device: the store, the
    /// parameters with the optimizer's state, the bytes alive when the
    /// backward pass starts, and the largest single allocation.
    ///
    /// `batch_rows` and `parts` are as for the epoch constructors; with both
    /// `None` the estimate is for whole-graph batches of the automatic size
    /// when the dataset has several graphs, and for automatic node parts
    /// otherwise.
    pub fn memory_estimate(
        &self,
        batch_rows: Option<usize>,
        parts: Option<usize>,
    ) -> MemoryEstimate {
        let n = self.store.n_nodes();
        let rows = match (batch_rows, parts) {
            (Some(rows), _) => round_up(rows.max(1), self.spec.row_quantum),
            (None, Some(parts)) => n.div_ceil(parts.max(1)),
            (None, None) if self.store.n_graphs() > 1 => {
                self.auto_rows().min(round_up(n, self.spec.row_quantum))
            }
            (None, None) => n.div_ceil(self.auto_parts()),
        };
        self.estimate_for(rows)
    }

    fn estimate_for(&self, rows: usize) -> MemoryEstimate {
        let elem = core::mem::size_of::<E>();
        let degree = self.avg_degree();
        let largest = self.per_row_peak() * rows as f64;
        let features = token_feature_bytes(&self.spec, elem) * rows as f64;
        let threshold = tensor_threshold(self.store.device());
        MemoryEstimate {
            store: self.store.bytes(),
            // Weights and AdamW's two moments.
            parameters: 3 * self.spec.parameter_count() * elem,
            live: (live_bytes_per_row(&self.spec, elem, degree) * rows as f64) as usize,
            // Token features above the threshold are produced in chunks.
            largest_allocation: largest.max(features.min(threshold as f64)) as usize,
            rows,
            threshold,
        }
    }

    /// Refuse a batch size whose step would not fit the configured bound.
    fn check_memory(&self, rows: usize) -> Result<()> {
        let estimate = self.estimate_for(rows);
        if estimate.total() > max_bytes() {
            return Err(Error::config(format!(
                "a step of {rows} rows needs about {} bytes on the device (store {}, parameters \
                 {}, live {}), above the limit of {} ({MAX_BYTES_ENV}); use a smaller batch_rows \
                 or more parts, fewer token layers, a 16-bit dtype, or raise the limit",
                estimate.total(),
                estimate.store,
                estimate.parameters,
                estimate.live,
                max_bytes()
            )));
        }
        Ok(())
    }
}

/// The widths of one mixer block as it runs.
struct BlockWidths {
    /// Width of the fused input projection.
    proj: usize,
    /// `n_heads · head_dim`.
    inner: usize,
    /// Channels of the convolution: `x`, `B` and `C`.
    conv: usize,
    /// Scan state per position: `heads · head_dim · d_state`.
    state: usize,
}

fn block_widths(ssm: &crate::ssm::config::SsmConfig, bidirectional: bool) -> BlockWidths {
    let mut cfg = ssm.clone();
    if bidirectional {
        cfg.n_heads *= 2;
        cfg.n_groups *= 2;
    }
    BlockWidths {
        proj: cfg.in_proj_width(),
        inner: cfg.d_inner(),
        conv: cfg.d_inner() + 2 * cfg.bc_width(),
        state: cfg.n_heads * cfg.head_dim * cfg.d_state,
    }
}

/// The per-node stage's blocks, first to last, as `(widths, is_tail)`.
fn token_blocks(spec: &GraphMambaSpec) -> Vec<(BlockWidths, bool)> {
    (0..spec.token_layers)
        .map(|layer| {
            let tail = layer + 1 == spec.token_layers;
            let bidirectional = !tail || spec.token_tail == TokenTail::Bidirectional;
            (block_widths(&spec.token_ssm, bidirectional), tail)
        })
        .collect()
}

/// Padded positions per row a whole-graph batch is assumed to scan in the node
/// stage (buckets keep the measured figure below this).
const PAD_FACTOR: f64 = 1.5;

/// Every row-proportional allocation of a step, as bytes per row: the
/// enumeration the automatic batch size takes its maximum over.
fn allocations(spec: &GraphMambaSpec, elem: usize, avg_degree: f64) -> Vec<(&'static str, f64)> {
    let d = spec.d_model;
    let mut out: Vec<(&'static str, f64)> = Vec::new();
    if spec.has_token_stage() {
        let len = spec.tokens.len();
        out.push(("token tables", (len * spec.tokens.cap() * 4) as f64));
        out.push(("token embedding", (len * d * elem) as f64));
        for (widths, tail) in token_blocks(spec) {
            // The forward tail projects the gate band for the last row only.
            let split = tail && spec.token_tail == TokenTail::Forward;
            let projected = if split {
                widths.proj - widths.inner
            } else {
                widths.proj
            };
            out.push(("token projection", (len * projected * elem) as f64));
            out.push(("token convolution", (len * widths.conv * elem) as f64));
            // One `f32` state checkpoint every eight positions.
            out.push(("scan checkpoints", (len.div_ceil(8) * widths.state * 4) as f64));
        }
    }
    let node = block_widths(&spec.node_ssm, spec.direction == ScanDirection::Bidirectional);
    out.push(("node projection", PAD_FACTOR * (node.proj * elem) as f64));
    out.push(("feed-forward", (2 * d * elem) as f64));
    if spec.needs_reverse_index() {
        out.push(("edge states", avg_degree * (d * elem) as f64));
    }
    out
}

/// Bytes per row of the token features, the one position-sized constant that
/// can be wider than everything else (it is chunked when it is).
fn token_feature_bytes(spec: &GraphMambaSpec, elem: usize) -> f64 {
    if spec.has_token_stage() {
        (spec.tokens.len() * spec.input_width() * elem) as f64
    } else {
        0.0
    }
}

/// Bytes alive per row when the backward pass starts.
///
/// The measured fit for a mixer block is about sixteen bytes per position and
/// projection column in `f32` — four buffers as wide as the projection — to
/// which the constants a step keeps are added: the token tables and features,
/// the scan's checkpoints, and the handful of `d`-wide buffers of the
/// embedding, the message passing, the feed-forward block and the head.
fn live_bytes_per_row(spec: &GraphMambaSpec, elem: usize, avg_degree: f64) -> f64 {
    let d = spec.d_model as f64;
    let e = elem as f64;
    let mut bytes = 0.0;
    if spec.has_token_stage() {
        let len = spec.tokens.len() as f64;
        // Node ids, weights and statistics; then the features.
        bytes += len * (spec.tokens.cap() as f64 * 8.0 + 12.0);
        bytes += token_feature_bytes(spec, elem);
        // The embedding, the statistics' projection, their sum and the GELU.
        bytes += len * d * e * 6.0;
        for (widths, tail) in token_blocks(spec) {
            let split = tail && spec.token_tail == TokenTail::Forward;
            let projected = if split {
                widths.proj - widths.inner
            } else {
                widths.proj
            };
            bytes += 4.0 * len * projected as f64 * e;
            bytes += (spec.tokens.len().div_ceil(8) * widths.state * 4) as f64;
        }
    }
    let node = block_widths(&spec.node_ssm, spec.direction == ScanDirection::Bidirectional);
    let per_layer = 4.0 * PAD_FACTOR * node.proj as f64 * e
        // Padding in and out, the residual sums, the norm, the feed-forward
        // block's two products and its activation.
        + 12.0 * d * e
        + match spec.mpnn {
            None => 0.0,
            Some(MpnnKind::Gine) => 5.0 * d * e,
            Some(MpnnKind::GatedGcn) => (8.0 + 6.0 * avg_degree) * d * e,
        };
    bytes += spec.node_layers as f64 * per_layer;
    bytes += 4.0 * d * e;
    bytes
}

/// What the host knows about one whole-graph batch.
#[derive(Debug, Clone, Copy)]
struct BatchPlan {
    /// The padded length: the longest graph, rounded up to a multiple of 32.
    nmax: usize,
    /// Whether every slot is used by a graph of exactly `nmax` nodes.
    all_full: bool,
}

/// One pass over a split: the plan its batches are cut from.
pub struct Epoch<R: Runtime, E: FloatElem> {
    store: Rc<GraphStore<R, E>>,
    mode: BatchMode,
    table: Option<Rc<EpochTable<R>>>,
    batches: Vec<Vec<u32>>,
    plans: Vec<BatchPlan>,
    rows: usize,
    edges: Option<usize>,
    graphs: usize,
    parts: usize,
    sequences: usize,
    epoch: u64,
    split: Option<Split>,
    seed: u64,
    threshold: usize,
}

impl<R: Runtime, E: FloatElem> Epoch<R, E> {
    /// Number of batches.
    pub fn len(&self) -> usize {
        match self.mode {
            BatchMode::Graphs => self.batches.len(),
            BatchMode::NodeSubset => self.parts,
        }
    }

    /// Whether the epoch has no batch.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// How the batches are cut.
    pub fn mode(&self) -> BatchMode {
        self.mode
    }

    /// Row capacity of every batch.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Edge capacity of every batch, when batches are laid out with edges.
    pub fn edges(&self) -> Option<usize> {
        self.edges
    }

    /// Graph capacity of every batch.
    pub fn graphs(&self) -> usize {
        self.graphs
    }

    /// The split the epoch runs over; `None` is every item.
    pub fn split(&self) -> Option<Split> {
        self.split
    }

    /// The epoch number.
    pub fn number(&self) -> u64 {
        self.epoch
    }

    /// The dataset's device tables.
    pub fn store(&self) -> &Rc<GraphStore<R, E>> {
        &self.store
    }

    /// The dataset graph ids of batch `index`, in slot order (whole-graph
    /// batches).
    pub fn batch_graphs(&self, index: usize) -> &[u32] {
        match self.mode {
            BatchMode::Graphs => &self.batches[index],
            BatchMode::NodeSubset => &[0],
        }
    }

    /// The padded length of batch `index`.
    pub fn padded_len(&self, index: usize) -> usize {
        match self.mode {
            BatchMode::Graphs => self.plans[index].nmax,
            BatchMode::NodeSubset => self.rows / self.sequences,
        }
    }

    /// Batch `index`: one launch, no upload, no read.
    pub fn batch(&self, index: usize) -> Result<GraphBatch<R, E>> {
        if index >= self.len() {
            return Err(Error::shape(format!(
                "batch {index} of an epoch with {} batches",
                self.len()
            )));
        }
        let adjacency = self.store.adjacency();
        let epoch_seed = mix(self.seed, self.epoch);
        let (layout, max, all_full) = match self.mode {
            BatchMode::Graphs => {
                let table = self.table.as_ref().expect("whole-graph epochs have a table");
                let layout =
                    batch_rows_graphs(adjacency, &table.slots(index)?, self.rows, self.edges)?;
                let plan = self.plans[index];
                (layout, plan.nmax, plan.all_full)
            }
            BatchMode::NodeSubset => {
                let layout = batch_rows_subset(
                    adjacency,
                    self.parts,
                    index,
                    halves(epoch_seed),
                    self.sequences,
                )?;
                let n = self.store.n_nodes();
                let blocks = n.div_ceil(self.parts);
                // Only a last block that reaches past the end can pick an
                // absent node; rounding rows are absent by construction.
                let all_full =
                    n.is_multiple_of(self.parts) && blocks.is_multiple_of(self.sequences);
                (layout, self.rows / self.sequences, all_full)
            }
        };
        let lengths = RaggedLengths::new(layout.lengths().clone(), max, all_full)?;
        // Distinct for every (epoch, batch): the number of batches can differ
        // from one shuffled epoch to the next, so it is not the stride.
        let position = self
            .epoch
            .wrapping_mul(STEP_STRIDE)
            .wrapping_add(index as u64);
        Ok(GraphBatch {
            store: self.store.clone(),
            mode: self.mode,
            rows: self.rows,
            edges: self.edges.unwrap_or(0),
            graphs: self.graphs,
            layout,
            lengths,
            split: self.split,
            epoch: self.epoch,
            index,
            token_seed: halves(mix(self.seed, 0x70CE)),
            step_counter: (position as u32).wrapping_add(1),
            epoch_counter: (self.epoch as u32).wrapping_add(1),
            sign_seed: halves(mix(epoch_seed, index as u64 ^ 0x5167)),
            threshold: self.threshold,
        })
    }
}

/// One batch: a descriptor of which rows it holds and how they are laid out.
pub struct GraphBatch<R: Runtime, E: FloatElem> {
    /// The dataset the batch indexes; the model reads its tables through here.
    pub store: Rc<GraphStore<R, E>>,
    /// How the batch was cut.
    pub mode: BatchMode,
    /// Row capacity; trailing rows are absent.
    pub rows: usize,
    /// Edge capacity; 0 when the batch was laid out without its edges.
    pub edges: usize,
    /// Graph capacity (whole-graph batches).
    pub graphs: usize,
    /// The rows: dataset node of each, slot of each, and the edge rows.
    pub layout: BatchRows<R>,
    /// True length of every graph slot or node sequence, with the padded
    /// length.
    pub lengths: RaggedLengths<R>,
    /// The split whose targets the loss reads; `None` for prediction.
    pub split: Option<Split>,
    /// The epoch the batch belongs to.
    pub epoch: u64,
    /// The batch's index in its epoch.
    pub index: usize,
    /// Seed of the token walks: a property of the dataset, not of the batch.
    pub token_seed: (u32, u32),
    /// Token counter of [`TokenSampling::PerStep`]: unique per epoch and batch.
    pub step_counter: u32,
    /// Token counter of [`TokenSampling::PerEpoch`].
    pub epoch_counter: u32,
    /// Seed of the encoding's sign flips for this batch.
    pub sign_seed: (u32, u32),
    /// The allocation size position-sized constants are chunked under.
    pub threshold: usize,
}

impl<R: Runtime, E: FloatElem> GraphBatch<R, E> {
    /// The counter the tokens are sampled with: fixed when not training or
    /// with [`TokenSampling::Static`], otherwise the step or the epoch.
    pub fn token_counter(&self, sampling: TokenSampling, training: bool) -> u32 {
        match (training, sampling) {
            (false, _) | (_, TokenSampling::Static) => 0,
            (true, TokenSampling::PerEpoch) => self.epoch_counter,
            (true, TokenSampling::PerStep) => self.step_counter,
        }
    }

    /// Row ranges `(first, count)` to produce `bytes_per_row`-wide constants
    /// in, so that no chunk is a single allocation above the threshold. Equal
    /// chunks, so a run sees at most two chunk sizes.
    pub fn row_chunks(&self, bytes_per_row: usize) -> Vec<(usize, usize)> {
        let total = self.rows * bytes_per_row;
        let chunks = total.div_ceil(self.threshold.max(1)).max(1);
        let per_chunk = self.rows.div_ceil(chunks);
        (0..self.rows)
            .step_by(per_chunk.max(1))
            .map(|first| (first, per_chunk.min(self.rows - first)))
            .collect()
    }
}
