//! Generic host batch (ENTITY_MODEL_PLAN.md G4, composed oracle).
//!
//! [`HostArrays`] is keyed exactly as §1.3; [`EntityBatch::from_host`]
//! validates every key, shape and id range (messages name the key) and builds
//! input tensors, presence tensors, anchor ids, per-head labels, keep-weight
//! tensors (step weights folded in) and per-head host divisors — all on the
//! host, as `PlannerBatch::from_host` does.

use std::collections::{BTreeMap, BTreeSet};

use cubecl::prelude::Runtime;

use crate::backend::{Device, FloatElem};
use crate::error::{Error, Result};
use crate::models::entity::spec::{EntityModelSpec, HeadKind, StepSelection};
use crate::tensor::Tensor;
use crate::tensor::ops::entity_model::IGNORE;
use crate::tensor::ops::index::IdTensor;

/// Everything is named, so one dictionary describes a batch (§1.3). Shapes
/// use `B` (this batch); the model code never sees sample indices.
#[derive(Debug, Clone, Default)]
pub struct HostArrays {
    /// Float arrays keyed as §1.3: `(shape, row-major data)`.
    pub f32s: BTreeMap<String, (Vec<usize>, Vec<f32>)>,
    /// Integer arrays keyed as §1.3: `(shape, row-major data)`, `-1` = IGNORE.
    pub ints: BTreeMap<String, (Vec<usize>, Vec<i64>)>,
}

impl HostArrays {
    /// An empty dictionary.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a float array.
    pub fn insert_f32(&mut self, key: &str, shape: Vec<usize>, data: Vec<f32>) {
        self.f32s.insert(key.to_string(), (shape, data));
    }

    /// Insert an integer array (`-1` = IGNORE).
    pub fn insert_int(&mut self, key: &str, shape: Vec<usize>, data: Vec<i64>) {
        self.ints.insert(key.to_string(), (shape, data));
    }
}

/// Per-head training labels on the device.
pub enum HeadLabels<R: Runtime, E: FloatElem> {
    /// Pointer and categorical heads: `ids` is `[rows]` (`0` where ignored),
    /// `keep` is `[rows]` (step weight where kept, else 0).
    Class {
        /// `[rows]` label ids.
        ids: IdTensor<R>,
        /// `[rows]` keep weights.
        keep: Tensor<R, E>,
        /// Keep sum clamped to ≥ 1.
        div: f32,
    },
    /// Multi-label heads: `targets` is `[rows, labels]`.
    Multi {
        /// `[rows, labels]` 0/1 targets.
        targets: Tensor<R, E>,
        /// `[rows]` keep weights.
        keep: Tensor<R, E>,
        /// Keep sum clamped to ≥ 1.
        div: f32,
    },
    /// Regression heads: `targets` is `[rows, outputs]`.
    Reg {
        /// `[rows, outputs]` targets.
        targets: Tensor<R, E>,
        /// `[rows]` keep weights.
        keep: Tensor<R, E>,
        /// Keep sum clamped to ≥ 1.
        div: f32,
    },
}

/// A dataset split uploaded once (K1). After this, a training step moves no
/// data from the host: [`EntityBatch::from_ids`] gathers the `[B]` samples'
/// rows on the device. Per-sample keep sums stay on the host
/// ([`EntityDataset::sample_weights`]), so batch divisors need no read.
pub struct EntityDataset<R: Runtime, E: FloatElem> {
    /// Samples in this split.
    pub samples: usize,
    /// `[S, F]` concatenated float rows (features, presence, keeps,
    /// targets, legal masks).
    pub floats: Tensor<R, E>,
    /// `[S, I]` concatenated id rows (anchors, class ids, choice ids).
    pub ids: IdTensor<R>,
    /// Section layout of the two tables.
    pub layout: DatasetLayout,
    /// `[S * H]` host per-sample keep sums in spec-head order.
    pub sample_weights: Vec<f32>,
    /// Phantom element type.
    pub _elem: std::marker::PhantomData<E>,
}

/// Section layout of an [`EntityDataset`]'s tables.
#[derive(Debug, Clone)]
pub struct DatasetLayout {
    /// `(name, offset, width)` float sections.
    pub float_sections: Vec<(String, usize, usize)>,
    /// `(name, offset, width)` id sections.
    pub id_sections: Vec<(String, usize, usize)>,
    /// Per-head label metadata in spec order.
    pub heads: Vec<DatasetHead>,
    /// Total float row width.
    pub n_float: usize,
    /// Total id row width.
    pub n_id: usize,
    /// Legal-mask reshape (dims without the batch axis) per pointer head.
    pub legal_shapes: BTreeMap<String, Vec<usize>>,
}

/// Per-head label metadata for dataset rebuilds.
#[derive(Debug, Clone)]
pub struct DatasetHead {
    /// Head name.
    pub name: String,
    /// Label kind: class (pointer/categorical), multi, or regression.
    pub kind: DatasetHeadKind,
    /// Label rows per sample (`M` for `First`, `M*K` else).
    pub rows_ps: usize,
    /// Target width (classes/labels/outputs; `0` for class heads' targets
    /// since ids ride the id table).
    pub width: usize,
    /// Whether this head had labels at upload.
    pub labelled: bool,
}

/// Label kinds stored in an [`EntityDataset`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DatasetHeadKind {
    /// Pointer and categorical heads.
    Class,
    /// Multi-label heads.
    Multi,
    /// Regression heads.
    Reg,
}

/// Segment-table row kinds for [`seg_loss_rows`](crate::tensor::ops::entity_model::seg_loss_rows).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegKind {
    /// Cross-entropy over `width` logits (pointer, categorical).
    Class,
    /// BCE averaged over `width` logits (multilabel).
    Multi,
    /// MSE averaged over `width` outputs (regression).
    Reg,
}

/// Static segment layout for a spec's all-steps heads: one table drives the
/// fused row kernel (K3). `First` heads stay on the composed path.
#[derive(Debug, Clone)]
pub struct SegLayout {
    /// All-steps head names, spec order.
    pub heads: Vec<String>,
    /// One `[kind, width, logit_src, logit_off, ft_off]` row per head:
    /// `logit_src` 0 = conditioned shared output, 1 = unconditioned shared
    /// output, 2 = packed pointer logits.
    pub seg: Vec<[u32; 5]>,
    /// Packed float-target row width (Σ head widths).
    pub wtot: usize,
    /// Packed pointer-logit row width (Σ pointer head widths).
    pub wptr: usize,
}

/// Build the segment layout for `spec`: all-steps heads in spec order over
/// the two shared head Linears plus the packed pointer table. Returns `None`
/// when no head takes all steps.
pub fn seg_layout(spec: &EntityModelSpec) -> Option<SegLayout> {
    // Column assignment must match EntityModel's shared Linears (each side
    // packs its heads in spec order) and the packed pointer table (pointer
    // heads in spec order).
    let mut offs = [0usize; 3];
    let mut heads = Vec::new();
    let mut seg = Vec::new();
    let mut wtot = 0usize;
    for h in &spec.heads {
        if !matches!(h.steps, StepSelection::All) {
            continue;
        }
        let width = h.width(spec)?;
        let kind = match &h.kind {
            HeadKind::Pointer { .. } | HeadKind::Categorical { .. } => SegKind::Class,
            HeadKind::MultiLabel { .. } => SegKind::Multi,
            HeadKind::Regression { .. } => SegKind::Reg,
        };
        let src = match &h.kind {
            HeadKind::Pointer { .. } => 2,
            _ if h.condition_on.is_some() => 0,
            _ => 1,
        };
        let off = offs[src];
        offs[src] += width;
        heads.push(h.name.clone());
        seg.push([
            match kind {
                SegKind::Class => 0,
                SegKind::Multi => 1,
                SegKind::Reg => 2,
            },
            width as u32,
            src as u32,
            off as u32,
            wtot as u32,
        ]);
        wtot += width;
    }
    if heads.is_empty() {
        return None;
    }
    Some(SegLayout {
        heads,
        seg,
        wtot,
        wptr: offs[2],
    })
}

/// Per-batch fused-loss inputs (K3): packed label tables over `[R, ·]`
/// (`R = B*M*K`) plus the uploaded segment table. Built once per batch (not
/// per step); `First` heads are not packed.
pub struct SegData<R: Runtime, E: FloatElem> {
    /// All-steps head names, spec order.
    pub heads: Vec<String>,
    /// `[R, H]` class ids (`0` where N/A or ignored).
    pub class_ids: IdTensor<R>,
    /// `[R, H]` keep weights (step weights folded in).
    pub keep: Tensor<R, E>,
    /// `[R, Wtot]` packed float targets (`0` where N/A).
    pub ft: Tensor<R, E>,
    /// `[H]` host divisors (keep sums clamped ≥ 1).
    pub divs: Vec<f32>,
    /// `[H]` host loss weights.
    pub wts: Vec<f32>,
    /// `[H]` device coefficients (`loss_weight / div`).
    pub coef: Tensor<R, E>,
    /// `[Hs, 5]` uploaded segment table.
    pub seg_dev: IdTensor<R>,
    /// `[H]` device inverse widths (`1 / width`).
    pub inv_width: Tensor<R, E>,
}

/// A device batch with every keep mask and divisor precomputed on the host.
pub struct EntityBatch<R: Runtime, E: FloatElem> {
    /// Batch size.
    pub b: usize,
    /// Context features, one `[B, N_s, F_s]` tensor per set (spec order).
    pub ctx_feats: Vec<Tensor<R, E>>,
    /// Context presence, one `[B, N_s]` tensor per set (spec order).
    pub ctx_presence: Vec<Tensor<R, E>>,
    /// Globals `[B, G]` (`None` when `G = 0`).
    pub globals: Option<Tensor<R, E>>,
    /// Query features `[B, M, F]` (`None` without a query set).
    pub q_feats: Option<Tensor<R, E>>,
    /// Query presence `[B, M]` (`None` without a query set).
    pub q_presence: Option<Tensor<R, E>>,
    /// Anchor ids `[B*M]` as global context indices (`IGNORE` = no anchor;
    /// `None` without anchors).
    pub anchor_ids: Option<Vec<u32>>,
    /// Extra action masking per pointer head: `[B, N+E]` or `[B, M, N+E]`
    /// 0/1 floats (absent = no extra masking; presence still applies).
    pub legal: BTreeMap<String, Tensor<R, E>>,
    /// Per-head labels (`None` entries for heads without `label.<name>`).
    pub labels: BTreeMap<String, Option<HeadLabels<R, E>>>,
    /// Host choice ids per pointer head: `[B*M*K]` (`IGNORE` = no label),
    /// for teacher-forced tokens.
    pub choice_ids: BTreeMap<String, Vec<u32>>,
    /// Device choice ids per pointer head: `[rows]` (`IGNORE` preserved),
    /// for the fused path (no host upload).
    pub choice_dev: BTreeMap<String, IdTensor<R>>,
    /// Device anchor ids `[B*M]` (`IGNORE` = no anchor; `None` without
    /// anchors).
    pub anchor_dev: Option<IdTensor<R>>,
    /// Whether this batch was gathered from an [`EntityDataset`] (no host
    /// id maps; the composed path refuses these batches).
    pub resident: bool,
    /// Fused-loss inputs (`None` without all-steps heads).
    pub seg: Option<SegData<R, E>>,
}

fn get_f32(
    a: &HostArrays,
    key: &str,
    seen: &mut BTreeSet<String>,
) -> Result<Option<(Vec<usize>, Vec<f32>)>> {
    seen.insert(key.to_string());
    Ok(a.f32s.get(key).cloned())
}

fn need_f32(
    a: &HostArrays,
    key: &str,
    seen: &mut BTreeSet<String>,
) -> Result<(Vec<usize>, Vec<f32>)> {
    get_f32(a, key, seen)?
        .ok_or_else(|| Error::shape(format!("entity batch is missing required key {key:?}")))
}

fn get_int(
    a: &HostArrays,
    key: &str,
    seen: &mut BTreeSet<String>,
) -> Result<Option<(Vec<usize>, Vec<i64>)>> {
    seen.insert(key.to_string());
    Ok(a.ints.get(key).cloned())
}

fn check_len(key: &str, shape: &[usize], data_len: usize) -> Result<usize> {
    let want: usize = shape.iter().product();
    if data_len != want {
        return Err(Error::shape(format!(
            "entity batch key {key:?} holds {data_len} elements, shape {shape:?} wants {want}"
        )));
    }
    Ok(want)
}

fn check_batch(key: &str, shape: &[usize], rank: usize, b: usize) -> Result<()> {
    if shape.len() != rank || shape[0] != b {
        return Err(Error::shape(format!(
            "entity batch key {key:?} has shape {shape:?}, expected rank {rank} with batch {b}"
        )));
    }
    Ok(())
}

impl<R: Runtime, E: FloatElem> EntityBatch<R, E> {
    /// Row count of a head's label table (`B*M` for `First`, `B*M*K` else).
    pub fn head_rows(spec: &EntityModelSpec, b: usize, steps: StepSelection) -> usize {
        let (m, k) = match &spec.queries {
            Some(q) => (q.count, q.steps),
            None => return 0,
        };
        match steps {
            StepSelection::First => b * m,
            StepSelection::All => b * m * k,
        }
    }

    /// Build a device batch from host arrays, computing every mask and
    /// divisor on the host. Label keys (`label.<head>`) are optional: heads
    /// without one get `None` (greedy inference needs no labels).
    pub fn from_host(spec: &EntityModelSpec, a: &HostArrays, device: &Device<R>) -> Result<Self> {
        spec.validate()?;
        let mut seen = BTreeSet::new();
        // Batch size from the first context set.
        let first = &spec.context[0];
        let (shape0, _) = need_f32(a, &first.name, &mut seen)?;
        if shape0.len() != 3 {
            return Err(Error::shape(format!(
                "entity batch key {:?} has shape {shape0:?}, expected [B, count, features]",
                first.name
            )));
        }
        let b = shape0[0];
        let (m, k) = match &spec.queries {
            Some(q) => (q.count, q.steps),
            None => (0, 0),
        };

        // Context sets.
        let mut ctx_feats = Vec::with_capacity(spec.context.len());
        let mut ctx_presence = Vec::with_capacity(spec.context.len());
        for s in &spec.context {
            let (shape, data) = need_f32(a, &s.name, &mut seen)?;
            check_batch(&s.name, &shape, 3, b)?;
            if shape[1] != s.count || shape[2] != s.features {
                return Err(Error::shape(format!(
                    "entity batch key {:?} has shape {shape:?}, expected [B, {}, {}]",
                    s.name, s.count, s.features
                )));
            }
            check_len(&s.name, &shape, data.len())?;
            ctx_feats.push(Tensor::from_f32(&data, shape, device)?);
            let pkey = format!("{}.presence", s.name);
            match get_f32(a, &pkey, &mut seen)? {
                Some((shape, data)) => {
                    check_batch(&pkey, &shape, 2, b)?;
                    if shape[1] != s.count {
                        return Err(Error::shape(format!(
                            "entity batch key {pkey:?} has shape {shape:?}, expected [B, {}]",
                            s.count
                        )));
                    }
                    check_len(&pkey, &shape, data.len())?;
                    for (i, &v) in data.iter().enumerate() {
                        if v != 0.0 && v != 1.0 {
                            return Err(Error::shape(format!(
                                "entity batch key {pkey:?}[{i}] = {v} is not 0 or 1"
                            )));
                        }
                    }
                    ctx_presence.push(Tensor::from_f32(&data, shape, device)?);
                }
                None => {
                    ctx_presence.push(Tensor::from_f32(
                        &vec![1.0; b * s.count],
                        vec![b, s.count],
                        device,
                    )?);
                }
            }
        }

        // Globals.
        let globals = if spec.globals > 0 {
            let (shape, data) = need_f32(a, "globals", &mut seen)?;
            check_batch("globals", &shape, 2, b)?;
            if shape[1] != spec.globals {
                return Err(Error::shape(format!(
                    "entity batch key \"globals\" has shape {shape:?}, expected [B, {}]",
                    spec.globals
                )));
            }
            check_len("globals", &shape, data.len())?;
            Some(Tensor::from_f32(&data, shape, device)?)
        } else {
            None
        };

        // Queries.
        let (q_feats, q_presence, anchor_ids) = match &spec.queries {
            Some(q) => {
                let (shape, data) = need_f32(a, &q.name, &mut seen)?;
                check_batch(&q.name, &shape, 3, b)?;
                if shape[1] != q.count || shape[2] != q.features {
                    return Err(Error::shape(format!(
                        "entity batch key {:?} has shape {shape:?}, expected [B, {}, {}]",
                        q.name, q.count, q.features
                    )));
                }
                check_len(&q.name, &shape, data.len())?;
                let feats = Tensor::from_f32(&data, shape, device)?;
                let pkey = format!("{}.presence", q.name);
                let presence = match get_f32(a, &pkey, &mut seen)? {
                    Some((shape, data)) => {
                        check_batch(&pkey, &shape, 2, b)?;
                        if shape[1] != q.count {
                            return Err(Error::shape(format!(
                                "entity batch key {pkey:?} has shape {shape:?}, expected [B, {}]",
                                q.count
                            )));
                        }
                        check_len(&pkey, &shape, data.len())?;
                        for (i, &v) in data.iter().enumerate() {
                            if v != 0.0 && v != 1.0 {
                                return Err(Error::shape(format!(
                                    "entity batch key {pkey:?}[{i}] = {v} is not 0 or 1"
                                )));
                            }
                        }
                        Tensor::from_f32(&data, shape, device)?
                    }
                    None => Tensor::from_f32(&vec![1.0; b * q.count], vec![b, q.count], device)?,
                };
                let anchors = match &q.anchor {
                    Some(set) => {
                        let akey = format!("{}.anchor", q.name);
                        let (shape, data) = get_int(a, &akey, &mut seen)?.ok_or_else(|| {
                            Error::shape(format!(
                                "entity batch is missing required key {akey:?} (queries.anchor is set)"
                            ))
                        })?;
                        check_batch(&akey, &shape, 2, b)?;
                        if shape[1] != q.count {
                            return Err(Error::shape(format!(
                                "entity batch key {akey:?} has shape {shape:?}, expected [B, {}]",
                                q.count
                            )));
                        }
                        check_len(&akey, &shape, data.len())?;
                        let off = spec.set_offset(set).ok_or_else(|| {
                            Error::config(format!("queries.anchor {set:?} names no context set"))
                        })?;
                        let count = spec.context.iter().find(|s| s.name == *set).unwrap().count;
                        let mut ids = Vec::with_capacity(b * m);
                        for (i, &v) in data.iter().enumerate() {
                            if v == -1 {
                                ids.push(IGNORE);
                            } else if 0 <= v && (v as usize) < count {
                                ids.push((off + v as usize) as u32);
                            } else {
                                return Err(Error::shape(format!(
                                    "entity batch key {akey:?}[{i}] = {v} is not -1 or 0..{count}"
                                )));
                            }
                        }
                        Some(ids)
                    }
                    None => None,
                };
                (Some(feats), Some(presence), anchors)
            }
            None => (None, None, None),
        };

        // Legal masks per pointer head.
        let mut legal = BTreeMap::new();
        for h in &spec.heads {
            let (set, extra) = match &h.kind {
                HeadKind::Pointer { set, extra_actions } => (set, *extra_actions),
                _ => continue,
            };
            let lkey = format!("legal.{}", h.name);
            if let Some((shape, data)) = get_f32(a, &lkey, &mut seen)? {
                let n = spec.context.iter().find(|s| s.name == *set).unwrap().count;
                let per_query = shape.len() == 3;
                if per_query {
                    check_batch(&lkey, &shape, 3, b)?;
                    if shape[1] != m || shape[2] != n + extra {
                        return Err(Error::shape(format!(
                            "entity batch key {lkey:?} has shape {shape:?}, expected [B, {m}, {}]",
                            n + extra
                        )));
                    }
                } else {
                    check_batch(&lkey, &shape, 2, b)?;
                    if shape[1] != n + extra {
                        return Err(Error::shape(format!(
                            "entity batch key {lkey:?} has shape {shape:?}, expected [B, {}]",
                            n + extra
                        )));
                    }
                }
                check_len(&lkey, &shape, data.len())?;
                for (i, &v) in data.iter().enumerate() {
                    if v != 0.0 && v != 1.0 {
                        return Err(Error::shape(format!(
                            "entity batch key {lkey:?}[{i}] = {v} is not 0 or 1"
                        )));
                    }
                }
                legal.insert(h.name.clone(), Tensor::from_f32(&data, shape, device)?);
            }
        }

        // Labels per head.
        let mut labels = BTreeMap::new();
        let mut choice_ids = BTreeMap::new();
        let mut choice_dev = BTreeMap::new();
        let mut anchor_dev: Option<IdTensor<R>> = None;
        if let Some(ids) = &anchor_ids {
            anchor_dev = Some(IdTensor::from_slice(ids, vec![b * m], device)?);
        }
        for h in &spec.heads {
            let lkey = format!("label.{}", h.name);
            let rows = Self::head_rows(spec, b, h.steps);
            let step_w: Vec<f32> = match &h.step_weights {
                Some(w) => w.clone(),
                None => vec![1.0; k.max(1)],
            };
            let w_at = |row: usize| -> f32 {
                match h.steps {
                    StepSelection::First => step_w[0],
                    StepSelection::All => step_w[(row % k).min(step_w.len() - 1)],
                }
            };
            match &h.kind {
                HeadKind::Pointer { set, extra_actions } => {
                    let n = spec.context.iter().find(|s| s.name == *set).unwrap().count;
                    let width = n + extra_actions;
                    match get_int(a, &lkey, &mut seen)? {
                        Some((shape, data)) => {
                            let rank = if matches!(h.steps, StepSelection::First) {
                                2
                            } else {
                                3
                            };
                            check_batch(&lkey, &shape, rank, b)?;
                            if shape[1] != m || (rank == 3 && shape[2] != k) {
                                return Err(Error::shape(format!(
                                    "entity batch key {lkey:?} has shape {shape:?}, expected [B, {m}, {k}]"
                                )));
                            }
                            check_len(&lkey, &shape, data.len())?;
                            let mut ids = vec![0u32; rows];
                            let mut keep = vec![0.0f32; rows];
                            let mut host = vec![IGNORE; rows];
                            for (i, &v) in data.iter().enumerate() {
                                if v == -1 {
                                    continue;
                                }
                                if !(0 <= v && (v as usize) < width) {
                                    return Err(Error::shape(format!(
                                        "entity batch key {lkey:?}[{i}] = {v} is not -1 or 0..{width}"
                                    )));
                                }
                                ids[i] = v as u32;
                                host[i] = v as u32;
                                keep[i] = w_at(i);
                            }
                            let div = keep.iter().sum::<f32>().max(1.0);
                            labels.insert(
                                h.name.clone(),
                                Some(HeadLabels::Class {
                                    ids: IdTensor::from_slice(&ids, vec![rows], device)?,
                                    keep: Tensor::from_f32(&keep, vec![rows], device)?,
                                    div,
                                }),
                            );
                            choice_ids.insert(h.name.clone(), host.clone());
                            choice_dev.insert(
                                h.name.clone(),
                                IdTensor::from_slice(&host, vec![rows], device)?,
                            );
                        }
                        None => {
                            labels.insert(h.name.clone(), None);
                        }
                    }
                }
                HeadKind::Categorical { classes } => match get_int(a, &lkey, &mut seen)? {
                    Some((shape, data)) => {
                        let rank = if matches!(h.steps, StepSelection::First) {
                            2
                        } else {
                            3
                        };
                        check_batch(&lkey, &shape, rank, b)?;
                        if shape[1] != m || (rank == 3 && shape[2] != k) {
                            return Err(Error::shape(format!(
                                "entity batch key {lkey:?} has shape {shape:?}, expected [B, {m}, {k}]"
                            )));
                        }
                        check_len(&lkey, &shape, data.len())?;
                        let mut ids = vec![0u32; rows];
                        let mut keep = vec![0.0f32; rows];
                        for (i, &v) in data.iter().enumerate() {
                            if v == -1 {
                                continue;
                            }
                            if !(0 <= v && (v as usize) < *classes) {
                                return Err(Error::shape(format!(
                                    "entity batch key {lkey:?}[{i}] = {v} is not -1 or 0..{classes}"
                                )));
                            }
                            ids[i] = v as u32;
                            keep[i] = w_at(i);
                        }
                        let div = keep.iter().sum::<f32>().max(1.0);
                        labels.insert(
                            h.name.clone(),
                            Some(HeadLabels::Class {
                                ids: IdTensor::from_slice(&ids, vec![rows], device)?,
                                keep: Tensor::from_f32(&keep, vec![rows], device)?,
                                div,
                            }),
                        );
                    }
                    None => {
                        labels.insert(h.name.clone(), None);
                    }
                },
                HeadKind::MultiLabel { labels: n_labels } => match get_f32(a, &lkey, &mut seen)? {
                    Some((shape, data)) => {
                        let rank = if matches!(h.steps, StepSelection::First) {
                            3
                        } else {
                            4
                        };
                        check_batch(&lkey, &shape, rank, b)?;
                        let w = shape[rank - 1];
                        if shape[1] != m || w != *n_labels || (rank == 4 && shape[2] != k) {
                            return Err(Error::shape(format!(
                                "entity batch key {lkey:?} has shape {shape:?}, expected [B, {m}, {k}, {n_labels}]"
                            )));
                        }
                        check_len(&lkey, &shape, data.len())?;
                        let mut targets = vec![0.0f32; rows * n_labels];
                        let mut keep = vec![0.0f32; rows];
                        for r in 0..rows {
                            let mut bad = false;
                            for l in 0..*n_labels {
                                let v = data[r * n_labels + l];
                                if v.is_nan() {
                                    bad = true;
                                    break;
                                }
                                if v != 0.0 && v != 1.0 {
                                    return Err(Error::shape(format!(
                                        "entity batch key {lkey:?}[{r},{l}] = {v} is not 0, 1 or NaN"
                                    )));
                                }
                                targets[r * n_labels + l] = v;
                            }
                            if !bad {
                                keep[r] = w_at(r);
                            }
                        }
                        let div = keep.iter().sum::<f32>().max(1.0);
                        labels.insert(
                            h.name.clone(),
                            Some(HeadLabels::Multi {
                                targets: Tensor::from_f32(&targets, vec![rows, *n_labels], device)?,
                                keep: Tensor::from_f32(&keep, vec![rows], device)?,
                                div,
                            }),
                        );
                    }
                    None => {
                        labels.insert(h.name.clone(), None);
                    }
                },
                HeadKind::Regression { outputs } => match get_f32(a, &lkey, &mut seen)? {
                    Some((shape, data)) => {
                        let rank = if matches!(h.steps, StepSelection::First) {
                            3
                        } else {
                            4
                        };
                        check_batch(&lkey, &shape, rank, b)?;
                        let w = shape[rank - 1];
                        if shape[1] != m || w != *outputs || (rank == 4 && shape[2] != k) {
                            return Err(Error::shape(format!(
                                "entity batch key {lkey:?} has shape {shape:?}, expected [B, {m}, {k}, {outputs}]"
                            )));
                        }
                        check_len(&lkey, &shape, data.len())?;
                        let mut targets = vec![0.0f32; rows * outputs];
                        let mut keep = vec![0.0f32; rows];
                        for r in 0..rows {
                            let mut bad = false;
                            for o in 0..*outputs {
                                let v = data[r * outputs + o];
                                if v.is_nan() {
                                    bad = true;
                                    break;
                                }
                                targets[r * outputs + o] = v;
                            }
                            if !bad {
                                keep[r] = w_at(r);
                            }
                        }
                        let div = keep.iter().sum::<f32>().max(1.0);
                        labels.insert(
                            h.name.clone(),
                            Some(HeadLabels::Reg {
                                targets: Tensor::from_f32(&targets, vec![rows, *outputs], device)?,
                                keep: Tensor::from_f32(&keep, vec![rows], device)?,
                                div,
                            }),
                        );
                    }
                    None => {
                        labels.insert(h.name.clone(), None);
                    }
                },
            }
        }

        // Unknown keys are typos until proven otherwise.
        for key in a.f32s.keys().chain(a.ints.keys()) {
            if !seen.contains(key) {
                return Err(Error::shape(format!(
                    "entity batch has unknown key {key:?}; expected keys are globals, <set>, \
                     <set>.presence, <queries>, <queries>.presence, <queries>.anchor, \
                     legal.<head>, label.<head>"
                )));
            }
        }

        let seg = Self::build_seg(spec, &labels, b, device)?;
        Ok(Self {
            b,
            ctx_feats,
            ctx_presence,
            globals,
            q_feats,
            q_presence,
            anchor_ids,
            legal,
            labels,
            choice_ids,
            choice_dev,
            anchor_dev,
            resident: false,
            seg,
        })
    }

    /// Pack fused-loss inputs for the spec's all-steps heads (K3): reads the
    /// per-head labels back once per batch (construction, not per step).
    fn build_seg(
        spec: &EntityModelSpec,
        labels: &BTreeMap<String, Option<HeadLabels<R, E>>>,
        b: usize,
        device: &Device<R>,
    ) -> Result<Option<SegData<R, E>>> {
        let layout = match seg_layout(spec) {
            Some(l) => l,
            None => return Ok(None),
        };
        let (m, k) = match &spec.queries {
            Some(q) => (q.count, q.steps),
            None => return Ok(None),
        };
        let r = b * m * k;
        let h = layout.heads.len();
        let mut class_host = vec![0u32; r * h];
        let mut keep_host = vec![0.0f32; r * h];
        let mut ft_host = vec![0.0f32; r * layout.wtot];
        let mut divs = vec![1.0f32; h];
        let mut wts = vec![1.0f32; h];
        for (hi, name) in layout.heads.iter().enumerate() {
            let spec_head = spec.head(name).unwrap();
            wts[hi] = spec_head.loss_weight;
            let ft_off = layout.seg[hi][4] as usize;
            let width = layout.seg[hi][1] as usize;
            match labels.get(name).and_then(|l| l.as_ref()) {
                Some(HeadLabels::Class { ids, keep, div }) => {
                    let iv = ids.to_vec();
                    let kv = keep.to_f32();
                    for row in 0..r {
                        class_host[row * h + hi] = iv[row];
                        keep_host[row * h + hi] = kv[row];
                    }
                    divs[hi] = *div;
                }
                Some(HeadLabels::Multi { targets, keep, div }) => {
                    let tv = targets.to_f32();
                    let kv = keep.to_f32();
                    for row in 0..r {
                        keep_host[row * h + hi] = kv[row];
                        ft_host[row * layout.wtot + ft_off..row * layout.wtot + ft_off + width]
                            .copy_from_slice(&tv[row * width..(row + 1) * width]);
                    }
                    divs[hi] = *div;
                }
                Some(HeadLabels::Reg { targets, keep, div }) => {
                    let tv = targets.to_f32();
                    let kv = keep.to_f32();
                    for row in 0..r {
                        keep_host[row * h + hi] = kv[row];
                        ft_host[row * layout.wtot + ft_off..row * layout.wtot + ft_off + width]
                            .copy_from_slice(&tv[row * width..(row + 1) * width]);
                    }
                    divs[hi] = *div;
                }
                None => {}
            }
        }
        let coef: Vec<f32> = wts.iter().zip(divs.iter()).map(|(w, d)| w / d).collect();
        let inv_width: Vec<f32> = layout.seg.iter().map(|row| 1.0 / row[1] as f32).collect();
        let seg_flat: Vec<u32> = layout.seg.iter().flat_map(|r| r.iter().copied()).collect();
        Ok(Some(SegData {
            heads: layout.heads,
            class_ids: IdTensor::from_slice(&class_host, vec![r, h], device)?,
            keep: Tensor::from_f32(&keep_host, vec![r, h], device)?,
            ft: Tensor::from_f32(&ft_host, vec![r, layout.wtot], device)?,
            divs,
            wts,
            coef: Tensor::from_f32(&coef, vec![h], device)?,
            seg_dev: IdTensor::from_slice(&seg_flat, vec![h, 5], device)?,
            inv_width: Tensor::from_f32(&inv_width, vec![h], device)?,
        }))
    }
}

impl<R: Runtime, E: FloatElem> EntityDataset<R, E> {
    /// Upload a split once: run the host batch build over all `S` samples,
    /// then flatten every tensor to concatenated `[S, F]` / `[S, I]` rows.
    /// Every id range is validated by [`EntityBatch::from_host`]: an
    /// out-of-range id has no bounds check on the device.
    pub fn from_arrays(spec: &EntityModelSpec, a: &HostArrays, device: &Device<R>) -> Result<Self> {
        let full = EntityBatch::<R, E>::from_host(spec, a, device)?;
        let s = full.b;
        if s == 0 {
            return Err(Error::shape(
                "entity dataset needs at least one sample".to_string(),
            ));
        }
        let flat_f = |t: &Tensor<R, E>| -> Vec<f32> { t.to_f32() };
        let flat_i = |t: &IdTensor<R>| -> Vec<u32> { t.to_vec() };
        // Sections are collected per sample-major row: row s holds every
        // section's s-th row back to back.
        let mut fsecs: Vec<(String, usize, Vec<f32>)> = Vec::new();
        let mut isecs: Vec<(String, usize, Vec<u32>)> = Vec::new();
        let mut push_f = |name: String, w: usize, data: Vec<f32>| {
            debug_assert_eq!(data.len(), s * w);
            fsecs.push((name, w, data));
        };
        let mut push_i = |name: String, w: usize, data: Vec<u32>| {
            debug_assert_eq!(data.len(), s * w);
            isecs.push((name, w, data));
        };
        // Context, globals, queries.
        for (i, t) in full.ctx_feats.iter().enumerate() {
            let w = t.shape().dims()[1..].iter().product();
            push_f(format!("ctxfeat:{i}"), w, flat_f(t));
        }
        for (i, t) in full.ctx_presence.iter().enumerate() {
            let w = t.shape().dims()[1..].iter().product();
            push_f(format!("ctxpres:{i}"), w, flat_f(t));
        }
        if let Some(t) = &full.globals {
            let w = t.shape().dims()[1..].iter().product();
            push_f("globals".to_string(), w, flat_f(t));
        }
        if let Some(t) = &full.q_feats {
            let w = t.shape().dims()[1..].iter().product();
            push_f("qfeat".to_string(), w, flat_f(t));
        }
        if let Some(t) = &full.q_presence {
            let w = t.shape().dims()[1..].iter().product();
            push_f("qpres".to_string(), w, flat_f(t));
        }
        if let Some(t) = &full.anchor_dev {
            push_i("anchor".to_string(), t.len() / s, flat_i(t));
        }
        // Per-head labels: keep + targets ride floats; ids ride the id table.
        let mut heads = Vec::with_capacity(spec.heads.len());
        let mut sample_weights = vec![0.0f32; s * spec.heads.len()];
        let mut legal_shapes = BTreeMap::new();
        for (hi, h) in spec.heads.iter().enumerate() {
            let rows_ps = EntityBatch::<R, E>::head_rows(spec, 1, h.steps);
            let (kind, width) = match &h.kind {
                HeadKind::Pointer { .. } => (DatasetHeadKind::Class, 0),
                HeadKind::Categorical { .. } => (DatasetHeadKind::Class, 0),
                HeadKind::MultiLabel { labels } => (DatasetHeadKind::Multi, *labels),
                HeadKind::Regression { outputs } => (DatasetHeadKind::Reg, *outputs),
            };
            let labelled = full.labels.get(&h.name).and_then(|l| l.as_ref()).is_some();
            heads.push(DatasetHead {
                name: h.name.clone(),
                kind,
                rows_ps,
                width,
                labelled,
            });
            if !labelled {
                continue;
            }
            match full.labels.get(&h.name).and_then(|l| l.as_ref()).unwrap() {
                HeadLabels::Class { ids, keep, .. } => {
                    push_i(format!("class:{}", h.name), rows_ps, flat_i(ids));
                    let kv = flat_f(keep);
                    push_f(format!("keep:{}", h.name), rows_ps, kv.clone());
                    for (si, w) in kv.chunks(rows_ps).enumerate() {
                        sample_weights[si * spec.heads.len() + hi] = w.iter().sum();
                    }
                }
                HeadLabels::Multi { targets, keep, .. } => {
                    let kv = flat_f(keep);
                    let tv = flat_f(targets);
                    push_f(format!("keep:{}", h.name), rows_ps, kv.clone());
                    push_f(format!("target:{}", h.name), rows_ps * width, tv);
                    for (si, w) in kv.chunks(rows_ps).enumerate() {
                        sample_weights[si * spec.heads.len() + hi] = w.iter().sum();
                    }
                }
                HeadLabels::Reg { targets, keep, .. } => {
                    let kv = flat_f(keep);
                    let tv = flat_f(targets);
                    push_f(format!("keep:{}", h.name), rows_ps, kv.clone());
                    push_f(format!("target:{}", h.name), rows_ps * width, tv);
                    for (si, w) in kv.chunks(rows_ps).enumerate() {
                        sample_weights[si * spec.heads.len() + hi] = w.iter().sum();
                    }
                }
            }
            if full.choice_dev.contains_key(&h.name) {
                let t = &full.choice_dev[&h.name];
                push_i(format!("choice:{}", h.name), rows_ps, flat_i(t));
            }
        }
        for (name, t) in &full.legal {
            let dims = t.shape().dims().to_vec();
            let rest = dims[1..].to_vec();
            legal_shapes.insert(name.clone(), rest.clone());
            let w: usize = rest.iter().product();
            push_f(format!("legal:{name}"), w, flat_f(t));
        }
        // Fused-loss tables: repack per sample (construction reads, once).
        if let Some(seg) = &full.seg {
            let (m, k) = match &spec.queries {
                Some(q) => (q.count, q.steps),
                None => (0, 0),
            };
            let rps = m * k;
            let hs = seg.heads.len();
            let cv = flat_i(&seg.class_ids);
            let kv = flat_f(&seg.keep);
            let tv = flat_f(&seg.ft);
            let wtot = tv.len() / (s * rps).max(1);
            let mut sc = Vec::with_capacity(s * rps * hs);
            let mut sk = Vec::with_capacity(s * rps * hs);
            let mut sf = Vec::with_capacity(s * rps * wtot);
            for si in 0..s {
                sc.extend_from_slice(&cv[si * rps * hs..(si + 1) * rps * hs]);
                sk.extend_from_slice(&kv[si * rps * hs..(si + 1) * rps * hs]);
                sf.extend_from_slice(&tv[si * rps * wtot..(si + 1) * rps * wtot]);
            }
            push_i("seg_class".to_string(), rps * hs, sc);
            push_f("seg_keep".to_string(), rps * hs, sk);
            push_f("seg_ft".to_string(), rps * wtot, sf);
        }
        // Interleave sample-major rows: row s holds every section's s-th row.
        let mut float_sections = Vec::with_capacity(fsecs.len());
        let mut off = 0usize;
        for (name, w, _) in &fsecs {
            float_sections.push((name.clone(), off, *w));
            off += w;
        }
        let n_float = off;
        let mut floats = vec![0.0f32; s * n_float];
        for si in 0..s {
            for ((_, w, data), (_, o, _)) in fsecs.iter().zip(float_sections.iter()) {
                floats[si * n_float + o..si * n_float + o + w]
                    .copy_from_slice(&data[si * w..(si + 1) * w]);
            }
        }
        let mut id_sections = Vec::with_capacity(isecs.len());
        let mut off = 0usize;
        for (name, w, _) in &isecs {
            id_sections.push((name.clone(), off, *w));
            off += w;
        }
        let n_id = off;
        let mut ids = vec![0u32; s * n_id];
        for si in 0..s {
            for ((_, w, data), (_, o, _)) in isecs.iter().zip(id_sections.iter()) {
                ids[si * n_id + o..si * n_id + o + w].copy_from_slice(&data[si * w..(si + 1) * w]);
            }
        }
        Ok(Self {
            samples: s,
            floats: Tensor::from_f32(&floats, vec![s, n_float], device)?,
            ids: if n_id > 0 {
                IdTensor::from_slice(&ids, vec![s, n_id], device)?
            } else {
                IdTensor::from_slice(&[], vec![s, 0], device)?
            },
            layout: DatasetLayout {
                float_sections,
                id_sections,
                heads,
                n_float,
                n_id,
                legal_shapes,
            },
            sample_weights,
            _elem: std::marker::PhantomData,
        })
    }
}

impl<R: Runtime, E: FloatElem> EntityBatch<R, E> {
    /// Assemble a batch on the device from a resident split: one
    /// [`gather_rows_multi`](crate::tensor::ops::entity_model::gather_rows_multi)
    /// launch, no reads. `ids` are host sample indices (validated here); only
    /// the `[B]` id buffer crosses to the device per step.
    ///
    /// Resident batches carry no host id maps, so the composed path refuses
    /// them (it maps host ids); the fused path gathers from device ids.
    pub fn from_ids(
        spec: &EntityModelSpec,
        data: &EntityDataset<R, E>,
        ids: &[u32],
    ) -> Result<Self> {
        spec.validate()?;
        let b = ids.len();
        if b == 0 {
            return Err(Error::shape(
                "entity batch needs at least one sample id".to_string(),
            ));
        }
        for (i, &t) in ids.iter().enumerate() {
            if (t as usize) >= data.samples {
                return Err(Error::shape(format!(
                    "entity sample_ids[{i}] = {t} is outside 0..{}",
                    data.samples
                )));
            }
        }
        let device = data.floats.device();
        let id_buf = IdTensor::from_slice(ids, vec![b], device)?;
        // One launch, no reads: the batch is gathered and sliced on device.
        let (gathered_f, gathered_i) =
            crate::tensor::ops::entity_model::gather_rows_multi(&data.floats, &data.ids, &id_buf)?;
        let section_f = |name: &str| -> Option<(usize, usize)> {
            data.layout
                .float_sections
                .iter()
                .find(|(n, _, _)| n == name)
                .map(|(_, o, w)| (*o, *w))
        };
        let section_i = |name: &str| -> Option<(usize, usize)> {
            data.layout
                .id_sections
                .iter()
                .find(|(n, _, _)| n == name)
                .map(|(_, o, w)| (*o, *w))
        };
        let take_f = |name: &str, shape: Vec<usize>| -> Result<Tensor<R, E>> {
            let (o, w) = section_f(name).ok_or_else(|| {
                Error::shape(format!("entity dataset is missing float section {name:?}"))
            })?;
            crate::tensor::ops::movement::slice(&gathered_f, 1, o, w)?.reshape(shape)
        };
        let take_i = |name: &str, rows: usize| -> Result<IdTensor<R>> {
            let (o, w) = section_i(name).ok_or_else(|| {
                Error::shape(format!("entity dataset is missing id section {name:?}"))
            })?;
            crate::tensor::ops::index::slice_ids_along(&gathered_i, 1, o, w)?.reshape(vec![rows])
        };
        // Context, globals, queries.
        let mut ctx_feats = Vec::with_capacity(spec.context.len());
        let mut ctx_presence = Vec::with_capacity(spec.context.len());
        for (i, s) in spec.context.iter().enumerate() {
            ctx_feats.push(take_f(
                &format!("ctxfeat:{i}"),
                vec![b, s.count, s.features],
            )?);
            ctx_presence.push(take_f(&format!("ctxpres:{i}"), vec![b, s.count])?);
        }
        let globals = if spec.globals > 0 {
            Some(take_f("globals", vec![b, spec.globals])?)
        } else {
            None
        };
        let (q_feats, q_presence, anchor_ids, anchor_dev) = match &spec.queries {
            Some(q) => {
                let f = take_f("qfeat", vec![b, q.count, q.features])?;
                let p = take_f("qpres", vec![b, q.count])?;
                let (a, ad) = match &q.anchor {
                    Some(_) => {
                        let v = take_i("anchor", b * q.count)?;
                        (None, Some(v.reshape(vec![b * q.count])?))
                    }
                    None => (None, None),
                };
                (Some(f), Some(p), a, ad)
            }
            None => (None, None, None, None),
        };
        // Labels per head; divisors from the host per-sample sums.
        let mut labels = BTreeMap::new();
        let mut choice_dev = BTreeMap::new();
        for (hi, h) in spec.heads.iter().enumerate() {
            let meta = &data.layout.heads[hi];
            let rows = b * meta.rows_ps;
            if !meta.labelled {
                labels.insert(h.name.clone(), None);
                continue;
            }
            let kv = take_f(&format!("keep:{}", h.name), vec![rows])?;
            let mut div = 0.0f32;
            for &t in ids {
                div += data.sample_weights[t as usize * spec.heads.len() + hi];
            }
            div = div.max(1.0);
            match meta.kind {
                DatasetHeadKind::Class => {
                    let v = take_i(&format!("class:{}", h.name), rows)?;
                    labels.insert(
                        h.name.clone(),
                        Some(HeadLabels::Class {
                            ids: v,
                            keep: kv,
                            div,
                        }),
                    );
                }
                DatasetHeadKind::Multi => {
                    let tv = take_f(&format!("target:{}", h.name), vec![rows, meta.width])?;
                    labels.insert(
                        h.name.clone(),
                        Some(HeadLabels::Multi {
                            targets: tv,
                            keep: kv,
                            div,
                        }),
                    );
                }
                DatasetHeadKind::Reg => {
                    let tv = take_f(&format!("target:{}", h.name), vec![rows, meta.width])?;
                    labels.insert(
                        h.name.clone(),
                        Some(HeadLabels::Reg {
                            targets: tv,
                            keep: kv,
                            div,
                        }),
                    );
                }
            }
            if data
                .layout
                .id_sections
                .iter()
                .any(|(n, _, _)| n == &format!("choice:{}", h.name))
            {
                let v = take_i(&format!("choice:{}", h.name), rows)?;
                choice_dev.insert(h.name.clone(), v);
            }
        }
        let mut legal = BTreeMap::new();
        for (name, rest) in &data.layout.legal_shapes {
            let mut shape = vec![b];
            shape.extend(rest.clone());
            legal.insert(name.clone(), take_f(&format!("legal:{name}"), shape)?);
        }
        // Fused-loss tables from the seg sections.
        let seg = match seg_layout(spec) {
            Some(layout) => {
                let (m, k) = match &spec.queries {
                    Some(q) => (q.count, q.steps),
                    None => (0, 0),
                };
                let (rps, hs) = (m * k, layout.heads.len());
                let rows = b * rps;
                let class_ids = take_i("seg_class", rows * hs)?.reshape(vec![rows, hs])?;
                let keep = take_f("seg_keep", vec![rows, hs])?;
                let ft = take_f("seg_ft", vec![rows, layout.wtot])?;
                let mut divs = vec![1.0f32; hs];
                let mut wts = vec![1.0f32; hs];
                for (sh, name) in layout.heads.iter().enumerate() {
                    let hi = spec.heads.iter().position(|h| &h.name == name).unwrap();
                    wts[sh] = spec.heads[hi].loss_weight;
                    let mut div = 0.0f32;
                    for &t in ids {
                        div += data.sample_weights[t as usize * spec.heads.len() + hi];
                    }
                    divs[sh] = div.max(1.0);
                }
                let coef: Vec<f32> = wts.iter().zip(divs.iter()).map(|(w, d)| w / d).collect();
                let inv_width: Vec<f32> =
                    layout.seg.iter().map(|row| 1.0 / row[1] as f32).collect();
                let seg_flat: Vec<u32> =
                    layout.seg.iter().flat_map(|r| r.iter().copied()).collect();
                Some(SegData {
                    heads: layout.heads,
                    class_ids,
                    keep,
                    ft,
                    divs,
                    wts,
                    coef: Tensor::from_f32(&coef, vec![hs], device)?,
                    seg_dev: IdTensor::from_slice(&seg_flat, vec![hs, 5], device)?,
                    inv_width: Tensor::from_f32(&inv_width, vec![hs], device)?,
                })
            }
            None => None,
        };
        Ok(Self {
            b,
            ctx_feats,
            ctx_presence,
            globals,
            q_feats,
            q_presence,
            anchor_ids,
            legal,
            labels,
            choice_ids: BTreeMap::new(),
            choice_dev,
            anchor_dev,
            resident: true,
            seg,
        })
    }
}
