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
use crate::tensor::ops::index::IdTensor;
use crate::tensor::ops::entity_model::IGNORE;

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
    get_f32(a, key, seen)?.ok_or_else(|| {
        Error::shape(format!("entity batch is missing required key {key:?}"))
    })
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
    pub fn from_host(
        spec: &EntityModelSpec,
        a: &HostArrays,
        device: &Device<R>,
    ) -> Result<Self> {
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
                            let rank = if matches!(h.steps, StepSelection::First) { 2 } else { 3 };
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
                            choice_ids.insert(h.name.clone(), host);
                        }
                        None => {
                            labels.insert(h.name.clone(), None);
                        }
                    }
                }
                HeadKind::Categorical { classes } => {
                    match get_int(a, &lkey, &mut seen)? {
                        Some((shape, data)) => {
                            let rank = if matches!(h.steps, StepSelection::First) { 2 } else { 3 };
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
                    }
                }
                HeadKind::MultiLabel { labels: n_labels } => {
                    match get_f32(a, &lkey, &mut seen)? {
                        Some((shape, data)) => {
                            let rank = if matches!(h.steps, StepSelection::First) { 3 } else { 4 };
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
                    }
                }
                HeadKind::Regression { outputs } => {
                    match get_f32(a, &lkey, &mut seen)? {
                        Some((shape, data)) => {
                            let rank = if matches!(h.steps, StepSelection::First) { 3 } else { 4 };
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
                    }
                }
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
        })
    }
}
