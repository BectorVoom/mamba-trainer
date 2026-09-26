//! Domain-free entity-to-plan model spec (ENTITY_MODEL_PLAN.md §1.2, task G1).
//!
//! These types name no domain (no tiles, units, ops, crops): they describe
//! numbered sets of entities, one query set with a plan length `K`, and a list
//! of prediction heads. The Kaggriculture planner is one such spec, built
//! outside the library.

use crate::error::{Error, Result};
use crate::ssm::config::SsmConfig;

/// Layout of a context set's slots.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SetLayout {
    /// Entities in the order given.
    Sequence,
    /// Row-major `height × width` grid (`count` must equal `height * width`).
    /// With `alternate_axes`, every second encoder layer scans the grid
    /// column-major, so vertical neighbours are adjacent in the scan.
    Grid {
        /// Grid height (rows).
        height: usize,
        /// Grid width (columns).
        width: usize,
        /// Transpose the set's token range on odd encoder layers.
        #[serde(default = "default_true")]
        alternate_axes: bool,
    },
}

fn default_true() -> bool {
    true
}

impl Default for SetLayout {
    /// The default layout: entities in the order given.
    fn default() -> Self {
        SetLayout::Sequence
    }
}

/// One named set of context entities: up to `count` entities with `features`
/// floats each and a presence flag.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ContextSetSpec {
    /// Set name; keys the data contract (§1.3) and pointer targets.
    pub name: String,
    /// Slots in the set (batches always pad to this).
    pub count: usize,
    /// Feature floats per entity.
    pub features: usize,
    /// Slot ordering (default [`SetLayout::Sequence`]).
    #[serde(default)]
    pub layout: SetLayout,
    /// Learned per-slot embedding (default true).
    #[serde(default = "default_true")]
    pub position_embedding: bool,
}

impl ContextSetSpec {
    /// A set of `count` entities with `features` features each.
    pub fn new(name: impl Into<String>, count: usize, features: usize) -> Self {
        Self {
            name: name.into(),
            count,
            features,
            layout: SetLayout::Sequence,
            position_embedding: true,
        }
    }

    /// Set the slot ordering.
    pub fn with_layout(mut self, layout: SetLayout) -> Self {
        self.layout = layout;
        self
    }

    /// Toggle the learned per-slot embedding.
    pub fn with_position_embedding(mut self, on: bool) -> Self {
        self.position_embedding = on;
        self
    }
}

impl Default for ContextSetSpec {
    /// An empty placeholder; [`EntityModelSpec::validate`] rejects it.
    fn default() -> Self {
        Self {
            name: String::new(),
            count: 0,
            features: 0,
            layout: SetLayout::Sequence,
            position_embedding: true,
        }
    }
}

impl From<&crate::rl::spec::EntitySet> for ContextSetSpec {
    /// Reuse an RL observation spec's set (sequence layout, embeddings on).
    fn from(s: &crate::rl::spec::EntitySet) -> Self {
        ContextSetSpec::new(s.name.clone(), s.count, s.features)
    }
}

/// The query set: up to `count` queries with `features` floats each, a
/// presence flag, an optional anchor into a context set, and a plan length.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct QuerySetSpec {
    /// Query-set name; keys the data contract (§1.3).
    pub name: String,
    /// Query slots (batches always pad to this).
    pub count: usize,
    /// Feature floats per query.
    pub features: usize,
    /// Name of a context set; each query then carries an index into it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub anchor: Option<String>,
    /// Plan length `K` (number of steps predicted per query).
    pub steps: usize,
    /// Name of the pointer head whose previous choices condition later steps;
    /// `None` means steps are independent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub autoregressive_on: Option<String>,
    /// How many previous steps condition a step (1 = only j-1).
    /// Ignored when `autoregressive_on` is `None`. Defaults to `steps - 1`.
    pub lags: usize,
}

impl QuerySetSpec {
    /// A query set with `steps` plan steps and `lags = steps - 1`.
    pub fn new(name: impl Into<String>, count: usize, features: usize, steps: usize) -> Self {
        Self {
            name: name.into(),
            count,
            features,
            anchor: None,
            steps,
            autoregressive_on: None,
            lags: steps.saturating_sub(1),
        }
    }

    /// Anchor each query at one entity of the named context set.
    pub fn with_anchor(mut self, set: impl Into<String>) -> Self {
        self.anchor = Some(set.into());
        self
    }

    /// Condition later steps on the named pointer head's previous choices.
    pub fn with_autoregressive(mut self, head: impl Into<String>) -> Self {
        self.autoregressive_on = Some(head.into());
        self
    }

    /// Override how many previous steps condition a step.
    pub fn with_lags(mut self, lags: usize) -> Self {
        self.lags = lags;
        self
    }
}

/// One prediction per (query, step).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum HeadKind {
    /// Scores the entities of `set` plus `extra_actions` learned extras.
    /// Absent entities are masked.
    Pointer {
        /// Context set the pointer selects over.
        set: String,
        /// Learned extra choices appended to the entities (e.g. NONE).
        extra_actions: usize,
    },
    /// `classes` logits (cross-entropy).
    Categorical {
        /// Number of classes.
        classes: usize,
    },
    /// `labels` independent Bernoulli logits (BCE).
    MultiLabel {
        /// Number of independent labels.
        labels: usize,
    },
    /// `outputs` real numbers (MSE).
    Regression {
        /// Number of real-valued outputs.
        outputs: usize,
    },
}

/// Which steps of a head carry labels and outputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StepSelection {
    /// Every step has a label and an output.
    #[default]
    All,
    /// Only step 0 has a label and an output (e.g. eta).
    First,
}

/// One prediction head.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct HeadSpec {
    /// Head name; keys labels (`label.<name>`) and outputs.
    pub name: String,
    /// What the head predicts.
    #[serde(flatten)]
    pub kind: HeadKind,
    /// Name of a pointer head: this head also sees the token of the entity
    /// chosen there (the true entity in training, the decoded one at
    /// inference). `None` means it sees only the query state.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub condition_on: Option<String>,
    /// Which steps carry labels and outputs (default all).
    #[serde(default)]
    pub steps: StepSelection,
    /// Multiplier on this head's loss (default 1.0).
    #[serde(default = "default_loss_weight")]
    pub loss_weight: f32,
    /// Per-step loss weights, length = steps (default all 1.0).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub step_weights: Option<Vec<f32>>,
}

fn default_loss_weight() -> f32 {
    1.0
}

impl HeadSpec {
    /// A pointer head over `set` with `extra_actions` learned extras.
    pub fn pointer(name: impl Into<String>, set: impl Into<String>, extra_actions: usize) -> Self {
        Self {
            name: name.into(),
            kind: HeadKind::Pointer {
                set: set.into(),
                extra_actions,
            },
            condition_on: None,
            steps: StepSelection::All,
            loss_weight: 1.0,
            step_weights: None,
        }
    }

    /// A categorical head with `classes` logits.
    pub fn categorical(name: impl Into<String>, classes: usize) -> Self {
        Self {
            name: name.into(),
            kind: HeadKind::Categorical { classes },
            condition_on: None,
            steps: StepSelection::All,
            loss_weight: 1.0,
            step_weights: None,
        }
    }

    /// A multi-label head with `labels` Bernoulli logits.
    pub fn multilabel(name: impl Into<String>, labels: usize) -> Self {
        Self {
            name: name.into(),
            kind: HeadKind::MultiLabel { labels },
            condition_on: None,
            steps: StepSelection::All,
            loss_weight: 1.0,
            step_weights: None,
        }
    }

    /// A regression head with `outputs` real numbers.
    pub fn regression(name: impl Into<String>, outputs: usize) -> Self {
        Self {
            name: name.into(),
            kind: HeadKind::Regression { outputs },
            condition_on: None,
            steps: StepSelection::All,
            loss_weight: 1.0,
            step_weights: None,
        }
    }

    /// Also feed this head the token of the entity chosen at `head`.
    pub fn condition_on(mut self, head: impl Into<String>) -> Self {
        self.condition_on = Some(head.into());
        self
    }

    /// Set the multiplier on this head's loss.
    pub fn loss_weight(mut self, w: f32) -> Self {
        self.loss_weight = w;
        self
    }

    /// Set the per-step loss weights (length must equal the plan steps).
    pub fn step_weights(mut self, w: Vec<f32>) -> Self {
        self.step_weights = Some(w);
        self
    }

    /// Keep only step 0's label and output.
    pub fn first_step_only(mut self) -> Self {
        self.steps = StepSelection::First;
        self
    }

    /// Width of the head's output per (query, step): `N + E` for pointers,
    /// `classes` / `labels` / `outputs` otherwise (`None` when a pointer's
    /// set names nothing in `spec`).
    pub fn width(&self, spec: &EntityModelSpec) -> Option<usize> {
        match &self.kind {
            HeadKind::Pointer { set, extra_actions } => spec
                .context
                .iter()
                .find(|s| s.name == *set)
                .map(|s| s.count + extra_actions),
            HeadKind::Categorical { classes } => Some(*classes),
            HeadKind::MultiLabel { labels } => Some(*labels),
            HeadKind::Regression { outputs } => Some(*outputs),
        }
    }

    /// Whether this head is a pointer head.
    pub fn is_pointer(&self) -> bool {
        matches!(self.kind, HeadKind::Pointer { .. })
    }
}

/// How the decoder mixes context and query tokens.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DecoderMode {
    /// All queries and steps in one bidirectional scan after the context.
    /// Only valid without `autoregressive_on`.
    Joint,
    /// Step-major sequence `[context ; step 0 queries ; step 1 queries ; …]`
    /// with forward-only scans. With `crew_symmetric`, a second forward scan
    /// runs with the queries reversed inside each step block and the two are
    /// summed.
    StepCausal {
        /// Add the within-step-reversed second scan.
        crew_symmetric: bool,
    },
}

impl Default for DecoderMode {
    /// The default mode: one bidirectional scan.
    fn default() -> Self {
        DecoderMode::Joint
    }
}

fn default_d_model() -> usize {
    128
}

fn default_layers() -> usize {
    3
}

fn default_norm_eps() -> f32 {
    1e-5
}

fn default_ssm() -> SsmConfig {
    SsmConfig {
        d_model: 128,
        n_heads: 4,
        head_dim: 64,
        d_state: 32,
        n_groups: 1,
        ..SsmConfig::default()
    }
}

/// The full domain-free model configuration.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct EntityModelSpec {
    /// Global feature floats per sample (may be 0).
    #[serde(default)]
    pub globals: usize,
    /// Context entity sets (at least one).
    #[serde(default)]
    pub context: Vec<ContextSetSpec>,
    /// Query set (`None` is reserved for context-head models, §8).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub queries: Option<QuerySetSpec>,
    /// Prediction heads (at least one).
    #[serde(default)]
    pub heads: Vec<HeadSpec>,
    /// Residual stream width (default 128).
    #[serde(default = "default_d_model")]
    pub d_model: usize,
    /// Encoder (context) mixer layers (default 3).
    #[serde(default = "default_layers")]
    pub context_layers: usize,
    /// Decoder mixer layers (default 3).
    #[serde(default = "default_layers")]
    pub decoder_layers: usize,
    /// Decoder scan mode (default [`DecoderMode::Joint`]; use
    /// [`EntityModelSpec::default_decoder`] for the query-dependent default).
    #[serde(default)]
    pub decoder: DecoderMode,
    /// Mixer settings (default as TASK_PLANNER_PLAN §2.2).
    #[serde(default = "default_ssm")]
    pub ssm: SsmConfig,
    /// Scan chunk size (`None` = auto, [`EntityModelSpec::chunk_for`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chunk_size: Option<usize>,
    /// Normalisation epsilon.
    #[serde(default = "default_norm_eps")]
    pub norm_eps: f32,
    /// Initialisation seed.
    #[serde(default)]
    pub seed: u64,
}

impl Default for EntityModelSpec {
    /// An empty placeholder; [`EntityModelSpec::validate`] rejects it
    /// (no context sets, no heads).
    fn default() -> Self {
        Self {
            globals: 0,
            context: Vec::new(),
            queries: None,
            heads: Vec::new(),
            d_model: 128,
            context_layers: 3,
            decoder_layers: 3,
            decoder: DecoderMode::Joint,
            ssm: default_ssm(),
            chunk_size: None,
            norm_eps: 1e-5,
            seed: 0,
        }
    }
}

impl EntityModelSpec {
    /// Total number of context tokens: Σ counts.
    pub fn n_ctx(&self) -> usize {
        self.context.iter().map(|s| s.count).sum()
    }

    /// Offset of the named context set in the concatenated context sequence.
    pub fn set_offset(&self, name: &str) -> Option<usize> {
        let mut at = 0;
        for s in &self.context {
            if s.name == name {
                return Some(at);
            }
            at += s.count;
        }
        None
    }

    /// Number of query tokens per step-block layout: `count * steps`
    /// (0 when there is no query set).
    pub fn query_tokens(&self) -> usize {
        match &self.queries {
            Some(q) => q.count * q.steps,
            None => 0,
        }
    }

    /// The head called `name`.
    pub fn head(&self, name: &str) -> Option<&HeadSpec> {
        self.heads.iter().find(|h| h.name == name)
    }

    /// The one pointer head whose choice conditions later steps, if any.
    pub fn plan_head(&self) -> Option<&HeadSpec> {
        match &self.queries {
            Some(q) => match &q.autoregressive_on {
                Some(name) => self.head(name),
                None => None,
            },
            None => None,
        }
    }

    /// Default decoder for the given queries: `StepCausal { crew_symmetric:
    /// true }` when `autoregressive_on` is set, else `Joint`.
    pub fn default_decoder(queries: &Option<QuerySetSpec>) -> DecoderMode {
        match queries {
            Some(q) if q.autoregressive_on.is_some() => DecoderMode::StepCausal {
                crew_symmetric: true,
            },
            _ => DecoderMode::Joint,
        }
    }

    /// Pick the largest chunk `c ∈ {64, 50, 48, 40, 32, 25, 20, 16}` that
    /// divides `len`; 32 when none does.
    pub fn chunk_for(len: usize) -> usize {
        for c in [64usize, 50, 48, 40, 32, 25, 20, 16] {
            if len.is_multiple_of(c) {
                return c;
            }
        }
        32
    }

    /// Check internal consistency, naming the offending field.
    pub fn validate(&self) -> Result<()> {
        if self.context.is_empty() {
            return Err(Error::config(
                "entity spec context is empty; at least one context set is required",
            ));
        }
        if self.heads.is_empty() {
            return Err(Error::config(
                "entity spec heads is empty; at least one head is required",
            ));
        }
        if self.d_model == 0 {
            return Err(Error::config("entity spec d_model must be positive"));
        }
        if self.context_layers == 0 {
            return Err(Error::config("entity spec context_layers must be positive"));
        }
        if self.decoder_layers == 0 {
            return Err(Error::config("entity spec decoder_layers must be positive"));
        }
        if self.norm_eps.is_nan() || self.norm_eps <= 0.0 {
            return Err(Error::config("entity spec norm_eps must be positive"));
        }
        // Context sets.
        for (i, s) in self.context.iter().enumerate() {
            if s.name.is_empty() {
                return Err(Error::config(format!(
                    "entity spec context[{i}].name is empty; every context set needs a name"
                )));
            }
            if self.context[..i].iter().any(|o| o.name == s.name) {
                return Err(Error::config(format!(
                    "entity spec context[{i}].name {:?} is a duplicate set name",
                    s.name
                )));
            }
            if s.count == 0 {
                return Err(Error::config(format!(
                    "entity spec context {:?}.count must be positive",
                    s.name
                )));
            }
            if s.features == 0 {
                return Err(Error::config(format!(
                    "entity spec context {:?}.features must be positive",
                    s.name
                )));
            }
            if let SetLayout::Grid {
                height,
                width,
                alternate_axes: _,
            } = &s.layout
                && height.checked_mul(*width) != Some(s.count)
            {
                return Err(Error::config(format!(
                    "entity spec context {:?}.layout Grid({}x{}) does not match count {}",
                    s.name, height, width, s.count
                )));
            }
        }
        // Query set.
        if let Some(q) = &self.queries {
            if q.name.is_empty() {
                return Err(Error::config(
                    "entity spec queries.name is empty; the query set needs a name",
                ));
            }
            if self.context.iter().any(|s| s.name == q.name) {
                return Err(Error::config(format!(
                    "entity spec queries.name {:?} is a duplicate set name (clashes with a context set)",
                    q.name
                )));
            }
            if q.count == 0 {
                return Err(Error::config(format!(
                    "entity spec queries {:?}.count must be positive",
                    q.name
                )));
            }
            if q.features == 0 {
                return Err(Error::config(format!(
                    "entity spec queries {:?}.features must be positive",
                    q.name
                )));
            }
            if q.steps == 0 {
                return Err(Error::config(format!(
                    "entity spec queries {:?}.steps must be positive",
                    q.name
                )));
            }
            if let Some(anchor) = &q.anchor
                && self.set_offset(anchor).is_none()
            {
                return Err(Error::config(format!(
                    "entity spec queries {:?}.anchor {:?} names no context set",
                    q.name, anchor
                )));
            }
            if let Some(plan) = &q.autoregressive_on {
                match self.head(plan) {
                    Some(h) if h.is_pointer() => {}
                    Some(_) => {
                        return Err(Error::config(format!(
                            "entity spec queries {:?}.autoregressive_on {:?} names a head that is not a pointer",
                            q.name, plan
                        )));
                    }
                    None => {
                        return Err(Error::config(format!(
                            "entity spec queries {:?}.autoregressive_on {:?} names no head",
                            q.name, plan
                        )));
                    }
                }
                if matches!(self.decoder, DecoderMode::Joint) {
                    return Err(Error::config(
                        "entity spec decoder Joint cannot be used with queries.autoregressive_on \
                         (it would leak future steps); use decoder StepCausal"
                            .to_string(),
                    ));
                }
            }
        }
        // Heads.
        for (i, h) in self.heads.iter().enumerate() {
            if h.name.is_empty() {
                return Err(Error::config(format!(
                    "entity spec heads[{i}].name is empty; every head needs a name"
                )));
            }
            if self.heads[..i].iter().any(|o| o.name == h.name) {
                return Err(Error::config(format!(
                    "entity spec heads[{i}].name {:?} is a duplicate head name",
                    h.name
                )));
            }
            match &h.kind {
                HeadKind::Pointer {
                    set,
                    extra_actions: _,
                } => {
                    if self.set_offset(set).is_none() {
                        return Err(Error::config(format!(
                            "entity spec head {:?}.kind Pointer.set {:?} names no context set",
                            h.name, set
                        )));
                    }
                    if h.condition_on.is_some() {
                        return Err(Error::config(format!(
                            "entity spec head {:?}.condition_on is not supported on pointer heads \
                             (pointer logits read the query state alone)",
                            h.name
                        )));
                    }
                }
                HeadKind::Categorical { classes } => {
                    if *classes == 0 {
                        return Err(Error::config(format!(
                            "entity spec head {:?}.kind Categorical.classes must be positive",
                            h.name
                        )));
                    }
                }
                HeadKind::MultiLabel { labels } => {
                    if *labels == 0 {
                        return Err(Error::config(format!(
                            "entity spec head {:?}.kind MultiLabel.labels must be positive",
                            h.name
                        )));
                    }
                }
                HeadKind::Regression { outputs } => {
                    if *outputs == 0 {
                        return Err(Error::config(format!(
                            "entity spec head {:?}.kind Regression.outputs must be positive",
                            h.name
                        )));
                    }
                }
            }
            if let Some(cond) = &h.condition_on {
                match self.head(cond) {
                    Some(c) if c.is_pointer() => {}
                    Some(_) => {
                        return Err(Error::config(format!(
                            "entity spec head {:?}.condition_on {:?} names a head that is not a pointer",
                            h.name, cond
                        )));
                    }
                    None => {
                        return Err(Error::config(format!(
                            "entity spec head {:?}.condition_on {:?} names no head",
                            h.name, cond
                        )));
                    }
                }
            }
            if h.loss_weight.is_nan() || h.loss_weight <= 0.0 {
                return Err(Error::config(format!(
                    "entity spec head {:?}.loss_weight must be positive",
                    h.name
                )));
            }
            if let Some(w) = &h.step_weights {
                let steps = self.queries.as_ref().map(|q| q.steps).unwrap_or(1);
                if w.len() != steps {
                    return Err(Error::config(format!(
                        "entity spec head {:?}.step_weights has length {} but queries.steps is {}",
                        h.name,
                        w.len(),
                        steps
                    )));
                }
            }
            // StepSelection::First is meaningless on the plan head (it would
            // leave later steps without the choices that condition them).
            if matches!(h.steps, StepSelection::First)
                && self
                    .queries
                    .as_ref()
                    .and_then(|q| q.autoregressive_on.as_ref())
                    .is_some_and(|p| p == &h.name)
            {
                return Err(Error::config(format!(
                    "entity spec head {:?}.steps First cannot be used on the plan head \
                     (autoregressive_on); the plan head needs every step",
                    h.name
                )));
            }
        }
        let mut ssm = self.ssm.clone();
        ssm.d_model = self.d_model;
        ssm.validate()
            .map_err(|e| Error::config(format!("entity spec ssm is invalid: {e}")))?;
        Ok(())
    }
}

/// The reference Kaggriculture spec from ENTITY_MODEL_PLAN.md §1.4, for tests
/// and examples. Kept behind `cfg(test)` so the library stays domain-free —
/// callers outside the crate build the same spec field by field.
#[cfg(test)]
pub(crate) fn kaggriculture_spec() -> EntityModelSpec {
    EntityModelSpec {
        globals: 114,
        context: vec![
            ContextSetSpec::new("tiles", 100, 48).with_layout(SetLayout::Grid {
                height: 10,
                width: 10,
                alternate_axes: true,
            }),
        ],
        queries: Some(
            QuerySetSpec::new("units", 20, 36, 3)
                .with_anchor("tiles")
                .with_autoregressive("target"),
        ),
        heads: vec![
            HeadSpec::pointer("target", "tiles", 1).step_weights(vec![1.0, 0.5, 0.5]),
            HeadSpec::categorical("op", 13).condition_on("target"),
            HeadSpec::multilabel("opset", 13)
                .condition_on("target")
                .loss_weight(0.3),
            HeadSpec::categorical("crop", 5)
                .condition_on("target")
                .loss_weight(0.3),
            HeadSpec::regression("eta", 1)
                .first_step_only()
                .loss_weight(0.1),
        ],
        d_model: 128,
        context_layers: 3,
        decoder_layers: 3,
        decoder: DecoderMode::StepCausal {
            crew_symmetric: true,
        },
        ssm: default_ssm(),
        chunk_size: None,
        norm_eps: 1e-5,
        seed: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kaggriculture_spec_validates() {
        kaggriculture_spec().validate().unwrap();
    }
}
