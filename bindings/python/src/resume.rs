//! Learner checkpoints: what a learner saves beside its weights, and how a load
//! decides whether what it restored is a continuation.
//!
//! A learner checkpoint is a [`Checkpoint`] whose metadata carries
//!
//! ```text
//! {"format": "mamba3-learner", "kind": "ppo" | "imitation",
//!  "rounds": <integer>,
//!  "policy": <architecture, as Policy.save writes it>,
//!  "trainer_config": {"version": 1, "learning_rate", "lr_schedule",
//!                     "max_grad_norm", "optimizer", "algorithm", "policy",
//!                     "reference"},
//!  "continuation": {"exact": bool, "notes": [...]}}
//! ```
//!
//! Every counter is a JSON integer (see `mamba3::train::checkpoint`'s notes on
//! counters). Loading is all or nothing: the configuration is compared, the
//! weights staged, and the optimizer restored into a *new* trainer before the
//! learner itself is touched; the learner then swaps everything in at once.

use mamba3::train::{AdamW, AdamWConfig, Checkpoint, LrSchedule, RestoreReport, Trainer, TrainerConfig};
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use serde_json::{Value, json};

use crate::err::IntoPyResult;
use crate::{E, R};

/// `metadata.format` of a learner checkpoint.
const LEARNER_FORMAT: &str = "mamba3-learner";
/// `metadata.trainer_config.version` this build writes and reads.
const TRAINER_CONFIG_VERSION: u64 = 1;
/// Configuration fields a load with `config="checkpoint"` may adopt. Everything
/// else — the architecture, the reference weights, the learner kind — is a
/// property of the objects the learner was built around and cannot be adopted.
const ADOPTABLE: [&str; 5] = ["learning_rate", "lr_schedule", "max_grad_norm", "optimizer", "algorithm"];

/// A load failure as the exception a caller would catch: an unreadable file is
/// an `OSError`, anything wrong with its contents a `ValueError`.
pub fn load_error(err: mamba3::error::Error) -> PyErr {
    match err {
        mamba3::error::Error::Io(_) => PyIOError::new_err(err.to_string()),
        other => PyValueError::new_err(other.to_string()),
    }
}

/// The optimizer half of a learner, which both loops configure the same way.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OptimSettings {
    pub learning_rate: f32,
    /// Advances on optimizer updates, i.e. `Trainer::step_count()` — PPO epochs
    /// and minibatches each take one, so a window trained over `epochs *
    /// minibatches` times advances the schedule that many steps, not one.
    pub schedule: LrSchedule,
    pub max_grad_norm: f32,
    pub weight_decay: f32,
    pub betas: (f32, f32),
    pub eps: f32,
}

impl OptimSettings {
    fn adamw(&self) -> AdamWConfig {
        AdamWConfig::builder()
            .learning_rate(self.learning_rate)
            .betas(self.betas.0, self.betas.1)
            .eps(self.eps)
            .weight_decay(self.weight_decay)
            .build()
    }

    /// A fresh trainer with these settings.
    pub fn trainer(&self) -> PyResult<Trainer<R, E, AdamW<R, E>>> {
        let config = TrainerConfig::builder()
            .learning_rate(self.learning_rate)
            .max_grad_norm(self.max_grad_norm)
            .schedule(self.schedule)
            .build()
            .py()?;
        Ok(Trainer::new(config, self.adamw().init::<R, E>()))
    }

    fn to_json(self) -> Value {
        let adamw = self.adamw();
        json!({
            "learning_rate": self.learning_rate,
            "lr_schedule": self.schedule,
            "max_grad_norm": self.max_grad_norm,
            "optimizer": {
                "type": "adamw",
                "beta1": adamw.beta1,
                "beta2": adamw.beta2,
                "eps": adamw.eps,
                "weight_decay": adamw.weight_decay,
                "decay_matrices_only": adamw.decay_matrices_only,
            },
        })
    }

    fn from_json(config: &Value) -> PyResult<Self> {
        let number = |value: Option<&Value>, what: &str| -> PyResult<f32> {
            value.and_then(Value::as_f64).map(|v| v as f32).ok_or_else(|| {
                PyValueError::new_err(format!("the checkpoint's trainer_config.{what} is not a number"))
            })
        };
        let optimizer = config.get("optimizer").cloned().unwrap_or(Value::Null);
        if optimizer.get("type").and_then(Value::as_str) != Some("adamw") {
            return Err(PyValueError::new_err(
                "the checkpoint's optimizer is not AdamW, the only one these learners run",
            ));
        }
        if optimizer.get("decay_matrices_only").and_then(Value::as_bool) != Some(true) {
            return Err(PyValueError::new_err(
                "the checkpoint's optimizer decays every parameter, which these learners cannot run",
            ));
        }
        let schedule: LrSchedule = serde_json::from_value(
            config.get("lr_schedule").cloned().unwrap_or(Value::Null),
        )
        .map_err(|e| PyValueError::new_err(format!("the checkpoint's lr_schedule is unusable: {e}")))?;
        schedule.validate().py()?;
        Ok(Self {
            learning_rate: number(config.get("learning_rate"), "learning_rate")?,
            schedule,
            max_grad_norm: number(config.get("max_grad_norm"), "max_grad_norm")?,
            weight_decay: number(optimizer.get("weight_decay"), "optimizer.weight_decay")?,
            betas: (
                number(optimizer.get("beta1"), "optimizer.beta1")?,
                number(optimizer.get("beta2"), "optimizer.beta2")?,
            ),
            eps: number(optimizer.get("eps"), "optimizer.eps")?,
        })
    }
}

/// How a load treats a training configuration that differs from the live one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigMode {
    /// Refuse, listing every difference. The default.
    Verify,
    /// Adopt the checkpoint's optimizer, schedule and algorithm settings.
    Checkpoint,
    /// Keep the live settings and record that the run is not an exact
    /// continuation.
    Live,
}

impl ConfigMode {
    pub fn parse(value: &str) -> PyResult<Self> {
        match value {
            "verify" => Ok(Self::Verify),
            "checkpoint" => Ok(Self::Checkpoint),
            "live" => Ok(Self::Live),
            other => Err(PyValueError::new_err(format!(
                "config must be 'verify', 'checkpoint' or 'live', got {other:?}"
            ))),
        }
    }
}

/// Whether a learner's history is one uninterrupted run under one
/// configuration, and if not, why not.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Continuation {
    pub exact: bool,
    pub notes: Vec<String>,
}

impl Continuation {
    pub fn fresh() -> Self {
        Self { exact: true, notes: Vec::new() }
    }

    fn to_json(&self) -> Value {
        json!({"exact": self.exact, "notes": self.notes})
    }

    fn from_metadata(metadata: &Value) -> Self {
        let Some(saved) = metadata.get("continuation") else {
            return Self::fresh();
        };
        Self {
            exact: saved.get("exact").and_then(Value::as_bool).unwrap_or(false),
            notes: saved
                .get("notes")
                .and_then(Value::as_array)
                .map(|notes| notes.iter().filter_map(|n| n.as_str().map(str::to_string)).collect())
                .unwrap_or_default(),
        }
    }

    pub fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("exact", self.exact)?;
        dict.set_item("notes", self.notes.clone())?;
        Ok(dict)
    }
}

/// Everything about a learner that decides what its next update does, as
/// saved under `trainer_config`.
pub struct LiveConfig {
    pub kind: &'static str,
    pub optim: OptimSettings,
    pub algorithm: Value,
    pub policy: Value,
    /// `null`, or `{"coeff": …, "weights_fnv1a64": "…"}`.
    pub reference: Value,
}

impl LiveConfig {
    fn trainer_config(&self) -> Value {
        let mut config = self.optim.to_json();
        let object = config.as_object_mut().expect("to_json builds an object");
        object.insert("version".into(), json!(TRAINER_CONFIG_VERSION));
        object.insert("algorithm".into(), self.algorithm.clone());
        // The initialisation seed only decides weights a restore replaces, so it
        // is not part of what a continuation has to agree on.
        let mut policy = self.policy.clone();
        if let Some(fields) = policy.as_object_mut() {
            fields.remove("seed");
        }
        object.insert("policy".into(), policy);
        object.insert("reference".into(), self.reference.clone());
        config
    }
}

/// Write a learner checkpoint.
pub fn save(
    path: &str,
    policy: &mamba3::rl::Mamba3Policy<R, E>,
    trainer: &Trainer<R, E, AdamW<R, E>>,
    rounds: u64,
    live: &LiveConfig,
    continuation: &Continuation,
) -> PyResult<()> {
    let metadata = json!({
        "format": LEARNER_FORMAT,
        "kind": live.kind,
        "rounds": rounds,
        "policy": live.policy,
        "trainer_config": live.trainer_config(),
        "continuation": continuation.to_json(),
    });
    Checkpoint::capture::<R, E, _>(policy, trainer.step_count())
        .with_optimizer(policy, trainer.optimizer())
        .with_metadata(metadata)
        .save(path)
        .py()
}

/// What a successful load produced, for the learner to swap in.
pub struct Loaded {
    pub trainer: Trainer<R, E, AdamW<R, E>>,
    pub optim: OptimSettings,
    /// The checkpoint's algorithm settings, when `config="checkpoint"` adopted
    /// them.
    pub adopted_algorithm: Option<Value>,
    pub rounds: u64,
    pub continuation: Continuation,
    report: RestoreReport,
    counters: bool,
    config: &'static str,
    notes: Vec<String>,
}

impl Loaded {
    /// The summary `load_checkpoint` returns.
    pub fn summary<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("weights", self.report.weights)?;
        dict.set_item("optimizer", self.report.optimizer)?;
        dict.set_item("counters", self.counters)?;
        dict.set_item("config", self.config)?;
        dict.set_item("exact", self.continuation.exact)?;
        dict.set_item("notes", self.notes.clone())?;
        Ok(dict)
    }
}

/// Every leaf of `saved` that differs from `live`, as `path: checkpoint X, live Y`.
fn differences(path: &str, saved: &Value, live: &Value, out: &mut Vec<String>) {
    match (saved, live) {
        (Value::Object(a), Value::Object(b)) => {
            let mut keys: Vec<&String> = a.keys().chain(b.keys()).collect();
            keys.sort();
            keys.dedup();
            for key in keys {
                let child = if path.is_empty() { key.clone() } else { format!("{path}.{key}") };
                differences(
                    &child,
                    a.get(key).unwrap_or(&Value::Null),
                    b.get(key).unwrap_or(&Value::Null),
                    out,
                );
            }
        }
        // Every float here started life as an `f32`, and serde_json's decimal
        // parsing is not guaranteed to reproduce the last bit of the `f64` it
        // printed, so floats are equal when they name the same `f32`.
        (Value::Number(a), Value::Number(b)) if a.is_f64() || b.is_f64() => {
            let same = match (a.as_f64(), b.as_f64()) {
                (Some(x), Some(y)) => (x as f32).to_bits() == (y as f32).to_bits(),
                _ => false,
            };
            if !same {
                out.push(format!("{path}: checkpoint {a}, live {b}"));
            }
        }
        (a, b) if a != b => out.push(format!("{path}: checkpoint {a}, live {b}")),
        _ => {}
    }
}

/// Load a learner checkpoint into a *new* trainer built for the configuration
/// `mode` settles on, restoring `policy`'s weights last. On error nothing —
/// not `policy`, not the caller's trainer — has changed.
///
/// `check_algorithm` validates the checkpoint's algorithm settings before
/// anything is restored, when `config="checkpoint"` is about to adopt them.
pub fn load(
    checkpoint: &Checkpoint,
    policy: &mamba3::rl::Mamba3Policy<R, E>,
    live: &LiveConfig,
    strict: bool,
    mode: ConfigMode,
    check_algorithm: &dyn Fn(&Value) -> PyResult<()>,
) -> PyResult<Loaded> {
    let metadata = &checkpoint.metadata;
    if checkpoint.optimizer.is_none() && strict {
        return Err(PyValueError::new_err(
            "this checkpoint carries no optimizer state to restore (Policy.save writes \
             weights only); pass strict=False for an explicit warm start",
        ));
    }
    let warm_start = checkpoint.optimizer.is_none();

    let mut notes = Vec::new();
    let mut optim = live.optim;
    let mut adopted_algorithm = None;
    // This load's own verdict; the checkpoint's recorded history is folded in below.
    let mut exact = true;
    let config_outcome;

    if warm_start {
        // Weights alone start a new run; there is no configuration to continue.
        config_outcome = "warm_start";
        exact = false;
        notes.push("warm start from weights only: optimizer and counters restart".to_string());
    } else if metadata.get("trainer_config").is_none() {
        if mode != ConfigMode::Live {
            return Err(PyValueError::new_err(
                "this checkpoint carries no trainer_config (it predates configuration \
                 persistence, or was not written by a learner), so whether the run \
                 continues under the same settings cannot be checked; pass \
                 config='live' to load it as a non-exact continuation",
            ));
        }
        config_outcome = "legacy";
        exact = false;
        notes.push("legacy checkpoint without trainer_config".to_string());
    } else {
        let saved = &metadata["trainer_config"];
        if saved.get("version").and_then(Value::as_u64) != Some(TRAINER_CONFIG_VERSION) {
            return Err(PyValueError::new_err(format!(
                "trainer_config version {} is not supported; this build reads version \
                 {TRAINER_CONFIG_VERSION}",
                saved.get("version").unwrap_or(&Value::Null)
            )));
        }
        let saved_kind = metadata.get("kind").and_then(Value::as_str).unwrap_or("");
        let mut all = Vec::new();
        if saved_kind != live.kind {
            all.push(format!("kind: checkpoint {saved_kind:?}, live {:?}", live.kind));
        }
        differences("", saved, &live.trainer_config(), &mut all);
        let (adoptable, fixed): (Vec<String>, Vec<String>) = all.into_iter().partition(|d| {
            ADOPTABLE.iter().any(|field| d == field || d.starts_with(&format!("{field}.")) || d.starts_with(&format!("{field}:")))
        });

        match mode {
            ConfigMode::Verify => {
                let every: Vec<String> = fixed.iter().chain(&adoptable).cloned().collect();
                if !every.is_empty() {
                    return Err(PyValueError::new_err(format!(
                        "the checkpoint was trained under a different configuration; pass \
                         config='checkpoint' to adopt its optimizer, schedule and algorithm \
                         settings, or config='live' to keep these as a non-exact \
                         continuation. Differences:\n  {}",
                        every.join("\n  ")
                    )));
                }
                config_outcome = "verified";
            }
            ConfigMode::Checkpoint => {
                if !fixed.is_empty() {
                    return Err(PyValueError::new_err(format!(
                        "config='checkpoint' can adopt optimizer, schedule and algorithm \
                         settings, not these:\n  {}",
                        fixed.join("\n  ")
                    )));
                }
                optim = OptimSettings::from_json(saved)?;
                let algorithm = saved.get("algorithm").cloned().unwrap_or(Value::Null);
                check_algorithm(&algorithm)?;
                adopted_algorithm = Some(algorithm);
                notes.extend(adoptable.iter().map(|d| format!("adopted {d}")));
                config_outcome = "adopted";
            }
            ConfigMode::Live => {
                if !(fixed.is_empty() && adoptable.is_empty()) {
                    exact = false;
                    notes.extend(fixed.iter().chain(&adoptable).map(|d| format!("kept live {d}")));
                }
                config_outcome = "live";
            }
        }
    }

    // Counters, validated before anything is restored.
    let rounds = if warm_start {
        0
    } else {
        match metadata.get("rounds") {
            Some(value) => value.as_u64().ok_or_else(|| {
                PyValueError::new_err(format!(
                    "the checkpoint's rounds is {value}, not a non-negative integer"
                ))
            })?,
            None if strict => {
                return Err(PyValueError::new_err(
                    "this checkpoint records no round counter; it was not written by a \
                     learner's save()",
                ));
            }
            None => 0,
        }
    };

    let mut trainer = optim.trainer()?;
    let report = checkpoint
        .restore_training::<R, E, _, _>(policy, trainer.optimizer_mut(), strict)
        .map_err(load_error)?;
    trainer.set_step_count(if warm_start { 0 } else { checkpoint.step });

    let mut continuation = Continuation::from_metadata(metadata);
    continuation.exact = continuation.exact && exact;
    if warm_start {
        continuation = Continuation { exact: false, notes: Vec::new() };
    }
    continuation.notes.extend(notes.iter().cloned());

    Ok(Loaded {
        trainer,
        optim,
        adopted_algorithm,
        rounds,
        continuation,
        report,
        counters: !warm_start,
        config: config_outcome,
        notes,
    })
}

/// A stable fingerprint of a policy's weights: FNV-1a over every parameter's
/// path, shape and `f32` bit pattern, in path order. One host read.
pub fn weights_fingerprint(policy: &mamba3::rl::Mamba3Policy<R, E>) -> String {
    use mamba3::nn::Module;
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = OFFSET;
    let mut feed = |bytes: &[u8]| {
        for &b in bytes {
            hash ^= u64::from(b);
            hash = hash.wrapping_mul(PRIME);
        }
    };
    for (name, tensor) in Module::<R, E>::state_dict(policy).entries {
        feed(name.as_bytes());
        for dim in &tensor.shape {
            feed(&(*dim as u64).to_le_bytes());
        }
        for value in &tensor.data {
            feed(&value.to_bits().to_le_bytes());
        }
    }
    format!("{hash:016x}")
}

/// Read the trainer configuration a learner checkpoint was saved with, for a
/// constructor that builds a learner from it.
pub fn saved_trainer_config(checkpoint: &Checkpoint, kind: &str) -> PyResult<(Value, OptimSettings)> {
    let metadata = &checkpoint.metadata;
    if metadata.get("format").and_then(Value::as_str) != Some(LEARNER_FORMAT) {
        return Err(PyValueError::new_err(
            "not a learner checkpoint: from_checkpoint needs one written by a learner's save()",
        ));
    }
    let saved_kind = metadata.get("kind").and_then(Value::as_str).unwrap_or("");
    if saved_kind != kind {
        return Err(PyValueError::new_err(format!(
            "this is a {saved_kind:?} learner checkpoint, not a {kind:?} one"
        )));
    }
    let saved = metadata.get("trainer_config").ok_or_else(|| {
        PyValueError::new_err("this learner checkpoint predates trainer_config; construct the learner and use load_checkpoint(config='live')")
    })?;
    Ok((saved.clone(), OptimSettings::from_json(saved)?))
}
