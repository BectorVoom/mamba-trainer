//! Experiment configuration (TOML).

use std::path::{Path, PathBuf};

use clm_eval::folds::{FoldConfig, Scheme};
use clm_model::{
    ContrastiveConfig, EncoderConfig, HeadConfig, LossConfig, MambaConfig, ModelConfig, Pooling,
    Scoring, SourceConfig, TcnConfig, Temperature, TrainConfig, action_attributes,
};
use serde::{Deserialize, Serialize};

/// A whole experiment.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExperimentConfig {
    /// Run name.
    pub name: String,
    /// Directory of `<SYMBOL>.parquet` + `universe.json`, relative to the config file.
    pub bars_dir: PathBuf,
    /// Base seed.
    pub seed: u64,
    /// Input window (time steps).
    pub window: usize,
    /// Labels.
    pub labels: LabelSection,
    /// Folds.
    pub folds: FoldSection,
    /// Optimisation defaults (each model may override through its grid).
    #[serde(default)]
    pub train: TrainSection,
    /// One-way trading cost used by the policy backtest, in basis points.
    pub cost_bps: f64,
    /// Decision-threshold candidates tried on the calibration segment (the
    /// secondary policy).
    #[serde(default = "default_taus")]
    pub tau_grid: Vec<f64>,
    /// The threshold of the primary policy, the same for every model.
    #[serde(default)]
    pub policy_tau: f64,
    /// The model every other is compared against in the report.
    pub reference: String,
    /// Models.
    pub models: Vec<ModelSpec>,
}

fn default_taus() -> Vec<f64> {
    vec![0.0, 0.02, 0.05, 0.1, 0.2]
}

/// Label settings (CONTRACTS C3).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LabelSection {
    /// Horizon in trading days.
    pub horizon: usize,
    /// One-way cost assumed by the labels, in basis points.
    pub kappa_bps: f64,
    /// Margin multiplier `m` of `theta = 2 kappa + m sigma_h`.
    pub margin_mult: f64,
    /// Stop-loss fraction for the risk target.
    pub stop_frac: f64,
}

/// Fold settings (CONTRACTS C4).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FoldSection {
    /// `expanding` or `rolling`.
    pub scheme: String,
    /// Training days for `rolling`.
    #[serde(default)]
    pub train_days: usize,
    /// First test year.
    pub first_test_year: i32,
    /// Last test year.
    pub last_test_year: i32,
    /// Validation days.
    pub val_days: usize,
    /// Calibration days.
    pub cal_days: usize,
    /// Embargo days.
    pub embargo: usize,
    /// Minimum training days for a fold to be emitted.
    pub min_train_days: usize,
}

/// Optimisation settings shared by all models.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct TrainSection {
    /// Most steps.
    pub max_steps: u64,
    /// Batch size.
    pub batch_size: usize,
    /// Learning rate.
    pub learning_rate: f32,
    /// Weight decay.
    pub weight_decay: f32,
    /// Warmup steps.
    pub warmup_steps: u64,
    /// Gradient clip.
    pub grad_clip: f32,
    /// Evaluation interval.
    pub eval_every: u64,
    /// Early-stopping patience (evaluations).
    pub patience: u32,
    /// Evaluation batch.
    pub eval_batch: usize,
}

impl Default for TrainSection {
    fn default() -> Self {
        let t = TrainConfig::default();
        Self {
            max_steps: t.max_steps,
            batch_size: t.batch_size,
            learning_rate: t.learning_rate,
            weight_decay: t.weight_decay,
            warmup_steps: t.warmup_steps,
            grad_clip: t.grad_clip,
            eval_every: t.eval_every,
            patience: t.patience,
            eval_batch: t.eval_batch,
        }
    }
}

/// One compared model.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelSpec {
    /// Name (used in paths and the report).
    pub name: String,
    /// `mamba`, `tcn`, `mlp` (last row through a hidden layer), `linear` (last row),
    /// or `external` (predictions from another program, e.g. LightGBM).
    pub encoder: String,
    /// `linear`, `mlp` or `contrastive` (ignored for `external`).
    #[serde(default = "default_head")]
    pub head: String,
    /// Directory with `fold_<year>/pred_{cal,test}.parquet` for `external`.
    #[serde(default)]
    pub external_dir: Option<PathBuf>,
    /// Mamba / TCN width.
    #[serde(default = "d64")]
    pub d_model: usize,
    /// Mamba / TCN depth.
    #[serde(default = "d2")]
    pub layers: usize,
    /// Mamba state size.
    #[serde(default = "d16")]
    pub d_state: usize,
    /// Mamba expansion.
    #[serde(default = "d2")]
    pub expand: usize,
    /// Mamba conv taps / TCN kernel.
    #[serde(default = "d4")]
    pub d_conv: usize,
    /// `last` or `mean`.
    #[serde(default = "default_pool")]
    pub pooling: String,
    /// Hidden width of MLP heads and bodies.
    #[serde(default = "d64")]
    pub hidden: usize,
    /// Contrastive embedding width.
    #[serde(default = "d64")]
    pub embed_dim: usize,
    /// Contrastive temperature (initial value when learned).
    #[serde(default = "default_tau")]
    pub temperature: f32,
    /// Learn the temperature.
    #[serde(default = "yes")]
    pub learned_temperature: bool,
    /// `cosine` or `bilinear`.
    #[serde(default = "default_scoring")]
    pub scoring: String,
    /// Supervised-contrastive weight.
    #[serde(default)]
    pub supcon: f32,
    /// Supervised-contrastive temperature.
    #[serde(default = "default_tau")]
    pub supcon_temperature: f32,
    /// Auxiliary return weight (0 = no head).
    #[serde(default)]
    pub aux_return: f32,
    /// Auxiliary stop-loss weight (0 = no head).
    #[serde(default)]
    pub aux_risk: f32,
    /// Multi-positive band (fraction of sigma_h), 0 = off.
    #[serde(default)]
    pub ambiguity_band: f64,
    /// Inverse-frequency class weights.
    #[serde(default)]
    pub class_balance: bool,
    /// Hyperparameter candidates; each is tried and the best on validation kept.
    #[serde(default)]
    pub grid: Vec<GridPoint>,
    /// Added to the run seed for this model (0: the shared per-fold seed, which
    /// every model of the main comparison uses; >0 for extra-seed replicates).
    #[serde(default)]
    pub seed_offset: u64,
}

/// Overrides for one hyperparameter candidate.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct GridPoint {
    /// Learning rate.
    pub learning_rate: Option<f32>,
    /// Weight decay.
    pub weight_decay: Option<f32>,
    /// Width override (`d_model` / `hidden`).
    pub d_model: Option<usize>,
}

fn default_head() -> String {
    "mlp".into()
}
fn default_pool() -> String {
    "last".into()
}
fn default_scoring() -> String {
    "cosine".into()
}
fn default_tau() -> f32 {
    0.1
}
fn yes() -> bool {
    true
}
fn d2() -> usize {
    2
}
fn d4() -> usize {
    4
}
fn d16() -> usize {
    16
}
fn d64() -> usize {
    64
}

impl ExperimentConfig {
    /// Read and validate a config; relative paths are resolved against its directory.
    pub fn load(path: &Path) -> anyhow_like::Result<Self> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| anyhow_like::err(format!("reading {}: {e}", path.display())))?;
        let mut cfg: Self = toml::from_str(&text)
            .map_err(|e| anyhow_like::err(format!("parsing {}: {e}", path.display())))?;
        let base = path.parent().unwrap_or(Path::new("."));
        if cfg.bars_dir.is_relative() {
            cfg.bars_dir = base.join(&cfg.bars_dir);
        }
        for m in &mut cfg.models {
            if let Some(d) = &m.external_dir
                && d.is_relative()
            {
                m.external_dir = Some(base.join(d));
            }
        }
        cfg.validate()?;
        Ok(cfg)
    }

    fn validate(&self) -> anyhow_like::Result<()> {
        let mut names = std::collections::HashSet::new();
        for m in &self.models {
            if !names.insert(&m.name) {
                return Err(anyhow_like::err(format!("duplicate model name {}", m.name)));
            }
            if !["mamba", "tcn", "mlp", "linear", "external"].contains(&m.encoder.as_str()) {
                return Err(anyhow_like::err(format!("{}: unknown encoder {}", m.name, m.encoder)));
            }
            if !["linear", "mlp", "contrastive"].contains(&m.head.as_str()) {
                return Err(anyhow_like::err(format!("{}: unknown head {}", m.name, m.head)));
            }
            if m.encoder == "external" && m.external_dir.is_none() {
                return Err(anyhow_like::err(format!("{}: external needs external_dir", m.name)));
            }
        }
        if !names.contains(&self.reference) {
            return Err(anyhow_like::err(format!("reference model {} not among models", self.reference)));
        }
        if !["expanding", "rolling"].contains(&self.folds.scheme.as_str()) {
            return Err(anyhow_like::err(format!("unknown fold scheme {}", self.folds.scheme)));
        }
        let fail = |what: &str| Err(anyhow_like::err(format!("invalid config: {what}")));
        let finite_nonneg = |x: f64| x.is_finite() && x >= 0.0;
        if self.window == 0 {
            return fail("window must be >= 1");
        }
        if !finite_nonneg(self.cost_bps) || !finite_nonneg(self.labels.kappa_bps) {
            return fail("costs must be finite and >= 0");
        }
        if !finite_nonneg(self.labels.margin_mult) || !(self.labels.stop_frac > 0.0 && self.labels.stop_frac.is_finite()) {
            return fail("margin_mult >= 0 and stop_frac > 0 required");
        }
        if self.labels.horizon == 0 {
            return fail("horizon must be >= 1");
        }
        if self.tau_grid.is_empty() || !self.tau_grid.iter().all(|&t| finite_nonneg(t)) || !finite_nonneg(self.policy_tau) {
            return fail("tau values must be finite and >= 0 (grid non-empty)");
        }
        let t = &self.train;
        if t.max_steps == 0 || t.batch_size == 0 || t.eval_every == 0 || t.eval_batch == 0 || t.patience == 0 {
            return fail("max_steps, batch_size, eval_every, eval_batch, patience must be >= 1");
        }
        if t.warmup_steps > t.max_steps {
            return fail("warmup_steps > max_steps");
        }
        if !(t.learning_rate > 0.0 && t.learning_rate.is_finite())
            || !(t.weight_decay >= 0.0 && t.weight_decay.is_finite() && t.grad_clip >= 0.0 && t.grad_clip.is_finite())
        {
            return fail("learning_rate > 0, weight_decay >= 0, grad_clip >= 0 required");
        }
        if self.folds.val_days == 0 || self.folds.cal_days == 0 || self.folds.first_test_year > self.folds.last_test_year {
            return fail("val_days, cal_days >= 1 and first_test_year <= last_test_year required");
        }
        for m in &self.models {
            if m.encoder == "external" {
                continue;
            }
            if !["last", "mean"].contains(&m.pooling.as_str()) || !["cosine", "bilinear"].contains(&m.scoring.as_str()) {
                return fail(&format!("{}: pooling must be last|mean and scoring cosine|bilinear", m.name));
            }
            let pos = |x: usize| x >= 1;
            if !(pos(m.d_model) && pos(m.layers) && pos(m.d_state) && m.d_state <= 64 && pos(m.expand) && pos(m.d_conv) && pos(m.hidden) && pos(m.embed_dim)) {
                return fail(&format!("{}: dimensions must be >= 1 (d_state <= 64)", m.name));
            }
            if !(m.temperature > 0.0 && m.temperature.is_finite() && m.supcon_temperature > 0.0 && m.supcon_temperature.is_finite()) {
                return fail(&format!("{}: temperatures must be finite and > 0", m.name));
            }
            if [m.supcon, m.aux_return, m.aux_risk].iter().any(|w| !(w.is_finite() && *w >= 0.0)) || !finite_nonneg(m.ambiguity_band) {
                return fail(&format!("{}: loss weights and ambiguity_band must be finite and >= 0", m.name));
            }
            for g in m.candidates() {
                if g.learning_rate.is_some_and(|v| !(v > 0.0 && v.is_finite()))
                    || g.weight_decay.is_some_and(|v| !(v >= 0.0 && v.is_finite()))
                    || g.d_model == Some(0)
                {
                    return fail(&format!("{}: invalid grid point {g:?}", m.name));
                }
            }
        }
        Ok(())
    }

    /// Everything a run's results depend on besides the data: this config and the
    /// source tree the binary was built from.
    pub fn signature(&self, dataset_version: &str) -> String {
        let cfg = serde_json::to_string(self).unwrap_or_default();
        let mut h: u64 = 0xcbf29ce484222325;
        for b in cfg.bytes().chain(dataset_version.bytes()).chain(env!("CLM_SOURCE_HASH").bytes()) {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        format!("{h:016x}")
    }

    /// The one-way policy cost as a fraction.
    pub fn kappa(&self) -> f64 {
        self.cost_bps * 1e-4
    }

    /// Fold configuration for clm-eval.
    pub fn fold_config(&self) -> FoldConfig {
        FoldConfig {
            scheme: if self.folds.scheme == "rolling" {
                Scheme::Rolling {
                    train_days: self.folds.train_days,
                }
            } else {
                Scheme::Expanding
            },
            first_test_year: self.folds.first_test_year,
            last_test_year: self.folds.last_test_year,
            val_days: self.folds.val_days,
            cal_days: self.folds.cal_days,
            horizon: self.labels.horizon,
            embargo: self.folds.embargo,
            min_train_days: self.folds.min_train_days,
        }
    }

    /// Label configuration for clm-data.
    pub fn label_config(&self) -> clm_data::LabelConfig {
        clm_data::LabelConfig {
            horizon: self.labels.horizon,
            kappa: self.labels.kappa_bps * 1e-4,
            margin_mult: self.labels.margin_mult,
            stop_frac: self.labels.stop_frac,
        }
    }
}

impl ModelSpec {
    /// Whether this model is trained here (not read from another program).
    pub fn is_internal(&self) -> bool {
        self.encoder != "external"
    }

    /// The candidates to try (at least one).
    pub fn candidates(&self) -> Vec<GridPoint> {
        if self.grid.is_empty() {
            vec![GridPoint::default()]
        } else {
            self.grid.clone()
        }
    }

    /// The model configuration for one candidate.
    pub fn model_config(&self, exp: &ExperimentConfig, features: usize, g: &GridPoint, seed: u64) -> ModelConfig {
        let width = g.d_model.unwrap_or(self.d_model);
        let pooling = if self.pooling == "mean" { Pooling::Mean } else { Pooling::Last };
        let encoder = match self.encoder.as_str() {
            "mamba" => EncoderConfig::Mamba(MambaConfig {
                input_dim: features,
                d_model: width,
                d_state: self.d_state,
                d_conv: self.d_conv,
                expand: self.expand,
                num_layers: self.layers,
                dt_rank: 0,
                dt_min: 1e-3,
                dt_max: 1e-1,
                pooling,
            }),
            "tcn" => EncoderConfig::Tcn(TcnConfig {
                input_dim: features,
                channels: width,
                kernel: self.d_conv,
                num_layers: self.layers,
                pooling,
            }),
            "mlp" => EncoderConfig::LastRow {
                input_dim: features,
                hidden: g.d_model.unwrap_or(self.hidden),
            },
            _ => EncoderConfig::LastRow {
                input_dim: features,
                hidden: 0,
            },
        };
        let head = match self.head.as_str() {
            "linear" => HeadConfig::Linear,
            "contrastive" => HeadConfig::Contrastive(ContrastiveConfig {
                embed_dim: self.embed_dim,
                hidden: self.hidden,
                temperature: if self.learned_temperature {
                    Temperature::Learned(self.temperature)
                } else {
                    Temperature::Fixed(self.temperature)
                },
                scoring: if self.scoring == "bilinear" {
                    Scoring::Bilinear
                } else {
                    Scoring::Cosine
                },
            }),
            _ => HeadConfig::Mlp { hidden: self.hidden },
        };
        let kappa = exp.labels.kappa_bps * 1e-4;
        ModelConfig {
            encoder,
            head,
            aux_return: self.aux_return > 0.0,
            aux_risk: self.aux_risk > 0.0,
            aux_hidden: self.hidden,
            action_attrs: action_attributes(exp.labels.horizon, (2.0 * kappa) as f32, kappa as f32),
            seed,
        }
    }

    /// The optimisation settings for one candidate.
    pub fn train_config(&self, exp: &ExperimentConfig, g: &GridPoint) -> TrainConfig {
        let t = &exp.train;
        TrainConfig {
            max_steps: t.max_steps,
            batch_size: t.batch_size,
            learning_rate: g.learning_rate.unwrap_or(t.learning_rate),
            weight_decay: g.weight_decay.unwrap_or(t.weight_decay),
            warmup_steps: t.warmup_steps,
            grad_clip: t.grad_clip,
            eval_every: t.eval_every,
            patience: t.patience,
            eval_batch: t.eval_batch,
            loss: LossConfig {
                decision: 1.0,
                supcon: self.supcon,
                supcon_temperature: self.supcon_temperature,
                ret: self.aux_return,
                risk: self.aux_risk,
                ret_clip: 5.0,
            },
            max_bad_steps: 20,
        }
    }

    /// Sampling settings.
    pub fn source_config(&self, exp: &ExperimentConfig, seed: u64) -> SourceConfig {
        SourceConfig {
            window: exp.window,
            class_balance: self.class_balance,
            ambiguity_band: self.ambiguity_band,
            seed,
        }
    }
}

/// A tiny error type so the CLI needs no extra dependency.
pub mod anyhow_like {
    /// CLI error.
    #[derive(Debug)]
    pub struct Error(pub String);

    impl std::fmt::Display for Error {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str(&self.0)
        }
    }

    impl std::error::Error for Error {}

    /// CLI result.
    pub type Result<T> = std::result::Result<T, Error>;

    /// Build an error.
    pub fn err(msg: impl Into<String>) -> Error {
        Error(msg.into())
    }

    /// Convert any displayable error.
    pub fn wrap<E: std::fmt::Display>(context: &str) -> impl FnOnce(E) -> Error + '_ {
        move |e| Error(format!("{context}: {e}"))
    }
}
