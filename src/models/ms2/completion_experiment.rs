//! Synthetic parent-relative molecular-completion experiment driver (MC4).
//!
//! This builds the numbers the design's "Primary prediction target" and "A
//! direct test still needs" sections ask for, in the **synthetic,
//! parent-relative, oracle-formula** setting only: each query conditions on a
//! held-out molecule's exact composition plus substructures cut from that
//! molecule with its own hydrogen counts
//! ([`extract_patterns`](super::completion_data::extract_patterns)). The
//! report says so in [`EXPERIMENT_SCOPE`]; nothing here measures
//! mass-derived formulas, real fragment evidence, calibration or physical
//! validity.
//!
//! * [`ExperimentArgs`] carries every CLI value; [`run`] executes the run.
//! * Arms: [`ExperimentArm::Full`] trains and generates with patterns,
//!   [`ExperimentArm::FormulaOnly`] trains with no patterns and generates
//!   with `condition_on_patterns = false` while its requests carry the same
//!   patterns (so the acceptance filter is identical), and
//!   [`ExperimentArm::Untrained`] skips training and generates like `Full`.
//! * The training seed stream (`extraction_seed`, `draw = epoch`) never
//!   touches the evaluation stream
//!   ([`eval_extraction_seed`], [`EVAL_DRAW]).
//! * Headline denominators count every validation molecule read: skipped
//!   molecules are misses with reason `out_of_domain:<reason>`. The kept
//!   (eligible) subgroup is reported separately.
//! * The validation export is the model-selection set (best teacher NLL
//!   picks the checkpoint); no untouched test set is used. The report states
//!   this in its scope string.
//! * Outputs are aggregates and 64-bit key hashes only: no molecule keys,
//!   no SMILES (the export data is CC BY-NC).

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::Instant;

use cubecl::prelude::Runtime;
use serde::{Deserialize, Serialize};

use crate::backend::Device;
use crate::error::{Error, Result};
use crate::tensor::ops::ms2::Ms2Constants;

use super::chem::{CHEMISTRY_VERSION, Composition, composition_mass};
use super::completion::contains_pattern;
use super::completion::stable_hash;
use super::completion_data::{
    COMPLETION_DATA_VERSION, CompletionSet, ExtractionConfig, FunctionalGroupConfig, PatternSource,
    SplitMix64, functional_group_patterns,
};
use super::completion_diagnostics::{DoomReason, classify_dead_end};
use super::completion_eval::{
    OutcomeCounts, QueryScore, RecoveryReport, recovery_report, score_query,
};
use super::completion_formula::{FormulaAllocation, FormulaPruning, MassQuery, run_mass_completion};
use super::completion_model::{
    COMPLETION_MODEL_VERSION, CompletionGenerationConfig, CompletionModelConfig, CompletionRequest,
    CompletionTrainConfig, CompletionTrainer, FormulaArtifacts, MAX_PATTERNS, PATTERN_SLOTS,
    QueryOutcome, SubstructureSemantics,
};
use super::contain::Containment;
use super::contract::candidate_status;
use super::dataset::ExportFile;
use super::experiment::{export_provenance, sha256_hex};
use super::functional_groups::{
    AROMATICITY_VERSION, FUNCTIONAL_GROUPS_VERSION, aromatic_rings, functional_groups,
};
use super::grammar::{CANONICAL_WORK_LIMIT, COMPLETION_GRAMMAR_VERSION, Limits, Token};
use super::graph::MolGraph;

/// What the numbers in [`ExperimentReport`] do and do not establish.
///
/// Synthetic parent-relative patterns, oracle exact composition,
/// identity-fold validation used for model selection; not a
/// mass-conditioned, calibrated or physically verified result.
pub const EXPERIMENT_SCOPE: &str = "synthetic parent-relative patterns, oracle exact composition, identity-fold validation used for model selection; not a mass-conditioned, calibrated or physically verified result";

/// What the numbers establish under functional-group patterns: the query
/// gives the molecule's composition and the functional groups (Ertl) of the
/// target with its own hydrogen counts.
pub const FUNCTIONAL_GROUPS_SCOPE: &str = "synthetic functional groups (Ertl) of the target with its own hydrogen counts, oracle exact composition, identity-fold validation used for model selection; not a mass-conditioned, calibrated or physically verified result";

/// Offset of the evaluation pattern stream from `--extraction-seed`.
///
/// Training draws use `draw = epoch` on the raw seed; evaluation uses
/// [`EVAL_DRAW`] on `seed + OFFSET`, a stream training never uses. See
/// [`eval_extraction_seed`].
pub const EVAL_SEED_OFFSET: u64 = 1_000_003;

/// Resampling index of the evaluation patterns (always the first draw).
pub const EVAL_DRAW: u64 = 0;

/// The evaluation pattern seed for an `--extraction-seed` base value.
///
/// `base.wrapping_add(EVAL_SEED_OFFSET)`: a seed stream training never
/// uses (training draws count epochs from 0 on the raw seed).
pub fn eval_extraction_seed(base: u64) -> u64 {
    base.wrapping_add(EVAL_SEED_OFFSET)
}

/// The generation config of an evaluation round: the containment and
/// identity work limits are the module defaults (100,000 each); the
/// sampling parameters are `trajectories` per query (`K`), the softmax
/// `temperature`, the RNG `seed` (`--gen-seed`), the shortlist size
/// `returned` (`--returned`), and whether the device conditions on the
/// patterns (`condition_on_patterns`: `false` is the formula-only control;
/// acceptance still requires each request's own patterns).
pub fn eval_generation_config(
    trajectories: u32,
    temperature: f32,
    seed: u64,
    returned: u32,
    condition_on_patterns: bool,
) -> CompletionGenerationConfig {
    CompletionGenerationConfig {
        trajectories,
        temperature,
        seed,
        returned,
        condition_on_patterns,
        ..CompletionGenerationConfig::default()
    }
}

/// Scope string for the mass arm: synthetic exact neutral masses.
pub const MASS_SCOPE_SUFFIX: &str = "; mass arm uses synthetic exact neutral masses (composition_mass of the target) with the same patterns and acceptance as the oracle arm; no learned formula prior";

/// Fit formula artifacts from kept training compositions and attach them.
///
/// Uses `EnumDomain::from_compositions(.., margin 0)` with `heavy_max`
/// clamped to the model's `max_atoms` and `RatioBounds::fit(..,
/// quantile_margin 0)`. The weights are untouched. The same function backs
/// `--attach-formula-artifacts`, so the weights stay byte-identical there.
pub fn fit_and_attach<R: Runtime>(
    trainer: &mut CompletionTrainer<R, f32>,
    compositions: &[Composition],
    max_atoms: u32,
    source: String,
) -> Result<()> {
    let artifacts = FormulaArtifacts::fit(compositions, max_atoms, 0, 0, source)?;
    trainer.set_formula_artifacts(artifacts);
    Ok(())
}

/// Median / 90th percentile / max of `values` as a [`CountDistribution`].
fn distribution_of(values: &[usize]) -> CountDistribution {
    if values.is_empty() {
        return CountDistribution {
            median: 0.0,
            p90: 0.0,
            max: 0,
        };
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let median = if sorted.len() % 2 == 1 {
        sorted[sorted.len() / 2] as f64
    } else {
        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) as f64 / 2.0
    };
    let rank = ((sorted.len() - 1) as f64 * 0.9).round() as usize;
    CountDistribution {
        median,
        p90: sorted[rank.min(sorted.len() - 1)] as f64,
        max: *sorted.last().unwrap_or(&0),
    }
}

/// Which conditioning a run trains and generates with.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExperimentArm {
    /// Train with [`ExtractionConfig::default`] patterns and generate with
    /// them.
    Full,
    /// Train with no patterns (`max_patterns = 0`); generate with
    /// `condition_on_patterns = false` while the requests carry the same
    /// patterns as the full arm, so the acceptance filter is identical.
    FormulaOnly,
    /// Skip training (fresh weights from `--seed`); generate like `Full`.
    Untrained,
}

impl ExperimentArm {
    /// The CLI spelling of the arm.
    pub fn as_str(self) -> &'static str {
        match self {
            ExperimentArm::Full => "full",
            ExperimentArm::FormulaOnly => "formula_only",
            ExperimentArm::Untrained => "untrained",
        }
    }

    /// Parse a CLI arm spelling.
    pub fn parse(text: &str) -> Result<Self> {
        match text {
            "full" => Ok(ExperimentArm::Full),
            "formula_only" => Ok(ExperimentArm::FormulaOnly),
            "untrained" => Ok(ExperimentArm::Untrained),
            other => Err(Error::config(format!(
                "ExperimentArm::parse: unknown arm {other:?} (expected full|formula_only|untrained)"
            ))),
        }
    }
}

/// Which formula source an evaluation uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FormulaSource {
    /// Oracle exact composition (the pre-task setting).
    Oracle,
    /// Mass-derived formula hypotheses (synthetic exact masses).
    Mass,
}

impl FormulaSource {
    /// The CLI spelling.
    pub fn as_str(self) -> &'static str {
        match self {
            FormulaSource::Oracle => "oracle",
            FormulaSource::Mass => "mass",
        }
    }

    /// Parse a CLI spelling.
    pub fn parse(text: &str) -> Result<Self> {
        match text {
            "oracle" => Ok(FormulaSource::Oracle),
            "mass" => Ok(FormulaSource::Mass),
            other => Err(Error::config(format!(
                "FormulaSource::parse: unknown source {other:?} (expected oracle|mass)"
            ))),
        }
    }
}

/// Which pattern source an evaluation uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum PatternArg {
    /// Random connected patches (the pre-task setting).
    #[default]
    Random,
    /// Functional groups (Ertl) of the target.
    FunctionalGroups,
}

impl PatternArg {
    /// The CLI spelling.
    pub fn as_str(self) -> &'static str {
        match self {
            PatternArg::Random => "random",
            PatternArg::FunctionalGroups => "functional_groups",
        }
    }

    /// Parse a CLI spelling.
    pub fn parse(text: &str) -> Result<Self> {
        match text {
            "random" => Ok(PatternArg::Random),
            "functional_groups" => Ok(PatternArg::FunctionalGroups),
            other => Err(Error::config(format!(
                "PatternArg::parse: unknown patterns {other:?} (expected random|functional_groups)"
            ))),
        }
    }
}

/// Every CLI value of one experiment run.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExperimentArgs {
    /// Train export file.
    pub train: PathBuf,
    /// Validation (model-selection) export file.
    pub validation: PathBuf,
    /// Output directory (created when missing).
    pub out: PathBuf,
    /// Run name.
    pub name: String,
    /// Conditioning arm.
    pub arm: ExperimentArm,
    /// Model preset (`small` or `base`).
    pub model: String,
    /// Domain atom cap override (`None` means the model preset's value).
    pub max_atoms: Option<usize>,
    /// Domain ring-closure cap override (`None` means the preset's value).
    pub max_closures: Option<usize>,
    /// Optimizer steps.
    pub steps: usize,
    /// Queries per training batch.
    pub batch: usize,
    /// AdamW base learning rate.
    pub lr: f32,
    /// AdamW decoupled weight decay.
    pub weight_decay: f32,
    /// Global gradient-norm clip (`None` disables it).
    pub grad_clip: Option<f32>,
    /// Seed for weight initialisation and the train shuffle.
    pub seed: u64,
    /// Report (device-read) the loss every N steps.
    pub report_every: usize,
    /// Teacher-forced validation NLL every N steps (`0` disables it).
    pub eval_every: usize,
    /// First N kept validation examples in the eval subset.
    pub eval_subset: usize,
    /// Trajectories per query (`K`, the total budget for the mass arm).
    pub trajectories: u32,
    /// Softmax temperature.
    pub temperature: f32,
    /// Shortlist size (`1..=25`).
    pub returned: u32,
    /// Queries per `generate` call.
    pub gen_batch: usize,
    /// Generation RNG seed.
    pub gen_seed: u64,
    /// Pattern-extraction sampling seed.
    pub extraction_seed: u64,
    /// First N kept train examples (`None` keeps all).
    pub limit_train: Option<usize>,
    /// First N kept validation examples (`None` keeps all).
    pub limit_validation: Option<usize>,
    /// Optional subgroups file (`{label: [molecule key, ...]}`).
    pub subgroups: Option<PathBuf>,
    /// Checkpoint to load (required with `eval_only`).
    pub load: Option<PathBuf>,
    /// Checkpoint to write (plus `<ckpt>.best` on validation improvement).
    pub save: Option<PathBuf>,
    /// Skip training; evaluate the loaded checkpoint.
    pub eval_only: bool,
    /// Bootstrap replicates per hit-rate interval.
    pub bootstrap: usize,
    /// Print live progress lines to stderr (step/loss/eval during training,
    /// queries during generation). No extra device reads.
    pub progress: bool,
    /// Formula source (`oracle` or `mass`; default `oracle`).
    #[serde(default = "default_formula_source")]
    pub formula_source: FormulaSource,
    /// Mass tolerance in tenths of a ppm for the mass arm (default 50).
    #[serde(default = "default_mass_ppm")]
    pub mass_ppm_tenths: u32,
    /// Mass uncertainty in micro-dalton for the mass arm (default 50).
    #[serde(default = "default_mass_uncertainty")]
    pub mass_uncertainty_uda: u32,
    /// Formula hypotheses selected per query for the mass arm (default 8).
    #[serde(default = "default_formula_hypotheses")]
    pub formula_hypotheses: u32,
    /// Formula-search pruning for the mass arm (`train_fit` by default;
    /// `chemical_only` drops the empirical train-fit bounds).
    #[serde(default)]
    pub formula_pruning: FormulaPruning,
    /// Trajectory allocation over the selected formulas for the mass arm
    /// (`equal` by default; `train_frequency` weights by training counts).
    #[serde(default)]
    pub formula_allocation: FormulaAllocation,
    /// Attach formula artifacts without training (`--attach-formula-artifacts`).
    #[serde(default)]
    pub attach_formula_artifacts: bool,
    /// Pattern source (`random` or `functional_groups`; default `random`).
    #[serde(default)]
    pub patterns: PatternArg,
    /// Training-time per-group keep probability in percent for functional
    /// groups (default 100).
    #[serde(default = "default_fg_keep")]
    pub fg_keep_percent: u8,
    /// Add aromatic rings without a marked atom as groups (`--fg-aromatic-rings`).
    #[serde(default)]
    pub fg_aromatic_rings: bool,
    /// How the supplied substructures constrain host acceptance
    /// (`--substructure-semantics contained|disjoint|complete`; default
    /// `contained`).
    #[serde(default)]
    pub substructure_semantics: SubstructureSemantics,
    /// Which evidence the model sees (`patterns` by default; `fingerprint`
    /// sees no patterns; `both` sees both).
    #[serde(default)]
    pub evidence: super::completion_fingerprint::Evidence,
    /// True-bits file for training (`--fp-train`).
    #[serde(default)]
    pub fp_train: Option<PathBuf>,
    /// True-bits file for validation (`--fp-validation`).
    #[serde(default)]
    pub fp_validation: Option<PathBuf>,
    /// Noise file for MIST-like sampling (`--fp-noise`).
    #[serde(default)]
    pub fp_noise: Option<PathBuf>,
    /// Training fingerprint mode (`exact` or `mist_like`; default `exact`).
    #[serde(default = "default_fp_train_mode")]
    pub fp_train_mode: super::completion_fingerprint::FingerprintMode,
    /// Evaluation fingerprint mode (`exact`, `mist_like` or `predicted`;
    /// default `exact`).
    #[serde(default)]
    pub fp_eval_mode: super::completion_fingerprint::FingerprintEvalMode,
    /// Which noise histogram set `mist_like` sampling draws from
    /// (`spectrum` or `molecule`; default `spectrum`).
    #[serde(default)]
    pub fp_noise_level: super::completion_fingerprint::FingerprintNoiseLevel,
    /// Fingerprint token threshold (`--fp-threshold`; default 0.1).
    #[serde(default = "default_fp_threshold")]
    pub fp_threshold: f32,
    /// Fingerprint slots (`--fp-slots`; default 128).
    #[serde(default = "default_fp_slots")]
    pub fp_slots: u32,
    /// Panel file whose identity groups are removed from training
    /// (`--exclude-identity-groups`).
    #[serde(default)]
    pub exclude_identity_groups: Option<PathBuf>,
    /// JSONL candidate dump (`--dump-candidates`; refused inside the repo).
    #[serde(default)]
    pub dump_candidates: Option<PathBuf>,
}

/// Default `--fp-train-mode` (`exact`).
fn default_fp_train_mode() -> super::completion_fingerprint::FingerprintMode {
    super::completion_fingerprint::FingerprintMode::Exact
}

/// Default `--fp-threshold` (0.1).
fn default_fp_threshold() -> f32 {
    0.1
}

/// Default `--fp-slots` (128).
fn default_fp_slots() -> u32 {
    super::completion_fingerprint::FINGERPRINT_SLOTS as u32
}

/// Default `formula_source` (`oracle`).
fn default_formula_source() -> FormulaSource {
    FormulaSource::Oracle
}

/// Default `--mass-ppm-tenths` (50).
fn default_mass_ppm() -> u32 {
    50
}

/// Default `--mass-uncertainty-uda` (50).
fn default_mass_uncertainty() -> u32 {
    50
}

/// Default `--formula-hypotheses` (8).
fn default_formula_hypotheses() -> u32 {
    8
}

/// Default `--fg-keep-percent` (100).
fn default_fg_keep() -> u8 {
    100
}

/// File provenance of one export used by a run.
#[derive(Clone, Debug, Serialize)]
pub struct ExportProvenance {
    /// File name (no directories).
    pub file: String,
    /// File size in bytes.
    pub bytes: u64,
    /// SHA-256 of the file bytes, lowercase hex.
    pub sha256: String,
    /// The export's `source` string.
    pub source: String,
    /// Allow-listed scalar header via
    /// [`export_provenance`](super::experiment::export_provenance) (no
    /// per-molecule rows).
    pub provenance: serde_json::Value,
}

/// Version constants behind a run.
#[derive(Clone, Debug, Serialize)]
pub struct VersionInfo {
    /// [`COMPLETION_GRAMMAR_VERSION`].
    pub grammar: String,
    /// [`COMPLETION_DATA_VERSION`].
    pub data: String,
    /// [`COMPLETION_MODEL_VERSION`].
    pub model: String,
    /// [`CHEMISTRY_VERSION`].
    pub chemistry: String,
    /// Pattern source (`random` or `functional_groups`).
    pub pattern_source: String,
    /// [`FUNCTIONAL_GROUPS_VERSION`] when functional groups are used.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub functional_groups: Option<String>,
    /// [`AROMATICITY_VERSION`] when functional groups are used.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aromaticity: Option<String>,
}

/// Domain limits behind a run.
#[derive(Clone, Debug, Serialize)]
pub struct DomainInfo {
    /// Maximum atoms per molecule.
    pub max_atoms: usize,
    /// Maximum ring closures per molecule.
    pub max_closures: usize,
    /// Canonicalization budget of the set build.
    pub work_limit: usize,
}

/// Read/kept/skipped accounting of both exports.
#[derive(Clone, Debug, Serialize)]
pub struct AccountingInfo {
    /// Molecules in the validation file.
    pub validation_read: usize,
    /// Kept validation examples actually evaluated.
    pub validation_kept: usize,
    /// Validation molecules left out, by domain reason.
    pub validation_skipped: BTreeMap<String, u64>,
    /// Kept validation examples past `--limit-validation` (not evaluated).
    pub limit_excluded: usize,
    /// Molecules in the train file.
    pub train_read: usize,
    /// Kept train examples actually trained on.
    pub train_kept: usize,
    /// Train molecules left out, by domain reason.
    pub train_skipped: BTreeMap<String, u64>,
    /// Kept train examples past `--limit-train` (not trained on).
    pub train_limit_excluded: usize,
    /// Evaluated validation examples whose trace is in the train set.
    pub overlap_strict: usize,
    /// Evaluated validation examples whose non-empty skeleton trace is in
    /// the train set.
    pub overlap_skeleton: usize,
    /// Distinct subgroup keys found in no export molecule.
    pub subgroup_unknown_keys: usize,
    /// Evaluated denominator per subgroup label.
    pub subgroup_denominators: BTreeMap<String, usize>,
    /// Training molecules removed by `--exclude-identity-groups` (`0`
    /// without the flag).
    #[serde(default)]
    pub train_excluded_identity_groups: usize,
}

/// One training-curve entry.
#[derive(Clone, Debug, Serialize)]
pub struct CurvePoint {
    /// Optimizer steps completed.
    pub step: u64,
    /// Reported pre-update loss (`None` on non-report steps).
    pub loss: Option<f32>,
    /// Eval-subset mean NLL per example (`None` when no eval ran here).
    pub eval_nll_per_example: Option<f64>,
    /// Eval-subset mean NLL per trace token (`None` when no eval ran).
    pub eval_nll_per_token: Option<f64>,
    /// Run-relative seconds at this entry.
    pub elapsed_seconds: f64,
}

/// Generation settings and cost of the final evaluation.
#[derive(Clone, Debug, Serialize)]
pub struct GenerationInfo {
    /// Sampling hyperparameters.
    pub config: CompletionGenerationConfig,
    /// Queries per `generate` call.
    pub gen_batch: usize,
    /// Wall seconds of the whole generation sweep.
    pub seconds: f64,
    /// Trajectories per second over the sweep.
    pub trajectories_per_second: f64,
}

/// One frequent functional-group signature.
#[derive(Clone, Debug, Serialize)]
pub struct TopGroup {
    /// Order-independent signature of the group's typed subgraph.
    pub signature: String,
    /// Queries containing the signature.
    pub queries: usize,
}

/// Aggregate pattern statistics of the evaluation patterns.
#[derive(Clone, Debug, Serialize)]
pub struct EvalPatternStats {
    /// `extraction_seed + 1_000_003` (a stream training never uses).
    pub seed: u64,
    /// Always 0 (the first draw).
    pub draw: u64,
    /// Pattern source (`random` or `functional_groups`).
    #[serde(default)]
    pub pattern_source: String,
    /// Mean pattern count per evaluated query (mean groups for FG).
    pub mean_patterns: f64,
    /// Mean atoms per pattern (mean atoms per group for FG).
    pub mean_pattern_atoms: f64,
    /// Mean covered parent-atom fraction per evaluated query.
    pub mean_coverage: f64,
    /// Fraction of eligible queries with no functional group (`None` for random).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fraction_no_group: Option<f64>,
    /// Fraction of eligible queries with a truncated list (`None` for random).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fraction_truncated: Option<f64>,
    /// Fraction of eligible queries with a dropped oversized group
    /// (`None` for random).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fraction_dropped_oversized: Option<f64>,
    /// The 20 most frequent group signatures with query counts
    /// (`None` for random; signatures are generic chemistry, not data rows).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_groups: Option<Vec<TopGroup>>,
    /// Queries whose untruncated functional-group list exceeds the encoder
    /// limits (8 patterns / 24 atoms), so the model saw the seeded fitting
    /// subset while acceptance used the full list. Only nonzero with
    /// `--patterns functional_groups --substructure-semantics complete`.
    #[serde(default)]
    pub full_list_acceptance_queries: usize,
}

/// Dead-end cause diagnostics over the generation evaluation: every
/// `no_valid_action` trajectory classified by
/// [`classify_dead_end`](super::completion_diagnostics::classify_dead_end).
///
/// `mean_doomed_at`, `median_doomed_at`, `mean_steps_after_doomed` and
/// `median_steps_after_doomed` run over the doomed trajectories only (those
/// with a `Some` reason); they are `0.0` when no trajectory was doomed.
/// `fraction_doomed_at_root` is the fraction of all dead-end trajectories with
/// `doomed_at <= 2` (the root choice).
#[derive(Clone, Debug, Serialize)]
pub struct DeadEndDiagnostics {
    /// Dead-end (`no_valid_action`) trajectories in total.
    pub total: usize,
    /// Doomed by the hydrogen bound.
    pub hydrogen_bound: usize,
    /// Doomed with no open site.
    pub no_open_site: usize,
    /// Doomed with open valence but no atoms left.
    pub open_valence_without_atoms: usize,
    /// Doomed by the valence bound.
    pub valence_bound: usize,
    /// Dead ends no necessary condition explains.
    pub unexplained: usize,
    /// Mean tokens of the dead-end traces.
    pub mean_dead_end_step: f64,
    /// Median tokens of the dead-end traces.
    pub median_dead_end_step: f64,
    /// Mean doom step over the doomed trajectories (`0.0` when none).
    pub mean_doomed_at: f64,
    /// Median doom step over the doomed trajectories (`0.0` when none).
    pub median_doomed_at: f64,
    /// Mean `dead_end_step - doomed_at` over the doomed trajectories.
    pub mean_steps_after_doomed: f64,
    /// Median `dead_end_step - doomed_at` over the doomed trajectories.
    pub median_steps_after_doomed: f64,
    /// Fraction of dead ends with `doomed_at <= 2`.
    pub fraction_doomed_at_root: f64,
}

/// Finished-but-rejected-by-containment breakdown: composition-complete graphs
/// that miss at least one required pattern, split into graphs missing every
/// pattern versus only some (at least one pattern contained), and by the
/// cheap type-count necessary condition
/// ([`type_counts_cover`](super::completion_diagnostics::type_counts_cover)):
/// a pattern can be contained only if every atom type count of the molecule
/// covers the pattern's.
#[derive(Clone, Debug, Serialize)]
pub struct RejectedContainmentDetail {
    /// Rejected-by-containment trajectories in total.
    pub total: usize,
    /// Graphs containing no required pattern.
    pub missing_all: usize,
    /// Graphs containing at least one required pattern.
    pub missing_some: usize,
    /// `missing_all / total` (`0.0` when empty).
    pub fraction_missing_all: f64,
    /// Rejected graphs where some required pattern fails the type-count
    /// cover test (a type-level mask could have prevented the miss).
    pub type_shortfall: usize,
    /// Rejected graphs where every required pattern passes the type-count
    /// cover test (the miss is one of connectivity).
    pub type_sufficient: usize,
    /// Fraction of queries with at least one rejected trajectory whose
    /// rejected trajectories are all `type_shortfall` (`0.0` when no query
    /// has a rejected trajectory).
    pub fraction_queries_all_shortfall: f64,
    /// Fraction of all finished trajectories that replay to a valid complete
    /// graph and pass the type-count cover test for every required pattern
    /// (how often the model satisfies the necessary condition at all; `0.0`
    /// when no finished trajectory yields a valid complete graph).
    pub finished_type_cover_fraction: f64,
}

/// Trajectory-outcome fractions over the eligible queries.
///
/// Each fraction is the mean over queries of `count / trajectories`.
#[derive(Clone, Debug, Serialize)]
pub struct DiagnosticsInfo {
    /// Mean accepted identities per query (before the `returned` cut).
    pub mean_distinct: f64,
    /// Mean unresolved trajectories per query (the `unresolved` shortlist,
    /// never hits).
    pub mean_unresolved: f64,
    /// Queries with an empty shortlist.
    pub zero_candidate_queries: usize,
    /// Device status FINISHED.
    pub finished_fraction: f64,
    /// Device status `no_valid_action`.
    pub dead_end_fraction: f64,
    /// Device status TRUNCATED.
    pub truncated_fraction: f64,
    /// Host exact replay rejected.
    pub rejected_replay_fraction: f64,
    /// A required pattern is not contained.
    pub rejected_containment_fraction: f64,
    /// A containment check hit its node limit.
    pub containment_unresolved_fraction: f64,
    /// An identity comparison hit its work limit.
    pub identity_unresolved_fraction: f64,
    /// Finished candidates with extra functional groups under
    /// `complete_functional_groups` (a split of the rejected-containment
    /// total; 0 under the other rules).
    pub rejected_extra_groups: u32,
    /// Finished candidates lacking a group under
    /// `complete_functional_groups` (a split of the rejected-containment
    /// total; 0 under the other rules).
    pub rejected_missing_groups: u32,
    /// Fraction of finished trajectories whose replayed graph passes the
    /// `contained` rule (computed for every run, regardless of the active
    /// semantics).
    pub pass_contained_fraction: f64,
    /// Fraction of finished trajectories whose replayed graph passes the
    /// `disjoint_occurrences` rule (computed for every run).
    pub pass_disjoint_fraction: f64,
    /// Fraction of finished trajectories whose replayed graph passes the
    /// `complete_functional_groups` rule (computed for every run).
    pub pass_complete_fraction: f64,
    /// Dead-end cause breakdown (reason counts sum to the dead-end total).
    pub dead_ends: DeadEndDiagnostics,
    /// Finished-but-rejected-by-containment breakdown.
    pub rejected_containment_detail: RejectedContainmentDetail,
    /// Mean fingerprint tokens per query (`0.0` without fingerprint evidence).
    #[serde(default)]
    pub fp_tokens_mean: f64,
    /// Max fingerprint tokens over queries (`0` without fingerprint evidence).
    #[serde(default)]
    pub fp_tokens_max: usize,
    /// Mean entries dropped by the slot limit (`0.0` without evidence).
    #[serde(default)]
    pub fp_entries_dropped_mean: f64,
    /// Mean true bits missing from the tokens (`None` for exact mode or
    /// without fingerprint evidence).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fp_true_missing_mean: Option<f64>,
    /// Mean false tokens (`None` for exact mode or without evidence).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fp_false_mean: Option<f64>,
    /// Empirical true-bit retention rate of the noise file at the run
    /// threshold and noise level (`None` without `--fp-noise`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fp_noise_retention_rate: Option<f64>,
    /// Empirical mean false tokens per spectrum of the noise file (`None`
    /// without `--fp-noise`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fp_noise_mean_false_tokens: Option<f64>,
    /// Noise histogram set the run sampled (`spectrum` or `molecule`;
    /// `--fp-noise-level`, `None` without `--fp-noise`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fp_noise_level: Option<String>,
}

/// One `predictions.jsonl` line: a validation molecule read.
///
/// The molecule key appears only as [`stable_hash`] (`key_hash`): no keys,
/// no SMILES. `source_index` is the export row position (rows may share one
/// key, so it disambiguates them).
#[derive(Clone, Debug, Serialize)]
pub struct PredictionLine {
    /// 64-bit hash of the molecule key.
    pub key_hash: u64,
    /// Position in the validation export's molecule list.
    pub source_index: usize,
    /// `eligible`, `out_of_domain:<reason>` or `limit_excluded`.
    pub status: String,
    /// Whether the query was generated and scored.
    pub eligible: bool,
    /// Identity group of the export row.
    pub identity_group: u64,
    /// Whether the target trace is in the train set (`None` for skipped).
    pub identity_in_train: Option<bool>,
    /// Heavy-atom count of the export row.
    pub atoms: usize,
    /// Evaluation pattern count (`0` when not evaluated).
    pub pattern_count: usize,
    /// Evaluation pattern atoms in total (`0` when not evaluated).
    pub pattern_atoms: usize,
    /// 1-based rank of the target (`None` when absent).
    pub rank: Option<u32>,
    /// 1-based skeleton rank (`None` when absent).
    pub skeleton_rank: Option<u32>,
    /// Shortlist length returned.
    pub candidates: u32,
    /// Accepted identities before the `returned` cut.
    pub distinct: u32,
    /// The outcome accounting (`None` when not evaluated).
    pub outcome: Option<OutcomeCounts>,
    /// Trajectory count behind the top candidate (`None` when empty).
    pub top_samples: Option<u32>,
    /// Best trace log-probability of the top candidate.
    pub top_best_log_prob: Option<f32>,
    /// Trajectory count behind the target's candidate (`None` when absent).
    pub target_samples: Option<u32>,
    /// Best trace log-probability of the target's candidate.
    pub target_best_log_prob: Option<f32>,
    /// Formulas joined by the enumerator (`None` for the oracle arm).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub formula_joined: Option<usize>,
    /// Survivors after the model-domain filter (`None` for oracle).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub formula_after_domain: Option<usize>,
    /// Survivors after the substructure bound (`None` for oracle).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub formula_after_substructures: Option<usize>,
    /// Survivors after completability (`None` for oracle).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub formula_after_completability: Option<usize>,
    /// Formulas selected (`None` for oracle).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub formula_selected: Option<usize>,
    /// Formulas sampled (`None` for oracle).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub formula_sampled: Option<usize>,
    /// Stage the true formula reached (`None` for oracle or skipped).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub true_formula_stage: Option<String>,
    /// Whether the train-fit bounds exclude the true formula, checked
    /// directly with `EnumDomain::contains` and the `RatioBounds::passes_*`
    /// predicates (`None` for oracle or skipped). An empirical modelling
    /// choice, not chemistry.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub true_excluded_by_train_fit: Option<bool>,
    /// Whether the true formula is absent from the actual search's joined
    /// rows (`None` for oracle or skipped).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub true_absent_from_search: Option<bool>,
    /// Functional groups found on the target (`None` when not evaluated).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub groups_found: Option<usize>,
    /// Whether the evaluation pattern list was truncated (`None` when not evaluated).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncated: Option<bool>,
    /// Pre-check failure reason when the query was infeasible (no trajectory
    /// sampled; the query scores as a miss). `None` when the query ran.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub infeasible_reason: Option<String>,
}

/// Distribution summary (median, 90th percentile, max).
#[derive(Clone, Debug, Serialize)]
pub struct CountDistribution {
    /// Median.
    pub median: f64,
    /// 90th percentile.
    pub p90: f64,
    /// Maximum.
    pub max: usize,
}

/// Formula-search diagnostics for the mass arm (`None` for oracle).
///
/// Fractions run over all eligible queries; stage fractions are monotone
/// (joined `>=` after_domain `>=` after_substructures `>=`
/// after_completability `>=` selected `>=` sampled). There is no learned
/// formula ranker.
#[derive(Clone, Debug, Serialize)]
pub struct FormulaSearchDiagnostics {
    /// The pruning the mass arm enumerated under.
    pub pruning: String,
    /// The trajectory allocation over the selected formulas.
    pub allocation: String,
    /// Fraction whose true formula was joined by the enumerator.
    pub fraction_joined: f64,
    /// Fraction surviving the model-domain filter.
    pub fraction_after_domain: f64,
    /// Fraction surviving the substructure bound.
    pub fraction_after_substructures: f64,
    /// Fraction surviving completability.
    pub fraction_after_completability: f64,
    /// Fraction selected.
    pub fraction_selected: f64,
    /// Fraction sampled.
    pub fraction_sampled: f64,
    /// Eligible queries whose true formula the train-fit bounds exclude
    /// (direct `EnumDomain::contains` / `RatioBounds::passes_*` check).
    pub true_excluded_by_train_fit_queries: usize,
    /// Fraction of eligible queries whose true formula the train-fit bounds
    /// exclude (direct check).
    pub fraction_true_excluded_by_train_fit: f64,
    /// Eligible queries whose true formula is absent from the actual
    /// search's joined rows.
    pub true_absent_from_search_queries: usize,
    /// Fraction of eligible queries whose true formula is absent from the
    /// actual search's joined rows.
    pub fraction_true_absent_from_search: f64,
    /// Joined-count distribution.
    pub joined: CountDistribution,
    /// Surviving-count (after completability) distribution.
    pub surviving: CountDistribution,
    /// Selected-count distribution.
    pub selected: CountDistribution,
    /// True-formula residual-rank distribution (1-based, over queries where
    /// the true formula survived to selection).
    pub true_rank: CountDistribution,
    /// Queries truncated by the hypotheses cap.
    pub truncated_queries: usize,
    /// Queries where the enumerator reported exhaustion.
    pub search_exhausted_queries: usize,
    /// Recovery when the true formula was sampled.
    pub recovery_sampled_true: RecoveryReport,
    /// Recovery when the true formula was not sampled.
    pub recovery_sampled_false: RecoveryReport,
}

/// The serializable result of [`run`]: everything `report.json` holds
/// except the per-query lines (those go to `predictions.jsonl`).
///
/// [`QueryOutcome`] implements neither `Clone` nor `Debug`, so this report
/// is serializable and comparable but not cloneable.
#[derive(Serialize)]
pub struct ExperimentReport {
    /// Run name.
    pub name: String,
    /// Arm spelling.
    pub arm: String,
    /// [`EXPERIMENT_SCOPE`].
    pub scope: String,
    /// Every CLI value, echoed.
    pub args: ExperimentArgs,
    /// Train export provenance.
    pub train_provenance: ExportProvenance,
    /// Validation export provenance.
    pub validation_provenance: ExportProvenance,
    /// Version constants.
    pub versions: VersionInfo,
    /// Model hyperparameters.
    pub model_config: CompletionModelConfig,
    /// Training hyperparameters.
    pub train_config: CompletionTrainConfig,
    /// Domain limits.
    pub domain: DomainInfo,
    /// Read/kept/skipped accounting.
    pub accounting: AccountingInfo,
    /// Model preset name (`small` or `base`).
    pub model_preset: String,
    /// First N kept validation examples in the eval subset.
    pub eval_subset: usize,
    /// Training curve (loss reports and eval NLLs).
    pub curve: Vec<CurvePoint>,
    /// Which checkpoint the final generation used.
    pub checkpoint_evaluated: String,
    /// Step of the best validation NLL (`None` when no eval ran).
    pub best_step: Option<u64>,
    /// Best eval-subset mean NLL per example.
    pub best_eval_nll_per_example: Option<f64>,
    /// Generation settings and cost.
    pub generation: GenerationInfo,
    /// Evaluation pattern statistics.
    pub eval_patterns: EvalPatternStats,
    /// Recovery over every validation molecule read (skips are misses).
    pub metrics_all: RecoveryReport,
    /// Recovery over the eligible (kept, evaluated) queries.
    pub metrics_eligible: RecoveryReport,
    /// Recovery per subgroup label, intersected with the read set.
    pub metrics_subgroups: BTreeMap<String, RecoveryReport>,
    /// Recovery over eligible queries with the target trace in train.
    pub metrics_identity_in_train_true: RecoveryReport,
    /// Recovery over eligible queries with the target trace absent.
    pub metrics_identity_in_train_false: RecoveryReport,
    /// Trajectory-outcome diagnostics over the eligible queries.
    pub diagnostics: DiagnosticsInfo,
    /// Train wall seconds.
    pub train_seconds: f64,
    /// Train wall seconds per 100 optimizer steps (`0` when untrained).
    pub seconds_per_100_steps: f64,
    /// Formula-search diagnostics for the mass arm (`None` for oracle).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub formula_search: Option<FormulaSearchDiagnostics>,
    /// The prediction lines, in validation file order (also written to
    /// `predictions.jsonl`).
    pub predictions: Vec<PredictionLine>,
    /// The scored outcomes of the eligible queries in file order (kept in
    /// memory for test hooks; never written to disk).
    #[serde(skip)]
    pub outcomes: Vec<QueryOutcome>,
}

/// Lexically normalize an absolute path (resolve `.` and `..` without
/// touching the filesystem).
fn normalize_absolute(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for part in path.components() {
        use std::path::Component;
        match part {
            Component::ParentDir => {
                out.pop();
            }
            Component::CurDir => {}
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// One validation molecule read, with its evaluation status.
struct ValidationRow {
    /// Stable molecule key (memory only; the report keeps its hash).
    key: String,
    /// Position in the validation export's molecule list: the fingerprint
    /// sidecar lookup key and the request-identity distinguisher. Molecule
    /// keys may repeat across rows (stereoisomers share them), so nothing
    /// here maps rows by key.
    source_index: usize,
    /// Identity group of the export row.
    identity_group: u64,
    /// Heavy-atom count of the export row.
    atoms: usize,
    /// Index into the validation [`CompletionSet`] when kept.
    example: Option<usize>,
    /// Domain skip reason when not kept and not limit-excluded.
    skip_reason: Option<String>,
    /// Kept but past `--limit-validation`.
    limit_excluded: bool,
}

/// Classify every validation molecule in file order.
///
/// Kept rows point at the set's examples by `source_index`; skipped rows
/// carry the same reason [`CompletionSet::from_export`] counted.
/// Consistency with `set` is asserted positionally: every kept example's
/// `source_index` names its export row. Canonical traces are recomputed
/// for skipped molecules only.
fn classify_validation(
    file: &ExportFile,
    set: &CompletionSet,
    limits: Limits,
    work_limit: usize,
    limit: Option<usize>,
) -> Result<(Vec<ValidationRow>, usize)> {
    let mut by_source: HashMap<usize, usize> = HashMap::with_capacity(set.examples.len());
    for (i, example) in set.examples.iter().enumerate() {
        by_source.insert(example.source_index, i);
    }
    let keep = limit.map_or(set.examples.len(), |n| n.min(set.examples.len()));
    let mut rows = Vec::with_capacity(file.molecules.len());
    let mut kept_traces: Vec<Vec<Token>> = Vec::new();
    let mut kept_ptr = 0usize;
    let mut limit_excluded = 0usize;
    for (source_index, mol) in file.molecules.iter().enumerate() {
        let atoms = mol.atoms.len();
        if let Some(&index) = by_source.get(&source_index) {
            let expected = &set.examples[kept_ptr];
            if expected.source_index != source_index {
                return Err(Error::config(format!(
                    "classify_validation: kept example order diverged at export row {source_index}"
                )));
            }
            kept_traces.push(expected.trace.clone());
            kept_ptr += 1;
            if index < keep {
                rows.push(ValidationRow {
                    key: mol.key.clone(),
                    source_index,
                    identity_group: mol.identity_group,
                    atoms,
                    example: Some(index),
                    skip_reason: None,
                    limit_excluded: false,
                });
            } else {
                limit_excluded += 1;
                rows.push(ValidationRow {
                    key: mol.key.clone(),
                    source_index,
                    identity_group: mol.identity_group,
                    atoms,
                    example: Some(index),
                    skip_reason: None,
                    limit_excluded: true,
                });
            }
            continue;
        }
        let reason = skip_reason_of(mol, &kept_traces, limits, work_limit)?;
        rows.push(ValidationRow {
            key: mol.key.clone(),
            source_index,
            identity_group: mol.identity_group,
            atoms,
            example: None,
            skip_reason: Some(reason),
            limit_excluded: false,
        });
    }
    if kept_ptr != set.examples.len() {
        return Err(Error::config(format!(
            "classify_validation: {} kept examples but only {kept_ptr} matched export rows",
            set.examples.len()
        )));
    }
    Ok((rows, limit_excluded))
}

/// The [`CompletionSet::from_export`] skip reason of one molecule known to
/// be absent from the kept set, mirroring that constructor's checks in
/// order. `kept_traces` holds the canonical traces of the earlier kept
/// examples for the duplicate test.
fn skip_reason_of(
    mol: &super::dataset::ExportMolecule,
    kept_traces: &[Vec<Token>],
    limits: Limits,
    work_limit: usize,
) -> Result<String> {
    let graph = match mol.graph() {
        Ok(graph) => graph,
        Err(_) => return Ok("graph_error".to_string()),
    };
    if !graph.is_connected() {
        return Ok("not_connected".to_string());
    }
    if graph.atoms().len() > limits.max_atoms() {
        return Ok("too_many_atoms".to_string());
    }
    if graph.ring_closures() > limits.max_closures() {
        return Ok("too_many_closures".to_string());
    }
    let canonical = match super::grammar::canonical_trace(&graph, limits, work_limit) {
        Ok(canonical) => canonical,
        Err(e) => {
            if e.to_string().contains("canonicalization_budget_exceeded") {
                return Ok("canonicalization_work_limit".to_string());
            }
            return Err(Error::config(format!(
                "classify_validation: molecule {} canonicalization failed: {e}",
                mol.key
            )));
        }
    };
    if kept_traces.contains(&canonical.trace) {
        return Ok("duplicate_identity".to_string());
    }
    Err(Error::config(format!(
        "classify_validation: molecule {} replays as kept but is absent from the set",
        mol.key
    )))
}

/// Entry keys of an export file in order (`<key>|<identity_group>`, the
/// `bits` sidecar's `keys_by_molecule` convention): the order binding behind
/// [`FingerprintStore::assert_keys_match`](super::completion_fingerprint::FingerprintStore::assert_keys_match).
fn export_entry_keys(file: &ExportFile) -> Vec<String> {
    file.molecules
        .iter()
        .map(|m| format!("{}|{}", m.key, m.identity_group))
        .collect()
}

/// Read one export file: bytes for the SHA-256, the parsed file, and its
/// provenance record.
fn read_export(path: &Path) -> Result<(ExportFile, ExportProvenance)> {
    let bytes = std::fs::read(path)?;
    let text = std::str::from_utf8(&bytes).map_err(|e| {
        Error::config(format!(
            "read_export: {} is not valid UTF-8: {e}",
            path.display()
        ))
    })?;
    let file = ExportFile::from_json(text)?;
    let raw: serde_json::Value = serde_json::from_str(text)?;
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string());
    let sha256 = sha256_hex(&bytes);
    let provenance = export_provenance(&raw, &name, &sha256);
    Ok((
        file.clone(),
        ExportProvenance {
            file: name,
            bytes: bytes.len() as u64,
            sha256,
            source: file.source.clone(),
            provenance,
        },
    ))
}

/// Read a panel file (`fp_mist_panel.json`, molecule-export schema) as a
/// validation export: synthetic [`ExportFile`] with empty spectra plus the
/// per-molecule predicted probabilities (`fp_pred_mean`) and true bits
/// (`fp_true`) by panel position (the [`CompletionExample::source_index`]
/// lookup key).
///
/// Returns the file, its provenance, `panel_pred[source_index]` and
/// `panel_true[source_index]`.
#[allow(clippy::type_complexity)]
fn load_panel_validation(
    path: &Path,
) -> Result<(
    ExportFile,
    ExportProvenance,
    HashMap<usize, Vec<(u16, f32)>>,
    HashMap<usize, Vec<u16>>,
)> {
    let bytes = std::fs::read(path)?;
    let text = std::str::from_utf8(&bytes).map_err(|e| {
        Error::config(format!(
            "load_panel_validation: {} is not valid UTF-8: {e}",
            path.display()
        ))
    })?;
    let raw: serde_json::Value = serde_json::from_str(text)?;
    let fingerprint = raw
        .get("fingerprint")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    if fingerprint != "morgan4096" {
        return Err(Error::config(format!(
            "load_panel_validation: fingerprint {fingerprint:?} is not \"morgan4096\""
        )));
    }
    let schema_version = raw
        .get("schema_version")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| {
            Error::config(format!(
                "load_panel_validation: {} has no schema_version",
                path.display()
            ))
        })?;
    if schema_version != 1 {
        return Err(Error::config(format!(
            "load_panel_validation: schema_version {schema_version} is not 1"
        )));
    }
    let molecules = raw.get("molecules").ok_or_else(|| {
        Error::config(format!(
            "load_panel_validation: {} has no molecules",
            path.display()
        ))
    })?;
    let list = molecules.as_array().ok_or_else(|| {
        Error::config(format!(
            "load_panel_validation: {} molecules is not a list",
            path.display()
        ))
    })?;
    let mut out_molecules = Vec::with_capacity(list.len());
    let mut panel_pred: HashMap<usize, Vec<(u16, f32)>> = HashMap::new();
    let mut panel_true: HashMap<usize, Vec<u16>> = HashMap::new();
    for (i, mol) in list.iter().enumerate() {
        let key = mol
            .get("key")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                Error::config(format!("load_panel_validation: molecule {i} has no key"))
            })?
            .to_string();
        let identity_group = mol
            .get("identity_group")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| {
                Error::config(format!(
                    "load_panel_validation: molecule {i} has no identity_group"
                ))
            })?;
        let fold_identity = mol
            .get("fold_identity")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32;
        let atoms_v = mol.get("atoms").ok_or_else(|| {
            Error::config(format!("load_panel_validation: molecule {i} has no atoms"))
        })?;
        let atoms: Vec<u8> = atoms_v
            .as_array()
            .ok_or_else(|| {
                Error::config(format!(
                    "load_panel_validation: molecule {i} atoms is not a list"
                ))
            })?
            .iter()
            .map(|v| {
                let n = v.as_u64().ok_or_else(|| {
                    Error::config(format!(
                        "load_panel_validation: molecule {i} atoms holds a non-integer"
                    ))
                })?;
                if n > u8::MAX as u64 {
                    return Err(Error::config(format!(
                        "load_panel_validation: molecule {i} atom type {n} does not fit u8"
                    )));
                }
                Ok(n as u8)
            })
            .collect::<Result<Vec<u8>>>()?;
        let bonds_v = mol.get("bonds").ok_or_else(|| {
            Error::config(format!("load_panel_validation: molecule {i} has no bonds"))
        })?;
        let bonds: Vec<(usize, usize, u8)> = bonds_v
            .as_array()
            .ok_or_else(|| {
                Error::config(format!(
                    "load_panel_validation: molecule {i} bonds is not a list"
                ))
            })?
            .iter()
            .map(|b| {
                let triple = b.as_array().ok_or_else(|| {
                    Error::config(format!(
                        "load_panel_validation: molecule {i} bonds entries must be lists"
                    ))
                })?;
                if triple.len() != 3 {
                    return Err(Error::config(format!(
                        "load_panel_validation: molecule {i} bonds entries must have 3 fields"
                    )));
                }
                let a = triple[0].as_u64().ok_or_else(|| {
                    Error::config(format!(
                        "load_panel_validation: molecule {i} bonds endpoint is not an integer"
                    ))
                })?;
                let c = triple[1].as_u64().ok_or_else(|| {
                    Error::config(format!(
                        "load_panel_validation: molecule {i} bonds endpoint is not an integer"
                    ))
                })?;
                let o = triple[2].as_u64().ok_or_else(|| {
                    Error::config(format!(
                        "load_panel_validation: molecule {i} bonds order is not an integer"
                    ))
                })?;
                if a > usize::MAX as u64 || c > usize::MAX as u64 {
                    return Err(Error::config(format!(
                        "load_panel_validation: molecule {i} bonds endpoint does not fit usize"
                    )));
                }
                if o > u8::MAX as u64 {
                    return Err(Error::config(format!(
                        "load_panel_validation: molecule {i} bonds order {o} does not fit u8"
                    )));
                }
                Ok((a as usize, c as usize, o as u8))
            })
            .collect::<Result<Vec<(usize, usize, u8)>>>()?;
        out_molecules.push(super::dataset::ExportMolecule {
            key,
            identity_group,
            fold_identity,
            atoms,
            bonds,
            spectra: Vec::new(),
        });
        if let Some(pred) = mol.get("fp_pred_mean") {
            let entries = pred
                .as_array()
                .ok_or_else(|| {
                    Error::config(format!(
                        "load_panel_validation: molecule {i} fp_pred_mean is not a list"
                    ))
                })?
                .iter()
                .map(|entry| {
                    let pair = entry.as_array().ok_or_else(|| {
                        Error::config(format!(
                            "load_panel_validation: molecule {i} fp_pred_mean entries must be pairs"
                        ))
                    })?;
                    if pair.len() != 2 {
                        return Err(Error::config(format!(
                            "load_panel_validation: molecule {i} fp_pred_mean entry holds {} fields (needs 2)",
                            pair.len()
                        )));
                    }
                    let bit = pair[0].as_u64().ok_or_else(|| {
                        Error::config(format!(
                            "load_panel_validation: molecule {i} fp_pred_mean bit is not an integer"
                        ))
                    })?;
                    if bit >= super::completion_fingerprint::FINGERPRINT_BITS as u64 {
                        return Err(Error::config(format!(
                            "load_panel_validation: molecule {i} fp_pred_mean bit {bit} is past {}",
                            super::completion_fingerprint::FINGERPRINT_BITS
                        )));
                    }
                    let prob = pair[1].as_f64().ok_or_else(|| {
                        Error::config(format!(
                            "load_panel_validation: molecule {i} fp_pred_mean probability is not a number"
                        ))
                    })?;
                    if !(prob.is_finite() && prob > 0.0 && prob <= 1.0) {
                        return Err(Error::config(format!(
                            "load_panel_validation: molecule {i} fp_pred_mean probability {prob} is not in (0, 1]"
                        )));
                    }
                    Ok((bit as u16, prob as f32))
                })
                .collect::<Result<Vec<(u16, f32)>>>()?;
            panel_pred.insert(i, entries);
        }
        if let Some(truth) = mol.get("fp_true") {
            let bits = truth
                .as_array()
                .ok_or_else(|| {
                    Error::config(format!(
                        "load_panel_validation: molecule {i} fp_true is not a list"
                    ))
                })?
                .iter()
                .map(|v| {
                    let n = v.as_u64().ok_or_else(|| {
                        Error::config(format!(
                            "load_panel_validation: molecule {i} fp_true holds a non-integer"
                        ))
                    })?;
                    if n >= super::completion_fingerprint::FINGERPRINT_BITS as u64 {
                        return Err(Error::config(format!(
                            "load_panel_validation: molecule {i} fp_true bit {n} is past {}",
                            super::completion_fingerprint::FINGERPRINT_BITS
                        )));
                    }
                    Ok(n as u16)
                })
                .collect::<Result<Vec<u16>>>()?;
            panel_true.insert(i, bits);
        }
    }
    let chemistry = raw
        .get("chemistry")
        .and_then(|v| v.as_str())
        .unwrap_or(super::chem::CHEMISTRY_VERSION)
        .to_string();
    let source = raw
        .get("source")
        .and_then(|v| v.as_str())
        .unwrap_or("panel")
        .to_string();
    let file = ExportFile {
        schema_version: 1,
        chemistry,
        rdkit: raw
            .get("rdkit")
            .and_then(|v| v.as_str())
            .unwrap_or("panel")
            .to_string(),
        source: source.clone(),
        seed: 0,
        n_raw: 0,
        spectra_per_molecule: 0,
        skipped_spectra: BTreeMap::new(),
        subset: "validation".to_string(),
        molecules: out_molecules,
    };
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string());
    let sha256 = sha256_hex(&bytes);
    let provenance = export_provenance(
        &serde_json::json!({"source": source}),
        &name,
        &sha256,
    );
    Ok((
        file,
        ExportProvenance {
            file: name,
            bytes: bytes.len() as u64,
            sha256,
            source,
            provenance,
        },
        panel_pred,
        panel_true,
    ))
}
/// Build evaluation fingerprints for `eval_subset` positions (indices into
/// `eval_indices`, which index `validation_set.examples`).
///
/// True bits come from the validation store by `source_index`, or from the
/// panel `panel_true` map when the validation is a panel file. `Exact` gives
/// probability-1 bits; `MistLike` samples from `noise` (required) with the
/// evaluation stream (`seed`, key, `EVAL_DRAW`); `Predicted` uses the panel's
/// `fp_pred_mean` (required, so the validation must be a panel file).
/// `threshold` drops low entries in the latter two modes.
fn build_eval_fingerprints(
    validation_set: &CompletionSet,
    eval_indices: &[usize],
    eval_subset: &[usize],
    store: Option<&super::completion_fingerprint::FingerprintStore>,
    noise: Option<&super::completion_fingerprint::FingerprintNoise>,
    noise_level: super::completion_fingerprint::FingerprintNoiseLevel,
    panel_pred: &HashMap<usize, Vec<(u16, f32)>>,
    panel_true: &HashMap<usize, Vec<u16>>,
    mode: super::completion_fingerprint::FingerprintEvalMode,
    threshold: f32,
    seed: u64,
) -> Result<Vec<super::completion_fingerprint::SparseFingerprint>> {
    use super::completion_fingerprint::{FingerprintEvalMode, SparseFingerprint};
    let mut out = Vec::with_capacity(eval_subset.len());
    for &pos in eval_subset {
        let example = &validation_set.examples[eval_indices[pos]];
        let true_bits: Vec<u16> = if let Some(bits) = panel_true.get(&example.source_index) {
            bits.clone()
        } else if let Some(store) = store {
            store.get_by_index(example.source_index)?.to_vec()
        } else {
            return Err(Error::config(
                "build_eval_fingerprints: no validation fingerprint source (need --fp-validation or a panel --validation)".to_string(),
            ));
        };
        match mode {
            FingerprintEvalMode::Exact => out.push(SparseFingerprint::from_bits(&true_bits)?),
            FingerprintEvalMode::MistLike => {
                let noise = noise.ok_or_else(|| {
                    Error::config(
                        "build_eval_fingerprints: mist_like eval needs --fp-noise".to_string(),
                    )
                })?;
                out.push(noise.sample_at_level(
                    &true_bits,
                    seed,
                    &example.key,
                    EVAL_DRAW,
                    threshold,
                    noise_level,
                )?);
            }
            FingerprintEvalMode::Predicted => {
                let pred = panel_pred.get(&example.source_index).ok_or_else(|| {
                    Error::config(
                        "build_eval_fingerprints: predicted eval needs a panel --validation with fp_pred_mean".to_string(),
                    )
                })?;
                out.push(SparseFingerprint::from_probabilities(pred, threshold)?);
            }
        }
    }
    Ok(out)
}
fn check_out_dir(out: &Path) -> Result<()> {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let docs = normalize_absolute(&manifest.join("docs"));
    let src = normalize_absolute(&manifest.join("src"));
    let cwd = std::env::current_dir()?;
    let absolute = if out.is_absolute() {
        out.to_path_buf()
    } else {
        cwd.join(out)
    };
    let normalized = normalize_absolute(&absolute);
    for forbidden in [&docs, &src] {
        if normalized.starts_with(forbidden) {
            return Err(Error::config(format!(
                "refusing to write inside {}: {}",
                forbidden.display(),
                out.display()
            )));
        }
    }
    Ok(())
}

/// Resolve `absolute` through its existing parent chain (symlinks), so a
/// path that only lexically looks outside the repository cannot smuggle
/// data-derived output inside it. Non-existing trailing components are kept
/// lexically; when nothing canonicalizes, the lexical normalization is kept.
fn resolve_existing_chain(absolute: &Path) -> PathBuf {
    use std::ffi::OsString;
    let mut base = absolute.to_path_buf();
    let mut rest: Vec<OsString> = Vec::new();
    while !base.exists() {
        match base.file_name() {
            Some(name) => {
                rest.push(name.to_os_string());
                base.pop();
            }
            None => break,
        }
    }
    match std::fs::canonicalize(&base) {
        Ok(mut canon) => {
            for comp in rest.iter().rev() {
                canon.push(comp);
            }
            canon
        }
        Err(_) => normalize_absolute(absolute),
    }
}

/// Check that `--dump-candidates` does not point inside the repository: the
/// file holds data-derived structures. The existing parent chain is resolved
/// first, so a path through a symlink into the repository is refused too.
fn check_dump_path(path: &Path) -> Result<()> {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root = resolve_existing_chain(&manifest);
    let cwd = std::env::current_dir()?;
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        cwd.join(path)
    };
    let resolved = resolve_existing_chain(&absolute);
    if resolved.starts_with(&root) {
        return Err(Error::config(format!(
            "refusing to write data-derived structures inside the repository: {}",
            path.display()
        )));
    }
    Ok(())
}

/// One subgroup label with its molecule-key set.
type Subgroup = (String, HashSet<String>);

/// Subgroup labels to molecule-key sets, dropping `_`-prefixed labels.
fn read_subgroups(path: &Path, read_keys: &HashSet<&str>) -> Result<(Vec<Subgroup>, usize)> {
    let text = std::fs::read_to_string(path)?;
    let raw: serde_json::Value = serde_json::from_str(&text)?;
    let object = raw.as_object().ok_or_else(|| {
        Error::config(format!(
            "read_subgroups: {} is not a JSON object",
            path.display()
        ))
    })?;
    let mut groups = Vec::new();
    let mut unknown = HashSet::new();
    for (label, keys) in object {
        if label.starts_with('_') {
            continue;
        }
        let list = keys.as_array().ok_or_else(|| {
            Error::config(format!(
                "read_subgroups: label {label:?} is not an array of molecule keys"
            ))
        })?;
        let mut set = HashSet::new();
        for key in list {
            let key = key.as_str().ok_or_else(|| {
                Error::config(format!(
                    "read_subgroups: label {label:?} holds a non-string key"
                ))
            })?;
            if key.starts_with('_') {
                continue;
            }
            if !read_keys.contains(key) {
                unknown.insert(key.to_string());
            }
            set.insert(key.to_string());
        }
        groups.push((label.clone(), set));
    }
    let unknown_count = unknown.len();
    Ok((groups, unknown_count))
}

/// Untruncated functional-group list of `parent` as typed induced subgraphs:
/// every `functional-groups-ertl-v1` group (plus each unmarked aromatic ring
/// when `aromatic_rings_as_groups`, exactly the candidate set behind
/// [`functional_group_patterns`](super::completion_data::functional_group_patterns)),
/// with no keep-probability drop, no oversized drop and no encoder fit.
/// Each graph is the induced subgraph on ascending atoms (deterministic;
/// group identity is order-independent). Used as the acceptance pattern set
/// with `CompleteFunctionalGroups`, where evaluation queries carry the full
/// list even when it exceeds the encoder limits.
fn full_functional_group_list(
    parent: &MolGraph,
    aromatic_rings_as_groups: bool,
) -> Result<Vec<MolGraph>> {
    let groups = functional_groups(parent).map_err(|e| {
        Error::config(format!("full_functional_group_list: functional groups failed: {e}"))
    })?;
    let marked_union: HashSet<usize> =
        groups.iter().flat_map(|g| g.atoms.iter().copied()).collect();
    let mut atom_lists: Vec<Vec<usize>> = groups.into_iter().map(|g| g.atoms).collect();
    if aromatic_rings_as_groups {
        for ring in aromatic_rings(parent) {
            if ring.iter().any(|a| marked_union.contains(a)) {
                continue;
            }
            let mut atoms = ring;
            atoms.sort_unstable();
            atoms.dedup();
            if !atom_lists.contains(&atoms) {
                atom_lists.push(atoms);
            }
        }
        atom_lists.sort_by_key(|atoms| atoms[0]);
    }
    let mut out = Vec::with_capacity(atom_lists.len());
    for atoms in &atom_lists {
        out.push(parent.induced(atoms).map_err(|e| {
            Error::config(format!("full_functional_group_list: induced group failed: {e}"))
        })?);
    }
    Ok(out)
}

/// A zero-score miss: no candidates, no ranks, no trajectories.
fn miss_score() -> QueryScore {
    QueryScore {
        rank: None,
        skeleton_rank: None,
        outcome_counts: OutcomeCounts {
            candidates: 0,
            distinct: 0,
            trajectories: 0,
            finished: 0,
            dead_end: 0,
            truncated: 0,
            rejected_replay: 0,
            rejected_containment: 0,
            containment_unresolved: 0,
            identity_unresolved: 0,
            other_status: 0,
        },
    }
}

/// Mean of `pick(counts) / trajectories` over scores (0 trajectories give
/// 0), the same convention as
/// [`recovery_report`](super::completion_eval::recovery_report).
fn mean_fraction(scores: &[QueryScore], pick: fn(&OutcomeCounts) -> u32) -> f64 {
    if scores.is_empty() {
        return 0.0;
    }
    scores
        .iter()
        .map(|score| {
            let counts = &score.outcome_counts;
            if counts.trajectories == 0 {
                0.0
            } else {
                f64::from(pick(counts)) / f64::from(counts.trajectories)
            }
        })
        .sum::<f64>()
        / scores.len() as f64
}

/// Fraction of finished trajectories passing one acceptance rule, pooled
/// over `outcomes` (`0.0` when nothing finished). The three per-rule pass
/// counts are recorded under every semantics, so one run shows the cost of
/// each rule.
fn finished_pass_fraction(outcomes: &[QueryOutcome], pick: fn(&QueryOutcome) -> u32) -> f64 {
    let finished: u32 = outcomes.iter().map(|o| o.finished).sum();
    if finished == 0 {
        return 0.0;
    }
    let pass: u32 = outcomes.iter().map(pick).sum();
    f64::from(pass) / f64::from(finished)
}

/// Mean of a slice (`0.0` when empty).
fn mean_of(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

/// Median of a slice (`0.0` when empty; the average of the two middle values
/// when even).
fn median_of(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 1 {
        sorted[mid]
    } else {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    }
}

/// Aggregate dead-end causes over the generation evaluation.
///
/// Every `no_valid_action` trajectory of every `outcome` is classified with
/// [`classify_dead_end`] against its query's composition; a trace that fails
/// to replay counts as unexplained.
fn aggregate_dead_ends(
    outcomes: &[QueryOutcome],
    compositions: &[Composition],
    limits: Limits,
) -> DeadEndDiagnostics {
    let mut hydrogen_bound = 0usize;
    let mut no_open_site = 0usize;
    let mut open_valence_without_atoms = 0usize;
    let mut valence_bound = 0usize;
    let mut unexplained = 0usize;
    let mut dead_steps: Vec<f64> = Vec::new();
    let mut doomed_ats: Vec<f64> = Vec::new();
    let mut delays: Vec<f64> = Vec::new();
    let mut early = 0usize;
    let mut total = 0usize;
    for (outcome, composition) in outcomes.iter().zip(compositions.iter()) {
        for sampled in &outcome.sampled {
            if sampled.status & candidate_status::NO_VALID_ACTION == 0 {
                continue;
            }
            total += 1;
            let cause = classify_dead_end(&sampled.trace, composition, limits);
            let (reason, dead_step, doomed_at) = match &cause {
                Ok(cause) => (cause.reason, cause.dead_end_step, cause.doomed_at),
                Err(_) => (None, sampled.trace.len(), None),
            };
            dead_steps.push(dead_step as f64);
            match reason {
                Some(DoomReason::HydrogenBound) => hydrogen_bound += 1,
                Some(DoomReason::NoOpenSite) => no_open_site += 1,
                Some(DoomReason::OpenValenceWithoutAtoms) => {
                    open_valence_without_atoms += 1;
                }
                Some(DoomReason::ValenceBound) => valence_bound += 1,
                None => unexplained += 1,
            }
            if let Some(at) = doomed_at {
                doomed_ats.push(at as f64);
                delays.push(dead_step as f64 - at as f64);
                if at <= 2 {
                    early += 1;
                }
            }
        }
    }
    DeadEndDiagnostics {
        total,
        hydrogen_bound,
        no_open_site,
        open_valence_without_atoms,
        valence_bound,
        unexplained,
        mean_dead_end_step: mean_of(&dead_steps),
        median_dead_end_step: median_of(&dead_steps),
        mean_doomed_at: mean_of(&doomed_ats),
        median_doomed_at: median_of(&doomed_ats),
        mean_steps_after_doomed: mean_of(&delays),
        median_steps_after_doomed: median_of(&delays),
        fraction_doomed_at_root: if total == 0 {
            0.0
        } else {
            early as f64 / total as f64
        },
    }
}

/// Aggregate finished-but-rejected-by-containment trajectories: replay every
/// FINISHED trajectory exactly as
/// [`generate`](super::completion_model::CompletionModel::generate) does and
/// keep the composition-complete graphs the containment filter rejects.
///
/// `patterns_per_query` holds the required substructures of each query;
/// `node_limit` is the same containment work limit generation used. A graph
/// containing none of its patterns counts as missing all, one containing at
/// least one (but missing another, since it was rejected) as missing some.
/// The type-count split uses
/// [`type_counts_cover`](super::completion_diagnostics::type_counts_cover):
/// a rejected graph whose every required pattern passes the cover test is a
/// connectivity miss (`type_sufficient`), otherwise a `type_shortfall`.
fn aggregate_rejected_containment(
    outcomes: &[QueryOutcome],
    compositions: &[Composition],
    patterns_per_query: &[Vec<MolGraph>],
    limits: Limits,
    node_limit: usize,
) -> RejectedContainmentDetail {
    let mut total = 0usize;
    let mut missing_all = 0usize;
    let mut missing_some = 0usize;
    let mut type_shortfall = 0usize;
    let mut type_sufficient = 0usize;
    let mut queries_with_rejected = 0usize;
    let mut queries_all_shortfall = 0usize;
    let mut finished_valid = 0usize;
    let mut finished_cover = 0usize;
    for ((outcome, composition), patterns) in outcomes
        .iter()
        .zip(compositions.iter())
        .zip(patterns_per_query.iter())
    {
        let mut query_rejected = 0usize;
        let mut query_shortfall = 0usize;
        for sampled in &outcome.sampled {
            if sampled.status & candidate_status::FINISHED == 0 {
                continue;
            }
            let end = match super::grammar::replay_exact(&sampled.trace, limits, *composition) {
                Ok(end) => end,
                Err(_) => continue,
            };
            if !end.stopped() || !end.is_complete() {
                continue;
            }
            let graph = match end.graph() {
                Ok(graph) => graph,
                Err(_) => continue,
            };
            if !graph.is_connected() || graph.composition() != *composition {
                continue;
            }
            finished_valid += 1;
            let covers = patterns
                .iter()
                .all(|p| super::completion_diagnostics::type_counts_cover(p, &graph));
            if covers {
                finished_cover += 1;
            }
            let mut first_failure: Option<bool> = None;
            for pattern in patterns {
                match contains_pattern(&graph, pattern, node_limit) {
                    Containment::Contained => {}
                    Containment::NotContained => {
                        first_failure = Some(false);
                        break;
                    }
                    Containment::WorkLimit => {
                        first_failure = Some(true);
                        break;
                    }
                }
            }
            if first_failure != Some(false) {
                continue;
            }
            total += 1;
            query_rejected += 1;
            let mut any_contained = false;
            for pattern in patterns {
                if contains_pattern(&graph, pattern, node_limit) == Containment::Contained {
                    any_contained = true;
                    break;
                }
            }
            if any_contained {
                missing_some += 1;
            } else {
                missing_all += 1;
            }
            if covers {
                type_sufficient += 1;
            } else {
                type_shortfall += 1;
                query_shortfall += 1;
            }
        }
        if query_rejected > 0 {
            queries_with_rejected += 1;
            if query_shortfall == query_rejected {
                queries_all_shortfall += 1;
            }
        }
    }
    RejectedContainmentDetail {
        total,
        missing_all,
        missing_some,
        fraction_missing_all: if total == 0 {
            0.0
        } else {
            missing_all as f64 / total as f64
        },
        type_shortfall,
        type_sufficient,
        fraction_queries_all_shortfall: if queries_with_rejected == 0 {
            0.0
        } else {
            queries_all_shortfall as f64 / queries_with_rejected as f64
        },
        finished_type_cover_fraction: if finished_valid == 0 {
            0.0
        } else {
            finished_cover as f64 / finished_valid as f64
        },
    }
}

/// Per-query mass evaluation detail (mass arm only).
#[derive(Clone, Debug)]
struct MassEvalDetail {
    /// Joined count.
    joined: usize,
    /// After domain.
    after_domain: usize,
    /// After substructures.
    after_substructures: usize,
    /// After completability.
    after_completability: usize,
    /// Selected.
    selected: usize,
    /// Sampled.
    sampled: usize,
    /// True-formula stage.
    stage: String,
    /// True-formula rank (1-based, when selected).
    true_rank: Option<usize>,
    /// Truncated.
    truncated: bool,
    /// Enumerator exhausted.
    exhausted: bool,
    /// True formula sampled.
    sampled_true: bool,
    /// True formula excluded by the train-fit bounds (direct
    /// `EnumDomain::contains` / `RatioBounds::passes_*` check).
    excluded_by_train_fit: bool,
    /// True formula absent from the actual search's joined rows.
    absent_from_search: bool,
}

/// Whether the train-fit bounds exclude a composition, checked directly:
/// the domain must contain it and every ratio stage must pass it.
fn excluded_by_train_fit_direct(
    artifacts: &super::completion_model::FormulaArtifacts,
    c: &Composition,
) -> bool {
    if !artifacts.domain.contains(c) {
        return true;
    }
    let bounds = &artifacts.bounds;
    if !bounds.passes_cap(c) {
        return true;
    }
    if !bounds.passes_rare_max(c) || !bounds.passes_rare_min(c) {
        return true;
    }
    for k in 0..super::formula_enum::RATIO_FEATURES.len() {
        if !bounds.passes_ratio(k, c) {
            return true;
        }
    }
    if !bounds.passes_ratio_dbe(c) {
        return true;
    }
    false
}

/// True-formula stage when not selected/sampled: recompute the filter chain
/// for the true composition.
fn mass_true_stage(
    true_comp: &Composition,
    patterns: &[MolGraph],
    limits: Limits,
    max_atoms: u32,
    result: &super::completion_formula::MassCompletionResult,
) -> String {
    // Heavy-domain check (model cap).
    let mut heavy: u32 = 0;
    for e in [0usize, 2, 3, 4, 5, 6, 7, 8, 9] {
        heavy += u32::from(true_comp[e]);
    }
    if heavy > max_atoms {
        // Joined but cut by the model domain? If it was joined at all, the
        // stage is `joined`; otherwise `absent`. The enumerator respects its
        // own domain, not the model cap, so a heavy-over-cap true could still
        // have joined. Report `joined` when the enumerator joined anything
        // (conservative), else `absent`.
        if result.formula_search.joined > 0 {
            return "joined".to_string();
        }
        return "absent".to_string();
    }
    // Substructure bound.
    {
        let mut ok = true;
        for pattern in patterns {
            let mut need_heavy = [0u16; 10];
            let mut need_h: u32 = 0;
            for &id in pattern.atoms() {
                if let Some(t) = super::chem::atom_type(id) {
                    if t.element == super::chem::HYDROGEN {
                        need_h += 1;
                    } else {
                        need_heavy[t.element] += 1;
                        need_h += u32::from(t.hydrogens);
                    }
                }
            }
            for e in [0usize, 2, 3, 4, 5, 6, 7, 8, 9] {
                if need_heavy[e] > true_comp[e] {
                    ok = false;
                }
            }
            if need_h > u32::from(true_comp[super::chem::HYDROGEN]) {
                ok = false;
            }
        }
        if !ok {
            return "after_domain".to_string();
        }
    }
    // Completability.
    {
        let state = super::grammar::TraceState::new_exact(limits, *true_comp);
        if !state.feasibility().all() {
            return "after_substructures".to_string();
        }
    }
    // Passed all filters but not selected (truncated).
    "after_completability".to_string()
}

/// Check every accounting identity of the run.
fn check_accounting(
    report: &ExperimentReport,
    predictions: usize,
    per_query_trajectories: u32,
) -> Result<()> {
    let a = &report.accounting;
    let skipped: usize = a.validation_skipped.values().sum::<u64>() as usize;
    if a.validation_read != a.validation_kept + skipped + a.limit_excluded {
        return Err(Error::config(format!(
            "accounting: validation_read {} != kept {} + skipped {skipped} + limit_excluded {}",
            a.validation_read, a.validation_kept, a.limit_excluded
        )));
    }
    let train_skipped: usize = a.train_skipped.values().sum::<u64>() as usize;
    if a.train_read
        != a.train_kept + train_skipped + a.train_limit_excluded + a.train_excluded_identity_groups
    {
        return Err(Error::config(format!(
            "accounting: train_read {} != kept {} + skipped {train_skipped} + limit_excluded {} + excluded {}",
            a.train_read, a.train_kept, a.train_limit_excluded, a.train_excluded_identity_groups
        )));
    }
    if predictions != a.validation_read {
        return Err(Error::config(format!(
            "accounting: {predictions} prediction lines for {} molecules read",
            a.validation_read
        )));
    }
    for line in &report.predictions {
        if line.eligible {
            let counts = line.outcome.as_ref().ok_or_else(|| {
                Error::config("accounting: an eligible line has no outcome counts".to_string())
            })?;
            // Generation deliberately returns zero trajectories for
            // infeasible requests (no device work): expect zero with the
            // serialized reason, otherwise exactly K. The query scores as a
            // miss either way.
            if line.infeasible_reason.is_some() {
                if counts.trajectories != 0 {
                    return Err(Error::config(format!(
                        "accounting: infeasible query ran {} trajectories, want 0",
                        counts.trajectories
                    )));
                }
            } else if counts.trajectories != per_query_trajectories {
                return Err(Error::config(format!(
                    "accounting: query trajectories {} != K {per_query_trajectories}",
                    counts.trajectories
                )));
            }
            let total = counts.finished + counts.dead_end + counts.truncated + counts.other_status;
            if total != counts.trajectories {
                return Err(Error::config(format!(
                    "accounting: query status classes sum to {total}, want {}",
                    counts.trajectories
                )));
            }
        }
    }
    // The identity accounting of every evaluated outcome: finished
    // trajectories are rejected, certified (before the cut) or unresolved.
    // The shortlist cut only drops whole certified identities, so the
    // candidate samples are a lower bound of the certified total while the
    // unresolved shortlist is never cut.
    for (pos, outcome) in report.outcomes.iter().enumerate() {
        // Zero-trajectory infeasible outcomes are expected (see above);
        // otherwise every eligible query ran exactly K trajectories.
        if outcome.infeasible.is_some() {
            if outcome.trajectories != 0 {
                return Err(Error::config(format!(
                    "accounting: eligible query {pos} is infeasible but ran {} trajectories, want 0",
                    outcome.trajectories
                )));
            }
        } else if outcome.trajectories != per_query_trajectories {
            return Err(Error::config(format!(
                "accounting: eligible query {pos} ran {} trajectories, want K {per_query_trajectories}",
                outcome.trajectories
            )));
        }
        let status = outcome.finished + outcome.dead_end + outcome.truncated + outcome.other_status;
        if status != outcome.trajectories {
            return Err(Error::config(format!(
                "accounting: eligible query {pos} status classes sum to {status}, want {}",
                outcome.trajectories
            )));
        }
        let candidate_samples: u32 = outcome.candidates.iter().map(|c| c.samples).sum();
        let unresolved_samples: u32 = outcome.unresolved.iter().map(|c| c.samples).sum();
        if outcome.identity_unresolved != unresolved_samples {
            return Err(Error::config(format!(
                "accounting: eligible query {pos} counts {} identity-unresolved over {unresolved_samples} unresolved samples",
                outcome.identity_unresolved
            )));
        }
        let accounted = outcome
            .rejected_replay
            .saturating_add(outcome.rejected_containment)
            .saturating_add(outcome.containment_unresolved)
            .saturating_add(candidate_samples)
            .saturating_add(unresolved_samples);
        if accounted > outcome.finished {
            return Err(Error::config(format!(
                "accounting: eligible query {pos} accounts {accounted} finished trajectories over {}",
                outcome.finished
            )));
        }
        if outcome.distinct < outcome.candidates.len() as u32 {
            return Err(Error::config(format!(
                "accounting: eligible query {pos} reports {} distinct identities under {} candidates",
                outcome.distinct,
                outcome.candidates.len()
            )));
        }
    }
    for (name, metrics) in [
        ("all", &report.metrics_all),
        ("eligible", &report.metrics_eligible),
    ] {
        check_rates(name, metrics)?;
    }
    check_rates(
        "identity_in_train_true",
        &report.metrics_identity_in_train_true,
    )?;
    check_rates(
        "identity_in_train_false",
        &report.metrics_identity_in_train_false,
    )?;
    for (label, metrics) in report.metrics_subgroups.iter() {
        check_rates(&format!("subgroup:{label}"), metrics)?;
    }
    Ok(())
}

/// One group's `hits <= queries` check behind [`check_accounting`].
fn check_rates(name: &str, metrics: &RecoveryReport) -> Result<()> {
    for rate in [
        &metrics.top1,
        &metrics.top10,
        &metrics.top25,
        &metrics.skeleton_top25,
    ] {
        if rate.hits > rate.queries {
            return Err(Error::config(format!(
                "accounting: {name} has {} hits over {} queries",
                rate.hits, rate.queries
            )));
        }
    }
    Ok(())
}

/// Mass-arm accounting: like [`check_accounting`] but the executed
/// trajectories may sit below the requested total (the floor remainder is
/// unused), so per-query `trajectories <= K` with the status classes still
/// summing exactly. Every other identity is exact; exit is non-zero on any
/// failure.
fn check_accounting_mass(
    report: &ExperimentReport,
    predictions: usize,
    per_query_trajectories: u32,
) -> Result<()> {
    let a = &report.accounting;
    let skipped: usize = a.validation_skipped.values().sum::<u64>() as usize;
    if a.validation_read != a.validation_kept + skipped + a.limit_excluded {
        return Err(Error::config(format!(
            "accounting: validation_read {} != kept {} + skipped {skipped} + limit_excluded {}",
            a.validation_read, a.validation_kept, a.limit_excluded
        )));
    }
    let train_skipped: usize = a.train_skipped.values().sum::<u64>() as usize;
    if a.train_read
        != a.train_kept + train_skipped + a.train_limit_excluded + a.train_excluded_identity_groups
    {
        return Err(Error::config(format!(
            "accounting: train_read {} != kept {} + skipped {train_skipped} + limit_excluded {} + excluded {}",
            a.train_read, a.train_kept, a.train_limit_excluded, a.train_excluded_identity_groups
        )));
    }
    if predictions != a.validation_read {
        return Err(Error::config(format!(
            "accounting: {predictions} prediction lines for {} molecules read",
            a.validation_read
        )));
    }
    for line in &report.predictions {
        if line.eligible {
            let counts = line.outcome.as_ref().ok_or_else(|| {
                Error::config("accounting: an eligible line has no outcome counts".to_string())
            })?;
            if counts.trajectories > per_query_trajectories {
                return Err(Error::config(format!(
                    "accounting: query trajectories {} over K {per_query_trajectories}",
                    counts.trajectories
                )));
            }
            let total = counts.finished + counts.dead_end + counts.truncated + counts.other_status;
            if total != counts.trajectories {
                return Err(Error::config(format!(
                    "accounting: query status classes sum to {total}, want {}",
                    counts.trajectories
                )));
            }
        }
    }
    for (pos, outcome) in report.outcomes.iter().enumerate() {
        if outcome.trajectories > per_query_trajectories {
            return Err(Error::config(format!(
                "accounting: eligible query {pos} ran {} trajectories, over K {per_query_trajectories}",
                outcome.trajectories
            )));
        }
        let status = outcome.finished + outcome.dead_end + outcome.truncated + outcome.other_status;
        if status != outcome.trajectories {
            return Err(Error::config(format!(
                "accounting: eligible query {pos} status classes sum to {status}, want {}",
                outcome.trajectories
            )));
        }
        let candidate_samples: u32 = outcome.candidates.iter().map(|c| c.samples).sum();
        let unresolved_samples: u32 = outcome.unresolved.iter().map(|c| c.samples).sum();
        if outcome.identity_unresolved != unresolved_samples {
            return Err(Error::config(format!(
                "accounting: eligible query {pos} counts {} identity-unresolved over {unresolved_samples} unresolved samples",
                outcome.identity_unresolved
            )));
        }
        let accounted = outcome
            .rejected_replay
            .saturating_add(outcome.rejected_containment)
            .saturating_add(outcome.containment_unresolved)
            .saturating_add(candidate_samples)
            .saturating_add(unresolved_samples);
        if accounted > outcome.finished {
            return Err(Error::config(format!(
                "accounting: eligible query {pos} accounts {accounted} finished trajectories over {}",
                outcome.finished
            )));
        }
        if outcome.distinct < outcome.candidates.len() as u32 {
            return Err(Error::config(format!(
                "accounting: eligible query {pos} reports {} distinct identities under {} candidates",
                outcome.distinct,
                outcome.candidates.len()
            )));
        }
    }
    for (name, metrics) in [
        ("all", &report.metrics_all),
        ("eligible", &report.metrics_eligible),
    ] {
        check_rates(name, metrics)?;
    }
    check_rates(
        "identity_in_train_true",
        &report.metrics_identity_in_train_true,
    )?;
    check_rates(
        "identity_in_train_false",
        &report.metrics_identity_in_train_false,
    )?;
    for (label, metrics) in report.metrics_subgroups.iter() {
        check_rates(&format!("subgroup:{label}"), metrics)?;
    }
    if let Some(fs) = &report.formula_search {
        for name in [
            fs.fraction_joined,
            fs.fraction_after_domain,
            fs.fraction_after_substructures,
            fs.fraction_after_completability,
            fs.fraction_selected,
            fs.fraction_sampled,
            fs.fraction_true_excluded_by_train_fit,
            fs.fraction_true_absent_from_search,
        ] {
            if !(0.0..=1.0).contains(&name) {
                return Err(Error::config(format!(
                    "accounting: formula_search fraction {name} outside [0, 1]"
                )));
            }
        }
        if !(fs.fraction_joined + 1e-12 >= fs.fraction_after_domain
            && fs.fraction_after_domain + 1e-12 >= fs.fraction_after_substructures
            && fs.fraction_after_substructures + 1e-12 >= fs.fraction_after_completability
            && fs.fraction_after_completability + 1e-12 >= fs.fraction_selected
            && fs.fraction_selected + 1e-12 >= fs.fraction_sampled)
        {
            return Err(Error::config(format!(
                "accounting: formula_search fractions are not monotone: {} {} {} {} {} {}",
                fs.fraction_joined,
                fs.fraction_after_domain,
                fs.fraction_after_substructures,
                fs.fraction_after_completability,
                fs.fraction_selected,
                fs.fraction_sampled
            )));
        }
    }
    Ok(())
}

/// Run the experiment: train (unless the arm or flags skip it), evaluate,
/// generate, score, and write `report.json`, `predictions.jsonl` and the
/// checkpoint(s) into `args.out`.
///
/// The device sees only the usual trainer/generator traffic; the only
/// device reads of training are the `request_report` losses and the
/// batched eval NLLs. Returns the in-memory [`ExperimentReport`] (with
/// scored outcomes attached for test hooks); files are written as a side
/// effect.
pub fn run<R: Runtime>(args: &ExperimentArgs, device: &Device<R>) -> Result<ExperimentReport> {
    if args.batch == 0 {
        return Err(Error::config("run: --batch must be at least 1".to_string()));
    }
    if args.report_every == 0 {
        return Err(Error::config(
            "run: --report-every must be at least 1".to_string(),
        ));
    }
    if args.gen_batch == 0 {
        return Err(Error::config(
            "run: --gen-batch must be at least 1".to_string(),
        ));
    }
    if !(1..=32).contains(&args.formula_hypotheses) || args.formula_hypotheses == 0 {
        return Err(Error::config(format!(
            "run: --formula-hypotheses {} is not in 1..=32",
            args.formula_hypotheses
        )));
    }
    if args.mass_ppm_tenths > 1000 {
        return Err(Error::config(format!(
            "run: --mass-ppm-tenths {} exceeds the 1000 proof bound",
            args.mass_ppm_tenths
        )));
    }
    if !(args.lr.is_finite() && args.lr > 0.0) {
        return Err(Error::config(format!(
            "run: --lr {} is not finite and positive",
            args.lr
        )));
    }
    if !(args.weight_decay.is_finite() && args.weight_decay >= 0.0) {
        return Err(Error::config(format!(
            "run: --weight-decay {} is not finite and non-negative",
            args.weight_decay
        )));
    }
    if let Some(clip) = args.grad_clip
        && !(clip.is_finite() && clip > 0.0)
    {
        return Err(Error::config(format!(
            "run: --grad-clip {clip} is not finite and positive"
        )));
    }
    if !(args.temperature.is_finite() && args.temperature > 0.0) {
        return Err(Error::config(format!(
            "run: --temperature {} is not finite and positive",
            args.temperature
        )));
    }
    if args.eval_only && args.load.is_none() {
        return Err(Error::config(
            "run: --eval-only needs --load <ckpt>".to_string(),
        ));
    }
    check_out_dir(&args.out)?;
    std::fs::create_dir_all(&args.out)?;

    let started = Instant::now();
    let uses_fp = args.evidence.uses_fingerprint();
    if !(args.fp_threshold.is_finite() && args.fp_threshold > 0.0 && args.fp_threshold <= 1.0) {
        return Err(Error::config(format!(
            "run: --fp-threshold {} is not in (0, 1]",
            args.fp_threshold
        )));
    }
    if uses_fp && (args.fp_slots == 0 || args.fp_slots > 4096) {
        return Err(Error::config(format!(
            "run: --fp-slots {} is not in 1..=4096",
            args.fp_slots
        )));
    }
    if uses_fp && !args.eval_only && args.fp_train.is_none() {
        return Err(Error::config(
            "run: fingerprint evidence needs --fp-train <bits json>".to_string(),
        ));
    }
    if uses_fp
        && args.fp_eval_mode != super::completion_fingerprint::FingerprintEvalMode::Predicted
        && args.fp_validation.is_none()
    {
        return Err(Error::config(
            "run: fingerprint evidence needs --fp-validation <bits json> (or --fp-eval-mode predicted with a panel --validation)".to_string(),
        ));
    }
    if uses_fp
        && (args.fp_train_mode == super::completion_fingerprint::FingerprintMode::MistLike
            || args.fp_eval_mode == super::completion_fingerprint::FingerprintEvalMode::MistLike)
        && args.fp_noise.is_none()
    {
        return Err(Error::config(
            "run: mist_like fingerprint sampling needs --fp-noise <noise json>".to_string(),
        ));
    }
    // Predicted-panel evaluation must not select the checkpoint: the panel
    // scores would pick `.best`, which is then generated on the same panel
    // (evaluation leakage identity-group exclusion cannot prevent). Such a
    // run is `--eval-only` with a checkpoint chosen beforehand.
    if uses_fp
        && args.fp_eval_mode == super::completion_fingerprint::FingerprintEvalMode::Predicted
        && (!args.eval_only || args.load.is_none())
    {
        return Err(Error::config(
            "run: --fp-eval-mode predicted evaluates on the predicted panel, whose scores must not select the checkpoint: use --eval-only with a --load checkpoint chosen beforehand (e.g. on exact fingerprints or a separate development set)".to_string(),
        ));
    }
    if let Some(dump) = &args.dump_candidates {
        check_dump_path(dump)?;
    }
    let mut model_config = match args.model.as_str() {
        "small" => CompletionModelConfig::small(),
        "base" if uses_fp => CompletionModelConfig::base_fingerprint(),
        "base" => CompletionModelConfig::base(),
        _ => {
            return Err(Error::config(format!(
                "run: unknown --model {:?} (expected small|base)",
                args.model
            )));
        }
    };
    if uses_fp {
        model_config.fingerprint_slots = args.fp_slots;
    }
    let max_atoms = args.max_atoms.unwrap_or(model_config.max_atoms as usize);
    let max_closures = args
        .max_closures
        .unwrap_or(model_config.max_ring_closures as usize);
    model_config.max_atoms = u32::try_from(max_atoms)
        .map_err(|_| Error::config(format!("run: --max-atoms {max_atoms} does not fit u32")))?;
    model_config.max_ring_closures = u32::try_from(max_closures).map_err(|_| {
        Error::config(format!(
            "run: --max-closures {max_closures} does not fit u32"
        ))
    })?;
    model_config.validate()?;
    let limits = Limits::new(max_atoms, max_closures)?;
    let work_limit = CANONICAL_WORK_LIMIT;

    let (train_file, train_provenance) = read_export(&args.train)?;
    // Validation: either a regular export or, with `--fp-eval-mode predicted`,
    // a panel file in the molecule-export schema carrying `fp_pred_mean`.
    let predicted_eval = uses_fp
        && args.fp_eval_mode == super::completion_fingerprint::FingerprintEvalMode::Predicted;
    let (validation_file, validation_provenance, panel_pred, panel_true) = if predicted_eval {
        load_panel_validation(&args.validation)?
    } else {
        let (file, prov) = read_export(&args.validation)?;
        (file, prov, HashMap::new(), HashMap::new())
    };
    let mut train_set = CompletionSet::from_export(&train_file, limits, work_limit)?;
    let validation_set = CompletionSet::from_export(&validation_file, limits, work_limit)?;
    // Fingerprint sidecars.
    let fp_train_store = if uses_fp && !args.eval_only {
        match &args.fp_train {
            Some(path) => {
                let store = super::completion_fingerprint::FingerprintStore::load(path)?;
                store.assert_molecule_count(train_file.molecules.len())?;
                store.assert_keys_match(&export_entry_keys(&train_file))?;
                Some(store)
            }
            None => None,
        }
    } else if uses_fp {
        match &args.fp_train {
            Some(path) => {
                let store = super::completion_fingerprint::FingerprintStore::load(path)?;
                store.assert_molecule_count(train_file.molecules.len())?;
                store.assert_keys_match(&export_entry_keys(&train_file))?;
                Some(store)
            }
            None => None,
        }
    } else {
        None
    };
    let fp_validation_store = if uses_fp && !predicted_eval {
        match &args.fp_validation {
            Some(path) => {
                let store = super::completion_fingerprint::FingerprintStore::load(path)?;
                store.assert_molecule_count(validation_file.molecules.len())?;
                store.assert_keys_match(&export_entry_keys(&validation_file))?;
                Some(store)
            }
            None => None,
        }
    } else {
        None
    };
    let fp_noise = if uses_fp
        && (args.fp_train_mode == super::completion_fingerprint::FingerprintMode::MistLike
            || args.fp_eval_mode == super::completion_fingerprint::FingerprintEvalMode::MistLike)
    {
        match &args.fp_noise {
            Some(path) => Some(super::completion_fingerprint::FingerprintNoise::load(path)?),
            None => None,
        }
    } else if uses_fp && args.fp_noise.is_some() {
        Some(super::completion_fingerprint::FingerprintNoise::load(
            args.fp_noise.as_ref().expect("checked above"),
        )?)
    } else {
        None
    };
    // The molecule-averaged histogram set only exists in v2 noise files.
    if uses_fp
        && args.fp_noise_level == super::completion_fingerprint::FingerprintNoiseLevel::Molecule
    {
        match fp_noise.as_ref() {
            Some(noise) if noise.has_molecule_histograms => {}
            Some(_) => {
                return Err(Error::config(
                    "run: --fp-noise-level molecule needs a v2 noise file with hist_pred_given_true_on/off_molecule (regenerate with tools/ms2/export_fingerprints_mist.py)".to_string(),
                ));
            }
            None => {
                return Err(Error::config(
                    "run: --fp-noise-level molecule needs --fp-noise <noise json>".to_string(),
                ));
            }
        }
    }
    // Identity-group exclusion (`--exclude-identity-groups <panel json>`).
    let mut train_excluded_molecules = 0usize;
    if let Some(exclude_path) = &args.exclude_identity_groups {
        let bytes = std::fs::read(exclude_path)?;
        let text = std::str::from_utf8(&bytes).map_err(|e| {
            Error::config(format!(
                "run: {} is not valid UTF-8: {e}",
                exclude_path.display()
            ))
        })?;
        let raw: serde_json::Value = serde_json::from_str(text)?;
        let groups: Vec<u64> = match raw.get("panel_identity_groups") {
            Some(list) => list
                .as_array()
                .ok_or_else(|| {
                    Error::config(
                        "run: --exclude-identity-groups panel has no panel_identity_groups list"
                            .to_string(),
                    )
                })?
                .iter()
                .map(|v| {
                    v.as_u64().ok_or_else(|| {
                        Error::config(
                            "run: --exclude-identity-groups panel_identity_groups holds a non-integer"
                                .to_string(),
                        )
                    })
                })
                .collect::<Result<Vec<u64>>>()?,
            None => {
                return Err(Error::config(
                    "run: --exclude-identity-groups file has no panel_identity_groups".to_string(),
                ));
            }
        };
        let excluded: HashSet<u64> = groups.into_iter().collect();
        let before = train_set.examples.len();
        train_set.examples.retain(|e| !excluded.contains(&e.identity_group));
        train_excluded_molecules = before - train_set.examples.len();
    }
    let train_full_kept = train_set.examples.len();
    let train_keep = args
        .limit_train
        .map_or(train_full_kept, |n| n.min(train_full_kept));
    let (rows, limit_excluded) = classify_validation(
        &validation_file,
        &validation_set,
        limits,
        work_limit,
        args.limit_validation,
    )?;
    let train_skipped = train_set.skipped.clone();
    {
        let mut rebuilt: BTreeMap<String, u64> = BTreeMap::new();
        for row in &rows {
            if let Some(reason) = &row.skip_reason {
                *rebuilt.entry(reason.clone()).or_default() += 1;
            }
        }
        if rebuilt != validation_set.skipped {
            return Err(Error::config(format!(
                "run: per-molecule skip counts {rebuilt:?} disagree with the set's {:?}",
                validation_set.skipped
            )));
        }
    }

    // Evaluated validation examples in file order, with fixed evaluation
    // patterns from a stream training never uses.
    let eval_seed = eval_extraction_seed(args.extraction_seed);
    let eval_extraction = ExtractionConfig::default();
    eval_extraction.validate()?;
    let use_fg = args.patterns == PatternArg::FunctionalGroups;
    let eval_fg_config = FunctionalGroupConfig {
        max_groups: 8,
        max_total_atoms: 24,
        max_group_atoms: 24,
        keep_probability_percent: 100,
        aromatic_rings_as_groups: args.fg_aromatic_rings,
    };
    if use_fg {
        eval_fg_config.validate()?;
        if args.fg_keep_percent > 100 {
            return Err(Error::config(format!(
                "run: --fg-keep-percent {} exceeds 100",
                args.fg_keep_percent
            )));
        }
    }
    let mut eval_indices: Vec<usize> = Vec::new();
    for row in &rows {
        if let Some(i) = row.example
            && !row.limit_excluded
        {
            eval_indices.push(i);
        }
    }
    // Exclusion check: no evaluation identity group may remain in training.
    // Non-zero exit (an error) when one does.
    if args.exclude_identity_groups.is_some() {
        let train_groups: HashSet<u64> =
            train_set.examples.iter().map(|e| e.identity_group).collect();
        let mut remaining: Vec<u64> = Vec::new();
        for &i in &eval_indices {
            let group = validation_set.examples[i].identity_group;
            if train_groups.contains(&group) && !remaining.contains(&group) {
                remaining.push(group);
            }
        }
        if !remaining.is_empty() {
            return Err(Error::config(format!(
                "run: {} evaluation identity groups remain in training after --exclude-identity-groups (e.g. {})",
                remaining.len(),
                remaining[0]
            )));
        }
    }
    // With `--patterns functional_groups --substructure-semantics complete`,
    // evaluation queries carry the untruncated full group list for
    // acceptance, while the model sees the seeded fitting subset: the
    // encoder limits (8 patterns / 24 atoms) still bind the device input.
    let use_full_acceptance = use_fg
        && args.substructure_semantics == SubstructureSemantics::CompleteFunctionalGroups;
    let mut eval_patterns = Vec::with_capacity(eval_indices.len());
    let mut eval_acceptance: Vec<Option<Vec<MolGraph>>> = Vec::with_capacity(eval_indices.len());
    let mut full_list_acceptance_queries = 0usize;
    // Per-eval-query functional-group audit: (groups_found, truncated, dropped).
    let mut eval_fg_meta: Vec<(usize, bool, usize)> = Vec::with_capacity(eval_indices.len());
    for &i in &eval_indices {
        let example = &validation_set.examples[i];
        // With `--evidence fingerprint` the model sees no patterns: every
        // query conditions on the fingerprint alone (acceptance is exact
        // composition and a valid complete molecule only, which empty
        // patterns give). With `both` the patterns below apply as usual.
        if args.evidence == super::completion_fingerprint::Evidence::Fingerprint {
            eval_fg_meta.push((0, false, 0));
            eval_patterns.push(Vec::new());
            eval_acceptance.push(None);
            continue;
        }
        if use_fg {
            let draw = functional_group_patterns(
                &example.target,
                &eval_fg_config,
                eval_seed,
                &example.key,
                EVAL_DRAW,
            )?;
            eval_fg_meta.push((draw.groups_found, draw.truncated, draw.dropped_oversized));
            eval_patterns.push(draw.patterns);
            if use_full_acceptance {
                // Complete acceptance uses the full Ertl list only: optional
                // unmarked aromatic rings stay in the encoder conditioning
                // (the fitting subset above) but never enter complete
                // acceptance, where they would overlap the Ertl groups and
                // reject their own target.
                let full = full_functional_group_list(
                    &example.target,
                    false,
                )?;
                let full_atoms: usize = full.iter().map(|g| g.atoms().len()).sum();
                if full.len() > MAX_PATTERNS || full_atoms > PATTERN_SLOTS {
                    full_list_acceptance_queries += 1;
                }
                eval_acceptance.push(Some(full));
            } else {
                eval_acceptance.push(None);
            }
        } else {
            let patterns = example.patterns(&eval_extraction, eval_seed, EVAL_DRAW)?;
            eval_fg_meta.push((patterns.len(), false, 0));
            eval_patterns.push(patterns);
            eval_acceptance.push(None);
        }
    }
    // Overlap against the train examples actually trained on, for every
    // kept validation example (evaluated or limit-excluded).
    let mut train_traces = HashSet::new();
    let mut train_skeletons = HashSet::new();
    for example in train_set.examples.iter().take(train_keep) {
        train_traces.insert(example.trace.clone());
        if !example.skeleton_trace.is_empty() {
            train_skeletons.insert(example.skeleton_trace.clone());
        }
    }
    let mut strict_by_example: HashMap<usize, bool> = HashMap::new();
    for (i, example) in validation_set.examples.iter().enumerate() {
        strict_by_example.insert(i, train_traces.contains(&example.trace));
    }
    let mut identity_in_train = Vec::with_capacity(eval_indices.len());
    let mut overlap_strict = 0usize;
    let mut overlap_skeleton = 0usize;
    for &i in &eval_indices {
        let example = &validation_set.examples[i];
        let strict = strict_by_example.get(&i).copied().unwrap_or(false);
        identity_in_train.push(strict);
        if strict {
            overlap_strict += 1;
        }
        if !example.skeleton_trace.is_empty() && train_skeletons.contains(&example.skeleton_trace) {
            overlap_skeleton += 1;
        }
    }

    // Pattern statistics of the evaluation patterns.
    let mut pattern_total = 0usize;
    let mut pattern_atoms = 0usize;
    let mut coverage_sum = 0.0f64;
    for (pos, &i) in eval_indices.iter().enumerate() {
        let example = &validation_set.examples[i];
        let patterns = &eval_patterns[pos];
        pattern_total += patterns.len();
        let mut covered = vec![false; example.target.atoms().len()];
        for pattern in patterns {
            pattern_atoms += pattern.graph.atoms().len();
            for atom in &pattern.parent_atoms {
                covered[*atom] = true;
            }
        }
        let hit = covered.iter().filter(|&&c| c).count();
        coverage_sum += hit as f64 / example.target.atoms().len().max(1) as f64;
    }
    let n_eval = eval_indices.len().max(1);
    // Functional-group fractions and top signatures (eligible queries only).
    let (fraction_no_group, fraction_truncated, fraction_dropped_oversized, top_groups) =
        if use_fg && !eval_indices.is_empty() {
            let no_group = eval_fg_meta
                .iter()
                .filter(|(found, _, _)| *found == 0)
                .count();
            let truncated = eval_fg_meta.iter().filter(|(_, t, _)| *t).count();
            let dropped = eval_fg_meta.iter().filter(|(_, _, d)| *d > 0).count();
            let n = eval_indices.len() as f64;
            // Query counts per signature over the found groups (generic
            // chemistry, not data rows): one count per query per signature.
            let mut per_signature: BTreeMap<String, usize> = BTreeMap::new();
            for &i in &eval_indices {
                let example = &validation_set.examples[i];
                let found = functional_groups(&example.target).map_err(|e| {
                    Error::config(format!("run: functional groups failed: {e}"))
                })?;
                let mut seen = std::collections::HashSet::new();
                for group in &found {
                    if seen.insert(group.signature.clone()) {
                        *per_signature.entry(group.signature.clone()).or_default() += 1;
                    }
                }
            }
            let mut ranked: Vec<(String, usize)> = per_signature.into_iter().collect();
            ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            ranked.truncate(20);
            let top = ranked
                .into_iter()
                .map(|(signature, queries)| TopGroup {
                    signature,
                    queries,
                })
                .collect();
            (
                Some(no_group as f64 / n),
                Some(truncated as f64 / n),
                Some(dropped as f64 / n),
                Some(top),
            )
        } else if use_fg {
            (Some(0.0), Some(0.0), Some(0.0), Some(Vec::new()))
        } else {
            (None, None, None, None)
        };
    let eval_pattern_stats = EvalPatternStats {
        seed: eval_seed,
        draw: EVAL_DRAW,
        pattern_source: args.patterns.as_str().to_string(),
        mean_patterns: pattern_total as f64 / n_eval as f64,
        mean_pattern_atoms: if pattern_total == 0 {
            0.0
        } else {
            pattern_atoms as f64 / pattern_total as f64
        },
        mean_coverage: coverage_sum / n_eval as f64,
        fraction_no_group,
        fraction_truncated,
        fraction_dropped_oversized,
        top_groups,
        full_list_acceptance_queries,
    };

    // Trainer: fresh or loaded; the formula-only arm trains on empty
    // patterns, as does `--evidence fingerprint`.
    let no_patterns = args.arm == ExperimentArm::FormulaOnly
        || args.evidence == super::completion_fingerprint::Evidence::Fingerprint;
    let train_extraction = if no_patterns {
        ExtractionConfig {
            min_patterns: 0,
            max_patterns: 0,
            ..ExtractionConfig::default()
        }
    } else {
        ExtractionConfig::default()
    };
    train_extraction.validate()?;
    let train_pattern_source = if no_patterns {
        PatternSource::RandomPatches(ExtractionConfig {
            min_patterns: train_extraction.min_patterns,
            max_patterns: train_extraction.max_patterns,
            min_pattern_atoms: train_extraction.min_pattern_atoms,
            max_pattern_atoms: train_extraction.max_pattern_atoms,
            max_total_atoms: train_extraction.max_total_atoms,
        })
    } else if use_fg {
        let train_fg = FunctionalGroupConfig {
            max_groups: 8,
            max_total_atoms: 24,
            max_group_atoms: 24,
            keep_probability_percent: args.fg_keep_percent,
            aromatic_rings_as_groups: args.fg_aromatic_rings,
        };
        train_fg.validate()?;
        PatternSource::FunctionalGroups(train_fg)
    } else {
        PatternSource::RandomPatches(ExtractionConfig {
            min_patterns: train_extraction.min_patterns,
            max_patterns: train_extraction.max_patterns,
            min_pattern_atoms: train_extraction.min_pattern_atoms,
            max_pattern_atoms: train_extraction.max_pattern_atoms,
            max_total_atoms: train_extraction.max_total_atoms,
        })
    };
    let train_config = CompletionTrainConfig {
        lr: args.lr,
        weight_decay: args.weight_decay,
        grad_clip: args.grad_clip,
        seed: args.seed,
        extraction: ExtractionConfig {
            min_patterns: train_extraction.min_patterns,
            max_patterns: train_extraction.max_patterns,
            min_pattern_atoms: train_extraction.min_pattern_atoms,
            max_pattern_atoms: train_extraction.max_pattern_atoms,
            max_total_atoms: train_extraction.max_total_atoms,
        },
        extraction_seed: args.extraction_seed,
        pattern_source: train_pattern_source,
        fingerprint_mode: if uses_fp {
            Some(args.fp_train_mode)
        } else {
            None
        },
        fingerprint_threshold: args.fp_threshold,
    };
    let mut trainer = match &args.load {
        Some(path) => {
            let loaded = CompletionTrainer::<R, f32>::load(path, device)?;
            if loaded.model().config != model_config {
                return Err(Error::config(format!(
                    "run: checkpoint model config does not match --model {} with domain ({max_atoms}, {max_closures})",
                    args.model
                )));
            }
            loaded
        }
        None => CompletionTrainer::new(&model_config, &train_config, device)?,
    };
    // Resume conflicts: continued training follows the checkpoint's train
    // config (pattern source, extraction, fingerprint mode/threshold,
    // optimizer settings), never the new CLI values. Reject a resume
    // whose CLI training options disagree instead of silently training under
    // the old ones while the report echoes the new ones. Only the weight
    // initialisation seed is exempt (it is inert after loading). The noise
    // level is not in the checkpoint header: the driver sets it from the CLI
    // on every run, so it cannot conflict.
    let trains_on = args.load.is_some() && !args.eval_only && args.arm != ExperimentArm::Untrained;
    if trains_on {
        let kept = trainer.train_config();
        let mut conflicts: Vec<&str> = Vec::new();
        if kept.lr != train_config.lr {
            conflicts.push("--lr");
        }
        if kept.weight_decay != train_config.weight_decay {
            conflicts.push("--weight-decay");
        }
        if kept.grad_clip != train_config.grad_clip {
            conflicts.push("--grad-clip");
        }
        if kept.extraction != train_config.extraction {
            conflicts.push("extraction (arm/evidence)");
        }
        if kept.extraction_seed != train_config.extraction_seed {
            conflicts.push("--extraction-seed");
        }
        if kept.pattern_source != train_config.pattern_source {
            conflicts.push("--patterns/--fg-*");
        }
        if kept.fingerprint_mode != train_config.fingerprint_mode {
            conflicts.push("--fp-train-mode/--evidence");
        }
        if kept.fingerprint_threshold != train_config.fingerprint_threshold {
            conflicts.push("--fp-threshold");
        }
        if !conflicts.is_empty() {
            return Err(Error::config(format!(
                "run: --load resume conflicts with the checkpoint's training configuration (continued training would use the checkpoint's values while the report echoes the CLI ones): {} (resume with matching options, or start a fresh run)",
                conflicts.join(", ")
            )));
        }
    }
    // The effective trainer configuration (the checkpoint's own after
    // --load): the report records this, never the resume CLI values.
    let effective_train_config = trainer.train_config().clone();
    // Fingerprint sidecars onto the trainer (training and evaluation with
    // the encoder need them; harmless otherwise).
    if uses_fp {
        if let Some(store) = fp_train_store {
            trainer.set_fingerprint_store(store);
        }
        if let Some(noise) = fp_noise.clone() {
            trainer.set_fingerprint_noise(noise);
        }
        // The run's noise level (not persisted in the checkpoint, so set on
        // every run, fresh or resumed).
        trainer.set_fingerprint_noise_level(args.fp_noise_level);
        // Validation fingerprints for eval diagnostics come from the
        // validation store below (or the panel maps); the trainer's
        // evaluation path (`teacher_eval`) uses the train store, which
        // covers only train indices. The eval-NLL path below therefore
        // goes through `teacher_eval_with_fingerprints` with explicitly
        // built validation fingerprints (see the eval closure).
    }
    // Formula artifacts bound to the checkpoint, fitted on the kept
    // training compositions (margin 0, quantile margin 0, heavy_max clamped
    // to the model domain). Stored before every save below.
    {
        let compositions: Vec<Composition> = train_set.examples[..train_keep]
            .iter()
            .map(|e| e.composition)
            .collect();
        let source = format!("completion_experiment:train:{}", train_provenance.file);
        fit_and_attach(&mut trainer, &compositions, max_atoms as u32, source)?;
    }
    let start_step = trainer.step_count();
    let target_steps = start_step + args.steps as u64;

    // Training loop: seeded shuffle per epoch, `draw = epoch`.
    let mut curve: Vec<CurvePoint> = Vec::new();
    let mut best_nll: Option<f64> = None;
    let mut best_step: Option<u64> = None;
    let train_seconds;
    let best_path: Option<PathBuf> = args.save.as_ref().map(|p| {
        let mut s = p.as_os_str().to_owned();
        s.push(".best");
        PathBuf::from(s)
    });
    let mut wrote_best = false;
    // Live-progress state: the previous line's time and step, for the
    // steps-per-second rate. No extra device reads: every printed value is
    // one the driver already read.
    let mut progress_at = Instant::now();
    let mut progress_step = start_step;
    let progress_line = |step: u64,
                         loss: Option<f32>,
                         per_token: Option<f64>,
                         progress_at: &mut Instant,
                         progress_step: &mut u64| {
        let now = Instant::now();
        let elapsed = started.elapsed().as_secs_f64();
        let dt = now.duration_since(*progress_at).as_secs_f64().max(1e-9);
        let rate = step.saturating_sub(*progress_step) as f64 / dt;
        let loss_text = loss.map_or("na".to_string(), |v| format!("{v:.4}"));
        let eval_text = per_token.map_or("na".to_string(), |v| format!("{v:.4}"));
        eprintln!(
            "[progress] step={step} loss={loss_text} eval_nll_per_token={eval_text} elapsed_s={elapsed:.1} steps_per_s={rate:.2}"
        );
        *progress_at = now;
        *progress_step = step;
    };
    let eval_subset: Vec<usize> = (0..eval_indices.len().min(args.eval_subset)).collect();
    // Fixed eval-pattern batches for the subset (same stream as generation).
    let mut eval_subset_patterns: Vec<Vec<MolGraph>> = Vec::new();
    for &pos in &eval_subset {
        eval_subset_patterns.push(
            eval_patterns[pos]
                .iter()
                .map(|p| {
                    MolGraph::new(p.graph.atoms().to_vec(), p.graph.bonds().to_vec())
                        .expect("evaluation patterns rebuild")
                })
                .collect(),
        );
    }
    let eval_nlls = |trainer: &mut CompletionTrainer<R, f32>| -> Result<(f64, f64)> {
        if eval_subset.is_empty() {
            return Ok((f64::NAN, f64::NAN));
        }
        let mut patterns: Vec<&[MolGraph]> = Vec::with_capacity(eval_subset.len());
        let mut traces = Vec::with_capacity(eval_subset.len());
        let mut compositions = Vec::with_capacity(eval_subset.len());
        for (k, &pos) in eval_subset.iter().enumerate() {
            let example = &validation_set.examples[eval_indices[pos]];
            // The formula-only arm never shows patterns to the model, so its
            // validation loss (which selects the checkpoint) must not either:
            // scoring it with patterns measured an input it was never trained
            // on and picked an arbitrary checkpoint. `--evidence fingerprint`
            // likewise sees no patterns.
            patterns.push(
                if args.arm == ExperimentArm::FormulaOnly
                    || args.evidence == super::completion_fingerprint::Evidence::Fingerprint
                {
                    &[]
                } else {
                    eval_subset_patterns[k].as_slice()
                },
            );
            traces.push(example.trace.as_slice());
            compositions.push(example.composition);
        }
        let nlls = if uses_fp {
            let fps = build_eval_fingerprints(
                &validation_set,
                &eval_indices,
                &eval_subset,
                fp_validation_store.as_ref(),
                fp_noise.as_ref(),
                args.fp_noise_level,
                &panel_pred,
                &panel_true,
                args.fp_eval_mode,
                args.fp_threshold,
                eval_seed,
            )?;
            trainer.teacher_eval_with_fingerprints(&patterns, &fps, &traces, &compositions)?
        } else {
            trainer.teacher_eval_with(&patterns, &traces, &compositions)?
        };
        let tokens: usize = eval_subset
            .iter()
            .map(|&pos| validation_set.examples[eval_indices[pos]].trace.len())
            .sum();
        let sum: f64 = nlls.iter().map(|&v| f64::from(v)).sum();
        Ok((sum / nlls.len() as f64, sum / tokens.max(1) as f64))
    };
    if args.eval_only || args.arm == ExperimentArm::Untrained {
        train_seconds = 0.0;
        if args.eval_every > 0 && !eval_subset.is_empty() {
            let (per_example, per_token) = eval_nlls(&mut trainer)?;
            best_nll = Some(per_example);
            best_step = Some(trainer.step_count());
            curve.push(CurvePoint {
                step: trainer.step_count(),
                loss: None,
                eval_nll_per_example: Some(per_example),
                eval_nll_per_token: Some(per_token),
                elapsed_seconds: started.elapsed().as_secs_f64(),
            });
            if args.progress {
                progress_line(
                    trainer.step_count(),
                    None,
                    Some(per_token),
                    &mut progress_at,
                    &mut progress_step,
                );
            }
        }
    } else {
        if train_keep == 0 {
            return Err(Error::config(
                "run: the train set keeps no examples under this domain".to_string(),
            ));
        }
        let train_started = Instant::now();
        let mut order: Vec<usize> = (0..train_keep).collect();
        let mut epoch: u64 = 0;
        if args.eval_every > 0 && !eval_subset.is_empty() {
            let (per_example, per_token) = eval_nlls(&mut trainer)?;
            best_nll = Some(per_example);
            best_step = Some(start_step);
            curve.push(CurvePoint {
                step: start_step,
                loss: None,
                eval_nll_per_example: Some(per_example),
                eval_nll_per_token: Some(per_token),
                elapsed_seconds: started.elapsed().as_secs_f64(),
            });
            // The initial best names a checkpoint too: without this save,
            // generation would use the final weights despite reporting the
            // earlier best step when no later score improves.
            if let Some(path) = &best_path {
                trainer.save(path)?;
                wrote_best = true;
            }
            if args.progress {
                progress_line(
                    start_step,
                    None,
                    Some(per_token),
                    &mut progress_at,
                    &mut progress_step,
                );
            }
        }
        while trainer.step_count() < target_steps {
            let mut rng = SplitMix64::new(args.seed.wrapping_add(epoch));
            for i in (1..order.len()).rev() {
                let j = rng.below(i as u64 + 1) as usize;
                order.swap(i, j);
            }
            for chunk in order.chunks(args.batch.max(1)) {
                if trainer.step_count() >= target_steps {
                    break;
                }
                if trainer.step_count() % args.report_every as u64 == 0 {
                    trainer.request_report();
                }
                let loss = trainer.step(&train_set, chunk, epoch)?;
                let step = trainer.step_count();
                if let Some(loss) = loss {
                    curve.push(CurvePoint {
                        step,
                        loss: Some(loss),
                        eval_nll_per_example: None,
                        eval_nll_per_token: None,
                        elapsed_seconds: started.elapsed().as_secs_f64(),
                    });
                    if args.progress {
                        progress_line(step, Some(loss), None, &mut progress_at, &mut progress_step);
                    }
                }
                if args.eval_every > 0
                    && step % args.eval_every as u64 == 0
                    && !eval_subset.is_empty()
                {
                    let (per_example, per_token) = eval_nlls(&mut trainer)?;
                    curve.push(CurvePoint {
                        step,
                        loss: None,
                        eval_nll_per_example: Some(per_example),
                        eval_nll_per_token: Some(per_token),
                        elapsed_seconds: started.elapsed().as_secs_f64(),
                    });
                    if args.progress {
                        progress_line(
                            step,
                            None,
                            Some(per_token),
                            &mut progress_at,
                            &mut progress_step,
                        );
                    }
                    if best_nll.is_none_or(|best| per_example < best) {
                        best_nll = Some(per_example);
                        best_step = Some(step);
                        if let Some(path) = &best_path {
                            trainer.save(path)?;
                            wrote_best = true;
                        }
                    }
                }
            }
            epoch += 1;
        }
        train_seconds = train_started.elapsed().as_secs_f64();
    }
    let steps_done = trainer.step_count() - start_step;
    if let Some(path) = &args.save {
        trainer.save(path)?;
    }

    // Which checkpoint the final generation uses: the best one when this
    // run wrote it, else the final (or loaded) trainer.
    let mut eval_trainer: Option<CompletionTrainer<R, f32>> = None;
    let checkpoint_evaluated = if wrote_best {
        let path = best_path.as_ref().expect("best path when wrote_best");
        eval_trainer = Some(CompletionTrainer::load(path, device)?);
        format!("{} (best validation NLL)", path.display())
    } else if let Some(path) = &args.load
        && (args.eval_only || args.arm == ExperimentArm::Untrained)
    {
        format!("{} (loaded, no training)", path.display())
    } else if let Some(path) = &args.save {
        format!("{} (final)", path.display())
    } else if args.arm == ExperimentArm::Untrained || args.eval_only {
        "in-memory weights (no training, no --save)".to_string()
    } else {
        "in-memory final weights (no --save)".to_string()
    };
    let gen_model_trainer = eval_trainer.as_ref().unwrap_or(&trainer);

    // Final generation over the eligible queries in file order.
    let mut gen_config = eval_generation_config(
        args.trajectories,
        args.temperature,
        args.gen_seed,
        args.returned,
        args.arm != ExperimentArm::FormulaOnly
            && args.evidence.uses_patterns(),
    );
    gen_config.substructure_semantics = args.substructure_semantics;
    gen_config.validate()?;
    let constants = Ms2Constants::new(device);
    let gen_started = Instant::now();
    let mut outcomes: Vec<QueryOutcome> = Vec::with_capacity(eval_indices.len());
    // Fingerprints for every eligible query (oracle path).
    let all_positions: Vec<usize> = (0..eval_indices.len()).collect();
    let eval_fingerprints: Vec<super::completion_fingerprint::SparseFingerprint> = if uses_fp {
        build_eval_fingerprints(
            &validation_set,
            &eval_indices,
            &all_positions,
            fp_validation_store.as_ref(),
            fp_noise.as_ref(),
            args.fp_noise_level,
            &panel_pred,
            &panel_true,
            args.fp_eval_mode,
            args.fp_threshold,
            eval_seed,
        )?
    } else {
        Vec::new()
    };
    // Per-query fingerprint diagnostics.
    let mut fp_tokens: Vec<usize> = Vec::new();
    let mut fp_dropped: Vec<usize> = Vec::new();
    let mut fp_true_missing: Vec<usize> = Vec::new();
    let mut fp_false: Vec<usize> = Vec::new();
    if uses_fp {
        let slots = args.fp_slots as usize;
        for (pos, fp) in eval_fingerprints.iter().enumerate() {
            let example = &validation_set.examples[eval_indices[pos]];
            let true_bits: Vec<u16> = if let Some(bits) = panel_true.get(&example.source_index) {
                bits.clone()
            } else if let Some(store) = fp_validation_store.as_ref() {
                store.get_by_index(example.source_index)?.to_vec()
            } else {
                Vec::new()
            };
            let stats = super::completion_fingerprint::FingerprintQueryStats::score(
                fp, &true_bits, slots,
            );
            fp_tokens.push(stats.tokens_used);
            fp_dropped.push(stats.entries_dropped);
            let noisy = args.fp_eval_mode
                != super::completion_fingerprint::FingerprintEvalMode::Exact;
            if noisy {
                fp_true_missing.push(stats.true_missing);
                fp_false.push(stats.false_tokens);
            }
        }
    }
    // Mass-arm per-query details, in eval order (only when mass).
    let mut mass_details: Vec<Option<MassEvalDetail>> = vec![None; eval_indices.len()];
    if args.formula_source == FormulaSource::Oracle {
        let mut gen_calls = 0usize;
        for chunk in (0..eval_indices.len()).step_by(args.gen_batch.max(1)) {
            let end = (chunk + args.gen_batch.max(1)).min(eval_indices.len());
            let mut owned: Vec<Vec<MolGraph>> = Vec::with_capacity(end - chunk);
            let mut owned_acceptance: Vec<Option<Vec<MolGraph>>> =
                Vec::with_capacity(end - chunk);
            for pos in chunk..end {
                let patterns = &eval_patterns[pos];
                owned.push(
                    patterns
                        .iter()
                        .map(|p| {
                            MolGraph::new(p.graph.atoms().to_vec(), p.graph.bonds().to_vec())
                                .expect("evaluation patterns rebuild")
                        })
                        .collect(),
                );
                owned_acceptance.push(eval_acceptance[pos].as_ref().map(|full| {
                    full.iter()
                        .map(|g| {
                            MolGraph::new(g.atoms().to_vec(), g.bonds().to_vec())
                                .expect("evaluation acceptance rebuild")
                        })
                        .collect()
                }));
            }
            let mut requests = Vec::with_capacity(end - chunk);
            for (k, pos) in (chunk..end).enumerate() {
                let example = &validation_set.examples[eval_indices[pos]];
                // Request ids mix the source position in: export keys may
                // repeat across rows, so the key alone does not identify a
                // query.
                let id_seed = example.source_index.to_string();
                requests.push(CompletionRequest {
                    id: stable_hash(&[example.key.as_str(), id_seed.as_str()]),
                    composition: example.composition,
                    patterns: owned[k].as_slice(),
                    acceptance_patterns: owned_acceptance[k].as_deref(),
                    fingerprint: if uses_fp {
                        Some(&eval_fingerprints[pos])
                    } else {
                        None
                    },
                });
            }
            let mut chunk_outcomes =
                gen_model_trainer
                    .model()
                    .generate(&requests, &gen_config, &constants, device)?;
            outcomes.append(&mut chunk_outcomes);
            gen_calls += 1;
            if args.progress && (gen_calls.is_multiple_of(10) || outcomes.len() == eval_indices.len()) {
                let elapsed = gen_started.elapsed().as_secs_f64();
                let done = outcomes.len() as f64 * f64::from(args.trajectories);
                let rate = if elapsed > 0.0 { done / elapsed } else { 0.0 };
                eprintln!(
                    "[progress] generate queries={}/{} trajectories_per_s={rate:.1}",
                    outcomes.len(),
                    eval_indices.len(),
                );
            }
        }
    } else {
        // Mass arm: synthetic exact neutral masses, same patterns and
        // acceptance as the oracle arm, through the shared library path.
        let artifacts = gen_model_trainer.formula_artifacts().ok_or_else(|| {
            Error::config("run: mass evaluation needs formula artifacts (fit from train)".to_string())
        })?;
        let nodes_visited_max: u64 = 2_000_000;
        for (pos, &ei) in eval_indices.iter().enumerate() {
            let example = &validation_set.examples[ei];
            let neutral = composition_mass(&example.composition)?;
            let uncertainty = if args.mass_uncertainty_uda == u32::MAX {
                None
            } else {
                Some(args.mass_uncertainty_uda)
            };
            let mass_query = MassQuery::Neutral {
                value: neutral,
                ppm_tenths: args.mass_ppm_tenths,
                uncertainty,
            };
            let rebuilt: Vec<MolGraph> = eval_patterns[pos]
                .iter()
                .map(|p| {
                    MolGraph::new(p.graph.atoms().to_vec(), p.graph.bonds().to_vec())
                        .expect("evaluation patterns rebuild")
                })
                .collect();
            // The mass request id mixes the source position in, like the
            // oracle request ids (export keys may repeat across rows).
            let mass_request_id =
                format!("{}|{}", example.key, example.source_index);
            let result = run_mass_completion(
                gen_model_trainer.model(),
                &constants,
                device,
                artifacts,
                max_atoms as u32,
                max_closures as u32,
                &rebuilt,
                eval_acceptance[pos].as_deref(),
                &mass_query,
                args.formula_hypotheses,
                nodes_visited_max,
                args.trajectories,
                args.temperature,
                args.gen_seed,
                args.returned,
                &mass_request_id,
                args.arm != ExperimentArm::FormulaOnly
                    && args.evidence.uses_patterns(),
                args.formula_pruning,
                args.formula_allocation,
                args.substructure_semantics,
                if uses_fp {
                    Some(&eval_fingerprints[pos])
                } else {
                    None
                },
            )?;
            // True-formula stage and rank.
            let true_comp = example.composition;
            let fs = &result.formula_search;
            let mut true_rank: Option<usize> = None;
            // Rank among selected ordered by residual (formulas already in
            // rank order).
            for (r, f) in fs.formulas.iter().enumerate() {
                if f.composition == true_comp {
                    true_rank = Some(r + 1);
                    break;
                }
            }
            let in_selected = fs.formulas.iter().any(|f| f.composition == true_comp);
            let sampled_true = fs
                .formulas
                .iter()
                .any(|f| f.composition == true_comp && f.trajectories > 0);
            let stage = if sampled_true {
                "sampled".to_string()
            } else if in_selected {
                "selected".to_string()
            } else {
                // Fall back to filter checks for finer stages.
                mass_true_stage(
                    &true_comp,
                    &rebuilt,
                    limits,
                    max_atoms as u32,
                    &result,
                )
            };
            mass_details[pos] = Some(MassEvalDetail {
                joined: fs.joined,
                after_domain: fs.after_domain,
                after_substructures: fs.after_substructures,
                after_completability: fs.after_completability,
                selected: fs.selected,
                sampled: fs.sampled,
                stage: stage.clone(),
                true_rank,
                truncated: fs.truncated,
                exhausted: fs.enumerator.exhausted,
                sampled_true,
                excluded_by_train_fit: excluded_by_train_fit_direct(artifacts, &true_comp),
                absent_from_search: !result.joined_compositions.contains(&true_comp),
            });
            // Synthetic outcome for scoring: pooled candidates only.
            let mut cands: Vec<super::completion_model::CompletionCandidate> = Vec::new();
            for pc in &result.candidates {
                let graph =
                    MolGraph::new(pc.atoms.clone(), pc.bonds.clone()).expect("pooled rebuild");
                cands.push(super::completion_model::CompletionCandidate {
                    graph,
                    trace: pc.trace.clone(),
                    samples: pc.samples,
                    best_log_prob: pc.best_log_prob,
                });
            }
            outcomes.push(QueryOutcome {
                candidates: cands,
                unresolved: Vec::new(),
                sampled: Vec::new(),
                distinct: result.distinct_before_cut,
                trajectories: result.accounting.trajectories,
                finished: result.accounting.finished,
                dead_end: result.accounting.dead_end,
                truncated: result.accounting.truncated,
                rejected_replay: result.accounting.rejected_replay,
                rejected_containment: result.accounting.rejected_containment,
                containment_unresolved: result.accounting.containment_unresolved,
                rejected_extra_groups: result.accounting.rejected_extra_groups,
                rejected_missing_groups: result.accounting.rejected_missing_groups,
                pass_contained: result.accounting.pass_contained,
                pass_disjoint: result.accounting.pass_disjoint,
                pass_complete: result.accounting.pass_complete,
                identity_unresolved: result.accounting.identity_unresolved,
                other_status: result.accounting.other_status,
                infeasible: None,
            });
            if args.progress && (pos + 1) % 10 == 0 || pos + 1 == eval_indices.len() {
                if args.progress {
                    let elapsed = gen_started.elapsed().as_secs_f64();
                    let done = (pos + 1) as f64 * f64::from(args.trajectories);
                    let rate = if elapsed > 0.0 { done / elapsed } else { 0.0 };
                    eprintln!(
                        "[progress] generate queries={}/{} trajectories_per_s={rate:.1}",
                        pos + 1,
                        eval_indices.len(),
                    );
                }
            }
        }
    }
    let gen_seconds = gen_started.elapsed().as_secs_f64();
    let trajectories_total = outcomes.len() as f64 * f64::from(args.trajectories);

    // Scores in file order of the evaluated queries.
    let identity_work = gen_config.identity_work_limit as usize;
    let mut scores: Vec<QueryScore> = Vec::with_capacity(outcomes.len());
    let mut groups: Vec<u64> = Vec::with_capacity(outcomes.len());
    for (pos, outcome) in outcomes.iter().enumerate() {
        let example = &validation_set.examples[eval_indices[pos]];
        scores.push(score_query(outcome, &example.target, identity_work));
        groups.push(example.identity_group);
    }

    // All-read scores: every skipped molecule is a miss.
    let mut example_pos: HashMap<usize, usize> = HashMap::new();
    for (pos, &i) in eval_indices.iter().enumerate() {
        example_pos.insert(i, pos);
    }
    let mut all_scores = Vec::with_capacity(rows.len());
    let mut all_groups = Vec::with_capacity(rows.len());
    for row in &rows {
        all_groups.push(row.identity_group);
        if row.limit_excluded {
            all_scores.push(miss_score());
        } else if let Some(i) = row.example {
            match example_pos.get(&i) {
                Some(&pos) => all_scores.push(scores[pos].clone()),
                None => all_scores.push(miss_score()),
            }
        } else {
            all_scores.push(miss_score());
        }
    }

    // Subgroups intersected with the read set.
    let read_keys: HashSet<&str> = rows.iter().map(|r| r.key.as_str()).collect();
    let (subgroups, subgroup_unknown_keys) = match &args.subgroups {
        Some(path) => read_subgroups(path, &read_keys)?,
        None => (Vec::new(), 0),
    };
    let mut metrics_subgroups = BTreeMap::new();
    let mut subgroup_denominators = BTreeMap::new();
    for (label, keys) in &subgroups {
        let mut sub_scores = Vec::new();
        let mut sub_groups = Vec::new();
        for (row, score) in rows.iter().zip(all_scores.iter()) {
            if keys.contains(&row.key) {
                sub_scores.push(score.clone());
                sub_groups.push(row.identity_group);
            }
        }
        subgroup_denominators.insert(label.clone(), sub_scores.len());
        metrics_subgroups.insert(
            label.clone(),
            recovery_report(&sub_scores, &sub_groups, args.bootstrap, args.gen_seed),
        );
    }

    // identity_in_train true/false over the eligible queries.
    let mut true_scores = Vec::new();
    let mut true_groups = Vec::new();
    let mut false_scores = Vec::new();
    let mut false_groups = Vec::new();
    for pos in 0..eval_indices.len() {
        if identity_in_train[pos] {
            true_scores.push(scores[pos].clone());
            true_groups.push(groups[pos]);
        } else {
            false_scores.push(scores[pos].clone());
            false_groups.push(groups[pos]);
        }
    }
    let metrics_all = recovery_report(&all_scores, &all_groups, args.bootstrap, args.gen_seed);
    let metrics_eligible = recovery_report(&scores, &groups, args.bootstrap, args.gen_seed);
    let metrics_identity_in_train_true =
        recovery_report(&true_scores, &true_groups, args.bootstrap, args.gen_seed);
    let metrics_identity_in_train_false =
        recovery_report(&false_scores, &false_groups, args.bootstrap, args.gen_seed);
    // Dead-end causes and containment misses over the same evaluated queries.
    let mut diag_compositions: Vec<Composition> = Vec::with_capacity(eval_indices.len());
    let mut diag_patterns: Vec<Vec<MolGraph>> = Vec::with_capacity(eval_indices.len());
    for (pos, &i) in eval_indices.iter().enumerate() {
        diag_compositions.push(validation_set.examples[i].composition);
        diag_patterns.push(
            eval_patterns[pos]
                .iter()
                .map(|p| {
                    MolGraph::new(p.graph.atoms().to_vec(), p.graph.bonds().to_vec())
                        .expect("evaluation patterns rebuild")
                })
                .collect(),
        );
    }
    let dead_ends = aggregate_dead_ends(&outcomes, &diag_compositions, limits);
    let rejected_containment_detail = aggregate_rejected_containment(
        &outcomes,
        &diag_compositions,
        &diag_patterns,
        limits,
        gen_config.containment_node_limit as usize,
    );
    let diagnostics = DiagnosticsInfo {
        mean_distinct: metrics_eligible.mean_distinct,
        mean_unresolved: metrics_eligible.mean_unresolved,
        zero_candidate_queries: metrics_eligible.zero_candidate_queries,
        finished_fraction: metrics_eligible.mean_finished_fraction,
        dead_end_fraction: metrics_eligible.mean_dead_end_fraction,
        truncated_fraction: metrics_eligible.mean_truncated_fraction,
        rejected_replay_fraction: mean_fraction(&scores, |c| c.rejected_replay),
        rejected_containment_fraction: mean_fraction(&scores, |c| c.rejected_containment),
        containment_unresolved_fraction: mean_fraction(&scores, |c| c.containment_unresolved),
        identity_unresolved_fraction: mean_fraction(&scores, |c| c.identity_unresolved),
        rejected_extra_groups: outcomes.iter().map(|o| o.rejected_extra_groups).sum(),
        rejected_missing_groups: outcomes.iter().map(|o| o.rejected_missing_groups).sum(),
        pass_contained_fraction: finished_pass_fraction(&outcomes, |o| o.pass_contained),
        pass_disjoint_fraction: finished_pass_fraction(&outcomes, |o| o.pass_disjoint),
        pass_complete_fraction: finished_pass_fraction(&outcomes, |o| o.pass_complete),
        dead_ends,
        rejected_containment_detail,
        fp_tokens_mean: super::completion_fingerprint::mean_of(
            &fp_tokens.iter().map(|&n| n as f64).collect::<Vec<f64>>(),
        ),
        fp_tokens_max: fp_tokens.into_iter().max().unwrap_or(0),
        fp_entries_dropped_mean: super::completion_fingerprint::mean_of(
            &fp_dropped.iter().map(|&n| n as f64).collect::<Vec<f64>>(),
        ),
        fp_true_missing_mean: if fp_true_missing.is_empty() {
            None
        } else {
            Some(super::completion_fingerprint::mean_of(
                &fp_true_missing.iter().map(|&n| n as f64).collect::<Vec<f64>>(),
            ))
        },
        fp_false_mean: if fp_false.is_empty() {
            None
        } else {
            Some(super::completion_fingerprint::mean_of(
                &fp_false.iter().map(|&n| n as f64).collect::<Vec<f64>>(),
            ))
        },
        fp_noise_retention_rate: fp_noise.as_ref().map(|noise| {
            noise.retention_rate_at_level(args.fp_threshold, args.fp_noise_level)
        }),
        fp_noise_mean_false_tokens: fp_noise.as_ref().map(|noise| {
            noise.mean_false_tokens_at_level(args.fp_threshold, args.fp_noise_level)
        }),
        fp_noise_level: fp_noise
            .as_ref()
            .map(|_| args.fp_noise_level.as_str().to_string()),
    };

    // Prediction lines in validation file order.
    let mut predictions = Vec::with_capacity(rows.len());
    for row in &rows {
        let key_hash = stable_hash(&[row.key.as_str()]);
        if row.limit_excluded {
            let flag = row.example.and_then(|i| strict_by_example.get(&i).copied());
            predictions.push(PredictionLine {
                key_hash,
                source_index: row.source_index,
                status: "limit_excluded".to_string(),
                eligible: false,
                identity_group: row.identity_group,
                identity_in_train: flag,
                atoms: row.atoms,
                pattern_count: 0,
                pattern_atoms: 0,
                rank: None,
                skeleton_rank: None,
                candidates: 0,
                distinct: 0,
                outcome: None,
                top_samples: None,
                top_best_log_prob: None,
                target_samples: None,
                target_best_log_prob: None,
                formula_joined: None,
                formula_after_domain: None,
                formula_after_substructures: None,
                formula_after_completability: None,
                formula_selected: None,
                formula_sampled: None,
                true_formula_stage: None,
                true_excluded_by_train_fit: None,
                true_absent_from_search: None,
                groups_found: None,
                truncated: None,
                infeasible_reason: None,
            });
            continue;
        }
        let Some(i) = row.example else {
            let reason = row
                .skip_reason
                .clone()
                .unwrap_or_else(|| "unknown".to_string());
            predictions.push(PredictionLine {
                key_hash,
                source_index: row.source_index,
                status: format!("out_of_domain:{reason}"),
                eligible: false,
                identity_group: row.identity_group,
                identity_in_train: None,
                atoms: row.atoms,
                pattern_count: 0,
                pattern_atoms: 0,
                rank: None,
                skeleton_rank: None,
                candidates: 0,
                distinct: 0,
                outcome: None,
                top_samples: None,
                top_best_log_prob: None,
                target_samples: None,
                target_best_log_prob: None,
                formula_joined: None,
                formula_after_domain: None,
                formula_after_substructures: None,
                formula_after_completability: None,
                formula_selected: None,
                formula_sampled: None,
                true_formula_stage: None,
                true_excluded_by_train_fit: None,
                true_absent_from_search: None,
                groups_found: None,
                truncated: None,
                infeasible_reason: None,
            });
            continue;
        };
        let pos = example_pos
            .get(&i)
            .copied()
            .expect("kept row was evaluated");
        let outcome = &outcomes[pos];
        let score = &scores[pos];
        let atoms = validation_set.examples[i].target.atoms().len();
        let pattern_atoms: usize = eval_patterns[pos]
            .iter()
            .map(|p| p.graph.atoms().len())
            .sum();
        let (top_samples, top_best_log_prob) = match outcome.candidates.first() {
            Some(top) => (Some(top.samples), Some(top.best_log_prob)),
            None => (None, None),
        };
        let (target_samples, target_best_log_prob) = match score.rank {
            Some(rank) => {
                let target = &outcome.candidates[rank as usize - 1];
                (Some(target.samples), Some(target.best_log_prob))
            }
            None => (None, None),
        };
        let (fj, fad, fsub, fcomp, fsel, fsamp, tstage, texcl, tabsent) =
            if args.formula_source == FormulaSource::Mass {
                match &mass_details[pos] {
                    Some(d) => (
                        Some(d.joined),
                        Some(d.after_domain),
                        Some(d.after_substructures),
                        Some(d.after_completability),
                        Some(d.selected),
                        Some(d.sampled),
                        Some(d.stage.clone()),
                        Some(d.excluded_by_train_fit),
                        Some(d.absent_from_search),
                    ),
                    None => (None, None, None, None, None, None, None, None, None),
                }
            } else {
                (None, None, None, None, None, None, None, None, None)
            };
        predictions.push(PredictionLine {
            key_hash,
            source_index: row.source_index,
            status: "eligible".to_string(),
            eligible: true,
            identity_group: row.identity_group,
            identity_in_train: Some(identity_in_train[pos]),
            atoms,
            pattern_count: eval_patterns[pos].len(),
            pattern_atoms,
            rank: score.rank,
            skeleton_rank: score.skeleton_rank,
            candidates: outcome.candidates.len() as u32,
            distinct: outcome.distinct,
            outcome: Some(OutcomeCounts::from(outcome)),
            top_samples,
            top_best_log_prob,
            target_samples,
            target_best_log_prob,
            formula_joined: fj,
            formula_after_domain: fad,
            formula_after_substructures: fsub,
            formula_after_completability: fcomp,
            formula_selected: fsel,
            formula_sampled: fsamp,
            true_formula_stage: tstage,
            true_excluded_by_train_fit: texcl,
            true_absent_from_search: tabsent,
            groups_found: Some(eval_fg_meta[pos].0),
            truncated: Some(eval_fg_meta[pos].1),
            infeasible_reason: outcome.infeasible.clone(),
        });
    }

    // Mass-arm report additions.
    let is_mass = args.formula_source == FormulaSource::Mass;
    let base_scope = if use_fg {
        FUNCTIONAL_GROUPS_SCOPE
    } else {
        EXPERIMENT_SCOPE
    };
    let scope = if is_mass {
        // The base scope describes the oracle arm; in the mass arm the
        // composition is not supplied and the masses are synthetic, so say
        // exactly that instead of contradicting the suffix.
        format!(
            "{}{MASS_SCOPE_SUFFIX}",
            base_scope
                .replace(
                    "oracle exact composition",
                    "composition hypotheses derived from a synthetic exact neutral mass"
                )
                .replace("not a mass-conditioned,", "not a measured-mass,")
        )
    } else {
        base_scope.to_string()
    };
    let scope = format!(
        "{scope}; substructure_semantics={}",
        args.substructure_semantics.as_str()
    );
    // Evidence scope: which conditioning the model saw and whether the
    // fingerprint was exact, synthetic or predicted.
    let scope = if uses_fp {
        let fp_kind = match args.fp_eval_mode {
            super::completion_fingerprint::FingerprintEvalMode::Exact => "exact",
            super::completion_fingerprint::FingerprintEvalMode::MistLike => "synthetic",
            super::completion_fingerprint::FingerprintEvalMode::Predicted => "predicted",
        };
        format!(
            "{scope}; evidence={} fingerprint={}",
            args.evidence.as_str(),
            fp_kind
        )
    } else {
        format!("{scope}; evidence={}", args.evidence.as_str())
    };
    // Noise-level scope: which histogram set mist_like sampling drew from.
    let scope = if fp_noise.is_some() {
        format!("{scope}; fp_noise_level={}", args.fp_noise_level.as_str())
    } else {
        scope
    };
    let formula_search: Option<FormulaSearchDiagnostics> = if is_mass {
        let n = eval_indices.len().max(1) as f64;
        let mut c_joined = 0usize;
        let mut c_domain = 0usize;
        let mut c_sub = 0usize;
        let mut c_comp = 0usize;
        let mut c_sel = 0usize;
        let mut c_samp = 0usize;
        let mut joined_counts: Vec<usize> = Vec::with_capacity(eval_indices.len());
        let mut surviving_counts: Vec<usize> = Vec::with_capacity(eval_indices.len());
        let mut selected_counts: Vec<usize> = Vec::with_capacity(eval_indices.len());
        let mut rank_vals: Vec<usize> = Vec::new();
        let mut trunc_q = 0usize;
        let mut exh_q = 0usize;
        let mut excl_q = 0usize;
        let mut absent_q = 0usize;
        for d in mass_details.iter().flatten() {
            joined_counts.push(d.joined);
            surviving_counts.push(d.after_completability);
            selected_counts.push(d.selected);
            if d.truncated {
                trunc_q += 1;
            }
            if d.exhausted {
                exh_q += 1;
            }
            if d.excluded_by_train_fit {
                excl_q += 1;
            }
            if d.absent_from_search {
                absent_q += 1;
            }
            if let Some(r) = d.true_rank {
                rank_vals.push(r);
            }
            match d.stage.as_str() {
                "sampled" => {
                    c_joined += 1;
                    c_domain += 1;
                    c_sub += 1;
                    c_comp += 1;
                    c_sel += 1;
                    c_samp += 1;
                }
                "selected" => {
                    c_joined += 1;
                    c_domain += 1;
                    c_sub += 1;
                    c_comp += 1;
                    c_sel += 1;
                }
                "after_completability" => {
                    c_joined += 1;
                    c_domain += 1;
                    c_sub += 1;
                    c_comp += 1;
                }
                "after_substructures" => {
                    c_joined += 1;
                    c_domain += 1;
                    c_sub += 1;
                }
                "after_domain" => {
                    c_joined += 1;
                    c_domain += 1;
                }
                "joined" => {
                    c_joined += 1;
                }
                _ => {}
            }
        }
        // Recovery split by sampled-true.
        let mut samp_true_scores: Vec<QueryScore> = Vec::new();
        let mut samp_true_groups: Vec<u64> = Vec::new();
        let mut samp_false_scores: Vec<QueryScore> = Vec::new();
        let mut samp_false_groups: Vec<u64> = Vec::new();
        for (pos, d) in mass_details.iter().enumerate() {
            let sampled = d.as_ref().is_some_and(|v| v.sampled_true);
            if sampled {
                samp_true_scores.push(scores[pos].clone());
                samp_true_groups.push(groups[pos]);
            } else {
                samp_false_scores.push(scores[pos].clone());
                samp_false_groups.push(groups[pos]);
            }
        }
        Some(FormulaSearchDiagnostics {
            pruning: args.formula_pruning.as_str().to_string(),
            allocation: args.formula_allocation.as_str().to_string(),
            fraction_joined: c_joined as f64 / n,
            fraction_after_domain: c_domain as f64 / n,
            fraction_after_substructures: c_sub as f64 / n,
            fraction_after_completability: c_comp as f64 / n,
            fraction_selected: c_sel as f64 / n,
            fraction_sampled: c_samp as f64 / n,
            true_excluded_by_train_fit_queries: excl_q,
            fraction_true_excluded_by_train_fit: excl_q as f64 / n,
            true_absent_from_search_queries: absent_q,
            fraction_true_absent_from_search: absent_q as f64 / n,
            joined: distribution_of(&joined_counts),
            surviving: distribution_of(&surviving_counts),
            selected: distribution_of(&selected_counts),
            true_rank: distribution_of(&rank_vals),
            truncated_queries: trunc_q,
            search_exhausted_queries: exh_q,
            recovery_sampled_true: recovery_report(
                &samp_true_scores,
                &samp_true_groups,
                args.bootstrap,
                args.gen_seed,
            ),
            recovery_sampled_false: recovery_report(
                &samp_false_scores,
                &samp_false_groups,
                args.bootstrap,
                args.gen_seed,
            ),
        })
    } else {
        None
    };

    let report = ExperimentReport {
        name: args.name.clone(),
        arm: args.arm.as_str().to_string(),
        scope,
        args: args.clone(),
        train_provenance,
        validation_provenance,
        versions: VersionInfo {
            grammar: COMPLETION_GRAMMAR_VERSION.to_string(),
            data: COMPLETION_DATA_VERSION.to_string(),
            model: COMPLETION_MODEL_VERSION.to_string(),
            chemistry: CHEMISTRY_VERSION.to_string(),
            pattern_source: args.patterns.as_str().to_string(),
            functional_groups: use_fg.then(|| FUNCTIONAL_GROUPS_VERSION.to_string()),
            aromaticity: use_fg.then(|| AROMATICITY_VERSION.to_string()),
        },
        model_config: model_config.clone(),
        // The effective trainer configuration (the checkpoint's own after
        // --load), never the resume CLI values.
        train_config: effective_train_config,
        domain: DomainInfo {
            max_atoms,
            max_closures,
            work_limit,
        },
        accounting: AccountingInfo {
            validation_read: rows.len(),
            validation_kept: eval_indices.len(),
            validation_skipped: validation_set.skipped.clone(),
            limit_excluded,
            train_read: train_file.molecules.len(),
            train_kept: train_keep,
            train_skipped,
            train_limit_excluded: train_full_kept - train_keep,
            overlap_strict,
            overlap_skeleton,
            subgroup_unknown_keys,
            subgroup_denominators,
            train_excluded_identity_groups: train_excluded_molecules,
        },
        model_preset: args.model.clone(),
        eval_subset: eval_subset.len(),
        curve,
        checkpoint_evaluated,
        best_step,
        best_eval_nll_per_example: best_nll,
        generation: GenerationInfo {
            config: gen_config,
            gen_batch: args.gen_batch,
            seconds: gen_seconds,
            trajectories_per_second: if gen_seconds > 0.0 {
                trajectories_total / gen_seconds
            } else {
                0.0
            },
        },
        eval_patterns: eval_pattern_stats,
        metrics_all,
        metrics_eligible,
        metrics_subgroups,
        metrics_identity_in_train_true,
        metrics_identity_in_train_false,
        diagnostics,
        train_seconds,
        seconds_per_100_steps: if steps_done == 0 {
            0.0
        } else {
            train_seconds / steps_done as f64 * 100.0
        },
        formula_search,
        predictions,
        outcomes,
    };
    // Accounting identities: for the mass arm the executed trajectories may
    // sit below the requested total (floor remainder unused), so allow
    // `<= K` there while keeping every other identity exact.
    if is_mass {
        check_accounting_mass(&report, report.predictions.len(), args.trajectories)?;
    } else {
        check_accounting(&report, report.predictions.len(), args.trajectories)?;
    }

    std::fs::write(
        args.out.join("report.json"),
        serde_json::to_string_pretty(&report)?,
    )?;
    let mut lines = String::new();
    for line in &report.predictions {
        lines.push_str(&serde_json::to_string(line)?);
        lines.push('\n');
    }
    std::fs::write(args.out.join("predictions.jsonl"), lines)?;
    // Candidate dump for an external re-ranking tool: one JSON object per
    // evaluation query with the target typed graph, the query composition,
    // the query's source position (so the tool matches queries to
    // fingerprints by position instead of by structure) and every accepted
    // candidate in rank order. Refused inside the repo (see
    // `check_dump_path`).
    if let Some(dump) = &args.dump_candidates {
        let mut dump_lines = String::new();
        for (pos, &i) in eval_indices.iter().enumerate() {
            let example = &validation_set.examples[i];
            let outcome = &report.outcomes[pos];
            let candidates: Vec<serde_json::Value> = outcome
                .candidates
                .iter()
                .map(|c| {
                    serde_json::json!({
                        "atoms": c.graph.atoms(),
                        "bonds": c.graph.bonds(),
                        "samples": c.samples,
                        "best_log_prob": c.best_log_prob,
                    })
                })
                .collect();
            let line = serde_json::json!({
                "target": {
                    "atoms": example.target.atoms(),
                    "bonds": example.target.bonds(),
                },
                "composition": example.composition,
                "source_index": example.source_index,
                "candidates": candidates,
            });
            dump_lines.push_str(&serde_json::to_string(&line)?);
            dump_lines.push('\n');
        }
        if let Some(parent) = dump.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        std::fs::write(dump, dump_lines)?;
    }
    Ok(report)
}
