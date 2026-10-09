//! Spectrum-conditioned molecular completion: train and evaluate.
//!
//! Trains a [`CompletionModel`] on molecules with measured MS/MS spectra and
//! evaluates generation from four inputs per query: a `morgan4096`
//! fingerprint (true bits, or probabilities predicted by an external model
//! such as MIST and supplied as a JSONL file), the fragment peaks, the adduct
//! and the neutral mass (the measured precursor m/z minus the adduct shift).
//!
//! ```text
//! cargo run --release --example ms2_spectral_completion -- \
//!   --train data/ms2/specgen/msgym_train.json --train-fp data/ms2/specgen/msgym_train_fp.json \
//!   --validation data/ms2/specgen/msgym_validation.json \
//!   --validation-fp data/ms2/specgen/msgym_validation_fp.json \
//!   --steps 20000 --save data/ms2/specgen/run/model.ckpt --out data/ms2/specgen/run
//! ```
//!
//! Options (defaults in brackets):
//!
//! * `--evidence fingerprint+spectrum|fingerprint|spectrum|none` [both]: which
//!   inputs the model owns an encoder for. `none` is the formula-only control.
//! * `--predictions FILE` (repeatable): JSONL of `{"id": "MassSpecGymID0000042",
//!   "bits": [[index, probability], ...]}` per spectrum (the id's digits are
//!   the export's `spectrum_id`).
//! * `--fp-train exact|predicted|mist_like|channel|predicted_channel` [exact],
//!   `--fp-eval ...` [same as --fp-train], `--fp-threshold P` [0.1]:
//!   where a query's fingerprint comes from. A spectrum without a prediction
//!   uses an empty fingerprint in both training and evaluation; missing
//!   predictions never substitute answer-derived bits and are counted.
//!   `mist_like` degrades the true bits with the measured noise
//!   of a real predictor (`--fp-noise FILE` from
//!   `tools/ms2/mist_noise_from_jsonl.py`), so a molecule with no spectrum of
//!   its own still trains on a realistic fingerprint; training redraws the
//!   noise every step, evaluation uses one fixed draw.
//!   `channel` degrades the true bits with a per-bit error channel fitted on
//!   real predictions for molecules the predictor never saw (`--fp-channel
//!   FILE` from `tools/ms2/fit_fingerprint_channel.py`): unlike `mist_like`
//!   it keeps which bits the predictor is right about and how bad a whole
//!   molecule can be, so structure-only molecules train on what a real
//!   prediction looks like. `predicted_channel` uses the real prediction of
//!   a spectrum when there is one and the channel otherwise (structure-only
//!   molecules, and spectra left out of `--predictions` because the
//!   predictor trained on their molecule).
//! * `--formula oracle|mass` [mass]: the composition a query is generated with:
//!   the target's own (an oracle input) or every formula hypothesis of the
//!   neutral mass within `--mass-ppm-tenths` [100] (at most `--hypotheses` [8]).
//! * `--element-predictions FILE` (repeatable): JSONL of `{"id": "<spectrum id>",
//!   "elements": {"C": x, "H": x, ...}}` with a predictor's `ln(1 + count)` per
//!   element for a query (same id convention as `--predictions`). With
//!   `--formula mass`, a query that has one orders its formula hypotheses by
//!   distance to those counts instead of by mass residual and splits the
//!   search budget by `exp(-distance / --formula-temperature)` [0.25]
//!   (`ElementPrior`); a query without one keeps the default order and is
//!   counted. An element missing from a line counts as predicted absent.
//! * `--steps`, `--batch` [16], `--lr` [3e-4], `--seed` [1], `--eval-every`
//!   [1000], `--eval-subset` [256], `--report-every` [100].
//!   With `--load`, an explicit `--lr` overrides the saved rate; omitting it
//!   preserves the checkpoint's rate. Reports record the effective rate.
//!   `--eval-offset` [0] skips this many validation molecules for NLL and
//!   checkpoint selection; generation still uses the first `--queries`.
//! * `--returned N` [25]: distinct candidates kept per query. 25 is the
//!   competition shortlist; a larger value keeps the whole accepted pool, so
//!   `--dump-candidates` can be re-ranked over everything the model found
//!   (`tools/ms2/completion_fp_rerank.py` reports top-1/10/25 from the
//!   re-ranked list). The reported top-k always counts the first 25 of the
//!   model's own order, whatever this is.
//! * `--survival N` [0]: instead of sampling, run teacher forcing on the
//!   first N validation queries and write the per-step log-probability of
//!   the target's own action (the target-survival curve) to
//!   `<out>/<tag>_survival.json`. This says whether the target's trace is
//!   uniformly improbable (a modelling limit) or improbable at only a few
//!   steps (which a search or repair strategy could address).
//! * `--scaffold FILE`: supply each molecule's Murcko scaffold (from
//!   `tools/ms2/export_scaffold_patterns.py`, which writes the scaffold's
//!   atom indices in the molecule's own numbering) as the query's single
//!   substructure, carrying the parent's atom types. The scaffold is an
//!   oracle input taken from the answer: the question it answers is whether
//!   knowing the scaffold would let the decoder reach the molecule, not
//!   whether a spectrum model can predict one. A scaffold that does not fit
//!   the pattern encoder (over [`PATTERN_SLOTS`] atoms) or is empty (an
//!   acyclic molecule)
//!   is supplied as no pattern and counted. Both training and evaluation
//!   use it when given, and acceptance then requires the candidate to
//!   contain it.
//! * `--temperature X` [1.0]: sampling temperature. The recorded
//!   log-probability stays the temperature-one, grammar-masked score.
//! * `--beam W` [0]: search with a beam of `W` rows per query instead of
//!   `--trajectories K` independent samples (0 keeps sampling). The rows are
//!   split across the formula hypotheses exactly as trajectories are, so
//!   `--beam W` and `--trajectories W` spend the same decoder rows; the
//!   report carries `decoder_row_steps` for both arms so the comparison is
//!   by work and not by `K`. Every other stage — formula enumeration from
//!   the neutral mass, acceptance, the `--returned` pool, the dumps — is the
//!   same code. `--temperature` and `--gen-seed` are unused with a beam.
//! * `--queries N` [200]: evaluation queries (the first N kept validation
//!   molecules, first exported spectrum each); `--trajectories K` [64];
//!   `--gen-batch` [8]; `--gen-seed` [7].
//! * `--extra-train FILE --extra-train-fp FILE`: structure-only molecules
//!   (`tools/ms2/export_molecules_fp.py`) added to the training pool. They
//!   use --fp-train and no spectral evidence (predicted mode supplies empty
//!   fingerprints because these molecules have no predicted spectra); any whose
//!   identity is a validation or training molecule is dropped and counted.
//! * `--evidence-dropout P` [0]: during training each query independently
//!   loses its fingerprint with probability `P` and its spectral evidence
//!   with probability `P`, so one model can be evaluated with an input
//!   removed. `--eval-drop fingerprint|spectrum|both` (repeatable) removes
//!   that input from every evaluation query (the model then sees an empty
//!   fingerprint or no spectral evidence); the formula search still uses the
//!   neutral mass.
//! * `--save-every N` [0]: write the current weights to `--save` every `N`
//!   training steps (atomically, with the formula artifacts attached), so an
//!   interrupted run loses at most `N` steps. `--until-step T` trains until
//!   the checkpoint's own step count reaches `T` instead of for `--steps`
//!   more steps: rerunning the same command with `--load` set to the saved
//!   file resumes where the run stopped and ends at the same total.
//! * `--load FILE`, `--save FILE` (plus `<save>.best` on validation
//!   improvement), `--eval-only`, `--out DIR` (writes `report.json` and
//!   `predictions.jsonl`), `--tag NAME` [eval] (prefix of the two files).
//!
//! * `--dump-candidates FILE`: additionally write the shortlists in the
//!   `--dump-candidates` line format of the completion experiment (`target`,
//!   `composition`, `candidates`), plus `FILE.bits.json` (the true on-bits of
//!   each query's molecule, one entry per line) and `FILE.panel.json` (per
//!   query the typed graph, `fp_true` and `fp_pred_mean`, the fingerprint
//!   the query supplied), the inputs of `tools/ms2/completion_fp_rerank.py`.
//!
//! `predictions.jsonl` holds one line per query with the inputs as the model
//! saw them, the target, and every returned candidate (atom types, bonds,
//! formula, mass and its residual against the input mass, and whether it is
//! the target). The crate cannot compute a `morgan4096` fingerprint, so the
//! agreement of a candidate's own fingerprint with the input fingerprint is
//! not scored here: `tools/ms2/completion_fp_rerank.py` scores it from the
//! `--dump-candidates` files.

#![cfg(feature = "backend")]
#![recursion_limit = "256"]

use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use mamba3::backend::{Device, check_launches};
use mamba3::backends::Auto;
use mamba3::error::{Error, Result};
use mamba3::models::ms2::chem::{Composition, composition_mass, tolerance};
use mamba3::models::ms2::completion_data::{
    CompletionSet, ExtractionConfig, PatternSource, same_identity,
};
use mamba3::models::ms2::completion_experiment::fit_and_attach;
use mamba3::models::ms2::completion_fingerprint::{
    FINGERPRINT_SLOTS, FingerprintChannel, FingerprintMode, FingerprintNoise,
    FingerprintNoiseLevel, FingerprintStore, SparseFingerprint,
};
use mamba3::models::ms2::completion_formula::{
    CompletionSearch, ElementPrior, FormulaAllocation, FormulaPruning, MassQuery, formula_text,
    run_mass_completion_search_with_prior,
};
use mamba3::models::ms2::completion_model::{
    BeamStats, CompletionGenerationConfig, CompletionModelConfig, CompletionRequest,
    CompletionTrainConfig, CompletionTrainer, PATTERN_SLOTS, SubstructureSemantics,
};
use mamba3::models::ms2::completion_spectrum::{
    ADDUCT_CONVERSION_ERROR_UDA, SPECTRUM_SLOTS, SpectrumEvidence, completion_adduct,
    neutral_mass_of, precursor_mz_of,
};
use mamba3::models::ms2::dataset::{ExportFile, ExportSpectrum};
use mamba3::models::ms2::grammar::{CANONICAL_WORK_LIMIT, Limits, Token};
use mamba3::models::ms2::graph::MolGraph;
use serde_json::json;

type R = Auto;

/// Noise draw of every evaluation fingerprint: one fixed realisation, so two
/// evaluations of one checkpoint see the same inputs.
const EVAL_DRAW: u64 = 1;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum FpSource {
    Exact,
    Predicted,
    MistLike,
    /// True bits degraded by the fitted per-bit channel.
    Channel,
    /// The real prediction when the spectrum has one, the channel otherwise.
    PredictedChannel,
}

impl FpSource {
    /// Whether the source reads `--predictions`.
    fn uses_predictions(self) -> bool {
        matches!(self, FpSource::Predicted | FpSource::PredictedChannel)
    }

    /// Whether the source samples `--fp-channel`.
    fn uses_channel(self) -> bool {
        matches!(self, FpSource::Channel | FpSource::PredictedChannel)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum FormulaSource {
    Oracle,
    Mass,
}

struct Args {
    train: Option<PathBuf>,
    train_fp: Option<PathBuf>,
    validation: PathBuf,
    validation_fp: PathBuf,
    predictions: Vec<PathBuf>,
    use_fingerprint: bool,
    use_spectrum: bool,
    fp_train: FpSource,
    fp_eval: FpSource,
    fp_noise: Option<PathBuf>,
    fp_channel: Option<PathBuf>,
    scaffold: Option<PathBuf>,
    fp_threshold: f32,
    formula: FormulaSource,
    mass_ppm_tenths: u32,
    hypotheses: u32,
    element_predictions: Vec<PathBuf>,
    formula_temperature: f32,
    steps: usize,
    batch: usize,
    lr: Option<f32>,
    seed: u64,
    eval_every: usize,
    eval_subset: usize,
    eval_offset: usize,
    report_every: usize,
    queries: usize,
    returned: u32,
    temperature: f32,
    survival: usize,
    trajectories: u32,
    beam: u32,
    gen_batch: usize,
    gen_seed: u64,
    load: Option<PathBuf>,
    save: Option<PathBuf>,
    eval_only: bool,
    out: Option<PathBuf>,
    tag: String,
    max_atoms: usize,
    max_closures: usize,
    evidence_dropout: f64,
    eval_drop_fingerprint: bool,
    eval_drop_spectrum: bool,
    dump_candidates: Option<PathBuf>,
    extra_train: Option<PathBuf>,
    extra_train_fp: Option<PathBuf>,
    save_every: usize,
    until_step: Option<u64>,
    /// Per-step structural progress features: a fresh model is built with
    /// them, and a checkpoint without them is resumed with them through the
    /// non-strict load that initialises only the new projection.
    progress_features: bool,
}

fn usage(message: &str) -> ! {
    eprintln!(
        "ms2_spectral_completion: {message}\nsee the header of examples/ms2_spectral_completion.rs for the options"
    );
    std::process::exit(2)
}

fn default_args() -> Args {
    Args {
        train: None,
        train_fp: None,
        validation: PathBuf::new(),
        validation_fp: PathBuf::new(),
        predictions: Vec::new(),
        use_fingerprint: true,
        use_spectrum: true,
        fp_train: FpSource::Exact,
        fp_eval: FpSource::Exact,
        fp_noise: None,
        fp_channel: None,
        scaffold: None,
        fp_threshold: 0.1,
        formula: FormulaSource::Mass,
        mass_ppm_tenths: 100,
        hypotheses: 8,
        element_predictions: Vec::new(),
        formula_temperature: 0.25,
        steps: 0,
        batch: 16,
        lr: None,
        seed: 1,
        eval_every: 1000,
        eval_subset: 256,
        eval_offset: 0,
        report_every: 100,
        queries: 200,
        returned: 25,
        temperature: 1.0,
        survival: 0,
        trajectories: 64,
        beam: 0,
        gen_batch: 8,
        gen_seed: 7,
        load: None,
        save: None,
        eval_only: false,
        out: None,
        tag: "eval".to_string(),
        max_atoms: 32,
        max_closures: 6,
        evidence_dropout: 0.0,
        eval_drop_fingerprint: false,
        eval_drop_spectrum: false,
        dump_candidates: None,
        extra_train: None,
        extra_train_fp: None,
        save_every: 0,
        until_step: None,
        progress_features: false,
    }
}

fn parse_args() -> Args {
    parse_args_from(std::env::args().skip(1))
}

fn parse_args_from(args: impl Iterator<Item = String>) -> Args {
    let mut a = default_args();
    let mut fp_eval_explicit = false;
    let mut it = args;
    let fp_source = |text: &str| match text {
        "exact" => FpSource::Exact,
        "predicted" => FpSource::Predicted,
        "mist_like" => FpSource::MistLike,
        "channel" => FpSource::Channel,
        "predicted_channel" => FpSource::PredictedChannel,
        other => usage(&format!("unknown fingerprint source {other:?}")),
    };
    while let Some(flag) = it.next() {
        let mut value = || {
            it.next()
                .unwrap_or_else(|| usage(&format!("{flag} needs a value")))
        };
        fn num<T: std::str::FromStr>(flag: &str, text: String) -> T {
            text.parse()
                .unwrap_or_else(|_| usage(&format!("{flag}: cannot parse {text:?}")))
        }
        match flag.as_str() {
            "--train" => a.train = Some(value().into()),
            "--train-fp" => a.train_fp = Some(value().into()),
            "--validation" => a.validation = value().into(),
            "--validation-fp" => a.validation_fp = value().into(),
            "--predictions" => a.predictions.push(value().into()),
            "--evidence" => match value().as_str() {
                "fingerprint+spectrum" => (a.use_fingerprint, a.use_spectrum) = (true, true),
                "fingerprint" => (a.use_fingerprint, a.use_spectrum) = (true, false),
                "spectrum" => (a.use_fingerprint, a.use_spectrum) = (false, true),
                "none" => (a.use_fingerprint, a.use_spectrum) = (false, false),
                other => usage(&format!("unknown evidence {other:?}")),
            },
            "--fp-train" => a.fp_train = fp_source(&value()),
            "--fp-eval" => {
                a.fp_eval = fp_source(&value());
                fp_eval_explicit = true;
            }
            "--fp-noise" => a.fp_noise = Some(value().into()),
            "--fp-channel" => a.fp_channel = Some(value().into()),
            "--scaffold" => a.scaffold = Some(value().into()),
            "--fp-threshold" => a.fp_threshold = num(&flag, value()),
            "--formula" => {
                a.formula = match value().as_str() {
                    "oracle" => FormulaSource::Oracle,
                    "mass" => FormulaSource::Mass,
                    other => usage(&format!("unknown formula source {other:?}")),
                }
            }
            "--mass-ppm-tenths" => a.mass_ppm_tenths = num(&flag, value()),
            "--hypotheses" => a.hypotheses = num(&flag, value()),
            "--element-predictions" => a.element_predictions.push(value().into()),
            "--formula-temperature" => a.formula_temperature = num(&flag, value()),
            "--steps" => a.steps = num(&flag, value()),
            "--batch" => a.batch = num(&flag, value()),
            "--lr" => a.lr = Some(num(&flag, value())),
            "--seed" => a.seed = num(&flag, value()),
            "--eval-every" => a.eval_every = num(&flag, value()),
            "--eval-subset" => a.eval_subset = num(&flag, value()),
            "--eval-offset" => a.eval_offset = num(&flag, value()),
            "--report-every" => a.report_every = num(&flag, value()),
            "--queries" => a.queries = num(&flag, value()),
            "--returned" => a.returned = num(&flag, value()),
            "--temperature" => a.temperature = num(&flag, value()),
            "--survival" => a.survival = num(&flag, value()),
            "--trajectories" => a.trajectories = num(&flag, value()),
            "--beam" => a.beam = num(&flag, value()),
            "--gen-batch" => a.gen_batch = num(&flag, value()),
            "--gen-seed" => a.gen_seed = num(&flag, value()),
            "--load" => a.load = Some(value().into()),
            "--save" => a.save = Some(value().into()),
            "--eval-only" => a.eval_only = true,
            "--out" => a.out = Some(value().into()),
            "--tag" => a.tag = value(),
            "--max-atoms" => a.max_atoms = num(&flag, value()),
            "--max-closures" => a.max_closures = num(&flag, value()),
            "--dump-candidates" => a.dump_candidates = Some(value().into()),
            "--save-every" => a.save_every = num(&flag, value()),
            "--until-step" => a.until_step = Some(num(&flag, value())),
            "--progress-features" => a.progress_features = true,
            "--extra-train" => a.extra_train = Some(value().into()),
            "--extra-train-fp" => a.extra_train_fp = Some(value().into()),
            "--evidence-dropout" => a.evidence_dropout = num(&flag, value()),
            "--eval-drop" => match value().as_str() {
                "fingerprint" => a.eval_drop_fingerprint = true,
                "spectrum" => a.eval_drop_spectrum = true,
                "both" => (a.eval_drop_fingerprint, a.eval_drop_spectrum) = (true, true),
                other => usage(&format!("unknown --eval-drop {other:?}")),
            },
            other => usage(&format!("unknown argument {other:?}")),
        }
    }
    if !fp_eval_explicit {
        a.fp_eval = a.fp_train;
    }
    if a.validation.as_os_str().is_empty() || a.validation_fp.as_os_str().is_empty() {
        usage("--validation and --validation-fp are required");
    }
    if a.save_every > 0 && a.save.is_none() {
        usage("--save-every needs --save");
    }
    if !a.eval_only
        && (a.steps > 0 || a.until_step.is_some())
        && (a.train.is_none() || a.train_fp.is_none())
    {
        usage("training needs --train and --train-fp");
    }
    if !(0.0..1.0).contains(&a.evidence_dropout) {
        usage("--evidence-dropout must be in [0, 1)");
    }
    if a.batch == 0 || a.gen_batch == 0 {
        usage("--batch and --gen-batch must be positive");
    }
    a
}

/// SplitMix64 (the crate's generator is crate-private): the train shuffle
/// and the per-step spectrum pick.
struct Mix(u64);

impl Mix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, bound: usize) -> usize {
        (self.next() % bound as u64) as usize
    }

    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// One loaded fold: the export, the kept examples and the true bits.
struct Fold {
    file: ExportFile,
    set: CompletionSet,
    bits: FingerprintStore,
    /// Murcko scaffold atom indices per export molecule, when `--scaffold`
    /// named a sidecar for this fold (empty otherwise).
    scaffolds: Vec<Vec<usize>>,
}

/// Scaffold atom indices per export molecule, checked against the export's
/// own keys so a reordered sidecar is an error rather than a silent
/// misattribution (the rule the fingerprint sidecar follows).
fn load_scaffolds(path: &Path, file: &ExportFile) -> Result<Vec<Vec<usize>>> {
    let value: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path)?)?;
    let lists = value["atoms_by_molecule"]
        .as_array()
        .ok_or_else(|| Error::config(format!("{}: missing atoms_by_molecule", path.display())))?;
    if lists.len() != file.molecules.len() {
        return Err(Error::config(format!(
            "{}: holds {} entries for an export of {} molecules",
            path.display(),
            lists.len(),
            file.molecules.len()
        )));
    }
    if let Some(keys) = value["keys_by_molecule"].as_array() {
        for (i, (key, molecule)) in keys.iter().zip(file.molecules.iter()).enumerate() {
            let want = format!("{}|{}", molecule.key, molecule.identity_group);
            if key.as_str() != Some(want.as_str()) {
                return Err(Error::config(format!(
                    "{}: entry {i} is {key} for export molecule {want}",
                    path.display()
                )));
            }
        }
    }
    let mut out = Vec::with_capacity(lists.len());
    for (i, list) in lists.iter().enumerate() {
        let atoms = list
            .as_array()
            .ok_or_else(|| Error::config(format!("{}: entry {i} is not a list", path.display())))?;
        let mut indices = Vec::with_capacity(atoms.len());
        for value in atoms {
            indices.push(value.as_u64().ok_or_else(|| {
                Error::config(format!("{}: entry {i} holds a non-integer", path.display()))
            })? as usize);
        }
        out.push(indices);
    }
    Ok(out)
}

fn load_fold(path: &Path, fp_path: &Path, limits: Limits) -> Result<Fold> {
    let file = ExportFile::load(path)?;
    let set = CompletionSet::from_export(&file, limits, CANONICAL_WORK_LIMIT)?;
    let bits = FingerprintStore::load(fp_path)?;
    bits.assert_molecule_count(file.molecules.len())?;
    let keys: Vec<String> = file
        .molecules
        .iter()
        .map(|m| format!("{}|{}", m.key, m.identity_group))
        .collect();
    bits.assert_keys_match(&keys)?;
    Ok(Fold {
        file,
        set,
        bits,
        scaffolds: Vec::new(),
    })
}

/// The scaffold pattern of example `index` of `fold`: the induced subgraph of
/// the target on its scaffold atoms, carrying the parent's atom types. `None`
/// when no sidecar was loaded, the scaffold is empty (an acyclic molecule) or
/// it does not fit the pattern encoder.
fn scaffold_pattern(fold: &Fold, index: usize) -> Option<MolGraph> {
    let example = fold.set.examples.get(index)?;
    let atoms = fold.scaffolds.get(example.source_index)?;
    if atoms.is_empty() || atoms.len() > PATTERN_SLOTS {
        return None;
    }
    example.target.induced(atoms).ok()
}

/// Predicted fingerprints by spectrum id: `(bit, probability)` pairs.
/// Predicted `ln(1 + count)` per element by spectrum id, in `ELEMENTS` order.
fn load_element_predictions(paths: &[PathBuf]) -> Result<HashMap<u64, [f32; 10]>> {
    let mut out = HashMap::new();
    for path in paths {
        let reader = std::io::BufReader::new(std::fs::File::open(path)?);
        for (number, line) in reader.lines().enumerate() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let bad =
                |what: &str| Error::config(format!("{}:{}: {what}", path.display(), number + 1));
            let value: serde_json::Value = serde_json::from_str(&line)?;
            let id = value["id"]
                .as_str()
                .ok_or_else(|| bad("missing string id"))?;
            let digits: String = id.chars().filter(char::is_ascii_digit).collect();
            let id: u64 = digits.parse().map_err(|_| bad("id holds no number"))?;
            let elements = value["elements"]
                .as_object()
                .ok_or_else(|| bad("missing elements object"))?;
            let mut counts = [0.0f32; 10];
            for (symbol, v) in elements {
                let e = mamba3::models::ms2::chem::ELEMENTS
                    .iter()
                    .position(|el| el.symbol == symbol.as_str())
                    .ok_or_else(|| bad("unknown element symbol"))?;
                let x = v.as_f64().ok_or_else(|| bad("element value is not a number"))? as f32;
                if !x.is_finite() {
                    return Err(bad("element value is not finite"));
                }
                counts[e] = x.max(0.0);
            }
            out.insert(id, counts);
        }
    }
    Ok(out)
}

fn load_predictions(paths: &[PathBuf]) -> Result<HashMap<u64, Vec<(u16, f32)>>> {
    let mut out = HashMap::new();
    for path in paths {
        let reader = std::io::BufReader::new(std::fs::File::open(path)?);
        for (number, line) in reader.lines().enumerate() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let bad =
                |what: &str| Error::config(format!("{}:{}: {what}", path.display(), number + 1));
            let value: serde_json::Value = serde_json::from_str(&line)?;
            let id = value["id"]
                .as_str()
                .ok_or_else(|| bad("missing string id"))?;
            let digits: String = id.chars().filter(char::is_ascii_digit).collect();
            let id: u64 = digits.parse().map_err(|_| bad("id holds no number"))?;
            let bits = value["bits"]
                .as_array()
                .ok_or_else(|| bad("missing bits list"))?;
            let mut pairs = Vec::with_capacity(bits.len());
            for pair in bits {
                let index = pair[0]
                    .as_u64()
                    .ok_or_else(|| bad("bit index is not an integer"))?;
                let probability = pair[1]
                    .as_f64()
                    .ok_or_else(|| bad("probability is not a number"))?;
                if index >= 4096 {
                    return Err(bad("bit index is past 4096"));
                }
                pairs.push((index as u16, (probability as f32).clamp(0.0, 1.0)));
            }
            out.insert(id, pairs);
        }
    }
    Ok(out)
}

/// Spectral evidence of one exported spectrum: its peaks, precursor and
/// adduct, with the neutral mass derived from the measured precursor.
fn evidence_of(spectrum: &ExportSpectrum) -> Result<SpectrumEvidence> {
    let neutral =
        neutral_mass_of(spectrum.precursor_mz_udalton, spectrum.adduct).ok_or_else(|| {
            Error::config(format!(
                "spectrum {}: no neutral mass for precursor {} under adduct id {}",
                spectrum.spectrum_id,
                spectrum.precursor_mz_udalton,
                spectrum.adduct
            ))
        })?;
    Ok(SpectrumEvidence {
        peaks: spectrum
            .mz_udalton
            .iter()
            .zip(spectrum.intensity.iter())
            .map(|(&mz, &intensity)| (mz, intensity as f32))
            .collect(),
        precursor_mz: spectrum.precursor_mz_udalton,
        adduct: spectrum.adduct,
        neutral_mass: neutral,
    })
}

/// The fingerprint a query carries and where it came from.
#[allow(clippy::too_many_arguments)]
fn fingerprint_of(
    source: FpSource,
    true_bits: &[u16],
    spectrum_id: Option<u64>,
    key: &str,
    draw: u64,
    predictions: &HashMap<u64, Vec<(u16, f32)>>,
    noise: Option<&FingerprintNoise>,
    channel: Option<&FingerprintChannel>,
    threshold: f32,
    seed: u64,
) -> Result<(SparseFingerprint, &'static str)> {
    let from_channel = || -> Result<(SparseFingerprint, &'static str)> {
        let channel = channel.ok_or_else(|| {
            Error::config("a channel fingerprint source needs --fp-channel".to_string())
        })?;
        Ok((channel.sample(true_bits, seed, key, draw)?, "channel"))
    };
    match source {
        FpSource::Exact => Ok((SparseFingerprint::from_bits(true_bits)?, "exact")),
        FpSource::Predicted => match spectrum_id.and_then(|id| predictions.get(&id)) {
            Some(pairs) => Ok((
                SparseFingerprint::from_probabilities(pairs, threshold)?,
                "predicted",
            )),
            None => Ok((
                SparseFingerprint {
                    entries: Vec::new(),
                },
                "missing",
            )),
        },
        FpSource::MistLike => {
            let noise = noise.ok_or_else(|| {
                Error::config("a mist_like fingerprint source needs --fp-noise".to_string())
            })?;
            let fp = noise.sample_at_level(
                true_bits,
                seed,
                key,
                draw,
                threshold,
                FingerprintNoiseLevel::Spectrum,
            )?;
            Ok((fp, "mist_like"))
        }
        FpSource::Channel => from_channel(),
        FpSource::PredictedChannel => match spectrum_id.and_then(|id| predictions.get(&id)) {
            Some(pairs) => Ok((
                SparseFingerprint::from_probabilities(pairs, threshold)?,
                "predicted",
            )),
            None => from_channel(),
        },
    }
}

/// The per-query pattern lists of a chunk: the scaffold when one was loaded
/// and fits the encoder, else no pattern. Owned, so the caller takes slices.
fn chunk_patterns(fold: &Fold, picks: &[(usize, usize)]) -> Vec<Vec<MolGraph>> {
    picks
        .iter()
        .map(|&(i, _)| {
            scaffold_pattern(fold, i)
                .map(|g| vec![g])
                .unwrap_or_default()
        })
        .collect()
}

fn graph_json(graph: &MolGraph) -> serde_json::Value {
    json!({
        "atoms": graph.atoms(),
        "bonds": graph.bonds().iter().map(|&(a, b, o)| json!([a, b, o])).collect::<Vec<_>>(),
    })
}

fn main() -> Result<()> {
    let args = parse_args();
    let device = Device::<R>::default();
    let limits = Limits::new(args.max_atoms, args.max_closures)?;
    let started = Instant::now();

    let mut validation = load_fold(&args.validation, &args.validation_fp, limits)?;
    if let Some(path) = &args.scaffold {
        let sidecar = path.with_file_name(format!(
            "{}_scaffold.json",
            args.validation
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default()
        ));
        let sidecar = if path.is_dir() { sidecar } else { path.clone() };
        validation.scaffolds = load_scaffolds(&sidecar, &validation.file)?;
        let supplied = (0..validation.set.examples.len())
            .filter(|&i| scaffold_pattern(&validation, i).is_some())
            .count();
        eprintln!(
            "scaffolds: {} of {} validation examples carry one that fits the encoder",
            supplied,
            validation.set.examples.len()
        );
    }
    eprintln!(
        "validation: {} molecules in the file, {} kept, skipped {:?}",
        validation.file.molecules.len(),
        validation.set.examples.len(),
        validation.set.skipped
    );
    let noise = match &args.fp_noise {
        Some(path) => {
            let noise = FingerprintNoise::load(path)?;
            eprintln!(
                "fingerprint noise: {} spectra, {} molecules",
                noise.n_spectra, noise.n_molecules
            );
            Some(noise)
        }
        None => None,
    };
    let predictions = load_predictions(&args.predictions)?;
    if !args.predictions.is_empty() {
        eprintln!("predictions: {} spectra", predictions.len());
    }
    let element_predictions = load_element_predictions(&args.element_predictions)?;
    if !args.element_predictions.is_empty() {
        eprintln!("element predictions: {} spectra", element_predictions.len());
    }
    if (args.fp_train.uses_predictions() || args.fp_eval.uses_predictions())
        && predictions.is_empty()
    {
        usage("a predicted fingerprint source needs --predictions");
    }
    let channel_wanted = args.fp_train.uses_channel() || args.fp_eval.uses_channel();
    let channel = match args.fp_channel.as_ref().filter(|_| channel_wanted) {
        Some(path) => {
            let channel = FingerprintChannel::load(path)?;
            // A token the channel emits must be one the threshold keeps, and
            // a prediction the threshold keeps must be one the channel models.
            if (channel.threshold - args.fp_threshold).abs() > 1e-6 {
                return Err(Error::config(format!(
                    "--fp-channel was fitted at threshold {} but --fp-threshold is {}",
                    channel.threshold, args.fp_threshold
                )));
            }
            eprintln!(
                "fingerprint channel: {} classes, weights {:?}",
                channel.classes, channel.weights
            );
            Some(channel)
        }
        None => None,
    };
    if channel_wanted && channel.is_none() {
        usage("a channel fingerprint source needs --fp-channel");
    }
    if (args.fp_train == FpSource::MistLike || args.fp_eval == FpSource::MistLike)
        && noise.is_none()
    {
        usage("a mist_like fingerprint source needs --fp-noise");
    }

    let mut model_config = CompletionModelConfig::base();
    model_config.max_atoms = args.max_atoms as u32;
    model_config.max_ring_closures = args.max_closures as u32;
    model_config.fingerprint_slots = if args.use_fingerprint {
        FINGERPRINT_SLOTS as u32
    } else {
        0
    };
    model_config.spectrum_slots = if args.use_spectrum {
        SPECTRUM_SLOTS as u32
    } else {
        0
    };
    model_config.progress_features = args.progress_features;
    let extraction = ExtractionConfig {
        min_patterns: 0,
        max_patterns: 0,
        ..ExtractionConfig::default()
    };
    let train_config = CompletionTrainConfig {
        lr: args.lr.unwrap_or(3e-4),
        weight_decay: 0.01,
        grad_clip: Some(1.0),
        seed: args.seed,
        extraction_seed: args.seed,
        pattern_source: PatternSource::RandomPatches(extraction.clone()),
        extraction,
        fingerprint_mode: args.use_fingerprint.then_some(FingerprintMode::Exact),
        fingerprint_threshold: args.fp_threshold,
    };
    let mut trainer = match &args.load {
        Some(path) => {
            // `--progress-features` on a checkpoint without them: the
            // non-strict load that initialises only the new projection (zero,
            // so step 0 of the resumed run is the checkpoint).
            let trainer = if args.progress_features {
                CompletionTrainer::<R, f32>::load_with_progress_features(path, &device, true)?
            } else {
                CompletionTrainer::<R, f32>::load(path, &device)?
            };
            let config = &trainer.model().config;
            if (config.fingerprint_slots > 0) != args.use_fingerprint
                || (config.spectrum_slots > 0) != args.use_spectrum
            {
                usage("--evidence does not match the loaded checkpoint's encoders");
            }
            eprintln!("loaded {} at step {}", path.display(), trainer.step_count());
            trainer
        }
        None => CompletionTrainer::<R, f32>::new(&model_config, &train_config, &device)?,
    };
    if let Some(lr) = args.lr {
        trainer.set_learning_rate(lr)?;
    }
    eprintln!("effective learning rate: {}", trainer.train_config().lr);
    check_launches(&device)?;
    let parameters: usize = {
        use mamba3::nn::Module;
        trainer.model().num_parameters()
    };
    eprintln!("model: {parameters} parameters");

    // One query: example index of a fold plus the spectrum it is read with.
    let empty: Vec<MolGraph> = Vec::new();
    let build = |fold: &Fold,
                 picks: &[(usize, usize)],
                 source: FpSource,
                 draw: u64|
     -> Result<(
        Vec<SparseFingerprint>,
        Vec<SpectrumEvidence>,
        Vec<&'static str>,
    )> {
        let mut fps = Vec::with_capacity(picks.len());
        let mut spectra = Vec::with_capacity(picks.len());
        let mut sources = Vec::with_capacity(picks.len());
        for &(example, spectrum) in picks {
            let example = &fold.set.examples[example];
            let molecule = &fold.file.molecules[example.source_index];
            let spectrum = &molecule.spectra[spectrum];
            let (fp, origin) = fingerprint_of(
                source,
                fold.bits.get_by_index(example.source_index)?,
                Some(spectrum.spectrum_id),
                &example.key,
                draw,
                &predictions,
                noise.as_ref(),
                channel.as_ref(),
                args.fp_threshold,
                args.seed,
            )?;
            fps.push(fp);
            sources.push(origin);
            spectra.push(evidence_of(spectrum)?);
        }
        Ok((fps, spectra, sources))
    };

    // Teacher-forced validation NLL per token on a fixed subset.
    let eval_picks: Vec<(usize, usize)> = (args.eval_offset
        ..validation
            .set
            .examples
            .len()
            .min(args.eval_offset.saturating_add(args.eval_subset)))
        .map(|i| (i, 0))
        .collect();
    if eval_picks.is_empty() {
        return Err(Error::config(
            "--eval-offset and --eval-subset select no validation molecules".to_string(),
        ));
    }
    let validation_nll = |trainer: &mut CompletionTrainer<R, f32>| -> Result<(f64, f64)> {
        let mut total = 0.0f64;
        let mut tokens = 0usize;
        for chunk in eval_picks.chunks(args.batch) {
            let (mut fps, spectra, _) = build(&validation, chunk, args.fp_eval, EVAL_DRAW)?;
            if args.eval_drop_fingerprint {
                fps.iter_mut().for_each(|fp| fp.entries.clear());
            }
            let owned_patterns = chunk_patterns(&validation, chunk);
            let refs: Vec<&[MolGraph]> = owned_patterns.iter().map(Vec::as_slice).collect();
            let traces: Vec<&[Token]> = chunk
                .iter()
                .map(|&(i, _)| validation.set.examples[i].trace.as_slice())
                .collect();
            let compositions: Vec<Composition> = chunk
                .iter()
                .map(|&(i, _)| validation.set.examples[i].composition)
                .collect();
            let spectra_refs: Vec<Option<&SpectrumEvidence>> = spectra
                .iter()
                .map(|s| (args.use_spectrum && !args.eval_drop_spectrum).then_some(s))
                .collect();
            let nll = trainer.teacher_eval_with_evidence(
                &refs,
                args.use_fingerprint.then_some(fps.as_slice()),
                &spectra_refs,
                &traces,
                &compositions,
            )?;
            total += nll.iter().map(|&v| f64::from(v)).sum::<f64>();
            tokens += traces.iter().map(|t| t.len() - 1).sum::<usize>();
        }
        Ok((
            total / eval_picks.len().max(1) as f64,
            total / tokens.max(1) as f64,
        ))
    };

    let mut curve: Vec<serde_json::Value> = Vec::new();
    let mut train_info = json!(null);
    // Steps to run now: `--until-step` counts from the checkpoint's own step.
    let steps_to_run = match args.until_step {
        Some(target) => target.saturating_sub(trainer.step_count()) as usize,
        None => args.steps,
    };
    if !args.eval_only && steps_to_run > 0 {
        let mut train = load_fold(
            args.train.as_deref().expect("checked in parse_args"),
            args.train_fp.as_deref().expect("checked in parse_args"),
            limits,
        )?;
        if args.scaffold.is_some() {
            let path = args
                .train
                .as_deref()
                .expect("checked in parse_args")
                .with_extension("")
                .with_file_name(format!(
                    "{}_scaffold.json",
                    args.train
                        .as_deref()
                        .and_then(|p| p.file_stem())
                        .map(|s| s.to_string_lossy().into_owned())
                        .unwrap_or_default()
                ));
            train.scaffolds = load_scaffolds(&path, &train.file)?;
            eprintln!("train scaffolds from {}", path.display());
        }
        eprintln!(
            "train: {} molecules in the file, {} kept, skipped {:?}",
            train.file.molecules.len(),
            train.set.examples.len(),
            train.set.skipped
        );
        // No validation identity may be a training identity.
        let (shared_traces, shared_skeletons) = train.set.overlap(&validation.set);
        eprintln!(
            "train/validation overlap: {shared_traces} identical traces, {shared_skeletons} identical skeletons"
        );
        if shared_traces != 0 {
            return Err(Error::config(format!(
                "{shared_traces} validation molecules are training molecules"
            )));
        }
        let n = train.set.examples.len();
        let predicted_molecules = train
            .set
            .examples
            .iter()
            .filter(|e| {
                train.file.molecules[e.source_index]
                    .spectra
                    .iter()
                    .any(|s| predictions.contains_key(&s.spectrum_id))
            })
            .count();
        if args.use_fingerprint && args.fp_train.uses_predictions() {
            if predicted_molecules == 0 {
                return Err(Error::config(
                    "--fp-train predicted: no training molecule has a prediction; provide training predictions, not only validation predictions".to_string(),
                ));
            }
            eprintln!(
                "predicted training coverage: {predicted_molecules}/{n} molecules; {} use {}",
                n - predicted_molecules,
                if args.fp_train.uses_channel() {
                    "channel samples"
                } else {
                    "empty fingerprints"
                }
            );
        }
        if args.fp_train != args.fp_eval {
            eprintln!(
                "warning: training fingerprints {:?} differ from checkpoint-selection fingerprints {:?}",
                args.fp_train, args.fp_eval
            );
        }
        // Structure-only molecules: indices `n..total` of the training pool.
        let extra = match (&args.extra_train, &args.extra_train_fp) {
            (Some(path), Some(fp_path)) => {
                let mut fold = load_fold(path, fp_path, limits)?;
                if args.scaffold.is_some() {
                    let sidecar = path.with_file_name(format!(
                        "{}_scaffold.json",
                        path.file_stem()
                            .map(|s| s.to_string_lossy().into_owned())
                            .unwrap_or_default()
                    ));
                    fold.scaffolds = load_scaffolds(&sidecar, &fold.file)?;
                    eprintln!("extra scaffolds from {}", sidecar.display());
                }
                let known: std::collections::HashSet<&[Token]> = validation
                    .set
                    .examples
                    .iter()
                    .chain(train.set.examples.iter())
                    .map(|e| e.trace.as_slice())
                    .collect();
                let before = fold.set.examples.len();
                fold.set
                    .examples
                    .retain(|e| !known.contains(e.trace.as_slice()));
                eprintln!(
                    "extra train: {} molecules in the file, {} kept, {} dropped as validation or training identities, skipped {:?}",
                    fold.file.molecules.len(),
                    fold.set.examples.len(),
                    before - fold.set.examples.len(),
                    fold.set.skipped
                );
                let (shared, _) = fold.set.overlap(&validation.set);
                if shared != 0 {
                    return Err(Error::config(format!(
                        "{shared} validation molecules are extra training molecules"
                    )));
                }
                Some(fold)
            }
            (None, None) => None,
            _ => usage("--extra-train and --extra-train-fp go together"),
        };
        let extra_n = extra.as_ref().map_or(0, |fold| fold.set.examples.len());
        let total = n + extra_n;
        // The shuffle stream depends on the starting step, so a resumed run
        // does not replay the order it already trained on.
        let mut rng = Mix(args.seed ^ 0x5EED_5EED ^ trainer.step_count().rotate_left(24));
        // Formula artifacts go into every checkpoint this run writes.
        let artifact_compositions: Vec<Composition> =
            train.set.examples.iter().map(|e| e.composition).collect();
        fit_and_attach(
            &mut trainer,
            &artifact_compositions,
            args.max_atoms as u32,
            format!("{} kept training molecules", artifact_compositions.len()),
        )?;
        eprintln!(
            "training {steps_to_run} steps from step {} (save every {})",
            trainer.step_count(),
            args.save_every
        );
        let mut order: Vec<usize> = (0..total).collect();
        let mut cursor = total;
        let mut best = f64::INFINITY;
        let mut train_fp_sources: HashMap<&str, usize> = HashMap::new();
        let mut running = 0.0f64;
        let mut reports = 0usize;
        let train_started = Instant::now();
        for step in 1..=steps_to_run {
            let mut picks = Vec::with_capacity(args.batch);
            let mut extra_picks: Vec<usize> = Vec::new();
            while picks.len() + extra_picks.len() < args.batch {
                if cursor == total {
                    for i in (1..total).rev() {
                        order.swap(i, rng.below(i + 1));
                    }
                    cursor = 0;
                }
                let example = order[cursor];
                cursor += 1;
                if example >= n {
                    extra_picks.push(example - n);
                    continue;
                }
                let molecule_spectra =
                    &train.file.molecules[train.set.examples[example].source_index].spectra;
                // With predicted fingerprints, pick among the spectra that
                // have a prediction (any spectrum when none has one, which
                // then trains on an empty fingerprint and is counted).
                let predicted: Vec<usize> = if args.fp_train.uses_predictions() {
                    (0..molecule_spectra.len())
                        .filter(|&s| predictions.contains_key(&molecule_spectra[s].spectrum_id))
                        .collect()
                } else {
                    Vec::new()
                };
                let pick = if predicted.is_empty() {
                    rng.below(molecule_spectra.len())
                } else {
                    predicted[rng.below(predicted.len())]
                };
                picks.push((example, pick));
            }
            let (mut fps, spectra, sources) =
                build(&train, &picks, args.fp_train, trainer.step_count())?;
            for source in sources {
                *train_fp_sources.entry(source).or_default() += 1;
            }
            // Structure-only molecules follow the spectra molecules in the
            // batch: the same fingerprint policy, no spectral evidence.
            if let Some(extra) = &extra {
                for &i in &extra_picks {
                    let example = &extra.set.examples[i];
                    // These have no spectrum of their own, so `predicted` is
                    // unavailable: use the same empty-input fallback as
                    // spectral queries, never silently switch to true bits.
                    let (fp, origin) = fingerprint_of(
                        args.fp_train,
                        extra.bits.get_by_index(example.source_index)?,
                        None,
                        &example.key,
                        trainer.step_count(),
                        &predictions,
                        noise.as_ref(),
                        channel.as_ref(),
                        args.fp_threshold,
                        args.seed,
                    )?;
                    fps.push(fp);
                    *train_fp_sources.entry(origin).or_default() += 1;
                }
            }
            let batch_len = picks.len() + extra_picks.len();
            let mut keep_spectrum = vec![true; picks.len()];
            if args.evidence_dropout > 0.0 {
                for q in 0..batch_len {
                    if rng.unit() < args.evidence_dropout {
                        fps[q].entries.clear();
                    }
                    if q < picks.len() && rng.unit() < args.evidence_dropout {
                        keep_spectrum[q] = false;
                    }
                }
            }
            let mut owned_patterns = chunk_patterns(&train, &picks);
            if let Some(extra) = &extra {
                for &i in &extra_picks {
                    owned_patterns.push(
                        scaffold_pattern(extra, i)
                            .map(|g| vec![g])
                            .unwrap_or_default(),
                    );
                }
            }
            let refs: Vec<&[MolGraph]> = owned_patterns.iter().map(Vec::as_slice).collect();
            let mut traces: Vec<&[Token]> = picks
                .iter()
                .map(|&(i, _)| train.set.examples[i].trace.as_slice())
                .collect();
            let mut compositions: Vec<Composition> = picks
                .iter()
                .map(|&(i, _)| train.set.examples[i].composition)
                .collect();
            let mut spectra_refs: Vec<Option<&SpectrumEvidence>> = spectra
                .iter()
                .zip(keep_spectrum.iter())
                .map(|(s, &keep)| (args.use_spectrum && keep).then_some(s))
                .collect();
            if let Some(extra) = &extra {
                for &i in &extra_picks {
                    traces.push(extra.set.examples[i].trace.as_slice());
                    compositions.push(extra.set.examples[i].composition);
                    spectra_refs.push(None);
                }
            }
            let report = step % args.report_every == 0;
            if report {
                trainer.request_report();
            }
            let loss = trainer.step_with_evidence(
                &refs,
                args.use_fingerprint.then_some(fps.as_slice()),
                &spectra_refs,
                &traces,
                &compositions,
            )?;
            if let Some(loss) = loss {
                check_launches(&device)?;
                running += f64::from(loss);
                reports += 1;
                let rate = step as f64 / train_started.elapsed().as_secs_f64();
                eprintln!("step {step} loss {loss:.4} ({rate:.2} steps/s)");
            }
            if args.save_every > 0 && step % args.save_every == 0 {
                if let Some(save) = &args.save {
                    trainer.save(save)?;
                    eprintln!("saved step {} to {}", trainer.step_count(), save.display());
                }
            }
            if args.eval_every > 0 && (step % args.eval_every == 0 || step == steps_to_run) {
                let (per_query, per_token) = validation_nll(&mut trainer)?;
                check_launches(&device)?;
                let improved = per_query < best;
                eprintln!(
                    "eval step {step}: validation NLL {per_query:.3} per molecule, {per_token:.4} per token{}",
                    if improved { " (best)" } else { "" }
                );
                curve.push(json!({
                    "step": trainer.step_count(), "validation_nll_per_molecule": per_query,
                    "validation_nll_per_token": per_token,
                    "train_loss_mean_since_last": if reports > 0 { json!(running / reports as f64) } else { json!(null) },
                }));
                running = 0.0;
                reports = 0;
                if improved {
                    best = per_query;
                    if let Some(save) = &args.save {
                        let mut best_path = save.clone().into_os_string();
                        best_path.push(".best");
                        trainer.save(Path::new(&best_path))?;
                    }
                }
            }
        }
        let compositions: Vec<Composition> =
            train.set.examples.iter().map(|e| e.composition).collect();
        fit_and_attach(
            &mut trainer,
            &compositions,
            args.max_atoms as u32,
            format!("{} kept training molecules", compositions.len()),
        )?;
        if let Some(save) = &args.save {
            trainer.save(save)?;
            // The best checkpoint gets the same formula artifacts.
            let mut best_path = save.clone().into_os_string();
            best_path.push(".best");
            let best_path = PathBuf::from(best_path);
            if best_path.exists() {
                let mut best_trainer = CompletionTrainer::<R, f32>::load(&best_path, &device)?;
                fit_and_attach(
                    &mut best_trainer,
                    &compositions,
                    args.max_atoms as u32,
                    format!("{} kept training molecules", compositions.len()),
                )?;
                best_trainer.save(&best_path)?;
            }
        }
        train_info = json!({
            "molecules_in_file": train.file.molecules.len(),
            "molecules_kept": n,
            "extra_structure_only_molecules": extra_n,
            "skipped": train.set.skipped,
            "steps": steps_to_run,
            "final_step": trainer.step_count(),
            "save_every": args.save_every,
            "batch": args.batch,
            "lr": trainer.train_config().lr,
            "requested_lr": args.lr,
            "seed": args.seed,
            "fp_train": format!("{:?}", args.fp_train).to_lowercase(),
            "fp_noise": args.fp_noise.as_ref().and_then(|p| p.file_name()).map(|n| n.to_string_lossy().into_owned()),
            "fp_channel": args.fp_channel.as_ref().and_then(|p| p.file_name()).map(|n| n.to_string_lossy().into_owned()),
            "evidence_dropout": args.evidence_dropout,
            "fingerprint_sources": train_fp_sources,
            "molecules_with_predictions": predicted_molecules,
            "predicted_fingerprint_fallbacks_to_exact": 0,
            "seconds": train_started.elapsed().as_secs_f64(),
            "best_validation_nll_per_molecule": best,
            "validation_overlap": {"traces": shared_traces, "skeletons": shared_skeletons},
        });
    }

    // A checkpoint saved during training has no formula artifacts yet: fit
    // them from the training file when one is given (the weights are
    // untouched).
    if args.formula == FormulaSource::Mass
        && trainer.formula_artifacts().is_none()
        && let (Some(train), Some(train_fp)) = (&args.train, &args.train_fp)
    {
        let train = load_fold(train, train_fp, limits)?;
        let compositions: Vec<Composition> =
            train.set.examples.iter().map(|e| e.composition).collect();
        fit_and_attach(
            &mut trainer,
            &compositions,
            args.max_atoms as u32,
            format!("{} kept training molecules", compositions.len()),
        )?;
        eprintln!(
            "attached formula artifacts from {} training molecules",
            compositions.len()
        );
        if let Some(save) = &args.save {
            trainer.save(save)?;
        }
    }

    // Teacher-forced NLL of the evaluated checkpoint under the evaluation
    // inputs (fingerprint source and any removed input), on the same subset
    // the training curve uses.
    let (eval_nll_molecule, eval_nll_token) = validation_nll(&mut trainer)?;
    check_launches(&device)?;
    eprintln!(
        "teacher-forced NLL on {} molecules: {eval_nll_molecule:.3} per molecule, {eval_nll_token:.4} per token",
        eval_picks.len()
    );

    // Target-survival curve: the model's own probability for each action of
    // the target's trace, under the evaluation inputs.
    if args.survival > 0 {
        let picks: Vec<(usize, usize)> = (0..validation.set.examples.len().min(args.survival))
            .map(|i| (i, 0))
            .collect();
        let mut rows: Vec<serde_json::Value> = Vec::with_capacity(picks.len());
        for chunk in picks.chunks(args.batch) {
            let (mut fps, spectra, _) = build(&validation, chunk, args.fp_eval, EVAL_DRAW)?;
            if args.eval_drop_fingerprint {
                fps.iter_mut().for_each(|fp| fp.entries.clear());
            }
            let owned_patterns = chunk_patterns(&validation, chunk);
            let refs: Vec<&[MolGraph]> = owned_patterns.iter().map(Vec::as_slice).collect();
            let traces: Vec<&[Token]> = chunk
                .iter()
                .map(|&(i, _)| validation.set.examples[i].trace.as_slice())
                .collect();
            let compositions: Vec<Composition> = chunk
                .iter()
                .map(|&(i, _)| validation.set.examples[i].composition)
                .collect();
            let spectra_refs: Vec<Option<&SpectrumEvidence>> = spectra
                .iter()
                .map(|s| (args.use_spectrum && !args.eval_drop_spectrum).then_some(s))
                .collect();
            let (flat, steps) = trainer.teacher_steps_with_evidence(
                &refs,
                args.use_fingerprint.then_some(fps.as_slice()),
                &spectra_refs,
                &traces,
                &compositions,
            )?;
            check_launches(&device)?;
            for (q, &(i, _)) in chunk.iter().enumerate() {
                let example = &validation.set.examples[i];
                // The scored positions are the trace's tokens after START.
                let scored = example.trace.len().saturating_sub(1);
                let mut per_step = Vec::with_capacity(scored);
                for t in 0..scored.min(steps) {
                    let base = (q * steps + t) * 4;
                    per_step.push(flat[base..base + 4].iter().sum::<f32>());
                }
                rows.push(json!({
                    "query": rows.len(),
                    "key": example.key,
                    "heavy_atoms": example.target.atoms().len(),
                    "trace_length": example.trace.len(),
                    "step_log_prob": per_step,
                }));
            }
        }
        let path = args
            .out
            .clone()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(format!("{}_survival.json", args.tag));
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&path, serde_json::to_string(&json!({"queries": rows}))?)?;
        eprintln!("wrote {}", path.display());
    }

    // Evaluation: sampled shortlists for the first `queries` kept molecules.
    let n_queries = validation.set.examples.len().min(args.queries);
    let mut lines: Vec<String> = Vec::with_capacity(n_queries);
    let mut dump_lines: Vec<String> = Vec::new();
    let mut dump_bits: Vec<Vec<u16>> = Vec::new();
    let mut dump_panel: Vec<serde_json::Value> = Vec::new();
    let mut hits = [0usize; 3];
    let mut target_in_pool = 0usize;
    let mut any_candidate = 0usize;
    let mut candidates_total = 0usize;
    let mut candidates_in_tolerance = 0usize;
    let mut finished = 0u64;
    let mut dead_end = 0u64;
    let mut trajectories = 0u64;
    let mut true_formula_sampled = 0usize;
    let mut hypotheses_total = 0usize;
    let mut element_prior_used = 0usize;
    // Decoder row-steps actually executed, so the beam and sampling arms
    // compare by work rather than by `K`.
    let mut row_steps = 0u64;
    let mut beam_dropped = 0u64;
    let mut acceptance_totals = std::collections::BTreeMap::<String, u64>::new();
    let mut beam_totals = std::collections::BTreeMap::<String, u64>::new();
    let mut fp_sources: HashMap<&'static str, usize> = HashMap::new();
    let generation_started = Instant::now();
    let model = trainer.model();
    let constants = trainer.constants();
    let gen_config = CompletionGenerationConfig {
        trajectories: args.trajectories,
        temperature: args.temperature,
        seed: args.gen_seed,
        returned: args.returned,
        substructure_semantics: SubstructureSemantics::Contained,
        ..CompletionGenerationConfig::default()
    };
    let all_picks: Vec<(usize, usize)> = (0..n_queries).map(|i| (i, 0)).collect();
    for chunk in all_picks.chunks(args.gen_batch) {
        let (mut fps, spectra, mut sources) = build(&validation, chunk, args.fp_eval, EVAL_DRAW)?;
        if args.eval_drop_fingerprint {
            fps.iter_mut().for_each(|fp| fp.entries.clear());
            sources.iter_mut().for_each(|s| *s = "dropped");
        }
        let spectrum_seen = args.use_spectrum && !args.eval_drop_spectrum;
        // Per query: candidates as (graph, formula, samples, log-probability)
        // plus the accounting and the formula record.
        struct Shortlist {
            candidates: Vec<(MolGraph, String, u32, f32)>,
            accounting: serde_json::Value,
            beam_stats: Vec<BeamStats>,
            formula: serde_json::Value,
            finished: u32,
            dead_end: u32,
            trajectories: u32,
            true_formula_sampled: bool,
            hypotheses: usize,
        }
        let chunk_scaffolds = chunk_patterns(&validation, chunk);
        let mut shortlists: Vec<Shortlist> = Vec::with_capacity(chunk.len());
        match args.formula {
            FormulaSource::Oracle => {
                let requests: Vec<CompletionRequest> = chunk
                    .iter()
                    .enumerate()
                    .map(|(q, &(i, _))| CompletionRequest {
                        id: validation.set.examples[i].identity_group * 16 + 1,
                        composition: validation.set.examples[i].composition,
                        patterns: chunk_scaffolds[q].as_slice(),
                        acceptance_patterns: None,
                        fingerprint: args.use_fingerprint.then_some(&fps[q]),
                    })
                    .collect();
                let spectra_refs: Vec<Option<&SpectrumEvidence>> =
                    spectra.iter().map(|s| spectrum_seen.then_some(s)).collect();
                let (outcomes, beam_stats) = if args.beam > 0 {
                    let (outcomes, stats) = model.generate_beam_with_spectra(
                        &requests,
                        &spectra_refs,
                        &gen_config,
                        args.beam,
                        constants,
                        &device,
                    )?;
                    for report in stats.iter() {
                        row_steps += report.row_steps;
                        beam_dropped += report.candidates_dropped;
                    }
                    (outcomes, stats)
                } else {
                    row_steps += u64::from(args.trajectories)
                        * u64::from(limits.max_steps() as u32 - 1)
                        * chunk.len() as u64;
                    (
                        model.generate_with_spectra(
                            &requests,
                            &spectra_refs,
                            &gen_config,
                            constants,
                            &device,
                        )?,
                        Vec::new(),
                    )
                };
                for (q, (outcome, &(i, _))) in outcomes.iter().zip(chunk.iter()).enumerate() {
                    let formula = formula_text(&validation.set.examples[i].composition);
                    shortlists.push(Shortlist {
                        candidates: outcome
                            .candidates
                            .iter()
                            .map(|c| (c.graph.clone(), formula.clone(), c.samples, c.best_log_prob))
                            .collect(),
                        accounting: json!({
                            "trajectories": outcome.trajectories, "finished": outcome.finished,
                            "dead_end": outcome.dead_end, "truncated": outcome.truncated,
                            "rejected_replay": outcome.rejected_replay, "distinct": outcome.distinct,
                            "identity_unresolved": outcome.identity_unresolved,
                            "other_status": outcome.other_status,
                            "rejected_containment": outcome.rejected_containment,
                            "containment_unresolved": outcome.containment_unresolved,
                            "pass_contained": outcome.pass_contained,
                            "pass_disjoint": outcome.pass_disjoint,
                            "pass_complete": outcome.pass_complete,
                            "rejected_extra_groups": outcome.rejected_extra_groups,
                            "rejected_missing_groups": outcome.rejected_missing_groups,
                        }),
                        beam_stats: beam_stats.get(q).cloned().into_iter().collect(),
                        formula: json!({"source": "oracle", "formulas": [formula]}),
                        finished: outcome.finished,
                        dead_end: outcome.dead_end,
                        trajectories: outcome.trajectories,
                        true_formula_sampled: true,
                        hypotheses: 1,
                    });
                }
            }
            FormulaSource::Mass => {
                let artifacts = trainer.formula_artifacts().ok_or_else(|| {
                    Error::config(
                        "--formula mass needs a checkpoint with formula artifacts (train with this driver, which attaches them)".to_string(),
                    )
                })?;
                for (q, &(i, s)) in chunk.iter().enumerate() {
                    let example = &validation.set.examples[i];
                    let spectrum = &validation.file.molecules[example.source_index].spectra[s];
                    let query = MassQuery::Neutral {
                        value: spectra[q].neutral_mass,
                        ppm_tenths: args.mass_ppm_tenths,
                        uncertainty: Some(
                            spectrum.precursor_uncertainty_udalton + ADDUCT_CONVERSION_ERROR_UDA,
                        ),
                    };
                    let (rows_total, mode) = if args.beam > 0 {
                        (args.beam, CompletionSearch::Beam)
                    } else {
                        (args.trajectories, CompletionSearch::Sampling)
                    };
                    let prior = element_predictions.get(&spectrum.spectrum_id).map(|counts| {
                        ElementPrior {
                            log1p_counts: *counts,
                            temperature: args.formula_temperature,
                        }
                    });
                    element_prior_used += usize::from(prior.is_some());
                    let result = run_mass_completion_search_with_prior(
                        model,
                        constants,
                        &device,
                        artifacts,
                        args.max_atoms as u32,
                        args.max_closures as u32,
                        chunk_scaffolds[q].as_slice(),
                        None,
                        &query,
                        args.hypotheses,
                        2_000_000,
                        rows_total,
                        args.temperature,
                        args.gen_seed,
                        args.returned,
                        &format!("{}|{}", example.key, spectrum.spectrum_id),
                        true,
                        FormulaPruning::TrainFit,
                        FormulaAllocation::Equal,
                        SubstructureSemantics::Contained,
                        args.use_fingerprint.then_some(&fps[q]),
                        spectrum_seen.then_some(&spectra[q]),
                        mode,
                        prior.as_ref(),
                    )?;
                    if args.beam > 0 {
                        row_steps += result.beam_row_steps;
                        beam_dropped += result.beam_candidates_dropped;
                    } else {
                        // Sampling runs every row for every step by
                        // construction, so its rows are exact without
                        // instrumenting the loop.
                        row_steps += u64::from(result.accounting.trajectories)
                            * u64::from(limits.max_steps() as u32 - 1);
                    }
                    let true_formula = formula_text(&example.composition);
                    let search = &result.formula_search;
                    let sampled_true = search
                        .formulas
                        .iter()
                        .any(|f| f.formula == true_formula && f.trajectories > 0);
                    let mut candidates = Vec::with_capacity(result.candidates.len());
                    for c in &result.candidates {
                        candidates.push((
                            MolGraph::new(c.atoms.clone(), c.bonds.clone())?,
                            c.formula.clone(),
                            c.samples,
                            c.best_log_prob,
                        ));
                    }
                    shortlists.push(Shortlist {
                        candidates,
                        accounting: json!({
                            "trajectories": result.accounting.trajectories,
                            "finished": result.accounting.finished,
                            "dead_end": result.accounting.dead_end,
                            "truncated": result.accounting.truncated,
                            "rejected_replay": result.accounting.rejected_replay,
                            "distinct": result.distinct_before_cut,
                            "identity_unresolved": result.accounting.identity_unresolved,
                            "other_status": result.accounting.other_status,
                            "rejected_containment": result.accounting.rejected_containment,
                            "containment_unresolved": result.accounting.containment_unresolved,
                            "pass_contained": result.accounting.pass_contained,
                            "pass_disjoint": result.accounting.pass_disjoint,
                            "pass_complete": result.accounting.pass_complete,
                            "rejected_extra_groups": result.accounting.rejected_extra_groups,
                            "rejected_missing_groups": result.accounting.rejected_missing_groups,
                        }),
                        beam_stats: search.formulas.iter().filter_map(|f| f.beam_stats.clone()).collect(),
                        formula: json!({
                            "source": "mass",
                            "mass_evidence": result.mass_evidence_status,
                            "search_status": search.status,
                            "joined": search.joined,
                            "selected": search.selected,
                            "unsampled_reason": search.unsampled_reason,
                            "true_formula": true_formula,
                            "true_formula_joined": result.joined_compositions.contains(&example.composition),
                            "true_formula_sampled": sampled_true,
                            "formulas": search.formulas.iter().map(|f| json!({
                                "formula": f.formula, "computed_uda": f.computed_uda,
                                "residual_uda": f.residual_uda, "trajectories": f.trajectories,
                                "accepted_candidates": f.accepted_candidates,
                                "finished": f.finished,
                                "beam_stats": f.beam_stats,
                            })).collect::<Vec<_>>(),
                        }),
                        finished: result.accounting.finished,
                        dead_end: result.accounting.dead_end,
                        trajectories: result.accounting.trajectories,
                        true_formula_sampled: sampled_true,
                        hypotheses: search.formulas.iter().filter(|f| f.trajectories > 0).count(),
                    });
                }
            }
        }
        check_launches(&device)?;
        for (q, (&(i, s), shortlist)) in chunk.iter().zip(shortlists.iter()).enumerate() {
            for (name, value) in shortlist.accounting.as_object().expect("accounting object") {
                if let Some(count) = value.as_u64() {
                    *acceptance_totals.entry(name.clone()).or_default() += count;
                }
            }
            for stats in &shortlist.beam_stats {
                for (name, value) in serde_json::to_value(stats)?
                    .as_object()
                    .expect("beam stats object")
                {
                    if let Some(count) = value.as_u64() {
                        *beam_totals.entry(name.clone()).or_default() += count;
                    }
                }
            }
            let example = &validation.set.examples[i];
            let molecule = &validation.file.molecules[example.source_index];
            let spectrum = &molecule.spectra[s];
            let evidence = &spectra[q];
            let allowed = tolerance(evidence.neutral_mass, args.mass_ppm_tenths)
                + spectrum.precursor_uncertainty_udalton
                + ADDUCT_CONVERSION_ERROR_UDA;
            let mut target_rank: Option<usize> = None;
            let mut rows = Vec::with_capacity(shortlist.candidates.len());
            for (rank, (graph, formula, samples, log_prob)) in
                shortlist.candidates.iter().enumerate()
            {
                let mass = composition_mass(&graph.composition())?;
                let residual = mass.abs_diff(evidence.neutral_mass);
                let is_target = same_identity(graph, &example.target, 100_000) == Some(true);
                if is_target && target_rank.is_none() {
                    target_rank = Some(rank + 1);
                }
                candidates_total += 1;
                if residual <= allowed {
                    candidates_in_tolerance += 1;
                }
                let mut row = graph_json(graph);
                row["rank"] = json!(rank + 1);
                row["formula"] = json!(formula);
                row["mass_uda"] = json!(mass);
                row["mass_residual_uda"] = json!(residual);
                row["mass_residual_ppm"] =
                    json!(residual as f64 / evidence.neutral_mass as f64 * 1e6);
                row["mass_within_tolerance"] = json!(residual <= allowed);
                row["precursor_mz_uda"] = json!(precursor_mz_of(mass, evidence.adduct));
                row["samples"] = json!(samples);
                row["best_log_prob"] = json!(log_prob);
                row["is_target"] = json!(is_target);
                rows.push(row);
            }
            if target_rank.is_some() {
                target_in_pool += 1;
            }
            if let Some(rank) = target_rank {
                for (slot, limit) in [1usize, 10, 25].iter().enumerate() {
                    if rank <= *limit {
                        hits[slot] += 1;
                    }
                }
            }
            if !shortlist.candidates.is_empty() {
                any_candidate += 1;
            }
            finished += u64::from(shortlist.finished);
            dead_end += u64::from(shortlist.dead_end);
            trajectories += u64::from(shortlist.trajectories);
            true_formula_sampled += usize::from(shortlist.true_formula_sampled);
            hypotheses_total += shortlist.hypotheses;
            *fp_sources.entry(sources[q]).or_default() += 1;
            let fp = &fps[q];
            if args.dump_candidates.is_some() {
                let true_bits = validation.bits.get_by_index(example.source_index)?;
                dump_lines.push(
                    json!({
                        "target": graph_json(&example.target),
                        "composition": example.composition,
                        "candidates": shortlist.candidates.iter().map(|(graph, _, samples, log_prob)| {
                            let mut row = graph_json(graph);
                            row["samples"] = json!(samples);
                            row["best_log_prob"] = json!(log_prob);
                            row
                        }).collect::<Vec<_>>(),
                    })
                    .to_string(),
                );
                dump_bits.push(true_bits.to_vec());
                let mut molecule = graph_json(&example.target);
                molecule["fp_true"] = json!(true_bits);
                molecule["fp_pred_mean"] = json!(
                    fp.entries
                        .iter()
                        .map(|&(b, p)| json!([b, p]))
                        .collect::<Vec<_>>()
                );
                dump_panel.push(molecule);
            }
            let (token_ids, _, token_valid) = fp.tokens(FINGERPRINT_SLOTS);
            let tokens_used = token_valid.iter().filter(|&&v| v != 0.0).count();
            let mut target = graph_json(&example.target);
            target["formula"] = json!(formula_text(&example.composition));
            target["mass_uda"] = json!(composition_mass(&example.composition)?);
            let line = json!({
                "query": lines.len(),
                "key": example.key,
                "identity_group": example.identity_group,
                "spectrum_id": spectrum.spectrum_id,
                "inputs": {
                    // Whether this query was handed a Murcko scaffold that
                    // fits the pattern encoder, and how many atoms of the
                    // target it covers: the two populations a scaffold run
                    // has to be read by.
                    "scaffold": {
                        "supplied": !chunk_scaffolds[q].is_empty(),
                        "atoms": chunk_scaffolds[q].iter().map(|g| g.atoms().len()).sum::<usize>(),
                        "target_atoms": example.target.atoms().len(),
                    },
                    "adduct_id": evidence.adduct,
                    "adduct": completion_adduct(evidence.adduct).map(|a| a.name),
                    "precursor_mz_uda": evidence.precursor_mz,
                    "neutral_mass_uda": evidence.neutral_mass,
                    "mass_ppm_tenths": args.mass_ppm_tenths,
                    "mass_tolerance_uda": allowed,
                    "spectrum": {
                        "seen_by_model": spectrum_seen,
                        "peaks_supplied": evidence.peaks.len(),
                        "peaks_used": evidence.selected(SPECTRUM_SLOTS).len(),
                        "peaks": evidence.selected(SPECTRUM_SLOTS).iter()
                            .map(|&(mz, r)| json!([mz, r])).collect::<Vec<_>>(),
                    },
                    "fingerprint": {
                        "seen_by_model": args.use_fingerprint && !args.eval_drop_fingerprint,
                        "source": sources[q],
                        "threshold": args.fp_threshold,
                        "entries": fp.entries.len(),
                        "tokens_used": tokens_used,
                        "token_bits": token_ids.iter().filter(|&&t| t != 0).map(|&t| t - 1).collect::<Vec<_>>(),
                        "bits": fp.entries.iter().map(|&(b, p)| json!([b, p])).collect::<Vec<_>>(),
                    },
                },
                "target": target,
                "formula_search": shortlist.formula,
                "accounting": shortlist.accounting,
                "beam_stats": shortlist.beam_stats,
                "target_rank": target_rank,
                "candidates": rows,
            });
            lines.push(line.to_string());
        }
        eprintln!(
            "generated {}/{} queries ({:.1}s): top-1 {} top-10 {} top-25 {}",
            lines.len(),
            n_queries,
            generation_started.elapsed().as_secs_f64(),
            hits[0],
            hits[1],
            hits[2]
        );
    }
    if let Some(path) = &args.dump_candidates {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut file = std::fs::File::create(path)?;
        for line in &dump_lines {
            writeln!(file, "{line}")?;
        }
        let sibling = |suffix: &str| {
            let mut name = path.clone().into_os_string();
            name.push(suffix);
            PathBuf::from(name)
        };
        std::fs::write(
            sibling(".bits.json"),
            json!({"fingerprint": "morgan4096", "bits_by_molecule": dump_bits}).to_string(),
        )?;
        std::fs::write(
            sibling(".panel.json"),
            json!({"molecules": dump_panel}).to_string(),
        )?;
        eprintln!(
            "wrote {} with its .bits.json and .panel.json",
            path.display()
        );
    }
    let rate = |count: usize| {
        if n_queries > 0 {
            count as f64 / n_queries as f64
        } else {
            0.0
        }
    };
    let report = json!({
        "tool": "ms2_spectral_completion",
        "backend": std::any::type_name::<R>(),
        "model": {
            "config": trainer.model().config,
            "parameters": parameters,
            "checkpoint_steps": trainer.step_count(),
            "effective_learning_rate": trainer.train_config().lr,
            "evidence": {"fingerprint": args.use_fingerprint, "spectrum": args.use_spectrum},
        },
        "data": {
            "validation_file": args.validation.file_name().map(|n| n.to_string_lossy().into_owned()),
            "validation_molecules_in_file": validation.file.molecules.len(),
            "validation_molecules_kept": validation.set.examples.len(),
            "validation_skipped": validation.set.skipped,
            "prediction_files": args.predictions.iter()
                .map(|p| p.file_name().map(|n| n.to_string_lossy().into_owned())).collect::<Vec<_>>(),
            "predicted_spectra_loaded": predictions.len(),
        },
        "train": train_info,
        "curve": curve,
        "evaluation": {
            "teacher_forced_nll": {
                "molecules": eval_picks.len(),
                "offset": args.eval_offset,
                "per_molecule": eval_nll_molecule,
                "per_token": eval_nll_token,
            },
            "queries": n_queries,
            "returned": args.returned,
            "temperature": args.temperature,
            "trajectories_per_query": args.trajectories,
            "search": if args.beam > 0 { "beam" } else { "sampling" },
            "beam_width": args.beam,
            "decoder_row_steps": row_steps,
            "decoder_row_steps_per_query": if n_queries > 0 { row_steps as f64 / n_queries as f64 } else { 0.0 },
            "beam_candidates_dropped": beam_dropped,
            "beam_stats_totals": beam_totals,
            "acceptance_accounting": acceptance_totals,
            "accounting_semantics": if args.beam > 0 {
                "acceptance accounting describes admitted STOP traces; beam_stats_totals measures the search; trajectories_per_query is the requested width budget split across formulas"
            } else { "acceptance accounting describes independent sampled trajectories" },
            "formula_source": format!("{:?}", args.formula).to_lowercase(),
            "mass_ppm_tenths": args.mass_ppm_tenths,
            "hypotheses_max": args.hypotheses,
            "element_prior": {"queries_with_prediction": element_prior_used,
                              "temperature": args.formula_temperature},
            "fp_eval": format!("{:?}", args.fp_eval).to_lowercase(),
            "eval_drop": {"fingerprint": args.eval_drop_fingerprint, "spectrum": args.eval_drop_spectrum},
            "fingerprint_sources": fp_sources,
            "top1": hits[0], "top10": hits[1], "top25": hits[2],
            "top1_rate": rate(hits[0]), "top10_rate": rate(hits[1]), "top25_rate": rate(hits[2]),
            "target_in_pool": target_in_pool,
            "target_in_pool_rate": rate(target_in_pool),
            "queries_with_a_candidate": any_candidate,
            "candidates_returned": candidates_total,
            "candidates_with_mass_in_tolerance": candidates_in_tolerance,
            "trajectories": trajectories, "finished": finished, "dead_end": dead_end,
            "true_formula_sampled": true_formula_sampled,
            "mean_sampled_formulas": if n_queries > 0 { hypotheses_total as f64 / n_queries as f64 } else { 0.0 },
            "seconds": generation_started.elapsed().as_secs_f64(),
        },
        "total_seconds": started.elapsed().as_secs_f64(),
    });
    println!("{}", serde_json::to_string_pretty(&report["evaluation"])?);
    if let Some(out) = &args.out {
        std::fs::create_dir_all(out)?;
        std::fs::write(
            out.join(format!("{}_report.json", args.tag)),
            serde_json::to_string_pretty(&report)?,
        )?;
        let mut file = std::fs::File::create(out.join(format!("{}_predictions.jsonl", args.tag)))?;
        for line in &lines {
            writeln!(file, "{line}")?;
        }
        eprintln!(
            "wrote {}/{}_report.json and {}_predictions.jsonl",
            out.display(),
            args.tag,
            args.tag
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(extra: &[&str]) -> Args {
        parse_args_from(
            ["--validation", "val.json", "--validation-fp", "val_fp.json"]
                .into_iter()
                .chain(extra.iter().copied())
                .map(str::to_string),
        )
    }

    #[test]
    fn predicted_training_defaults_to_predicted_validation() {
        assert_eq!(
            args(&["--fp-train", "predicted"]).fp_eval,
            FpSource::Predicted
        );
        assert_eq!(
            args(&["--fp-train", "mist_like"]).fp_eval,
            FpSource::MistLike
        );
        assert_eq!(args(&[]).fp_eval, FpSource::Exact);
        assert_eq!(
            args(&["--fp-eval", "exact", "--fp-train", "predicted"]).fp_eval,
            FpSource::Exact,
            "an explicit oracle control remains available"
        );
        assert_eq!(
            args(&[]).lr,
            None,
            "resume preserves the saved rate by default"
        );
        assert_eq!(args(&["--lr", "0.0001"]).lr, Some(1e-4));
    }

    #[test]
    fn predicted_conditioning_never_substitutes_target_bits() {
        let predictions = HashMap::from([(0, vec![(3, 0.7)]), (42, vec![(5, 0.11), (9, 0.8)])]);
        let get = |id, truth: &[u16]| {
            fingerprint_of(
                FpSource::Predicted,
                truth,
                id,
                "molecule",
                1,
                &predictions,
                None,
                None,
                0.1,
                7,
            )
            .unwrap()
        };
        let (fp, origin) = get(Some(42), &[1, 2]);
        assert_eq!(origin, "predicted");
        assert_eq!(fp.entries, vec![(5, 0.11), (9, 0.8)]);
        assert_eq!(
            fp,
            get(Some(42), &[10, 11, 12]).0,
            "truth cannot change model input"
        );
        for id in [Some(99), None] {
            let (fp, origin) = get(id, &[1, 2]);
            assert_eq!(origin, "missing");
            assert!(
                fp.entries.is_empty(),
                "missing spectra and structure-only data use empty input"
            );
            assert_eq!(fp, get(id, &[10, 11, 12]).0);
        }
    }

    /// A one-class channel: bit 7 always becomes a bucket-8 token when on,
    /// bit 9 a bucket-1 token when off, every other bit never a token.
    fn tiny_channel() -> FingerprintChannel {
        let none = serde_json::json!([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let mut on = vec![none.clone(); 4096];
        let mut off = vec![none; 4096];
        on[7] = serde_json::json!([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
        off[9] = serde_json::json!([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let doc = serde_json::json!({
            "format": "fingerprint_channel_v1", "fingerprint": "morgan4096",
            "threshold": 0.1, "buckets": 8, "classes": 1, "weights": [1.0],
            "on": [on], "off": [off],
        });
        FingerprintChannel::load_json(&doc.to_string()).unwrap()
    }

    #[test]
    fn channel_sources_parse_and_sample_without_a_prediction() {
        assert_eq!(
            args(&["--fp-train", "predicted_channel"]).fp_train,
            FpSource::PredictedChannel
        );
        assert_eq!(args(&["--fp-train", "channel"]).fp_eval, FpSource::Channel);
        assert!(FpSource::PredictedChannel.uses_predictions());
        assert!(FpSource::PredictedChannel.uses_channel());
        assert!(!FpSource::Channel.uses_predictions());

        let channel = tiny_channel();
        let predictions = HashMap::from([(42, vec![(5, 0.11), (9, 0.8)])]);
        let get = |source, id, truth: &[u16]| {
            fingerprint_of(
                source,
                truth,
                id,
                "molecule",
                1,
                &predictions,
                None,
                Some(&channel),
                0.1,
                7,
            )
            .unwrap()
        };
        // A real prediction wins and ignores the truth.
        let (fp, origin) = get(FpSource::PredictedChannel, Some(42), &[7]);
        assert_eq!(origin, "predicted");
        assert_eq!(fp.entries, vec![(5, 0.11), (9, 0.8)]);
        // No prediction (another spectrum, or a structure-only molecule):
        // the channel degrades the true bits.
        for id in [Some(99), None] {
            let (fp, origin) = get(FpSource::PredictedChannel, id, &[7, 8]);
            assert_eq!(origin, "channel");
            let bits: Vec<u16> = fp.entries.iter().map(|e| e.0).collect();
            assert_eq!(bits, vec![7, 9], "bit 8 is missed, bit 9 is a false token");
            assert_eq!(SparseFingerprint::bucket(fp.entries[0].1), 8);
            assert_eq!(SparseFingerprint::bucket(fp.entries[1].1), 1);
            assert!(fp.entries[1].1 >= 0.1);
        }
        // The pure channel source never reads the prediction.
        let (fp, origin) = get(FpSource::Channel, Some(42), &[]);
        assert_eq!(origin, "channel");
        assert_eq!(fp.entries.len(), 1);
        assert_eq!(fp.entries[0].0, 9);
        // Without a channel file the source is an error, not a fallback.
        assert!(
            fingerprint_of(
                FpSource::Channel,
                &[7],
                None,
                "molecule",
                1,
                &predictions,
                None,
                None,
                0.1,
                7
            )
            .is_err()
        );
    }
}
