//! Motif-level decoder: train and search.
//!
//! A [`Mamba3Lm`] reads a conditioning prefix (formula and fingerprint, see
//! [`Layout::prefix`]) and writes a molecule as a sequence of whole ring
//! systems, groups and single atoms (`mamba3::models::ms2::motif`). The
//! inputs come from `tools/ms2/motif_tokens.py prepare`.
//!
//! ```text
//! cargo run --release --no-default-features --features wgpu --example ms2_motif_decoder -- \
//!   --vocab data/ms2/specgen/motif/lm_vocab.json \
//!   --train data/ms2/specgen/motif/msgym_train.lm.jsonl \
//!   --train data/ms2/specgen/motif/extra163k_structures.lm.jsonl \
//!   --validation data/ms2/specgen/motif/msgym_validation.lm.jsonl \
//!   --steps 40000 --save run/motif.ckpt
//! ```
//!
//! Options (defaults in brackets):
//!
//! * `--arch prefix|encoders` [prefix]: how the decoder is conditioned.
//!   `prefix` is the plain sequence model above. `encoders` is
//!   [`MotifModel`]: the completion model's fingerprint and spectrum
//!   encoders feed a memory every decoder layer attends to, as in the
//!   atom-level decoder; `--spectra FILE` (repeatable) names the exports
//!   whose fragment peaks, adduct and neutral mass it reads by spectrum id,
//!   `--no-spectrum` builds it without the spectrum encoder, and
//!   `--evidence-dropout` [0.1] is the share of training rows that lose
//!   their spectrum.
//! * `--vocab FILE`, `--train FILE` (repeatable), `--validation FILE`.
//! * `--predictions FILE` (repeatable): a fingerprint predictor's JSONL by
//!   spectrum id; `--fp-channel FILE`: its error channel
//!   (`tools/ms2/fit_fingerprint_channel.py`).
//! * Training fingerprints, drawn per row: `--p-empty` [0.1] none,
//!   `--p-exact` [0.25] the true bits, otherwise a predicted-looking one: the
//!   real prediction of one of the molecule's spectra with probability
//!   `--p-real` [0.5] when it has one, else a channel sample (the true bits
//!   when no channel was given).
//! * `--steps`, `--batch` [32], `--lr` [3e-4], `--warmup` [1000], `--seed`
//!   [1], `--d-model` [192], `--layers` [6], `--max-target` [128] (longer
//!   molecules are left out of training and counted).
//! * `--eval-every` [1000], `--eval-subset` [300], `--eval-offset` [300]:
//!   teacher-forced NLL per molecule on validation molecules, with
//!   `--fp-eval exact|predicted|removed|channel` [exact].
//! * `--load FILE`, `--save FILE`, `--save-every` [1000].
//! * Search: `--queries N` [0] validation molecules from `--query-offset`
//!   [0], `--beam W` [256], `--returned` [1024], `--hypotheses FILE` (a JSON
//!   map from molecule key to formula candidates, each ten counts in the
//!   order C H N O F P S Cl Br I; the beam is split evenly across them and
//!   the pooled sequences are ordered by log-probability, which treats the
//!   formulas as equally likely; without the file the query's own formula is
//!   used, an oracle input),
//!   `--out FILE`: one JSON line per query with the best `--returned`
//!   finished sequences of the beam and their log-probability under the
//!   mask.
//!
//! `--load` restores weights only: the optimizer and the schedule start
//! afresh. A query whose answer the converter gave no sequence is still
//! searched; scoring is done by `tools/ms2/motif_score.py`, which rebuilds
//! the molecules and compares them under the competition's identity.

#![cfg(feature = "backend")]

use std::cell::Cell;
use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::time::Instant;

use mamba3::models::ms2::completion_fingerprint::{
    FINGERPRINT_SLOTS, FingerprintBatch, FingerprintChannel, SparseFingerprint,
};
use mamba3::models::ms2::completion_spectrum::{
    SPECTRUM_SLOTS, SpectrumBatch, SpectrumEvidence, neutral_mass_of,
};
use mamba3::models::ms2::dataset::ExportFile;
use mamba3::models::ms2::motif_model::{MotifModel, MotifModelConfig};
use mamba3::models::ms2::motif::{
    BeamOutput, Formula, Layout, MotifMachine, MotifVocab, PAD, PREFIX_LEN, beam_search,
    gather_lm_cache,
};
use mamba3::nn::{Module, Param, StateDict};
use mamba3::prelude::*;
use mamba3::tensor::ops::index::IdTensor;
use mamba3::train::Checkpoint;
use mamba3::train::loss::{CrossEntropyConfig, cross_entropy_with};
use mamba3::train::trainer::QueuedStep;
use mamba3::train::trainer::TrainStep;
use serde::Deserialize;
use serde_json::json;

type R = mamba3::backends::Auto;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum FpEval {
    Exact,
    Predicted,
    Removed,
    Channel,
}

struct Args {
    encoders: bool,
    spectra: Vec<PathBuf>,
    no_spectrum: bool,
    evidence_dropout: f64,
    vocab: PathBuf,
    train: Vec<PathBuf>,
    validation: PathBuf,
    predictions: Vec<PathBuf>,
    fp_channel: Option<PathBuf>,
    p_empty: f64,
    p_exact: f64,
    p_real: f64,
    steps: u64,
    batch: usize,
    lr: f32,
    warmup: u64,
    seed: u64,
    d_model: usize,
    layers: usize,
    d_state: usize,
    max_target: usize,
    eval_every: u64,
    eval_subset: usize,
    eval_offset: usize,
    fp_eval: FpEval,
    load: Option<PathBuf>,
    save: Option<PathBuf>,
    save_every: u64,
    queries: usize,
    query_offset: usize,
    beam: usize,
    returned: usize,
    hypotheses: Option<PathBuf>,
    out: Option<PathBuf>,
}

fn usage(message: &str) -> ! {
    eprintln!("ms2_motif_decoder: {message}\nsee the header of examples/ms2_motif_decoder.rs");
    std::process::exit(2)
}

fn parse_args() -> Args {
    let mut a = Args {
        encoders: false,
        spectra: Vec::new(),
        no_spectrum: false,
        evidence_dropout: 0.1,
        vocab: PathBuf::new(),
        train: Vec::new(),
        validation: PathBuf::new(),
        predictions: Vec::new(),
        fp_channel: None,
        p_empty: 0.1,
        p_exact: 0.25,
        p_real: 0.5,
        steps: 0,
        batch: 32,
        lr: 3e-4,
        warmup: 1000,
        seed: 1,
        d_model: 192,
        layers: 6,
        d_state: 32,
        max_target: 128,
        eval_every: 1000,
        eval_subset: 300,
        eval_offset: 300,
        fp_eval: FpEval::Exact,
        load: None,
        save: None,
        save_every: 1000,
        queries: 0,
        query_offset: 0,
        beam: 256,
        returned: 1024,
        hypotheses: None,
        out: None,
    };
    let mut it = std::env::args().skip(1);
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
            "--arch" => {
                a.encoders = match value().as_str() {
                    "prefix" => false,
                    "encoders" => true,
                    other => usage(&format!("unknown --arch {other:?}")),
                }
            }
            "--spectra" => a.spectra.push(value().into()),
            "--no-spectrum" => a.no_spectrum = true,
            "--evidence-dropout" => a.evidence_dropout = num(&flag, value()),
            "--vocab" => a.vocab = value().into(),
            "--train" => a.train.push(value().into()),
            "--validation" => a.validation = value().into(),
            "--predictions" => a.predictions.push(value().into()),
            "--fp-channel" => a.fp_channel = Some(value().into()),
            "--p-empty" => a.p_empty = num(&flag, value()),
            "--p-exact" => a.p_exact = num(&flag, value()),
            "--p-real" => a.p_real = num(&flag, value()),
            "--steps" => a.steps = num(&flag, value()),
            "--batch" => a.batch = num(&flag, value()),
            "--lr" => a.lr = num(&flag, value()),
            "--warmup" => a.warmup = num(&flag, value()),
            "--seed" => a.seed = num(&flag, value()),
            "--d-model" => a.d_model = num(&flag, value()),
            "--layers" => a.layers = num(&flag, value()),
            "--d-state" => a.d_state = num(&flag, value()),
            "--max-target" => a.max_target = num(&flag, value()),
            "--eval-every" => a.eval_every = num(&flag, value()),
            "--eval-subset" => a.eval_subset = num(&flag, value()),
            "--eval-offset" => a.eval_offset = num(&flag, value()),
            "--fp-eval" => {
                a.fp_eval = match value().as_str() {
                    "exact" => FpEval::Exact,
                    "predicted" => FpEval::Predicted,
                    "removed" => FpEval::Removed,
                    "channel" => FpEval::Channel,
                    other => usage(&format!("unknown --fp-eval {other:?}")),
                }
            }
            "--load" => a.load = Some(value().into()),
            "--save" => a.save = Some(value().into()),
            "--save-every" => a.save_every = num(&flag, value()),
            "--queries" => a.queries = num(&flag, value()),
            "--query-offset" => a.query_offset = num(&flag, value()),
            "--beam" => a.beam = num(&flag, value()),
            "--returned" => a.returned = num(&flag, value()),
            "--hypotheses" => a.hypotheses = Some(value().into()),
            "--out" => a.out = Some(value().into()),
            other => usage(&format!("unknown argument {other:?}")),
        }
    }
    if a.vocab.as_os_str().is_empty() || a.validation.as_os_str().is_empty() {
        usage("--vocab and --validation are required");
    }
    if a.steps > 0 && a.train.is_empty() {
        usage("training needs --train");
    }
    if !(0.0..1.0).contains(&a.evidence_dropout) {
        usage("--evidence-dropout must be in [0, 1)");
    }
    if a.batch == 0 || a.beam == 0 || a.max_target == 0 {
        usage("--batch, --beam and --max-target must be positive");
    }
    if !(0.0..=1.0).contains(&a.p_empty)
        || !(0.0..=1.0).contains(&a.p_exact)
        || a.p_empty + a.p_exact > 1.0
        || !(0.0..=1.0).contains(&a.p_real)
    {
        usage("--p-empty, --p-exact and --p-real are probabilities, and the first two sum to at most 1");
    }
    a
}

/// SplitMix64: the shuffle and the per-row source draws.
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

/// One molecule of a `*.lm.jsonl` file.
#[derive(Deserialize, Clone)]
struct Row {
    index: usize,
    key: String,
    tokens: Option<Vec<u32>>,
    formula: Option<Formula>,
    bits: Vec<u16>,
    spectra: Vec<u64>,
}

fn load_rows(path: &PathBuf) -> Result<Vec<Row>> {
    let reader = std::io::BufReader::new(std::fs::File::open(path)?);
    let mut rows = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if !line.trim().is_empty() {
            rows.push(serde_json::from_str(&line)?);
        }
    }
    Ok(rows)
}

fn load_predictions(paths: &[PathBuf]) -> Result<HashMap<u64, Vec<(u16, f32)>>> {
    let mut out = HashMap::new();
    for path in paths {
        let reader = std::io::BufReader::new(std::fs::File::open(path)?);
        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let value: serde_json::Value = serde_json::from_str(&line)?;
            let digits: String = value["id"]
                .as_str()
                .unwrap_or("")
                .chars()
                .filter(char::is_ascii_digit)
                .collect();
            let Ok(id) = digits.parse::<u64>() else {
                return Err(Error::config(format!("{}: a row has no numeric id", path.display())));
            };
            let pairs = value["bits"]
                .as_array()
                .map(|bits| {
                    bits.iter()
                        .filter_map(|pair| {
                            let bit = pair[0].as_u64().filter(|&bit| bit < 4096)?;
                            Some((bit as u16, pair[1].as_f64()? as f32))
                        })
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            out.insert(id, pairs);
        }
    }
    Ok(out)
}

/// Spectral evidence of every spectrum of the given exports, by spectrum id.
fn load_spectra(paths: &[PathBuf]) -> Result<HashMap<u64, SpectrumEvidence>> {
    let mut out = HashMap::new();
    for path in paths {
        for molecule in &ExportFile::load(path)?.molecules {
            for spectrum in &molecule.spectra {
                let Some(neutral) = neutral_mass_of(spectrum.precursor_mz_udalton, spectrum.adduct)
                else {
                    continue;
                };
                out.insert(
                    spectrum.spectrum_id,
                    SpectrumEvidence {
                        peaks: spectrum
                            .mz_udalton
                            .iter()
                            .zip(spectrum.intensity.iter())
                            .map(|(&mz, &intensity)| (mz, intensity as f32))
                            .collect(),
                        precursor_mz: spectrum.precursor_mz_udalton,
                        adduct: spectrum.adduct,
                        neutral_mass: neutral,
                    },
                );
            }
        }
    }
    Ok(out)
}

/// The decoder: a plain sequence model reading a prefix, or the model with
/// conditioning encoders.
enum Net {
    Prefix(Mamba3Lm<R, f32>),
    Encoders(MotifModel<R, f32>),
}

impl Net {
    fn parameters(&self) -> Vec<Param<R, f32>> {
        match self {
            Net::Prefix(model) => model.parameters(),
            Net::Encoders(model) => model.parameters(),
        }
    }

    fn num_parameters(&self) -> usize {
        match self {
            Net::Prefix(model) => model.num_parameters(),
            Net::Encoders(model) => model.num_parameters(),
        }
    }

    fn set_training(&self, training: bool) {
        match self {
            Net::Prefix(model) => model.set_training(training),
            Net::Encoders(model) => model.set_training(training),
        }
    }

    fn restore(&self, checkpoint: &Checkpoint) -> Result<()> {
        match self {
            Net::Prefix(model) => checkpoint.restore(model, true),
            Net::Encoders(model) => checkpoint.restore(model, true),
        }
    }

    fn capture(&self, step: u64) -> Checkpoint {
        match self {
            Net::Prefix(model) => Checkpoint::capture(model, step),
            Net::Encoders(model) => Checkpoint::capture(model, step),
        }
    }

    fn uses_spectrum(&self) -> bool {
        matches!(self, Net::Encoders(model) if model.config().spectrum_slots > 0)
    }
}

/// One query of a batch: the molecule and the evidence it is read with.
struct Item<'a> {
    row: &'a Row,
    fingerprint: SparseFingerprint,
    spectrum: Option<&'a SpectrumEvidence>,
}

/// A training or validation batch in the form its decoder reads.
enum NetBatch {
    /// Inputs `[batch, PREFIX_LEN + T - 1]` and targets `[batch, T]`.
    Prefix {
        inputs: IdTensor<R>,
        targets: IdTensor<R>,
        width: usize,
    },
    /// Inputs `[batch, T]` (the start token, then the target shifted by
    /// one), targets `[batch, T]` and the conditioning inputs.
    Encoders {
        inputs: IdTensor<R>,
        targets: IdTensor<R>,
        formulas: Vec<Formula>,
        fingerprints: FingerprintBatch,
        spectra: Option<SpectrumBatch>,
    },
}

/// A prefix padded on the left to [`PREFIX_LEN`]. Padding ids pass through
/// the recurrent stack like any other, so the amount of padding is part of
/// what the decoder is conditioned on: training, validation and search all
/// use this one function, which makes a query's conditioning the same in all
/// three and independent of what else is in its batch.
fn padded_prefix(layout: &Layout, formula: &Formula, fingerprint: &SparseFingerprint) -> Vec<u32> {
    let prefix = layout.prefix(formula, fingerprint);
    let mut out = vec![PAD; PREFIX_LEN - prefix.len()];
    out.extend(prefix);
    out
}

/// Build the batch of `items` for `net`; also returns its target tokens.
fn build_batch(
    net: &Net,
    layout: &Layout,
    items: &[Item],
    device: &Device<R>,
) -> Result<(NetBatch, usize)> {
    let width = items
        .iter()
        .map(|item| item.row.tokens.as_ref().map_or(0, Vec::len))
        .max()
        .unwrap_or(1)
        .max(1);
    let mut targets = Vec::with_capacity(items.len() * width);
    let mut tokens = 0;
    for item in items {
        let target = item.row.tokens.as_ref().expect("batch rows carry tokens");
        targets.extend_from_slice(target);
        targets.extend(std::iter::repeat_n(PAD, width - target.len()));
        tokens += target.len();
    }
    let targets = IdTensor::from_slice(&targets, vec![items.len(), width], device)?;
    let formula_of = |item: &Item| *item.row.formula.as_ref().expect("batch rows carry a formula");
    let batch = match net {
        Net::Prefix(_) => {
            let seq = PREFIX_LEN + width - 1;
            let mut inputs = Vec::with_capacity(items.len() * seq);
            for item in items {
                let target = item.row.tokens.as_ref().expect("checked above");
                inputs.extend(padded_prefix(layout, &formula_of(item), &item.fingerprint));
                inputs.extend_from_slice(&target[..target.len() - 1]);
                inputs.extend(std::iter::repeat_n(PAD, width - target.len()));
            }
            NetBatch::Prefix {
                inputs: IdTensor::from_host(inputs, vec![items.len(), seq], device)?,
                targets,
                width,
            }
        }
        Net::Encoders(model) => {
            let mut inputs = Vec::with_capacity(items.len() * width);
            for item in items {
                let target = item.row.tokens.as_ref().expect("checked above");
                inputs.push(model.start_token());
                inputs.extend_from_slice(&target[..target.len() - 1]);
                inputs.extend(std::iter::repeat_n(PAD, width - target.len()));
            }
            let fingerprints: Vec<SparseFingerprint> =
                items.iter().map(|item| item.fingerprint.clone()).collect();
            let spectra: Vec<Option<&SpectrumEvidence>> =
                items.iter().map(|item| item.spectrum).collect();
            NetBatch::Encoders {
                inputs: IdTensor::from_host(inputs, vec![items.len(), width], device)?,
                targets,
                formulas: items.iter().map(formula_of).collect(),
                fingerprints: FingerprintBatch::build(&fingerprints, FINGERPRINT_SLOTS)?,
                spectra: if net.uses_spectrum() {
                    Some(SpectrumBatch::build(&spectra, SPECTRUM_SLOTS)?)
                } else {
                    None
                },
            }
        }
    };
    Ok((batch, tokens))
}

/// Next-token loss over the target tokens only.
struct MotifTask<'a> {
    net: &'a Net,
    params: Vec<Param<R, f32>>,
    training: Cell<bool>,
    device: &'a Device<R>,
}

impl MotifTask<'_> {
    fn batch_loss(&self, batch: &NetBatch, train: bool) -> Result<Var<R, f32>> {
        let config = CrossEntropyConfig::default().with_ignore_index(PAD);
        match (self.net, batch) {
            (
                Net::Prefix(model),
                NetBatch::Prefix {
                    inputs,
                    targets,
                    width,
                },
            ) => {
                let hidden = model.hidden(inputs, train)?;
                let window = hidden.slice(1, PREFIX_LEN - 1, *width)?;
                cross_entropy_with(&model.logits_from(&window)?, targets, config)
            }
            (
                Net::Encoders(model),
                NetBatch::Encoders {
                    inputs,
                    targets,
                    formulas,
                    fingerprints,
                    spectra,
                },
            ) => {
                let conditioning =
                    model.encode(formulas, fingerprints, spectra.as_ref(), self.device)?;
                cross_entropy_with(&model.logits(inputs, &conditioning)?, targets, config)
            }
            _ => Err(Error::config("a batch built for the other decoder".to_string())),
        }
    }
}

impl TrainStep<R, f32> for MotifTask<'_> {
    type Batch = NetBatch;

    fn parameters(&self) -> Vec<Param<R, f32>> {
        self.params.clone()
    }

    fn loss(&self, batch: &Self::Batch) -> Result<Var<R, f32>> {
        if self.training.get() {
            self.batch_loss(batch, true)
        } else {
            let _guard = mamba3::autograd::no_grad();
            self.batch_loss(batch, false)
        }
    }

    fn set_training(&self, training: bool) {
        self.training.set(training);
        self.net.set_training(training);
    }
}

/// Pad a pre-padding checkpoint's head to `width` output ids.
///
/// Old checkpoints store `head.weight` as `[d, old]` and `head.bias` as
/// `[old]`; the model now builds the head `width` wide. Padded rows/entries
/// (zeros for the weight, `-1e4` for the bias) keep the file loadable. If a
/// key is missing, it is left alone.
fn pad_head(state: &mut StateDict, width: usize) {
    let mut padded: Option<(usize, usize)> = None;
    if let Some(weight) = state.entries.get_mut("head.weight") {
        if weight.shape.len() == 2 {
            let (d, old) = (weight.shape[0], weight.shape[1]);
            if old < width && weight.data.len() == d * old {
                let mut data = Vec::with_capacity(d * width);
                for row in 0..d {
                    data.extend_from_slice(&weight.data[row * old..(row + 1) * old]);
                    data.extend(std::iter::repeat_n(0.0f32, width - old));
                }
                weight.data = data;
                weight.shape = vec![d, width];
                padded = Some((old, width));
            }
        }
    }
    if let Some(bias) = state.entries.get_mut("head.bias") {
        if bias.shape.len() == 1 {
            let old = bias.shape[0];
            if old < width && bias.data.len() == old {
                bias.data.extend(std::iter::repeat_n(-1.0e4_f32, width - old));
                bias.shape = vec![width];
                if padded.is_none() {
                    padded = Some((old, width));
                }
            }
        }
    }
    if let Some((old, new)) = padded {
        eprintln!("padded head {old} -> {new}");
    }
}

/// Beam search for one query and one formula ([`beam_search`]).
///
/// The prefix decoder prefills the padded prefix and then takes one cached
/// step per token; the encoder decoder encodes the query once and steps
/// against that memory. Either way the cache rows are reordered to each kept
/// hypothesis's parent before a step.
#[allow(clippy::too_many_arguments)]
fn search(
    net: &Net,
    vocab: &MotifVocab,
    layout: &Layout,
    formula: &Formula,
    fingerprint: &SparseFingerprint,
    spectrum: Option<&SpectrumEvidence>,
    width: usize,
    max_tokens: usize,
    device: &Device<R>,
) -> Result<BeamOutput> {
    let n_out = layout.n_out as usize;
    match net {
        Net::Prefix(model) => {
            let prefix = padded_prefix(layout, formula, fingerprint);
            let mut cache = model.empty_cache(1, device);
            let ids = IdTensor::from_slice(&prefix, vec![1, prefix.len()], device)?;
            // The prefill's logits cover every prefix position; the last one
            // (`SEP`) predicts the root motif, where the training loss starts.
            let prefill = model.forward_cached(&ids, &mut cache)?;
            let first = prefill
                .slice(1, prefix.len() - 1, 1)?
                .slice(2, 0, n_out)?
                .to_f32();
            let mut rows = 1usize;
            beam_search(vocab, formula, n_out, first, width, max_tokens, |parents, tokens| {
                let parent_ids = IdTensor::from_slice(parents, vec![parents.len()], device)?;
                cache = gather_lm_cache(&cache, &parent_ids, rows)?;
                rows = parents.len();
                let step = IdTensor::from_slice(tokens, vec![tokens.len(), 1], device)?;
                Ok(model
                    .forward_cached(&step, &mut cache)?
                    .slice(2, 0, n_out)?
                    .to_f32())
            })
        }
        Net::Encoders(model) => {
            let n_out = model.config().n_out_padded();
            let fingerprints =
                FingerprintBatch::build(std::slice::from_ref(fingerprint), FINGERPRINT_SLOTS)?;
            let spectra = if net.uses_spectrum() {
                Some(SpectrumBatch::build(&[spectrum], SPECTRUM_SLOTS)?)
            } else {
                None
            };
            let conditioning = model.encode(
                std::slice::from_ref(formula),
                &fingerprints,
                spectra.as_ref(),
                device,
            )?;
            let mut state = model.start(&conditioning, device)?;
            let start = IdTensor::from_slice(&[model.start_token()], vec![1], device)?;
            let first = model.step(&mut state, &start, device)?.to_f32();
            // Position `i` predicts token `i`, so a model of `P` positions
            // can write `P` tokens, which takes `P - 1` advances.
            let limit = max_tokens.min(model.config().max_tokens);
            // Pad rows to a multiple of 32 so the device sees few matmul shapes;
            // extra rows copy the last row and their logits are truncated away.
            beam_search(vocab, formula, n_out, first, width, limit, |parents, tokens| {
                let rows = parents.len();
                let padded = rows.div_ceil(32) * 32;
                let mut parents_padded = parents.to_vec();
                parents_padded.resize(padded, *parents.last().expect("beam_search never advances with no rows"));
                let mut tokens_padded = tokens.to_vec();
                tokens_padded.resize(padded, *tokens.last().expect("beam_search never advances with no rows"));
                model.gather_host(&mut state, &parents_padded, device)?;
                let step = IdTensor::from_slice(&tokens_padded, vec![padded], device)?;
                let mut logits = model.step(&mut state, &step, device)?.to_f32();
                logits.truncate(rows * n_out);
                Ok(logits)
            })
        }
    }
}

fn main() -> Result<()> {
    let args = parse_args();
    let device = Device::<R>::default();
    let started = Instant::now();
    let vocab = MotifVocab::load(&args.vocab)?;
    let layout = Layout::new(vocab.len());
    eprintln!(
        "backend {}; {} motifs, {} output ids, {} ids in all",
        device.name(),
        vocab.len(),
        layout.n_out,
        layout.vocab_size
    );
    let predictions = load_predictions(&args.predictions)?;
    if !args.predictions.is_empty() {
        eprintln!("predictions: {} spectra", predictions.len());
    }
    let channel = match &args.fp_channel {
        Some(path) => Some(FingerprintChannel::load(path)?),
        None => None,
    };
    if args.fp_eval == FpEval::Channel && channel.is_none() {
        usage("--fp-eval channel needs --fp-channel");
    }
    if args.fp_eval == FpEval::Predicted && predictions.is_empty() {
        usage("--fp-eval predicted needs --predictions");
    }

    let net = if args.encoders {
        Net::Encoders(MotifModel::init(
            &MotifModelConfig {
                n_out: layout.n_out as usize,
                d_model: args.d_model,
                layers: args.layers,
                attention_heads: 4,
                d_state: args.d_state,
                max_tokens: args.max_target,
                fingerprint_slots: FINGERPRINT_SLOTS,
                spectrum_slots: if args.no_spectrum { 0 } else { SPECTRUM_SLOTS },
                seed: args.seed,
            },
            &device,
        )?)
    } else {
        Net::Prefix(
            Mamba3LmConfig::builder()
                .vocab_size(layout.vocab_size as usize)
                .d_model(args.d_model)
                .n_layers(args.layers)
                .with_ssm(|s| {
                    s.n_heads = (2 * args.d_model / 64).max(1);
                    s.n_groups = s.n_heads;
                    s.head_dim = 64;
                    s.d_state = args.d_state;
                    s.chunk_size = 64;
                })
                .seed(args.seed)
                .build()?
                .init::<R, f32>(&device)?,
        )
    };
    let mut start_step = 0u64;
    if let Some(path) = &args.load {
        let mut checkpoint = Checkpoint::load(path)?;
        if let Net::Encoders(model) = &net {
            pad_head(&mut checkpoint.state, model.config().n_out_padded());
        }
        net.restore(&checkpoint)?;
        start_step = checkpoint.step;
        eprintln!("loaded {} at step {start_step}", path.display());
    }
    eprintln!(
        "model: {} parameters ({})",
        net.num_parameters(),
        if args.encoders { "encoders" } else { "prefix" }
    );
    let spectra = if net.uses_spectrum() {
        let spectra = load_spectra(&args.spectra)?;
        eprintln!("spectral evidence: {} spectra", spectra.len());
        spectra
    } else {
        HashMap::new()
    };
    let eval_spectrum = |row: &Row| row.spectra.first().and_then(|id| spectra.get(id));

    let validation = load_rows(&args.validation)?;
    let eval_fingerprint = |row: &Row| -> Result<SparseFingerprint> {
        Ok(match args.fp_eval {
            FpEval::Exact => SparseFingerprint::from_bits(&row.bits)?,
            FpEval::Removed => SparseFingerprint { entries: Vec::new() },
            FpEval::Predicted => match row.spectra.first().and_then(|id| predictions.get(id)) {
                Some(pairs) => SparseFingerprint::from_probabilities(pairs, 0.1)?,
                None => SparseFingerprint { entries: Vec::new() },
            },
            FpEval::Channel => channel
                .as_ref()
                .expect("checked above")
                .sample(&row.bits, args.seed, &row.key, 1)?,
        })
    };
    // Validation molecules the alphabet can write and the window can hold.
    let usable = |row: &&Row| {
        row.formula.is_some()
            && row
                .tokens
                .as_ref()
                .is_some_and(|t| t.len() <= args.max_target)
    };
    let eval_rows: Vec<&Row> = validation
        .iter()
        .skip(args.eval_offset)
        .take(args.eval_subset)
        .filter(usable)
        .collect();
    let task = MotifTask {
        params: net.parameters(),
        net: &net,
        training: Cell::new(true),
        device: &device,
    };
    let validation_nll = |task: &MotifTask| -> Result<(f64, f64)> {
        task.set_training(false);
        let mut total = 0.0f64;
        let mut tokens = 0usize;
        for chunk in eval_rows.chunks(args.batch) {
            let items: Vec<Item> = chunk
                .iter()
                .map(|row| {
                    Ok(Item {
                        row,
                        fingerprint: eval_fingerprint(row)?,
                        spectrum: eval_spectrum(row),
                    })
                })
                .collect::<Result<_>>()?;
            let (batch, count) = build_batch(&net, &layout, &items, &device)?;
            total += f64::from(task.loss(&batch)?.to_f32()[0]) * count as f64;
            tokens += count;
        }
        task.set_training(true);
        Ok((
            total / eval_rows.len().max(1) as f64,
            total / tokens.max(1) as f64,
        ))
    };

    let mut curve = Vec::new();
    if args.steps > 0 {
        let mut train: Vec<Row> = Vec::new();
        let (mut unwritable, mut too_long, mut rejected) = (0usize, 0usize, 0usize);
        for path in &args.train {
            for row in load_rows(path)? {
                let (Some(tokens), Some(formula)) = (&row.tokens, &row.formula) else {
                    unwritable += 1;
                    continue;
                };
                if tokens.len() > args.max_target {
                    too_long += 1;
                    continue;
                }
                // Every training sequence must be one the mask accepts for
                // its own formula, or the search could never produce it.
                if MotifMachine::run(&vocab, Some(formula), tokens).is_err() {
                    rejected += 1;
                    continue;
                }
                train.push(row);
            }
        }
        eprintln!(
            "train: {} molecules; left out: {unwritable} outside the alphabet, {too_long} over {} tokens, {rejected} rejected by the machine",
            train.len(),
            args.max_target
        );
        if train.is_empty() {
            return Err(Error::config("no training molecule is usable".to_string()));
        }
        if rejected > 0 {
            return Err(Error::config(format!(
                "{rejected} training sequences are not accepted by the machine for their own formula"
            )));
        }
        let config = TrainerConfig::builder()
            .learning_rate(args.lr)
            .schedule(LrSchedule::CosineWithWarmup {
                warmup_steps: args.warmup.min(args.steps.saturating_sub(1)).max(1),
                total_steps: args.steps,
                min_ratio: 0.1,
            })
            .max_grad_norm(1.0)
            .build()?;
        let mut trainer = Trainer::new(
            config,
            AdamWConfig::builder()
                .learning_rate(args.lr)
                .weight_decay(0.01)
                .build()
                .init::<R, f32>(),
        );
        let mut rng = Mix(args.seed ^ 0x6D6F_7469_66 ^ start_step.rotate_left(20));
        let mut order: Vec<usize> = (0..train.len()).collect();
        let mut cursor = train.len();
        let mut sources: HashMap<&str, usize> = HashMap::new();
        let mut running = 0.0f64;
        let mut reports = 0usize;
        let mut best = f64::INFINITY;
        let clock = Instant::now();
        let mut queued: Vec<QueuedStep<R, f32>> = Vec::new();
        for step in 1..=args.steps {
            let mut items: Vec<Item> = Vec::with_capacity(args.batch);
            while items.len() < args.batch {
                if cursor == train.len() {
                    for i in (1..order.len()).rev() {
                        order.swap(i, rng.below(i + 1));
                    }
                    // Group similar-length rows so build_batch pads to a
                    // smaller batch maximum instead of the longest of 32
                    // random rows.
                    let pool = args.batch * 16;
                    let mut start = 0;
                    while start < order.len() {
                        let end = (start + pool).min(order.len());
                        order[start..end].sort_by_key(|&i| {
                            train[i].tokens.as_ref().map_or(0, Vec::len)
                        });
                        let batches = (end - start) / args.batch;
                        for b in (1..batches).rev() {
                            let other = rng.below(b + 1);
                            for k in 0..args.batch {
                                order.swap(start + b * args.batch + k, start + other * args.batch + k);
                            }
                        }
                        start = end;
                    }
                    cursor = 0;
                }
                let row = &train[order[cursor]];
                cursor += 1;
                let draw = rng.unit();
                // The spectrum the row is read with: one of the molecule's
                // own, and the one whose prediction is used when a real
                // prediction is drawn.
                let mut spectrum_id = (!row.spectra.is_empty())
                    .then(|| row.spectra[rng.below(row.spectra.len())]);
                let (fingerprint, source) = if draw < args.p_empty {
                    (SparseFingerprint { entries: Vec::new() }, "empty")
                } else if draw < args.p_empty + args.p_exact {
                    (SparseFingerprint::from_bits(&row.bits)?, "exact")
                } else {
                    let real: Vec<u64> = row
                        .spectra
                        .iter()
                        .copied()
                        .filter(|id| predictions.contains_key(id))
                        .collect();
                    if !real.is_empty() && rng.unit() < args.p_real {
                        let id = real[rng.below(real.len())];
                        spectrum_id = Some(id);
                        let pairs = &predictions[&id];
                        (SparseFingerprint::from_probabilities(pairs, 0.1)?, "predicted")
                    } else if let Some(channel) = &channel {
                        (
                            channel.sample(&row.bits, args.seed, &row.key, start_step + step)?,
                            "channel",
                        )
                    } else {
                        (SparseFingerprint::from_bits(&row.bits)?, "exact")
                    }
                };
                *sources.entry(source).or_default() += 1;
                let spectrum = spectrum_id
                    .and_then(|id| spectra.get(&id))
                    .filter(|_| rng.unit() >= args.evidence_dropout);
                if net.uses_spectrum() {
                    *sources
                        .entry(if spectrum.is_some() { "with spectrum" } else { "without spectrum" })
                        .or_default() += 1;
                }
                items.push(Item {
                    row,
                    fingerprint,
                    spectrum,
                });
            }
            let (batch, _) = build_batch(&net, &layout, &items, &device)?;
            queued.push(trainer.queue_step(&task, std::slice::from_ref(&batch))?);
            let last = step == args.steps;
            // The loss is read only when it is reported: every read waits for the
            // whole queue, and a read per step left the device idle while the host
            // built and enqueued the next step.
            if step % 100 == 0 || last {
                let mut learning_rate = 0.0f32;
                for info in trainer.read_steps(&queued)? {
                    running += f64::from(info.loss);
                    reports += 1;
                    learning_rate = info.learning_rate;
                }
                queued.clear();
                eprintln!(
                    "step {step} loss/token {:.4} lr {:.2e} ({:.2} steps/s)",
                    running / reports as f64,
                    learning_rate,
                    step as f64 / clock.elapsed().as_secs_f64()
                );
                running = 0.0;
                reports = 0;
            }
            if args.eval_every > 0 && (step % args.eval_every == 0 || last) {
                let (per_molecule, per_token) = validation_nll(&task)?;
                let improved = per_molecule < best;
                eprintln!(
                    "eval step {step}: validation NLL {per_molecule:.3} per molecule, {per_token:.4} per token{}",
                    if improved { " (best)" } else { "" }
                );
                curve.push(json!({"step": start_step + step, "nll_per_molecule": per_molecule, "nll_per_token": per_token}));
                if improved {
                    best = per_molecule;
                    if let Some(save) = &args.save {
                        let mut path = save.clone().into_os_string();
                        path.push(".best");
                        net.capture(start_step + step).save(PathBuf::from(path))?;
                    }
                }
            }
            if let Some(save) = &args.save
                && ((args.save_every > 0 && step % args.save_every == 0) || last)
            {
                net.capture(start_step + step)
                    .with_metadata(json!({"tool": "ms2_motif_decoder", "arch": if args.encoders { "encoders" } else { "prefix" }, "d_model": args.d_model, "layers": args.layers, "d_state": args.d_state, "motifs": vocab.len()}))
                    .save(save)?;
            }
        }
        eprintln!("training fingerprint sources: {sources:?}");
    }

    let (per_molecule, per_token) = validation_nll(&task)?;
    eprintln!(
        "teacher-forced NLL on {} molecules ({:?} fingerprints): {per_molecule:.3} per molecule, {per_token:.4} per token",
        eval_rows.len(),
        args.fp_eval
    );

    let mut summary = json!({
        "tool": "ms2_motif_decoder",
        "arch": if args.encoders { "encoders" } else { "prefix" },
        "spectrum_encoder": net.uses_spectrum(),
        "fp_eval": format!("{:?}", args.fp_eval).to_lowercase(),
        "eval_molecules": eval_rows.len(),
        "eval_offset": args.eval_offset,
        "nll_per_molecule": per_molecule,
        "nll_per_token": per_token,
        "curve": curve,
        "seconds": started.elapsed().as_secs_f64(),
    });

    if args.queries > 0 {
        let out_path = args
            .out
            .clone()
            .unwrap_or_else(|| usage("--queries needs --out"));
        let hypotheses: HashMap<String, Vec<Formula>> = match &args.hypotheses {
            Some(path) => serde_json::from_str(&std::fs::read_to_string(path)?)?,
            None => HashMap::new(),
        };
        task.set_training(false);
        let _guard = mamba3::autograd::no_grad();
        let mut stream = std::io::BufWriter::new(std::fs::File::create(&out_path)?);
        let clock = Instant::now();
        let (mut with_candidates, mut total_rows, mut candidates_total) = (0usize, 0usize, 0usize);
        for (q, row) in validation
            .iter()
            .skip(args.query_offset)
            .take(args.queries)
            .enumerate()
        {
            let formulas: Vec<Formula> = match hypotheses.get(&row.key) {
                Some(list) if !list.is_empty() => list.clone(),
                _ if args.hypotheses.is_some() => Vec::new(),
                _ => row.formula.iter().copied().collect(),
            };
            let fingerprint = eval_fingerprint(row)?;
            let mut pool: Vec<(Vec<u32>, f64, usize)> = Vec::new();
            let width = (args.beam / formulas.len().max(1)).max(1);
            for (f, formula) in formulas.iter().enumerate() {
                let heavy: usize = formula
                    .iter()
                    .enumerate()
                    .filter(|(e, _)| *e != 1)
                    .map(|(_, &c)| usize::from(c))
                    .sum();
                let found = search(
                    &net,
                    &vocab,
                    &layout,
                    formula,
                    &fingerprint,
                    eval_spectrum(row),
                    width,
                    5 * heavy.max(1),
                    &device,
                )?;
                total_rows += found.row_steps;
                pool.extend(
                    found
                        .finished
                        .into_iter()
                        .map(|(tokens, score)| (tokens, score, f)),
                );
            }
            pool.sort_by(|x, y| y.1.total_cmp(&x.1));
            pool.truncate(args.returned);
            with_candidates += usize::from(!pool.is_empty());
            candidates_total += pool.len();
            let line = json!({
                "query": args.query_offset + q,
                "index": row.index,
                "key": row.key,
                "spectrum_id": row.spectra.first(),
                "target_tokens": row.tokens,
                "target_formula": row.formula,
                "formulas": formulas,
                "fingerprint_tokens": fingerprint.entries.len(),
                "candidates": pool.iter().map(|(tokens, score, f)| json!({"tokens": tokens, "log_prob": score, "formula": f})).collect::<Vec<_>>(),
            });
            writeln!(stream, "{line}")?;
            if (q + 1) % 20 == 0 {
                eprintln!(
                    "searched {}/{} queries ({:.1}s)",
                    q + 1,
                    args.queries.min(validation.len().saturating_sub(args.query_offset)),
                    clock.elapsed().as_secs_f64()
                );
            }
        }
        stream.flush()?;
        summary["search"] = json!({
            "queries": args.queries,
            "query_offset": args.query_offset,
            "beam": args.beam,
            "returned": args.returned,
            "formula_source": if args.hypotheses.is_some() { "hypotheses file" } else { "the query's own formula (oracle)" },
            "queries_with_a_candidate": with_candidates,
            "candidates": candidates_total,
            "decoder_row_steps": total_rows,
            "seconds": clock.elapsed().as_secs_f64(),
        });
        let mut report = out_path.clone().into_os_string();
        report.push(".report.json");
        std::fs::write(PathBuf::from(report), serde_json::to_string_pretty(&summary)?)?;
    }
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}
