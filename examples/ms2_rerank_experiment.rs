//! Reranker and calibration experiment (task RK; plan items P6.3, P7.9).
//!
//! Trains the [`Reranker`](mamba3::models::ms2::rerank::Reranker) of
//! `docs/MS2_V1_ARCHITECTURE.md` §4.3 on candidates a frozen generator
//! proposes for the `rank` split, fits Platt scaling on the `calibration`
//! split (once for the reranker logit, once for the raw score), and judges
//! both rankings plus calibration on the `report` split.
//!
//! Usage:
//! ```text
//! cargo run --release --no-default-features --features cpu \
//!   --example ms2_rerank_experiment -- \
//!   --generator <trainer checkpoint> --table <formula table json> \
//!   --rank <export> --calibration <export> --report <export> \
//!   [--k 8] [--batch 16] [--epochs 40] [--examples-per-step 1024] \
//!   [--lr 0.01] [--seed 1] [--bootstrap 1000] [--contain-work N] \
//!   [--limit-spectra N] [--save-reranker <path>] \
//!   [--save-calibration <path>] --out <json>
//! ```
//!
//! Labels are containment in the true parent (contracts §7.3), a pseudo-label
//! metric. Features are the 8 of architecture §4.3 through the host twin
//! [`compute_features`](mamba3::models::ms2::rerank::compute_features): the
//! device record rows are rebuilt from the read-back [`CandidateBatch`]
//! (token words, open valences, length), `scores` from
//! `(trace_log_prob, formula_log_prob)`, `evidence` from
//! `(evidence_status, evidence_count)`, and `evidence_f` from the retained
//! evidence records (largest log-probability; smallest |residual| in units of
//! the request's fragment tolerance, recovered per record from the export
//! peak m/z with the default tolerance when the export carries none, exactly
//! as the device twin computes it; `(0, 1)` when there is no evidence).

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::time::Instant;

use mamba3::backend::{Device, reset_transfer_counters, runtime_read_count};
use mamba3::backends::Auto;
use mamba3::error::Error;
use mamba3::models::ms2::calibration::{
    Binning, CalibrationArtifact, ConfigKeys, brier_by_stratum, brier_score, ece_by_stratum,
    expected_calibration_error, fit_platt, reliability_by_stratum, reliability_table,
};
use mamba3::models::ms2::chem::CHEMISTRY_VERSION;
use mamba3::models::ms2::contain::Containment;
use mamba3::models::ms2::contract::{
    AllocationMode, CandidateBatch, Control, EVIDENCE_CAP, FormulaSource, GenerationConfig,
    IdentityMode,
};
use mamba3::models::ms2::dataset::percentile;
use mamba3::models::ms2::experiment::ExperimentSet;
use mamba3::models::ms2::formula::FormulaTable;
use mamba3::models::ms2::metrics::evaluate_candidates;
use mamba3::models::ms2::rerank::{
    EVIDENCE_STRIDE, N_FEATURES, RERANKER_VERSION, RerankTrainer, Reranker, compute_features,
    eligible_examples,
};
use mamba3::models::ms2::rerank_eval::{
    FitProvenance, SearchPolicyInputs, build_search_policy, check_generator_fit, find_shared_key,
    paired_bootstrap, roc_auc, top1_precision, trace_atom_count, validate_bootstrap,
};
use mamba3::models::ms2::targets::RecipeLimits;
use mamba3::models::ms2::train::{Ms2Trainer, TRAIN_EVAL_WORK_LIMIT};
use mamba3::tensor::Tensor;
use mamba3::tensor::ops::random::Rng;

type R = Auto;
type E = f32;

/// Retained formulas per spectrum for generation (the `F` of `ConfigKeys`).
const FORMULAS: u32 = 4;

fn usage() -> ! {
    eprintln!(
        "usage: ms2_rerank_experiment --generator <ckpt> --generator-fit <export> --table <table.json> \
         --rank <export> --calibration <export> --report <export> \
         [--k 8] [--batch 16] [--epochs 40] [--examples-per-step 1024] \
         [--lr 0.01] [--seed 1] [--bootstrap 1000] [--contain-work N] \
         [--limit-spectra N] [--save-reranker <path>] \
         [--save-calibration <path>] --out <report.json>"
    );
    std::process::exit(2);
}

fn fail(msg: String) -> ! {
    eprintln!("ms2_rerank_experiment: {msg}");
    std::process::exit(1);
}

/// Refusal of task step (h): the rank split carries one class only.
fn refuse(msg: String) -> ! {
    eprintln!("ms2_rerank_experiment: {msg}");
    std::process::exit(2);
}

/// p50/p95 of a sample by linear interpolation (empty samples give zeros).
fn p50p95(mut values: Vec<f64>) -> (f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0);
    }
    values.sort_by(|a, b| a.total_cmp(b));
    (percentile(&values, 50.0), percentile(&values, 95.0))
}

/// SHA-256 of `bytes` as lowercase hex (FIPS 180-4; a copy of the private
/// helper in `models::ms2::experiment`, which exposes no public hash).
fn sha256_hex(bytes: &[u8]) -> String {
    let mut h: [u32; 8] = [
        0x6a09_e667,
        0xbb67_ae85,
        0x3c6e_f372,
        0xa54f_f53a,
        0x510e_527f,
        0x9b05_688c,
        0x1f83_d9ab,
        0x5be0_cd19,
    ];
    const K: [u32; 64] = [
        0x428a_2f98, 0x7137_4491, 0xb5c0_fbcf, 0xe9b5_dba5, 0x3956_c25b, 0x59f1_11f1, 0x923f_82a4,
        0xab1c_5ed5, 0xd807_aa98, 0x1283_5b01, 0x2431_85be, 0x550c_7dc3, 0x72be_5d74, 0x80de_b1fe,
        0x9bdc_06a7, 0xc19b_f174, 0xe49b_69c1, 0xefbe_4786, 0x0fc1_9dc6, 0x240c_a1cc, 0x2de9_2c6f,
        0x4a74_84aa, 0x5cb0_a9dc, 0x76f9_88da, 0x983e_5152, 0xa831_c66d, 0xb003_27c8, 0xbf59_7fc7,
        0xc6e0_0bf3, 0xd5a7_9147, 0x06ca_6351, 0x1429_2967, 0x27b7_0a85, 0x2e1b_2138, 0x4d2c_6dfc,
        0x5338_0d13, 0x650a_7354, 0x766a_0abb, 0x81c2_c92e, 0x9272_2c85, 0xa2bf_e8a1, 0xa81a_664b,
        0xc24b_8b70, 0xc76c_51a3, 0xd192_e819, 0xd699_0624, 0xf40e_3585, 0x106a_a070, 0x19a4_c116,
        0x1e37_6c08, 0x2748_774c, 0x34b0_bcb5, 0x391c_0cb3, 0x4ed8_aa4a, 0x5b9c_ca4f, 0x682e_6ff3,
        0x748f_82ee, 0x78a5_636f, 0x84c8_7814, 0x8cc7_0208, 0x90be_fffa, 0xa450_6ceb, 0xbef9_a3f7,
        0xc671_78f2,
    ];
    let mut msg = bytes.to_vec();
    let bit_len = (bytes.len() as u64).wrapping_mul(8);
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());
    for block in msg.chunks_exact(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] =
                u32::from_be_bytes([block[4 * i], block[4 * i + 1], block[4 * i + 2], block[4 * i + 3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
        }
        let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh) =
            (h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = hh.wrapping_add(s1).wrapping_add(ch).wrapping_add(K[i]).wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }
    let mut out = String::with_capacity(64);
    for v in h {
        out.push_str(&format!("{v:08x}"));
    }
    out
}

/// Deterministic 64-bit generator for the epoch shuffles (SplitMix64, the
/// same constants as the experiment loader, so a seed shuffles the same on
/// any platform).
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// One training example: an eligible candidate with its §4.3 features.
struct Example {
    /// Position of the spectrum in the split's used spectrum list.
    group: usize,
    /// Molecule index in the split's [`ExperimentSet`].
    molecule: usize,
    /// Replayed atom count (the size stratum).
    atoms: usize,
    /// Raw score: `formula_log_prob + trace_log_prob`.
    raw: f64,
    /// The 8 features of architecture §4.3.
    features: [f32; 8],
    /// Containment label (`1.0` contained, `0.0` not).
    label: f64,
}

/// All examples of one split plus its exclusion counts.
struct SplitExamples {
    examples: Vec<Example>,
    /// Molecules of the used spectra, in used-spectrum order.
    molecule_of_group: Vec<usize>,
    not_finished: usize,
    invalid: usize,
    duplicate: usize,
    work_limit: usize,
}

/// Stable sigmoid.
fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Mean over molecules of per-molecule means with a percentile bootstrap
/// over molecules (95% interval, seeded SplitMix64, linear-interpolation
/// percentiles — the same rule as `rerank_eval::paired_bootstrap`).
fn molecule_interval(
    per_group: &[f64],
    molecule_of_group: &[usize],
    n: usize,
    seed: u64,
) -> (f64, f64, f64) {
    assert_eq!(per_group.len(), molecule_of_group.len());
    let mut acc: BTreeMap<usize, (f64, usize)> = BTreeMap::new();
    for (i, &m) in molecule_of_group.iter().enumerate() {
        let v = per_group[i];
        if !v.is_nan() {
            let e = acc.entry(m).or_insert((0.0, 0));
            e.0 += v;
            e.1 += 1;
        }
    }
    let means: Vec<f64> = acc.values().map(|&(s, c)| s / c as f64).collect();
    if means.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let point = means.iter().sum::<f64>() / means.len() as f64;
    if n == 0 {
        return (point, point, point);
    }
    let mut rng = SplitMix64::new(seed);
    let m = means.len();
    let mut boots: Vec<f64> = Vec::with_capacity(n);
    for _ in 0..n {
        let mut sum = 0.0;
        for _ in 0..m {
            sum += means[(rng.next() % m as u64) as usize];
        }
        boots.push(sum / m as f64);
    }
    boots.sort_by(|a, b| a.total_cmp(b));
    let rank = |p: f64| {
        if boots.len() == 1 {
            boots[0]
        } else {
            let r = (boots.len() - 1) as f64 * p / 100.0;
            let lo = r.floor() as usize;
            let hi = r.ceil() as usize;
            if lo == hi {
                boots[lo]
            } else {
                boots[lo] + (boots[hi] - boots[lo]) * (r - lo as f64)
            }
        }
    };
    (point, rank(2.5), rank(97.5))
}

/// Per-group top-1 hits (1.0/0.0) in group order.
fn group_top1(scores: &[f64], labels: &[f64], egroup: &[usize], n_groups: usize) -> Vec<f64> {
    let mut out = vec![0.0; n_groups];
    for g in 0..n_groups {
        let mut best: Option<usize> = None;
        for (i, &eg) in egroup.iter().enumerate() {
            if eg != g {
                continue;
            }
            match best {
                None => best = Some(i),
                Some(b) => {
                    if scores[i] > scores[b] {
                        best = Some(i);
                    }
                }
            }
        }
        if let Some(b) = best
            && labels[b] != 0.0
        {
            out[g] = 1.0;
        }
    }
    out
}

/// Per-group precision at `r` in group order.
fn group_precision_at(
    scores: &[f64],
    labels: &[f64],
    egroup: &[usize],
    n_groups: usize,
    r: usize,
) -> Vec<f64> {
    let mut out = vec![0.0; n_groups];
    if r == 0 {
        return out;
    }
    for g in 0..n_groups {
        let mut pos: Vec<usize> = egroup
            .iter()
            .enumerate()
            .filter_map(|(i, &eg)| (eg == g).then_some(i))
            .collect();
        if pos.is_empty() {
            continue;
        }
        pos.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]).then_with(|| a.cmp(&b)));
        let take = r.min(pos.len());
        let mut hits = 0.0;
        for &i in &pos[..take] {
            if labels[i] != 0.0 {
                hits += 1.0;
            }
        }
        out[g] = hits / take as f64;
    }
    out
}

/// Rebuild the [`compute_features`] input buffers for every record of `batch`.
///
/// Device record rows come from the read-back batch: token words and length
/// from `actions`/`length`, open valences from `open_valence` (status words
/// are unread by the feature lane and stay `0`); `scores` from
/// `(trace_log_prob, formula_log_prob)`; `evidence` from
/// `(evidence_status, evidence_count)`; `evidence_f` from the retained
/// evidence records — the largest `evidence_log_prob` and the smallest
/// |residual| in units of the request's fragment tolerance. The tolerance
/// repeats the device twin's `u32` computation at the record's peak m/z
/// (looked up in `set` by original `peak_id`) with the default fragment
/// tolerance the export adapter uses; `(0, 1)` when there is no evidence.
fn features_of_batch(
    batch: &CandidateBatch,
    set: &ExperimentSet,
    chunk: &[usize],
) -> Result<Vec<f32>, Error> {
    let rows = batch.batch * batch.trajectories;
    let steps = batch.max_steps;
    let atoms = batch.max_atoms;
    let stride = steps * 4 + atoms + 4;
    let mut actions = vec![0u32; rows * stride];
    let mut scores = vec![0.0f32; rows * 2];
    let mut evidence = vec![0u32; rows * EVIDENCE_STRIDE];
    let mut evidence_f = vec![0.0f32; rows * 2];
    // Default fragment tolerance in tenths of a ppm: the export adapter
    // stores 0 (the default 100), as `SpectrumBatch::fragment_tolerance` maps.
    const PPM_TENTHS_DEFAULT: u32 = 100;
    for r in 0..rows {
        let tbase = r * steps * 4;
        for s in 0..steps {
            for c in 0..4 {
                actions[r * stride + s * 4 + c] = batch.actions[tbase + s * 4 + c];
            }
        }
        for j in 0..atoms {
            actions[r * stride + steps * 4 + j] = u32::from(batch.open_valence[r * atoms + j]);
        }
        actions[r * stride + steps * 4 + atoms] = batch.length[r];
        scores[r * 2] = batch.trace_log_prob[r];
        scores[r * 2 + 1] = batch.formula_log_prob[r];
        evidence[r * EVIDENCE_STRIDE] = u32::from(batch.evidence_status[r]);
        evidence[r * EVIDENCE_STRIDE + 1] = u32::from(batch.evidence_count[r]);
        let count = usize::from(batch.evidence_count[r]);
        if count == 0 {
            evidence_f[r * 2] = 0.0;
            evidence_f[r * 2 + 1] = 1.0;
            continue;
        }
        // Retained records: largest log-probability, smallest |residual| in
        // tolerance units at the record's peak.
        let b = r / batch.trajectories;
        let export = &set.spectra[chunk[b]].spectrum;
        let mut max_lp = f32::NEG_INFINITY;
        let mut min_unit = f32::INFINITY;
        for e in 0..count.min(EVIDENCE_CAP) {
            let lp = batch.evidence_log_prob[r * EVIDENCE_CAP + e];
            if lp > max_lp {
                max_lp = lp;
            }
            let rabs = batch.evidence_residual[r * EVIDENCE_CAP + e].unsigned_abs();
            let pid = batch.evidence_peak_id[r * EVIDENCE_CAP + e];
            let mut mz = 0u32;
            for (pos, &id) in export.peak_id.iter().enumerate() {
                if id == pid {
                    mz = export.mz_udalton[pos];
                    break;
                }
            }
            let ppm = PPM_TENTHS_DEFAULT;
            let hi = mz / 10_000;
            let lo = mz % 10_000;
            let qq = hi.wrapping_mul(ppm);
            let tol = qq / 1000 + ((qq % 1000) * 10_000 + lo.wrapping_mul(ppm)) / 10_000_000;
            let tol_f = if tol == 0 { 1.0f32 } else { tol as f32 };
            let unit = rabs as f32 / tol_f;
            if unit < min_unit {
                min_unit = unit;
            }
        }
        evidence_f[r * 2] = max_lp;
        evidence_f[r * 2 + 1] = min_unit;
    }
    let (features, _) =
        compute_features(&actions, steps, atoms as u32, &scores, &evidence, &evidence_f)?;
    Ok(features)
}

#[allow(clippy::too_many_lines)]
fn main() {
    let started = Instant::now();
    let argv: Vec<String> = std::env::args().collect();
    let mut generator: Option<PathBuf> = None;
    let mut generator_fit: Option<PathBuf> = None;
    let mut table_path: Option<PathBuf> = None;
    let mut rank_path: Option<PathBuf> = None;
    let mut cal_path: Option<PathBuf> = None;
    let mut rep_path: Option<PathBuf> = None;
    let mut k = 8u32;
    let mut batch = 16usize;
    let mut epochs = 40usize;
    let mut examples_per_step = 1024usize;
    let mut lr = 0.01f32;
    let mut seed = 1u64;
    let mut bootstrap = 1000usize;
    let mut contain_work = TRAIN_EVAL_WORK_LIMIT;
    let mut limit_spectra: Option<usize> = None;
    let mut save_reranker: Option<PathBuf> = None;
    let mut save_calibration: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut next = || args.next().unwrap_or_else(|| usage());
        match arg.as_str() {
            "--generator" => generator = Some(PathBuf::from(next())),
            "--generator-fit" => generator_fit = Some(PathBuf::from(next())),
            "--table" => table_path = Some(PathBuf::from(next())),
            "--rank" => rank_path = Some(PathBuf::from(next())),
            "--calibration" => cal_path = Some(PathBuf::from(next())),
            "--report" => rep_path = Some(PathBuf::from(next())),
            "--k" => k = next().parse().unwrap_or_else(|_| usage()),
            "--batch" => batch = next().parse().unwrap_or_else(|_| usage()),
            "--epochs" => epochs = next().parse().unwrap_or_else(|_| usage()),
            "--examples-per-step" => {
                examples_per_step = next().parse().unwrap_or_else(|_| usage());
            }
            "--lr" => lr = next().parse().unwrap_or_else(|_| usage()),
            "--seed" => seed = next().parse().unwrap_or_else(|_| usage()),
            "--bootstrap" => bootstrap = next().parse().unwrap_or_else(|_| usage()),
            "--contain-work" => contain_work = next().parse().unwrap_or_else(|_| usage()),
            "--limit-spectra" => limit_spectra = Some(next().parse().unwrap_or_else(|_| usage())),
            "--save-reranker" => save_reranker = Some(PathBuf::from(next())),
            "--save-calibration" => save_calibration = Some(PathBuf::from(next())),
            "--out" => out = Some(PathBuf::from(next())),
            _ => usage(),
        }
    }
    let (Some(generator), Some(generator_fit), Some(table_path), Some(rank_path), Some(cal_path), Some(rep_path), Some(out)) =
        (generator, generator_fit, table_path, rank_path, cal_path, rep_path, out)
    else {
        usage()
    };
    if !(1..=64).contains(&k) {
        fail("--k must be in 1..=64".to_string());
    }
    if batch == 0 {
        fail("--batch must be non-zero".to_string());
    }
    if epochs == 0 {
        fail("--epochs must be non-zero".to_string());
    }
    if examples_per_step == 0 {
        fail("--examples-per-step must be non-zero".to_string());
    }
    if !(lr.is_finite() && lr > 0.0) {
        fail("--lr must be finite and positive".to_string());
    }
    if let Err(msg) = validate_bootstrap(bootstrap) {
        refuse(msg);
    }

    // (a) Load the frozen generator exactly as `ms2_experiment --load` does.
    let device = Device::<R>::default();
    let table_text = std::fs::read_to_string(&table_path)
        .unwrap_or_else(|e| fail(format!("cannot read {}: {e}", table_path.display())));
    let table = FormulaTable::from_json(&table_text)
        .unwrap_or_else(|e| fail(format!("cannot parse {}: {e}", table_path.display())));
    let ckpt_bytes = std::fs::read(&generator)
        .unwrap_or_else(|e| fail(format!("cannot read {}: {e}", generator.display())));
    let ckpt_sha256 = sha256_hex(&ckpt_bytes);
    let trainer = Ms2Trainer::<R, E>::load(&generator, &table, &device)
        .unwrap_or_else(|e| fail(format!("cannot load {}: {e}", generator.display())));
    let train_config = trainer.train_config().clone();
    if train_config.control != Control::None {
        fail(format!(
            "checkpoint control {:?} is not None: the reranker experiment needs an uncontrolled generator",
            train_config.control
        ));
    }
    let evidence = trainer.model.config.assignment.is_some();
    let gen_config = GenerationConfig {
        trajectories: k,
        formulas: FORMULAS,
        seed,
        formula_source: train_config.formula_source,
        formula_window: train_config.formula_window,
        enum_lanes_max: train_config.enum_lanes_max,
        enum_lane_visits_max: train_config.enum_lane_visits_max,
        enum_dispatch_visits_max: train_config.enum_dispatch_visits_max,
        allocation: AllocationMode::RoundRobin,
        identity: IdentityMode::Graph,
        identity_work_max: 4096,
        evidence,
        ..GenerationConfig::default()
    };
    let source_name = match gen_config.formula_source {
        FormulaSource::Table => "table",
        FormulaSource::Enumerate => "enumerate",
    };
    let allocation_name = match gen_config.allocation {
        AllocationMode::RoundRobin => "round_robin",
        AllocationMode::Proportional => "proportional",
    };
    let identity_name = match gen_config.identity {
        IdentityMode::Graph => "graph",
        IdentityMode::TraceOnly => "trace_only",
    };
    let table_sha_full = trainer.table_sha256().to_string();
    let table_sha16: String = table_sha_full.chars().take(16).collect();
    let artifacts = trainer.model.config.formula_artifacts.clone();
    let enum_domain_16: Option<String> = artifacts
        .as_ref()
        .map(|a| a.domain_sha256.chars().take(16).collect());
    let enum_bounds_16: Option<String> = artifacts
        .as_ref()
        .map(|a| a.bounds_sha256.chars().take(16).collect());
    let search_policy = build_search_policy(&SearchPolicyInputs {
        formula_source: source_name,
        window_m: gen_config.formula_window,
        formula_rows_scored_max: gen_config.formula_rows_scored_max,
        formula_rows_visited_max: gen_config.formula_rows_visited_max,
        enum_lane_visits_max: gen_config.enum_lane_visits_max,
        enum_lanes_max: gen_config.enum_lanes_max,
        allocation: allocation_name,
        identity: identity_name,
        returned: gen_config.effective_returned(),
        table_sha256_16: &table_sha16,
        enum_domain_sha256_16: enum_domain_16.as_deref(),
        enum_bounds_sha256_16: enum_bounds_16.as_deref(),
    });
    let keys_reranker = ConfigKeys {
        domain: CHEMISTRY_VERSION.to_string(),
        k: k.to_string(),
        f: FORMULAS.to_string(),
        precision: "f32".to_string(),
        ranking: RERANKER_VERSION.to_string(),
        search_policy: search_policy.clone(),
    };
    let keys_raw = ConfigKeys {
        ranking: "raw".to_string(),
        ..keys_reranker.clone()
    };

    // Load the three splits plus the export the generator (and every fitted
    // artifact it uses) was trained on.
    let load_set = |path: &PathBuf, role: &str| -> ExperimentSet {
        ExperimentSet::load(path, &RecipeLimits::V0)
            .unwrap_or_else(|e| fail(format!("cannot load --{role} {}: {e}", path.display())))
    };
    let rank_set = load_set(&rank_path, "rank");
    let cal_set = load_set(&cal_path, "calibration");
    let rep_set = load_set(&rep_path, "report");
    let fit_set = load_set(&generator_fit, "generator-fit");

    // (b) Leakage guard before any generation.
    let leakage = |a_name: &str, a: &[String], b_name: &str, b: &[String]| {
        if let Some(key) = find_shared_key(a, b) {
            let e = Error::config(format!(
                "molecule keys of --{a_name} and --{b_name} are not disjoint (first shared key '{key}')"
            ));
            fail(format!("{e}"));
        }
    };
    leakage("rank", &rank_set.molecules, "calibration", &cal_set.molecules);
    leakage("rank", &rank_set.molecules, "report", &rep_set.molecules);
    leakage("calibration", &cal_set.molecules, "report", &rep_set.molecules);
    // No experiment split may reuse a molecule the generator was fitted on.
    for (role, set) in [
        ("rank", &rank_set),
        ("calibration", &cal_set),
        ("report", &rep_set),
    ] {
        if let Some(key) = find_shared_key(&set.molecules, &fit_set.molecules) {
            refuse(format!(
                "molecule keys of --{role} and --generator-fit are not disjoint (first shared key '{key}'): \
                 the generator was fitted on this export"
            ));
        }
    }
    // The checkpoint's recorded fit provenance must match the supplied
    // `--generator-fit` export (name or SHA-256); a checkpoint that records
    // nothing is taken on the caller's word and reported as such.
    let fit_provenance = match check_generator_fit(
        train_config.enum_fit_name.as_deref(),
        train_config.enum_fit_sha256.as_deref(),
        &fit_set.name,
        &fit_set.source_sha256,
    ) {
        Ok(p) => p,
        Err(msg) => refuse(msg),
    };
    let generator_fit_provenance = match fit_provenance {
        FitProvenance::Matches => "matches the checkpoint's recorded fit export",
        FitProvenance::Unrecorded => "supplied by the caller, not recorded in the checkpoint",
    };
    for (role, set) in [
        ("rank", &rank_set),
        ("calibration", &cal_set),
        ("report", &rep_set),
    ] {
        let hit = train_config
            .enum_fit_sha256
            .as_ref()
            .is_some_and(|sha| sha == &set.source_sha256)
            || train_config
                .enum_fit_name
                .as_ref()
                .is_some_and(|name| name == &set.name);
        if hit {
            let e = Error::config(format!(
                "--{role} {} equals the checkpoint's recorded train export (name/sha): \
                 the reranker experiment needs train-disjoint splits",
                set.name
            ));
            fail(format!("{e}"));
        }
    }

    // (c) Generate candidates per split, then containment, examples, features.
    let mut gen_seconds: Vec<f64> = Vec::new();
    let mut gen_reads: Vec<usize> = Vec::new();
    // First call per chunk-size class is reported but not read-asserted
    // (autotune may read on a new shape); steady-state calls assert one read.
    let mut seen_sizes: BTreeSet<usize> = BTreeSet::new();

    let mut run_split = |set: &ExperimentSet, role: &str| -> SplitExamples {
        let total = limit_spectra.unwrap_or(set.spectra.len()).min(set.spectra.len());
        let indices: Vec<usize> = (0..total).collect();
        let mut examples: Vec<Example> = Vec::new();
        let mut molecule_of_group: Vec<usize> = Vec::with_capacity(total);
        let (mut not_finished, mut invalid, mut duplicate, mut work_limit) = (0, 0, 0, 0);
        for chunk in indices.chunks(batch.max(1)) {
            let r0 = runtime_read_count();
            let t0 = Instant::now();
            let chunk_batch = trainer
                .generate_candidates(set, chunk, &gen_config)
                .unwrap_or_else(|e| fail(format!("{role} generate: {e}")));
            let dt = t0.elapsed().as_secs_f64();
            let reads = runtime_read_count() - r0;
            gen_seconds.push(dt);
            gen_reads.push(reads);
            if !seen_sizes.insert(chunk.len()) && reads != 1 {
                fail(format!(
                    "{role} generate call read {reads} time(s), expected exactly one"
                ));
            }
            let evals = evaluate_candidates(set, chunk, &chunk_batch, contain_work)
                .unwrap_or_else(|e| fail(format!("{role} containment: {e}")));
            let mut contain: Vec<Containment> = Vec::with_capacity(chunk.len() * k as usize);
            for ev in &evals {
                for c in &ev.candidates {
                    contain.push(c.contained);
                }
            }
            // Size stratum of an example: atoms of the candidate's own
            // trace (atom-adding actions of the valid finished record),
            // NOT the replay-under-the-true-parent count (0 when replay
            // under the parent's composition fails). Containment under the
            // true parent stays the label.
            let t = chunk_batch.max_steps;
            let mut atoms_of: Vec<usize> = Vec::with_capacity(chunk.len() * k as usize);
            for r in 0..chunk.len() * k as usize {
                let base = r * t * 4;
                atoms_of.push(trace_atom_count(
                    &chunk_batch.actions[base..base + t * 4],
                    chunk_batch.length[r],
                ));
            }
            let (idx, lab, excl) = eligible_examples(&chunk_batch, &contain)
                .unwrap_or_else(|e| fail(format!("{role} eligible_examples: {e}")));
            not_finished += excl.not_finished;
            invalid += excl.invalid;
            duplicate += excl.duplicate;
            work_limit += excl.work_limit;
            let flat = features_of_batch(&chunk_batch, set, chunk)
                .unwrap_or_else(|e| fail(format!("{role} features: {e}")));
            let group_base = molecule_of_group.len();
            for &global in chunk {
                molecule_of_group.push(set.spectra[global].molecule);
            }
            for (pos, &r) in idx.iter().enumerate() {
                let mut feat = [0.0f32; N_FEATURES];
                feat.copy_from_slice(&flat[r * N_FEATURES..(r + 1) * N_FEATURES]);
                examples.push(Example {
                    group: group_base + r / k as usize,
                    molecule: set.spectra[chunk[r / k as usize]].molecule,
                    atoms: atoms_of[r],
                    raw: f64::from(chunk_batch.formula_log_prob[r] + chunk_batch.trace_log_prob[r]),
                    features: feat,
                    label: f64::from(lab[pos]),
                });
            }
        }
        SplitExamples {
            examples,
            molecule_of_group,
            not_finished,
            invalid,
            duplicate,
            work_limit,
        }
    };

    let rank = run_split(&rank_set, "rank");
    let cal = run_split(&cal_set, "calibration");
    let rep = run_split(&rep_set, "report");
    // Per-example molecule/group consistency: every example's molecule must
    // match its spectrum's molecule in the split's group list.
    for (split, name) in [(&rank, "rank"), (&cal, "calibration"), (&rep, "report")] {
        for ex in &split.examples {
            if ex.molecule != split.molecule_of_group[ex.group] {
                fail(format!(
                    "{name} example group {} names molecule {} but the spectrum has {}",
                    ex.group, ex.molecule, split.molecule_of_group[ex.group]
                ));
            }
        }
    }

    // (h) Refuse to train on one class.
    let rank_pos = rank.examples.iter().filter(|e| e.label != 0.0).count();
    if rank_pos == 0 || rank_pos == rank.examples.len() {
        refuse(format!(
            "--rank has {} positive and {} negative examples after filtering: \
             the reranker needs both classes",
            rank_pos,
            rank.examples.len() - rank_pos
        ));
    }

    // (d) Train the reranker on the rank examples only.
    let n_rank = rank.examples.len();
    let steps_per_epoch = n_rank.div_ceil(examples_per_step).max(1);
    let reranker = {
        let mut rng = Rng::seeded(seed);
        Reranker::<R, E>::init(&device, &mut rng)
    };
    let mut rerank_trainer = RerankTrainer::<R, E>::new(lr, steps_per_epoch);
    // Zero-weight warmup: autotune without touching parameters, then the
    // read budget below covers training only.
    {
        let zf = vec![0.0f32; examples_per_step * N_FEATURES];
        let zl = vec![0.0f32; examples_per_step];
        let zw = vec![0.0f32; examples_per_step];
        let f_t = Tensor::<R, E>::from_f32(&zf, vec![examples_per_step, N_FEATURES], &device)
            .unwrap_or_else(|e| fail(format!("warmup upload: {e}")));
        let l_t = Tensor::<R, E>::from_f32(&zl, vec![examples_per_step], &device)
            .unwrap_or_else(|e| fail(format!("warmup upload: {e}")));
        let w_t = Tensor::<R, E>::from_f32(&zw, vec![examples_per_step], &device)
            .unwrap_or_else(|e| fail(format!("warmup upload: {e}")));
        rerank_trainer
            .step(&reranker, &f_t, &l_t, &w_t, 0)
            .unwrap_or_else(|e| fail(format!("warmup step: {e}")));
    }
    reset_transfer_counters();
    let train_reads_before = runtime_read_count();
    let mut loss_curve: Vec<serde_json::Value> = Vec::with_capacity(epochs);
    let mut step_seconds: Vec<f64> = Vec::with_capacity(epochs * steps_per_epoch);
    let mut order: Vec<usize> = (0..n_rank).collect();
    let mut cold_reads = 0;
    for epoch in 0..epochs {
        let mut rng = SplitMix64::new(seed.wrapping_add(epoch as u64));
        for i in (1..order.len()).rev() {
            let j = (rng.next() % (i as u64 + 1)) as usize;
            order.swap(i, j);
        }
        for step in order.chunks(examples_per_step) {
            // Last partial step padded with zero-weight rows.
            let mut f = vec![0.0f32; examples_per_step * N_FEATURES];
            let mut l = vec![0.0f32; examples_per_step];
            let mut w = vec![0.0f32; examples_per_step];
            for (row, &ei) in step.iter().enumerate() {
                let ex = &rank.examples[ei];
                f[row * N_FEATURES..(row + 1) * N_FEATURES].copy_from_slice(&ex.features);
                l[row] = ex.label as f32;
                w[row] = 1.0;
            }
            let t0 = Instant::now();
            let f_t =
                Tensor::<R, E>::from_f32(&f, vec![examples_per_step, N_FEATURES], &device)
                    .unwrap_or_else(|e| fail(format!("train upload: {e}")));
            let l_t = Tensor::<R, E>::from_f32(&l, vec![examples_per_step], &device)
                .unwrap_or_else(|e| fail(format!("train upload: {e}")));
            let w_t = Tensor::<R, E>::from_f32(&w, vec![examples_per_step], &device)
                .unwrap_or_else(|e| fail(format!("train upload: {e}")));
            let reported = rerank_trainer
                .step(&reranker, &f_t, &l_t, &w_t, step.len())
                .unwrap_or_else(|e| fail(format!("reranker step: {e}")));
            step_seconds.push(t0.elapsed().as_secs_f64());
            if let Some(loss) = reported {
                loss_curve.push(serde_json::json!({"epoch": epoch, "loss": loss}));
            }
        }
        if epoch == 0 {
            // The first epoch also holds the cold-start reads of the runtime
            // (matmul autotuning on a GPU backend), so the read budget is
            // asserted on the warmed epochs only.
            cold_reads = runtime_read_count() - train_reads_before;
        }
    }
    let train_reads = runtime_read_count() - train_reads_before;
    let warm_reads = train_reads - cold_reads;
    if warm_reads > epochs.saturating_sub(1) {
        fail(format!(
            "reranker training read {warm_reads} time(s) over {} warmed epochs (at most one loss read per epoch; the first epoch read {cold_reads} time(s))",
            epochs.saturating_sub(1)
        ));
    }

    // (e) Score calibration and report with the trained reranker.
    let score_split = |split: &SplitExamples, role: &str| -> Vec<f32> {
        let n = split.examples.len();
        if n == 0 {
            return Vec::new();
        }
        let mut f = vec![0.0f32; n * N_FEATURES];
        for (row, ex) in split.examples.iter().enumerate() {
            f[row * N_FEATURES..(row + 1) * N_FEATURES].copy_from_slice(&ex.features);
        }
        let f_t = Tensor::<R, E>::from_f32(&f, vec![n, N_FEATURES], &device)
            .unwrap_or_else(|e| fail(format!("{role} score upload: {e}")));
        // A new matrix shape makes a GPU backend autotune, and the tuner's
        // correctness check reads the device: warm the shape with one
        // unmeasured call, then require exactly one read of the measured one.
        let _warm = reranker
            .logits(&f_t)
            .unwrap_or_else(|e| fail(format!("{role} score warm-up: {e}")))
            .try_to_f32()
            .unwrap_or_else(|e| fail(format!("{role} score warm-up read: {e}")));
        let r0 = runtime_read_count();
        let logits = reranker
            .logits(&f_t)
            .unwrap_or_else(|e| fail(format!("{role} score: {e}")))
            .try_to_f32()
            .unwrap_or_else(|e| fail(format!("{role} score read: {e}")));
        if runtime_read_count() - r0 != 1 {
            fail(format!("{role} scoring did not read exactly once"));
        }
        logits
    };
    let cal_logits = score_split(&cal, "calibration");
    let rep_logits = score_split(&rep, "report");

    let labels_of = |split: &SplitExamples| -> Vec<f64> {
        split.examples.iter().map(|e| e.label).collect()
    };
    let raws_of = |split: &SplitExamples| -> Vec<f64> {
        split.examples.iter().map(|e| e.raw).collect()
    };
    let atoms_of = |split: &SplitExamples| -> Vec<usize> {
        split.examples.iter().map(|e| e.atoms).collect()
    };
    let cal_labels = labels_of(&cal);
    let cal_raws = raws_of(&cal);
    let rep_labels = labels_of(&rep);
    let rep_raws = raws_of(&rep);
    let cal_logits_f64: Vec<f64> = cal_logits.iter().map(|&x| f64::from(x)).collect();
    let rep_logits_f64: Vec<f64> = rep_logits.iter().map(|&x| f64::from(x)).collect();

    let cal_pos = cal_labels.iter().filter(|&&y| y != 0.0).count();
    let params_reranker = fit_platt(&cal_logits_f64, &cal_labels)
        .unwrap_or_else(|e| fail(format!("fit_platt reranker: {e}")));
    let params_raw = fit_platt(&cal_raws, &cal_labels)
        .unwrap_or_else(|e| fail(format!("fit_platt raw: {e}")));
    let art_reranker = CalibrationArtifact::new(
        &params_reranker,
        "calibration",
        cal_labels.len(),
        cal_pos,
        keys_reranker.clone(),
    );
    let art_raw = CalibrationArtifact::new(
        &params_raw,
        "calibration",
        cal_labels.len(),
        cal_pos,
        keys_raw.clone(),
    );

    // (f) Ranking metrics on report (held out) and calibration (not held out
    // for calibration), plus calibration quality on report.
    let rank_metrics = |split: &SplitExamples,
                        logits: &[f64],
                        role: &str|
     -> serde_json::Value {
        let labels = labels_of(split);
        let raws = raws_of(split);
        // Groups are spectra with at least one example: remap the split's
        // group ids to a compact 0..G so empty spectra contribute no zeros.
        let mut present: Vec<usize> = split.examples.iter().map(|e| e.group).collect();
        present.sort_unstable();
        present.dedup();
        let n_groups = present.len();
        let egroup: Vec<usize> = split
            .examples
            .iter()
            .map(|e| present.iter().position(|&g| g == e.group).expect("present covers egroup"))
            .collect();
        let mol_of_group: Vec<usize> = present
            .iter()
            .map(|&g| split.molecule_of_group[g])
            .collect();
        let base_rate = if labels.is_empty() {
            0.0
        } else {
            labels.iter().sum::<f64>() / labels.len() as f64
        };
        // Spectra and molecules with at least one eligible candidate: the
        // `present` groups and their molecules, next to the split totals.
        let mut eligible_molecules: Vec<usize> = mol_of_group.clone();
        eligible_molecules.sort_unstable();
        eligible_molecules.dedup();
        let n_molecules_eligible = eligible_molecules.len();
        let one = |scores: &[f64], name: &str| {
            let auc_value = roc_auc(scores, &labels).map_or(serde_json::Value::Null, |v| {
                serde_json::json!(v)
            });
            let top1_groups = group_top1(scores, &labels, &egroup, n_groups);
            let p4_groups = group_precision_at(scores, &labels, &egroup, n_groups, 4);
            let p8_groups = group_precision_at(scores, &labels, &egroup, n_groups, 8);
            let (_, t1_n) = top1_precision(scores, &labels, &egroup);
            let (t1m, t1lo, t1hi) =
                molecule_interval(&top1_groups, &mol_of_group, bootstrap, seed);
            let (p4m, p4lo, p4hi) =
                molecule_interval(&p4_groups, &mol_of_group, bootstrap, seed);
            let (p8m, p8lo, p8hi) =
                molecule_interval(&p8_groups, &mol_of_group, bootstrap, seed);
            serde_json::json!({
                "auc": {"value": auc_value, "aggregation": "pooled eligible examples"},
                "top1": {"point": t1m, "lo": t1lo, "hi": t1hi, "n_groups": t1_n,
                    "aggregation": "mean over molecules of per-spectrum values, spectra with at least one eligible candidate"},
                "p_at_4": {"point": p4m, "lo": p4lo, "hi": p4hi,
                    "aggregation": "mean over molecules of per-spectrum values, spectra with at least one eligible candidate"},
                "p_at_8": {"point": p8m, "lo": p8lo, "hi": p8hi,
                    "aggregation": "mean over molecules of per-spectrum values, spectra with at least one eligible candidate"},
                "note": name,
            })
        };
        let raw_m = one(&raws, "raw");
        let rr_m = one(logits, "ms2-reranker-v1");
        // Paired reranker − raw differences with molecule-bootstrap intervals.
        // No paired group (no spectrum defined in both rankings) is JSON
        // null, never zero in place of a measured value.
        let pair_json = |opt: Option<(f64, f64, f64)>| -> serde_json::Value {
            match opt {
                Some((point, lo, hi)) => serde_json::json!({"point": point, "lo": lo, "hi": hi}),
                None => serde_json::json!({"point": null, "lo": null, "hi": null}),
            }
        };
        let diff = |r: usize| {
            let a = group_precision_at(logits, &labels, &egroup, n_groups, r);
            let b = group_precision_at(&raws, &labels, &egroup, n_groups, r);
            pair_json(paired_bootstrap(&a, &b, &mol_of_group, bootstrap, seed))
        };
        let a_top1 = group_top1(logits, &labels, &egroup, n_groups);
        let b_top1 = group_top1(&raws, &labels, &egroup, n_groups);
        let d1 = pair_json(paired_bootstrap(&a_top1, &b_top1, &mol_of_group, bootstrap, seed));
        serde_json::json!({
            "role": role,
            "n_examples": labels.len(),
            "n_groups": n_groups,
            "n_spectra_eligible": n_groups,
            "n_molecules_eligible": n_molecules_eligible,
            "base_rate": {"value": base_rate, "aggregation": "pooled eligible examples"},
            "raw": raw_m,
            "reranker": rr_m,
            "paired_reranker_minus_raw": {
                "top1": d1,
                "p_at_4": diff(4),
                "p_at_8": diff(8),
            },
        })
    };
    let report_ranking = rank_metrics(&rep, &rep_logits_f64, "report");
    let mut calibration_ranking = rank_metrics(&cal, &cal_logits_f64, "calibration");
    calibration_ranking["note"] =
        serde_json::json!("not held out for calibration: ranking only, calibration was fitted here");

    // Calibration quality on report, per ranking: uncalibrated (sigmoid)
    // reference plus calibrated ECE/Brier/reliability overall and per stratum.
    let calibration_quality = |logits: &[f64], artifact: &CalibrationArtifact| -> serde_json::Value {
        let probs = artifact
            .apply(&artifact.config, logits)
            .unwrap_or_else(|e| fail(format!("calibration apply: {e}")));
        let uncal: Vec<f64> = logits.iter().map(|&x| sigmoid(x)).collect();
        let atoms = atoms_of(&rep);
        let one_side = |p: &[f64]| {
            let ew = expected_calibration_error(p, &rep_labels, 15, Binning::EqualWidth)
                .unwrap_or_else(|e| fail(format!("ECE width: {e}")));
            let em = expected_calibration_error(p, &rep_labels, 15, Binning::EqualMass)
                .unwrap_or_else(|e| fail(format!("ECE mass: {e}")));
            let brier = brier_score(p, &rep_labels)
                .unwrap_or_else(|e| fail(format!("Brier: {e}")));
            let table_w = reliability_table(p, &rep_labels, 15, Binning::EqualWidth)
                .unwrap_or_else(|e| fail(format!("reliability: {e}")));
            let table_m = reliability_table(p, &rep_labels, 15, Binning::EqualMass)
                .unwrap_or_else(|e| fail(format!("reliability: {e}")));
            let ece_w_s = ece_by_stratum(p, &rep_labels, &atoms, 15, Binning::EqualWidth)
                .unwrap_or_else(|e| fail(format!("stratum ECE: {e}")));
            let ece_m_s = ece_by_stratum(p, &rep_labels, &atoms, 15, Binning::EqualMass)
                .unwrap_or_else(|e| fail(format!("stratum ECE: {e}")));
            let brier_s = brier_by_stratum(p, &rep_labels, &atoms)
                .unwrap_or_else(|e| fail(format!("stratum Brier: {e}")));
            let tables_s = reliability_by_stratum(p, &rep_labels, &atoms, 15, Binning::EqualWidth)
                .unwrap_or_else(|e| fail(format!("stratum reliability: {e}")));
            serde_json::json!({
                "ece_width_15": ew,
                "ece_mass_15": em,
                "brier": brier,
                "reliability_width_15": table_w,
                "reliability_mass_15": table_m,
                "ece_width_15_by_stratum": ece_w_s,
                "ece_mass_15_by_stratum": ece_m_s,
                "brier_by_stratum": brier_s,
                "reliability_width_15_by_stratum": tables_s,
            })
        };
        serde_json::json!({
            "platt": {"a": artifact.a, "b": artifact.b},
            "fit_split": artifact.fit_split,
            "fit_count": artifact.fit_count,
            "fit_positives": artifact.fit_positives,
            "uncalibrated": one_side(&uncal),
            "calibrated": one_side(&probs),
        })
    };
    let report_cal_reranker = calibration_quality(&rep_logits_f64, &art_reranker);
    let report_cal_raw = calibration_quality(&rep_raws, &art_raw);

    // (g) JSON report.
    let split_summary = |set: &ExperimentSet, split: &SplitExamples| {
        let pos = split.examples.iter().filter(|e| e.label != 0.0).count();
        let mols: BTreeSet<usize> = split.molecule_of_group.iter().copied().collect();
        serde_json::json!({
            "file": set.name,
            "source_sha256": set.source_sha256,
            "n_spectra_used": split.molecule_of_group.len(),
            "n_molecules_used": mols.len(),
            "n_examples": split.examples.len(),
            "n_positives": pos,
            "base_rate": if split.examples.is_empty() { 0.0 } else { pos as f64 / split.examples.len() as f64 },
            "excluded": {
                "not_finished": split.not_finished,
                "invalid": split.invalid,
                "duplicate": split.duplicate,
                "work_limit": split.work_limit,
            },
        })
    };
    let table_file_name = table_path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| table_path.display().to_string());
    let gen_config_json = serde_json::to_value(&gen_config)
        .unwrap_or_else(|e| fail(format!("generation config JSON: {e}")));
    let (g50, g95) = p50p95(gen_seconds.clone());
    let (s50, s95) = p50p95(step_seconds.clone());
    let report = serde_json::json!({
        "schema_version": 1,
        "command_line": argv,
        "labels": "containment in the true parent (contracts §7.3); pseudo-label metric",
        "provenance": {
            "checkpoint_path": generator.display().to_string(),
            "checkpoint_sha256": ckpt_sha256,
            "generator_fit": {"file": generator_fit.display().to_string(), "name": fit_set.name, "source_sha256": fit_set.source_sha256},
            "generator_fit_provenance": generator_fit_provenance,
            "rank": {"file": rank_set.name, "source_sha256": rank_set.source_sha256},
            "calibration": {"file": cal_set.name, "source_sha256": cal_set.source_sha256},
            "report": {"file": rep_set.name, "source_sha256": rep_set.source_sha256},
            "table_file": table_file_name,
            "table_sha256": trainer.table_sha256(),
            "table_rows": table.len(),
            "model_config": trainer.model.config,
            "train_config": train_config,
            "generation_config": gen_config_json.clone(),
            "config_keys": {"reranker": keys_reranker, "raw": keys_raw},
        },
        "splits": {
            "rank": split_summary(&rank_set, &rank),
            "calibration": split_summary(&cal_set, &cal),
            "report": split_summary(&rep_set, &rep),
        },
        "rerank_training": {
            "epochs": epochs,
            "examples_per_step": examples_per_step,
            "steps_per_epoch": steps_per_epoch,
            "device_reads": train_reads,
            "loss_curve": loss_curve,
        },
        "ranking_report": report_ranking,
        "ranking_calibration": calibration_ranking,
        "calibration_report": {
            "reranker": report_cal_reranker,
            "raw": report_cal_raw,
        },
        "timing": {
            "generate_seconds_per_call": {"p50": g50, "p95": g95, "n": gen_reads.len()},
            "reads_per_generate_call": gen_reads,
            "rerank_step_seconds": {"p50": s50, "p95": s95, "n": step_seconds.len()},
            "total_seconds": started.elapsed().as_secs_f64(),
        },
    });
    if let Some(parent) = out.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)
            .unwrap_or_else(|e| fail(format!("cannot create {}: {e}", parent.display())));
    }
    std::fs::write(&out, serde_json::to_string_pretty(&report).unwrap())
        .unwrap_or_else(|e| fail(format!("cannot write {}: {e}", out.display())));

    if let Some(path) = &save_reranker {
        reranker
            .save(path, &gen_config_json)
            .unwrap_or_else(|e| fail(format!("cannot save reranker {}: {e}", path.display())));
    }
    if let Some(path) = &save_calibration {
        let both = serde_json::json!({"reranker": art_reranker, "raw": art_raw});
        std::fs::write(path, serde_json::to_string_pretty(&both).unwrap())
            .unwrap_or_else(|e| fail(format!("cannot save calibration {}: {e}", path.display())));
    }

    // Short table to stdout.
    let get3 = |v: &serde_json::Value| -> (f64, f64, f64) {
        (
            v["point"].as_f64().unwrap_or(f64::NAN),
            v["lo"].as_f64().unwrap_or(f64::NAN),
            v["hi"].as_f64().unwrap_or(f64::NAN),
        )
    };
    println!("split      spectra  molecules  examples  positives  base_rate");
    for (name, split) in [("rank", &rank), ("cal", &cal), ("rep", &rep)] {
        let pos = split.examples.iter().filter(|e| e.label != 0.0).count();
        let mols: BTreeSet<usize> = split.molecule_of_group.iter().copied().collect();
        println!(
            "{name:10} {:>7}  {:>9}  {:>8}  {:>9}  {:.4}",
            split.molecule_of_group.len(),
            mols.len(),
            split.examples.len(),
            pos,
            pos as f64 / split.examples.len().max(1) as f64,
        );
    }
    println!(
        "excluded   rank (unfin/inv/dup/work) {}/{}/{}/{}; cal {}/{}/{}/{}; rep {}/{}/{}/{}",
        rank.not_finished,
        rank.invalid,
        rank.duplicate,
        rank.work_limit,
        cal.not_finished,
        cal.invalid,
        cal.duplicate,
        cal.work_limit,
        rep.not_finished,
        rep.invalid,
        rep.duplicate,
        rep.work_limit,
    );
    for (name, m) in [("raw", &report_ranking["raw"]), ("reranker", &report_ranking["reranker"])] {
        let auc = m["auc"]["value"].as_f64().unwrap_or(f64::NAN);
        let (t1, t1lo, t1hi) = get3(&m["top1"]);
        let (p4, p4lo, p4hi) = get3(&m["p_at_4"]);
        let (p8, p8lo, p8hi) = get3(&m["p_at_8"]);
        println!("report {name:8} AUC {auc:.4} top1 {t1:.4} [{t1lo:.4},{t1hi:.4}] p@4 {p4:.4} [{p4lo:.4},{p4hi:.4}] p@8 {p8:.4} [{p8lo:.4},{p8hi:.4}]");
    }
    for name in ["top1", "p_at_4", "p_at_8"] {
        let (d, lo, hi) = get3(&report_ranking["paired_reranker_minus_raw"][name]);
        println!("paired {name:4} reranker-raw {d:.4} [{lo:.4},{hi:.4}]");
    }
    for (name, q) in [("reranker", &report_cal_reranker), ("raw", &report_cal_raw)] {
        println!(
            "calib {name:8} uncal ECEw {:.4} Brier {:.4} | cal ECEw {:.4} ECEq {:.4} Brier {:.4}",
            q["uncalibrated"]["ece_width_15"].as_f64().unwrap_or(f64::NAN),
            q["uncalibrated"]["brier"].as_f64().unwrap_or(f64::NAN),
            q["calibrated"]["ece_width_15"].as_f64().unwrap_or(f64::NAN),
            q["calibrated"]["ece_mass_15"].as_f64().unwrap_or(f64::NAN),
            q["calibrated"]["brier"].as_f64().unwrap_or(f64::NAN),
        );
    }
    println!(
        "timing     gen {g50:.4}s p50/call; rerank step {s50:.4}s p50; train reads {train_reads}"
    );
    println!("out        {}", out.display());
}
