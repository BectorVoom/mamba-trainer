//! The dataset of one experiment and the per-fold sample sets.

use std::collections::HashMap;

use clm_data::{FeatureTensor, Labels, Normalizer, Panel, SampleIndex, compute_features, compute_labels};
use clm_eval::folds::{Fold, SegmentKind};
use clm_eval::purging::{check_no_leakage, split_samples};
use clm_model::{Split, input_valid};

use crate::config::ExperimentConfig;
use crate::config::anyhow_like::{Result, err, wrap};

/// Panel, raw features and labels.
pub struct Dataset {
    /// Bars.
    pub panel: Panel,
    /// Raw (un-normalised) features.
    pub features: FeatureTensor,
    /// Labels.
    pub labels: Labels,
    /// `yyyy-mm-dd` per day.
    pub dates: Vec<String>,
    /// Year per day.
    pub years: Vec<i32>,
    /// A fingerprint of the bars (symbol, first/last date, row count, sum of closes).
    pub version: String,
}

impl Dataset {
    /// Load bars and compute features and labels.
    pub fn load(cfg: &ExperimentConfig) -> Result<Self> {
        let (panel, report) = Panel::load_dir(&cfg.bars_dir).map_err(wrap("loading bars"))?;
        eprintln!(
            "loaded {} stocks x {} days ({} .. {}); {} symbol reports",
            panel.stocks.len(),
            panel.dates.len(),
            panel.dates.first().copied().unwrap_or(0),
            panel.dates.last().copied().unwrap_or(0),
            report.symbols.len()
        );
        let features = compute_features(&panel);
        let labels = compute_labels(&panel, &cfg.label_config()).map_err(wrap("labels"))?;
        let dates = panel
            .dates
            .iter()
            .map(|d| format!("{:04}-{:02}-{:02}", d / 10000, d / 100 % 100, d % 100))
            .collect();
        let years = panel.years();
        let mut h: u64 = 0xcbf29ce484222325;
        let mut mix = |x: u64| {
            h ^= x;
            h = h.wrapping_mul(0x100000001b3);
        };
        for d in &panel.dates {
            mix(*d as u64);
        }
        // Every field features, labels and the backtest read, plus presence.
        for s in panel.stocks.iter().chain(std::iter::once(&panel.market)).chain(&panel.sectors) {
            for b in s.symbol.bytes() {
                mix(b as u64);
            }
            for t in 0..s.present.len() {
                mix(s.present[t] as u64);
                if s.present[t] {
                    for v in [s.open[t], s.high[t], s.low[t], s.close[t], s.volume[t], s.dollar_volume[t]] {
                        mix(v.to_bits());
                    }
                }
            }
        }
        for (i, sec) in panel.sector_of.iter().enumerate() {
            mix(((i as u64) << 32) | *sec as u64);
        }
        let version = format!("bars-{h:016x}-{}d", panel.dates.len());
        Ok(Self {
            panel,
            features,
            labels,
            dates,
            years,
            version,
        })
    }

    /// Daily opens per stock (NaN when missing), for the backtest.
    pub fn opens(&self) -> Vec<Vec<f64>> {
        self.panel
            .stocks
            .iter()
            .map(|s| {
                s.open
                    .iter()
                    .zip(&s.present)
                    .map(|(o, p)| if *p { *o } else { f64::NAN })
                    .collect()
            })
            .collect()
    }
}

/// One fold's normalised features and sample sets.
pub struct FoldData {
    /// Features normalised with this fold's training statistics.
    pub normalized: FeatureTensor,
    /// The normaliser (saved with the fold).
    pub normalizer: Normalizer,
    /// Labelled, kept, input-valid samples per split; `Test` holds every
    /// input-valid test day (labelled or not), `CalTrade` is not a split here.
    pub splits: HashMap<Split, Vec<SampleIndex>>,
}

/// Prepare a fold: normalise on training days only, purge, check for leakage.
pub fn prepare_fold(ds: &Dataset, fold: &Fold, cfg: &ExperimentConfig) -> Result<FoldData> {
    let w = cfg.window;
    // C4: statistics from the feature rows of training days only.
    let normalizer = Normalizer::fit(&ds.features, fold.train.days()).map_err(wrap("normaliser"))?;
    let normalized = normalizer.apply(&ds.features);
    let ns = ds.features.num_securities;
    let mut labelled = Vec::new();
    let mut test_all = Vec::new();
    for s in 0..ns {
        for t in fold.train.start..fold.test.end {
            if !input_valid(&ds.features, s, t, w) {
                continue;
            }
            if fold.test.contains(t) {
                test_all.push(SampleIndex::new(s, t));
            }
            if ds.labels.class_of(s, t).is_some() {
                labelled.push(SampleIndex::new(s, t));
            }
        }
    }
    let purged = split_samples(fold, &labelled);
    check_no_leakage(fold, &purged, cfg.labels.horizon, cfg.folds.embargo)
        .map_err(|e| err(format!("fold {}: leakage guard: {e}", fold.test_year)))?;
    let mut splits = HashMap::new();
    splits.insert(Split::Train, purged.get(SegmentKind::Train).to_vec());
    splits.insert(Split::Val, purged.get(SegmentKind::Val).to_vec());
    splits.insert(Split::Cal, purged.get(SegmentKind::Cal).to_vec());
    splits.insert(Split::Test, test_all);
    for (name, split) in [("train", Split::Train), ("val", Split::Val), ("cal", Split::Cal)] {
        if splits[&split].is_empty() {
            return Err(err(format!("fold {}: empty {name} split", fold.test_year)));
        }
    }
    Ok(FoldData {
        normalized,
        normalizer,
        splits,

    })
}

/// FNV-1a of a list of samples (order-sensitive), for manifests.
pub fn samples_hash(samples: &[SampleIndex]) -> String {
    let mut h: u64 = 0xcbf29ce484222325;
    for smp in samples {
        for x in [smp.s as u64, smp.t as u64] {
            h ^= x;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    format!("{h:016x}-{}", samples.len())
}

/// What an external baseline must echo back for its predictions of a fold to be
/// accepted: the data, the fold boundaries and the exact sample keys of each split.
pub fn fold_manifest(ds: &Dataset, fold: &Fold, fd: &FoldData) -> serde_json::Value {
    let split = |k: Split| samples_hash(&fd.splits[&k]);
    let norm = serde_json::to_string(&fd.normalizer).unwrap_or_default();
    let mut h: u64 = 0xcbf29ce484222325;
    for b in norm.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    serde_json::json!({
        "dataset_version": ds.version,
        "normalizer": format!("{h:016x}"),
        "fold": clm_eval::folds::render_folds_line(fold),
        "train": split(Split::Train),
        "val": split(Split::Val),
        "cal": split(Split::Cal),
        "test": split(Split::Test),
    })
}
