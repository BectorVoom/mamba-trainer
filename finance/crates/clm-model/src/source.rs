//! [`BatchSource`] over a normalised feature panel and its labels.

use std::collections::HashMap;

use clm_data::{Action, FeatureTensor, Labels, SampleIndex, gather_windows};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

use crate::batch::{Batch, BatchSource, Split};

/// How training batches are drawn and weighted.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SourceConfig {
    /// Time steps per input window.
    pub window: usize,
    /// Weight samples by inverse class frequency of the training split.
    pub class_balance: bool,
    /// A sample whose forward return is within `band * sigma_h` of a class
    /// threshold also counts the class on the other side as a positive
    /// (multi-positive InfoNCE). 0 turns it off.
    pub ambiguity_band: f64,
    /// Sampling seed.
    pub seed: u64,
}

impl Default for SourceConfig {
    fn default() -> Self {
        Self {
            window: 64,
            class_balance: false,
            ambiguity_band: 0.0,
            seed: 0,
        }
    }
}

/// Samples of one fold over a normalised panel.
pub struct PanelSource<'a> {
    features: &'a FeatureTensor,
    labels: &'a Labels,
    splits: HashMap<Split, Vec<SampleIndex>>,
    class_weight: [f32; 3],
    cfg: SourceConfig,
    rng: StdRng,
}

impl<'a> PanelSource<'a> {
    /// `features` must already be normalised with the fold's training statistics.
    /// Every sample of `train`, `val` and `cal` must have a label; `test` samples
    /// may not (they are still predicted and traded).
    pub fn new(
        features: &'a FeatureTensor,
        labels: &'a Labels,
        splits: HashMap<Split, Vec<SampleIndex>>,
        cfg: SourceConfig,
    ) -> Self {
        let mut counts = [0usize; 3];
        for s in splits.get(&Split::Train).into_iter().flatten() {
            let c = labels
                .class_of(s.security(), s.day())
                .expect("training samples carry labels");
            counts[c.id()] += 1;
        }
        let total: usize = counts.iter().sum();
        let class_weight = if cfg.class_balance && total > 0 {
            counts.map(|n| if n == 0 { 0.0 } else { total as f32 / (3.0 * n as f32) })
        } else {
            [1.0; 3]
        };
        let rng = StdRng::seed_from_u64(cfg.seed);
        Self {
            features,
            labels,
            splits,
            class_weight,
            cfg,
            rng,
        }
    }

    /// The samples of a split.
    pub fn samples(&self, split: Split) -> &[SampleIndex] {
        self.splits.get(&split).map_or(&[], Vec::as_slice)
    }

    fn positives(&self, s: SampleIndex, label: Action) -> [f32; 3] {
        let mut p = [0.0f32; 3];
        p[label.id()] = 1.0;
        if self.cfg.ambiguity_band > 0.0 {
            let (i, t) = (s.security(), s.day());
            let idx = i * self.labels.num_days + t;
            let r = self.labels.fwd_ret[idx];
            let theta = self.labels.theta[idx];
            let band = self.cfg.ambiguity_band * self.labels.sigma_h[idx];
            // Near +theta: BUY and HOLD are both defensible; near -theta: SELL and HOLD.
            if (r - theta).abs() < band {
                p[Action::Buy.id()] = 1.0;
                p[Action::Hold.id()] = 1.0;
            }
            if (r + theta).abs() < band {
                p[Action::Sell.id()] = 1.0;
                p[Action::Hold.id()] = 1.0;
            }
        }
        p
    }

    fn build(&self, samples: &[SampleIndex]) -> Batch {
        let n = samples.len();
        let mut b = Batch {
            size: n,
            window: self.cfg.window,
            features: self.features.num_features,
            inputs: gather_windows(self.features, samples, self.cfg.window),
            labels: Vec::with_capacity(n),
            positives: Vec::with_capacity(3 * n),
            weights: Vec::with_capacity(n),
            ret_target: Vec::with_capacity(n),
            stop_target: Vec::with_capacity(n),
        };
        for &s in samples {
            let (i, t) = (s.security(), s.day());
            match self.labels.class_of(i, t) {
                Some(label) => {
                    b.labels.push(label.id() as u32);
                    b.positives.extend(self.positives(s, label));
                    b.weights.push(self.class_weight[label.id()]);
                    b.ret_target.push(self.labels.vol_ret_of(i, t) as f32);
                    b.stop_target.push(if self.labels.stop_of(i, t) { 1.0 } else { 0.0 });
                }
                None => {
                    // An unlabelled test day: predicted and traded, never scored.
                    b.labels.push(Action::Hold.id() as u32);
                    b.positives.extend([0.0, 0.0, 1.0]);
                    b.weights.push(0.0);
                    b.ret_target.push(0.0);
                    b.stop_target.push(0.0);
                }
            }
        }
        b
    }
}

impl BatchSource for PanelSource<'_> {
    fn train_batch(&mut self, size: usize) -> Batch {
        let train = self.splits.get(&Split::Train).map_or(&[][..], Vec::as_slice);
        assert!(!train.is_empty(), "empty training split");
        let picks: Vec<SampleIndex> = (0..size)
            .map(|_| train[self.rng.random_range(0..train.len())])
            .collect();
        self.build(&picks)
    }

    fn eval_len(&self, split: Split) -> usize {
        self.samples(split).len()
    }

    fn eval_batch(&self, split: Split, index: usize, size: usize) -> Batch {
        let all = self.samples(split);
        let lo = (index * size).min(all.len());
        let hi = (lo + size).min(all.len());
        self.build(&all[lo..hi])
    }
}

/// Days `t` of security `s` whose whole input window `t-W+1..=t` is valid.
pub fn input_valid(features: &FeatureTensor, s: usize, t: usize, window: usize) -> bool {
    t + 1 >= window && (t + 1 - window..=t).all(|d| features.is_valid(s, d))
}
