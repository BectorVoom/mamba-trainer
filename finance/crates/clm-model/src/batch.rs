//! Host-side batches, independent of where the samples come from.

use serde::{Deserialize, Serialize};

/// One batch of samples, on the host.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Batch {
    /// Samples.
    pub size: usize,
    /// Time steps per sample.
    pub window: usize,
    /// Features per step.
    pub features: usize,
    /// `[size][window][features]`, normalised.
    pub inputs: Vec<f32>,
    /// Class ids (BUY 0, SELL 1, HOLD 2).
    pub labels: Vec<u32>,
    /// `[size][3]`: 1 for every action counted as a positive (the label, plus any
    /// neighbour within the ambiguity band).
    pub positives: Vec<f32>,
    /// Per-sample loss weight.
    pub weights: Vec<f32>,
    /// Vol-scaled forward return.
    pub ret_target: Vec<f32>,
    /// Stop-loss flag (0/1).
    pub stop_target: Vec<f32>,
}

impl Batch {
    /// Check the field lengths agree.
    pub fn validate(&self) -> Result<(), String> {
        let n = self.size;
        let checks = [
            ("inputs", self.inputs.len(), n * self.window * self.features),
            ("labels", self.labels.len(), n),
            ("positives", self.positives.len(), 3 * n),
            ("weights", self.weights.len(), n),
            ("ret_target", self.ret_target.len(), n),
            ("stop_target", self.stop_target.len(), n),
        ];
        for (name, got, want) in checks {
            if got != want {
                return Err(format!("batch field {name}: {got} values, expected {want}"));
            }
        }
        if let Some(&l) = self.labels.iter().find(|&&l| l >= 3) {
            return Err(format!("label {l} out of range"));
        }
        for i in 0..n {
            let row = &self.positives[3 * i..3 * i + 3];
            if row[self.labels[i] as usize] != 1.0 {
                return Err(format!("sample {i}: its label is not among its positives"));
            }
        }
        Ok(())
    }
}

/// Where training and evaluation batches come from.
pub trait BatchSource {
    /// A random training batch of `size` (sampled with the given seed stream).
    fn train_batch(&mut self, size: usize) -> Batch;
    /// Number of samples in the named evaluation split.
    fn eval_len(&self, split: Split) -> usize;
    /// Evaluation batch `index` (in order, `size` samples each, the last shorter).
    fn eval_batch(&self, split: Split, index: usize, size: usize) -> Batch;
}

/// Evaluation splits of a fold.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Split {
    /// Training samples (for diagnostics only).
    Train,
    /// Hyperparameter / early-stopping validation.
    Val,
    /// Calibration.
    Cal,
    /// Outer test.
    Test,
}
