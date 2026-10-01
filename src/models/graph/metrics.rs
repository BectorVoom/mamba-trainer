//! Metrics of the graph tasks (GRAPH_MAMBA_PLAN.md GM7).
//!
//! The countable metrics are read off values the device already reduced:
//! [`accuracy`] and [`f1_macro`] from a confusion matrix summed over a split's
//! batches, [`weighted_mean`] from per-batch means and counts. The rank
//! metrics, [`average_precision`] and [`roc_auc`], sort — so they run here, on
//! one read of the scores with their labels and masks.

/// The fraction of correct predictions in a `[classes, classes]` confusion
/// matrix (row: true class, column: predicted class); 0 for an empty one.
pub fn accuracy(confusion: &[u64], classes: usize) -> f32 {
    let total: u64 = confusion.iter().sum();
    if total == 0 {
        return 0.0;
    }
    let correct: u64 = (0..classes).map(|c| confusion[c * classes + c]).sum();
    correct as f32 / total as f32
}

/// The unweighted mean of the per-class F1 scores, over the classes that occur
/// among the true or the predicted labels — scikit-learn's `f1_score(average =
/// "macro")` with its default labels. A class the split does not contain and
/// the model never predicts is not averaged in; 0 for an empty matrix.
pub fn f1_macro(confusion: &[u64], classes: usize) -> f32 {
    let (mut sum, mut present) = (0.0f64, 0usize);
    for c in 0..classes {
        let hit = confusion[c * classes + c] as f64;
        let actual: f64 = (0..classes).map(|p| confusion[c * classes + p] as f64).sum();
        let predicted: f64 = (0..classes).map(|t| confusion[t * classes + c] as f64).sum();
        if actual + predicted > 0.0 {
            sum += 2.0 * hit / (actual + predicted);
            present += 1;
        }
    }
    (sum / present.max(1) as f64) as f32
}

/// `Σ mean·count / Σ count` over batches: a masked mean over a whole split
/// from each batch's masked mean and its count. 0 when nothing was counted.
pub fn weighted_mean(parts: &[(f32, f32)]) -> f32 {
    let total: f64 = parts.iter().map(|&(_, count)| count as f64).sum();
    if total == 0.0 {
        return 0.0;
    }
    let sum: f64 = parts
        .iter()
        .map(|&(mean, count)| mean as f64 * count as f64)
        .sum();
    (sum / total) as f32
}

/// The kept `(score, is positive)` pairs of column `column` of row-major
/// `[rows, columns]` scores, labels and mask.
fn column(
    scores: &[f32],
    labels: &[f32],
    mask: &[f32],
    columns: usize,
    column: usize,
) -> Vec<(f32, bool)> {
    (column..scores.len())
        .step_by(columns)
        .filter(|&i| mask[i] != 0.0)
        .map(|i| (scores[i], labels[i] > 0.5))
        .collect()
}

/// Average precision of one column: the area under the precision–recall
/// curve taken at every distinct score, as scikit-learn computes it. `None`
/// when the column has no positive or no negative; `NaN` when a score is `NaN`
/// (a diverged model has no ranking).
fn average_precision_of(mut pairs: Vec<(f32, bool)>) -> Option<f64> {
    let positives = pairs.iter().filter(|p| p.1).count();
    if positives == 0 || positives == pairs.len() {
        return None;
    }
    if pairs.iter().any(|p| p.0.is_nan()) {
        return Some(f64::NAN);
    }
    pairs.sort_by(|a, b| b.0.total_cmp(&a.0));
    let (mut hits, mut seen, mut area, mut recall) = (0usize, 0usize, 0.0f64, 0.0f64);
    let mut i = 0;
    while i < pairs.len() {
        // Every item tied at this score enters at once.
        let score = pairs[i].0;
        while i < pairs.len() && pairs[i].0 == score {
            hits += pairs[i].1 as usize;
            seen += 1;
            i += 1;
        }
        let now = hits as f64 / positives as f64;
        area += (now - recall) * hits as f64 / seen as f64;
        recall = now;
    }
    Some(area)
}

/// Area under the ROC curve of one column, by ranks with ties averaged.
/// `None` when the column has no positive or no negative; `NaN` when a score
/// is `NaN`.
fn roc_auc_of(mut pairs: Vec<(f32, bool)>) -> Option<f64> {
    let positives = pairs.iter().filter(|p| p.1).count();
    let negatives = pairs.len() - positives;
    if positives == 0 || negatives == 0 {
        return None;
    }
    if pairs.iter().any(|p| p.0.is_nan()) {
        return Some(f64::NAN);
    }
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut rank_sum = 0.0f64;
    let mut i = 0;
    while i < pairs.len() {
        let mut j = i;
        while j < pairs.len() && pairs[j].0 == pairs[i].0 {
            j += 1;
        }
        // Ranks `i + 1 ..= j` share their mean.
        let rank = (i + 1 + j) as f64 / 2.0;
        rank_sum += rank * pairs[i..j].iter().filter(|p| p.1).count() as f64;
        i = j;
    }
    let p = positives as f64;
    Some((rank_sum - p * (p + 1.0) / 2.0) / (p * negatives as f64))
}

/// The mean of a per-column metric over the columns it is defined for.
fn mean_over_columns(
    scores: &[f32],
    labels: &[f32],
    mask: &[f32],
    columns: usize,
    metric: fn(Vec<(f32, bool)>) -> Option<f64>,
) -> Option<f32> {
    assert!(columns > 0 && scores.len() == labels.len() && scores.len() == mask.len());
    let values: Vec<f64> = (0..columns)
        .filter_map(|c| metric(column(scores, labels, mask, columns, c)))
        .collect();
    (!values.is_empty()).then(|| (values.iter().sum::<f64>() / values.len() as f64) as f32)
}

/// Mean average precision over the label columns of row-major
/// `[rows, columns]` scores, 0/1 labels and mask. A column whose kept labels
/// are all of one class is skipped; `None` if every column is.
pub fn average_precision(
    scores: &[f32],
    labels: &[f32],
    mask: &[f32],
    columns: usize,
) -> Option<f32> {
    mean_over_columns(scores, labels, mask, columns, average_precision_of)
}

/// Mean ROC AUC over the label columns of row-major `[rows, columns]` scores,
/// 0/1 labels and mask; ranks are averaged over ties. A column whose kept
/// labels are all of one class is skipped; `None` if every column is.
pub fn roc_auc(scores: &[f32], labels: &[f32], mask: &[f32], columns: usize) -> Option<f32> {
    mean_over_columns(scores, labels, mask, columns, roc_auc_of)
}
