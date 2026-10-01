//! GM7: synthetic tasks the model must learn, each built so that one part of
//! the model is what makes it learnable.
//!
//! The thresholds are first targets. Each test prints what it measured.

#![cfg(feature = "backend")]

use mamba3::backend::Device;
use mamba3::backends::Auto;
use mamba3::models::graph::{
    EvalOptions, FeatureSpec, Features, GraphData, GraphDataset, GraphMamba, GraphMambaSpec,
    GraphPool, GraphTaskSpec, GraphTrainConfig, GraphTrainer, Labels, LocalEncoder, Metric,
    MpnnKind, NodeOrder, RegressionLoss, Split, Splits,
};
use mamba3::models::vision::ScanDirection;

type R = Auto;
type Model = GraphMamba<R, f32>;
type Dataset = GraphDataset<R, f32>;

fn dev() -> Device<R> {
    Device::<R>::default()
}

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }

    fn below(&mut self, n: usize) -> usize {
        self.next() as usize % n
    }
}

/// A 60 / 20 / 20 split of `items`, by a hash of the index.
fn split_60_20_20(items: usize, seed: u64) -> Splits {
    let mut rng = Lcg(seed);
    let draw: Vec<usize> = (0..items).map(|_| rng.below(10)).collect();
    Splits {
        train: draw.iter().map(|&d| d < 6).collect(),
        val: draw.iter().map(|&d| (6..8).contains(&d)).collect(),
        test: draw.iter().map(|&d| d >= 8).collect(),
    }
}

/// Train for `steps` optimizer steps, one epoch after another.
fn train(
    model: &Model,
    dataset: &Dataset,
    steps: u64,
    learning_rate: f32,
    epoch_of: &dyn Fn(u64) -> mamba3::models::graph::Epoch<R, f32>,
) -> Vec<f32> {
    let _ = dataset;
    let mut trainer = GraphTrainer::new(&GraphTrainConfig {
        learning_rate,
        ..Default::default()
    })
    .unwrap();
    let mut epoch = 0;
    while trainer.step_count() < steps {
        model.train_epoch(&mut trainer, &epoch_of(epoch)).unwrap();
        epoch += 1;
    }
    trainer.read_losses().unwrap().iter().map(|i| i.loss).collect()
}

fn curve(losses: &[f32]) -> String {
    let every = (losses.len() / 6).max(1);
    losses
        .iter()
        .step_by(every)
        .chain(losses.last())
        .map(|l| format!("{l:.3}"))
        .collect::<Vec<_>>()
        .join(" → ")
}

// ---------------------------------------------------------------------------
// 1 and 5. Neighbour majority
// ---------------------------------------------------------------------------

/// A random graph of 300 nodes with mean degree 6, one random bit per node,
/// and the label "more than half of my neighbours have bit 1" (a tie, or no
/// neighbour, is unlabelled).
fn neighbour_majority() -> GraphData {
    let n = 300usize;
    let mut rng = Lcg(2024);
    let (mut src, mut dst) = (Vec::new(), Vec::new());
    let mut neighbours = vec![std::collections::BTreeSet::new(); n];
    for _ in 0..n * 3 {
        let (a, b) = (rng.below(n), rng.below(n));
        if a != b {
            src.extend([a as u32, b as u32]);
            dst.extend([b as u32, a as u32]);
            neighbours[a].insert(b);
            neighbours[b].insert(a);
        }
    }
    let bits: Vec<usize> = (0..n).map(|_| rng.below(2)).collect();
    let labels = (0..n)
        .map(|v| {
            let ones: usize = neighbours[v].iter().map(|&u| bits[u]).sum();
            match (2 * ones).cmp(&neighbours[v].len()) {
                std::cmp::Ordering::Greater => 1,
                std::cmp::Ordering::Less => 0,
                std::cmp::Ordering::Equal => -1,
            }
        })
        .collect();
    let mut data = GraphData::new(
        n,
        src,
        dst,
        Features::Float {
            dim: 1,
            data: bits.iter().map(|&b| b as f32 * 2.0 - 1.0).collect(),
        },
    );
    data.y = Labels::Node(labels);
    data.masks = Some(split_60_20_20(n, 7));
    data
}

fn majority_spec(max_hops: usize) -> GraphMambaSpec {
    GraphMambaSpec::new(
        FeatureSpec::Float { dim: 1 },
        GraphTaskSpec::NodeClass { classes: 2 },
    )
    .with_tokens(max_hops, 16, 2)
    .with_local(LocalEncoder::Mean)
    .with_node_layers(1)
    .with_seed(1)
}

fn majority_accuracy(max_hops: usize, parts: usize) -> f32 {
    let device = dev();
    let spec = majority_spec(max_hops);
    let dataset = Dataset::new(&spec, neighbour_majority(), &device).unwrap();
    let model = Model::init(&spec, &device).unwrap();
    let losses = train(&model, &dataset, 300, 3e-3, &|epoch| {
        dataset.epoch_nodes(Some(parts), epoch, Split::Train).unwrap()
    });
    let options = EvalOptions {
        parts: Some(parts),
        ..Default::default()
    };
    let test = model
        .evaluate(&dataset, Split::Test, Metric::Accuracy, &options)
        .unwrap();
    println!(
        "neighbour majority, max_hops = {max_hops}, parts = {parts}: test accuracy {:.3} on {} \
         nodes; loss {}",
        test.value,
        test.count,
        curve(&losses)
    );
    test.value
}

#[test]
fn neighbour_majority_needs_the_token_stage() {
    let with_tokens = majority_accuracy(1, 1);
    let without = majority_accuracy(0, 1);
    assert!(with_tokens > 0.85, "with walk tokens: {with_tokens}");
    assert!(without < 0.65, "node tokens only: {without}");
}

#[test]
fn neighbour_majority_trains_on_node_partitions() {
    let accuracy = majority_accuracy(1, 3);
    assert!(accuracy > 0.85, "three node parts: {accuracy}");
}

// ---------------------------------------------------------------------------
// 2. Long range
// ---------------------------------------------------------------------------

/// Paths of 40 nodes; node 0 of each carries one of four colours as a one-hot
/// feature, every other node has zero features, and every node's label is
/// that colour.
fn coloured_paths(graphs: usize) -> GraphData {
    let len = 40usize;
    let mut rng = Lcg(99);
    let (mut src, mut dst, mut ptr) = (Vec::new(), Vec::new(), vec![0u32]);
    let mut x = vec![0.0f32; graphs * len * 4];
    let mut y = Vec::with_capacity(graphs * len);
    for g in 0..graphs {
        let base = g * len;
        for i in 0..len - 1 {
            src.extend([(base + i) as u32, (base + i + 1) as u32]);
            dst.extend([(base + i + 1) as u32, (base + i) as u32]);
        }
        let colour = rng.below(4);
        x[base * 4 + colour] = 1.0;
        y.extend(std::iter::repeat_n(colour as i64, len));
        ptr.push((base + len) as u32);
    }
    let n = graphs * len;
    let mut data = GraphData::new(n, src, dst, Features::Float { dim: 4, data: x });
    data.graph_ptr = ptr;
    data.y = Labels::Node(y);
    // Whole graphs go to one split: 4 of 5 train, the rest test.
    data.masks = Some(Splits {
        train: (0..n).map(|i| (i / len) % 5 != 4).collect(),
        val: vec![false; n],
        test: (0..n).map(|i| (i / len) % 5 == 4).collect(),
    });
    data
}

/// Test accuracy over all nodes, and over the 38 interior nodes of each path
/// (those a descending-degree order scans before node 0).
fn long_range(direction: ScanDirection) -> (f32, f32) {
    let device = dev();
    let graphs = 100usize;
    let spec = GraphMambaSpec::new(
        FeatureSpec::Float { dim: 4 },
        GraphTaskSpec::NodeClass { classes: 4 },
    )
    .with_tokens(0, 1, 1)
    .with_node_layers(2)
    .with_direction(direction)
    // Interior nodes (degree 2) first, the two ends last: node 0 is scanned
    // second to last.
    .with_order(NodeOrder::Degree { descending: true })
    .with_seed(2);
    let data = coloured_paths(graphs);
    let labels = match &data.y {
        Labels::Node(y) => y.clone(),
        _ => unreachable!(),
    };
    let dataset = Dataset::new(&spec, data, &device).unwrap();
    let model = Model::init(&spec, &device).unwrap();
    let losses = train(&model, &dataset, 300, 3e-3, &|epoch| {
        dataset.epoch_graphs(Some(800), epoch, Split::Train).unwrap()
    });
    let predictions = model
        .predict(
            &dataset,
            &EvalOptions {
                batch_rows: Some(800),
                ..Default::default()
            },
        )
        .unwrap();
    let (mut all, mut all_hit, mut interior, mut interior_hit) = (0, 0, 0, 0);
    for g in (0..graphs).filter(|g| g % 5 == 4) {
        for i in 0..40 {
            let node = g * 40 + i;
            let row = &predictions[node * 4..(node + 1) * 4];
            let best = (0..4).max_by(|&a, &b| row[a].total_cmp(&row[b])).unwrap();
            let hit = (best as i64 == labels[node]) as usize;
            all += 1;
            all_hit += hit;
            if (1..39).contains(&i) {
                interior += 1;
                interior_hit += hit;
            }
        }
    }
    let (overall, before) = (
        all_hit as f32 / all as f32,
        interior_hit as f32 / interior as f32,
    );
    println!(
        "long range, {direction:?}: test accuracy {overall:.3}, {before:.3} on the nodes scanned \
         before node 0; loss {}",
        curve(&losses)
    );
    (overall, before)
}

#[test]
fn long_range_needs_both_directions() {
    let (bidirectional, _) = long_range(ScanDirection::Bidirectional);
    let (_, forward_before) = long_range(ScanDirection::Forward);
    assert!(bidirectional > 0.95, "bidirectional: {bidirectional}");
    assert!(
        forward_before < 0.5,
        "forward only, nodes scanned before the coloured one: {forward_before}"
    );
}

// ---------------------------------------------------------------------------
// 3. Graph regression
// ---------------------------------------------------------------------------

/// Random 20-node graphs with constant features; the target is the triangle
/// count over its standard deviation across the dataset.
fn triangle_counts(graphs: usize) -> (GraphData, Vec<f32>) {
    let n = 20usize;
    let mut rng = Lcg(5);
    let (mut src, mut dst, mut ptr) = (Vec::new(), Vec::new(), vec![0u32]);
    let mut counts = Vec::with_capacity(graphs);
    for g in 0..graphs {
        let base = g * n;
        let mut adjacent = vec![false; n * n];
        // Densities from sparse to dense, so the counts vary.
        let density = 15 + rng.below(25);
        for a in 0..n {
            for b in a + 1..n {
                if rng.below(100) < density {
                    adjacent[a * n + b] = true;
                    adjacent[b * n + a] = true;
                    src.extend([(base + a) as u32, (base + b) as u32]);
                    dst.extend([(base + b) as u32, (base + a) as u32]);
                }
            }
        }
        let mut triangles = 0usize;
        for a in 0..n {
            for b in a + 1..n {
                for c in b + 1..n {
                    triangles +=
                        (adjacent[a * n + b] && adjacent[b * n + c] && adjacent[a * n + c]) as usize;
                }
            }
        }
        counts.push(triangles as f32);
        ptr.push((base + n) as u32);
    }
    let mean = counts.iter().sum::<f32>() / graphs as f32;
    let std = (counts.iter().map(|c| (c - mean).powi(2)).sum::<f32>() / graphs as f32).sqrt();
    let targets: Vec<f32> = counts.iter().map(|c| c / std).collect();
    let total = graphs * n;
    let mut data = GraphData::new(
        total,
        src,
        dst,
        Features::Float {
            dim: 1,
            data: vec![1.0; total],
        },
    );
    data.graph_ptr = ptr;
    data.y = Labels::Graph {
        targets: 1,
        values: targets.clone(),
    };
    data.masks = Some(Splits {
        train: (0..graphs).map(|g| g % 5 != 4).collect(),
        val: vec![false; graphs],
        test: (0..graphs).map(|g| g % 5 == 4).collect(),
    });
    (data, targets)
}

#[test]
fn triangle_counts_are_regressed_from_walk_tokens() {
    let device = dev();
    let graphs = 400usize;
    let (data, targets) = triangle_counts(graphs);
    let spec = GraphMambaSpec::new(
        FeatureSpec::Float { dim: 1 },
        GraphTaskSpec::GraphRegression {
            targets: 1,
            pool: GraphPool::Mean,
            loss: RegressionLoss::L1,
        },
    )
    .with_tokens(2, 8, 2)
    .with_local(LocalEncoder::Sgc { hops: 1 })
    .with_mpnn(Some(MpnnKind::Gine))
    .with_node_layers(2)
    .with_seed(3);
    let dataset = Dataset::new(&spec, data, &device).unwrap();
    let model = Model::init(&spec, &device).unwrap();
    let losses = train(&model, &dataset, 300, 2e-3, &|epoch| {
        dataset.epoch_graphs(Some(1024), epoch, Split::Train).unwrap()
    });
    let mae = model
        .evaluate(
            &dataset,
            Split::Test,
            Metric::Mae,
            &EvalOptions {
                batch_rows: Some(1024),
                ..Default::default()
            },
        )
        .unwrap();
    // The baseline: predicting the training mean for every held-out graph.
    let train_targets: Vec<f32> = (0..graphs).filter(|g| g % 5 != 4).map(|g| targets[g]).collect();
    let mean = train_targets.iter().sum::<f32>() / train_targets.len() as f32;
    let test_targets: Vec<f32> = (0..graphs).filter(|g| g % 5 == 4).map(|g| targets[g]).collect();
    let baseline =
        test_targets.iter().map(|t| (t - mean).abs()).sum::<f32>() / test_targets.len() as f32;
    println!(
        "triangle counts: held-out MAE {:.3} on {} graphs against {baseline:.3} for the training \
         mean (ratio {:.2}); loss {}",
        mae.value,
        mae.count,
        mae.value / baseline,
        curve(&losses)
    );
    assert_eq!(mae.count, test_targets.len());
    assert!(
        mae.value < 0.7 * baseline,
        "held-out MAE {} against a baseline of {baseline}",
        mae.value
    );
}

// ---------------------------------------------------------------------------
// 4. Multi-label
// ---------------------------------------------------------------------------

#[test]
fn multi_label_loss_decreases_with_missing_labels() {
    let device = dev();
    let graphs = 120usize;
    let (mut data, targets) = triangle_counts(graphs);
    // Three labels from the triangle count: above the median, in the top
    // quarter (missing on every other graph), and even.
    let mut sorted = targets.clone();
    sorted.sort_by(f32::total_cmp);
    let (median, quartile) = (sorted[graphs / 2], sorted[3 * graphs / 4]);
    let mut values = Vec::with_capacity(graphs * 3);
    for (g, &t) in targets.iter().enumerate() {
        values.push((t > median) as u32 as f32);
        values.push(if g % 2 == 0 {
            f32::NAN
        } else {
            (t > quartile) as u32 as f32
        });
        values.push((g % 3 == 0) as u32 as f32);
    }
    data.y = Labels::Graph { targets: 3, values };
    let spec = GraphMambaSpec::new(
        FeatureSpec::Float { dim: 1 },
        GraphTaskSpec::GraphMultiLabel {
            labels: 3,
            pool: GraphPool::Mean,
        },
    )
    .with_tokens(2, 8, 2)
    .with_node_layers(1)
    .with_seed(4);
    let dataset = Dataset::new(&spec, data, &device).unwrap();
    let model = Model::init(&spec, &device).unwrap();
    let losses = train(&model, &dataset, 80, 3e-3, &|epoch| {
        dataset.epoch_graphs(Some(1024), epoch, Split::Train).unwrap()
    });
    println!("multi-label: loss {}", curve(&losses));
    assert!(losses.iter().all(|l| l.is_finite()), "the loss stays finite");
    let first: f32 = losses[..5].iter().sum::<f32>() / 5.0;
    let last: f32 = losses[losses.len() - 5..].iter().sum::<f32>() / 5.0;
    assert!(last < first, "the loss decreases: {first} → {last}");
    let ap = model
        .evaluate(
            &dataset,
            Split::Test,
            Metric::AveragePrecision,
            &EvalOptions {
                batch_rows: Some(1024),
                ..Default::default()
            },
        )
        .unwrap();
    assert!(ap.value.is_finite() && ap.value > 0.0);
    // Half of the second label's targets are missing and are not counted.
    assert_eq!(ap.count, 24 * 2 + 12);
}
