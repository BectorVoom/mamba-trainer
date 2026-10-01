//! The graph model's training step moves nothing between host and device, and
//! its launch count is pinned.
//!
//! Alone in its binary for the reason `entity_footprint.rs` gives: the launch,
//! read and upload counters are process-wide, and a test running beside this
//! one would add to them.

#![cfg(feature = "backend")]

use mamba3::backend::{
    Device, launch_count, read_count, reset_launch_count, reset_read_count, reset_upload_count,
    upload_count,
};
use mamba3::backends::Auto;
use mamba3::models::graph::{
    Epoch, FeatureSpec, Features, GraphData, GraphDataset, GraphMamba, GraphMambaSpec, GraphPool,
    GraphTask, GraphTaskSpec, GraphTrainConfig, GraphTrainer, Labels, LocalEncoder, MpnnKind,
    Split, Splits,
};
use mamba3::train::{AdamW, Trainer, TrainerConfig};

type R = Auto;

/// The counters are process-wide: every test holds this for its whole body.
static COUNT_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn lock() -> std::sync::MutexGuard<'static, ()> {
    COUNT_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

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

    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n)
            .map(|_| self.next() as f32 / (1u64 << 30) as f32 - 1.0)
            .collect()
    }
}

const F: usize = 8;

fn graphs(sizes: &[usize], seed: u64) -> GraphData {
    let mut rng = Lcg(seed);
    let (mut src, mut dst, mut ptr) = (Vec::new(), Vec::new(), vec![0u32]);
    let mut at = 0usize;
    for &size in sizes {
        for _ in 0..2 * size {
            let (a, b) = (at + rng.below(size), at + rng.below(size));
            if a != b {
                src.extend([a as u32, b as u32]);
                dst.extend([b as u32, a as u32]);
            }
        }
        at += size;
        ptr.push(at as u32);
    }
    let mut data = GraphData::new(
        at,
        src,
        dst,
        Features::Float {
            dim: F,
            data: rng.vec(at * F),
        },
    );
    data.graph_ptr = ptr;
    data
}

const GRAPH: GraphTaskSpec = GraphTaskSpec::GraphClass {
    classes: 3,
    pool: GraphPool::Mean,
};
const NODE: GraphTaskSpec = GraphTaskSpec::NodeClass { classes: 3 };

/// The two pinned specs: node tokens with GatedGCN, and walk tokens with GINE.
fn gated(task: GraphTaskSpec) -> GraphMambaSpec {
    GraphMambaSpec::new(FeatureSpec::Float { dim: F }, task)
        .with_d_model(32)
        .with_tokens(0, 1, 1)
        .with_mpnn(Some(MpnnKind::GatedGcn))
        .with_row_quantum(64)
}

fn tokens(task: GraphTaskSpec) -> GraphMambaSpec {
    GraphMambaSpec::new(FeatureSpec::Float { dim: F }, task)
        .with_d_model(32)
        .with_tokens(2, 4, 2)
        .with_local(LocalEncoder::Sgc { hops: 1 })
        .with_mpnn(Some(MpnnKind::Gine))
        .with_row_quantum(64)
}

/// Forty graphs with graph classes, all in the training split.
fn graph_dataset(spec: &GraphMambaSpec, device: &Device<R>) -> GraphDataset<R, f32> {
    let sizes: Vec<usize> = (0..40).map(|i| 12 + (i * 5) % 17).collect();
    let mut data = graphs(&sizes, 3);
    data.y = Labels::GraphClass((0..40).map(|i| i % 3).collect());
    data.masks = Some(Splits {
        train: vec![true; 40],
        val: vec![false; 40],
        test: vec![false; 40],
    });
    GraphDataset::new(spec, data, device).unwrap()
}

/// One graph of 600 nodes with node classes.
fn node_dataset(spec: &GraphMambaSpec, device: &Device<R>) -> GraphDataset<R, f32> {
    let mut data = graphs(&[600], 4);
    data.y = Labels::Node((0..600).map(|i| (i % 3) as i64).collect());
    data.masks = Some(Splits {
        train: vec![true; 600],
        val: vec![false; 600],
        test: vec![false; 600],
    });
    GraphDataset::new(spec, data, device).unwrap()
}

/// Two warm-up epochs over the same batches (so every shape has been seen),
/// then the counters of one queued step of batch 0.
fn steady_step(
    model: &GraphMamba<R, f32>,
    epoch: &Epoch<R, f32>,
) -> (usize, usize, usize) {
    let task = GraphTask::new(model);
    let mut trainer = Trainer::new(TrainerConfig::default(), AdamW::<R, f32>::new(1e-3));
    for _ in 0..2 {
        let mut queued = Vec::new();
        for index in 0..epoch.len() {
            let batch = epoch.batch(index).unwrap();
            queued.push(trainer.queue_step(&task, std::slice::from_ref(&batch)).unwrap());
        }
        trainer.read_steps(&queued).unwrap();
    }
    reset_launch_count();
    reset_read_count();
    reset_upload_count();
    let batch = epoch.batch(0).unwrap();
    let queued = trainer.queue_step(&task, std::slice::from_ref(&batch)).unwrap();
    let counts = (launch_count(), read_count(), upload_count());
    trainer.read_steps(&[queued]).unwrap();
    counts
}

#[test]
fn a_whole_graph_step_reads_and_uploads_nothing() {
    let _guard = lock();
    let device = dev();
    for (name, spec, pin) in [
        // 587 before the adjoint of the edge pre-activation masked the edge
        // rows a batch does not use: one launch per GatedGCN layer.
        ("node tokens + GatedGCN", gated(GRAPH), 589usize),
        ("walk tokens + Sgc + GINE", tokens(GRAPH), 704),
    ] {
        let dataset = graph_dataset(&spec, &device);
        let model = GraphMamba::<R, f32>::init(&spec, &device).unwrap();
        let epoch = dataset.epoch_graphs(Some(256), 0, Split::Train).unwrap();
        let (launches, reads, uploads) = steady_step(&model, &epoch);
        println!("{name}: {launches} launches per whole-graph step");
        assert_eq!(reads, 0, "{name}: a step reads nothing");
        assert_eq!(uploads, 0, "{name}: a step uploads nothing");
        // Launch counts are a property of the backend's dispatch: pinned on the
        // CPU runtime only, where the scan is the composed one.
        if device.name() == "cpu" {
            assert_eq!(launches, pin, "{name}: launches per step");
        }
    }
}

#[test]
fn a_node_partition_step_reads_and_uploads_nothing() {
    let _guard = lock();
    let device = dev();
    for (name, spec, pin) in [
        (
            "node tokens + GINE",
            gated(NODE).with_mpnn(Some(MpnnKind::Gine)),
            529usize,
        ),
        ("walk tokens + Sgc + GINE", tokens(NODE), 728),
        (
            "walk tokens, four node sequences",
            tokens(NODE).with_node_sequences(4),
            736,
        ),
    ] {
        let dataset = node_dataset(&spec, &device);
        let model = GraphMamba::<R, f32>::init(&spec, &device).unwrap();
        let epoch = dataset.epoch_nodes(Some(3), 0, Split::Train).unwrap();
        let (launches, reads, uploads) = steady_step(&model, &epoch);
        println!("{name}: {launches} launches per node-partition step");
        assert_eq!(reads, 0, "{name}: a step reads nothing");
        assert_eq!(uploads, 0, "{name}: a step uploads nothing");
        if device.name() == "cpu" {
            assert_eq!(launches, pin, "{name}: launches per step");
        }
    }
}

#[test]
fn an_epoch_of_ten_batches_uploads_once_and_reads_once() {
    let _guard = lock();
    let device = dev();
    let spec = tokens(GRAPH);
    // Forty graphs of twenty nodes at eighty rows a batch: ten batches.
    let mut data = graphs(&[20; 40], 6);
    data.y = Labels::GraphClass((0..40).map(|i| i % 3).collect());
    data.masks = Some(Splits {
        train: vec![true; 40],
        val: vec![false; 40],
        test: vec![false; 40],
    });
    let dataset = GraphDataset::new(&spec, data, &device).unwrap();
    let model = GraphMamba::<R, f32>::init(&spec, &device).unwrap();
    let mut trainer = GraphTrainer::new(&GraphTrainConfig::default()).unwrap();
    // Two warm-up epochs, then the counted one, with ten batches each.
    for epoch in 0..2 {
        let epoch = dataset.epoch_graphs(Some(80), epoch, Split::Train).unwrap();
        assert_eq!(epoch.len(), 10);
        model.train_epoch(&mut trainer, &epoch).unwrap();
        trainer.read_losses().unwrap();
    }
    reset_read_count();
    reset_upload_count();
    let epoch = dataset.epoch_graphs(Some(80), 2, Split::Train).unwrap();
    assert_eq!(model.train_epoch(&mut trainer, &epoch).unwrap(), 10);
    assert_eq!(upload_count(), 1, "the epoch table is the only upload");
    assert_eq!(read_count(), 0, "queueing an epoch reads nothing");
    let losses = trainer.read_losses().unwrap();
    assert_eq!(losses.len(), 10);
    assert_eq!(read_count(), 1, "every loss of the epoch in one read");
    assert!(losses.iter().all(|info| info.loss.is_finite()));
    assert_eq!(losses.last().unwrap().step, 30);
}

#[test]
fn prediction_and_evaluation_read_once() {
    use mamba3::models::graph::{EvalOptions, Metric};
    let _guard = lock();
    let device = dev();
    // Whole graphs: five batches.
    let spec = tokens(GRAPH);
    let mut data = graphs(&[20; 40], 6);
    data.y = Labels::GraphClass((0..40).map(|i| i % 3).collect());
    data.masks = Some(Splits {
        train: (0..40).map(|i| i % 2 == 0).collect(),
        val: (0..40).map(|i| i % 2 == 1).collect(),
        test: vec![false; 40],
    });
    let dataset = GraphDataset::new(&spec, data, &device).unwrap();
    let model = GraphMamba::<R, f32>::init(&spec, &device).unwrap();
    let options = EvalOptions {
        batch_rows: Some(160),
        ..Default::default()
    };
    model.predict(&dataset, &options).unwrap();
    reset_read_count();
    reset_upload_count();
    let predictions = model.predict(&dataset, &options).unwrap();
    assert_eq!(predictions.len(), 40 * 3);
    assert_eq!(read_count(), 1, "predict over five batches is one read");
    assert_eq!(upload_count(), 1, "and one upload: the epoch table");
    reset_read_count();
    let accuracy = model
        .evaluate(&dataset, Split::Val, Metric::Accuracy, &options)
        .unwrap();
    assert_eq!(accuracy.count, 20);
    assert_eq!(read_count(), 1, "evaluate is one read");

    // Node partitions: the row ids come back in the same read.
    let spec = tokens(NODE);
    let dataset = node_dataset(&spec, &device);
    let model = GraphMamba::<R, f32>::init(&spec, &device).unwrap();
    let options = EvalOptions {
        parts: Some(5),
        ..Default::default()
    };
    model.predict(&dataset, &options).unwrap();
    reset_read_count();
    reset_upload_count();
    assert_eq!(model.predict(&dataset, &options).unwrap().len(), 600 * 3);
    assert_eq!(read_count(), 1, "predict over five parts is one read");
    assert_eq!(upload_count(), 0, "node partitions upload nothing");
    reset_read_count();
    model
        .evaluate(&dataset, Split::Train, Metric::F1Macro, &options)
        .unwrap();
    assert_eq!(read_count(), 1);
}
