//! GM6: datasets, epochs and batches — capacities, ordering, the automatic row
//! budget, chunked token features, interleaved node sequences and the memory
//! estimate.
//!
//! Every test holds [`LOCK`]: several read the process-wide upload and
//! allocation counters, and two set environment variables.

#![cfg(feature = "backend")]

use std::collections::HashMap;

use mamba3::autograd::Var;
use mamba3::backend::{
    Device, peak_alloc_bytes, read_count, reset_peak_alloc, reset_read_count, reset_upload_count,
    upload_count,
};
use mamba3::backends::Auto;
use mamba3::models::entity::blocks::BiBlock;
use mamba3::models::graph::{
    BatchMode, EpochOptions, EvalOptions, FeatureSpec, Features, GraphData, GraphDataset,
    GraphMamba, GraphMambaSpec, GraphPool, GraphTask, GraphTaskSpec, Labels, MAX_BYTES_ENV,
    MpnnKind, Split, Splits, TENSOR_MAX_ENV, TokenSampling, TokenTail, tensor_threshold,
};
use mamba3::nn::linear::LinearConfig;
use mamba3::nn::mlp::{Activation, MlpConfig};
use mamba3::nn::module::Module;
use mamba3::nn::norm::RmsNormConfig;
use mamba3::tensor::ops::movement::RaggedLengths;
use mamba3::tensor::ops::random::Rng;
use mamba3::tensor::ops::IGNORE;
use mamba3::tensor::Tensor;
use mamba3::train::TrainStep;

type R = Auto;
type Model = GraphMamba<R, f32>;
type Dataset = GraphDataset<R, f32>;

static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn lock() -> std::sync::MutexGuard<'static, ()> {
    LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn dev() -> Device<R> {
    Device::<R>::default()
}

/// Set an environment variable for the length of a scope. Callers hold
/// [`LOCK`], so no other test of this binary reads the environment meanwhile.
struct EnvGuard(&'static str);

impl EnvGuard {
    fn set(name: &'static str, value: usize) -> Self {
        // SAFETY: every test of this binary holds `LOCK` for its whole body,
        // so nothing reads or writes the environment concurrently.
        unsafe { std::env::set_var(name, value.to_string()) };
        Self(name)
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        // SAFETY: as in `set`.
        unsafe { std::env::remove_var(self.0) };
    }
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

    fn unit(&mut self) -> f32 {
        self.next() as f32 / (1u64 << 30) as f32 - 1.0
    }

    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.unit()).collect()
    }
}

fn assert_close(actual: &[f32], expected: &[f32], tol: f32, what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= tol * (1.0 + e.abs()),
            "{what}: index {i} got {a}, want {e}"
        );
    }
}

/// Graphs of the given sizes with about two random edges per node and `dim`
/// float features per node.
fn graphs(sizes: &[usize], dim: usize, seed: u64) -> GraphData {
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
            dim,
            data: rng.vec(at * dim),
        },
    );
    data.graph_ptr = ptr;
    data
}

fn with_node_labels(mut data: GraphData, classes: usize) -> GraphData {
    let n = data.n_nodes;
    let mut rng = Lcg(n as u64);
    data.y = Labels::Node((0..n).map(|_| rng.below(classes) as i64).collect());
    data.masks = Some(Splits {
        train: (0..n).map(|i| i % 3 != 2).collect(),
        val: (0..n).map(|i| i % 3 == 2).collect(),
        test: vec![false; n],
    });
    data
}

const NODE: GraphTaskSpec = GraphTaskSpec::NodeClass { classes: 3 };

fn spec(dim: usize) -> GraphMambaSpec {
    GraphMambaSpec::new(FeatureSpec::Float { dim }, NODE)
        .with_d_model(16)
        .with_tokens(2, 3, 2)
        .with_node_layers(1)
        .with_row_quantum(16)
}

// ---------------------------------------------------------------------------
// Whole-graph epochs
// ---------------------------------------------------------------------------

#[test]
fn batches_are_concatenations_with_fixed_capacities() {
    let _guard = lock();
    let device = dev();
    let sizes: Vec<usize> = (0..40).map(|i| 5 + (i * 7) % 23).collect();
    let s = spec(4).with_mpnn(Some(MpnnKind::GatedGcn));
    let data = with_node_labels(graphs(&sizes, 4, 3), 3);
    let dataset = Dataset::new(&s, data, &device).unwrap();
    let store = dataset.store();
    let (ptr, edge_ptr) = (store.graph_ptr().to_vec(), store.edge_ptr().to_vec());

    reset_upload_count();
    let epoch = dataset.epoch_graphs(Some(100), 0, Split::Train).unwrap();
    assert_eq!(upload_count(), 1, "the epoch table is the one upload");
    assert_eq!(epoch.mode(), BatchMode::Graphs);
    assert_eq!(epoch.rows(), 112, "the budget rounded up to the row quantum");
    assert_eq!(epoch.graphs() % 8, 0);
    let edge_cap = epoch.edges().unwrap();
    assert_eq!(edge_cap % 4096, 0);

    let mut seen = vec![0usize; sizes.len()];
    reset_upload_count();
    reset_read_count();
    let mut batches = Vec::new();
    for index in 0..epoch.len() {
        batches.push(epoch.batch(index).unwrap());
    }
    assert_eq!(upload_count(), 0, "a batch uploads nothing");
    assert_eq!(read_count(), 0, "a batch reads nothing");

    for (index, batch) in batches.iter().enumerate() {
        assert_eq!(batch.rows, 112);
        assert_eq!(batch.edges, edge_cap);
        assert_eq!(batch.graphs, epoch.graphs());
        assert_eq!(batch.lengths.max() % 32, 0);
        assert_eq!(batch.lengths.max(), epoch.padded_len(index));
        let ids = epoch.batch_graphs(index);
        let (mut gid, mut slot, mut lengths) = (Vec::new(), Vec::new(), vec![0u32; batch.graphs]);
        let mut edges = 0usize;
        for (s, &g) in ids.iter().enumerate() {
            seen[g as usize] += 1;
            lengths[s] = ptr[g as usize + 1] - ptr[g as usize];
            for node in ptr[g as usize]..ptr[g as usize + 1] {
                gid.push(node);
                slot.push(s as u32);
            }
            edges += (edge_ptr[g as usize + 1] - edge_ptr[g as usize]) as usize;
        }
        assert!(gid.len() <= 100, "filled to the budget, not past it");
        assert!(*lengths.iter().max().unwrap() as usize <= batch.lengths.max());
        assert!(batch.lengths.max() < *lengths.iter().max().unwrap() as usize + 32);
        gid.resize(112, IGNORE);
        slot.resize(112, IGNORE);
        assert_eq!(batch.layout.gid().to_vec(), gid, "rows of batch {index}");
        assert_eq!(batch.layout.row_graph().to_vec(), slot);
        assert_eq!(batch.lengths.ids().to_vec(), lengths);
        let eid = batch.layout.edges().unwrap().eid().to_vec();
        assert_eq!(eid.iter().filter(|&&e| e != IGNORE).count(), edges);
    }
    assert!(seen.iter().all(|&c| c == 1), "every graph exactly once");

    // Another epoch, another order; the same epoch, the same order.
    let order = |epoch: u64| -> Vec<u32> {
        let e = dataset.epoch_graphs(Some(100), epoch, Split::Train).unwrap();
        (0..e.len()).flat_map(|i| e.batch_graphs(i).to_vec()).collect()
    };
    assert_ne!(order(0), order(1));
    assert_eq!(order(1), order(1));
    // Evaluation is not shuffled.
    let val = |epoch: u64| -> Vec<u32> {
        let e = dataset.epoch_graphs(Some(100), epoch, Split::Val).unwrap();
        (0..e.len()).flat_map(|i| e.batch_graphs(i).to_vec()).collect()
    };
    assert_eq!(val(0), val(5));

    // A graph larger than the budget is named.
    let err = dataset
        .epoch_graphs(Some(20), 0, Split::Train)
        .err()
        .unwrap()
        .to_string();
    assert!(err.contains("graph") && err.contains("row budget"), "{err}");
    assert!(dataset.epoch_graphs(Some(0), 0, Split::Train).is_err());
    // Node partitions cut one graph, not forty.
    assert!(dataset.epoch_nodes(Some(2), 0, Split::Train).is_err());
}

#[test]
fn size_buckets_keep_the_padding_short() {
    let _guard = lock();
    let device = dev();
    let mut rng = Lcg(77);
    let sizes: Vec<usize> = (0..2000).map(|_| 10 + rng.below(391)).collect();
    let s = spec(2).with_tokens(0, 1, 1);
    let data = with_node_labels(graphs(&sizes, 2, 5), 3);
    let dataset = Dataset::new(&s, data, &device).unwrap();
    let ptr = dataset.store().graph_ptr().to_vec();
    let ratio = |bucket: bool| -> f64 {
        let epoch = dataset
            .epoch_graphs_with(
                Some(2400),
                3,
                Some(Split::Train),
                EpochOptions {
                    shuffle: true,
                    bucket,
                },
            )
            .unwrap();
        let ratios: Vec<f64> = (0..epoch.len())
            .map(|index| {
                let ids = epoch.batch_graphs(index);
                let nodes: usize = ids
                    .iter()
                    .map(|&g| (ptr[g as usize + 1] - ptr[g as usize]) as usize)
                    .sum();
                epoch.padded_len(index) as f64 / (nodes as f64 / ids.len() as f64)
            })
            .collect();
        ratios.iter().sum::<f64>() / ratios.len() as f64
    };
    let (bucketed, plain) = (ratio(true), ratio(false));
    println!("padded / mean length: {bucketed:.3} with buckets, {plain:.3} without");
    assert!(bucketed < 1.3, "padded / mean length with buckets: {bucketed}");
    assert!(plain > 1.8, "padded / mean length without buckets: {plain}");
}

#[test]
fn graph_targets_select_their_split_and_empty_splits_are_refused() {
    let _guard = lock();
    let device = dev();
    let sizes = [6usize, 9, 5, 7, 8, 4];
    let task = GraphTaskSpec::GraphClass {
        classes: 2,
        pool: GraphPool::Mean,
    };
    let s = GraphMambaSpec::new(FeatureSpec::Float { dim: 3 }, task)
        .with_d_model(16)
        .with_tokens(0, 1, 1)
        .with_row_quantum(16);
    let mut data = graphs(&sizes, 3, 9);
    data.y = Labels::GraphClass(vec![0, 1, 1, 0, 1, 0]);
    data.masks = Some(Splits {
        train: vec![true, true, false, true, false, false],
        val: vec![false, false, true, false, true, true],
        test: vec![false; 6],
    });
    let dataset = Dataset::new(&s, data, &device).unwrap();
    let ids = |split: Option<Split>| -> Vec<u32> {
        let e = dataset.epoch_graphs(Some(64), 0, split).unwrap();
        let mut ids: Vec<u32> = (0..e.len()).flat_map(|i| e.batch_graphs(i).to_vec()).collect();
        ids.sort_unstable();
        ids
    };
    assert_eq!(ids(Some(Split::Train)), vec![0, 1, 3]);
    assert_eq!(ids(Some(Split::Val)), vec![2, 4, 5]);
    assert_eq!(ids(None), vec![0, 1, 2, 3, 4, 5]);
    let err = dataset
        .epoch_graphs(Some(64), 0, Split::Test)
        .err()
        .unwrap()
        .to_string();
    assert!(err.contains("test split"), "{err}");

    // A dataset without targets can be predicted on, not trained on.
    let unlabelled = Dataset::new(&s, graphs(&sizes, 3, 9), &device).unwrap();
    assert!(!unlabelled.has_targets());
    assert!(unlabelled.epoch_graphs(Some(64), 0, None).is_ok());
    let err = unlabelled
        .epoch_graphs(Some(64), 0, Split::Train)
        .err()
        .unwrap()
        .to_string();
    assert!(err.contains("no targets"), "{err}");
    let model = Model::init(&s, &device).unwrap();
    assert_eq!(
        model
            .predict(&unlabelled, &EvalOptions::default())
            .unwrap()
            .len(),
        6 * 2
    );
}

#[test]
fn the_dataset_checks_itself_against_the_spec() {
    let _guard = lock();
    let device = dev();
    let s = spec(4);
    let good = || with_node_labels(graphs(&[20], 4, 1), 3);
    let fails = |spec: &GraphMambaSpec, data: GraphData, name: &str| {
        let text = Dataset::new(spec, data, &device).err().unwrap().to_string();
        assert!(text.contains(name), "expected `{name}` in: {text}");
    };
    Dataset::new(&s, good(), &device).unwrap();
    fails(&spec(5), good(), "x");
    fails(&s.clone().with_pe_dim(2), good(), "pe");
    let mut data = good();
    data.pe = Some((3, vec![0.0; 60]));
    fails(&s, data, "pe");
    let mut data = good();
    data.y = Labels::Node(vec![3; 20]);
    fails(&s, data, "y");
    let mut data = good();
    data.y = Labels::GraphClass(vec![0]);
    fails(&s, data, "y");
    fails(
        &s.clone()
            .with_mpnn(Some(MpnnKind::Gine))
            .with_edge_features(Some(FeatureSpec::Float { dim: 2 })),
        good(),
        "edge_attr",
    );
    // Edge features the spec does not read are not uploaded.
    let mut data = good();
    data.edge_attr = Some(Features::Float {
        dim: 1,
        data: vec![0.0; data.edge_src.len()],
    });
    assert!(Dataset::new(&s, data, &device).unwrap().store().edge_x().is_none());

    // The size guard names the limit and the remedy.
    let limit = EnvGuard::set(MAX_BYTES_ENV, 256);
    let err = Dataset::new(&s, good(), &device).err().unwrap().to_string();
    assert!(err.contains(MAX_BYTES_ENV) && err.contains("subset"), "{err}");
    drop(limit);
    // And a step that would not fit is refused when its epoch is built.
    let dataset = Dataset::new(&s, good(), &device).unwrap();
    let limit = EnvGuard::set(MAX_BYTES_ENV, dataset.store().bytes() + 1024);
    let err = dataset
        .epoch_nodes(Some(1), 0, Split::Train)
        .err()
        .unwrap()
        .to_string();
    assert!(err.contains(MAX_BYTES_ENV) && err.contains("parts"), "{err}");
    drop(limit);
}

// ---------------------------------------------------------------------------
// Node partitions
// ---------------------------------------------------------------------------

#[test]
fn node_partitions_upload_nothing() {
    let _guard = lock();
    let device = dev();
    let s = spec(4);
    let dataset = Dataset::new(&s, with_node_labels(graphs(&[103], 4, 2), 3), &device).unwrap();
    reset_upload_count();
    reset_read_count();
    let epoch = dataset.epoch_nodes(Some(4), 7, Split::Train).unwrap();
    assert_eq!(epoch.len(), 4);
    assert_eq!(epoch.rows(), 26);
    let mut seen = vec![0usize; 103];
    let mut batches = Vec::new();
    for part in 0..4 {
        batches.push(epoch.batch(part).unwrap());
    }
    assert_eq!(upload_count(), 0, "node partitions upload nothing");
    assert_eq!(read_count(), 0);
    for batch in &batches {
        assert_eq!(batch.mode, BatchMode::NodeSubset);
        assert_eq!(batch.lengths.max(), 26);
        for node in batch.layout.gid().to_vec() {
            if node != IGNORE {
                seen[node as usize] += 1;
            }
        }
    }
    assert!(seen.iter().all(|&c| c == 1));
    // The step counter is different for every batch of every epoch; the token
    // seed is the dataset's.
    let other = dataset.epoch_nodes(Some(4), 8, Split::Train).unwrap().batch(0).unwrap();
    assert_eq!(other.token_seed, batches[0].token_seed);
    assert_ne!(other.step_counter, batches[0].step_counter);
    assert_ne!(batches[1].step_counter, batches[0].step_counter);
    assert_eq!(batches[0].token_counter(TokenSampling::Static, true), 0);
    assert_eq!(batches[0].token_counter(TokenSampling::PerStep, false), 0);
    assert_eq!(
        batches[0].token_counter(TokenSampling::PerEpoch, true),
        batches[1].token_counter(TokenSampling::PerEpoch, true)
    );
    assert!(dataset.epoch_nodes(Some(0), 0, Split::Train).is_err());
    assert!(dataset.epoch_nodes(Some(104), 0, Split::Train).is_err());
    // A split nobody is in.
    let err = dataset
        .epoch_nodes(Some(4), 0, Split::Test)
        .err()
        .unwrap()
        .to_string();
    assert!(err.contains("test split"), "{err}");
}

/// Copy the model's parameters under `prefix` into `module`, by name.
fn load_from<M: Module<R, f32>>(model: &Model, prefix: &str, module: &M) {
    let source: HashMap<String, _> = model.named_parameters().into_iter().collect();
    for (name, param) in module.named_parameters() {
        param.set(source[&format!("{prefix}{name}")].value());
    }
}

#[test]
fn interleaved_node_sequences_equal_a_host_permuted_reference() {
    let _guard = lock();
    let device = dev();
    let (n, d, f, k) = (30usize, 16usize, 4usize, 4usize);
    let s = spec(f).with_tokens(0, 1, 1).with_node_sequences(k);
    let data = with_node_labels(graphs(&[n], f, 6), 3);
    let x_original = match &data.x {
        Features::Float { data, .. } => data.clone(),
        _ => unreachable!(),
    };
    let dataset = Dataset::new(&s, data, &device).unwrap();
    let model = Model::init(&s, &device).unwrap();
    model.set_training(false);
    let epoch = dataset.epoch_nodes(Some(1), 0, Split::Train).unwrap();
    // 30 rows are not a multiple of 4: two absent rows round the batch up.
    assert_eq!(epoch.rows(), 32);
    let batch = epoch.batch(0).unwrap();
    assert_eq!(batch.lengths.ids().to_vec(), vec![8, 8, 7, 7]);
    assert_eq!(batch.lengths.max(), 8);
    let task = GraphTask::new(&model);
    let loss = task.loss(&batch).unwrap();
    let got = model.forward(&batch).unwrap().to_f32();
    let got_grads = loss.backward().unwrap();

    // The reference: the sequences gathered on the host, one row of a padded
    // batch each, and the block applied to them directly.
    let perm = dataset.store().perm();
    let mut rng = Rng::seeded(1);
    let embed = LinearConfig::new(f, d).init::<R, f32>(&device, &mut rng);
    load_from(&model, "embed.", &embed);
    let block = BiBlock::<R, f32>::new(d, &s.node_ssm, s.norm_eps, 1, &device, &mut rng).unwrap();
    load_from(&model, "node.0.mixer.", &block);
    let ffn_norm = RmsNormConfig::new(d).init::<R, f32>(&device, &mut rng);
    load_from(&model, "node.0.ffn.norm.", &ffn_norm);
    let mlp = MlpConfig::new(d, 2 * d)
        .with_gated(false)
        .with_activation(Activation::Gelu)
        .with_bias(true)
        .init::<R, f32>(&device, &mut rng);
    load_from(&model, "node.0.ffn.mlp.", &mlp);
    let head_norm = RmsNormConfig::new(d).init::<R, f32>(&device, &mut rng);
    load_from(&model, "head.norm.", &head_norm);
    let head = LinearConfig::new(d, 3).init::<R, f32>(&device, &mut rng);
    load_from(&model, "head.out.", &head);

    // Sequence `q` holds rows q, q + 4, …; pad the short ones with loud rows.
    let mut sequences = vec![1.0e3f32; k * 8 * f];
    for row in 0..n {
        let (q, p) = (row % k, row / k);
        let node = perm[row] as usize;
        sequences[(q * 8 + p) * f..(q * 8 + p + 1) * f]
            .copy_from_slice(&x_original[node * f..(node + 1) * f]);
    }
    let h0 = embed
        .apply(&Var::constant(
            Tensor::<R, f32>::from_f32(&sequences, vec![k, 8, f], &device).unwrap(),
        ))
        .unwrap();
    let lengths = RaggedLengths::<R>::from_host(&[8, 8, 7, 7], 8, &device).unwrap();
    let h = h0.add(&block.branch_ragged(&h0, &lengths).unwrap()).unwrap();
    let h = h
        .add(&mlp.apply(&ffn_norm.apply(&h).unwrap()).unwrap())
        .unwrap();
    let out = head.apply(&head_norm.apply(&h).unwrap()).unwrap();
    let reference = out.to_f32();
    for row in 0..n {
        let (q, p) = (row % k, row / k);
        assert_close(
            &got[row * 3..(row + 1) * 3],
            &reference[(q * 8 + p) * 3..(q * 8 + p + 1) * 3],
            1e-5,
            &format!("row {row}"),
        );
    }

    // The same loss on the reference: every parameter's gradient agrees.
    let labels: Vec<i64> = {
        let mut rng = Lcg(n as u64);
        (0..n).map(|_| rng.below(3) as i64).collect()
    };
    let (mut ids, mut mask) = (vec![0u32; k * 8], vec![0.0f32; k * 8]);
    for row in 0..n {
        let node = perm[row] as usize;
        if node % 3 != 2 {
            ids[(row % k) * 8 + row / k] = labels[node] as u32;
            mask[(row % k) * 8 + row / k] = 1.0;
        }
    }
    let ids = mamba3::tensor::ops::index::IdTensor::<R>::from_slice(&ids, vec![k * 8], &device).unwrap();
    let mask = Tensor::<R, f32>::from_f32(&mask, vec![k * 8], &device).unwrap();
    let reference_loss = out
        .reshape(vec![k * 8, 3])
        .unwrap()
        .cross_entropy_rows(&ids, 0.0)
        .unwrap()
        .masked_mean(&mask)
        .unwrap();
    assert!((reference_loss.to_f32()[0] - loss.to_f32()[0]).abs() < 1e-5);
    let reference_grads = reference_loss.backward().unwrap();
    let model_params: HashMap<String, _> = model.named_parameters().into_iter().collect();
    let mut compared = 0;
    for (prefix, params) in [
        ("embed.", embed.named_parameters()),
        ("node.0.mixer.", block.named_parameters()),
        ("node.0.ffn.norm.", ffn_norm.named_parameters()),
        ("node.0.ffn.mlp.", mlp.named_parameters()),
        ("head.norm.", head_norm.named_parameters()),
        ("head.out.", head.named_parameters()),
    ] {
        for (name, param) in params {
            let key = format!("{prefix}{name}");
            let got = got_grads.get(model_params[&key].id()).unwrap().to_f32();
            let want = reference_grads.get(param.id()).unwrap().to_f32();
            assert_close(&got, &want, 2e-4, &format!("gradient of {key}"));
            compared += 1;
        }
    }
    assert_eq!(compared, model_params.len(), "every parameter was compared");

    // Interleaving is a property of node partitions.
    let many = Dataset::new(&s, with_node_labels(graphs(&[9, 8], f, 6), 3), &device);
    assert!(many.unwrap().epoch_graphs(Some(32), 0, Split::Train).is_err());
}

// ---------------------------------------------------------------------------
// The automatic row budget, chunked features, the memory estimate
// ---------------------------------------------------------------------------

/// One training step's loss and backward pass, returning the largest single
/// allocation it made.
fn step_peak(model: &Model, dataset: &Dataset) -> usize {
    let epoch = dataset.epoch_nodes(None, 0, Split::Train).unwrap();
    let batch = epoch.batch(0).unwrap();
    let task = GraphTask::new(model);
    reset_peak_alloc();
    drop(task.loss(&batch).unwrap().backward().unwrap());
    dataset.store().device().synchronize();
    peak_alloc_bytes()
}

#[test]
fn the_automatic_budget_keeps_every_allocation_under_the_threshold() {
    let _guard = lock();
    let device = dev();
    let threshold = 1 << 20;
    let limit = EnvGuard::set(TENSOR_MAX_ENV, threshold);
    assert_eq!(tensor_threshold(&device), threshold);
    let n = 6000usize;
    let base = GraphMambaSpec::new(FeatureSpec::Float { dim: 8 }, NODE)
        .with_d_model(64)
        .with_node_layers(1)
        .with_row_quantum(4);
    for (name, s) in [
        ("forward tail, L = 17", base.clone().with_tokens(4, 4, 4)),
        (
            "bidirectional tail, L = 17",
            base.clone()
                .with_tokens(4, 4, 4)
                .with_token_tail(TokenTail::Bidirectional),
        ),
        ("forward tail, L = 65", base.clone().with_tokens(4, 4, 16)),
        (
            "two token layers, L = 65",
            base.clone().with_tokens(4, 4, 16).with_token_layers(2),
        ),
        ("node tokens only", base.clone().with_tokens(0, 1, 1)),
    ] {
        let dataset = Dataset::new(&s, with_node_labels(graphs(&[n], 8, 4), 3), &device).unwrap();
        let rows = dataset.auto_rows();
        assert_eq!(rows % 4, 0);
        let estimate = dataset.memory_estimate(None, None);
        assert_eq!(estimate.threshold, threshold);
        assert!(estimate.largest_allocation <= threshold, "{name}");
        let model = Model::init(&s, &device).unwrap();
        let peak = step_peak(&model, &dataset);
        println!(
            "{name}: auto rows {rows}, parts {}, largest allocation {peak} of {threshold} \
             (estimated {})",
            dataset.auto_parts(),
            estimate.largest_allocation
        );
        assert!(peak <= threshold, "{name}: an allocation of {peak} bytes passed {threshold}");
        // The budget is not timid either: the largest allocation is a real
        // fraction of the threshold.
        assert!(peak * 4 >= threshold, "{name}: the largest allocation is only {peak} bytes");
    }
    drop(limit);

    // The budget follows the threshold.
    let s = base.with_tokens(4, 4, 4);
    let dataset = Dataset::new(&s, with_node_labels(graphs(&[n], 8, 4), 3), &device).unwrap();
    let small = {
        let _limit = EnvGuard::set(TENSOR_MAX_ENV, 1 << 20);
        dataset.auto_rows()
    };
    let large = {
        let _limit = EnvGuard::set(TENSOR_MAX_ENV, 4 << 20);
        dataset.auto_rows()
    };
    // Four times the threshold, four times the rows, up to the row quantum.
    assert!(large >= 4 * small && large <= 4 * (small + 4), "{small} -> {large}");
}

#[test]
fn wide_token_features_come_in_chunks() {
    let _guard = lock();
    let device = dev();
    let (n, dim) = (400usize, 316usize);
    let s = GraphMambaSpec::new(FeatureSpec::Float { dim }, NODE)
        .with_d_model(16)
        .with_tokens(2, 3, 2)
        .with_node_layers(1)
        .with_token_sampling(TokenSampling::Static)
        .with_row_quantum(16);
    let dataset = Dataset::new(&s, with_node_labels(graphs(&[n], dim, 8), 3), &device).unwrap();
    let model = Model::init(&s, &device).unwrap();
    model.set_training(false);
    let forward = |threshold: usize| {
        let _limit = EnvGuard::set(TENSOR_MAX_ENV, threshold);
        let epoch = dataset.epoch_nodes(Some(1), 0, Split::Train).unwrap();
        let batch = epoch.batch(0).unwrap();
        // 400 rows × 5 tokens × 316 columns × 4 bytes = 2.5 MB of features.
        let chunks = batch.row_chunks(5 * dim * 4);
        reset_peak_alloc();
        let out = model.forward(&batch).unwrap().to_f32();
        (out, chunks, peak_alloc_bytes())
    };
    let (whole, one, peak_whole) = forward(64 << 20);
    assert_eq!(one, vec![(0, 400)]);
    assert!(peak_whole >= 400 * 5 * dim * 4);
    let threshold = 600_000;
    let (chunked, chunks, peak) = forward(threshold);
    assert_eq!(chunks.len(), 5, "{chunks:?}");
    assert_eq!(chunks.iter().map(|c| c.1).sum::<usize>(), 400);
    assert!(chunks.iter().all(|c| c.1 * 5 * dim * 4 <= threshold));
    assert!(peak <= threshold, "a chunk of {peak} bytes passed {threshold}");
    assert_close(&chunked, &whole, 1e-6, "chunked token features");
}

#[test]
fn the_memory_estimate_is_within_reach_of_the_measurement() {
    let _guard = lock();
    let device = dev();
    let task = GraphTaskSpec::GraphClass {
        classes: 3,
        pool: GraphPool::Mean,
    };
    // The two specs of the footprint test.
    let sizes: Vec<usize> = (0..48).map(|i| 20 + (i * 11) % 30).collect();
    for (name, s, edges) in [
        (
            "node tokens with GatedGCN",
            GraphMambaSpec::new(FeatureSpec::Float { dim: 8 }, task)
                .with_d_model(64)
                .with_tokens(0, 1, 1)
                .with_mpnn(Some(MpnnKind::GatedGcn)),
            false,
        ),
        (
            "walk tokens with GINE",
            GraphMambaSpec::new(FeatureSpec::Float { dim: 8 }, task)
                .with_d_model(64)
                .with_tokens(2, 8, 2)
                .with_mpnn(Some(MpnnKind::Gine)),
            false,
        ),
    ] {
        let mut data = graphs(&sizes, 8, 5);
        let _ = edges;
        data.y = Labels::GraphClass((0..48).map(|i| i % 3).collect());
        data.masks = Some(Splits {
            train: vec![true; 48],
            val: vec![false; 48],
            test: vec![false; 48],
        });
        let dataset = Dataset::new(&s, data, &device).unwrap();
        let model = Model::init(&s, &device).unwrap();
        let rows = 1024;
        let estimate = dataset.memory_estimate(Some(rows), None);
        assert_eq!(estimate.rows, rows);
        assert_eq!(estimate.store, dataset.store().bytes());
        assert_eq!(estimate.parameters, 3 * model.num_parameters() * 4);
        let epoch = dataset.epoch_graphs(Some(rows), 0, Split::Train).unwrap();
        let batch = epoch.batch(0).unwrap();
        let in_use = || {
            device.synchronize();
            device.client().memory_usage().unwrap().bytes_in_use as usize
        };
        let task = GraphTask::new(&model);
        // Warm, so the tuner's probes are not counted.
        drop(task.loss(&batch).unwrap());
        let before = in_use();
        let loss = task.loss(&batch).unwrap();
        let live = in_use() - before;
        drop(loss);
        let ratio = estimate.live as f64 / live as f64;
        println!("{name}: live {live} bytes, estimated {} (ratio {ratio:.2})", estimate.live);
        assert!(
            (1.0 / 1.5..=1.5).contains(&ratio),
            "{name}: the estimate {} is not within a factor 1.5 of the measured {live}",
            estimate.live
        );
    }
}
