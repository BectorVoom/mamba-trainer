//! GM7: the Graph Mamba model — shapes, structure, invariances, gradients,
//! persistence, metrics and the golden file shared with the Python bindings.
//!
//! Every test holds [`LOCK`]: one of them reads the process-wide launch tally.

#![cfg(feature = "backend")]

use std::collections::HashMap;

use mamba3::autograd::Var;
use mamba3::backend::{Device, launch_tally_detailed, start_launch_tally, stop_launch_tally};
use mamba3::backends::Auto;
use mamba3::models::entity::blocks::BiBlock;
use mamba3::models::graph::metrics::{accuracy, average_precision, f1_macro, roc_auc, weighted_mean};
use mamba3::models::graph::{
    EvalOptions, FeatureSpec, Features, GraphData, GraphDataset, GraphMamba, GraphMambaSpec,
    GraphPool, GraphTask, GraphTaskSpec, GraphTrainConfig, GraphTrainer, Labels, LocalEncoder,
    Metric, MpnnKind, NodeOrder, RegressionLoss, Split, Splits, TokenSampling, TokenTail,
    graph_loss,
};
use mamba3::models::vision::ScanDirection;
use mamba3::nn::linear::LinearConfig;
use mamba3::nn::mlp::{Activation, MlpConfig};
use mamba3::nn::module::Module;
use mamba3::nn::norm::RmsNormConfig;
use mamba3::tensor::ops::random::Rng;
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

const F: usize = 6;

/// Graphs of the given sizes with about two random edges per node, float node
/// features, and (optionally) three float edge features.
fn graphs(sizes: &[usize], edge_features: bool, seed: u64) -> GraphData {
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
    if edge_features {
        data.edge_attr = Some(Features::Float {
            dim: 3,
            data: rng.vec(data.edge_src.len() * 3),
        });
    }
    data
}

/// Node labels in `0..classes` with a train / val / test split by thirds.
fn with_node_labels(mut data: GraphData, classes: usize, seed: u64) -> GraphData {
    let n = data.n_nodes;
    let mut rng = Lcg(seed);
    data.y = Labels::Node((0..n).map(|_| rng.below(classes) as i64).collect());
    data.masks = Some(Splits {
        train: (0..n).map(|i| i % 3 == 0).collect(),
        val: (0..n).map(|i| i % 3 == 1).collect(),
        test: (0..n).map(|i| i % 3 == 2).collect(),
    });
    data
}

fn all_train(items: usize) -> Splits {
    Splits {
        train: vec![true; items],
        val: vec![false; items],
        test: vec![false; items],
    }
}

/// A small spec: width 16, tokens of two hops, one node layer.
fn spec(task: GraphTaskSpec) -> GraphMambaSpec {
    GraphMambaSpec::new(FeatureSpec::Float { dim: F }, task)
        .with_d_model(16)
        .with_tokens(2, 3, 2)
        .with_node_layers(1)
        .with_row_quantum(16)
        .with_seed(3)
}

const NODE: GraphTaskSpec = GraphTaskSpec::NodeClass { classes: 4 };

// ---------------------------------------------------------------------------
// Shapes and structure
// ---------------------------------------------------------------------------

#[test]
fn output_shapes_of_every_task() {
    let _guard = lock();
    let device = dev();
    let sizes = [7usize, 12, 5, 9, 6];
    let graph_count = sizes.len();

    // Node classification, whole graphs and node parts.
    let s = spec(NODE);
    let data = with_node_labels(graphs(&sizes, false, 1), 4, 2);
    let dataset = Dataset::new(&s, data, &device).unwrap();
    let model = Model::init(&s, &device).unwrap();
    let epoch = dataset.epoch_graphs(Some(32), 0, Split::Train).unwrap();
    assert!(epoch.len() >= 2);
    let batch = epoch.batch(0).unwrap();
    assert_eq!(model.forward(&batch).unwrap().dims(), &[32, 4]);
    let single = with_node_labels(graphs(&[40], false, 1), 4, 2);
    let dataset = Dataset::new(&s, single, &device).unwrap();
    let epoch = dataset.epoch_nodes(Some(3), 0, Split::Train).unwrap();
    assert_eq!(epoch.len(), 3);
    assert_eq!(
        model.forward(&epoch.batch(1).unwrap()).unwrap().dims(),
        &[14, 4]
    );

    // Graph tasks: one row per graph slot.
    for (task, labels) in [
        (
            GraphTaskSpec::GraphClass {
                classes: 3,
                pool: GraphPool::Mean,
            },
            Labels::GraphClass(vec![0, 1, 2, 1, 0]),
        ),
        (
            GraphTaskSpec::GraphRegression {
                targets: 2,
                pool: GraphPool::Sum,
                loss: RegressionLoss::L1,
            },
            Labels::Graph {
                targets: 2,
                values: Lcg(4).vec(graph_count * 2),
            },
        ),
        (
            GraphTaskSpec::GraphMultiLabel {
                labels: 5,
                pool: GraphPool::Mean,
            },
            Labels::Graph {
                targets: 5,
                values: (0..graph_count * 5).map(|i| (i % 2) as f32).collect(),
            },
        ),
    ] {
        let s = spec(task);
        let mut data = graphs(&sizes, false, 1);
        data.y = labels;
        data.masks = Some(all_train(graph_count));
        let dataset = Dataset::new(&s, data, &device).unwrap();
        let model = Model::init(&s, &device).unwrap();
        let epoch = dataset.epoch_graphs(Some(64), 0, Split::Train).unwrap();
        let batch = epoch.batch(0).unwrap();
        let out = model.forward(&batch).unwrap();
        assert_eq!(out.dims(), &[8, task.outputs()], "{task:?}");
        let loss = graph_loss(&task, &out, &batch).unwrap();
        assert_eq!(loss.dims(), &[1]);
        assert!(loss.to_f32()[0].is_finite(), "{task:?}");
    }
}

#[test]
fn parameter_names_and_count_follow_the_spec() {
    let _guard = lock();
    let device = dev();
    let specs = [
        spec(NODE),
        spec(NODE).with_tokens(0, 1, 1),
        spec(NODE)
            .with_token_layers(2)
            .with_token_tail(TokenTail::Bidirectional)
            .with_mpnn(Some(MpnnKind::Gine))
            .with_node_layers(2)
            .with_pe_dim(4),
        spec(GraphTaskSpec::GraphClass {
            classes: 3,
            pool: GraphPool::Mean,
        })
        .with_mpnn(Some(MpnnKind::GatedGcn))
        .with_node_layers(3)
        .with_edge_features(Some(FeatureSpec::Float { dim: 3 })),
        spec(GraphTaskSpec::GraphRegression {
            targets: 2,
            pool: GraphPool::Sum,
            loss: RegressionLoss::Mse,
        })
        .with_mpnn(Some(MpnnKind::GatedGcn))
        .with_node_layers(2)
        .with_direction(ScanDirection::Forward)
        .with_node_heads(2)
        .with_d_state(4),
        GraphMambaSpec::new(
            FeatureSpec::Categorical { vocab: vec![5, 3] },
            NODE,
        )
        .with_d_model(32)
        .with_mpnn(Some(MpnnKind::Gine))
        .with_edge_features(Some(FeatureSpec::Categorical { vocab: vec![4] })),
    ];
    for s in specs {
        let model = Model::init(&s, &device).unwrap();
        assert_eq!(model.num_parameters(), s.parameter_count(), "{s:?}");
        let names: Vec<String> = model.named_parameters().into_iter().map(|p| p.0).collect();
        for name in &names {
            let known = ["embed.", "local.", "token.", "edge.", "node.", "head."]
                .iter()
                .any(|prefix| name.starts_with(prefix));
            assert!(known, "unexpected parameter {name}");
        }
        assert!(names.iter().any(|n| n == "embed.weight"));
        assert!(names.iter().any(|n| n == "node.0.mixer.mixer.in_proj.weight"));
        assert!(names.iter().any(|n| n == "node.0.ffn.mlp.up.weight"));
        assert!(names.iter().any(|n| n == "head.out.weight"));
        assert_eq!(names.iter().any(|n| n.starts_with("local.")), s.has_token_stage());
        assert_eq!(
            names.iter().any(|n| n.starts_with("node.0.mpnn.")),
            s.mpnn.is_some()
        );
        assert_eq!(
            names.iter().any(|n| n.starts_with("edge.")),
            s.mpnn == Some(MpnnKind::GatedGcn)
        );
    }
}

/// Copy the model's parameters under `prefix` into `module`, by name.
fn load_from<M: Module<R, f32>>(model: &Model, prefix: &str, module: &M) {
    let source: HashMap<String, _> = model.named_parameters().into_iter().collect();
    for (name, param) in module.named_parameters() {
        let key = format!("{prefix}{name}");
        let value = source
            .get(&key)
            .unwrap_or_else(|| panic!("the model has no parameter {key}"))
            .value();
        param.set(value);
    }
}

#[test]
fn node_tokens_only_is_embed_block_ffn_head() {
    let _guard = lock();
    let device = dev();
    let s = spec(NODE).with_tokens(0, 1, 1);
    let data = with_node_labels(graphs(&[30], false, 5), 4, 6);
    let x_original = match &data.x {
        Features::Float { data, .. } => data.clone(),
        _ => unreachable!(),
    };
    let dataset = Dataset::new(&s, data, &device).unwrap();
    let model = Model::init(&s, &device).unwrap();
    model.set_training(false);
    let epoch = dataset.epoch_nodes(Some(1), 0, Split::Train).unwrap();
    let got = model.forward(&epoch.batch(0).unwrap()).unwrap().to_f32();

    // The same computation written out, on the degree-sorted sequence.
    let (n, d) = (30usize, 16usize);
    let perm = dataset.store().perm();
    let mut x = Vec::with_capacity(n * F);
    for &original in perm {
        x.extend_from_slice(&x_original[original as usize * F..(original as usize + 1) * F]);
    }
    let mut rng = Rng::seeded(99);
    let embed = LinearConfig::new(F, d).init::<R, f32>(&device, &mut rng);
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
    let head = LinearConfig::new(d, 4).init::<R, f32>(&device, &mut rng);
    load_from(&model, "head.out.", &head);

    let h0 = embed
        .apply(&Var::constant(
            Tensor::<R, f32>::from_f32(&x, vec![n, F], &device).unwrap(),
        ))
        .unwrap();
    let branch = block
        .branch(&h0.reshape(vec![1, n, d]).unwrap())
        .unwrap()
        .reshape(vec![n, d])
        .unwrap();
    let h = h0.add(&branch).unwrap();
    let h = h
        .add(&mlp.apply(&ffn_norm.apply(&h).unwrap()).unwrap())
        .unwrap();
    let want = head.apply(&head_norm.apply(&h).unwrap()).unwrap().to_f32();
    assert_close(&got, &want, 1e-5, "the composition written out");
}

// ---------------------------------------------------------------------------
// Invariances
// ---------------------------------------------------------------------------

#[test]
fn a_graph_does_not_depend_on_its_batch() {
    let _guard = lock();
    let device = dev();
    let sizes = [7usize, 12, 5, 9, 33];
    for s in [
        spec(NODE).with_token_sampling(TokenSampling::Static),
        spec(NODE)
            .with_token_sampling(TokenSampling::Static)
            .with_mpnn(Some(MpnnKind::Gine))
            .with_node_layers(2),
        spec(NODE)
            .with_tokens(0, 1, 1)
            .with_mpnn(Some(MpnnKind::GatedGcn))
            .with_node_layers(2),
    ] {
        let data = with_node_labels(graphs(&sizes, false, 7), 4, 8);
        let dataset = Dataset::new(&s, data, &device).unwrap();
        let model = Model::init(&s, &device).unwrap();
        // Graph 1 (12 nodes): alone; with two others; later in a batch that
        // also holds the long graph (another padded length); at another row
        // budget (other absent rows).
        let epoch_of = |batches: Vec<Vec<u32>>, rows: usize| {
            dataset
                .epoch_from_batches(batches, rows, 0, Some(Split::Train))
                .unwrap()
        };
        let rows_of = |batches: Vec<Vec<u32>>, rows: usize, skip: usize| -> Vec<f32> {
            let epoch = epoch_of(batches, rows);
            let out = model.forward(&epoch.batch(0).unwrap()).unwrap().to_f32();
            out[skip * 4..(skip + 12) * 4].to_vec()
        };
        let alone = rows_of(vec![vec![1]], 16, 0);
        assert_close(&rows_of(vec![vec![1, 0, 2]], 32, 0), &alone, 1e-4, "with two others");
        assert_close(&rows_of(vec![vec![2, 1, 3]], 32, 5), &alone, 1e-4, "second in its batch");
        assert_close(&rows_of(vec![vec![4, 1]], 48, 33), &alone, 1e-4, "after a longer graph");
        assert_close(&rows_of(vec![vec![1]], 64, 0), &alone, 1e-4, "another row budget");
    }
}

/// A graph whose degrees are all different: `i ~ j` iff `i + j >= n`, with
/// self-loops on the upper half.
fn distinct_degrees(n: usize) -> (Vec<u32>, Vec<u32>) {
    let (mut src, mut dst) = (Vec::new(), Vec::new());
    for i in 0..n {
        for j in 0..n {
            if i + j >= n && (i != j || i >= n / 2) {
                src.push(i as u32);
                dst.push(j as u32);
            }
        }
    }
    (src, dst)
}

#[test]
fn renumbering_the_nodes_does_not_change_their_predictions() {
    let _guard = lock();
    let device = dev();
    let n = 12usize;
    let (src, dst) = distinct_degrees(n);
    let features = Lcg(21).vec(n * F);
    let s = spec(NODE).with_tokens(0, 1, 1).with_node_layers(2);
    let model = Model::init(&s, &device).unwrap();
    let options = EvalOptions {
        parts: Some(1),
        ..Default::default()
    };
    let predict = |src: Vec<u32>, dst: Vec<u32>, x: Vec<f32>| {
        let data = GraphData::new(n, src, dst, Features::Float { dim: F, data: x });
        let dataset = Dataset::new(&s, data, &device).unwrap();
        // Every degree is different, so the order is the graph's own.
        let degrees: Vec<usize> = (0..n).map(|i| dataset.store().graph_len(0).min(i)).collect();
        assert_eq!(degrees.len(), n);
        model.predict(&dataset, &options).unwrap()
    };
    let reference = predict(src.clone(), dst.clone(), features.clone());
    assert_eq!(reference.len(), n * 4);

    // A random renumbering: node `i` becomes node `pi[i]`.
    let mut pi: Vec<usize> = (0..n).collect();
    let mut rng = Lcg(5);
    for i in (1..n).rev() {
        pi.swap(i, rng.below(i + 1));
    }
    let map = |ids: &[u32]| ids.iter().map(|&i| pi[i as usize] as u32).collect::<Vec<_>>();
    let mut x = vec![0.0f32; n * F];
    for i in 0..n {
        x[pi[i] * F..(pi[i] + 1) * F].copy_from_slice(&features[i * F..(i + 1) * F]);
    }
    let renumbered = predict(map(&src), map(&dst), x);
    for i in 0..n {
        assert_close(
            &renumbered[pi[i] * 4..(pi[i] + 1) * 4],
            &reference[i * 4..(i + 1) * 4],
            1e-5,
            &format!("node {i}"),
        );
    }
}

#[test]
fn forward_only_node_blocks_change_the_output() {
    let _guard = lock();
    let device = dev();
    let data = with_node_labels(graphs(&[25], false, 9), 4, 9);
    let bi = spec(NODE).with_tokens(0, 1, 1);
    let fwd = bi.clone().with_direction(ScanDirection::Forward);
    let (model_bi, model_fwd) = (
        Model::init(&bi, &device).unwrap(),
        Model::init(&fwd, &device).unwrap(),
    );
    // The fused bidirectional mixer has both directions' heads.
    let width = |model: &Model| {
        model
            .named_parameters()
            .into_iter()
            .find(|(name, _)| name == "node.0.mixer.mixer.in_proj.weight")
            .unwrap()
            .1
            .shape()
            .dim(1)
    };
    assert_eq!(width(&model_bi), 2 * width(&model_fwd));
    let out = |s: &GraphMambaSpec, model: &Model| {
        let dataset = Dataset::new(s, data.clone(), &device).unwrap();
        model
            .predict(
                &dataset,
                &EvalOptions {
                    parts: Some(1),
                    ..Default::default()
                },
            )
            .unwrap()
    };
    assert_ne!(out(&bi, &model_bi), out(&fwd, &model_fwd));
}

#[test]
fn a_forward_tail_launches_no_reversal() {
    let _guard = lock();
    let device = dev();
    let data = with_node_labels(graphs(&[25], false, 9), 4, 9);
    let reversals = |s: GraphMambaSpec| {
        let dataset = Dataset::new(&s, data.clone(), &device).unwrap();
        let model = Model::init(&s, &device).unwrap();
        let batch = dataset
            .epoch_nodes(Some(1), 0, Split::Train)
            .unwrap()
            .batch(0)
            .unwrap();
        model.forward(&batch).unwrap();
        start_launch_tally();
        model.forward(&batch).unwrap();
        stop_launch_tally();
        launch_tally_detailed()
            .into_iter()
            .filter(|row| row.op.starts_with("reverse_bands"))
            .map(|row| row.count)
            .sum::<usize>()
    };
    // One forward token layer and a forward node stage: nothing is reversed.
    let forward = spec(NODE).with_direction(ScanDirection::Forward);
    assert_eq!(reversals(forward.clone()), 0);
    // The paper-literal tail reverses on the way in and on the way out.
    assert_eq!(
        reversals(forward.with_token_tail(TokenTail::Bidirectional)),
        2
    );
    // A full row count makes the ragged reversal the plain one.
    assert_eq!(reversals(spec(NODE)), 2);
}

// ---------------------------------------------------------------------------
// Gradients
// ---------------------------------------------------------------------------

#[test]
fn gradients_reach_every_parameter() {
    let _guard = lock();
    let device = dev();
    let sizes = [7usize, 12, 5, 9];
    let task = GraphTaskSpec::GraphClass {
        classes: 3,
        pool: GraphPool::Mean,
    };
    for local in [LocalEncoder::Mean, LocalEncoder::Sgc { hops: 1 }] {
        for mpnn in [None, Some(MpnnKind::Gine), Some(MpnnKind::GatedGcn)] {
            for tail in [TokenTail::Forward, TokenTail::Bidirectional] {
                for edges in [false, true] {
                    if edges && mpnn.is_none() {
                        continue;
                    }
                    let s = spec(task)
                        .with_local(local)
                        .with_mpnn(mpnn)
                        .with_token_tail(tail)
                        .with_node_layers(2)
                        .with_pe_dim(2)
                        .with_edge_features(edges.then_some(FeatureSpec::Float { dim: 3 }));
                    let mut data = graphs(&sizes, edges, 11);
                    data.pe = Some((2, Lcg(12).vec(data.n_nodes * 2)));
                    data.y = Labels::GraphClass(vec![0, 1, 2, 1]);
                    data.masks = Some(all_train(4));
                    let dataset = Dataset::new(&s, data, &device).unwrap();
                    let model = Model::init(&s, &device).unwrap();
                    let batch = dataset
                        .epoch_graphs(Some(48), 0, Split::Train)
                        .unwrap()
                        .batch(0)
                        .unwrap();
                    let task = GraphTask::new(&model);
                    let grads = task.loss(&batch).unwrap().backward().unwrap();
                    let what = format!("{local:?} {mpnn:?} {tail:?} edges={edges}");
                    for (name, param) in model.named_parameters() {
                        let grad = grads
                            .get(param.id())
                            .unwrap_or_else(|| panic!("{what}: {name} has no gradient"))
                            .to_f32();
                        assert!(
                            grad.iter().all(|g| g.is_finite()),
                            "{what}: {name} has a non-finite gradient"
                        );
                        assert!(
                            grad.iter().any(|&g| g != 0.0),
                            "{what}: {name} has an all-zero gradient"
                        );
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Training mode, persistence
// ---------------------------------------------------------------------------

#[test]
fn evaluation_switches_dropout_off_and_restores_the_mode() {
    let _guard = lock();
    let device = dev();
    let s = spec(NODE).with_dropout(0.5);
    let data = with_node_labels(graphs(&[60], false, 13), 4, 14);
    let dataset = Dataset::new(&s, data, &device).unwrap();
    let model = Model::init(&s, &device).unwrap();
    let mut trainer = GraphTrainer::new(&GraphTrainConfig::default()).unwrap();
    let epoch = dataset.epoch_nodes(Some(2), 0, Split::Train).unwrap();
    assert_eq!(model.train_epoch(&mut trainer, &epoch).unwrap(), 2);
    let infos = trainer.read_losses().unwrap();
    assert_eq!(infos.len(), 2);
    assert!(infos.iter().all(|i| i.loss.is_finite() && i.grad_norm > 0.0));
    assert_eq!(infos[1].step, 2);
    assert!(model.is_training());

    let options = EvalOptions {
        parts: Some(2),
        ..Default::default()
    };
    // With dropout on, two passes would differ.
    let a = model
        .evaluate(&dataset, Split::Val, Metric::Accuracy, &options)
        .unwrap();
    let b = model
        .evaluate(&dataset, Split::Val, Metric::Accuracy, &options)
        .unwrap();
    assert_eq!(a, b);
    assert_eq!(a.count, 20);
    assert_eq!(
        model.predict(&dataset, &options).unwrap(),
        model.predict(&dataset, &options).unwrap()
    );
    assert!(model.is_training(), "the mode is restored");

    // Accuracy equals the accuracy of the predictions.
    let predictions = model.predict(&dataset, &options).unwrap();
    let labels: Vec<usize> = {
        let mut rng = Lcg(14);
        (0..60).map(|_| rng.below(4)).collect()
    };
    let correct = (0..60)
        .filter(|i| i % 3 == 1)
        .filter(|&i| {
            let row = &predictions[i * 4..(i + 1) * 4];
            let best = (0..4).max_by(|&a, &b| row[a].total_cmp(&row[b])).unwrap();
            best == labels[i]
        })
        .count();
    assert!((a.value - correct as f32 / 20.0).abs() < 1e-6);

    // A metric that does not fit the task is refused by name.
    let err = model
        .evaluate(&dataset, Split::Val, Metric::Mae, &options)
        .unwrap_err()
        .to_string();
    assert!(err.contains("metric"), "{err}");
    assert!(Metric::parse("roc_auc").is_ok());
    assert!(Metric::parse("auc").unwrap_err().to_string().contains("metric"));
}

#[test]
fn save_and_load_give_identical_predictions() {
    let _guard = lock();
    let device = dev();
    let s = spec(NODE).with_mpnn(Some(MpnnKind::Gine)).with_node_layers(2);
    let data = with_node_labels(graphs(&[9, 14, 6], false, 15), 4, 16);
    let dataset = Dataset::new(&s, data, &device).unwrap();
    let model = Model::init(&s, &device).unwrap();
    let mut trainer = GraphTrainer::new(&GraphTrainConfig::default()).unwrap();
    let epoch = dataset.epoch_graphs(Some(32), 0, Split::Train).unwrap();
    model.train_epoch(&mut trainer, &epoch).unwrap();
    trainer.read_losses().unwrap();

    let dir = std::env::temp_dir().join(format!("mamba3_graph_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("model.m3ck");
    model.save(&path, trainer.step_count()).unwrap();
    let loaded = Model::load(&path, &device).unwrap();
    std::fs::remove_dir_all(&dir).ok();
    assert_eq!(loaded.spec(), model.spec());
    let options = EvalOptions {
        batch_rows: Some(32),
        ..Default::default()
    };
    assert_eq!(
        loaded.predict(&dataset, &options).unwrap(),
        model.predict(&dataset, &options).unwrap()
    );
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

#[test]
fn metrics_match_hand_computed_values() {
    // Rows are true classes, columns predictions.
    let confusion = [3u64, 1, 0, 0, 2, 2, 1, 0, 1];
    assert!((accuracy(&confusion, 3) - 6.0 / 10.0).abs() < 1e-6);
    // F1: class 0: 2·3 / (4 + 4) = 0.75; class 1: 2·2 / (4 + 3) = 4/7;
    // class 2: 2·1 / (2 + 3) = 0.4.
    let want = (0.75 + 4.0 / 7.0 + 0.4) / 3.0;
    assert!((f1_macro(&confusion, 3) - want).abs() < 1e-6);
    assert_eq!(accuracy(&[0; 4], 2), 0.0);
    assert_eq!(f1_macro(&[0; 4], 2), 0.0);
    assert!((weighted_mean(&[(1.0, 2.0), (4.0, 1.0), (9.0, 0.0)]) - 2.0).abs() < 1e-6);
    assert_eq!(weighted_mean(&[]), 0.0);

    // ROC AUC with ties: scores 0.1, 0.4, 0.4, 0.8 and labels 0, 0, 1, 1. Of
    // the four positive–negative pairs three are ordered and one is tied.
    let scores = [0.1f32, 0.4, 0.4, 0.8];
    let labels = [0.0f32, 0.0, 1.0, 1.0];
    let all = [1.0f32; 4];
    assert!((roc_auc(&scores, &labels, &all, 1).unwrap() - 3.5 / 4.0).abs() < 1e-6);
    // Average precision at distinct thresholds: at 0.8 recall 1/2 with
    // precision 1; at 0.4 recall 1 with precision 2/3 (the tie enters whole).
    let ap = 0.5 * 1.0 + 0.5 * (2.0 / 3.0);
    assert!((average_precision(&scores, &labels, &all, 1).unwrap() - ap).abs() < 1e-6);
    // A masked row does not count.
    let masked = [1.0f32, 0.0, 1.0, 1.0];
    assert!((roc_auc(&scores, &labels, &masked, 1).unwrap() - 1.0).abs() < 1e-6);

    // Two label columns, the second with a single class: it is skipped, so
    // the mean is the first column's value.
    let scores2 = [0.1f32, 0.9, 0.4, 0.2, 0.4, 0.7, 0.8, 0.3];
    let labels2 = [0.0f32, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let all2 = [1.0f32; 8];
    assert!((average_precision(&scores2, &labels2, &all2, 2).unwrap() - ap).abs() < 1e-6);
    assert!((roc_auc(&scores2, &labels2, &all2, 2).unwrap() - 3.5 / 4.0).abs() < 1e-6);
    // No column with both classes: undefined.
    assert!(average_precision(&[0.3, 0.6], &[1.0, 1.0], &[1.0, 1.0], 1).is_none());
    assert!(roc_auc(&[0.3, 0.6], &[0.0, 0.0], &[1.0, 1.0], 1).is_none());
}

#[test]
fn rank_and_error_metrics_read_the_right_targets() {
    let _guard = lock();
    let device = dev();
    let sizes = [7usize, 12, 5, 9, 6, 8];
    let masks = Splits {
        train: vec![true, true, false, false, false, false],
        val: vec![false, false, true, true, true, true],
        test: vec![false; 6],
    };
    let options = EvalOptions {
        batch_rows: Some(32),
        ..Default::default()
    };

    // Regression: MAE over the real targets of the split, from predict.
    let task = GraphTaskSpec::GraphRegression {
        targets: 2,
        pool: GraphPool::Mean,
        loss: RegressionLoss::L1,
    };
    let s = spec(task);
    let mut values = Lcg(31).vec(12);
    values[5] = f32::NAN;
    let mut data = graphs(&sizes, false, 30);
    data.y = Labels::Graph {
        targets: 2,
        values: values.clone(),
    };
    data.masks = Some(masks.clone());
    let dataset = Dataset::new(&s, data, &device).unwrap();
    let model = Model::init(&s, &device).unwrap();
    let predictions = model.predict(&dataset, &options).unwrap();
    assert_eq!(predictions.len(), 12);
    let (mut abs, mut sq, mut count) = (0.0f64, 0.0f64, 0usize);
    for g in 2..6 {
        for t in 0..2 {
            let target = values[g * 2 + t];
            if !target.is_nan() {
                let error = (predictions[g * 2 + t] - target) as f64;
                abs += error.abs();
                sq += error * error;
                count += 1;
            }
        }
    }
    let mae = model
        .evaluate(&dataset, Split::Val, Metric::Mae, &options)
        .unwrap();
    assert_eq!(mae.count, count);
    assert_eq!(count, 7);
    assert!((mae.value as f64 - abs / count as f64).abs() < 1e-5);
    let mse = model
        .evaluate(&dataset, Split::Val, Metric::Mse, &options)
        .unwrap();
    assert!((mse.value as f64 - sq / count as f64).abs() < 1e-5);

    // Multi-label: AP and AUC of the split, from predict.
    let task = GraphTaskSpec::GraphMultiLabel {
        labels: 2,
        pool: GraphPool::Mean,
    };
    let s = spec(task);
    let labels = vec![0.0f32, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0];
    let mut data = graphs(&sizes, false, 30);
    data.y = Labels::Graph {
        targets: 2,
        values: labels.clone(),
    };
    data.masks = Some(masks);
    let dataset = Dataset::new(&s, data, &device).unwrap();
    let model = Model::init(&s, &device).unwrap();
    let predictions = model.predict(&dataset, &options).unwrap();
    let ones = [1.0f32; 8];
    let ap = model
        .evaluate(&dataset, Split::Val, Metric::AveragePrecision, &options)
        .unwrap();
    assert_eq!(ap.count, 8);
    let want = average_precision(&predictions[4..], &labels[4..], &ones, 2).unwrap();
    assert!((ap.value - want).abs() < 1e-6);
    let auc = model
        .evaluate(&dataset, Split::Val, Metric::RocAuc, &options)
        .unwrap();
    let want = roc_auc(&predictions[4..], &labels[4..], &ones, 2).unwrap();
    assert!((auc.value - want).abs() < 1e-6);

    // Two node classes: the score is the margin of class 1.
    let node = GraphTaskSpec::NodeClass { classes: 2 };
    let s = spec(node);
    let data = with_node_labels(graphs(&[45], false, 33), 2, 34);
    let dataset = Dataset::new(&s, data, &device).unwrap();
    let model = Model::init(&s, &device).unwrap();
    let parts = EvalOptions {
        parts: Some(2),
        ..Default::default()
    };
    let predictions = model.predict(&dataset, &parts).unwrap();
    let node_labels: Vec<f32> = {
        let mut rng = Lcg(34);
        (0..45).map(|_| rng.below(2) as f32).collect()
    };
    let test: Vec<usize> = (0..45).filter(|i| i % 3 == 2).collect();
    let scores: Vec<f32> = test
        .iter()
        .map(|&i| predictions[i * 2 + 1] - predictions[i * 2])
        .collect();
    let truth: Vec<f32> = test.iter().map(|&i| node_labels[i]).collect();
    let auc = model
        .evaluate(&dataset, Split::Test, Metric::RocAuc, &parts)
        .unwrap();
    assert_eq!(auc.count, test.len());
    let want = roc_auc(&scores, &truth, &vec![1.0; test.len()], 1).unwrap();
    assert!((auc.value - want).abs() < 1e-6);
}

// ---------------------------------------------------------------------------
// Expressivity
// ---------------------------------------------------------------------------

/// A 6-cycle and two triangles as two graphs with constant features: both are
/// 2-regular, so message passing cannot tell them apart.
fn cycle_and_triangles() -> GraphData {
    let cycle = [(0u32, 1u32), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)];
    let triangles = [(6u32, 7u32), (7, 8), (8, 6), (9, 10), (10, 11), (11, 9)];
    let (mut src, mut dst) = (Vec::new(), Vec::new());
    for &(a, b) in cycle.iter().chain(&triangles) {
        src.extend([a, b]);
        dst.extend([b, a]);
    }
    let mut data = GraphData::new(
        12,
        src,
        dst,
        Features::Float {
            dim: F,
            data: vec![1.0; 12 * F],
        },
    );
    data.graph_ptr = vec![0, 6, 12];
    data.y = Labels::GraphClass(vec![0, 1]);
    data.masks = Some(all_train(2));
    data
}

#[test]
fn walk_tokens_separate_what_message_passing_cannot() {
    let _guard = lock();
    let device = dev();
    let task = GraphTaskSpec::GraphClass {
        classes: 2,
        pool: GraphPool::Mean,
    };
    let options = EvalOptions {
        batch_rows: Some(16),
        ..Default::default()
    };

    // Node tokens with message passing: 1-WL, the two graphs are the same.
    let s = spec(task).with_tokens(0, 1, 1).with_mpnn(Some(MpnnKind::Gine));
    let dataset = Dataset::new(&s, cycle_and_triangles(), &device).unwrap();
    let model = Model::init(&s, &device).unwrap();
    let out = model.predict(&dataset, &options).unwrap();
    assert_close(&out[..2], &out[2..], 1e-5, "1-WL cannot separate the pair");
    let blind = (out[0] - out[2]).abs().max((out[1] - out[3]).abs());

    // Subgraph tokens: the triangle's 2-hop neighbourhood is three nodes and
    // three edges, the cycle's five nodes and four edges.
    let s = spec(task)
        .with_tokens(2, 8, 2)
        .with_local(LocalEncoder::Sgc { hops: 1 })
        .with_token_sampling(TokenSampling::Static);
    let dataset = Dataset::new(&s, cycle_and_triangles(), &device).unwrap();
    let model = Model::init(&s, &device).unwrap();
    let out = model.predict(&dataset, &options).unwrap();
    // At initialisation the mixers' output projections are scaled down, so the
    // gap is small — but it is there, where message passing has none at all.
    let gap = (out[0] - out[2]).abs().max((out[1] - out[3]).abs());
    println!("gap with message passing {blind:e}, with walk tokens {gap:e}");
    assert!(
        gap > 1e-7 && gap > 20.0 * blind,
        "the token statistics should separate the pair (gap {gap:e} against {blind:e})"
    );

    let mut trainer = GraphTrainer::new(&GraphTrainConfig {
        learning_rate: 3e-3,
        ..Default::default()
    })
    .unwrap();
    for epoch in 0..100 {
        let epoch = dataset.epoch_graphs(Some(16), epoch, Split::Train).unwrap();
        model.train_epoch(&mut trainer, &epoch).unwrap();
    }
    let losses = trainer.read_losses().unwrap();
    assert_eq!(losses.len(), 100);
    assert!(losses[99].loss < losses[0].loss);
    let accuracy = model
        .evaluate(&dataset, Split::Train, Metric::Accuracy, &options)
        .unwrap();
    assert_eq!(accuracy.value, 1.0, "100 steps classify the pair");
}

// ---------------------------------------------------------------------------
// The golden file shared with the Python bindings
// ---------------------------------------------------------------------------

#[derive(serde::Serialize, serde::Deserialize)]
struct Golden {
    spec: serde_json::Value,
    n_nodes: usize,
    edge_src: Vec<u32>,
    edge_dst: Vec<u32>,
    x: Vec<f32>,
    y: Vec<i64>,
    train_mask: Vec<bool>,
    learning_rate: f32,
    losses: Vec<f32>,
    predictions: Vec<f32>,
}

#[test]
fn the_golden_file_is_reproduced() {
    let _guard = lock();
    let device = dev();
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden/graph_tiny.json");
    let n = 12usize;
    let s = spec(GraphTaskSpec::NodeClass { classes: 3 })
        .with_mpnn(Some(MpnnKind::Gine))
        .with_token_sampling(TokenSampling::PerStep);
    let data = graphs(&[n], false, 41);
    let x = match &data.x {
        Features::Float { data, .. } => data.clone(),
        _ => unreachable!(),
    };
    let y: Vec<i64> = (0..n as i64).map(|i| i % 3).collect();
    let train_mask: Vec<bool> = (0..n).map(|i| i % 4 != 3).collect();
    let learning_rate = 2e-3f32;

    let mut labelled = data.clone();
    labelled.y = Labels::Node(y.clone());
    labelled.masks = Some(Splits {
        train: train_mask.clone(),
        val: train_mask.iter().map(|m| !m).collect(),
        test: vec![false; n],
    });
    let dataset = Dataset::new(&s, labelled, &device).unwrap();
    let model = Model::init(&s, &device).unwrap();
    let mut trainer = GraphTrainer::new(&GraphTrainConfig {
        learning_rate,
        ..Default::default()
    })
    .unwrap();
    for epoch in 0..5 {
        let epoch = dataset.epoch_nodes(Some(1), epoch, Split::Train).unwrap();
        model.train_epoch(&mut trainer, &epoch).unwrap();
    }
    let losses: Vec<f32> = trainer.read_losses().unwrap().iter().map(|i| i.loss).collect();
    let predictions = model
        .predict(
            &dataset,
            &EvalOptions {
                parts: Some(1),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(losses.len(), 5);
    assert_eq!(predictions.len(), n * 3);

    if std::env::var_os("MAMBA3_WRITE_GOLDEN").is_some() {
        let golden = Golden {
            spec: serde_json::to_value(&s).unwrap(),
            n_nodes: n,
            edge_src: data.edge_src.clone(),
            edge_dst: data.edge_dst.clone(),
            x,
            y,
            train_mask,
            learning_rate,
            losses,
            predictions,
        };
        std::fs::write(&path, serde_json::to_string_pretty(&golden).unwrap()).unwrap();
        println!("wrote {}", path.display());
        return;
    }
    let golden: Golden = serde_json::from_str(
        &std::fs::read_to_string(&path)
            .unwrap_or_else(|_| panic!("{} is missing; run with MAMBA3_WRITE_GOLDEN=1", path.display())),
    )
    .unwrap();
    assert_eq!(golden.spec, serde_json::to_value(&s).unwrap(), "the spec changed");
    assert_eq!(golden.edge_src, data.edge_src);
    assert_eq!(golden.x, x);
    // Written on the CPU runtime; a GPU agrees to rounding.
    let tol = if device.name() == "cpu" { 1e-4 } else { 1e-3 };
    assert_close(&losses, &golden.losses, tol, "losses");
    assert_close(&predictions, &golden.predictions, tol, "predictions");
}

#[test]
fn node_order_is_part_of_the_spec() {
    let _guard = lock();
    let device = dev();
    let data = with_node_labels(graphs(&[30], false, 17), 4, 18);
    let by_degree = spec(NODE).with_tokens(0, 1, 1);
    let given = by_degree.clone().with_order(NodeOrder::Given);
    let perm = |s: &GraphMambaSpec| Dataset::new(s, data.clone(), &device).unwrap().store().perm().to_vec();
    assert_eq!(perm(&given), (0..30).collect::<Vec<u32>>());
    assert_ne!(perm(&by_degree), perm(&given));
}
