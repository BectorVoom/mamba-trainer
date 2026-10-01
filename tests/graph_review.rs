//! Regressions found by the review pass of the graph model (GM14): each test
//! is a case that the suites written with the code did not reach.

#![cfg(feature = "backend")]

use mamba3::backend::Device;
use mamba3::backends::Auto;
use mamba3::models::graph::metrics::{average_precision, f1_macro, roc_auc};
use mamba3::models::graph::{
    CanonicalizeOptions, EvalOptions, FeatureSpec, Features, GraphData, GraphDataset, GraphMamba,
    GraphMambaSpec, GraphTask, GraphTaskSpec, GraphTrainConfig, GraphTrainer, Labels, Metric,
    MpnnKind, NodeOrder, Split, Splits, canonicalize,
};
use mamba3::models::graph::GraphStore;
use mamba3::tensor::Tensor;
use mamba3::tensor::ops::graph::{
    Adjacency, EpochTable, batch_rows_graphs, batch_rows_subset, gine_aggregate,
    gine_aggregate_du, mask_edge_rows, safe_class_targets,
};
use mamba3::tensor::ops::index::IdTensor;
use mamba3::train::TrainStep;

type R = Auto;

fn dev() -> Device<R> {
    Device::<R>::default()
}

/// A ring of `n` nodes with chords, `features` constant-ish features, and
/// `classes` classes by node index; every node is in the training split.
fn ring(n: usize, features: usize, classes: i64) -> GraphData {
    let (mut src, mut dst) = (Vec::new(), Vec::new());
    for i in 0..n as u32 {
        for other in [(i + 1) % n as u32, (i * 5 + 2) % n as u32] {
            if other != i {
                src.push(i);
                dst.push(other);
            }
        }
    }
    let mut data = GraphData::new(
        n,
        src,
        dst,
        Features::Float {
            dim: features,
            data: (0..n * features)
                .map(|i| ((i * 37 % 19) as f32 - 9.0) / 9.0)
                .collect(),
        },
    );
    data.y = Labels::Node((0..n as i64).map(|i| i % classes).collect());
    data.masks = Some(Splits {
        train: vec![true; n],
        val: vec![true; n],
        test: vec![false; n],
    });
    data
}

fn small(classes: usize) -> GraphMambaSpec {
    GraphMambaSpec::new(
        FeatureSpec::Float { dim: 3 },
        GraphTaskSpec::NodeClass { classes },
    )
    .with_d_model(16)
    .with_tokens(1, 2, 1)
    .with_node_layers(1)
}

/// A node task on one graph whose model cannot run on node partitions
/// (GatedGcn) trains on the whole graph, and is predicted and evaluated on it.
#[test]
fn a_single_graph_with_gated_gcn_is_evaluated_whole() {
    let device = dev();
    let spec = small(3).with_mpnn(Some(MpnnKind::GatedGcn));
    let dataset = GraphDataset::<R, f32>::new(&spec, ring(40, 3, 3), &device).unwrap();
    let model = GraphMamba::<R, f32>::init(&spec, &device).unwrap();
    let mut trainer = GraphTrainer::new(&GraphTrainConfig::default()).unwrap();
    let plan = dataset.epoch_graphs(None, 0, Split::Train).unwrap();
    assert_eq!(model.train_epoch(&mut trainer, &plan).unwrap(), 1);
    assert!(trainer.read_losses().unwrap()[0].loss.is_finite());

    let options = EvalOptions::default();
    let predictions = model.predict(&dataset, &options).unwrap();
    assert_eq!(predictions.len(), 40 * 3);
    assert!(predictions.iter().all(|v| v.is_finite()));
    let result = model
        .evaluate(&dataset, Split::Val, Metric::Accuracy, &options)
        .unwrap();
    assert_eq!(result.count, 40);

    // Asking for node partitions of such a model is refused, with the reason.
    let parts = EvalOptions {
        parts: Some(2),
        ..Default::default()
    };
    let err = model.predict(&dataset, &parts).unwrap_err().to_string();
    assert!(err.contains("GatedGcn"), "{err}");

    // A model that can run on partitions still does, by default.
    let plain = small(3);
    let dataset = GraphDataset::<R, f32>::new(&plain, ring(40, 3, 3), &device).unwrap();
    let model = GraphMamba::<R, f32>::init(&plain, &device).unwrap();
    let by_parts = model.predict(&dataset, &parts).unwrap();
    let by_rows = model
        .predict(
            &dataset,
            &EvalOptions {
                batch_rows: Some(64),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(by_parts.len(), by_rows.len());
}

/// A model with fewer classes than the dataset's labels is refused before its
/// loss indexes past its logits.
#[test]
fn a_model_with_fewer_classes_than_the_labels_is_refused() {
    let device = dev();
    let dataset = GraphDataset::<R, f32>::new(&small(5), ring(40, 3, 5), &device).unwrap();
    assert_eq!(dataset.store().class_bound(), 5);
    let model = GraphMamba::<R, f32>::init(&small(3), &device).unwrap();
    let batch = dataset
        .epoch_nodes(Some(1), 0, Split::Train)
        .unwrap()
        .batch(0)
        .unwrap();
    let err = GraphTask::new(&model).loss(&batch).err().expect("refused").to_string();
    assert!(err.starts_with("invalid configuration: y:") || err.contains("y: the dataset holds class 4"), "{err}");
    assert!(
        model
            .evaluate(&dataset, Split::Val, Metric::Accuracy, &EvalOptions::default())
            .is_err()
    );
    // More classes than the labels use is fine.
    let wide = GraphMamba::<R, f32>::init(&small(7), &device).unwrap();
    assert!(GraphTask::new(&wide).loss(&batch).is_ok());
}

/// The rank metrics return on a NaN score, and say so.
#[test]
fn rank_metrics_terminate_on_a_nan_score() {
    let scores = [0.9, f32::NAN, 0.2, 0.4];
    let labels = [1.0, 0.0, 1.0, 0.0];
    let mask = [1.0; 4];
    assert!(average_precision(&scores, &labels, &mask, 1).unwrap().is_nan());
    assert!(roc_auc(&scores, &labels, &mask, 1).unwrap().is_nan());
    // A masked NaN is not a score.
    let mask = [1.0, 0.0, 1.0, 1.0];
    assert!(average_precision(&scores, &labels, &mask, 1).unwrap().is_finite());
    assert!(roc_auc(&scores, &labels, &mask, 1).unwrap().is_finite());
}

/// Macro-F1 averages over the classes that occur, as scikit-learn's default.
#[test]
fn macro_f1_ignores_classes_that_do_not_occur() {
    // Three classes; the split holds classes 0 and 1 only and is predicted
    // perfectly.
    let confusion = [4u64, 0, 0, 0, 6, 0, 0, 0, 0];
    assert!((f1_macro(&confusion, 3) - 1.0).abs() < 1e-6);
    // A class that is predicted but never true does occur, and scores 0.
    let confusion = [4u64, 0, 0, 0, 5, 1, 0, 0, 0];
    let want = (1.0 + 2.0 * 5.0 / (6.0 + 5.0) + 0.0) / 3.0;
    assert!((f1_macro(&confusion, 3) - want).abs() < 1e-6);
}

/// The k-core order is that of the undirected graph, whether or not the
/// dataset is symmetrised.
#[test]
fn the_core_order_does_not_depend_on_symmetrising() {
    // A triangle 0-1-2 given in one direction, with a pendant path 2 → 3 → 4.
    let mut data = GraphData::new(
        5,
        vec![0, 1, 2, 2, 3],
        vec![1, 2, 0, 3, 4],
        Features::Float {
            dim: 1,
            data: vec![0.0; 5],
        },
    );
    data.y = Labels::None;
    let perm_of = |symmetrize: bool| {
        canonicalize::<f32>(
            &data.view(),
            &CanonicalizeOptions {
                order: NodeOrder::KCore,
                symmetrize,
                reverse_index: false,
            },
        )
        .unwrap()
        .perm
    };
    // Cores: the path nodes 3 and 4 are 1, the triangle is 2; ties keep the
    // given order.
    assert_eq!(perm_of(true), vec![3, 4, 0, 1, 2]);
    assert_eq!(perm_of(false), perm_of(true));
}

/// A reverse-edge index must pair each edge with its reverse, not merely be
/// an involution.
#[test]
fn a_reverse_index_that_is_only_an_involution_is_refused() {
    let device = dev();
    // The path 0 - 1 - 2: row 0 = [1], row 1 = [0, 2], row 2 = [1].
    let (off, col, dst) = (vec![0, 1, 3, 4], vec![1, 0, 2, 1], vec![0, 1, 1, 2]);
    let upload = |rev: Vec<u32>| {
        Adjacency::<R>::upload(off.clone(), col.clone(), Some(rev), Some(dst.clone()), &device)
    };
    assert!(upload(vec![1, 0, 3, 2]).is_ok());
    // The identity and a swap of unrelated edges are involutions.
    for wrong in [vec![0, 1, 2, 3], vec![3, 2, 1, 0], vec![2, 3, 0, 1]] {
        let err = upload(wrong.clone()).err().expect("refused").to_string();
        assert!(err.contains("reverse-edge index"), "{wrong:?}: {err}");
    }
}

/// Step counters, which seed the walks, are distinct over a run even when the
/// number of batches differs from one epoch to the next.
#[test]
fn step_counters_are_distinct_across_epochs() {
    let device = dev();
    let spec = small(3);
    let dataset = GraphDataset::<R, f32>::new(&spec, ring(40, 3, 3), &device).unwrap();
    let mut seen = std::collections::BTreeSet::new();
    for (epoch, parts) in [(0u64, 4usize), (1, 3), (2, 5), (3, 4), (4, 1), (5, 6)] {
        let plan = dataset.epoch_nodes(Some(parts), epoch, Split::Train).unwrap();
        for index in 0..plan.len() {
            let batch = plan.batch(index).unwrap();
            assert!(batch.step_counter != 0, "0 is the static sample");
            assert!(
                seen.insert(batch.step_counter),
                "epoch {epoch} batch {index} repeats a counter"
            );
        }
    }
}

/// `symmetrize` reports edge features of the wrong length instead of slicing
/// past them.
#[test]
fn symmetrize_checks_the_edge_feature_rows() {
    let mut data = ring(6, 3, 3);
    data.edge_attr = Some(Features::Float {
        dim: 2,
        data: vec![0.0; 6],
    });
    let err = data.symmetrize().unwrap_err().to_string();
    assert!(err.contains("edge_attr"), "{err}");
}

/// Dropout is applied in training mode and is a function of the spec's seed
/// (each layer has its own stream, seeded from it).
#[test]
fn dropout_is_on_in_training_and_reproducible() {
    let device = dev();
    let outputs = |seed: u64, dropout: f32| {
        let spec = small(3)
            .with_tokens(0, 1, 1)
            .with_node_layers(2)
            .with_dropout(dropout)
            .with_seed(seed);
        let dataset = GraphDataset::<R, f32>::new(&spec, ring(40, 3, 3), &device).unwrap();
        let model = GraphMamba::<R, f32>::init(&spec, &device).unwrap();
        let batch = dataset
            .epoch_nodes(Some(1), 0, Split::Train)
            .unwrap()
            .batch(0)
            .unwrap();
        // Training mode: dropout is on.
        model.forward(&batch).unwrap().to_f32()
    };
    // Dropout changes the training-mode output at all.
    assert_ne!(outputs(1, 0.5), outputs(1, 0.0));
    // The same seed reproduces it.
    assert_eq!(outputs(1, 0.5), outputs(1, 0.5));
}

// ---------------------------------------------------------------------------
// The launchers check what they used to take on trust
// ---------------------------------------------------------------------------

/// Two graphs of 4 and 3 nodes (rings), symmetrised, in the given order.
fn two_rings(device: &Device<R>) -> GraphStore<R, f32> {
    let pairs = [(0u32, 1u32), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 4)];
    let mut data = GraphData::new(
        7,
        pairs.iter().map(|p| p.0).collect(),
        pairs.iter().map(|p| p.1).collect(),
        Features::Float {
            dim: 1,
            data: vec![0.0; 7],
        },
    );
    data.graph_ptr = vec![0, 4, 7];
    let canon = canonicalize::<f32>(
        &data.view(),
        &CanonicalizeOptions {
            order: NodeOrder::Given,
            symmetrize: true,
            reverse_index: true,
        },
    )
    .unwrap();
    GraphStore::upload(canon, device).unwrap()
}

/// The adjoint of the edge pre-activation with respect to `C e` is zero on
/// the edge rows a batch does not use, as its forward is.
#[test]
fn absent_edge_rows_get_no_gradient() {
    let device = dev();
    let store = two_rings(&device);
    let table = EpochTable::build(
        &[vec![1, 0]],
        store.graph_ptr(),
        store.edge_ptr(),
        4,
        &device,
    )
    .unwrap();
    let slots = table.slots(0).unwrap();
    let used = slots.edges_used();
    assert_eq!(used, 14, "both rings, in both directions");

    let padded = batch_rows_graphs(store.adjacency(), &slots, 8, Some(used + 6)).unwrap();
    assert!(!padded.edges_full());
    let ones = Tensor::<R, f32>::ones(vec![used + 6, 3], &device);
    let masked = mask_edge_rows(&ones, &padded).unwrap().to_f32();
    assert!(masked[..used * 3].iter().all(|&v| v == 1.0));
    assert!(masked[used * 3..].iter().all(|&v| v == 0.0));

    let exact = batch_rows_graphs(store.adjacency(), &slots, 8, Some(used)).unwrap();
    assert!(exact.edges_full(), "no absent edge row, no masking launch");
    // A batch laid out without edges has none to mask.
    let plain = batch_rows_graphs(store.adjacency(), &slots, 8, None).unwrap();
    assert!(mask_edge_rows(&ones, &plain).is_err());
}

/// The adjoint of message passing is refused on a graph that is not its own
/// transpose, instead of returning the gradient of another graph.
#[test]
fn message_passing_is_not_differentiated_on_a_directed_graph() {
    let device = dev();
    // 0 → 1 → 2, one direction only: row 1 = [0], row 2 = [1].
    let directed =
        Adjacency::<R>::upload(vec![0, 0, 1, 2], vec![0, 1], None, None, &device).unwrap();
    assert!(!directed.symmetric());
    let rows = batch_rows_subset(&directed, 1, 0, (0, 0), 1).unwrap();
    let u = Tensor::<R, f32>::ones(vec![3, 2], &device);
    // The forward pass is defined on any graph.
    assert_eq!(
        gine_aggregate(&u, None, &directed, &rows).unwrap().to_f32(),
        vec![1.0, 1.0, 2.0, 2.0, 2.0, 2.0]
    );
    let err = gine_aggregate_du(&u, &u, None, &directed, &rows)
        .err()
        .expect("refused")
        .to_string();
    assert!(err.contains("symmetric"), "{err}");

    let both = Adjacency::<R>::upload(vec![0, 1, 3, 4], vec![1, 0, 2, 1], None, None, &device)
        .unwrap();
    assert!(both.symmetric());
    let rows = batch_rows_subset(&both, 1, 0, (0, 0), 1).unwrap();
    assert!(gine_aggregate_du(&u, &u, None, &both, &rows).is_ok());
}

/// A batch laid out from one dataset is refused with another's adjacency,
/// also when the two have the same number of nodes.
#[test]
fn a_batch_is_tied_to_its_datasets_edges() {
    let device = dev();
    let store = two_rings(&device);
    let table = EpochTable::build(
        &[vec![0, 1]],
        store.graph_ptr(),
        store.edge_ptr(),
        2,
        &device,
    )
    .unwrap();
    let rows = batch_rows_graphs(store.adjacency(), &table.slots(0).unwrap(), 8, None).unwrap();
    let u = Tensor::<R, f32>::ones(vec![8, 2], &device);
    assert!(gine_aggregate(&u, None, store.adjacency(), &rows).is_ok());
    // Seven nodes again, with fewer edges.
    let other = Adjacency::<R>::upload(
        vec![0, 1, 2, 2, 2, 2, 2, 2],
        vec![1, 0],
        None,
        None,
        &device,
    )
    .unwrap();
    let err = gine_aggregate(&u, None, &other, &rows)
        .err()
        .expect("refused")
        .to_string();
    assert!(err.contains("different dataset"), "{err}");
}

/// The tables an epoch is described from, and the targets a batch gathers,
/// are checked against the dataset.
#[test]
fn epoch_offsets_and_target_tables_are_validated() {
    let device = dev();
    let store = two_rings(&device);
    let build = |graph_ptr: &[u32], edge_ptr: &[u32]| {
        EpochTable::<R>::build(&[vec![0, 1]], graph_ptr, edge_ptr, 2, &device)
    };
    assert!(build(store.graph_ptr(), store.edge_ptr()).is_ok());
    for (graph_ptr, edge_ptr) in [
        (vec![0u32, 4, 3], store.edge_ptr().to_vec()),
        (vec![1, 4, 7], store.edge_ptr().to_vec()),
        (store.graph_ptr().to_vec(), vec![0, 8, 6]),
    ] {
        let err = build(&graph_ptr, &edge_ptr).err().expect("refused").to_string();
        assert!(err.contains("offsets"), "{err}");
    }

    let table = build(store.graph_ptr(), store.edge_ptr()).unwrap();
    let rows = batch_rows_graphs(store.adjacency(), &table.slots(0).unwrap(), 8, None).unwrap();
    let labels = |count: usize| IdTensor::<R>::from_vec(vec![0; count], vec![count], &device).unwrap();
    // One label per graph of the dataset: two.
    assert!(safe_class_targets::<R, f32>(&labels(2), &labels(2), &rows, true, 1).is_ok());
    for count in [1usize, 3] {
        assert!(
            safe_class_targets::<R, f32>(&labels(count), &labels(count), &rows, true, 1).is_err(),
            "{count} labels for 2 graphs"
        );
    }
}

/// Masks without targets select nothing and are an error, not a silence.
#[test]
fn masks_without_targets_are_refused() {
    let mut data = ring(6, 3, 3);
    data.y = Labels::None;
    let err = canonicalize::<f32>(
        &data.view(),
        &CanonicalizeOptions {
            order: NodeOrder::Given,
            symmetrize: true,
            reverse_index: false,
        },
    )
    .err()
    .expect("refused")
    .to_string();
    assert!(err.contains("train_mask"), "{err}");
}
