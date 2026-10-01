//! GM1: host graph data — canonical order, the CSR, node orders, validation.
//!
//! Host-only: nothing here touches a device.

use mamba3::models::graph::data::{core_numbers, pagerank};
use mamba3::models::graph::{
    Bools, CanonicalGraph, CanonicalLabels, CanonicalizeOptions, FeatureTable, Features,
    FeaturesView, Floats, GraphData, GraphDataView, HostCsr, Ints, Labels, LabelsView, NodeOrder,
    SPLIT_TEST, SPLIT_TRAIN, SPLIT_VAL, Splits, SplitsView, canonicalize,
};
use mamba3::tensor::ops::IGNORE;

/// Undirected edges as both directed pairs.
fn undirected(pairs: &[(u32, u32)]) -> (Vec<u32>, Vec<u32>) {
    let mut src = Vec::new();
    let mut dst = Vec::new();
    for &(a, b) in pairs {
        src.extend([a, b]);
        dst.extend([b, a]);
    }
    (src, dst)
}

/// One float feature per node holding the node's own id.
fn id_features(n: usize) -> Features {
    Features::Float {
        dim: 1,
        data: (0..n).map(|i| i as f32).collect(),
    }
}

/// Degrees 4, 2, 2, 1, 1, 0 for nodes 0..6.
fn six_nodes() -> GraphData {
    let (src, dst) = undirected(&[(0, 1), (0, 2), (0, 3), (1, 2), (4, 0)]);
    GraphData::new(6, src, dst, id_features(6))
}

fn canon(data: &GraphData, order: NodeOrder) -> CanonicalGraph<f32> {
    canonicalize::<f32>(
        &data.view(),
        &CanonicalizeOptions {
            order,
            symmetrize: true,
            reverse_index: true,
        },
    )
    .unwrap()
}

fn float_data(table: &FeatureTable<f32>) -> &[f32] {
    match table {
        FeatureTable::Float { data, .. } => data,
        FeatureTable::Categorical { .. } => panic!("expected float features"),
    }
}

fn message(err: mamba3::error::Error) -> String {
    err.to_string()
}

#[test]
fn canonical_order_by_degree_and_relabelled_csr() {
    let c = canon(&six_nodes(), NodeOrder::Degree { descending: false });
    // Degrees ascending; ties (3, 4) and (1, 2) keep their original order.
    assert_eq!(c.perm, vec![5, 3, 4, 1, 2, 0]);
    assert_eq!(c.adj_off, vec![0, 0, 1, 2, 4, 6, 10]);
    assert_eq!(c.adj_col, vec![5, 5, 4, 5, 3, 5, 1, 2, 3, 4]);
    assert_eq!(c.graph_ptr, vec![0, 6]);
    assert_eq!(c.edge_ptr, vec![0, 10]);
    assert_eq!(c.n_edges(), 10);
    // Features moved with their nodes.
    assert_eq!(float_data(&c.x), &[5.0, 3.0, 4.0, 1.0, 2.0, 0.0]);
}

#[test]
fn descending_degree_and_given_order() {
    let c = canon(&six_nodes(), NodeOrder::Degree { descending: true });
    assert_eq!(c.perm, vec![0, 1, 2, 3, 4, 5]);
    let c = canon(&six_nodes(), NodeOrder::Given);
    assert_eq!(c.perm, vec![0, 1, 2, 3, 4, 5]);
    assert_eq!(c.adj_col[..4], [1, 2, 3, 4]);
}

#[test]
fn permutation_maps_predictions_back() {
    let c = canon(&six_nodes(), NodeOrder::Degree { descending: false });
    // A "prediction" per canonical row that is a function of the node's feature.
    let predictions: Vec<f32> = float_data(&c.x).iter().map(|v| v * 10.0).collect();
    let mut original = vec![0.0f32; 6];
    for (row, &node) in c.perm.iter().enumerate() {
        original[node as usize] = predictions[row];
    }
    assert_eq!(original, vec![0.0, 10.0, 20.0, 30.0, 40.0, 50.0]);
}

#[test]
fn rows_are_sorted_and_deduplicated_and_self_loops_kept_once() {
    // Duplicates of 0 -> 1, a self-loop on 2 given twice, and a one-way edge.
    let data = GraphData::new(
        3,
        vec![0, 0, 2, 2, 1],
        vec![1, 1, 2, 2, 2],
        id_features(3),
    );
    let csr = HostCsr::from_view(&data.view(), true).unwrap();
    assert_eq!(csr.off, vec![0, 1, 3, 5]);
    assert_eq!(csr.col, vec![1, 0, 2, 1, 2]);
    let directed = HostCsr::from_view(&data.view(), false).unwrap();
    assert_eq!(directed.off, vec![0, 0, 1, 3]);
    assert_eq!(directed.col, vec![0, 1, 2]);
}

#[test]
fn symmetrize_is_idempotent() {
    let mut data = GraphData::new(
        4,
        vec![0, 1, 3, 3, 0],
        vec![1, 2, 3, 0, 1],
        id_features(4),
    );
    data.edge_attr = Some(Features::Float {
        dim: 1,
        data: vec![10.0, 20.0, 30.0, 40.0, 50.0],
    });
    data.symmetrize().unwrap();
    let once = data.clone();
    assert_eq!(once.edge_dst, vec![0, 0, 1, 1, 2, 3, 3]);
    assert_eq!(once.edge_src, vec![1, 3, 0, 2, 1, 0, 3]);
    // 1 -> 0 mirrors edge 0 (the duplicate, edge 4, lost); 3 -> 0 was given;
    // 0 -> 1 is edge 0; 2 -> 1 mirrors edge 1; 1 -> 2 is edge 1; 0 -> 3 mirrors
    // edge 3; the self-loop is edge 2.
    assert_eq!(
        once.edge_attr,
        Some(Features::Float {
            dim: 1,
            data: vec![10.0, 40.0, 10.0, 20.0, 20.0, 40.0, 30.0],
        })
    );
    data.symmetrize().unwrap();
    assert_eq!(data, once);
}

#[test]
fn reverse_index_is_an_involution_on_reverse_pairs() {
    let c = canon(&six_nodes(), NodeOrder::Degree { descending: false });
    let rev = c.adj_rev.as_ref().unwrap();
    let dst = c.adj_dst.as_ref().unwrap();
    for e in 0..c.n_edges() {
        let r = rev[e] as usize;
        assert_eq!(rev[r] as usize, e, "rev is an involution");
        assert_eq!(c.adj_col[r], dst[e], "the reverse starts where the edge ends");
        assert_eq!(dst[r], c.adj_col[e], "the reverse ends where the edge starts");
    }
    for u in 0..c.n_nodes {
        for e in c.adj_off[u]..c.adj_off[u + 1] {
            assert_eq!(dst[e as usize] as usize, u);
        }
    }
}

#[test]
fn reverse_index_refuses_a_directed_graph() {
    let data = GraphData::new(2, vec![0], vec![1], id_features(2));
    let err = canonicalize::<f32>(
        &data.view(),
        &CanonicalizeOptions {
            order: NodeOrder::Given,
            symmetrize: false,
            reverse_index: true,
        },
    )
    .unwrap_err();
    let text = message(err);
    assert!(text.contains("edge_index") && text.contains("symmetr"), "{text}");
}

#[test]
fn edge_features_follow_their_edges() {
    // 0 -> 1 with feature 7, 2 -> 1 with feature 9; symmetrised, relabelled.
    let mut data = GraphData::new(3, vec![0, 2], vec![1, 1], id_features(3));
    data.edge_attr = Some(Features::Float {
        dim: 1,
        data: vec![7.0, 9.0],
    });
    let c = canon(&data, NodeOrder::Degree { descending: false });
    // Degrees 1, 2, 1 -> order 0, 2, 1.
    assert_eq!(c.perm, vec![0, 2, 1]);
    assert_eq!(c.adj_off, vec![0, 1, 2, 4]);
    assert_eq!(c.adj_col, vec![2, 2, 0, 1]);
    assert_eq!(float_data(c.edge_x.as_ref().unwrap()), &[7.0, 9.0, 7.0, 9.0]);
}

#[test]
fn two_graphs_stay_separate() {
    // Graph 0: a path 0-1-2. Graph 1: a star with centre 3 and leaves 4, 5, 6.
    let (src, dst) = undirected(&[(0, 1), (1, 2), (3, 4), (3, 5), (3, 6)]);
    let mut data = GraphData::new(7, src, dst, id_features(7));
    data.graph_ptr = vec![0, 3, 7];
    data.y = Labels::GraphClass(vec![1, IGNORE]);
    let c = canon(&data, NodeOrder::Degree { descending: false });
    assert_eq!(c.n_graphs, 2);
    assert_eq!(c.perm, vec![0, 2, 1, 4, 5, 6, 3]);
    assert_eq!(c.graph_ptr, vec![0, 3, 7]);
    assert_eq!(c.edge_ptr, vec![0, 4, 10]);
    for g in 0..2 {
        let (a, b) = (c.graph_ptr[g], c.graph_ptr[g + 1]);
        for u in a..b {
            for e in c.adj_off[u as usize]..c.adj_off[u as usize + 1] {
                let v = c.adj_col[e as usize];
                assert!(v >= a && v < b, "edge leaves graph {g}");
            }
        }
    }
    assert_eq!(c.y, CanonicalLabels::GraphClass(vec![1, IGNORE]));
    assert_eq!(c.split, vec![SPLIT_TRAIN, SPLIT_TRAIN]);
    assert_eq!(c.labelled, [1, 0, 0]);
}

#[test]
fn pagerank_ranks_the_centre_of_a_star_last() {
    let (src, dst) = undirected(&[(2, 0), (2, 1), (2, 3), (2, 4)]);
    let data = GraphData::new(5, src, dst, id_features(5));
    let c = canon(&data, NodeOrder::ppr());
    assert_eq!(*c.perm.last().unwrap(), 2);
    assert_eq!(c.perm[..4], [0, 1, 3, 4]);
    let csr = HostCsr::from_view(&data.view(), true).unwrap();
    let rank = pagerank(&csr, &[0, 5], 0.15, 50);
    assert!((rank.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    assert!(rank[2] > rank[0]);
}

#[test]
fn core_numbers_of_a_triangle_with_a_pendant() {
    let (src, dst) = undirected(&[(0, 1), (1, 2), (2, 0), (2, 3)]);
    let data = GraphData::new(4, src, dst, id_features(4));
    let csr = HostCsr::from_view(&data.view(), true).unwrap();
    assert_eq!(core_numbers(&csr), vec![2, 2, 2, 1]);
    let c = canon(&data, NodeOrder::KCore);
    assert_eq!(c.perm, vec![3, 0, 1, 2]);
    // A 4-clique with a tail of two: cores 3, 3, 3, 3, 1, 1.
    let (src, dst) = undirected(&[
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
        (2, 3),
        (3, 4),
        (4, 5),
    ]);
    let data = GraphData::new(6, src, dst, id_features(6));
    let csr = HostCsr::from_view(&data.view(), true).unwrap();
    assert_eq!(core_numbers(&csr), vec![3, 3, 3, 3, 1, 1]);
}

#[test]
fn node_labels_and_splits_follow_the_permutation() {
    let mut data = six_nodes();
    data.y = Labels::Node(vec![0, 1, 2, -1, 1, 0]);
    data.masks = Some(Splits {
        train: vec![true, true, false, true, false, false],
        val: vec![false, false, true, false, false, false],
        test: vec![false, false, false, false, true, true],
    });
    let c = canon(&data, NodeOrder::Degree { descending: false });
    // perm = [5, 3, 4, 1, 2, 0]
    assert_eq!(c.y, CanonicalLabels::Node(vec![0, IGNORE, 1, 1, 2, 0]));
    assert_eq!(
        c.split,
        vec![
            SPLIT_TEST,
            SPLIT_TRAIN,
            SPLIT_TEST,
            SPLIT_TRAIN,
            SPLIT_VAL,
            SPLIT_TRAIN
        ]
    );
    // Node 3 is in the training mask but has no label.
    assert_eq!(c.labelled, [2, 1, 2]);
}

#[test]
fn graph_targets_keep_nan_as_missing() {
    let (src, dst) = undirected(&[(0, 1), (2, 3)]);
    let mut data = GraphData::new(4, src, dst, id_features(4));
    data.graph_ptr = vec![0, 2, 4];
    data.y = Labels::Graph {
        targets: 2,
        values: vec![1.0, f32::NAN, f32::NAN, f32::NAN],
    };
    data.masks = Some(Splits {
        train: vec![true, true],
        val: vec![false, false],
        test: vec![false, false],
    });
    let c = canon(&data, NodeOrder::Given);
    let CanonicalLabels::Graph { targets, values } = &c.y else {
        panic!("expected graph targets");
    };
    assert_eq!(*targets, 2);
    assert_eq!(values[0], 1.0);
    assert!(values[1..].iter().all(|v| v.is_nan()));
    // Graph 1 has no target at all, so it is not a labelled item.
    assert_eq!(c.labelled, [1, 0, 0]);
}

#[test]
fn categorical_features_get_field_offsets() {
    let (src, dst) = undirected(&[(0, 1)]);
    let data = GraphData::new(
        2,
        src,
        dst,
        Features::Categorical {
            fields: 2,
            vocab: vec![3, 5],
            ids: vec![2, 4, 0, 1],
        },
    );
    let c = canon(&data, NodeOrder::Given);
    assert_eq!(
        c.x,
        FeatureTable::Categorical {
            field_offset: vec![0, 3, 8],
            ids: vec![2, 4, 0, 1],
        }
    );
    assert_eq!(c.x.width(), 8);
}

#[test]
fn wide_views_give_the_same_tables() {
    let data = {
        let mut d = six_nodes();
        d.pe = Some((2, (0..12).map(|i| i as f32 * 0.5).collect()));
        d.y = Labels::Node(vec![0, 1, 2, -1, 1, 0]);
        d.masks = Some(Splits {
            train: vec![true, true, false, true, false, false],
            val: vec![false, false, true, false, false, false],
            test: vec![false, false, false, false, true, true],
        });
        d
    };
    let reference = canon(&data, NodeOrder::Degree { descending: false });

    let src: Vec<i64> = data.edge_src.iter().map(|&v| v as i64).collect();
    let dst: Vec<i32> = data.edge_dst.iter().map(|&v| v as i32).collect();
    let x: Vec<f64> = (0..6).map(|i| i as f64).collect();
    let pe: Vec<f64> = (0..12).map(|i| i as f64 * 0.5).collect();
    let y: Vec<i32> = vec![0, 1, 2, -1, 1, 0];
    let ptr: Vec<i64> = vec![0, 6];
    let train: Vec<u8> = vec![1, 1, 0, 7, 0, 0];
    let val = [false, false, true, false, false, false];
    let test = [false, false, false, false, true, true];
    let view = GraphDataView {
        n_nodes: 6,
        edge_src: Ints::I64(&src),
        edge_dst: Ints::I32(&dst),
        x: FeaturesView::Float {
            dim: 1,
            data: Floats::F64(&x),
        },
        edge_attr: None,
        pe: Some((2, Floats::F64(&pe))),
        y: LabelsView::Node(Ints::I32(&y)),
        graph_ptr: Some(Ints::I64(&ptr)),
        masks: Some(SplitsView {
            train: Bools::U8(&train),
            val: Bools::Bool(&val),
            test: Bools::Bool(&test),
        }),
    };
    let wide = canonicalize::<f32>(
        &view,
        &CanonicalizeOptions {
            order: NodeOrder::Degree { descending: false },
            symmetrize: true,
            reverse_index: true,
        },
    )
    .unwrap();
    assert_eq!(wide, reference);

    // The same data in a 16-bit store: same structure, features converted.
    let narrow = canonicalize::<half::f16>(
        &view,
        &CanonicalizeOptions {
            order: NodeOrder::Degree { descending: false },
            symmetrize: true,
            reverse_index: true,
        },
    )
    .unwrap();
    assert_eq!(narrow.adj_col, reference.adj_col);
    let FeatureTable::Float { data: narrow_x, .. } = &narrow.x else {
        panic!("expected float features");
    };
    let widened: Vec<f32> = narrow_x.iter().map(|v| v.to_f32()).collect();
    assert_eq!(widened, float_data(&reference.x));
}

#[test]
fn edge_ids_out_of_range_name_edge_index() {
    let x: Vec<f32> = vec![0.0; 3];
    let features = FeaturesView::Float {
        dim: 1,
        data: Floats::F32(&x),
    };
    let base = |src: Ints<'_>, dst: Ints<'_>| {
        let view = GraphDataView {
            n_nodes: 3,
            edge_src: src,
            edge_dst: dst,
            x: features,
            edge_attr: None,
            pe: None,
            y: LabelsView::None,
            graph_ptr: None,
            masks: None,
        };
        message(canonicalize::<f32>(&view, &CanonicalizeOptions::default()).unwrap_err())
    };
    let text = base(Ints::I64(&[0, -1]), Ints::I64(&[1, 2]));
    assert!(text.contains("edge_index") && text.contains("negative"), "{text}");
    let text = base(Ints::I64(&[0, 1 << 40]), Ints::I64(&[1, 2]));
    assert!(text.contains("edge_index") && text.contains("outside"), "{text}");
    let text = base(Ints::U64(&[0, u64::MAX]), Ints::U64(&[1, 2]));
    assert!(text.contains("edge_index"), "{text}");
    let text = base(Ints::U32(&[0, 3]), Ints::U32(&[1, 2]));
    assert!(text.contains("edge_index") && text.contains("outside 0..3"), "{text}");
    let text = base(Ints::U32(&[0]), Ints::U32(&[1, 2]));
    assert!(text.contains("edge_index"), "{text}");
}

#[test]
fn every_validation_error_names_its_field() {
    let good = || {
        let (src, dst) = undirected(&[(0, 1), (2, 3)]);
        let mut d = GraphData::new(4, src, dst, id_features(4));
        d.graph_ptr = vec![0, 2, 4];
        d
    };
    let fails = |data: GraphData, field: &str| {
        let text = message(data.validate().unwrap_err());
        assert!(text.contains(field), "expected `{field}` in: {text}");
    };
    good().validate().unwrap();

    let mut d = good();
    d.graph_ptr = vec![1, 2, 4];
    fails(d, "graph_ptr");
    let mut d = good();
    d.graph_ptr = vec![0, 3, 2, 4];
    fails(d, "graph_ptr");
    let mut d = good();
    d.graph_ptr = vec![0, 2, 5];
    fails(d, "graph_ptr");
    let mut d = good();
    d.graph_ptr = vec![0];
    fails(d, "graph_ptr");

    // An edge from graph 0 into graph 1.
    let mut d = good();
    d.edge_src.push(1);
    d.edge_dst.push(2);
    fails(d, "edge_index");

    let mut d = good();
    d.x = Features::Float {
        dim: 2,
        data: vec![0.0; 7],
    };
    fails(d, "x");
    let mut d = good();
    d.x = Features::Float {
        dim: 1,
        data: vec![0.0, f32::NAN, 0.0, 0.0],
    };
    fails(d, "x");
    let mut d = good();
    d.x = Features::Categorical {
        fields: 1,
        vocab: vec![3],
        ids: vec![0, 1, 3, 2],
    };
    fails(d, "x");
    let mut d = good();
    d.x = Features::Categorical {
        fields: 2,
        vocab: vec![3],
        ids: vec![0; 8],
    };
    fails(d, "x");

    let mut d = good();
    d.edge_attr = Some(Features::Float {
        dim: 1,
        data: vec![0.0; 3],
    });
    fails(d, "edge_attr");

    let mut d = good();
    d.pe = Some((2, vec![0.0; 7]));
    fails(d, "pe");
    let mut d = good();
    d.pe = Some((1, vec![0.0, f32::INFINITY, 0.0, 0.0]));
    fails(d, "pe");

    let mut d = good();
    d.y = Labels::Node(vec![0; 3]);
    fails(d, "y");
    let mut d = good();
    d.y = Labels::GraphClass(vec![0; 3]);
    fails(d, "y");
    let mut d = good();
    d.y = Labels::Graph {
        targets: 2,
        values: vec![0.0; 3],
    };
    fails(d, "y");
    let mut d = good();
    d.y = Labels::Graph {
        targets: 1,
        values: vec![0.0, f32::INFINITY],
    };
    fails(d, "y");

    let mask = |n: usize| Splits {
        train: vec![true; n],
        val: vec![false; 4],
        test: vec![false; 4],
    };
    let mut d = good();
    d.y = Labels::Node(vec![0; 4]);
    d.masks = Some(mask(3));
    fails(d, "train_mask");
    let mut d = good();
    d.y = Labels::Node(vec![0; 4]);
    d.masks = Some(Splits {
        train: vec![true; 4],
        val: vec![false; 4],
        test: vec![false; 2],
    });
    fails(d, "test_mask");

    let d = GraphData::new(0, vec![], vec![], id_features(0));
    fails(d, "n_nodes");
}

#[test]
fn owned_canonicalize_relabels_everything() {
    let mut data = six_nodes();
    data.y = Labels::Node(vec![0, 1, 2, -1, 1, 0]);
    let (sorted, perm) = data
        .canonicalize(NodeOrder::Degree { descending: false })
        .unwrap();
    assert_eq!(perm, vec![5, 3, 4, 1, 2, 0]);
    assert_eq!(sorted.y, Labels::Node(vec![0, -1, 1, 1, 2, 0]));
    assert_eq!(sorted.edge_src, vec![5, 5, 4, 5, 3, 5, 1, 2, 3, 4]);
    assert_eq!(sorted.edge_dst, vec![1, 2, 3, 3, 4, 4, 5, 5, 5, 5]);
    // Already canonical: a second pass changes nothing.
    let (again, perm) = sorted
        .canonicalize(NodeOrder::Degree { descending: false })
        .unwrap();
    assert_eq!(perm, vec![0, 1, 2, 3, 4, 5]);
    assert_eq!(again, sorted);
}
