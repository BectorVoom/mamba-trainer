//! GM5: the message-passing layers, the local encoder and the spec.
//!
//! The layers are checked against host implementations of the formulas that
//! read the layer's own weights, and every nonlinear kernel's adjoint against
//! finite differences.

#![cfg(feature = "backend")]

use std::collections::HashMap;

use mamba3::autograd::Var;
use mamba3::backend::Device;
use mamba3::backends::Auto;
use mamba3::models::graph::{
    BatchMode, CanonicalGraph, CanonicalizeOptions, FeatureSpec, FeatureTable, Features, GatedGcn,
    Gine, GraphData, GraphMambaSpec, GraphPool, GraphStore, GraphTaskSpec, LocalEncoder,
    LocalEncoderModule, MpnnKind, NodeOrder, RegressionLoss, TokenTail, canonicalize,
};
use mamba3::models::vision::ScanDirection;
use mamba3::nn::linear::LinearConfig;
use mamba3::nn::module::Module;
use mamba3::tensor::ops::graph::{
    BatchRows, EpochTable, WalkShape, batch_rows_graphs, batch_rows_subset, edge_inputs,
    gated_edge, gated_edge_backward, gated_node, gated_node_db, gated_node_dehat, gine_aggregate,
    gine_aggregate_dee, gine_aggregate_du, node_inputs, token_features, walk_tokens,
};
use mamba3::tensor::ops::random::Rng;
use mamba3::tensor::ops::IGNORE;
use mamba3::tensor::Tensor;

type R = Auto;

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

fn dot(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum()
}

fn tensor(data: &[f32], shape: Vec<usize>, device: &Device<R>) -> Tensor<R, f32> {
    Tensor::from_f32(data, shape, device).unwrap()
}

/// Five nodes: a triangle 0-1-2, a tail 2-3 with a self-loop on 3, and node 4
/// with no edge. Edge features are three floats per directed edge.
fn five_nodes(edge_features: bool) -> GraphData {
    let pairs = [(0u32, 1u32), (1, 2), (2, 0), (2, 3)];
    let mut src: Vec<u32> = pairs.iter().flat_map(|p| [p.0, p.1]).collect();
    let mut dst: Vec<u32> = pairs.iter().flat_map(|p| [p.1, p.0]).collect();
    src.push(3);
    dst.push(3);
    let mut rng = Lcg(5);
    let mut data = GraphData::new(
        5,
        src,
        dst,
        Features::Float {
            dim: 2,
            data: rng.vec(10),
        },
    );
    if edge_features {
        data.edge_attr = Some(Features::Float {
            dim: 3,
            data: rng.vec(data.edge_src.len() * 3),
        });
    }
    data
}

/// Several random graphs; with `edge_features`, three floats per edge.
fn many_graphs(sizes: &[usize], edge_features: bool, seed: u64) -> GraphData {
    let mut rng = Lcg(seed);
    let (mut src, mut dst, mut ptr) = (Vec::new(), Vec::new(), vec![0u32]);
    let mut at = 0usize;
    for &size in sizes {
        for _ in 0..2 * size {
            let (a, b) = (at + rng.below(size), at + rng.below(size));
            // Both directions, so that each carries its own features.
            src.extend([a as u32, b as u32]);
            dst.extend([b as u32, a as u32]);
        }
        at += size;
        ptr.push(at as u32);
    }
    let mut data = GraphData::new(
        at,
        src,
        dst,
        Features::Float {
            dim: 2,
            data: rng.vec(at * 2),
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

/// A batch on the device and its graph in row space on the host.
struct Fixture {
    canon: CanonicalGraph<f32>,
    store: GraphStore<R, f32>,
    rows: BatchRows<R>,
    /// For every row, its in-edges as `(source row, edge row)`.
    incoming: Vec<Vec<(usize, usize)>>,
    n_rows: usize,
    n_edges: usize,
}

impl Fixture {
    /// Whole graphs `batch` of `data`, laid out with their edges.
    fn graphs(data: &GraphData, batch: &[u32], row_cap: usize, edge_cap: usize) -> Self {
        let device = dev();
        let canon = canonicalize::<f32>(
            &data.view(),
            &CanonicalizeOptions {
                order: NodeOrder::default(),
                symmetrize: true,
                reverse_index: true,
            },
        )
        .unwrap();
        let store = GraphStore::upload(canon.clone(), &device).unwrap();
        let epoch = EpochTable::build(
            &[batch.to_vec()],
            &canon.graph_ptr,
            &canon.edge_ptr,
            8,
            &device,
        )
        .unwrap();
        let rows = batch_rows_graphs(
            store.adjacency(),
            &epoch.slots(0).unwrap(),
            row_cap,
            Some(edge_cap),
        )
        .unwrap();
        let gid = rows.gid().to_vec();
        let eid = rows.edges().unwrap().eid().to_vec();
        let row_of: HashMap<u32, usize> = gid
            .iter()
            .enumerate()
            .filter(|(_, v)| **v != IGNORE)
            .map(|(r, &v)| (v, r))
            .collect();
        let edge_of: HashMap<u32, usize> = eid
            .iter()
            .enumerate()
            .filter(|(_, e)| **e != IGNORE)
            .map(|(r, &e)| (e, r))
            .collect();
        let incoming = gid
            .iter()
            .map(|&node| {
                if node == IGNORE {
                    return Vec::new();
                }
                (canon.adj_off[node as usize]..canon.adj_off[node as usize + 1])
                    .map(|e| (row_of[&canon.adj_col[e as usize]], edge_of[&e]))
                    .collect()
            })
            .collect();
        Self {
            canon,
            store,
            rows,
            incoming,
            n_rows: row_cap,
            n_edges: edge_cap,
        }
    }

    /// Part `part` of `parts` of a single graph; edges into nodes outside the
    /// batch are left out of `incoming`.
    fn subset(data: &GraphData, parts: usize, part: usize) -> Self {
        let device = dev();
        let canon = canonicalize::<f32>(&data.view(), &CanonicalizeOptions::default()).unwrap();
        let store = GraphStore::upload(canon.clone(), &device).unwrap();
        let rows = batch_rows_subset(store.adjacency(), parts, part, (11, 12), 1).unwrap();
        let gid = rows.gid().to_vec();
        let row_of: HashMap<u32, usize> = gid
            .iter()
            .enumerate()
            .filter(|(_, v)| **v != IGNORE)
            .map(|(r, &v)| (v, r))
            .collect();
        let incoming = gid
            .iter()
            .map(|&node| {
                if node == IGNORE {
                    return Vec::new();
                }
                (canon.adj_off[node as usize]..canon.adj_off[node as usize + 1])
                    .filter_map(|e| row_of.get(&canon.adj_col[e as usize]).map(|&s| (s, 0)))
                    .collect()
            })
            .collect();
        let n_rows = gid.len();
        Self {
            canon,
            store,
            rows,
            incoming,
            n_rows,
            n_edges: 0,
        }
    }

    /// The batch's `[edges, 3]` edge inputs.
    fn edge_inputs(&self) -> Tensor<R, f32> {
        edge_inputs(
            &self.store.edge_x().unwrap().source(),
            &self.rows,
            self.store.n_edges(),
        )
        .unwrap()
    }
}

/// A module's parameters on the host, by name.
struct Weights(HashMap<String, Vec<f32>>);

impl Weights {
    fn of<M: Module<R, f32>>(module: &M) -> Self {
        Self(
            module
                .named_parameters()
                .into_iter()
                .map(|(name, param)| (name, param.value().to_f32()))
                .collect(),
        )
    }

    fn get(&self, name: &str) -> &[f32] {
        self.0
            .get(name)
            .unwrap_or_else(|| panic!("no parameter {name} among {:?}", self.0.keys()))
    }

    /// `x @ W + b` for the `Linear` called `name`; `x` is `[rows, d_in]`.
    fn linear(&self, name: &str, x: &[f32], d_in: usize) -> Vec<f32> {
        let weight = self.get(&format!("{name}.weight"));
        let bias = self.get(&format!("{name}.bias"));
        let d_out = bias.len();
        assert_eq!(weight.len(), d_in * d_out);
        let rows = x.len() / d_in;
        let mut out = vec![0.0f32; rows * d_out];
        for r in 0..rows {
            for o in 0..d_out {
                let mut acc = bias[o] as f64;
                for i in 0..d_in {
                    acc += x[r * d_in + i] as f64 * weight[i * d_out + o] as f64;
                }
                out[r * d_out + o] = acc as f32;
            }
        }
        out
    }

    /// `x / sqrt(mean(x²) + eps) * g` for the `RmsNorm` called `name`.
    fn rms_norm(&self, name: &str, x: &[f32], eps: f32) -> Vec<f32> {
        let gain = self.get(&format!("{name}.weight"));
        let d = gain.len();
        let mut out = x.to_vec();
        for row in out.chunks_mut(d) {
            let mean: f32 = row.iter().map(|v| v * v).sum::<f32>() / d as f32;
            let scale = 1.0 / (mean + eps).sqrt();
            for (v, g) in row.iter_mut().zip(gain) {
                *v *= scale * g;
            }
        }
        out
    }
}

fn relu(x: &[f32]) -> Vec<f32> {
    x.iter().map(|v| v.max(0.0)).collect()
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// GINE's aggregation on the host.
fn host_gine_aggregate(fixture: &Fixture, u: &[f32], ee: Option<&[f32]>, d: usize) -> Vec<f32> {
    let mut out = u.to_vec();
    for (r, edges) in fixture.incoming.iter().enumerate() {
        for &(s, e) in edges {
            for j in 0..d {
                let msg = u[s * d + j] + ee.map_or(0.0, |ee| ee[e * d + j]);
                out[r * d + j] += msg.max(0.0);
            }
        }
    }
    out
}

const D: usize = 8;

#[test]
fn gine_matches_the_host_formula() {
    let device = dev();
    for (data, batch, edges) in [
        (five_nodes(false), vec![0u32], false),
        (five_nodes(true), vec![0], true),
        (many_graphs(&[6, 4, 7], false, 3), vec![2, 0], false),
        (many_graphs(&[6, 4, 7], true, 3), vec![2, 0], true),
    ] {
        let fixture = Fixture::graphs(&data, &batch, 16, 64);
        let layer = Gine::<R, f32>::new(D, edges.then_some(3), &device, &mut Rng::seeded(2));
        let weights = Weights::of(&layer);
        let u_host = Lcg(8).vec(fixture.n_rows * D);
        let u = Var::constant(tensor(&u_host, vec![fixture.n_rows, D], &device));
        let edge_in = edges.then(|| fixture.edge_inputs());
        let got = layer
            .apply(
                &u,
                edge_in.clone().map(Var::constant).as_ref(),
                fixture.store.adjacency(),
                &fixture.rows,
            )
            .unwrap()
            .to_f32();

        let ee = edge_in.map(|x| weights.linear("edge", &x.to_f32(), 3));
        let aggregated = host_gine_aggregate(&fixture, &u_host, ee.as_deref(), D);
        let hidden = relu(&weights.linear("lin1", &aggregated, D));
        let want = weights.linear("lin2", &hidden, D);
        assert_close(&got, &want, 1e-5, "gine branch");
        assert!(got.iter().all(|v| v.is_finite()));
    }
}

#[test]
fn a_node_without_neighbours_keeps_its_own_value() {
    let device = dev();
    let fixture = Fixture::graphs(&five_nodes(false), &[0], 16, 64);
    let u_host = Lcg(9).vec(16 * D);
    let u = tensor(&u_host, vec![16, D], &device);
    let got = gine_aggregate(&u, None, fixture.store.adjacency(), &fixture.rows)
        .unwrap()
        .to_f32();
    // Ascending degree puts the isolated node first.
    assert!(fixture.incoming[0].is_empty());
    assert_eq!(&got[..D], &u_host[..D]);
    // Absent rows pass through too.
    assert_eq!(&got[5 * D..], &u_host[5 * D..]);
    // The self-loop sends the node's own message to itself.
    let looped = (0..5)
        .find(|&r| fixture.incoming[r].iter().any(|&(s, _)| s == r))
        .expect("one node has a self-loop");
    assert_eq!(fixture.incoming[looped].len(), 2, "its neighbour and itself");
    assert_close(
        &got,
        &host_gine_aggregate(&fixture, &u_host, None, D),
        1e-6,
        "aggregation",
    );
}

/// GatedGCN on the host: the branch and the updated edge states.
fn host_gated_gcn(
    fixture: &Fixture,
    weights: &Weights,
    u: &[f32],
    e: &[f32],
    update: bool,
) -> (Vec<f32>, Option<Vec<f32>>) {
    let (rows, edges) = (fixture.n_rows, fixture.n_edges);
    let (au, bu, ce, du, eu) = (
        weights.linear("a", u, D),
        weights.linear("b", u, D),
        weights.linear("c", e, D),
        weights.linear("d", u, D),
        weights.linear("e", u, D),
    );
    let mut ehat = vec![0.0f32; edges * D];
    let mut z = vec![0.0f32; rows * D];
    for (r, incoming) in fixture.incoming.iter().enumerate() {
        for j in 0..D {
            let (mut num, mut den) = (0.0f32, 1.0e-6f32);
            for &(s, edge) in incoming {
                let pre = ce[edge * D + j] + du[r * D + j] + eu[s * D + j];
                ehat[edge * D + j] = pre;
                num += sigmoid(pre) * bu[s * D + j];
                den += sigmoid(pre);
            }
            z[r * D + j] = num / den;
        }
    }
    let sum: Vec<f32> = au.iter().zip(&z).map(|(a, z)| a + z).collect();
    let branch = relu(&weights.rms_norm("norm_node", &sum, 1e-5));
    let updated = update.then(|| {
        let step = relu(&weights.rms_norm("norm_edge", &ehat, 1e-5));
        e.iter().zip(&step).map(|(e, s)| e + s).collect()
    });
    (branch, updated)
}

#[test]
fn gated_gcn_matches_the_host_formula() {
    let device = dev();
    for (data, batch) in [
        (five_nodes(false), vec![0u32]),
        (many_graphs(&[6, 4, 7], false, 4), vec![1, 2]),
    ] {
        for update in [true, false] {
            let fixture = Fixture::graphs(&data, &batch, 16, 64);
            let layer = GatedGcn::<R, f32>::new(D, update, 1e-5, &device, &mut Rng::seeded(3));
            assert_eq!(layer.updates_edges(), update);
            let weights = Weights::of(&layer);
            let u_host = Lcg(10).vec(fixture.n_rows * D);
            let e_host = Lcg(11).vec(fixture.n_edges * D);
            let u = Var::constant(tensor(&u_host, vec![fixture.n_rows, D], &device));
            let e = Var::constant(tensor(&e_host, vec![fixture.n_edges, D], &device));
            let (branch, updated) = layer
                .apply(&u, &e, fixture.store.adjacency(), &fixture.rows)
                .unwrap();
            let (want_branch, want_updated) =
                host_gated_gcn(&fixture, &weights, &u_host, &e_host, update);
            assert_close(&branch.to_f32(), &want_branch, 2e-5, "gated branch");
            assert!(branch.to_f32().iter().all(|v| v.is_finite()));
            match (updated, want_updated) {
                (Some(got), Some(want)) => {
                    // Real edge rows only: an absent edge row keeps its state.
                    let real: Vec<usize> = fixture
                        .incoming
                        .iter()
                        .flat_map(|edges| edges.iter().map(|&(_, e)| e))
                        .collect();
                    let got = got.to_f32();
                    for &edge in &real {
                        assert_close(
                            &got[edge * D..(edge + 1) * D],
                            &want[edge * D..(edge + 1) * D],
                            2e-5,
                            "edge update",
                        );
                    }
                }
                (None, None) => {}
                _ => panic!("the edge update is computed exactly when asked for"),
            }
        }
    }
}

/// `⟨J δ, g⟩` by central differences of `f` along `delta`.
fn directional(
    f: &dyn Fn(&[f32]) -> Vec<f32>,
    x: &[f32],
    delta: &[f32],
    g: &[f32],
    h: f32,
) -> f64 {
    let plus: Vec<f32> = x.iter().zip(delta).map(|(x, d)| x + h * d).collect();
    let minus: Vec<f32> = x.iter().zip(delta).map(|(x, d)| x - h * d).collect();
    let (fp, fm) = (f(&plus), f(&minus));
    fp.iter()
        .zip(&fm)
        .zip(g)
        .map(|((p, m), g)| (*p as f64 - *m as f64) / (2.0 * h as f64) * *g as f64)
        .sum()
}

fn assert_vjp(numeric: f64, analytic: f64, what: &str) {
    assert!(
        (numeric - analytic).abs() <= 2e-2 * (1.0 + analytic.abs()),
        "{what}: finite differences give {numeric}, the adjoint {analytic}"
    );
}

#[test]
fn gine_adjoints_match_finite_differences() {
    let device = dev();
    let data = many_graphs(&[6, 4, 7], true, 5);
    let fixture = Fixture::graphs(&data, &[0, 2], 16, 64);
    let (rows, edges) = (fixture.n_rows, fixture.n_edges);
    let adjacency = fixture.store.adjacency();
    let u_host = Lcg(20).vec(rows * D);
    let ee_host = Lcg(21).vec(edges * D);
    let g_host = Lcg(22).vec(rows * D);
    let g = tensor(&g_host, vec![rows, D], &device);
    let u = tensor(&u_host, vec![rows, D], &device);
    let ee = tensor(&ee_host, vec![edges, D], &device);

    for with_edges in [true, false] {
        let ee_opt = with_edges.then_some(&ee);
        let f_u = |x: &[f32]| {
            gine_aggregate(&tensor(x, vec![rows, D], &device), ee_opt, adjacency, &fixture.rows)
                .unwrap()
                .to_f32()
        };
        let delta = Lcg(23).vec(rows * D);
        let du = gine_aggregate_du(&g, &u, ee_opt, adjacency, &fixture.rows)
            .unwrap()
            .to_f32();
        assert_vjp(
            directional(&f_u, &u_host, &delta, &g_host, 1e-3),
            dot(&delta, &du),
            "gine d/du",
        );
        if with_edges {
            let f_ee = |x: &[f32]| {
                gine_aggregate(
                    &u,
                    Some(&tensor(x, vec![edges, D], &device)),
                    adjacency,
                    &fixture.rows,
                )
                .unwrap()
                .to_f32()
            };
            let delta = Lcg(24).vec(edges * D);
            let dee = gine_aggregate_dee(&g, &u, &ee, &fixture.rows)
                .unwrap()
                .to_f32();
            assert_vjp(
                directional(&f_ee, &ee_host, &delta, &g_host, 1e-3),
                dot(&delta, &dee),
                "gine d/dee",
            );
            // Absent edge rows get no gradient.
            let real: std::collections::HashSet<usize> = fixture
                .incoming
                .iter()
                .flat_map(|edges| edges.iter().map(|&(_, e)| e))
                .collect();
            for edge in 0..edges {
                if !real.contains(&edge) {
                    assert!(dee[edge * D..(edge + 1) * D].iter().all(|&v| v == 0.0));
                }
            }
        }
    }
}

#[test]
fn gated_adjoints_match_finite_differences() {
    let device = dev();
    let data = many_graphs(&[6, 4, 7], false, 6);
    let fixture = Fixture::graphs(&data, &[2, 1], 16, 64);
    let (rows, edges) = (fixture.n_rows, fixture.n_edges);
    let adjacency = fixture.store.adjacency();
    let node = |seed: u64| Lcg(seed).vec(rows * D);
    let edge = |seed: u64| Lcg(seed).vec(edges * D);

    // The edge pre-activation is linear: its adjoint is exact.
    let (ce, du, eu) = (edge(30), node(31), node(32));
    let ehat = gated_edge(
        &tensor(&ce, vec![edges, D], &device),
        &tensor(&du, vec![rows, D], &device),
        &tensor(&eu, vec![rows, D], &device),
        &fixture.rows,
    )
    .unwrap();
    let ehat_host = ehat.to_f32();
    let g_edge = edge(33);
    let (d_du, d_eu) = gated_edge_backward(
        &tensor(&g_edge, vec![edges, D], &device),
        adjacency,
        &fixture.rows,
    )
    .unwrap();
    // <ehat, g> = <ce, g> + <du, d_du> + <eu, d_eu> on real edges; absent
    // edge rows are zero in `ehat` and take no part.
    let real: std::collections::HashSet<usize> = fixture
        .incoming
        .iter()
        .flat_map(|edges| edges.iter().map(|&(_, e)| e))
        .collect();
    let masked = |x: &[f32]| -> Vec<f32> {
        x.iter()
            .enumerate()
            .map(|(i, v)| if real.contains(&(i / D)) { *v } else { 0.0 })
            .collect()
    };
    let lhs = dot(&ehat_host, &g_edge);
    let rhs = dot(&masked(&ce), &g_edge) + dot(&du, &d_du.to_f32()) + dot(&eu, &d_eu.to_f32());
    assert!((lhs - rhs).abs() < 1e-3 * (1.0 + lhs.abs()), "{lhs} vs {rhs}");

    // The gated aggregation: adjoints in `ê` and in `b`.
    let b_host = node(34);
    let g_host = node(35);
    let (b, g) = (
        tensor(&b_host, vec![rows, D], &device),
        tensor(&g_host, vec![rows, D], &device),
    );
    let (z, den) = gated_node(&ehat, &b, adjacency, &fixture.rows).unwrap();
    let f_ehat = |x: &[f32]| {
        gated_node(&tensor(x, vec![edges, D], &device), &b, adjacency, &fixture.rows)
            .unwrap()
            .0
            .to_f32()
    };
    let delta = masked(&edge(36));
    let dehat = gated_node_dehat(&g, &b, &z, &den, &ehat, &fixture.rows)
        .unwrap()
        .to_f32();
    assert_vjp(
        directional(&f_ehat, &ehat_host, &delta, &g_host, 1e-3),
        dot(&delta, &dehat),
        "gated d/dehat",
    );
    let f_b = |x: &[f32]| {
        gated_node(&ehat, &tensor(x, vec![rows, D], &device), adjacency, &fixture.rows)
            .unwrap()
            .0
            .to_f32()
    };
    let delta = node(37);
    let db = gated_node_db(&g, &den, &ehat, adjacency, &fixture.rows)
        .unwrap()
        .to_f32();
    assert_vjp(
        directional(&f_b, &b_host, &delta, &g_host, 1e-3),
        dot(&delta, &db),
        "gated d/db",
    );
}

/// The derivative of `loss` in element `index` of the parameter called `name`,
/// by central differences.
///
/// The layers are piecewise linear in their weights (ReLU), so a step that
/// crosses a kink gives a wrong slope however small it is chosen in advance.
/// The step is therefore halved until two consecutive estimates agree: on one
/// side of every kink they do at once, across one they cannot.
fn weight_derivative<M: Module<R, f32>>(
    module: &M,
    name: &str,
    index: usize,
    loss: &dyn Fn() -> f64,
) -> f64 {
    let (_, param) = module
        .named_parameters()
        .into_iter()
        .find(|(n, _)| n == name)
        .unwrap_or_else(|| panic!("no parameter {name}"));
    let original = param.value();
    let mut host = original.to_f32();
    let base = host[index];
    let mut slope = |h: f32| {
        let mut at = |value: f32| {
            host[index] = value;
            param.set(Tensor::from_f32(&host, param.shape(), original.device()).unwrap());
            loss()
        };
        let (plus, minus) = (at(base + h), at(base - h));
        (plus - minus) / (2.0 * h as f64)
    };
    let mut h = 4e-3f32;
    let mut previous = slope(h);
    let mut found = None;
    for _ in 0..7 {
        h *= 0.5;
        let current = slope(h);
        if (current - previous).abs() <= 5e-3 * (1.0 + current.abs()) {
            found = Some(current);
            break;
        }
        previous = current;
    }
    param.set(original);
    found.unwrap_or_else(|| panic!("the finite differences of {name}[{index}] do not settle"))
}

#[test]
fn layer_weights_have_the_gradients_finite_differences_predict() {
    let device = dev();
    for edges in [false, true] {
        let data = many_graphs(&[6, 4, 7], edges, 7);
        let fixture = Fixture::graphs(&data, &[0, 1], 16, 64);
        let (rows, n_edges) = (fixture.n_rows, fixture.n_edges);
        let adjacency = fixture.store.adjacency();
        let u = tensor(&Lcg(40).vec(rows * D), vec![rows, D], &device);
        let w = tensor(&Lcg(41).vec(rows * D), vec![rows, D], &device);

        let gine = Gine::<R, f32>::new(D, edges.then_some(3), &device, &mut Rng::seeded(5));
        let edge_in = edges.then(|| fixture.edge_inputs());
        let w_host = w.to_f32();
        let out = || {
            gine.apply(
                &Var::constant(u.clone()),
                edge_in.clone().map(Var::constant).as_ref(),
                adjacency,
                &fixture.rows,
            )
            .unwrap()
        };
        // The loss the tape differentiates, and the same sum in `f64` for the
        // finite differences.
        let grads = out()
            .mul(&Var::constant(w.clone()))
            .unwrap()
            .sum()
            .unwrap()
            .backward()
            .unwrap();
        let host_loss = || dot(&out().to_f32(), &w_host);
        let mut names = vec![("lin1.weight", 5usize), ("lin2.weight", 11), ("lin1.bias", 2)];
        if edges {
            names.push(("edge.weight", 7));
        }
        for (name, index) in names {
            let param = gine
                .named_parameters()
                .into_iter()
                .find(|(n, _)| n == name)
                .unwrap()
                .1;
            let analytic = grads.get(param.id()).unwrap().to_f32()[index] as f64;
            let numeric = weight_derivative(&gine, name, index, &host_loss);
            assert!(
                (numeric - analytic).abs() <= 3e-2 * (1.0 + analytic.abs()),
                "gine {name}[{index}] (edges = {edges}): numeric {numeric}, analytic {analytic}"
            );
        }

        // GatedGCN with the edge update: the loss reads both outputs.
        let gated = GatedGcn::<R, f32>::new(D, true, 1e-5, &device, &mut Rng::seeded(6));
        let e = tensor(&Lcg(42).vec(n_edges * D), vec![n_edges, D], &device);
        let we = tensor(&Lcg(43).vec(n_edges * D), vec![n_edges, D], &device);
        let we_host = we.to_f32();
        let out = || {
            let (branch, updated) = gated
                .apply(
                    &Var::constant(u.clone()),
                    &Var::constant(e.clone()),
                    adjacency,
                    &fixture.rows,
                )
                .unwrap();
            (branch, updated.unwrap())
        };
        let (branch, updated) = out();
        let grads = branch
            .mul(&Var::constant(w.clone()))
            .unwrap()
            .sum()
            .unwrap()
            .add(
                &updated
                    .mul(&Var::constant(we.clone()))
                    .unwrap()
                    .sum()
                    .unwrap(),
            )
            .unwrap()
            .backward()
            .unwrap();
        let host_loss = || {
            let (branch, updated) = out();
            dot(&branch.to_f32(), &w_host) + dot(&updated.to_f32(), &we_host)
        };
        for (name, index) in [
            ("a.weight", 3usize),
            ("b.weight", 9),
            ("c.weight", 14),
            ("d.weight", 20),
            ("e.weight", 33),
            ("norm_node.weight", 1),
            ("norm_edge.weight", 4),
        ] {
            let param = gated
                .named_parameters()
                .into_iter()
                .find(|(n, _)| n == name)
                .unwrap()
                .1;
            let analytic = grads.get(param.id()).unwrap().to_f32()[index] as f64;
            let numeric = weight_derivative(&gated, name, index, &host_loss);
            assert!(
                (numeric - analytic).abs() <= 3e-2 * (1.0 + analytic.abs()),
                "gated {name}[{index}]: numeric {numeric}, analytic {analytic}"
            );
        }
    }
}

#[test]
fn the_last_gated_layer_has_no_parameter_without_a_gradient() {
    let device = dev();
    let data = many_graphs(&[6, 4, 7], false, 8);
    let fixture = Fixture::graphs(&data, &[0, 1, 2], 32, 128);
    let (rows, edges) = (fixture.n_rows, fixture.n_edges);
    let u = Var::constant(tensor(&Lcg(50).vec(rows * D), vec![rows, D], &device));
    let e = Var::constant(tensor(&Lcg(51).vec(edges * D), vec![edges, D], &device));
    let branch_loss = |layer: &GatedGcn<R, f32>| {
        layer
            .apply(&u, &e, fixture.store.adjacency(), &fixture.rows)
            .unwrap()
            .0
            .sum()
            .unwrap()
            .backward()
            .unwrap()
    };

    let last = GatedGcn::<R, f32>::new(D, false, 1e-5, &device, &mut Rng::seeded(9));
    let (_, updated) = last
        .apply(&u, &e, fixture.store.adjacency(), &fixture.rows)
        .unwrap();
    assert!(updated.is_none(), "the last layer computes no edge update");
    let grads = branch_loss(&last);
    for (name, param) in last.named_parameters() {
        assert!(!name.starts_with("norm_edge"));
        let grad = grads
            .get(param.id())
            .unwrap_or_else(|| panic!("{name} has no gradient"));
        assert!(grad.to_f32().iter().any(|&g| g != 0.0), "{name} has a zero gradient");
    }

    // With the update but only the branch read, the edge norm would be a
    // parameter no gradient reaches: the reason the last layer drops it.
    let inner = GatedGcn::<R, f32>::new(D, true, 1e-5, &device, &mut Rng::seeded(9));
    let grads = branch_loss(&inner);
    let (_, edge_norm) = inner
        .named_parameters()
        .into_iter()
        .find(|(name, _)| name == "norm_edge.weight")
        .unwrap();
    assert!(grads.get(edge_norm.id()).is_none());
}

#[test]
fn node_subsets_skip_neighbours_outside_the_batch() {
    let device = dev();
    let data = many_graphs(&[40], false, 9);
    let fixture = Fixture::subset(&data, 3, 1);
    let rows = fixture.n_rows;
    let adjacency = fixture.store.adjacency();
    // Some neighbours are in the batch and some are not.
    let inside: usize = fixture.incoming.iter().map(Vec::len).sum();
    let gid = fixture.rows.gid().to_vec();
    let all: usize = gid
        .iter()
        .filter(|&&v| v != IGNORE)
        .map(|&v| (fixture.canon.adj_off[v as usize + 1] - fixture.canon.adj_off[v as usize]) as usize)
        .sum();
    assert!(inside > 0 && inside < all, "{inside} of {all} edges stay inside");

    let u_host = Lcg(60).vec(rows * D);
    let u = tensor(&u_host, vec![rows, D], &device);
    let got = gine_aggregate(&u, None, adjacency, &fixture.rows)
        .unwrap()
        .to_f32();
    assert_close(
        &got,
        &host_gine_aggregate(&fixture, &u_host, None, D),
        1e-6,
        "subset aggregation",
    );
    let g_host = Lcg(61).vec(rows * D);
    let g = tensor(&g_host, vec![rows, D], &device);
    let f = |x: &[f32]| {
        gine_aggregate(&tensor(x, vec![rows, D], &device), None, adjacency, &fixture.rows)
            .unwrap()
            .to_f32()
    };
    let delta = Lcg(62).vec(rows * D);
    let du = gine_aggregate_du(&g, &u, None, adjacency, &fixture.rows)
        .unwrap()
        .to_f32();
    assert_vjp(
        directional(&f, &u_host, &delta, &g_host, 1e-3),
        dot(&delta, &du),
        "subset d/du",
    );
    // The host adjoint, with the same edges skipped.
    let mut want = g_host.clone();
    for (r, edges) in fixture.incoming.iter().enumerate() {
        for &(s, _) in edges {
            // Row `r` received from `s`; by symmetry `r` also sent to `s`.
            for j in 0..D {
                if u_host[r * D + j] > 0.0 {
                    want[r * D + j] += g_host[s * D + j];
                }
            }
        }
    }
    assert_close(&du, &want, 1e-5, "subset adjoint");

    // The layer runs in this mode, and refuses edge features it cannot have.
    let layer = Gine::<R, f32>::new(D, None, &device, &mut Rng::seeded(1));
    assert!(
        layer
            .apply(&Var::constant(u.clone()), None, adjacency, &fixture.rows)
            .is_ok()
    );
    let with_edges = Gine::<R, f32>::new(D, Some(3), &device, &mut Rng::seeded(1));
    assert!(
        with_edges
            .apply(&Var::constant(u), None, adjacency, &fixture.rows)
            .is_err()
    );
}

/// `erf` to about 1e-7 (Abramowitz and Stegun 7.1.26).
fn erf(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let poly = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    let y = 1.0 - poly * (-x * x).exp();
    if x < 0.0 { -y } else { y }
}

fn gelu(x: f32) -> f32 {
    (x as f64 * 0.5 * (1.0 + erf(x as f64 / std::f64::consts::SQRT_2))) as f32
}

#[test]
fn the_local_encoder_is_the_host_formula_and_commutes_with_the_embedding() {
    let device = dev();
    let mut data = many_graphs(&[60], false, 12);
    data.x = Features::Float {
        dim: 5,
        data: Lcg(70).vec(60 * 5),
    };
    data.pe = Some((3, Lcg(71).vec(60 * 3)));
    let canon = canonicalize::<f32>(&data.view(), &CanonicalizeOptions::default()).unwrap();
    let store = GraphStore::<R, f32>::upload(canon.clone(), &device).unwrap();
    let rows = batch_rows_subset(store.adjacency(), 1, 0, (0, 0), 1).unwrap();
    let width = 8usize;
    let mut rng = Rng::seeded(4);
    let embed = LinearConfig::new(width, D).init::<R, f32>(&device, &mut rng);
    let encoder = LocalEncoderModule::<R, f32>::new(D, &device, &mut rng);

    for sgc in [true, false] {
        let shape = WalkShape {
            hops: 2,
            walks: 4,
            repeats: 2,
            sgc,
        };
        let (len, cap) = (shape.len(), shape.cap());
        let tokens = walk_tokens(store.adjacency(), &rows, shape, (5, 5), 0).unwrap();
        let features = token_features(&store.x().source(), store.pe(), &tokens, &rows, None).unwrap();
        let got = encoder
            .apply(&embed, &features, tokens.stats())
            .unwrap()
            .to_f32();

        let (node, w, stats) = (
            tokens.node().to_vec(),
            tokens.w().to_f32(),
            tokens.stats().to_f32(),
        );
        // The host formula: aggregate, embed, add the statistics, GELU.
        let FeatureTable::Float { data: x, .. } = &canon.x else {
            panic!("expected float features");
        };
        let pe = &canon.pe.as_ref().unwrap().1;
        let joined = |id: usize| -> Vec<f32> {
            let mut row = x[id * 5..(id + 1) * 5].to_vec();
            row.extend_from_slice(&pe[id * 3..(id + 1) * 3]);
            row
        };
        let n_tokens = 60 * len;
        let mut aggregated = vec![0.0f32; n_tokens * width];
        for token in 0..n_tokens {
            for c in 0..cap {
                let id = node[token * cap + c];
                if id == IGNORE {
                    break;
                }
                for (j, v) in joined(id as usize).iter().enumerate() {
                    aggregated[token * width + j] += w[token * cap + c] * v;
                }
            }
        }
        let embed_w = Weights::of(&embed);
        let enc_w = Weights::of(&encoder);
        let embedded = {
            let weight = embed_w.get("weight");
            let bias = embed_w.get("bias");
            let mut out = vec![0.0f32; n_tokens * D];
            for t in 0..n_tokens {
                for o in 0..D {
                    let mut acc = bias[o];
                    for i in 0..width {
                        acc += aggregated[t * width + i] * weight[i * D + o];
                    }
                    out[t * D + o] = acc;
                }
            }
            out
        };
        let projected = enc_w.linear("stats_proj", &stats, 3);
        let want: Vec<f32> = embedded
            .iter()
            .zip(&projected)
            .map(|(a, b)| gelu(a + b))
            .collect();
        assert_close(&got, &want, 2e-5, "token encoding");

        // Commutation: embedding the aggregate is the aggregate of the embedded
        // nodes, because the weights of a token sum to one.
        let inputs = node_inputs(&store.x().source(), store.pe(), &rows, None).unwrap();
        let per_node = embed.apply(&Var::constant(inputs)).unwrap().to_f32();
        let on_tokens = embed.apply(&Var::constant(features)).unwrap().to_f32();
        let mut mean = vec![0.0f32; n_tokens * D];
        for token in 0..n_tokens {
            for c in 0..cap {
                let id = node[token * cap + c];
                if id == IGNORE {
                    break;
                }
                for j in 0..D {
                    mean[token * D + j] += w[token * cap + c] * per_node[id as usize * D + j];
                }
            }
        }
        assert_close(&on_tokens, &mean, 1e-5, "embed(aggregate) = aggregate(embed)");
    }
}

#[test]
fn token_statistics_separate_tokens_of_different_size() {
    let device = dev();
    // Constant features: only the statistics can tell two tokens apart.
    let mut data = many_graphs(&[50], false, 13);
    data.x = Features::Float {
        dim: 4,
        data: vec![1.0; 50 * 4],
    };
    let canon = canonicalize::<f32>(&data.view(), &CanonicalizeOptions::default()).unwrap();
    let store = GraphStore::<R, f32>::upload(canon, &device).unwrap();
    let rows = batch_rows_subset(store.adjacency(), 1, 0, (0, 0), 1).unwrap();
    let shape = WalkShape {
        hops: 2,
        walks: 6,
        repeats: 1,
        sgc: true,
    };
    let tokens = walk_tokens(store.adjacency(), &rows, shape, (1, 2), 0).unwrap();
    let features = token_features(&store.x().source(), None, &tokens, &rows, None).unwrap();
    assert!(
        features.to_f32().iter().all(|v| (v - 1.0).abs() < 1e-6),
        "every token aggregates to the same features"
    );
    let mut rng = Rng::seeded(2);
    let embed = LinearConfig::new(4, D).init::<R, f32>(&device, &mut rng);
    let encoder = LocalEncoderModule::<R, f32>::new(D, &device, &mut rng);
    let encoded = encoder
        .apply(&embed, &features, tokens.stats())
        .unwrap()
        .to_f32();
    let node = tokens.node().to_vec();
    let size = |token: usize| {
        node[token * shape.cap()..(token + 1) * shape.cap()]
            .iter()
            .filter(|&&s| s != IGNORE)
            .count()
    };
    // The last node (largest degree): its 2-hop token against itself alone.
    let (big, small) = (49 * shape.len(), 49 * shape.len() + 2);
    assert!(size(big) > size(small) && size(small) == 1);
    let differ = (0..D).any(|j| (encoded[big * D + j] - encoded[small * D + j]).abs() > 1e-4);
    assert!(differ, "tokens of different size must not encode alike");
}

fn spec() -> GraphMambaSpec {
    GraphMambaSpec::new(
        FeatureSpec::Float { dim: 6 },
        GraphTaskSpec::NodeClass { classes: 4 },
    )
}

#[test]
fn the_spec_round_trips_through_json() {
    let specs = [
        spec(),
        spec()
            .with_d_model(32)
            .with_tokens(2, 5, 3)
            .with_mpnn(Some(MpnnKind::Gine))
            .with_pe_dim(8)
            .with_pe_sign_flip(Some((2, 8)))
            .with_order(NodeOrder::ppr())
            .with_token_tail(TokenTail::Bidirectional)
            .with_local(LocalEncoder::Mean)
            .with_dropout(0.1)
            .with_seed(9),
        GraphMambaSpec::new(
            FeatureSpec::Categorical {
                vocab: vec![28, 4, 9],
            },
            GraphTaskSpec::GraphRegression {
                targets: 11,
                pool: GraphPool::Sum,
                loss: RegressionLoss::Mse,
            },
        )
        .with_tokens(0, 1, 1)
        .with_mpnn(Some(MpnnKind::GatedGcn))
        .with_edge_features(Some(FeatureSpec::Categorical { vocab: vec![4] }))
        .with_direction(ScanDirection::Forward),
    ];
    for original in specs {
        original.validate().unwrap();
        let json = original.to_json().unwrap();
        assert_eq!(GraphMambaSpec::from_json(&json).unwrap(), original);
    }
    assert_eq!(spec().tokens.len(), 17);
    assert_eq!(spec().tokens.cap(), 33);
    assert_eq!(spec().input_width(), 6);
    assert_eq!(spec().with_d_model(32).token_ssm.head_dim, 32);
    assert!(spec().with_tokens(0, 1, 1).walk_shape().is_none());
    assert!(spec().walk_shape().unwrap().sgc);
    // One head per direction as wide as the model: the widths of §2.7.
    assert_eq!(spec().token_ssm.in_proj_width(), 150);
    // A malformed or invalid document is an error, not a default.
    assert!(GraphMambaSpec::from_json("{").is_err());
    let invalid = spec().with_node_layers(0);
    let json = serde_json::to_string(&invalid).unwrap();
    assert!(GraphMambaSpec::from_json(&json).is_err());
}

#[test]
fn every_spec_error_names_its_field() {
    let fails = |spec: GraphMambaSpec, name: &str| {
        let text = spec.validate().unwrap_err().to_string();
        assert!(text.contains(name), "expected `{name}` in: {text}");
    };
    fails(spec().with_d_model(0), "d_model");
    let mut s = spec();
    s.token_layers = 2;
    s.tokens.max_hops = 0;
    fails(s, "token_layers");
    fails(spec().with_token_layers(0), "token_layers");
    let mut s = spec();
    s.tokens.walks = 0;
    fails(s, "tokens.walks");
    let mut s = spec();
    s.tokens.repeats = 0;
    fails(s, "tokens.repeats");
    fails(spec().with_tokens(8, 16, 1), "tokens");
    fails(spec().with_local(LocalEncoder::Sgc { hops: 2 }), "local");
    fails(spec().with_local(LocalEncoder::Mpnn { layers: 1 }), "local");
    fails(
        spec().with_edge_features(Some(FeatureSpec::Float { dim: 2 })),
        "edge_features",
    );
    fails(
        spec()
            .with_mpnn(Some(MpnnKind::Gine))
            .with_edge_features(Some(FeatureSpec::Float { dim: 0 })),
        "edge_features",
    );
    fails(spec().with_node_layers(0), "node_layers");
    fails(spec().with_node_sequences(0), "node_sequences");
    fails(spec().with_d_state(7), "token_ssm");
    let mut s = spec();
    s.node_ssm.d_state = 7;
    fails(s, "node_ssm");
    let mut s = spec();
    s.node_ssm.d_model = 32;
    fails(s, "node_ssm");
    let mut s = spec();
    s.node_ssm.post_gate_norm = true;
    fails(s, "node_ssm");
    let mut s = spec().with_token_tail(TokenTail::Bidirectional);
    s.token_ssm.post_gate_norm = true;
    fails(s, "token_ssm");
    // A forward tail alone may keep its post-gate norm.
    let mut s = spec();
    s.token_ssm.post_gate_norm = true;
    s.validate().unwrap();
    fails(spec().with_dropout(1.0), "dropout");
    fails(spec().with_row_quantum(0), "row_quantum");
    fails(spec().with_pe_sign_flip(Some((0, 2))), "pe_sign_flip");
    fails(spec().with_pe_dim(4).with_pe_sign_flip(Some((3, 3))), "pe_sign_flip");
    let mut s = spec();
    s.norm_eps = 0.0;
    fails(s, "norm_eps");
    let mut s = spec();
    s.task = GraphTaskSpec::NodeClass { classes: 1 };
    fails(s, "task");
    let mut s = spec();
    s.task = GraphTaskSpec::GraphMultiLabel {
        labels: 0,
        pool: GraphPool::Mean,
    };
    fails(s, "task");
    let mut s = spec();
    s.node_features = FeatureSpec::Float { dim: 0 };
    fails(s, "node_features");
    let mut s = spec();
    s.node_features = FeatureSpec::Categorical {
        vocab: vec![600, 500],
    };
    fails(s, "node_features");
    let mut s = spec();
    s.node_features = FeatureSpec::Categorical { vocab: vec![3, 0] };
    fails(s, "node_features");

    // What depends on how batches are cut.
    let fails_for = |spec: GraphMambaSpec, mode: BatchMode, name: &str| {
        let text = spec.validate_for(mode).unwrap_err().to_string();
        assert!(text.contains(name), "expected `{name}` in: {text}");
    };
    spec().validate_for(BatchMode::Graphs).unwrap();
    spec().validate_for(BatchMode::NodeSubset).unwrap();
    fails_for(spec().with_node_sequences(4), BatchMode::Graphs, "node_sequences");
    spec()
        .with_node_sequences(4)
        .validate_for(BatchMode::NodeSubset)
        .unwrap();
    fails_for(
        spec().with_mpnn(Some(MpnnKind::GatedGcn)),
        BatchMode::NodeSubset,
        "mpnn",
    );
    fails_for(
        spec()
            .with_mpnn(Some(MpnnKind::Gine))
            .with_edge_features(Some(FeatureSpec::Float { dim: 2 })),
        BatchMode::NodeSubset,
        "edge_features",
    );
    let mut s = spec();
    s.task = GraphTaskSpec::GraphClass {
        classes: 3,
        pool: GraphPool::Mean,
    };
    fails_for(s, BatchMode::NodeSubset, "task");
}
