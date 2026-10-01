//! GM3: positional and structural encodings against dense references.
//!
//! Host-only: nothing here touches a device.

use mamba3::models::graph::{
    Features, GraphData, laplacian_pe, laplacian_pe_with, rwse, rwse_with,
};

fn undirected(pairs: &[(u32, u32)]) -> (Vec<u32>, Vec<u32>) {
    let mut src = Vec::new();
    let mut dst = Vec::new();
    for &(a, b) in pairs {
        src.extend([a, b]);
        dst.extend([b, a]);
    }
    (src, dst)
}

fn graph(n: usize, pairs: &[(u32, u32)]) -> GraphData {
    let (src, dst) = undirected(pairs);
    GraphData::new(
        n,
        src,
        dst,
        Features::Float {
            dim: 1,
            data: vec![0.0; n],
        },
    )
}

/// Dense adjacency of the symmetrised graph, self-loops kept.
fn dense_adjacency(data: &GraphData) -> Vec<f64> {
    let n = data.n_nodes;
    let mut a = vec![0.0f64; n * n];
    for (&s, &d) in data.edge_src.iter().zip(&data.edge_dst) {
        a[s as usize * n + d as usize] = 1.0;
        a[d as usize * n + s as usize] = 1.0;
    }
    a
}

fn matmul(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; n * n];
    for i in 0..n {
        for l in 0..n {
            let x = a[i * n + l];
            if x != 0.0 {
                for j in 0..n {
                    out[i * n + j] += x * b[l * n + j];
                }
            }
        }
    }
    out
}

/// Diagonals of `(D⁻¹A)^j`, `j = 1..=k`, by dense matrix powers.
fn rwse_reference(data: &GraphData, k: usize) -> Vec<f32> {
    let n = data.n_nodes;
    let a = dense_adjacency(data);
    let mut p = vec![0.0f64; n * n];
    for i in 0..n {
        let degree: f64 = a[i * n..(i + 1) * n].iter().sum();
        if degree > 0.0 {
            for j in 0..n {
                p[i * n + j] = a[i * n + j] / degree;
            }
        }
    }
    let mut out = vec![0.0f32; n * k];
    let mut power = p.clone();
    for j in 0..k {
        for i in 0..n {
            out[i * k + j] = power[i * n + i] as f32;
        }
        power = matmul(&power, &p, n);
    }
    out
}

fn assert_close(actual: &[f32], expected: &[f32], tol: f32, what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!((a - e).abs() <= tol, "{what}: index {i} got {a}, want {e}");
    }
}

#[test]
fn rwse_of_a_triangle_and_a_four_cycle() {
    let triangle = graph(3, &[(0, 1), (1, 2), (2, 0)]);
    let got = rwse(&triangle.view(), 5).unwrap();
    for v in 0..3 {
        assert_close(
            &got[v * 5..(v + 1) * 5],
            &[0.0, 0.5, 0.25, 0.375, 0.3125],
            1e-6,
            "triangle",
        );
    }
    assert_close(&got, &rwse_reference(&triangle, 5), 1e-6, "triangle reference");

    let cycle = graph(4, &[(0, 1), (1, 2), (2, 3), (3, 0)]);
    let got = rwse(&cycle.view(), 6).unwrap();
    for v in 0..4 {
        for j in [0usize, 2, 4] {
            assert_eq!(got[v * 6 + j], 0.0, "an odd walk on a bipartite graph cannot return");
        }
        assert!((got[v * 6 + 1] - 0.5).abs() < 1e-6);
    }
    assert_close(&got, &rwse_reference(&cycle, 6), 1e-6, "cycle reference");
}

#[test]
fn rwse_of_an_irregular_graph_with_an_isolated_node() {
    // Two graphs, a pendant path and a node with no edge at all.
    let mut data = graph(
        9,
        &[(0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (5, 6), (6, 7), (7, 5), (5, 7)],
    );
    data.graph_ptr = vec![0, 5, 9];
    let got = rwse(&data.view(), 8).unwrap();
    assert_close(&got, &rwse_reference(&data, 8), 1e-5, "irregular graph");
    assert!(got[8 * 8..].iter().all(|&v| v == 0.0), "an isolated node never returns");
    assert_eq!(rwse(&data.view(), 0).unwrap().len(), 0);
}

#[test]
fn rwse_refuses_a_ball_that_is_too_large() {
    let star: Vec<(u32, u32)> = (1..50).map(|leaf| (0, leaf)).collect();
    let data = graph(50, &star);
    let err = rwse_with(&data.view(), 3, 20).unwrap_err().to_string();
    assert!(err.contains("rwse") && err.contains("smaller k"), "{err}");
    assert!(rwse_with(&data.view(), 3, 50).is_ok());
}

/// The symmetric normalised Laplacian of nodes `a..b`, dense.
fn laplacian(data: &GraphData, a: usize, b: usize) -> Vec<f64> {
    let n = b - a;
    let full = data.n_nodes;
    let adjacency = dense_adjacency(data);
    let at = |i: usize, j: usize| if i == j { 0.0 } else { adjacency[(a + i) * full + a + j] };
    let degree: Vec<f64> = (0..n).map(|i| (0..n).map(|j| at(i, j)).sum()).collect();
    let mut l = vec![0.0f64; n * n];
    for i in 0..n {
        l[i * n + i] = 1.0;
        for j in 0..n {
            if at(i, j) != 0.0 {
                l[i * n + j] = -1.0 / (degree[i] * degree[j]).sqrt();
            }
        }
    }
    l
}

/// Check the eigenpairs of graph `g` (nodes `a..b`): residuals, order, unit
/// norm, mutual orthogonality, no trivial vector. Returns how many are real.
fn check_graph(
    data: &GraphData,
    vectors: &[f32],
    values: &[f32],
    g: usize,
    a: usize,
    b: usize,
    k: usize,
) -> usize {
    let n = b - a;
    let l = laplacian(data, a, b);
    let mut found = 0;
    let mut previous = 0.0f32;
    let column = |j: usize| -> Vec<f64> { (0..n).map(|i| vectors[(a + i) * k + j] as f64).collect() };
    for j in 0..k {
        let value = values[g * k + j];
        let v = column(j);
        if value.is_nan() {
            assert!(v.iter().all(|&x| x == 0.0), "a padded column is zero");
            continue;
        }
        assert_eq!(found, j, "real columns come first");
        found += 1;
        assert!(value > 1e-6, "the trivial eigenvector is excluded (got {value})");
        assert!(value >= previous - 1e-6, "eigenvalues ascend");
        previous = value;
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4, "unit norm, got {norm}");
        let mut residual = 0.0f64;
        for i in 0..n {
            let lv: f64 = (0..n).map(|m| l[i * n + m] * v[m]).sum();
            residual += (lv - value as f64 * v[i]).powi(2);
        }
        assert!(residual.sqrt() < 1e-4, "‖Lv − λv‖ = {}", residual.sqrt());
        for other in 0..j {
            let w = column(other);
            let dot: f64 = v.iter().zip(&w).map(|(x, y)| x * y).sum();
            assert!(dot.abs() < 1e-4, "columns {other} and {j} are not orthogonal: {dot}");
        }
    }
    found
}

#[test]
fn laplacian_pe_eigenpairs_of_two_graphs() {
    // Graph 0: a 6-cycle with a chord. Graph 1: a path of four and, apart
    // from it, a triangle (two connected components).
    let mut data = graph(
        13,
        &[
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 0),
            (0, 3),
            (6, 7),
            (7, 8),
            (8, 9),
            (10, 11),
            (11, 12),
            (12, 10),
        ],
    );
    data.graph_ptr = vec![0, 6, 13];
    let k = 4;
    let pe = laplacian_pe_with(&data.view(), k, 2048).unwrap();
    assert_eq!(pe.vectors.len(), 13 * k);
    assert_eq!(pe.values.len(), 2 * k);
    assert_eq!(check_graph(&data, &pe.vectors, &pe.values, 0, 0, 6, k), 4);
    assert_eq!(check_graph(&data, &pe.vectors, &pe.values, 1, 6, 13, k), 4);
    assert_eq!(laplacian_pe(&data.view(), k).unwrap(), pe.vectors);

    // Asking for every non-trivial vector: 5 of graph 0; graph 1 has two
    // components, so two trivial vectors and 5 of 7 left.
    let k = 6;
    let pe = laplacian_pe_with(&data.view(), k, 2048).unwrap();
    assert_eq!(check_graph(&data, &pe.vectors, &pe.values, 0, 0, 6, k), 5);
    assert_eq!(check_graph(&data, &pe.vectors, &pe.values, 1, 6, 13, k), 5);
}

#[test]
fn laplacian_pe_pads_small_graphs_and_refuses_large_ones() {
    // A single edge has one non-trivial eigenvector; a lone node has none.
    let mut data = graph(3, &[(0, 1)]);
    data.graph_ptr = vec![0, 2, 3];
    let k = 3;
    let pe = laplacian_pe_with(&data.view(), k, 2048).unwrap();
    assert_eq!(check_graph(&data, &pe.vectors, &pe.values, 0, 0, 2, k), 1);
    assert!((pe.values[0] - 2.0).abs() < 1e-5, "K2 has eigenvalues 0 and 2");
    // The lone node's Laplacian is the 1 × 1 identity: one eigenvalue of one.
    assert_eq!(check_graph(&data, &pe.vectors, &pe.values, 1, 2, 3, k), 1);
    assert!(pe.vectors[6..].iter().skip(1).all(|&v| v == 0.0));

    let path: Vec<(u32, u32)> = (0..39).map(|i| (i, i + 1)).collect();
    let data = graph(40, &path);
    let err = laplacian_pe_with(&data.view(), 4, 32).unwrap_err().to_string();
    assert!(err.contains("laplacian_pe") && err.contains("rwse"), "{err}");
    let pe = laplacian_pe_with(&data.view(), 4, 40).unwrap();
    assert_eq!(check_graph(&data, &pe.vectors, &pe.values, 0, 0, 40, 4), 4);
    // The Fiedler value of a path of n nodes under the normalised Laplacian is
    // 1 − cos(π / (n − 1)).
    let fiedler = 1.0 - (std::f32::consts::PI / 39.0).cos();
    assert!((pe.values[0] - fiedler).abs() < 1e-5, "{} vs {fiedler}", pe.values[0]);
}

#[test]
fn encodings_validate_their_input() {
    let mut data = graph(3, &[(0, 1)]);
    data.edge_src.push(7);
    data.edge_dst.push(0);
    assert!(rwse(&data.view(), 2).unwrap_err().to_string().contains("edge_index"));
    assert!(
        laplacian_pe(&data.view(), 2)
            .unwrap_err()
            .to_string()
            .contains("edge_index")
    );
}
