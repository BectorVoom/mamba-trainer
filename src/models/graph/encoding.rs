//! Positional and structural encodings (GRAPH_MAMBA_PLAN.md §2.3, GM3).
//!
//! Both run once, on the host, when a dataset is prepared; their result is a
//! `[n_nodes, k]` table in the caller's node order, written into
//! [`crate::models::graph::GraphData::pe`] and uploaded with everything else.
//!
//! * [`rwse`] — random-walk structural encoding: for each node, the
//!   probability that a walk returns to it after `1, …, k` steps.
//! * [`laplacian_pe`] — the eigenvectors of the symmetric normalised Laplacian
//!   with the smallest non-zero eigenvalues, per graph, by a dense Jacobi
//!   eigensolver. Dense, so limited to small graphs; larger ones use RWSE.

use crate::error::{Error, Result};
use crate::models::graph::data::{GraphDataView, HostCsr};

/// Entries a node's `k`-hop ball may hold before [`rwse`] refuses.
pub const RWSE_MAX_BALL: usize = 200_000;

/// Nodes a graph may have before [`laplacian_pe`] refuses.
pub const LAPLACIAN_MAX_NODES: usize = 2048;

fn threads() -> usize {
    std::thread::available_parallelism().map_or(1, |n| n.get())
}

/// `graph_ptr` of a view as `usize` offsets (`[0, n]` when absent). The CSR
/// builder has already validated it.
fn graph_offsets(view: &GraphDataView<'_>, csr: &HostCsr) -> Vec<usize> {
    let n = csr.n_nodes();
    // Re-derive the graphs from the view through an order-preserving
    // canonicalisation would be a detour; the offsets are small, so read them.
    use crate::models::graph::data::Ints;
    macro_rules! read {
        ($s:expr) => {
            $s.iter().map(|&v| v as usize).collect::<Vec<usize>>()
        };
    }
    match view.graph_ptr {
        None => vec![0, n],
        Some(Ints::I8(s)) => read!(s),
        Some(Ints::I16(s)) => read!(s),
        Some(Ints::I32(s)) => read!(s),
        Some(Ints::I64(s)) => read!(s),
        Some(Ints::U8(s)) => read!(s),
        Some(Ints::U16(s)) => read!(s),
        Some(Ints::U32(s)) => read!(s),
        Some(Ints::U64(s)) => read!(s),
    }
}

/// Random-walk structural encoding with the default ball limit
/// ([`RWSE_MAX_BALL`]); see [`rwse_with`].
pub fn rwse(view: &GraphDataView<'_>, k: usize) -> Result<Vec<f32>> {
    rwse_with(view, k, RWSE_MAX_BALL)
}

/// `[n_nodes, k]`: column `j − 1` of row `v` is the diagonal entry
/// `((D⁻¹A)^j)[v, v]`, the probability that a `j`-step random walk on the
/// symmetrised graph that starts at `v` ends at `v`.
///
/// Exact: a sparse indicator vector is propagated from every node, in parallel
/// over nodes. A node whose `k`-hop ball holds more than `max_ball` nodes is
/// refused, since the propagation would touch the whole graph from every node.
pub fn rwse_with(view: &GraphDataView<'_>, k: usize, max_ball: usize) -> Result<Vec<f32>> {
    rwse_csr(&HostCsr::from_view(view, true)?, k, max_ball)
}

/// [`rwse_with`] on a graph already in CSR form (symmetrised): the part that
/// reads none of the caller's arrays.
pub fn rwse_csr(csr: &HostCsr, k: usize, max_ball: usize) -> Result<Vec<f32>> {
    let n = csr.n_nodes();
    let mut out = vec![0.0f32; n * k];
    if k == 0 {
        return Ok(out);
    }
    let workers = threads().min(n).max(1);
    let chunk = n.div_ceil(workers);
    let failed = std::sync::Mutex::new(None::<(usize, usize)>);
    std::thread::scope(|scope| {
        for (w, rows) in out.chunks_mut(chunk * k).enumerate() {
            let failed = &failed;
            scope.spawn(move || {
                // Dense scratch per worker, cleared through the touched lists.
                let mut mass = vec![0.0f64; n];
                let mut next = vec![0.0f64; n];
                let mut touched: Vec<u32> = Vec::new();
                let mut next_touched: Vec<u32> = Vec::new();
                for (i, row) in rows.chunks_mut(k).enumerate() {
                    let v = w * chunk + i;
                    mass[v] = 1.0;
                    touched.push(v as u32);
                    for slot in row.iter_mut() {
                        for &u in &touched {
                            let neighbours = csr.row(u as usize);
                            if neighbours.is_empty() {
                                continue;
                            }
                            let share = mass[u as usize] / neighbours.len() as f64;
                            for &c in neighbours {
                                if next[c as usize] == 0.0 {
                                    next_touched.push(c);
                                }
                                next[c as usize] += share;
                            }
                        }
                        for &u in &touched {
                            mass[u as usize] = 0.0;
                        }
                        core::mem::swap(&mut mass, &mut next);
                        core::mem::swap(&mut touched, &mut next_touched);
                        next_touched.clear();
                        if touched.len() > max_ball {
                            let mut slot = failed.lock().expect("no panic holds the lock");
                            slot.get_or_insert((v, touched.len()));
                            break;
                        }
                        *slot = mass[v] as f32;
                    }
                    for &u in &touched {
                        mass[u as usize] = 0.0;
                    }
                    touched.clear();
                }
            });
        }
    });
    if let Some((node, ball)) = failed.into_inner().expect("no panic holds the lock") {
        return Err(Error::config(format!(
            "rwse: the {k}-hop neighbourhood of node {node} holds {ball} nodes, above the limit \
             of {max_ball}; use a smaller k"
        )));
    }
    Ok(out)
}

/// The eigenvectors [`laplacian_pe_with`] returns, with their eigenvalues.
#[derive(Debug, Clone, PartialEq)]
pub struct LaplacianPe {
    /// `[n_nodes, k]` eigenvector entries, zero where a graph has fewer than
    /// `k` non-trivial eigenvectors.
    pub vectors: Vec<f32>,
    /// `[n_graphs, k]` eigenvalues, ascending; `NaN` where padded.
    pub values: Vec<f32>,
}

/// Laplacian positional encoding with the default size limit
/// ([`LAPLACIAN_MAX_NODES`]); see [`laplacian_pe_with`].
pub fn laplacian_pe(view: &GraphDataView<'_>, k: usize) -> Result<Vec<f32>> {
    Ok(laplacian_pe_with(view, k, LAPLACIAN_MAX_NODES)?.vectors)
}

/// The `k` eigenvectors of each graph's symmetric normalised Laplacian
/// `I − D^{-1/2} A D^{-1/2}` with the smallest non-zero eigenvalues, ascending.
///
/// Self-loops are ignored and an isolated node keeps a diagonal of one. The
/// trivial eigenvectors (eigenvalue zero, one per connected component) are
/// skipped; each returned vector has unit norm and its largest entry positive.
/// A graph with fewer than `k` non-trivial eigenvectors is padded with zero
/// columns. Graphs above `max_dense_nodes` are refused: the solver is dense
/// and cubic.
pub fn laplacian_pe_with(
    view: &GraphDataView<'_>,
    k: usize,
    max_dense_nodes: usize,
) -> Result<LaplacianPe> {
    let csr = HostCsr::from_view(view, true)?;
    let ptr = graph_offsets(view, &csr);
    laplacian_pe_csr(&csr, &ptr, k, max_dense_nodes)
}

/// The `[n_graphs + 1]` node offsets of a view, as [`laplacian_pe_csr`] takes
/// them. The view must already have been validated (building its CSR does).
pub fn graph_offsets_of(view: &GraphDataView<'_>, csr: &HostCsr) -> Vec<usize> {
    graph_offsets(view, csr)
}

/// [`laplacian_pe_with`] on a graph already in CSR form (symmetrised), with
/// its `[n_graphs + 1]` node offsets: the part that reads none of the caller's
/// arrays.
pub fn laplacian_pe_csr(
    csr: &HostCsr,
    ptr: &[usize],
    k: usize,
    max_dense_nodes: usize,
) -> Result<LaplacianPe> {
    let n = csr.n_nodes();
    let n_graphs = ptr.len() - 1;
    if let Some(g) = (0..n_graphs).find(|&g| ptr[g + 1] - ptr[g] > max_dense_nodes) {
        return Err(Error::config(format!(
            "laplacian_pe: graph {g} has {} nodes, above the dense eigensolver's limit of \
             {max_dense_nodes}; use rwse for large graphs",
            ptr[g + 1] - ptr[g]
        )));
    }
    let mut vectors = vec![0.0f32; n * k];
    let mut values = vec![f32::NAN; n_graphs * k];
    if k == 0 {
        return Ok(LaplacianPe { vectors, values });
    }

    // One graph is one job; graphs are dealt to the workers round-robin and
    // each worker returns its graphs' columns.
    let workers = threads().min(n_graphs).max(1);
    let results: Vec<Vec<(usize, Vec<f64>, Vec<f64>)>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..workers)
            .map(|w| {
                scope.spawn(move || {
                    (w..n_graphs)
                        .step_by(workers)
                        .map(|g| {
                            let (vals, vecs) = graph_eigenvectors(csr, ptr[g], ptr[g + 1], k);
                            (g, vals, vecs)
                        })
                        .collect()
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("the eigensolver does not panic"))
            .collect()
    });
    for (g, vals, vecs) in results.into_iter().flatten() {
        let size = ptr[g + 1] - ptr[g];
        let found = vals.len();
        for (j, &value) in vals.iter().enumerate() {
            values[g * k + j] = value as f32;
            for i in 0..size {
                vectors[(ptr[g] + i) * k + j] = vecs[j * size + i] as f32;
            }
        }
        debug_assert!(found <= k);
    }
    Ok(LaplacianPe { vectors, values })
}

/// Up to `k` non-trivial eigenpairs of the graph on nodes `a..b`: the
/// eigenvalues ascending and the eigenvectors, one after the other.
fn graph_eigenvectors(csr: &HostCsr, a: usize, b: usize, k: usize) -> (Vec<f64>, Vec<f64>) {
    let n = b - a;
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    let degree: Vec<f64> = (a..b)
        .map(|u| csr.row(u).iter().filter(|&&c| c as usize != u).count() as f64)
        .collect();
    let mut matrix = vec![0.0f64; n * n];
    for i in 0..n {
        matrix[i * n + i] = 1.0;
        for &c in csr.row(a + i) {
            let j = c as usize - a;
            if j != i {
                matrix[i * n + j] = -1.0 / (degree[i] * degree[j]).sqrt();
            }
        }
    }
    let (eigenvalues, eigenvectors) = jacobi_eigh(&mut matrix, n);
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&x, &y| eigenvalues[x].total_cmp(&eigenvalues[y]));

    let mut vals = Vec::with_capacity(k);
    let mut vecs = Vec::with_capacity(k * n);
    for &col in &order {
        if vals.len() == k {
            break;
        }
        // The trivial eigenvectors: one per connected component.
        if eigenvalues[col] < 1e-8 {
            continue;
        }
        let column: Vec<f64> = (0..n).map(|i| eigenvectors[i * n + col]).collect();
        // A deterministic sign: the largest entry is positive.
        let peak = column
            .iter()
            .copied()
            .fold(0.0f64, |m, v| if v.abs() > m.abs() { v } else { m });
        let sign = if peak < 0.0 { -1.0 } else { 1.0 };
        vals.push(eigenvalues[col]);
        vecs.extend(column.iter().map(|v| v * sign));
    }
    (vals, vecs)
}

/// Eigenvalues and eigenvectors of the symmetric `n × n` matrix `a` (row-major,
/// destroyed) by cyclic Jacobi rotations. The eigenvectors are the columns of
/// the returned row-major matrix, in the order of the eigenvalues.
fn jacobi_eigh(a: &mut [f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut v = vec![0.0f64; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    let scale: f64 = a.iter().map(|x| x * x).sum::<f64>().max(f64::MIN_POSITIVE);
    for _sweep in 0..64 {
        let mut off = 0.0f64;
        for p in 0..n {
            for q in p + 1..n {
                off += a[p * n + q] * a[p * n + q];
            }
        }
        if off <= 1e-22 * scale {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                let apq = a[p * n + q];
                if apq.abs() < 1e-300 {
                    continue;
                }
                let theta = (a[q * n + q] - a[p * n + p]) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for i in 0..n {
                    let (aip, aiq) = (a[i * n + p], a[i * n + q]);
                    a[i * n + p] = c * aip - s * aiq;
                    a[i * n + q] = s * aip + c * aiq;
                }
                for j in 0..n {
                    let (apj, aqj) = (a[p * n + j], a[q * n + j]);
                    a[p * n + j] = c * apj - s * aqj;
                    a[q * n + j] = s * apj + c * aqj;
                }
                for i in 0..n {
                    let (vip, viq) = (v[i * n + p], v[i * n + q]);
                    v[i * n + p] = c * vip - s * viq;
                    v[i * n + q] = s * vip + c * viq;
                }
            }
        }
    }
    ((0..n).map(|i| a[i * n + i]).collect(), v)
}
