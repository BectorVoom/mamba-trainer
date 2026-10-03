//! Induced-subgraph containment of contract §7.3 (the primary objective).
//!
//! A candidate graph is correct when an injective map of its atoms into the
//! parent's preserves atom type and bond presence with order: two candidate
//! atoms are bonded with order `b` exactly when their images are. A parent
//! bond between two images that the candidate lacks is therefore a mismatch
//! (induced), not an extra that can be ignored.

use super::graph::MolGraph;

/// Outcome of [`contains_induced`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Containment {
    /// An induced type- and bond-preserving injection exists.
    Contained,
    /// No such injection exists (proven exhaustively within the work limit).
    NotContained,
    /// The work limit was spent before the search finished: never a guess.
    WorkLimit,
}

/// Whether `candidate` is an induced labeled subgraph of `parent`.
///
/// The search backtracks in a connectivity order of the candidate (each next
/// atom is adjacent to an already-mapped one where the candidate's
/// connectivity allows), pruned by atom type and by degree (a candidate atom
/// needs `candidate degree <= parent degree`, which any induced image
/// satisfies). `work_limit` counts extension attempts: trying one more
/// parent image for the next candidate atom. Exceeding it returns
/// [`Containment::WorkLimit`]. An empty candidate is
/// [`Containment::NotContained`]: the recipe never emits one and the empty
/// map would carry no objective.
pub fn contains_induced(parent: &MolGraph, candidate: &MolGraph, work_limit: usize) -> Containment {
    let p_atoms = parent.atoms();
    let c_atoms = candidate.atoms();
    let n_p = p_atoms.len();
    let n_c = c_atoms.len();
    if n_c == 0 {
        return Containment::NotContained;
    }
    if n_c > n_p {
        return Containment::NotContained;
    }
    // Adjacency as order maps (0 means no bond), plus neighbour counts.
    let mut p_adj = vec![vec![0u8; n_p]; n_p];
    for (a, b, o) in parent.bonds() {
        p_adj[*a][*b] = *o;
        p_adj[*b][*a] = *o;
    }
    let mut c_adj = vec![vec![0u8; n_c]; n_c];
    for (a, b, o) in candidate.bonds() {
        c_adj[*a][*b] = *o;
        c_adj[*b][*a] = *o;
    }
    let p_deg: Vec<usize> = (0..n_p)
        .map(|i| p_adj[i].iter().filter(|&&o| o != 0).count())
        .collect();
    let c_deg: Vec<usize> = (0..n_c)
        .map(|i| c_adj[i].iter().filter(|&&o| o != 0).count())
        .collect();
    // Type-count reject: a type the candidate uses more often than the parent
    // has can never inject. This is exact, not a heuristic.
    {
        use std::collections::BTreeMap;
        let mut need: BTreeMap<u8, usize> = BTreeMap::new();
        let mut have: BTreeMap<u8, usize> = BTreeMap::new();
        for t in c_atoms {
            *need.entry(*t).or_default() += 1;
        }
        for t in p_atoms {
            *have.entry(*t).or_default() += 1;
        }
        for (t, n) in &need {
            if have.get(t).copied().unwrap_or(0) < *n {
                return Containment::NotContained;
            }
        }
    }
    // Connectivity order: breadth-first from atom 0, each next atom adjacent
    // to an earlier one where the remaining graph allows; a disconnected
    // component restarts from its smallest unvisited atom.
    let mut order: Vec<usize> = Vec::with_capacity(n_c);
    {
        let mut visited = vec![false; n_c];
        while order.len() < n_c {
            let start = (0..n_c).find(|&i| !visited[i]).expect("unvisited remains");
            visited[start] = true;
            order.push(start);
            // Grow the visited set while some unvisited atom touches it.
            loop {
                let mut next: Option<usize> = None;
                for c in 0..n_c {
                    if visited[c] {
                        continue;
                    }
                    let touches = (0..n_c).any(|d| visited[d] && c_adj[c][d] != 0);
                    if touches {
                        next = Some(c);
                        break;
                    }
                }
                match next {
                    Some(v) => {
                        visited[v] = true;
                        order.push(v);
                    }
                    None => break,
                }
            }
        }
    }
    let mut mapping: Vec<Option<usize>> = vec![None; n_c];
    let mut used = vec![false; n_p];
    let mut attempts: usize = 0;
    let mut over_budget = false;
    let found = dfs(
        0,
        &order,
        p_atoms,
        c_atoms,
        &p_adj,
        &c_adj,
        &p_deg,
        &c_deg,
        &mut mapping,
        &mut used,
        &mut attempts,
        work_limit,
        &mut over_budget,
    );
    if over_budget {
        Containment::WorkLimit
    } else if found {
        Containment::Contained
    } else {
        Containment::NotContained
    }
}

/// Depth-first backtracking over the connectivity order.
///
/// `mapping[d]` is the parent image of `order[d]` for `d < depth`.
#[allow(clippy::too_many_arguments)]
fn dfs(
    depth: usize,
    order: &[usize],
    p_atoms: &[u8],
    c_atoms: &[u8],
    p_adj: &[Vec<u8>],
    c_adj: &[Vec<u8>],
    p_deg: &[usize],
    c_deg: &[usize],
    mapping: &mut [Option<usize>],
    used: &mut [bool],
    attempts: &mut usize,
    work_limit: usize,
    over_budget: &mut bool,
) -> bool {
    if *over_budget {
        return false;
    }
    if depth == order.len() {
        return true;
    }
    let c = order[depth];
    for p in 0..p_atoms.len() {
        if used[p] {
            continue;
        }
        if p_atoms[p] != c_atoms[c] {
            continue;
        }
        if c_deg[c] > p_deg[p] {
            continue;
        }
        // Induced bond check against every already-mapped atom: the
        // candidate order and the parent order must agree exactly,
        // presence and bond order both.
        let mut ok = true;
        for d in 0..depth {
            let c2 = order[d];
            let p2 = mapping[d].expect("earlier depths are mapped");
            if c_adj[c][c2] != p_adj[p][p2] {
                ok = false;
                break;
            }
        }
        if !ok {
            continue;
        }
        *attempts += 1;
        if *attempts > work_limit {
            *over_budget = true;
            return false;
        }
        mapping[depth] = Some(p);
        used[p] = true;
        if dfs(
            depth + 1,
            order,
            p_atoms,
            c_atoms,
            p_adj,
            c_adj,
            p_deg,
            c_deg,
            mapping,
            used,
            attempts,
            work_limit,
            over_budget,
        ) {
            return true;
        }
        mapping[depth] = None;
        used[p] = false;
        if *over_budget {
            return false;
        }
    }
    false
}
