//! K1 kernel-versus-twin tests for graph identity and trajectory allocation.
//!
//! Every device call runs on poisoned outputs, is followed by
//! [`check_launches`], and is compared element-for-element with the host twin:
//! a dropped launch (stale poison) or a wrong word fails. Sizes stay small on
//! the CPU runtime; the supervisor runs the same file on wgpu.

#![cfg(feature = "backend")]

use mamba3::backend::{Device, check_launches};
use mamba3::backends::Auto;
use mamba3::models::ms2::allocate::{
    ALLOC_PROPORTIONAL, ALLOC_ROUND_ROBIN, allocate_checked, allocate_lane,
};
use mamba3::models::ms2::contract::{
    CandidateBatch, NO_FORMULA, SCHEMA_VERSION, candidate_status,
};
use mamba3::models::ms2::grammar::{
    ADD_ATOM, CANONICAL_WORK_LIMIT, CLOSE_RING, Limits, START, STOP, Token, canonical_trace, replay,
};
use mamba3::models::ms2::graph::MolGraph;
use mamba3::models::ms2::identity::{
    DUPLICATE_GRAPH, IDENTITY_UNRESOLVED, graph_hash_lane, graph_scratch_len, identity_batch,
    identity_record_len, identity_stack_len,
};
use mamba3::tensor::Tensor;
use mamba3::tensor::ops::index::IdTensor;
use mamba3::tensor::ops::ms2_identity;
use mamba3::tensor::ops::ms2_identity::{check_device_len, check_device_scalar};
use serde_json::Value;

type R = Auto;

fn dev() -> Device<R> {
    Device::<R>::default()
}

fn upload_ids(data: &[u32], shape: Vec<usize>, device: &Device<R>) -> IdTensor<R> {
    IdTensor::from_slice(data, shape, device).unwrap()
}

fn upload_f(data: &[f32], shape: Vec<usize>, device: &Device<R>) -> Tensor<R, f32> {
    Tensor::<R, f32>::from_f32(data, shape, device).unwrap()
}

fn poison_ids(len: usize, device: &Device<R>) -> Vec<u32> {
    let _ = device;
    vec![0xDEAD_BEEF; len]
}

fn assert_ids(actual: &[u32], expected: &[u32], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_eq!(*a, *e, "{what}: word {i} differs");
    }
}

// ---------------------------------------------------------------------------
// Chemistry fixture and trace builders (same source as the twin tests)
// ---------------------------------------------------------------------------

fn fixture() -> Value {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/ms2/chemistry_v0.json");
    let text = std::fs::read_to_string(path).expect("fixture readable");
    serde_json::from_str(&text).expect("fixture parses")
}

fn bonds_of(m: &Value) -> Vec<(usize, usize, u8)> {
    m["bonds"]
        .as_array()
        .expect("bonds")
        .iter()
        .map(|b| {
            let b = b.as_array().expect("bond triple");
            (
                b[0].as_u64().expect("a") as usize,
                b[1].as_u64().expect("b") as usize,
                b[2].as_u64().expect("order") as u8,
            )
        })
        .collect()
}

fn molecule_by_name(fixture: &Value, name: &str) -> MolGraph {
    let m = fixture["molecules"]
        .as_array()
        .expect("molecules")
        .iter()
        .find(|m| m["name"] == name)
        .unwrap_or_else(|| panic!("molecule {name} in fixture"));
    let atoms: Vec<u8> = m["atoms"]
        .as_array()
        .expect("atoms")
        .iter()
        .map(|a| a.as_u64().expect("atom type") as u8)
        .collect();
    MolGraph::new(atoms, bonds_of(m)).expect("fixture molecule builds")
}

fn all_graphs() -> Vec<(String, MolGraph)> {
    let f = fixture();
    let mut out = Vec::new();
    for m in f["molecules"].as_array().expect("molecules") {
        let name = m["name"].as_str().expect("name");
        let g = molecule_by_name(&f, name);
        if let Some(subs) = m.get("subgraphs").and_then(Value::as_array) {
            for (i, sub) in subs.iter().enumerate() {
                let idx: Vec<usize> = sub["atoms"]
                    .as_array()
                    .expect("subgraph atoms")
                    .iter()
                    .map(|a| a.as_u64().expect("index") as usize)
                    .collect();
                let sg = g.induced(&idx).expect("subgraph induces");
                out.push((format!("{name}#{i}"), sg));
            }
        }
        out.push((name.to_string(), g));
    }
    out
}

fn limits_for(a: u32, r: u32) -> Limits {
    Limits::new(a as usize, r as usize).expect("test caps fit")
}

fn canon_with(graph: &MolGraph, a: u32, r: u32) -> Vec<Token> {
    canonical_trace(graph, limits_for(a, r), CANONICAL_WORK_LIMIT)
        .expect("fixture graph canonicalizes")
        .trace
}

/// One alternative legal re-trace from another root (rotated neighbour order).
fn alt_retrace(graph: &MolGraph, a: u32, r: u32) -> Vec<Token> {
    let lim = limits_for(a, r);
    let n = graph.atoms().len();
    let types = graph.atoms().to_vec();
    let mut adj: Vec<Vec<(usize, u8)>> = vec![Vec::new(); n];
    for (x, y, order) in graph.bonds() {
        adj[*x].push((*y, *order));
        adj[*y].push((*x, *order));
    }
    let order_of = |u: usize, v: usize| {
        adj[u]
            .iter()
            .find(|(x, _)| *x == v)
            .map(|(_, o)| *o)
            .expect("bond exists")
    };
    let root = if n > 1 { 1 } else { 0 };
    let mut order = vec![root];
    let mut placed = vec![false; n];
    placed[root] = true;
    let mut index_of = vec![0usize; n];
    let mut tokens = vec![
        Token {
            kind: START,
            atom_type: 0,
            bond: 0,
            pointer: 0,
        },
        Token {
            kind: ADD_ATOM,
            atom_type: types[root],
            bond: 0,
            pointer: 0,
        },
    ];
    let mut head = 0usize;
    while order.len() < n {
        while head < order.len() && !adj[order[head]].iter().any(|(v, _)| !placed[*v]) {
            head += 1;
        }
        if head >= order.len() {
            break;
        }
        let u = order[head];
        let mut nbrs: Vec<usize> = adj[u]
            .iter()
            .filter(|(v, _)| !placed[*v])
            .map(|(v, _)| *v)
            .collect();
        nbrs.sort_by_key(|v| (types[*v], *v));
        if !nbrs.is_empty() {
            let m = nbrs.len();
            nbrs.rotate_left(root % m);
        }
        for v in nbrs {
            tokens.push(Token {
                kind: ADD_ATOM,
                atom_type: types[v],
                bond: order_of(u, v),
                pointer: head as u8,
            });
            let mut closers: Vec<usize> = adj[v]
                .iter()
                .filter(|(w, _)| placed[*w] && *w != u)
                .map(|(w, _)| *w)
                .collect();
            closers.sort_by_key(|w| index_of[*w]);
            for w in closers {
                tokens.push(Token {
                    kind: CLOSE_RING,
                    atom_type: 0,
                    bond: order_of(v, w),
                    pointer: index_of[w] as u8,
                });
            }
            index_of[v] = order.len();
            placed[v] = true;
            order.push(v);
        }
    }
    tokens.push(Token {
        kind: STOP,
        atom_type: 0,
        bond: 0,
        pointer: 0,
    });
    // Fall back to canonical when the rotated walk is illegal or trivial.
    let canon = canon_with(graph, a, r);
    if mamba3::models::ms2::grammar::first_illegal_step(&tokens, lim, None).is_some() {
        return canon;
    }
    tokens
}

/// Pack traces into device records of width `steps * 4 + atoms + 4`.
fn pack_traces(traces: &[Vec<Token>], statuses: &[u32], steps: u32, atoms: u32) -> Vec<u32> {
    let rs = steps as usize * 4 + atoms as usize + 4;
    let mut flat = vec![0u32; traces.len() * rs];
    for (r, trace) in traces.iter().enumerate() {
        let base = r * rs;
        for (s, t) in trace.iter().enumerate() {
            if s < steps as usize {
                flat[base + s * 4] = u32::from(t.kind);
                flat[base + s * 4 + 1] = u32::from(t.atom_type);
                flat[base + s * 4 + 2] = u32::from(t.bond);
                flat[base + s * 4 + 3] = u32::from(t.pointer);
            }
        }
        let len_field = steps as usize * 4 + atoms as usize;
        flat[base + len_field] = trace.len() as u32;
        flat[base + len_field + 1] = statuses[r];
        flat[base + len_field + 2] = 0;
        flat[base + len_field + 3] = u32::MAX;
    }
    flat
}

/// Twin hashes and scratches for these packed rows.
fn twin_hash(
    packed: &[u32],
    rows: usize,
    record_stride: u32,
    steps: u32,
    atoms: u32,
    closures: u32,
    mask: u32,
) -> (Vec<u32>, Vec<u32>) {
    let ss = graph_scratch_len(atoms, closures);
    let mut hashes = vec![0u32; rows];
    let mut scratch = vec![0u32; rows * ss];
    for r in 0..rows as u32 {
        graph_hash_lane(
            packed,
            r,
            record_stride,
            steps,
            atoms,
            closures,
            mask,
            &mut hashes,
            &mut scratch,
            ss as u32,
        );
    }
    (hashes, scratch)
}

#[allow(clippy::too_many_arguments)]
fn run_hash_kernel(
    packed: &[u32],
    rows: usize,
    record_stride: usize,
    scratch_stride: usize,
    steps: usize,
    atoms: u32,
    closures: u32,
    mask: u32,
) -> (Vec<u32>, Vec<u32>) {
    let device = dev();
    let actions_t = upload_ids(packed, vec![rows, record_stride], &device);
    let mut hash_t = upload_ids(&poison_ids(rows, &device), vec![rows], &device);
    let mut scratch_t =
        upload_ids(&poison_ids(rows * scratch_stride, &device), vec![rows, scratch_stride], &device);
    ms2_identity::graph_hash(&actions_t, &mut hash_t, &mut scratch_t, steps, atoms, closures, mask)
        .unwrap();
    check_launches(&device).unwrap();
    (hash_t.try_to_vec().unwrap(), scratch_t.try_to_vec().unwrap())
}

fn check_hash_config(atoms: u32, closures: u32, steps: u32, masks: &[u32]) {
    let graphs = all_graphs();
    assert!(graphs.len() > 700, "fixture holds molecules and subgraphs");
    // Every graph: canonical plus one alternative legal trace.
    let mut traces: Vec<Vec<Token>> = Vec::new();
    let mut names: Vec<String> = Vec::new();
    for (name, g) in graphs.iter().take(400) {
        let c = canon_with(g, atoms, closures);
        if c.len() as u32 > steps {
            continue;
        }
        let alt = alt_retrace(g, atoms, closures);
        traces.push(c);
        names.push(name.clone());
        if alt.len() as u32 <= steps {
            traces.push(alt);
            names.push(format!("{name}-alt"));
        }
    }
    // Ineligible records: unfinished, invalid, empty.
    let empty: Vec<Token> = Vec::new();
    traces.push(empty);
    names.push("empty".to_string());
    let singleton = vec![
        Token {
            kind: START,
            atom_type: 0,
            bond: 0,
            pointer: 0,
        },
        Token {
            kind: ADD_ATOM,
            atom_type: 4,
            bond: 0,
            pointer: 0,
        },
        Token {
            kind: STOP,
            atom_type: 0,
            bond: 0,
            pointer: 0,
        },
    ];
    traces.push(singleton);
    names.push("singleton".to_string());
    let rows = traces.len();
    // Statuses: mostly finished, with unfinished/invalid/empty mixed in.
    let mut statuses = vec![candidate_status::FINISHED; rows];
    if rows > 2 {
        statuses[rows - 2] = 0;
        statuses[rows - 1] = candidate_status::FINISHED | candidate_status::INVALID_FINAL;
    }
    // Several spectra: B*K rows (hash ignores spectra, but the layout must).
    let packed = pack_traces(&traces, &statuses, steps, atoms);
    let rs = steps * 4 + atoms + 4;
    let ss = graph_scratch_len(atoms, closures);
    assert_eq!(identity_record_len(steps, atoms), rs as usize);
    for mask in masks {
        let (want_hash, want_scratch) =
            twin_hash(&packed, rows, rs, steps, atoms, closures, *mask);
        let (got_hash, got_scratch) = run_hash_kernel(
            &packed,
            rows,
            rs as usize,
            ss,
            steps as usize,
            atoms,
            closures,
            *mask,
        );
        assert_ids(&got_hash, &want_hash, &format!("hash A{atoms} mask {mask:#x}"));
        assert_ids(
            &got_scratch,
            &want_scratch,
            &format!("scratch A{atoms} mask {mask:#x}"),
        );
    }
    let _ = names;
}

#[test]
fn hash_kernels_match_twin_small() {
    for mask in [u32::MAX, 0x3, 0xff, 0] {
        check_hash_config(16, 4, 22, &[mask]);
    }
}

#[test]
fn hash_kernels_match_twin_large() {
    for mask in [u32::MAX, 0x3] {
        check_hash_config(32, 8, 42, &[mask]);
    }
}

// ---------------------------------------------------------------------------
// Identity
// ---------------------------------------------------------------------------

fn make_batch(traces: &[Vec<Token>], statuses: &[u32], atoms: u32, closures: u32) -> CandidateBatch {
    assert_eq!(traces.len(), statuses.len());
    let k = traces.len();
    let steps = traces.iter().map(Vec::len).max().unwrap_or(1).max(1);
    let mut actions = vec![0u32; k * steps * 4];
    let mut length = vec![0u32; k];
    for (r, trace) in traces.iter().enumerate() {
        length[r] = trace.len() as u32;
        for (s, token) in trace.iter().enumerate() {
            let base = (r * steps + s) * 4;
            actions[base] = u32::from(token.kind);
            actions[base + 1] = u32::from(token.atom_type);
            actions[base + 2] = u32::from(token.bond);
            actions[base + 3] = u32::from(token.pointer);
        }
    }
    CandidateBatch {
        schema_version: SCHEMA_VERSION,
        batch: 1,
        trajectories: k,
        max_steps: steps,
        max_atoms: atoms as usize,
        max_ring_closures: closures as usize,
        spectrum_id: vec![7],
        trajectory: (0..k).map(|i| i as u32).collect(),
        actions,
        length,
        formula_row: vec![NO_FORMULA; k],
        formula_log_prob: vec![0.0; k],
        trace_log_prob: vec![0.0; k],
        open_valence: vec![0; k * atoms as usize],
        attachment_partition: vec![0; k],
        status: statuses.to_vec(),
        evidence_status: vec![0; k],
        evidence_count: vec![0; k],
        evidence_peak_id: vec![0; (k) * 4],
        evidence_hypothesis: vec![0; (k) * 4],
        evidence_shift: vec![0; (k) * 4],
        evidence_residual: vec![0; (k) * 4],
        evidence_log_prob: vec![0.0; (k) * 4],
        identity_resolution: vec![0; k],
        request_status: vec![0],
        rows_visited: vec![0],
        rows_joined: vec![0],
        rows_scored: vec![0],
        formula_support_complete: vec![0],
        formula_mass_retained: vec![0.0],
        peaks_kept: vec![0],
        intensity_retained: vec![0.0],
        // Schema-2 provenance (no formula here: all-zero counts with
        // `u32::MAX` ranks, as `CandidateBatch::validate` requires).
        formula_counts: vec![0; k * 10],
        formula_source: vec![0; 1],
        formula_rank: vec![NO_FORMULA; k],
    }
}

/// Pack a `CandidateBatch` exactly like `identity_batch` does.
fn pack_batch(batch: &CandidateBatch) -> (Vec<u32>, u32, u32) {
    let steps = batch.max_steps as u32;
    let atoms = batch.max_atoms as u32;
    let rs = steps * 4 + atoms + 4;
    let n = batch.batch * batch.trajectories;
    let mut flat = vec![0u32; n * rs as usize];
    for r in 0..n {
        let abase = r as u32 * rs;
        let tok_base = r * batch.max_steps * 4;
        for w in 0..batch.max_steps * 4 {
            if tok_base + w < batch.actions.len() {
                flat[abase as usize + w] = batch.actions[tok_base + w];
            }
        }
        let len_field = steps * 4 + atoms;
        let base = abase as usize;
        if r < batch.length.len() {
            flat[base + len_field as usize] = batch.length[r];
        }
        if r < batch.status.len() {
            flat[base + len_field as usize + 1] = batch.status[r];
        }
        flat[base + len_field as usize + 2] = 0;
        flat[base + len_field as usize + 3] = batch.formula_row.get(r).copied().unwrap_or(u32::MAX);
    }
    (flat, rs, steps)
}

fn run_identity_kernel(
    batch: &CandidateBatch,
    hash_mask: u32,
    work_max: u32,
) -> (Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>) {
    let device = dev();
    let atoms = batch.max_atoms as u32;
    let closures = batch.max_ring_closures as u32;
    let steps = batch.max_steps;
    let (packed, rs, steps_u) = pack_batch(batch);
    let rows = batch.batch * batch.trajectories;
    let per = batch.trajectories;
    let bonds = atoms.saturating_sub(1).saturating_add(closures);
    let ss = graph_scratch_len(atoms, closures);
    let st = identity_stack_len(atoms);
    // Hash first (poisoned), as generation does.
    let actions_t = upload_ids(&packed, vec![rows, rs as usize], &device);
    let mut hash_t = upload_ids(&poison_ids(rows, &device), vec![rows], &device);
    let mut scratch_t =
        upload_ids(&poison_ids(rows * ss, &device), vec![rows, ss], &device);
    ms2_identity::graph_hash(&actions_t, &mut hash_t, &mut scratch_t, steps, atoms, closures, hash_mask)
        .unwrap();
    check_launches(&device).unwrap();
    // Identity on poisoned outputs.
    let mut ident_t = upload_ids(&poison_ids(rows * 2, &device), vec![rows, 2], &device);
    let mut stack_t = upload_ids(&poison_ids(rows * st, &device), vec![rows, st], &device);
    ms2_identity::graph_identity(
        &actions_t,
        &hash_t,
        &scratch_t,
        &mut ident_t,
        &mut stack_t,
        steps,
        atoms,
        bonds,
        per,
        work_max,
    )
    .unwrap();
    check_launches(&device).unwrap();
    let _ = steps_u;
    (
        hash_t.try_to_vec().unwrap(),
        scratch_t.try_to_vec().unwrap(),
        ident_t.try_to_vec().unwrap(),
        stack_t.try_to_vec().unwrap(),
    )
}

fn check_identity_batch(batch: &CandidateBatch, mask: u32, work: u32, what: &str) {
    let want = identity_batch(batch, mask, work);
    let (got_hash, _, got_ident, _) = run_identity_kernel(batch, mask, work);
    assert_ids(&got_hash, &want.graph_hash, &format!("{what} hash"));
    // Identity rows are [bits, resolution-as-u32].
    let mut want_ident = vec![0u32; want.status_bits.len() * 2];
    for (i, (b, r)) in want.status_bits.iter().zip(want.resolution.iter()).enumerate() {
        want_ident[i * 2] = *b;
        want_ident[i * 2 + 1] = u32::from(*r);
    }
    assert_ids(&got_ident, &want_ident, &format!("{what} identity"));
}

#[test]
fn identity_duplicates_and_ineligible() {
    let atoms = 32;
    let closures = 8;
    let f = fixture();
    let benzene = molecule_by_name(&f, "benzene");
    let t0 = canon_with(&benzene, atoms, closures);
    let t1 = alt_retrace(&benzene, atoms, closures);
    // Exact duplicates plus ineligible records.
    let batch = make_batch(
        &[t0.clone(), t1.clone(), t0.clone()],
        &[
            candidate_status::FINISHED,
            candidate_status::FINISHED,
            candidate_status::FINISHED,
        ],
        atoms,
        closures,
    );
    check_identity_batch(&batch, u32::MAX, 1_000_000, "duplicates");
    // Ineligible: unfinished, invalid, empty mix with identical traces.
    let batch = make_batch(
        &[t0.clone(), t0.clone(), t1.clone()],
        &[
            candidate_status::FINISHED,
            0,
            candidate_status::FINISHED | candidate_status::INVALID_FINAL,
        ],
        atoms,
        closures,
    );
    check_identity_batch(&batch, u32::MAX, 1_000_000, "ineligible");
}

#[test]
fn identity_forced_collisions_and_prism() {
    let atoms = 32;
    let closures = 8;
    // Same-hash non-isomorphic pairs via a tiny mask, over fixture isomers.
    let graphs = all_graphs();
    let mut traces: Vec<Vec<Token>> = Vec::new();
    for (_, g) in graphs.iter().take(24) {
        let t = canon_with(g, atoms, closures);
        traces.push(t);
        if traces.len() >= 8 {
            break;
        }
    }
    let batch = make_batch(&traces, &vec![candidate_status::FINISHED; traces.len()], atoms, closures);
    check_identity_batch(&batch, 0x3, 1_000_000, "forced-collision");
    // Prism vs K3,3: 1-WL blind spot, exact says different.
    let prism = MolGraph::new(
        vec![1; 6],
        vec![
            (0, 1, 1),
            (1, 2, 1),
            (0, 2, 1),
            (3, 4, 1),
            (4, 5, 1),
            (3, 5, 1),
            (0, 3, 1),
            (1, 4, 1),
            (2, 5, 1),
        ],
    )
    .expect("prism builds");
    let mut bipartite = Vec::new();
    for a in 0..3usize {
        for b in 3..6usize {
            bipartite.push((a, b, 1));
        }
    }
    let complete = MolGraph::new(vec![1; 6], bipartite).expect("K3,3 builds");
    let tp = canon_with(&prism, atoms, closures);
    let tc = canon_with(&complete, atoms, closures);
    let batch = make_batch(&[tp, tc], &[candidate_status::FINISHED; 2], atoms, closures);
    check_identity_batch(&batch, u32::MAX, 1_000_000, "prism-k33");
}

#[test]
fn identity_unresolved_budget_one() {
    // Symmetric 6-ring with work_max = 1: unresolved on the later only.
    let atoms = 32;
    let closures = 8;
    let ring = MolGraph::new(
        vec![3; 6],
        vec![(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 4, 1), (4, 5, 1), (0, 5, 1)],
    )
    .expect("ring builds");
    let trace = canon_with(&ring, atoms, closures);
    let batch = make_batch(
        &[trace.clone(), trace],
        &[candidate_status::FINISHED; 2],
        atoms,
        closures,
    );
    let want = identity_batch(&batch, u32::MAX, 1);
    assert_eq!(want.status_bits, vec![0, IDENTITY_UNRESOLVED]);
    assert_eq!(want.resolution, vec![1, 2]);
    check_identity_batch(&batch, u32::MAX, 1, "budget-one");
    check_identity_batch(&batch, u32::MAX, 1_000_000, "budget-funded");
    // Duplicate bit value itself.
    assert_eq!(DUPLICATE_GRAPH, 1 << 7);
}

// ---------------------------------------------------------------------------
// Allocation
// ---------------------------------------------------------------------------

fn alloc_buffers(f: usize) -> (Vec<u32>, Vec<u32>) {
    let mut top = Vec::with_capacity(f * 2);
    let mut counts = vec![0u32; f * 10];
    for s in 0..f as u32 {
        top.push(100 + s);
        top.push(s);
        for e in 0..10 {
            counts[s as usize * 10 + e] = s * 10 + e as u32;
        }
    }
    (top, counts)
}

fn twin_alloc(
    top: &[u32],
    counts: &[u32],
    lp: &[f32],
    top_count: u32,
    f: usize,
    k: usize,
    mode: u32,
) -> Vec<u32> {
    let tc = vec![top_count];
    let mut out = vec![0u32; k * 12];
    allocate_lane(top, counts, lp, &tc, 0, f as u32, k as u32, mode, &mut out);
    out
}

#[allow(clippy::too_many_arguments)]
fn run_alloc_kernel(
    top: &[u32],
    counts: &[u32],
    lp: &[f32],
    top_count: &[u32],
    batch: usize,
    f: usize,
    k: usize,
    mode: u32,
) -> Vec<u32> {
    let device = dev();
    let top_t = upload_ids(top, vec![batch, f, 2], &device);
    let counts_t = upload_ids(counts, vec![batch, f, 10], &device);
    let lp_t = upload_f(lp, vec![batch, f], &device);
    let count_t = upload_ids(top_count, vec![batch], &device);
    let mut out_t = upload_ids(&poison_ids(batch * k * 12, &device), vec![batch, k, 12], &device);
    ms2_identity::allocate(&top_t, &counts_t, &lp_t, &count_t, &mut out_t, mode).unwrap();
    check_launches(&device).unwrap();
    out_t.try_to_vec().unwrap()
}

fn check_alloc_case(lp: &[f32], top_count: u32, f: usize, k: usize, mode: u32, what: &str) {
    let (top, counts) = alloc_buffers(f);
    let want = twin_alloc(&top, &counts, lp, top_count, f, k, mode);
    // Batch-of-1 kernel form.
    let top_b = top.clone();
    let counts_b = counts.clone();
    let lp_b = lp.to_vec();
    let tc_b = vec![top_count];
    let got = run_alloc_kernel(&top_b, &counts_b, &lp_b, &tc_b, 1, f, k, mode);
    assert_ids(&got, &want, &format!("{what} complete records"));
    // The wrapper agrees with the lane on legal inputs.
    if k <= 64 && f <= 8 && top_count <= f as u32 {
        let mut via_checked = vec![0u32; k * 12];
        allocate_checked(&top_b, &counts_b, &lp_b, &tc_b, 1, f as u32, k as u32, mode, &mut via_checked)
            .unwrap();
        assert_ids(&via_checked, &want, &format!("{what} checked"));
    }
}

#[test]
fn allocate_kernels_match_twin() {
    // Round robin and proportional, top_count 0 / 1 / F, K < top_count.
    check_alloc_case(&[0.0; 4], 4, 4, 9, ALLOC_ROUND_ROBIN, "rr");
    check_alloc_case(&[0.0; 4], 0, 4, 6, ALLOC_ROUND_ROBIN, "rr-zero");
    check_alloc_case(&[0.0; 4], 0, 4, 6, ALLOC_PROPORTIONAL, "prop-zero");
    check_alloc_case(&[1.5], 1, 1, 8, ALLOC_PROPORTIONAL, "single");
    check_alloc_case(&[0.0, -1.0, -2.0], 3, 3, 8, ALLOC_PROPORTIONAL, "hand");
    check_alloc_case(&[0.0, 0.0, 0.0], 3, 3, 7, ALLOC_PROPORTIONAL, "ties");
    check_alloc_case(&[0.0, -0.2, -0.4, -0.6, -0.8], 5, 5, 2, ALLOC_PROPORTIONAL, "k-lt-count");
    check_alloc_case(&[0.0, -1.0, -2.0, -3.0], 4, 4, 11, 7, "unknown-mode");
    // Equal probabilities and fallback for entries outside the validated domain.
    check_alloc_case(&[0.0; 4], 4, 4, 6, ALLOC_PROPORTIONAL, "equal");
    check_alloc_case(&[0.0, f32::NAN], 2, 2, 7, ALLOC_PROPORTIONAL, "nan");
    check_alloc_case(&[f32::INFINITY, -1.0], 2, 2, 7, ALLOC_PROPORTIONAL, "inf");
    // Multi-spectrum batch with mixed modes per spectrum is split by mode here;
    // batch routing itself is checked below.
    check_alloc_case(&[0.3, -0.1, -1.7, -0.9], 4, 4, 19, ALLOC_PROPORTIONAL, "det");
}

#[test]
fn allocate_multispectrum_batch() {
    // B = 3 spectra in one launch: round robin, proportional, and empty.
    let device = dev();
    let (f, k) = (4usize, 7usize);
    let mut top = Vec::new();
    let mut counts = Vec::new();
    let mut lp = Vec::new();
    for _ in 0..3 {
        let (t, c) = alloc_buffers(f);
        top.extend(t);
        counts.extend(c);
    }
    lp.extend([0.0f32, -1.0, -2.0, -3.0]);
    lp.extend([0.0f32; 4]);
    lp.extend([0.0f32, -1.0, -2.0, -3.0]);
    let tc = vec![4u32, 4, 0];
    // Spectrum 0 round robin, spectrum 2 empty; spectrum 1 proportional is
    // checked lane-by-lane against the twin below (one mode per launch).
    let top_t = upload_ids(&top, vec![3, f, 2], &device);
    let counts_t = upload_ids(&counts, vec![3, f, 10], &device);
    let lp_t = upload_f(&lp, vec![3, f], &device);
    let tc_t = upload_ids(&tc, vec![3], &device);
    let mut out_t = upload_ids(&poison_ids(3 * k * 12, &device), vec![3, k, 12], &device);
    ms2_identity::allocate(&top_t, &counts_t, &lp_t, &tc_t, &mut out_t, ALLOC_ROUND_ROBIN).unwrap();
    check_launches(&device).unwrap();
    let got = out_t.try_to_vec().unwrap();
    // Twin per spectrum (full batch out, sliced).
    for b in 0..3 {
        let mut want_full = vec![0u32; 3 * k * 12];
        allocate_lane(&top, &counts, &lp, &tc, b as u32, f as u32, k as u32, ALLOC_ROUND_ROBIN, &mut want_full);
        let base = b * k * 12;
        assert_ids(&got[base..base + k * 12], &want_full[base..base + k * 12], &format!("batch spectrum {b}"));
    }
    // Sentinel rows for the empty spectrum are complete records.
    let empty_base = 2 * k * 12;
    for t in 0..k {
        assert_eq!(got[empty_base + t * 12], u32::MAX, "empty slot");
        assert_eq!(got[empty_base + t * 12 + 1], u32::MAX, "empty source");
        for e in 0..10 {
            assert_eq!(got[empty_base + t * 12 + 2 + e], 0, "empty count");
        }
    }
}

#[test]
fn hash_and_identity_cover_full_32_atom_39_bond_graph() {
    // A real 32-atom / 39-bond graph (a 31-bond carbon chain closed by 8
    // ring closures) through the hash and identity kernels at A = 32,
    // R = 8: scratch rows are the full 181 words, filled to the bond cap.
    let atoms = 32u32;
    let closures = 8u32;
    assert_eq!(graph_scratch_len(atoms, closures), 181, "full scratch row");
    let mut bonds: Vec<(usize, usize, u8)> = (0..31).map(|i| (i, i + 1, 1)).collect();
    for (a, b) in [(0, 5), (4, 9), (8, 13), (12, 17), (16, 21), (20, 25), (24, 29), (27, 31)] {
        bonds.push((a, b, 1));
    }
    let graph = MolGraph::new(vec![1; 32], bonds).expect("32-atom graph builds");
    let trace = canon_with(&graph, atoms, closures);
    let rebuilt = replay(&trace, limits_for(atoms, closures), None)
        .expect("trace replays")
        .graph()
        .expect("graph builds");
    assert_eq!(rebuilt.atoms().len(), 32, "all atoms replay");
    assert_eq!(rebuilt.bonds().len(), 39, "all bonds replay");
    let alt = alt_retrace(&graph, atoms, closures);
    let batch = make_batch(
        &[trace.clone(), alt.clone()],
        &[candidate_status::FINISHED; 2],
        atoms,
        closures,
    );
    let want = identity_batch(&batch, u32::MAX, 1_000_000);
    assert_eq!(want.graph_hash[0], want.graph_hash[1], "retraces hash equal");
    assert_eq!(want.status_bits, vec![0, DUPLICATE_GRAPH], "later flagged duplicate");
    assert_eq!(want.resolution, vec![1, 1], "both decided");
    check_identity_batch(&batch, u32::MAX, 1_000_000, "full-32-atom");
}

#[test]
fn allocate_full_f8_k64_matches_twin() {
    // The spec caps on the device: F = 8 retained formulas, K = 64
    // trajectories, in both modes.
    let lp = vec![0.0f32, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0, -3.5];
    check_alloc_case(&lp, 8, 8, 64, ALLOC_PROPORTIONAL, "f8k64-prop");
    check_alloc_case(&lp, 8, 8, 64, ALLOC_ROUND_ROBIN, "f8k64-rr");
    // Extreme finite values strictly inside (-3e38, 3e38) stay proportional
    // (exp(-3e37) underflows to 0, so that slot takes no extra trajectory).
    check_alloc_case(
        &[0.0f32, -1.0, -3e37, -2.0, -0.5, -4.0, -1.5, -0.25],
        8,
        8,
        64,
        ALLOC_PROPORTIONAL,
        "f8k64-extreme",
    );
}

#[test]
fn allocate_rejects_non_f32() {
    // The contract is FP32: reduced-precision tensors are refused with
    // Error::Unsupported naming the dtype, before any launch.
    use half::f16;
    let device = dev();
    let (top, counts) = alloc_buffers(2);
    let top_t = upload_ids(&top, vec![1, 2, 2], &device);
    let counts_t = upload_ids(&counts, vec![1, 2, 10], &device);
    let lp_t = Tensor::<R, f16>::from_f32(&[0.0f32, -1.0], vec![1, 2], &device).unwrap();
    let tc_t = upload_ids(&[2u32], vec![1], &device);
    let mut out_t = upload_ids(&poison_ids(24, &device), vec![1, 2, 12], &device);
    let err =
        ms2_identity::allocate(&top_t, &counts_t, &lp_t, &tc_t, &mut out_t, ALLOC_ROUND_ROBIN)
            .unwrap_err();
    assert!(
        matches!(err, mamba3::error::Error::Unsupported(_)),
        "unsupported dtype, got {err}"
    );
    let msg = format!("{err}");
    assert!(msg.contains("f32"), "names the FP32 contract: {msg}");
    assert!(msg.contains("f16"), "names the rejected dtype: {msg}");
}

#[test]
fn launchers_reject_addresses_beyond_u32() {
    // The device-address checks behind every launcher, unit-tested directly
    // (a bound buffer this large cannot be allocated in a test).
    assert!(check_device_len("actions", u32::MAX as usize).is_ok());
    assert!(check_device_len("actions", u32::MAX as usize + 1).is_err());
    assert_eq!(check_device_scalar("stride", 204).unwrap(), 204);
    assert_eq!(check_device_scalar("stride", u32::MAX as usize).unwrap(), u32::MAX);
    assert!(check_device_scalar("stride", u32::MAX as usize + 1).is_err());
    // The reviewer's overflow shape: B = 21_846, F = 8, N = 256, J = 8.
    // The B * F * N lanes fit u32, but the ion buffer (lanes * J * 12)
    // holds 4_295_098_368 words, so the last spectrum's base wraps to zero
    // on device: the buffer-length check refuses it.
    let lanes = 21_846usize * 8 * 256;
    let ion_words = lanes * 8 * 12;
    assert_eq!(ion_words, 4_295_098_368, "reviewer ion buffer size");
    assert!(check_device_len("ion", ion_words).is_err());
}

#[test]
fn allocate_rejects_bad_shapes() {    let device = dev();
    let (top, counts) = alloc_buffers(2);
    let lp = vec![0.0f32, -1.0];
    let tc = vec![2u32];
    let top_t = upload_ids(&top, vec![1, 2, 2], &device);
    let counts_t = upload_ids(&counts, vec![1, 2, 10], &device);
    let lp_t = upload_f(&lp, vec![1, 2], &device);
    let tc_t = upload_ids(&tc, vec![1], &device);
    // Wrong last dimension on `out`.
    let mut bad = IdTensor::from_slice(&[0u32; 20], vec![1, 2, 10], &device).unwrap();
    assert!(ms2_identity::allocate(&top_t, &counts_t, &lp_t, &tc_t, &mut bad, ALLOC_ROUND_ROBIN).is_err());
    // Wrong rank on `top`.
    let flat_top = upload_ids(&top, vec![4], &device);
    let mut out = upload_ids(&poison_ids(24, &device), vec![1, 2, 12], &device);
    assert!(ms2_identity::allocate(&flat_top, &counts_t, &lp_t, &tc_t, &mut out, ALLOC_ROUND_ROBIN).is_err());
}

#[test]
fn allocate_window_matches_twin() {
    // The retained-to-window slot translation against
    // `alloc_window_lane`: every record is copied with word 0 mapped through
    // `top[(b, s), 1]`; sentinel and out-of-range slots pass through.
    use mamba3::models::ms2::allocate::alloc_window_lane;
    let device = dev();
    let (batch, f, k) = (2usize, 3usize, 4usize);
    // Distinct window slots per retained entry.
    let top: Vec<u32> = vec![
        10, 2, 11, 0, 12, 1,
        20, 1, 21, 2, 22, 0,
    ];
    // Retained slots: valid, valid, valid, none / valid, valid,
    // out-of-range, valid.
    let slots = [0u32, 1, 2, u32::MAX, 2, 0, 7, 1];
    let mut traj_in = vec![0u32; batch * k * 12];
    for (r, &s) in slots.iter().enumerate() {
        traj_in[r * 12] = s;
        traj_in[r * 12 + 1] = 50 + r as u32;
        for e in 0..10 {
            traj_in[r * 12 + 2 + e] = r as u32 * 10 + e as u32;
        }
    }
    let mut want = vec![0u32; batch * k * 12];
    for b in 0..batch {
        for kk in 0..k {
            alloc_window_lane(
                &traj_in,
                &top,
                b as u32,
                kk as u32,
                f as u32,
                k as u32,
                &mut want,
            );
        }
    }
    // Spot values: b0 slot 0 -> window 2, slot 1 -> 0, slot 2 -> 1, MAX
    // passes through; b1 slot 2 -> window 0; slot 7 passes through.
    assert_eq!(want[0], 2);
    assert_eq!(want[12], 0);
    assert_eq!(want[24], 1);
    assert_eq!(want[36], u32::MAX);
    assert_eq!(want[48], 0);
    assert_eq!(want[72], 7);
    let traj_t = upload_ids(&traj_in, vec![batch, k, 12], &device);
    let top_t = upload_ids(&top, vec![batch, f, 2], &device);
    let mut out_t = upload_ids(&poison_ids(batch * k * 12, &device), vec![batch, k, 12], &device);
    ms2_identity::allocate_window(&traj_t, &top_t, &mut out_t).unwrap();
    check_launches(&device).unwrap();
    assert_ids(&out_t.try_to_vec().unwrap(), &want, "allocate_window");
}

#[test]
fn allocate_window_rejects_bad_shapes() {
    let device = dev();
    let traj_t = upload_ids(&vec![0u32; 24], vec![1, 2, 12], &device);
    let top_t = upload_ids(&vec![0u32; 4], vec![1, 2, 2], &device);
    let mut out_t = upload_ids(&vec![0u32; 24], vec![1, 2, 12], &device);
    // Wrong last dimension on `out`.
    let mut bad = IdTensor::from_slice(&vec![0u32; 20], vec![1, 2, 10], &device).unwrap();
    assert!(ms2_identity::allocate_window(&traj_t, &top_t, &mut bad).is_err());
    // Wrong rank on `top`.
    let flat_top = upload_ids(&vec![0u32; 4], vec![4], &device);
    assert!(ms2_identity::allocate_window(&traj_t, &flat_top, &mut out_t).is_err());
    let _ = out_t;
}

#[test]
fn allocate_bf16_matches_f32_twin() {
    // Finding R1-C3: the lane loads the log-probabilities, widens to f32
    // and does ALL allocation arithmetic (max, exp, sum, quotas, fractions)
    // in f32 whatever the neural element type, as the host twin and spec
    // §3.2 do. bf16 inputs therefore give exactly the f32 twin's records —
    // including the reviewer's arithmetic counterexample (F = 3, K = 15,
    // [-0.6796875, -1.375, -3.21875] → [9, 5, 1]), where bf16 arithmetic
    // would give [8, 5, 2].
    use half::bf16;
    let device = dev();
    let f = 3usize;
    let k = 15usize;
    let lp_f32 = vec![-0.6796875f32, -1.375, -3.21875];
    let (top, counts) = alloc_buffers(f);
    let want = twin_alloc(&top, &counts, &lp_f32, 3, f, k, ALLOC_PROPORTIONAL);
    let want_slots: Vec<u32> = want.iter().step_by(12).copied().collect();
    assert_eq!(
        want_slots,
        vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2],
        "the f32 twin assigns 9 trajectories to slot 0, 5 to slot 1, 1 to slot 2"
    );
    // The same values as bf16 (all three are exactly representable).
    let lp_bf16: Vec<bf16> = lp_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    assert!(
        lp_bf16.iter().zip(lp_f32.iter()).all(|(b, f)| b.to_f32() == *f),
        "the fixture values are exactly representable in bf16"
    );
    let top_t = upload_ids(&top, vec![1, f, 2], &device);
    let counts_t = upload_ids(&counts, vec![1, f, 10], &device);
    let lp_t =
        Tensor::<R, bf16>::from_f32(&lp_f32, vec![1, f], &device).unwrap();
    let count_t = upload_ids(&[3u32], vec![1], &device);
    let mut out_t = upload_ids(&poison_ids(k * 12, &device), vec![1, k, 12], &device);
    ms2_identity::allocate(&top_t, &counts_t, &lp_t, &count_t, &mut out_t, ALLOC_PROPORTIONAL)
        .unwrap();
    check_launches(&device).unwrap();
    let got = out_t.try_to_vec().unwrap();
    assert_ids(&got, &want, "bf16 device records equal the f32 twin");
    // And the f32 device path agrees too.
    let lp_f32_t = upload_f(&lp_f32, vec![1, f], &device);
    let mut out_f32_t = upload_ids(&poison_ids(k * 12, &device), vec![1, k, 12], &device);
    ms2_identity::allocate(
        &top_t,
        &counts_t,
        &lp_f32_t,
        &count_t,
        &mut out_f32_t,
        ALLOC_PROPORTIONAL,
    )
    .unwrap();
    check_launches(&device).unwrap();
    assert_ids(
        &out_f32_t.try_to_vec().unwrap(),
        &want,
        "f32 device records equal the f32 twin",
    );
}
