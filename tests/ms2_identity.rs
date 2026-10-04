//! H6.1 host-only tests for graph identity
//! (`mamba3::models::ms2::identity`). No tensors, no kernels.
//!
//! The reference is equality by `canonical_trace` of the graphs rebuilt from
//! the traces: every verdict below is checked against it. The fixture is
//! `tests/fixtures/ms2/chemistry_v0.json` (molecule whole graphs plus every
//! recipe subgraph).
//!
//! What each test proves:
//!
//! * `alternative_traces_hash_equal_and_flagged` — every fixture graph
//!   against an alternative legal trace of the same graph (re-traced from
//!   another root with the existing traversal rules): equal hash under full
//!   and tiny masks, the exact lane says equal, and `identity_batch` flags
//!   the later trajectory while keeping both.
//! * `refinement_banks_hold_typed_labels_and_final_bank` — the bank fix
//!   (review finding 1): an independent `Vec`/`HashMap` reimplementation of
//!   refinement pins the expected hashes and final bank-0 labels, including
//!   the reviewer's singleton hashes; bank 1 holds the round-3 labels.
//! * `hashes_ignore_atom_order` — permuted atom orders re-traced from
//!   scratch give identical hashes (wrapping arithmetic is order-free).
//! * `same_formula_isomers_differ` — same-composition graphs with different
//!   canonical traces are different (hash differs, or the exact check says
//!   different) and are never flagged.
//! * `forced_collisions_keep_distinct_graphs_distinct` — with
//!   `hash_mask = 0x3` every pair's verdict equals canonical-trace equality:
//!   no distinct graphs merge.
//! * `refinement_blind_spot_prism_vs_k33` — triangular prism versus K3,3 with
//!   one atom type and single bonds: the hashes collide (1-WL blind spot)
//!   and the exact check still says different.
//! * `prism_and_k33_traces_are_legal_at_r_max_4` — both traces replay legally
//!   under `R_max = 4` with exactly four closures each.
//! * `symmetric_graph_budget_one_is_unresolved` — a uniform 6-ring with
//!   `work_max = 1` is unresolved (both kept, bit 8 on the later only) and
//!   equal with a large budget.
//! * `extra_ring_bond_is_different_with_equal_labels` — a graph and the same
//!   graph plus one extra ring-closure bond are different even with the
//!   refined labels overwritten equal (induced equality, not bond-presence
//!   subset).
//! * `same_counts_nonisomorphic_with_equal_labels_stay_different` — same atom
//!   count and same bond count, not isomorphic, with labels (and hashes)
//!   forced equal, plus a same-topology different-bond-order pair.
//! * `mismatched_stack_layout_is_unresolved` — the reviewer's capacity case:
//!   rows hashed with `A = 32` compared through an `A = 29` stack are
//!   unresolved, never different.
//! * `hash_wrapping_is_explicit_and_debug_safe` — the reviewer's two-atom
//!   trace: both final labels are `0xfd5474be`, whose sum overflows `u32`,
//!   so the twin's `wrapping_add` (not plain `+=`) is what keeps debug
//!   builds alive; a `u64`-arithmetic proof pins the wrapped hash.
//! * `incomplete_layouts_are_unresolved_never_verdicts` — the reviewer's two
//!   extent cases: an empty scratch row forges equality, a truncated scratch
//!   forges difference; both are unresolved.
//! * `identity_batch_isolates_spectra` — two spectra sharing one graph:
//!   duplicates flag only within the same spectrum.
//! * `duplicate_and_unresolved_share_one_trajectory` — one trajectory proves
//!   equality against an earlier copy and exhausts its budget against a
//!   1-WL-blind non-isomorphic graph: bit 7 and bit 8 together, resolution 2.
//! * `random_pairs_match_canonical_when_resolved` — 2,016 random legal trace
//!   pairs: the lane verdict equals canonical-trace equality whenever the
//!   small budget resolves it.
//! * `random_retraces_give_resolved_nontrivial_equal_pairs` — random legal
//!   traces re-traced from another root give resolved equal pairs, with a
//!   minimum count of non-trivial (>= 4 atoms) ones.
//! * `lane_edge_cases` — count mismatches, empty traces, zero budgets,
//!   malformed scratch and non-eligible batch records.

use std::collections::HashMap;

use serde_json::Value;

use mamba3::models::ms2::Composition;
use mamba3::models::ms2::contract::{
    CandidateBatch, NO_FORMULA, SCHEMA_VERSION, candidate_status,
};
use mamba3::models::ms2::grammar::{
    ADD_ATOM, CANONICAL_WORK_LIMIT, CLOSE_RING, Limits, START, STOP, Token, TraceState,
    canonical_trace, first_illegal_step, replay,
};
use mamba3::models::ms2::graph::MolGraph;
use mamba3::models::ms2::identity::{
    DUPLICATE_GRAPH, IDENTITY_UNRESOLVED, graph_equal_lane, graph_hash_lane, graph_scratch_len,
    identity_batch, identity_lane,
};

/// Lane atom cap for every test: above every fixture graph (22 atoms at
/// most) and the crafted cases.
const ACAP: u32 = 32;
/// Lane ring-closure cap matching `ACAP` (cubane needs 5).
const RCAP: u32 = 8;
/// Lane bond capacity matching the caps (`32 - 1 + 8`).
const BCAP: u32 = 39;
/// Exact-comparison budget that decides every fixture pair.
const BIG_WORK: u32 = 1_000_000;

/// Grammar limits matching the lane caps.
fn limits() -> Limits {
    Limits::new(ACAP as usize, RCAP as usize).expect("test caps fit")
}

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

/// Every fixture graph: each molecule's whole graph plus every recipe
/// subgraph (induced by its atom index list), named `molecule` and
/// `molecule#i`.
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

/// Canonical trace of a graph through the existing traversal code.
fn canon(graph: &MolGraph) -> Vec<Token> {
    canonical_trace(graph, limits(), CANONICAL_WORK_LIMIT)
        .expect("fixture graph canonicalizes")
        .trace
}

/// One breadth-first legal re-trace per root, with the neighbour discovery
/// order rotated by the root for variety. Each trace replays legally and
/// rebuilds the same labelled graph up to atom order.
fn retrace(graph: &MolGraph) -> Vec<Vec<Token>> {
    let n = graph.atoms().len();
    let types = graph.atoms().to_vec();
    let mut adj: Vec<Vec<(usize, u8)>> = vec![Vec::new(); n];
    for (a, b, order) in graph.bonds() {
        adj[*a].push((*b, *order));
        adj[*b].push((*a, *order));
    }
    let order_of = |u: usize, v: usize| {
        adj[u]
            .iter()
            .find(|(x, _)| *x == v)
            .map(|(_, o)| *o)
            .expect("bond exists")
    };
    let mut out = Vec::new();
    for root in 0..n {
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
                let rot = root % nbrs.len();
                nbrs.rotate_left(rot);
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
        assert_eq!(
            first_illegal_step(&tokens, limits(), None),
            None,
            "re-trace from root {root} is legal"
        );
        out.push(tokens);
    }
    out
}

/// Token words of one trace (`T * 4` words; kept for documentation).
#[allow(dead_code)]
fn words_of(trace: &[Token]) -> Vec<u32> {
    trace
        .iter()
        .flat_map(|t| {
            [
                u32::from(t.kind),
                u32::from(t.atom_type),
                u32::from(t.bond),
                u32::from(t.pointer),
            ]
        })
        .collect()
}

/// Device-record stride for these caps and this token capacity.
fn record_stride_for(steps: u32) -> u32 {
    steps.wrapping_mul(4).wrapping_add(ACAP).wrapping_add(4)
}

/// Pack traces into device-record layout (`steps * 4 + ACAP + 4` words per
/// row: tokens, open-valence zeros, length, status, zero bits, `u32::MAX`
/// formula row) with a common `steps`.
fn pack_records(traces: &[Vec<Token>], statuses: &[u32], steps: u32) -> Vec<u32> {
    let rs = record_stride_for(steps) as usize;
    let mut flat = vec![0u32; traces.len() * rs];
    for (r, trace) in traces.iter().enumerate() {
        let base = r * rs;
        for (s, token) in trace.iter().enumerate() {
            flat[base + s * 4] = u32::from(token.kind);
            flat[base + s * 4 + 1] = u32::from(token.atom_type);
            flat[base + s * 4 + 2] = u32::from(token.bond);
            flat[base + s * 4 + 3] = u32::from(token.pointer);
        }
        let len_field = steps as usize * 4 + ACAP as usize;
        flat[base + len_field] = trace.len() as u32;
        flat[base + len_field + 1] = statuses[r];
        flat[base + len_field + 2] = 0;
        flat[base + len_field + 3] = u32::MAX;
    }
    flat
}

/// Lane hash and scratch of one trace through the full-buffer lane.
fn lane_hash(trace: &[Token], mask: u32) -> (u32, Vec<u32>) {
    use mamba3::models::ms2::identity::identity_record_len;
    let steps = trace.len() as u32;
    let rs = record_stride_for(steps);
    debug_assert_eq!(identity_record_len(steps, ACAP), rs as usize);
    let packed = pack_records(&[trace.to_vec()], &[candidate_status::FINISHED], steps);
    let ss = graph_scratch_len(ACAP, RCAP) as u32;
    let mut hashes = vec![0u32; 1];
    let mut scratch = vec![0u32; ss as usize];
    graph_hash_lane(&packed, 0, rs, steps, ACAP, RCAP, mask, &mut hashes, &mut scratch, ss);
    (hashes[0], scratch)
}

/// Atom types, atom count and bond count of a trace through the independent
/// replay/graph code (not through the lane internals).
fn info_of(trace: &[Token]) -> (Vec<u32>, u32, u32) {
    let graph = replay(trace, limits(), None)
        .expect("trace replays")
        .graph()
        .expect("graph builds");
    let types = graph.atoms().iter().map(|t| u32::from(*t)).collect();
    (types, graph.atoms().len() as u32, graph.bonds().len() as u32)
}

/// Exact-lane verdict between two traces through the full-buffer lanes.
fn lane_equal(a: &[Token], b: &[Token], mask: u32, work: u32) -> u32 {
    let steps = (a.len().max(b.len())) as u32;
    let rs = record_stride_for(steps);
    let ss = graph_scratch_len(ACAP, RCAP) as u32;
    let packed = pack_records(
        &[a.to_vec(), b.to_vec()],
        &[candidate_status::FINISHED, candidate_status::FINISHED],
        steps,
    );
    let mut hashes = vec![0u32; 2];
    let mut scratch = vec![0u32; 2 * ss as usize];
    for r in 0..2u32 {
        graph_hash_lane(&packed, r, rs, steps, ACAP, RCAP, mask, &mut hashes, &mut scratch, ss);
    }
    let stack_stride = 3 * ACAP;
    let mut stack = vec![0u32; stack_stride as usize];
    graph_equal_lane(
        &packed,
        0,
        1,
        rs,
        steps,
        &hashes,
        &scratch,
        ss,
        ACAP,
        BCAP,
        work,
        &mut stack,
        0,
        stack_stride,
    )
}

/// One-spectrum batch of traces with these statuses.
fn make_batch(traces: &[Vec<Token>], statuses: &[u32]) -> CandidateBatch {
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
        max_atoms: ACAP as usize,
        max_ring_closures: RCAP as usize,
        spectrum_id: vec![7],
        trajectory: (0..k).map(|i| i as u32).collect(),
        actions,
        length,
        formula_row: vec![NO_FORMULA; k],
        formula_log_prob: vec![0.0; k],
        trace_log_prob: vec![0.0; k],
        open_valence: vec![0; k * ACAP as usize],
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
        // Schema-2 provenance (every record has no formula here, so the
        // counts are all zero exactly when the rank is `u32::MAX`, as
        // `CandidateBatch::validate` requires).
        formula_counts: vec![0; k * 10],
        formula_source: vec![0; 1],
        formula_rank: vec![NO_FORMULA; k],
    }
}

#[test]
fn alternative_traces_hash_equal_and_flagged() {
    let graphs = all_graphs();
    assert!(graphs.len() > 700, "fixture holds molecules and subgraphs");
    for (name, graph) in &graphs {
        let canonical = canon(graph);
        let alts = retrace(graph);
        assert!(!alts.is_empty(), "{name} re-traces");
        let alt = alts
            .into_iter()
            .find(|t| *t != canonical)
            .unwrap_or(canonical.clone());
        // Reference: the rebuilt graph has the same canonical trace.
        let rebuilt = replay(&alt, limits(), None)
            .expect("alternative replays")
            .graph()
            .expect("graph builds");
        assert_eq!(canon(&rebuilt), canonical, "{name}: same graph");
        // Equal hashes under the full and a tiny mask.
        for mask in [u32::MAX, 0x3] {
            let (ha, _) = lane_hash(&canonical, mask);
            let (hb, _) = lane_hash(&alt, mask);
            assert_eq!(ha, hb, "{name} under mask {mask:#x}");
        }
        assert_eq!(lane_equal(&canonical, &alt, u32::MAX, BIG_WORK), 1, "{name}");
        // The batch flags the later trajectory and keeps both records.
        let batch = make_batch(
            &[canonical.clone(), alt.clone()],
            &[candidate_status::FINISHED; 2],
        );
        let out = identity_batch(&batch, u32::MAX, BIG_WORK);
        assert_eq!(out.graph_hash.len(), 2, "{name}");
        assert_eq!(out.graph_hash[0], out.graph_hash[1], "{name}");
        assert_eq!(out.status_bits, vec![0, DUPLICATE_GRAPH], "{name}");
        assert_eq!(out.resolution, vec![1, 1], "{name}");
    }
}

/// Independent copy of the mixer's avalanche (a straightforward
/// re-derivation, not a call into the lane's helper).
fn mix3(index: u32, seed_lo: u32, seed_hi: u32) -> u32 {
    let mut h = index ^ seed_lo;
    h ^= h >> 16;
    h = h.wrapping_mul(0x7feb352d);
    h ^= h >> 15;
    h = h.wrapping_mul(0x846ca68b);
    h ^= seed_hi;
    h ^= h >> 16;
    h
}

/// Independent two-argument hash: `hash(atom type, degree)` and friends.
fn ref_pair(first: u32, second: u32) -> u32 {
    mix3(first, second, 0)
}

/// Independent refinement over a `Vec`/`HashMap` adjacency (not the lane's
/// bond triples or scratch banks): initial typed labels, four synchronous
/// rounds, the final labels, the round-3 labels and the graph hash.
fn ref_refine(types: &[u32], bonds: &[(usize, usize, u8)]) -> (Vec<u32>, Vec<u32>, u32) {
    let n = types.len();
    let mut degree = vec![0u32; n];
    let mut adj: HashMap<usize, Vec<(usize, u8)>> = HashMap::new();
    for (a, b, order) in bonds {
        degree[*a] += 1;
        degree[*b] += 1;
        adj.entry(*a).or_default().push((*b, *order));
        adj.entry(*b).or_default().push((*a, *order));
    }
    let mut current: Vec<u32> = types
        .iter()
        .zip(degree.iter())
        .map(|(ty, deg)| ref_pair(*ty, *deg))
        .collect();
    let mut round3 = current.clone();
    for r in 0..4 {
        let mut next = vec![0u32; n];
        for (a, slot) in next.iter_mut().enumerate() {
            let mut acc = 0u32;
            if let Some(nbrs) = adj.get(&a) {
                for (m, order) in nbrs {
                    acc = acc.wrapping_add(ref_pair(u32::from(*order), current[*m]));
                }
            }
            *slot = ref_pair(current[a], acc);
        }
        if r == 2 {
            round3 = next.clone();
        }
        current = next;
    }
    let mut sum = 0u32;
    for label in &current {
        sum = sum.wrapping_add(*label);
    }
    let hash = mix3(sum, n as u32, bonds.len() as u32);
    (current, round3, hash)
}

/// A legal singleton trace of one atom of this type.
fn singleton_trace(atom_type: u8) -> Vec<Token> {
    vec![
        Token {
            kind: START,
            atom_type: 0,
            bond: 0,
            pointer: 0,
        },
        Token {
            kind: ADD_ATOM,
            atom_type,
            bond: 0,
            pointer: 0,
        },
        Token {
            kind: STOP,
            atom_type: 0,
            bond: 0,
            pointer: 0,
        },
    ]
}

#[test]
fn refinement_banks_hold_typed_labels_and_final_bank() {
    // The reviewer's singleton hashes: the independent implementation
    // reproduces them, so they pin the fixed algorithm (typed initial labels
    // plus four synchronous rounds starting from bank 0).
    for (atom_type, expected) in [(1u8, 0xe96f5975u32), (4u8, 0x97e8f25eu32)] {
        let trace = singleton_trace(atom_type);
        assert_eq!(
            first_illegal_step(&trace, limits(), None),
            None,
            "singleton type {atom_type} is legal"
        );
        let (final_labels, _, expected_hash) = ref_refine(&[u32::from(atom_type)], &[]);
        assert_eq!(expected_hash, expected, "independent type {atom_type}");
        assert_eq!(final_labels.len(), 1, "one atom");
        let (hash, scratch) = lane_hash(&trace, u32::MAX);
        assert_eq!(hash, expected, "lane type {atom_type}");
        let bank = 3 * (ACAP - 1 + RCAP) as usize;
        assert_eq!(scratch[bank], final_labels[0], "final label in bank 0");
    }
    // Non-trivial graphs: lane hash and bank-0 labels equal the independent
    // computation, and bank 1 holds the round-3 labels (proving the
    // alternation ends in bank 0 after four rounds).
    let f = fixture();
    for name in ["benzene", "2-butanol", "glucose"] {
        let g = molecule_by_name(&f, name);
        let trace = canon(&g);
        let types: Vec<u32> = g.atoms().iter().map(|t| u32::from(*t)).collect();
        let bonds: Vec<(usize, usize, u8)> = g.bonds().to_vec();
        let (final_labels, round3, expected_hash) = ref_refine(&types, &bonds);
        let (hash, scratch) = lane_hash(&trace, u32::MAX);
        assert_eq!(hash, expected_hash, "{name} hash");
        let bank = 3 * (ACAP - 1 + RCAP) as usize;
        let bank1 = bank + ACAP as usize;
        let n = types.len();
        // Trace order is the traversal order, not the graph index order, so
        // the per-atom comparison is order-free (sorted multisets); the
        // singletons above already pin exact per-atom placement.
        let mut got_final = scratch[bank..bank + n].to_vec();
        let mut want_final = final_labels.clone();
        got_final.sort_unstable();
        want_final.sort_unstable();
        assert_eq!(got_final, want_final, "{name} bank 0");
        let mut got_r3 = scratch[bank1..bank1 + n].to_vec();
        let mut want_r3 = round3.clone();
        got_r3.sort_unstable();
        want_r3.sort_unstable();
        assert_eq!(got_r3, want_r3, "{name} bank 1");
    }
}

#[test]
fn hash_wrapping_is_explicit_and_debug_safe() {
    // Review finding 1: START; ADD(type=4); ADD(type=4, order=1, pointer=0);
    // STOP. Both final labels are 0xfd5474be, whose sum 8_500_341_116
    // exceeds u32::MAX: plain `+=` panics in debug builds while device
    // arithmetic wraps, so the twin spells `wrapping_add` (the kernel keeps
    // plain device addition). Running the twin here exercises that path in
    // debug builds; the u64 computation below proves the wrapped value
    // independently of the build profile.
    let trace = vec![
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
            kind: ADD_ATOM,
            atom_type: 4,
            bond: 1,
            pointer: 0,
        },
        Token {
            kind: STOP,
            atom_type: 0,
            bond: 0,
            pointer: 0,
        },
    ];
    assert_eq!(
        first_illegal_step(&trace, limits(), None),
        None,
        "two-atom trace is legal"
    );
    let (hash, scratch) = lane_hash(&trace, u32::MAX);
    let bank = 3 * (ACAP - 1 + RCAP) as usize;
    assert_eq!(scratch[bank], 0xfd5474be, "final label of atom 0");
    assert_eq!(scratch[bank + 1], 0xfd5474be, "final label of atom 1");
    // Profile-independent proof with u64 arithmetic reduced mod 2^32.
    let sum32 = ((0xfd5474beu64 + 0xfd5474beu64) % (1u64 << 32)) as u32;
    assert_eq!(hash, mix3(sum32, 2, 1), "wrapped final sum mixes to the hash");
}

#[test]
fn hashes_ignore_atom_order() {    let f = fixture();
    for name in ["benzene", "neopentane", "glucose"] {
        let g = molecule_by_name(&f, name);
        let canonical = canon(&g);
        let (expected, _) = lane_hash(&canonical, u32::MAX);
        let n = g.atoms().len();
        let mut perms: Vec<Vec<usize>> = vec![(0..n).rev().collect()];
        for r in 1..n.min(6) {
            let mut p: Vec<usize> = (0..n).collect();
            p.rotate_left(r);
            perms.push(p);
        }
        for perm in &perms {
            let permuted = g.permuted(perm).expect("permutes");
            assert_eq!(canon(&permuted), canonical, "{name} canonical invariant");
            for alt in retrace(&permuted) {
                let (hash, _) = lane_hash(&alt, u32::MAX);
                assert_eq!(hash, expected, "{name} perm {perm:?}");
            }
        }
    }
}

#[test]
fn same_formula_isomers_differ() {
    let graphs = all_graphs();
    let items: Vec<(String, Composition, Vec<Token>)> = graphs
        .iter()
        .map(|(name, g)| (name.clone(), g.composition(), canon(g)))
        .collect();
    // The named same-formula pair is present and different.
    let butanol = items
        .iter()
        .find(|(n, _, _)| n == "2-butanol")
        .expect("2-butanol");
    let isobutanol = items
        .iter()
        .find(|(n, _, _)| n == "isobutanol")
        .expect("isobutanol");
    assert_eq!(butanol.1, isobutanol.1, "same formula");
    assert_ne!(butanol.2, isobutanol.2, "different graphs");
    let mut pairs = 0u32;
    for i in 0..items.len() {
        for j in (i + 1)..items.len() {
            if items[i].1 == items[j].1 && items[i].2 != items[j].2 {
                if pairs >= 24 {
                    continue;
                }
                let (hi, _) = lane_hash(&items[i].2, u32::MAX);
                let (hj, _) = lane_hash(&items[j].2, u32::MAX);
                let verdict = lane_equal(&items[i].2, &items[j].2, u32::MAX, BIG_WORK);
                assert!(
                    hi != hj || verdict == 0,
                    "{} vs {}: hash differs or exact says different",
                    items[i].0,
                    items[j].0
                );
                let batch = make_batch(
                    &[items[i].2.clone(), items[j].2.clone()],
                    &[candidate_status::FINISHED; 2],
                );
                let out = identity_batch(&batch, u32::MAX, BIG_WORK);
                assert_eq!(out.status_bits, vec![0, 0], "{} vs {}", items[i].0, items[j].0);
                assert_eq!(out.resolution, vec![1, 1], "{} vs {}", items[i].0, items[j].0);
                pairs += 1;
            }
        }
    }
    assert!(pairs > 0, "fixture holds same-formula isomer pairs");
}

#[test]
fn forced_collisions_keep_distinct_graphs_distinct() {
    let graphs = all_graphs();
    let traces: Vec<Vec<Token>> = graphs.iter().map(|(_, g)| canon(g)).collect();
    let batch = make_batch(&traces, &vec![candidate_status::FINISHED; traces.len()]);
    let out = identity_batch(&batch, 0x3, BIG_WORK);
    assert_eq!(out.graph_hash.len(), traces.len());
    assert_eq!(out.status_bits.len(), traces.len());
    assert_eq!(out.resolution.len(), traces.len());
    for (k, canonical) in traces.iter().enumerate() {
        // Expected verdict from the canonical-trace reference.
        let duplicate = traces.iter().take(k).any(|t| *t == *canonical);
        let want = if duplicate { DUPLICATE_GRAPH } else { 0 };
        assert_eq!(
            out.status_bits[k] & (DUPLICATE_GRAPH | IDENTITY_UNRESOLVED),
            want,
            "record {k} ({})",
            graphs[k].0
        );
        assert_eq!(out.resolution[k], 1, "record {k} ({}) decided", graphs[k].0);
    }
}

#[test]
fn refinement_blind_spot_prism_vs_k33() {
    // Triangular prism: two triangles with a matching between them. K3,3:
    // the complete bipartite graph on {0,1,2} x {3,4,5}. Both have 6 atoms
    // of one type, 9 single bonds and 4 ring closures, so both fit
    // A <= 16 and R_max <= 4 under the grammar.
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
    let trace_prism = canon(&prism);
    let trace_complete = canon(&complete);
    assert_ne!(trace_prism, trace_complete, "different graphs");
    // Uniform 3-regular labels refine identically: the hashes collide.
    let (hash_prism, _) = lane_hash(&trace_prism, u32::MAX);
    let (hash_complete, _) = lane_hash(&trace_complete, u32::MAX);
    assert_eq!(hash_prism, hash_complete, "1-WL blind spot collides");
    // The exact check still separates them.
    assert_eq!(
        lane_equal(&trace_prism, &trace_complete, u32::MAX, BIG_WORK),
        0,
        "exact check says different"
    );
}

#[test]
fn prism_and_k33_traces_are_legal_at_r_max_4() {
    // Both constructions need exactly four closures, so their traces replay
    // legally under R_max = 4 (not just the R = 8 the other tests use).
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
    let narrow = Limits::new(16, 4).expect("narrow limits fit");
    for (name, graph) in [("prism", &prism), ("K3,3", &complete)] {
        let trace = canonical_trace(graph, limits(), CANONICAL_WORK_LIMIT)
            .expect("canonicalizes")
            .trace;
        let closures = trace.iter().filter(|t| t.kind == CLOSE_RING).count();
        assert_eq!(closures, 4, "{name} needs exactly four closures");
        assert_eq!(
            first_illegal_step(&trace, narrow, None),
            None,
            "{name} legal at R_max = 4"
        );
        let rebuilt = replay(&trace, narrow, None)
            .expect("replays at R_max = 4")
            .graph()
            .expect("graph builds");
        assert_eq!(rebuilt.bonds().len(), 9, "{name} keeps nine bonds");
    }
}

#[test]
fn symmetric_graph_budget_one_is_unresolved() {
    // Uniform 6-ring: every placement is viable, so one work unit cannot get
    // past the second atom.
    let ring = MolGraph::new(
        vec![3; 6],
        vec![
            (0, 1, 1),
            (1, 2, 1),
            (2, 3, 1),
            (3, 4, 1),
            (4, 5, 1),
            (0, 5, 1),
        ],
    )
    .expect("ring builds");
    let trace = canon(&ring);
    assert_eq!(lane_equal(&trace, &trace, u32::MAX, 1), 2);
    assert_eq!(lane_equal(&trace, &trace, u32::MAX, BIG_WORK), 1);
    let batch = make_batch(&[trace.clone(), trace], &[candidate_status::FINISHED; 2]);
    let starved = identity_batch(&batch, u32::MAX, 1);
    assert_eq!(starved.status_bits, vec![0, IDENTITY_UNRESOLVED]);
    assert_eq!(starved.resolution, vec![1, 2]);
    let funded = identity_batch(&batch, u32::MAX, BIG_WORK);
    assert_eq!(funded.status_bits, vec![0, DUPLICATE_GRAPH]);
    assert_eq!(funded.resolution, vec![1, 1]);
}

#[test]
fn extra_ring_bond_is_different_with_equal_labels() {
    // A path of four identical atoms, and the same atoms plus one extra
    // ring-closure bond: same atom count, different bond count.
    let open =
        MolGraph::new(vec![3; 4], vec![(0, 1, 1), (1, 2, 1), (2, 3, 1)]).expect("path builds");
    let shut = MolGraph::new(
        vec![3; 4],
        vec![(0, 1, 1), (1, 2, 1), (2, 3, 1), (0, 3, 1)],
    )
    .expect("cycle builds");
    let trace_open = canon(&open);
    let trace_shut = canon(&shut);
    assert_ne!(trace_open, trace_shut, "different graphs");
    assert_eq!(lane_equal(&trace_open, &trace_shut, u32::MAX, BIG_WORK), 0);
    // Force the refined labels equal by overwriting both bank-0 rows with
    // one constant: the verdict must stay different (induced equality
    // compares order 0 for non-bonds, so extra bonds cannot hide).
    // Full-buffer form with a common step capacity.
    let steps_os = (trace_open.len().max(trace_shut.len())) as u32;
    let rs_os = record_stride_for(steps_os);
    let ss_os = graph_scratch_len(ACAP, RCAP) as u32;
    let packed_os = pack_records(
        &[trace_open.clone(), trace_shut.clone()],
        &[candidate_status::FINISHED, candidate_status::FINISHED],
        steps_os,
    );
    let mut hashes_os = vec![0u32; 2];
    let mut scratch_os = vec![0u32; 2 * ss_os as usize];
    for r in 0..2u32 {
        graph_hash_lane(&packed_os, r, rs_os, steps_os, ACAP, RCAP, u32::MAX, &mut hashes_os, &mut scratch_os, ss_os);
    }
    let bank_os = 3 * (ACAP - 1 + RCAP);
    for i in 0..4u32 {
        let a0 = i as usize;
        let a1 = ss_os as usize + i as usize;
        scratch_os[bank_os as usize + a0] = 0x1234_5678;
        scratch_os[bank_os as usize + a1] = 0x1234_5678;
    }
    // Hashes differ by bond count; force them equal for the label test by
    // writing the first hash over the second.
    hashes_os[1] = hashes_os[0];
    let stack_stride_os = 3 * ACAP;
    let mut stack_os = vec![0u32; stack_stride_os as usize];
    assert_eq!(
        graph_equal_lane(
            &packed_os,
            0,
            1,
            rs_os,
            steps_os,
            &hashes_os,
            &scratch_os,
            ss_os,
            ACAP,
            BCAP,
            BIG_WORK,
            &mut stack_os,
            0,
            stack_stride_os,
        ),
        0,
        "different with equal labels"
    );
}

/// Overwrite the first `n` bank-0 labels of each row of a flat scratch
/// buffer (stride `ss`) with one constant.
fn force_labels_equal(scratch: &mut [u32], ss: u32, rows: u32, n: u32) {
    let bank = 3u32.wrapping_mul(ACAP.wrapping_sub(1).wrapping_add(RCAP));
    let mut r = 0u32;
    while r < rows {
        let mut i = 0u32;
        while i < n {
            let addr = r.wrapping_mul(ss).wrapping_add(bank.wrapping_add(i));
            if (addr as usize) < scratch.len() {
                scratch[addr as usize] = 0x1234_5678;
            }
            i = i.wrapping_add(1);
        }
        r = r.wrapping_add(1);
    }
}

/// Hash two traces with a common step capacity (test helper kept for
/// documentation; pair tests inline the same packing).
#[allow(dead_code)]
fn hash_pair_common(a: &[Token], b: &[Token], mask: u32) -> (u32, Vec<u32>, Vec<u32>, Vec<u32>) {
    let steps = (a.len().max(b.len())) as u32;
    let rs = record_stride_for(steps);
    let ss = graph_scratch_len(ACAP, RCAP) as u32;
    let packed = pack_records(
        &[a.to_vec(), b.to_vec()],
        &[candidate_status::FINISHED, candidate_status::FINISHED],
        steps,
    );
    let mut hashes = vec![0u32; 2];
    let mut scratch = vec![0u32; 2 * ss as usize];
    for r in 0..2u32 {
        graph_hash_lane(&packed, r, rs, steps, ACAP, RCAP, mask, &mut hashes, &mut scratch, ss);
    }
    (rs, packed, hashes, scratch)
}

#[test]
fn same_counts_nonisomorphic_with_equal_labels_stay_different() {
    // A fork tree and a path: 6 atoms and 5 bonds each, uniform type 1, but
    // not isomorphic (the fork has a degree-3 vertex). Same counts, so the
    // count pre-check cannot decide; the induced adjacency check must.
    let fork = MolGraph::new(
        vec![1; 6],
        vec![(0, 1, 1), (1, 2, 1), (2, 3, 1), (2, 4, 1), (4, 5, 1)],
    )
    .expect("fork builds");
    let path = MolGraph::new(
        vec![1; 6],
        vec![(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 4, 1), (4, 5, 1)],
    )
    .expect("path builds");
    let trace_fork = canon(&fork);
    let trace_path = canon(&path);
    assert_ne!(trace_fork, trace_path, "different graphs");
    let (_, atoms_fork, bonds_fork) = info_of(&trace_fork);
    let (_, atoms_path, bonds_path) = info_of(&trace_path);
    assert_eq!((atoms_fork, bonds_fork), (6, 5));
    assert_eq!((atoms_path, bonds_path), (6, 5));
    assert_eq!(lane_equal(&trace_fork, &trace_path, u32::MAX, BIG_WORK), 0);
    // Force refined labels AND the compared hashes equal: the verdict still
    // rests on the induced adjacency check (as under forced collisions).
    let steps_fp = (trace_fork.len().max(trace_path.len())) as u32;
    let rs_fp = record_stride_for(steps_fp);
    let ss_fp = graph_scratch_len(ACAP, RCAP) as u32;
    let packed_fp = pack_records(
        &[trace_fork.clone(), trace_path.clone()],
        &[candidate_status::FINISHED, candidate_status::FINISHED],
        steps_fp,
    );
    let mut hashes_fp = vec![0u32; 2];
    let mut scratch_fp = vec![0u32; 2 * ss_fp as usize];
    for r in 0..2u32 {
        graph_hash_lane(&packed_fp, r, rs_fp, steps_fp, ACAP, RCAP, u32::MAX, &mut hashes_fp, &mut scratch_fp, ss_fp);
    }
    force_labels_equal(&mut scratch_fp, ss_fp, 2, 6);
    hashes_fp[1] = hashes_fp[0];
    let stack_stride_fp = 3 * ACAP;
    let mut stack_fp = vec![0u32; stack_stride_fp as usize];
    assert_eq!(
        graph_equal_lane(
            &packed_fp,
            0,
            1,
            rs_fp,
            steps_fp,
            &hashes_fp,
            &scratch_fp,
            ss_fp,
            ACAP,
            BCAP,
            BIG_WORK,
            &mut stack_fp,
            0,
            stack_stride_fp,
        ),
        0,
        "fork vs path with equal labels"
    );
    // Same topology, different bond orders: a 3-atom chain with single bonds
    // versus single-plus-double. Same counts (3 atoms, 2 bonds), uniform
    // types, different graphs.
    let single = MolGraph::new(vec![1; 3], vec![(0, 1, 1), (1, 2, 1)]).expect("chain builds");
    let mixed = MolGraph::new(vec![1; 3], vec![(0, 1, 1), (1, 2, 2)]).expect("mixed builds");
    let trace_single = canon(&single);
    let trace_mixed = canon(&mixed);
    assert_ne!(trace_single, trace_mixed, "different bond orders");
    let (_, atoms_single, bonds_single) = info_of(&trace_single);
    let (_, atoms_mixed, bonds_mixed) = info_of(&trace_mixed);
    assert_eq!((atoms_single, bonds_single), (3, 2));
    assert_eq!((atoms_mixed, bonds_mixed), (3, 2));
    assert_eq!(lane_equal(&trace_single, &trace_mixed, u32::MAX, BIG_WORK), 0);
    let steps_sm = (trace_single.len().max(trace_mixed.len())) as u32;
    let rs_sm = record_stride_for(steps_sm);
    let ss_sm = graph_scratch_len(ACAP, RCAP) as u32;
    let packed_sm = pack_records(
        &[trace_single.clone(), trace_mixed.clone()],
        &[candidate_status::FINISHED, candidate_status::FINISHED],
        steps_sm,
    );
    let mut hashes_sm = vec![0u32; 2];
    let mut scratch_sm = vec![0u32; 2 * ss_sm as usize];
    for r in 0..2u32 {
        graph_hash_lane(&packed_sm, r, rs_sm, steps_sm, ACAP, RCAP, u32::MAX, &mut hashes_sm, &mut scratch_sm, ss_sm);
    }
    force_labels_equal(&mut scratch_sm, ss_sm, 2, 3);
    hashes_sm[1] = hashes_sm[0];
    assert_eq!(
        graph_equal_lane(
            &packed_sm,
            0,
            1,
            rs_sm,
            steps_sm,
            &hashes_sm,
            &scratch_sm,
            ss_sm,
            ACAP,
            BCAP,
            BIG_WORK,
            &mut stack_fp,
            0,
            stack_stride_fp,
        ),
        0,
        "same topology, different bond orders, equal labels"
    );
}

#[test]
fn mismatched_stack_layout_is_unresolved() {
    // The reviewer's capacity case: a seven-atom type-3 path hashed with
    // A = 32, R = 8 (181-word rows). Its endpoint-root and centre-root BFS
    // re-traces are isomorphic, hence hash-equal.
    let path = MolGraph::new(
        vec![3; 7],
        vec![(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 4, 1), (4, 5, 1), (5, 6, 1)],
    )
    .expect("path builds");
    let traces = retrace(&path);
    assert_eq!(traces.len(), 7);
    let end = traces[0].clone();
    let centre = traces[3].clone();
    assert_ne!(end, centre, "different roots trace differently");
    let steps_mm = (end.len().max(centre.len())) as u32;
    let rs_mm = record_stride_for(steps_mm);
    let ss_mm = graph_scratch_len(ACAP, RCAP) as u32;
    let packed_mm = pack_records(
        &[end.clone(), centre.clone()],
        &[candidate_status::FINISHED, candidate_status::FINISHED],
        steps_mm,
    );
    let mut hashes_mm = vec![0u32; 2];
    let mut scratch_mm = vec![0u32; 2 * ss_mm as usize];
    for r in 0..2u32 {
        graph_hash_lane(&packed_mm, r, rs_mm, steps_mm, ACAP, RCAP, u32::MAX, &mut hashes_mm, &mut scratch_mm, ss_mm);
    }
    assert_eq!(hashes_mm[0], hashes_mm[1], "isomorphic traces hash equal");
    assert_eq!(ss_mm as usize, 181, "A = 32 rows");
    // The correctly sized stack (A = 32) decides equality.
    let stack_stride_mm = 3 * ACAP;
    let mut stack_mm = vec![0u32; stack_stride_mm as usize];
    assert_eq!(
        graph_equal_lane(
            &packed_mm,
            0,
            1,
            rs_mm,
            steps_mm,
            &hashes_mm,
            &scratch_mm,
            ss_mm,
            ACAP,
            BCAP,
            BIG_WORK,
            &mut stack_mm,
            0,
            stack_stride_mm,
        ),
        1,
        "correct layout decides equal"
    );
    // An 87-word stack (A = 29 geometry) against A = 32 caps is an
    // inconsistent layout: unresolved, never "different".
    let mut short = vec![0u32; 87];
    assert_eq!(
        graph_equal_lane(
            &packed_mm,
            0,
            1,
            rs_mm,
            steps_mm,
            &hashes_mm,
            &scratch_mm,
            ss_mm,
            ACAP,
            BCAP,
            BIG_WORK,
            &mut short,
            0,
            87,
        ),
        2,
        "mismatched stack is unresolved"
    );
}

/// A legal two-atom trace of this bond order between two type-1 atoms.
fn two_atom_trace(order: u8) -> Vec<Token> {
    let trace = vec![
        Token {
            kind: START,
            atom_type: 0,
            bond: 0,
            pointer: 0,
        },
        Token {
            kind: ADD_ATOM,
            atom_type: 1,
            bond: 0,
            pointer: 0,
        },
        Token {
            kind: ADD_ATOM,
            atom_type: 1,
            bond: order,
            pointer: 0,
        },
        Token {
            kind: STOP,
            atom_type: 0,
            bond: 0,
            pointer: 0,
        },
    ];
    assert_eq!(
        first_illegal_step(&trace, limits(), None),
        None,
        "two-atom order-{order} trace is legal"
    );
    trace
}

#[test]
fn incomplete_layouts_are_unresolved_never_verdicts() {
    // Review finding 7, case 1: two legal two-atom type-1 traces (single vs
    // double bond) with zeroed hashes, correct scalar strides, an empty
    // scratch, and complete actions and stack. Every missing label and bond
    // order reads as zero, which forges full agreement; the lane reports
    // unresolved, never equal.
    let single = two_atom_trace(1);
    let double = two_atom_trace(2);
    assert_ne!(single, double, "different bond orders");
    let steps_sd = 4u32;
    let rs_sd = record_stride_for(steps_sd);
    let ss_sd = graph_scratch_len(ACAP, RCAP) as u32;
    let packed_sd = pack_records(
        &[single.clone(), double.clone()],
        &[candidate_status::FINISHED, candidate_status::FINISHED],
        steps_sd,
    );
    let hashes_sd = vec![0u32; 2];
    let scratch_sd: Vec<u32> = Vec::new();
    let stack_stride_sd = 3 * ACAP;
    let mut stack_sd = vec![0u32; stack_stride_sd as usize];
    assert_eq!(
        graph_equal_lane(
            &packed_sd,
            0,
            1,
            rs_sd,
            steps_sd,
            &hashes_sd,
            &scratch_sd,
            ss_sd,
            ACAP,
            BCAP,
            BIG_WORK,
            &mut stack_sd,
            0,
            stack_stride_sd,
        ),
        2,
        "empty scratch is unresolved, not equal"
    );
    // Review finding 7, case 2: two identical nonempty graphs with only the
    // first scratch row retained. The shifted (zero) labels read as a
    // mismatch; the lane reports unresolved, never different.
    let packed_same = pack_records(
        &[single.clone(), single.clone()],
        &[candidate_status::FINISHED, candidate_status::FINISHED],
        steps_sd,
    );
    let mut hashes_full = vec![0u32; 2];
    let mut scratch_full = vec![0u32; 2 * ss_sd as usize];
    for r in 0..2u32 {
        graph_hash_lane(
            &packed_same,
            r,
            rs_sd,
            steps_sd,
            ACAP,
            RCAP,
            u32::MAX,
            &mut hashes_full,
            &mut scratch_full,
            ss_sd,
        );
    }
    assert_eq!(hashes_full[0], hashes_full[1], "identical traces hash equal");
    let scratch_trunc = scratch_full[..ss_sd as usize].to_vec();
    let mut stack_full = vec![0u32; stack_stride_sd as usize];
    assert_eq!(
        graph_equal_lane(
            &packed_same,
            0,
            1,
            rs_sd,
            steps_sd,
            &hashes_full,
            &scratch_trunc,
            ss_sd,
            ACAP,
            BCAP,
            BIG_WORK,
            &mut stack_full,
            0,
            stack_stride_sd,
        ),
        2,
        "truncated scratch is unresolved, not different"
    );
    // The complete layout still decides equal.
    assert_eq!(
        graph_equal_lane(
            &packed_same,
            0,
            1,
            rs_sd,
            steps_sd,
            &hashes_full,
            &scratch_full,
            ss_sd,
            ACAP,
            BCAP,
            BIG_WORK,
            &mut stack_full,
            0,
            stack_stride_sd,
        ),
        1,
        "complete layout decides equal"
    );
}

#[test]
fn identity_batch_isolates_spectra() {
    // Two spectra share one graph: the duplicate flags only the later
    // trajectory of the same spectrum, never the cross-spectrum copy.
    let f = fixture();
    let benzene = molecule_by_name(&f, "benzene");
    let other = molecule_by_name(&f, "isobutanol");
    let g0 = canon(&benzene);
    let g1 = retrace(&benzene)[1].clone();
    let h = canon(&other);
    let steps = g0.len().max(g1.len()).max(h.len()) as u32;
    let k = 4usize;
    let mut actions = vec![0u32; k * steps as usize * 4];
    let mut length = vec![0u32; k];
    for (r, trace) in [g0, g1, canon(&benzene), h].iter().enumerate() {
        length[r] = trace.len() as u32;
        for (s, token) in trace.iter().enumerate() {
            let base = (r * steps as usize + s) * 4;
            actions[base] = u32::from(token.kind);
            actions[base + 1] = u32::from(token.atom_type);
            actions[base + 2] = u32::from(token.bond);
            actions[base + 3] = u32::from(token.pointer);
        }
    }
    let batch = CandidateBatch {
        schema_version: SCHEMA_VERSION,
        batch: 2,
        trajectories: 2,
        max_steps: steps as usize,
        max_atoms: ACAP as usize,
        max_ring_closures: RCAP as usize,
        spectrum_id: vec![7, 7, 9, 9],
        trajectory: vec![0, 1, 0, 1],
        actions,
        length,
        formula_row: vec![NO_FORMULA; k],
        formula_log_prob: vec![0.0; k],
        trace_log_prob: vec![0.0; k],
        open_valence: vec![0; k * ACAP as usize],
        attachment_partition: vec![0; k],
        status: vec![candidate_status::FINISHED; k],
        evidence_status: vec![0; k],
        evidence_count: vec![0; k],
        evidence_peak_id: vec![0; (k) * 4],
        evidence_hypothesis: vec![0; (k) * 4],
        evidence_shift: vec![0; (k) * 4],
        evidence_residual: vec![0; (k) * 4],
        evidence_log_prob: vec![0.0; (k) * 4],
        identity_resolution: vec![0; k],
        request_status: vec![0, 0],
        rows_visited: vec![0, 0],
        rows_joined: vec![0, 0],
        rows_scored: vec![0, 0],
        formula_support_complete: vec![0, 0],
        formula_mass_retained: vec![0.0, 0.0],
        peaks_kept: vec![0, 0],
        intensity_retained: vec![0.0, 0.0],
        formula_counts: vec![0; k * 10],
        formula_source: vec![0; 2],
        formula_rank: vec![NO_FORMULA; k],
    };
    let out = identity_batch(&batch, u32::MAX, BIG_WORK);
    assert_eq!(out.status_bits, vec![0, DUPLICATE_GRAPH, 0, 0]);
    assert_eq!(out.resolution, vec![1, 1, 1, 1]);
}

#[test]
fn duplicate_and_unresolved_share_one_trajectory() {
    // One trajectory proves equality against an earlier copy of its own
    // graph and exhausts its budget against a 1-WL-blind non-isomorphic
    // graph: bit 7 and bit 8 together, resolution 2.
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
    let prism_canon = canon(&prism);
    let prism_alt = retrace(&prism)
        .into_iter()
        .find(|t| *t != prism_canon)
        .unwrap_or(prism_canon.clone());
    let k33_canon = canon(&complete);
    // Calibrate a budget where the equal pair resolves but the blind-spot
    // pair still spends everything: equal verdicts persist with more budget
    // (deterministic first-solution search), unresolved ones decide later.
    let mut star = 0u32;
    for wm in [2u32, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096, 16384, 65536] {
        if lane_equal(&prism_canon, &prism_alt, u32::MAX, wm) == 1 {
            star = wm;
            break;
        }
    }
    assert_ne!(star, 0, "equal prism pair resolves within the scan");
    assert_eq!(
        lane_equal(&prism_canon, &k33_canon, u32::MAX, star),
        2,
        "blind-spot pair is still unresolved at work {star}"
    );
    // Three records, one spectrum: [prism, K3,3, prism-alt]. The hashes of
    // prism and K3,3 collide (blind spot); the retraces hash equal.
    let steps_du = prism_canon
        .len()
        .max(prism_alt.len())
        .max(k33_canon.len()) as u32;
    let rs_du = record_stride_for(steps_du);
    let ss_du = graph_scratch_len(ACAP, RCAP) as u32;
    let packed_du = pack_records(
        &[prism_canon, k33_canon, prism_alt],
        &[candidate_status::FINISHED; 3],
        steps_du,
    );
    let mut hashes_du = vec![0u32; 3];
    let mut scratch_du = vec![0u32; 3 * ss_du as usize];
    for r in 0..3u32 {
        graph_hash_lane(
            &packed_du,
            r,
            rs_du,
            steps_du,
            ACAP,
            RCAP,
            u32::MAX,
            &mut hashes_du,
            &mut scratch_du,
            ss_du,
        );
    }
    assert_eq!(hashes_du[0], hashes_du[1], "blind-spot hashes collide");
    assert_eq!(hashes_du[0], hashes_du[2], "retraces hash equal");
    let stack_stride_du = 3 * ACAP;
    let mut stacks_du = vec![0u32; 3 * stack_stride_du as usize];
    let (bits, resolution) = identity_lane(
        &packed_du,
        2,
        rs_du,
        steps_du,
        3,
        ACAP,
        BCAP,
        star,
        &hashes_du,
        &scratch_du,
        ss_du,
        &mut stacks_du,
        2 * stack_stride_du,
        stack_stride_du,
    );
    assert_eq!(
        bits,
        DUPLICATE_GRAPH | IDENTITY_UNRESOLVED,
        "equal against record 0, unresolved against record 1"
    );
    assert_eq!(resolution, 2);
}

#[test]
fn lane_edge_cases() {
    // Unequal counts are different without spending any budget.
    let one = vec![
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
    let two = vec![
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
            kind: ADD_ATOM,
            atom_type: 4,
            bond: 1,
            pointer: 0,
        },
        Token {
            kind: STOP,
            atom_type: 0,
            bond: 0,
            pointer: 0,
        },
    ];
    assert_eq!(lane_equal(&one, &two, u32::MAX, 0), 0);
    // Empty traces are equal.
    let empty: Vec<Token> = Vec::new();
    assert_eq!(lane_equal(&empty, &empty, u32::MAX, 0), 1);
    // Zero budget on a non-empty equal pair is unresolved.
    assert_eq!(lane_equal(&one, &one, u32::MAX, 0), 2);
    // A stack whose length is not a multiple of 3 is unresolved, not a guess.
    let steps_one = one.len() as u32;
    let rs_one = record_stride_for(steps_one);
    let ss_one = graph_scratch_len(ACAP, RCAP) as u32;
    let packed_one = pack_records(std::slice::from_ref(&one), &[candidate_status::FINISHED], steps_one);
    let mut hashes_one = vec![0u32; 1];
    let mut scratch_one = vec![0u32; ss_one as usize];
    graph_hash_lane(&packed_one, 0, rs_one, steps_one, ACAP, RCAP, u32::MAX, &mut hashes_one, &mut scratch_one, ss_one);
    let mut bad = vec![0u32; 4];
    assert_eq!(
        graph_equal_lane(
            &packed_one,
            0,
            0,
            rs_one,
            steps_one,
            &hashes_one,
            &scratch_one,
            ss_one,
            ACAP,
            BCAP,
            BIG_WORK,
            &mut bad,
            0,
            4,
        ),
        2
    );
    // Records that are not finished-and-valid get resolution 1 and no bit,
    // and identical unfinished traces are never flagged.
    let batch = make_batch(
        &[one.clone(), one.clone(), two.clone()],
        &[
            candidate_status::FINISHED,
            0,
            candidate_status::FINISHED | candidate_status::INVALID_FINAL,
        ],
    );
    let out = identity_batch(&batch, u32::MAX, BIG_WORK);
    assert_eq!(out.status_bits, vec![0, 0, 0]);
    assert_eq!(out.resolution, vec![1, 1, 1]);
}

// ---------------------------------------------------------------------------
// Randomized cross-check against the canonical-trace reference.
// ---------------------------------------------------------------------------

/// One set bit of a legality mask, picked by the caller.
fn pick_bit(next: &mut impl FnMut() -> u32, mask: u32) -> u32 {
    assert_ne!(mask, 0, "a legal value exists");
    let mut opts = Vec::new();
    let mut b = 0u32;
    while b < 32 {
        if mask & (1u32 << b) != 0 {
            opts.push(b);
        }
        b += 1;
    }
    opts[next() as usize % opts.len()]
}

/// One random legal trace from the grammar: at each step every field is
/// picked uniformly among its legal values through `TraceState::masks`.
fn random_trace(next: &mut impl FnMut() -> u32, limits: Limits) -> Vec<Token> {
    let blank = Token {
        kind: 0,
        atom_type: 0,
        bond: 0,
        pointer: 0,
    };
    let mut state = TraceState::new(limits, None);
    let mut tokens = Vec::new();
    while !state.stopped() {
        let kinds = state.masks(blank).kinds;
        assert_ne!(kinds, 0, "a legal kind exists without budget");
        let kind = pick_bit(next, kinds);
        let mut token = Token {
            kind: kind as u8,
            atom_type: 0,
            bond: 0,
            pointer: 0,
        };
        if kind == u32::from(ADD_ATOM) {
            if state.atoms() == 0 {
                let mask = state
                    .masks(Token {
                        kind: token.kind,
                        ..blank
                    })
                    .atom_types;
                token.atom_type = pick_bit(next, mask) as u8;
            } else {
                let mut opts = Vec::new();
                for id in 1..=17u32 {
                    let probe = state.masks(Token {
                        kind: token.kind,
                        atom_type: id as u8,
                        ..blank
                    });
                    if probe.atom_types & (1u32 << id) != 0 {
                        opts.push(id);
                    }
                }
                assert!(!opts.is_empty(), "a legal atom type exists");
                let id = opts[next() as usize % opts.len()];
                token.atom_type = id as u8;
                let bonds = state
                    .masks(Token {
                        kind: token.kind,
                        atom_type: token.atom_type,
                        ..blank
                    })
                    .bonds;
                token.bond = pick_bit(next, bonds) as u8;
                let pointers = state
                    .masks(Token {
                        kind: token.kind,
                        atom_type: token.atom_type,
                        bond: token.bond,
                        ..blank
                    })
                    .pointers;
                token.pointer = pick_bit(next, pointers) as u8;
            }
        } else if kind == u32::from(CLOSE_RING) {
            let bonds = state.masks(Token { kind: token.kind, ..blank }).bonds;
            token.bond = pick_bit(next, bonds) as u8;
            let pointers = state
                .masks(Token {
                    kind: token.kind,
                    bond: token.bond,
                    ..blank
                })
                .pointers;
            token.pointer = pick_bit(next, pointers) as u8;
        }
        state.apply(token).expect("generated token is legal");
        tokens.push(token);
        assert!(tokens.len() <= limits.max_steps() + 1, "the walk terminates");
    }
    tokens
}

#[test]
fn random_pairs_match_canonical_when_resolved() {
    let small = Limits::new(6, 2).expect("small limits fit");
    let mut rng = 0x243f_6a88_85a3_08d3u64;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng >> 32) as u32
    };
    let mut traces = Vec::new();
    while traces.len() < 64 {
        traces.push(random_trace(&mut next, small));
    }
    // Reference canonical traces of the rebuilt graphs: 64 traces give
    // 64 * 63 / 2 = 2,016 pairs.
    let canons: Vec<Vec<Token>> = traces
        .iter()
        .map(|trace| {
            let graph = replay(trace, small, None)
                .expect("random trace replays")
                .graph()
                .expect("graph builds");
            canonical_trace(&graph, small, CANONICAL_WORK_LIMIT)
                .expect("small graph canonicalizes")
                .trace
        })
        .collect();
    let mut resolved = 0u32;
    let mut total = 0u32;
    for i in 0..traces.len() {
        for j in (i + 1)..traces.len() {
            total += 1;
            let expected = canons[i] == canons[j];
            let (hi, _) = lane_hash(&traces[i], u32::MAX);
            let (hj, _) = lane_hash(&traces[j], u32::MAX);
            if hi != hj {
                // A hash mismatch decides different: sound because equal
                // graphs have equal hashes under synchronous refinement.
                assert!(!expected, "pair {i}/{j}: hashes differ but graphs equal");
                resolved += 1;
            } else {
                let verdict = lane_equal(&traces[i], &traces[j], u32::MAX, 256);
                if verdict == 0 {
                    assert!(!expected, "pair {i}/{j}: different but graphs equal");
                    resolved += 1;
                } else if verdict == 1 {
                    assert!(expected, "pair {i}/{j}: equal but graphs differ");
                    resolved += 1;
                }
            }
        }
    }
    assert_eq!(total, 2016);
    assert!(
        resolved >= total - total / 20,
        "{resolved}/{total} pairs resolved at work 256"
    );
}

#[test]
fn random_retraces_give_resolved_nontrivial_equal_pairs() {
    // Deliberate isomorphic pairs: each random legal trace plus a re-trace
    // of the same rebuilt graph from another root with a rotated neighbour
    // order. Every pair must resolve equal at full budget.
    let small = Limits::new(6, 2).expect("small limits fit");
    let mut rng = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng >> 32) as u32
    };
    let mut nontrivial = 0u32;
    let mut draws = 0u32;
    while nontrivial < 24 && draws < 512 {
        draws += 1;
        let trace = random_trace(&mut next, small);
        let graph = replay(&trace, small, None)
            .expect("random trace replays")
            .graph()
            .expect("graph builds");
        let n_atoms = graph.atoms().len();
        if n_atoms < 4 {
            continue;
        }
        let retraces = retrace(&graph);
        assert!(!retraces.is_empty(), "nontrivial graph re-traces");
        // Prefer a re-trace from another root that differs from the
        // original token stream.
        let mut alt = retraces[0].clone();
        for cand in retraces.iter().skip(1) {
            alt = cand.clone();
            if *cand != trace {
                break;
            }
        }
        let rebuilt = replay(&alt, limits(), None)
            .expect("re-trace replays")
            .graph()
            .expect("graph builds");
        assert_eq!(
            canonical_trace(&rebuilt, limits(), CANONICAL_WORK_LIMIT)
                .expect("canonicalizes")
                .trace,
            canonical_trace(&graph, limits(), CANONICAL_WORK_LIMIT)
                .expect("canonicalizes")
                .trace,
            "re-trace rebuilds the same graph"
        );
        let (ha, _) = lane_hash(&trace, u32::MAX);
        let (hb, _) = lane_hash(&alt, u32::MAX);
        assert_eq!(ha, hb, "isomorphic pair hashes equal");
        assert_eq!(
            lane_equal(&trace, &alt, u32::MAX, BIG_WORK),
            1,
            "isomorphic pair resolves equal"
        );
        nontrivial += 1;
    }
    assert_eq!(nontrivial, 24, "found 24 non-trivial equal pairs");
}

#[test]
fn zero_stride_empty_actions_is_unresolved() {
    // Finding R2-A: the reviewer's remaining scenario for finding 7 — A =
    // 32, bonds = 39, steps = 42, records 1 and 0, `record_stride` = 0,
    // empty actions, hashes [0, 0], 362 scratch words with stride 181, and
    // a complete 96-word stack. The extent checks used to accept the empty
    // action ranges and return equal; the record stride, indices and every
    // extent are now validated by guarded u32 comparisons (division-based
    // guards, no `usize` addition) before any multiplication, so the
    // invalid layout is unresolved.
    let actions: Vec<u32> = Vec::new();
    let hashes = vec![0u32, 0u32];
    let scratch = vec![0u32; 362];
    let mut stack = vec![0u32; 96];
    assert_eq!(
        graph_equal_lane(
            &actions, 1, 0, 0, 42, &hashes, &scratch, 181, 32, 39, BIG_WORK, &mut stack, 0, 96,
        ),
        2,
        "stride 0 with empty actions is unresolved, not equal"
    );
}

#[test]
fn oversized_record_index_is_unresolved_without_panic() {
    // Finding R2-A: record-base multiplication used to precede validation,
    // so an oversized record index could panic in debug builds before the
    // promised unresolved result. Indices are now validated before any
    // multiplication.
    let single = two_atom_trace(1);
    let steps = 4u32;
    let rs = record_stride_for(steps);
    let ss = graph_scratch_len(ACAP, RCAP) as u32;
    let packed = pack_records(
        &[single.clone(), single.clone()],
        &[candidate_status::FINISHED, candidate_status::FINISHED],
        steps,
    );
    let mut hashes = vec![0u32; 2];
    let mut scratch = vec![0u32; 2 * ss as usize];
    for r in 0..2u32 {
        graph_hash_lane(
            &packed, r, rs, steps, ACAP, RCAP, u32::MAX, &mut hashes, &mut scratch, ss,
        );
    }
    let stack_stride = 3 * ACAP;
    let mut stack = vec![0u32; stack_stride as usize];
    assert_eq!(
        graph_equal_lane(
            &packed,
            u32::MAX,
            1,
            rs,
            steps,
            &hashes,
            &scratch,
            ss,
            ACAP,
            BCAP,
            BIG_WORK,
            &mut stack,
            0,
            stack_stride,
        ),
        2,
        "oversized record index is unresolved, never a panic"
    );
    assert_eq!(
        graph_equal_lane(
            &packed,
            0,
            u32::MAX,
            rs,
            steps,
            &hashes,
            &scratch,
            ss,
            ACAP,
            BCAP,
            BIG_WORK,
            &mut stack,
            0,
            stack_stride,
        ),
        2,
        "oversized second index is unresolved, never a panic"
    );
}

#[test]
fn short_record_stride_is_unresolved() {
    // Finding R2-A: the action stride must cover the layout's actual
    // requirement (`record_stride >= steps * 4 + atoms_cap + 4`); a stride
    // holding only the token words is an invalid layout, unresolved rather
    // than decided on truncated rows.
    let single = two_atom_trace(1);
    let steps = 4u32;
    let rs = record_stride_for(steps);
    let ss = graph_scratch_len(ACAP, RCAP) as u32;
    let packed = pack_records(
        &[single.clone(), single.clone()],
        &[candidate_status::FINISHED, candidate_status::FINISHED],
        steps,
    );
    let mut hashes = vec![0u32; 2];
    let mut scratch = vec![0u32; 2 * ss as usize];
    for r in 0..2u32 {
        graph_hash_lane(
            &packed, r, rs, steps, ACAP, RCAP, u32::MAX, &mut hashes, &mut scratch, ss,
        );
    }
    let stack_stride = 3 * ACAP;
    let mut stack = vec![0u32; stack_stride as usize];
    assert_eq!(
        graph_equal_lane(
            &packed,
            0,
            1,
            steps * 4,
            steps,
            &hashes,
            &scratch,
            ss,
            ACAP,
            BCAP,
            BIG_WORK,
            &mut stack,
            0,
            stack_stride,
        ),
        2,
        "a stride missing the valence/trailing words is unresolved"
    );
}
