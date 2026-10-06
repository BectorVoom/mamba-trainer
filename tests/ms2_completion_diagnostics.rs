//! MC6 tests: dead-end cause diagnostics under the exact-completion grammar.
//!
//! Host only: no device, no kernels. Hand-built budgets and traces plus a
//! soundness sweep asserting that every completable exact-mode prefix passes
//! all four [`feasibility`](mamba3::models::ms2::completion_diagnostics::feasibility)
//! checks.

use std::collections::HashMap;

use mamba3::models::ms2::chem::Composition;
use mamba3::models::ms2::completion_diagnostics::{
    DoomReason, classify_dead_end, feasibility, first_doomed_step,
};
use mamba3::models::ms2::grammar::{
    ADD_ATOM, CLOSE_RING, Limits, START, STOP, Token, TraceState, canonical_trace, replay_exact,
    replay_exact_v1,
};
use mamba3::models::ms2::graph::MolGraph;

fn blank() -> Token {
    Token {
        kind: 0,
        atom_type: 0,
        bond: 0,
        pointer: 0,
    }
}

fn start_token() -> Token {
    Token {
        kind: START,
        atom_type: 0,
        bond: 0,
        pointer: 0,
    }
}

fn root_token(atom_type: u8) -> Token {
    Token {
        kind: ADD_ATOM,
        atom_type,
        bond: 0,
        pointer: 0,
    }
}

fn add_token(atom_type: u8, bond: u8, pointer: u8) -> Token {
    Token {
        kind: ADD_ATOM,
        atom_type,
        bond,
        pointer,
    }
}

fn soundness_budgets() -> Vec<(&'static str, Composition)> {
    vec![
        ("C2H6O", [2, 6, 0, 1, 0, 0, 0, 0, 0, 0]),
        ("C3H8O", [3, 8, 0, 1, 0, 0, 0, 0, 0, 0]),
        ("C2H7N", [2, 7, 1, 0, 0, 0, 0, 0, 0, 0]),
        ("C3H4", [3, 4, 0, 0, 0, 0, 0, 0, 0, 0]),
        ("C4H8", [4, 8, 0, 0, 0, 0, 0, 0, 0, 0]),
        ("C3H7NO", [3, 7, 1, 1, 0, 0, 0, 0, 0, 0]),
        ("F2", [0, 0, 0, 0, 2, 0, 0, 0, 0, 0]),
    ]
}

/// Every legal single-token extension of this exact-mode prefix except STOP.
fn extensions(state: &TraceState) -> Vec<Token> {
    let mut out = Vec::new();
    let kinds = state.masks(blank()).kinds;
    if kinds & (1u32 << ADD_ATOM) != 0 {
        if state.step() == 1 {
            let types = state
                .masks(Token {
                    kind: ADD_ATOM,
                    ..blank()
                })
                .atom_types;
            for id in 1..=17u8 {
                if types & (1u32 << id) != 0 {
                    out.push(Token {
                        kind: ADD_ATOM,
                        atom_type: id,
                        ..blank()
                    });
                }
            }
        } else {
            let types = state
                .masks(Token {
                    kind: ADD_ATOM,
                    ..blank()
                })
                .atom_types;
            for id in 1..=17u8 {
                if types & (1u32 << id) == 0 {
                    continue;
                }
                let bonds = state
                    .masks(Token {
                        kind: ADD_ATOM,
                        atom_type: id,
                        ..blank()
                    })
                    .bonds;
                for b in 1..=3u8 {
                    if bonds & (1u32 << b) == 0 {
                        continue;
                    }
                    let pointers = state
                        .masks(Token {
                            kind: ADD_ATOM,
                            atom_type: id,
                            bond: b,
                            ..blank()
                        })
                        .pointers;
                    for p in 0..32u8 {
                        if pointers & (1u32 << p) != 0 {
                            out.push(Token {
                                kind: ADD_ATOM,
                                atom_type: id,
                                bond: b,
                                pointer: p,
                            });
                        }
                    }
                }
            }
        }
    }
    if kinds & (1u32 << CLOSE_RING) != 0 {
        let bonds = state
            .masks(Token {
                kind: CLOSE_RING,
                ..blank()
            })
            .bonds;
        for b in 1..=3u8 {
            if bonds & (1u32 << b) == 0 {
                continue;
            }
            let pointers = state
                .masks(Token {
                    kind: CLOSE_RING,
                    bond: b,
                    ..blank()
                })
                .pointers;
            for p in 0..32u8 {
                if pointers & (1u32 << p) != 0 {
                    out.push(Token {
                        kind: CLOSE_RING,
                        bond: b,
                        pointer: p,
                        ..blank()
                    });
                }
            }
        }
    }
    out
}

/// Depth-first walk over the independent upper-bound oracle
/// (`TraceState::new` with the budget, as `closed_prefix_rule_is_sound` does),
/// recording `(prefix, completable)` in post-order and returning whether this
/// prefix can still reach a complete molecule.
///
/// The oracle never applies the rule under test (the v2 lookahead), so the
/// sweep can see a completable prefix the lookahead wrongly removed.
fn walk_completable(
    prefix: &mut Vec<Token>,
    limits: Limits,
    budget: Composition,
    records: &mut Vec<(Vec<Token>, bool)>,
) -> bool {
    let mut state = TraceState::new(limits, Some(budget));
    for tok in prefix.iter() {
        state.apply(*tok).expect("walk extends legal tokens only");
    }
    let mut completable = state.is_complete();
    for child in extensions(&state) {
        prefix.push(child);
        if walk_completable(prefix, limits, budget, records) {
            completable = true;
        }
        prefix.pop();
    }
    records.push((prefix.clone(), completable));
    completable
}

#[test]
fn soundness_over_upper_bound_prefixes() {
    let limits = Limits::new(6, 1).unwrap();
    for (name, budget) in soundness_budgets() {
        let mut records = Vec::new();
        let mut prefix = vec![start_token()];
        walk_completable(&mut prefix, limits, budget, &mut records);
        let by_prefix: HashMap<Vec<Token>, bool> = records.iter().cloned().collect();
        let mut completable_count = 0usize;
        let mut non_h = 0usize;
        let mut non_o = 0usize;
        let mut non_c = 0usize;
        let mut non_v = 0usize;
        let mut non_uncaught = 0usize;
        let mut first_h = 0usize;
        let mut first_o = 0usize;
        let mut first_c = 0usize;
        let mut first_v = 0usize;
        let mut first_uncaught = 0usize;
        let mut first_total = 0usize;
        for (tokens, completable) in &records {
            // Feasibility of the oracle prefix itself (the upper-bound state
            // carries the same used counts, residuals and closures the exact
            // state would, since `feasibility` never reads the exact flag).
            let mut oracle = TraceState::new(limits, Some(budget));
            for tok in tokens.iter() {
                oracle.apply(*tok).expect("oracle prefix replays");
            }
            let f = feasibility(&oracle, &budget, limits);
            if *completable {
                completable_count += 1;
                // Rebuild the exact-mode state by `apply_v1` token by token
                // and assert every check passes: a completable oracle prefix
                // must survive the lookahead.
                let mut exact = TraceState::new_exact(limits, budget);
                for (i, tok) in tokens.iter().enumerate() {
                    exact.apply_v1(*tok).unwrap_or_else(|e| {
                        panic!("{name}: completable prefix fails apply_v1 at {i}: {e}")
                    });
                    let g = feasibility(&exact, &budget, limits);
                    assert!(
                        g.hydrogen && g.open_site && g.closable && g.valence,
                        "{name}: completable prefix {tokens:?} fails feasibility at {i}: {g:?}"
                    );
                }
                assert!(
                    f.hydrogen && f.open_site && f.closable && f.valence,
                    "{name}: completable prefix {tokens:?} fails feasibility {f:?}"
                );
            } else {
                if !f.hydrogen {
                    non_h += 1;
                }
                if !f.open_site {
                    non_o += 1;
                }
                if !f.closable {
                    non_c += 1;
                }
                if !f.valence {
                    non_v += 1;
                }
                if f.hydrogen && f.open_site && f.closable && f.valence {
                    non_uncaught += 1;
                }
                // First doomed prefix of its maximal doomed subtree: the
                // parent is completable (the empty parent counts as such).
                let parent_completable = if tokens.len() <= 1 {
                    true
                } else {
                    by_prefix
                        .get(&tokens[..tokens.len() - 1].to_vec())
                        .copied()
                        .unwrap_or(true)
                };
                if parent_completable {
                    first_total += 1;
                    if !f.hydrogen {
                        first_h += 1;
                    }
                    if !f.open_site {
                        first_o += 1;
                    }
                    if !f.closable {
                        first_c += 1;
                    }
                    if !f.valence {
                        first_v += 1;
                    }
                    if f.hydrogen && f.open_site && f.closable && f.valence {
                        first_uncaught += 1;
                    }
                }
            }
        }
        let non_total = records.len() - completable_count;
        println!(
            "{name}: prefixes={} completable={completable_count} non-completable={non_total} \
             caught hydrogen={non_h} open-site={non_o} closable={non_c} valence={non_v} uncaught={non_uncaught}",
            records.len(),
        );
        println!(
            "{name}: first-doomed={first_total} \
             caught hydrogen={first_h} open-site={first_o} closable={first_c} valence={first_v} uncaught={first_uncaught}"
        );
    }
}

#[test]
fn hydrogen_bound_at_root() {
    // Root C with zero hydrogens (id 1) for C2H6: the one remaining carbon
    // can bring at most 3 hydrogens, but 6 remain.
    let limits = Limits::new(6, 1).unwrap();
    let budget: Composition = [2, 6, 0, 0, 0, 0, 0, 0, 0, 0];
    let trace = vec![start_token(), root_token(1)];
    let state = replay_exact_v1(&trace, limits, budget).unwrap();
    let f = feasibility(&state, &budget, limits);
    assert!(!f.hydrogen, "hydrogen check fires");
    assert_eq!(
        first_doomed_step(&trace, &budget, limits).unwrap(),
        Some((1, DoomReason::HydrogenBound))
    );
    let cause = classify_dead_end(&trace, &budget, limits).unwrap();
    assert_eq!(cause.reason, Some(DoomReason::HydrogenBound));
    assert_eq!(cause.doomed_at, Some(1));
}

#[test]
fn no_open_site_after_closed_pair() {
    // Root C(H3) plus a child C(H3) on bond 1: residuals [0, 0] with heavy
    // atoms still missing, and the closed-prefix floor already passed them.
    let limits = Limits::new(6, 1).unwrap();
    let budget: Composition = [3, 8, 0, 1, 0, 0, 0, 0, 0, 0];
    let trace = vec![start_token(), root_token(4), add_token(4, 1, 0)];
    let state = replay_exact_v1(&trace, limits, budget).unwrap();
    assert_eq!(state.residual_valence(), &[0, 0]);
    let f = feasibility(&state, &budget, limits);
    assert!(f.hydrogen, "hydrogen still satisfiable");
    assert!(!f.open_site, "no open site remains");
    assert_eq!(
        first_doomed_step(&trace, &budget, limits).unwrap(),
        Some((2, DoomReason::NoOpenSite))
    );
}

#[test]
fn open_valence_without_atoms() {
    // All three carbons of C3H4 placed with H3+H1+H0 = H4 but residuals
    // [0, 1, 3]: the newest (3) cannot equal the others' sum (1).
    let limits = Limits::new(6, 1).unwrap();
    let budget: Composition = [3, 4, 0, 0, 0, 0, 0, 0, 0, 0];
    let trace = vec![
        start_token(),
        root_token(4),
        add_token(2, 1, 0),
        add_token(1, 1, 1),
    ];
    let state = replay_exact_v1(&trace, limits, budget).unwrap();
    assert_eq!(state.residual_valence(), &[0, 1, 3]);
    let f = feasibility(&state, &budget, limits);
    assert!(f.hydrogen, "hydrogens exact with m == 0");
    assert!(f.open_site, "vacuous with m == 0");
    assert!(!f.closable, "newest cannot close the others");
    assert_eq!(
        first_doomed_step(&trace, &budget, limits).unwrap(),
        Some((3, DoomReason::OpenValenceWithoutAtoms))
    );
}

#[test]
fn valence_bound_on_odd_total() {
    // Budget C1H2F1: root C(H2) leaves one F (V = 1) against R = 2, so every
    // total is odd.
    let limits = Limits::new(6, 1).unwrap();
    let budget: Composition = [1, 2, 0, 0, 1, 0, 0, 0, 0, 0];
    let trace = vec![start_token(), root_token(3)];
    let state = replay_exact_v1(&trace, limits, budget).unwrap();
    assert_eq!(state.residual_valence(), &[2]);
    let f = feasibility(&state, &budget, limits);
    assert!(f.hydrogen, "F needs no hydrogens");
    assert!(f.open_site, "the root is open");
    assert!(f.closable, "vacuous with atoms remaining");
    assert!(!f.valence, "R + V stays odd");
    assert_eq!(
        first_doomed_step(&trace, &budget, limits).unwrap(),
        Some((1, DoomReason::ValenceBound))
    );
}

#[test]
fn completable_prefixes_pass() {
    let limits = Limits::new(6, 1).unwrap();
    let molecules = vec![
        MolGraph::new(vec![4, 3, 9], vec![(0, 1, 1), (1, 2, 1)]).unwrap(),
        MolGraph::new(vec![4, 8, 4], vec![(0, 1, 1), (1, 2, 1)]).unwrap(),
        MolGraph::new(vec![3, 3, 3], vec![(0, 1, 1), (1, 2, 1), (0, 2, 1)]).unwrap(),
        MolGraph::new(vec![4, 1, 2], vec![(0, 1, 1), (1, 2, 3)]).unwrap(),
    ];
    for (i, graph) in molecules.iter().enumerate() {
        let budget = graph.composition();
        let trace = canonical_trace(graph, limits, 100_000).unwrap().trace;
        assert!(trace.len() >= 3, "molecule {i} has a 3-token prefix");
        let prefix = &trace[..3];
        let state = replay_exact(prefix, limits, budget).unwrap();
        let f = feasibility(&state, &budget, limits);
        assert!(
            f.hydrogen && f.open_site && f.closable && f.valence,
            "molecule {i} completable prefix passes: {f:?}"
        );
        assert_eq!(
            first_doomed_step(prefix, &budget, limits).unwrap(),
            None,
            "molecule {i} prefix not doomed"
        );
    }
    assert_eq!(
        first_doomed_step(&[start_token()], &[2, 6, 0, 1, 0, 0, 0, 0, 0, 0], limits).unwrap(),
        None,
        "the empty prefix is never doomed"
    );
}

#[test]
fn classify_dead_end_found_by_search() {
    let limits = Limits::new(6, 1).unwrap();
    // First dead-end prefix of the DFS under the current rule: kinds == 0,
    // hence no legal action. Small budgets leave no dead end at all under the
    // v2 lookahead, so the search walks the soundness budgets in order and
    // takes the first dead end it meets.
    let mut found: Option<(Composition, Vec<Token>)> = None;
    for (_, budget) in soundness_budgets() {
        let mut stack = vec![vec![start_token()]];
        let mut dead: Option<Vec<Token>> = None;
        while let Some(prefix) = stack.pop() {
            let mut state = TraceState::new_exact(limits, budget);
            for tok in &prefix {
                state.apply(*tok).unwrap();
            }
            if state.masks(blank()).kinds == 0 {
                dead = Some(prefix);
                break;
            }
            for child in extensions(&state) {
                let mut next = prefix.clone();
                next.push(child);
                stack.push(next);
            }
        }
        if let Some(trace) = dead {
            found = Some((budget, trace));
            break;
        }
    }
    let (budget, trace) = found.expect("some budget has a dead-end prefix");
    let cause = classify_dead_end(&trace, &budget, limits).unwrap();
    assert_eq!(cause.dead_end_step, trace.len());
    let end = replay_exact(&trace, limits, budget).unwrap();
    assert_eq!(cause.atoms, end.atoms());
    assert_eq!(
        cause.open_valence as usize,
        end.residual_valence()
            .iter()
            .map(|&r| r as usize)
            .sum::<usize>()
    );
    assert_eq!(cause.reason.is_some(), cause.doomed_at.is_some());
    if let Some(at) = cause.doomed_at {
        assert!(at < trace.len(), "doomed inside the trace");
    }
    let _ = STOP;
}
