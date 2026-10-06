//! MC1b/MC7 tests: the exact-completion grammar (`completion-exact-v2`): STOP
//! only when complete, the closed-prefix pointer rule, and the feasibility
//! lookahead (every ADD_ATOM and CLOSE_RING must leave a feasible state).
//!
//! The host [`TraceState`](mamba3::models::ms2::grammar::TraceState) gains an
//! exact mode, the device grammar helpers gain the budget flag value 2, and
//! the twins mirror it. Every device call is followed by [`check_launches`].

#![cfg(feature = "backend")]

use std::collections::HashSet;

use mamba3::backend::{Device, check_launches};
use mamba3::backends::Auto;
use mamba3::models::ms2::chem::Composition;
use mamba3::models::ms2::completion_diagnostics::first_doomed_step;
use mamba3::models::ms2::contract::candidate_status;
use mamba3::models::ms2::grammar::{
    ADD_ATOM, CLOSE_RING, Limits, START, STOP, Token, TraceState, canonical_trace, replay,
    replay_exact,
};
use mamba3::models::ms2::graph::MolGraph;
use mamba3::models::ms2::twin as host;
use mamba3::tensor::Tensor;
use mamba3::tensor::ops::index::IdTensor;
use mamba3::tensor::ops::ms2::{self, Ms2Constants, ReplayBuffers};
use mamba3::tensor::ops::random::Rng;

type R = Auto;
type E = f32;

fn dev() -> Device<R> {
    Device::<R>::default()
}

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

fn stop_token() -> Token {
    Token {
        kind: STOP,
        atom_type: 0,
        bond: 0,
        pointer: 0,
    }
}

/// The four canonical molecules, built from [`chem::ATOM_TYPES`](mamba3::models::ms2::chem::ATOM_TYPES)
/// ids: ethanol `[C(H3), C(H2), O(H1)]`, dimethyl ether `[C(H3), O(H0),
/// C(H3)]`, cyclopropane (three `C(H2)` in a ring) and propyne
/// `[C(H3), C(H0), C(H1)]` with a triple bond.
fn molecules() -> Vec<(&'static str, MolGraph)> {
    vec![
        (
            "ethanol",
            MolGraph::new(vec![4, 3, 9], vec![(0, 1, 1), (1, 2, 1)]).unwrap(),
        ),
        (
            "dimethyl-ether",
            MolGraph::new(vec![4, 8, 4], vec![(0, 1, 1), (1, 2, 1)]).unwrap(),
        ),
        (
            "cyclopropane",
            MolGraph::new(vec![3, 3, 3], vec![(0, 1, 1), (1, 2, 1), (0, 2, 1)]).unwrap(),
        ),
        (
            "propyne",
            MolGraph::new(vec![4, 1, 2], vec![(0, 1, 1), (1, 2, 3)]).unwrap(),
        ),
    ]
}

fn exact_limits() -> Limits {
    // The audit allowed 1 closure (V0 allows 4); 6 atoms cover every test
    // molecule (at most 4 heavy atoms).
    Limits::new(6, 1).unwrap()
}

#[test]
fn stop_is_legal_only_when_complete() {
    let limits = exact_limits();
    for (name, graph) in molecules() {
        let budget = graph.composition();
        let trace = canonical_trace(&graph, limits, 100_000).unwrap().trace;
        assert_eq!(
            trace.first().unwrap().kind,
            START,
            "{name}: starts with START"
        );
        assert_eq!(trace.last().unwrap().kind, STOP, "{name}: ends with STOP");
        // Walk every prefix: no STOP bit and no legal STOP before the
        // pre-STOP state, STOP offered exactly there.
        let mut st = TraceState::new_exact(limits, budget);
        assert!(st.is_exact(), "{name}: exact mode reports itself");
        for (i, tok) in trace.iter().enumerate() {
            let kinds = st.masks(blank()).kinds;
            if i + 1 == trace.len() {
                assert_eq!(tok.kind, STOP, "{name}: the final token is STOP");
                assert_ne!(
                    kinds & (1u32 << STOP),
                    0,
                    "{name}: STOP offered at completion"
                );
                assert!(
                    st.is_legal(stop_token()),
                    "{name}: STOP legal at completion"
                );
            } else {
                assert_eq!(kinds & (1u32 << STOP), 0, "{name} prefix {i}: no STOP bit");
                assert!(
                    !st.is_legal(stop_token()),
                    "{name} prefix {i}: STOP illegal"
                );
            }
            st.apply(*tok).unwrap();
        }
        assert!(st.is_complete(), "{name}: the full trace completes");
        assert_eq!(st.used(), &budget, "{name}: used equals the budget");
        // The full trace replays exactly.
        let end = replay_exact(&trace, limits, budget).unwrap();
        assert!(end.is_complete(), "{name}: replay_exact completes");
        // Dropping the last non-STOP token makes the STOP fail: the prefix
        // is unchanged, so the STOP itself is the first illegal step.
        let mut short = trace.clone();
        let drop = short.iter().rposition(|t| t.kind != STOP).unwrap();
        short.remove(drop);
        let err = replay_exact(&short, limits, budget)
            .err()
            .expect("the early STOP is illegal under exact completion");
        assert!(
            err.to_string()
                .contains(&format!("step {}", short.len() - 1)),
            "{name}: the early STOP fails at its own index: {err}"
        );
        // Unchanged subgraph semantics: the same shortened trace replays
        // fine with an upper-bound budget.
        replay(&short, limits, Some(budget)).unwrap();
    }
}

/// Depth-first search over every token the masks admit (kinds, then atom
/// types, bonds and pointers, each conditioned as `masks` documents) under
/// `new_exact`. Records the canonical identity of every completed state,
/// every dead-end prefix (`kinds == 0`, hence no legal action) and the dead-end
/// count.
fn enumerate_completed(
    budget: Composition,
    limits: Limits,
) -> (HashSet<Vec<Token>>, Vec<Vec<Token>>) {
    let mut identities: HashSet<Vec<Token>> = HashSet::new();
    let mut dead_prefixes: Vec<Vec<Token>> = Vec::new();
    fn dfs(
        prefix: &mut Vec<Token>,
        limits: Limits,
        budget: Composition,
        identities: &mut HashSet<Vec<Token>>,
        dead_prefixes: &mut Vec<Vec<Token>>,
    ) {
        let mut st = TraceState::new_exact(limits, budget);
        for tok in prefix.iter() {
            st.apply(*tok)
                .expect("the search only extends legal tokens");
        }
        let kinds = st.masks(blank()).kinds;
        if kinds == 0 {
            assert!(!st.has_legal_action(), "kinds == 0 means no legal action");
            dead_prefixes.push(prefix.clone());
            return;
        }
        if kinds & (1u32 << STOP) != 0 {
            // STOP is offered only on a complete molecule.
            assert!(st.is_complete(), "STOP implies a complete molecule");
            assert_eq!(st.used(), &budget, "a completed state uses the budget");
            assert!(
                st.residual_valence().iter().all(|&r| r == 0),
                "a completed state has no open valence"
            );
            let graph = st.graph().unwrap();
            assert_eq!(
                graph.composition(),
                budget,
                "the graph has the budget formula"
            );
            let canon = canonical_trace(&graph, limits, 100_000).unwrap().trace;
            identities.insert(canon);
        }
        if kinds & (1u32 << ADD_ATOM) != 0 {
            if st.step() == 1 {
                let types = st
                    .masks(Token {
                        kind: ADD_ATOM,
                        ..blank()
                    })
                    .atom_types;
                for id in 1..=17u8 {
                    if types & (1u32 << id) == 0 {
                        continue;
                    }
                    prefix.push(Token {
                        kind: ADD_ATOM,
                        atom_type: id,
                        ..blank()
                    });
                    dfs(prefix, limits, budget, identities, dead_prefixes);
                    prefix.pop();
                }
            } else {
                let types = st
                    .masks(Token {
                        kind: ADD_ATOM,
                        ..blank()
                    })
                    .atom_types;
                for id in 1..=17u8 {
                    if types & (1u32 << id) == 0 {
                        continue;
                    }
                    let bonds = st
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
                        let pointers = st
                            .masks(Token {
                                kind: ADD_ATOM,
                                atom_type: id,
                                bond: b,
                                ..blank()
                            })
                            .pointers;
                        for p in 0..32u8 {
                            if pointers & (1u32 << p) == 0 {
                                continue;
                            }
                            prefix.push(Token {
                                kind: ADD_ATOM,
                                atom_type: id,
                                bond: b,
                                pointer: p,
                            });
                            dfs(prefix, limits, budget, identities, dead_prefixes);
                            prefix.pop();
                        }
                    }
                }
            }
        }
        if kinds & (1u32 << CLOSE_RING) != 0 {
            let bonds = st
                .masks(Token {
                    kind: CLOSE_RING,
                    ..blank()
                })
                .bonds;
            for b in 1..=3u8 {
                if bonds & (1u32 << b) == 0 {
                    continue;
                }
                let pointers = st
                    .masks(Token {
                        kind: CLOSE_RING,
                        bond: b,
                        ..blank()
                    })
                    .pointers;
                for p in 0..32u8 {
                    if pointers & (1u32 << p) == 0 {
                        continue;
                    }
                    prefix.push(Token {
                        kind: CLOSE_RING,
                        bond: b,
                        pointer: p,
                        ..blank()
                    });
                    dfs(prefix, limits, budget, identities, dead_prefixes);
                    prefix.pop();
                }
            }
        }
        // A STOP extension stays stopped with no legal action, so the search
        // records the completed identity above and never extends past STOP.
    }
    let mut prefix = vec![start_token()];
    dfs(
        &mut prefix,
        limits,
        budget,
        &mut identities,
        &mut dead_prefixes,
    );
    (identities, dead_prefixes)
}

#[test]
fn exact_enumeration_matches_known_isomer_counts() {
    let limits = exact_limits();
    // Budgets over C/H/N/O only: `type_fits` admits exactly the C/N/O atom
    // types (ids 1-9), so no extra type restriction is needed.
    let cases: [(&str, Composition, usize, Option<usize>); 5] = [
        ("C2H6O", [2, 6, 0, 1, 0, 0, 0, 0, 0, 0], 2, Some(242)),
        ("C3H8O", [3, 8, 0, 1, 0, 0, 0, 0, 0, 0], 3, Some(3439)),
        ("C2H7N", [2, 7, 1, 0, 0, 0, 0, 0, 0, 0], 2, Some(489)),
        ("C3H4", [3, 4, 0, 0, 0, 0, 0, 0, 0, 0], 3, Some(205)),
        ("C4H8", [4, 8, 0, 0, 0, 0, 0, 0, 0, 0], 5, None),
    ];
    for (name, budget, want, old_dead) in cases {
        let (identities, dead_prefixes) = enumerate_completed(budget, limits);
        let dead_ends = dead_prefixes.len();
        println!(
            "{name}: {} identities, {dead_ends} dead ends",
            identities.len()
        );
        assert_eq!(
            identities.len(),
            want,
            "{name}: distinct completed identities"
        );
        // Dead-end counts after the closed-prefix rule. Before the rule (MC1
        // measurement, STOP-only exact mode) they were C2H6O 242, C3H8O 3439,
        // C2H7N 489, C3H4 205; the rule must strictly reduce each of them.
        if let Some(old) = old_dead {
            assert!(
                dead_ends < old,
                "{name}: {dead_ends} dead ends is strictly below the pre-rule {old}"
            );
        }
        if name == "C2H6O" {
            // Under the v2 lookahead C2H6O keeps no dead-end prefix at all:
            // every feasible prefix completes.
            assert_eq!(
                dead_ends, 0,
                "C2H6O leaves no dead-end prefix under the lookahead"
            );
        }
    }
}

/// Depth-first search over every token the masks admit under the upper-bound
/// grammar (`TraceState::new` with the budget), keeping the canonical identity
/// of every STOP state that [`TraceState::is_complete`] (the reference set for
/// the soundness check). Every prefix that reaches a complete state is also
/// checked to replay under [`replay_exact`]: the closed-prefix rule must not
/// forbid any completing trace.
fn enumerate_upper_bound(
    budget: Composition,
    limits: Limits,
) -> (HashSet<Vec<Token>>, Vec<Vec<Token>>) {
    let mut identities: HashSet<Vec<Token>> = HashSet::new();
    let mut completing: Vec<Vec<Token>> = Vec::new();
    fn dfs(
        prefix: &mut Vec<Token>,
        limits: Limits,
        budget: Composition,
        identities: &mut HashSet<Vec<Token>>,
        completing: &mut Vec<Vec<Token>>,
    ) {
        let mut st = TraceState::new(limits, Some(budget));
        for tok in prefix.iter() {
            st.apply(*tok)
                .expect("the search only extends legal tokens");
        }
        let kinds = st.masks(blank()).kinds;
        if kinds == 0 {
            return;
        }
        if kinds & (1u32 << STOP) != 0 && st.is_complete() {
            assert_eq!(st.used(), &budget, "a completed state uses the budget");
            assert!(
                st.residual_valence().iter().all(|&r| r == 0),
                "a completed state has no open valence"
            );
            let graph = st.graph().unwrap();
            assert_eq!(
                graph.composition(),
                budget,
                "the graph has the budget formula"
            );
            let canon = canonical_trace(&graph, limits, 100_000).unwrap().trace;
            identities.insert(canon);
            // The completing trace itself (prefix plus STOP) must stay legal
            // under the exact rule: soundness of the closed-prefix pruning.
            let mut full = prefix.clone();
            full.push(stop_token());
            replay_exact(&full, limits, budget)
                .expect("every upper-bound completing trace replays exactly");
            completing.push(full);
        }
        if kinds & (1u32 << ADD_ATOM) != 0 {
            if st.step() == 1 {
                let types = st
                    .masks(Token {
                        kind: ADD_ATOM,
                        ..blank()
                    })
                    .atom_types;
                for id in 1..=17u8 {
                    if types & (1u32 << id) == 0 {
                        continue;
                    }
                    prefix.push(Token {
                        kind: ADD_ATOM,
                        atom_type: id,
                        ..blank()
                    });
                    dfs(prefix, limits, budget, identities, completing);
                    prefix.pop();
                }
            } else {
                let types = st
                    .masks(Token {
                        kind: ADD_ATOM,
                        ..blank()
                    })
                    .atom_types;
                for id in 1..=17u8 {
                    if types & (1u32 << id) == 0 {
                        continue;
                    }
                    let bonds = st
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
                        let pointers = st
                            .masks(Token {
                                kind: ADD_ATOM,
                                atom_type: id,
                                bond: b,
                                ..blank()
                            })
                            .pointers;
                        for p in 0..32u8 {
                            if pointers & (1u32 << p) == 0 {
                                continue;
                            }
                            prefix.push(Token {
                                kind: ADD_ATOM,
                                atom_type: id,
                                bond: b,
                                pointer: p,
                            });
                            dfs(prefix, limits, budget, identities, completing);
                            prefix.pop();
                        }
                    }
                }
            }
        }
        if kinds & (1u32 << CLOSE_RING) != 0 {
            let bonds = st
                .masks(Token {
                    kind: CLOSE_RING,
                    ..blank()
                })
                .bonds;
            for b in 1..=3u8 {
                if bonds & (1u32 << b) == 0 {
                    continue;
                }
                let pointers = st
                    .masks(Token {
                        kind: CLOSE_RING,
                        bond: b,
                        ..blank()
                    })
                    .pointers;
                for p in 0..32u8 {
                    if pointers & (1u32 << p) == 0 {
                        continue;
                    }
                    prefix.push(Token {
                        kind: CLOSE_RING,
                        bond: b,
                        pointer: p,
                        ..blank()
                    });
                    dfs(prefix, limits, budget, identities, completing);
                    prefix.pop();
                }
            }
        }
    }
    let mut prefix = vec![start_token()];
    dfs(
        &mut prefix,
        limits,
        budget,
        &mut identities,
        &mut completing,
    );
    (identities, completing)
}

#[test]
fn closed_prefix_rule_is_sound() {
    let limits = Limits::new(6, 1).unwrap();
    let budgets: [(&str, Composition); 6] = [
        ("C2H6O", [2, 6, 0, 1, 0, 0, 0, 0, 0, 0]),
        ("C3H8O", [3, 8, 0, 1, 0, 0, 0, 0, 0, 0]),
        ("C2H7N", [2, 7, 1, 0, 0, 0, 0, 0, 0, 0]),
        ("C3H4", [3, 4, 0, 0, 0, 0, 0, 0, 0, 0]),
        ("C4H8", [4, 8, 0, 0, 0, 0, 0, 0, 0, 0]),
        ("C3H7NO", [3, 7, 1, 1, 0, 0, 0, 0, 0, 0]),
    ];
    for (name, budget) in budgets {
        let (upper, _) = enumerate_upper_bound(budget, limits);
        let (pruned, _) = enumerate_completed(budget, limits);
        println!(
            "{name}: upper-bound {} identities, exact {} identities",
            upper.len(),
            pruned.len()
        );
        assert_eq!(
            upper, pruned,
            "{name}: the closed-prefix rule removes no molecule"
        );
    }
}

#[test]
fn feasibility_lookahead_is_sound() {
    // The v2 lookahead removes no molecule: the completed-identity set under
    // `new_exact` equals the upper-bound reference set, and every completing
    // trace of the reference DFS is legal token by token under `new_exact`
    // (every prefix of a completing trace passes all four checks, since the
    // checks are necessary for completability).
    let limits = Limits::new(6, 1).unwrap();
    let budgets: [(&str, Composition); 6] = [
        ("C2H6O", [2, 6, 0, 1, 0, 0, 0, 0, 0, 0]),
        ("C3H8O", [3, 8, 0, 1, 0, 0, 0, 0, 0, 0]),
        ("C2H7N", [2, 7, 1, 0, 0, 0, 0, 0, 0, 0]),
        ("C3H4", [3, 4, 0, 0, 0, 0, 0, 0, 0, 0]),
        ("C4H8", [4, 8, 0, 0, 0, 0, 0, 0, 0, 0]),
        ("C3H7NO", [3, 7, 1, 1, 0, 0, 0, 0, 0, 0]),
    ];
    for (name, budget) in budgets {
        let (upper, completing) = enumerate_upper_bound(budget, limits);
        let (exact, _) = enumerate_completed(budget, limits);
        println!(
            "{name}: upper-bound {} identities, exact {} identities, {} completing traces",
            upper.len(),
            exact.len(),
            completing.len()
        );
        assert_eq!(upper, exact, "{name}: the lookahead removes no molecule");
        assert!(
            !completing.is_empty(),
            "{name}: the reference search completes at least one trace"
        );
        for trace in &completing {
            let mut st = TraceState::new_exact(limits, budget);
            for (i, tok) in trace.iter().enumerate() {
                assert!(
                    st.is_legal(*tok),
                    "{name}: completing trace stays legal at token {i} ({tok:?})"
                );
                st.apply(*tok).unwrap();
            }
            assert!(st.is_complete(), "{name}: the completing trace completes");
        }
    }
}

/// Every prefix the exact-mode search visits (all of them v2-legal by
/// construction).
fn collect_prefixes(budget: Composition, limits: Limits) -> Vec<Vec<Token>> {
    fn dfs(
        prefix: &mut Vec<Token>,
        limits: Limits,
        budget: Composition,
        out: &mut Vec<Vec<Token>>,
    ) {
        out.push(prefix.clone());
        let mut st = TraceState::new_exact(limits, budget);
        for tok in prefix.iter() {
            st.apply(*tok)
                .expect("the search only extends legal tokens");
        }
        let kinds = st.masks(blank()).kinds;
        if kinds == 0 {
            return;
        }
        if kinds & (1u32 << ADD_ATOM) != 0 {
            if st.step() == 1 {
                let types = st
                    .masks(Token {
                        kind: ADD_ATOM,
                        ..blank()
                    })
                    .atom_types;
                for id in 1..=17u8 {
                    if types & (1u32 << id) == 0 {
                        continue;
                    }
                    prefix.push(Token {
                        kind: ADD_ATOM,
                        atom_type: id,
                        ..blank()
                    });
                    dfs(prefix, limits, budget, out);
                    prefix.pop();
                }
            } else {
                let types = st
                    .masks(Token {
                        kind: ADD_ATOM,
                        ..blank()
                    })
                    .atom_types;
                for id in 1..=17u8 {
                    if types & (1u32 << id) == 0 {
                        continue;
                    }
                    let bonds = st
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
                        let pointers = st
                            .masks(Token {
                                kind: ADD_ATOM,
                                atom_type: id,
                                bond: b,
                                ..blank()
                            })
                            .pointers;
                        for p in 0..32u8 {
                            if pointers & (1u32 << p) == 0 {
                                continue;
                            }
                            prefix.push(Token {
                                kind: ADD_ATOM,
                                atom_type: id,
                                bond: b,
                                pointer: p,
                            });
                            dfs(prefix, limits, budget, out);
                            prefix.pop();
                        }
                    }
                }
            }
        }
        if kinds & (1u32 << CLOSE_RING) != 0 {
            let bonds = st
                .masks(Token {
                    kind: CLOSE_RING,
                    ..blank()
                })
                .bonds;
            for b in 1..=3u8 {
                if bonds & (1u32 << b) == 0 {
                    continue;
                }
                let pointers = st
                    .masks(Token {
                        kind: CLOSE_RING,
                        bond: b,
                        ..blank()
                    })
                    .pointers;
                for p in 0..32u8 {
                    if pointers & (1u32 << p) == 0 {
                        continue;
                    }
                    prefix.push(Token {
                        kind: CLOSE_RING,
                        bond: b,
                        pointer: p,
                        ..blank()
                    });
                    dfs(prefix, limits, budget, out);
                    prefix.pop();
                }
            }
        }
    }
    let mut out = Vec::new();
    let mut prefix = vec![start_token()];
    dfs(&mut prefix, limits, budget, &mut out);
    out
}

/// Deterministic 64-bit mix for fixed-seed sampling (splitmix64).
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

#[test]
fn device_masks_match_host_on_enumerated_prefixes() {
    let limits = Limits::new(6, 1).unwrap();
    let (a, rmax) = (6usize, 1u32);
    let device = dev();
    let constants = Ms2Constants::new(&device);
    let c2: Composition = [2, 6, 0, 1, 0, 0, 0, 0, 0, 0];
    let c3n: Composition = [3, 7, 1, 1, 0, 0, 0, 0, 0, 0];
    // All C2H6O exact-mode prefixes, plus 256 C3H7NO prefixes sampled with a
    // fixed seed.
    let mut prefixes: Vec<(Vec<Token>, Composition)> = collect_prefixes(c2, limits)
        .into_iter()
        .map(|p| (p, c2))
        .collect();
    let all_c3n = collect_prefixes(c3n, limits);
    assert!(
        all_c3n.len() >= 256,
        "C3H7NO visits at least 256 prefixes, got {}",
        all_c3n.len()
    );
    let mut seed = 0xC70FFEEu64;
    let mut picked = vec![false; all_c3n.len()];
    let mut n_picked = 0usize;
    while n_picked < 256 {
        let i = (splitmix64(&mut seed) % all_c3n.len() as u64) as usize;
        if !picked[i] {
            picked[i] = true;
            n_picked += 1;
        }
    }
    for (i, prefix) in all_c3n.into_iter().enumerate() {
        if picked[i] {
            prefixes.push((prefix, c3n));
        }
    }
    println!(
        "enumerated prefixes: {} total (C2H6O all, C3H7NO 256 sampled)",
        prefixes.len()
    );
    // Rows: for every prefix, the prefix extended by each offered token (all
    // legal by construction), plus for each prefix the first token the v1
    // rules allow but v2 forbids (when one exists).
    let mut traces: Vec<Vec<Token>> = Vec::new();
    let mut budgets: Vec<Composition> = Vec::new();
    let mut doomed_rows = 0usize;
    for (prefix, budget) in &prefixes {
        let mut st = TraceState::new_exact(limits, *budget);
        for tok in prefix.iter() {
            st.apply(*tok).expect("collected prefixes stay legal");
        }
        let kinds = st.masks(blank()).kinds;
        if kinds & (1u32 << STOP) != 0 {
            let mut trace = prefix.clone();
            trace.push(stop_token());
            traces.push(trace);
            budgets.push(*budget);
        }
        if kinds & (1u32 << ADD_ATOM) != 0 {
            if st.step() == 1 {
                let types = st
                    .masks(Token {
                        kind: ADD_ATOM,
                        ..blank()
                    })
                    .atom_types;
                for id in 1..=17u8 {
                    if types & (1u32 << id) == 0 {
                        continue;
                    }
                    let mut trace = prefix.clone();
                    trace.push(Token {
                        kind: ADD_ATOM,
                        atom_type: id,
                        ..blank()
                    });
                    traces.push(trace);
                    budgets.push(*budget);
                }
            } else {
                let types = st
                    .masks(Token {
                        kind: ADD_ATOM,
                        ..blank()
                    })
                    .atom_types;
                for id in 1..=17u8 {
                    if types & (1u32 << id) == 0 {
                        continue;
                    }
                    let bonds = st
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
                        let pointers = st
                            .masks(Token {
                                kind: ADD_ATOM,
                                atom_type: id,
                                bond: b,
                                ..blank()
                            })
                            .pointers;
                        for p in 0..32u8 {
                            if pointers & (1u32 << p) == 0 {
                                continue;
                            }
                            let mut trace = prefix.clone();
                            trace.push(Token {
                                kind: ADD_ATOM,
                                atom_type: id,
                                bond: b,
                                pointer: p,
                            });
                            traces.push(trace);
                            budgets.push(*budget);
                        }
                    }
                }
            }
        }
        if kinds & (1u32 << CLOSE_RING) != 0 {
            let bonds = st
                .masks(Token {
                    kind: CLOSE_RING,
                    ..blank()
                })
                .bonds;
            for b in 1..=3u8 {
                if bonds & (1u32 << b) == 0 {
                    continue;
                }
                let pointers = st
                    .masks(Token {
                        kind: CLOSE_RING,
                        bond: b,
                        ..blank()
                    })
                    .pointers;
                for p in 0..32u8 {
                    if pointers & (1u32 << p) == 0 {
                        continue;
                    }
                    let mut trace = prefix.clone();
                    trace.push(Token {
                        kind: CLOSE_RING,
                        bond: b,
                        pointer: p,
                        ..blank()
                    });
                    traces.push(trace);
                    budgets.push(*budget);
                }
            }
        }
        // The first v1-legal but v2-illegal token at this prefix, if any: the
        // v1 oracle is a clone with `apply_v1`, the v2 verdict `is_legal`.
        let mut candidates: Vec<Token> = vec![start_token(), stop_token()];
        if st.step() == 1 {
            for id in 1..=17u8 {
                candidates.push(Token {
                    kind: ADD_ATOM,
                    atom_type: id,
                    ..blank()
                });
            }
        } else if st.step() > 1 {
            for id in 1..=17u8 {
                for b in 1..=3u8 {
                    for p in 0..st.atoms().max(1) as u8 {
                        candidates.push(Token {
                            kind: ADD_ATOM,
                            atom_type: id,
                            bond: b,
                            pointer: p,
                        });
                    }
                }
            }
            for b in 1..=3u8 {
                for p in 0..st.atoms().max(1) as u8 {
                    candidates.push(Token {
                        kind: CLOSE_RING,
                        bond: b,
                        pointer: p,
                        ..blank()
                    });
                }
            }
        }
        for tok in candidates {
            let mut v1 = st.clone();
            if v1.apply_v1(tok).is_ok() && !st.is_legal(tok) {
                let mut trace = prefix.clone();
                trace.push(tok);
                traces.push(trace);
                budgets.push(*budget);
                doomed_rows += 1;
                break;
            }
        }
    }
    println!(
        "{} extended rows ({} with a v1-allowed v2-forbidden last token)",
        traces.len(),
        doomed_rows
    );
    assert!(!traces.is_empty(), "the enumeration yields replay rows");
    let rows = traces.len();
    let t = traces.iter().map(|tr| tr.len()).max().unwrap();
    let mut tokens = vec![0u32; rows * t * 4];
    let mut meta = vec![0u32; rows * 12];
    for (r, (trace, budget)) in traces.iter().zip(budgets.iter()).enumerate() {
        for (s, tok) in trace.iter().enumerate() {
            tokens[(r * t + s) * 4] = u32::from(tok.kind);
            tokens[(r * t + s) * 4 + 1] = u32::from(tok.atom_type);
            tokens[(r * t + s) * 4 + 2] = u32::from(tok.bond);
            tokens[(r * t + s) * 4 + 3] = u32::from(tok.pointer);
        }
        meta[r * 12] = trace.len() as u32;
        meta[r * 12 + 1] = 2;
        for (e, count) in budget.iter().enumerate() {
            meta[r * 12 + 2 + e] = u32::from(*count);
        }
    }
    let tokens_t = IdTensor::from_slice(&tokens, vec![rows, t, 4], &device).unwrap();
    let meta_t = IdTensor::from_slice(&meta, vec![rows, 12], &device).unwrap();
    let out = ReplayBuffers::poisoned(rows, t, a, &device).unwrap();
    ms2::grammar_replay(&tokens_t, &meta_t, &constants, a as u32, rmax, &out).unwrap();
    check_launches(&device).unwrap();
    let replay = out.replay.try_to_vec().unwrap();
    let atoms = out.atoms.try_to_vec().unwrap();
    for (r, (trace, budget)) in traces.iter().zip(budgets.iter()).enumerate() {
        let host = host_replay(trace, limits, *budget, 2, a);
        let length = trace.len();
        let illegal = host.first_illegal;
        for s in 0..t {
            let base = (r * t + s) * (4 + a);
            if s >= length || (illegal != u32::MAX && s > illegal as usize) {
                for w in 0..4 + a {
                    assert_eq!(replay[base + w], 0, "row {r} step {s}: zero word {w}");
                }
                continue;
            }
            assert_eq!(replay[base], host.kinds[s], "row {r} step {s}: kinds");
            assert_eq!(replay[base + 1], host.types[s], "row {r} step {s}: types");
            assert_eq!(replay[base + 2], host.bonds[s], "row {r} step {s}: bonds");
            assert_eq!(
                replay[base + 3],
                host.pointers[s],
                "row {r} step {s}: pointers"
            );
            for j in 0..a {
                assert_eq!(
                    replay[base + 4 + j],
                    host.resids[s][j],
                    "row {r} step {s}: residual {j}"
                );
            }
        }
        for j in 0..a {
            assert_eq!(
                atoms[r * (a + 1) + j],
                host.add_steps[j],
                "row {r}: atom {j} step"
            );
        }
        assert_eq!(atoms[r * (a + 1) + a], illegal, "row {r}: first illegal");
    }
}

#[test]
fn lookahead_removes_dead_ends() {
    // The exact-mode DFS dead-end prefix counts under v2, strictly below the
    // v1 (closed-prefix, no-lookahead) measurements. The dead-end states that
    // remain pass every necessary condition by construction (each of their
    // tokens was legal when applied, hence left a feasible state): the
    // lookahead cannot see them coming.
    let limits = Limits::new(6, 1).unwrap();
    let cases: [(&str, Composition, usize); 5] = [
        ("C2H6O", [2, 6, 0, 1, 0, 0, 0, 0, 0, 0], 174),
        ("C3H8O", [3, 8, 0, 1, 0, 0, 0, 0, 0, 0], 1707),
        ("C2H7N", [2, 7, 1, 0, 0, 0, 0, 0, 0, 0], 336),
        ("C3H4", [3, 4, 0, 0, 0, 0, 0, 0, 0, 0], 131),
        ("C4H8", [4, 8, 0, 0, 0, 0, 0, 0, 0, 0], 1457),
    ];
    for (name, budget, v1_dead) in cases {
        let (_, dead_prefixes) = enumerate_completed(budget, limits);
        let mut violated_h = 0usize;
        let mut violated_o = 0usize;
        let mut violated_c = 0usize;
        let mut violated_v = 0usize;
        for prefix in &dead_prefixes {
            let end = replay_exact(prefix, limits, budget)
                .expect("a dead-end prefix reached under v2 replays");
            assert_eq!(
                end.masks(blank()).kinds,
                0,
                "{name}: the dead-end prefix has no legal action"
            );
            let f = end.feasibility();
            if !f.hydrogen {
                violated_h += 1;
            }
            if !f.open_site {
                violated_o += 1;
            }
            if !f.closable {
                violated_c += 1;
            }
            if !f.valence {
                violated_v += 1;
            }
            assert!(
                f.all(),
                "{name}: a dead-end state reached under v2 passes every check: {prefix:?} {f:?}"
            );
        }
        println!(
            "{name}: v1 {v1_dead} dead ends, v2 {} dead ends (state violations: hydrogen={violated_h} open-site={violated_o} closable={violated_c} valence={violated_v})",
            dead_prefixes.len()
        );
        assert!(
            dead_prefixes.len() < v1_dead,
            "{name}: {} dead ends is strictly below the v1 {v1_dead}",
            dead_prefixes.len()
        );
    }
}

/// One host replay of a trace under the given budget flag: 0 no budget, 1
/// upper bound, 2 exact completion.
#[test]
fn closed_prefix_pointer_masks() {
    let limits = exact_limits();
    // Budget C3H8O covers every prefix below (at most 3 carbons, 6 hydrogens).
    let budget: Composition = [3, 8, 0, 1, 0, 0, 0, 0, 0, 0];
    // Prefix with atoms 0 and 1 both open: root C(H2, id 3, residual 2), then
    // a child C(H1, id 2, residual 3) on pointer 0 with bond 1. Residuals are
    // [1, 2]: atom 0 stays open, so an exact ADD on pointer 1 would abandon
    // it forever. Under v2 the second ADD is itself doomed (the remaining
    // carbon can bring at most 3 of the 5 hydrogens still missing), so the
    // prefix dies at step 2; under the upper bound it stays legal and the
    // closed-prefix distinction shows on the pointer mask.
    let open_prefix = vec![
        start_token(),
        Token {
            kind: ADD_ATOM,
            atom_type: 3,
            ..blank()
        },
        Token {
            kind: ADD_ATOM,
            atom_type: 2,
            bond: 1,
            pointer: 0,
        },
    ];
    // Upper bound: the prefix stays legal, pointer 1 is offered (no
    // closed-prefix rule), and the probe is legal.
    let mut st = TraceState::new(limits, Some(budget));
    for tok in &open_prefix {
        assert!(st.is_legal(*tok), "flag 1: prefix stays legal");
        st.apply(*tok).unwrap();
    }
    assert_eq!(st.residual_valence(), &[1, 2], "flag 1: both open");
    let taken = Token {
        kind: ADD_ATOM,
        atom_type: 3,
        bond: 1,
        ..blank()
    };
    assert_eq!(
        st.masks(taken).pointers,
        (1u32 << 0) | (1u32 << 1),
        "flag 1: ADD pointer mask"
    );
    let probe = Token {
        kind: ADD_ATOM,
        atom_type: 3,
        bond: 1,
        pointer: 1,
    };
    assert!(st.is_legal(probe), "flag 1: pointer 1 stays legal");
    // Exact mode: the doomed ADD at step 2 is the first illegal token.
    let host = host_replay(&open_prefix, limits, budget, 2, 6);
    assert_eq!(
        host.first_illegal, 2,
        "flag 2: the hydrogen-doomed ADD fails at itself"
    );
    // A v2-legal prefix with the same shape: root C(H3, id 4, residual 1),
    // then a child C(H2, id 3) on pointer 0 with bond 1. Residuals are
    // [0, 1]: atom 0 is closed, so pointer 1 passes the closed-prefix rule,
    // and the prefix completes (via C(H2) then O(H1) on the newest atom).
    let live_prefix = vec![
        start_token(),
        Token {
            kind: ADD_ATOM,
            atom_type: 4,
            ..blank()
        },
        Token {
            kind: ADD_ATOM,
            atom_type: 3,
            bond: 1,
            pointer: 0,
        },
    ];
    for flag in [1u32, 2] {
        let mut st = match flag {
            2 => TraceState::new_exact(limits, budget),
            _ => TraceState::new(limits, Some(budget)),
        };
        for tok in &live_prefix {
            assert!(st.is_legal(*tok), "flag {flag}: prefix stays legal");
            st.apply(*tok).unwrap();
        }
        assert_eq!(st.residual_valence(), &[0, 1], "flag {flag}: 0 closed");
        // C(H2, id 3) on pointer 1 with bond 1 completes the chain: legal in
        // both modes.
        let probe = Token {
            kind: ADD_ATOM,
            atom_type: 3,
            bond: 1,
            pointer: 1,
        };
        assert!(st.is_legal(probe), "flag {flag}: pointer 1 stays legal");
        let taken = Token {
            kind: ADD_ATOM,
            atom_type: 3,
            bond: 1,
            ..blank()
        };
        assert_ne!(
            st.masks(taken).pointers & (1u32 << 1),
            0,
            "flag {flag}: mask keeps pointer 1"
        );
        // C(H3, id 4) on pointer 1 with bond 1 strands the remaining oxygen
        // (no open site left for it): legal under the upper bound, doomed
        // under exact completion.
        let doomed = Token {
            kind: ADD_ATOM,
            atom_type: 4,
            bond: 1,
            pointer: 1,
        };
        assert_eq!(
            st.is_legal(doomed),
            flag != 2,
            "flag {flag}: the stranding ADD is legal only above exact"
        );
        let taken4 = Token {
            kind: ADD_ATOM,
            atom_type: 4,
            bond: 1,
            ..blank()
        };
        assert_eq!(
            st.masks(taken4).pointers & (1u32 << 1) != 0,
            flag != 2,
            "flag {flag}: the stranding pointer is masked only above exact"
        );
    }
}

fn host_replay(
    trace: &[Token],
    limits: Limits,
    budget: Composition,
    flag: u32,
    a: usize,
) -> host::ReplayRows {
    host::replay_rows(trace, limits, budget, flag, a)
}

#[test]
fn device_replay_matches_host_under_all_three_flags() {
    let limits = exact_limits();
    let (a, rmax) = (6usize, 1u32);
    let device = dev();
    let constants = Ms2Constants::new(&device);
    // Rows: every molecule's full canonical trace under flags 0, 1, 2, plus
    // the shortened trace (last non-STOP token removed, early STOP) under
    // flags 1 and 2, plus a closed-prefix-breaking trace (legal under flags
    // 0/1, first illegal at the offending ADD under flag 2) under all three
    // flags.
    let mut traces: Vec<Vec<Token>> = Vec::new();
    let mut budgets: Vec<Composition> = Vec::new();
    let mut flags: Vec<u32> = Vec::new();
    let mut names: Vec<String> = Vec::new();
    for (name, graph) in molecules() {
        let budget = graph.composition();
        let full = canonical_trace(&graph, limits, 100_000).unwrap().trace;
        let mut short = full.clone();
        short.remove(short.iter().rposition(|t| t.kind != STOP).unwrap());
        for flag in [0u32, 1, 2] {
            traces.push(full.clone());
            budgets.push(budget);
            flags.push(flag);
            names.push(format!("{name}-full-{flag}"));
        }
        for flag in [1u32, 2] {
            traces.push(short.clone());
            budgets.push(budget);
            flags.push(flag);
            names.push(format!("{name}-short-{flag}"));
        }
    }
    // Closed-prefix breaker: root C(H2, id 3) then child C(H1, id 2) on
    // pointer 0 with bond 1 leaves atom 0 open (residuals [1, 2]); a third
    // atom on pointer 1 is legal under the subgraph semantics but abandons
    // atom 0, so it is the first illegal step under exact completion. The
    // trailing STOP keeps the flag-0/1 rows legal to the end.
    let breaker_budget: Composition = [3, 8, 0, 1, 0, 0, 0, 0, 0, 0];
    let breaker = vec![
        start_token(),
        Token {
            kind: ADD_ATOM,
            atom_type: 3,
            ..blank()
        },
        Token {
            kind: ADD_ATOM,
            atom_type: 2,
            bond: 1,
            pointer: 0,
        },
        Token {
            kind: ADD_ATOM,
            atom_type: 4,
            bond: 1,
            pointer: 1,
        },
        stop_token(),
    ];
    for flag in [0u32, 1, 2] {
        traces.push(breaker.clone());
        budgets.push(breaker_budget);
        flags.push(flag);
        names.push(format!("closed-break-{flag}"));
    }
    let rows = traces.len();
    let t = traces.iter().map(|tr| tr.len()).max().unwrap();
    let mut tokens = vec![0u32; rows * t * 4];
    let mut meta = vec![0u32; rows * 12];
    for (r, ((trace, budget), flag)) in traces
        .iter()
        .zip(budgets.iter())
        .zip(flags.iter())
        .enumerate()
    {
        for (s, tok) in trace.iter().enumerate() {
            tokens[(r * t + s) * 4] = u32::from(tok.kind);
            tokens[(r * t + s) * 4 + 1] = u32::from(tok.atom_type);
            tokens[(r * t + s) * 4 + 2] = u32::from(tok.bond);
            tokens[(r * t + s) * 4 + 3] = u32::from(tok.pointer);
        }
        meta[r * 12] = trace.len() as u32;
        if *flag != 0 {
            meta[r * 12 + 1] = *flag;
            for (e, count) in budget.iter().enumerate() {
                meta[r * 12 + 2 + e] = u32::from(*count);
            }
        }
    }
    let tokens_t = IdTensor::from_slice(&tokens, vec![rows, t, 4], &device).unwrap();
    let meta_t = IdTensor::from_slice(&meta, vec![rows, 12], &device).unwrap();
    let out = ReplayBuffers::poisoned(rows, t, a, &device).unwrap();
    ms2::grammar_replay(&tokens_t, &meta_t, &constants, a as u32, rmax, &out).unwrap();
    check_launches(&device).unwrap();
    let replay = out.replay.try_to_vec().unwrap();
    let atoms = out.atoms.try_to_vec().unwrap();
    for (r, (((trace, budget), flag), name)) in traces
        .iter()
        .zip(budgets.iter())
        .zip(flags.iter())
        .zip(names.iter())
        .enumerate()
    {
        let host = host_replay(trace, limits, *budget, *flag, a);
        let length = trace.len();
        let illegal = host.first_illegal;
        if name.contains("-short-") {
            // The early STOP is legal under the upper bound. Under exact
            // completion the shortened trace fails no later than the STOP
            // itself (an incomplete STOP is already illegal under v1, and the
            // lookahead may doom an earlier ADD first).
            if *flag == 1 {
                assert_eq!(illegal, u32::MAX, "{name}: subgraph STOP stays legal");
            } else {
                assert_ne!(
                    illegal,
                    u32::MAX,
                    "{name}: the shortened trace fails under exact completion"
                );
                assert!(
                    illegal <= (length - 1) as u32,
                    "{name}: the failure is at or before the early STOP"
                );
            }
        }
        if name.contains("closed-break-") {
            // The offending ADD at index 3 is legal under flags 0/1. Under
            // exact completion the trace dies no later than that ADD: here
            // the ADD at index 2 is already hydrogen-doomed (the last carbon
            // can bring at most 3 of the 5 hydrogens still missing), so the
            // lookahead forbids it at its own index.
            if *flag == 2 {
                assert_eq!(illegal, 2, "{name}: breaker fails at the doomed ADD");
            } else {
                assert_eq!(illegal, u32::MAX, "{name}: breaker stays legal");
            }
        }
        for s in 0..t {
            let base = (r * t + s) * (4 + a);
            if s >= length || (illegal != u32::MAX && s > illegal as usize) {
                for w in 0..4 + a {
                    assert_eq!(
                        replay[base + w],
                        0,
                        "row {r} ({name}) step {s}: zero word {w}"
                    );
                }
                continue;
            }
            assert_eq!(
                replay[base], host.kinds[s],
                "row {r} ({name}) step {s}: kinds"
            );
            assert_eq!(
                replay[base + 1],
                host.types[s],
                "row {r} ({name}) step {s}: types"
            );
            assert_eq!(
                replay[base + 2],
                host.bonds[s],
                "row {r} ({name}) step {s}: bonds"
            );
            assert_eq!(
                replay[base + 3],
                host.pointers[s],
                "row {r} ({name}) step {s}: pointers"
            );
            for j in 0..a {
                assert_eq!(
                    replay[base + 4 + j],
                    host.resids[s][j],
                    "row {r} ({name}) step {s}: residual {j}"
                );
            }
        }
        for j in 0..a {
            assert_eq!(
                atoms[r * (a + 1) + j],
                host.add_steps[j],
                "row {r} ({name}): atom {j} step"
            );
        }
        assert_eq!(
            atoms[r * (a + 1) + a],
            illegal,
            "row {r} ({name}): first illegal"
        );
    }
}

/// Compare one device replay row against its host replay element for
/// element (masks, residuals, atom steps, first illegal).
fn check_device_row(
    device: &Device<R>,
    constants: &Ms2Constants<R>,
    trace: &[Token],
    limits: Limits,
    budget: Composition,
    flag: u32,
    a: usize,
    rmax: u32,
    label: &str,
) {
    let rows = 1usize;
    let t = trace.len();
    let mut tokens = vec![0u32; rows * t * 4];
    for (s, tok) in trace.iter().enumerate() {
        tokens[s * 4] = u32::from(tok.kind);
        tokens[s * 4 + 1] = u32::from(tok.atom_type);
        tokens[s * 4 + 2] = u32::from(tok.bond);
        tokens[s * 4 + 3] = u32::from(tok.pointer);
    }
    let mut meta = vec![0u32; rows * 12];
    meta[0] = t as u32;
    if flag != 0 {
        meta[1] = flag;
        for (e, count) in budget.iter().enumerate() {
            meta[2 + e] = u32::from(*count);
        }
    }
    let tokens_t = IdTensor::from_slice(&tokens, vec![rows, t, 4], device).unwrap();
    let meta_t = IdTensor::from_slice(&meta, vec![rows, 12], device).unwrap();
    let out = ReplayBuffers::poisoned(rows, t, a, device).unwrap();
    ms2::grammar_replay(&tokens_t, &meta_t, constants, a as u32, rmax, &out).unwrap();
    check_launches(device).unwrap();
    let replay = out.replay.try_to_vec().unwrap();
    let atoms = out.atoms.try_to_vec().unwrap();
    let host = host_replay(trace, limits, budget, flag, a);
    let illegal = host.first_illegal;
    assert_eq!(illegal, u32::MAX, "{label}: flag {flag} trace stays legal");
    for s in 0..t {
        let base = s * (4 + a);
        assert_eq!(replay[base], host.kinds[s], "{label} step {s}: kinds");
        assert_eq!(replay[base + 1], host.types[s], "{label} step {s}: types");
        assert_eq!(replay[base + 2], host.bonds[s], "{label} step {s}: bonds");
        assert_eq!(
            replay[base + 3],
            host.pointers[s],
            "{label} step {s}: pointers"
        );
        for j in 0..a {
            assert_eq!(
                replay[base + 4 + j],
                host.resids[s][j],
                "{label} step {s}: residual {j}"
            );
        }
    }
    for j in 0..a {
        assert_eq!(atoms[j], host.add_steps[j], "{label}: atom {j} step");
    }
    assert_eq!(atoms[a], illegal, "{label}: first illegal");
}

#[test]
fn boundary_32_atoms_6_closures() {
    let limits = Limits::new(32, 6).unwrap();
    // T = 2 + 32 + 6 = 40.
    assert_eq!(limits.max_steps(), 40);
    let (a, rmax) = (32usize, 6u32);
    let device = dev();
    let constants = Ms2Constants::new(&device);
    // (a) The 32-carbon single-bond chain: CH3, 30 x CH2, CH3.
    let mut chain_types = vec![0u8; 32];
    chain_types[0] = 4;
    chain_types[31] = 4;
    for i in 1..31 {
        chain_types[i] = 3;
    }
    let mut chain_bonds = Vec::with_capacity(31);
    for i in 0..31 {
        chain_bonds.push((i, i + 1, 1));
    }
    let chain = MolGraph::new(chain_types, chain_bonds).unwrap();
    let chain_budget = chain.composition();
    assert_eq!(&chain_budget[..4], &[32, 66, 0, 0]);
    let chain_canon = canonical_trace(&chain, limits, 5_000_000).unwrap();
    println!("chain-32: {} expansions", chain_canon.expansions);
    // START + 32 ADD + STOP.
    assert_eq!(chain_canon.trace.len(), 34);
    assert_eq!(chain_canon.trace.last().unwrap().kind, STOP);
    let chain_end = replay_exact(&chain_canon.trace, limits, chain_budget).unwrap();
    assert!(chain_end.is_complete(), "the chain completes exactly");
    assert!(chain_end.stopped(), "the chain ends in STOP");
    // It completes at the 32nd atom, before the STOP.
    let chain_prefix = &chain_canon.trace[..chain_canon.trace.len() - 1];
    let chain_pre = replay_exact(chain_prefix, limits, chain_budget).unwrap();
    assert_eq!(chain_pre.atoms(), 32);
    assert!(chain_pre.is_complete(), "complete at the 32nd atom");
    // A 33rd ADD is illegal by the atom cap: with no budget the cap alone
    // forbids it (add_pointers returns 0 at max_atoms).
    let cap_pre = replay(chain_prefix, limits, None).unwrap();
    assert_eq!(cap_pre.atoms(), 32);
    let probe = Token {
        kind: ADD_ATOM,
        atom_type: 4,
        bond: 1,
        pointer: 31,
    };
    assert!(
        !chain_pre.is_legal(probe),
        "33rd ADD illegal under the budget"
    );
    assert!(!cap_pre.is_legal(probe), "33rd ADD illegal by the atom cap");
    assert_eq!(
        chain_pre.masks(blank()).kinds & (1u32 << ADD_ATOM),
        0,
        "no ADD kind at the cap"
    );
    check_device_row(
        &device,
        &constants,
        &chain_canon.trace,
        limits,
        chain_budget,
        2,
        a,
        rmax,
        "chain-32",
    );
    // Pointer-bit coverage: bit 31 can never be a legal ADD pointer under the
    // 32-atom cap (`add_pointers` returns 0 once `n == max_atoms`, so a mask
    // containing bit 31 would need 33 atoms). The canonical chain starts from
    // a middle C(H2) root (id 3 < id 4, so the lexicographically smallest BFS
    // root is interior), hence the queue fans out in both directions and the
    // highest exercised pointer stays well below 30. What is exercised is the
    // full 32-atom width itself: atom index 31 is added (`add_steps[31]`), and
    // the scan below prints the highest pointer bit the canonical order
    // needed.
    let host32 = host_replay(&chain_canon.trace, limits, chain_budget, 2, a);
    let mut top_bit: i32 = -1;
    for (s, mask) in host32.pointers.iter().enumerate() {
        assert_eq!(mask & (1u32 << 31), 0, "step {s}: bit 31 never legal");
        for b in 0..32u32 {
            if mask & (1u32 << b) != 0 {
                top_bit = top_bit.max(b as i32);
            }
        }
    }
    println!("chain-32: highest pointer bit {top_bit}");
    assert!(top_bit >= 0, "some ADD pointer is exercised");
    assert_ne!(
        host32.add_steps[31],
        u32::MAX,
        "atom 31 is added (highest index exercised)"
    );
    // (b) A 32-atom all-carbon molecule with 6 ring closures: the 32-chain
    // plus 6 single-bond chords, each chord its own separate ring (six
    // separate rings on a chain, no shared edges). Degrees stay <= 3, so
    // hydrogens close every residual (degree 1 -> H3 id 4, degree 2 -> H2 id
    // 3, degree 3 -> H1 id 2): START + 32 ADD + 6 CLOSE + STOP = 40 tokens.
    let mut bonds: Vec<(usize, usize, u8)> = (0..31).map(|i| (i, i + 1, 1)).collect();
    for (x, y) in [
        (0usize, 4usize),
        (5, 9),
        (10, 14),
        (15, 19),
        (20, 24),
        (25, 29),
    ] {
        bonds.push((x, y, 1));
    }
    let mut degree = vec![0u8; 32];
    for (x, y, _) in &bonds {
        degree[*x] += 1;
        degree[*y] += 1;
    }
    let types: Vec<u8> = degree
        .iter()
        .map(|d| match d {
            1 => 4,
            2 => 3,
            3 => 2,
            4 => 1,
            _ => panic!("degree {d} exceeds carbon valence"),
        })
        .collect();
    let separate = MolGraph::new(types, bonds).unwrap();
    assert!(
        separate.is_connected(),
        "the separate-rings graph is connected"
    );
    assert_eq!(separate.ring_closures(), 6);
    assert!(separate.residual_valence().iter().all(|&r| r == 0));
    let separate_budget = separate.composition();
    let separate_canon = canonical_trace(&separate, limits, 5_000_000).unwrap();
    println!("separate-32-6: {} expansions", separate_canon.expansions);
    assert!(separate_canon.trace.len() <= 40, "fits in T = 40");
    assert_eq!(separate_canon.trace.last().unwrap().kind, STOP);
    assert!(
        (separate_canon.trace.len() as u32 - 1) <= 39,
        "STOP at step <= 39"
    );
    let separate_end = replay_exact(&separate_canon.trace, limits, separate_budget).unwrap();
    assert!(
        separate_end.is_complete(),
        "the separate-rings molecule completes"
    );
    for flag in [1u32, 2] {
        check_device_row(
            &device,
            &constants,
            &separate_canon.trace,
            limits,
            separate_budget,
            flag,
            a,
            rmax,
            &format!("separate-32-6 flag {flag}"),
        );
    }
    // (c) The 32-chain replayed in non-canonical linear order root -> end:
    // atom `i > 0` is added on pointer `i - 1`, so the last ADD uses pointer
    // 30 (bit 30, the highest legal ADD pointer: bit 31 would need 33 atoms).
    // A legal BFS trace even though it is not the canonical one (the
    // canonical root is interior, so its order fans out both ways).
    let mut linear = vec![
        start_token(),
        Token {
            kind: ADD_ATOM,
            atom_type: 4,
            ..blank()
        },
    ];
    for i in 1..32 {
        linear.push(Token {
            kind: ADD_ATOM,
            atom_type: if i == 31 { 4 } else { 3 },
            bond: 1,
            pointer: (i - 1) as u8,
        });
    }
    linear.push(stop_token());
    // START + 32 ADD + STOP; the last ADD is token 32.
    assert_eq!(linear.len(), 34);
    assert_eq!(linear[32].pointer, 30);
    let linear_host = host_replay(&linear, limits, chain_budget, 2, a);
    assert_eq!(
        linear_host.first_illegal,
        u32::MAX,
        "the linear order stays legal under exact completion"
    );
    assert_ne!(
        linear_host.pointers[32] & (1u32 << 30),
        0,
        "bit 30 is offered at the last ADD"
    );
    let linear_end = replay_exact(&linear, limits, chain_budget).unwrap();
    assert!(linear_end.is_complete(), "the linear chain completes");
    check_device_row(
        &device,
        &constants,
        &linear,
        limits,
        chain_budget,
        2,
        a,
        rmax,
        "chain-32-linear",
    );
    // (d) Three linearly fused six-membered rings with shared edges (14
    // carbons): ring 0-1-2-3-4-5-0, ring 2-3-6-7-8-9-2 sharing edge 2-3,
    // ring 7-8-10-11-12-13-7 sharing edge 7-8. Degrees 2 (H2 id 3) and 3
    // (H1 id 2) close every valence.
    let fused_bonds = vec![
        (0, 1, 1),
        (1, 2, 1),
        (2, 3, 1),
        (3, 4, 1),
        (4, 5, 1),
        (5, 0, 1),
        (3, 6, 1),
        (6, 7, 1),
        (7, 8, 1),
        (8, 9, 1),
        (9, 2, 1),
        (7, 13, 1),
        (13, 12, 1),
        (12, 11, 1),
        (11, 10, 1),
        (10, 8, 1),
    ];
    let mut fused_degree = vec![0u8; 14];
    for (x, y, _) in &fused_bonds {
        fused_degree[*x] += 1;
        fused_degree[*y] += 1;
    }
    let fused_types: Vec<u8> = fused_degree
        .iter()
        .map(|d| match d {
            1 => 4,
            2 => 3,
            3 => 2,
            4 => 1,
            _ => panic!("degree {d} exceeds carbon valence"),
        })
        .collect();
    let fused = MolGraph::new(fused_types, fused_bonds).unwrap();
    assert!(fused.is_connected(), "the fused rings are connected");
    assert_eq!(fused.ring_closures(), 3);
    assert!(fused.residual_valence().iter().all(|&r| r == 0));
    let fused_budget = fused.composition();
    let fused_canon = canonical_trace(&fused, limits, 5_000_000).unwrap();
    println!("fused-3x6: {} expansions", fused_canon.expansions);
    assert_eq!(fused_canon.trace.last().unwrap().kind, STOP);
    let fused_end = replay_exact(&fused_canon.trace, limits, fused_budget).unwrap();
    assert!(fused_end.is_complete(), "the fused rings complete");
    check_device_row(
        &device,
        &constants,
        &fused_canon.trace,
        limits,
        fused_budget,
        2,
        a,
        rmax,
        "fused-3x6",
    );
    // (e) A spiro centre joining two six-membered rings (11 carbons):
    // centre 0 with degree 4 (H0 id 1), the rest degree 2 (H2 id 3).
    let spiro_bonds = vec![
        (0, 1, 1),
        (1, 2, 1),
        (2, 3, 1),
        (3, 4, 1),
        (4, 5, 1),
        (5, 0, 1),
        (0, 6, 1),
        (6, 7, 1),
        (7, 8, 1),
        (8, 9, 1),
        (9, 10, 1),
        (10, 0, 1),
    ];
    let mut spiro_degree = vec![0u8; 11];
    for (x, y, _) in &spiro_bonds {
        spiro_degree[*x] += 1;
        spiro_degree[*y] += 1;
    }
    let spiro_types: Vec<u8> = spiro_degree
        .iter()
        .map(|d| match d {
            1 => 4,
            2 => 3,
            3 => 2,
            4 => 1,
            _ => panic!("degree {d} exceeds carbon valence"),
        })
        .collect();
    assert_eq!(spiro_types[0], 1, "the spiro centre carries no hydrogens");
    let spiro = MolGraph::new(spiro_types, spiro_bonds).unwrap();
    assert!(spiro.is_connected(), "the spiro rings are connected");
    assert_eq!(spiro.ring_closures(), 2);
    assert!(spiro.residual_valence().iter().all(|&r| r == 0));
    let spiro_budget = spiro.composition();
    let spiro_canon = canonical_trace(&spiro, limits, 5_000_000).unwrap();
    println!("spiro-2x6: {} expansions", spiro_canon.expansions);
    assert_eq!(spiro_canon.trace.last().unwrap().kind, STOP);
    let spiro_end = replay_exact(&spiro_canon.trace, limits, spiro_budget).unwrap();
    assert!(spiro_end.is_complete(), "the spiro rings complete");
    check_device_row(
        &device,
        &constants,
        &spiro_canon.trace,
        limits,
        spiro_budget,
        2,
        a,
        rmax,
        "spiro-2x6",
    );
    // (f) A K4 tetrahedron (4 carbons, each degree 3, H1 id 2): the last
    // atom closes two rings (two CLOSE_RING tokens after the final ADD_ATOM).
    let k4_bonds = vec![
        (0, 1, 1),
        (0, 2, 1),
        (0, 3, 1),
        (1, 2, 1),
        (1, 3, 1),
        (2, 3, 1),
    ];
    let k4 = MolGraph::new(vec![2, 2, 2, 2], k4_bonds).unwrap();
    assert!(k4.is_connected(), "K4 is connected");
    assert_eq!(k4.ring_closures(), 3);
    assert!(k4.residual_valence().iter().all(|&r| r == 0));
    let k4_budget = k4.composition();
    let k4_canon = canonical_trace(&k4, limits, 5_000_000).unwrap();
    println!("k4-double-close: {} expansions", k4_canon.expansions);
    assert_eq!(k4_canon.trace.last().unwrap().kind, STOP);
    let last_add = k4_canon
        .trace
        .iter()
        .rposition(|t| t.kind == ADD_ATOM)
        .expect("K4 has an ADD_ATOM");
    assert_eq!(
        k4_canon.trace[last_add + 1].kind,
        CLOSE_RING,
        "the last atom closes a first ring"
    );
    assert_eq!(
        k4_canon.trace[last_add + 2].kind,
        CLOSE_RING,
        "the last atom closes a second ring"
    );
    let k4_end = replay_exact(&k4_canon.trace, limits, k4_budget).unwrap();
    assert!(k4_end.is_complete(), "K4 completes");
    check_device_row(
        &device,
        &constants,
        &k4_canon.trace,
        limits,
        k4_budget,
        2,
        a,
        rmax,
        "k4-double-close",
    );
    // (g) A triple bond next to a ring: cyclopropyl-C.identC-H (ring atoms
    // 0-1-2, 0 substituted; 0-3 single, 3-4 triple). Types close every
    // valence by construction.
    let ethynyl = MolGraph::new(
        vec![2, 3, 3, 1, 2],
        vec![(0, 1, 1), (1, 2, 1), (2, 0, 1), (0, 3, 1), (3, 4, 3)],
    )
    .unwrap();
    assert!(ethynyl.is_connected(), "the ethynyl ring is connected");
    assert_eq!(ethynyl.ring_closures(), 1);
    assert!(
        ethynyl.bonds().iter().any(|(_, _, o)| *o == 3),
        "a triple bond is present"
    );
    assert!(ethynyl.residual_valence().iter().all(|&r| r == 0));
    let ethynyl_budget = ethynyl.composition();
    let ethynyl_canon = canonical_trace(&ethynyl, limits, 5_000_000).unwrap();
    println!("ethynyl-ring: {} expansions", ethynyl_canon.expansions);
    assert_eq!(ethynyl_canon.trace.last().unwrap().kind, STOP);
    let ethynyl_end = replay_exact(&ethynyl_canon.trace, limits, ethynyl_budget).unwrap();
    assert!(ethynyl_end.is_complete(), "the ethynyl ring completes");
    check_device_row(
        &device,
        &constants,
        &ethynyl_canon.trace,
        limits,
        ethynyl_budget,
        2,
        a,
        rmax,
        "ethynyl-ring",
    );
}

/// Drive `sample_step` for `T` steps from fresh exact trajectories and check
/// every step against the twin. Returns the final `(traj, state, actions)`.
#[allow(clippy::too_many_arguments)]
fn drive_exact_batch(
    device: &Device<R>,
    constants: &Ms2Constants<R>,
    budget: Composition,
    rows: usize,
    t: usize,
    a: usize,
    rmax: u32,
    seed_lo: u32,
    seed_hi: u32,
    rng_seed: u64,
    stop_boost: f32,
) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let s = ms2::replay_state_width(a);
    let record = ms2::sample_record_width(t, a);
    let width = ms2::sample_logits_width(a);
    let atable = host::atom_table_rows();
    let mut rng = Rng::seeded(rng_seed);
    let tables_host: Vec<f32> = rng.uniform_vec(19 * 4, -1.0, 1.0);
    // Fresh trajectories: the started word is 2 (exact completion).
    let mut traj = vec![0u32; rows * 14];
    for r in 0..rows {
        traj[r * 14] = r as u32;
        traj[r * 14 + 1] = 0;
        traj[r * 14 + 2] = r as u32;
        traj[r * 14 + 3] = 2;
        for e in 0..10 {
            traj[r * 14 + 4 + e] = u32::from(budget[e]);
        }
    }
    // START already applied: the step word is 1, the record holds START.
    let mut states = vec![0u32; rows * s];
    let mut actions_vec = vec![0u32; rows * record];
    for r in 0..rows {
        states[r * s + 3 * a + 4] = 1;
        actions_vec[r * record] = u32::from(START);
        actions_vec[r * record + t * 4 + a] = 1;
    }
    let mut twin_states = states.clone();
    let mut twin_actions = actions_vec.clone();
    let tables_t = Tensor::<R, E>::from_f32(&tables_host, vec![19, 4], device).unwrap();
    let traj_t = IdTensor::from_slice(&traj, vec![rows, 14], device).unwrap();
    let mut state_t = IdTensor::from_slice(&states, vec![rows, s], device).unwrap();
    let mut actions_t = IdTensor::from_slice(&actions_vec, vec![rows, record], device).unwrap();
    for step in 1..t {
        // Seeded random logits uploaded per step; `stop_boost` is added to
        // the STOP kind logit (0.0 removes the boost entirely).
        let mut logits_host: Vec<f32> = rng.uniform_vec(rows * width, -2.0, 2.0);
        for r in 0..rows {
            logits_host[r * width + usize::from(STOP)] += stop_boost;
        }
        for r in 0..rows {
            host::sample_step(
                &logits_host[r * width..(r + 1) * width],
                &tables_host,
                &traj[r * 14..(r + 1) * 14],
                &mut twin_states[r * s..(r + 1) * s],
                &mut twin_actions[r * record..(r + 1) * record],
                step as u32,
                seed_lo,
                seed_hi,
                1.0,
                t,
                a,
                rmax as usize,
                &atable,
            );
        }
        let logits_t = Tensor::<R, E>::from_f32(&logits_host, vec![rows, width], device).unwrap();
        ms2::sample_step(
            &logits_t,
            &tables_t,
            &traj_t,
            &mut state_t,
            &mut actions_t,
            step as u32,
            seed_lo,
            seed_hi,
            1.0,
            t,
            a,
            rmax,
            &constants.atom_table,
        )
        .unwrap();
        check_launches(device).unwrap();
        let state_got = state_t.try_to_vec().unwrap();
        let actions_got = actions_t.try_to_vec().unwrap();
        assert_eq!(
            state_got, twin_states,
            "step {step}: grammar rows match exactly"
        );
        for r in 0..rows {
            let abase = r * record;
            for w in 0..record {
                if w == t * 4 + a + 2 {
                    continue;
                }
                assert_eq!(
                    actions_got[abase + w],
                    twin_actions[abase + w],
                    "step {step} row {r} word {w}: exact"
                );
            }
            let g = f32::from_bits(actions_got[abase + t * 4 + a + 2]);
            let w = f32::from_bits(twin_actions[abase + t * 4 + a + 2]);
            assert!(
                (g - w).abs() <= 1e-5,
                "step {step} row {r}: trace_log_prob {g} vs twin {w}"
            );
        }
        // Carry the device outputs into the next step's inputs, exactly as
        // the generation loop does (the twin rows already advanced).
        state_t = IdTensor::from_slice(&state_got, vec![rows, s], device).unwrap();
        actions_t = IdTensor::from_slice(&actions_got, vec![rows, record], device).unwrap();
        twin_states.clone_from_slice(&state_got);
        twin_actions.clone_from_slice(&actions_got);
    }
    let state_got = state_t.try_to_vec().unwrap();
    let actions_got = actions_t.try_to_vec().unwrap();
    (traj, state_got, actions_got)
}

#[test]
fn sampler_under_exact_rule_matches_twin_and_never_stops_incomplete() {
    let limits = exact_limits();
    let (a, rmax) = (6usize, 1u32);
    let t = limits.max_steps();
    let record = ms2::sample_record_width(t, a);
    let device = dev();
    let constants = Ms2Constants::new(&device);
    let atable = host::atom_table_rows();
    // Seeds 20261005/77 with the same seeds as the v1 measurement (5/59 and
    // 2/62 finished of 64 for C2H6O and C3H8O). No STOP-logit boost is
    // applied: under exact completion STOP never competes (a complete state
    // admits no ADD or CLOSE, an incomplete state forbids STOP), so the boost
    // is a no-op.
    for (budget, rng_seed, v1_finished) in [
        ([2, 6, 0, 1, 0, 0, 0, 0, 0, 0], 20261005u64, 5usize),
        ([3, 8, 0, 1, 0, 0, 0, 0, 0, 0], 77u64, 2usize),
    ] {
        let rows = 64usize;
        let (traj, _state, actions) = drive_exact_batch(
            &device,
            &constants,
            budget,
            rows,
            t,
            a,
            rmax,
            0x1234_5678,
            0x9ABC_DEF0,
            rng_seed,
            0.0,
        );
        let mut n_finished_complete = 0usize;
        let mut n_dead_end = 0usize;
        for r in 0..rows {
            let abase = r * record;
            let st = actions[abase + t * 4 + a + 1];
            let len = actions[abase + t * 4 + a] as usize;
            if st & candidate_status::FINISHED != 0 {
                // A finished row replays exactly to a complete molecule.
                let mut toks = Vec::with_capacity(len);
                for i in 0..len {
                    toks.push(Token {
                        kind: actions[abase + i * 4] as u8,
                        atom_type: actions[abase + i * 4 + 1] as u8,
                        bond: actions[abase + i * 4 + 2] as u8,
                        pointer: actions[abase + i * 4 + 3] as u8,
                    });
                }
                let end =
                    replay_exact(&toks, limits, budget).expect("a finished exact trace replays");
                assert!(end.is_complete(), "row {r}: finished means complete");
                n_finished_complete += 1;
            } else {
                assert_eq!(
                    st & candidate_status::NO_VALID_ACTION,
                    candidate_status::NO_VALID_ACTION,
                    "row {r}: every unfinished row carries no_valid_action (status {st})"
                );
                assert_eq!(
                    st & candidate_status::TRUNCATED,
                    0,
                    "row {r}: no row truncates"
                );
                n_dead_end += 1;
            }
            assert!(
                !(st & candidate_status::FINISHED != 0
                    && st & candidate_status::NO_VALID_ACTION != 0),
                "row {r}: finished and failed are exclusive (status {st})"
            );
        }
        println!(
            "budget {:?}: {n_finished_complete} finished-complete, {n_dead_end} dead ends (v1 finished {v1_finished} of 64)",
            &budget[..4]
        );
        assert!(
            n_finished_complete > 0,
            "at least one row finishes complete"
        );
        assert!(
            n_finished_complete > v1_finished,
            "the lookahead finishes strictly more rows than v1's {v1_finished}"
        );
        // Validation and its twin agree, and no finished row is invalid:
        // under flag 2 an incomplete STOP would already have failed the
        // replay above, so `invalid_final` only needs checking, not forcing.
        let mut device_actions =
            IdTensor::from_slice(&actions, vec![rows, record], &device).unwrap();
        let traj_t = IdTensor::from_slice(&traj, vec![rows, 14], &device).unwrap();
        let mut scratch = IdTensor::empty(vec![rows, ms2::replay_state_width(a)], &device);
        ms2::validate_trajectories(
            &mut device_actions,
            &traj_t,
            &mut scratch,
            &constants.atom_table,
            1,
            rows,
            t,
            a,
            rmax,
        )
        .unwrap();
        check_launches(&device).unwrap();
        let validated = device_actions.try_to_vec().unwrap();
        let mut twin_validated = actions.clone();
        host::validate(
            &mut twin_validated,
            &traj,
            1,
            rows,
            t,
            a,
            rmax as usize,
            &atable,
        );
        assert_eq!(
            validated, twin_validated,
            "validation matches its twin exactly"
        );
        for r in 0..rows {
            let st = validated[r * record + t * 4 + a + 1];
            if st & candidate_status::FINISHED != 0 {
                assert_eq!(
                    st & candidate_status::INVALID_FINAL,
                    0,
                    "row {r}: no finished row is invalid_final"
                );
            }
        }
    }
}

#[test]
fn exact_flag_leaves_upper_bound_rows_untouched() {
    // Batch independence: flag-1 rows inside a mixed batch equal the same
    // rows sampled alone, and both equal the twin (which keeps flags 0/1 on
    // the subgraph path bit for bit).
    let limits = exact_limits();
    let (a, rmax) = (6usize, 1u32);
    let t = limits.max_steps();
    let s = ms2::replay_state_width(a);
    let record = ms2::sample_record_width(t, a);
    let width = ms2::sample_logits_width(a);
    let device = dev();
    let constants = Ms2Constants::new(&device);
    let atable = host::atom_table_rows();
    let budget: Composition = [2, 6, 0, 1, 0, 0, 0, 0, 0, 0];
    let n = 8usize;
    let m = 8usize;
    let mut rng = Rng::seeded(424242u64);
    let tables_host: Vec<f32> = rng.uniform_vec(19 * 4, -1.0, 1.0);
    let tables_t = Tensor::<R, E>::from_f32(&tables_host, vec![19, 4], &device).unwrap();
    // The mixed batch holds n flag-1 rows then m flag-2 rows; the lone batch
    // holds the same n flag-1 rows. Trajectory words agree on rows 0..n.
    let mut traj_mixed = vec![0u32; (n + m) * 14];
    for r in 0..n + m {
        traj_mixed[r * 14] = r as u32;
        traj_mixed[r * 14 + 1] = 0;
        traj_mixed[r * 14 + 2] = r as u32;
        traj_mixed[r * 14 + 3] = if r < n { 1 } else { 2 };
        for e in 0..10 {
            traj_mixed[r * 14 + 4 + e] = u32::from(budget[e]);
        }
    }
    let traj_lone: Vec<u32> = traj_mixed[..n * 14].to_vec();
    let init = |rows: usize| -> (Vec<u32>, Vec<u32>) {
        let mut states = vec![0u32; rows * s];
        let mut acts = vec![0u32; rows * record];
        for r in 0..rows {
            states[r * s + 3 * a + 4] = 1;
            acts[r * record] = u32::from(START);
            acts[r * record + t * 4 + a] = 1;
        }
        (states, acts)
    };
    let (states_mixed, acts_mixed) = init(n + m);
    let (states_lone, acts_lone) = init(n);
    let mut twin_mixed_s = states_mixed.clone();
    let mut twin_mixed_a = acts_mixed.clone();
    let mut twin_lone_s = states_lone.clone();
    let mut twin_lone_a = acts_lone.clone();
    let traj_mixed_t = IdTensor::from_slice(&traj_mixed, vec![n + m, 14], &device).unwrap();
    let traj_lone_t = IdTensor::from_slice(&traj_lone, vec![n, 14], &device).unwrap();
    let mut state_mixed_t = IdTensor::from_slice(&states_mixed, vec![n + m, s], &device).unwrap();
    let mut acts_mixed_t = IdTensor::from_slice(&acts_mixed, vec![n + m, record], &device).unwrap();
    let mut state_lone_t = IdTensor::from_slice(&states_lone, vec![n, s], &device).unwrap();
    let mut acts_lone_t = IdTensor::from_slice(&acts_lone, vec![n, record], &device).unwrap();
    for step in 1..t {
        // One shared logits row tiled over both batches, so rows 0..n see
        // identical inputs in the mixed and the lone batch.
        let row: Vec<f32> = rng.uniform_vec(width, -2.0, 2.0);
        let mut logits_mixed = vec![0.0f32; (n + m) * width];
        for r in 0..n + m {
            logits_mixed[r * width..(r + 1) * width].copy_from_slice(&row);
        }
        let mut logits_lone = vec![0.0f32; n * width];
        for r in 0..n {
            logits_lone[r * width..(r + 1) * width].copy_from_slice(&row);
        }
        for r in 0..n + m {
            host::sample_step(
                &logits_mixed[r * width..(r + 1) * width],
                &tables_host,
                &traj_mixed[r * 14..(r + 1) * 14],
                &mut twin_mixed_s[r * s..(r + 1) * s],
                &mut twin_mixed_a[r * record..(r + 1) * record],
                step as u32,
                11,
                22,
                1.0,
                t,
                a,
                rmax as usize,
                &atable,
            );
        }
        for r in 0..n {
            host::sample_step(
                &logits_lone[r * width..(r + 1) * width],
                &tables_host,
                &traj_lone[r * 14..(r + 1) * 14],
                &mut twin_lone_s[r * s..(r + 1) * s],
                &mut twin_lone_a[r * record..(r + 1) * record],
                step as u32,
                11,
                22,
                1.0,
                t,
                a,
                rmax as usize,
                &atable,
            );
        }
        let logits_mixed_t =
            Tensor::<R, E>::from_f32(&logits_mixed, vec![n + m, width], &device).unwrap();
        let logits_lone_t =
            Tensor::<R, E>::from_f32(&logits_lone, vec![n, width], &device).unwrap();
        ms2::sample_step(
            &logits_mixed_t,
            &tables_t,
            &traj_mixed_t,
            &mut state_mixed_t,
            &mut acts_mixed_t,
            step as u32,
            11,
            22,
            1.0,
            t,
            a,
            rmax,
            &constants.atom_table,
        )
        .unwrap();
        ms2::sample_step(
            &logits_lone_t,
            &tables_t,
            &traj_lone_t,
            &mut state_lone_t,
            &mut acts_lone_t,
            step as u32,
            11,
            22,
            1.0,
            t,
            a,
            rmax,
            &constants.atom_table,
        )
        .unwrap();
        check_launches(&device).unwrap();
        let mixed_s = state_mixed_t.try_to_vec().unwrap();
        let mixed_a = acts_mixed_t.try_to_vec().unwrap();
        let lone_s = state_lone_t.try_to_vec().unwrap();
        let lone_a = acts_lone_t.try_to_vec().unwrap();
        assert_eq!(
            mixed_s, twin_mixed_s,
            "step {step}: mixed states match the twin"
        );
        assert_eq!(
            lone_s, twin_lone_s,
            "step {step}: lone states match the twin"
        );
        for r in 0..n {
            // Batch independence: the flag-1 rows agree across batches.
            assert_eq!(
                mixed_s[r * s..(r + 1) * s],
                lone_s[r * s..(r + 1) * s],
                "step {step} row {r}: state independent of later flag-2 rows"
            );
            for w in 0..record {
                if w == t * 4 + a + 2 {
                    continue;
                }
                assert_eq!(
                    mixed_a[r * record + w],
                    lone_a[r * record + w],
                    "step {step} row {r} word {w}: actions independent of later rows"
                );
                assert_eq!(
                    mixed_a[r * record + w],
                    twin_lone_a[r * record + w],
                    "step {step} row {r} word {w}: device equals the twin"
                );
            }
        }
        state_mixed_t = IdTensor::from_slice(&mixed_s, vec![n + m, s], &device).unwrap();
        acts_mixed_t = IdTensor::from_slice(&mixed_a, vec![n + m, record], &device).unwrap();
        state_lone_t = IdTensor::from_slice(&lone_s, vec![n, s], &device).unwrap();
        acts_lone_t = IdTensor::from_slice(&lone_a, vec![n, record], &device).unwrap();
        twin_mixed_s.clone_from_slice(&mixed_s);
        twin_mixed_a.clone_from_slice(&mixed_a);
        twin_lone_s.clone_from_slice(&lone_s);
        twin_lone_a.clone_from_slice(&lone_a);
    }
    let _ = limits;
}

/// One 14-word trajectory row: spectrum ids from `id`, this started flag and
/// budget.
fn make_traj(id: u32, started: u32, budget: Composition) -> Vec<u32> {
    let mut row = vec![0u32; 14];
    row[0] = id;
    row[1] = 0;
    row[2] = id;
    row[3] = started;
    for e in 0..10 {
        row[4 + e] = u32::from(budget[e]);
    }
    row
}

/// One action record holding `trace` with this status word (length set,
/// zeroed open valence, zero log-probability, formula row 0).
fn make_record(trace: &[Token], t: usize, a: usize, status: u32) -> Vec<u32> {
    let record = ms2::sample_record_width(t, a);
    let mut rec = vec![0u32; record];
    for (s, tok) in trace.iter().enumerate() {
        rec[s * 4] = u32::from(tok.kind);
        rec[s * 4 + 1] = u32::from(tok.atom_type);
        rec[s * 4 + 2] = u32::from(tok.bond);
        rec[s * 4 + 3] = u32::from(tok.pointer);
    }
    rec[t * 4 + a] = trace.len() as u32;
    rec[t * 4 + a + 1] = status;
    rec
}

/// Status word of row `r` in a flat action buffer.
fn record_status(actions: &[u32], r: usize, t: usize, a: usize) -> u32 {
    actions[r * ms2::sample_record_width(t, a) + t * 4 + a + 1]
}

/// Run `validate_trajectories` on the device and `twin::validate` on the host
/// over the same inputs; returns both flat action buffers.
fn validate_both(
    device: &Device<R>,
    constants: &Ms2Constants<R>,
    actions: &[u32],
    traj: &[u32],
    spectra: usize,
    per: usize,
    t: usize,
    a: usize,
    rmax: u32,
) -> (Vec<u32>, Vec<u32>) {
    let rows = spectra * per;
    let record = ms2::sample_record_width(t, a);
    let mut device_actions = IdTensor::from_slice(actions, vec![rows, record], device).unwrap();
    let traj_t = IdTensor::from_slice(traj, vec![rows, 14], device).unwrap();
    let mut scratch = IdTensor::empty(vec![rows, ms2::replay_state_width(a)], device);
    ms2::validate_trajectories(
        &mut device_actions,
        &traj_t,
        &mut scratch,
        &constants.atom_table,
        spectra,
        per,
        t,
        a,
        rmax,
    )
    .unwrap();
    check_launches(device).unwrap();
    let got = device_actions.try_to_vec().unwrap();
    let mut twin = actions.to_vec();
    host::validate(
        &mut twin,
        traj,
        spectra,
        per,
        t,
        a,
        rmax as usize,
        &host::atom_table_rows(),
    );
    (got, twin)
}

#[test]
fn validator_rejects_forged_exact_rows() {
    let limits = exact_limits();
    let (a, rmax) = (6usize, 1u32);
    let t = limits.max_steps();
    let record = ms2::sample_record_width(t, a);
    let device = dev();
    let constants = Ms2Constants::new(&device);
    let budget: Composition = [2, 6, 0, 1, 0, 0, 0, 0, 0, 0];
    let graph = MolGraph::new(vec![4, 3, 9], vec![(0, 1, 1), (1, 2, 1)]).unwrap();
    assert_eq!(graph.composition(), budget, "ethanol is C2H6O");
    let full = canonical_trace(&graph, limits, 100_000).unwrap().trace;
    replay_exact(&full, limits, budget).expect("the genuine ethanol trace completes");
    // (a) An incomplete STOP: START, one C(H3, id 4), STOP.
    let forged_stop = vec![
        start_token(),
        Token {
            kind: ADD_ATOM,
            atom_type: 4,
            ..blank()
        },
        stop_token(),
    ];
    // (b) A closed-prefix breaker inside the budget: root C(H2, id 3), then a
    // child C(H1, id 2) on pointer 0 with bond 1 (residuals [1, 2], atom 0
    // open), then an O(H0, id 8) on pointer 1, which would abandon atom 0.
    // The second ADD is already hydrogen-doomed under the lookahead (the last
    // carbon can bring at most 3 of the 5 hydrogens still missing), so the
    // exact replay fails at step 2, before the breaking ADD. The trailing
    // STOP is otherwise legal.
    let forged_breaker = vec![
        start_token(),
        Token {
            kind: ADD_ATOM,
            atom_type: 3,
            ..blank()
        },
        Token {
            kind: ADD_ATOM,
            atom_type: 2,
            bond: 1,
            pointer: 0,
        },
        Token {
            kind: ADD_ATOM,
            atom_type: 8,
            bond: 1,
            pointer: 1,
        },
        stop_token(),
    ];
    // (c) The complete history with its final STOP removed (length shortened).
    let mut forged_nostop = full.clone();
    assert_eq!(
        forged_nostop.pop().unwrap().kind,
        STOP,
        "row (c) drops the final STOP"
    );
    // Host sanity: (a) and (b) fail the exact replay and stay legal under the
    // upper bound; (c) is a legal exact prefix.
    assert!(
        replay_exact(&forged_stop, limits, budget).is_err(),
        "row (a): the incomplete STOP fails exactly"
    );
    let breaker_err = replay_exact(&forged_breaker, limits, budget)
        .err()
        .expect("row (b): the doomed ADD fails exactly");
    assert!(
        breaker_err.to_string().contains("step 2"),
        "row (b): the doomed ADD fails at its own index: {breaker_err}"
    );
    replay(&forged_stop, limits, Some(budget)).expect("row (a) stays legal above exact");
    replay(&forged_breaker, limits, Some(budget)).expect("row (b) stays legal above exact");
    replay_exact(&forged_nostop, limits, budget).expect("row (c) is a legal exact prefix");
    // All four rows marked FINISHED under the exact flag: the three forged
    // rows carry INVALID_FINAL on device and twin alike, the genuine one does
    // not. The traces are pairwise distinct, so no DUPLICATE_TRACE fires.
    let traces = [&forged_stop, &forged_breaker, &forged_nostop, &full];
    let rows = traces.len();
    let mut actions = vec![0u32; rows * record];
    let mut traj = vec![0u32; rows * 14];
    for (r, trace) in traces.iter().enumerate() {
        actions[r * record..(r + 1) * record].copy_from_slice(&make_record(
            trace,
            t,
            a,
            candidate_status::FINISHED,
        ));
        traj[r * 14..(r + 1) * 14].copy_from_slice(&make_traj(r as u32, 2, budget));
    }
    let (got, twin) = validate_both(&device, &constants, &actions, &traj, 1, rows, t, a, rmax);
    assert_eq!(got, twin, "validation matches its twin exactly");
    for r in 0..3 {
        assert_ne!(
            record_status(&got, r, t, a) & candidate_status::INVALID_FINAL,
            0,
            "forged row {r} carries invalid_final"
        );
    }
    assert_eq!(
        record_status(&got, 3, t, a) & candidate_status::INVALID_FINAL,
        0,
        "the genuine trace stays valid"
    );
    // The same records with started word 1: row (a) is not invalid (a
    // subgraph STOP is legal), showing the flag is what decides. Row (b) is
    // valid too; row (c) stays invalid because a FINISHED record must end in
    // STOP under every flag.
    for r in 0..rows {
        traj[r * 14 + 3] = 1;
    }
    let (got1, twin1) = validate_both(&device, &constants, &actions, &traj, 1, rows, t, a, rmax);
    assert_eq!(got1, twin1, "flag-1 validation matches its twin exactly");
    assert_eq!(
        record_status(&got1, 0, t, a) & candidate_status::INVALID_FINAL,
        0,
        "row (a): the subgraph STOP is legal, the flag decides"
    );
    assert_eq!(
        record_status(&got1, 1, t, a) & candidate_status::INVALID_FINAL,
        0,
        "row (b): legal above exact"
    );
    assert_ne!(
        record_status(&got1, 2, t, a) & candidate_status::INVALID_FINAL,
        0,
        "row (c): finished without STOP stays invalid"
    );
    assert_eq!(
        record_status(&got1, 3, t, a) & candidate_status::INVALID_FINAL,
        0,
        "the genuine trace stays valid above exact too"
    );
}

#[test]
fn truncated_exact_rows_are_not_finished() {
    let (a, rmax) = (6usize, 1u32);
    // C3H8O needs START + 4 ADD + STOP = 6 tokens at minimum (4 heavy atoms),
    // so a horizon of 4 can never fit a complete trace.
    let budget: Composition = [3, 8, 0, 1, 0, 0, 0, 0, 0, 0];
    let t = 4usize;
    let device = dev();
    let constants = Ms2Constants::new(&device);
    let rows = 64usize;
    // `drive_exact_batch` already checks device equals twin after every step.
    let (traj, _state, actions) = drive_exact_batch(
        &device,
        &constants,
        budget,
        rows,
        t,
        a,
        rmax,
        0x1234_5678,
        0x9ABC_DEF0,
        20261005,
        0.0,
    );
    for r in 0..rows {
        let st = record_status(&actions, r, t, a);
        assert_eq!(
            st & candidate_status::FINISHED,
            0,
            "row {r}: nothing finishes in 4 steps (status {st})"
        );
        assert_ne!(
            st & (candidate_status::TRUNCATED | candidate_status::NO_VALID_ACTION),
            0,
            "row {r}: every live row ends truncated or failed (status {st})"
        );
    }
    let (got, twin) = validate_both(&device, &constants, &actions, &traj, 1, rows, t, a, rmax);
    assert_eq!(got, twin, "validation matches its twin exactly");
    for r in 0..rows {
        let st = record_status(&got, r, t, a);
        assert_eq!(
            st & candidate_status::FINISHED,
            0,
            "row {r}: validation finishes nothing (status {st})"
        );
        assert_ne!(
            st & (candidate_status::TRUNCATED | candidate_status::NO_VALID_ACTION),
            0,
            "row {r}: validation keeps the truncated/failed bits (status {st})"
        );
    }
}

#[test]
fn not_started_and_absorbing_rows_stay_untouched() {
    let limits = exact_limits();
    let (a, rmax) = (6usize, 1u32);
    let t = limits.max_steps();
    let s = ms2::replay_state_width(a);
    let record = ms2::sample_record_width(t, a);
    let width = ms2::sample_logits_width(a);
    let device = dev();
    let constants = Ms2Constants::new(&device);
    let atable = host::atom_table_rows();
    let budget: Composition = [2, 6, 0, 1, 0, 0, 0, 0, 0, 0];
    // The forged incomplete-STOP row: an illegal history under the exact flag
    // (fix 1's regression: the twin used to panic replaying it).
    let forged = vec![
        start_token(),
        Token {
            kind: ADD_ATOM,
            atom_type: 4,
            ..blank()
        },
        stop_token(),
    ];
    let fresh = vec![start_token()];
    // (started flag, trace, status): one not-started row, live rows under
    // flags 1 and 2, and FINISHED / NO_VALID_ACTION / INVALID_FINAL rows
    // under both flags.
    let spec: Vec<(u32, Vec<Token>, u32)> = vec![
        (0, vec![], 0),
        (1, fresh.clone(), 0),
        (2, fresh.clone(), 0),
        (1, forged.clone(), candidate_status::FINISHED),
        (2, forged.clone(), candidate_status::FINISHED),
        (1, fresh.clone(), candidate_status::NO_VALID_ACTION),
        (2, fresh.clone(), candidate_status::NO_VALID_ACTION),
        (1, forged.clone(), candidate_status::INVALID_FINAL),
        (2, forged.clone(), candidate_status::INVALID_FINAL),
    ];
    let rows = spec.len();
    let mut traj = vec![0u32; rows * 14];
    let mut states = vec![0u32; rows * s];
    let mut acts = vec![0u32; rows * record];
    for (r, (started, trace, status)) in spec.iter().enumerate() {
        traj[r * 14..(r + 1) * 14].copy_from_slice(&make_traj(r as u32, *started, budget));
        acts[r * record..(r + 1) * record].copy_from_slice(&make_record(trace, t, a, *status));
        if *started != 0 && *status == 0 {
            // Live rows carry the fresh sampler state (START applied).
            states[r * s + 3 * a + 4] = 1;
        }
    }
    let mut rng = Rng::seeded(777u64);
    let tables_host: Vec<f32> = rng.uniform_vec(19 * 4, -1.0, 1.0);
    let logits_host: Vec<f32> = rng.uniform_vec(rows * width, -2.0, 2.0);
    let tables_t = Tensor::<R, E>::from_f32(&tables_host, vec![19, 4], &device).unwrap();
    let logits_t = Tensor::<R, E>::from_f32(&logits_host, vec![rows, width], &device).unwrap();
    let traj_t = IdTensor::from_slice(&traj, vec![rows, 14], &device).unwrap();
    let mut state_t = IdTensor::from_slice(&states, vec![rows, s], &device).unwrap();
    let mut acts_t = IdTensor::from_slice(&acts, vec![rows, record], &device).unwrap();
    let step = 1u32;
    let (seed_lo, seed_hi) = (11u32, 22u32);
    ms2::sample_step(
        &logits_t,
        &tables_t,
        &traj_t,
        &mut state_t,
        &mut acts_t,
        step,
        seed_lo,
        seed_hi,
        1.0,
        t,
        a,
        rmax,
        &constants.atom_table,
    )
    .unwrap();
    check_launches(&device).unwrap();
    let state_got = state_t.try_to_vec().unwrap();
    let acts_got = acts_t.try_to_vec().unwrap();
    // The twin must not panic on any row (fix 1) and must agree with the
    // device everywhere.
    let mut twin_states = states.clone();
    let mut twin_acts = acts.clone();
    for r in 0..rows {
        let draw = host::sample_step(
            &logits_host[r * width..(r + 1) * width],
            &tables_host,
            &traj[r * 14..(r + 1) * 14],
            &mut twin_states[r * s..(r + 1) * s],
            &mut twin_acts[r * record..(r + 1) * record],
            step,
            seed_lo,
            seed_hi,
            1.0,
            t,
            a,
            rmax as usize,
            &atable,
        );
        let (started, _, status) = &spec[r];
        if *started == 0 || *status != 0 {
            // Absorbing (or not started): byte-identical before and after on
            // the device, unchanged on the twin, and a zero-mask draw.
            assert_eq!(
                state_got[r * s..(r + 1) * s],
                states[r * s..(r + 1) * s],
                "row {r}: absorbing device state untouched"
            );
            assert_eq!(
                acts_got[r * record..(r + 1) * record],
                acts[r * record..(r + 1) * record],
                "row {r}: absorbing device actions untouched"
            );
            assert_eq!(
                twin_states[r * s..(r + 1) * s],
                states[r * s..(r + 1) * s],
                "row {r}: absorbing twin state untouched"
            );
            assert_eq!(
                twin_acts[r * record..(r + 1) * record],
                acts[r * record..(r + 1) * record],
                "row {r}: absorbing twin actions untouched"
            );
            assert_eq!(draw.token, [0, 0, 0, 0], "row {r}: zero token");
            assert_eq!(draw.kinds, 0, "row {r}: zero kind mask");
            assert_eq!(draw.types, 0, "row {r}: zero type mask");
            assert_eq!(draw.bonds, 0, "row {r}: zero bond mask");
            assert_eq!(draw.pointers, 0, "row {r}: zero pointer mask");
            let h = host::hash_u32_host(seed_hi, seed_lo, 0);
            let key = host::hash_u32_host(traj[r * 14], h, traj[r * 14 + 1]);
            let base = host::hash_u32_host(traj[r * 14 + 2], key, 0);
            for f in 0..4 {
                let want =
                    ((host::hash_u32_host(step * 4 + f as u32, base, 0) >> 8) as f32) / 16777216.0;
                assert_eq!(
                    draw.u[f].to_bits(),
                    want.to_bits(),
                    "row {r} field {f}: the draw is still the hash word"
                );
            }
        } else {
            // Live rows: the device equals the twin (log-probability up to
            // the usual device rounding).
            assert_eq!(
                state_got[r * s..(r + 1) * s],
                twin_states[r * s..(r + 1) * s],
                "row {r}: live state matches the twin"
            );
            for w in 0..record {
                if w == t * 4 + a + 2 {
                    continue;
                }
                assert_eq!(
                    acts_got[r * record + w],
                    twin_acts[r * record + w],
                    "row {r} word {w}: live actions match the twin"
                );
            }
            let g = f32::from_bits(acts_got[r * record + t * 4 + a + 2]);
            let w = f32::from_bits(twin_acts[r * record + t * 4 + a + 2]);
            assert!(
                (g - w).abs() <= 1e-5,
                "row {r}: trace_log_prob {g} vs twin {w}"
            );
            assert_eq!(
                acts_got[r * record + t * 4 + a],
                2,
                "row {r}: the live row sampled its root type"
            );
        }
    }
    let _ = limits;
}

#[test]
fn replay_after_stop_is_illegal() {
    let limits = exact_limits();
    let (a, rmax) = (6usize, 1u32);
    let device = dev();
    let constants = Ms2Constants::new(&device);
    let graph = MolGraph::new(vec![4, 3, 9], vec![(0, 1, 1), (1, 2, 1)]).unwrap();
    let budget = graph.composition();
    let mut trace = canonical_trace(&graph, limits, 100_000).unwrap().trace;
    trace.push(stop_token());
    let illegal = (trace.len() - 1) as u32;
    let t = trace.len();
    for flag in [1u32, 2] {
        let host = host_replay(&trace, limits, budget, flag, a);
        assert_eq!(
            host.first_illegal, illegal,
            "flag {flag}: the appended index is the first illegal step"
        );
        assert_eq!(
            host.kinds[illegal as usize], 0,
            "flag {flag}: zero kind mask"
        );
        assert_eq!(
            host.types[illegal as usize], 0,
            "flag {flag}: zero type mask"
        );
        assert_eq!(
            host.bonds[illegal as usize], 0,
            "flag {flag}: zero bond mask"
        );
        assert_eq!(
            host.pointers[illegal as usize], 0,
            "flag {flag}: zero pointer mask"
        );
        let mut tokens = vec![0u32; t * 4];
        for (s2, tok) in trace.iter().enumerate() {
            tokens[s2 * 4] = u32::from(tok.kind);
            tokens[s2 * 4 + 1] = u32::from(tok.atom_type);
            tokens[s2 * 4 + 2] = u32::from(tok.bond);
            tokens[s2 * 4 + 3] = u32::from(tok.pointer);
        }
        let mut meta = vec![0u32; 12];
        meta[0] = t as u32;
        if flag != 0 {
            meta[1] = flag;
            for (e, count) in budget.iter().enumerate() {
                meta[2 + e] = u32::from(*count);
            }
        }
        let tokens_t = IdTensor::from_slice(&tokens, vec![1, t, 4], &device).unwrap();
        let meta_t = IdTensor::from_slice(&meta, vec![1, 12], &device).unwrap();
        let out = ReplayBuffers::poisoned(1, t, a, &device).unwrap();
        ms2::grammar_replay(&tokens_t, &meta_t, &constants, a as u32, rmax, &out).unwrap();
        check_launches(&device).unwrap();
        let replay = out.replay.try_to_vec().unwrap();
        let atoms = out.atoms.try_to_vec().unwrap();
        for s2 in 0..t {
            let base = s2 * (4 + a);
            assert_eq!(replay[base], host.kinds[s2], "flag {flag} step {s2}: kinds");
            assert_eq!(
                replay[base + 1],
                host.types[s2],
                "flag {flag} step {s2}: types"
            );
            assert_eq!(
                replay[base + 2],
                host.bonds[s2],
                "flag {flag} step {s2}: bonds"
            );
            assert_eq!(
                replay[base + 3],
                host.pointers[s2],
                "flag {flag} step {s2}: pointers"
            );
            for j in 0..a {
                assert_eq!(
                    replay[base + 4 + j],
                    host.resids[s2][j],
                    "flag {flag} step {s2}: residual {j}"
                );
            }
        }
        let base = illegal as usize * (4 + a);
        for w in 0..4 + a {
            assert_eq!(
                replay[base + w],
                0,
                "flag {flag}: all-zero masks at the appended index (word {w})"
            );
        }
        assert_eq!(
            atoms[a], illegal,
            "flag {flag}: the device reports the appended index"
        );
    }
}

#[test]
fn exact_stop_checks_every_count() {
    let limits = exact_limits();
    let (a, rmax) = (6usize, 1u32);
    let device = dev();
    let constants = Ms2Constants::new(&device);
    // (a) A hydrogen-free complete molecule: difluorine, F-F from two id-10
    // fluorines. A linear C#C-C#C from zero-hydrogen carbons cannot close
    // (each outer triple leaves residual 1), and a single H0 atom can never
    // complete (every H0 valence is >= 1), so F2 is the smallest H0 molecule
    // the table allows.
    let f2 = MolGraph::new(vec![10, 10], vec![(0, 1, 1)]).unwrap();
    let f2_budget = f2.composition();
    assert_eq!(&f2_budget[..5], &[0, 0, 0, 0, 2], "F2 uses no H");
    assert_eq!(f2_budget[1], 0, "F2 is hydrogen-free");
    let f2_trace = canonical_trace(&f2, limits, 100_000).unwrap().trace;
    assert_eq!(f2_trace.len(), 4, "START + 2 ADD + STOP");
    let f2_end = replay_exact(&f2_trace, limits, f2_budget).unwrap();
    assert!(f2_end.is_complete(), "F2 completes under its own budget");
    assert_eq!(f2_end.used(), &f2_budget, "F2 uses exactly its budget");
    check_device_row(
        &device, &constants, &f2_trace, limits, f2_budget, 2, a, rmax, "F2",
    );
    // (b) For a complete ethanol trace, +1 on each of the 10 element counts
    // in turn makes the final STOP illegal under exact completion (host and
    // device) while the upper bound still accepts it.
    let graph = MolGraph::new(vec![4, 3, 9], vec![(0, 1, 1), (1, 2, 1)]).unwrap();
    let budget = graph.composition();
    let trace = canonical_trace(&graph, limits, 100_000).unwrap().trace;
    let n = trace.len();
    let rows = 20usize;
    let mut tokens = vec![0u32; rows * n * 4];
    let mut meta = vec![0u32; rows * 12];
    let mut budgets = Vec::with_capacity(rows);
    let mut flags = Vec::with_capacity(rows);
    for e in 0..10 {
        let mut perturbed = budget;
        perturbed[e] += 1;
        budgets.push(perturbed);
        flags.push(2u32);
    }
    for e in 0..10 {
        let mut perturbed = budget;
        perturbed[e] += 1;
        budgets.push(perturbed);
        flags.push(1u32);
    }
    for (r, (perturbed, flag)) in budgets.iter().zip(flags.iter()).enumerate() {
        let e = r % 10;
        for (s2, tok) in trace.iter().enumerate() {
            tokens[(r * n + s2) * 4] = u32::from(tok.kind);
            tokens[(r * n + s2) * 4 + 1] = u32::from(tok.atom_type);
            tokens[(r * n + s2) * 4 + 2] = u32::from(tok.bond);
            tokens[(r * n + s2) * 4 + 3] = u32::from(tok.pointer);
        }
        meta[r * 12] = n as u32;
        meta[r * 12 + 1] = *flag;
        for (e, count) in perturbed.iter().enumerate() {
            meta[r * 12 + 2 + e] = u32::from(*count);
        }
        // Host side: nothing is illegal above exact completion; under exact
        // completion the perturbed trace fails no later than the STOP (an
        // incomplete STOP is already illegal under v1, and the lookahead may
        // doom an earlier token first).
        let host = host_replay(&trace, limits, *perturbed, *flag, a);
        if *flag == 2 {
            assert_ne!(
                host.first_illegal,
                u32::MAX,
                "element {e}: the perturbed trace fails under exact completion"
            );
            assert!(
                host.first_illegal <= (n - 1) as u32,
                "element {e}: the failure is at or before the STOP"
            );
            assert!(
                replay_exact(&trace, limits, *perturbed).is_err(),
                "element {e}: the perturbed trace fails exactly"
            );
        } else {
            assert_eq!(
                host.first_illegal,
                u32::MAX,
                "element {e}: the upper bound still accepts the perturbed STOP"
            );
        }
    }
    let tokens_t = IdTensor::from_slice(&tokens, vec![rows, n, 4], &device).unwrap();
    let meta_t = IdTensor::from_slice(&meta, vec![rows, 12], &device).unwrap();
    let out = ReplayBuffers::poisoned(rows, n, a, &device).unwrap();
    ms2::grammar_replay(&tokens_t, &meta_t, &constants, a as u32, rmax, &out).unwrap();
    check_launches(&device).unwrap();
    let replay = out.replay.try_to_vec().unwrap();
    let atoms = out.atoms.try_to_vec().unwrap();
    for (r, (perturbed, flag)) in budgets.iter().zip(flags.iter()).enumerate() {
        let host = host_replay(&trace, limits, *perturbed, *flag, a);
        let illegal = host.first_illegal;
        if *flag == 2 {
            assert_ne!(
                illegal,
                u32::MAX,
                "row {r}: the perturbed trace fails under exact completion"
            );
            assert!(
                illegal <= (n - 1) as u32,
                "row {r}: the failure is at or before the STOP"
            );
        }
        for s2 in 0..n {
            // The host stops at the first illegal step; the device writes
            // zeros there and past it.
            if illegal != u32::MAX && s2 > illegal as usize {
                for w in 0..4 + a {
                    assert_eq!(
                        replay[(r * n + s2) * (4 + a) + w],
                        0,
                        "row {r} step {s2}: zero word {w}"
                    );
                }
                continue;
            }
            let base = (r * n + s2) * (4 + a);
            assert_eq!(replay[base], host.kinds[s2], "row {r} step {s2}: kinds");
            assert_eq!(replay[base + 1], host.types[s2], "row {r} step {s2}: types");
            assert_eq!(replay[base + 2], host.bonds[s2], "row {r} step {s2}: bonds");
            assert_eq!(
                replay[base + 3],
                host.pointers[s2],
                "row {r} step {s2}: pointers"
            );
            for j in 0..a {
                assert_eq!(
                    replay[base + 4 + j],
                    host.resids[s2][j],
                    "row {r} step {s2}: residual {j}"
                );
            }
        }
        assert_eq!(atoms[r * (a + 1) + a], illegal, "row {r}: first illegal");
    }
}

#[test]
fn empty_state_valence_uses_parent_bonds() {
    // Fix 1: `valence` is not a necessary condition when no atom exists.
    // Budget F2 right after START has R = 0, m = 2, V = 2: the old bound
    // R + V >= 2 * m (2 >= 4) reported infeasible although START, ADD F,
    // ADD F completes. With zero atoms the root needs no parent bond, so
    // the bound is 2 * (m - 1).
    let limits = exact_limits();
    let f2 = MolGraph::new(vec![10, 10], vec![(0, 1, 1)]).unwrap();
    let f2_budget = f2.composition();
    let mut start = TraceState::new_exact(limits, f2_budget);
    start.apply(start_token()).unwrap();
    assert_eq!(start.atoms(), 0, "no atom after START");
    let f = start.feasibility();
    assert!(
        f.hydrogen && f.open_site && f.closable && f.valence,
        "the F2 start state passes all four conditions: {f:?}"
    );
    // `first_doomed_step` is None on the complete canonical trace of every
    // molecule in `molecules()` and of F2.
    for (name, graph) in molecules() {
        let budget = graph.composition();
        let trace = canonical_trace(&graph, limits, 100_000).unwrap().trace;
        assert_eq!(
            first_doomed_step(&trace, &budget, limits).unwrap(),
            None,
            "{name}: the complete canonical trace is never doomed"
        );
    }
    let f2_trace = canonical_trace(&f2, limits, 100_000).unwrap().trace;
    assert_eq!(
        first_doomed_step(&f2_trace, &f2_budget, limits).unwrap(),
        None,
        "F2: the complete canonical trace is never doomed"
    );
}

/// FNV-1a over the little-endian bytes of `words`, for pinning device
/// outputs as constants.
fn fnv1a64(words: &[u32]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for &w in words {
        for b in w.to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

#[test]
fn upper_bound_flags_are_untouched() {
    // Pinned pre-v2 outputs (MC7): the same batch replayed under flags 0 and
    // 1. The v2 feasibility lookahead must not move any flag-0/1 output bit:
    // the `replay` and `atoms` buffers below are compared against hashes
    // captured from the tree before the change. Any drift fails here.
    const REPLAY_FLAG0: u64 = 0x18d9ccf463ce1f74;
    const ATOMS_FLAG0: u64 = 0xbe3028f4203d47ba;
    const REPLAY_FLAG1: u64 = 0x65855e9b41845c4e;
    const ATOMS_FLAG1: u64 = 0xbe3028f4203d47ba;
    let limits = exact_limits();
    let (a, rmax) = (6usize, 1u32);
    let device = dev();
    let constants = Ms2Constants::new(&device);
    // Rows: every molecule's full canonical trace, its shortened trace (last
    // non-STOP token removed, early STOP) and the closed-prefix breaker.
    let mut traces: Vec<Vec<Token>> = Vec::new();
    let mut budgets: Vec<Composition> = Vec::new();
    for (_, graph) in molecules() {
        let budget = graph.composition();
        let full = canonical_trace(&graph, limits, 100_000).unwrap().trace;
        let mut short = full.clone();
        short.remove(short.iter().rposition(|t| t.kind != STOP).unwrap());
        traces.push(full);
        budgets.push(budget);
        traces.push(short);
        budgets.push(budget);
    }
    let breaker_budget: Composition = [3, 8, 0, 1, 0, 0, 0, 0, 0, 0];
    traces.push(vec![
        start_token(),
        Token {
            kind: ADD_ATOM,
            atom_type: 3,
            ..blank()
        },
        Token {
            kind: ADD_ATOM,
            atom_type: 2,
            bond: 1,
            pointer: 0,
        },
        Token {
            kind: ADD_ATOM,
            atom_type: 4,
            bond: 1,
            pointer: 1,
        },
        stop_token(),
    ]);
    budgets.push(breaker_budget);
    let rows = traces.len();
    let t = traces.iter().map(|tr| tr.len()).max().unwrap();
    for flag in [0u32, 1u32] {
        let mut tokens = vec![0u32; rows * t * 4];
        let mut meta = vec![0u32; rows * 12];
        for (r, (trace, budget)) in traces.iter().zip(budgets.iter()).enumerate() {
            for (s, tok) in trace.iter().enumerate() {
                tokens[(r * t + s) * 4] = u32::from(tok.kind);
                tokens[(r * t + s) * 4 + 1] = u32::from(tok.atom_type);
                tokens[(r * t + s) * 4 + 2] = u32::from(tok.bond);
                tokens[(r * t + s) * 4 + 3] = u32::from(tok.pointer);
            }
            meta[r * 12] = trace.len() as u32;
            if flag != 0 {
                meta[r * 12 + 1] = flag;
                for (e, count) in budget.iter().enumerate() {
                    meta[r * 12 + 2 + e] = u32::from(*count);
                }
            }
        }
        let tokens_t = IdTensor::from_slice(&tokens, vec![rows, t, 4], &device).unwrap();
        let meta_t = IdTensor::from_slice(&meta, vec![rows, 12], &device).unwrap();
        let out = ReplayBuffers::poisoned(rows, t, a, &device).unwrap();
        ms2::grammar_replay(&tokens_t, &meta_t, &constants, a as u32, rmax, &out).unwrap();
        check_launches(&device).unwrap();
        let replay = out.replay.try_to_vec().unwrap();
        let atoms = out.atoms.try_to_vec().unwrap();
        let (rh, ah) = (fnv1a64(&replay), fnv1a64(&atoms));
        println!("flag {flag}: replay hash {rh:016x}, atoms hash {ah:016x}");
        match flag {
            0 => {
                assert_eq!(rh, REPLAY_FLAG0, "flag 0 replay drifted");
                assert_eq!(ah, ATOMS_FLAG0, "flag 0 atoms drifted");
            }
            _ => {
                assert_eq!(rh, REPLAY_FLAG1, "flag 1 replay drifted");
                assert_eq!(ah, ATOMS_FLAG1, "flag 1 atoms drifted");
            }
        }
    }
    let _ = limits;
}

/// Every prefix the exact-mode search visits, with the non-root ADD offers
/// checked along the way: each offered atom type has a non-zero bond mask,
/// each offered bond a non-zero pointer mask, and each offered CLOSE_RING
/// bond a non-zero pointer mask. Returns the visited offers as (prefix,
/// type, bond, one offered pointer).
fn collect_offered_pairs(
    budget: Composition,
    limits: Limits,
    cap: usize,
) -> Vec<(Vec<Token>, u8, u8, u8)> {
    fn dfs(
        prefix: &mut Vec<Token>,
        limits: Limits,
        budget: Composition,
        out: &mut Vec<(Vec<Token>, u8, u8, u8)>,
        cap: usize,
    ) {
        let mut st = TraceState::new_exact(limits, budget);
        for tok in prefix.iter() {
            st.apply(*tok)
                .expect("the search only extends legal tokens");
        }
        // Non-root ADD offers only: the root takes kind and type alone, so
        // its bond and pointer masks are 0 by construction.
        if st.step() > 1 {
            let types = st
                .masks(Token {
                    kind: ADD_ATOM,
                    ..blank()
                })
                .atom_types;
            for id in 1..=17u8 {
                if types & (1u32 << id) == 0 {
                    continue;
                }
                let bonds = st
                    .masks(Token {
                        kind: ADD_ATOM,
                        atom_type: id,
                        ..blank()
                    })
                    .bonds;
                assert_ne!(bonds, 0, "every offered type has a bond");
                for b in 1..=3u8 {
                    if bonds & (1u32 << b) == 0 {
                        continue;
                    }
                    let pointers = st
                        .masks(Token {
                            kind: ADD_ATOM,
                            atom_type: id,
                            bond: b,
                            ..blank()
                        })
                        .pointers;
                    assert_ne!(pointers, 0, "every offered (type, bond) has a pointer");
                    if out.len() < cap {
                        out.push((prefix.clone(), id, b, pointers.trailing_zeros() as u8));
                    }
                }
            }
        }
        let close_bonds = st
            .masks(Token {
                kind: CLOSE_RING,
                ..blank()
            })
            .bonds;
        for b in 1..=3u8 {
            if close_bonds & (1u32 << b) == 0 {
                continue;
            }
            let pointers = st
                .masks(Token {
                    kind: CLOSE_RING,
                    bond: b,
                    ..blank()
                })
                .pointers;
            assert_ne!(pointers, 0, "every offered close bond has a pointer");
        }
        let kinds = st.masks(blank()).kinds;
        if kinds == 0 {
            return;
        }
        if kinds & (1u32 << ADD_ATOM) != 0 {
            if st.step() == 1 {
                let types = st
                    .masks(Token {
                        kind: ADD_ATOM,
                        ..blank()
                    })
                    .atom_types;
                for id in 1..=17u8 {
                    if types & (1u32 << id) == 0 {
                        continue;
                    }
                    prefix.push(Token {
                        kind: ADD_ATOM,
                        atom_type: id,
                        ..blank()
                    });
                    dfs(prefix, limits, budget, out, cap);
                    prefix.pop();
                }
            } else {
                let types = st
                    .masks(Token {
                        kind: ADD_ATOM,
                        ..blank()
                    })
                    .atom_types;
                for id in 1..=17u8 {
                    if types & (1u32 << id) == 0 {
                        continue;
                    }
                    let bonds = st
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
                        let pointers = st
                            .masks(Token {
                                kind: ADD_ATOM,
                                atom_type: id,
                                bond: b,
                                ..blank()
                            })
                            .pointers;
                        for p in 0..32u8 {
                            if pointers & (1u32 << p) == 0 {
                                continue;
                            }
                            prefix.push(Token {
                                kind: ADD_ATOM,
                                atom_type: id,
                                bond: b,
                                pointer: p,
                            });
                            dfs(prefix, limits, budget, out, cap);
                            prefix.pop();
                        }
                    }
                }
            }
        }
        if kinds & (1u32 << CLOSE_RING) != 0 {
            let bonds = st
                .masks(Token {
                    kind: CLOSE_RING,
                    ..blank()
                })
                .bonds;
            for b in 1..=3u8 {
                if bonds & (1u32 << b) == 0 {
                    continue;
                }
                let pointers = st
                    .masks(Token {
                        kind: CLOSE_RING,
                        bond: b,
                        ..blank()
                    })
                    .pointers;
                for p in 0..32u8 {
                    if pointers & (1u32 << p) == 0 {
                        continue;
                    }
                    prefix.push(Token {
                        kind: CLOSE_RING,
                        bond: b,
                        pointer: p,
                        ..blank()
                    });
                    dfs(prefix, limits, budget, out, cap);
                    prefix.pop();
                }
            }
        }
    }
    let mut out = Vec::new();
    let mut prefix = vec![start_token()];
    dfs(&mut prefix, limits, budget, &mut out, cap);
    out
}

#[test]
fn offered_types_and_bonds_always_have_a_pointer() {
    let limits = Limits::new(6, 1).unwrap();
    let (a, rmax) = (6usize, 1u32);
    let device = dev();
    let constants = Ms2Constants::new(&device);
    // The collector asserts the property at every visited prefix of both
    // budgets; the device then replays a sample of the offers.
    let c2: Composition = [2, 6, 0, 1, 0, 0, 0, 0, 0, 0];
    let c3n: Composition = [3, 7, 1, 1, 0, 0, 0, 0, 0, 0];
    let offers2 = collect_offered_pairs(c2, limits, usize::MAX);
    let offers3n = collect_offered_pairs(c3n, limits, usize::MAX);
    println!(
        "C2H6O {} offers, C3H7NO {} offers",
        offers2.len(),
        offers3n.len()
    );
    // 64 offers per budget (or all of them when fewer): at least 64 rows.
    let mut sample: Vec<(Vec<Token>, u8, u8, u8, Composition)> = Vec::new();
    for (offers, budget) in [(offers2, c2), (offers3n, c3n)] {
        for (prefix, id, b, p) in offers.into_iter().take(64) {
            sample.push((prefix, id, b, p, budget));
        }
    }
    assert!(
        sample.len() >= 64,
        "at least 64 sampled prefixes, got {}",
        sample.len()
    );
    // One row per offer: the prefix plus an ADD carrying the offered (type,
    // bond) pair on an offered pointer (legal by construction).
    let mut traces: Vec<Vec<Token>> = Vec::with_capacity(sample.len());
    let mut budgets: Vec<Composition> = Vec::with_capacity(sample.len());
    for (prefix, id, b, p, budget) in &sample {
        let mut trace = prefix.clone();
        trace.push(Token {
            kind: ADD_ATOM,
            atom_type: *id,
            bond: *b,
            pointer: *p,
        });
        traces.push(trace);
        budgets.push(*budget);
    }
    let rows = traces.len();
    let t = traces.iter().map(|tr| tr.len()).max().unwrap();
    let mut tokens = vec![0u32; rows * t * 4];
    let mut meta = vec![0u32; rows * 12];
    for (r, (trace, budget)) in traces.iter().zip(budgets.iter()).enumerate() {
        for (s2, tok) in trace.iter().enumerate() {
            tokens[(r * t + s2) * 4] = u32::from(tok.kind);
            tokens[(r * t + s2) * 4 + 1] = u32::from(tok.atom_type);
            tokens[(r * t + s2) * 4 + 2] = u32::from(tok.bond);
            tokens[(r * t + s2) * 4 + 3] = u32::from(tok.pointer);
        }
        meta[r * 12] = trace.len() as u32;
        meta[r * 12 + 1] = 2;
        for (e, count) in budget.iter().enumerate() {
            meta[r * 12 + 2 + e] = u32::from(*count);
        }
    }
    let tokens_t = IdTensor::from_slice(&tokens, vec![rows, t, 4], &device).unwrap();
    let meta_t = IdTensor::from_slice(&meta, vec![rows, 12], &device).unwrap();
    let out = ReplayBuffers::poisoned(rows, t, a, &device).unwrap();
    ms2::grammar_replay(&tokens_t, &meta_t, &constants, a as u32, rmax, &out).unwrap();
    check_launches(&device).unwrap();
    let replay = out.replay.try_to_vec().unwrap();
    let atoms = out.atoms.try_to_vec().unwrap();
    for (r, (trace, budget)) in traces.iter().zip(budgets.iter()).enumerate() {
        let host = host_replay(trace, limits, *budget, 2, a);
        assert_eq!(
            host.first_illegal,
            u32::MAX,
            "row {r}: the offered pair stays legal"
        );
        let length = trace.len();
        for s2 in 0..t {
            let base = (r * t + s2) * (4 + a);
            if s2 >= length {
                for w in 0..4 + a {
                    assert_eq!(replay[base + w], 0, "row {r} step {s2}: zero word {w}");
                }
                continue;
            }
            assert_eq!(replay[base], host.kinds[s2], "row {r} step {s2}: kinds");
            assert_eq!(replay[base + 1], host.types[s2], "row {r} step {s2}: types");
            assert_eq!(replay[base + 2], host.bonds[s2], "row {r} step {s2}: bonds");
            assert_eq!(
                replay[base + 3],
                host.pointers[s2],
                "row {r} step {s2}: pointers"
            );
            for j in 0..a {
                assert_eq!(
                    replay[base + 4 + j],
                    host.resids[s2][j],
                    "row {r} step {s2}: residual {j}"
                );
            }
        }
        // The device pointer mask at the offer step equals the host's
        // non-zero mask.
        let base = (r * t + length - 1) * (4 + a);
        assert_eq!(
            replay[base + 3],
            host.pointers[length - 1],
            "row {r}: the device offer mask equals the host's"
        );
        assert_ne!(
            replay[base + 3],
            0,
            "row {r}: the device offer mask is non-zero"
        );
        for j in 0..a {
            assert_eq!(
                atoms[r * (a + 1) + j],
                host.add_steps[j],
                "row {r}: atom {j} step"
            );
        }
        assert_eq!(atoms[r * (a + 1) + a], u32::MAX, "row {r}: legal");
    }
}
