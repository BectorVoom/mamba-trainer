//! MC3b tests: exact-rule completion sampling, host acceptance, the ranked
//! shortlist and recovery metrics.
//!
//! Hand-built molecules only (atom type ids from
//! [`chem::ATOM_TYPES`](mamba3::models::ms2::chem::ATOM_TYPES)), copied from
//! tests/ms2_completion_model.rs: ethanol, dimethyl ether, propan-1-ol,
//! propan-2-ol, methoxyethane, ethylamine, dimethylamine, cyclopropane and a
//! kekulized benzene ring with a methyl. The shared setup trains
//! [`CompletionModelConfig::small`](mamba3::models::ms2::completion_model::CompletionModelConfig::small)
//! on the nine molecules until the mean NLL is below 30% of its start (at
//! most 300 steps, fixed seeds). Every device call is followed by
//! [`check_launches`].
//!
//! Counter-sensitive tests (reads, launches) need exact process-global
//! counters, so every test in this file holds the file-local serial lock for
//! its whole body: the binary runs serially and no concurrent test can add
//! launches or reads inside a measurement.

#![cfg(feature = "backend")]

use std::collections::BTreeMap;

use mamba3::backend::{
    Device, check_launches, launch_count, reset_launch_count, runtime_read_count,
};
use mamba3::backends::Auto;
use mamba3::models::ms2::chem::Composition;
use mamba3::models::ms2::completion::contains_pattern;
use mamba3::models::ms2::completion_data::{
    CompletionExample, CompletionSet, ExtractionConfig, PatternSource, same_identity, skeleton,
};
use mamba3::models::ms2::completion_eval::{
    OutcomeCounts, QueryScore, recovery_report, score_query,
};
use mamba3::models::ms2::completion_model::{
    Acceptance, CompletionCandidate, CompletionGenerationConfig, CompletionModelConfig, CompletionRequest,
    CompletionTrainConfig, CompletionTrainer, QueryOutcome, SubstructureSemantics, accepts,
};
use mamba3::models::ms2::contain::Containment;
use mamba3::models::ms2::contract::candidate_status;
use mamba3::models::ms2::grammar::{
    CANONICAL_WORK_LIMIT, Limits, Token, canonical_trace, replay, replay_exact,
};
use mamba3::models::ms2::graph::MolGraph;
use mamba3::models::ms2::twin;
use mamba3::tensor::ops::ms2::Ms2Constants;

type R = Auto;
type E = f32;

/// File-local serial lock: counter-sensitive tests need exact process-global
/// launch/read counters, so the whole binary runs serially.
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn serial() -> std::sync::MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

fn dev() -> Device<R> {
    Device::<R>::default()
}

fn limits() -> Limits {
    Limits::new(16, 4).unwrap()
}

/// Ethanol `[C(H3), C(H2), O(H1)]`.
fn ethanol() -> MolGraph {
    MolGraph::new(vec![4, 3, 9], vec![(0, 1, 1), (1, 2, 1)]).unwrap()
}

/// Dimethyl ether `[C(H3), O(H0), C(H3)]`, a C2H6O isomer of ethanol.
fn dimethyl_ether() -> MolGraph {
    MolGraph::new(vec![4, 8, 4], vec![(0, 1, 1), (1, 2, 1)]).unwrap()
}

/// Propan-1-ol: a C(H3)-C(H2)-C(H2)-O(H1) chain.
fn propan_1_ol() -> MolGraph {
    MolGraph::new(vec![4, 3, 3, 9], vec![(0, 1, 1), (1, 2, 1), (2, 3, 1)]).unwrap()
}

/// Propan-2-ol: central C(H1) with two methyls and one O(H1).
fn propan_2_ol() -> MolGraph {
    MolGraph::new(vec![4, 2, 4, 9], vec![(0, 1, 1), (1, 2, 1), (1, 3, 1)]).unwrap()
}

/// Methoxyethane: C(H3)-O(H0)-C(H2)-C(H3), a C3H8O isomer.
fn methoxyethane() -> MolGraph {
    MolGraph::new(vec![4, 8, 3, 4], vec![(0, 1, 1), (1, 2, 1), (2, 3, 1)]).unwrap()
}

/// Ethylamine: C(H3)-C(H2)-N(H2).
fn ethylamine() -> MolGraph {
    MolGraph::new(vec![4, 3, 7], vec![(0, 1, 1), (1, 2, 1)]).unwrap()
}

/// Dimethylamine: C(H3)-N(H1)-C(H3), a C2H7N isomer of ethylamine (the
/// two-bond nitrogen carries one hydrogen, type 6).
fn dimethylamine() -> MolGraph {
    MolGraph::new(vec![4, 6, 4], vec![(0, 1, 1), (1, 2, 1)]).unwrap()
}

/// Cyclopropane: three `C(H2)` (id 3) in a single-bond ring.
fn cyclopropane() -> MolGraph {
    MolGraph::new(vec![3, 3, 3], vec![(0, 1, 1), (1, 2, 1), (2, 0, 1)]).unwrap()
}

/// Kekulized benzene ring with a methyl: six ring atoms with alternating
/// single/double bonds (`C(H1)` id 2, ipso `C(H0)` id 1) plus a `C(H3)`
/// methyl (id 4) on atom 0.
fn methylbenzene() -> MolGraph {
    MolGraph::new(
        vec![1, 2, 2, 2, 2, 2, 4],
        vec![
            (0, 1, 1),
            (1, 2, 2),
            (2, 3, 1),
            (3, 4, 2),
            (4, 5, 1),
            (5, 0, 2),
            (0, 6, 1),
        ],
    )
    .unwrap()
}

/// Ethene `[C(H2), C(H2)]` with a double bond (closed-shell).
fn ethene() -> MolGraph {
    MolGraph::new(vec![3, 3], vec![(0, 1, 2)]).unwrap()
}

/// The bond-order isomer of [`ethene`]: the same atoms with a single bond
/// (a valid open graph, residual 1 on each carbon). Strictly non-identical
/// to ethene with an identical skeleton: the documented case where the
/// skeleton merges bond-order isomers.
fn ethene_single_bond() -> MolGraph {
    MolGraph::new(vec![3, 3], vec![(0, 1, 1)]).unwrap()
}

/// A C=O double-bond pattern (C(H1) id 2 to O(H0) id 8): no complete C2H6O
/// molecule (saturated, single bonds only) can contain it.
fn carbonyl_pattern() -> MolGraph {
    MolGraph::new(vec![2, 8], vec![(0, 1, 2)]).unwrap()
}

/// A lone C(H3) methyl: contained in every complete C2H6O molecule (both
/// isomers carry methyls).
fn methyl_pattern() -> MolGraph {
    MolGraph::new(vec![4], vec![]).unwrap()
}

/// The nine overfit molecules in a fixed order.
fn nine_molecules() -> Vec<(&'static str, MolGraph)> {
    vec![
        ("ethanol", ethanol()),
        ("dimethyl ether", dimethyl_ether()),
        ("propan-1-ol", propan_1_ol()),
        ("propan-2-ol", propan_2_ol()),
        ("methoxyethane", methoxyethane()),
        ("ethylamine", ethylamine()),
        ("dimethylamine", dimethylamine()),
        ("cyclopropane", cyclopropane()),
        ("methylbenzene", methylbenzene()),
    ]
}

/// Canonical trace and composition of `graph`, asserting the trace replays
/// under the exact-completion grammar to a stopped, complete state.
fn trace_and_composition(graph: &MolGraph) -> (Vec<Token>, Composition) {
    let canonical = canonical_trace(graph, limits(), CANONICAL_WORK_LIMIT).unwrap();
    let composition = graph.composition();
    let end = replay_exact(&canonical.trace, limits(), composition).unwrap();
    assert!(
        end.stopped() && end.is_complete(),
        "canonical trace replays exact to a complete molecule"
    );
    (canonical.trace, composition)
}

/// A [`CompletionSet`] over the nine hand-built molecules.
fn nine_set() -> CompletionSet {
    let mut examples = Vec::new();
    for (i, (key, graph)) in nine_molecules().into_iter().enumerate() {
        let (trace, composition) = trace_and_composition(&graph);
        examples.push(CompletionExample {
            key: key.to_string(),
            identity_group: i as u64,
            source_index: i,
            target: graph,
            composition,
            trace,
            skeleton_trace: Vec::new(),
        });
    }
    CompletionSet {
        limits: limits(),
        examples,
        skipped: BTreeMap::new(),
        max_expansions: CANONICAL_WORK_LIMIT,
    }
}

fn train_config() -> CompletionTrainConfig {
    CompletionTrainConfig {
        lr: 3e-3,
        weight_decay: 0.0,
        grad_clip: None,
        seed: 1,
        extraction: ExtractionConfig::default(),
        extraction_seed: 11,
        pattern_source: PatternSource::RandomPatches(ExtractionConfig::default()),
        fingerprint_mode: None,
        fingerprint_threshold: 0.1,
    }
}

/// Train [`CompletionModelConfig::small`] on the nine molecules until the
/// mean NLL is below 30% of its start (at most 300 steps, fixed seeds),
/// returning the trainer, the set and the (initial, final) losses.
fn train_overfit(device: &Device<R>) -> (CompletionTrainer<R, E>, CompletionSet, f32, f32) {
    let set = nine_set();
    let indices: Vec<usize> = (0..set.examples.len()).collect();
    let mut trainer =
        CompletionTrainer::new(&CompletionModelConfig::small(), &train_config(), device).unwrap();
    trainer.request_report();
    let initial = trainer.step(&set, &indices, 0).unwrap().unwrap();
    println!("mc3b overfit step 0: loss {initial}");
    check_launches(device).unwrap();
    let mut reported = initial;
    let mut done = false;
    for step in 1..=300 {
        if step % 10 == 0 {
            trainer.request_report();
        }
        if let Some(loss) = trainer.step(&set, &indices, 0).unwrap() {
            reported = loss;
            println!("mc3b overfit step {step}: loss {loss}");
            if loss < 0.3 * initial {
                done = true;
                break;
            }
        }
    }
    check_launches(device).unwrap();
    assert!(
        done && reported < 0.3 * initial,
        "loss falls below 30% of its start (initial {initial}, final {reported})"
    );
    (trainer, set, initial, reported)
}

fn gen_config(trajectories: u32, seed: u64) -> CompletionGenerationConfig {
    CompletionGenerationConfig {
        trajectories,
        temperature: 1.0,
        seed,
        returned: 25,
        containment_node_limit: 100_000,
        identity_work_limit: 100_000,
        condition_on_patterns: true,
        substructure_semantics: SubstructureSemantics::Contained,
    }
}

/// Patterns each example sees at draw 0 (the training draw), in set order.
fn draw_zero_patterns(set: &CompletionSet) -> Vec<Vec<MolGraph>> {
    set.examples
        .iter()
        .map(|example| {
            example
                .patterns(
                    &train_config().extraction,
                    train_config().extraction_seed,
                    0,
                )
                .unwrap()
                .into_iter()
                .map(|pattern| pattern.graph)
                .collect()
        })
        .collect()
}

/// Compare two outcomes field for field (graphs by atoms and bonds; floats
/// by bits).
fn assert_outcomes_equal(a: &QueryOutcome, b: &QueryOutcome, what: &str) {
    assert_eq!(a.trajectories, b.trajectories, "{what}: trajectories");
    assert_eq!(a.finished, b.finished, "{what}: finished");
    assert_eq!(a.dead_end, b.dead_end, "{what}: dead_end");
    assert_eq!(a.truncated, b.truncated, "{what}: truncated");
    assert_eq!(
        a.rejected_replay, b.rejected_replay,
        "{what}: rejected_replay"
    );
    assert_eq!(
        a.rejected_containment, b.rejected_containment,
        "{what}: rejected_containment"
    );
    assert_eq!(
        a.containment_unresolved, b.containment_unresolved,
        "{what}: containment_unresolved"
    );
    assert_eq!(
        a.rejected_extra_groups, b.rejected_extra_groups,
        "{what}: rejected_extra_groups"
    );
    assert_eq!(
        a.rejected_missing_groups, b.rejected_missing_groups,
        "{what}: rejected_missing_groups"
    );
    assert_eq!(
        a.pass_contained, b.pass_contained,
        "{what}: pass_contained"
    );
    assert_eq!(
        a.pass_disjoint, b.pass_disjoint,
        "{what}: pass_disjoint"
    );
    assert_eq!(
        a.pass_complete, b.pass_complete,
        "{what}: pass_complete"
    );
    assert_eq!(
        a.infeasible, b.infeasible,
        "{what}: infeasible"
    );
    assert_eq!(
        a.identity_unresolved, b.identity_unresolved,
        "{what}: identity_unresolved"
    );
    assert_eq!(a.other_status, b.other_status, "{what}: other_status");
    assert_eq!(a.distinct, b.distinct, "{what}: distinct");
    assert_eq!(
        a.unresolved.len(),
        b.unresolved.len(),
        "{what}: unresolved count"
    );
    for (i, (ca, cb)) in a.unresolved.iter().zip(b.unresolved.iter()).enumerate() {
        assert_eq!(ca.trace, cb.trace, "{what}: unresolved {i} trace");
        assert_eq!(ca.samples, cb.samples, "{what}: unresolved {i} samples");
        assert_eq!(
            ca.best_log_prob.to_bits(),
            cb.best_log_prob.to_bits(),
            "{what}: unresolved {i} log-probability"
        );
        assert_eq!(
            ca.graph.atoms(),
            cb.graph.atoms(),
            "{what}: unresolved {i} atoms"
        );
        assert_eq!(
            ca.graph.bonds(),
            cb.graph.bonds(),
            "{what}: unresolved {i} bonds"
        );
    }
    assert_eq!(
        a.candidates.len(),
        b.candidates.len(),
        "{what}: candidate count"
    );
    for (i, (ca, cb)) in a.candidates.iter().zip(b.candidates.iter()).enumerate() {
        assert_eq!(ca.trace, cb.trace, "{what}: candidate {i} trace");
        assert_eq!(ca.samples, cb.samples, "{what}: candidate {i} samples");
        assert_eq!(
            ca.best_log_prob.to_bits(),
            cb.best_log_prob.to_bits(),
            "{what}: candidate {i} log-probability"
        );
        assert_eq!(
            ca.graph.atoms(),
            cb.graph.atoms(),
            "{what}: candidate {i} atoms"
        );
        assert_eq!(
            ca.graph.bonds(),
            cb.graph.bonds(),
            "{what}: candidate {i} bonds"
        );
    }
    assert_eq!(a.sampled.len(), b.sampled.len(), "{what}: sampled count");
    for (i, (sa, sb)) in a.sampled.iter().zip(b.sampled.iter()).enumerate() {
        assert_eq!(sa.trajectory, sb.trajectory, "{what}: sampled {i} index");
        assert_eq!(sa.trace, sb.trace, "{what}: sampled {i} trace");
        assert_eq!(
            sa.log_prob.to_bits(),
            sb.log_prob.to_bits(),
            "{what}: sampled {i} log-probability"
        );
        assert_eq!(sa.status, sb.status, "{what}: sampled {i} status");
    }
}

/// The acceptance invariants of section 3, shared by the trained-model test
/// and the untrained-baseline test.
fn check_acceptance_invariants(
    outcomes: &[QueryOutcome],
    requests: &[CompletionRequest],
    returned: usize,
    k: usize,
) {
    assert_eq!(outcomes.len(), requests.len(), "one outcome per request");
    for (q, (outcome, request)) in outcomes.iter().zip(requests.iter()).enumerate() {
        let at = format!("query {q}");
        assert_eq!(outcome.trajectories as usize, k, "{at}: trajectories");
        assert_eq!(outcome.sampled.len(), k, "{at}: sampled rows");
        assert_eq!(
            outcome.trajectories,
            outcome.finished + outcome.dead_end + outcome.truncated + outcome.other_status,
            "{at}: the counts add up to trajectories"
        );
        // The device counts match the exposed status words.
        let mut finished = 0u32;
        let mut dead_end = 0u32;
        let mut truncated = 0u32;
        let mut other = 0u32;
        for row in &outcome.sampled {
            if row.status & candidate_status::FINISHED != 0 {
                finished += 1;
            } else if row.status & candidate_status::NO_VALID_ACTION != 0 {
                dead_end += 1;
            } else if row.status & candidate_status::TRUNCATED != 0 {
                truncated += 1;
            } else {
                other += 1;
            }
        }
        assert_eq!(
            finished, outcome.finished,
            "{at}: finished matches statuses"
        );
        assert_eq!(
            dead_end, outcome.dead_end,
            "{at}: dead_end matches statuses"
        );
        assert_eq!(
            truncated, outcome.truncated,
            "{at}: truncated matches statuses"
        );
        assert_eq!(other, outcome.other_status, "{at}: other matches statuses");
        // The device never reports a FINISHED row the host cannot replay:
        // a recorded FINISHED is never trusted on its own.
        assert_eq!(
            outcome.rejected_replay, 0,
            "{at}: every FINISHED row replays exact to stopped + complete"
        );
        assert!(
            outcome.candidates.len() <= returned,
            "{at}: at most `returned` candidates"
        );
        if outcome.candidates.len() < returned {
            assert_eq!(
                outcome.distinct,
                outcome.candidates.len() as u32,
                "{at}: a short shortlist keeps every identity"
            );
        }
        // The ranking order: samples desc, best_log_prob desc, trace asc.
        for pair in outcome.candidates.windows(2) {
            let (a, c) = (&pair[0], &pair[1]);
            let order = c
                .samples
                .cmp(&a.samples)
                .then_with(|| c.best_log_prob.total_cmp(&a.best_log_prob))
                .then_with(|| a.trace.cmp(&c.trace));
            assert!(
                order != std::cmp::Ordering::Greater,
                "{at}: candidates are ranked ({} samples then {})",
                a.samples,
                c.samples
            );
        }
        // Candidates of a query are pairwise non-identical.
        for (i, a) in outcome.candidates.iter().enumerate() {
            // Independent re-check: stopped + complete, on composition,
            // connected, every pattern contained.
            let end =
                replay_exact(&a.trace, limits(), request.composition).expect("{at}: replayable");
            assert!(
                end.stopped() && end.is_complete(),
                "{at}: stopped + complete"
            );
            let graph = end.graph().expect("{at}: graph builds");
            assert_eq!(
                graph.composition(),
                request.composition,
                "{at}: composition equal"
            );
            assert!(graph.is_connected(), "{at}: connected");
            for pattern in request.patterns {
                assert_eq!(
                    contains_pattern(&graph, pattern, 100_000),
                    Containment::Contained,
                    "{at}: every pattern contained"
                );
            }
            for (j, c) in outcome.candidates.iter().enumerate().skip(i + 1) {
                assert_eq!(
                    same_identity(&a.graph, &c.graph, 100_000),
                    Some(false),
                    "{at}: candidates {i} and {j} are non-identical"
                );
            }
        }
        // Exact per-rule pass counts: every successfully replayed graph
        // counts under each rule it passes, including trajectories the
        // active rule rejects. Recomputed here from the recorded sampled
        // rows, so an omission (pass counts over accepted trajectories
        // only) fails.
        let pats = request.acceptance_patterns.unwrap_or(request.patterns);
        let mut expect = [0u32; 3];
        for row in &outcome.sampled {
            if row.status & candidate_status::FINISHED == 0 {
                continue;
            }
            let end = match replay_exact(&row.trace, limits(), request.composition) {
                Ok(end) => end,
                Err(_) => continue,
            };
            if !end.stopped() || !end.is_complete() {
                continue;
            }
            let graph = match end.graph() {
                Ok(graph) => graph,
                Err(_) => continue,
            };
            if !graph.is_connected() || graph.composition() != request.composition {
                continue;
            }
            if accepts(&graph, pats, SubstructureSemantics::Contained, 100_000, 100_000)
                == Acceptance::Accepted
            {
                expect[0] += 1;
            }
            if accepts(
                &graph,
                pats,
                SubstructureSemantics::DisjointOccurrences,
                100_000,
                100_000,
            ) == Acceptance::Accepted
            {
                expect[1] += 1;
            }
            if accepts(
                &graph,
                pats,
                SubstructureSemantics::CompleteFunctionalGroups,
                100_000,
                100_000,
            ) == Acceptance::Accepted
            {
                expect[2] += 1;
            }
        }
        assert_eq!(
            outcome.pass_contained, expect[0],
            "{at}: pass_contained counts every replayed pass"
        );
        assert_eq!(
            outcome.pass_disjoint, expect[1],
            "{at}: pass_disjoint counts every replayed pass"
        );
        assert_eq!(
            outcome.pass_complete, expect[2],
            "{at}: pass_complete counts every replayed pass"
        );
        // The identity accounting: finished trajectories are rejected,
        // certified or unresolved. The shortlist cut only drops whole
        // certified identities, so with no cut the candidate samples are the
        // certified total and the identity holds exactly.
        let candidate_samples: u32 = outcome.candidates.iter().map(|c| c.samples).sum();
        let unresolved_samples: u32 = outcome.unresolved.iter().map(|c| c.samples).sum();
        assert_eq!(
            outcome.identity_unresolved, unresolved_samples,
            "{at}: identity_unresolved counts every unresolved trajectory once"
        );
        let rejected =
            outcome.rejected_replay + outcome.rejected_containment + outcome.containment_unresolved;
        if outcome.candidates.len() == outcome.distinct as usize {
            assert_eq!(
                outcome.finished,
                rejected + candidate_samples + unresolved_samples,
                "{at}: the accounting identity holds with no cut"
            );
        } else {
            assert!(
                rejected + candidate_samples + unresolved_samples <= outcome.finished,
                "{at}: accounted trajectories within finished {}",
                outcome.finished
            );
        }
    }
}

#[test]
fn twin_helper_matches_patched_twin() {
    let _lock = serial();
    // The host init helper must equal the twin's started rows with the
    // started word patched from 1 to 2.
    let ids = [7u64, 0x1_0000_0005u64];
    let compositions = [ethanol().composition(), dimethyl_ether().composition()];
    let (k, steps, atoms) = (3usize, limits().max_steps(), 16usize);
    let (traj_meta, state, actions) =
        twin::init_completion_trajectories(&ids, &compositions, k, steps, atoms);
    let rows = ids.len() * k;
    assert_eq!(traj_meta.len(), rows * 14);
    assert_eq!(state.len(), rows * (3 * atoms + 16));
    assert_eq!(actions.len(), rows * (steps * 4 + atoms + 4));
    // The same inputs through the bare twin, patched by hand.
    let mut traj_formula = vec![0u32; rows * 12];
    for r in 0..rows {
        let b = r / k;
        traj_formula[r * 12] = 0;
        traj_formula[r * 12 + 1] = u32::MAX;
        for e in 0..10 {
            traj_formula[r * 12 + 2 + e] = u32::from(compositions[b][e]);
        }
    }
    let mut spectra_meta = vec![0u32; ids.len() * 8];
    for (b, id) in ids.iter().enumerate() {
        spectra_meta[b * 8] = 1;
        spectra_meta[b * 8 + 6] = (*id & 0xFFFF_FFFF) as u32;
        spectra_meta[b * 8 + 7] = (*id >> 32) as u32;
    }
    let (mut want_meta, want_state, want_actions) = twin::init_trajectories(
        &traj_formula,
        &spectra_meta,
        ids.len(),
        k,
        steps,
        atoms,
        false,
    );
    for r in 0..rows {
        // Every row started (START applied, length 1); the patch flips the
        // started word to exact-completion sampling.
        assert_eq!(want_meta[r * 14 + 3], 1, "row {r} started");
        want_meta[r * 14 + 3] = 2;
        assert_eq!(traj_meta[r * 14], spectra_meta[(r / k) * 8 + 6]);
        assert_eq!(traj_meta[r * 14 + 1], spectra_meta[(r / k) * 8 + 7]);
        assert_eq!(traj_meta[r * 14 + 2], (r % k) as u32);
        for e in 0..10 {
            assert_eq!(traj_meta[r * 14 + 4 + e], u32::from(compositions[r / k][e]));
        }
    }
    assert_eq!(traj_meta, want_meta, "traj_meta is the patched twin");
    assert_eq!(state, want_state, "state is the twin's START-applied row");
    assert_eq!(actions, want_actions, "actions is the twin's record");
}

#[test]
fn sampled_log_prob_matches_teacher() {
    let _lock = serial();
    let device = dev();
    let (mut trainer, set, _, _) = train_overfit(&device);
    let patterns = draw_zero_patterns(&set);
    let refs: Vec<&[MolGraph]> = patterns.iter().map(Vec::as_slice).collect();
    let requests: Vec<CompletionRequest> = set
        .examples
        .iter()
        .enumerate()
        .map(|(i, example)| CompletionRequest {
            id: example.identity_group,
            composition: example.composition,
            patterns: refs[i],
            acceptance_patterns: None,
            fingerprint: None,
        })
        .collect();
    let config = gen_config(32, 101);
    let constants = Ms2Constants::new(&device);
    let outcomes = trainer
        .model()
        .generate(&requests, &config, &constants, &device)
        .unwrap();
    check_launches(&device).unwrap();
    // Every finished trajectory whose trace replays exact: the recorded
    // trace log-probability against minus the teacher-forced NLL of that
    // same trace under the same request.
    let mut batch_patterns: Vec<&[MolGraph]> = Vec::new();
    let mut batch_traces: Vec<&[Token]> = Vec::new();
    let mut batch_comps: Vec<Composition> = Vec::new();
    let mut recorded: Vec<f32> = Vec::new();
    for (outcome, request) in outcomes.iter().zip(requests.iter()) {
        for row in &outcome.sampled {
            if row.status & candidate_status::FINISHED == 0 {
                continue;
            }
            let end = replay_exact(&row.trace, limits(), request.composition)
                .expect("a FINISHED row replays exact");
            assert!(
                end.stopped() && end.is_complete(),
                "a FINISHED row ends stopped + complete"
            );
            batch_patterns.push(request.patterns);
            batch_traces.push(&row.trace);
            batch_comps.push(request.composition);
            recorded.push(row.log_prob);
        }
    }
    assert!(
        !recorded.is_empty(),
        "the trained model finishes trajectories to compare"
    );
    let nlls = trainer
        .teacher_eval_with(&batch_patterns, &batch_traces, &batch_comps)
        .unwrap();
    check_launches(&device).unwrap();
    let mut worst = 0.0f32;
    for (lp, nll) in recorded.iter().zip(nlls.iter()) {
        let diff = (*lp + *nll).abs();
        worst = worst.max(diff);
        assert!(
            diff <= 1e-3,
            "sampled {lp} against teacher {} (diff {diff})",
            -nll
        );
    }
    println!(
        "sampled_log_prob_matches_teacher: {} finished traces, largest |sampled − teacher| = {worst}",
        recorded.len()
    );
}

#[test]
fn generation_is_deterministic_and_seeded() {
    let _lock = serial();
    let device = dev();
    let (trainer, set, _, _) = train_overfit(&device);
    let patterns = draw_zero_patterns(&set);
    let refs: Vec<&[MolGraph]> = patterns.iter().map(Vec::as_slice).collect();
    let requests: Vec<CompletionRequest> = set
        .examples
        .iter()
        .map(|example| CompletionRequest {
            id: example.identity_group,
            composition: example.composition,
            patterns: refs[example.identity_group as usize],
            acceptance_patterns: None,
            fingerprint: None,
        })
        .collect();
    let config = gen_config(32, 202);
    let constants = Ms2Constants::new(&device);
    let model = trainer.model();
    let first = model
        .generate(&requests, &config, &constants, &device)
        .unwrap();
    check_launches(&device).unwrap();
    let second = model
        .generate(&requests, &config, &constants, &device)
        .unwrap();
    check_launches(&device).unwrap();
    for (q, (a, b)) in first.iter().zip(second.iter()).enumerate() {
        assert_outcomes_equal(a, b, &format!("query {q}: same config repeats"));
    }
    let traces_of = |outcomes: &[QueryOutcome]| -> Vec<Vec<Token>> {
        outcomes
            .iter()
            .flat_map(|outcome| outcome.sampled.iter().map(|row| row.trace.clone()))
            .collect()
    };
    // A different seed samples different traces somewhere.
    let mut other_config = config.clone();
    other_config.seed = 999;
    let other = model
        .generate(&requests, &other_config, &constants, &device)
        .unwrap();
    check_launches(&device).unwrap();
    assert_ne!(
        traces_of(&first),
        traces_of(&other),
        "a different seed moves the traces"
    );
    // A different request id samples different traces somewhere.
    let mut requests2: Vec<CompletionRequest> = set
        .examples
        .iter()
        .map(|example| CompletionRequest {
            id: example.identity_group,
            composition: example.composition,
            patterns: refs[example.identity_group as usize],
            acceptance_patterns: None,
            fingerprint: None,
        })
        .collect();
    requests2[8].id = 1_000_000;
    // The diverted query is methylbenzene (index 8): under the v2 lookahead
    // the eight smaller molecules sample a single forced trace each, so only
    // a query with sampling freedom can observe the id in its traces.
    let moved = model
        .generate(&requests2, &config, &constants, &device)
        .unwrap();
    check_launches(&device).unwrap();
    assert_ne!(
        traces_of(&first),
        traces_of(&moved),
        "a different request id moves the traces"
    );
    // Batch independence: a batch of 3 against the three singletons.
    let trio: Vec<CompletionRequest> = requests[0..3]
        .iter()
        .map(|request| CompletionRequest {
            id: request.id,
            composition: request.composition,
            patterns: request.patterns,
            acceptance_patterns: None,
            fingerprint: None,
        })
        .collect();
    let batched = model.generate(&trio, &config, &constants, &device).unwrap();
    check_launches(&device).unwrap();
    for (q, request) in trio.iter().enumerate() {
        let alone = model
            .generate(
                &[CompletionRequest {
                    id: request.id,
                    composition: request.composition,
                    patterns: request.patterns,
                    acceptance_patterns: None,
                    fingerprint: None,
                }],
                &config,
                &constants,
                &device,
            )
            .unwrap();
        check_launches(&device).unwrap();
        assert_outcomes_equal(
            &batched[q],
            &alone[0],
            &format!("query {q}: batch vs singleton"),
        );
    }
}

#[test]
fn every_candidate_is_complete_and_contains_the_patterns() {
    let _lock = serial();
    let device = dev();
    let (trainer, set, _, _) = train_overfit(&device);
    let patterns = draw_zero_patterns(&set);
    let refs: Vec<&[MolGraph]> = patterns.iter().map(Vec::as_slice).collect();
    let requests: Vec<CompletionRequest> = set
        .examples
        .iter()
        .map(|example| CompletionRequest {
            id: example.identity_group,
            composition: example.composition,
            patterns: refs[example.identity_group as usize],
            acceptance_patterns: None,
            fingerprint: None,
        })
        .collect();
    let config = gen_config(32, 303);
    let constants = Ms2Constants::new(&device);
    let outcomes = trainer
        .model()
        .generate(&requests, &config, &constants, &device)
        .unwrap();
    check_launches(&device).unwrap();
    check_acceptance_invariants(&outcomes, &requests, 25, 32);
}

#[test]
fn overfit_model_recovers_training_molecules() {
    let _lock = serial();
    let device = dev();
    let (trainer, set, _, _) = train_overfit(&device);
    let patterns = draw_zero_patterns(&set);
    let refs: Vec<&[MolGraph]> = patterns.iter().map(Vec::as_slice).collect();
    let requests: Vec<CompletionRequest> = set
        .examples
        .iter()
        .map(|example| CompletionRequest {
            id: example.identity_group,
            composition: example.composition,
            patterns: refs[example.identity_group as usize],
            acceptance_patterns: None,
            fingerprint: None,
        })
        .collect();
    let config = gen_config(64, 404);
    let constants = Ms2Constants::new(&device);
    let outcomes = trainer
        .model()
        .generate(&requests, &config, &constants, &device)
        .unwrap();
    check_launches(&device).unwrap();
    let mut ranks: Vec<Option<u32>> = Vec::new();
    let mut finished_sum = 0u32;
    let mut dead_end_sum = 0u32;
    let mut traj_sum = 0u32;
    for ((outcome, example), name) in outcomes
        .iter()
        .zip(set.examples.iter())
        .zip(nine_molecules().into_iter().map(|(name, _)| name))
    {
        let score = score_query(outcome, &example.target, 100_000);
        println!(
            "{name}: rank {:?} ({} candidates, {} distinct)",
            score.rank,
            outcome.candidates.len(),
            outcome.distinct
        );
        ranks.push(score.rank);
        finished_sum += outcome.finished;
        dead_end_sum += outcome.dead_end;
        traj_sum += outcome.trajectories;
    }
    let first = ranks.iter().filter(|rank| **rank == Some(1)).count();
    let present = ranks.iter().filter(|rank| rank.is_some()).count();
    println!(
        "overfit recovery: {first}/9 ranked first, {present}/9 in the shortlist; \
         finished fraction {:.3}, dead-end fraction {:.3}",
        f64::from(finished_sum) / f64::from(traj_sum),
        f64::from(dead_end_sum) / f64::from(traj_sum)
    );
    assert_eq!(ranks.len(), 9, "one rank per molecule");
    assert!(
        present == 9,
        "the target is in the shortlist for all 9 molecules: {ranks:?}"
    );
    assert!(
        first >= 7,
        "the target is ranked first for at least 7 of the 9 molecules: {ranks:?}"
    );
}

#[test]
fn identity_budget_exhaustion_never_duplicates() {
    let _lock = serial();
    let device = dev();
    let (trainer, set, _, _) = train_overfit(&device);
    let patterns = draw_zero_patterns(&set);
    let refs: Vec<&[MolGraph]> = patterns.iter().map(Vec::as_slice).collect();
    let requests: Vec<CompletionRequest> = set
        .examples
        .iter()
        .enumerate()
        .map(|(i, example)| CompletionRequest {
            id: example.identity_group,
            composition: example.composition,
            patterns: refs[i],
            acceptance_patterns: None,
            fingerprint: None,
        })
        .collect();
    let constants = Ms2Constants::new(&device);
    // Zero budget means no search: every non-trace-equal comparison in a
    // bucket is unresolved. K = 16 stays below `returned`, so no cut drops a
    // certified identity and the accounting identity is exact.
    let mut exhausted = gen_config(16, 404);
    exhausted.identity_work_limit = 0;
    let starved = trainer
        .model()
        .generate(&requests, &exhausted, &constants, &device)
        .unwrap();
    check_launches(&device).unwrap();
    for (q, outcome) in starved.iter().enumerate() {
        let at = format!("query {q} (no search)");
        assert_eq!(
            outcome.distinct as usize,
            outcome.candidates.len(),
            "{at}: no cut at K = 16"
        );
        // Candidates hold pairwise distinct traces, and no candidate graph
        // is identical to another even with a full search budget.
        for (i, a) in outcome.candidates.iter().enumerate() {
            for (j, c) in outcome.candidates.iter().enumerate().skip(i + 1) {
                assert_ne!(
                    a.trace, c.trace,
                    "{at}: candidates {i} and {j} share a trace"
                );
                assert_eq!(
                    same_identity(&a.graph, &c.graph, 100_000),
                    Some(false),
                    "{at}: candidates {i} and {j} are identical under a full budget"
                );
            }
        }
        // Unresolved entries repeat neither each other nor any candidate
        // trace: repeats merge by exact-trace equality.
        for entry in &outcome.unresolved {
            assert!(
                !outcome.candidates.iter().any(|c| c.trace == entry.trace),
                "{at}: an unresolved trace repeats a candidate trace"
            );
        }
        for (i, a) in outcome.unresolved.iter().enumerate() {
            for c in outcome.unresolved.iter().skip(i + 1) {
                assert_ne!(
                    a.trace, c.trace,
                    "{at}: unresolved entries {i} share a trace"
                );
            }
        }
        let certified: u32 = outcome.candidates.iter().map(|c| c.samples).sum();
        let waiting: u32 = outcome.unresolved.iter().map(|c| c.samples).sum();
        assert_eq!(
            outcome.identity_unresolved, waiting,
            "{at}: identity_unresolved counts every unresolved trajectory once"
        );
        assert_eq!(
            outcome.finished,
            outcome.rejected_replay
                + outcome.rejected_containment
                + outcome.containment_unresolved
                + certified
                + waiting,
            "{at}: finished == rejected + certified + unresolved"
        );
    }
    // Repeats do merge: some identity holds more than one trajectory.
    assert!(
        starved.iter().any(|outcome| outcome
            .candidates
            .iter()
            .chain(outcome.unresolved.iter())
            .any(|c| c.samples >= 2)),
        "identical traces merge instead of splitting"
    );
    // The budget only regroups identical draws: the default budget samples
    // the same trajectories at the same seed.
    let default16 = gen_config(16, 404);
    let regrouped = trainer
        .model()
        .generate(&requests, &default16, &constants, &device)
        .unwrap();
    check_launches(&device).unwrap();
    for (q, (s, c)) in starved.iter().zip(regrouped.iter()).enumerate() {
        let a: Vec<&Vec<Token>> = s.sampled.iter().map(|row| &row.trace).collect();
        let b: Vec<&Vec<Token>> = c.sampled.iter().map(|row| &row.trace).collect();
        assert_eq!(a, b, "query {q}: the budget only regroups identical draws");
    }
    // With the default budget nothing is unresolved, and the nine ranks match
    // the pre-change behaviour (every target present, most ranked first).
    for outcome in &regrouped {
        assert!(outcome.unresolved.is_empty());
        assert_eq!(outcome.identity_unresolved, 0);
    }
    let full = gen_config(64, 404);
    let outcomes = trainer
        .model()
        .generate(&requests, &full, &constants, &device)
        .unwrap();
    check_launches(&device).unwrap();
    let mut ranks = Vec::new();
    for (outcome, example) in outcomes.iter().zip(set.examples.iter()) {
        assert!(outcome.unresolved.is_empty());
        ranks.push(score_query(outcome, &example.target, 100_000).rank);
    }
    println!("default-budget ranks: {ranks:?}");
    assert!(
        ranks.iter().all(|rank| rank.is_some()),
        "every target is in the shortlist: {ranks:?}"
    );
    assert!(
        ranks.iter().filter(|rank| **rank == Some(1)).count() >= 7,
        "at least 7 targets ranked first: {ranks:?}"
    );
}

#[test]
fn structured_seeds_and_ids_do_not_collide() {
    let _lock = serial();
    let device = dev();
    let (trainer, set, _, _) = train_overfit(&device);
    let patterns = draw_zero_patterns(&set);
    let refs: Vec<&[MolGraph]> = patterns.iter().map(Vec::as_slice).collect();
    // Methylbenzene (index 8): the eight smaller molecules sample a single
    // forced trace each under the v2 lookahead, so the seed/id streams are
    // only observable through a query with sampling freedom.
    let composition = set.examples[8].composition;
    let request_with = |id: u64| CompletionRequest {
        id,
        composition,
        patterns: refs[8],
        acceptance_patterns: None,
        fingerprint: None,
    };
    let constants = Ms2Constants::new(&device);
    let traces_of = |seed: u64, id: u64| {
        let mut outcomes = trainer
            .model()
            .generate(
                &[request_with(id)],
                &gen_config(32, seed),
                &constants,
                &device,
            )
            .unwrap();
        outcomes
            .pop()
            .unwrap()
            .sampled
            .into_iter()
            .map(|row| row.trace)
            .collect::<Vec<_>>()
    };
    // Seeds 1 and 1 << 32 shared every draw under the old XOR mix.
    let low = traces_of(1, 0);
    check_launches(&device).unwrap();
    let high = traces_of(1 << 32, 0);
    check_launches(&device).unwrap();
    assert_ne!(low, high, "seeds 1 and 1 << 32 sample different streams");
    // Ids 0 and 0x6889_f849_0000_0001 collided under the old scheme at seed 0.
    let id_a = traces_of(0, 0);
    check_launches(&device).unwrap();
    let id_b = traces_of(0, 0x6889_f849_0000_0001);
    check_launches(&device).unwrap();
    assert_ne!(id_a, id_b, "structured ids sample different streams");
    // Two requests with the same id in one call are rejected.
    let err = match trainer.model().generate(
        &[request_with(9), request_with(9)],
        &gen_config(8, 0),
        &constants,
        &device,
    ) {
        Err(e) => e,
        Ok(_) => panic!("duplicate ids are rejected"),
    };
    assert!(
        err.to_string().contains("requests 0 and 1") && err.to_string().contains("share id"),
        "the rejection names both indices: {err}"
    );
    check_launches(&device).unwrap();
}

#[test]
fn patterns_are_a_hard_filter() {
    let _lock = serial();
    let device = dev();
    let (trainer, _, _, _) = train_overfit(&device);
    let (_, ethanol_comp) = trace_and_composition(&ethanol());
    let impossible = vec![carbonyl_pattern()];
    let request = CompletionRequest {
        id: 555,
        composition: ethanol_comp,
        patterns: impossible.as_slice(),
        acceptance_patterns: None,
        fingerprint: None,
    };
    let config = gen_config(32, 505);
    let constants = Ms2Constants::new(&device);
    let outcomes = trainer
        .model()
        .generate(&[request], &config, &constants, &device)
        .unwrap();
    check_launches(&device).unwrap();
    assert_eq!(outcomes.len(), 1);
    let outcome = &outcomes[0];
    assert!(
        outcome.candidates.is_empty(),
        "a C=O pattern admits no complete C2H6O molecule"
    );
    assert_eq!(outcome.distinct, 0);
    assert!(
        outcome.finished > 0,
        "the test is vacuous without finished trajectories"
    );
    assert_eq!(
        outcome.rejected_containment, outcome.finished,
        "every finished trajectory counts as rejected_containment"
    );
    assert_eq!(outcome.rejected_replay, 0);
    assert_eq!(outcome.containment_unresolved, 0);
}

#[test]
fn formula_only_control_keeps_the_filter() {
    let _lock = serial();
    let device = dev();
    let (trainer, _, _, _) = train_overfit(&device);
    let (_, ethanol_comp) = trace_and_composition(&ethanol());
    let permissive = vec![methyl_pattern()];
    let impossible = vec![carbonyl_pattern()];
    let mut config = gen_config(64, 607);
    config.condition_on_patterns = false;
    let constants = Ms2Constants::new(&device);
    // The two requests share composition and id but run in separate calls
    // (one call never holds two requests with the same id): only their
    // patterns differ, so the device samples identical traces for both.
    let request_a = CompletionRequest {
        id: 606,
        composition: ethanol_comp,
        patterns: permissive.as_slice(),
        acceptance_patterns: None,
        fingerprint: None,
    };
    let out_a = trainer
        .model()
        .generate(&[request_a], &config, &constants, &device)
        .unwrap();
    check_launches(&device).unwrap();
    let request_b = CompletionRequest {
        id: 606,
        composition: ethanol_comp,
        patterns: impossible.as_slice(),
        acceptance_patterns: None,
        fingerprint: None,
    };
    let out_b = trainer
        .model()
        .generate(&[request_b], &config, &constants, &device)
        .unwrap();
    check_launches(&device).unwrap();
    assert_eq!(out_a.len(), 1);
    assert_eq!(out_b.len(), 1);
    let traces_a: Vec<&Vec<Token>> = out_a[0].sampled.iter().map(|row| &row.trace).collect();
    let traces_b: Vec<&Vec<Token>> = out_b[0].sampled.iter().map(|row| &row.trace).collect();
    assert_eq!(
        traces_a, traces_b,
        "the formula-only control samples identical traces for identical (composition, id)"
    );
    // Acceptance still applies each request's own patterns.
    assert!(
        !out_a[0].candidates.is_empty(),
        "the permissive methyl pattern admits candidates"
    );
    assert!(
        out_b[0].candidates.is_empty(),
        "the impossible C=O pattern admits none"
    );
    for candidate in &out_a[0].candidates {
        assert_eq!(
            contains_pattern(&candidate.graph, &methyl_pattern(), 100_000),
            Containment::Contained,
            "every returned candidate contains its request's patterns"
        );
    }
    // The same control request repeated matches its first outcome.
    let request_a2 = CompletionRequest {
        id: 606,
        composition: ethanol_comp,
        patterns: permissive.as_slice(),
        acceptance_patterns: None,
        fingerprint: None,
    };
    let alone = trainer
        .model()
        .generate(&[request_a2], &config, &constants, &device)
        .unwrap();
    check_launches(&device).unwrap();
    assert_outcomes_equal(&out_a[0], &alone[0], "control: repeat call");
}

#[test]
fn uniform_sampler_baseline_runs() {
    let _lock = serial();
    let device = dev();
    let (_, set, _, _) = train_overfit(&device);
    let patterns = draw_zero_patterns(&set);
    let refs: Vec<&[MolGraph]> = patterns.iter().map(Vec::as_slice).collect();
    let requests: Vec<CompletionRequest> = set
        .examples
        .iter()
        .map(|example| CompletionRequest {
            id: example.identity_group,
            composition: example.composition,
            patterns: refs[example.identity_group as usize],
            acceptance_patterns: None,
            fingerprint: None,
        })
        .collect();
    // An untrained model with the same config (fresh seed): the control
    // arm's code path under the same acceptance rules.
    let untrained: CompletionTrainer<R, E> = CompletionTrainer::new(
        &CompletionModelConfig::small(),
        &CompletionTrainConfig {
            seed: 777,
            ..train_config()
        },
        &device,
    )
    .unwrap();
    let config = gen_config(64, 608);
    let constants = Ms2Constants::new(&device);
    let outcomes = untrained
        .model()
        .generate(&requests, &config, &constants, &device)
        .unwrap();
    check_launches(&device).unwrap();
    check_acceptance_invariants(&outcomes, &requests, 25, 64);
    let (mut finished, mut dead_end, mut traj) = (0u32, 0u32, 0u32);
    for outcome in &outcomes {
        finished += outcome.finished;
        dead_end += outcome.dead_end;
        traj += outcome.trajectories;
    }
    println!(
        "untrained baseline: finished fraction {:.3}, dead-end fraction {:.3}",
        f64::from(finished) / f64::from(traj),
        f64::from(dead_end) / f64::from(traj)
    );
}

#[test]
fn infeasible_requests_return_zero_trajectories_with_reasons() {
    let _lock = serial();
    let device = dev();
    // No training needed: the feasibility pre-check runs before any device
    // work, so an untrained model suffices.
    let trainer: CompletionTrainer<R, E> =
        CompletionTrainer::new(&CompletionModelConfig::small(), &train_config(), &device)
            .unwrap();
    // Two hydroxyl patterns need two oxygens: C2H6O (one oxygen) is
    // infeasible under disjoint and complete, feasible under contained.
    let hydroxyl = || MolGraph::new(vec![9], vec![]).unwrap();
    let patterns = vec![hydroxyl(), hydroxyl()];
    let mut composition: Composition = [0; 10];
    composition[0] = 2;
    composition[1] = 6;
    composition[3] = 1;
    let request = |semantics| CompletionGenerationConfig {
        trajectories: 8,
        temperature: 1.0,
        seed: 0,
        returned: 25,
        containment_node_limit: 100_000,
        identity_work_limit: 100_000,
        condition_on_patterns: true,
        substructure_semantics: semantics,
    };
    let constants = Ms2Constants::new(&device);
    for semantics in [
        SubstructureSemantics::DisjointOccurrences,
        SubstructureSemantics::CompleteFunctionalGroups,
    ] {
        let requests = [CompletionRequest {
            id: 1,
            composition,
            patterns: &patterns,
            acceptance_patterns: None,
            fingerprint: None,
        }];
        let outcomes = trainer
            .model()
            .generate(&requests, &request(semantics), &constants, &device)
            .unwrap();
        assert_eq!(outcomes.len(), 1);
        let outcome = &outcomes[0];
        assert_eq!(outcome.trajectories, 0, "{semantics:?}: no trajectories");
        assert!(outcome.sampled.is_empty());
        assert!(outcome.candidates.is_empty());
        let reason = outcome.infeasible.as_ref().expect("a reason is kept");
        assert!(reason.contains('O'), "{semantics:?}: reason names oxygen: {reason}");
        assert_eq!(outcome.finished, 0);
        assert_eq!(outcome.pass_contained, 0);
        assert_eq!(outcome.pass_disjoint, 0);
        assert_eq!(outcome.pass_complete, 0);
        // Scoring keeps the query as a miss, never a hit.
        let score = score_query(outcome, &ethanol(), 100_000);
        assert!(score.rank.is_none());
        assert_eq!(score.outcome_counts.trajectories, 0);
    }
    // Under contained the same query runs the full K trajectories.
    let requests = [CompletionRequest {
        id: 1,
        composition,
        patterns: &patterns,
        acceptance_patterns: None,
        fingerprint: None,
    }];
    let outcomes = trainer
        .model()
        .generate(
            &requests,
            &request(SubstructureSemantics::Contained),
            &constants,
            &device,
        )
        .unwrap();
    assert_eq!(outcomes[0].trajectories, 8);
    assert!(outcomes[0].infeasible.is_none());
}

#[test]
fn score_and_report() {
    let _lock = serial();
    // Hand-made outcomes against the ethanol target. Candidates carry
    // their canonical trace replayed without a budget (open bond-order
    // isomers have no complete exact replay, which is fine: scoring only
    // reads the graphs).
    let target = ethanol();
    let trace_of = |graph: &MolGraph| {
        canonical_trace(graph, limits(), CANONICAL_WORK_LIMIT)
            .unwrap()
            .trace
    };
    let outcome_with = |graphs: Vec<(MolGraph, u32, f32)>| QueryOutcome {
        candidates: graphs
            .into_iter()
            .map(|(graph, samples, lp)| {
                let trace = trace_of(&graph);
                let replayed = replay(&trace, limits(), None).unwrap().graph().unwrap();
                CompletionCandidate {
                    graph: replayed,
                    trace,
                    samples,
                    best_log_prob: lp,
                }
            })
            .collect(),
        unresolved: Vec::new(),
        sampled: Vec::new(),
        distinct: 0,
        trajectories: 8,
        finished: 5,
        dead_end: 2,
        truncated: 1,
        rejected_replay: 0,
        rejected_containment: 0,
        containment_unresolved: 0,
        rejected_extra_groups: 0,
        rejected_missing_groups: 0,
        pass_contained: 0,
        pass_disjoint: 0,
        pass_complete: 0,
        identity_unresolved: 0,
        other_status: 0,
        infeasible: None,
    };
    let first = outcome_with(vec![(ethanol(), 4, -1.0), (dimethyl_ether(), 2, -3.0)]);
    let third = outcome_with(vec![
        (dimethyl_ether(), 4, -1.0),
        (propan_1_ol(), 2, -2.0),
        (ethanol(), 1, -3.0),
    ]);
    let absent = outcome_with(vec![(dimethyl_ether(), 4, -1.0), (propan_1_ol(), 2, -2.0)]);
    let empty = QueryOutcome {
        candidates: Vec::new(),
        unresolved: Vec::new(),
        sampled: Vec::new(),
        distinct: 0,
        trajectories: 8,
        finished: 0,
        dead_end: 0,
        truncated: 8,
        rejected_replay: 0,
        rejected_containment: 0,
        containment_unresolved: 0,
        rejected_extra_groups: 0,
        rejected_missing_groups: 0,
        pass_contained: 0,
        pass_disjoint: 0,
        pass_complete: 0,
        identity_unresolved: 0,
        other_status: 0,
        infeasible: None,
    };
    let score_first = score_query(&first, &target, 100_000);
    let score_third = score_query(&third, &target, 100_000);
    let score_absent = score_query(&absent, &target, 100_000);
    let score_empty = score_query(&empty, &target, 100_000);
    assert_eq!(score_first.rank, Some(1), "target first");
    assert_eq!(score_third.rank, Some(3), "target third");
    assert_eq!(score_absent.rank, None, "target absent");
    assert_eq!(score_empty.rank, None, "empty shortlist is a miss");
    // Bond-order isomer: strict miss, skeleton hit. (The shifted Kekulé
    // form of a methylbenzene ring is isomorphic to it by ring reversal,
    // so the ethene single/double pair stands in as the documented case
    // where the skeleton merges bond-order isomers.)
    assert_eq!(
        same_identity(&ethene_single_bond(), &ethene(), 100_000),
        Some(false),
        "the bond-order isomers are strictly non-identical"
    );
    let isomer = outcome_with(vec![(ethene_single_bond(), 3, -1.5)]);
    let score_isomer = score_query(&isomer, &ethene(), 100_000);
    assert_eq!(score_isomer.rank, None, "strict rank misses the isomer");
    assert_eq!(
        score_isomer.skeleton_rank,
        Some(1),
        "skeleton rank catches the bond-order isomer"
    );
    assert_eq!(skeleton(&ethanol()).unwrap().bonds().len(), 2);
    // The report over [first, third, absent, empty]: denominators include
    // the empty outcome.
    let scores = vec![score_first, score_third, score_absent, score_empty];
    let groups = vec![21u64, 21, 22, 23];
    let report = recovery_report(&scores, &groups, 200, 7);
    assert_eq!(report.queries, 4);
    assert_eq!((report.top1.hits, report.top1.queries), (1, 4));
    assert_eq!((report.top10.hits, report.top10.queries), (2, 4));
    assert_eq!((report.top25.hits, report.top25.queries), (2, 4));
    for (name, rate) in [
        ("top1", &report.top1),
        ("top10", &report.top10),
        ("top25", &report.top25),
        ("skeleton_top25", &report.skeleton_top25),
    ] {
        assert!(
            rate.lo <= rate.rate && rate.rate <= rate.hi,
            "{name}: the interval contains the point estimate"
        );
    }
    let again = recovery_report(&scores, &groups, 200, 7);
    assert_eq!(report, again, "the report is deterministic in the seed");
    let other_seed = recovery_report(&scores, &groups, 200, 8);
    println!(
        "seed 7 top25 [{}, {}], seed 8 top25 [{}, {}]",
        report.top25.lo, report.top25.hi, other_seed.top25.lo, other_seed.top25.hi
    );
    // Group resampling keeps a group's queries together: two queries of one
    // group collapse every interval to its point estimate.
    let pair = vec![
        score_query(&first, &target, 100_000),
        score_query(&absent, &target, 100_000),
    ];
    let collapsed = recovery_report(&pair, &[5u64, 5], 200, 7);
    for (name, rate) in [
        ("top1", &collapsed.top1),
        ("top10", &collapsed.top10),
        ("top25", &collapsed.top25),
        ("skeleton_top25", &collapsed.skeleton_top25),
    ] {
        assert_eq!(
            (rate.lo, rate.hi),
            (rate.rate, rate.rate),
            "{name}: one group collapses the interval to the point"
        );
    }
}

#[test]
fn ranks_beyond_the_cut_are_misses() {
    let _lock = serial();
    let counts = || OutcomeCounts {
        candidates: 0,
        distinct: 0,
        trajectories: 8,
        finished: 8,
        dead_end: 0,
        truncated: 0,
        rejected_replay: 0,
        rejected_containment: 0,
        containment_unresolved: 0,
        identity_unresolved: 0,
        other_status: 0,
    };
    let score = |rank: Option<u32>, skeleton_rank: Option<u32>| QueryScore {
        rank,
        skeleton_rank,
        outcome_counts: counts(),
    };
    let scores = vec![
        score(Some(1), Some(1)),
        score(Some(10), Some(10)),
        score(Some(11), Some(11)),
        score(Some(25), Some(25)),
        score(Some(26), Some(26)),
        score(None, None),
    ];
    let groups = vec![1u64, 2, 3, 4, 5, 6];
    let report = recovery_report(&scores, &groups, 0, 7);
    assert_eq!(report.queries, 6);
    assert_eq!(report.top1.hits, 1, "only rank 1 is top-1");
    assert_eq!(report.top10.hits, 2, "ranks 1 and 10 are top-10, not 11");
    assert_eq!(
        report.top25.hits, 4,
        "ranks 1, 10, 11 and 25 are top-25, not 26"
    );
    assert_eq!(
        report.skeleton_top25.hits, 4,
        "skeleton rank 26 is not skeleton-top-25"
    );
}

#[test]
fn bootstrap_matches_an_independent_oracle() {
    let _lock = serial();
    let counts = |unresolved: u32| OutcomeCounts {
        candidates: 0,
        distinct: 0,
        trajectories: 8,
        finished: 8,
        dead_end: 0,
        truncated: 0,
        rejected_replay: 0,
        rejected_containment: 0,
        containment_unresolved: 0,
        identity_unresolved: unresolved,
        other_status: 0,
    };
    // Unequal groups (sizes 1, 2 and 4) with mixed hits.
    let ranks = [Some(1), Some(3), None, Some(1), None, Some(26), Some(2)];
    let groups = vec![10u64, 20, 20, 30, 30, 30, 30];
    let scores: Vec<QueryScore> = ranks
        .iter()
        .map(|rank| QueryScore {
            rank: *rank,
            skeleton_rank: *rank,
            outcome_counts: counts(0),
        })
        .collect();
    let report = recovery_report(&scores, &groups, 200, 7);
    // The independent oracle: its own SplitMix64 with rejection sampling,
    // the same group pooling and the same linear-interpolation percentiles.
    struct Oracle {
        state: u64,
    }
    impl Oracle {
        fn next(&mut self) -> u64 {
            self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn below(&mut self, bound: u64) -> u64 {
            let accept = (u64::MAX / bound) * bound;
            loop {
                let x = self.next();
                if x < accept {
                    return x % bound;
                }
            }
        }
    }
    fn percentile(sorted: &[f64], p: f64) -> f64 {
        if sorted.len() == 1 {
            return sorted[0];
        }
        let rank = (sorted.len() - 1) as f64 * p / 100.0;
        let lo = rank.floor() as usize;
        let hi = rank.ceil() as usize;
        if lo == hi {
            sorted[lo]
        } else {
            sorted[lo] + (sorted[hi] - sorted[lo]) * (rank - lo as f64)
        }
    }
    let hit = |rank: &Option<u32>, which: u8| match which {
        0 => *rank == Some(1),
        1 => rank.is_some_and(|r| r <= 10),
        2 => rank.is_some_and(|r| r <= 25),
        _ => rank.is_some_and(|r| r <= 25),
    };
    let members: Vec<Vec<usize>> = vec![vec![0], vec![1, 2], vec![3, 4, 5, 6]];
    let mut rng = Oracle { state: 7 };
    let mut replicates: [Vec<f64>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    for _ in 0..200 {
        let mut pooled = Vec::new();
        for _ in 0..3 {
            pooled.extend_from_slice(&members[rng.below(3) as usize]);
        }
        for (which, rates) in replicates.iter_mut().enumerate() {
            let hits = pooled
                .iter()
                .filter(|i| hit(&ranks[**i], which as u8))
                .count();
            rates.push(hits as f64 / pooled.len() as f64);
        }
    }
    for rates in replicates.iter_mut() {
        rates.sort_by(|a, b| a.total_cmp(b));
    }
    let point =
        |which: u8| ranks.iter().filter(|r| hit(r, which)).count() as f64 / ranks.len() as f64;
    let rates = [
        &report.top1,
        &report.top10,
        &report.top25,
        &report.skeleton_top25,
    ];
    for (which, rate) in rates.iter().enumerate() {
        assert_eq!(rate.rate, point(which as u8), "point rate {which}");
        assert_eq!(rate.lo, percentile(&replicates[which], 2.5), "lo {which}");
        assert_eq!(rate.hi, percentile(&replicates[which], 97.5), "hi {which}");
    }
    // `bootstrap = 0` reports the point as both bounds, and an empty score
    // list is a defined zero report: neither panics.
    let point_report = recovery_report(&scores, &groups, 0, 7);
    for rate in [
        &point_report.top1,
        &point_report.top10,
        &point_report.top25,
        &point_report.skeleton_top25,
    ] {
        assert_eq!((rate.lo, rate.hi), (rate.rate, rate.rate));
    }
    let empty = recovery_report(&[], &[], 50, 7);
    assert_eq!(empty.queries, 0);
    assert_eq!((empty.top1.hits, empty.top1.queries), (0, 0));
    assert_eq!(empty.top25.rate, 0.0);
    assert_eq!(empty.mean_distinct, 0.0);
    assert_eq!(empty.mean_unresolved, 0.0);
}

#[test]
fn unresolved_is_never_a_hit() {
    let _lock = serial();
    let target = ethanol();
    let trace_of = |graph: &MolGraph| {
        canonical_trace(graph, limits(), CANONICAL_WORK_LIMIT)
            .unwrap()
            .trace
    };
    let entry = |graph: &MolGraph, samples: u32| {
        let trace = trace_of(graph);
        let graph = replay(&trace, limits(), None).unwrap().graph().unwrap();
        CompletionCandidate {
            graph,
            trace,
            samples,
            best_log_prob: -1.0,
        }
    };
    let outcome_of = |candidates: Vec<CompletionCandidate>,
                      unresolved: Vec<CompletionCandidate>,
                      waiting: u32| {
        QueryOutcome {
            candidates,
            unresolved,
            sampled: Vec::new(),
            distinct: 1,
            trajectories: 8,
            finished: 5,
            dead_end: 2,
            truncated: 1,
            rejected_replay: 0,
            rejected_containment: 0,
            containment_unresolved: 0,
            rejected_extra_groups: 0,
            rejected_missing_groups: 0,
            pass_contained: 0,
            pass_disjoint: 0,
            pass_complete: 0,
            identity_unresolved: waiting,
            other_status: 0,
            infeasible: None,
        }
    };
    // The shortlist misses the target while the unresolved shortlist holds
    // it: neither rank hits.
    let miss = outcome_of(
        vec![entry(&dimethyl_ether(), 3)],
        vec![entry(&target, 2)],
        2,
    );
    let score = score_query(&miss, &target, 100_000);
    assert_eq!(score.rank, None, "an unresolved entry is never a hit");
    assert_eq!(
        score.skeleton_rank, None,
        "an unresolved entry is never a skeleton hit either"
    );
    // When the shortlist holds the target its rank ignores the unresolved
    // entries.
    let hit = outcome_of(
        vec![entry(&dimethyl_ether(), 3), entry(&target, 1)],
        vec![entry(&target, 1)],
        1,
    );
    let score_hit = score_query(&hit, &target, 100_000);
    assert_eq!(score_hit.rank, Some(2));
    // The report counts the unresolved trajectories next to the diagnostics.
    let report = recovery_report(&[score, score_hit], &[1u64, 2], 0, 7);
    assert_eq!(report.top25.hits, 1);
    assert_eq!(report.mean_unresolved, 1.5);
}

#[test]
fn generate_reads_the_device_once() {
    let _lock = serial();
    let device = dev();
    let (trainer, set, _, _) = train_overfit(&device);
    let patterns = draw_zero_patterns(&set);
    let refs: Vec<&[MolGraph]> = patterns.iter().map(Vec::as_slice).collect();
    let constants = Ms2Constants::new(&device);
    let one = |seed: u64, k: u32| {
        let request = CompletionRequest {
            id: set.examples[0].identity_group,
            composition: set.examples[0].composition,
            patterns: refs[0],
            acceptance_patterns: None,
            fingerprint: None,
        };
        let config = gen_config(k, seed);
        trainer
            .model()
            .generate(&[request], &config, &constants, &device)
            .unwrap()
    };
    // Warm up (settles matmul routing and the allocator) before measuring.
    for seed in [701u64, 702] {
        one(seed, 32);
        check_launches(&device).unwrap();
    }
    // Exactly one device read per generate call: the minimum over 3 calls
    // (a concurrent test can only add reads, never remove this thread's).
    let mut deltas = Vec::new();
    for seed in [703u64, 704, 705] {
        let r0 = runtime_read_count();
        one(seed, 32);
        check_launches(&device).unwrap();
        deltas.push(runtime_read_count() - r0);
    }
    println!("generate read deltas: {deltas:?}");
    // One final read plus at most one early-exit poll every 8 steps.
    let min_reads = deltas.into_iter().min().unwrap();
    let steps_total = limits().max_steps();
    assert!(
        (1..=1 + steps_total / 8).contains(&min_reads),
        "a generate call reads the device once at the end plus at most one poll per 8 steps, got {min_reads}"
    );
    // The launch count per step does not depend on K: warmed totals at K =
    // 8 and K = 64 agree within 2 launches per step (the tolerance of
    // `step_logits_no_read_and_row_independent_launches`).
    let steps_done = limits().max_steps() - 1;
    for seed in [706u64, 707] {
        for k in [8u32, 64] {
            one(seed, k);
            check_launches(&device).unwrap();
        }
    }
    let mut totals = Vec::new();
    for k in [8u32, 64] {
        device.synchronize();
        reset_launch_count();
        one(708, k);
        check_launches(&device).unwrap();
        device.synchronize();
        totals.push((k, launch_count()));
    }
    println!("generate launch totals (K, launches): {totals:?}");
    let diff = (totals[0].1 as i64 - totals[1].1 as i64).unsigned_abs() as usize;
    let per_step = diff as f64 / steps_done as f64;
    println!("launch diff {diff} over {steps_done} steps = {per_step:.3} per step");
    assert!(
        diff <= 2 * steps_done,
        "launch count per step is K-independent within 2 (K=8: {}, K=64: {})",
        totals[0].1,
        totals[1].1
    );
}
