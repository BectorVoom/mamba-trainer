//! Beam-search primitives for the completion model: re-ordering the per-row
//! generation state, reading one step's masked action distribution, and
//! committing a caller-chosen action.
//!
//! The search itself is not here — these are the three pieces it stands on:
//! [`DecoderState::gather`](mamba3::models::ms2::decoder::DecoderState::gather)
//! plus [`gather_generation_rows`] for the rows,
//! [`composed_step_log_probs`] for the distribution and
//! [`apply_chosen_action`] for the commit.
//!
//! Hand-built molecules only (atom type ids from
//! [`chem::ATOM_TYPES`](mamba3::models::ms2::chem::ATOM_TYPES)), as in
//! `ms2_completion_spectrum.rs`: ethane (every step of its exact-completion
//! trace is forced), ethanol and kekulized methylbenzene. Every device call
//! is followed by [`check_launches`].

#![cfg(feature = "backend")]

use mamba3::autograd::{Var, no_grad};
use mamba3::backend::{Device, check_launches, runtime_read_count};
use mamba3::backends::Auto;
use mamba3::models::ms2::chem::Composition;
use mamba3::models::ms2::completion_data::{ExtractionConfig, PatternSource, same_identity};
use mamba3::models::ms2::completion_model::{
    BeamAudit, CompletionGenerationConfig, CompletionModel, CompletionModelConfig,
    CompletionRequest, CompletionTrainConfig, CompletionTrainer, PatternBatch,
    SubstructureSemantics,
};
use mamba3::models::ms2::completion_spectrum::SpectrumEvidence;
use mamba3::models::ms2::decoder::DecoderState;
use mamba3::models::ms2::encoder::EncoderOutput;
use mamba3::models::ms2::generate::{
    StepLogProbs, apply_chosen_action, composed_decode_step, composed_step_heads,
    composed_step_log_probs, gather_generation_rows, step_field_log_probs,
};
use mamba3::models::ms2::grammar::{
    CANONICAL_WORK_LIMIT, Limits, Token, TraceState, canonical_trace, replay_exact,
};
use mamba3::models::ms2::graph::MolGraph;
use mamba3::models::ms2::targets_batch::TargetBatch;
use mamba3::models::ms2::twin;
use mamba3::tensor::Tensor;
use mamba3::tensor::ops::index::IdTensor;
use mamba3::tensor::ops::ms2::{self, Ms2Constants};
use mamba3::tensor::ops::random::Rng;

type R = Auto;
type E = f32;

static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn serial() -> std::sync::MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

fn dev() -> Device<R> {
    Device::<R>::default()
}

/// The model config every test here uses: `small()`, so 16 atom slots, 4 ring
/// closures and `T = 22` steps.
fn config() -> CompletionModelConfig {
    CompletionModelConfig::small()
}

fn limits() -> Limits {
    Limits::new(16, 4).expect("16 atoms and 4 closures are in range")
}

/// Ethane `C2H6`: two carbons with three hydrogens each, one single bond.
/// Every step of its exact-completion trace has exactly one legal action,
/// which `forced_trace_is_the_only_trace` checks and relies on.
fn ethane() -> MolGraph {
    MolGraph::new(vec![4, 4], vec![(0, 1, 1)]).expect("ethane is a valid graph")
}

/// Ethanol `C2H6O`.
fn ethanol() -> MolGraph {
    MolGraph::new(vec![4, 3, 9], vec![(0, 1, 1), (1, 2, 1)]).expect("ethanol is a valid graph")
}

/// Kekulized methylbenzene `C7H8`: 7 atoms and a ring closure, so its trace
/// is long and its steps have many legal actions.
fn toluene() -> MolGraph {
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
    .expect("methylbenzene is a valid graph")
}

/// The canonical exact-completion trace of a molecule, with its composition.
fn trace_and_composition(graph: &MolGraph) -> (Vec<Token>, Composition) {
    let canonical = canonical_trace(graph, limits(), CANONICAL_WORK_LIMIT).expect("a trace");
    let composition = graph.composition();
    let end = replay_exact(&canonical.trace, limits(), composition).expect("the trace replays");
    assert!(end.stopped() && end.is_complete());
    (canonical.trace, composition)
}

/// One generation loop, held exactly as `CompletionModel::generate` holds it:
/// the encoder output, the per-trajectory composition embedding, the four
/// per-row trajectory buffers, the decoder state and the packed-logits row.
struct Loop {
    encoded: EncoderOutput<R, E>,
    traj_formula: Var<R, E>,
    actions: IdTensor<R>,
    replay: IdTensor<R>,
    step_token: IdTensor<R>,
    traj_meta: IdTensor<R>,
    state: DecoderState<R, E>,
    logits: Tensor<R, E>,
    rows: usize,
    k: usize,
    steps: usize,
    atoms: usize,
    closures: u32,
}

/// Start a loop over `compositions` with `k` trajectories per query, the same
/// way `CompletionModel::generate` starts one: patterns off, the composition
/// embedding expanded to one row per trajectory, and the host-built initial
/// rows of `twin::init_completion_trajectories` (START applied, started word
/// 2).
fn start(
    model: &CompletionModel<R, E>,
    compositions: &[Composition],
    k: usize,
    ids: &[u64],
    device: &Device<R>,
) -> Loop {
    assert_eq!(ids.len(), compositions.len());
    let b = compositions.len();
    let rows = b * k;
    let atoms = 16usize;
    let closures = 4u32;
    let steps = limits().max_steps();
    let d = 64usize;
    let empty: Vec<MolGraph> = Vec::new();
    let pattern_refs: Vec<&[MolGraph]> = compositions.iter().map(|_| empty.as_slice()).collect();
    let batch = PatternBatch::build(&pattern_refs, compositions).expect("a pattern batch");
    let encoded = model.encode(&batch, device).expect("the encoder runs");
    check_launches(device).unwrap();
    let traj_formula = encoded
        .context
        .clone()
        .reshape(vec![b, 1, d])
        .unwrap()
        .expand(vec![b, k, d])
        .unwrap()
        .reshape(vec![rows, d])
        .unwrap();
    check_launches(device).unwrap();
    let (traj_meta_host, state_host, actions_host) =
        twin::init_completion_trajectories(ids, compositions, k, steps, atoms);
    let traj_meta =
        IdTensor::from_slice(&traj_meta_host, vec![rows, ms2::TRAJ_META_WIDTH], device).unwrap();
    let replay = IdTensor::from_slice(
        &state_host,
        vec![rows, ms2::replay_state_width(atoms)],
        device,
    )
    .unwrap();
    let actions = IdTensor::from_slice(
        &actions_host,
        vec![rows, ms2::sample_record_width(steps, atoms)],
        device,
    )
    .unwrap();
    let state = model
        .decoder()
        .start_state(&encoded, rows, device)
        .expect("a composed decoder state");
    check_launches(device).unwrap();
    Loop {
        encoded,
        traj_formula,
        actions,
        replay,
        step_token: IdTensor::empty(vec![rows, 4], device),
        traj_meta,
        state,
        logits: Tensor::<R, E>::empty(vec![rows, ms2::sample_logits_width(atoms)], device),
        rows,
        k,
        steps,
        atoms,
        closures,
    }
}

/// One sampling step of the loop: the shared prologue and
/// `composed_decode_step`, exactly as `CompletionModel::generate` runs them.
fn sample_step(
    run: &mut Loop,
    model: &CompletionModel<R, E>,
    constants: &Ms2Constants<R>,
    step: usize,
    temperature: f32,
    seed_lo: u32,
    seed_hi: u32,
    device: &Device<R>,
) {
    ms2::step_token(&run.actions, &mut run.step_token, run.steps, run.atoms).unwrap();
    check_launches(device).unwrap();
    composed_decode_step(
        model.decoder(),
        &run.encoded,
        &run.traj_formula,
        &mut run.actions,
        &mut run.step_token,
        &mut run.replay,
        &mut run.logits,
        &run.traj_meta,
        &mut run.state,
        &model.decoder().bond_by_type_value(),
        &constants.atom_table,
        step,
        seed_lo,
        seed_hi,
        temperature,
        run.steps,
        run.atoms,
        run.closures,
        run.k,
        run.rows,
    )
    .unwrap();
    check_launches(device).unwrap();
}

/// Run sampling steps `from..to` of the loop.
#[allow(clippy::too_many_arguments)]
fn sample_steps(
    run: &mut Loop,
    model: &CompletionModel<R, E>,
    constants: &Ms2Constants<R>,
    from: usize,
    to: usize,
    temperature: f32,
    seed: u64,
    device: &Device<R>,
) {
    let seed_lo = seed as u32;
    let seed_hi = (seed >> 32) as u32;
    for step in from..to {
        sample_step(
            run,
            model,
            constants,
            step,
            temperature,
            seed_lo,
            seed_hi,
            device,
        );
    }
}

/// Re-order every per-row buffer of the loop by `parents`, with one draw key
/// per child: `DecoderState::gather` for the recurrent state and
/// `gather_generation_rows` for the trajectory buffers.
fn gather(run: &mut Loop, parents: &[u32], child_keys: &[u64], device: &Device<R>) {
    let ids = IdTensor::from_slice(parents, vec![parents.len()], device).unwrap();
    let rows = parents.len();
    let state = run.state.gather(&ids, device).expect("the state gathers");
    check_launches(device).unwrap();
    let gathered = gather_generation_rows(
        &run.actions,
        &run.replay,
        &run.step_token,
        &run.traj_meta,
        &ids,
        child_keys,
        run.steps,
        run.atoms,
        device,
    )
    .expect("the trajectory buffers gather");
    check_launches(device).unwrap();
    run.state = state;
    run.actions = gathered.actions;
    run.replay = gathered.replay;
    run.step_token = gathered.step_token;
    run.traj_meta = gathered.traj_meta;
    run.rows = rows;
    run.logits = Tensor::<R, E>::empty(vec![rows, ms2::sample_logits_width(run.atoms)], device);
}

/// The `[rows, record]` action records as host words.
fn records(run: &Loop) -> Vec<u32> {
    run.actions.try_to_vec().expect("the record reads back")
}

/// The trace words of row `r` of a record buffer (`length * 4` words).
fn trace_words(record: &[u32], steps: usize, atoms: usize, r: usize) -> Vec<u32> {
    let width = ms2::sample_record_width(steps, atoms);
    let base = r * width;
    let length = record[base + steps * 4 + atoms] as usize;
    record[base..base + length * 4].to_vec()
}

/// The four words of a token, as a record holds them.
fn token_words(token: &Token) -> [u32; 4] {
    [
        u32::from(token.kind),
        u32::from(token.atom_type),
        u32::from(token.bond),
        u32::from(token.pointer),
    ]
}

/// The trace words a whole trace should land in a record.
fn expected_trace_words(trace: &[Token]) -> Vec<u32> {
    trace.iter().flat_map(token_words).collect()
}

/// A model with the small config, drawn from `seed`.
fn model_of(seed: u64, device: &Device<R>) -> CompletionModel<R, E> {
    let mut rng = Rng::seeded(seed);
    let model = CompletionModel::init(&config(), device, &mut rng).expect("the model builds");
    check_launches(device).unwrap();
    model
}

#[test]
fn mass_search_preserves_per_formula_beam_counters() {
    use mamba3::models::ms2::chem::composition_mass;
    use mamba3::models::ms2::completion_formula::{
        CompletionSearch, FormulaAllocation, FormulaPruning, MassQuery,
        run_mass_completion_search,
    };
    use mamba3::models::ms2::completion_model::FormulaArtifacts;
    let _lock = serial();
    let device = dev();
    let model = model_of(23, &device);
    let constants = Ms2Constants::new(&device);
    let composition = ethane().composition();
    let artifacts = FormulaArtifacts::fit(&[composition], 16, 0, 0, "beam-stats".into()).unwrap();
    let mass = MassQuery::Neutral {
        value: composition_mass(&composition).unwrap(),
        ppm_tenths: 100,
        uncertainty: Some(1),
    };
    let result = run_mass_completion_search(
        &model,
        &constants,
        &device,
        &artifacts,
        16,
        4,
        &[],
        None,
        &mass,
        1,
        100_000,
        4,
        1.0,
        7,
        16,
        "beam-stats",
        true,
        FormulaPruning::TrainFit,
        FormulaAllocation::Equal,
        SubstructureSemantics::Contained,
        None,
        None,
        CompletionSearch::Beam,
    )
    .unwrap();
    check_launches(&device).unwrap();
    assert_eq!(result.formula_search.formulas.len(), 1);
    let entry = &result.formula_search.formulas[0];
    let stats = entry
        .beam_stats
        .as_ref()
        .expect("beam diagnostics retained");
    let request = CompletionRequest {
        id: 1,
        composition,
        patterns: &[],
        acceptance_patterns: None,
        fingerprint: None,
    };
    let config = CompletionGenerationConfig {
        trajectories: 4,
        returned: 16,
        ..Default::default()
    };
    let (_, direct) = model
        .generate_beam(&[request], &config, 4, &constants, &device)
        .unwrap();
    check_launches(&device).unwrap();
    assert_eq!(
        stats, &direct[0],
        "formula grouping must preserve the actual counters"
    );
    assert_eq!(stats.finished, entry.finished);
    assert_eq!(stats.row_steps, result.beam_row_steps);
    assert_eq!(stats.candidates_dropped, result.beam_candidates_dropped);
    assert!(stats.row_steps > 0);
    let sampling = run_mass_completion_search(
        &model,
        &constants,
        &device,
        &artifacts,
        16,
        4,
        &[],
        None,
        &mass,
        1,
        100_000,
        4,
        1.0,
        7,
        16,
        "sampling-stats",
        true,
        FormulaPruning::TrainFit,
        FormulaAllocation::Equal,
        SubstructureSemantics::Contained,
        None,
        None,
        CompletionSearch::Sampling,
    )
    .unwrap();
    check_launches(&device).unwrap();
    assert!(
        sampling
            .formula_search
            .formulas
            .iter()
            .all(|f| f.beam_stats.is_none())
    );
}

#[test]
fn identity_gather_changes_nothing() {
    let _lock = serial();
    let _guard = no_grad();
    let device = dev();
    let model = model_of(7, &device);
    let constants = Ms2Constants::new(&device);
    let (_, composition) = trace_and_composition(&toluene());
    let k = 4usize;
    let id = 0x5151_2727_abcd_0001u64;
    let seed = 12345u64;
    let steps = limits().max_steps();

    // One run straight through, one that gathers with the identity after
    // three steps and then continues. The child key of an identity gather is
    // the parent's request id: the key words are derived from it exactly as
    // the request id's are, so the row keeps its own stream.
    let mut plain = start(&model, &[composition], k, &[id], &device);
    sample_steps(&mut plain, &model, &constants, 1, steps, 1.0, seed, &device);
    let want = records(&plain);

    let mut gathered = start(&model, &[composition], k, &[id], &device);
    sample_steps(&mut gathered, &model, &constants, 1, 4, 1.0, seed, &device);
    let parents: Vec<u32> = (0..k as u32).collect();
    gather(&mut gathered, &parents, &vec![id; k], &device);
    sample_steps(
        &mut gathered,
        &model,
        &constants,
        4,
        steps,
        1.0,
        seed,
        &device,
    );
    let got = records(&gathered);
    assert_eq!(got.len(), want.len());
    assert_eq!(
        got, want,
        "an identity gather must leave every action record bit-identical"
    );
    // The run did produce traces, so the comparison is not vacuous.
    let lengths: Vec<u32> = (0..k)
        .map(|r| want[r * ms2::sample_record_width(steps, 16) + steps * 4 + 16])
        .collect();
    assert!(
        lengths.iter().all(|&l| l > 2),
        "the sampled rows are too short to test anything: {lengths:?}"
    );
    println!("identity gather: record lengths {lengths:?}");
}

#[test]
fn a_duplicated_parent_diverges_only_through_the_key() {
    let _lock = serial();
    let _guard = no_grad();
    let device = dev();
    let model = model_of(11, &device);
    let constants = Ms2Constants::new(&device);
    let (_, composition) = trace_and_composition(&toluene());
    let k = 2usize;
    let id = 0x1234_5678_9abc_def0u64;
    let seed = 999u64;
    let steps = limits().max_steps();

    // Both children copy parent row 0. With the same fresh key they are the
    // same row and must step identically for ever; with different keys their
    // draws differ and the traces must part.
    let twin_run = |keys: [u64; 2]| -> Vec<u32> {
        let mut run = start(&model, &[composition], k, &[id], &device);
        sample_steps(&mut run, &model, &constants, 1, 3, 1.0, seed, &device);
        gather(&mut run, &[0, 0], &keys, &device);
        sample_steps(&mut run, &model, &constants, 3, steps, 1.0, seed, &device);
        records(&run)
    };

    let same = twin_run([id, id]);
    let width = ms2::sample_record_width(steps, 16);
    assert_eq!(
        same[0..width],
        same[width..2 * width],
        "two children of one parent with the same key must stay identical"
    );

    let differing = twin_run([id, id ^ 0x9e37_79b9_7f4a_7c15]);
    assert_ne!(
        differing[0..width],
        differing[width..2 * width],
        "two children of one parent with different keys must diverge"
    );
    // It is the trace that differs, not only a status word.
    let left = trace_words(&differing, steps, 16, 0);
    let right = trace_words(&differing, steps, 16, 1);
    assert_ne!(left, right, "the two children's traces must differ");
    println!(
        "duplicated parent: {} and {} trace words, first difference at {:?}",
        left.len(),
        right.len(),
        left.iter().zip(right.iter()).position(|(a, b)| a != b)
    );
}

#[test]
fn a_permuted_gather_permutes_the_outcome() {
    let _lock = serial();
    let _guard = no_grad();
    let device = dev();
    let model = model_of(13, &device);
    let constants = Ms2Constants::new(&device);
    let (_, composition) = trace_and_composition(&toluene());
    let k = 4usize;
    let id = 0x7777_0000_1111_2222u64;
    let seed = 4242u64;
    let steps = limits().max_steps();
    let width = ms2::sample_record_width(steps, 16);

    let mut plain = start(&model, &[composition], k, &[id], &device);
    sample_steps(&mut plain, &model, &constants, 1, steps, 1.0, seed, &device);
    let want = records(&plain);

    // The reversing permutation, with the request id as every child's key:
    // child `i` then carries parent `k - 1 - i`'s trajectory word *and* its
    // key, so its whole stream is that parent's.
    let mut run = start(&model, &[composition], k, &[id], &device);
    sample_steps(&mut run, &model, &constants, 1, 3, 1.0, seed, &device);
    let parents: Vec<u32> = (0..k as u32).rev().collect();
    gather(&mut run, &parents, &vec![id; k], &device);
    sample_steps(&mut run, &model, &constants, 3, steps, 1.0, seed, &device);
    let got = records(&run);
    for i in 0..k {
        let src = k - 1 - i;
        assert_eq!(
            got[i * width..(i + 1) * width],
            want[src * width..(src + 1) * width],
            "row {i} of the reversed run must be row {src} of the plain run"
        );
    }
    println!("permuted gather: {k} rows of {width} words matched in reverse order");
}

/// Teacher-force a trace with the beam primitives, returning per step the
/// four masked field log-probabilities at the target indices, the four
/// legality masks the device computed, and the four whole distributions.
struct ForcedStep {
    /// The four values at the target action's indices.
    field: [f32; 4],
    /// The device masks (kind, atom type, bond, pointer).
    masks: [u32; 4],
    /// The four whole rows.
    rows: [Vec<f32>; 4],
}

/// Drive one row along `trace` with `composed_step_log_probs` and
/// `apply_chosen_action`, collecting one [`ForcedStep`] per scored step
/// (`1..trace.len()`). The committed step log-probability is the summed
/// field value, so the record's accumulator is the sampler's.
fn force_trace(
    model: &CompletionModel<R, E>,
    constants: &Ms2Constants<R>,
    composition: Composition,
    trace: &[Token],
    id: u64,
    device: &Device<R>,
) -> (Loop, Vec<ForcedStep>) {
    let mut run = start(model, &[composition], 1, &[id], device);
    let mut out = Vec::new();
    let mut buffers = StepLogProbs::<R, E>::new(run.rows, run.atoms, device);
    let bond_table = model.decoder().bond_by_type_value();
    for step in 1..trace.len() {
        let token = trace[step];
        let words = token_words(&token);
        let context = IdTensor::from_slice(&words, vec![1, 4], device).unwrap();
        ms2::step_token(&run.actions, &mut run.step_token, run.steps, run.atoms).unwrap();
        check_launches(device).unwrap();
        composed_step_log_probs(
            model.decoder(),
            &run.encoded,
            &run.traj_formula,
            &run.step_token,
            &run.replay,
            &run.traj_meta,
            &context,
            &mut run.state,
            &bond_table,
            &constants.atom_table,
            &mut buffers,
            step,
            run.atoms,
            run.closures,
            run.k,
            run.rows,
        )
        .unwrap();
        check_launches(device).unwrap();
        let kind = buffers.kind.try_to_f32().unwrap();
        let atom_type = buffers.atom_type.try_to_f32().unwrap();
        let bond = buffers.bond.try_to_f32().unwrap();
        let pointer = buffers.pointer.try_to_f32().unwrap();
        let plan = buffers.plan.try_to_vec().unwrap();
        check_launches(device).unwrap();
        let field = [
            kind[words[0] as usize],
            atom_type[words[1] as usize],
            bond[words[2] as usize],
            pointer[words[3] as usize],
        ];
        out.push(ForcedStep {
            field,
            masks: [plan[0], plan[1], plan[2], plan[3]],
            rows: [kind, atom_type, bond, pointer],
        });
        // Commit the target action with its own joint log-probability.
        let joint: f32 = field.iter().sum();
        let lp = Tensor::<R, E>::from_f32(&[joint], vec![1], device).unwrap();
        apply_chosen_action(
            &context,
            &mut run.actions,
            &mut run.step_token,
            &mut run.replay,
            &run.traj_meta,
            &mut run.state,
            &lp,
            &constants.atom_table,
            step,
            run.steps,
            run.atoms,
            run.closures,
        )
        .unwrap();
        check_launches(device).unwrap();
    }
    (run, out)
}

#[test]
fn the_returned_distribution_agrees_with_teacher_forcing() {
    let _lock = serial();
    let _guard = no_grad();
    let device = dev();
    let model = model_of(17, &device);
    let constants = Ms2Constants::new(&device);
    // The tolerance: the stepped recurrence and the parallel teacher pass
    // are the same arithmetic in a different order, compared at 1e-4 — the
    // tolerance `ms2_decoder.rs::stepped_parity_with_teacher` already uses
    // for exactly this comparison.
    let tolerance = 1e-4f32;
    let mut worst = 0.0f32;
    for graph in [ethane(), ethanol(), toluene()] {
        let (trace, composition) = trace_and_composition(&graph);
        // The teacher pass over the same trace, with the same encoder.
        let empty: Vec<MolGraph> = Vec::new();
        let batch = PatternBatch::build(&[empty.as_slice()], &[composition]).unwrap();
        let targets =
            TargetBatch::build_exact(&[trace.as_slice()], &[composition], limits()).unwrap();
        let teacher = model
            .teacher(&batch, &targets, &constants, &device)
            .unwrap()
            .0;
        check_launches(&device).unwrap();
        let fields = teacher.field_log_prob.try_to_f32().unwrap();
        check_launches(&device).unwrap();

        let (_, forced) = force_trace(
            &model,
            &constants,
            composition,
            &trace,
            0x0bea_0000_0000_0001,
            &device,
        );
        assert_eq!(forced.len(), trace.len() - 1);
        for (i, got) in forced.iter().enumerate() {
            // Teacher output position `i` scores token `i + 1`, which is the
            // token step `i + 1` of the loop commits.
            let want: f32 = fields[i * 4..i * 4 + 4].iter().sum();
            let sum: f32 = got.field.iter().sum();
            let delta = (sum - want).abs();
            worst = worst.max(delta);
            assert!(
                delta <= tolerance,
                "{} atoms, step {}: beam {sum} vs teacher {want} (fields {:?} vs {:?})",
                graph.atoms().len(),
                i + 1,
                got.field,
                &fields[i * 4..i * 4 + 4]
            );
            // Field by field too, so a compensating pair of errors cannot
            // hide inside the sum.
            for f in 0..4 {
                let delta = (got.field[f] - fields[i * 4 + f]).abs();
                worst = worst.max(delta);
                assert!(
                    delta <= tolerance,
                    "{} atoms, step {}, field {f}: beam {} vs teacher {}",
                    graph.atoms().len(),
                    i + 1,
                    got.field[f],
                    fields[i * 4 + f]
                );
            }
        }
    }
    println!("teacher agreement: worst deviation {worst:e} (tolerance {tolerance:e})");
}

#[test]
fn masked_actions_are_impossible_and_the_legal_set_normalises() {
    let _lock = serial();
    let _guard = no_grad();
    let device = dev();
    let model = model_of(19, &device);
    let constants = Ms2Constants::new(&device);
    // Any log-probability at or below this floor is "impossible": no backend
    // spells `-inf`, so the kernels write `f32::MIN` where `mask_logits`
    // writes it.
    let floor = -1e30f32;
    let mut checked = 0usize;
    for graph in [ethanol(), toluene()] {
        let (trace, composition) = trace_and_composition(&graph);
        let (_, forced) = force_trace(
            &model,
            &constants,
            composition,
            &trace,
            0x0bea_0000_0000_0002,
            &device,
        );
        // The host grammar, driven along the same trace.
        let mut host = TraceState::new_exact(limits(), composition);
        host.apply(trace[0]).expect("START applies");
        for (i, got) in forced.iter().enumerate() {
            let token = trace[i + 1];
            let want = host.masks(token);
            let host_masks = [want.kinds, want.atom_types, want.bonds, want.pointers];
            // Fields the kind does not use come back as 0 from the host and
            // as "index 0 only" from the device (the convention that keeps
            // the four fields summing to the joint log-probability).
            let used = [
                true,
                token.kind == 2,
                (token.kind == 2 && host.step() > 1) || token.kind == 3,
                (token.kind == 2 && host.step() > 1) || token.kind == 3,
            ];
            for f in 0..4 {
                let device_mask = got.masks[f];
                if used[f] {
                    assert_eq!(
                        device_mask,
                        host_masks[f],
                        "{} atoms, step {}, field {f}: device mask {device_mask:#x} vs host {:#x}",
                        graph.atoms().len(),
                        i + 1,
                        host_masks[f]
                    );
                } else {
                    assert_eq!(
                        device_mask,
                        1,
                        "{} atoms, step {}, field {f}: an unused field is index 0 only",
                        graph.atoms().len(),
                        i + 1
                    );
                }
                let row = &got.rows[f];
                let mut total = 0.0f64;
                for (j, &value) in row.iter().enumerate() {
                    if device_mask & (1 << j) != 0 {
                        assert!(
                            value <= 0.0 && value.is_finite(),
                            "{} atoms, step {}, field {f}, index {j}: a legal action has log-probability {value}",
                            graph.atoms().len(),
                            i + 1
                        );
                        total += f64::from(value).exp();
                    } else {
                        assert!(
                            value <= floor,
                            "{} atoms, step {}, field {f}, index {j}: an illegal action has log-probability {value}, above the floor {floor:e}",
                            graph.atoms().len(),
                            i + 1
                        );
                    }
                }
                assert!(
                    (total - 1.0).abs() <= 1e-5,
                    "{} atoms, step {}, field {f}: the legal set sums to {total}, not 1",
                    graph.atoms().len(),
                    i + 1
                );
                checked += 1;
            }
            host.apply(token).expect("a canonical token applies");
        }
    }
    println!("masking: {checked} field distributions checked");
}

#[test]
fn forced_trace_is_the_only_trace() {
    let _lock = serial();
    let _guard = no_grad();
    let device = dev();
    let model = model_of(23, &device);
    let constants = Ms2Constants::new(&device);
    let steps = limits().max_steps();
    let (trace, composition) = trace_and_composition(&ethane());
    let id = 0x0bea_0000_0000_0003u64;

    // Drive the trace with the beam primitives, and check on the way that
    // every used field had exactly one legal index: the sampler then has no
    // choice to make, whatever it draws.
    let (forced_run, forced) = force_trace(&model, &constants, composition, &trace, id, &device);
    for (i, got) in forced.iter().enumerate() {
        for f in 0..4 {
            let mask = got.masks[f];
            assert_eq!(
                mask.count_ones(),
                1,
                "ethane step {}, field {f}: mask {mask:#x} is not a singleton, so the trace is not forced",
                i + 1
            );
        }
    }
    let committed = records(&forced_run);

    // The same loop, sampled. A forced trace leaves the draw nothing to do,
    // so the record must match word for word — including the trace
    // log-probability, which is exactly 0 on a singleton legal set.
    let mut sampled = start(&model, &[composition], 1, &[id], &device);
    sample_steps(
        &mut sampled,
        &model,
        &constants,
        1,
        steps,
        1.0,
        777,
        &device,
    );
    let drawn = records(&sampled);
    assert_eq!(
        trace_words(&drawn, steps, 16, 0),
        expected_trace_words(&trace),
        "the sampled trace of a forced molecule must be the canonical one"
    );
    assert_eq!(
        committed, drawn,
        "committing the chosen action must leave the record the sampler leaves"
    );
    println!("forced trace: {} tokens, records equal", trace.len());
}

#[test]
fn a_forced_prefix_lands_the_record_teacher_forcing_expects() {
    let _lock = serial();
    let _guard = no_grad();
    let device = dev();
    let model = model_of(29, &device);
    let constants = Ms2Constants::new(&device);
    let steps = limits().max_steps();
    let atoms = 16usize;
    for graph in [ethanol(), toluene()] {
        let (trace, composition) = trace_and_composition(&graph);
        let (run, forced) = force_trace(
            &model,
            &constants,
            composition,
            &trace,
            0x0bea_0000_0000_0004,
            &device,
        );
        let record = records(&run);
        let width = ms2::sample_record_width(steps, atoms);
        assert_eq!(record.len(), width);
        // The trace words, the length and the finished status.
        assert_eq!(
            trace_words(&record, steps, atoms, 0),
            expected_trace_words(&trace),
            "the forced trace"
        );
        assert_eq!(
            record[steps * 4 + atoms] as usize,
            trace.len(),
            "the record's length is the trace's"
        );
        assert_eq!(record[steps * 4 + atoms + 1] & 1, 1, "the row is finished");
        // A complete molecule has no open valence left.
        let end = replay_exact(&trace, limits(), composition).unwrap();
        assert!(end.is_complete());
        for j in 0..atoms {
            assert_eq!(
                record[steps * 4 + j],
                0,
                "atom {j} of a complete molecule has an open valence"
            );
        }
        // The trace log-probability is the sum of the committed steps.
        let want: f32 = forced.iter().map(|s| s.field.iter().sum::<f32>()).sum();
        let got = f32::from_bits(record[steps * 4 + atoms + 2]);
        assert!(
            (got - want).abs() <= 1e-5 * want.abs().max(1.0),
            "the record's trace log-probability is {got}, not the committed {want}"
        );
        println!(
            "forced prefix ({} atoms): {} tokens, trace log-probability {got}",
            graph.atoms().len(),
            trace.len()
        );
    }
}

#[test]
fn gather_refuses_states_and_parent_vectors_it_cannot_serve() {
    let _lock = serial();
    let _guard = no_grad();
    let device = dev();
    let model = model_of(31, &device);
    let constants = Ms2Constants::new(&device);
    let (_, composition) = trace_and_composition(&ethanol());
    let k = 2usize;
    let run = start(&model, &[composition], k, &[5], &device);
    let parents = IdTensor::from_slice(&[0u32, 1], vec![2], &device).unwrap();

    // An in-place-carry state and a fused state are both refused by name.
    // Either may fall back to the composed state on a backend that cannot
    // run the fused step, and then there is nothing to refuse.
    let unobserved = model
        .decoder()
        .start_state_unobserved(&run.encoded, run.rows, &device)
        .unwrap();
    check_launches(&device).unwrap();
    if unobserved.carries_in_place() {
        let error = unobserved
            .gather(&parents, &device)
            .err()
            .expect("an in-place-carry state cannot be gathered")
            .to_string();
        assert!(error.contains("in place"), "{error}");
    } else {
        println!("in-place carries are not available on this backend; nothing to refuse");
    }
    let fused = model
        .decoder()
        .start_state_fused(&run.encoded, run.rows, &device)
        .unwrap();
    check_launches(&device).unwrap();
    if fused.fused.is_some() && !fused.carries_in_place() {
        let error = fused
            .gather(&parents, &device)
            .err()
            .expect("a fused state cannot be gathered")
            .to_string();
        assert!(error.contains("fused step"), "{error}");
    } else {
        println!("the fused step is not available on this backend; nothing to refuse");
    }

    // An out-of-range parent is an error, not an out-of-bounds read.
    let past_end = IdTensor::from_slice(&[0u32, k as u32], vec![2], &device).unwrap();
    let error = run
        .state
        .gather(&past_end, &device)
        .err()
        .expect("a parent past the last row is an error")
        .to_string();
    assert!(error.contains("out of range"), "{error}");
    let error = gather_generation_rows(
        &run.actions,
        &run.replay,
        &run.step_token,
        &run.traj_meta,
        &past_end,
        &[1, 2],
        run.steps,
        run.atoms,
        &device,
    )
    .err()
    .expect("a parent past the last row is an error")
    .to_string();
    assert!(error.contains("out of range"), "{error}");

    // A parent vector of the wrong rank, and a key vector of the wrong
    // length, are errors too.
    let two_dim = IdTensor::from_slice(&[0u32, 1, 0, 1], vec![2, 2], &device).unwrap();
    let error = run
        .state
        .gather(&two_dim, &device)
        .err()
        .expect("a two-column parent vector is an error")
        .to_string();
    assert!(error.contains("row ids"), "{error}");
    let error = gather_generation_rows(
        &run.actions,
        &run.replay,
        &run.step_token,
        &run.traj_meta,
        &parents,
        &[7],
        run.steps,
        run.atoms,
        &device,
    )
    .err()
    .expect("one key for two children is an error")
    .to_string();
    assert!(error.contains("child keys"), "{error}");

    // The step scorer refuses the same two states, and step 0.
    let mut buffers = StepLogProbs::<R, E>::new(run.rows, run.atoms, &device);
    let bond_table = model.decoder().bond_by_type_value();
    let context = IdTensor::from_slice(&[4u32, 0, 0, 0, 4, 0, 0, 0], vec![2, 4], &device).unwrap();
    let mut state = model
        .decoder()
        .start_state(&run.encoded, run.rows, &device)
        .unwrap();
    let error = composed_step_log_probs(
        model.decoder(),
        &run.encoded,
        &run.traj_formula,
        &run.step_token,
        &run.replay,
        &run.traj_meta,
        &context,
        &mut state,
        &bond_table,
        &constants.atom_table,
        &mut buffers,
        0,
        run.atoms,
        run.closures,
        run.k,
        run.rows,
    )
    .err()
    .expect("step 0 is not scored")
    .to_string();
    assert!(error.contains("step 0"), "{error}");
    check_launches(&device).unwrap();
}

#[test]
fn identity_gather_copies_every_buffer_word_for_word() {
    let _lock = serial();
    let _guard = no_grad();
    let device = dev();
    let model = model_of(37, &device);
    let constants = Ms2Constants::new(&device);
    let (_, composition) = trace_and_composition(&toluene());
    let k = 4usize;
    let id = 0x3737_0000_0000_0001u64;
    let mut run = start(&model, &[composition], k, &[id], &device);
    sample_steps(&mut run, &model, &constants, 1, 4, 1.0, 5, &device);

    let before_ids: Vec<Vec<u32>> = vec![
        run.actions.try_to_vec().unwrap(),
        run.replay.try_to_vec().unwrap(),
        run.step_token.try_to_vec().unwrap(),
        run.traj_meta.try_to_vec().unwrap(),
        run.state.resid_ids.try_to_vec().unwrap(),
    ];
    let mut before_floats: Vec<Vec<f32>> = vec![
        run.state.prev_h.try_to_f32().unwrap(),
        run.state.atom_memory.try_to_f32().unwrap(),
    ];
    for cache in &run.state.caches {
        before_floats.push(cache.ssm.h.try_to_f32().unwrap());
        before_floats.push(cache.ssm.last_u.try_to_f32().unwrap());
        if let Some(angle) = &cache.ssm.angle {
            before_floats.push(angle.try_to_f32().unwrap());
        }
        if let Some(conv) = &cache.conv {
            before_floats.push(conv.try_to_f32().unwrap());
        }
    }
    check_launches(&device).unwrap();

    let parents: Vec<u32> = (0..k as u32).collect();
    gather(&mut run, &parents, &vec![id; k], &device);

    let after_ids: Vec<Vec<u32>> = vec![
        run.actions.try_to_vec().unwrap(),
        run.replay.try_to_vec().unwrap(),
        run.step_token.try_to_vec().unwrap(),
        run.traj_meta.try_to_vec().unwrap(),
        run.state.resid_ids.try_to_vec().unwrap(),
    ];
    let mut after_floats: Vec<Vec<f32>> = vec![
        run.state.prev_h.try_to_f32().unwrap(),
        run.state.atom_memory.try_to_f32().unwrap(),
    ];
    for cache in &run.state.caches {
        after_floats.push(cache.ssm.h.try_to_f32().unwrap());
        after_floats.push(cache.ssm.last_u.try_to_f32().unwrap());
        if let Some(angle) = &cache.ssm.angle {
            after_floats.push(angle.try_to_f32().unwrap());
        }
        if let Some(conv) = &cache.conv {
            after_floats.push(conv.try_to_f32().unwrap());
        }
    }
    check_launches(&device).unwrap();

    let names = ["actions", "replay", "step_token", "traj_meta", "resid_ids"];
    for (i, name) in names.iter().enumerate() {
        assert_eq!(
            after_ids[i], before_ids[i],
            "{name} changed under an identity gather"
        );
    }
    assert_eq!(after_floats.len(), before_floats.len(), "buffer count");
    for (i, (a, b)) in after_floats.iter().zip(before_floats.iter()).enumerate() {
        let diffs: Vec<usize> = a
            .iter()
            .zip(b.iter())
            .enumerate()
            .filter(|(_, (x, y))| x.to_bits() != y.to_bits())
            .map(|(j, _)| j)
            .collect();
        assert!(
            diffs.is_empty(),
            "float buffer {i} ({} elements) differs at {} places, first {:?}",
            a.len(),
            diffs.len(),
            &diffs[..diffs.len().min(4)]
        );
    }
    println!(
        "identity gather: {} id buffers and {} float buffers copied exactly",
        before_ids.len(),
        before_floats.len()
    );
}

#[test]
fn scoring_and_committing_read_nothing_and_a_gather_reads_only_its_parents() {
    let _lock = serial();
    let _guard = no_grad();
    let device = dev();
    let model = model_of(41, &device);
    let constants = Ms2Constants::new(&device);
    let (trace, composition) = trace_and_composition(&ethanol());
    let mut run = start(&model, &[composition], 1, &[0x4141], &device);
    let mut buffers = StepLogProbs::<R, E>::new(run.rows, run.atoms, &device);
    let bond_table = model.decoder().bond_by_type_value();
    let zero = Tensor::<R, E>::from_f32(&[0.0], vec![1], &device).unwrap();
    // The read counter is process-global, so a delta can only be inflated by
    // another thread, never deflated: the minimum over the steps is this
    // thread's own count.
    let mut score_reads = Vec::new();
    let mut commit_reads = Vec::new();
    for step in 1..trace.len() {
        let words = token_words(&trace[step]);
        let context = IdTensor::from_slice(&words, vec![1, 4], &device).unwrap();
        let before = runtime_read_count();
        ms2::step_token(&run.actions, &mut run.step_token, run.steps, run.atoms).unwrap();
        composed_step_log_probs(
            model.decoder(),
            &run.encoded,
            &run.traj_formula,
            &run.step_token,
            &run.replay,
            &run.traj_meta,
            &context,
            &mut run.state,
            &bond_table,
            &constants.atom_table,
            &mut buffers,
            step,
            run.atoms,
            run.closures,
            run.k,
            run.rows,
        )
        .unwrap();
        score_reads.push(runtime_read_count() - before);
        let before = runtime_read_count();
        apply_chosen_action(
            &context,
            &mut run.actions,
            &mut run.step_token,
            &mut run.replay,
            &run.traj_meta,
            &mut run.state,
            &zero,
            &constants.atom_table,
            step,
            run.steps,
            run.atoms,
            run.closures,
        )
        .unwrap();
        commit_reads.push(runtime_read_count() - before);
        check_launches(&device).unwrap();
    }
    println!("score read deltas {score_reads:?}, commit read deltas {commit_reads:?}");
    assert_eq!(
        score_reads.iter().min(),
        Some(&0),
        "scoring a step must read nothing"
    );
    assert_eq!(
        commit_reads.iter().min(),
        Some(&0),
        "committing an action must read nothing"
    );

    // A gather reads exactly one buffer: the parent vector it validates.
    let parents = IdTensor::from_slice(&[0u32], vec![1], &device).unwrap();
    let mut state_reads = Vec::new();
    let mut rows_reads = Vec::new();
    for _ in 0..4 {
        let before = runtime_read_count();
        let gathered = run.state.gather(&parents, &device).unwrap();
        state_reads.push(runtime_read_count() - before);
        let before = runtime_read_count();
        let rows = gather_generation_rows(
            &run.actions,
            &run.replay,
            &run.step_token,
            &run.traj_meta,
            &parents,
            &[0x4141],
            run.steps,
            run.atoms,
            &device,
        )
        .unwrap();
        rows_reads.push(runtime_read_count() - before);
        check_launches(&device).unwrap();
        drop(gathered);
        drop(rows);
    }
    println!("gather read deltas: state {state_reads:?}, rows {rows_reads:?}");
    assert_eq!(
        state_reads.iter().min(),
        Some(&1),
        "a state gather reads the parent vector and nothing else"
    );
    assert_eq!(
        rows_reads.iter().min(),
        Some(&1),
        "a trajectory gather reads the parent vector and nothing else"
    );
}

// Beam search over the completion grammar
// ---------------------------------------------------------------------------
// The search is `CompletionModel::generate_beam_with_spectra`; the primitives
// above are what it stands on. The check that matters most is
// `the_beam_scores_actions_exactly_as_the_device_does`: the search enumerates
// and scores a row's actions on the host, from the one packed head row of the
// step, and a wrong composition there would rank actions wrongly while every
// other property below still held.

/// Requests for a list of (composition, id), with no patterns.
fn plain_requests<'a>(compositions: &[Composition], ids: &[u64]) -> Vec<CompletionRequest<'a>> {
    compositions
        .iter()
        .zip(ids.iter())
        .map(|(composition, id)| CompletionRequest {
            id: *id,
            composition: *composition,
            patterns: &[],
            acceptance_patterns: None,
            fingerprint: None,
        })
        .collect()
}

fn beam_config(returned: u32) -> CompletionGenerationConfig {
    CompletionGenerationConfig {
        trajectories: 1,
        temperature: 1.0,
        seed: 7,
        returned,
        containment_node_limit: 100_000,
        identity_work_limit: 100_000,
        condition_on_patterns: true,
        substructure_semantics: SubstructureSemantics::Contained,
    }
}

#[test]
fn the_beam_scores_actions_exactly_as_the_device_does() {
    let _lock = serial();
    let device = dev();
    let model = model_of(101, &device);
    let constants = Ms2Constants::new(&device);
    // Tolerance: the host and the device run the same masked log-softmax over
    // the same f32 head values in the same index order, so only the last bits
    // may differ; 1e-5 on a per-action joint log-probability is far inside
    // what a wrong composition would cost (whole nats).
    let tolerance = 1e-5f32;
    let graphs = [ethanol(), toluene()];
    let compositions: Vec<Composition> = graphs
        .iter()
        .map(|graph| trace_and_composition(graph).1)
        .collect();
    let ids: Vec<u64> = vec![0xbea0_0001, 0xbea0_0002];
    let requests = plain_requests(&compositions, &ids);
    let none: Vec<Option<&SpectrumEvidence>> = vec![None; requests.len()];
    let mut audit = BeamAudit::default();
    let (outcomes, stats) = model
        .generate_beam_audited(
            &requests,
            &none,
            &beam_config(25),
            6,
            &constants,
            &device,
            &mut audit,
        )
        .unwrap();
    check_launches(&device).unwrap();
    assert_eq!(outcomes.len(), 2);
    assert!(
        !audit.actions.is_empty(),
        "the audit recorded no action, so nothing was checked"
    );
    let (worst, action) = audit.worst().expect("a worst action");
    for entry in audit.actions.iter() {
        assert!(
            (entry.host - entry.device).abs() <= tolerance,
            "step {} row {} token {:?}: the search scored {} and the device {}",
            entry.step,
            entry.row,
            entry.token,
            entry.host,
            entry.device
        );
    }
    let stops = audit.actions.iter().filter(|a| a.stop).count();
    println!(
        "beam scoring: {} actions checked ({stops} STOP), worst deviation {worst:e} at step {} token {:?} (tolerance {tolerance:e})",
        audit.actions.len(),
        action.step,
        action.token
    );
    println!(
        "beam stats: {:?}",
        stats
            .iter()
            .map(|s| s.candidates_scored)
            .collect::<Vec<_>>()
    );
    assert!(
        stops > 0,
        "no STOP was audited, so no finished candidate was checked"
    );
}

#[test]
fn width_one_is_greedy() {
    let _lock = serial();
    let device = dev();
    let model = model_of(103, &device);
    let constants = Ms2Constants::new(&device);
    let (_, composition) = trace_and_composition(&ethanol());
    let id = 0x9001u64;
    let requests = plain_requests(&[composition], &[id]);
    let none: Vec<Option<&SpectrumEvidence>> = vec![None];
    let (outcomes, _) = model
        .generate_beam_with_spectra(&requests, &none, &beam_config(25), 1, &constants, &device)
        .unwrap();
    check_launches(&device).unwrap();
    let beam_traces: Vec<Vec<Token>> = outcomes[0]
        .sampled
        .iter()
        .map(|t| t.trace.clone())
        .collect();

    // The hand-rolled reference: at every step take the joint argmax over all
    // legal actions, with every number coming from the device scorer
    // (`composed_step_heads` once, then `step_field_log_probs` at each legal
    // (kind, type, bond) context — the heads of a step do not depend on the
    // context, so one decoder pass serves them all).
    let mut run = start(&model, &[composition], 1, &[id], &device);
    let mut buffers = StepLogProbs::<R, E>::new(1, run.atoms, &device);
    let bond_table = model.decoder().bond_by_type_value();
    let mut host = TraceState::new_exact(limits(), composition);
    host.apply(Token {
        kind: 1,
        atom_type: 0,
        bond: 0,
        pointer: 0,
    })
    .unwrap();
    let mut greedy: Vec<Token> = vec![Token {
        kind: 1,
        atom_type: 0,
        bond: 0,
        pointer: 0,
    }];
    let mut greedy_lp = 0.0f64;
    for step in 1..run.steps {
        ms2::step_token(&run.actions, &mut run.step_token, run.steps, run.atoms).unwrap();
        composed_step_heads(
            model.decoder(),
            &run.encoded,
            &run.traj_formula,
            &run.step_token,
            &run.replay,
            &mut run.state,
            &mut buffers.logits,
            step,
            run.atoms,
            run.k,
            run.rows,
        )
        .unwrap();
        check_launches(&device).unwrap();
        // Every legal (kind, type, bond) context of this row, from the host
        // grammar; the pointer comes from the context's own field.
        let kinds = host
            .masks(Token {
                kind: 0,
                atom_type: 0,
                bond: 0,
                pointer: 0,
            })
            .kinds;
        if kinds == 0 {
            break;
        }
        let mut contexts: Vec<(u8, u8, u8)> = Vec::new();
        for kind in [2u8, 3u8, 4u8] {
            if kinds & (1 << kind) == 0 {
                continue;
            }
            if kind == 4 {
                contexts.push((4, 0, 0));
                continue;
            }
            if kind == 2 {
                let types = host
                    .masks(Token {
                        kind,
                        atom_type: 0,
                        bond: 0,
                        pointer: 0,
                    })
                    .atom_types;
                for ty in 0..18u8 {
                    if types & (1 << ty) == 0 {
                        continue;
                    }
                    if host.step() == 1 {
                        contexts.push((2, ty, 0));
                        continue;
                    }
                    let bonds = host
                        .masks(Token {
                            kind,
                            atom_type: ty,
                            bond: 0,
                            pointer: 0,
                        })
                        .bonds;
                    for bond in 0..4u8 {
                        if bonds & (1 << bond) != 0 {
                            contexts.push((2, ty, bond));
                        }
                    }
                }
            } else {
                let bonds = host
                    .masks(Token {
                        kind,
                        atom_type: 0,
                        bond: 0,
                        pointer: 0,
                    })
                    .bonds;
                for bond in 0..4u8 {
                    if bonds & (1 << bond) != 0 {
                        contexts.push((3, 0, bond));
                    }
                }
            }
        }
        let mut best: Option<(f32, Token)> = None;
        for (kind, atom_type, bond) in contexts {
            let probe = Token {
                kind,
                atom_type,
                bond,
                pointer: 0,
            };
            let words = token_words(&probe);
            let context = IdTensor::from_slice(&words, vec![1, 4], &device).unwrap();
            step_field_log_probs(
                &run.replay,
                &run.traj_meta,
                &context,
                &bond_table,
                &constants.atom_table,
                &mut buffers,
                run.atoms,
                run.closures,
            )
            .unwrap();
            let kind_lp = buffers.kind.try_to_f32().unwrap();
            let type_lp = buffers.atom_type.try_to_f32().unwrap();
            let bond_lp = buffers.bond.try_to_f32().unwrap();
            let ptr_lp = buffers.pointer.try_to_f32().unwrap();
            let plan = buffers.plan.try_to_vec().unwrap();
            check_launches(&device).unwrap();
            let head = kind_lp[usize::from(kind)]
                + type_lp[usize::from(atom_type)]
                + bond_lp[usize::from(bond)];
            if kind == 4 || (kind == 2 && host.step() == 1) {
                // No pointer field: index 0 carries log-probability 0.
                let joint = head + ptr_lp[0];
                let token = Token {
                    kind,
                    atom_type,
                    bond,
                    pointer: 0,
                };
                if best
                    .as_ref()
                    .is_none_or(|(b, t)| joint > *b || (joint == *b && token < *t))
                {
                    best = Some((joint, token));
                }
                continue;
            }
            for pointer in 0..run.atoms {
                if plan[3] & (1 << pointer) == 0 {
                    continue;
                }
                let joint = head + ptr_lp[pointer];
                let token = Token {
                    kind,
                    atom_type,
                    bond,
                    pointer: pointer as u8,
                };
                if best
                    .as_ref()
                    .is_none_or(|(b, t)| joint > *b || (joint == *b && token < *t))
                {
                    best = Some((joint, token));
                }
            }
        }
        let Some((joint, token)) = best else { break };
        greedy.push(token);
        greedy_lp += f64::from(joint);
        host.apply(token).unwrap();
        let words = token_words(&token);
        let context = IdTensor::from_slice(&words, vec![1, 4], &device).unwrap();
        let lp = Tensor::<R, E>::from_f32(&[joint], vec![1], &device).unwrap();
        apply_chosen_action(
            &context,
            &mut run.actions,
            &mut run.step_token,
            &mut run.replay,
            &run.traj_meta,
            &mut run.state,
            &lp,
            &constants.atom_table,
            step,
            run.steps,
            run.atoms,
            run.closures,
        )
        .unwrap();
        check_launches(&device).unwrap();
        if token.kind == 4 {
            break;
        }
    }
    println!(
        "greedy reference: {} tokens, log-probability {greedy_lp:.4}; beam traces {}",
        greedy.len(),
        beam_traces.len()
    );
    assert_eq!(beam_traces.len(), 1, "width 1 finishes exactly one trace");
    assert_eq!(
        beam_traces[0], greedy,
        "the width-1 beam must take the joint argmax at every step"
    );
    let beam_lp = outcomes[0].sampled[0].log_prob;
    assert!(
        (f64::from(beam_lp) - greedy_lp).abs() <= 1e-4,
        "width-1 trace log-probability {beam_lp} vs the greedy reference {greedy_lp}"
    );
}

#[test]
fn every_returned_candidate_is_legal_on_composition_and_distinct() {
    let _lock = serial();
    let device = dev();
    let model = model_of(107, &device);
    let constants = Ms2Constants::new(&device);
    let graphs = [ethanol(), toluene(), ethane()];
    let compositions: Vec<Composition> = graphs
        .iter()
        .map(|graph| trace_and_composition(graph).1)
        .collect();
    let ids: Vec<u64> = vec![11, 22, 33];
    let requests = plain_requests(&compositions, &ids);
    let none: Vec<Option<&SpectrumEvidence>> = vec![None; requests.len()];
    let (outcomes, stats) = model
        .generate_beam_with_spectra(&requests, &none, &beam_config(25), 16, &constants, &device)
        .unwrap();
    check_launches(&device).unwrap();
    let mut total = 0usize;
    for (q, outcome) in outcomes.iter().enumerate() {
        for candidate in outcome.candidates.iter() {
            total += 1;
            let end = replay_exact(&candidate.trace, limits(), compositions[q])
                .expect("a returned candidate replays");
            assert!(end.stopped() && end.is_complete(), "query {q}: complete");
            let graph = end.graph().expect("a returned candidate has a graph");
            assert!(graph.is_connected(), "query {q}: connected");
            assert_eq!(
                graph.composition(),
                compositions[q],
                "query {q}: on composition"
            );
        }
        // No two returned candidates are the same molecule.
        for i in 0..outcome.candidates.len() {
            for j in i + 1..outcome.candidates.len() {
                let left = replay_exact(&outcome.candidates[i].trace, limits(), compositions[q])
                    .unwrap()
                    .graph()
                    .unwrap();
                let right = replay_exact(&outcome.candidates[j].trace, limits(), compositions[q])
                    .unwrap()
                    .graph()
                    .unwrap();
                assert_ne!(
                    same_identity(&left, &right, 100_000),
                    Some(true),
                    "query {q}: candidates {i} and {j} are the same molecule"
                );
            }
        }
        println!(
            "query {q}: {} candidates, {} finished, {} dropped by the width",
            outcome.candidates.len(),
            stats[q].finished,
            stats[q].candidates_dropped
        );
    }
    assert!(total > 0, "the beam returned nothing to check");
}

#[test]
fn the_beam_is_deterministic() {
    let _lock = serial();
    let device = dev();
    let model = model_of(109, &device);
    let constants = Ms2Constants::new(&device);
    let graphs = [toluene(), ethanol()];
    let compositions: Vec<Composition> = graphs
        .iter()
        .map(|graph| trace_and_composition(graph).1)
        .collect();
    let requests = plain_requests(&compositions, &[5, 6]);
    let none: Vec<Option<&SpectrumEvidence>> = vec![None; requests.len()];
    let run = || {
        let (outcomes, stats) = model
            .generate_beam_with_spectra(&requests, &none, &beam_config(25), 8, &constants, &device)
            .unwrap();
        check_launches(&device).unwrap();
        (outcomes, stats)
    };
    let (first, first_stats) = run();
    let (second, second_stats) = run();
    for q in 0..2 {
        let traces = |outcome: &mamba3::models::ms2::completion_model::QueryOutcome| {
            outcome
                .candidates
                .iter()
                .map(|c| (c.trace.clone(), c.samples, c.best_log_prob.to_bits()))
                .collect::<Vec<_>>()
        };
        assert_eq!(
            traces(&first[q]),
            traces(&second[q]),
            "query {q}: the same inputs must give the same candidates in the same order"
        );
        assert_eq!(
            first_stats[q], second_stats[q],
            "query {q}: the same accounting"
        );
    }
    println!(
        "determinism: {} and {} candidates, {} and {} candidates scored",
        first[0].candidates.len(),
        first[1].candidates.len(),
        first_stats[0].candidates_scored,
        first_stats[1].candidates_scored
    );
}

#[test]
fn the_beam_accounting_adds_up() {
    let _lock = serial();
    let device = dev();
    let model = model_of(113, &device);
    let constants = Ms2Constants::new(&device);
    let graphs = [ethanol(), toluene()];
    let compositions: Vec<Composition> = graphs
        .iter()
        .map(|graph| trace_and_composition(graph).1)
        .collect();
    let requests = plain_requests(&compositions, &[71, 72]);
    let none: Vec<Option<&SpectrumEvidence>> = vec![None; requests.len()];
    for width in [1u32, 4, 16] {
        let (outcomes, stats) = model
            .generate_beam_with_spectra(
                &requests,
                &none,
                &beam_config(25),
                width,
                &constants,
                &device,
            )
            .unwrap();
        check_launches(&device).unwrap();
        for (q, report) in stats.iter().enumerate() {
            // Every scored action either entered the beam, left it finished,
            // or fell below the width's cut.
            assert_eq!(
                report.candidates_scored,
                report.admitted_live
                    + u64::from(report.finished)
                    + report.candidates_dropped
                    + report.refused_tokens,
                "width {width} query {q}: scored actions are admitted, finished or dropped ({report:?})"
            );
            // Every row the search ever held — the one it started with plus
            // the admitted ones — was expanded or was still live at the end.
            assert_eq!(
                1 + report.admitted_live,
                report.rows_expanded + u64::from(report.live_at_limit),
                "width {width} query {q}: every row held was expanded or is live ({report:?})"
            );
            assert_eq!(
                report.refused_tokens, 0,
                "width {width} query {q}: no legal action was refused"
            );
            assert_eq!(
                report.row_steps,
                u64::from(width) * u64::from(report.steps_run),
                "width {width} query {q}: row-steps are width times the steps run"
            );
            assert!(
                report.live_row_steps <= report.row_steps,
                "width {width} query {q}: live row-steps cannot exceed the executed ones"
            );
            assert_eq!(
                u64::from(outcomes[q].trajectories),
                u64::from(report.finished),
                "width {width} query {q}: every finished candidate reaches the host path"
            );
            println!(
                "width {width} query {q}: scored {} admitted {} finished {} dropped {} rows {} live-at-limit {} row-steps {} (live {})",
                report.candidates_scored,
                report.admitted_live,
                report.finished,
                report.candidates_dropped,
                report.rows_expanded,
                report.live_at_limit,
                report.row_steps,
                report.live_row_steps
            );
        }
    }
}

#[test]
fn a_wider_beam_keeps_what_a_narrower_one_found() {
    let _lock = serial();
    let device = dev();
    let model = model_of(127, &device);
    let constants = Ms2Constants::new(&device);
    let graphs = [ethanol(), toluene()];
    let compositions: Vec<Composition> = graphs
        .iter()
        .map(|graph| trace_and_composition(graph).1)
        .collect();
    let requests = plain_requests(&compositions, &[81, 82]);
    let none: Vec<Option<&SpectrumEvidence>> = vec![None; requests.len()];
    // `returned` at its maximum, so the shortlist cut cannot hide a
    // candidate: monotonicity is a property of the search, not of the cut.
    let pool = |width: u32| {
        let (outcomes, _) = model
            .generate_beam_with_spectra(
                &requests,
                &none,
                &beam_config(1024),
                width,
                &constants,
                &device,
            )
            .unwrap();
        check_launches(&device).unwrap();
        outcomes
    };
    for (narrow_w, wide_w) in [(2u32, 4u32), (4, 8)] {
        let narrow = pool(narrow_w);
        let wide = pool(wide_w);
        for q in 0..2 {
            let mut missing = Vec::new();
            for candidate in narrow[q].candidates.iter() {
                let left = replay_exact(&candidate.trace, limits(), compositions[q])
                    .unwrap()
                    .graph()
                    .unwrap();
                let found = wide[q].candidates.iter().any(|other| {
                    let right = replay_exact(&other.trace, limits(), compositions[q])
                        .unwrap()
                        .graph()
                        .unwrap();
                    same_identity(&left, &right, 100_000) == Some(true)
                });
                if !found {
                    missing.push(candidate.trace.clone());
                }
            }
            println!(
                "query {q}: width {narrow_w} found {} identities, width {wide_w} found {} ({} of the narrower ones missing)",
                narrow[q].candidates.len(),
                wide[q].candidates.len(),
                missing.len()
            );
            // Beam search is not monotone in the width in general: a wider
            // beam keeps more hypotheses at every step, and a hypothesis the
            // narrow beam carried to a STOP can be crowded out of the wider
            // beam by better-scoring prefixes that never finish. The
            // assertion is therefore on the measured case, and it names what
            // was lost if the case ever stops holding.
            assert!(
                missing.is_empty(),
                "query {q}: width {wide_w} lost {} identities width {narrow_w} found: {missing:?}",
                missing.len()
            );
            assert!(
                wide[q].candidates.len() >= narrow[q].candidates.len(),
                "query {q}: width {wide_w} returned fewer identities than width {narrow_w}"
            );
        }
    }
}

/// Nine hand-built molecules (the set `ms2_completion_fingerprint.rs` and
/// `ms2_completion_spectrum.rs` overfit), for the sampling comparison.
fn nine_molecules() -> Vec<MolGraph> {
    let chain =
        |atoms: Vec<u8>, bonds: Vec<(usize, usize, u8)>| MolGraph::new(atoms, bonds).unwrap();
    vec![
        chain(vec![4, 3, 9], vec![(0, 1, 1), (1, 2, 1)]),
        chain(vec![4, 8, 4], vec![(0, 1, 1), (1, 2, 1)]),
        chain(vec![4, 3, 3, 9], vec![(0, 1, 1), (1, 2, 1), (2, 3, 1)]),
        chain(vec![4, 2, 4, 9], vec![(0, 1, 1), (1, 2, 1), (1, 3, 1)]),
        chain(vec![4, 8, 3, 4], vec![(0, 1, 1), (1, 2, 1), (2, 3, 1)]),
        chain(vec![4, 3, 7], vec![(0, 1, 1), (1, 2, 1)]),
        chain(vec![4, 6, 4], vec![(0, 1, 1), (1, 2, 1)]),
        chain(vec![3, 3, 3], vec![(0, 1, 1), (1, 2, 1), (2, 0, 1)]),
        toluene(),
    ]
}

#[test]
fn a_beam_finds_a_molecule_sampling_misses_at_the_same_rows() {
    let _lock = serial();
    let device = dev();
    let molecules = nine_molecules();
    let pairs: Vec<(Vec<Token>, Composition)> =
        molecules.iter().map(trace_and_composition).collect();
    let empty: Vec<Vec<MolGraph>> = molecules.iter().map(|_| Vec::new()).collect();
    let refs: Vec<&[MolGraph]> = empty.iter().map(Vec::as_slice).collect();
    let traces: Vec<&[Token]> = pairs.iter().map(|p| p.0.as_slice()).collect();
    let compositions: Vec<Composition> = pairs.iter().map(|p| p.1).collect();
    // The regime this compares in is the interesting one and has to be hit
    // deliberately: a saturated model makes every trace probability ~1, so
    // sampling finds the target too (that is what happens by 120 steps,
    // where the mean trace NLL is ~0.7 — a trace probability of ~0.5, which
    // four samples almost always hit), and an untrained one makes them all
    // tiny, so neither arm finds anything. Training stops at the first
    // check below `TARGET_NLL` nats, where the target's trace is the most
    // probable one but carries only `exp(-2.5)` ~ 8% of the mass, so an
    // equal-row sample budget is unlikely to draw it and a beam still ranks
    // it first.
    let train = CompletionTrainConfig {
        lr: 3e-3,
        weight_decay: 0.0,
        grad_clip: None,
        seed: 1,
        extraction_seed: 11,
        pattern_source: PatternSource::RandomPatches(ExtractionConfig {
            min_patterns: 0,
            max_patterns: 0,
            ..ExtractionConfig::default()
        }),
        extraction: ExtractionConfig {
            min_patterns: 0,
            max_patterns: 0,
            ..ExtractionConfig::default()
        },
        fingerprint_mode: None,
        fingerprint_threshold: 0.1,
    };
    const TARGET_NLL: f64 = 2.5;
    let mut trainer = CompletionTrainer::<R, E>::new(&config(), &train, &device).unwrap();
    let mean = |v: &[f32]| v.iter().map(|&x| f64::from(x)).sum::<f64>() / v.len() as f64;
    let initial = mean(
        &trainer
            .teacher_eval_with(&refs, &traces, &compositions)
            .unwrap(),
    );
    let mut trained = initial;
    let mut steps = 0usize;
    while steps < 400 && trained > TARGET_NLL {
        for _ in 0..10 {
            trainer.step_with(&refs, &traces, &compositions).unwrap();
        }
        steps += 10;
        trained = mean(
            &trainer
                .teacher_eval_with(&refs, &traces, &compositions)
                .unwrap(),
        );
    }
    check_launches(&device).unwrap();
    println!("overfit: trace NLL {initial:.3} -> {trained:.3} in {steps} steps");
    assert!(
        trained <= TARGET_NLL,
        "training did not reach the comparison regime: NLL {trained:.3} after {steps} steps"
    );
    let model = trainer.model();
    let constants = trainer.constants();

    // Equal rows: a beam of `width` rows against `width` independent
    // samples, so both arms run `width * steps` decoder row-steps.
    let mut wins: Vec<(usize, u32)> = Vec::new();
    let mut table: Vec<String> = Vec::new();
    for width in [2u32, 4, 8] {
        for (m, composition) in compositions.iter().enumerate() {
            let requests = plain_requests(&[*composition], &[1000 + m as u64]);
            let none: Vec<Option<&SpectrumEvidence>> = vec![None];
            let mut config = beam_config(25);
            config.trajectories = width;
            let (beam, stats) = model
                .generate_beam_with_spectra(&requests, &none, &config, width, constants, &device)
                .unwrap();
            let (sampled, _) = (
                model
                    .generate_with_spectra(&requests, &none, &config, constants, &device)
                    .unwrap(),
                (),
            );
            check_launches(&device).unwrap();
            let found = |outcome: &mamba3::models::ms2::completion_model::QueryOutcome| {
                outcome
                    .candidates
                    .iter()
                    .any(|c| same_identity(&c.graph, &molecules[m], 100_000) == Some(true))
            };
            let beam_hit = found(&beam[0]);
            let sample_hit = found(&sampled[0]);
            if beam_hit && !sample_hit {
                wins.push((m, width));
            }
            table.push(format!(
                "width {width} molecule {m} ({} atoms): beam {} ({} candidates, {} row-steps), sampling {} ({} candidates)",
                molecules[m].atoms().len(),
                if beam_hit { "HIT" } else { "miss" },
                beam[0].candidates.len(),
                stats[0].row_steps,
                if sample_hit { "HIT" } else { "miss" },
                sampled[0].candidates.len()
            ));
        }
    }
    for line in table.iter() {
        println!("{line}");
    }
    let beam_hits = table
        .iter()
        .filter(|line| line.contains("beam HIT"))
        .count();
    let sample_hits = table
        .iter()
        .filter(|line| line.contains("sampling HIT"))
        .count();
    println!("beam found {beam_hits} targets, sampling at the same rows found {sample_hits}");
    assert!(
        beam_hits >= sample_hits,
        "the beam found fewer targets than sampling at the same rows ({beam_hits} against {sample_hits}); tried:\n{}",
        table.join("\n")
    );
    assert!(
        !wins.is_empty(),
        "no (molecule, width) where the beam found the target and sampling at the same rows did not; tried:\n{}",
        table.join("\n")
    );
    println!(
        "the beam won at {} of {} (molecule, width) pairs: {wins:?}",
        wins.len(),
        table.len()
    );
}

#[test]
fn the_one_sweep_continuations_are_the_per_context_masks() {
    // `TraceState::add_continuations` and `close_continuations` are what the
    // beam enumerates a row with; each must equal `masks` at every context
    // it covers, which is the property that lets the search call them once
    // instead of once per (type, bond) pair.
    let mut checked = 0usize;
    for graph in [ethanol(), toluene(), ethane()] {
        let (trace, composition) = trace_and_composition(&graph);
        let mut state = TraceState::new_exact(limits(), composition);
        for token in trace.iter() {
            let mut adds = Vec::new();
            let mut closes = Vec::new();
            state.add_continuations(&mut adds);
            state.close_continuations(&mut closes);
            // Every (type, bond) pair `masks` admits is in the sweep with
            // the same pointer mask, and the sweep holds nothing else.
            let mut from_masks: Vec<(u8, u8, u32)> = Vec::new();
            for ty in 1..=17u8 {
                for bond in 1..=3u8 {
                    let m = state.masks(Token {
                        kind: 2,
                        atom_type: ty,
                        bond,
                        pointer: 0,
                    });
                    if m.bonds & (1 << bond) != 0 && m.pointers != 0 {
                        from_masks.push((ty, bond, m.pointers));
                    }
                }
            }
            assert_eq!(adds, from_masks, "the ADD_ATOM sweep matches the masks");
            let mut close_masks: Vec<(u8, u32)> = Vec::new();
            for bond in 1..=3u8 {
                let m = state.masks(Token {
                    kind: 3,
                    atom_type: 0,
                    bond,
                    pointer: 0,
                });
                if m.bonds & (1 << bond) != 0 && m.pointers != 0 {
                    close_masks.push((bond, m.pointers));
                }
            }
            assert_eq!(
                closes, close_masks,
                "the CLOSE_RING sweep matches the masks"
            );
            checked += 1;
            if state.apply(*token).is_err() {
                break;
            }
        }
    }
    println!("continuation sweeps: {checked} states checked against the per-context masks");
}

/// `ln(1 + count)` of a composition, the space an [`ElementPrior`] lives in.
fn log1p_counts(c: &Composition) -> [f32; 10] {
    let mut out = [0.0f32; 10];
    for e in 0..10 {
        out[e] = (f64::from(c[e])).ln_1p() as f32;
    }
    out
}

#[test]
fn element_prior_distance_and_weights() {
    use mamba3::models::ms2::completion_formula::ElementPrior;
    // C2H6O and CH6N2 in ELEMENTS order (C, H, N, O, F, P, S, Cl, Br, I).
    let ethanol: Composition = [2, 6, 0, 1, 0, 0, 0, 0, 0, 0];
    let hydrazine: Composition = [1, 6, 2, 0, 0, 0, 0, 0, 0, 0];
    let prior = ElementPrior {
        log1p_counts: log1p_counts(&hydrazine),
        temperature: 0.5,
    };
    assert!(prior.distance(&hydrazine) < 1e-6);
    // |ln3 - ln2| + |0 - ln3| + |ln2 - 0| = ln 9 - ... computed directly.
    let expected = (3f64.ln() - 2f64.ln()) + 3f64.ln() + 2f64.ln();
    assert!((prior.distance(&ethanol) - expected).abs() < 1e-5);
    let weights = prior.weights(&[prior.distance(&hydrazine), prior.distance(&ethanol)]);
    assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    assert!((weights[1] / weights[0] - (-expected / 0.5).exp()).abs() < 1e-6);
    // Shifting every distance by a constant changes nothing.
    let shifted = prior.weights(&[3.0, 3.0 + expected]);
    assert!((shifted[0] - weights[0]).abs() < 1e-6);
    assert!(prior.weights(&[]).is_empty());
}

#[test]
fn element_prior_reorders_formula_hypotheses() {
    use mamba3::models::ms2::chem::composition_mass;
    use mamba3::models::ms2::completion_formula::{
        CompletionSearch, ElementPrior, FormulaAllocation, FormulaPruning, MassQuery,
        run_mass_completion_search, run_mass_completion_search_with_prior,
    };
    use mamba3::models::ms2::completion_model::FormulaArtifacts;
    let _lock = serial();
    let device = dev();
    let model = model_of(23, &device);
    let constants = Ms2Constants::new(&device);
    // C6H12O2 (116.0837 Da) and C5H12N2O (116.0950 Da) differ by the N2/CO
    // gap, 97 ppm here: the widest window (100 ppm) around the first holds
    // both, the first with residual zero.
    let acid: Composition = [6, 12, 0, 2, 0, 0, 0, 0, 0, 0];
    let amide: Composition = [5, 12, 2, 1, 0, 0, 0, 0, 0, 0];
    let artifacts =
        FormulaArtifacts::fit(&[acid, amide], 16, 0, 0, "element-prior".into()).unwrap();
    let mass = MassQuery::Neutral {
        value: composition_mass(&acid).unwrap(),
        ppm_tenths: 1000,
        uncertainty: Some(1),
    };
    let run = |prior: Option<&ElementPrior>| {
        let result = run_mass_completion_search_with_prior(
            &model,
            &constants,
            &device,
            &artifacts,
            16,
            4,
            &[],
            None,
            &mass,
            8,
            100_000,
            8,
            1.0,
            7,
            16,
            "element-prior",
            true,
            FormulaPruning::ChemicalOnly,
            FormulaAllocation::Equal,
            SubstructureSemantics::Contained,
            None,
            None,
            CompletionSearch::Beam,
            prior,
        )
        .unwrap();
        check_launches(&device).unwrap();
        result
    };
    // Default: nearest mass first, even split.
    let default = run(None);
    let formulas = &default.formula_search.formulas;
    assert!(formulas.len() >= 2, "both formulas are hypotheses");
    assert_eq!(formulas[0].formula, "C6H12O2");
    let position = |result: &mamba3::models::ms2::completion_formula::MassCompletionResult,
                    text: &str| {
        result
            .formula_search
            .formulas
            .iter()
            .position(|f| f.formula == text)
            .expect("the formula is a hypothesis")
    };
    assert!(position(&default, "C5H12N2O") > 0);
    // `None` is the plain entry point.
    let plain = run_mass_completion_search(
        &model,
        &constants,
        &device,
        &artifacts,
        16,
        4,
        &[],
        None,
        &mass,
        8,
        100_000,
        8,
        1.0,
        7,
        16,
        "element-prior",
        true,
        FormulaPruning::ChemicalOnly,
        FormulaAllocation::Equal,
        SubstructureSemantics::Contained,
        None,
        None,
        CompletionSearch::Beam,
    )
    .unwrap();
    check_launches(&device).unwrap();
    let texts = |r: &mamba3::models::ms2::completion_formula::MassCompletionResult| {
        r.formula_search
            .formulas
            .iter()
            .map(|f| (f.formula.clone(), f.trajectories))
            .collect::<Vec<_>>()
    };
    assert_eq!(texts(&plain), texts(&default));
    // A prior that predicts C5H12N2O puts it first and, at a low
    // temperature, gives it the whole budget.
    let prior = ElementPrior {
        log1p_counts: log1p_counts(&amide),
        temperature: 0.05,
    };
    let guided = run(Some(&prior));
    assert_eq!(guided.formula_search.formulas[0].formula, "C5H12N2O");
    // Its weight is all but one, and no other formula is searched wider. (The
    // allocation keeps one row for every selected hypothesis, and a beam
    // reports the rows it finished, so the counts are compared, not fixed.)
    let first = &guided.formula_search.formulas[0];
    assert!(first.weight > 0.99);
    assert!(guided.formula_search.formulas[1..].iter().all(|f| f.weight < 1e-6));
    assert!(
        guided.formula_search.formulas[1..]
            .iter()
            .all(|f| f.trajectories <= first.trajectories)
    );
    assert!(
        first.trajectories
            > default.formula_search.formulas[position(&default, "C5H12N2O")].trajectories
    );
    assert!(guided.formula_search.ranking.contains("element counts"));
    // A non-positive temperature is refused, not silently replaced.
    let bad = ElementPrior {
        log1p_counts: log1p_counts(&amide),
        temperature: 0.0,
    };
    assert!(
        run_mass_completion_search_with_prior(
            &model,
            &constants,
            &device,
            &artifacts,
            16,
            4,
            &[],
            None,
            &mass,
            8,
            100_000,
            8,
            1.0,
            7,
            16,
            "element-prior",
            true,
            FormulaPruning::ChemicalOnly,
            FormulaAllocation::Equal,
            SubstructureSemantics::Contained,
            None,
            None,
            CompletionSearch::Beam,
            Some(&bad),
        )
        .is_err()
    );
}
