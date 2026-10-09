//! Per-step structural progress features for the completion model: the
//! remaining composition, the unplaced pattern atom types and the four state
//! scalars the decoder is told at every step
//! (`ms2::progress_features`, `ms2::progress_features_teacher`,
//! `CompletionModelConfig::progress_features`).
//!
//! Hand-built molecules only (atom type ids from
//! [`chem::ATOM_TYPES`](mamba3::models::ms2::chem::ATOM_TYPES)), as in
//! `ms2_completion_spectrum.rs`. Every device call is followed by
//! [`check_launches`].
//!
//! The expected feature rows are computed on the host straight from the trace
//! tokens and the atom table — never from the device state row and never from
//! a twin of the kernel — so a kernel that agrees with its own bookkeeping
//! but not with the grammar still fails.

#![cfg(feature = "backend")]

use mamba3::backend::{Device, check_launches};
use mamba3::backends::Auto;
use mamba3::models::ms2::chem::{ATOM_TYPES, Composition};
use mamba3::models::ms2::completion_data::{ExtractionConfig, PatternSource};
use mamba3::models::ms2::completion_model::{
    CompletionGenerationConfig, CompletionModel, CompletionModelConfig, CompletionRequest,
    CompletionTrainConfig, CompletionTrainer, PatternBatch,
};
use mamba3::models::ms2::grammar::{
    ADD_ATOM, CANONICAL_WORK_LIMIT, CLOSE_RING, Limits, STOP, Token, canonical_trace, replay_exact,
};
use mamba3::models::ms2::graph::MolGraph;
use mamba3::models::ms2::targets_batch::TargetBatch;
use mamba3::nn::Module;
use mamba3::tensor::Tensor;
use mamba3::tensor::ops::index::IdTensor;
use mamba3::tensor::ops::ms2::{
    self, Ms2Constants, PROGRESS_FEATURES, PROGRESS_PATTERN_BASE, PROGRESS_SCALAR_BASE,
    ReplayBuffers, replay_state_width,
};
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

/// The limits the small test config is built for: 16 atoms, 4 closures.
fn limits() -> Limits {
    Limits::new(16, 4).unwrap()
}

/// Kekulized methylbenzene: a ring closure, two elements, a terminal methyl —
/// enough that every feature moves along the trace.
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
    .unwrap()
}

/// Propan-1-ol: a second molecule with an oxygen and no closure.
fn propanol() -> MolGraph {
    MolGraph::new(vec![4, 3, 3, 9], vec![(0, 1, 1), (1, 2, 1), (2, 3, 1)]).unwrap()
}

fn trace_of(graph: &MolGraph) -> (Vec<Token>, Composition) {
    let canonical = canonical_trace(graph, limits(), CANONICAL_WORK_LIMIT).unwrap();
    let composition = graph.composition();
    let end = replay_exact(&canonical.trace, limits(), composition).unwrap();
    assert!(end.stopped() && end.is_complete());
    (canonical.trace, composition)
}

/// `(element, hydrogens, valence)` of an atom type id, from the chemistry
/// table (ids are 1-based; 0 is padding and has no entry).
fn atom_type_row(id: u8) -> (usize, u32, u32) {
    let row = ATOM_TYPES
        .iter()
        .find(|t| t.id == id)
        .unwrap_or_else(|| panic!("atom type {id} is not in ATOM_TYPES"));
    (row.element, u32::from(row.hydrogens), u32::from(row.valence))
}

/// The [`PROGRESS_FEATURES`] words the kernels must produce after applying
/// `prefix`, computed from the tokens and the chemistry table alone.
///
/// `prefix` is a trace prefix including its leading START. `pattern_counts`
/// is the count of each atom type among the supplied pattern atoms.
fn host_progress(
    prefix: &[Token],
    composition: Composition,
    pattern_counts: &[u32; 18],
    atoms: usize,
    max_closures: u32,
    steps: usize,
) -> Vec<f32> {
    let mut used = [0u32; 10];
    let mut types: Vec<u8> = Vec::new();
    let mut residual: Vec<i64> = Vec::new();
    let mut closures = 0u32;
    for (i, token) in prefix.iter().enumerate() {
        match token.kind {
            ADD_ATOM => {
                let (element, hydrogens, valence) = atom_type_row(token.atom_type);
                used[element] += 1;
                used[1] += hydrogens;
                residual.push(i64::from(valence) - i64::from(hydrogens));
                types.push(token.atom_type);
                // The root is the ADD that follows START and nothing else;
                // every later ADD spends its bond on both ends.
                if i != 1 {
                    let n = residual.len() - 1;
                    residual[usize::from(token.pointer)] -= i64::from(token.bond);
                    residual[n] -= i64::from(token.bond);
                }
            }
            CLOSE_RING => {
                let n = residual.len() - 1;
                residual[usize::from(token.pointer)] -= i64::from(token.bond);
                residual[n] -= i64::from(token.bond);
                closures += 1;
            }
            _ => {}
        }
    }
    let ln1p = |x: i64| (1.0f32 + x.max(0) as f32).ln();
    let mut out = vec![0.0f32; PROGRESS_FEATURES];
    for e in 0..10 {
        out[e] = ln1p(i64::from(composition[e]) - i64::from(used[e]));
    }
    for t in 0..18usize {
        let placed = types.iter().filter(|&&ty| usize::from(ty) == t).count();
        out[PROGRESS_PATTERN_BASE + t] =
            ln1p(i64::from(pattern_counts[t]) - placed as i64);
    }
    assert!(types.len() <= atoms, "the prefix placed more atoms than fit");
    let open = residual.iter().filter(|&&r| r != 0).count();
    out[PROGRESS_SCALAR_BASE] = ln1p(types.len() as i64);
    out[PROGRESS_SCALAR_BASE + 1] = ln1p(i64::from(max_closures) - i64::from(closures));
    out[PROGRESS_SCALAR_BASE + 2] = ln1p(steps as i64 - prefix.len() as i64);
    out[PROGRESS_SCALAR_BASE + 3] = ln1p(open as i64);
    out
}

/// A one-row `[rows, 18]` pattern type-count tensor from a host array.
fn counts_tensor(counts: &[u32], rows: usize, device: &Device<R>) -> IdTensor<R> {
    IdTensor::from_slice(counts, vec![rows, 18], device).unwrap()
}

/// The teacher feature block of one trace: `[steps, PROGRESS_FEATURES]` host
/// floats, through `progress_features_teacher`.
fn device_teacher_block(
    trace: &[Token],
    composition: Composition,
    pattern_counts: &[u32; 18],
    atoms: usize,
    max_closures: u32,
    device: &Device<R>,
) -> (Vec<f32>, usize) {
    let constants = Ms2Constants::<R>::new(device);
    let steps = limits().max_steps();
    let mut tokens = vec![0u32; steps * 4];
    for (i, token) in trace.iter().enumerate() {
        tokens[i * 4] = u32::from(token.kind);
        tokens[i * 4 + 1] = u32::from(token.atom_type);
        tokens[i * 4 + 2] = u32::from(token.bond);
        tokens[i * 4 + 3] = u32::from(token.pointer);
    }
    let mut meta = vec![0u32; 12];
    meta[0] = trace.len() as u32;
    meta[1] = 2;
    for e in 0..10 {
        meta[2 + e] = u32::from(composition[e]);
    }
    let tokens = IdTensor::from_slice(&tokens, vec![1, steps, 4], device).unwrap();
    let meta = IdTensor::from_slice(&meta, vec![1, 12], device).unwrap();
    let counts = counts_tensor(pattern_counts, 1, device);
    let mut scratch = IdTensor::empty(vec![1, replay_state_width(atoms)], device);
    let mut block = Tensor::<R, E>::empty(vec![1, steps, PROGRESS_FEATURES], device);
    ms2::progress_features_teacher(
        &tokens,
        &meta,
        &counts,
        &constants,
        &mut scratch,
        &mut block,
        atoms,
        max_closures,
    )
    .unwrap();
    check_launches(device).unwrap();
    (block.try_to_f32().unwrap(), steps)
}

/// The stepping feature row after a trace prefix: `grammar_replay` on the
/// prefix gives the production `[rows, 3A + 16]` state row the sampler keeps,
/// and `progress_features` reads it exactly as the decode loop does.
///
/// Returns the feature row and the state row, so a caller can check the
/// per-atom word layout the features rest on.
fn device_step_row(
    prefix: &[Token],
    composition: Composition,
    pattern_counts: &[u32; 18],
    atoms: usize,
    max_closures: u32,
    device: &Device<R>,
) -> (Vec<f32>, Vec<u32>) {
    let constants = Ms2Constants::<R>::new(device);
    let steps = limits().max_steps();
    let mut tokens = vec![0u32; steps * 4];
    for (i, token) in prefix.iter().enumerate() {
        tokens[i * 4] = u32::from(token.kind);
        tokens[i * 4 + 1] = u32::from(token.atom_type);
        tokens[i * 4 + 2] = u32::from(token.bond);
        tokens[i * 4 + 3] = u32::from(token.pointer);
    }
    let mut meta = vec![0u32; 12];
    meta[0] = prefix.len() as u32;
    meta[1] = 2;
    for e in 0..10 {
        meta[2 + e] = u32::from(composition[e]);
    }
    let tokens_t = IdTensor::from_slice(&tokens, vec![1, steps, 4], device).unwrap();
    let meta_t = IdTensor::from_slice(&meta, vec![1, 12], device).unwrap();
    let buffers = ReplayBuffers::<R>::new(1, steps, atoms, device);
    ms2::grammar_replay(
        &tokens_t,
        &meta_t,
        &constants,
        atoms as u32,
        max_closures,
        &buffers,
    )
    .unwrap();
    check_launches(device).unwrap();
    let atom_steps = buffers.atoms.try_to_vec().unwrap();
    assert_eq!(
        atom_steps[atoms],
        u32::MAX,
        "the prefix must replay legally: first illegal step {}",
        atom_steps[atoms]
    );
    let state = buffers.state.try_to_vec().unwrap();
    // The trajectory metadata the sampler carries: the request composition in
    // words 4..14, the exact-completion flag in word 3.
    let mut traj_meta = vec![0u32; ms2::TRAJ_META_WIDTH];
    traj_meta[3] = 2;
    for e in 0..10 {
        traj_meta[4 + e] = u32::from(composition[e]);
    }
    let traj_meta = IdTensor::from_slice(&traj_meta, vec![1, ms2::TRAJ_META_WIDTH], device).unwrap();
    let counts = counts_tensor(pattern_counts, 1, device);
    let mut row = Tensor::<R, E>::empty(vec![1, PROGRESS_FEATURES], device);
    ms2::progress_features(
        &buffers.state,
        &traj_meta,
        &counts,
        &mut row,
        atoms,
        1,
        max_closures,
        steps,
    )
    .unwrap();
    check_launches(device).unwrap();
    (row.try_to_f32().unwrap(), state)
}

/// The type counts of a pattern set, as [`PatternBatch::type_counts`] builds
/// them, in the fixed-size array the host twin takes.
fn pattern_counts_of(patterns: &[MolGraph], composition: Composition) -> [u32; 18] {
    let refs: Vec<&[MolGraph]> = vec![patterns];
    let batch = PatternBatch::build(&refs, &[composition]).unwrap();
    batch.validate().unwrap();
    let counts = batch.type_counts();
    let mut out = [0u32; 18];
    out.copy_from_slice(&counts[..18]);
    out
}

fn small_config(progress: bool) -> CompletionModelConfig {
    let mut config = CompletionModelConfig::small();
    config.progress_features = progress;
    config
}

fn train_config() -> CompletionTrainConfig {
    let extraction = ExtractionConfig {
        min_patterns: 0,
        max_patterns: 0,
        ..ExtractionConfig::default()
    };
    CompletionTrainConfig {
        lr: 3e-3,
        weight_decay: 0.0,
        grad_clip: Some(1.0),
        seed: 7,
        extraction_seed: 7,
        pattern_source: PatternSource::RandomPatches(extraction.clone()),
        extraction,
        fingerprint_mode: None,
        fingerprint_threshold: 0.1,
    }
}

/// 1. The features are what they claim: along a molecule's canonical trace,
///    at every step, the remaining composition is the target composition
///    minus what the prefix placed, and the unplaced-pattern-type entries are
///    the host-computed bound.
#[test]
fn progress_features_match_the_host_along_a_trace() {
    let _guard = serial();
    let device = dev();
    let atoms = limits().max_atoms();
    let closures = limits().max_closures() as u32;
    for graph in [toluene(), propanol()] {
        let (trace, composition) = trace_of(&graph);
        // A pattern set that really overlaps the answer: the first three
        // atoms of the molecule, induced.
        let pattern = graph.induced(&[0, 1, 2]).unwrap();
        let patterns = vec![pattern];
        let counts = pattern_counts_of(&patterns, composition);
        let (block, steps) = device_teacher_block(
            &trace,
            composition,
            &counts,
            atoms,
            closures,
            &device,
        );
        for position in 0..steps {
            // Position `i` describes the state after tokens `0..=i`; past the
            // trace the kernel holds the final state.
            let taken = (position + 1).min(trace.len());
            let want = host_progress(
                &trace[..taken],
                composition,
                &counts,
                atoms,
                closures,
                steps,
            );
            let got = &block[position * PROGRESS_FEATURES..(position + 1) * PROGRESS_FEATURES];
            for w in 0..PROGRESS_FEATURES {
                assert!(
                    (got[w] - want[w]).abs() <= 1e-6,
                    "position {position} word {w}: device {} host {}",
                    got[w],
                    want[w]
                );
            }
        }
        // The claim the features rest on: per-atom word `j` of the state row
        // is atom `j`'s type id, so placed-atom-type counts come from the
        // same row the sampler already carries.
        let (_, state) = device_step_row(&trace, composition, &counts, atoms, closures, &device);
        let placed: Vec<u8> = trace
            .iter()
            .filter(|token| token.kind == ADD_ATOM)
            .map(|token| token.atom_type)
            .collect();
        for (j, &ty) in placed.iter().enumerate() {
            assert_eq!(
                state[j],
                u32::from(ty),
                "state word {j} should be atom {j}'s type id"
            );
        }
        assert_eq!(state[3 * atoms], placed.len() as u32, "atom count word");
    }
}

/// 2. Host and device agree: the block the teacher path builds equals the row
///    the stepping path builds, step for step, on the same trace.
#[test]
fn teacher_block_and_stepping_row_agree_step_for_step() {
    let _guard = serial();
    let device = dev();
    let atoms = limits().max_atoms();
    let closures = limits().max_closures() as u32;
    let graph = toluene();
    let (trace, composition) = trace_of(&graph);
    let patterns = vec![graph.induced(&[0, 1, 2, 6]).unwrap()];
    let counts = pattern_counts_of(&patterns, composition);
    let (block, _) =
        device_teacher_block(&trace, composition, &counts, atoms, closures, &device);
    // A prefix of length `L` leaves the state the teacher's position `L - 1`
    // describes, so the two must agree word for word there.
    for taken in 1..=trace.len() {
        let (row, _) = device_step_row(
            &trace[..taken],
            composition,
            &counts,
            atoms,
            closures,
            &device,
        );
        let position = taken - 1;
        let want = &block[position * PROGRESS_FEATURES..(position + 1) * PROGRESS_FEATURES];
        for w in 0..PROGRESS_FEATURES {
            assert!(
                (row[w] - want[w]).abs() <= 1e-6,
                "prefix {taken} word {w}: stepping {} teacher {}",
                row[w],
                want[w]
            );
        }
    }
}

/// The unplaced-pattern-type words are a bound, never an over-count: with the
/// whole molecule as its own pattern, every entry falls to 0 by the end, and
/// no entry is ever negative on the way.
#[test]
fn unplaced_pattern_types_are_a_bound_that_reaches_zero() {
    let _guard = serial();
    let device = dev();
    let atoms = limits().max_atoms();
    let closures = limits().max_closures() as u32;
    let graph = propanol();
    let (trace, composition) = trace_of(&graph);
    let patterns = vec![graph.clone()];
    let counts = pattern_counts_of(&patterns, composition);
    let (block, steps) =
        device_teacher_block(&trace, composition, &counts, atoms, closures, &device);
    for position in 0..steps {
        let row = &block[position * PROGRESS_FEATURES..(position + 1) * PROGRESS_FEATURES];
        for t in 0..18 {
            assert!(
                row[PROGRESS_PATTERN_BASE + t] >= 0.0,
                "position {position} type {t} is {} (ln(1 + bound) is never negative)",
                row[PROGRESS_PATTERN_BASE + t]
            );
        }
    }
    // The final position: every pattern atom has a placed atom of its type.
    let last = (trace.len() - 1) * PROGRESS_FEATURES;
    for t in 0..18 {
        assert_eq!(
            block[last + PROGRESS_PATTERN_BASE + t],
            0.0,
            "type {t} should be fully accounted for once the molecule is built"
        );
    }
    // And so does the remaining composition.
    for e in 0..10 {
        assert_eq!(block[last + e], 0.0, "element {e} should be fully placed");
    }
}

/// 3. Off is bit-identical. With the config field off the model owns no
///    projection, names no extra parameter, and a config JSON written before
///    the field existed still loads.
///
/// And the stronger claim the resume rests on: because the projection is
/// zero-initialised and built after every other parameter, a model with the
/// feature *on* gives the same loss and the same sampled traces as the
/// off model at the same seed — which is what makes a non-strict load of an
/// old checkpoint the old model at step 0.
#[test]
fn the_feature_off_and_a_fresh_on_model_agree() {
    let _guard = serial();
    let device = dev();
    // Serde: the field is `#[serde(default)]`, so a JSON without it loads.
    let mut value = serde_json::to_value(CompletionModelConfig::small()).unwrap();
    value.as_object_mut().unwrap().remove("progress_features");
    let reloaded: CompletionModelConfig = serde_json::from_value(value).unwrap();
    assert!(
        !reloaded.progress_features,
        "a config without the field must load with the feature off"
    );
    assert_eq!(reloaded, CompletionModelConfig::small());

    let graph = propanol();
    let (trace, composition) = trace_of(&graph);
    let patterns = vec![graph.induced(&[0, 1]).unwrap()];
    let pattern_refs: Vec<&[MolGraph]> = vec![&patterns];
    let constants = Ms2Constants::<R>::new(&device);
    let batch = PatternBatch::build(&pattern_refs, &[composition]).unwrap();
    let targets =
        TargetBatch::build_exact(&[&trace], &[composition], limits()).unwrap();

    let mut losses = Vec::new();
    let mut traces = Vec::new();
    let mut names = Vec::new();
    for progress in [false, true] {
        let mut rng = Rng::seeded(11);
        let model =
            CompletionModel::<R, E>::init(&small_config(progress), &device, &mut rng).unwrap();
        assert_eq!(model.has_progress(), progress);
        let keys: Vec<String> = model
            .state_dict()
            .entries
            .keys()
            .filter(|k| k.starts_with("progress"))
            .cloned()
            .collect();
        assert_eq!(
            keys.is_empty(),
            !progress,
            "progress parameters present: {keys:?}"
        );
        names.push(keys);
        let (_, loss) = model
            .teacher(&batch, &targets, &constants, &device)
            .unwrap();
        check_launches(&device).unwrap();
        losses.push(loss.try_to_f32().unwrap()[0]);
        let request = CompletionRequest {
            id: 5,
            composition,
            patterns: &patterns,
            acceptance_patterns: None,
            fingerprint: None,
        };
        let config = CompletionGenerationConfig {
            trajectories: 8,
            returned: 4,
            seed: 3,
            ..CompletionGenerationConfig::default()
        };
        let outcomes = model
            .generate(&[request], &config, &constants, &device)
            .unwrap();
        check_launches(&device).unwrap();
        traces.push(
            outcomes[0]
                .sampled
                .iter()
                .map(|t| t.trace.clone())
                .collect::<Vec<_>>(),
        );
    }
    assert_eq!(
        losses[0], losses[1],
        "the zero-initialised projection must add exactly nothing"
    );
    assert_eq!(
        traces[0], traces[1],
        "the sampled traces must be the off model's"
    );
    assert_eq!(names[1].len(), 1, "one new parameter: {:?}", names[1]);
}

/// 4. Gradients reach the new parameter: finite, and not all zero.
#[test]
fn gradients_reach_the_progress_projection() {
    let _guard = serial();
    let device = dev();
    let graph = toluene();
    let (trace, composition) = trace_of(&graph);
    let patterns = vec![graph.induced(&[0, 1, 2]).unwrap()];
    let pattern_refs: Vec<&[MolGraph]> = vec![&patterns];
    let constants = Ms2Constants::<R>::new(&device);
    let mut rng = Rng::seeded(13);
    let model = CompletionModel::<R, E>::init(&small_config(true), &device, &mut rng).unwrap();
    let batch = PatternBatch::build(&pattern_refs, &[composition]).unwrap();
    let targets = TargetBatch::build_exact(&[&trace], &[composition], limits()).unwrap();
    let (_, loss) = model
        .teacher(&batch, &targets, &constants, &device)
        .unwrap();
    let grads = loss.backward().unwrap();
    check_launches(&device).unwrap();
    let named = model.named_parameters();
    let (name, param) = named
        .iter()
        .find(|(name, _)| name.starts_with("progress"))
        .expect("the model owns a progress parameter");
    let grad = grads
        .get(param.id())
        .unwrap_or_else(|| panic!("no gradient reached `{name}`"));
    let values = grad.try_to_f32().unwrap();
    assert_eq!(
        values.len(),
        PROGRESS_FEATURES * small_config(true).d_model as usize,
        "`{name}` should be the [32, d] projection"
    );
    assert!(
        values.iter().all(|v| v.is_finite()),
        "`{name}` has a non-finite gradient"
    );
    assert!(
        values.iter().any(|v| *v != 0.0),
        "`{name}` has an all-zero gradient"
    );
}

/// 5. The features carry information: trained briefly on a handful of
///    molecules at the same seed and step count, the model with them reaches
///    a lower teacher loss than the model without.
#[test]
fn the_features_lower_the_teacher_loss() {
    let _guard = serial();
    let device = dev();
    // Five molecules whose scaffolds really are most of the answer, each
    // trained with its own first atoms as the supplied pattern — the regime
    // the feature is for.
    let graphs = vec![
        toluene(),
        propanol(),
        MolGraph::new(vec![4, 3, 3, 3, 9], vec![
            (0, 1, 1),
            (1, 2, 1),
            (2, 3, 1),
            (3, 4, 1),
        ])
        .unwrap(),
        MolGraph::new(vec![4, 2, 4, 9], vec![(0, 1, 1), (1, 2, 1), (1, 3, 1)]).unwrap(),
        MolGraph::new(vec![3, 3, 3], vec![(0, 1, 1), (1, 2, 1), (2, 0, 1)]).unwrap(),
    ];
    let mut traces = Vec::new();
    let mut compositions = Vec::new();
    let mut owned_patterns = Vec::new();
    for graph in &graphs {
        let (trace, composition) = trace_of(graph);
        let take: Vec<usize> = (0..graph.atoms().len().saturating_sub(1)).collect();
        owned_patterns.push(vec![graph.induced(&take).unwrap()]);
        traces.push(trace);
        compositions.push(composition);
    }
    let pattern_refs: Vec<&[MolGraph]> =
        owned_patterns.iter().map(Vec::as_slice).collect();
    let trace_refs: Vec<&[Token]> = traces.iter().map(Vec::as_slice).collect();

    let mut final_loss = Vec::new();
    for progress in [false, true] {
        let mut trainer = CompletionTrainer::<R, E>::new(
            &small_config(progress),
            &train_config(),
            &device,
        )
        .unwrap();
        for _ in 0..120 {
            trainer
                .step_with(&pattern_refs, &trace_refs, &compositions)
                .unwrap();
        }
        check_launches(&device).unwrap();
        let nll = trainer
            .teacher_eval_with(&pattern_refs, &trace_refs, &compositions)
            .unwrap();
        check_launches(&device).unwrap();
        let mean = nll.iter().sum::<f32>() / nll.len() as f32;
        assert!(mean.is_finite(), "the teacher loss must stay finite");
        final_loss.push(mean);
    }
    // The result, not a hope: if the feature does not help here, this fails
    // and says by how much.
    assert!(
        final_loss[1] < final_loss[0],
        "progress features on: {} nats, off: {} nats (the feature should help)",
        final_loss[1],
        final_loss[0]
    );
}

/// The beam path takes the same conditioning as the sampler: an on model
/// searches and returns candidates, with no launch left unaccounted for.
#[test]
fn the_beam_path_runs_with_the_progress_conditioning() {
    let _guard = serial();
    let device = dev();
    let graph = propanol();
    let (_, composition) = trace_of(&graph);
    let patterns = vec![graph.induced(&[0, 1]).unwrap()];
    let constants = Ms2Constants::<R>::new(&device);
    let mut rng = Rng::seeded(17);
    let model = CompletionModel::<R, E>::init(&small_config(true), &device, &mut rng).unwrap();
    let request = CompletionRequest {
        id: 9,
        composition,
        patterns: &patterns,
        acceptance_patterns: None,
        fingerprint: None,
    };
    let config = CompletionGenerationConfig {
        trajectories: 16,
        returned: 8,
        seed: 2,
        ..CompletionGenerationConfig::default()
    };
    let (outcomes, _) = model
        .generate_beam(&[request], &config, 16, &constants, &device)
        .unwrap();
    check_launches(&device).unwrap();
    assert_eq!(outcomes.len(), 1);
    assert!(
        outcomes[0].trajectories > 0,
        "the search should have expanded rows"
    );
    for candidate in &outcomes[0].candidates {
        let end = replay_exact(&candidate.trace, limits(), composition).unwrap();
        assert!(
            end.stopped() && end.is_complete(),
            "every returned candidate is a complete molecule on the composition"
        );
    }
    // STOP is the only kind that ends a returned trace.
    for candidate in &outcomes[0].candidates {
        assert_eq!(candidate.trace.last().map(|t| t.kind), Some(STOP));
    }
}
