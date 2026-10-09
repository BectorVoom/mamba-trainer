//! Pattern-atom capacity of the substructure encoder
//! ([`PATTERN_SLOTS`], raised from the 24 the shipped checkpoints were
//! trained at to the decoder's own `max_atoms` of 32).
//!
//! Hand-built molecules only (atom type ids from
//! [`chem::ATOM_TYPES`](mamba3::models::ms2::chem::ATOM_TYPES)): ethanol and
//! two fused-ring skeletons of 32 and 33 carbons, standing in for the Murcko
//! scaffolds the driver supplies. Every device call is followed by
//! [`check_launches`]. Tolerances are stated per assertion; the capacity
//! test asserts bit equality, because that is what "an existing checkpoint
//! keeps its behaviour" means.

#![cfg(feature = "backend")]

use mamba3::backend::{Device, check_launches};
use mamba3::backends::Auto;
use mamba3::models::ms2::chem::Composition;
use mamba3::models::ms2::completion_model::{
    CompletionModel, CompletionModelConfig, PATTERN_SLOTS, PatternBatch,
};
use mamba3::models::ms2::grammar::{
    CANONICAL_WORK_LIMIT, Limits, Token, canonical_trace, replay_exact,
};
use mamba3::models::ms2::graph::MolGraph;
use mamba3::models::ms2::targets_batch::TargetBatch;
use mamba3::tensor::ops::ms2::Ms2Constants;
use mamba3::tensor::ops::random::Rng;

type R = Auto;
type E = f32;

/// File-local serial lock: the launch counters [`check_launches`] reads are
/// process-global, so the whole binary runs serially.
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

fn ethanol() -> MolGraph {
    MolGraph::new(vec![4, 3, 9], vec![(0, 1, 1), (1, 2, 1)]).unwrap()
}

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

/// A saturated carbon skeleton of `n` atoms: a single chain closed into a
/// ring by one extra bond, so every atom carries a type id and a bond and the
/// graph is connected, as an induced scaffold is.
///
/// Atom type 3 is `C` with two hydrogens, which has the residual valence a
/// chain interior atom needs; the ring closure is what makes the shape a
/// stand-in for a scaffold rather than a loose chain.
fn ring_of(n: usize) -> MolGraph {
    let atoms = vec![3u8; n];
    let mut bonds: Vec<(usize, usize, u8)> = (0..n - 1).map(|i| (i, i + 1, 1)).collect();
    bonds.push((n - 1, 0, 1));
    MolGraph::new(atoms, bonds).unwrap()
}

#[test]
fn thirty_two_atoms_fit_and_thirty_three_do_not() {
    let _lock = serial();
    assert_eq!(
        PATTERN_SLOTS, 32,
        "the capacity this suite pins is the decoder's own max_atoms"
    );
    let comp = ethanol().composition();
    // Exactly the capacity: accepted, every slot valid, no padding left.
    let full = vec![ring_of(PATTERN_SLOTS)];
    let batch = PatternBatch::build(&[full.as_slice()], &[comp]).unwrap();
    batch.validate().unwrap();
    assert_eq!(batch.slots, PATTERN_SLOTS);
    assert_eq!(batch.types.len(), PATTERN_SLOTS);
    assert!(
        batch.valid.iter().all(|&v| v == 1.0),
        "a pattern of {PATTERN_SLOTS} atoms leaves no padding slot"
    );
    assert!(batch.types.iter().all(|&t| t == 3));
    // The ring's bonds: the chain plus the closure, each stored twice.
    let ones: f32 = batch.adjacency.iter().sum();
    assert_eq!(ones, 2.0 * PATTERN_SLOTS as f32, "one bond per atom, mirrored");
    // The type counts the progress features read see all of them.
    let counts = batch.type_counts();
    assert_eq!(counts[3], PATTERN_SLOTS as u32);
    assert_eq!(counts.iter().sum::<u32>(), PATTERN_SLOTS as u32);
    // One atom past the capacity: `Error::Config` naming the total and the
    // limit, not a panic and not a silent truncation.
    let over = vec![ring_of(PATTERN_SLOTS + 1)];
    let err = PatternBatch::build(&[over.as_slice()], &[comp]).unwrap_err();
    assert!(
        matches!(err, mamba3::error::Error::Config(_)),
        "a pattern past the capacity is Error::Config: {err}"
    );
    let text = err.to_string();
    assert!(
        text.contains(&format!("{} pattern atoms", PATTERN_SLOTS + 1))
            && text.contains(&format!("limit of {PATTERN_SLOTS} slots")),
        "the refusal names the offending total and the limit: {text}"
    );
    // A width outside `1..=PATTERN_SLOTS` is refused at the batch, so no
    // caller can build a batch the encoder cannot upload.
    assert!(
        PatternBatch::build_with_slots(&[full.as_slice()], &[comp], PATTERN_SLOTS + 1).is_err()
    );
    assert!(PatternBatch::build_with_slots(&[full.as_slice()], &[comp], 0).is_err());
}

#[test]
fn a_full_width_pattern_encodes_into_the_memory() {
    let _lock = serial();
    let device = dev();
    let mut rng = Rng::seeded(71);
    let model: CompletionModel<R, E> =
        CompletionModel::init(&CompletionModelConfig::small(), &device, &mut rng).unwrap();
    let comp = ethanol().composition();
    let full = vec![ring_of(PATTERN_SLOTS)];
    let batch = PatternBatch::build(&[full.as_slice()], &[comp]).unwrap();
    let out = model.encode(&batch, &device).unwrap();
    check_launches(&device).unwrap();
    // `[context; pattern atoms]`, every pattern slot unmasked.
    let d = 64usize;
    assert_eq!(out.memory.dims(), [1, 1 + PATTERN_SLOTS, d]);
    let mask = out.memory_mask.to_f32();
    assert_eq!(mask.len(), 1 + PATTERN_SLOTS);
    assert!(mask.iter().all(|&v| v == 1.0), "no slot is masked: {mask:?}");
    // Every atom row carries signal: the widened slots are encoded, not
    // merely allocated.
    let x = out.x.try_to_f32().unwrap();
    assert_eq!(x.len(), PATTERN_SLOTS * d);
    for slot in 0..PATTERN_SLOTS {
        let row = &x[slot * d..(slot + 1) * d];
        assert!(
            row.iter().all(|v| v.is_finite()) && row.iter().any(|&v| v != 0.0),
            "slot {slot} is finite and non-zero"
        );
    }
    // The pool is the composition row plus the clamped mean over all
    // `PATTERN_SLOTS` valid rows, so it differs from the formula-only pool.
    let pool = out.pool.try_to_f32().unwrap();
    let context = out.context.try_to_f32().unwrap();
    assert!(pool.iter().all(|v| v.is_finite()));
    assert_ne!(pool, context, "a full-width pattern set moves the pool");
    // The teacher pass runs over the wider memory.
    let (trace, target_comp) = trace_and_composition(&ethanol());
    let targets = TargetBatch::build_exact(&[trace.as_slice()], &[target_comp], limits()).unwrap();
    let constants = Ms2Constants::new(&device);
    let (_, loss) = model
        .teacher(&batch, &targets, &constants, &device)
        .unwrap();
    check_launches(&device).unwrap();
    let value = loss.try_to_f32().unwrap()[0];
    assert!(value.is_finite() && value > 0.0, "teacher loss {value}");
}

#[test]
fn encoding_is_invariant_to_pattern_order_at_full_width() {
    let _lock = serial();
    let device = dev();
    let mut rng = Rng::seeded(72);
    let model: CompletionModel<R, E> =
        CompletionModel::init(&CompletionModelConfig::small(), &device, &mut rng).unwrap();
    let comp = ethanol().composition();
    // Three patterns filling the capacity exactly (12 + 14 + 6 = 32), so the
    // permutations below move atoms across the whole width.
    let pats = vec![ring_of(12), ring_of(14), ring_of(6)];
    let total: usize = pats.iter().map(|g| g.atoms().len()).sum();
    assert_eq!(total, PATTERN_SLOTS);
    // Reversed pattern order, and each pattern's atoms rotated by one: the
    // encoder has no pattern-index and no position embedding, so neither
    // may reach the output.
    let rotated = |g: &MolGraph| {
        let n = g.atoms().len();
        let perm: Vec<usize> = (0..n).map(|i| (i + 1) % n).collect();
        g.permuted(&perm).unwrap()
    };
    let permuted = vec![rotated(&pats[2]), rotated(&pats[1]), rotated(&pats[0])];
    let batch_a = PatternBatch::build(&[pats.as_slice()], &[comp]).unwrap();
    let batch_b = PatternBatch::build(&[permuted.as_slice()], &[comp]).unwrap();
    let out_a = model.encode(&batch_a, &device).unwrap();
    let out_b = model.encode(&batch_b, &device).unwrap();
    check_launches(&device).unwrap();
    let close = |a: &[f32], b: &[f32], tol: f32, what: &str| {
        assert_eq!(a.len(), b.len(), "{what}: lengths");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).abs() <= tol * x.abs().max(1.0),
                "{what}[{i}]: {x} vs {y}"
            );
        }
    };
    close(
        &out_a.pool.try_to_f32().unwrap(),
        &out_b.pool.try_to_f32().unwrap(),
        1e-5,
        "pool",
    );
    close(
        &out_a.context.try_to_f32().unwrap(),
        &out_b.context.try_to_f32().unwrap(),
        1e-5,
        "context",
    );
    // The atom rows as a multiset: sorted lexicographically, since the
    // encoder is equivariant to the atom order, not invariant.
    let sorted_rows = |x: &[f32], valid: &[f32]| {
        let d = 64usize;
        let mut rows: Vec<Vec<f32>> = Vec::new();
        for (slot, &v) in valid.iter().enumerate() {
            if v != 0.0 {
                rows.push(x[slot * d..(slot + 1) * d].to_vec());
            }
        }
        rows.sort_by(|a, b| a.partial_cmp(b).expect("finite encoder rows"));
        rows.concat()
    };
    let x_a = out_a.x.try_to_f32().unwrap();
    let x_b = out_b.x.try_to_f32().unwrap();
    close(
        &sorted_rows(&x_a, &batch_a.valid),
        &sorted_rows(&x_b, &batch_b.valid),
        1e-4,
        "valid x rows as a multiset",
    );
    // And the teacher NLL of the query, which is what training sees.
    let (trace, target_comp) = trace_and_composition(&ethanol());
    let targets = TargetBatch::build_exact(&[trace.as_slice()], &[target_comp], limits()).unwrap();
    let constants = Ms2Constants::new(&device);
    let (tout_a, _) = model
        .teacher(&batch_a, &targets, &constants, &device)
        .unwrap();
    let (tout_b, _) = model
        .teacher(&batch_b, &targets, &constants, &device)
        .unwrap();
    check_launches(&device).unwrap();
    close(
        &tout_a.nll.try_to_f32().unwrap(),
        &tout_b.nll.try_to_f32().unwrap(),
        1e-4,
        "teacher NLL",
    );
}

#[test]
fn the_old_capacity_is_bit_identical_at_the_new_one() {
    let _lock = serial();
    let device = dev();
    let mut rng = Rng::seeded(73);
    let model: CompletionModel<R, E> =
        CompletionModel::init(&CompletionModelConfig::small(), &device, &mut rng).unwrap();
    let comp = ethanol().composition();
    // A pattern set that fits the 24 slots the shipped checkpoints were
    // trained at, laid out at both widths. Nothing learned is shaped by the
    // width, so the slots the wider layout adds are padding: selected to
    // exact zeros after every round and masked out of the decoder memory.
    let old = 24usize;
    let pats = vec![ring_of(10), ring_of(9), ring_of(5)];
    let total: usize = pats.iter().map(|g| g.atoms().len()).sum();
    assert_eq!(total, old);
    let narrow = PatternBatch::build_with_slots(&[pats.as_slice()], &[comp], old).unwrap();
    let wide = PatternBatch::build(&[pats.as_slice()], &[comp]).unwrap();
    narrow.validate().unwrap();
    wide.validate().unwrap();
    assert_eq!(narrow.slots, old);
    assert_eq!(wide.slots, PATTERN_SLOTS);
    // The host arrays agree where they overlap; `type_counts` does not see
    // the width at all, which is what keeps the progress features identical.
    assert_eq!(narrow.types, wide.types[..old]);
    assert_eq!(narrow.open, wide.open[..old]);
    assert_eq!(narrow.valid, wide.valid[..old]);
    assert_eq!(narrow.features, wide.features);
    assert_eq!(narrow.type_counts(), wide.type_counts());
    let out_n = model.encode(&narrow, &device).unwrap();
    let out_w = model.encode(&wide, &device).unwrap();
    check_launches(&device).unwrap();
    // Bit equality, not a tolerance: the same weights see the same values in
    // the same order, and the extra slots contribute nothing.
    assert_eq!(
        out_n.pool.try_to_f32().unwrap(),
        out_w.pool.try_to_f32().unwrap(),
        "pool"
    );
    assert_eq!(
        out_n.context.try_to_f32().unwrap(),
        out_w.context.try_to_f32().unwrap(),
        "context"
    );
    let d = 64usize;
    let x_n = out_n.x.try_to_f32().unwrap();
    let x_w = out_w.x.try_to_f32().unwrap();
    assert_eq!(x_n.len(), old * d);
    assert_eq!(x_w.len(), PATTERN_SLOTS * d);
    assert_eq!(x_n, x_w[..old * d], "the occupied atom rows");
    assert!(
        x_w[old * d..].iter().all(|&v| v == 0.0),
        "the slots the wider layout adds are exact zeros"
    );
    // The memory keeps `[context; pattern atoms]`, so the wider one is the
    // narrower one plus masked zero slots.
    let mem_n = out_n.memory.try_to_f32().unwrap();
    let mem_w = out_w.memory.try_to_f32().unwrap();
    let kept = (1 + old) * d;
    assert_eq!(mem_n.len(), kept);
    assert_eq!(mem_n, mem_w[..kept], "the occupied memory slots");
    assert!(mem_w[kept..].iter().all(|&v| v == 0.0));
    let mask_n = out_n.memory_mask.to_f32();
    let mask_w = out_w.memory_mask.to_f32();
    assert_eq!(mask_n, mask_w[..1 + old]);
    assert!(mask_w[1 + old..].iter().all(|&v| v == 0.0));
    // The teacher NLL, which is what a resumed run's first step sees.
    let (trace, target_comp) = trace_and_composition(&ethanol());
    let targets = TargetBatch::build_exact(&[trace.as_slice()], &[target_comp], limits()).unwrap();
    let constants = Ms2Constants::new(&device);
    let (tout_n, loss_n) = model
        .teacher(&narrow, &targets, &constants, &device)
        .unwrap();
    let (tout_w, loss_w) = model.teacher(&wide, &targets, &constants, &device).unwrap();
    check_launches(&device).unwrap();
    assert_eq!(
        tout_n.nll.try_to_f32().unwrap(),
        tout_w.nll.try_to_f32().unwrap(),
        "teacher NLL at the old capacity is unchanged by the new one"
    );
    assert_eq!(
        loss_n.try_to_f32().unwrap(),
        loss_w.try_to_f32().unwrap(),
        "teacher loss"
    );
}
