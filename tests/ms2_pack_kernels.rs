//! K3 kernel-versus-twin tests for ranking, compaction and the packed output.
//!
//! Every device call runs on poisoned outputs, is followed by
//! [`check_launches`], and is compared element-for-element with the host twin
//! lanes (and, through [`assemble`], with [`pack`] on the same data): a
//! dropped launch (stale poison) or a wrong word fails. Sizes stay small on
//! the CPU runtime; the supervisor runs the same file on wgpu.

#![cfg(feature = "backend")]

use mamba3::backend::{Device, check_launches};
use mamba3::backends::Auto;
use mamba3::error::Error;
use mamba3::models::ms2::contract::{
    CandidateBatch, NO_FORMULA, SCHEMA_VERSION, candidate_status, request_status,
};
use mamba3::models::ms2::pack::{
    EVIDENCE_STRIDE, ScoreKind, TRAJ_FORMULA_STRIDE, WF, assemble, pack, pack_evidence_lane,
    pack_from_device_layout, record_width, scores_fill_lane,
};
use mamba3::tensor::Tensor;
use mamba3::tensor::ops::index::IdTensor;
use mamba3::tensor::ops::ms2_pack;

type R = Auto;
type E = f32;

const T: usize = 6;
const A: usize = 8;
const RMAX: usize = 4;

fn dev() -> Device<R> {
    Device::<R>::default()
}

fn upload_ids(data: &[u32], shape: Vec<usize>, device: &Device<R>) -> IdTensor<R> {
    IdTensor::from_slice(data, shape, device).unwrap()
}

fn upload_f(data: &[f32], shape: Vec<usize>, device: &Device<R>) -> Tensor<R, E> {
    Tensor::<R, f32>::from_f32(data, shape, device).unwrap()
}

fn assert_ids(actual: &[u32], expected: &[u32], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_eq!(*a, *e, "{what}: word {i} differs");
    }
}

fn assert_f32_bits(actual: &[f32], expected: &[f32], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_eq!(a.to_bits(), e.to_bits(), "{what}: word {i} differs");
    }
}

// ---------------------------------------------------------------------------
// Batch builder (same legal traces as the host tests)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum Tmpl {
    Fin,
    Unfin,
    Trunc,
    Fail,
}

#[derive(Clone, Copy)]
struct Rec {
    tmpl: Tmpl,
    extra_status: u32,
    flp: f32,
    tlp: f32,
    formula: bool,
}

fn fin(flp: f32, tlp: f32) -> Rec {
    Rec { tmpl: Tmpl::Fin, extra_status: 0, flp, tlp, formula: true }
}

fn tokens_of(tmpl: Tmpl) -> (Vec<u32>, u32, Vec<u8>, u32) {
    match tmpl {
        Tmpl::Fin => {
            let mut tok = vec![0u32; T * 4];
            tok[0..4].copy_from_slice(&[1, 0, 0, 0]);
            tok[4..8].copy_from_slice(&[2, 4, 0, 0]);
            tok[8..12].copy_from_slice(&[4, 0, 0, 0]);
            let mut open = vec![0u8; A];
            open[0] = 1;
            (tok, 3, open, candidate_status::FINISHED)
        }
        Tmpl::Unfin => {
            let mut tok = vec![0u32; T * 4];
            tok[0..4].copy_from_slice(&[1, 0, 0, 0]);
            tok[4..8].copy_from_slice(&[2, 4, 0, 0]);
            (tok, 2, vec![0u8; A], 0)
        }
        Tmpl::Trunc => {
            let mut tok = vec![0u32; T * 4];
            tok[0..4].copy_from_slice(&[1, 0, 0, 0]);
            tok[4..8].copy_from_slice(&[2, 1, 0, 0]);
            for (i, parent) in [0u32, 1, 2, 3].iter().enumerate() {
                let base = (i + 2) * 4;
                tok[base..base + 4].copy_from_slice(&[2, 1, 1, *parent]);
            }
            (tok, T as u32, vec![0u8; A], candidate_status::TRUNCATED)
        }
        Tmpl::Fail => (
            vec![0u32; T * 4],
            0,
            vec![0u8; A],
            candidate_status::REQUEST_FAILED,
        ),
    }
}

fn make_batch(b: usize, k: usize, recs: &[Rec], req: &[u32]) -> CandidateBatch {
    let n = b * k;
    let mut actions = vec![0u32; n * T * 4];
    let mut length = vec![0u32; n];
    let mut status = vec![0u32; n];
    let mut open_valence = vec![0u8; n * A];
    let mut flp = vec![0.0f32; n];
    let mut tlp = vec![0.0f32; n];
    let mut formula_row = vec![NO_FORMULA; n];
    let mut formula_rank = vec![NO_FORMULA; n];
    let mut formula_counts = vec![0u16; n * 10];
    let mut spectrum_id = vec![0u64; n];
    let mut trajectory = vec![0u32; n];
    for (r, rec) in recs.iter().enumerate() {
        let bb = r / k;
        let kk = r % k;
        let (tok, len, open, base_status) = tokens_of(rec.tmpl);
        actions[r * T * 4..(r + 1) * T * 4].copy_from_slice(&tok);
        length[r] = len;
        status[r] = base_status | rec.extra_status;
        open_valence[r * A..(r + 1) * A].copy_from_slice(&open);
        flp[r] = rec.flp;
        tlp[r] = rec.tlp;
        spectrum_id[r] = 9000 + bb as u64;
        trajectory[r] = kk as u32;
        if rec.formula {
            formula_rank[r] = (r % 3) as u32;
            formula_row[r] = 7;
            let counts = [4u16, 8, 2, 1, 0, 0, 0, 0, 0, 0];
            formula_counts[r * 10..(r + 1) * 10].copy_from_slice(&counts);
        }
    }
    CandidateBatch {
        schema_version: SCHEMA_VERSION,
        batch: b,
        trajectories: k,
        max_steps: T,
        max_atoms: A,
        max_ring_closures: RMAX,
        spectrum_id,
        trajectory,
        actions,
        length,
        formula_row,
        formula_log_prob: flp,
        trace_log_prob: tlp,
        open_valence,
        attachment_partition: vec![0; n],
        status,
        evidence_status: vec![0; n],
        evidence_count: vec![0; n],
        evidence_peak_id: vec![0; (n) * 4],
        evidence_hypothesis: vec![0; (n) * 4],
        evidence_shift: vec![0; (n) * 4],
        evidence_residual: vec![0; (n) * 4],
        evidence_log_prob: vec![0.0; (n) * 4],
        identity_resolution: vec![0; n],
        request_status: req.to_vec(),
        rows_visited: vec![3; b],
        rows_joined: vec![3; b],
        rows_scored: vec![3; b],
        formula_support_complete: vec![1; b],
        formula_mass_retained: vec![0.7; b],
        peaks_kept: vec![5; b],
        intensity_retained: vec![0.9; b],
        formula_counts,
        formula_source: vec![0; b],
        formula_rank,
    }
}

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 11
    }

    fn below(&mut self, m: u64) -> u64 {
        self.next() % m
    }
}

/// Device-layout host buffers for one batch, exactly as `pack` builds them.
struct DeviceLayout {
    /// Trajectory records `[rows, T*4 + A + 4]`.
    actions: Vec<u32>,
    /// Formula slots `[B*K*12]`.
    traj: Vec<u32>,
    /// Log-probability pairs `[rows*2]`.
    scores: Vec<f32>,
    /// Caller scores `[rows]`.
    rerank: Vec<f32>,
    /// Evidence words `[rows*18]`.
    evidence: Vec<u32>,
    /// Identity pairs `[rows*2]`.
    identity: Vec<u32>,
}

/// Device-layout host buffers for one batch, exactly as `pack` builds them:
///
/// `actions [rows, S]`, `traj_formula [B*K*12]`, `scores [rows*2]`,
/// `rerank [rows]`, `evidence [rows*18]`, `identity [rows*2]`.
#[allow(clippy::too_many_arguments)]
fn device_layout(
    batch: &CandidateBatch,
    rng: &mut Lcg,
    use_rerank: bool,
) -> DeviceLayout {
    let b = batch.batch;
    let k = batch.trajectories;
    let n = b * k;
    let stride = T * 4 + A + 4;
    let mut actions = vec![0u32; n * stride];
    for r in 0..n {
        let abase = r * stride;
        for w in 0..T * 4 {
            actions[abase + w] = batch.actions[r * T * 4 + w];
        }
        for v in 0..A {
            actions[abase + T * 4 + v] = u32::from(batch.open_valence[r * A + v]);
        }
        actions[abase + T * 4 + A] = batch.length[r];
        actions[abase + T * 4 + A + 1] = batch.status[r];
        actions[abase + T * 4 + A + 2] = batch.trace_log_prob[r].to_bits();
        actions[abase + T * 4 + A + 3] = batch.formula_row[r];
    }
    let mut traj = vec![0u32; n * TRAJ_FORMULA_STRIDE as usize];
    for r in 0..n {
        traj[r * 12] = batch.formula_rank[r];
        traj[r * 12 + 1] = batch.formula_row[r];
        for e in 0..10 {
            traj[r * 12 + 2 + e] = u32::from(batch.formula_counts[r * 10 + e]);
        }
    }
    let mut scores = vec![0.0f32; n * 2];
    for r in 0..n {
        scores[r * 2] = batch.trace_log_prob[r];
        scores[r * 2 + 1] = batch.formula_log_prob[r];
    }
    let mut rerank = vec![0.0f32; n];
    if use_rerank {
        for v in rerank.iter_mut() {
            *v = if rng.below(16) == 0 {
                f32::NAN
            } else {
                rng.below(9) as f32 - 4.0
            };
        }
    }
    // Word 0 is the evidence status; the other 17 words are random to prove
    // the lanes ignore them.
    let mut evidence = vec![0u32; n * EVIDENCE_STRIDE as usize];
    for r in 0..n {
        evidence[r * 18] = u32::from(batch.evidence_status[r]);
        for w in 1..18 {
            evidence[r * 18 + w] = rng.next() as u32;
        }
    }
    // Identity words; the resolution follows the pack rule (0 without
    // identity, else 1, or 2 when unresolved).
    let mut identity = vec![0u32; n * 2];
    for r in 0..n {
        let mut bits = 0u32;
        if rng.below(4) == 0 {
            bits |= 1 << 7;
        }
        if rng.below(4) == 0 {
            bits |= 1 << 8;
        }
        identity[r * 2] = bits;
        identity[r * 2 + 1] = if bits & (1 << 8) != 0 { 2 } else { 1 };
    }
    DeviceLayout { actions, traj, scores, rerank, evidence, identity }
}

fn random_recs(rng: &mut Lcg, b: usize, k: usize, fatal_b: Option<usize>) -> Vec<Rec> {
    let mut recs = Vec::with_capacity(b * k);
    for r in 0..b * k {
        if Some(r / k) == fatal_b {
            recs.push(Rec { tmpl: Tmpl::Fail, ..fin(0.0, 0.0) });
            continue;
        }
        let tmpl = match rng.below(10) {
            0 | 1 => Tmpl::Unfin,
            2 => Tmpl::Trunc,
            _ => Tmpl::Fin,
        };
        let mut extra = 0u32;
        if tmpl == Tmpl::Fin {
            match rng.below(8) {
                0 => extra |= candidate_status::INVALID_FINAL,
                1 => extra |= candidate_status::DUPLICATE_TRACE,
                _ => {}
            }
        }
        let mut flp = -0.5 * rng.below(12) as f32;
        let mut tlp = -0.5 * rng.below(12) as f32;
        if rng.below(20) == 0 {
            if rng.below(2) == 0 {
                flp = f32::NAN;
            } else {
                tlp = f32::NAN;
            }
        }
        recs.push(Rec { tmpl, extra_status: extra, flp, tlp,
            // No trajectory starts without a formula (finding R1-C6).
            formula: rng.below(4) != 0 || tmpl == Tmpl::Fin });
    }
    recs
}

#[allow(clippy::too_many_lines)]
fn check_config(
    b: usize,
    k: usize,
    rr: usize,
    use_identity: bool,
    use_rerank: bool,
    seed: u64,
    what: &str,
) {
    let device = dev();
    let rows = b * k;
    let stride = T * 4 + A + 4;
    let width = record_width(T, A);
    let mut rng = Lcg(seed);
    let fatal_b = if rng.below(4) == 0 { Some(rng.below(b as u64) as usize) } else { None };
    let recs = random_recs(&mut rng, b, k, fatal_b);
    let mut req = vec![0u32; b];
    if let Some(fb) = fatal_b {
        req[fb] = request_status::EMPTY_SPECTRUM;
    }
    let batch = make_batch(b, k, &recs, &req);
    let layout = device_layout(&batch, &mut rng, use_rerank);
    let actions = layout.actions;
    let traj = layout.traj;
    let scores = layout.scores;
    let rerank = layout.rerank;
    let evidence = layout.evidence;
    let identity = layout.identity;
    let use_graph = u32::from(use_identity);
    let use_rr = u32::from(use_rerank);

    // Twin pipeline over the same host buffers.
    let words = pack_from_device_layout(
        &actions,
        stride,
        &traj,
        &evidence,
        &identity,
        use_graph,
        &batch.trace_log_prob,
        &batch.formula_log_prob,
        &rerank,
        use_rr,
        b,
        k,
        T as u32,
        A as u32,
        rr,
    )
    .unwrap();
    let want_packed = words.packed;
    let want_packed_f = words.packed_f;
    let want_counts = words.returned_count;
    let want_ranks = words.ranks;

    // Device: rank on a poisoned output.
    let actions_t = upload_ids(&actions, vec![rows, stride], &device);
    let identity_t = upload_ids(&identity, vec![rows, 2], &device);
    let scores_t = upload_f(&scores, vec![rows, 2], &device);
    let rerank_t = upload_f(&rerank, vec![rows], &device);
    let mut rank_t = upload_ids(&vec![0xDEAD_BEEF; rows], vec![rows], &device);
    ms2_pack::rank(
        &actions_t,
        &identity_t,
        &scores_t,
        &rerank_t,
        &mut rank_t,
        T,
        A as u32,
        k,
        use_graph,
        use_rr,
    )
    .unwrap();
    check_launches(&device).unwrap();
    assert_ids(&rank_t.try_to_vec().unwrap(), &want_ranks, &format!("{what} rank"));

    // Device: integer + float records on poisoned outputs.
    let traj_t = upload_ids(&traj, vec![b, k, 12], &device);
    let evidence_t = upload_ids(&evidence, vec![rows, 18], &device);
    let mut record_t =
        upload_ids(&vec![0xDEAD_BEEF; rows * width], vec![rows, width], &device);
    ms2_pack::record_pack(
        &actions_t,
        &traj_t,
        &evidence_t,
        &identity_t,
        &mut record_t,
        T,
        A as u32,
        k,
        use_graph,
    )
    .unwrap();
    check_launches(&device).unwrap();
    let mut record_f_t = upload_f(&vec![f32::NAN; rows * WF], vec![rows, WF], &device);
    ms2_pack::record_pack_f(&scores_t, &rerank_t, &mut record_f_t, use_rr).unwrap();
    check_launches(&device).unwrap();
    // Element-wise check against the twin gather lane per record.
    let mut exp_rec = vec![0u32; rows * width];
    let mut exp_rec_f = vec![0.0f32; rows * WF];
    for r in 0..rows {
        mamba3::models::ms2::pack::record_pack_lane(
            &actions,
            stride as u32,
            T as u32,
            A as u32,
            &traj,
            &evidence,
            EVIDENCE_STRIDE,
            &identity,
            use_graph,
            &batch.trace_log_prob,
            &batch.formula_log_prob,
            &rerank,
            use_rr,
            r as u32,
            (r / k) as u32,
            (r % k) as u32,
            &mut exp_rec,
            (r * width) as u32,
            &mut exp_rec_f,
            (r * WF) as u32,
        );
    }
    assert_ids(
        &record_t.try_to_vec().unwrap(),
        &exp_rec,
        &format!("{what} record"),
    );
    assert_f32_bits(
        &record_f_t.try_to_f32().unwrap(),
        &exp_rec_f,
        &format!("{what} record_f"),
    );

    // Device: pack on poisoned outputs.
    let rank_in_t = upload_ids(&want_ranks, vec![rows], &device);
    let mut packed_t =
        upload_ids(&vec![0xDEAD_BEEF; b * rr * width], vec![b, rr, width], &device);
    let mut packed_f_t = upload_f(&vec![f32::NAN; b * rr * WF], vec![b, rr, WF], &device);
    let mut counts_t = upload_ids(&vec![0xDEAD_BEEF; b], vec![b], &device);
    ms2_pack::pack(
        &rank_in_t,
        &record_t,
        &record_f_t,
        &mut packed_t,
        &mut packed_f_t,
        &mut counts_t,
        T,
        A as u32,
        k,
        rr,
    )
    .unwrap();
    check_launches(&device).unwrap();
    assert_ids(
        &packed_t.try_to_vec().unwrap(),
        &want_packed,
        &format!("{what} packed"),
    );
    assert_f32_bits(
        &packed_f_t.try_to_f32().unwrap(),
        &want_packed_f,
        &format!("{what} packed_f"),
    );
    assert_ids(
        &counts_t.try_to_vec().unwrap(),
        &want_counts,
        &format!("{what} counts"),
    );

    // The device words assemble to exactly `pack` on the same data (the
    // reranker path only when the device ran it too).
    let assembled = assemble(&batch, &want_packed, &want_packed_f, &want_counts, rr).unwrap();
    assembled.validate().unwrap();
    let score_kind = if use_rerank {
        ScoreKind::Reranker(&rerank)
    } else {
        ScoreKind::Raw
    };
    let id_bits_owned: Vec<u32> = identity.iter().step_by(2).copied().collect();
    let expected = pack(
        &batch,
        if use_identity { Some(&id_bits_owned) } else { None },
        score_kind,
        rr,
    )
    .unwrap();
    assert_eq!(assembled, expected, "{what}: assembled != pack");
}

#[test]
fn pack_kernels_match_twin() {
    let mut seed = 0x009a_c077_u64;
    for b in [1usize, 3] {
        for k in [1usize, 4, 8] {
            for r in [1usize, 0] {
                let rr = if r == 0 { k } else { 1 };
                for use_identity in [false, true] {
                    for use_rerank in [false, true] {
                        seed += 1;
                        check_config(
                            b,
                            k,
                            rr,
                            use_identity,
                            use_rerank,
                            seed,
                            &format!("B{b} K{k} R{rr} id{use_identity} rr{use_rerank}"),
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn pack_kernel_failed_request_returns_zero() {
    let device = dev();
    let (b, k, rr) = (2usize, 3usize, 3usize);
    let rows = b * k;
    let stride = T * 4 + A + 4;
    let width = record_width(T, A);
    let recs = vec![
        Rec { tmpl: Tmpl::Fail, ..fin(0.0, 0.0) },
        Rec { tmpl: Tmpl::Fail, ..fin(0.0, 0.0) },
        Rec { tmpl: Tmpl::Fail, ..fin(0.0, 0.0) },
        fin(-0.1, -0.1),
        fin(-0.2, -0.2),
        fin(-0.3, -0.3),
    ];
    let fatal = request_status::EMPTY_SPECTRUM;
    let batch = make_batch(b, k, &recs, &[fatal, 0]);
    let mut rng = Lcg(42);
    let layout = device_layout(&batch, &mut rng, false);
    let actions = layout.actions;
    let traj = layout.traj;
    let scores = layout.scores;
    let rerank = layout.rerank;
    let evidence = layout.evidence;
    // Deterministic identity (all zero): no graph-duplicate exclusion, so the
    // failed spectrum returns 0 and the live one returns all 3.
    let identity = vec![0u32; rows * 2];
    let actions_t = upload_ids(&actions, vec![rows, stride], &device);
    let identity_t = upload_ids(&identity, vec![rows, 2], &device);
    let scores_t = upload_f(&scores, vec![rows, 2], &device);
    let rerank_t = upload_f(&rerank, vec![rows], &device);
    let mut rank_t = upload_ids(&vec![0xDEAD_BEEF; rows], vec![rows], &device);
    ms2_pack::rank(&actions_t, &identity_t, &scores_t, &rerank_t, &mut rank_t, T, A as u32, k, 1, 0)
        .unwrap();
    check_launches(&device).unwrap();
    let traj_t = upload_ids(&traj, vec![b, k, 12], &device);
    let evidence_t = upload_ids(&evidence, vec![rows, 18], &device);
    let mut record_t =
        upload_ids(&vec![0xDEAD_BEEF; rows * width], vec![rows, width], &device);
    ms2_pack::record_pack(
        &actions_t,
        &traj_t,
        &evidence_t,
        &identity_t,
        &mut record_t,
        T,
        A as u32,
        k,
        1,
    )
    .unwrap();
    check_launches(&device).unwrap();
    let mut record_f_t = upload_f(&vec![f32::NAN; rows * WF], vec![rows, WF], &device);
    ms2_pack::record_pack_f(&scores_t, &rerank_t, &mut record_f_t, 0).unwrap();
    check_launches(&device).unwrap();
    let mut packed_t =
        upload_ids(&vec![0xDEAD_BEEF; b * rr * width], vec![b, rr, width], &device);
    let mut packed_f_t = upload_f(&vec![f32::NAN; b * rr * WF], vec![b, rr, WF], &device);
    let mut counts_t = upload_ids(&vec![0xDEAD_BEEF; b], vec![b], &device);
    ms2_pack::pack(
        &rank_t,
        &record_t,
        &record_f_t,
        &mut packed_t,
        &mut packed_f_t,
        &mut counts_t,
        T,
        A as u32,
        k,
        rr,
    )
    .unwrap();
    check_launches(&device).unwrap();
    assert_ids(&counts_t.try_to_vec().unwrap(), &[0, 3], "failed counts");
    let assembled = assemble(
        &batch,
        &packed_t.try_to_vec().unwrap(),
        &packed_f_t.try_to_f32().unwrap(),
        &counts_t.try_to_vec().unwrap(),
        rr,
    )
    .unwrap();
    assembled.validate().unwrap();
}

#[test]
fn pack_kernels_reject_bad_shapes() {
    let device = dev();
    let (b, k, rr) = (1usize, 2usize, 2usize);
    let rows = b * k;
    let stride = T * 4 + A + 4;
    let width = record_width(T, A);
    let recs = vec![fin(-0.1, -0.1), fin(-0.2, -0.2)];
    let batch = make_batch(b, k, &recs, &[0]);
    let mut rng = Lcg(7);
    let layout = device_layout(&batch, &mut rng, false);
    let actions = layout.actions;
    let traj = layout.traj;
    let scores = layout.scores;
    let rerank = layout.rerank;
    let evidence = layout.evidence;
    let identity = layout.identity;
    let actions_t = upload_ids(&actions, vec![rows, stride], &device);
    let identity_t = upload_ids(&identity, vec![rows, 2], &device);
    let scores_t = upload_f(&scores, vec![rows, 2], &device);
    let rerank_t = upload_f(&rerank, vec![rows], &device);
    let mut rank_t = upload_ids(&vec![0u32; rows], vec![rows], &device);
    // Wrong rank on `actions`.
    let flat = upload_ids(&actions, vec![rows * stride], &device);
    assert!(matches!(
        ms2_pack::rank(&flat, &identity_t, &scores_t, &rerank_t, &mut rank_t, T, A as u32, k, 1, 0),
        Err(Error::Shape(_))
    ));
    // Wrong last dimension on `identity`.
    let bad_id = upload_ids(&vec![0u32; rows * 3], vec![rows, 3], &device);
    assert!(matches!(
        ms2_pack::rank(&actions_t, &bad_id, &scores_t, &rerank_t, &mut rank_t, T, A as u32, k, 1, 0),
        Err(Error::Shape(_))
    ));
    // `per_spectrum` that does not divide the rows.
    assert!(matches!(
        ms2_pack::rank(&actions_t, &identity_t, &scores_t, &rerank_t, &mut rank_t, T, A as u32, 3, 1, 0),
        Err(Error::Shape(_))
    ));
    // Wrong `traj_formula` batch.
    let traj_t = upload_ids(&traj, vec![b, k, 12], &device);
    let bad_traj = upload_ids(&vec![0u32; (b + 1) * k * 12], vec![b + 1, k, 12], &device);
    let evidence_t = upload_ids(&evidence, vec![rows, 18], &device);
    let mut record_t = upload_ids(&vec![0u32; rows * width], vec![rows, width], &device);
    assert!(matches!(
        ms2_pack::record_pack(&actions_t, &bad_traj, &evidence_t, &identity_t, &mut record_t, T, A as u32, k, 1),
        Err(Error::Shape(_))
    ));
    // Wrong float width on `record_f`.
    let mut bad_f = upload_f(&vec![0.0f32; rows * 2], vec![rows, 2], &device);
    assert!(matches!(
        ms2_pack::record_pack_f(&scores_t, &rerank_t, &mut bad_f, 0),
        Err(Error::Shape(_))
    ));
    // Wrong packed width.
    let rank_in = upload_ids(&vec![0u32; rows], vec![rows], &device);
    ms2_pack::record_pack(&actions_t, &traj_t, &evidence_t, &identity_t, &mut record_t, T, A as u32, k, 1)
        .unwrap();
    let record_f = upload_f(&vec![0.0f32; rows * WF], vec![rows, WF], &device);
    let mut bad_packed = upload_ids(&vec![0u32; b * rr * (width + 1)], vec![b, rr, width + 1], &device);
    let mut packed_f = upload_f(&vec![0.0f32; b * rr * WF], vec![b, rr, WF], &device);
    let mut counts = upload_ids(&vec![0u32; b], vec![b], &device);
    assert!(matches!(
        ms2_pack::pack(&rank_in, &record_t, &record_f, &mut bad_packed, &mut packed_f, &mut counts, T, A as u32, k, rr),
        Err(Error::Shape(_))
    ));
    // `returned == 0` is refused.
    let mut packed = upload_ids(&vec![0u32; b * rr * width], vec![b, rr, width], &device);
    assert!(matches!(
        ms2_pack::pack(&rank_in, &record_t, &record_f, &mut packed, &mut packed_f, &mut counts, T, A as u32, k, 0),
        Err(Error::Shape(_))
    ));
}

// ---------------------------------------------------------------------------
// Finding E3: the validated score domain, on the device rank path
// ---------------------------------------------------------------------------

#[test]
fn rank_excludes_out_of_domain_scores() {
    // Finite-extreme, infinite, NaN and an overflowing sum of two in-domain
    // terms (2e38 + 2e38 is +inf) are all ineligible on the device, exactly
    // as the host twin says. Record 5 isolates the reranker rule: its raw
    // terms are in-domain but its caller score is out of it. Every output
    // starts poisoned, so a dropped lane would fail the comparison.
    let device = dev();
    let (b, k, rr) = (1usize, 6usize, 6usize);
    let rows = b * k;
    let stride = T * 4 + A + 4;
    let width = record_width(T, A);
    let recs = vec![
        fin(-0.1, -0.1),
        Rec { flp: 3.1e38, ..fin(-0.1, -0.1) },
        Rec { tlp: f32::INFINITY, ..fin(-0.1, -0.1) },
        Rec { flp: f32::NAN, ..fin(-0.1, -0.1) },
        Rec { flp: 2.0e38, tlp: 2.0e38, ..fin(-0.1, -0.1) },
        Rec { flp: -0.2, tlp: -0.1, ..fin(-0.1, -0.1) },
    ];
    let batch = make_batch(b, k, &recs, &[0]);
    let mut rng = Lcg(1234);
    let layout = device_layout(&batch, &mut rng, false);
    // Deterministic identity (all zero): no graph-duplicate exclusion.
    let identity = vec![0u32; rows * 2];
    let rerank_raw = vec![0.0f32; rows];
    let rerank_bad = vec![-0.5f32, -0.6, -0.7, -0.8, -0.9, 3.1e38f32];

    // Twin expectations, raw mode first: record 0 scores −0.2 (rank 0),
    // record 5 scores −0.3 (rank 1), the rest are ineligible.
    let twin_raw = pack_from_device_layout(
        &layout.actions,
        stride,
        &layout.traj,
        &layout.evidence,
        &identity,
        1,
        &batch.trace_log_prob,
        &batch.formula_log_prob,
        &rerank_raw,
        0,
        b,
        k,
        T as u32,
        A as u32,
        rr,
    )
    .unwrap();
    assert_ids(
        &twin_raw.ranks,
        &[0, NO_FORMULA, NO_FORMULA, NO_FORMULA, NO_FORMULA, 1],
        "twin raw ranks",
    );
    // Reranker mode: record 5's caller score is out of domain, so it stays
    // out; record 4's caller score is in-domain (its overflowing raw sum is
    // not the ranking score here, and both raw terms are in-domain), so it
    // ranks second.
    let twin_rr = pack_from_device_layout(
        &layout.actions,
        stride,
        &layout.traj,
        &layout.evidence,
        &identity,
        1,
        &batch.trace_log_prob,
        &batch.formula_log_prob,
        &rerank_bad,
        1,
        b,
        k,
        T as u32,
        A as u32,
        rr,
    )
    .unwrap();
    assert_ids(
        &twin_rr.ranks,
        &[0, NO_FORMULA, NO_FORMULA, NO_FORMULA, 1, NO_FORMULA],
        "twin reranker ranks",
    );

    let actions_t = upload_ids(&layout.actions, vec![rows, stride], &device);
    let identity_t = upload_ids(&identity, vec![rows, 2], &device);
    let scores_t = upload_f(&layout.scores, vec![rows, 2], &device);
    let rerank_raw_t = upload_f(&rerank_raw, vec![rows], &device);
    let rerank_bad_t = upload_f(&rerank_bad, vec![rows], &device);
    let mut rank_t = upload_ids(&vec![0xDEAD_BEEF; rows], vec![rows], &device);
    ms2_pack::rank(
        &actions_t,
        &identity_t,
        &scores_t,
        &rerank_raw_t,
        &mut rank_t,
        T,
        A as u32,
        k,
        1,
        0,
    )
    .unwrap();
    check_launches(&device).unwrap();
    assert_ids(&rank_t.try_to_vec().unwrap(), &twin_raw.ranks, "device raw ranks");
    let mut rank_rr_t = upload_ids(&vec![0xDEAD_BEEF; rows], vec![rows], &device);
    ms2_pack::rank(
        &actions_t,
        &identity_t,
        &scores_t,
        &rerank_bad_t,
        &mut rank_rr_t,
        T,
        A as u32,
        k,
        1,
        1,
    )
    .unwrap();
    check_launches(&device).unwrap();
    assert_ids(
        &rank_rr_t.try_to_vec().unwrap(),
        &twin_rr.ranks,
        "device reranker ranks",
    );

    // The raw-mode words pack to two filled slots and validate.
    let traj_t = upload_ids(&layout.traj, vec![b, k, 12], &device);
    let evidence_t = upload_ids(&layout.evidence, vec![rows, 18], &device);
    let mut record_t =
        upload_ids(&vec![0xDEAD_BEEF; rows * width], vec![rows, width], &device);
    ms2_pack::record_pack(
        &actions_t,
        &traj_t,
        &evidence_t,
        &identity_t,
        &mut record_t,
        T,
        A as u32,
        k,
        1,
    )
    .unwrap();
    check_launches(&device).unwrap();
    let mut record_f_t = upload_f(&vec![f32::NAN; rows * WF], vec![rows, WF], &device);
    ms2_pack::record_pack_f(&scores_t, &rerank_raw_t, &mut record_f_t, 0).unwrap();
    check_launches(&device).unwrap();
    let mut packed_t =
        upload_ids(&vec![0xDEAD_BEEF; b * rr * width], vec![b, rr, width], &device);
    let mut packed_f_t = upload_f(&vec![f32::NAN; b * rr * WF], vec![b, rr, WF], &device);
    let mut counts_t = upload_ids(&vec![0xDEAD_BEEF; b], vec![b], &device);
    ms2_pack::pack(
        &rank_t,
        &record_t,
        &record_f_t,
        &mut packed_t,
        &mut packed_f_t,
        &mut counts_t,
        T,
        A as u32,
        k,
        rr,
    )
    .unwrap();
    check_launches(&device).unwrap();
    assert_ids(&counts_t.try_to_vec().unwrap(), &[2], "out-of-domain counts");
    let assembled = assemble(
        &batch,
        &packed_t.try_to_vec().unwrap(),
        &packed_f_t.try_to_f32().unwrap(),
        &counts_t.try_to_vec().unwrap(),
        rr,
    )
    .unwrap();
    assembled.validate().unwrap();
    assert_eq!(assembled.trajectory[0], 0);
    assert_eq!(assembled.trajectory[1], 5);
}

// ---------------------------------------------------------------------------
// Finding E4: `returned > per_spectrum` is refused; `returned > eligible`
// stays legal
// ---------------------------------------------------------------------------

#[test]
fn pack_rejects_returned_above_per_spectrum() {
    // B = 1, K = 1, R = 2 with matching output shapes is refused with
    // `Error::Config` before any launch, so the poisoned outputs survive.
    let device = dev();
    let (b, k, rr) = (1usize, 1usize, 2usize);
    let rows = b * k;
    let stride = T * 4 + A + 4;
    let width = record_width(T, A);
    let batch = make_batch(b, k, &[fin(-0.1, -0.1)], &[0]);
    let mut rng = Lcg(99);
    let layout = device_layout(&batch, &mut rng, false);
    let identity = vec![0u32; rows * 2];
    let actions_t = upload_ids(&layout.actions, vec![rows, stride], &device);
    let identity_t = upload_ids(&identity, vec![rows, 2], &device);
    let scores_t = upload_f(&layout.scores, vec![rows, 2], &device);
    let rerank_t = upload_f(&layout.rerank, vec![rows], &device);
    let mut rank_t = upload_ids(&vec![0xDEAD_BEEF; rows], vec![rows], &device);
    ms2_pack::rank(&actions_t, &identity_t, &scores_t, &rerank_t, &mut rank_t, T, A as u32, k, 1, 0)
        .unwrap();
    check_launches(&device).unwrap();
    let traj_t = upload_ids(&layout.traj, vec![b, k, 12], &device);
    let evidence_t = upload_ids(&layout.evidence, vec![rows, 18], &device);
    let mut record_t =
        upload_ids(&vec![0xDEAD_BEEF; rows * width], vec![rows, width], &device);
    ms2_pack::record_pack(
        &actions_t,
        &traj_t,
        &evidence_t,
        &identity_t,
        &mut record_t,
        T,
        A as u32,
        k,
        1,
    )
    .unwrap();
    check_launches(&device).unwrap();
    let mut record_f_t = upload_f(&vec![f32::NAN; rows * WF], vec![rows, WF], &device);
    ms2_pack::record_pack_f(&scores_t, &rerank_t, &mut record_f_t, 0).unwrap();
    check_launches(&device).unwrap();
    let mut packed_t =
        upload_ids(&vec![0xDEAD_BEEF; b * rr * width], vec![b, rr, width], &device);
    let mut packed_f_t = upload_f(&vec![f32::NAN; b * rr * WF], vec![b, rr, WF], &device);
    let mut counts_t = upload_ids(&vec![0xDEAD_BEEF; b], vec![b], &device);
    assert!(matches!(
        ms2_pack::pack(
            &rank_t,
            &record_t,
            &record_f_t,
            &mut packed_t,
            &mut packed_f_t,
            &mut counts_t,
            T,
            A as u32,
            k,
            rr,
        ),
        Err(Error::Config(_))
    ));
    check_launches(&device).unwrap();
    assert_ids(
        &counts_t.try_to_vec().unwrap(),
        &[0xDEAD_BEEF],
        "refused pack launches nothing",
    );
}

#[test]
fn pack_accepts_returned_above_eligible() {
    // The legal side of finding E4, on the device: R = 2 > eligible = 1
    // with K = 2 launches and leaves the second slot unfilled.
    let device = dev();
    let (b, k, rr) = (1usize, 2usize, 2usize);
    let rows = b * k;
    let stride = T * 4 + A + 4;
    let width = record_width(T, A);
    let recs = vec![fin(-0.1, -0.1), Rec { tmpl: Tmpl::Unfin, ..fin(0.0, 0.0) }];
    let batch = make_batch(b, k, &recs, &[0]);
    let mut rng = Lcg(100);
    let layout = device_layout(&batch, &mut rng, false);
    let identity = vec![0u32; rows * 2];
    let actions_t = upload_ids(&layout.actions, vec![rows, stride], &device);
    let identity_t = upload_ids(&identity, vec![rows, 2], &device);
    let scores_t = upload_f(&layout.scores, vec![rows, 2], &device);
    let rerank_t = upload_f(&layout.rerank, vec![rows], &device);
    let mut rank_t = upload_ids(&vec![0xDEAD_BEEF; rows], vec![rows], &device);
    ms2_pack::rank(&actions_t, &identity_t, &scores_t, &rerank_t, &mut rank_t, T, A as u32, k, 1, 0)
        .unwrap();
    check_launches(&device).unwrap();
    let traj_t = upload_ids(&layout.traj, vec![b, k, 12], &device);
    let evidence_t = upload_ids(&layout.evidence, vec![rows, 18], &device);
    let mut record_t =
        upload_ids(&vec![0xDEAD_BEEF; rows * width], vec![rows, width], &device);
    ms2_pack::record_pack(
        &actions_t,
        &traj_t,
        &evidence_t,
        &identity_t,
        &mut record_t,
        T,
        A as u32,
        k,
        1,
    )
    .unwrap();
    check_launches(&device).unwrap();
    let mut record_f_t = upload_f(&vec![f32::NAN; rows * WF], vec![rows, WF], &device);
    ms2_pack::record_pack_f(&scores_t, &rerank_t, &mut record_f_t, 0).unwrap();
    check_launches(&device).unwrap();
    let mut packed_t =
        upload_ids(&vec![0xDEAD_BEEF; b * rr * width], vec![b, rr, width], &device);
    let mut packed_f_t = upload_f(&vec![f32::NAN; b * rr * WF], vec![b, rr, WF], &device);
    let mut counts_t = upload_ids(&vec![0xDEAD_BEEF; b], vec![b], &device);
    ms2_pack::pack(
        &rank_t,
        &record_t,
        &record_f_t,
        &mut packed_t,
        &mut packed_f_t,
        &mut counts_t,
        T,
        A as u32,
        k,
        rr,
    )
    .unwrap();
    check_launches(&device).unwrap();
    assert_ids(&counts_t.try_to_vec().unwrap(), &[1], "one eligible of two");
    let assembled = assemble(
        &batch,
        &packed_t.try_to_vec().unwrap(),
        &packed_f_t.try_to_f32().unwrap(),
        &counts_t.try_to_vec().unwrap(),
        rr,
    )
    .unwrap();
    assembled.validate().unwrap();
    assert_eq!(assembled.trajectory, vec![0, NO_FORMULA]);
}

#[test]
fn scores_fill_matches_twin() {
    // The ranking-score fill against `scores_fill_lane`: trace
    // log-probabilities from the record bits, formula log-probabilities
    // gathered by the trajectory's retained slot (0.0 when there is none,
    // including an out-of-range slot).
    let device = dev();
    let (b, k, f) = (2usize, 3usize, 2usize);
    let rows = b * k;
    let stride = T * 4 + A + 4;
    let len_field = (T * 4 + A) as u32;
    let trace_lp = [-0.5f32, -1.0, -1.5, -2.0, -2.5, -3.0];
    let mut actions = vec![0u32; rows * stride];
    for r in 0..rows {
        actions[r * stride + len_field as usize + 2] = trace_lp[r].to_bits();
    }
    let mut traj = vec![0u32; rows * 12];
    let slots = [0u32, 1, u32::MAX, 1, 0, 7];
    for (r, &s) in slots.iter().enumerate() {
        traj[r * 12] = s;
    }
    let top_lp = [-0.1f32, -0.2, -0.3, -0.4];
    let mut want = vec![0.0f32; rows * 2];
    for r in 0..rows {
        let (tl, fl) = scores_fill_lane(
            &actions,
            stride as u32,
            len_field,
            &traj,
            &top_lp,
            f as u32,
            r as u32,
            k as u32,
        );
        want[r * 2] = tl;
        want[r * 2 + 1] = fl;
    }
    assert_eq!(want[5 * 2 + 1].to_bits(), 0.0f32.to_bits(), "out-of-range slot reads 0.0");
    assert_eq!(want[2 * 2 + 1].to_bits(), 0.0f32.to_bits(), "sentinel slot reads 0.0");
    let actions_t = upload_ids(&actions, vec![rows, stride], &device);
    let traj_t = upload_ids(&traj, vec![b, k, 12], &device);
    let lp_t = upload_f(&top_lp, vec![b, f], &device);
    let mut scores_t =
        upload_f(&vec![f32::NAN; rows * 2], vec![rows, 2], &device);
    ms2_pack::scores_fill(&actions_t, &traj_t, &lp_t, &mut scores_t, T, A as u32, k, f)
        .unwrap();
    check_launches(&device).unwrap();
    assert_f32_bits(&scores_t.try_to_f32().unwrap(), &want, "scores_fill");
}

#[test]
fn scores_fill_rejects_bad_shapes() {
    let device = dev();
    let stride = T * 4 + A + 4;
    let actions_t = upload_ids(&vec![0u32; 2 * stride], vec![2, stride], &device);
    let traj_t = upload_ids(&vec![0u32; 2 * 12], vec![1, 2, 12], &device);
    let lp_t = upload_f(&vec![0.0f32; 2], vec![1, 2], &device);
    let mut scores_t = upload_f(&vec![0.0f32; 4], vec![2, 2], &device);
    // per_spectrum 0 is refused before any launch.
    assert!(
        ms2_pack::scores_fill(&actions_t, &traj_t, &lp_t, &mut scores_t, T, A as u32, 0, 2)
            .is_err()
    );
    // Wrong scores width is refused before any launch.
    let mut bad = upload_f(&vec![0.0f32; 6], vec![2, 3], &device);
    assert!(
        ms2_pack::scores_fill(&actions_t, &traj_t, &lp_t, &mut bad, T, A as u32, 2, 2).is_err()
    );
}

// ---------------------------------------------------------------------------
// `ms2_pack_evidence` (I3b) against its twin
// ---------------------------------------------------------------------------

/// Randomised `pack_evidence` fixtures: ranks with eligible and ineligible
/// trajectories, evidence rows with counts below, at and above `E = 4`
/// (statuses 0/1/2 with and without bit 7), formula slots inside and outside
/// `F`, and log-probabilities spanning the classes. The kernel runs on
/// poisoned outputs and every packed word is compared with the twin lane.
#[test]
fn pack_evidence_matches_twin_on_poisoned_outputs() {
    let device = dev();
    for seed in [21u64, 22, 23] {
        let mut rng = Lcg(seed);
        let (b, k, rr, f, n, j) = (2usize, 4usize, 3usize, 2usize, 5usize, 3usize);
        let rows = b * k;
        let slots = b * rr;
        let width = j + 1;
        let mut ranks = vec![u32::MAX; rows];
        for t in 0..rows {
            if rng.below(4) != 0 {
                ranks[t] = rng.below(k as u64) as u32;
            }
        }
        let mut evidence = vec![0u32; rows * EVIDENCE_STRIDE as usize];
        for t in 0..rows {
            let base = t * EVIDENCE_STRIDE as usize;
            let base_status = [0u32, 1, 2][rng.below(3) as usize];
            let incomplete = rng.below(3) == 0;
            evidence[base] = base_status + if incomplete { 128 } else { 0 };
            let count = rng.below(7) as u32; // 0..=6: below, at and above E
            evidence[base + 1] = count;
            for q in 0..count.min(4) as usize {
                evidence[base + 2 + q * 4] = rng.below(n as u64) as u32;
                evidence[base + 2 + q * 4 + 1] = rng.below(j as u64) as u32;
                evidence[base + 2 + q * 4 + 2] = rng.next() as u32;
                evidence[base + 2 + q * 4 + 3] = rng.next() as u32;
            }
        }
        let mut traj_slot = vec![0u32; rows * 2];
        for t in 0..rows {
            traj_slot[t * 2] = if rng.below(6) == 0 {
                u32::MAX // invalid slot: the lane clamps to 0
            } else {
                rng.below(f as u64) as u32
            };
            traj_slot[t * 2 + 1] = 1;
        }
        let mut log_prob = vec![0.0f32; b * f * n * width];
        for v in log_prob.iter_mut() {
            *v = -(rng.below(800) as f32) / 100.0;
        }
        // Twin lanes.
        let mut want_ev = vec![0u32; slots * EVIDENCE_STRIDE as usize];
        let mut want_evf = vec![0.0f32; slots * 4];
        for s in 0..slots {
            pack_evidence_lane(
                &ranks,
                &evidence,
                &traj_slot,
                &log_prob,
                f as u32,
                n as u32,
                j as u32,
                (s / rr) as u32,
                (s % rr) as u32,
                k as u32,
                &mut want_ev,
                (s * EVIDENCE_STRIDE as usize) as u32,
                &mut want_evf,
                (s * 4) as u32,
            );
        }
        // Kernel on poisoned outputs.
        let rank_t = upload_ids(&ranks, vec![rows], &device);
        let ev_t = upload_ids(&evidence, vec![rows, EVIDENCE_STRIDE as usize], &device);
        let slot_t = upload_ids(&traj_slot, vec![rows, 2], &device);
        let lp_t = upload_f(&log_prob, vec![b, f, n, width], &device);
        let mut packed_t = upload_ids(
            &vec![0xDEAD_BEEF; slots * EVIDENCE_STRIDE as usize],
            vec![slots, EVIDENCE_STRIDE as usize],
            &device,
        );
        let mut packed_f_t = upload_f(&vec![f32::NAN; slots * 4], vec![slots, 4], &device);
        ms2_pack::pack_evidence(
            &rank_t,
            &ev_t,
            &slot_t,
            &lp_t,
            &mut packed_t,
            &mut packed_f_t,
            f,
            n,
            j,
            k,
            rr,
        )
        .unwrap();
        check_launches(&device).unwrap();
        assert_ids(
            &packed_t.try_to_vec().unwrap(),
            &want_ev,
            &format!("seed {seed}: packed evidence rows"),
        );
        assert_f32_bits(
            &packed_f_t.try_to_f32().unwrap(),
            &want_evf,
            &format!("seed {seed}: packed evidence log-probs"),
        );
    }
}

/// Launch-error checks for `pack_evidence`: shape mismatches are refused
/// before any launch, and `returned > per_spectrum` is `Error::Config`.
#[test]
fn pack_evidence_rejects_bad_shapes() {
    let device = dev();
    let (b, k, rr, f, n, j) = (1usize, 2usize, 2usize, 2usize, 3usize, 2usize);
    let rows = b * k;
    let width = j + 1;
    let rank_t = upload_ids(&vec![0u32; rows], vec![rows], &device);
    let ev_t = upload_ids(&vec![0u32; rows * 18], vec![rows, 18], &device);
    let slot_t = upload_ids(&vec![0u32; rows * 2], vec![rows, 2], &device);
    let lp_t = upload_f(&vec![0.0f32; b * f * n * width], vec![b, f, n, width], &device);
    let mut packed_t = upload_ids(&vec![0u32; b * rr * 18], vec![b * rr, 18], &device);
    let mut packed_f_t = upload_f(&vec![0.0f32; b * rr * 4], vec![b * rr, 4], &device);
    // `returned > per_spectrum` is refused.
    assert!(matches!(
        ms2_pack::pack_evidence(&rank_t, &ev_t, &slot_t, &lp_t, &mut packed_t, &mut packed_f_t, f, n, j, k, k + 1),
        Err(Error::Config(_))
    ));
    // Evidence width must be 18.
    let bad_ev = upload_ids(&vec![0u32; rows * 17], vec![rows, 17], &device);
    assert!(ms2_pack::pack_evidence(&rank_t, &bad_ev, &slot_t, &lp_t, &mut packed_t, &mut packed_f_t, f, n, j, k, rr).is_err());
    // Log-probability width must be J + 1.
    let bad_lp = upload_f(&vec![0.0f32; b * f * n * j], vec![b, f, n, j], &device);
    assert!(ms2_pack::pack_evidence(&rank_t, &ev_t, &slot_t, &bad_lp, &mut packed_t, &mut packed_f_t, f, n, j, k, rr).is_err());
    // Packed float width must be 4.
    let mut bad_pf = upload_f(&vec![0.0f32; b * rr * 3], vec![b * rr, 3], &device);
    assert!(ms2_pack::pack_evidence(&rank_t, &ev_t, &slot_t, &lp_t, &mut packed_t, &mut bad_pf, f, n, j, k, rr).is_err());
}

#[test]
fn bf16_ranking_uses_f32_sums_and_f32_scores() {
    // Finding R1-C4: the ranking score is the f32 sum of the f32-widened
    // terms on every neural dtype, equal to the host `pack`'s arithmetic,
    // and the packed score buffer is f32. The reviewer's near-tie (formula
    // -0.69140625, trace -1.0078125: host sum -1.69921875, bf16 sum
    // -1.703125) ranks as the host does: the later candidate first. Every
    // packed field equals host `pack`.
    use half::bf16;
    let device = dev();
    let rows = 2usize;
    let stride = T * 4 + A + 4;
    let len_field = T * 4 + A;
    // FINISHED status words; lengths unused by rank/record_pack_f.
    let mut actions = vec![0u32; rows * stride];
    for r in 0..rows {
        actions[r * stride + len_field + 1] = candidate_status::FINISHED;
    }
    let identity = vec![0u32; rows * 2];
    // (trace, formula) per record: traj 0 scores exactly -1.703125, traj 1
    // is the near-tie at host -1.69921875. All four terms are exactly
    // representable in bf16.
    let tlp = [-0.703125f32, -1.0078125f32];
    let flp = [-1.0f32, -0.69140625f32];
    assert_eq!(tlp[0] + flp[0], -1.703125f32, "traj 0 host sum");
    assert_eq!(tlp[1] + flp[1], -1.69921875f32, "traj 1 host sum (near-tie)");
    let scores_f32 = vec![tlp[0], flp[0], tlp[1], flp[1]];
    for &v in &scores_f32 {
        assert_eq!(bf16::from_f32(v).to_f32(), v, "term {v} is bf16-exact");
    }
    // The gathered ranking terms live in an f32 buffer on every neural
    // dtype (spec §4.4); `E = bf16` only affects the caller-score (`rerank`)
    // buffer here.
    let scores_t =
        Tensor::<R, f32>::from_f32(&scores_f32, vec![rows, 2], &device).unwrap();
    let rerank_t = match Tensor::<R, bf16>::from_f32(&vec![0.0f32; rows], vec![rows], &device) {
        Ok(t) => t,
        Err(Error::Unsupported(msg)) => {
            assert!(
                msg.contains("bf16"),
                "the bf16 refusal names bf16: {msg}"
            );
            println!(
                "bf16 comparison is a cpu-runtime case: this backend cannot store bf16 ({msg})"
            );
            return;
        }
        Err(e) => panic!("unexpected rerank upload error: {e}"),
    };
    let actions_t = upload_ids(&actions, vec![rows, stride], &device);
    let identity_t = upload_ids(&identity, vec![rows, 2], &device);
    let mut rank_t = upload_ids(&vec![0xDEAD_BEEF; rows], vec![rows], &device);
    ms2_pack::rank(&actions_t, &identity_t, &scores_t, &rerank_t, &mut rank_t, T, A as u32, 2, 0, 0)
        .unwrap();
    check_launches(&device).unwrap();
    // Host order: traj 1 (-1.69921875) before traj 0 (-1.703125). bf16
    // arithmetic would tie both at -1.703125 and rank traj 0 first.
    assert_eq!(
        rank_t.try_to_vec().unwrap(),
        vec![1u32, 0u32],
        "bf16 rank order equals the host f32 order"
    );
    // Float records: the f32 buffer carries the f32-widened terms and sums.
    let mut record_f_t = Tensor::<R, f32>::from_f32(
        &vec![f32::NAN; rows * WF],
        vec![rows, WF],
        &device,
    )
    .unwrap();
    ms2_pack::record_pack_f(&scores_t, &rerank_t, &mut record_f_t, 0).unwrap();
    check_launches(&device).unwrap();
    assert_f32_bits(
        &record_f_t.try_to_f32().unwrap(),
        &[-1.0f32, -0.703125, -1.703125, -0.69140625, -1.0078125, -1.69921875],
        "record_f holds f32 terms and f32 sums",
    );
    // Compaction routes by rank and copies the f32 scores through.
    let width = record_width(T, A);
    let record_t = upload_ids(&vec![7u32; rows * width], vec![rows, width], &device);
    let rank_in_t = upload_ids(&vec![1u32, 0u32], vec![rows], &device);
    let mut packed_t = upload_ids(&vec![0xDEAD_BEEF; 1 * 2 * width], vec![1, 2, width], &device);
    let mut packed_f_t =
        Tensor::<R, f32>::from_f32(&vec![f32::NAN; 2 * WF], vec![1, 2, WF], &device).unwrap();
    let mut counts_t = upload_ids(&vec![0xDEAD_BEEF; 1], vec![1], &device);
    ms2_pack::pack(&rank_in_t, &record_t, &record_f_t, &mut packed_t, &mut packed_f_t, &mut counts_t, T, A as u32, 2, 2)
        .unwrap();
    check_launches(&device).unwrap();
    assert_f32_bits(
        &packed_f_t.try_to_f32().unwrap(),
        &[-0.69140625f32, -1.0078125, -1.69921875, -1.0, -0.703125, -1.703125],
        "packed_f carries the winner first with f32 scores",
    );
    assert_eq!(
        counts_t.try_to_vec().unwrap(),
        vec![2u32],
        "both trajectories eligible"
    );
}

#[test]
fn bf16_scores_fill_producer_through_ranking_matches_pack() {
    // Finding N1 / I-C4: the trace log-probability rides as f32 bits in the
    // u32 actions record on every neural dtype. Gathering it with
    // `F::reinterpret` panics during kernel expansion for bf16 (4 bytes to 2)
    // and the launch is dropped silently. The kernel decodes the word with
    // `f32::reinterpret`, then narrows. This test runs the PRODUCER
    // (`scores_fill`) in bf16 — never scores uploaded directly — through
    // ranking and packing on the CPU runtime, and every packed field equals
    // host `CandidateBatch::pack` (all terms are bf16-exact, so the gathered
    // f32 device terms equal the host f32 terms bit for bit).
    use half::bf16;
    if dev().name() != "cpu" {
        println!("bf16 producer runs only on the CPU runtime: skipped");
        return;
    }
    let device = dev();
    let recs = vec![fin(-0.5, -1.0), fin(-0.25, -0.75)];
    let batch = make_batch(1, 2, &recs, &[0]);
    batch.validate().expect("test batch validates");
    let rows = 2usize;
    let stride = T * 4 + A + 4;
    let width = record_width(T, A);
    let mut rng = Lcg(0x5eed);
    let layout = device_layout(&batch, &mut rng, false);
    // Formula log-probabilities by retained slot: `make_batch` ranks record
    // `r` as `r % 3` over real table row 7, so slot `s` holds the formula
    // log-probability of record `s`.
    let top_lp = vec![batch.formula_log_prob[0], batch.formula_log_prob[1], 0.0f32];
    for &v in top_lp.iter().chain(batch.trace_log_prob.iter()) {
        assert_eq!(bf16::from_f32(v).to_f32(), v, "term {v} is bf16-exact");
    }
    // The producer in bf16, on poisoned outputs. The gathered ranking terms
    // are f32 on every neural dtype; only the resident formula table
    // (`top`) and the caller scores (`rerank`) are bf16 here.
    let actions_t = upload_ids(&layout.actions, vec![rows, stride], &device);
    let traj_t = upload_ids(&layout.traj, vec![1, 2, 12], &device);
    let top_t = match Tensor::<R, bf16>::from_f32(&top_lp, vec![1, 3], &device) {
        Ok(t) => t,
        Err(Error::Unsupported(msg)) => {
            assert!(
                msg.contains("bf16"),
                "the bf16 refusal names bf16: {msg}"
            );
            println!(
                "bf16 comparison is a cpu-runtime case: this backend cannot store bf16 ({msg})"
            );
            return;
        }
        Err(e) => panic!("unexpected top upload error: {e}"),
    };
    let mut scores_t =
        Tensor::<R, f32>::from_f32(&vec![f32::NAN; rows * 2], vec![rows, 2], &device)
            .unwrap();
    ms2_pack::scores_fill(&actions_t, &traj_t, &top_t, &mut scores_t, T, A as u32, 2, 3)
        .unwrap();
    check_launches(&device).unwrap();
    // The gathered scores equal the host twin bit for bit (before the fix
    // the dropped launch left the NaN poison here).
    let mut want_scores = vec![0.0f32; rows * 2];
    for r in 0..rows {
        let (tl, fl) = scores_fill_lane(
            &layout.actions,
            stride as u32,
            (T * 4 + A) as u32,
            &layout.traj,
            &top_lp,
            3,
            r as u32,
            2,
        );
        want_scores[r * 2] = tl;
        want_scores[r * 2 + 1] = fl;
    }
    assert_f32_bits(
        &scores_t.try_to_f32().unwrap(),
        &want_scores,
        "bf16 producer scores",
    );
    // Ranking, records and compaction in bf16, on poisoned outputs.
    let identity = vec![0u32; rows * 2];
    let identity_t = upload_ids(&identity, vec![rows, 2], &device);
    let rerank_t = match Tensor::<R, bf16>::from_f32(&vec![0.0f32; rows], vec![rows], &device) {
        Ok(t) => t,
        Err(Error::Unsupported(msg)) => {
            assert!(
                msg.contains("bf16"),
                "the bf16 refusal names bf16: {msg}"
            );
            println!(
                "bf16 comparison is a cpu-runtime case: this backend cannot store bf16 ({msg})"
            );
            return;
        }
        Err(e) => panic!("unexpected rerank upload error: {e}"),
    };
    let mut rank_t = upload_ids(&vec![0xDEAD_BEEF; rows], vec![rows], &device);
    ms2_pack::rank(
        &actions_t, &identity_t, &scores_t, &rerank_t, &mut rank_t, T, A as u32, 2, 0, 0,
    )
    .unwrap();
    check_launches(&device).unwrap();
    let evidence_t = upload_ids(&layout.evidence, vec![rows, 18], &device);
    let mut record_t =
        upload_ids(&vec![0xDEAD_BEEF; rows * width], vec![rows, width], &device);
    ms2_pack::record_pack(
        &actions_t, &traj_t, &evidence_t, &identity_t, &mut record_t, T, A as u32, 2, 0,
    )
    .unwrap();
    check_launches(&device).unwrap();
    let mut record_f_t =
        Tensor::<R, f32>::from_f32(&vec![f32::NAN; rows * WF], vec![rows, WF], &device)
            .unwrap();
    ms2_pack::record_pack_f(&scores_t, &rerank_t, &mut record_f_t, 0).unwrap();
    check_launches(&device).unwrap();
    let ranks = rank_t.try_to_vec().unwrap();
    let rank_in_t = upload_ids(&ranks, vec![rows], &device);
    let mut packed_t =
        upload_ids(&vec![0xDEAD_BEEF; 2 * width], vec![1, 2, width], &device);
    let mut packed_f_t =
        Tensor::<R, f32>::from_f32(&vec![f32::NAN; 2 * WF], vec![1, 2, WF], &device)
            .unwrap();
    let mut counts_t = upload_ids(&vec![0xDEAD_BEEF; 1], vec![1], &device);
    ms2_pack::pack(
        &rank_in_t, &record_t, &record_f_t, &mut packed_t, &mut packed_f_t, &mut counts_t,
        T, A as u32, 2, 2,
    )
    .unwrap();
    check_launches(&device).unwrap();
    // Every packed field equals host `CandidateBatch::pack` on the same batch.
    let assembled = assemble(
        &batch,
        &packed_t.try_to_vec().unwrap(),
        &packed_f_t.try_to_f32().unwrap(),
        &counts_t.try_to_vec().unwrap(),
        2,
    )
    .unwrap();
    assembled.validate().unwrap();
    let expected = pack(&batch, None, ScoreKind::Raw, 2).unwrap();
    assert_eq!(assembled, expected, "bf16 producer pipeline != host pack");
}

#[test]
fn bf16_producer_preserves_f32_trace_terms_for_ranking() {
    // Finding A-P1: the ranking score is the f32 sum of f32-widened terms on
    // every neural dtype (spec §4.4). The reviewer's two-trajectory case has
    // the same formula log-probability −0.5 with f32 trace accumulators
    // −1.00390625 (trajectory 0) and −1.0009765625 (trajectory 1): host
    // scores −1.50390625 and −1.5009765625, so trajectory 1 ranks first.
    // Both trace terms round to bf16 −1, so a narrowed device sum ties at
    // −1.5 with trajectory 0 first. This test runs the PRODUCER
    // (`ms2_scores_fill` → rank → pack) with `E = bf16` and requires every
    // packed field to equal host `CandidateBatch::pack`.
    use half::bf16;
    if dev().name() != "cpu" {
        println!("bf16 producer runs only on the CPU runtime: skipped");
        return;
    }
    let device = dev();
    let recs = vec![fin(-0.5, -1.00390625), fin(-0.5, -1.0009765625)];
    let batch = make_batch(1, 2, &recs, &[0]);
    batch.validate().expect("test batch validates");
    // Host order: trajectory 1 (−1.5009765625) before trajectory 0
    // (−1.50390625).
    assert_eq!(
        batch.trace_log_prob[0] + batch.formula_log_prob[0],
        -1.50390625f32,
        "traj 0 host sum"
    );
    assert_eq!(
        batch.trace_log_prob[1] + batch.formula_log_prob[1],
        -1.5009765625f32,
        "traj 1 host sum"
    );
    // Both trace terms narrow to bf16 −1: a narrowed device sum would tie.
    assert_eq!(bf16::from_f32(-1.00390625f32).to_f32(), -1.0f32);
    assert_eq!(bf16::from_f32(-1.0009765625f32).to_f32(), -1.0f32);
    let rows = 2usize;
    let stride = T * 4 + A + 4;
    let width = record_width(T, A);
    let mut rng = Lcg(0x5eed);
    let layout = device_layout(&batch, &mut rng, false);
    let top_lp = vec![batch.formula_log_prob[0], batch.formula_log_prob[1], 0.0f32];
    let actions_t = upload_ids(&layout.actions, vec![rows, stride], &device);
    let traj_t = upload_ids(&layout.traj, vec![1, 2, 12], &device);
    let top_t = match Tensor::<R, bf16>::from_f32(&top_lp, vec![1, 3], &device) {
        Ok(t) => t,
        Err(Error::Unsupported(msg)) => {
            assert!(
                msg.contains("bf16"),
                "the bf16 refusal names bf16: {msg}"
            );
            println!(
                "bf16 comparison is a cpu-runtime case: this backend cannot store bf16 ({msg})"
            );
            return;
        }
        Err(e) => panic!("unexpected top upload error: {e}"),
    };
    let mut scores_t =
        Tensor::<R, f32>::from_f32(&vec![f32::NAN; rows * 2], vec![rows, 2], &device)
            .unwrap();
    ms2_pack::scores_fill(&actions_t, &traj_t, &top_t, &mut scores_t, T, A as u32, 2, 3)
        .unwrap();
    check_launches(&device).unwrap();
    // The gathered trace terms are the stored f32 bits unchanged.
    assert_f32_bits(
        &scores_t.try_to_f32().unwrap(),
        &[-1.00390625f32, -0.5, -1.0009765625, -0.5],
        "bf16 producer keeps f32 trace terms",
    );
    let identity = vec![0u32; rows * 2];
    let identity_t = upload_ids(&identity, vec![rows, 2], &device);
    let rerank_t = match Tensor::<R, bf16>::from_f32(&vec![0.0f32; rows], vec![rows], &device) {
        Ok(t) => t,
        Err(Error::Unsupported(msg)) => {
            assert!(
                msg.contains("bf16"),
                "the bf16 refusal names bf16: {msg}"
            );
            println!(
                "bf16 comparison is a cpu-runtime case: this backend cannot store bf16 ({msg})"
            );
            return;
        }
        Err(e) => panic!("unexpected rerank upload error: {e}"),
    };
    let mut rank_t = upload_ids(&vec![0xDEAD_BEEF; rows], vec![rows], &device);
    ms2_pack::rank(
        &actions_t, &identity_t, &scores_t, &rerank_t, &mut rank_t, T, A as u32, 2, 0, 0,
    )
    .unwrap();
    check_launches(&device).unwrap();
    assert_eq!(
        rank_t.try_to_vec().unwrap(),
        vec![1u32, 0u32],
        "trajectory 1 ranks first, as the host f32 sums do"
    );
    let evidence_t = upload_ids(&layout.evidence, vec![rows, 18], &device);
    let mut record_t =
        upload_ids(&vec![0xDEAD_BEEF; rows * width], vec![rows, width], &device);
    ms2_pack::record_pack(
        &actions_t, &traj_t, &evidence_t, &identity_t, &mut record_t, T, A as u32, 2, 0,
    )
    .unwrap();
    check_launches(&device).unwrap();
    let mut record_f_t =
        Tensor::<R, f32>::from_f32(&vec![f32::NAN; rows * WF], vec![rows, WF], &device)
            .unwrap();
    ms2_pack::record_pack_f(&scores_t, &rerank_t, &mut record_f_t, 0).unwrap();
    check_launches(&device).unwrap();
    assert_f32_bits(
        &record_f_t.try_to_f32().unwrap(),
        &[-0.5f32, -1.00390625, -1.50390625, -0.5, -1.0009765625, -1.5009765625],
        "record_f holds f32 terms and f32 sums",
    );
    let ranks = rank_t.try_to_vec().unwrap();
    let rank_in_t = upload_ids(&ranks, vec![rows], &device);
    let mut packed_t =
        upload_ids(&vec![0xDEAD_BEEF; 2 * width], vec![1, 2, width], &device);
    let mut packed_f_t =
        Tensor::<R, f32>::from_f32(&vec![f32::NAN; 2 * WF], vec![1, 2, WF], &device)
            .unwrap();
    let mut counts_t = upload_ids(&vec![0xDEAD_BEEF; 1], vec![1], &device);
    ms2_pack::pack(
        &rank_in_t, &record_t, &record_f_t, &mut packed_t, &mut packed_f_t, &mut counts_t,
        T, A as u32, 2, 2,
    )
    .unwrap();
    check_launches(&device).unwrap();
    let assembled = assemble(
        &batch,
        &packed_t.try_to_vec().unwrap(),
        &packed_f_t.try_to_f32().unwrap(),
        &counts_t.try_to_vec().unwrap(),
        2,
    )
    .unwrap();
    assembled.validate().unwrap();
    let expected = pack(&batch, None, ScoreKind::Raw, 2).unwrap();
    assert_eq!(assembled, expected, "bf16 producer pipeline != host pack");
}
