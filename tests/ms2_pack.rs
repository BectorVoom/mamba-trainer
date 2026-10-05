//! K3 host tests for ranking, compaction and the packed output.
//!
//! Hand-built `CandidateBatch`es cover ties, every ineligibility reason, NaN
//! scores, `R > eligible`, `R == K`, failed requests and identity bits
//! present/absent; `validate` is shown to accept `pack`'s output and to reject
//! each broken invariant; and `pack` is compared with an independent
//! straightforward implementation (sort + filter with `Vec`) on 200 random
//! batches.

use mamba3::error::Error;
use mamba3::models::ms2::contract::{
    CandidateBatch, NO_FORMULA, SCHEMA_VERSION, candidate_status, request_status,
};
use mamba3::models::ms2::pack::{
    PackedCandidateBatch, SCORE_FINITE_MAX, ScoreKind, check_u32_len, check_u32_product, pack,
    score_in_domain,
};

// ---------------------------------------------------------------------------
// Batch builder (traces stay legal so `validate` passes unless NaN is used)
// ---------------------------------------------------------------------------

const T: usize = 6;
const A: usize = 8;
const RMAX: usize = 4;
const K_DEFAULT: usize = 4;

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
    Rec {
        tmpl: Tmpl::Fin,
        extra_status: 0,
        flp,
        tlp,
        formula: true,
    }
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

#[allow(clippy::too_many_arguments)]
fn make_batch(
    b: usize,
    k: usize,
    recs: &[Rec],
    req: &[u32],
    src: &[u8],
    rank_seq: usize,
    check: bool,
) -> CandidateBatch {
    assert_eq!(recs.len(), b * k);
    assert_eq!(req.len(), b);
    assert_eq!(src.len(), b);
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
        spectrum_id[r] = 7000 + bb as u64;
        trajectory[r] = kk as u32;
        if rec.formula {
            formula_rank[r] = ((r + rank_seq) % 3) as u32;
            if src[bb] == 0 {
                formula_row[r] = 7;
            }
            let counts = [4u16, 8, 2, 1, 0, 0, 0, 0, 0, 0];
            formula_counts[r * 10..(r + 1) * 10].copy_from_slice(&counts);
        }
    }
    let batch = CandidateBatch {
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
        formula_source: src.to_vec(),
        formula_rank,
    };
    if check {
        batch.validate().expect("test batch validates");
    }
    batch
}

fn bit7(n: usize) -> Vec<u32> {
    vec![0u32; n]
}

fn id_bits_with(n: usize, dup: &[usize], unres: &[usize]) -> Vec<u32> {
    let mut v = vec![0u32; n];
    for &r in dup {
        v[r] |= 1 << 7;
    }
    for &r in unres {
        v[r] |= 1 << 8;
    }
    v
}

fn slot_trajectories(got: &PackedCandidateBatch, b: usize) -> Vec<u32> {
    let r = got.returned;
    got.trajectory[b * r..(b + 1) * r].to_vec()
}

// ---------------------------------------------------------------------------
// Directed tests
// ---------------------------------------------------------------------------

#[test]
fn ties_break_by_smaller_trajectory() {
    let recs = vec![fin(-0.5, -0.5); K_DEFAULT];
    let batch = make_batch(1, K_DEFAULT, &recs, &[0], &[0], 0, true);
    let got = pack(&batch, Some(&bit7(4)), ScoreKind::Raw, K_DEFAULT).unwrap();
    got.validate().unwrap();
    assert_eq!(got.returned_count, vec![4]);
    assert_eq!(slot_trajectories(&got, 0), vec![0, 1, 2, 3]);
    for v in &got.score {
        assert_eq!(*v, -1.0);
    }
}

#[test]
fn ranking_orders_by_decreasing_score() {
    let recs = vec![
        fin(-0.5, -2.0),
        fin(-0.1, -0.1),
        fin(-1.0, -0.1),
        fin(-0.1, -2.0),
    ];
    let batch = make_batch(1, K_DEFAULT, &recs, &[0], &[0], 0, true);
    let got = pack(&batch, Some(&bit7(4)), ScoreKind::Raw, K_DEFAULT).unwrap();
    got.validate().unwrap();
    // Scores: -2.5, -0.2, -1.1, -2.1 → trajectories 1, 2, 3, 0.
    assert_eq!(slot_trajectories(&got, 0), vec![1, 2, 3, 0]);
}

#[test]
fn each_ineligibility_reason_excludes() {
    // Record 0 carries the reason under test; records 1–2 are eligible with
    // lower scores, so an exclusion shows as [1, 2] with count 2.
    let good1 = fin(-2.0, -2.0);
    let good2 = fin(-3.0, -3.0);
    let cand0 = fin(-0.1, -0.1);
    let cases: Vec<(&str, Rec, Option<Vec<u32>>)> = vec![
        ("unfinished", Rec { tmpl: Tmpl::Unfin, ..cand0 }, Some(bit7(3))),
        ("truncated", Rec { tmpl: Tmpl::Trunc, ..cand0 }, Some(bit7(3))),
        (
            "invalid_final",
            Rec { extra_status: candidate_status::INVALID_FINAL, ..cand0 },
            Some(bit7(3)),
        ),
        (
            "duplicate_trace",
            Rec { extra_status: candidate_status::DUPLICATE_TRACE, ..cand0 },
            Some(bit7(3)),
        ),
        ("duplicate_graph", cand0, Some(id_bits_with(3, &[0], &[]))),
        ("score_outside_validated_domain", Rec { tlp: f32::NAN, ..cand0 }, Some(bit7(3))),
        ("formula_outside_validated_domain", Rec { flp: f32::NAN, ..cand0 }, Some(bit7(3))),
    ];
    for (what, rec0, bits) in &cases {
        let recs = vec![*rec0, good1, good2];
        let check = !what.contains("outside_validated_domain");
        let batch = make_batch(1, 3, &recs, &[0], &[0], 0, check);
        let got = pack(&batch, bits.as_deref(), ScoreKind::Raw, 3).unwrap();
        got.validate().unwrap();
        assert_eq!(got.returned_count, vec![2], "{what}");
        assert_eq!(slot_trajectories(&got, 0)[..2], [1, 2], "{what}");
        assert_eq!(got.trajectory[2], NO_FORMULA, "{what}");
    }
}

#[test]
fn identity_absent_keeps_graph_duplicates_eligible() {
    let recs = vec![fin(-0.1, -0.1), fin(-2.0, -2.0)];
    let batch = make_batch(1, 2, &recs, &[0], &[0], 0, true);
    let bits = id_bits_with(2, &[0, 1], &[]);
    let got = pack(&batch, None, ScoreKind::Raw, 2).unwrap();
    got.validate().unwrap();
    assert_eq!(got.returned_count, vec![2]);
    // The duplicate bits were given but must not matter without them passed.
    let _ = bits;
    let got2 = pack(&batch, Some(&bits), ScoreKind::Raw, 2).unwrap();
    got2.validate().unwrap();
    assert_eq!(got2.returned_count, vec![0]);
}

#[test]
fn unresolved_identity_stays_eligible() {
    let recs = vec![fin(-0.1, -0.1), fin(-2.0, -2.0)];
    let batch = make_batch(1, 2, &recs, &[0], &[0], 0, true);
    let bits = id_bits_with(2, &[], &[0]);
    let got = pack(&batch, Some(&bits), ScoreKind::Raw, 2).unwrap();
    got.validate().unwrap();
    assert_eq!(got.returned_count, vec![2]);
    assert_eq!(got.identity_resolution, vec![2, 1]);
    assert_eq!(got.status[0] & (1 << 8), 1 << 8);
}

#[test]
fn r_larger_than_eligible_leaves_unfilled_slots() {
    let recs = vec![
        fin(-0.1, -0.1),
        Rec { tmpl: Tmpl::Unfin, ..fin(-0.1, -0.1) },
        fin(-0.2, -0.2),
        Rec { tmpl: Tmpl::Unfin, ..fin(-0.1, -0.1) },
    ];
    let batch = make_batch(1, K_DEFAULT, &recs, &[0], &[0], 0, true);
    let got = pack(&batch, Some(&bit7(4)), ScoreKind::Raw, K_DEFAULT).unwrap();
    got.validate().unwrap();
    assert_eq!(got.returned_count, vec![2]);
    assert_eq!(slot_trajectories(&got, 0), vec![0, 2, NO_FORMULA, NO_FORMULA]);
    assert_eq!(got.status[2], 0);
    assert_eq!(got.status[3], 0);
    assert_eq!(got.score[2], 0.0);
    assert_eq!(got.length[2], 0);
}

#[test]
fn failed_request_returns_nothing() {
    let recs = vec![
        Rec { tmpl: Tmpl::Fail, ..fin(0.0, 0.0) },
        Rec { tmpl: Tmpl::Fail, ..fin(0.0, 0.0) },
    ];
    let fatal = request_status::EMPTY_SPECTRUM;
    let batch = make_batch(1, 2, &recs, &[fatal], &[0], 0, true);
    let got = pack(&batch, Some(&bit7(2)), ScoreKind::Raw, 2).unwrap();
    got.validate().unwrap();
    assert_eq!(got.returned_count, vec![0]);
    assert_eq!(got.request_status, vec![fatal]);
    assert_eq!(slot_trajectories(&got, 0), vec![NO_FORMULA, NO_FORMULA]);
}

#[test]
fn mixed_spectra_pack_independently() {
    // Spectrum 0: two eligible + one unfinished, R = 3. Spectrum 1: failed.
    let recs = vec![
        fin(-0.3, -0.3),
        Rec { tmpl: Tmpl::Unfin, ..fin(-0.1, -0.1) },
        fin(-0.1, -0.1),
        Rec { tmpl: Tmpl::Fail, ..fin(0.0, 0.0) },
        Rec { tmpl: Tmpl::Fail, ..fin(0.0, 0.0) },
        Rec { tmpl: Tmpl::Fail, ..fin(0.0, 0.0) },
    ];
    let fatal = request_status::EMPTY_SPECTRUM;
    let batch = make_batch(2, 3, &recs, &[0, fatal], &[0, 0], 0, true);
    let got = pack(&batch, Some(&bit7(6)), ScoreKind::Raw, 3).unwrap();
    got.validate().unwrap();
    assert_eq!(got.returned_count, vec![2, 0]);
    assert_eq!(slot_trajectories(&got, 0), vec![2, 0, NO_FORMULA]);
    assert_eq!(
        slot_trajectories(&got, 1),
        vec![NO_FORMULA, NO_FORMULA, NO_FORMULA]
    );
}

#[test]
fn reranker_scores_reorder_and_exclude_outside_validated_domain() {
    let recs = vec![fin(-0.1, -0.1), fin(-0.1, -0.1), fin(-0.1, -0.1)];
    let batch = make_batch(1, 3, &recs, &[0], &[0], 0, true);
    // NaN is outside the validated domain, like infinities and finite extremes.
    let scores = vec![1.0f32, f32::NAN, 2.0];
    let got = pack(&batch, Some(&bit7(3)), ScoreKind::Reranker(&scores), 3).unwrap();
    got.validate().unwrap();
    assert_eq!(got.returned_count, vec![2]);
    assert_eq!(slot_trajectories(&got, 0), vec![2, 0, NO_FORMULA]);
    assert_eq!(got.score[0], 2.0);
    assert_eq!(got.score[1], 1.0);
}

#[test]
fn pack_rejects_bad_arguments() {
    let recs = vec![fin(-0.1, -0.1); K_DEFAULT];
    let batch = make_batch(1, K_DEFAULT, &recs, &[0], &[0], 0, true);
    assert!(pack(&batch, Some(&bit7(4)), ScoreKind::Raw, 0).is_err());
    assert!(pack(&batch, Some(&bit7(4)), ScoreKind::Raw, K_DEFAULT + 1).is_err());
    assert!(pack(&batch, Some(&bit7(3)), ScoreKind::Raw, 2).is_err());
    assert!(pack(&batch, None, ScoreKind::Reranker(&[0.0; 3]), 2).is_err());
    let mut short = batch.clone();
    short.length.pop();
    assert!(pack(&short, None, ScoreKind::Raw, 2).is_err());
}

// ---------------------------------------------------------------------------
// `validate` rejects each broken invariant
// ---------------------------------------------------------------------------

fn good_packed() -> PackedCandidateBatch {
    let recs = vec![fin(-0.1, -0.1), fin(-0.5, -0.5)];
    let batch = make_batch(1, 2, &recs, &[0], &[0], 0, true);
    let got = pack(&batch, Some(&bit7(2)), ScoreKind::Raw, 2).unwrap();
    got.validate().unwrap();
    got
}

#[test]
fn validate_rejects_broken_invariants() {
    let mut ok = good_packed();
    ok.schema_version = 99;
    assert!(ok.validate().is_err());

    let mut ok = good_packed();
    ok.returned = 3;
    assert!(ok.validate().is_err());

    let mut ok = good_packed();
    ok.returned_count[0] = 3;
    assert!(ok.validate().is_err());

    // Rank order broken: the second slot outscores the first.
    let mut ok = good_packed();
    ok.score[1] = ok.score[0] + 1.0;
    assert!(ok.validate().is_err());

    // Tie with the wrong trajectory order.
    let mut ok = good_packed();
    ok.score[1] = ok.score[0];
    ok.trajectory.swap(0, 1);
    assert!(ok.validate().is_err());

    // Unfilled slot with a real trajectory.
    // Make room: pack with R = 2 over K = 2 has no unfilled slot, so rebuild
    // with R = 2 over one eligible of two.
    let recs = vec![fin(-0.1, -0.1), Rec { tmpl: Tmpl::Unfin, ..fin(0.0, 0.0) }];
    let batch = make_batch(1, 2, &recs, &[0], &[0], 0, true);
    let mut ok = pack(&batch, Some(&bit7(2)), ScoreKind::Raw, 2).unwrap();
    ok.validate().unwrap();
    ok.trajectory[1] = 1;
    assert!(ok.validate().is_err());

    let mut ok = good_packed();
    ok.status[0] |= candidate_status::INVALID_FINAL;
    assert!(ok.validate().is_err());

    let mut ok = good_packed();
    ok.actions[0] = 9;
    assert!(ok.validate().is_err());

    let mut ok = good_packed();
    ok.actions.pop();
    assert!(ok.validate().is_err());

    // Failed request with a filled slot.
    let mut ok = good_packed();
    ok.request_status[0] = request_status::EMPTY_SPECTRUM;
    assert!(ok.validate().is_err());
}

// ---------------------------------------------------------------------------
// Random batches against an independent implementation
// ---------------------------------------------------------------------------

fn reference_pack(
    batch: &CandidateBatch,
    identity_bits: Option<&[u32]>,
    score: ScoreKind<'_>,
    returned: usize,
) -> PackedCandidateBatch {
    let k = batch.trajectories;
    let rerank_of = |r: usize| match score {
        ScoreKind::Raw => batch.formula_log_prob[r] + batch.trace_log_prob[r],
        ScoreKind::Reranker(s) => s[r],
    };
    let eligible = |r: usize| {
        let st = batch.status[r];
        if st & candidate_status::FINISHED == 0 {
            return None;
        }
        if st & candidate_status::INVALID_FINAL != 0 {
            return None;
        }
        if st & candidate_status::DUPLICATE_TRACE != 0 {
            return None;
        }
        if st & candidate_status::REQUEST_FAILED != 0 {
            return None;
        }
        if let Some(bits) = identity_bits
            && bits[r] & (1 << 7) != 0
        {
            return None;
        }
        let tl = batch.trace_log_prob[r];
        let fl = batch.formula_log_prob[r];
        if !(tl > -SCORE_FINITE_MAX
            && tl < SCORE_FINITE_MAX
            && fl > -SCORE_FINITE_MAX
            && fl < SCORE_FINITE_MAX)
        {
            return None;
        }
        let s = rerank_of(r);
        if !(s > -SCORE_FINITE_MAX && s < SCORE_FINITE_MAX) {
            return None;
        }
        Some(s)
    };
    let mut trajectory = vec![NO_FORMULA; batch.batch * returned];
    let mut score_out = vec![0.0f32; batch.batch * returned];
    let mut returned_count = vec![0u32; batch.batch];
    for b in 0..batch.batch {
        let mut cand: Vec<(f32, u32)> = Vec::new();
        for kk in 0..k {
            let r = b * k + kk;
            if let Some(s) = eligible(r) {
                cand.push((s, kk as u32));
            }
        }
        cand.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .expect("scores outside the validated domain are excluded before sorting")
                .then(a.1.cmp(&b.1))
        });
        let take = cand.len().min(returned);
        returned_count[b] = take as u32;
        for (i, (s, kk)) in cand.iter().take(take).enumerate() {
            trajectory[b * returned + i] = *kk;
            score_out[b * returned + i] = *s;
        }
    }
    let mut out = PackedCandidateBatch {
        schema_version: SCHEMA_VERSION,
        batch: batch.batch,
        returned,
        trajectories: k,
        max_steps: batch.max_steps,
        max_atoms: batch.max_atoms,
        max_ring_closures: batch.max_ring_closures,
        spectrum_id: vec![0; batch.batch * returned],
        trajectory,
        actions: vec![0; batch.batch * returned * batch.max_steps * 4],
        length: vec![0; batch.batch * returned],
        formula_row: vec![NO_FORMULA; batch.batch * returned],
        formula_rank: vec![NO_FORMULA; batch.batch * returned],
        formula_counts: vec![0; batch.batch * returned * 10],
        formula_log_prob: vec![0.0; batch.batch * returned],
        trace_log_prob: vec![0.0; batch.batch * returned],
        score: score_out,
        open_valence: vec![0; batch.batch * returned * batch.max_atoms],
        status: vec![0; batch.batch * returned],
        evidence_status: vec![0; batch.batch * returned],
        evidence_count: vec![0; batch.batch * returned],
        evidence_peak_id: vec![0; (batch.batch * returned) * 4],
        evidence_hypothesis: vec![0; (batch.batch * returned) * 4],
        evidence_shift: vec![0; (batch.batch * returned) * 4],
        evidence_residual: vec![0; (batch.batch * returned) * 4],
        evidence_log_prob: vec![0.0; (batch.batch * returned) * 4],
        identity_resolution: vec![0; batch.batch * returned],
        attachment_partition: vec![0; batch.batch * returned],
        returned_count,
        request_status: batch.request_status.clone(),
        rows_visited: batch.rows_visited.clone(),
        rows_joined: batch.rows_joined.clone(),
        rows_scored: batch.rows_scored.clone(),
        formula_support_complete: batch.formula_support_complete.clone(),
        formula_mass_retained: batch.formula_mass_retained.clone(),
        peaks_kept: batch.peaks_kept.clone(),
        intensity_retained: batch.intensity_retained.clone(),
        formula_source: batch.formula_source.clone(),
    };
    for b in 0..batch.batch {
        for r in 0..returned {
            let s = b * returned + r;
            let kk = out.trajectory[s];
            if kk == NO_FORMULA {
                continue;
            }
            let rec = b * k + kk as usize;
            out.spectrum_id[s] = batch.spectrum_id[rec];
            out.length[s] = batch.length[rec];
            out.formula_row[s] = batch.formula_row[rec];
            out.formula_rank[s] = batch.formula_rank[rec];
            out.formula_counts[s * 10..(s + 1) * 10]
                .copy_from_slice(&batch.formula_counts[rec * 10..(rec + 1) * 10]);
            out.formula_log_prob[s] = batch.formula_log_prob[rec];
            out.trace_log_prob[s] = batch.trace_log_prob[rec];
            out.open_valence[s * batch.max_atoms..(s + 1) * batch.max_atoms]
                .copy_from_slice(&batch.open_valence[rec * batch.max_atoms..(rec + 1) * batch.max_atoms]);
            out.actions[s * batch.max_steps * 4..(s + 1) * batch.max_steps * 4]
                .copy_from_slice(
                    &batch.actions[rec * batch.max_steps * 4..(rec + 1) * batch.max_steps * 4],
                );
            let mut st = batch.status[rec];
            let mut res = 0u8;
            if let Some(bits) = identity_bits {
                st |= bits[rec] & ((1 << 7) | (1 << 8));
                res = 1;
                if bits[rec] & (1 << 8) != 0 {
                    res = 2;
                }
            }
            out.status[s] = st;
            out.identity_resolution[s] = res;
        }
    }
    // Every spectrum's slots share the spectrum id, filled or not.
    for b in 0..batch.batch {
        let id = batch.spectrum_id[b * k.max(1)];
        for r in 0..returned {
            out.spectrum_id[b * returned + r] = id;
        }
    }
    out
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

#[test]
fn pack_matches_reference_on_random_batches() {
    let mut rng = Lcg(0x5eed_1234);
    for iter in 0..200 {
        let b = 1 + rng.below(3) as usize;
        let k = 1 + rng.below(8) as usize;
        let fatal_b = if rng.below(5) == 0 {
            Some(rng.below(b as u64) as usize)
        } else {
            None
        };
        let mut recs = Vec::with_capacity(b * k);
        for r in 0..b * k {
            let bb = r / k;
            if Some(bb) == fatal_b {
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
            // Scores in 0.5 steps (ties are common); occasional NaN.
            let q = rng.below(12) as f32;
            let mut flp = -0.5 * q;
            let mut tlp = -0.5 * (rng.below(12) as f32);
            if rng.below(20) == 0 {
                if rng.below(2) == 0 {
                    flp = f32::NAN;
                } else {
                    tlp = f32::NAN;
                }
            }
            // No trajectory starts without a formula (finding R1-C6): every
            // finished record carries one, so packed filled slots always
            // have real provenance and validate.
            let formula = rng.below(4) != 0 || tmpl == Tmpl::Fin;
            recs.push(Rec { tmpl, extra_status: extra, flp, tlp, formula });
        }
        // Truncated records are only finished-shaped when T fits; the
        // builder's truncated template needs T == 6, which holds here.
        let mut req = vec![0u32; b];
        if let Some(fb) = fatal_b {
            req[fb] = request_status::EMPTY_SPECTRUM;
        } else if rng.below(4) == 0 {
            req[rng.below(b as u64) as usize] = 1 << 16;
        }
        let mut src = vec![0u8; b];
        for s in src.iter_mut() {
            if rng.below(3) == 0 {
                *s = 1;
            }
        }
        // Enumeration spectra name no table row.
        let batch = make_batch(b, k, &recs, &req, &src, iter, false);
        let has_nan = batch
            .formula_log_prob
            .iter()
            .chain(batch.trace_log_prob.iter())
            .any(|v| v.is_nan());
        if !has_nan {
            batch.validate().expect("random batch validates");
        }
        let use_identity = rng.below(2) == 0;
        let id_bits: Vec<u32> = (0..b * k)
            .map(|_| {
                let mut w = 0u32;
                if rng.below(6) == 0 {
                    w |= 1 << 7;
                }
                if rng.below(6) == 0 {
                    w |= 1 << 8;
                }
                w
            })
            .collect();
        let identity = if use_identity { Some(id_bits.as_slice()) } else { None };
        let returned = 1 + rng.below(k as u64) as usize;
        let use_reranker = rng.below(2) == 0;
        if use_reranker {
            let scores: Vec<f32> = (0..b * k)
                .map(|_| {
                    let q = rng.below(8) as f32;
                    if rng.below(20) == 0 { f32::NAN } else { q }
                })
                .collect();
            let got = pack(&batch, identity, ScoreKind::Reranker(&scores), returned)
                .expect("random reranker pack succeeds");
            got.validate().expect("random reranker pack validates");
            let want = reference_pack(&batch, identity, ScoreKind::Reranker(&scores), returned);
            assert_eq!(got, want, "iter {iter} reranker");
        } else {
            let got = pack(&batch, identity, ScoreKind::Raw, returned)
                .expect("random raw pack succeeds");
            got.validate().expect("random raw pack validates");
            let want = reference_pack(&batch, identity, ScoreKind::Raw, returned);
            assert_eq!(got, want, "iter {iter} raw");
        }
    }
}

// ---------------------------------------------------------------------------
// Finding E1: `validate` corruption tests, one per rule
// ---------------------------------------------------------------------------

#[test]
fn validate_rejects_score_outside_validated_domain_on_its_own() {
    // The first filled slot has no earlier neighbour to compare against, so
    // its score must be checked on its own. Every value outside (−3e38,
    // 3e38) — NaN, infinities and finite extremes included — is rejected;
    // just inside the boundary still validates.
    for bad in [
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        3.1e38,
        -3.1e38,
        3.0e38,
        -3.0e38,
    ] {
        let mut ok = good_packed();
        ok.score[0] = bad;
        assert!(ok.validate().is_err(), "score {bad} passes validate");
    }
    let mut ok = good_packed();
    ok.score[0] = 2.9e38;
    ok.score[1] = -1.0;
    ok.validate().expect("in-domain first-slot score validates");
}

#[test]
fn validate_rejects_enumeration_table_row() {
    // An enumerated formula names no table row.
    let recs = vec![fin(-0.1, -0.1), fin(-0.5, -0.5)];
    let batch = make_batch(1, 2, &recs, &[0], &[1], 0, true);
    let mut got = pack(&batch, Some(&bit7(2)), ScoreKind::Raw, 2).unwrap();
    got.validate().unwrap();
    assert_eq!(got.formula_row, vec![NO_FORMULA, NO_FORMULA]);
    got.formula_row[0] = 7;
    assert!(got.validate().is_err());
}

#[test]
fn validate_rejects_exhausted_enumeration_claiming_completeness() {
    // Finding N3: `formula_support_complete == 1` is rejected when the
    // request carries `FORMULA_SEARCH_EXHAUSTED` or a counter is saturated
    // (contracts §9). The reviewer's input on a valid filled enumeration
    // packed record: keep valid provenance and `joined == scored == 1`,
    // saturate `visited`, add exhaustion, keep `complete == 1`.
    let recs = vec![fin(-0.1, -0.1)];
    let batch = make_batch(1, 1, &recs, &[0], &[1], 0, true);
    let mut got = pack(&batch, Some(&bit7(1)), ScoreKind::Raw, 1).unwrap();
    got.validate().unwrap();
    got.rows_visited[0] = u32::MAX - 1;
    got.rows_joined[0] = 1;
    got.rows_scored[0] = 1;
    got.request_status[0] |= request_status::FORMULA_SEARCH_EXHAUSTED;
    assert_eq!(got.formula_support_complete[0], 1);
    let err = got.validate().unwrap_err();
    assert!(
        err.to_string().contains("formula_support_complete"),
        "exhausted completeness names completeness: {err}"
    );
    // Clearing the false claim restores validity: saturated counters with
    // exhaustion and `complete == 0` are a legal lower bound.
    let mut ok = got.clone();
    ok.formula_support_complete[0] = 0;
    ok.validate().expect("cleared completeness validates");
}

#[test]
fn validate_rejects_counts_rank_mismatch() {
    // Counts are all zero exactly when the rank is `u32::MAX`.
    let mut ok = good_packed();
    for c in ok.formula_counts[0..10].iter_mut() {
        *c = 0;
    }
    assert!(ok.validate().is_err());
    let mut ok = good_packed();
    ok.formula_rank[0] = NO_FORMULA;
    assert!(ok.validate().is_err());
}

#[test]
fn validate_rejects_rank_above_rows_scored() {
    // A real rank is a slot in the spectrum's scored support.
    let mut ok = good_packed();
    ok.formula_rank[0] = ok.rows_scored[0];
    assert!(ok.validate().is_err());
    // The boundary itself stays legal.
    let mut ok = good_packed();
    ok.formula_rank[0] = ok.rows_scored[0] - 1;
    ok.validate().expect("rank below rows_scored validates");
}

#[test]
fn validate_rejects_table_missing_row() {
    // A table-source record with a formula names a real table row.
    let mut ok = good_packed();
    ok.formula_row[0] = NO_FORMULA;
    assert!(ok.validate().is_err());
}

#[test]
fn validate_rejects_bad_identity_resolution() {
    let mut ok = good_packed();
    ok.identity_resolution[0] = 3;
    assert!(ok.validate().is_err());
    for v in [0u8, 1, 2] {
        let mut ok = good_packed();
        ok.identity_resolution[0] = v;
        ok.validate().expect("resolution {v} validates");
    }
}

#[test]
fn validate_rejects_duplicate_trajectories() {
    // Both slots stay in rank order (scores −0.2 then −1.0), so only the
    // uniqueness rule can fire.
    let mut ok = good_packed();
    ok.trajectory[1] = ok.trajectory[0];
    assert!(ok.validate().is_err());
}

#[test]
fn validate_rejects_trace_over_formula_budget() {
    // Slot 0 conditions on C4 H8 N2 O1 and emits one C(H3) atom; starving
    // the hydrogen budget makes the identical trace illegal under it.
    let mut ok = good_packed();
    ok.formula_counts[1] = 0;
    assert!(ok.validate().is_err());
}

// ---------------------------------------------------------------------------
// Finding E2: the u32 address checks, unit-tested without allocating
// ---------------------------------------------------------------------------

#[test]
fn u32_address_checks_reject_oversized_shapes() {
    let over = u32::MAX as usize + 1;
    assert!(matches!(check_u32_len("rows", over), Err(Error::Shape(_))));
    assert_eq!(check_u32_len("rows", u32::MAX as usize).unwrap(), u32::MAX);
    assert_eq!(check_u32_len("rows", 0).unwrap(), 0);
    // A product above the domain and an overflowing product are both
    // `Error::Shape`.
    assert!(matches!(
        check_u32_product("records", 1 << 32, 2),
        Err(Error::Shape(_))
    ));
    assert!(matches!(
        check_u32_product("records", usize::MAX, 2),
        Err(Error::Shape(_))
    ));
    assert_eq!(
        check_u32_product("records", 1 << 16, (1 << 16) - 1).unwrap(),
        (1 << 16) * ((1 << 16) - 1)
    );
}

// ---------------------------------------------------------------------------
// Finding E3: the validated score domain, on the host pack path
// ---------------------------------------------------------------------------

#[test]
fn validated_score_domain_matches_contract() {
    assert!(score_in_domain(0.0));
    assert!(score_in_domain(-1.0));
    assert!(score_in_domain(2.9e38));
    assert!(!score_in_domain(f32::NAN));
    assert!(!score_in_domain(f32::INFINITY));
    assert!(!score_in_domain(f32::NEG_INFINITY));
    assert!(!score_in_domain(3.1e38));
    assert!(!score_in_domain(-3.1e38));
    // At or above 3e38 is out by contract, even when finite.
    assert!(!score_in_domain(3.0e38));
    assert!(!score_in_domain(-3.0e38));
}

#[test]
fn pack_excludes_out_of_domain_scores() {
    // Finite-extreme, infinite and NaN raw terms are ineligible on the host;
    // so is an overflowing sum of two in-domain terms (2e38 + 2e38 is +inf).
    let cases = [
        ("extreme_formula", 3.1e38f32, -0.5f32),
        ("extreme_trace", -0.5f32, -3.1e38f32),
        ("infinite", f32::INFINITY, -0.5f32),
        ("negative_infinite", -0.5f32, f32::NEG_INFINITY),
        ("nan_formula", f32::NAN, -0.5f32),
        ("overflowing_sum", 2.0e38f32, 2.0e38f32),
    ];
    for (what, flp, tlp) in cases {
        let recs = vec![
            Rec { flp, tlp, ..fin(-0.1, -0.1) },
            fin(-2.0, -2.0),
            fin(-3.0, -3.0),
        ];
        let batch = make_batch(1, 3, &recs, &[0], &[0], 0, false);
        let got = pack(&batch, Some(&bit7(3)), ScoreKind::Raw, 3).unwrap();
        got.validate().unwrap();
        assert_eq!(got.returned_count, vec![2], "{what}");
        assert_eq!(slot_trajectories(&got, 0)[..2], [1, 2], "{what}");
        assert_eq!(got.trajectory[2], NO_FORMULA, "{what}");
    }
    // A caller-supplied reranker extreme is excluded too, even with
    // in-domain raw terms.
    let recs = vec![fin(-0.1, -0.1), fin(-0.2, -0.2)];
    let batch = make_batch(1, 2, &recs, &[0], &[0], 0, true);
    let got = pack(
        &batch,
        Some(&bit7(2)),
        ScoreKind::Reranker(&[3.1e38, -0.5]),
        2,
    )
    .unwrap();
    got.validate().unwrap();
    assert_eq!(got.returned_count, vec![1]);
    assert_eq!(got.trajectory[0], 1);
}

// ---------------------------------------------------------------------------
// Finding E5: every consumed field is length-checked (`Error::Shape`)
// ---------------------------------------------------------------------------

#[test]
fn pack_rejects_every_truncated_field_with_shape() {
    let recs = vec![fin(-0.1, -0.1), fin(-0.2, -0.2)];
    let batch = make_batch(1, 2, &recs, &[0], &[0], 0, true);
    let mut cases: Vec<(&str, CandidateBatch)> = Vec::new();
    let mut truncated = |name: &'static str, f: fn(&mut CandidateBatch)| {
        let mut short = batch.clone();
        f(&mut short);
        cases.push((name, short));
    };
    truncated("spectrum_id", |b| {
        b.spectrum_id.pop();
    });
    truncated("trajectory", |b| {
        b.trajectory.pop();
    });
    truncated("length", |b| {
        b.length.pop();
    });
    truncated("formula_row", |b| {
        b.formula_row.pop();
    });
    truncated("formula_rank", |b| {
        b.formula_rank.pop();
    });
    truncated("formula_log_prob", |b| {
        b.formula_log_prob.pop();
    });
    truncated("trace_log_prob", |b| {
        b.trace_log_prob.pop();
    });
    truncated("status", |b| {
        b.status.pop();
    });
    truncated("evidence_status", |b| {
        b.evidence_status.pop();
    });
    truncated("actions", |b| {
        b.actions.pop();
    });
    truncated("open_valence", |b| {
        b.open_valence.pop();
    });
    truncated("formula_counts", |b| {
        b.formula_counts.pop();
    });
    truncated("request_status", |b| {
        b.request_status.pop();
    });
    truncated("rows_visited", |b| {
        b.rows_visited.pop();
    });
    truncated("rows_joined", |b| {
        b.rows_joined.pop();
    });
    truncated("rows_scored", |b| {
        b.rows_scored.pop();
    });
    truncated("formula_support_complete", |b| {
        b.formula_support_complete.pop();
    });
    truncated("formula_mass_retained", |b| {
        b.formula_mass_retained.pop();
    });
    truncated("peaks_kept", |b| {
        b.peaks_kept.pop();
    });
    truncated("intensity_retained", |b| {
        b.intensity_retained.pop();
    });
    truncated("formula_source", |b| {
        b.formula_source.pop();
    });
    for (what, short) in &cases {
        assert!(
            matches!(
                pack(short, Some(&bit7(2)), ScoreKind::Raw, 2),
                Err(Error::Shape(_))
            ),
            "{what} truncation is not Error::Shape"
        );
    }
    // The side inputs are consumed too.
    assert!(matches!(
        pack(&batch, Some(&[0u32; 1]), ScoreKind::Raw, 2),
        Err(Error::Shape(_))
    ));
    assert!(matches!(
        pack(&batch, None, ScoreKind::Reranker(&[0.0; 1]), 2),
        Err(Error::Shape(_))
    ));
}

#[test]
fn validate_rejects_filled_slot_without_formula_provenance() {
    // Finding R1-C6: every FILLED candidate (finished or not) must have real
    // formula provenance — a source, a rank below `rows_scored`, counts that
    // are not all zero and that replay. The reviewer's corruption (finished
    // graph, zero counts, MAX row/rank, zero counters, nonfatal status) is
    // rejected, as is the slot-only variant with consistent counters.
    // Slot-only corruption: zero the counts and MAX the row/rank of a filled
    // slot while the spectrum counters stay consistent.
    let mut slot_only = good_packed();
    assert!(slot_only.returned_count[0] > 0, "slot 0 is filled");
    for e in 0..10 {
        slot_only.formula_counts[e] = 0;
    }
    slot_only.formula_row[0] = NO_FORMULA;
    slot_only.formula_rank[0] = NO_FORMULA;
    let err = slot_only.validate().unwrap_err();
    assert!(
        err.to_string().contains("no formula provenance"),
        "slot-only corruption names provenance: {err}"
    );
    // The full reviewer corruption: zero search counters and all.
    let mut full = good_packed();
    for e in 0..10 {
        full.formula_counts[e] = 0;
    }
    full.formula_row[0] = NO_FORMULA;
    full.formula_rank[0] = NO_FORMULA;
    full.rows_visited[0] = 0;
    full.rows_joined[0] = 0;
    full.rows_scored[0] = 0;
    let err = full.validate().unwrap_err();
    assert!(
        err.to_string().contains("no formula provenance"),
        "full corruption names provenance: {err}"
    );
}

#[test]
fn host_pack_preflights_output_domains_before_any_staging() {    // Finding I-C8: host `pack` preflights every staging/output size product
    // (checked u32/usize) BEFORE the first staging allocation AND before any
    // length validation that would need the big buffers. The reviewer's shape
    // (`B = 39,768,214`, `K = R = 1`, `T = 22`, `A = 16`: input stride 108
    // words fits u32, packed width 123 words does not) is refused through
    // `pack` itself with a batch whose vectors are EMPTY — the domain error
    // must fire before the length checks (which would name a field length)
    // and without allocating gigabytes.
    let rows = (u32::MAX as usize) / 108 - 1;
    assert_eq!(rows, 39_768_214);
    let batch = CandidateBatch {
        schema_version: SCHEMA_VERSION,
        batch: rows,
        trajectories: 1,
        max_steps: 22,
        max_atoms: 16,
        max_ring_closures: 4,
        spectrum_id: Vec::new(),
        trajectory: Vec::new(),
        actions: Vec::new(),
        length: Vec::new(),
        formula_row: Vec::new(),
        formula_log_prob: Vec::new(),
        trace_log_prob: Vec::new(),
        open_valence: Vec::new(),
        attachment_partition: Vec::new(),
        status: Vec::new(),
        evidence_status: Vec::new(),
        evidence_count: Vec::new(),
        evidence_peak_id: Vec::new(),
        evidence_hypothesis: Vec::new(),
        evidence_shift: Vec::new(),
        evidence_residual: Vec::new(),
        evidence_log_prob: Vec::new(),
        identity_resolution: Vec::new(),
        request_status: Vec::new(),
        rows_visited: Vec::new(),
        rows_joined: Vec::new(),
        rows_scored: Vec::new(),
        formula_support_complete: Vec::new(),
        formula_mass_retained: Vec::new(),
        peaks_kept: Vec::new(),
        intensity_retained: Vec::new(),
        formula_counts: Vec::new(),
        formula_source: Vec::new(),
        formula_rank: Vec::new(),
    };
    let err = pack(&batch, None, ScoreKind::Raw, 1).unwrap_err();
    assert!(
        matches!(err, Error::Shape(_)),
        "oversized output is Error::Shape: {err}"
    );
    let msg = err.to_string();
    assert!(
        msg.contains("u32 address domain"),
        "the output product is refused as a domain error: {msg}"
    );
    assert!(
        !msg.contains("has length"),
        "the domain preflight fires before any length validation: {msg}"
    );
}
