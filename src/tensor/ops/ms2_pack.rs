//! GPU kernels for ranking, compaction and the packed output (K3).
//!
//! Architecture `docs/MS2_V1_ARCHITECTURE.md` §4.4. Each kernel is a
//! line-for-line copy of its host twin's lane function(s) in
//! `crate::models::ms2::pack` (the twins were written first in the
//! kernel-expressible form: no fixed-size local arrays, no `wrapping_*`
//! methods, full buffers with explicit indices/strides/base offsets), so both
//! stay identical. Integer buffers are [`IdTensor`] (`u32`); every launch goes
//! through [`crate::backend::launch_1d_spans`] with one lane per output item;
//! shapes are checked to [`crate::error::Error::Shape`] before any launch.
//!
//! At most 6 arrays per kernel. Every output element is written by exactly one
//! lane (`returned_count` only by the lane with `r == 0`); padding gets an
//! explicit value; selection is by comparison, never by multiplying with a
//! mask. Loop-carried variables start from literals or buffer loads, never a
//! plain copy of a scalar argument. Inner-loop buffer loads are unconditional
//! (selection applies where the value is used), so no load serialises on
//! memory latency behind a branch.
//!
//! Layout words below name the constants of `crate::models::ms2::pack` they
//! equal: header offsets 0 (spectrum), 1 (trajectory), 2 (length), 3 (formula
//! row), 4 (formula rank), 5–14 (10 counts), 15 (status), 16 (evidence), 17
//! (resolution), 18 (attachment), 19 (token tail), float words 0 (formula
//! log-probability), 1 (trace log-probability), 2 (score). The kernels spell
//! them as literals, as `ms2_identity` does with its status bits.
//!
//! Deviations from the spec text: `ms2_rank` binds 5 arrays, not 4 — the extra
//! `rerank [B*K]` buffer carries caller-supplied scores for the `Reranker`
//! mode (zeros when unused); the raw score is `trace + formula` from the
//! `scores [B*K, 2]` buffer. `ms2_record_pack` is split into two launches
//! (integer `record`, 5 arrays, and float `record_f`, 3 arrays), since the
//! single-launch form would bind 7 arrays against wgpu's limit of 6; the float
//! launch carries the ranking score (raw sum or reranker value), so packed
//! rank order and packed scores always agree. Eligibility needs scores in the
//! validated score domain (−3e38, 3e38); anything outside it (NaN,
//! infinities, finite extremes at or beyond the bound, overflowing sums) is
//! treated as invalid and excluded from ranking, by specification: a NaN self-comparison is folded to
//! false by fast-math backends, so NaN would rank, and exact IEEE
//! classification is deliberately not implemented in the kernel. The host
//! twin ([`crate::models::ms2::pack`]) and
//! [`PackedCandidateBatch::validate`](crate::models::ms2::pack::PackedCandidateBatch::validate)
//! use the same range rule.
//!
//! Checked u32 addressing: every wrapper checks every stride, count and
//! largest accessed address of every input/output buffer against the u32
//! domain before any launch ([`Error::Shape`]); the device [`pack`] wrapper
//! additionally rejects `returned > per_spectrum` ([`Error::Config`]).
//!
//! What the integration task (P6.5) must provide per call: `actions`
//! `[B*K, T*4 + A + 4]` (the trajectory records generation already holds),
//! `traj_formula` `[B, K, 12]` (the allocation buffer), `scores` `[B*K, 2]`
//! f32 (trace log-probability from the `actions` record bits unchanged,
//! formula log-probability widened from `top_log_prob [B, F]` by the
//! trajectory's formula slot),
//! `evidence` `[B*K, 18]` (word 0 is the evidence status), `identity`
//! `[B*K, 2]` (as `ms2_graph_identity` writes it). Per-spectrum fields other
//! than `returned_count` never touch the device.
//!
//! CubeCL 0.10 has no `RangeInclusive::contains` or `usize::is_multiple_of`
//! in kernels, so lanes use explicit comparisons; the clippy lints for those
//! patterns are allowed here rather than rewritten. The lanes also guard
//! every `u32` division with an explicit zero check, which clippy reads as a
//! manual `checked_div`: CubeCL has no `checked_div` or `Option`, so the
//! kernel form stays and the lint is allowed here rather than rewritten.
#![allow(clippy::manual_range_contains, clippy::manual_is_multiple_of)]
#![allow(clippy::manual_checked_ops)]

use cubecl::prelude::*;

use crate::backend::{FloatElem, launch_1d_spans};
use crate::error::{Error, Result};
use crate::models::ms2::pack::{
    EVIDENCE_STRIDE, TRAJ_FORMULA_STRIDE, WF, check_u32_len, check_u32_product, record_width,
};
use crate::tensor::base::Tensor;
use crate::tensor::ops::index::IdTensor;

// ---------------------------------------------------------------------------
// Shared `#[cube]` helpers (copies of the twin helpers)
// ---------------------------------------------------------------------------

/// One guarded full-buffer read.
#[cube]
fn ms2_pack_slot(buf: &Array<u32>, base: u32, idx: u32) -> u32 {
    let addr = base + idx;
    let mut out: u32 = 0u32;
    if (addr as usize) < buf.len() {
        out = buf[addr as usize];
    }
    out
}

/// One guarded full-buffer write.
#[cube]
fn ms2_pack_put(buf: &mut Array<u32>, base: u32, idx: u32, value: u32) {
    let addr = base + idx;
    if (addr as usize) < buf.len() {
        buf[addr as usize] = value;
    }
}

/// Whether a trajectory may be ranked (copy of the twin's `eligible_lane`):
/// finished (bit 0), valid (bit 3 clear), not `duplicate_trace` (bit 4), not
/// `duplicate_graph` (bit 7 of `bits` when `use_graph` is 1), not
/// `request_failed` (bit 6); `bad` is 1 exactly when a raw log-probability or
/// the ranking score lies outside the validated score domain (−3e38, 3e38);
/// anything outside it is treated as invalid and excluded from ranking, by
/// specification. The
/// finiteness test is a range test: a NaN self-comparison is folded to false
/// by fast-math backends, so NaN would rank; exact IEEE classification is
/// deliberately not implemented in the kernel.
#[cube]
fn ms2_pack_eligible(status: u32, bits: u32, use_graph: u32, bad: u32) -> u32 {
    let mut e: u32 = 1u32;
    if status & 1u32 == 0u32 {
        e = 0u32;
    }
    if status & 8u32 != 0u32 {
        e = 0u32;
    }
    if status & 16u32 != 0u32 {
        e = 0u32;
    }
    if status & 64u32 != 0u32 {
        e = 0u32;
    }
    if use_graph == 1u32 && bits & 128u32 != 0u32 {
        e = 0u32;
    }
    if bad == 1u32 {
        e = 0u32;
    }
    e
}

// ---------------------------------------------------------------------------
// `ms2_rank`: lane per trajectory
// ---------------------------------------------------------------------------

/// Lane per trajectory of [`rank`]; a line-for-line copy of
/// `pack::rank_lane` over `Array`s.
///
/// `scores` is always f32 (the gathered ranking terms); `rerank` is the
/// neural dtype, widened to f32 on load.
#[allow(clippy::too_many_arguments)]
#[allow(unused_assignments)]
#[cube(launch_unchecked)]
fn ms2_rank_kernel<F: Float + CubeElement>(
    actions: &Array<u32>,
    identity: &Array<u32>,
    scores: &Array<f32>,
    rerank: &Array<F>,
    rank_out: &mut Array<u32>,
    record_stride: u32,
    len_field: u32,
    per_spectrum: u32,
    use_graph: u32,
    use_rerank: u32,
    lanes: usize,
    span: usize,
) {
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    for pos in start..end {
        let record = pos as u32;
        let abase = record * record_stride;
        let status = ms2_pack_slot(actions, abase, len_field + 1u32);
        let bits = ms2_pack_slot(identity, record * 2u32, 0u32);
        // The ranking score is the f32 sum of the f32-widened terms on every
        // neural dtype, equal to the host `pack`'s arithmetic (spec §4.4):
        // each term is widened to f32 on load and the sum is f32.
        let mut trace_v: f32 = 0.0f32;
        if ((record * 2u32) as usize) < scores.len() {
            trace_v = scores[(record * 2u32) as usize];
        }
        let mut formula_v: f32 = 0.0f32;
        if ((record * 2u32 + 1u32) as usize) < scores.len() {
            formula_v = scores[(record * 2u32 + 1u32) as usize];
        }
        let mut re_v: f32 = 0.0f32;
        if (record as usize) < rerank.len() {
            re_v = f32::cast_from(rerank[record as usize]);
        }
        let mut my: f32 = trace_v + formula_v;
        if use_rerank == 1u32 {
            my = re_v;
        }
        let mut bad: u32 = 1u32;
        if trace_v > -3.0e38_f32
            && trace_v < 3.0e38_f32
            && formula_v > -3.0e38_f32
            && formula_v < 3.0e38_f32
            && my > -3.0e38_f32
            && my < 3.0e38_f32
        {
            bad = 0u32;
        }
        let mine = ms2_pack_eligible(status, bits, use_graph, bad);
        let mut out_rank: u32 = 4294967295u32;
        if mine == 1u32 && per_spectrum > 0u32 {
            let base = (record / per_spectrum) * per_spectrum;
            let mut r: u32 = 0u32;
            let mut j: u32 = 0u32;
            while j < per_spectrum {
                let other = base + j;
                let obase = other * record_stride;
                let st = ms2_pack_slot(actions, obase, len_field + 1u32);
                let ob = ms2_pack_slot(identity, other * 2u32, 0u32);
                let mut tv: f32 = 0.0f32;
                if ((other * 2u32) as usize) < scores.len() {
                    tv = scores[(other * 2u32) as usize];
                }
                let mut fv: f32 = 0.0f32;
                if ((other * 2u32 + 1u32) as usize) < scores.len() {
                    fv = scores[(other * 2u32 + 1u32) as usize];
                }
                let mut rv: f32 = 0.0f32;
                if (other as usize) < rerank.len() {
                    rv = f32::cast_from(rerank[other as usize]);
                }
                let mut s: f32 = tv + fv;
                if use_rerank == 1u32 {
                    s = rv;
                }
                let mut sbad: u32 = 1u32;
                if tv > -3.0e38_f32
                    && tv < 3.0e38_f32
                    && fv > -3.0e38_f32
                    && fv < 3.0e38_f32
                    && s > -3.0e38_f32
                    && s < 3.0e38_f32
                {
                    sbad = 0u32;
                }
                let e = ms2_pack_eligible(st, ob, use_graph, sbad);
                if e == 1u32 && (s > my || (s == my && other < record)) {
                    r += 1u32;
                }
                j += 1u32;
            }
            out_rank = r;
        }
        if (record as usize) < rank_out.len() {
            rank_out[record as usize] = out_rank;
        }
    }
}

/// Ranks of one `(B, K)` bucket, lane per trajectory.
///
/// `actions` is `[rows, T*4 + A + 4]`, `identity` is `[rows, 2]`, `scores` is
/// `[rows, 2]` f32 `(trace_log_prob, formula_log_prob)` on every neural
/// dtype, `rerank` is `[rows]` caller scores (read only when `use_rerank` is 1), `rank_out` is
/// `[rows]`. Flags are 0/1. A trajectory is eligible only with its raw
/// log-probabilities and ranking score inside the validated score domain
/// (−3e38, 3e38); values outside it, NaN and infinities are ineligible BY
/// specification (range test, never NaN self-comparison). Every stride, count and
/// largest accessed address is checked against the u32 domain before any
/// launch ([`Error::Shape`]).
#[allow(clippy::too_many_arguments)]
pub fn rank<R: Runtime, E: FloatElem>(
    actions: &IdTensor<R>,
    identity: &IdTensor<R>,
    scores: &Tensor<R, f32>,
    rerank: &Tensor<R, E>,
    rank_out: &mut IdTensor<R>,
    steps: usize,
    atoms_cap: u32,
    per_spectrum: usize,
    use_graph: u32,
    use_rerank: u32,
) -> Result<()> {
    if actions.shape().rank() != 2
        || identity.shape().rank() != 2
        || scores.shape().rank() != 2
        || rerank.shape().rank() != 1
        || rank_out.shape().rank() != 1
    {
        return Err(Error::shape(format!(
            "rank needs actions [rows, stride], identity [rows, 2], scores [rows, 2], rerank [rows] and rank [rows], got {} and {} and {} and {} and {}",
            actions.shape(),
            identity.shape(),
            scores.shape(),
            rerank.shape(),
            rank_out.shape()
        )));
    }
    let rows = actions.shape().dim(0);
    let record_stride = actions.shape().dim(1);
    let want_stride = steps
        .checked_mul(4)
        .and_then(|v| v.checked_add(atoms_cap as usize))
        .and_then(|v| v.checked_add(4));
    if want_stride != Some(record_stride) {
        return Err(Error::shape(format!(
            "rank needs actions stride T*4 + A + 4 = {want_stride:?} for T {steps} A {atoms_cap}, got {record_stride}"
        )));
    }
    let want_identity: &[usize] = &[rows, 2];
    let want_scores: &[usize] = &[rows, 2];
    if identity.shape().dims() != want_identity
        || scores.shape().dims() != want_scores
        || rerank.len() != rows
        || rank_out.len() != rows
    {
        return Err(Error::shape(format!(
            "rank needs identity [{rows}, 2], scores [{rows}, 2], rerank [{rows}] and rank [{rows}], got {} and {} and {} and {}",
            identity.shape(),
            scores.shape(),
            rerank.shape(),
            rank_out.shape()
        )));
    }
    if atoms_cap > 32 {
        return Err(Error::shape(format!(
            "rank needs atoms_cap <= 32, got {atoms_cap}"
        )));
    }
    if per_spectrum == 0 || rows % per_spectrum != 0 {
        return Err(Error::shape(format!(
            "rank needs rows {rows} divisible by per_spectrum {per_spectrum}"
        )));
    }
    if use_graph > 1 || use_rerank > 1 {
        return Err(Error::shape(format!(
            "rank needs use_graph and use_rerank in 0..=1, got {use_graph} and {use_rerank}"
        )));
    }
    // Checked u32 addressing (finding E2): every stride, count and largest
    // accessed address before any launch; unsupported sizes are
    // `Error::Shape`, never a narrowed `as u32`.
    check_u32_len("rank rows", rows)?;
    check_u32_len("rank record_stride", record_stride)?;
    check_u32_len("rank per_spectrum", per_spectrum)?;
    let _ = check_u32_product("rank actions addresses", rows, record_stride)?;
    let _ = check_u32_product("rank identity/scores addresses", rows, 2)?;
    if rows == 0 {
        return Ok(());
    }
    let len_field_usize = steps
        .checked_mul(4)
        .and_then(|v| v.checked_add(atoms_cap as usize))
        .ok_or_else(|| {
            Error::shape(format!(
                "rank needs T*4 + A to fit usize for T {steps} A {atoms_cap}"
            ))
        })?;
    let len_field = check_u32_len("rank len_field", len_field_usize)?;
    let client = actions.client();
    let (count, dim, span) = launch_1d_spans(client, rows, record_stride);
    unsafe {
        ms2_rank_kernel::launch_unchecked::<E, R>(
            client,
            count,
            dim,
            actions.arg(),
            identity.arg(),
            scores.arg(),
            rerank.arg(),
            rank_out.arg(),
            record_stride as u32,
            len_field,
            per_spectrum as u32,
            use_graph,
            use_rerank,
            rows,
            span,
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// `ms2_record_pack`: lane per trajectory (integer record)
// ---------------------------------------------------------------------------

/// Lane per trajectory of [`scores_fill`]; a line-for-line copy of
/// `pack::scores_fill_lane` over `Array`s.
///
/// The trace term is the stored f32 bits unchanged; the formula term is the
/// resident `top_log_prob` value widened to f32. The gathered `scores`
/// buffer is always f32, on every neural dtype (spec §4.4).
#[cube(launch_unchecked)]
fn ms2_scores_fill_kernel<F: Float + CubeElement>(
    actions: &Array<u32>,
    traj_formula: &Array<u32>,
    top_log_prob: &Array<F>,
    scores: &mut Array<f32>,
    record_stride: u32,
    len_field: u32,
    per_spectrum: u32,
    formulas: u32,
    lanes: usize,
    span: usize,
) {
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    for pos in start..end {
        let record = pos as u32;
        let abase = record * record_stride;
        let mut tl: f32 = 0.0f32;
        if ((abase + len_field + 2u32) as usize) < actions.len() {
            // The trace log-probability rides as f32 bits on every neural
            // dtype: decode as f32 and store unchanged (a direct
            // F::reinterpret would need equal sizes and panics for bf16
            // during kernel expansion).
            tl = f32::reinterpret(
                actions[(abase + len_field + 2u32) as usize],
            );
        }
        let mut fl: f32 = 0.0f32;
        let b = record / per_spectrum;
        let k = record % per_spectrum;
        let s = ms2_pack_slot(traj_formula, (b * per_spectrum + k) * 12u32, 0u32);
        if s != 4294967295u32 && s < formulas
            && ((b * formulas + s) as usize) < top_log_prob.len()
        {
            fl = f32::cast_from(top_log_prob[(b * formulas + s) as usize]);
        }
        if ((record * 2u32) as usize) < scores.len() {
            scores[(record * 2u32) as usize] = tl;
        }
        if ((record * 2u32 + 1u32) as usize) < scores.len() {
            scores[(record * 2u32 + 1u32) as usize] = fl;
        }
    }
}

/// Ranking scores of one `(B, K)` bucket, lane per trajectory.
///
/// `actions` is `[rows, T*4 + A + 4]` (the trace log-probability rides as
/// `f32` bits in the spare word), `traj_formula` is `[B, K, 12]` (word 0 is
/// the retained slot, `u32::MAX` when there is none), `top_log_prob` is
/// `[B, F]` of the neural dtype and `scores` is `[rows, 2]` f32
/// `(trace_log_prob, formula_log_prob)` on every neural dtype. Exactly 1 launch.
#[allow(clippy::too_many_arguments)]
pub fn scores_fill<R: Runtime, E: FloatElem>(
    actions: &IdTensor<R>,
    traj_formula: &IdTensor<R>,
    top_log_prob: &Tensor<R, E>,
    scores: &mut Tensor<R, f32>,
    steps: usize,
    atoms_cap: u32,
    per_spectrum: usize,
    formulas: usize,
) -> Result<()> {
    if actions.shape().rank() != 2
        || traj_formula.shape().rank() != 3
        || top_log_prob.shape().rank() != 2
        || scores.shape().rank() != 2
    {
        return Err(Error::shape(format!(
            "scores_fill needs actions [rows, stride], traj_formula [B, K, 12], top_log_prob [B, F] and scores [rows, 2], got {} and {} and {} and {}",
            actions.shape(),
            traj_formula.shape(),
            top_log_prob.shape(),
            scores.shape()
        )));
    }
    let rows = actions.shape().dim(0);
    let record_stride = actions.shape().dim(1);
    let want_stride = steps
        .checked_mul(4)
        .and_then(|v| v.checked_add(atoms_cap as usize))
        .and_then(|v| v.checked_add(4));
    if want_stride != Some(record_stride) {
        return Err(Error::shape(format!(
            "scores_fill needs actions stride T*4 + A + 4 = {want_stride:?} for T {steps} A {atoms_cap}, got {record_stride}"
        )));
    }
    if atoms_cap > 32 {
        return Err(Error::shape(format!(
            "scores_fill needs atoms_cap <= 32, got {atoms_cap}"
        )));
    }
    if per_spectrum == 0 || rows % per_spectrum != 0 {
        return Err(Error::shape(format!(
            "scores_fill needs rows {rows} divisible by per_spectrum {per_spectrum}"
        )));
    }
    let batch = rows / per_spectrum;
    let want_tf: &[usize] = &[batch, per_spectrum, TRAJ_FORMULA_STRIDE as usize];
    let want_lp: &[usize] = &[batch, formulas];
    let want_scores: &[usize] = &[rows, 2];
    if traj_formula.shape().dims() != want_tf
        || top_log_prob.shape().dims() != want_lp
        || scores.shape().dims() != want_scores
    {
        return Err(Error::shape(format!(
            "scores_fill needs traj_formula [{batch}, {per_spectrum}, 12], top_log_prob [{batch}, {formulas}] and scores [{rows}, 2], got {} and {} and {}",
            traj_formula.shape(),
            top_log_prob.shape(),
            scores.shape()
        )));
    }
    // Checked u32 addressing before any launch.
    check_u32_len("scores_fill rows", rows)?;
    check_u32_len("scores_fill record_stride", record_stride)?;
    check_u32_len("scores_fill per_spectrum", per_spectrum)?;
    check_u32_len("scores_fill formulas", formulas)?;
    let _ = check_u32_product("scores_fill actions addresses", rows, record_stride)?;
    let _ = check_u32_product("scores_fill scores addresses", rows, 2)?;
    if rows == 0 {
        return Ok(());
    }
    let len_field_usize = steps
        .checked_mul(4)
        .and_then(|v| v.checked_add(atoms_cap as usize))
        .ok_or_else(|| {
            Error::shape(format!(
                "scores_fill needs T*4 + A to fit usize for T {steps} A {atoms_cap}"
            ))
        })?;
    let len_field = check_u32_len("scores_fill len_field", len_field_usize)?;
    let client = actions.client();
    let (count, dim, span) = launch_1d_spans(client, rows, record_stride);
    unsafe {
        ms2_scores_fill_kernel::launch_unchecked::<E, R>(
            client,
            count,
            dim,
            actions.arg(),
            traj_formula.arg(),
            top_log_prob.arg(),
            scores.arg(),
            record_stride as u32,
            len_field,
            per_spectrum as u32,
            formulas as u32,
            rows,
            span,
        );
    }
    Ok(())
}

/// Lane per trajectory of [`record_pack`]; a line-for-line copy of
/// `pack::record_pack_lane` over `Array`s.
#[allow(clippy::too_many_arguments)]
#[allow(unused_assignments)]
#[cube(launch_unchecked)]
fn ms2_record_pack_kernel(
    actions: &Array<u32>,
    traj_formula: &Array<u32>,
    evidence: &Array<u32>,
    identity: &Array<u32>,
    record: &mut Array<u32>,
    record_stride: u32,
    steps: u32,
    atoms_cap: u32,
    record_w: u32,
    per_spectrum: u32,
    ev_stride: u32,
    use_identity: u32,
    lanes: usize,
    span: usize,
) {
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    for pos in start..end {
        let traj_record = pos as u32;
        let abase = traj_record * record_stride;
        let tfbase = traj_record * 12u32;
        let evbase = traj_record * ev_stride;
        let idbase = traj_record * 2u32;
        let len_field = steps * 4u32 + atoms_cap;
        let length = ms2_pack_slot(actions, abase, len_field);
        let status_in = ms2_pack_slot(actions, abase, len_field + 1u32);
        let formula_row = ms2_pack_slot(actions, abase, len_field + 3u32);
        let formula_slot = ms2_pack_slot(traj_formula, tfbase, 0u32);
        let ev_status = ms2_pack_slot(evidence, evbase, 0u32);
        let mut status = status_in;
        let mut resolution: u32 = 0u32;
        if use_identity == 1u32 {
            let bits = ms2_pack_slot(identity, idbase, 0u32);
            status |= bits;
            resolution = ms2_pack_slot(identity, idbase, 1u32);
        }
        let mut spectrum: u32 = 0u32;
        let mut traj: u32 = traj_record;
        if per_spectrum > 0u32 {
            spectrum = traj_record / per_spectrum;
            traj = traj_record % per_spectrum;
        }
        let out_base = traj_record * record_w;
        ms2_pack_put(record, out_base, 0u32, spectrum);
        ms2_pack_put(record, out_base, 1u32, traj);
        ms2_pack_put(record, out_base, 2u32, length);
        ms2_pack_put(record, out_base, 3u32, formula_row);
        ms2_pack_put(record, out_base, 4u32, formula_slot);
        let mut e: u32 = 0u32;
        while e < 10u32 {
            ms2_pack_put(
                record,
                out_base,
                5u32 + e,
                ms2_pack_slot(traj_formula, tfbase, 2u32 + e),
            );
            e += 1u32;
        }
        ms2_pack_put(record, out_base, 15u32, status);
        ms2_pack_put(record, out_base, 16u32, ev_status);
        ms2_pack_put(record, out_base, 17u32, resolution);
        ms2_pack_put(record, out_base, 18u32, 0u32);
        let mut w: u32 = 0u32;
        while w < steps * 4u32 {
            ms2_pack_put(
                record,
                out_base,
                19u32 + w,
                ms2_pack_slot(actions, abase, w),
            );
            w += 1u32;
        }
        let mut v: u32 = 0u32;
        while v < atoms_cap {
            ms2_pack_put(
                record,
                out_base,
                19u32 + steps * 4u32 + v,
                ms2_pack_slot(actions, abase, steps * 4u32 + v),
            );
            v += 1u32;
        }
    }
}

/// Integer records of one `(B, K)` bucket, lane per trajectory.
///
/// `actions` is `[rows, T*4 + A + 4]`, `traj_formula` is `[B, K, 12]`,
/// `evidence` is `[rows, 18]`, `identity` is `[rows, 2]`, `record` is
/// `[rows, W]`.
#[allow(clippy::too_many_arguments)]
pub fn record_pack<R: Runtime>(
    actions: &IdTensor<R>,
    traj_formula: &IdTensor<R>,
    evidence: &IdTensor<R>,
    identity: &IdTensor<R>,
    record: &mut IdTensor<R>,
    steps: usize,
    atoms_cap: u32,
    per_spectrum: usize,
    use_identity: u32,
) -> Result<()> {
    if actions.shape().rank() != 2
        || traj_formula.shape().rank() != 3
        || evidence.shape().rank() != 2
        || identity.shape().rank() != 2
        || record.shape().rank() != 2
    {
        return Err(Error::shape(format!(
            "record_pack needs actions [rows, stride], traj_formula [B, K, 12], evidence [rows, 18], identity [rows, 2] and record [rows, W], got {} and {} and {} and {} and {}",
            actions.shape(),
            traj_formula.shape(),
            evidence.shape(),
            identity.shape(),
            record.shape()
        )));
    }
    let rows = actions.shape().dim(0);
    let record_stride = actions.shape().dim(1);
    let want_stride = steps
        .checked_mul(4)
        .and_then(|v| v.checked_add(atoms_cap as usize))
        .and_then(|v| v.checked_add(4));
    if want_stride != Some(record_stride) {
        return Err(Error::shape(format!(
            "record_pack needs actions stride T*4 + A + 4 = {want_stride:?} for T {steps} A {atoms_cap}, got {record_stride}"
        )));
    }
    if atoms_cap > 32 {
        return Err(Error::shape(format!(
            "record_pack needs atoms_cap <= 32, got {atoms_cap}"
        )));
    }
    if per_spectrum == 0 || rows % per_spectrum != 0 {
        return Err(Error::shape(format!(
            "record_pack needs rows {rows} divisible by per_spectrum {per_spectrum}"
        )));
    }
    let width = record_width(steps, atoms_cap as usize);
    let batch = rows / per_spectrum;
    let want_tf: &[usize] = &[batch, per_spectrum, TRAJ_FORMULA_STRIDE as usize];
    let want_ev: &[usize] = &[rows, EVIDENCE_STRIDE as usize];
    let want_id: &[usize] = &[rows, 2];
    let want_rec: &[usize] = &[rows, width];
    if traj_formula.shape().dims() != want_tf
        || evidence.shape().dims() != want_ev
        || identity.shape().dims() != want_id
        || record.shape().dims() != want_rec
    {
        return Err(Error::shape(format!(
            "record_pack needs traj_formula [{batch}, {per_spectrum}, 12], evidence [{rows}, 18], identity [{rows}, 2] and record [{rows}, {width}], got {} and {} and {} and {}",
            traj_formula.shape(),
            evidence.shape(),
            identity.shape(),
            record.shape()
        )));
    }
    if use_identity > 1 {
        return Err(Error::shape(format!(
            "record_pack needs use_identity in 0..=1, got {use_identity}"
        )));
    }
    // Checked u32 addressing (finding E2) before any launch.
    check_u32_len("record_pack rows", rows)?;
    check_u32_len("record_pack record_stride", record_stride)?;
    check_u32_len("record_pack per_spectrum", per_spectrum)?;
    check_u32_len("record_pack steps", steps)?;
    check_u32_len("record_pack width", width)?;
    let _ = check_u32_product("record_pack actions addresses", rows, record_stride)?;
    let _ = check_u32_product(
        "record_pack traj_formula addresses",
        rows,
        TRAJ_FORMULA_STRIDE as usize,
    )?;
    let _ = check_u32_product(
        "record_pack evidence addresses",
        rows,
        EVIDENCE_STRIDE as usize,
    )?;
    let _ = check_u32_product("record_pack identity addresses", rows, 2)?;
    let _ = check_u32_product("record_pack record addresses", rows, width)?;
    if rows == 0 {
        return Ok(());
    }
    let client = actions.client();
    let (count, dim, span) = launch_1d_spans(client, rows, width);
    unsafe {
        ms2_record_pack_kernel::launch_unchecked::<R>(
            client,
            count,
            dim,
            actions.arg(),
            traj_formula.arg(),
            evidence.arg(),
            identity.arg(),
            record.arg(),
            record_stride as u32,
            steps as u32,
            atoms_cap,
            width as u32,
            per_spectrum as u32,
            EVIDENCE_STRIDE,
            use_identity,
            rows,
            span,
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// `ms2_record_pack_f`: lane per trajectory (float record)
// ---------------------------------------------------------------------------

/// Lane per trajectory of [`record_pack_f`]: the formula log-probability,
/// the trace log-probability and the ranking score (the f32 raw sum of the
/// f32 terms, or `rerank[record]` widened to f32 when `use_rerank` is
/// 1) from `scores [rows, 2]` (always f32). The packed float record is always f32, on
/// every neural dtype, so device words equal the host `pack` exactly.
#[cube(launch_unchecked)]
fn ms2_record_pack_f_kernel<F: Float + CubeElement>(
    scores: &Array<f32>,
    rerank: &Array<F>,
    record_f: &mut Array<f32>,
    use_rerank: u32,
    lanes: usize,
    span: usize,
) {
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    for pos in start..end {
        let record = pos as u32;
        let mut tl: f32 = 0.0f32;
        if ((record * 2u32) as usize) < scores.len() {
            tl = scores[(record * 2u32) as usize];
        }
        let mut fl: f32 = 0.0f32;
        if ((record * 2u32 + 1u32) as usize) < scores.len() {
            fl = scores[(record * 2u32 + 1u32) as usize];
        }
        let mut rv: f32 = 0.0f32;
        if (record as usize) < rerank.len() {
            rv = f32::cast_from(rerank[record as usize]);
        }
        let mut sc: f32 = tl + fl;
        if use_rerank == 1u32 {
            sc = rv;
        }
        let base = record * 3u32;
        if (base as usize) < record_f.len() {
            record_f[base as usize] = fl;
        }
        if ((base + 1u32) as usize) < record_f.len() {
            record_f[(base + 1u32) as usize] = tl;
        }
        if ((base + 2u32) as usize) < record_f.len() {
            record_f[(base + 2u32) as usize] = sc;
        }
    }
}

/// Float records of one `(B, K)` bucket, lane per trajectory: `scores` is
/// `[rows, 2]` f32 on every neural dtype, `rerank` is `[rows]` caller scores (read only when
/// `use_rerank` is 1), `record_f` is `[rows, 3]` and always f32, on every
/// neural dtype, so device words equal the host `pack` exactly.
pub fn record_pack_f<R: Runtime, E: FloatElem>(
    scores: &Tensor<R, f32>,
    rerank: &Tensor<R, E>,
    record_f: &mut Tensor<R, f32>,
    use_rerank: u32,
) -> Result<()> {
    if scores.shape().rank() != 2 || rerank.shape().rank() != 1 || record_f.shape().rank() != 2 {
        return Err(Error::shape(format!(
            "record_pack_f needs scores [rows, 2], rerank [rows] and record_f [rows, 3], got {} and {} and {}",
            scores.shape(),
            rerank.shape(),
            record_f.shape()
        )));
    }
    let rows = scores.shape().dim(0);
    let want_scores: &[usize] = &[rows, 2];
    let want_f: &[usize] = &[rows, WF];
    if scores.shape().dims() != want_scores
        || rerank.len() != rows
        || record_f.shape().dims() != want_f
    {
        return Err(Error::shape(format!(
            "record_pack_f needs scores [{rows}, 2], rerank [{rows}] and record_f [{rows}, 3], got {} and {} and {}",
            scores.shape(),
            rerank.shape(),
            record_f.shape()
        )));
    }
    if use_rerank > 1 {
        return Err(Error::shape(format!(
            "record_pack_f needs use_rerank in 0..=1, got {use_rerank}"
        )));
    }
    // Checked u32 addressing (finding E2) before any launch.
    check_u32_len("record_pack_f rows", rows)?;
    let _ = check_u32_product("record_pack_f scores addresses", rows, 2)?;
    let _ = check_u32_product("record_pack_f record_f addresses", rows, WF)?;
    if rows == 0 {
        return Ok(());
    }
    let client = scores.client();
    let (count, dim, span) = launch_1d_spans(client, rows, WF);
    unsafe {
        ms2_record_pack_f_kernel::launch_unchecked::<E, R>(
            client,
            count,
            dim,
            scores.arg(),
            rerank.arg(),
            record_f.arg(),
            use_rerank,
            rows,
            span,
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// `ms2_pack`: lane per output slot `(b, r)`
// ---------------------------------------------------------------------------

/// Lane per output slot of [`pack`]; a line-for-line copy of
/// `pack::pack_lane` (with `pack::returned_count_lane` inlined for the
/// `r == 0` lane, the single writer of `returned_count`) over `Array`s.
#[allow(clippy::too_many_arguments)]
#[allow(unused_assignments)]
#[cube(launch_unchecked)]
fn ms2_pack_kernel(
    rank_in: &Array<u32>,
    record: &Array<u32>,
    record_f: &Array<f32>,
    packed: &mut Array<u32>,
    packed_f: &mut Array<f32>,
    returned_count: &mut Array<u32>,
    record_w: u32,
    record_fw: u32,
    per_spectrum: u32,
    returned: u32,
    lanes: usize,
    span: usize,
) {
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    for pos in start..end {
        let slot_pos = pos as u32;
        let mut b: u32 = 0u32;
        let mut slot_r: u32 = slot_pos;
        if returned > 0u32 {
            b = slot_pos / returned;
            slot_r = slot_pos % returned;
        }
        let mut found: u32 = 4294967295u32;
        let mut j: u32 = 0u32;
        let base = b * per_spectrum;
        while j < per_spectrum {
            if ms2_pack_slot(rank_in, 0u32, base + j) == slot_r {
                found = j;
            }
            j += 1u32;
        }
        let out_base = slot_pos * record_w;
        let out_fbase = slot_pos * record_fw;
        if found == 4294967295u32 {
            let mut w: u32 = 0u32;
            while w < record_w {
                ms2_pack_put(packed, out_base, w, 0u32);
                w += 1u32;
            }
            ms2_pack_put(packed, out_base, 0u32, b);
            ms2_pack_put(packed, out_base, 1u32, 4294967295u32);
            ms2_pack_put(packed, out_base, 3u32, 4294967295u32);
            ms2_pack_put(packed, out_base, 4u32, 4294967295u32);
            let mut q: u32 = 0u32;
            while q < record_fw {
                let dst = out_fbase + q;
                if (dst as usize) < packed_f.len() {
                    packed_f[dst as usize] = 0.0f32;
                }
                q += 1u32;
            }
        } else {
            let src = (base + found) * record_w;
            let mut w: u32 = 0u32;
            while w < record_w {
                ms2_pack_put(packed, out_base, w, ms2_pack_slot(record, src, w));
                w += 1u32;
            }
            let src_f = (base + found) * record_fw;
            let mut q: u32 = 0u32;
            while q < record_fw {
                let saddr = src_f + q;
                let mut v: f32 = 0.0f32;
                if (saddr as usize) < record_f.len() {
                    v = record_f[saddr as usize];
                }
                let daddr = out_fbase + q;
                if (daddr as usize) < packed_f.len() {
                    packed_f[daddr as usize] = v;
                }
                q += 1u32;
            }
        }
        if slot_r == 0u32 {
            let mut count: u32 = 0u32;
            let mut k: u32 = 0u32;
            while k < per_spectrum {
                if ms2_pack_slot(rank_in, 0u32, base + k) != 4294967295u32 {
                    count += 1u32;
                }
                k += 1u32;
            }
            if count > returned {
                count = returned;
            }
            if (b as usize) < returned_count.len() {
                returned_count[b as usize] = count;
            }
        }
    }
}

/// Packed records of one `(B, R)` output, lane per slot `(b, r)`.
///
/// `rank_in` is `[B*K]`, `record` is `[B*K, W]`, `record_f` is `[B*K, 3]`
/// (always f32, on every neural dtype), `packed` is `[B, R, W]`, `packed_f`
/// is `[B, R, 3]` (always f32), `returned_count` is `[B]`. `1 <= R <= K`:
/// `returned > per_spectrum` is [`Error::Config`]
/// (finding E4), while `returned > eligible` stays legal and leaves unfilled
/// slots. Every stride, count and largest accessed address is checked against
/// the u32 domain before any launch ([`Error::Shape`], finding E2).
#[allow(clippy::too_many_arguments)]
pub fn pack<R: Runtime>(
    rank_in: &IdTensor<R>,
    record: &IdTensor<R>,
    record_f: &Tensor<R, f32>,
    packed: &mut IdTensor<R>,
    packed_f: &mut Tensor<R, f32>,
    returned_count: &mut IdTensor<R>,
    steps: usize,
    atoms_cap: u32,
    per_spectrum: usize,
    returned: usize,
) -> Result<()> {
    if rank_in.shape().rank() != 1
        || record.shape().rank() != 2
        || record_f.shape().rank() != 2
        || packed.shape().rank() != 3
        || packed_f.shape().rank() != 3
        || returned_count.shape().rank() != 1
    {
        return Err(Error::shape(format!(
            "pack needs rank [rows], record [rows, W], record_f [rows, 3], packed [B, R, W], packed_f [B, R, 3] and returned_count [B], got {} and {} and {} and {} and {} and {}",
            rank_in.shape(),
            record.shape(),
            record_f.shape(),
            packed.shape(),
            packed_f.shape(),
            returned_count.shape()
        )));
    }
    let rows = rank_in.len();
    if atoms_cap > 32 {
        return Err(Error::shape(format!(
            "pack needs atoms_cap <= 32, got {atoms_cap}"
        )));
    }
    let batch = rows.checked_div(per_spectrum).ok_or_else(|| {
        Error::shape(format!(
            "pack needs per_spectrum >= 1, got {per_spectrum}"
        ))
    })?;
    if rows % per_spectrum != 0 {
        return Err(Error::shape(format!(
            "pack needs rows {rows} divisible by per_spectrum {per_spectrum}"
        )));
    }
    if returned == 0 {
        return Err(Error::shape(format!(
            "pack needs returned >= 1, got {returned}"
        )));
    }
    // Finding E4: `returned > per_spectrum` (R > K) is `Error::Config`,
    // refused separately from `returned > eligible`, which stays legal and
    // leaves unfilled slots.
    if returned > per_spectrum {
        return Err(Error::config(format!(
            "pack needs returned {returned} <= per_spectrum {per_spectrum} (1 <= R <= K)"
        )));
    }
    let width = record_width(steps, atoms_cap as usize);
    let want_record: &[usize] = &[rows, width];
    let want_record_f: &[usize] = &[rows, WF];
    let want_packed: &[usize] = &[batch, returned, width];
    let want_packed_f: &[usize] = &[batch, returned, WF];
    let want_counts: &[usize] = &[batch];
    if record.shape().dims() != want_record
        || record_f.shape().dims() != want_record_f
        || packed.shape().dims() != want_packed
        || packed_f.shape().dims() != want_packed_f
        || returned_count.shape().dims() != want_counts
    {
        return Err(Error::shape(format!(
            "pack needs record [{rows}, {width}], record_f [{rows}, 3], packed [{batch}, {returned}, {width}], packed_f [{batch}, {returned}, 3] and returned_count [{batch}], got {} and {} and {} and {} and {}",
            record.shape(),
            record_f.shape(),
            packed.shape(),
            packed_f.shape(),
            returned_count.shape()
        )));
    }
    // Checked u32 addressing (finding E2): every stride, count and largest
    // accessed address before any launch; unsupported sizes are
    // `Error::Shape`, never a narrowed `as u32`.
    check_u32_len("pack rows", rows)?;
    check_u32_len("pack batch", batch)?;
    check_u32_len("pack per_spectrum", per_spectrum)?;
    check_u32_len("pack returned", returned)?;
    check_u32_len("pack width", width)?;
    check_u32_len("pack steps", steps)?;
    let lanes = check_u32_product("pack slots", batch, returned)?;
    let _ = check_u32_product("pack record addresses", rows, width)?;
    let _ = check_u32_product("pack record_f addresses", rows, WF)?;
    let _ = check_u32_product("pack packed addresses", lanes, width)?;
    let _ = check_u32_product("pack packed_f addresses", lanes, WF)?;
    if lanes == 0 {
        return Ok(());
    }
    let client = rank_in.client();
    let (count, dim, span) = launch_1d_spans(client, lanes, width);
    unsafe {
        ms2_pack_kernel::launch_unchecked::<R>(
            client,
            count,
            dim,
            rank_in.arg(),
            record.arg(),
            record_f.arg(),
            packed.arg(),
            packed_f.arg(),
            returned_count.arg(),
            width as u32,
            WF as u32,
            per_spectrum as u32,
            returned as u32,
            lanes,
            span,
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// `ms2_pack_evidence`: lane per output slot `(b, r)` (I3b)
// ---------------------------------------------------------------------------

/// Lane per output slot of [`pack_evidence`]; a line-for-line copy of
/// `pack::pack_evidence_lane` over `Array`s.
///
/// The trajectory of rank `slot_r` is found by scanning the spectrum's ranks
/// (as in [`ms2_pack_kernel`]); its 18 evidence words ride as kept-peak
/// positions (the host maps them to original peak ids at readout exactly as
/// `generate` does) and each stored record's assignment log-probability is
/// looked up by `(slot, position, hypothesis)`. All loads are unconditional
/// (clamped); masks apply where the value is used. Per-lane work is
/// `O(K + E)` with `E = 4`, never quadratic in an unbounded window.
#[allow(clippy::too_many_arguments)]
#[allow(unused_assignments)]
#[cube(launch_unchecked)]
fn ms2_pack_evidence_kernel<F: Float + CubeElement>(
    rank_in: &Array<u32>,
    evidence: &Array<u32>,
    traj_slot: &Array<u32>,
    log_prob: &Array<F>,
    packed_ev: &mut Array<u32>,
    packed_ev_f: &mut Array<F>,
    per_spectrum: u32,
    returned: u32,
    formulas: u32,
    n: u32,
    j: u32,
    lanes: usize,
    span: usize,
) {
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    for pos in start..end {
        let slot_pos = pos as u32;
        let mut b: u32 = 0u32;
        let mut slot_r: u32 = slot_pos;
        if returned > 0u32 {
            b = slot_pos / returned;
            slot_r = slot_pos % returned;
        }
        let base = b * per_spectrum;
        let mut found: u32 = 4294967295u32;
        let mut t: u32 = 0u32;
        while t < per_spectrum {
            if ms2_pack_slot(rank_in, 0u32, base + t) == slot_r {
                found = t;
            }
            t += 1u32;
        }
        let mut valid: u32 = 0u32;
        if found != 4294967295u32 {
            valid = 1u32;
        }
        let mut src: u32 = base;
        if valid == 1u32 {
            src = base + found;
        }
        // The 18 evidence words (unconditional loads, masked at use).
        let ev_base = src * 18u32;
        let out_base = slot_pos * 18u32;
        let mut w: u32 = 0u32;
        while w < 18u32 {
            let v = ms2_pack_slot(evidence, ev_base, w);
            let mut o: u32 = 0u32;
            if valid == 1u32 {
                o = v;
            }
            ms2_pack_put(packed_ev, out_base, w, o);
            w += 1u32;
        }
        let count_all = ms2_pack_slot(evidence, ev_base, 1u32);
        let mut count: u32 = 0u32;
        if valid == 1u32 {
            count = count_all;
        }
        let mut kept_c: u32 = count;
        if kept_c > 4u32 {
            kept_c = 4u32;
        }
        let slot_t = ms2_pack_slot(traj_slot, src * 2u32, 0u32);
        let mut slot_c: u32 = 0u32;
        if slot_t < formulas {
            slot_c = slot_t;
        }
        let width = j + 1u32;
        let mut q: u32 = 0u32;
        while q < 4u32 {
            let rw = ev_base + 2u32 + q * 4u32;
            let p = ms2_pack_slot(evidence, rw, 0u32);
            let bq = ms2_pack_slot(evidence, rw, 1u32);
            let apos = (b * formulas + slot_c) * n + p;
            let lp_idx = apos * width + bq;
            let mut lp: F = F::new(0.0_f32);
            if (lp_idx as usize) < log_prob.len() {
                lp = log_prob[lp_idx as usize];
            }
            let mut on: u32 = 0u32;
            if valid == 1u32 && q < kept_c {
                on = 1u32;
            }
            let mut v: F = F::new(0.0_f32);
            if on == 1u32 {
                v = lp;
            }
            let faddr = slot_pos * 4u32 + q;
            if (faddr as usize) < packed_ev_f.len() {
                packed_ev_f[faddr as usize] = v;
            }
            q += 1u32;
        }
    }
}

/// Packed evidence of one `(B, R)` output, lane per slot `(b, r)`.
///
/// `rank_in` is `[B*K]`, `evidence` is `[B*K, 18]`, `traj_slot` is `[B*K, 2]`
/// (formula slot, adduct), `log_prob` is `[B, F, N, J + 1]` floats,
/// `packed_ev` is `[B*R, 18]`, `packed_ev_f` is `[B*R, 4]`. `1 <= R <= K`:
/// `returned > per_spectrum` is [`Error::Config`]. Every stride, count and
/// largest accessed address is checked against the u32 domain before any
/// launch ([`Error::Shape`]).
#[allow(clippy::too_many_arguments)]
pub fn pack_evidence<R: Runtime, E: FloatElem>(
    rank_in: &IdTensor<R>,
    evidence: &IdTensor<R>,
    traj_slot: &IdTensor<R>,
    log_prob: &Tensor<R, E>,
    packed_ev: &mut IdTensor<R>,
    packed_ev_f: &mut Tensor<R, E>,
    formulas: usize,
    n: usize,
    j: usize,
    per_spectrum: usize,
    returned: usize,
) -> Result<()> {
    if rank_in.shape().rank() != 1
        || evidence.shape().rank() != 2
        || traj_slot.shape().rank() != 2
        || log_prob.shape().rank() != 4
        || packed_ev.shape().rank() != 2
        || packed_ev_f.shape().rank() != 2
    {
        return Err(Error::shape(format!(
            "pack_evidence needs rank [rows], evidence [rows, 18], traj_slot [rows, 2], log_prob [B, F, N, J + 1], packed_ev [slots, 18] and packed_ev_f [slots, 4], got {} and {} and {} and {} and {} and {}",
            rank_in.shape(),
            evidence.shape(),
            traj_slot.shape(),
            log_prob.shape(),
            packed_ev.shape(),
            packed_ev_f.shape()
        )));
    }
    let rows = rank_in.len();
    if per_spectrum == 0 || rows % per_spectrum != 0 {
        return Err(Error::shape(format!(
            "pack_evidence needs rows {rows} divisible by per_spectrum {per_spectrum}"
        )));
    }
    let batch = rows / per_spectrum;
    if returned == 0 {
        return Err(Error::shape(format!(
            "pack_evidence needs returned >= 1, got {returned}"
        )));
    }
    if returned > per_spectrum {
        return Err(Error::config(format!(
            "pack_evidence needs returned {returned} <= per_spectrum {per_spectrum} (1 <= R <= K)"
        )));
    }
    let want_ev: &[usize] = &[rows, EVIDENCE_STRIDE as usize];
    let want_slot: &[usize] = &[rows, 2];
    let slots = batch.checked_mul(returned).ok_or_else(|| {
        Error::shape(format!(
            "pack_evidence needs batch {batch} times returned {returned} to fit usize"
        ))
    })?;
    let want_packed: &[usize] = &[slots, EVIDENCE_STRIDE as usize];
    let want_packed_f: &[usize] = &[slots, 4];
    if evidence.shape().dims() != want_ev
        || traj_slot.shape().dims() != want_slot
        || packed_ev.shape().dims() != want_packed
        || packed_ev_f.shape().dims() != want_packed_f
    {
        return Err(Error::shape(format!(
            "pack_evidence needs evidence [{rows}, 18], traj_slot [{rows}, 2], packed_ev [{slots}, 18] and packed_ev_f [{slots}, 4], got {} and {} and {} and {}",
            evidence.shape(),
            traj_slot.shape(),
            packed_ev.shape(),
            packed_ev_f.shape()
        )));
    }
    if log_prob.shape().dims() != [batch, formulas, n, j + 1] {
        return Err(Error::shape(format!(
            "pack_evidence needs log_prob [{batch}, {formulas}, {n}, {}], got {}",
            j + 1,
            log_prob.shape()
        )));
    }
    check_u32_len("pack_evidence rows", rows)?;
    check_u32_len("pack_evidence batch", batch)?;
    check_u32_len("pack_evidence per_spectrum", per_spectrum)?;
    check_u32_len("pack_evidence returned", returned)?;
    check_u32_len("pack_evidence formulas", formulas)?;
    check_u32_len("pack_evidence n", n)?;
    check_u32_len("pack_evidence j", j)?;
    let lanes = check_u32_product("pack_evidence slots", batch, returned)?;
    let _ = check_u32_product("pack_evidence evidence addresses", rows, EVIDENCE_STRIDE as usize)?;
    let _ = check_u32_product("pack_evidence packed_ev addresses", lanes, EVIDENCE_STRIDE as usize)?;
    let _ = check_u32_product("pack_evidence packed_ev_f addresses", lanes, 4)?;
    if lanes == 0 {
        return Ok(());
    }
    let client = rank_in.client();
    let (count, dim, span) = launch_1d_spans(client, lanes, 18);
    unsafe {
        ms2_pack_evidence_kernel::launch_unchecked::<E, R>(
            client,
            count,
            dim,
            rank_in.arg(),
            evidence.arg(),
            traj_slot.arg(),
            log_prob.arg(),
            packed_ev.arg(),
            packed_ev_f.arg(),
            per_spectrum as u32,
            returned as u32,
            formulas as u32,
            n as u32,
            j as u32,
            lanes,
            span,
        );
    }
    Ok(())
}
