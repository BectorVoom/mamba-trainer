//! GPU kernel for the reranker features (K5).
//!
//! Architecture `docs/MS2_V1_ARCHITECTURE.md` §4.3. The kernel is a
//! line-for-line copy of its host twin's lane
//! ([`feature_lane`](crate::models::ms2::rerank::feature_lane) in
//! `crate::models::ms2::rerank`, written first in the kernel-expressible
//! form), so both stay identical. Integer buffers are [`IdTensor`] (`u32`);
//! the launch goes through [`crate::backend::launch_1d_spans`] with one lane
//! per trajectory; shapes are checked to [`crate::error::Error::Shape`]
//! before any launch.
//!
//! The 6 bound arrays are `actions`, `scores`, `evidence`, `evidence_f`,
//! `features`, `feature_ok` (wgpu's limit). Every output element is written
//! by exactly one lane; selection is by comparison, never by multiplying with
//! a mask. Loop-carried variables start from literals or buffer loads, never
//! a plain copy of a scalar argument. Buffer loads on the counting loops are
//! unconditional (the length guard applies where the value is used), so no
//! load serialises on memory latency behind a branch.
//!
//! Layout words below name the constants of `crate::models::ms2::rerank`
//! they equal: `scores` holds `(trace_log_prob, formula_log_prob)`;
//! `evidence` holds the status at word 0 and the count at word 1 of an
//! 18-word row; `evidence_f` holds `(largest evidence log-probability,
//! smallest |residual| in tolerance units)`.
//!
//! What the integration task must provide per call: `actions`
//! `[B*K, T*4 + A + 4]` (the trajectory records generation already holds),
//! `scores` `[B*K, 2]` (trace log-probability from the `actions` record,
//! formula log-probability by indexing `top_log_prob [B, F]` with the
//! trajectory's formula slot — the same packing `ms2_rank` needs), `evidence`
//! `[B*K, 18]` (as `ms2_ion_evidence` writes it), `evidence_f` `[B*K, 2]`
//! (from the assignment head's log-probabilities divided by the per-peak
//! fragment tolerance; `(0, 1)` when the candidate has no evidence).
//!
//! CubeCL 0.10 has no `RangeInclusive::contains` or `usize::is_multiple_of`
//! in kernels, so lanes use explicit comparisons; the clippy lints for those
//! patterns are allowed here rather than rewritten.
//! The lane also guards every `u32` division with an explicit non-zero check,
//! which clippy reads as a manual `checked_div`: CubeCL has no `checked_div`
//! or `Option`, so the kernel form stays and the lint is allowed here rather
//! than rewritten.
#![allow(clippy::manual_range_contains, clippy::manual_is_multiple_of)]
#![allow(clippy::manual_checked_ops)]

use cubecl::prelude::*;

use crate::backend::{FloatElem, launch_1d_spans};
use crate::error::{Error, Result};
use crate::models::ms2::rerank::{EVIDENCE_STRIDE, N_FEATURES};
use crate::tensor::base::Tensor;
use crate::tensor::ops::index::IdTensor;

// ---------------------------------------------------------------------------
// Shared `#[cube]` helper (copy of the twin helper)
// ---------------------------------------------------------------------------

/// One guarded full-buffer read.
#[cube]
fn ms2_rerank_slot(buf: &Array<u32>, base: u32, idx: u32) -> u32 {
    let addr = base + idx;
    let mut out: u32 = 0u32;
    if (addr as usize) < buf.len() {
        out = buf[addr as usize];
    }
    out
}

// ---------------------------------------------------------------------------
// `ms2_rerank_features`: lane per trajectory
// ---------------------------------------------------------------------------

/// Lane per trajectory of [`features`]; a line-for-line copy of
/// `rerank::feature_lane` over `Array`s (the 8 feature scalars are written
/// out one by one on both sides).
#[allow(clippy::too_many_arguments)]
#[allow(unused_assignments)]
#[cube(launch_unchecked)]
fn ms2_rerank_features_kernel<F: Float + CubeElement>(
    actions: &Array<u32>,
    scores: &Array<F>,
    evidence: &Array<u32>,
    evidence_f: &Array<F>,
    features: &mut Array<F>,
    feature_ok: &mut Array<u32>,
    record_stride: u32,
    steps: u32,
    atoms_cap: u32,
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
        let len_field = steps * 4u32 + atoms_cap;
        let length = ms2_rerank_slot(actions, abase, len_field);
        let mut trace_v: F = F::new(0.0_f32);
        if ((record * 2u32) as usize) < scores.len() {
            trace_v = scores[(record * 2u32) as usize];
        }
        let mut formula_v: F = F::new(0.0_f32);
        if ((record * 2u32 + 1u32) as usize) < scores.len() {
            formula_v = scores[(record * 2u32 + 1u32) as usize];
        }
        // Finiteness is a range test: a NaN self-comparison is folded to
        // false by fast-math backends, so NaN would pass.
        let mut good: u32 = 0u32;
        if trace_v > F::new(-3.0e38_f32)
            && trace_v < F::new(3.0e38_f32)
            && formula_v > F::new(-3.0e38_f32)
            && formula_v < F::new(3.0e38_f32)
        {
            good = 1u32;
        }
        let mut f0: F = F::new(0.0_f32);
        let mut f1: F = F::new(0.0_f32);
        let mut f2: F = F::new(0.0_f32);
        let mut f3: F = F::new(0.0_f32);
        let mut f4: F = F::new(0.0_f32);
        let mut f5: F = F::new(0.0_f32);
        let mut f6: F = F::new(0.0_f32);
        let mut f7: F = F::new(0.0_f32);
        if good == 1u32 {
            let mut atoms: u32 = 0u32;
            let mut s: u32 = 0u32;
            while s < steps {
                let kind = ms2_rerank_slot(actions, abase, s * 4u32);
                if s < length && kind == 2u32 {
                    atoms += 1u32;
                }
                s += 1u32;
            }
            let mut open: u32 = 0u32;
            let mut a: u32 = 0u32;
            while a < atoms_cap {
                open += ms2_rerank_slot(actions, abase, steps * 4u32 + a);
                a += 1u32;
            }
            let ev_status = ms2_rerank_slot(evidence, record * 18u32, 0u32);
            let ev_count = ms2_rerank_slot(evidence, record * 18u32, 1u32);
            let mut max_lp: F = F::new(0.0_f32);
            if ((record * 2u32) as usize) < evidence_f.len() {
                max_lp = evidence_f[(record * 2u32) as usize];
            }
            let mut min_resid: F = F::new(0.0_f32);
            if ((record * 2u32 + 1u32) as usize) < evidence_f.len() {
                min_resid = evidence_f[(record * 2u32 + 1u32) as usize];
            }
            if ev_count == 0u32 {
                max_lp = F::new(0.0_f32);
                min_resid = F::new(1.0_f32);
            } else {
                let mut max_ok: u32 = 0u32;
                if max_lp > F::new(-3.0e38_f32) && max_lp < F::new(3.0e38_f32) {
                    max_ok = 1u32;
                }
                if max_ok == 0u32 {
                    max_lp = F::new(0.0_f32);
                }
                let mut resid_ok: u32 = 0u32;
                if min_resid > F::new(-3.0e38_f32) && min_resid < F::new(3.0e38_f32) {
                    resid_ok = 1u32;
                }
                if resid_ok == 0u32 {
                    min_resid = F::new(1.0_f32);
                }
            }
            let mut incomplete: F = F::new(0.0_f32);
            if ev_count != 0u32 && ev_status & 128u32 != 0u32 {
                incomplete = F::new(1.0_f32);
            }
            f0 = trace_v;
            f1 = formula_v;
            if atoms_cap > 0u32 {
                f2 = F::cast_from(atoms) / F::cast_from(atoms_cap);
                f3 = F::cast_from(open) / (F::cast_from(atoms_cap) + F::cast_from(atoms_cap));
            }
            // Retained evidence count: the buffer's count word is the total
            // number of qualifying peaks (unbounded); §2.4 retains at most
            // E = 4, so clamp before dividing.
            let mut retained = ev_count;
            if retained > 4u32 {
                retained = 4u32;
            }
            f4 = F::cast_from(retained) / F::new(4.0_f32);
            f5 = max_lp;
            f6 = min_resid;
            f7 = incomplete;
        }
        let out_base = record * 8u32;
        if (out_base as usize) < features.len() {
            features[out_base as usize] = f0;
        }
        if ((out_base + 1u32) as usize) < features.len() {
            features[(out_base + 1u32) as usize] = f1;
        }
        if ((out_base + 2u32) as usize) < features.len() {
            features[(out_base + 2u32) as usize] = f2;
        }
        if ((out_base + 3u32) as usize) < features.len() {
            features[(out_base + 3u32) as usize] = f3;
        }
        if ((out_base + 4u32) as usize) < features.len() {
            features[(out_base + 4u32) as usize] = f4;
        }
        if ((out_base + 5u32) as usize) < features.len() {
            features[(out_base + 5u32) as usize] = f5;
        }
        if ((out_base + 6u32) as usize) < features.len() {
            features[(out_base + 6u32) as usize] = f6;
        }
        if ((out_base + 7u32) as usize) < features.len() {
            features[(out_base + 7u32) as usize] = f7;
        }
        if (record as usize) < feature_ok.len() {
            feature_ok[record as usize] = good;
        }
    }
}

/// Reranker features of one `(B, K)` bucket, lane per trajectory.
///
/// `actions` is `[rows, T*4 + A + 4]`, `scores` is `[rows, 2]` floats
/// `(trace_log_prob, formula_log_prob)`, `evidence` is `[rows, 18]`,
/// `evidence_f` is `[rows, 2]` floats, `features_out` is `[rows, 8]`,
/// `feature_ok` is `[rows]` (`1` finite, `0` the score was non-finite and the
/// row is all zeros).
#[allow(clippy::too_many_arguments)]
pub fn features<R: Runtime, E: FloatElem>(
    actions: &IdTensor<R>,
    scores: &Tensor<R, E>,
    evidence: &IdTensor<R>,
    evidence_f: &Tensor<R, E>,
    features_out: &mut Tensor<R, E>,
    feature_ok: &mut IdTensor<R>,
    steps: usize,
    atoms_cap: u32,
) -> Result<()> {
    if actions.shape().rank() != 2
        || scores.shape().rank() != 2
        || evidence.shape().rank() != 2
        || evidence_f.shape().rank() != 2
        || features_out.shape().rank() != 2
        || feature_ok.shape().rank() != 1
    {
        return Err(Error::shape(format!(
            "rerank features needs actions [rows, stride], scores [rows, 2], evidence [rows, 18], evidence_f [rows, 2], features [rows, 8] and feature_ok [rows], got {} and {} and {} and {} and {} and {}",
            actions.shape(),
            scores.shape(),
            evidence.shape(),
            evidence_f.shape(),
            features_out.shape(),
            feature_ok.shape()
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
            "rerank features needs actions stride T*4 + A + 4 = {want_stride:?} for T {steps} A {atoms_cap}, got {record_stride}"
        )));
    }
    if atoms_cap == 0 || atoms_cap > 32 {
        return Err(Error::shape(format!(
            "rerank features needs 1 <= atoms_cap <= 32, got {atoms_cap}"
        )));
    }
    if steps == 0 {
        return Err(Error::shape(
            "rerank features needs steps >= 1".to_string(),
        ));
    }
    let want_scores: &[usize] = &[rows, 2];
    let want_evidence: &[usize] = &[rows, EVIDENCE_STRIDE];
    let want_evidence_f: &[usize] = &[rows, 2];
    let want_features: &[usize] = &[rows, N_FEATURES];
    if scores.shape().dims() != want_scores
        || evidence.shape().dims() != want_evidence
        || evidence_f.shape().dims() != want_evidence_f
        || features_out.shape().dims() != want_features
        || feature_ok.len() != rows
    {
        return Err(Error::shape(format!(
            "rerank features needs scores [{rows}, 2], evidence [{rows}, {EVIDENCE_STRIDE}], evidence_f [{rows}, 2], features [{rows}, {N_FEATURES}] and feature_ok [{rows}], got {} and {} and {} and {} and {}",
            scores.shape(),
            evidence.shape(),
            evidence_f.shape(),
            features_out.shape(),
            feature_ok.shape()
        )));
    }
    if rows == 0 {
        return Ok(());
    }
    let client = actions.client();
    let (count, dim, span) = launch_1d_spans(client, rows, record_stride);
    unsafe {
        ms2_rerank_features_kernel::launch_unchecked::<E, R>(
            client,
            count,
            dim,
            actions.arg(),
            scores.arg(),
            evidence.arg(),
            evidence_f.arg(),
            features_out.arg(),
            feature_ok.arg(),
            record_stride as u32,
            steps as u32,
            atoms_cap,
            rows,
            span,
        );
    }
    Ok(())
}
