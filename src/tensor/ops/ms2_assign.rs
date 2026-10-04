//! GPU kernel for the fragment-ion assignment class mask (K6).
//!
//! Architecture `docs/MS2_V1_ARCHITECTURE.md` §2.2 (assignment distribution).
//! [`ion_class_mask`] turns `ion_meta [B, F, N, 4]` into the float class mask
//! `[B, F, N, J + 1]` the [`AssignmentHead`](crate::models::ms2::assign::AssignmentHead)
//! scores under: class `j < J` exists exactly when `j` names a kept
//! hypothesis (`j < min(kept, J)` with `kept = ion_meta[.., 2]`), class `J`
//! (unassigned) always exists. A peak with `ion_unavailable` (bit 2 of
//! `ion_meta[.., 3]`) keeps no class but unassigned: its row is the one-hot
//! of the unassigned class, so its masked log-softmax is log-probability 0 at
//! class `J` and the masked value elsewhere, and the loss (eligibility 0)
//! sends it no gradient. The lane's padding peaks carry the same bit, so they
//! need no separate input. Status bits 0 (exhausted) and 1 (capacity
//! exceeded) do not change the mask: the distribution is conditional on the
//! retained hypotheses and is reported with that status.
//!
//! The host twin is
//! [`ion_class_mask_host`](crate::models::ms2::assign::ion_class_mask_host),
//! written first in kernel-expressible form (full buffers with explicit
//! indices, `u32` arithmetic, no early `return`); the kernel below copies it.
//! Integer buffers are [`IdTensor`] (`u32`); the launch goes through
//! [`crate::backend::launch_1d_spans`] with one lane per output element;
//! shapes are checked to [`crate::error::Error::Shape`] before any launch.
//!
//! At most 6 arrays per kernel (2 bound here). Every output element is
//! written by exactly one lane; selection is by comparison, never by
//! multiplying with a mask. Loop-carried variables start from literals or
//! buffer loads, never a plain copy of a scalar argument.

use cubecl::prelude::*;

use crate::backend::{FloatElem, launch_1d_spans};
use crate::error::{Error, Result};
use crate::tensor::base::Tensor;
use crate::tensor::ops::index::IdTensor;
use crate::tensor::ops::ms2_identity::{check_device_len, check_device_scalar};
use crate::tensor::Shape;

/// Largest hypothesis capacity the mask supports (architecture §2.2:
/// `J <= 8`).
pub const ION_CLASS_MAX: usize = 8;

/// Lane per output element of [`ion_class_mask_into`]; a copy of
/// [`ion_class_mask_host`](crate::models::ms2::assign::ion_class_mask_host)
/// over `Array`s.
///
/// `row = lane / width`, `col = lane % width` with `width = j + 1`;
/// `kept` and `status` are unconditionally loaded from the row's `ion_meta`
/// words and the guards apply where the value is used.
#[cube(launch_unchecked)]
fn ms2_ion_class_mask_kernel<F: Float + CubeElement>(
    ion_meta: &Array<u32>,
    mask: &mut Array<F>,
    j: u32,
    lanes: usize,
    span: usize,
) {
    let one = F::new(1.0_f32);
    let zero = F::new(0.0_f32);
    let width = j + 1u32;
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    for pos in start..end {
        let lane = pos as u32;
        let row = lane / width;
        let col = lane - row * width;
        let kept = ion_meta[(row * 4u32 + 2u32) as usize];
        let status = ion_meta[(row * 4u32 + 3u32) as usize];
        let mut unavailable: u32 = 0u32;
        if status & 4u32 != 0u32 {
            unavailable = 1u32;
        }
        let mut eff = kept;
        if eff > j {
            eff = j;
        }
        if unavailable == 1u32 {
            eff = 0u32;
        }
        let mut v = zero;
        if col < eff {
            v = one;
        }
        if col == j {
            v = one;
        }
        mask[pos] = v;
    }
}

/// Write the assignment class mask into caller-provided `out`: the launch
/// path behind [`ion_class_mask`].
///
/// `ion_meta` is `[B, F, N, 4]` (accepted, ambiguous, kept, status with bit 2
/// `ion_unavailable`); `out` is `[B, F, N, J + 1]` floats (`J` from `out`'s
/// last dimension, `1 <= J <= 8`). Tests poison `out` first (NaN floats), so a
/// lane the kernel skips fails the every-element comparison against the host
/// twin. One launch.
pub fn ion_class_mask_into<R: Runtime, E: FloatElem>(
    ion_meta: &IdTensor<R>,
    out: &mut Tensor<R, E>,
) -> Result<()> {
    if ion_meta.shape().rank() != 4 || out.rank() != 4 {
        return Err(Error::shape(format!(
            "ion_class_mask needs ion_meta [B, F, N, 4] and out [B, F, N, J + 1], got {} and {}",
            ion_meta.shape(),
            out.shape()
        )));
    }
    let batch = ion_meta.shape().dim(0);
    let f = ion_meta.shape().dim(1);
    let n = ion_meta.shape().dim(2);
    if ion_meta.shape().dims() != [batch, f, n, 4] {
        return Err(Error::shape(format!(
            "ion_class_mask needs ion_meta [{batch}, {f}, {n}, 4], got {}",
            ion_meta.shape()
        )));
    }
    if out.shape().dim(3) < 2 {
        return Err(Error::shape(format!(
            "ion_class_mask needs out [B, F, N, J + 1] with J >= 1, got {}",
            out.shape()
        )));
    }
    let j = out.shape().dim(3) - 1;
    if j > ION_CLASS_MAX {
        return Err(Error::shape(format!(
            "ion_class_mask needs J <= {ION_CLASS_MAX} (architecture §2.2), got J = {j}"
        )));
    }
    let want_out: &[usize] = &[batch, f, n, j + 1];
    if out.shape().dims() != want_out {
        return Err(Error::shape(format!(
            "ion_class_mask needs out [{batch}, {f}, {n}, {}], got {}",
            j + 1,
            out.shape()
        )));
    }
    if out.is_empty() {
        return Ok(());
    }
    check_device_len("ion_meta", ion_meta.len())?;
    check_device_len("ion_class_mask out", out.len())?;
    let lanes = batch
        .checked_mul(f)
        .and_then(|v| v.checked_mul(n))
        .and_then(|v| v.checked_mul(j + 1))
        .ok_or_else(|| Error::shape("ion_class_mask: B * F * N * (J + 1) overflows usize".to_string()))?;
    if lanes != out.len() {
        return Err(Error::shape(format!(
            "ion_class_mask: out holds {} words but [B, F, N, J + 1] needs {lanes}",
            out.len()
        )));
    }
    // `lanes` fits `u32`, so every `pos < lanes` narrows to a `u32` lane
    // without wrapping.
    check_device_scalar("ion_class_mask B * F * N * (J + 1) lanes", lanes)?;
    let j_u32 = check_device_scalar("ion_class_mask J", j)?;
    let client = ion_meta.client();
    let (count, dim, span) = launch_1d_spans(client, lanes, j + 1);
    unsafe {
        ms2_ion_class_mask_kernel::launch_unchecked::<E, R>(
            client,
            count,
            dim,
            ion_meta.arg(),
            out.arg(),
            j_u32,
            lanes,
            span,
        );
    }
    Ok(())
}

/// The `[B, F, N, J + 1]` float class mask of `ion_meta` (`[B, F, N, 4]`)
/// with the hypothesis capacity `j` (`1 <= J <= 8`): 1 on the kept hypothesis
/// classes and on unassigned, 0 elsewhere; unassigned-only under
/// `ion_unavailable`. One launch.
pub fn ion_class_mask<R: Runtime, E: FloatElem>(
    ion_meta: &IdTensor<R>,
    j: usize,
) -> Result<Tensor<R, E>> {
    if ion_meta.shape().rank() != 4 || ion_meta.shape().dim(3) != 4 {
        return Err(Error::shape(format!(
            "ion_class_mask needs ion_meta [B, F, N, 4], got {}",
            ion_meta.shape()
        )));
    }
    if j == 0 || j > ION_CLASS_MAX {
        return Err(Error::shape(format!(
            "ion_class_mask needs 1 <= J <= {ION_CLASS_MAX} (architecture §2.2), got J = {j}"
        )));
    }
    let batch = ion_meta.shape().dim(0);
    let f = ion_meta.shape().dim(1);
    let n = ion_meta.shape().dim(2);
    let mut out = Tensor::empty(
        Shape::new(vec![batch, f, n, j + 1]),
        ion_meta.device(),
    );
    ion_class_mask_into(ion_meta, &mut out)?;
    Ok(out)
}
