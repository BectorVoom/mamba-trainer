//! Kernels of the graph-matching task ([`crate::models::graph::matching`]).
//!
//! Two gathers, in the style of [`crate::tensor::ops::graph`]: one unit owns an
//! output element and loops over its sources; nothing scatters and nothing
//! uses atomics.
//!
//! * [`match_bag`] writes the dense `[rows, n_bits]` bag of weighted bits of
//!   each row's query — the constant input of the query encoder's first,
//!   bias-free projection (an embedding bag computed as a product).
//! * [`match_layout`] writes one batch's legal mask `[groups, slots]` (each
//!   group's own member slots), the answer slot of every group and the mask
//!   of the real groups.
//!
//! Both read a **group table**: `3` ids per row — the query id (or
//! [`IGNORE`] for a padding row), and the first and one-past-last position,
//! in the same table, of the row's member slots, answer first. The launchers
//! are `pub(crate)`: their tables are only built by
//! [`crate::models::graph::matching`], which validates every id on the host
//! before the one upload, so the unchecked launches index in bounds.

use cubecl::prelude::*;

use crate::backend::{FloatElem, launch_1d_spans};
use crate::error::{Error, Result};
use crate::tensor::base::Tensor;
use crate::tensor::ops::IGNORE;
use crate::tensor::ops::index::IdTensor;

/// Ids per row of a group table: query, first member position, end.
pub(crate) const GROUP_STRIDE: usize = 3;

/// One unit per bag element `(row, bit)`: the summed weight of `bit` in the
/// row's query; zero for a padding row.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn match_bag_kernel<F: Float + CubeElement>(
    index: &Array<u32>,
    bits: &Array<u32>,
    weights: &Array<F>,
    counts: &Array<u32>,
    bag: &mut Array<F>,
    index_at: usize,
    stride: usize,
    k: usize,
    n_bits: usize,
    lanes: usize,
    span: usize,
    ignore: u32,
) {
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    for pos in start..end {
        let row = pos / n_bits;
        let bit = pos % n_bits;
        let query = index[index_at + row * stride];
        let mut acc = 0.0f32;
        if query != ignore {
            let base = query as usize * k;
            let count = counts[query as usize] as usize;
            for j in 0..count {
                if bits[base + j] as usize == bit {
                    acc += f32::cast_from(weights[base + j]);
                }
            }
        }
        bag[pos] = F::cast_from(acc);
    }
}

/// The dense bag `[rows, n_bits]` of the queries named by `index`: row `r`'s
/// query id is `index[index_at + r * stride]` ([`IGNORE`]: a zero row). The
/// query table is `[queries, k]` bits and weights, of which the first
/// `counts[q]` entries of query `q` are real. One launch.
///
/// The caller guarantees that every id in `index` is below the query count or
/// [`IGNORE`], every count at most `k`, and every real bit below `n_bits`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn match_bag<R: Runtime, E: FloatElem>(
    index: &IdTensor<R>,
    index_at: usize,
    stride: usize,
    rows: usize,
    bits: &IdTensor<R>,
    weights: &Tensor<R, E>,
    counts: &IdTensor<R>,
    n_bits: usize,
) -> Result<Tensor<R, E>> {
    let _op = crate::backend::tally_op_scope("match_bag");
    let queries = counts.len();
    if queries == 0 || bits.len() != weights.len() || !bits.len().is_multiple_of(queries) {
        return Err(Error::shape(format!(
            "match_bag: a query table of {} bits, {} weights and {queries} counts",
            bits.len(),
            weights.len()
        )));
    }
    if rows > 0 && index_at + (rows - 1) * stride >= index.len() {
        return Err(Error::shape(format!(
            "match_bag: {rows} rows at {index_at} with stride {stride} leave an index of {}",
            index.len()
        )));
    }
    let k = bits.len() / queries;
    let device = index.device();
    let bag = Tensor::<R, E>::empty(vec![rows, n_bits], device);
    let lanes = rows * n_bits;
    if lanes == 0 {
        return Ok(bag);
    }
    let client = device.client();
    let (cube_count, cube_dim, span) = launch_1d_spans(client, lanes, k.min(64));
    // SAFETY: the index range was checked above; the ids it holds, the
    // counts and the bits were validated by the matching module before their
    // upload (ids below `queries` or IGNORE, counts at most `k`, real bits
    // below `n_bits`), and every output position is below `lanes`.
    unsafe {
        match_bag_kernel::launch_unchecked::<E, R>(
            client,
            cube_count,
            cube_dim,
            index.arg(),
            bits.arg(),
            weights.arg(),
            counts.arg(),
            bag.arg(),
            index_at,
            stride,
            k,
            n_bits,
            lanes,
            span,
            IGNORE,
        );
    }
    Ok(bag)
}

/// One unit per `(group, slot)`: whether the slot is one of the group's
/// members; the unit of slot 0 also writes the group's target and mask.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn match_layout_kernel<F: Float + CubeElement>(
    table: &Array<u32>,
    legal: &mut Array<F>,
    targets: &mut Array<u32>,
    valid: &mut Array<F>,
    group_at: usize,
    slots: usize,
    lanes: usize,
    span: usize,
    ignore: u32,
) {
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    for pos in start..end {
        let row = pos / slots;
        let slot = pos % slots;
        let at = group_at + row * 3;
        let query = table[at];
        let first = table[at + 1] as usize;
        let last = table[at + 2] as usize;
        let padding = query == ignore;
        // A padding group keeps one legal slot, its safe target.
        let mut hit = select(padding && slot == 0usize, 1u32, 0u32);
        if !padding {
            for j in first..last {
                if table[j] as usize == slot {
                    hit = 1u32;
                }
            }
        }
        legal[pos] = F::cast_from(hit);
        if slot == 0usize {
            // `first` is 0 for a padding row: the read stays in the table.
            targets[row] = select(padding, 0u32, table[first]);
            valid[row] = F::cast_from(select(padding, 0u32, 1u32));
        }
    }
}

/// One batch's legal mask `[groups, slots]`, answer slots `[groups]` and
/// real-group mask `[groups]` from the group rows at `group_at` of `table`.
/// A padding group gets slot 0 as its only legal slot and target, and a zero
/// mask. One launch.
///
/// The caller guarantees that every member position lies in `table` and every
/// member slot is below `slots`.
pub(crate) fn match_layout<R: Runtime, E: FloatElem>(
    table: &IdTensor<R>,
    group_at: usize,
    groups: usize,
    slots: usize,
) -> Result<(Tensor<R, E>, IdTensor<R>, Tensor<R, E>)> {
    let _op = crate::backend::tally_op_scope("match_layout");
    if groups == 0 || slots == 0 || group_at + groups * GROUP_STRIDE > table.len() {
        return Err(Error::shape(format!(
            "match_layout: {groups} groups at {group_at} over {slots} slots in a table of {}",
            table.len()
        )));
    }
    let device = table.device();
    let legal = Tensor::<R, E>::empty(vec![groups, slots], device);
    let targets = IdTensor::empty(vec![groups], device);
    let valid = Tensor::<R, E>::empty(vec![groups], device);
    let lanes = groups * slots;
    let client = device.client();
    let (cube_count, cube_dim, span) = launch_1d_spans(client, lanes, 8);
    // SAFETY: the group rows lie in the table (checked above); the matching
    // module wrote member positions inside the table and member slots below
    // the batch's slot capacity, which is `slots`.
    unsafe {
        match_layout_kernel::launch_unchecked::<E, R>(
            client,
            cube_count,
            cube_dim,
            table.arg(),
            legal.arg(),
            targets.arg(),
            valid.arg(),
            group_at,
            slots,
            lanes,
            span,
            IGNORE,
        );
    }
    Ok((legal, targets, valid))
}
