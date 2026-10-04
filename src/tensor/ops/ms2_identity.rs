//! GPU kernels for graph identity and trajectory allocation (K1).
//!
//! Architecture `docs/MS2_V1_ARCHITECTURE.md` §4.2 (graph identity) and §3.2
//! (allocation). Each kernel is a line-for-line copy of its host twin's lane
//! function(s) in `crate::models::ms2::identity` and
//! `crate::models::ms2::allocate`; the twins were written first in the
//! kernel-expressible form (no Rust fixed-size local arrays, no `wrapping_*`
//! methods, full buffers with explicit indices/strides/base offsets), so both
//! stay identical. Integer buffers are [`IdTensor`] (`u32`); every launch goes
//! through [`crate::backend::launch_1d_spans`] with one lane per output item;
//! shapes are checked to [`crate::error::Error::Shape`] before any launch.
//!
//! At most 6 arrays per kernel. Every output element is written by exactly one
//! lane; padding gets an explicit value; selection is by comparison, never by
//! multiplying with a mask. Loop-carried variables start from literals or
//! buffer loads, never a plain copy of a scalar argument. The hash reuses the
//! sampler's `#[cube]` [`hash_u32`]; no new hash is introduced.

//! CubeCL 0.10 has no `RangeInclusive::contains` or `usize::is_multiple_of`
//! in kernels, so lanes use explicit comparisons; the clippy lints for those
//! patterns are allowed here rather than rewritten.
#![allow(clippy::manual_range_contains, clippy::manual_is_multiple_of)]

use cubecl::prelude::*;

use crate::backend::{DType, FloatElem, launch_1d_spans};
use crate::error::{Error, Result};
use crate::tensor::base::Tensor;
use crate::tensor::ops::index::IdTensor;
use crate::tensor::ops::random::hash_u32;
use crate::tensor::shape::Shape;

// ---------------------------------------------------------------------------
// Device-address checks (review finding 3)
// ---------------------------------------------------------------------------

/// Reject a bound buffer whose word count leaves the `u32` device address
/// range: every kernel below addresses its bindings in `u32`, so a buffer
/// with more than `u32::MAX` words would wrap its tail lanes onto earlier
/// ones, breaking single-writer ownership. Shape checks already tie each
/// buffer's length to its lane grid (`rows * stride`), so a length inside
/// `u32` proves every lane address the kernel forms for it fits `u32`
/// without wrapping.
pub fn check_device_len(name: &str, len: usize) -> Result<()> {
    if len > u32::MAX as usize {
        return Err(Error::shape(format!(
            "{name} holds {len} words, beyond the u32 device address range"
        )));
    }
    Ok(())
}

/// Narrow a host `usize` shape to a `u32` kernel scalar, rejecting values
/// that leave the `u32` range instead of wrapping them with `as u32`.
pub fn check_device_scalar(name: &str, value: usize) -> Result<u32> {
    u32::try_from(value).map_err(|_| {
        Error::shape(format!(
            "{name} is {value}, beyond the u32 device scalar range"
        ))
    })
}

// ---------------------------------------------------------------------------
// Shared `#[cube]` helpers (copies of the twin helpers)
// ---------------------------------------------------------------------------

/// Two-argument hash through the sampler's mixer with a zero third word.
#[cube]
fn ms2_id_hash_pair(first: u32, second: u32) -> u32 {
    hash_u32(first, second, 0u32)
}

/// One guarded full-buffer read.
#[cube]
fn ms2_id_slot(buf: &Array<u32>, base: u32, idx: u32) -> u32 {
    let addr = base + idx;
    let mut out: u32 = 0u32;
    if (addr as usize) < buf.len() {
        out = buf[addr as usize];
    }
    out
}

/// One guarded full-buffer write.
#[cube]
fn ms2_id_put(buf: &mut Array<u32>, base: u32, idx: u32, value: u32) {
    let addr = base + idx;
    if (addr as usize) < buf.len() {
        buf[addr as usize] = value;
    }
}

/// One guarded token-field read from a device record.
#[cube]
fn ms2_id_token_word(actions: &Array<u32>, abase: u32, field: u32, or: u32) -> u32 {
    let addr = abase + field;
    let mut out: u32 = or;
    if (addr as usize) < actions.len() {
        out = actions[addr as usize];
    }
    out
}

/// Bond order between `x` and `y` in the bond list at `sbase`, else 0.
#[cube]
fn ms2_id_bond_between(scratch: &Array<u32>, sbase: u32, n_bonds: u32, x: u32, y: u32) -> u32 {
    let mut order: u32 = 0u32;
    let mut bi: u32 = 0u32;
    while bi < n_bonds {
        let triple = bi * 3;
        let a = ms2_id_slot(scratch, sbase, triple);
        let b = ms2_id_slot(scratch, sbase, triple + 1);
        let o = ms2_id_slot(scratch, sbase, triple + 2);
        let hits = (a == x && b == y) || (a == y && b == x);
        if hits {
            order = o;
        }
        bi += 1;
    }
    order
}

/// Atom type of the `target`-th atom in a device record, or 0.
#[cube]
fn ms2_id_atom_type_at(
    actions: &Array<u32>,
    abase: u32,
    length: u32,
    atoms_cap: u32,
    target: u32,
) -> u32 {
    let mut cap_a: u32 = atoms_cap;
    if cap_a > 32u32 {
        cap_a = 32u32;
    }
    let mut out: u32 = 0u32;
    let mut n: u32 = 0u32;
    let mut s: u32 = 0u32;
    while s < length {
        let tok = s * 4;
        let kind = ms2_id_token_word(actions, abase, tok, 4294967295u32);
        let ty = ms2_id_token_word(actions, abase, tok + 1, 0u32);
        if kind == 2u32 && n < cap_a {
            if n == target {
                out = ty;
            }
            n += 1;
        }
        s += 1;
    }
    out
}

/// Whether candidate image `cand` fits depth `depth` (copy of the twin, with
/// types re-decoded on demand and no local arrays).
#[allow(clippy::too_many_arguments)]
#[cube]
fn ms2_id_candidate_fits(
    cand: u32,
    depth: u32,
    actions: &Array<u32>,
    abase_k: u32,
    abase_j: u32,
    length_k: u32,
    length_j: u32,
    atoms_cap: u32,
    scratch: &Array<u32>,
    sbase_k: u32,
    sbase_j: u32,
    bank: u32,
    n_bonds: u32,
    stack: &Array<u32>,
    stack_base: u32,
) -> u32 {
    let mut fits: u32 = 1u32;
    if ms2_id_atom_type_at(actions, abase_k, length_k, atoms_cap, depth)
        != ms2_id_atom_type_at(actions, abase_j, length_j, atoms_cap, cand)
    {
        fits = 0;
    }
    if ms2_id_slot(scratch, sbase_k, bank + depth)
        != ms2_id_slot(scratch, sbase_j, bank + cand)
    {
        fits = 0;
    }
    let mut d: u32 = 0u32;
    while d < depth && fits == 1 {
        let earlier = ms2_id_slot(stack, stack_base, d);
        let order_k = ms2_id_bond_between(scratch, sbase_k, n_bonds, depth, d);
        let order_j = ms2_id_bond_between(scratch, sbase_j, n_bonds, cand, earlier);
        if order_k != order_j {
            fits = 0;
        }
        d += 1;
    }
    fits
}

// ---------------------------------------------------------------------------
// `ms2_graph_hash`: lane per trajectory
// ---------------------------------------------------------------------------

/// Lane per trajectory of [`graph_hash`]; a line-for-line copy of
/// `identity::graph_hash_lane` over `Array`s.
#[allow(clippy::too_many_arguments)]
#[allow(unused_assignments)]
#[cube(launch_unchecked)]
fn ms2_graph_hash_kernel(
    actions: &Array<u32>,
    graph_hash: &mut Array<u32>,
    scratch: &mut Array<u32>,
    record_stride: u32,
    scratch_stride: u32,
    steps: u32,
    atoms_cap: u32,
    closures_cap: u32,
    hash_mask: u32,
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
        let add_kind = 2u32;
        let close_kind = 3u32;
        let mut cap_a: u32 = atoms_cap;
        if cap_a == 0 {
            cap_a = 1;
        }
        if cap_a > 32u32 {
            cap_a = 32u32;
        }
        let mut bonds_max: u32 = cap_a - 1 + closures_cap;
        if cap_a == 0 {
            bonds_max = closures_cap;
        }
        if bonds_max > 39u32 {
            bonds_max = 39u32;
        }
        let bank_words = bonds_max * 3;
        let bank1 = bank_words + cap_a;
        let abase = record * record_stride;
        let sbase = record * scratch_stride;
        let len_field = steps * 4 + cap_a;
        let length = ms2_id_token_word(actions, abase, len_field, 0u32);
        let mut zi: u32 = 0u32;
        while zi < scratch_stride {
            ms2_id_put(scratch, sbase, zi, 0u32);
            zi += 1;
        }
        let mut n_atoms: u32 = 0u32;
        let mut n_bonds: u32 = 0u32;
        let mut s: u32 = 0u32;
        while s < length {
            let tok = s * 4;
            let kind = ms2_id_token_word(actions, abase, tok, 4294967295u32);
            let order = ms2_id_token_word(actions, abase, tok + 2, 0u32);
            let ptr = ms2_id_token_word(actions, abase, tok + 3, 0u32);
            if kind == add_kind {
                if n_atoms < cap_a {
                    if n_atoms > 0 && ptr < n_atoms && order >= 1u32 && order <= 3u32 && n_bonds < bonds_max
                    {
                        let triple = n_bonds * 3;
                        ms2_id_put(scratch, sbase, triple, n_atoms);
                        ms2_id_put(scratch, sbase, triple + 1, ptr);
                        ms2_id_put(scratch, sbase, triple + 2, order);
                        n_bonds += 1;
                    }
                    n_atoms += 1;
                }
            } else if kind == close_kind && n_atoms > 0 {
                let newest = n_atoms - 1;
                if ptr < newest && order >= 1u32 && order <= 3u32 && n_bonds < bonds_max {
                    let triple = n_bonds * 3;
                    ms2_id_put(scratch, sbase, triple, newest);
                    ms2_id_put(scratch, sbase, triple + 1, ptr);
                    ms2_id_put(scratch, sbase, triple + 2, order);
                    n_bonds += 1;
                }
            }
            s += 1;
        }
        let mut i: u32 = 0u32;
        while i < cap_a {
            let mut label: u32 = 0u32;
            if i < n_atoms {
                let mut degree: u32 = 0u32;
                let mut bi: u32 = 0u32;
                while bi < n_bonds {
                    let triple = bi * 3;
                    let a = ms2_id_slot(scratch, sbase, triple);
                    let b = ms2_id_slot(scratch, sbase, triple + 1);
                    if a == i || b == i {
                        degree += 1;
                    }
                    bi += 1;
                }
                let ty = ms2_id_atom_type_at(actions, abase, length, atoms_cap, i);
                label = ms2_id_hash_pair(ty, degree);
            }
            ms2_id_put(scratch, sbase, bank_words + i, label);
            i += 1;
        }
        let mut round: u32 = 0u32;
        while round < 4 {
            let mut read: u32 = bank_words;
            let mut write: u32 = bank1;
            if round % 2 == 1 {
                read = bank1;
                write = bank_words;
            }
            let mut a: u32 = 0u32;
            while a < cap_a {
                let mut next: u32 = 0u32;
                if a < n_atoms {
                    let old = ms2_id_slot(scratch, sbase, read + a);
                    let mut acc: u32 = 0u32;
                    let mut bi: u32 = 0u32;
                    while bi < n_bonds {
                        let triple = bi * 3;
                        let x = ms2_id_slot(scratch, sbase, triple);
                        let y = ms2_id_slot(scratch, sbase, triple + 1);
                        let o = ms2_id_slot(scratch, sbase, triple + 2);
                        if x == a && y < n_atoms {
                            acc += ms2_id_hash_pair(o, ms2_id_slot(scratch, sbase, read + y));
                        } else if y == a && x < n_atoms {
                            acc += ms2_id_hash_pair(o, ms2_id_slot(scratch, sbase, read + x));
                        }
                        bi += 1;
                    }
                    next = ms2_id_hash_pair(old, acc);
                }
                ms2_id_put(scratch, sbase, write + a, next);
                a += 1;
            }
            round += 1;
        }
        let mut sum: u32 = 0u32;
        let mut f: u32 = 0u32;
        while f < n_atoms {
            sum += ms2_id_slot(scratch, sbase, bank_words + f);
            f += 1;
        }
        let hash = hash_u32(sum, n_atoms, n_bonds) & hash_mask;
        if (record as usize) < graph_hash.len() {
            graph_hash[record as usize] = hash;
        }
    }
}

/// Graph hashes of one `(B, K)` bucket, lane per trajectory.
///
/// Producer preconditions (trusted, not re-read): every record's `length`
/// fits `steps` (token reads past the row are guarded and decode as skips).
/// Every bound buffer's word count and every narrowed scalar fit `u32`
/// (checked below with [`check_device_len`]/[`check_device_scalar`]);
/// larger layouts would wrap device addresses and are refused with
/// [`Error::Shape`] before launch.
#[allow(clippy::too_many_arguments)]
pub fn graph_hash<R: Runtime>(
    actions: &IdTensor<R>,
    graph_hash: &mut IdTensor<R>,
    scratch: &mut IdTensor<R>,
    steps: usize,
    atoms_cap: u32,
    closures_cap: u32,
    hash_mask: u32,
) -> Result<()> {
    if actions.shape().rank() != 2 || graph_hash.shape().rank() != 1 || scratch.shape().rank() != 2 {
        return Err(Error::shape(format!(
            "graph_hash needs actions [rows, stride], graph_hash [rows] and scratch [rows, G], got {} and {} and {}",
            actions.shape(),
            graph_hash.shape(),
            scratch.shape()
        )));
    }
    let rows = actions.shape().dim(0);
    let record_stride = actions.shape().dim(1);
    let scratch_stride = if rows == 0 { 0 } else { scratch.shape().dim(1) };
    let want_actions: &[usize] = &[rows, record_stride];
    let want_hash: &[usize] = &[rows];
    let want_scratch: &[usize] = &[rows, scratch_stride];
    if actions.shape().dims() != want_actions
        || graph_hash.shape().dims() != want_hash
        || scratch.shape().dims() != want_scratch
    {
        return Err(Error::shape(format!(
            "graph_hash needs actions [{rows}, {record_stride}], graph_hash [{rows}] and scratch [{rows}, {scratch_stride}], got {} and {} and {}",
            actions.shape(),
            graph_hash.shape(),
            scratch.shape()
        )));
    }
    let mut cap_a: u32 = atoms_cap;
    if cap_a == 0 {
        cap_a = 1;
    }
    if cap_a > 32 {
        return Err(Error::shape(format!(
            "graph_hash needs atoms_cap 1..=32, got {atoms_cap}"
        )));
    }
    let bonds = cap_a - 1 + closures_cap;
    if bonds > 39 {
        return Err(Error::shape(format!(
            "graph_hash needs A - 1 + R <= 39, got {bonds}"
        )));
    }
    let expect_stride = (bonds as usize) * 3 + (cap_a as usize) * 2;
    if scratch_stride != expect_stride {
        return Err(Error::shape(format!(
            "graph_hash needs scratch stride {expect_stride} for A {cap_a} R {closures_cap}, got {scratch_stride}"
        )));
    }
    let expect_record = steps * 4 + cap_a as usize + 4;
    if record_stride != expect_record {
        return Err(Error::shape(format!(
            "graph_hash needs record stride {expect_record} for T {steps} A {cap_a}, got {record_stride}"
        )));
    }
    check_device_len("actions", actions.len())?;
    check_device_len("graph_hash", graph_hash.len())?;
    check_device_len("scratch", scratch.len())?;
    let record_stride_u32 = check_device_scalar("graph_hash record stride", record_stride)?;
    let scratch_stride_u32 = check_device_scalar("graph_hash scratch stride", scratch_stride)?;
    let steps_u32 = check_device_scalar("graph_hash steps", steps)?;
    if rows == 0 {
        return Ok(());
    }
    let client = actions.client();
    let (count, dim, span) = launch_1d_spans(client, rows, expect_record);
    unsafe {
        ms2_graph_hash_kernel::launch_unchecked::<R>(
            client,
            count,
            dim,
            actions.arg(),
            graph_hash.arg(),
            scratch.arg(),
            record_stride_u32,
            scratch_stride_u32,
            steps_u32,
            atoms_cap,
            closures_cap,
            hash_mask,
            rows,
            span,
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// `ms2_graph_identity`: lane per trajectory `k`
// ---------------------------------------------------------------------------

/// Lane per trajectory `k` of [`graph_identity`]; a line-for-line copy of
/// `identity::identity_lane` (pair work inlined through the shared helpers).
#[allow(clippy::too_many_arguments)]
#[allow(unused_assignments)]
#[cube(launch_unchecked)]
fn ms2_graph_identity_kernel(
    actions: &Array<u32>,
    graph_hash: &Array<u32>,
    scratch: &Array<u32>,
    identity: &mut Array<u32>,
    stack: &mut Array<u32>,
    record_stride: u32,
    scratch_stride: u32,
    stack_stride: u32,
    steps: u32,
    atoms_cap: u32,
    bonds_cap: u32,
    per_spectrum: u32,
    work_max: u32,
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
        let stack_base = record * stack_stride;
        let mut bits: u32 = 0u32;
        let mut decided: u32 = 1u32;
        let mut eligible_k: u32 = 0u32;
        let len_field = steps * 4 + atoms_cap;
        let abase_k = record * record_stride;
        let status_k = ms2_id_token_word(actions, abase_k, len_field + 1, 0u32);
        if status_k & 1u32 != 0u32 && status_k & 8u32 == 0u32 {
            eligible_k = 1;
        }
        if eligible_k == 1 && per_spectrum > 0u32 {
            let b = record / per_spectrum;
            let k = record % per_spectrum;
            let base = b * per_spectrum;
            let mut j: u32 = 0u32;
            while j < k {
                let prev = base + j;
                let abase_j = prev * record_stride;
                let status_j = ms2_id_token_word(actions, abase_j, len_field + 1, 0u32);
                let mut eligible_j: u32 = 0u32;
                if status_j & 1u32 != 0u32 && status_j & 8u32 == 0u32 {
                    eligible_j = 1;
                }
                if eligible_j == 1 {
                    let hash_k = ms2_id_slot(graph_hash, 0u32, record);
                    let hash_j = ms2_id_slot(graph_hash, 0u32, prev);
                    if hash_k == hash_j {
                        let mut layout_ok: u32 = 1u32;
                        // All layout validation BEFORE any address
                        // multiplication, by guarded comparisons only
                        // (division-based guards, no `usize` addition): an
                        // invalid layout returns unresolved (2), never a
                        // verdict, and never panics. Kernel and twin share
                        // this block line for line.
                        if atoms_cap == 0u32 || atoms_cap > 32u32 {
                            layout_ok = 0u32;
                        }
                        if bonds_cap > 39u32 {
                            layout_ok = 0u32;
                        }
                        if stack_stride / 3u32 != atoms_cap || stack_stride % 3u32 != 0u32 {
                            layout_ok = 0u32;
                        }
                        let mut expect_row: u32 = 0u32;
                        if layout_ok == 1u32 {
                            expect_row = bonds_cap * 3u32 + atoms_cap * 2u32;
                        }
                        if scratch_stride != expect_row {
                            layout_ok = 0u32;
                        }
                        if record_stride < 4u32 {
                            layout_ok = 0u32;
                        } else if record_stride - 4u32 < atoms_cap {
                            layout_ok = 0u32;
                        } else if steps > (record_stride - 4u32 - atoms_cap) / 4u32 {
                            layout_ok = 0u32;
                        }
                        if record_stride == 0u32 {
                            layout_ok = 0u32;
                        } else if (record as usize) >= actions.len() / (record_stride as usize) {
                            layout_ok = 0u32;
                        } else if (prev as usize) >= actions.len() / (record_stride as usize) {
                            layout_ok = 0u32;
                        }
                        if (record as usize) >= graph_hash.len() {
                            layout_ok = 0u32;
                        }
                        if (prev as usize) >= graph_hash.len() {
                            layout_ok = 0u32;
                        }
                        if scratch_stride == 0u32 {
                            layout_ok = 0u32;
                        } else if (record as usize) >= scratch.len() / (scratch_stride as usize) {
                            layout_ok = 0u32;
                        } else if (prev as usize) >= scratch.len() / (scratch_stride as usize) {
                            layout_ok = 0u32;
                        }
                        if (stack_base as usize) > stack.len() {
                            layout_ok = 0u32;
                        } else if (stack_stride as usize) > stack.len() - (stack_base as usize) {
                            layout_ok = 0u32;
                        }
                        let mut abase_kk: u32 = 0u32;
                        let mut abase_jj: u32 = 0u32;
                        let mut sbase_k: u32 = 0u32;
                        let mut sbase_j: u32 = 0u32;
                        let mut len_field: u32 = 0u32;
                        if layout_ok == 1u32 {
                            abase_kk = record * record_stride;
                            abase_jj = prev * record_stride;
                            sbase_k = record * scratch_stride;
                            sbase_j = prev * scratch_stride;
                            len_field = steps * 4u32 + atoms_cap;
                        }
                        let length_k = ms2_id_token_word(actions, abase_kk, len_field, 0u32);
                        let length_j = ms2_id_token_word(actions, abase_jj, len_field, 0u32);
                        let mut cap_a: u32 = atoms_cap;
                        if cap_a > 32u32 {
                            cap_a = 32u32;
                        }
                        // Decode counts (no arrays).
                        let mut atoms_k: u32 = 0u32;
                        let mut bonds_k: u32 = 0u32;
                        let mut s: u32 = 0u32;
                        while s < length_k {
                            let tok = s * 4;
                            let kind = ms2_id_token_word(actions, abase_kk, tok, 4294967295u32);
                            let order = ms2_id_token_word(actions, abase_kk, tok + 2, 0u32);
                            let ptr = ms2_id_token_word(actions, abase_kk, tok + 3, 0u32);
                            if kind == 2u32 {
                                if atoms_k < cap_a {
                                    if atoms_k > 0 && ptr < atoms_k && order >= 1u32 && order <= 3u32 && bonds_k < bonds_cap
                                    {
                                        bonds_k += 1;
                                    }
                                    atoms_k += 1;
                                }
                            } else if kind == 3u32 && atoms_k > 0 {
                                let newest = atoms_k - 1;
                                if ptr < newest && order >= 1u32 && order <= 3u32 && bonds_k < bonds_cap
                                {
                                    bonds_k += 1;
                                }
                            }
                            s += 1;
                        }
                        let mut atoms_j: u32 = 0u32;
                        let mut bonds_j: u32 = 0u32;
                        let mut t: u32 = 0u32;
                        while t < length_j {
                            let tok = t * 4;
                            let kind = ms2_id_token_word(actions, abase_jj, tok, 4294967295u32);
                            let order = ms2_id_token_word(actions, abase_jj, tok + 2, 0u32);
                            let ptr = ms2_id_token_word(actions, abase_jj, tok + 3, 0u32);
                            if kind == 2u32 {
                                if atoms_j < cap_a {
                                    if atoms_j > 0 && ptr < atoms_j && order >= 1u32 && order <= 3u32 && bonds_j < bonds_cap
                                    {
                                        bonds_j += 1;
                                    }
                                    atoms_j += 1;
                                }
                            } else if kind == 3u32 && atoms_j > 0 {
                                let newest = atoms_j - 1;
                                if ptr < newest && order >= 1u32 && order <= 3u32 && bonds_j < bonds_cap
                                {
                                    bonds_j += 1;
                                }
                            }
                            t += 1;
                        }
                        let mut counts_ok: u32 = 1u32;
                        if atoms_k != atoms_j || bonds_k != bonds_j {
                            counts_ok = 0;
                        }
                        let mut hash_ok: u32 = 1u32;
                        if hash_k != hash_j {
                            hash_ok = 0;
                        }
                        let mut verdict: u32 = 0u32;
                        if layout_ok == 0 {
                            verdict = 2;
                        } else if counts_ok == 0 || hash_ok == 0 {
                            verdict = 0;
                        } else if atoms_k == 0 {
                            verdict = 1;
                        } else {
                            let n = atoms_k;
                            let nb = bonds_k;
                            let bank = bonds_cap * 3;
                            let two_a = atoms_cap * 2;
                            let mut w: u32 = 0u32;
                            while w < stack_stride {
                                ms2_id_put(stack, stack_base, w, 0u32);
                                w += 1;
                            }
                            let mut dd: u32 = 0u32;
                            while dd < atoms_cap {
                                ms2_id_put(stack, stack_base, dd, 4294967295u32);
                                dd += 1;
                            }
                            let mut depth: u32 = 0u32;
                            let mut used: u32 = 0u32;
                            let mut done: u32 = 0u32;
                            let mut found: u32 = 0u32;
                            let mut spent: u32 = 0u32;
                            while done == 0 {
                                if depth == n {
                                    found = 1;
                                    done = 1;
                                } else {
                                    let mut cand =
                                        ms2_id_slot(stack, stack_base, two_a + depth);
                                    let mut placed: u32 = 0u32;
                                    while cand < n && placed == 0 && spent == 0 {
                                        let mut viable: u32 = 0u32;
                                        if ms2_id_slot(stack, stack_base, atoms_cap + cand) == 0
                                        {
                                            viable = ms2_id_candidate_fits(
                                                cand,
                                                depth,
                                                actions,
                                                abase_kk,
                                                abase_jj,
                                                length_k,
                                                length_j,
                                                atoms_cap,
                                                scratch,
                                                sbase_k,
                                                sbase_j,
                                                bank,
                                                nb,
                                                stack,
                                                stack_base,
                                            );
                                        }
                                        if viable == 1 {
                                            if used >= work_max {
                                                spent = 1;
                                            } else {
                                                used += 1;
                                                ms2_id_put(stack, stack_base, depth, cand);
                                                ms2_id_put(
                                                    stack,
                                                    stack_base,
                                                    atoms_cap + cand,
                                                    1u32,
                                                );
                                                ms2_id_put(
                                                    stack,
                                                    stack_base,
                                                    two_a + depth,
                                                    cand + 1,
                                                );
                                                depth += 1;
                                                placed = 1;
                                            }
                                        }
                                        if placed == 0 && spent == 0 {
                                            cand += 1;
                                        }
                                    }
                                    if spent == 1 {
                                        done = 1;
                                    } else if placed == 0 {
                                        ms2_id_put(stack, stack_base, depth, 4294967295u32);
                                        ms2_id_put(stack, stack_base, two_a + depth, 0u32);
                                        if depth == 0 {
                                            done = 1;
                                        } else {
                                            depth -= 1;
                                            let prev_img = ms2_id_slot(stack, stack_base, depth);
                                            ms2_id_put(
                                                stack,
                                                stack_base,
                                                atoms_cap + prev_img,
                                                0u32,
                                            );
                                            ms2_id_put(stack, stack_base, depth, 4294967295u32);
                                        }
                                    }
                                }
                            }
                            if spent == 1 {
                                verdict = 2;
                            } else if found == 1 {
                                verdict = 1;
                            } else {
                                verdict = 0;
                            }
                        }
                        if verdict == 1 {
                            bits |= 128u32;
                        } else if verdict == 2 {
                            bits |= 256u32;
                            decided = 0;
                        }
                    }
                }
                j += 1;
            }
        }
        let mut resolution: u32 = 1u32;
        if decided == 0 {
            resolution = 2;
        }
        if eligible_k == 0 {
            bits = 0;
            resolution = 1;
        }
        if (record as usize) < identity.len() / 2 {
            identity[record as usize * 2] = bits;
            identity[record as usize * 2 + 1] = resolution;
        }
    }
}

/// Identity bits of one `(B, K)` bucket, lane per trajectory.
///
/// Producer preconditions (trusted, not re-read): every record's `length`
/// fits `steps`. The pair block additionally validates the decoded rows'
/// actual extents before decoding (an incomplete layout is unresolved, never
/// a verdict). Every bound buffer's word count and every narrowed scalar fit
/// `u32` (checked below); larger layouts would wrap device addresses and are
/// refused with [`Error::Shape`] before launch.
#[allow(clippy::too_many_arguments)]
pub fn graph_identity<R: Runtime>(
    actions: &IdTensor<R>,
    graph_hash: &IdTensor<R>,
    scratch: &IdTensor<R>,
    identity: &mut IdTensor<R>,
    stack: &mut IdTensor<R>,
    steps: usize,
    atoms_cap: u32,
    bonds_cap: u32,
    per_spectrum: usize,
    work_max: u32,
) -> Result<()> {
    if actions.shape().rank() != 2
        || graph_hash.shape().rank() != 1
        || scratch.shape().rank() != 2
        || identity.shape().rank() != 2
        || stack.shape().rank() != 2
    {
        return Err(Error::shape(format!(
            "graph_identity needs actions [rows, stride], graph_hash [rows], scratch [rows, G], identity [rows, 2] and stack [rows, 3A], got {} and {} and {} and {} and {}",
            actions.shape(),
            graph_hash.shape(),
            scratch.shape(),
            identity.shape(),
            stack.shape()
        )));
    }
    let rows = actions.shape().dim(0);
    let record_stride = actions.shape().dim(1);
    let scratch_stride = if rows == 0 { 0 } else { scratch.shape().dim(1) };
    let stack_stride = if rows == 0 { 0 } else { stack.shape().dim(1) };
    let want_identity: &[usize] = &[rows, 2];
    if graph_hash.len() != rows
        || scratch.shape().dims() != [rows, scratch_stride]
        || identity.shape().dims() != want_identity
        || stack.shape().dims() != [rows, stack_stride]
    {
        return Err(Error::shape(format!(
            "graph_identity needs graph_hash [{rows}], scratch [{rows}, {scratch_stride}], identity [{rows}, 2] and stack [{rows}, {stack_stride}], got {} and {} and {} and {}",
            graph_hash.shape(),
            scratch.shape(),
            identity.shape(),
            stack.shape()
        )));
    }
    let mut cap_a: u32 = atoms_cap;
    if cap_a == 0 {
        cap_a = 1;
    }
    if cap_a > 32 {
        return Err(Error::shape(format!(
            "graph_identity needs atoms_cap 1..=32, got {atoms_cap}"
        )));
    }
    if bonds_cap > 39 {
        return Err(Error::shape(format!(
            "graph_identity needs bonds_cap <= 39, got {bonds_cap}"
        )));
    }
    let expect_scratch = (bonds_cap as usize) * 3 + (cap_a as usize) * 2;
    if scratch_stride != expect_scratch {
        return Err(Error::shape(format!(
            "graph_identity needs scratch stride {expect_scratch} for A {cap_a} bonds {bonds_cap}, got {scratch_stride}"
        )));
    }
    if stack_stride != (cap_a as usize) * 3 {
        return Err(Error::shape(format!(
            "graph_identity needs stack stride {} for A {cap_a}, got {stack_stride}",
            (cap_a as usize) * 3
        )));
    }
    let expect_record = steps * 4 + cap_a as usize + 4;
    if record_stride != expect_record {
        return Err(Error::shape(format!(
            "graph_identity needs record stride {expect_record} for T {steps} A {cap_a}, got {record_stride}"
        )));
    }
    if per_spectrum == 0 || rows % per_spectrum != 0 {
        return Err(Error::shape(format!(
            "graph_identity needs rows {rows} divisible by per_spectrum {per_spectrum}"
        )));
    }
    check_device_len("actions", actions.len())?;
    check_device_len("graph_hash", graph_hash.len())?;
    check_device_len("scratch", scratch.len())?;
    check_device_len("identity", identity.len())?;
    check_device_len("stack", stack.len())?;
    let record_stride_u32 = check_device_scalar("graph_identity record stride", record_stride)?;
    let scratch_stride_u32 = check_device_scalar("graph_identity scratch stride", scratch_stride)?;
    let stack_stride_u32 = check_device_scalar("graph_identity stack stride", stack_stride)?;
    let steps_u32 = check_device_scalar("graph_identity steps", steps)?;
    let per_spectrum_u32 = check_device_scalar("graph_identity per_spectrum", per_spectrum)?;
    if rows == 0 {
        return Ok(());
    }
    let client = actions.client();
    let (count, dim, span) = launch_1d_spans(client, rows, expect_record);
    unsafe {
        ms2_graph_identity_kernel::launch_unchecked::<R>(
            client,
            count,
            dim,
            actions.arg(),
            graph_hash.arg(),
            scratch.arg(),
            identity.arg(),
            stack.arg(),
            record_stride_u32,
            scratch_stride_u32,
            stack_stride_u32,
            steps_u32,
            atoms_cap,
            bonds_cap,
            per_spectrum_u32,
            work_max,
            rows,
            span,
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// `ms2_allocate`: lane per spectrum
// ---------------------------------------------------------------------------

/// One guarded `u32` read from a flat buffer.
#[cube]
fn ms2_alloc_word(buf: &Array<u32>, base: u32, idx: u32, or: u32) -> u32 {
    let addr = base + idx;
    let mut out: u32 = or;
    if (addr as usize) < buf.len() {
        out = buf[addr as usize];
    }
    out
}

/// One guarded 12-word record write with 10 scalar counts.
#[allow(clippy::too_many_arguments)]
#[cube]
fn ms2_alloc_put_record(
    out: &mut Array<u32>,
    base: u32,
    t: u32,
    slot_v: u32,
    source: u32,
    c0: u32,
    c1: u32,
    c2: u32,
    c3: u32,
    c4: u32,
    c5: u32,
    c6: u32,
    c7: u32,
    c8: u32,
    c9: u32,
) {
    let row = base + t * 12u32;
    if (row as usize) < out.len() {
        out[row as usize] = slot_v;
    }
    if ((row + 1) as usize) < out.len() {
        out[(row + 1) as usize] = source;
    }
    if ((row + 2) as usize) < out.len() {
        out[(row + 2) as usize] = c0;
    }
    if ((row + 3) as usize) < out.len() {
        out[(row + 3) as usize] = c1;
    }
    if ((row + 4) as usize) < out.len() {
        out[(row + 4) as usize] = c2;
    }
    if ((row + 5) as usize) < out.len() {
        out[(row + 5) as usize] = c3;
    }
    if ((row + 6) as usize) < out.len() {
        out[(row + 6) as usize] = c4;
    }
    if ((row + 7) as usize) < out.len() {
        out[(row + 7) as usize] = c5;
    }
    if ((row + 8) as usize) < out.len() {
        out[(row + 8) as usize] = c6;
    }
    if ((row + 9) as usize) < out.len() {
        out[(row + 9) as usize] = c7;
    }
    if ((row + 10) as usize) < out.len() {
        out[(row + 10) as usize] = c8;
    }
    if ((row + 11) as usize) < out.len() {
        out[(row + 11) as usize] = c9;
    }
}

// (CubeCL 0.10 note for the f32 locals below: a method call on a concrete
// cube `f32` resolves to the host inherent method, and a bare host literal
// does not lift to a cube value, so the lane spells constants as
// `<f32 as Float>::new(..)` and `exp`/`floor` as `<f32 as Exp>::exp(..)` /
// `<f32 as Floor>::floor(..)`; widening loads stay `f32::cast_from(..)`,
// as in the pack kernel. See the removed probe battery for the matrix.)

/// Lane per spectrum of [`allocate`]; a line-for-line copy of
/// `allocate::allocate_lane` over `Array`s with the same `f32` operation order.
/// Each loaded log-probability is widened to f32 on load, so ALL allocation
/// arithmetic below (max, exp, sum, quotas, fractions) is f32 whatever the
/// neural element type — exactly as the host twin and spec §3.2 do.
#[allow(clippy::too_many_arguments)]
#[allow(unused_assignments)]
#[cube(launch_unchecked)]
fn ms2_allocate_kernel<F: Float + CubeElement>(
    top: &Array<u32>,
    top_counts: &Array<u32>,
    top_log_prob: &Array<F>,
    top_count: &Array<u32>,
    out: &mut Array<u32>,
    f: u32,
    k: u32,
    mode: u32,
    lanes: usize,
    span: usize,
) {
    // Finite means strictly inside (-3e38, 3e38): the `ms2.rs` FINITE_MAX
    // range test, required because fast-math backends fold `v - v == 0`
    // into a constant. The validated input domain is that open interval;
    // entries with |v| >= 3e38 fall back to round robin with NaN/±infinity.
    // (The bounds spell the literals inline, as the pack kernel does.)
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    for pos in start..end {
        let spectrum = pos as u32;
        let top_base = spectrum * (f * 2);
        let counts_base = spectrum * (f * 10);
        let lp_base = spectrum * f;
        let out_base = spectrum * (k * 12u32);
        let mut n: u32 = 0u32;
        if (spectrum as usize) < top_count.len() {
            n = top_count[spectrum as usize];
        }
        if n > f {
            n = f;
        }
        let mut ok: u32 = 1u32;
        if k > 64u32 {
            ok = 0;
        }
        if f > 8u32 {
            ok = 0;
        }
        if (spectrum as usize) < top_count.len() && top_count[spectrum as usize] > f {
            ok = 0;
        }
        if (top.len() as u32) < (spectrum + 1) * (f * 2) {
            ok = 0;
        }
        if (top_counts.len() as u32) < (spectrum + 1) * (f * 10) {
            ok = 0;
        }
        if (top_log_prob.len() as u32) < (spectrum + 1) * f {
            ok = 0;
        }
        if (out.len() as u32) < (spectrum + 1) * (k * 12u32) {
            ok = 0;
        }
        if ok == 0 || n == 0 {
            let mut t: u32 = 0u32;
            while t < k {
                ms2_alloc_put_record(
                    out, out_base, t, 4294967295u32, 4294967295u32, 0u32, 0u32, 0u32,
                    0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32,
                );
                t += 1;
            }
        } else if mode == 0u32 {
            let mut t: u32 = 0u32;
            while t < k {
                let s = t % n;
                let source = ms2_alloc_word(top, top_base, s * 2, 4294967295u32);
                let c0 = ms2_alloc_word(top_counts, counts_base, s * 10, 0u32);
                let c1 = ms2_alloc_word(top_counts, counts_base, s * 10 + 1, 0u32);
                let c2 = ms2_alloc_word(top_counts, counts_base, s * 10 + 2, 0u32);
                let c3 = ms2_alloc_word(top_counts, counts_base, s * 10 + 3, 0u32);
                let c4 = ms2_alloc_word(top_counts, counts_base, s * 10 + 4, 0u32);
                let c5 = ms2_alloc_word(top_counts, counts_base, s * 10 + 5, 0u32);
                let c6 = ms2_alloc_word(top_counts, counts_base, s * 10 + 6, 0u32);
                let c7 = ms2_alloc_word(top_counts, counts_base, s * 10 + 7, 0u32);
                let c8 = ms2_alloc_word(top_counts, counts_base, s * 10 + 8, 0u32);
                let c9 = ms2_alloc_word(top_counts, counts_base, s * 10 + 9, 0u32);
                ms2_alloc_put_record(out, out_base, t, s, source, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9);
                t += 1;
            }
        } else if k <= n {
            let mut t: u32 = 0u32;
            while t < k {
                let s = t;
                let source = ms2_alloc_word(top, top_base, s * 2, 4294967295u32);
                let c0 = ms2_alloc_word(top_counts, counts_base, s * 10, 0u32);
                let c1 = ms2_alloc_word(top_counts, counts_base, s * 10 + 1, 0u32);
                let c2 = ms2_alloc_word(top_counts, counts_base, s * 10 + 2, 0u32);
                let c3 = ms2_alloc_word(top_counts, counts_base, s * 10 + 3, 0u32);
                let c4 = ms2_alloc_word(top_counts, counts_base, s * 10 + 4, 0u32);
                let c5 = ms2_alloc_word(top_counts, counts_base, s * 10 + 5, 0u32);
                let c6 = ms2_alloc_word(top_counts, counts_base, s * 10 + 6, 0u32);
                let c7 = ms2_alloc_word(top_counts, counts_base, s * 10 + 7, 0u32);
                let c8 = ms2_alloc_word(top_counts, counts_base, s * 10 + 8, 0u32);
                let c9 = ms2_alloc_word(top_counts, counts_base, s * 10 + 9, 0u32);
                ms2_alloc_put_record(out, out_base, t, s, source, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9);
                t += 1;
            }
        } else {
            let mut finite: u32 = 1u32;
            let mut fchk: u32 = 0u32;
            while fchk < n {
                // Widened to f32 on load: all allocation arithmetic is f32.
                let v: f32 = f32::cast_from(top_log_prob[(lp_base + fchk) as usize]);
                let mut flag: u32 = 0u32;
                if v > -3.0e38_f32 && v < 3.0e38_f32 {
                    flag = 1;
                }
                if flag == 0 {
                    finite = 0;
                }
                fchk += 1;
            }
            if finite == 0 {
                let mut t: u32 = 0u32;
                while t < k {
                    let s = t % n;
                    let source = ms2_alloc_word(top, top_base, s * 2, 4294967295u32);
                    let c0 = ms2_alloc_word(top_counts, counts_base, s * 10, 0u32);
                    let c1 = ms2_alloc_word(top_counts, counts_base, s * 10 + 1, 0u32);
                    let c2 = ms2_alloc_word(top_counts, counts_base, s * 10 + 2, 0u32);
                    let c3 = ms2_alloc_word(top_counts, counts_base, s * 10 + 3, 0u32);
                    let c4 = ms2_alloc_word(top_counts, counts_base, s * 10 + 4, 0u32);
                    let c5 = ms2_alloc_word(top_counts, counts_base, s * 10 + 5, 0u32);
                    let c6 = ms2_alloc_word(top_counts, counts_base, s * 10 + 6, 0u32);
                    let c7 = ms2_alloc_word(top_counts, counts_base, s * 10 + 7, 0u32);
                    let c8 = ms2_alloc_word(top_counts, counts_base, s * 10 + 8, 0u32);
                    let c9 = ms2_alloc_word(top_counts, counts_base, s * 10 + 9, 0u32);
                    ms2_alloc_put_record(out, out_base, t, s, source, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9);
                    t += 1;
                }
            } else {
                let mut max: f32 = f32::cast_from(top_log_prob[lp_base as usize]);
                let mut fi: u32 = 1u32;
                while fi < n {
                    let v: f32 = f32::cast_from(top_log_prob[(lp_base + fi) as usize]);
                    if v > max {
                        max = v;
                    }
                    fi += 1;
                }
                let mut sum: f32 = <f32 as Float>::new(0.0);
                let mut g: u32 = 0u32;
                while g < n {
                    sum = sum + <f32 as Exp>::exp(f32::cast_from(top_log_prob[(lp_base + g) as usize]) - max);
                    g += 1;
                }
                let rest_k = k - n;
                let rest_f: f32 = f32::cast_from(rest_k);
                let mut f0: u32 = 0u32;
                let mut f1: u32 = 0u32;
                let mut f2: u32 = 0u32;
                let mut f3: u32 = 0u32;
                let mut f4: u32 = 0u32;
                let mut f5: u32 = 0u32;
                let mut f6: u32 = 0u32;
                let mut f7: u32 = 0u32;
                let mut e0: f32 = <f32 as Float>::new(0.0);
                let mut e1: f32 = <f32 as Float>::new(0.0);
                let mut e2: f32 = <f32 as Float>::new(0.0);
                let mut e3: f32 = <f32 as Float>::new(0.0);
                let mut e4: f32 = <f32 as Float>::new(0.0);
                let mut e5: f32 = <f32 as Float>::new(0.0);
                let mut e6: f32 = <f32 as Float>::new(0.0);
                let mut e7: f32 = <f32 as Float>::new(0.0);
                let mut d0: u32 = 0u32;
                let mut d1: u32 = 0u32;
                let mut d2: u32 = 0u32;
                let mut d3: u32 = 0u32;
                let mut d4: u32 = 0u32;
                let mut d5: u32 = 0u32;
                let mut d6: u32 = 0u32;
                let mut d7: u32 = 0u32;
                let mut h: u32 = 0u32;
                while h < n {
                    let p: f32 = <f32 as Exp>::exp(f32::cast_from(top_log_prob[(lp_base + h) as usize]) - max) / sum;
                    let quota = rest_f * p;
                    let floor = u32::cast_from(<f32 as Floor>::floor(quota));
                    let frac: f32 = quota - f32::cast_from(floor);
                    if h == 0 {
                        f0 = floor;
                        e0 = frac;
                    } else if h == 1 {
                        f1 = floor;
                        e1 = frac;
                    } else if h == 2 {
                        f2 = floor;
                        e2 = frac;
                    } else if h == 3 {
                        f3 = floor;
                        e3 = frac;
                    } else if h == 4 {
                        f4 = floor;
                        e4 = frac;
                    } else if h == 5 {
                        f5 = floor;
                        e5 = frac;
                    } else if h == 6 {
                        f6 = floor;
                        e6 = frac;
                    } else {
                        f7 = floor;
                        e7 = frac;
                    }
                    h += 1;
                }
                let mut fsum: u32 = 0u32;
                if n > 0 {
                    fsum += f0;
                }
                if n > 1 {
                    fsum += f1;
                }
                if n > 2 {
                    fsum += f2;
                }
                if n > 3 {
                    fsum += f3;
                }
                if n > 4 {
                    fsum += f4;
                }
                if n > 5 {
                    fsum += f5;
                }
                if n > 6 {
                    fsum += f6;
                }
                if n > 7 {
                    fsum += f7;
                }
                if fsum > rest_k {
                    let mut excess: u32 = fsum - rest_k;
                    if n > 0 {
                        let mut take: u32 = f0;
                        if take > excess {
                            take = excess;
                        }
                        f0 -= take;
                        excess -= take;
                    }
                    if n > 1 {
                        let mut take: u32 = f1;
                        if take > excess {
                            take = excess;
                        }
                        f1 -= take;
                        excess -= take;
                    }
                    if n > 2 {
                        let mut take: u32 = f2;
                        if take > excess {
                            take = excess;
                        }
                        f2 -= take;
                        excess -= take;
                    }
                    if n > 3 {
                        let mut take: u32 = f3;
                        if take > excess {
                            take = excess;
                        }
                        f3 -= take;
                        excess -= take;
                    }
                    if n > 4 {
                        let mut take: u32 = f4;
                        if take > excess {
                            take = excess;
                        }
                        f4 -= take;
                        excess -= take;
                    }
                    if n > 5 {
                        let mut take: u32 = f5;
                        if take > excess {
                            take = excess;
                        }
                        f5 -= take;
                        excess -= take;
                    }
                    if n > 6 {
                        let mut take: u32 = f6;
                        if take > excess {
                            take = excess;
                        }
                        f6 -= take;
                        excess -= take;
                    }
                    if n > 7 {
                        let mut take: u32 = f7;
                        if take > excess {
                            take = excess;
                        }
                        f7 -= take;
                        excess -= take;
                    }
                    fsum = rest_k;
                }
                let mut rest: u32 = rest_k - fsum;
                let mut picks: u32 = 0u32;
                while rest > 0 && picks < n {
                    let mut best: u32 = 0u32;
                    let mut best_frac: f32 = <f32 as Float>::new(-1.0);
                    let mut q: u32 = 0u32;
                    while q < n {
                        let fq = if q == 0 {
                            e0
                        } else if q == 1 {
                            e1
                        } else if q == 2 {
                            e2
                        } else if q == 3 {
                            e3
                        } else if q == 4 {
                            e4
                        } else if q == 5 {
                            e5
                        } else if q == 6 {
                            e6
                        } else {
                            e7
                        };
                        let pq = if q == 0 {
                            d0
                        } else if q == 1 {
                            d1
                        } else if q == 2 {
                            d2
                        } else if q == 3 {
                            d3
                        } else if q == 4 {
                            d4
                        } else if q == 5 {
                            d5
                        } else if q == 6 {
                            d6
                        } else {
                            d7
                        };
                        if pq == 0 && fq > best_frac {
                            best_frac = fq;
                            best = q;
                        }
                        q += 1;
                    }
                    if best == 0 {
                        d0 = 1;
                        f0 += 1;
                    } else if best == 1 {
                        d1 = 1;
                        f1 += 1;
                    } else if best == 2 {
                        d2 = 1;
                        f2 += 1;
                    } else if best == 3 {
                        d3 = 1;
                        f3 += 1;
                    } else if best == 4 {
                        d4 = 1;
                        f4 += 1;
                    } else if best == 5 {
                        d5 = 1;
                        f5 += 1;
                    } else if best == 6 {
                        d6 = 1;
                        f6 += 1;
                    } else {
                        d7 = 1;
                        f7 += 1;
                    }
                    rest -= 1;
                    picks += 1;
                }
                let mut t: u32 = 0u32;
                let mut s: u32 = 0u32;
                while s < n {
                    let mut count: u32 = if s == 0 {
                        f0 + 1
                    } else if s == 1 {
                        f1 + 1
                    } else if s == 2 {
                        f2 + 1
                    } else if s == 3 {
                        f3 + 1
                    } else if s == 4 {
                        f4 + 1
                    } else if s == 5 {
                        f5 + 1
                    } else if s == 6 {
                        f6 + 1
                    } else {
                        f7 + 1
                    };
                    while count > 0 && t < k {
                        let source = ms2_alloc_word(top, top_base, s * 2, 4294967295u32);
                        let c0 = ms2_alloc_word(top_counts, counts_base, s * 10, 0u32);
                        let c1 = ms2_alloc_word(top_counts, counts_base, s * 10 + 1, 0u32);
                        let c2 = ms2_alloc_word(top_counts, counts_base, s * 10 + 2, 0u32);
                        let c3 = ms2_alloc_word(top_counts, counts_base, s * 10 + 3, 0u32);
                        let c4 = ms2_alloc_word(top_counts, counts_base, s * 10 + 4, 0u32);
                        let c5 = ms2_alloc_word(top_counts, counts_base, s * 10 + 5, 0u32);
                        let c6 = ms2_alloc_word(top_counts, counts_base, s * 10 + 6, 0u32);
                        let c7 = ms2_alloc_word(top_counts, counts_base, s * 10 + 7, 0u32);
                        let c8 = ms2_alloc_word(top_counts, counts_base, s * 10 + 8, 0u32);
                        let c9 = ms2_alloc_word(top_counts, counts_base, s * 10 + 9, 0u32);
                        ms2_alloc_put_record(out, out_base, t, s, source, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9);
                        t += 1;
                        count -= 1;
                    }
                    s += 1;
                }
                while t < k {
                    let sv = n - 1;
                    let source = ms2_alloc_word(top, top_base, sv * 2, 4294967295u32);
                    let c0 = ms2_alloc_word(top_counts, counts_base, sv * 10, 0u32);
                    let c1 = ms2_alloc_word(top_counts, counts_base, sv * 10 + 1, 0u32);
                    let c2 = ms2_alloc_word(top_counts, counts_base, sv * 10 + 2, 0u32);
                    let c3 = ms2_alloc_word(top_counts, counts_base, sv * 10 + 3, 0u32);
                    let c4 = ms2_alloc_word(top_counts, counts_base, sv * 10 + 4, 0u32);
                    let c5 = ms2_alloc_word(top_counts, counts_base, sv * 10 + 5, 0u32);
                    let c6 = ms2_alloc_word(top_counts, counts_base, sv * 10 + 6, 0u32);
                    let c7 = ms2_alloc_word(top_counts, counts_base, sv * 10 + 7, 0u32);
                    let c8 = ms2_alloc_word(top_counts, counts_base, sv * 10 + 8, 0u32);
                    let c9 = ms2_alloc_word(top_counts, counts_base, sv * 10 + 9, 0u32);
                    ms2_alloc_put_record(out, out_base, t, sv, source, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9);
                    t += 1;
                }
            }
        }
    }
}

/// Trajectory allocation of one `(B, K)` bucket, lane per spectrum.
///
/// The contract is FP32 (architecture §3.2): the loaded log-probabilities
/// are widened to f32 on load and every max, `exp`, sum, quota and remainder
/// is f32 arithmetic in the lane's fixed order, whatever the neural element
/// type (`f32` or `bf16`, the validated dtypes of contracts §3.3); any other
/// float dtype is refused with [`Error::Unsupported`] naming the dtype.
/// Finite means strictly inside `(-3e38, 3e38)` (the `ms2.rs` `FINITE_MAX`
/// range test, required by fast-math backends): the validated input domain
/// of `top_log_prob` is that open interval, and entries with `|v| >= 3e38`
/// fall back to round robin together with NaN and ±infinity. Every bound
/// buffer's word count and every narrowed scalar fit `u32` (checked below);
/// larger layouts would wrap device addresses and are refused with
/// [`Error::Shape`] before launch.
pub fn allocate<R: Runtime, E: FloatElem>(
    top: &IdTensor<R>,
    top_counts: &IdTensor<R>,
    top_log_prob: &Tensor<R, E>,
    top_count: &IdTensor<R>,
    out: &mut IdTensor<R>,
    mode: u32,
) -> Result<()> {
    if top.shape().rank() != 3
        || top_counts.shape().rank() != 3
        || top_log_prob.shape().rank() != 2
        || top_count.shape().rank() != 1
        || out.shape().rank() != 3
    {
        return Err(Error::shape(format!(
            "allocate needs top [B, F, 2], top_counts [B, F, 10], top_log_prob [B, F], top_count [B] and out [B, K, 12], got {} and {} and {} and {} and {}",
            top.shape(),
            top_counts.shape(),
            top_log_prob.shape(),
            top_count.shape(),
            out.shape()
        )));
    }
    let batch = top.shape().dim(0);
    let f = top.shape().dim(1);
    let k = out.shape().dim(1);
    let top_last = top.shape().dim(2);
    let counts_last = top_counts.shape().dim(2);
    let out_last = out.shape().dim(2);
    if top_last != 2 || counts_last != 10 || out_last != 12 {
        return Err(Error::shape(format!(
            "allocate needs top [B, F, 2], top_counts [B, F, 10] and out [B, K, 12], got last dimensions {top_last} and {counts_last} and {out_last}"
        )));
    }
    let want_top: &[usize] = &[batch, f, 2];
    let want_counts: &[usize] = &[batch, f, 10];
    let want_lp: &[usize] = &[batch, f];
    let want_count: &[usize] = &[batch];
    let want_out: &[usize] = &[batch, k, 12];
    if top.shape().dims() != want_top
        || top_counts.shape().dims() != want_counts
        || top_log_prob.shape().dims() != want_lp
        || top_count.shape().dims() != want_count
        || out.shape().dims() != want_out
    {
        return Err(Error::shape(format!(
            "allocate needs top [{batch}, {f}, 2], top_counts [{batch}, {f}, 10], top_log_prob [{batch}, {f}], top_count [{batch}] and out [{batch}, {k}, 12], got {} and {} and {} and {} and {}",
            top.shape(),
            top_counts.shape(),
            top_log_prob.shape(),
            top_count.shape(),
            out.shape()
        )));
    }
    if f > 8 {
        return Err(Error::shape(format!(
            "allocate needs F <= 8, got {f}"
        )));
    }
    if k > 64 {
        return Err(Error::shape(format!(
            "allocate needs K <= 64, got {k}"
        )));
    }
    if !matches!(E::DTYPE, DType::F32 | DType::BF16) {
        return Err(Error::Unsupported(format!(
            "allocate needs an f32 or bf16 top_log_prob (the validated dtypes of contracts §3.3), got {}",
            E::DTYPE.name()
        )));
    }
    check_device_len("top", top.len())?;
    check_device_len("top_counts", top_counts.len())?;
    check_device_len("top_log_prob", top_log_prob.len())?;
    check_device_len("top_count", top_count.len())?;
    check_device_len("traj_formula", out.len())?;
    let f_u32 = check_device_scalar("allocate F", f)?;
    let k_u32 = check_device_scalar("allocate K", k)?;
    if batch == 0 {
        return Ok(());
    }
    let _shape: Shape = out.shape().clone();
    let client = top.client();
    let (count, dim, span) = launch_1d_spans(client, batch, k);
    unsafe {
        ms2_allocate_kernel::launch_unchecked::<E, R>(
            client,
            count,
            dim,
            top.arg(),
            top_counts.arg(),
            top_log_prob.arg(),
            top_count.arg(),
            out.arg(),
            f_u32,
            k_u32,
            mode,
            batch,
            span,
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// `ms2_allocate_window`: lane per trajectory
// ---------------------------------------------------------------------------

/// Lane per trajectory of [`allocate_window`]; a line-for-line copy of
/// `allocate::alloc_window_lane` over `Array`s.
#[cube(launch_unchecked)]
fn ms2_allocate_window_kernel(
    traj_in: &Array<u32>,
    top: &Array<u32>,
    out: &mut Array<u32>,
    f: u32,
    k: u32,
    lanes: usize,
    span: usize,
) {
    let start = ABSOLUTE_POS * span;
    let mut end = start + span;
    if end > lanes {
        end = lanes;
    }
    for pos in start..end {
        let traj = pos as u32;
        let b = traj / k;
        let kk = traj % k;
        let src = (b * k + kk) * 12u32;
        let mut s: u32 = 4294967295u32;
        if (src as usize) < traj_in.len() {
            s = traj_in[src as usize];
        }
        let mut w: u32 = s;
        if s != 4294967295u32 && s < f && (((b * f + s) * 2u32 + 1u32) as usize) < top.len() {
            w = top[((b * f + s) * 2u32 + 1u32) as usize];
        }
        let dst = (b * k + kk) * 12u32;
        let mut i: u32 = 0u32;
        while i < 12u32 {
            let mut v: u32 = w;
            if i != 0u32 {
                v = 0u32;
                if ((src + i) as usize) < traj_in.len() {
                    v = traj_in[(src + i) as usize];
                }
            }
            if ((dst + i) as usize) < out.len() {
                out[(dst + i) as usize] = v;
            }
            i += 1;
        }
    }
}

/// Window slots of one `(B, K)` allocation: lane per trajectory.
///
/// `traj_in` is the `[B, K, 12]` buffer [`allocate`] wrote, `top` is
/// `[B, F, 2]` and `out` is `[B, K, 12]`. Every record is copied with word 0
/// (the retained formula index) mapped through `top[(b, s), 1]` to the
/// window slot — the formula rank the packed record carries (V1 §4.4). A
/// sentinel or out-of-range slot passes through unchanged. 3 arrays, exactly
/// 1 launch per packed/resident call.
pub fn allocate_window<R: Runtime>(
    traj_in: &IdTensor<R>,
    top: &IdTensor<R>,
    out: &mut IdTensor<R>,
) -> Result<()> {
    if traj_in.shape().rank() != 3 || top.shape().rank() != 3 || out.shape().rank() != 3 {
        return Err(Error::shape(format!(
            "allocate_window needs traj_in [B, K, 12], top [B, F, 2] and out [B, K, 12], got {} and {} and {}",
            traj_in.shape(),
            top.shape(),
            out.shape()
        )));
    }
    let batch = traj_in.shape().dim(0);
    let k = traj_in.shape().dim(1);
    let f = top.shape().dim(1);
    if traj_in.shape().dim(2) != 12 || out.shape().dim(2) != 12 || top.shape().dim(2) != 2 {
        return Err(Error::shape(format!(
            "allocate_window needs traj_in [B, K, 12], top [B, F, 2] and out [B, K, 12], got last dimensions {} and {} and {}",
            traj_in.shape().dim(2),
            top.shape().dim(2),
            out.shape().dim(2)
        )));
    }
    let want_top: &[usize] = &[batch, f, 2];
    let want_out: &[usize] = &[batch, k, 12];
    if top.shape().dims() != want_top || out.shape().dims() != want_out {
        return Err(Error::shape(format!(
            "allocate_window needs top [{batch}, {f}, 2] and out [{batch}, {k}, 12], got {} and {}",
            top.shape(),
            out.shape()
        )));
    }
    if f > 8 {
        return Err(Error::shape(format!(
            "allocate_window needs F <= 8, got {f}"
        )));
    }
    if k > 64 {
        return Err(Error::shape(format!(
            "allocate_window needs K <= 64, got {k}"
        )));
    }
    check_device_len("traj_in", traj_in.len())?;
    check_device_len("top", top.len())?;
    check_device_len("traj_window", out.len())?;
    let f_u32 = check_device_scalar("allocate_window F", f)?;
    let k_u32 = check_device_scalar("allocate_window K", k)?;
    let rows = batch * k;
    if rows == 0 {
        return Ok(());
    }
    // One lane per trajectory; `rows > 0` implies `k > 0`, so the kernel's
    // division by `k` is exact.
    let client = traj_in.client();
    let (count, dim, span) = launch_1d_spans(client, rows, 12);
    unsafe {
        ms2_allocate_window_kernel::launch_unchecked::<R>(
            client,
            count,
            dim,
            traj_in.arg(),
            top.arg(),
            out.arg(),
            f_u32,
            k_u32,
            rows,
            span,
        );
    }
    Ok(())
}
