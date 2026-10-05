//! Graph identity host twins of `docs/MS2_V1_ARCHITECTURE.md` §4.2 (plan item
//! P6.2, host side).
//!
//! Pure host Rust with `u32` arithmetic only: no tensors, no kernels, no
//! floats, no recursion. The `#[cube]` kernels `ms2_graph_hash` and
//! `ms2_graph_identity` copy [`graph_hash_lane`] and [`identity_lane`] (with
//! [`graph_equal_lane`] as a shared helper) line for line, while
//! [`identity_batch`] is the host wrapper around those lanes and is NOT copied
//! to the kernel: it owns the `Vec` allocation and packs `CandidateBatch`
//! records into device-record layout.
//!
//! Kernel-expressible form (CubeCL 0.10 cannot express Rust fixed-size local
//! arrays `[u32; N]`, so the twins use none): every value is `u32` with plain
//! `+`/`-`/`*`/`/`/`%` (wrapping in release and on device, where `u32`
//! arithmetic wraps), except that intentional wrapping of the hash sums is
//! spelled `wrapping_add` on the host so debug builds do not panic while the
//! kernel keeps plain device addition (same values, review finding 1); `usize` appears only to bound
//! a slice or to index it; loops are `while` loops over `u32` counters;
//! scalar capacities and every buffer offset are passed explicitly as `u32`;
//! full bound buffers are addressed with explicit record indices, strides and
//! base offsets (no record-local slices); atom types are re-decoded from
//! `actions` on demand by [`atom_type_at`] (no local type arrays); every lane
//! has a single exit; nesting stays at most 5 deep. The refinement sum stays
//! commutative with wrapping `u32` arithmetic, so the hash does not depend on
//! atom order.
//!
//! Device-record layout (what the kernels bind): one row per trajectory of
//! `record_stride = steps * 4 + atoms_cap + 4` words holding `steps * 4` token
//! words, `atoms_cap` open-valence words, then length, status,
//! trace-log-probability bits and formula row.

//! CubeCL 0.10 has no `RangeInclusive::contains` in kernels, so the lanes use
//! explicit comparisons; the clippy lint for that pattern is allowed here
//! rather than rewritten, keeping the twin line-for-line identical with the
//! kernel (as in `crate::tensor::ops::ms2_identity`).
#![allow(clippy::manual_range_contains)]

use super::contract::{CandidateBatch, candidate_status};
use super::grammar::{ADD_ATOM, CLOSE_RING};
use super::twin::hash_u32_host;

/// Candidate status bit 7: an exact comparison proved equality with an
/// earlier trajectory of the same spectrum (contracts §8).
pub const DUPLICATE_GRAPH: u32 = candidate_status::DUPLICATE_GRAPH;
/// Candidate status bit 8: some comparison of this (the later) trajectory ran
/// out of budget (contracts §8).
pub const IDENTITY_UNRESOLVED: u32 = candidate_status::IDENTITY_UNRESOLVED;
/// `identity_resolution` value: every comparison of this candidate was decided.
pub const RESOLUTION_EXACT: u8 = 1;
/// `identity_resolution` value: some comparison of this candidate was
/// unresolved (budget spent before the search finished).
pub const RESOLUTION_UNRESOLVED: u8 = 2;
/// Largest atom cap a lane handles.
pub const ATOMS_MAX: u32 = 32;
/// Largest bond count a lane handles (`32 - 1 + 8`).
pub const BONDS_MAX: u32 = 39;
/// Request-level exact-comparison work bound of V1 §4.2:
/// `B * K * (K - 1) / 2 * identity_work_max` must not exceed this, else the
/// request is refused before dispatch.
pub const IDENTITY_REQUEST_WORK_MAX: u64 = 1 << 28;

/// Scratch words of [`graph_hash_lane`] for these caps:
/// `G = 3 * (A - 1 + R_max) + 2 * A` words holding the bond list as
/// `(atom, atom, order)` triples followed by the two label banks of `A`
/// words each. Saturates instead of overflowing on absurd caps.
pub fn graph_scratch_len(atoms_cap: u32, closures_cap: u32) -> usize {
    let bonds = atoms_cap.saturating_sub(1).saturating_add(closures_cap) as usize;
    let atoms = atoms_cap as usize;
    3usize.saturating_mul(bonds).saturating_add(2usize.saturating_mul(atoms))
}

/// Lane-owned scratch words of [`graph_equal_lane`] for this atom cap:
/// `3 * A` words holding the assignment (`A`), the used flags (`A`) and the
/// cursor per depth (`A`).
pub fn identity_stack_len(atoms_cap: u32) -> usize {
    3usize.saturating_mul(atoms_cap as usize)
}

/// Words of one device trajectory record for these caps:
/// `steps * 4 + atoms_cap + 4` (tokens, open valence, length, status,
/// trace-log-probability bits, formula row). Saturates on absurd caps.
pub fn identity_record_len(steps: u32, atoms_cap: u32) -> usize {
    (steps as usize)
        .saturating_mul(4)
        .saturating_add(atoms_cap as usize)
        .saturating_add(4)
}

/// Two-argument hash through the sampler's mixing function.
fn hash_pair(first: u32, second: u32) -> u32 {
    hash_u32_host(first, second, 0)
}

/// One guarded full-buffer read: `buf[base + idx]` when it exists, else `0`.
fn slot(buf: &[u32], base: u32, idx: u32) -> u32 {
    let addr = base + idx;
    let mut out = 0u32;
    if (addr as usize) < buf.len() {
        out = buf[addr as usize];
    }
    out
}

/// One guarded full-buffer write.
fn put(buf: &mut [u32], base: u32, idx: u32, value: u32) {
    let addr = base + idx;
    if (addr as usize) < buf.len() {
        buf[addr as usize] = value;
    }
}

/// One guarded token-field read from a device record.
fn token_word(actions: &[u32], abase: u32, field: u32, or: u32) -> u32 {
    let addr = abase + field;
    let mut out = or;
    if (addr as usize) < actions.len() {
        out = actions[addr as usize];
    }
    out
}

/// Bond order between `x` and `y` in the bond list at `sbase`, else 0.
fn bond_between(scratch: &[u32], sbase: u32, n_bonds: u32, x: u32, y: u32) -> u32 {
    let mut order = 0u32;
    let mut bi = 0u32;
    while bi < n_bonds {
        let triple = bi * 3;
        let a = slot(scratch, sbase, triple);
        let b = slot(scratch, sbase, triple + 1);
        let o = slot(scratch, sbase, triple + 2);
        let hits = (a == x && b == y) || (a == y && b == x);
        if hits {
            order = o;
        }
        bi += 1;
    }
    order
}

/// Atom type of the `target`-th atom (0-based in trace order) in a device
/// record, or 0 when there is none.
///
/// Scans the `ADD_ATOM` tokens with the lane's skip rule for atoms (words past
/// `atoms_cap` are skipped; bond validity does not affect the atom count, so
/// only the kind and the cap matter here). No local array is needed.
fn atom_type_at(actions: &[u32], abase: u32, length: u32, atoms_cap: u32, target: u32) -> u32 {
    let add_kind = u32::from(ADD_ATOM);
    let mut cap_a = atoms_cap;
    if cap_a > ATOMS_MAX {
        cap_a = ATOMS_MAX;
    }
    let mut out = 0u32;
    let mut n = 0u32;
    let mut s = 0u32;
    while s < length {
        let tok = s * 4;
        let kind = token_word(actions, abase, tok, u32::MAX);
        let ty = token_word(actions, abase, tok + 1, 0);
        if kind == add_kind && n < cap_a {
            if n == target {
                out = ty;
            }
            n += 1;
        }
        s += 1;
    }
    out
}

/// Atom and bond counts of a device record with the lane's skip rules exactly
/// (root has no bond; out-of-range pointers, orders outside 1–3 and words past
/// the caps are skipped), so the counts agree with what [`graph_hash_lane`]
/// hashed. No type array is staged.
fn decode_counts(
    actions: &[u32],
    abase: u32,
    length: u32,
    atoms_cap: u32,
    bonds_cap: u32,
) -> (u32, u32) {
    let add_kind = u32::from(ADD_ATOM);
    let close_kind = u32::from(CLOSE_RING);
    let mut cap_a = atoms_cap;
    if cap_a > ATOMS_MAX {
        cap_a = ATOMS_MAX;
    }
    let mut n_atoms = 0u32;
    let mut n_bonds = 0u32;
    let mut s = 0u32;
    while s < length {
        let tok = s * 4;
        let kind = token_word(actions, abase, tok, u32::MAX);
        let ty = token_word(actions, abase, tok + 1, 0);
        let order = token_word(actions, abase, tok + 2, 0);
        let ptr = token_word(actions, abase, tok + 3, 0);
        let _ = ty;
        if kind == add_kind {
            if n_atoms < cap_a {
                if n_atoms > 0 && ptr < n_atoms && order >= 1 && order <= 3 && n_bonds < bonds_cap
                {
                    n_bonds += 1;
                }
                n_atoms += 1;
            }
        } else if kind == close_kind && n_atoms > 0 {
            let newest = n_atoms - 1;
            if ptr < newest && order >= 1 && order <= 3 && n_bonds < bonds_cap {
                n_bonds += 1;
            }
        }
        s += 1;
    }
    (n_atoms, n_bonds)
}

/// The graph hash lane: the kernel twin of `ms2_graph_hash`.
///
/// Full-buffer form over the kernel's bound buffers (see the module docs).
/// Rebuilds the bond list and runs 4 synchronous refinement rounds; atoms
/// past the caps are skipped deterministically and padding atoms keep label 0.
/// Malformed tokens are skipped; the lane assumes legal traces.
#[allow(clippy::too_many_arguments)]
pub fn graph_hash_lane(
    actions: &[u32],
    record: u32,
    record_stride: u32,
    steps: u32,
    atoms_cap: u32,
    closures_cap: u32,
    hash_mask: u32,
    graph_hash: &mut [u32],
    scratch: &mut [u32],
    scratch_stride: u32,
) -> u32 {
    let add_kind = u32::from(ADD_ATOM);
    let close_kind = u32::from(CLOSE_RING);
    let mut cap_a = atoms_cap;
    if cap_a == 0 {
        cap_a = 1;
    }
    if cap_a > ATOMS_MAX {
        cap_a = ATOMS_MAX;
    }
    let mut bonds_max = cap_a - 1 + closures_cap;
    if bonds_max > BONDS_MAX {
        bonds_max = BONDS_MAX;
    }
    // Guard against a zero cap wrapping the subtraction above in debug builds:
    // `cap_a` is at least 1 here, so the subtraction is exact.
    let bank_words = bonds_max * 3;
    let bank1 = bank_words + cap_a;
    let abase = record * record_stride;
    let sbase = record * scratch_stride;
    let len_field = steps * 4 + cap_a;
    let length = token_word(actions, abase, len_field, 0);
    let mut zi = 0u32;
    while zi < scratch_stride {
        put(scratch, sbase, zi, 0);
        zi += 1;
    }
    let mut n_atoms = 0u32;
    let mut n_bonds = 0u32;
    let mut s = 0u32;
    while s < length {
        let tok = s * 4;
        let kind = token_word(actions, abase, tok, u32::MAX);
        let order = token_word(actions, abase, tok + 2, 0);
        let ptr = token_word(actions, abase, tok + 3, 0);
        if kind == add_kind {
            if n_atoms < cap_a {
                if n_atoms > 0 && ptr < n_atoms && order >= 1 && order <= 3 && n_bonds < bonds_max
                {
                    let triple = n_bonds * 3;
                    put(scratch, sbase, triple, n_atoms);
                    put(scratch, sbase, triple + 1, ptr);
                    put(scratch, sbase, triple + 2, order);
                    n_bonds += 1;
                }
                n_atoms += 1;
            }
        } else if kind == close_kind && n_atoms > 0 {
            let newest = n_atoms - 1;
            if ptr < newest && order >= 1 && order <= 3 && n_bonds < bonds_max {
                let triple = n_bonds * 3;
                put(scratch, sbase, triple, newest);
                put(scratch, sbase, triple + 1, ptr);
                put(scratch, sbase, triple + 2, order);
                n_bonds += 1;
            }
        }
        s += 1;
    }
    let mut i = 0u32;
    while i < cap_a {
        let mut label = 0u32;
        if i < n_atoms {
            let mut degree = 0u32;
            let mut bi = 0u32;
            while bi < n_bonds {
                let triple = bi * 3;
                let a = slot(scratch, sbase, triple);
                let b = slot(scratch, sbase, triple + 1);
                if a == i || b == i {
                    degree += 1;
                }
                bi += 1;
            }
            let ty = atom_type_at(actions, abase, length, atoms_cap, i);
            label = hash_pair(ty, degree);
        }
        put(scratch, sbase, bank_words + i, label);
        i += 1;
    }
    let mut round = 0u32;
    while round < 4 {
        let mut read = bank_words;
        let mut write = bank1;
        if round % 2 == 1 {
            read = bank1;
            write = bank_words;
        }
        let mut a = 0u32;
        while a < cap_a {
            let mut next = 0u32;
            if a < n_atoms {
                let old = slot(scratch, sbase, read + a);
                let mut acc = 0u32;
                let mut bi = 0u32;
                while bi < n_bonds {
                    let triple = bi * 3;
                    let x = slot(scratch, sbase, triple);
                    let y = slot(scratch, sbase, triple + 1);
                    let o = slot(scratch, sbase, triple + 2);
                    if x == a && y < n_atoms {
                        acc = acc.wrapping_add(hash_pair(o, slot(scratch, sbase, read + y)));
                    } else if y == a && x < n_atoms {
                        acc = acc.wrapping_add(hash_pair(o, slot(scratch, sbase, read + x)));
                    }
                    bi += 1;
                }
                next = hash_pair(old, acc);
            }
            put(scratch, sbase, write + a, next);
            a += 1;
        }
        round += 1;
    }
    // Intentional wrapping: device `u32` addition wraps, so the host twin
    // spells it with `wrapping_add` (plain `+=` would panic in debug builds
    // on ordinary graphs: two `0xfd5474be` final labels already overflow).
    let mut sum = 0u32;
    let mut f = 0u32;
    while f < n_atoms {
        sum = sum.wrapping_add(slot(scratch, sbase, bank_words + f));
        f += 1;
    }
    let hash = hash_u32_host(sum, n_atoms, n_bonds) & hash_mask;
    if (record as usize) < graph_hash.len() {
        graph_hash[record as usize] = hash;
    }
    hash
}

/// Whether candidate image `cand` fits depth `depth`: atom types (re-decoded
/// on demand, never staged) and refined labels agree, and for every already
/// assigned pair the bond order between them is the same in both graphs, order
/// 0 (no bond) included. Returns 1 when the candidate fits, else 0.
#[allow(clippy::too_many_arguments)]
fn candidate_fits(
    cand: u32,
    depth: u32,
    actions: &[u32],
    abase_k: u32,
    abase_j: u32,
    length_k: u32,
    length_j: u32,
    atoms_cap: u32,
    scratch: &[u32],
    sbase_k: u32,
    sbase_j: u32,
    bank: u32,
    n_bonds: u32,
    stack: &[u32],
    stack_base: u32,
) -> u32 {
    let mut fits = 1u32;
    if atom_type_at(actions, abase_k, length_k, atoms_cap, depth)
        != atom_type_at(actions, abase_j, length_j, atoms_cap, cand)
    {
        fits = 0;
    }
    if slot(scratch, sbase_k, bank + depth) != slot(scratch, sbase_j, bank + cand) {
        fits = 0;
    }
    let mut d = 0u32;
    while d < depth && fits == 1 {
        let earlier = slot(stack, stack_base, d);
        let order_k = bond_between(scratch, sbase_k, n_bonds, depth, d);
        let order_j = bond_between(scratch, sbase_j, n_bonds, cand, earlier);
        if order_k != order_j {
            fits = 0;
        }
        d += 1;
    }
    fits
}

/// The exact-comparison lane helper for one pair.
///
/// Full-buffer form (see the module docs). Returns 0 (different), 1 (equal)
/// or 2 (unresolved: budget spent or an unindexable layout — never a guess).
/// Each tentative assignment costs one unit of `work_max`, counted up from a
/// literal zero so no loop-carried variable starts as a copy of the scalar
/// argument.
#[allow(clippy::too_many_arguments)]
pub fn graph_equal_lane(
    actions: &[u32],
    record_k: u32,
    record_j: u32,
    record_stride: u32,
    steps: u32,
    graph_hash: &[u32],
    scratch: &[u32],
    scratch_stride: u32,
    atoms_cap: u32,
    bonds_cap: u32,
    work_max: u32,
    stack: &mut [u32],
    stack_base: u32,
    stack_stride: u32,
) -> u32 {
    let mut layout_ok: u32 = 1u32;
    // All layout validation BEFORE any address multiplication, by guarded
    // comparisons only (division-based guards, no `usize` addition): an
    // invalid layout returns unresolved (2), never a verdict, and never
    // panics — even for an oversized record index in debug builds. Kernel
    // and twin share this block line for line.
    if atoms_cap == 0 || atoms_cap > ATOMS_MAX {
        layout_ok = 0;
    }
    if bonds_cap > BONDS_MAX {
        layout_ok = 0;
    }
    // `stack_stride == atoms_cap * 3` by division guards (nonzero-literal
    // divisors, so no multiplication of unvalidated inputs runs).
    if stack_stride / 3 != atoms_cap || stack_stride % 3 != 0 {
        layout_ok = 0;
    }
    // `scratch_stride == bonds_cap * 3 + atoms_cap * 2`: the products below
    // run only on validated caps (atoms <= 32, bonds <= 39), so they cannot
    // overflow; anything else already failed above.
    let mut expect_row: u32 = 0u32;
    if layout_ok == 1 {
        expect_row = bonds_cap * 3 + atoms_cap * 2;
    }
    if scratch_stride != expect_row {
        layout_ok = 0;
    }
    // The record must hold the `steps * 4` token words, the `atoms_cap`
    // open-valence words and the 4 trailing words (`record_stride >=
    // steps * 4 + atoms_cap + 4`), by division guards: no multiplication
    // of the unvalidated `steps` runs.
    if record_stride < 4 {
        layout_ok = 0;
    } else if record_stride - 4 < atoms_cap {
        layout_ok = 0;
    } else if steps > (record_stride - 4 - atoms_cap) / 4 {
        layout_ok = 0;
    }
    // Record indices and extents, before any address multiplication: with
    // a nonzero stride, record `r` is fully inside exactly when
    // `r < len / stride` (the division guard for `(r + 1) * stride <= len`
    // without forming the product). A zero stride holds no record. The
    // validated buffer lengths are narrowed to u32 once and every extent
    // below runs in u32 (`usize` appears only inside index expressions);
    // an oversized buffer narrows to nothing and fails the extents below.
    // Kernel and twin share this block line for line (the kernel's wrappers
    // checked the same lengths before any launch, so its narrows are exact).
    let mut lens_ok: u32 = 1u32;
    let mut actions_len: u32 = 0u32;
    if (actions.len() as u64) > (u32::MAX as u64) {
        lens_ok = 0;
    } else {
        actions_len = actions.len() as u32;
    }
    let mut hash_len: u32 = 0u32;
    if (graph_hash.len() as u64) > (u32::MAX as u64) {
        lens_ok = 0;
    } else {
        hash_len = graph_hash.len() as u32;
    }
    let mut scratch_len: u32 = 0u32;
    if (scratch.len() as u64) > (u32::MAX as u64) {
        lens_ok = 0;
    } else {
        scratch_len = scratch.len() as u32;
    }
    let mut stack_len: u32 = 0u32;
    if (stack.len() as u64) > (u32::MAX as u64) {
        lens_ok = 0;
    } else {
        stack_len = stack.len() as u32;
    }
    if lens_ok == 0 {
        layout_ok = 0;
    }
    if record_stride == 0 {
        layout_ok = 0;
    } else if record_k >= actions_len / record_stride {
        layout_ok = 0;
    } else if record_j >= actions_len / record_stride {
        layout_ok = 0;
    }
    if record_k >= hash_len {
        layout_ok = 0;
    }
    if record_j >= hash_len {
        layout_ok = 0;
    }
    if scratch_stride == 0 {
        layout_ok = 0;
    } else if record_k >= scratch_len / scratch_stride {
        layout_ok = 0;
    } else if record_j >= scratch_len / scratch_stride {
        layout_ok = 0;
    }
    // `stack_base + stack_stride <= stack.len()` without forming the sum:
    // an out-of-range base is invalid, else the stride must fit the tail.
    if stack_base > stack_len {
        layout_ok = 0;
    } else if stack_stride > stack_len - stack_base {
        layout_ok = 0;
    }
    // Addresses form only on a valid layout (every product exact here),
    // else stay zero: every read below is guarded, and the verdict is
    // unresolved.
    let mut abase_k: u32 = 0u32;
    let mut abase_j: u32 = 0u32;
    let mut sbase_k: u32 = 0u32;
    let mut sbase_j: u32 = 0u32;
    let mut len_field: u32 = 0u32;
    if layout_ok == 1 {
        abase_k = record_k * record_stride;
        abase_j = record_j * record_stride;
        sbase_k = record_k * scratch_stride;
        sbase_j = record_j * scratch_stride;
        len_field = steps * 4 + atoms_cap;
    }
    let length_k = token_word(actions, abase_k, len_field, 0);
    let length_j = token_word(actions, abase_j, len_field, 0);
    let hash_k = slot(graph_hash, 0, record_k);
    let hash_j = slot(graph_hash, 0, record_j);
    let (atoms_k, bonds_k) = decode_counts(actions, abase_k, length_k, atoms_cap, bonds_cap);
    let (atoms_j, bonds_j) = decode_counts(actions, abase_j, length_j, atoms_cap, bonds_cap);
    let mut counts_ok = 1u32;
    if atoms_k != atoms_j || bonds_k != bonds_j {
        counts_ok = 0;
    }
    let mut hash_ok = 1u32;
    if hash_k != hash_j {
        hash_ok = 0;
    }
    let result: u32;
    if layout_ok == 0 {
        result = 2;
    } else if counts_ok == 0 || hash_ok == 0 {
        result = 0;
    } else if atoms_k == 0 {
        result = 1;
    } else {
        let n = atoms_k;
        let nb = bonds_k;
        let cap_a = atoms_cap;
        let bank = bonds_cap * 3;
        let two_a = cap_a * 2;
        let mut w = 0u32;
        while w < stack_stride {
            put(stack, stack_base, w, 0);
            w += 1;
        }
        let mut d = 0u32;
        while d < cap_a {
            put(stack, stack_base, d, u32::MAX);
            d += 1;
        }
        let mut depth = 0u32;
        let mut used = 0u32;
        let mut done = 0u32;
        let mut found = 0u32;
        let mut spent = 0u32;
        while done == 0 {
            if depth == n {
                found = 1;
                done = 1;
            } else {
                let mut cand = slot(stack, stack_base, two_a + depth);
                let mut placed = 0u32;
                while cand < n && placed == 0 && spent == 0 {
                    let mut viable = 0u32;
                    if slot(stack, stack_base, cap_a + cand) == 0 {
                        viable = candidate_fits(
                            cand,
                            depth,
                            actions,
                            abase_k,
                            abase_j,
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
                            put(stack, stack_base, depth, cand);
                            put(stack, stack_base, cap_a + cand, 1);
                            put(stack, stack_base, two_a + depth, cand + 1);
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
                    put(stack, stack_base, depth, u32::MAX);
                    put(stack, stack_base, two_a + depth, 0);
                    if depth == 0 {
                        done = 1;
                    } else {
                        depth -= 1;
                        let prev = slot(stack, stack_base, depth);
                        put(stack, stack_base, cap_a + prev, 0);
                        put(stack, stack_base, depth, u32::MAX);
                    }
                }
            }
        }
        if spent == 1 {
            result = 2;
        } else if found == 1 {
            result = 1;
        } else {
            result = 0;
        }
    }
    result
}

/// Whether a candidate status word names a finished-and-valid trajectory.
fn finished_valid(status: u32) -> bool {
    status & candidate_status::FINISHED != 0 && status & candidate_status::INVALID_FINAL == 0
}

/// The per-trajectory identity lane for one later trajectory.
///
/// Full-buffer form over the kernel's bound buffers. Compares `record` with
/// every earlier finished-and-valid trajectory `j < k` of the same spectrum
/// with an equal graph hash through [`graph_equal_lane`]. Returns
/// `(bits, resolution)` as `u32` words; ineligible records return `(0, 1)`.
#[allow(clippy::too_many_arguments)]
pub fn identity_lane(
    actions: &[u32],
    record: u32,
    record_stride: u32,
    steps: u32,
    per_spectrum: u32,
    atoms_cap: u32,
    bonds_cap: u32,
    work_max: u32,
    graph_hash: &[u32],
    scratch: &[u32],
    scratch_stride: u32,
    stack: &mut [u32],
    stack_base: u32,
    stack_stride: u32,
) -> (u32, u32) {
    let mut bits = 0u32;
    let mut decided = 1u32;
    let mut eligible_k = 0u32;
    let len_field = steps * 4 + atoms_cap;
    let abase_k = record * record_stride;
    let status_k = token_word(actions, abase_k, len_field + 1, 0);
    if finished_valid(status_k) {
        eligible_k = 1;
    }
    if eligible_k == 1 && per_spectrum > 0 {
        let b = record / per_spectrum;
        let k = record % per_spectrum;
        let base = b * per_spectrum;
        let mut j = 0u32;
        while j < k {
            let prev = base + j;
            let abase_j = prev * record_stride;
            let status_j = token_word(actions, abase_j, len_field + 1, 0);
            let mut eligible_j = 0u32;
            if finished_valid(status_j) {
                eligible_j = 1;
            }
            if eligible_j == 1 {
                let hash_k = slot(graph_hash, 0, record);
                let hash_j = slot(graph_hash, 0, prev);
                if hash_k == hash_j {
                    let verdict = graph_equal_lane(
                        actions,
                        record,
                        prev,
                        record_stride,
                        steps,
                        graph_hash,
                        scratch,
                        scratch_stride,
                        atoms_cap,
                        bonds_cap,
                        work_max,
                        stack,
                        stack_base,
                        stack_stride,
                    );
                    if verdict == 1 {
                        bits |= DUPLICATE_GRAPH;
                    } else if verdict == 2 {
                        bits |= IDENTITY_UNRESOLVED;
                        decided = 0;
                    }
                }
            }
            j += 1;
        }
    }
    let mut resolution = 1u32;
    if decided == 0 {
        resolution = 2;
    }
    if eligible_k == 0 {
        bits = 0;
        resolution = 1;
    }
    (bits, resolution)
}

/// Mirror of the lane's trace parsing for the host wrapper (host-only).
#[allow(dead_code)]
fn decode_trace(
    actions: &[u32],
    length: u32,
    atoms_cap: u32,
    closures_cap: u32,
) -> (Vec<u32>, u32, u32) {
    let add_kind = u32::from(ADD_ATOM);
    let close_kind = u32::from(CLOSE_RING);
    let mut cap_a = atoms_cap;
    if cap_a == 0 {
        cap_a = 1;
    }
    let bonds_max = cap_a.saturating_sub(1).saturating_add(closures_cap);
    let mut types: Vec<u32> = Vec::new();
    let mut n_bonds = 0u32;
    let mut s = 0u32;
    while s < length {
        let base = (s as usize).saturating_mul(4);
        let mut kind = u32::MAX;
        let mut ty = 0u32;
        let mut order = 0u32;
        let mut ptr = 0u32;
        if base.saturating_add(4) <= actions.len() {
            kind = actions[base];
            ty = actions[base.wrapping_add(1)];
            order = actions[base.wrapping_add(2)];
            ptr = actions[base.wrapping_add(3)];
        }
        if kind == add_kind {
            if (types.len() as u32) < cap_a {
                if !types.is_empty()
                    && ptr < types.len() as u32
                    && (1..=3).contains(&order)
                    && n_bonds < bonds_max
                {
                    n_bonds += 1;
                }
                types.push(ty);
            }
        } else if kind == close_kind && !types.is_empty() {
            let newest = types.len() as u32 - 1;
            if ptr < newest && (1..=3).contains(&order) && n_bonds < bonds_max {
                n_bonds += 1;
            }
        }
        s += 1;
    }
    let n_atoms = types.len() as u32;
    (types, n_atoms, n_bonds)
}

/// Per-batch duplicate detection: the host wrapper (NOT a kernel twin).
pub struct IdentityResult {
    /// Graph hash per record (`B * K` entries in record order).
    pub graph_hash: Vec<u32>,
    /// Identity status bits per record (bits 7/8 only, to OR into the
    /// candidate status by the caller).
    pub status_bits: Vec<u32>,
    /// Identity resolution per record (1 exact, 2 unresolved).
    pub resolution: Vec<u8>,
}

/// Per-spectrum duplicate detection over a candidate batch.
pub fn identity_batch(batch: &CandidateBatch, hash_mask: u32, work_max: u32) -> IdentityResult {
    let trajectories = batch.trajectories;
    let steps = batch.max_steps;
    let atoms_cap = batch.max_atoms as u32;
    let closures_cap = batch.max_ring_closures as u32;
    let bonds_cap = atoms_cap.saturating_sub(1).saturating_add(closures_cap);
    let n_records = batch.batch.saturating_mul(trajectories);
    let record_stride = (steps.saturating_mul(4))
        .saturating_add(atoms_cap as usize)
        .saturating_add(4) as u32;
    let scratch_stride = bonds_cap * 3 + atoms_cap * 2;
    let stack_stride = atoms_cap * 3;
    let per_spectrum = trajectories as u32;
    let mut actions_flat = vec![0u32; n_records.saturating_mul(record_stride as usize)];
    let mut r = 0usize;
    while r < n_records {
        let abase = (r as u32) * record_stride;
        let tok_base = r.saturating_mul(steps.saturating_mul(4));
        let tok_end = tok_base
            .saturating_add(steps.saturating_mul(4))
            .min(batch.actions.len());
        let mut w = 0usize;
        while tok_base.saturating_add(w) < tok_end {
            let addr = (abase as usize).saturating_add(w);
            if addr < actions_flat.len() {
                actions_flat[addr] = batch.actions[tok_base.saturating_add(w)];
            }
            w = w.saturating_add(1);
        }
        let length = if r < batch.length.len() {
            batch.length[r]
        } else {
            0
        };
        let status = if r < batch.status.len() {
            batch.status[r]
        } else {
            0
        };
        let formula = if r < batch.formula_row.len() {
            batch.formula_row[r]
        } else {
            u32::MAX
        };
        let len_field = (steps as u32) * 4 + atoms_cap;
        put(&mut actions_flat, abase, len_field, length);
        put(&mut actions_flat, abase, len_field + 1, status);
        put(&mut actions_flat, abase, len_field + 2, 0);
        put(&mut actions_flat, abase, len_field + 3, formula);
        r += 1;
    }
    let mut hashes: Vec<u32> = vec![0; n_records];
    let mut scratches: Vec<u32> = vec![0; n_records.saturating_mul(scratch_stride as usize)];
    let mut rr = 0u32;
    while (rr as usize) < n_records {
        graph_hash_lane(
            &actions_flat,
            rr,
            record_stride,
            steps as u32,
            atoms_cap,
            closures_cap,
            hash_mask,
            &mut hashes,
            &mut scratches,
            scratch_stride,
        );
        rr += 1;
    }
    let mut bits: Vec<u32> = vec![0; n_records];
    let mut resolution: Vec<u8> = vec![RESOLUTION_EXACT; n_records];
    let mut stacks: Vec<u32> = vec![0; n_records.saturating_mul(stack_stride as usize)];
    let mut rec = 0u32;
    while (rec as usize) < n_records {
        let stack_base = rec * stack_stride;
        let (b, res) = identity_lane(
            &actions_flat,
            rec,
            record_stride,
            steps as u32,
            per_spectrum,
            atoms_cap,
            bonds_cap,
            work_max,
            &hashes,
            &scratches,
            scratch_stride,
            &mut stacks,
            stack_base,
            stack_stride,
        );
        bits[rec as usize] = b;
        resolution[rec as usize] = res as u8;
        rec += 1;
    }
    IdentityResult {
        graph_hash: hashes,
        status_bits: bits,
        resolution,
    }
}
