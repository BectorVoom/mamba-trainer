//! Host twins of the MS2 device kernels in [`crate::tensor::ops::ms2`].
//!
//! Plain-slice `f32` functions with the same arithmetic and the same iteration
//! order as the kernels, so the kernel tests compare against these rather
//! than re-deriving the specification. Integer outputs must match exactly;
//! float outputs match up to the usual device rounding.

use crate::tensor::ops::ms2::{
    INTENSITY_FLOOR, META_FEATURES, MS2_WAVELENGTHS, PEAK_FEATURES, PRECURSOR_MARGIN,
};

use super::contract::candidate_status;
use super::grammar::{Limits, Token, TraceState, replay};

/// The host twin of [`crate::tensor::ops::random::hash_u32`], bit-identical
/// to `crate::models::graph::tokenize::hash_u32_host` (that module is not
/// wired into this working copy's `models` tree, so the twin spells the
/// same avalanche locally rather than importing an uncompiled path).
#[inline]
pub fn hash_u32_host(index: u32, seed_lo: u32, seed_hi: u32) -> u32 {
    let mut h = index ^ seed_lo;
    h ^= h >> 16;
    h = h.wrapping_mul(0x7feb352d);
    h ^= h >> 15;
    h = h.wrapping_mul(0x846ca68b);
    h ^= seed_hi;
    h ^= h >> 16;
    h
}

/// A draw in `[0, 1)` from a hash word: the 24-bit value over 2^24, exact in
/// `f32`. The sampler's per-field draw is `unit(hash_u32_host(4 * step +
/// field, base, 0))` with `s = hash_u32_host(seed_hi, seed_lo, 0)`, `key =
/// hash_u32_host(id_lo, s, id_hi)`, `base = hash_u32_host(trajectory, key, 0)`
/// (architecture §3.6).
fn unit(h: u32) -> f32 {
    ((h >> 8) as f32) / 16777216.0
}

/// `2π` in `f32`: the same rounded value the feature kernel receives.
const TWO_PI_F32: f32 = 2.0 * core::f32::consts::PI;

/// Flat outputs of [`peak_select`]: `stats` is `B * 3`, `rank` and `position`
/// are `B * n_raw`, `kept` is `B * n_keep * 3`, `kept_f` is `B * n_keep * 2`
/// and `summary` is `B * 2` elements.
pub struct PeakSelection {
    /// Per-spectrum max, total and retained relative intensity.
    pub stats: Vec<f32>,
    /// Intensity rank among kept peaks, `u32::MAX` when not kept.
    pub rank: Vec<u32>,
    /// m/z order among selected peaks, `u32::MAX` when not selected.
    pub position: Vec<u32>,
    /// Per kept slot: raw index, m/z, reverse index.
    pub kept: Vec<u32>,
    /// Per kept slot: relative intensity, valid flag.
    pub kept_f: Vec<f32>,
    /// Per spectrum: kept count and status bits.
    pub summary: Vec<u32>,
}

/// Transformed intensity: the value itself for scale 0, its square for
/// scale 1 (an overflowing square is infinite, hence ineligible).
fn transformed(x: f32, intensity_scale: u32) -> f32 {
    if intensity_scale == 1 { x * x } else { x }
}

/// Eligibility of a valid slot: non-zero m/z within the precursor bound, and
/// a positive transformed intensity below `FINITE_MAX`: the same range test as
/// the kernels, which is false for NaN, infinity and an overflowed square.
fn eligible(m: u32, t: f32, precursor: u32) -> bool {
    if m == 0 {
        return false;
    }
    if m > precursor && m - precursor > PRECURSOR_MARGIN {
        return false;
    }
    t > 0.0 && t < crate::tensor::ops::ms2::FINITE_MAX
}

/// Largest eligible transformed intensity, 0 when none is eligible.
fn spectrum_max(
    mz: &[u32],
    intensity: &[f32],
    base: usize,
    count: usize,
    precursor: u32,
    scale: u32,
) -> f32 {
    let mut max = 0.0f32;
    for i in 0..count {
        let t = transformed(intensity[base + i], scale);
        if eligible(mz[base + i], t, precursor) && t > max {
            max = t;
        }
    }
    max
}

/// Mirror of [`crate::tensor::ops::ms2::peak_select`]: `mz` and `intensity`
/// are `batch * n_raw` flat, `meta` is `batch * 8` flat.
pub fn peak_select(
    mz: &[u32],
    intensity: &[f32],
    meta: &[u32],
    batch: usize,
    n_raw: usize,
    n_keep: usize,
    intensity_scale: u32,
) -> PeakSelection {
    let mut stats = vec![0.0f32; batch * 3];
    let mut rank = vec![u32::MAX; batch * n_raw];
    let mut position = vec![u32::MAX; batch * n_raw];
    let mut kept = vec![0u32; batch * n_keep * 3];
    let mut kept_f = vec![0.0f32; batch * n_keep * 2];
    let mut summary = vec![0u32; batch * 2];
    for b in 0..batch {
        let base = b * n_raw;
        // Clamp to `n_raw`, like the kernels: invalid metadata must never
        // index past the inputs.
        let mut count = meta[b * 8] as usize;
        if count > n_raw {
            count = n_raw;
        }
        let precursor = meta[b * 8 + 1];
        let max = spectrum_max(mz, intensity, base, count, precursor, intensity_scale);
        let floor = INTENSITY_FLOOR * max;
        let mut total = 0.0f32;
        if max > 0.0 {
            for i in 0..count {
                let t = transformed(intensity[base + i], intensity_scale);
                if eligible(mz[base + i], t, precursor) && t >= floor {
                    total += t / max;
                }
            }
        }
        stats[b * 3] = max;
        stats[b * 3 + 1] = total;
        stats[b * 3 + 2] = 0.0;
        for i in 0..n_raw {
            if i >= count {
                continue;
            }
            let t = transformed(intensity[base + i], intensity_scale);
            let keep = eligible(mz[base + i], t, precursor) && max > 0.0 && t >= floor;
            if !keep {
                continue;
            }
            let mut r = 0u32;
            for j in 0..count {
                let tj = transformed(intensity[base + j], intensity_scale);
                let kept_j = eligible(mz[base + j], tj, precursor) && max > 0.0 && tj >= floor;
                if kept_j && (tj > t || (tj == t && j < i)) {
                    r += 1;
                }
            }
            rank[base + i] = r;
        }
        for i in 0..n_raw {
            if i >= count {
                continue;
            }
            if rank[base + i] as usize >= n_keep {
                continue;
            }
            let m = mz[base + i];
            let mut p = 0u32;
            for j in 0..count {
                if (rank[base + j] as usize) < n_keep {
                    let mj = mz[base + j];
                    if mj < m || (mj == m && j < i) {
                        p += 1;
                    }
                }
            }
            position[base + i] = p;
        }
        let max2 = spectrum_max(mz, intensity, base, count, precursor, intensity_scale);
        let mut len = 0u32;
        for i in 0..count {
            if position[base + i] != u32::MAX {
                len += 1;
            }
        }
        for p in 0..n_keep {
            let slot = (b * n_keep + p) * 3;
            let slot_f = (b * n_keep + p) * 2;
            if (p as u32) < len {
                let found = (0..count)
                    .find(|&i| position[base + i] as usize == p)
                    .expect("a selected position below len is taken");
                let t = transformed(intensity[base + found], intensity_scale);
                kept[slot] = found as u32;
                kept[slot + 1] = mz[base + found];
                kept[slot + 2] = len - 1 - p as u32;
                kept_f[slot_f] = t / max2;
                kept_f[slot_f + 1] = 1.0;
            } else {
                kept[slot] = u32::MAX;
                kept[slot + 1] = 0;
                kept[slot + 2] = p as u32;
                kept_f[slot_f] = 0.0;
                kept_f[slot_f + 1] = 0.0;
            }
        }
        let mut kept_n = 0u32;
        let mut sel = 0.0f32;
        for i in 0..count {
            if rank[base + i] != u32::MAX {
                kept_n += 1;
            }
            if (rank[base + i] as usize) < n_keep && total > 0.0 {
                let t = transformed(intensity[base + i], intensity_scale);
                sel += t / max;
            }
        }
        let mut bits = 0u32;
        if (kept_n as usize) > n_keep {
            bits = 1u32 << 19;
        }
        let mut retained = 0.0f32;
        if total > 0.0 {
            retained = sel / total;
        }
        summary[b * 2] = len;
        summary[b * 2 + 1] = bits;
        stats[b * 3 + 2] = retained;
    }
    PeakSelection {
        stats,
        rank,
        position,
        kept,
        kept_f,
        summary,
    }
}

/// Mirror of [`crate::tensor::ops::ms2::peak_features`]: `kept` is `batch *
/// n_keep * 3` flat, `kept_f` is `batch * n_keep * 2` flat, `meta` is `batch *
/// 8` flat; the output is `batch * n_keep * 71` flat.
pub fn peak_features(
    kept: &[u32],
    kept_f: &[f32],
    meta: &[u32],
    batch: usize,
    n_keep: usize,
) -> Vec<f32> {
    let ln101 = 101.0f32.ln();
    let mut out = vec![0.0f32; batch * n_keep * PEAK_FEATURES];
    for b in 0..batch {
        for p in 0..n_keep {
            let slot = (b * n_keep + p) * 3;
            let slot_f = (b * n_keep + p) * 2;
            let base = (b * n_keep + p) * PEAK_FEATURES;
            if kept[slot] == u32::MAX {
                continue;
            }
            let m = kept[slot + 1];
            let r = kept_f[slot_f];
            let c = meta[b * 8 + 1];
            out[base] = m as f32 * 1e-9;
            out[base + 1] = (c as f32 - m as f32) * 1e-9;
            out[base + 2] = m as f32 / c as f32;
            out[base + 3] = r;
            out[base + 4] = r.sqrt();
            out[base + 5] = (1.0 + 100.0 * r).ln() / ln101;
            let gap = if p == 0 {
                0.0
            } else {
                (m - kept[slot - 3 + 1]) as f32 * 1e-6
            };
            out[base + 6] = (1.0 + gap).ln();
            for k in 0..16 {
                let w = MS2_WAVELENGTHS[k];
                let phase = TWO_PI_F32 * (m % w) as f32 / w as f32;
                out[base + 7 + 2 * k] = phase.sin();
                out[base + 8 + 2 * k] = phase.cos();
            }
            let v = c.abs_diff(m);
            for k in 0..16 {
                let w = MS2_WAVELENGTHS[k];
                let phase = TWO_PI_F32 * (v % w) as f32 / w as f32;
                out[base + 39 + 2 * k] = phase.sin();
                out[base + 40 + 2 * k] = phase.cos();
            }
        }
    }
    out
}

/// Mirror of [`crate::tensor::ops::ms2::select_valid`]: `x` is flat with
/// last dimension `d`, `valid` selects whole rows of it.
pub fn select_valid(x: &[f32], valid: &[f32], d: usize) -> Vec<f32> {
    x.iter()
        .enumerate()
        .map(|(pos, &v)| if valid[pos / d] != 0.0 { v } else { 0.0 })
        .collect()
}

/// Mirror of [`crate::tensor::ops::ms2::bits_to_mask`].
pub fn bits_to_mask(bits: &[u32], width: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; bits.len() * width];
    for (row, &word) in bits.iter().enumerate() {
        for col in 0..width {
            out[row * width + col] = ((word >> col) & 1) as f32;
        }
    }
    out
}

/// Mirror of [`crate::tensor::ops::ms2::lookup`]: `table` is `rows * d`
/// flat, `ids` selects rows, out-of-range ids give zero rows.
pub fn lookup(table: &[f32], d: usize, ids: &[u32]) -> Vec<f32> {
    let table_rows = table.len() / d;
    let mut out = vec![0.0f32; ids.len() * d];
    for (row, &id) in ids.iter().enumerate() {
        if (id as usize) < table_rows {
            for col in 0..d {
                out[row * d + col] = table[(id as usize) * d + col];
            }
        }
    }
    out
}

/// Mirror of [`crate::tensor::ops::ms2::lookup_backward`]: per table row, the
/// index-ordered sum of the gradient rows that select it.
pub fn lookup_backward(grad: &[f32], d: usize, ids: &[u32], table_rows: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; table_rows * d];
    for v in 0..table_rows {
        for col in 0..d {
            let mut acc = 0.0f32;
            for (r, &id) in ids.iter().enumerate() {
                if (id as usize) == v {
                    acc += grad[r * d + col];
                }
            }
            out[v * d + col] = acc;
        }
    }
    out
}

/// Mirror of [`crate::tensor::ops::ms2::meta_features`]: `meta` is `batch * 8`
/// flat, `energy` is `batch * 2` flat holding the collision energy in eV and
/// the known flag; the output is `batch * 34` flat with `min(ce, 400) / 100`
/// when the flag is non-zero, `precursor * 1e-9`, and the 32 Fourier
/// features of the precursor.
pub fn meta_features(meta: &[u32], energy: &[f32], batch: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; batch * META_FEATURES];
    for b in 0..batch {
        let ce = energy[b * 2];
        let known = energy[b * 2 + 1];
        out[b * META_FEATURES] = if known != 0.0 {
            ce.min(400.0) / 100.0
        } else {
            0.0
        };
        let precursor = meta[b * 8 + 1];
        out[b * META_FEATURES + 1] = precursor as f32 * 1e-9;
        for k in 0..16 {
            let w = MS2_WAVELENGTHS[k];
            let phase = TWO_PI_F32 * (precursor % w) as f32 / w as f32;
            out[b * META_FEATURES + 2 + 2 * k] = phase.sin();
            out[b * META_FEATURES + 3 + 2 * k] = phase.cos();
        }
    }
    out
}

/// Mirror of [`crate::tensor::ops::ms2::kept_column`]: one column of the
/// `batch * n_keep * 3` flat `kept` buffer as a flat `batch * n_keep` vector.
pub fn kept_column(kept: &[u32], batch: usize, n_keep: usize, column: usize) -> Vec<u32> {
    let mut out = vec![0u32; batch * n_keep];
    for b in 0..batch {
        for p in 0..n_keep {
            out[b * n_keep + p] = kept[(b * n_keep + p) * 3 + column];
        }
    }
    out
}

/// Mirror of [`crate::tensor::ops::ms2::formula_window`], written by calling
/// [`crate::models::ms2::formula::FormulaTable::window`] per spectrum, so the
/// kernel is compared with the P1 reference itself rather than with a mirror
/// of the kernel.
///
/// `table` is the formula table, `meta` is `batch * 8` flat per-spectrum
/// metadata (precursor at `meta[b, 1]`, uncertainty at `[b, 2]`, adduct at
/// `[b, 3]`, precursor tolerance at `[b, 5]`), `m` the window capacity and
/// `max_error` the table's largest row bound (it must equal
/// `table.max_error()`: the kernel takes it as a scalar while the reference
/// derives it from its rows). Returns the `[B, M, 2]` window flat (row,
/// `u32::MAX` padding; flag 0 none, 1 accept, 2 ambiguous) and the `[B, 5]`
/// counters flat (rows_visited, rows_joined, rows_scored, status bits,
/// complete as 0/1).
pub fn formula_window(
    table: &super::formula::FormulaTable,
    meta: &[u32],
    batch: usize,
    m: usize,
    max_error: u32,
    rows_visited_max: u32,
    rows_scored_max: u32,
) -> (Vec<u32>, Vec<u32>) {
    debug_assert_eq!(
        max_error,
        table.max_error(),
        "the kernel scalar must be the table's own largest row bound"
    );
    let mut window = vec![0u32; batch * m * 2];
    let mut counters = vec![0u32; batch * 5];
    for b in 0..batch {
        let query = super::formula::WindowQuery {
            precursor_mz: meta[b * 8 + 1],
            adduct: meta[b * 8 + 3] as u16,
            ppm_tenths: meta[b * 8 + 5],
            precursor_uncertainty: meta[b * 8 + 2],
            rows_visited_max,
            rows_scored_max: rows_scored_max.min(m as u32),
        };
        let found = table.window(&query);
        // The joined rows fill the first slots in table order; the rest stays
        // empty. `rows_scored` is what the kernel wrote.
        for slot in 0..m {
            let base = (b * m + slot) * 2;
            if slot < found.rows_scored as usize {
                window[base] = found.joined[slot] as u32;
                window[base + 1] = if found.ambiguous[slot] { 2 } else { 1 };
            } else {
                window[base] = u32::MAX;
                window[base + 1] = 0;
            }
        }
        counters[b * 5] = found.rows_visited;
        counters[b * 5 + 1] = found.rows_joined;
        counters[b * 5 + 2] = found.rows_scored;
        counters[b * 5 + 3] = found.status;
        counters[b * 5 + 4] = u32::from(found.complete);
    }
    (window, counters)
}

/// Mirror of [`crate::tensor::ops::ms2::formula_top`]: the `F` largest
/// `log_prob` values among the slots whose flag is non-zero, ties by smaller
/// slot. `log_prob` is `batch * m` flat, `window` the `[B, M, 2]` flat window
/// of [`formula_window`]. Returns the `[B, F, 2]` top flat (table row and
/// window slot, `u32::MAX` padding), the `[B, F]` top log-probabilities flat
/// (0 in padding) and the `[B]` counts flat: the number of entries actually
/// written, compacted densely so every slot below the count is a real row.
///
/// The rank of a slot counts the flagged slots strictly better than it
/// (larger log-probability, or an equal one at a smaller slot), comparing
/// floats with `>`/`==` only, exactly like the kernel.
pub fn formula_top(
    log_prob: &[f32],
    window: &[u32],
    batch: usize,
    m: usize,
    f: usize,
) -> (Vec<u32>, Vec<f32>, Vec<u32>) {
    let mut top = vec![u32::MAX; batch * f * 2];
    let mut top_log_prob = vec![0.0f32; batch * f];
    let mut top_count = vec![0u32; batch];
    for b in 0..batch {
        let mut joined = 0u32;
        for slot in 0..m {
            if window[(b * m + slot) * 2 + 1] != 0 {
                joined += 1;
            }
        }
        let count = joined.min(f as u32);
        let mut written = 0u32;
        for pick in 0..f {
            if (pick as u32) < count {
                // The slot of rank `pick`: like the kernel, only a rank
                // produced exactly once is written, so a NaN or tie anomaly
                // leaves no entry for that pick instead of indexing out of
                // bounds; written entries compact densely.
                let mut best = u32::MAX;
                let mut found = false;
                for slot in 0..m {
                    if window[(b * m + slot) * 2 + 1] == 0 {
                        continue;
                    }
                    let mut rank = 0u32;
                    for other in 0..m {
                        if window[(b * m + other) * 2 + 1] == 0 {
                            continue;
                        }
                        let here = log_prob[b * m + slot];
                        let there = log_prob[b * m + other];
                        if there > here || (there == here && other < slot) {
                            rank += 1;
                        }
                    }
                    if rank == pick as u32 {
                        best = slot as u32;
                        found = true;
                        break;
                    }
                }
                if found {
                    let dest = (b * f + written as usize) * 2;
                    top[dest] = window[(b * m + best as usize) * 2];
                    top[dest + 1] = best;
                    top_log_prob[b * f + written as usize] = log_prob[b * m + best as usize];
                    written += 1;
                }
            }
        }
        top_count[b] = written;
    }
    (top, top_log_prob, top_count)
}

/// The 18 atom-type rows as one `[18, 3]` flat buffer holding
/// `(element, hydrogens, valence)` in [`crate::models::ms2::chem`] order; row
/// 0 is unused (all zeros). The host copy of the resident device table the
/// sampler kernels read.
pub fn atom_table_rows() -> Vec<u32> {
    let mut types = vec![0u32; 18 * 3];
    for (i, t) in super::chem::ATOM_TYPES.iter().enumerate() {
        types[(i + 1) * 3] = t.element as u32;
        types[(i + 1) * 3 + 1] = u32::from(t.hydrogens);
        types[(i + 1) * 3 + 2] = u32::from(t.valence);
    }
    types
}

/// Host mirror of the sampler's `apply_token`: apply one token onto a state
/// row of `3 * a + 16` words (the layout of
/// [`crate::tensor::ops::ms2::grammar_replay`]), with the exact word updates
/// the device helper performs. The caller checks legality first.
pub fn apply_token_row(
    state: &mut [u32],
    a: usize,
    atom_table: &[u32],
    kind: u32,
    ty: u32,
    b: u32,
    p: u32,
) {
    if kind == 2 {
        let elem = atom_table[ty as usize * 3] as usize;
        let hyd = atom_table[ty as usize * 3 + 1];
        let val = atom_table[ty as usize * 3 + 2];
        let n = state[3 * a] as usize;
        state[3 * a + 6 + elem] += 1;
        state[3 * a + 6 + 1] += hyd;
        state[n] = ty;
        state[a + n] = val - hyd;
        if state[3 * a + 4] == 1 {
            state[2 * a + n] = u32::MAX;
        } else {
            let q = p as usize;
            state[a + q] -= b;
            state[a + n] -= b;
            state[2 * a + n] = p;
            state[3 * a + 1] = p;
        }
        state[3 * a + 2] = 0;
        state[3 * a] += 1;
    }
    if kind == 3 {
        let n = state[3 * a] as usize;
        let newest = n - 1;
        let q = p as usize;
        state[a + q] -= b;
        state[a + newest] -= b;
        state[3 * a + 2] = p + 1;
        state[3 * a + 3] += 1;
    }
    if kind == 4 {
        state[3 * a + 5] = 1;
    }
    state[3 * a + 4] += 1;
}

/// One sampled field with the exact operation order of the sampler kernel:
/// the maximum over the legal set, the temperature-scaled normaliser, the
/// increasing-order inverse-CDF pick (first cumulative probability above `u`,
/// last legal index on rounding), and the log-softmax at temperature 1.
/// `logits` holds the field's values at `base..base + width`; `mask` has bit
/// `i` for legal index `i`.
fn sample_field(
    logits: &[f32],
    base: usize,
    width: usize,
    mask: u32,
    u: f32,
    temp: f32,
) -> (u32, f32) {
    let at = |i: usize| logits[base + i];
    let mut first = true;
    let mut m = 0.0f32;
    for i in 0..width {
        if mask & (1 << i) != 0 {
            if first {
                m = at(i);
                first = false;
            } else if at(i) > m {
                m = at(i);
            }
        }
    }
    let mut sum = 0.0f32;
    for i in 0..width {
        if mask & (1 << i) != 0 {
            sum += ((at(i) - m) / temp).exp();
        }
    }
    let mut cum = 0.0f32;
    let mut pick = 0u32;
    let mut last = 0u32;
    let mut done = false;
    for i in 0..width {
        if mask & (1 << i) != 0 {
            last = i as u32;
            if !done {
                cum += ((at(i) - m) / temp).exp() / sum;
                if cum > u {
                    pick = i as u32;
                    done = true;
                }
            }
        }
    }
    if !done {
        pick = last;
    }
    let mut first1 = true;
    let mut m1 = 0.0f32;
    for i in 0..width {
        if mask & (1 << i) != 0 {
            if first1 {
                m1 = at(i);
                first1 = false;
            } else if at(i) > m1 {
                m1 = at(i);
            }
        }
    }
    let mut sum1 = 0.0f32;
    for i in 0..width {
        if mask & (1 << i) != 0 {
            sum1 += (at(i) - m1).exp();
        }
    }
    (pick, (at(pick as usize) - m1) - sum1.ln())
}

/// What [`sample_step`] drew: the token, the four per-field draws (0.0 for a
/// field the kind does not use) and the legality masks the draws ran on.
pub struct SampleDraw {
    /// `(kind, atom_type, bond_order, pointer)`.
    pub token: [u32; 4],
    /// The `unit(hash_u32(4 * step + field, base, 0))` draws for kind, atom
    /// type, bond and pointer (0.0 for an unused field).
    pub u: [f32; 4],
    /// Legal kinds.
    pub kinds: u32,
    /// Legal atom types for the sampled kind.
    pub types: u32,
    /// Legal bond orders for the sampled kind and type.
    pub bonds: u32,
    /// Legal pointers for the sampled kind, type and bond.
    pub pointers: u32,
}

/// Mirror of [`crate::tensor::ops::ms2::sample_step`] for one trajectory:
/// `logits` is one packed row (the layout of
/// [`crate::tensor::ops::ms2::sample_logits_offsets`]), `tables` the
/// `bond_by_type [19, 4]` rows flat, `traj` the 14-word metadata row, `state`
/// the `3 * a + 16` grammar row (in/out), `actions` the `t * 4 + a + 4`
/// record (in/out, trace-log-probability as `f32` bits in the spare word).
/// Same legality (via [`TraceState`]), same draws, same `f32` operation
/// order; finished, failed and not-started rows are untouched.
#[allow(clippy::too_many_arguments)]
pub fn sample_step(
    logits: &[f32],
    tables: &[f32],
    traj: &[u32],
    state: &mut [u32],
    actions: &mut [u32],
    step: u32,
    seed_lo: u32,
    seed_hi: u32,
    temperature: f32,
    t: usize,
    a: usize,
    rmax: usize,
    atom_table: &[u32],
) -> SampleDraw {
    let len_off = t * 4 + a;
    let st_off = len_off + 1;
    let lp_off = len_off + 2;
    let started = traj[3];
    let st = actions[st_off];
    let absorbing = started == 0 || st & (1 | 2 | 4 | 8 | 16 | 64) != 0;
    // The draws are pure functions of the inputs, so they are reported even
    // for an absorbing row (the kernel returns before drawing).
    let s = hash_u32_host(seed_hi, seed_lo, 0);
    let key = hash_u32_host(traj[0], s, traj[1]);
    let draw_base = hash_u32_host(traj[2], key, 0);
    let mut u = [0.0f32; 4];
    for (f, slot) in u.iter_mut().enumerate() {
        *slot = unit(hash_u32_host(step * 4 + f as u32, draw_base, 0));
    }
    // Replay the emitted prefix to recover the grammar legality state.
    let limits = Limits::new(a, rmax).expect("twin limits fit");
    let mut budget = [0u16; 10];
    for e in 0..10 {
        budget[e] = traj[4 + e] as u16;
    }
    let len = actions[len_off] as usize;
    let mut tokens = Vec::with_capacity(len);
    for i in 0..len {
        tokens.push(Token {
            kind: actions[i * 4] as u8,
            atom_type: actions[i * 4 + 1] as u8,
            bond: actions[i * 4 + 2] as u8,
            pointer: actions[i * 4 + 3] as u8,
        });
    }
    let gram = replay(&tokens, limits, Some(budget)).expect("twin replays a legal prefix");
    let blank = Token {
        kind: 0,
        atom_type: 0,
        bond: 0,
        pointer: 0,
    };
    let kinds = gram.masks(blank).kinds;
    let mut draw = SampleDraw {
        token: [0, 0, 0, 0],
        u,
        kinds,
        types: 0,
        bonds: 0,
        pointers: 0,
    };
    if absorbing || kinds == 0 {
        if !absorbing {
            actions[st_off] = st | candidate_status::NO_VALID_ACTION;
        }
        return draw;
    }
    let (k, kind_lp) = sample_field(logits, 0, 5, kinds, u[0], temperature);
    let mut lp = kind_lp;
    let (mut ty, mut b, mut p) = (0u32, 0u32, 0u32);
    if k == 2 {
        if gram.step() == 1 {
            let types = gram.masks(Token { kind: 2, ..blank }).atom_types;
            draw.types = types;
            let (id, id_lp) = sample_field(logits, 5, 18, types, u[1], temperature);
            ty = id;
            lp += id_lp;
        } else {
            let mut types = 0u32;
            for id in 1..=17u32 {
                let probe = gram.masks(Token {
                    kind: 2,
                    atom_type: id as u8,
                    ..blank
                });
                if probe.atom_types & (1 << id) != 0 {
                    types |= 1 << id;
                }
            }
            draw.types = types;
            let (id, id_lp) = sample_field(logits, 5, 18, types, u[1], temperature);
            ty = id;
            let c = ty;
            lp += id_lp;
            let bonds = gram
                .masks(Token {
                    kind: 2,
                    atom_type: ty as u8,
                    ..blank
                })
                .bonds;
            draw.bonds = bonds;
            let mut bond_logits = [0.0f32; 4];
            for i in 0..4 {
                bond_logits[i] = logits[23 + i] + tables[c as usize * 4 + i];
            }
            let (bd, bd_lp) = sample_field(&bond_logits, 0, 4, bonds, u[2], temperature);
            b = bd;
            lp += bd_lp;
            let pointers = gram
                .masks(Token {
                    kind: 2,
                    atom_type: ty as u8,
                    bond: b as u8,
                    ..blank
                })
                .pointers;
            draw.pointers = pointers;
            let mut ptr_logits = vec![0.0f32; a];
            for j in 0..a {
                ptr_logits[j] = logits[27 + j]
                    + logits[27 + a + c as usize * a + j]
                    + logits[27 + a + 19 * a + b as usize * a + j];
            }
            let (pt, pt_lp) = sample_field(&ptr_logits, 0, a, pointers, u[3], temperature);
            p = pt;
            lp += pt_lp;
        }
    } else if k == 3 {
        let c = 18;
        let bonds = gram.masks(Token { kind: 3, ..blank }).bonds;
        draw.bonds = bonds;
        let mut bond_logits = [0.0f32; 4];
        for i in 0..4 {
            bond_logits[i] = logits[23 + i] + tables[c as usize * 4 + i];
        }
        let (bd, bd_lp) = sample_field(&bond_logits, 0, 4, bonds, u[2], temperature);
        b = bd;
        lp += bd_lp;
        let pointers = gram
            .masks(Token {
                kind: 3,
                bond: b as u8,
                ..blank
            })
            .pointers;
        draw.pointers = pointers;
        let mut ptr_logits = vec![0.0f32; a];
        for j in 0..a {
            ptr_logits[j] = logits[27 + j]
                + logits[27 + a + c as usize * a + j]
                + logits[27 + a + 19 * a + b as usize * a + j];
        }
        let (pt, pt_lp) = sample_field(&ptr_logits, 0, a, pointers, u[3], temperature);
        p = pt;
        lp += pt_lp;
    }
    draw.token = [k, ty, b, p];
    // Zero the draws of fields the kind does not use, matching what the
    // kernel consumed.
    let used = match k {
        2 if gram.step() == 1 => [true, true, false, false],
        2 => [true, true, true, true],
        3 => [true, false, true, true],
        _ => [true, false, false, false],
    };
    for (f, keep) in used.iter().enumerate() {
        if !keep {
            draw.u[f] = 0.0;
        }
    }
    apply_token_row(state, a, atom_table, k, ty, b, p);
    actions[len * 4..len * 4 + 4].copy_from_slice(&draw.token);
    actions[len_off] = len as u32 + 1;
    let old = f32::from_bits(actions[lp_off]);
    actions[lp_off] = (old + lp).to_bits();
    if k == 4 {
        actions[st_off] = st | candidate_status::FINISHED;
        let n = state[3 * a] as usize;
        for j in 0..a {
            actions[t * 4 + j] = if j < n { state[a + j] } else { 0 };
        }
    } else if step == t as u32 - 1 {
        actions[st_off] = st | candidate_status::TRUNCATED;
    }
    draw
}

/// Mirror of [`crate::tensor::ops::ms2::validate_trajectories`]: replay every
/// started trajectory of the flat `actions`/`traj` buffers from scratch,
/// setting `invalid_final` on an illegal step or a failed final rule (at
/// least one atom, caps, budget), then flagging exact trace-and-formula
/// duplicates of earlier trajectories of the same spectrum.
pub fn validate(
    actions: &mut [u32],
    traj: &[u32],
    spectra: usize,
    per_spectrum: usize,
    t: usize,
    a: usize,
    rmax: usize,
    atom_table: &[u32],
) {
    let width = t * 4 + a + 4;
    let limits = Limits::new(a, rmax).expect("twin limits fit");
    for r in 0..spectra * per_spectrum {
        let base = r * width;
        let len_off = base + t * 4 + a;
        let st_off = len_off + 1;
        let form_off = len_off + 3;
        let st = actions[st_off];
        if st & candidate_status::REQUEST_FAILED != 0 {
            continue;
        }
        let mut budget = [0u16; 10];
        for e in 0..10 {
            budget[e] = traj[r * 14 + 4 + e] as u16;
        }
        let len = actions[len_off] as usize;
        let mut tst = TraceState::new(limits, Some(budget));
        let mut used = [0u16; 10];
        let mut closes = 0u32;
        let mut bad = false;
        for i in 0..len {
            let tok = Token {
                kind: actions[base + i * 4] as u8,
                atom_type: actions[base + i * 4 + 1] as u8,
                bond: actions[base + i * 4 + 2] as u8,
                pointer: actions[base + i * 4 + 3] as u8,
            };
            if tst.is_legal(tok) {
                if tok.kind == 2 {
                    let ty = tok.atom_type as usize;
                    let elem = atom_table[ty * 3] as usize;
                    let hyd = atom_table[ty * 3 + 1];
                    used[elem] += 1;
                    used[1] += hyd as u16;
                }
                if tok.kind == 3 {
                    closes += 1;
                }
                tst.apply(tok).expect("twin applies a legal token");
            } else {
                bad = true;
                break;
            }
        }
        if tst.atoms() == 0 {
            bad = true;
        }
        if tst.atoms() > a || closes > rmax as u32 {
            bad = true;
        }
        for e in 0..10 {
            if used[e] > budget[e] {
                bad = true;
            }
        }
        let b = r / per_spectrum;
        let k = r % per_spectrum;
        let len_rel = t * 4 + a;
        let st_rel = len_rel + 1;
        let form_rel = len_rel + 3;
        let mut dup = false;
        for k2 in 0..k {
            let r2 = b * per_spectrum + k2;
            let b2 = r2 * width;
            if actions[b2 + st_rel] & candidate_status::REQUEST_FAILED != 0 {
                continue;
            }
            if actions[b2 + form_rel] == actions[form_off]
                && actions[b2 + len_rel] == actions[len_off]
                && actions[b2..b2 + t * 4] == actions[base..base + t * 4]
            {
                dup = true;
                break;
            }
        }
        if bad {
            actions[st_off] |= candidate_status::INVALID_FINAL;
        }
        if dup {
            actions[st_off] |= candidate_status::DUPLICATE_TRACE;
        }
    }
}
