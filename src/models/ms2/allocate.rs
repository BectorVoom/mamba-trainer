//! Trajectory allocation host twin of `docs/MS2_V1_ARCHITECTURE.md` §3.2 (plan
//! item P5.5, host side).
//!
//! Pure host Rust with no tensors and no kernels. The `#[cube]` kernel
//! `ms2_allocate` copies [`allocate_lane`] line for line; [`allocate_checked`]
//! is the host wrapper around that lane and is NOT copied to the kernel.
//!
//! Kernel-expressible form (CubeCL 0.10 cannot express Rust fixed-size local
//! arrays nor `wrapping_*` methods, so the lane uses neither): `u32` counters
//! with `usize` only to bound a slice or index it, `while` loops, plain
//! `u32` arithmetic (wrapping in release and on device), scalar capacities
//! passed explicitly, full bound buffers with explicit spectrum indices and
//! base offsets (no record-local slices), per-slot shares as individual
//! scalars `f0..f7`/`e0..e7`/`d0..d7` under the spec caps (`F <= 8`,
//! `K <= 64`), a single exit and nesting at most 5 deep. The lane also uses
//! `f32`, whose operation order is fixed below so a kernel copies it bit for
//! bit on one backend.
//!
//! Output layout: `traj_formula` flat `[B, K, 12]` with slot, source id, 10
//! counts; sentinel slot/source and zero counts when `top_count == 0`.

use crate::error::{Error, Result};

/// Allocation mode 0: round robin, trajectory `k` uses formula
/// `k mod top_count` (the V0 rule).
pub const ALLOC_ROUND_ROBIN: u32 = 0;
/// Allocation mode 1: proportional to the renormalised retained probabilities.
pub const ALLOC_PROPORTIONAL: u32 = 1;
/// Largest trajectory count a lane handles (spec §3.2: `K <= 64`).
pub const ALLOC_K_MAX: u32 = 64;
/// Largest retained-formula count a lane handles (spec §3.2: `F <= 8`).
pub const ALLOC_F_MAX: u32 = 8;
/// Words per trajectory in `traj_formula`: slot, source id, 10 counts.
pub const ALLOC_RECORD: u32 = 12;
/// Finite-range bound for the kernel-expressible outside-validated-domain fallback.
///
/// An alias of the one shared bound ([`crate::tensor::ops::ms2::FINITE_MAX`]):
/// an open-interval range test, because fast-math backends (Metal compiles
/// WGSL with fast-math) fold `t - t == 0` and `t != t` to constants, so an
/// exact NaN/infinity test is not expressible in the kernel subset. A value
/// is "in the validated domain" when strictly inside (−3e38, 3e38); anything
/// else (NaN, infinities, finite extremes at or beyond it) takes the
/// round-robin fallback together with overflowing sums.
pub const ALLOC_FINITE_MAX: f32 = crate::tensor::ops::ms2::FINITE_MAX;

/// Whether `v` counts as finite under the kernel's range test: 1 finite, else 0.
///
/// Finite means strictly inside `(-ALLOC_FINITE_MAX, ALLOC_FINITE_MAX)`; the
/// validated input domain of `top_log_prob` is that open interval, and
/// entries with `|v| >= 3e38` fall back to round robin (see
/// [`ALLOC_FINITE_MAX`]).
fn finite_flag(v: f32) -> u32 {
    let mut out = 0u32;
    if v > -ALLOC_FINITE_MAX && v < ALLOC_FINITE_MAX {
        out = 1;
    }
    out
}

/// One guarded `u32` read from a flat buffer.
fn word_at(buf: &[u32], base: u32, idx: u32, or: u32) -> u32 {
    let addr = base + idx;
    let mut out = or;
    if (addr as usize) < buf.len() {
        out = buf[addr as usize];
    }
    out
}

/// One guarded `f32` read from a flat buffer.
fn lp_at(buf: &[f32], base: u32, idx: u32, or: f32) -> f32 {
    let addr = base + idx;
    let mut out = or;
    if (addr as usize) < buf.len() {
        out = buf[addr as usize];
    }
    out
}

/// One guarded 12-word record write with 10 scalar counts.
#[allow(clippy::too_many_arguments)]
fn put_record(
    out: &mut [u32],
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
    let row = base + t * ALLOC_RECORD;
    if (row as usize) < out.len() {
        out[row as usize] = slot_v;
    }
    if ((row + 1) as usize) < out.len() {
        out[(row + 1) as usize] = source;
    }
    // Unrolled count stores, each guarded.
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

/// Read the 10 counts of slot `s` as scalars.
fn read_counts(top_counts: &[u32], counts_base: u32, s: u32) -> (u32, u32, u32, u32, u32, u32, u32, u32, u32, u32) {
    let b = s * 10;
    (
        word_at(top_counts, counts_base, b, 0),
        word_at(top_counts, counts_base, b + 1, 0),
        word_at(top_counts, counts_base, b + 2, 0),
        word_at(top_counts, counts_base, b + 3, 0),
        word_at(top_counts, counts_base, b + 4, 0),
        word_at(top_counts, counts_base, b + 5, 0),
        word_at(top_counts, counts_base, b + 6, 0),
        word_at(top_counts, counts_base, b + 7, 0),
        word_at(top_counts, counts_base, b + 8, 0),
        word_at(top_counts, counts_base, b + 9, 0),
    )
}

/// Reconcile a floor-sum overshoot before any unsigned subtraction.
///
/// `floors` holds the per-slot `floor(K' p_f)` extra shares of the first `n`
/// slots and `rest_k` the remaining trajectories to share. An `f32` rounding
/// artefact can push the floor sum past `rest_k`: the excess is shaved in slot
/// order and the remaining picks are reported (0 after a shave). Host-only
/// helper with the same slot-order shave the lane inlines with scalars;
/// tested directly.
///
/// Bounds (review finding 6): the lane runs with `n <= 8` (`F <= 8`) and
/// `rest_k <= 64` (`K <= 64`). The helper enforces `n <= 8` by clamping and
/// accumulates the floor sum in `u64`, so neither the `n = 9` indexing past
/// the fixed array nor the `sum = 2^32` wrap (`n = 2`,
/// `rest_k = u32::MAX`, floors `[2_147_483_648, 2_147_483_648]`) can panic
/// or wrap: the adversarial sum shaves to `rest_k` and reports 0 remaining
/// picks.
pub fn reconcile_extra_shares(floors: &mut [u32; 8], n: u32, rest_k: u32) -> u32 {
    let mut slots = n;
    if slots > 8 {
        slots = 8;
    }
    let mut sum: u64 = 0;
    let mut i = 0u32;
    while i < slots {
        sum += floors[i as usize] as u64;
        i += 1;
    }
    let rest = rest_k as u64;
    if sum > rest {
        let mut excess = sum - rest;
        let mut s = 0u32;
        while s < slots && excess > 0 {
            let held = floors[s as usize] as u64;
            let mut take = held;
            if take > excess {
                take = excess;
            }
            floors[s as usize] = (held - take) as u32;
            excess -= take;
            s += 1;
        }
        return 0;
    }
    (rest - sum) as u32
}

/// One spectrum's trajectory-to-formula assignment: the kernel twin of
/// `ms2_allocate`.
///
/// Full-buffer form over the kernel's 5 bound arrays (see the module docs).
/// Exactly the lane's `K` 12-word records are written. Contract violations are
/// a guarded sentinel no-op; [`allocate_checked`] refuses them.
///
/// Round robin writes `t mod top_count`. Proportional gives every retained
/// formula one trajectory first (first `K` one each when `K <= top_count`),
/// shares `K' = K - top_count` as `floor(K' * p_f)` with largest-remainder
/// picks (ties by smaller slot), and fills trajectories in slot order. A
/// retained log-probability outside the validated domain (range test) falls
/// back to round robin:
/// finite means strictly inside `(-3e38, 3e38)` (the `ms2.rs` `FINITE_MAX`
/// rule, required by fast-math backends), so the validated input domain of
/// `top_log_prob` is that open interval and entries with `|v| >= 3e38` fall
/// back together with NaN and ±infinity.
///
/// Fixed `f32` order: max by sequential scan (strict `>`); `e_f` per slot;
/// `sum` from `0.0` in slot order; `p_f = e_f / sum`; `quota_f = K' * p_f`;
/// `floor_f`; `frac_f`; scalar slot-order reconcile; largest-remainder picks
/// by sequential scan per pick (strict `>`).
#[allow(clippy::too_many_arguments)]
pub fn allocate_lane(
    top: &[u32],
    top_counts: &[u32],
    top_log_prob: &[f32],
    top_count: &[u32],
    spectrum: u32,
    f: u32,
    k: u32,
    mode: u32,
    out: &mut [u32],
) {
    let top_base = spectrum * (f * 2);
    let counts_base = spectrum * (f * 10);
    let lp_base = spectrum * f;
    let out_base = spectrum * (k * ALLOC_RECORD);
    let mut n = 0u32;
    if (spectrum as usize) < top_count.len() {
        n = top_count[spectrum as usize];
    }
    if n > f {
        n = f;
    }
    let mut ok = 1u32;
    if k > ALLOC_K_MAX {
        ok = 0;
    }
    if f > ALLOC_F_MAX {
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
    if (out.len() as u32) < (spectrum + 1) * (k * ALLOC_RECORD) {
        ok = 0;
    }
    if ok == 0 || n == 0 {
        let mut t = 0u32;
        while t < k {
            put_record(out, out_base, t, u32::MAX, u32::MAX, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
            t += 1;
        }
    } else if mode == ALLOC_ROUND_ROBIN {
        let mut t = 0u32;
        while t < k {
            let s = t % n;
            let source = word_at(top, top_base, s * 2, u32::MAX);
            let (c0, c1, c2, c3, c4, c5, c6, c7, c8, c9) = read_counts(top_counts, counts_base, s);
            put_record(out, out_base, t, s, source, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9);
            t += 1;
        }
    } else if k <= n {
        let mut t = 0u32;
        while t < k {
            let s = t;
            let source = word_at(top, top_base, s * 2, u32::MAX);
            let (c0, c1, c2, c3, c4, c5, c6, c7, c8, c9) = read_counts(top_counts, counts_base, s);
            put_record(out, out_base, t, s, source, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9);
            t += 1;
        }
    } else {
        let mut finite = 1u32;
        let mut fchk = 0u32;
        while fchk < n {
            let v = lp_at(top_log_prob, lp_base, fchk, 0.0);
            if finite_flag(v) == 0 {
                finite = 0;
            }
            fchk += 1;
        }
        if finite == 0 {
            let mut t = 0u32;
            while t < k {
                let s = t % n;
                let source = word_at(top, top_base, s * 2, u32::MAX);
                let (c0, c1, c2, c3, c4, c5, c6, c7, c8, c9) = read_counts(top_counts, counts_base, s);
                put_record(out, out_base, t, s, source, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9);
                t += 1;
            }
        } else {
            let mut max = lp_at(top_log_prob, lp_base, 0, 0.0);
            let mut fi = 1u32;
            while fi < n {
                let v = lp_at(top_log_prob, lp_base, fi, 0.0);
                if v > max {
                    max = v;
                }
                fi += 1;
            }
            let mut sum = 0.0f32;
            let mut g = 0u32;
            while g < n {
                sum += (lp_at(top_log_prob, lp_base, g, 0.0) - max).exp();
                g += 1;
            }
            let rest_k = k - n;
            let rest_f = rest_k as f32;
            // Per-slot shares as scalars (no local arrays).
            let mut f0 = 0u32;
            let mut f1 = 0u32;
            let mut f2 = 0u32;
            let mut f3 = 0u32;
            let mut f4 = 0u32;
            let mut f5 = 0u32;
            let mut f6 = 0u32;
            let mut f7 = 0u32;
            let mut e0 = 0.0f32;
            let mut e1 = 0.0f32;
            let mut e2 = 0.0f32;
            let mut e3 = 0.0f32;
            let mut e4 = 0.0f32;
            let mut e5 = 0.0f32;
            let mut e6 = 0.0f32;
            let mut e7 = 0.0f32;
            let mut d0 = 0u32;
            let mut d1 = 0u32;
            let mut d2 = 0u32;
            let mut d3 = 0u32;
            let mut d4 = 0u32;
            let mut d5 = 0u32;
            let mut d6 = 0u32;
            let mut d7 = 0u32;
            let mut h = 0u32;
            while h < n {
                let p = (lp_at(top_log_prob, lp_base, h, 0.0) - max).exp() / sum;
                let quota = rest_f * p;
                let floor = quota.floor() as u32;
                let frac = quota - floor as f32;
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
            // Slot-order reconcile of a floor-sum overshoot (same shave as
            // `reconcile_extra_shares`, inlined with scalars).
            let mut fsum = 0u32;
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
                let mut excess = fsum - rest_k;
                if n > 0 {
                    let mut take = f0;
                    if take > excess {
                        take = excess;
                    }
                    f0 -= take;
                    excess -= take;
                }
                if n > 1 {
                    let mut take = f1;
                    if take > excess {
                        take = excess;
                    }
                    f1 -= take;
                    excess -= take;
                }
                if n > 2 {
                    let mut take = f2;
                    if take > excess {
                        take = excess;
                    }
                    f2 -= take;
                    excess -= take;
                }
                if n > 3 {
                    let mut take = f3;
                    if take > excess {
                        take = excess;
                    }
                    f3 -= take;
                    excess -= take;
                }
                if n > 4 {
                    let mut take = f4;
                    if take > excess {
                        take = excess;
                    }
                    f4 -= take;
                    excess -= take;
                }
                if n > 5 {
                    let mut take = f5;
                    if take > excess {
                        take = excess;
                    }
                    f5 -= take;
                    excess -= take;
                }
                if n > 6 {
                    let mut take = f6;
                    if take > excess {
                        take = excess;
                    }
                    f6 -= take;
                    excess -= take;
                }
                if n > 7 {
                    let mut take = f7;
                    if take > excess {
                        take = excess;
                    }
                    f7 -= take;
                    excess -= take;
                }
                let _ = excess;
                fsum = rest_k;
            }
            let mut rest = rest_k - fsum;
            let mut picks = 0u32;
            while rest > 0 && picks < n {
                let mut best = 0u32;
                let mut best_frac = -1.0f32;
                let mut q = 0u32;
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
            // Trajectories to formulas in slot order, written directly (no
            // assign array); the tail keeps exactly K writes.
            let mut t = 0u32;
            let mut s = 0u32;
            while s < n {
                let mut count = if s == 0 {
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
                    let source = word_at(top, top_base, s * 2, u32::MAX);
                    let (c0, c1, c2, c3, c4, c5, c6, c7, c8, c9) =
                        read_counts(top_counts, counts_base, s);
                    put_record(out, out_base, t, s, source, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9);
                    t += 1;
                    count -= 1;
                }
                s += 1;
            }
            while t < k {
                let sv = n - 1;
                let source = word_at(top, top_base, sv * 2, u32::MAX);
                let (c0, c1, c2, c3, c4, c5, c6, c7, c8, c9) =
                    read_counts(top_counts, counts_base, sv);
                put_record(out, out_base, t, sv, source, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9);
                t += 1;
            }
        }
    }
}

/// One spectrum's retained-to-window slot translation: the kernel twin of
/// `ms2_allocate_window`.
///
/// Full-buffer form over the kernel's 3 bound arrays. Copies all 12 words of
/// every `[B, K, 12]` allocation record, mapping word 0 (the retained formula
/// index `ms2_allocate` wrote, `u32::MAX` when there is none) through
/// `top[(b, s), 1]` to the window slot — the formula rank the packed record
/// carries. A sentinel or out-of-range slot passes through unchanged (such
/// rows are never selected; the kernel and the twin agree on the copy).
pub fn alloc_window_lane(
    traj_in: &[u32],
    top: &[u32],
    spectrum: u32,
    trajectory: u32,
    f: u32,
    k: u32,
    out: &mut [u32],
) {
    let src = (spectrum * k + trajectory) * ALLOC_RECORD;
    let mut s = u32::MAX;
    if (src as usize) < traj_in.len() {
        s = traj_in[src as usize];
    }
    let mut w = s;
    if s != u32::MAX && s < f {
        let addr = (spectrum * f + s) * 2 + 1;
        if (addr as usize) < top.len() {
            w = top[addr as usize];
        }
    }
    let dst = (spectrum * k + trajectory) * ALLOC_RECORD;
    let mut i = 0u32;
    while i < ALLOC_RECORD {
        let mut v = w;
        if i != 0 {
            let addr = src + i;
            v = 0;
            if (addr as usize) < traj_in.len() {
                v = traj_in[addr as usize];
            }
        }
        if ((dst + i) as usize) < out.len() {
            out[(dst + i) as usize] = v;
        }
        i += 1;
    }
}

/// Host wrapper around [`allocate_lane`] (NOT a kernel twin).
#[allow(clippy::too_many_arguments)]
pub fn allocate_checked(
    top: &[u32],
    top_counts: &[u32],
    top_log_prob: &[f32],
    top_count: &[u32],
    batch: u32,
    f: u32,
    k: u32,
    mode: u32,
    out: &mut [u32],
) -> Result<()> {
    if k > ALLOC_K_MAX {
        return Err(Error::config(format!(
            "allocate: K {k} exceeds {ALLOC_K_MAX}"
        )));
    }
    if f > ALLOC_F_MAX {
        return Err(Error::config(format!(
            "allocate: F {f} exceeds {ALLOC_F_MAX}"
        )));
    }
    if (top_count.len() as u32) < batch {
        return Err(Error::config(format!(
            "allocate: top_count holds {} words for B {batch}",
            top_count.len()
        )));
    }
    let mut b = 0u32;
    while b < batch {
        if top_count[b as usize] > f {
            return Err(Error::config(format!(
                "allocate: top_count {} exceeds F {f}",
                top_count[b as usize]
            )));
        }
        b += 1;
    }
    if (top.len() as u32) < batch * (f * 2) {
        return Err(Error::config("allocate: top shorter than [B, F, 2]"));
    }
    if (top_counts.len() as u32) < batch * (f * 10) {
        return Err(Error::config("allocate: top_counts shorter than [B, F, 10]"));
    }
    if (top_log_prob.len() as u32) < batch * f {
        return Err(Error::config("allocate: top_log_prob shorter than [B, F]"));
    }
    if (out.len() as u32) < batch * (k * ALLOC_RECORD) {
        return Err(Error::config(format!(
            "allocate: out holds {} words for [B {batch}, K {k}, 12]",
            out.len()
        )));
    }
    let mut s = 0u32;
    while s < batch {
        allocate_lane(top, top_counts, top_log_prob, top_count, s, f, k, mode, out);
        s += 1;
    }
    Ok(())
}
