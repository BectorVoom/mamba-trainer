//! Batched matrix multiplication.
//!
//! Batch dimensions are collapsed on the host into one leading axis, and a stride
//! of `0` marks a broadcast operand — that is what lets `[b, m, k] @ [k, n]` work
//! without materialising the broadcast.
//!
//! # Two kernels, and why the default is the naive one
//!
//! [`MatmulKernel::Tiled`] is the textbook shared-memory kernel: stage a tile of
//! each operand, `sync_cube`, accumulate, `sync_cube`. It is the right shape for a
//! GPU. It is also *catastrophically* wrong for a CPU backend, where a cube barrier
//! is not a hardware instruction — measured on CubeCL's CPU runtime, one 64x64x64
//! product costs **1.48 s** tiled against **0.12 ms** with no barriers at all.
//!
//! So the default is [`MatmulKernel::Simple`]: one unit per output *vector*, a plain
//! loop over `k`, no shared memory and no synchronisation. It is portable and never
//! pathological; on a GPU it is merely unoptimal rather than unusable. Backends
//! where barriers are cheap can opt into the tiled path with [`set_default_kernel`].
//!
//! "Per output vector" is where the speed comes from. A unit owns `line` adjacent
//! columns of one output row: the `lhs` element is a scalar splat, and the `rhs`
//! row and the accumulator are [`Vector`]s, so the inner loop is a vector FMA per
//! step instead of a scalar one. Measured on CubeCL's CPU runtime, a 256x256x256
//! product costs 13.3 ms scalar against 51 us at 16 lanes wide.

use core::sync::atomic::{AtomicU8, Ordering};

use cubecl::prelude::*;

use crate::backend::{FloatElem, launch_1d, line_size_for};
use crate::error::{Error, Result};
use crate::tensor::base::Tensor;
use crate::tensor::shape::Shape;

/// Which matmul kernel to launch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatmulKernel {
    /// Pick per device: [`MatmulKernel::RowTiled`] where a cube has real hardware
    /// planes, [`MatmulKernel::Simple`] otherwise. This is the default.
    Auto,
    /// One unit per output vector; no shared memory, no barriers.
    Simple,
    /// One unit per `ROWS` output vectors stacked down a column, accumulating in
    /// registers. No shared memory and no barriers either.
    RowTiled,
    /// Shared-memory tiles with cube barriers. Only choose this on a backend where
    /// `sync_cube` is cheap.
    Tiled,
    /// Shared-memory block tiles with a two-dimensional register tile per unit.
    /// The fastest kernel here on a GPU, and the worst on a CPU runtime.
    BlockTiled,
}

static DEFAULT_KERNEL: AtomicU8 = AtomicU8::new(0);

/// Choose the kernel used by [`matmul`] from now on.
///
/// A backend tuning knob, not a semantic one: every kernel computes the same
/// product to within floating-point associativity.
pub fn set_default_kernel(kernel: MatmulKernel) {
    DEFAULT_KERNEL.store(
        match kernel {
            MatmulKernel::Auto => 0,
            MatmulKernel::Simple => 1,
            MatmulKernel::RowTiled => 2,
            MatmulKernel::Tiled => 3,
            MatmulKernel::BlockTiled => 4,
        },
        Ordering::Relaxed,
    );
}

/// The kernel [`matmul`] will use, before per-device resolution.
pub fn default_kernel() -> MatmulKernel {
    match DEFAULT_KERNEL.load(Ordering::Relaxed) {
        1 => MatmulKernel::Simple,
        2 => MatmulKernel::RowTiled,
        3 => MatmulKernel::Tiled,
        4 => MatmulKernel::BlockTiled,
        _ => MatmulKernel::Auto,
    }
}

/// Tile edge for [`MatmulKernel::Tiled`].
const TILE: usize = 16;

/// Output rows one unit accumulates in [`MatmulKernel::RowTiled`].
///
/// Every step of the `k` loop is one vector load from `rhs` reused across all
/// `ROWS` accumulators, so arithmetic per byte read grows linearly with this number
/// until the register file runs out — and then falls off a cliff. Measured on an
/// RDNA3.5 iGPU at 1024x1024x1024, in GFLOP/s:
///
/// | rows | 2 | 4 | 8 | 12 | 16 |
/// |---|---|---|---|---|---|
/// | wgpu | 452 | 548 | **586** | 335 | 324 |
/// | HIP | - | 403 | **617** | 230 | 211 |
const ROWS: usize = 8;

/// One unit per output vector.
///
/// `n_lines`, `rhs_batch_lines` and the `rhs`/`out` indices are all counted in
/// vectors of `N` elements; `lhs` stays scalar because a unit reads one `lhs` value
/// per step and splats it across the accumulator.
#[cube(launch_unchecked)]
fn matmul_simple_kernel<F: Float + CubeElement, N: Size>(
    lhs: &Array<F>,
    rhs: &Array<Vector<F, N>>,
    out: &mut Array<Vector<F, N>>,
    m: usize,
    n_lines: usize,
    k: usize,
    lhs_batch_stride: usize,
    rhs_batch_lines: usize,
) {
    if ABSOLUTE_POS < out.len() {
        let batch = ABSOLUTE_POS / (m * n_lines);
        let within = ABSOLUTE_POS % (m * n_lines);
        let row = within / n_lines;
        let col = within % n_lines;
        let lhs_base = batch * lhs_batch_stride + row * k;
        let rhs_base = batch * rhs_batch_lines + col;
        let mut acc = Vector::<F, N>::new(F::new(0.0_f32));
        for p in 0..k {
            acc += Vector::<F, N>::new(lhs[lhs_base + p]) * rhs[rhs_base + p * n_lines];
        }
        out[ABSOLUTE_POS] = acc;
    }
}

/// One unit per `rows` stacked output vectors.
///
/// The `rhs` vector loaded on each `k` step is reused by every accumulator, and the
/// `lhs` values a unit reads are the same for every unit in the plane — a broadcast
/// out of cache. That turns the roughly one-FLOP-per-byte of the simple kernel into
/// `2 * rows` FLOPs per byte, which is what a GPU needs to leave memory-bound
/// territory. Nothing is shared and nothing synchronises, so it is safe on the CPU
/// runtime too; it is simply not faster there.
#[cube(launch_unchecked)]
fn matmul_row_tiled_kernel<F: Float + CubeElement, N: Size>(
    lhs: &Array<F>,
    rhs: &Array<Vector<F, N>>,
    out: &mut Array<Vector<F, N>>,
    m: usize,
    n_lines: usize,
    k: usize,
    row_tiles: usize,
    lanes: usize,
    lhs_batch_stride: usize,
    rhs_batch_lines: usize,
    #[comptime] rows: usize,
) {
    if ABSOLUTE_POS < lanes {
        let tile_span = row_tiles * n_lines;
        let batch = ABSOLUTE_POS / tile_span;
        let within = ABSOLUTE_POS % tile_span;
        let row0 = (within / n_lines) * rows;
        let col = within % n_lines;
        let rhs_base = batch * rhs_batch_lines + col;
        let lhs_base = batch * lhs_batch_stride;

        // Hoist the tail guard: a row past the end reads row 0 instead of running
        // off the buffer, and its result is simply never stored.
        let mut row_offset = Array::<usize>::new(rows);
        let mut acc = Array::<Vector<F, N>>::new(rows);
        #[unroll]
        for i in 0..rows {
            let row = select(row0 + i < m, row0 + i, 0usize);
            row_offset[i] = lhs_base + row * k;
            acc[i] = Vector::<F, N>::new(F::new(0.0_f32));
        }

        for p in 0..k {
            let rv = rhs[rhs_base + p * n_lines];
            #[unroll]
            for i in 0..rows {
                acc[i] += Vector::<F, N>::new(lhs[row_offset[i] + p]) * rv;
            }
        }

        let out_base = batch * m * n_lines + col;
        #[unroll]
        for i in 0..rows {
            if row0 + i < m {
                out[out_base + (row0 + i) * n_lines] = acc[i];
            }
        }
    }
}

/// A block-tiling shape: the output rectangle one cube owns, how deep it steps
/// through `k`, and the register tile one unit owns inside it.
///
/// A cube is `(bm / tm) * (bn / tn)` units. Each `bk` step stages `bm * bk + bk * bn`
/// elements in shared memory and spends `bm * bn * bk` multiply-adds on them, so
/// arithmetic per staged byte grows with the block edge — until the accumulators
/// stop fitting in registers, at which point it collapses. `tn` doubles as the vector
/// width, which is what makes the inner loop a `tm`-long run of vector multiply-adds
/// against a single shared-memory vector load.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct BlockShape {
    bm: usize,
    bn: usize,
    bk: usize,
    tm: usize,
    tn: usize,
}

impl BlockShape {
    const fn units(self) -> usize {
        (self.bm / self.tm) * (self.bn / self.tn)
    }

    /// Whether every unit in the cube gets a whole number of staging tasks.
    ///
    /// Both kernels stage their tiles with `#[unroll] for t in 0..(tile / units)`, so
    /// a tile smaller than the cube does not merely waste units — it makes the loop
    /// run zero times and leaves shared memory holding whatever was there before.
    /// That is a silently wrong answer rather than a slow one, and a fast-looking one
    /// too, which is exactly the sort of candidate a tuner would happily pick. It is
    /// checked rather than commented on.
    const fn stages_evenly(self) -> bool {
        let units = self.units();
        (self.bm * self.bk).is_multiple_of(units)
            && (self.bk * (self.bn / self.tn)).is_multiple_of(units)
            && (self.bk * self.bn).is_multiple_of(units)
    }

    /// The same property for the vector-staged kernel, whose `lhs` tasks are counted
    /// in vectors of `tn`: every unit needs a whole number of vector loads, and the
    /// vector columns of one `bk` step must divide across the units.
    const fn stages_evenly_vec(self) -> bool {
        let units = self.units();
        self.bk.is_multiple_of(self.tn)
            && (self.bm * self.bk / self.tn).is_multiple_of(units)
            && units.is_multiple_of(self.bk / self.tn)
    }
}

/// The tiling shapes the tuner may choose between.
///
/// Measured on an RDNA3.5 iGPU, no single one is best at more than about half the
/// shapes a training step issues, and the reasons pull against each other: a taller
/// block reuses more staged data per unit but needs rows to fill it, and a deeper
/// `bk` amortises more of the barrier pair but costs shared memory, which is what
/// limits how many cubes a compute unit can hold at once.
///
/// Every shape here satisfies [`BlockShape::stages_evenly`]; the list is checked
/// against that below, because a shape that does not is wrong rather than slow.
/// It is also deliberately short — each entry costs one compiled kernel variant and
/// four timed launches the first time a shape is seen.
const BLOCK_CANDIDATES: [BlockShape; 4] = [
    BlockShape { bm: 128, bn: 64, bk: 16, tm: 8, tn: 4 },
    BlockShape { bm: 128, bn: 64, bk: 32, tm: 8, tn: 4 },
    BlockShape { bm: 128, bn: 32, bk: 16, tm: 8, tn: 4 },
    BlockShape { bm: 64, bn: 64, bk: 16, tm: 4, tn: 4 },
];

const _: () = {
    let mut i = 0;
    while i < BLOCK_CANDIDATES.len() {
        assert!(
            BLOCK_CANDIDATES[i].stages_evenly(),
            "a block shape whose tiles do not divide across its units would stage \
             nothing and return whatever was in shared memory"
        );
        i += 1;
    }
};

/// The shape [`MatmulKernel::BlockTiled`] uses when it is asked for explicitly,
/// rather than resolved by measurement. The best single choice on the shapes this
/// crate issues, and the first candidate the tuner tries.
const BLOCK_TALL: BlockShape = BLOCK_CANDIDATES[0];

/// The same block tiling, for an operand that is stored transposed.
///
/// The adjoint of a matrix product is two more matrix products against transposed
/// operands — `dA = G Bᵀ` and `dB = Aᵀ G` — and materialising those transposes was
/// costing more than the products themselves: a `permute` that swaps the contiguous
/// axis is the one case the vectorised strided copy cannot help with, and a training
/// step does two of them per matmul on tensors of a few million elements.
///
/// Nothing about the algorithm changes; only where a staged element is read from.
/// Because both tiles are staged through shared memory anyway, transposition is one
/// index expression, chosen at compile time, and the inner loop is untouched. What is
/// lost relative to [`matmul_block_tiled_kernel`] is the vectorised staging — a
/// transposed operand is not contiguous along the axis the vector would span — so
/// this kernel is used only when a transpose is actually asked for, and never in
/// place of the plain one.
///
/// `sb` is padded for the same reason `sa` is: with `rhs_t` set, consecutive units
/// stage consecutive `k` for one column and their shared-memory addresses are a
/// block edge apart.
#[cube(launch_unchecked)]
fn matmul_block_tiled_t_kernel<F: Float + CubeElement>(
    lhs: &Array<F>,
    rhs: &Array<F>,
    out: &mut Array<F>,
    m: usize,
    n: usize,
    k: usize,
    lhs_batch_stride: usize,
    rhs_batch_stride: usize,
    col_blocks: usize,
    #[comptime] bm: usize,
    #[comptime] bn: usize,
    #[comptime] bk: usize,
    #[comptime] tm: usize,
    #[comptime] tn: usize,
    #[comptime] units: usize,
    #[comptime] lhs_t: bool,
    #[comptime] rhs_t: bool,
) {
    let unit = UNIT_POS_X as usize;
    let block = CUBE_POS_X as usize;
    let batch = CUBE_POS_Y as usize;
    let row0 = (block / col_blocks) * bm;
    let col0 = (block % col_blocks) * bn;

    let lhs_base = batch * lhs_batch_stride;
    let rhs_base = batch * rhs_batch_stride;

    let tile_row = (unit / (bn / tn)) * tm;
    let tile_col = (unit % (bn / tn)) * tn;

    let a_stride = bm + 1;
    let b_stride = bn + 1;
    let mut sa = SharedMemory::<F>::new(bk * (bm + 1));
    let mut sb = SharedMemory::<F>::new(bk * (bn + 1));

    let mut acc = Array::<F>::new(tm * tn);
    #[unroll]
    for i in 0..tm {
        #[unroll]
        for j in 0..tn {
            acc[i * tn + j] = F::new(0.0_f32);
        }
    }
    let mut a_reg = Array::<F>::new(tm);
    let mut b_reg = Array::<F>::new(tn);

    // Which axis consecutive units walk depends on which one is contiguous in
    // memory, and that is exactly what transposing changes. Getting this wrong
    // is not a small penalty: with `rhs` transposed and units walking columns,
    // consecutive units read addresses `k` floats apart and every load is its own
    // memory transaction. Choosing the assignment at compile time from the same
    // flag that chooses the index expression keeps both cases coalesced.
    //
    // The next tiles are prefetched into registers one `bk` step ahead, exactly as
    // in the plain kernel: the loads for step `s + 1` are issued right after the
    // first barrier and retire behind step `s`'s arithmetic, instead of stalling
    // the whole cube between the barriers.
    let mut a_pref = Array::<F>::new(bm * bk / units);
    let mut b_pref = Array::<F>::new(bk * bn / units);

    #[unroll]
    for t in 0..(bm * bk / units) {
        let mut row = unit / bk + t * (units / bk);
        let mut kk = unit % bk;
        if lhs_t {
            // `lhs` is stored `[k, m]`: rows are contiguous.
            row = unit % bm;
            kk = unit / bm + t * (units / bm);
        }
        let global_row = row0 + row;
        let mut v = F::new(0.0_f32);
        if global_row < m && kk < k {
            if lhs_t {
                v = lhs[lhs_base + kk * m + global_row];
            } else {
                v = lhs[lhs_base + global_row * k + kk];
            }
        }
        a_pref[t] = v;
    }
    #[unroll]
    for t in 0..(bk * bn / units) {
        let mut kk = unit / bn + t * (units / bn);
        let mut col = unit % bn;
        if rhs_t {
            // `rhs` is stored `[n, k]`: the contraction axis is contiguous.
            kk = unit % bk;
            col = unit / bk + t * (units / bk);
        }
        let global_col = col0 + col;
        let mut v = F::new(0.0_f32);
        if kk < k && global_col < n {
            if rhs_t {
                v = rhs[rhs_base + global_col * k + kk];
            } else {
                v = rhs[rhs_base + kk * n + global_col];
            }
        }
        b_pref[t] = v;
    }

    let steps = k.div_ceil(bk);
    for step in 0..steps {
        #[unroll]
        for t in 0..(bm * bk / units) {
            let mut row = unit / bk + t * (units / bk);
            let mut kk = unit % bk;
            if lhs_t {
                row = unit % bm;
                kk = unit / bm + t * (units / bm);
            }
            sa[kk * a_stride + row] = a_pref[t];
        }
        #[unroll]
        for t in 0..(bk * bn / units) {
            let mut kk = unit / bn + t * (units / bn);
            let mut col = unit % bn;
            if rhs_t {
                kk = unit % bk;
                col = unit / bk + t * (units / bk);
            }
            sb[kk * b_stride + col] = b_pref[t];
        }

        sync_cube();

        if step + 1 < steps {
            let k0 = (step + 1) * bk;
            #[unroll]
            for t in 0..(bm * bk / units) {
                let mut row = unit / bk + t * (units / bk);
                let mut kk = unit % bk;
                if lhs_t {
                    row = unit % bm;
                    kk = unit / bm + t * (units / bm);
                }
                let global_row = row0 + row;
                let global_k = k0 + kk;
                let mut v = F::new(0.0_f32);
                if global_row < m && global_k < k {
                    if lhs_t {
                        v = lhs[lhs_base + global_k * m + global_row];
                    } else {
                        v = lhs[lhs_base + global_row * k + global_k];
                    }
                }
                a_pref[t] = v;
            }
            #[unroll]
            for t in 0..(bk * bn / units) {
                let mut kk = unit / bn + t * (units / bn);
                let mut col = unit % bn;
                if rhs_t {
                    kk = unit % bk;
                    col = unit / bk + t * (units / bk);
                }
                let global_k = k0 + kk;
                let global_col = col0 + col;
                let mut v = F::new(0.0_f32);
                if global_k < k && global_col < n {
                    if rhs_t {
                        v = rhs[rhs_base + global_col * k + global_k];
                    } else {
                        v = rhs[rhs_base + global_k * n + global_col];
                    }
                }
                b_pref[t] = v;
            }
        }

        #[unroll]
        for kk in 0..bk {
            #[unroll]
            for i in 0..tm {
                a_reg[i] = sa[kk * a_stride + tile_row + i];
            }
            #[unroll]
            for j in 0..tn {
                b_reg[j] = sb[kk * b_stride + tile_col + j];
            }
            #[unroll]
            for i in 0..tm {
                #[unroll]
                for j in 0..tn {
                    acc[i * tn + j] += a_reg[i] * b_reg[j];
                }
            }
        }

        sync_cube();
    }

    let out_base = batch * m * n;
    #[unroll]
    for i in 0..tm {
        let global_row = row0 + tile_row + i;
        if global_row < m {
            #[unroll]
            for j in 0..tn {
                let global_col = col0 + tile_col + j;
                if global_col < n {
                    out[out_base + global_row * n + global_col] = acc[i * tn + j];
                }
            }
        }
    }
}

/// Shared-memory block tiling with a two-dimensional, vectorised register tile.
///
/// One cube owns a `BM x BN` rectangle of the output and walks `k` in steps of `BK`.
/// Each step stages two tiles in shared memory:
///
/// * the `BM x BK` piece of `lhs`, **transposed** to `sa[kk][row]`, so the `TM`
///   values a unit wants for one `kk` are adjacent — and padded by one element per
///   row, because without the padding the transposed store puts all `BK` units
///   staging a row in the same memory bank and the kernel loses a factor of two;
/// * the `BK x BN` piece of `rhs` as it lies, held as `BN / TN` **vectors** per row,
///   which is exactly the granularity a unit consumes.
///
/// The inner loop is then `TM` vector fused multiply-adds against one vector read
/// and `TM` scalar reads: sixteen multiply-adds for five shared-memory accesses,
/// where the row-tiled kernel manages eight for nine — and its nine are from global
/// memory rather than shared. Global reads are vectorised on `rhs` and on the
/// output, so a cube issues a quarter of the load instructions it otherwise would.
///
/// Bounds are checked on the way in and on the way out rather than by padding, so
/// any `m`, `n`, `k` works as long as the vector width divides `n`; a shape that
/// divides the block simply never takes the false branch.
#[cube(launch_unchecked)]
fn matmul_block_tiled_kernel<F: Float + CubeElement, N: Size>(
    lhs: &Array<F>,
    rhs: &Array<Vector<F, N>>,
    out: &mut Array<Vector<F, N>>,
    m: usize,
    n_lines: usize,
    k: usize,
    lhs_batch_stride: usize,
    rhs_batch_lines: usize,
    col_blocks: usize,
    #[comptime] bm: usize,
    #[comptime] bn: usize,
    #[comptime] bk: usize,
    #[comptime] tm: usize,
    #[comptime] tn: usize,
    #[comptime] units: usize,
) {
    let block_lines = bn / tn;
    let unit = UNIT_POS_X as usize;
    let block = CUBE_POS_X as usize;
    let batch = CUBE_POS_Y as usize;
    let row0 = (block / col_blocks) * bm;
    let col0 = (block % col_blocks) * bn;
    let col0_lines = col0 / tn;

    let lhs_base = batch * lhs_batch_stride;
    let rhs_base = batch * rhs_batch_lines;

    // Where this unit's register tile sits inside the block.
    let tile_row = (unit / block_lines) * tm;
    let tile_line = unit % block_lines;

    // Where this unit's share of each staging pass sits. Consecutive units take
    // consecutive `k` for `lhs` and consecutive column vectors for `rhs`, which is
    // what makes both global reads coalesce.
    let a_row = unit / bk;
    let a_col = unit % bk;
    let b_row = unit / block_lines;
    let b_line = unit % block_lines;

    let a_stride = bm + 1;
    let mut sa = SharedMemory::<F>::new(bk * (bm + 1));
    let mut sb = SharedMemory::<Vector<F, N>>::new(bk * (bn / tn));

    let mut acc = Array::<Vector<F, N>>::new(tm);
    #[unroll]
    for i in 0..tm {
        acc[i] = Vector::<F, N>::new(F::new(0.0_f32));
    }

    // The next tiles, prefetched into registers one `bk` step ahead. Without this
    // the global loads sit on the critical path between the two barriers: the whole
    // cube stages, waits on DRAM, computes, and repeats. Fetching step `s + 1` right
    // after the first barrier lets those loads retire behind step `s`'s arithmetic —
    // the wait moves to the staging store, which happens a full compute phase later.
    let mut a_pref = Array::<F>::new(bm * bk / units);
    let mut b_pref = Array::<Vector<F, N>>::new(bk * bn / tn / units);

    #[unroll]
    for t in 0..(bm * bk / units) {
        let row = a_row + t * (units / bk);
        let global_row = row0 + row;
        let mut v = F::new(0.0_f32);
        if global_row < m && a_col < k {
            v = lhs[lhs_base + global_row * k + a_col];
        }
        a_pref[t] = v;
    }
    #[unroll]
    for t in 0..(bk * bn / tn / units) {
        let kk = b_row + t * (units / block_lines);
        let global_line = col0_lines + b_line;
        let mut v = Vector::<F, N>::new(F::new(0.0_f32));
        if kk < k && global_line < n_lines {
            v = rhs[rhs_base + kk * n_lines + global_line];
        }
        b_pref[t] = v;
    }

    let steps = k.div_ceil(bk);
    for step in 0..steps {
        #[unroll]
        for t in 0..(bm * bk / units) {
            let row = a_row + t * (units / bk);
            sa[a_col * a_stride + row] = a_pref[t];
        }
        #[unroll]
        for t in 0..(bk * bn / tn / units) {
            let kk = b_row + t * (units / block_lines);
            sb[kk * block_lines + b_line] = b_pref[t];
        }

        sync_cube();

        if step + 1 < steps {
            let k0 = (step + 1) * bk;
            #[unroll]
            for t in 0..(bm * bk / units) {
                let row = a_row + t * (units / bk);
                let global_row = row0 + row;
                let global_k = k0 + a_col;
                let mut v = F::new(0.0_f32);
                if global_row < m && global_k < k {
                    v = lhs[lhs_base + global_row * k + global_k];
                }
                a_pref[t] = v;
            }
            #[unroll]
            for t in 0..(bk * bn / tn / units) {
                let kk = b_row + t * (units / block_lines);
                let global_k = k0 + kk;
                let global_line = col0_lines + b_line;
                let mut v = Vector::<F, N>::new(F::new(0.0_f32));
                if global_k < k && global_line < n_lines {
                    v = rhs[rhs_base + global_k * n_lines + global_line];
                }
                b_pref[t] = v;
            }
        }

        #[unroll]
        for kk in 0..bk {
            let b_vec = sb[kk * block_lines + tile_line];
            #[unroll]
            for i in 0..tm {
                acc[i] += Vector::<F, N>::new(sa[kk * a_stride + tile_row + i]) * b_vec;
            }
        }

        sync_cube();
    }

    let out_base = batch * m * n_lines;
    let global_line = col0_lines + tile_line;
    if global_line < n_lines {
        #[unroll]
        for i in 0..tm {
            let global_row = row0 + tile_row + i;
            if global_row < m {
                out[out_base + global_row * n_lines + global_line] = acc[i];
            }
        }
    }
}

#[cube(launch_unchecked)]
fn matmul_tiled_kernel<F: Float + CubeElement>(
    lhs: &Array<F>,
    rhs: &Array<F>,
    out: &mut Array<F>,
    m: usize,
    n: usize,
    k: usize,
    lhs_batch_stride: usize,
    rhs_batch_stride: usize,
    #[comptime] tile: usize,
) {
    let batch = CUBE_POS_Z as usize;
    let ty = UNIT_POS_Y as usize;
    let tx = UNIT_POS_X as usize;
    let row = CUBE_POS_Y as usize * tile + ty;
    let col = CUBE_POS_X as usize * tile + tx;

    let mut tile_a = SharedMemory::<F>::new(tile * tile);
    let mut tile_b = SharedMemory::<F>::new(tile * tile);

    let lhs_base = batch * lhs_batch_stride;
    let rhs_base = batch * rhs_batch_stride;

    let mut acc = F::new(0.0_f32);
    let num_tiles = k.div_ceil(tile);

    for t in 0..num_tiles {
        let a_col = t * tile + tx;
        let b_row = t * tile + ty;

        if row < m && a_col < k {
            tile_a[ty * tile + tx] = lhs[lhs_base + row * k + a_col];
        } else {
            tile_a[ty * tile + tx] = F::new(0.0_f32);
        }
        if col < n && b_row < k {
            tile_b[ty * tile + tx] = rhs[rhs_base + b_row * n + col];
        } else {
            tile_b[ty * tile + tx] = F::new(0.0_f32);
        }

        sync_cube();

        for i in 0..tile {
            acc += tile_a[ty * tile + i] * tile_b[i * tile + tx];
        }

        sync_cube();
    }

    if row < m && col < n {
        out[batch * m * n + row * n + col] = acc;
    }
}

/// [`matmul_block_tiled_kernel`] with `lhs` read as vectors of the same width as
/// `rhs`.
///
/// The plain kernel's `lhs` staging is its one scalar global access: `bm * bk`
/// loads per step, each a separate instruction using a fraction of a cache line.
/// Rows of `lhs` are contiguous along `k`, so when the vector width divides `k` a
/// unit can fetch `N` consecutive `k` in one load and scatter them into the
/// transposed shared tile — a quarter of the load instructions for the same bytes,
/// and each transaction now spans a whole cache line.
///
/// The scatter changes which units store to shared memory at once, so the padding
/// changes with it: with vector staging the simultaneous stores come from `tn`
/// consecutive `k` columns across a run of rows, and a pad of *two* puts them on
/// distinct banks for every block edge this crate uses (all multiples of eight),
/// where the plain kernel's pad of one leaves half of them colliding.
#[cube(launch_unchecked)]
fn matmul_block_tiled_vec_kernel<F: Float + CubeElement, N: Size>(
    lhs: &Array<Vector<F, N>>,
    rhs: &Array<Vector<F, N>>,
    out: &mut Array<Vector<F, N>>,
    m: usize,
    n_lines: usize,
    k_lines: usize,
    lhs_batch_lines: usize,
    rhs_batch_lines: usize,
    col_blocks: usize,
    #[comptime] bm: usize,
    #[comptime] bn: usize,
    #[comptime] bk: usize,
    #[comptime] tm: usize,
    #[comptime] tn: usize,
    #[comptime] units: usize,
) {
    let block_lines = bn / tn;
    let bk_lines = bk / tn;
    let unit = UNIT_POS_X as usize;
    let block = CUBE_POS_X as usize;
    let batch = CUBE_POS_Y as usize;
    let row0 = (block / col_blocks) * bm;
    let col0_lines = ((block % col_blocks) * bn) / tn;

    let lhs_base = batch * lhs_batch_lines;
    let rhs_base = batch * rhs_batch_lines;

    let tile_row = (unit / block_lines) * tm;
    let tile_line = unit % block_lines;

    // Consecutive units take consecutive `k` vectors for a run of rows of `lhs`,
    // and consecutive column vectors of `rhs` — both coalesce.
    let a_row = unit / bk_lines;
    let a_col = unit % bk_lines;
    let b_row = unit / block_lines;
    let b_line = unit % block_lines;

    let a_stride = bm + 2;
    let mut sa = SharedMemory::<F>::new(bk * (bm + 2));
    let mut sb = SharedMemory::<Vector<F, N>>::new(bk * (bn / tn));

    let mut acc = Array::<Vector<F, N>>::new(tm);
    #[unroll]
    for i in 0..tm {
        acc[i] = Vector::<F, N>::new(F::new(0.0_f32));
    }

    // Prefetched next tiles, exactly as in the plain kernel.
    let mut a_pref = Array::<Vector<F, N>>::new(bm * bk / tn / units);
    let mut b_pref = Array::<Vector<F, N>>::new(bk * bn / tn / units);

    #[unroll]
    for t in 0..(bm * bk / tn / units) {
        let row = a_row + t * (units / bk_lines);
        let global_row = row0 + row;
        let mut v = Vector::<F, N>::new(F::new(0.0_f32));
        if global_row < m && a_col < k_lines {
            v = lhs[lhs_base + global_row * k_lines + a_col];
        }
        a_pref[t] = v;
    }
    #[unroll]
    for t in 0..(bk * bn / tn / units) {
        let kk = b_row + t * (units / block_lines);
        let global_line = col0_lines + b_line;
        let mut v = Vector::<F, N>::new(F::new(0.0_f32));
        if kk < k_lines * tn && global_line < n_lines {
            v = rhs[rhs_base + kk * n_lines + global_line];
        }
        b_pref[t] = v;
    }

    let steps = k_lines.div_ceil(bk_lines);
    for step in 0..steps {
        #[unroll]
        for t in 0..(bm * bk / tn / units) {
            let row = a_row + t * (units / bk_lines);
            let v = a_pref[t];
            #[unroll]
            for j in 0..tn {
                sa[(a_col * tn + j) * a_stride + row] = v[j];
            }
        }
        #[unroll]
        for t in 0..(bk * bn / tn / units) {
            let kk = b_row + t * (units / block_lines);
            sb[kk * block_lines + b_line] = b_pref[t];
        }

        sync_cube();

        if step + 1 < steps {
            let k0_lines = (step + 1) * bk_lines;
            #[unroll]
            for t in 0..(bm * bk / tn / units) {
                let row = a_row + t * (units / bk_lines);
                let global_row = row0 + row;
                let global_col = k0_lines + a_col;
                let mut v = Vector::<F, N>::new(F::new(0.0_f32));
                if global_row < m && global_col < k_lines {
                    v = lhs[lhs_base + global_row * k_lines + global_col];
                }
                a_pref[t] = v;
            }
            #[unroll]
            for t in 0..(bk * bn / tn / units) {
                let kk = b_row + t * (units / block_lines);
                let global_k = (step + 1) * bk + kk;
                let global_line = col0_lines + b_line;
                let mut v = Vector::<F, N>::new(F::new(0.0_f32));
                if global_k < k_lines * tn && global_line < n_lines {
                    v = rhs[rhs_base + global_k * n_lines + global_line];
                }
                b_pref[t] = v;
            }
        }

        #[unroll]
        for kk in 0..bk {
            let b_vec = sb[kk * block_lines + tile_line];
            #[unroll]
            for i in 0..tm {
                acc[i] += Vector::<F, N>::new(sa[kk * a_stride + tile_row + i]) * b_vec;
            }
        }

        sync_cube();
    }

    let out_base = batch * m * n_lines;
    let global_line = col0_lines + tile_line;
    if global_line < n_lines {
        #[unroll]
        for i in 0..tm {
            let global_row = row0 + tile_row + i;
            if global_row < m {
                out[out_base + global_row * n_lines + global_line] = acc[i];
            }
        }
    }
}

/// One plane per output element, for products that are almost all reduction.
///
/// A shape like `[8, 4096] @ [4096, 8]ᵀ` is sixty-four dot products of half a
/// million elements — there is no output rectangle worth tiling, and the block
/// kernels collapse on it: a cube of 256 units owns at most 64 output elements and
/// spends its life at barriers, measured at 8 GFLOP/s where the elementwise kernels
/// move two orders of magnitude more data.
///
/// The right shape is the one the fused reductions already use: a plane per output
/// element, lanes striding the contraction axis in vectors, one `plane_sum` at the
/// end. Every load is contiguous across the lanes that issue it. This only works
/// when *both* operands are contiguous along `k` — `lhs` in its natural layout,
/// `rhs` stored `[n, k]` — which is exactly the `dA = G Bᵀ` adjoint that produces
/// these shapes in the first place.
#[cube(launch_unchecked)]
fn matmul_plane_dot_kernel<F: Float + CubeElement, N: Size>(
    lhs: &Array<Vector<F, N>>,
    rhs: &Array<Vector<F, N>>,
    out: &mut Array<F>,
    m: usize,
    cols: usize,
    k_lines: usize,
    lhs_batch_lines: usize,
    rhs_batch_lines: usize,
) {
    let width = PLANE_DIM as usize;
    let lane = UNIT_POS_PLANE as usize;
    let idx = ABSOLUTE_POS / width;
    let live = idx < out.len();
    let safe = select(live, idx, 0usize);
    let batch = safe / (m * cols);
    let within = safe % (m * cols);
    let lhs_base = batch * lhs_batch_lines + (within / cols) * k_lines;
    let rhs_base = batch * rhs_batch_lines + (within % cols) * k_lines;

    let mut acc = Vector::<F, N>::new(F::new(0.0_f32));
    let steps = k_lines.div_ceil(width);
    for s in 0..steps {
        let i = lane + s * width;
        if i < k_lines {
            acc += lhs[lhs_base + i] * rhs[rhs_base + i];
        }
    }
    let mut total = acc[0];
    #[unroll]
    for l in 1..N::value() {
        total += acc[l];
    }
    let dot = plane_sum(total);
    if live && lane == 0 {
        out[idx] = dot;
    }
}

/// Planes per cube for [`matmul_plane_dot_kernel`], matching the fused reductions.
const DOT_PLANES: usize = 4;

/// What a matmul call will actually launch: a kernel and, for the block-tiled ones,
/// the tiling shape to launch it with.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Plan {
    Simple,
    RowTiled,
    Tiled,
    /// Block-tiled, both operands in their natural layout.
    Block(BlockShape),
    /// Block-tiled with one operand read transposed.
    BlockT(BlockShape, bool, bool),
    /// Block-tiled with vectorised `lhs` staging; needs the vector width to divide `k`.
    BlockV(BlockShape),
    /// One plane per output element; only for `lhs @ rhsᵀ` with a small output.
    PlaneDot,
}

/// Launch one specific plan into an existing output buffer.
#[allow(clippy::too_many_arguments)]
fn launch_matmul<R: Runtime, E: FloatElem>(
    plan: Plan,
    lhs: &Tensor<R, E>,
    rhs: &Tensor<R, E>,
    out: &Tensor<R, E>,
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
    lhs_batch_stride: usize,
    rhs_batch_stride: usize,
) {
    match plan {
        Plan::Simple => {
            // Widths that divide `n` also divide both possible `rhs` batch strides
            // (`0` when broadcast, `k * n` otherwise), so no unit's vector can
            // straddle two rows.
            let line = line_size_for::<R, E>(lhs.client(), n);
            let lanes = batch * m * (n / line);
            let (count, dim) = launch_1d(lhs.client(), lanes, k * line);
            unsafe {
                matmul_simple_kernel::launch_unchecked::<E, R>(
                    lhs.client(),
                    count,
                    dim,
                    line,
                    lhs.arg(),
                    rhs.arg(),
                    out.arg(),
                    m,
                    n / line,
                    k,
                    lhs_batch_stride,
                    rhs_batch_stride / line,
                );
            }
        }
        Plan::RowTiled => {
            let line = line_size_for::<R, E>(lhs.client(), n);
            let n_lines = n / line;
            let row_tiles = m.div_ceil(ROWS);
            let lanes = batch * row_tiles * n_lines;
            let (count, dim) = launch_1d(lhs.client(), lanes, k * line * ROWS);
            unsafe {
                matmul_row_tiled_kernel::launch_unchecked::<E, R>(
                    lhs.client(),
                    count,
                    dim,
                    line,
                    lhs.arg(),
                    rhs.arg(),
                    out.arg(),
                    m,
                    n_lines,
                    k,
                    row_tiles,
                    lanes,
                    lhs_batch_stride,
                    rhs_batch_stride / line,
                    ROWS,
                );
            }
        }
        Plan::Block(shape) => {
            let row_blocks = m.div_ceil(shape.bm);
            let col_blocks = n.div_ceil(shape.bn);
            // One cube per output block, batches on the y axis. Counting launches
            // here keeps `launch_count` honest; this kernel does not go through
            // `launch_1d`, because its geometry is fixed by the block shape rather
            // than derived from an element count.
            crate::backend::count_launch();
            unsafe {
                matmul_block_tiled_kernel::launch_unchecked::<E, R>(
                    lhs.client(),
                    CubeCount::Static((row_blocks * col_blocks) as u32, batch as u32, 1),
                    CubeDim::new_1d(shape.units() as u32),
                    shape.tn,
                    lhs.arg(),
                    rhs.arg(),
                    out.arg(),
                    m,
                    n / shape.tn,
                    k,
                    lhs_batch_stride,
                    rhs_batch_stride / shape.tn,
                    col_blocks,
                    shape.bm,
                    shape.bn,
                    shape.bk,
                    shape.tm,
                    shape.tn,
                    shape.units(),
                );
            }
        }
        Plan::BlockT(shape, lhs_t, rhs_t) => {
            let row_blocks = m.div_ceil(shape.bm);
            let col_blocks = n.div_ceil(shape.bn);
            crate::backend::count_launch();
            unsafe {
                matmul_block_tiled_t_kernel::launch_unchecked::<E, R>(
                    lhs.client(),
                    CubeCount::Static((row_blocks * col_blocks) as u32, batch as u32, 1),
                    CubeDim::new_1d(shape.units() as u32),
                    lhs.arg(),
                    rhs.arg(),
                    out.arg(),
                    m,
                    n,
                    k,
                    lhs_batch_stride,
                    rhs_batch_stride,
                    col_blocks,
                    shape.bm,
                    shape.bn,
                    shape.bk,
                    shape.tm,
                    shape.tn,
                    shape.units(),
                    lhs_t,
                    rhs_t,
                );
            }
        }
        Plan::BlockV(shape) => {
            let line = shape.tn;
            let row_blocks = m.div_ceil(shape.bm);
            let col_blocks = n.div_ceil(shape.bn);
            crate::backend::count_launch();
            unsafe {
                matmul_block_tiled_vec_kernel::launch_unchecked::<E, R>(
                    lhs.client(),
                    CubeCount::Static((row_blocks * col_blocks) as u32, batch as u32, 1),
                    CubeDim::new_1d(shape.units() as u32),
                    line,
                    lhs.arg(),
                    rhs.arg(),
                    out.arg(),
                    m,
                    n / line,
                    k / line,
                    lhs_batch_stride / line,
                    rhs_batch_stride / line,
                    col_blocks,
                    shape.bm,
                    shape.bn,
                    shape.bk,
                    shape.tm,
                    shape.tn,
                    shape.units(),
                );
            }
        }
        Plan::PlaneDot => {
            let line = line_size_for::<R, E>(lhs.client(), k);
            let plane = lhs.client().properties().hardware.plane_size_max as usize;
            let cube_dim = CubeDim::new_1d((plane * DOT_PLANES) as u32);
            let count =
                crate::backend::cube_count_for(batch * m * n * plane, cube_dim.num_elems());
            crate::backend::count_launch();
            unsafe {
                matmul_plane_dot_kernel::launch_unchecked::<E, R>(
                    lhs.client(),
                    count,
                    cube_dim,
                    line,
                    lhs.arg(),
                    rhs.arg(),
                    out.arg(),
                    m,
                    n,
                    k / line,
                    lhs_batch_stride / line,
                    rhs_batch_stride / line,
                );
            }
        }
        Plan::Tiled => {
            let cube_count = CubeCount::Static(
                n.div_ceil(TILE) as u32,
                m.div_ceil(TILE) as u32,
                batch as u32,
            );
            unsafe {
                matmul_tiled_kernel::launch_unchecked::<E, R>(
                    lhs.client(),
                    cube_count,
                    CubeDim::new_2d(TILE as u32, TILE as u32),
                    lhs.arg(),
                    rhs.arg(),
                    out.arg(),
                    m,
                    n,
                    k,
                    lhs_batch_stride,
                    rhs_batch_stride,
                    TILE,
                );
            }
        }
    }
}

/// Launch `plan` for a possibly-transposed problem, staging the transpose first if
/// the plan has no transposed form.
///
/// Staging is not always the loser. The transposed kernel has no row-tiled
/// counterpart, and on the scan's batch of tiny products — 512 separate 64x64x64
/// adjoints — a block of 256 units spends most of its life at barriers: 116 GFLOP/s
/// against 355 for the row-tiled kernel on a materialised transpose, which pays for
/// the copy several times over. So staging is a candidate the tuner gets to weigh
/// like any other, with the copy included in what it measures.
#[allow(clippy::too_many_arguments)]
fn launch_plan<R: Runtime, E: FloatElem>(
    plan: Plan,
    lhs: &Tensor<R, E>,
    rhs: &Tensor<R, E>,
    out: &Tensor<R, E>,
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
    lhs_batch_stride: usize,
    rhs_batch_stride: usize,
    lhs_t: bool,
    rhs_t: bool,
) {
    if (lhs_t || rhs_t) && !matches!(plan, Plan::BlockT(..) | Plan::PlaneDot) {
        let staged_lhs = if lhs_t {
            transpose_batched(lhs, batch, k, m, lhs_batch_stride)
        } else {
            lhs.clone()
        };
        let staged_rhs = if rhs_t {
            transpose_batched(rhs, batch, n, k, rhs_batch_stride)
        } else {
            rhs.clone()
        };
        launch_matmul(
            plan,
            &staged_lhs,
            &staged_rhs,
            out,
            batch,
            m,
            n,
            k,
            lhs_batch_stride,
            rhs_batch_stride,
        );
        return;
    }
    launch_matmul(
        plan, lhs, rhs, out, batch, m, n, k, lhs_batch_stride, rhs_batch_stride,
    );
}

/// Timed runs of each candidate before one is chosen.
///
/// Each probe is a launch and a synchronisation, so the cost is `PROBES` times the
/// number of candidates, once per distinct problem shape. Five is enough that a
/// single scheduling hiccup does not decide the winner.
const PROBES: usize = 5;

/// Winning plan per problem shape and transposition, measured once and remembered.
type TuneKey = (usize, usize, usize, usize, bool, bool);
static TUNED: std::sync::OnceLock<std::sync::Mutex<std::collections::HashMap<TuneKey, Plan>>> =
    std::sync::OnceLock::new();

/// Which plan to use for this problem, measuring if it has not been seen before.
///
/// No kernel and no tiling shape dominates. Measured on an RDNA3.5 iGPU in GFLOP/s:
///
/// | shape | row-tiled | block 128x64 | block 64x64 |
/// |---|---|---|---|
/// | 1024³ | 724 | **953** | 624 |
/// | 2048x4640x512 (input projection) | 629 | **898** | 581 |
/// | 512x4640x2048 (its weight adjoint) | 300 | **858** | 498 |
/// | 2048x512x1024 (output projection) | 699 | **875** | 647 |
/// | 512 batched 64³ (the scan) | 346 | 383 | **472** |
///
/// The pattern is legible — a tall block wants rows to fill it, and the scan's
/// products only have 64 — but every number in that table is a property of one
/// device's cache and register file. Rather than encode a rule, the problem is timed
/// once, every applicable way, and the winner is cached for the process. A training
/// run issues a few dozen distinct shapes and repeats them every step, so the probe
/// amortises to nothing, and a device with a different register file simply produces
/// a different table.
#[allow(clippy::too_many_arguments)]
fn tuned_plan<R: Runtime, E: FloatElem>(
    lhs: &Tensor<R, E>,
    rhs: &Tensor<R, E>,
    out: &Tensor<R, E>,
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
    lhs_batch_stride: usize,
    rhs_batch_stride: usize,
    lhs_t: bool,
    rhs_t: bool,
) -> Plan {
    let key = (batch, m, n, k, lhs_t, rhs_t);
    let cache = TUNED.get_or_init(Default::default);
    if let Some(found) = cache.lock().expect("matmul tuning cache").get(&key) {
        return *found;
    }

    let mut candidates: Vec<Plan> = Vec::with_capacity(2 * BLOCK_CANDIDATES.len() + 2);
    if lhs_t || rhs_t {
        for shape in BLOCK_CANDIDATES {
            candidates.push(Plan::BlockT(shape, lhs_t, rhs_t));
        }
    }
    // `dA = G Bᵀ` with a small output rectangle is a bundle of long dot products,
    // not a tileable matmul; the plane-per-element kernel needs both operands
    // contiguous along `k`, which is exactly that transposition.
    if !lhs_t && rhs_t && m * n <= 1024 {
        candidates.push(Plan::PlaneDot);
    }
    // Always in the running: with a transposed problem these stage the transpose
    // first, and `launch_plan` prices that in.
    candidates.push(Plan::RowTiled);
    // The block-tiled kernel reads `rhs` and writes the output as whole vectors, so
    // its width has to divide `n`; the vector-staged variant asks the same of `k`.
    // Where both qualify only the vector-staged one runs: measured shape by shape
    // on the training step's whole repertoire, it beat or tied the scalar-staged
    // kernel every time, and a candidate that never wins is pure probe cost.
    for shape in BLOCK_CANDIDATES {
        if n.is_multiple_of(shape.tn) {
            if k.is_multiple_of(shape.tn) && shape.stages_evenly_vec() {
                candidates.push(Plan::BlockV(shape));
            } else {
                candidates.push(Plan::Block(shape));
            }
        }
    }

    // Warm every candidate first, so no one of them pays for compilation or for the
    // first allocation of its output.
    for candidate in &candidates {
        launch_plan(
            *candidate, lhs, rhs, out, batch, m, n, k, lhs_batch_stride, rhs_batch_stride,
            lhs_t, rhs_t,
        );
    }
    let _ = cubecl::future::block_on(lhs.client().sync());

    // Set `MAMBA3_TUNE_CHECK` to verify every candidate against the simple kernel
    // before any of them can win. The tuner's one failure mode is a kernel that is
    // wrong *and* fast — a staging bug that leaves shared memory stale looks like a
    // speedup — and ordinary tests only assert the winner. Running a real model
    // under this flag checks every candidate on every shape that model issues, with
    // its real operands. Tolerance is relative to `k`: the kernels sum the same
    // products in different orders, and a longer sum accumulates more rounding.
    if std::env::var_os("MAMBA3_TUNE_CHECK").is_some() {
        launch_plan(
            Plan::Simple, lhs, rhs, out, batch, m, n, k, lhs_batch_stride, rhs_batch_stride,
            lhs_t, rhs_t,
        );
        let _ = cubecl::future::block_on(lhs.client().sync());
        let want = out.to_f32();
        let scale = want.iter().fold(1.0_f32, |a, v| a.max(v.abs()));
        let tol = scale * 1e-5 * (k as f32).sqrt().max(1.0);
        for candidate in &candidates {
            launch_plan(
                *candidate, lhs, rhs, out, batch, m, n, k, lhs_batch_stride,
                rhs_batch_stride, lhs_t, rhs_t,
            );
            let _ = cubecl::future::block_on(lhs.client().sync());
            let got = out.to_f32();
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                assert!(
                    (g - w).abs() <= tol,
                    "matmul candidate {candidate:?} disagrees with Simple on \
                     {key:?} at element {i}: {g} vs {w} (tol {tol})"
                );
            }
        }
    }

    // Then time them round-robin rather than one at a time, keeping each one's best.
    // This matters on a machine that is doing anything else: measuring candidate A
    // five times and then candidate B five times gives whichever of them happened to
    // coincide with a busy moment a permanent handicap, and the choice is cached for
    // the process. Interleaving makes a transient spike cost every candidate a
    // little instead of one candidate everything.
    let mut times = vec![f64::INFINITY; candidates.len()];
    for _ in 0..PROBES {
        for (candidate, slot) in candidates.iter().zip(&mut times) {
            let start = std::time::Instant::now();
            launch_plan(
                *candidate, lhs, rhs, out, batch, m, n, k, lhs_batch_stride, rhs_batch_stride,
                lhs_t, rhs_t,
            );
            let _ = cubecl::future::block_on(lhs.client().sync());
            *slot = slot.min(start.elapsed().as_secs_f64());
        }
    }
    let (winner, best_time) = times
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.total_cmp(b.1))
        .expect("at least one candidate");
    let best = candidates[winner];
    let best_time = *best_time;

    // Set `MAMBA3_TUNE_LOG` to see what was chosen and what it achieved. Worth doing
    // once on a new device: the shapes a model issues are not obvious from its
    // configuration, and a shape that lands far below the others is a lead.
    if let Some(level) = std::env::var_os("MAMBA3_TUNE_LOG") {
        let flops = 2.0 * (batch * m * n * k) as f64;
        eprintln!("tune {key:?} -> {best:?}  ({:.0} GFLOP/s)", flops / best_time / 1e9);
        if level == "2" {
            for (candidate, time) in candidates.iter().zip(&times) {
                eprintln!("     {:>6.0} GFLOP/s  {candidate:?}", flops / time / 1e9);
            }
        }
    }
    cache.lock().expect("matmul tuning cache").insert(key, best);
    best
}

/// Raw 3-D matmul: `[batch, m, k] @ [batch, k, n] -> [batch, m, n]`.
///
/// A batch stride of `0` broadcasts that operand across the batch.
pub fn matmul_3d<R: Runtime, E: FloatElem>(
    lhs: &Tensor<R, E>,
    rhs: &Tensor<R, E>,
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
    lhs_batch_stride: usize,
    rhs_batch_stride: usize,
) -> Tensor<R, E> {
    matmul_3d_t(
        lhs,
        rhs,
        batch,
        m,
        n,
        k,
        lhs_batch_stride,
        rhs_batch_stride,
        false,
        false,
    )
}

/// [`matmul_3d`], with either operand read as though it were transposed.
///
/// `lhs_t` means `lhs` is stored `[batch, k, m]` and used as `[batch, m, k]`;
/// `rhs_t` means `rhs` is stored `[batch, n, k]` and used as `[batch, k, n]`. The
/// element count and the batch strides are the same either way, because a transpose
/// does not change how many numbers there are — only which one a given `(i, p)` maps
/// to, which is a compile-time choice inside the kernel.
///
/// Runtimes with no shared memory worth staging through, and callers that have pinned
/// a specific non-block kernel, materialise the transpose and take the ordinary path.
/// That is exactly what every caller used to do.
#[allow(clippy::too_many_arguments)]
pub fn matmul_3d_t<R: Runtime, E: FloatElem>(
    lhs: &Tensor<R, E>,
    rhs: &Tensor<R, E>,
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
    lhs_batch_stride: usize,
    rhs_batch_stride: usize,
    lhs_t: bool,
    rhs_t: bool,
) -> Tensor<R, E> {
    let out = Tensor::empty(Shape::new(vec![batch, m, n]), lhs.device());
    if out.is_empty() {
        return out;
    }

    // A hardware plane means a GPU-like machine: many units per cube, high latency to
    // memory, registers to spare, and a cheap `sync_cube`. Without one, on CubeCL's
    // CPU runtime, a barrier is an expensive emulation and shared memory is just
    // another array — the barrier-free kernel wins there by four orders of magnitude
    // and there is nothing to tune.
    let gpu = lhs.client().properties().hardware.plane_size_max > 1;
    let requested = default_kernel();
    let plan = match requested {
        MatmulKernel::Simple => Plan::Simple,
        MatmulKernel::RowTiled => Plan::RowTiled,
        MatmulKernel::Tiled => Plan::Tiled,
        MatmulKernel::BlockTiled if lhs_t || rhs_t => Plan::BlockT(BLOCK_TALL, lhs_t, rhs_t),
        MatmulKernel::BlockTiled
            if n.is_multiple_of(BLOCK_TALL.tn) && k.is_multiple_of(BLOCK_TALL.tn) =>
        {
            Plan::BlockV(BLOCK_TALL)
        }
        MatmulKernel::BlockTiled if n.is_multiple_of(BLOCK_TALL.tn) => Plan::Block(BLOCK_TALL),
        MatmulKernel::BlockTiled => Plan::RowTiled,
        MatmulKernel::Auto if !gpu => Plan::Simple,
        MatmulKernel::Auto => tuned_plan(
            lhs,
            rhs,
            &out,
            batch,
            m,
            n,
            k,
            lhs_batch_stride,
            rhs_batch_stride,
            lhs_t,
            rhs_t,
        ),
    };

    launch_plan(
        plan, lhs, rhs, &out, batch, m, n, k, lhs_batch_stride, rhs_batch_stride, lhs_t, rhs_t,
    );
    out
}

/// Materialise `[batch, rows, cols] -> [batch, cols, rows]`, honouring a broadcast
/// batch stride of `0`.
fn transpose_batched<R: Runtime, E: FloatElem>(
    src: &Tensor<R, E>,
    batch: usize,
    rows: usize,
    cols: usize,
    batch_stride: usize,
) -> Tensor<R, E> {
    let effective = if batch_stride == 0 { 1 } else { batch };
    let view = src
        .reshape(Shape::new(vec![effective, rows, cols]))
        .expect("transpose view");
    crate::tensor::ops::movement::transpose(&view).expect("transpose")
}

/// Batched matmul with broadcasting over leading dimensions.
///
/// * `lhs`: `[..., m, k]`
/// * `rhs`: `[..., k, n]`
///
/// The leading dimensions are broadcast against each other, matching NumPy's
/// `matmul` semantics. Vector operands (rank 1) are not auto-promoted; reshape
/// explicitly so the intent stays visible at the call site.
pub fn matmul<R: Runtime, E: FloatElem>(
    lhs: &Tensor<R, E>,
    rhs: &Tensor<R, E>,
) -> Result<Tensor<R, E>> {
    matmul_t(lhs, rhs, false, false)
}

/// `lhs @ rhsᵀ`, contracting the trailing axis of both operands.
///
/// This is the shape the adjoint of a matrix product wants for its left operand,
/// and taking it directly is what lets the backward pass skip materialising `rhsᵀ`.
pub fn matmul_nt<R: Runtime, E: FloatElem>(
    lhs: &Tensor<R, E>,
    rhs: &Tensor<R, E>,
) -> Result<Tensor<R, E>> {
    matmul_t(lhs, rhs, false, true)
}

/// `lhsᵀ @ rhs`, contracting the *leading* matrix axis of both operands.
///
/// The adjoint's right operand, for the same reason as [`matmul_nt`].
pub fn matmul_tn<R: Runtime, E: FloatElem>(
    lhs: &Tensor<R, E>,
    rhs: &Tensor<R, E>,
) -> Result<Tensor<R, E>> {
    matmul_t(lhs, rhs, true, false)
}

/// The general form: `op(lhs) @ op(rhs)`, where `op` is a transpose of the trailing
/// two axes when the corresponding flag is set.
pub fn matmul_t<R: Runtime, E: FloatElem>(
    lhs: &Tensor<R, E>,
    rhs: &Tensor<R, E>,
    lhs_t: bool,
    rhs_t: bool,
) -> Result<Tensor<R, E>> {
    if lhs.rank() < 2 || rhs.rank() < 2 {
        return Err(Error::shape(format!(
            "matmul needs rank >= 2 operands, got {} and {}",
            lhs.shape, rhs.shape
        )));
    }
    // Logical dimensions after the notional transpose; the physical matrix size is
    // `m * k` and `k * n` either way, which is what the batch strides count.
    let (m, k1) = if lhs_t {
        (lhs.shape.dim_from_end(0), lhs.shape.dim_from_end(1))
    } else {
        (lhs.shape.dim_from_end(1), lhs.shape.dim_from_end(0))
    };
    let (k2, n) = if rhs_t {
        (rhs.shape.dim_from_end(0), rhs.shape.dim_from_end(1))
    } else {
        (rhs.shape.dim_from_end(1), rhs.shape.dim_from_end(0))
    };
    if k1 != k2 {
        return Err(Error::shape(format!(
            "matmul inner dimensions disagree: {} vs {}",
            lhs.shape, rhs.shape
        )));
    }

    let lhs_batch = Shape::new(lhs.dims()[..lhs.rank() - 2].to_vec());
    let rhs_batch = Shape::new(rhs.dims()[..rhs.rank() - 2].to_vec());
    let batch_shape = Shape::broadcast(&lhs_batch, &rhs_batch)?;
    let batch = batch_shape.num_elements();

    // A matching batch layout, or a single operand broadcast across the batch,
    // both avoid materialising anything. Anything else is expanded first.
    let lhs_stride = if lhs_batch.num_elements() == batch {
        m * k1
    } else if lhs_batch.num_elements() == 1 {
        0
    } else {
        usize::MAX
    };
    let rhs_stride = if rhs_batch.num_elements() == batch {
        k1 * n
    } else if rhs_batch.num_elements() == 1 {
        0
    } else {
        usize::MAX
    };

    let (lhs, lhs_stride) = if lhs_stride == usize::MAX {
        let target = {
            let mut d = batch_shape.dims().to_vec();
            d.push(lhs.shape.dim_from_end(1));
            d.push(lhs.shape.dim_from_end(0));
            Shape::new(d)
        };
        (crate::tensor::ops::elemwise::expand(lhs, &target)?, m * k1)
    } else {
        (lhs.clone(), lhs_stride)
    };
    let (rhs, rhs_stride) = if rhs_stride == usize::MAX {
        let target = {
            let mut d = batch_shape.dims().to_vec();
            d.push(rhs.shape.dim_from_end(1));
            d.push(rhs.shape.dim_from_end(0));
            Shape::new(d)
        };
        (crate::tensor::ops::elemwise::expand(rhs, &target)?, k1 * n)
    } else {
        (rhs.clone(), rhs_stride)
    };

    let out = matmul_3d_t(
        &lhs, &rhs, batch, m, n, k1, lhs_stride, rhs_stride, lhs_t, rhs_t,
    );

    let mut out_dims = batch_shape.dims().to_vec();
    out_dims.push(m);
    out_dims.push(n);
    out.reshape(Shape::new(out_dims))
}
