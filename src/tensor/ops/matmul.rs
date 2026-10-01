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
    /// Shared-memory block tiles whose inner product runs on the matrix cores.
    ///
    /// Requires a reduced-precision mode *and* a device that implements the
    /// fragment shape; where either is missing this falls back to
    /// [`MatmulKernel::RowTiled`] rather than failing, so asking for it is
    /// always safe. Mostly useful for forcing the path under test — the tuner
    /// offers it on its own merits when the mode is on.
    Cmma,
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
            MatmulKernel::Cmma => 5,
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
        5 => MatmulKernel::Cmma,
        _ => MatmulKernel::Auto,
    }
}

/// Storage precision for matmul operands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatmulPrecision {
    /// Operands as the caller stored them. The default.
    F32,
    /// Round `f32` operands to `bf16` once per call and read the narrow copy;
    /// every product is still accumulated in `f32`.
    Bf16,
    /// The same, in `f16`.
    ///
    /// `f16` keeps three more mantissa bits than `bf16` and five fewer exponent
    /// bits, so it is the more accurate of the two per product and the one that
    /// can overflow: its largest value is 65504, which activations reach far
    /// sooner than `bf16`'s `f32`-sized range. That is why `Bf16` is the
    /// training recommendation and this is not. It exists because it is what
    /// the *hardware* offers: every tensor-core generation does `f16`, while
    /// `bf16` fragments arrived with Ampere and RDNA3 — a Turing T4, the free
    /// Colab GPU, has the first and not the second. Keeping both means the
    /// matrix-core path can be validated on hardware anyone can reach.
    F16,
}

static PRECISION: AtomicU8 = AtomicU8::new(0);

/// Choose the storage precision used by [`matmul`] from now on.
///
/// Unlike [`set_default_kernel`] this is a *semantic* knob, which is why it is a
/// mode and never a tuner candidate: the crate's promise that every kernel
/// computes the same product to within floating-point associativity holds within
/// a mode, not across modes. `Bf16` keeps `f32` master weights, gradients and
/// accumulation — only what the product kernels *read* is rounded, which is the
/// mixed-precision training recipe and needs no loss scaling. It halves the
/// bytes a matmul moves, which is the ceiling the memory-bound products are at,
/// and it is the storage type matrix cores want. The default is `F32`; nothing
/// changes unless a caller opts in.
pub fn set_matmul_precision(precision: MatmulPrecision) {
    PRECISION.store(
        match precision {
            MatmulPrecision::F32 => 0,
            MatmulPrecision::Bf16 => 1,
            MatmulPrecision::F16 => 2,
        },
        Ordering::Relaxed,
    );
}

/// Whether `device`'s backend can compile kernels that read `precision`.
///
/// The mode is a global, but whether it can be honoured is a property of the
/// runtime, and getting that wrong is not a graceful failure: WGSL has no `bf16`
/// type at all, so a kernel asking for one panics inside the compiler, on the
/// device thread, once per launch. A run that set the mode and walked away comes
/// back to tens of thousands of identical panics and no result.
///
/// Answered by the runtime's own per-type capability table
/// ([`crate::backend::supports_dtype`]), not by the device's name. Verified on the
/// CPU runtime (both narrow types) and on wgpu compiling WGSL on Apple M1 (`f16`
/// yes, `bf16` no); `tests/mixed_precision.rs::the_capability_query_matches_what_the_kernels_do`
/// checks the answer against what the kernels actually do on whichever backend
/// runs it.
pub fn supports_matmul_precision<R: Runtime>(
    device: &crate::backend::Device<R>,
    precision: MatmulPrecision,
) -> bool {
    crate::backend::supports_dtype(device, precision_dtype(precision))
}

/// The element type a matmul reads its operands at under `precision`.
fn precision_dtype(precision: MatmulPrecision) -> crate::backend::DType {
    match precision {
        MatmulPrecision::F32 => crate::backend::DType::F32,
        MatmulPrecision::Bf16 => crate::backend::DType::BF16,
        MatmulPrecision::F16 => crate::backend::DType::F16,
    }
}

/// The refusal [`try_set_matmul_precision`] and the matmul entry points share.
fn check_matmul_precision<R: Runtime>(
    device: &crate::backend::Device<R>,
    precision: MatmulPrecision,
) -> crate::error::Result<()> {
    if supports_matmul_precision(device, precision) {
        return Ok(());
    }
    Err(crate::error::Error::config(format!(
        "the {} backend cannot compile {precision:?} matrix products; \
         its shader language has no such type. Use F32 or F16 here, or build \
         for a backend that does: cuda, hip, or wgpu through spirv or msl",
        device.name(),
    )))
}

/// Set the precision, refusing one this backend cannot compile.
///
/// [`set_matmul_precision`] is a plain store and stays one — it is runtime-agnostic
/// and some callers know what they are doing. This is the checked door, and the one
/// a user-facing surface should go through.
pub fn try_set_matmul_precision<R: Runtime>(
    device: &crate::backend::Device<R>,
    precision: MatmulPrecision,
) -> crate::error::Result<()> {
    check_matmul_precision(device, precision)?;
    set_matmul_precision(precision);
    Ok(())
}

/// The storage precision [`matmul`] is using.
pub fn matmul_precision() -> MatmulPrecision {
    match PRECISION.load(Ordering::Relaxed) {
        1 => MatmulPrecision::Bf16,
        2 => MatmulPrecision::F16,
        _ => MatmulPrecision::F32,
    }
}

/// Parse a kernel name, case-insensitively: `auto`, `simple`, `row_tiled`,
/// `tiled`, `block_tiled` or `cmma`.
///
/// Shared by [`try_set_kernel_from_env`] and the Python setter, like
/// [`parse_matmul_precision`].
pub fn parse_matmul_kernel(value: &str) -> Option<MatmulKernel> {
    match value.to_ascii_lowercase().as_str() {
        "auto" => Some(MatmulKernel::Auto),
        "simple" => Some(MatmulKernel::Simple),
        "row_tiled" => Some(MatmulKernel::RowTiled),
        "tiled" => Some(MatmulKernel::Tiled),
        "block_tiled" => Some(MatmulKernel::BlockTiled),
        "cmma" => Some(MatmulKernel::Cmma),
        _ => None,
    }
}

/// The name [`parse_matmul_kernel`] accepts for `kernel`.
pub fn matmul_kernel_name(kernel: MatmulKernel) -> &'static str {
    match kernel {
        MatmulKernel::Auto => "auto",
        MatmulKernel::Simple => "simple",
        MatmulKernel::RowTiled => "row_tiled",
        MatmulKernel::Tiled => "tiled",
        MatmulKernel::BlockTiled => "block_tiled",
        MatmulKernel::Cmma => "cmma",
    }
}

/// Read `MAMBA3_MATMUL_KERNEL` and set the default kernel, refusing a name that is
/// not one. Unset leaves the default (`auto`) alone.
///
/// Why pin one: on a GPU, `auto` times the candidate kernels for every new shape
/// and keeps the fastest *for this process*. The candidates compute the same
/// product in different summation orders, so two processes can train a few ulp
/// apart. A run that must be bit-reproducible across processes — a restore from a
/// full checkpoint compared against a run that never stopped — pins a kernel in
/// both.
pub fn try_set_kernel_from_env() -> crate::error::Result<()> {
    let Ok(value) = std::env::var("MAMBA3_MATMUL_KERNEL") else {
        return Ok(());
    };
    let kernel = parse_matmul_kernel(&value).ok_or_else(|| {
        crate::error::Error::config(format!(
            "MAMBA3_MATMUL_KERNEL={value:?} is not a kernel; expected 'auto', 'simple', \
             'row_tiled', 'tiled', 'block_tiled' or 'cmma' (case-insensitive)"
        ))
    })?;
    set_default_kernel(kernel);
    Ok(())
}

/// Parse one of the accepted precision spellings, case-insensitively.
///
/// The single place that decides what counts as a valid value, shared by the
/// environment-variable path below and the Python setter (`set_matmul_precision`
/// in `bindings/python/src/lib.rs`), so the two cannot silently drift apart on
/// which strings they accept or how they treat case.
pub fn parse_matmul_precision(value: &str) -> Option<MatmulPrecision> {
    match value.to_ascii_lowercase().as_str() {
        "f32" => Some(MatmulPrecision::F32),
        "bf16" => Some(MatmulPrecision::Bf16),
        "f16" => Some(MatmulPrecision::F16),
        _ => None,
    }
}

/// Read `MAMBA3_MATMUL_PRECISION` (`f32`, `bf16` or `f16`, case-insensitive) and
/// set the mode, checked against a default device's capability before anything
/// is stored.
///
/// An **absent** variable returns `Ok(())` immediately and touches no device at
/// all — deferred backend initialisation stays deferred, so merely importing a
/// configuration surface that happens to call this does not force one. A
/// **present** variable is validated eagerly, right here, before the caller's
/// own first device operation: an unrecognised value or one this backend cannot
/// compile is an error, not a silent fallback to `F32` or a value quietly
/// different from what was asked for. That is deliberate — an explicitly
/// requested precision must not silently change — which is why this differs
/// from the old unchecked `set_matmul_precision`-under-a-`match` this replaces:
/// that stored an unsupported mode first and left validation to whatever ran a
/// kernel under it, which on WGSL means a worker-thread abort inside the shader
/// compiler rather than a message anyone could act on.
pub fn try_set_precision_from_env<R: Runtime>() -> crate::error::Result<()> {
    let Ok(value) = std::env::var("MAMBA3_MATMUL_PRECISION") else {
        return Ok(());
    };
    let precision = parse_matmul_precision(&value).ok_or_else(|| {
        crate::error::Error::config(format!(
            "MAMBA3_MATMUL_PRECISION={value:?} is not a recognised precision; \
             expected 'f32', 'bf16' or 'f16' (case-insensitive)"
        ))
    })?;
    let device = crate::backend::Device::<R>::default();
    try_set_matmul_precision(&device, precision)
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
///
/// Every kernel here is generic over a storage element `FS` beside the compute
/// element `F`: operands are read as `FS` and widened to `F` at the multiply,
/// which is the whole of the mixed-precision mode. The ordinary path instantiates
/// `FS = F`, where the widening cast is the identity and nothing changes.
#[cube(launch_unchecked)]
fn matmul_simple_kernel<FS: Float + CubeElement, F: Float + CubeElement, N: Size>(
    lhs: &Array<FS>,
    rhs: &Array<Vector<FS, N>>,
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
            acc += Vector::<F, N>::new(F::cast_from(lhs[lhs_base + p]))
                * Vector::<F, N>::cast_from(rhs[rhs_base + p * n_lines]);
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
fn matmul_row_tiled_kernel<FS: Float + CubeElement, F: Float + CubeElement, N: Size>(
    lhs: &Array<FS>,
    rhs: &Array<Vector<FS, N>>,
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
            let rv = Vector::<F, N>::cast_from(rhs[rhs_base + p * n_lines]);
            #[unroll]
            for i in 0..rows {
                acc[i] += Vector::<F, N>::new(F::cast_from(lhs[row_offset[i] + p])) * rv;
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
const BLOCK_CANDIDATES: [BlockShape; 6] = [
    BlockShape {
        bm: 128,
        bn: 64,
        bk: 16,
        tm: 8,
        tn: 4,
    },
    BlockShape {
        bm: 128,
        bn: 64,
        bk: 16,
        tm: 16,
        tn: 4,
    },
    BlockShape {
        bm: 128,
        bn: 128,
        bk: 16,
        tm: 8,
        tn: 4,
    },
    BlockShape {
        bm: 128,
        bn: 64,
        bk: 32,
        tm: 8,
        tn: 4,
    },
    BlockShape {
        bm: 128,
        bn: 32,
        bk: 16,
        tm: 8,
        tn: 4,
    },
    BlockShape {
        bm: 64,
        bn: 64,
        bk: 16,
        tm: 4,
        tn: 4,
    },
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
fn matmul_block_tiled_t_kernel<FS: Float + CubeElement, F: Float + CubeElement>(
    lhs: &Array<FS>,
    rhs: &Array<FS>,
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
    let mut sa = SharedMemory::<FS>::new(bk * (bm + 1));
    let mut sb = SharedMemory::<FS>::new(bk * (bn + 1));

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
    let mut a_pref = Array::<FS>::new(bm * bk / units);
    let mut b_pref = Array::<FS>::new(bk * bn / units);

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
        let mut v = FS::new(0.0_f32);
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
        let mut v = FS::new(0.0_f32);
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
                let mut v = FS::new(0.0_f32);
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
                let mut v = FS::new(0.0_f32);
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
                a_reg[i] = F::cast_from(sa[kk * a_stride + tile_row + i]);
            }
            #[unroll]
            for j in 0..tn {
                b_reg[j] = F::cast_from(sb[kk * b_stride + tile_col + j]);
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
fn matmul_block_tiled_kernel<FS: Float + CubeElement, F: Float + CubeElement, N: Size>(
    lhs: &Array<FS>,
    rhs: &Array<Vector<FS, N>>,
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
    let mut sa = SharedMemory::<FS>::new(bk * (bm + 1));
    let mut sb = SharedMemory::<Vector<FS, N>>::new(bk * (bn / tn));

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
    let mut a_pref = Array::<FS>::new(bm * bk / units);
    let mut b_pref = Array::<Vector<FS, N>>::new(bk * bn / tn / units);

    #[unroll]
    for t in 0..(bm * bk / units) {
        let row = a_row + t * (units / bk);
        let global_row = row0 + row;
        let mut v = FS::new(0.0_f32);
        if global_row < m && a_col < k {
            v = lhs[lhs_base + global_row * k + a_col];
        }
        a_pref[t] = v;
    }
    #[unroll]
    for t in 0..(bk * bn / tn / units) {
        let kk = b_row + t * (units / block_lines);
        let global_line = col0_lines + b_line;
        let mut v = Vector::<FS, N>::new(FS::new(0.0_f32));
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
                let mut v = FS::new(0.0_f32);
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
                let mut v = Vector::<FS, N>::new(FS::new(0.0_f32));
                if global_k < k && global_line < n_lines {
                    v = rhs[rhs_base + global_k * n_lines + global_line];
                }
                b_pref[t] = v;
            }
        }

        #[unroll]
        for kk in 0..bk {
            let b_vec = Vector::<F, N>::cast_from(sb[kk * block_lines + tile_line]);
            #[unroll]
            for i in 0..tm {
                acc[i] +=
                    Vector::<F, N>::new(F::cast_from(sa[kk * a_stride + tile_row + i])) * b_vec;
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
fn matmul_tiled_kernel<FS: Float + CubeElement, F: Float + CubeElement>(
    lhs: &Array<FS>,
    rhs: &Array<FS>,
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

    let mut tile_a = SharedMemory::<FS>::new(tile * tile);
    let mut tile_b = SharedMemory::<FS>::new(tile * tile);

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
            tile_a[ty * tile + tx] = FS::new(0.0_f32);
        }
        if col < n && b_row < k {
            tile_b[ty * tile + tx] = rhs[rhs_base + b_row * n + col];
        } else {
            tile_b[ty * tile + tx] = FS::new(0.0_f32);
        }

        sync_cube();

        for i in 0..tile {
            acc += F::cast_from(tile_a[ty * tile + i]) * F::cast_from(tile_b[i * tile + tx]);
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
fn matmul_block_tiled_vec_kernel<FS: Float + CubeElement, F: Float + CubeElement, N: Size>(
    lhs: &Array<Vector<FS, N>>,
    rhs: &Array<Vector<FS, N>>,
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
    let mut sa = SharedMemory::<FS>::new(bk * (bm + 2));
    let mut sb = SharedMemory::<Vector<FS, N>>::new(bk * (bn / tn));

    let mut acc = Array::<Vector<F, N>>::new(tm);
    #[unroll]
    for i in 0..tm {
        acc[i] = Vector::<F, N>::new(F::new(0.0_f32));
    }

    // Prefetched next tiles, exactly as in the plain kernel.
    let mut a_pref = Array::<Vector<FS, N>>::new(bm * bk / tn / units);
    let mut b_pref = Array::<Vector<FS, N>>::new(bk * bn / tn / units);

    #[unroll]
    for t in 0..(bm * bk / tn / units) {
        let row = a_row + t * (units / bk_lines);
        let global_row = row0 + row;
        let mut v = Vector::<FS, N>::new(FS::new(0.0_f32));
        if global_row < m && a_col < k_lines {
            v = lhs[lhs_base + global_row * k_lines + a_col];
        }
        a_pref[t] = v;
    }
    #[unroll]
    for t in 0..(bk * bn / tn / units) {
        let kk = b_row + t * (units / block_lines);
        let global_line = col0_lines + b_line;
        let mut v = Vector::<FS, N>::new(FS::new(0.0_f32));
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
                let mut v = Vector::<FS, N>::new(FS::new(0.0_f32));
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
                let mut v = Vector::<FS, N>::new(FS::new(0.0_f32));
                if global_k < k_lines * tn && global_line < n_lines {
                    v = rhs[rhs_base + global_k * n_lines + global_line];
                }
                b_pref[t] = v;
            }
        }

        #[unroll]
        for kk in 0..bk {
            let b_vec = Vector::<F, N>::cast_from(sb[kk * block_lines + tile_line]);
            #[unroll]
            for i in 0..tm {
                acc[i] +=
                    Vector::<F, N>::new(F::cast_from(sa[kk * a_stride + tile_row + i])) * b_vec;
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

/// The edge of one cooperative-matrix instruction.
///
/// 16×16×16 is the shape every tensor-core generation supports for the half
/// types — the one entry that is present on Turing, Ampere, Ada, Hopper and
/// RDNA3 alike — so it is the only one this kernel asks for. Larger fragments
/// exist on newer parts and would be a per-device tuner axis of their own;
/// membership in the capability set is checked either way, so a device that
/// lacks even this simply never sees the candidate.
const MMA_TILE: usize = 16;

/// Depth of one k-step for the warp-tiled CMMA kernel.
///
/// 32 halves are 64 bytes per row of `sa`/`sb` staging; two 16-wide
/// `cmma::execute` calls consume them. Deeper steps amortise the two barriers
/// per step over more tensor-core work without blowing shared memory.
const CMMA_BK: usize = 32;

/// Whether [`MatmulKernel::Cmma`] would actually reach the matrix cores on this
/// device under `precision`.
///
/// `false` means the request would fall back to a register kernel — either the
/// mode is `F32` (matrix cores take half-precision operands) or the device does
/// not implement the fragment. Tests and benchmarks use this to skip rather than
/// to assert, since whether a machine has tensor cores is not something the
/// crate gets to decide.
///
/// On the CUDA backend `Bf16` additionally requires `MAMBA3_CMMA_BF16=1`; see
/// [`cmma_supported`] for why the feature table alone cannot be trusted there.
pub fn cmma_available<R: Runtime>(
    device: &crate::backend::Device<R>,
    precision: MatmulPrecision,
) -> bool {
    match precision {
        MatmulPrecision::F32 => false,
        MatmulPrecision::Bf16 => cmma_supported::<R, half::bf16, f32>(device.client()),
        MatmulPrecision::F16 => cmma_supported::<R, half::f16, f32>(device.client()),
    }
}

/// Shared-memory block tiling whose inner product runs on the matrix cores.
///
/// One cube owns a `bm × bn` rectangle of the output and walks `k` in steps of
/// [`CMMA_BK`]. Each plane owns a `16·wm × 16·wn` sub-block of 16×16
/// accumulator fragments held in a comptime [`Sequence`](cubecl::prelude::Sequence):
/// `(wm, wn) = (2, 2)` for the 64×64 and 32×32 blocks (a 32×32 sub-block, 4
/// planes and 1 plane) and `(4, 2)` for 128×128 (a 64×32 sub-block, 8 planes in
/// a 2 rows × 4 columns arrangement). The accumulators live across the whole
/// `k` walk and are written out once.
///
/// Staging is two padded tiles: `sa` is `[bm][BK+8]` (row `i` of A holds `k`
/// values contiguous, padded by 8 halves against bank conflicts) and `sb` is
/// `[BK][bn+8]` (row `kk` of B holds `n` values contiguous, padded the same
/// way). Both fragments are loaded `RowMajor` with those strides, so the
/// staging layout is exactly what `cmma::load` wants.
///
/// Global reads are vectorised along the operand's contiguous axis with
/// `Vector<FS, 8>` (16 bytes): A contiguous along `k` when `lhs_t` is false
/// and along `m` when true, mirrored for B. A vector that would cross the end
/// of its row or of `m`/`n`/`k` falls back to scalar guarded loads (zero
/// outside). Vector loads need 16-byte alignment, so the launch arm only takes
/// the vector path when the contiguous extent and the batch stride are both
/// multiples of 8 (`vec_ok_*`, decided at comptime); otherwise the scalar
/// grid-stride path runs. The same buffer is bound twice — once as
/// `Array<FS>`, once as `Array<Vector<FS, Const<8>>>` — which is how CubeCL
/// reinterprets it, the way [`matmul_block_tiled_vec_kernel`] does.
///
/// The first step's tiles are prefetched into per-unit registers before the
/// loop. Each iteration writes the registers to shared memory, `sync_cube`,
/// issues the next step's global loads into the same registers when there is
/// a next step, runs this step's MMAs from shared memory, and `sync_cube`
/// again — so step `s + 1`'s loads retire behind step `s`'s tensor-core work.
/// The register files hold scalar `FI` elements (the vector paths scatter each
/// 8-wide load into the 8 slots owned by the same vector task) and are sized
/// for 32-lane planes with a bounds guard, so wider planes leave the tail
/// slots idle; the `FI` → `FS` rounding stays at the shared-memory write.
/// Per k-step each plane loads its `wm` A fragments and `wn` B fragments and
/// issues `wm·wn` `cmma::execute` calls. Transposition of either operand is
/// absorbed into the staging index — the same compile-time choice
/// [`matmul_block_tiled_t_kernel`] makes — so one kernel covers all four
/// combinations.
///
/// The epilogue is per-plane: a shared scratch of `planes·256` `f32` (one
/// 16×16 tile per plane, 8 KB for 128×128), so no `bm × bn` scratch and
/// 128×128 fits in 48 KB. Each plane stores one fragment at a time into its
/// own slice (row stride 16), `sync_plane`, and the plane's 32 lanes copy the
/// 256 values to global memory with the bounds guard (lane `j` writes elements
/// `j, j+32, …`, i.e. consecutive lanes write consecutive columns), then
/// `sync_plane` again before the next fragment reuses the slice. The scratch
/// bounce is the known cost the CubeCL manual describes: the cooperative
/// store's lane mapping is opaque, so there is no way to bounds-check a direct
/// global write.
#[cube(launch_unchecked)]
fn matmul_cmma_kernel<FI: Float + CubeElement, FS: Float + CubeElement, F: Float + CubeElement>(
    lhs: &Array<FI>,
    lhs_vec: &Array<Vector<FI, Const<8>>>,
    rhs: &Array<FI>,
    rhs_vec: &Array<Vector<FI, Const<8>>>,
    out: &mut Array<F>,
    m: usize,
    n: usize,
    k: usize,
    lhs_batch_stride: usize,
    rhs_batch_stride: usize,
    col_blocks: usize,
    #[comptime] bm: usize,
    #[comptime] bn: usize,
    #[comptime] wm: usize,
    #[comptime] wn: usize,
    #[comptime] planes: usize,
    #[comptime] tile: usize,
    #[comptime] lhs_t: bool,
    #[comptime] rhs_t: bool,
    #[comptime] vec_ok_lhs: bool,
    #[comptime] vec_ok_rhs: bool,
) {
    let units = CUBE_DIM as usize;
    let unit = UNIT_POS_X as usize;
    let plane_idx = unit / PLANE_DIM as usize;
    let block = CUBE_POS_X as usize;
    let batch = CUBE_POS_Y as usize;
    let row0 = (block / col_blocks) * bm;
    let col0 = (block % col_blocks) * bn;
    let lhs_base = batch * lhs_batch_stride;
    let rhs_base = batch * rhs_batch_stride;

    // Planes tile the block `bn / (16·wn)` across: plane p covers rows
    // `(p / planes_n)·(16·wm)` and columns `(p % planes_n)·(16·wn)`.
    let planes_n = bn / (16 * wn);
    let plane_row0 = (plane_idx / planes_n) * (16 * wm);
    let plane_col0 = (plane_idx % planes_n) * (16 * wn);

    let a_stride = CMMA_BK + 8;
    let b_stride = bn + 8;
    let mut sa = SharedMemory::<FS>::new(bm * (CMMA_BK + 8));
    let mut sb = SharedMemory::<FS>::new(CMMA_BK * (bn + 8));
    // Per-plane epilogue scratch: one 16×16 `f32` tile per plane, reused for
    // each fragment in turn (8 planes → 8 KB for 128×128).
    let mut sp = SharedMemory::<F>::new(planes * 256);

    // The accumulators: `wm·wn` 16×16 fragments per plane, live across the
    // whole `k` walk. A flat comptime sequence, indexed by literals and bare
    // loop variables only — never by computed expressions, which CubeCL does
    // not accept as constant indices.
    let mut acc = Sequence::<cmma::Matrix<F>>::new();
    #[unroll]
    for _ in 0..wm * wn {
        acc.push(cmma::Matrix::<F>::from_value(
            cmma::MatrixIdent::Accumulator,
            tile,
            tile,
            tile,
            cmma::MatrixLayout::Undefined,
            F::new(0.0_f32),
        ));
    }

    // One k-step's staging share per unit, held in registers so the next
    // step's global loads can be issued before this step's MMAs run. Scalar
    // `FI` slots: the 8-wide paths scatter each vector into the 8 slots owned
    // by the same vector task. Sized for 32-lane planes — every candidate
    // divides evenly (16/16 slots for 64×64 and 128×128, 32/32 for 32×32) —
    // with a bounds guard, so wider planes leave the tail slots idle. This
    // kernel only launches where the fragment exists (never on the CPU
    // runtime), so narrower planes need not be covered.
    let mut a_pref = Array::<FI>::new(bm * CMMA_BK / (planes * 32));
    let mut b_pref = Array::<FI>::new(CMMA_BK * bn / (planes * 32));

    // Prologue: step 0's tiles are already in registers before the loop, so
    // the first iteration stores without waiting on global loads.
    let k0 = 0usize;
    // Prefetch A of step `k0` into registers (zero outside m/k).
    if lhs_t {
        // Stored [k][m], contiguous along m.
        if vec_ok_lhs {
            let m_lines = m / 8;
            let lhs_vec_base = batch * (lhs_batch_stride / 8);
            let row0_vec = row0 / 8;
            #[unroll]
            for t in 0..((bm / 8) * CMMA_BK / (planes * 32)) {
                let v = unit + t * units;
                if v < (bm / 8) * CMMA_BK {
                    let vec_row = v % (bm / 8);
                    let kk = v / (bm / 8);
                    let global_k = k0 + kk;
                    let vec_m = row0_vec + vec_row;
                    if global_k < k && vec_m < m_lines {
                        let vv = lhs_vec[lhs_vec_base + global_k * m_lines + vec_m];
                        #[unroll]
                        for j in 0usize..8usize {
                            a_pref[t * 8 + j] = vv[j];
                        }
                    } else {
                        #[unroll]
                        for j in 0usize..8usize {
                            let global_row = row0 + vec_row * 8 + j;
                            let mut sv = FI::new(0.0_f32);
                            if global_k < k && global_row < m {
                                sv = lhs[lhs_base + global_k * m + global_row];
                            }
                            a_pref[t * 8 + j] = sv;
                        }
                    }
                }
            }
        } else {
            #[unroll]
            for t in 0..(bm * CMMA_BK / (planes * 32)) {
                let i = unit + t * units;
                if i < bm * CMMA_BK {
                    let row = i % bm;
                    let kk = i / bm;
                    let global_row = row0 + row;
                    let global_k = k0 + kk;
                    let mut v = FI::new(0.0_f32);
                    if global_row < m && global_k < k {
                        v = lhs[lhs_base + global_k * m + global_row];
                    }
                    a_pref[t] = v;
                }
            }
        }
    } else if vec_ok_lhs {
        // Stored [m][k], contiguous along k.
        let k_lines = k / 8;
        let lhs_vec_base = batch * (lhs_batch_stride / 8);
        let k0_vec = k0 / 8;
        #[unroll]
        for t in 0..(bm * (CMMA_BK / 8) / (planes * 32)) {
            let v = unit + t * units;
            if v < bm * (CMMA_BK / 8) {
                let row = v / (CMMA_BK / 8);
                let vec_col = v % (CMMA_BK / 8);
                let global_row = row0 + row;
                let vec_k = k0_vec + vec_col;
                if global_row < m && vec_k < k_lines {
                    let vv = lhs_vec[lhs_vec_base + global_row * k_lines + vec_k];
                    #[unroll]
                    for j in 0usize..8usize {
                        a_pref[t * 8 + j] = vv[j];
                    }
                } else {
                    #[unroll]
                    for j in 0usize..8usize {
                        let global_k = k0 + vec_col * 8 + j;
                        let mut sv = FI::new(0.0_f32);
                        if global_row < m && global_k < k {
                            sv = lhs[lhs_base + global_row * k + global_k];
                        }
                        a_pref[t * 8 + j] = sv;
                    }
                }
            }
        }
    } else {
        #[unroll]
        for t in 0..(bm * CMMA_BK / (planes * 32)) {
            let i = unit + t * units;
            if i < bm * CMMA_BK {
                let row = i / CMMA_BK;
                let kk = i % CMMA_BK;
                let global_row = row0 + row;
                let global_k = k0 + kk;
                let mut v = FI::new(0.0_f32);
                if global_row < m && global_k < k {
                    v = lhs[lhs_base + global_row * k + global_k];
                }
                a_pref[t] = v;
            }
        }
    }
    // Prefetch B of step `k0` into registers (zero outside k/n).
    if rhs_t {
        // Stored [n][k], contiguous along k.
        if vec_ok_rhs {
            let k_lines = k / 8;
            let rhs_vec_base = batch * (rhs_batch_stride / 8);
            let k0_vec = k0 / 8;
            #[unroll]
            for t in 0..(bn * (CMMA_BK / 8) / (planes * 32)) {
                let v = unit + t * units;
                if v < bn * (CMMA_BK / 8) {
                    let col = v / (CMMA_BK / 8);
                    let vec_kk = v % (CMMA_BK / 8);
                    let global_col = col0 + col;
                    let vec_k = k0_vec + vec_kk;
                    if global_col < n && vec_k < k_lines {
                        let vv = rhs_vec[rhs_vec_base + global_col * k_lines + vec_k];
                        #[unroll]
                        for j in 0usize..8usize {
                            b_pref[t * 8 + j] = vv[j];
                        }
                    } else {
                        #[unroll]
                        for j in 0usize..8usize {
                            let global_k = k0 + vec_kk * 8 + j;
                            let mut sv = FI::new(0.0_f32);
                            if global_col < n && global_k < k {
                                sv = rhs[rhs_base + global_col * k + global_k];
                            }
                            b_pref[t * 8 + j] = sv;
                        }
                    }
                }
            }
        } else {
            #[unroll]
            for t in 0..(CMMA_BK * bn / (planes * 32)) {
                let i = unit + t * units;
                if i < CMMA_BK * bn {
                    let col = i / CMMA_BK;
                    let kk = i % CMMA_BK;
                    let global_col = col0 + col;
                    let global_k = k0 + kk;
                    let mut v = FI::new(0.0_f32);
                    if global_k < k && global_col < n {
                        v = rhs[rhs_base + global_col * k + global_k];
                    }
                    b_pref[t] = v;
                }
            }
        }
    } else if vec_ok_rhs {
        // Stored [k][n], contiguous along n.
        let n_lines = n / 8;
        let rhs_vec_base = batch * (rhs_batch_stride / 8);
        let col0_vec = col0 / 8;
        #[unroll]
        for t in 0..(CMMA_BK * (bn / 8) / (planes * 32)) {
            let v = unit + t * units;
            if v < CMMA_BK * (bn / 8) {
                let kk = v / (bn / 8);
                let vec_col = v % (bn / 8);
                let global_k = k0 + kk;
                let vec_n = col0_vec + vec_col;
                if global_k < k && vec_n < n_lines {
                    let vv = rhs_vec[rhs_vec_base + global_k * n_lines + vec_n];
                    #[unroll]
                    for j in 0usize..8usize {
                        b_pref[t * 8 + j] = vv[j];
                    }
                } else {
                    #[unroll]
                    for j in 0usize..8usize {
                        let global_col = col0 + vec_col * 8 + j;
                        let mut sv = FI::new(0.0_f32);
                        if global_k < k && global_col < n {
                            sv = rhs[rhs_base + global_k * n + global_col];
                        }
                        b_pref[t * 8 + j] = sv;
                    }
                }
            }
        }
    } else {
        #[unroll]
        for t in 0..(CMMA_BK * bn / (planes * 32)) {
            let i = unit + t * units;
            if i < CMMA_BK * bn {
                let kk = i / bn;
                let col = i % bn;
                let global_k = k0 + kk;
                let global_col = col0 + col;
                let mut v = FI::new(0.0_f32);
                if global_k < k && global_col < n {
                    v = rhs[rhs_base + global_k * n + global_col];
                }
                b_pref[t] = v;
            }
        }
    }

    let steps = k.div_ceil(CMMA_BK);
    for step in 0..steps {
        // Write this step's prefetched A elements into shared memory,
        // rounding to the fragment type at the write.
        if lhs_t {
            if vec_ok_lhs {
                #[unroll]
                for t in 0..((bm / 8) * CMMA_BK / (planes * 32)) {
                    let v = unit + t * units;
                    if v < (bm / 8) * CMMA_BK {
                        let vec_row = v % (bm / 8);
                        let kk = v / (bm / 8);
                        #[unroll]
                        for j in 0usize..8usize {
                            sa[(vec_row * 8 + j) * a_stride + kk] =
                                FS::cast_from(a_pref[t * 8 + j]);
                        }
                    }
                }
            } else {
                #[unroll]
                for t in 0..(bm * CMMA_BK / (planes * 32)) {
                    let i = unit + t * units;
                    if i < bm * CMMA_BK {
                        let row = i % bm;
                        let kk = i / bm;
                        sa[row * a_stride + kk] = FS::cast_from(a_pref[t]);
                    }
                }
            }
        } else if vec_ok_lhs {
            #[unroll]
            for t in 0..(bm * (CMMA_BK / 8) / (planes * 32)) {
                let v = unit + t * units;
                if v < bm * (CMMA_BK / 8) {
                    let row = v / (CMMA_BK / 8);
                    let vec_col = v % (CMMA_BK / 8);
                    #[unroll]
                    for j in 0usize..8usize {
                        sa[row * a_stride + vec_col * 8 + j] =
                            FS::cast_from(a_pref[t * 8 + j]);
                    }
                }
            }
        } else {
            #[unroll]
            for t in 0..(bm * CMMA_BK / (planes * 32)) {
                let i = unit + t * units;
                if i < bm * CMMA_BK {
                    let row = i / CMMA_BK;
                    let kk = i % CMMA_BK;
                    sa[row * a_stride + kk] = FS::cast_from(a_pref[t]);
                }
            }
        }

        // Write this step's prefetched B elements into shared memory,
        // rounding to the fragment type at the write.
        if rhs_t {
            if vec_ok_rhs {
                #[unroll]
                for t in 0..(bn * (CMMA_BK / 8) / (planes * 32)) {
                    let v = unit + t * units;
                    if v < bn * (CMMA_BK / 8) {
                        let col = v / (CMMA_BK / 8);
                        let vec_kk = v % (CMMA_BK / 8);
                        #[unroll]
                        for j in 0usize..8usize {
                            sb[(vec_kk * 8 + j) * b_stride + col] =
                                FS::cast_from(b_pref[t * 8 + j]);
                        }
                    }
                }
            } else {
                #[unroll]
                for t in 0..(CMMA_BK * bn / (planes * 32)) {
                    let i = unit + t * units;
                    if i < CMMA_BK * bn {
                        let col = i / CMMA_BK;
                        let kk = i % CMMA_BK;
                        sb[kk * b_stride + col] = FS::cast_from(b_pref[t]);
                    }
                }
            }
        } else if vec_ok_rhs {
            #[unroll]
            for t in 0..(CMMA_BK * (bn / 8) / (planes * 32)) {
                let v = unit + t * units;
                if v < CMMA_BK * (bn / 8) {
                    let kk = v / (bn / 8);
                    let vec_col = v % (bn / 8);
                    #[unroll]
                    for j in 0usize..8usize {
                        sb[kk * b_stride + vec_col * 8 + j] =
                            FS::cast_from(b_pref[t * 8 + j]);
                    }
                }
            }
        } else {
            #[unroll]
            for t in 0..(CMMA_BK * bn / (planes * 32)) {
                let i = unit + t * units;
                if i < CMMA_BK * bn {
                    let kk = i / bn;
                    let col = i % bn;
                    sb[kk * b_stride + col] = FS::cast_from(b_pref[t]);
                }
            }
        }

        sync_cube();

        // Issue the next step's global loads while this step's MMAs below run.
        if step + 1 < steps {
            let k0 = (step + 1) * CMMA_BK;
            // Prefetch A of step `k0` into registers (zero outside m/k).
            if lhs_t {
                // Stored [k][m], contiguous along m.
                if vec_ok_lhs {
                    let m_lines = m / 8;
                    let lhs_vec_base = batch * (lhs_batch_stride / 8);
                    let row0_vec = row0 / 8;
                    #[unroll]
                    for t in 0..((bm / 8) * CMMA_BK / (planes * 32)) {
                        let v = unit + t * units;
                        if v < (bm / 8) * CMMA_BK {
                            let vec_row = v % (bm / 8);
                            let kk = v / (bm / 8);
                            let global_k = k0 + kk;
                            let vec_m = row0_vec + vec_row;
                            if global_k < k && vec_m < m_lines {
                                let vv = lhs_vec[lhs_vec_base + global_k * m_lines + vec_m];
                                #[unroll]
                                for j in 0usize..8usize {
                                    a_pref[t * 8 + j] = vv[j];
                                }
                            } else {
                                #[unroll]
                                for j in 0usize..8usize {
                                    let global_row = row0 + vec_row * 8 + j;
                                    let mut sv = FI::new(0.0_f32);
                                    if global_k < k && global_row < m {
                                        sv = lhs[lhs_base + global_k * m + global_row];
                                    }
                                    a_pref[t * 8 + j] = sv;
                                }
                            }
                        }
                    }
                } else {
                    #[unroll]
                    for t in 0..(bm * CMMA_BK / (planes * 32)) {
                        let i = unit + t * units;
                        if i < bm * CMMA_BK {
                            let row = i % bm;
                            let kk = i / bm;
                            let global_row = row0 + row;
                            let global_k = k0 + kk;
                            let mut v = FI::new(0.0_f32);
                            if global_row < m && global_k < k {
                                v = lhs[lhs_base + global_k * m + global_row];
                            }
                            a_pref[t] = v;
                        }
                    }
                }
            } else if vec_ok_lhs {
                // Stored [m][k], contiguous along k.
                let k_lines = k / 8;
                let lhs_vec_base = batch * (lhs_batch_stride / 8);
                let k0_vec = k0 / 8;
                #[unroll]
                for t in 0..(bm * (CMMA_BK / 8) / (planes * 32)) {
                    let v = unit + t * units;
                    if v < bm * (CMMA_BK / 8) {
                        let row = v / (CMMA_BK / 8);
                        let vec_col = v % (CMMA_BK / 8);
                        let global_row = row0 + row;
                        let vec_k = k0_vec + vec_col;
                        if global_row < m && vec_k < k_lines {
                            let vv = lhs_vec[lhs_vec_base + global_row * k_lines + vec_k];
                            #[unroll]
                            for j in 0usize..8usize {
                                a_pref[t * 8 + j] = vv[j];
                            }
                        } else {
                            #[unroll]
                            for j in 0usize..8usize {
                                let global_k = k0 + vec_col * 8 + j;
                                let mut sv = FI::new(0.0_f32);
                                if global_row < m && global_k < k {
                                    sv = lhs[lhs_base + global_row * k + global_k];
                                }
                                a_pref[t * 8 + j] = sv;
                            }
                        }
                    }
                }
            } else {
                #[unroll]
                for t in 0..(bm * CMMA_BK / (planes * 32)) {
                    let i = unit + t * units;
                    if i < bm * CMMA_BK {
                        let row = i / CMMA_BK;
                        let kk = i % CMMA_BK;
                        let global_row = row0 + row;
                        let global_k = k0 + kk;
                        let mut v = FI::new(0.0_f32);
                        if global_row < m && global_k < k {
                            v = lhs[lhs_base + global_row * k + global_k];
                        }
                        a_pref[t] = v;
                    }
                }
            }
            // Prefetch B of step `k0` into registers (zero outside k/n).
            if rhs_t {
                // Stored [n][k], contiguous along k.
                if vec_ok_rhs {
                    let k_lines = k / 8;
                    let rhs_vec_base = batch * (rhs_batch_stride / 8);
                    let k0_vec = k0 / 8;
                    #[unroll]
                    for t in 0..(bn * (CMMA_BK / 8) / (planes * 32)) {
                        let v = unit + t * units;
                        if v < bn * (CMMA_BK / 8) {
                            let col = v / (CMMA_BK / 8);
                            let vec_kk = v % (CMMA_BK / 8);
                            let global_col = col0 + col;
                            let vec_k = k0_vec + vec_kk;
                            if global_col < n && vec_k < k_lines {
                                let vv = rhs_vec[rhs_vec_base + global_col * k_lines + vec_k];
                                #[unroll]
                                for j in 0usize..8usize {
                                    b_pref[t * 8 + j] = vv[j];
                                }
                            } else {
                                #[unroll]
                                for j in 0usize..8usize {
                                    let global_k = k0 + vec_kk * 8 + j;
                                    let mut sv = FI::new(0.0_f32);
                                    if global_col < n && global_k < k {
                                        sv = rhs[rhs_base + global_col * k + global_k];
                                    }
                                    b_pref[t * 8 + j] = sv;
                                }
                            }
                        }
                    }
                } else {
                    #[unroll]
                    for t in 0..(CMMA_BK * bn / (planes * 32)) {
                        let i = unit + t * units;
                        if i < CMMA_BK * bn {
                            let col = i / CMMA_BK;
                            let kk = i % CMMA_BK;
                            let global_col = col0 + col;
                            let global_k = k0 + kk;
                            let mut v = FI::new(0.0_f32);
                            if global_k < k && global_col < n {
                                v = rhs[rhs_base + global_col * k + global_k];
                            }
                            b_pref[t] = v;
                        }
                    }
                }
            } else if vec_ok_rhs {
                // Stored [k][n], contiguous along n.
                let n_lines = n / 8;
                let rhs_vec_base = batch * (rhs_batch_stride / 8);
                let col0_vec = col0 / 8;
                #[unroll]
                for t in 0..(CMMA_BK * (bn / 8) / (planes * 32)) {
                    let v = unit + t * units;
                    if v < CMMA_BK * (bn / 8) {
                        let kk = v / (bn / 8);
                        let vec_col = v % (bn / 8);
                        let global_k = k0 + kk;
                        let vec_n = col0_vec + vec_col;
                        if global_k < k && vec_n < n_lines {
                            let vv = rhs_vec[rhs_vec_base + global_k * n_lines + vec_n];
                            #[unroll]
                            for j in 0usize..8usize {
                                b_pref[t * 8 + j] = vv[j];
                            }
                        } else {
                            #[unroll]
                            for j in 0usize..8usize {
                                let global_col = col0 + vec_col * 8 + j;
                                let mut sv = FI::new(0.0_f32);
                                if global_k < k && global_col < n {
                                    sv = rhs[rhs_base + global_k * n + global_col];
                                }
                                b_pref[t * 8 + j] = sv;
                            }
                        }
                    }
                }
            } else {
                #[unroll]
                for t in 0..(CMMA_BK * bn / (planes * 32)) {
                    let i = unit + t * units;
                    if i < CMMA_BK * bn {
                        let kk = i / bn;
                        let col = i % bn;
                        let global_k = k0 + kk;
                        let global_col = col0 + col;
                        let mut v = FI::new(0.0_f32);
                        if global_k < k && global_col < n {
                            v = rhs[rhs_base + global_k * n + global_col];
                        }
                        b_pref[t] = v;
                    }
                }
            }
        }

        #[unroll]
        for ks in 0..CMMA_BK / MMA_TILE {
            let mut a_frags = Sequence::<cmma::Matrix<FS>>::new();
            #[unroll]
            for i in 0..wm {
                a_frags.push(cmma::Matrix::<FS>::from_slice(
                    cmma::MatrixIdent::A,
                    tile,
                    tile,
                    tile,
                    cmma::MatrixLayout::RowMajor,
                    &sa.to_slice().slice(
                        (plane_row0 + i * tile) * a_stride + ks * tile,
                        bm * (CMMA_BK + 8),
                    ),
                    a_stride as u32,
                ));
            }
            let mut b_frags = Sequence::<cmma::Matrix<FS>>::new();
            #[unroll]
            for j in 0..wn {
                b_frags.push(cmma::Matrix::<FS>::from_slice(
                    cmma::MatrixIdent::B,
                    tile,
                    tile,
                    tile,
                    cmma::MatrixLayout::RowMajor,
                    &sb.to_slice().slice(
                        ks * tile * b_stride + plane_col0 + j * tile,
                        CMMA_BK * (bn + 8),
                    ),
                    b_stride as u32,
                ));
            }
            // The flat accumulator index is a computed expression, which
            // CubeCL does not accept as a constant `Sequence` index, so the
            // two grids are spelled out with literals under a static branch.
            if comptime![wm == 4usize] {
                cmma::execute::<FS, FS, F, F>(
                    a_frags.index(0usize),
                    b_frags.index(0usize),
                    acc.index(0usize),
                    acc.index(0usize),
                );
                cmma::execute::<FS, FS, F, F>(
                    a_frags.index(0usize),
                    b_frags.index(1usize),
                    acc.index(1usize),
                    acc.index(1usize),
                );
                cmma::execute::<FS, FS, F, F>(
                    a_frags.index(1usize),
                    b_frags.index(0usize),
                    acc.index(2usize),
                    acc.index(2usize),
                );
                cmma::execute::<FS, FS, F, F>(
                    a_frags.index(1usize),
                    b_frags.index(1usize),
                    acc.index(3usize),
                    acc.index(3usize),
                );
                cmma::execute::<FS, FS, F, F>(
                    a_frags.index(2usize),
                    b_frags.index(0usize),
                    acc.index(4usize),
                    acc.index(4usize),
                );
                cmma::execute::<FS, FS, F, F>(
                    a_frags.index(2usize),
                    b_frags.index(1usize),
                    acc.index(5usize),
                    acc.index(5usize),
                );
                cmma::execute::<FS, FS, F, F>(
                    a_frags.index(3usize),
                    b_frags.index(0usize),
                    acc.index(6usize),
                    acc.index(6usize),
                );
                cmma::execute::<FS, FS, F, F>(
                    a_frags.index(3usize),
                    b_frags.index(1usize),
                    acc.index(7usize),
                    acc.index(7usize),
                );
            } else {
                cmma::execute::<FS, FS, F, F>(
                    a_frags.index(0usize),
                    b_frags.index(0usize),
                    acc.index(0usize),
                    acc.index(0usize),
                );
                cmma::execute::<FS, FS, F, F>(
                    a_frags.index(0usize),
                    b_frags.index(1usize),
                    acc.index(1usize),
                    acc.index(1usize),
                );
                cmma::execute::<FS, FS, F, F>(
                    a_frags.index(1usize),
                    b_frags.index(0usize),
                    acc.index(2usize),
                    acc.index(2usize),
                );
                cmma::execute::<FS, FS, F, F>(
                    a_frags.index(1usize),
                    b_frags.index(1usize),
                    acc.index(3usize),
                    acc.index(3usize),
                );
            }
        }

        // The next step overwrites both tiles, so every plane has to be done
        // reading them before the staging loop above runs again.
        sync_cube();
    }

    // Per-plane epilogue: each plane moves one fragment at a time through its
    // own 256-float slice of `sp`. Lane `j` writes elements `j, j+32, …`, so
    // consecutive lanes write consecutive columns of the tile.
    let lane = UNIT_POS_PLANE as usize;
    let out_base = batch * m * n;
    if comptime![wm == 4usize] {
        cmma::store(
            &mut sp.to_slice_mut().slice_mut(plane_idx * 256, planes * 256),
            acc.index(0usize),
            16u32,
            cmma::MatrixLayout::RowMajor,
        );
        sync_plane();
        #[unroll]
        for t in 0usize..8usize {
            let e = lane + t * 32;
            let global_row = row0 + plane_row0 + e / 16;
            let global_col = col0 + plane_col0 + e % 16;
            if global_row < m && global_col < n {
                out[out_base + global_row * n + global_col] = sp[plane_idx * 256 + e];
            }
        }
        sync_plane();
        cmma::store(
            &mut sp.to_slice_mut().slice_mut(plane_idx * 256, planes * 256),
            acc.index(1usize),
            16u32,
            cmma::MatrixLayout::RowMajor,
        );
        sync_plane();
        #[unroll]
        for t in 0usize..8usize {
            let e = lane + t * 32;
            let global_row = row0 + plane_row0 + e / 16;
            let global_col = col0 + plane_col0 + 16 + e % 16;
            if global_row < m && global_col < n {
                out[out_base + global_row * n + global_col] = sp[plane_idx * 256 + e];
            }
        }
        sync_plane();
        cmma::store(
            &mut sp.to_slice_mut().slice_mut(plane_idx * 256, planes * 256),
            acc.index(2usize),
            16u32,
            cmma::MatrixLayout::RowMajor,
        );
        sync_plane();
        #[unroll]
        for t in 0usize..8usize {
            let e = lane + t * 32;
            let global_row = row0 + plane_row0 + 16 + e / 16;
            let global_col = col0 + plane_col0 + e % 16;
            if global_row < m && global_col < n {
                out[out_base + global_row * n + global_col] = sp[plane_idx * 256 + e];
            }
        }
        sync_plane();
        cmma::store(
            &mut sp.to_slice_mut().slice_mut(plane_idx * 256, planes * 256),
            acc.index(3usize),
            16u32,
            cmma::MatrixLayout::RowMajor,
        );
        sync_plane();
        #[unroll]
        for t in 0usize..8usize {
            let e = lane + t * 32;
            let global_row = row0 + plane_row0 + 16 + e / 16;
            let global_col = col0 + plane_col0 + 16 + e % 16;
            if global_row < m && global_col < n {
                out[out_base + global_row * n + global_col] = sp[plane_idx * 256 + e];
            }
        }
        sync_plane();
        cmma::store(
            &mut sp.to_slice_mut().slice_mut(plane_idx * 256, planes * 256),
            acc.index(4usize),
            16u32,
            cmma::MatrixLayout::RowMajor,
        );
        sync_plane();
        #[unroll]
        for t in 0usize..8usize {
            let e = lane + t * 32;
            let global_row = row0 + plane_row0 + 32 + e / 16;
            let global_col = col0 + plane_col0 + e % 16;
            if global_row < m && global_col < n {
                out[out_base + global_row * n + global_col] = sp[plane_idx * 256 + e];
            }
        }
        sync_plane();
        cmma::store(
            &mut sp.to_slice_mut().slice_mut(plane_idx * 256, planes * 256),
            acc.index(5usize),
            16u32,
            cmma::MatrixLayout::RowMajor,
        );
        sync_plane();
        #[unroll]
        for t in 0usize..8usize {
            let e = lane + t * 32;
            let global_row = row0 + plane_row0 + 32 + e / 16;
            let global_col = col0 + plane_col0 + 16 + e % 16;
            if global_row < m && global_col < n {
                out[out_base + global_row * n + global_col] = sp[plane_idx * 256 + e];
            }
        }
        sync_plane();
        cmma::store(
            &mut sp.to_slice_mut().slice_mut(plane_idx * 256, planes * 256),
            acc.index(6usize),
            16u32,
            cmma::MatrixLayout::RowMajor,
        );
        sync_plane();
        #[unroll]
        for t in 0usize..8usize {
            let e = lane + t * 32;
            let global_row = row0 + plane_row0 + 48 + e / 16;
            let global_col = col0 + plane_col0 + e % 16;
            if global_row < m && global_col < n {
                out[out_base + global_row * n + global_col] = sp[plane_idx * 256 + e];
            }
        }
        sync_plane();
        cmma::store(
            &mut sp.to_slice_mut().slice_mut(plane_idx * 256, planes * 256),
            acc.index(7usize),
            16u32,
            cmma::MatrixLayout::RowMajor,
        );
        sync_plane();
        #[unroll]
        for t in 0usize..8usize {
            let e = lane + t * 32;
            let global_row = row0 + plane_row0 + 48 + e / 16;
            let global_col = col0 + plane_col0 + 16 + e % 16;
            if global_row < m && global_col < n {
                out[out_base + global_row * n + global_col] = sp[plane_idx * 256 + e];
            }
        }
        sync_plane();
    } else {
        cmma::store(
            &mut sp.to_slice_mut().slice_mut(plane_idx * 256, planes * 256),
            acc.index(0usize),
            16u32,
            cmma::MatrixLayout::RowMajor,
        );
        sync_plane();
        #[unroll]
        for t in 0usize..8usize {
            let e = lane + t * 32;
            let global_row = row0 + plane_row0 + e / 16;
            let global_col = col0 + plane_col0 + e % 16;
            if global_row < m && global_col < n {
                out[out_base + global_row * n + global_col] = sp[plane_idx * 256 + e];
            }
        }
        sync_plane();
        cmma::store(
            &mut sp.to_slice_mut().slice_mut(plane_idx * 256, planes * 256),
            acc.index(1usize),
            16u32,
            cmma::MatrixLayout::RowMajor,
        );
        sync_plane();
        #[unroll]
        for t in 0usize..8usize {
            let e = lane + t * 32;
            let global_row = row0 + plane_row0 + e / 16;
            let global_col = col0 + plane_col0 + 16 + e % 16;
            if global_row < m && global_col < n {
                out[out_base + global_row * n + global_col] = sp[plane_idx * 256 + e];
            }
        }
        sync_plane();
        cmma::store(
            &mut sp.to_slice_mut().slice_mut(plane_idx * 256, planes * 256),
            acc.index(2usize),
            16u32,
            cmma::MatrixLayout::RowMajor,
        );
        sync_plane();
        #[unroll]
        for t in 0usize..8usize {
            let e = lane + t * 32;
            let global_row = row0 + plane_row0 + 16 + e / 16;
            let global_col = col0 + plane_col0 + e % 16;
            if global_row < m && global_col < n {
                out[out_base + global_row * n + global_col] = sp[plane_idx * 256 + e];
            }
        }
        sync_plane();
        cmma::store(
            &mut sp.to_slice_mut().slice_mut(plane_idx * 256, planes * 256),
            acc.index(3usize),
            16u32,
            cmma::MatrixLayout::RowMajor,
        );
        sync_plane();
        #[unroll]
        for t in 0usize..8usize {
            let e = lane + t * 32;
            let global_row = row0 + plane_row0 + 16 + e / 16;
            let global_col = col0 + plane_col0 + 16 + e % 16;
            if global_row < m && global_col < n {
                out[out_base + global_row * n + global_col] = sp[plane_idx * 256 + e];
            }
        }
        sync_plane();
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
fn matmul_plane_dot_kernel<FS: Float + CubeElement, F: Float + CubeElement, N: Size>(
    lhs: &Array<Vector<FS, N>>,
    rhs: &Array<Vector<FS, N>>,
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
            acc += Vector::<F, N>::cast_from(lhs[lhs_base + i])
                * Vector::<F, N>::cast_from(rhs[rhs_base + i]);
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
    /// Block-tiled with the inner product on the matrix cores:
    /// `(bm, bn, lhs_t, rhs_t)`.
    ///
    /// Carries the transposition flags for the same reason [`Plan::BlockT`]
    /// does — the kernel absorbs them into its staging index, so a transposed
    /// problem needs no materialised copy.
    ///
    /// Only ever offered when a reduced-precision mode is on *and* the device
    /// reports the exact fragment combination this needs.
    Cmma(usize, usize, bool, bool),
}

/// Output block shapes the matrix-core candidate may be launched with.
///
/// A cube is one plane per 16×16 fragment of the block: 4 planes for (64, 64)
/// with a 2×2 fragment grid per plane, 1 plane for (32, 32), and 8 planes for
/// (128, 128) with a 4×2 grid — 128, 32 and 256 units at a 32-lane plane.
/// Bigger blocks reuse more staged data per plane (a 128×128 block spends
/// eight tensor-core MMAs per 32-deep step where 64×64 spends four); smaller
/// ones fit shapes with fewer rows, which is what the scan's batched products
/// are. `(64, 64)` stays first: it is what an explicit `MAMBA3_MATMUL_KERNEL`
/// request for `cmma` launches.
const CMMA_CANDIDATES: [(usize, usize); 3] = [(64, 64), (128, 128), (32, 32)];

/// Whether this device can run [`Plan::Cmma`] with `ES` operands accumulating
/// into `E`.
///
/// The runtimes register a set of exactly the `(a, b, cd, m, n, k)` combinations
/// their architecture implements, so this is a membership test rather than a
/// guess from a device name: Ampere and RDNA3 carry both half-type fragments,
/// and CPU and wgpu carry none and so never see the candidate at all.
///
/// One entry in CubeCL 0.10's CUDA table cannot be taken at face value: it lists
/// the 16x16x16 `bf16` fragment from compute capability 7.0, but
/// `nvcuda::wmma` `bf16` fragments only compile for 8.0 (Ampere) and newer, so a
/// Turing T4 passes the membership test and then fails at kernel-compile time.
/// The runtime exposes nothing to correct for — `HardwareProperties` has no
/// compute-capability field, `R::name` is just `"cuda"`, and the CUDA server's
/// `Info` is `()` — so on the CUDA backend the `bf16` branch additionally
/// requires an explicit opt-in, `MAMBA3_CMMA_BF16=1`, and reports unsupported
/// without it. The `f16` branch is unchanged: a T4 does implement that fragment.
fn cmma_supported<R: Runtime, ES: FloatElem, E: FloatElem>(client: &ComputeClient<R>) -> bool {
    use crate::backend::DType;
    use cubecl::ir::{ElemType, FloatKind, StorageType};

    fn storage(dtype: DType) -> StorageType {
        match dtype {
            DType::F32 => ElemType::Float(FloatKind::F32).into(),
            DType::F16 => ElemType::Float(FloatKind::F16).into(),
            DType::BF16 => ElemType::Float(FloatKind::BF16).into(),
        }
    }

    let operand = storage(ES::DTYPE);
    if !client
        .features()
        .matmul
        .cmma
        .contains(&cubecl::ir::features::MmaConfig {
            a_type: operand,
            b_type: operand,
            cd_type: storage(E::DTYPE),
            m: MMA_TILE as u32,
            n: MMA_TILE as u32,
            k: MMA_TILE as u32,
        })
    {
        return false;
    }
    // CubeCL 0.10's CUDA table lists the 16x16x16 bf16 fragment from sm_70, but
    // nvcc only accepts `nvcuda::wmma` bf16 fragments on sm_80+. With no compute
    // capability to test against, stay off the fragment on CUDA unless told the
    // device is new enough; the tuner and the explicit `Cmma` request then fall
    // back to the register kernels instead of aborting the process at compile
    // time.
    if ES::DTYPE == DType::BF16 && R::name(client) == "cuda" && !cmma_bf16_opt_in() {
        return false;
    }
    true
}

/// Whether `MAMBA3_CMMA_BF16=1` is set, opting the CUDA backend into the `bf16`
/// matrix-core fragment; see [`cmma_supported`].
fn cmma_bf16_opt_in() -> bool {
    static OPT_IN: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OPT_IN.get_or_init(|| std::env::var("MAMBA3_CMMA_BF16").as_deref() == Ok("1"))
}

/// Launch a [`Plan::Cmma`] block directly, with the operand storage type `FI`
/// decoupled from the fragment type `FS`.
///
/// The ordinary path has `FI = FS` and the cast on staging is a no-op. The
/// mixed-precision fast path passes `f32` storage with a narrow fragment type
/// so the rounding happens while staging into shared memory instead of in two
/// separate cast launches. `allow_vec` gates the 8-wide vector path, which is
/// for 16-bit operands: 8 `f32` would be a 32-byte load.
#[allow(clippy::too_many_arguments)]
fn launch_cmma<R: Runtime, FI: FloatElem, FS: FloatElem, E: FloatElem>(
    lhs: &Tensor<R, FI>,
    rhs: &Tensor<R, FI>,
    out: &Tensor<R, E>,
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
    lhs_batch_stride: usize,
    rhs_batch_stride: usize,
    bm: usize,
    bn: usize,
    lhs_t: bool,
    rhs_t: bool,
    allow_vec: bool,
) {
    let row_blocks = m.div_ceil(bm);
    let col_blocks = n.div_ceil(bn);
    let plane = lhs.client().properties().hardware.plane_size_max as usize;
    // Fragment rows × columns per plane: (4, 2) for 128×128, (2, 2) otherwise.
    let (wm, wn) = if bm == 128 && bn == 128 {
        (4, 2)
    } else {
        (2, 2)
    };
    let planes = (bm / (16 * wm)) * (bn / (16 * wn));
    // The register prefetch files assume 32-lane planes and divide the tiles
    // evenly; every candidate below does.
    debug_assert_eq!(bm * CMMA_BK % (planes * 32), 0);
    debug_assert_eq!(CMMA_BK * bn % (planes * 32), 0);
    // 16-byte alignment for the 8-wide vector path: the contiguous
    // extent and the per-batch base offset must both be multiples of 8.
    let lhs_contig = if lhs_t { m } else { k };
    let rhs_contig = if rhs_t { k } else { n };
    let vec_ok_lhs =
        allow_vec && lhs_contig.is_multiple_of(8) && lhs_batch_stride.is_multiple_of(8);
    let vec_ok_rhs =
        allow_vec && rhs_contig.is_multiple_of(8) && rhs_batch_stride.is_multiple_of(8);
    crate::backend::count_launch();
    unsafe {
        matmul_cmma_kernel::launch_unchecked::<FI, FS, E, R>(
            lhs.client(),
            CubeCount::Static((row_blocks * col_blocks) as u32, batch as u32, 1),
            CubeDim::new_1d((planes * plane) as u32),
            lhs.arg(),
            lhs.arg(),
            rhs.arg(),
            rhs.arg(),
            out.arg(),
            m,
            n,
            k,
            lhs_batch_stride,
            rhs_batch_stride,
            col_blocks,
            bm,
            bn,
            wm,
            wn,
            planes,
            MMA_TILE,
            lhs_t,
            rhs_t,
            vec_ok_lhs,
            vec_ok_rhs,
        );
    }
}

/// Launch one specific plan into an existing output buffer.
///
/// `ES` is the element type the operands are stored in and `E` the type the
/// product is accumulated in and written out as. The ordinary path has `ES = E`;
/// the mixed-precision mode passes `bf16` operands with an `f32` output.
#[allow(clippy::too_many_arguments)]
fn launch_matmul<R: Runtime, ES: FloatElem, E: FloatElem>(
    plan: Plan,
    lhs: &Tensor<R, ES>,
    rhs: &Tensor<R, ES>,
    out: &Tensor<R, E>,
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
    lhs_batch_stride: usize,
    rhs_batch_stride: usize,
) {
    // Input and output vectors share an element count, so the width has to be
    // one the device supports for both element types.
    let line_for = |d: usize| {
        line_size_for::<R, E>(lhs.client(), d).min(line_size_for::<R, ES>(lhs.client(), d))
    };
    match plan {
        Plan::Simple => {
            // Widths that divide `n` also divide both possible `rhs` batch strides
            // (`0` when broadcast, `k * n` otherwise), so no unit's vector can
            // straddle two rows.
            let line = line_for(n);
            let lanes = batch * m * (n / line);
            let (count, dim) = launch_1d(lhs.client(), lanes, k * line);
            unsafe {
                matmul_simple_kernel::launch_unchecked::<ES, E, R>(
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
            let line = line_for(n);
            let n_lines = n / line;
            let row_tiles = m.div_ceil(ROWS);
            let lanes = batch * row_tiles * n_lines;
            let (count, dim) = launch_1d(lhs.client(), lanes, k * line * ROWS);
            unsafe {
                matmul_row_tiled_kernel::launch_unchecked::<ES, E, R>(
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
                matmul_block_tiled_kernel::launch_unchecked::<ES, E, R>(
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
                matmul_block_tiled_t_kernel::launch_unchecked::<ES, E, R>(
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
                matmul_block_tiled_vec_kernel::launch_unchecked::<ES, E, R>(
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
        Plan::Cmma(bm, bn, lhs_t, rhs_t) => {
            launch_cmma::<R, ES, ES, E>(
                lhs,
                rhs,
                out,
                batch,
                m,
                n,
                k,
                lhs_batch_stride,
                rhs_batch_stride,
                bm,
                bn,
                lhs_t,
                rhs_t,
                true,
            );
        }
        Plan::PlaneDot => {
            let line = line_for(k);
            let plane = lhs.client().properties().hardware.plane_size_max as usize;
            let cube_dim = CubeDim::new_1d((plane * DOT_PLANES) as u32);
            let count = crate::backend::cube_count_for(batch * m * n * plane, cube_dim.num_elems());
            crate::backend::count_launch();
            unsafe {
                matmul_plane_dot_kernel::launch_unchecked::<ES, E, R>(
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
                matmul_tiled_kernel::launch_unchecked::<ES, E, R>(
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
fn launch_plan<R: Runtime, ES: FloatElem, E: FloatElem>(
    plan: Plan,
    lhs: &Tensor<R, ES>,
    rhs: &Tensor<R, ES>,
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
    if (lhs_t || rhs_t) && !matches!(plan, Plan::BlockT(..) | Plan::PlaneDot | Plan::Cmma(..)) {
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
        plan,
        lhs,
        rhs,
        out,
        batch,
        m,
        n,
        k,
        lhs_batch_stride,
        rhs_batch_stride,
    );
}

/// One matrix product as the dispatcher saw it: `batch` products of an
/// `[m, k]` by a `[k, n]` operand, either of which may be read transposed.
///
/// These six numbers, with the two element types, are the autotuner's key: a
/// `Linear` over `R` rows of width `I` to `O` outputs is `(1, R, O, I)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MatmulShape {
    /// Independent products.
    pub batch: usize,
    /// Output rows.
    pub m: usize,
    /// Output columns.
    pub n: usize,
    /// Contracted extent.
    pub k: usize,
    /// The left operand is stored `[k, m]`.
    pub lhs_t: bool,
    /// The right operand is stored `[n, k]`.
    pub rhs_t: bool,
}

/// Whether [`MATMUL_LOG`] is recording, checked before the lock is taken.
static MATMUL_LOG_ON: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);

/// The products dispatched since [`start_matmul_log`].
static MATMUL_LOG: std::sync::Mutex<Vec<MatmulShape>> = std::sync::Mutex::new(Vec::new());

/// Begin recording the shape of every matrix product, discarding any earlier
/// log. What `MAMBA3_TRACE` prints, as values: which products a step issues,
/// and so how many distinct shapes the autotuner has to learn.
pub fn start_matmul_log() {
    MATMUL_LOG.lock().expect("matmul log is not poisoned").clear();
    MATMUL_LOG_ON.store(true, core::sync::atomic::Ordering::Relaxed);
}

/// Stop recording and return the products dispatched since
/// [`start_matmul_log`], in order.
pub fn take_matmul_log() -> Vec<MatmulShape> {
    MATMUL_LOG_ON.store(false, core::sync::atomic::Ordering::Relaxed);
    core::mem::take(&mut *MATMUL_LOG.lock().expect("matmul log is not poisoned"))
}

/// Timed runs of each candidate before one is chosen.
///
/// Each probe is a launch and a synchronisation, so the cost is `PROBES` times the
/// number of candidates, once per distinct problem shape. Five is enough that a
/// single scheduling hiccup does not decide the winner.
const PROBES: usize = 5;

/// Winning plan per problem shape, transposition and element-type pair, measured
/// once and remembered. The storage and accumulator types are part of the key
/// because the mixed-precision mode changes both the candidates' relative speed
/// and what a cached plan would launch.
type TuneKey = (
    usize,
    usize,
    usize,
    usize,
    bool,
    bool,
    crate::backend::DType,
    crate::backend::DType,
);
static TUNED: std::sync::OnceLock<std::sync::Mutex<std::collections::HashMap<TuneKey, Plan>>> =
    std::sync::OnceLock::new();

/// Problem shapes measured since the last [`reset_tune_miss_count`].
static TUNE_MISSES: core::sync::atomic::AtomicUsize = core::sync::atomic::AtomicUsize::new(0);

/// Matrix-product shapes the autotuner has had to measure so far.
///
/// A shape it has not seen — in this process or, through the on-disk cache, in
/// an earlier one — is timed every applicable way before it is used: a handful
/// of launches, a synchronisation each and a host read that
/// [`crate::backend::read_count`] does not count. A training run should stop
/// adding to this after its first epoch; one that keeps adding is issuing
/// shapes that never repeat.
pub fn tune_miss_count() -> usize {
    TUNE_MISSES.load(core::sync::atomic::Ordering::Relaxed)
}

/// Set [`tune_miss_count`] back to zero.
pub fn reset_tune_miss_count() {
    TUNE_MISSES.store(0, core::sync::atomic::Ordering::Relaxed);
}

/// Distinct problem shapes the autotuner holds a plan for in this process.
pub fn tuned_shape_count() -> usize {
    TUNED
        .get()
        .map_or(0, |cache| cache.lock().expect("matmul tuning cache").len())
}

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
fn tuned_plan<R: Runtime, ES: FloatElem, E: FloatElem>(
    lhs: &Tensor<R, ES>,
    rhs: &Tensor<R, ES>,
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
    let key = (batch, m, n, k, lhs_t, rhs_t, ES::DTYPE, E::DTYPE);
    let cache = TUNED.get_or_init(Default::default);
    if let Some(found) = cache.lock().expect("matmul tuning cache").get(&key) {
        return *found;
    }
    // A choice an earlier process measured on this device, unless every
    // candidate is being verified.
    let check = std::env::var_os("MAMBA3_TUNE_CHECK").is_some();
    if !check && let Some(found) = tune_disk::lookup(lhs.client(), &key) {
        return *cache
            .lock()
            .expect("matmul tuning cache")
            .entry(key)
            .or_insert(found);
    }

    // From here on the shape is measured: launches, synchronisations and a read
    // that `read_count` does not see.
    TUNE_MISSES.fetch_add(1, core::sync::atomic::Ordering::Relaxed);

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
    // The matrix cores, when the mode put narrow operands in front of us and the
    // device implements this exact fragment. It is a candidate rather than a
    // rule for the usual reason — a fragment pipeline needs rows and columns to
    // fill it, and the scan's 64-row products may still prefer a register
    // kernel — and `MAMBA3_TUNE_CHECK` verifies it against the simple kernel
    // reading the same operands before it is allowed to win.
    if cmma_supported::<R, ES, E>(lhs.client()) {
        for (bm, bn) in CMMA_CANDIDATES {
            candidates.push(Plan::Cmma(bm, bn, lhs_t, rhs_t));
        }
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

    // Drop candidates that do not produce the product before timing them.
    //
    // A launch the device silently drops — on Metal a pipeline whose threadgroup
    // is larger than its register use allows never runs, and neither wgpu nor
    // CubeCL reports it — leaves the output buffer holding whatever it held
    // before. Such a candidate times as the fastest and would win every shape it
    // is offered for, poisoning every later product of that shape with stale
    // memory. So every candidate has to prove it writes the product first: the
    // simple kernel runs once as the reference, then each candidate runs from a
    // NaN-poisoned buffer, and both are reduced on the device to a sum and a sum
    // of absolute values. All the scalars come back with one host read, which is
    // what makes this affordable on wgpu, where a read costs a full device wait;
    // tuning happens once per shape per process and is then cached. A candidate
    // survives only if both its scalars are finite and within the same k-scaled
    // relative tolerance the `MAMBA3_TUNE_CHECK` block below uses, measured
    // against the sum of absolute values. A non-finite reference (inf/NaN
    // inputs) disables the filter, and if nothing survives the simple kernel —
    // which always runs — is the plan.
    let candidates: Vec<Plan> = {
        use crate::tensor::ops::{elemwise, index, reduce};
        launch_plan(
            Plan::Simple,
            lhs,
            rhs,
            out,
            batch,
            m,
            n,
            k,
            lhs_batch_stride,
            rhs_batch_stride,
            lhs_t,
            rhs_t,
        );
        let ref_sum = reduce::sum_all(out).expect("tuning reference sum");
        let ref_abs_sum =
            reduce::sum_all(&elemwise::abs(out)).expect("tuning reference absolute sum");
        let mut sums = Vec::with_capacity(candidates.len());
        let mut abs_sums = Vec::with_capacity(candidates.len());
        for candidate in &candidates {
            // Poison the output first, for the same reason the check block does:
            // a dropped launch must read back as NaN, not as the reference.
            elemwise::fill_(out, f32::NAN);
            launch_plan(
                *candidate,
                lhs,
                rhs,
                out,
                batch,
                m,
                n,
                k,
                lhs_batch_stride,
                rhs_batch_stride,
                lhs_t,
                rhs_t,
            );
            sums.push(reduce::sum_all(out).expect("tuning candidate sum"));
            abs_sums.push(reduce::sum_all(&elemwise::abs(out)).expect("tuning candidate sum"));
        }
        let mut floats: Vec<&Tensor<R, E>> = Vec::with_capacity(2 + 2 * candidates.len());
        floats.push(&ref_sum);
        floats.push(&ref_abs_sum);
        for s in &sums {
            floats.push(s);
        }
        for s in &abs_sums {
            floats.push(s);
        }
        let ids: &[&index::IdTensor<R>] = &[];
        let (_, values) = index::read_all(ids, &floats).expect("tuning scalar read");
        let sum_ref = values[0][0];
        let abs_ref = values[1][0];
        if sum_ref.is_finite() && abs_ref.is_finite() {
            let scale = abs_ref.max(1.0);
            let tol = scale * 1e-5 * (k as f32).sqrt().max(1.0);
            let log = std::env::var_os("MAMBA3_TUNE_LOG").is_some();
            let mut kept = Vec::with_capacity(candidates.len());
            for (i, candidate) in candidates.iter().enumerate() {
                let s = values[2 + i][0];
                let a = values[2 + candidates.len() + i][0];
                if s.is_finite()
                    && a.is_finite()
                    && (s - sum_ref).abs() <= tol
                    && (a - abs_ref).abs() <= tol
                {
                    kept.push(*candidate);
                } else if log {
                    eprintln!("tune drop {candidate:?}: launch produced no/incorrect output");
                }
            }
            kept
        } else {
            candidates
        }
    };

    // Warm every candidate first, so no one of them pays for compilation or for the
    // first allocation of its output.
    for candidate in &candidates {
        launch_plan(
            *candidate,
            lhs,
            rhs,
            out,
            batch,
            m,
            n,
            k,
            lhs_batch_stride,
            rhs_batch_stride,
            lhs_t,
            rhs_t,
        );
    }
    // A checked sync, not a discarded one: a candidate whose kernel failed to
    // compile does no work, so it would time as the fastest and win.
    lhs.device().synchronize();

    // Set `MAMBA3_TUNE_CHECK` to verify every candidate against the simple kernel
    // before any of them can win. The tuner's one failure mode is a kernel that is
    // wrong *and* fast — a staging bug that leaves shared memory stale looks like a
    // speedup — and ordinary tests only assert the winner. Running a real model
    // under this flag checks every candidate on every shape that model issues, with
    // its real operands. Tolerance is relative to `k`: the kernels sum the same
    // products in different orders, and a longer sum accumulates more rounding.
    if std::env::var_os("MAMBA3_TUNE_CHECK").is_some() {
        launch_plan(
            Plan::Simple,
            lhs,
            rhs,
            out,
            batch,
            m,
            n,
            k,
            lhs_batch_stride,
            rhs_batch_stride,
            lhs_t,
            rhs_t,
        );
        lhs.device().synchronize();
        let want = out.to_f32();
        let scale = want.iter().fold(1.0_f32, |a, v| a.max(v.abs()));
        let tol = scale * 1e-5 * (k as f32).sqrt().max(1.0);
        for candidate in &candidates {
            // Poison the output first. Without this a candidate that does no
            // work at all — a launch the device silently drops, e.g. a Metal
            // pipeline whose threadgroup is larger than its register use allows
            // — leaves `Simple`'s answer in place and passes the check.
            crate::tensor::ops::elemwise::fill_(out, f32::NAN);
            launch_plan(
                *candidate,
                lhs,
                rhs,
                out,
                batch,
                m,
                n,
                k,
                lhs_batch_stride,
                rhs_batch_stride,
                lhs_t,
                rhs_t,
            );
            lhs.device().synchronize();
            let got = out.to_f32();
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                assert!(
                    g.is_finite() && (g - w).abs() <= tol,
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
    //
    // If the dropped-launch filter above kept nothing, there is nothing to time:
    // the simple kernel, which the reference run just proved works, is the plan.
    let (best, best_time, times) = if candidates.is_empty() {
        (Plan::Simple, f64::INFINITY, Vec::new())
    } else {
        let mut times = vec![f64::INFINITY; candidates.len()];
        for _ in 0..PROBES {
            for (candidate, slot) in candidates.iter().zip(&mut times) {
                let start = std::time::Instant::now();
                launch_plan(
                    *candidate,
                    lhs,
                    rhs,
                    out,
                    batch,
                    m,
                    n,
                    k,
                    lhs_batch_stride,
                    rhs_batch_stride,
                    lhs_t,
                    rhs_t,
                );
                lhs.device().synchronize();
                *slot = slot.min(start.elapsed().as_secs_f64());
            }
        }
        let (winner, best_time) = times
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.total_cmp(b.1))
            .expect("at least one candidate");
        (candidates[winner], *best_time, times)
    };

    // Set `MAMBA3_TUNE_LOG` to see what was chosen and what it achieved. Worth doing
    // once on a new device: the shapes a model issues are not obvious from its
    // configuration, and a shape that lands far below the others is a lead.
    if let Some(level) = std::env::var_os("MAMBA3_TUNE_LOG") {
        let flops = 2.0 * (batch * m * n * k) as f64;
        eprintln!(
            "tune {key:?} -> {best:?}  ({:.0} GFLOP/s)",
            flops / best_time / 1e9
        );
        if level == "2" {
            for (candidate, time) in candidates.iter().zip(&times) {
                eprintln!("     {:>6.0} GFLOP/s  {candidate:?}", flops / time / 1e9);
            }
        }
    }
    // The first plan recorded for a shape is the one every call uses from then
    // on. Two threads can tune the same shape at once; overwriting here let the
    // second winner replace a plan the first thread had already computed with,
    // so one process ran two kernels for one shape — different summation orders,
    // results a few ulp apart, and two identical training runs that did not agree.
    let chosen = *cache
        .lock()
        .expect("matmul tuning cache")
        .entry(key)
        .or_insert(best);
    tune_disk::record(lhs.client(), &key, chosen);
    chosen
}

/// [`tuned_plan`]'s choices, remembered across processes.
///
/// Tuning costs seconds at the start of every process — each new shape times
/// every candidate kernel several times with a device sync in between — and a
/// training script issues a few dozen shapes. The winners are appended to a small
/// text file in the user cache directory (one line per shape, keyed by the device
/// and this crate's version) and read back by later processes, which then run the
/// same kernels from their first step. Reusing a choice is also what makes two
/// processes compute bit-identical products. `MAMBA3_TUNE_CACHE=0` turns it off;
/// `MAMBA3_TUNE_CACHE_DIR` moves it.
mod tune_disk {
    use super::{BLOCK_CANDIDATES, BlockShape, CMMA_CANDIDATES, Plan, TuneKey};
    use crate::backend::DType;
    use cubecl::prelude::{ComputeClient, Runtime};
    use std::collections::HashMap;
    use std::io::Write;
    use std::sync::{Mutex, OnceLock};

    /// Bump when a plan's meaning changes, so old files stop being read.
    ///
    /// v3: plans recorded before the dropped-launch filter (a Metal pipeline whose
    /// threadgroup exceeds its register budget never runs and leaves stale memory
    /// behind, which timed as the fastest and won) may name a candidate that does
    /// not produce the product at all; every such file is ignored.
    const FORMAT: u32 = 3;

    fn path() -> Option<std::path::PathBuf> {
        if std::env::var("MAMBA3_TUNE_CACHE").as_deref() == Ok("0") {
            return None;
        }
        let dir = match std::env::var_os("MAMBA3_TUNE_CACHE_DIR") {
            Some(dir) => std::path::PathBuf::from(dir),
            None => std::env::var_os("XDG_CACHE_HOME")
                .map(std::path::PathBuf::from)
                .or_else(|| std::env::var_os("HOME").map(|h| std::path::PathBuf::from(h).join(".cache")))?
                .join("mamba3"),
        };
        Some(dir.join(format!(
            "matmul-tune-v{FORMAT}-{}.txt",
            env!("CARGO_PKG_VERSION")
        )))
    }

    fn device_tag<R: Runtime>(client: &ComputeClient<R>) -> String {
        let hw = &client.properties().hardware;
        format!(
            "{}/p{}-{}/b{}",
            R::name(client),
            hw.plane_size_min,
            hw.plane_size_max,
            hw.max_bindings
        )
    }

    fn dtype(d: DType) -> &'static str {
        match d {
            DType::F32 => "f32",
            DType::F16 => "f16",
            DType::BF16 => "bf16",
        }
    }

    fn key_string<R: Runtime>(client: &ComputeClient<R>, key: &TuneKey) -> String {
        let (batch, m, n, k, lhs_t, rhs_t, es, e) = *key;
        format!(
            "{}|{batch},{m},{n},{k},{},{},{},{}",
            device_tag(client),
            lhs_t as u8,
            rhs_t as u8,
            dtype(es),
            dtype(e)
        )
    }

    fn shape_string(s: BlockShape) -> String {
        format!("{},{},{},{},{}", s.bm, s.bn, s.bk, s.tm, s.tn)
    }

    fn plan_string(plan: Plan) -> String {
        match plan {
            Plan::Simple => "simple".into(),
            Plan::RowTiled => "row_tiled".into(),
            Plan::Tiled => "tiled".into(),
            Plan::PlaneDot => "plane_dot".into(),
            Plan::Block(s) => format!("block:{}", shape_string(s)),
            Plan::BlockV(s) => format!("block_v:{}", shape_string(s)),
            Plan::BlockT(s, lt, rt) => {
                format!("block_t:{},{},{}", shape_string(s), lt as u8, rt as u8)
            }
            Plan::Cmma(bm, bn, lt, rt) => format!("cmma:{bm},{bn},{},{}", lt as u8, rt as u8),
        }
    }

    /// The inverse of [`plan_string`], accepting only shapes this build can
    /// launch: a file from another build must not smuggle in a tiling that does
    /// not stage evenly.
    fn parse_plan(text: &str) -> Option<Plan> {
        let (name, args) = text.split_once(':').unwrap_or((text, ""));
        let nums: Vec<usize> = if args.is_empty() {
            Vec::new()
        } else {
            args.split(',').map(|v| v.parse().ok()).collect::<Option<_>>()?
        };
        let shape = |v: &[usize]| -> Option<BlockShape> {
            let s = BlockShape {
                bm: *v.first()?,
                bn: *v.get(1)?,
                bk: *v.get(2)?,
                tm: *v.get(3)?,
                tn: *v.get(4)?,
            };
            BLOCK_CANDIDATES.contains(&s).then_some(s)
        };
        let flag = |v: Option<&usize>| -> Option<bool> {
            match v? {
                0 => Some(false),
                1 => Some(true),
                _ => None,
            }
        };
        Some(match name {
            "simple" => Plan::Simple,
            "row_tiled" => Plan::RowTiled,
            "tiled" => Plan::Tiled,
            "plane_dot" => Plan::PlaneDot,
            "block" if nums.len() == 5 => Plan::Block(shape(&nums)?),
            "block_v" if nums.len() == 5 => Plan::BlockV(shape(&nums)?),
            "block_t" if nums.len() == 7 => {
                Plan::BlockT(shape(&nums)?, flag(nums.get(5))?, flag(nums.get(6))?)
            }
            "cmma" if nums.len() == 4 => {
                let (bm, bn) = (nums[0], nums[1]);
                if !CMMA_CANDIDATES.contains(&(bm, bn)) {
                    return None;
                }
                Plan::Cmma(bm, bn, flag(nums.get(2))?, flag(nums.get(3))?)
            }
            _ => return None,
        })
    }

    fn table() -> &'static Mutex<HashMap<String, Plan>> {
        static TABLE: OnceLock<Mutex<HashMap<String, Plan>>> = OnceLock::new();
        TABLE.get_or_init(|| {
            let mut map = HashMap::new();
            if let Some(text) = path().and_then(|p| std::fs::read_to_string(p).ok()) {
                for line in text.lines() {
                    if let Some((key, plan)) = line.rsplit_once('|')
                        && let Some(plan) = parse_plan(plan)
                    {
                        // First line wins, like the in-process table.
                        map.entry(key.to_string()).or_insert(plan);
                    }
                }
            }
            Mutex::new(map)
        })
    }

    pub(super) fn lookup<R: Runtime>(client: &ComputeClient<R>, key: &TuneKey) -> Option<Plan> {
        path()?;
        table().lock().ok()?.get(&key_string(client, key)).copied()
    }

    pub(super) fn record<R: Runtime>(client: &ComputeClient<R>, key: &TuneKey, plan: Plan) {
        let Some(path) = path() else {
            return;
        };
        let key = key_string(client, key);
        {
            let Ok(mut table) = table().lock() else {
                return;
            };
            if table.contains_key(&key) {
                return;
            }
            table.insert(key.clone(), plan);
        }
        // Best effort: a read-only or missing cache directory only costs the
        // next process its tuning time.
        let _ = std::fs::create_dir_all(path.parent().unwrap_or(std::path::Path::new(".")));
        if let Ok(mut file) = std::fs::OpenOptions::new().create(true).append(true).open(&path) {
            let _ = writeln!(file, "{key}|{}", plan_string(plan));
        }
    }
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
    // The mixed-precision mode: round each operand to a narrow type once, then
    // run the same kernels reading the narrow copies and accumulating in `E`.
    // Applied here rather than per kernel so every entry point — including the
    // tuner's probes and the transposed adjoint forms — sees one consistent
    // mode. Only `f32` callers are eligible: a model already stored in a narrow
    // type has nothing to round.
    let mode = matmul_precision();
    if E::DTYPE == crate::backend::DType::F32 {
        // [`set_matmul_precision`] is unchecked, so a mode the device cannot
        // compile can arrive here. Stop on the calling thread with the reason
        // rather than let the shader compiler panic on the device thread.
        if mode != MatmulPrecision::F32
            && let Err(err) = check_matmul_precision(lhs.device(), mode)
        {
            panic!("{err}");
        }
        macro_rules! staged {
            ($narrow:ty) => {{
                let l = crate::tensor::ops::elemwise::cast::<R, E, $narrow>(lhs);
                let r = crate::tensor::ops::elemwise::cast::<R, E, $narrow>(rhs);
                return matmul_3d_inner::<R, $narrow, E>(
                    &l,
                    &r,
                    batch,
                    m,
                    n,
                    k,
                    lhs_batch_stride,
                    rhs_batch_stride,
                    lhs_t,
                    rhs_t,
                );
            }};
        }
        // The casts stay separate launches on purpose. Rounding inside the
        // matrix-core kernel's staging instead (f32 operands in, no cast
        // launches) was measured on a T4 and made the f16 training step slower,
        // 128 -> 165 ms: the kernel re-reads each operand tile from L2 once per
        // output block, about ten times, so f32 operands double those bytes.
        match mode {
            MatmulPrecision::Bf16 => staged!(half::bf16),
            MatmulPrecision::F16 => staged!(half::f16),
            MatmulPrecision::F32 => {}
        }
    }
    matmul_3d_inner::<R, E, E>(
        lhs,
        rhs,
        batch,
        m,
        n,
        k,
        lhs_batch_stride,
        rhs_batch_stride,
        lhs_t,
        rhs_t,
    )
}

/// How many `k` slices [`matmul_3d_inner`] splits a transposed-left product into,
/// if any: only on GPU-like devices, for one matrix whose `k` is long against a
/// small output, and only into slices of at least 512 that divide `k` exactly.
#[allow(clippy::too_many_arguments)]
fn split_k_factor<R: Runtime>(
    client: &ComputeClient<R>,
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
    lhs_t: bool,
    rhs_t: bool,
) -> Option<usize> {
    if batch != 1 || !lhs_t || rhs_t || k < 4096 || m * n > 512 * 512 {
        return None;
    }
    if client.properties().hardware.plane_size_max <= 1 || !split_k_enabled() {
        return None;
    }
    // Aim for ~1-2k of `k` per slice.
    let want = (k / 1536).clamp(2, 32);
    (2..=want).rev().find(|s| k.is_multiple_of(*s) && k / s >= 512)
}

/// `MAMBA3_SPLIT_K=0` turns split-K off.
fn split_k_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MAMBA3_SPLIT_K").as_deref() != Ok("0"))
}

/// [`matmul_3d_t`] after the precision mode has been resolved into an operand
/// element type `ES` and an accumulator/output type `E`.
#[allow(clippy::too_many_arguments)]
fn matmul_3d_inner<R: Runtime, ES: FloatElem, E: FloatElem>(
    lhs: &Tensor<R, ES>,
    rhs: &Tensor<R, ES>,
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
    lhs_batch_stride: usize,
    rhs_batch_stride: usize,
    lhs_t: bool,
    rhs_t: bool,
) -> Tensor<R, E> {
    crate::backend::trace_shape!(
        "TRACE matmul batch={batch} m={m} n={n} k={k} lhs_t={lhs_t} rhs_t={rhs_t} \
         lhs_bstride={lhs_batch_stride} rhs_bstride={rhs_batch_stride}"
    );
    if MATMUL_LOG_ON.load(core::sync::atomic::Ordering::Relaxed) {
        MATMUL_LOG
            .lock()
            .expect("matmul log is not poisoned")
            .push(MatmulShape {
                batch,
                m,
                n,
                k,
                lhs_t,
                rhs_t,
            });
    }
    // Split-K for the weight gradient's shape, `Xᵀ G` with the whole batch of
    // positions as `k`: a few dozen output tiles each walking tens of thousands of
    // `k` leave most of a GPU idle. Both operands are stored `[k, ·]`, so a slice of
    // `k` is a batch stride; the `[splits, m, n]` partials are then summed.
    if let Some(splits) = split_k_factor(lhs.client(), batch, m, n, k, lhs_t, rhs_t) {
        let part = k / splits;
        let partials = matmul_3d_inner::<R, ES, E>(
            lhs,
            rhs,
            splits,
            m,
            n,
            part,
            part * m,
            part * n,
            true,
            false,
        );
        return crate::tensor::ops::reduce::sum_dim(&partials, 0)
            .expect("summing split-k partials over a valid axis");
    }

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
        MatmulKernel::Cmma if cmma_supported::<R, ES, E>(lhs.client()) => {
            Plan::Cmma(CMMA_CANDIDATES[0].0, CMMA_CANDIDATES[0].1, lhs_t, rhs_t)
        }
        // Asking for matrix cores where there are none is a request the device
        // cannot honour, not an error; fall back rather than fail.
        MatmulKernel::Cmma => Plan::RowTiled,
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
        plan,
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
    // Generic over whatever element the operand is stored in, including the
    // mixed mode's bf16 staging copies.
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
    let _op = crate::backend::tally_op_scope("matmul");
    if E::DTYPE == crate::backend::DType::F32 {
        let mode = matmul_precision();
        if mode != MatmulPrecision::F32 {
            check_matmul_precision(lhs.device(), mode)?;
        }
    }
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

    // A shared (broadcast) right operand against a batch of untransposed left
    // matrices is one tall product: `[batch * m, k] @ [k, n]` reads the same bytes
    // in the same order and writes the same output, and the tall shape tiles far
    // better than `batch` short ones (a linear layer on `[B, L, d]` activations,
    // 1.4-1.7x on the entity model's projections).
    let out = if rhs_stride == 0 && lhs_stride == m * k1 && !lhs_t && batch > 1 {
        matmul_3d_t(
            &lhs,
            &rhs,
            1,
            batch * m,
            n,
            k1,
            batch * m * k1,
            0,
            false,
            rhs_t,
        )
    } else {
        matmul_3d_t(
            &lhs, &rhs, batch, m, n, k1, lhs_stride, rhs_stride, lhs_t, rhs_t,
        )
    };

    let mut out_dims = batch_shape.dims().to_vec();
    out_dims.push(m);
    out_dims.push(n);
    out.reshape(Shape::new(out_dims))
}
