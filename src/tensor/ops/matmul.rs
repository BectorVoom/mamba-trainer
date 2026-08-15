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
//! So the default is [`MatmulKernel::Simple`]: one unit per output element, a plain
//! loop over `k`, no shared memory and no synchronisation. It is portable and never
//! pathological; on a GPU it is merely unoptimal rather than unusable. Backends
//! where barriers are cheap can opt into the tiled path with [`set_default_kernel`].

use core::sync::atomic::{AtomicU8, Ordering};

use cubecl::prelude::*;

use crate::backend::{FloatElem, cube_count_for};
use crate::error::{Error, Result};
use crate::tensor::base::Tensor;
use crate::tensor::shape::Shape;

/// Which matmul kernel to launch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatmulKernel {
    /// One unit per output element; no shared memory, no barriers.
    Simple,
    /// Shared-memory tiles with cube barriers. Only choose this on a backend where
    /// `sync_cube` is cheap.
    Tiled,
}

static DEFAULT_KERNEL: AtomicU8 = AtomicU8::new(0);

/// Choose the kernel used by [`matmul`] from now on.
///
/// A backend tuning knob, not a semantic one: both kernels compute the same
/// product to within floating-point associativity.
pub fn set_default_kernel(kernel: MatmulKernel) {
    DEFAULT_KERNEL.store(
        match kernel {
            MatmulKernel::Simple => 0,
            MatmulKernel::Tiled => 1,
        },
        Ordering::Relaxed,
    );
}

/// The kernel [`matmul`] will use.
pub fn default_kernel() -> MatmulKernel {
    match DEFAULT_KERNEL.load(Ordering::Relaxed) {
        1 => MatmulKernel::Tiled,
        _ => MatmulKernel::Simple,
    }
}

/// Tile edge for [`MatmulKernel::Tiled`].
const TILE: usize = 16;

/// Units per cube for [`MatmulKernel::Simple`].
const SIMPLE_CUBE_DIM: u32 = 64;

#[cube(launch_unchecked)]
fn matmul_simple_kernel<F: Float + CubeElement>(
    lhs: &Array<F>,
    rhs: &Array<F>,
    out: &mut Array<F>,
    m: usize,
    n: usize,
    k: usize,
    lhs_batch_stride: usize,
    rhs_batch_stride: usize,
) {
    if ABSOLUTE_POS < out.len() {
        let batch = ABSOLUTE_POS / (m * n);
        let within = ABSOLUTE_POS % (m * n);
        let row = within / n;
        let col = within % n;
        let lhs_base = batch * lhs_batch_stride + row * k;
        let rhs_base = batch * rhs_batch_stride + col;
        let mut acc = F::new(0.0_f32);
        for p in 0..k {
            acc += lhs[lhs_base + p] * rhs[rhs_base + p * n];
        }
        out[ABSOLUTE_POS] = acc;
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
    let out = Tensor::empty(Shape::new(vec![batch, m, n]), lhs.device());
    if out.is_empty() {
        return out;
    }

    match default_kernel() {
        MatmulKernel::Simple => {
            let total = batch * m * n;
            unsafe {
                matmul_simple_kernel::launch_unchecked::<E, R>(
                    lhs.client(),
                    cube_count_for(total, SIMPLE_CUBE_DIM),
                    CubeDim::new_1d(SIMPLE_CUBE_DIM),
                    lhs.arg(),
                    rhs.arg(),
                    out.arg(),
                    m,
                    n,
                    k,
                    lhs_batch_stride,
                    rhs_batch_stride,
                );
            }
        }
        MatmulKernel::Tiled => {
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
    out
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
    if lhs.rank() < 2 || rhs.rank() < 2 {
        return Err(Error::shape(format!(
            "matmul needs rank >= 2 operands, got {} and {}",
            lhs.shape, rhs.shape
        )));
    }
    let (m, k1) = (lhs.shape.dim_from_end(1), lhs.shape.dim_from_end(0));
    let (k2, n) = (rhs.shape.dim_from_end(1), rhs.shape.dim_from_end(0));
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
            d.push(m);
            d.push(k1);
            Shape::new(d)
        };
        (crate::tensor::ops::elemwise::expand(lhs, &target)?, m * k1)
    } else {
        (lhs.clone(), lhs_stride)
    };
    let (rhs, rhs_stride) = if rhs_stride == usize::MAX {
        let target = {
            let mut d = batch_shape.dims().to_vec();
            d.push(k1);
            d.push(n);
            Shape::new(d)
        };
        (crate::tensor::ops::elemwise::expand(rhs, &target)?, k1 * n)
    } else {
        (rhs.clone(), rhs_stride)
    };

    let out = matmul_3d(&lhs, &rhs, batch, m, n, k1, lhs_stride, rhs_stride);

    let mut out_dims = batch_shape.dims().to_vec();
    out_dims.push(m);
    out_dims.push(n);
    out.reshape(Shape::new(out_dims))
}
