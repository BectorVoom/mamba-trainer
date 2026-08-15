//! Shapes, strides and broadcasting rules.

use crate::error::{Error, Result};

/// Maximum tensor rank. Kernels carry shape metadata in fixed-size buffers, so
/// this bound is baked into the launch protocol.
pub const MAX_RANK: usize = 8;

/// A row-major (C-contiguous) shape.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default, serde::Serialize, serde::Deserialize)]
pub struct Shape {
    dims: Vec<usize>,
}

impl Shape {
    /// Build a shape from dimensions.
    pub fn new(dims: impl Into<Vec<usize>>) -> Self {
        let dims = dims.into();
        assert!(
            dims.len() <= MAX_RANK,
            "rank {} exceeds MAX_RANK {MAX_RANK}",
            dims.len()
        );
        Self { dims }
    }

    /// The scalar shape `[]`.
    pub fn scalar() -> Self {
        Self { dims: Vec::new() }
    }

    /// Number of dimensions.
    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    /// Total number of elements.
    pub fn num_elements(&self) -> usize {
        self.dims.iter().product()
    }

    /// Dimensions as a slice.
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Size of dimension `i`, supporting negative-style indexing from the end
    /// through [`Shape::dim_from_end`].
    pub fn dim(&self, i: usize) -> usize {
        self.dims[i]
    }

    /// Size of the `i`-th dimension counted from the end (`0` == last).
    pub fn dim_from_end(&self, i: usize) -> usize {
        self.dims[self.dims.len() - 1 - i]
    }

    /// Row-major strides.
    pub fn strides(&self) -> Vec<usize> {
        let mut strides = vec![1usize; self.dims.len()];
        for i in (0..self.dims.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * self.dims[i + 1];
        }
        strides
    }

    /// Product of all dimensions before `axis`.
    pub fn outer(&self, axis: usize) -> usize {
        self.dims[..axis].iter().product()
    }

    /// Product of all dimensions after `axis`.
    pub fn inner(&self, axis: usize) -> usize {
        self.dims[axis + 1..].iter().product()
    }

    /// Error unless the rank matches.
    pub fn expect_rank(&self, expected: usize) -> Result<()> {
        if self.rank() == expected {
            Ok(())
        } else {
            Err(Error::Rank {
                expected,
                got: self.rank(),
                shape: self.dims.clone(),
            })
        }
    }

    /// A copy with `axis` removed.
    pub fn without(&self, axis: usize) -> Shape {
        let mut dims = self.dims.clone();
        dims.remove(axis);
        Shape::new(dims)
    }

    /// A copy with `axis` set to 1.
    pub fn with_dim(&self, axis: usize, size: usize) -> Shape {
        let mut dims = self.dims.clone();
        dims[axis] = size;
        Shape::new(dims)
    }

    /// A copy with a size-1 axis inserted at `axis`.
    pub fn unsqueezed(&self, axis: usize) -> Shape {
        let mut dims = self.dims.clone();
        dims.insert(axis, 1);
        Shape::new(dims)
    }

    /// Left-pad with ones so the rank becomes `rank`.
    pub fn left_padded(&self, rank: usize) -> Shape {
        let mut dims = vec![1usize; rank.saturating_sub(self.rank())];
        dims.extend_from_slice(&self.dims);
        Shape::new(dims)
    }

    /// NumPy broadcasting: the shape both operands expand to.
    pub fn broadcast(lhs: &Shape, rhs: &Shape) -> Result<Shape> {
        let rank = lhs.rank().max(rhs.rank());
        let l = lhs.left_padded(rank);
        let r = rhs.left_padded(rank);
        let mut out = Vec::with_capacity(rank);
        for i in 0..rank {
            let (a, b) = (l.dims[i], r.dims[i]);
            if a == b || b == 1 {
                out.push(a);
            } else if a == 1 {
                out.push(b);
            } else {
                return Err(Error::shape(format!(
                    "cannot broadcast {:?} with {:?}",
                    lhs.dims, rhs.dims
                )));
            }
        }
        Ok(Shape::new(out))
    }

    /// Strides to read a tensor of this shape as if it had shape `target`,
    /// with `0` on broadcast axes.
    pub fn broadcast_strides(&self, target: &Shape) -> Result<Vec<usize>> {
        let rank = target.rank();
        let padded = self.left_padded(rank);
        let strides = padded.strides();
        let mut out = Vec::with_capacity(rank);
        for i in 0..rank {
            if padded.dims[i] == target.dims[i] {
                out.push(strides[i]);
            } else if padded.dims[i] == 1 {
                out.push(0);
            } else {
                return Err(Error::shape(format!(
                    "cannot broadcast {:?} to {:?}",
                    self.dims, target.dims
                )));
            }
        }
        Ok(out)
    }
}

impl From<Vec<usize>> for Shape {
    fn from(dims: Vec<usize>) -> Self {
        Shape::new(dims)
    }
}

impl From<&[usize]> for Shape {
    fn from(dims: &[usize]) -> Self {
        Shape::new(dims.to_vec())
    }
}

impl<const N: usize> From<[usize; N]> for Shape {
    fn from(dims: [usize; N]) -> Self {
        Shape::new(dims.to_vec())
    }
}

impl core::ops::Index<usize> for Shape {
    type Output = usize;

    fn index(&self, i: usize) -> &usize {
        &self.dims[i]
    }
}

impl core::fmt::Display for Shape {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{:?}", self.dims)
    }
}

/// Pack shape + strides metadata into the fixed-size `u32` layout kernels expect.
///
/// Layout: `[rank, out_shape[MAX_RANK], lhs_strides[MAX_RANK], rhs_strides[MAX_RANK]]`.
pub(crate) fn pack_broadcast_meta(
    out: &Shape,
    lhs_strides: &[usize],
    rhs_strides: &[usize],
) -> Vec<u32> {
    let mut meta = Vec::with_capacity(1 + 3 * MAX_RANK);
    meta.push(out.rank() as u32);
    let mut push_padded = |vals: &[usize]| {
        for i in 0..MAX_RANK {
            meta.push(vals.get(i).copied().unwrap_or(0) as u32);
        }
    };
    push_padded(out.dims());
    push_padded(lhs_strides);
    push_padded(rhs_strides);
    meta
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strides_are_row_major() {
        let s = Shape::new(vec![2, 3, 4]);
        assert_eq!(s.strides(), vec![12, 4, 1]);
        assert_eq!(s.num_elements(), 24);
    }

    #[test]
    fn broadcast_matches_numpy() {
        let a = Shape::new(vec![4, 1, 3]);
        let b = Shape::new(vec![5, 3]);
        let out = Shape::broadcast(&a, &b).unwrap();
        assert_eq!(out.dims(), &[4, 5, 3]);
        assert_eq!(a.broadcast_strides(&out).unwrap(), vec![3, 0, 1]);
        assert_eq!(b.broadcast_strides(&out).unwrap(), vec![0, 3, 1]);
    }

    #[test]
    fn incompatible_broadcast_errors() {
        let a = Shape::new(vec![2, 3]);
        let b = Shape::new(vec![4, 3]);
        assert!(Shape::broadcast(&a, &b).is_err());
    }
}
