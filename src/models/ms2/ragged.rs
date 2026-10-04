//! Ragged sequences as packed rows.
//!
//! A padded batch `[segments, width, d]` whose segments use only a prefix of
//! their row costs a causal layer `width` positions a segment, however short
//! the segment. Here the used prefixes lie end to end in rows of `row_len`
//! positions, with a reset where each begins after another: a reset-aware
//! layer ([`crate::models::mamba3::Mamba3Block::apply_with_state_masked`])
//! over those rows computes, for every used position, what it computes in the
//! padded batch, and the positions no segment uses are never computed. Two
//! index maps take a padded tensor to the packed rows and back
//! ([`crate::autograd::Var::ms2_take_rows`]).
//!
//! The layout is decided on the host from lengths the host already has, and
//! uploaded: no device read.

use cubecl::prelude::Runtime;

use crate::autograd::Var;
use crate::backend::{Device, FloatElem};
use crate::error::Result;
use crate::tensor::Tensor;
use crate::tensor::ops::index::IdTensor;

/// Where each segment of a padded batch sits in the packed rows.
#[derive(Debug, Clone, PartialEq)]
pub struct RowPacking {
    /// Packed rows, a multiple of the bucket asked for.
    pub rows: usize,
    /// Positions per packed row.
    pub row_len: usize,
    /// Positions per segment row of the padded layout.
    pub width: usize,
    /// `[segments * width]`: the packed cell of padded position `(s, i)`,
    /// `u32::MAX` past the segment's length.
    pub unpack: Vec<u32>,
    /// `[rows * row_len]`: the padded position of each packed cell,
    /// `u32::MAX` for an unused cell. The inverse of `unpack`.
    pub pack: Vec<u32>,
    /// `[rows * row_len]`: 1 where a segment begins after another in its row.
    pub reset: Vec<f32>,
    /// `[rows]`: the group of the segments in each row, when rows were kept
    /// to one group (0 for an empty row).
    pub row_group: Vec<u32>,
}

impl RowPacking {
    /// Lay `lengths[s]` positions of each segment `s` (clamped to `width`)
    /// into rows of `row_len` (raised to `width` if shorter): longest first,
    /// each into the first row with room. With `groups`, a row takes
    /// segments of one group only (`groups[s]` must be non-decreasing, as the
    /// slots of a batch are); without, any segment fits any row. The rows
    /// are padded with empty ones to a multiple of `row_bucket`.
    pub fn new(
        lengths: &[usize],
        width: usize,
        row_len: usize,
        row_bucket: usize,
        groups: Option<&[u32]>,
    ) -> Self {
        let row_len = row_len.max(width).max(1);
        let row_bucket = row_bucket.max(1);
        let length = |s: usize| lengths[s].min(width);
        let mut order: Vec<usize> = (0..lengths.len()).filter(|&s| length(s) > 0).collect();
        // Longest first; within a group when rows are kept to one group.
        order.sort_by_key(|&s| (groups.map_or(0, |g| g[s]), core::cmp::Reverse(length(s))));
        let mut used: Vec<usize> = Vec::new();
        let mut row_group: Vec<u32> = Vec::new();
        let mut place: Vec<(usize, usize)> = vec![(0, 0); lengths.len()];
        for s in order {
            let group = groups.map_or(0, |g| g[s]);
            let fits = |r: usize, used: &[usize], row_group: &[u32]| {
                row_group[r] == group && used[r] + length(s) <= row_len
            };
            let row = match (0..used.len()).find(|&r| fits(r, &used, &row_group)) {
                Some(row) => row,
                None => {
                    used.push(0);
                    row_group.push(group);
                    used.len() - 1
                }
            };
            place[s] = (row, used[row]);
            used[row] += length(s);
        }
        let rows = used.len().max(1).next_multiple_of(row_bucket);
        row_group.resize(rows, 0);
        let mut unpack = vec![u32::MAX; lengths.len() * width];
        let mut pack = vec![u32::MAX; rows * row_len];
        let mut reset = vec![0.0f32; rows * row_len];
        for s in 0..lengths.len() {
            let (row, offset) = place[s];
            let base = row * row_len + offset;
            if length(s) > 0 && offset > 0 {
                reset[base] = 1.0;
            }
            for i in 0..length(s) {
                unpack[s * width + i] = (base + i) as u32;
                pack[base + i] = (s * width + i) as u32;
            }
        }
        Self {
            rows,
            row_len,
            width,
            unpack,
            pack,
            reset,
            row_group,
        }
    }

    /// Packed cells.
    pub fn cells(&self) -> usize {
        self.rows * self.row_len
    }

    /// Upload the two maps and the reset flags: 3 uploads, no launch, no read.
    pub fn upload<R: Runtime, E: FloatElem>(&self, device: &Device<R>) -> Result<DeviceRowPacking<R, E>> {
        Ok(DeviceRowPacking {
            rows: self.rows,
            row_len: self.row_len,
            unpack: IdTensor::from_slice(&self.unpack, vec![self.unpack.len()], device)?,
            pack: IdTensor::from_slice(&self.pack, vec![self.pack.len()], device)?,
            reset: Tensor::<R, E>::from_f32(&self.reset, vec![self.rows, self.row_len], device)?,
        })
    }
}

/// Device copy of a [`RowPacking`].
pub struct DeviceRowPacking<R: Runtime, E: FloatElem> {
    /// Packed rows.
    pub rows: usize,
    /// Positions per packed row.
    pub row_len: usize,
    /// `[segments * width]`.
    pub unpack: IdTensor<R>,
    /// `[rows * row_len]`.
    pub pack: IdTensor<R>,
    /// `[rows, row_len]`.
    pub reset: Tensor<R, E>,
}

impl<R: Runtime, E: FloatElem> DeviceRowPacking<R, E> {
    /// A padded `[segments, width, d]` as packed rows `[rows, row_len, d]`
    /// (zeros in the unused cells).
    pub fn pack(&self, padded: &Var<R, E>) -> Result<Var<R, E>> {
        let d = padded.shape().dim_from_end(0);
        Var::ms2_take_rows(
            &padded.reshape(vec![self.unpack.len(), d])?,
            &self.pack,
            &self.unpack,
        )?
        .reshape(vec![self.rows, self.row_len, d])
    }

    /// Packed rows `[rows, row_len, d]` back as `like`'s padded shape (zeros
    /// past each segment's length).
    pub fn unpack(&self, packed: &Var<R, E>, like: &[usize]) -> Result<Var<R, E>> {
        let d = packed.shape().dim_from_end(0);
        Var::ms2_take_rows(
            &packed.reshape(vec![self.pack.len(), d])?,
            &self.unpack,
            &self.pack,
        )?
        .reshape(like.to_vec())
    }
}
