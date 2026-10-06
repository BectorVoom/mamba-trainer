//! Host target batch for the graph-action decoder (architecture §4.4).
//!
//! [`TargetBatch::build`] packs every spectrum's kept pseudo-label targets
//! into flat host buffers, validating each trace against the grammar under
//! the parent-formula budget; [`TargetBatch::upload`] copies them to the
//! device. Training conditions on the true parent formula (contracts §7.2):
//! the budget of every target slot is the parent composition.

use cubecl::prelude::Runtime;

use crate::backend::{Device, FloatElem};
use crate::error::{Error, Result};
use crate::tensor::Tensor;
use crate::tensor::ops::index::IdTensor;

use super::chem::Composition;
use super::grammar::{ADD_ATOM, CLOSE_RING, Limits, STOP, Token, replay, replay_exact};
use super::ragged::{DeviceRowPacking, RowPacking};
use super::targets::Labels;

/// Host-side target batch: `B` spectra times `G` target slots.
pub struct TargetBatch {
    /// Spectra per batch.
    pub spectra: usize,
    /// Target slots per spectrum.
    pub slots: usize,
    /// Trace length (`limits.max_steps()`).
    pub max_steps: usize,
    /// `[B*G*T*4]` token fields per step.
    pub tokens: Vec<u32>,
    /// `[B*G*12]` per-slot metadata: length, budget flag, 10 budget counts.
    pub meta: Vec<u32>,
    /// `[B*G]` target weights, 0 for empty slots.
    pub q: Vec<f32>,
    /// `[B*G*T*4]` field-use indicators: 1 where the field of token `i + 1`
    /// is used and `i + 1 < length` (START excluded, STOP included).
    pub use_mask: Vec<f32>,
    /// `[B]`, 1 when the spectrum has at least one target.
    pub labeled: Vec<u8>,
}

/// A target batch as ragged sequences ([`TargetBatch::pack`]).
///
/// The decoder's layers cost the same at a position past the end of a trace
/// as at one inside it, and traces fill about half of the horizon. Here the
/// traces lie end to end in packed rows, with a reset where each begins: the
/// layers run over those rows, and their output is taken back to one trace
/// per row for the heads, which need a trace's positions side by side.
pub struct PackedTargets {
    /// The occupied targets, one per row (`slots = 1`), padded with empty
    /// rows: what the heads and the grammar replay see.
    pub traces: TargetBatch,
    /// `[scan cells * 4]` token fields in the scan layout.
    pub tokens: Vec<u32>,
    /// `[scan cells]` position of each cell inside its own trace.
    pub steps: Vec<u32>,
    /// `[scan cells]` the spectrum of each cell's trace, `u32::MAX` for an
    /// unused cell.
    pub cell_owner: Vec<u32>,
    /// `[scan cells]` the attention cell of each scan cell, `u32::MAX` for
    /// an unused one.
    pub to_scan: Vec<u32>,
    /// `[attention cells]` the scan cell of each attention cell, `u32::MAX`
    /// for an unused one: the inverse of `to_scan`.
    pub to_attn: Vec<u32>,
    /// The scan layout: rows of any spectrum's traces. Its `unpack` and
    /// `pack` relate it to the trace rows.
    pub scan: RowPacking,
    /// The attention layout: rows of one spectrum's traces each, the
    /// spectrum in `row_group`.
    pub attn: RowPacking,
}

impl PackedTargets {
    /// Upload the packing: 10 uploads, no launch, no read.
    pub fn upload<R: Runtime, E: FloatElem>(&self, device: &Device<R>) -> Result<DevicePacking<R, E>> {
        let cells = self.scan.cells();
        Ok(DevicePacking {
            tokens: IdTensor::from_slice(
                &self.tokens,
                vec![self.scan.rows, self.scan.row_len, 4],
                device,
            )?,
            steps: IdTensor::from_slice(&self.steps, vec![cells], device)?,
            cell_owner: IdTensor::from_slice(&self.cell_owner, vec![cells], device)?,
            to_scan: IdTensor::from_slice(&self.to_scan, vec![cells], device)?,
            to_attn: IdTensor::from_slice(&self.to_attn, vec![self.attn.cells()], device)?,
            attn_owner: IdTensor::from_slice(&self.attn.row_group, vec![self.attn.rows], device)?,
            attn_len: self.attn.row_len,
            scan: self.scan.upload(device)?,
        })
    }
}

/// Device copy of a [`PackedTargets`] layout (the traces themselves upload
/// as [`DeviceTargets`]).
pub struct DevicePacking<R: Runtime, E: FloatElem> {
    /// `[P, row_len, 4]` tokens in the scan layout.
    pub tokens: IdTensor<R>,
    /// `[P * row_len]`.
    pub steps: IdTensor<R>,
    /// `[P * row_len]`.
    pub cell_owner: IdTensor<R>,
    /// `[P * row_len]`.
    pub to_scan: IdTensor<R>,
    /// `[Q * attn_len]`.
    pub to_attn: IdTensor<R>,
    /// `[Q]` the spectrum of each attention row.
    pub attn_owner: IdTensor<R>,
    /// Positions per attention row.
    pub attn_len: usize,
    /// The scan layout's maps to the trace rows, and its reset flags.
    pub scan: DeviceRowPacking<R, E>,
}

/// Device copy of a [`TargetBatch`].
pub struct DeviceTargets<R: Runtime, E: FloatElem> {
    /// `[B*G, T, 4]` target tokens.
    pub tokens: IdTensor<R>,
    /// `[B*G, 12]` per-slot metadata: length, budget flag, 10 budget counts.
    pub meta: IdTensor<R>,
    /// `[B*G]` target weights, 0 for empty slots.
    pub q: Tensor<R, E>,
    /// `[B*G, T, 4]` field-use indicators.
    pub use_mask: Tensor<R, E>,
}

impl TargetBatch {
    /// Pack every spectrum's kept targets in [`Labels`] order: at most
    /// `slots` per spectrum (`Err` when a spectrum holds more — the recipe
    /// already keeps at most 16, so `slots = 16` never truncates and `q`
    /// keeps its frozen normalisation), and at most 16 regardless of `slots`
    /// (contract §9). Every `q` must be finite and > 0 and each labeled
    /// spectrum's `q` must sum to 1 within 1e-5 (`Err` naming the spectrum
    /// and the offending value). `labels[b]` is `None` for an
    /// unlabeled spectrum; `parents[b]` is the training budget, the true
    /// parent formula (contracts §7.2, architecture §4.4). Empty slots hold
    /// length 0, `q` 0 and all-zero tokens. Every target trace must replay
    /// legally under the budget, end with STOP and fit `T` (`Err` otherwise).
    pub fn build(
        labels: &[Option<&Labels>],
        parents: &[Composition],
        slots: usize,
        limits: Limits,
    ) -> Result<Self> {
        if labels.len() != parents.len() {
            return Err(Error::config(format!(
                "TargetBatch::build: {} label entries for {} parents",
                labels.len(),
                parents.len()
            )));
        }
        let spectra = labels.len();
        let max_steps = limits.max_steps();
        let mut tokens = vec![0u32; spectra * slots * max_steps * 4];
        let mut meta = vec![0u32; spectra * slots * 12];
        let mut q = vec![0.0f32; spectra * slots];
        let mut use_mask = vec![0.0f32; spectra * slots * max_steps * 4];
        let mut labeled = vec![0u8; spectra];
        for b in 0..spectra {
            let Some(set) = labels[b] else {
                continue;
            };
            if set.targets.len() > 16 {
                return Err(Error::config(format!(
                    "TargetBatch::build: spectrum {b} holds {} targets, past the contract cap of 16",
                    set.targets.len()
                )));
            }
            if set.targets.len() > slots {
                return Err(Error::config(format!(
                    "TargetBatch::build: spectrum {b} holds {} targets but only {slots} slots",
                    set.targets.len()
                )));
            }
            if set.targets.is_empty() {
                continue;
            }
            // Contract §7.2: every q finite and > 0, and the kept q sums to 1
            // within 1e-5.
            for (g, target) in set.targets.iter().enumerate() {
                if !(target.q.is_finite() && target.q > 0.0) {
                    return Err(Error::config(format!(
                        "TargetBatch::build: spectrum {b} target {g} has q {} (needs finite q > 0)",
                        target.q
                    )));
                }
            }
            let q_sum: f64 = set.targets.iter().map(|t| t.q).sum();
            if (q_sum - 1.0).abs() > 1e-5 {
                return Err(Error::config(format!(
                    "TargetBatch::build: spectrum {b} q sums to {q_sum} (needs 1 within 1e-5)"
                )));
            }
            labeled[b] = 1;
            for (g, target) in set.targets.iter().enumerate() {
                let trace = &target.trace;
                if trace.len() > max_steps {
                    return Err(Error::config(format!(
                        "TargetBatch::build: spectrum {b} target {g} has length {} past T = {max_steps}",
                        trace.len()
                    )));
                }
                let last = trace.last().ok_or_else(|| {
                    Error::config(format!(
                        "TargetBatch::build: spectrum {b} target {g} is empty (no STOP)"
                    ))
                })?;
                if last.kind != STOP {
                    return Err(Error::config(format!(
                        "TargetBatch::build: spectrum {b} target {g} does not end with STOP"
                    )));
                }
                replay(trace, limits, Some(parents[b])).map_err(|e| {
                    Error::config(format!(
                        "TargetBatch::build: spectrum {b} target {g} fails the grammar replay: {e}"
                    ))
                })?;
                let row = b * slots + g;
                Self::write_row(
                    &mut tokens,
                    &mut meta,
                    &mut q,
                    &mut use_mask,
                    row,
                    max_steps,
                    trace,
                    &parents[b],
                    1,
                    target.q as f32,
                );
            }
        }
        Ok(Self {
            spectra,
            slots,
            max_steps,
            tokens,
            meta,
            q,
            use_mask,
            labeled,
        })
    }

    /// Write one target row: the token fields, the metadata, the weight and
    /// the field-use indicators. Shared by [`TargetBatch::build`] (budget
    /// flag 1, the pseudo-label weight) and [`TargetBatch::build_exact`]
    /// (budget flag 2, weight 1), so both layouts score the same positions.
    #[allow(clippy::too_many_arguments)]
    fn write_row(
        tokens: &mut [u32],
        meta: &mut [u32],
        q: &mut [f32],
        use_mask: &mut [f32],
        row: usize,
        max_steps: usize,
        trace: &[Token],
        budget: &Composition,
        meta_flag: u32,
        weight: f32,
    ) {
        let length = trace.len();
        for (t, token) in trace.iter().enumerate() {
            let base = (row * max_steps + t) * 4;
            tokens[base] = u32::from(token.kind);
            tokens[base + 1] = u32::from(token.atom_type);
            tokens[base + 2] = u32::from(token.bond);
            tokens[base + 3] = u32::from(token.pointer);
        }
        meta[row * 12] = length as u32;
        meta[row * 12 + 1] = meta_flag;
        for (e, count) in budget.iter().enumerate() {
            meta[row * 12 + 2 + e] = u32::from(*count);
        }
        q[row] = weight;
        // Output position `i` predicts token `i + 1`: a field counts
        // when its token is inside the trace and the kind uses it.
        // Token 0 is START (never scored); the root ADD_ATOM at
        // position 1 uses kind and atom type only.
        for i in 0..max_steps {
            let pos = i + 1;
            if pos >= length {
                break;
            }
            let kind = trace[pos].kind;
            let use_kind = 1.0f32;
            let mut use_type = 0.0f32;
            let mut use_bond = 0.0f32;
            let mut use_ptr = 0.0f32;
            if kind == ADD_ATOM {
                use_type = 1.0;
                if pos > 1 {
                    use_bond = 1.0;
                    use_ptr = 1.0;
                }
            } else if kind == CLOSE_RING {
                use_bond = 1.0;
                use_ptr = 1.0;
            }
            let base = (row * max_steps + i) * 4;
            use_mask[base] = use_kind;
            use_mask[base + 1] = use_type;
            use_mask[base + 2] = use_bond;
            use_mask[base + 3] = use_ptr;
        }
    }

    /// Pack one exact-completion trace per query: `slots = 1`, `q = 1`,
    /// `labeled = 1` and per-slot metadata `[length, 2, counts...]` (flag 2
    /// is the exact-completion budget of [`replay_exact`], whose device twin
    /// reads the same flag). Every trace must replay with [`replay_exact`]
    /// under `limits` and the stated composition to a state that is stopped
    /// and complete, and must fit `T = limits.max_steps()`; anything else is
    /// [`Error::Config`] naming the query index. The token and `use_mask`
    /// rows are written by the same code [`TargetBatch::build`] uses.
    pub fn build_exact(
        traces: &[&[Token]],
        compositions: &[Composition],
        limits: Limits,
    ) -> Result<Self> {
        if traces.len() != compositions.len() {
            return Err(Error::config(format!(
                "TargetBatch::build_exact: {} traces for {} compositions",
                traces.len(),
                compositions.len()
            )));
        }
        let spectra = traces.len();
        let slots = 1;
        let max_steps = limits.max_steps();
        let mut tokens = vec![0u32; spectra * slots * max_steps * 4];
        let mut meta = vec![0u32; spectra * slots * 12];
        let mut q = vec![0.0f32; spectra * slots];
        let mut use_mask = vec![0.0f32; spectra * slots * max_steps * 4];
        let mut labeled = vec![0u8; spectra];
        for (i, (trace, composition)) in traces.iter().zip(compositions.iter()).enumerate() {
            if trace.len() > max_steps {
                return Err(Error::config(format!(
                    "TargetBatch::build_exact: query {i} has length {} past T = {max_steps}",
                    trace.len()
                )));
            }
            let last = trace.last().ok_or_else(|| {
                Error::config(format!(
                    "TargetBatch::build_exact: query {i} is empty (no STOP)"
                ))
            })?;
            if last.kind != STOP {
                return Err(Error::config(format!(
                    "TargetBatch::build_exact: query {i} does not end with STOP"
                )));
            }
            let end = replay_exact(trace, limits, *composition).map_err(|e| {
                Error::config(format!(
                    "TargetBatch::build_exact: query {i} fails the exact replay: {e}"
                ))
            })?;
            if !end.stopped() || !end.is_complete() {
                return Err(Error::config(format!(
                    "TargetBatch::build_exact: query {i} replays to a stopped={} complete={} state (needs a complete molecule)",
                    end.stopped(),
                    end.is_complete()
                )));
            }
            labeled[i] = 1;
            Self::write_row(
                &mut tokens,
                &mut meta,
                &mut q,
                &mut use_mask,
                i,
                max_steps,
                trace,
                composition,
                2,
                1.0,
            );
        }
        Ok(Self {
            spectra,
            slots,
            max_steps,
            tokens,
            meta,
            q,
            use_mask,
            labeled,
        })
    }

    /// The same targets without the empty slots: every spectrum's occupied
    /// slots in groups of `group`, each group a *virtual spectrum* of `group`
    /// slots (the last group of a spectrum padded with empty slots), and the
    /// virtual spectra padded with all-empty ones up to a multiple of
    /// `bucket`, so the row counts a device sees are few. Returns the compact
    /// batch and, per virtual spectrum, the spectrum it belongs to (0 for an
    /// all-empty one, whose `q` is 0 throughout).
    ///
    /// A teacher pass over it with each virtual spectrum given its
    /// spectrum's memory and conditioning scores the same rows as a pass
    /// over `self`: a row's result depends on its own tokens and its
    /// spectrum only. What it leaves out is the work on rows whose weight is
    /// zero — about half of the rows of a 16-slot batch of real spectra.
    pub fn compact(&self, group: usize, bucket: usize) -> (TargetBatch, Vec<u32>) {
        let (group, bucket) = (group.max(1), bucket.max(1));
        let t4 = self.max_steps * 4;
        let mut owner: Vec<u32> = Vec::new();
        let mut rows: Vec<Option<usize>> = Vec::new();
        for b in 0..self.spectra {
            let occupied: Vec<usize> = (0..self.slots)
                .map(|g| b * self.slots + g)
                .filter(|&row| self.meta[row * 12] != 0)
                .collect();
            for part in occupied.chunks(group) {
                owner.push(b as u32);
                rows.extend(part.iter().map(|&row| Some(row)));
                rows.extend((part.len()..group).map(|_| None));
            }
        }
        let spectra = owner.len().max(1).next_multiple_of(bucket);
        owner.resize(spectra, 0);
        rows.resize(spectra * group, None);
        let mut out = TargetBatch {
            spectra,
            slots: group,
            max_steps: self.max_steps,
            tokens: vec![0u32; rows.len() * t4],
            meta: vec![0u32; rows.len() * 12],
            q: vec![0.0f32; rows.len()],
            use_mask: vec![0.0f32; rows.len() * t4],
            labeled: vec![0u8; spectra],
        };
        for (to, from) in rows.iter().enumerate() {
            let Some(from) = *from else {
                continue;
            };
            out.tokens[to * t4..(to + 1) * t4].copy_from_slice(&self.tokens[from * t4..(from + 1) * t4]);
            out.meta[to * 12..(to + 1) * 12].copy_from_slice(&self.meta[from * 12..(from + 1) * 12]);
            out.q[to] = self.q[from];
            out.use_mask[to * t4..(to + 1) * t4]
                .copy_from_slice(&self.use_mask[from * t4..(from + 1) * t4]);
            out.labeled[to / group] = 1;
        }
        (out, owner)
    }

    /// The occupied slots as a ragged batch ([`PackedTargets`]): the traces
    /// one to a row for the heads, and laid end to end in rows of `row_len`
    /// positions for the decoder layers.
    ///
    /// The layers need two things of a layout and are given one each. The
    /// scan only needs a trace to be contiguous, so its rows take any
    /// spectrum's traces, longest first, each into the first row with room:
    /// they come out nearly full. Cross-attention needs the positions of a
    /// row to share a memory, so its rows hold one spectrum's traces each,
    /// and the unfilled end of every spectrum's last row is paid there only.
    /// Scan rows are padded to a multiple of `scan_bucket`, attention rows
    /// of `attn_bucket`, and the traces of `trace_bucket`, so the shapes a
    /// device sees are few. `row_len` is raised to the horizon if shorter.
    pub fn pack(
        &self,
        row_len: usize,
        scan_bucket: usize,
        attn_bucket: usize,
        trace_bucket: usize,
    ) -> PackedTargets {
        let t = self.max_steps;
        let trace_bucket = trace_bucket.max(1);
        let t4 = t * 4;
        let occupied: Vec<usize> = (0..self.spectra * self.slots)
            .filter(|&row| self.meta[row * 12] != 0)
            .collect();
        let rows = occupied.len().max(1).next_multiple_of(trace_bucket);
        let mut traces = TargetBatch {
            spectra: rows,
            slots: 1,
            max_steps: t,
            tokens: vec![0u32; rows * t4],
            meta: vec![0u32; rows * 12],
            q: vec![0.0f32; rows],
            use_mask: vec![0.0f32; rows * t4],
            labeled: vec![0u8; rows],
        };
        let mut lengths = vec![0usize; rows];
        let mut spectrum = vec![0u32; rows];
        for (to, &from) in occupied.iter().enumerate() {
            traces.tokens[to * t4..(to + 1) * t4]
                .copy_from_slice(&self.tokens[from * t4..(from + 1) * t4]);
            traces.meta[to * 12..(to + 1) * 12]
                .copy_from_slice(&self.meta[from * 12..(from + 1) * 12]);
            traces.q[to] = self.q[from];
            traces.use_mask[to * t4..(to + 1) * t4]
                .copy_from_slice(&self.use_mask[from * t4..(from + 1) * t4]);
            traces.labeled[to] = 1;
            lengths[to] = (self.meta[from * 12] as usize).min(t);
            spectrum[to] = (from / self.slots) as u32;
        }
        // The padding traces follow the last spectrum's, so the groups stay
        // sorted; their length is zero and they take no cell.
        let last = spectrum[..occupied.len()].last().copied().unwrap_or(0);
        spectrum[occupied.len()..].fill(last);
        let scan = RowPacking::new(&lengths, t, row_len, scan_bucket, None);
        let attn = RowPacking::new(&lengths, t, row_len, attn_bucket, Some(&spectrum));
        let mut tokens = vec![0u32; scan.cells() * 4];
        let mut steps = vec![0u32; scan.cells()];
        let mut cell_owner = vec![u32::MAX; scan.cells()];
        let mut to_scan = vec![u32::MAX; scan.cells()];
        let mut to_attn = vec![u32::MAX; attn.cells()];
        for (cell, &position) in scan.pack.iter().enumerate() {
            if position == u32::MAX {
                continue;
            }
            let position = position as usize;
            tokens[cell * 4..cell * 4 + 4]
                .copy_from_slice(&traces.tokens[position * 4..position * 4 + 4]);
            steps[cell] = (position % t) as u32;
            cell_owner[cell] = spectrum[position / t];
            let attn_cell = attn.unpack[position];
            to_scan[cell] = attn_cell;
            to_attn[attn_cell as usize] = cell as u32;
        }
        PackedTargets {
            traces,
            tokens,
            steps,
            cell_owner,
            to_scan,
            to_attn,
            scan,
            attn,
        }
    }

    /// Upload the batch: exactly 4 uploads (tokens, meta, q, use_mask), no
    /// launch, no read.
    pub fn upload<R: Runtime, E: FloatElem>(
        &self,
        device: &Device<R>,
    ) -> Result<DeviceTargets<R, E>> {
        let rows = self.spectra * self.slots;
        let tokens = IdTensor::from_slice(&self.tokens, vec![rows, self.max_steps, 4], device)?;
        let meta = IdTensor::from_slice(&self.meta, vec![rows, 12], device)?;
        let q = Tensor::<R, E>::from_f32(&self.q, vec![rows], device)?;
        let use_mask =
            Tensor::<R, E>::from_f32(&self.use_mask, vec![rows, self.max_steps, 4], device)?;
        Ok(DeviceTargets {
            tokens,
            meta,
            q,
            use_mask,
        })
    }
}
