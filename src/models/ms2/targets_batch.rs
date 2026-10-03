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
use super::grammar::{ADD_ATOM, CLOSE_RING, Limits, STOP, replay};
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
                let length = trace.len();
                for (t, token) in trace.iter().enumerate() {
                    let base = (row * max_steps + t) * 4;
                    tokens[base] = u32::from(token.kind);
                    tokens[base + 1] = u32::from(token.atom_type);
                    tokens[base + 2] = u32::from(token.bond);
                    tokens[base + 3] = u32::from(token.pointer);
                }
                meta[row * 12] = length as u32;
                meta[row * 12 + 1] = 1;
                for (e, count) in parents[b].iter().enumerate() {
                    meta[row * 12 + 2 + e] = u32::from(*count);
                }
                q[row] = target.q as f32;
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
