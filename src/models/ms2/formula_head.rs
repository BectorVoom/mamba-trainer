//! Formula head: scores the joined precursor window (architecture §4.2).
//!
//! [`DeviceFormulaTable::upload`] copies the host [`FormulaTable`] once: the
//! `[R, 2]` integer search table (mass, per-row arithmetic bound) the
//! [`crate::tensor::ops::ms2::formula_window`] kernel reads, the `[R, 10]`
//! `ln(1 + count)` row features the head embeds, and the `[R, 10]` exact
//! element counts the trajectory initialiser reads. [`FormulaHead::score`]
//! looks the joined window rows up, embeds them, dots them with the pooled
//! spectrum vector and normalises over the window; [`FormulaHead::loss`] is
//! the cross-entropy to the gold window slot, computed on the device with no
//! read.

use cubecl::prelude::Runtime;

use crate::autograd::Var;
use crate::backend::{Device, FloatElem};
use crate::error::{Error, Result};
use crate::nn::linear::{Linear, LinearConfig};
use crate::nn::module::{Module, ModuleVisitor};
use crate::tensor::Tensor;
use crate::tensor::ops::index::{IdTensor, ids_to_float, slice_ids_along};
use crate::tensor::ops::ms2::{FormulaBuffers, nonzero_mask, safe_ids};
use crate::tensor::ops::random::Rng;
use crate::tensor::ops::{elemwise, movement, reduce};

use super::chem::{ELEMENTS, composition_error_nda};
use super::contract::ModelConfig;
use super::formula::{FormulaTable, WindowQuery};

/// Element counts per formula-table row, in [`ELEMENTS`] order.
const FORMULA_ELEMENTS: usize = ELEMENTS.len();

/// SHA-256 over `bytes`, as lowercase hex: the fingerprint
/// [`DeviceFormulaTable::upload`] binds to the uploaded rows. Written out
/// (rather than taken from a hash crate) because the crate has no hash
/// dependency; the `tests/ms2_formula.rs` table hash pins it against the
/// system `sha256sum` over the same canonical row bytes.
fn sha256_hex(bytes: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let mut msg = bytes.to_vec();
    let bit_len = (bytes.len() as u64).wrapping_mul(8);
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());
    for chunk in msg.chunks_exact(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                chunk[4 * i],
                chunk[4 * i + 1],
                chunk[4 * i + 2],
                chunk[4 * i + 3],
            ]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh) =
            (h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }
    let mut out = String::with_capacity(64);
    for v in h {
        out.push_str(&format!("{v:08x}"));
    }
    out
}

/// The resident formula table: uploaded once, never per request.
pub struct DeviceFormulaTable<R: Runtime, E: FloatElem> {
    /// Table rows.
    pub rows: usize,
    /// Largest per-row arithmetic bound (the `formula_window` scalar).
    pub max_error: u32,
    /// `[R, 2]`: integer mass, per-row arithmetic bound.
    pub table: IdTensor<R>,
    /// `[R, 10]`: `ln(1 + count)` per element in [`ELEMENTS`] order.
    pub features: Tensor<R, E>,
    /// `[R, 10]` u32 exact element counts in [`ELEMENTS`] order, uploaded
    /// once so [`crate::tensor::ops::ms2::init_trajectories`] reads exact
    /// chemistry budgets without a float round-trip.
    pub counts: IdTensor<R>,
    /// SHA-256 of [`FormulaTable::to_json`]: the fingerprint binding the
    /// upload to the checkpoint's table (checked by [`check`]).
    ///
    /// [`check`]: DeviceFormulaTable::check
    pub sha256: String,
}

impl<R: Runtime, E: FloatElem> DeviceFormulaTable<R, E> {
    /// Upload `table` once: exactly 3 uploads (the integer search table,
    /// the float row features and the exact element counts), no launch,
    /// no read.
    pub fn upload(table: &FormulaTable, device: &Device<R>) -> Result<Self> {
        let rows = table.len();
        let max_error = table.max_error();
        let mut ids = Vec::with_capacity(rows * 2);
        let mut feats = Vec::with_capacity(rows * FORMULA_ELEMENTS);
        let mut counts = Vec::with_capacity(rows * FORMULA_ELEMENTS);
        for row in 0..rows {
            let composition = table.composition(row);
            let mass = table.mass(row);
            // The same bound the reference search derives per row.
            let error = composition_error_nda(composition).div_ceil(1000) as u32;
            ids.push(mass);
            ids.push(error);
            for count in composition.iter() {
                feats.push((1.0 + f32::from(*count)).ln());
                counts.push(u32::from(*count));
            }
        }
        let table_t = IdTensor::from_slice(&ids, vec![rows, 2], device)?;
        let features_t = Tensor::<R, E>::from_f32(&feats, vec![rows, FORMULA_ELEMENTS], device)?;
        let counts_t = IdTensor::from_slice(&counts, vec![rows, FORMULA_ELEMENTS], device)?;
        // The fingerprint is the SHA-256 of `FormulaTable::to_json()` (the
        // same bytes `tools/ms2/formula_table.py` hashes), binding the upload
        // to the exact table the checkpoint names.
        let sha256 = sha256_hex(table.to_json().as_bytes());
        Ok(Self {
            rows,
            max_error,
            table: table_t,
            features: features_t,
            counts: counts_t,
            sha256,
        })
    }

    /// Check the upload against the checkpoint's table reference: the row
    /// count and the SHA-256 must both match `model.formula_table`, or this
    /// is [`Error::Config`] naming both values.
    pub fn check(&self, model: &ModelConfig) -> Result<()> {
        if self.rows as u32 != model.formula_table.rows {
            return Err(Error::Config(format!(
                "DeviceFormulaTable::check: uploaded {} rows but model.formula_table has {} rows",
                self.rows, model.formula_table.rows
            )));
        }
        if self.sha256 != model.formula_table.sha256 {
            return Err(Error::Config(format!(
                "DeviceFormulaTable::check: uploaded sha256 {} does not match model.formula_table sha256 {}",
                self.sha256, model.formula_table.sha256
            )));
        }
        Ok(())
    }
}

/// The formula head of architecture §4.2: row embeddings from the table
/// features, dotted with a query projection of the pooled spectrum vector.
pub struct FormulaHead<R: Runtime, E: FloatElem> {
    /// `Linear(10 → d)` over the row features.
    row_in: Linear<R, E>,
    /// `Linear(d → d)` over the SiLU hidden state.
    row_out: Linear<R, E>,
    /// `Linear(d → d)` query projection of the pooled spectrum vector.
    pool_query: Linear<R, E>,
    /// Residual width.
    d_model: usize,
}

/// Output of [`FormulaHead::score`].
pub struct FormulaOutput<R: Runtime, E: FloatElem> {
    /// `[B, M]` masked log-softmax over the joined rows.
    pub log_prob: Var<R, E>,
    /// `[B, M, d]` row embeddings.
    pub embedding: Var<R, E>,
    /// `[B, M]` 1/0 join mask (0 for every slot of a row with no joined
    /// slot, whose logits still use the all-masked rule of architecture
    /// §3.8).
    pub mask: Tensor<R, E>,
}

impl<R: Runtime, E: FloatElem> FormulaHead<R, E> {
    /// Build the head for `model` on `device`.
    pub fn init(model: &ModelConfig, device: &Device<R>, rng: &mut Rng) -> Result<Self> {
        let d = model.d_model as usize;
        let row_in = LinearConfig::new(FORMULA_ELEMENTS, d).init(device, rng);
        let row_out = LinearConfig::new(d, d).init(device, rng);
        let pool_query = LinearConfig::new(d, d).init(device, rng);
        Ok(Self {
            row_in,
            row_out,
            pool_query,
            d_model: d,
        })
    }

    /// Score the window in `buffers` against `pool` (`[B, d]`): look the
    /// window rows up with [`Var::ms2_lookup`] (`u32::MAX` gives zero rows),
    /// `e = Linear(SiLU(Linear(features)))`, `score = (e · Linear(pool)) /
    /// sqrt(d)`, `mask_logits` with the window mask, `log_softmax` over `M`.
    /// A row with no joined slot uses the all-masked rule of architecture
    /// §3.8 (mask 1 at slot 0 for the logits) and is reported by [`mask`]
    /// with all zeros.
    ///
    /// [`mask`]: FormulaOutput::mask
    pub fn score(
        &self,
        table: &DeviceFormulaTable<R, E>,
        buffers: &FormulaBuffers<R, E>,
        pool: &Var<R, E>,
    ) -> Result<FormulaOutput<R, E>> {
        // Every input rank is checked before any dimension is read, so a
        // malformed shape is `Error::Shape` rather than a panic.
        if buffers.window.shape().rank() != 3
            || pool.rank() != 2
            || table.features.shape().rank() != 2
        {
            return Err(Error::shape(format!(
                "FormulaHead::score needs window [B, M, 2], pool [B, d] and features [R, 10], got {} and {} and {}",
                buffers.window.shape(),
                pool.shape(),
                table.features.shape()
            )));
        }
        let batch = buffers.window.shape().dim(0);
        let m = buffers.window.shape().dim(1);
        let want_window: &[usize] = &[batch, m, 2];
        if buffers.window.shape().dims() != want_window {
            return Err(Error::shape(format!(
                "FormulaHead::score needs window [B, M, 2], got {}",
                buffers.window.shape()
            )));
        }
        let want_pool: &[usize] = &[batch, self.d_model];
        let want_features: &[usize] = &[table.rows, FORMULA_ELEMENTS];
        if pool.dims() != want_pool || table.features.shape().dims() != want_features {
            return Err(Error::shape(format!(
                "FormulaHead::score needs pool [B, d] with d = {} and features [R, 10], got {} and {}",
                self.d_model,
                pool.shape(),
                table.features.shape()
            )));
        }
        // Window rows as flat `[B * M]` ids (column 0 of the window).
        let ids = slice_ids_along(&buffers.window, 2, 0, 1)?.reshape(vec![batch * m])?;
        let feats = Var::constant(table.features.clone());
        let looked = Var::ms2_lookup(&feats, &ids)?.reshape(vec![batch, m, FORMULA_ELEMENTS])?;
        let e = self.row_out.apply(&self.row_in.apply(&looked)?.silu()?)?;
        let query = self.pool_query.apply(pool)?;
        let query = query.unsqueeze(1)?.expand(vec![batch, m, self.d_model])?;
        let scores = e
            .mul(&query)?
            .sum_dim(2)?
            .squeeze(2)?
            .mul_scalar(1.0 / (self.d_model as f32).sqrt());
        // The join mask: 1 where the window flag is non-zero. A row with no
        // joined slot keeps the reported all-zero mask but scores with the
        // all-masked rule (mask 1 at slot 0 only), so its log-softmax is 0
        // at slot 0 rather than uniform.
        let mask = nonzero_mask(&buffers.window)?;
        let eff = if m == 0 {
            mask.clone()
        } else {
            let has = reduce::sum_dim(&mask, 1)?;
            let empty = elemwise::eq_scalar(&has, 0.0);
            let first = movement::slice(&mask, 1, 0, 1)?;
            let new_first = elemwise::maximum(&first, &empty)?;
            if m == 1 {
                new_first
            } else {
                let rest = movement::slice(&mask, 1, 1, m - 1)?;
                movement::cat(&[new_first, rest], 1)?
            }
        };
        let log_prob = scores.mask_logits(&eff)?.log_softmax(1)?;
        Ok(FormulaOutput {
            log_prob,
            embedding: e,
            mask,
        })
    }

    /// `L_formula = sum over spectra with a gold slot of
    /// −log_prob[b, gold] / max(1, count of such spectra)`, computed on the
    /// device with no read: the gold slot picks its log-probability, the
    /// `u32::MAX` slots are masked out, and the count is a device reduction,
    /// never a host value. `gold_slot` is `[B]` (`u32::MAX` when the gold
    /// formula is not scored).
    pub fn loss(&self, out: &FormulaOutput<R, E>, gold_slot: &IdTensor<R>) -> Result<Var<R, E>> {
        // Every input rank is checked before any dimension is read, so a
        // malformed shape is `Error::Shape` rather than a panic.
        if out.log_prob.rank() != 2 || gold_slot.shape().rank() != 1 {
            return Err(Error::shape(format!(
                "FormulaHead::loss needs log_prob [B, M] and gold_slot [B], got {} and {}",
                out.log_prob.shape(),
                gold_slot.shape()
            )));
        }
        let batch = out.log_prob.dims()[0];
        if gold_slot.len() != batch {
            return Err(Error::shape(format!(
                "FormulaHead::loss needs log_prob [B, M] and gold_slot [B], got {} and {}",
                out.log_prob.shape(),
                gold_slot.shape()
            )));
        }
        // The scored slot per spectrum, safe at 0 for the masked-out rows.
        let safe = safe_ids(gold_slot, 0)?;
        let picked = out.log_prob.take_along_last(&safe)?;
        // Validity without a read: `u32::MAX` casts to `2^32` in `f32`, every
        // real slot well below it.
        let gold_f: Tensor<R, E> = ids_to_float(gold_slot);
        let is_absent = elemwise::eq_scalar(&gold_f, u32::MAX as f32);
        let valid = elemwise::rsub_scalar(&is_absent, 1.0);
        let valid_var = Var::constant(valid);
        let masked = picked.mul(&valid_var)?.neg();
        let total = masked.sum()?;
        let denom_unclamped = valid_var.sum()?;
        let one = Var::constant(Tensor::full(
            denom_unclamped.shape().clone(),
            1.0,
            out.log_prob.device(),
        ));
        let denom = denom_unclamped.maximum(&one)?;
        total.div(&denom)
    }
}

impl<R: Runtime, E: FloatElem> Module<R, E> for FormulaHead<R, E> {
    fn visit(&self, visitor: &mut ModuleVisitor<'_, R, E>) {
        visitor.child("row_in", &self.row_in);
        visitor.child("row_out", &self.row_out);
        visitor.child("pool_query", &self.pool_query);
    }
}

/// Host helper for data preparation only: the slot of each spectrum's gold
/// table row in a host copy of the `[B, M, 2]` window flat, `u32::MAX` when
/// the gold row has no slot. `gold_rows` holds one table row per spectrum
/// (`u32::MAX` when the spectrum has no gold formula).
pub fn gold_slots(window: &[u32], gold_rows: &[u32]) -> Vec<u32> {
    let batch = gold_rows.len();
    if batch == 0 {
        return Vec::new();
    }
    let m = window.len() / (batch * 2);
    let mut out = vec![u32::MAX; batch];
    for b in 0..batch {
        let gold = gold_rows[b];
        if gold == u32::MAX {
            continue;
        }
        for slot in 0..m {
            if window[(b * m + slot) * 2] == gold {
                out[b] = slot as u32;
                break;
            }
        }
    }
    out
}

/// Host helper for the training loop: the window slot of each spectrum's gold
/// table row, `u32::MAX` when the gold formula is not scored. `queries` and
/// `gold_rows` hold one entry per spectrum (`u32::MAX` gold when the spectrum
/// has no gold formula).
///
/// `window_capacity` is the device window width `M`: the query runs with
/// `rows_scored_max = min(rows_scored_max, M)`, the same scored cap the
/// [`crate::tensor::ops::ms2::formula_window`] kernel applies, so a gold row
/// past the cap reports `u32::MAX` here exactly as on the device.
///
/// No device read is involved: the window is deterministic, so the host
/// reference [`FormulaTable::window`] — the exact search the
/// [`crate::tensor::ops::ms2::formula_window`] kernel reproduces, counters
/// included — gives the same slots as the kernel without reading the device.
/// The training loop computes this once per batch on the host from the formula
/// table and the gold composition.
pub fn gold_slots_host(
    table: &FormulaTable,
    queries: &[WindowQuery],
    gold_rows: &[u32],
    window_capacity: usize,
) -> Vec<u32> {
    assert_eq!(
        queries.len(),
        gold_rows.len(),
        "one query and one gold row per spectrum"
    );
    let mut out = vec![u32::MAX; gold_rows.len()];
    for (b, (&gold, query)) in gold_rows.iter().zip(queries.iter()).enumerate() {
        if gold == u32::MAX {
            continue;
        }
        // The same scored cap as the kernel: at most the first
        // `min(rows_scored_max, M)` joined rows are scored.
        let capped = WindowQuery {
            rows_scored_max: query.rows_scored_max.min(window_capacity as u32),
            ..*query
        };
        let found = table.window(&capped);
        // The joined rows fill the window slots in table order, so the slot
        // is the index within the scored prefix.
        for (slot, &row) in found.joined.iter().enumerate() {
            if slot >= found.rows_scored as usize {
                break;
            }
            if row as u32 == gold {
                out[b] = slot as u32;
                break;
            }
        }
    }
    out
}
