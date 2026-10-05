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
use crate::tensor::ops::index::{IdTensor, ids_to_float};
use crate::tensor::ops::ms2::{FormulaBuffers, cand_mask, safe_ids};
use crate::tensor::ops::random::Rng;
use crate::tensor::ops::{elemwise, movement, reduce};

use super::chem::{ELEMENTS, composition_error_nda};
use super::contract::{FormulaFeatures, ModelConfig};
use super::formula::{FormulaTable, WindowQuery};
use super::formula_enum::{
    EnumDomain, RatioBounds, pack_device_bounds, rare_table, validate_device_artifacts,
};
use super::workspace::Ms2Capabilities;

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
    /// Largest hydrogen count over all rows (for the evidence dispatch
    /// sizing: `h_cap_max = max_hydrogen + 3`, known without a read).
    pub max_hydrogen: u16,
    /// `[R, 2]`: integer mass, per-row arithmetic bound.
    pub table: IdTensor<R>,
    /// `[R, 10]`: `ln(1 + count)` per element in [`ELEMENTS`] order.
    pub features: Tensor<R, E>,
    /// `[R, 10]` u32 exact element counts in [`ELEMENTS`] order, uploaded
    /// once for the table-source gather.
    pub counts: IdTensor<R>,
    /// `[1024]` resident `ln(1 + n)` table (V1 §1.2): `log_table[n]` is the
    /// host's `(1.0 + n as f32).ln()` uploaded once, so `count_features` is
    /// the same bits as V0's uploaded table features on every backend.
    pub log_table: Tensor<R, E>,
    /// SHA-256 of [`FormulaTable::to_json`]: the fingerprint binding the
    /// upload to the checkpoint's table (checked by [`check`]).
    ///
    /// [`check`]: DeviceFormulaTable::check
    pub sha256: String,
}

impl<R: Runtime, E: FloatElem> DeviceFormulaTable<R, E> {
    /// Upload `table` once: exactly 4 uploads (the integer search table,
    /// the float row features, the exact element counts and the resident
    /// `log_table [1024]`), no launch, no read.
    ///
    /// A table row with an element count above 1023 is refused
    /// (`Error::Config`): the device `log_table` has 1024 entries and V1
    /// applies this bound whatever the document's version.
    pub fn upload(table: &FormulaTable, device: &Device<R>) -> Result<Self> {
        // Dtype gate (contracts §3.3): the actual neural element type must be
        // in the validated set — refused before any upload, so it can never
        // reach a kernel launch.
        Ms2Capabilities::check_dtype(&device.name(), E::DTYPE)?;
        let rows = table.len();
        let max_error = table.max_error();
        let max_hydrogen = table.max_hydrogen();
        let mut ids = Vec::with_capacity(rows * 2);
        let mut feats = Vec::with_capacity(rows * FORMULA_ELEMENTS);
        let mut counts = Vec::with_capacity(rows * FORMULA_ELEMENTS);
        for row in 0..rows {
            let composition = table.composition(row);
            let mass = table.mass(row);
            for (e, count) in composition.iter().enumerate() {
                if u32::from(*count) > 1023 {
                    return Err(Error::config(format!(
                        "DeviceFormulaTable::upload: row {row} element {e} count {count} exceeds 1023 (V1 device bound)"
                    )));
                }
            }
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
        let log_feats: Vec<f32> = (0..1024).map(|n| (1.0 + n as f32).ln()).collect();
        let log_table = Tensor::<R, E>::from_f32(&log_feats, vec![1024], device)?;
        // The fingerprint is the SHA-256 of `FormulaTable::to_json()` (the
        // same bytes `tools/ms2/formula_table.py` hashes), binding the upload
        // to the exact table the checkpoint names.
        let sha256 = sha256_hex(table.to_json().as_bytes());
        Ok(Self {
            rows,
            max_error,
            max_hydrogen,
            table: table_t,
            features: features_t,
            counts: counts_t,
            log_table,
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

    /// Upper bound of `c[H] + 3` over every candidate the table source can
    /// score (task E4F item 2): the table's largest hydrogen count plus 3
    /// (the lane's `h_pos + 2` with `h_pos <= 1`), known on the host without
    /// a read.
    pub fn hydrogen_cap_max(&self) -> u32 {
        u32::from(self.max_hydrogen) + 3
    }
}

/// Resident enumeration artifacts (V1 §1.4): the rare table `[P, 8]` and
/// the packed bounds, uploaded once and reused by the enumerating source.
///
/// `upload` validates with [`validate_device_artifacts`], uploads `rare`
/// and the packed `bounds`, and records the domain/bounds versions and the
/// SHA-256 of the two artifacts' JSON. A version-1 or table-only config
/// has none (`ModelConfig::formula_artifacts` is `None`); a mismatch at
/// load is [`Error::Config`].
pub struct DeviceEnumArtifacts<R: Runtime> {
    /// Rare combinations (`P` rows).
    pub p: usize,
    /// `[P, 8]` rare rows (6 counts, mass, sum).
    pub rare: IdTensor<R>,
    /// Packed bounds buffer.
    pub bounds: IdTensor<R>,
    /// Largest per-composition arithmetic bound over the domain (for the
    /// superset window `half`).
    pub domain_max_error: u32,
    /// Maximum hydrogen count of the enum domain (for the evidence dispatch
    /// sizing: `h_cap_max = hydrogen_max + 3`, known without a read).
    pub hydrogen_max: u16,
    /// Enum domain version string.
    pub domain_version: String,
    /// Ratio bounds version string.
    pub bounds_version: String,
    /// SHA-256 of the enum domain JSON.
    pub domain_sha256: String,
    /// SHA-256 of the ratio bounds JSON.
    pub bounds_sha256: String,
    /// Enum domain JSON (for checkpointing).
    pub domain_json: String,
    /// Ratio bounds JSON (for checkpointing).
    pub bounds_json: String,
}

impl<R: Runtime> DeviceEnumArtifacts<R> {
    /// Validate and upload `domain` and `bounds` once: exactly 2 uploads
    /// (the rare table and the packed bounds), no launch, no read.
    pub fn upload(
        domain: &EnumDomain,
        bounds: &RatioBounds,
        device: &Device<R>,
    ) -> Result<Self> {
        validate_device_artifacts(domain, bounds)?;
        let rows = rare_table(domain, bounds)?;
        let packed = pack_device_bounds(domain, bounds)?;
        let p = rows.len();
        let mut flat = Vec::with_capacity(p * 8);
        for row in &rows {
            flat.extend_from_slice(row);
        }
        let rare = IdTensor::from_slice(&flat, vec![p, 8], device)?;
        let bounds_t = IdTensor::from_slice(&packed, vec![packed.len()], device)?;
        let domain_json = domain.to_json();
        let bounds_json = bounds.to_json();
        let hydrogen_max = domain.hydrogen_max;
        Ok(Self {
            p,
            rare,
            bounds: bounds_t,
            domain_max_error: domain.max_error(),
            hydrogen_max,
            domain_version: domain.version.clone(),
            bounds_version: bounds.version.clone(),
            domain_sha256: sha256_hex(domain_json.as_bytes()),
            bounds_sha256: sha256_hex(bounds_json.as_bytes()),
            domain_json,
            bounds_json,
        })
    }

    /// Check the upload against the checkpoint's artifact reference: both
    /// versions and both SHA-256 hashes must match
    /// `model.formula_artifacts`, or this is [`Error::Config`] naming both
    /// values. A table-only config (`None`) is `Error::Config` here: the
    /// enumerating source needs resident artifacts.
    pub fn check(&self, model: &ModelConfig) -> Result<()> {
        let Some(want) = model.formula_artifacts.as_ref() else {
            return Err(Error::Config(
                "DeviceEnumArtifacts::check: model has no formula_artifacts (table-only config)".to_string(),
            ));
        };
        if self.domain_version != want.domain_version {
            return Err(Error::Config(format!(
                "DeviceEnumArtifacts::check: uploaded domain version {:?} does not match model {:?}",
                self.domain_version, want.domain_version
            )));
        }
        if self.domain_sha256 != want.domain_sha256 {
            return Err(Error::Config(format!(
                "DeviceEnumArtifacts::check: uploaded domain sha256 {} does not match model {}",
                self.domain_sha256, want.domain_sha256
            )));
        }
        if self.bounds_version != want.bounds_version {
            return Err(Error::Config(format!(
                "DeviceEnumArtifacts::check: uploaded bounds version {:?} does not match model {:?}",
                self.bounds_version, want.bounds_version
            )));
        }
        if self.bounds_sha256 != want.bounds_sha256 {
            return Err(Error::Config(format!(
                "DeviceEnumArtifacts::check: uploaded bounds sha256 {} does not match model {}",
                self.bounds_sha256, want.bounds_sha256
            )));
        }
        Ok(())
    }

    /// Upper bound of `c[H] + 3` over every candidate the enumerating source
    /// can score (task E4F item 2): the artifacts' hydrogen bound plus 3
    /// (the lane's `h_pos + 2` with `h_pos <= 1`), known on the host without
    /// a read.
    pub fn hydrogen_cap_max(&self) -> u32 {
        u32::from(self.hydrogen_max) + 3
    }
}

/// The formula head of architecture §4.2: row embeddings from the table
/// features, dotted with a query projection of the pooled spectrum vector.
///
/// In the `Evidence` layout (architecture §1.6) the head additionally owns
/// the additive evidence branch `Linear(6 → 32)`, SiLU, `Linear(32 → 1)`
/// over features 10..16, whose scalar output is added to the score. The
/// branch exists only in that layout; a `Counts` head has exactly the
/// parameters, names and operations of §1.2.
pub struct FormulaHead<R: Runtime, E: FloatElem> {
    /// `Linear(10 → d)` over the row features.
    row_in: Linear<R, E>,
    /// `Linear(d → d)` over the SiLU hidden state.
    row_out: Linear<R, E>,
    /// `Linear(d → d)` query projection of the pooled spectrum vector.
    pool_query: Linear<R, E>,
    /// `Linear(6 → 32)` evidence branch input (architecture §1.6):
    /// `Some` only in the `Evidence` layout.
    evidence_in: Option<Linear<R, E>>,
    /// `Linear(32 → 1)` evidence branch output, zero-initialised so an
    /// untrained branch adds exactly 0: `Some` only in the `Evidence`
    /// layout.
    evidence_out: Option<Linear<R, E>>,
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
    ///
    /// With `FormulaFeatures::Counts` this is exactly the §1.2 head. With
    /// `Evidence` it additionally builds the evidence branch
    /// (`evidence_in: Linear(6 → 32)` with the default initialiser,
    /// `evidence_out: Linear(32 → 1)` with zero weight and bias, so an
    /// untrained branch adds exactly 0).
    pub fn init(model: &ModelConfig, device: &Device<R>, rng: &mut Rng) -> Result<Self> {
        let d = model.d_model as usize;
        let row_in = LinearConfig::new(FORMULA_ELEMENTS, d).init(device, rng);
        let row_out = LinearConfig::new(d, d).init(device, rng);
        let pool_query = LinearConfig::new(d, d).init(device, rng);
        let (evidence_in, evidence_out) = match model.formula_features {
            FormulaFeatures::Counts => (None, None),
            FormulaFeatures::Evidence => {
                use crate::nn::init::Initializer;
                let evidence_in = LinearConfig::new(6, 32).init(device, rng);
                let evidence_out = LinearConfig::new(32, 1)
                    .with_initializer(Initializer::Zeros)
                    .with_bias_initializer(Initializer::Zeros)
                    .init(device, rng);
                (Some(evidence_in), Some(evidence_out))
            }
        };
        Ok(Self {
            row_in,
            row_out,
            pool_query,
            evidence_in,
            evidence_out,
            d_model: d,
        })
    }

    /// Whether this head owns the evidence branch (the `Evidence` layout).
    pub fn has_evidence_branch(&self) -> bool {
        self.evidence_in.is_some()
    }

    /// Row embedding of an arbitrary feature tensor whose last dimension is
    /// 10 (`ln(1 + count)` in [`ELEMENTS`] order): `e = Linear(SiLU(Linear))`
    /// with the head's own weights, in the same op order as [`score`], so the
    /// same features give the same bits. Used for the gold composition
    /// (teacher forcing, V1 §1.2). No device read.
    pub fn embed_rows(&self, features: &Var<R, E>) -> Result<Var<R, E>> {
        let rank = features.rank();
        if rank < 1 || features.dims()[rank - 1] != FORMULA_ELEMENTS {
            return Err(Error::shape(format!(
                "FormulaHead::embed_rows needs features [.., 10], got {}",
                features.shape()
            )));
        }
        self.row_out.apply(&self.row_in.apply(features)?.silu()?)
    }

    /// Score the candidates in `buffers` against `pool` (`[B, d]`) from
    /// `cand_feat` (V1 §1.2): `e = Linear(SiLU(Linear(cand_feat)))` via
    /// [`embed_rows`], `score = (e · Linear(pool)) / sqrt(d)`, `mask_logits`
    /// with the `cand` mask, `log_softmax` over `M`. Same weights and same op
    /// order as V0, so the same features give the same bits for the same
    /// features. A spectrum with an empty support (no flagged slot) uses the
    /// all-masked rule of architecture §3.8 (mask 1 at slot 0 for the logits)
    /// and is reported by [`mask`] with all zeros.
    ///
    /// In the `Evidence` layout (architecture §1.6) the row network still
    /// embeds features 0..10 exactly as above, and the scalar output of the
    /// evidence branch over `cand_xfeat [B, M, 6]` (features 10..16,
    /// constants of the graph: no gradient flows into them) is added to the
    /// score before the mask: `score = (e . q) / sqrt(d) + branch(x)`. A
    /// head with the branch returns [`Error::Shape`] when `cand_xfeat` is
    /// missing or mis-shaped; a head without the branch ignores it.
    ///
    /// [`mask`]: FormulaOutput::mask
    /// [`embed_rows`]: FormulaHead::embed_rows
    pub fn score(
        &self,
        buffers: &FormulaBuffers<R, E>,
        pool: &Var<R, E>,
    ) -> Result<FormulaOutput<R, E>> {
        // Every input rank is checked before any dimension is read, so a
        // malformed shape is `Error::Shape` rather than a panic.
        if buffers.cand.shape().rank() != 3
            || buffers.cand_feat.shape().rank() != 3
            || pool.rank() != 2
        {
            return Err(Error::shape(format!(
                "FormulaHead::score needs cand [B, M, 13], cand_feat [B, M, 10] and pool [B, d], got {} and {} and {}",
                buffers.cand.shape(),
                buffers.cand_feat.shape(),
                pool.shape(),
            )));
        }
        let batch = buffers.cand.shape().dim(0);
        let m = buffers.cand.shape().dim(1);
        if buffers.cand.shape().dims() != [batch, m, 13]
            || buffers.cand_feat.shape().dims() != [batch, m, 10]
        {
            return Err(Error::shape(format!(
                "FormulaHead::score needs cand [B, M, 13] and cand_feat [B, M, 10], got {} and {}",
                buffers.cand.shape(),
                buffers.cand_feat.shape(),
            )));
        }
        let want_pool: &[usize] = &[batch, self.d_model];
        if pool.dims() != want_pool {
            return Err(Error::shape(format!(
                "FormulaHead::score needs pool [B, d] with d = {}, got {}",
                self.d_model,
                pool.shape(),
            )));
        }
        let feats = Var::constant(buffers.cand_feat.clone()).reshape(vec![batch, m, FORMULA_ELEMENTS])?;
        let e = self.embed_rows(&feats)?;
        let query = self.pool_query.apply(pool)?;
        let query = query.unsqueeze(1)?.expand(vec![batch, m, self.d_model])?;
        let mut scores = e
            .mul(&query)?
            .sum_dim(2)?
            .squeeze(2)?
            .mul_scalar(1.0 / (self.d_model as f32).sqrt());
        // The evidence branch (architecture §1.6): features 10..16 are
        // constants of the graph (wrapped in `Var::constant`, so no
        // gradient flows into them); gradients flow into the branch
        // parameters through the two linears.
        if let (Some(ev_in), Some(ev_out)) = (&self.evidence_in, &self.evidence_out) {
            let Some(xfeat) = buffers.cand_xfeat.as_ref() else {
                return Err(Error::shape(format!(
                    "FormulaHead::score needs cand_xfeat [B, M, 6] in the Evidence layout, got none (batch {batch}, M {m})"
                )));
            };
            if xfeat.shape().dims() != [batch, m, 6] {
                return Err(Error::shape(format!(
                    "FormulaHead::score needs cand_xfeat [{batch}, {m}, 6], got {}",
                    xfeat.shape()
                )));
            }
            let x = Var::constant(xfeat.clone()).reshape(vec![batch, m, 6])?;
            let hidden = ev_in.apply(&x)?.silu()?;
            let branch = ev_out.apply(&hidden)?.squeeze(2)?;
            scores = scores.add(&branch)?;
        }
        // The join mask: 1 where the cand flag is non-zero. A spectrum with
        // no scored candidate keeps the reported all-zero mask but scores
        // with the all-masked rule (mask 1 at slot 0 only), so its
        // log-softmax is 0 at slot 0 rather than uniform (empty-support
        // handling, V1 §1.2).
        let mask = cand_mask(&buffers.cand)?;
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
        if let Some(ev_in) = &self.evidence_in {
            visitor.child("evidence_in", ev_in);
        }
        if let Some(ev_out) = &self.evidence_out {
            visitor.child("evidence_out", ev_out);
        }
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
