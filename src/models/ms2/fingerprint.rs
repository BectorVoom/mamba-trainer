//! Whole-parent fingerprint auxiliary supervision (task K10 / P7.3).
//!
//! Design `docs/MS2_SUBSTRUCTURE_DESIGN.md` §8: "An optional FPNet-style
//! fingerprint head supplies whole-parent auxiliary supervision. It is not a
//! connectivity representation or a required bottleneck."
//!
//! Every spectrum of a molecule shares that molecule's whole-parent
//! fingerprint: the Morgan radius-2 1024-bit fingerprint of the complete
//! parent molecule (see `tools/ms2/export_fingerprints.py`). This is
//! whole-parent auxiliary supervision only. It is never used as a target for
//! an individual fragment or candidate: a whole-parent fingerprint is not a
//! target fingerprint for every subgraph (design §3.6), so the loss and the
//! labels below must not be wired to per-fragment or per-candidate targets.
//! Standalone components: no hook into the model, the trainer or the drivers
//! (a later integration task wires the loss weight `lambda_fp`, default 0 =
//! off, the `pool` input from the encoder and the sidecar in the driver).

use cubecl::prelude::Runtime;
use serde::Deserialize;

use crate::autograd::Var;
use crate::backend::{Device, FloatElem};
use crate::error::{Error, Result};
use crate::nn::linear::{Linear, LinearConfig};
use crate::nn::module::{Module, ModuleVisitor};
use crate::tensor::Tensor;
use crate::tensor::ops::random::Rng;

use super::dataset::ExportFile;
use super::rerank::bce_with_logits;

/// Bits per whole-parent fingerprint.
pub const FINGERPRINT_BITS: usize = 1024;
/// Floor for the spectrum-weight sum in [`FingerprintHead::loss`]: the loss is
/// the weighted mean `sum_b w_b * l_b / max(sum_b w_b, eps)` of the
/// per-spectrum bit-means `l_b`, with `eps` tiny so fractional weight sums
/// below one keep their full weight (one spectrum at weight 0.5 and zero
/// logits against all-zero targets gives exactly `ln 2`); a zero weight sum
/// gives exactly 0 (the numerator is 0 then).
pub const FINGERPRINT_WEIGHT_EPS: f32 = 1e-12;
/// Fingerprint id shared by the exporter, the sidecar and these labels.
pub const FINGERPRINT_VERSION: &str = "morgan-r2-1024";
/// Sidecar schema version `tools/ms2/export_fingerprints.py` writes.
const SIDECAR_SCHEMA_VERSION: u32 = 1;

/// One molecule row of the fingerprint sidecar.
#[derive(Deserialize)]
struct SidecarMolecule {
    /// Stable molecule key (same order as the export).
    key: String,
    /// Parent SMILES the fingerprint was computed from.
    smiles: String,
    /// Sorted on-bit indices of the whole-parent fingerprint.
    bits: Vec<u32>,
}

/// The JSON sidecar `tools/ms2/export_fingerprints.py` writes.
#[derive(Deserialize)]
struct Sidecar {
    /// Schema version; only [`SIDECAR_SCHEMA_VERSION`] is accepted.
    schema_version: u32,
    /// Fingerprint id; must be [`FINGERPRINT_VERSION`].
    fingerprint: String,
    /// Bit width; must be [`FINGERPRINT_BITS`].
    bits: usize,
    /// RDKit version the sidecar was computed under.
    rdkit: String,
    /// Base file name of the source export.
    source_export: String,
    /// SHA-256 of the source export file bytes.
    source_sha256: String,
    /// One row per molecule, in export order.
    molecules: Vec<SidecarMolecule>,
}

/// Whole-parent fingerprint labels for one export.
///
/// Row `m` holds the Morgan radius-2 1024-bit fingerprint of molecule `m`'s
/// complete parent structure. Every spectrum of that molecule shares this one
/// row: whole-parent auxiliary supervision, never a target for an individual
/// fragment or candidate.
pub struct FingerprintLabels {
    /// Fingerprint id ([`FINGERPRINT_VERSION`]).
    pub version: String,
    /// RDKit version the sidecar was computed under.
    pub rdkit: String,
    /// Base file name of the source export.
    pub source_export: String,
    /// SHA-256 of the source export file bytes.
    pub source_sha256: String,
    /// Stable molecule keys, in export order.
    pub keys: Vec<String>,
    /// Parent SMILES per molecule, in export order.
    pub smiles: Vec<String>,
    /// Sorted on-bit indices per molecule, in export order.
    pub bits: Vec<Vec<u32>>,
}

impl FingerprintLabels {
    /// Parse the sidecar from a file path or from JSON text.
    ///
    /// When `path_or_text` names an existing file it is read; otherwise it
    /// is parsed as JSON directly. Rejects a schema version, fingerprint id
    /// or bit width it does not know, and any on-bit index at or above
    /// [`FINGERPRINT_BITS`] (all [`Error::Config`]).
    pub fn from_json(path_or_text: &str) -> Result<Self> {
        let text = if std::path::Path::new(path_or_text).is_file() {
            std::fs::read_to_string(path_or_text)?
        } else {
            path_or_text.to_string()
        };
        let sidecar: Sidecar = serde_json::from_str(&text)?;
        if sidecar.schema_version != SIDECAR_SCHEMA_VERSION {
            return Err(Error::config(format!(
                "FingerprintLabels::from_json: schema_version {} does not match {SIDECAR_SCHEMA_VERSION}",
                sidecar.schema_version
            )));
        }
        if sidecar.fingerprint != FINGERPRINT_VERSION {
            return Err(Error::config(format!(
                "FingerprintLabels::from_json: fingerprint {:?} does not match {FINGERPRINT_VERSION:?}",
                sidecar.fingerprint
            )));
        }
        if sidecar.bits != FINGERPRINT_BITS {
            return Err(Error::config(format!(
                "FingerprintLabels::from_json: bits {} does not match {FINGERPRINT_BITS}",
                sidecar.bits
            )));
        }
        let mut keys = Vec::with_capacity(sidecar.molecules.len());
        let mut smiles = Vec::with_capacity(sidecar.molecules.len());
        let mut bits = Vec::with_capacity(sidecar.molecules.len());
        for mol in sidecar.molecules {
            for &bit in &mol.bits {
                if bit as usize >= FINGERPRINT_BITS {
                    return Err(Error::config(format!(
                        "FingerprintLabels::from_json: molecule {:?} bit {bit} out of range",
                        mol.key
                    )));
                }
            }
            let mut row = mol.bits;
            row.sort_unstable();
            row.dedup();
            keys.push(mol.key);
            smiles.push(mol.smiles);
            bits.push(row);
        }
        Ok(Self {
            version: FINGERPRINT_VERSION.to_string(),
            rdkit: sidecar.rdkit,
            source_export: sidecar.source_export,
            source_sha256: sidecar.source_sha256,
            keys,
            smiles,
            bits,
        })
    }

    /// Molecules covered, in export order.
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    /// Whether the sidecar holds no molecule.
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// Check the sidecar covers exactly the export's molecules in the same
    /// order, with the same parent SMILES, and — when `export_bytes` (the raw
    /// export file bytes) is supplied — that the sidecar's `source_sha256`
    /// matches the bytes' SHA-256. Any length, key, SMILES or hash mismatch
    /// is [`Error::Config`] naming the first difference.
    ///
    /// The parsed [`ExportFile`] drops per-molecule SMILES (it models only the
    /// graph and spectra), so the SMILES comparison reads the `smiles` field
    /// of each entry of the export JSON's `molecules` array from
    /// `export_bytes`; export molecules without a `smiles` field are skipped
    /// for that row. Pass `None` only when the bytes are unavailable: then
    /// just the keys are checked.
    pub fn matches_export(
        &self,
        export: &ExportFile,
        export_bytes: Option<&[u8]>,
    ) -> Result<()> {
        if self.keys.len() != export.molecules.len() {
            return Err(Error::config(format!(
                "FingerprintLabels::matches_export: sidecar holds {} molecules but the export holds {}",
                self.keys.len(),
                export.molecules.len()
            )));
        }
        for (i, (mine, theirs)) in self
            .keys
            .iter()
            .zip(export.molecules.iter().map(|m| &m.key))
            .enumerate()
        {
            if mine != theirs {
                return Err(Error::config(format!(
                    "FingerprintLabels::matches_export: molecule {i} is {mine:?} in the sidecar but {theirs:?} in the export"
                )));
            }
        }
        let Some(bytes) = export_bytes else {
            return Ok(());
        };
        let digest = sha256_hex(bytes);
        if digest != self.source_sha256 {
            return Err(Error::config(format!(
                "FingerprintLabels::matches_export: sidecar source_sha256 {:?} does not match the export bytes' SHA-256 {digest:?}",
                self.source_sha256
            )));
        }
        let raw: serde_json::Value = serde_json::from_slice(bytes).map_err(|e| {
            Error::config(format!(
                "FingerprintLabels::matches_export: export bytes are not JSON: {e}"
            ))
        })?;
        let Some(mols) = raw.get("molecules").and_then(|v| v.as_array()) else {
            return Ok(());
        };
        for (i, (mol, want)) in mols.iter().zip(self.smiles.iter()).enumerate() {
            let Some(theirs) = mol.get("smiles").and_then(|v| v.as_str()) else {
                continue;
            };
            if theirs != want {
                return Err(Error::config(format!(
                    "FingerprintLabels::matches_export: molecule {i} SMILES {want:?} in the sidecar but {theirs:?} in the export"
                )));
            }
        }
        Ok(())
    }

    /// Dense 0/1 rows for a batch of spectra, row-major `[B, 1024]`.
    ///
    /// `molecule_indices` holds one molecule index per spectrum; every
    /// spectrum of a molecule shares that molecule's whole-parent
    /// fingerprint row.
    pub fn dense(&self, molecule_indices: &[usize]) -> Vec<f32> {
        let mut out = vec![0.0f32; molecule_indices.len() * FINGERPRINT_BITS];
        for (row, &m) in molecule_indices.iter().enumerate() {
            let base = row * FINGERPRINT_BITS;
            for &bit in &self.bits[m] {
                out[base + bit as usize] = 1.0;
            }
        }
        out
    }
}

/// SHA-256 of `bytes` as lowercase hex (FIPS 180-4, no dependencies).
///
/// Exposed so drivers and tests can recompute an export file's hash when
/// calling [`FingerprintLabels::matches_export`] with the export bytes.
pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut h: [u32; 8] = [
        0x6a09_e667,
        0xbb67_ae85,
        0x3c6e_f372,
        0xa54f_f53a,
        0x510e_527f,
        0x9b05_688c,
        0x1f83_d9ab,
        0x5be0_cd19,
    ];
    const K: [u32; 64] = [
        0x428a_2f98, 0x7137_4491, 0xb5c0_fbcf, 0xe9b5_dba5, 0x3956_c25b, 0x59f1_11f1,
        0x923f_82a4, 0xab1c_5ed5, 0xd807_aa98, 0x1283_5b01, 0x2431_85be, 0x550c_7dc3,
        0x72be_5d74, 0x80de_b1fe, 0x9bdc_06a7, 0xc19b_f174, 0xe49b_69c1, 0xefbe_4786,
        0x0fc1_9dc6, 0x240c_a1cc, 0x2de9_2c6f, 0x4a74_84aa, 0x5cb0_a9dc, 0x76f9_88da,
        0x983e_5152, 0xa831_c66d, 0xb003_27c8, 0xbf59_7fc7, 0xc6e0_0bf3, 0xd5a7_9147,
        0x06ca_6351, 0x1429_2967, 0x27b7_0a85, 0x2e1b_2138, 0x4d2c_6dfc, 0x5338_0d13,
        0x650a_7354, 0x766a_0abb, 0x81c2_c92e, 0x9272_2c85, 0xa2bf_e8a1, 0xa81a_664b,
        0xc24b_8b70, 0xc76c_51a3, 0xd192_e819, 0xd699_0624, 0xf40e_3585, 0x106a_a070,
        0x19a4_c116, 0x1e37_6c08, 0x2748_774c, 0x34b0_bcb5, 0x391c_0cb3, 0x4ed8_aa4a,
        0x5b9c_ca4f, 0x682e_6ff3, 0x748f_82ee, 0x78a5_636f, 0x84c8_7814, 0x8cc7_0208,
        0x90be_fffa, 0xa450_6ceb, 0xbef9_a3f7, 0xc671_78f2,
    ];
    let mut msg = bytes.to_vec();
    let bit_len = (bytes.len() as u64).wrapping_mul(8);
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());
    for block in msg.as_chunks::<64>().0 {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                block[4 * i],
                block[4 * i + 1],
                block[4 * i + 2],
                block[4 * i + 3],
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

/// Whole-parent fingerprint head: `Linear(d → d)`, SiLU, `Linear(d → 1024)`
/// on the pooled spectrum vector `pool [B, d]` (architecture §4.1).
///
/// The head predicts the whole parent's fingerprint as auxiliary supervision.
/// It is never used as a target for an individual fragment or candidate.
pub struct FingerprintHead<R: Runtime, E: FloatElem> {
    /// `Linear(d → d)` hidden projection.
    fc1: Linear<R, E>,
    /// `Linear(d → 1024)` bit-logit projection.
    fc2: Linear<R, E>,
    /// Pooled width.
    d_model: usize,
}

impl<R: Runtime, E: FloatElem> FingerprintHead<R, E> {
    /// Build the head for pooled width `d_model` on `device`.
    pub fn init(d_model: usize, device: &Device<R>, rng: &mut Rng) -> Result<Self> {
        if d_model == 0 {
            return Err(Error::config(
                "FingerprintHead::init: d_model must be at least 1".to_string(),
            ));
        }
        Ok(Self {
            fc1: LinearConfig::new(d_model, d_model).init(device, rng),
            fc2: LinearConfig::new(d_model, FINGERPRINT_BITS).init(device, rng),
            d_model,
        })
    }

    /// Pooled width this head was built for.
    pub fn d_model(&self) -> usize {
        self.d_model
    }

    /// Bit logits of `pool` (`[B, d]`), as `Var [B, 1024]`. No device read.
    pub fn logits(&self, pool: &Var<R, E>) -> Result<Var<R, E>> {
        let shape = pool.dims().to_vec();
        if shape.len() != 2 || shape[1] != self.d_model {
            return Err(Error::shape(format!(
                "FingerprintHead::logits needs pool [B, {}], got {shape:?}",
                self.d_model
            )));
        }
        let hidden = self.fc1.apply(pool)?.silu()?;
        self.fc2.apply(&hidden)
    }

    /// Weighted mean over spectra of the per-spectrum mean over bits of the
    /// shared stable binary cross-entropy with logits
    /// ([`bce_with_logits`](super::rerank::bce_with_logits):
    /// `softplus(x) − x·z`, whose derivative is `sigmoid(x) − z` everywhere
    /// including zero): `sum_b w_b * l_b / max(sum_b w_b, eps)` with
    /// `eps = `[`FINGERPRINT_WEIGHT_EPS`] (tiny, so fractional weight sums
    /// below one keep their full weight) and exactly 0 when the weight sum is
    /// 0 (the numerator is 0 then). Computed on the device with no read.
    ///
    /// `logits` is `[B, 1024]`, `targets` is `[B, 1024]` floats (`0`/`1` by
    /// contract; values are not re-checked here) and `weights` is `[B]`.
    /// Whole-parent auxiliary supervision: `targets` rows are whole-parent
    /// fingerprints shared by every spectrum of a molecule, never targets
    /// for individual fragments or candidates.
    pub fn loss(
        &self,
        logits: &Var<R, E>,
        targets: &Tensor<R, E>,
        weights: &Tensor<R, E>,
    ) -> Result<Var<R, E>> {
        let shape = logits.dims().to_vec();
        if shape.len() != 2 || shape[1] != FINGERPRINT_BITS {
            return Err(Error::shape(format!(
                "FingerprintHead::loss needs logits [B, {FINGERPRINT_BITS}], got {shape:?}"
            )));
        }
        let batch = shape[0];
        if targets.rank() != 2 || targets.dims().to_vec() != vec![batch, FINGERPRINT_BITS] {
            return Err(Error::shape(format!(
                "FingerprintHead::loss needs targets [{batch}, {FINGERPRINT_BITS}], got {:?}",
                targets.shape()
            )));
        }
        if weights.rank() != 1 || weights.len() != batch {
            return Err(Error::shape(format!(
                "FingerprintHead::loss needs weights [{batch}], got {:?}",
                weights.shape()
            )));
        }
        if batch == 0 {
            return Ok(Var::constant(Tensor::full(
                Vec::<usize>::new(),
                0.0,
                logits.device(),
            )));
        }
        let z = Var::constant(targets.clone());
        let w = Var::constant(weights.clone());
        let per = bce_with_logits(logits, &z)?;
        let spread = w.unsqueeze(1)?.expand(vec![batch, FINGERPRINT_BITS])?;
        let num = per.mul(&spread)?.sum()?;
        let den = w.sum()?;
        let eps = Var::constant(Tensor::full(
            Vec::<usize>::new(),
            FINGERPRINT_WEIGHT_EPS,
            logits.device(),
        ));
        let denom = den.maximum(&eps)?.mul_scalar(FINGERPRINT_BITS as f32);
        num.div(&denom)
    }
}

impl<R: Runtime, E: FloatElem> Module<R, E> for FingerprintHead<R, E> {
    fn visit(&self, visitor: &mut ModuleVisitor<'_, R, E>) {
        visitor.child("fc1", &self.fc1);
        visitor.child("fc2", &self.fc2);
    }
}

/// Sigmoid in `f64`, the probability map of the reporting metrics.
fn sigmoid_f64(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Check host slices hold exactly one `[batch, 1024]` row-major pair.
fn check_metric_shapes(logits: &[f32], targets: &[f32], batch: usize, what: &str) {
    assert_eq!(
        logits.len(),
        batch * FINGERPRINT_BITS,
        "{what}: logits hold {} values, want {}",
        logits.len(),
        batch * FINGERPRINT_BITS
    );
    assert_eq!(
        targets.len(),
        batch * FINGERPRINT_BITS,
        "{what}: targets hold {} values, want {}",
        targets.len(),
        batch * FINGERPRINT_BITS
    );
}

/// Mean bit accuracy over all `batch * 1024` bits, from read-back logits, for
/// reporting only: a bit counts as correct when `(logit >= 0)` agrees with
/// `(target >= 0.5)`. 1.0 for an empty batch.
pub fn bit_accuracy(logits: &[f32], targets: &[f32], batch: usize) -> f64 {
    check_metric_shapes(logits, targets, batch, "bit_accuracy");
    if batch == 0 {
        return 1.0;
    }
    let mut correct = 0u64;
    for (x, t) in logits.iter().zip(targets.iter()) {
        if (*x >= 0.0) == (*t >= 0.5) {
            correct += 1;
        }
    }
    correct as f64 / (batch * FINGERPRINT_BITS) as f64
}

/// Mean over spectra of the Tanimoto of the thresholded prediction
/// (`logit >= 0`) against the label (`target >= 0.5`), from read-back logits,
/// for reporting only: `|P ∩ L| / |P ∪ L|` per spectrum, defined as 1 when
/// both the prediction and the label are empty (union 0). 1.0 for an empty
/// batch.
pub fn tanimoto(logits: &[f32], targets: &[f32], batch: usize) -> f64 {
    check_metric_shapes(logits, targets, batch, "tanimoto");
    if batch == 0 {
        return 1.0;
    }
    let mut sum = 0.0;
    for b in 0..batch {
        let base = b * FINGERPRINT_BITS;
        let mut inter = 0u32;
        let mut union = 0u32;
        for i in 0..FINGERPRINT_BITS {
            let p = logits[base + i] >= 0.0;
            let l = targets[base + i] >= 0.5;
            if p && l {
                inter += 1;
            }
            if p || l {
                union += 1;
            }
        }
        sum += if union == 0 {
            1.0
        } else {
            inter as f64 / union as f64
        };
    }
    sum / batch as f64
}

/// Per-spectrum cosine of the probabilities (`sigmoid` of the read-back
/// logits) against the labels, for reporting only: `p·t / (|p| |t|)` per
/// spectrum, 1.0 when both are all-zero, 0.0 when exactly one is.
pub fn cosine_per_spectrum(logits: &[f32], targets: &[f32], batch: usize) -> Vec<f64> {
    check_metric_shapes(logits, targets, batch, "cosine_per_spectrum");
    let mut out = Vec::with_capacity(batch);
    for b in 0..batch {
        let base = b * FINGERPRINT_BITS;
        let mut dot = 0.0;
        let mut pp = 0.0;
        let mut tt = 0.0;
        for i in 0..FINGERPRINT_BITS {
            let p = sigmoid_f64(f64::from(logits[base + i]));
            let t = f64::from(targets[base + i]);
            dot += p * t;
            pp += p * p;
            tt += t * t;
        }
        out.push(if pp == 0.0 || tt == 0.0 {
            if pp == 0.0 && tt == 0.0 { 1.0 } else { 0.0 }
        } else {
            dot / (pp.sqrt() * tt.sqrt())
        });
    }
    out
}
