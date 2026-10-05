//! Memoised device enumeration for the enumerating formula source (task T6).
//!
//! The enumeration result of a spectrum depends only on its precursor data
//! and the frozen artifacts — not on any model parameter — yet the uncached
//! path recomputes it for every spectrum at every step of every epoch and at
//! every evaluation. [`EnumCache`] is an exact memo of that device
//! enumeration: the key is the spectrum's 8-word enumeration meta row exactly
//! as [`build_enum_meta`](super::formula_enum::build_enum_meta) produces it
//! (so a different precursor, tolerance, uncertainty, adduct, lane budget or
//! scored cap is a different key), and the value is that spectrum's
//! `counters` row (5 words) plus its scored candidates (rows `0 ..
//! rows_scored` of `cand`; every slot at or after `rows_scored` is the fixed
//! padding row and is regenerated, never stored).
//!
//! Compact storage: per candidate the 9 heavy counts as `u8`, hydrogen as
//! `u16` and the flag as `u8` (artifact validation guarantees caps `<= 255`
//! and hydrogen `<= 1023`; [`EnumCache::insert`] asserts it with
//! [`Error::Config`](crate::error::Error::Config) otherwise). The integer
//! mass and the source id (`u32::MAX`) are recomputed when a row is expanded
//! (the mass through [`composition_mass`](super::chem::composition_mass), the
//! same checked function the kernels' host twin uses; `insert` tests equality
//! with the device value for every stored candidate).
//!
//! The cache is exact or absent: no approximate matching, no reuse across
//! headers. The header is checked on every use (build, load, expand);
//! [`Error::Config`](crate::error::Error::Config) names the field on
//! mismatch.
//!
//! Pure host Rust: no tensors, no kernels, no device code.

use std::collections::HashMap;
use std::path::Path;

use cubecl::prelude::Runtime;
use serde::{Deserialize, Serialize};

use crate::backend::{Device, FloatElem};
use crate::error::{Error, Result};
use crate::tensor::ops::index::{IdTensor, read_all};

use super::chem::{CHEMISTRY_VERSION, composition_mass};
use super::contract::SpectrumBatch;
use super::formula_enum::build_enum_meta;
use super::formula_head::DeviceEnumArtifacts;
use crate::tensor::ops::ms2_enum::{EnumLaunch, cand_pad, enum_offsets};

/// Magic bytes of the [`EnumCache`] binary file (`save` / `load`).
const CACHE_MAGIC: [u8; 8] = *b"MS2ENUMC";

/// Binary format version of the [`EnumCache`] file.
pub const ENUM_CACHE_FORMAT_VERSION: u32 = 1;

/// Chemistry version this cache was built against (today's domain).
const CACHE_CHEMISTRY_VERSION: &str = CHEMISTRY_VERSION;

/// Words per candidate record of `cand [B, M, 13]`.
const CAND_WORDS: usize = 13;

/// `counters [B, 5]` words per spectrum: visited, joined, scored, status,
/// complete.
const COUNTER_WORDS: usize = 5;

/// Compact bytes per stored candidate: 9 heavy counts (`u8`), hydrogen
/// (`u16`, little-endian) and the flag (`u8`).
const COMPACT_CAND_BYTES: usize = 12;

/// Header of an [`EnumCache`]: everything the cached enumeration depends on.
///
/// Two runs share cache entries only when every field agrees: the artifacts'
/// SHA-256 (domain and bounds), the rare-table depth `P`, the window `M`,
/// `formula_rows_scored_max`, `enum_lane_visits_max` and the chemistry
/// version. The format version guards the binary layout itself.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EnumCacheHeader {
    /// Binary format version ([`ENUM_CACHE_FORMAT_VERSION`]).
    pub format_version: u32,
    /// SHA-256 of the enum domain JSON the entries were enumerated with.
    pub domain_sha256: String,
    /// SHA-256 of the ratio bounds JSON the entries were enumerated with.
    pub bounds_sha256: String,
    /// Rare-table depth `P` of the enumerating source.
    pub p: u32,
    /// Scored-candidate window `M` the entries were enumerated with.
    pub window_m: u32,
    /// Scored-cap argument the entries were enumerated with
    /// (`min(formula_rows_scored_max, M)` lands in the meta row).
    pub formula_rows_scored_max: u32,
    /// Per-lane visit budget the entries were enumerated with (lands in the
    /// meta row).
    pub enum_lane_visits_max: u32,
    /// Chemistry version ([`CHEMISTRY_VERSION`]).
    pub chemistry_version: String,
}

impl EnumCacheHeader {
    /// Build the header callers stamp on a fresh cache: today's format and
    /// chemistry versions with the given enumeration parameters.
    pub fn new(
        domain_sha256: String,
        bounds_sha256: String,
        p: u32,
        window_m: u32,
        formula_rows_scored_max: u32,
        enum_lane_visits_max: u32,
    ) -> Self {
        Self {
            format_version: ENUM_CACHE_FORMAT_VERSION,
            domain_sha256,
            bounds_sha256,
            p,
            window_m,
            formula_rows_scored_max,
            enum_lane_visits_max,
            chemistry_version: CACHE_CHEMISTRY_VERSION.to_string(),
        }
    }

    /// Check `other` against `self` field by field: the first mismatch is
    /// [`Error::Config`](crate::error::Error::Config) naming the field, so a
    /// stale cache is never silently reused across headers.
    pub fn check_compatible(&self, other: &EnumCacheHeader) -> Result<()> {
        if self.format_version != other.format_version {
            return Err(Error::config(format!(
                "EnumCache header mismatch on field `format_version`: cache {} != expected {}",
                self.format_version, other.format_version
            )));
        }
        if self.domain_sha256 != other.domain_sha256 {
            return Err(Error::config(format!(
                "EnumCache header mismatch on field `domain_sha256`: cache {} != expected {}",
                self.domain_sha256, other.domain_sha256
            )));
        }
        if self.bounds_sha256 != other.bounds_sha256 {
            return Err(Error::config(format!(
                "EnumCache header mismatch on field `bounds_sha256`: cache {} != expected {}",
                self.bounds_sha256, other.bounds_sha256
            )));
        }
        if self.p != other.p {
            return Err(Error::config(format!(
                "EnumCache header mismatch on field `p`: cache {} != expected {}",
                self.p, other.p
            )));
        }
        if self.window_m != other.window_m {
            return Err(Error::config(format!(
                "EnumCache header mismatch on field `window_m`: cache {} != expected {}",
                self.window_m, other.window_m
            )));
        }
        if self.formula_rows_scored_max != other.formula_rows_scored_max {
            return Err(Error::config(format!(
                "EnumCache header mismatch on field `formula_rows_scored_max`: cache {} != expected {}",
                self.formula_rows_scored_max, other.formula_rows_scored_max
            )));
        }
        if self.enum_lane_visits_max != other.enum_lane_visits_max {
            return Err(Error::config(format!(
                "EnumCache header mismatch on field `enum_lane_visits_max`: cache {} != expected {}",
                self.enum_lane_visits_max, other.enum_lane_visits_max
            )));
        }
        if self.chemistry_version != other.chemistry_version {
            return Err(Error::config(format!(
                "EnumCache header mismatch on field `chemistry_version`: cache {:?} != expected {:?}",
                self.chemistry_version, other.chemistry_version
            )));
        }
        Ok(())
    }
}

/// One compact scored candidate: the 9 heavy counts in `ELEMENTS` order
/// without hydrogen (C, N, O, F, P, S, Cl, Br, I), the hydrogen count and the
/// verdict flag (`1` accept, `2` ambiguous).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CompactCand {
    /// Heavy counts in `ELEMENTS` order without hydrogen.
    heavy: [u8; 9],
    /// Hydrogen count.
    h: u16,
    /// Verdict flag (`1` accept, `2` ambiguous).
    flag: u8,
}

impl CompactCand {
    /// Pack one device `cand` record (13 words) into the compact form,
    /// checking the recomputed mass against the device value.
    ///
    /// [`Error::Config`](crate::error::Error::Config) on any out-of-range
    /// count, on a hydrogen count above 1023, on a flag other than 1 or 2, on
    /// a source id other than `u32::MAX`, or when the recomputed mass differs
    /// from the device word.
    fn pack(record: &[u32]) -> Result<Self> {
        if record.len() != CAND_WORDS {
            return Err(Error::config(format!(
                "EnumCache::insert: candidate record has {} words, needs {CAND_WORDS}",
                record.len()
            )));
        }
        let mut heavy = [0u8; 9];
        // ELEMENTS order is C, H, N, O, F, P, S, Cl, Br, I: the heavy slots
        // are record words 0, 2, 3, 4, 5, 6, 7, 8, 9.
        const HEAVY_WORDS: [usize; 9] = [0, 2, 3, 4, 5, 6, 7, 8, 9];
        for (i, &w) in HEAVY_WORDS.iter().enumerate() {
            if record[w] > 255 {
                return Err(Error::config(format!(
                    "EnumCache::insert: heavy count {i} {} exceeds 255",
                    record[w]
                )));
            }
            heavy[i] = record[w] as u8;
        }
        if record[1] > 1023 {
            return Err(Error::config(format!(
                "EnumCache::insert: hydrogen count {} exceeds 1023",
                record[1]
            )));
        }
        let h = record[1] as u16;
        if record[11] != 1 && record[11] != 2 {
            return Err(Error::config(format!(
                "EnumCache::insert: flag {} is not 1 (accept) or 2 (ambiguous)",
                record[11]
            )));
        }
        if record[12] != u32::MAX {
            return Err(Error::config(format!(
                "EnumCache::insert: source id {} is not u32::MAX",
                record[12]
            )));
        }
        // Recompute the mass through the same checked composition-mass
        // function the kernels' host twin uses, and test equality with the
        // device value.
        let mut comp = [0u16; 10];
        comp[0] = u16::from(heavy[0]);
        comp[1] = h;
        comp[2] = u16::from(heavy[1]);
        comp[3] = u16::from(heavy[2]);
        comp[4] = u16::from(heavy[3]);
        comp[5] = u16::from(heavy[4]);
        comp[6] = u16::from(heavy[5]);
        comp[7] = u16::from(heavy[6]);
        comp[8] = u16::from(heavy[7]);
        comp[9] = u16::from(heavy[8]);
        let mass = composition_mass(&comp).map_err(|e| {
            Error::config(format!("EnumCache::insert: recomputed mass overflows u32: {e}"))
        })?;
        if mass != record[10] {
            return Err(Error::config(format!(
                "EnumCache::insert: recomputed mass {mass} != device mass {}",
                record[10]
            )));
        }
        Ok(Self {
            heavy,
            h,
            flag: record[11] as u8,
        })
    }

    /// Expand back to one 13-word device `cand` record (mass recomputed,
    /// source id `u32::MAX`).
    fn expand(&self) -> [u32; CAND_WORDS] {
        let mut comp = [0u16; 10];
        comp[0] = u16::from(self.heavy[0]);
        comp[1] = self.h;
        comp[2] = u16::from(self.heavy[1]);
        comp[3] = u16::from(self.heavy[2]);
        comp[4] = u16::from(self.heavy[3]);
        comp[5] = u16::from(self.heavy[4]);
        comp[6] = u16::from(self.heavy[5]);
        comp[7] = u16::from(self.heavy[6]);
        comp[8] = u16::from(self.heavy[7]);
        comp[9] = u16::from(self.heavy[8]);
        // Packed by `pack`, so the mass fits `u32` (ELEMENTS masses are
        // fixed); a corrupt in-memory entry surfaces here rather than
        // wrapping.
        let mass = composition_mass(&comp).unwrap_or(0);
        let mut out = [0u32; CAND_WORDS];
        out[0] = u32::from(self.heavy[0]);
        out[1] = u32::from(self.h);
        out[2] = u32::from(self.heavy[1]);
        out[3] = u32::from(self.heavy[2]);
        out[4] = u32::from(self.heavy[3]);
        out[5] = u32::from(self.heavy[4]);
        out[6] = u32::from(self.heavy[5]);
        out[7] = u32::from(self.heavy[6]);
        out[8] = u32::from(self.heavy[7]);
        out[9] = u32::from(self.heavy[8]);
        out[10] = mass;
        out[11] = u32::from(self.flag);
        out[12] = u32::MAX;
        out
    }
}

/// One cached spectrum: its `counters` row and its scored candidates in
/// compact form.
#[derive(Clone, Debug, PartialEq, Eq)]
struct CacheEntry {
    /// `counters` row: visited, joined, scored, status, complete.
    counters: [u32; COUNTER_WORDS],
    /// Scored candidates (`rows_scored` of them).
    cands: Vec<CompactCand>,
}

/// A borrowed view of one cached spectrum's value.
#[derive(Clone, Copy, Debug)]
pub struct EntryRef<'a> {
    /// `counters` row: visited, joined, scored, status, complete.
    pub counters: &'a [u32; COUNTER_WORDS],
    /// Scored candidates in compact form (`rows_scored` of them).
    pub cands: &'a [CompactCand],
}

/// Exact memo of the device enumeration for the enumerating formula source.
///
/// See the module docs. The map key is the spectrum's 8-word enumeration
/// meta row exactly as `build_enum_meta` produces it; the value is that
/// spectrum's `counters` row and its scored candidates. A partially cached
/// batch is never mixed: [`EnumCache::expand_batch`] returns `None` when ANY
/// key misses, and callers then run the device enumeration exactly as
/// without a cache.
#[derive(Clone, Debug)]
pub struct EnumCache {
    /// What the entries were enumerated with; checked on every use.
    header: EnumCacheHeader,
    /// Meta row → (`counters` row, scored candidates).
    map: HashMap<[u32; 8], CacheEntry>,
}

impl EnumCache {
    /// An empty cache for `header`.
    pub fn new(header: EnumCacheHeader) -> Self {
        Self {
            header,
            map: HashMap::new(),
        }
    }

    /// The header the entries were enumerated with.
    pub fn header(&self) -> &EnumCacheHeader {
        &self.header
    }

    /// Look up one spectrum's meta row.
    pub fn get(&self, key: &[u32; 8]) -> Option<EntryRef<'_>> {
        self.map.get(key).map(|e| EntryRef {
            counters: &e.counters,
            cands: &e.cands,
        })
    }

    /// Insert one spectrum: its meta-row `key`, its `counters` row and its
    /// scored candidate rows (`rows 0 .. rows_scored` of `cand`, each 13
    /// words, where `rows_scored = counters[2]`).
    ///
    /// Every stored candidate is range-checked and its recomputed mass is
    /// tested against the device value ([`CompactCand::pack`]);
    /// [`Error::Config`](crate::error::Error::Config) otherwise. Re-inserting
    /// a present key overwrites it. `m` must equal the header's `window_m`
    /// (the width the caller read `cand` at), else `Error::Config`: entries
    /// read at another width do not belong in this cache.
    pub fn insert(
        &mut self,
        key: [u32; 8],
        counters: [u32; COUNTER_WORDS],
        cand_rows: &[u32],
        m: usize,
    ) -> Result<()> {
        if m as u64 != u64::from(self.header.window_m) {
            return Err(Error::config(format!(
                "EnumCache::insert: cand width {m} != header window_m {}",
                self.header.window_m
            )));
        }
        let scored = counters[2] as usize;
        if cand_rows.len() != scored * CAND_WORDS {
            return Err(Error::config(format!(
                "EnumCache::insert: {} cand words != rows_scored {scored} * {CAND_WORDS}",
                cand_rows.len()
            )));
        }
        if scored > m {
            return Err(Error::config(format!(
                "EnumCache::insert: rows_scored {scored} exceeds cand width {m}"
            )));
        }
        let mut cands = Vec::with_capacity(scored);
        for r in 0..scored {
            cands.push(CompactCand::pack(
                &cand_rows[r * CAND_WORDS..(r + 1) * CAND_WORDS],
            )?);
        }
        self.map.insert(key, CacheEntry { counters, cands });
        Ok(())
    }

    /// Spectra cached.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Whether no spectrum is cached.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Exact byte size of the [`EnumCache::save`] file for these entries.
    pub fn bytes(&self) -> usize {
        let header_json = self.header_json();
        let mut n = CACHE_MAGIC.len() + 4 + 4 + header_json.len() + 8;
        for e in self.map.values() {
            n += 8 * 4 + COUNTER_WORDS * 4 + 4 + e.cands.len() * COMPACT_CAND_BYTES;
        }
        n
    }

    /// The header as JSON (the form `save` stores).
    fn header_json(&self) -> String {
        serde_json::to_string(&self.header).unwrap_or_else(|_| String::from("{}"))
    }

    /// Expand a whole batch: `cand [B, M, 13]` (scored rows plus the fixed
    /// padding rows) and `counters [B, 5]`, flat and row-major.
    ///
    /// Returns `None` when ANY key misses, when `m` differs from the
    /// header's `window_m`, or when a stored `rows_scored` exceeds `m`: a
    /// partially cached batch takes the device path (no mixing). Otherwise
    /// the returned buffers are bit-identical to what the device enumeration
    /// produced, including padding slots and status bits.
    pub fn expand_batch(
        &self,
        keys: &[[u32; 8]],
        m: usize,
    ) -> Option<(Vec<u32> /* cand [B, M, 13] */, Vec<u32> /* counters [B, 5] */)> {
        if m as u64 != u64::from(self.header.window_m) {
            return None;
        }
        let b = keys.len();
        let mut cand = vec![0u32; b * m * CAND_WORDS];
        let mut counters = vec![0u32; b * COUNTER_WORDS];
        for (i, key) in keys.iter().enumerate() {
            let entry = self.map.get(key)?;
            let scored = entry.counters[2] as usize;
            if scored > m || entry.cands.len() != scored {
                return None;
            }
            counters[i * COUNTER_WORDS..(i + 1) * COUNTER_WORDS]
                .copy_from_slice(&entry.counters);
            for (r, c) in entry.cands.iter().enumerate() {
                let words = c.expand();
                cand[(i * m + r) * CAND_WORDS..(i * m + r + 1) * CAND_WORDS]
                    .copy_from_slice(&words);
            }
            // Slots at or after `rows_scored` are the fixed padding row
            // (all `0` except source `u32::MAX`); `cand` starts zeroed, so
            // only the source words need writing.
            for r in scored..m {
                cand[(i * m + r) * CAND_WORDS + CAND_WORDS - 1] = u32::MAX;
            }
        }
        Some((cand, counters))
    }

    /// Save to `path`: magic, version, header JSON, then entries,
    /// little-endian. Written atomically via a temp file and rename, so a
    /// crash never leaves a half-written cache behind.
    pub fn save(&self, path: &Path) -> Result<()> {
        let header_json = self.header_json();
        let header_bytes = header_json.as_bytes();
        let mut buf = Vec::with_capacity(self.bytes());
        buf.extend_from_slice(&CACHE_MAGIC);
        buf.extend_from_slice(&ENUM_CACHE_FORMAT_VERSION.to_le_bytes());
        buf.extend_from_slice(&(header_bytes.len() as u32).to_le_bytes());
        buf.extend_from_slice(header_bytes);
        buf.extend_from_slice(&(self.map.len() as u64).to_le_bytes());
        // Sorted keys, so the file is deterministic run to run.
        let mut keys: Vec<&[u32; 8]> = self.map.keys().collect();
        keys.sort();
        for key in keys {
            let entry = &self.map[key];
            for w in key.iter() {
                buf.extend_from_slice(&w.to_le_bytes());
            }
            for w in entry.counters.iter() {
                buf.extend_from_slice(&w.to_le_bytes());
            }
            buf.extend_from_slice(&(entry.cands.len() as u32).to_le_bytes());
            for c in &entry.cands {
                buf.extend_from_slice(&c.heavy);
                buf.extend_from_slice(&c.h.to_le_bytes());
                buf.push(c.flag);
            }
        }
        debug_assert_eq!(buf.len(), self.bytes());
        let mut tmp = path.as_os_str().to_owned();
        tmp.push(format!(".tmp-{}", std::process::id()));
        let tmp_path = Path::new(&tmp);
        std::fs::write(tmp_path, &buf).map_err(|e| {
            Error::config(format!(
                "EnumCache::save: cannot write {}: {e}",
                tmp_path.display()
            ))
        })?;
        std::fs::rename(tmp_path, path).map_err(|e| {
            Error::config(format!(
                "EnumCache::save: cannot rename {} to {}: {e}",
                tmp_path.display(),
                path.display()
            ))
        })?;
        Ok(())
    }

    /// Load from `path`, checking the stored header against `expected`
    /// field by field ([`Error::Config`](crate::error::Error::Config)
    /// naming the field on mismatch — never silently rebuild over a
    /// mismatching file).
    ///
    /// A truncated or corrupted file is an error, never a panic.
    pub fn load(path: &Path, expected: &EnumCacheHeader) -> Result<Self> {
        let bytes = std::fs::read(path).map_err(|e| {
            Error::config(format!(
                "EnumCache::load: cannot read {}: {e}",
                path.display()
            ))
        })?;
        let cache = Self::parse(&bytes, path)?;
        // Check the stored header against the expected one before trusting
        // any entry (`check_compatible` names the field on mismatch).
        cache.header.check_compatible(expected)?;
        Ok(cache)
    }

    /// Parse the binary form, validating every length before slicing.
    fn parse(bytes: &[u8], path: &Path) -> Result<Self> {
        let err = |what: &str| {
            Error::config(format!(
                "EnumCache::load: {} is truncated or corrupted ({what})",
                path.display()
            ))
        };
        // Advance `cursor` by `n`, handing back the slice. Every length is
        // validated before slicing, so corrupt lengths are errors, never
        // panics.
        let mut cursor = 0usize;
        let mut take = |n: usize| -> Result<&[u8]> {
            let end = cursor.checked_add(n).ok_or_else(|| err("length overflow"))?;
            if end > bytes.len() {
                return Err(err("unexpected end"));
            }
            let out = &bytes[cursor..end];
            cursor = end;
            Ok(out)
        };
        let magic = take(CACHE_MAGIC.len())?;
        if magic != CACHE_MAGIC {
            return Err(Error::config(format!(
                "EnumCache::load: {} has bad magic (not an enum cache)",
                path.display()
            )));
        }
        let version_bytes = take(4)?;
        let version = u32::from_le_bytes(version_bytes.try_into().map_err(|_| err("version"))?);
        if version != ENUM_CACHE_FORMAT_VERSION {
            return Err(Error::config(format!(
                "EnumCache::load: {} has format version {version}, needs {}",
                path.display(),
                ENUM_CACHE_FORMAT_VERSION
            )));
        }
        let hlen_bytes = take(4)?;
        let hlen =
            u32::from_le_bytes(hlen_bytes.try_into().map_err(|_| err("header length"))?) as usize;
        let hbytes = take(hlen)?;
        let header_text =
            std::str::from_utf8(hbytes).map_err(|_| err("header is not UTF-8"))?;
        let header: EnumCacheHeader =
            serde_json::from_str(header_text).map_err(|_| err("header JSON does not parse"))?;
        let count_bytes = take(8)?;
        let count =
            u64::from_le_bytes(count_bytes.try_into().map_err(|_| err("entry count"))?) as usize;
        if count > 10_000_000 {
            return Err(err("entry count is implausible"));
        }
        let mut map = HashMap::with_capacity(count.min(1_000_000));
        for _ in 0..count {
            let mut key = [0u32; 8];
            for w in key.iter_mut() {
                let wb = take(4)?;
                *w = u32::from_le_bytes(wb.try_into().map_err(|_| err("key"))?);
            }
            let mut counters = [0u32; COUNTER_WORDS];
            for w in counters.iter_mut() {
                let wb = take(4)?;
                *w = u32::from_le_bytes(wb.try_into().map_err(|_| err("counters"))?);
            }
            let nb = take(4)?;
            let n = u32::from_le_bytes(nb.try_into().map_err(|_| err("candidate count"))?) as usize;
            if n > 100_000 {
                return Err(err("candidate count is implausible"));
            }
            if n as u64 != u64::from(counters[2]) {
                return Err(Error::config(format!(
                    "EnumCache::load: {} stores {n} candidates but rows_scored is {}",
                    path.display(),
                    counters[2]
                )));
            }
            let mut cands = Vec::with_capacity(n);
            for _ in 0..n {
                let hb = take(9)?;
                let mut heavy = [0u8; 9];
                heavy.copy_from_slice(hb);
                let hhb = take(2)?;
                let h = u16::from_le_bytes(hhb.try_into().map_err(|_| err("hydrogen"))?);
                let fb = take(1)?;
                let flag = fb[0];
                if h > 1023 {
                    return Err(Error::config(format!(
                        "EnumCache::load: {} stores hydrogen {h} above 1023",
                        path.display()
                    )));
                }
                if flag != 1 && flag != 2 {
                    return Err(Error::config(format!(
                        "EnumCache::load: {} stores flag {flag} (needs 1 or 2)",
                        path.display()
                    )));
                }
                // The stored mass is recomputed at expand time through the
                // checked composition mass; validate the counts here so a
                // corrupt file cannot smuggle an unrepresentable composition
                // past the load.
                let mut comp = [0u16; 10];
                comp[0] = u16::from(heavy[0]);
                comp[1] = h;
                comp[2] = u16::from(heavy[1]);
                comp[3] = u16::from(heavy[2]);
                comp[4] = u16::from(heavy[3]);
                comp[5] = u16::from(heavy[4]);
                comp[6] = u16::from(heavy[5]);
                comp[7] = u16::from(heavy[6]);
                comp[8] = u16::from(heavy[7]);
                comp[9] = u16::from(heavy[8]);
                composition_mass(&comp).map_err(|_| err("candidate mass overflows u32"))?;
                cands.push(CompactCand { heavy, h, flag });
            }
            if map.insert(key, CacheEntry { counters, cands }).is_some() {
                return Err(Error::config(format!(
                    "EnumCache::load: {} stores a duplicate key",
                    path.display()
                )));
            }
        }
        if cursor != bytes.len() {
            return Err(Error::config(format!(
                "EnumCache::load: {} has {} trailing bytes",
                path.display(),
                bytes.len() - cursor
            )));
        }
        Ok(Self { header, map })
    }
}

/// Run the EXISTING device enumeration (count → offsets → fill → pad) for
/// one batch and insert every spectrum not yet in `cache`.
///
/// This is the production enumeration, not a copy: the same
/// [`build_enum_meta`], [`EnumLaunch::count`] / [`EnumLaunch::fill`],
/// [`enum_offsets`] and [`cand_pad`] the search stage calls, over caller
/// buffers of the same shapes (`meta [B, 8]`, `lane_stats [B * P, 2]`,
/// `offsets [B * P]`, `cand [B, M, 13]`, `counters [B, 5]`). `cand` and
/// `counters` come back in one batched read (this is a precompute pass;
/// reads are expected). Batches whose spectra are all cached already are
/// skipped without any launch or read. `m` (`M`) must equal the cache
/// header's `window_m`.
#[allow(clippy::too_many_arguments)]
pub fn run_device_enumeration_into<R: Runtime, E: FloatElem>(
    device: &Device<R>,
    artifacts: &DeviceEnumArtifacts<R>,
    batch: &SpectrumBatch,
    scored_cap: u32,
    lanes_max: u32,
    dispatch_visits_max: u32,
    lane_visits_max: u32,
    m: usize,
    cache: &mut EnumCache,
) -> Result<()> {
    let b = batch.len();
    if b == 0 {
        return Ok(());
    }
    if m as u64 != u64::from(cache.header().window_m) {
        return Err(Error::config(format!(
            "run_device_enumeration_into: cand width {m} != cache header window_m {}",
            cache.header().window_m
        )));
    }
    let meta_host = build_enum_meta(
        batch,
        artifacts.domain_max_error,
        lane_visits_max,
        scored_cap,
    );
    let keys: Vec<[u32; 8]> = meta_host
        .chunks_exact(8)
        .map(|r| {
            let mut k = [0u32; 8];
            k.copy_from_slice(r);
            k
        })
        .collect();
    if keys.iter().all(|k| cache.get(k).is_some()) {
        return Ok(());
    }
    let p = artifacts.p;
    let meta_t = IdTensor::from_slice(&meta_host, vec![b, 8], device)?;
    let lane_stats = IdTensor::empty(vec![b * p, 2], device);
    let offsets = IdTensor::empty(vec![b * p], device);
    let cand_t = IdTensor::empty(vec![b, m, CAND_WORDS], device);
    let counters_t = IdTensor::empty(vec![b, COUNTER_WORDS], device);
    let launch = EnumLaunch::from_chemistry();
    launch.count(
        &meta_t,
        &artifacts.rare,
        &artifacts.bounds,
        &lane_stats,
        lanes_max,
        dispatch_visits_max,
        lane_visits_max,
    )?;
    enum_offsets(
        &lane_stats,
        &meta_t,
        &offsets,
        &counters_t,
        scored_cap,
        m,
        lanes_max,
    )?;
    launch.fill(
        &meta_t,
        &artifacts.rare,
        &artifacts.bounds,
        &offsets,
        &cand_t,
        scored_cap,
        lanes_max,
        dispatch_visits_max,
        lane_visits_max,
    )?;
    cand_pad(&counters_t, &cand_t, p, lanes_max)?;
    // One batched read per batch.
    let (ids, _) = read_all::<R, E>(&[&cand_t, &counters_t], &[])?;
    let (cand_host, counters_host) = (&ids[0], &ids[1]);
    for (i, key) in keys.iter().enumerate() {
        if cache.get(key).is_some() {
            continue;
        }
        let mut counters = [0u32; COUNTER_WORDS];
        counters.copy_from_slice(
            &counters_host[i * COUNTER_WORDS..(i + 1) * COUNTER_WORDS],
        );
        let scored = counters[2] as usize;
        let base = i * m * CAND_WORDS;
        cache.insert(
            *key,
            counters,
            &cand_host[base..base + scored * CAND_WORDS],
            m,
        )?;
    }
    Ok(())
}
