//! Memoised device enumeration for the enumerating formula source (task T6)
//! with the formula-evidence memo (task T6B).
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
//! Task T6B adds an exact memo of the formula-evidence stage (`cand_ev [B, M,
//! 4]`) per (spectrum, candidate set), in a sibling map of the same cache
//! object and file. Like the enumeration, the evidence of a candidate depends
//! only on the spectrum (its uploaded peaks, fragment tolerance, m/z
//! uncertainty, adduct) and the candidate — on nothing the model learns. The
//! key is the spectrum's 8-word enumeration meta row (which fixes the
//! candidate rows) PLUS a 128-bit content hash of everything else
//! `evidence_peaks` and `formula_evidence` read for that spectrum (see
//! [`evidence_key_for_batch`] for the canonical encoding). The value is, per
//! scored slot, the explained count (`u8`), the explained weight (`f32`
//! bits, unchanged) and one bit for complete, plus per spectrum the number of
//! valid evidence peaks (`u8`); padding slots regenerate as zeros. Each entry
//! additionally stores the canonical evidence inputs it was built from (the
//! uploaded peak row words and every scalar that enters the key, exactly the
//! bytes [`evidence_key_for_batch`] hashes) alongside the 128-bit hash: on a
//! hash hit the stored bytes are compared in full and a mismatch is a MISS
//! (counted in [`EnumCache::collisions`]), never a served entry. Colliding
//! keys coexist in per-key buckets, so the memo is exact, not probabilistic.
//! Memory cost per evidence entry (after task F8 item 3; before, the entry
//! held no canonical bytes): the 48-byte key, `36 + 8 * slots` canonical
//! bytes, `rows` explained bytes, `4 * rows` weight bytes, `rows` complete
//! bytes, plus the [`EnumCache::resident_bytes`] overhead estimate.
//!
//! Evidence identity (task F8 items 1–2): the header carries the model's
//! kept-peak capacity `n_peaks` (the evidence stage sees only the kept
//! peaks, so a cache built at `n_peaks = 16` is refused for a model with
//! `n_peaks = 32`, naming `n_peaks`) and the floating element dtype the
//! evidence was computed in (`evidence_dtype`, e.g. `"f32"`). Enumeration
//! entries are pure integers and stay dtype-independent: a cache built under
//! `f32` still serves ENUMERATION hits under `bf16`, while the evidence
//! section always MISSES there (the step runs the device evidence stage, so
//! exactness is preserved). The dtype is enforced when evidence is built
//! ([`run_device_evidence_into`] refuses a mismatching header) and on every
//! evidence lookup ([`EnumCache::expand_evidence_batch`] misses); it is NOT
//! part of the attach/load compatibility check, so enumeration reuse across
//! dtypes keeps working.
//!
//! Host memory budget (task F8 item 4): [`EnumCache::resident_bytes`] is the
//! estimated resident host footprint (vector capacities plus map overhead,
//! see its formula — an estimate, not an allocator readout), separate from
//! [`EnumCache::bytes`] (the serialized file size). Inserts that would push
//! the resident estimate past `max_resident_bytes` (default
//! [`DEFAULT_ENUM_CACHE_MAX_RESIDENT_BYTES`], driver flag
//! `--enum-cache-max-mb`) are REFUSED with a counted
//! [`EnumCache::budget_refusals`]: the row stays uncached and the step runs
//! the device path for it, so exactness is preserved.
//!
//! Integrity and concurrency semantics (task F8 item 6): the trailing
//! checksum detects accidental corruption, not tampering (anyone who can
//! rewrite the payload can recompute the trailer); two processes saving one
//! path race with last-writer-wins and no merge; a crash between temporary
//! creation and rename leaves an orphan `<path>.tmp-*` file behind, which
//! [`EnumCache::stale_temp_files`] lists (callers print, never auto-delete)
//! while the destination itself is never half-written; the containing
//! directory is not fsynced, so rename atomicity must not be read as
//! durability across power loss. Concurrent readers through `Arc` are safe
//! (lookups borrow `&self`; hit/miss accounting is atomic); insertion needs
//! `&mut self` and external synchronization for shared mutation.
//!
//! The per-use header check allocates nothing: callers compare borrowed
//! fields through [`ExpectedHeader`] / [`EnumCache::check_expected`]
//! (artifact hashes as `&str`, chemistry version as `&'static str`, scalars
//! directly). Evidence key hashing likewise hashes in place over the row
//! slices (no per-spectrum buffering). The hit path's remaining host cost is
//! candidate/counter expansion, evidence-output allocation and one upload per
//! served stage — no device readback, no relaunches.
//!
//! Batch-composition note (shuffled-spectrum control): the precompute pass is
//! given the same batches training and evaluation will upload. Under the
//! shuffled-spectrum control the uploaded rows carry donor peaks, and the
//! content hash then simply keys the donor-peak row. Donor assignment is
//! batch-composition-dependent, so training hits are unlikely under that
//! control; that is correct behaviour, not an error.
//!
//! Host data structures are pure host Rust; the `run_device_*_into`
//! precompute helpers drive the production device kernels (no copies).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use cubecl::prelude::Runtime;
use serde::{Deserialize, Serialize};

use crate::backend::{Device, FloatElem};
use crate::error::{Error, Result};
use crate::tensor::ops::index::{IdTensor, read_all};

use super::batch::DeviceSpectra;
use super::chem::{CHEMISTRY_VERSION, composition_mass};
use super::contract::SpectrumBatch;
use super::formula_enum::build_enum_meta;
use super::formula_evidence::EVIDENCE_PEAKS;
use super::formula_head::DeviceEnumArtifacts;
use crate::tensor::ops::ms2::{PeakBuffers, peak_select};
use crate::tensor::ops::ms2_enum::{EnumLaunch, cand_pad, enum_offsets};
use crate::tensor::ops::ms2_formula_evidence::{evidence_peaks, formula_evidence};

/// Magic bytes of the [`EnumCache`] binary file (`save` / `load`).
const CACHE_MAGIC: [u8; 8] = *b"MS2ENUMC";

/// Binary format version of the [`EnumCache`] file.
///
/// Version 4 adds the evidence identity fields to the header (`n_peaks`,
/// `evidence_dtype`) and the per-entry canonical evidence inputs with
/// collision buckets (task F8): the 128-bit hash is verified against the
/// stored canonical bytes on every hit. Version 1 (no evidence section),
/// version 2 (no checksum) and version 3 (no evidence identity, unverified
/// hash hits) files are refused by name (never silently reinterpreted).
pub const ENUM_CACHE_FORMAT_VERSION: u32 = 4;

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

/// Trailing checksum bytes of the [`EnumCache`] binary file (task F7A):
/// two little-endian `u64` words, the two independent 64-bit FNV-1a passes
/// (seeds [`FNV64_BASIS_A`] / [`FNV64_BASIS_B`]) over every preceding file
/// byte (magic, version, header JSON and all enumeration and evidence
/// entries alike). Verified on load before any entry is used.
const CHECKSUM_BYTES: usize = 16;

/// Minimum bytes of one enumeration entry on the wire with zero stored
/// candidates (key `8 * 4`, counters `5 * 4`, candidate-count word `4`).
/// Load reservations are bounded by `remaining_bytes / MIN_ENUM_ENTRY_BYTES`,
/// never by the advertised count alone (task F7A item A4).
const MIN_ENUM_ENTRY_BYTES: usize = 8 * 4 + COUNTER_WORDS * 4 + 4;

/// Minimum bytes of one evidence entry on the wire with zero scored rows
/// (meta `8 * 4`, hashes `8 + 8`, peak count `4`, m/z sum `8`, `n_ev` `1`,
/// row-count word `4`, canonical length `4`, empty canonical minimum `36`).
const MIN_EVIDENCE_ENTRY_BYTES: usize = 8 * 4 + 8 + 8 + 4 + 8 + 1 + 4 + 4 + 36;

/// Minimum bytes of one stored candidate on the wire.
const MIN_CAND_BYTES: usize = COMPACT_CAND_BYTES;

/// Minimum bytes of one scored evidence row on the wire (explained `1`,
/// weight `4`; the complete bits ride along at 1 bit per row).
const MIN_EVIDENCE_ROW_BYTES: usize = 1 + 4;

/// Default host resident-byte budget of a fresh [`EnumCache`] (task F8 item
/// 4): 4096 MiB. The driver surfaces it as `--enum-cache-max-mb`.
pub const DEFAULT_ENUM_CACHE_MAX_RESIDENT_BYTES: u64 = 4096 * 1024 * 1024;

/// Per-entry resident-overhead estimate of one enumeration entry (task F8
/// item 4): hash-map slot plus entry bookkeeping. Part of the
/// [`EnumCache::resident_bytes`] formula.
const ENUM_ENTRY_OVERHEAD: usize = 64 + 8;

/// Per-entry resident-overhead estimate of one evidence entry (task F8 item
/// 4): bucket slot plus entry bookkeeping. Part of the
/// [`EnumCache::resident_bytes`] formula.
const EVIDENCE_ENTRY_OVERHEAD: usize = 80 + 8;

/// Map-table bytes charged per physical BUCKET of the enumeration map
/// (task F10 item B1): `size_of::<(K, V)>()` plus the one hashbrown control
/// byte per bucket. Charged per physical bucket (the `buckets` high-water
/// mark below), not per usable slot: `map.capacity()` is what hashbrown
/// reports for admission-free growth, but the allocation holds every bucket
/// (usable slots are 7/8 of the buckets; tombstone removals can further
/// lower `capacity()` without releasing any bucket).
const ENUM_MAP_BUCKET_BYTES: usize =
    std::mem::size_of::<([u32; 8], CacheEntry)>() + 1;

/// See [`ENUM_MAP_BUCKET_BYTES`]: the per-bucket charge for the evidence map
/// (`EvidenceKey` plus its collision bucket).
const EVIDENCE_MAP_BUCKET_BYTES: usize =
    std::mem::size_of::<(EvidenceKey, Vec<EvidenceEntry>)>() + 1;

/// Flat per-map table overhead (task F10 item B1): hashbrown keeps an extra
/// control group past the buckets plus alignment padding for the data.
/// 32 bytes covers the trailing group (`Group::WIDTH` is 16 at most) and the
/// alignment slack on every supported platform. Counted once per allocated
/// map (no table, no overhead).
const MAP_GROUP_PAD_BYTES: usize = 32;

/// Bytes per unit of collision-bucket `Vec` capacity (task F9 item A2): a
/// bucket reserves `capacity * size_of::<EvidenceEntry>()` bytes, so unused
/// bucket capacity is part of the resident estimate, not just the stored
/// entries.
const EVIDENCE_BUCKET_SLOT_BYTES: usize = std::mem::size_of::<EvidenceEntry>();

/// Physical buckets hashbrown allocates for `want` elements (task F10 item
/// B1): the smallest power of two (minimum 4) whose usable slots cover
/// `want` — hashbrown 0.15.5's `capacity_to_buckets` for `(K, V)` with
/// `size_of > 3`, which both cache maps satisfy (the small-table
/// minimum-capacity rule agrees with it for those sizes; the large-table
/// rule is exactly 7/8 load rounded to a power of two).
///
/// The cache grows a table ONLY through its own budgeted `reserve(1)` calls
/// before inserts of NEW keys (replacements go through `get_mut`, a pure
/// lookup that never reserves): hashbrown's insert/entry paths grow only
/// through their leading `reserve(1)` (`find_or_find_insert_slot` reserves
/// before probing; `insert_in_slot` never grows), and the explicit reserve
/// guarantees `growth_left >= 1` afterwards, so the insert's own reserve is
/// a no-op and its probe is guaranteed an EMPTY slot — reusing a tombstone
/// or taking it without growing. A budgeted reserve for `want` elements
/// therefore lands on exactly this function (verified after every op by the
/// debug asserts, and by the randomised budget test in every build).
///
/// Because growth passes only through budgeted reserves, tombstones cannot
/// cause unbilled growth: a removal never lowers the tracked number, and
/// inserting into a tombstone only raises `len` towards the already-charged
/// usable count. Rehash transients (old plus new table while an explicitly
/// budgeted `reserve` reallocates) are bounded by twice the checked bucket
/// count and are momentary; the steady-state charge always covers the
/// surviving table.
///
/// `0` means unallocated; every positive value is a power of two.
fn required_buckets(want: usize) -> usize {
    if want == 0 {
        return 0;
    }
    let mut buckets = 4usize;
    while usable_buckets(buckets) < want {
        buckets = buckets.checked_mul(2).expect("bucket count overflow");
    }
    buckets
}

/// Usable element slots of a hashbrown table with `buckets` physical
/// buckets: `buckets - 1` for tiny tables (one empty slot is always kept),
/// 7/8 of the buckets above 8 (12.5% kept empty). This is hashbrown 0.15.5's
/// `bucket_mask_to_capacity`, and inverts [`required_buckets`].
fn usable_buckets(buckets: usize) -> usize {
    if buckets <= 8 {
        buckets.saturating_sub(1)
    } else {
        buckets / 8 * 7
    }
}

/// Prospective physical buckets for a NEW key when the table is full
/// (`len == capacity`, so the budgeted `reserve` will realloc — task F10
/// item B1): hashbrown resizes for `max(len + 1, usable + 1)` ("at least
/// the next size up, to avoid churning deletes into frequent rehashes") —
/// the `usable + 1` term is what turns the reviewer's 28-usable table into
/// 64 buckets even for a key that takes an EMPTY slot. `usable` here is
/// the tracked high-water's usable count, an upper bound of the table's
/// own (buckets never shrink), so the charge always covers the resize. When
/// the reserve turns out to rehash in place (room locked in tombstones),
/// the charge is conservative by at most one tier — safe, and the tracked
/// mark never goes down anyway.
fn growth_buckets(len: usize, tracked: usize) -> usize {
    required_buckets(
        len.saturating_add(1)
            .max(usable_buckets(tracked).saturating_add(1)),
    )
    .max(tracked)
}

/// Prospective `Vec` capacity after pushing one element (task F9 item A2):
/// the current capacity when the push fits, otherwise a documented upper
/// bound of the grown capacity (`Vec` doubles on growth; `max(4, _)` covers
/// the cold `0 -> 4` first allocation). The randomised-budget regression
/// test asserts this bound against the real post-push capacity.
fn prospective_vec_capacity(len: usize, capacity: usize) -> usize {
    if len < capacity {
        return capacity;
    }
    if capacity == 0 {
        return 4;
    }
    capacity.saturating_mul(2).max(len.saturating_add(1))
}

/// Header of an [`EnumCache`]: everything the cached enumeration depends on.
///
/// Two runs share cache entries only when every field agrees: the artifacts'
/// SHA-256 (domain and bounds), the rare-table depth `P`, the window `M`,
/// `formula_rows_scored_max`, `enum_lane_visits_max`, the model's kept-peak
/// capacity `n_peaks` (task F8 item 1: the evidence stage sees only the kept
/// peaks) and the chemistry version. The format version guards the binary
/// layout itself. The floating element dtype the evidence was computed in
/// (`evidence_dtype`, e.g. `"f32"`) rides along for the evidence section
/// only (task F8 item 2): it is enforced when evidence is built and on every
/// evidence lookup, but NOT by [`EnumCacheHeader::check_compatible`], so
/// integer enumeration entries stay reusable across dtypes.
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
    /// Kept-peak capacity `N` the evidence entries were built with (task F8
    /// item 1).
    pub n_peaks: u32,
    /// Floating element dtype the evidence entries were computed in, e.g.
    /// `"f32"` (task F8 item 2; enumeration entries are dtype-independent).
    pub evidence_dtype: String,
    /// Chemistry version ([`CHEMISTRY_VERSION`]).
    pub chemistry_version: String,
}

/// The borrowed expectation a per-use header check compares against (task F8
/// item 5): every field is borrowed (`&str`) or a scalar, so
/// [`EnumCacheHeader::check_expected`] allocates nothing. The evidence dtype
/// is deliberately absent: it is enforced per evidence operation (build
/// refuses, lookup misses) so enumeration reuse across dtypes keeps working.
#[derive(Clone, Copy, Debug)]
pub struct ExpectedHeader<'a> {
    /// Binary format version ([`ENUM_CACHE_FORMAT_VERSION`]).
    pub format_version: u32,
    /// SHA-256 of the enum domain JSON the caller enumerates with.
    pub domain_sha256: &'a str,
    /// SHA-256 of the ratio bounds JSON the caller enumerates with.
    pub bounds_sha256: &'a str,
    /// Rare-table depth `P` of the caller's enumerating source.
    pub p: u32,
    /// Scored-candidate window `M` the caller enumerates with.
    pub window_m: u32,
    /// Scored-cap argument the caller enumerates with.
    pub formula_rows_scored_max: u32,
    /// Per-lane visit budget the caller enumerates with.
    pub enum_lane_visits_max: u32,
    /// Kept-peak capacity `N` of the caller's model.
    pub n_peaks: u32,
    /// Chemistry version ([`CHEMISTRY_VERSION`]).
    pub chemistry_version: &'a str,
}

/// Build the borrowed per-use expectation from artifact fields and scalar
/// configuration (task F8 item 5): no allocation, so the cached search path
/// can check the header on every batch without cloning the SHA strings.
///
/// `p` is the rare-table depth as `u32` (callers convert from `usize` with
/// the same overflow refusal as the owned header constructors).
pub fn expected_header<'a>(
    domain_sha256: &'a str,
    bounds_sha256: &'a str,
    p: u32,
    window_m: u32,
    formula_rows_scored_max: u32,
    enum_lane_visits_max: u32,
    n_peaks: u32,
) -> ExpectedHeader<'a> {
    ExpectedHeader {
        format_version: ENUM_CACHE_FORMAT_VERSION,
        domain_sha256,
        bounds_sha256,
        p,
        window_m,
        formula_rows_scored_max,
        enum_lane_visits_max,
        n_peaks,
        chemistry_version: CACHE_CHEMISTRY_VERSION,
    }
}

impl EnumCacheHeader {
    /// Build the header callers stamp on a fresh cache: today's format and
    /// chemistry versions with the given enumeration parameters, the model's
    /// kept-peak capacity `n_peaks` (task F8 item 1) and the floating element
    /// dtype the evidence is computed in (task F8 item 2, e.g. `"f32"`).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        domain_sha256: String,
        bounds_sha256: String,
        p: u32,
        window_m: u32,
        formula_rows_scored_max: u32,
        enum_lane_visits_max: u32,
        n_peaks: u32,
        evidence_dtype: &str,
    ) -> Self {
        Self {
            format_version: ENUM_CACHE_FORMAT_VERSION,
            domain_sha256,
            bounds_sha256,
            p,
            window_m,
            formula_rows_scored_max,
            enum_lane_visits_max,
            n_peaks,
            evidence_dtype: evidence_dtype.to_string(),
            chemistry_version: CACHE_CHEMISTRY_VERSION.to_string(),
        }
    }

    /// Check `expected` against `self` field by field without allocating
    /// (task F8 item 5): the first mismatch is
    /// [`Error::Config`](crate::error::Error::Config) naming the field, so a
    /// stale cache is never silently reused across headers. The evidence
    /// dtype is not compared here (see the [`EnumCacheHeader`] docs): it is
    /// enforced per evidence operation instead.
    pub fn check_expected(&self, expected: &ExpectedHeader<'_>) -> Result<()> {
        if self.format_version != expected.format_version {
            return Err(Error::config(format!(
                "EnumCache header mismatch on field `format_version`: cache {} != expected {}",
                self.format_version, expected.format_version
            )));
        }
        if self.domain_sha256 != expected.domain_sha256 {
            return Err(Error::config(format!(
                "EnumCache header mismatch on field `domain_sha256`: cache {} != expected {}",
                self.domain_sha256, expected.domain_sha256
            )));
        }
        if self.bounds_sha256 != expected.bounds_sha256 {
            return Err(Error::config(format!(
                "EnumCache header mismatch on field `bounds_sha256`: cache {} != expected {}",
                self.bounds_sha256, expected.bounds_sha256
            )));
        }
        if self.p != expected.p {
            return Err(Error::config(format!(
                "EnumCache header mismatch on field `p`: cache {} != expected {}",
                self.p, expected.p
            )));
        }
        if self.window_m != expected.window_m {
            return Err(Error::config(format!(
                "EnumCache header mismatch on field `window_m`: cache {} != expected {}",
                self.window_m, expected.window_m
            )));
        }
        if self.formula_rows_scored_max != expected.formula_rows_scored_max {
            return Err(Error::config(format!(
                "EnumCache header mismatch on field `formula_rows_scored_max`: cache {} != expected {}",
                self.formula_rows_scored_max, expected.formula_rows_scored_max
            )));
        }
        if self.enum_lane_visits_max != expected.enum_lane_visits_max {
            return Err(Error::config(format!(
                "EnumCache header mismatch on field `enum_lane_visits_max`: cache {} != expected {}",
                self.enum_lane_visits_max, expected.enum_lane_visits_max
            )));
        }
        if self.n_peaks != expected.n_peaks {
            return Err(Error::config(format!(
                "EnumCache header mismatch on field `n_peaks`: cache {} != expected {}",
                self.n_peaks, expected.n_peaks
            )));
        }
        if self.chemistry_version != expected.chemistry_version {
            return Err(Error::config(format!(
                "EnumCache header mismatch on field `chemistry_version`: cache {:?} != expected {:?}",
                self.chemistry_version, expected.chemistry_version
            )));
        }
        Ok(())
    }

    /// Check `other` against `self` field by field: the first mismatch is
    /// [`Error::Config`](crate::error::Error::Config) naming the field, so a
    /// stale cache is never silently reused across headers. Borrowed
    /// comparison (no allocation); the evidence dtype is not compared (see
    /// the [`EnumCacheHeader`] docs).
    pub fn check_compatible(&self, other: &EnumCacheHeader) -> Result<()> {
        self.check_expected(&ExpectedHeader {
            format_version: other.format_version,
            domain_sha256: other.domain_sha256.as_str(),
            bounds_sha256: other.bounds_sha256.as_str(),
            p: other.p,
            window_m: other.window_m,
            formula_rows_scored_max: other.formula_rows_scored_max,
            enum_lane_visits_max: other.enum_lane_visits_max,
            n_peaks: other.n_peaks,
            chemistry_version: other.chemistry_version.as_str(),
        })
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

/// FNV-1a 64-bit prime (shared by both evidence hash passes).
const FNV64_PRIME: u64 = 1_099_511_628_211;

/// First evidence-hash seed (the FNV-1a offset basis).
const FNV64_BASIS_A: u64 = 0xcbf2_9ce4_8422_2325;
/// Second evidence-hash seed (bitwise complement of the first, so the two
/// 64-bit passes are independent).
const FNV64_BASIS_B: u64 = 0x3d0d_613b_7bde_ddda;

/// One FNV-1a 64-bit pass over `bytes` from `basis`.
fn fnv1a64(bytes: &[u8], basis: u64) -> u64 {
    let mut h = basis;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(FNV64_PRIME);
    }
    h
}

/// The 128-bit payload checksum of a [`EnumCache::save`] body (task F7A):
/// two independent 64-bit FNV-1a passes (seeds [`FNV64_BASIS_A`] /
/// [`FNV64_BASIS_B`]) over every byte of the body (magic, version, header
/// JSON and all enumeration and evidence entries alike). Stored as the last
/// [`CHECKSUM_BYTES`] file bytes, verified on load before any entry is
/// used.
fn payload_checksum(body: &[u8]) -> (u64, u64) {
    (fnv1a64(body, FNV64_BASIS_A), fnv1a64(body, FNV64_BASIS_B))
}

/// Bound a load-time reservation by what the file can still hold (task F7A
/// item A4): `min(advertised, remaining / min_bytes)`, so a tiny file
/// advertising a million entries reserves a handful of slots instead of a
/// million before the parse rejects it.
fn bounded_cap(advertised: usize, remaining: usize, min_bytes: usize) -> usize {
    advertised.min(remaining / min_bytes.max(1))
}

/// Running resident estimate while parsing (task F9 item A4, re-homed on
/// physical buckets by task F10 item B1): the same function
/// [`EnumCache::resident_bytes`] reports — tracked entry bytes (including
/// bucket slack, folded into `tracked` on both the insert and the load path)
/// plus the per-map physical-bucket terms at their tracked bucket counts.
fn parse_resident(tracked: usize, enum_buckets: usize, evidence_buckets: usize) -> usize {
    tracked
        .saturating_add(map_table_bytes(ENUM_MAP_BUCKET_BYTES, enum_buckets))
        .saturating_add(map_table_bytes(EVIDENCE_MAP_BUCKET_BYTES, evidence_buckets))
}

/// Physical table bytes for a map with `buckets` tracked physical buckets
/// (task F10 item B1): per-bucket bytes plus the flat group pad, or zero
/// when the map is unallocated.
fn map_table_bytes(per_bucket: usize, buckets: usize) -> usize {
    if buckets == 0 {
        0
    } else {
        per_bucket
            .saturating_mul(buckets)
            .saturating_add(MAP_GROUP_PAD_BYTES)
    }
}

/// Sum of uploaded m/z words (the cheap pre-check stored per evidence
/// entry, ahead of the full canonical verification).
fn mz_sum_of(mz_row: &[u32]) -> u64 {
    mz_row.iter().map(|&m| u64::from(m)).sum()
}

/// The 128-bit evidence key of one spectrum (task T6B).
///
/// The enumeration meta row fixes the candidate rows; the 128-bit content
/// hash fixes everything else `evidence_peaks` and `formula_evidence` read
/// for that spectrum (see [`evidence_key_for_batch`]). The hash is verified
/// against the stored canonical inputs on every hit (task F8 item 3), so two
/// keys that compare equal with different canonical bytes coexist in
/// per-key buckets instead of aliasing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EvidenceKey {
    /// The spectrum's 8-word enumeration meta row (fixes the candidates).
    pub meta: [u32; 8],
    /// First 64 bits of the content hash.
    pub h0: u64,
    /// Second 64 bits of the content hash (independent seed).
    pub h1: u64,
}

/// Build an evidence key from explicit words (test hook, hidden from the
/// public docs): lets a regression test force two different canonical inputs
/// onto the same 128-bit key and check both entries stay servable. Never
/// used on the production path, where keys come from
/// [`evidence_key_for_batch`].
#[doc(hidden)]
pub fn evidence_key_with_forced_hash(meta: [u32; 8], h0: u64, h1: u64) -> EvidenceKey {
    EvidenceKey { meta, h0, h1 }
}

/// One cached spectrum's evidence value (task T6B): per scored slot the
/// explained count, the explained weight bits and the complete bit, plus the
/// per-spectrum valid evidence-peak count. The stored peak count and m/z sum
/// are a cheap pre-check; the stored canonical bytes are the exactness
/// check (task F8 item 3): a hash hit whose canonical bytes differ from the
/// lookup's is a miss, never silent corruption.
#[derive(Clone, Debug, PartialEq, Eq)]
struct EvidenceEntry {
    /// Number of valid evidence peaks of this spectrum (`<= 32`).
    n_ev: u8,
    /// Uploaded peak count this entry was built from (pre-check).
    peak_count: u32,
    /// Sum of the uploaded m/z words (pre-check).
    mz_sum: u64,
    /// Explained count per scored slot (`<= 32`).
    explained: Vec<u8>,
    /// Explained-weight bits per scored slot (unchanged `f32` bits).
    weight_bits: Vec<u32>,
    /// Complete flag per scored slot (`0`/`1`).
    complete: Vec<u8>,
    /// Canonical evidence inputs this entry was built from (task F8 item 3):
    /// exactly the bytes the content hash covers (see
    /// [`evidence_canonical_bytes`]).
    canonical: Vec<u8>,
}

/// Tracked resident bytes of one enumeration entry with `cands_cap`
/// candidate slots (task F9 item A2): the 8-word key (32 bytes), the
/// `counters` row (`COUNTER_WORDS` words) and the compact candidates, plus
/// the per-entry overhead. Part of the [`EnumCache::resident_bytes`] formula.
fn enum_entry_tracked(cands_cap: usize) -> usize {
    32 + COUNTER_WORDS * 4 + cands_cap * COMPACT_CAND_BYTES + ENUM_ENTRY_OVERHEAD
}

/// Tracked resident bytes of one evidence entry, by vector CAPACITY (task F9
/// item A2): callers shrink entry vectors to their lengths before admitting,
/// so capacity equals length at insert time. Part of the
/// [`EnumCache::resident_bytes`] formula.
fn evidence_entry_tracked(e: &EvidenceEntry) -> usize {
    48 + e.canonical.capacity()
        + e.explained.capacity()
        + 4 * e.weight_bits.capacity()
        + e.complete.capacity()
        + 16
        + EVIDENCE_ENTRY_OVERHEAD
}

/// A borrowed view of one cached spectrum's evidence value.
#[derive(Clone, Debug)]
pub struct EvidenceRef<'a> {
    /// Number of valid evidence peaks of this spectrum.
    pub n_ev: u8,
    /// Uploaded peak count this entry was built from (pre-check).
    pub peak_count: u32,
    /// Sum of the uploaded m/z words (pre-check).
    pub mz_sum: u64,
    /// Explained count per scored slot.
    pub explained: &'a [u8],
    /// Explained-weight bits per scored slot.
    pub weight_bits: &'a [u32],
    /// Complete flag per scored slot.
    pub complete: &'a [u8],
}

/// Canonical byte encoding of the evidence content hash (task T6B).
///
/// Little-endian `u32` words in this fixed order: uploaded peak count,
/// intensity scale, precursor word (the precursor `evidence_peaks` selects
/// peaks under, `meta[1]`), adduct id, fragment tolerance in tenths of a ppm
/// (stored `0` resolved to the default `100`), m/z uncertainty `U`,
/// `formula_evidence_work_max`, `EVIDENCE_PEAKS` (`P`), the hydrogen-cap bound
/// the kernel clamps with (`h_cap_max`); then per kept-peak slot in slot
/// order the uploaded m/z word and the uploaded intensity bits
/// (`f32::to_bits`). Only the first `peak_count` slots (clamped to `n_raw`)
/// are encoded: padding slots are never read. Precursor tolerance,
/// precursor uncertainty, spectrum ids and the dispatch sizing (`dispatch
/// bound`, `tol_max`) do not affect `cand_ev` (the lane takes the absolute
/// lane index; `formula_features` is recomputed, not memoised) and are not
/// encoded.
///
/// No new dependency: two independent 64-bit FNV-1a passes with different
/// seeds over these bytes.
///
/// Task F8 item 5: the bytes are hashed in place over the row slices (no
/// per-spectrum buffering); [`evidence_canonical_bytes`] materialises the
/// same layout for storage on the build path only.
fn evidence_content_hash(
    mz_row: &[u32],
    intensity_row: &[f32],
    intensity_scale: u32,
    precursor: u32,
    adduct: u32,
    fragment_ppm: u32,
    mz_uncertainty: u32,
    peak_count: u32,
    work_max: u32,
    p: u32,
    h_cap_max: u32,
) -> (u64, u64) {
    debug_assert_eq!(mz_row.len(), intensity_row.len());
    let mut h0 = FNV64_BASIS_A;
    let mut h1 = FNV64_BASIS_B;
    // One macro-free closure: feed one `u32` word (little-endian) into both
    // passes without materialising any buffer.
    let mut feed = |w: u32| {
        for b in w.to_le_bytes() {
            h0 ^= u64::from(b);
            h0 = h0.wrapping_mul(FNV64_PRIME);
            h1 ^= u64::from(b);
            h1 = h1.wrapping_mul(FNV64_PRIME);
        }
    };
    for w in [
        peak_count,
        intensity_scale,
        precursor,
        adduct,
        fragment_ppm,
        mz_uncertainty,
        work_max,
        p,
        h_cap_max,
    ] {
        feed(w);
    }
    // `p` above is `EVIDENCE_PEAKS`; the loop below is the kept peaks (the
    // ninth header word count differs from this loop's length on purpose).
    for (m, f) in mz_row.iter().zip(intensity_row.iter()) {
        feed(*m);
        feed(f.to_bits());
    }
    (h0, h1)
}

/// Borrowed canonical evidence inputs of one spectrum (task F8 item 3): the
/// exact uploaded row slices plus every scalar that enters the evidence key.
/// Borrowed throughout, so lookups verify hash hits without allocating.
#[derive(Clone, Copy, Debug)]
pub struct EvidenceInputs<'a> {
    /// Uploaded m/z words of the kept slots (`count = min(peak_count,
    /// `n_raw`) of them, in slot order).
    pub mz_row: &'a [u32],
    /// Uploaded intensities of the same slots (compared by
    /// [`f32::to_bits`]).
    pub intensity_row: &'a [f32],
    /// Intensity scale (`0` linear, `1` square root).
    pub intensity_scale: u32,
    /// Precursor word the peaks were selected under.
    pub precursor: u32,
    /// Adduct id.
    pub adduct: u32,
    /// Fragment tolerance in tenths of a ppm (`0` resolved to 100).
    pub fragment_ppm: u32,
    /// m/z uncertainty `U`.
    pub mz_uncertainty: u32,
    /// EXACT uploaded peak count (`0` for a fatal host status).
    pub peak_count: u32,
    /// `formula_evidence_work_max` the kernel runs with.
    pub work_max: u32,
    /// `EVIDENCE_PEAKS` (`P`).
    pub p: u32,
    /// Hydrogen-cap bound the kernel clamps with.
    pub h_cap_max: u32,
}

/// One verified evidence lookup (task F8 item 3): the hash key, the scored
/// count the entry's vectors must match, and the borrowed canonical inputs
/// the stored bytes are compared against on a hash hit.
#[derive(Clone, Copy, Debug)]
pub struct EvidenceQuery<'a> {
    /// The spectrum's evidence key (meta row plus content hash).
    pub key: EvidenceKey,
    /// Scored count from the spectrum's enumeration `counters` row.
    pub rows_scored: usize,
    /// Borrowed canonical inputs for the full hit verification.
    pub inputs: EvidenceInputs<'a>,
}

/// Borrow the canonical evidence inputs of uploaded row `b` of `batch`
/// (task F8 item 3): no allocation, so the hit path verifies hash hits over
/// the row slices in place.
///
/// `uploaded_peak_count` is the EXACT uploaded peak count (`0` for a
/// spectrum with a fatal host status). Under the shuffled-spectrum control
/// callers pass the rotated (donor-peak) batch — never the pre-rotation
/// request — and under trainer donor assembly the donor-assembled batch.
pub fn evidence_inputs_for_batch(
    batch: &SpectrumBatch,
    b: usize,
    uploaded_peak_count: u32,
    work_max: u32,
    h_cap_max: u32,
) -> EvidenceInputs<'_> {
    use super::formula_evidence::EVIDENCE_PEAKS;
    let n_raw = batch.n_raw as usize;
    let base = b * n_raw;
    let count = (uploaded_peak_count as usize).min(n_raw);
    EvidenceInputs {
        mz_row: &batch.mz_udalton[base..base + count],
        intensity_row: &batch.intensity[base..base + count],
        intensity_scale: u32::from(batch.intensity_scale),
        precursor: batch.precursor_mz_udalton[b],
        adduct: u32::from(batch.adduct[b]),
        fragment_ppm: batch.fragment_tolerance(b),
        mz_uncertainty: batch.mz_uncertainty_udalton[b],
        peak_count: uploaded_peak_count,
        work_max,
        p: EVIDENCE_PEAKS as u32,
        h_cap_max,
    }
}

/// Materialise the canonical bytes of `inputs` (task F8 item 3): the exact
/// layout [`evidence_content_hash`] hashes, stored per entry on the build
/// path and compared in full on every hash hit. Allocates; the lookup path
/// uses [`canonical_inputs_match`] instead and never materialises these.
pub fn evidence_canonical_bytes(inputs: &EvidenceInputs<'_>) -> Vec<u8> {
    let mut buf = Vec::with_capacity(9 * 4 + inputs.mz_row.len() * 8);
    for w in [
        inputs.peak_count,
        inputs.intensity_scale,
        inputs.precursor,
        inputs.adduct,
        inputs.fragment_ppm,
        inputs.mz_uncertainty,
        inputs.work_max,
        inputs.p,
        inputs.h_cap_max,
    ] {
        buf.extend_from_slice(&w.to_le_bytes());
    }
    debug_assert_eq!(inputs.mz_row.len(), inputs.intensity_row.len());
    for (m, f) in inputs.mz_row.iter().zip(inputs.intensity_row.iter()) {
        buf.extend_from_slice(&m.to_le_bytes());
        buf.extend_from_slice(&f.to_bits().to_le_bytes());
    }
    buf
}

/// Whether `stored` (bytes of [`evidence_canonical_bytes`]) equals `inputs`
/// (task F8 item 3): the full hit verification, streaming over the stored
/// bytes without allocating. A malformed `stored` (short, or not a multiple
/// of a slot past the 9-word scalar prefix) never matches.
fn canonical_inputs_match(stored: &[u8], inputs: &EvidenceInputs<'_>) -> bool {
    if inputs.mz_row.len() != inputs.intensity_row.len() {
        return false;
    }
    if stored.len() != 9 * 4 + inputs.mz_row.len() * 8 {
        return false;
    }
    let mut words = stored.chunks_exact(4);
    let scalar = |w: &[u8]| u32::from_le_bytes([w[0], w[1], w[2], w[3]]);
    let expect = [
        inputs.peak_count,
        inputs.intensity_scale,
        inputs.precursor,
        inputs.adduct,
        inputs.fragment_ppm,
        inputs.mz_uncertainty,
        inputs.work_max,
        inputs.p,
        inputs.h_cap_max,
    ];
    for e in expect {
        match words.next() {
            Some(w) if scalar(w) == e => {}
            _ => return false,
        }
    }
    for (m, f) in inputs.mz_row.iter().zip(inputs.intensity_row.iter()) {
        match (words.next(), words.next()) {
            (Some(wm), Some(wi)) if scalar(wm) == *m && scalar(wi) == f.to_bits() => {}
            _ => return false,
        }
    }
    true
}

/// Whether `stored` canonical bytes are structurally valid (task F8): the
/// 9-word scalar prefix plus whole slots, at most 512 slots (`n_raw` never
/// exceeds 512), with the embedded peak-count word equal to `peak_count`.
/// Load rejects anything else as corruption, never parses past it.
fn canonical_bytes_valid(stored: &[u8], peak_count: u32) -> bool {
    if stored.len() < 9 * 4 || (stored.len() - 9 * 4) % 8 != 0 {
        return false;
    }
    if (stored.len() - 9 * 4) / 8 > 512 {
        return false;
    }
    u32::from_le_bytes([stored[0], stored[1], stored[2], stored[3]]) == peak_count
}

/// Build the evidence key of uploaded row `b` of `batch` (task T6B).
///
/// `meta_row` is the spectrum's 8-word enumeration meta row (fixes the
/// candidates), `uploaded_peak_count` the EXACT uploaded peak count (`0` for
/// a spectrum with a fatal host status — what [`DeviceSpectra`](super::batch::DeviceSpectra)
/// uploads), `work_max` the `formula_evidence_work_max` the kernel runs with,
/// `h_cap_max` the hydrogen-cap bound the kernel clamps with, and `p`
/// `EVIDENCE_PEAKS`. Returns the key and the m/z sum (the collision second
/// check alongside `uploaded_peak_count`).
///
/// Callers hash the EXACT uploaded row: under the shuffled-spectrum control
/// that is the donor-peak row (rotate first), and under trainer donor
/// assembly it is the donor-assembled batch — never the pre-rotation
/// request. A different uploaded peak, intensity bit, fragment ppm, m/z
/// uncertainty, adduct, `work_max`, `P` or `h_cap_max` is a different key
/// (a miss, never an approximate match).
pub fn evidence_key_for_batch(
    batch: &SpectrumBatch,
    b: usize,
    uploaded_peak_count: u32,
    meta_row: [u32; 8],
    work_max: u32,
    h_cap_max: u32,
) -> (EvidenceKey, u64) {
    use super::formula_evidence::EVIDENCE_PEAKS;
    let n_raw = batch.n_raw as usize;
    let base = b * n_raw;
    let count = (uploaded_peak_count as usize).min(n_raw);
    let mz_row = &batch.mz_udalton[base..base + count];
    let intensity_row = &batch.intensity[base..base + count];
    let mut mz_sum: u64 = 0;
    for &m in mz_row {
        mz_sum += u64::from(m);
    }
    // Task F8 item 5: hash in place over the row slices — the two
    // per-spectrum buffering allocations are gone.
    let (h0, h1) = evidence_content_hash(
        mz_row,
        intensity_row,
        u32::from(batch.intensity_scale),
        batch.precursor_mz_udalton[b],
        u32::from(batch.adduct[b]),
        batch.fragment_tolerance(b),
        batch.mz_uncertainty_udalton[b],
        uploaded_peak_count,
        work_max,
        EVIDENCE_PEAKS as u32,
        h_cap_max,
    );
    (EvidenceKey { meta: meta_row, h0, h1 }, mz_sum)
}

/// Per-process [`EnumCache::save`] sequence: together with the pid and the
/// wall clock it makes every save's temporary file unique, so concurrent
/// saves to the same destination never share a temporary file (task F7A
/// item A3).
static SAVE_SEQ: AtomicU64 = AtomicU64::new(0);

/// A per-call unique temporary path in the destination's directory (task
/// F7A item A3): `<path>.tmp-<pid>-<seq>-<nanos>`, created with exclusive
/// creation by [`EnumCache::save`].
fn unique_tmp_path(path: &Path) -> PathBuf {
    let seq = SAVE_SEQ.fetch_add(1, Ordering::Relaxed);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(format!(".tmp-{}-{seq}-{nanos}", std::process::id()));
    PathBuf::from(tmp)
}

/// Phase of [`EnumCache::save`] just completed (test hook, hidden from the
/// public docs): lets a regression test force two saves to overlap
/// deterministically (barrier between phases) instead of hoping threads
/// collide. The production path never sets a hook.
///
/// Task F9 item A6: this hook exists only under `cfg(any(test, feature =
/// "test-support"))`, so a normal build takes no mutex per save phase and
/// exposes no setter. The `ms2_enum_cache` test target enables the
/// `test-support` feature (see `required-features` in `Cargo.toml`).
#[cfg(any(test, feature = "test-support"))]
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SavePhase {
    /// The unique temporary file was created exclusively.
    Created,
    /// The body was written and fsynced, before the rename.
    Written,
    /// The temporary file was renamed over the destination.
    Renamed,
}

/// Global save-phase hook (test hook, hidden from the public docs): when
/// set, [`EnumCache::save`] calls it after each [`SavePhase`]. The hook
/// lives only under `cfg(any(test, feature = "test-support"))` (task F9 item
/// A6); callers must reset to `None` when done (a set hook serialises nothing
/// by itself; pair it with a barrier).
#[cfg(any(test, feature = "test-support"))]
#[doc(hidden)]
pub fn set_save_phase_hook(hook: Option<fn(SavePhase)>) {
    *SAVE_PHASE_HOOK.lock().unwrap_or_else(|e| e.into_inner()) = hook;
}

/// The global save-phase hook slot (see [`set_save_phase_hook`]): only
/// present under `cfg(any(test, feature = "test-support"))` (task F9 item
/// A6), so production saves take no lock.
#[cfg(any(test, feature = "test-support"))]
static SAVE_PHASE_HOOK: Mutex<Option<fn(SavePhase)>> = Mutex::new(None);

/// Run the global save-phase hook, if any (never fails the save). The hook
/// runs WITHOUT the slot lock held (a barrier-style hook must be able to
/// rendezvous with another save inside the hook). Only present under
/// `cfg(any(test, feature = "test-support"))` (task F9 item A6).
#[cfg(any(test, feature = "test-support"))]
fn save_phase(phase: SavePhase) {
    let hook: Option<fn(SavePhase)> = *SAVE_PHASE_HOOK
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    if let Some(hook) = hook {
        hook(phase);
    }
}

/// Exact memo of the device enumeration for the enumerating formula source.
///
/// See the module docs. The map key is the spectrum's 8-word enumeration
/// meta row exactly as `build_enum_meta` produces it; the value is that
/// spectrum's `counters` row and its scored candidates. A partially cached
/// batch is never mixed: [`EnumCache::expand_batch`] returns `None` when ANY
/// key misses, and callers then run the device enumeration exactly as
/// without a cache.
///
/// Task T6B adds the sibling evidence map: [`EvidenceKey`] (meta row plus
/// the 128-bit content hash) to per-key BUCKETS of per-spectrum `cand_ev`
/// values ([`EnumCache::insert_evidence`] /
/// [`EnumCache::expand_evidence_batch`]). Buckets hold colliding keys side by
/// side (task F8 item 3); every hash hit is verified against the stored
/// canonical inputs and a mismatch is a miss counted in [`collisions`](EnumCache::collisions).
/// The two stages hit or miss independently: an enumeration hit with an
/// evidence miss still serves the enumeration (uploaded `cand`/`counters`)
/// and runs the device evidence stage. The table source has no enumeration
/// entry, so the evidence memo is never used there (the device path runs).
///
/// Concurrent readers through `Arc` are safe (lookups borrow `&self`;
/// [`collisions`](EnumCache::collisions) and
/// [`budget_refusals`](EnumCache::budget_refusals) are atomic); insertion
/// needs `&mut self`.
#[derive(Debug)]
pub struct EnumCache {
    /// What the entries were enumerated with; checked on every use.
    header: EnumCacheHeader,
    /// Meta row → (`counters` row, scored candidates).
    map: HashMap<[u32; 8], CacheEntry>,
    /// Evidence key → bucket of per-spectrum `cand_ev` values (task T6B;
    /// buckets coexist on 128-bit key collision, task F8 item 3).
    evidence: HashMap<EvidenceKey, Vec<EvidenceEntry>>,
    /// Hash hits refused by the canonical-input verification (task F8 item
    /// 3). Atomic so concurrent readers through `Arc` keep exact totals.
    collisions: AtomicU64,
    /// Inserts refused by the resident-byte budget (task F8 item 4). Atomic
    /// so concurrent readers through `Arc` keep exact totals.
    budget_refusals: AtomicU64,
    /// Batch enumeration expansions attempted through
    /// [`EnumCache::expand_batch`]. Atomic so concurrent readers through
    /// `Arc` keep exact totals (task F8 item 7).
    enum_lookups: AtomicU64,
    /// Batch enumeration expansions fully served from the cache. Atomic (see
    /// [`EnumCache::enum_lookups`](Self::enum_lookups)).
    enum_hits: AtomicU64,
    /// Batch evidence expansions attempted through
    /// [`EnumCache::expand_evidence_batch`]. Atomic (see
    /// [`EnumCache::enum_lookups`](Self::enum_lookups)).
    evidence_lookups: AtomicU64,
    /// Batch evidence expansions fully served from the cache. Atomic (see
    /// [`EnumCache::enum_lookups`](Self::enum_lookups)).
    evidence_hits: AtomicU64,
    /// Tracked entry bytes of the [`EnumCache::resident_bytes`] formula
    /// (entry estimates plus unused collision-bucket capacity, without the
    /// live map-table terms, which [`EnumCache::resident_bytes`] adds from
    /// the bucket high-water marks below).
    tracked: usize,
    /// Tracked physical buckets of the enumeration map (task F10 item B1):
    /// a high-water mark (a power of two, `0` when unallocated) the cache
    /// updates only through the explicit `reserve` calls it makes before
    /// inserts — never implicitly, never down on removal. The map cannot
    /// hold more physical buckets than this (debug-asserted after every
    /// mutating op), so charging it always covers the table.
    enum_buckets: usize,
    /// Tracked physical buckets of the evidence map: the same high-water
    /// mark for the sibling table.
    evidence_buckets: usize,
    /// Resident-byte budget: an insert that would push
    /// [`EnumCache::resident_bytes`] past it is refused (counted, the row
    /// stays uncached). Not persisted: a runtime limit, not cache identity.
    max_resident_bytes: u64,
}

impl Clone for EnumCache {
    /// Clone the entries, the header and the budget; counters restart from
    /// their current values (a clone is a new accounting scope, not a new
    /// observation).
    fn clone(&self) -> Self {
        Self {
            header: self.header.clone(),
            map: self.map.clone(),
            evidence: self.evidence.clone(),
            collisions: AtomicU64::new(self.collisions.load(Ordering::Relaxed)),
            budget_refusals: AtomicU64::new(self.budget_refusals.load(Ordering::Relaxed)),
            enum_lookups: AtomicU64::new(self.enum_lookups.load(Ordering::Relaxed)),
            enum_hits: AtomicU64::new(self.enum_hits.load(Ordering::Relaxed)),
            evidence_lookups: AtomicU64::new(self.evidence_lookups.load(Ordering::Relaxed)),
            evidence_hits: AtomicU64::new(self.evidence_hits.load(Ordering::Relaxed)),
            tracked: self.tracked,
            enum_buckets: self.enum_buckets,
            evidence_buckets: self.evidence_buckets,
            max_resident_bytes: self.max_resident_bytes,
        }
    }
}

/// Load-time reservation statistics (test hook, hidden from the public
/// docs): the capacities [`EnumCache::parse`] actually reserved, so a
/// regression test can assert the reservation itself is bounded by what the
/// file can still hold — not just that the parse rejects. See
/// [`EnumCache::parse_with_stats`].
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParseStats {
    /// Capacity reserved for the enumeration map.
    pub enum_reservation: usize,
    /// Capacity reserved for the evidence map.
    pub evidence_reservation: usize,
    /// Candidate slots reserved across all enumeration entries.
    pub candidate_reservation: usize,
    /// Scored evidence rows reserved across all evidence entries.
    pub evidence_row_reservation: usize,
}

impl EnumCache {
    /// An empty cache for `header`, with the default resident-byte budget
    /// ([`DEFAULT_ENUM_CACHE_MAX_RESIDENT_BYTES`]).
    pub fn new(header: EnumCacheHeader) -> Self {
        Self {
            header,
            map: HashMap::new(),
            evidence: HashMap::new(),
            collisions: AtomicU64::new(0),
            budget_refusals: AtomicU64::new(0),
            enum_lookups: AtomicU64::new(0),
            enum_hits: AtomicU64::new(0),
            evidence_lookups: AtomicU64::new(0),
            evidence_hits: AtomicU64::new(0),
            tracked: 0,
            enum_buckets: 0,
            evidence_buckets: 0,
            max_resident_bytes: DEFAULT_ENUM_CACHE_MAX_RESIDENT_BYTES,
        }
    }

    /// An empty cache for `header` with resident-byte budget `max` (task F8
    /// item 4).
    pub fn with_max_resident_bytes(header: EnumCacheHeader, max: u64) -> Self {
        let mut cache = Self::new(header);
        cache.max_resident_bytes = max;
        cache
    }

    /// Set the resident-byte budget (task F8 item 4): an insert that would
    /// push [`EnumCache::resident_bytes`] past `max` is refused (counted,
    /// the row stays uncached).
    pub fn set_max_resident_bytes(&mut self, max: u64) {
        self.max_resident_bytes = max;
    }

    /// The resident-byte budget (task F8 item 4).
    pub fn max_resident_bytes(&self) -> u64 {
        self.max_resident_bytes
    }

    /// The header the entries were enumerated with.
    pub fn header(&self) -> &EnumCacheHeader {
        &self.header
    }

    /// Hash hits refused by the canonical-input verification (task F8 item
    /// 3): a mismatch is a MISS, never a served entry.
    pub fn collisions(&self) -> u64 {
        self.collisions.load(Ordering::Relaxed)
    }

    /// Batch expansions attempted and served, as
    /// `(enum_lookups, enum_hits, evidence_lookups, evidence_hits)` (task F8
    /// item 7): every [`EnumCache::expand_batch`] call counts one enumeration
    /// lookup (and one hit when the whole batch was served), every
    /// [`EnumCache::expand_evidence_batch`] call one evidence lookup (and
    /// one hit when the whole batch was served). Atomic, so concurrent
    /// readers through `Arc` keep exact totals after workers finish
    /// (single-row [`EnumCache::get_evidence`] lookups and the build pass's
    /// internal checks do not count here). Not persisted across save/load.
    pub fn lookup_stats(&self) -> (u64, u64, u64, u64) {
        (
            self.enum_lookups.load(Ordering::Relaxed),
            self.enum_hits.load(Ordering::Relaxed),
            self.evidence_lookups.load(Ordering::Relaxed),
            self.evidence_hits.load(Ordering::Relaxed),
        )
    }

    /// Inserts refused by the resident-byte budget (task F8 item 4): the row
    /// stayed uncached and its step ran the device path.
    pub fn budget_refusals(&self) -> u64 {
        self.budget_refusals.load(Ordering::Relaxed)
    }

    /// Estimated resident host bytes of this cache (task F8 item 4, fixed by
    /// task F9 item A2, re-homed on physical buckets by task F10 item B1):
    /// an ESTIMATE, not an allocator readout, with this exact formula —
    ///
    /// ```text
    /// resident_bytes() = tracked
    ///     + ENUM_MAP_BUCKET_BYTES * enum_buckets (+ group pad when allocated)
    ///     + EVIDENCE_MAP_BUCKET_BYTES * evidence_buckets (+ group pad when allocated)
    /// tracked = Σ enum entries [32 key + 20 counters + 12 * cands.capacity() + ENUM_ENTRY_OVERHEAD]
    ///         + Σ evidence entries [48 key + canonical.capacity() + explained.capacity()
    ///           + 4 * weight_bits.capacity() + complete.capacity() + 16 scalar words
    ///           + EVIDENCE_ENTRY_OVERHEAD]
    ///         + Σ evidence buckets [bucket.capacity() * size_of::<EvidenceEntry>()]
    /// ```
    ///
    /// Entry vectors are shrunk to their lengths on insert
    /// ([`EnumCache::insert`], [`EnumCache::insert_evidence`]), so the
    /// capacity terms above equal the stored lengths; the map terms charge
    /// physical buckets (`size_of::<(K, V)>()` + 1 control byte per bucket,
    /// plus one group of padding per allocated map) at the tracked
    /// high-water marks, and the bucket term covers unused collision-bucket
    /// capacity. Admission ([`EnumCache::insert`],
    /// [`EnumCache::insert_evidence`]) evaluates this same function for the
    /// state AFTER the insert — prospective entry bytes and the prospective
    /// bucket high-water marks (bucket `Vec` growth via
    /// [`prospective_vec_capacity`], table growth via [`required_buckets`]
    /// checked BEFORE the budgeted `reserve`), minus any replaced entry — so
    /// every accepted insert keeps
    /// `resident_bytes() <= max_resident_bytes`.
    ///
    /// Separate from [`EnumCache::bytes`] (the serialized file size, which
    /// counts lengths rather than capacities and no overhead).
    pub fn resident_bytes(&self) -> usize {
        self.tracked
            .saturating_add(map_table_bytes(ENUM_MAP_BUCKET_BYTES, self.enum_buckets))
            .saturating_add(map_table_bytes(
                EVIDENCE_MAP_BUCKET_BYTES,
                self.evidence_buckets,
            ))
    }

    /// Whether inserting an entry with prospective footprint `need` (tracked
    /// bytes, computed for the state after the insert) and prospective table
    /// bucket counts `enum_buckets` / `evidence_buckets` would exceed the
    /// budget (task F8 item 4, fixed by task F10 item B1): the SAME function
    /// [`EnumCache::resident_bytes`] reports, evaluated prospectively.
    fn over_budget_prospective(&self, need: usize, enum_buckets: usize, evidence_buckets: usize) -> bool {
        (need as u64)
            .saturating_add(map_table_bytes(ENUM_MAP_BUCKET_BYTES, enum_buckets) as u64)
            .saturating_add(map_table_bytes(EVIDENCE_MAP_BUCKET_BYTES, evidence_buckets) as u64)
            > self.max_resident_bytes
    }

    /// Tracked physical buckets of the two tables, as `(enum, evidence)`
    /// (task F10 item B1): the high-water marks the budget accounts.
    /// Test support for the budget-invariant regression tests.
    pub fn table_buckets(&self) -> (usize, usize) {
        (self.enum_buckets, self.evidence_buckets)
    }

    /// Live usable capacities of the two tables, as `(enum, evidence)`:
    /// what `map.capacity()` reports. Test support: together with
    /// [`EnumCache::table_buckets`] the invariant test asserts the tracked
    /// buckets always cover the implied physical buckets.
    pub fn table_capacities(&self) -> (usize, usize) {
        (self.map.capacity(), self.evidence.capacity())
    }

    /// Whether the bucket invariant holds (task F10 item B1): every table's
    /// live usable capacity fits in its tracked physical buckets. The debug
    /// asserts inside the mutating ops check the same after every op; the
    /// invariant test asserts this in all builds.
    pub fn bucket_invariant_holds(&self) -> bool {
        required_buckets(self.map.capacity()) <= self.enum_buckets
            && required_buckets(self.evidence.capacity()) <= self.evidence_buckets
    }

    /// Debug-check the bucket invariant after a mutating op (task F10 item
    /// B1): the live usable capacity of each table fits in its tracked
    /// physical buckets — i.e. the table never grew past what the cache
    /// budgeted and reserved.
    fn debug_check_buckets(&self) {
        debug_assert!(
            required_buckets(self.map.capacity()) <= self.enum_buckets,
            "enum table outgrew its tracked buckets: capacity {} needs {} buckets, tracked {}",
            self.map.capacity(),
            required_buckets(self.map.capacity()),
            self.enum_buckets
        );
        debug_assert!(
            required_buckets(self.evidence.capacity()) <= self.evidence_buckets,
            "evidence table outgrew its tracked buckets: capacity {} needs {} buckets, tracked {}",
            self.evidence.capacity(),
            required_buckets(self.evidence.capacity()),
            self.evidence_buckets
        );
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
    /// a present key overwrites it AND invalidates every evidence entry keyed
    /// with its meta row (task F9 item A5: the old evidence was computed for
    /// the old candidates). `m` must equal the header's `window_m`
    /// (the width the caller read `cand` at), else `Error::Config`: entries
    /// read at another width do not belong in this cache.
    ///
    /// Admission (task F10 item B1) is computed from the PROSPECTIVE
    /// bucket-based footprint — the same function
    /// [`EnumCache::resident_bytes`] reports, evaluated for the state after
    /// the insert (the entry vector shrunk to its length, the prospective
    /// physical-bucket high-water marks, minus the replaced entry and its
    /// invalidated evidence). The table `reserve` runs only after admission,
    /// so the table cannot grow past the checked buckets. Returns `true`
    /// when the entry was stored, `false` when the insert would push
    /// [`EnumCache::resident_bytes`] past the budget and was
    /// REFUSED (counted in [`EnumCache::budget_refusals`]; the row stays
    /// uncached and its step runs the device path — exactness preserved). An
    /// accepted insert always leaves `resident_bytes() <= max_resident_bytes`.
    pub fn insert(
        &mut self,
        key: [u32; 8],
        counters: [u32; COUNTER_WORDS],
        cand_rows: &[u32],
        m: usize,
    ) -> Result<bool> {
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
        // Shrink the over-reserved vector so capacity equals length (task F9
        // item A2): admission below is computed with this exact capacity.
        cands.shrink_to_fit();
        let is_new = !self.map.contains_key(&key);
        let old_tracked = self
            .map
            .get(&key)
            .map(|old| enum_entry_tracked(old.cands.capacity()))
            .unwrap_or(0);
        // Task F9 item A5: the replacement candidates change what evidence
        // means, so the old evidence keyed with this meta row is evicted with
        // the replacement — and its freed bytes count toward admission.
        let evicted_tracked = if is_new {
            0
        } else {
            self.evidence_tracked_for_meta(&key)
        };
        let prospective_tracked = self
            .tracked
            .saturating_sub(old_tracked)
            .saturating_sub(evicted_tracked)
            .saturating_add(enum_entry_tracked(cands.capacity()));
        // Task F10 item B1: the prospective physical buckets. Only a NEW
        // key can grow the table: a replacement overwrites its entry in
        // place through `get_mut` below (a pure lookup — no reserve, no
        // rehash, no growth), so it needs no new buckets. For a new key
        // into a full table (`len == capacity`) the budgeted `reserve`
        // reallocates for `max(len + 1, usable + 1)` (see `growth_buckets`);
        // into a non-full table the reserve is a no-op (`growth_left =
        // capacity - len >= 1` already) and the tracked number stands. The
        // budget is checked with THIS number before the reserve, so the
        // table cannot grow on its own.
        let prospective_enum_buckets = if is_new && self.map.len() == self.map.capacity() {
            growth_buckets(self.map.len(), self.enum_buckets)
        } else {
            self.enum_buckets
        };
        if self.over_budget_prospective(
            prospective_tracked,
            prospective_enum_buckets,
            self.evidence_buckets,
        ) {
            self.budget_refusals.fetch_add(1, Ordering::Relaxed);
            return Ok(false);
        }
        // The budgeted reserve first (new keys only); the tracked
        // high-water follows. A removal never lowers it (tombstones keep
        // their buckets), so the charge is a monotone upper bound of the
        // physical table.
        if is_new {
            self.map.reserve(1);
            self.enum_buckets = prospective_enum_buckets;
        }
        self.debug_check_buckets();
        if !is_new {
            self.evict_evidence_for_meta(&key);
            self.tracked = self.tracked.saturating_sub(old_tracked);
        }
        self.tracked = self
            .tracked
            .saturating_add(enum_entry_tracked(cands.capacity()));
        if is_new {
            self.map.insert(key, CacheEntry { counters, cands });
        } else if let Some(slot) = self.map.get_mut(&key) {
            slot.counters = counters;
            slot.cands = cands;
        }
        self.debug_check_buckets();
        debug_assert!(
            (self.resident_bytes() as u64) <= self.max_resident_bytes,
            "accepted enum insert left resident_bytes() over budget"
        );
        Ok(true)
    }

    /// Tracked resident bytes (entries plus unused bucket capacity) of every
    /// evidence bucket whose key meta row equals `meta` (task F9 item A5).
    fn evidence_tracked_for_meta(&self, meta: &[u32; 8]) -> usize {
        self.evidence
            .iter()
            .filter(|(k, _)| k.meta == *meta)
            .map(|(_, bucket)| {
                bucket.iter().map(evidence_entry_tracked).sum::<usize>().saturating_add(
                    bucket.capacity().saturating_mul(EVIDENCE_BUCKET_SLOT_BYTES),
                )
            })
            .sum()
    }

    /// Remove every evidence bucket whose key meta row equals `meta`,
    /// subtracting its tracked bytes (task F9 item A5): replacing an
    /// enumeration entry invalidates the evidence computed for the old
    /// candidates. Map tables never shrink on removal (and the tracked
    /// bucket high-water marks never go down, task F10 item B1), so the
    /// table charge is unchanged and `resident_bytes()` can only fall.
    fn evict_evidence_for_meta(&mut self, meta: &[u32; 8]) {
        let evicted = self.evidence_tracked_for_meta(meta);
        self.evidence.retain(|k, _| k.meta != *meta);
        self.tracked = self.tracked.saturating_sub(evicted);
    }

    /// Spectra cached.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Whether no spectrum is cached.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Spectra with an evidence entry cached (task T6B; colliding keys
    /// contribute every bucket member, task F8 item 3).
    pub fn evidence_len(&self) -> usize {
        self.evidence.values().map(Vec::len).sum()
    }

    /// Look up one spectrum's evidence value (task T6B), VERIFIED against
    /// the canonical inputs (task F8 item 3): a hash hit whose stored
    /// canonical bytes differ from `inputs` is a MISS (counted in
    /// [`EnumCache::collisions`]), never a served entry. Colliding keys
    /// coexist, so the matching bucket member is served.
    pub fn get_evidence<'s, 'q>(
        &'s self,
        key: &EvidenceKey,
        inputs: &EvidenceInputs<'q>,
    ) -> Option<EvidenceRef<'s>> {
        let bucket = self.evidence.get(key)?;
        for e in bucket {
            if e.peak_count != inputs.peak_count || e.mz_sum != mz_sum_of(inputs.mz_row) {
                self.collisions.fetch_add(1, Ordering::Relaxed);
                continue;
            }
            if !canonical_inputs_match(&e.canonical, inputs) {
                self.collisions.fetch_add(1, Ordering::Relaxed);
                continue;
            }
            return Some(EvidenceRef {
                n_ev: e.n_ev,
                peak_count: e.peak_count,
                mz_sum: e.mz_sum,
                explained: &e.explained,
                weight_bits: &e.weight_bits,
                complete: &e.complete,
            });
        }
        None
    }

    /// Insert one spectrum's evidence value (task T6B): the number of valid
    /// evidence peaks `n_ev` (`<= 32`), the uploaded `peak_count` and `mz_sum`
    /// pre-check, per scored slot the explained count, the explained weight
    /// bits and the complete flag (`0`/`1`), and the `canonical` evidence
    /// inputs the entry was built from ([`evidence_canonical_bytes`], task F8
    /// item 3).
    ///
    /// Every vector must have `rows_scored` elements (`rows_scored <= m`),
    /// every explained count and `n_ev` must be `<= 32`, every complete flag
    /// `0` or `1`, and `canonical` must be structurally valid for
    /// `peak_count` (9-word scalar prefix plus whole slots, embedded
    /// peak-count word equal); `m` must equal the header's `window_m` (the
    /// width the caller read `cand_ev` at), else
    /// [`Error::Config`](crate::error::Error::Config). Re-inserting an
    /// identical (key, canonical) entry overwrites it; a colliding key with
    /// different canonical bytes coexists in the key's bucket.
    ///
    /// Returns `true` when the entry was stored, `false` when the insert
    /// would push [`EnumCache::resident_bytes`] past the budget and was
    /// REFUSED (counted in [`EnumCache::budget_refusals`]; the row stays
    /// uncached and its step runs the device path — exactness preserved). An
    /// accepted insert always leaves `resident_bytes() <= max_resident_bytes`.
    ///
    /// Admission (task F10 item B1) is computed from the PROSPECTIVE
    /// bucket-based footprint — the same function
    /// [`EnumCache::resident_bytes`] reports, evaluated for the state after
    /// the insert: the entry vectors are shrunk to their lengths first
    /// (`shrink_to_fit`, so capacity equals length), and the estimate covers
    /// the replacement delta, the bucket `Vec` capacity growth and the
    /// prospective physical-bucket high-water marks (the table `reserve`
    /// runs only after admission).
    #[allow(clippy::too_many_arguments)]
    pub fn insert_evidence(
        &mut self,
        key: EvidenceKey,
        peak_count: u32,
        mz_sum: u64,
        n_ev: u8,
        mut explained: Vec<u8>,
        mut weight_bits: Vec<u32>,
        mut complete: Vec<u8>,
        m: usize,
        mut canonical: Vec<u8>,
    ) -> Result<bool> {
        if m as u64 != u64::from(self.header.window_m) {
            return Err(Error::config(format!(
                "EnumCache::insert_evidence: cand_ev width {m} != header window_m {}",
                self.header.window_m
            )));
        }
        let rows = explained.len();
        if weight_bits.len() != rows || complete.len() != rows {
            return Err(Error::config(format!(
                "EnumCache::insert_evidence: {} explained vs {} weights vs {} complete",
                rows,
                weight_bits.len(),
                complete.len()
            )));
        }
        if rows > m {
            return Err(Error::config(format!(
                "EnumCache::insert_evidence: rows_scored {rows} exceeds cand_ev width {m}"
            )));
        }
        if n_ev > 32 {
            return Err(Error::config(format!(
                "EnumCache::insert_evidence: n_ev {n_ev} exceeds 32"
            )));
        }
        for (i, &e) in explained.iter().enumerate() {
            if e > 32 {
                return Err(Error::config(format!(
                    "EnumCache::insert_evidence: explained[{i}] {e} exceeds 32"
                )));
            }
        }
        for (i, &c) in complete.iter().enumerate() {
            if c != 0 && c != 1 {
                return Err(Error::config(format!(
                    "EnumCache::insert_evidence: complete[{i}] {c} is not 0 or 1"
                )));
            }
        }
        if !canonical_bytes_valid(&canonical, peak_count) {
            return Err(Error::config(
                "EnumCache::insert_evidence: canonical evidence inputs are malformed for peak_count".to_string(),
            ));
        }
        // Shrink the over-reserved vectors so capacity equals length (task F9
        // item A2): admission below is computed with these exact capacities,
        // and the stored entry keeps them.
        explained.shrink_to_fit();
        weight_bits.shrink_to_fit();
        complete.shrink_to_fit();
        canonical.shrink_to_fit();
        // Capacity-based footprint of the new entry (same terms as
        // `evidence_entry_tracked`, with the shrunk capacities above).
        let new_entry_tracked = 48
            + canonical.capacity()
            + explained.capacity()
            + 4 * weight_bits.capacity()
            + complete.capacity()
            + 16
            + EVIDENCE_ENTRY_OVERHEAD;
        let (bucket_len, bucket_cap, replaces) = match self.evidence.get(&key) {
            None => (0, 0, false),
            Some(bucket) => (
                bucket.len(),
                bucket.capacity(),
                bucket.iter().any(|e| e.canonical == canonical),
            ),
        };
        let replaced_tracked = if replaces {
            self.evidence
                .get(&key)
                .and_then(|b| b.iter().find(|e| e.canonical == canonical))
                .map(evidence_entry_tracked)
                .unwrap_or(0)
        } else {
            0
        };
        let prospective_bucket_cap = if replaces {
            bucket_cap
        } else {
            prospective_vec_capacity(bucket_len, bucket_cap)
        };
        let bucket_slack_delta = prospective_bucket_cap
            .saturating_mul(EVIDENCE_BUCKET_SLOT_BYTES)
            .saturating_sub(bucket_cap.saturating_mul(EVIDENCE_BUCKET_SLOT_BYTES));
        let prospective_tracked = self
            .tracked
            .saturating_sub(replaced_tracked)
            .saturating_add(new_entry_tracked)
            .saturating_add(bucket_slack_delta);
        // Task F10 item B1: only a NEW key can grow the table (a new
        // collision-bucket member for a present key, or an in-place
        // overwrite of an identical entry, touches no table slot — both go
        // through `get_mut`, a pure lookup that never reserves). A new key
        // into a full table reallocates for `max(len + 1, usable + 1)` (see
        // `growth_buckets`); into a non-full table the reserve is a no-op.
        let key_present = self.evidence.contains_key(&key);
        let prospective_ev_buckets = if key_present || self.evidence.len() < self.evidence.capacity() {
            self.evidence_buckets
        } else {
            growth_buckets(self.evidence.len(), self.evidence_buckets)
        };
        if self.over_budget_prospective(
            prospective_tracked,
            self.enum_buckets,
            prospective_ev_buckets,
        ) {
            self.budget_refusals.fetch_add(1, Ordering::Relaxed);
            return Ok(false);
        }
        // The budgeted reserve first (new keys only; see `insert`); the
        // tracked high-water follows.
        if !key_present {
            self.evidence.reserve(1);
            self.evidence_buckets = prospective_ev_buckets;
        }
        self.debug_check_buckets();
        if key_present {
            let bucket = self
                .evidence
                .get_mut(&key)
                .expect("checked present above");
            if let Some(pos) = bucket.iter().position(|e| e.canonical == canonical) {
                self.tracked = self.tracked.saturating_sub(evidence_entry_tracked(&bucket[pos]));
                bucket[pos] = EvidenceEntry {
                    n_ev,
                    peak_count,
                    mz_sum,
                    explained,
                    weight_bits,
                    complete,
                    canonical,
                };
                self.tracked = self
                    .tracked
                    .saturating_add(evidence_entry_tracked(&bucket[pos]));
            } else {
                let old_bucket_cap = bucket.capacity();
                let entry = EvidenceEntry {
                    n_ev,
                    peak_count,
                    mz_sum,
                    explained,
                    weight_bits,
                    complete,
                    canonical,
                };
                self.tracked = self.tracked.saturating_add(evidence_entry_tracked(&entry));
                bucket.push(entry);
                // The bucket may have grown: track the new unused bucket capacity
                // (task F9 item A2).
                self.tracked = self.tracked.saturating_add(
                    bucket
                        .capacity()
                        .saturating_sub(old_bucket_cap)
                        .saturating_mul(EVIDENCE_BUCKET_SLOT_BYTES),
                );
            }
        } else {
            let entry = EvidenceEntry {
                n_ev,
                peak_count,
                mz_sum,
                explained,
                weight_bits,
                complete,
                canonical,
            };
            self.tracked = self.tracked.saturating_add(evidence_entry_tracked(&entry));
            let mut bucket = Vec::new();
            bucket.push(entry);
            // The new bucket's unused capacity is part of the resident
            // estimate, like the growth above (`Vec` grows `0 -> 4` here,
            // covered by the prospective `bucket_slack_delta`).
            self.tracked = self.tracked.saturating_add(
                bucket
                    .capacity()
                    .saturating_mul(EVIDENCE_BUCKET_SLOT_BYTES),
            );
            self.evidence.insert(key, bucket);
        }
        debug_assert!(
            (self.resident_bytes() as u64) <= self.max_resident_bytes,
            "accepted evidence insert left resident_bytes() over budget"
        );
        self.debug_check_buckets();
        Ok(true)
    }

    /// Expand a whole batch's `cand_ev [B, M, 4]` (flat row-major `f32`) from
    /// the evidence map (task T6B).
    ///
    /// `queries[i]` carries spectrum `i`'s key, its scored count (from its
    /// enumeration `counters` row) and its borrowed canonical inputs. Every
    /// hash hit is VERIFIED against the stored canonical bytes (task F8 item
    /// 3): a mismatch is a miss (counted in [`EnumCache::collisions`]),
    /// never a served entry. `n_peaks` must equal the header's `n_peaks`
    /// (task F8 item 1) and `dtype` (e.g. `"f32"`) the header's
    /// `evidence_dtype` (task F8 item 2), else the whole batch misses and the
    /// caller runs the device evidence stage.
    ///
    /// Returns `None` when ANY query misses, when `m` differs from the
    /// header's `window_m`, or when a stored length differs from its
    /// `rows_scored`: a partially cached batch takes the device path (no
    /// mixing). Otherwise the buffer is bit-identical to what the device
    /// evidence stage produced (explained count, weight bits, `n_ev`,
    /// complete flag), with padding slots exact `0`. Counts one evidence
    /// lookup (and one hit when served) in [`EnumCache::lookup_stats`].
    pub fn expand_evidence_batch(
        &self,
        queries: &[EvidenceQuery<'_>],
        m: usize,
        n_peaks: u32,
        dtype: &str,
    ) -> Option<Vec<f32>> {
        self.evidence_lookups.fetch_add(1, Ordering::Relaxed);
        let out = self.expand_evidence_batch_core(queries, m, n_peaks, dtype);
        if out.is_some() {
            self.evidence_hits.fetch_add(1, Ordering::Relaxed);
        }
        out
    }

    /// [`EnumCache::expand_evidence_batch`] without the
    /// [`EnumCache::lookup_stats`] accounting: the build pass's internal use
    /// (a fully cached batch is skipped without counting a lookup).
    pub(crate) fn expand_evidence_batch_core(
        &self,
        queries: &[EvidenceQuery<'_>],
        m: usize,
        n_peaks: u32,
        dtype: &str,
    ) -> Option<Vec<f32>> {
        if m as u64 != u64::from(self.header.window_m) {
            return None;
        }
        if n_peaks != self.header.n_peaks || dtype != self.header.evidence_dtype.as_str() {
            return None;
        }
        let b = queries.len();
        let mut out = vec![0.0f32; b * m * 4];
        for (i, q) in queries.iter().enumerate() {
            let bucket = self.evidence.get(&q.key)?;
            let mut served = false;
            for entry in bucket {
                if entry.peak_count != q.inputs.peak_count
                    || entry.mz_sum != mz_sum_of(q.inputs.mz_row)
                {
                    self.collisions.fetch_add(1, Ordering::Relaxed);
                    continue;
                }
                if !canonical_inputs_match(&entry.canonical, &q.inputs) {
                    self.collisions.fetch_add(1, Ordering::Relaxed);
                    continue;
                }
                let scored = q.rows_scored;
                if entry.explained.len() != scored
                    || entry.weight_bits.len() != scored
                    || entry.complete.len() != scored
                    || scored > m
                {
                    return None;
                }
                let nev_f = f32::from(entry.n_ev);
                for r in 0..scored {
                    let base = (i * m + r) * 4;
                    out[base] = f32::from(entry.explained[r]);
                    out[base + 1] = f32::from_bits(entry.weight_bits[r]);
                    out[base + 2] = nev_f;
                    out[base + 3] = f32::from(entry.complete[r]);
                }
                served = true;
                break;
            }
            if !served {
                return None;
            }
        }
        Some(out)
    }

    /// Exact byte size of the [`EnumCache::save`] file for these entries
    /// (trailing [`CHECKSUM_BYTES`] checksum included).
    pub fn bytes(&self) -> usize {
        let header_json = self.header_json();
        let mut n = CACHE_MAGIC.len() + 4 + 4 + header_json.len() + 8;
        for e in self.map.values() {
            n += 8 * 4 + COUNTER_WORDS * 4 + 4 + e.cands.len() * COMPACT_CAND_BYTES;
        }
        // Evidence section: count word, then per entry the key (8 meta words,
        // two hash words), the pre-check (peak count, m/z sum), `n_ev`,
        // `rows_scored`, the per-slot explained/weight bytes, the complete
        // bits packed LSB-first, and the canonical-input length plus bytes
        // (task F8 item 3).
        n += 8;
        for bucket in self.evidence.values() {
            for e in bucket {
                n += 8 * 4 + 8 + 8 + 4 + 8 + 1 + 4;
                n += e.explained.len() * (1 + 4) + e.complete.len().div_ceil(8);
                n += 4 + e.canonical.len();
            }
        }
        n += CHECKSUM_BYTES;
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
    /// produced, including padding slots and status bits. Counts one
    /// enumeration lookup (and one hit when served) in
    /// [`EnumCache::lookup_stats`].
    pub fn expand_batch(
        &self,
        keys: &[[u32; 8]],
        m: usize,
    ) -> Option<(Vec<u32> /* cand [B, M, 13] */, Vec<u32> /* counters [B, 5] */)> {
        self.enum_lookups.fetch_add(1, Ordering::Relaxed);
        let out = self.expand_batch_core(keys, m);
        if out.is_some() {
            self.enum_hits.fetch_add(1, Ordering::Relaxed);
        }
        out
    }

    /// [`EnumCache::expand_batch`] without the
    /// [`EnumCache::lookup_stats`] accounting: the build pass's internal
    /// use (a fully cached batch is skipped without counting a lookup).
    pub(crate) fn expand_batch_core(
        &self,
        keys: &[[u32; 8]],
        m: usize,
    ) -> Option<(Vec<u32>, Vec<u32>)> {
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

    /// Save to `path`: magic, version, header JSON, enum entries, then the
    /// evidence entries (each with its canonical inputs, task T6B/F8), then
    /// the 128-bit payload checksum over all of it (task F7A),
    /// little-endian. Written atomically: the body is written to a per-call
    /// unique temporary file created with exclusive creation in the
    /// destination directory, fsynced, then renamed over the destination, so
    /// concurrent saves never share a temporary file and a crash never
    /// leaves a half-written cache behind. A failed save removes its
    /// temporary file.
    ///
    /// Two processes saving one path race with last-writer-wins and no merge
    /// (task F8 item 6); a crash between creation and rename leaves an orphan
    /// `<path>.tmp-*` file behind — listed by
    /// [`EnumCache::stale_temp_files`], never auto-deleted. The containing
    /// directory is not fsynced, so the rename must not be read as
    /// durability across power loss.
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
        // Evidence section (task T6B): count, then sorted entries, then per
        // entry the canonical-input length plus bytes (task F8 item 3).
        // Buckets flatten to one record per entry, sorted by (key,
        // canonical), so the file is deterministic run to run.
        buf.extend_from_slice(&(self.evidence_len() as u64).to_le_bytes());
        let mut erecs: Vec<(&EvidenceKey, &EvidenceEntry)> = self
            .evidence
            .iter()
            .flat_map(|(k, bucket)| bucket.iter().map(move |e| (k, e)))
            .collect();
        erecs.sort_by(|a, b| (a.0, &a.1.canonical).cmp(&(b.0, &b.1.canonical)));
        for (key, entry) in erecs {
            for w in key.meta.iter() {
                buf.extend_from_slice(&w.to_le_bytes());
            }
            buf.extend_from_slice(&key.h0.to_le_bytes());
            buf.extend_from_slice(&key.h1.to_le_bytes());
            buf.extend_from_slice(&entry.peak_count.to_le_bytes());
            buf.extend_from_slice(&entry.mz_sum.to_le_bytes());
            buf.push(entry.n_ev);
            buf.extend_from_slice(&(entry.explained.len() as u32).to_le_bytes());
            for (e, wb) in entry.explained.iter().zip(entry.weight_bits.iter()) {
                buf.push(*e);
                buf.extend_from_slice(&wb.to_le_bytes());
            }
            // Complete bits packed LSB-first.
            let nbytes = entry.complete.len().div_ceil(8);
            for i in 0..nbytes {
                let mut byte = 0u8;
                for bit in 0..8 {
                    let r = i * 8 + bit;
                    if r < entry.complete.len() && entry.complete[r] == 1 {
                        byte |= 1 << bit;
                    }
                }
                buf.push(byte);
            }
            // Canonical evidence inputs (task F8 item 3).
            buf.extend_from_slice(&(entry.canonical.len() as u32).to_le_bytes());
            buf.extend_from_slice(&entry.canonical);
        }
        debug_assert_eq!(buf.len() + CHECKSUM_BYTES, self.bytes());
        let (chk0, chk1) = payload_checksum(&buf);
        buf.extend_from_slice(&chk0.to_le_bytes());
        buf.extend_from_slice(&chk1.to_le_bytes());
        debug_assert_eq!(buf.len(), self.bytes());
        let tmp_path = unique_tmp_path(path);
        let result: Result<()> = (|| {
            use std::io::Write;
            let mut f = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&tmp_path)
                .map_err(|e| {
                    Error::config(format!(
                        "EnumCache::save: cannot create {}: {e}",
                        tmp_path.display()
                    ))
                })?;
            #[cfg(any(test, feature = "test-support"))]
            save_phase(SavePhase::Created);
            f.write_all(&buf).map_err(|e| {
                Error::config(format!(
                    "EnumCache::save: cannot write {}: {e}",
                    tmp_path.display()
                ))
            })?;
            f.sync_all().map_err(|e| {
                Error::config(format!(
                    "EnumCache::save: cannot sync {}: {e}",
                    tmp_path.display()
                ))
            })?;
            #[cfg(any(test, feature = "test-support"))]
            save_phase(SavePhase::Written);
            drop(f);
            std::fs::rename(&tmp_path, path).map_err(|e| {
                Error::config(format!(
                    "EnumCache::save: cannot rename {} to {}: {e}",
                    tmp_path.display(),
                    path.display()
                ))
            })?;
            #[cfg(any(test, feature = "test-support"))]
            save_phase(SavePhase::Renamed);
            Ok(())
        })();
        if result.is_err() {
            std::fs::remove_file(&tmp_path).ok();
        }
        result
    }

    /// Orphan temporary files of `path` (task F8 item 6): entries of `path`'s
    /// directory whose name starts with `<file>.tmp-` — what a crash between
    /// temporary creation and rename leaves behind. Sorted. Callers print
    /// them; they are never auto-deleted here (deletion races a live saver).
    /// An unreadable directory yields an empty list, never an error.
    pub fn stale_temp_files(path: &Path) -> Vec<PathBuf> {
        let dir = path.parent().filter(|p| !p.as_os_str().is_empty());
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            return Vec::new();
        };
        let prefix = format!("{name}.tmp-");
        let mut out = Vec::new();
        let dir_path: &Path = dir.unwrap_or(Path::new("."));
        let Ok(entries) = std::fs::read_dir(dir_path) else {
            return Vec::new();
        };
        for entry in entries.flatten() {
            let entry_name = entry.file_name();
            let Some(entry_str) = entry_name.to_str() else {
                continue;
            };
            if entry_str.starts_with(&prefix) {
                out.push(entry.path());
            }
        }
        out.sort();
        out
    }

    /// Load from `path`, checking the stored header against `expected`
    /// field by field ([`Error::Config`](crate::error::Error::Config)
    /// naming the field on mismatch — never silently rebuild over a
    /// mismatching file).
    ///
    /// The version, the payload checksum and the header compatibility are
    /// all checked BEFORE any entry is parsed or used; a truncated or
    /// corrupted file is an error, never a panic, and no reservation is
    /// ever sized by an advertised count alone (task F7A items A2, A4).
    ///
    /// The budget is the default ([`DEFAULT_ENUM_CACHE_MAX_RESIDENT_BYTES`]);
    /// to enforce a smaller host limit use
    /// [`EnumCache::load_with_budget`].
    pub fn load(path: &Path, expected: &EnumCacheHeader) -> Result<Self> {
        Self::load_with_budget(path, expected, DEFAULT_ENUM_CACHE_MAX_RESIDENT_BYTES)
    }

    /// Load from `path` like [`EnumCache::load`], with resident-byte budget
    /// `max_resident_bytes` (task F9 item A4): the requested host limit
    /// governs cache acceptance, including load time.
    ///
    /// Refuses with [`Error::Config`](crate::error::Error::Config) naming the
    /// budget and the size — never a partially loaded cache — in two stages:
    /// BEFORE reading the payload when the file size alone exceeds the
    /// budget, and during parsing when the running resident estimate (the
    /// same function [`EnumCache::resident_bytes`] reports) exceeds it. The
    /// loaded cache carries this budget (further inserts refuse past it).
    ///
    /// The file is opened ONCE (task F10 item B2): the size comes from that
    /// handle's metadata, and the payload is read through
    /// [`Read::take`](std::io::Read::take) bounded by that size — so a file
    /// replaced or grown after the check cannot be read past the limit. A
    /// replacement that is shorter fails the payload checksum instead of
    /// loading partially.
    pub fn load_with_budget(
        path: &Path,
        expected: &EnumCacheHeader,
        max_resident_bytes: u64,
    ) -> Result<Self> {
        let file = std::fs::File::open(path).map_err(|e| {
            Error::config(format!(
                "EnumCache::load: cannot read {}: {e}",
                path.display()
            ))
        })?;
        let file_len = file
            .metadata()
            .map_err(|e| {
                Error::config(format!(
                    "EnumCache::load: cannot read {}: {e}",
                    path.display()
                ))
            })?
            .len();
        if file_len > max_resident_bytes {
            return Err(Error::config(format!(
                "EnumCache::load: {} is {file_len} file bytes, over the {max_resident_bytes}-byte resident budget (--enum-cache-max-mb governs loads too): refusing the cached file before reading it",
                path.display()
            )));
        }
        Self::load_from_reader(file, file_len, path, expected, max_resident_bytes)
    }

    /// Load from an open handle with a declared size (task F10 item B2):
    /// the handle-based worker behind [`EnumCache::load_with_budget`]. At
    /// most `size_limit` bytes are ever read (through
    /// [`Read::take`](std::io::Read::take)), so a stream longer than its
    /// declared size is truncated to the declaration — a truncated body
    /// fails the payload checksum rather than loading partially. Test hook,
    /// hidden from the public docs: the regression test passes a reader
    /// longer than its declared size and asserts the read stays bounded.
    #[doc(hidden)]
    pub fn load_from_reader<R: std::io::Read>(
        reader: R,
        size_limit: u64,
        path: &Path,
        expected: &EnumCacheHeader,
        max_resident_bytes: u64,
    ) -> Result<Self> {
        let mut bounded = reader.take(size_limit);
        let mut bytes = Vec::new();
        use std::io::Read as _;
        bounded.read_to_end(&mut bytes).map_err(|e| {
            Error::config(format!(
                "EnumCache::load: cannot read {}: {e}",
                path.display()
            ))
        })?;
        Ok(Self::parse(&bytes, path, Some(expected), max_resident_bytes)?.0)
    }

    /// Parse the binary form and report the load-time reservations (test
    /// hook, hidden from the public docs): like [`EnumCache::load`] without
    /// the expected-header check requirement (pass `None` for structural
    /// validation only), plus the [`ParseStats`] the parse actually
    /// reserved. A regression test asserts these reservations stay bounded
    /// by what the file can still hold. `max_resident_bytes` is enforced
    /// during parsing like [`EnumCache::load_with_budget`].
    #[doc(hidden)]
    pub fn parse_with_stats(
        bytes: &[u8],
        path: &Path,
        expected: Option<&EnumCacheHeader>,
        max_resident_bytes: u64,
    ) -> Result<(Self, ParseStats)> {
        Self::parse(bytes, path, expected, max_resident_bytes)
    }

    /// Parse the binary form, validating every length before slicing.
    ///
    /// With `expected`, the stored header is checked against it (naming the
    /// field on mismatch) before any entry is parsed. Without `expected`
    /// the header is only structurally validated. `max_resident_bytes` is
    /// enforced during parsing (task F9 item A4): when the running resident
    /// estimate exceeds it the parse fails with `Error::Config` naming the
    /// budget and the size, never a partially loaded cache.
    fn parse(
        bytes: &[u8],
        path: &Path,
        expected: Option<&EnumCacheHeader>,
        max_resident_bytes: u64,
    ) -> Result<(Self, ParseStats)> {
        let err = |what: &str| {
            Error::config(format!(
                "EnumCache::load: {} is truncated or corrupted ({what})",
                path.display()
            ))
        };
        // Magic and version first, so a foreign file or an older format is
        // refused by name before the checksum (which covers the version
        // word) is even consulted.
        if bytes.len() < CACHE_MAGIC.len() + 4 {
            return Err(err("unexpected end"));
        }
        if &bytes[..CACHE_MAGIC.len()] != CACHE_MAGIC {
            return Err(Error::config(format!(
                "EnumCache::load: {} has bad magic (not an enum cache)",
                path.display()
            )));
        }
        let version = u32::from_le_bytes(
            bytes[CACHE_MAGIC.len()..CACHE_MAGIC.len() + 4]
                .try_into()
                .map_err(|_| err("version"))?,
        );
        if version != ENUM_CACHE_FORMAT_VERSION {
            return Err(Error::config(format!(
                "EnumCache::load: {} has format version {version}, needs {}",
                path.display(),
                ENUM_CACHE_FORMAT_VERSION
            )));
        }
        // Split the trailing checksum and verify it over every preceding
        // byte (magic, version, header and all entries alike) before any
        // entry is parsed or used.
        if bytes.len() < CHECKSUM_BYTES {
            return Err(err("unexpected end"));
        }
        let (body, trailer) = bytes.split_at(bytes.len() - CHECKSUM_BYTES);
        let stored0 = u64::from_le_bytes(
            trailer[..8].try_into().map_err(|_| err("checksum"))?,
        );
        let stored1 = u64::from_le_bytes(
            trailer[8..].try_into().map_err(|_| err("checksum"))?,
        );
        let (chk0, chk1) = payload_checksum(body);
        if (stored0, stored1) != (chk0, chk1) {
            return Err(Error::config(format!(
                "EnumCache::load: {} fails the payload checksum (file is corrupted)",
                path.display()
            )));
        }
        // A checked slicing cursor over the checksum-verified body: every
        // length is validated before slicing, so corrupt lengths are
        // errors, never panics. (`take` advances, `remaining` reports what
        // the file can still hold, so reservations stay bounded by it.)
        struct Cursor<'b> {
            body: &'b [u8],
            pos: usize,
        }
        impl<'b> Cursor<'b> {
            fn take(&mut self, n: usize, err: &dyn Fn(&str) -> Error) -> Result<&'b [u8]> {
                let end = self.pos.checked_add(n).ok_or_else(|| err("length overflow"))?;
                if end > self.body.len() {
                    return Err(err("unexpected end"));
                }
                let out = &self.body[self.pos..end];
                self.pos = end;
                Ok(out)
            }
            fn remaining(&self) -> usize {
                self.body.len() - self.pos
            }
        }
        let mut cursor = Cursor {
            body,
            pos: CACHE_MAGIC.len() + 4,
        };
        let hlen_bytes = cursor.take(4, &err)?;
        let hlen =
            u32::from_le_bytes(hlen_bytes.try_into().map_err(|_| err("header length"))?) as usize;
        let hbytes = cursor.take(hlen, &err)?;
        let header_text =
            std::str::from_utf8(hbytes).map_err(|_| err("header is not UTF-8"))?;
        let header: EnumCacheHeader =
            serde_json::from_str(header_text).map_err(|_| err("header JSON does not parse"))?;
        // Header compatibility before any entry is parsed: a stale cache is
        // never trusted for even one entry (`check_compatible` names the
        // field on mismatch).
        if let Some(expected) = expected {
            header.check_compatible(expected)?;
        }
        let count_bytes = cursor.take(8, &err)?;
        let count =
            u64::from_le_bytes(count_bytes.try_into().map_err(|_| err("entry count"))?) as usize;
        if count > 10_000_000 {
            return Err(err("entry count is implausible"));
        }
        let mut map =
            HashMap::with_capacity(bounded_cap(count, cursor.remaining(), MIN_ENUM_ENTRY_BYTES));
        // Task F10 item B1: the load path tracks the same physical-bucket
        // high-water marks as the insert path — refreshed from the live
        // usable capacity after every reservation and insert (capacities
        // only grow while parsing, so the marks never go down), and the
        // running budget check prices them.
        let mut enum_buckets = required_buckets(map.capacity());
        let mut stats = ParseStats {
            enum_reservation: map.capacity(),
            evidence_reservation: 0,
            candidate_reservation: 0,
            evidence_row_reservation: 0,
        };
        let mut tracked: usize = 0;
        // Task F9 item A4: the running resident estimate (the same function
        // `resident_bytes()` reports) is enforced while parsing — never a
        // partially loaded cache.
        let budget_err = |running: usize| {
            Error::config(format!(
                "EnumCache::load: {} needs {running} resident bytes, over the {max_resident_bytes}-byte resident budget (--enum-cache-max-mb governs loads too): refusing the cached file instead of partially loading it",
                path.display()
            ))
        };
        if parse_resident(tracked, enum_buckets, 0) as u64 > max_resident_bytes {
            return Err(budget_err(parse_resident(tracked, enum_buckets, 0)));
        }
        for _ in 0..count {
            let mut key = [0u32; 8];
            for w in key.iter_mut() {
                let wb = cursor.take(4, &err)?;
                *w = u32::from_le_bytes(wb.try_into().map_err(|_| err("key"))?);
            }
            let mut counters = [0u32; COUNTER_WORDS];
            for w in counters.iter_mut() {
                let wb = cursor.take(4, &err)?;
                *w = u32::from_le_bytes(wb.try_into().map_err(|_| err("counters"))?);
            }
            let nb = cursor.take(4, &err)?;
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
            let mut cands =
                Vec::with_capacity(bounded_cap(n, cursor.remaining(), MIN_CAND_BYTES));
            for _ in 0..n {
                let hb = cursor.take(9, &err)?;
                let mut heavy = [0u8; 9];
                heavy.copy_from_slice(hb);
                let hhb = cursor.take(2, &err)?;
                let h = u16::from_le_bytes(hhb.try_into().map_err(|_| err("hydrogen"))?);
                let fb = cursor.take(1, &err)?;
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
            let cand_cap = cands.capacity();
            if map.insert(key, CacheEntry { counters, cands }).is_some() {
                return Err(Error::config(format!(
                    "EnumCache::load: {} stores a duplicate key",
                    path.display()
                )));
            }
            stats.candidate_reservation =
                stats.candidate_reservation.saturating_add(cand_cap);
            tracked = tracked.saturating_add(enum_entry_tracked(cand_cap));
            enum_buckets = enum_buckets.max(required_buckets(map.capacity()));
            if parse_resident(tracked, enum_buckets, 0) as u64 > max_resident_bytes {
                return Err(budget_err(parse_resident(tracked, enum_buckets, 0)));
            }
        }
        // Evidence section (task T6B): versions 1 to 3 are already refused
        // above by the version check naming the version.
        let ecount_bytes = cursor.take(8, &err)?;
        let ecount =
            u64::from_le_bytes(ecount_bytes.try_into().map_err(|_| err("evidence count"))?)
                as usize;
        if ecount > 10_000_000 {
            return Err(err("evidence entry count is implausible"));
        }
        let mut evidence: HashMap<EvidenceKey, Vec<EvidenceEntry>> =
            HashMap::with_capacity(bounded_cap(
                ecount,
                cursor.remaining(),
                MIN_EVIDENCE_ENTRY_BYTES,
            ));
        stats.evidence_reservation = evidence.capacity();
        let mut evidence_buckets = required_buckets(evidence.capacity());
        if parse_resident(tracked, enum_buckets, evidence_buckets) as u64 > max_resident_bytes
        {
            return Err(budget_err(parse_resident(
                tracked,
                enum_buckets,
                evidence_buckets,
            )));
        }
        for _ in 0..ecount {
            let mut meta = [0u32; 8];
            for w in meta.iter_mut() {
                let wb = cursor.take(4, &err)?;
                *w = u32::from_le_bytes(wb.try_into().map_err(|_| err("evidence key"))?);
            }
            let h0b = cursor.take(8, &err)?;
            let h0 = u64::from_le_bytes(h0b.try_into().map_err(|_| err("evidence hash"))?);
            let h1b = cursor.take(8, &err)?;
            let h1 = u64::from_le_bytes(h1b.try_into().map_err(|_| err("evidence hash"))?);
            let pcb = cursor.take(4, &err)?;
            let peak_count =
                u32::from_le_bytes(pcb.try_into().map_err(|_| err("evidence peak count"))?);
            let msb = cursor.take(8, &err)?;
            let mz_sum = u64::from_le_bytes(msb.try_into().map_err(|_| err("evidence m/z sum"))?);
            let nevb = cursor.take(1, &err)?;
            let n_ev = nevb[0];
            if n_ev > 32 {
                return Err(Error::config(format!(
                    "EnumCache::load: {} stores n_ev {n_ev} above 32",
                    path.display()
                )));
            }
            let rsb = cursor.take(4, &err)?;
            let rows =
                u32::from_le_bytes(rsb.try_into().map_err(|_| err("evidence rows"))?) as usize;
            if rows > 100_000 {
                return Err(err("evidence row count is implausible"));
            }
            let remaining = cursor.remaining();
            let mut explained =
                Vec::with_capacity(bounded_cap(rows, remaining, MIN_EVIDENCE_ROW_BYTES));
            let mut weight_bits =
                Vec::with_capacity(bounded_cap(rows, remaining, MIN_EVIDENCE_ROW_BYTES));
            for _ in 0..rows {
                let eb = cursor.take(1, &err)?;
                if eb[0] > 32 {
                    return Err(Error::config(format!(
                        "EnumCache::load: {} stores explained {} above 32",
                        path.display(),
                        eb[0]
                    )));
                }
                explained.push(eb[0]);
                let wb = cursor.take(4, &err)?;
                weight_bits.push(u32::from_le_bytes(
                    wb.try_into().map_err(|_| err("evidence weight"))?,
                ));
            }
            let nbytes = rows.div_ceil(8);
            let packed = cursor.take(nbytes, &err)?;
            // `rows <= 100_000` was enforced above, so this reservation is at
            // most 100 KiB.
            let mut complete = Vec::with_capacity(rows);
            for r in 0..rows {
                let bit = (packed[r / 8] >> (r % 8)) & 1;
                complete.push(bit);
            }
            let key = EvidenceKey {
                meta,
                h0,
                h1,
            };
            // Canonical inputs (task F8 item 3): length plus bytes ride at
            // the end of every v4 evidence record.
            let clenb = cursor.take(4, &err)?;
            let clen =
                u32::from_le_bytes(clenb.try_into().map_err(|_| err("evidence canonical length"))?)
                    as usize;
            if clen > 36 + 512 * 8 {
                return Err(err("evidence canonical length is implausible"));
            }
            let cbytes = cursor.take(clen, &err)?.to_vec();
            if !canonical_bytes_valid(&cbytes, peak_count) {
                return Err(Error::config(format!(
                    "EnumCache::load: {} stores malformed canonical evidence inputs",
                    path.display()
                )));
            }
            let entry = EvidenceEntry {
                n_ev,
                peak_count,
                mz_sum,
                explained,
                weight_bits,
                complete,
                canonical: cbytes,
            };
            stats.evidence_row_reservation = stats
                .evidence_row_reservation
                .saturating_add(entry.explained.capacity());
            tracked = tracked.saturating_add(evidence_entry_tracked(&entry));
            let bucket = evidence.entry(key).or_default();
            // An exact (key, canonical) duplicate is corruption; a colliding
            // key with different canonical bytes coexists in the bucket
            // (task F8 item 3).
            if bucket.iter().any(|e| e.canonical == entry.canonical) {
                return Err(Error::config(format!(
                    "EnumCache::load: {} stores a duplicate evidence key",
                    path.display()
                )));
            }
            let old_bucket_cap = bucket.capacity();
            bucket.push(entry);
            // Unused bucket capacity is part of the resident estimate (task F9
            // item A2), like on the insert path.
            tracked = tracked.saturating_add(
                bucket
                    .capacity()
                    .saturating_sub(old_bucket_cap)
                    .saturating_mul(EVIDENCE_BUCKET_SLOT_BYTES),
            );
            evidence_buckets = evidence_buckets.max(required_buckets(evidence.capacity()));
            if parse_resident(tracked, enum_buckets, evidence_buckets) as u64
                > max_resident_bytes
            {
                return Err(budget_err(parse_resident(
                    tracked,
                    enum_buckets,
                    evidence_buckets,
                )));
            }
        }
        if cursor.pos != body.len() {
            return Err(Error::config(format!(
                "EnumCache::load: {} has {} trailing bytes",
                path.display(),
                cursor.remaining()
            )));
        }
        // The load-time budget is the caller's (`load_with_budget`, task F9
        // item A4): parsing above already refused files whose running
        // resident estimate exceeds it, and the loaded cache carries the
        // budget for further inserts.
        Ok((
            Self {
                header,
                map,
                evidence,
                collisions: AtomicU64::new(0),
                budget_refusals: AtomicU64::new(0),
                enum_lookups: AtomicU64::new(0),
                enum_hits: AtomicU64::new(0),
                evidence_lookups: AtomicU64::new(0),
                evidence_hits: AtomicU64::new(0),
                tracked,
                enum_buckets,
                evidence_buckets,
                max_resident_bytes,
            },
            stats,
        ))
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
        // A budget refusal (`Ok(false)`) skips the row: it stays uncached
        // and its step runs the device path (task F8 item 4 — exactness
        // preserved, never an error).
        let _ = cache.insert(
            *key,
            counters,
            &cand_host[base..base + scored * CAND_WORDS],
            m,
        )?;
    }
    Ok(())
}

/// Run the PRODUCTION formula-evidence stage (`peak_select` →
/// `evidence_peaks` → `formula_evidence`) for one batch and insert every
/// spectrum not yet in the evidence map (task T6B).
///
/// This is the production evidence stage, not a copy: the same
/// [`peak_select`](crate::tensor::ops::ms2::peak_select),
/// [`evidence_peaks`] and [`formula_evidence`] the search stage calls, over
/// caller buffers of the same shapes. `cand_ev` and the valid flags of
/// `ev_peaks` come back in one batched read (this is a precompute pass;
/// reads are expected). Batches whose spectra are all cached already are
/// skipped without any launch or read.
///
/// `enum_keys` are the batch's 8-word enumeration meta rows (they fix the
/// candidate rows), `counters_host`/`cand_host` the enumeration output the
/// caller expanded from the cache (`cand [B, M, 13]` flat, `counters [B, 5]`
/// flat), `m` (`M`) the cache header's `window_m`, `work_max` the
/// `formula_evidence_work_max` the kernel runs with, `dispatch_max` the
/// evidence dispatch bound (results do not depend on it — the lane takes
/// the absolute lane index), `h_cap_max` the hydrogen-cap bound the kernel
/// clamps with, and `n_peaks` the model's kept-peak capacity (`N`).
///
/// Entries are inserted for every spectrum of the batch from its EXACT
/// uploaded row (see [`evidence_key_for_batch`]): under the
/// shuffled-spectrum control the uploaded rows carry donor peaks and the
/// content hash simply keys the donor-peak row (training hits are then
/// unlikely, which is correct behaviour, not an error).
#[allow(clippy::too_many_arguments)]
pub fn run_device_evidence_into<R: Runtime, E: FloatElem>(
    device: &Device<R>,
    batch: &SpectrumBatch,
    enum_keys: &[[u32; 8]],
    counters_host: &[u32],
    cand_host: &[u32],
    m: usize,
    work_max: u32,
    dispatch_max: u64,
    h_cap_max: u32,
    n_peaks: usize,
    cache: &mut EnumCache,
) -> Result<()> {
    let b = batch.len();
    if b == 0 {
        return Ok(());
    }
    if m as u64 != u64::from(cache.header().window_m) {
        return Err(Error::config(format!(
            "run_device_evidence_into: cand_ev width {m} != cache header window_m {}",
            cache.header().window_m
        )));
    }
    // Task F8 items 1–2, build enforcement: the evidence entries about to be
    // built belong to this `n_peaks` capacity and this `E` dtype. A mismatch
    // is `Error::Config` naming the field — never a silently mislabelled
    // entry.
    if cache.header().n_peaks as usize != n_peaks {
        return Err(Error::config(format!(
            "run_device_evidence_into: header mismatch on field `n_peaks`: cache {} != expected {n_peaks}",
            cache.header().n_peaks
        )));
    }
    if cache.header().evidence_dtype.as_str() != E::DTYPE.name() {
        return Err(Error::config(format!(
            "run_device_evidence_into: header mismatch on field `evidence_dtype`: cache {:?} != expected {}",
            cache.header().evidence_dtype,
            E::DTYPE.name()
        )));
    }
    let n_peaks_u32 = u32::try_from(n_peaks).map_err(|_| {
        Error::config(format!(
            "run_device_evidence_into: n_peaks {n_peaks} exceeds u32"
        ))
    })?;
    if enum_keys.len() != b {
        return Err(Error::config(format!(
            "run_device_evidence_into: {} enum keys for {b} spectra",
            enum_keys.len()
        )));
    }
    // Uploaded rows (validates the batch; the error propagates).
    let spectra = DeviceSpectra::upload(batch, device)?;
    // Evidence queries from the EXACT uploaded rows (borrowed canonical
    // inputs: no per-spectrum buffering, task F8 item 5).
    let mut queries = Vec::with_capacity(b);
    for i in 0..b {
        let upc = spectra.peak_count.get(i).copied().unwrap_or(0);
        let (key, _) =
            evidence_key_for_batch(batch, i, upc, enum_keys[i], work_max, h_cap_max);
        queries.push(EvidenceQuery {
            key,
            rows_scored: counters_host[i * COUNTER_WORDS + 2] as usize,
            inputs: evidence_inputs_for_batch(batch, i, upc, work_max, h_cap_max),
        });
    }
    if cache
        .expand_evidence_batch_core(&queries, m, n_peaks_u32, E::DTYPE.name())
        .is_some()
    {
        return Ok(());
    }
    // Production evidence stage over caller buffers of the production
    // shapes (`kept [B, N_raw→N]`, `ev_peaks [B, P, 4]`, `cand_ev [B, M,
    // 4]` with `P = EVIDENCE_PEAKS`).
    let n_raw = batch.n_raw as usize;
    let peaks = PeakBuffers::new(b, n_raw, n_peaks, device);
    peak_select(
        &spectra.mz,
        &spectra.intensity,
        &spectra.meta,
        spectra.intensity_scale,
        &peaks,
    )?;
    let cand_t = IdTensor::from_slice(cand_host, vec![b, m, CAND_WORDS], device)?;
    let spec_t = spectra.evidence_spec(device)?;
    let mut ev_peaks_t = IdTensor::empty(vec![b, EVIDENCE_PEAKS, 4], device);
    let mut ev_w_t =
        crate::tensor::Tensor::<R, E>::empty(vec![b, EVIDENCE_PEAKS], device);
    evidence_peaks(
        &peaks.kept,
        &peaks.kept_f,
        &spectra.meta,
        &spec_t,
        &mut ev_peaks_t,
        &mut ev_w_t,
    )?;
    let mut cand_ev_t =
        crate::tensor::Tensor::<R, E>::empty(vec![b, m, 4], device);
    formula_evidence(
        &cand_t,
        &ev_peaks_t,
        &ev_w_t,
        &spectra.meta,
        &spec_t,
        &mut cand_ev_t,
        work_max,
        dispatch_max,
        h_cap_max,
        spectra.uploaded_tol_max(),
    )?;
    // One batched read per batch: the valid flags of `ev_peaks` (for the
    // per-spectrum `n_ev`, including zero-scored spectra whose `cand_ev`
    // rows are all padding) and the candidate evidence.
    let (ids, floats) = read_all::<R, E>(&[&ev_peaks_t], &[&cand_ev_t])?;
    let (ev_ids, ev) = (&ids[0], &floats[0]);
    let stride = EVIDENCE_PEAKS * 4;
    for i in 0..b {
        // Verified: a colliding key with different canonical inputs does not
        // count as present (task F8 item 3).
        if cache.get_evidence(&queries[i].key, &queries[i].inputs).is_some() {
            continue;
        }
        let scored = queries[i].rows_scored;
        let mut n_ev = 0u8;
        for s in 0..EVIDENCE_PEAKS {
            if ev_ids[i * stride + s * 4 + 3] == 1 {
                n_ev += 1;
            }
        }
        let mut explained = Vec::with_capacity(scored);
        let mut weight_bits = Vec::with_capacity(scored);
        let mut complete = Vec::with_capacity(scored);
        for r in 0..scored {
            let base = (i * m + r) * 4;
            // Device values are exact integers `0..=32` (explained, `n_ev`)
            // and `0.0`/`1.0` (complete); the cast is exact there.
            explained.push(ev[base] as u8);
            weight_bits.push(ev[base + 1].to_bits());
            complete.push(ev[base + 3] as u8);
        }
        // A budget refusal stays uncached (counted inside; task F8 item 4 —
        // the step runs the device path for the row, exactness preserved).
        cache.insert_evidence(
            queries[i].key,
            queries[i].inputs.peak_count,
            mz_sum_of(queries[i].inputs.mz_row),
            n_ev,
            explained,
            weight_bits,
            complete,
            m,
            evidence_canonical_bytes(&queries[i].inputs),
        )?;
    }
    Ok(())
}
