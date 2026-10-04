//! Ranking, compaction and the packed output (`docs/MS2_V1_ARCHITECTURE.md`
//! §4.4; plan items P6.4 and the host side of P6.5).
//!
//! Pure host Rust: no tensors, no kernels, no floats beyond the candidate
//! score. [`pack`] is implemented by calling the kernel twins [`rank_lane`],
//! [`record_pack_lane`], [`pack_lane`] and [`returned_count_lane`], so the
//! host function and the kernels in [`crate::tensor::ops::ms2_pack`] share one
//! definition: the `#[cube]` kernels copy the lanes line for line.
//!
//! Validated score domain (by specification, not by code): a ranking term or
//! score is "in the validated domain" when it lies strictly inside (−3e38,
//! 3e38); anything else (NaN, infinities, finite extremes at or beyond it,
//! overflowing sums) is treated as invalid and excluded from ranking.
//! Finiteness tests are not portable across shader backends, so the kernels
//! classify with a range test (NaN self-comparison is unreliable under
//! fast-math) and [`PackedCandidateBatch::validate`] uses the same range
//! rule, never exact IEEE classification. The bound value lives in exactly
//! one place ([`crate::tensor::ops::ms2::FINITE_MAX`]); [`SCORE_FINITE_MAX`]
//! is an alias of it.
//!
//! Checked u32 addressing (finding E2): every stride, count and largest
//! accessed address of every input/output buffer is checked against the u32
//! domain ([`check_u32_len`], [`check_u32_product]) before any launch or host
//! lane call, in the kernel wrappers and in the host [`pack`] path. An
//! unsupported size is [`Error::Shape`], never a narrowed `as u32`.
//!
//! Kernel-expressible form (CubeCL 0.10 cannot express Rust fixed-size local
//! arrays `[u32; N]` nor `wrapping_*` methods, so the lanes use neither):
//! every value is `u32` (or `f32` in the fixed operation order
//! `trace_log_prob + formula_log_prob`) with plain arithmetic, `usize` appears
//! only to bound a slice or to index it, loops are `while` loops over `u32`
//! counters, scalar capacities and every buffer offset are passed explicitly
//! as `u32`, full bound buffers are addressed with explicit record indices,
//! strides and base offsets (no record-local slices), every lane has a single
//! exit and nesting stays at most 5 deep. Loop-carried variables start from
//! literals or buffer loads, never from a scalar argument.
//!
//! Record layout. One trajectory's integer fields gather into `W` words and
//! its float fields into [`WF`] words. The fixed header ([`W`] words, offsets
//! [`O_SPECTRUM`]..[`O_ATTACHMENT`]) comes first, then the `steps * 4` token
//! words at [`O_TOKENS`], then the `atoms` open-valence words at
//! [`valence_offset`]. The full integer width is [`record_width`]; it is a
//! function of `(steps, atoms)`, not a constant, because the token and valence
//! tails vary with the request shape. What the device kernels bind:
//!
//! * `actions`: `[rows, steps * 4 + atoms + 4]` device trajectory records
//!   (tokens, open valence, length, status, trace-log-probability bits,
//!   formula row) — the layout [`crate::tensor::ops::ms2::sample_record_width`]
//!   describes and [`Ms2Model::generate`] reads.
//! * `traj_formula`: `[B, K, 12]` (retained formula slot, source id, 10
//!   counts) — the buffer the allocation kernel writes. `ms2_rank` and
//!   `ms2_scores_fill` read it directly (the formula log-probability is
//!   gathered by the retained slot); `ms2_record_pack` binds the translated
//!   `traj_window` instead, whose word 0 is the window slot (the formula
//!   rank the packed record carries), written by `ms2_allocate_window`.
//! * `scores`: `[rows, 2]` floats `(trace_log_prob, formula_log_prob)` per
//!   trajectory, filled on the device by `ms2_scores_fill`: the trace
//!   log-probability from the `actions` record bits, the formula
//!   log-probability by indexing `top_log_prob [B, F]` with the trajectory's
//!   retained slot (`u32::MAX` when there is none).
//! * `evidence`: `[rows, 18]` words per trajectory; word 0 is the evidence
//!   status (read by `ms2_record_pack` into the record header), the full row
//!   rides [`pack_evidence_lane`] into the packed evidence buffers.
//! * `identity`: `[rows, 2]` (status bits to OR in, resolution) as
//!   `ms2_graph_identity` writes it.
//!
//! Per-spectrum fields other than `returned_count` (`request_status`, the
//! counters, `formula_source`, …) never touch the device: the host copies them
//! from the [`CandidateBatch`], which already holds them.
//!
//! [`Ms2Model::generate`]: crate::models::ms2::generate::Ms2Model::generate
//! [`Ms2Model::generate_readout`]: crate::models::ms2::generate::Ms2Model::generate_readout

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

use super::contract::{
    CandidateBatch, NO_FORMULA, SCHEMA_VERSION, candidate_status, request_status,
};
use super::grammar::{Limits, Token, replay};
use super::identity::{DUPLICATE_GRAPH, IDENTITY_UNRESOLVED};

// ---------------------------------------------------------------------------
// Record layout constants
// ---------------------------------------------------------------------------

/// Integer header words before the token tail (offsets [`O_SPECTRUM`] through
/// [`O_ATTACHMENT`]).
pub const W: usize = 19;
/// Float record words per trajectory: formula log-probability, trace
/// log-probability, score.
pub const WF: usize = 3;
/// Offset of the spectrum index in a record.
pub const O_SPECTRUM: u32 = 0;
/// Offset of the original trajectory index (`u32::MAX` when unfilled).
pub const O_TRAJECTORY: u32 = 1;
/// Offset of the emitted length.
pub const O_LENGTH: u32 = 2;
/// Offset of the conditioning formula row (`u32::MAX` when there is none).
pub const O_FORMULA_ROW: u32 = 3;
/// Offset of the formula rank in the scored support (`u32::MAX` when none).
pub const O_FORMULA_RANK: u32 = 4;
/// Offset of the 10 formula counts (10 words).
pub const O_COUNTS: u32 = 5;
/// Offset of the candidate status (with the identity bits ORed in).
pub const O_STATUS: u32 = 15;
/// Offset of the evidence status (word 0 of the `evidence` row).
pub const O_EVIDENCE: u32 = 16;
/// Offset of the identity resolution (0 trace only, 1 exact, 2 unresolved).
pub const O_RESOLUTION: u32 = 17;
/// Offset of the attachment partition (0 unknown, the only current value).
pub const O_ATTACHMENT: u32 = 18;
/// Offset of the token tail (`steps * 4` words of `(kind, type, bond,
/// pointer)`).
pub const O_TOKENS: u32 = 19;
/// Offset of the formula log-probability in a float record.
pub const OF_FORMULA_LP: u32 = 0;
/// Offset of the trace log-probability in a float record.
pub const OF_TRACE_LP: u32 = 1;
/// Offset of the ranking score in a float record.
pub const OF_SCORE: u32 = 2;
/// Words per `evidence` row; only word 0 is read here.
pub const EVIDENCE_STRIDE: u32 = 18;
/// Words per `traj_formula` row: slot, source id, 10 counts.
pub const TRAJ_FORMULA_STRIDE: u32 = 12;
/// Finite-range bound of the ranking score: the validated score domain is
/// (−3e38, 3e38); a ranking term or score is "in the validated domain" when
/// it lies strictly inside that interval, and anything else (NaN,
/// infinities, finite extremes at or beyond it, overflowing sums) is treated
/// as invalid and excluded from ranking. This rule is by specification, not
/// by code: finiteness tests are not portable across shader backends, so the
/// kernels classify with a range test (NaN self-comparison is unreliable
/// under fast-math) and [`PackedCandidateBatch::validate`] uses the same
/// range rule, never exact IEEE classification. Log-probabilities are finite
/// by contract, so this only bites on caller-supplied reranker values and on
/// corrupted input.
///
/// The bound value itself lives in exactly one place,
/// [`crate::tensor::ops::ms2::FINITE_MAX`]; this name is an alias of it, kept
/// so existing imports keep working.
pub const SCORE_FINITE_MAX: f32 = crate::tensor::ops::ms2::FINITE_MAX;

/// Full integer record width for these caps: the [`W`] header words, the
/// `steps * 4` token words, then the `atoms` open-valence words.
pub fn record_width(steps: usize, atoms: usize) -> usize {
    W.saturating_add(steps.saturating_mul(4)).saturating_add(atoms)
}

/// Offset of the open-valence tail (`atoms` words) for this step count.
pub fn valence_offset(steps: usize) -> usize {
    W.saturating_add(steps.saturating_mul(4))
}

// ---------------------------------------------------------------------------
// Checked u32 address domains (finding E2) and the score-domain rule (E3)
// ---------------------------------------------------------------------------

/// Whether a score or log-probability lies in the validated score domain:
/// strictly inside (−[`SCORE_FINITE_MAX`], [`SCORE_FINITE_MAX`]) — a range
/// test, never a NaN self-comparison, so NaN, ±infinity, finite extremes at
/// or beyond the bound and overflowing sums all fail.
/// [`PackedCandidateBatch::validate`] and the ranking lanes share this rule.
pub fn score_in_domain(value: f32) -> bool {
    value > -SCORE_FINITE_MAX && value < SCORE_FINITE_MAX
}

/// Check one element count or stride against the u32 address domain.
///
/// Returns the value narrowed to `u32`, or [`Error::Shape`] when it exceeds
/// `u32::MAX`. Takes a shape, never a buffer, so oversized shapes are
/// unit-testable without allocating.
pub fn check_u32_len(field: &str, value: usize) -> Result<u32> {
    if (value as u64) > u32::MAX as u64 {
        return Err(Error::shape(format!(
            "pack: {field} length {value} exceeds the u32 address domain (u32::MAX)"
        )));
    }
    Ok(value as u32)
}

/// Check one `a * b` element domain (a record count, a buffer length, the
/// largest accessed address of a strided buffer) against the u32 address
/// domain: overflow of `usize` or a product above `u32::MAX` is
/// [`Error::Shape`]. Returns the product for the caller to reuse, so no
/// second multiplication can wrap.
pub fn check_u32_product(field: &str, a: usize, b: usize) -> Result<usize> {
    let product = a.checked_mul(b).ok_or_else(|| {
        Error::shape(format!(
            "pack: {field} address product {a} * {b} overflows usize"
        ))
    })?;
    if (product as u64) > u32::MAX as u64 {
        return Err(Error::shape(format!(
            "pack: {field} address product {a} * {b} = {product} exceeds the u32 address domain (u32::MAX)"
        )));
    }
    Ok(product)
}

// ---------------------------------------------------------------------------
// Score selection
// ---------------------------------------------------------------------------

/// Which score orders the candidates of [`pack`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ScoreKind<'a> {
    /// `formula_log_prob + trace_log_prob` (architecture §3.2), in that
    /// operation order everywhere (host lanes and kernels alike).
    Raw,
    /// A caller-supplied per-trajectory score (the reranker logit of §4.3,
    /// once P6.3 builds it); one entry per `B * K` record in record order.
    Reranker(&'a [f32]),
}

// ---------------------------------------------------------------------------
// Packed output schema
// ---------------------------------------------------------------------------

/// Ranked, compacted candidates of a batch (architecture §4.4): `B * R`
/// records in `(spectrum, rank)` order, where `R` is the configured
/// `returned` count. A separate type on purpose: [`CandidateBatch`] keeps its
/// `B * K` trajectory-ordered contract.
///
/// Per record (`[B * R]` unless noted): `spectrum_id`, the original
/// `trajectory` (`u32::MAX` for an unfilled slot), `actions [.., T, 4]`,
/// `length`, `formula_row`, `formula_rank`, `formula_counts [.., 10]`,
/// `formula_log_prob`, `trace_log_prob`, the ranking `score`, `open_valence
/// [.., A]`, `status`, `evidence_status`, `identity_resolution`,
/// `attachment_partition`. Per spectrum (`[B]`): `returned_count`,
/// `request_status`, `formula_source`, and the other per-spectrum fields of
/// [`CandidateBatch`].
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct PackedCandidateBatch {
    /// Schema version; only [`SCHEMA_VERSION`] (2) is accepted.
    pub schema_version: u32,
    /// Spectra per batch.
    pub batch: usize,
    /// Slots per spectrum (`R`, the `returned` of [`pack`]).
    pub returned: usize,
    /// Trajectories per spectrum of the source batch (`K`).
    pub trajectories: usize,
    /// Maximum trace steps.
    pub max_steps: usize,
    /// Maximum atoms per candidate (open-valence width).
    pub max_atoms: usize,
    /// Maximum ring closures per candidate (grammar limit).
    pub max_ring_closures: usize,
    /// Provenance per slot.
    pub spectrum_id: Vec<u64>,
    /// Original trajectory per slot (`u32::MAX` for an unfilled slot).
    pub trajectory: Vec<u32>,
    /// `(kind, atom_type, bond_order, pointer)` per step; steps at or after
    /// `length` are PAD (all zero).
    pub actions: Vec<u32>,
    /// Tokens emitted per slot.
    pub length: Vec<u32>,
    /// Conditioning formula row; [`NO_FORMULA`] when there is none.
    pub formula_row: Vec<u32>,
    /// Rank of the formula in the scored support; [`NO_FORMULA`] when none.
    pub formula_rank: Vec<u32>,
    /// Composition of the conditioning formula in `ELEMENTS` order.
    pub formula_counts: Vec<u16>,
    /// `log p(formula | spectrum)`.
    pub formula_log_prob: Vec<f32>,
    /// Summed action log-probabilities up to `length`.
    pub trace_log_prob: Vec<f32>,
    /// The ranking score: the raw sum, or the caller-supplied reranker score.
    pub score: Vec<f32>,
    /// Residual valence per atom; meaningful only for a filled slot.
    pub open_valence: Vec<u8>,
    /// Candidate bits of [`candidate_status`]; `0` for an unfilled slot.
    pub status: Vec<u32>,
    /// Evidence status per slot.
    pub evidence_status: Vec<u8>,
    /// Evidence record count per slot (`u8 [B*R]`, at most 4).
    #[serde(default)]
    pub evidence_count: Vec<u8>,
    /// Original `peak_id` per evidence record (`u32 [B*R, E]`).
    #[serde(default)]
    pub evidence_peak_id: Vec<u32>,
    /// Hypothesis index among the kept `J` per record (`u8 [B*R, E]`).
    #[serde(default)]
    pub evidence_hypothesis: Vec<u8>,
    /// Hydrogen shift `s` per record (`i8 [B*R, E]`).
    #[serde(default)]
    pub evidence_shift: Vec<i8>,
    /// Signed residual in integer mass units per record (`i32 [B*R, E]`).
    #[serde(default)]
    pub evidence_residual: Vec<i32>,
    /// Assignment log-probability per record (`f32 [B*R, E]`).
    #[serde(default)]
    pub evidence_log_prob: Vec<f32>,
    /// Identity resolution per slot (0 trace only, 1 exact, 2 unresolved).
    pub identity_resolution: Vec<u8>,
    /// Attachment partition per slot (`0` unknown).
    pub attachment_partition: Vec<u8>,
    /// Filled slots per spectrum (`min(R, eligible candidates)`; `0` for a
    /// failed request).
    pub returned_count: Vec<u32>,
    /// Request bits of [`request_status`], per spectrum.
    pub request_status: Vec<u32>,
    /// Table rows compared against the mass window, per spectrum.
    pub rows_visited: Vec<u32>,
    /// Rows inside the window, per spectrum.
    pub rows_joined: Vec<u32>,
    /// Rows given a neural score, per spectrum.
    pub rows_scored: Vec<u32>,
    /// `1` when every joined row was scored, else `0`.
    pub formula_support_complete: Vec<u8>,
    /// Probability mass of the retained formulas within the scored window.
    pub formula_mass_retained: Vec<f32>,
    /// Peaks kept after device selection, per spectrum.
    pub peaks_kept: Vec<u32>,
    /// Fraction of filtered intensity the kept peaks hold, per spectrum.
    pub intensity_retained: Vec<f32>,
    /// Formula source per spectrum (`0` table, `1` enumeration).
    pub formula_source: Vec<u8>,
}

impl PackedCandidateBatch {
    /// Check every invariant of architecture §4.4:
    ///
    /// 1. shapes: per-record fields hold `B * R` entries (`actions`
    ///    `B * R * T * 4`, `formula_counts` `B * R * 10`, `open_valence`
    ///    `B * R * A`), per-spectrum fields hold `B`;
    /// 2. `1 <= R <= K` and `returned_count <= R` on every spectrum;
    /// 3. rank order: every filled slot's score is independently in the
    ///    validated score domain (−3e38, 3e38) — NaN, infinities and extremes
    ///    are rejected on their own, not merely by comparison with a
    ///    neighbour — and the filled slots of a spectrum hold non-increasing
    ///    scores, ties by smaller trajectory; exactly the first
    ///    `returned_count` slots are filled;
    /// 4. an unfilled slot has trajectory, formula row and formula rank
    ///    `u32::MAX`, status `0` and zero payload (length, log-probabilities,
    ///    score, tokens, counts, valence, evidence, resolution, attachment);
    /// 5. a filled slot carries the source/row/rank/count relationships
    ///    [`CandidateBatch::validate`] enforces (enumeration source: formula
    ///    row `u32::MAX`; counts all zero exactly when the rank is
    ///    `u32::MAX`, and then the row is `u32::MAX` as well; a real rank
    ///    below that spectrum's `rows_scored`; a table-source record with a
    ///    formula names a real table row) AND, beyond that, real formula
    ///    provenance: a filled slot is never formula-less (rank `u32::MAX`
    ///    with zero counts is rejected even when self-consistent, because
    ///    no trajectory starts without a formula), an `identity_resolution` in
    ///    `0..=2`, an original trajectory below `trajectories` and unique
    ///    within its spectrum, finite log-probabilities at most `1e-4` inside
    ///    the validated score domain, and a trace that replays legally with
    ///    the grammar [`replay`] — the same function
    ///    [`CandidateBatch::validate`] uses — under the formula's composition
    ///    budget (`Some` composition from `formula_counts` when the slot has
    ///    a formula, `None` when it has none); the record is `finished`
    ///    (without `truncated`, `invalid_final`, `duplicate_trace`,
    ///    `duplicate_graph` or `request_failed`);
    /// 6. a failed request (a fatal [`request_status`] bit) has
    ///    `returned_count == 0` and no filled slot.
    ///
    /// What validation cannot show: it checks the packed records in front of
    /// it — order, provenance, replay and the score domain — but it cannot
    /// show that omitted candidates had lower scores, because the original
    /// candidate set is gone. A batch that drops its best candidate and packs
    /// the rest in order still validates.
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != SCHEMA_VERSION {
            return Err(Error::config(format!(
                "PackedCandidateBatch::validate: unknown schema_version {} (expected {SCHEMA_VERSION})",
                self.schema_version
            )));
        }
        if self.trajectories == 0 {
            return Err(Error::config(
                "PackedCandidateBatch::validate: trajectories is 0: slots need at least one trajectory per spectrum"
                    .to_string(),
            ));
        }
        if self.returned == 0 || self.returned > self.trajectories {
            return Err(Error::config(format!(
                "PackedCandidateBatch::validate: returned {} is not in 1..=trajectories {}",
                self.returned, self.trajectories
            )));
        }
        let n = self.batch.checked_mul(self.returned).ok_or_else(|| {
            Error::config(format!(
                "PackedCandidateBatch::validate: batch {} times returned {} overflows usize",
                self.batch, self.returned
            ))
        })?;
        let per_record: [(&str, usize); 13] = [
            ("spectrum_id", self.spectrum_id.len()),
            ("trajectory", self.trajectory.len()),
            ("length", self.length.len()),
            ("formula_row", self.formula_row.len()),
            ("formula_rank", self.formula_rank.len()),
            ("formula_log_prob", self.formula_log_prob.len()),
            ("trace_log_prob", self.trace_log_prob.len()),
            ("score", self.score.len()),
            ("status", self.status.len()),
            ("evidence_status", self.evidence_status.len()),
            ("evidence_count", self.evidence_count.len()),
            ("identity_resolution", self.identity_resolution.len()),
            ("attachment_partition", self.attachment_partition.len()),
        ];
        for (field, len) in per_record {
            if len != n {
                return Err(Error::config(format!(
                    "PackedCandidateBatch::validate: field {field} has length {len} for {n} slots"
                )));
            }
        }
        if self.actions.len() != n * self.max_steps * 4 {
            return Err(Error::config(format!(
                "PackedCandidateBatch::validate: field actions has length {} for {n} slots of {} steps",
                self.actions.len(),
                self.max_steps
            )));
        }
        if self.formula_counts.len() != n * 10 {
            return Err(Error::config(format!(
                "PackedCandidateBatch::validate: field formula_counts has length {} for {n} slots of 10 counts",
                self.formula_counts.len()
            )));
        }
        if self.open_valence.len() != n * self.max_atoms {
            return Err(Error::config(format!(
                "PackedCandidateBatch::validate: field open_valence has length {} for {n} slots of {} atoms",
                self.open_valence.len(),
                self.max_atoms
            )));
        }
        for (field, len) in [
            ("evidence_peak_id", self.evidence_peak_id.len()),
            ("evidence_hypothesis", self.evidence_hypothesis.len()),
            ("evidence_shift", self.evidence_shift.len()),
            ("evidence_residual", self.evidence_residual.len()),
            ("evidence_log_prob", self.evidence_log_prob.len()),
        ] {
            if len != n * crate::models::ms2::contract::EVIDENCE_CAP {
                return Err(Error::config(format!(
                    "PackedCandidateBatch::validate: field {field} has length {len} for {n} slots of {} evidence slots",
                    crate::models::ms2::contract::EVIDENCE_CAP
                )));
            }
        }
        let per_spectrum: [(&str, usize); 10] = [
            ("returned_count", self.returned_count.len()),
            ("request_status", self.request_status.len()),
            ("rows_visited", self.rows_visited.len()),
            ("rows_joined", self.rows_joined.len()),
            ("rows_scored", self.rows_scored.len()),
            (
                "formula_support_complete",
                self.formula_support_complete.len(),
            ),
            ("formula_mass_retained", self.formula_mass_retained.len()),
            ("peaks_kept", self.peaks_kept.len()),
            ("intensity_retained", self.intensity_retained.len()),
            ("formula_source", self.formula_source.len()),
        ];
        for (field, len) in per_spectrum {
            if len != self.batch {
                return Err(Error::config(format!(
                    "PackedCandidateBatch::validate: field {field} has length {len} for {} spectra",
                    self.batch
                )));
            }
        }
        let limits =
            Limits::new(self.max_atoms, self.max_ring_closures).map_err(|e| {
                Error::config(format!(
                    "PackedCandidateBatch::validate: replay limits from max_atoms {} and max_ring_closures {} rejected: {e}",
                    self.max_atoms, self.max_ring_closures
                ))
            })?;
        for b in 0..self.batch {
            if self.returned_count[b] > self.returned as u32 {
                return Err(Error::config(format!(
                    "PackedCandidateBatch::validate: spectrum {b} returned_count {} exceeds returned {}",
                    self.returned_count[b], self.returned
                )));
            }
            if self.formula_source[b] > 1 {
                return Err(Error::config(format!(
                    "PackedCandidateBatch::validate: spectrum {b} formula_source {} is not 0 (table) or 1 (enumeration)",
                    self.formula_source[b]
                )));
            }
            // Rule 6: a failed request returns nothing.
            let fatal = self.request_status[b] & request_status::FATAL_MASK != 0;
            if fatal && self.returned_count[b] != 0 {
                return Err(Error::config(format!(
                    "PackedCandidateBatch::validate: spectrum {b} has fatal request status {} but returned_count {} (expected 0)",
                    self.request_status[b], self.returned_count[b]
                )));
            }
            let first = self.spectrum_id[b * self.returned];
            for r in 0..self.returned {
                let s = b * self.returned + r;
                if self.spectrum_id[s] != first {
                    return Err(Error::config(format!(
                        "PackedCandidateBatch::validate: spectrum {b} slot {r} breaks (spectrum, rank) order"
                    )));
                }
                let filled = r < self.returned_count[b] as usize;
                let traj_none = self.trajectory[s] == NO_FORMULA;
                if filled == traj_none {
                    return Err(Error::config(format!(
                        "PackedCandidateBatch::validate: spectrum {b} slot {r} filled {filled} disagrees with trajectory {} (filled exactly when trajectory is not u32::MAX)",
                        self.trajectory[s]
                    )));
                }
                if !filled {
                    // Rule 4: the unfilled pattern.
                    if self.formula_row[s] != NO_FORMULA {
                        return Err(Error::config(format!(
                            "PackedCandidateBatch::validate: spectrum {b} slot {r} is unfilled but formula_row is {} (expected u32::MAX)",
                            self.formula_row[s]
                        )));
                    }
                    if self.formula_rank[s] != NO_FORMULA {
                        return Err(Error::config(format!(
                            "PackedCandidateBatch::validate: spectrum {b} slot {r} is unfilled but formula_rank is {} (expected u32::MAX)",
                            self.formula_rank[s]
                        )));
                    }
                    if self.status[s] != 0 {
                        return Err(Error::config(format!(
                            "PackedCandidateBatch::validate: spectrum {b} slot {r} is unfilled but status is {} (expected 0)",
                            self.status[s]
                        )));
                    }
                    if self.length[s] != 0
                        || self.formula_log_prob[s] != 0.0
                        || self.trace_log_prob[s] != 0.0
                        || self.score[s] != 0.0
                        || self.evidence_status[s] != 0
                        || self.evidence_count[s] != 0
                        || self.identity_resolution[s] != 0
                        || self.attachment_partition[s] != 0
                    {
                        return Err(Error::config(format!(
                            "PackedCandidateBatch::validate: spectrum {b} slot {r} is unfilled but carries a non-zero payload"
                        )));
                    }
                    {
                        let ecap = crate::models::ms2::contract::EVIDENCE_CAP;
                        for q in 0..ecap {
                            if self.evidence_peak_id[s * ecap + q] != 0
                                || self.evidence_hypothesis[s * ecap + q] != 0
                                || self.evidence_shift[s * ecap + q] != 0
                                || self.evidence_residual[s * ecap + q] != 0
                                || self.evidence_log_prob[s * ecap + q] != 0.0
                            {
                                return Err(Error::config(format!(
                                    "PackedCandidateBatch::validate: spectrum {b} slot {r} is unfilled but carries non-zero evidence record {q}"
                                )));
                            }
                        }
                    }
                    let abase = s * self.max_steps * 4;
                    if self.actions[abase..abase + self.max_steps * 4]
                        .iter()
                        .any(|&v| v != 0)
                    {
                        return Err(Error::config(format!(
                            "PackedCandidateBatch::validate: spectrum {b} slot {r} is unfilled but carries non-zero tokens"
                        )));
                    }
                    let cbase = s * 10;
                    if self.formula_counts[cbase..cbase + 10].iter().any(|&v| v != 0) {
                        return Err(Error::config(format!(
                            "PackedCandidateBatch::validate: spectrum {b} slot {r} is unfilled but carries non-zero formula counts"
                        )));
                    }
                    let vbase = s * self.max_atoms;
                    if self.open_valence[vbase..vbase + self.max_atoms]
                        .iter()
                        .any(|&v| v != 0)
                    {
                        return Err(Error::config(format!(
                            "PackedCandidateBatch::validate: spectrum {b} slot {r} is unfilled but carries non-zero open valence"
                        )));
                    }
                } else {
                    if fatal {
                        return Err(Error::config(format!(
                            "PackedCandidateBatch::validate: spectrum {b} slot {r} is filled but the request is fatal"
                        )));
                    }
                    if self.trajectory[s] >= self.trajectories as u32 {
                        return Err(Error::config(format!(
                            "PackedCandidateBatch::validate: spectrum {b} slot {r} trajectory {} is not below trajectories {}",
                            self.trajectory[s], self.trajectories
                        )));
                    }
                    // Rule 3: every filled score is independently in the
                    // validated domain (−3e38, 3e38); NaN, infinities and
                    // out-of-range extremes are rejected on their own, by a
                    // range test and never by NaN self-comparison.
                    let here_score = self.score[s];
                    if !score_in_domain(here_score) {
                        return Err(Error::config(format!(
                            "PackedCandidateBatch::validate: spectrum {b} slot {r} score {here_score} is outside the validated score domain (-3e38, 3e38); values outside it, NaN and infinities are ineligible"
                        )));
                    }
                    // Rank order: non-increasing scores, ties by smaller
                    // trajectory. Both scores are in-domain here (no NaN),
                    // so the comparisons decide.
                    if r > 0 {
                        let p = s - 1;
                        let there = self.score[p];
                        let ordered = here_score < there
                            || (here_score == there
                                && self.trajectory[p] < self.trajectory[s]);
                        if !ordered {
                            return Err(Error::config(format!(
                                "PackedCandidateBatch::validate: spectrum {b} slots {} and {r} break rank order (scores {there} then {here_score}, trajectories {} then {})",
                                r - 1,
                                self.trajectory[p],
                                self.trajectory[s]
                            )));
                        }
                    }
                    // Rule 5a: the source/row/rank/count relationships
                    // `CandidateBatch::validate` enforces.
                    let src = self.formula_source[b];
                    if src == 1 && self.formula_row[s] != NO_FORMULA {
                        return Err(Error::config(format!(
                            "PackedCandidateBatch::validate: spectrum {b} slot {r} has formula_source 1 (enumeration) but formula_row {} (expected u32::MAX)",
                            self.formula_row[s]
                        )));
                    }
                    let cbase = s * 10;
                    let all_zero = self.formula_counts[cbase..cbase + 10]
                        .iter()
                        .all(|&c| c == 0);
                    let rank_none = self.formula_rank[s] == NO_FORMULA;
                    if all_zero != rank_none {
                        return Err(Error::config(format!(
                            "PackedCandidateBatch::validate: spectrum {b} slot {r} counts all-zero {all_zero} disagrees with formula_rank {} (all zero exactly when rank is u32::MAX)",
                            self.formula_rank[s]
                        )));
                    }
                    if rank_none && self.formula_row[s] != NO_FORMULA {
                        return Err(Error::config(format!(
                            "PackedCandidateBatch::validate: spectrum {b} slot {r} has no formula (rank u32::MAX) but formula_row {} (expected u32::MAX)",
                            self.formula_row[s]
                        )));
                    }
                    // Every filled slot needs real formula provenance: a
                    // rank below `rows_scored` with non-zero counts that
                    // replay (a finished graph with zero counts and MAX
                    // row/rank is corrupt, not formula-less).
                    if rank_none {
                        return Err(Error::config(format!(
                            "PackedCandidateBatch::validate: spectrum {b} slot {r} is filled but has no formula provenance (rank u32::MAX with zero counts); every filled candidate needs a real formula rank below rows_scored {} with non-zero counts that replay",
                            self.rows_scored[b]
                        )));
                    }
                    if !rank_none {
                        let scored = self.rows_scored[b];
                        if self.formula_rank[s] >= scored {
                            return Err(Error::config(format!(
                                "PackedCandidateBatch::validate: spectrum {b} slot {r} formula_rank {} is not below rows_scored {scored}",
                                self.formula_rank[s]
                            )));
                        }
                        if src == 0 && self.formula_row[s] == NO_FORMULA {
                            return Err(Error::config(format!(
                                "PackedCandidateBatch::validate: spectrum {b} slot {r} has table source with formula_rank {} but formula_row u32::MAX (expected a real table row)",
                                self.formula_rank[s]
                            )));
                        }
                    }
                    // Rule 5b: the identity resolution is a known value.
                    if self.identity_resolution[s] > 2 {
                        return Err(Error::config(format!(
                            "PackedCandidateBatch::validate: spectrum {b} slot {r} identity_resolution {} is not in 0..=2 (0 trace only, 1 exact, 2 unresolved)",
                            self.identity_resolution[s]
                        )));
                    }
                    // Evidence pseudo-label invariants: status base 0,1,2
                    // with optional bit 7; count 0..=E; zero padding beyond
                    // the count; finite log-probs on filled records.
                    {
                        let ev = self.evidence_status[s];
                        let base = ev & 0x7F;
                        if base != 0 && base != 1 && base != 2 {
                            return Err(Error::config(format!(
                                "PackedCandidateBatch::validate: spectrum {b} slot {r} evidence_status {ev} has base {base} outside 0, 1, 2"
                            )));
                        }
                        if ev & !(0x7F | 0x80) != 0 {
                            return Err(Error::config(format!(
                                "PackedCandidateBatch::validate: spectrum {b} slot {r} evidence_status {ev} carries reserved bits"
                            )));
                        }
                        let ecap = crate::models::ms2::contract::EVIDENCE_CAP;
                        let count = self.evidence_count[s] as usize;
                        if count > ecap {
                            return Err(Error::config(format!(
                                "PackedCandidateBatch::validate: spectrum {b} slot {r} evidence_count {count} exceeds {ecap}"
                            )));
                        }
                        if (base == 0) != (count == 0) {
                            return Err(Error::config(format!(
                                "PackedCandidateBatch::validate: spectrum {b} slot {r} evidence_status {ev} disagrees with evidence_count {count} (0 exactly when unassigned)"
                            )));
                        }
                        for q in 0..ecap {
                            let lp = self.evidence_log_prob[s * ecap + q];
                            if q >= count {
                                if self.evidence_peak_id[s * ecap + q] != 0
                                    || self.evidence_hypothesis[s * ecap + q] != 0
                                    || self.evidence_shift[s * ecap + q] != 0
                                    || self.evidence_residual[s * ecap + q] != 0
                                    || lp != 0.0
                                {
                                    return Err(Error::config(format!(
                                        "PackedCandidateBatch::validate: spectrum {b} slot {r} evidence slot {q} beyond count {count} is not zero-padded"
                                    )));
                                }
                            } else if !(lp.is_finite() && lp <= 1e-4) {
                                return Err(Error::config(format!(
                                    "PackedCandidateBatch::validate: spectrum {b} slot {r} evidence slot {q} log_prob {lp} is not a log-probability"
                                )));
                            }
                        }
                    }
                    // Rule 5c: original trajectories are unique within a
                    // spectrum (they are compared against the earlier filled
                    // slots only; unfilled slots carry `u32::MAX`).
                    for q in 0..r {
                        let t = b * self.returned + q;
                        let earlier_filled = q < self.returned_count[b] as usize;
                        if earlier_filled && self.trajectory[t] == self.trajectory[s] {
                            return Err(Error::config(format!(
                                "PackedCandidateBatch::validate: spectrum {b} slots {q} and {r} share original trajectory {} (trajectories are unique within a spectrum)",
                                self.trajectory[s]
                            )));
                        }
                    }
                    // Rule 5d: a packed record is finished and valid.
                    let st = self.status[s];
                    if st & candidate_status::FINISHED == 0
                        || st & candidate_status::TRUNCATED != 0
                        || st & candidate_status::INVALID_FINAL != 0
                        || st & candidate_status::DUPLICATE_TRACE != 0
                        || st & candidate_status::REQUEST_FAILED != 0
                        || st & DUPLICATE_GRAPH != 0
                    {
                        return Err(Error::config(format!(
                            "PackedCandidateBatch::validate: spectrum {b} slot {r} status {st} is not a finished, valid, non-duplicate candidate"
                        )));
                    }
                    for (field, value) in [
                        ("formula_log_prob", self.formula_log_prob[s]),
                        ("trace_log_prob", self.trace_log_prob[s]),
                    ] {
                        if !(score_in_domain(value) && value <= 1e-4) {
                            return Err(Error::config(format!(
                                "PackedCandidateBatch::validate: spectrum {b} slot {r} {field} {value} is outside the log-probability range (inside (-3e38, 3e38) and <= 1e-4)"
                            )));
                        }
                    }
                    let len = self.length[s] as usize;
                    if len == 0 || len > self.max_steps {
                        return Err(Error::config(format!(
                            "PackedCandidateBatch::validate: spectrum {b} slot {r} length {len} is not in 1..=max_steps {}",
                            self.max_steps
                        )));
                    }
                    let abase = s * self.max_steps * 4;
                    for step in len..self.max_steps {
                        if self.actions[abase + step * 4..abase + step * 4 + 4] != [0, 0, 0, 0]
                        {
                            return Err(Error::config(format!(
                                "PackedCandidateBatch::validate: spectrum {b} slot {r} step {step} past length {len} is not PAD"
                            )));
                        }
                    }
                    let mut tokens = Vec::with_capacity(len);
                    for step in 0..len {
                        let fields = &self.actions[abase + step * 4..abase + step * 4 + 4];
                        let mut bytes = [0u8; 4];
                        for (f, v) in fields.iter().enumerate() {
                            if *v > u32::from(u8::MAX) {
                                return Err(Error::config(format!(
                                    "PackedCandidateBatch::validate: spectrum {b} slot {r} step {step} field {f} value {v} does not fit u8 for the grammar replay"
                                )));
                            }
                            bytes[f] = *v as u8;
                        }
                        tokens.push(Token {
                            kind: bytes[0],
                            atom_type: bytes[1],
                            bond: bytes[2],
                            pointer: bytes[3],
                        });
                    }
                    // Rule 5e: the emitted prefix replays legally with the
                    // grammar (`replay`, the same function
                    // `CandidateBatch::validate` uses) under the formula's
                    // composition budget: the slot's `formula_counts` when it
                    // has a formula, no budget when it has none.
                    let mut composition = [0u16; 10];
                    for (e, slot) in composition.iter_mut().enumerate() {
                        *slot = self.formula_counts[s * 10 + e];
                    }
                    let budget = if rank_none { None } else { Some(composition) };
                    let state = replay(&tokens, limits, budget).map_err(|e| {
                        Error::config(format!(
                            "PackedCandidateBatch::validate: spectrum {b} slot {r} emitted prefix fails the grammar replay under the formula composition budget: {e}"
                        ))
                    })?;
                    if !state.stopped() {
                        return Err(Error::config(format!(
                            "PackedCandidateBatch::validate: spectrum {b} slot {r} is finished but the replayed prefix is not stopped"
                        )));
                    }
                    if state.atoms() == 0 {
                        return Err(Error::config(format!(
                            "PackedCandidateBatch::validate: spectrum {b} slot {r} is finished with no atom"
                        )));
                    }
                    let atoms = state.atoms();
                    let residual = state.residual_valence();
                    let open =
                        &self.open_valence[s * self.max_atoms..(s + 1) * self.max_atoms];
                    if open[..atoms] != residual[..] || open[atoms..].iter().any(|&v| v != 0)
                    {
                        return Err(Error::config(format!(
                            "PackedCandidateBatch::validate: spectrum {b} slot {r} open valence {open:?} does not match the replayed residual valences {residual:?} with zeros after"
                        )));
                    }
                    let oracle = st & candidate_status::FORMULA_SOURCE_ORACLE != 0;
                    if oracle && self.formula_log_prob[s] != 0.0 {
                        return Err(Error::config(format!(
                            "PackedCandidateBatch::validate: spectrum {b} slot {r} has formula_source_oracle with formula_log_prob {} (expected 0)",
                            self.formula_log_prob[s]
                        )));
                    }
                }
            }
            // Per-spectrum contract rules carried over from `CandidateBatch`.
            for (field, value) in [
                ("intensity_retained", self.intensity_retained[b]),
                ("formula_mass_retained", self.formula_mass_retained[b]),
            ] {
                if !(0.0..=1.0 + 1e-4).contains(&value) {
                    return Err(Error::config(format!(
                        "PackedCandidateBatch::validate: spectrum {b} {field} {value} is outside the retained-fraction range [0, 1 + 1e-4]"
                    )));
                }
            }
            if self.formula_support_complete[b] > 1 {
                return Err(Error::config(format!(
                    "PackedCandidateBatch::validate: spectrum {b} formula_support_complete {} is not 0 or 1",
                    self.formula_support_complete[b]
                )));
            }
            if self.formula_support_complete[b] == 1 && self.rows_scored[b] != self.rows_joined[b]
            {
                return Err(Error::config(format!(
                    "PackedCandidateBatch::validate: spectrum {b} formula_support_complete is 1 but rows_scored {} != rows_joined {}",
                    self.rows_scored[b], self.rows_joined[b]
                )));
            }
            // Per-spectrum search counters: the SAME source-specific rule as
            // `CandidateBatch::validate` ([`super::contract::validate_search_counters`]),
            // so legal enumeration output (e.g. visited 1, joined 2,
            // scored 2) validates in every mode.
            super::contract::validate_search_counters(
                "PackedCandidateBatch::validate",
                b,
                self.formula_source[b] == 1,
                self.rows_visited[b],
                self.rows_joined[b],
                self.rows_scored[b],
                self.request_status[b],
            )?;
        }
        for a in 0..self.batch {
            for c in (a + 1)..self.batch {
                if self.spectrum_id[a * self.returned] == self.spectrum_id[c * self.returned] {
                    return Err(Error::config(format!(
                        "PackedCandidateBatch::validate: spectra {a} and {c} share spectrum_id {} against (spectrum, rank) order",
                        self.spectrum_id[a * self.returned]
                    )));
                }
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Kernel-expressible buffer helpers (copies of the `#[cube]` helpers)
// ---------------------------------------------------------------------------

/// One guarded full-buffer read: `buf[base + idx]` when it exists, else `0`.
fn slot(buf: &[u32], base: u32, idx: u32) -> u32 {
    let addr = base + idx;
    let mut out = 0u32;
    if (addr as usize) < buf.len() {
        out = buf[addr as usize];
    }
    out
}

/// One guarded full-buffer write.
fn put(buf: &mut [u32], base: u32, idx: u32, value: u32) {
    let addr = base + idx;
    if (addr as usize) < buf.len() {
        buf[addr as usize] = value;
    }
}

/// One guarded float read: `buf[base + idx]` when it exists, else `0.0`.
fn fslot(buf: &[f32], base: u32, idx: u32) -> f32 {
    let addr = (base + idx) as usize;
    let mut out = 0.0f32;
    if addr < buf.len() {
        out = buf[addr];
    }
    out
}

/// One guarded float write.
fn fput(buf: &mut [f32], base: u32, idx: u32, value: f32) {
    let addr = (base + idx) as usize;
    if addr < buf.len() {
        buf[addr] = value;
    }
}

/// Whether a trajectory may be ranked: finished, valid (`invalid_final`
/// clear), not `duplicate_trace`, not `duplicate_graph` (bit 7 of `bits`
/// when `use_graph` is 1), not `request_failed`. An unresolved identity (bit
/// 8) stays eligible by contract. `bad` is 1 exactly when a raw
/// log-probability or the ranking score lies outside the validated score
/// domain (−3e38, 3e38); anything else (NaN, infinities, finite extremes at
/// or beyond the bound, overflowing sums) is treated as invalid and excluded
/// from ranking, by specification. The finiteness test is a range test, never a NaN
/// self-comparison, which fast-math backends fold to false so NaN would rank.
/// Raw terms are always required in-domain, in both score modes, because a
/// packed record carries them and they must satisfy the log-probability
/// range.
fn eligible_lane(status: u32, bits: u32, use_graph: u32, bad: u32) -> u32 {
    let mut e = 1u32;
    if status & candidate_status::FINISHED == 0 {
        e = 0;
    }
    if status & candidate_status::INVALID_FINAL != 0 {
        e = 0;
    }
    if status & candidate_status::DUPLICATE_TRACE != 0 {
        e = 0;
    }
    if status & candidate_status::REQUEST_FAILED != 0 {
        e = 0;
    }
    if use_graph == 1 && bits & DUPLICATE_GRAPH != 0 {
        e = 0;
    }
    if bad == 1 {
        e = 0;
    }
    e
}

/// The bad-score flag of one trajectory: 1 unless both raw log-probabilities
/// and the ranking score are inside the validated score domain (−3e38,
/// 3e38); anything outside it is treated as invalid and excluded from
/// ranking, by specification (see [`SCORE_FINITE_MAX`]).
fn score_bad_lane(trace_lp: f32, formula_lp: f32, score: f32) -> u32 {
    let mut bad = 1u32;
    if trace_lp > -SCORE_FINITE_MAX
        && trace_lp < SCORE_FINITE_MAX
        && formula_lp > -SCORE_FINITE_MAX
        && formula_lp < SCORE_FINITE_MAX
        && score > -SCORE_FINITE_MAX
        && score < SCORE_FINITE_MAX
    {
        bad = 0;
    }
    bad
}

/// The ranking score of one trajectory: `trace_log_prob + formula_log_prob`
/// in that order, or the caller-supplied reranker score when `use_rerank` is
/// 1.
fn score_of(trace_lp: f32, formula_lp: f32, rerank_v: f32, use_rerank: u32) -> f32 {
    let mut s = trace_lp + formula_lp;
    if use_rerank == 1 {
        s = rerank_v;
    }
    s
}

/// Per-trajectory ranking scores: the kernel twin of `ms2_scores_fill`
/// (copied line for line by `ops::ms2_pack`).
///
/// Full-buffer form: `actions` holds device trajectory records of
/// `record_stride` words with the trace log-probability as `f32` bits at
/// `len_field + 2` (`len_field = steps * 4 + atoms`); `traj_formula` is
/// `[rows, 12]` flat (word 0 is the retained slot, `u32::MAX` when there is
/// none); `top_log_prob` is `[B, F]` flat. Returns `(trace_log_prob,
/// formula_log_prob)` (0.0 for a missing formula).
#[allow(clippy::too_many_arguments)]
pub fn scores_fill_lane(
    actions: &[u32],
    record_stride: u32,
    len_field: u32,
    traj_formula: &[u32],
    top_log_prob: &[f32],
    formulas: u32,
    record: u32,
    per_spectrum: u32,
) -> (f32, f32) {
    let abase = record * record_stride;
    let mut tl = 0.0f32;
    if ((abase + len_field + 2) as usize) < actions.len() {
        tl = f32::from_bits(actions[(abase + len_field + 2) as usize]);
    }
    let mut fl = 0.0f32;
    let b = record / per_spectrum;
    let k = record % per_spectrum;
    let s = slot(traj_formula, (b * per_spectrum + k) * TRAJ_FORMULA_STRIDE, 0);
    if s != u32::MAX && s < formulas && ((b * formulas + s) as usize) < top_log_prob.len() {
        fl = top_log_prob[(b * formulas + s) as usize];
    }
    (tl, fl)
}

// ---------------------------------------------------------------------------
// Twin lanes (copied line for line by `ops::ms2_pack`)
// ---------------------------------------------------------------------------

/// Per-trajectory rank among the eligible trajectories of its spectrum by
/// decreasing score, ties by smaller trajectory; `u32::MAX` when ineligible
/// (never selected). `O(K)` comparisons.
///
/// Full-buffer form: `actions` holds device trajectory records of
/// `record_stride` words with the status at `len_field + 1`
/// (`len_field = steps * 4 + atoms`); `id_bits` holds one identity word per
/// record (bit 7 excludes when `use_graph` is 1); the score reads
/// `trace_lp`/`formula_lp` (or `rerank` when `use_rerank` is 1), one entry per
/// record.
#[allow(clippy::too_many_arguments)]
pub fn rank_lane(
    actions: &[u32],
    record_stride: u32,
    len_field: u32,
    id_bits: &[u32],
    use_graph: u32,
    trace_lp: &[f32],
    formula_lp: &[f32],
    rerank: &[f32],
    use_rerank: u32,
    record: u32,
    per_spectrum: u32,
) -> u32 {
    let abase = record * record_stride;
    let status = slot(actions, abase, len_field + 1);
    let bits = slot(id_bits, 0, record);
    let tl0 = fslot(trace_lp, 0, record);
    let fl0 = fslot(formula_lp, 0, record);
    let my = score_of(tl0, fl0, fslot(rerank, 0, record), use_rerank);
    let mine = eligible_lane(status, bits, use_graph, score_bad_lane(tl0, fl0, my));
    let mut rank = u32::MAX;
    if mine == 1 && per_spectrum > 0 {
        let base = (record / per_spectrum) * per_spectrum;
        let mut r = 0u32;
        let mut j = 0u32;
        while j < per_spectrum {
            let other = base + j;
            let obase = other * record_stride;
            let st = slot(actions, obase, len_field + 1);
            let ob = slot(id_bits, 0, other);
            let tl = fslot(trace_lp, 0, other);
            let fl = fslot(formula_lp, 0, other);
            let s = score_of(tl, fl, fslot(rerank, 0, other), use_rerank);
            let e = eligible_lane(st, ob, use_graph, score_bad_lane(tl, fl, s));
            if e == 1 && (s > my || (s == my && other < record)) {
                r += 1;
            }
            j += 1;
        }
        rank = r;
    }
    rank
}

/// Per-trajectory gather into a [`record_width`] integer record at
/// `out_base` and a [`WF`] float record at `out_fbase`.
///
/// `traj_formula` is `[rows, 12]` flat (slot, source id, 10 counts);
/// `evidence` is `[rows, 18]` flat (word 0 is the evidence status);
/// `identity` is `[rows, 2]` flat (bits to OR into the status, resolution).
/// The status written is the `actions` status with the identity bits ORed in
/// (when `use_identity` is 1); the resolution is the identity word (0 when
/// identity is absent). The float score word is the ranking score: the raw
/// sum, or `rerank[record]` when `use_rerank` is 1.
#[allow(clippy::too_many_arguments)]
pub fn record_pack_lane(
    actions: &[u32],
    record_stride: u32,
    steps: u32,
    atoms_cap: u32,
    traj_formula: &[u32],
    evidence: &[u32],
    ev_stride: u32,
    identity: &[u32],
    use_identity: u32,
    trace_lp: &[f32],
    formula_lp: &[f32],
    rerank: &[f32],
    use_rerank: u32,
    record: u32,
    spectrum: u32,
    traj: u32,
    out: &mut [u32],
    out_base: u32,
    out_f: &mut [f32],
    out_fbase: u32,
) {
    let abase = record * record_stride;
    let tfbase = record * TRAJ_FORMULA_STRIDE;
    let evbase = record * ev_stride;
    let idbase = record * 2;
    let len_field = steps * 4 + atoms_cap;
    let length = slot(actions, abase, len_field);
    let status_in = slot(actions, abase, len_field + 1);
    let formula_row = slot(actions, abase, len_field + 3);
    let formula_slot = slot(traj_formula, tfbase, 0);
    let ev_status = slot(evidence, evbase, 0);
    let mut status = status_in;
    let mut resolution = 0u32;
    if use_identity == 1 {
        let bits = slot(identity, idbase, 0);
        status |= bits;
        resolution = slot(identity, idbase, 1);
    }
    put(out, out_base, O_SPECTRUM, spectrum);
    put(out, out_base, O_TRAJECTORY, traj);
    put(out, out_base, O_LENGTH, length);
    put(out, out_base, O_FORMULA_ROW, formula_row);
    put(out, out_base, O_FORMULA_RANK, formula_slot);
    let mut e = 0u32;
    while e < 10 {
        put(out, out_base, O_COUNTS + e, slot(traj_formula, tfbase, 2 + e));
        e += 1;
    }
    put(out, out_base, O_STATUS, status);
    put(out, out_base, O_EVIDENCE, ev_status);
    put(out, out_base, O_RESOLUTION, resolution);
    put(out, out_base, O_ATTACHMENT, 0);
    let mut w = 0u32;
    while w < steps * 4 {
        put(out, out_base, O_TOKENS + w, slot(actions, abase, w));
        w += 1;
    }
    let mut v = 0u32;
    while v < atoms_cap {
        put(
            out,
            out_base,
            O_TOKENS + steps * 4 + v,
            slot(actions, abase, steps * 4 + v),
        );
        v += 1;
    }
    let tl = fslot(trace_lp, 0, record);
    let fl = fslot(formula_lp, 0, record);
    let mut sc = tl + fl;
    if use_rerank == 1 {
        sc = fslot(rerank, 0, record);
    }
    fput(out_f, out_fbase, OF_FORMULA_LP, fl);
    fput(out_f, out_fbase, OF_TRACE_LP, tl);
    fput(out_f, out_fbase, OF_SCORE, sc);
}

/// Per-output-slot `(spectrum, slot_r)` compaction: the trajectory of rank
/// `slot_r` is found by scanning the spectrum's ranks and its records are
/// copied, or the unfilled pattern is written (trajectory, formula row and
/// formula rank `u32::MAX`, status `0`, zero payload).
///
/// `record_stride` is the full integer record width ([`record_width`]) and
/// `record_fstride` is [`WF`].
#[allow(clippy::too_many_arguments)]
pub fn pack_lane(
    ranks: &[u32],
    rec: &[u32],
    record_stride: u32,
    rec_f: &[f32],
    record_fstride: u32,
    spectrum: u32,
    slot_r: u32,
    per_spectrum: u32,
    out: &mut [u32],
    out_base: u32,
    out_f: &mut [f32],
    out_fbase: u32,
) {
    let mut found = u32::MAX;
    let mut j = 0u32;
    let base = spectrum * per_spectrum;
    while j < per_spectrum {
        if slot(ranks, 0, base + j) == slot_r {
            found = j;
        }
        j += 1;
    }
    if found == u32::MAX {
        let mut w = 0u32;
        while w < record_stride {
            put(out, out_base, w, 0);
            w += 1;
        }
        put(out, out_base, O_SPECTRUM, spectrum);
        put(out, out_base, O_TRAJECTORY, u32::MAX);
        put(out, out_base, O_FORMULA_ROW, u32::MAX);
        put(out, out_base, O_FORMULA_RANK, u32::MAX);
        let mut q = 0u32;
        while q < record_fstride {
            fput(out_f, out_fbase, q, 0.0);
            q += 1;
        }
    } else {
        let src = (base + found) * record_stride;
        let mut w = 0u32;
        while w < record_stride {
            put(out, out_base, w, slot(rec, src, w));
            w += 1;
        }
        let src_f = (base + found) * record_fstride;
        let mut q = 0u32;
        while q < record_fstride {
            fput(out_f, out_fbase, q, fslot(rec_f, src_f, q));
            q += 1;
        }
    }
}

/// Filled slots of one spectrum: the eligible count capped at `returned`.
pub fn returned_count_lane(
    ranks: &[u32],
    spectrum: u32,
    per_spectrum: u32,
    returned: u32,
) -> u32 {
    let mut count = 0u32;
    let mut j = 0u32;
    let base = spectrum * per_spectrum;
    while j < per_spectrum {
        if slot(ranks, 0, base + j) != u32::MAX {
            count += 1;
        }
        j += 1;
    }
    if count > returned {
        count = returned;
    }
    count
}

/// Per-output-slot `(spectrum, slot_r)` evidence compaction: the kernel twin
/// of `ms2_pack_evidence` (copied line for line by `ops::ms2_pack`).
///
/// `ranks` holds one rank per `B * K` record (`u32::MAX` when ineligible);
/// `evidence` is `[rows, 18]` flat (status, total qualifying count, then
/// records of kept-peak position, hypothesis index, shift offset by 2^31,
/// residual offset by 2^31); `traj_slot` is `[rows, 2]` flat (formula slot,
/// adduct); `log_prob` is `[B, F, N, J + 1]` flat assignment
/// log-probabilities. The trajectory of rank `slot_r` is found by scanning
/// the spectrum's ranks (as in [`pack_lane`); its 18 evidence words are
/// copied (kept-peak positions, not original ids: the host maps those at
/// readout exactly as `generate` does) and the assignment log-probability of
/// each stored record is looked up by `(slot, position, hypothesis)`.
/// An unfilled slot writes zeros. `count` beyond
/// [`EVIDENCE_CAP`](super::contract::EVIDENCE_CAP) keeps its total in the
/// row while only the stored records' log-probabilities are packed.
#[allow(clippy::too_many_arguments)]
pub fn pack_evidence_lane(
    ranks: &[u32],
    evidence: &[u32],
    traj_slot: &[u32],
    log_prob: &[f32],
    formulas: u32,
    n: u32,
    j: u32,
    spectrum: u32,
    slot_r: u32,
    per_spectrum: u32,
    out_ev: &mut [u32],
    out_base: u32,
    out_f: &mut [f32],
    out_fbase: u32,
) {
    let ecap = super::contract::EVIDENCE_CAP as u32;
    let mut found = u32::MAX;
    let mut t = 0u32;
    let base = spectrum * per_spectrum;
    while t < per_spectrum {
        if slot(ranks, 0, base + t) == slot_r {
            found = t;
        }
        t += 1;
    }
    if found == u32::MAX {
        let mut w = 0u32;
        while w < EVIDENCE_STRIDE {
            put(out_ev, out_base, w, 0);
            w += 1;
        }
        let mut q = 0u32;
        while q < ecap {
            fput(out_f, out_fbase, q, 0.0);
            q += 1;
        }
    } else {
        let src = base + found;
        let mut w = 0u32;
        while w < EVIDENCE_STRIDE {
            put(out_ev, out_base, w, slot(evidence, src * EVIDENCE_STRIDE, w));
            w += 1;
        }
        let total = slot(evidence, src * EVIDENCE_STRIDE, 1);
        let mut kept_c = total;
        if kept_c > ecap {
            kept_c = ecap;
        }
        let slot_t = slot(traj_slot, src * 2, 0);
        let mut slot_c = 0u32;
        if slot_t < formulas {
            slot_c = slot_t;
        }
        let width = j + 1;
        let mut q = 0u32;
        while q < ecap {
            let mut v = 0.0f32;
            if q < kept_c {
                let wpos = src * EVIDENCE_STRIDE + 2 + q * 4;
                let p = slot(evidence, wpos, 0);
                let bq = slot(evidence, wpos, 1);
                let lp_idx = (spectrum * formulas + slot_c) * n * width + p * width + bq;
                if (lp_idx as usize) < log_prob.len() {
                    v = log_prob[lp_idx as usize];
                }
            }
            fput(out_f, out_fbase, q, v);
            q += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Host packing (calls the lanes above)
// ---------------------------------------------------------------------------

/// Packed device words of one `(B, K)` bucket: the host form of
/// `ms2_record_pack` + `ms2_pack` over already-read buffers. See
/// [`pack_from_device_layout`].
pub struct DevicePackWords {
    /// Integer records in `(spectrum, rank)` order: `[B * R, W]`.
    pub packed: Vec<u32>,
    /// Float records in `(spectrum, rank)` order: `[B * R, WF]`.
    pub packed_f: Vec<f32>,
    /// Filled slots per spectrum.
    pub returned_count: Vec<u32>,
    /// Rank per trajectory in record order (`u32::MAX` when ineligible).
    pub ranks: Vec<u32>,
}

/// Pack the device-layout buffers of one `(B, K)` bucket into packed records:
/// the host form of `ms2_record_pack` + `ms2_pack` over already-read buffers.
/// Used by [`pack`] and by the kernel tests to compare device words with the
/// twin; the integration task (P6.5) calls this on its final read.
///
/// Every stride, count and largest accessed address is checked against the
/// u32 domain ([`check_u32_len`], [`check_u32_product`]) before any host lane
/// call: an unsupported size is [`Error::Shape`]. `returned > per_spectrum`
/// is [`Error::Config`], mirroring the device [`pack`](crate::tensor::ops::ms2_pack::pack)
/// wrapper (`1 <= R <= K`).
#[allow(clippy::too_many_arguments)]
pub fn pack_from_device_layout(
    actions: &[u32],
    record_stride: usize,
    traj_formula: &[u32],
    evidence: &[u32],
    identity: &[u32],
    use_identity: u32,
    trace_lp: &[f32],
    formula_lp: &[f32],
    rerank: &[f32],
    use_rerank: u32,
    batch: usize,
    per_spectrum: usize,
    steps: u32,
    atoms: u32,
    returned: usize,
) -> Result<DevicePackWords> {
    if per_spectrum == 0 {
        return Err(Error::shape(
            "pack_from_device_layout: per_spectrum is 0: lanes need at least one trajectory per spectrum"
                .to_string(),
        ));
    }
    if returned > per_spectrum {
        return Err(Error::config(format!(
            "pack_from_device_layout: returned {returned} exceeds per_spectrum {per_spectrum} (1 <= R <= K)"
        )));
    }
    // Checked u32 addressing (finding E2): every narrowing below is guarded
    // by these checks, so no `as u32` can truncate and no lane address
    // (`record * stride + offset`, at most `rows * stride - 1`) can wrap.
    let rows = check_u32_product("pack_from_device_layout records", batch, per_spectrum)?;
    let record_stride32 = check_u32_len("pack_from_device_layout record_stride", record_stride)?;
    let steps32 = check_u32_len("pack_from_device_layout steps", steps as usize)?;
    let atoms32 = check_u32_len("pack_from_device_layout atoms", atoms as usize)?;
    let per_spectrum32 = check_u32_len("pack_from_device_layout per_spectrum", per_spectrum)?;
    let returned32 = check_u32_len("pack_from_device_layout returned", returned)?;
    let _ = check_u32_product("pack_from_device_layout actions addresses", rows, record_stride)?;
    let _ = check_u32_product("pack_from_device_layout traj_formula addresses", rows, TRAJ_FORMULA_STRIDE as usize)?;
    let _ = check_u32_product("pack_from_device_layout evidence addresses", rows, EVIDENCE_STRIDE as usize)?;
    let _ = check_u32_product("pack_from_device_layout identity addresses", rows, 2)?;
    let _ = check_u32_product("pack_from_device_layout scores addresses", rows, 2)?;
    let width = record_width(steps32 as usize, atoms32 as usize);
    let width32 = check_u32_len("pack_from_device_layout record width", width)?;
    let _ = check_u32_product("pack_from_device_layout record addresses", rows, width)?;
    let _ = check_u32_product("pack_from_device_layout record_f addresses", rows, WF)?;
    let slots = check_u32_product("pack_from_device_layout slots", batch, returned)?;
    let _ = check_u32_product("pack_from_device_layout packed addresses", slots, width)?;
    let _ = check_u32_product("pack_from_device_layout packed_f addresses", slots, WF)?;
    let mut ranks = vec![u32::MAX; rows];
    let mut id_bits = vec![0u32; rows];
    let mut i = 0usize;
    while i < rows {
        id_bits[i] = slot(identity, (i as u32) * 2, 0);
        i += 1;
    }
    let len_field32 = steps32
        .checked_mul(4)
        .and_then(|v| v.checked_add(atoms32))
        .ok_or_else(|| {
            Error::shape(format!(
                "pack_from_device_layout: steps {steps32} * 4 + atoms {atoms32} overflows u32"
            ))
        })?;
    let mut rr = 0u32;
    while (rr as usize) < rows {
        ranks[rr as usize] = rank_lane(
            actions,
            record_stride32,
            len_field32,
            &id_bits,
            use_identity,
            trace_lp,
            formula_lp,
            rerank,
            use_rerank,
            rr,
            per_spectrum32,
        );
        rr += 1;
    }
    let mut rec = vec![0u32; rows * width];
    let mut rec_f = vec![0.0f32; rows * WF];
    let mut t = 0u32;
    while (t as usize) < rows {
        let spectrum = t / per_spectrum32;
        let traj = t % per_spectrum32;
        record_pack_lane(
            actions,
            record_stride32,
            steps32,
            atoms32,
            traj_formula,
            evidence,
            EVIDENCE_STRIDE,
            identity,
            use_identity,
            trace_lp,
            formula_lp,
            rerank,
            use_rerank,
            t,
            spectrum,
            traj,
            &mut rec,
            t * width32,
            &mut rec_f,
            t * WF as u32,
        );
        t += 1;
    }
    let mut packed = vec![0u32; slots * width];
    let mut packed_f = vec![0.0f32; slots * WF];
    let mut counts = vec![0u32; batch];
    let mut b = 0u32;
    while (b as usize) < batch {
        counts[b as usize] =
            returned_count_lane(&ranks, b, per_spectrum32, returned32);
        let mut s = 0u32;
        while (s as usize) < returned {
            let slot_base = (b as usize * returned + s as usize) * width;
            pack_lane(
                &ranks,
                &rec,
                width32,
                &rec_f,
                WF as u32,
                b,
                s,
                per_spectrum32,
                &mut packed,
                slot_base as u32,
                &mut packed_f,
                (slot_base / width * WF) as u32,
            );
            s += 1;
        }
        b += 1;
    }
    Ok(DevicePackWords {
        packed,
        packed_f,
        returned_count: counts,
        ranks,
    })
}

/// Assemble a [`PackedCandidateBatch`] from packed device words (in
/// `(spectrum, rank)` order) and the per-spectrum fields of `batch`: the host
/// side of the packed readout. Used by [`pack`] and by the integration task
/// on its final read.
pub fn assemble(
    batch: &CandidateBatch,
    packed: &[u32],
    packed_f: &[f32],
    returned_count: &[u32],
    returned: usize,
) -> Result<PackedCandidateBatch> {
    let steps = batch.max_steps;
    let atoms = batch.max_atoms;
    let steps_x4 = steps.checked_mul(4).ok_or_else(|| {
        Error::shape(format!(
            "pack::assemble: max_steps {steps} * 4 overflows usize"
        ))
    })?;
    let width = W
        .checked_add(steps_x4)
        .and_then(|v| v.checked_add(atoms))
        .ok_or_else(|| {
            Error::shape(format!(
                "pack::assemble: record width {W} + {steps}*4 + {atoms} overflows usize"
            ))
        })?;
    let slots = check_u32_product("pack::assemble slots", batch.batch, returned)?;
    let want = check_u32_product("pack::assemble packed words", slots, width)?;
    if packed.len() != want {
        return Err(Error::shape(format!(
            "pack::assemble: packed holds {} words for [B {}, R {returned}, W {width}] ({want} expected)",
            packed.len(),
            batch.batch
        )));
    }
    let want_f = check_u32_product("pack::assemble packed_f words", slots, WF)?;
    if packed_f.len() != want_f {
        return Err(Error::shape(format!(
            "pack::assemble: packed_f holds {} words for [B {}, R {returned}, WF {WF}] ({want_f} expected)",
            packed_f.len(),
            batch.batch
        )));
    }
    if returned_count.len() != batch.batch {
        return Err(Error::shape(format!(
            "pack::assemble: returned_count holds {} words for {} spectra",
            returned_count.len(),
            batch.batch
        )));
    }
    let n = slots;
    let toks = check_u32_product("pack::assemble action words", slots, steps_x4)?;
    let valence_words = check_u32_product("pack::assemble valence words", slots, atoms)?;
    let count_words = check_u32_product("pack::assemble count words", slots, 10)?;
    let mut out = PackedCandidateBatch {
        schema_version: SCHEMA_VERSION,
        batch: batch.batch,
        returned,
        trajectories: batch.trajectories,
        max_steps: steps,
        max_atoms: atoms,
        max_ring_closures: batch.max_ring_closures,
        spectrum_id: vec![0; n],
        trajectory: vec![0; n],
        actions: vec![0; toks],
        length: vec![0; n],
        formula_row: vec![NO_FORMULA; n],
        formula_rank: vec![NO_FORMULA; n],
        formula_counts: vec![0; count_words],
        formula_log_prob: vec![0.0; n],
        trace_log_prob: vec![0.0; n],
        score: vec![0.0; n],
        open_valence: vec![0; valence_words],
        status: vec![0; n],
        evidence_status: vec![0; n],
        evidence_count: vec![0; n],
        evidence_peak_id: vec![0; n * crate::models::ms2::contract::EVIDENCE_CAP],
        evidence_hypothesis: vec![0; n * crate::models::ms2::contract::EVIDENCE_CAP],
        evidence_shift: vec![0; n * crate::models::ms2::contract::EVIDENCE_CAP],
        evidence_residual: vec![0; n * crate::models::ms2::contract::EVIDENCE_CAP],
        evidence_log_prob: vec![0.0; n * crate::models::ms2::contract::EVIDENCE_CAP],
        identity_resolution: vec![0; n],
        attachment_partition: vec![0; n],
        returned_count: returned_count.to_vec(),
        request_status: batch.request_status.clone(),
        rows_visited: batch.rows_visited.clone(),
        rows_joined: batch.rows_joined.clone(),
        rows_scored: batch.rows_scored.clone(),
        formula_support_complete: batch.formula_support_complete.clone(),
        formula_mass_retained: batch.formula_mass_retained.clone(),
        peaks_kept: batch.peaks_kept.clone(),
        intensity_retained: batch.intensity_retained.clone(),
        formula_source: batch.formula_source.clone(),
    };
    let toks = steps_x4;
    for b in 0..batch.batch {
        for r in 0..returned {
            let s = b * returned + r;
            let base = s * width;
            out.spectrum_id[s] = batch.spectrum_id[b * batch.trajectories.max(1)];
            out.trajectory[s] = packed[base + O_TRAJECTORY as usize];
            out.length[s] = packed[base + O_LENGTH as usize];
            out.formula_row[s] = packed[base + O_FORMULA_ROW as usize];
            out.formula_rank[s] = packed[base + O_FORMULA_RANK as usize];
            for e in 0..10 {
                out.formula_counts[s * 10 + e] =
                    packed[base + O_COUNTS as usize + e] as u16;
            }
            out.status[s] = packed[base + O_STATUS as usize];
            out.evidence_status[s] = packed[base + O_EVIDENCE as usize] as u8;
            out.identity_resolution[s] = packed[base + O_RESOLUTION as usize] as u8;
            out.attachment_partition[s] = packed[base + O_ATTACHMENT as usize] as u8;
            let abase = s * toks;
            for w in 0..toks {
                out.actions[abase + w] = packed[base + O_TOKENS as usize + w];
            }
            let vbase = s * atoms;
            for v in 0..atoms {
                out.open_valence[vbase + v] =
                    packed[base + valence_offset(steps) + v] as u8;
            }
            let fbase = s * WF;
            out.formula_log_prob[s] = packed_f[fbase + OF_FORMULA_LP as usize];
            out.trace_log_prob[s] = packed_f[fbase + OF_TRACE_LP as usize];
            out.score[s] = packed_f[fbase + OF_SCORE as usize];
        }
    }
    Ok(out)
}

/// Re-inflate a packed batch to a [`CandidateBatch`] with `trajectories =
/// returned` (V1 §4.4, driver support): filled slots become their records,
/// unfilled slots become empty unfinished records (length 0, status 0, no
/// formula). The ranking score has no unpacked field and is dropped (it is
/// `formula_log_prob + trace_log_prob`, both carried).
///
/// Lets the packed top-R run through the trajectory-ordered metrics
/// (`evaluate_candidates`) for precision/coverage at R.
impl PackedCandidateBatch {
    pub fn to_candidate_batch(&self) -> Result<CandidateBatch> {
    let n = check_u32_product("pack::to_candidate_batch slots", self.batch, self.returned)?;
    let steps_x4 = self
        .max_steps
        .checked_mul(4)
        .ok_or_else(|| {
            Error::shape(format!(
                "pack::to_candidate_batch: max_steps {} * 4 overflows usize",
                self.max_steps
            ))
        })?;
    let toks = check_u32_product("pack::to_candidate_batch action words", n, steps_x4)?;
    if self.actions.len() != toks {
        return Err(Error::shape(format!(
            "pack::to_candidate_batch: actions hold {} words for {n} slots of {} steps ({toks} expected)",
            self.actions.len(),
            self.max_steps
        )));
    }
    let valence_words = check_u32_product(
        "pack::to_candidate_batch open-valence words",
        n,
        self.max_atoms,
    )?;
    if self.open_valence.len() != valence_words {
        return Err(Error::shape(format!(
            "pack::to_candidate_batch: open_valence holds {} words for {n} slots of {} atoms ({valence_words} expected)",
            self.open_valence.len(),
            self.max_atoms
        )));
    }
    let out = CandidateBatch {
        schema_version: SCHEMA_VERSION,
        batch: self.batch,
        trajectories: self.returned,
        max_steps: self.max_steps,
        max_atoms: self.max_atoms,
        max_ring_closures: self.max_ring_closures,
        spectrum_id: self.spectrum_id.clone(),
        trajectory: (0..n).map(|s| (s % self.returned.max(1)) as u32).collect(),
        actions: self.actions.clone(),
        length: self.length.clone(),
        formula_row: self.formula_row.clone(),
        formula_log_prob: self.formula_log_prob.clone(),
        trace_log_prob: self.trace_log_prob.clone(),
        open_valence: self.open_valence.clone(),
        attachment_partition: self.attachment_partition.clone(),
        status: self.status.clone(),
        evidence_status: self.evidence_status.clone(),
        identity_resolution: self.identity_resolution.clone(),
        request_status: self.request_status.clone(),
        rows_visited: self.rows_visited.clone(),
        rows_joined: self.rows_joined.clone(),
        rows_scored: self.rows_scored.clone(),
        formula_support_complete: self.formula_support_complete.clone(),
        formula_mass_retained: self.formula_mass_retained.clone(),
        peaks_kept: self.peaks_kept.clone(),
        intensity_retained: self.intensity_retained.clone(),
        formula_counts: self.formula_counts.clone(),
        formula_source: self.formula_source.clone(),
        formula_rank: self.formula_rank.clone(),
        evidence_count: self.evidence_count.clone(),
        evidence_peak_id: self.evidence_peak_id.clone(),
        evidence_hypothesis: self.evidence_hypothesis.clone(),
        evidence_shift: self.evidence_shift.clone(),
        evidence_residual: self.evidence_residual.clone(),
        evidence_log_prob: self.evidence_log_prob.clone(),
    };
    out.validate()?;
    Ok(out)
    }
}

/// Rank and compact one [`CandidateBatch`] to `returned` slots per spectrum.
///
/// Eligibility per trajectory: finished, valid (`invalid_final` clear), not
/// `duplicate_trace`, not `duplicate_graph` (bit 7 of `identity_bits` when
/// given — bit 8, `identity_unresolved`, stays eligible by contract),
/// `request_failed` clear, and raw log-probabilities and ranking score inside
/// the validated score domain (−3e38, 3e38); values outside it, NaN and
/// infinities are ineligible. The score is [`ScoreKind::Raw`]
/// (`formula_log_prob + trace_log_prob`) or the caller-supplied
/// [`ScoreKind::Reranker`] slice. Rank is by
/// decreasing score, ties by smaller trajectory. `returned_count` is
/// `min(returned, eligible)`; a failed request returns nothing.
///
/// `identity_bits` holds one identity word per `B * K` record (bits 7/8 as
/// `ms2_graph_identity` writes them); `None` means trace-only identity (no
/// graph-duplicate exclusion, resolution 0).
///
/// Every [`pack`] step runs through the twin lanes, so this function and the
/// kernels share one definition. `batch` itself is not re-validated here: a
/// NaN log-probability (rejected by `CandidateBatch::validate`) must still
/// pack, with the NaN record excluded.
///
/// Lengths: every consumed field is length-checked with checked products
/// before any device-layout buffer is built (finding E5); a malformed input
/// is [`Error::Shape`], never a panic. Every stride, count and largest
/// accessed address is checked against the u32 domain before any host lane
/// call (finding E2).
pub fn pack(
    batch: &CandidateBatch,
    identity_bits: Option<&[u32]>,
    score: ScoreKind<'_>,
    returned: usize,
) -> Result<PackedCandidateBatch> {
    let k = batch.trajectories;
    if k == 0 {
        return Err(Error::config(
            "pack: batch trajectories is 0: slots need at least one trajectory per spectrum"
                .to_string(),
        ));
    }
    if returned == 0 || returned > k {
        return Err(Error::config(format!(
            "pack: returned {returned} is not in 1..=trajectories {k}"
        )));
    }
    // Finding E5: every consumed field is length-checked with checked
    // products before any device-layout buffer is built; a malformed input
    // is `Error::Shape`, never a panic. Finding E2: every stride, count and
    // largest accessed address is checked against the u32 domain before any
    // host lane call, so no `as u32` below can truncate.
    let n = check_u32_product("pack records", batch.batch, k)?;
    let steps_x4 = batch
        .max_steps
        .checked_mul(4)
        .ok_or_else(|| {
            Error::shape(format!(
                "pack: max_steps {} * 4 overflows usize",
                batch.max_steps
            ))
        })?;
    let toks = check_u32_product("pack action words", n, steps_x4)?;
    let valence_words = check_u32_product("pack open-valence words", n, batch.max_atoms)?;
    let count_words = check_u32_product("pack formula-count words", n, 10)?;
    let per_record: [(&str, usize); 10] = [
        ("spectrum_id", batch.spectrum_id.len()),
        ("trajectory", batch.trajectory.len()),
        ("length", batch.length.len()),
        ("formula_row", batch.formula_row.len()),
        ("formula_rank", batch.formula_rank.len()),
        ("formula_log_prob", batch.formula_log_prob.len()),
        ("trace_log_prob", batch.trace_log_prob.len()),
        ("status", batch.status.len()),
        ("evidence_status", batch.evidence_status.len()),
        ("attachment_partition", batch.attachment_partition.len()),
    ];
    for (field, len) in per_record {
        if len != n {
            return Err(Error::shape(format!(
                "pack: field {field} has length {len} for {n} records"
            )));
        }
    }
    if batch.actions.len() != toks {
        return Err(Error::shape(format!(
            "pack: field actions has length {} for {n} records of {} steps ({toks} expected)",
            batch.actions.len(),
            batch.max_steps
        )));
    }
    if batch.open_valence.len() != valence_words {
        return Err(Error::shape(format!(
            "pack: field open_valence has length {} for {n} records of {} atoms ({valence_words} expected)",
            batch.open_valence.len(),
            batch.max_atoms
        )));
    }
    if batch.formula_counts.len() != count_words {
        return Err(Error::shape(format!(
            "pack: field formula_counts has length {} for {n} records of 10 counts ({count_words} expected)",
            batch.formula_counts.len()
        )));
    }
    let per_spectrum_fields: [(&str, usize); 9] = [
        ("request_status", batch.request_status.len()),
        ("rows_visited", batch.rows_visited.len()),
        ("rows_joined", batch.rows_joined.len()),
        ("rows_scored", batch.rows_scored.len()),
        (
            "formula_support_complete",
            batch.formula_support_complete.len(),
        ),
        ("formula_mass_retained", batch.formula_mass_retained.len()),
        ("peaks_kept", batch.peaks_kept.len()),
        ("intensity_retained", batch.intensity_retained.len()),
        ("formula_source", batch.formula_source.len()),
    ];
    for (field, len) in per_spectrum_fields {
        if len != batch.batch {
            return Err(Error::shape(format!(
                "pack: field {field} has length {len} for {} spectra",
                batch.batch
            )));
        }
    }
    if let Some(bits) = identity_bits
        && bits.len() != n
    {
        return Err(Error::shape(format!(
            "pack: identity_bits has length {} for {n} records",
            bits.len()
        )));
    }
    let (rerank, use_rerank) = match score {
        ScoreKind::Raw => (&[][..], 0u32),
        ScoreKind::Reranker(s) => {
            if s.len() != n {
                return Err(Error::shape(format!(
                    "pack: reranker scores hold {} entries for {n} records",
                    s.len()
                )));
            }
            (s, 1u32)
        }
    };
    let use_identity = if identity_bits.is_some() { 1u32 } else { 0u32 };
    let steps = batch.max_steps;
    let atoms = batch.max_atoms;
    let stride = steps_x4
        .checked_add(atoms)
        .and_then(|v| v.checked_add(4))
        .ok_or_else(|| {
            Error::shape(format!(
                "pack: record stride {steps}*4 + {atoms} + 4 overflows usize"
            ))
        })?;
    let _ = check_u32_len("pack record stride", stride)?;
    let _ = check_u32_len("pack per_spectrum", k)?;
    let _ = check_u32_len("pack steps", steps)?;
    let _ = check_u32_len("pack atoms", atoms)?;
    let _ = check_u32_len("pack returned", returned)?;
    let _ = check_u32_product("pack actions_flat words", n, stride)?;
    // Device-layout buffers, as the kernels would bind them. Every length
    // and every `as u32` below is guarded by the checked products above.
    let mut actions_flat = vec![0u32; n * stride];
    let stride32 = stride as u32;
    let steps_x4_usize = steps_x4;
    let mut r = 0usize;
    while r < n {
        let abase = (r as u32) * stride32;
        let tok_base = r * steps_x4_usize;
        let mut w = 0usize;
        while w < steps_x4_usize {
            put(&mut actions_flat, abase, w as u32, batch.actions[tok_base + w]);
            w += 1;
        }
        let mut v = 0usize;
        while v < atoms {
            put(
                &mut actions_flat,
                abase,
                (steps_x4_usize + v) as u32,
                u32::from(batch.open_valence[r * atoms + v]),
            );
            v += 1;
        }
        put(&mut actions_flat, abase, (steps_x4_usize + atoms) as u32, batch.length[r]);
        put(
            &mut actions_flat,
            abase,
            (steps_x4_usize + atoms + 1) as u32,
            batch.status[r],
        );
        put(
            &mut actions_flat,
            abase,
            (steps_x4_usize + atoms + 2) as u32,
            batch.trace_log_prob[r].to_bits(),
        );
        put(
            &mut actions_flat,
            abase,
            (steps_x4_usize + atoms + 3) as u32,
            batch.formula_row[r],
        );
        r += 1;
    }
    let mut traj_formula = vec![0u32; n * TRAJ_FORMULA_STRIDE as usize];
    let mut t = 0usize;
    while t < n {
        let base = (t as u32) * TRAJ_FORMULA_STRIDE;
        put(&mut traj_formula, base, 0, batch.formula_rank[t]);
        put(&mut traj_formula, base, 1, batch.formula_row[t]);
        let mut e = 0u32;
        while e < 10 {
            put(
                &mut traj_formula,
                base,
                2 + e,
                u32::from(batch.formula_counts[t * 10 + e as usize]),
            );
            e += 1;
        }
        t += 1;
    }
    let mut evidence = vec![0u32; n * EVIDENCE_STRIDE as usize];
    let mut q = 0usize;
    while q < n {
        put(
            &mut evidence,
            (q as u32) * EVIDENCE_STRIDE,
            0,
            u32::from(batch.evidence_status[q]),
        );
        q += 1;
    }
    let mut identity = vec![0u32; n * 2];
    let mut d = 0usize;
    while d < n {
        let bits = identity_bits.map_or(0, |b| b[d]);
        let mut res = 0u32;
        if use_identity == 1 {
            res = 1;
            if bits & IDENTITY_UNRESOLVED != 0 {
                res = 2;
            }
        }
        put(&mut identity, (d as u32) * 2, 0, bits);
        put(&mut identity, (d as u32) * 2, 1, res);
        d += 1;
    }
    let words = pack_from_device_layout(
        &actions_flat,
        stride,
        &traj_formula,
        &evidence,
        &identity,
        use_identity,
        &batch.trace_log_prob,
        &batch.formula_log_prob,
        rerank,
        use_rerank,
        batch.batch,
        k,
        steps as u32,
        atoms as u32,
        returned,
    )?;
    let mut out = assemble(
        batch,
        &words.packed,
        &words.packed_f,
        &words.returned_count,
        returned,
    )?;
    // Evidence details ride the host path from the `CandidateBatch` fields
    // (I3b): rank-select status, count and the `E` records per packed slot.
    // Unfilled slots stay zero (as `assemble` left them).
    {
        let ecap = super::contract::EVIDENCE_CAP;
        if batch.evidence_count.len() != n
            || batch.evidence_peak_id.len() != n * ecap
            || batch.evidence_hypothesis.len() != n * ecap
            || batch.evidence_shift.len() != n * ecap
            || batch.evidence_residual.len() != n * ecap
            || batch.evidence_log_prob.len() != n * ecap
        {
            return Err(Error::shape(format!(
                "pack: evidence fields hold {} and {} and {} and {} and {} and {} entries for {n} records of {ecap} slots",
                batch.evidence_count.len(),
                batch.evidence_peak_id.len(),
                batch.evidence_hypothesis.len(),
                batch.evidence_shift.len(),
                batch.evidence_residual.len(),
                batch.evidence_log_prob.len(),
            )));
        }
        for b in 0..batch.batch {
            for r in 0..returned {
                let s = b * returned + r;
                if out.trajectory[s] == u32::MAX {
                    continue;
                }
                let mut found = n;
                for t in b * k..(b + 1) * k {
                    if words.ranks[t] == r as u32 {
                        found = t;
                    }
                }
                if found >= n {
                    continue;
                }
                out.evidence_status[s] = batch.evidence_status[found];
                out.evidence_count[s] = batch.evidence_count[found];
                for q in 0..ecap {
                    out.evidence_peak_id[s * ecap + q] =
                        batch.evidence_peak_id[found * ecap + q];
                    out.evidence_hypothesis[s * ecap + q] =
                        batch.evidence_hypothesis[found * ecap + q];
                    out.evidence_shift[s * ecap + q] = batch.evidence_shift[found * ecap + q];
                    out.evidence_residual[s * ecap + q] =
                        batch.evidence_residual[found * ecap + q];
                    out.evidence_log_prob[s * ecap + q] =
                        batch.evidence_log_prob[found * ecap + q];
                }
            }
        }
    }
    out.validate()?;
    Ok(out)
}
