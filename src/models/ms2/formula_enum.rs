//! Bounded closed-form-hydrogen enumeration over compositions (P4.9 host reference).
//!
//! This module compares the indexed [`FormulaTable`](super::formula::FormulaTable)
//! (contract §9) with enumeration that solves for hydrogen in closed form.
//! Pure host Rust with integer masses only: no tensors, no kernels, no device code.
//!
//! * [`EnumDomain`] bounds the search: per heavy element a maximum count (the 9
//!   non-hydrogen elements of contracts §4.1 in the crate's element order), a
//!   maximum heavy-atom total, hydrogen bounds and a version string.
//! * [`enumerate`] runs the search for an [`EnumQuery`] under [`EnumLimits`].
//!   The query carries exactly what [`WindowQuery`](super::formula::WindowQuery)
//!   carries (precursor m/z, precursor uncertainty, adduct, tolerance in tenths
//!   of a ppm) and reuses the same parent-mass derivation ([`parent_mass`]),
//!   tolerance function ([`tolerance`]) and §5 decision rule ([`decide`]) as the
//!   table path; nothing is re-implemented.
//!
//! Algorithm:
//!
//! 1. Depth-first enumeration of heavy-atom count vectors in a fixed
//!    documented order. Without ratio bounds the order is heaviest element
//!    first (I, Br, Cl, S, P, F, O, N, C). With [`RatioBounds`] (see
//!    [`EnumLimits::ratio`]) carbon goes first (C, I, Br, Cl, S, P, F, O, N),
//!    so the carbon-conditional caps bound every later element. A branch is
//!    pruned when its heavy mass already exceeds the upper superset bound.
//!    Masses are positive, so this prune is exact: any completion only adds
//!    mass. Counts at each level run ascending from 0, so the first overweight
//!    count ends the level. With ratio bounds a count is additionally skipped
//!    (`continue`, counted in `pruned_ratio_cap` / `pruned_rare`) when a
//!    carbon- or heavy-total-bucketed maximum of stage (i) or a rare-atom
//!    maximum of stage (iii) is already violated; skips do not end the level
//!    because bucketed maxima need not be monotone in the count.
//! 2. For each complete heavy vector, hydrogen is solved in closed form: with
//!    `lo`/`hi` the superset window ends, the integer hydrogen counts `h` whose
//!    total mass can fall in the window are the integers in
//!    `[ceil((lo − heavy) / m_H), floor((hi − heavy) / m_H)]`, computed with
//!    integer arithmetic (`i128`, so no overflow is possible). Every integer in
//!    that range is checked (there may be 0, 1 or more) with the exact §5
//!    verdict using that composition's own arithmetic bound (`E_arith` from the
//!    residuals, plus the bound of the query, exactly as the table path adds
//!    `E_row + bound`). Accepted and ambiguous compositions are joined
//!    (ambiguous flagged); rejected ones are not.
//! 3. Chemical filters, each separately switchable in [`EnumLimits`] and each
//!    counted when it rejects, applied in order after the verdict:
//!    (a) hydrogen within `[0, h_max(heavy)]`, (b) the parity/integer-DBE rule,
//!    (c) DBE `>= 0`. Only rules exact for the V0 chemistry domain are used;
//!    no heuristic ratio rules.
//! 4. Train-fit pruning stages ([`RatioBounds`]), only when
//!    [`EnumLimits::ratio`] is present, applied after the exact filters in a
//!    fixed documented order, each with its own reject counter, so recall is
//!    reportable before and after each stage: the (i)/(iii) completion
//!    refinement of the DFS caps (`rejected_ratio_cap`, `rejected_rare`; the
//!    DFS skips themselves are `pruned_ratio_cap` / `pruned_rare`), then the
//!    six (ii) ratios H/C, N/C, O/C, (F+Cl+Br+I)/C, S/C, P/C
//!    (`rejected_ratio_hc/nc/oc/hal/s/p`, integer cross-multiplication, no
//!    floats), then the (iv) DBE-per-heavy-bucket range
//!    (`rejected_ratio_dbe`).
//!
//! Saturated-acyclic hydrogen bound. A heavy atom of maximum valence `v` carries
//! at most `v` hydrogens when isolated and spends one valence stub per
//! heavy–heavy single bond. A connected acyclic heavy skeleton of `n >= 1`
//! atoms has `n − 1` single bonds using `2 * (n − 1)` stubs, so
//!
//! ```text
//! h_max(heavy) = sum_e v_max(e) * n_e − 2 * n_heavy + 2,
//! ```
//!
//! clamped at zero (a negative value means no connected saturated composition
//! exists; the DBE filter then rejects the rest). Per-element maximum valences
//! are derived from the atom-type table in the code ([`max_valence`]):
//! C 4, N 3, O 2, F 1, P 5, S 6, Cl 1, Br 1, I 1. Using the maximum (S 6 covers
//! the v2 state) keeps the bound sound: no valid composition is rejected.
//!
//! DBE rule. For a neutral closed-shell molecule every bond order consumes two
//! valence stubs, so the stub total `sum_e v(e) * n_e + h` is even, and with the
//! maximum valences
//!
//! ```text
//! 2 * DBE = 2 + sum_e (v_max(e) − 2) * n_e − h.
//! ```
//!
//! Filter (b) requires this numerator to be even (equivalently, `sum v*n + h`
//! even); filter (c) requires it to be non-negative. Both use the maximum
//! valences, so any composition admittable under a lower valence state (S v2)
//! has a numerator between the computed one and the true one and can only be
//! accepted more readily here: sound, never rejecting a valid composition.
//!
//! Work accounting mirrors contracts §9: `nodes_visited` counts every heavy
//! vector prefix expanded (one per depth-first call, including the root and
//! every complete vector); `hydrogen_checks` counts every integer hydrogen
//! count for which the §5 verdict runs; `rows_joined` counts joined rows before
//! any cap; `rows_scored` counts the kept output rows, the first
//! `min(scored_max, capacity)` joined rows in the canonical output order.
//! Per-filter reject counts record verdict-passing compositions each filter
//! removes, so recall can be reported before and after each pruning stage;
//! `pruned_ratio_cap` / `pruned_rare` count DFS branches (count values)
//! skipped by the ratio caps before any hydrogen work.
//!
//! Capacity keeps the canonical prefix: every join is counted in
//! `rows_joined`, while a bounded max-heap retains only the `capacity`
//! smallest `(mass, composition)` rows, so memory stays O(capacity) and the
//! kept output equals the first `capacity` rows of the uncapped canonical
//! output. Under `nodes_visited_max` exhaustion the joins stop part-way
//! through the fixed enumeration order; the kept rows are then the canonical
//! prefix of the rows joined before the stop, which is deterministic (the
//! same inputs always stop at the same node and keep the same rows).
//!
//! Stage recall for a gold composition (window, exact-mass verdict, exact
//! filters) is computed by [`gold_stages`], shared with the report so the
//! unknown-precision sentinel and invalid parent masses contribute to no
//! recall stage while staying in the denominator.
//!
//! Statuses mirror [`FormulaTable::window`](super::formula::FormulaTable::window):
//! `exhausted` when `nodes_visited_max`, the join capacity or `scored_max` is
//! hit (deterministic: the same inputs always stop at the same node and keep
//! the same rows, because the enumeration order is fixed); `absent` only when
//! the search completed and joined nothing; `support_complete` only when it
//! completed and every joined row was scored; `mass_overflow` and
//! `exact_mass_unavailable` exactly as the table reports them.
//!
//! Output order is canonical and independent of the enumeration order: sorted
//! by integer mass, then by element counts (the table's sort), so results are
//! comparable with [`FormulaTable::window`](super::formula::FormulaTable::window)
//! row for row.

// The device-order kernel twins below mirror what the CubeCL IR can express
// (same reason as `tensor/ops/ms2.rs`' collapsible-if allow): nested
// statement `if`s instead of `&&` (short-circuiting cannot be relied on
// where a guard keeps a division exact), and guarded divisions/remainders
// instead of `checked_div` / `is_multiple_of` (neither exists on the device).
#![allow(clippy::collapsible_if)]
#![allow(clippy::manual_checked_ops)]
#![allow(clippy::manual_is_multiple_of)]

use std::collections::BinaryHeap;

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

use super::chem::{
    ATOM_TYPES, Composition, ELEMENTS, HYDROGEN, composition_error_nda, decide, parent_mass,
    tolerance_u32,
};
use super::contract::{SpectrumBatch, request_status};
use super::formula::WindowQuery;

/// Heavy elements in the crate's element order: C, N, O, F, P, S, Cl, Br, I.
///
/// Index `i` of [`EnumDomain::heavy_caps`] bounds [`HEAVY_ELEMENTS[i]`].
pub const HEAVY_ELEMENTS: [usize; 9] = [0, 2, 3, 4, 5, 6, 7, 8, 9];

/// Depth-first enumeration order over [`HEAVY_ELEMENTS`] positions, heaviest
/// element first by integer mass: I, Br, Cl, S, P, F, O, N, C.
const DFS_ORDER: [usize; 9] = [8, 7, 6, 5, 4, 3, 2, 1, 0];

/// Depth-first enumeration order with [`RatioBounds`]: carbon first, then
/// heaviest first (C, I, Br, Cl, S, P, F, O, N), so the carbon-conditional
/// caps bound every later element. The scored output order is unchanged
/// (canonical mass/count sort); only the traversal order differs.
const DFS_ORDER_CARBON_FIRST: [usize; 9] = [0, 8, 7, 6, 5, 4, 3, 2, 1];

/// Version string stamped on bounds built here.
///
/// `ms2-ratio-v2` tracks carbon/heavy bucket presence explicitly: a bucket
/// the fit never saw admits nothing, whatever the margin. `ms2-ratio-v1`
/// JSON (without the presence fields) does not parse as v2.
pub const RATIO_BOUNDS_VERSION: &str = "ms2-ratio-v2";

/// Carbon-count bucket width of [`RatioBounds`]: bucket `b` holds carbon
/// counts `4*b ..= 4*b + 3`.
pub const RATIO_CARBON_BUCKET: u16 = 4;

/// Heavy-atom-total bucket width of [`RatioBounds`]: bucket `b` holds totals
/// `4*b ..= 4*b + 3`.
pub const RATIO_HEAVY_BUCKET: u16 = 4;

/// Heavy element ids treated as "rare" heteroatoms for [`RatioBounds`]:
/// F, P, S, Cl, Br, I, read as "everything except C, N, O" among the heavy
/// heteroatoms. Hydrogen is excluded by construction: it is solved in closed
/// form and bounded by the exact hydrogen ceiling instead.
pub const RARE_ELEMENTS: [usize; 6] = [4, 5, 6, 7, 8, 9];

/// Positions of [`RARE_ELEMENTS`] inside [`HEAVY_ELEMENTS`].
const RARE_POS: [usize; 6] = [3, 4, 5, 6, 7, 8];

/// Ratio features of [`RatioBounds`] in stage order: H/C, N/C, O/C,
/// (F+Cl+Br+I)/C, S/C, P/C.
pub const RATIO_FEATURES: [&str; 6] = ["H/C", "N/C", "O/C", "hal/C", "S/C", "P/C"];

/// Maximum rare-element combinations of [`rare_table`] (spec §1.4 `P_max`).
pub const RARE_TABLE_MAX: usize = 16_384;

/// Largest element cap the device arithmetic relies on (spec §1.4).
pub const DEVICE_CAP_MAX: u16 = 255;

/// Largest hydrogen bound the device arithmetic relies on (spec §1.4).
pub const DEVICE_HYDROGEN_MAX: u16 = 1023;

/// Largest ratio numerator or denominator the device arithmetic relies on
/// (spec §1.4: every ratio numerator and denominator `<= 2^20`).
pub const DEVICE_RATIO_MAX: u32 = 1 << 20;

/// Largest bucket-table row count the device arithmetic relies on (spec §1.4).
pub const DEVICE_BUCKETS_MAX: usize = 64;

/// Scope restriction of spec §1.4: a spectrum is searched only when
/// `half <= 1,511,737` (`floor(3 * m_H / 2)` with `m_H = 1,007,825`), which
/// bounds the hydrogen range of a heavy vector to 4 integers. A wider window
/// is `formula_search_exhausted` with nothing joined. `half` is formed with
/// saturating additions, so an unknown or huge uncertainty lands there.
pub const DEVICE_HALF_MAX: u32 = 1_511_737;

/// Version string stamped on domains built here.
pub const ENUM_DOMAIN_VERSION: &str = "ms2-enum-v1";

/// Maximum valence of an element over the V0 atom types ([`ATOM_TYPES`]).
///
/// Hydrogen (which has no atom type; hydrogens are counts on heavy atoms)
/// reports 1. Every other element reports the maximum valence among its atom
/// types: C 4, N 3, O 2, F 1, P 5, S 6, Cl 1, Br 1, I 1.
pub fn max_valence(element: usize) -> u8 {
    if element == HYDROGEN {
        return 1;
    }
    ATOM_TYPES
        .iter()
        .filter(|t| t.element == element)
        .map(|t| t.valence)
        .max()
        .unwrap_or(0)
}

/// Saturated-acyclic hydrogen ceiling of a composition.
///
/// `sum_e v_max(e) * n_e − 2 * n_heavy + 2` for `n_heavy >= 1`, clamped at
/// zero; `0` when there is no heavy atom (hydrogen-only compositions never
/// join). The `u16` ceiling saturates at `u16::MAX`: a ceiling above the
/// representable range accepts every `u16` hydrogen count, which is sound.
pub fn hydrogen_ceiling(c: &Composition) -> Result<u16> {
    let mut heavy_n: u64 = 0;
    let mut stubs: u64 = 0;
    for e in HEAVY_ELEMENTS.iter() {
        let n = u64::from(c[*e]);
        heavy_n = heavy_n
            .checked_add(n)
            .ok_or_else(|| Error::config("hydrogen_ceiling: heavy atom total overflows u64"))?;
        let add = n
            .checked_mul(u64::from(max_valence(*e)))
            .ok_or_else(|| Error::config("hydrogen_ceiling: valence stubs overflow u64"))?;
        stubs = stubs
            .checked_add(add)
            .ok_or_else(|| Error::config("hydrogen_ceiling: valence stubs overflow u64"))?;
    }
    if heavy_n == 0 {
        return Ok(0);
    }
    let twice = heavy_n
        .checked_mul(2)
        .ok_or_else(|| Error::config("hydrogen_ceiling: heavy atom total overflows u64"))?;
    let ceiling = (stubs as i128) + 2 - (twice as i128);
    if ceiling <= 0 {
        return Ok(0);
    }
    if ceiling > u128::from(u16::MAX) as i128 {
        return Ok(u16::MAX);
    }
    Ok(ceiling as u16)
}

/// Twice the double-bond equivalent of a composition under maximum valences.
///
/// `2 + sum_e (v_max(e) − 2) * n_e − h` as `i64`. Filter (b) requires it even;
/// filter (c) requires it non-negative.
pub fn dbe_twice(c: &Composition) -> Result<i64> {
    let mut total: i64 = 2;
    for e in HEAVY_ELEMENTS {
        let v = i64::from(max_valence(e)) - 2;
        let n = i64::from(c[e]);
        let add = v
            .checked_mul(n)
            .ok_or_else(|| Error::config("dbe_twice: valence term overflows i64"))?;
        total = total
            .checked_add(add)
            .ok_or_else(|| Error::config("dbe_twice: valence sum overflows i64"))?;
    }
    total = total
        .checked_sub(i64::from(c[HYDROGEN]))
        .ok_or_else(|| Error::config("dbe_twice: hydrogen term overflows i64"))?;
    Ok(total)
}

/// Filter (b): the parity/integer-DBE rule for a neutral closed-shell molecule.
///
/// True exactly when [`dbe_twice`] is even.
pub fn passes_parity(c: &Composition) -> Result<bool> {
    Ok(dbe_twice(c)? % 2 == 0)
}

/// Filter (c): DBE `>= 0` under maximum valences.
///
/// True exactly when [`dbe_twice`] is non-negative. Call after [`passes_parity`]
/// so the two stages stay separable in the reject counts.
pub fn passes_dbe(c: &Composition) -> Result<bool> {
    Ok(dbe_twice(c)? >= 0)
}

/// Filter (a): hydrogen within `[0, h_max(heavy)]` of [`hydrogen_ceiling`].
pub fn passes_hydrogen_ceiling(c: &Composition) -> Result<bool> {
    Ok(c[HYDROGEN] <= hydrogen_ceiling(c)?)
}

/// Bounded search domain over compositions.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EnumDomain {
    /// Domain version string ([`ENUM_DOMAIN_VERSION`] when built here).
    pub version: String,
    /// Maximum count per heavy element in [`HEAVY_ELEMENTS`] order
    /// (C, N, O, F, P, S, Cl, Br, I).
    pub heavy_caps: [u16; 9],
    /// Maximum heavy-atom total over all 9 heavy elements.
    pub heavy_max: u16,
    /// Minimum hydrogen count (0 from [`EnumDomain::from_compositions`]).
    pub hydrogen_min: u16,
    /// Maximum hydrogen count.
    pub hydrogen_max: u16,
}

impl EnumDomain {
    /// Derive a dataset-justified domain from observed compositions.
    ///
    /// Each cap is the observed maximum plus `margin` (checked: overflow is an
    /// error, never wrapping); `heavy_max` is the observed maximum heavy-atom
    /// total plus `margin`; `hydrogen_min` is 0 and `hydrogen_max` the observed
    /// maximum hydrogen count plus `margin`. An empty input observes all zeros.
    pub fn from_compositions(
        items: impl IntoIterator<Item = Composition>,
        margin: u16,
    ) -> Result<Self> {
        let mut caps: [u16; 9] = [0; 9];
        let mut heavy_max: u16 = 0;
        let mut hydrogen_max: u16 = 0;
        for c in items {
            let mut total: u32 = 0;
            for (i, e) in HEAVY_ELEMENTS.iter().enumerate() {
                caps[i] = caps[i].max(c[*e]);
                total += u32::from(c[*e]);
            }
            if total <= u32::from(u16::MAX) {
                heavy_max = heavy_max.max(total as u16);
            } else {
                return Err(Error::config(
                    "EnumDomain::from_compositions: heavy atom total exceeds u16".to_string(),
                ));
            }
            hydrogen_max = hydrogen_max.max(c[HYDROGEN]);
        }
        let mut heavy_caps: [u16; 9] = [0; 9];
        for (i, cap) in caps.iter().enumerate() {
            heavy_caps[i] = cap.checked_add(margin).ok_or_else(|| {
                Error::config(format!(
                    "EnumDomain::from_compositions: heavy cap {cap} plus margin {margin} overflows u16"
                ))
            })?;
        }
        let heavy_max = heavy_max.checked_add(margin).ok_or_else(|| {
            Error::config(format!(
                "EnumDomain::from_compositions: heavy total {heavy_max} plus margin {margin} overflows u16"
            ))
        })?;
        let hydrogen_max = hydrogen_max.checked_add(margin).ok_or_else(|| {
            Error::config(format!(
                "EnumDomain::from_compositions: hydrogen max {hydrogen_max} plus margin {margin} overflows u16"
            ))
        })?;
        Ok(Self {
            version: ENUM_DOMAIN_VERSION.to_string(),
            heavy_caps,
            heavy_max,
            hydrogen_min: 0,
            hydrogen_max,
        })
    }

    /// Whether a composition lies inside the domain: every heavy count within
    /// its cap, the heavy total within `heavy_max`, hydrogen within bounds.
    pub fn contains(&self, c: &Composition) -> bool {
        let mut total: u32 = 0;
        for (i, e) in HEAVY_ELEMENTS.iter().enumerate() {
            if c[*e] > self.heavy_caps[i] {
                return false;
            }
            total += u32::from(c[*e]);
        }
        if total > u32::from(self.heavy_max) {
            return false;
        }
        if c[HYDROGEN] < self.hydrogen_min || c[HYDROGEN] > self.hydrogen_max {
            return false;
        }
        true
    }

    /// Largest per-composition arithmetic bound over the domain:
    /// `ceil((sum caps * residual + hydrogen_max * residual_H) / 1000)`.
    ///
    /// Saturating (never wrapping); the true sums are far inside `u64`
    /// (at most ~3e8 nano-dalton), so saturation never engages.
    pub fn max_error(&self) -> u32 {
        let mut nda: u64 = 0;
        for (i, e) in HEAVY_ELEMENTS.iter().enumerate() {
            nda = nda.saturating_add(
                (u64::from(self.heavy_caps[i]))
                    .saturating_mul(u64::from(ELEMENTS[*e].residual_nda)),
            );
        }
        nda = nda.saturating_add(
            u64::from(self.hydrogen_max).saturating_mul(u64::from(ELEMENTS[HYDROGEN].residual_nda)),
        );
        nda.div_ceil(1000).min(u64::from(u32::MAX)) as u32
    }

    /// Serialized size in bytes (the JSON form below).
    pub fn bytes(&self) -> usize {
        self.to_json().len()
    }

    /// Serialize the domain (`version`, caps, totals, hydrogen bounds).
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| {
            // The struct is plain data; serialization cannot fail. The
            // fallback keeps this infallible without `unwrap()`.
            String::from("{}")
        })
    }

    /// Parse [`EnumDomain::to_json`], rejecting `hydrogen_min > hydrogen_max`.
    pub fn from_json(text: &str) -> Result<Self> {
        let domain: Self = serde_json::from_str(text)?;
        if domain.hydrogen_min > domain.hydrogen_max {
            return Err(Error::config(format!(
                "EnumDomain::from_json: hydrogen_min {} exceeds hydrogen_max {}",
                domain.hydrogen_min, domain.hydrogen_max
            )));
        }
        Ok(domain)
    }
}

/// Train-fit pruning bounds over compositions (plan P4.3/P4.9).
///
/// What is fitted, by [`RatioBounds::fit`] over the given TRAIN compositions
/// only (min/max of the observed train values, each widened by `margin`):
///
/// * (i) per heavy element, the maximum count as a step function of the
///   carbon-count bucket (`max_by_carbon[carbon / carbon_bucket]`, buckets of
///   [`RATIO_CARBON_BUCKET`] carbons) and, separately, of the heavy-atom-total
///   bucket (`max_by_heavy[heavy_total / heavy_bucket]`). Each stored maximum
///   is the observed maximum plus `margin` (saturating at `u16::MAX`). There
///   is no minimum: only upper bounds. Each bucket records whether the fit
///   saw it (`carbon_seen` / `heavy_seen`): a bucket the fit never saw —
///   interior or past the tables — admits NOTHING, whatever the margin
///   ([`RatioBounds::passes_cap`] is false there); the margin widens observed
///   maxima only.
/// * (ii) the ratios H/C, N/C, O/C, (F+Cl+Br+I)/C, S/C, P/C
///   ([`RATIO_FEATURES`]) as observed minimum/maximum fractions over the
///   compositions with carbon `> 0`. Each bound is the extremal train fraction
///   `(num, den)`; the fitted lower numerator is `num − margin` (saturating)
///   and the fitted upper numerator `num + margin` (saturating), denominators
///   unchanged. When several train fractions tie as real numbers, the stored
///   extremal fraction is the one with the smallest denominator (hence the
///   smallest numerator among the tied observations with that denominator),
///   so the fitted bound is a function of the composition multiset and does
///   not depend on the order the compositions were fitted in. Decisions use
///   integer cross-multiplication only, no floats:
///   `n_x * den_lo >= num_lo * n_c` and `n_x * den_hi <= num_hi * n_c`.
/// * (iii) the total count of rare heteroatoms ([`RARE_ELEMENTS`]) and the
///   number of distinct rare elements present, each as `[min, max]` over
///   train (`min − margin` saturating, `max + margin` saturating).
/// * (iv) [`dbe_twice`] as `[min, max]` per heavy-atom-total bucket
///   (`dbe_by_heavy[heavy_total / heavy_bucket]`; `min − margin`, `max +
///   margin` in `i64`, saturating).
///
/// Compositions with zero carbons are handled explicitly: the (ii) ratios
/// divide by the carbon count, so they pass exactly when the fit observed at
/// least one zero-carbon composition (`zero_carbon_seen`); every other stage
/// applies to them normally. Measured on the V0 exports: the pilot train
/// export holds no zero-carbon molecule (carbon range 3–51), while the larger
/// scale train export holds one (carbon range 0–60); with bounds fitted on
/// the pilot train, zero-carbon candidates fail the ratio stages.
///
/// An empty fit observes nothing: every bucket table is empty, no carbon was
/// seen, and the rare ranges are `[0, 0]`; every composition then fails at
/// least one stage (no train support), which is the honest reading.
///
/// Buckets past the fitted tables (heavier than train) fail the corresponding
/// stage, and so does every interior bucket the fit never saw: there is no
/// train support there, and the recall cost is measured. With margin 0 an
/// unseen interior row holds all-zero caps, so only the (never joining)
/// all-zero heavy vector would pass it; the explicit presence check makes
/// the rejection exact for every margin.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RatioBounds {
    /// Bounds version string ([`RATIO_BOUNDS_VERSION`] when built here).
    pub version: String,
    /// Carbon-count bucket width ([`RATIO_CARBON_BUCKET`] when built here).
    pub carbon_bucket: u16,
    /// Heavy-atom-total bucket width ([`RATIO_HEAVY_BUCKET`] when built here).
    pub heavy_bucket: u16,
    /// Widening margin the fit applied (see the struct docs).
    pub margin: u16,
    /// (i) per heavy element (in [`HEAVY_ELEMENTS`] order) the maximum count
    /// for each carbon-count bucket.
    pub max_by_carbon: Vec<[u16; 9]>,
    /// (i) whether the fit saw each carbon-count bucket (same length as
    /// [`RatioBounds::max_by_carbon`]). Unseen buckets admit nothing,
    /// whatever the margin.
    pub carbon_seen: Vec<bool>,
    /// (i) per heavy element the maximum count for each heavy-atom-total
    /// bucket.
    pub max_by_heavy: Vec<[u16; 9]>,
    /// (i)/(iv) whether the fit saw each heavy-atom-total bucket (same
    /// length as [`RatioBounds::max_by_heavy`] and
    /// [`RatioBounds::dbe_by_heavy`]). Unseen buckets admit nothing,
    /// whatever the margin.
    pub heavy_seen: Vec<bool>,
    /// (ii) lower-bound numerators of [`RATIO_FEATURES`].
    pub ratio_lo_num: [u32; 6],
    /// (ii) lower-bound denominators of [`RATIO_FEATURES`] (never 0 when
    /// [`RatioBounds::has_carbon`] holds).
    pub ratio_lo_den: [u32; 6],
    /// (ii) upper-bound numerators of [`RATIO_FEATURES`].
    pub ratio_hi_num: [u32; 6],
    /// (ii) upper-bound denominators of [`RATIO_FEATURES`].
    pub ratio_hi_den: [u32; 6],
    /// Whether the fit saw a composition with carbon `> 0`.
    pub has_carbon: bool,
    /// Whether the fit saw a zero-carbon composition.
    pub zero_carbon_seen: bool,
    /// (iii) `[min, max]` of the rare-heteroatom total.
    pub rare_total: [u32; 2],
    /// (iii) `[min, max]` of the distinct rare-element count.
    pub rare_distinct: [u16; 2],
    /// (iv) `[min, max]` of [`dbe_twice`] per heavy-atom-total bucket.
    pub dbe_by_heavy: Vec<[i64; 2]>,
}

/// Numerators of [`RATIO_FEATURES`] for a composition: H, N, O, F+Cl+Br+I,
/// S, P. The halogen sum is at most `4 * u16::MAX`, so `u32` never overflows.
fn ratio_numerators(c: &Composition) -> [u32; 6] {
    let hal = u32::from(c[4]) + u32::from(c[7]) + u32::from(c[8]) + u32::from(c[9]);
    [
        u32::from(c[HYDROGEN]),
        u32::from(c[2]),
        u32::from(c[3]),
        hal,
        u32::from(c[6]),
        u32::from(c[5]),
    ]
}

/// Heavy-atom total of a composition as `u64` (at most `9 * u16::MAX`).
fn heavy_total_of(c: &Composition) -> u64 {
    let mut total: u64 = 0;
    for e in HEAVY_ELEMENTS {
        total += u64::from(c[e]);
    }
    total
}

/// Rare-heteroatom total and distinct-element count of a composition.
/// The total is at most `6 * u16::MAX`, so `u32` never overflows.
fn rare_of(c: &Composition) -> (u32, u16) {
    let mut total: u32 = 0;
    let mut distinct: u16 = 0;
    for e in RARE_ELEMENTS {
        total += u32::from(c[e]);
        if c[e] > 0 {
            distinct += 1;
        }
    }
    (total, distinct)
}

impl RatioBounds {
    /// Derive bounds from TRAIN compositions with widening `quantile_margin`.
    ///
    /// Exactly what is fitted is stated on the struct docs; every observed
    /// maximum is the observed train maximum plus `quantile_margin`
    /// (saturating), every minimum the observed minimum minus
    /// `quantile_margin` (saturating), ratio numerators widened likewise with
    /// denominators unchanged (ties broken by smallest denominator, so the
    /// result is independent of the input order), and DBE ranges widened in
    /// `i64`. Buckets the input never saw are recorded as unseen and admit
    /// nothing, whatever the margin. With `quantile_margin == 0` every fitted
    /// composition passes every stage by construction.
    pub fn fit(
        items: impl IntoIterator<Item = Composition>,
        quantile_margin: u16,
    ) -> Result<Self> {
        let carbon_width = u64::from(RATIO_CARBON_BUCKET);
        let heavy_width = u64::from(RATIO_HEAVY_BUCKET);
        let mut max_by_carbon: Vec<[u16; 9]> = Vec::new();
        let mut carbon_seen: Vec<bool> = Vec::new();
        let mut max_by_heavy: Vec<[u16; 9]> = Vec::new();
        let mut heavy_seen: Vec<bool> = Vec::new();
        let mut lo_num: [u32; 6] = [0; 6];
        let mut lo_den: [u32; 6] = [1; 6];
        let mut hi_num: [u32; 6] = [0; 6];
        let mut hi_den: [u32; 6] = [1; 6];
        let mut ratio_seen = [false; 6];
        let mut has_carbon = false;
        let mut zero_carbon_seen = false;
        let mut rare_min: u32 = u32::MAX;
        let mut rare_max: u32 = 0;
        let mut distinct_min: u16 = u16::MAX;
        let mut distinct_max: u16 = 0;
        let mut dbe_by_heavy: Vec<[i64; 2]> = Vec::new();
        let mut any = false;
        for c in items {
            any = true;
            let carbon = c[0];
            let heavy = heavy_total_of(&c);
            if carbon > 0 {
                has_carbon = true;
            } else {
                zero_carbon_seen = true;
            }
            let carbon_bucket = (u64::from(carbon) / carbon_width) as usize;
            while max_by_carbon.len() <= carbon_bucket {
                max_by_carbon.push([0; 9]);
                carbon_seen.push(false);
            }
            carbon_seen[carbon_bucket] = true;
            let heavy_bucket = (heavy / heavy_width) as usize;
            while max_by_heavy.len() <= heavy_bucket {
                max_by_heavy.push([0; 9]);
                heavy_seen.push(false);
            }
            heavy_seen[heavy_bucket] = true;
            while dbe_by_heavy.len() <= heavy_bucket {
                dbe_by_heavy.push([i64::MAX, i64::MIN]);
            }
            for (i, e) in HEAVY_ELEMENTS.iter().enumerate() {
                if c[*e] > max_by_carbon[carbon_bucket][i] {
                    max_by_carbon[carbon_bucket][i] = c[*e];
                }
                if c[*e] > max_by_heavy[heavy_bucket][i] {
                    max_by_heavy[heavy_bucket][i] = c[*e];
                }
            }
            if carbon > 0 {
                let nums = ratio_numerators(&c);
                for (k, n) in nums.iter().enumerate() {
                    // Fractions compare by cross-multiplication in `u64`:
                    // `n <= 4 * u16::MAX`, `den <= u16::MAX`, so no product
                    // exceeds `2^34`. Ties (equal fractions as real numbers)
                    // keep the fraction with the smallest denominator, so the
                    // stored bound is a function of the composition multiset
                    // and `fit` is independent of the input order, including
                    // under a positive margin (the margin widens the stored
                    // numerator only).
                    let carbon32 = u32::from(carbon);
                    if !ratio_seen[k] {
                        lo_num[k] = *n;
                        lo_den[k] = carbon32;
                        hi_num[k] = *n;
                        hi_den[k] = carbon32;
                    } else {
                        let left_lo = u64::from(*n) * u64::from(lo_den[k]);
                        let right_lo = u64::from(lo_num[k]) * u64::from(carbon);
                        if left_lo < right_lo
                            || (left_lo == right_lo && carbon32 < lo_den[k])
                        {
                            lo_num[k] = *n;
                            lo_den[k] = carbon32;
                        }
                        let left_hi = u64::from(*n) * u64::from(hi_den[k]);
                        let right_hi = u64::from(hi_num[k]) * u64::from(carbon);
                        if left_hi > right_hi
                            || (left_hi == right_hi && carbon32 < hi_den[k])
                        {
                            hi_num[k] = *n;
                            hi_den[k] = carbon32;
                        }
                    }
                    ratio_seen[k] = true;
                }
            }
            let (rare, distinct) = rare_of(&c);
            rare_min = rare_min.min(rare);
            rare_max = rare_max.max(rare);
            distinct_min = distinct_min.min(distinct);
            distinct_max = distinct_max.max(distinct);
            let twice = dbe_twice(&c)?;
            let range = &mut dbe_by_heavy[heavy_bucket];
            range[0] = range[0].min(twice);
            range[1] = range[1].max(twice);
        }
        if !any {
            rare_min = 0;
            distinct_min = 0;
        }
        // The margin widens observed buckets only: unseen interior buckets
        // keep all-zero caps and admit nothing, whatever the margin.
        for (row, seen) in max_by_carbon.iter_mut().zip(carbon_seen.iter()) {
            if *seen {
                for cap in row.iter_mut() {
                    *cap = cap.saturating_add(quantile_margin);
                }
            }
        }
        for (row, seen) in max_by_heavy.iter_mut().zip(heavy_seen.iter()) {
            if *seen {
                for cap in row.iter_mut() {
                    *cap = cap.saturating_add(quantile_margin);
                }
            }
        }
        let margin32 = u32::from(quantile_margin);
        for k in 0..6 {
            lo_num[k] = lo_num[k].saturating_sub(margin32);
            hi_num[k] = hi_num[k].saturating_add(margin32);
        }
        for range in dbe_by_heavy.iter_mut() {
            if range[0] != i64::MAX {
                range[0] = range[0].saturating_sub(i64::from(quantile_margin));
            }
            if range[1] != i64::MIN {
                range[1] = range[1].saturating_add(i64::from(quantile_margin));
            }
        }
        Ok(Self {
            version: RATIO_BOUNDS_VERSION.to_string(),
            carbon_bucket: RATIO_CARBON_BUCKET,
            heavy_bucket: RATIO_HEAVY_BUCKET,
            margin: quantile_margin,
            max_by_carbon,
            carbon_seen,
            max_by_heavy,
            heavy_seen,
            ratio_lo_num: lo_num,
            ratio_lo_den: lo_den,
            ratio_hi_num: hi_num,
            ratio_hi_den: hi_den,
            has_carbon,
            zero_carbon_seen,
            rare_total: [
                rare_min.saturating_sub(margin32),
                rare_max.saturating_add(margin32),
            ],
            rare_distinct: [
                distinct_min.saturating_sub(quantile_margin),
                distinct_max.saturating_add(quantile_margin),
            ],
            dbe_by_heavy,
        })
    }

    /// (i) carbon- and heavy-total-bucketed maximum counts hold.
    ///
    /// False when a bucket lies past the fitted tables or was never seen in
    /// the fit (no train support there, whatever the margin), and on a
    /// hand-built degenerate bound (zero widths, or presence/table length
    /// mismatch).
    pub fn passes_cap(&self, c: &Composition) -> bool {
        if self.carbon_bucket == 0 || self.heavy_bucket == 0 {
            return false;
        }
        let carbon_bucket =
            (u64::from(c[0]) / u64::from(self.carbon_bucket)) as usize;
        let heavy_bucket =
            (heavy_total_of(c) / u64::from(self.heavy_bucket)) as usize;
        if self.carbon_seen.get(carbon_bucket) != Some(&true) {
            return false;
        }
        if self.heavy_seen.get(heavy_bucket) != Some(&true) {
            return false;
        }
        let (Some(carbon_row), Some(heavy_row)) = (
            self.max_by_carbon.get(carbon_bucket),
            self.max_by_heavy.get(heavy_bucket),
        ) else {
            return false;
        };
        for (i, e) in HEAVY_ELEMENTS.iter().enumerate() {
            if c[*e] > carbon_row[i] || c[*e] > heavy_row[i] {
                return false;
            }
        }
        true
    }

    /// Largest heavy-bucketed maximum for heavy position `pos` over buckets
    /// `from_bucket..`, or 0 when no fitted bucket remains (no train support
    /// for heavier completions). Used to prune DFS prefixes whose final total
    /// is not yet known; it is an upper bound of the exact final check, so
    /// pruning on it is sound, and the leaf recheck makes it exact.
    fn heavy_cap_from(&self, pos: usize, from_bucket: usize) -> u16 {
        self.max_by_heavy
            .iter()
            .skip(from_bucket)
            .map(|row| row[pos])
            .max()
            .unwrap_or(0)
    }

    /// (iii) rare-heteroatom total and distinct count within their maxima.
    /// The minima are leaf-checked ([`RatioBounds::passes_rare_min`]); this
    /// maximum half is what prunes the DFS.
    pub fn passes_rare_max(&self, c: &Composition) -> bool {
        let (rare, distinct) = rare_of(c);
        rare <= self.rare_total[1] && distinct <= self.rare_distinct[1]
    }

    /// (iii) rare-heteroatom total and distinct count within their minima.
    pub fn passes_rare_min(&self, c: &Composition) -> bool {
        let (rare, distinct) = rare_of(c);
        rare >= self.rare_total[0] && distinct >= self.rare_distinct[0]
    }

    /// (ii) one ratio feature of [`RATIO_FEATURES`] within its fitted
    /// interval, by integer cross-multiplication (no floats). A zero-carbon
    /// composition passes exactly when the fit saw zero-carbon train
    /// compositions; a zero denominator (a degenerate bound) fails.
    pub fn passes_ratio(&self, feature: usize, c: &Composition) -> bool {
        if feature >= RATIO_FEATURES.len() {
            return false;
        }
        if c[0] == 0 {
            return self.zero_carbon_seen;
        }
        let n = u64::from(ratio_numerators(c)[feature]);
        let carbon = u64::from(c[0]);
        let (lo_num, lo_den, hi_num, hi_den) = (
            u64::from(self.ratio_lo_num[feature]),
            u64::from(self.ratio_lo_den[feature]),
            u64::from(self.ratio_hi_num[feature]),
            u64::from(self.ratio_hi_den[feature]),
        );
        if lo_den == 0 || hi_den == 0 {
            return false;
        }
        // `n <= 4 * u16::MAX`, denominators and `carbon <= u16::MAX`: every
        // product is below `2^34`, so `u64` cannot overflow.
        n * lo_den >= lo_num * carbon && n * hi_den <= hi_num * carbon
    }

    /// (iv) [`dbe_twice`] within the fitted range of the composition's
    /// heavy-atom-total bucket. False past the fitted buckets and in buckets
    /// the fit never saw (no train support), and on arithmetic error (never
    /// for representable counts).
    pub fn passes_ratio_dbe(&self, c: &Composition) -> bool {
        if self.heavy_bucket == 0 {
            return false;
        }
        let bucket = (heavy_total_of(c) / u64::from(self.heavy_bucket)) as usize;
        if self.heavy_seen.get(bucket) != Some(&true) {
            return false;
        }
        let Some(range) = self.dbe_by_heavy.get(bucket) else {
            return false;
        };
        match dbe_twice(c) {
            Ok(twice) => twice >= range[0] && twice <= range[1],
            Err(_) => false,
        }
    }

    /// Serialized size in bytes (the JSON form below).
    pub fn bytes(&self) -> usize {
        self.to_json().len()
    }

    /// Serialize the bounds.
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| {
            // The struct is plain data; serialization cannot fail. The
            // fallback keeps this infallible without `unwrap()`.
            String::from("{}")
        })
    }

    /// Parse [`RatioBounds::to_json`].
    pub fn from_json(text: &str) -> Result<Self> {
        let bounds: Self = serde_json::from_str(text)?;
        Ok(bounds)
    }
}

/// One precursor enumeration query: exactly what
/// [`WindowQuery`](super::formula::WindowQuery) carries for the mass window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EnumQuery {
    /// Precursor m/z in integer units.
    pub precursor_mz: u32,
    /// Adduct id of the precursor.
    pub adduct: u16,
    /// Precursor tolerance in tenths of a ppm.
    pub ppm_tenths: u32,
    /// Precursor uncertainty; `u32::MAX` means unknown (skips the search).
    pub precursor_uncertainty: u32,
}

impl From<&WindowQuery> for EnumQuery {
    /// Drop the table work limits; the mass window fields carry over exactly.
    fn from(q: &WindowQuery) -> Self {
        Self {
            precursor_mz: q.precursor_mz,
            adduct: q.adduct,
            ppm_tenths: q.ppm_tenths,
            precursor_uncertainty: q.precursor_uncertainty,
        }
    }
}

/// Work limits and chemical-filter switches of [`enumerate`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EnumLimits {
    /// Stop with `exhausted` once more prefix expansions would be needed.
    pub nodes_visited_max: u64,
    /// Keep at most this many joined rows before reporting `exhausted`.
    /// The kept rows are the `capacity` smallest `(mass, composition)` rows
    /// among ALL joins (a bounded max-heap, O(capacity) memory), i.e. the
    /// canonical prefix of the uncapped output; `rows_joined` still counts
    /// every join.
    pub capacity: usize,
    /// Score at most this many joined rows (canonical order) before `exhausted`.
    pub scored_max: usize,
    /// Apply filter (a), the saturated-acyclic hydrogen ceiling.
    pub filter_h_max: bool,
    /// Apply filter (b), the parity/integer-DBE rule.
    pub filter_parity: bool,
    /// Apply filter (c), DBE `>= 0`.
    pub filter_dbe: bool,
    /// Train-fit pruning bounds of [`RatioBounds`], or `None` for the exact
    /// search. When present the DFS visits carbon first (the scored output
    /// order is unchanged), the (i)/(iii) maxima prune DFS branches (counted
    /// in `pruned_ratio_cap` / `pruned_rare`), and the leaf applies the
    /// (i)/(iii) completion refinement plus the (ii)/(iv) filters in the
    /// fixed order of the module docs, each with its own reject counter.
    pub ratio: Option<RatioBounds>,
}

impl Default for EnumLimits {
    /// No work limits; every chemical filter applies; no ratio pruning.
    fn default() -> Self {
        Self {
            nodes_visited_max: u64::MAX,
            capacity: usize::MAX,
            scored_max: usize::MAX,
            filter_h_max: true,
            filter_parity: true,
            filter_dbe: true,
            ratio: None,
        }
    }
}

impl EnumLimits {
    /// No work limits and no chemical filters (the brute-force comparison path).
    pub fn unfiltered() -> Self {
        Self {
            filter_h_max: false,
            filter_parity: false,
            filter_dbe: false,
            ..Self::default()
        }
    }
}

/// Outcome of [`enumerate`]; counters mirror contracts §9.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EnumResult {
    /// Neutral parent mass, or `None` when [`parent_mass`] errors.
    pub parent_mass: Option<u32>,
    /// Observation part of the §5 verdict error bound: the query's
    /// uncertainty (`precursor_uncertainty` / neutral `uncertainty`).
    pub error_observation: u32,
    /// Neutralisation part of the §5 verdict error bound: `1` on the
    /// precursor path (the adduct conversion's rounding bound, contract
    /// §4.3: 33 + 421 nano-dalton, rounded up), `0` for an already-neutral
    /// mass (no conversion happens). The per-candidate verdict error is
    /// `ceil(composition_error_nda / 1000) + error_observation +
    /// error_neutralisation` (saturating).
    pub error_neutralisation: u32,
    /// Scored compositions in canonical order (mass, then element counts).
    pub compositions: Vec<Composition>,
    /// Integer mass per scored composition.
    pub masses: Vec<u32>,
    /// Per scored composition, whether its verdict was Ambiguous.
    pub ambiguous: Vec<bool>,
    /// Heavy vector prefixes expanded (every depth-first call).
    pub nodes_visited: u64,
    /// Integer hydrogen counts for which the §5 verdict ran.
    pub hydrogen_checks: u64,
    /// Joined rows before the capacity/scored caps.
    pub rows_joined: u64,
    /// Joined rows kept (the scored output length).
    pub rows_scored: u64,
    /// Verdict-passing compositions removed by filter (a).
    pub rejected_h_max: u64,
    /// Verdict-passing compositions removed by filter (b).
    pub rejected_parity: u64,
    /// Verdict-passing compositions removed by filter (c).
    pub rejected_dbe: u64,
    /// Compositions removed by the §5 verdict itself.
    pub rejected_mass: u64,
    /// DFS branches (count values) skipped by the (i) carbon/heavy-bucketed
    /// maxima. Zero without [`EnumLimits::ratio`].
    pub pruned_ratio_cap: u64,
    /// DFS branches skipped by the (iii) rare-atom maxima. Zero without
    /// [`EnumLimits::ratio`].
    pub pruned_rare: u64,
    /// Complete compositions failing the (i) final-total refinement at the
    /// leaf: they passed the looser DFS-prefix check but violate the exact
    /// bucketed maxima. Zero without [`EnumLimits::ratio`].
    pub rejected_ratio_cap: u64,
    /// Complete compositions failing the (iii) rare minima or the final-total
    /// refinement of the rare maxima at the leaf. Zero without
    /// [`EnumLimits::ratio`].
    pub rejected_rare: u64,
    /// Verdict- and exact-filter-passing compositions removed by the (ii)
    /// H/C stage.
    pub rejected_ratio_hc: u64,
    /// Removed by the (ii) N/C stage.
    pub rejected_ratio_nc: u64,
    /// Removed by the (ii) O/C stage.
    pub rejected_ratio_oc: u64,
    /// Removed by the (ii) (F+Cl+Br+I)/C stage.
    pub rejected_ratio_hal: u64,
    /// Removed by the (ii) S/C stage.
    pub rejected_ratio_s: u64,
    /// Removed by the (ii) P/C stage.
    pub rejected_ratio_p: u64,
    /// Verdict- and exact-filter-passing compositions removed by the (iv)
    /// DBE-per-heavy-bucket stage.
    pub rejected_ratio_dbe: u64,
    /// A work limit was reached: the joined set may be incomplete.
    pub exhausted: bool,
    /// The search completed and no composition joined.
    pub absent: bool,
    /// The search finished and every joined row was scored (never exhausted).
    pub support_complete: bool,
    /// Request status bits of [`request_status`]: `MASS_OVERFLOW` when the
    /// parent mass leaves the `u32` range, `EXACT_MASS_UNAVAILABLE |
    /// FORMULA_ABSENT` for the unknown-precision sentinel,
    /// `FORMULA_SEARCH_EXHAUSTED` when a work limit stopped or truncated the
    /// search, `FORMULA_ABSENT` when the search completed and joined nothing.
    pub status: u32,
}

impl EnumResult {
    /// Whether a composition is in the scored output.
    pub fn scored_contains(&self, c: &Composition) -> bool {
        self.compositions.iter().any(|o| o == c)
    }
}

/// `ceil(a / b)` for `b > 0` in `i128` (no overflow: inputs are ±2³²).
fn ceil_div(a: i128, b: i128) -> i128 {
    if a >= 0 {
        (a + b - 1) / b
    } else {
        a / b
    }
}

/// `floor(a / b)` for `b > 0` in `i128` (no overflow: inputs are ±2³²).
fn floor_div(a: i128, b: i128) -> i128 {
    if a >= 0 {
        a / b
    } else {
        (a - b + 1) / b
    }
}

/// Method-independent gold stage recall of one spectrum.
///
/// `window` needs the method's superset half-width and domain membership
/// (`in_domain`); the remaining stages are method-independent. With unknown
/// precursor precision (`precursor_uncertainty == u32::MAX`) or an invalid
/// parent mass (`parent == None`) the search evaluates nothing, so the gold
/// contributes to NO recall stage while the spectrum stays in the
/// denominator; that case reports `exact_mass_unavailable` and every other
/// stage false. A missing gold (an unbuildable molecule graph) is a miss at
/// every stage without unavailability.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GoldStages {
    /// Gold in the method's domain and superset window.
    pub window: bool,
    /// `window` and the §5 verdict accepts the gold.
    pub accepted: bool,
    /// `window` and the §5 verdict accepts the gold or finds it ambiguous.
    pub accepted_or_ambiguous: bool,
    /// The above and the gold passes the hydrogen ceiling.
    pub after_h_max: bool,
    /// The above and the gold passes the parity rule.
    pub after_parity: bool,
    /// The above and the gold passes DBE `>= 0`.
    pub after_dbe: bool,
    /// The search evaluates nothing for this spectrum.
    pub exact_mass_unavailable: bool,
}

/// Compute [`GoldStages`] with the same arithmetic the searches use
/// (contracts §4.3 and §5): `bound = precursor_uncertainty + 1` saturating,
/// the gold's own `ceil(error_nda / 1000)` plus that bound as the verdict
/// error, and `half_width` (already `tol + bound + E_table/domain`, in `u64`)
/// for the window, saturated at `u32::MAX` exactly as the searches saturate.
pub fn gold_stages(
    parent: Option<u32>,
    gold: Option<&Composition>,
    gold_mass: Option<u32>,
    tol: u32,
    precursor_uncertainty: u32,
    half_width: u64,
    in_domain: bool,
) -> GoldStages {
    let unavailable = GoldStages {
        window: false,
        accepted: false,
        accepted_or_ambiguous: false,
        after_h_max: false,
        after_parity: false,
        after_dbe: false,
        exact_mass_unavailable: true,
    };
    if precursor_uncertainty == u32::MAX || parent.is_none() {
        return unavailable;
    }
    let miss = GoldStages {
        exact_mass_unavailable: false,
        ..unavailable
    };
    let (Some(p), Some(g), Some(m)) = (parent, gold, gold_mass) else {
        return miss;
    };
    let bound = precursor_uncertainty.saturating_add(1);
    let row_error =
        (composition_error_nda(g).div_ceil(1000)).min(u64::from(u32::MAX)) as u32;
    let error = row_error.saturating_add(bound);
    let (accepted, either) = match decide(p, m, error, tol) {
        super::chem::Verdict::Accept => (true, true),
        super::chem::Verdict::Ambiguous => (false, true),
        super::chem::Verdict::Reject => (false, false),
    };
    let window =
        in_domain && p.abs_diff(m) as u64 <= half_width.min(u64::from(u32::MAX));
    let pass_h = passes_hydrogen_ceiling(g).unwrap_or(false);
    let pass_p = passes_parity(g).unwrap_or(false);
    let pass_d = passes_dbe(g).unwrap_or(false);
    GoldStages {
        window,
        accepted: window && accepted,
        accepted_or_ambiguous: window && either,
        after_h_max: window && either && pass_h,
        after_parity: window && either && pass_h && pass_p,
        after_dbe: window && either && pass_h && pass_p && pass_d,
        exact_mass_unavailable: false,
    }
}

/// Mutable enumeration state threaded through the depth-first search.
struct EnumState {
    /// Prefixes expanded so far.
    nodes_visited: u64,
    /// The node limit stopped the search part-way.
    hit_limit: bool,
    /// Hydrogen verdicts run so far.
    hydrogen_checks: u64,
    /// Filter and verdict reject counts.
    rejected_h_max: u64,
    rejected_parity: u64,
    rejected_dbe: u64,
    rejected_mass: u64,
    /// DFS branches skipped by the (i) caps.
    pruned_ratio_cap: u64,
    /// DFS branches skipped by the (iii) maxima.
    pruned_rare: u64,
    /// Leaf completions failing the (i) refinement.
    rejected_ratio_cap: u64,
    /// Leaf completions failing the (iii) stages.
    rejected_rare: u64,
    /// Leaf completions failing each (ii) ratio, in [`RATIO_FEATURES`] order.
    rejected_ratio: [u64; 6],
    /// Leaf completions failing the (iv) DBE stage.
    rejected_ratio_dbe: u64,
    /// Joined rows in total (before caps).
    total_joined: u64,
    /// Retained joined rows: a max-heap of at most `capacity` entries, so the
    /// survivors are the canonical prefix of all joins.
    heap: BinaryHeap<(u32, Composition, bool)>,
}

/// Depth-first search over heavy-atom count vectors.
///
/// `counts` holds the current prefix (positions before `depth` are fixed);
/// `heavy_mass`/`heavy_n` are the prefix mass (u64, never overflowing: at most
/// ~8e13) and heavy-atom total. Each call counts one prefix in `nodes_visited`
/// and stops with `hit_limit` when `nodes_visited_max` is reached, so the same
/// inputs always stop at the same node. Levels run counts ascending from 0 and
/// break at the first overweight count (exact prune: masses are positive, so
/// larger counts only add mass) and past `heavy_max` (exact: totals only grow).
/// With ratio bounds the traversal `order` puts carbon first and each count
/// is additionally checked against the (i) carbon/heavy-bucketed maxima and
/// the (iii) rare maxima for the prefix extended by that count; violations
/// skip the count (`continue`, counted) because bucketed maxima need not be
/// monotone in the count. `carbon` is the fixed carbon count for depths past
/// the carbon level (`None` while carbon is unassigned, in which case the
/// carbon-bucketed check falls back to the global maximum over buckets).
#[allow(clippy::too_many_arguments)]
fn dfs(
    domain: &EnumDomain,
    parent: u32,
    tol: u32,
    bound: u32,
    hi: u32,
    limits: &EnumLimits,
    ratio: Option<&RatioBounds>,
    order: &[usize; 9],
    depth: usize,
    counts: &mut [u16; 9],
    heavy_mass: u64,
    heavy_n: u64,
    state: &mut EnumState,
) {
    if state.nodes_visited >= limits.nodes_visited_max {
        state.hit_limit = true;
        return;
    }
    // `nodes_visited < nodes_visited_max <= u64::MAX`, so `+ 1` cannot wrap.
    state.nodes_visited += 1;
    if depth == order.len() {
        check_hydrogen(
            domain, parent, tol, bound, heavy_mass, heavy_n, counts, limits, ratio, state,
        );
        return;
    }
    let pos = order[depth];
    let carbon = if depth >= 1 && order[0] == 0 {
        Some(counts[0])
    } else {
        None
    };
    let cap = u64::from(domain.heavy_caps[pos]);
    let emass = u64::from(ELEMENTS[HEAVY_ELEMENTS[pos]].mass);
    let heavy_limit = u64::from(domain.heavy_max);
    for count in 0..=cap {
        let new_n = heavy_n + count;
        if new_n > heavy_limit {
            break;
        }
        // `count * emass` is at most ~8e12: `checked_*` only documents the
        // bound; overflow would mean a mass past any window, i.e. a prune.
        let add = match count.checked_mul(emass) {
            Some(add) => add,
            None => break,
        };
        let new_mass = match heavy_mass.checked_add(add) {
            Some(mass) => mass,
            None => break,
        };
        if new_mass > u64::from(hi) {
            break;
        }
        if let Some(ratio) = ratio {
            let n = count as u16;
            if !ratio_cap_prefix_allows(ratio, pos, n, new_n, carbon) {
                // Unreachable bound: one increment per skipped count value;
                // saturation keeps the counter monotone without wrapping.
                state.pruned_ratio_cap = state.pruned_ratio_cap.saturating_add(1);
                continue;
            }
            if !ratio_rare_prefix_allows(ratio, counts, pos, n) {
                state.pruned_rare = state.pruned_rare.saturating_add(1);
                continue;
            }
        }
        counts[pos] = count as u16;
        dfs(
            domain, parent, tol, bound, hi, limits, ratio, order, depth + 1, counts, new_mass,
            new_n, state,
        );
        if state.hit_limit {
            return;
        }
    }
    counts[pos] = 0;
}

/// (i) prefix check for assigning count `n` at heavy position `pos`:
/// the carbon-bucketed maximum (exact once carbon is known; the global maximum
/// over buckets while it is not, except for carbon itself, whose bucket is
/// known from `n`) and the heavy-bucketed maximum over all buckets reachable
/// from the partial total (an upper bound of the exact final check).
fn ratio_cap_prefix_allows(
    ratio: &RatioBounds,
    pos: usize,
    n: u16,
    partial_heavy: u64,
    carbon: Option<u16>,
) -> bool {
    if ratio.carbon_bucket == 0 || ratio.heavy_bucket == 0 {
        return false;
    }
    let carbon_width = u64::from(ratio.carbon_bucket);
    let heavy_width = u64::from(ratio.heavy_bucket);
    // Carbon-bucketed maximum.
    let carbon_ok = if pos == 0 {
        let bucket = (u64::from(n) / carbon_width) as usize;
        ratio
            .max_by_carbon
            .get(bucket)
            .is_some_and(|row| n <= row[0])
    } else if let Some(carbon) = carbon {
        let bucket = (u64::from(carbon) / carbon_width) as usize;
        ratio
            .max_by_carbon
            .get(bucket)
            .is_some_and(|row| n <= row[pos])
    } else {
        ratio
            .max_by_carbon
            .iter()
            .map(|row| row[pos])
            .max()
            .is_some_and(|cap| n <= cap)
    };
    if !carbon_ok {
        return false;
    }
    // Heavy-bucketed maximum over all buckets reachable from the partial
    // total (the final total only grows, so this is an upper bound of the
    // exact final check; the leaf recheck makes the stage exact).
    let from_bucket = (partial_heavy / heavy_width) as usize;
    n <= ratio.heavy_cap_from(pos, from_bucket)
}

/// (iii) prefix check for assigning count `n` at heavy position `pos`:
/// the rare total and distinct count of the prefix extended by `n` must fit
/// the fitted maxima (prefix sums only grow, so a violation can never heal).
fn ratio_rare_prefix_allows(
    ratio: &RatioBounds,
    counts: &[u16; 9],
    pos: usize,
    n: u16,
) -> bool {
    let mut total: u32 = 0;
    let mut distinct: u16 = 0;
    for p in RARE_POS {
        let v = if p == pos { n } else { counts[p] };
        total += u32::from(v);
        if v > 0 {
            distinct += 1;
        }
    }
    total <= ratio.rare_total[1] && distinct <= ratio.rare_distinct[1]
}

/// Closed-form hydrogen loop for one complete heavy vector.
///
/// Hydrogen-only vectors (`heavy_n == 0`) never join. Otherwise every integer
/// `h` in the closed-form range intersected with the domain hydrogen bounds
/// gets the exact §5 verdict with its own arithmetic bound, then the enabled
/// chemical filters in order, then — when `ratio` is present — the ratio
/// stages in the fixed order of the module docs (each rejection counted at
/// its own stage): the (i) completion refinement of the DFS caps, the (iii)
/// rare stages, the six (ii) ratios, the (iv) DBE bucket.
#[allow(clippy::too_many_arguments)]
fn check_hydrogen(
    domain: &EnumDomain,
    parent: u32,
    tol: u32,
    bound: u32,
    heavy_mass: u64,
    heavy_n: u64,
    counts: &[u16; 9],
    limits: &EnumLimits,
    ratio: Option<&RatioBounds>,
    state: &mut EnumState,
) {
    if heavy_n == 0 {
        return;
    }
    let lo = parent.saturating_sub(
        (u64::from(tol) + u64::from(bound) + u64::from(domain.max_error()))
            .min(u64::from(u32::MAX)) as u32,
    );
    let hi = parent.saturating_add(
        (u64::from(tol) + u64::from(bound) + u64::from(domain.max_error()))
            .min(u64::from(u32::MAX)) as u32,
    );
    let h_unit = ELEMENTS[HYDROGEN].mass as i128;
    let h_lo = ceil_div(lo as i128 - heavy_mass as i128, h_unit)
        .max(i128::from(domain.hydrogen_min))
        .max(0);
    let h_hi = floor_div(hi as i128 - heavy_mass as i128, h_unit)
        .min(i128::from(domain.hydrogen_max))
        .min(i128::from(u16::MAX));
    if h_lo > h_hi {
        return;
    }
    for h in h_lo..=h_hi {
        // Unreachable bound: billions of checks per leaf would be needed to
        // wrap; saturation keeps the counter monotone without wrapping.
        state.hydrogen_checks = state.hydrogen_checks.saturating_add(1);
        let mut c: Composition = [0; 10];
        for (i, e) in HEAVY_ELEMENTS.iter().enumerate() {
            c[*e] = counts[i];
        }
        // `h` is clamped to `[0, u16::MAX]` above, so the cast is exact.
        c[HYDROGEN] = h as u16;
        // By construction `heavy + h * m_H <= hi <= u32::MAX`; the checked
        // arithmetic only documents the bound (overflow would be a prune).
        let h_scaled = match (h as u64).checked_mul(h_unit as u64) {
            Some(scaled) => scaled,
            None => continue,
        };
        let mass = match heavy_mass
            .checked_add(h_scaled)
            .filter(|m| *m <= u64::from(u32::MAX))
        {
            Some(mass) => mass as u32,
            None => continue,
        };
        let row_error =
            (composition_error_nda(&c).div_ceil(1000)).min(u64::from(u32::MAX)) as u32;
        let error = row_error.saturating_add(bound);
        let ambiguous = match decide(parent, mass, error, tol) {
            super::chem::Verdict::Accept => false,
            super::chem::Verdict::Ambiguous => true,
            super::chem::Verdict::Reject => {
                state.rejected_mass = state.rejected_mass.saturating_add(1);
                continue;
            }
        };
        if limits.filter_h_max && !passes_hydrogen_ceiling(&c).unwrap_or(false) {
            state.rejected_h_max = state.rejected_h_max.saturating_add(1);
            continue;
        }
        if limits.filter_parity && !passes_parity(&c).unwrap_or(false) {
            state.rejected_parity = state.rejected_parity.saturating_add(1);
            continue;
        }
        if limits.filter_dbe && !passes_dbe(&c).unwrap_or(false) {
            state.rejected_dbe = state.rejected_dbe.saturating_add(1);
            continue;
        }
        if let Some(ratio) = ratio {
            // The DFS prefix checks are upper bounds of the exact
            // bucketed checks (the final total only grows), so the leaf
            // rechecks them on the complete composition: what joins here
            // is exactly the unpruned result filtered by the complete
            // predicates [`RatioBounds::passes_cap`] and
            // [`RatioBounds::passes_rare_max`].
            if !ratio.passes_cap(&c) {
                state.rejected_ratio_cap = state.rejected_ratio_cap.saturating_add(1);
                continue;
            }
            if !ratio.passes_rare_max(&c) || !ratio.passes_rare_min(&c) {
                state.rejected_rare = state.rejected_rare.saturating_add(1);
                continue;
            }
            let mut ratio_hit: Option<usize> = None;
            for k in 0..RATIO_FEATURES.len() {
                if !ratio.passes_ratio(k, &c) {
                    ratio_hit = Some(k);
                    break;
                }
            }
            if let Some(k) = ratio_hit {
                state.rejected_ratio[k] = state.rejected_ratio[k].saturating_add(1);
                continue;
            }
            if !ratio.passes_ratio_dbe(&c) {
                state.rejected_ratio_dbe = state.rejected_ratio_dbe.saturating_add(1);
                continue;
            }
        }
        state.total_joined = state.total_joined.saturating_add(1);
        state.heap.push((mass, c, ambiguous));
        if state.heap.len() > limits.capacity {
            state.heap.pop();
        }
    }
}

/// Shared enumeration core after the parent mass is known.
///
/// Runs the depth-first heavy enumeration with the closed-form hydrogen
/// loop under `parent`, `tol` and the §5 verdict error bound
/// `observation + neutralisation` (saturating; `bound` below), then
/// canonicalizes the joins. Both [`enumerate`] and [`enumerate_neutral`]
/// call this, so their searches differ only in how `parent`/`tol` and the
/// two bound parts were derived: the precursor path passes
/// (`precursor_uncertainty`, `1`) — the `1` is the adduct conversion's
/// rounding bound (contract §4.3) — while the neutral path passes
/// (`uncertainty`, `0`), because a supplied neutral mass undergoes no
/// conversion. The tolerance is always taken at the observed mass of the
/// respective path (precursor m/z for [`enumerate`], the neutral mass for
/// [`enumerate_neutral`], contract §5).
fn enumerate_core(
    domain: &EnumDomain,
    parent: u32,
    tol: u32,
    observation: u32,
    neutralisation: u32,
    limits: &EnumLimits,
) -> Result<EnumResult> {
    let bound = observation.saturating_add(neutralisation);
    let width =
        (u64::from(tol) + u64::from(bound) + u64::from(domain.max_error())).min(u64::from(u32::MAX))
            as u32;
    let hi = parent.saturating_add(width);
    let mut state = EnumState {
        nodes_visited: 0,
        hit_limit: false,
        hydrogen_checks: 0,
        rejected_h_max: 0,
        rejected_parity: 0,
        rejected_dbe: 0,
        rejected_mass: 0,
        pruned_ratio_cap: 0,
        pruned_rare: 0,
        rejected_ratio_cap: 0,
        rejected_rare: 0,
        rejected_ratio: [0; 6],
        rejected_ratio_dbe: 0,
        total_joined: 0,
        heap: BinaryHeap::new(),
    };
    let ratio = limits.ratio.as_ref();
    // Carbon first when ratio bounds are present (so the carbon-conditional
    // caps bound every later element); the scored output order is canonical
    // either way.
    let order: &[usize; 9] = if ratio.is_some() {
        &DFS_ORDER_CARBON_FIRST
    } else {
        &DFS_ORDER
    };
    let mut counts: [u16; 9] = [0; 9];
    dfs(
        domain, parent, tol, bound, hi, limits, ratio, order, 0, &mut counts, 0, 0,
        &mut state,
    );
    // Canonical output order, independent of the enumeration order: integer
    // mass, then element counts (the table's sort). The heap is a max-heap,
    // so `into_sorted_vec` yields ascending order directly.
    let rows = state.heap.into_sorted_vec();
    let scored_len = rows.len().min(limits.scored_max);
    let exhausted = state.hit_limit
        || state.total_joined > rows.len() as u64
        || rows.len() as u64 > scored_len as u64;
    let absent = !exhausted && state.total_joined == 0;
    let mut status = 0u32;
    if exhausted {
        status |= request_status::FORMULA_SEARCH_EXHAUSTED;
    }
    if absent {
        status |= request_status::FORMULA_ABSENT;
    }
    let mut compositions = Vec::with_capacity(scored_len);
    let mut masses = Vec::with_capacity(scored_len);
    let mut ambiguous = Vec::with_capacity(scored_len);
    for (mass, composition, is_ambiguous) in rows.iter().take(scored_len) {
        compositions.push(*composition);
        masses.push(*mass);
        ambiguous.push(*is_ambiguous);
    }
    Ok(EnumResult {
        parent_mass: Some(parent),
        error_observation: observation,
        error_neutralisation: neutralisation,
        compositions,
        masses,
        ambiguous,
        nodes_visited: state.nodes_visited,
        hydrogen_checks: state.hydrogen_checks,
        rows_joined: state.total_joined,
        rows_scored: scored_len as u64,
        rejected_h_max: state.rejected_h_max,
        rejected_parity: state.rejected_parity,
        rejected_dbe: state.rejected_dbe,
        rejected_mass: state.rejected_mass,
        pruned_ratio_cap: state.pruned_ratio_cap,
        pruned_rare: state.pruned_rare,
        rejected_ratio_cap: state.rejected_ratio_cap,
        rejected_rare: state.rejected_rare,
        rejected_ratio_hc: state.rejected_ratio[0],
        rejected_ratio_nc: state.rejected_ratio[1],
        rejected_ratio_oc: state.rejected_ratio[2],
        rejected_ratio_hal: state.rejected_ratio[3],
        rejected_ratio_s: state.rejected_ratio[4],
        rejected_ratio_p: state.rejected_ratio[5],
        rejected_ratio_dbe: state.rejected_ratio_dbe,
        exhausted,
        absent,
        support_complete: !exhausted,
        status,
    })
}

/// Bounded closed-form-hydrogen enumeration from a neutral mass.
///
/// Same search as [`enumerate`] after the parent-mass step, but the parent is
/// the supplied neutral mass directly: no [`parent_mass`] derivation, no
/// adduct hydrogen shift and no electron-mass term anywhere. The per-candidate
/// verdict keeps exactly the neutral-composition error bound plus the
/// observation uncertainty:
/// `error = ceil(composition_error_nda(c) / 1000) + uncertainty` with
/// `tolerance = tolerance_u32(neutral_mass, ppm_tenths)` and
/// `decide(parent = neutral_mass, mass(c), error, tolerance)`. In particular
/// there is NO `+1` neutralisation term (contract §4.3's rounding bound covers
/// the precursor m/z → neutral parent conversion, which does not happen
/// here), no electron residual (`ELECTRON_RESIDUAL_NDA`) is added and no
/// adduct hydrogen count enters the mass or the bound; the charged
/// [`enumerate`] path keeps its (`precursor_uncertainty`, `1`) bound and its
/// tolerance at the observed precursor m/z. The two paths share
/// [`enumerate_core`] and agree when the budgets are matched
/// (`uncertainty_neutral == precursor_uncertainty + 1` with equal
/// tolerances). There is no learned formula ranker: output order is the
/// enumerator's canonical order.
///
/// `uncertainty == u32::MAX` keeps the unknown-precision meaning (no search,
/// `EXACT_MASS_UNAVAILABLE`); `ppm_tenths > 1000` is an error.
pub fn enumerate_neutral(
    domain: &EnumDomain,
    neutral_mass: u32,
    ppm_tenths: u32,
    uncertainty: u32,
    limits: &EnumLimits,
) -> Result<EnumResult> {
    macro_rules! no_ratio {
        () => {
            (
                0, // pruned_ratio_cap
                0, // pruned_rare
                0, // rejected_ratio_cap
                0, // rejected_rare
                0, // rejected_ratio_hc
                0, // rejected_ratio_nc
                0, // rejected_ratio_oc
                0, // rejected_ratio_hal
                0, // rejected_ratio_s
                0, // rejected_ratio_p
                0, // rejected_ratio_dbe
            )
        };
    }
    if ppm_tenths > 1000 {
        return Err(Error::config(format!(
            "formula enumeration: ppm_tenths {ppm_tenths} exceeds 1000 (contracts §3.1)"
        )));
    }
    if uncertainty == u32::MAX {
        let (
            pruned_ratio_cap,
            pruned_rare,
            rejected_ratio_cap,
            rejected_rare,
            rejected_ratio_hc,
            rejected_ratio_nc,
            rejected_ratio_oc,
            rejected_ratio_hal,
            rejected_ratio_s,
            rejected_ratio_p,
            rejected_ratio_dbe,
        ) = no_ratio!();
        return Ok(EnumResult {
            parent_mass: Some(neutral_mass),
            error_observation: uncertainty,
            error_neutralisation: 0,
            compositions: Vec::new(),
            masses: Vec::new(),
            ambiguous: Vec::new(),
            nodes_visited: 0,
            hydrogen_checks: 0,
            rows_joined: 0,
            rows_scored: 0,
            rejected_h_max: 0,
            rejected_parity: 0,
            rejected_dbe: 0,
            rejected_mass: 0,
            pruned_ratio_cap,
            pruned_rare,
            rejected_ratio_cap,
            rejected_rare,
            rejected_ratio_hc,
            rejected_ratio_nc,
            rejected_ratio_oc,
            rejected_ratio_hal,
            rejected_ratio_s,
            rejected_ratio_p,
            rejected_ratio_dbe,
            exhausted: false,
            absent: true,
            support_complete: false,
            status: request_status::EXACT_MASS_UNAVAILABLE | request_status::FORMULA_ABSENT,
        });
    }
    // Exact for `ppm_tenths <= 1000` (checked above): the tolerance is never
    // narrowed by a wrapping cast. The tolerance is taken at the supplied
    // neutral mass (contract §5: at the observed mass); the bound carries no
    // neutralisation term (no adduct conversion happens on this path).
    let tol = tolerance_u32(neutral_mass, ppm_tenths)?;
    enumerate_core(domain, neutral_mass, tol, uncertainty, 0, limits)
}
/// Bounded closed-form-hydrogen enumeration over an [`EnumDomain`].
///
/// Runs the algorithm of the module docs: fixed-order depth-first heavy
/// enumeration with the exact heavy-mass prune, closed-form integer hydrogen
/// ranges with the exact §5 verdict per count, then the enabled chemical
/// filters in order, then the ratio stages when [`EnumLimits::ratio`] is
/// present. Infallible on chemistry: `mass_overflow` and
/// `exact_mass_unavailable` are reported in the result exactly as
/// [`FormulaTable::window`](super::formula::FormulaTable::window) reports them.
///
/// Fallible on the query: `ppm_tenths > 1000` is `Error::Config` (contracts
/// §3.1), matching [`enumerate_device_order`]. The tolerance itself comes
/// from [`tolerance_u32`], which is exact for `ppm_tenths <= 1000`, so a
/// tolerance is never narrowed by a wrapping cast.
///
/// The `capacity` rows kept are the canonical prefix of all joins (a bounded
/// max-heap), while `rows_joined` counts every join. Under
/// `nodes_visited_max` exhaustion the kept rows are the canonical prefix of
/// the rows joined before the stop; the stop node is a deterministic function
/// of the inputs, so the kept set is too.
///
/// The search body after the parent-mass step is [`enumerate_core`], shared
/// with [`enumerate_neutral`].
pub fn enumerate(
    domain: &EnumDomain,
    query: &EnumQuery,
    limits: &EnumLimits,
) -> Result<EnumResult> {
    // Zeroed ratio counters shared by the early exits (no search ran, so no
    // ratio stage rejected or pruned anything).
    macro_rules! no_ratio {
        () => {
            (
                0, // pruned_ratio_cap
                0, // pruned_rare
                0, // rejected_ratio_cap
                0, // rejected_rare
                0, // rejected_ratio_hc
                0, // rejected_ratio_nc
                0, // rejected_ratio_oc
                0, // rejected_ratio_hal
                0, // rejected_ratio_s
                0, // rejected_ratio_p
                0, // rejected_ratio_dbe
            )
        };
    }
    if query.ppm_tenths > 1000 {
        return Err(Error::config(format!(
            "formula enumeration: ppm_tenths {} exceeds 1000 (contracts §3.1)",
            query.ppm_tenths
        )));
    }
    if query.precursor_uncertainty == u32::MAX {
        let (
            pruned_ratio_cap,
            pruned_rare,
            rejected_ratio_cap,
            rejected_rare,
            rejected_ratio_hc,
            rejected_ratio_nc,
            rejected_ratio_oc,
            rejected_ratio_hal,
            rejected_ratio_s,
            rejected_ratio_p,
            rejected_ratio_dbe,
        ) = no_ratio!();
        return Ok(EnumResult {
            parent_mass: parent_mass(query.precursor_mz, query.adduct).ok(),
            error_observation: query.precursor_uncertainty,
            error_neutralisation: 1,
            compositions: Vec::new(),
            masses: Vec::new(),
            ambiguous: Vec::new(),
            nodes_visited: 0,
            hydrogen_checks: 0,
            rows_joined: 0,
            rows_scored: 0,
            rejected_h_max: 0,
            rejected_parity: 0,
            rejected_dbe: 0,
            rejected_mass: 0,
            pruned_ratio_cap,
            pruned_rare,
            rejected_ratio_cap,
            rejected_rare,
            rejected_ratio_hc,
            rejected_ratio_nc,
            rejected_ratio_oc,
            rejected_ratio_hal,
            rejected_ratio_s,
            rejected_ratio_p,
            rejected_ratio_dbe,
            exhausted: false,
            absent: true,
            support_complete: false,
            status: request_status::EXACT_MASS_UNAVAILABLE | request_status::FORMULA_ABSENT,
        });
    }
    let parent = match parent_mass(query.precursor_mz, query.adduct) {
        Ok(parent) => parent,
        Err(_) => {
            let (
                pruned_ratio_cap,
                pruned_rare,
                rejected_ratio_cap,
                rejected_rare,
                rejected_ratio_hc,
                rejected_ratio_nc,
                rejected_ratio_oc,
                rejected_ratio_hal,
                rejected_ratio_s,
                rejected_ratio_p,
                rejected_ratio_dbe,
            ) = no_ratio!();
            return Ok(EnumResult {
                parent_mass: None,
                error_observation: query.precursor_uncertainty,
                error_neutralisation: 1,
                compositions: Vec::new(),
                masses: Vec::new(),
                ambiguous: Vec::new(),
                nodes_visited: 0,
                hydrogen_checks: 0,
                rows_joined: 0,
                rows_scored: 0,
                rejected_h_max: 0,
                rejected_parity: 0,
                rejected_dbe: 0,
                rejected_mass: 0,
                pruned_ratio_cap,
                pruned_rare,
                rejected_ratio_cap,
                rejected_rare,
                rejected_ratio_hc,
                rejected_ratio_nc,
                rejected_ratio_oc,
                rejected_ratio_hal,
                rejected_ratio_s,
                rejected_ratio_p,
                rejected_ratio_dbe,
                exhausted: false,
                absent: false,
                support_complete: false,
                status: request_status::MASS_OVERFLOW,
        });
        }
    };
    // Exact for `ppm_tenths <= 1000` (checked above): the tolerance is never
    // narrowed by a wrapping cast. The tolerance is taken at the observed
    // precursor m/z (contract §5); the `+1` is the adduct conversion's
    // rounding bound (contract §4.3), counted once.
    let tol = tolerance_u32(query.precursor_mz, query.ppm_tenths)?;
    enumerate_core(domain, parent, tol, query.precursor_uncertainty, 1, limits)
}

// ---------------------------------------------------------------------------
// Device-order enumeration twin (spec §1.4): PACKED artifact, kernel lanes,
// host wrapper.
//
// The GPU kernel binds exactly one packed `u32` buffer plus the rare table:
// [`pack_device_bounds`] builds that buffer on the host, and [`rare_table`]
// provides the `[P, 8]` `u32` rows. The per-spectrum words (parent,
// tolerance, bound, window, budget, cap) travel as an 8-word `meta` slice
// (indices [`META_PARENT`]..[`META_RESERVED`]).
//
// Kernel twins (copied line for line into the `#[cube]` kernels of
// `tensor/ops/ms2_enum.rs`, one shared lane function in two modes, except
// for the slice/array parameter types). A kernel twin takes ONLY `&[u32]`
// slices and `u32` scalars, uses ONLY `u32` values for arithmetic, loop
// counters and indices (cast to `usize` only at the point of indexing a
// slice), calls no function that uses `i64`/`u64`/`i128` or `Option` or tuple
// returns, allocates nothing, has no early `return` and no value-form
// `if`/`match`, and names no `u32::MAX` or `ELEMENTS` entry: buffer lengths,
// `max_u32` and the fourteen chemistry scalars of [`chem_words`] all travel
// as explicit `u32` parameters, and every combination is a nested statement
// `if` over `1`/`0` flags. The twins are:
// * [`decide_u32`] — the §5 verdict as a `0`/`1`/`2` flag
//   (`ms2_enum_count`/`ms2_enum_fill` use it per candidate).
// * [`sat_add_counter`] and [`sat_add_saturates`] — the saturating counter
//   step and its saturation test (`ms2_enum_offsets` uses both).
// * [`kernel_lane`] — THE shared count/fill lane function (`ms2_enum_count`
//   with `mode 0`, `ms2_enum_fill` with `mode 1`). Count and fill are the
//   same function with a mode flag; the two launches stop at the same
//   vector. It returns `(joined, visited_with_exhausted_bit)` for the
//   caller to store; mode 1 additionally writes 13-word records of rank
//   below `cap` at `out_base + rank * 13`.
// * [`kernel_offsets`] — the offsets/counters lane (`ms2_enum_offsets`):
//   clamped exclusive prefix sums plus the 5 counters, one spectrum.
// * [`kernel_pad`] — the padding lane (`ms2_cand_pad`): one slot, padded
//   exactly when `scored <= slot < cap`, so every element of `cand` has
//   exactly one writer.
//
// Everything else in this section is HOST side: [`pack_device_bounds`]
// builds the packed artifact, [`enumerate_device_order`] allocates, runs
// count for every lane, runs [`kernel_offsets`], runs fill (which rewrites
// the same stats words), and converts the scored prefix to
// [`DeviceEnumResult`] — the host stores the scored prefix only, so the pad
// step is vacuous there and [`kernel_pad`] is exercised slot by slot in its
// own tests. The unknown-precision, bad-parent and wide-window early exits
// are host setup, as are the `u16` composition assembly and the `Vec`
// storage of the result.
//

// Packed device bounds: `pack_device_bounds` output layout, in `u32` words.
// Fixed header, then the variable section (row-major, 9-wide cap rows):
//
// | Words | Contents |
// |---|---|
// | 0 | carbon bucket row count |
// | 1 | heavy bucket row count |
// | 2 | carbon bucket width |
// | 3 | heavy bucket width |
// | 4..13 | domain heavy caps (C, N, O, F, P, S, Cl, Br, I) |
// | 13 | domain heavy-atom total max |
// | 14 | domain hydrogen min |
// | 15 | domain hydrogen max |
// | 16..22 | ratio lower numerators (H/C, N/C, O/C, hal/C, S/C, P/C) |
// | 22..28 | ratio lower denominators |
// | 28..34 | ratio upper numerators |
// | 34..40 | ratio upper denominators |
// | 40 | fit saw carbon (0/1) |
// | 41 | fit saw zero carbon (0/1) |
// | 42..44 | rare total range `[lo, hi]` |
// | 44..46 | rare distinct range `[lo, hi]` |
// | 46.. | carbon caps (`rows * 9`), carbon presence (`rows`), heavy caps |
// | | (`rows * 9`), heavy presence (`rows`), DBE intervals, biased |
// | | (`rows * 2`: lo, hi) |
//
/// Offset of the carbon bucket row count in [`pack_device_bounds`] output.
pub const PACK_N_CARBON_ROWS: u32 = 0;
/// Offset of the heavy bucket row count in [`pack_device_bounds`] output.
pub const PACK_N_HEAVY_ROWS: u32 = 1;
/// Offset of the carbon bucket width in [`pack_device_bounds`] output.
pub const PACK_CARBON_WIDTH: u32 = 2;
/// Offset of the heavy bucket width in [`pack_device_bounds`] output.
pub const PACK_HEAVY_WIDTH: u32 = 3;
/// Offset of the 9 domain heavy caps in [`pack_device_bounds`] output.
pub const PACK_HEAVY_CAPS: u32 = 4;
/// Offset of the domain heavy-atom total max in [`pack_device_bounds`] output.
pub const PACK_HEAVY_MAX: u32 = 13;
/// Offset of the domain hydrogen min in [`pack_device_bounds`] output.
pub const PACK_HYDROGEN_MIN: u32 = 14;
/// Offset of the domain hydrogen max in [`pack_device_bounds`] output.
pub const PACK_HYDROGEN_MAX: u32 = 15;
/// Offset of the 6 ratio lower numerators in [`pack_device_bounds`] output.
pub const PACK_RATIO_LO_NUM: u32 = 16;
/// Offset of the 6 ratio lower denominators in [`pack_device_bounds`] output.
pub const PACK_RATIO_LO_DEN: u32 = 22;
/// Offset of the 6 ratio upper numerators in [`pack_device_bounds`] output.
pub const PACK_RATIO_HI_NUM: u32 = 28;
/// Offset of the 6 ratio upper denominators in [`pack_device_bounds`] output.
pub const PACK_RATIO_HI_DEN: u32 = 34;
/// Offset of the saw-carbon flag in [`pack_device_bounds`] output.
pub const PACK_HAS_CARBON: u32 = 40;
/// Offset of the saw-zero-carbon flag in [`pack_device_bounds`] output.
pub const PACK_ZERO_CARBON: u32 = 41;
/// Offset of the rare total range lo in [`pack_device_bounds`] output.
pub const PACK_RARE_TOTAL_LO: u32 = 42;
/// Offset of the rare total range hi in [`pack_device_bounds`] output.
pub const PACK_RARE_TOTAL_HI: u32 = 43;
/// Offset of the rare distinct range lo in [`pack_device_bounds`] output.
pub const PACK_RARE_DISTINCT_LO: u32 = 44;
/// Offset of the rare distinct range hi in [`pack_device_bounds`] output.
pub const PACK_RARE_DISTINCT_HI: u32 = 45;
/// Length of the fixed header: the variable section starts here.
pub const PACK_HEADER_LEN: u32 = 46;
/// Bias of the stored DBE intervals: word `= (endpoint + 2^31)`, clamped to
/// `u32` (spec §1.4: the DBE intervals are stored UNSIGNED with this
/// documented bias). The lane compares `pos + bias` against `neg + stored`
/// without ever forming the signed difference.
pub const PACK_DBE_BIAS: u32 = 1 << 31;

/// Per-spectrum meta words bound alongside the packed artifact: the lane
/// reads the query through this 8-word `u32` slice.
pub const META_PARENT: u32 = 0;
/// Meta word: §5 tolerance.
pub const META_TOL: u32 = 1;
/// Meta word: query bound (`precursor_uncertainty + 1`, saturating).
pub const META_BOUND: u32 = 2;
/// Meta word: window low end.
pub const META_LO: u32 = 3;
/// Meta word: window high end.
pub const META_HI: u32 = 4;
/// Meta word: per-lane visit budget.
pub const META_BUDGET: u32 = 5;
/// Meta word: scored-candidate capacity (`M`).
pub const META_SCORED_CAP: u32 = 6;
/// Meta word: reserved (0).
pub const META_RESERVED: u32 = 7;
/// Length of the meta slice.
pub const META_LEN: u32 = 8;

/// Lane output record width in `u32` words: the 10 element counts in
/// `ELEMENTS` order, the integer mass, the flag (`1` accept, `2` ambiguous),
/// the source id (`u32::MAX` for an enumerated candidate).
pub const LANE_RECORD_WORDS: u32 = 13;
/// Lane mode: count (return `(joined, visited_with_exhausted_bit)`).
pub const LANE_MODE_COUNT: u32 = 0;
/// Lane mode: fill (write records of rank below `cap`).
pub const LANE_MODE_FILL: u32 = 1;
/// Top bit of the lane visited word: the lane exhausted its budget.
pub const LANE_EXHAUSTED_BIT: u32 = 1 << 31;
/// Mask of the visit count inside the lane visited word.
pub const LANE_VISITED_MASK: u32 = 0x7FFF_FFFF;
/// Largest visit count representable alongside the exhausted bit.
pub const LANE_VISITS_REPRESENTABLE_MAX: u32 = 0x7FFF_FFFF;

/// Pack the validated enumeration artifacts into the single `u32` buffer the
/// GPU kernel binds (layout constants [`PACK_N_CARBON_ROWS`]..[`PACK_HEADER_LEN`]).
///
/// HOST side (uses `u64`/`i64` checked arithmetic for the packing): validates
/// with [`validate_device_artifacts`], then writes domain caps and hydrogen
/// bounds, both bucket tables with their presence flags, the ratio
/// fractions, the rare ranges, and the DBE intervals biased by
/// [`PACK_DBE_BIAS`] (clamped; unseen-bucket sentinels land outside the
/// lane's reachable range and those buckets are gated by presence anyway).
/// `Error::Config` on any invalid artifact.
pub fn pack_device_bounds(domain: &EnumDomain, bounds: &RatioBounds) -> Result<Vec<u32>> {
    validate_device_artifacts(domain, bounds)?;
    let mut out: Vec<u32> = vec![
        bounds.max_by_carbon.len() as u32,
        bounds.max_by_heavy.len() as u32,
        u32::from(bounds.carbon_bucket),
        u32::from(bounds.heavy_bucket),
    ];
    for cap in domain.heavy_caps {
        out.push(u32::from(cap));
    }
    out.push(u32::from(domain.heavy_max));
    out.push(u32::from(domain.hydrogen_min));
    out.push(u32::from(domain.hydrogen_max));
    for v in bounds.ratio_lo_num {
        out.push(v);
    }
    for v in bounds.ratio_lo_den {
        out.push(v);
    }
    for v in bounds.ratio_hi_num {
        out.push(v);
    }
    for v in bounds.ratio_hi_den {
        out.push(v);
    }
    out.push(u32::from(bounds.has_carbon));
    out.push(u32::from(bounds.zero_carbon_seen));
    out.push(bounds.rare_total[0]);
    out.push(bounds.rare_total[1]);
    out.push(u32::from(bounds.rare_distinct[0]));
    out.push(u32::from(bounds.rare_distinct[1]));
    debug_assert!(out.len() as u32 == PACK_HEADER_LEN);
    for row in &bounds.max_by_carbon {
        for cap in row {
            out.push(u32::from(*cap));
        }
    }
    for seen in &bounds.carbon_seen {
        out.push(u32::from(*seen));
    }
    for row in &bounds.max_by_heavy {
        for cap in row {
            out.push(u32::from(*cap));
        }
    }
    for seen in &bounds.heavy_seen {
        out.push(u32::from(*seen));
    }
    for range in &bounds.dbe_by_heavy {
        out.push(bias_dbe(range[0]));
        out.push(bias_dbe(range[1]));
    }
    Ok(out)
}

/// Bias one DBE endpoint for [`pack_device_bounds`] (HOST side): the stored
/// word is `value + 2^31`, clamped to `u32`. Uses checked `i128` arithmetic
/// so an out-of-range endpoint can never wrap or panic: values below
/// `i32::MIN` land at 0, above `u32::MAX - 2^31` at `u32::MAX`.
/// [`validate_device_artifacts`] rejects non-sentinel endpoints outside the
/// documented exact range before packing, so production packing is exact.
fn bias_dbe(value: i64) -> u32 {
    let shifted = (value as i128) + (1i128 << 31);
    shifted.clamp(0, i128::from(u32::MAX)) as u32
}

/// Largest twice-DBE endpoint the device comparison keeps exact (spec §1.4).
/// The lane holds `pos = 2 + 2C + N + 3P + 4S <= 2,552` and
/// `neg = H + F + Cl + Br + I <= 2,043` under the cap validation
/// (`DEVICE_CAP_MAX = 255`, `DEVICE_HYDROGEN_MAX = 1023`), then compares
/// `pos + BIAS` against `neg + stored` with guarded additions. `stored =
/// endpoint + 2^31` is exact for endpoints in `[i32::MIN, i32::MAX]`; the
/// sums stay below `u32::MAX` exactly when `stored <= u32::MAX - 2,043`,
/// i.e. `endpoint <= 2^31 - 1 - 2,043 = 2,147,481,604`. The documented
/// exact range is therefore `[-2,147,483,648, 2,147,481,604]`; the unseen-
/// bucket sentinel `[i64::MAX, i64::MIN]` is exempt (those buckets are gated
/// by presence and join nothing).
pub const DBE_ENDPOINT_MIN: i64 = -(1i64 << 31);
/// Largest exact twice-DBE endpoint; see [`DBE_ENDPOINT_MIN`].
pub const DBE_ENDPOINT_MAX: i64 = (1i64 << 31) - 1 - 2043;

/// Default lane ceiling of spec §1.4 (`enum_lanes_max`): `B * P` above this
/// is refused before any launch.
pub const ENUM_LANES_MAX_DEFAULT: u32 = 262_144;

/// Default per-lane visit budget of spec §1.4 (`enum_lane_visits_max`).
pub const ENUM_LANE_VISITS_DEFAULT: u32 = 4_096;

/// Default worst-case visits per count/fill launch of spec §1.4
/// (`enum_dispatch_visits_max`): 16,000,000 (task T6B, from
/// `bench/results/ms2/p4_enum_dispatch_bench_wgpu_radeon860m.json`).
pub const ENUM_DISPATCH_VISITS_DEFAULT: u32 = 16_000_000;

/// Lanes covered by one count or fill launch (spec §1.4 bounded dispatch):
/// `max(1, dispatch_visits_max / lane_visits_max)`, so one launch's worst
/// case is about `dispatch_visits_max` visits. Both args must be non-zero
/// (validated by the configs); a zero here yields 1 lane per dispatch.
pub fn enum_lanes_per_dispatch(dispatch_visits_max: u32, lane_visits_max: u32) -> usize {
    let d = dispatch_visits_max as usize;
    let l = (lane_visits_max as usize).max(1);
    (d / l).max(1)
}

/// Number of count (or fill) launches covering `total_lanes` lanes at
/// `lanes_per_dispatch` lanes each: `ceil(total / per)` (0 when empty). A
/// pure function of the bucket shape and the config, so no device read
/// decides it.
pub fn enum_dispatches(total_lanes: usize, lanes_per_dispatch: usize) -> usize {
    if total_lanes == 0 {
        return 0;
    }
    let per = lanes_per_dispatch.max(1);
    total_lanes.div_ceil(per)
}

/// Validate a `(B, P, M)` enum dispatch without allocating (spec §1.4 work
/// bound and addressing): `B * P <= lanes_max` (else `Error::Config`), every
/// dimension fits `u32`, and every bound array's largest address fits `u32`
/// with checked host arithmetic (else `Error::Shape`). Used by all four
/// wrappers before any launch; tests exercise it directly (e.g. B = 17,
/// P = 16,384 exceeds the default 262,144).
pub fn validate_enum_dispatch(batch: usize, p: usize, m: usize, lanes_max: u32) -> Result<()> {
    let lanes = batch.checked_mul(p).ok_or_else(|| {
        Error::shape(format!(
            "validate_enum_dispatch: batch {batch} times {p} lanes overflows usize"
        ))
    })?;
    if lanes as u64 > u64::from(lanes_max) {
        return Err(Error::config(format!(
            "validate_enum_dispatch: B * P {lanes} exceeds enum_lanes_max {lanes_max} (refused before any launch)"
        )));
    }
    // Every dimension narrowed to `u32` fits.
    for (name, value) in [("B", batch), ("P", p), ("M", m)] {
        if value as u64 > u64::from(u32::MAX) {
            return Err(Error::shape(format!(
                "validate_enum_dispatch: {name} {value} exceeds u32"
            )));
        }
    }
    let b = batch as u64;
    let pp = p as u64;
    let mm = m as u64;
    // Largest addresses with checked `u64` arithmetic, then narrowed.
    let mut worst: u64 = 0;
    if b > 0 {
        worst = worst.max(b.checked_mul(8).ok_or_else(|| {
            Error::shape("validate_enum_dispatch: B * 8 overflows".to_string())
        })?);
        worst = worst.max(b.checked_mul(5).ok_or_else(|| {
            Error::shape("validate_enum_dispatch: B * 5 overflows".to_string())
        })?);
    }
    if pp > 0 {
        worst = worst.max(pp.checked_mul(8).ok_or_else(|| {
            Error::shape("validate_enum_dispatch: P * 8 overflows".to_string())
        })?);
    }
    if lanes > 0 {
        let l = lanes as u64;
        worst = worst.max(l.checked_mul(2).ok_or_else(|| {
            Error::shape("validate_enum_dispatch: B * P * 2 overflows".to_string())
        })?);
    }
    if b > 0 && mm > 0 {
        worst = worst.max(
            b.checked_mul(mm)
                .and_then(|v| v.checked_mul(13))
                .ok_or_else(|| {
                    Error::shape("validate_enum_dispatch: B * M * 13 overflows".to_string())
                })?,
        );
        worst = worst.max(
            b.checked_mul(mm)
                .ok_or_else(|| {
                    Error::shape("validate_enum_dispatch: B * M overflows".to_string())
                })?,
        );
    }
    if worst > u64::from(u32::MAX) {
        return Err(Error::shape(format!(
            "validate_enum_dispatch: largest bound address {worst} exceeds u32"
        )));
    }
    Ok(())
}

/// Validate the enumeration artifacts against every bound the device
/// arithmetic relies on (spec §1.4), using checked 64-bit arithmetic for
/// the derived products: every domain cap `<= 255`, hydrogen bounds
/// `<= 1023`, every ratio numerator and denominator `<= 2^20`, bucket tables
/// of at most 64 rows, non-zero bucket widths, presence tables matching
/// their bucket tables, the heavy caps/presence/DBE tables at equal lengths
/// (and the carbon caps/presence tables likewise), every cap of BOTH bucket
/// tables `<= 255`, every fitted twice-DBE endpoint of a seen heavy bucket
/// inside [`DBE_ENDPOINT_MIN`]..=[`DBE_ENDPOINT_MAX`] (the unseen-bucket
/// sentinel `[i64::MAX, i64::MIN]` is exempt; unseen buckets join nothing),
/// and at most [`RARE_TABLE_MAX`] rare combinations.
/// `Error::Config` on any violation.
pub fn validate_device_artifacts(domain: &EnumDomain, bounds: &RatioBounds) -> Result<()> {
    for (i, cap) in domain.heavy_caps.iter().enumerate() {
        if *cap > DEVICE_CAP_MAX {
            return Err(Error::config(format!(
                "validate_device_artifacts: heavy cap {i} {cap} exceeds {DEVICE_CAP_MAX}"
            )));
        }
    }
    if domain.hydrogen_min > DEVICE_HYDROGEN_MAX {
        return Err(Error::config(format!(
            "validate_device_artifacts: hydrogen_min {} exceeds {DEVICE_HYDROGEN_MAX}",
            domain.hydrogen_min
        )));
    }
    if domain.hydrogen_max > DEVICE_HYDROGEN_MAX {
        return Err(Error::config(format!(
            "validate_device_artifacts: hydrogen_max {} exceeds {DEVICE_HYDROGEN_MAX}",
            domain.hydrogen_max
        )));
    }
    if bounds.carbon_bucket == 0 || bounds.heavy_bucket == 0 {
        return Err(Error::config(
            "validate_device_artifacts: carbon/heavy bucket width is zero".to_string(),
        ));
    }
    if bounds.max_by_carbon.len() > DEVICE_BUCKETS_MAX {
        return Err(Error::config(format!(
            "validate_device_artifacts: carbon bucket table {} exceeds {DEVICE_BUCKETS_MAX}",
            bounds.max_by_carbon.len()
        )));
    }
    if bounds.max_by_heavy.len() > DEVICE_BUCKETS_MAX {
        return Err(Error::config(format!(
            "validate_device_artifacts: heavy bucket table {} exceeds {DEVICE_BUCKETS_MAX}",
            bounds.max_by_heavy.len()
        )));
    }
    if bounds.dbe_by_heavy.len() > DEVICE_BUCKETS_MAX {
        return Err(Error::config(format!(
            "validate_device_artifacts: DBE bucket table {} exceeds {DEVICE_BUCKETS_MAX}",
            bounds.dbe_by_heavy.len()
        )));
    }
    if bounds.carbon_seen.len() != bounds.max_by_carbon.len() {
        return Err(Error::config(
            "validate_device_artifacts: carbon presence length mismatches the carbon table".to_string(),
        ));
    }
    if bounds.heavy_seen.len() != bounds.max_by_heavy.len() {
        return Err(Error::config(
            "validate_device_artifacts: heavy presence length mismatches the heavy table".to_string(),
        ));
    }
    if bounds.dbe_by_heavy.len() != bounds.max_by_heavy.len() {
        return Err(Error::config(
            "validate_device_artifacts: DBE table length mismatches the heavy table".to_string(),
        ));
    }
    for (b, row) in bounds.max_by_carbon.iter().enumerate() {
        for (i, cap) in row.iter().enumerate() {
            if *cap > DEVICE_CAP_MAX {
                return Err(Error::config(format!(
                    "validate_device_artifacts: carbon bucket {b} cap {i} {cap} exceeds {DEVICE_CAP_MAX}"
                )));
            }
        }
    }
    for (b, row) in bounds.max_by_heavy.iter().enumerate() {
        for (i, cap) in row.iter().enumerate() {
            if *cap > DEVICE_CAP_MAX {
                return Err(Error::config(format!(
                    "validate_device_artifacts: heavy bucket {b} cap {i} {cap} exceeds {DEVICE_CAP_MAX}"
                )));
            }
        }
    }
    for k in 0..RATIO_FEATURES.len() {
        for (name, value) in [
            ("lo_num", bounds.ratio_lo_num[k]),
            ("lo_den", bounds.ratio_lo_den[k]),
            ("hi_num", bounds.ratio_hi_num[k]),
            ("hi_den", bounds.ratio_hi_den[k]),
        ] {
            if value > DEVICE_RATIO_MAX {
                return Err(Error::config(format!(
                    "validate_device_artifacts: ratio {name}[{k}] {value} exceeds {DEVICE_RATIO_MAX}"
                )));
            }
        }
    }
    for (b, range) in bounds.dbe_by_heavy.iter().enumerate() {
        let seen = bounds.heavy_seen.get(b) == Some(&true);
        if !seen {
            // Unseen buckets join nothing (gated by presence); the fit
            // sentinel `[i64::MAX, i64::MIN]` and any other endpoint are
            // exempt here.
            continue;
        }
        for (side, endpoint) in [("lo", range[0]), ("hi", range[1])] {
            if endpoint < DBE_ENDPOINT_MIN || endpoint > DBE_ENDPOINT_MAX {
                return Err(Error::config(format!(
                    "validate_device_artifacts: DBE bucket {b} {side} {endpoint} outside [{DBE_ENDPOINT_MIN}, {DBE_ENDPOINT_MAX}] (pos <= 2552, neg <= 2043 under the cap validation)"
                )));
            }
        }
        if range[0] > range[1] {
            return Err(Error::config(format!(
                "validate_device_artifacts: DBE bucket {b} interval [{}, {}] is empty",
                range[0], range[1]
            )));
        }
    }
    // The rare-combination count with checked 64-bit arithmetic and an early
    // stop past the limit (the walker below aborts there).
    let mut counts = [0u32; 6];
    let mut kept: usize = 0;
    rare_walk(
        &rare_caps_of(domain),
        &rare_masses(),
        bounds.rare_total[0],
        bounds.rare_total[1],
        bounds.rare_distinct[0],
        bounds.rare_distinct[1],
        0,
        0,
        0,
        &mut counts,
        None,
        &mut kept,
    )?;
    if kept > RARE_TABLE_MAX {
        return Err(Error::config(format!(
            "validate_device_artifacts: {kept} rare combinations exceed {RARE_TABLE_MAX}"
        )));
    }
    Ok(())
}

/// Domain caps of the six rare elements in rare-tuple order (F, P, S, Cl,
/// Br, I), i.e. [`EnumDomain::heavy_caps`] positions 3–8.
fn rare_caps_of(domain: &EnumDomain) -> [u32; 6] {
    [
        u32::from(domain.heavy_caps[3]),
        u32::from(domain.heavy_caps[4]),
        u32::from(domain.heavy_caps[5]),
        u32::from(domain.heavy_caps[6]),
        u32::from(domain.heavy_caps[7]),
        u32::from(domain.heavy_caps[8]),
    ]
}

/// Integer masses of the six rare elements in rare-tuple order, as `u64`
/// for the checked accumulation.
fn rare_masses() -> [u64; 6] {
    [
        u64::from(ELEMENTS[4].mass),
        u64::from(ELEMENTS[5].mass),
        u64::from(ELEMENTS[6].mass),
        u64::from(ELEMENTS[7].mass),
        u64::from(ELEMENTS[8].mass),
        u64::from(ELEMENTS[9].mass),
    ]
}

/// Depth-first walk over the rare-element combinations in increasing
/// lexicographic order of `(F, P, S, Cl, Br, I)` (position `pos` fixed by
/// the caller, counts ascending), pruned by the rare total/distinct ranges.
///
/// Each mass-representable leaf inside the ranges appends
/// `[F, P, S, Cl, Br, I, mass, sum]` to `out` (when present) and counts
/// toward `kept`; combinations whose true mass exceeds `u32::MAX` are
/// dropped without counting (their heavy mass already exceeds every window
/// end `hi <= u32::MAX`, so they join nothing). Past [`RARE_TABLE_MAX`]
/// kept rows the walk is `Error::Config`. Totals prune with `break`
/// (ascending counts only grow the total) and minima with `continue` against
/// the remaining caps, so narrow ranges over huge caps stay cheap; the abort
/// past the limit keeps wide ranges over huge caps cheap. All derived
/// products use checked 64-bit arithmetic.
#[allow(clippy::too_many_arguments)]
fn rare_walk(
    caps: &[u32; 6],
    masses: &[u64; 6],
    lo_total: u32,
    hi_total: u32,
    lo_dist: u16,
    hi_dist: u16,
    pos: usize,
    prefix_total: u32,
    prefix_dist: u16,
    counts: &mut [u32; 6],
    mut out: Option<&mut Vec<[u32; 8]>>,
    kept: &mut usize,
) -> Result<()> {
    if pos == 6 {
        if prefix_total < lo_total || prefix_total > hi_total {
            return Ok(());
        }
        if prefix_dist < lo_dist || prefix_dist > hi_dist {
            return Ok(());
        }
        // Mass BEFORE the `P_max` limit: unrepresentable tuples are dropped
        // without counting (their heavy mass already exceeds every window
        // end `hi <= u32::MAX`, so they join nothing); only a representable
        // row can trip the limit.
        let mut mass: u64 = 0;
        for i in 0..6 {
            let add = (u64::from(counts[i]))
                .checked_mul(masses[i])
                .ok_or_else(|| Error::config("rare_table: rare mass product overflows u64"))?;
            mass = mass
                .checked_add(add)
                .ok_or_else(|| Error::config("rare_table: rare mass sum overflows u64"))?;
        }
        if mass > u64::from(u32::MAX) {
            // Dropped, not counted: above every window end.
            return Ok(());
        }
        if *kept >= RARE_TABLE_MAX {
            return Err(Error::config(format!(
                "rare_table: more than {RARE_TABLE_MAX} rare combinations"
            )));
        }
        if let Some(rows) = out {
            rows.push([
                counts[0],
                counts[1],
                counts[2],
                counts[3],
                counts[4],
                counts[5],
                mass as u32,
                prefix_total,
            ]);
        }
        *kept += 1;
        return Ok(());
    }
    // Sum of the caps after `pos` (checked): the most the rest can still add.
    let mut rest_max: u64 = 0;
    for q in caps.iter().skip(pos + 1).take(6) {
        rest_max = rest_max
            .checked_add(u64::from(*q))
            .ok_or_else(|| Error::config("rare_table: remaining cap sum overflows u64"))?;
    }
    let rest_positions: u16 = (6 - pos - 1) as u16;
    let mut c: u32 = 0;
    while c <= caps[pos] {
        let total = (u64::from(prefix_total))
            .checked_add(u64::from(c))
            .ok_or_else(|| Error::config("rare_table: prefix total overflows u64"))?;
        if total > u64::from(hi_total) {
            break;
        }
        if total.saturating_add(rest_max) < u64::from(lo_total) {
            c += 1;
            continue;
        }
        let distinct = prefix_dist.saturating_add(u16::from(c > 0));
        if distinct > hi_dist {
            if c == 0 {
                // The prefix alone already exceeds the distinct maximum.
                return Ok(());
            }
            // Every larger `c` keeps the same distinct count.
            break;
        }
        if u32::from(distinct).saturating_add(u32::from(rest_positions)) < u32::from(lo_dist) {
            if c == 0 {
                c += 1;
                continue;
            }
            break;
        }
        counts[pos] = c;
        rare_walk(
            caps, masses, lo_total, hi_total, lo_dist, hi_dist, pos + 1,
            total as u32, distinct, counts, out.as_deref_mut(), kept,
        )?;
        // `c <= caps[pos] <= u32::MAX - 1` in practice (caps come from
        // `u16`), so the increment cannot wrap past the loop bound.
        debug_assert!(c < u32::MAX);
        c += 1;
    }
    counts[pos] = 0;
    Ok(())
}

/// The rare-element combinations `(F, P, S, Cl, Br, I)` allowed by the
/// domain caps and the rare total/distinct ranges, in increasing
/// lexicographic order, without duplicates.
///
/// Mass-representable policy (spec §1.4, revision 3): the rows are exactly
/// the mass-representable tuples — an allowed tuple whose true mass exceeds
/// `u32::MAX` cannot lie in any window (`hi <= u32::MAX`) and is not a row.
/// `P` counts the rows, not the allowed tuples. Combination 0 is all zeros
/// only when the rare ranges allow it (positive rare minima exclude it).
///
/// Row `= [F, P, S, Cl, Br, I, mass, sum]`: the 6 counts, their integer
/// mass, their count sum. `Error::Config` when more than [`RARE_TABLE_MAX`]
/// mass-representable combinations remain.
pub fn rare_table(domain: &EnumDomain, bounds: &RatioBounds) -> Result<Vec<[u32; 8]>> {
    let mut rows: Vec<[u32; 8]> = Vec::new();
    let mut kept: usize = 0;
    let mut counts = [0u32; 6];
    rare_walk(
        &rare_caps_of(domain),
        &rare_masses(),
        bounds.rare_total[0],
        bounds.rare_total[1],
        bounds.rare_distinct[0],
        bounds.rare_distinct[1],
        0,
        0,
        0,
        &mut counts,
        Some(&mut rows),
        &mut kept,
    )?;
    Ok(rows)
}

/// Work limits of [`enumerate_device_order`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceEnumLimits {
    /// Per-lane visit budget: a lane stops BEFORE the visit that would
    /// exceed it and raises its exhausted flag (spec §1.4, default 65,536).
    /// The visit unit is one `(C, N, O)` heavy vector with a non-empty
    /// hydrogen range. Budgets above [`LANE_VISITS_REPRESENTABLE_MAX`] stop
    /// at that count with exhaustion (the visited word carries the exhausted
    /// flag in its top bit).
    pub lane_visits_max: u32,
    /// Scored-candidate capacity per spectrum (spec §1.4 `M`, one of 32,
    /// 128, 512, 2048): the scored support is ranks below
    /// `min(joined, scored_cap)`.
    pub scored_cap: u32,
}

impl Default for DeviceEnumLimits {
    /// Spec defaults: 65,536 lane visits, `M = 2048`.
    fn default() -> Self {
        Self {
            lane_visits_max: 65_536,
            scored_cap: 2048,
        }
    }
}

/// Per-lane outcome of [`enumerate_device_order`], in rare-table order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceLaneStats {
    /// Joined candidates of this lane.
    pub joined: u32,
    /// `(C, N, O)` heavy vectors with a non-empty hydrogen range examined.
    pub visited: u32,
    /// The lane stopped before the visit that would exceed its budget.
    pub exhausted: bool,
}

/// Outcome of [`enumerate_device_order`]; the five counters are `visited`,
/// `joined`, `scored`, `status` and `complete` (spec §1.4 `counters [B, 5]`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceEnumResult {
    /// Neutral parent mass, or `None` when [`parent_mass`] errors.
    pub parent_mass: Option<u32>,
    /// Scored compositions in rank order (lane, then `C`, `N`, `O`, `H`).
    pub compositions: Vec<Composition>,
    /// Integer mass per scored composition.
    pub masses: Vec<u32>,
    /// Per scored composition: 1 = accept, 2 = ambiguous.
    pub flags: Vec<u8>,
    /// Per-lane stats in rare-table order.
    pub lanes: Vec<DeviceLaneStats>,
    /// Heavy vectors examined, all lanes (saturating at `u32::MAX - 1`; a
    /// saturated counter sets `formula_search_exhausted` and is a lower
    /// bound, per spec §1.4).
    pub visited: u32,
    /// Joined candidates, all lanes (saturating at `u32::MAX - 1`, likewise).
    pub joined: u32,
    /// Scored candidates kept (`min(joined, scored_cap)`).
    pub scored: u32,
    /// A lane budget, the scored cap, a saturated counter or the scope
    /// restriction cut the search.
    pub exhausted: bool,
    /// The search completed and joined nothing.
    pub absent: bool,
    /// The search finished every lane within budget and cap (`!exhausted`).
    pub complete: bool,
    /// Request status bits of [`request_status`], as in [`enumerate`].
    pub status: u32,
}

impl DeviceEnumResult {
    /// Whether a composition is in the scored output.
    pub fn scored_contains(&self, c: &Composition) -> bool {
        self.compositions.iter().any(|o| o == c)
    }

    /// Per scored composition, whether its verdict was Ambiguous.
    pub fn ambiguous(&self) -> Vec<bool> {
        self.flags.iter().map(|f| *f == 2).collect()
    }
}

/// Element masses and residuals of [`ELEMENTS`] in element order as the
/// fourteen chemistry scalars the lane twins take: `[m_C, m_N, m_O, m_H,
/// res_C, res_H, res_N, res_O, res_F, res_P, res_S, res_Cl, res_Br, res_I]`.
/// HOST side (plain indexing of the chemistry table): the kernel receives the
/// same fourteen words as scalar launch arguments, so the lane bodies stay
/// identical.
fn chem_words() -> [u32; 14] {
    [
        ELEMENTS[0].mass,
        ELEMENTS[2].mass,
        ELEMENTS[3].mass,
        ELEMENTS[HYDROGEN].mass,
        ELEMENTS[0].residual_nda,
        ELEMENTS[1].residual_nda,
        ELEMENTS[2].residual_nda,
        ELEMENTS[3].residual_nda,
        ELEMENTS[4].residual_nda,
        ELEMENTS[5].residual_nda,
        ELEMENTS[6].residual_nda,
        ELEMENTS[7].residual_nda,
        ELEMENTS[8].residual_nda,
        ELEMENTS[9].residual_nda,
    ]
}

/// KERNEL TWIN — the §5 verdict with `u32` arithmetic only (the kernel's copy
/// of [`decide`]): `r = |observed - computed|`; accept when `r <= tolerance`
/// with `error <= tolerance - r`, reject when `r > tolerance` with
/// `r - tolerance > error`, else ambiguous. Returns `1` on accept, `0` on
/// reject and `2` on ambiguous: a single `u32` flag, so the body is a
/// line-for-line copy of the kernel helper. Statement form only (no early
/// return, no value-form `if`): every subtraction sits behind the comparison
/// that makes it exact, exactly like the saturation tests of the twin below.
pub fn decide_u32(observed: u32, computed: u32, error: u32, tolerance: u32) -> u32 {
    let mut r: u32 = 0u32;
    if observed >= computed {
        r = observed - computed;
    }
    if computed > observed {
        r = computed - observed;
    }
    let mut accept: u32 = 0u32;
    if r <= tolerance {
        if error <= tolerance - r {
            accept = 1u32;
        }
    }
    let mut reject: u32 = 0u32;
    if r > tolerance {
        if r - tolerance > error {
            reject = 1u32;
        }
    }
    let mut flag: u32 = 2u32;
    if accept != 0u32 {
        flag = 1u32;
    }
    if reject != 0u32 {
        flag = 0u32;
    }
    flag
}

/// KERNEL TWIN — saturating counter step of the device counters: `a + b`,
/// saturating at `max_u32 - 1` (normally `u32::MAX - 1`; the bound travels as
/// a scalar because the kernel cannot name `u32::MAX`). A saturated counter
/// sets `formula_search_exhausted` and is then a lower bound, per spec §1.4.
/// Statement form: the sum is formed only behind the comparison that keeps it
/// below `max_u32`.
pub fn sat_add_counter(a: u32, b: u32, max_u32: u32) -> u32 {
    let mut s: u32 = max_u32 - 1u32;
    if a < max_u32 - b {
        s = a + b;
    }
    s
}

/// KERNEL TWIN — whether `a + b` saturates the device counter, i.e. the true
/// sum exceeds `max_u32 - 1`. [`kernel_offsets`] keeps the OR of this flag
/// over its accumulation and reports exhaustion on it. Returns `1` on
/// saturation, `0` otherwise, in statement form like the counter step.
pub fn sat_add_saturates(a: u32, b: u32, max_u32: u32) -> u32 {
    let lim = max_u32 - 1u32;
    let mut sat: u32 = 0u32;
    if b > lim {
        sat = 1u32;
    }
    if b <= lim {
        if a > lim - b {
            sat = 1u32;
        }
    }
    sat
}

/// KERNEL TWIN helper — guarded word read: `words[index]` when
/// `index < len`, else 0. The buffer length travels as a scalar (`len`):
/// the kernel cannot query an array length, so the twin takes it explicitly
/// too and the bodies stay identical. The `usize` cast happens only at the
/// point of indexing.
fn k_read(words: &[u32], index: u32, len: u32) -> u32 {
    let mut v: u32 = 0u32;
    if index < len {
        v = words[index as usize];
    }
    v
}

/// KERNEL TWIN helper — guarded word write: `words[index] = value` when
/// `index < len`, ignored past the end. Length handling as in [`k_read`].
fn k_write(words: &mut [u32], index: u32, value: u32, len: u32) {
    if index < len {
        words[index as usize] = value;
    }
}

/// KERNEL TWIN helper — whether `base + count * stride` is representable
/// (the overflow test of the old `k_span`, as a `1`/`0` flag). `stride == 0`
/// admits every count; otherwise `count <= (max_u32 - base) / stride` keeps
/// both the product and the sum below `max_u32`. Statement form, no
/// `checked_*`.
fn k_span_ok(base: u32, count: u32, stride: u32, max_u32: u32) -> u32 {
    let mut ok: u32 = 0u32;
    if stride == 0u32 {
        ok = 1u32;
    }
    if stride != 0u32 {
        if count <= (max_u32 - base) / stride {
            ok = 1u32;
        }
    }
    ok
}

/// KERNEL TWIN helper — `base + count * stride`, or 0 on overflow (the
/// address half of the old `k_span`). The caller checks [`k_span_ok`] first;
/// the address is only used when it is set.
fn k_span_at(base: u32, count: u32, stride: u32, max_u32: u32) -> u32 {
    let mut addr: u32 = 0u32;
    if stride == 0u32 {
        addr = base;
    }
    if stride != 0u32 {
        if count <= (max_u32 - base) / stride {
            addr = base + count * stride;
        }
    }
    addr
}

/// KERNEL TWIN helper — `base + index * stride`, or `max_u32` (which reads
/// as absent through [`k_read`]) on overflow. `max_u32` travels as a scalar
/// because the kernel cannot name `u32::MAX`.
fn k_at(base: u32, index: u32, stride: u32, max_u32: u32) -> u32 {
    let mut addr: u32 = max_u32;
    if stride == 0u32 {
        addr = base;
    }
    if stride != 0u32 {
        if index <= (max_u32 - base) / stride {
            addr = base + index * stride;
        }
    }
    addr
}

/// KERNEL TWIN helper — admission flag of one count level (the boolean half
/// of the old `k_allow`): `1` exactly when the window end covers the base
/// with a non-zero element mass, i.e. `mass != 0` and `base <= hi`.
fn k_allow_ok(hi: u32, base: u32, mass: u32) -> u32 {
    let mut ok: u32 = 0u32;
    if mass != 0u32 {
        if base <= hi {
            ok = 1u32;
        }
    }
    ok
}

/// KERNEL TWIN helper — admitted count bound of one count level (the value
/// half of the old `k_allow`): `(hi - base) / mass`, or 0 unless admitted.
/// A count is used only while `count <= k_allow_n(...)`; the product
/// `count * mass` is then below `hi - base`, so forming it cannot wrap.
fn k_allow_n(hi: u32, base: u32, mass: u32) -> u32 {
    let mut n: u32 = 0u32;
    if mass != 0u32 {
        if base <= hi {
            n = (hi - base) / mass;
        }
    }
    n
}

/// KERNEL TWIN helper — whether `rare_sum + c + n + o` exceeds the heavy
/// total max, as a `1`/`0` flag (an overflowing total is past every window,
/// so it is over). Each addition sits behind the comparison that keeps it
/// below `max`; statement form, no `checked_*`.
fn k_over_total(rare_sum: u32, c: u32, n: u32, o: u32, max: u32) -> u32 {
    let mut over: u32 = 1u32;
    if rare_sum <= max {
        if c <= max - rare_sum {
            let t1 = rare_sum + c;
            if n <= max - t1 {
                let t2 = t1 + n;
                if o <= max - t2 {
                    over = 0u32;
                }
            }
        }
    }
    over
}

/// KERNEL TWIN helper — whether the heavy total `rare_sum + c + n + o` is
/// representable, as a `1`/`0` flag (the boolean half of the old `k_total`).
/// `max_u32` is the arithmetic bound the additions are guarded against.
fn k_total_ok(rare_sum: u32, c: u32, n: u32, o: u32, max_u32: u32) -> u32 {
    let mut ok: u32 = 0u32;
    if c <= max_u32 - rare_sum {
        let t1 = rare_sum + c;
        if n <= max_u32 - t1 {
            let t2 = t1 + n;
            if o <= max_u32 - t2 {
                ok = 1u32;
            }
        }
    }
    ok
}

/// KERNEL TWIN helper — the heavy total `rare_sum + c + n + o` (the value
/// half of the old `k_total`), or 0 unless [`k_total_ok`] is set. Only read
/// when the flag is set.
fn k_total_v(rare_sum: u32, c: u32, n: u32, o: u32, max_u32: u32) -> u32 {
    let mut v: u32 = 0u32;
    if c <= max_u32 - rare_sum {
        let t1 = rare_sum + c;
        if n <= max_u32 - t1 {
            let t2 = t1 + n;
            if o <= max_u32 - t2 {
                v = t2 + o;
            }
        }
    }
    v
}

/// KERNEL TWIN helper — the §1.4 exact stages as `u32` magnitudes:
/// `pos = 2 + 2C + N + 3P + 4S` against `neg = H + F + Cl + Br + I`, compared
/// before any subtraction, plus the evenness of their sum (the hydrogen
/// ceiling and DBE `>= 0` are the same comparison; parity is the evenness of
/// the same difference). The three helpers share one guarded accumulation:
/// [`k_exact_ok`] is the verdict flag (`1` exactly when no guard tripped,
/// `neg <= pos` and the parities of `pos` and `neg` agree — comparing the
/// parities avoids forming the possibly-overflowing sum), while
/// [`k_exact_pos`] and [`k_exact_neg`] are the magnitudes for the DBE-bucket
/// stage (zero unless exact). Statement form, no `checked_*`.
#[allow(clippy::too_many_arguments)]
fn k_exact_ok(
    c: u32,
    h: u32,
    n: u32,
    f: u32,
    p: u32,
    s: u32,
    cl: u32,
    br: u32,
    ii: u32,
    max_u32: u32,
) -> u32 {
    let mut pos: u32 = 2u32;
    let mut ok: u32 = 1u32;
    if c > (max_u32 - pos) / 2u32 {
        ok = 0u32;
    }
    if ok != 0u32 {
        pos += c * 2u32;
    }
    if n > max_u32 - pos {
        ok = 0u32;
    }
    if ok != 0u32 {
        pos += n;
    }
    if p > (max_u32 - pos) / 3u32 {
        ok = 0u32;
    }
    if ok != 0u32 {
        pos += p * 3u32;
    }
    if s > (max_u32 - pos) / 4u32 {
        ok = 0u32;
    }
    if ok != 0u32 {
        pos += s * 4u32;
    }
    let mut neg: u32 = h;
    if f > max_u32 - neg {
        ok = 0u32;
    }
    if ok != 0u32 {
        neg += f;
    }
    if cl > max_u32 - neg {
        ok = 0u32;
    }
    if ok != 0u32 {
        neg += cl;
    }
    if br > max_u32 - neg {
        ok = 0u32;
    }
    if ok != 0u32 {
        neg += br;
    }
    if ii > max_u32 - neg {
        ok = 0u32;
    }
    if ok != 0u32 {
        neg += ii;
    }
    if neg > pos {
        ok = 0u32;
    }
    if pos % 2u32 != neg % 2u32 {
        ok = 0u32;
    }
    ok
}

/// KERNEL TWIN helper — the `pos` magnitude of [`k_exact_ok`], or 0 unless
/// [`k_exact_ok`] is set. Only read when the flag is set.
#[allow(clippy::too_many_arguments)]
fn k_exact_pos(
    c: u32,
    h: u32,
    n: u32,
    f: u32,
    p: u32,
    s: u32,
    cl: u32,
    br: u32,
    ii: u32,
    max_u32: u32,
) -> u32 {
    let mut pos: u32 = 2u32;
    let mut ok: u32 = 1u32;
    if c > (max_u32 - pos) / 2u32 {
        ok = 0u32;
    }
    if ok != 0u32 {
        pos += c * 2u32;
    }
    if n > max_u32 - pos {
        ok = 0u32;
    }
    if ok != 0u32 {
        pos += n;
    }
    if p > (max_u32 - pos) / 3u32 {
        ok = 0u32;
    }
    if ok != 0u32 {
        pos += p * 3u32;
    }
    if s > (max_u32 - pos) / 4u32 {
        ok = 0u32;
    }
    if ok != 0u32 {
        pos += s * 4u32;
    }
    let mut neg: u32 = h;
    if f > max_u32 - neg {
        ok = 0u32;
    }
    if ok != 0u32 {
        neg += f;
    }
    if cl > max_u32 - neg {
        ok = 0u32;
    }
    if ok != 0u32 {
        neg += cl;
    }
    if br > max_u32 - neg {
        ok = 0u32;
    }
    if ok != 0u32 {
        neg += br;
    }
    if ii > max_u32 - neg {
        ok = 0u32;
    }
    if ok != 0u32 {
        neg += ii;
    }
    if neg > pos {
        ok = 0u32;
    }
    if pos % 2u32 != neg % 2u32 {
        ok = 0u32;
    }
    let mut out: u32 = 0u32;
    if ok != 0u32 {
        out = pos;
    }
    out
}

/// KERNEL TWIN helper — the `neg` magnitude of [`k_exact_ok`], or 0 unless
/// [`k_exact_ok`] is set. Only read when the flag is set.
#[allow(clippy::too_many_arguments)]
fn k_exact_neg(
    c: u32,
    h: u32,
    n: u32,
    f: u32,
    p: u32,
    s: u32,
    cl: u32,
    br: u32,
    ii: u32,
    max_u32: u32,
) -> u32 {
    let mut pos: u32 = 2u32;
    let mut ok: u32 = 1u32;
    if c > (max_u32 - pos) / 2u32 {
        ok = 0u32;
    }
    if ok != 0u32 {
        pos += c * 2u32;
    }
    if n > max_u32 - pos {
        ok = 0u32;
    }
    if ok != 0u32 {
        pos += n;
    }
    if p > (max_u32 - pos) / 3u32 {
        ok = 0u32;
    }
    if ok != 0u32 {
        pos += p * 3u32;
    }
    if s > (max_u32 - pos) / 4u32 {
        ok = 0u32;
    }
    if ok != 0u32 {
        pos += s * 4u32;
    }
    let mut neg: u32 = h;
    if f > max_u32 - neg {
        ok = 0u32;
    }
    if ok != 0u32 {
        neg += f;
    }
    if cl > max_u32 - neg {
        ok = 0u32;
    }
    if ok != 0u32 {
        neg += cl;
    }
    if br > max_u32 - neg {
        ok = 0u32;
    }
    if ok != 0u32 {
        neg += br;
    }
    if ii > max_u32 - neg {
        ok = 0u32;
    }
    if ok != 0u32 {
        neg += ii;
    }
    if neg > pos {
        ok = 0u32;
    }
    if pos % 2u32 != neg % 2u32 {
        ok = 0u32;
    }
    let mut out: u32 = 0u32;
    if ok != 0u32 {
        out = neg;
    }
    out
}

/// KERNEL TWIN helper — the §5 verdict of one lane candidate in `u32` only:
/// the representation-error bound (`ceil(nda / 1000)` plus the query bound,
/// saturating) through [`decide_u32`], returned as its `0`/`1`/`2` flag. An
/// overflowing bound joins nothing (flag `0`, the skip arm). The element
/// masses and residuals travel as scalars (`m_*` are read by the lane;
/// `res_*` here): the kernel cannot index the host's `ELEMENTS` table, so
/// the twin takes the same scalars and the bodies stay identical. Each
/// product sits behind the comparison that keeps the running total below
/// `max_u32`; statement form, no `checked_*`.
#[allow(clippy::too_many_arguments)]
fn k_verdict(
    parent: u32,
    tol: u32,
    bound: u32,
    mass: u32,
    c: u32,
    h: u32,
    n: u32,
    o: u32,
    f: u32,
    p: u32,
    s: u32,
    cl: u32,
    br: u32,
    ii: u32,
    res_c: u32,
    res_h: u32,
    res_n: u32,
    res_o: u32,
    res_f: u32,
    res_p: u32,
    res_s: u32,
    res_cl: u32,
    res_br: u32,
    res_ii: u32,
    max_u32: u32,
) -> u32 {
    let mut nda: u32 = 0u32;
    let mut have: u32 = 1u32;
    if res_c != 0u32 {
        if c > (max_u32 - nda) / res_c {
            have = 0u32;
        }
    }
    if have != 0u32 {
        nda += c * res_c;
    }
    if res_h != 0u32 {
        if h > (max_u32 - nda) / res_h {
            have = 0u32;
        }
    }
    if have != 0u32 {
        nda += h * res_h;
    }
    if res_n != 0u32 {
        if n > (max_u32 - nda) / res_n {
            have = 0u32;
        }
    }
    if have != 0u32 {
        nda += n * res_n;
    }
    if res_o != 0u32 {
        if o > (max_u32 - nda) / res_o {
            have = 0u32;
        }
    }
    if have != 0u32 {
        nda += o * res_o;
    }
    if res_f != 0u32 {
        if f > (max_u32 - nda) / res_f {
            have = 0u32;
        }
    }
    if have != 0u32 {
        nda += f * res_f;
    }
    if res_p != 0u32 {
        if p > (max_u32 - nda) / res_p {
            have = 0u32;
        }
    }
    if have != 0u32 {
        nda += p * res_p;
    }
    if res_s != 0u32 {
        if s > (max_u32 - nda) / res_s {
            have = 0u32;
        }
    }
    if have != 0u32 {
        nda += s * res_s;
    }
    if res_cl != 0u32 {
        if cl > (max_u32 - nda) / res_cl {
            have = 0u32;
        }
    }
    if have != 0u32 {
        nda += cl * res_cl;
    }
    if res_br != 0u32 {
        if br > (max_u32 - nda) / res_br {
            have = 0u32;
        }
    }
    if have != 0u32 {
        nda += br * res_br;
    }
    if res_ii != 0u32 {
        if ii > (max_u32 - nda) / res_ii {
            have = 0u32;
        }
    }
    if have != 0u32 {
        nda += ii * res_ii;
    }
    // `nda / 1000 <= u32::MAX / 1000`, so the `+ 1` cannot wrap; the query
    // bound adds saturating.
    let mut flag: u32 = 0u32;
    if have != 0u32 {
        let mut error: u32 = nda / 1000u32;
        if nda % 1000u32 != 0u32 {
            error += 1u32;
        }
        if error > max_u32 - bound {
            error = max_u32;
        }
        if error <= max_u32 - bound {
            error += bound;
        }
        flag = decide_u32(parent, mass, error, tol);
    }
    flag
}

/// KERNEL TWIN helper — stage (i): the carbon- and heavy-total-bucketed
/// per-element maxima over the packed tables, as a `1`/`0` flag. Unseen
/// buckets (or zero widths, or an overflowing heavy total) admit nothing.
/// Row addresses come from the precomputed section bases; out-of-range reads
/// return 0 through [`k_read`], which rejects any non-trivial count. The
/// nine cap comparisons are unrolled (the kernel has no local arrays). Every
/// table read is hoisted into a `let` binding before its comparison (never a
/// call inside an `if` condition: deeply inlined calls in conditions break
/// the CPU backend's GVN pass), and every combination is a nested statement
/// `if`, never `&&`.
#[allow(clippy::too_many_arguments)]
fn k_buckets(
    c: u32,
    n: u32,
    o: u32,
    f: u32,
    p: u32,
    s: u32,
    cl: u32,
    br: u32,
    ii: u32,
    rare_sum: u32,
    bounds: &[u32],
    bounds_len: u32,
    n_carbon: u32,
    n_heavy: u32,
    cb_width: u32,
    hb_width: u32,
    caps_c: u32,
    seen_c: u32,
    caps_h: u32,
    seen_h: u32,
    max_u32: u32,
) -> u32 {
    let total_ok = k_total_ok(rare_sum, c, n, o, max_u32);
    let mut ok: u32 = 0u32;
    if total_ok != 0u32 {
        if cb_width != 0u32 {
            if hb_width != 0u32 {
                ok = 1u32;
            }
        }
    }
    let total_v = k_total_v(rare_sum, c, n, o, max_u32);
    let mut cb: u32 = 0u32;
    if cb_width != 0u32 {
        cb = c / cb_width;
    }
    let mut hb: u32 = 0u32;
    if hb_width != 0u32 {
        hb = total_v / hb_width;
    }
    if cb >= n_carbon {
        ok = 0u32;
    }
    if hb >= n_heavy {
        ok = 0u32;
    }
    let seen_c_addr = k_at(seen_c, cb, 1u32, max_u32);
    let seen_c_v = k_read(bounds, seen_c_addr, bounds_len);
    if seen_c_v != 1u32 {
        ok = 0u32;
    }
    let seen_h_addr = k_at(seen_h, hb, 1u32, max_u32);
    let seen_h_v = k_read(bounds, seen_h_addr, bounds_len);
    if seen_h_v != 1u32 {
        ok = 0u32;
    }
    let crow = k_at(caps_c, cb, 9u32, max_u32);
    let hrow = k_at(caps_h, hb, 9u32, max_u32);
    // The nine cap comparisons as a loop over the element position (the
    // kernel has no local arrays, so the count selects through an `if`
    // chain): reusing the same few variables keeps the backend's value
    // table small, where the unrolled form broke its GVN pass.
    let mut caps_ok: u32 = ok;
    let mut q: u32 = 0u32;
    let mut q_more: u32 = 1u32;
    while q_more != 0u32 {
        let mut have_q: u32 = c;
        if q == 1u32 {
            have_q = n;
        }
        if q == 2u32 {
            have_q = o;
        }
        if q == 3u32 {
            have_q = f;
        }
        if q == 4u32 {
            have_q = p;
        }
        if q == 5u32 {
            have_q = s;
        }
        if q == 6u32 {
            have_q = cl;
        }
        if q == 7u32 {
            have_q = br;
        }
        if q == 8u32 {
            have_q = ii;
        }
        let crow_addr = k_at(crow, q, 1u32, max_u32);
        let crow_cap = k_read(bounds, crow_addr, bounds_len);
        if have_q > crow_cap {
            caps_ok = 0u32;
        }
        if caps_ok != 0u32 {
            let hrow_addr = k_at(hrow, q, 1u32, max_u32);
            let hrow_cap = k_read(bounds, hrow_addr, bounds_len);
            if have_q > hrow_cap {
                caps_ok = 0u32;
            }
        }
        if q == 8u32 {
            q_more = 0u32;
        }
        if q_more != 0u32 {
            q += 1u32;
        }
    }
    caps_ok
}

/// KERNEL TWIN helper — stage (ii): the six cross-multiplied ratios with
/// `hal = F + Cl + Br + I` as the fourth numerator, as a `1`/`0` flag; zero
/// carbon passes exactly when the fit saw a zero-carbon composition. Every
/// sum and product sits behind the comparison that keeps it below `max_u32`
/// (`false` on any overflow, as the old `checked_*` arms); the six
/// numerators are selected by an `if` chain (the kernel has no local
/// arrays). Statement form, no `&&`.
#[allow(clippy::too_many_arguments)]
fn k_ratios(
    h: u32,
    n: u32,
    o: u32,
    c: u32,
    f: u32,
    p: u32,
    s: u32,
    cl: u32,
    br: u32,
    ii: u32,
    bounds: &[u32],
    bounds_len: u32,
    max_u32: u32,
) -> u32 {
    let mut hal: u32 = 0u32;
    let mut ok: u32 = 1u32;
    if f > max_u32 - cl {
        ok = 0u32;
    }
    if ok != 0u32 {
        hal = f + cl;
        if br > max_u32 - hal {
            ok = 0u32;
        }
    }
    if ok != 0u32 {
        hal += br;
        if ii > max_u32 - hal {
            ok = 0u32;
        }
    }
    if ok != 0u32 {
        hal += ii;
    }
    let hal_v = hal;
    let zc = k_read(bounds, PACK_ZERO_CARBON, bounds_len);
    let mut k: u32 = 0u32;
    let mut k_more: u32 = 1u32;
    while k_more != 0u32 {
        let mut num: u32 = h;
        if k == 1u32 {
            num = n;
        }
        if k == 2u32 {
            num = o;
        }
        if k == 3u32 {
            num = hal_v;
        }
        if k == 4u32 {
            num = s;
        }
        if k == 5u32 {
            num = p;
        }
        let mut stage_ok: u32 = 0u32;
        if c == 0u32 {
            if zc == 1u32 {
                stage_ok = 1u32;
            }
        }
        if c != 0u32 {
            let lo_num = k_read(bounds, PACK_RATIO_LO_NUM + k, bounds_len);
            let lo_den = k_read(bounds, PACK_RATIO_LO_DEN + k, bounds_len);
            let hi_num = k_read(bounds, PACK_RATIO_HI_NUM + k, bounds_len);
            let hi_den = k_read(bounds, PACK_RATIO_HI_DEN + k, bounds_len);
            let mut prod_ok: u32 = 1u32;
            let mut a: u32 = 0u32;
            if lo_den != 0u32 {
                if num <= max_u32 / lo_den {
                    a = num * lo_den;
                }
                if num > max_u32 / lo_den {
                    prod_ok = 0u32;
                }
            }
            if lo_den == 0u32 {
                prod_ok = 0u32;
            }
            let mut b: u32 = 0u32;
            if lo_num != 0u32 {
                if c <= max_u32 / lo_num {
                    b = lo_num * c;
                }
                if c > max_u32 / lo_num {
                    prod_ok = 0u32;
                }
            }
            let mut cc: u32 = 0u32;
            if hi_den != 0u32 {
                if num <= max_u32 / hi_den {
                    cc = num * hi_den;
                }
                if num > max_u32 / hi_den {
                    prod_ok = 0u32;
                }
            }
            if hi_den == 0u32 {
                prod_ok = 0u32;
            }
            let mut d: u32 = 0u32;
            if hi_num != 0u32 {
                if c <= max_u32 / hi_num {
                    d = hi_num * c;
                }
                if c > max_u32 / hi_num {
                    prod_ok = 0u32;
                }
            }
            let mut cmp_ok: u32 = 0u32;
            if prod_ok != 0u32 {
                if a >= b {
                    if cc <= d {
                        cmp_ok = 1u32;
                    }
                }
            }
            if lo_den == 0u32 {
                cmp_ok = 0u32;
            }
            if hi_den == 0u32 {
                cmp_ok = 0u32;
            }
            stage_ok = cmp_ok;
        }
        if stage_ok == 0u32 {
            ok = 0u32;
        }
        if k == 5u32 {
            k_more = 0u32;
        }
        if k_more != 0u32 {
            k += 1u32;
        }
    }
    ok
}

/// KERNEL TWIN helper — stage (iv): the twice-DBE interval of the
/// heavy-total bucket, stored UNSIGNED with [`PACK_DBE_BIAS`], as a `1`/`0`
/// flag. With `pos - neg` the signed difference, `lo <= pos - neg <= hi` is
/// `pos + bias >= neg + stored_lo` and `pos + bias <= neg + stored_hi`;
/// every sum sits behind the comparison that keeps it below `max_u32`.
/// An unseen bucket (or a zero width, or an overflowing total) admits
/// nothing. Statement form, no `checked_*`.
#[allow(clippy::too_many_arguments)]
fn k_dbe(
    pos: u32,
    neg: u32,
    rare_sum: u32,
    c: u32,
    n: u32,
    o: u32,
    bounds: &[u32],
    bounds_len: u32,
    n_heavy: u32,
    hb_width: u32,
    seen_h: u32,
    dbe_base: u32,
    max_u32: u32,
) -> u32 {
    let mut ok: u32 = 0u32;
    if k_total_ok(rare_sum, c, n, o, max_u32) != 0u32 {
        if hb_width != 0u32 {
            ok = 1u32;
        }
    }
    let total_v = k_total_v(rare_sum, c, n, o, max_u32);
    let mut hb: u32 = 0u32;
    if hb_width != 0u32 {
        hb = total_v / hb_width;
    }
    if hb >= n_heavy {
        ok = 0u32;
    }
    if k_read(bounds, k_at(seen_h, hb, 1u32, max_u32), bounds_len) != 1u32 {
        ok = 0u32;
    }
    let lo_addr = k_at(dbe_base, hb, 2u32, max_u32);
    let mut hi_addr: u32 = max_u32;
    if lo_addr < max_u32 {
        hi_addr = lo_addr + 1u32;
    }
    let slo = k_read(bounds, lo_addr, bounds_len);
    let shi = k_read(bounds, hi_addr, bounds_len);
    let mut range_ok: u32 = 0u32;
    if pos <= max_u32 - PACK_DBE_BIAS {
        let lhs = pos + PACK_DBE_BIAS;
        if neg <= max_u32 - slo {
            let rhs_lo = neg + slo;
            if neg <= max_u32 - shi {
                let rhs_hi = neg + shi;
                if lhs >= rhs_lo {
                    if lhs <= rhs_hi {
                        range_ok = 1u32;
                    }
                }
            }
        }
    }
    if ok == 0u32 {
        range_ok = 0u32;
    }
    range_ok
}

/// KERNEL TWIN — THE shared count/fill lane function (spec §1.4
/// `ms2_enum_count` with `mode == LANE_MODE_COUNT`, `ms2_enum_fill` with
/// `mode == LANE_MODE_FILL`): fixed rare counts, nested ascending `C`, then
/// `N`, then `O`, then `H`.
///
/// `meta`/`bounds`/`rare` are the FULL bound buffers with explicit word
/// lengths (`meta_len`, `bounds_len`, `rare_len`); the lane reads its
/// spectrum row at `mbase` and its rare-table row at `rbase`, so the body is
/// a line-for-line copy of the kernel lane. Count and fill are the same
/// function with a mode flag, so the two launches stop at the same vector.
/// The lane owns ONE mutable buffer `buf`: in mode 0 (`LANE_MODE_COUNT`) it
/// writes `(joined, visited_with_exhausted_bit)` at `stats_base` (guarded by
/// `buf_len`) and never touches records; in mode 1 (`LANE_MODE_FILL`) it
/// writes the lane's candidates of global rank below the EFFECTIVE cap
/// (`min(meta[META_SCORED_CAP], cap arg)`, so offsets, fill and pad agree;
/// a spectrum with cap 0 gets no fill write at all) into `buf` as
/// [`LANE_RECORD_WORDS`]-word records at `out_base + rank *
/// LANE_RECORD_WORDS` (guarded by `buf_len`) and never touches the stats
/// words. One buffer per mode keeps every output element single-writer
/// without aliasing two mutable borrows in the caller.
///
/// A fill lane whose first rank is at or above the effective cap enumerates
/// nothing (visited stays 0), and every fill lane stops as soon as its next
/// joined rank reaches the cap; count mode enumerates fully either way.
/// Returns the raw visit count (without the exhausted bit) so tests can
/// assert the 0 / reduced visits; the stats words are still written only in
/// count mode.
///
/// The element masses (`m_c`, `m_n`, `m_o`, `m_h`) and residuals (`res_*`)
/// travel as scalars: the kernel cannot index the host's `ELEMENTS` table.
/// `max_u32` (normally `u32::MAX`) stands in for every `u32::MAX` the kernel
/// cannot name, including the enumerated source id of a joined record.
///
/// Every product is admitted by a division guard first (`count <= (hi - m) /
/// element mass`, compared before the product is formed in the admitted
/// branch only), so no `u32` product or sum can overflow; every combination
/// is a nested statement `if` over `1`/`0` flags, never `&&`/`||`/`!`. The
/// visit unit is one `(C, N, O)` heavy vector with a non-empty hydrogen
/// range: a lane stops BEFORE the visit that would exceed its budget and
/// raises the exhausted flag. `distinct` is the lane's rare distinct-element
/// count; the rare maxima hold by the lane gate (the rare table enforces the
/// ranges), so only the rare minima are re-checked per candidate. Stage (i)
/// (`k_buckets`) is hydrogen-independent, so it is evaluated once per visit
/// before the `h` loop rather than once per candidate: same value, less
/// device work and a shallower inlining context for the backend.
///
/// Kernel-portability shape: the body is flat (no block nests deeper than
/// what a chain of statement `if`s needs). Each filter is a small single-exit
/// `u32`-only helper; filters combine as `if cond == 0 { ok = 0 }`;
/// arithmetic that is only valid when admitted is formed inside the admitted
/// branch (`if live` statements) or is unconditionally safe (guarded by a
/// comparison, or `saturating` by an explicit branch); loops carry running
/// flags in single-comparison `while` conditions with literal starts, and
/// exit through an explicit `*_more` flag so the increment provably cannot
/// wrap.
#[allow(clippy::too_many_arguments)]
pub fn kernel_lane(
    meta: &[u32],
    meta_len: u32,
    bounds: &[u32],
    bounds_len: u32,
    rare: &[u32],
    rare_len: u32,
    mbase: u32,
    rbase: u32,
    mode: u32,
    offset: u32,
    cap: u32,
    buf: &mut [u32],
    buf_len: u32,
    out_base: u32,
    stats_base: u32,
    m_c: u32,
    m_n: u32,
    m_o: u32,
    m_h: u32,
    res_c: u32,
    res_h: u32,
    res_n: u32,
    res_o: u32,
    res_f: u32,
    res_p: u32,
    res_s: u32,
    res_cl: u32,
    res_br: u32,
    res_ii: u32,
    max_u32: u32,
    lane_budget: u32,
) -> u32 {
    let parent = k_read(meta, mbase + META_PARENT, meta_len);
    let tol = k_read(meta, mbase + META_TOL, meta_len);
    let bound = k_read(meta, mbase + META_BOUND, meta_len);
    let lo = k_read(meta, mbase + META_LO, meta_len);
    let hi = k_read(meta, mbase + META_HI, meta_len);
    let budget = k_read(meta, mbase + META_BUDGET, meta_len);
    // The lane enforces the visit bound the chunk sizing assumes:
    // `min(metadata budget, lane_budget)`, then the representable clamp.
    // `budget_eff` starts from the buffer-loaded budget (never from a copy
    // of the scalar argument), in count and fill identically so ranks agree.
    let mut budget_eff: u32 = budget;
    if budget_eff > lane_budget {
        budget_eff = lane_budget;
    }
    if budget_eff > LANE_VISITS_REPRESENTABLE_MAX {
        budget_eff = LANE_VISITS_REPRESENTABLE_MAX;
    }
    // One effective scored cap (finding 1): `min(meta word 6, cap arg)`.
    // In count mode `cap` is 0 and unused, so the effective cap equals the
    // arg there; in fill mode the lane reads the per-spectrum word too, so
    // offsets, fill and pad agree slot for slot. A spectrum with cap 0 gets
    // no fill write at all.
    let mcap = k_read(meta, mbase + META_SCORED_CAP, meta_len);
    let mut eff_cap: u32 = cap;
    if mode == LANE_MODE_FILL {
        if mcap < eff_cap {
            eff_cap = mcap;
        }
    }
    let f = k_read(rare, rbase, rare_len);
    let p = k_read(rare, rbase + 1u32, rare_len);
    let s = k_read(rare, rbase + 2u32, rare_len);
    let cl = k_read(rare, rbase + 3u32, rare_len);
    let br = k_read(rare, rbase + 4u32, rare_len);
    let ii = k_read(rare, rbase + 5u32, rare_len);
    let rare_mass = k_read(rare, rbase + 6u32, rare_len);
    let rare_sum = k_read(rare, rbase + 7u32, rare_len);
    let mut distinct: u32 = 0u32;
    if f > 0u32 {
        distinct += 1u32;
    }
    if p > 0u32 {
        distinct += 1u32;
    }
    if s > 0u32 {
        distinct += 1u32;
    }
    if cl > 0u32 {
        distinct += 1u32;
    }
    if br > 0u32 {
        distinct += 1u32;
    }
    if ii > 0u32 {
        distinct += 1u32;
    }
    let n_carbon = k_read(bounds, PACK_N_CARBON_ROWS, bounds_len);
    let n_heavy = k_read(bounds, PACK_N_HEAVY_ROWS, bounds_len);
    let cb_width = k_read(bounds, PACK_CARBON_WIDTH, bounds_len);
    let hb_width = k_read(bounds, PACK_HEAVY_WIDTH, bounds_len);
    let caps_c = PACK_HEADER_LEN;
    let seen_c = k_span_at(caps_c, n_carbon, 9u32, max_u32);
    let caps_h = k_span_at(seen_c, n_carbon, 1u32, max_u32);
    let seen_h = k_span_at(caps_h, n_heavy, 9u32, max_u32);
    let dbe_base = k_span_at(seen_h, n_heavy, 1u32, max_u32);
    let mut lane_ok: u32 = 0u32;
    if cb_width != 0u32 {
        if hb_width != 0u32 {
            if k_span_ok(caps_c, n_carbon, 9u32, max_u32) != 0u32 {
                if k_span_ok(seen_c, n_carbon, 1u32, max_u32) != 0u32 {
                    if k_span_ok(caps_h, n_heavy, 9u32, max_u32) != 0u32 {
                        if k_span_ok(seen_h, n_heavy, 1u32, max_u32) != 0u32 {
                            lane_ok = 1u32;
                        }
                    }
                }
            }
        }
    }
    if rare_sum < k_read(bounds, PACK_RARE_TOTAL_LO, bounds_len) {
        lane_ok = 0u32;
    }
    if rare_sum > k_read(bounds, PACK_RARE_TOTAL_HI, bounds_len) {
        lane_ok = 0u32;
    }
    if distinct < k_read(bounds, PACK_RARE_DISTINCT_LO, bounds_len) {
        lane_ok = 0u32;
    }
    if distinct > k_read(bounds, PACK_RARE_DISTINCT_HI, bounds_len) {
        lane_ok = 0u32;
    }
    if rare_mass > hi {
        lane_ok = 0u32;
    }
    // Fill capacity exit (finding 4): a fill lane whose first rank is at or
    // above the effective cap owns no slots, so it enumerates nothing
    // (visited stays 0). Count mode is unchanged. No early `return` (the
    // twin has none): clearing `lane_ok` skips the same traversal.
    if mode == LANE_MODE_FILL {
        if offset >= eff_cap {
            lane_ok = 0u32;
        }
    }
    let cap_c = k_read(bounds, PACK_HEAVY_CAPS, bounds_len);
    let cap_n = k_read(bounds, PACK_HEAVY_CAPS + 1u32, bounds_len);
    let cap_o = k_read(bounds, PACK_HEAVY_CAPS + 2u32, bounds_len);
    let h_min = k_read(bounds, PACK_HYDROGEN_MIN, bounds_len);
    let h_max = k_read(bounds, PACK_HYDROGEN_MAX, bounds_len);
    let heavy_max = k_read(bounds, PACK_HEAVY_MAX, bounds_len);
    let mut visited: u32 = 0;
    let mut joined: u32 = 0;
    let mut exhausted: u32 = 0u32;
    let mut stop: u32 = 0u32;
    if lane_ok != 0u32 {
        let mut c: u32 = 0;
        let mut c_done: u32 = 0u32;
        let mut c_more: u32 = 1u32;
        while c_more != 0u32 {
            // Guard FIRST: `c` is admitted only while `c * m_c <= hi -
            // rare_mass` (past it the level ends: larger `c` only adds mass
            // and atoms). The product below is formed only in the admitted
            // branch.
            let c_allow_ok = k_allow_ok(hi, rare_mass, m_c);
            let c_allow = k_allow_n(hi, rare_mass, m_c);
            let mut c_over: u32 = 0u32;
            if c_allow_ok == 0u32 {
                c_over = 1u32;
            }
            if c > c_allow {
                c_over = 1u32;
            }
            if k_over_total(rare_sum, c, 0u32, 0u32, heavy_max) != 0u32 {
                c_over = 1u32;
            }
            let mut c_live: u32 = 0u32;
            if c_allow_ok != 0u32 {
                if c_over == 0u32 {
                    c_live = 1u32;
                }
            }
            if c_over != 0u32 {
                c_done = 1u32;
            }
            if c_allow_ok == 0u32 {
                c_done = 1u32;
            }
            if c_done != 0u32 {
                c_live = 0u32;
            }
            let mut c_prod: u32 = 0u32;
            let mut c_prod_ok: u32 = 0u32;
            if c_live != 0u32 {
                if m_c != 0u32 {
                    if c <= max_u32 / m_c {
                        c_prod = c * m_c;
                        c_prod_ok = 1u32;
                    }
                }
            }
            let mut m_after_c: u32 = hi;
            if c_prod_ok != 0u32 {
                if c_prod <= max_u32 - rare_mass {
                    m_after_c = rare_mass + c_prod;
                }
            }
            let mut c_go: u32 = 0u32;
            if c_live != 0u32 {
                if c_prod_ok != 0u32 {
                    if m_after_c <= hi {
                        c_go = 1u32;
                    }
                }
            }
            if c_go != 0u32 {
                let mut n: u32 = 0;
                let mut n_done: u32 = 0u32;
                let mut n_more: u32 = 1u32;
                while n_more != 0u32 {
                    // Same guard shape for `n`: compared first, multiplied
                    // only when admitted.
                    let n_allow_ok = k_allow_ok(hi, m_after_c, m_n);
                    let n_allow = k_allow_n(hi, m_after_c, m_n);
                    let mut n_over: u32 = 0u32;
                    if n_allow_ok == 0u32 {
                        n_over = 1u32;
                    }
                    if n > n_allow {
                        n_over = 1u32;
                    }
                    if k_over_total(rare_sum, c, n, 0u32, heavy_max) != 0u32 {
                        n_over = 1u32;
                    }
                    let mut n_live: u32 = 0u32;
                    if c_go != 0u32 {
                        if n_allow_ok != 0u32 {
                            if n_over == 0u32 {
                                n_live = 1u32;
                            }
                        }
                    }
                    if n_over != 0u32 {
                        n_done = 1u32;
                    }
                    if n_allow_ok == 0u32 {
                        n_done = 1u32;
                    }
                    if n_done != 0u32 {
                        n_live = 0u32;
                    }
                    let mut n_prod: u32 = 0u32;
                    let mut n_prod_ok: u32 = 0u32;
                    if n_live != 0u32 {
                        if m_n != 0u32 {
                            if n <= max_u32 / m_n {
                                n_prod = n * m_n;
                                n_prod_ok = 1u32;
                            }
                        }
                    }
                    let mut m_after_n: u32 = hi;
                    if n_prod_ok != 0u32 {
                        if n_prod <= max_u32 - m_after_c {
                            m_after_n = m_after_c + n_prod;
                        }
                    }
                    let mut n_go: u32 = 0u32;
                    if n_live != 0u32 {
                        if n_prod_ok != 0u32 {
                            if m_after_n <= hi {
                                n_go = 1u32;
                            }
                        }
                    }
                    if n_go != 0u32 {
                        let mut o: u32 = 0;
                        let mut o_done: u32 = 0u32;
                        let mut o_more: u32 = 1u32;
                        while o_more != 0u32 {
                            // Same guard shape for `o`.
                            let o_allow_ok = k_allow_ok(hi, m_after_n, m_o);
                            let o_allow = k_allow_n(hi, m_after_n, m_o);
                            let mut o_over: u32 = 0u32;
                            if o_allow_ok == 0u32 {
                                o_over = 1u32;
                            }
                            if o > o_allow {
                                o_over = 1u32;
                            }
                            if k_over_total(rare_sum, c, n, o, heavy_max) != 0u32 {
                                o_over = 1u32;
                            }
                            let mut o_live: u32 = 0u32;
                            if c_go != 0u32 {
                                if n_go != 0u32 {
                                    if o_allow_ok != 0u32 {
                                        if o_over == 0u32 {
                                            o_live = 1u32;
                                        }
                                    }
                                }
                            }
                            if o_over != 0u32 {
                                o_done = 1u32;
                            }
                            if o_allow_ok == 0u32 {
                                o_done = 1u32;
                            }
                            if o_done != 0u32 {
                                o_live = 0u32;
                            }
                            let mut o_prod: u32 = 0u32;
                            let mut o_prod_ok: u32 = 0u32;
                            if o_live != 0u32 {
                                if m_o != 0u32 {
                                    if o <= max_u32 / m_o {
                                        o_prod = o * m_o;
                                        o_prod_ok = 1u32;
                                    }
                                }
                            }
                            let mut m: u32 = max_u32;
                            if o_prod_ok != 0u32 {
                                if o_prod <= max_u32 - m_after_n {
                                    m = m_after_n + o_prod;
                                }
                            }
                            let mut o_go: u32 = 0u32;
                            if o_live != 0u32 {
                                if o_prod_ok != 0u32 {
                                    if m <= hi {
                                        o_go = 1u32;
                                    }
                                }
                            }
                            // Unconditionally safe closed-form hydrogen ends
                            // (`hi >= m` when admitted, so the saturating
                            // difference is exact); the flags below decide
                            // whether the `h` loop runs.
                            let m_ok = k_allow_ok(hi, m, m_h);
                            let h_allow = k_allow_n(hi, m, m_h);
                            let mut h_hi: u32 = h_max;
                            if h_allow < h_max {
                                h_hi = h_allow;
                            }
                            let mut over: u32 = 0u32;
                            if lo > m {
                                over = lo - m;
                            }
                            let mut h_need: u32 = 0u32;
                            if m_h != 0u32 {
                                h_need = over / m_h;
                                if over % m_h != 0u32 {
                                    h_need += 1u32;
                                }
                            }
                            let mut h_lo: u32 = h_need;
                            if h_min > h_need {
                                h_lo = h_min;
                            }
                            let mut ordered: u32 = 0u32;
                            if o_go != 0u32 {
                                if m_ok != 0u32 {
                                    if h_lo <= h_hi {
                                        ordered = 1u32;
                                    }
                                }
                            }
                            let t_ok = k_total_ok(rare_sum, c, n, o, max_u32);
                            let total_v = k_total_v(rare_sum, c, n, o, max_u32);
                            // One visit: this `(c, n, o)` heavy vector with a
                            // non-empty hydrogen range. Stop BEFORE the visit
                            // that would exceed the budget.
                            let mut visit_this: u32 = 0u32;
                            if o_go != 0u32 {
                                if t_ok != 0u32 {
                                    if ordered != 0u32 {
                                        if total_v > 0u32 {
                                            visit_this = 1u32;
                                        }
                                    }
                                }
                            }
                            if visit_this != 0u32 {
                                if visited >= budget_eff {
                                    exhausted = 1u32;
                                    stop = 1u32;
                                }
                            }
                            let mut counted: u32 = 0u32;
                            if visit_this != 0u32 {
                                if visited < budget_eff {
                                    if stop == 0u32 {
                                        // `visited < budget_eff <= 2^31 - 1`,
                                        // so no wrap.
                                        visited += 1u32;
                                        counted = 1u32;
                                    }
                                }
                            }
                            if counted != 0u32 {
                                // Stage (i) is hydrogen-independent:
                                // hoisted out of the `h` loop (same
                                // value for every hydrogen count of
                                // this visit).
                                let buckets_ok = k_buckets(
                                    c, n, o, f, p, s, cl, br, ii, rare_sum, bounds,
                                    bounds_len, n_carbon, n_heavy, cb_width, hb_width,
                                    caps_c, seen_c, caps_h, seen_h, max_u32,
                                );
                                let mut h: u32 = h_lo;
                                let mut h_done: u32 = 0u32;
                                let mut h_more: u32 = 1u32;
                                while h_more != 0u32 {
                                    // Guard FIRST for `h` as well:
                                    // `h <= (hi - m) / m_h` (then the product
                                    // and the sum cannot wrap).
                                    let mut h_admit: u32 = 0u32;
                                    if h <= h_allow {
                                        h_admit = 1u32;
                                    }
                                    if h_admit == 0u32 {
                                        h_done = 1u32;
                                    }
                                    let mut h_prod: u32 = 0u32;
                                    let mut h_prod_ok: u32 = 0u32;
                                    if h_admit != 0u32 {
                                        if m_h != 0u32 {
                                            if h <= max_u32 / m_h {
                                                h_prod = h * m_h;
                                                h_prod_ok = 1u32;
                                            }
                                        }
                                    }
                                    let mut mass: u32 = max_u32;
                                    if h_prod_ok != 0u32 {
                                        if h_prod <= max_u32 - m {
                                            mass = m + h_prod;
                                        }
                                    }
                                    let mut mass_ok: u32 = 0u32;
                                    if h_admit != 0u32 {
                                        if h_prod_ok != 0u32 {
                                            if mass <= hi {
                                                mass_ok = 1u32;
                                            }
                                        }
                                    }
                                    let vflag = k_verdict(
                                        parent, tol, bound, mass, c, h, n, o, f, p, s,
                                        cl, br, ii, res_c, res_h, res_n, res_o, res_f,
                                        res_p, res_s, res_cl, res_br, res_ii, max_u32,
                                    );
                                    let mut v_join: u32 = 0u32;
                                    if vflag != 0u32 {
                                        v_join = 1u32;
                                    }
                                    let mut v_amb: u32 = 0u32;
                                    if vflag == 2u32 {
                                        v_amb = 1u32;
                                    }
                                    let exact_ok = k_exact_ok(
                                        c, h, n, f, p, s, cl, br, ii, max_u32,
                                    );
                                    let pos = k_exact_pos(
                                        c, h, n, f, p, s, cl, br, ii, max_u32,
                                    );
                                    let neg = k_exact_neg(
                                        c, h, n, f, p, s, cl, br, ii, max_u32,
                                    );
                                    let mut ok_all: u32 = mass_ok;
                                    if v_join == 0u32 {
                                        ok_all = 0u32;
                                    }
                                    if exact_ok == 0u32 {
                                        ok_all = 0u32;
                                    }
                                    if buckets_ok == 0u32 {
                                        ok_all = 0u32;
                                    }
                                    // Stage (iii) minima (the maxima hold by
                                    // the lane gate).
                                    if rare_sum
                                        < k_read(bounds, PACK_RARE_TOTAL_LO, bounds_len)
                                    {
                                        ok_all = 0u32;
                                    }
                                    if distinct
                                        < k_read(bounds, PACK_RARE_DISTINCT_LO, bounds_len)
                                    {
                                        ok_all = 0u32;
                                    }
                                    if k_ratios(
                                        h, n, o, c, f, p, s, cl, br, ii, bounds,
                                        bounds_len, max_u32,
                                    ) == 0u32
                                    {
                                        ok_all = 0u32;
                                    }
                                    if k_dbe(
                                        pos, neg, rare_sum, c, n, o, bounds, bounds_len,
                                        n_heavy, hb_width, seen_h, dbe_base, max_u32,
                                    ) == 0u32
                                    {
                                        ok_all = 0u32;
                                    }
                                    let mut gval: u32 = 0u32;
                                    let mut grank_ok: u32 = 0u32;
                                    if joined <= max_u32 - offset {
                                        gval = offset + joined;
                                        grank_ok = 1u32;
                                    }
                                    let mut wbase: u32 = 0u32;
                                    let mut wend_rel_ok: u32 = 0u32;
                                    if gval <= max_u32 / LANE_RECORD_WORDS {
                                        wbase = gval * LANE_RECORD_WORDS;
                                        if wbase <= max_u32 - LANE_RECORD_WORDS {
                                            wend_rel_ok = 1u32;
                                        }
                                    }
                                    let mut write_it: u32 = 0u32;
                                    if mode == LANE_MODE_FILL {
                                        write_it = 1u32;
                                    }
                                    if ok_all == 0u32 {
                                        write_it = 0u32;
                                    }
                                    if grank_ok == 0u32 {
                                        write_it = 0u32;
                                    }
                                    if gval >= eff_cap {
                                        write_it = 0u32;
                                    }
                                    if wend_rel_ok == 0u32 {
                                        write_it = 0u32;
                                    }
                                    if write_it != 0u32 {
                                        if out_base > max_u32 - (wbase + LANE_RECORD_WORDS) {
                                            write_it = 0u32;
                                        }
                                    }
                                    if write_it != 0u32 {
                                        if out_base + wbase + LANE_RECORD_WORDS > buf_len {
                                            write_it = 0u32;
                                        }
                                    }
                                    if write_it != 0u32 {
                                        let waddr = out_base + wbase;
                                        buf[waddr as usize] = c;
                                        buf[(waddr + 1u32) as usize] = h;
                                        buf[(waddr + 2u32) as usize] = n;
                                        buf[(waddr + 3u32) as usize] = o;
                                        buf[(waddr + 4u32) as usize] = f;
                                        buf[(waddr + 5u32) as usize] = p;
                                        buf[(waddr + 6u32) as usize] = s;
                                        buf[(waddr + 7u32) as usize] = cl;
                                        buf[(waddr + 8u32) as usize] = br;
                                        buf[(waddr + 9u32) as usize] = ii;
                                        buf[(waddr + 10u32) as usize] = mass;
                                        let mut flag: u32 = 1u32;
                                        if v_amb != 0u32 {
                                            flag = 2u32;
                                        }
                                        buf[(waddr + 11u32) as usize] = flag;
                                        buf[(waddr + 12u32) as usize] = max_u32;
                                    }
                                    if ok_all != 0u32 {
                                        // `joined` counts lane visits' joins
                                        // in both modes, so count and fill
                                        // agree.
                                        joined += 1u32;
                                        // Fill capacity stop (finding 4): the
                                        // next joined rank is `offset +
                                        // joined`; once it reaches the
                                        // effective cap no later join can be
                                        // written, so the lane stops. Count
                                        // mode is unchanged. The addition is
                                        // guarded: on overflow the lane stops
                                        // safely.
                                        if mode == LANE_MODE_FILL {
                                            if joined <= max_u32 - offset {
                                                if offset + joined >= eff_cap {
                                                    stop = 1u32;
                                                }
                                            }
                                            if joined > max_u32 - offset {
                                                stop = 1u32;
                                            }
                                        }
                                    }
                                    if h_done != 0u32 {
                                        h_more = 0u32;
                                    }
                                    if stop != 0u32 {
                                        h_more = 0u32;
                                    }
                                    if h == h_hi {
                                        h_more = 0u32;
                                    }
                                    if h_more != 0u32 {
                                        // `h < h_hi <= u32::MAX`, so no wrap.
                                        h += 1u32;
                                    }
                                }
                            }
                            if o_done != 0u32 {
                                o_more = 0u32;
                            }
                            if stop != 0u32 {
                                o_more = 0u32;
                            }
                            if o == cap_o {
                                o_more = 0u32;
                            }
                            if o_more != 0u32 {
                                // `o < cap_o <= u32::MAX`, so no wrap.
                                o += 1u32;
                            }
                        }
                    }
                    if n_done != 0u32 {
                        n_more = 0u32;
                    }
                    if stop != 0u32 {
                        n_more = 0u32;
                    }
                    if n == cap_n {
                        n_more = 0u32;
                    }
                    if n_more != 0u32 {
                        // `n < cap_n <= u32::MAX`, so no wrap.
                        n += 1u32;
                    }
                }
            }
            if c_done != 0u32 {
                c_more = 0u32;
            }
            if stop != 0u32 {
                c_more = 0u32;
            }
            if c == cap_c {
                c_more = 0u32;
            }
            if c_more != 0u32 {
                // `c < cap_c <= u32::MAX`, so no wrap.
                c += 1u32;
            }
        }
    }
    let mut visited_word: u32 = visited;
    if exhausted != 0u32 {
        visited_word = visited + LANE_EXHAUSTED_BIT;
    }
    if mode == LANE_MODE_COUNT {
        k_write(buf, stats_base, joined, buf_len);
        k_write(buf, stats_base + 1u32, visited_word, buf_len);
    }
    visited
}

/// KERNEL TWIN — the offsets/counters lane (spec §1.4 `ms2_enum_offsets`),
/// lane per spectrum `b`: the spectrum's `n_p` count results in `lane_stats`
/// (`[B * P, 2]` words: joined, then visited with the exhausted flag in the
/// top bit) → its `n_p` clamped offsets in `offsets` plus its 5 counters in
/// `counters` (visited, joined, scored, status bits, complete).
///
/// All buffers are the full bound buffers with explicit word lengths; the
/// spectrum's sections start at `b * n_p * 2` (stats), `b * n_p` (offsets),
/// `b * 8` (meta) and `b * 5` (counters). An offset is the exclusive prefix
/// sum CLAMPED to the scored cap (`min(prefix, cap)`), so no saturated value
/// is ever used as a rank. The cap is `min(meta[META_SCORED_CAP], cap_arg)`:
/// the wrapper passes the scored capacity it allocated for, so a skewed meta
/// word cannot address past the candidate buffer. `visited` and `joined`
/// saturate at `max_u32 - 1` via [`sat_add_counter`]; a saturation flag is
/// kept during aggregation ([`sat_add_saturates`]) and sets
/// `formula_search_exhausted` with `complete` cleared, exactly like lane
/// exhaustion or scored truncation. The status bits travel as scalars
/// (`bit_exhausted`, `bit_absent`) because the kernel combines no `|`
/// itself. Statement form, no `&&`: every combination is a nested `if` over
/// `1`/`0` flags.
#[allow(clippy::too_many_arguments)]
pub fn kernel_offsets(
    lane_stats: &[u32],
    stats_len: u32,
    meta: &[u32],
    meta_len: u32,
    offsets: &mut [u32],
    offsets_len: u32,
    counters: &mut [u32],
    counters_len: u32,
    b: u32,
    n_p: u32,
    cap_arg: u32,
    bit_exhausted: u32,
    bit_absent: u32,
    max_u32: u32,
) {
    let mbase = b * META_LEN;
    let mcap = k_read(meta, mbase + META_SCORED_CAP, meta_len);
    let mut cap: u32 = mcap;
    if cap_arg < cap {
        cap = cap_arg;
    }
    let sbase = b * n_p * 2u32;
    let obase = b * n_p;
    let cbase = b * 5u32;
    let mut running_j: u32 = 0;
    let mut running_v: u32 = 0;
    let mut any_exh: u32 = 0u32;
    let mut sat: u32 = 0u32;
    let mut r: u32 = 0;
    let mut r_more: u32 = 1u32;
    if n_p == 0u32 {
        r_more = 0u32;
    }
    while r_more != 0u32 {
        // `r < n_p`, so `sbase + 2 * r + 1` addresses this spectrum's words.
        let jr = k_read(lane_stats, sbase + r * 2u32, stats_len);
        let vw = k_read(lane_stats, sbase + r * 2u32 + 1u32, stats_len);
        let vr = vw & LANE_VISITED_MASK;
        let mut exh: u32 = 0u32;
        if vw & LANE_EXHAUSTED_BIT != 0u32 {
            exh = 1u32;
        }
        let mut off: u32 = cap;
        if running_j < cap {
            off = running_j;
        }
        k_write(offsets, obase + r, off, offsets_len);
        if sat_add_saturates(running_j, jr, max_u32) != 0u32 {
            sat = 1u32;
        }
        running_j = sat_add_counter(running_j, jr, max_u32);
        if sat_add_saturates(running_v, vr, max_u32) != 0u32 {
            sat = 1u32;
        }
        running_v = sat_add_counter(running_v, vr, max_u32);
        if exh != 0u32 {
            any_exh = 1u32;
        }
        if r + 1u32 >= n_p {
            r_more = 0u32;
        }
        if r_more != 0u32 {
            // `r < n_p - 1 <= u32::MAX - 1`, so no wrap.
            r += 1u32;
        }
    }
    let mut scored: u32 = cap;
    if running_j < cap {
        scored = running_j;
    }
    let mut trunc: u32 = 0u32;
    if running_j > scored {
        trunc = 1u32;
    }
    let mut exhausted: u32 = 0u32;
    if any_exh != 0u32 {
        exhausted = 1u32;
    }
    if trunc != 0u32 {
        exhausted = 1u32;
    }
    if sat != 0u32 {
        exhausted = 1u32;
    }
    let mut status: u32 = 0u32;
    if exhausted != 0u32 {
        status = bit_exhausted;
    }
    if exhausted == 0u32 {
        if running_j == 0u32 {
            status = bit_absent;
        }
    }
    let mut complete: u32 = 0u32;
    if exhausted == 0u32 {
        complete = 1u32;
    }
    k_write(counters, cbase, running_v, counters_len);
    k_write(counters, cbase + 1u32, running_j, counters_len);
    k_write(counters, cbase + 2u32, scored, counters_len);
    k_write(counters, cbase + 3u32, status, counters_len);
    k_write(counters, cbase + 4u32, complete, counters_len);
}

/// KERNEL TWIN — the padding lane (spec §1.4 `ms2_cand_pad`), lane per
/// candidate slot: the record at `base + slot * LANE_RECORD_WORDS` becomes a
/// padding record (all `0` except the source id `max_u32`, normally
/// `u32::MAX`) exactly when `scored <= slot < cap`, so every element of
/// `cand` has exactly one writer (fill owns ranks below `scored`, pad owns
/// the rest). The address guard runs through [`k_span_ok`]/[`k_span_at`]
/// and the writes through [`k_write`], so an unrepresentable address writes
/// nothing. Statement form, no `&&`.
pub fn kernel_pad(
    out: &mut [u32],
    out_len: u32,
    base: u32,
    slot: u32,
    scored: u32,
    cap: u32,
    max_u32: u32,
) {
    let mut pad: u32 = 0u32;
    if slot >= scored {
        if slot < cap {
            pad = 1u32;
        }
    }
    if k_span_ok(base, slot, LANE_RECORD_WORDS, max_u32) == 0u32 {
        pad = 0u32;
    }
    let wbase = k_span_at(base, slot, LANE_RECORD_WORDS, max_u32);
    if pad != 0u32 {
        // `wbase + 12` is in bounds: the span guard keeps
        // `base + slot * 13 <= max_u32`, and each write is length-guarded.
        k_write(out, wbase, 0u32, out_len);
        k_write(out, wbase + 1u32, 0u32, out_len);
        k_write(out, wbase + 2u32, 0u32, out_len);
        k_write(out, wbase + 3u32, 0u32, out_len);
        k_write(out, wbase + 4u32, 0u32, out_len);
        k_write(out, wbase + 5u32, 0u32, out_len);
        k_write(out, wbase + 6u32, 0u32, out_len);
        k_write(out, wbase + 7u32, 0u32, out_len);
        k_write(out, wbase + 8u32, 0u32, out_len);
        k_write(out, wbase + 9u32, 0u32, out_len);
        k_write(out, wbase + 10u32, 0u32, out_len);
        k_write(out, wbase + 11u32, 0u32, out_len);
        k_write(out, wbase + 12u32, max_u32, out_len);
    }
}

/// Bounded enumeration in device order over an [`EnumDomain`] with
/// [`RatioBounds`] (spec §1.4, HOST side).
///
/// HOST wrapper around the kernel twins: packs the artifacts once
/// ([`pack_device_bounds`]), runs the count lane ([`kernel_lane`] with mode
/// 0) for every rare-table row, runs the offsets lane ([`kernel_offsets`])
/// for the clamped offsets and the 5 counters, runs the fill lane (the SAME
/// [`kernel_lane`] with mode 1) into a record buffer, pads, and converts the
/// scored prefix to [`DeviceEnumResult`]. The host stores exactly the scored
/// prefix, so the pad step is vacuous here (the device pads slots
/// `[scored, cap)` with [`kernel_pad`], tested separately); every stored
/// slot is written by exactly one fill lane.
///
/// The joined rank of a candidate is its lane offset (the exclusive prefix
/// sum of the lane joined counts, clamped to `limits.scored_cap`, so no
/// saturated value is ever used as a rank) plus its index among its own
/// lane's joined candidates, and the scored support is ranks below
/// `min(joined, scored_cap)`.
///
/// `ppm_tenths > 1000` is `Error::Config` (contracts §3.1), matching
/// [`enumerate`]; the tolerance comes from [`tolerance_u32`](super::chem::tolerance_u32),
/// exact for `ppm_tenths <= 1000`, so it never narrows. The artifacts are
/// validated first ([`validate_device_artifacts`]); invalid artifacts are
/// `Error::Config`. The `u32::MAX` precursor uncertainty returns the
/// unknown-precision sentinel exactly as [`enumerate`]; an invalid parent
/// mass returns `mass_overflow`; a window half above [`DEVICE_HALF_MAX`] is
/// `formula_search_exhausted` with nothing joined. Lane exhaustion, scored
/// truncation, or a saturated counter sets `formula_search_exhausted` and
/// clears `complete`; a completed search that joined nothing reports
/// `formula_absent`.
pub fn enumerate_device_order(
    domain: &EnumDomain,
    bounds: &RatioBounds,
    query: &EnumQuery,
    limits: &DeviceEnumLimits,
) -> Result<DeviceEnumResult> {
    // Contracts §3.1: the tolerance never wraps. Validated on entry, before
    // any status-dependent early return (unknown precision, bad parent,
    // wide window), exactly like [`enumerate`].
    if query.ppm_tenths > 1000 {
        return Err(Error::config(format!(
            "enumerate_device_order: ppm_tenths {} exceeds 1000 (contracts §3.1)",
            query.ppm_tenths
        )));
    }
    validate_device_artifacts(domain, bounds)?;
    let rare = rare_table(domain, bounds)?;
    let packed = pack_device_bounds(domain, bounds)?;
    let lanes_zero = || {
        rare.iter()
            .map(|_| DeviceLaneStats {
                joined: 0,
                visited: 0,
                exhausted: false,
            })
            .collect::<Vec<DeviceLaneStats>>()
    };
    if query.precursor_uncertainty == u32::MAX {
        return Ok(DeviceEnumResult {
            parent_mass: parent_mass(query.precursor_mz, query.adduct).ok(),
            compositions: Vec::new(),
            masses: Vec::new(),
            flags: Vec::new(),
            lanes: lanes_zero(),
            visited: 0,
            joined: 0,
            scored: 0,
            exhausted: false,
            absent: true,
            complete: false,
            status: request_status::EXACT_MASS_UNAVAILABLE | request_status::FORMULA_ABSENT,
        });
    }
    let parent = match parent_mass(query.precursor_mz, query.adduct) {
        Ok(parent) => parent,
        Err(_) => {
            return Ok(DeviceEnumResult {
                parent_mass: None,
                compositions: Vec::new(),
                masses: Vec::new(),
                flags: Vec::new(),
                lanes: lanes_zero(),
                visited: 0,
                joined: 0,
                scored: 0,
                exhausted: false,
                absent: false,
                complete: false,
                status: request_status::MASS_OVERFLOW,
            });
        }
    };
    // Exact for `ppm_tenths <= 1000` (rejected above otherwise): never
    // narrowed by a wrapping cast.
    let tol = super::chem::tolerance_u32(query.precursor_mz, query.ppm_tenths)?;
    let bound = query.precursor_uncertainty.saturating_add(1);
    // `half` with saturating additions, exactly as [`enumerate`] saturates.
    let half = tol
        .saturating_add(bound)
        .saturating_add(domain.max_error());
    if half > DEVICE_HALF_MAX {
        return Ok(DeviceEnumResult {
            parent_mass: Some(parent),
            compositions: Vec::new(),
            masses: Vec::new(),
            flags: Vec::new(),
            lanes: lanes_zero(),
            visited: 0,
            joined: 0,
            scored: 0,
            exhausted: true,
            absent: false,
            complete: false,
            status: request_status::FORMULA_SEARCH_EXHAUSTED,
        });
    }
    let lo = parent.saturating_sub(half);
    let hi = parent.saturating_add(half);
    let meta_all: [u32; 8] = [
        parent,
        tol,
        bound,
        lo,
        hi,
        limits.lane_visits_max,
        limits.scored_cap,
        0,
    ];
    let meta_len = META_LEN;
    let packed_len = packed.len() as u32;
    let rare_all: Vec<u32> = rare.iter().flat_map(|row| row.iter().copied()).collect();
    let rare_len = rare_all.len() as u32;
    let stats_len = rare.len() as u32 * 2u32;
    let chem = chem_words();
    // Count pass: the same lane function in mode 0 for every rare row. Each
    // lane writes its `(joined, visited)` pair into the stats buffer; the
    // record output is inert in mode 0.
    let mut lane_stats = vec![0u32; rare.len().saturating_mul(2)];
    for r in 0..rare.len() {
        let rbase = r as u32 * 8u32;
        let stats_base = r as u32 * 2u32;
        kernel_lane(
            &meta_all, meta_len, &packed, packed_len, &rare_all, rare_len, 0u32,
            rbase, LANE_MODE_COUNT, 0u32, 0u32, &mut lane_stats, stats_len, 0u32,
            stats_base, chem[0], chem[1], chem[2], chem[3], chem[4], chem[5],
            chem[6], chem[7], chem[8], chem[9], chem[10], chem[11], chem[12],
            chem[13], u32::MAX, limits.lane_visits_max,
        );
    }
    // Offsets pass: clamped offsets plus the 5 counters.
    let mut offsets = vec![0u32; rare.len()];
    let mut counters = vec![0u32; 5];
    kernel_offsets(
        &lane_stats,
        stats_len,
        &meta_all,
        meta_len,
        &mut offsets,
        rare.len() as u32,
        &mut counters,
        5u32,
        0u32,
        rare.len() as u32,
        limits.scored_cap,
        request_status::FORMULA_SEARCH_EXHAUSTED,
        request_status::FORMULA_ABSENT,
        u32::MAX,
    );
    let visited_total = counters[0];
    let joined_total = counters[1];
    let scored = counters[2];
    // Fill pass: the SAME lane function in mode 1 at the clamped offsets,
    // writing records into the scored-prefix buffer (the stats words are
    // untouched in mode 1).
    let words = (scored as usize)
        .checked_mul(LANE_RECORD_WORDS as usize)
        .ok_or_else(|| Error::config("enumerate_device_order: scored record buffer overflows usize"))?;
    let mut out = vec![0u32; words];
    let out_len = words as u32;
    for r in 0..rare.len() {
        let rbase = r as u32 * 8u32;
        kernel_lane(
            &meta_all, meta_len, &packed, packed_len, &rare_all, rare_len, 0u32,
            rbase, LANE_MODE_FILL, offsets[r], limits.scored_cap, &mut out, out_len,
            0u32, 0u32, chem[0], chem[1], chem[2], chem[3], chem[4], chem[5],
            chem[6], chem[7], chem[8], chem[9], chem[10], chem[11], chem[12],
            chem[13], u32::MAX, limits.lane_visits_max,
        );
    }
    // Pad pass: the host stores exactly the scored prefix, so no padding
    // slot is stored here; on the device [`kernel_pad`] writes slots
    // `[scored, cap)` of the `[B, M, 13]` candidate buffer, tested
    // separately slot by slot.
    // Convert the scored prefix (host side: `u16` assembly and storage).
    let mut compositions: Vec<Composition> = Vec::with_capacity(scored as usize);
    let mut masses: Vec<u32> = Vec::with_capacity(scored as usize);
    let mut flags: Vec<u8> = Vec::with_capacity(scored as usize);
    for s in 0..scored {
        let base = (s as usize) * (LANE_RECORD_WORDS as usize);
        let word = |w: usize| out[base + w];
        debug_assert!(
            word(0) <= u32::from(u16::MAX)
                && word(1) <= u32::from(u16::MAX)
                && word(2) <= u32::from(u16::MAX)
                && word(3) <= u32::from(u16::MAX)
                && word(4) <= u32::from(u16::MAX)
                && word(5) <= u32::from(u16::MAX)
                && word(6) <= u32::from(u16::MAX)
                && word(7) <= u32::from(u16::MAX)
                && word(8) <= u32::from(u16::MAX)
                && word(9) <= u32::from(u16::MAX)
        );
        compositions.push(
            [
                word(0) as u16,
                word(1) as u16,
                word(2) as u16,
                word(3) as u16,
                word(4) as u16,
                word(5) as u16,
                word(6) as u16,
                word(7) as u16,
                word(8) as u16,
                word(9) as u16,
            ],
        );
        masses.push(word(10));
        flags.push(word(11) as u8);
        debug_assert!(out[base + 12] == u32::MAX);
    }
    let mut lanes: Vec<DeviceLaneStats> = Vec::with_capacity(rare.len());
    for r in 0..rare.len() {
        let visited_word = lane_stats[2 * r + 1];
        lanes.push(DeviceLaneStats {
            joined: lane_stats[2 * r],
            visited: visited_word & LANE_VISITED_MASK,
            exhausted: visited_word & LANE_EXHAUSTED_BIT != 0,
        });
    }
    let status = counters[3];
    let complete = counters[4] != 0;
    let exhausted = !complete;
    let absent = !exhausted && joined_total == 0;
    Ok(DeviceEnumResult {
        parent_mass: Some(parent),
        compositions,
        masses,
        flags,
        lanes,
        visited: visited_total,
        joined: joined_total,
        scored,
        exhausted,
        absent,
        complete,
        status,
    })
}

/// Build the per-spectrum `[B, 8]` enum meta rows for the device lanes
/// (spec §1.4): `[parent, tol, bound, lo, hi, budget, scored_cap, 0]`.
///
/// HOST side, shared by generation and training so both call the wrappers
/// the same way: the query fields come from the host [`SpectrumBatch`]
/// (precursor, adduct, tolerances, uncertainty), `domain_max_error` is the
/// resident domain's largest composition bound, `lane_visits_max` the
/// per-lane budget and `scored_cap` the scored capacity
/// `min(formula_rows_scored_max, M)`.
///
/// * Unknown precision (`precursor_uncertainty == u32::MAX`) or an invalid
///   parent mass: an empty window (`lo = 1`, `hi = 0`) with no visits, so
///   the lanes join nothing; the final statuses come from the host
///   (`exact_mass_unavailable` / `mass_overflow`) ORed in readout, matching
///   [`enumerate_device_order`].
/// * A window half above [`DEVICE_HALF_MAX`]: the normal `lo`/`hi` with
///   `budget = 0`, so the first visit exhausts the lane with nothing
///   joined (`formula_search_exhausted`), the scope restriction of §1.4.
/// * Otherwise the normal `lo`/`hi` with `budget = lane_visits_max`.
pub fn build_enum_meta(
    batch: &SpectrumBatch,
    domain_max_error: u32,
    lane_visits_max: u32,
    scored_cap: u32,
) -> Vec<u32> {
    let b = batch.len();
    let mut meta = vec![0u32; b * META_LEN as usize];
    for i in 0..b {
        let precursor = batch.precursor_mz_udalton[i];
        let adduct = batch.adduct[i];
        let unc = batch.precursor_uncertainty_udalton[i];
        let ppm = batch.precursor_tolerance(i);
        let (parent_ok, parent) = match parent_mass(precursor, adduct) {
            Ok(p) => (true, p),
            Err(_) => (false, 0),
        };
        if unc == u32::MAX || !parent_ok {
            meta[i * 8] = 0;
            meta[i * 8 + 1] = 0;
            meta[i * 8 + 2] = 0;
            meta[i * 8 + 3] = 1;
            meta[i * 8 + 4] = 0;
            meta[i * 8 + 5] = lane_visits_max;
            meta[i * 8 + 6] = scored_cap;
            meta[i * 8 + 7] = 0;
            continue;
        }
        let tol = tolerance_u32(precursor, ppm).unwrap_or(0);
        let bound = unc.saturating_add(1);
        let half = tol
            .saturating_add(bound)
            .saturating_add(domain_max_error);
        let lo = parent.saturating_sub(half);
        let hi = parent.saturating_add(half);
        let budget = if half > DEVICE_HALF_MAX {
            0
        } else {
            lane_visits_max
        };
        meta[i * 8] = parent;
        meta[i * 8 + 1] = tol;
        meta[i * 8 + 2] = bound;
        meta[i * 8 + 3] = lo;
        meta[i * 8 + 4] = hi;
        meta[i * 8 + 5] = budget;
        meta[i * 8 + 6] = scored_cap;
        meta[i * 8 + 7] = 0;
    }
    meta
}
