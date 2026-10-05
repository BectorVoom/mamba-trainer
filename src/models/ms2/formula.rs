//! Reference formula table and peak-relation edges (contract §9).
//!
//! The [`FormulaTable`] is the host reference the GPU precursor window
//! reproduces exactly, counters included; [`loss_edges`] is the deterministic
//! reference for the optional sparse-relation experiment. Pure host Rust with
//! integer masses only: no kernels, no tensors, no new dependencies.

use crate::error::{Error, Result};

use super::chem::{
    Composition, ELEMENTS, composition_error_nda, composition_mass, decide, parent_mass, tolerance,
};
use super::contract::request_status;

/// Bytes per formula-table row: ten `u16` counts plus a `u32` mass.
pub const FORMULA_ROW_BYTES: usize = 24;

/// One table row: an element composition with its integer mass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FormulaRow {
    /// Integer mass in units of 10⁻⁶ dalton.
    mass: u32,
    /// Atom count per element id.
    composition: Composition,
}

/// The V0 formula table: distinct compositions sorted by (mass, composition).
///
/// Rows hold ten `u16` element counts and a `u32` mass, 24 bytes each.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FormulaTable {
    /// Rows sorted by `(mass, composition)`, without duplicates.
    rows: Vec<FormulaRow>,
}

impl FormulaTable {
    /// Build a table from compositions: masses via [`composition_mass`]
    /// (checked, so an overflowing composition is an error), then sort by
    /// `(mass, composition)` and deduplicate.
    pub fn from_compositions(rows: impl IntoIterator<Item = Composition>) -> Result<Self> {
        let mut table: Vec<FormulaRow> = Vec::new();
        for composition in rows {
            table.push(FormulaRow {
                mass: composition_mass(&composition)?,
                composition,
            });
        }
        table.sort_by_key(|r| (r.mass, r.composition));
        table.dedup_by_key(|r| (r.mass, r.composition));
        Ok(Self { rows: table })
    }

    /// Read the `tools/ms2/formula_table.py` JSON format
    /// (`{"elements": [...], "rows": [[mass, [counts...]], ...]}`).
    ///
    /// Verifies the element order equals [`ELEMENTS`], every stored mass
    /// equals [`composition_mass`] of its counts, and the rows are strictly
    /// increasing in `(mass, composition)`; each error names its row.
    pub fn from_json(text: &str) -> Result<Self> {
        let value: serde_json::Value = serde_json::from_str(text)?;
        let elements = value
            .get("elements")
            .and_then(|e| e.as_array())
            .ok_or_else(|| {
                Error::config("formula_table: missing string array 'elements'".to_string())
            })?;
        if elements.len() != ELEMENTS.len() {
            return Err(Error::config(format!(
                "formula_table: expected {} elements, found {}",
                ELEMENTS.len(),
                elements.len()
            )));
        }
        for (i, (got, want)) in elements.iter().zip(ELEMENTS.iter()).enumerate() {
            if got.as_str() != Some(want.symbol) {
                return Err(Error::config(format!(
                    "formula_table: element order mismatch at position {i}: \
                     expected {:?}, found {got}",
                    want.symbol
                )));
            }
        }
        let rows = value
            .get("rows")
            .and_then(|r| r.as_array())
            .ok_or_else(|| Error::config("formula_table: missing array 'rows'".to_string()))?;
        let mut table = Vec::with_capacity(rows.len());
        let mut prev: Option<(u32, Composition)> = None;
        for (i, row) in rows.iter().enumerate() {
            let fail = |why: String| Error::config(format!("formula_table row {i}: {why}"));
            let pair = row
                .as_array()
                .ok_or_else(|| fail("row is not a [mass, counts] pair".to_string()))?;
            if pair.len() != 2 {
                return Err(fail(format!("expected 2 entries, found {}", pair.len())));
            }
            let mass = pair[0]
                .as_u64()
                .filter(|&m| m <= u64::from(u32::MAX))
                .ok_or_else(|| fail(format!("bad mass {}", pair[0])))?
                as u32;
            let counts = pair[1]
                .as_array()
                .ok_or_else(|| fail("counts are not an array".to_string()))?;
            if counts.len() != ELEMENTS.len() {
                return Err(fail(format!(
                    "expected {} element counts, found {}",
                    ELEMENTS.len(),
                    counts.len()
                )));
            }
            let mut composition: Composition = [0; 10];
            for (e, count) in counts.iter().enumerate() {
                composition[e] = count
                    .as_u64()
                    .filter(|&c| c <= u64::from(u16::MAX))
                    .ok_or_else(|| fail(format!("bad count {}", count)))?
                    as u16;
            }
            let expected = composition_mass(&composition).map_err(|e| fail(e.to_string()))?;
            if mass != expected {
                return Err(fail(format!(
                    "stored mass {mass} does not equal composition mass {expected}"
                )));
            }
            if let Some((prev_mass, prev_comp)) = prev
                && (mass, composition) <= (prev_mass, prev_comp)
            {
                return Err(fail(
                    "rows are not strictly increasing in (mass, composition)".to_string(),
                ));
            }
            prev = Some((mass, composition));
            table.push(FormulaRow { mass, composition });
        }
        Ok(Self { rows: table })
    }

    /// Write the `tools/ms2/formula_table.py` JSON format: compact output
    /// byte-identical to the Python writer's for the same rows.
    pub fn to_json(&self) -> String {
        let mut out = String::from("{\"elements\":[");
        for (i, e) in ELEMENTS.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push('"');
            out.push_str(e.symbol);
            out.push('"');
        }
        out.push_str("],\"rows\":[");
        for (i, row) in self.rows.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push('[');
            out.push_str(&row.mass.to_string());
            out.push_str(",[");
            for (e, count) in row.composition.iter().enumerate() {
                if e > 0 {
                    out.push(',');
                }
                out.push_str(&count.to_string());
            }
            out.push_str("]]");
        }
        out.push_str("]}");
        out
    }

    /// Number of rows.
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Whether the table holds no row.
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Serialized size: rows times [`FORMULA_ROW_BYTES`].
    pub fn bytes(&self) -> usize {
        self.rows.len() * FORMULA_ROW_BYTES
    }

    /// Integer mass of a row.
    ///
    /// # Panics
    ///
    /// Panics when `row` is out of range.
    pub fn mass(&self, row: usize) -> u32 {
        self.rows[row].mass
    }

    /// Element composition of a row.
    ///
    /// # Panics
    ///
    /// Panics when `row` is out of range.
    pub fn composition(&self, row: usize) -> &Composition {
        &self.rows[row].composition
    }

    /// Largest per-row arithmetic bound: the maximum over rows of
    /// `ceil(composition_error_nda / 1000)` (0 for an empty table).
    pub fn max_error(&self) -> u32 {
        self.rows
            .iter()
            .map(|r| composition_error_nda(&r.composition).div_ceil(1000) as u32)
            .max()
            .unwrap_or(0)
    }

    /// Largest hydrogen count over all rows (0 for an empty table): the
    /// host-known bound for the evidence dispatch sizing (`h_cap_max =
    /// max_hydrogen + 3`), known without a device read.
    pub fn max_hydrogen(&self) -> u16 {
        self.rows
            .iter()
            .map(|r| r.composition[super::chem::HYDROGEN])
            .max()
            .unwrap_or(0)
    }

    /// Precursor window search with the exact counters the GPU kernel
    /// reproduces (contract §9), in the contract's six steps:
    ///
    /// 1. `parent = parent_mass(precursor_mz, adduct)`: when that leaves the
    ///    `u32` range the request is `mass_overflow` (fatal) and nothing is
    ///    searched. An unknown precursor precision (`u32::MAX`) is
    ///    `exact_mass_unavailable` and `formula_absent`, with nothing
    ///    searched.
    /// 2. `tol = tol(precursor, precursor tolerance)`,
    ///    `bound = precursor uncertainty + 1` (the adduct's arithmetic
    ///    bound). The **superset window** is `parent ± (tol + bound +
    ///    E_table)`, `E_table` the largest row bound of the table; its ends
    ///    are found by two halving searches over the sorted masses (lower
    ///    bound of the low end, upper bound of the high end), saturating at
    ///    0 and `u32::MAX`.
    /// 3. Every row of the superset window gets the verdict of §5 with
    ///    `E = E_row + bound`. Accepted and ambiguous rows are **joined**
    ///    (an ambiguous row is flagged); rejected rows are not.
    /// 4. `rows_visited` counts every mass read: one per halving step, one
    ///    per row of the superset window. `rows_joined` counts joined rows.
    ///    The first `min(rows_scored_max, M)` joined rows in table order are
    ///    **scored**; `rows_scored` counts them.
    /// 5. A search that would need more than `rows_visited_max` reads stops
    ///    there and is **exhausted**: it keeps the counters and joined rows
    ///    it has, and the request carries `formula_search_exhausted`. More
    ///    joined rows than can be scored is exhausted too.
    /// 6. `formula_absent` (fatal) means the search **completed** and joined
    ///    no row. `formula_support_complete` is `1` exactly when the search
    ///    completed and every joined row was scored; only then are the
    ///    formula probabilities and `formula_mass_retained` statements about
    ///    the whole window. Equal `rows_scored` and `rows_joined` alone do
    ///    not show that.
    ///
    /// `status` carries the request bits of step 1 and steps 5–6;
    /// `complete` is true exactly when the search finished and every joined
    /// row was scored (never when `exhausted`), and `absent` exactly when
    /// the search completed and joined nothing.
    pub fn window(&self, query: &WindowQuery) -> WindowResult {
        // Score the joined rows of a finished scan: past `rows_scored_max`
        // only the first rows (table order) are kept and the search is
        // exhausted by the cap.
        let finish = |parent_mass: Option<u32>,
                      first: usize,
                      mut joined: Vec<usize>,
                      mut ambiguous: Vec<bool>,
                      visited: u64,
                      status: u32| {
            let rows_joined = joined.len() as u32;
            let (rows_scored, exhausted, status) = if rows_joined > query.rows_scored_max {
                joined.truncate(query.rows_scored_max as usize);
                ambiguous.truncate(query.rows_scored_max as usize);
                (
                    query.rows_scored_max,
                    true,
                    status | request_status::FORMULA_SEARCH_EXHAUSTED,
                )
            } else {
                (rows_joined, false, status)
            };
            let absent = !exhausted && rows_joined == 0;
            let status = if absent {
                status | request_status::FORMULA_ABSENT
            } else {
                status
            };
            WindowResult {
                parent_mass,
                first,
                joined,
                ambiguous,
                rows_visited: visited as u32,
                rows_joined,
                rows_scored,
                exhausted,
                absent,
                complete: !exhausted,
                status,
            }
        };
        // A search stopped by `rows_visited_max` keeps the counters and the
        // rows joined so far; `rows_scored` is then the scored cap over what
        // was joined.
        let stop = |parent: u32,
                    first: usize,
                    mut joined: Vec<usize>,
                    mut ambiguous: Vec<bool>,
                    visited: u64| {
            let rows_joined = joined.len() as u32;
            let rows_scored = rows_joined.min(query.rows_scored_max);
            joined.truncate(rows_scored as usize);
            ambiguous.truncate(rows_scored as usize);
            WindowResult {
                parent_mass: Some(parent),
                first,
                joined,
                ambiguous,
                rows_visited: visited as u32,
                rows_joined,
                rows_scored,
                exhausted: true,
                absent: false,
                complete: false,
                status: request_status::FORMULA_SEARCH_EXHAUSTED,
            }
        };
        if query.precursor_uncertainty == u32::MAX {
            return WindowResult {
                parent_mass: parent_mass(query.precursor_mz, query.adduct).ok(),
                first: 0,
                joined: Vec::new(),
                ambiguous: Vec::new(),
                rows_visited: 0,
                rows_joined: 0,
                rows_scored: 0,
                exhausted: false,
                absent: true,
                complete: false,
                status: request_status::EXACT_MASS_UNAVAILABLE | request_status::FORMULA_ABSENT,
            };
        }
        let parent = match parent_mass(query.precursor_mz, query.adduct) {
            Ok(parent) => parent,
            Err(_) => {
                return WindowResult {
                    parent_mass: None,
                    first: 0,
                    joined: Vec::new(),
                    ambiguous: Vec::new(),
                    rows_visited: 0,
                    rows_joined: 0,
                    rows_scored: 0,
                    exhausted: false,
                    absent: false,
                    complete: false,
                    status: request_status::MASS_OVERFLOW,
                };
            }
        };
        let tol = tolerance(query.precursor_mz, query.ppm_tenths);
        let bound = query.precursor_uncertainty.saturating_add(1);
        let width = (u64::from(tol) + u64::from(bound) + u64::from(self.max_error()))
            .min(u64::from(u32::MAX)) as u32;
        let n = self.rows.len();
        let mut visited: u64 = 0;
        let mut lower = 0usize;
        let mut upper = n;
        while lower < upper {
            if visited + 1 > u64::from(query.rows_visited_max) {
                return stop(parent, 0, Vec::new(), Vec::new(), visited);
            }
            visited += 1;
            let mid = lower + (upper - lower) / 2;
            if self.rows[mid].mass < parent.saturating_sub(width) {
                lower = mid + 1;
            } else {
                upper = mid;
            }
        }
        let first = lower;
        let mut lo = 0usize;
        let mut hi = n;
        while lo < hi {
            if visited + 1 > u64::from(query.rows_visited_max) {
                return stop(parent, 0, Vec::new(), Vec::new(), visited);
            }
            visited += 1;
            let mid = lo + (hi - lo) / 2;
            if self.rows[mid].mass <= parent.saturating_add(width) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        let mut joined = Vec::new();
        let mut ambiguous = Vec::new();
        for row in first..lo {
            if visited + 1 > u64::from(query.rows_visited_max) {
                return stop(parent, first, joined, ambiguous, visited);
            }
            visited += 1;
            let row_error =
                composition_error_nda(&self.rows[row].composition).div_ceil(1000) as u32;
            match decide(
                parent,
                self.rows[row].mass,
                row_error.saturating_add(bound),
                tol,
            ) {
                super::chem::Verdict::Accept => {
                    joined.push(row);
                    ambiguous.push(false);
                }
                super::chem::Verdict::Ambiguous => {
                    joined.push(row);
                    ambiguous.push(true);
                }
                super::chem::Verdict::Reject => {}
            }
        }
        finish(Some(parent), first, joined, ambiguous, visited, 0)
    }
}

/// One precursor window search over a [`FormulaTable`].
#[derive(Clone, Copy, Debug)]
pub struct WindowQuery {
    /// Precursor m/z in integer units.
    pub precursor_mz: u32,
    /// Adduct id of the precursor.
    pub adduct: u16,
    /// Precursor tolerance in tenths of a ppm.
    pub ppm_tenths: u32,
    /// Precursor uncertainty; `u32::MAX` means unknown (skips the search).
    pub precursor_uncertainty: u32,
    /// Stop with `exhausted` once more mass reads would be needed.
    pub rows_visited_max: u32,
    /// Keep at most this many joined rows (table order) before exhausting.
    pub rows_scored_max: u32,
}

/// Outcome of [`FormulaTable::window`]; counters match the GPU kernel exactly.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WindowResult {
    /// Neutral parent mass, or `None` when [`parent_mass`] errors.
    pub parent_mass: Option<u32>,
    /// Lower-bound index of the superset window (0 on visited-exhaustion).
    pub first: usize,
    /// Joined table rows, in table order (truncated to the scored cap).
    pub joined: Vec<usize>,
    /// Per joined row, whether its verdict was Ambiguous.
    pub ambiguous: Vec<bool>,
    /// Mass reads: binary-search steps plus one per row in the window.
    pub rows_visited: u32,
    /// Rows inside the window before the scored cap.
    pub rows_joined: u32,
    /// Joined rows kept (the cap when exhausted by it).
    pub rows_scored: u32,
    /// A work limit was reached: the joined set may be incomplete.
    pub exhausted: bool,
    /// The search completed and no row is in the window.
    pub absent: bool,
    /// The search finished and every joined row was scored (never when
    /// `exhausted`).
    pub complete: bool,
    /// Request status bits of [`request_status`]: `MASS_OVERFLOW` when the
    /// parent mass leaves the `u32` range, `EXACT_MASS_UNAVAILABLE |
    /// FORMULA_ABSENT` for the unknown-precision sentinel,
    /// `FORMULA_SEARCH_EXHAUSTED` when a work limit stopped or truncated the
    /// search, `FORMULA_ABSENT` when the search completed and joined
    /// nothing.
    pub status: u32,
}

/// One neutral loss of the peak-relation experiment.
pub struct Loss {
    /// Short name, e.g. `"H2O"`.
    pub name: &'static str,
    /// Neutral composition of the loss.
    pub composition: Composition,
}

/// Build a composition from C/H/N/O counts (all other elements are zero).
const fn chno(c: u16, h: u16, n: u16, o: u16) -> Composition {
    [c, h, n, o, 0, 0, 0, 0, 0, 0]
}

/// The eight losses of the sparse-relation experiment: H2, H2O, NH3, CO, CO2,
/// the methyl radical (C1 H3), HCN and C2H2. Masses come from
/// [`composition_mass`], never from literals.
pub const LOSSES: [Loss; 8] = [
    Loss {
        name: "H2",
        composition: chno(0, 2, 0, 0),
    },
    Loss {
        name: "H2O",
        composition: chno(0, 2, 0, 1),
    },
    Loss {
        name: "NH3",
        composition: chno(0, 3, 1, 0),
    },
    Loss {
        name: "CO",
        composition: chno(1, 0, 0, 1),
    },
    Loss {
        name: "CO2",
        composition: chno(1, 0, 0, 2),
    },
    Loss {
        name: "CH3",
        composition: chno(1, 3, 0, 0),
    },
    Loss {
        name: "HCN",
        composition: chno(1, 1, 1, 0),
    },
    Loss {
        name: "C2H2",
        composition: chno(2, 2, 0, 0),
    },
];

/// Peak-relation edges: for each destination peak, up to `degree` sources.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EdgeTable {
    /// Edges kept per destination peak.
    pub degree: usize,
    /// Source peak per (destination, slot): `[n * degree]`, `u32::MAX` empty.
    pub source: Vec<u32>,
    /// Loss index per (destination, slot): `[n * degree]` (0 when empty).
    pub loss: Vec<u8>,
    /// Per destination peak: more candidates than `degree` existed.
    pub overflow: Vec<bool>,
}

/// Relate peaks by neutral losses.
///
/// For destination peak `i` and loss `l`, a source peak `j > i` is a
/// candidate when `decide(mz[j] − mz[i], mass(l),
/// ceil(error_nda(l)/1000) + 2 * mz_uncertainty, tolerance(mz[j],
/// ppm_tenths))` accepts. Candidates of `i` are ordered by (absolute
/// residual, loss index, source) and the first `degree` are kept;
/// `overflow[i]` marks a longer candidate list. Sources come from one binary
/// search per `(i, loss)` plus a scan of the mass window (the residual minus
/// the tolerance only grows past the loss mass while `ppm_tenths <= 10⁷`, so
/// the early stop is gated on that); an unsorted `mz` is an error.
pub fn loss_edges(
    mz: &[u32],
    ppm_tenths: u32,
    mz_uncertainty: u32,
    degree: usize,
) -> Result<EdgeTable> {
    for i in 1..mz.len() {
        if mz[i] < mz[i - 1] {
            return Err(Error::config(format!(
                "loss_edges: mz[{i}] {} is below mz[{}] {}: peaks must be ascending",
                mz[i],
                i - 1,
                mz[i - 1]
            )));
        }
    }
    let n = mz.len();
    let mut loss_mass = [0u32; 8];
    let mut loss_error = [0u32; 8];
    for (k, loss) in LOSSES.iter().enumerate() {
        loss_mass[k] = composition_mass(&loss.composition)?;
        loss_error[k] = (composition_error_nda(&loss.composition).div_ceil(1000) as u32)
            .saturating_add(mz_uncertainty.saturating_mul(2));
    }
    let max_tol = mz.last().map(|&m| tolerance(m, ppm_tenths)).unwrap_or(0);
    // The early stop below needs the tolerance to grow slower than the m/z
    // gap, i.e. `ppm_tenths / 10⁷ <= 1`.
    let can_stop_early = ppm_tenths <= 10_000_000;
    let mut source = vec![u32::MAX; n * degree];
    let mut loss = vec![0u8; n * degree];
    let mut overflow = vec![false; n];
    for i in 0..n {
        let mut candidates: Vec<(u32, usize, usize)> = Vec::new();
        for (k, _) in LOSSES.iter().enumerate() {
            let target = u64::from(mz[i]) + u64::from(loss_mass[k]);
            let pad = u64::from(loss_error[k]) + u64::from(max_tol);
            let mut j = mz.partition_point(|&m| u64::from(m) < target.saturating_sub(pad));
            if j < i + 1 {
                j = i + 1;
            }
            while j < n {
                let gap = u64::from(mz[j]) - u64::from(mz[i]);
                let mass = u64::from(loss_mass[k]);
                let residual = gap.abs_diff(mass);
                let tol = u64::from(tolerance(mz[j], ppm_tenths));
                let error = u64::from(loss_error[k]);
                if residual + error <= tol {
                    candidates.push((residual as u32, k, j));
                } else if can_stop_early && gap >= mass && gap - mass > tol + error {
                    break;
                }
                j += 1;
            }
        }
        candidates.sort();
        if candidates.len() > degree {
            overflow[i] = true;
        }
        for (slot, &(_, k, j)) in candidates.iter().take(degree).enumerate() {
            source[i * degree + slot] = j as u32;
            loss[i * degree + slot] = k as u8;
        }
    }
    Ok(EdgeTable {
        degree,
        source,
        loss,
        overflow,
    })
}
