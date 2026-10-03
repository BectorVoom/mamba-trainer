//! Integer chemistry domain of `docs/MS2_CONTRACTS.md` (§§4.1–4.3 and 5).
//!
//! Element masses, atom types, adducts and every mass computation are done in
//! integer units of 10⁻⁶ dalton (`u32`), so no mass decision ever sees a float.
//! This is the host reference the GPU kernels are later tested against: exact
//! before fast.

use crate::error::{Error, Result};

/// Integer mass units per dalton (contract §5).
pub const MASS_SCALE: u32 = 1_000_000;

/// Version string of the chemistry domain (contract §3.2).
pub const CHEMISTRY_VERSION: &str = "ms2-chem-v0.1";

/// One domain element of contract §4.1.
///
/// The index in [`ELEMENTS`] is the element id used by atom types and
/// compositions.
pub struct Element {
    /// Chemical symbol.
    pub symbol: &'static str,
    /// Defining exact mass as a decimal string.
    pub exact: &'static str,
    /// Exact mass rounded once to integer units of 10⁻⁶ dalton.
    pub mass: u32,
    /// `|exact * 10⁶ − mass|` in nano-dalton, rounded up.
    pub residual_nda: u32,
}

/// The ten V0 elements in id order: C, H, N, O, F, P, S, Cl, Br, I.
pub const ELEMENTS: [Element; 10] = [
    Element {
        symbol: "C",
        exact: "12",
        mass: 12_000_000,
        residual_nda: 0,
    },
    Element {
        symbol: "H",
        exact: "1.00782503223",
        mass: 1_007_825,
        residual_nda: 33,
    },
    Element {
        symbol: "N",
        exact: "14.00307400443",
        mass: 14_003_074,
        residual_nda: 5,
    },
    Element {
        symbol: "O",
        exact: "15.99491461957",
        mass: 15_994_915,
        residual_nda: 381,
    },
    Element {
        symbol: "F",
        exact: "18.99840316273",
        mass: 18_998_403,
        residual_nda: 163,
    },
    Element {
        symbol: "P",
        exact: "30.97376199842",
        mass: 30_973_762,
        residual_nda: 2,
    },
    Element {
        symbol: "S",
        exact: "31.9720711744",
        mass: 31_972_071,
        residual_nda: 175,
    },
    Element {
        symbol: "Cl",
        exact: "34.968852682",
        mass: 34_968_853,
        residual_nda: 318,
    },
    Element {
        symbol: "Br",
        exact: "78.9183376",
        mass: 78_918_338,
        residual_nda: 400,
    },
    Element {
        symbol: "I",
        exact: "126.9044719",
        mass: 126_904_472,
        residual_nda: 100,
    },
];

/// Index of hydrogen in [`ELEMENTS`].
pub const HYDROGEN: usize = 1;

/// Defining exact mass of the electron as a decimal string.
pub const ELECTRON_EXACT: &str = "0.000548579909065";
/// Electron mass in integer units.
pub const ELECTRON_MASS: u32 = 549;
/// Electron rounding residual in nano-dalton, rounded up.
pub const ELECTRON_RESIDUAL_NDA: u32 = 421;

/// Index of an element symbol in [`ELEMENTS`], or `None` when outside §4.1.
pub fn element_index(symbol: &str) -> Option<usize> {
    ELEMENTS.iter().position(|e| e.symbol == symbol)
}

/// One V0 atom type of contract §4.2: `(element, parent hydrogens, valence)`.
///
/// All atoms are neutral. The hydrogen count is the one the atom has in the
/// parent molecule and never changes; the valence is that count plus the
/// atom's bond orders in the kekulized parent.
pub struct AtomType {
    /// Type id, 1–17 (0 is padding and has no entry).
    pub id: u8,
    /// Index into [`ELEMENTS`].
    pub element: usize,
    /// Parent hydrogen count.
    pub hydrogens: u8,
    /// Valence: hydrogens plus bond orders in the kekulized parent.
    pub valence: u8,
}

/// The 17 V0 atom types in id order (contract §4.2).
pub const ATOM_TYPES: [AtomType; 17] = [
    AtomType {
        id: 1,
        element: 0,
        hydrogens: 0,
        valence: 4,
    },
    AtomType {
        id: 2,
        element: 0,
        hydrogens: 1,
        valence: 4,
    },
    AtomType {
        id: 3,
        element: 0,
        hydrogens: 2,
        valence: 4,
    },
    AtomType {
        id: 4,
        element: 0,
        hydrogens: 3,
        valence: 4,
    },
    AtomType {
        id: 5,
        element: 2,
        hydrogens: 0,
        valence: 3,
    },
    AtomType {
        id: 6,
        element: 2,
        hydrogens: 1,
        valence: 3,
    },
    AtomType {
        id: 7,
        element: 2,
        hydrogens: 2,
        valence: 3,
    },
    AtomType {
        id: 8,
        element: 3,
        hydrogens: 0,
        valence: 2,
    },
    AtomType {
        id: 9,
        element: 3,
        hydrogens: 1,
        valence: 2,
    },
    AtomType {
        id: 10,
        element: 4,
        hydrogens: 0,
        valence: 1,
    },
    AtomType {
        id: 11,
        element: 7,
        hydrogens: 0,
        valence: 1,
    },
    AtomType {
        id: 12,
        element: 8,
        hydrogens: 0,
        valence: 1,
    },
    AtomType {
        id: 13,
        element: 6,
        hydrogens: 0,
        valence: 2,
    },
    AtomType {
        id: 14,
        element: 6,
        hydrogens: 1,
        valence: 2,
    },
    AtomType {
        id: 15,
        element: 6,
        hydrogens: 0,
        valence: 6,
    },
    AtomType {
        id: 16,
        element: 5,
        hydrogens: 0,
        valence: 5,
    },
    AtomType {
        id: 17,
        element: 9,
        hydrogens: 0,
        valence: 1,
    },
];

/// The atom type with this id, or `None` for 0 and anything above 17.
pub fn atom_type(id: u8) -> Option<&'static AtomType> {
    if id == 0 {
        return None;
    }
    ATOM_TYPES.get(usize::from(id) - 1)
}

/// The atom type with this `(element, hydrogens, valence)`, if it is in §4.2.
pub fn atom_type_of(element: usize, hydrogens: u8, valence: u8) -> Option<&'static AtomType> {
    ATOM_TYPES
        .iter()
        .find(|t| t.element == element && t.hydrogens == hydrogens && t.valence == valence)
}

/// One V0 adduct of contract §4.3 (id 0 means unknown and has no entry).
pub struct Adduct {
    /// Adduct id.
    pub id: u16,
    /// Adduct name, e.g. `[M+H]+`.
    pub name: &'static str,
    /// Hydrogens added to the neutral composition.
    pub hydrogens: i32,
    /// Signed charge.
    pub charge: i32,
}

/// The two V0 adducts.
pub const ADDUCTS: [Adduct; 2] = [
    Adduct {
        id: 1,
        name: "[M+H]+",
        hydrogens: 1,
        charge: 1,
    },
    Adduct {
        id: 2,
        name: "[M-H]-",
        hydrogens: -1,
        charge: -1,
    },
];

/// The adduct with this id, or `None` for 0 and unknown ids.
pub fn adduct(id: u16) -> Option<&'static Adduct> {
    ADDUCTS.iter().find(|a| a.id == id)
}

/// Atom count per element id; hydrogens live at [`HYDROGEN`].
pub type Composition = [u16; 10];

/// The largest value [`parse_decimal`] accepts, in integer units (`u32::MAX`).
const MAX_UDALTON: u64 = u32::MAX as u64;

/// Parse an exact decimal string to integer units, rounding half to even.
///
/// Accepts `digits[.digits]` only: no sign, no exponent, no empty parts.
/// Values above 4294.967295 overflow the `u32` range and are rejected.
pub fn parse_decimal(text: &str) -> Result<u32> {
    let invalid = || Error::config(format!("parse_decimal: invalid decimal string {text:?}"));
    if text.is_empty() {
        return Err(invalid());
    }
    let mut parts = text.split('.');
    let int_part = parts.next().ok_or_else(invalid)?;
    let frac_part = parts.next();
    if parts.next().is_some() {
        return Err(invalid());
    }
    if int_part.is_empty() || !int_part.bytes().all(|b| b.is_ascii_digit()) {
        return Err(invalid());
    }
    let frac = frac_part.unwrap_or("");
    // A present point must carry digits; a missing point means zero fraction.
    if frac_part.is_some() && (frac.is_empty() || !frac.bytes().all(|b| b.is_ascii_digit())) {
        return Err(invalid());
    }
    if !frac.bytes().all(|b| b.is_ascii_digit()) {
        return Err(invalid());
    }
    let mut int_val: u64 = 0;
    for b in int_part.bytes() {
        int_val = int_val
            .checked_mul(10)
            .and_then(|v| v.checked_add(u64::from(b - b'0')))
            .ok_or_else(|| {
                Error::config(format!(
                    "parse_decimal: {text:?} overflows u32 micro-dalton range"
                ))
            })?;
    }
    let base: u64 = if frac.len() <= 6 {
        let mut v: u64 = 0;
        for b in frac.bytes() {
            v = v * 10 + u64::from(b - b'0');
        }
        for _ in frac.len()..6 {
            v *= 10;
        }
        v
    } else {
        let head = &frac[..6];
        let rest = &frac[6..];
        let mut head_val: u64 = 0;
        for b in head.bytes() {
            head_val = head_val * 10 + u64::from(b - b'0');
        }
        // Round the dropped tail half to even: compare against 5 followed by
        // zeros (same length, so lexicographic order is numeric order).
        let mut half = String::with_capacity(rest.len());
        half.push('5');
        for _ in 1..rest.len() {
            half.push('0');
        }
        let round_up = if rest > half.as_str() {
            true
        } else if rest < half.as_str() {
            false
        } else {
            head_val % 2 == 1
        };
        head_val + u64::from(round_up)
    };
    let scaled = int_val
        .checked_mul(1_000_000)
        .and_then(|v| v.checked_add(base))
        .ok_or_else(|| {
            Error::config(format!(
                "parse_decimal: {text:?} overflows u32 micro-dalton range"
            ))
        })?;
    if scaled > MAX_UDALTON {
        return Err(Error::config(format!(
            "parse_decimal: {text:?} overflows u32 micro-dalton range"
        )));
    }
    Ok(scaled as u32)
}

/// Integer mass of a neutral composition; `Err` on `u32` overflow.
pub fn composition_mass(c: &Composition) -> Result<u32> {
    let mut total: u64 = 0;
    for (e, n) in c.iter().enumerate() {
        total += u64::from(*n) * u64::from(ELEMENTS[e].mass);
    }
    if total > MAX_UDALTON {
        return Err(Error::Unsupported(format!(
            "mass_overflow: composition mass {total} exceeds u32 range"
        )));
    }
    Ok(total as u32)
}

/// Representation error bound of a neutral composition, in nano-dalton.
pub fn composition_error_nda(c: &Composition) -> u64 {
    c.iter()
        .enumerate()
        .map(|(e, n)| u64::from(*n) * u64::from(ELEMENTS[e].residual_nda))
        .sum()
}

/// Neutral parent mass from a precursor m/z under an adduct (contract §4.3).
///
/// For these singly charged single-molecule adducts this is
/// `mz − hydrogens * m_H + charge * m_e`. Unknown adduct ids are an error,
/// as is any under/overflowing intermediate.
pub fn parent_mass(precursor_mz: u32, adduct_id: u16) -> Result<u32> {
    let a = adduct(adduct_id).ok_or_else(|| {
        Error::Unsupported(format!("unsupported_adduct: unknown adduct id {adduct_id}"))
    })?;
    let signed = i64::from(precursor_mz)
        - i64::from(a.hydrogens) * i64::from(ELEMENTS[HYDROGEN].mass)
        + i64::from(a.charge) * i64::from(ELECTRON_MASS);
    if signed < 0 || signed > i64::from(u32::MAX) {
        return Err(Error::Unsupported(format!(
            "mass_overflow: parent mass of precursor {precursor_mz} under adduct {adduct_id} leaves u32 range"
        )));
    }
    Ok(signed as u32)
}

/// A fragment-ion hypothesis: computed m/z and its arithmetic bound `E`.
pub struct Ion {
    /// Computed m/z in integer units.
    pub mz: u32,
    /// The bound `E` of contract §5, in integer units.
    pub error: u32,
}

/// m/z of ion hypothesis `(composition, adduct, shift)` (contract §4.3).
///
/// Returns `Ok(None)` when the ion's hydrogen count would be negative; any
/// overflowing intermediate is `Err`.
pub fn ion(c: &Composition, adduct_id: u16, shift: i32) -> Result<Option<Ion>> {
    let a = adduct(adduct_id).ok_or_else(|| {
        Error::Unsupported(format!("unsupported_adduct: unknown adduct id {adduct_id}"))
    })?;
    let extra_h = i64::from(a.hydrogens) + i64::from(shift);
    let hydrogens = i64::from(c[HYDROGEN]) + extra_h;
    if hydrogens < 0 {
        return Ok(None);
    }
    let base = i64::from(composition_mass(c)?);
    let mz = base + extra_h * i64::from(ELEMENTS[HYDROGEN].mass)
        - i64::from(a.charge) * i64::from(ELECTRON_MASS);
    if mz < 0 || mz > i64::from(u32::MAX) {
        return Err(Error::Unsupported(format!(
            "mass_overflow: ion m/z leaves u32 range (adduct {adduct_id}, shift {shift})"
        )));
    }
    // The ion's own composition is `c` with its hydrogen count changed by
    // `extra_h`; its bound adds the electron's residual, rounded up.
    let mut ion_c = *c;
    ion_c[HYDROGEN] = hydrogens as u16;
    let nda = composition_error_nda(&ion_c) + u64::from(ELECTRON_RESIDUAL_NDA);
    Ok(Some(Ion {
        mz: mz as u32,
        error: nda.div_ceil(1000) as u32,
    }))
}

/// `floor(mz * ppm_tenths / 10^7)` at the observed m/z (contract §5).
pub fn tolerance(mz: u32, ppm_tenths: u32) -> u32 {
    ((u64::from(mz) * u64::from(ppm_tenths)) / 10_000_000) as u32
}

/// The same tolerance using only `u32` arithmetic (the GPU-side algorithm).
///
/// Splits `mz = hi * 10^4 + lo` and returns
/// `q / 1000 + ((q % 1000) * 10^4 + lo * t) / 10^7` with `q = hi * t`.
///
/// No intermediate exceeds `u32` for `t <= 1000` and any `u32` `mz`: `hi <=
/// 429496` (since `mz <= 4294967295`), so `q = hi * t <= 429496000`; and
/// `(q % 1000) * 10^4 <= 9990000` while `lo * t <= 9999000`, so the second
/// numerator is at most `19989000`. Both fit in `u32` with wide margin.
/// Larger `t` is rejected because the proof above no longer holds.
pub fn tolerance_u32(mz: u32, ppm_tenths: u32) -> Result<u32> {
    if ppm_tenths > 1000 {
        return Err(Error::config(format!(
            "tolerance_u32: ppm_tenths {ppm_tenths} exceeds the 1000 proof bound"
        )));
    }
    let hi = mz / 10_000;
    let lo = mz % 10_000;
    let q = hi * ppm_tenths;
    Ok(q / 1000 + ((q % 1000) * 10_000 + lo * ppm_tenths) / 10_000_000)
}

/// Outcome of the exact-mass decision rule of contract §5.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// `r + error <= tolerance`: the hypothesis explains the peak.
    Accept,
    /// `r > tolerance + error`: the hypothesis is ruled out.
    Reject,
    /// Neither: counted, but neither a label nor a rejection.
    Ambiguous,
}

/// Decide a hypothesis with `r = |observed − computed|` in integer units.
///
/// Accepts when `r + error <= tolerance`, rejects when
/// `r > tolerance + error`, else reports ambiguity. The sums run in `u64`
/// so no input pair can overflow them.
pub fn decide(observed: u32, computed: u32, error: u32, tolerance: u32) -> Verdict {
    let r = observed.abs_diff(computed) as u64;
    if r + u64::from(error) <= u64::from(tolerance) {
        Verdict::Accept
    } else if r > u64::from(tolerance) + u64::from(error) {
        Verdict::Reject
    } else {
        Verdict::Ambiguous
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Re-derives each table mass and residual from its defining decimal
    // string with integer arithmetic only (u128 scale ladder, no f64):
    // mass is round-half-even of exact * 10^6, residual is the ceiling of
    // |exact * 10^9 − mass * 1000| / 10^frac_digits.
    fn derive(decimal: &str) -> (u32, u32) {
        let (int_part, frac_part) = match decimal.split_once('.') {
            Some((i, f)) => (i, f),
            None => (decimal, ""),
        };
        let mut digits: u128 = 0;
        for b in int_part.bytes().chain(frac_part.bytes()) {
            digits = digits * 10 + u128::from(b - b'0');
        }
        let f = frac_part.len() as u32;
        let num = digits * 1_000_000;
        let den = 10u128.pow(f);
        let q = num / den;
        let r = num % den;
        // Round half to even: up past the half, and at the half only from an odd quotient.
        let round_up = 2 * r > den || (2 * r == den && q % 2 == 1);
        let mass = q + u128::from(round_up);
        let num_nda = digits * 1_000_000_000;
        let den_nda = den;
        let exact_floor = num_nda / den_nda;
        let mass_nda = mass * 1000;
        let diff = exact_floor.abs_diff(mass_nda);
        let rem = num_nda % den_nda;
        // Ceiling of the distance from the exact value. Above the integer the
        // dropped fraction lengthens the floor distance, so it rounds up; below
        // it the fraction shortens that distance, whose ceiling is the floor
        // distance itself.
        let residual = if exact_floor >= mass_nda {
            diff + u128::from(rem != 0)
        } else {
            diff
        };
        (mass as u32, residual as u32)
    }

    #[test]
    fn element_table_rederives_from_exact_strings() {
        for e in ELEMENTS.iter() {
            let (mass, residual) = derive(e.exact);
            assert_eq!(mass, e.mass, "mass of {}", e.symbol);
            assert_eq!(residual, e.residual_nda, "residual of {}", e.symbol);
        }
        let (mass, residual) = derive(ELECTRON_EXACT);
        assert_eq!(mass, ELECTRON_MASS, "electron mass");
        assert_eq!(residual, ELECTRON_RESIDUAL_NDA, "electron residual");
    }
}
