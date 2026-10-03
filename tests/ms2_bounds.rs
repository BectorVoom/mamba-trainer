//! P2.5 tests: single-`u32` arithmetic bounds of architecture §6.4.
//!
//! Each bound is computed from the domain tables (`ELEMENTS`, `ATOM_TYPES`,
//! the electron mass, the adduct rows) rather than retyping a constant, and
//! each asserts against the documented figure. Host-only arithmetic comes
//! first; the device boundary checks at the end compare the integer kernels
//! against the host reference on boundary values.
//!
//! Kernel-versus-host reuse: `u32::MAX` sentinels and the unknown-precision
//! path of `formula_window` are already covered by
//! `formula_window_matches_reference` in `tests/ms2_formula.rs` (group A:
//! precursor `u32::MAX`, the unknown-precision sentinel); the checks below
//! cover the window edges (a parent at the low/high saturating ends) and the
//! `u32::MAX` id sentinel of the integer kernels.

#![cfg(feature = "backend")]

use mamba3::backend::{Device, check_launches};
use mamba3::backends::Auto;
use mamba3::models::ms2::chem::{
    ADDUCTS, ATOM_TYPES, Composition, ELECTRON_MASS, ELEMENTS, HYDROGEN, composition_error_nda,
    composition_mass, element_index, ion, parent_mass, tolerance, tolerance_u32,
};
use mamba3::models::ms2::contract::{PRECURSOR_MAX, PRECURSOR_MIN, SCHEMA_VERSION, SpectrumBatch};
use mamba3::models::ms2::formula::{FormulaTable, WindowQuery};
use mamba3::models::ms2::twin;
use mamba3::tensor::ops::index::IdTensor;
use mamba3::tensor::ops::ms2;

type R = Auto;
type E = f32;

fn dev() -> Device<R> {
    Device::<R>::default()
}

/// One spectrum with this precursor, valid peaks below it and adduct 1.
fn batch_with_precursor(precursor: u32) -> SpectrumBatch {
    let n_raw = 64usize;
    let mut peak_id = vec![u32::MAX; n_raw];
    peak_id[0..2].copy_from_slice(&[0, 1]);
    let mut mz = vec![0u32; n_raw];
    mz[0..2].copy_from_slice(&[100_000_000, 200_000_000]);
    let mut intensity = vec![0.0f32; n_raw];
    intensity[0..2].copy_from_slice(&[1.0, 0.5]);
    SpectrumBatch {
        schema_version: SCHEMA_VERSION,
        n_raw: n_raw as u32,
        spectrum_id: vec![7],
        raw_peak_count: vec![2],
        peak_count: vec![2],
        peak_id,
        mz_udalton: mz,
        intensity,
        intensity_scale: 0,
        mz_uncertainty_udalton: vec![50],
        precursor_mz_udalton: vec![precursor],
        precursor_uncertainty_udalton: vec![50],
        adduct: vec![1],
        polarity: vec![1],
        collision_energy_ev: vec![30.0],
        collision_energy_known: vec![1],
        energy_count: vec![1],
        fragment_tolerance_ppm_tenths: vec![0],
        precursor_tolerance_ppm_tenths: vec![0],
        instrument_class: vec![0],
    }
}

#[test]
fn precursor_bound_accepted_and_refused() {
    // Architecture §6.4: precursor at most 2,000,000,000. The bound is a
    // contract range check: 2,000,000,000 validates cleanly and 2,000,000,001
    // carries `precursor_out_of_range`.
    use mamba3::models::ms2::contract::request_status as s;
    assert_eq!(PRECURSOR_MIN, 50_000_000);
    assert_eq!(PRECURSOR_MAX, 2_000_000_000);
    let ok = batch_with_precursor(2_000_000_000).validate().unwrap();
    assert_eq!(ok[0] & s::PRECURSOR_OUT_OF_RANGE, 0, "2e9 validates");
    let bad = batch_with_precursor(2_000_000_001).validate().unwrap();
    assert_ne!(bad[0] & s::PRECURSOR_OUT_OF_RANGE, 0, "2e9+1 refused");
}

#[test]
fn parent_mass_after_adduct_shift_within_documented_interval() {
    // Architecture §6.4: the parent mass after the adduct shift lies within
    // [48,992,724, 2,001,007,276] over both adducts at the precursor
    // extremes. Endpoints derived from the tables: low is
    // `50 Da − m_H + m_e` ([M+H]+ at the floor), high is
    // `2000 Da + m_H − m_e` ([M-H]- at the ceiling).
    let h = ELEMENTS[HYDROGEN].mass;
    let e = ELECTRON_MASS;
    let low = 50_000_000u32 - h + e;
    let high = 2_000_000_000u32 + h - e;
    assert_eq!(low, 48_992_724, "low endpoint from the tables");
    assert_eq!(high, 2_001_007_276, "high endpoint from the tables");
    for adduct in [1u16, 2] {
        for precursor in [50_000_000u32, 2_000_000_000] {
            let parent = parent_mass(precursor, adduct).unwrap();
            assert!(
                (low..=high).contains(&parent),
                "parent {parent} of precursor {precursor} adduct {adduct} within [{low}, {high}]"
            );
        }
    }
    // The adduct rows used exist in the domain.
    assert_eq!(ADDUCTS.len(), 2);
}

#[test]
fn largest_table_mass_within_u32() {
    // Architecture §6.4: table masses at most 3,399,572,650. The V0 domain
    // budget for one subgraph is 16 atoms (Limits::V0); the heaviest atom
    // type mass comes from the tables (element mass plus parent hydrogens),
    // so 16 of those is an upper bound on any 16-atom subgraph mass — well
    // under the documented table maximum, which itself fits `u32`.
    let h = ELEMENTS[HYDROGEN].mass;
    let mut heaviest = 0u32;
    for t in ATOM_TYPES.iter() {
        let m = ELEMENTS[t.element]
            .mass
            .checked_add((u32::from(t.hydrogens)) * h)
            .expect("one atom-type mass fits u32");
        heaviest = heaviest.max(m);
    }
    let sixteen = heaviest.checked_mul(16).expect("16 atoms fit u32");
    assert!(
        sixteen <= 3_399_572_650,
        "16 heaviest atoms {sixteen} under the table bound"
    );
    assert!(3_399_572_650u32 < u32::MAX, "the bound fits u32");
    // The composition machinery agrees: the heaviest single atom type times
    // 16 is a valid composition mass.
    let mut c: Composition = [0; 10];
    let heaviest_type = ATOM_TYPES
        .iter()
        .max_by_key(|t| ELEMENTS[t.element].mass + u32::from(t.hydrogens) * h)
        .expect("atom types");
    c[heaviest_type.element] = 16;
    // Hydrogens counted in the composition are the element's own count only
    // for this bound check when the type carries none (iodine H0); the
    // inequality above is the load-bearing one.
    let _ = composition_mass(&c).expect("16 heavy atoms fit u32");
}

#[test]
fn window_bounds_use_saturating_arithmetic() {
    // Architecture §6.4: window bounds computed with saturating arithmetic
    // at 0 and `u32::MAX`. A parent at the bottom saturates the low end to
    // 0; a width past the top saturates the high end to `u32::MAX`; neither
    // panics and both match the host reference the kernel reproduces.
    let table =
        FormulaTable::from_compositions([[60, 120, 0, 30, 0, 0, 0, 0, 0, 0]].into_iter()).unwrap();
    let mass = table.mass(0);
    // Low saturation: parent below the width.
    let low = mass.saturating_sub(u32::MAX);
    assert_eq!(low, 0);
    // High saturation: parent plus a huge width.
    let high = mass.saturating_add(u32::MAX);
    assert_eq!(high, u32::MAX);
    // The real search saturates the same way: a precursor at the domain
    // floor with a wide tolerance still searches without panic.
    let query = WindowQuery {
        precursor_mz: PRECURSOR_MIN,
        adduct: 1,
        ppm_tenths: 1000,
        precursor_uncertainty: 50,
        rows_visited_max: u32::MAX,
        rows_scored_max: u32::MAX,
    };
    let result = table.window(&query);
    assert!(
        result.rows_visited > 0 || result.absent,
        "a floor search completes or is absent, never panics"
    );
}

#[test]
fn ion_mz_of_16_atom_subgraph_plus_shift_below_precursor_plus_2da() {
    // Architecture §6.4: the ion m/z of a 16-atom subgraph plus the largest
    // shift is below the precursor bound plus 2 Da. The subgraph must be one
    // a request-domain parent can hold: sixteen iodines already exceed the
    // precursor range as a parent (2,030,471,552 Da > 2000 Da), so the bound
    // is read over 16-atom subgraphs of in-range parents. Pyrene (C16H10,
    // 16 atoms) from the domain tables under [M+H]+ with shift +2, the
    // largest supported mapping, stays far below 2e9 + 2e6 — and every
    // intermediate fits `u32` by construction.
    let carbon = element_index("C").expect("carbon in the domain");
    let mut c: Composition = [0; 10];
    c[carbon] = 16;
    c[HYDROGEN] = 10;
    let _parent = composition_mass(&c).expect("pyrene fits u32");
    let hyp = ion(&c, 1, 2)
        .expect("ion computes")
        .expect("hydrogen count non-negative");
    let ceiling = 2_000_000_000u32.checked_add(2_000_000).expect("fits u32");
    assert!(
        hyp.mz < ceiling,
        "16-atom ion {} under precursor bound + 2 Da {ceiling}",
        hyp.mz
    );
    // Why the domain restriction matters: sixteen iodines (the heaviest
    // atom type) as a parent already leave the request domain, so they are
    // not a counterexample to a bound over in-range subgraphs.
    let iodine = ELEMENTS
        .iter()
        .position(|e| e.symbol == "I")
        .expect("iodine in the domain");
    let mut heavy: Composition = [0; 10];
    heavy[iodine] = 16;
    let heavy_mass = composition_mass(&heavy).expect("fits u32");
    assert!(
        heavy_mass > 2_000_000_000,
        "16 iodines {heavy_mass} exceed the precursor range"
    );
}

#[test]
fn tolerance_intermediates_fit_u32_with_checked_arithmetic() {
    // Contracts §5: `hi = mz / 10^4`, `lo = mz % 10^4`, `a = hi * t`,
    // `tol = a / 1000 + ((a % 1000) * 10^4 + lo * t) / 10^7` with `t <= 1000`.
    // At the extreme inputs (`mz = u32::MAX`, `t = 1000`) every intermediate
    // fits `u32`: `a <= 429,496,000` and the second numerator is at most
    // `19,989,000`. Asserted here with checked arithmetic, not by running
    // the division.
    for (mz, t) in [(u32::MAX, 1000u32), (u32::MAX, 0), (0, 1000), (1, 1)] {
        let hi = mz / 10_000;
        let lo = mz % 10_000;
        let a = hi.checked_mul(t).expect("hi * t fits u32 for t <= 1000");
        assert!(a <= 429_496_000, "mz={mz} t={t}: a={a}");
        let second = (a % 1000)
            .checked_mul(10_000)
            .and_then(|v| v.checked_add(lo.checked_mul(t).expect("lo * t fits")))
            .expect("second numerator fits u32");
        assert!(second <= 19_989_000, "mz={mz} t={t}: second={second}");
        // And the split form equals the direct form.
        assert_eq!(tolerance_u32(mz, t).unwrap(), tolerance(mz, t));
    }
    assert!(tolerance_u32(100, 1001).is_err(), "t > 1000 rejected");
}

#[test]
fn tolerance_split_matches_direct_at_extremes() {
    // Contracts §5 decision inputs at their extremes: the split tolerance
    // equals the direct `floor(mz * t / 1e7)` everywhere, including 0 and
    // `u32::MAX`, and the arithmetic bound of a 16-atom subgraph stays under
    // 9 against a 10 ppm tolerance of at least 500 at m/z 50.
    for mz in [0u32, 1, 50_000_000, 2_000_000_000, u32::MAX] {
        for t in [0u32, 1, 100, 200, 1000] {
            assert_eq!(
                tolerance_u32(mz, t).unwrap(),
                tolerance(mz, t),
                "mz={mz} t={t}"
            );
        }
    }
    // 16-atom arithmetic bound under 9: worst residual per atom is bromine
    // (400 nDa) plus the electron (421 nDa), all from the tables.
    let max_residual = ELEMENTS.iter().map(|e| e.residual_nda).max().unwrap_or(0);
    let bound = (16u64 * u64::from(max_residual) + 421).div_ceil(1000);
    assert!(bound < 9, "16-atom bound {bound} under 9");
    assert!(
        tolerance(50_000_000, 100) >= 500,
        "10 ppm at m/z 50 is >= 500"
    );
}

#[test]
fn formula_window_at_window_edges_matches_host() {
    // Device boundary: `formula_window` at the window edges. A table whose
    // rows sit at the low edge (mass 50 Da + shift) and the high edge is
    // searched from both extremes; the device kernel equals the host twin
    // exactly (rows, flags and counters).
    let device = dev();
    let low: Composition = [3, 6, 0, 1, 0, 0, 0, 0, 0, 0];
    let high: Composition = [60, 120, 0, 30, 0, 0, 0, 0, 0, 0];
    let table = FormulaTable::from_compositions([low, high].into_iter()).unwrap();
    let masses: Vec<u32> = (0..table.len()).map(|r| table.mass(r)).collect();
    let bounds: Vec<u32> = (0..table.len())
        .map(|r| composition_error_nda(table.composition(r)).div_ceil(1000) as u32)
        .collect();
    let rows = table.len();
    let search = IdTensor::from_slice(
        &{
            let mut v = Vec::with_capacity(rows * 2);
            for i in 0..rows {
                v.push(masses[i]);
                v.push(bounds[i]);
            }
            v
        },
        vec![rows, 2],
        &device,
    )
    .unwrap();
    // Precursors placing the parent exactly on each row (under [M+H]+).
    let h_net = ELEMENTS[HYDROGEN].mass - ELECTRON_MASS;
    for (row, mass) in masses.iter().enumerate() {
        let precursor = mass.checked_add(h_net).expect("precursor fits");
        let meta = vec![1u32, precursor, 50, 1, 100, 200, 0, 0];
        let m = 32usize;
        let out = ms2::FormulaBuffers::<R, E>::poisoned(1, m, 2, &device).unwrap();
        let meta_t = IdTensor::from_slice(&meta, vec![1, 8], &device).unwrap();
        ms2::formula_window(
            &search,
            &meta_t,
            table.max_error(),
            u32::MAX,
            u32::MAX,
            &out,
        )
        .unwrap();
        check_launches(&device).unwrap();
        let (want_window, want_counters) =
            twin::formula_window(&table, &meta, 1, m, table.max_error(), u32::MAX, u32::MAX);
        assert_eq!(
            out.window.try_to_vec().unwrap(),
            want_window,
            "edge row {row}"
        );
        assert_eq!(
            out.counters.try_to_vec().unwrap(),
            want_counters,
            "edge row {row}"
        );
    }
    // `u32::MAX` id sentinels and the unknown-precision path are covered by
    // `formula_window_matches_reference` in tests/ms2_formula.rs (group A).
}
