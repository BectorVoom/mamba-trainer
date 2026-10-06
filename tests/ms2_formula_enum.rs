//! H1.2 tests: bounded closed-form-hydrogen enumeration (host-only, no device).
//!
//! Every fixture-derived expected value comes from
//! `tests/fixtures/ms2/chemistry_v0.json`; hand-built cases state their
//! arithmetic inline. No tensors, no kernels, no device code.

use std::path::PathBuf;

use serde_json::Value;

use mamba3::error::Error;
use mamba3::models::ms2::contract::request_status;
use mamba3::models::ms2::formula::{FormulaTable, WindowQuery};
use mamba3::models::ms2::formula_enum::{
    DEVICE_HALF_MAX, DBE_ENDPOINT_MAX, DBE_ENDPOINT_MIN, DeviceEnumLimits, ENUM_LANES_MAX_DEFAULT,
    EnumDomain, EnumLimits, EnumQuery, LANE_EXHAUSTED_BIT,
    LANE_MODE_COUNT, LANE_MODE_FILL, LANE_RECORD_WORDS, LANE_VISITED_MASK, PACK_CARBON_WIDTH,
    PACK_DBE_BIAS, PACK_HEADER_LEN, PACK_HEAVY_CAPS, PACK_HEAVY_MAX, PACK_HEAVY_WIDTH,
    PACK_HYDROGEN_MAX, PACK_HYDROGEN_MIN, PACK_N_CARBON_ROWS, PACK_N_HEAVY_ROWS,
    PACK_RARE_TOTAL_HI, PACK_RARE_TOTAL_LO, PACK_RATIO_HI_DEN, PACK_RATIO_HI_NUM,
    PACK_RATIO_LO_DEN, PACK_RATIO_LO_NUM, PACK_ZERO_CARBON, RARE_TABLE_MAX, RATIO_FEATURES,
    RatioBounds, dbe_twice, decide_u32, enumerate, enumerate_device_order, enumerate_neutral,
    gold_stages, hydrogen_ceiling, kernel_lane, kernel_offsets, kernel_pad, max_valence,
    pack_device_bounds, rare_table, sat_add_counter, sat_add_saturates, validate_device_artifacts,
    validate_enum_dispatch,
};
use mamba3::models::ms2::chem::ELEMENTS;
use mamba3::models::ms2::{Composition, composition_mass, decide, element_index};

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/ms2/chemistry_v0.json")
}

fn fixture() -> Value {
    let text = std::fs::read_to_string(fixture_path()).expect("fixture readable");
    serde_json::from_str(&text).expect("fixture parses")
}

fn formula_of(m: &Value) -> Composition {
    let mut c: Composition = [0; 10];
    for (symbol, count) in m["formula"].as_object().expect("formula") {
        c[element_index(symbol).expect("known element")] = count.as_u64().unwrap() as u16;
    }
    c
}

fn fixture_compositions() -> Vec<Composition> {
    fixture()["molecules"]
        .as_array()
        .expect("molecules")
        .iter()
        .map(formula_of)
        .collect()
}

/// Protonated (`[M+H]+`) precursor m/z of a neutral composition.
///
/// Written with the test literals below (contract §4.3: `mz = M + m_H − m_e`
/// with `m_H = 1_007_825`, `m_e = 549`): the precursor is only a search input,
/// while the parent mass and tolerance the search is checked against come
/// from [`brute_parent`] and [`brute_tolerance`].
fn protonated(c: &Composition) -> u32 {
    brute_mass(c) + BRUTE_MASS[1] - 549
}

/// Deprotonated (`[M-H]-`) precursor m/z of a neutral composition.
///
/// Contract §4.3 with test literals: `mz = M − m_H + m_e`.
fn deprotonated(c: &Composition) -> u32 {
    brute_mass(c) - BRUTE_MASS[1] + 549
}

// ---------------------------------------------------------------------------
// Independent brute-force references (own literals, own sums).
// ---------------------------------------------------------------------------

/// Own copy of the §4.1 integer masses (test literals, not the crate table).
const BRUTE_MASS: [u32; 10] = [
    12_000_000,
    1_007_825,
    14_003_074,
    15_994_915,
    18_998_403,
    30_973_762,
    31_972_071,
    34_968_853,
    78_918_338,
    126_904_472,
];

/// Own copy of the §4.1 rounding residuals in nano-dalton.
const BRUTE_RES: [u32; 10] = [0, 33, 5, 381, 163, 2, 175, 318, 400, 100];

/// Own copy of the per-element maximum valences (C, H, N, O, F, P, S, Cl, Br, I).
const BRUTE_VMAX: [u8; 10] = [4, 1, 3, 2, 1, 5, 6, 1, 1, 1];

/// Heavy element ids in the crate's element order.
const BRUTE_HEAVY: [usize; 9] = [0, 2, 3, 4, 5, 6, 7, 8, 9];

/// Independent parent mass from contract §4.3, written with the test literals
/// above (`m_H = 1_007_825`, `m_e = 549`), not production helpers:
/// adduct 1 `[M+H]+` gives `M = mz − m_H + m_e`, adduct 2 `[M-H]-` gives
/// `M = mz + m_H − m_e`.
fn brute_parent(precursor_mz: u32, adduct: u16) -> u32 {
    match adduct {
        1 => precursor_mz - 1_007_825 + 549,
        2 => precursor_mz + 1_007_825 - 549,
        _ => panic!("brute_parent: unsupported adduct {adduct}"),
    }
}

/// Independent tolerance from contract §5, written inline:
/// `tol(mz, t) = floor(mz * t / 10^7)` for `t` in tenths of a ppm.
fn brute_tolerance(mz: u32, ppm_tenths: u32) -> u32 {
    ((u64::from(mz) * u64::from(ppm_tenths)) / 10_000_000) as u32
}

fn brute_mass(c: &Composition) -> u32 {
    let mut total: u64 = 0;
    for (e, n) in c.iter().enumerate() {
        total += u64::from(*n) * u64::from(BRUTE_MASS[e]);
    }
    assert!(total <= u64::from(u32::MAX), "brute mass fits u32");
    total as u32
}

fn brute_error(c: &Composition) -> u32 {
    let mut nda: u64 = 0;
    for (e, n) in c.iter().enumerate() {
        nda += u64::from(*n) * u64::from(BRUTE_RES[e]);
    }
    nda.div_ceil(1000) as u32
}

/// Independent §5 verdict: `(joined, ambiguous)`.
fn brute_verdict(parent: u32, mass: u32, error: u32, tol: u32) -> (bool, bool) {
    let r = parent.abs_diff(mass) as u64;
    let (e, t) = (u64::from(error), u64::from(tol));
    if r + e <= t {
        (true, false)
    } else if r > t + e {
        (false, false)
    } else {
        (true, true)
    }
}

fn brute_h_ceil(c: &Composition) -> u16 {
    let mut n: u32 = 0;
    let mut stubs: u32 = 0;
    for e in BRUTE_HEAVY {
        n += u32::from(c[e]);
        stubs += u32::from(c[e]) * u32::from(BRUTE_VMAX[e]);
    }
    if n == 0 {
        return 0;
    }
    (stubs + 2).saturating_sub(2 * n).min(u32::from(u16::MAX)) as u16
}

fn brute_parity(c: &Composition) -> bool {
    let mut stubs: u32 = 0;
    for e in BRUTE_HEAVY {
        stubs += u32::from(c[e]) * u32::from(BRUTE_VMAX[e]);
    }
    (stubs + u32::from(c[1])).is_multiple_of(2)
}

fn brute_dbe_ok(c: &Composition) -> bool {
    let mut twice: i64 = 2;
    for e in BRUTE_HEAVY {
        twice += (i64::from(BRUTE_VMAX[e]) - 2) * i64::from(c[e]);
    }
    twice - i64::from(c[1]) >= 0
}

/// The tiny brute-force domain: C 0–2, N 0–1, O 0–1, H 0–6, heavy total ≤ 4.
fn tiny_domain() -> EnumDomain {
    EnumDomain {
        version: "test-tiny".to_string(),
        heavy_caps: [2, 1, 1, 0, 0, 0, 0, 0, 0],
        heavy_max: 4,
        hydrogen_min: 0,
        hydrogen_max: 6,
    }
}

/// Every composition of the tiny domain with at least one heavy atom.
fn tiny_compositions() -> Vec<Composition> {
    let mut out = Vec::new();
    for c in 0..=2u16 {
        for n in 0..=1u16 {
            for o in 0..=1u16 {
                if c + n + o == 0 || c + n + o > 4 {
                    continue;
                }
                for h in 0..=6u16 {
                    let mut comp: Composition = [0; 10];
                    comp[0] = c;
                    comp[1] = h;
                    comp[2] = n;
                    comp[3] = o;
                    out.push(comp);
                }
            }
        }
    }
    out
}

/// Brute-force expected joins over the tiny domain for one query.
fn brute_expected(
    parent: u32,
    tol: u32,
    bound: u32,
    filtered: bool,
) -> Vec<(u32, Composition, bool)> {
    let mut out = Vec::new();
    for c in tiny_compositions() {
        let mass = brute_mass(&c);
        let error = brute_error(&c).saturating_add(bound);
        let (joined, ambiguous) = brute_verdict(parent, mass, error, tol);
        if !joined {
            continue;
        }
        if filtered && (c[1] > brute_h_ceil(&c) || !brute_parity(&c) || !brute_dbe_ok(&c)) {
            continue;
        }
        out.push((mass, c, ambiguous));
    }
    out.sort_by_key(|a| (a.0, a.1));
    out
}

fn enum_rows(
    domain: &EnumDomain,
    query: &EnumQuery,
    limits: &EnumLimits,
) -> Vec<(u32, Composition, bool)> {
    let found = enumerate(domain, query, limits).unwrap();
    let mut rows: Vec<(u32, Composition, bool)> = found
        .masses
        .iter()
        .zip(found.compositions.iter())
        .zip(found.ambiguous.iter())
        .map(|((m, c), a)| (*m, *c, *a))
        .collect();
    rows.sort_by_key(|a| (a.0, a.1));
    rows
}

#[test]
fn max_valence_matches_atom_type_table() {
    assert_eq!(
        [
            max_valence(0),
            max_valence(2),
            max_valence(3),
            max_valence(4),
            max_valence(5),
            max_valence(6),
            max_valence(7),
            max_valence(8),
            max_valence(9),
        ],
        [4, 3, 2, 1, 5, 6, 1, 1, 1]
    );
    assert_eq!(max_valence(1), 1);
}

#[test]
fn no_fixture_molecule_is_rejected_by_any_filter() {
    // H1.1 step 3: every in-domain fixture composition passes all three
    // filters, so the rules are exact for the V0 chemistry domain.
    let comps = fixture_compositions();
    assert_eq!(comps.len(), 28);
    for (i, c) in comps.iter().enumerate() {
        assert!(
            hydrogen_ceiling(c).unwrap() >= c[1],
            "molecule {i}: H{} above ceiling {}",
            c[1],
            hydrogen_ceiling(c).unwrap()
        );
        assert_eq!(dbe_twice(c).unwrap() % 2, 0, "molecule {i}: parity");
        assert!(
            dbe_twice(c).unwrap() >= 0,
            "molecule {i}: negative DBE {}",
            dbe_twice(c).unwrap()
        );
    }
}

#[test]
fn domain_from_compositions_margin_json_and_contains() {
    let comps = fixture_compositions();
    let domain = EnumDomain::from_compositions(comps.iter().copied(), 2).unwrap();
    assert_eq!(domain.version, "ms2-enum-v1");
    // Caps are observed maxima plus the margin.
    let mut heavy_max: [u16; 9] = [0; 9];
    let mut total_max: u16 = 0;
    let mut h_max: u16 = 0;
    for c in &comps {
        let mut total: u16 = 0;
        for (i, e) in [0usize, 2, 3, 4, 5, 6, 7, 8, 9].iter().enumerate() {
            heavy_max[i] = heavy_max[i].max(c[*e]);
            total += c[*e];
        }
        total_max = total_max.max(total);
        h_max = h_max.max(c[1]);
    }
    for (i, cap) in heavy_max.iter().enumerate() {
        assert_eq!(domain.heavy_caps[i], cap + 2, "heavy cap {i}");
    }
    assert_eq!(domain.heavy_max, total_max + 2);
    assert_eq!((domain.hydrogen_min, domain.hydrogen_max), (0, h_max + 2));
    assert!(domain.bytes() > 0);
    for c in &comps {
        assert!(domain.contains(c), "fixture composition is in-domain");
    }
    // JSON round trip.
    let back = EnumDomain::from_json(&domain.to_json()).unwrap();
    assert_eq!(back, domain);
    assert!(EnumDomain::from_json("{\"version\":\"x\",\"heavy_caps\":[0,0,0,0,0,0,0,0,0],\"heavy_max\":0,\"hydrogen_min\":5,\"hydrogen_max\":4}").is_err());
    // Margin overflow is an error, never wrapping.
    assert!(EnumDomain::from_compositions(comps.iter().copied(), u16::MAX).is_err());
}

#[test]
fn tiny_domain_matches_brute_force() {
    let domain = tiny_domain();
    let bases: Vec<Composition> = [
        [1, 4, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 6, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 2, 1, 1, 0, 0, 0, 0, 0, 0],
        [2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 6, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    ]
    .to_vec();
    let deltas: [i64; 9] = [-20_000, -1000, -100, -1, 0, 1, 100, 1000, 20_000];
    for adduct in [1u16, 2] {
        for ppm in [1u32, 100, 200, 1000] {
            for base in &bases {
                let anchor = if adduct == 1 {
                    protonated(base)
                } else {
                    deprotonated(base)
                };
                for delta in deltas {
                    let precursor = (anchor as i64 + delta) as u32;
                    for filtered in [false, true] {
                        let query = EnumQuery {
                            precursor_mz: precursor,
                            adduct,
                            ppm_tenths: ppm,
                            precursor_uncertainty: 50,
                        };
                        let limits = if filtered {
                            EnumLimits::default()
                        } else {
                            EnumLimits::unfiltered()
                        };
                        let found = enumerate(&domain, &query, &limits).unwrap();
                        assert!(!found.exhausted, "tiny domain never exhausts");
                        assert!(
                            found.support_complete,
                            "tiny domain always completes"
                        );
                        // Canonical order: mass, then element counts.
                        let mut sorted: Vec<(u32, Composition)> = found
                            .masses
                            .iter()
                            .zip(found.compositions.iter())
                            .map(|(m, c)| (*m, *c))
                            .collect();
                        let mut by_sort = sorted.clone();
                        by_sort.sort();
                        assert_eq!(sorted, by_sort, "canonical output order");
                        let _ = &mut sorted;
                        let parent = brute_parent(precursor, adduct);
                        let tol = brute_tolerance(precursor, ppm);
                        let expected = brute_expected(parent, tol, 51, filtered);
                        assert_eq!(
                            enum_rows(&domain, &query, &limits),
                            expected,
                            "adduct {adduct} ppm {ppm} delta {delta} filtered {filtered}"
                        );
                        // Counters reconcile: every hydrogen check ends in
                        // exactly one bucket.
                        let counted = found.rejected_mass
                            + found.rejected_h_max
                            + found.rejected_parity
                            + found.rejected_dbe
                            + found.rows_joined;
                        assert_eq!(
                            found.hydrogen_checks, counted,
                            "hydrogen checks partition"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn window_edge_precursors_match_brute_force() {
    // Masses at the tolerance edge: a parent exactly `tol + 1` past a row mass
    // is ambiguous, and both sides of the edge agree with brute force.
    let domain = tiny_domain();
    let base: Composition = [1, 4, 0, 0, 0, 0, 0, 0, 0, 0];
    let mass = brute_mass(&base);
    for adduct in [1u16, 2] {
        for ppm in [1u32, 200] {
            // Fixed-point the tolerance at the final precursor (two rounds).
            let mut parent = mass + 5;
            for _ in 0..3 {
                let precursor = if adduct == 1 {
                    parent + BRUTE_MASS[1] - 549
                } else {
                    parent - BRUTE_MASS[1] + 549
                };
                let tol = brute_tolerance(precursor, ppm);
                parent = mass + tol + 1;
            }
            let precursor = if adduct == 1 {
                parent + BRUTE_MASS[1] - 549
            } else {
                parent - BRUTE_MASS[1] + 549
            };
            let tol = brute_tolerance(precursor, ppm);
            assert_eq!(parent - mass, tol + 1);
            let query = EnumQuery {
                precursor_mz: precursor,
                adduct,
                ppm_tenths: ppm,
                precursor_uncertainty: 0,
            };
            let expected = brute_expected(parent, tol, 1, false);
            let edge = expected.iter().find(|(_, c, _)| *c == base).expect("edge row");
            assert!(edge.2, "the edge row is ambiguous");
            assert_eq!(enum_rows(&domain, &query, &EnumLimits::unfiltered()), expected);
        }
    }
}

#[test]
fn every_integer_hydrogen_count_in_a_wide_window_is_found() {
    // A window spanning two hydrogen counts for one heavy vector (large
    // uncertainty at low mass): both counts must be returned.
    let domain = EnumDomain {
        version: "test-h-span".to_string(),
        heavy_caps: [1, 0, 0, 0, 0, 0, 0, 0, 0],
        heavy_max: 1,
        hydrogen_min: 0,
        hydrogen_max: 12,
    };
    let h_unit = BRUTE_MASS[1];
    // Centre the parent between C1H5 and C1H6.
    let parent = BRUTE_MASS[0] + 5 * h_unit + h_unit / 2;
    let precursor = parent + BRUTE_MASS[1] - 549;
    let query = EnumQuery {
        precursor_mz: precursor,
        adduct: 1,
        ppm_tenths: 1000,
        precursor_uncertainty: 600_000,
    };
    let found = enumerate(&domain, &query, &EnumLimits::unfiltered()).unwrap();
    assert!(!found.exhausted);
    let expected = {
        let tol = brute_tolerance(precursor, 1000);
        let mut out = Vec::new();
        for h in 0..=12u16 {
            let mut c: Composition = [0; 10];
            c[0] = 1;
            c[1] = h;
            let (joined, ambiguous) =
                brute_verdict(parent, brute_mass(&c), brute_error(&c) + 600_001, tol);
            if joined {
                out.push((brute_mass(&c), c, ambiguous));
            }
        }
        out.sort_by_key(|a| (a.0, a.1));
        out
    };
    assert!(expected.len() >= 2, "the window spans two hydrogen counts");
    assert_eq!(enum_rows(&domain, &query, &EnumLimits::unfiltered()), expected);
    // With filters the same predicate-filtered set is returned.
    let limits = EnumLimits::default();
    let found = enumerate(&domain, &query, &limits).unwrap();
    let filtered: Vec<(u32, Composition, bool)> = expected
        .into_iter()
        .filter(|(_, c, _)| {
            c[1] <= brute_h_ceil(c) && brute_parity(c) && brute_dbe_ok(c)
        })
        .collect();
    assert_eq!(enum_rows(&domain, &query, &limits), filtered);
    let _ = found;
}

#[test]
fn superset_property_against_the_table() {
    // For a table built from the fixture compositions, every row
    // `FormulaTable::window` joins is also joined by `enumerate` over a domain
    // containing the fixture, with the same verdict flag.
    let comps = fixture_compositions();
    let table = FormulaTable::from_compositions(comps.iter().copied()).unwrap();
    let domain = EnumDomain::from_compositions(comps.iter().copied(), 0).unwrap();
    let limits = EnumLimits::unfiltered();
    for (i, gold) in comps.iter().enumerate() {
        for adduct in [1u16, 2] {
            let precursor = if adduct == 1 {
                protonated(gold)
            } else {
                deprotonated(gold)
            };
            let window = WindowQuery {
                precursor_mz: precursor,
                adduct,
                ppm_tenths: 200,
                precursor_uncertainty: 50,
                rows_visited_max: u32::MAX,
                rows_scored_max: u32::MAX,
            };
            let table_found = table.window(&window);
            assert!(
                !table_found.exhausted,
                "molecule {i} adduct {adduct}: table completes"
            );
            let query = EnumQuery::from(&window);
            let enum_found = enumerate(&domain, &query, &limits).unwrap();
            assert!(
                !enum_found.exhausted,
                "molecule {i} adduct {adduct}: enumeration completes"
            );
            for (row, ambiguous) in table_found
                .joined
                .iter()
                .zip(table_found.ambiguous.iter())
            {
                let comp = table.composition(*row);
                let pos = enum_found
                    .compositions
                    .iter()
                    .position(|c| c == comp)
                    .unwrap_or_else(|| panic!("molecule {i} adduct {adduct}: table row {row} missing"));
                assert_eq!(
                    enum_found.ambiguous[pos], *ambiguous,
                    "molecule {i} adduct {adduct}: same verdict flag"
                );
            }
            // The gold composition itself joins as accepted in both.
            let gold_row = table_found
                .joined
                .iter()
                .position(|r| table.composition(*r) == gold)
                .unwrap_or_else(|| panic!("molecule {i} adduct {adduct}: gold missing in table"));
            assert!(!table_found.ambiguous[gold_row]);
            let gold_pos = enum_found
                .scored_contains(gold)
                .then(|| {
                    enum_found
                        .compositions
                        .iter()
                        .position(|c| c == gold)
                        .unwrap()
                })
                .expect("gold joins the enumeration");
            assert!(!enum_found.ambiguous[gold_pos]);
        }
    }
}

#[test]
fn capacity_and_visit_limits_are_deterministic() {
    let domain = EnumDomain {
        version: "test-cap".to_string(),
        heavy_caps: [2, 0, 0, 0, 0, 0, 0, 0, 0],
        heavy_max: 3,
        hydrogen_min: 0,
        hydrogen_max: 10,
    };
    let query = EnumQuery {
        precursor_mz: 20_000_000 + BRUTE_MASS[1] - 549,
        adduct: 1,
        ppm_tenths: 1000,
        precursor_uncertainty: 5_000_000,
    };
    let open = EnumLimits::unfiltered();
    let full = enumerate(&domain, &query, &open).unwrap();
    assert!(!full.exhausted, "open search completes");
    assert!(full.rows_joined >= 3, "joins several rows");
    // A join capacity truncates deterministically and reports exhausted.
    let capped = EnumLimits {
        capacity: 2,
        ..EnumLimits::unfiltered()
    };
    let first = enumerate(&domain, &query, &capped).unwrap();
    let second = enumerate(&domain, &query, &capped).unwrap();
    assert_eq!(first, second, "capacity overflow is deterministic");
    assert!(first.exhausted);
    assert!(!first.absent);
    assert_eq!(first.rows_joined, full.rows_joined);
    assert_eq!(first.rows_scored, 2);
    assert_eq!(
        first.status & request_status::FORMULA_SEARCH_EXHAUSTED,
        request_status::FORMULA_SEARCH_EXHAUSTED
    );
    // A scored cap truncates the same way.
    let scored = EnumLimits {
        scored_max: 2,
        ..EnumLimits::unfiltered()
    };
    let found = enumerate(&domain, &query, &scored).unwrap();
    assert!(found.exhausted);
    assert_eq!(found.rows_scored, 2);
    // A visit limit stops part-way, deterministically, distinct from absent.
    let limited = EnumLimits {
        nodes_visited_max: 4,
        ..EnumLimits::unfiltered()
    };
    let a = enumerate(&domain, &query, &limited).unwrap();
    let b = enumerate(&domain, &query, &limited).unwrap();
    assert_eq!(a, b, "visit-limit stop is deterministic");
    assert!(a.exhausted);
    assert!(!a.absent);
    assert_eq!(a.nodes_visited, 4);
    // A far precursor is absent, not exhausted.
    let far = EnumQuery {
        precursor_mz: 2_000_000_000,
        ..query
    };
    let absent = enumerate(&domain, &far, &open).unwrap();
    assert!(absent.absent);
    assert!(!absent.exhausted);
    assert!(absent.support_complete);
    assert_eq!(absent.status, request_status::FORMULA_ABSENT);
}

#[test]
fn boundary_cases() {
    // Ambiguous verdicts join flagged.
    let single: Composition = [2, 4, 0, 1, 0, 0, 0, 0, 0, 0];
    let single_mass = composition_mass(&single).unwrap();
    let domain = EnumDomain::from_compositions([single], 0).unwrap();
    let mut parent = single_mass + 5;
    for _ in 0..3 {
        let precursor = parent + BRUTE_MASS[1] - 549;
        parent = single_mass + brute_tolerance(precursor, 1) + 1;
    }
    let precursor = parent + BRUTE_MASS[1] - 549;
    assert_eq!(parent - single_mass, brute_tolerance(precursor, 1) + 1);
    let query = EnumQuery {
        precursor_mz: precursor,
        adduct: 1,
        ppm_tenths: 1,
        precursor_uncertainty: 0,
    };
    let found = enumerate(&domain, &query, &EnumLimits::unfiltered()).unwrap();
    assert_eq!(found.compositions, vec![single]);
    assert_eq!(found.ambiguous, vec![true]);
    assert!(!found.absent);

    // The `u32::MAX` uncertainty sentinel skips the search exactly like the table.
    let sentinel = EnumQuery {
        precursor_uncertainty: u32::MAX,
        ..query
    };
    let found = enumerate(&domain, &sentinel, &EnumLimits::unfiltered()).unwrap();
    assert!(found.absent);
    assert!(!found.exhausted);
    assert_eq!(found.nodes_visited, 0);
    assert_eq!(found.hydrogen_checks, 0);
    assert!(found.compositions.is_empty());
    assert_eq!(
        found.status,
        request_status::EXACT_MASS_UNAVAILABLE | request_status::FORMULA_ABSENT
    );
    let table = FormulaTable::from_compositions([single]).unwrap();
    let table_found = table.window(&WindowQuery {
        precursor_mz: precursor,
        adduct: 1,
        ppm_tenths: 1,
        precursor_uncertainty: u32::MAX,
        rows_visited_max: u32::MAX,
        rows_scored_max: u32::MAX,
    });
    assert_eq!(found.status, table_found.status);
    assert_eq!(found.absent, table_found.absent);

    // Mass overflow: precursor 0 under adduct 1 underflows the parent mass.
    let overflow = EnumQuery {
        precursor_mz: 0,
        ..query
    };
    let found = enumerate(&domain, &overflow, &EnumLimits::unfiltered()).unwrap();
    assert_eq!(found.status, request_status::MASS_OVERFLOW);
    assert_eq!(found.parent_mass, None);
    assert!(!found.absent);
    assert!(!found.support_complete);

    // An empty domain completes and joins nothing.
    let empty = EnumDomain {
        version: "test-empty".to_string(),
        heavy_caps: [0; 9],
        heavy_max: 0,
        hydrogen_min: 0,
        hydrogen_max: 0,
    };
    let found = enumerate(&empty, &query, &EnumLimits::unfiltered()).unwrap();
    assert!(found.absent);
    assert!(!found.exhausted);
    assert!(found.support_complete);

    // A domain capped at zero for every heavy element holds hydrogen only,
    // which is never in-domain: nothing joins.
    let h_only = EnumDomain {
        version: "test-h-only".to_string(),
        heavy_caps: [0; 9],
        heavy_max: 0,
        hydrogen_min: 0,
        hydrogen_max: 10,
    };
    let found = enumerate(&h_only, &query, &EnumLimits::unfiltered()).unwrap();
    assert!(found.compositions.is_empty());
    assert!(found.absent);
    assert!(!found.exhausted);
}

#[test]
fn decide_agrees_with_brute_verdict_on_fixtures() {
    // The shared decision rule is what both paths call: spot-check `decide`
    // against the independently written verdict on fixture masses.
    for gold in fixture_compositions().iter().take(8) {
        let mass = composition_mass(gold).unwrap();
        assert_eq!(mass, brute_mass(gold));
        for adduct in [1u16, 2] {
            let precursor = if adduct == 1 {
                protonated(gold)
            } else {
                deprotonated(gold)
            };
            let parent = brute_parent(precursor, adduct);
            let tol = brute_tolerance(precursor, 200);
            let error = brute_error(gold) + 51;
            let (joined, ambiguous) = brute_verdict(parent, mass, error, tol);
            let verdict = decide(parent, mass, error, tol);
            assert_eq!(
                (joined, ambiguous),
                (
                    verdict != mamba3::models::ms2::Verdict::Reject,
                    verdict == mamba3::models::ms2::Verdict::Ambiguous
                )
            );
        }
    }
}

#[test]
fn capacity_keeps_the_canonical_prefix() {
    // Review finding 1, verbatim concrete case: heavy caps C=2, O=1, every
    // other cap zero; heavy total at most 2; hydrogen fixed at zero. Parent
    // 20_000_000, protonated precursor 21_007_276, uncertainty 10_000_000,
    // zero ppm, capacity 2. DFS joins C, C2, O, CO in that order; the
    // canonical first two are C (12 Da) and O (15.994915 Da), not C and C2.
    let domain = EnumDomain {
        version: "test-cap-prefix".to_string(),
        heavy_caps: [2, 0, 1, 0, 0, 0, 0, 0, 0],
        heavy_max: 2,
        hydrogen_min: 0,
        hydrogen_max: 0,
    };
    let query = EnumQuery {
        precursor_mz: 21_007_276,
        adduct: 1,
        ppm_tenths: 0,
        precursor_uncertainty: 10_000_000,
    };
    assert_eq!(brute_parent(query.precursor_mz, query.adduct), 20_000_000);
    let full = enumerate(&domain, &query, &EnumLimits::unfiltered()).unwrap();
    assert!(!full.exhausted, "open search completes");
    assert_eq!(
        full.masses,
        vec![12_000_000, 15_994_915, 24_000_000, 27_994_915]
    );
    let carbon: Composition = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let oxygen: Composition = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0];
    assert_eq!(full.compositions[0], carbon);
    assert_eq!(full.compositions[1], oxygen);
    let capped = enumerate(
        &domain,
        &query,
        &EnumLimits {
            capacity: 2,
            ..EnumLimits::unfiltered()
        },
    ).unwrap();
    assert!(capped.exhausted, "capacity overflow reports exhausted");
    assert_eq!(capped.rows_joined, full.rows_joined);
    assert_eq!(capped.rows_joined, 4);
    assert_eq!(capped.rows_scored, 2);
    assert_eq!(capped.masses, vec![12_000_000, 15_994_915]);
    assert_eq!(capped.compositions, vec![carbon, oxygen]);
    assert_eq!(capped.ambiguous, vec![true, true]);
}

#[test]
fn capped_output_is_the_uncapped_canonical_prefix() {
    // Randomized check with a deterministic xorshift64* stream (own
    // arithmetic, no dependency): for many queries, every capped output
    // equals the first `capacity` rows of the uncapped canonical output,
    // while every join is still counted. The scored cap truncates the same
    // prefix.
    let domain = EnumDomain {
        version: "test-cap-rand".to_string(),
        heavy_caps: [4, 2, 2, 0, 0, 0, 0, 0, 0],
        heavy_max: 8,
        hydrogen_min: 0,
        hydrogen_max: 8,
    };
    let ppms = [0u32, 1, 10, 100, 200, 1000];
    let mut rng: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = || {
        rng ^= rng >> 12;
        rng ^= rng << 25;
        rng ^= rng >> 27;
        rng = rng.wrapping_mul(0x2545_F491_4F6C_DD1D);
        rng
    };
    for _ in 0..200 {
        let base = 10_000_000 + (next() % 60_000_000) as u32;
        let adduct = if next() % 2 == 0 { 1u16 } else { 2 };
        let precursor = if adduct == 1 {
            base + 1_007_825 - 549
        } else {
            base - 1_007_825 + 549
        };
        let query = EnumQuery {
            precursor_mz: precursor,
            adduct,
            ppm_tenths: ppms[(next() % ppms.len() as u64) as usize],
            precursor_uncertainty: (next() % 3_000_001) as u32,
        };
        let full = enumerate(&domain, &query, &EnumLimits::unfiltered()).unwrap();
        assert!(!full.exhausted, "small domain never exhausts open");
        for cap in [0usize, 1, 2, 3, 5, 13] {
            let capped = enumerate(
                &domain,
                &query,
                &EnumLimits {
                    capacity: cap,
                    ..EnumLimits::unfiltered()
                },
            ).unwrap();
            let keep = full.compositions.len().min(cap);
            assert_eq!(&capped.compositions[..], &full.compositions[..keep]);
            assert_eq!(&capped.masses[..], &full.masses[..keep]);
            assert_eq!(&capped.ambiguous[..], &full.ambiguous[..keep]);
            assert_eq!(capped.rows_joined, full.rows_joined);
            assert_eq!(capped.exhausted, full.compositions.len() > cap);
            assert_eq!(capped.rows_scored, keep as u64);
            let scored = enumerate(
                &domain,
                &query,
                &EnumLimits {
                    scored_max: cap,
                    ..EnumLimits::unfiltered()
                },
            ).unwrap();
            assert_eq!(&scored.compositions[..], &full.compositions[..keep]);
            assert_eq!(scored.rows_scored, keep as u64);
            assert_eq!(scored.exhausted, full.compositions.len() > cap);
        }
    }
}

#[test]
fn gold_stages_gate_unknown_precision_and_bad_parent() {
    // Finding 6: with the unknown-precision sentinel or an invalid
    // parent-mass derivation the gold contributes to NO recall stage, even
    // with a universal window; the spectrum stays countable through
    // `exact_mass_unavailable`.
    let gold: Composition = [1, 4, 0, 0, 0, 0, 0, 0, 0, 0];
    let mass = brute_mass(&gold);
    let tol = 1000u32;
    let none = |stages: mamba3::models::ms2::formula_enum::GoldStages| {
        !stages.window
            && !stages.accepted
            && !stages.accepted_or_ambiguous
            && !stages.after_h_max
            && !stages.after_parity
            && !stages.after_dbe
    };
    // Exact gold, known precision: every stage holds, nothing unavailable.
    let ok = gold_stages(Some(mass), Some(&gold), Some(mass), tol, 50, 1_000_000, true);
    assert!(ok.window);
    assert!(ok.accepted);
    assert!(ok.accepted_or_ambiguous);
    assert!(ok.after_h_max);
    assert!(ok.after_parity);
    assert!(ok.after_dbe);
    assert!(!ok.exact_mass_unavailable);
    // The sentinel gates every stage, however wide the window.
    let gated = gold_stages(
        Some(mass),
        Some(&gold),
        Some(mass),
        tol,
        u32::MAX,
        u64::from(u32::MAX),
        true,
    );
    assert!(none(gated));
    assert!(gated.exact_mass_unavailable);
    // An invalid parent-mass derivation does the same.
    let bad = gold_stages(None, Some(&gold), Some(mass), tol, 50, 1_000_000, true);
    assert!(none(bad));
    assert!(bad.exact_mass_unavailable);
    // Out-of-domain gold is a miss at every stage without unavailability.
    let out = gold_stages(Some(mass), Some(&gold), Some(mass), tol, 50, 1_000_000, false);
    assert!(none(out));
    assert!(!out.exact_mass_unavailable);
    // Missing gold (an unbuildable molecule graph) is likewise a miss.
    let missing = gold_stages(Some(mass), None, None, tol, 50, 1_000_000, true);
    assert!(none(missing));
    assert!(!missing.exact_mass_unavailable);
}

#[test]
fn ratio_bounds_fit_passes_training_compositions() {
    // Soundness half 1: every composition used to fit a bound passes it
    // (by construction), checked on the fixture compositions with margin 0.
    assert_eq!(RATIO_FEATURES.len(), 6);
    let comps = fixture_compositions();
    let bounds = RatioBounds::fit(comps.iter().copied(), 0).unwrap();
    assert_eq!(bounds.version, "ms2-ratio-v2");
    assert!(bounds.bytes() > 0);
    for (i, c) in comps.iter().enumerate() {
        assert!(bounds.passes_cap(c), "molecule {i}: carbon/heavy caps");
        assert!(bounds.passes_rare_max(c), "molecule {i}: rare maxima");
        assert!(bounds.passes_rare_min(c), "molecule {i}: rare minima");
        for (k, name) in RATIO_FEATURES.iter().enumerate() {
            assert!(
                bounds.passes_ratio(k, c),
                "molecule {i}: ratio {name}",
            );
        }
        assert!(bounds.passes_ratio_dbe(c), "molecule {i}: DBE bucket");
        assert!(!bounds.passes_ratio(RATIO_FEATURES.len(), c));
    }
    // A widened fit still passes everything it was fitted on.
    let wide = RatioBounds::fit(comps.iter().copied(), 5).unwrap();
    for (i, c) in comps.iter().enumerate() {
        assert!(wide.passes_cap(c), "molecule {i}: widened caps");
        assert!(wide.passes_rare_min(c), "molecule {i}: widened rare minima");
        for k in 0..RATIO_FEATURES.len() {
            assert!(wide.passes_ratio(k, c), "molecule {i}: widened ratio {k}");
        }
        assert!(wide.passes_ratio_dbe(c), "molecule {i}: widened DBE");
    }
    // JSON round trip; malformed input is an error.
    let back = RatioBounds::from_json(&bounds.to_json()).unwrap();
    assert_eq!(back, bounds);
    assert!(RatioBounds::from_json("{not json").is_err());
}

/// Independently written ratio predicate over a training slice, with own
/// loops and the test literals only (carbon buckets of width 4,
/// heavy-total buckets of width 4, fractions by cross-multiplication,
/// rare totals/distinct, DBE from the test valences), widened by `margin`
/// exactly as production widens: observed bucketed maxima plus `margin`,
/// ratio numerators minus/plus `margin` with denominators unchanged (ties
/// broken by smallest denominator, mirroring production), rare/DBE ranges
/// widened both ways. Buckets the slice never saw admit nothing, whatever
/// the margin.
fn brute_ratio_ok(subset: &[Composition], c: &Composition) -> bool {
    brute_ratio_ok_margin(subset, c, 0)
}

fn brute_ratio_ok_margin(subset: &[Composition], c: &Composition, margin: u16) -> bool {
    let margin32 = u32::from(margin);
    // (i) carbon- and heavy-total-bucketed maxima (observed only, then margin).
    let carbon_bucket = usize::from(c[0]) / 4;
    let mut heavy: u32 = 0;
    for e in BRUTE_HEAVY {
        heavy += u32::from(c[e]);
    }
    let heavy_bucket = (heavy / 4) as usize;
    for e in BRUTE_HEAVY {
        let mut cap_c: Option<u16> = None;
        let mut cap_h: Option<u16> = None;
        for s in subset {
            if usize::from(s[0]) / 4 == carbon_bucket {
                cap_c = Some(cap_c.map_or(s[e], |v: u16| v.max(s[e])));
            }
            let mut sh: u32 = 0;
            for f in BRUTE_HEAVY {
                sh += u32::from(s[f]);
            }
            if (sh / 4) as usize == heavy_bucket {
                cap_h = Some(cap_h.map_or(s[e], |v: u16| v.max(s[e])));
            }
        }
        match (cap_c, cap_h) {
            (Some(a), Some(b)) => {
                let a = a.saturating_add(margin);
                let b = b.saturating_add(margin);
                if c[e] > a || c[e] > b {
                    return false;
                }
            }
            _ => return false,
        }
    }
    // (iii) rare total and distinct count within the observed minima/maxima,
    // widened by the margin.
    let rare_ids = [4usize, 5, 6, 7, 8, 9];
    let mut total = 0u32;
    let mut distinct = 0u32;
    for e in rare_ids {
        total += u32::from(c[e]);
        if c[e] > 0 {
            distinct += 1;
        }
    }
    let (mut lo, mut hi) = (u32::MAX, 0u32);
    let (mut dlo, mut dhi) = (u32::MAX, 0u32);
    for s in subset {
        let mut t = 0u32;
        let mut d = 0u32;
        for e in rare_ids {
            t += u32::from(s[e]);
            if s[e] > 0 {
                d += 1;
            }
        }
        lo = lo.min(t);
        hi = hi.max(t);
        dlo = dlo.min(d);
        dhi = dhi.max(d);
    }
    if subset.is_empty() {
        return false;
    }
    if total < lo.saturating_sub(margin32)
        || total > hi.saturating_add(margin32)
        || distinct < dlo.saturating_sub(margin32)
        || distinct > dhi.saturating_add(margin32)
    {
        return false;
    }
    // (ii) ratios: a zero-carbon candidate skips only the ratio comparisons
    // (it still faces the DBE stage below); otherwise every ratio must hold.
    // Both extrema start from the first positive-carbon observation, so a
    // positive minimum is learned; ties keep the smallest denominator.
    let zero_ok = subset.iter().any(|s| s[0] == 0);
    if c[0] == 0 && !zero_ok {
        return false;
    }
    if c[0] != 0 {
        let nums = [
            u32::from(c[1]),
            u32::from(c[2]),
            u32::from(c[3]),
            u32::from(c[4]) + u32::from(c[7]) + u32::from(c[8]) + u32::from(c[9]),
            u32::from(c[6]),
            u32::from(c[5]),
        ];
        for (k, n) in nums.iter().enumerate() {
            let mut first = true;
            let (mut ln, mut ld, mut hn, mut hd) = (0u64, 1u64, 0u64, 1u64);
            for s in subset {
                if s[0] == 0 {
                    continue;
                }
                let m = [
                    u32::from(s[1]),
                    u32::from(s[2]),
                    u32::from(s[3]),
                    u32::from(s[4]) + u32::from(s[7]) + u32::from(s[8]) + u32::from(s[9]),
                    u32::from(s[6]),
                    u32::from(s[5]),
                ][k] as u64;
                let d = u64::from(s[0]);
                if first {
                    ln = m;
                    ld = d;
                    hn = m;
                    hd = d;
                    first = false;
                    continue;
                }
                if m * ld < ln * d || (m * ld == ln * d && d < ld) {
                    ln = m;
                    ld = d;
                }
                if m * hd > hn * d || (m * hd == hn * d && d < hd) {
                    hn = m;
                    hd = d;
                }
            }
            if first {
                // No positive-carbon train: production holds the default
                // `(0, 1)` bounds widened by the margin, i.e. lower 0 and
                // upper `margin` over denominator 1, admitting exactly the
                // numerators `n <= margin * carbon` (the margin applies to
                // the default fractions, not just to observed ones).
                let allow = u64::from(margin32) * u64::from(c[0]);
                if u64::from(*n) > allow {
                    return false;
                }
                continue;
            }
            let ln = ln.saturating_sub(u64::from(margin32));
            let hn = hn.saturating_add(u64::from(margin32));
            let n = u64::from(*n);
            let carbon = u64::from(c[0]);
            if n * ld < ln * carbon || n * hd > hn * carbon {
                return false;
            }
        }
    }
    // (iv) DBE within the observed range of the heavy-total bucket, widened
    // by the margin.
    let mut twice: i64 = 2;
    for e in BRUTE_HEAVY {
        twice += (i64::from(BRUTE_VMAX[e]) - 2) * i64::from(c[e]);
    }
    twice -= i64::from(c[1]);
    let (mut blo, mut bhi) = (i64::MAX, i64::MIN);
    let mut seen = false;
    for s in subset {
        let mut sh: u32 = 0;
        for e in BRUTE_HEAVY {
            sh += u32::from(s[e]);
        }
        if (sh / 4) as usize != heavy_bucket {
            continue;
        }
        seen = true;
        let mut t: i64 = 2;
        for e in BRUTE_HEAVY {
            t += (i64::from(BRUTE_VMAX[e]) - 2) * i64::from(s[e]);
        }
        t -= i64::from(s[1]);
        blo = blo.min(t);
        bhi = bhi.max(t);
    }
    if !seen {
        return false;
    }
    seen && twice >= blo.saturating_sub(i64::from(margin)) && twice <= bhi.saturating_add(i64::from(margin))
}

/// All production ratio stages combined (the oracle's twin).
fn production_ratio_ok(bounds: &RatioBounds, c: &Composition) -> bool {
    bounds.passes_cap(c)
        && bounds.passes_rare_max(c)
        && bounds.passes_rare_min(c)
        && (0..RATIO_FEATURES.len()).all(|k| bounds.passes_ratio(k, c))
        && bounds.passes_ratio_dbe(c)
}

#[test]
fn pruned_enumeration_equals_filtered_brute_force() {
    // Soundness half 2: the pruned enumeration over the tiny domain with
    // bounds fitted on a strict subset equals the unpruned result filtered
    // by the independently written predicate above.
    let subset: Vec<Composition> = tiny_compositions()
        .into_iter()
        .filter(|c| c[0] <= 1 && c[1] <= 4)
        .collect();
    assert!(!subset.is_empty());
    assert!(subset.len() < tiny_compositions().len());
    let bounds = RatioBounds::fit(subset.iter().copied(), 0).unwrap();
    for c in &subset {
        assert!(bounds.passes_cap(c));
        assert!(bounds.passes_rare_max(c) && bounds.passes_rare_min(c));
        for k in 0..RATIO_FEATURES.len() {
            assert!(bounds.passes_ratio(k, c));
        }
        assert!(bounds.passes_ratio_dbe(c));
    }
    let domain = tiny_domain();
    let bases: Vec<Composition> = [
        [1, 4, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 6, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 2, 1, 1, 0, 0, 0, 0, 0, 0],
        [2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    .to_vec();
    let mut saw_prune = false;
    for adduct in [1u16, 2] {
        for ppm in [100u32, 1000] {
            for base in &bases {
                let anchor = if adduct == 1 {
                    protonated(base)
                } else {
                    deprotonated(base)
                };
                for delta in [-1000i64, 0, 1000] {
                    let precursor = (anchor as i64 + delta) as u32;
                    let query = EnumQuery {
                        precursor_mz: precursor,
                        adduct,
                        ppm_tenths: ppm,
                        precursor_uncertainty: 500,
                    };
                    let limits = EnumLimits {
                        ratio: Some(bounds.clone()),
                        ..EnumLimits::unfiltered()
                    };
                    let pruned = enumerate(&domain, &query, &limits).unwrap();
                    assert!(!pruned.exhausted, "tiny domain never exhausts");
                    assert!(pruned.support_complete);
                    let parent = brute_parent(precursor, adduct);
                    let tol = brute_tolerance(precursor, ppm);
                    let expected: Vec<(u32, Composition, bool)> =
                        brute_expected(parent, tol, 501, false)
                            .into_iter()
                            .filter(|(_, c, _)| brute_ratio_ok(&subset, c))
                            .collect();
                    assert_eq!(
                        enum_rows(&domain, &query, &limits),
                        expected,
                        "adduct {adduct} ppm {ppm} delta {delta}"
                    );
                    // Every hydrogen check ends in exactly one bucket,
                    // now including the ratio stages.
                    let ratio_rejects = pruned.rejected_ratio_hc
                        + pruned.rejected_ratio_nc
                        + pruned.rejected_ratio_oc
                        + pruned.rejected_ratio_hal
                        + pruned.rejected_ratio_s
                        + pruned.rejected_ratio_p;
                    let counted = pruned.rejected_mass
                        + pruned.rejected_h_max
                        + pruned.rejected_parity
                        + pruned.rejected_dbe
                        + pruned.rejected_ratio_cap
                        + pruned.rejected_rare
                        + ratio_rejects
                        + pruned.rejected_ratio_dbe
                        + pruned.rows_joined;
                    assert_eq!(pruned.hydrogen_checks, counted);
                    let open = enumerate(&domain, &query, &EnumLimits::unfiltered()).unwrap();
                    saw_prune =
                        saw_prune || pruned.rows_joined < open.rows_joined;
                }
            }
        }
    }
    assert!(saw_prune, "the subset-fitted bounds prune something");
}

#[test]
fn ratio_stages_gate_zero_carbon_explicitly() {
    // Zero-carbon compositions pass the (ii) ratios exactly when the fit saw
    // zero-carbon train compositions; every other stage applies normally.
    let bare: Composition = [0, 2, 1, 1, 0, 0, 0, 0, 0, 0];
    let meth: Composition = [1, 4, 0, 0, 0, 0, 0, 0, 0, 0];
    let with = RatioBounds::fit([bare, meth], 0).unwrap();
    assert!(with.zero_carbon_seen);
    for k in 0..RATIO_FEATURES.len() {
        assert!(with.passes_ratio(k, &bare), "ratio {k} with zero carbon");
    }
    let without = RatioBounds::fit([meth], 0).unwrap();
    assert!(!without.zero_carbon_seen);
    for k in 0..RATIO_FEATURES.len() {
        assert!(!without.passes_ratio(k, &bare), "ratio {k} gates zero carbon");
    }
}

// ---------------------------------------------------------------------------
// H3 finding 1: order-independent `RatioBounds::fit`.
// ---------------------------------------------------------------------------

#[test]
fn ratio_fit_is_order_independent() {
    // Review finding 1, verbatim case: fit `[C4H8, C6H12]` with margin 2.
    // The H/C fractions 8/4 and 12/6 tie at 2; the stored bound keeps the
    // smallest denominator (4), widening to `[6/4, 10/4]` in every order, so
    // pentane `C5H12` is admitted deterministically.
    let a: Composition = [4, 8, 0, 0, 0, 0, 0, 0, 0, 0];
    let b: Composition = [6, 12, 0, 0, 0, 0, 0, 0, 0, 0];
    let pentane: Composition = [5, 12, 0, 0, 0, 0, 0, 0, 0, 0];
    for margin in [0u16, 2] {
        let fwd = RatioBounds::fit([a, b], margin).unwrap();
        let rev = RatioBounds::fit([b, a], margin).unwrap();
        assert_eq!(
            fwd.to_json(),
            rev.to_json(),
            "margin {margin}: byte-identical bounds"
        );
        assert_eq!(fwd, rev);
    }
    let bounds = RatioBounds::fit([a, b], 2).unwrap();
    assert_eq!(
        (bounds.ratio_lo_num[0], bounds.ratio_lo_den[0]),
        (6, 4),
        "H/C lower bound keeps denominator 4"
    );
    assert_eq!(
        (bounds.ratio_hi_num[0], bounds.ratio_hi_den[0]),
        (10, 4),
        "H/C upper bound keeps denominator 4"
    );
    assert!(
        bounds.passes_ratio(0, &pentane),
        "pentane is admitted deterministically"
    );
    // Permutations of three compositions: byte-identical JSON and identical
    // enumeration, with margin 0 and margin 2.
    let c: Composition = [3, 6, 1, 1, 0, 0, 0, 0, 0, 0];
    let base = [a, b, c];
    let mut orders: Vec<[Composition; 3]> = Vec::new();
    for i in 0..3 {
        for j in 0..3 {
            if j == i {
                continue;
            }
            let k = 3 - i - j;
            if k == i || k == j || k > 2 {
                continue;
            }
            orders.push([base[i], base[j], base[k]]);
        }
    }
    assert_eq!(orders.len(), 6);
    let domain = EnumDomain {
        version: "test-perm".to_string(),
        heavy_caps: [6, 1, 1, 0, 0, 0, 0, 0, 0],
        heavy_max: 16,
        hydrogen_min: 0,
        hydrogen_max: 14,
    };
    let query = EnumQuery {
        precursor_mz: protonated(&b),
        adduct: 1,
        ppm_tenths: 200,
        precursor_uncertainty: 500,
    };
    for margin in [0u16, 2] {
        let first = RatioBounds::fit(orders[0].iter().copied(), margin).unwrap();
        for order in &orders[1..] {
            let other = RatioBounds::fit(order.iter().copied(), margin).unwrap();
            assert_eq!(first.to_json(), other.to_json(), "margin {margin}");
        }
        let expected = enum_rows(
            &domain,
            &query,
            &EnumLimits {
                ratio: Some(first.clone()),
                ..EnumLimits::unfiltered()
            },
        );
        assert!(!expected.is_empty(), "margin {margin}: non-empty support");
        for order in &orders[1..] {
            let fitted = RatioBounds::fit(order.iter().copied(), margin).unwrap();
            assert_eq!(
                enum_rows(
                    &domain,
                    &query,
                    &EnumLimits {
                        ratio: Some(fitted),
                        ..EnumLimits::unfiltered()
                    },
                ),
                expected,
                "margin {margin}: identical enumeration"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// H3 finding 2: the reviewer's oracle counterexamples.
// ---------------------------------------------------------------------------

#[test]
fn ratio_oracle_matches_production_on_review_counterexamples() {
    // Counterexample 1: train `[C2H2, C3H4]`, candidate `C1H0`, margin 0.
    // H/C below 1 rejects; the oracle learns the positive minimum from the
    // first positive-carbon observation.
    let t1: Composition = [2, 2, 0, 0, 0, 0, 0, 0, 0, 0];
    let t2: Composition = [3, 4, 0, 0, 0, 0, 0, 0, 0, 0];
    let cand: Composition = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let subset = [t1, t2];
    let bounds = RatioBounds::fit(subset.iter().copied(), 0).unwrap();
    assert!(
        !bounds.passes_ratio(0, &cand),
        "H/C below 1 is rejected by production"
    );
    assert!(
        !brute_ratio_ok(&subset, &cand),
        "H/C below 1 is rejected by the oracle"
    );
    assert_eq!(
        brute_ratio_ok(&subset, &cand),
        production_ratio_ok(&bounds, &cand),
        "oracle == production"
    );
    // Counterexample 2: train `[H2O]`, candidate `O`. Zero carbon skips only
    // the ratio comparisons; the DBE stage still rejects twice-DBE 2
    // outside `[0, 0]`.
    let h2o: Composition = [0, 2, 0, 1, 0, 0, 0, 0, 0, 0];
    let oxygen: Composition = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0];
    let subset = [h2o];
    let bounds = RatioBounds::fit(subset.iter().copied(), 0).unwrap();
    assert!(
        !bounds.passes_ratio_dbe(&oxygen),
        "production DBE rejects atomic oxygen"
    );
    assert!(
        !production_ratio_ok(&bounds, &oxygen),
        "production rejects atomic oxygen"
    );
    assert!(
        !brute_ratio_ok(&subset, &oxygen),
        "the oracle rejects atomic oxygen"
    );
    assert_eq!(
        brute_ratio_ok(&subset, &oxygen),
        production_ratio_ok(&bounds, &oxygen),
        "oracle == production"
    );
    // The supported zero-carbon train composition still passes both.
    assert!(production_ratio_ok(&bounds, &h2o));
    assert!(brute_ratio_ok(&subset, &h2o));
}

// ---------------------------------------------------------------------------
// H3 finding 4: unseen buckets admit nothing at any margin.
// ---------------------------------------------------------------------------

#[test]
fn unseen_buckets_admit_nothing_at_any_margin() {
    // Review finding 4, verbatim case: train `[CH4, CH4O4, C8H18]`.
    // Carbon bucket 1 was never observed, yet `C4H10` passed every fitted v1
    // stage at margin 4. With presence tracking it is rejected at any margin.
    let ch4: Composition = [1, 4, 0, 0, 0, 0, 0, 0, 0, 0];
    let ch4o4: Composition = [1, 4, 0, 4, 0, 0, 0, 0, 0, 0];
    let c8h18: Composition = [8, 18, 0, 0, 0, 0, 0, 0, 0, 0];
    let train = [ch4, ch4o4, c8h18];
    let c4h10: Composition = [4, 10, 0, 0, 0, 0, 0, 0, 0, 0];
    for margin in [0u16, 1, 4] {
        let bounds = RatioBounds::fit(train.iter().copied(), margin).unwrap();
        assert_eq!(bounds.version, "ms2-ratio-v2");
        assert!(
            !bounds.passes_cap(&c4h10),
            "margin {margin}: unseen carbon bucket 1 admits nothing"
        );
        assert!(
            !brute_ratio_ok_margin(&train, &c4h10, margin),
            "margin {margin}: the oracle rejects the unseen bucket"
        );
        assert_eq!(
            brute_ratio_ok_margin(&train, &c4h10, margin),
            production_ratio_ok(&bounds, &c4h10),
            "margin {margin}: oracle == production"
        );
    }
    // Presence is tracked explicitly: carbon buckets 0 and 2 seen, 1 unseen.
    let bounds = RatioBounds::fit(train.iter().copied(), 4).unwrap();
    assert_eq!(bounds.carbon_seen, vec![true, false, true]);
    assert_eq!(bounds.heavy_seen, vec![true, true, true]);
    // The presence vectors survive the JSON round trip.
    let back = RatioBounds::from_json(&bounds.to_json()).unwrap();
    assert_eq!(back, bounds);
    // Seen buckets still admit their train at margin 4.
    for c in &train {
        assert!(
            production_ratio_ok(&bounds, c),
            "widened bounds keep the train"
        );
    }
}

// ---------------------------------------------------------------------------
// H3 finding 3: exhaustive pruning coverage.
// ---------------------------------------------------------------------------

/// The domain of the extended pruning test: C 0–3, N 0–1, O 0–2, F 0–2,
/// S 0–1, H 0–8, heavy total at most 6.
fn rare_tiny_domain() -> EnumDomain {
    EnumDomain {
        version: "test-rare-tiny".to_string(),
        heavy_caps: [3, 1, 2, 2, 0, 1, 0, 0, 0],
        heavy_max: 6,
        hydrogen_min: 0,
        hydrogen_max: 8,
    }
}

/// Train compositions of the extended pruning test: the reviewer's bucket
/// boundary pair (`CH4`, `C2H2O2`), rare carriers (`CH3F`, `C2H6S`), the
/// non-monotone heavy-cap pair (`C1H4N1` in heavy bucket 0, `C2H2O2` in
/// heavy bucket 1 with no nitrogen), zero-carbon anchors (`[0,2,1,1]`, `O2`)
/// so the (ii) stages admit zero carbon, and DBE anchors (`C2`, `O2`) so
/// the heavy-bucket-0 DBE range spans `[0, 6]`.
fn extended_subset() -> Vec<Composition> {
    vec![
        [1, 4, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 2, 0, 2, 0, 0, 0, 0, 0, 0],
        [1, 3, 0, 0, 1, 0, 0, 0, 0, 0],
        [2, 6, 0, 0, 0, 0, 1, 0, 0, 0],
        [1, 4, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 2, 1, 1, 0, 0, 0, 0, 0, 0],
        [2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 2, 0, 0, 0, 0, 0, 0],
    ]
}

/// Brute-force expected joins over the rare-tiny domain: every heavy vector
/// in caps, every hydrogen count, the independent verdict, then the
/// independently written margin oracle.
#[allow(clippy::too_many_arguments)]
fn brute_expected_rare(
    parent: u32,
    tol: u32,
    bound: u32,
    subset: &[Composition],
    margin: u16,
) -> Vec<(u32, Composition, bool)> {
    let domain = rare_tiny_domain();
    let (c_max, n_max, o_max, f_max, s_max) = (
        domain.heavy_caps[0],
        domain.heavy_caps[1],
        domain.heavy_caps[2],
        domain.heavy_caps[3],
        domain.heavy_caps[5],
    );
    let mut out = Vec::new();
    for c in 0..=c_max {
        for n in 0..=n_max {
            for o in 0..=o_max {
                for f in 0..=f_max {
                    for s in 0..=s_max {
                        let heavy = c + n + o + f + s;
                        if heavy == 0 || heavy > domain.heavy_max {
                            continue;
                        }
                        for h in 0..=domain.hydrogen_max {
                            let mut comp: Composition = [0; 10];
                            comp[0] = c;
                            comp[1] = h;
                            comp[2] = n;
                            comp[3] = o;
                            comp[4] = f;
                            comp[6] = s;
                            let mass = brute_mass(&comp);
                            let error = brute_error(&comp).saturating_add(bound);
                            let (joined, ambiguous) = brute_verdict(parent, mass, error, tol);
                            if !joined {
                                continue;
                            }
                            if !brute_ratio_ok_margin(subset, &comp, margin) {
                                continue;
                            }
                            out.push((mass, comp, ambiguous));
                        }
                    }
                }
            }
        }
    }
    out.sort_by_key(|a| (a.0, a.1));
    out
}

/// Query centred on a composition (adduct 1, fixed ppm/uncertainty).
fn centred_query(center: &Composition, ppm: u32, uncertainty: u32) -> EnumQuery {
    EnumQuery {
        precursor_mz: protonated(center),
        adduct: 1,
        ppm_tenths: ppm,
        precursor_uncertainty: uncertainty,
    }
}

#[test]
fn pruned_enumeration_extended_buckets_rare_margins() {
    // Finding 3: rare elements present, carbon crossing a bucket boundary,
    // non-monotone heavy caps, positive margins, rare total/distinct limits.
    let domain = rare_tiny_domain();
    let subset = extended_subset();
    let gold: Composition = [2, 2, 0, 2, 0, 0, 0, 0, 0, 0];
    // N=1 with heavy total 4 (bucket 1, whose N cap is 0): checked, then
    // rejected by the (i) leaf refinement, not by the DFS prefix.
    let heavy_reject: Composition = [2, 4, 1, 1, 0, 0, 0, 0, 0, 0];
    // F1+S1 (rare total 2 above the train maximum 1): pruned in the DFS by
    // the (iii) maxima before any hydrogen work.
    let rare_reject: Composition = [1, 4, 0, 0, 1, 0, 1, 0, 0, 0];
    // F2 (above the train F maximum 1 with the domain cap at 2): pruned in
    // the DFS by the (iii) maxima before any hydrogen work.
    let rare_pruned: Composition = [2, 4, 0, 0, 2, 0, 0, 0, 0, 0];
    // S1H0: a multi-row query (O2H0 and C1O1H4 join alongside across the
    // S lane and lane 0) for the rank/prefix siblings below.
    let multi: Composition = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0];
    let queries = [
        centred_query(&gold, 1000, 500_000),
        centred_query(&heavy_reject, 1000, 500_000),
        centred_query(&rare_reject, 1000, 500_000),
        centred_query(&rare_pruned, 1000, 500_000),
        centred_query(&multi, 1000, 100_000),
    ];
    for margin in [0u16, 2] {
        let bounds = RatioBounds::fit(subset.iter().copied(), margin).unwrap();
        for c in &subset {
            assert!(
                production_ratio_ok(&bounds, c),
                "margin {margin}: the train passes production"
            );
            assert!(
                brute_ratio_ok_margin(&subset, c, margin),
                "margin {margin}: the train passes the oracle"
            );
        }
        let mut saw_joined = false;
        let mut saw_pruned_rare = false;
        let mut saw_rejected_cap = false;
        let mut saw_rejected_dbe = false;
        for query in &queries {
            let limits = EnumLimits {
                ratio: Some(bounds.clone()),
                ..EnumLimits::unfiltered()
            };
            let pruned = enumerate(&domain, query, &limits).unwrap();
            assert!(!pruned.exhausted, "margin {margin}: rare-tiny never exhausts");
            assert!(pruned.support_complete);
            let parent = brute_parent(query.precursor_mz, query.adduct);
            let tol = brute_tolerance(query.precursor_mz, query.ppm_tenths);
            let expected = brute_expected_rare(parent, tol, 500_001, &subset, margin);
            // NOTE: the multi-row query uses uncertainty 100_000, so its
            // bound here is 100_001, not 500_001.
            let expected = if query.precursor_uncertainty == 100_000 {
                brute_expected_rare(parent, tol, 100_001, &subset, margin)
            } else {
                expected
            };
            assert_eq!(
                enum_rows(&domain, query, &limits),
                expected,
                "margin {margin}: pruned enumeration equals the oracle-filtered brute force"
            );
            // Every hydrogen check ends in exactly one bucket, now including
            // the ratio stages.
            let ratio_rejects = pruned.rejected_ratio_hc
                + pruned.rejected_ratio_nc
                + pruned.rejected_ratio_oc
                + pruned.rejected_ratio_hal
                + pruned.rejected_ratio_s
                + pruned.rejected_ratio_p;
            let counted = pruned.rejected_mass
                + pruned.rejected_h_max
                + pruned.rejected_parity
                + pruned.rejected_dbe
                + pruned.rejected_ratio_cap
                + pruned.rejected_rare
                + ratio_rejects
                + pruned.rejected_ratio_dbe
                + pruned.rows_joined;
            assert_eq!(pruned.hydrogen_checks, counted, "margin {margin}: checks partition");
            saw_joined = saw_joined || pruned.rows_joined > 0;
            saw_pruned_rare = saw_pruned_rare || pruned.pruned_rare > 0;
            saw_rejected_cap = saw_rejected_cap || pruned.rejected_ratio_cap > 0;
            saw_rejected_dbe = saw_rejected_dbe || pruned.rejected_ratio_dbe > 0;
        }
        assert!(saw_joined, "margin {margin}: non-zero survivors");
        if margin == 0 {
            assert!(saw_pruned_rare, "the (iii) DFS prune fires");
            assert!(
                saw_rejected_cap,
                "the non-monotone heavy cap rejects at the leaf"
            );
        } else {
            // At margin 2 the widened maxima prune less, but the DBE bucket
            // still rejects (e.g. the F2 composition, twice-DBE 1 outside
            // the widened bucket-1 range `[2, 6]`).
            assert!(saw_rejected_dbe, "margin 2: the DBE bucket still rejects");
        }
        // The reviewer's bucket-boundary case survives at both margins: the
        // gold is scored, and the carbon assignment's partial heavy total 2
        // in bucket 0 does not prune the completion in bucket 1.
        let gold_found = enumerate(
            &domain,
            &queries[0],
            &EnumLimits {
                ratio: Some(bounds.clone()),
                ..EnumLimits::unfiltered()
            },
        ).unwrap();
        assert!(
            gold_found.scored_contains(&gold),
            "margin {margin}: C2H2O2 survives across the bucket boundary"
        );
    }
    // At margin 0 the targeted rejects never join (pruned or leaf-rejected).
    let bounds = RatioBounds::fit(subset.iter().copied(), 0).unwrap();
    for (query, rejected) in [&queries[1], &queries[2], &queries[3]]
        .iter()
        .zip([heavy_reject, rare_reject, rare_pruned])
    {
        let found = enumerate(
            &domain,
            query,
            &EnumLimits {
                ratio: Some(bounds.clone()),
                ..EnumLimits::unfiltered()
            },
        ).unwrap();
        assert!(
            !found.scored_contains(&rejected),
            "margin 0: {rejected:?} is pruned or rejected"
        );
    }
}

#[test]
fn pruned_enumeration_rare_minima_reject_at_the_leaf() {
    // Finding 3, rare minima half: the (iii) maxima prune in the DFS (so a
    // leaf maximum failure is unreachable by construction), while the minima
    // are leaf-checked. A fit whose every train composition carries rare
    // atoms rejects a rare-free candidate at the leaf with `rejected_rare`.
    let domain = rare_tiny_domain();
    let subset: Vec<Composition> = vec![
        [1, 3, 0, 0, 1, 0, 0, 0, 0, 0],
        [2, 6, 0, 0, 0, 0, 1, 0, 0, 0],
    ];
    let bounds = RatioBounds::fit(subset.iter().copied(), 0).unwrap();
    assert_eq!(bounds.rare_total, [1, 1]);
    assert_eq!(bounds.rare_distinct, [1, 1]);
    let carrier: Composition = [1, 3, 0, 0, 1, 0, 0, 0, 0, 0];
    let bare: Composition = [1, 4, 0, 0, 0, 0, 0, 0, 0, 0];
    for query in [centred_query(&carrier, 1000, 500_000), centred_query(&bare, 1000, 500_000)] {
        let limits = EnumLimits {
            ratio: Some(bounds.clone()),
            ..EnumLimits::unfiltered()
        };
        let found = enumerate(&domain, &query, &limits).unwrap();
        assert!(!found.exhausted);
        let parent = brute_parent(query.precursor_mz, query.adduct);
        let tol = brute_tolerance(query.precursor_mz, query.ppm_tenths);
        assert_eq!(
            enum_rows(&domain, &query, &limits),
            brute_expected_rare(parent, tol, 500_001, &subset, 0),
            "minima fit matches the oracle"
        );
    }
    let carrier_found = enumerate(
        &domain,
        &centred_query(&carrier, 1000, 500_000),
        &EnumLimits {
            ratio: Some(bounds.clone()),
            ..EnumLimits::unfiltered()
        },
    ).unwrap();
    assert!(carrier_found.scored_contains(&carrier), "non-zero survivors");
    let bare_found = enumerate(
        &domain,
        &centred_query(&bare, 1000, 500_000),
        &EnumLimits {
            ratio: Some(bounds.clone()),
            ..EnumLimits::unfiltered()
        },
    ).unwrap();
    assert!(
        !bare_found.scored_contains(&bare),
        "the rare-free candidate is rejected"
    );
    assert!(bare_found.rejected_rare > 0, "the minima reject at the leaf");
    assert_eq!(
        brute_ratio_ok(&subset, &bare),
        production_ratio_ok(&bounds, &bare)
    );
}

#[test]
fn ratio_limits_combine_deterministically() {
    // Finding 3, second half: node, capacity and scored limits combined with
    // ratio bounds. The capped output is the uncapped canonical prefix; the
    // node limit stops deterministically; every check is still counted.
    let domain = rare_tiny_domain();
    let subset = extended_subset();
    let bounds = RatioBounds::fit(subset.iter().copied(), 0).unwrap();
    let gold: Composition = [2, 2, 0, 2, 0, 0, 0, 0, 0, 0];
    let query = centred_query(&gold, 1000, 500_000);
    let ratio_limits = |over: EnumLimits| EnumLimits {
        ratio: Some(bounds.clone()),
        ..over
    };
    let open = enumerate(&domain, &query, &ratio_limits(EnumLimits::unfiltered())).unwrap();
    assert!(!open.exhausted, "open search completes");
    assert!(open.rows_joined > 0, "non-zero survivors");
    // Capacity alone keeps the canonical prefix while counting every join.
    // The multi-row query joins several rows, so a capacity of 1 truncates.
    let multi: Composition = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0];
    let multi_query = centred_query(&multi, 1000, 100_000);
    let multi_open = enumerate(&domain, &multi_query, &ratio_limits(EnumLimits::unfiltered())).unwrap();
    assert!(!multi_open.exhausted);
    assert!(multi_open.rows_joined >= 2, "the multi-row query joins several rows");
    let capped = enumerate(
        &domain,
        &multi_query,
        &ratio_limits(EnumLimits {
            capacity: 1,
            ..EnumLimits::unfiltered()
        }),
    ).unwrap();
    assert!(capped.exhausted);
    assert_eq!(capped.rows_joined, multi_open.rows_joined);
    assert_eq!(capped.rows_scored, 1);
    assert_eq!(&capped.compositions[..], &multi_open.compositions[..1]);
    assert_eq!(&capped.masses[..], &multi_open.masses[..1]);
    // Combined node, capacity and scored limits: deterministic exhaustion.
    // The node budget allows the full traversal (it is present and equal),
    // so truncation comes from the capacity/scored caps with survivors.
    let combined = EnumLimits {
        nodes_visited_max: multi_open.nodes_visited,
        capacity: 1,
        scored_max: 1,
        ..EnumLimits::unfiltered()
    };
    let first = enumerate(&domain, &multi_query, &ratio_limits(combined.clone())).unwrap();
    let second = enumerate(&domain, &multi_query, &ratio_limits(combined)).unwrap();
    assert_eq!(first, second, "combined limits stop deterministically");
    assert!(first.exhausted);
    assert_eq!(first.rows_scored, 1);
    assert_eq!(first.rows_joined, multi_open.rows_joined);
    assert!(first.rows_joined > 0, "non-zero survivors under limits");
    // A stricter node budget stops part-way, deterministically.
    let tight = EnumLimits {
        nodes_visited_max: multi_open.nodes_visited.max(2) / 2,
        ..EnumLimits::unfiltered()
    };
    let a = enumerate(&domain, &multi_query, &ratio_limits(tight.clone())).unwrap();
    let b = enumerate(&domain, &multi_query, &ratio_limits(tight)).unwrap();
    assert_eq!(a, b, "node-limit stop is deterministic");
    assert!(a.exhausted);
    let ratio_rejects = first.rejected_ratio_hc
        + first.rejected_ratio_nc
        + first.rejected_ratio_oc
        + first.rejected_ratio_hal
        + first.rejected_ratio_s
        + first.rejected_ratio_p;
    let counted = first.rejected_mass
        + first.rejected_h_max
        + first.rejected_parity
        + first.rejected_dbe
        + first.rejected_ratio_cap
        + first.rejected_rare
        + ratio_rejects
        + first.rejected_ratio_dbe
        + first.rows_joined;
    assert_eq!(first.hydrogen_checks, counted, "checks partition under limits");
}

// ---------------------------------------------------------------------------
// H3 part 2: the host twin of the device enumeration order.
// ---------------------------------------------------------------------------

/// Device rows sorted by (mass, composition) for set comparison.
fn device_rows_sorted(found: &mamba3::models::ms2::formula_enum::DeviceEnumResult) -> Vec<(u32, Composition, bool)> {
    let mut rows: Vec<(u32, Composition, bool)> = found
        .masses
        .iter()
        .zip(found.compositions.iter())
        .zip(found.flags.iter())
        .map(|((m, c), f)| (*m, *c, *f == 2))
        .collect();
    rows.sort_by_key(|a| (a.0, a.1));
    rows
}

/// Uncapped device limits for the equivalence checks.
fn device_open() -> DeviceEnumLimits {
    DeviceEnumLimits {
        lane_visits_max: u32::MAX,
        scored_cap: u32::MAX,
    }
}

#[test]
fn rare_table_is_lexicographic_and_bounded() {
    let domain = rare_tiny_domain();
    let subset = extended_subset();
    let bounds = RatioBounds::fit(subset.iter().copied(), 0).unwrap();
    let rows = rare_table(&domain, &bounds).unwrap();
    assert!(!rows.is_empty(), "non-empty rare table");
    assert!(rows.len() <= RARE_TABLE_MAX);
    // Combination 0 is all zeros when the ranges allow it.
    assert_eq!(&rows[0][..6], &[0, 0, 0, 0, 0, 0]);
    // Increasing lexicographic order, no duplicates.
    let keys: Vec<[u32; 6]> = rows
        .iter()
        .map(|r| [r[0], r[1], r[2], r[3], r[4], r[5]])
        .collect();
    let mut sorted = keys.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(keys, sorted, "lexicographic without duplicates");
    // Row contents: 6 counts, their mass, their count sum.
    for r in &rows {
        let mass = r[0] * BRUTE_MASS[4]
            + r[1] * BRUTE_MASS[5]
            + r[2] * BRUTE_MASS[6]
            + r[3] * BRUTE_MASS[7]
            + r[4] * BRUTE_MASS[8]
            + r[5] * BRUTE_MASS[9];
        assert_eq!(r[6], mass, "row mass");
        assert_eq!(r[7], r[0] + r[1] + r[2] + r[3] + r[4] + r[5], "row sum");
        assert!(r[0] <= u32::from(domain.heavy_caps[3]));
        assert!(r[1] <= u32::from(domain.heavy_caps[4]));
        assert!(r[2] <= u32::from(domain.heavy_caps[5]));
    }
    // The subset maxima bound the table: F at most 1, S at most 1.
    assert!(rows.iter().all(|r| r[0] <= 1 && r[2] <= 1));
    // More than 16,384 combinations is `Error::Config`, in both entries.
    let big = EnumDomain {
        version: "test-big".to_string(),
        heavy_caps: [0, 0, 0, 8, 8, 8, 8, 8, 8],
        heavy_max: 48,
        hydrogen_min: 0,
        hydrogen_max: 8,
    };
    let mut wide = bounds.clone();
    wide.rare_total = [0, 100];
    wide.rare_distinct = [0, 6];
    assert!(rare_table(&big, &wide).is_err());
    assert!(validate_device_artifacts(&big, &wide).is_err());
    // The ordinary artifacts validate.
    assert!(validate_device_artifacts(&domain, &bounds).is_ok());
}

/// Reference limits for the device-order equivalence checks: the exact
/// chemical filters on (as the device applies them unconditionally) with the
/// fitted ratio bounds.
fn filtered_ratio_limits(bounds: RatioBounds) -> EnumLimits {
    EnumLimits {
        ratio: Some(bounds),
        ..EnumLimits::default()
    }
}

#[test]
fn device_order_matches_enumerate_on_tiny_domains() {
    // (a) the joined SET equals `enumerate` with the same bounds whenever
    // neither is exhausted, on the tiny brute-force domains, both adducts;
    // flags equal.
    let domain = tiny_domain();
    let subset: Vec<Composition> = tiny_compositions()
        .into_iter()
        .filter(|c| c[0] <= 1 && c[1] <= 4)
        .collect();
    let bounds = RatioBounds::fit(subset.iter().copied(), 0).unwrap();
    assert!(validate_device_artifacts(&domain, &bounds).is_ok());
    let bases: Vec<Composition> = [
        [1, 4, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 6, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 2, 1, 1, 0, 0, 0, 0, 0, 0],
        [2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    .to_vec();
    for adduct in [1u16, 2] {
        for ppm in [100u32, 1000] {
            for base in &bases {
                let anchor = if adduct == 1 {
                    protonated(base)
                } else {
                    deprotonated(base)
                };
                for delta in [-1000i64, 0, 1000] {
                    let precursor = (anchor as i64 + delta) as u32;
                    let query = EnumQuery {
                        precursor_mz: precursor,
                        adduct,
                        ppm_tenths: ppm,
                        precursor_uncertainty: 500,
                    };
                    let expected = enumerate(&domain, &query, &filtered_ratio_limits(bounds.clone())).unwrap();
                    assert!(!expected.exhausted);
                    let found =
                        enumerate_device_order(&domain, &bounds, &query, &device_open()).unwrap();
                    assert!(
                        !found.exhausted,
                        "adduct {adduct} ppm {ppm} delta {delta}: device completes"
                    );
                    assert_eq!(
                        device_rows_sorted(&found),
                        enum_rows(&domain, &query, &filtered_ratio_limits(bounds.clone())),
                        "adduct {adduct} ppm {ppm} delta {delta}: same joined set and flags"
                    );
                    assert_eq!(found.joined as usize, expected.rows_joined as usize);
                }
            }
        }
    }
    // The same on the rare-tiny domain (several lanes), margins 0 and 2.
    let domain = rare_tiny_domain();
    let subset = extended_subset();
    let gold: Composition = [2, 2, 0, 2, 0, 0, 0, 0, 0, 0];
    for margin in [0u16, 2] {
        let bounds = RatioBounds::fit(subset.iter().copied(), margin).unwrap();
        assert!(validate_device_artifacts(&domain, &bounds).is_ok());
        for query in [
            centred_query(&gold, 1000, 500_000),
            centred_query(&[1, 3, 0, 0, 1, 0, 0, 0, 0, 0], 1000, 500_000),
        ] {
            let expected = enumerate(&domain, &query, &filtered_ratio_limits(bounds.clone())).unwrap();
            assert!(!expected.exhausted);
            let found =
                enumerate_device_order(&domain, &bounds, &query, &device_open()).unwrap();
            assert!(!found.exhausted, "margin {margin}: device completes");
            assert_eq!(
                device_rows_sorted(&found),
                enum_rows(&domain, &query, &filtered_ratio_limits(bounds.clone())),
                "margin {margin}: same joined set and flags"
            );
            // Per-lane counters reconcile with the spectra-wide counters.
            let mut joined: u32 = 0;
            let mut visited: u32 = 0;
            for lane in &found.lanes {
                joined = sat_add_counter(joined, lane.joined, u32::MAX);
                visited = sat_add_counter(visited, lane.visited, u32::MAX);
            }
            assert_eq!((joined, visited), (found.joined, found.visited));
            assert_eq!(found.lanes.len(), rare_table(&domain, &bounds).unwrap().len());
        }
    }
}

#[test]
fn device_order_matches_enumerate_on_fixture_sweep() {
    // (a) second half: a sweep of fixture-derived queries, both adducts.
    let mut comps = fixture_compositions();
    comps.sort_by_key(brute_mass);
    let subset: Vec<Composition> = comps.iter().take(6).copied().collect();
    let domain = EnumDomain::from_compositions(subset.iter().copied(), 0).unwrap();
    let bounds = RatioBounds::fit(subset.iter().copied(), 0).unwrap();
    assert!(validate_device_artifacts(&domain, &bounds).is_ok());
    let mut compared = 0usize;
    let mut gold_joined = 0usize;
    for gold in &subset {
        for adduct in [1u16, 2] {
            let precursor = if adduct == 1 {
                protonated(gold)
            } else {
                deprotonated(gold)
            };
            let query = EnumQuery {
                precursor_mz: precursor,
                adduct,
                ppm_tenths: 200,
                precursor_uncertainty: 50,
            };
            let expected = enumerate(&domain, &query, &filtered_ratio_limits(bounds.clone())).unwrap();
            let found =
                enumerate_device_order(&domain, &bounds, &query, &device_open()).unwrap();
            if expected.exhausted || found.exhausted {
                continue;
            }
            assert_eq!(
                device_rows_sorted(&found),
                enum_rows(&domain, &query, &filtered_ratio_limits(bounds.clone())),
                "same joined set and flags"
            );
            compared += 1;
            if expected.scored_contains(gold) {
                assert!(
                    found.scored_contains(gold),
                    "gold joins the device order too"
                );
                gold_joined += 1;
            }
        }
    }
    assert!(compared > 0, "non-vacuous sweep");
    assert!(gold_joined > 0, "some gold joins");
}

#[test]
fn device_rank_order_is_lexicographic() {
    // (b) rank order is exactly lexicographic `(r, C, N, O, H)`.
    let domain = rare_tiny_domain();
    let subset = extended_subset();
    let bounds = RatioBounds::fit(subset.iter().copied(), 0).unwrap();
    let table = rare_table(&domain, &bounds).unwrap();
    // A wide-uncertainty query joins several rows across lanes (S1H0 in the S lane
    // against O2H0 and C1O1H4 in lane 0).
    let center: Composition = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0];
    let query = centred_query(&center, 1000, 100_000);
    let found = enumerate_device_order(&domain, &bounds, &query, &device_open()).unwrap();
    assert!(!found.exhausted);
    assert!(found.compositions.len() >= 2, "several rows join");
    let lane_of = |c: &Composition| {
        table
            .iter()
            .position(|r| {
                r[0] == u32::from(c[4])
                    && r[1] == u32::from(c[5])
                    && r[2] == u32::from(c[6])
                    && r[3] == u32::from(c[7])
                    && r[4] == u32::from(c[8])
                    && r[5] == u32::from(c[9])
            })
            .expect("scored composition comes from a lane")
    };
    let rank_keys: Vec<(usize, u16, u16, u16, u16)> = found
        .compositions
        .iter()
        .map(|c| (lane_of(c), c[0], c[2], c[3], c[1]))
        .collect();
    let mut sorted = rank_keys.clone();
    sorted.sort();
    assert_eq!(rank_keys, sorted, "rank order is (r, C, N, O, H)");
    sorted.dedup();
    assert_eq!(sorted.len(), rank_keys.len(), "no duplicate ranks");
}

#[test]
fn device_scored_cap_keeps_prefix() {
    // (c) a capped run's scored support is the prefix of the uncapped run.
    let domain = rare_tiny_domain();
    let subset = extended_subset();
    let bounds = RatioBounds::fit(subset.iter().copied(), 0).unwrap();
    let center: Composition = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0];
    let query = centred_query(&center, 1000, 100_000);
    let full = enumerate_device_order(&domain, &bounds, &query, &device_open()).unwrap();
    assert!(!full.exhausted);
    assert!(full.compositions.len() >= 2);
    for cap in [0u32, 1, 2] {
        let capped = enumerate_device_order(
            &domain,
            &bounds,
            &query,
            &DeviceEnumLimits {
                lane_visits_max: u32::MAX,
                scored_cap: cap,
            },
        )
        .unwrap();
        let keep = (full.compositions.len() as u32).min(cap) as usize;
        assert_eq!(&capped.compositions[..], &full.compositions[..keep]);
        assert_eq!(&capped.masses[..], &full.masses[..keep]);
        assert_eq!(&capped.flags[..], &full.flags[..keep]);
        assert_eq!(capped.scored, keep as u32);
        assert_eq!(capped.joined, full.joined);
        assert_eq!(
            capped.exhausted,
            (full.joined as u64) > (cap as u64),
            "cap {cap}: exhaustion exactly on truncation"
        );
    }
}

#[test]
fn device_wide_window_joins_nothing() {
    // (e) the `half` restriction: a window half above 1,511,737 joins
    // nothing and is exhausted; exactly at the bound the search runs.
    let domain = tiny_domain();
    let subset: Vec<Composition> = tiny_compositions()
        .into_iter()
        .filter(|c| c[0] <= 1 && c[1] <= 4)
        .collect();
    let bounds = RatioBounds::fit(subset.iter().copied(), 0).unwrap();
    // `E_domain` is 1 here (584 nano-dalton); with zero ppm the half is
    // `uncertainty + 1 + 1`.
    assert_eq!(domain.max_error(), 1);
    let ch4: Composition = [1, 4, 0, 0, 0, 0, 0, 0, 0, 0];
    assert_eq!(brute_parent(protonated(&ch4), 1), brute_mass(&ch4));
    let at_bound = EnumQuery {
        precursor_mz: protonated(&ch4),
        adduct: 1,
        ppm_tenths: 0,
        precursor_uncertainty: DEVICE_HALF_MAX - 2,
    };
    let past_bound = EnumQuery {
        precursor_uncertainty: DEVICE_HALF_MAX - 1,
        ..at_bound
    };
    let searched =
        enumerate_device_order(&domain, &bounds, &at_bound, &device_open()).unwrap();
    assert!(
        !searched.exhausted,
        "half == 1,511,737 still searches (it may still join nothing)"
    );
    assert!(searched.complete);
    let wide = enumerate_device_order(&domain, &bounds, &past_bound, &device_open()).unwrap();
    assert!(wide.exhausted, "a wider window is exhausted");
    assert_eq!(wide.joined, 0, "a wider window joins nothing");
    assert_eq!(wide.scored, 0);
    assert!(!wide.absent);
    assert!(!wide.complete);
    assert!(wide.lanes.iter().all(|l| l.visited == 0 && l.joined == 0));
}

#[test]
fn device_u32_helpers_at_boundaries() {
    // (f) unit half: the `u32`-only primitives at their overflow guards.
    assert_eq!(sat_add_counter(u32::MAX - 1, 1, u32::MAX), u32::MAX - 1);
    assert_eq!(sat_add_counter(u32::MAX - 1, 0, u32::MAX), u32::MAX - 1);
    assert_eq!(
        sat_add_counter(u32::MAX - 1, u32::MAX, u32::MAX),
        u32::MAX - 1
    );
    assert_eq!(sat_add_counter(3, 4, u32::MAX), 7);
    assert_eq!(sat_add_saturates(3, 4, u32::MAX), 0);
    assert_eq!(sat_add_saturates(0, u32::MAX - 1, u32::MAX), 0);
    assert_eq!(sat_add_saturates(1, u32::MAX - 1, u32::MAX), 1);
    // Verdict flags: 1 accept, 0 reject, 2 ambiguous; the unrepresentable
    // `tolerance + error` arm never rejects (ambiguous instead).
    assert_eq!(decide_u32(100, 100, 5, 5), 1);
    assert_eq!(decide_u32(100, 200, 5, 5), 0);
    assert_eq!(decide_u32(100, 106, 5, 5), 2);
    assert_eq!(decide_u32(100, 110, u32::MAX, 5), 2);
    assert_eq!(decide_u32(100, 100, u32::MAX, u32::MAX), 1);
    // DBE bias: the packed words of a real fit are the fitted endpoints
    // plus 2^31 (independently biased here).
    let domain = rare_tiny_domain();
    let subset = extended_subset();
    let bounds = RatioBounds::fit(subset.iter().copied(), 0).unwrap();
    let packed = pack_device_bounds(&domain, &bounds).unwrap();
    let carbon_rows = bounds.max_by_carbon.len();
    let heavy_rows = bounds.max_by_heavy.len();
    let dbe = PACK_HEADER_LEN as usize + carbon_rows * 10 + heavy_rows * 10;
    for (b, range) in bounds.dbe_by_heavy.iter().enumerate() {
        for (w, v) in [range[0], range[1]].iter().enumerate() {
            let expected = (v + (1i64 << 31)).clamp(0, i64::from(u32::MAX)) as u32;
            assert_eq!(packed[dbe + b * 2 + w], expected);
        }
    }
}

#[test]
fn device_u32_boundary_caps_do_not_overflow() {
    // (f) system half: the largest validated caps with a precursor near
    // 2000 Da and near `u32::MAX` run without overflow (every lane product
    // is division-guarded, every sum checked/saturating).
    let domain = EnumDomain {
        version: "test-max".to_string(),
        heavy_caps: [255, 255, 255, 255, 255, 255, 255, 255, 255],
        heavy_max: u16::MAX,
        hydrogen_min: 0,
        hydrogen_max: 1023,
    };
    let one: Composition = [1, 4, 0, 0, 0, 0, 0, 0, 0, 0];
    let bounds = RatioBounds::fit([one], 0).unwrap();
    assert!(validate_device_artifacts(&domain, &bounds).is_ok());
    let limits = DeviceEnumLimits {
        lane_visits_max: 5000,
        scored_cap: 2048,
    };
    // Near 2000 Da: the mass guards engage (255 carbons alone exceed `hi`).
    let query = EnumQuery {
        precursor_mz: 2_000_000_000 + 1_007_825 - 549,
        adduct: 1,
        ppm_tenths: 200,
        precursor_uncertainty: 1000,
    };
    let first = enumerate_device_order(&domain, &bounds, &query, &limits).unwrap();
    let second = enumerate_device_order(&domain, &bounds, &query, &limits).unwrap();
    assert_eq!(first, second, "deterministic at the mass guards");
    assert!(first.visited > 0, "lanes run into the mass guards");
    // Near `u32::MAX`: `hi` saturates while the guards still hold.
    let top = EnumQuery {
        precursor_mz: u32::MAX - 5_000,
        adduct: 1,
        ppm_tenths: 200,
        precursor_uncertainty: 1000,
    };
    let found = enumerate_device_order(&domain, &bounds, &top, &limits).unwrap();
    let repeat = enumerate_device_order(&domain, &bounds, &top, &limits).unwrap();
    assert_eq!(found, repeat, "deterministic at saturating `hi`");
    assert!(found.visited > 0);
}

#[test]
fn validate_device_artifacts_rejects_each_bound() {
    // (g) every artifact bound the spec lists is refused.
    let domain = tiny_domain();
    let subset: Vec<Composition> = tiny_compositions()
        .into_iter()
        .filter(|c| c[0] <= 1 && c[1] <= 4)
        .collect();
    let bounds = RatioBounds::fit(subset.iter().copied(), 0).unwrap();
    assert!(validate_device_artifacts(&domain, &bounds).is_ok());
    // Element cap above 255.
    let mut bad = domain.clone();
    bad.heavy_caps[0] = 256;
    assert!(validate_device_artifacts(&bad, &bounds).is_err());
    // Hydrogen bounds above 1023.
    let mut bad = domain.clone();
    bad.hydrogen_max = 1024;
    assert!(validate_device_artifacts(&bad, &bounds).is_err());
    let mut bad = domain;
    bad.hydrogen_min = 2000;
    assert!(validate_device_artifacts(&bad, &bounds).is_err());
    // Ratio numerators and denominators above 2^20.
    for slot in 0..4 {
        let mut bad = bounds.clone();
        match slot {
            0 => bad.ratio_lo_num[0] = (1 << 20) + 1,
            1 => bad.ratio_lo_den[0] = (1 << 20) + 1,
            2 => bad.ratio_hi_num[0] = (1 << 20) + 1,
            _ => bad.ratio_hi_den[0] = (1 << 20) + 1,
        }
        assert!(
            validate_device_artifacts(&tiny_domain(), &bad).is_err(),
            "ratio slot {slot} is refused"
        );
    }
    // Bucket tables above 64 rows.
    let mut bad = bounds.clone();
    bad.max_by_carbon = vec![[0; 9]; 65];
    bad.carbon_seen = vec![true; 65];
    assert!(validate_device_artifacts(&tiny_domain(), &bad).is_err());
    let mut bad = bounds.clone();
    bad.max_by_heavy = vec![[0; 9]; 65];
    bad.heavy_seen = vec![true; 65];
    assert!(validate_device_artifacts(&tiny_domain(), &bad).is_err());
    let mut bad = bounds.clone();
    bad.dbe_by_heavy = vec![[0; 2]; 65];
    assert!(validate_device_artifacts(&tiny_domain(), &bad).is_err());
    // More than 16,384 rare combinations.
    let big = EnumDomain {
        version: "test-big".to_string(),
        heavy_caps: [0, 0, 0, 8, 8, 8, 8, 8, 8],
        heavy_max: 48,
        hydrogen_min: 0,
        hydrogen_max: 8,
    };
    let mut wide = bounds.clone();
    wide.rare_total = [0, 100];
    wide.rare_distinct = [0, 6];
    assert!(validate_device_artifacts(&big, &wide).is_err());
    // `enumerate_device_order` refuses invalid artifacts instead of running.
    assert!(
        enumerate_device_order(&big, &wide, &centred_query(&[1, 4, 0, 0, 0, 0, 0, 0, 0, 0], 200, 50), &device_open())
            .is_err()
    );
}

// ---------------------------------------------------------------------------
// H5 finding 1: the packed artifact and the kernel twins.
// ---------------------------------------------------------------------------

use mamba3::models::ms2::chem::tolerance_u32;

/// Per-spectrum meta words for the kernel twins, built from independent
/// values (the test computes parent/tolerance/window with its own helpers).
/// The fourteen chemistry scalars the lane twins take: `[m_C, m_N, m_O,
/// m_H, res_C, res_H, res_N, res_O, res_F, res_P, res_S, res_Cl, res_Br,
/// res_I]`, read off the chemistry table (test literals live in
/// `BRUTE_MASS`/`BRUTE_RES`; these are the production values the kernels
/// receive as launch scalars).
fn twin_chem() -> [u32; 14] {
    [
        ELEMENTS[0].mass,
        ELEMENTS[2].mass,
        ELEMENTS[3].mass,
        ELEMENTS[1].mass,
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

fn test_meta(
    parent: u32,
    tol: u32,
    bound: u32,
    lo: u32,
    hi: u32,
    budget: u32,
    cap: u32,
) -> [u32; 8] {
    [parent, tol, bound, lo, hi, budget, cap, 0]
}

/// Decode `n` 13-word lane records into `(mass, composition, flag)`.
fn decode_records(out: &[u32], n: u32) -> Vec<(u32, Composition, u8)> {
    let mut rows = Vec::new();
    for s in 0..n {
        let base = (s as usize) * 13;
        let comp: Composition = [
            out[base] as u16,
            out[base + 1] as u16,
            out[base + 2] as u16,
            out[base + 3] as u16,
            out[base + 4] as u16,
            out[base + 5] as u16,
            out[base + 6] as u16,
            out[base + 7] as u16,
            out[base + 8] as u16,
            out[base + 9] as u16,
        ];
        rows.push((out[base + 10], comp, out[base + 11] as u8));
    }
    rows
}

#[test]
fn packed_bounds_layout_matches_inputs() {
    // The packed buffer carries the documented fixed header plus the
    // variable section; the DBE words are the biased endpoints.
    let domain = rare_tiny_domain();
    let subset = extended_subset();
    let bounds = RatioBounds::fit(subset.iter().copied(), 0).unwrap();
    let packed = pack_device_bounds(&domain, &bounds).unwrap();
    let carbon_rows = bounds.max_by_carbon.len() as u32;
    let heavy_rows = bounds.max_by_heavy.len() as u32;
    assert_eq!(packed[PACK_N_CARBON_ROWS as usize], carbon_rows);
    assert_eq!(packed[PACK_N_HEAVY_ROWS as usize], heavy_rows);
    assert_eq!(
        packed[PACK_CARBON_WIDTH as usize],
        u32::from(bounds.carbon_bucket)
    );
    assert_eq!(
        packed[PACK_HEAVY_WIDTH as usize],
        u32::from(bounds.heavy_bucket)
    );
    for (i, cap) in domain.heavy_caps.iter().enumerate() {
        assert_eq!(packed[PACK_HEAVY_CAPS as usize + i], u32::from(*cap));
    }
    assert_eq!(
        packed[PACK_HEAVY_MAX as usize],
        u32::from(domain.heavy_max)
    );
    assert_eq!(
        packed[PACK_HYDROGEN_MIN as usize],
        u32::from(domain.hydrogen_min)
    );
    assert_eq!(
        packed[PACK_HYDROGEN_MAX as usize],
        u32::from(domain.hydrogen_max)
    );
    for k in 0..6 {
        assert_eq!(
            packed[PACK_RATIO_LO_NUM as usize + k],
            bounds.ratio_lo_num[k]
        );
        assert_eq!(
            packed[PACK_RATIO_LO_DEN as usize + k],
            bounds.ratio_lo_den[k]
        );
        assert_eq!(
            packed[PACK_RATIO_HI_NUM as usize + k],
            bounds.ratio_hi_num[k]
        );
        assert_eq!(
            packed[PACK_RATIO_HI_DEN as usize + k],
            bounds.ratio_hi_den[k]
        );
    }
    assert_eq!(
        packed[PACK_ZERO_CARBON as usize],
        u32::from(bounds.zero_carbon_seen)
    );
    assert_eq!(packed[PACK_RARE_TOTAL_LO as usize], bounds.rare_total[0]);
    assert_eq!(packed[PACK_RARE_TOTAL_HI as usize], bounds.rare_total[1]);
    // Variable section lengths: carbon caps + presence, heavy caps +
    // presence, biased DBE pairs.
    let carbon_rows_n = carbon_rows as usize;
    let heavy_rows_n = heavy_rows as usize;
    assert_eq!(
        packed.len(),
        PACK_HEADER_LEN as usize + carbon_rows_n * 10 + heavy_rows_n * 12
    );
    let caps_c = PACK_HEADER_LEN as usize;
    let seen_c = caps_c + carbon_rows_n * 9;
    let caps_h = seen_c + carbon_rows_n;
    let seen_h = caps_h + heavy_rows_n * 9;
    let dbe = seen_h + heavy_rows_n;
    for (b, row) in bounds.max_by_carbon.iter().enumerate() {
        for (i, cap) in row.iter().enumerate() {
            assert_eq!(packed[caps_c + b * 9 + i], u32::from(*cap));
        }
        assert_eq!(packed[seen_c + b], u32::from(bounds.carbon_seen[b]));
    }
    for (b, row) in bounds.max_by_heavy.iter().enumerate() {
        for (i, cap) in row.iter().enumerate() {
            assert_eq!(packed[caps_h + b * 9 + i], u32::from(*cap));
        }
        assert_eq!(packed[seen_h + b], u32::from(bounds.heavy_seen[b]));
    }
    // The DBE words are the endpoints plus 2^31 (independently biased here
    // with the test's own arithmetic).
    for (b, range) in bounds.dbe_by_heavy.iter().enumerate() {
        for (w, v) in [range[0], range[1]].iter().enumerate() {
            let expected =
                (v + (1i64 << 31)).clamp(0, i64::from(u32::MAX)) as u32;
            assert_eq!(packed[dbe + b * 2 + w], expected, "bucket {b} word {w}");
        }
    }
    // The bias constant is exactly 2^31.
    assert_eq!(PACK_DBE_BIAS, 1 << 31);
    // Packing rejects invalid artifacts instead of writing them.
    let mut bad = bounds.clone();
    bad.max_by_carbon[0][0] = 256;
    assert!(pack_device_bounds(&domain, &bad).is_err());
}

#[test]
fn kernel_count_fill_match_wrapper() {
    // Count and fill are the same lane function with a mode flag: driving
    // the twins directly (count per lane, offsets, fill at the clamped
    // offsets) reproduces `enumerate_device_order` record for record.
    let domain = rare_tiny_domain();
    let subset = extended_subset();
    let bounds = RatioBounds::fit(subset.iter().copied(), 0).unwrap();
    let packed = pack_device_bounds(&domain, &bounds).unwrap();
    let rare = rare_table(&domain, &bounds).unwrap();
    let gold: Composition = [2, 2, 0, 2, 0, 0, 0, 0, 0, 0];
    let query = centred_query(&gold, 1000, 500_000);
    let parent = brute_parent(query.precursor_mz, query.adduct);
    let tol = brute_tolerance(query.precursor_mz, query.ppm_tenths);
    let bound = 500_001u32;
    let e_domain = domain.max_error();
    let half = tol.saturating_add(bound).saturating_add(e_domain);
    let lo = parent.saturating_sub(half);
    let hi = parent.saturating_add(half);
    let limits = DeviceEnumLimits {
        lane_visits_max: 1000,
        scored_cap: 64,
    };
    let meta_all = test_meta(
        parent,
        tol,
        bound,
        lo,
        hi,
        limits.lane_visits_max,
        limits.scored_cap,
    );
    let chem = twin_chem();
    let packed_len = packed.len() as u32;
    let rare_all: Vec<u32> = rare.iter().flat_map(|row| row.iter().copied()).collect();
    let rare_len = rare_all.len() as u32;
    // Count pass through the twin.
    let mut stats_flat = vec![0u32; rare.len() * 2];
    let stats_len = stats_flat.len() as u32;
    for r in 0..rare.len() {
        let rbase = r as u32 * 8u32;
        let stats_base = r as u32 * 2u32;
        kernel_lane(
            &meta_all, 8u32, &packed, packed_len, &rare_all, rare_len, 0u32, rbase,
            LANE_MODE_COUNT, 0u32, 0u32, &mut stats_flat, stats_len, 0u32, stats_base,
            chem[0], chem[1], chem[2], chem[3], chem[4], chem[5], chem[6], chem[7],
            chem[8], chem[9], chem[10], chem[11], chem[12], chem[13], u32::MAX,
            meta_all[5],
        );
    }
    // Offsets pass through the twin.
    let mut offsets = vec![0u32; rare.len()];
    let mut counters = vec![0u32; 5];
    kernel_offsets(
        &stats_flat,
        stats_flat.len() as u32,
        &meta_all,
        8u32,
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
    // Fill pass through the SAME function in mode 1 (records only; the
    // stats words are untouched in mode 1). The record-for-record comparison
    // with the wrapper below proves both modes stop at the same vector.
    let scored = counters[2];
    let mut out = vec![0u32; scored as usize * LANE_RECORD_WORDS as usize];
    let out_len = out.len() as u32;
    for (r, off) in offsets.iter().enumerate() {
        let rbase = r as u32 * 8u32;
        kernel_lane(
            &meta_all, 8u32, &packed, packed_len, &rare_all, rare_len, 0u32, rbase,
            LANE_MODE_FILL, *off, limits.scored_cap, &mut out, out_len, 0u32,
            0u32, chem[0], chem[1], chem[2], chem[3], chem[4], chem[5], chem[6],
            chem[7], chem[8], chem[9], chem[10], chem[11], chem[12], chem[13],
            u32::MAX, meta_all[5],
        );
    }
    // Mode 1 with cap 0 writes nothing: the probe record stays poisoned.
    let mut probe = vec![0xDEAD_BEEFu32; 13];
    let probe_len = probe.len() as u32;
    for r in 0..rare.len() {
        let rbase = r as u32 * 8u32;
        kernel_lane(
            &meta_all, 8u32, &packed, packed_len, &rare_all, rare_len, 0u32, rbase,
            LANE_MODE_FILL, 0u32, 0u32, &mut probe, probe_len, 0u32, 0u32, chem[0],
            chem[1], chem[2], chem[3], chem[4], chem[5], chem[6], chem[7], chem[8],
            chem[9], chem[10], chem[11], chem[12], chem[13], u32::MAX,
            meta_all[5],
        );
    }
    assert!(probe.iter().all(|w| *w == 0xDEAD_BEEF));
    // The wrapper agrees record for record.
    let found = enumerate_device_order(&domain, &bounds, &query, &limits).unwrap();
    assert_eq!(found.visited, counters[0]);
    assert_eq!(found.joined, counters[1]);
    assert_eq!(found.scored, scored);
    assert_eq!(found.status, counters[3]);
    assert_eq!(found.complete, counters[4] != 0);
    let rows = decode_records(&out, scored);
    assert_eq!(rows.len(), found.compositions.len());
    for (i, (mass, comp, flag)) in rows.iter().enumerate() {
        assert_eq!(*mass, found.masses[i], "record {i} mass");
        assert_eq!(*comp, found.compositions[i], "record {i} composition");
        assert_eq!(*flag, found.flags[i], "record {i} flag");
    }
    for (r, lane) in found.lanes.iter().enumerate() {
        assert_eq!(lane.joined, stats_flat[2 * r]);
        assert_eq!(lane.visited, stats_flat[2 * r + 1] & LANE_VISITED_MASK);
        assert_eq!(
            lane.exhausted,
            stats_flat[2 * r + 1] & LANE_EXHAUSTED_BIT != 0
        );
    }
    assert!(found.scored_contains(&gold));
}

#[test]
fn kernel_pad_writes_padding_records() {
    // Slots at or past `scored` and below `cap` become padding (zeros,
    // source `u32::MAX`); earlier slots are untouched. The twin covers one
    // slot per call; the test drives the `[1, 3)` range slot by slot.
    let mut out = vec![0xDEAD_BEEFu32; 3 * LANE_RECORD_WORDS as usize];
    let out_len = out.len() as u32;
    for slot in 1..3u32 {
        kernel_pad(&mut out, out_len, 0u32, slot, 1u32, 3u32, u32::MAX);
    }
    assert!(out[..LANE_RECORD_WORDS as usize].iter().all(|w| *w == 0xDEAD_BEEF));
    for slot in 1..3u32 {
        let base = slot as usize * LANE_RECORD_WORDS as usize;
        assert_eq!(&out[base..base + 12], &[0u32; 12]);
        assert_eq!(out[base + 12], u32::MAX);
    }
    // Empty ranges are a no-op: a slot below `scored`, or at or past `cap`.
    let mut out = vec![7u32; 13];
    kernel_pad(&mut out, 13u32, 0u32, 0u32, 1u32, 1u32, u32::MAX);
    kernel_pad(&mut out, 13u32, 0u32, 5u32, 2u32, 1u32, u32::MAX);
    kernel_pad(&mut out, 13u32, 0u32, 0u32, 0u32, 0u32, u32::MAX);
    assert!(out.iter().all(|w| *w == 7));
}

// ---------------------------------------------------------------------------
// H5 finding 2: counter saturation sets exhaustion and clears `complete`.
// ---------------------------------------------------------------------------

#[test]
fn counter_saturation_sets_exhaustion() {
    // Aggregation near the saturation threshold, driven directly through
    // the offsets twin on a synthetic `lane_stats` array (no billions of
    // vectors enumerated).
    let meta = test_meta(0, 0, 0, 0, 0, 0, u32::MAX);
    // The reviewer's case: visits saturate at `u32::MAX - 1` while no
    // candidates pass and no lane raises exhaustion. Saturation alone must
    // report exhausted (never absent), with `complete` cleared. Three lanes
    // at half-maximum sum exactly past the threshold.
    let stats = [0u32, 0x7FFF_FFFF, 0, 0x7FFF_FFFF, 0, 0x7FFF_FFFF];
    let mut offsets = [0u32; 3];
    let mut counters = [0u32; 5];
    kernel_offsets(
        &stats,
        6u32,
        &meta,
        8u32,
        &mut offsets,
        3u32,
        &mut counters,
        5u32,
        0u32,
        3u32,
        u32::MAX,
        request_status::FORMULA_SEARCH_EXHAUSTED,
        request_status::FORMULA_ABSENT,
        u32::MAX,
    );
    assert_eq!(counters[0], u32::MAX - 1, "visited saturates");
    assert_eq!(counters[1], 0);
    assert_eq!(counters[2], 0);
    assert_eq!(
        counters[3], request_status::FORMULA_SEARCH_EXHAUSTED,
        "saturation sets exhaustion and nothing else"
    );
    assert_eq!(counters[4], 0, "saturation clears complete");
    assert_eq!(offsets, [0, 0, 0]);
    // Joined saturation likewise exhausts, with the joined total clamped.
    let stats = [0x7FFF_FFFFu32, 0, 0x7FFF_FFFF, 0, 0x7FFF_FFFF, 0];
    kernel_offsets(
        &stats,
        6u32,
        &meta,
        8u32,
        &mut offsets,
        3u32,
        &mut counters,
        5u32,
        0u32,
        3u32,
        u32::MAX,
        request_status::FORMULA_SEARCH_EXHAUSTED,
        request_status::FORMULA_ABSENT,
        u32::MAX,
    );
    assert_eq!(counters[1], u32::MAX - 1, "joined saturates");
    assert_eq!(counters[3], request_status::FORMULA_SEARCH_EXHAUSTED);
    assert_eq!(counters[4], 0);
    assert_eq!(offsets, [0, 0x7FFF_FFFF, 0xFFFF_FFFE]);
    // Ordinary truncation still exhausts with the scored prefix.
    let meta = test_meta(0, 0, 0, 0, 0, 0, 4);
    let stats = [3u32, 5, 2, 7];
    let mut offsets = [0u32; 2];
    let mut counters = [0u32; 5];
    kernel_offsets(
        &stats,
        4u32,
        &meta,
        8u32,
        &mut offsets,
        2u32,
        &mut counters,
        5u32,
        0u32,
        2u32,
        4u32,
        request_status::FORMULA_SEARCH_EXHAUSTED,
        request_status::FORMULA_ABSENT,
        u32::MAX,
    );
    assert_eq!(offsets, [0, 3]);
    assert_eq!(counters[0], 12);
    assert_eq!(counters[1], 5);
    assert_eq!(counters[2], 4);
    assert_eq!(counters[3], request_status::FORMULA_SEARCH_EXHAUSTED);
    assert_eq!(counters[4], 0);
    // An exact fit completes with no status bits.
    let meta = test_meta(0, 0, 0, 0, 0, 0, 5);
    kernel_offsets(
        &stats,
        4u32,
        &meta,
        8u32,
        &mut offsets,
        2u32,
        &mut counters,
        5u32,
        0u32,
        2u32,
        5u32,
        request_status::FORMULA_SEARCH_EXHAUSTED,
        request_status::FORMULA_ABSENT,
        u32::MAX,
    );
    assert_eq!(counters[2], 5);
    assert_eq!(counters[3], 0);
    assert_eq!(counters[4], 1);
    // The saturating step itself pins the threshold.
    assert_eq!(sat_add_counter(u32::MAX - 1, 0, u32::MAX), u32::MAX - 1);
    assert_eq!(sat_add_counter(u32::MAX - 1, 1, u32::MAX), u32::MAX - 1);
    assert_eq!(sat_add_saturates(u32::MAX - 2, 0, u32::MAX), 0);
    assert_eq!(sat_add_saturates(0x7FFF_FFFF, 0x7FFF_FFFF, u32::MAX), 0);
    assert_eq!(sat_add_saturates(u32::MAX - 1, 1, u32::MAX), 1);
    assert_eq!(sat_add_saturates(0x7FFF_FFFF, 0x8000_0000, u32::MAX), 1);
}

// ---------------------------------------------------------------------------
// H5 finding 3: the oracle applies the margin to the default fractions.
// ---------------------------------------------------------------------------

#[test]
fn oracle_margin_with_no_positive_carbon() {
    // The reviewer's counterexample: train `[H2O]`, margin 2, candidate
    // `CH2O`. Production fits upper ratio numerators to 2 over denominator
    // 1 and admits the candidate; the oracle must agree.
    let h2o: Composition = [0, 2, 0, 1, 0, 0, 0, 0, 0, 0];
    let ch2o: Composition = [1, 2, 0, 1, 0, 0, 0, 0, 0, 0];
    let subset = [h2o];
    let bounds = RatioBounds::fit(subset.iter().copied(), 2).unwrap();
    assert!(
        bounds.passes_ratio(0, &ch2o),
        "production admits CH2O at margin 2"
    );
    assert!(
        production_ratio_ok(&bounds, &ch2o),
        "production admits CH2O at every stage"
    );
    assert!(
        brute_ratio_ok_margin(&subset, &ch2o, 2),
        "the oracle admits CH2O at margin 2"
    );
    assert_eq!(
        brute_ratio_ok_margin(&subset, &ch2o, 2),
        production_ratio_ok(&bounds, &ch2o),
        "oracle == production"
    );
    // A numerator above `margin * carbon` is still rejected by both.
    let ch4: Composition = [1, 4, 0, 0, 0, 0, 0, 0, 0, 0];
    assert!(!bounds.passes_ratio(0, &ch4), "H/C 4/1 above 2/1");
    assert!(!brute_ratio_ok_margin(&subset, &ch4, 2));
    assert_eq!(
        brute_ratio_ok_margin(&subset, &ch4, 2),
        production_ratio_ok(&bounds, &ch4),
        "oracle == production on the reject"
    );
    // At margin 0 the default `(0, 1)` bounds admit only zero numerators.
    let bounds = RatioBounds::fit(subset.iter().copied(), 0).unwrap();
    assert!(!bounds.passes_ratio(0, &ch2o));
    assert!(!brute_ratio_ok_margin(&subset, &ch2o, 0));
    assert_eq!(
        brute_ratio_ok_margin(&subset, &ch2o, 0),
        production_ratio_ok(&bounds, &ch2o),
        "oracle == production at margin 0"
    );
}

// ---------------------------------------------------------------------------
// H5 finding 4: query validation and a tolerance that never narrows.
// ---------------------------------------------------------------------------

#[test]
fn ppm_above_1000_is_config() {
    // The reviewer's input: domain/bounds fitted from `CH4`, precursor
    // `u32::MAX`, adduct 1, `ppm_tenths = 10_000_001`. The true tolerance
    // (4,294,967,724) does not fit `u32` after the division; both entries
    // refuse the query instead of searching a wrapped window.
    let ch4: Composition = [1, 4, 0, 0, 0, 0, 0, 0, 0, 0];
    let domain = EnumDomain::from_compositions([ch4], 0).unwrap();
    let bounds = RatioBounds::fit([ch4], 0).unwrap();
    let query = EnumQuery {
        precursor_mz: u32::MAX,
        adduct: 1,
        ppm_tenths: 10_000_001,
        precursor_uncertainty: 0,
    };
    assert!(enumerate(&domain, &query, &EnumLimits::unfiltered()).is_err());
    assert!(enumerate_device_order(&domain, &bounds, &query, &device_open()).is_err());
    // The boundary: 1000 runs, 1001 is refused.
    let ok = EnumQuery {
        precursor_mz: protonated(&ch4),
        adduct: 1,
        ppm_tenths: 1000,
        precursor_uncertainty: 0,
    };
    assert!(enumerate(&domain, &ok, &EnumLimits::unfiltered()).is_ok());
    assert!(enumerate_device_order(&domain, &bounds, &ok, &device_open()).is_ok());
    let bad = EnumQuery { ppm_tenths: 1001, ..ok };
    assert!(enumerate(&domain, &bad, &EnumLimits::unfiltered()).is_err());
    assert!(enumerate_device_order(&domain, &bounds, &bad, &device_open()).is_err());
    // Within the contract range the `u32`-only tolerance never narrows: it
    // equals the `u64` reference at the structural extremes.
    for mz in [0u32, 1, 9_999, 10_000, 1_007_825, u32::MAX - 1, u32::MAX] {
        for ppm in [0u32, 1, 200, 999, 1000] {
            assert_eq!(
                tolerance_u32(mz, ppm).unwrap(),
                brute_tolerance(mz, ppm),
                "mz {mz} ppm {ppm}"
            );
        }
        assert!(tolerance_u32(mz, 1001).is_err());
    }
}

// ---------------------------------------------------------------------------
// H5 finding 5: the division guard admits the exact-boundary count.
// ---------------------------------------------------------------------------

#[test]
fn division_guard_boundary() {
    // Carbon-only single lane where `(hi - rare) / m_C` is exactly 2: `C2`
    // is admitted (its product then fits) and joins as ambiguous, `C1` is
    // rejected by the verdict, `C3` never forms a product. The lane visits
    // exactly the `C2` vector.
    let domain = EnumDomain {
        version: "test-guard".to_string(),
        heavy_caps: [3, 0, 0, 0, 0, 0, 0, 0, 0],
        heavy_max: 3,
        hydrogen_min: 0,
        hydrogen_max: 0,
    };
    let c2: Composition = [2, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let bounds = RatioBounds::fit([c2], 0).unwrap();
    assert!(validate_device_artifacts(&domain, &bounds).is_ok());
    // Parent exactly the `C2` mass: `hi = parent + 1`, so the carbon bound
    // is `(24_000_001 - 0) / 12_000_000 = 2`.
    let parent = 24_000_000u32;
    let query = EnumQuery {
        precursor_mz: parent + 1_007_825 - 549,
        adduct: 1,
        ppm_tenths: 0,
        precursor_uncertainty: 0,
    };
    assert_eq!(brute_parent(query.precursor_mz, 1), parent);
    let found = enumerate_device_order(&domain, &bounds, &query, &device_open()).unwrap();
    assert!(!found.exhausted);
    assert_eq!(found.compositions, vec![c2]);
    assert_eq!(found.flags, vec![2], "r = 0 within tol + error: ambiguous");
    assert_eq!(found.lanes.len(), 1);
    assert_eq!(found.lanes[0].visited, 1, "only the C2 vector is visited");
    assert_eq!(found.lanes[0].joined, 1);
    let expected = enumerate(
        &domain,
        &query,
        &EnumLimits {
            ratio: Some(bounds.clone()),
            ..EnumLimits::default()
        },
    )
    .unwrap();
    assert!(!expected.exhausted);
    assert_eq!(device_rows_sorted(&found), enum_rows(&domain, &query, &EnumLimits {
        ratio: Some(bounds.clone()),
        ..EnumLimits::default()
    }));
    assert_eq!(expected.rows_joined, 1);
}

// ---------------------------------------------------------------------------
// H5 finding 6: the rare table holds exactly the mass-representable rows.
// ---------------------------------------------------------------------------

#[test]
fn rare_table_counts_mass_representable_rows() {
    // The reviewer's case: iodine cap 34 with rare ranges `[0, 34]` /
    // `[0, 1]`. Tuple `(0,0,0,0,0,34)` (mass 4,314,752,048) is allowed but
    // not mass-representable, so it is not a row; `P` counts the 34 rows.
    let domain = EnumDomain {
        version: "test-iodine".to_string(),
        heavy_caps: [0, 0, 0, 0, 0, 0, 0, 0, 34],
        heavy_max: 34,
        hydrogen_min: 0,
        hydrogen_max: 0,
    };
    let mut bounds = RatioBounds::fit([], 0).unwrap();
    bounds.rare_total = [0, 34];
    bounds.rare_distinct = [0, 1];
    assert!(validate_device_artifacts(&domain, &bounds).is_ok());
    let rows = rare_table(&domain, &bounds).unwrap();
    assert_eq!(rows.len(), 34, "P counts the mass-representable rows");
    assert!(rows.len() <= RARE_TABLE_MAX);
    for (i, row) in rows.iter().enumerate() {
        assert_eq!(&row[..5], &[0, 0, 0, 0, 0]);
        assert_eq!(row[5], i as u32, "iodine counts ascend");
        assert_eq!(row[6], row[5] * BRUTE_MASS[9], "row mass");
        assert_eq!(row[7], row[5], "row sum");
    }
    assert!(
        rows.iter().all(|row| row[5] <= 33),
        "iodine 34 (mass 4,314,752,048) is absent"
    );
    // Combination 0 is all zeros here because the ranges allow it ...
    assert_eq!(&rows[0][..6], &[0, 0, 0, 0, 0, 0]);
    // ... and excluded under positive rare minima.
    let mut bounds = bounds.clone();
    bounds.rare_total = [1, 34];
    let rows = rare_table(&domain, &bounds).unwrap();
    assert!(!rows.is_empty());
    assert!(rows.iter().all(|row| row[7] >= 1));
}

// ---------------------------------------------------------------------------
// H5 finding 7: artifact validation checks every cap and every alignment.
// ---------------------------------------------------------------------------

#[test]
fn validate_device_artifacts_checks_tables_and_caps() {
    let domain = tiny_domain();
    let subset: Vec<Composition> = tiny_compositions()
        .into_iter()
        .filter(|c| c[0] <= 1 && c[1] <= 4)
        .collect();
    let bounds = RatioBounds::fit(subset.iter().copied(), 0).unwrap();
    assert!(validate_device_artifacts(&domain, &bounds).is_ok());
    // Every cap of BOTH bucket tables is checked: each of the 9 positions of
    // the first row of each table is refused at 256.
    for i in 0..9 {
        let mut bad = bounds.clone();
        bad.max_by_carbon[0][i] = 256;
        assert!(
            validate_device_artifacts(&domain, &bad).is_err(),
            "carbon table cap {i}"
        );
        let mut bad = bounds.clone();
        bad.max_by_heavy[0][i] = 256;
        assert!(
            validate_device_artifacts(&domain, &bad).is_err(),
            "heavy table cap {i}"
        );
    }
    // A deeper row of each table is checked too.
    if bounds.max_by_carbon.len() > 1 {
        let mut bad = bounds.clone();
        let last = bad.max_by_carbon.len() - 1;
        bad.max_by_carbon[last][4] = 256;
        assert!(validate_device_artifacts(&domain, &bad).is_err());
    }
    // Heavy caps, heavy presence and the DBE table must have equal lengths:
    // each of the three truncated/extended alone is refused.
    let mut bad = bounds.clone();
    bad.heavy_seen.pop();
    assert!(validate_device_artifacts(&domain, &bad).is_err());
    let mut bad = bounds.clone();
    bad.dbe_by_heavy.pop();
    assert!(validate_device_artifacts(&domain, &bad).is_err());
    let mut bad = bounds.clone();
    bad.dbe_by_heavy.push([0; 2]);
    assert!(validate_device_artifacts(&domain, &bad).is_err());
    let mut bad = bounds.clone();
    bad.max_by_heavy.pop();
    assert!(validate_device_artifacts(&domain, &bad).is_err());
    // Carbon caps and presence likewise.
    let mut bad = bounds.clone();
    bad.carbon_seen.pop();
    assert!(validate_device_artifacts(&domain, &bad).is_err());
    // Ratio numerators and denominators at exactly 2^20 validate; above any
    // one of the 24 slots is refused.
    let mut edge = bounds.clone();
    for k in 0..6 {
        edge.ratio_lo_num[k] = 1 << 20;
        edge.ratio_lo_den[k] = 1 << 20;
        edge.ratio_hi_num[k] = 1 << 20;
        edge.ratio_hi_den[k] = 1 << 20;
    }
    assert!(validate_device_artifacts(&domain, &edge).is_ok());
    for k in 0..6 {
        for slot in 0..4 {
            let mut bad = edge.clone();
            match slot {
                0 => bad.ratio_lo_num[k] = (1 << 20) + 1,
                1 => bad.ratio_lo_den[k] = (1 << 20) + 1,
                2 => bad.ratio_hi_num[k] = (1 << 20) + 1,
                _ => bad.ratio_hi_den[k] = (1 << 20) + 1,
            }
            assert!(
                validate_device_artifacts(&domain, &bad).is_err(),
                "ratio slot {slot} of feature {k}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// H5 finding 8: independently calculated visits, saturation, parity, rare
// elements and the precise equality statement.
// ---------------------------------------------------------------------------

/// Independent brute-force visits of one single-rare-lane query over the
/// tiny domain: `(C, N, O)` heavy vectors in ascending lane order with a
/// non-empty hydrogen range, using the test literals only. Returns the
/// visit count and the first visited vector.
fn brute_visits_tiny(
    parent: u32,
    tol: u32,
    bound: u32,
    h_max: u16,
    lo_override: Option<u64>,
    hi_override: Option<u64>,
) -> (u32, Option<(u16, u16, u16)>) {
    let half = (tol as u64) + (bound as u64) + 1;
    // `E_domain` of the tiny domain from the test literals: caps C2 N1 O1
    // H6 give `2*0 + 6*33 + 1*5 + 1*381 = 584` nano-dalton, i.e. 1 dalton.
    let lo = lo_override.unwrap_or((parent as u64).saturating_sub(half));
    let hi = hi_override.unwrap_or((parent as u64).saturating_add(half).min(u64::from(u32::MAX)));
    let m_h = 1_007_825u64;
    let mut visits: u32 = 0;
    let mut first: Option<(u16, u16, u16)> = None;
    for c in 0..=2u16 {
        for n in 0..=1u16 {
            for o in 0..=1u16 {
                if c + n + o == 0 || c + n + o > 4 {
                    continue;
                }
                let m = (c as u64) * 12_000_000 + (n as u64) * 14_003_074 + (o as u64) * 15_994_915;
                if m > hi {
                    continue;
                }
                let h_lo = if m >= lo { 0 } else { (lo - m).div_ceil(m_h) };
                let h_hi = ((hi - m) / m_h).min(u64::from(h_max));
                if h_lo <= h_hi {
                    visits += 1;
                    if first.is_none() {
                        first = Some((c, n, o));
                    }
                }
            }
        }
    }
    (visits, first)
}

#[test]
fn device_lane_budget() {
    // Budgets 0 / 1 / exact fit with INDEPENDENTLY calculated visit counts
    // and first visited vectors (the brute-force loop above, not the
    // implementation). A single rare lane keeps lane attribution trivial.
    let domain = tiny_domain();
    let subset: Vec<Composition> = tiny_compositions()
        .into_iter()
        .filter(|c| c[0] <= 1 && c[1] <= 4)
        .collect();
    let bounds = RatioBounds::fit(subset.iter().copied(), 0).unwrap();
    assert_eq!(rare_table(&domain, &bounds).unwrap().len(), 1);
    let base: Composition = [1, 4, 0, 0, 0, 0, 0, 0, 0, 0];
    let query = EnumQuery {
        precursor_mz: protonated(&base),
        adduct: 1,
        ppm_tenths: 1000,
        precursor_uncertainty: 500_000,
    };
    let parent = brute_parent(query.precursor_mz, query.adduct);
    let tol = brute_tolerance(query.precursor_mz, query.ppm_tenths);
    let (brute_visits, first) = brute_visits_tiny(parent, tol, 500_001, 6, None, None);
    assert!(brute_visits > 1, "the window visits several vectors");
    let first = first.expect("non-empty visits");
    assert_eq!(first, (0, 0, 1), "O1 is the first visited vector");
    // Budget 0: nothing visited, nothing joined, the lane exhausted.
    let zero = enumerate_device_order(
        &domain,
        &bounds,
        &query,
        &DeviceEnumLimits {
            lane_visits_max: 0,
            scored_cap: u32::MAX,
        },
    )
    .unwrap();
    assert_eq!(zero.visited, 0);
    assert_eq!(zero.joined, 0);
    assert_eq!(zero.scored, 0);
    assert!(zero.exhausted);
    assert!(!zero.absent, "exhaustion is distinct from absence");
    assert!(zero.lanes.iter().all(|l| l.exhausted));
    assert_eq!(
        zero.status & request_status::FORMULA_SEARCH_EXHAUSTED,
        request_status::FORMULA_SEARCH_EXHAUSTED
    );
    // Budget 1: exactly the first visited vector is examined, so every
    // joined composition carries its `(C, N, O)`; the lane exhausts because
    // more visits remain. Deterministic across runs.
    let one = enumerate_device_order(
        &domain,
        &bounds,
        &query,
        &DeviceEnumLimits {
            lane_visits_max: 1,
            scored_cap: u32::MAX,
        },
    )
    .unwrap();
    let again = enumerate_device_order(
        &domain,
        &bounds,
        &query,
        &DeviceEnumLimits {
            lane_visits_max: 1,
            scored_cap: u32::MAX,
        },
    )
    .unwrap();
    assert_eq!(one, again, "budgeted runs are deterministic");
    assert_eq!(one.visited, 1, "exactly one visit at budget 1");
    assert_eq!(one.lanes[0].visited, 1);
    assert!(one.lanes[0].exhausted);
    assert!(one.exhausted);
    for c in &one.compositions {
        assert_eq!(
            (c[0], c[2], c[3]),
            first,
            "budget 1 joins only the first visited vector"
        );
    }
    // The first visited vector's hydrogen admits exactly `O1H0` here: `h`
    // is `{0}` in the window, and it joins as ambiguous.
    let oh0: Composition = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0];
    assert_eq!(one.compositions, vec![oh0]);
    // Open run: the implementation's visits equal the brute-force count.
    let open = enumerate_device_order(&domain, &bounds, &query, &device_open()).unwrap();
    assert!(!open.exhausted);
    assert_eq!(open.visited, brute_visits, "visits match the brute force");
    assert_eq!(open.lanes[0].visited, brute_visits);
    // Exact-fit budget: the brute-force count reproduces the open run with
    // no exhaustion.
    let fitted = enumerate_device_order(
        &domain,
        &bounds,
        &query,
        &DeviceEnumLimits {
            lane_visits_max: brute_visits,
            scored_cap: u32::MAX,
        },
    )
    .unwrap();
    assert!(!fitted.exhausted, "exact-fit budget completes");
    assert_eq!(fitted.compositions, open.compositions);
    assert_eq!(fitted.masses, open.masses);
    assert_eq!(fitted.flags, open.flags);
    assert_eq!(fitted.visited, open.visited);
    // One below the exact fit exhausts.
    let tight = enumerate_device_order(
        &domain,
        &bounds,
        &query,
        &DeviceEnumLimits {
            lane_visits_max: brute_visits - 1,
            scored_cap: u32::MAX,
        },
    )
    .unwrap();
    assert!(tight.exhausted);
    assert_eq!(tight.visited, brute_visits - 1);
}

#[test]
fn hi_saturates_at_u32_max() {
    // An actual `hi` saturation case: the precursor is chosen so
    // `parent + half` exceeds `u32::MAX` while `half` stays within the
    // scope restriction. The run still searches (no wrap to a narrow
    // window) and agrees with `enumerate`.
    let domain = EnumDomain {
        version: "test-hi-sat".to_string(),
        heavy_caps: [100, 20, 0, 0, 0, 0, 0, 0, 22],
        heavy_max: 142,
        hydrogen_min: 0,
        hydrogen_max: 64,
    };
    let a: Composition = [100, 22, 20, 0, 0, 0, 0, 0, 0, 22];
    let b: Composition = [100, 24, 20, 0, 0, 0, 0, 0, 0, 22];
    let bounds = RatioBounds::fit([a, b], 0).unwrap();
    assert!(validate_device_artifacts(&domain, &bounds).is_ok());
    assert_eq!(rare_table(&domain, &bounds).unwrap().len(), 1);
    let query = EnumQuery {
        precursor_mz: u32::MAX,
        adduct: 1,
        ppm_tenths: 1000,
        precursor_uncertainty: 670_499,
    };
    // Independently: parent, tolerance, bound, domain error, half.
    let parent = brute_parent(query.precursor_mz, 1);
    assert_eq!(parent, 4_293_960_019);
    let tol = brute_tolerance(query.precursor_mz, 1000);
    assert_eq!(tol, 429_496);
    let bound = 670_500u32;
    // `E_domain` from the test literals: `20*5 + 64*33 + 22*100 = 4412`
    // nano-dalton, i.e. 5 daltons.
    assert_eq!(domain.max_error(), 5);
    let half = tol + bound + 5;
    assert_eq!(half, 1_100_001);
    assert!(half <= DEVICE_HALF_MAX, "the scope restriction still searches");
    assert!(
        (parent as u64) + (half as u64) > u64::from(u32::MAX),
        "parent + half leaves u32: hi saturates"
    );
    let found = enumerate_device_order(&domain, &bounds, &query, &device_open()).unwrap();
    assert_eq!(found.parent_mass, Some(parent));
    assert!(!found.exhausted, "the saturated window still completes");
    assert!(found.visited > 0, "lanes examine real vectors");
    // `C100 H22 N20 I22` is independently in the window and admitted: its
    // mass is 4,294,132,014, `r = 171,995 <= tol + error = 1,100,000`, and
    // its exact stages pass (ceiling 200, even DBE 178).
    let gold: Composition = [100, 22, 20, 0, 0, 0, 0, 0, 0, 22];
    assert_eq!(brute_mass(&gold), 4_294_132_014);
    assert!(brute_h_ceil(&gold) >= 22);
    assert!(brute_parity(&gold));
    assert!(brute_dbe_ok(&gold));
    assert!(production_ratio_ok(&bounds, &gold));
    assert!(found.scored_contains(&gold));
    // Set equality with `enumerate` under the same exact filters.
    let limits = EnumLimits {
        ratio: Some(bounds.clone()),
        ..EnumLimits::default()
    };
    let expected = enumerate(&domain, &query, &limits).unwrap();
    assert!(!expected.exhausted);
    assert_eq!(device_rows_sorted(&found), enum_rows(&domain, &query, &limits));
}

#[test]
fn adjacent_hydrogen_parity_admits_one() {
    // Parity admits only one of two adjacent hydrogen counts: a window wide
    // enough for both `h` and `h + 1` still joins only the even-DBE one.
    // Here the window covers `CH2` and `CH4`... no: it covers `CH2` and
    // `CH3` (1.007825 apart), both admitted by every ratio stage, of which
    // only `CH2` survives parity.
    let domain = tiny_domain();
    let subset: Vec<Composition> = tiny_compositions()
        .into_iter()
        .filter(|c| c[0] <= 1 && c[1] <= 4)
        .collect();
    let bounds = RatioBounds::fit(subset.iter().copied(), 0).unwrap();
    let ch2: Composition = [1, 2, 0, 0, 0, 0, 0, 0, 0, 0];
    let ch3: Composition = [1, 3, 0, 0, 0, 0, 0, 0, 0, 0];
    // Both masses lie in the window (test's own arithmetic).
    let parent = (brute_mass(&ch2) as u64 + brute_mass(&ch3) as u64) / 2;
    assert_eq!(brute_mass(&ch3) - brute_mass(&ch2), 1_007_825);
    let query = EnumQuery {
        precursor_mz: (parent as u32) + 1_007_825 - 549,
        adduct: 1,
        ppm_tenths: 1000,
        precursor_uncertainty: 502_360,
    };
    assert_eq!(brute_parent(query.precursor_mz, 1), parent as u32);
    let tol = brute_tolerance(query.precursor_mz, 1000);
    let half = (tol as u64) + 502_361 + 1;
    assert!(brute_mass(&ch2) as u64 + half >= parent);
    assert!((brute_mass(&ch3) as u64) >= parent - half.min(parent));
    // Parity is the decider: the ratio stages admit both.
    assert!(brute_parity(&ch2));
    assert!(!brute_parity(&ch3));
    assert!(production_ratio_ok(&bounds, &ch2));
    assert!(
        production_ratio_ok(&bounds, &ch3),
        "every ratio stage admits CH3, so parity decides"
    );
    let found = enumerate_device_order(&domain, &bounds, &query, &device_open()).unwrap();
    assert!(!found.exhausted);
    assert!(found.scored_contains(&ch2), "CH2 joins");
    assert!(!found.scored_contains(&ch3), "CH3 falls to parity");
    let expected = enumerate(
        &domain,
        &query,
        &EnumLimits {
            ratio: Some(bounds.clone()),
            ..EnumLimits::default()
        },
    )
    .unwrap();
    assert!(!expected.exhausted);
    assert!(expected.scored_contains(&ch2));
    assert!(!expected.scored_contains(&ch3));
}

#[test]
fn rare_elements_match_independent_sets() {
    // Domains with non-zero P, Cl, Br and I against an independently
    // computed expected set (the test's own loops and literals).
    let domain = EnumDomain {
        version: "test-rare-px".to_string(),
        heavy_caps: [1, 0, 1, 0, 1, 0, 1, 1, 1],
        heavy_max: 4,
        hydrogen_min: 0,
        hydrogen_max: 4,
    };
    // Every composition of the domain with at least one heavy atom.
    let mut subset: Vec<Composition> = Vec::new();
    for c in 0..=1u16 {
        for o in 0..=1u16 {
            for p in 0..=1u16 {
                for cl in 0..=1u16 {
                    for br in 0..=1u16 {
                        for ii in 0..=1u16 {
                            if c + o + p + cl + br + ii == 0
                                || c + o + p + cl + br + ii > 4
                            {
                                continue;
                            }
                            for h in 0..=4u16 {
                                let mut comp: Composition = [0; 10];
                                comp[0] = c;
                                comp[1] = h;
                                comp[3] = o;
                                comp[5] = p;
                                comp[7] = cl;
                                comp[8] = br;
                                comp[9] = ii;
                                subset.push(comp);
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(!subset.is_empty());
    assert!(subset.iter().any(|c| c[5] == 1), "P is exercised");
    assert!(subset.iter().any(|c| c[7] == 1), "Cl is exercised");
    assert!(subset.iter().any(|c| c[8] == 1), "Br is exercised");
    assert!(subset.iter().any(|c| c[9] == 1), "I is exercised");
    let bounds = RatioBounds::fit(subset.iter().copied(), 0).unwrap();
    assert!(validate_device_artifacts(&domain, &bounds).is_ok());
    let gold: Composition = [1, 3, 0, 1, 0, 1, 0, 0, 0, 0];
    assert!(subset.contains(&gold));
    let query = centred_query(&gold, 1000, 5000);
    let parent = brute_parent(query.precursor_mz, query.adduct);
    let tol = brute_tolerance(query.precursor_mz, query.ppm_tenths);
    let bound = 5001u32;
    // Independent expected joins: verdict, exact filters, oracle.
    let mut expected: Vec<(u32, Composition, bool)> = Vec::new();
    for c in &subset {
        let mass = brute_mass(c);
        let error = brute_error(c).saturating_add(bound);
        let (joined, ambiguous) = brute_verdict(parent, mass, error, tol);
        if !joined {
            continue;
        }
        if c[1] > brute_h_ceil(c) || !brute_parity(c) || !brute_dbe_ok(c) {
            continue;
        }
        if !brute_ratio_ok(&subset, c) {
            continue;
        }
        expected.push((mass, *c, ambiguous));
    }
    expected.sort_by_key(|a| (a.0, a.1));
    assert!(!expected.is_empty(), "non-vacuous rare domain");
    assert!(
        expected.iter().any(|(_, c, _)| c == &gold),
        "the P-bearing gold joins"
    );
    let limits = EnumLimits {
        ratio: Some(bounds.clone()),
        ..EnumLimits::default()
    };
    assert_eq!(enum_rows(&domain, &query, &limits), expected);
    let found = enumerate_device_order(&domain, &bounds, &query, &device_open()).unwrap();
    assert!(!found.exhausted);
    assert_eq!(device_rows_sorted(&found), expected);
    assert!(found.scored_contains(&gold));
}

#[test]
fn device_enumerate_equality_requires_exact_filters() {
    // The equality statement, precisely: with the same exact filters
    // enabled the device joined set equals `enumerate`'s; with the filters
    // off (`EnumLimits::unfiltered`) `enumerate` admits compositions the
    // device twin — which applies the exact stages unconditionally —
    // rejects. `CH3` (odd parity) is the witness.
    let domain = tiny_domain();
    let subset: Vec<Composition> = tiny_compositions()
        .into_iter()
        .filter(|c| c[0] <= 1 && c[1] <= 4)
        .collect();
    let bounds = RatioBounds::fit(subset.iter().copied(), 0).unwrap();
    let ch3: Composition = [1, 3, 0, 0, 0, 0, 0, 0, 0, 0];
    let query = centred_query(&ch3, 200, 500);
    let open = enumerate(&domain, &query, &EnumLimits::unfiltered()).unwrap();
    assert!(!open.exhausted);
    assert!(open.scored_contains(&ch3), "unfiltered enumerate admits CH3");
    let found = enumerate_device_order(&domain, &bounds, &query, &device_open()).unwrap();
    assert!(!found.exhausted);
    assert!(!found.scored_contains(&ch3), "the device twin rejects CH3");
    let filtered = EnumLimits {
        ratio: Some(bounds.clone()),
        ..EnumLimits::default()
    };
    let expected = enumerate(&domain, &query, &filtered).unwrap();
    assert!(!expected.exhausted);
    assert!(!expected.scored_contains(&ch3));
    assert_eq!(
        device_rows_sorted(&found),
        enum_rows(&domain, &query, &filtered),
        "same exact filters enabled: same joined set and flags"
    );
}

// ---------------------------------------------------------------------------
// EF finding 2: validated DBE endpoints keep the biased comparison exact.
// ---------------------------------------------------------------------------

#[test]
fn dbe_endpoint_validation_rejects_overflow() {
    // The reviewer's two inputs on valid CH4 artifacts. CH4 (twice-DBE 0)
    // belongs in `[0, 2_147_483_647]`, yet packing biases the upper endpoint
    // to `u32::MAX` so the kernel's guarded `neg <= max - shi` comparison
    // fails and CH4 is wrongly rejected; `i64::MAX` overflows the host bias
    // addition itself. Both are now `Error::Config`.
    // Documented exact range: pos <= 2,552 and neg <= 2,043 under the cap
    // validation, so `stored <= u32::MAX - 2,043`, i.e. endpoints in
    // `[-2^31, 2^31 - 1 - 2,043] = [-2,147,483,648, 2,147,481,604]`.
    assert_eq!(DBE_ENDPOINT_MIN, -(1i64 << 31));
    assert_eq!(DBE_ENDPOINT_MAX, (1i64 << 31) - 1 - 2043);
    let ch4: Composition = [1, 4, 0, 0, 0, 0, 0, 0, 0, 0];
    let domain = EnumDomain::from_compositions([ch4], 0).unwrap();
    let bounds = RatioBounds::fit([ch4], 0).unwrap();
    assert!(validate_device_artifacts(&domain, &bounds).is_ok());
    // Sanity: CH4 joins under the valid artifacts.
    let query = EnumQuery {
        precursor_mz: protonated(&ch4),
        adduct: 1,
        ppm_tenths: 200,
        precursor_uncertainty: 50,
    };
    let found = enumerate_device_order(&domain, &bounds, &query, &device_open()).unwrap();
    assert!(found.scored_contains(&ch4));
    // Reviewer's first input: upper endpoint `2_147_483_647`.
    let mut wide = bounds.clone();
    wide.dbe_by_heavy[0] = [0, 2_147_483_647];
    assert!(matches!(
        validate_device_artifacts(&domain, &wide),
        Err(Error::Config(_))
    ));
    assert!(pack_device_bounds(&domain, &wide).is_err());
    // Reviewer's second input: upper endpoint `i64::MAX`.
    let mut huge = bounds.clone();
    huge.dbe_by_heavy[0] = [0, i64::MAX];
    assert!(matches!(
        validate_device_artifacts(&domain, &huge),
        Err(Error::Config(_))
    ));
    assert!(pack_device_bounds(&domain, &huge).is_err());
    // The exact boundary: `DBE_ENDPOINT_MAX` validates, one past it fails.
    let mut edge = bounds.clone();
    edge.dbe_by_heavy[0] = [DBE_ENDPOINT_MIN, DBE_ENDPOINT_MAX];
    assert!(validate_device_artifacts(&domain, &edge).is_ok());
    let mut past = bounds.clone();
    past.dbe_by_heavy[0] = [DBE_ENDPOINT_MIN, DBE_ENDPOINT_MAX + 1];
    assert!(matches!(
        validate_device_artifacts(&domain, &past),
        Err(Error::Config(_))
    ));
    let mut below = bounds.clone();
    below.dbe_by_heavy[0] = [DBE_ENDPOINT_MIN - 1, 0];
    assert!(matches!(
        validate_device_artifacts(&domain, &below),
        Err(Error::Config(_))
    ));
}

// ---------------------------------------------------------------------------
// EF finding 3: the rare-table limit counts representable rows only.
// ---------------------------------------------------------------------------

#[test]
fn rare_table_exact_boundary_keeps_representable() {
    // The reviewer's exact-boundary case: rare caps (F, P, S, Cl, Br, I) =
    // (201, 13, 0, 0, 0, 5) with rare total `[0, 219]` and distinct `[0, 3]`.
    // Exactly 16,384 tuples have representable mass; the next tuple
    // `(201, 13, 0, 0, 0, 1)` (mass 4,348,242,381) must be dropped, not
    // counted against `P_max`. The old code checked the limit before the
    // mass and returned `Error::Config` here.
    let full: Composition = [0, 0, 0, 0, 201, 13, 0, 0, 0, 5];
    let empty: Composition = [0; 10];
    let domain = EnumDomain::from_compositions([empty, full], 0).unwrap();
    assert_eq!(
        domain.heavy_caps,
        [0, 0, 0, 201, 13, 0, 0, 0, 5],
        "rare caps (F, P, I) = (201, 13, 5)"
    );
    let bounds = RatioBounds::fit([empty, full], 0).unwrap();
    assert_eq!(bounds.rare_total, [0, 219]);
    assert_eq!(bounds.rare_distinct, [0, 3]);
    let rows = rare_table(&domain, &bounds).unwrap();
    assert_eq!(
        rows.len(),
        16_384,
        "exactly the 16,384 representable rows are kept"
    );
    assert!(validate_device_artifacts(&domain, &bounds).is_ok());
    // Every kept row is mass-representable and in range.
    for r in &rows {
        assert!(r[6] <= u32::MAX, "row mass fits u32");
        assert!(r[7] <= 219);
    }
}

// ---------------------------------------------------------------------------
// EF finding 6: ppm validation runs before status-dependent early returns.
// ---------------------------------------------------------------------------

#[test]
fn device_order_validates_ppm_before_status_exits() {
    // Valid CH4 artifacts; ppm 1001 must be `Error::Config` even when the
    // spectrum would otherwise take the unknown-precision or bad-parent
    // early exit. `enumerate` already validates first; the device entry now
    // matches it.
    let ch4: Composition = [1, 4, 0, 0, 0, 0, 0, 0, 0, 0];
    let domain = EnumDomain::from_compositions([ch4], 0).unwrap();
    let bounds = RatioBounds::fit([ch4], 0).unwrap();
    let limits = device_open();
    // Unknown precision + ppm 1001: Config, not the sentinel result.
    let unknown_bad = EnumQuery {
        precursor_mz: protonated(&ch4),
        adduct: 1,
        ppm_tenths: 1001,
        precursor_uncertainty: u32::MAX,
    };
    assert!(matches!(
        enumerate_device_order(&domain, &bounds, &unknown_bad, &limits),
        Err(Error::Config(_))
    ));
    // Invalid parent + ppm 1001: Config, not mass_overflow.
    let bad_parent = EnumQuery {
        precursor_mz: 0,
        adduct: 1,
        ppm_tenths: 1001,
        precursor_uncertainty: 50,
    };
    assert!(matches!(
        enumerate_device_order(&domain, &bounds, &bad_parent, &limits),
        Err(Error::Config(_))
    ));
    // And the valid-ppm counterparts still take their status exits.
    let unknown_ok = EnumQuery {
        ppm_tenths: 200,
        ..unknown_bad
    };
    let found = enumerate_device_order(&domain, &bounds, &unknown_ok, &limits).unwrap();
    assert_eq!(
        found.status,
        request_status::EXACT_MASS_UNAVAILABLE | request_status::FORMULA_ABSENT
    );
    let parent_bad_ok = EnumQuery {
        ppm_tenths: 200,
        ..bad_parent
    };
    let found = enumerate_device_order(&domain, &bounds, &parent_bad_ok, &limits).unwrap();
    assert_eq!(found.status, request_status::MASS_OVERFLOW);
}

// ---------------------------------------------------------------------------
// EF finding 5: the dispatch check without allocating.
// ---------------------------------------------------------------------------

#[test]
fn dispatch_check_enforces_lane_ceiling() {
    // The reviewer's case: rare caps admitting all 16,384 tuples with B = 17
    // gives 278,528 lanes, past the default 262,144. The check refuses it
    // with no buffer allocated.
    assert!(matches!(
        validate_enum_dispatch(17, 16_384, 32, ENUM_LANES_MAX_DEFAULT),
        Err(Error::Config(_))
    ));
    assert_eq!(17usize * 16_384, 278_528);
    // At exactly the ceiling the dispatch is accepted.
    assert!(validate_enum_dispatch(16, 16_384, 32, ENUM_LANES_MAX_DEFAULT).is_ok());
    assert_eq!(16usize * 16_384, 262_144);
    // Small dispatches pass; an explicit zero ceiling refuses any lane.
    assert!(validate_enum_dispatch(1, 1, 32, ENUM_LANES_MAX_DEFAULT).is_ok());
    assert!(validate_enum_dispatch(0, 0, 0, ENUM_LANES_MAX_DEFAULT).is_ok());
    assert!(matches!(
        validate_enum_dispatch(1, 1, 32, 0),
        Err(Error::Config(_))
    ));
    // A `u32`-unrepresentable largest address is `Error::Shape`.
    assert!(matches!(
        validate_enum_dispatch(1, 1, u32::MAX as usize, ENUM_LANES_MAX_DEFAULT),
        Err(Error::Shape(_))
    ));
}

// ---------------------------------------------------------------------------
// EF finding 7 (twin): CH2 and CH4 join as ambiguous; 2^20 ratios enumerate.
// ---------------------------------------------------------------------------

#[test]
fn two_hydrogens_ch2_ch4_join_ambiguous() {
    // Parent 15,023,475, adduct 1, ppm 1000, uncertainty 1,007,000, bounds
    // fitted from CH2 and CH4. Both hydrogen counts lie in the window
    // (|parent - mass| = m_H = 1,007,825 each side) with `r - tol < error`,
    // so both verdicts are Ambiguous; parity keeps both (twice-DBE 2 and 0)
    // while removing the intermediate CH3 (twice-DBE 1, odd). The numbers do
    // produce the reviewer's case, verified step by step below.
    let ch2: Composition = [1, 2, 0, 0, 0, 0, 0, 0, 0, 0];
    let ch3: Composition = [1, 3, 0, 0, 0, 0, 0, 0, 0, 0];
    let ch4: Composition = [1, 4, 0, 0, 0, 0, 0, 0, 0, 0];
    assert_eq!(brute_mass(&ch2), 14_015_650);
    assert_eq!(brute_mass(&ch4), 16_031_300);
    let domain = EnumDomain::from_compositions([ch2, ch4], 0).unwrap();
    let bounds = RatioBounds::fit([ch2, ch4], 0).unwrap();
    assert!(validate_device_artifacts(&domain, &bounds).is_ok());
    let parent = 15_023_475u32;
    let precursor = parent + 1_007_825 - 549;
    let query = EnumQuery {
        precursor_mz: precursor,
        adduct: 1,
        ppm_tenths: 1000,
        precursor_uncertainty: 1_007_000,
    };
    // Independent window check with test literals: both masses inside.
    let tol = brute_tolerance(precursor, 1000);
    let half = tol + 1_007_001 + 1;
    assert!(parent.abs_diff(brute_mass(&ch2)) as u64 <= half as u64);
    assert!(parent.abs_diff(brute_mass(&ch4)) as u64 <= half as u64);
    let found = enumerate_device_order(&domain, &bounds, &query, &device_open()).unwrap();
    assert!(!found.exhausted);
    for (want, name) in [(ch2, "CH2"), (ch4, "CH4")] {
        let pos = found
            .compositions
            .iter()
            .position(|c| c == &want)
            .unwrap_or_else(|| panic!("{name} joins"));
        assert_eq!(found.flags[pos], 2, "{name} is ambiguous");
    }
    assert!(
        !found.scored_contains(&ch3),
        "parity removes the intermediate CH3"
    );
}

#[test]
fn ratio_two_to_twenty_enumerates() {
    // Ratio factors exactly 2^20 (the validation maximum) through the real
    // enumeration: every candidate cross-multiplies `num * 2^20` with
    // numerators <= 1,023, so every product is <= 1,072,693,248 < 2^32 and
    // no overflow guard can trip. The gold still joins on the twin.
    let ch4: Composition = [1, 4, 0, 0, 0, 0, 0, 0, 0, 0];
    let domain = EnumDomain::from_compositions([ch4], 0).unwrap();
    let mut bounds = RatioBounds::fit([ch4], 0).unwrap();
    for k in 0..6 {
        bounds.ratio_lo_num[k] = 0;
        bounds.ratio_lo_den[k] = 1 << 20;
        bounds.ratio_hi_num[k] = 1 << 20;
        bounds.ratio_hi_den[k] = 1;
    }
    assert!(validate_device_artifacts(&domain, &bounds).is_ok());
    let query = EnumQuery {
        precursor_mz: protonated(&ch4),
        adduct: 1,
        ppm_tenths: 200,
        precursor_uncertainty: 50,
    };
    let found = enumerate_device_order(&domain, &bounds, &query, &device_open()).unwrap();
    assert!(!found.exhausted);
    assert!(
        found.scored_contains(&ch4),
        "gold joins with 2^20 ratio factors"
    );
}

// ---------------------------------------------------------------------------
// MC12: neutral-mass entry.
// ---------------------------------------------------------------------------

fn neutral_rows(
    domain: &EnumDomain,
    neutral_mass: u32,
    ppm_tenths: u32,
    uncertainty: u32,
    limits: &EnumLimits,
) -> Vec<(u32, Composition, bool)> {
    let found = enumerate_neutral(domain, neutral_mass, ppm_tenths, uncertainty, limits).unwrap();
    let mut rows: Vec<(u32, Composition, bool)> = found
        .masses
        .iter()
        .zip(found.compositions.iter())
        .zip(found.ambiguous.iter())
        .map(|((m, c), a)| (*m, *c, *a))
        .collect();
    rows.sort_by_key(|a| (a.0, a.1));
    rows
}

#[test]
fn neutral_equals_protonated_when_budget_matched() {
    // Zero ppm makes the tolerance identical (0) on both paths. The budgets
    // match when the neutral uncertainty is one above the precursor's: the
    // precursor bound carries the +1 adduct-conversion rounding term while
    // the neutral path carries none.
    let domain = tiny_domain();
    let ethanol: Composition = [2, 6, 0, 1, 0, 0, 0, 0, 0, 0];
    let neutral = brute_mass(&ethanol);
    let precursor = protonated(&ethanol);
    for filtered in [false, true] {
        let limits = if filtered {
            EnumLimits::default()
        } else {
            EnumLimits::unfiltered()
        };
        let query = EnumQuery {
            precursor_mz: precursor,
            adduct: 1,
            ppm_tenths: 0,
            precursor_uncertainty: 50,
        };
        let charged = enum_rows(&domain, &query, &limits);
        let neutral_found = neutral_rows(&domain, neutral, 0, 51, &limits);
        assert_eq!(neutral_found, charged, "filtered {filtered}: neutral equals protonated");
    }
}

#[test]
fn neutral_verdicts_equal_brute_decide_with_neutral_bound() {
    // Verdicts equal a brute-force loop over `decide` with the neutral error
    // bound only (no electron term, no adduct hydrogen, no +1): the bound is
    // `ceil(error_nda/1000) + uncertainty`.
    let domain = tiny_domain();
    let neutral: u32 = 30_000_000;
    let ppm = 200u32;
    let uncertainty = 500u32;
    let limits = EnumLimits::unfiltered();
    let found = enumerate_neutral(&domain, neutral, ppm, uncertainty, &limits).unwrap();
    let tol = brute_tolerance(neutral, ppm);
    let bound = uncertainty;
    let mut expected: Vec<(u32, Composition, bool)> = Vec::new();
    for c in tiny_compositions() {
        let mass = brute_mass(&c);
        let error = brute_error(&c).saturating_add(bound);
        let (joined, ambiguous) = brute_verdict(neutral, mass, error, tol);
        if joined {
            expected.push((mass, c, ambiguous));
        }
    }
    expected.sort_by_key(|a| (a.0, a.1));
    let mut got: Vec<(u32, Composition, bool)> = found
        .masses
        .iter()
        .zip(found.compositions.iter())
        .zip(found.ambiguous.iter())
        .map(|((m, c), a)| (*m, *c, *a))
        .collect();
    got.sort_by_key(|a| (a.0, a.1));
    assert_eq!(got, expected);
    // Production `decide` agrees on the same inputs.
    for (mass, c, ambiguous) in &expected {
        let error = brute_error(c).saturating_add(bound);
        let verdict = decide(neutral, *mass, error, tol);
        assert_eq!(
            (verdict != mamba3::models::ms2::Verdict::Reject,
             verdict == mamba3::models::ms2::Verdict::Ambiguous),
            (true, *ambiguous)
        );
    }
}

#[test]
fn neutral_unknown_precision_and_ppm_bound() {
    let domain = tiny_domain();
    let found =
        enumerate_neutral(&domain, 30_000_000, 100, u32::MAX, &EnumLimits::unfiltered()).unwrap();
    assert!(found.absent);
    assert!(!found.exhausted);
    assert_eq!(found.nodes_visited, 0);
    assert!(found.compositions.is_empty());
    assert_eq!(
        found.status,
        request_status::EXACT_MASS_UNAVAILABLE | request_status::FORMULA_ABSENT
    );
    assert_eq!(found.parent_mass, Some(30_000_000));
    assert!(enumerate_neutral(&domain, 30_000_000, 1001, 50, &EnumLimits::unfiltered()).is_err());
}
/// Pinned precursor-path regression: the exact `enumerate` output for a fixed
/// ethanol `[M+H]+` query.
///
/// The shared core may be refactored, but this output must stay
/// byte-identical: the precursor path (adduct conversion, tolerance at the
/// observed precursor m/z, `uncertainty + 1` bound) is frozen. Values were
/// captured from the pre-change implementation.
#[test]
fn precursor_path_pinned_ethanol() {
    let domain = EnumDomain {
        version: mamba3::models::ms2::formula_enum::ENUM_DOMAIN_VERSION.to_string(),
        heavy_caps: [3, 0, 2, 0, 0, 0, 0, 0, 0],
        heavy_max: 4,
        hydrogen_min: 0,
        hydrogen_max: 10,
    };
    let query = EnumQuery {
        precursor_mz: 47049141,
        adduct: 1,
        ppm_tenths: 200,
        precursor_uncertainty: 50,
    };
    let found = enumerate(&domain, &query, &EnumLimits::default()).unwrap();
    // Parent: 47049141 - 1007825 + 549 = 46041865 (ethanol, C2H6O).
    assert_eq!(found.parent_mass, Some(46041865));
    assert_eq!(found.compositions, vec![[2, 6, 0, 1, 0, 0, 0, 0, 0, 0]]);
    assert_eq!(found.masses, vec![46041865]);
    assert_eq!(found.ambiguous, vec![false]);
    assert_eq!(found.nodes_visited, 22);
    assert_eq!(found.hydrogen_checks, 1);
    assert_eq!((found.rows_joined, found.rows_scored), (1, 1));
    assert!(!found.exhausted && !found.absent && found.support_complete);
    assert_eq!(found.status, 0);
    assert_eq!(
        (
            found.rejected_h_max,
            found.rejected_parity,
            found.rejected_dbe,
            found.rejected_mass
        ),
        (0, 0, 0, 0)
    );
    // Named error parts of the precursor bound: observation 50, the +1 is the
    // adduct-conversion rounding bound (contract §4.3: 33 + 421 nDa, ceil 1).
    assert_eq!(found.error_observation, 50);
    assert_eq!(found.error_neutralisation, 1);
}

/// Neutral-path boundary: the `+1` adduct-conversion bound must NOT apply to
/// an already-neutral mass.
///
/// C2H6O has composition error `ceil((6 * 33 + 381) / 1000) = 1`. With
/// uncertainty 50 the neutral verdict error is 51; at `ppm_tenths = 12` the
/// tolerance at the neutral mass is `floor(46041865 * 12 / 1e7) = 55`, so a
/// residual of 4 (`r + E = 55`) is exactly Accept. Under the old precursor
/// bound (`E = 52`) the same residual (`r + E = 56 > 55`) is
/// boundary-ambiguous. The precursor path keeps the `+1`.
#[test]
fn neutral_path_drops_adduct_rounding_bound() {
    let domain = EnumDomain {
        version: mamba3::models::ms2::formula_enum::ENUM_DOMAIN_VERSION.to_string(),
        heavy_caps: [3, 0, 2, 0, 0, 0, 0, 0, 0],
        heavy_max: 4,
        hydrogen_min: 0,
        hydrogen_max: 10,
    };
    let found = enumerate_neutral(&domain, 46041869, 12, 50, &EnumLimits::default()).unwrap();
    assert_eq!(found.parent_mass, Some(46041869));
    assert_eq!(found.compositions, vec![[2, 6, 0, 1, 0, 0, 0, 0, 0, 0]]);
    assert_eq!(found.masses, vec![46041865]);
    // Accepted under the neutral bound ...
    assert_eq!(found.ambiguous, vec![false]);
    // ... while the old ion bound (+1) would leave it boundary-ambiguous.
    assert_eq!(
        decide(46041869, 46041865, 52, 55),
        mamba3::models::ms2::chem::Verdict::Ambiguous
    );
    assert_eq!(decide(46041869, 46041865, 51, 55), mamba3::models::ms2::chem::Verdict::Accept);
    // Named error parts of the neutral bound: no neutralisation term.
    assert_eq!(found.error_observation, 50);
    assert_eq!(found.error_neutralisation, 0);
}

/// A bound capacity never reads as a complete search: a window with more
/// joins than a small capacity keeps the canonical prefix (the caller's
/// selection re-orders it by its own key over the unbounded search) and
/// reports `search_exhausted`, never a silently complete truncation.
#[test]
fn bound_capacity_reports_search_exhausted() {
    let domain = tiny_domain();
    let neutral: u32 = 30_000_000;
    let (ppm, uncertainty) = (1000u32, 600_000u32);
    let open = EnumLimits::unfiltered();
    let full = enumerate_neutral(&domain, neutral, ppm, uncertainty, &open).unwrap();
    assert!(!full.exhausted && full.support_complete);
    assert!(full.rows_joined > 3, "the window joins several rows: {}", full.rows_joined);
    let mut capped = EnumLimits::unfiltered();
    capped.capacity = 2;
    let small = enumerate_neutral(&domain, neutral, ppm, uncertainty, &capped).unwrap();
    assert!(small.exhausted, "a bound capacity exhausts the search");
    assert!(!small.support_complete && !small.absent);
    assert_eq!(
        small.status & request_status::FORMULA_SEARCH_EXHAUSTED,
        request_status::FORMULA_SEARCH_EXHAUSTED
    );
    assert_eq!(small.rows_joined, full.rows_joined);
    assert_eq!(small.rows_scored, 2);
    // The kept rows are the canonical prefix of the uncapped output.
    assert_eq!(&small.compositions[..], &full.compositions[..2]);
    assert_eq!(&small.masses[..], &full.masses[..2]);
}
