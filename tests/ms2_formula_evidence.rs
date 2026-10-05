//! E1 host tests: the formula-evidence twins against `ion_assign` and
//! against hand-computed expectations.
//!
//! Host only (no device, no feature gate). Every `(candidate, peak)`
//! equivalence goes through the reviewed [`ion_assign`](mamba3::models::ms2::ion::ion_assign)
//! reference; every selection order and feature row is recomputed by hand or
//! with independent `u64`/`f64` arithmetic in the test, never by calling the
//! twin under test.

use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

use mamba3::models::ms2::MolGraph;
use mamba3::models::ms2::chem::{Composition, ELEMENTS, HYDROGEN, composition_mass, ion};
use mamba3::models::ms2::formula::FormulaTable;
use mamba3::models::ms2::formula_enum::EnumDomain;
use mamba3::models::ms2::formula_evidence::{
    EVIDENCE_PEAKS, evidence_peaks, formula_evidence, formula_evidence_lane,
    formula_evidence_lane_slow, formula_evidence_lane_slow_trials,
    formula_evidence_lane_trials, formula_features, formula_features_lane,
    hydrogen_s_max, hydrogen_trials_bound, hydrogen_wrapped_trials,
};
use mamba3::models::ms2::formula_evidence_ref::build_evidence_index;
use mamba3::models::ms2::ion::{ION_SEARCH_EXHAUSTED, IonLimits, ion_assign};
use mamba3::models::ms2::tolerance;
use serde_json::Value;

fn fixture() -> Value {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/ms2/chemistry_v0.json");
    let text = std::fs::read_to_string(path).expect("fixture readable");
    serde_json::from_str(&text).expect("fixture parses")
}

fn graph_of(name: &str) -> MolGraph {
    let f = fixture();
    let m = f["molecules"]
        .as_array()
        .expect("molecules")
        .iter()
        .find(|m| m["name"] == name)
        .unwrap_or_else(|| panic!("molecule {name} in fixture"));
    let atoms: Vec<u8> = m["atoms"]
        .as_array()
        .expect("atoms")
        .iter()
        .map(|a| a.as_u64().expect("atom type") as u8)
        .collect();
    let bonds: Vec<(usize, usize, u8)> = m["bonds"]
        .as_array()
        .expect("bonds")
        .iter()
        .map(|b| {
            let b = b.as_array().expect("bond triple");
            (
                b[0].as_u64().expect("a") as usize,
                b[1].as_u64().expect("b") as usize,
                b[2].as_u64().expect("order") as u8,
            )
        })
        .collect();
    MolGraph::new(atoms, bonds).expect("fixture molecule builds")
}

/// Target mass and fragment tolerance of a peak m/z, with independent `u64`
/// arithmetic (the `u32` algorithm cannot wrap at these magnitudes).
fn peak_row(mz: u32, adduct: u16, ppm: u32) -> (u32, u32) {
    let t = if adduct == 1 {
        mz.checked_add(549).expect("target fits")
    } else {
        mz.checked_sub(549).expect("target fits")
    };
    let hi = u64::from(mz / 10_000);
    let lo = u64::from(mz % 10_000);
    let q = hi * u64::from(ppm);
    let tol = (q / 1000 + ((q % 1000) * 10_000 + lo * u64::from(ppm)) / 10_000_000) as u32;
    (t, tol)
}

/// m/z of the ion hypothesis with the given heavy counts and parent
/// hydrogens at shift 0, via [`chem::ion`](mamba3::models::ms2::chem::ion).
fn ion_mz_of(heavy: &[u16; 9], parent_h: u16, adduct: u16) -> u32 {
    let mut c: Composition = [0; 10];
    const ORDER: [usize; 9] = [0, 2, 3, 4, 5, 6, 7, 8, 9];
    for (s, &e) in ORDER.iter().enumerate() {
        c[e] = heavy[s];
    }
    c[HYDROGEN] = parent_h;
    ion(&c, adduct, 0)
        .expect("ion mass fits")
        .expect("ion hydrogen non-negative")
        .mz
}

/// True-fragment peaks of a parent composition: the full heavy vector with
/// shifts, a half-carbon fragment, a single carbon, each via `chem::ion`,
/// plus two far decoy masses.
fn true_peaks(parent: &Composition, adduct: u16) -> Vec<u32> {
    const ORDER: [usize; 9] = [0, 2, 3, 4, 5, 6, 7, 8, 9];
    let mut full = [0u16; 9];
    for (s, &e) in ORDER.iter().enumerate() {
        full[s] = parent[e];
    }
    let mut peaks = Vec::new();
    for shift in [-1i32, 0, 1, 2] {
        let mut c = [0u16; 10];
        for (s, &e) in ORDER.iter().enumerate() {
            c[e] = full[s];
        }
        c[HYDROGEN] = parent[HYDROGEN];
        if let Some(hyp) = ion(&c, adduct, shift).expect("ion mass fits") {
            // Back to the peak m/z the target mass came from.
            let mz = if adduct == 1 {
                hyp.mz.checked_sub(549).expect("peak fits")
            } else {
                hyp.mz.checked_add(549).expect("peak fits")
            };
            peaks.push(mz);
        }
    }
    if parent[0] >= 2 {
        let mut half = full;
        half[0] /= 2;
        peaks.push(ion_mz_of(&half, parent[HYDROGEN], adduct));
    }
    let mut one = [0u16; 9];
    one[0] = 1;
    peaks.push(ion_mz_of(&one, parent[HYDROGEN], adduct));
    peaks.sort_unstable();
    peaks.dedup();
    peaks.push(10_000_000);
    let pm = composition_mass(parent).expect("parent mass fits") as u64;
    peaks.push((pm + 500_000_000).min(u64::from(u32::MAX)) as u32);
    peaks
}

/// Candidate set for a parent: the parent, two near variants and one
/// unrelated composition.
fn candidates_of(parent: &Composition) -> Vec<Composition> {
    let mut minus_c = *parent;
    minus_c[0] = minus_c[0].saturating_sub(1);
    let mut plus_o = *parent;
    plus_o[3] = plus_o[3].saturating_add(1);
    let unrelated: Composition = [2, 3, 0, 0, 0, 0, 0, 1, 0, 0];
    vec![*parent, minus_c, plus_o, unrelated]
}

/// Pack compositions into a `cand` buffer with integer masses, flag 1 and
/// source id `u32::MAX`, plus one trailing padding slot.
fn pack_cand(comps: &[Composition]) -> (Vec<u32>, usize) {
    let m = comps.len() + 1;
    let mut cand = vec![0u32; m * 13];
    for (i, c) in comps.iter().enumerate() {
        for e in 0..10 {
            cand[i * 13 + e] = u32::from(c[e]);
        }
        cand[i * 13 + 10] = composition_mass(c).expect("candidate mass fits");
        cand[i * 13 + 11] = 1;
        cand[i * 13 + 12] = u32::MAX;
    }
    cand[(m - 1) * 13 + 12] = u32::MAX;
    (cand, m)
}

/// One candidate as a single-slot `cand` buffer.
fn pack_one(comp: &Composition) -> Vec<u32> {
    let mut cand = vec![0u32; 13];
    for e in 0..10 {
        cand[e] = u32::from(comp[e]);
    }
    cand[10] = composition_mass(comp).expect("candidate mass fits");
    cand[11] = 1;
    cand[12] = u32::MAX;
    cand
}

fn assert_f32_close(actual: f32, expected: f32, tol: f32, what: &str) {
    assert!(
        (actual - expected).abs() <= tol,
        "{what}: got {actual}, want {expected}"
    );
}

// ---------------------------------------------------------------------------
// Twin of kernel 2 against `ion_assign`, pair by pair.
// ---------------------------------------------------------------------------

/// A peak is explained by `c` exactly when `ion_assign` with parent `c`
/// reports `accepted >= 1`: shown here for every (candidate, peak) of three
/// real molecules' compositions, with peaks from true sub-compositions
/// (via `chem::ion`) plus decoys.
///
/// Each pair runs through the real [`formula_evidence_lane`] on a
/// single-peak evidence buffer, so the verdict path is the lane's own.
#[test]
fn explained_equals_ion_assign_every_pair() {
    let cases = [
        ("acetonitrile", 1u16),
        ("cysteine", 1u16),
        ("cysteine", 2u16),
        ("2-butanol", 1u16),
    ];
    let work_max = 65_536u32;
    let (u, ppm) = (50u32, 100u32);
    let p = 4usize;
    for (name, adduct) in cases {
        let parent = graph_of(name).composition();
        let peaks = true_peaks(&parent, adduct);
        assert!(peaks.len() >= 6, "{name}: true peaks plus decoys");
        let comps = candidates_of(&parent);
        let meta = vec![0u32, 0, 0, u32::from(adduct), 0, 0, 0, 0];
        let spec = vec![u, 0];
        let limits = IonLimits {
            work_max,
            kept: 4,
        };
        let mut any_explained = false;
        let mut any_unexplained = false;
        for comp in comps.iter() {
            let cand = pack_one(comp);
            for &mz in peaks.iter() {
                let (t, tol_p) = peak_row(mz, adduct, ppm);
                let mut ev_peaks = vec![u32::MAX, 0, 0, 0];
                ev_peaks.extend_from_slice(&[0, t, tol_p, 1]);
                ev_peaks.extend_from_slice(&[u32::MAX, 0, 0, 0, u32::MAX, 0, 0, 0]);
                let ev_w = vec![0.0f32, 1.0, 0.0, 0.0];
                let mut out = vec![0.0f32; 4];
                formula_evidence_lane(
                    &cand, &ev_peaks, &ev_w, &meta, &spec, 0, 0, 1, p as u32, work_max, u32::MAX,
                    &mut out,
                );
                let assign =
                    ion_assign(comp, adduct, mz, u, ppm, &limits).expect("ion_assign runs");
                let explained = out[0] == 1.0;
                assert_eq!(
                    explained,
                    assign.accepted >= 1,
                    "{name} adduct {adduct} candidate {comp:?} peak {mz}: lane {out:?} vs accepted {}",
                    assign.accepted,
                );
                // The single-peak row carries its own count, weight, size
                // and completion flag.
                if assign.accepted >= 1 {
                    assert_eq!(&out[..], &[1.0, 1.0, 1.0, 1.0][..]);
                    any_explained = true;
                } else {
                    assert_eq!(&out[..], &[0.0, 0.0, 1.0, 1.0][..]);
                    any_unexplained = true;
                }
            }
        }
        assert!(any_explained, "{name}: some pair explains");
        assert!(any_unexplained, "{name}: some pair does not");
    }
}

/// The whole-batch path on the same fixtures: per-candidate counts and
/// weights equal the `ion_assign` verdicts summed over the selected peaks,
/// with `complete = 1` and the padding slot zeroed.
#[test]
fn batch_evidence_matches_ion_assign_sums() {
    let parent = graph_of("2-butanol").composition();
    let (adduct, u, ppm) = (1u16, 50u32, 100u32);
    let work_max = 65_536u32;
    let peaks = true_peaks(&parent, adduct);
    let n = peaks.len();
    let mut kept = vec![0u32; n * 3];
    let mut kept_f = vec![0.0f32; n * 2];
    for (p, &mz) in peaks.iter().enumerate() {
        kept[p * 3] = p as u32;
        kept[p * 3 + 1] = mz;
        kept_f[p * 2] = 1.0 - p as f32 * 0.01;
        kept_f[p * 2 + 1] = 1.0;
    }
    let meta = vec![n as u32, 0, 0, u32::from(adduct), ppm, 0, 0, 0];
    let spec = vec![u, 0];
    let (ev_peaks, ev_w) =
        evidence_peaks(&kept, &kept_f, &meta, &spec, 1, n, EVIDENCE_PEAKS);
    assert_f32_close(ev_w.iter().sum::<f32>(), 1.0, 1e-6, "weights sum to 1");
    let comps = candidates_of(&parent);
    let (cand, m) = pack_cand(&comps);
    let cand_ev = formula_evidence(
        &cand,
        &ev_peaks,
        &ev_w,
        &meta,
        &spec,
        1,
        m,
        EVIDENCE_PEAKS,
        work_max,
        u32::MAX,
    );
    let limits = IonLimits {
        work_max,
        kept: 4,
    };
    for (mi, comp) in comps.iter().enumerate() {
        let row = &cand_ev[mi * 4..mi * 4 + 4];
        let mut want_count = 0u32;
        let mut want_weight = 0.0f32;
        for s in 0..n {
            let pos = ev_peaks[s * 4] as usize;
            let mz = kept[pos * 3 + 1];
            if ion_assign(comp, adduct, mz, u, ppm, &limits)
                .expect("ion_assign runs")
                .accepted
                >= 1
            {
                want_count += 1;
                want_weight += ev_w[s];
            }
        }
        assert_eq!(row[0], want_count as f32, "candidate {mi} count");
        assert_f32_close(row[1], want_weight, 1e-6, "candidate {mi} weight");
        assert_eq!(row[2], n as f32, "candidate {mi} evidence size");
        assert_eq!(row[3], 1.0, "candidate {mi} complete");
    }
    assert_eq!(
        &cand_ev[(m - 1) * 4..m * 4],
        &[0.0; 4][..],
        "padding slot is zero"
    );
}

// ---------------------------------------------------------------------------
// E4: walk budget over the non-carbon vectors and its `ion_assign` relation.
// ---------------------------------------------------------------------------

/// Non-carbon radix product `J` of a composition (slots 1..8 in ELEMENTS
/// order: N, O, F, P, S, Cl, Br, I).
fn noncarbon_product(comp: &Composition) -> u64 {
    const SLOTS: [usize; 8] = [2, 3, 4, 5, 6, 7, 8, 9];
    SLOTS.iter().map(|&e| u64::from(comp[e]) + 1).product()
}

/// `ion_assign` oracle budget for `V = min(J, W)` non-carbon visits:
/// `V * (c[C] + 1) - 1`, exactly when it fits `u32` and is at least 1 (the
/// budget relation of [`formula_evidence_lane`]).
fn oracle_work_max(v: u64, c_c: u16) -> u32 {
    let w = v * (u64::from(c_c) + 1) - 1;
    assert!(
        w >= 1 && w <= u64::from(u32::MAX),
        "oracle budget {w} fits u32 and is >= 1"
    );
    w as u32
}

/// Explained-ness of one (candidate, peak) through the lane on a P = 1
/// evidence buffer. Returns the full row (explained, weight, n_ev,
/// complete).
fn lane_explains(
    comp: &Composition,
    t: u32,
    tol_p: u32,
    adduct: u16,
    u: u32,
    work_max: u32,
) -> [f32; 4] {
    let cand = pack_one(comp);
    lane_explains_raw_words(
        &cand,
        t,
        tol_p,
        adduct,
        u,
        work_max,
        false,
    )
}

/// Same as [`lane_explains`] over raw `cand` words (one slot), with the
/// slow twin on `slow`.
fn lane_explains_raw_words(
    cand: &[u32],
    t: u32,
    tol_p: u32,
    adduct: u16,
    u: u32,
    work_max: u32,
    slow: bool,
) -> [f32; 4] {
    let ev_peaks = vec![0, t, tol_p, 1];
    let ev_w = vec![1.0f32];
    let meta = vec![0u32, 0, 0, u32::from(adduct), 0, 0, 0, 0];
    let spec = vec![u, 0];
    let mut out = vec![0.0f32; 4];
    if slow {
        formula_evidence_lane_slow(
            cand, &ev_peaks, &ev_w, &meta, &spec, 0, 0, 1, 1, work_max, u32::MAX, &mut out,
        );
    } else {
        formula_evidence_lane(
            cand, &ev_peaks, &ev_w, &meta, &spec, 0, 0, 1, 1, work_max, u32::MAX, &mut out,
        );
    }
    [out[0], out[1], out[2], out[3]]
}

/// `J > W` gives `complete = 0` with the explained set of the first `V`
/// non-carbon visits: the lane equals
/// `ion_assign(work_max = V * (c[C] + 1) - 1)` on every (candidate, peak),
/// at `J == W + 1`, `J == W` and strictly inside, with and without carbon.
#[test]
fn truncated_budget_matches_ion_assign_prefix() {
    let (adduct, u, ppm) = (1u16, 50u32, 100u32);
    // N2 O1 H4: J = 3 * 2 = 6, c[C] = 0.
    let comp: Composition = [0, 4, 2, 1, 0, 0, 0, 0, 0, 0];
    assert_eq!(noncarbon_product(&comp), 6);
    // C3 N2 H4: J = 3, c[C] = 3.
    let comp_c: Composition = [3, 4, 2, 0, 0, 0, 0, 0, 0, 0];
    assert_eq!(noncarbon_product(&comp_c), 3);
    // Peaks: exact sub-composition ions (via chem::ion at shifts -1..=2) of
    // both candidates, plus a far decoy.
    let mut peaks = Vec::new();
    for c in [&comp, &comp_c] {
        for shift in [-1i32, 0, 1, 2] {
            if let Some(hyp) = ion(c, adduct, shift).expect("ion mass fits") {
                let mz = hyp.mz.checked_sub(549).expect("peak fits");
                peaks.push(mz);
            }
        }
    }
    peaks.push(10_000_000);
    peaks.sort_unstable();
    peaks.dedup();
    let mut compared = 0usize;
    for (c, w) in [
        (&comp, 5u32),   // J == W + 1
        (&comp, 6u32),   // J == W
        (&comp, 2u32),   // J > W
        (&comp_c, 2u32), // J == W + 1
        (&comp_c, 3u32), // J == W
        (&comp_c, 1u32), // J > W
    ] {
        let j = noncarbon_product(c);
        let v = j.min(w as u64);
        let oracle_w = oracle_work_max(v, c[0]);
        let limits = IonLimits {
            work_max: oracle_w,
            kept: 4,
        };
        for &mz in peaks.iter() {
            let (t, tol_p) = peak_row(mz, adduct, ppm);
            let out = lane_explains(c, t, tol_p, adduct, u, w);
            let assign = ion_assign(c, adduct, mz, u, ppm, &limits).expect("ion_assign runs");
            assert_eq!(
                out[0] == 1.0,
                assign.accepted >= 1,
                "J = {j} W = {w} candidate {c:?} peak {mz}"
            );
            assert_eq!(
                out[3] == 1.0,
                j <= w as u64,
                "J = {j} W = {w}: complete = (J <= W)"
            );
            assert_eq!(
                out[3] == 1.0,
                assign.status & ION_SEARCH_EXHAUSTED == 0,
                "J = {j} W = {w}: complete agrees with exhausted bit",
            );
            compared += 1;
        }
    }
    println!("E4-BUDGET compared={compared}");
}

/// Carbon-only candidates (`J = 1`) visit `n >= 1` only, and a flag-1
/// candidate with no heavy atom explains nothing but is complete.
#[test]
fn carbon_only_and_empty_candidates() {
    let (adduct, u, ppm) = (1u16, 50u32, 100u32);
    // C4 H2: J = 1, one non-carbon visit at any W >= 1.
    let c4: Composition = [4, 2, 0, 0, 0, 0, 0, 0, 0, 0];
    assert_eq!(noncarbon_product(&c4), 1);
    // Exact carbon ions n = 1..=4 at h = 0 explain; n = 5 does not.
    for n in 1u32..=5 {
        let t = n * 12_000_000;
        let mz = t - 549;
        let tol_p = peak_row(mz, adduct, ppm).1;
        let out = lane_explains(&c4, t, tol_p, adduct, u, 1);
        let want = n <= 4;
        let e = if want { 1.0 } else { 0.0 };
        assert_eq!(out, [e, e, 1.0, 1.0], "C4 carbon-only n = {n}");
        // Oracle: V = 1, c[C] = 4, so W' = 4 (the full walk).
        let assign = ion_assign(
            &c4,
            adduct,
            mz,
            u,
            ppm,
            &IonLimits {
                work_max: oracle_work_max(1, 4),
                kept: 4,
            },
        )
        .expect("ion_assign runs");
        assert_eq!(out[0] == 1.0, assign.accepted >= 1, "C4 n = {n} vs ion_assign");
    }
    // Flag-1 candidate with no heavy atom: nothing explained, complete.
    let empty: Composition = [0; 10];
    let out = lane_explains(&empty, 12_000_000, 119, adduct, u, 4096);
    assert_eq!(out, [0.0, 0.0, 1.0, 1.0], "empty candidate explains nothing");
    let assign = ion_assign(
        &empty,
        adduct,
        12_000_000 - 549,
        u,
        ppm,
        &IonLimits {
            work_max: 4096,
            kept: 4,
        },
    )
    .expect("ion_assign runs");
    assert_eq!(assign.accepted, 0, "empty parent has no accept");
    assert_ne!(
        assign.status & mamba3::models::ms2::ion::ION_UNAVAILABLE,
        0,
        "empty parent is unavailable"
    );
}

// ---------------------------------------------------------------------------
// E4: exhaustive equivalence against `EvidenceIndex` and `ion_assign`.
// ---------------------------------------------------------------------------

/// Random parent within the E4 caps (C 12, N 4, O 5, S 2, Cl 2, P 1, F 2,
/// with hydrogens).
fn random_parent_capped(rng: &mut StdRng) -> Composition {
    let mut c: Composition = [0; 10];
    c[0] = rng.random_range(0..=12);
    c[HYDROGEN] = rng.random_range(0..=12);
    c[2] = rng.random_range(0..=4);
    c[3] = rng.random_range(0..=5);
    c[4] = rng.random_range(0..=2);
    c[5] = rng.random_range(0..=1);
    c[6] = rng.random_range(0..=2);
    c[7] = rng.random_range(0..=2);
    if c.iter().all(|&n| n == 0) {
        c[0] = 1;
    }
    c
}

/// E4 exhaustive equivalence: with a complete walk (`W` above every `J`),
/// explained-ness per (candidate, peak) equals `EvidenceIndex::explains`
/// and `ion_assign(work_max = u32::MAX / 2).accepted >= 1` — per peak on
/// P = 1 lanes and as count/weight over full 32-peak spectra. Peaks sit ON
/// true sub-composition ions displaced by `-tol-3 ..= +tol+3` units, plus
/// decoys.
#[test]
fn exhaustive_equivalence_with_reference_index_and_ion_assign() {
    let ppm = 100u32;
    let mut rng = StdRng::seed_from_u64(0xE4u64);
    let work_max = 1u32 << 20;
    let ion_limits = IonLimits {
        work_max: u32::MAX / 2,
        kept: 4,
    };
    let mut compared = 0usize;
    let mut explained = 0usize;
    let mut unexplained = 0usize;
    for _ in 0..300 {
        let parent = random_parent_capped(&mut rng);
        for adduct in [1u16, 2] {
            for u in [0u32, 5, 40] {
                let h_pos: u32 = if adduct == 1 { 1 } else { 0 };
                let h_cap = u32::from(parent[HYDROGEN]) + h_pos + 2;
                // Four sub-vector draws; each yields an exact ion plus
                // displaced variants around the tolerance edge.
                let mut mzs: Vec<u32> = Vec::new();
                for _ in 0..4 {
                    const ORDER: [usize; 9] = [0, 2, 3, 4, 5, 6, 7, 8, 9];
                    let mut digits = [0u16; 9];
                    for (s, &e) in ORDER.iter().enumerate() {
                        digits[s] = rng.random_range(0..=parent[e]);
                    }
                    if digits.iter().all(|&d| d == 0) {
                        if parent[0] > 0 {
                            digits[0] = 1;
                        } else {
                            continue;
                        }
                    }
                    let h = rng.random_range(0..=h_cap.min(24));
                    let mut mass: u64 = 0;
                    for (s, &e) in ORDER.iter().enumerate() {
                        mass += u64::from(digits[s]) * u64::from(ELEMENTS[e].mass);
                    }
                    mass += u64::from(h) * u64::from(ELEMENTS[HYDROGEN].mass);
                    let Ok(mass) = u32::try_from(mass) else {
                        continue;
                    };
                    let mz = if adduct == 1 {
                        mass.checked_sub(549)
                    } else if mass <= u32::MAX - 549 {
                        Some(mass + 549)
                    } else {
                        None
                    };
                    let Some(mz) = mz.filter(|&v| v > 0) else {
                        continue;
                    };
                    let tol = tolerance(mz, ppm) as i64;
                    let shifts = [
                        0i64,
                        -(tol + 3),
                        tol + 3,
                        rng.random_range(-(tol + 3)..=tol + 3),
                    ];
                    for shift in shifts {
                        let peak = mz as i64 + shift;
                        if peak >= 1 && peak <= u64::from(u32::MAX) as i64 {
                            mzs.push(peak as u32);
                        }
                    }
                }
                for _ in 0..2 {
                    mzs.push(rng.random_range(5_000_000u32..150_000_000));
                }
                mzs.sort_unstable();
                mzs.dedup();
                let index = build_evidence_index(&parent, adduct, u, ppm)
                    .expect("index builds")
                    .expect("supported parent");
                // Per-peak P = 1 lanes against both oracles.
                for &mz in mzs.iter() {
                    let (t, tol_p) = peak_row(mz, adduct, ppm);
                    let out = lane_explains(&parent, t, tol_p, adduct, u, work_max);
                    assert_eq!(out[2], 1.0, "P = 1 evidence size");
                    assert_eq!(out[3], 1.0, "W = 2^20 completes J <= 1620");
                    let fast = out[0] == 1.0;
                    let slow = index.explains(mz).expect("explains runs");
                    let assign =
                        ion_assign(&parent, adduct, mz, u, ppm, &ion_limits).expect("ion runs");
                    let oracle = assign.accepted >= 1;
                    assert_eq!(
                        fast, slow,
                        "parent {parent:?} adduct {adduct} U {u} peak {mz}: twin vs index"
                    );
                    assert_eq!(
                        fast, oracle,
                        "parent {parent:?} adduct {adduct} U {u} peak {mz}: twin vs ion_assign"
                    );
                    compared += 1;
                    if fast {
                        explained += 1;
                    } else {
                        unexplained += 1;
                    }
                }
                // Full 32-peak spectrum: count and weight.
                let p = EVIDENCE_PEAKS;
                let mut ev_peaks = vec![u32::MAX, 0, 0, 0];
                ev_peaks = ev_peaks.repeat(p);
                let mut ev_w = vec![0.0f32; p];
                let mut want_count = 0u32;
                let mut want_weight = 0.0f32;
                for (s, &mz) in mzs.iter().take(p).enumerate() {
                    let (t, tol_p) = peak_row(mz, adduct, ppm);
                    ev_peaks[s * 4] = s as u32;
                    ev_peaks[s * 4 + 1] = t;
                    ev_peaks[s * 4 + 2] = tol_p;
                    ev_peaks[s * 4 + 3] = 1;
                    ev_w[s] = (s + 1) as f32;
                    if index.explains(mz).expect("explains runs") {
                        want_count += 1;
                        want_weight += ev_w[s];
                    }
                }
                let n_ev = mzs.len().min(p);
                let cand = pack_one(&parent);
                let meta = vec![0u32, 0, 0, u32::from(adduct), 0, 0, 0, 0];
                let spec = vec![u, 0];
                let mut out = vec![0.0f32; 4];
                formula_evidence_lane(
                    &cand, &ev_peaks, &ev_w, &meta, &spec, 0, 0, 1, p as u32, work_max, u32::MAX,
                    &mut out,
                );
                assert_eq!(out[0], want_count as f32, "32-peak count");
                assert_eq!(out[1], want_weight, "32-peak weight");
                assert_eq!(out[2], n_ev as f32, "32-peak evidence size");
                assert_eq!(out[3], 1.0, "32-peak complete");
            }
        }
    }
    println!("E4-EXHAUSTIVE compared={compared} explained={explained} unexplained={unexplained}");
    assert!(compared >= 20_000, "only {compared} (candidate, peak) pairs ran");
    assert!(explained >= 500, "only {explained} explained pairs");
    assert!(unexplained >= 500, "only {unexplained} unexplained pairs");
}

// ---------------------------------------------------------------------------
// E4: fast hydrogen range against the slow range.
// ---------------------------------------------------------------------------

/// The wrapped-range decision of [`formula_evidence_lane`], replicated here
/// (item 1 rule) to tally which range each comparison took: the wrapped
/// ranges run exactly when `(s_max + 1) * (2 tol / 7,825 + 2) <= h_cap + 1`.
fn fast_precondition(h_hi_abs: u32, tol_p: u32) -> bool {
    let s_max = (u64::from(h_hi_abs) * 7_825 + 2 * u64::from(tol_p)) / 1_000_000;
    let per_s = (2 * u64::from(tol_p)) / 7_825 + 2;
    (s_max + 1) * per_s <= u64::from(h_hi_abs) + 1
}

/// Hydrogen cap of a composition candidate under an adduct (the lane's
/// `h_hi_abs`; small counts never clamp at 65535).
fn lane_h_cap(comp: &Composition, adduct: u16) -> u32 {
    let h_pos: u32 = if adduct == 1 { 1 } else { 0 };
    (u32::from(comp[HYDROGEN]) + h_pos + 2).min(u32::from(u16::MAX))
}

/// E4 fast path against slow path: the `#[doc(hidden)]` slow twin agrees
/// with the fast lane on the random set plus adversarial ranges —
/// tolerances from 0 up to the largest for which the fast precondition
/// still holds and the first value for which it fails; `h_cap` at 0, 1, 99,
/// 127, 128 and values making `7,825 h_cap + 2 tol` exactly `999,999` and
/// exactly `1,000,000`; `Rm` at 0, at `999,999`, just below and above a
/// multiple of 7,825; `t + tol < m'`; `m'` near `u32::MAX`.
#[test]
fn fast_path_equals_slow_path() {
    let ppm = 100u32;
    let mut rng = StdRng::seed_from_u64(0xFA57u64);
    let work_max = 1u32 << 20;
    let mut compared = 0usize;
    let mut fast_taken = 0usize;
    let mut slow_taken = 0usize;
    // Random set: same draws as the exhaustive test (fewer parents); every
    // pair through both twins.
    for _ in 0..120 {
        let parent = random_parent_capped(&mut rng);
        for adduct in [1u16, 2] {
            for u in [0u32, 5, 40] {
                let h_pos: u32 = if adduct == 1 { 1 } else { 0 };
                let h_cap = u32::from(parent[HYDROGEN]) + h_pos + 2;
                let mut mzs: Vec<u32> = Vec::new();
                for _ in 0..3 {
                    const ORDER: [usize; 9] = [0, 2, 3, 4, 5, 6, 7, 8, 9];
                    let mut digits = [0u16; 9];
                    for (s, &e) in ORDER.iter().enumerate() {
                        digits[s] = rng.random_range(0..=parent[e]);
                    }
                    if digits.iter().all(|&d| d == 0) {
                        if parent[0] > 0 {
                            digits[0] = 1;
                        } else {
                            continue;
                        }
                    }
                    let h = rng.random_range(0..=h_cap.min(24));
                    let mut mass: u64 = 0;
                    for (s, &e) in ORDER.iter().enumerate() {
                        mass += u64::from(digits[s]) * u64::from(ELEMENTS[e].mass);
                    }
                    mass += u64::from(h) * u64::from(ELEMENTS[HYDROGEN].mass);
                    let Ok(mass) = u32::try_from(mass) else {
                        continue;
                    };
                    let mz = if adduct == 1 {
                        mass.checked_sub(549)
                    } else if mass <= u32::MAX - 549 {
                        Some(mass + 549)
                    } else {
                        None
                    };
                    let Some(mz) = mz.filter(|&v| v > 0) else {
                        continue;
                    };
                    let tol = tolerance(mz, ppm) as i64;
                    for shift in [0i64, -(tol + 3), tol + 3] {
                        let peak = mz as i64 + shift;
                        if peak >= 1 && peak <= u64::from(u32::MAX) as i64 {
                            mzs.push(peak as u32);
                        }
                    }
                }
                mzs.push(rng.random_range(5_000_000u32..150_000_000));
                mzs.sort_unstable();
                mzs.dedup();
                for &mz in mzs.iter() {
                    let (t, tol_p) = peak_row(mz, adduct, ppm);
                    let cand = pack_one(&parent);
                    let fast = lane_explains_raw_words(&cand, t, tol_p, adduct, u, work_max, false);
                    let slow = lane_explains_raw_words(&cand, t, tol_p, adduct, u, work_max, true);
                    assert_eq!(
                        fast, slow,
                        "parent {parent:?} adduct {adduct} U {u} peak {mz}"
                    );
                    if fast_precondition(lane_h_cap(&parent, adduct), tol_p) {
                        fast_taken += 1;
                    } else {
                        slow_taken += 1;
                    }
                    compared += 1;
                }
            }
        }
    }
    // Adversarial hand cases: (cand words, t, tol, adduct, U).
    let mut cases: Vec<(Vec<u32>, u32, u32, u16, u32)> = Vec::new();
    let words_of = |c: &Composition| -> Vec<u32> {
        let mut w = vec![0u32; 13];
        for e in 0..10 {
            w[e] = u32::from(c[e]);
        }
        w[10] = composition_mass(c).expect("mass fits");
        w[11] = 1;
        w[12] = u32::MAX;
        w
    };
    // Tolerances 0 up to the largest holding value and the first failing
    // one (C4 H6, adduct 1: h_cap = 9; 7,825 * 9 + 2 * 464,787 = 999,999).
    let c4h6: Composition = [4, 6, 0, 0, 0, 0, 0, 0, 0, 0];
    let t_exact = 4 * 12_000_000 + 6 * 1_007_825;
    for tol in [0u32, 1, 2, 100, 464_787, 464_788] {
        cases.push((words_of(&c4h6), t_exact, tol, 1, 0));
    }
    // h_cap at 99, 127, 128 (H = 96, 124, 125 under adduct 1).
    for h in [96u16, 124, 125] {
        let comp: Composition = [2, h, 0, 0, 0, 0, 0, 0, 0, 0];
        let t = 2 * 12_000_000 + 90 * 1_007_825;
        for tol in [0u32, 100] {
            cases.push((words_of(&comp), t, tol, 1, 0));
        }
    }
    // 7,825 * 127 + 2 * tol exactly 999,999 (tol = 3,112) and 1,000,001
    // (tol = 3,113); exactly 1,000,000 (h_cap = 64, tol = 249,600) and
    // 999,998 (tol = 249,599).
    let c2h124: Composition = [2, 124, 0, 0, 0, 0, 0, 0, 0, 0];
    let t127 = 2 * 12_000_000 + 60 * 1_007_825;
    for tol in [3_112u32, 3_113] {
        cases.push((words_of(&c2h124), t127, tol, 1, 0));
    }
    let c2h61: Composition = [2, 61, 0, 0, 0, 0, 0, 0, 0, 0];
    let t64 = 2 * 12_000_000 + 60 * 1_007_825;
    for tol in [249_599u32, 249_600] {
        cases.push((words_of(&c2h61), t64, tol, 1, 0));
    }
    // h_cap at 0 and 1 via wrapping hydrogen counts (raw words; E_ion wraps
    // too — agreement only, on both sides of the gate).
    for h_word in [u32::MAX - 2, u32::MAX - 1] {
        let mut w = vec![0u32; 13];
        w[0] = 2;
        w[1] = h_word;
        w[10] = 24_000_000;
        w[11] = 1;
        w[12] = u32::MAX;
        cases.push((w, 24_000_000, 100, 1, 0));
    }
    // Rm at 0, at 999,999, just below and above multiples of 7,825
    // (C1 H2, adduct 1, m' = 0, tol = 50: Rm = (t + 50) mod 1,000,000).
    let c1h2: Composition = [1, 2, 0, 0, 0, 0, 0, 0, 0, 0];
    for t in [
        999_950u32, // Rm = 0
        999_949,    // Rm = 999,999
        15_600,     // Rm = 15,650 = 2 * 7,825
        15_599,     // Rm = 15,649, just below
        15_601,     // Rm = 15,651, just above
        7_775,      // Rm = 7,825
        7_774,      // Rm = 7,824, just below
        7_776,      // Rm = 7,826, just above
    ] {
        cases.push((words_of(&c1h2), t, 50, 1, 0));
    }
    // t + tol < m' (S1: the j = 1 visit has m' = 31,972,071).
    let s1: Composition = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0];
    let (t_s, tol_s) = peak_row(31_972_071 - 549, 1, ppm);
    cases.push((words_of(&s1), t_s, tol_s, 1, 0));
    cases.push((words_of(&s1), 1_000_000, 100, 1, 0));
    // m' near u32::MAX (I33: max m' = 4,187,847,576).
    let i33: Composition = [0, 0, 0, 0, 0, 0, 0, 0, 0, 33];
    let (t_i, tol_i) = peak_row(4_187_847_576 - 549, 1, ppm);
    cases.push((words_of(&i33), t_i, tol_i, 1, 0));
    cases.push((words_of(&i33), u32::MAX - 100, 50, 1, 0));
    // Adduct 2 mirror of one exact case.
    let mz2 = t_exact + 549;
    let (t2, tol2) = peak_row(mz2, 2, ppm);
    assert_eq!(t2, t_exact);
    cases.push((words_of(&c4h6), t2, tol2, 2, 0));
    for (w, t, tol, adduct, u) in cases.iter() {
        let fast = lane_explains_raw_words(w, *t, *tol, *adduct, *u, work_max, false);
        let slow = lane_explains_raw_words(w, *t, *tol, *adduct, *u, work_max, true);
        assert_eq!(fast, slow, "t = {t} tol = {tol} adduct = {adduct}");
        // Tally the range the fast lane took (same guard as the lane).
        let h_pos: u32 = if *adduct == 1 { 1 } else { 0 };
        let h_cap = w[1].wrapping_add(h_pos).wrapping_add(2).min(u32::from(u16::MAX));
        if fast_precondition(h_cap, *tol) {
            fast_taken += 1;
        } else {
            slow_taken += 1;
        }
        compared += 1;
    }
    println!("E4-FAST-SLOW compared={compared} fast_taken={fast_taken} slow_taken={slow_taken}");
    assert!(compared >= 5_000, "only {compared} fast/slow pairs ran");
    assert!(fast_taken >= 1_000, "fast range never taken");
    assert!(slow_taken >= 5, "slow range never taken");
}

// ---------------------------------------------------------------------------
// Peak selection: order, ties, overflow, empty spectra.
// ---------------------------------------------------------------------------

/// Independent `u32` tolerance (same algorithm, `u64` headroom).
fn tol_u64(mz: u32, ppm: u32) -> u32 {
    let hi = u64::from(mz / 10_000);
    let lo = u64::from(mz % 10_000);
    (hi * u64::from(ppm) / 1000
        + ((hi * u64::from(ppm) % 1000) * 10_000 + lo * u64::from(ppm)) / 10_000_000)
        as u32
}

/// Selection order with ties and more than `P` peaks, plus a NaN that is
/// never selected: expected positions and weights recomputed here with a
/// plain sort and `f64` arithmetic, not with the twin.
#[test]
fn peak_selection_order_ties_overflow_nan() {
    let p = EVIDENCE_PEAKS;
    let n = 40usize;
    assert!(n > p, "more peaks than slots");
    let (adduct, u, ppm) = (1u16, 50u32, 100u32);
    // Intensities with ties; position 37 is NaN and must never be picked.
    let inten: Vec<f32> = (0..n)
        .map(|i| if i == 37 { f32::NAN } else { ((i * 7) % 13) as f32 })
        .collect();
    let mz: Vec<u32> = (0..n).map(|i| 100_000_000 + i as u32 * 1000).collect();
    // Second spectrum: only 3 eligible peaks, the rest is padding.
    let n2inten = [5.0f32, 5.0, 2.0];
    let mut kept = vec![0u32; 2 * n * 3];
    let mut kept_f = vec![0.0f32; 2 * n * 2];
    for i in 0..n {
        kept[i * 3] = i as u32;
        kept[i * 3 + 1] = mz[i];
        kept_f[i * 2] = inten[i];
        kept_f[i * 2 + 1] = 1.0;
        kept[(n + i) * 3] = i as u32;
        kept[(n + i) * 3 + 1] = if i < 3 { 200_000_000 + i as u32 } else { 0 };
        kept_f[(n + i) * 2] = if i < 3 { n2inten[i] } else { 0.0 };
        kept_f[(n + i) * 2 + 1] = 1.0;
    }
    let meta = vec![
        n as u32, 0, 0, u32::from(adduct), ppm, 0, 0, 0, //
        3, 0, 0, u32::from(adduct), ppm, 0, 0, 0,
    ];
    let spec = vec![u, 0, u, 0];
    let (ev_peaks, ev_w) = evidence_peaks(&kept, &kept_f, &meta, &spec, 2, n, p);
    // Expected order, computed here: NaN filtered out, then intensity
    // descending, position ascending, top `p`.
    let mut order: Vec<usize> = (0..n).filter(|&i| i != 37).collect();
    order.sort_by(|&a, &b| {
        inten[b]
            .partial_cmp(&inten[a])
            .expect("no NaN left")
            .then_with(|| a.cmp(&b))
    });
    let want: &[usize] = &order[..p];
    assert!(!want.contains(&37), "NaN never selected");
    for (s, &pos) in want.iter().enumerate() {
        let row = &ev_peaks[s * 4..s * 4 + 4];
        assert_eq!(row[0], pos as u32, "slot {s} position");
        assert_eq!(row[1], mz[pos] + 549, "slot {s} target");
        assert_eq!(row[2], tol_u64(mz[pos], ppm), "slot {s} tolerance");
        assert_eq!(row[3], 1, "slot {s} valid");
    }
    let sum: f64 = want.iter().map(|&i| f64::from(inten[i])).sum();
    for (s, &pos) in want.iter().enumerate() {
        let want_w = (f64::from(inten[pos]) / sum) as f32;
        assert_f32_close(ev_w[s], want_w, 1e-6, "slot {s} weight");
    }
    // Spectrum 2: three valid rows in (5, 5, 2) order with ties by position,
    // then exact padding.
    let base = p * 4;
    assert_eq!(&ev_peaks[base..base + 4], &[0, 200_000_000 + 549, tol_u64(200_000_000, ppm), 1]);
    assert_eq!(&ev_peaks[base + 4..base + 8], &[1, 200_000_001 + 549, tol_u64(200_000_001, ppm), 1]);
    assert_eq!(&ev_peaks[base + 8..base + 12], &[2, 200_000_002 + 549, tol_u64(200_000_002, ppm), 1]);
    for s in 3..p {
        assert_eq!(
            &ev_peaks[base + s * 4..base + s * 4 + 4],
            &[u32::MAX, 0, 0, 0][..],
            "spectrum 2 slot {s} padding"
        );
        assert_eq!(ev_w[p + s], 0.0, "spectrum 2 weight {s} zero");
    }
    let wsum: f32 = ev_w[p..p + 3].iter().sum();
    assert_f32_close(wsum, 1.0, 1e-6, "spectrum 2 weights sum to 1");
    assert_f32_close(ev_w[p], 5.0 / 12.0, 1e-6, "tied first weight");
    assert_f32_close(ev_w[p + 1], 5.0 / 12.0, 1e-6, "tied second weight");
    assert_f32_close(ev_w[p + 2], 2.0 / 12.0, 1e-6, "third weight");
}

/// A spectrum with no eligible peak selects nothing: exact padding rows and
/// zero weights, and its candidates carry `n_ev = 0`.
#[test]
fn spectrum_without_eligible_peak() {
    let (n, p) = (4usize, 8usize);
    let kept = vec![0u32; n * 3];
    let kept_f = vec![1.0f32; n * 2];
    let meta = vec![0u32, 0, 0, 1, 100, 0, 0, 0];
    let spec = vec![50u32, 0];
    let (ev_peaks, ev_w) = evidence_peaks(&kept, &kept_f, &meta, &spec, 1, n, p);
    for s in 0..p {
        assert_eq!(
            &ev_peaks[s * 4..s * 4 + 4],
            &[u32::MAX, 0, 0, 0][..],
            "slot {s} padding"
        );
        assert_eq!(ev_w[s], 0.0, "slot {s} weight zero");
    }
    // One real candidate (C1: one visit, complete) and one padding slot.
    let one_c: Composition = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let (cand, m) = pack_cand(&[one_c]);
    let cand_ev = formula_evidence(&cand, &ev_peaks, &ev_w, &meta, &spec, 1, m, p, 4096, u32::MAX);
    assert_eq!(&cand_ev[0..4], &[0.0, 0.0, 0.0, 1.0], "real slot");
    assert_eq!(&cand_ev[m * 4 - 4..m * 4], &[0.0; 4][..], "padding slot");
}

/// Unknown m/z uncertainty and unknown adduct each disable selection: exact
/// padding rows and zero weights.
#[test]
fn unknown_uncertainty_or_adduct_selects_nothing() {
    let (n, p) = (4usize, 6usize);
    let mut kept = vec![0u32; 2 * n * 3];
    let mut kept_f = vec![0.0f32; 2 * n * 2];
    for b in 0..2 {
        for i in 0..n {
            kept[(b * n + i) * 3] = i as u32;
            kept[(b * n + i) * 3 + 1] = 150_000_000 + i as u32;
            kept_f[(b * n + i) * 2] = 1.0;
            kept_f[(b * n + i) * 2 + 1] = 1.0;
        }
    }
    // Spectrum 0: unknown U. Spectrum 1: unknown adduct.
    let meta = vec![
        n as u32, 0, 0, 1, 100, 0, 0, 0, //
        n as u32, 0, 0, 0, 100, 0, 0, 0,
    ];
    let spec = vec![u32::MAX, 0, 50u32, 0];
    let (ev_peaks, ev_w) = evidence_peaks(&kept, &kept_f, &meta, &spec, 2, n, p);
    for b in 0..2 {
        for s in 0..p {
            assert_eq!(
                &ev_peaks[(b * p + s) * 4..(b * p + s) * 4 + 4],
                &[u32::MAX, 0, 0, 0][..],
                "spectrum {b} slot {s} padding"
            );
            assert_eq!(ev_w[b * p + s], 0.0, "spectrum {b} weight {s} zero");
        }
    }
}

// ---------------------------------------------------------------------------
// Feature rows: hand-computed expectations.
// ---------------------------------------------------------------------------

/// Resident log table, computed inline (not via the twin).
fn log_table_inline() -> Vec<f32> {
    (0..1024).map(|n| (1.0 + n as f32).ln()).collect()
}

/// Feature rows for padding, below/above matches, the `4w` clamp and
/// unknown precursor uncertainty, adduct and parent mass: expectations from
/// independent `u64`/`f64` arithmetic.
#[test]
fn feature_rows_hand_computed() {
    const H_NET: u64 = 1_007_825 - 549;
    let log = log_table_inline();
    // precursor 200M, adduct 1, frag ppm 100, precursor ppm 100, unc 500.
    let prec = 200_000_000u32;
    let (unc, ppm_pre) = (500u32, 100u32);
    let tol = tol_u64(prec, ppm_pre);
    assert_eq!(tol, 2000, "hand tolerance");
    let w = u64::from(tol) + u64::from(unc);
    let m_p = u64::from(prec) - H_NET;
    // Slots: padding (with garbage evidence), exact, below, above, clamp.
    let counts: Composition = [3, 8, 0, 1, 0, 0, 0, 0, 0, 0];
    let mass_of = |m: u64| m as u32;
    let mut cand = vec![0u32; 5 * 13];
    // Slot 0: padding, but a garbage evidence row to prove zeroing.
    cand[0 * 13 + 12] = u32::MAX;
    // Slot 1: exact match.
    for e in 0..10 {
        cand[1 * 13 + e] = u32::from(counts[e]);
    }
    cand[1 * 13 + 10] = mass_of(m_p);
    cand[1 * 13 + 11] = 1;
    cand[1 * 13 + 12] = 7;
    // Slot 2: 5000 below.
    for e in 0..10 {
        cand[2 * 13 + e] = u32::from(counts[e]);
    }
    cand[2 * 13 + 10] = mass_of(m_p - 5000);
    cand[2 * 13 + 11] = 1;
    cand[2 * 13 + 12] = 7;
    // Slot 3: 2500 above.
    for e in 0..10 {
        cand[3 * 13 + e] = u32::from(counts[e]);
    }
    cand[3 * 13 + 10] = mass_of(m_p + 2500);
    cand[3 * 13 + 11] = 1;
    cand[3 * 13 + 12] = 7;
    // Slot 4: far above, into the 4w clamp.
    for e in 0..10 {
        cand[4 * 13 + e] = u32::from(counts[e]);
    }
    cand[4 * 13 + 10] = mass_of(m_p + 100_000_000);
    cand[4 * 13 + 11] = 1;
    cand[4 * 13 + 12] = 7;
    let cand_ev = vec![
        3.0, 0.5, 5.0, 1.0, // padding-slot garbage
        2.0, 0.75, 4.0, 1.0, //
        2.0, 0.75, 4.0, 1.0, //
        0.0, 0.0, 0.0, 0.0, //
        1.0, 0.25, 2.0, 0.0,
    ];
    let meta = vec![0u32, prec, unc, 1, 100, ppm_pre, 0, 0];
    let out = formula_features(&cand, &cand_ev, &meta, &log, 1, 5);
    // Slot 0: exact zeros despite the garbage evidence row.
    assert_eq!(&out[0..16], &[0.0; 16][..], "padding row");
    // Shared count features, recomputed here.
    let mut want_counts = [0.0f32; 10];
    for e in 0..10 {
        want_counts[e] = (1.0 + f32::from(counts[e])).ln();
    }
    // Slot 1: exact mass match.
    assert_eq!(&out[16..26], &want_counts[..], "slot 1 counts");
    assert_f32_close(out[26], 0.0, 1e-6, "slot 1 abs");
    assert!(out[27] == 0.0 && out[27].is_sign_positive(), "slot 1 signed +0");
    assert_f32_close(out[28], 0.5, 1e-6, "slot 1 fraction");
    assert_f32_close(out[29], 0.75, 1e-6, "slot 1 weight");
    assert_f32_close(out[30], (1.0 + 2.0f32).ln(), 1e-6, "slot 1 log");
    assert_eq!(out[31], 1.0, "slot 1 complete");
    // Slot 2: 5000 below over w = 2500.
    let d2 = 5000.0f64 / w as f64;
    assert_f32_close(out[42], d2 as f32, 1e-6, "slot 2 abs");
    assert_f32_close(out[43], -(d2 as f32), 1e-6, "slot 2 signed");
    // Slot 3: 2500 above over w = 2500.
    assert_f32_close(out[58], 1.0, 1e-6, "slot 3 abs");
    assert_f32_close(out[59], 1.0, 1e-6, "slot 3 signed");
    assert_f32_close(out[60], 0.0, 1e-6, "slot 3 fraction of none");
    // Slot 4: clamped to 4w, so abs is 4 within rounding.
    assert_f32_close(out[74], 4.0, 1e-6, "slot 4 clamped abs");
    assert_f32_close(out[75], 4.0, 1e-6, "slot 4 clamped signed");
    assert_f32_close(out[76], 0.5, 1e-6, "slot 4 fraction");
    assert_eq!(out[79], 0.0, "slot 4 incomplete");
    // Unknown precursor uncertainty zeroes the residual pair only.
    let mut meta_unk = meta.clone();
    meta_unk[2] = u32::MAX;
    let mut lane_out = vec![0.0f32; 5 * 16];
    formula_features_lane(&cand, &cand_ev, &meta_unk, &log, 0, 2, 5, &mut lane_out);
    let lane_row = |lane_out: &[f32]| lane_out[2 * 16..3 * 16].to_vec();
    let row2 = lane_row(&lane_out);
    assert_eq!(row2[10], 0.0, "unknown unc abs");
    assert_eq!(row2[11], 0.0, "unknown unc signed");
    assert_f32_close(row2[12], 0.5, 1e-6, "unknown unc fraction kept");
    // Unknown adduct zeroes the residual pair only.
    meta_unk[2] = unc;
    meta_unk[3] = 0;
    formula_features_lane(&cand, &cand_ev, &meta_unk, &log, 0, 2, 5, &mut lane_out);
    let row2 = lane_row(&lane_out);
    assert_eq!(row2[10], 0.0, "unknown adduct abs");
    assert_eq!(row2[11], 0.0, "unknown adduct signed");
    // Parent mass leaving u32 zeroes the residual pair (both adduct sides).
    meta_unk[3] = 1;
    meta_unk[1] = 1000; // below h_net: adduct-1 subtraction underflows.
    formula_features_lane(&cand, &cand_ev, &meta_unk, &log, 0, 2, 5, &mut lane_out);
    let row2 = lane_row(&lane_out);
    assert_eq!(row2[10], 0.0, "adduct 1 overflow abs");
    assert_eq!(row2[11], 0.0, "adduct 1 overflow signed");
    meta_unk[3] = 2;
    meta_unk[1] = u32::MAX - 100; // above MAX - h_net: adduct-2 addition overflows.
    formula_features_lane(&cand, &cand_ev, &meta_unk, &log, 0, 2, 5, &mut lane_out);
    let row2 = lane_row(&lane_out);
    assert_eq!(row2[10], 0.0, "adduct 2 overflow abs");
    assert_eq!(row2[11], 0.0, "adduct 2 overflow signed");
    // The hardware present in this file is real: element masses resolve.
    assert_eq!(ELEMENTS[HYDROGEN].mass, 1_007_825);
}

// ---------------------------------------------------------------------------
// E1F: selection scope rule (finding 2), degenerate targets, weights.
// ---------------------------------------------------------------------------

/// Reviewer's finding-2 example, pinned as the intended rule: selection
/// applies the candidate-independent part `tol_p + U <= m_H`, while the
/// candidate-dependent `+ E_ion(c)` scope is decided in kernel 2.
///
/// Positions: m/z 50,000,000 (intensity 1.0) and 49,000,000 (0.5), adduct 1,
/// ppm-tenths 100, `U = 1,007,325`. Both pass the candidate-independent
/// gate (`500 + U = m_H` for position 0), so position 0 IS selected; for
/// candidate C1 (`[1,0,...]`, `E_ion = ceil(520/1000) = 1`) position 0 has
/// `half_p = 1,007,826 > m_H` and is unexplained for C1 while still counting
/// in `n_ev`. A candidate-independent in-scope peak (small m/z, small U)
/// can be explained by C1.
#[test]
fn selection_omits_e_ion_by_design() {
    use mamba3::models::ms2::formula_evidence::formula_evidence_lane;

    let m_h = ELEMENTS[HYDROGEN].mass;
    assert_eq!(m_h, 1_007_825);
    // Selection side: B=1, N=2, P=1.
    let kept = vec![0u32, 50_000_000, 0, 1, 49_000_000, 0];
    let kept_f = vec![1.0f32, 1.0, 0.5, 1.0];
    let u = 1_007_325u32;
    let ppm = 100u32;
    let meta = vec![2u32, 0, 0, 1, ppm, 0, 0, 0];
    let spec = vec![u, 0];
    // Independent tolerances (u64 headroom).
    let tol0 = tol_u64(50_000_000, ppm);
    let tol1 = tol_u64(49_000_000, ppm);
    assert_eq!(tol0, 500, "hand tolerance position 0");
    assert_eq!(tol1, 490, "hand tolerance position 1");
    assert_eq!(tol0 as u64 + u as u64, m_h as u64, "pos0 at the gate");
    assert!((tol1 as u64 + u as u64) < m_h as u64, "pos1 inside");
    let (ev_peaks, ev_w) = evidence_peaks(&kept, &kept_f, &meta, &spec, 1, 2, 1);
    // Position 0 IS selected (higher intensity, both eligible).
    assert_eq!(&ev_peaks[..], &[0, 50_000_000 + 549, 500, 1][..]);
    assert_eq!(ev_w, vec![1.0]);
    // Kernel-2 side: candidate C1 = single carbon.
    let c1: Composition = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let cand = pack_one(&c1);
    let mut out = vec![0.0f32; 4];
    formula_evidence_lane(&cand, &ev_peaks, &ev_w, &meta, &spec, 0, 0, 1, 1, 65_536, u32::MAX, &mut out);
    // Out-of-scope for C1: unexplained, but still counts in n_ev.
    assert_eq!(&out[..], &[0.0, 0.0, 1.0, 1.0][..]);
    // Independent oracle: ion_assign with parent C1 reports no accept.
    let assign = ion_assign(
        &c1,
        1,
        50_000_000,
        u,
        ppm,
        &IonLimits { work_max: 65_536, kept: 4 },
    )
    .expect("ion_assign runs");
    assert_eq!(assign.accepted, 0, "C1 does not explain position 0");
    // Contrast: a candidate-independent in-scope peak CAN be explained by
    // C1 (target 12,000,000 at h=0, small U).
    let mz_small = 12_000_000u32 - 549;
    let (t_small, tol_small) = peak_row(mz_small, 1, ppm);
    assert_eq!(t_small, 12_000_000);
    let u2 = 50u32;
    let meta2 = vec![1u32, 0, 0, 1, ppm, 0, 0, 0];
    let spec2 = vec![u2, 0];
    let ev2 = vec![0, t_small, tol_small, 1];
    let mut out2 = vec![0.0f32; 4];
    formula_evidence_lane(&cand, &ev2, &[1.0], &meta2, &spec2, 0, 0, 1, 1, 65_536, u32::MAX, &mut out2);
    assert_eq!(&out2[..], &[1.0, 1.0, 1.0, 1.0][..]);
    let assign2 = ion_assign(
        &c1,
        1,
        mz_small,
        u2,
        ppm,
        &IonLimits { work_max: 65_536, kept: 4 },
    )
    .expect("ion_assign runs");
    assert!(assign2.accepted >= 1, "C1 explains the in-scope peak");
}

/// Actual target underflow (adduct 2, `mz < 549`) and overflow (adduct 1,
/// `mz > u32::MAX - 549`) are ineligible, even with `ppm = 0` (tolerance 0)
/// and a tiny `U` so the scope gate alone would pass.
#[test]
fn selection_target_underflow_overflow_ineligible() {
    // Adduct 1: valid, overflow, zero-m/z.
    let kept1 = vec![0u32, 100_000_000, 0, 1, u32::MAX - 100, 0, 2, 0, 0];
    let kept_f1 = vec![1.0f32, 1.0, 0.9, 1.0, 0.8, 1.0];
    let meta1 = vec![3u32, 0, 0, 1, 0, 0, 0, 0];
    let spec1 = vec![50u32, 0];
    let (peaks1, w1) = evidence_peaks(&kept1, &kept_f1, &meta1, &spec1, 1, 3, 3);
    // Only position 0 selected; the overflow target (mz + 549 wraps) and
    // the zero m/z never appear.
    assert_eq!(&peaks1[0..4], &[0, 100_000_000 + 549, 0, 1][..]);
    for s in 1..3 {
        assert_eq!(&peaks1[s * 4..s * 4 + 4], &[u32::MAX, 0, 0, 0][..], "slot {s} padding");
        assert_eq!(w1[s], 0.0);
    }
    assert_eq!(w1[0], 1.0);
    // Adduct 2: underflow (mz = 548 < 549), valid, zero-m/z.
    let kept2 = vec![0u32, 548, 0, 1, 100_000_000, 0, 2, 0, 0];
    let kept_f2 = vec![1.0f32, 1.0, 0.9, 1.0, 0.8, 1.0];
    let meta2 = vec![3u32, 0, 0, 2, 0, 0, 0, 0];
    let spec2 = vec![50u32, 0];
    let (peaks2, w2) = evidence_peaks(&kept2, &kept_f2, &meta2, &spec2, 1, 3, 3);
    assert_eq!(&peaks2[0..4], &[1, 100_000_000 - 549, 0, 1][..]);
    for s in 1..3 {
        assert_eq!(&peaks2[s * 4..s * 4 + 4], &[u32::MAX, 0, 0, 0][..], "slot {s} padding");
        assert_eq!(w2[s], 0.0);
    }
    assert_eq!(w2[0], 1.0);
}

/// A saturated half-width (`U = u32::MAX - 1`) selects nothing: every
/// `tol_p + U` saturates to `u32::MAX > m_H`.
#[test]
fn selection_saturated_half_width_selects_nothing() {
    let kept = vec![0u32, 100_000_000, 0, 1, 200_000_000, 0];
    let kept_f = vec![1.0f32, 1.0, 0.5, 1.0];
    let meta = vec![2u32, 0, 0, 1, 100, 0, 0, 0];
    let spec = vec![u32::MAX - 1, 0];
    let (peaks, w) = evidence_peaks(&kept, &kept_f, &meta, &spec, 1, 2, 2);
    for s in 0..2 {
        assert_eq!(&peaks[s * 4..s * 4 + 4], &[u32::MAX, 0, 0, 0][..]);
        assert_eq!(w[s], 0.0);
    }
}

/// All-zero intensities: the eligible rows are still selected (valid flag
/// set) but every weight is 0 (the sum is not positive).
#[test]
fn selection_all_zero_intensities_valid_rows_zero_weights() {
    let kept = vec![0u32, 100_000_000, 0, 1, 100_001_000, 0, 2, 100_002_000, 0];
    let kept_f = vec![0.0f32, 1.0, 0.0, 1.0, 0.0, 1.0];
    let meta = vec![3u32, 0, 0, 1, 100, 0, 0, 0];
    let spec = vec![50u32, 0];
    let (peaks, w) = evidence_peaks(&kept, &kept_f, &meta, &spec, 1, 3, 2);
    // Slot order with ties by position: positions 0 then 1.
    assert_eq!(&peaks[0..4], &[0, 100_000_000 + 549, tol_u64(100_000_000, 100), 1][..]);
    assert_eq!(&peaks[4..8], &[1, 100_001_000 + 549, tol_u64(100_001_000, 100), 1][..]);
    assert_eq!(w, vec![0.0, 0.0]);
}

/// Order-sensitive `f32` weight sum: intensities whose `f32` sum depends on
/// the accumulation order. Expected weights come from summing in slot order
/// (selection order) with `f32` arithmetic, not from the twin.
#[test]
fn selection_order_sensitive_weight_sum() {
    // 16,777,216 + 1 rounds back to 16,777,216 in f32, so the slot-order
    // sum ((M + 1) + 1) is 16,777,216 while (1 + 1) + M is 16,777,218.
    let big = 16_777_216.0f32;
    let kept = vec![0u32, 100_000_000, 0, 1, 100_001_000, 0, 2, 100_002_000, 0];
    let kept_f = vec![big, 1.0, 1.0, 1.0, 1.0, 1.0];
    let meta = vec![3u32, 0, 0, 1, 0, 0, 0, 0];
    let spec = vec![50u32, 0];
    let (peaks, w) = evidence_peaks(&kept, &kept_f, &meta, &spec, 1, 3, 3);
    assert_eq!(peaks[0], 0);
    assert_eq!(peaks[4], 1);
    assert_eq!(peaks[8], 2);
    // Slot-order f32 sum, computed here.
    let mut sum = 0.0f32;
    sum += big;
    sum += 1.0f32;
    sum += 1.0f32;
    assert_eq!(sum.to_bits(), 16_777_216.0f32.to_bits(), "slot-order sum pins the rounding");
    let alt = (1.0f32 + 1.0f32) + big;
    assert_ne!(sum.to_bits(), alt.to_bits(), "order matters");
    assert_eq!(w[0].to_bits(), (big / sum).to_bits());
    assert_eq!(w[1].to_bits(), (1.0f32 / sum).to_bits());
    assert_eq!(w[2].to_bits(), (1.0f32 / sum).to_bits());
}

// ---------------------------------------------------------------------------
// E4F item 1: wrapped hydrogen ranges against both independent oracles.
// ---------------------------------------------------------------------------

/// Find `ppm_tenths` in `1..=1000` with `tolerance(mz, ppm) == tol`
/// (independent `u64` arithmetic, for oracle-consistent fixtures).
fn find_ppm_for_tol(mz: u32, tol: u32) -> Option<u32> {
    for ppm in 1..=1000u32 {
        if tolerance(mz, ppm) == tol {
            return Some(ppm);
        }
    }
    None
}

/// E4F wrapped ranges on many-hydrogen parents against BOTH independent
/// oracles: at least 200 random parents with `c[H]` in 120..=400, heavy
/// atoms small enough to keep the exhaustive oracles affordable (C 1..=6, N
/// 0..=2, O 0..=2, F/P/S/Cl 0..=1), both adducts, `U` in {0, 5}, ppm in
/// {100, 1000}, peaks on true sub-composition ions displaced by `-tol-3 ..=
/// +tol+3` (edges just outside acceptance plus interior) plus a decoy.
/// Expected values come from `EvidenceIndex::explains` and `ion_assign`
/// only; the twin under test supplies the verdict.
#[test]
fn e4f_wrapped_many_hydrogens_vs_both_oracles() {
    let mut rng = StdRng::seed_from_u64(0xE4F1u64);
    let work_max = 1u32 << 20;
    let mut compared = 0usize;
    let mut explained = 0usize;
    let mut unexplained = 0usize;
    let mut smax_ge1 = 0usize;
    for _ in 0..210 {
        let mut parent: Composition = [0; 10];
        parent[0] = rng.random_range(1..=6);
        parent[HYDROGEN] = rng.random_range(120..=400);
        parent[2] = rng.random_range(0..=2);
        parent[3] = rng.random_range(0..=2);
        parent[4] = rng.random_range(0..=1);
        parent[5] = rng.random_range(0..=1);
        parent[6] = rng.random_range(0..=1);
        parent[7] = rng.random_range(0..=1);
        let j = noncarbon_product(&parent);
        assert!(j <= 2016, "oracle-affordable heavies");
        for adduct in [1u16, 2] {
            for u in [0u32, 5] {
                for ppm in [100u32, 1000] {
                    let h_pos: u32 = if adduct == 1 { 1 } else { 0 };
                    let h_cap = u32::from(parent[HYDROGEN]) + h_pos + 2;
                    let mut mzs: Vec<u32> = Vec::new();
                    for _ in 0..3 {
                        const ORDER: [usize; 9] = [0, 2, 3, 4, 5, 6, 7, 8, 9];
                        let mut digits = [0u16; 9];
                        for (s, &e) in ORDER.iter().enumerate() {
                            digits[s] = rng.random_range(0..=parent[e]);
                        }
                        if digits.iter().all(|&d| d == 0) {
                            digits[0] = 1;
                        }
                        let h = rng.random_range(0..=h_cap);
                        let mut mass: u64 = 0;
                        for (s, &e) in ORDER.iter().enumerate() {
                            mass += u64::from(digits[s]) * u64::from(ELEMENTS[e].mass);
                        }
                        mass += u64::from(h) * u64::from(ELEMENTS[HYDROGEN].mass);
                        let Ok(mass) = u32::try_from(mass) else {
                            continue;
                        };
                        let mz = if adduct == 1 {
                            mass.checked_sub(549)
                        } else if mass <= u32::MAX - 549 {
                            Some(mass + 549)
                        } else {
                            None
                        };
                        let Some(mz) = mz.filter(|&v| v > 0) else {
                            continue;
                        };
                        let tol = tolerance(mz, ppm) as i64;
                        let mut shifts = vec![0i64, -(tol + 3), tol + 3];
                        shifts.push(rng.random_range(-(tol + 3)..=tol + 3));
                        for shift in shifts {
                            let peak = mz as i64 + shift;
                            if peak >= 1 && peak <= u64::from(u32::MAX) as i64 {
                                mzs.push(peak as u32);
                            }
                        }
                    }
                    mzs.push(rng.random_range(5_000_000u32..150_000_000));
                    mzs.sort_unstable();
                    mzs.dedup();
                    let index = build_evidence_index(&parent, adduct, u, ppm)
                        .expect("index builds")
                        .expect("supported parent");
                    let v = j.min(work_max as u64);
                    let oracle_w = oracle_work_max(v, parent[0]);
                    let limits = IonLimits { work_max: oracle_w, kept: 4 };
                    for &mz in mzs.iter() {
                        let (t, tol_p) = peak_row(mz, adduct, ppm);
                        let out = lane_explains(&parent, t, tol_p, adduct, u, work_max);
                        let fast = out[0] == 1.0;
                        let slow_hit = index.explains(mz).expect("explains runs");
                        let assign =
                            ion_assign(&parent, adduct, mz, u, ppm, &limits).expect("ion runs");
                        let oracle = assign.accepted >= 1;
                        assert_eq!(
                            fast, slow_hit,
                            "parent {parent:?} adduct {adduct} U {u} ppm {ppm} peak {mz}: twin vs index"
                        );
                        assert_eq!(
                            fast, oracle,
                            "parent {parent:?} adduct {adduct} U {u} ppm {ppm} peak {mz}: twin vs ion_assign"
                        );
                        if hydrogen_s_max(lane_h_cap(&parent, adduct), tol_p) >= 1 {
                            smax_ge1 += 1;
                        }
                        compared += 1;
                        if fast {
                            explained += 1;
                        } else {
                            unexplained += 1;
                        }
                    }
                }
            }
        }
    }
    println!(
        "E4F-MANY-H compared={compared} explained={explained} unexplained={unexplained} smax_ge1={smax_ge1}"
    );
    assert!(compared >= 15_000, "only {compared} (candidate, peak) pairs ran");
    assert!(explained >= 200, "only {explained} explained pairs");
    assert!(unexplained >= 200, "only {unexplained} unexplained pairs");
    assert!(smax_ge1 >= 5_000, "only {smax_ge1} pairs with s_max >= 1");
}

/// E4F wrapped ranges at the exact `7,825 h_cap + 2 tol` sums 999,999 /
/// 1,000,000 / 1,999,999 / 2,000,000 (hand-solved `(h_cap, tol)` pairs below,
/// asserted here): fast, slow and both oracles agree on accepted exact-ion
/// peaks and on far decoys, and the wrap counts pin the boundary arithmetic
/// (`s_max` 0, 1, 1, 2).
#[test]
fn e4f_wrapped_exact_sum_boundaries() {
    // 7,825 * 127 + 2 * 3,112 = 999,999; 7,825 * 126 + 2 * 7,025 =
    // 1,000,000; 7,825 * 255 + 2 * 2,312 = 1,999,999; 7,825 * 254 + 2 * 6,225
    // = 2,000,000 (hand division, asserted below).
    let cases = [
        (127u32, 3_112u32, 999_999u64, 0u32),
        (126u32, 7_025u32, 1_000_000u64, 1u32),
        (255u32, 2_312u32, 1_999_999u64, 1u32),
        (254u32, 6_225u32, 2_000_000u64, 2u32),
    ];
    let work_max = 4096u32;
    for (h_cap, tol, want_sum, want_smax) in cases {
        assert_eq!(
            7_825u64 * u64::from(h_cap) + 2 * u64::from(tol),
            want_sum,
            "hand sum"
        );
        assert_eq!(hydrogen_s_max(h_cap, tol), want_smax, "wrap count");
        assert!(
            hydrogen_wrapped_trials(h_cap, tol) <= u64::from(h_cap) + 1,
            "item-1 rule takes the wrapped ranges"
        );
        // Carbon-only candidate with exactly this cap (adduct 1); c[C] = 6
        // keeps every oracle's masses in `u32` (C400 would overflow the
        // reference index).
        let h_c = (h_cap - 3) as u16;
        let comp: Composition = [6, h_c, 0, 0, 0, 0, 0, 0, 0, 0];
        assert_eq!(lane_h_cap(&comp, 1), h_cap);
        let v = noncarbon_product(&comp);
        assert_eq!(v, 1);
        let oracle_w = oracle_work_max(1, 6);
        let limits = IonLimits { work_max: oracle_w, kept: 4 };
        // Accepted exact-ion peaks: search h upward for oracle-consistent
        // ppm realisations (small h cannot realise large tolerances).
        let mut tried = 0usize;
        for h in 0..=h_cap {
            let cand_hyp = 12_000_000u32 + h * 1_007_825;
            let bound = (33 * h + 421).div_ceil(1000);
            if bound > tol {
                continue;
            }
            let mz = cand_hyp - 549;
            let Some(ppm) = find_ppm_for_tol(mz, tol) else {
                continue;
            };
            let (t, tol_p) = (cand_hyp, tol);
            assert_eq!(tol_p, tolerance(mz, ppm));
            let out = lane_explains(&comp, t, tol_p, 1, 0, work_max);
            assert_eq!(out[0], 1.0, "h_cap {h_cap} h {h}: explained");
            let slow = lane_explains_raw_words(
                &pack_one(&comp),
                t,
                tol_p,
                1,
                0,
                work_max,
                true,
            );
            assert_eq!([slow[0], slow[1], slow[2], slow[3]], [out[0], out[1], out[2], out[3]]);
            let assign = ion_assign(&comp, 1, mz, 0, ppm, &limits).expect("ion runs");
            assert!(assign.accepted >= 1, "h_cap {h_cap} h {h}: oracle accepts");
            let index = build_evidence_index(&comp, 1, 0, ppm)
                .expect("index builds")
                .expect("supported");
            assert!(index.explains(mz).expect("explains runs"), "index explains");
            // Far decoy at its own tolerance is rejected everywhere.
            let mz_d = mz.saturating_add(tol + 100);
            let (t_d, tol_d) = peak_row(mz_d, 1, ppm);
            let out_d = lane_explains(&comp, t_d, tol_d, 1, 0, work_max);
            assert_eq!(out_d[0], 0.0, "h_cap {h_cap} h {h}: decoy unexplained");
            let assign_d = ion_assign(&comp, 1, mz_d, 0, ppm, &limits).expect("ion runs");
            assert_eq!(assign_d.accepted, 0, "decoy oracle rejects");
            tried += 1;
            if tried >= 3 {
                break;
            }
        }
        assert!(tried >= 1, "h_cap {h_cap} tol {tol}: no accepted peak ran");
    }
}

/// E4F accepted hypotheses exactly at wrap boundaries (hand-verified
/// fixtures below, asserted by recomputation here; verdicts from the
/// oracles): `Rm = 0` with `s = 1` (`S = 1,000,000`), and `Rm = 999,999`
/// with `s = 0` (`S = 999,999`).
///
/// Both use candidate C6 H124, adduct 1 (`h_cap = 127`), `U = 0`,
/// `tol = 6,299`, `ppm = 450`, hypothesis `(n = 1, h = 127, m' = 0)` with
/// `bound = ceil((33 * 127 + 421) / 1000) = 5`:
/// - `Rm = 0`: `t = 139,993,701` (`r = -74`), `X = t + tol = 140,000,000`,
///   `Rm = 0`, `D = r + tol = 6,225`, `S = 993,775 + 6,225 = 1,000,000`,
///   `s = 1`. `r + bound = 79 <= tol`: accepted.
/// - `Rm = 999,999`: `t = 139,993,700` (`r = -75`), `X = 139,999,999`,
///   `Rm = 999,999`, `D = 6,224`, `S = 999,999`, `s = 0`. Accepted.
/// `tolerance(139,993,152, 450) = tolerance(139,993,151, 450) = 6,299`
/// (asserted), so both peaks are oracle-consistent.
#[test]
fn e4f_wrap_boundary_accepted_rm_edges() {
    let comp: Composition = [6, 124, 0, 0, 0, 0, 0, 0, 0, 0];
    assert_eq!(lane_h_cap(&comp, 1), 127);
    let work_max = 4096u32;
    let oracle_w = oracle_work_max(1, 6);
    let limits = IonLimits { work_max: oracle_w, kept: 4 };
    // Hand arithmetic, asserted (fixture discovery, not verdicts).
    let cand: u32 = 12_000_000 + 127 * 1_007_825;
    assert_eq!(cand, 139_993_775);
    let bound: u32 = (33u32 * 127 + 421).div_ceil(1000);
    assert_eq!(bound, 5);
    assert_eq!(tolerance(139_993_152, 450), 6_299);
    assert_eq!(tolerance(139_993_151, 450), 6_299);
    assert_eq!(hydrogen_s_max(127, 6_299), 1);
    for (t, want_rm, want_s) in [(139_993_701u32, 0u32, 1u32), (139_993_700u32, 999_999u32, 0u32)] {
        let tol = 6_299u32;
        let x = u64::from(t) + u64::from(tol);
        let rm = (x % 1_000_000) as u32;
        assert_eq!(rm, want_rm, "t = {t}: Rm");
        let d = (x - u64::from(cand)) as u32;
        let s_val = 7_825u32 * 127 + d;
        assert_eq!(s_val % 1_000_000, want_rm);
        assert_eq!((s_val - want_rm) / 1_000_000, want_s, "t = {t}: wrap count");
        let mz = t - 549;
        let out = lane_explains(&comp, t, tol, 1, 0, work_max);
        assert_eq!(out, [1.0, 1.0, 1.0, 1.0], "t = {t}: explained");
        let slow = lane_explains_raw_words(&pack_one(&comp), t, tol, 1, 0, work_max, true);
        assert_eq!([slow[0], slow[1], slow[2], slow[3]], out);
        let assign = ion_assign(&comp, 1, mz, 0, 450, &limits).expect("ion runs");
        assert!(assign.accepted >= 1, "t = {t}: oracle accepts");
        let index = build_evidence_index(&comp, 1, 0, 450)
            .expect("index builds")
            .expect("supported");
        assert!(index.explains(mz).expect("explains runs"), "t = {t}: index explains");
    }
}

/// E4F non-vacuous accepted fast-range endpoints at tolerances 3,912 and
/// 3,913 (the review notes the existing residue-edge cases accept nothing).
///
/// `2 * 3,912 = 7,824 < 7,825`: with `s_max = 0` every range is the single
/// point `{floor(Rm / 7,825)}`, so an accepted `h` is both endpoints at once
/// (asserted). `2 * 3,913 = 7,826`: two-point ranges `{q - 1, q}` occur
/// exactly when `Rm % 7,825 <= 1` (with `Rm >= 7,826`); otherwise single
/// points. All cases use candidate C6 H20, adduct 1 (`h_cap = 23`,
/// `s_max = 0`, asserted), `U = 0`, hypothesis `(n = 6, h, m' = 0)` with
/// `bound(h) = ceil((33 h + 421) / 1000) = 1` for the `h` below, and
/// `tol << m_H / 2` (a single `(n, h)` hypothesis can accept, isolating the
/// endpoint claim to the exhibited `h`).
///
/// Hand-verified fixtures (recomputed and asserted in-test; verdicts from
/// the oracles):
/// - tol 3,912: found by `(n, h, ppm)` search below (any accepted `h` is
///   both endpoints).
/// - tol 3,913 UPPER endpoint: `h = 11`, `r = -3,912`,
///   `t = 72M + 11 * m_H - 3,912 = 83,082,163`,
///   `tolerance(83,081,614, 471) = 3,913`, `X = 83,086,076`, `Rm = 86,076`,
///   range `{10, 11}`: the accepted `h = 11` is the top.
/// - tol 3,913 LOWER endpoint: `h = 17`, `r = +3,912`,
///   `t = 72M + 17 * m_H + 3,912 = 89,136,937`,
///   `tolerance(89,136,388, 439) = 3,913`, `X = 89,140,850`, `Rm = 140,850`,
///   range `{17, 18}`: the accepted `h = 17` is the bottom.
#[test]
fn e4f_fast_endpoints_3912_3913_nonvacuous() {
    let comp: Composition = [6, 20, 0, 0, 0, 0, 0, 0, 0, 0];
    assert_eq!(lane_h_cap(&comp, 1), 23);
    assert_eq!(hydrogen_s_max(23, 3_912), 0);
    assert_eq!(hydrogen_s_max(23, 3_913), 0);
    let work_max = 4096u32;
    let oracle_w = oracle_work_max(1, 6);
    let limits = IonLimits { work_max: oracle_w, kept: 4 };
    // tol 3,912: single-point range; accepted h is both endpoints.
    {
        let tol = 3_912u32;
        let mut found = None;
        for n in 1..=6u32 {
            for h in 0..=23u32 {
                let bound = (33 * h + 421).div_ceil(1000);
                if bound > tol {
                    continue;
                }
                let t = n * 12_000_000 + h * 1_007_825;
                let mz = t - 549;
                if let Some(ppm) = find_ppm_for_tol(mz, tol) {
                    found = Some((n, h, ppm, mz, t));
                    break;
                }
            }
            if found.is_some() {
                break;
            }
        }
        let (n, h, ppm, mz, t) =
            found.expect("tol 3,912: an oracle-consistent accepted peak exists");
        let _ = n;
        let rm = ((u64::from(t) + u64::from(tol)) % 1_000_000) as u32;
        let lo = rm.saturating_sub(2 * tol).div_ceil(7_825);
        let hi = (rm / 7_825).min(23);
        assert_eq!(lo, hi, "tol 3,912 ranges are single points");
        assert_eq!(hi, h, "accepted h is the endpoint");
        let out = lane_explains(&comp, t, tol, 1, 0, work_max);
        assert_eq!(out[0], 1.0, "tol 3,912 accepts (non-vacuous)");
        let assign = ion_assign(&comp, 1, mz, 0, ppm, &limits).expect("ion runs");
        assert!(assign.accepted >= 1, "oracle accepts");
        let index = build_evidence_index(&comp, 1, 0, ppm)
            .expect("index builds")
            .expect("supported");
        assert!(index.explains(mz).expect("explains runs"));
    }
    // tol 3,913, UPPER endpoint of a two-point range.
    {
        let (tol, h, t, ppm) = (3_913u32, 11u32, 83_082_163u32, 471u32);
        assert_eq!((33 * h + 421).div_ceil(1000), 1, "hand bound");
        assert_eq!(t, 6 * 12_000_000 + h * 1_007_825 - 3_912, "hand t");
        let mz = t - 549;
        assert_eq!(mz, 83_081_614);
        assert_eq!(tolerance(mz, ppm), tol, "hand tol");
        let rm = ((u64::from(t) + u64::from(tol)) % 1_000_000) as u32;
        assert_eq!(rm, 86_076, "hand Rm");
        assert_eq!(rm % 7_825, 1, "two-point range");
        let lo = rm.saturating_sub(2 * tol).div_ceil(7_825);
        let hi = (rm / 7_825).min(23);
        assert_eq!((lo, hi), (10, 11), "range is {{10, 11}}");
        assert_eq!(h, hi, "accepted h is the upper endpoint");
        let out = lane_explains(&comp, t, tol, 1, 0, work_max);
        assert_eq!(out[0], 1.0, "tol 3,913 upper endpoint accepts (non-vacuous)");
        let assign = ion_assign(&comp, 1, mz, 0, ppm, &limits).expect("ion runs");
        assert!(assign.accepted >= 1, "oracle accepts");
        let index = build_evidence_index(&comp, 1, 0, ppm)
            .expect("index builds")
            .expect("supported");
        assert!(index.explains(mz).expect("explains runs"));
    }
    // tol 3,913, LOWER endpoint of a two-point range.
    {
        let (tol, h, t, ppm) = (3_913u32, 17u32, 89_136_937u32, 439u32);
        assert_eq!((33 * h + 421).div_ceil(1000), 1, "hand bound");
        assert_eq!(t, 6 * 12_000_000 + h * 1_007_825 + 3_912, "hand t");
        let mz = t - 549;
        assert_eq!(mz, 89_136_388);
        assert_eq!(tolerance(mz, ppm), tol, "hand tol");
        let rm = ((u64::from(t) + u64::from(tol)) % 1_000_000) as u32;
        assert_eq!(rm, 140_850, "hand Rm");
        assert_eq!(rm % 7_825, 0, "two-point range");
        let lo = rm.saturating_sub(2 * tol).div_ceil(7_825);
        let hi = (rm / 7_825).min(23);
        assert_eq!((lo, hi), (17, 18), "range is {{17, 18}}");
        assert_eq!(h, lo, "accepted h is the lower endpoint");
        let out = lane_explains(&comp, t, tol, 1, 0, work_max);
        assert_eq!(out[0], 1.0, "tol 3,913 lower endpoint accepts (non-vacuous)");
        let assign = ion_assign(&comp, 1, mz, 0, ppm, &limits).expect("ion runs");
        assert!(assign.accepted >= 1, "oracle accepts");
        let index = build_evidence_index(&comp, 1, 0, ppm)
            .expect("index builds")
            .expect("supported");
        assert!(index.explains(mz).expect("explains runs"));
    }
}

/// E4F hydrogen-trial counts: a test-only counter return on the hidden twin
/// variants pins the physical work per (visit, peak) to at most
/// `min(h_cap + 1, (s_max + 1)(2 tol / 7,825 + 2))`.
///
/// Reviewer scenario (review §P2): candidate
/// `[C100,H200,N7,O7,F7,P3]`, adduct 1, `U = 0`, ppm-tenths 100, 32 peaks
/// at m/z 50,499,451 (`t = 50,500,000`, `tol = 504`, `J = W = 2,048`,
/// `h_cap = 203`). Before: `2,048 * 32 * 204 = 13,369,344` trials (slow
/// range, asserted exactly). After: at most `2,048 * 32 * 4 = 262,144`
/// (`s_max = 1`, `2 * 504 / 7,825 + 2 = 2`), with identical rows.
#[test]
fn e4f_trial_count_bound_reviewer_and_random() {
    // Reviewer scenario.
    let comp: Composition = [100, 200, 7, 7, 7, 3, 0, 0, 0, 0];
    assert_eq!(composition_mass(&comp).expect("mass fits"), 1_837_461_030);
    let cand = pack_one(&comp);
    let (t, tol) = (50_500_000u32, 504u32);
    assert_eq!(peak_row(50_499_451, 1, 100), (t, tol));
    assert_eq!(lane_h_cap(&comp, 1), 203);
    assert_eq!(hydrogen_s_max(203, tol), 1);
    assert_eq!(hydrogen_trials_bound(203, tol), 4);
    let p = 32usize;
    let mut ev_peaks = vec![u32::MAX, 0, 0, 0];
    ev_peaks = ev_peaks.repeat(p);
    let mut ev_w = vec![0.0f32; p];
    for s in 0..p {
        ev_peaks[s * 4] = s as u32;
        ev_peaks[s * 4 + 1] = t;
        ev_peaks[s * 4 + 2] = tol;
        ev_peaks[s * 4 + 3] = 1;
        ev_w[s] = 1.0;
    }
    let meta = vec![0u32, 0, 0, 1, 100, 0, 0, 0];
    let spec = vec![0u32, 0];
    let work_max = 2048u32;
    let mut fast_row = vec![0.0f32; 4];
    let mut fast_trials = 0u64;
    formula_evidence_lane_trials(
        &cand, &ev_peaks, &ev_w, &meta, &spec, 0, 0, 1, p as u32, work_max, u32::MAX,
        &mut fast_row, &mut fast_trials,
    );
    let mut slow_row = vec![0.0f32; 4];
    let mut slow_trials = 0u64;
    formula_evidence_lane_slow_trials(
        &cand, &ev_peaks, &ev_w, &meta, &spec, 0, 0, 1, p as u32, work_max, u32::MAX,
        &mut slow_row, &mut slow_trials,
    );
    assert_eq!(fast_row, slow_row, "predicate unchanged");
    assert_eq!(fast_row, [0.0, 0.0, 32.0, 1.0], "nothing explained, complete walk");
    assert_eq!(slow_trials, 13_369_344, "slow trials before");
    assert!(fast_trials <= 262_144, "wrapped trials after: {fast_trials}");
    println!("E4F-TRIALS reviewer before={slow_trials} after={fast_trials}");
    // Independent oracle spot-check on one peak: nothing accepted.
    let assign = ion_assign(
        &comp, 1, 50_499_451, 0, 100,
        &IonLimits { work_max: oracle_work_max(2048, 100), kept: 4 },
    )
    .expect("ion_assign runs");
    assert_eq!(assign.accepted, 0, "reviewer scenario accepts nothing");
    // Per-(visit, peak) micro-cases: W = 1, P = 1, so the lane total IS the
    // single (visit, peak) count; it never exceeds the item-1 bound and
    // never exceeds the slow count, with identical rows.
    let mut checked = 0usize;
    for h_c in [0u16, 1, 5, 50, 124, 200, 400, 1000, 5000, 65534] {
        for tol in [0u32, 100, 504, 3_912, 50_000] {
            for adduct in [1u16, 2] {
                let parent: Composition = [2, h_c, 1, 0, 0, 0, 0, 0, 0, 0];
                let h_cap = lane_h_cap(&parent, adduct);
                let bound = hydrogen_trials_bound(h_cap, tol);
                assert!(bound <= u64::from(h_cap) + 1);
                for peak in 0..2 {
                    // Raw words: large hydrogen counts overflow
                    // `composition_mass`, which the lane never reads.
                    let t = if peak == 0 {
                        ((24_000_000u64 + u64::from(h_c) * 1_007_825) % 500_000_000) as u32
                    } else {
                        10_000_000u32
                    };
                    let words = vec![
                        2, u32::from(h_c), 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, u32::MAX,
                    ];
                    let mut fast_row = vec![0.0f32; 4];
                    let mut fast_n = 0u64;
                    formula_evidence_lane_trials(
                        &words,
                        &[0, t, tol, 1],
                        &[1.0],
                        &[0, 0, 0, u32::from(adduct), 0, 0, 0, 0],
                        &[0, 0],
                        0, 0, 1, 1, 1, u32::MAX,
                        &mut fast_row, &mut fast_n,
                    );
                    let mut slow_row = vec![0.0f32; 4];
                    let mut slow_n = 0u64;
                    formula_evidence_lane_slow_trials(
                        &words,
                        &[0, t, tol, 1],
                        &[1.0],
                        &[0, 0, 0, u32::from(adduct), 0, 0, 0, 0],
                        &[0, 0],
                        0, 0, 1, 1, 1, u32::MAX,
                        &mut slow_row, &mut slow_n,
                    );
                    assert_eq!(fast_row, slow_row, "rows agree");
                    assert!(fast_n <= bound, "per-(visit, peak) bound");
                    assert!(fast_n <= slow_n, "narrowing only");
                    if h_c == 65534 && adduct == 1 {
                        // Saturation: cap 65,535, plain range 65,536 trials.
                        assert_eq!(h_cap, 65_535);
                    }
                    checked += 1;
                }
            }
        }
    }
    // Saturation slow count pins the clamp: H5000 C1, W = 1, P = 1.
    {
        let mut words = vec![0u32; 13];
        words[0] = 1;
        words[1] = 5_000;
        words[10] = 0;
        words[11] = 1;
        words[12] = u32::MAX;
        let mut slow_row = vec![0.0f32; 4];
        let mut slow_n = 0u64;
        formula_evidence_lane_slow_trials(
            &words,
            &[0, 12_000_000, 119, 1],
            &[1.0],
            &[0, 0, 0, 1, 0, 0, 0, 0],
            &[0, 0],
            0, 0, 1, 1, 1, u32::MAX,
            &mut slow_row, &mut slow_n,
        );
        assert_eq!(slow_n, 5_004, "plain range 0..=5003");
        assert_eq!(slow_row[0], 1.0, "h = 0 accepted");
        let mut fast_n = 0u64;
        let mut fast_row = vec![0.0f32; 4];
        formula_evidence_lane_trials(
            &words,
            &[0, 12_000_000, 119, 1],
            &[1.0],
            &[0, 0, 0, 1, 0, 0, 0, 0],
            &[0, 0],
            0, 0, 1, 1, 1, u32::MAX,
            &mut fast_row, &mut fast_n,
        );
        assert_eq!(fast_row, slow_row);
        assert!(fast_n <= hydrogen_trials_bound(5_003, 119));
        assert_eq!(hydrogen_trials_bound(5_003, 119), 80);
    }
    println!("E4F-TRIALS micro checked={checked}");
    assert!(checked >= 200, "only {checked} micro cases ran");
}

/// E4F review's missing `u32`-edge cases (hand-verified fixtures, verdicts
/// from the oracles):
/// - `t + tol > u32::MAX` AND `t + delta > u32::MAX` with an accepted
///   representable hypothesis: candidate `[C8,H8,I33]`, adduct 1, `U = 0`,
///   hypothesis `(n = 8, h = 11, m' = 4,187,847,576)` with `bound = 5`,
///   `t = 4,294,933,651`, `tol = 42,949` (`tolerance(4,294,933,102, 100)`),
///   `top = t + delta` saturating to `u32::MAX`, `n1 = 8`.
/// - non-carbon mass overflow: candidate `[N1000]` (raw words) with visits
///   at `d_N >= 307` overflowing `u32`, accepted via `d_N = 1`.
/// - hydrogen term overflow: candidate `[C1,H5000]` (raw words) with
///   `h * m_H` overflowing at `h >= 4,262`, accepted at `h = 0`.
#[test]
fn e4f_u32_edge_acceptance_and_overflow() {
    let work_max = 4096u32;
    // `t + tol` and `t + delta` past `u32::MAX`, accepted.
    {
        let comp: Composition = [8, 8, 0, 0, 0, 0, 0, 0, 0, 33];
        let (t, tol, ppm) = (4_294_933_651u32, 42_949u32, 100u32);
        let mz = t - 549;
        assert_eq!(mz, 4_294_933_102);
        assert_eq!(tol_u64(mz, ppm), tol, "hand tol");
        assert_eq!(tolerance(mz, ppm), tol, "oracle tol");
        // Hand carbon/residual arithmetic (fixture, not verdicts).
        let mprime: u32 = 33 * 126_904_472;
        assert_eq!(mprime, 4_187_847_576);
        let base = mprime + 11 * 1_007_825;
        assert_eq!(base, 4_198_933_651);
        let bound: u32 = (33u32 * 100 + 33 * 11 + 421).div_ceil(1000);
        assert_eq!(bound, 5);
        let delta = tol - bound;
        assert_eq!(delta, 42_944);
        let t_sum = u64::from(t) + u64::from(tol);
        assert!(t_sum > u64::from(u32::MAX), "t + tol overflows");
        assert!(u64::from(t) + u64::from(delta) > u64::from(u32::MAX), "t + delta overflows");
        let n1 = (u32::MAX - base) / 12_000_000;
        assert_eq!(n1, 8, "saturated top still yields n1 = 8");
        let out = lane_explains(&comp, t, tol, 1, 0, work_max);
        assert_eq!(out[0], 1.0, "overflow edge accepts");
        let slow = lane_explains_raw_words(&pack_one(&comp), t, tol, 1, 0, work_max, true);
        assert_eq!([slow[0], slow[1], slow[2], slow[3]], out);
        let oracle_w = oracle_work_max(34, 8);
        assert_eq!(oracle_w, 305);
        let assign = ion_assign(
            &comp, 1, mz, 0, ppm,
            &IonLimits { work_max: oracle_w, kept: 4 },
        )
        .expect("ion_assign runs");
        assert!(assign.accepted >= 1, "oracle accepts past-MAX edge");
        let index = build_evidence_index(&comp, 1, 0, ppm)
            .expect("index builds")
            .expect("supported");
        assert!(index.explains(mz).expect("explains runs"));
    }
    // Non-carbon mass overflow: N1000 overflows at d_N >= 307.
    {
        assert!(307u64 * 14_003_074 > u64::from(u32::MAX));
        assert!(306u64 * 14_003_074 <= u64::from(u32::MAX));
        let mut words = vec![0u32; 13];
        words[2] = 1_000;
        words[1] = 4;
        words[10] = 0;
        words[11] = 1;
        words[12] = u32::MAX;
        let (t, tol, ppm) = (14_003_074u32, 140u32, 100u32);
        let mz = t - 549;
        assert_eq!(tol_u64(mz, ppm), tol);
        let out = lane_explains_raw_words(&words, t, tol, 1, 0, work_max, false);
        assert_eq!(out, [1.0, 1.0, 1.0, 1.0], "accepted via d_N = 1");
        let slow = lane_explains_raw_words(&words, t, tol, 1, 0, work_max, true);
        assert_eq!(slow, out);
        let parent: Composition = [0, 4, 1_000, 0, 0, 0, 0, 0, 0, 0];
        let assign = ion_assign(
            &parent, 1, mz, 0, ppm,
            &IonLimits { work_max: oracle_work_max(1_001, 0), kept: 4 },
        )
        .expect("ion_assign runs");
        assert!(assign.accepted >= 1, "oracle accepts despite overflow visits");
        // No `EvidenceIndex` here: its exhaustive build refuses parents with
        // a `u32`-overflowing sub-vector mass; `ion_assign` is the oracle.
    }
    // Hydrogen term overflow: H5000 overflows at h >= 4,262.
    {
        assert!(4_261u64 * 1_007_825 <= u64::from(u32::MAX));
        assert!(4_262u64 * 1_007_825 > u64::from(u32::MAX));
        let mut words = vec![0u32; 13];
        words[0] = 1;
        words[1] = 5_000;
        words[10] = 0;
        words[11] = 1;
        words[12] = u32::MAX;
        let (t, tol, ppm) = (12_000_000u32, 119u32, 100u32);
        let mz = t - 549;
        assert_eq!(tol_u64(mz, ppm), tol);
        let out = lane_explains_raw_words(&words, t, tol, 1, 0, work_max, false);
        assert_eq!(out[0], 1.0, "accepted at h = 0");
        let slow = lane_explains_raw_words(&words, t, tol, 1, 0, work_max, true);
        assert_eq!(slow, out);
        let parent: Composition = [1, 5_000, 0, 0, 0, 0, 0, 0, 0, 0];
        let assign = ion_assign(
            &parent, 1, mz, 0, ppm,
            &IonLimits { work_max: oracle_work_max(1, 1), kept: 4 },
        )
        .expect("ion_assign runs");
        assert!(assign.accepted >= 1, "oracle accepts despite overflow h");
        let index = build_evidence_index(&parent, 1, 0, ppm)
            .expect("index builds")
            .expect("supported");
        assert!(index.explains(mz).expect("explains runs"));
    }
}

/// E4F residual edges (verdicts from the oracles; hand residual arithmetic
/// asserted):
/// - accepted hypothesis exactly at `residual + bound == tol` next to the
///   ambiguous neighbour: candidate C1 H10, `h = 5`
///   (`res = 5 * 33 + 421 = 586`, `arith = 1`), `U = 50` (`bound = 51`),
///   `r = 100`, `tol = 151` (`tolerance(17,038,676, 89)`); the neighbour at
///   `r = 101` (`bound + r = tol + 1`) is rejected everywhere.
/// - residual-ceiling transition (`res' + 33 h + 421` crossing a multiple of
///   1000): `33 * 17 + 421 = 982` (`arith 1`) vs `33 * 18 + 421 = 1,015`
///   (`arith 2`). Candidate C1 H20, `U = 0`: peak at `t = cand_17` with
///   `tol = 2` (`tolerance(29,132,476, 1)`) is accepted via `h = 17`, while
///   `t = cand_18` with `tol = 0` (`ppm = 0`) is rejected (`bound 2 > 0`);
///   `t = cand_17` with `tol = 0` is rejected too (`bound 1 > 0`).
#[test]
fn e4f_residual_edge_acceptance() {
    let work_max = 4096u32;
    // Equality at residual + bound == tol, plus the rejected neighbour.
    {
        let comp: Composition = [1, 10, 0, 0, 0, 0, 0, 0, 0, 0];
        assert_eq!(5 * 33 + 421, 586);
        assert_eq!(586u32.div_ceil(1000), 1);
        let (u, bound) = (50u32, 51u32);
        let cand = 12_000_000u32 + 5 * 1_007_825;
        assert_eq!(cand, 17_039_125);
        let (r, tol, ppm) = (100u32, 151u32, 89u32);
        assert_eq!(r + bound, tol, "exactly at the edge");
        let t = cand + r;
        let mz = t - 549;
        assert_eq!(mz, 17_038_676);
        assert_eq!(tolerance(mz, ppm), tol, "oracle tol");
        let oracle_w = oracle_work_max(1, 1);
        let limits = IonLimits { work_max: oracle_w, kept: 4 };
        let out = lane_explains(&comp, t, tol, 1, u, work_max);
        assert_eq!(out[0], 1.0, "edge accepts");
        let assign = ion_assign(&comp, 1, mz, u, ppm, &limits).expect("ion runs");
        assert!(assign.accepted >= 1, "oracle accepts the edge");
        let index = build_evidence_index(&comp, 1, u, ppm)
            .expect("index builds")
            .expect("supported");
        assert!(index.explains(mz).expect("explains runs"));
        // Ambiguous neighbour at r + 1: bound + (r + 1) = tol + 1.
        let t_n = t + 1;
        let mz_n = mz + 1;
        assert_eq!(tolerance(mz_n, ppm), tol, "same tol next door");
        let out_n = lane_explains(&comp, t_n, tol, 1, u, work_max);
        assert_eq!(out_n[0], 0.0, "neighbour rejected");
        let assign_n = ion_assign(&comp, 1, mz_n, u, ppm, &limits).expect("ion runs");
        assert_eq!(assign_n.accepted, 0, "oracle rejects the neighbour");
        let index_n = build_evidence_index(&comp, 1, u, ppm)
            .expect("index builds")
            .expect("supported");
        assert!(!index_n.explains(mz_n).expect("explains runs"));
    }
    // Residual-ceiling transition across 1000.
    {
        assert_eq!(33 * 17 + 421, 982);
        assert_eq!(33 * 18 + 421, 1_015);
        assert_eq!(982u32.div_ceil(1000), 1);
        assert_eq!(1_015u32.div_ceil(1000), 2);
        let comp: Composition = [1, 20, 0, 0, 0, 0, 0, 0, 0, 0];
        let oracle_w = oracle_work_max(1, 1);
        let limits = IonLimits { work_max: oracle_w, kept: 4 };
        let t17 = 12_000_000u32 + 17 * 1_007_825;
        let mz17 = t17 - 549;
        assert_eq!(mz17, 29_132_476);
        assert_eq!(tolerance(mz17, 1), 2, "oracle tol");
        let out17 = lane_explains(&comp, t17, 2, 1, 0, work_max);
        assert_eq!(out17[0], 1.0, "below-ceiling accepts");
        let assign17 = ion_assign(
            &comp, 1, mz17, 0, 1,
            &IonLimits { work_max: oracle_w, kept: 4 },
        )
        .expect("ion runs");
        assert!(assign17.accepted >= 1);
        let _ = limits;
        let t18 = 12_000_000u32 + 18 * 1_007_825;
        let mz18 = t18 - 549;
        assert_eq!(tolerance(mz18, 0), 0, "oracle tol");
        let out18 = lane_explains(&comp, t18, 0, 1, 0, work_max);
        assert_eq!(out18[0], 0.0, "above-ceiling rejected at tol 0");
        let assign18 = ion_assign(
            &comp, 1, mz18, 0, 0,
            &IonLimits { work_max: oracle_w, kept: 4 },
        )
        .expect("ion runs");
        assert_eq!(assign18.accepted, 0);
        let out17_zero = lane_explains(&comp, t17, 0, 1, 0, work_max);
        assert_eq!(out17_zero[0], 0.0, "below-ceiling still needs tol");
    }
}

/// E4F degenerate parents (verdicts from the oracles):
/// - hydrogen-only non-empty parent `[H5]`: nothing explained (the `n1 == 0`
///   with all-zero `u'` rule), complete.
/// - `c[C] = 0` with `V = 1`: parent `[N1,H4]` under `W = 1` visits only
///   `j = 0` (zero-length `ion_assign` prefix, `work_max = 0`): nothing
///   explained, incomplete.
#[test]
fn e4f_degenerate_parents() {
    // Hydrogen-only parent.
    {
        let comp: Composition = [0, 5, 0, 0, 0, 0, 0, 0, 0, 0];
        let t = 5 * 1_007_825;
        let mz = t - 549;
        let (u, ppm) = (0u32, 100u32);
        let tol = tol_u64(mz, ppm);
        let out = lane_explains(&comp, t, tol, 1, u, 4096);
        assert_eq!(out, [0.0, 0.0, 1.0, 1.0], "hydrogen alone explains nothing");
        let assign = ion_assign(
            &comp, 1, mz, u, ppm,
            &IonLimits { work_max: 4096, kept: 4 },
        )
        .expect("ion_assign runs");
        assert_eq!(assign.accepted, 0, "oracle agrees");
        let index = build_evidence_index(&comp, 1, u, ppm)
            .expect("index builds")
            .expect("supported");
        assert!(!index.explains(mz).expect("explains runs"));
    }
    // c[C] = 0 with V = 1.
    {
        let comp: Composition = [0, 4, 1, 0, 0, 0, 0, 0, 0, 0];
        assert_eq!(noncarbon_product(&comp), 2);
        let t = 14_003_074u32;
        let mz = t - 549;
        let (u, ppm) = (0u32, 100u32);
        let tol = tol_u64(mz, ppm);
        let cand = pack_one(&comp);
        let mut out = vec![0.0f32; 4];
        formula_evidence_lane(
            &cand,
            &[0, t, tol, 1],
            &[1.0],
            &[0, 0, 0, 1, 0, 0, 0, 0],
            &[u, 0],
            0, 0, 1, 1, 1, u32::MAX,
            &mut out,
        );
        assert_eq!(out, [0.0, 0.0, 1.0, 0.0], "V = 1 explains nothing, incomplete");
        // Zero-length ion_assign prefix: V * (c[C] + 1) - 1 = 0.
        let assign = ion_assign(
            &comp, 1, mz, u, ppm,
            &IonLimits { work_max: 0, kept: 4 },
        )
        .expect("ion_assign runs");
        assert_eq!(assign.accepted, 0, "zero-length prefix accepts nothing");
    }
}

/// E4F item 2 on the host: a deliberately small `h_cap_max` cuts the range
/// (twin and slow twin agree on the cut), a true bound reproduces the
/// unclamped row, and the host-known accessors report `max + 3`.
#[test]
fn e4f_dispatch_clamp_host() {
    // Candidate C1 H10 (cap 13 under adduct 1); peak at h = 10.
    let comp: Composition = [1, 10, 0, 0, 0, 0, 0, 0, 0, 0];
    assert_eq!(lane_h_cap(&comp, 1), 13);
    let t = 12_000_000u32 + 10 * 1_007_825;
    let mz = t - 549;
    let ppm = 100u32;
    let tol = tol_u64(mz, ppm);
    let cand = pack_one(&comp);
    let meta = vec![0u32, 0, 0, 1, 0, 0, 0, 0];
    let spec = vec![0u32, 0];
    let run = |h_cap_max: u32| {
        let mut out = vec![0.0f32; 4];
        formula_evidence_lane(
            &cand, &[0, t, tol, 1], &[1.0], &meta, &spec, 0, 0, 1, 1, 4096, h_cap_max,
            &mut out,
        );
        let mut slow = vec![0.0f32; 4];
        formula_evidence_lane_slow(
            &cand, &[0, t, tol, 1], &[1.0], &meta, &spec, 0, 0, 1, 1, 4096, h_cap_max,
            &mut slow,
        );
        (out, slow)
    };
    let (cut, cut_slow) = run(5);
    assert_eq!(cut, [0.0, 0.0, 1.0, 1.0], "small bound cuts h = 10");
    assert_eq!(cut_slow, cut, "slow twin agrees on the cut");
    let (full, full_slow) = run(13);
    assert_eq!(full, [1.0, 1.0, 1.0, 1.0], "true bound explains");
    assert_eq!(full_slow, full);
    let (wide, _) = run(u32::MAX);
    assert_eq!(wide, full, "wider bound is identical");
    // Independent oracle: h = 10 accepted, h <= 5 impossible.
    let assign = ion_assign(
        &comp, 1, mz, 0, ppm,
        &IonLimits { work_max: oracle_work_max(1, 1), kept: 4 },
    )
    .expect("ion_assign runs");
    assert!(assign.accepted >= 1);
    // Host-known accessors report max hydrogen + 3.
    let table = FormulaTable::from_compositions(
        [[1u16, 10, 0, 0, 0, 0, 0, 0, 0, 0], [2, 3, 1, 0, 0, 0, 0, 0, 0, 0]].into_iter(),
    )
    .expect("table builds");
    assert_eq!(table.max_hydrogen(), 10);
    assert_eq!(u32::from(table.max_hydrogen()) + 3, 13);
    let domain = EnumDomain::from_compositions(
        [[1u16, 10, 0, 0, 0, 0, 0, 0, 0, 0]].into_iter(),
        2,
    )
    .expect("domain builds");
    assert_eq!(domain.hydrogen_max, 12);
    assert_eq!(u32::from(domain.hydrogen_max) + 3, 15);
}
