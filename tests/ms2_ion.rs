//! H4.2 host-only tests for the fragment-ion assignment twin
//! (`mamba3::models::ms2::ion`). No tensors, no kernels, no neural code.
//!
//! The brute-force cross-check implements its own arithmetic from the
//! contract formulas (literal masses, `floor(mz * t / 1e7)` tolerance, the
//! electron correction, the §5 rule) and never calls production helpers.

use std::collections::BTreeSet;

use serde_json::Value;

use mamba3::models::ms2::chem::{ELEMENTS, HYDROGEN};
use mamba3::models::ms2::ion::{
    EVIDENCE_ROW_WORDS, EVIDENCE_SUPPORT_INCOMPLETE, ION_CAPACITY_EXCEEDED, ION_SEARCH_EXHAUSTED,
    ION_UNAVAILABLE, IonAssignment, IonHypothesis, IonLabel, IonLabels, IonLimits, LANE_ACCEPT,
    LANE_AMBIGUOUS, LANE_BIAS, LANE_REJECT, embedding_ion, evidence_status, ion_assign, ion_labels,
    label_mask, lane_verdict_u32, lane_visits_u32, mapping_is_supported,
};
use mamba3::models::ms2::targets::{Candidates, Peak, RecipeLimits, enumerate_embeddings};
use mamba3::models::ms2::{Composition, MolGraph};

// ---------------------------------------------------------------------------
// Independent arithmetic (contract formulas, literal constants).
// ---------------------------------------------------------------------------

/// Literal integer masses in `ELEMENTS` order (test-local copy of the table).
const MASS: [u32; 10] = [
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
/// Literal rounding residuals in `ELEMENTS` order.
const RES: [u32; 10] = [0, 33, 5, 381, 163, 2, 175, 318, 400, 100];
const ME: u32 = 549;
const ME_RES: u32 = 421;
const MH: u32 = 1_007_825;
const HEAVY: [usize; 9] = [0, 2, 3, 4, 5, 6, 7, 8, 9];

/// Brute-force verdict counts with independent arithmetic.
struct Brute {
    /// Accepted `(counts with ion H, mass, residual)`, sorted for set compare.
    accepted: Vec<(Composition, u32, i32)>,
    /// Accepted in spec visit order (the kept-prefix reference).
    in_order: Vec<(Composition, u32, i32)>,
    /// Ambiguous count.
    ambiguous: u32,
    /// True when the peak is not searched (sentinel or wide window).
    unavailable: bool,
}

fn adduct_params(adduct: u16) -> (i64, i64) {
    match adduct {
        1 => (1, 1),
        2 => (-1, -1),
        _ => panic!("test uses adducts 1 and 2 only"),
    }
}

/// Residual-bound numerator of a hypothesis composition (own arithmetic).
fn bound_of(counts: &Composition) -> u64 {
    let mut nda: u64 = 0;
    for e in 0..10 {
        nda += u64::from(counts[e]) * u64::from(RES[e]);
    }
    (nda + u64::from(ME_RES)).div_ceil(1000)
}

/// E_ion upper bound of a parent composition (own arithmetic).
fn e_ion_of(parent: &Composition) -> u64 {
    let mut nda: u64 = 0;
    for e in 0..10 {
        nda += u64::from(parent[e]) * u64::from(RES[e]);
    }
    (nda + 3 * 33 + u64::from(ME_RES)).div_ceil(1000)
}

/// Spec visit order written independently: index 1 upward, carbon least
/// significant.
fn visit_order(parent: &Composition) -> Vec<Composition> {
    let rad: Vec<u64> = HEAVY.iter().map(|&e| u64::from(parent[e]) + 1).collect();
    let total: u64 = rad.iter().product();
    (1..total)
        .map(|mut k| {
            let mut c: Composition = [0; 10];
            for (i, &e) in HEAVY.iter().enumerate() {
                c[e] = (k % rad[i]) as u16;
                k /= rad[i];
            }
            c
        })
        .collect()
}

/// Brute-force query: one peak under one parent composition.
struct BruteQuery<'a> {
    parent: &'a Composition,
    adduct: u16,
    mz: u32,
    uncertainty: u32,
    ppm: u32,
}

fn brute(q: &BruteQuery<'_>) -> Brute {
    let (parent, adduct, mz, uncertainty, ppm) =
        (q.parent, q.adduct, q.mz, q.uncertainty, q.ppm);
    let (h_a, z) = adduct_params(adduct);
    let t = if z > 0 {
        u64::from(mz) + u64::from(ME)
    } else {
        u64::from(mz) - u64::from(ME)
    };
    let tol = u64::from(mz) * u64::from(ppm) / 10_000_000;
    let half = tol + u64::from(uncertainty) + e_ion_of(parent);
    if uncertainty == u32::MAX || half > u64::from(MH) {
        return Brute {
            accepted: Vec::new(),
            in_order: Vec::new(),
            ambiguous: 0,
            unavailable: true,
        };
    }
    let h_cap = u64::from(parent[1]) + h_a.max(0) as u64 + 2;
    // Nested enumeration over heavy digits (any order: the accepted set and
    // the ambiguous count do not depend on it).
    let ctx = RecCtx {
        parent,
        t,
        tol,
        u: uncertainty,
        h_cap,
    };
    let mut acc: Vec<(Composition, u32, i32)> = Vec::new();
    let mut amb: u32 = 0;
    /// Shared brute-force inputs behind one peak.
    struct RecCtx<'a> {
        parent: &'a Composition,
        t: u64,
        tol: u64,
        u: u32,
        h_cap: u64,
    }
    fn rec(
        depth: usize,
        ctx: &RecCtx<'_>,
        cur: &mut Composition,
        acc: &mut Vec<(Composition, u32, i32)>,
        amb: &mut u32,
    ) {
        if depth == HEAVY.len() {
            if cur.iter().all(|&n| n == 0) {
                return;
            }
            let mut m: u64 = 0;
            for &e in &HEAVY {
                m += u64::from(cur[e]) * u64::from(MASS[e]);
            }
            for h in 0..=ctx.h_cap {
                let mass = m + h * u64::from(MH);
                if mass > u64::from(u32::MAX) {
                    continue;
                }
                let mut hyp = *cur;
                hyp[1] = h as u16;
                let e = bound_of(&hyp) + u64::from(ctx.u);
                let r = ctx.t.abs_diff(mass);
                if r + e <= ctx.tol {
                    // Accepted verdicts have r <= tol (tiny here), so the
                    // signed residual fits i32 exactly.
                    let signed = if mass >= ctx.t {
                        r as i32
                    } else {
                        -(r as i32)
                    };
                    acc.push((hyp, mass as u32, signed));
                } else if r > ctx.tol + e {
                    // Reject.
                } else {
                    *amb += 1;
                }
            }
            return;
        }
        let e = HEAVY[depth];
        for d in 0..=ctx.parent[e] {
            cur[e] = d;
            rec(depth + 1, ctx, cur, acc, amb);
        }
        cur[e] = 0;
    }
    let mut cur: Composition = [0; 10];
    rec(0, &ctx, &mut cur, &mut acc, &mut amb);
    // Residuals fit i32 for accepted verdicts (see push site).
    let mut accepted: Vec<(Composition, u32, i32)> = acc;
    accepted.sort();
    // Kept-prefix reference: accepted hypotheses in spec visit order.
    let set: BTreeSet<(Composition, u32, i32)> = accepted.iter().copied().collect();
    let mut in_order = Vec::new();
    for mut heavy in visit_order(parent) {
        let mut m: u64 = 0;
        for &e in &HEAVY {
            m += u64::from(heavy[e]) * u64::from(MASS[e]);
        }
        for h in 0..=h_cap {
            let mass = m + h * u64::from(MH);
            if mass > u64::from(u32::MAX) {
                continue;
            }
            heavy[1] = h as u16;
            let r = (mass as i64 - t as i64) as i32;
            if set.contains(&(heavy, mass as u32, r)) {
                in_order.push((heavy, mass as u32, r));
            }
        }
    }
    Brute {
        accepted,
        ambiguous: amb,
        in_order,
        unavailable: false,
    }
}

// ---------------------------------------------------------------------------
// Fixture helpers.
// ---------------------------------------------------------------------------

fn fixture() -> Value {
    let path =
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/ms2/chemistry_v0.json");
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

// ---------------------------------------------------------------------------
// Brute-force cross-check.
// ---------------------------------------------------------------------------

/// Sweep peaks: exact ion masses plus accept/ambiguous/reject boundary
/// values, clamped into `u32` range.
fn sweep_peaks(parent: &Composition, adduct: u16, u: u32, ppm: u32) -> Vec<u32> {
    let (h_a, z) = adduct_params(adduct);
    let mut heavy_full = [0u16; 10];
    for &e in &HEAVY {
        heavy_full[e] = parent[e];
    }
    // Fragments: full parent, half carbons, single carbon.
    let mut frags: Vec<(Composition, u32)> = Vec::new();
    let h0 = u64::from(parent[1]);
    for s in [-1i64, 0, 1] {
        let h = h0 as i64 + h_a + s;
        if h >= 0 {
            frags.push((heavy_full, h as u32));
        }
    }
    if parent[0] >= 2 {
        let mut half = [0u16; 10];
        half[0] = parent[0] / 2;
        for &e in &HEAVY[1..] {
            half[e] = parent[e];
        }
        let h = (h0 / 2) as i64 + h_a;
        if h >= 0 {
            frags.push((half, h as u32));
        }
    }
    let mut one = [0u16; 10];
    one[0] = 1;
    for s in [0i64, 1] {
        let h = 1 + h_a + s;
        if h >= 0 {
            frags.push((one, h as u32));
        }
    }
    let mut peaks = BTreeSet::new();
    for (heavy, ion_h) in &frags {
        let mut m: u64 = 0;
        for &e in &HEAVY {
            m += u64::from(heavy[e]) * u64::from(MASS[e]);
        }
        m += u64::from(*ion_h) * u64::from(MH);
        let mz0 = m as i64 - z * i64::from(ME);
        if mz0 <= 0 || mz0 > u64::from(u32::MAX) as i64 {
            continue;
        }
        let mut hyp = *heavy;
        hyp[1] = *ion_h as u16;
        let e0 = bound_of(&hyp) + u64::from(u);
        let tol = mz0 as u64 * u64::from(ppm) / 10_000_000;
        let edges: Vec<i64> = vec![
            0,
            tol as i64 - e0 as i64,
            tol as i64 - e0 as i64 + 1,
            tol as i64 + e0 as i64,
            tol as i64 + e0 as i64 + 1,
            tol as i64 + e0 as i64 + 5000,
        ];
        for d in edges {
            if d < 0 {
                continue;
            }
            let v = mz0 + d;
            if v > 0 && v <= u64::from(u32::MAX) as i64 {
                peaks.insert(v as u32);
            }
            if d > 0 && mz0 - d > 0 {
                peaks.insert((mz0 - d) as u32);
            }
        }
    }
    // A peak above the parent mass.
    let mut pm: u64 = 0;
    for e in 0..10 {
        pm += u64::from(parent[e]) * u64::from(MASS[e]);
    }
    let above = pm + 5_000_000;
    if above <= u64::from(u32::MAX) {
        peaks.insert(above as u32);
    }
    peaks.into_iter().collect()
}

fn check_parent(parent: &Composition, adduct: u16) {
    let (u, ppm) = (50u32, 100u32);
    let limits = IonLimits {
        work_max: 1_000_000,
        kept: 100_000,
    };
    for mz in sweep_peaks(parent, adduct, u, ppm) {
        // Adduct 2 needs mz >= 549 for a u32 target mass.
        if adduct == 2 && mz < ME {
            continue;
        }
        let b = brute(&BruteQuery {
            parent,
            adduct,
            mz,
            uncertainty: u,
            ppm,
        });
        assert!(!b.unavailable, "sweep peak {mz} must be searched");
        let got = ion_assign(parent, adduct, mz, u, ppm, &limits).expect("ion_assign runs");
        assert_eq!(got.status, 0, "large limits leave peak {mz} complete");
        let mut kept: Vec<(Composition, u32, i32)> = got
            .kept
            .iter()
            .map(|h| (h.counts, h.mass, h.residual))
            .collect();
        kept.sort();
        assert_eq!(kept, b.accepted, "accepted set at peak {mz}");
        assert_eq!(got.accepted as usize, b.accepted.len(), "accepted count {mz}");
        assert_eq!(got.ambiguous, b.ambiguous, "ambiguous count {mz}");
        let order: Vec<(Composition, u32, i32)> = got
            .kept
            .iter()
            .map(|h| (h.counts, h.mass, h.residual))
            .collect();
        assert_eq!(order, b.in_order, "kept visit-order prefix {mz}");
        for h in &got.kept {
            assert!(
                HEAVY.iter().any(|&e| h.counts[e] > 0),
                "no hydrogen-only ion at {mz}"
            );
            let t = if adduct == 1 {
                mz as i64 + 549
            } else {
                mz as i64 - 549
            };
            assert_eq!(h.residual as i64, h.mass as i64 - t, "residual is mass − t at {mz}");
        }
    }
}

#[test]
fn brute_force_matches_small_parents() {
    let parents: Vec<Composition> = vec![
        [3, 8, 0, 1, 0, 0, 0, 0, 0, 0], // C3H8O
        [2, 4, 0, 2, 0, 0, 0, 0, 0, 0], // C2H4O2
        [6, 6, 0, 0, 0, 0, 0, 0, 0, 0], // C6H6
        [1, 3, 0, 0, 0, 0, 0, 1, 0, 0], // CH3Cl
    ];
    for parent in &parents {
        for adduct in [1u16, 2] {
            check_parent(parent, adduct);
        }
    }
}

#[test]
fn brute_force_matches_fixture_molecules_with_heteroatoms() {
    for name in ["cysteine", "bromobenzene", "chloramphenicol-like", "acetonitrile"] {
        let g = graph_of(name);
        let comp = g.composition();
        for adduct in [1u16, 2] {
            check_parent(&comp, adduct);
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel-twin lane: inline verdict rule and visit budget.
// ---------------------------------------------------------------------------

/// The lane's inline rule equals `chem::decide` on a dense sweep covering
/// the accept/ambiguous/reject edges and values near `u32::MAX`.
#[test]
fn lane_verdict_matches_decide_on_dense_sweep() {
    use mamba3::models::ms2::chem::{Verdict, decide};
    let code_of = |o: u32, c: u32, e: u32, tol: u32| -> u32 {
        match decide(o, c, e, tol) {
            Verdict::Accept => LANE_ACCEPT,
            Verdict::Reject => LANE_REJECT,
            Verdict::Ambiguous => LANE_AMBIGUOUS,
        }
    };
    // Dense small sweep over tolerance, error and residual distance, both
    // mass directions.
    for tol in 0..=8u32 {
        for err in 0..=8u32 {
            for r in 0..=16u32 {
                let o = 5000u32;
                let c_hi = o + r;
                assert_eq!(
                    lane_verdict_u32(o, c_hi, err, tol),
                    code_of(o, c_hi, err, tol),
                    "tol {tol} err {err} r {r} above"
                );
                let c_lo = o - r;
                assert_eq!(
                    lane_verdict_u32(o, c_lo, err, tol),
                    code_of(o, c_lo, err, tol),
                    "tol {tol} err {err} r {r} below"
                );
            }
        }
    }
    // Edges from wide arithmetic: accept edge `r + e == tol`, the ambiguous
    // band and the reject edge `r == tol + e + 1`, clamped into `u32`.
    let mut edge_radii: Vec<u64> = vec![0, 1, u64::from(u32::MAX) - 1, u64::from(u32::MAX)];
    for tol in [0u64, 1, 2, 50, 1000, 429_496, u64::from(u32::MAX) - 1, u64::from(u32::MAX)] {
        for err in [0u64, 1, 2, 50, 1000, u64::from(u32::MAX)] {
            if err <= tol {
                edge_radii.push(tol - err);
            }
            edge_radii.push(tol + err);
            edge_radii.push(tol + err + 1);
        }
    }
    for tol in [0u32, 1, 50, 1000, u32::MAX - 1, u32::MAX] {
        for err in [0u32, 1, 50, 1000, u32::MAX] {
            for r in &edge_radii {
                if *r > u64::from(u32::MAX) {
                    continue;
                }
                let r = *r as u32;
                // Realise |o − c| == r without wrapping: small radii sit
                // above a base peak, large radii span from r down to 0.
                let (o, c) = if r <= u32::MAX - 5000 {
                    (5000u32, 5000 + r)
                } else {
                    (r, 0u32)
                };
                assert_eq!(o.abs_diff(c), r);
                assert_eq!(
                    lane_verdict_u32(o, c, err, tol),
                    code_of(o, c, err, tol),
                    "tol {tol} err {err} r {r}"
                );
            }
        }
    }
    // Values near `u32::MAX` in every seat.
    let max_cases: Vec<(u32, u32, u32, u32)> = vec![
        (u32::MAX, 0, 0, 0),
        (u32::MAX, u32::MAX, 0, 0),
        (0, u32::MAX, u32::MAX, u32::MAX),
        (u32::MAX, 0, u32::MAX, u32::MAX),
        (u32::MAX - 1, u32::MAX, 1, 0),
        (u32::MAX, u32::MAX - 100, 50, 49),
        (u32::MAX, u32::MAX - 100, 50, 50),
        (u32::MAX, u32::MAX - 100, 51, 50),
        (0, 0, u32::MAX, u32::MAX),
        (1, 0, u32::MAX, 0),
        (u32::MAX, u32::MAX / 2, u32::MAX / 2, u32::MAX / 2),
        (u32::MAX / 2, u32::MAX, u32::MAX / 2, u32::MAX / 2),
    ];
    for (o, c, e, tol) in max_cases {
        assert_eq!(
            lane_verdict_u32(o, c, e, tol),
            code_of(o, c, e, tol),
            "o {o} c {c} e {e} tol {tol}"
        );
    }
}

/// The lane's guarded visit budget equals the exact radix product reference,
/// including `work_max == u32::MAX` and the exact-`2^32` product.
#[test]
fn lane_visits_matches_product_reference() {
    fn reference(radices: &[u32; 9], work_max: u32) -> (u32, u32) {
        let bound = u128::from(work_max) + 1;
        let mut total: u128 = 1;
        let mut cut = false;
        for &r in radices {
            total *= u128::from(r);
            if total > bound {
                cut = true;
                break;
            }
        }
        let visits = if cut { work_max } else { (total - 1) as u32 };
        (visits, u32::from(cut))
    }
    let full = |r0: u32| -> [u32; 9] {
        let mut rad = [1u32; 9];
        rad[0] = r0;
        rad
    };
    let mut cases: Vec<([u32; 9], u32)> = vec![
        ([1; 9], 0),
        ([1; 9], u32::MAX),
        (full(2), 0),
        (full(2), 1),
        (full(2), u32::MAX),
        (full(3), 1),
        ([3, 2, 1, 1, 1, 1, 1, 1, 1], 3),
        ([u16::MAX as u32 + 1; 9], 3),
        ([u16::MAX as u32 + 1; 9], u32::MAX),
        // Exact 2^32 product: complete under MAX, exhausted under MAX − 1.
        ([65536, 65536, 1, 1, 1, 1, 1, 1, 1], u32::MAX),
        ([65536, 65536, 1, 1, 1, 1, 1, 1, 1], u32::MAX - 1),
        ([65536, 65536, 2, 1, 1, 1, 1, 1, 1], u32::MAX),
        ([2, 2, 2, 2, 2, 2, 2, 2, 2], 4095),
        ([2, 2, 2, 2, 2, 2, 2, 2, 2], 4096),
        ([2, 2, 2, 2, 2, 2, 2, 2, 2], u32::MAX),
    ];
    // Parents from the sweep tests under several budgets.
    for parent in [
        [3u16, 8, 0, 1, 0, 0, 0, 0, 0, 0],
        [8, 10, 4, 2, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    ] {
        let mut rad = [1u32; 9];
        for (slot, &e) in HEAVY.iter().enumerate() {
            rad[slot] = u32::from(parent[e]) + 1;
        }
        for w in [0u32, 1, 3, 4095, 4096, 1_000_000, u32::MAX - 1, u32::MAX] {
            cases.push((rad, w));
        }
    }
    for (radices, work_max) in cases {
        assert_eq!(
            lane_visits_u32(radices, work_max),
            reference(&radices, work_max),
            "radices {radices:?} work_max {work_max}"
        );
    }
}

// ---------------------------------------------------------------------------
// Superset property against the label recipe.
// ---------------------------------------------------------------------------

#[test]
fn accepted_hypotheses_cover_every_supported_mapping() {
    let f = fixture();
    let names: Vec<String> = f["molecules"]
        .as_array()
        .expect("molecules")
        .iter()
        .map(|m| m["name"].as_str().expect("name").to_string())
        .collect();
    let limits = IonLimits {
        work_max: 1_000_000,
        kept: 100_000,
    };
    let (u, ppm) = (50u32, 100u32);
    // Coverage per adduct (index 0/1 for adduct 1/2) per shift (-2..=2):
    // supported counts come from the test's own rule, never production.
    let mut supported = [[0u32; 5]; 2];
    let mut covered = [[0u32; 5]; 2];
    for name in &names {
        let g = graph_of(name);
        let parent = g.composition();
        let embeddings = enumerate_embeddings(&g, &RecipeLimits::V0);
        for emb in &embeddings {
            // Own P6.8 support rule, written out: |s| <= min(c, 2).
            let cap = emb.boundary.min(2) as i32;
            for s in -2..=2i32 {
                if s.abs() > cap {
                    continue;
                }
                for (ai, adduct) in [1u16, 2].iter().enumerate() {
                    let h_a: i64 = if *adduct == 1 { 1 } else { -1 };
                    let z: i64 = h_a;
                    // Fragment composition from the structure itself.
                    let frag = g.induced(&emb.atoms).expect("embedding induces");
                    let comp = frag.composition();
                    let hg = i64::from(comp[1]);
                    let h = hg + h_a + i64::from(s);
                    // Negative (or unrepresentable) ion hydrogen: no mapping.
                    if h < 0 || h > i64::from(u16::MAX) {
                        // The production helper must agree it is None (a
                        // mapping that cannot form has no ion).
                        if h < 0 {
                            assert!(
                                embedding_ion(&g, emb, *adduct, s)
                                    .expect("embedding_ion runs")
                                    .is_none(),
                                "{name} boundary {} shift {s} adduct {adduct}: negative hydrogen must be None",
                                emb.boundary
                            );
                        }
                        continue;
                    }
                    // Own mass arithmetic from the test literals: the full
                    // neutral fragment mass over all 10 elements.
                    let mut neutral: u64 = 0;
                    for e in 0..10 {
                        neutral += u64::from(comp[e]) * u64::from(MASS[e]);
                    }
                    let ion_mass_i =
                        neutral as i64 + (h_a + i64::from(s)) * MH as i64;
                    let mz_i = ion_mass_i - z * ME as i64;
                    if mz_i <= 0 || mz_i > u64::from(u32::MAX) as i64 {
                        continue;
                    }
                    if ion_mass_i <= 0 || ion_mass_i > u64::from(u32::MAX) as i64 {
                        continue;
                    }
                    let mz = mz_i as u32;
                    let ion_mass = ion_mass_i as u32;
                    let mut ion_comp = comp;
                    ion_comp[1] = h as u16;
                    // The production helper agrees with the independent
                    // oracle: a supported mapping is `Some` with the same
                    // composition and mass (a silent `None` for some shift
                    // or adduct would skip coverage without failing).
                    let produced =
                        embedding_ion(&g, emb, *adduct, s).expect("embedding_ion runs");
                    let Some((produced_comp, produced_ion)) = produced else {
                        panic!(
                            "{name} boundary {} shift {s} adduct {adduct}: embedding_ion returned None for a supported mapping",
                            emb.boundary
                        )
                    };
                    assert_eq!(
                        produced_comp, ion_comp,
                        "{name} boundary {} shift {s} adduct {adduct}: composition",
                        emb.boundary
                    );
                    assert_eq!(
                        produced_ion.mz, mz,
                        "{name} boundary {} shift {s} adduct {adduct}: mass",
                        emb.boundary
                    );
                    supported[ai][(s + 2) as usize] += 1;
                    let assign =
                        ion_assign(&parent, *adduct, mz, u, ppm, &limits).expect("ion_assign runs");
                    assert_eq!(
                        assign.status, 0,
                        "large limits leave {name} boundary {} shift {s} adduct {adduct} complete",
                        emb.boundary
                    );
                    // An independently supported mapping must be accepted:
                    // no production `None` may skip it silently.
                    let hyp = assign.kept.iter().find(|hh| hh.counts == ion_comp);
                    let Some(hyp) = hyp else {
                        panic!(
                            "{name} boundary {} shift {s} adduct {adduct}: independently derived ion missing from accepted",
                            emb.boundary
                        )
                    };
                    assert_eq!(hyp.mass, ion_mass, "hypothesis mass for {name}");
                    // The shift recomputed from the hypothesis equals s.
                    assert_eq!(
                        i64::from(hyp.counts[1]) - hg - h_a,
                        i64::from(s),
                        "shift round-trips for {name}"
                    );
                    covered[ai][(s + 2) as usize] += 1;
                }
            }
        }
    }
    for (ai, adduct) in [1u16, 2].iter().enumerate() {
        for s in -2..=2i32 {
            let sup = supported[ai][(s + 2) as usize];
            let cov = covered[ai][(s + 2) as usize];
            assert!(
                sup > 0,
                "fixture supports adduct {adduct} shift {s} (got {sup} cases)"
            );
            assert_eq!(
                cov, sup,
                "every supported adduct {adduct} shift {s} mapping is covered"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Electron sign.
// ---------------------------------------------------------------------------

#[test]
fn electron_sign_matches_review_numbers() {
    // The review's numeric example lives in the t relation t = mz + z_a m_e.
    assert_eq!(100_000_000u32 - 549, 99_999_451);
    assert_eq!(100_000_000u32 + 549, 100_000_549);
    // End to end on methane: the same atomic mass maps to m/z 549 lower for
    // [M+H]+ and 549 higher for [M-H]-.
    let methane: Composition = [1, 4, 0, 0, 0, 0, 0, 0, 0, 0];
    let mass = 12_000_000 + 4 * 1_007_825;
    let pos = mamba3::models::ms2::ion(&methane, 1, 0)
        .expect("ion runs")
        .expect("ion exists");
    let neg = mamba3::models::ms2::ion(&methane, 2, 0)
        .expect("ion runs")
        .expect("ion exists");
    assert_eq!(pos.mz, mass + 1_007_825 - 549);
    assert_eq!(neg.mz, mass - 1_007_825 + 549);
    assert_eq!(pos.mz, 17_038_576);
    assert_eq!(neg.mz, 15_024_024);
    // Through ion_assign: the hypothesis mass is accepted at mz = mass − 549
    // (positive) and mz = mass + 549 (negative) with residual 0.
    let limits = IonLimits {
        work_max: 4096,
        kept: 4,
    };
    let hyp_mass = 12_000_000 + 5 * 1_007_825;
    let a = ion_assign(&methane, 1, hyp_mass - 549, 50, 100, &limits).expect("runs");
    assert!(a.kept.iter().any(|h| h.mass == hyp_mass && h.residual == 0));
    let b = ion_assign(&methane, 2, hyp_mass + 549, 50, 100, &limits).expect("runs");
    assert!(b.kept.iter().any(|h| h.mass == hyp_mass && h.residual == 0));
}

// ---------------------------------------------------------------------------
// Radix guard.
// ---------------------------------------------------------------------------

#[test]
fn radix_guard_completeness_and_exhaustion() {
    let small = IonLimits {
        work_max: 3,
        kept: 100,
    };
    // C1 O1: radices 2 * 2 = 4, exactly 3 non-empty vectors: complete.
    let c1o1: Composition = [1, 0, 0, 1, 0, 0, 0, 0, 0, 0];
    let a = ion_assign(&c1o1, 1, 12_000_000, 50, 100, &small).expect("runs");
    assert_eq!(a.status & ION_SEARCH_EXHAUSTED, 0, "product 4 with max 3 is complete");
    // C2 O1: radices 3 * 2 = 6, 5 non-empty vectors > 3: exhausted.
    let c2o1: Composition = [2, 0, 0, 1, 0, 0, 0, 0, 0, 0];
    let b = ion_assign(&c2o1, 1, 12_000_000, 50, 100, &small).expect("runs");
    assert_ne!(b.status & ION_SEARCH_EXHAUSTED, 0, "product 6 with max 3 is exhausted");
    // A radix product overflowing u32 (65536^2 = 2^32) never overflows the
    // guard and is exhausted under a small budget.
    let huge: Composition = [u16::MAX, 0, u16::MAX, 0, 0, 0, 0, 0, 0, 0];
    let c = ion_assign(&huge, 1, 12_000_000, 50, 100, &small).expect("runs");
    assert_ne!(c.status & ION_SEARCH_EXHAUSTED, 0, "overflowing product is exhausted");
}

// ---------------------------------------------------------------------------
// Window restriction, sentinel, overflow, above-parent, J truncation.
// ---------------------------------------------------------------------------

#[test]
fn window_restriction_sentinel_and_overflow() {
    let parent: Composition = [3, 8, 0, 1, 0, 0, 0, 0, 0, 0];
    let mz = 60_000_000u32;
    let limits = IonLimits {
        work_max: 4096,
        kept: 4,
    };
    // Huge uncertainty: half_p saturates past one hydrogen mass.
    let wide = ion_assign(&parent, 1, mz, 2_000_000, 100, &limits).expect("runs");
    assert_eq!(wide.status & ION_UNAVAILABLE, ION_UNAVAILABLE);
    assert_eq!(wide.accepted, 0);
    assert!(wide.kept.is_empty());
    // The unknown-precision sentinel disables decisions the same way.
    let sent = ion_assign(&parent, 1, mz, u32::MAX, 100, &limits).expect("runs");
    assert_eq!(sent.status & ION_UNAVAILABLE, ION_UNAVAILABLE);
    // The exact boundary half_p == m_H is still searched.
    let tol = u64::from(mz) * 100 / 10_000_000;
    let e_ion = e_ion_of(&parent);
    assert!(MH as u64 > tol + e_ion);
    let edge_u = (MH as u64 - tol - e_ion) as u32;
    let edge = ion_assign(&parent, 1, mz, edge_u, 100, &limits).expect("runs");
    assert_eq!(edge.status & ION_UNAVAILABLE, 0, "half_p == m_H is searched");
    let over = ion_assign(&parent, 1, mz, edge_u + 1, 100, &limits).expect("runs");
    assert_eq!(over.status & ION_UNAVAILABLE, ION_UNAVAILABLE);
    // A target mass outside u32 is mass_overflow, both directions.
    let err = ion_assign(&parent, 2, 100, 50, 100, &limits).expect_err("t underflows");
    assert!(err.to_string().contains("mass_overflow"), "got {err}");
    let err = ion_assign(&parent, 1, u32::MAX, 50, 100, &limits).expect_err("t overflows");
    assert!(err.to_string().contains("mass_overflow"), "got {err}");
    // Unknown adduct and over-wide tolerance are errors, not statuses.
    let err = ion_assign(&parent, 7, mz, 50, 100, &limits).expect_err("bad adduct");
    assert!(err.to_string().contains("adduct"), "got {err}");
    ion_assign(&parent, 1, mz, 50, 1001, &limits).expect_err("ppm bound");
    // A peak above the parent mass accepts nothing but stays complete.
    let above = ion_assign(&parent, 1, 70_000_000, 50, 100, &limits).expect("runs");
    assert_eq!(above.accepted, 0);
    assert_eq!(above.status, 0);
    // A hydrogen-only parent has no heavy vector to visit.
    let honly: Composition = [0, 8, 0, 0, 0, 0, 0, 0, 0, 0];
    let h = ion_assign(&honly, 1, 8_000_000, 50, 100, &limits).expect("runs");
    assert_eq!(h.accepted, 0);
    assert_eq!(h.status, 0);
}

#[test]
fn capacity_truncation_keeps_visit_order_prefix() {
    // Caffeine at 1000 ppm: the C8O-ion peak accepts 4 hypotheses through
    // near mass coincidences (e.g. C7N2 within 11,233 units), so J = 2
    // truncates honestly.
    let parent: Composition = [8, 10, 4, 2, 0, 0, 0, 0, 0, 0];
    // Exact m/z of the C8 O1 ion with ion hydrogen 10 (adduct 1).
    let mz = 8 * 12_000_000 + 15_994_915 + 11 * 1_007_825 - 549;
    let big = IonLimits {
        work_max: 1_000_000,
        kept: 100_000,
    };
    let small = IonLimits {
        work_max: 1_000_000,
        kept: 2,
    };
    let full = ion_assign(&parent, 1, mz, 50, 1000, &big).expect("runs");
    assert_eq!(full.status, 0);
    assert!(full.accepted >= 3, "peak accepts {}", full.accepted);
    let cut = ion_assign(&parent, 1, mz, 50, 1000, &small).expect("runs");
    assert_ne!(cut.status & ION_CAPACITY_EXCEEDED, 0);
    assert_eq!(cut.kept.len(), 2);
    assert_eq!(cut.kept, full.kept[..2]);
    assert_eq!(cut.accepted, full.accepted);
    assert_eq!(cut.ambiguous, full.ambiguous);
}

// ---------------------------------------------------------------------------
// Label sets.
// ---------------------------------------------------------------------------

/// Real recipe labels on 2-butanol with a peak at every (embedding, s = 0)
/// ion mass, so several targets anchor.
fn butanol_labels() -> (
    mamba3::models::ms2::targets::Labels,
    Vec<Peak>,
    u32,
) {
    let g = graph_of("2-butanol");
    let candidates = Candidates::new(&g, &RecipeLimits::V0).expect("candidates build");
    let embeddings = enumerate_embeddings(&g, &RecipeLimits::V0);
    let mut peaks = Vec::new();
    let mut whole_id = 0u32;
    for (i, emb) in embeddings.iter().enumerate() {
        let frag = g.induced(&emb.atoms).expect("embedding induces");
        let comp = frag.composition();
        // Ion hydrogen H(g) + h_a with s = 0 under adduct 1; masses are
        // test-local literals, the structure comes from the graph.
        let mut m: u64 = 0;
        for &e in &HEAVY {
            m += u64::from(comp[e]) * u64::from(MASS[e]);
        }
        let ion_h = u64::from(comp[1]) + 1;
        m += ion_h * u64::from(MH);
        let mz = m as i64 - 549;
        assert!(mz > 0 && mz <= u64::from(u32::MAX) as i64);
        let id = 100 + i as u32;
        peaks.push(Peak {
            id,
            mz: mz as u32,
            intensity: 1.0 - i as f64 * 0.01,
        });
        if emb.atoms.len() == g.atoms().len() {
            whole_id = id;
        }
    }
    assert_ne!(whole_id, 0, "the whole molecule is a candidate");
    let labels = candidates
        .label(&peaks, 1, 100, 50)
        .expect("labeling runs");
    assert!(!labels.targets.is_empty(), "crafted peaks anchor targets");
    (labels, peaks, whole_id)
}

fn raw_map(id: u32) -> Option<u32> {
    if (100..400).contains(&id) {
        Some(id - 100)
    } else {
        None
    }
}

#[test]
fn label_sets_dedup_cap_and_overflow() {
    let (labels, _, whole_id) = butanol_labels();
    let whole_raw = whole_id - 100;
    let full = ion_labels(&labels, 1, raw_map, 64);
    assert!(!full.labels.is_empty());
    // Sorted by (raw index, counts lexicographically).
    let keys: Vec<(u32, Composition)> = full
        .labels
        .iter()
        .map(|l| (l.raw_index, l.counts))
        .collect();
    let mut sorted = keys.clone();
    sorted.sort();
    assert_eq!(keys, sorted);
    assert_eq!(full.overflow, 0);
    // The whole-molecule ion composition (C4, H10 + 1) is labeled.
    let whole: Composition = [4, 11, 0, 1, 0, 0, 0, 0, 0, 0];
    assert!(
        full.labels
            .iter()
            .any(|l| l.raw_index == whole_raw && l.counts == whole),
        "whole ion labeled"
    );
    // Duplicating a target changes nothing: labels are deduplicated.
    let mut doubled = labels.clone();
    doubled.targets.push(doubled.targets[0].clone());
    let dedup = ion_labels(&doubled, 1, raw_map, 64);
    assert_eq!(dedup.labels, full.labels);
    assert_eq!(dedup.overflow, 0);
    // Cap: deterministic prefix plus counted overflow.
    let total = full.labels.len();
    assert!(total >= 2, "need 2+ labels for the cap test, got {total}");
    let capped = ion_labels(&labels, 1, raw_map, 1);
    assert_eq!(capped.labels, full.labels[..1]);
    assert_eq!(capped.overflow, total - 1);
    assert_eq!(ion_labels(&labels, 1, raw_map, 1), capped, "cap order deterministic");
    // Anchors of peaks outside the batch never appear.
    assert!(full.labels.iter().all(|l| l.raw_index < 300));
    // An anchor whose peak id is absent from the batch changes nothing: the
    // complete output (labels and overflow) equals the unchanged baseline.
    let mut with_absent = labels.clone();
    with_absent.targets[0].anchors.push((999_999, 0));
    assert_eq!(
        ion_labels(&with_absent, 1, raw_map, 64),
        full,
        "absent peak id is skipped"
    );
}

#[test]
fn label_negative_hydrogen_shifts_and_adduct_two() {
    let (labels, _, whole_id) = butanol_labels();
    let whole_raw = whole_id - 100;
    // Adduct 2 with zero shifts: the whole ion loses one hydrogen.
    let neg = ion_labels(&labels, 2, raw_map, 64);
    let whole_adduct2: Composition = [4, 9, 0, 1, 0, 0, 0, 0, 0, 0];
    assert!(
        neg.labels
            .iter()
            .any(|l| l.raw_index == whole_raw && l.counts == whole_adduct2),
        "adduct 2 whole ion labeled"
    );
    // Non-zero shifts: every anchor shifted by +1 under adduct 1.
    let mut shifted = labels.clone();
    for t in &mut shifted.targets {
        for a in &mut t.anchors {
            a.1 += 1;
        }
    }
    let plus = ion_labels(&shifted, 1, raw_map, 64);
    let whole_plus: Composition = [4, 12, 0, 1, 0, 0, 0, 0, 0, 0];
    assert!(
        plus.labels
            .iter()
            .any(|l| l.raw_index == whole_raw && l.counts == whole_plus),
        "shift +1 whole ion labeled"
    );
    // Negative ion hydrogen is omitted under either adduct.
    let mut negh = labels.clone();
    for t in &mut negh.targets {
        for a in &mut t.anchors {
            a.1 = -100;
        }
    }
    let empty1 = ion_labels(&negh, 1, raw_map, 64);
    assert!(empty1.labels.is_empty() && empty1.overflow == 0);
    let empty2 = ion_labels(&negh, 2, raw_map, 64);
    assert!(empty2.labels.is_empty() && empty2.overflow == 0);
}

#[test]
fn label_mask_states_and_partial_overlap() {
    let comp_a: Composition = [1, 4, 0, 0, 0, 0, 0, 0, 0, 0];
    let comp_b: Composition = [1, 2, 0, 0, 0, 0, 0, 0, 0, 0];
    let comp_c: Composition = [2, 6, 0, 0, 0, 0, 0, 0, 0, 0];
    let comp_d: Composition = [2, 4, 0, 0, 0, 0, 0, 0, 0, 0];
    let hyps = vec![
        IonAssignment {
            accepted: 1,
            ambiguous: 0,
            kept: vec![IonHypothesis {
                counts: comp_a,
                mass: 1,
                residual: 0,
            }],
            status: 0,
        },
        IonAssignment {
            accepted: 1,
            ambiguous: 0,
            kept: vec![IonHypothesis {
                counts: comp_b,
                mass: 2,
                residual: 0,
            }],
            status: 0,
        },
        IonAssignment {
            accepted: 0,
            ambiguous: 0,
            kept: Vec::new(),
            status: ION_UNAVAILABLE,
        },
    ];
    let labels = IonLabels {
        labels: vec![
            IonLabel {
                raw_index: 3,
                counts: comp_a,
            },
            IonLabel {
                raw_index: 3,
                counts: comp_d,
            },
            IonLabel {
                raw_index: 5,
                counts: comp_c,
            },
        ],
        overflow: 0,
    };
    let (mask, state) = label_mask(&[3, 5, u32::MAX], &hyps, &labels, 2);
    // Slot 0 is true partial (state 3): comp_a kept, comp_d of the same peak
    // not; unassigned is 0.
    assert_eq!(state, vec![3, 2, 0]);
    assert_eq!(&mask[0..3], &[1.0, 0.0, 0.0]);
    // Slots outside states 1 and 3 are the one-hot of unassigned.
    assert_eq!(&mask[3..6], &[0.0, 0.0, 1.0]);
    assert_eq!(&mask[6..9], &[0.0, 0.0, 1.0]);
    // A peak with no label at all is state 0 with a one-hot row.
    let (mask, state) = label_mask(&[11], &hyps[..1], &labels, 2);
    assert_eq!(state, vec![0]);
    assert_eq!(&mask[0..3], &[0.0, 0.0, 1.0]);
}

// ---------------------------------------------------------------------------
// Evidence.
// ---------------------------------------------------------------------------

fn assignment_of(counts_list: &[(Composition, i32)], status: u32) -> IonAssignment {
    IonAssignment {
        accepted: counts_list.len() as u32,
        ambiguous: 0,
        kept: counts_list
            .iter()
            .map(|(c, r)| IonHypothesis {
                counts: *c,
                mass: 0,
                residual: *r,
            })
            .collect(),
        status,
    }
}

/// Decode an offset-binary shift/residual word of a packed evidence row.
fn unbias(ob: u32) -> i32 {
    ob.wrapping_sub(LANE_BIAS) as i32
}

#[test]
fn evidence_reviewer_cases() {
    // Acetylene fragment CH: one carbon with one parent hydrogen and no
    // bonds left has open valence 3, so c_lo = 1 and c_hi = 3.
    let ch: Composition = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let hyp = |h: u16| {
        let mut c = ch;
        c[1] = h;
        c
    };
    let hyps = vec![
        assignment_of(&[(hyp(2), 5)], 0),   // s = 0: universal
        assignment_of(&[(hyp(4), -3)], 0),  // s = +2: consistent only
        assignment_of(&[(hyp(5), 7)], 0),   // s = +3: unassigned
        assignment_of(&[], 0),              // no matching heavy vector
    ];
    let sum = evidence_status(&ch, 1, &[3], 1, &hyps, 4);
    // Only mass-consistent shifts count: s = 0 (universal) and s = +2
    // (consistent only); s = +3 is unsupported and contributes nothing.
    assert_eq!(sum.count, 2);
    assert_eq!(sum.status, 2);
    assert_eq!(&sum.peak[..2], &[0, 1]);
    assert_eq!(&sum.hypothesis[..2], &[0, 0]);
    assert_eq!(unbias(sum.shift_ob[0]), 0);
    assert_eq!(unbias(sum.shift_ob[1]), 2);
    assert_eq!(sum.residual_ob[0], 5u32.wrapping_add(LANE_BIAS));
    assert_eq!(sum.residual_ob[1], (-3i32 as u32).wrapping_add(LANE_BIAS));
    // The record area past the qualifying count stays zero.
    assert_eq!(&sum.peak[2..], &[0, 0]);
    assert_eq!(&sum.shift_ob[2..], &[0, 0]);
    assert_eq!(&sum.residual_ob[2..], &[0, 0]);
    // The flat row is status, count, then the qualifying records in peak order.
    assert_eq!(
        &sum.to_row()[..14],
        &[
            2,
            2,
            0,
            0,
            LANE_BIAS,
            5u32.wrapping_add(LANE_BIAS),
            1,
            0,
            2u32.wrapping_add(LANE_BIAS),
            (-3i32 as u32).wrapping_add(LANE_BIAS),
            0,
            0,
            0,
            0,
        ]
    );
    // A whole molecule has no open valence: only s = 0 is consistent.
    let c2: Composition = [2, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let hyp2 = |h: u16| {
        let mut c = c2;
        c[1] = h;
        c
    };
    let hyps = vec![
        assignment_of(&[(hyp2(3), 1)], 0), // H(g) = 2, h_a = 1: s = 0
        assignment_of(&[(hyp2(4), 2)], 0), // s = +1: outside min(0, 2)
    ];
    let sum = evidence_status(&c2, 2, &[], 1, &hyps, 4);
    assert_eq!(sum.count, 1);
    assert_eq!(sum.status, 2);
    assert_eq!(&sum.peak[..1], &[0]);
    assert_eq!(unbias(sum.shift_ob[0]), 0);
    assert_eq!(&sum.peak[1..], &[0, 0, 0]);
    assert_eq!(&sum.to_row()[6..], &[0u32; 12][..]);
    // Review finding 2: a whole 2-butanol candidate (zero open valence,
    // positive adduct) with four peaks at s = +1 followed by one at s = 0.
    // Only the last peak is candidate evidence: status 2, count 1, and the
    // four unsupported shifts leave no records.
    let butanol = graph_of("2-butanol");
    let parent = butanol.composition();
    let hg_butanol = u32::from(parent[HYDROGEN]);
    let mk = |s: i64, r: i32| {
        let mut c = parent;
        c[HYDROGEN] = (hg_butanol as i64 + 1 + s) as u16;
        (c, r)
    };
    let hyps = vec![
        assignment_of(&[mk(1, 11)], 0),
        assignment_of(&[mk(1, 12)], 0),
        assignment_of(&[mk(1, 13)], 0),
        assignment_of(&[mk(1, 14)], 0),
        assignment_of(&[mk(0, 15)], 0),
    ];
    let open = vec![0u8; butanol.atoms().len()];
    assert!(open.iter().all(|&o| o == 0), "whole molecule has no open valence");
    let sum = evidence_status(&parent, hg_butanol, &open, 1, &hyps, 4);
    assert_eq!(sum.count, 1);
    assert_eq!(sum.status, 2);
    assert_eq!(&sum.peak[..1], &[4]);
    assert_eq!(&sum.hypothesis[..1], &[0]);
    assert_eq!(unbias(sum.shift_ob[0]), 0);
    assert_eq!(sum.residual_ob[0], 15u32.wrapping_add(LANE_BIAS));
    assert_eq!(&sum.to_row()[6..], &[0u32; 12][..]);
    // Incomplete support carries bit 7 next to any status.
    let hyps = vec![assignment_of(&[(hyp(2), 5)], ION_SEARCH_EXHAUSTED)];
    let sum = evidence_status(&ch, 1, &[3], 1, &hyps, 4);
    assert_eq!(sum.status, 2 | EVIDENCE_SUPPORT_INCOMPLETE);
    assert_eq!(sum.count, 1);
    let hyps = vec![assignment_of(&[], ION_CAPACITY_EXCEEDED)];
    let sum = evidence_status(&ch, 1, &[3], 1, &hyps, 4);
    assert_eq!(sum.status, EVIDENCE_SUPPORT_INCOMPLETE);
    assert_eq!(sum.count, 0);
    // More qualifying matches than slots: the count stays total, the records
    // are the first E qualifying matches in peak order.
    let many: Vec<IonAssignment> = (0..6)
        .map(|h| assignment_of(&[(hyp(2), h)], 0))
        .collect();
    let sum = evidence_status(&ch, 1, &[3], 1, &many, 4);
    assert_eq!(sum.count, 6);
    assert_eq!(sum.status, 2);
    assert_eq!(&sum.peak, &[0, 1, 2, 3]);
    assert_eq!(
        &sum.residual_ob,
        &[
            LANE_BIAS,
            1u32.wrapping_add(LANE_BIAS),
            2u32.wrapping_add(LANE_BIAS),
            3u32.wrapping_add(LANE_BIAS),
        ]
    );
    // An unknown adduct yields a zeroed row.
    let hyps = vec![assignment_of(&[(hyp(2), 5)], 0)];
    let sum = evidence_status(&ch, 1, &[3], 7, &hyps, 4);
    assert_eq!(sum.to_row(), vec![0u32; EVIDENCE_ROW_WORDS]);
}

#[test]
fn evidence_lane_matches_status_on_decoded_inputs() {
    use mamba3::models::ms2::grammar::{ADD_ATOM, START, STOP};
    use mamba3::models::ms2::ion::evidence_lane;
    // Trace: START, ADD_ATOM(type 4: carbon with 3 parent hydrogens), STOP,
    // with open valence 0 on the single atom.
    let steps = 3u32;
    let atoms = 1u32;
    let stride = steps as usize * 4 + atoms as usize + 4;
    let mut actions = vec![0u32; stride];
    actions[0] = u32::from(START);
    actions[4] = u32::from(ADD_ATOM);
    actions[5] = 4;
    actions[8] = u32::from(STOP);
    actions[12] = 0;
    actions[13] = 3;
    actions[14] = 1;
    // One peak whose hypothesis is C1 H4: s = 4 − 3 − 1 = 0, universal.
    let mut c: Composition = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    c[1] = 4;
    let hyps = vec![assignment_of(&[(c, 9)], 0)];
    let mut ion = vec![0u32; 4 * 12];
    for e in 0..10 {
        ion[e] = u32::from(c[e]);
    }
    ion[11] = 9u32.wrapping_add(LANE_BIAS);
    let ion_meta = vec![1u32, 0, 1, 0];
    let mut row = vec![0u32; EVIDENCE_ROW_WORDS];
    evidence_lane(&actions, &[0, 1], &ion, &ion_meta, 0, steps, atoms, 1, 1, 1, 4, &mut row);
    let via_status = evidence_status(&[1, 3, 0, 0, 0, 0, 0, 0, 0, 0], 3, &[0], 1, &hyps, 4);
    assert_eq!(row, via_status.to_row());
    assert_eq!(via_status.status, 2);
    assert_eq!(via_status.count, 1);
    assert_eq!(unbias(via_status.shift_ob[0]), 0);
}

#[test]
fn mapping_support_boundaries() {
    assert!(mapping_is_supported(0, 0));
    assert!(!mapping_is_supported(0, 1));
    assert!(!mapping_is_supported(0, -1));
    assert!(mapping_is_supported(1, 1));
    assert!(mapping_is_supported(1, -1));
    assert!(!mapping_is_supported(1, 2));
    assert!(mapping_is_supported(2, 2));
    assert!(mapping_is_supported(2, -2));
    assert!(!mapping_is_supported(2, 3));
    assert!(mapping_is_supported(99, 2));
    assert!(!mapping_is_supported(99, -3));
}

#[test]
fn element_table_matches_production() {
    use mamba3::models::ms2::chem::ATOM_TYPES;
    use mamba3::models::ms2::ion::{
        HEAVY_MASS_U32, HEAVY_RES_U32, atom_fields_u32, heavy_mass_u32, heavy_res_u32,
    };
    for (e, table) in ELEMENTS.iter().enumerate() {
        assert_eq!(table.mass, MASS[e], "mass of {}", table.symbol);
    }
    // The kernel-portable heavy tables match the production table in HEAVY
    // order, and the if-chain selectors match the tables.
    const HEAVY_ORDER: [usize; 9] = [0, 2, 3, 4, 5, 6, 7, 8, 9];
    for (s, &e) in HEAVY_ORDER.iter().enumerate() {
        assert_eq!(HEAVY_MASS_U32[s], MASS[e], "heavy mass slot {s}");
        assert_eq!(HEAVY_MASS_U32[s], ELEMENTS[e].mass, "heavy mass slot {s}");
        assert_eq!(HEAVY_RES_U32[s], RES[e], "heavy residual slot {s}");
        assert_eq!(heavy_mass_u32(s as u32), HEAVY_MASS_U32[s], "mass selector {s}");
        assert_eq!(heavy_res_u32(s as u32), HEAVY_RES_U32[s], "residual selector {s}");
    }
    // The packed atom-type fields match the production atom-type table.
    for ty in 1..=17u32 {
        let entry = &ATOM_TYPES[ty as usize - 1];
        assert_eq!(entry.id as u32, ty, "atom type id order");
        let fields = atom_fields_u32(ty);
        assert_eq!(fields / 65536, entry.element as u32, "element of type {ty}");
        assert_eq!(
            fields - (fields / 65536) * 65536,
            u32::from(entry.hydrogens),
            "hydrogens of type {ty}"
        );
    }
}
