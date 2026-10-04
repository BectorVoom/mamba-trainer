//! K2 kernel-versus-twin tests for fragment-ion assignment.
//!
//! Every device call runs on poisoned outputs, is followed by
//! [`check_launches`], and is compared element-for-element with the host
//! twin lane: a dropped launch (stale poison) or a wrong word fails. Sizes
//! stay small on the CPU runtime; the supervisor runs the same file on wgpu.

#![cfg(feature = "backend")]

use mamba3::backend::{Device, check_launches};
use mamba3::backends::Auto;
use mamba3::models::ms2::chem::{ELECTRON_RESIDUAL_NDA, ELEMENTS, HYDROGEN};
use mamba3::models::ms2::grammar::{
    ADD_ATOM, CANONICAL_WORK_LIMIT, Limits, START, STOP, canonical_trace, replay,
};
use mamba3::models::ms2::ion::{
    EVIDENCE_ROW_WORDS, EVIDENCE_SLOTS, EVIDENCE_SUPPORT_INCOMPLETE, ION_CAPACITY_EXCEEDED,
    ION_SEARCH_EXHAUSTED, ION_UNAVAILABLE, IonAssignment, IonHypothesis, LANE_BIAS,
    evidence_features_lane, evidence_lane, evidence_lane_scored, evidence_status, ion_assign_lane,
    label_mask_lane,
};
use mamba3::tensor::ops::random::Rng;
use mamba3::models::ms2::targets::{RecipeLimits, enumerate_embeddings};
use mamba3::models::ms2::twin::peak_select;
use mamba3::models::ms2::{Composition, MolGraph};
use mamba3::tensor::Tensor;
use mamba3::tensor::ops::index::IdTensor;
use mamba3::tensor::ops::ms2_identity::{check_device_len, check_device_scalar};
use mamba3::tensor::ops::ms2_ion;
use serde_json::Value;

type R = Auto;

fn dev() -> Device<R> {
    Device::<R>::default()
}

fn upload_ids(data: &[u32], shape: Vec<usize>, device: &Device<R>) -> IdTensor<R> {
    IdTensor::from_slice(data, shape, device).unwrap()
}

fn upload_f(data: &[f32], shape: Vec<usize>, device: &Device<R>) -> Tensor<R, f32> {
    Tensor::<R, f32>::from_f32(data, shape, device).unwrap()
}

fn poison_ids(len: usize, device: &Device<R>) -> Vec<u32> {
    let _ = device;
    vec![0xDEAD_BEEF; len]
}

fn poison_f(len: usize) -> Vec<f32> {
    vec![f32::NAN; len]
}

fn assert_ids(actual: &[u32], expected: &[u32], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_eq!(*a, *e, "{what}: word {i} differs");
    }
}

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

// ---------------------------------------------------------------------------
// Assignment harness: twin lanes versus the kernel on poisoned outputs.
// ---------------------------------------------------------------------------

/// One spectrum's assignment inputs for [`run_assign_kernel`].
struct AssignSpectrum {
    formulas: Vec<Composition>,
    adduct: u16,
    ppm: u32,
    uncertainty: u32,
    peaks: Vec<u32>,
    peak_count: u32,
}

/// Run the twin [`ion_assign_lane`] per `(b, f, p)` and the
/// [`ms2_ion::ion_assign`] kernel on poisoned outputs; compare every word.
fn run_assign_kernel(spectra: &[AssignSpectrum], j: u32, work_max: u32) -> (Vec<u32>, Vec<u32>) {
    let b = spectra.len();
    let f = spectra.iter().map(|s| s.formulas.len()).max().unwrap_or(0);
    let n = spectra.iter().map(|s| s.peaks.len()).max().unwrap_or(0);
    assert!(f > 0 && n > 0, "need at least one formula and one peak");
    for s in spectra {
        assert_eq!(s.formulas.len(), f, "uniform F across spectra");
        assert_eq!(s.peaks.len(), n, "uniform N across spectra");
    }
    let mut top_counts = vec![0u32; b * f * 10];
    let mut kept = vec![0u32; b * n * 3];
    let mut meta = vec![0u32; b * 8];
    let mut spec = vec![0u32; b * 2];
    for (bi, s) in spectra.iter().enumerate() {
        for (fi, comp) in s.formulas.iter().enumerate() {
            for e in 0..10 {
                top_counts[(bi * f + fi) * 10 + e] = u32::from(comp[e]);
            }
        }
        for (pi, &mz) in s.peaks.iter().enumerate() {
            kept[(bi * n + pi) * 3] = (bi * n + pi) as u32;
            kept[(bi * n + pi) * 3 + 1] = mz;
        }
        meta[bi * 8] = s.peak_count;
        meta[bi * 8 + 3] = u32::from(s.adduct);
        meta[bi * 8 + 4] = s.ppm;
        spec[bi * 2] = s.uncertainty;
    }
    // Twin lanes.
    let mut want_ion = vec![0u32; b * f * n * j as usize * 12];
    let mut want_meta = vec![0u32; b * f * n * 4];
    for bi in 0..b {
        for fi in 0..f {
            for pi in 0..n {
                ion_assign_lane(
                    &top_counts,
                    &kept,
                    &meta,
                    &spec,
                    bi as u32,
                    fi as u32,
                    pi as u32,
                    f as u32,
                    n as u32,
                    j,
                    work_max,
                    &mut want_ion,
                    &mut want_meta,
                );
            }
        }
    }
    // Kernel on poisoned outputs.
    let device = dev();
    let top_t = upload_ids(&top_counts, vec![b, f, 10], &device);
    let kept_t = upload_ids(&kept, vec![b, n, 3], &device);
    let meta_t = upload_ids(&meta, vec![b, 8], &device);
    let spec_t = upload_ids(&spec, vec![b, 2], &device);
    let mut ion_t = upload_ids(
        &poison_ids(b * f * n * j as usize * 12, &device),
        vec![b, f, n, j as usize, 12],
        &device,
    );
    let mut im_t = upload_ids(&poison_ids(b * f * n * 4, &device), vec![b, f, n, 4], &device);
    ms2_ion::ion_assign(&top_t, &kept_t, &meta_t, &spec_t, &mut ion_t, &mut im_t, work_max)
        .unwrap();
    check_launches(&device).unwrap();
    let got_ion = ion_t.try_to_vec().unwrap();
    let got_meta = im_t.try_to_vec().unwrap();
    assert_ids(&got_ion, &want_ion, "ion rows");
    assert_ids(&got_meta, &want_meta, "ion_meta rows");
    (got_ion, got_meta)
}

/// Residual-bound numerator of a hypothesis composition (test-local).
fn bound_numer(counts: &Composition) -> u64 {
    let mut nda: u64 = 0;
    for e in 0..10 {
        nda += u64::from(counts[e]) * u64::from(ELEMENTS[e].residual_nda);
    }
    nda
}

/// Sweep peaks over a few fragments with accept/ambiguous/reject edge
/// offsets (test-local arithmetic from the contract formulas).
fn sweep_peaks(parent: &Composition, adduct: u16, u: u32, ppm: u32) -> Vec<u32> {
    const HEAVY: [usize; 9] = [0, 2, 3, 4, 5, 6, 7, 8, 9];
    let (h_a, z): (i64, i64) = if adduct == 1 { (1, 1) } else { (-1, -1) };
    let mass_of = |counts: &Composition| -> u64 {
        let mut m: u64 = 0;
        for e in 0..10 {
            m += u64::from(counts[e]) * u64::from(ELEMENTS[e].mass);
        }
        m
    };
    // Fragments: full parent heavy vector with shifts, half carbons, one carbon.
    let mut frags: Vec<(Composition, u32)> = Vec::new();
    let mut full = [0u16; 10];
    for &e in &HEAVY {
        full[e] = parent[e];
    }
    for s in [-1i64, 0, 1] {
        let h = i64::from(parent[HYDROGEN]) + h_a + s;
        if h >= 0 {
            frags.push((full, h as u32));
        }
    }
    if parent[0] >= 2 {
        let mut half = [0u16; 10];
        half[0] = parent[0] / 2;
        for &e in &HEAVY[1..] {
            half[e] = parent[e];
        }
        let h = i64::from(parent[HYDROGEN]) / 2 + h_a;
        if h >= 0 {
            frags.push((half, h as u32));
        }
    }
    let mut peaks = std::collections::BTreeSet::new();
    for (heavy, ion_h) in &frags {
        let mut m: u64 = 0;
        for &e in &HEAVY {
            m += u64::from(heavy[e]) * u64::from(ELEMENTS[e].mass);
        }
        m += u64::from(*ion_h) * u64::from(ELEMENTS[HYDROGEN].mass);
        let mz0 = m as i64 - z * 549;
        if mz0 <= 0 || mz0 > u64::from(u32::MAX) as i64 {
            continue;
        }
        let mut hyp = *heavy;
        hyp[1] = *ion_h as u16;
        let e0 = (bound_numer(&hyp) + u64::from(ELECTRON_RESIDUAL_NDA)).div_ceil(1000) + u64::from(u);
        let tol = mz0 as u64 * u64::from(ppm) / 10_000_000;
        let edges: Vec<i64> = vec![
            0,
            tol as i64 - e0 as i64,
            tol as i64 - e0 as i64 + 1,
            tol as i64 + e0 as i64,
            tol as i64 + e0 as i64 + 1,
        ];
        for d in edges {
            if mz0 + d > 0 && mz0 + d <= u64::from(u32::MAX) as i64 {
                peaks.insert((mz0 + d) as u32);
            }
            if d > 0 && mz0 - d > 0 {
                peaks.insert((mz0 - d) as u32);
            }
        }
    }
    // A peak above the parent mass (reject, complete).
    let pm = mass_of(parent);
    if pm + 5_000_000 <= u64::from(u32::MAX) {
        peaks.insert((pm + 5_000_000) as u32);
    }
    let mut out: Vec<u32> = peaks.into_iter().collect();
    if adduct == 2 {
        out.retain(|&mz| mz >= 549);
    }
    out
}

fn small_parents() -> Vec<Composition> {
    vec![
        [3, 8, 0, 1, 0, 0, 0, 0, 0, 0], // C3H8O
        [2, 4, 0, 2, 0, 0, 0, 0, 0, 0], // C2H4O2
        [6, 6, 0, 0, 0, 0, 0, 0, 0, 0], // C6H6
        [1, 3, 0, 0, 0, 0, 0, 1, 0, 0], // CH3Cl
    ]
}

#[test]
fn assign_small_parents_both_adducts() {
    for parent in &small_parents() {
        for adduct in [1u16, 2] {
            let (u, ppm) = (50u32, 100u32);
            let mut peaks = sweep_peaks(parent, adduct, u, ppm);
            assert!(!peaks.is_empty(), "sweep yields peaks");
            peaks.truncate(40);
            let n = peaks.len();
            let spectra = vec![AssignSpectrum {
                formulas: vec![*parent],
                adduct,
                ppm,
                uncertainty: u,
                peaks,
                peak_count: n as u32,
            }];
            let (_, meta) = run_assign_kernel(&spectra, 4, 4096);
            // The sweep hits every verdict class through the twin.
            let mut any_accept = false;
            let mut any_ambiguous = false;
            let mut any_reject_complete = false;
            for p in 0..n {
                let a = meta[p * 4];
                let amb = meta[p * 4 + 1];
                let st = meta[p * 4 + 3];
                if a > 0 {
                    any_accept = true;
                }
                if amb > 0 {
                    any_ambiguous = true;
                }
                if a == 0 && amb == 0 && st == 0 {
                    any_reject_complete = true;
                }
                assert_eq!(st, 0, "large limits leave every sweep peak complete");
            }
            assert!(any_accept, "sweep accepts for {parent:?} adduct {adduct}");
            assert!(any_ambiguous, "sweep is ambiguous somewhere for {parent:?}");
            assert!(any_reject_complete, "sweep rejects somewhere for {parent:?}");
        }
    }
}

#[test]
fn assign_heteroatom_fixture_and_b2_independence() {
    // Fixture molecule with N and S plus one with Cl (small work).
    let acet = graph_of("acetonitrile").composition(); // C2H3N
    let cyst = graph_of("cysteine").composition(); // C3H7NO2S
    let peaks_a = sweep_peaks(&acet, 1, 50, 100);
    let peaks_c = sweep_peaks(&cyst, 2, 75, 200);
    let na = peaks_a.len().min(24);
    let nc = peaks_c.len().min(24);
    let spectra = vec![
        AssignSpectrum {
            formulas: vec![acet],
            adduct: 1,
            ppm: 100,
            uncertainty: 50,
            peaks: peaks_a[..na].to_vec(),
            peak_count: na as u32,
        },
        AssignSpectrum {
            formulas: vec![cyst],
            adduct: 2,
            ppm: 200,
            uncertainty: 75,
            peaks: peaks_c[..nc].to_vec(),
            peak_count: nc as u32,
        },
    ];
    // Uniform N across spectra for one launch.
    let n = na.max(nc);
    let mut spectra = spectra;
    for s in &mut spectra {
        while s.peaks.len() < n {
            s.peaks.push(0);
        }
    }
    let (ion2, meta2) = run_assign_kernel(&spectra, 4, 4096);
    // Row independence: each spectrum alone produces the same words.
    for (bi, s) in spectra.iter().enumerate() {
        let solo = vec![AssignSpectrum {
            formulas: s.formulas.clone(),
            adduct: s.adduct,
            ppm: s.ppm,
            uncertainty: s.uncertainty,
            peaks: s.peaks.clone(),
            peak_count: s.peak_count,
        }];
        let (ion1, meta1) = run_assign_kernel(&solo, 4, 4096);
        let row = n * 4 * 12;
        assert_ids(&ion2[bi * row..(bi + 1) * row], &ion1, "B=2 row {bi} ion");
        let mrow = n * 4;
        assert_ids(&meta2[bi * mrow..(bi + 1) * mrow], &meta1, "B=2 row {bi} meta");
    }
}

#[test]
fn assign_truncation_exhaustion_and_unavailable() {
    // Caffeine at 1000 ppm accepts several hypotheses at the C8O-ion peak.
    let parent: Composition = [8, 10, 4, 2, 0, 0, 0, 0, 0, 0];
    let mz = 8 * 12_000_000 + 15_994_915 + 11 * 1_007_825 - 549;
    let full = AssignSpectrum {
        formulas: vec![parent],
        adduct: 1,
        ppm: 1000,
        uncertainty: 50,
        peaks: vec![mz],
        peak_count: 1,
    };
    let (ion_big, meta_big) = run_assign_kernel(std::slice::from_ref(&full), 8, 1_000_000);
    assert_eq!(meta_big[3], 0, "wide limits leave the peak complete");
    assert!(meta_big[0] >= 3, "peak accepts {}", meta_big[0]);
    let cut = AssignSpectrum {
        formulas: vec![parent],
        adduct: 1,
        ppm: 1000,
        uncertainty: 50,
        peaks: vec![mz],
        peak_count: 1,
    };
    let (ion_cut, meta_cut) = run_assign_kernel(std::slice::from_ref(&cut), 2, 1_000_000);
    assert_ne!(meta_cut[3] & ION_CAPACITY_EXCEEDED, 0, "J = 2 truncates");
    assert_eq!(meta_cut[0], meta_big[0], "accepted counts agree");
    assert_eq!(meta_cut[1], meta_big[1], "ambiguous counts agree");
    assert_ids(&ion_cut, &ion_big[..ion_cut.len()], "kept prefix in visit order");
    // Exhaustion under a tiny work budget.
    let c2o1: Composition = [2, 0, 0, 1, 0, 0, 0, 0, 0, 0];
    let exh = AssignSpectrum {
        formulas: vec![c2o1],
        adduct: 1,
        ppm: 100,
        uncertainty: 50,
        peaks: vec![12_000_000],
        peak_count: 1,
    };
    let (_, meta_exh) = run_assign_kernel(std::slice::from_ref(&exh), 4, 3);
    assert_ne!(meta_exh[3] & ION_SEARCH_EXHAUSTED, 0, "tiny budget exhausts");
    // Unavailable: wide window, sentinel, padding, unstarted slot, F = 3.
    let wide = AssignSpectrum {
        formulas: vec![parent, parent, [0; 10]],
        adduct: 1,
        ppm: 100,
        uncertainty: 2_000_000,
        peaks: vec![mz, 0, mz, mz],
        peak_count: 3,
        // slot 1 is padding (mz 0), slot 3 is past peak_count, formula slot
        // 2 is unstarted; the wide uncertainty covers every real peak.
    };
    let (ion_w, meta_w) = run_assign_kernel(std::slice::from_ref(&wide), 4, 4096);
    for fi in 0..3usize {
        for p in 0..4usize {
            let m = meta_w[(fi * 4 + p) * 4..(fi * 4 + p) * 4 + 4].to_vec();
            assert_eq!(m[3] & ION_UNAVAILABLE, ION_UNAVAILABLE, "f {fi} p {p} unavailable");
            assert_eq!(&m[..3], &[0, 0, 0], "f {fi} p {p} counts zero");
            let row = ion_w[(fi * 4 + p) * 4 * 12..(fi * 4 + p + 1) * 4 * 12].to_vec();
            assert!(row.iter().all(|&w| w == 0), "f {fi} p {p} ion zeros");
        }
    }
    // The sentinel behaves like the wide window.
    let sent = AssignSpectrum {
        formulas: vec![parent],
        adduct: 1,
        ppm: 100,
        uncertainty: u32::MAX,
        peaks: vec![mz],
        peak_count: 1,
    };
    let (_, meta_s) = run_assign_kernel(std::slice::from_ref(&sent), 4, 4096);
    assert_eq!(meta_s[3] & ION_UNAVAILABLE, ION_UNAVAILABLE);
}

#[test]
fn assign_iodine_and_high_counts_near_guards() {
    // Iodine (the heaviest element, 126_904_472 per atom): guarded heavy
    // products stay exact and the whole-vector hypothesis is accepted with
    // complete support.
    let parent: Composition = [2, 3, 0, 0, 0, 0, 0, 0, 0, 5]; // C2H3I5
    let mass: u64 = 2 * 12_000_000 + 5 * 126_904_472 + 4 * 1_007_825;
    let mz = (mass - 549) as u32;
    let spectra = vec![AssignSpectrum {
        formulas: vec![parent],
        adduct: 1,
        ppm: 100,
        uncertainty: 50,
        peaks: vec![mz],
        peak_count: 1,
    }];
    let (ion, meta) = run_assign_kernel(&spectra, 4, 65_536);
    assert!(meta[0] >= 1, "iodine parent accepts {}", meta[0]);
    assert_eq!(meta[3] & (ION_SEARCH_EXHAUSTED | ION_UNAVAILABLE), 0, "complete support");
    assert!(meta[2] >= 1, "keeps {}", meta[2]);
    assert_eq!(ion[0], 2, "kept carbon count");
    assert_eq!(ion[9], 5, "kept iodine count");
    // Counts near the u16 ceiling: the radix product is decided exactly by
    // division guards, and heavy-mass accumulation never overflows (every
    // kept hypothesis has an exact u32 mass, so no wrapped product was
    // stored). The C1H97 hypothesis is accepted.
    let heavy: Composition = [1023, 96, 0, 0, 0, 0, 0, 0, 0, 0];
    let spectra = vec![AssignSpectrum {
        formulas: vec![heavy],
        adduct: 1,
        ppm: 100,
        uncertainty: 50,
        peaks: vec![12_000_000 + 97 * 1_007_825 - 549],
        peak_count: 1,
    }];
    let (ion_h, meta_h) = run_assign_kernel(&spectra, 4, 4096);
    assert!(meta_h[0] >= 1, "C1H97 accepts, got {}", meta_h[0]);
    assert_eq!(meta_h[3] & (ION_SEARCH_EXHAUSTED | ION_UNAVAILABLE), 0, "guarded run is complete");
    for q in 0..meta_h[2] as usize {
        let w = q * 12;
        let mut m: u64 = 0;
        for e in 0..10 {
            m += u64::from(ion_h[w + e]) * u64::from(ELEMENTS[e].mass);
        }
        assert!(m <= u64::from(u32::MAX), "kept mass fits u32");
        assert_eq!(ion_h[w + 10], m as u32, "kept mass is exact (guarded)");
    }
    // The same counts exhaust a tiny budget through the exact product rule.
    let (_, meta_e) = run_assign_kernel(&spectra, 4, 3);
    assert_ne!(meta_e[3] & ION_SEARCH_EXHAUSTED, 0, "giant radix product exhausts");
}

#[test]
fn assign_hydrogen_window_equality_edge() {
    // half_p exactly one hydrogen mass is still searched (the scope
    // restriction is `half_p <= m_H`); one unit more is unavailable.
    // Parent C1: E_ion = ceil((0 + 3 * 33 + 421) / 1000) = 1.
    let parent: Composition = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let mz = 100_000_000u32;
    assert_eq!(mz / 100_000, 1000, "tol_p is 1000 at ppm 100");
    let u_edge = 1_007_825 - 1000 - 1;
    let spectra = vec![
        AssignSpectrum {
            formulas: vec![parent],
            adduct: 1,
            ppm: 100,
            uncertainty: u_edge,
            peaks: vec![mz],
            peak_count: 1,
        },
        AssignSpectrum {
            formulas: vec![parent],
            adduct: 1,
            ppm: 100,
            uncertainty: u_edge + 1,
            peaks: vec![mz],
            peak_count: 1,
        },
    ];
    let (ion, meta) = run_assign_kernel(&spectra, 4, 4096);
    assert_eq!(meta[3], 0, "equality edge is searched and complete");
    assert_eq!(meta[7] & ION_UNAVAILABLE, ION_UNAVAILABLE, "one unit wider is unavailable");
    assert_eq!(&meta[4..7], &[0, 0, 0], "unavailable counts are zero");
    assert!(ion[4 * 12..5 * 12].iter().all(|&w| w == 0), "unavailable row is zero");
}

#[test]
fn assign_padding_and_unstarted_guard_matching_windows() {
    // A padding peak and an unstarted formula slot each write zeros with
    // ION_UNAVAILABLE even though the window itself would match: the guard,
    // not the window, zeroes the row.
    let parent: Composition = [1, 4, 0, 0, 0, 0, 0, 0, 0, 0];
    let mz = 12_000_000 + 5 * 1_007_825 - 549;
    let spectra = vec![AssignSpectrum {
        formulas: vec![parent, [0; 10]],
        adduct: 1,
        ppm: 100,
        uncertainty: 50,
        peaks: vec![mz, mz],
        peak_count: 1,
    }];
    let (ion, meta) = run_assign_kernel(&spectra, 4, 4096);
    // (f0, p0) is searched: the whole-vector hypothesis is accepted.
    assert!(meta[0] >= 1, "matching lane accepts {}", meta[0]);
    assert_eq!(meta[3] & (ION_SEARCH_EXHAUSTED | ION_UNAVAILABLE), 0, "complete");
    // (f0, p1) is a padding peak, (f1, p0) an unstarted slot, (f1, p1) both.
    for lane in [1usize, 2, 3] {
        let m = &meta[lane * 4..lane * 4 + 4];
        assert_eq!(m[3] & ION_UNAVAILABLE, ION_UNAVAILABLE, "lane {lane} unavailable");
        assert_eq!(&m[..3], &[0, 0, 0], "lane {lane} counts zero");
        let row = &ion[lane * 4 * 12..(lane + 1) * 4 * 12];
        assert!(row.iter().all(|&w| w == 0), "lane {lane} ion zeros");
    }
}

#[test]
fn assign_fixture_spectra_from_recipe_masses() {
    // Synthetic spectra at recipe ion masses, kept peaks from twin::peak_select.
    let cases = [("2-butanol", 1u16), ("acetonitrile", 2u16)];
    let mut spectra: Vec<AssignSpectrum> = Vec::new();
    for (name, adduct) in cases {
        let g = graph_of(name);
        let parent = g.composition();
        let embeddings = enumerate_embeddings(&g, &RecipeLimits::V0);
        assert!(!embeddings.is_empty(), "{name} has recipe embeddings");
        let (h_a, z): (i64, i64) = if adduct == 1 { (1, 1) } else { (-1, -1) };
        let mut mz: Vec<u32> = Vec::new();
        for emb in embeddings.iter().take(12) {
            let frag = g.induced(&emb.atoms).expect("embedding induces");
            let comp = frag.composition();
            let mut m: u64 = 0;
            for e in 0..10 {
                m += u64::from(comp[e]) * u64::from(ELEMENTS[e].mass);
            }
            let ion_h = u64::from(comp[HYDROGEN]) as i64 + h_a;
            if ion_h < 0 {
                continue;
            }
            let ion_mass = m as i64 + ion_h * u64::from(ELEMENTS[HYDROGEN].mass) as i64;
            let mz_i = ion_mass - z * 549;
            if mz_i > 0 && mz_i <= u64::from(u32::MAX) as i64 {
                mz.push(mz_i as u32);
            }
        }
        mz.sort_unstable();
        mz.dedup();
        mz.truncate(12);
        let n_raw = mz.len();
        let intensity: Vec<f32> = (0..n_raw).map(|i| 1.0 - i as f32 * 0.01).collect();
        let mut parent_mass: u64 = 0;
        for e in 0..10 {
            parent_mass += u64::from(parent[e]) * u64::from(ELEMENTS[e].mass);
        }
        let meta = vec![
            n_raw as u32,
            (parent_mass + 2_000_000).min(u64::from(u32::MAX)) as u32,
            0,
            u32::from(adduct),
            100,
            0,
            0,
            0,
        ];
        let n_keep = 8usize;
        let sel = peak_select(&mz, &intensity, &meta, 1, n_raw, n_keep, 0);
        let mut peaks = vec![0u32; n_keep];
        for (p, slot) in peaks.iter_mut().enumerate() {
            *slot = sel.kept[p * 3 + 1];
        }
        assert!(peaks.iter().any(|&m| m != 0), "{name} keeps real peaks");
        let mut sub = parent;
        sub[0] = sub[0].saturating_sub(1);
        spectra.push(AssignSpectrum {
            formulas: vec![parent, sub],
            adduct,
            ppm: 100,
            uncertainty: 50,
            peaks,
            peak_count: n_raw as u32,
        });
    }
    let n = spectra.iter().map(|s| s.peaks.len()).max().unwrap();
    for s in &mut spectra {
        while s.peaks.len() < n {
            s.peaks.push(0);
        }
    }
    run_assign_kernel(&spectra, 4, 4096);
}

#[test]
fn assign_rejects_bad_shapes() {
    let device = dev();
    let top_t = upload_ids(&[0u32; 10], vec![1, 1, 10], &device);
    let kept_t = upload_ids(&[0u32; 6], vec![1, 2, 3], &device);
    let meta_t = upload_ids(&[0u32; 8], vec![1, 8], &device);
    let spec_t = upload_ids(&[0u32; 2], vec![1, 2], &device);
    let mut ion_t = upload_ids(&[0u32; 96], vec![1, 1, 2, 4, 12], &device);
    let mut im_t = upload_ids(&[0u32; 8], vec![1, 1, 2, 4], &device);
    // Wrong kept width.
    let bad_kept = upload_ids(&[0u32; 4], vec![1, 2, 2], &device);
    assert!(ms2_ion::ion_assign(&top_t, &bad_kept, &meta_t, &spec_t, &mut ion_t, &mut im_t, 8).is_err());
    // Wrong ion rank.
    let mut bad_ion = upload_ids(&[0u32; 96], vec![96], &device);
    assert!(ms2_ion::ion_assign(&top_t, &kept_t, &meta_t, &spec_t, &mut bad_ion, &mut im_t, 8).is_err());
    // N mismatch between kept and ion.
    let mut bad_n = upload_ids(&[0u32; 48], vec![1, 1, 1, 4, 12], &device);
    assert!(ms2_ion::ion_assign(&top_t, &kept_t, &meta_t, &spec_t, &mut bad_n, &mut im_t, 8).is_err());
}

// ---------------------------------------------------------------------------
// Label mask: twin lanes versus the kernel.
// ---------------------------------------------------------------------------

/// Pack hand-built assignments and labels into the kernel buffers.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn pack_label_case(
    hyps: &[Vec<(Composition, u32, i32)>],
    statuses: &[u32],
    labels: &[(u32, Composition, bool)],
    kept_raw: &[u32],
    j: usize,
) -> (Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>) {
    let n = kept_raw.len();
    let l = labels.len();
    let mut ion = vec![0u32; n * j * 12];
    let mut ion_meta = vec![0u32; n * 4];
    for (p, h) in hyps.iter().enumerate() {
        let take = h.len().min(j);
        for (q, (counts, mass, res)) in h.iter().take(take).enumerate() {
            let w = (p * j + q) * 12;
            for e in 0..10 {
                ion[w + e] = u32::from(counts[e]);
            }
            ion[w + 10] = *mass;
            ion[w + 11] = (*res as u32).wrapping_add(LANE_BIAS);
        }
        let m = p * 4;
        ion_meta[m] = take as u32;
        ion_meta[m + 1] = 0;
        ion_meta[m + 2] = take as u32;
        ion_meta[m + 3] = statuses[p];
    }
    let mut lab = vec![0u32; l * 12];
    for (i, (raw, counts, valid)) in labels.iter().enumerate() {
        lab[i * 12] = *raw;
        for e in 0..10 {
            lab[i * 12 + 1 + e] = u32::from(counts[e]);
        }
        lab[i * 12 + 11] = u32::from(*valid);
    }
    let mut kept = vec![0u32; n * 3];
    for (p, &raw) in kept_raw.iter().enumerate() {
        kept[p * 3] = raw;
        kept[p * 3 + 1] = 1000 + p as u32;
    }
    (ion, ion_meta, lab, kept)
}

#[allow(clippy::too_many_arguments)]
fn run_label_mask_kernel(
    ion: &[u32],
    ion_meta: &[u32],
    lab: &[u32],
    kept: &[u32],
    b: usize,
    f: usize,
    n: usize,
    j: usize,
    l: usize,
    f_slot: u32,
) -> (Vec<f32>, Vec<u32>) {
    // Twin lanes.
    let mut want_mask = vec![0.0f32; b * n * (j + 1)];
    let mut want_state = vec![0u32; b * n];
    for bi in 0..b {
        for pi in 0..n {
            label_mask_lane(
                lab,
                ion,
                ion_meta,
                kept,
                bi as u32,
                pi as u32,
                f_slot,
                f as u32,
                n as u32,
                j as u32,
                l as u32,
                &mut want_mask,
                &mut want_state,
            );
        }
    }
    // Kernel on poisoned outputs.
    let device = dev();
    let lab_t = upload_ids(lab, vec![b, l, 12], &device);
    let ion_t = upload_ids(ion, vec![b, f, n, j, 12], &device);
    let im_t = upload_ids(ion_meta, vec![b, f, n, 4], &device);
    let kept_t = upload_ids(kept, vec![b, n, 3], &device);
    let mut mask_t = upload_f(&poison_f(b * n * (j + 1)), vec![b, n, j + 1], &device);
    let mut state_t = upload_ids(&poison_ids(b * n, &device), vec![b, n], &device);
    ms2_ion::ion_label_mask(&lab_t, &ion_t, &im_t, &kept_t, &mut mask_t, &mut state_t, f_slot)
        .unwrap();
    check_launches(&device).unwrap();
    let got_mask = mask_t.to_f32();
    let got_state = state_t.try_to_vec().unwrap();
    // The harness compares the complete outputs with the twin, not just
    // selected words: a lane that skips hypothesis indices above zero (or a
    // spectrum) fails here.
    assert_eq!(got_mask, want_mask, "complete label_mask matches the twin");
    assert_ids(&got_state, &want_state, "complete label_state matches the twin");
    (got_mask, got_state)
}

#[test]
fn label_mask_states_partial_and_invalid_rows() {
    let comp_a: Composition = [1, 4, 0, 0, 0, 0, 0, 0, 0, 0];
    let comp_b: Composition = [1, 2, 0, 0, 0, 0, 0, 0, 0, 0];
    let comp_c: Composition = [2, 6, 0, 0, 0, 0, 0, 0, 0, 0];
    let comp_d: Composition = [2, 4, 0, 0, 0, 0, 0, 0, 0, 0];
    let j = 2usize;
    // Peaks: partial overlap (a kept, d not), labels-but-none-kept,
    // padding, and a peak with no label at all.
    let hyps = vec![
        vec![(comp_a, 1u32, 0i32)],
        vec![(comp_b, 2u32, 0i32)],
        vec![],
        vec![(comp_a, 3u32, 0i32)],
    ];
    let statuses = vec![0u32; 4];
    let labels = vec![
        (3u32, comp_a, true),
        (3u32, comp_d, true),
        (5u32, comp_c, true),
        // Invalid overflow row: matches peak 11 but must never match.
        (11u32, comp_a, false),
        // Invalid row matching a kept hypothesis: must not set state 1.
        (5u32, comp_b, false),
    ];
    let kept_raw = vec![3u32, 5, u32::MAX, 11];
    let (ion, ion_meta, lab, kept) = pack_label_case(&hyps, &statuses, &labels, &kept_raw, j);
    let (mask, state) = run_label_mask_kernel(&ion, &ion_meta, &lab, &kept, 1, 1, 4, j, 5, 0);
    // Peak 0 is true partial (state 3): label `a` is kept while label `d` of
    // the same peak is not among the kept hypotheses.
    assert_eq!(state, vec![3, 2, 0, 0]);
    assert_eq!(&mask[0..3], &[1.0, 0.0, 0.0]);
    assert_eq!(&mask[3..6], &[0.0, 0.0, 1.0]);
    assert_eq!(&mask[6..9], &[0.0, 0.0, 1.0]);
    assert_eq!(&mask[9..12], &[0.0, 0.0, 1.0]);
}

#[test]
fn label_mask_second_formula_slot() {
    // F = 2 with distinct hypotheses per slot; the lane reads f_slot.
    let comp_a: Composition = [1, 4, 0, 0, 0, 0, 0, 0, 0, 0];
    let comp_b: Composition = [1, 2, 0, 0, 0, 0, 0, 0, 0, 0];
    let j = 2usize;
    let n = 1usize;
    let mut ion = vec![0u32; 2 * n * j * 12];
    let mut ion_meta = vec![0u32; 2 * n * 4];
    for (slot, comp) in [comp_a, comp_b].iter().enumerate() {
        let w = (slot * n) * j * 12;
        for e in 0..10 {
            ion[w + e] = u32::from(comp[e]);
        }
        ion[w + 11] = LANE_BIAS;
        let m = (slot * n) * 4;
        ion_meta[m] = 1;
        ion_meta[m + 2] = 1;
    }
    let mut lab = vec![0u32; 2 * 12];
    lab[0] = 7;
    for e in 0..10 {
        lab[1 + e] = u32::from(comp_b[e]);
    }
    lab[11] = 1;
    lab[12] = 7;
    for e in 0..10 {
        lab[12 + 1 + e] = u32::from(comp_a[e]);
    }
    lab[23] = 1;
    let kept = vec![7u32, 5000, 0];
    // Slot 0 holds comp_a: label row 1 matches, row 0 (comp_b) is a dropped
    // label of the same peak, so the state is true partial (3).
    let (mask0, state0) = run_label_mask_kernel(&ion, &ion_meta, &lab, &kept, 1, 2, n, j, 2, 0);
    assert_eq!(state0, vec![3]);
    assert_eq!(mask0, vec![1.0, 0.0, 0.0]);
    // Slot 1 holds comp_b: label row 0 matches, row 1 dropped: partial too.
    let (mask1, state1) = run_label_mask_kernel(&ion, &ion_meta, &lab, &kept, 1, 2, n, j, 2, 1);
    assert_eq!(state1, vec![3]);
    assert_eq!(mask1, vec![1.0, 0.0, 0.0]);
}

#[test]
fn label_mask_positive_match_at_hypothesis_one() {
    // The matching hypothesis sits at kept index 1, not 0: a lane that only
    // reads hypothesis 0 reports state 2 with the unassigned one-hot, while
    // the twin (and the kernel) report state 1 with the mask on class 1.
    let comp_a: Composition = [1, 4, 0, 0, 0, 0, 0, 0, 0, 0];
    let comp_other: Composition = [2, 6, 0, 0, 0, 0, 0, 0, 0, 0];
    let j = 2usize;
    let hyps = vec![vec![(comp_other, 7u32, 0i32), (comp_a, 9u32, 1i32)]];
    let labels = vec![(0u32, comp_a, true)];
    let kept_raw = vec![0u32];
    let (ion, ion_meta, lab, kept) = pack_label_case(&hyps, &[0], &labels, &kept_raw, j);
    let (mask, state) = run_label_mask_kernel(&ion, &ion_meta, &lab, &kept, 1, 1, 1, j, 1, 0);
    assert_eq!(state, vec![1]);
    assert_eq!(mask, vec![0.0, 1.0, 0.0]);
}

#[test]
fn label_mask_multispectrum_distinct_labels() {
    // B = 2 spectra with distinct labels: each spectrum's lane reads only
    // its own label rows and hypothesis rows, so cross-spectrum hypotheses
    // never match.
    let comp_a: Composition = [1, 4, 0, 0, 0, 0, 0, 0, 0, 0];
    let comp_b: Composition = [1, 2, 0, 0, 0, 0, 0, 0, 0, 0];
    let j = 1usize;
    let (ion0, meta0, lab0, kept0) =
        pack_label_case(&[vec![(comp_a, 5u32, 0i32)]], &[0], &[(0u32, comp_a, true)], &[0u32], j);
    let (ion1, meta1, lab1, kept1) =
        pack_label_case(&[vec![(comp_b, 6u32, 0i32)]], &[0], &[(1u32, comp_b, true)], &[1u32], j);
    let ion = [ion0, ion1].concat();
    let ion_meta = [meta0, meta1].concat();
    let lab = [lab0, lab1].concat();
    let kept = [kept0, kept1].concat();
    let (mask, state) = run_label_mask_kernel(&ion, &ion_meta, &lab, &kept, 2, 1, 1, j, 1, 0);
    assert_eq!(state, vec![1, 1]);
    assert_eq!(mask, vec![1.0, 0.0, 1.0, 0.0]);
}

#[test]
fn label_mask_rejects_bad_shapes() {
    let device = dev();
    let lab_t = upload_ids(&[0u32; 12], vec![1, 1, 12], &device);
    let ion_t = upload_ids(&[0u32; 24], vec![1, 1, 1, 2, 12], &device);
    let im_t = upload_ids(&[0u32; 4], vec![1, 1, 1, 4], &device);
    let kept_t = upload_ids(&[0u32; 3], vec![1, 1, 3], &device);
    let mut mask_t = upload_f(&[0.0; 3], vec![1, 1, 3], &device);
    let mut state_t = upload_ids(&[0u32; 1], vec![1, 1], &device);
    // f_slot beyond F.
    assert!(ms2_ion::ion_label_mask(&lab_t, &ion_t, &im_t, &kept_t, &mut mask_t, &mut state_t, 1).is_err());
    // Mask width must be J + 1.
    let mut bad_mask = upload_f(&[0.0; 2], vec![1, 1, 2], &device);
    assert!(ms2_ion::ion_label_mask(&lab_t, &ion_t, &im_t, &kept_t, &mut bad_mask, &mut state_t, 0).is_err());
}

// ---------------------------------------------------------------------------
// Evidence: twin lanes versus the kernel.
// ---------------------------------------------------------------------------

/// Pack a token trace with open valence words into an evidence actions row.
fn pack_actions(trace: &[(u8, u8, u8, u8)], open: &[u32], status: u32, steps: usize, atoms: usize) -> Vec<u32> {
    let stride = steps * 4 + atoms + 4;
    let mut row = vec![0u32; stride];
    for (s, &(kind, ty, bond, ptr)) in trace.iter().enumerate() {
        assert!(s < steps, "trace fits steps");
        row[s * 4] = u32::from(kind);
        row[s * 4 + 1] = u32::from(ty);
        row[s * 4 + 2] = u32::from(bond);
        row[s * 4 + 3] = u32::from(ptr);
    }
    for (a, &o) in open.iter().enumerate() {
        assert!(a < atoms, "open fits atoms");
        row[steps * 4 + a] = o;
    }
    row[steps * 4 + atoms] = trace.len() as u32;
    row[steps * 4 + atoms + 1] = status;
    row
}

/// Canonical trace tokens of a molecule with replayed open valence.
type TraceRows = (Vec<(u8, u8, u8, u8)>, Vec<u32>);

fn trace_with_open(name: &str, atoms: &[usize], steps: usize, cap: usize) -> TraceRows {
    let g = graph_of(name);
    let sub = if atoms.len() == g.atoms().len() {
        g
    } else {
        g.induced(atoms).expect("subgraph induces")
    };
    let lim = Limits::new(cap, 4).expect("caps fit");
    let canon = canonical_trace(&sub, lim, CANONICAL_WORK_LIMIT)
        .expect("fixture graph canonicalizes")
        .trace;
    assert!(canon.len() <= steps, "{name} trace fits steps");
    let state = replay(
        &canon,
        Limits::new(cap, 4).expect("caps fit"),
        None,
    )
    .expect("canonical trace replays");
    let tokens = canon
        .iter()
        .map(|t| (t.kind, t.atom_type, t.bond, t.pointer))
        .collect();
    let open: Vec<u32> = state.residual_valence().iter().map(|&o| u32::from(o)).collect();
    (tokens, open)
}

/// Run the twin [`evidence_lane`] per trajectory and the
/// [`ms2_ion::ion_evidence`] kernel on poisoned outputs; compare every word.
#[allow(clippy::too_many_arguments)]
fn run_evidence_kernel(
    actions: &[u32],
    traj_slot: &[u32],
    ion: &[u32],
    ion_meta: &[u32],
    rows: usize,
    steps: u32,
    atoms: u32,
    k: usize,
    f: usize,
    n: usize,
    j: usize,
) -> Vec<u32> {
    let b = rows / k;
    assert_eq!(b * k, rows, "rows split into spectra");
    let mut want = vec![0u32; rows * EVIDENCE_ROW_WORDS];
    for r in 0..rows {
        evidence_lane(
            actions,
            traj_slot,
            ion,
            ion_meta,
            r as u32,
            steps,
            atoms,
            k as u32,
            f as u32,
            n as u32,
            j as u32,
            &mut want,
        );
    }
    let device = dev();
    let stride = steps as usize * 4 + atoms as usize + 4;
    let actions_t = upload_ids(actions, vec![rows, stride], &device);
    let slot_t = upload_ids(traj_slot, vec![rows, 2], &device);
    let ion_t = upload_ids(ion, vec![b, f, n, j, 12], &device);
    let im_t = upload_ids(ion_meta, vec![b, f, n, 4], &device);
    let mut ev_t = upload_ids(
        &poison_ids(rows * EVIDENCE_ROW_WORDS, &device),
        vec![rows, EVIDENCE_ROW_WORDS],
        &device,
    );
    ms2_ion::ion_evidence(
        &actions_t,
        &slot_t,
        &ion_t,
        &im_t,
        &mut ev_t,
        steps,
        atoms,
        k as u32,
    )
    .unwrap();
    check_launches(&device).unwrap();
    let got = ev_t.try_to_vec().unwrap();
    assert_ids(&got, &want, "evidence rows");
    got
}

fn unbias(ob: u32) -> i32 {
    ob.wrapping_sub(LANE_BIAS) as i32
}

/// Assignment rows for the evidence tests: the twin lane over crafted peaks.
fn evidence_ion_buffers(
    parent: &Composition,
    adduct: u16,
    peaks: &[u32],
    peak_count: u32,
    j: u32,
    work_max: u32,
) -> (Vec<u32>, Vec<u32>) {
    let n = peaks.len();
    let mut top_counts = vec![0u32; 10];
    for e in 0..10 {
        top_counts[e] = u32::from(parent[e]);
    }
    let mut kept = vec![0u32; n * 3];
    for (p, &mz) in peaks.iter().enumerate() {
        kept[p * 3] = p as u32;
        kept[p * 3 + 1] = mz;
    }
    let meta = vec![peak_count, 0, 0, u32::from(adduct), 100, 0, 0, 0];
    let spec = vec![50u32, 0];
    let mut ion = vec![0u32; n * j as usize * 12];
    let mut ion_meta = vec![0u32; n * 4];
    for p in 0..n {
        ion_assign_lane(
            &top_counts,
            &kept,
            &meta,
            &spec,
            0,
            0,
            p as u32,
            1,
            n as u32,
            j,
            work_max,
            &mut ion,
            &mut ion_meta,
        );
    }
    (ion, ion_meta)
}

#[test]
fn evidence_whole_molecule_only_zero_shift() {
    // Whole 2-butanol candidate: no open valence, so only s = 0 is
    // mass-consistent for every boundary count.
    let g = graph_of("2-butanol");
    let parent = g.composition();
    let hg_parent = parent[HYDROGEN];
    let mh = ELEMENTS[HYDROGEN].mass;
    // Peaks at the whole-molecule heavy vector with shifts s in -1..=2
    // (h stays within parent[H] + 3, the hypothesis ceiling).
    let mut heavy_mass: u64 = 0;
    for e in [0usize, 2, 3] {
        heavy_mass += u64::from(parent[e]) * u64::from(ELEMENTS[e].mass);
    }
    let mut peaks = Vec::new();
    for s in [-1i64, 0, 0, 1, 2, 2] {
        let h = hg_parent as i64 + 1 + s;
        assert!(h >= 0, "ion hydrogen stays non-negative");
        let ion_mass = heavy_mass as i64 + h * mh as i64;
        peaks.push((ion_mass - 549) as u32);
    }
    let n = peaks.len();
    let j = 4u32;
    let (ion, ion_meta) = evidence_ion_buffers(&parent, 1, &peaks, n as u32, j, 65_536);
    // Only mass-consistent shifts are evidence: with zero open valence the
    // s = 0 peaks qualify and the s = ±1, ±2 peaks contribute nothing, so
    // the count is 2 and the records are the two qualifying peaks in order.
    let steps = 24usize;
    let atoms = 8usize;
    let all: Vec<usize> = (0..g.atoms().len()).collect();
    let (tokens, open) = trace_with_open("2-butanol", &all, steps, 16);
    assert!(open.iter().all(|&o| o == 0), "whole molecule has no open valence");
    let actions = pack_actions(&tokens, &open, 1, steps, atoms);
    let got = run_evidence_kernel(&actions, &[0, 1], &ion, &ion_meta, 1, steps as u32, atoms as u32, 1, 1, n, j as usize);
    assert_eq!(got[0] & 127, 2, "universal match present");
    assert_eq!(got[0] & EVIDENCE_SUPPORT_INCOMPLETE, 0, "support complete");
    assert_eq!(got[1], 2, "count is qualifying matches only");
    for (q, (peak, s)) in [(1u32, 0i32), (2, 0)].iter().enumerate() {
        assert_eq!(got[2 + q * 4], *peak, "record {q} is peak {peak}");
        assert_eq!(unbias(got[2 + q * 4 + 2]), *s, "record {q} shift");
    }
    assert_eq!(&got[10..18], &[0u32; 8][..], "record area past the count is zero");
    assert_eq!(EVIDENCE_SLOTS, 4, "records fill all E slots");
}

#[test]
fn evidence_fragment_open_valence_and_incomplete() {
    // Single carbon atom with open valence 3: c_lo = 1, c_hi = 3.
    let steps = 4usize;
    let atoms = 2usize;
    let trace = vec![(START, 0, 0, 0), (ADD_ATOM, 4, 0, 0), (STOP, 0, 0, 0)];
    let actions = pack_actions(&trace, &[3, 0], 1, steps, atoms);
    // Peaks with s = 0 (universal), s = +2 (consistent only), s = +3 (out).
    // Carbon ion masses with H(g) = 3, h_a = 1: ion H = 4 + s.
    let mh = ELEMENTS[HYDROGEN].mass;
    let mk = |h: u32| 12_000_000 + h * mh - 549;
    let peaks = vec![mk(4), mk(6), mk(7)];
    let j = 4u32;
    // The assignment parent carries one extra hydrogen so h = 7 stays
    // within the hypothesis ceiling; the candidate H(g) = 3 comes from
    // the trace.
    let parent: Composition = [1, 4, 0, 0, 0, 0, 0, 0, 0, 0];
    let (ion, ion_meta) = evidence_ion_buffers(&parent, 1, &peaks, 3, j, 65_536);
    let got = run_evidence_kernel(&actions, &[0, 1], &ion, &ion_meta, 1, steps as u32, atoms as u32, 1, 1, 3, j as usize);
    assert_eq!(got[0], 2, "best status is universal");
    // Only s = 0 (universal) and s = +2 (consistent only) qualify; the
    // s = +3 peak contributes no count and no record.
    assert_eq!(got[1], 2, "unsupported shifts do not count");
    assert_eq!(unbias(got[2 + 2]), 0);
    assert_eq!(unbias(got[2 + 4 + 2]), 2);
    assert_eq!(&got[10..18], &[0u32; 8][..], "record area past the count is zero");
    // Exhausted assignment support sets bit 7 next to the status.
    let (ion_x, meta_x) = evidence_ion_buffers(&parent, 1, &peaks, 3, j, 0);
    assert_ne!(meta_x[3] & ION_SEARCH_EXHAUSTED, 0, "zero budget exhausts");
    let got_x = run_evidence_kernel(&actions, &[0, 1], &ion_x, &meta_x, 1, steps as u32, atoms as u32, 1, 1, 3, j as usize);
    assert_ne!(got_x[0] & EVIDENCE_SUPPORT_INCOMPLETE, 0, "incomplete support flagged");
}

#[test]
fn evidence_unfinished_trajectory_writes_zeros() {
    let steps = 4usize;
    let atoms = 2usize;
    let trace = vec![(START, 0, 0, 0), (ADD_ATOM, 4, 0, 0), (STOP, 0, 0, 0)];
    // Same finished trace twice: once unfinished (status 0), once with an
    // invalid formula slot.
    let mut actions = pack_actions(&trace, &[3, 0], 0, steps, atoms);
    actions.extend(pack_actions(&trace, &[3, 0], 1, steps, atoms));
    let mh = ELEMENTS[HYDROGEN].mass;
    let parent: Composition = [1, 3, 0, 0, 0, 0, 0, 0, 0, 0];
    let (ion, ion_meta) = evidence_ion_buffers(&parent, 1, &[12_000_000 + 4 * mh - 549], 1, 4, 65_536);
    let got = run_evidence_kernel(
        &actions,
        &[0, 1, u32::MAX, 1],
        &ion,
        &ion_meta,
        2,
        steps as u32,
        atoms as u32,
        2,
        1,
        1,
        4,
    );
    assert_eq!(&got[..EVIDENCE_ROW_WORDS], &vec![0u32; EVIDENCE_ROW_WORDS], "unfinished writes zeros");
    assert_eq!(
        &got[EVIDENCE_ROW_WORDS..],
        &vec![0u32; EVIDENCE_ROW_WORDS],
        "invalid slot writes zeros"
    );
}

#[test]
fn evidence_butanol_unsupported_shifts_ignored_on_device() {
    // Review finding 2 through the kernel: a whole 2-butanol candidate
    // (zero open valence, positive adduct) with four peaks at s = +1
    // followed by one at s = 0. Only the last peak is candidate evidence.
    let g = graph_of("2-butanol");
    let parent = g.composition();
    let hg = u32::from(parent[HYDROGEN]);
    let mh = ELEMENTS[HYDROGEN].mass;
    let mut heavy_mass: u64 = 0;
    for e in [0usize, 2, 3] {
        heavy_mass += u64::from(parent[e]) * u64::from(ELEMENTS[e].mass);
    }
    let mk = |s: i64| (heavy_mass as i64 + (hg as i64 + 1 + s) * mh as i64 - 549) as u32;
    let peaks = vec![mk(1), mk(1), mk(1), mk(1), mk(0)];
    let n = peaks.len();
    let j = 4u32;
    let (ion, ion_meta) = evidence_ion_buffers(&parent, 1, &peaks, n as u32, j, 65_536);
    let steps = 24usize;
    let atoms = 8usize;
    let all: Vec<usize> = (0..g.atoms().len()).collect();
    let (tokens, open) = trace_with_open("2-butanol", &all, steps, 16);
    let actions = pack_actions(&tokens, &open, 1, steps, atoms);
    let got = run_evidence_kernel(&actions, &[0, 1], &ion, &ion_meta, 1, steps as u32, atoms as u32, 1, 1, n, j as usize);
    assert_eq!(got[0] & 127, 2, "universal match present");
    assert_eq!(got[0] & EVIDENCE_SUPPORT_INCOMPLETE, 0, "support complete");
    assert_eq!(got[1], 1, "only the s = 0 peak counts");
    assert_eq!(got[2], 4, "record is peak 4");
    assert_eq!(got[3], 0, "record is hypothesis 0");
    assert_eq!(unbias(got[4]), 0, "record shift is 0");
    assert_eq!(&got[6..18], &[0u32; 12][..], "unsupported shifts leave no records");
}

#[test]
fn evidence_routes_across_spectra_slots_and_adduct_two() {
    // B = 2 spectra, F = 2 slots, N = 1, J = 2: (b0, slot0) under adduct 1
    // and (b1, slot1) under adduct 2 each match their own spectrum's peak
    // at s = 0; the crossed slots match nothing. Each trajectory reads only
    // its own (b, slot) rows.
    let steps = 4usize;
    let atoms = 2usize;
    let trace = vec![(START, 0, 0, 0), (ADD_ATOM, 4, 0, 0), (STOP, 0, 0, 0)];
    // Type 4 is carbon with 3 parent hydrogens; open valence 3 gives
    // c_lo = 1, c_hi = 3, so s = 0 is universal.
    let row = pack_actions(&trace, &[3, 0], 1, steps, atoms);
    let actions = [row.clone(), row.clone(), row.clone(), row.clone()].concat();
    let mh = ELEMENTS[HYDROGEN].mass;
    // Adduct 1: ion H = 3 + 1 + 0 = 4; adduct 2: ion H = 3 - 1 + 0 = 2.
    let mz0 = 12_000_000 + 4 * mh - 549;
    let mz1 = 12_000_000 + 2 * mh + 549;
    let j = 2usize;
    let parent0: Composition = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0];
    let parent1: Composition = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let (ion00, meta00) = evidence_ion_buffers(&parent0, 1, &[mz0], 1, j as u32, 65_536);
    let (ion11, meta11) = evidence_ion_buffers(&parent1, 2, &[mz1], 1, j as u32, 65_536);
    assert_eq!(meta00[2], 1, "spectrum 0 keeps one hypothesis");
    assert_eq!(meta11[2], 1, "spectrum 1 keeps one hypothesis");
    let zero_ion = vec![0u32; j * 12];
    let zero_meta = vec![0u32; 4];
    // Layout [B, F, N, J, 12] and [B, F, N, 4] with (b, f) rows in order.
    let ion = [ion00, zero_ion.clone(), zero_ion.clone(), ion11].concat();
    let ion_meta = [meta00, zero_meta.clone(), zero_meta.clone(), meta11].concat();
    // Trajectories: (b0, slot0, adduct1), (b0, slot1, adduct1),
    // (b1, slot0, adduct2), (b1, slot1, adduct2).
    let traj_slot = vec![0, 1, 1, 1, 0, 2, 1, 2];
    let got = run_evidence_kernel(
        &actions,
        &traj_slot,
        &ion,
        &ion_meta,
        4,
        steps as u32,
        atoms as u32,
        2,
        2,
        1,
        j,
    );
    // Matched rows carry the exact residual 0 hypothesis at peak 0.
    assert_eq!(&got[..6], &[2, 1, 0, 0, LANE_BIAS, LANE_BIAS][..], "b0 slot0 matches");
    assert_eq!(&got[6..18], &[0u32; 12][..], "b0 slot0 has no further records");
    // Crossed slots match nothing and write zeroed rows.
    assert_eq!(&got[18..36], &[0u32; 18][..], "b0 slot1 is zero");
    assert_eq!(&got[36..54], &[0u32; 18][..], "b1 slot0 is zero");
    assert_eq!(&got[54..60], &[2, 1, 0, 0, LANE_BIAS, LANE_BIAS][..], "b1 slot1 matches");
    assert_eq!(&got[60..72], &[0u32; 12][..], "b1 slot1 has no further records");
}

#[test]
fn evidence_status_form_matches_reference_rules() {
    // Cross-check the packed `evidence_status` wrapper against the lane on
    // equivalent inputs (uses the host wrapper, no device).
    let ch: Composition = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let mk = |h: u16, r: i32| {
        let mut c = ch;
        c[1] = h;
        (c, 0u32, r)
    };
    let wrap = |v: &[(Composition, u32, i32)]| IonAssignment {
        accepted: v.len() as u32,
        ambiguous: 0,
        kept: v
            .iter()
            .map(|(c, m, r)| IonHypothesis {
                counts: *c,
                mass: *m,
                residual: *r,
            })
            .collect(),
        status: 0,
    };
    let hyps = vec![wrap(&[mk(2, 5)]), wrap(&[mk(4, -3)]), wrap(&[mk(5, 7)]), wrap(&[])];
    let sum = evidence_status(&ch, 1, &[3], 1, &hyps, 4);
    // Only the s = 0 and s = +2 peaks qualify; the s = +3 peak contributes
    // no count and no record.
    assert_eq!(sum.count, 2);
    assert_eq!(sum.status, 2);
    // Same inputs through the lane: identical rows.
    let steps = 4usize;
    let atoms = 2usize;
    let trace = vec![(START, 0, 0, 0), (ADD_ATOM, 2, 0, 0), (STOP, 0, 0, 0)];
    let actions = pack_actions(&trace, &[3, 0], 1, steps, atoms);
    // Type 2 is carbon with one parent hydrogen: H(g) = 1, heavy C1.
    let mut ion = vec![0u32; 4 * 4 * 12];
    let mut ion_meta = vec![0u32; 4 * 4];
    for (p, (c, m, r)) in [mk(2, 5), mk(4, -3), mk(5, 7)].iter().enumerate() {
        for e in 0..10 {
            ion[(p * 4) * 12 + e] = u32::from(c[e]);
        }
        ion[(p * 4) * 12 + 10] = *m;
        ion[(p * 4) * 12 + 11] = (*r as u32).wrapping_add(LANE_BIAS);
        ion_meta[p * 4] = 1;
        ion_meta[p * 4 + 2] = 1;
    }
    let mut row = vec![0u32; EVIDENCE_ROW_WORDS];
    evidence_lane(&actions, &[0, 1], &ion, &ion_meta, 0, steps as u32, atoms as u32, 1, 1, 4, 4, &mut row);
    // Peak 3 has no hypothesis in the lane buffers (kept 0); the first two
    // records agree with the wrapper and the rest of the row is zero.
    assert_eq!(&row[..14], &sum.to_row()[..14]);
}

#[test]
fn launchers_reject_addresses_beyond_u32() {
    // The device-address checks behind every ion launcher, unit-tested
    // directly (a bound buffer this large cannot be allocated in a test).
    // Every `ion`/`ion_meta` word and every narrowed lane scalar must fit
    // `u32`; larger layouts would wrap device addresses lane onto lane.
    assert!(check_device_len("ion", u32::MAX as usize).is_ok());
    assert!(check_device_len("ion", u32::MAX as usize + 1).is_err());
    assert_eq!(check_device_scalar("ion_assign J", 8).unwrap(), 8);
    assert!(check_device_scalar("ion_assign J", u32::MAX as usize + 1).is_err());
    // The reviewer's overflow shape: B = 21_846, F = 8, N = 256, J = 8.
    // The B * F * N lanes fit u32, but the ion buffer (lanes * J * 12)
    // holds 4_295_098_368 words, so the last spectrum's base wraps to zero
    // on device: the buffer-length check refuses it.
    let lanes = 21_846usize * 8 * 256;
    let ion_words = lanes * 8 * 12;
    assert_eq!(ion_words, 4_295_098_368, "reviewer ion buffer size");
    assert!(check_device_len("ion", ion_words).is_err());
}

#[test]
fn evidence_rejects_bad_shapes() {    let device = dev();
    let actions_t = upload_ids(&[0u32; 20], vec![1, 20], &device);
    let slot_t = upload_ids(&[0u32; 2], vec![1, 2], &device);
    let ion_t = upload_ids(&[0u32; 48], vec![1, 1, 1, 4, 12], &device);
    let im_t = upload_ids(&[0u32; 4], vec![1, 1, 1, 4], &device);
    let mut ev_t = upload_ids(&[0u32; 18], vec![1, 18], &device);
    // Stride mismatch (steps * 4 + atoms + 4 = 12, not 20).
    assert!(ms2_ion::ion_evidence(&actions_t, &slot_t, &ion_t, &im_t, &mut ev_t, 1, 4, 1).is_err());
    // Evidence width must be 18.
    let actions_ok = upload_ids(&[0u32; 12], vec![1, 12], &device);
    let mut bad_ev = upload_ids(&[0u32; 8], vec![1, 8], &device);
    assert!(ms2_ion::ion_evidence(&actions_ok, &slot_t, &ion_t, &im_t, &mut bad_ev, 1, 4, 1).is_err());
    // Rows must equal B * K.
    assert!(ms2_ion::ion_evidence(&actions_ok, &slot_t, &ion_t, &im_t, &mut ev_t, 1, 4, 2).is_err());
}

// ---------------------------------------------------------------------------
// Scored evidence (`ms2_ion_evidence_scored`) and evidence features
// (`ms2_ion_evidence_features`) versus their twins (I3a kernels).
// ---------------------------------------------------------------------------

/// Crafted `[B, 1, N, J, 12]` / `[B, 1, N, 4]` ion buffers for the scored
/// tests: the candidate is one C H3 atom with no open valence, so a kept
/// hypothesis qualifies exactly when its heavy vector is C1 with hydrogen 4
/// (`s = 0`). Peaks in `qualifying` hold that hypothesis at kept index 0;
/// the rest hold a non-matching C2 vector. Peaks in `incomplete` carry an
/// exhausted status word (bit 7 behind the evidence status).
fn scored_ion_buffers(
    b: usize,
    n: usize,
    j: usize,
    qualifying: &[Vec<usize>],
    incomplete: &[Vec<usize>],
) -> (Vec<u32>, Vec<u32>) {
    let mut ion = vec![0u32; b * n * j * 12];
    let mut ion_meta = vec![0u32; b * n * 4];
    for bi in 0..b {
        for p in 0..n {
            let w = (bi * n + p) * j * 12;
            let m = (bi * n + p) * 4;
            let qual = qualifying[bi].contains(&p);
            if qual {
                ion[w] = 1; // C
                ion[w + 1] = 4; // H: s = 4 - 3 - 1 = 0
            } else {
                ion[w] = 2; // C2 never matches the C1 candidate
                ion[w + 1] = 4;
            }
            ion[w + 10] = 24_000_000 + p as u32;
            ion[w + 11] = LANE_BIAS;
            ion_meta[m] = 1;
            ion_meta[m + 2] = 1;
            if incomplete[bi].contains(&p) {
                ion_meta[m + 3] = ION_SEARCH_EXHAUSTED;
            }
        }
    }
    (ion, ion_meta)
}

fn assert_evidence_f_eq(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length mismatch");
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        let tol = 1e-6 * w.abs().max(1.0);
        assert!(
            (g - w).abs() <= tol,
            "{what}: index {i} got {g}, want {w}"
        );
    }
}

/// Random scored-evidence fixtures over the two I3a kernels: no evidence,
/// 1..=6 qualifying peaks (more than `E = 4`), ties in log-probability
/// (smaller peak position wins), unfinished trajectories, incomplete support
/// (bit 7) and invalid slots. Every output element is compared against the
/// twin on poisoned outputs, for both kernels.
#[test]
fn scored_evidence_and_features_match_twins() {
    let steps = 4usize;
    let atoms = 2usize;
    let (b, f, n, j, k) = (2usize, 1usize, 6usize, 2usize, 2usize);
    let rows = b * k;
    // One C H3 atom, finished; the unfinished row reuses the trace with a
    // clear status word.
    let trace = vec![(START, 0, 0, 0), (ADD_ATOM, 4, 0, 0), (STOP, 0, 0, 0)];
    let finished = pack_actions(&trace, &[0, 0], 1, steps, atoms);
    let open_row = pack_actions(&trace, &[0, 0], 0, steps, atoms);
    for seed in [11u64, 12, 13] {
        let mut rng = Rng::seeded(seed);
        // Qualifying peaks per spectrum: (6, 5), (4, 2), (1, 0) by seed, so
        // the seeds cover every count 0..=6: more than `E`, within `E` and
        // no-evidence spectra.
        let (nq0, nq1) = [(6usize, 5usize), (4, 2), (1, 0)][(seed - 11) as usize];
        let qual0: Vec<usize> = (0..nq0).collect();
        let qual1: Vec<usize> = (0..nq1).collect();
        // One incomplete-support peak per spectrum (peak 3 of spectrum 0 is
        // qualifying but outside the top 4: the flag still applies).
        let (ion, ion_meta) = scored_ion_buffers(
            b,
            n,
            j,
            &[qual0.clone(), qual1.clone()],
            &[vec![3], vec![0]],
        );
        // Log-probabilities: random in [-5, 0), with an exact tie between
        // peaks 1 and 2 of spectrum 0 (the smaller position must win).
        let width = j + 1;
        let mut log_prob = vec![-10.0f32; b * f * n * width];
        let draws = rng.uniform_vec(b * n, -5.0, 0.0);
        for bi in 0..b {
            for p in 0..n {
                log_prob[(bi * n + p) * width] = draws[bi * n + p];
            }
        }
        log_prob[(0 * n + 2) * width] = log_prob[(0 * n + 1) * width];
        // Exact tie at the top between peaks 0 and 1 of spectrum 0 (0.0
        // beats every draw in [-5, 0)): the smaller position must win.
        log_prob[(0 * n) * width] = 0.0;
        log_prob[(0 * n + 1) * width] = 0.0;
        // Trajectories: (b0 finished), (b0 unfinished), (b1 finished),
        // (b1 finished with an invalid slot).
        let actions = [finished.clone(), open_row.clone(), finished.clone(), finished.clone()]
            .concat();
        let traj_slot = vec![0, 1, 0, 1, 0, 1, u32::MAX, 1];
        // Twin rows.
        let mut want = vec![0u32; rows * EVIDENCE_ROW_WORDS];
        for r in 0..rows {
            evidence_lane_scored(
                &actions,
                &traj_slot,
                &ion,
                &ion_meta,
                &log_prob,
                r as u32,
                steps as u32,
                atoms as u32,
                k as u32,
                f as u32,
                n as u32,
                j as u32,
                &mut want,
            );
        }
        // Kernel on poisoned outputs.
        let device = dev();
        let stride = steps * 4 + atoms + 4;
        let actions_t = upload_ids(&actions, vec![rows, stride], &device);
        let slot_t = upload_ids(&traj_slot, vec![rows, 2], &device);
        let ion_t = upload_ids(&ion, vec![b, f, n, j, 12], &device);
        let im_t = upload_ids(&ion_meta, vec![b, f, n, 4], &device);
        let lp_t = upload_f(&log_prob, vec![b, f, n, width], &device);
        let mut ev_t = upload_ids(
            &poison_ids(rows * EVIDENCE_ROW_WORDS, &device),
            vec![rows, EVIDENCE_ROW_WORDS],
            &device,
        );
        ms2_ion::ion_evidence_scored(
            &actions_t,
            &slot_t,
            &ion_t,
            &im_t,
            &lp_t,
            &mut ev_t,
            steps as u32,
            atoms as u32,
            k as u32,
        )
        .unwrap();
        check_launches(&device).unwrap();
        let got = ev_t.try_to_vec().unwrap();
        assert_ids(&got, &want, "seed {seed}: scored evidence rows");
        // Spot rules on top of the twin comparison: spectrum 0 keeps the
        // best `min(count, 4)` of its qualifying peaks by log-probability
        // with ties broken by position, and carries bit 7 from the
        // incomplete peak.
        let r0 = &got[..EVIDENCE_ROW_WORDS];
        assert_eq!(r0[1] as usize, nq0, "seed {seed}: qualifying peaks counted");
        assert_ne!(
            r0[0] & EVIDENCE_SUPPORT_INCOMPLETE,
            0,
            "seed {seed}: incomplete support flagged"
        );
        let mut order: Vec<usize> = qual0.clone();
        order.sort_by(|&a, &c| {
            log_prob[a * width]
                .partial_cmp(&log_prob[c * width])
                .unwrap()
                .reverse()
                .then(a.cmp(&c))
        });
        for (q, &p) in order.iter().take(nq0.min(4)).enumerate() {
            assert_eq!(r0[2 + q * 4], p as u32, "seed {seed}: record {q} peak");
            assert_eq!(r0[2 + q * 4 + 1], 0, "seed {seed}: record {q} hypothesis");
        }
        if nq0 >= 2 {
            // The exact top tie goes to the smaller peak position.
            assert_eq!(r0[2], 0, "seed {seed}: tie broken by smaller position");
            assert_eq!(r0[6], 1, "seed {seed}: tie broken by smaller position");
        }
        // Unfinished and invalid-slot rows are zero.
        assert_eq!(
            &got[EVIDENCE_ROW_WORDS..2 * EVIDENCE_ROW_WORDS],
            &vec![0u32; EVIDENCE_ROW_WORDS],
            "seed {seed}: unfinished writes zeros"
        );
        assert_eq!(
            &got[3 * EVIDENCE_ROW_WORDS..4 * EVIDENCE_ROW_WORDS],
            &vec![0u32; EVIDENCE_ROW_WORDS],
            "seed {seed}: invalid slot writes zeros"
        );
        // Spectrum-1 row follows its qualifying count (6, 3 or 0 records).
        let r2 = &got[2 * EVIDENCE_ROW_WORDS..3 * EVIDENCE_ROW_WORDS];
        assert_eq!(r2[1] as usize, nq1, "seed {seed}: spectrum-1 count");
        if nq1 == 0 {
            // No qualifying peak, but peak 0 has exhausted support: status
            // carries bit 7 with count 0.
            assert_eq!(r2[0], EVIDENCE_SUPPORT_INCOMPLETE, "seed {seed}: incomplete bit only");
        }
        // Record area past the stored count is zero.
        for q in nq1.min(4)..4 {
            assert_eq!(
                &r2[2 + q * 4..2 + q * 4 + 4],
                &[0u32; 4],
                "seed {seed}: spectrum-1 record {q} padding is zero"
            );
        }
        // Features kernel on the kernel's evidence rows: the twin runs on
        // the same rows, so the comparison isolates the features lane.
        let mut kept = vec![0u32; b * n * 3];
        for bi in 0..b {
            for p in 0..n {
                kept[(bi * n + p) * 3] = p as u32;
                kept[(bi * n + p) * 3 + 1] = 100_000_000 + p as u32;
            }
        }
        let mut meta = vec![0u32; b * 8];
        for bi in 0..b {
            meta[bi * 8 + 4] = 100;
        }
        let mut want_f = vec![0.0f32; rows * 2];
        for r in 0..rows {
            evidence_features_lane(
                &got,
                &log_prob,
                &traj_slot,
                &kept,
                &meta,
                r as u32,
                k as u32,
                f as u32,
                n as u32,
                j as u32,
                &mut want_f,
            );
        }
        let kept_t = upload_ids(&kept, vec![b, n, 3], &device);
        let meta_t = upload_ids(&meta, vec![b, 8], &device);
        let mut ef_t = upload_f(&poison_f(rows * 2), vec![rows, 2], &device);
        ms2_ion::ion_evidence_features(
            &ev_t,
            &lp_t,
            &slot_t,
            &kept_t,
            &meta_t,
            &mut ef_t,
            k as u32,
        )
        .unwrap();
        check_launches(&device).unwrap();
        let got_f = ef_t.to_f32();
        assert_evidence_f_eq(&got_f, &want_f, "seed {seed}: evidence features");
        // No-evidence rows take the (0, 1) defaults exactly.
        for r in 0..rows {
            if want[r * EVIDENCE_ROW_WORDS + 1] == 0 {
                assert_eq!(got_f[r * 2], 0.0, "seed {seed}: row {r} max default");
                assert_eq!(got_f[r * 2 + 1], 1.0, "seed {seed}: row {r} residual default");
            }
        }
    }
}

/// Launch-error checks for the two I3a kernels: shape mismatches are
/// refused before any launch, never a panic or a partial write.
#[test]
fn scored_evidence_and_features_reject_bad_shapes() {
    let device = dev();
    let steps = 4u32;
    let atoms = 2u32;
    let (b, f, n, j, k) = (1usize, 1usize, 2usize, 2usize, 1usize);
    let stride = steps as usize * 4 + atoms as usize + 4;
    let actions_t = upload_ids(&vec![0u32; stride], vec![1, stride], &device);
    let slot_t = upload_ids(&vec![0u32; 2], vec![1, 2], &device);
    let ion_t = upload_ids(&vec![0u32; b * f * n * j * 12], vec![b, f, n, j, 12], &device);
    let im_t = upload_ids(&vec![0u32; b * f * n * 4], vec![b, f, n, 4], &device);
    let lp_t = upload_f(&vec![0.0; b * f * n * (j + 1)], vec![b, f, n, j + 1], &device);
    let mut ev_t = upload_ids(&vec![0u32; 18], vec![1, 18], &device);
    // Evidence width must be 18.
    let mut bad_ev = upload_ids(&vec![0u32; 17], vec![1, 17], &device);
    assert!(ms2_ion::ion_evidence_scored(
        &actions_t, &slot_t, &ion_t, &im_t, &lp_t, &mut bad_ev, steps, atoms, k as u32
    )
    .is_err());
    // Log-probability width must be J + 1.
    let bad_lp = upload_f(&vec![0.0; b * f * n * j], vec![b, f, n, j], &device);
    assert!(ms2_ion::ion_evidence_scored(
        &actions_t, &slot_t, &ion_t, &im_t, &bad_lp, &mut ev_t, steps, atoms, k as u32
    )
    .is_err());
    // Rows must equal B * K.
    assert!(ms2_ion::ion_evidence_scored(
        &actions_t, &slot_t, &ion_t, &im_t, &lp_t, &mut ev_t, steps, atoms, 2
    )
    .is_err());
    // Features: evidence_f width must be 2.
    let kept_t = upload_ids(&vec![0u32; 1 * 2 * 3], vec![1, 2, 3], &device);
    let meta_t = upload_ids(&vec![0u32; 8], vec![1, 8], &device);
    let mut bad_ef = upload_f(&vec![0.0; 3], vec![1, 3], &device);
    assert!(ms2_ion::ion_evidence_features(
        &ev_t, &lp_t, &slot_t, &kept_t, &meta_t, &mut bad_ef, k as u32
    )
    .is_err());
    // Features: evidence width must be 18.
    let mut ef_t = upload_f(&vec![0.0; 2], vec![1, 2], &device);
    assert!(ms2_ion::ion_evidence_features(
        &bad_ev, &lp_t, &slot_t, &kept_t, &meta_t, &mut ef_t, k as u32
    )
    .is_err());
}

#[test]
fn ion_labels_pack_with_valid_flags() {
    // The upload rows behind `ms2_ion_label_mask`: raw index, 10 counts, a
    // valid flag; overflow rows carry valid 0 and never match (covered in
    // `label_mask_states_partial_and_invalid_rows`). Packing is the exact
    // inverse of unpacking, in label sort order.
    use mamba3::models::ms2::ion::{IonLabel, IonLabels};
    let labels = IonLabels {
        labels: vec![
            IonLabel {
                raw_index: 3,
                counts: [1, 4, 0, 0, 0, 0, 0, 0, 0, 0],
            },
            IonLabel {
                raw_index: 5,
                counts: [2, 6, 0, 0, 0, 0, 0, 0, 0, 0],
            },
        ],
        overflow: 2,
    };
    let mut packed = vec![0u32; labels.labels.len() * 12];
    for (i, l) in labels.labels.iter().enumerate() {
        packed[i * 12] = l.raw_index;
        for e in 0..10 {
            packed[i * 12 + 1 + e] = u32::from(l.counts[e]);
        }
        packed[i * 12 + 11] = 1;
    }
    let mut unpacked = Vec::new();
    for i in (0..packed.len()).step_by(12) {
        let row = &packed[i..i + 12];
        assert_eq!(row[11], 1, "valid flag survives");
        let mut counts = [0u16; 10];
        for e in 0..10 {
            counts[e] = row[1 + e] as u16;
        }
        unpacked.push(IonLabel {
            raw_index: row[0],
            counts,
        });
    }
    assert_eq!(unpacked, labels.labels);
    let keys: Vec<(u32, Composition)> = unpacked.iter().map(|l| (l.raw_index, l.counts)).collect();
    let mut sorted = keys.clone();
    sorted.sort();
    assert_eq!(keys, sorted, "upload order is sorted");
}
