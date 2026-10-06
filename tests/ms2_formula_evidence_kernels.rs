//! E1 kernel-versus-twin tests for formula evidence.
//!
//! Every device call runs on poisoned outputs, is followed by
//! [`check_launches`], and is compared element-for-element with the host
//! twin lane: a dropped launch (stale poison) or a wrong word fails. Integer
//! outputs compare exactly, floats within `1e-6`. Sizes stay small on the
//! CPU runtime; the supervisor runs the same file on wgpu.
//!
//! Launch-counter assertions live in tests that hold `serial()` below: the
//! counters are process-global, so every measured window holds this guard
//! to stay exclusive.

#![cfg(feature = "backend")]

use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

use mamba3::backend::{Device, check_launches, launch_count, reset_launch_count};
use mamba3::backends::Auto;
use mamba3::error::Error;
use mamba3::models::ms2::MolGraph;
use mamba3::models::ms2::chem::{Composition, ELEMENTS, HYDROGEN, composition_mass, ion};
use mamba3::models::ms2::formula::FormulaTable;
use mamba3::models::ms2::formula_evidence as twin;
use mamba3::models::ms2::formula_head::DeviceFormulaTable;
use mamba3::models::ms2::ion::{IonLimits, ion_assign};
use mamba3::tensor::Tensor;
use mamba3::tensor::ops::index::IdTensor;
use mamba3::tensor::ops::ms2;
use mamba3::tensor::ops::ms2_formula_evidence as kernels;
use serde_json::Value;

type R = Auto;

/// Serialises the tests of this binary: kernel launches share the
/// process-global launch counter, so the counting test below holds this
/// guard while it measures, and every other test holds it while it
/// launches, keeping every measured window exclusive.
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn serial() -> std::sync::MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

fn dev() -> Device<R> {
    Device::<R>::default()
}

fn upload_ids(data: &[u32], shape: Vec<usize>, device: &Device<R>) -> IdTensor<R> {
    IdTensor::from_slice(data, shape, device).unwrap()
}

fn upload_f(data: &[f32], shape: Vec<usize>, device: &Device<R>) -> Tensor<R, f32> {
    Tensor::<R, f32>::from_f32(data, shape, device).unwrap()
}

fn assert_ids(actual: &[u32], expected: &[u32], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_eq!(*a, *e, "{what}: word {i} differs");
    }
}

fn assert_f32_close(actual: &[f32], expected: &[f32], tol: f32, what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(!a.is_nan(), "{what}: word {i} is NaN (stale poison)");
        assert!(
            (a - e).abs() <= tol,
            "{what}: word {i} differs: got {a}, want {e}"
        );
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

/// Peak m/z values from true ion masses (via `chem::ion`) plus a far decoy.
fn spectrum_peaks(parent: &Composition, adduct: u16) -> Vec<u32> {
    const ORDER: [usize; 9] = [0, 2, 3, 4, 5, 6, 7, 8, 9];
    let mut peaks = Vec::new();
    for shift in [-1i32, 0, 1, 2] {
        if let Some(hyp) = ion(parent, adduct, shift).expect("ion mass fits") {
            let mz = if adduct == 1 {
                hyp.mz.checked_sub(549).expect("peak fits")
            } else {
                hyp.mz.checked_add(549).expect("peak fits")
            };
            peaks.push(mz);
        }
    }
    // A half-carbon fragment.
    if parent[0] >= 2 {
        let mut c = *parent;
        c[0] /= 2;
        if let Some(hyp) = ion(&c, adduct, 0).expect("ion mass fits") {
            let mz = if adduct == 1 {
                hyp.mz - 549
            } else {
                hyp.mz + 549
            };
            peaks.push(mz);
        }
    }
    let _ = ORDER;
    peaks.sort_unstable();
    peaks.dedup();
    peaks.push(10_000_000);
    peaks
}

/// Pad peak m/z values with zeros (padding slots) to width `n`.
fn pad_to(mut peaks: Vec<u32>, n: usize) -> Vec<u32> {
    peaks.truncate(n);
    while peaks.len() < n {
        peaks.push(0);
    }
    peaks
}

/// Pack one spectrum's kept buffers with descending intensities.
fn pack_kept(peaks: &[u32], start_intensity: f32) -> (Vec<u32>, Vec<f32>) {
    let n = peaks.len();
    let mut kept = vec![0u32; n * 3];
    let mut kept_f = vec![0.0f32; n * 2];
    for (p, &mz) in peaks.iter().enumerate() {
        kept[p * 3] = p as u32;
        kept[p * 3 + 1] = mz;
        kept_f[p * 2] = start_intensity - p as f32 * 0.01;
        kept_f[p * 2 + 1] = 1.0;
    }
    (kept, kept_f)
}

fn pack_cand(comps: &[Composition]) -> Vec<u32> {
    let mut cand = vec![0u32; comps.len() * 13];
    for (i, c) in comps.iter().enumerate() {
        for e in 0..10 {
            cand[i * 13 + e] = u32::from(c[e]);
        }
        cand[i * 13 + 10] = composition_mass(c).expect("candidate mass fits");
        cand[i * 13 + 11] = 1;
        cand[i * 13 + 12] = u32::MAX;
    }
    cand
}

// ---------------------------------------------------------------------------
// Kernel 1: evidence peaks versus the twin.
// ---------------------------------------------------------------------------

#[test]
fn evidence_peaks_matches_twin() {
    let _guard = serial();
    let device = dev();
    let (b, n, p) = (3usize, 8usize, 5usize);
    let butanol = graph_of("2-butanol").composition();
    let aceto = graph_of("acetonitrile").composition();
    // Six true peaks each; padded with zeros to N. The NaN intensity sits
    // on a real peak of spectrum 0.
    let peaks0 = pad_to(spectrum_peaks(&butanol, 1), n);
    let peaks1 = pad_to(spectrum_peaks(&aceto, 2), n);
    let (kept0, mut kept_f0) = pack_kept(&peaks0, 1.0);
    // A NaN intensity is never selected, on either side.
    kept_f0[3 * 2] = f32::NAN;
    let (kept1, kept_f1) = pack_kept(&peaks1, 0.9);
    let kept2 = vec![0u32; n * 3];
    let kept_f2 = vec![0.5f32; n * 2];
    let mut kept = Vec::new();
    kept.extend_from_slice(&kept0);
    kept.extend_from_slice(&kept1);
    kept.extend_from_slice(&kept2);
    let mut kept_f = Vec::new();
    kept_f.extend_from_slice(&kept_f0);
    kept_f.extend_from_slice(&kept_f1);
    kept_f.extend_from_slice(&kept_f2);
    // Spectrum 0: adduct 1, all 8 eligible. Spectrum 1: adduct 2, 5 peaked.
    // Spectrum 2: unknown adduct, nothing eligible.
    let meta = vec![
        n as u32, 0, 0, 1, 100, 0, 0, 0, //
        5, 0, 0, 2, 200, 0, 0, 0, //
        n as u32, 0, 0, 0, 100, 0, 0, 0,
    ];
    let spec = vec![50u32, 0, 75, 0, 50, 0];
    let (want_peaks, want_w) = twin::evidence_peaks(&kept, &kept_f, &meta, &spec, b, n, p);
    // The NaN peak of spectrum 0 is absent from the selection.
    assert!(!want_peaks[0..p * 4].chunks(4).any(|r| r[0] == 3));
    let kept_t = upload_ids(&kept, vec![b, n, 3], &device);
    let kept_f_t = upload_f(&kept_f, vec![b, n, 2], &device);
    let meta_t = upload_ids(&meta, vec![b, 8], &device);
    let spec_t = upload_ids(&spec, vec![b, 2], &device);
    let mut peaks_t = upload_ids(&vec![u32::MAX; b * p * 4], vec![b, p, 4], &device);
    let mut w_t = upload_f(&vec![f32::NAN; b * p], vec![b, p], &device);
    kernels::evidence_peaks(&kept_t, &kept_f_t, &meta_t, &spec_t, &mut peaks_t, &mut w_t)
        .unwrap();
    check_launches(&device).unwrap();
    assert_ids(&peaks_t.try_to_vec().unwrap(), &want_peaks, "ev_peaks");
    assert_f32_close(&w_t.to_f32(), &want_w, 1e-6, "ev_w");
}

// ---------------------------------------------------------------------------
// Kernel 2: formula evidence versus the twin, single and chunked.
// ---------------------------------------------------------------------------

/// Two spectra with distinct productive candidates, hand-packed once.
struct EvFixture {
    b: usize,
    m: usize,
    n: usize,
    p: usize,
    work_max: u32,
    kept: Vec<u32>,
    kept_f: Vec<f32>,
    meta: Vec<u32>,
    spec: Vec<u32>,
    cand: Vec<u32>,
    ev_peaks: Vec<u32>,
    ev_w: Vec<f32>,
    want_ev: Vec<f32>,
}

fn ev_fixture() -> EvFixture {
    let (b, m, n, p) = (2usize, 3usize, 6usize, 5usize);
    let work_max = 512u32;
    let butanol = graph_of("2-butanol").composition();
    let aceto = graph_of("acetonitrile").composition();
    let peaks0 = pad_to(spectrum_peaks(&butanol, 1), n);
    let peaks1 = pad_to(spectrum_peaks(&aceto, 2), n);
    let (kept0, kept_f0) = pack_kept(&peaks0, 1.0);
    let (kept1, kept_f1) = pack_kept(&peaks1, 0.8);
    let mut kept = Vec::new();
    kept.extend_from_slice(&kept0);
    kept.extend_from_slice(&kept1);
    let mut kept_f = Vec::new();
    kept_f.extend_from_slice(&kept_f0);
    kept_f.extend_from_slice(&kept_f1);
    let meta = vec![
        n as u32, 0, 0, 1, 100, 0, 0, 0, //
        n as u32, 0, 0, 2, 200, 0, 0, 0,
    ];
    let spec = vec![50u32, 0, 75, 0];
    let (ev_peaks, ev_w) = twin::evidence_peaks(&kept, &kept_f, &meta, &spec, b, n, p);
    // Distinct productive candidates per spectrum, with padding slots.
    let mut minus_c = butanol;
    minus_c[0] -= 1;
    let mut plus_o = aceto;
    plus_o[3] += 1;
    let mut cand = pack_cand(&[butanol, minus_c, [0; 10]]);
    cand[2 * 13 + 11] = 0; // padding slot of spectrum 0
    let mut cand1 = pack_cand(&[aceto, plus_o, [0; 10]]);
    cand1[2 * 13 + 11] = 0; // padding slot of spectrum 1
    let mut cand_full = Vec::new();
    cand_full.extend_from_slice(&cand);
    cand_full.extend_from_slice(&cand1);
    // Both spectra explain at least one peak (productive).
    let want_ev = twin::formula_evidence(
        &cand_full, &ev_peaks, &ev_w, &meta, &spec, b, m, p, work_max, u32::MAX,
    );
    assert!(want_ev[0] >= 1.0, "spectrum 0 productive");
    assert!(want_ev[m * 4] >= 1.0, "spectrum 1 productive");
    EvFixture {
        b,
        m,
        n,
        p,
        work_max,
        kept,
        kept_f,
        meta,
        spec,
        cand: cand_full,
        ev_peaks,
        ev_w,
        want_ev,
    }
}

fn run_formula_evidence(
    fx: &EvFixture,
    dispatch_tests_max: u64,
    h_cap_max: u32,
    tol_max: u32,
) -> Vec<f32> {
    let device = dev();
    let cand_t = upload_ids(&fx.cand, vec![fx.b, fx.m, 13], &device);
    let peaks_t = upload_ids(&fx.ev_peaks, vec![fx.b, fx.p, 4], &device);
    let w_t = upload_f(&fx.ev_w, vec![fx.b, fx.p], &device);
    let meta_t = upload_ids(&fx.meta, vec![fx.b, 8], &device);
    let spec_t = upload_ids(&fx.spec, vec![fx.b, 2], &device);
    let mut ev_t = upload_f(
        &vec![f32::NAN; fx.b * fx.m * 4],
        vec![fx.b, fx.m, 4],
        &device,
    );
    kernels::formula_evidence(
        &cand_t,
        &peaks_t,
        &w_t,
        &meta_t,
        &spec_t,
        &mut ev_t,
        fx.work_max,
        dispatch_tests_max,
        h_cap_max,
        tol_max,
    )
    .unwrap();
    check_launches(&device).unwrap();
    ev_t.to_f32()
}

#[test]
fn formula_evidence_matches_twin() {
    let _guard = serial();
    let fx = ev_fixture();
    let got = run_formula_evidence(&fx, u64::MAX, u32::MAX, 1_007_825);
    assert_f32_close(&got, &fx.want_ev, 1e-6, "cand_ev single launch");
    // Padding slots are exact zeros.
    for b in 0..fx.b {
        let row = &got[(b * fx.m + 2) * 4..(b * fx.m + 3) * 4];
        assert_eq!(row, &[0.0; 4][..], "spectrum {b} padding slot");
    }
}

/// Per-lane hydrogen-trial bound of item 1, replicated here with plain
/// `u64` arithmetic for the dispatch expectations below.
fn trials_bound_of(h_cap_max: u32, tol_max: u32) -> u64 {
    let s_max = (u64::from(h_cap_max) * 7_825 + 2 * u64::from(tol_max)) / 1_000_000;
    let per_s = (2 * u64::from(tol_max)) / 7_825 + 2;
    ((s_max + 1) * per_s).min(u64::from(h_cap_max) + 1)
}

/// Chunked dispatch (1 lane, 2 lanes, spectrum-crossing chunks) equals the
/// single-launch result exactly.
#[test]
fn formula_evidence_chunking_equals_single_launch() {
    let _guard = serial();
    let fx = ev_fixture();
    // True bounds for the fixture (max H + 3 <= 20; tolerances <= 2000):
    // the clamp never engages, chunking changes no result.
    let (h_cap_max, tol_max) = (20u32, 2000u32);
    let trials = trials_bound_of(h_cap_max, tol_max);
    let per_lane = u64::from(fx.work_max) * fx.p as u64 * trials;
    let single = run_formula_evidence(&fx, u64::MAX, h_cap_max, tol_max);
    let one_lane = run_formula_evidence(&fx, per_lane, h_cap_max, tol_max);
    assert_eq!(one_lane, single, "1-lane chunks equal single launch");
    let two_lanes = run_formula_evidence(&fx, 2 * per_lane, h_cap_max, tol_max);
    assert_eq!(two_lanes, single, "2-lane chunks equal single launch");
    // 4-lane chunks cross the spectrum boundary (lanes 0-2 are spectrum 0,
    // lanes 3-5 spectrum 1 with distinct candidates).
    let crossing = run_formula_evidence(&fx, 4 * per_lane, h_cap_max, tol_max);
    assert_eq!(crossing, single, "crossing chunks equal single launch");
}

// ---------------------------------------------------------------------------
// Kernel 3: formula features versus the twin and `count_features`.
// ---------------------------------------------------------------------------

#[test]
fn formula_features_matches_twin_and_count_features() {
    let _guard = serial();
    let device = dev();
    let fx = ev_fixture();
    let log_host = mamba3::models::ms2::twin::log_table();
    let want = twin::formula_features(&fx.cand, &fx.want_ev, &fx.meta, &log_host, fx.b, fx.m);
    let cand_t = upload_ids(&fx.cand, vec![fx.b, fx.m, 13], &device);
    let ev_t = upload_f(&fx.want_ev, vec![fx.b, fx.m, 4], &device);
    let meta_t = upload_ids(&fx.meta, vec![fx.b, 8], &device);
    let log_t = upload_f(&log_host, vec![1024], &device);
    let mut out_t = upload_f(
        &vec![f32::NAN; fx.b * fx.m * 16],
        vec![fx.b, fx.m, 16],
        &device,
    );
    kernels::formula_features(&cand_t, &ev_t, &meta_t, &log_t, &mut out_t).unwrap();
    check_launches(&device).unwrap();
    let got = out_t.to_f32();
    assert_f32_close(&got, &want, 1e-6, "cand_feat16");
    // Padding slots are exact zeros in all 16 words.
    for b in 0..fx.b {
        let row = &got[(b * fx.m + 2) * 16..(b * fx.m + 3) * 16];
        assert_eq!(row, &[0.0; 16][..], "spectrum {b} padding row");
    }
    // The first 10 features are bit-equal to `count_features` on the same
    // candidate records.
    let rec_t = cand_t.reshape(vec![fx.b * fx.m, 13]).unwrap();
    let mut cf_t = upload_f(&vec![f32::NAN; fx.b * fx.m * 10], vec![fx.b * fx.m, 10], &device);
    ms2::count_features(&rec_t, &log_t, &mut cf_t, 13).unwrap();
    check_launches(&device).unwrap();
    let cf = cf_t.to_f32();
    for r in 0..fx.b * fx.m {
        assert_eq!(
            &got[r * 16..r * 16 + 10],
            &cf[r * 10..r * 10 + 10],
            "row {r} first 10 features bit-equal to count_features"
        );
    }
}

// ---------------------------------------------------------------------------
// E1F: independent `u32` tolerance, single-lane device runners.
// ---------------------------------------------------------------------------

/// Independent tolerance with `u64` headroom (same algorithm, no wrap).
fn tol_u64(mz: u32, ppm: u32) -> u32 {
    let hi = u64::from(mz / 10_000);
    let lo = u64::from(mz % 10_000);
    (hi * u64::from(ppm) / 1000
        + ((hi * u64::from(ppm) % 1000) * 10_000 + lo * u64::from(ppm)) / 10_000_000)
        as u32
}

fn pack_one_cand(comp: &Composition) -> Vec<u32> {
    let mut cand = vec![0u32; 13];
    for e in 0..10 {
        cand[e] = u32::from(comp[e]);
    }
    cand[10] = composition_mass(comp).expect("candidate mass fits");
    cand[11] = 1;
    cand[12] = u32::MAX;
    cand
}

/// Run kernel 2 on the device for one `(b, m)` lane with explicit buffers.
fn run_evidence_lane_device(
    cand: &[u32],
    ev_peaks: &[u32],
    ev_w: &[f32],
    meta: &[u32],
    spec: &[u32],
    p: usize,
    work_max: u32,
) -> Vec<f32> {
    let device = dev();
    let cand_t = upload_ids(cand, vec![1, 1, 13], &device);
    let peaks_t = upload_ids(ev_peaks, vec![1, p, 4], &device);
    let w_t = upload_f(ev_w, vec![1, p], &device);
    let meta_t = upload_ids(meta, vec![1, 8], &device);
    let spec_t = upload_ids(spec, vec![1, 2], &device);
    let mut out_t = upload_f(&vec![f32::NAN; 4], vec![1, 1, 4], &device);
    kernels::formula_evidence(
        &cand_t, &peaks_t, &w_t, &meta_t, &spec_t, &mut out_t, work_max, u64::MAX, u32::MAX,
        1_007_825,
    )
    .unwrap();
    check_launches(&device).unwrap();
    out_t.to_f32()
}

/// Run kernel 1 on the device for one spectrum with explicit buffers.
fn run_peaks_device(
    kept: &[u32],
    kept_f: &[f32],
    meta: &[u32],
    spec: &[u32],
    n: usize,
    p: usize,
) -> (Vec<u32>, Vec<f32>) {
    let device = dev();
    let kept_t = upload_ids(kept, vec![1, n, 3], &device);
    let kept_f_t = upload_f(kept_f, vec![1, n, 2], &device);
    let meta_t = upload_ids(meta, vec![1, 8], &device);
    let spec_t = upload_ids(spec, vec![1, 2], &device);
    let mut peaks_t = upload_ids(&vec![u32::MAX; p * 4], vec![1, p, 4], &device);
    let mut w_t = upload_f(&vec![f32::NAN; p], vec![1, p], &device);
    kernels::evidence_peaks(&kept_t, &kept_f_t, &meta_t, &spec_t, &mut peaks_t, &mut w_t)
        .unwrap();
    check_launches(&device).unwrap();
    (peaks_t.try_to_vec().unwrap(), w_t.to_f32())
}

/// Run kernel 3 on the device for one `(b, m)` lane with explicit buffers.
fn run_features_lane_device(
    cand: &[u32],
    cand_ev: &[f32],
    meta: &[u32],
    log_table: &[f32],
) -> Vec<f32> {
    let device = dev();
    let cand_t = upload_ids(cand, vec![1, 1, 13], &device);
    let ev_t = upload_f(cand_ev, vec![1, 1, 4], &device);
    let meta_t = upload_ids(meta, vec![1, 8], &device);
    let log_t = upload_f(log_table, vec![1024], &device);
    let mut out_t = upload_f(&vec![f32::NAN; 16], vec![1, 1, 16], &device);
    kernels::formula_features(&cand_t, &ev_t, &meta_t, &log_t, &mut out_t).unwrap();
    check_launches(&device).unwrap();
    out_t.to_f32()
}

// ---------------------------------------------------------------------------
// E1F selection on device (underflow/overflow, saturated width, zero sum,
// order-sensitive sum, E_ion scope rule).
// ---------------------------------------------------------------------------

#[test]
fn evidence_peaks_device_boundary_cases() {
    let _guard = serial();
    // Adduct 1 overflow (`mz > MAX - 549`) ineligible with ppm = 0.
    let kept = vec![0u32, 100_000_000, 0, 1, u32::MAX - 100, 0, 2, 0, 0];
    let kept_f = vec![1.0f32, 1.0, 0.9, 1.0, 0.8, 1.0];
    let meta = vec![3u32, 0, 0, 1, 0, 0, 0, 0];
    let spec = vec![50u32, 0];
    let want = twin::evidence_peaks(&kept, &kept_f, &meta, &spec, 1, 3, 3);
    let (got_p, got_w) = run_peaks_device(&kept, &kept_f, &meta, &spec, 3, 3);
    assert_ids(&got_p, &want.0, "adduct1 overflow peaks");
    assert_f32_close(&got_w, &want.1, 1e-6, "adduct1 overflow weights");
    assert_eq!(&got_p[0..4], &[0, 100_000_000 + 549, 0, 1][..]);
    // Adduct 2 underflow (`mz < 549`) ineligible.
    let kept2 = vec![0u32, 548, 0, 1, 100_000_000, 0, 2, 0, 0];
    let kept_f2 = vec![1.0f32, 1.0, 0.9, 1.0, 0.8, 1.0];
    let meta2 = vec![3u32, 0, 0, 2, 0, 0, 0, 0];
    let (got_p2, got_w2) = run_peaks_device(&kept2, &kept_f2, &meta2, &spec, 3, 3);
    let want2 = twin::evidence_peaks(&kept2, &kept_f2, &meta2, &spec, 1, 3, 3);
    assert_ids(&got_p2, &want2.0, "adduct2 underflow peaks");
    assert_f32_close(&got_w2, &want2.1, 1e-6, "adduct2 underflow weights");
    assert_eq!(&got_p2[0..4], &[1, 100_000_000 - 549, 0, 1][..]);
    // Saturated half-width selects nothing.
    let kept3 = vec![0u32, 100_000_000, 0, 1, 200_000_000, 0];
    let kept_f3 = vec![1.0f32, 1.0, 0.5, 1.0];
    let meta3 = vec![2u32, 0, 0, 1, 100, 0, 0, 0];
    let spec3 = vec![u32::MAX - 1, 0];
    let (got_p3, got_w3) = run_peaks_device(&kept3, &kept_f3, &meta3, &spec3, 2, 2);
    assert_ids(&got_p3, &vec![u32::MAX, 0, 0, 0, u32::MAX, 0, 0, 0], "saturated peaks");
    assert_eq!(&got_w3, &[0.0, 0.0][..]);
    // All-zero intensities: rows valid, weights 0.
    let kept4 = vec![0u32, 100_000_000, 0, 1, 100_001_000, 0, 2, 100_002_000, 0];
    let kept_f4 = vec![0.0f32, 1.0, 0.0, 1.0, 0.0, 1.0];
    let meta4 = vec![3u32, 0, 0, 1, 100, 0, 0, 0];
    let (got_p4, got_w4) = run_peaks_device(&kept4, &kept_f4, &meta4, &spec, 3, 2);
    assert_eq!(got_p4[3], 1);
    assert_eq!(got_p4[7], 1);
    assert_eq!(&got_w4, &[0.0, 0.0][..]);
    // Order-sensitive f32 sum in slot order.
    let big = 16_777_216.0f32;
    let kept5 = vec![0u32, 100_000_000, 0, 1, 100_001_000, 0, 2, 100_002_000, 0];
    let kept_f5 = vec![big, 1.0, 1.0, 1.0, 1.0, 1.0];
    let meta5 = vec![3u32, 0, 0, 1, 0, 0, 0, 0];
    let (got_p5, got_w5) = run_peaks_device(&kept5, &kept_f5, &meta5, &spec, 3, 3);
    let want5 = twin::evidence_peaks(&kept5, &kept_f5, &meta5, &spec, 1, 3, 3);
    assert_ids(&got_p5, &want5.0, "order peaks");
    assert_f32_close(&got_w5, &want5.1, 1e-6, "order weights");
    let mut sum = 0.0f32;
    sum += big;
    sum += 1.0;
    sum += 1.0;
    assert_eq!(sum.to_bits(), 16_777_216.0f32.to_bits());
    assert_eq!(got_w5[0].to_bits(), (big / sum).to_bits());
}

/// Reviewer's finding-2 example on the device: position 0 IS selected, and
/// kernel 2 reports it unexplained for C1 while an in-scope peak explains.
#[test]
fn evidence_peaks_device_e_ion_scope_rule() {
    let _guard = serial();
    let u = 1_007_325u32;
    let ppm = 100u32;
    let kept = vec![0u32, 50_000_000, 0, 1, 49_000_000, 0];
    let kept_f = vec![1.0f32, 1.0, 0.5, 1.0];
    let meta = vec![2u32, 0, 0, 1, ppm, 0, 0, 0];
    let spec = vec![u, 0];
    let (got_p, got_w) = run_peaks_device(&kept, &kept_f, &meta, &spec, 2, 1);
    assert_eq!(&got_p[..], &[0, 50_000_000 + 549, 500, 1][..]);
    assert_eq!(&got_w[..], &[1.0][..]);
    let c1: Composition = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let cand = pack_one_cand(&c1);
    let got = run_evidence_lane_device(&cand, &got_p, &got_w, &meta, &spec, 1, 65_536);
    let want = twin::formula_evidence(&cand, &got_p, &got_w, &meta, &spec, 1, 1, 1, 65_536, u32::MAX);
    assert_f32_close(&got, &want, 1e-6, "C1 out-of-scope");
    assert_eq!(&got[..], &[0.0, 0.0, 1.0, 1.0][..]);
    let assign = ion_assign(
        &c1, 1, 50_000_000, u, ppm,
        &IonLimits { work_max: 65_536, kept: 4 },
    )
    .expect("ion_assign runs");
    assert_eq!(assign.accepted, 0);
    // In-scope contrast: C1 explains target 12,000,000 with small U.
    let mz_small = 12_000_000u32 - 549;
    let tol_small = tol_u64(mz_small, ppm);
    let meta2 = vec![1u32, 0, 0, 1, ppm, 0, 0, 0];
    let spec2 = vec![50u32, 0];
    let ev2 = vec![0, 12_000_000, tol_small, 1];
    let got2 = run_evidence_lane_device(&cand, &ev2, &[1.0], &meta2, &spec2, 1, 65_536);
    assert_eq!(&got2[..], &[1.0, 1.0, 1.0, 1.0][..]);
}

// ---------------------------------------------------------------------------
// E1F evidence on device (budgets, half_p, H-cap, windows, visits, slots).
// ---------------------------------------------------------------------------

#[test]
fn formula_evidence_device_budget_boundaries() {
    let _guard = serial();
    // N2 O1 H4: non-carbon J = 3 * 2 = 6, c[C] = 0, so the walk visits the
    // mixed-radix indices j = 0..min(6, W) and carbon never contributes.
    let comp: Composition = [0, 4, 2, 1, 0, 0, 0, 0, 0, 0];
    let cand = pack_one_cand(&comp);
    let ppm = 100u32;
    // Peak A: the full-vector ion (N2 O1, h = 0) at j = 2 + 3 * 1 = 5.
    let t_a = 2 * 14_003_074 + 15_994_915;
    assert_eq!(t_a, 44_001_063);
    let mz_a = t_a - 549;
    let tol_a = tol_u64(mz_a, ppm);
    // Peak B: the N1 ion (h = 0) at j = 1.
    let t_b = 14_003_074u32;
    let mz_b = t_b - 549;
    let tol_b = tol_u64(mz_b, ppm);
    let meta = vec![1u32, 0, 0, 1, ppm, 0, 0, 0];
    let spec = vec![0u32, 0];
    // (W, hand row for A, hand row for B): J == W + 1, J == W, J > W.
    for (w, want_a, want_b) in [
        (
            5u32,
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 0.0],
        ),
        (
            6u32,
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ),
        (
            2u32,
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 0.0],
        ),
    ] {
        for (t, tol, want, mz) in [
            (t_a, tol_a, want_a, mz_a),
            (t_b, tol_b, want_b, mz_b),
        ] {
            let ev_peaks = vec![0, t, tol, 1];
            let ev_w = vec![1.0f32];
            let got = run_evidence_lane_device(&cand, &ev_peaks, &ev_w, &meta, &spec, 1, w);
            let twin_out =
                twin::formula_evidence(&cand, &ev_peaks, &ev_w, &meta, &spec, 1, 1, 1, w, u32::MAX);
            assert_f32_close(&got, &twin_out, 1e-6, "W = {w} t = {t} device vs twin");
            assert_eq!(&got[..], &want[..], "W = {w} t = {t} hand expectation");
            // Oracle: V = min(6, W) visits, c[C] = 0, so W' = V - 1.
            let v = 6u64.min(w as u64);
            let oracle_w = (v * 1 - 1) as u32;
            assert!(oracle_w >= 1, "oracle budget fits and is >= 1");
            let assign = ion_assign(
                &comp, 1, mz, 0, ppm,
                &IonLimits { work_max: oracle_w, kept: 4 },
            )
            .expect("ion_assign runs");
            assert_eq!(got[0] == 1.0, assign.accepted >= 1, "W = {w} t = {t} vs ion_assign");
        }
    }
}

#[test]
fn formula_evidence_device_half_p_boundaries() {
    let _guard = serial();
    let m_h = ELEMENTS[HYDROGEN].mass;
    let ppm = 100u32;
    let mz = 100_000_000u32;
    let tol = tol_u64(mz, ppm);
    assert_eq!(tol, 1000);
    // C1: E_ion = 1, so U0 puts half_p == m_H, U1 puts m_H + 1.
    let u0 = m_h - tol - 1;
    let u1 = u0 + 1;
    let t = mz + 549;
    let ev_peaks = vec![0, t, tol, 1];
    let ev_w = vec![1.0f32];
    let c1: Composition = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let cand = pack_one_cand(&c1);
    let meta = vec![1u32, 0, 0, 1, ppm, 0, 0, 0];
    for (u, unavailable) in [(u0, false), (u1, true)] {
        let spec = vec![u, 0];
        let got = run_evidence_lane_device(&cand, &ev_peaks, &ev_w, &meta, &spec, 1, 4096);
        let want = twin::formula_evidence(&cand, &ev_peaks, &ev_w, &meta, &spec, 1, 1, 1, 4096, u32::MAX);
        assert_f32_close(&got, &want, 1e-6, "U = {u} device vs twin");
        // Both are ambiguous-or-skipped (huge bound), so unexplained.
        assert_eq!(&got[..], &[0.0, 0.0, 1.0, 1.0][..], "U = {u}");
        let assign = ion_assign(
            &c1, 1, mz, u, ppm,
            &IonLimits { work_max: 4096, kept: 4 },
        )
        .expect("ion_assign runs");
        assert_eq!(assign.accepted, 0, "U = {u} accepted");
        assert_eq!(
            assign.status & mamba3::models::ms2::ion::ION_UNAVAILABLE != 0,
            unavailable,
            "U = {u} unavailable"
        );
    }
}

#[test]
fn formula_evidence_device_hydrogen_cap_endpoints() {
    let _guard = serial();
    let ppm = 100u32;
    // Candidate C1 H2: adduct 1 cap = 2 + 1 + 2 = 5, adduct 2 cap = 4.
    let comp: Composition = [1, 2, 0, 0, 0, 0, 0, 0, 0, 0];
    let cand = pack_one_cand(&comp);
    let m_h = ELEMENTS[HYDROGEN].mass;
    // Adduct 1: h = 5 explained, h = 6 not.
    for (h, explained) in [(5u32, true), (6u32, false)] {
        let t = 12_000_000u32 + h * m_h;
        let mz = t - 549;
        let tol = tol_u64(mz, ppm);
        let ev = vec![0, t, tol, 1];
        let meta = vec![1u32, 0, 0, 1, ppm, 0, 0, 0];
        let spec = vec![0u32, 0];
        let got = run_evidence_lane_device(&cand, &ev, &[1.0], &meta, &spec, 1, 65_536);
        let want = twin::formula_evidence(&cand, &ev, &[1.0], &meta, &spec, 1, 1, 1, 65_536, u32::MAX);
        assert_f32_close(&got, &want, 1e-6, "adduct1 h = {h}");
        assert_eq!(got[0] == 1.0, explained, "adduct1 h = {h}");
        let assign = ion_assign(
            &comp, 1, mz, 0, ppm,
            &IonLimits { work_max: 65_536, kept: 4 },
        )
        .expect("ion_assign runs");
        assert_eq!(assign.accepted >= 1, explained, "adduct1 h = {h} ion_assign");
    }
    // Adduct 2: h = 4 explained, h = 5 not.
    for (h, explained) in [(4u32, true), (5u32, false)] {
        let t = 12_000_000u32 + h * m_h;
        let mz = t + 549;
        let tol = tol_u64(mz, ppm);
        let ev = vec![0, t, tol, 1];
        let meta = vec![1u32, 0, 0, 2, ppm, 0, 0, 0];
        let spec = vec![0u32, 0];
        let got = run_evidence_lane_device(&cand, &ev, &[1.0], &meta, &spec, 1, 65_536);
        let want = twin::formula_evidence(&cand, &ev, &[1.0], &meta, &spec, 1, 1, 1, 65_536, u32::MAX);
        assert_f32_close(&got, &want, 1e-6, "adduct2 h = {h}");
        assert_eq!(got[0] == 1.0, explained, "adduct2 h = {h}");
        let assign = ion_assign(
            &comp, 2, mz, 0, ppm,
            &IonLimits { work_max: 65_536, kept: 4 },
        )
        .expect("ion_assign runs");
        assert_eq!(assign.accepted >= 1, explained, "adduct2 h = {h} ion_assign");
    }
}

#[test]
fn formula_evidence_device_three_hydrogen_window() {
    let _guard = serial();
    let m_h = ELEMENTS[HYDROGEN].mass;
    // t = 2 * m_H with half_p = m_H: at hm = 0 the interval is [1, 3],
    // exactly three hydrogen counts (hand arithmetic).
    let t = 2 * m_h;
    let ppm = 100u32;
    let mz = t - 549;
    let tol = tol_u64(mz, ppm);
    let c1: Composition = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let cand = pack_one_cand(&c1);
    // E_ion(C1) = 1, so U = m_H - tol - 1 puts half_p == m_H.
    let u = m_h - tol - 1;
    let half = tol.saturating_add(u).saturating_add(1);
    assert_eq!(half, m_h, "half_p is exactly m_H");
    let lo = t.saturating_sub(half);
    let hi = t.saturating_add(half);
    // At hm = 0 (visit k = 2 of C1): hand interval.
    let h_lo = lo.div_ceil(m_h);
    let h_hi = (hi / m_h).min(3);
    assert_eq!((h_lo, h_hi), (1, 3), "exactly three hydrogen counts");
    assert_eq!(h_hi - h_lo + 1, 3);
    let ev = vec![0, t, tol, 1];
    let meta = vec![1u32, 0, 0, 1, ppm, 0, 0, 0];
    let spec = vec![u, 0];
    let got = run_evidence_lane_device(&cand, &ev, &[1.0], &meta, &spec, 1, 65_536);
    let want = twin::formula_evidence(&cand, &ev, &[1.0], &meta, &spec, 1, 1, 1, 65_536, u32::MAX);
    assert_f32_close(&got, &want, 1e-6, "three-H window device vs twin");
    let assign = ion_assign(
        &c1, 1, mz, u, ppm,
        &IonLimits { work_max: 65_536, kept: 4 },
    )
    .expect("ion_assign runs");
    assert_eq!(got[0] == 1.0, assign.accepted >= 1, "vs ion_assign");
    assert_eq!(assign.status & mamba3::models::ms2::ion::ION_UNAVAILABLE, 0, "searched");
}

#[test]
fn formula_evidence_device_zero_heavy_and_empty_unknown() {
    let _guard = serial();
    // Flag-1 candidate with zero heavy atoms: zero visits, complete.
    let zero: Composition = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let mut cand = pack_one_cand(&zero);
    cand[10] = 0;
    let ev = vec![0, 12_000_000, 119, 1];
    let meta = vec![1u32, 0, 0, 1, 100, 0, 0, 0];
    let spec = vec![50u32, 0];
    let got = run_evidence_lane_device(&cand, &ev, &[1.0], &meta, &spec, 1, 4096);
    let want = twin::formula_evidence(&cand, &ev, &[1.0], &meta, &spec, 1, 1, 1, 4096, u32::MAX);
    assert_f32_close(&got, &want, 1e-6, "zero heavy");
    assert_eq!(&got[..], &[0.0, 0.0, 1.0, 1.0][..], "zero visits, complete");
    // Zero evidence peaks and unknown U given directly to kernel 2.
    let c1: Composition = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let cand1 = pack_one_cand(&c1);
    let empty = vec![u32::MAX, 0, 0, 0, u32::MAX, 0, 0, 0];
    let spec_unk = vec![u32::MAX, 0];
    let got2 = run_evidence_lane_device(&cand1, &empty, &[0.0, 0.0], &meta, &spec_unk, 2, 4096);
    let want2 = twin::formula_evidence(&cand1, &empty, &[0.0, 0.0], &meta, &spec_unk, 1, 1, 2, 4096, u32::MAX);
    assert_f32_close(&got2, &want2, 1e-6, "empty unknown");
    assert_eq!(&got2[..], &[0.0, 0.0, 0.0, 1.0][..]);
}

#[test]
fn formula_evidence_device_slot31_and_early_exit() {
    let _guard = serial();
    let ppm = 100u32;
    // Explained peak in slot 31 (mask bit 31) with P = 32.
    let c1: Composition = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let cand = pack_one_cand(&c1);
    let mz = 12_000_000u32 - 549;
    let tol = tol_u64(mz, ppm);
    let mut peaks = vec![u32::MAX, 0, 0, 0];
    peaks = peaks.repeat(32);
    peaks[31 * 4] = 7;
    peaks[31 * 4 + 1] = 12_000_000;
    peaks[31 * 4 + 2] = tol;
    peaks[31 * 4 + 3] = 1;
    let mut w = vec![0.0f32; 32];
    w[31] = 1.0;
    let meta = vec![8u32, 0, 0, 1, ppm, 0, 0, 0];
    let spec = vec![50u32, 0];
    let got = run_evidence_lane_device(&cand, &peaks, &w, &meta, &spec, 32, 65_536);
    let want = twin::formula_evidence(&cand, &peaks, &w, &meta, &spec, 1, 1, 32, 65_536, u32::MAX);
    assert_f32_close(&got, &want, 1e-6, "slot 31");
    assert_eq!(&got[..], &[1.0, 1.0, 1.0, 1.0][..], "bit 31 explained");
    let assign = ion_assign(
        &c1, 1, mz, 50, ppm,
        &IonLimits { work_max: 65_536, kept: 4 },
    )
    .expect("ion_assign runs");
    assert!(assign.accepted >= 1);
    // Early exit with complete = 0: C1 N2 O1 H4 (non-carbon J = 3 * 2 = 6)
    // under W = 3 explains both peaks (j = 0 carbon-only, j = 1 N1) inside
    // the prefix.
    let comp_cno: Composition = [1, 4, 2, 1, 0, 0, 0, 0, 0, 0];
    let cand_cno = pack_one_cand(&comp_cno);
    let mz_a = 12_000_000u32 - 549;
    let mz_b = 14_003_074u32 - 549;
    let ev2 = vec![0, 12_000_000, tol_u64(mz_a, ppm), 1, 1, 14_003_074, tol_u64(mz_b, ppm), 1];
    let w2 = vec![0.25f32, 0.75];
    let got4 = run_evidence_lane_device(&cand_cno, &ev2, &w2, &meta, &spec, 2, 3);
    let want4 = twin::formula_evidence(&cand_cno, &ev2, &w2, &meta, &spec, 1, 1, 2, 3, u32::MAX);
    assert_f32_close(&got4, &want4, 1e-6, "early exit");
    assert_eq!(got4[0], 2.0, "both peaks explained");
    assert_eq!(got4[2], 2.0, "n_ev");
    assert_eq!(got4[3], 0.0, "complete = 0 under truncation");
    assert!((got4[1] - 1.0).abs() <= 1e-6, "weight sums both slots");
    // Oracle: V = 3 visits, c[C] = 1, so W' = 3 * 2 - 1 = 5.
    for &mz in &[mz_a, mz_b] {
        let a = ion_assign(
            &comp_cno, 1, mz, 50, ppm,
            &IonLimits { work_max: 5, kept: 4 },
        )
        .expect("ion_assign runs");
        assert!(a.accepted >= 1, "peak {mz} in prefix");
    }
}

// ---------------------------------------------------------------------------
// E1F features on device (precursors, unknowns, w = 0, clamp, counts).
// ---------------------------------------------------------------------------

#[test]
fn formula_features_device_hand_computed() {
    let _guard = serial();
    const H_NET: u64 = 1_007_825 - 549;
    let log_host = mamba3::models::ms2::twin::log_table();
    // Non-zero precursors for both adducts, m_c above and below m_p.
    for adduct in [1u32, 2u32] {
        let prec = 200_000_000u32;
        let (unc, ppm_pre) = (500u32, 100u32);
        let tol = tol_u64(prec, ppm_pre);
        assert_eq!(tol, 2000, "hand tolerance");
        let w = u64::from(tol) + u64::from(unc);
        let m_p = if adduct == 1 {
            u64::from(prec) - H_NET
        } else {
            u64::from(prec) + H_NET
        };
        let counts: Composition = [3, 8, 0, 1, 0, 0, 0, 0, 0, 0];
        for (delta, sign) in [(0i64, 1.0f32), (-5000, -1.0f32), (2500, 1.0f32)] {
            let m_c = (m_p as i64 + delta) as u64 as u32;
            let mut cand = pack_one_cand(&counts);
            cand[10] = m_c;
            let cand_ev = vec![2.0f32, 0.75, 4.0, 1.0];
            let meta = vec![0u32, prec, unc, adduct, 100, ppm_pre, 0, 0];
            let got = run_features_lane_device(&cand, &cand_ev, &meta, &log_host);
            let want = twin::formula_features(&cand, &cand_ev, &meta, &log_host, 1, 1);
            assert_f32_close(&got, &want, 1e-6, "adduct {adduct} delta {delta}");
            // Hand arithmetic for words 10/11.
            let d = delta.unsigned_abs() as f64;
            let hand = (d / w as f64) as f32;
            assert!((got[10] - hand).abs() <= 1e-6, "adduct {adduct} abs");
            if delta < 0 {
                assert!((got[11] + hand).abs() <= 1e-6, "adduct {adduct} signed");
            } else {
                assert!((got[11] - hand).abs() <= 1e-6, "adduct {adduct} signed");
                if delta == 0 {
                    assert!(got[11] == 0.0 && got[11].is_sign_positive(), "+0");
                }
            }
            let _ = sign;
        }
    }
    // w = 0 path (tolerance 0 and uncertainty 0 gives w = 1).
    {
        let prec = 2_000_000u32;
        let meta = vec![0u32, prec, 0, 1, 0, 0, 0, 0];
        let m_p = prec - 1_007_276u32;
        let counts: Composition = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let mut cand = pack_one_cand(&counts);
        cand[10] = m_p + 1;
        let cand_ev = vec![0.0f32, 0.0, 1.0, 1.0];
        let got = run_features_lane_device(&cand, &cand_ev, &meta, &log_host);
        let want = twin::formula_features(&cand, &cand_ev, &meta, &log_host, 1, 1);
        assert_f32_close(&got, &want, 1e-6, "w = 0");
        assert!((got[10] - 1.0).abs() <= 1e-6, "d = 1 over w = 1");
        assert!((got[11] - 1.0).abs() <= 1e-6);
    }
    // 4w clamp with a large w: abs is 4.
    {
        let prec = 200_000_000u32;
        let unc = 10_000_000u32;
        let ppm_pre = 100u32;
        let tol = tol_u64(prec, ppm_pre);
        let w = u64::from(tol) + u64::from(unc);
        let m_p = u64::from(prec) - H_NET;
        let counts: Composition = [3, 8, 0, 1, 0, 0, 0, 0, 0, 0];
        let mut cand = pack_one_cand(&counts);
        cand[10] = (m_p + 100_000_000) as u32;
        let cand_ev = vec![1.0f32, 0.25, 2.0, 0.0];
        let meta = vec![0u32, prec, unc, 1, 100, ppm_pre, 0, 0];
        let got = run_features_lane_device(&cand, &cand_ev, &meta, &log_host);
        assert!((got[10] - 4.0).abs() <= 1e-6, "clamped abs {}", got[10]);
        assert!((got[11] - 4.0).abs() <= 1e-6);
        let _ = w;
    }
    // Counts 0, 1 and 1023 read the table; bit equality below covers them.
    {
        let meta = vec![0u32, 200_000_000, 500, 1, 100, 100, 0, 0];
        let cand_ev = vec![0.0f32, 0.0, 0.0, 1.0];
        for count in [0u32, 1, 1023] {
            let mut comp = [0u16; 10];
            comp[0] = count.min(1023) as u16;
            // 1023 fits u16; build via raw words to avoid composition_mass overflow checks.
            let mut cand = vec![0u32; 13];
            cand[0] = count;
            cand[10] = 12_000_000u32.wrapping_mul(count);
            cand[11] = 1;
            cand[12] = u32::MAX;
            let got = run_features_lane_device(&cand, &cand_ev, &meta, &log_host);
            let hand = ((1.0 + count as f32).ln()).to_bits();
            assert_eq!(got[0].to_bits(), hand, "count {count}");
            let _ = comp;
        }
    }
    // Unknown uncertainty, unknown adduct and both parent overflows zero
    // the residual pair on the device, matching the twin and hand (0).
    {
        let prec = 200_000_000u32;
        let counts: Composition = [3, 8, 0, 1, 0, 0, 0, 0, 0, 0];
        let mut cand = pack_one_cand(&counts);
        cand[10] = 199_000_000;
        let cand_ev = vec![2.0f32, 0.75, 4.0, 1.0];
        let base = vec![0u32, prec, 500, 1, 100, 100, 0, 0];
        // Unknown uncertainty.
        let mut m1 = base.clone();
        m1[2] = u32::MAX;
        let g1 = run_features_lane_device(&cand, &cand_ev, &m1, &log_host);
        let w1 = twin::formula_features(&cand, &cand_ev, &m1, &log_host, 1, 1);
        assert_f32_close(&g1, &w1, 1e-6, "unknown unc");
        assert_eq!(g1[10], 0.0);
        assert_eq!(g1[11], 0.0);
        // Unknown adduct.
        let mut m2 = base.clone();
        m2[3] = 0;
        let g2 = run_features_lane_device(&cand, &cand_ev, &m2, &log_host);
        assert_eq!(g2[10], 0.0);
        assert_eq!(g2[11], 0.0);
        // Parent overflow both sides.
        let mut m3 = base.clone();
        m3[1] = 1000;
        let g3 = run_features_lane_device(&cand, &cand_ev, &m3, &log_host);
        assert_eq!(g3[10], 0.0);
        assert_eq!(g3[11], 0.0);
        let mut m4 = base;
        m4[3] = 2;
        m4[1] = u32::MAX - 100;
        let g4 = run_features_lane_device(&cand, &cand_ev, &m4, &log_host);
        assert_eq!(g4[10], 0.0);
        assert_eq!(g4[11], 0.0);
    }
    // Bit equality with count_features, asserted with to_bits().
    {
        let device = dev();
        let fx = ev_fixture();
        let cand_t = upload_ids(&fx.cand, vec![fx.b, fx.m, 13], &device);
        let ev_t = upload_f(&fx.want_ev, vec![fx.b, fx.m, 4], &device);
        let meta_t = upload_ids(&fx.meta, vec![fx.b, 8], &device);
        let log_t = upload_f(&log_host, vec![1024], &device);
        let mut out_t = upload_f(&vec![f32::NAN; fx.b * fx.m * 16], vec![fx.b, fx.m, 16], &device);
        kernels::formula_features(&cand_t, &ev_t, &meta_t, &log_t, &mut out_t).unwrap();
        check_launches(&device).unwrap();
        let got = out_t.to_f32();
        let rec_t = cand_t.reshape(vec![fx.b * fx.m, 13]).unwrap();
        let mut cf_t = upload_f(&vec![f32::NAN; fx.b * fx.m * 10], vec![fx.b * fx.m, 10], &device);
        ms2::count_features(&rec_t, &log_t, &mut cf_t, 13).unwrap();
        check_launches(&device).unwrap();
        let cf = cf_t.to_f32();
        for r in 0..fx.b * fx.m {
            for e in 0..10 {
                assert_eq!(
                    got[r * 16 + e].to_bits(),
                    cf[r * 10 + e].to_bits(),
                    "row {r} word {e} bit-equal"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Launch counts and error variants, in one test (process-global counters).
// ---------------------------------------------------------------------------

#[test]
fn formula_evidence_launch_counts_and_errors() {
    let _guard = serial();
    let device = dev();
    let fx = ev_fixture();
    let kept_t = upload_ids(&fx.kept, vec![fx.b, fx.n, 3], &device);
    let kept_f_t = upload_f(&fx.kept_f, vec![fx.b, fx.n, 2], &device);
    let meta_t = upload_ids(&fx.meta, vec![fx.b, 8], &device);
    let spec_t = upload_ids(&fx.spec, vec![fx.b, 2], &device);
    let cand_t = upload_ids(&fx.cand, vec![fx.b, fx.m, 13], &device);
    let peaks_t = upload_ids(&fx.ev_peaks, vec![fx.b, fx.p, 4], &device);
    let w_t = upload_f(&fx.ev_w, vec![fx.b, fx.p], &device);
    let log_host = mamba3::models::ms2::twin::log_table();
    let log_t = upload_f(&log_host, vec![1024], &device);
    let mut peaks_o = upload_ids(&vec![u32::MAX; fx.b * fx.p * 4], vec![fx.b, fx.p, 4], &device);
    let mut w_o = upload_f(&vec![f32::NAN; fx.b * fx.p], vec![fx.b, fx.p], &device);
    let mut ev_o = upload_f(
        &vec![f32::NAN; fx.b * fx.m * 4],
        vec![fx.b, fx.m, 4],
        &device,
    );
    let mut feat_o = upload_f(
        &vec![f32::NAN; fx.b * fx.m * 16],
        vec![fx.b, fx.m, 16],
        &device,
    );
    reset_launch_count();
    let delta = |before: usize| launch_count() - before;
    let mut before = launch_count();
    kernels::evidence_peaks(&kept_t, &kept_f_t, &meta_t, &spec_t, &mut peaks_o, &mut w_o).unwrap();
    assert_eq!(delta(before), 1, "evidence_peaks launches");
    before = launch_count();
    kernels::formula_features(&cand_t, &ev_o, &meta_t, &log_t, &mut feat_o).unwrap();
    assert_eq!(delta(before), 1, "formula_features launches");
    // 6 lanes at per_lane hydrogen trials per lane: huge budget is one launch.
    // True bounds (20, 2000): per_lane = 512 * 5 * 2 = 5120.
    let (h_cap_max, tol_max) = (20u32, 2000u32);
    let per_lane = u64::from(fx.work_max) * fx.p as u64 * trials_bound_of(h_cap_max, tol_max);
    assert_eq!(per_lane, 5120, "hand per-lane bound");
    before = launch_count();
    kernels::formula_evidence(
        &cand_t, &peaks_t, &w_t, &meta_t, &spec_t, &mut ev_o, fx.work_max, u64::MAX, h_cap_max,
        tol_max,
    )
    .unwrap();
    assert_eq!(delta(before), 1, "formula_evidence single launch");
    // 2 lanes per launch: 3 launches.
    before = launch_count();
    kernels::formula_evidence(
        &cand_t, &peaks_t, &w_t, &meta_t, &spec_t, &mut ev_o, fx.work_max, 2 * per_lane,
        h_cap_max, tol_max,
    )
    .unwrap();
    assert_eq!(delta(before), 3, "formula_evidence 2-lane chunks");
    // 1 lane per launch: 6 launches.
    before = launch_count();
    kernels::formula_evidence(
        &cand_t, &peaks_t, &w_t, &meta_t, &spec_t, &mut ev_o, fx.work_max, per_lane,
        h_cap_max, tol_max,
    )
    .unwrap();
    assert_eq!(delta(before), 6, "formula_evidence 1-lane chunks");
    check_launches(&device).unwrap();
    // Shape errors return Error::Shape before any launch.
    before = launch_count();
    let bad_kept_f = upload_f(&vec![0.0; fx.b * fx.n], vec![fx.b, fx.n, 1], &device);
    assert!(matches!(
        kernels::evidence_peaks(&kept_t, &bad_kept_f, &meta_t, &spec_t, &mut peaks_o, &mut w_o),
        Err(Error::Shape(_))
    ));
    assert_eq!(delta(before), 0, "evidence_peaks shape error launches nothing");
    before = launch_count();
    let bad_cand = upload_ids(&vec![0u32; fx.b * fx.m * 12], vec![fx.b, fx.m, 12], &device);
    assert!(matches!(
        kernels::formula_evidence(
            &bad_cand, &peaks_t, &w_t, &meta_t, &spec_t, &mut ev_o, fx.work_max, u64::MAX,
            h_cap_max, tol_max,
        ),
        Err(Error::Shape(_))
    ));
    assert_eq!(delta(before), 0, "formula_evidence shape error launches nothing");
    before = launch_count();
    let mut bad_feat = upload_f(&vec![0.0; fx.b * fx.m * 10], vec![fx.b, fx.m, 10], &device);
    assert!(matches!(
        kernels::formula_features(&cand_t, &ev_o, &meta_t, &log_t, &mut bad_feat),
        Err(Error::Shape(_))
    ));
    assert_eq!(delta(before), 0, "formula_features shape error launches nothing");
    // Zero budgets are Error::Config before any launch.
    before = launch_count();
    assert!(matches!(
        kernels::formula_evidence(
            &cand_t, &peaks_t, &w_t, &meta_t, &spec_t, &mut ev_o, 0, u64::MAX, h_cap_max,
            tol_max,
        ),
        Err(Error::Config(_))
    ));
    assert!(matches!(
        kernels::formula_evidence(&cand_t, &peaks_t, &w_t, &meta_t, &spec_t, &mut ev_o, fx.work_max, 0, h_cap_max, tol_max),
        Err(Error::Config(_))
    ));
    assert_eq!(delta(before), 0, "config errors launch nothing");
}

#[test]
fn formula_evidence_dispatch_degenerate_and_tail() {
    let _guard = serial();
    let device = dev();
    let fx = ev_fixture();
    let delta = |before: usize| launch_count() - before;
    // B = 0 returns Ok with zero launches (kernel 1).
    {
        let before = launch_count();
        let kept = upload_ids(&[], vec![0, fx.n, 3], &device);
        let kept_f = upload_f(&[], vec![0, fx.n, 2], &device);
        let meta = upload_ids(&[], vec![0, 8], &device);
        let spec = upload_ids(&[], vec![0, 2], &device);
        let mut peaks = upload_ids(&[], vec![0, fx.p, 4], &device);
        let mut w = upload_f(&[], vec![0, fx.p], &device);
        kernels::evidence_peaks(&kept, &kept_f, &meta, &spec, &mut peaks, &mut w).unwrap();
        assert_eq!(delta(before), 0, "B = 0 launches nothing");
    }
    // B = 0 and M = 0 return Ok with zero launches (kernels 2 and 3).
    {
        let before = launch_count();
        let cand = upload_ids(&[], vec![0, fx.m, 13], &device);
        let peaks = upload_ids(&[], vec![0, fx.p, 4], &device);
        let w = upload_f(&[], vec![0, fx.p], &device);
        let meta = upload_ids(&[], vec![0, 8], &device);
        let spec = upload_ids(&[], vec![0, 2], &device);
        let mut ev = upload_f(&[], vec![0, fx.m, 4], &device);
        kernels::formula_evidence(&cand, &peaks, &w, &meta, &spec, &mut ev, fx.work_max, u64::MAX, u32::MAX, 1_007_825)
            .unwrap();
        assert_eq!(delta(before), 0, "B = 0 launches nothing");
        let cand0 = upload_ids(&vec![0u32; fx.b * 13], vec![fx.b, 1, 13], &device);
        let _ = cand0;
        // M = 0 with B = 1.
        let cand_m0 = upload_ids(&[], vec![1, 0, 13], &device);
        let peaks_m0 = upload_ids(&vec![0u32; fx.p * 4], vec![1, fx.p, 4], &device);
        let w_m0 = upload_f(&vec![0.0; fx.p], vec![1, fx.p], &device);
        let meta_m0 = upload_ids(&vec![0u32; 8], vec![1, 8], &device);
        let spec_m0 = upload_ids(&vec![0u32; 2], vec![1, 2], &device);
        let mut ev_m0 = upload_f(&[], vec![1, 0, 4], &device);
        kernels::formula_evidence(
            &cand_m0, &peaks_m0, &w_m0, &meta_m0, &spec_m0, &mut ev_m0, fx.work_max, u64::MAX,
            u32::MAX, 1_007_825,
        )
        .unwrap();
        assert_eq!(delta(before), 0, "M = 0 launches nothing");
        let log_host = mamba3::models::ms2::twin::log_table();
        let log_t = upload_f(&log_host, vec![1024], &device);
        let mut out_m0 = upload_f(&[], vec![1, 0, 16], &device);
        kernels::formula_features(&cand_m0, &ev_m0, &meta_m0, &log_t, &mut out_m0).unwrap();
        assert_eq!(delta(before), 0, "features M = 0 launches nothing");
    }
    // Refused shapes return Error::Shape before any launch.
    {
        // evidence_peaks: N == 0.
        let before = launch_count();
        let kept = upload_ids(&[], vec![1, 0, 3], &device);
        let kept_f = upload_f(&[], vec![1, 0, 2], &device);
        let meta = upload_ids(&vec![0u32; 8], vec![1, 8], &device);
        let spec = upload_ids(&vec![0u32; 2], vec![1, 2], &device);
        let mut peaks = upload_ids(&vec![u32::MAX; fx.p * 4], vec![1, fx.p, 4], &device);
        let mut w = upload_f(&vec![0.0; fx.p], vec![1, fx.p], &device);
        assert!(matches!(
            kernels::evidence_peaks(&kept, &kept_f, &meta, &spec, &mut peaks, &mut w),
            Err(Error::Shape(_))
        ));
        assert_eq!(delta(before), 0, "N == 0 launches nothing");
        // evidence_peaks: P == 0.
        let before = launch_count();
        let kept_n = upload_ids(&vec![0u32; fx.n * 3], vec![1, fx.n, 3], &device);
        let kept_fn = upload_f(&vec![0.0; fx.n * 2], vec![1, fx.n, 2], &device);
        let mut peaks0 = upload_ids(&[], vec![1, 0, 4], &device);
        let mut w0 = upload_f(&[], vec![1, 0], &device);
        assert!(matches!(
            kernels::evidence_peaks(&kept_n, &kept_fn, &meta, &spec, &mut peaks0, &mut w0),
            Err(Error::Shape(_))
        ));
        assert_eq!(delta(before), 0, "P == 0 launches nothing");
        // evidence_peaks: P == 33.
        let before = launch_count();
        let mut peaks33 = upload_ids(&vec![u32::MAX; 33 * 4], vec![1, 33, 4], &device);
        let mut w33 = upload_f(&vec![0.0; 33], vec![1, 33], &device);
        assert!(matches!(
            kernels::evidence_peaks(&kept_n, &kept_fn, &meta, &spec, &mut peaks33, &mut w33),
            Err(Error::Shape(_))
        ));
        assert_eq!(delta(before), 0, "P == 33 launches nothing");
        // formula_evidence: P == 0 and P == 33.
        let cand_t = upload_ids(&fx.cand, vec![fx.b, fx.m, 13], &device);
        let meta_t = upload_ids(&fx.meta, vec![fx.b, 8], &device);
        let spec_t = upload_ids(&fx.spec, vec![fx.b, 2], &device);
        let mut ev_o =
            upload_f(&vec![f32::NAN; fx.b * fx.m * 4], vec![fx.b, fx.m, 4], &device);
        let before = launch_count();
        let peaks0 = upload_ids(&[], vec![fx.b, 0, 4], &device);
        let w0 = upload_f(&[], vec![fx.b, 0], &device);
        assert!(matches!(
            kernels::formula_evidence(
                &cand_t, &peaks0, &w0, &meta_t, &spec_t, &mut ev_o, fx.work_max, u64::MAX,
                u32::MAX, 1_007_825,
            ),
            Err(Error::Shape(_))
        ));
        assert_eq!(delta(before), 0, "evidence P == 0 launches nothing");
        let before = launch_count();
        let peaks33 = upload_ids(&vec![0u32; fx.b * 33 * 4], vec![fx.b, 33, 4], &device);
        let w33 = upload_f(&vec![0.0; fx.b * 33], vec![fx.b, 33], &device);
        assert!(matches!(
            kernels::formula_evidence(
                &cand_t, &peaks33, &w33, &meta_t, &spec_t, &mut ev_o, fx.work_max, u64::MAX,
                u32::MAX, 1_007_825,
            ),
            Err(Error::Shape(_))
        ));
        assert_eq!(delta(before), 0, "evidence P == 33 launches nothing");
    }
    // Tail launch count: 6 lanes at 4 lanes per launch is 2 launches.
    {
        let cand_t = upload_ids(&fx.cand, vec![fx.b, fx.m, 13], &device);
        let peaks_t = upload_ids(&fx.ev_peaks, vec![fx.b, fx.p, 4], &device);
        let w_t = upload_f(&fx.ev_w, vec![fx.b, fx.p], &device);
        let meta_t = upload_ids(&fx.meta, vec![fx.b, 8], &device);
        let spec_t = upload_ids(&fx.spec, vec![fx.b, 2], &device);
        let mut ev_o =
            upload_f(&vec![f32::NAN; fx.b * fx.m * 4], vec![fx.b, fx.m, 4], &device);
        let before = launch_count();
        let (h_cap_max, tol_max) = (20u32, 2000u32);
        let per_lane = u64::from(fx.work_max) * fx.p as u64 * trials_bound_of(h_cap_max, tol_max);
        kernels::formula_evidence(
            &cand_t, &peaks_t, &w_t, &meta_t, &spec_t, &mut ev_o, fx.work_max, 4 * per_lane,
            h_cap_max, tol_max,
        )
        .unwrap();
        // 6 lanes, 4 per launch: chunks of 4 + 2, so 2 launches.
        assert_eq!(delta(before), 2, "tail launch count");
        check_launches(&device).unwrap();
    }
}

// ---------------------------------------------------------------------------
// E4: the explained-peak walk without the carbon digit, on device.
// ---------------------------------------------------------------------------

/// Peak target and fragment tolerance of a peak m/z (independent `u64`
/// arithmetic).
fn peak_row_e4(mz: u32, adduct: u16, ppm: u32) -> (u32, u32) {
    let t = if adduct == 1 {
        mz.checked_add(549).expect("target fits")
    } else {
        mz.checked_sub(549).expect("target fits")
    };
    (t, tol_u64(mz, ppm))
}

/// One P = 1 evidence lane on the device and through both host twins.
fn run_e4_pair(
    cand: &[u32],
    t: u32,
    tol: u32,
    adduct: u16,
    u: u32,
    work_max: u32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let ev_peaks = vec![0, t, tol, 1];
    let ev_w = vec![1.0f32];
    let meta = vec![1u32, 0, 0, u32::from(adduct), 100, 0, 0, 0];
    let spec = vec![u, 0];
    let got = run_evidence_lane_device(cand, &ev_peaks, &ev_w, &meta, &spec, 1, work_max);
    let mut fast = vec![0.0f32; 4];
    twin::formula_evidence_lane(cand, &ev_peaks, &ev_w, &meta, &spec, 0, 0, 1, 1, work_max, u32::MAX, &mut fast);
    let mut slow = vec![0.0f32; 4];
    twin::formula_evidence_lane_slow(
        cand, &ev_peaks, &ev_w, &meta, &spec, 0, 0, 1, 1, work_max, u32::MAX, &mut slow,
    );
    (got, fast, slow)
}

/// E4 random peaks on device: every fixture family of the host
/// exhaustive/fast/slow/budget tests through the kernel on poisoned
/// outputs against both twins (integers exact, floats within 1e-6).
#[test]
fn formula_evidence_device_exhaustive_matches_twins() {
    let _guard = serial();
    let ppm = 100u32;
    let mut rng = StdRng::seed_from_u64(0xDE9);
    let work_max = 1u32 << 20;
    let mut compared = 0usize;
    let mut explained = 0usize;
    let mut unexplained = 0usize;
    for _ in 0..24 {
        let mut parent: Composition = [0; 10];
        parent[0] = rng.random_range(0..=12);
        parent[HYDROGEN] = rng.random_range(0..=12);
        parent[2] = rng.random_range(0..=4);
        parent[3] = rng.random_range(0..=5);
        parent[4] = rng.random_range(0..=2);
        parent[5] = rng.random_range(0..=1);
        parent[6] = rng.random_range(0..=2);
        parent[7] = rng.random_range(0..=2);
        if parent.iter().all(|&n| n == 0) {
            parent[0] = 1;
        }
        let cand = pack_one_cand(&parent);
        for adduct in [1u16, 2] {
            for u in [0u32, 40] {
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
                    let tol = tol_u64(mz, ppm) as i64;
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
                    let (t, tol) = peak_row_e4(mz, adduct, ppm);
                    let (got, fast, slow) = run_e4_pair(&cand, t, tol, adduct, u, work_max);
                    assert_f32_close(&got, &fast, 1e-6, "device vs fast twin peak {mz}");
                    assert_f32_close(&got, &slow, 1e-6, "device vs slow twin peak {mz}");
                    assert_eq!(fast, slow, "fast vs slow peak {mz}");
                    assert_eq!(fast[3], 1.0, "complete walk");
                    compared += 1;
                    if fast[0] == 1.0 {
                        explained += 1;
                    } else {
                        unexplained += 1;
                    }
                }
                // One P = 8 spectrum per combo: count and weight.
                let p = 8usize;
                let mut ev_peaks = vec![u32::MAX, 0, 0, 0];
                ev_peaks = ev_peaks.repeat(p);
                let mut ev_w = vec![0.0f32; p];
                for (s, &mz) in mzs.iter().take(p).enumerate() {
                    let (t, tol) = peak_row_e4(mz, adduct, ppm);
                    ev_peaks[s * 4] = s as u32;
                    ev_peaks[s * 4 + 1] = t;
                    ev_peaks[s * 4 + 2] = tol;
                    ev_peaks[s * 4 + 3] = 1;
                    ev_w[s] = (s + 1) as f32;
                }
                let meta = vec![1u32, 0, 0, u32::from(adduct), ppm, 0, 0, 0];
                let spec = vec![u, 0];
                let device = dev();
                let cand_t = upload_ids(&cand, vec![1, 1, 13], &device);
                let peaks_t = upload_ids(&ev_peaks, vec![1, p, 4], &device);
                let w_t = upload_f(&ev_w, vec![1, p], &device);
                let meta_t = upload_ids(&meta, vec![1, 8], &device);
                let spec_t = upload_ids(&spec, vec![1, 2], &device);
                let mut out_t = upload_f(&vec![f32::NAN; 4], vec![1, 1, 4], &device);
                kernels::formula_evidence(
                    &cand_t, &peaks_t, &w_t, &meta_t, &spec_t, &mut out_t, work_max, u64::MAX,
                    u32::MAX, 1_007_825,
                )
                .unwrap();
                check_launches(&device).unwrap();
                let got = out_t.to_f32();
                let want = twin::formula_evidence(&cand, &ev_peaks, &ev_w, &meta, &spec, 1, 1, p, work_max, u32::MAX);
                assert_f32_close(&got, &want, 1e-6, "P = 8 device vs twin");
                compared += 1;
            }
        }
    }
    println!("E4-DEVICE-EXHAUSTIVE compared={compared} explained={explained} unexplained={unexplained}");
    assert!(compared >= 800, "only {compared} device pairs ran");
    assert!(explained >= 50, "only {explained} explained pairs");
    assert!(unexplained >= 50, "only {unexplained} unexplained pairs");
}

/// E4 adversarial ranges on device: tolerance edges of the fast
/// precondition, `h_cap` boundaries, `Rm` edges, `t + tol < m'` and `m'`
/// near `u32::MAX` through the kernel against both twins.
#[test]
fn formula_evidence_device_fast_slow_adversarial() {
    let _guard = serial();
    let ppm = 100u32;
    let work_max = 1u32 << 20;
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
    let mut cases: Vec<(Vec<u32>, u32, u32, u16, u32)> = Vec::new();
    let c4h6: Composition = [4, 6, 0, 0, 0, 0, 0, 0, 0, 0];
    let t_exact = 4 * 12_000_000 + 6 * 1_007_825;
    for tol in [0u32, 100, 464_787, 464_788] {
        cases.push((words_of(&c4h6), t_exact, tol, 1, 0));
    }
    for h in [96u16, 124, 125] {
        let comp: Composition = [2, h, 0, 0, 0, 0, 0, 0, 0, 0];
        let t = 2 * 12_000_000 + 90 * 1_007_825;
        cases.push((words_of(&comp), t, 100, 1, 0));
    }
    let c2h124: Composition = [2, 124, 0, 0, 0, 0, 0, 0, 0, 0];
    let t127 = 2 * 12_000_000 + 60 * 1_007_825;
    cases.push((words_of(&c2h124), t127, 3_112, 1, 0));
    cases.push((words_of(&c2h124), t127, 3_113, 1, 0));
    let c2h61: Composition = [2, 61, 0, 0, 0, 0, 0, 0, 0, 0];
    cases.push((words_of(&c2h61), t127, 249_599, 1, 0));
    cases.push((words_of(&c2h61), t127, 249_600, 1, 0));
    for h_word in [u32::MAX - 2, u32::MAX - 1] {
        let mut w = vec![0u32; 13];
        w[0] = 2;
        w[1] = h_word;
        w[10] = 24_000_000;
        w[11] = 1;
        w[12] = u32::MAX;
        cases.push((w, 24_000_000, 100, 1, 0));
    }
    let c1h2: Composition = [1, 2, 0, 0, 0, 0, 0, 0, 0, 0];
    for t in [999_950u32, 999_949, 15_600, 15_599, 15_601, 7_775, 7_774, 7_776] {
        cases.push((words_of(&c1h2), t, 50, 1, 0));
    }
    let s1: Composition = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0];
    let (t_s, tol_s) = peak_row_e4(31_972_071 - 549, 1, ppm);
    cases.push((words_of(&s1), t_s, tol_s, 1, 0));
    cases.push((words_of(&s1), 1_000_000, 100, 1, 0));
    let i33: Composition = [0, 0, 0, 0, 0, 0, 0, 0, 0, 33];
    let (t_i, tol_i) = peak_row_e4(4_187_847_576 - 549, 1, ppm);
    cases.push((words_of(&i33), t_i, tol_i, 1, 0));
    cases.push((words_of(&i33), u32::MAX - 100, 50, 1, 0));
    let mut compared = 0usize;
    for (w, t, tol, adduct, u) in cases.iter() {
        let (got, fast, slow) = run_e4_pair(w, *t, *tol, *adduct, *u, work_max);
        assert_f32_close(&got, &fast, 1e-6, "t = {t} tol = {tol} device vs fast");
        assert_f32_close(&got, &slow, 1e-6, "t = {t} tol = {tol} device vs slow");
        compared += 1;
    }
    println!("E4-DEVICE-ADVERSARIAL compared={compared}");
}

/// E4 budget fixtures on device: truncated, boundary, carbon-only and
/// empty candidates against both twins and the `ion_assign` prefix oracle.
#[test]
fn formula_evidence_device_budget_prefix_matches_oracle() {
    let _guard = serial();
    let ppm = 100u32;
    let mut compared = 0usize;
    // N2 O1 H4 (J = 6, c[C] = 0) at W = 2, 5, 6.
    let comp: Composition = [0, 4, 2, 1, 0, 0, 0, 0, 0, 0];
    let cand = pack_one_cand(&comp);
    let t_a = 44_001_063u32;
    let t_b = 14_003_074u32;
    for (w, ta_expl, tb_expl) in [(2u32, false, true), (5, false, true), (6, true, true)] {
        for (t, want) in [(t_a, ta_expl), (t_b, tb_expl)] {
            let mz = t - 549;
            let tol = tol_u64(mz, ppm);
            let (got, fast, slow) = run_e4_pair(&cand, t, tol, 1, 0, w);
            assert_f32_close(&got, &fast, 1e-6, "W = {w} t = {t} device vs fast");
            assert_eq!(fast, slow, "W = {w} t = {t} fast vs slow");
            assert_eq!(got[0] == 1.0, want, "W = {w} t = {t} hand");
            assert_eq!(got[3] == 1.0, 6 <= w as u64, "W = {w} complete");
            let v = 6u64.min(w as u64);
            let oracle_w = (v - 1) as u32;
            let assign = ion_assign(
                &comp, 1, mz, 0, ppm,
                &IonLimits { work_max: oracle_w, kept: 4 },
            )
            .expect("ion_assign runs");
            assert_eq!(got[0] == 1.0, assign.accepted >= 1, "W = {w} t = {t} oracle");
            compared += 1;
        }
    }
    // C4 H2 carbon-only (J = 1): n = 1..=4 explain at W = 1.
    let c4: Composition = [4, 2, 0, 0, 0, 0, 0, 0, 0, 0];
    let cand4 = pack_one_cand(&c4);
    for n in 1u32..=5 {
        let t = n * 12_000_000;
        let tol = tol_u64(t - 549, ppm);
        let (got, fast, slow) = run_e4_pair(&cand4, t, tol, 1, 50, 1);
        assert_f32_close(&got, &fast, 1e-6, "C4 n = {n} device vs fast");
        assert_eq!(fast, slow, "C4 n = {n} fast vs slow");
        assert_eq!(got, vec![if n <= 4 { 1.0 } else { 0.0 }, if n <= 4 { 1.0 } else { 0.0 }, 1.0, 1.0]);
        compared += 1;
    }
    // Flag-1 candidate with no heavy atom: nothing explained, complete.
    let empty = pack_one_cand(&[0; 10]);
    let (got, fast, slow) = run_e4_pair(&empty, 12_000_000, 119, 1, 50, 4096);
    assert_f32_close(&got, &fast, 1e-6, "empty device vs fast");
    assert_eq!(fast, slow, "empty fast vs slow");
    assert_eq!(&got[..], &[0.0, 0.0, 1.0, 1.0][..], "empty hand");
    compared += 1;
    println!("E4-DEVICE-BUDGET compared={compared}");
}

// ---------------------------------------------------------------------------
// E4F item 2: dispatch sizing counts hydrogen trials; the kernel enforces
// the host bound by clamping. Every device call runs on poisoned outputs
// against the twin.
// ---------------------------------------------------------------------------

/// Run kernel 2 on the device for one `(b, m)` lane with explicit hydrogen
/// bounds.
fn run_evidence_lane_device_bounds(
    cand: &[u32],
    ev_peaks: &[u32],
    ev_w: &[f32],
    meta: &[u32],
    spec: &[u32],
    p: usize,
    work_max: u32,
    h_cap_max: u32,
    tol_max: u32,
) -> Vec<f32> {
    let device = dev();
    let cand_t = upload_ids(cand, vec![1, 1, 13], &device);
    let peaks_t = upload_ids(ev_peaks, vec![1, p, 4], &device);
    let w_t = upload_f(ev_w, vec![1, p], &device);
    let meta_t = upload_ids(meta, vec![1, 8], &device);
    let spec_t = upload_ids(spec, vec![1, 2], &device);
    let mut out_t = upload_f(&vec![f32::NAN; 4], vec![1, 1, 4], &device);
    kernels::formula_evidence(
        &cand_t, &peaks_t, &w_t, &meta_t, &spec_t, &mut out_t, work_max, u64::MAX, h_cap_max,
        tol_max,
    )
    .unwrap();
    check_launches(&device).unwrap();
    out_t.to_f32()
}

/// E4F launch counts for given (`h_cap_max`, `tol_max`, `dispatch_max`) by
/// the item-2 formula: `trials_bound = min(h_cap_max + 1, (s_max + 1) * (2
/// tol_max / 7,825 + 2))`, `per_lane = work_max * P * trials_bound`,
/// `lanes_per_launch = max(1, dispatch_max / per_lane)`,
/// `launches = ceil(lanes / lanes_per_launch)` (6 lanes here).
#[test]
fn e4f_dispatch_launch_counts_by_formula() {
    let _guard = serial();
    let device = dev();
    let fx = ev_fixture();
    let lanes = (fx.b * fx.m) as u64;
    assert_eq!(lanes, 6);
    let launches_for = |h_cap_max: u32, tol_max: u32, dispatch_max: u64| -> usize {
        let cand_t = upload_ids(&fx.cand, vec![fx.b, fx.m, 13], &device);
        let peaks_t = upload_ids(&fx.ev_peaks, vec![fx.b, fx.p, 4], &device);
        let w_t = upload_f(&fx.ev_w, vec![fx.b, fx.p], &device);
        let meta_t = upload_ids(&fx.meta, vec![fx.b, 8], &device);
        let spec_t = upload_ids(&fx.spec, vec![fx.b, 2], &device);
        let mut ev_o = upload_f(
            &vec![f32::NAN; fx.b * fx.m * 4],
            vec![fx.b, fx.m, 4],
            &device,
        );
        let before = launch_count();
        kernels::formula_evidence(
            &cand_t, &peaks_t, &w_t, &meta_t, &spec_t, &mut ev_o, fx.work_max, dispatch_max,
            h_cap_max, tol_max,
        )
        .unwrap();
        check_launches(&device).unwrap();
        launch_count() - before
    };
    // (20, 2000): trials_bound = min(21, 1 * 2) = 2, per_lane = 5120.
    let (h_cap_max, tol_max) = (20u32, 2000u32);
    assert_eq!(trials_bound_of(h_cap_max, tol_max), 2);
    let per_lane = u64::from(fx.work_max) * fx.p as u64 * 2;
    assert_eq!(per_lane, 5120);
    assert_eq!(launches_for(h_cap_max, tol_max, 6 * per_lane), 1, "exact fit");
    assert_eq!(launches_for(h_cap_max, tol_max, 6 * per_lane - 1), 2, "one short");
    assert_eq!(launches_for(h_cap_max, tol_max, 2 * per_lane), 3, "2 lanes/launch");
    assert_eq!(launches_for(h_cap_max, tol_max, per_lane), 6, "1 lane/launch");
    assert_eq!(launches_for(h_cap_max, tol_max, per_lane - 1), 6, "below one lane");
    // (0, 0): trials_bound = min(1, 1 * 2) = 1, per_lane = 2560.
    assert_eq!(trials_bound_of(0, 0), 1);
    assert_eq!(launches_for(0, 0, u64::MAX), 1, "huge budget, tiny trials");
    assert_eq!(launches_for(0, 0, 2560), 6, "per_lane 2560");
    // Saturated bounds: trials_bound = min(2^32, (s_max + 1) * per_s) with
    // s_max = 33,608,121, per_s = 259: the wrapped product is ~8.7e12, so
    // trials_bound = 2^32 and per_lane = 2560 * 2^32 = 10,995,116,277,760.
    assert_eq!(trials_bound_of(u32::MAX, 1_007_825), 4_294_967_296);
    let big_per_lane = 2560u64 * 4_294_967_296;
    assert_eq!(big_per_lane, 10_995_116_277_760);
    assert_eq!(
        launches_for(u32::MAX, 1_007_825, u64::MAX),
        1,
        "u64::MAX still fits 6 lanes"
    );
    assert_eq!(
        launches_for(u32::MAX, 1_007_825, big_per_lane - 1),
        6,
        "below one per-lane bound serialises"
    );
    // Chunked equals unchunked under a true bound.
    let single = run_formula_evidence(&fx, u64::MAX, h_cap_max, tol_max);
    let chunked = run_formula_evidence(&fx, 2 * per_lane, h_cap_max, tol_max);
    assert_eq!(chunked, single, "chunked equals unchunked");
}

/// E4F kernel clamp: a deliberately small `h_cap_max` cuts the hydrogen
/// range on the device exactly like the twin (C1 H10, peak at `h = 10`);
/// a true bound reproduces the unclamped row; chunked equals unchunked.
#[test]
fn e4f_kernel_clamp_cuts_range() {
    let _guard = serial();
    let comp: Composition = [1, 10, 0, 0, 0, 0, 0, 0, 0, 0];
    let cand = pack_one_cand(&comp);
    let t = 12_000_000u32 + 10 * 1_007_825;
    let mz = t - 549;
    let tol = tol_u64(mz, 100);
    let ev = vec![0, t, tol, 1];
    let meta = vec![1u32, 0, 0, 1, 100, 0, 0, 0];
    let spec = vec![0u32, 0];
    let want_cut = twin::formula_evidence(&cand, &ev, &[1.0], &meta, &spec, 1, 1, 1, 4096, 5);
    assert_eq!(&want_cut[..], &[0.0, 0.0, 1.0, 1.0][..], "twin cut");
    let got_cut = run_evidence_lane_device_bounds(&cand, &ev, &[1.0], &meta, &spec, 1, 4096, 5, tol);
    assert_f32_close(&got_cut, &want_cut, 1e-6, "device cut matches twin cut");
    let want_full = twin::formula_evidence(&cand, &ev, &[1.0], &meta, &spec, 1, 1, 1, 4096, 13);
    assert_eq!(&want_full[..], &[1.0, 1.0, 1.0, 1.0][..], "twin true bound");
    let got_full = run_evidence_lane_device_bounds(&cand, &ev, &[1.0], &meta, &spec, 1, 4096, 13, tol);
    assert_f32_close(&got_full, &want_full, 1e-6, "device true bound");
    // Chunked equals unchunked under the true bound: two identical lanes,
    // one launch at u64::MAX versus one lane per launch.
    let device = dev();
    let trials = trials_bound_of(13, tol);
    let per_lane = 4096u64 * 1 * trials;
    let mut cand2 = cand.clone();
    cand2.extend_from_slice(&cand);
    let cand_t = upload_ids(&cand2, vec![1, 2, 13], &device);
    let peaks_t = upload_ids(&ev, vec![1, 1, 4], &device);
    let w_t = upload_f(&[1.0], vec![1, 1], &device);
    let meta_t = upload_ids(&meta, vec![1, 8], &device);
    let spec_t = upload_ids(&spec, vec![1, 2], &device);
    let run_two = |dispatch: u64| {
        let mut out = upload_f(&vec![f32::NAN; 8], vec![1, 2, 4], &device);
        kernels::formula_evidence(
            &cand_t, &peaks_t, &w_t, &meta_t, &spec_t, &mut out, 4096, dispatch, 13, tol,
        )
        .unwrap();
        check_launches(&device).unwrap();
        out.to_f32()
    };
    let before = launch_count();
    let single = run_two(u64::MAX);
    assert_eq!(launch_count() - before, 1, "single launch");
    let before = launch_count();
    let chunked = run_two(per_lane);
    assert_eq!(launch_count() - before, 2, "two launches");
    assert_eq!(chunked, single, "chunked clamp equals unchunked");
    assert_eq!(&single[0..4], &want_full[..], "lane 0");
    assert_eq!(&single[4..8], &want_full[..], "lane 1");
}

/// E4F device coverage of the new fixture families: many-hydrogen parents,
/// wrap-boundary peaks, 3,912/3,913 endpoints, the past-MAX I33 edge,
/// hydrogen-only and saturated-cap raw words — kernel on poisoned outputs
/// against fast and slow twins.
#[test]
fn e4f_device_new_families_match_twins() {
    let _guard = serial();
    let mut rng = StdRng::seed_from_u64(0xE4FDu64);
    let work_max = 1u32 << 20;
    let mut compared = 0usize;
    // Many-hydrogen sample: 12 parents, H in 120..=400, small heavies.
    for _ in 0..12 {
        let mut parent: Composition = [0; 10];
        parent[0] = rng.random_range(1..=6);
        parent[HYDROGEN] = rng.random_range(120..=400);
        parent[2] = rng.random_range(0..=2);
        parent[3] = rng.random_range(0..=2);
        let cand = pack_one_cand(&parent);
        for adduct in [1u16, 2] {
            let h_cap = u32::from(parent[HYDROGEN]) + if adduct == 1 { 3 } else { 2 };
            let h = rng.random_range(0..=h_cap.min(400));
            let mass = u64::from(parent[0]) * 12_000_000
                + u64::from(parent[2]) * 14_003_074
                + u64::from(parent[3]) * 15_994_915
                + u64::from(h) * 1_007_825;
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
            for ppm in [100u32, 1000] {
                let tol = tol_u64(mz, ppm);
                let (t, _) = peak_row_e4(mz, adduct, ppm);
                let (got, fast, slow) = run_e4_pair(&cand, t, tol, adduct, 0, work_max);
                assert_f32_close(&got, &fast, 1e-6, "many-H device vs fast");
                assert_eq!(fast, slow, "many-H fast vs slow");
                compared += 1;
            }
        }
    }
    // Wrap-boundary and endpoint fixtures (hand values from the host tests).
    let comp6: Composition = [6, 124, 0, 0, 0, 0, 0, 0, 0, 0];
    let cand6 = pack_one_cand(&comp6);
    for (t, tol) in [(139_993_701u32, 6_299u32), (139_993_700u32, 6_299u32)] {
        let (got, fast, slow) = run_e4_pair(&cand6, t, tol, 1, 0, 4096);
        assert_f32_close(&got, &fast, 1e-6, "wrap edge device vs fast");
        assert_eq!(fast, slow);
        assert_eq!(fast[0], 1.0, "wrap edge accepts");
        compared += 1;
    }
    let comp20: Composition = [6, 20, 0, 0, 0, 0, 0, 0, 0, 0];
    let cand20 = pack_one_cand(&comp20);
    for (t, tol) in [(83_082_163u32, 3_913u32), (89_136_937u32, 3_913u32)] {
        let (got, fast, slow) = run_e4_pair(&cand20, t, tol, 1, 0, 4096);
        assert_f32_close(&got, &fast, 1e-6, "endpoint device vs fast");
        assert_eq!(fast, slow);
        assert_eq!(fast[0], 1.0, "endpoint accepts");
        compared += 1;
    }
    // Past-MAX I33 edge.
    let comp_i: Composition = [8, 8, 0, 0, 0, 0, 0, 0, 0, 33];
    let cand_i = pack_one_cand(&comp_i);
    let (got, fast, slow) = run_e4_pair(&cand_i, 4_294_933_651, 42_949, 1, 0, 4096);
    assert_f32_close(&got, &fast, 1e-6, "past-MAX device vs fast");
    assert_eq!(fast, slow);
    assert_eq!(fast[0], 1.0, "past-MAX accepts");
    compared += 1;
    // Hydrogen-only parent and saturated-cap raw words.
    let cand_h = pack_one_cand(&[0, 5, 0, 0, 0, 0, 0, 0, 0, 0]);
    let (got, fast, slow) = run_e4_pair(&cand_h, 5 * 1_007_825, 119, 1, 0, 4096);
    assert_f32_close(&got, &fast, 1e-6, "H-only device vs fast");
    assert_eq!(fast, slow);
    assert_eq!(fast[0], 0.0, "H-only explains nothing");
    compared += 1;
    for adduct in [1u16, 2] {
        let h_word = if adduct == 1 { 65_534u32 } else { 65_533u32 };
        let words = vec![2, h_word, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, u32::MAX];
        let (got, fast, slow) = run_e4_pair(&words, 24_000_000, 100, adduct, 0, 4096);
        assert_f32_close(&got, &fast, 1e-6, "saturated cap device vs fast");
        assert_eq!(fast, slow, "saturated cap fast vs slow");
        compared += 1;
    }
    println!("E4F-DEVICE-NEW compared={compared}");
    assert!(compared >= 40, "only {compared} device pairs ran");
}

/// E4F host-known hydrogen bounds on uploaded tables: the device table
/// reports `max_hydrogen + 3` without a read.
#[test]
fn e4f_device_table_hydrogen_cap_max() {
    let _guard = serial();
    let device = dev();
    let table = FormulaTable::from_compositions(
        [[1u16, 10, 0, 0, 0, 0, 0, 0, 0, 0], [2, 3, 1, 0, 0, 0, 0, 0, 0, 0]].into_iter(),
    )
    .expect("table builds");
    assert_eq!(table.max_hydrogen(), 10);
    let dtable = DeviceFormulaTable::<R, f32>::upload(&table, &device).expect("upload works");
    assert_eq!(dtable.max_hydrogen, 10);
    assert_eq!(dtable.hydrogen_cap_max(), 13);
}

/// E5F Part A main regression: direct shuffled generation must size from the
/// EXACT uploaded rows.
///
/// Reviewer request: table `C20H200` (mass 441,565,000, `h_cap_max = 203`);
/// rows with fragment ppm-tenths 1000 and 1; peaks at 50,000,000 and
/// 400,499,451; `ShuffledSpectrum`; `formula_evidence_work_max = 1`;
/// dispatch budget 192. The pre-rotation batch gives `tol_max = 5000`;
/// the rotated upload gives `40,049`. Through the hidden trial counter no
/// lane exceeds the per-lane bound sized with the TRUE bound, and the
/// launch count is what the formula gives for the TRUE bound.
#[test]
fn e5f_shuffled_tol_max_dispatch() {
    let _guard = serial();
    use mamba3::models::ms2::batch::{DeviceSpectra, rotate_peaks};
    use mamba3::models::ms2::chem::tolerance;
    use mamba3::models::ms2::contract::SPECTRUM_SCHEMA_VERSION;
    use mamba3::models::ms2::contract::SpectrumBatch;
    let device = dev();
    // Table mass check by independent arithmetic.
    let mass: u64 = 20 * 12_000_000 + 200 * 1_007_825;
    assert_eq!(mass, 441_565_000);
    // Host batch as the reviewer describes it.
    let n_raw = 64usize;
    let mut peak_id = vec![0u32; 2 * n_raw];
    let mut mz = vec![0u32; 2 * n_raw];
    let mut intensity = vec![0.0f32; 2 * n_raw];
    for k in 0..32 {
        peak_id[k] = k as u32;
        mz[k] = 50_000_000;
        intensity[k] = 1.0;
        peak_id[n_raw + k] = 100 + k as u32;
        mz[n_raw + k] = 400_499_451;
        intensity[n_raw + k] = 1.0;
    }
    let batch = SpectrumBatch {
        schema_version: SPECTRUM_SCHEMA_VERSION,
        n_raw: n_raw as u32,
        spectrum_id: vec![1, 2],
        raw_peak_count: vec![132, 132],
        peak_count: vec![32, 32],
        peak_id,
        mz_udalton: mz,
        intensity,
        intensity_scale: 0,
        mz_uncertainty_udalton: vec![0, 0],
        precursor_mz_udalton: vec![442_572_276, 442_572_276],
        precursor_uncertainty_udalton: vec![0, 0],
        adduct: vec![1, 1],
        polarity: vec![1, 1],
        collision_energy_ev: vec![30.0, 30.0],
        collision_energy_known: vec![1, 1],
        energy_count: vec![1, 1],
        fragment_tolerance_ppm_tenths: vec![1000, 1],
        precursor_tolerance_ppm_tenths: vec![0, 0],
        instrument_class: vec![0, 0],
    };
    // Pre-rotation bound (the bug's source): 5000.
    assert_eq!(tolerance(50_000_000, 1000), 5_000);
    assert_eq!(tolerance(400_499_451, 1), 40);
    assert_eq!(batch.max_fragment_tolerance(), 5_000);
    // Exact uploaded rows (rotation): row 0 gets the 400M peaks with its
    // own ppm 1000 -> 40,049; row 1 gets 50M with ppm 1 -> 5.
    assert_eq!(tolerance(400_499_451, 1000), 40_049);
    assert_eq!(tolerance(50_000_000, 1), 5);
    let rotated = rotate_peaks(&batch);
    let spectra = DeviceSpectra::<R, f32>::upload(&rotated, &device).unwrap();
    assert_eq!(spectra.uploaded_tol_max(), 40_049);
    // Dispatch bounds by the item-2 formula (independent arithmetic here).
    let h_cap_max = 203u32;
    let tol_false = 5_000u32;
    let tol_true = 40_049u32;
    assert_eq!(trials_bound_of(h_cap_max, tol_false), 6);
    assert_eq!(trials_bound_of(h_cap_max, tol_true), 24);
    let per_lane_false = 1u64 * 32 * 6;
    let per_lane_true = 1u64 * 32 * 24;
    assert_eq!(per_lane_false, 192);
    assert_eq!(per_lane_true, 768);
    // Hidden trial counter on the twin: C20H200, one visit (W = 1), 32
    // peaks at (t = 400,500,000, tol = 40,049), adduct 1, U = 0.
    let comp: Composition = [20, 200, 0, 0, 0, 0, 0, 0, 0, 0];
    let cand = pack_one_cand(&comp);
    let (t, tol) = (400_500_000u32, 40_049u32);
    assert_eq!(400_499_451u32 + 549, t);
    let mut ev_peaks = Vec::new();
    let mut ev_w = Vec::new();
    for s in 0..32 {
        ev_peaks.extend_from_slice(&[s, t, tol, 1]);
        ev_w.push(1.0 / 32.0);
    }
    let meta = vec![32u32, 0, 0, 1, 1000, 0, 0, 0];
    let spec = vec![0u32, 0];
    let mut row = vec![0.0f32; 4];
    let mut trials = 0u64;
    twin::formula_evidence_lane_trials(
        &cand, &ev_peaks, &ev_w, &meta, &spec, 0, 0, 1, 32, 1, h_cap_max,
        &mut row, &mut trials,
    );
    // Reviewer values: 21 trials per peak (ranges 59..=69 and 187..=196),
    // 672 per lane.
    assert_eq!(trials, 672, "reviewer lane executes 672 hydrogen trials");
    assert!(
        trials <= per_lane_true,
        "no lane exceeds the TRUE per-lane bound {per_lane_true}: {trials}"
    );
    assert!(
        trials > per_lane_false,
        "the FALSE bound {per_lane_false} would have been violated"
    );
    // Launch count through the real wrapper with the TRUE bound and budget
    // 192: lanes = 2 (B = 2, M = 1), per = max(1, 192 / 768) = 1, so 2
    // launches — exactly what the formula gives for the TRUE bound.
    let lanes = 2u64;
    let per = (192u64 / per_lane_true).max(1) as usize;
    assert_eq!(per, 1);
    assert_eq!(lanes.div_ceil(per as u64), 2);
    let cand2 = {
        let mut c = cand.clone();
        c.extend_from_slice(&cand);
        c
    };
    let meta2 = {
        let mut m = meta.clone();
        m.extend_from_slice(&[32u32, 0, 0, 1, 1, 0, 0, 0]);
        m
    };
    let spec2 = vec![0u32, 0, 0, 0];
    let peaks2 = {
        let mut p = ev_peaks.clone();
        // Second row: same shape (32 peaks at its own t/tol); the count is
        // what matters for sizing (P = 32 each).
        p.extend_from_slice(&ev_peaks);
        p
    };
    let w2 = {
        let mut w = ev_w.clone();
        w.extend_from_slice(&ev_w);
        w
    };
    let cand_t = upload_ids(&cand2, vec![2, 1, 13], &device);
    let peaks_t = upload_ids(&peaks2, vec![2, 32, 4], &device);
    let w_t = upload_f(&w2, vec![2, 32], &device);
    let meta_t = upload_ids(&meta2, vec![2, 8], &device);
    let spec_t = upload_ids(&spec2, vec![2, 2], &device);
    let mut out_t = upload_f(&vec![f32::NAN; 8], vec![2, 1, 4], &device);
    reset_launch_count();
    let before = launch_count();
    kernels::formula_evidence(
        &cand_t, &peaks_t, &w_t, &meta_t, &spec_t, &mut out_t, 1, 192, h_cap_max,
        tol_true,
    )
    .unwrap();
    check_launches(&device).unwrap();
    assert_eq!(launch_count() - before, 2, "launch count for the TRUE bound");
}

/// E5F-d: the zero-carbon prefix (`c[C] = 0`, `V = 1`: zero-length
/// `ion_assign` prefix) through the DEVICE kernel on poisoned outputs.
///
/// Parent `[N1,H4]` under `W = 1` visits only `j = 0`; nothing explained,
/// incomplete. Device output (poisoned with NaN) equals the twin row.
#[test]
fn e5f_d_zero_carbon_prefix_device() {
    let _guard = serial();
    let comp: Composition = [0, 4, 1, 0, 0, 0, 0, 0, 0, 0];
    let cand = pack_one_cand(&comp);
    let t = 14_003_074u32;
    let mz = t - 549;
    let tol = tol_u64(mz, 100);
    let ev = vec![0, t, tol, 1];
    let meta = vec![0u32, 0, 0, 1, 0, 0, 0, 0];
    let spec = vec![0u32, 0];
    let want = twin::formula_evidence(&cand, &ev, &[1.0], &meta, &spec, 1, 1, 1, 1, u32::MAX);
    assert_eq!(&want[..], &[0.0, 0.0, 1.0, 0.0][..], "V = 1 explains nothing, incomplete");
    let got = run_evidence_lane_device_bounds(&cand, &ev, &[1.0], &meta, &spec, 1, 1, u32::MAX, 1_007_825);
    assert_f32_close(&got, &want, 1e-6, "device zero-carbon prefix matches twin");
    // Zero-length ion_assign prefix accepts nothing.
    let assign = ion_assign(
        &comp, 1, mz, 0, 100,
        &IonLimits { work_max: 0, kept: 4 },
    )
    .expect("ion_assign runs");
    assert_eq!(assign.accepted, 0, "zero-length prefix accepts nothing");
}
