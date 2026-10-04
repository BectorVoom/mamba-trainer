//! V0-A tests: the formula window and top-F kernels against their host
//! twins, the window mask, and the formula head (score, loss, gold slots).
//!
//! Every device call is followed by [`check_launches`], so a kernel that
//! failed to compile or run is an error rather than stale data.

#![cfg(feature = "backend")]

use mamba3::autograd::Var;
use mamba3::backend::{Device, check_launches};
use mamba3::backends::Auto;
use mamba3::models::ms2::MolGraph;
use mamba3::models::ms2::batch::DeviceSpectra;
use mamba3::models::ms2::chem::{Composition, composition_error_nda, composition_mass};
use mamba3::models::ms2::contract::{Control, ModelConfig, request_status};
use mamba3::models::ms2::dataset::ExportSpectrum;
use mamba3::models::ms2::decoder::{ReplayView, graph_loss};
use mamba3::models::ms2::experiment::{
    ExperimentSet, ExperimentSpectrum, SpectrumDomain, spectrum_batch_for, target_batch_for,
};
use mamba3::models::ms2::formula::{FormulaTable, WindowQuery};
use mamba3::models::ms2::formula_head::{
    DeviceFormulaTable, FormulaHead, gold_slots, gold_slots_host,
};
use mamba3::models::ms2::grammar::{ADD_ATOM, Limits, START, STOP, Token, TraceState};
use mamba3::models::ms2::targets::{Labels, Target};
use mamba3::models::ms2::targets_batch::DeviceTargets;
use mamba3::models::ms2::train::{
    GoldFormulaConditioning, Ms2Trainer, TRAIN_ROWS_SCORED_MAX, TRAIN_WINDOW_M, TrainConfig,
};
use mamba3::models::ms2::{tolerance, twin};
use mamba3::nn::Module;
use mamba3::tensor::Tensor;
use mamba3::tensor::ops::index::IdTensor;
use mamba3::tensor::ops::ms2;
use mamba3::tensor::ops::random::Rng;

type R = Auto;
type E = f32;

fn dev() -> Device<R> {
    Device::<R>::default()
}

/// Net hydrogen shift of contract §4.3, `m_H - m_e`.
const H_NET: u32 = 1_007_825 - 549;

/// A C/H/N/O composition; every other element is zero.
fn comp(c: u16, h: u16, n: u16, o: u16) -> Composition {
    [c, h, n, o, 0, 0, 0, 0, 0, 0]
}

/// A few well-known small compositions: the "fixture" half of the tables.
fn fixture_compositions() -> Vec<Composition> {
    vec![
        comp(0, 2, 0, 1),  // water
        comp(1, 0, 0, 2),  // carbon dioxide
        comp(1, 4, 0, 0),  // methane
        comp(2, 6, 0, 1),  // ethanol
        comp(6, 6, 0, 0),  // benzene
        comp(6, 12, 0, 6), // glucose
        comp(8, 10, 4, 2), // caffeine
        comp(2, 4, 0, 1),  // the ambiguous-row composition of ms2_targets
        comp(0, 3, 1, 0),  // ammonia
        comp(1, 0, 0, 1),  // carbon monoxide
        comp(7, 5, 1, 3),  // a nitro-aromatic fragment
        comp(5, 5, 1, 0),  // pyridine-ish
    ]
}

/// Seeded synthetic compositions: small C/H/N/O counts with occasional F, P,
/// S, Cl, so the table is dense enough for multi-row windows.
fn synthetic_compositions(seed: u64, n: usize) -> Vec<Composition> {
    let mut rng = Rng::seeded(seed);
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let u = rng.uniform_vec(10, 0.0, 1.0);
        let c = (u[0] * 21.0) as u16;
        let h = (u[1] * 41.0) as u16;
        let ni = (u[2] * 6.0) as u16;
        let o = (u[3] * 11.0) as u16;
        let mut e = [c, h, ni, o, 0, 0, 0, 0, 0, 0];
        e[4] = (u[4] * 3.0) as u16;
        e[5] = (u[5] * 2.0) as u16;
        e[6] = (u[6] * 2.0) as u16;
        e[7] = (u[7] * 2.0) as u16;
        if e.iter().all(|&v| v == 0) {
            continue;
        }
        out.push(e);
    }
    out
}

/// The `[R, 2]` integer search table (mass, per-row arithmetic bound) the
/// [`ms2::formula_window`] kernel reads.
fn upload_search_table(table: &FormulaTable, device: &Device<R>) -> IdTensor<R> {
    let mut v = Vec::with_capacity(table.len() * 2);
    for row in 0..table.len() {
        v.push(table.mass(row));
        v.push((composition_error_nda(table.composition(row)).div_ceil(1000)) as u32);
    }
    IdTensor::from_slice(&v, vec![table.len(), 2], device).unwrap()
}

/// One `[B, 8]` metadata row: peak count 3 (the search never reads it),
/// precursor, uncertainty, adduct, fragment tolerance, precursor tolerance.
fn meta_row(precursor: u32, unc: u32, adduct: u32, ppm_tenths: u32) -> [u32; 8] {
    [3, precursor, unc, adduct, 100, ppm_tenths, 0, 0]
}

/// Precursor m/z of a neutral parent mass under an adduct.
fn precursor_of(parent: u32, adduct: u16) -> u32 {
    match adduct {
        1 => parent + H_NET,
        2 => parent - H_NET,
        _ => panic!("test adduct"),
    }
}

fn assert_windows_eq(
    got_window: &[u32],
    got_counters: &[u32],
    want_window: &[u32],
    want_counters: &[u32],
    what: &str,
) {
    assert_eq!(got_window, want_window, "{what}: window rows/flags differ");
    assert_eq!(got_counters, want_counters, "{what}: counters differ");
}

fn assert_close(actual: &[f32], expected: &[f32], tol: f32, what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        let ok = if e.abs() <= 1.0 {
            (a - e).abs() <= tol
        } else {
            (a - e).abs() <= tol * e.abs()
        };
        assert!(ok, "{what}: index {i} got {a}, want {e}");
    }
}

/// Run [`ms2::formula_window`] on poisoned buffers and read both outputs back.
fn run_window(
    search: &IdTensor<R>,
    meta: &[u32],
    batch: usize,
    m: usize,
    max_error: u32,
    rows_visited_max: u32,
    rows_scored_max: u32,
) -> (Vec<u32>, Vec<u32>) {
    let device = dev();
    let meta_t = IdTensor::from_slice(meta, vec![batch, 8], &device).unwrap();
    let out = ms2::FormulaBuffers::<R, E>::poisoned(batch, m, 4, &device).unwrap();
    ms2::formula_window(
        search,
        &meta_t,
        max_error,
        rows_visited_max,
        rows_scored_max,
        &out,
    )
    .unwrap();
    check_launches(&device).unwrap();
    (
        out.window.try_to_vec().unwrap(),
        out.counters.try_to_vec().unwrap(),
    )
}

fn check_window(
    table: &FormulaTable,
    search: &IdTensor<R>,
    meta: &[u32],
    batch: usize,
    m: usize,
    rows_visited_max: u32,
    rows_scored_max: u32,
    what: &str,
) {
    let (got_window, got_counters) = run_window(
        search,
        meta,
        batch,
        m,
        table.max_error(),
        rows_visited_max,
        rows_scored_max,
    );
    let (want_window, want_counters) = twin::formula_window(
        table,
        meta,
        batch,
        m,
        table.max_error(),
        rows_visited_max,
        rows_scored_max,
    );
    assert_windows_eq(
        &got_window,
        &got_counters,
        &want_window,
        &want_counters,
        what,
    );
}

/// Expand a `[B, M, 2]` window flat into `[B, M, 13]` cand flat for
/// [`ms2::formula_top`]: counts and mass are zero, flag and source row pass
/// through (the kernel reads only the flag at +11 and the source at +12).
fn cand_from_window(window: &[u32], batch: usize, m: usize) -> Vec<u32> {
    let mut cand = vec![0u32; batch * m * 13];
    for b in 0..batch {
        for mm in 0..m {
            let row = window[(b * m + mm) * 2];
            let flag = window[(b * m + mm) * 2 + 1];
            let base = (b * m + mm) * 13;
            cand[base + 11] = flag;
            cand[base + 12] = row;
        }
    }
    cand
}

/// A table with a three-row isobaric cluster: a heavy base composition plus
/// N2, CO and C2H4, whose pairwise mass gaps (at most 36,385 units) sit well
/// inside the 1000-tenth tolerance at ~1,320 Da (~132,000 units), so one
/// precursor joins all three. Returns the table and that precursor.
fn cluster_table() -> (FormulaTable, u32) {
    let base: Composition = [60, 120, 0, 30, 0, 0, 0, 0, 0, 0];
    let add = |extra: Composition| {
        let mut c = base;
        for (a, b) in c.iter_mut().zip(extra.iter()) {
            *a += b;
        }
        c
    };
    let table = FormulaTable::from_compositions(
        [
            add(comp(0, 0, 2, 0)),
            add(comp(1, 0, 0, 1)),
            add(comp(2, 4, 0, 0)),
        ]
        .into_iter(),
    )
    .unwrap();
    assert_eq!(table.len(), 3, "the cluster rows are distinct");
    // The middle row by mass: both neighbours join its window.
    let precursor = table.mass(1) + H_NET;
    let query = WindowQuery {
        precursor_mz: precursor,
        adduct: 1,
        ppm_tenths: 1000,
        precursor_uncertainty: 50,
        rows_visited_max: u32::MAX,
        rows_scored_max: u32::MAX,
    };
    assert_eq!(
        table.window(&query).rows_joined,
        3,
        "the cluster precursor joins all three rows"
    );
    (table, precursor)
}

#[test]
fn formula_window_matches_reference() {
    let device = dev();
    let table = FormulaTable::from_compositions(
        fixture_compositions()
            .into_iter()
            .chain(synthetic_compositions(11, 64).into_iter()),
    )
    .unwrap();
    let search = upload_search_table(&table, &device);
    // A hit row: precursor computed from its mass under [M+H]+.
    let hit_row = 3usize;
    let hit_mass = table.mass(hit_row);
    let m = 32usize;
    // Group A: one batch under the default limits, covering hits ([M+H]+ and
    // [M-H]-), an absent precursor, an ambiguous row, MASS_OVERFLOW, the
    // unknown-precision sentinel, a 1000-tenth tolerance and precursor u32::MAX.
    let amb = {
        // As in tests/ms2_targets.rs: a parent one unit past the tolerance
        // edge joins as Ambiguous (r + E > tol while r <= tol + E).
        let c = comp(2, 4, 0, 1);
        let one = FormulaTable::from_compositions([c]).unwrap();
        let mass = one.mass(0);
        let mut parent = mass + 5;
        for _ in 0..3 {
            let precursor = parent + H_NET;
            let tol = tolerance(precursor, 1);
            parent = mass + tol + 1;
        }
        (one, parent)
    };
    let amb_precursor = amb.1 + H_NET;
    let rows: Vec<[u32; 8]> = vec![
        meta_row(precursor_of(hit_mass, 1), 50, 1, 200),
        meta_row(precursor_of(hit_mass, 2), 50, 2, 200),
        // Absent: 1 Da has no table row near it.
        meta_row(1_000_000 + H_NET, 50, 1, 200),
        // The ambiguous precursor, also run against the big table: whatever
        // it joins there must agree with the reference.
        meta_row(amb_precursor, 0, 1, 1),
        // Precursor 0 under [M+H]+: the parent leaves u32.
        meta_row(0, 50, 1, 200),
        // Unknown precursor precision: nothing searched.
        meta_row(precursor_of(hit_mass, 1), u32::MAX, 1, 200),
        // Wide tolerance.
        meta_row(precursor_of(hit_mass, 1), 50, 1, 1000),
        // Precursor u32::MAX under [M-H]-: the parent leaves u32.
        meta_row(u32::MAX, 50, 2, 200),
    ];
    let batch = rows.len();
    let meta: Vec<u32> = rows.iter().flat_map(|r| r.iter().copied()).collect();
    check_window(
        &table,
        &search,
        &meta,
        batch,
        m,
        u32::MAX,
        u32::MAX,
        "group-a",
    );
    // The absent precursor really is absent, the overflow really overflows.
    let (_, counters) = twin::formula_window(
        &table,
        &meta,
        batch,
        m,
        table.max_error(),
        u32::MAX,
        u32::MAX,
    );
    assert!(
        counters[2 * 5 + 3] & request_status::FORMULA_ABSENT != 0,
        "spectrum 2 is absent"
    );
    assert!(
        counters[4 * 5 + 3] & request_status::MASS_OVERFLOW != 0,
        "spectrum 4 overflows"
    );
    assert_eq!(
        counters[5 * 5 + 3],
        request_status::EXACT_MASS_UNAVAILABLE | request_status::FORMULA_ABSENT,
        "spectrum 5 is the sentinel"
    );
    assert!(
        counters[7 * 5 + 3] & request_status::MASS_OVERFLOW != 0,
        "spectrum 7 overflows"
    );
    // The ambiguous construction against its single-row table.
    let amb_search = upload_search_table(&amb.0, &device);
    let amb_meta = meta_row(amb_precursor, 0, 1, 1);
    check_window(
        &amb.0,
        &amb_search,
        &amb_meta,
        1,
        m,
        u32::MAX,
        u32::MAX,
        "ambiguous",
    );
    let (amb_window, _) = twin::formula_window(
        &amb.0,
        &amb_meta,
        1,
        m,
        amb.0.max_error(),
        u32::MAX,
        u32::MAX,
    );
    assert_eq!(amb_window[0], 0, "ambiguous row joins");
    assert_eq!(amb_window[1], 2, "ambiguous row is flagged 2");

    // Group B: visit limits — 1, 3, and one that stops the row scan
    // part-way (one visit short of the full search over a multi-row window).
    // The cluster table's three-row window gives the scan something to stop in.
    let (cluster, dense_precursor) = cluster_table();
    let cluster_search = upload_search_table(&cluster, &device);
    let wide_meta: Vec<u32> = vec![meta_row(dense_precursor, 50, 1, 1000)]
        .into_iter()
        .flat_map(|r| r.into_iter())
        .collect();
    let (_, full_counters) = twin::formula_window(
        &cluster,
        &wide_meta,
        1,
        m,
        cluster.max_error(),
        u32::MAX,
        u32::MAX,
    );
    let full_visited = full_counters[0];
    assert!(full_visited > 4, "the wide search reads several rows");
    for &limit in &[1u32, 3, full_visited - 1] {
        check_window(
            &cluster,
            &cluster_search,
            &wide_meta,
            1,
            m,
            limit,
            u32::MAX,
            &format!("visit-limit-{limit}"),
        );
    }
    let (_, limited) = twin::formula_window(
        &cluster,
        &wide_meta,
        1,
        m,
        cluster.max_error(),
        full_visited - 1,
        u32::MAX,
    );
    assert!(
        limited[3] & request_status::FORMULA_SEARCH_EXHAUSTED != 0,
        "the part-way stop is exhausted"
    );

    // Group C: one scored row while several join.
    check_window(
        &cluster,
        &cluster_search,
        &wide_meta,
        1,
        m,
        u32::MAX,
        1,
        "scored-max-1",
    );
    let (_, capped) =
        twin::formula_window(&cluster, &wide_meta, 1, m, cluster.max_error(), u32::MAX, 1);
    assert_eq!(capped[1], 3, "three rows join");
    assert_eq!(capped[2], 1, "one row is scored");
    assert!(
        capped[3] & request_status::FORMULA_SEARCH_EXHAUSTED != 0,
        "the cap exhausts"
    );

    // Group D: M smaller than the joined count.
    check_window(
        &cluster,
        &cluster_search,
        &wide_meta,
        1,
        2,
        u32::MAX,
        u32::MAX,
        "m-2",
    );

    // Group E: a batch of 64 seeded random queries over a 2,000-row
    // synthetic table.
    let big =
        FormulaTable::from_compositions(synthetic_compositions(2026, 2200).into_iter()).unwrap();
    assert!(big.len() >= 2000, "the synthetic table has 2,000 rows");
    let big_search = upload_search_table(&big, &device);
    let mut rng = Rng::seeded(99);
    let u = rng.uniform_vec(64 * 3, 0.0, 1.0);
    let mut erows = Vec::with_capacity(64);
    for i in 0..64 {
        if i % 2 == 0 {
            let row = (u[i * 3] * big.len() as f32) as usize % big.len();
            let ad = if u[i * 3 + 1] < 0.5 || big.mass(row) < H_NET {
                1
            } else {
                2
            };
            let prec = if ad == 1 {
                big.mass(row) + H_NET
            } else {
                big.mass(row) - H_NET
            };
            erows.push(meta_row(prec, 50, ad, 200));
        } else {
            let precursor = 50_000_000 + ((u[i * 3] * 450_000_000.0) as u32);
            erows.push(meta_row(precursor, 50, 1, 200));
        }
    }
    let emeta: Vec<u32> = erows.iter().flat_map(|r| r.iter().copied()).collect();
    check_window(
        &big,
        &big_search,
        &emeta,
        64,
        32,
        u32::MAX,
        4096,
        "random-64",
    );
}

#[test]
fn formula_top_matches_twin() {
    let device = dev();
    // Hand case: ties among flagged slots, and fewer joined rows than F.
    let (m, f) = (6usize, 4usize);
    let window: Vec<u32> = vec![
        10, 1, 11, 0, 12, 2, 13, 1, 14, 0, 15, 2, // spectrum 0: slots 0,2,3,5
        20, 1, 21, 0, 22, 0, 23, 0, 24, 0, 25, 0, // spectrum 1: slot 0 only
    ];
    let log_prob: Vec<f32> = vec![
        0.5, -99.0, 0.5, 0.25, -99.0, 0.75, // tie at slots 0 and 2
        1.0, -99.0, -99.0, -99.0, -99.0, -99.0,
    ];
    let lp_t = Tensor::<R, E>::from_f32(&log_prob, vec![2, m], &device).unwrap();
    let out = ms2::FormulaBuffers::<R, E>::poisoned(2, m, f, &device).unwrap();
    let cand = cand_from_window(&window, 2, m);
    let cand_t = IdTensor::from_slice(&cand, vec![2, m, 13], &device).unwrap();
    ms2::formula_top(&lp_t, &cand_t, &out).unwrap();
    check_launches(&device).unwrap();
    let (want_top, want_lp, want_count) = twin::formula_top(&log_prob, &window, 2, m, f);
    assert_eq!(out.top.try_to_vec().unwrap(), want_top, "top rows/slots");
    assert_close(
        &out.top_log_prob.try_to_f32().unwrap(),
        &want_lp,
        1e-5,
        "top log-probs",
    );
    assert_eq!(
        out.top_count.try_to_vec().unwrap(),
        want_count,
        "top counts"
    );
    // The tie breaks by smaller slot; the single-join spectrum pads the rest.
    assert_eq!(&want_top[0..8], &[15, 5, 10, 0, 12, 2, 13, 3], "tie order");
    assert_eq!(want_count, vec![4, 1]);
    assert_eq!(
        &want_top[8..16],
        &[
            20,
            0,
            u32::MAX,
            u32::MAX,
            u32::MAX,
            u32::MAX,
            u32::MAX,
            u32::MAX
        ]
    );

    // Random case over a real search window.
    let table =
        FormulaTable::from_compositions(synthetic_compositions(7, 128).into_iter()).unwrap();
    let search = upload_search_table(&table, &device);
    let mass = table.mass(table.len() / 2);
    let meta: Vec<u32> = vec![
        meta_row(mass + H_NET, 50, 1, 1000),
        meta_row(mass + H_NET, 50, 1, 200),
    ]
    .into_iter()
    .flat_map(|r| r.into_iter())
    .collect();
    let (got_window, _) = run_window(&search, &meta, 2, 16, table.max_error(), u32::MAX, u32::MAX);
    let mut rng = Rng::seeded(3);
    let lp = rng.uniform_vec(2 * 16, -2.0, 0.0);
    let lp_t = Tensor::<R, E>::from_f32(&lp, vec![2, 16], &device).unwrap();
    let out = ms2::FormulaBuffers::<R, E>::poisoned(2, 16, 4, &device).unwrap();
    let cand = cand_from_window(&got_window, 2, 16);
    let cand_t = IdTensor::from_slice(&cand, vec![2, 16, 13], &device).unwrap();
    ms2::formula_top(&lp_t, &cand_t, &out).unwrap();
    check_launches(&device).unwrap();
    let (want_top, want_lp, want_count) = twin::formula_top(&lp, &got_window, 2, 16, 4);
    assert_eq!(out.top.try_to_vec().unwrap(), want_top, "random top");
    assert_close(
        &out.top_log_prob.try_to_f32().unwrap(),
        &want_lp,
        1e-5,
        "random top log-probs",
    );
    assert_eq!(
        out.top_count.try_to_vec().unwrap(),
        want_count,
        "random counts"
    );
}

#[test]
fn gold_slots_host_applies_the_scored_cap() {
    // A three-row cluster with window capacity M = 1: the kernel scores only
    // the first joined row, so gold row 1 reports u32::MAX, not 1.
    let device = dev();
    let (table, precursor) = cluster_table();
    let query = WindowQuery {
        precursor_mz: precursor,
        adduct: 1,
        ppm_tenths: 1000,
        precursor_uncertainty: 50,
        rows_visited_max: u32::MAX,
        rows_scored_max: u32::MAX,
    };
    assert_eq!(table.window(&query).rows_joined, 3);
    let queries = vec![query];
    // Gold row 1 (the middle row by mass) with M = 1.
    assert_eq!(gold_slots_host(&table, &queries, &[1], 1), vec![u32::MAX]);
    // With M = 3 the same gold row is scored at slot 1.
    assert_eq!(gold_slots_host(&table, &queries, &[1], 3), vec![1]);
    // And the device window agrees: slot u32::MAX, not 1.
    let search = upload_search_table(&table, &device);
    let meta: Vec<u32> = vec![meta_row(precursor, 50, 1, 1000)]
        .into_iter()
        .flat_map(|r| r.into_iter())
        .collect();
    let meta_t = IdTensor::from_slice(&meta, vec![1, 8], &device).unwrap();
    let buffers = ms2::FormulaBuffers::<R, E>::new(1, 1, 4, &device);
    ms2::formula_window(
        &search,
        &meta_t,
        table.max_error(),
        u32::MAX,
        u32::MAX,
        &buffers,
    )
    .unwrap();
    check_launches(&device).unwrap();
    let window = buffers.window.try_to_vec().unwrap();
    assert_eq!(gold_slots(&window, &[1]), vec![u32::MAX]);
}

#[test]
fn formula_top_nan_never_writes_out_of_bounds() {
    let device = dev();
    // Both slots joined; the second holds NaN. NaN lies outside the
    // validated domain, so it is never selected: the finite slot is written
    // and the count compacts to it — no crash, no out-of-bounds write, no
    // padding slot selected by `k mod top_count`.
    let (m, f) = (2usize, 2usize);
    let window: Vec<u32> = vec![40, 1, 41, 1];
    let log_prob: Vec<f32> = vec![0.0, f32::NAN];
    let cand = cand_from_window(&window, 1, m);
    let cand_t = IdTensor::from_slice(&cand, vec![1, m, 13], &device).unwrap();
    let lp_t = Tensor::<R, E>::from_f32(&log_prob, vec![1, m], &device).unwrap();
    let out = ms2::FormulaBuffers::<R, E>::poisoned(1, m, f, &device).unwrap();
    ms2::formula_top(&lp_t, &cand_t, &out).unwrap();
    check_launches(&device).unwrap();
    let (want_top, want_lp, want_count) = twin::formula_top(&log_prob, &window, 1, m, f);
    assert_eq!(
        out.top.try_to_vec().unwrap(),
        want_top,
        "kernel matches twin"
    );
    assert_eq!(
        want_top,
        vec![40, 0, u32::MAX, u32::MAX],
        "finite first, rest padding"
    );
    assert_eq!(want_lp[0], 0.0);
    assert_eq!(want_lp[1], 0.0);
    assert_eq!(want_count, vec![1]);
    let got_top = out.top.try_to_vec().unwrap();
    let got_count = out.top_count.try_to_vec().unwrap();
    assert_eq!(
        got_count,
        vec![1],
        "kernel count compacts to written entries"
    );
    assert_eq!(got_count[0] as usize, 1);
    assert_ne!(got_top[0], u32::MAX, "slot below top_count is a real row");
    // The mirrored order `[NaN, 0]`: NaN is never selected, so the finite
    // slot 1 wins the only pick and `top_count` is 1; every slot below it
    // is a valid row index, on the kernel and the twin.
    let log_prob: Vec<f32> = vec![f32::NAN, 0.0];
    let lp_t = Tensor::<R, E>::from_f32(&log_prob, vec![1, m], &device).unwrap();
    let out = ms2::FormulaBuffers::<R, E>::poisoned(1, m, f, &device).unwrap();
    ms2::formula_top(&lp_t, &cand_t, &out).unwrap();
    check_launches(&device).unwrap();
    let (want_top, _, want_count) = twin::formula_top(&log_prob, &window, 1, m, f);
    assert_eq!(
        out.top.try_to_vec().unwrap(),
        want_top,
        "kernel matches twin"
    );
    assert_eq!(
        want_top,
        vec![41, 1, u32::MAX, u32::MAX],
        "finite slot wins over NaN"
    );
    assert_eq!(want_count, vec![1]);
    for pick in 0..f {
        let row = want_top[pick * 2];
        assert!(
            row == 40 || row == 41 || row == u32::MAX,
            "pick {pick} row {row} is in bounds"
        );
    }
    let got_top = out.top.try_to_vec().unwrap();
    let got_count = out.top_count.try_to_vec().unwrap();
    assert_eq!(
        got_count, want_count,
        "kernel and twin agree on the compacted count"
    );
    let non_padding = got_top
        .chunks_exact(2)
        .filter(|pair| pair[0] != u32::MAX)
        .count();
    assert_eq!(
        got_count[0] as usize, non_padding,
        "[NaN, 0] top_count equals the number of non-padding slots"
    );
    for pick in 0..got_count[0] as usize {
        assert_ne!(
            got_top[pick * 2],
            u32::MAX,
            "slot {pick} below top_count is a real row"
        );
        assert!(
            got_top[pick * 2] == 40 || got_top[pick * 2] == 41,
            "slot {pick} row is valid"
        );
    }
    let twin_non_padding = want_top
        .chunks_exact(2)
        .filter(|pair| pair[0] != u32::MAX)
        .count();
    assert_eq!(
        want_count[0] as usize, twin_non_padding,
        "twin top_count equals the number of non-padding slots"
    );
    for pick in 0..want_count[0] as usize {
        assert_ne!(
            want_top[pick * 2],
            u32::MAX,
            "twin slot {pick} below top_count is a real row"
        );
    }
}

#[test]
fn nonzero_mask_maps_flags() {
    let device = dev();
    // Flags 0/1/2 become 0/1/1.
    let window = vec![7u32, 0, 8, 1, 9, 2, 10, 0, 11, 1, 12, 2];
    let window_t = IdTensor::from_slice(&window, vec![2, 3, 2], &device).unwrap();
    let mask: Tensor<R, E> = ms2::nonzero_mask(&window_t).unwrap();
    check_launches(&device).unwrap();
    assert_close(
        &mask.try_to_f32().unwrap(),
        &[0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
        0.0,
        "nonzero_mask",
    );
}

fn small_head(device: &Device<R>) -> (FormulaHead<R, E>, ModelConfig) {
    let mut model = ModelConfig::v0();
    model.d_model = 8;
    let mut rng = Rng::seeded(5);
    let head = FormulaHead::<R, E>::init(&model, device, &mut rng).unwrap();
    (head, model)
}

#[test]
fn formula_head_score_normalises_over_joined_slots() {
    let device = dev();
    let (head, _) = small_head(&device);
    let (table, hit) = cluster_table();
    let uploaded = DeviceFormulaTable::<R, E>::upload(&table, &device).unwrap();
    let m = 8usize;
    // Spectrum 0 joins all three cluster rows; spectrum 1 joins nothing.
    let meta: Vec<u32> = vec![
        meta_row(hit, 50, 1, 1000),
        meta_row(1_000_000 + H_NET, 50, 1, 200),
    ]
    .into_iter()
    .flat_map(|r| r.into_iter())
    .collect();
    let search = upload_search_table(&table, &device);
    let meta_t = IdTensor::from_slice(&meta, vec![2, 8], &device).unwrap();
    let mut buffers = ms2::FormulaBuffers::<R, E>::new(2, m, 4, &device);
    ms2::formula_window(
        &search,
        &meta_t,
        table.max_error(),
        u32::MAX,
        u32::MAX,
        &buffers,
    )
    .unwrap();
    ms2::formula_gather(
        &buffers.window,
        &uploaded.table,
        &uploaded.counts,
        &mut buffers.cand,
    )
    .unwrap();
    ms2::count_features(
        &buffers.cand.reshape(vec![2 * m, 13]).unwrap(),
        &uploaded.log_table,
        &mut buffers.cand_feat.reshape(vec![2 * m, 10]).unwrap(),
        13,
    )
    .unwrap();
    check_launches(&device).unwrap();
    let mut rng = Rng::seeded(17);
    let pool_host = rng.uniform_vec(2 * 8, -0.5, 0.5);
    let pool = Var::constant(Tensor::<R, E>::from_f32(&pool_host, vec![2, 8], &device).unwrap());
    let out = head.score(&buffers, &pool).unwrap();
    check_launches(&device).unwrap();
    let log_prob = out.log_prob.try_to_f32().unwrap();
    let mask = out.mask.try_to_f32().unwrap();
    let window = buffers.window.try_to_vec().unwrap();
    // Spectrum 0: exponentiated log-probs sum to 1 over the joined slots
    // (within 1e-5); masked slots share the finite minimum and sit below.
    let joined0: Vec<usize> = (0..m).filter(|&s| window[s * 2 + 1] != 0).collect();
    assert!(joined0.len() >= 2, "the hit spectrum joins rows");
    let mut total = 0.0f32;
    let mut masked_min = f32::INFINITY;
    for s in 0..m {
        let lp = log_prob[s];
        assert!(lp.is_finite(), "slot {s} is finite");
        if window[s * 2 + 1] != 0 {
            total += lp.exp();
            assert_eq!(mask[s], 1.0, "joined slot {s} is masked in");
        } else {
            masked_min = masked_min.min(lp);
            assert_eq!(mask[s], 0.0, "rejected slot {s} is masked out");
        }
    }
    assert!(
        (total - 1.0).abs() <= 1e-5,
        "joined mass sums to 1: {total}"
    );
    for &s in &joined0 {
        assert!(
            log_prob[s] > masked_min,
            "joined slot {s} beats masked slots"
        );
    }
    // Spectrum 1: no joined row gives an all-zero mask and a finite log-prob.
    assert!(
        mask[m..].iter().all(|&v| v == 0.0),
        "the empty row reports a zero mask"
    );
    assert!(
        log_prob[m..].iter().all(|v| v.is_finite()),
        "the empty row is finite"
    );
    assert!(
        (log_prob[m] - 0.0).abs() <= 1e-5,
        "the all-masked rule puts log-prob 0 at slot 0"
    );
}

#[test]
fn formula_head_loss_gradient_matches_finite_differences() {
    let device = dev();
    let (head, _) = small_head(&device);
    let table = FormulaTable::from_compositions(fixture_compositions().into_iter()).unwrap();
    let uploaded = DeviceFormulaTable::<R, E>::upload(&table, &device).unwrap();
    let m = 6usize;
    let hit0 = precursor_of(table.mass(2), 1);
    let hit1 = precursor_of(table.mass(6), 1);
    let meta: Vec<u32> = vec![
        meta_row(hit0, 50, 1, 1000),
        meta_row(hit1, 50, 1, 1000),
        meta_row(1_000_000 + H_NET, 50, 1, 200),
    ]
    .into_iter()
    .flat_map(|r| r.into_iter())
    .collect();
    let search = upload_search_table(&table, &device);
    let meta_t = IdTensor::from_slice(&meta, vec![3, 8], &device).unwrap();
    let mut buffers = ms2::FormulaBuffers::<R, E>::new(3, m, 4, &device);
    ms2::formula_window(
        &search,
        &meta_t,
        table.max_error(),
        u32::MAX,
        u32::MAX,
        &buffers,
    )
    .unwrap();
    ms2::formula_gather(
        &buffers.window,
        &uploaded.table,
        &uploaded.counts,
        &mut buffers.cand,
    )
    .unwrap();
    ms2::count_features(
        &buffers.cand.reshape(vec![3 * m, 13]).unwrap(),
        &uploaded.log_table,
        &mut buffers.cand_feat.reshape(vec![3 * m, 10]).unwrap(),
        13,
    )
    .unwrap();
    check_launches(&device).unwrap();
    let window = buffers.window.try_to_vec().unwrap();
    // Gold rows: two scored rows and one absent formula.
    let gold_rows = vec![2u32, 6, u32::MAX];
    let slots = gold_slots(&window, &gold_rows);
    assert!(
        slots[0] != u32::MAX && slots[1] != u32::MAX,
        "golds are scored"
    );
    let gold_t = IdTensor::from_slice(&slots, vec![3], &device).unwrap();
    let mut rng = Rng::seeded(23);
    let pool_host = rng.uniform_vec(3 * 8, -0.5, 0.5);
    let pool_of =
        |host: &[f32]| Var::constant(Tensor::<R, E>::from_f32(host, vec![3, 8], &device).unwrap());
    let loss_of = |head: &FormulaHead<R, E>, host: &[f32]| {
        let out = head.score(&buffers, &pool_of(host)).unwrap();
        head.loss(&out, &gold_t).unwrap().to_f32()[0]
    };
    let out = head.score(&buffers, &pool_of(&pool_host)).unwrap();
    check_launches(&device).unwrap();
    let loss = head.loss(&out, &gold_t).unwrap();
    let grads = loss.backward_retain().unwrap();
    // The manual mean: both scored spectra contribute.
    let lp = out.log_prob.try_to_f32().unwrap();
    let want = -(lp[slots[0] as usize] + lp[m + slots[1] as usize]) / 2.0;
    assert!(
        (loss.to_f32()[0] - want).abs() <= 1e-5,
        "the loss averages the scored spectra"
    );
    for target in ["row_in.weight", "row_out.weight", "pool_query.weight"] {
        let param = head
            .named_parameters()
            .into_iter()
            .find(|(n, _)| n == target)
            .unwrap_or_else(|| panic!("missing param {target}"))
            .1;
        let analytic = grads.get(param.id()).unwrap().to_f32();
        let shape = param.shape().dims().to_vec();
        let base = param.value().to_f32();
        for &idx in &[0usize, 1, 2] {
            let mut up = base.clone();
            up[idx] += 1e-2;
            param.set(Tensor::<R, E>::from_f32(&up, shape.clone(), &device).unwrap());
            let fu = loss_of(&head, &pool_host);
            let mut down = base.clone();
            down[idx] -= 1e-2;
            param.set(Tensor::<R, E>::from_f32(&down, shape.clone(), &device).unwrap());
            let fd = loss_of(&head, &pool_host);
            param.set(Tensor::<R, E>::from_f32(&base, shape.clone(), &device).unwrap());
            let numeric = (fu - fd) / 2e-2;
            let a = analytic[idx];
            assert!(
                (a - numeric).abs() <= 2e-2 * numeric.abs() + 1e-3,
                "{target}[{idx}]: analytic={a} numeric={numeric}"
            );
        }
    }
    check_launches(&device).unwrap();
    // Spectra whose gold slot is u32::MAX do not move the loss: changing
    // their pool leaves every bit unchanged.
    let before = loss_of(&head, &pool_host);
    let mut changed = pool_host.clone();
    for c in 0..8 {
        changed[2 * 8 + c] += 1.0;
    }
    let after = loss_of(&head, &changed);
    assert_eq!(
        before.to_bits(),
        after.to_bits(),
        "absent-gold spectra do not move the loss"
    );
}

#[test]
fn gold_slots_host_matches_device_window() {
    let device = dev();
    let big =
        FormulaTable::from_compositions(synthetic_compositions(2026, 2200).into_iter()).unwrap();
    let m = 32usize;
    // The random batch of `formula_window_matches_reference`, rebuilt here so
    // this test stands alone: half the spectra target a known row.
    let mut rng = Rng::seeded(99);
    let u = rng.uniform_vec(64 * 3, 0.0, 1.0);
    let mut queries = Vec::with_capacity(64);
    let mut gold_rows = Vec::with_capacity(64);
    let mut meta = Vec::with_capacity(64 * 8);
    for i in 0..64 {
        let (precursor, adduct, gold) = if i % 2 == 0 {
            let row = (u[i * 3] * big.len() as f32) as usize % big.len();
            let ad = if u[i * 3 + 1] < 0.5 || big.mass(row) < H_NET {
                1
            } else {
                2
            };
            let prec = if ad == 1 {
                big.mass(row) + H_NET
            } else {
                big.mass(row) - H_NET
            };
            (prec, ad, row as u32)
        } else {
            (
                50_000_000 + ((u[i * 3] * 450_000_000.0) as u32),
                1,
                u32::MAX,
            )
        };
        queries.push(WindowQuery {
            precursor_mz: precursor,
            adduct: adduct as u16,
            ppm_tenths: 200,
            precursor_uncertainty: 50,
            rows_visited_max: u32::MAX,
            rows_scored_max: 4096,
        });
        gold_rows.push(gold);
        meta.extend_from_slice(&meta_row(precursor, 50, adduct, 200));
    }
    let want = gold_slots_host(&big, &queries, &gold_rows, m);
    // One read of the device window for the whole batch.
    let search = upload_search_table(&big, &device);
    let meta_t = IdTensor::from_slice(&meta, vec![64, 8], &device).unwrap();
    let buffers = ms2::FormulaBuffers::<R, E>::new(64, m, 4, &device);
    ms2::formula_window(&search, &meta_t, big.max_error(), u32::MAX, 4096, &buffers).unwrap();
    check_launches(&device).unwrap();
    let window = buffers.window.try_to_vec().unwrap();
    let got = gold_slots(&window, &gold_rows);
    // `gold_slots_host` caps its queries at min(rows_scored_max, M) exactly
    // like the kernel call above, so every spectrum agrees.
    assert_eq!(got, want, "host slots match the device window");
    assert!(
        want.iter().any(|&s| s != u32::MAX),
        "some gold formulas are scored"
    );
    assert!(
        want.iter().any(|&s| s == u32::MAX),
        "some gold formulas are absent"
    );
}

#[test]
fn device_formula_table_upload_pins_rows_and_hash() {
    let device = dev();
    let table = FormulaTable::from_compositions([comp(0, 2, 0, 1), comp(6, 6, 0, 0)]).unwrap();
    let uploaded = DeviceFormulaTable::<R, E>::upload(&table, &device).unwrap();
    assert_eq!(uploaded.rows, 2);
    assert_eq!(uploaded.max_error, table.max_error());
    assert_eq!(
        uploaded.table.try_to_vec().unwrap(),
        vec![
            table.mass(0),
            composition_error_nda(table.composition(0)).div_ceil(1000) as u32,
            table.mass(1),
            composition_error_nda(table.composition(1)).div_ceil(1000) as u32,
        ]
    );
    let feats = uploaded.features.try_to_f32().unwrap();
    for (row, want) in [
        [0u16, 2, 0, 1, 0, 0, 0, 0, 0, 0],
        [6, 6, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    .iter()
    .enumerate()
    {
        for (e, &c) in want.iter().enumerate() {
            assert!(
                (feats[row * 10 + e] - (1.0 + c as f32).ln()).abs() <= 1e-6,
                "row {row} element {e}"
            );
        }
    }
    // Pinned against the SHA-256 of `FormulaTable::to_json()` (the same
    // bytes `tools/ms2/formula_table.py` hashes).
    assert_eq!(
        uploaded.sha256, "d4a7c80d8824ec0e8804470020d0704ea4fc00bf626382c608c70f3af8f7951e",
        "table hash"
    );
}

#[test]
fn device_formula_table_check_binds_rows_and_hash() {
    let device = dev();
    let table = FormulaTable::from_compositions([comp(0, 2, 0, 1), comp(6, 6, 0, 0)]).unwrap();
    let uploaded = DeviceFormulaTable::<R, E>::upload(&table, &device).unwrap();
    let mut model = ModelConfig::v0();
    model.formula_table.rows = table.len() as u32;
    model.formula_table.sha256 = uploaded.sha256.clone();
    uploaded.check(&model).unwrap();
    // A row-count mismatch names both values.
    let mut wrong_rows = model.clone();
    wrong_rows.formula_table.rows += 1;
    let err = uploaded.check(&wrong_rows).unwrap_err().to_string();
    assert!(
        err.contains(&uploaded.rows.to_string()),
        "names upload rows: {err}"
    );
    assert!(
        err.contains(&wrong_rows.formula_table.rows.to_string()),
        "names model rows: {err}"
    );
    // A hash mismatch names both values.
    let mut wrong_hash = model.clone();
    wrong_hash.formula_table.sha256 = "0".repeat(64);
    let err = uploaded.check(&wrong_hash).unwrap_err().to_string();
    assert!(err.contains(&uploaded.sha256), "names upload hash: {err}");
    assert!(
        err.contains(&wrong_hash.formula_table.sha256),
        "names model hash: {err}"
    );
}

/// Bitwise float comparison, NaN payloads included: the V1 kernels copy
/// floats or read them from the resident `log_table`, so device and twin
/// must agree on every bit, not just within a tolerance.
fn assert_bits_eq(actual: &[f32], expected: &[f32], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            a.to_bits(),
            e.to_bits(),
            "{what}: index {i} got {a}, want {e}"
        );
    }
}

/// A `Result` that must be `Error::Shape`: a malformed shape never panics
/// and never launches.
fn assert_shape<T>(r: mamba3::error::Result<T>, what: &str) {
    assert!(
        matches!(r, Err(mamba3::error::Error::Shape(_))),
        "{what}: expected Error::Shape"
    );
}

/// The `[R, 10]` exact element counts of `table` flat, the `table_counts`
/// input of [`ms2::formula_gather`].
fn table_counts_flat(table: &FormulaTable) -> Vec<u32> {
    let mut counts = Vec::with_capacity(table.len() * 10);
    for row in 0..table.len() {
        for c in table.composition(row).iter() {
            counts.push(u32::from(*c));
        }
    }
    counts
}

/// Run [`ms2::formula_gather`] on a poisoned `cand` and compare with
/// [`twin::formula_gather`] exactly.
fn check_gather(table: &FormulaTable, window: &[u32], batch: usize, m: usize, what: &str) {
    let device = dev();
    let search = upload_search_table(table, &device);
    let counts = table_counts_flat(table);
    let counts_t = IdTensor::from_slice(&counts, vec![table.len(), 10], &device).unwrap();
    let window_t = IdTensor::from_slice(window, vec![batch, m, 2], &device).unwrap();
    let mut out = ms2::FormulaBuffers::<R, E>::poisoned(batch, m, 4, &device).unwrap();
    ms2::formula_gather(&window_t, &search, &counts_t, &mut out.cand).unwrap();
    check_launches(&device).unwrap();
    let got = out.cand.try_to_vec().unwrap();
    let search_flat = search.try_to_vec().unwrap();
    let want = twin::formula_gather(window, &search_flat, &counts, batch, m, table.len());
    assert_eq!(got, want, "{what}: cand differs");
}

#[test]
fn formula_gather_matches_twin() {
    let device = dev();
    let table = FormulaTable::from_compositions(fixture_compositions().into_iter()).unwrap();
    let rows = table.len();
    // Fixture window: two spectra over M = 8 with valid rows under flags 1
    // and 2, a valid row under flag 0 (gathers to padding), MAX/0 padding
    // and out-of-range rows under live flags (also padding, no OOB read).
    let window: Vec<u32> = vec![
        0, 1, 1, 2, u32::MAX, 0, 2, 0, rows as u32 + 5, 1, 3, 1, u32::MAX, 0, 4, 2,
        5, 1, u32::MAX, 0, 6, 0, rows as u32 + 100, 2, 7, 1, u32::MAX, 0, u32::MAX, 0, 8, 1,
    ];
    check_gather(&table, &window, 2, 8, "fixture");
    // The flag-0 and out-of-range slots really are padding.
    let search_flat = upload_search_table(&table, &device)
        .try_to_vec()
        .unwrap();
    let want = twin::formula_gather(
        &window,
        &search_flat,
        &table_counts_flat(&table),
        2,
        8,
        rows,
    );
    // Slot (0, 3): valid row under flag 0 gives zero counts and MAX source.
    assert_eq!(&want[3 * 13..4 * 13], &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, u32::MAX]);
    // Slot (0, 4): out-of-range row under a live flag gives the same padding.
    assert_eq!(&want[4 * 13..5 * 13], &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, u32::MAX]);
    // Slot (0, 0): a live row carries its counts, mass, flag and source.
    let c0: Vec<u32> = table.composition(0).iter().map(|&c| u32::from(c)).collect();
    assert_eq!(&want[0..10], &c0[..]);
    assert_eq!(want[10], table.mass(0));
    assert_eq!(want[11], 1);
    assert_eq!(want[12], 0);

    // Randomised windows: R = 16 rows, B = 3 spectra, M = 12 slots, with
    // rows past the end and MAX rows sprinkled in.
    let big =
        FormulaTable::from_compositions(synthetic_compositions(4242, 24).into_iter()).unwrap();
    let r = big.len();
    let (b, m) = (3usize, 12usize);
    let mut rng = Rng::seeded(4243);
    let u = rng.uniform_vec(b * m * 3, 0.0, 1.0);
    let mut rand_window = Vec::with_capacity(b * m * 2);
    for i in 0..b * m {
        let row = if u[i * 3 + 2] < 0.1 {
            u32::MAX
        } else {
            ((u[i * 3] * (r + 3) as f32) as usize).min(r + 2) as u32
        };
        let flag = (u[i * 3 + 1] * 3.0) as u32;
        rand_window.push(row);
        rand_window.push(flag);
    }
    check_gather(&big, &rand_window, b, m, "random");

    // Shape errors are Error::Shape, never panics.
    let search = upload_search_table(&table, &device);
    let counts = table_counts_flat(&table);
    let counts_t = IdTensor::from_slice(&counts, vec![rows, 10], &device).unwrap();
    let window_t = IdTensor::from_slice(&window, vec![2, 8, 2], &device).unwrap();
    // Window rank 2 instead of [B, M, 2].
    let flat = IdTensor::from_slice(&window, vec![2 * 8, 2], &device).unwrap();
    let mut cand = IdTensor::empty(vec![2, 8, 13], &device);
    assert_shape(
        ms2::formula_gather(&flat, &search, &counts_t, &mut cand),
        "window rank",
    );
    // Window last dimension 3.
    let wide = IdTensor::from_slice(&vec![0u32; 2 * 8 * 3], vec![2, 8, 3], &device).unwrap();
    assert_shape(
        ms2::formula_gather(&wide, &search, &counts_t, &mut cand),
        "window last dim",
    );
    // Table [R, 3] instead of [R, 2].
    let bad_table = IdTensor::from_slice(&vec![0u32; rows * 3], vec![rows, 3], &device).unwrap();
    assert_shape(
        ms2::formula_gather(&window_t, &bad_table, &counts_t, &mut cand),
        "table dims",
    );
    // Counts [R, 9] instead of [R, 10].
    let bad_counts = IdTensor::from_slice(&vec![0u32; rows * 9], vec![rows, 9], &device).unwrap();
    assert_shape(
        ms2::formula_gather(&window_t, &search, &bad_counts, &mut cand),
        "counts dims",
    );
    // Cand [B, M, 12] instead of [B, M, 13].
    let mut bad_cand = IdTensor::empty(vec![2, 8, 12], &device);
    assert_shape(
        ms2::formula_gather(&window_t, &search, &counts_t, &mut bad_cand),
        "cand dims",
    );
}

#[test]
fn formula_gather_empty_table_writes_padding() {
    // B1-fix finding 1: `rows == 0` with a fully padded window (B = 1,
    // M = 32). No table element may be loaded; all 13 words of every
    // candidate are still written (padding). The wrapper takes the
    // padding-only path and never binds a zero-length array to a kernel
    // that indexes it; the poisoned output proves every word is written.
    let device = dev();
    let (batch, m) = (1usize, 32usize);
    let window = vec![u32::MAX, 0u32]
        .into_iter()
        .cycle()
        .take(batch * m * 2)
        .collect::<Vec<u32>>();
    // A mixed window too: live flags and row ids with no valid fallback
    // index in an empty array must still gather to padding.
    let mut mixed = window.clone();
    mixed[0] = 0;
    mixed[1] = 1;
    mixed[2] = 7;
    mixed[3] = 2;
    for (what, win) in [("padded", window), ("mixed", mixed)] {
        let table = FormulaTable::from_compositions(std::iter::empty()).unwrap();
        assert!(table.is_empty());
        let search = upload_search_table(&table, &device);
        assert_eq!(search.shape().dims(), &[0, 2]);
        let counts_t = IdTensor::from_slice(&[], vec![0, 10], &device).unwrap();
        let window_t = IdTensor::from_slice(&win, vec![batch, m, 2], &device).unwrap();
        let mut out = ms2::FormulaBuffers::<R, E>::poisoned(batch, m, 4, &device).unwrap();
        ms2::formula_gather(&window_t, &search, &counts_t, &mut out.cand).unwrap();
        check_launches(&device).unwrap();
        let got = out.cand.try_to_vec().unwrap();
        let want = twin::formula_gather(&win, &[], &[], batch, m, 0);
        assert_eq!(got, want, "{what}: cand differs");
        for mm in 0..m {
            assert_eq!(
                &got[mm * 13..mm * 13 + 13],
                &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, u32::MAX],
                "{what}: slot {mm} is not padding"
            );
        }
    }
}

/// Run [`ms2::count_features`] on NaN-poisoned output and compare with
/// [`twin::count_features`] bit-for-bit.
fn check_count_features(records: &[u32], log_host: &[f32], n: usize, w: usize, what: &str) {
    let device = dev();
    let rec_t = IdTensor::from_slice(records, vec![n, w], &device).unwrap();
    let log_t = Tensor::<R, E>::from_f32(log_host, vec![1024], &device).unwrap();
    let mut out_t =
        Tensor::<R, E>::from_f32(&vec![f32::NAN; n * 10], vec![n, 10], &device).unwrap();
    ms2::count_features(&rec_t, &log_t, &mut out_t, w).unwrap();
    check_launches(&device).unwrap();
    let got = out_t.try_to_f32().unwrap();
    let want = twin::count_features(records, log_host, n, w);
    assert_bits_eq(&got, &want, what);
}

#[test]
fn count_features_matches_twin_and_v0_bits() {
    let device = dev();
    let log_host = twin::log_table();
    assert_eq!(log_host.len(), 1024, "the resident table has 1024 entries");
    // Fixture records at three widths: cand-like (w = 13, counts plus
    // mass/flag/source filler), gold-like (w = 10) and w = 12. Counts cover
    // 0, small values, the 1023 bound and the 1024/u32::MAX edge, which must
    // read no entry and write exact 0.
    let counts_rows: Vec<[u32; 10]> = vec![
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1023, 1023, 1023, 1023, 1023, 1023, 1023, 1023, 1023, 1023],
        [1024, u32::MAX, 0, 1, 2, 3, 4, 5, 6, 7],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ];
    let mut rec13 = Vec::with_capacity(4 * 13);
    for (i, row) in counts_rows.iter().enumerate() {
        rec13.extend_from_slice(row);
        rec13.extend_from_slice(&[100 + i as u32, 1, 7]); // mass, flag, source
    }
    let rec10: Vec<u32> = counts_rows.iter().flat_map(|r| r.iter().copied()).collect();
    let mut rec12 = Vec::with_capacity(4 * 12);
    for (i, row) in counts_rows.iter().enumerate() {
        rec12.extend_from_slice(row);
        rec12.extend_from_slice(&[100 + i as u32, 7]);
    }
    check_count_features(&rec13, &log_host, 4, 13, "fixture w=13");
    check_count_features(&rec10, &log_host, 4, 10, "fixture w=10");
    check_count_features(&rec12, &log_host, 4, 12, "fixture w=12");
    // The edge row really is zeros past the bound.
    let edge = twin::count_features(&rec10, &log_host, 4, 10);
    assert_eq!(edge[2 * 10], 0.0, "count 1024 gives 0");
    assert_eq!(edge[2 * 10 + 1], 0.0, "count u32::MAX gives 0");
    assert_eq!(edge[1 * 10], log_host[1023], "count 1023 reads the last entry");

    // Randomised counts in 0..1030 (past the bound) with MAX sprinkled in.
    let mut rng = Rng::seeded(77);
    let u = rng.uniform_vec(64 * 13 * 2, 0.0, 1.0);
    let mut rand_rec = Vec::with_capacity(64 * 13);
    for i in 0..64 * 13 {
        let e = i % 13;
        if e < 10 {
            let c = (u[i] * 1030.0) as u32;
            rand_rec.push(if u[64 * 13 + i] < 0.05 { u32::MAX } else { c });
        } else {
            rand_rec.push((u[i] * 1000.0) as u32); // filler words
        }
    }
    check_count_features(&rand_rec, &log_host, 64, 13, "random w=13");

    // V0 parity: the kernel over the uploaded exact counts is the uploaded
    // float features bit-for-bit.
    let small = FormulaTable::from_compositions(
        [comp(0, 2, 0, 1), comp(6, 6, 0, 0), comp(2, 6, 0, 1)].into_iter(),
    )
    .unwrap();
    let uploaded = DeviceFormulaTable::<R, E>::upload(&small, &device).unwrap();
    let n = small.len();
    let rec = uploaded.counts.reshape(vec![n, 10]).unwrap();
    let mut out =
        Tensor::<R, E>::from_f32(&vec![f32::NAN; n * 10], vec![n, 10], &device).unwrap();
    ms2::count_features(&rec, &uploaded.log_table, &mut out, 10).unwrap();
    check_launches(&device).unwrap();
    assert_bits_eq(
        &out.try_to_f32().unwrap(),
        &uploaded.features.try_to_f32().unwrap(),
        "v0 parity",
    );

    // Shape errors are Error::Shape, never panics or OOB reads.
    let rec_t = IdTensor::from_slice(&rec10, vec![4, 10], &device).unwrap();
    let log_t = Tensor::<R, E>::from_f32(&log_host, vec![1024], &device).unwrap();
    let mut out_t = Tensor::<R, E>::from_f32(&vec![0.0; 40], vec![4, 10], &device).unwrap();
    // Record width below 10.
    let narrow = IdTensor::from_slice(&vec![0u32; 4 * 9], vec![4, 9], &device).unwrap();
    assert_shape(
        ms2::count_features(&narrow, &log_t, &mut out_t, 9),
        "width below 10",
    );
    // Width argument disagreeing with the record shape.
    let wide_rec = IdTensor::from_slice(&vec![0u32; 4 * 13], vec![4, 13], &device).unwrap();
    assert_shape(
        ms2::count_features(&wide_rec, &log_t, &mut out_t, 10),
        "width arg mismatch",
    );
    // Resident table of 1023 instead of 1024.
    let short_log =
        Tensor::<R, E>::from_f32(&log_host[..1023], vec![1023], &device).unwrap();
    assert_shape(
        ms2::count_features(&rec_t, &short_log, &mut out_t, 10),
        "log_table length",
    );
    // Output last dimension 9 instead of 10.
    let mut narrow_out = Tensor::<R, E>::from_f32(&vec![0.0; 36], vec![4, 9], &device).unwrap();
    assert_shape(
        ms2::count_features(&rec_t, &log_t, &mut narrow_out, 10),
        "output dims",
    );
}

/// Run [`ms2::formula_top`] on poisoned top buffers with `cand` from the
/// gather twin, and compare with [`twin::formula_top_from_cand`] exactly
/// (integer top, bitwise log-probs, counts).
fn check_top_from_cand(
    log_prob: &[f32],
    cand: &[u32],
    batch: usize,
    m: usize,
    f: usize,
    what: &str,
) -> (Vec<u32>, Vec<f32>, Vec<u32>) {
    let device = dev();
    let lp_t = Tensor::<R, E>::from_f32(log_prob, vec![batch, m], &device).unwrap();
    let cand_t = IdTensor::from_slice(cand, vec![batch, m, 13], &device).unwrap();
    let out = ms2::FormulaBuffers::<R, E>::poisoned(batch, m, f, &device).unwrap();
    ms2::formula_top(&lp_t, &cand_t, &out).unwrap();
    check_launches(&device).unwrap();
    let (got_top, got_lp, got_count) = (
        out.top.try_to_vec().unwrap(),
        out.top_log_prob.try_to_f32().unwrap(),
        out.top_count.try_to_vec().unwrap(),
    );
    let (want_top, want_lp, want_count) =
        twin::formula_top_from_cand(log_prob, cand, batch, m, f);
    assert_eq!(got_top, want_top, "{what}: top rows/slots differ");
    assert_bits_eq(&got_lp, &want_lp, &format!("{what}: top log-probs"));
    assert_eq!(got_count, want_count, "{what}: top counts differ");
    (got_top, got_lp, got_count)
}

#[test]
fn formula_top_from_cand_matches_twin() {
    let device = dev();
    // Fixture: the three-row cluster joins slots 0..2 of spectrum 0 while
    // spectrum 1 is empty. The log-probs hold a tie (slots 0 and 2) and a
    // NaN at flagged slot 1, which is never selected; the empty spectrum
    // must report count 0 with MAX/0 padding.
    let (table, precursor) = cluster_table();
    let (m, f) = (6usize, 4usize);
    let meta: Vec<u32> = vec![
        meta_row(precursor, 50, 1, 1000),
        meta_row(1_000_000 + H_NET, 50, 1, 200),
    ]
    .into_iter()
    .flat_map(|r| r.into_iter())
    .collect();
    let (window, _) =
        twin::formula_window(&table, &meta, 2, m, table.max_error(), u32::MAX, u32::MAX);
    let search_flat = upload_search_table(&table, &device)
        .try_to_vec()
        .unwrap();
    let cand = twin::formula_gather(
        &window,
        &search_flat,
        &table_counts_flat(&table),
        2,
        m,
        table.len(),
    );
    let log_prob: Vec<f32> = vec![
        0.5, f32::NAN, 0.5, -99.0, -99.0, -99.0, // tie at slots 0 and 2, NaN at 1
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // empty support: values are irrelevant
    ];
    let (top, lp, count) = check_top_from_cand(&log_prob, &cand, 2, m, f, "fixture");
    assert_eq!(count[0], 2, "tie slots 0 and 2 win, NaN never selected");
    assert_eq!(
        &[top[1], top[3]],
        &[0, 2],
        "the NaN slot 1 is never selected"
    );
    assert_eq!(count[1], 0, "the empty spectrum fills no entry");
    assert_eq!(
        &top[f * 2..],
        &[u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX],
        "the empty spectrum pads everything"
    );
    assert_eq!(&lp[f..], &[0.0, 0.0, 0.0, 0.0], "padding log-probs are 0");
    // Every written slot of spectrum 0 is below its count and holds a real
    // row; the NaN slot is never selected and nothing is written out of
    // bounds.
    for pick in 0..count[0] as usize {
        assert_ne!(top[pick * 2], u32::MAX, "pick {pick} is a real row");
    }

    // Randomised: a wide-uncertainty search over a synthetic table joins a
    // band of rows per hit spectrum; quantised log-probs force ties and
    // every ninth slot is NaN. The last spectrum is absent (empty support).
    let big =
        FormulaTable::from_compositions(synthetic_compositions(5150, 256).into_iter()).unwrap();
    let (b, m, f) = (4usize, 16usize, 4usize);
    let mut meta = Vec::with_capacity(b * 8);
    for i in 0..3 {
        let row = big.len() * (i + 1) / 4;
        meta.extend_from_slice(&meta_row(big.mass(row) + H_NET, 10_000_000, 1, 1000));
    }
    meta.extend_from_slice(&meta_row(1_000_000 + H_NET, 50, 1, 200));
    let (window, _) =
        twin::formula_window(&big, &meta, b, m, big.max_error(), u32::MAX, u32::MAX);
    let big_search = upload_search_table(&big, &device)
        .try_to_vec()
        .unwrap();
    let cand = twin::formula_gather(
        &window,
        &big_search,
        &table_counts_flat(&big),
        b,
        m,
        big.len(),
    );
    let mut rng = Rng::seeded(5151);
    let u = rng.uniform_vec(b * m, -2.0, 0.0);
    let log_prob: Vec<f32> = u
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            if i % 9 == 4 {
                f32::NAN
            } else {
                (v * 2.0).round() / 2.0 // quantise to force ties
            }
        })
        .collect();
    let (_, _, count) = check_top_from_cand(&log_prob, &cand, b, m, f, "random");
    assert_eq!(count[3], 0, "the absent spectrum fills no entry");

    // Shape errors are Error::Shape, never panics.
    let lp_t = Tensor::<R, E>::from_f32(&log_prob, vec![b, m], &device).unwrap();
    let cand_t = IdTensor::from_slice(&cand, vec![b, m, 13], &device).unwrap();
    let out = ms2::FormulaBuffers::<R, E>::poisoned(b, m, f, &device).unwrap();
    // Log-prob one slot too wide.
    let bad_lp = Tensor::<R, E>::from_f32(&vec![0.0; b * (m + 1)], vec![b, m + 1], &device).unwrap();
    assert_shape(ms2::formula_top(&bad_lp, &cand_t, &out), "log_prob dims");
    // Log-prob rank 3.
    let rank_lp =
        Tensor::<R, E>::from_f32(&vec![0.0; b * m], vec![b, m, 1], &device).unwrap();
    assert_shape(ms2::formula_top(&rank_lp, &cand_t, &out), "log_prob rank");
    // Cand last dimension 12.
    let bad_cand = IdTensor::from_slice(&vec![0u32; b * m * 12], vec![b, m, 12], &device).unwrap();
    assert_shape(ms2::formula_top(&lp_t, &bad_cand, &out), "cand dims");
    // Top buffers one entry too wide (top [B, F+1, 2] against
    // top_log_prob [B, F]): hand-mixed fields, since one `poisoned` call is
    // always self-consistent.
    let base = ms2::FormulaBuffers::<R, E>::poisoned(b, m, f, &device).unwrap();
    let wide_top =
        IdTensor::from_slice(&vec![0xDEAD_BEEFu32; b * (f + 1) * 2], vec![b, f + 1, 2], &device)
            .unwrap();
    let mixed = ms2::FormulaBuffers::<R, E> {
        window: base.window,
        counters: base.counters,
        top: wide_top,
        top_log_prob: base.top_log_prob,
        top_count: base.top_count,
        cand: base.cand,
        cand_feat: base.cand_feat,
        top_counts: base.top_counts,
    };
    assert_shape(ms2::formula_top(&lp_t, &cand_t, &mixed), "top dims");
}

/// Run [`ms2::formula_top_counts`] on poisoned output and compare with
/// [`twin::formula_top_counts`] exactly.
fn check_top_counts(top: &[u32], cand: &[u32], batch: usize, m: usize, f: usize, what: &str) {
    let device = dev();
    let top_t = IdTensor::from_slice(top, vec![batch, f, 2], &device).unwrap();
    let cand_t = IdTensor::from_slice(cand, vec![batch, m, 13], &device).unwrap();
    let mut out_t =
        IdTensor::from_slice(&vec![0xDEAD_BEEF; batch * f * 10], vec![batch, f, 10], &device)
            .unwrap();
    ms2::formula_top_counts(&top_t, &cand_t, &mut out_t).unwrap();
    check_launches(&device).unwrap();
    let got = out_t.try_to_vec().unwrap();
    let want = twin::formula_top_counts(top, cand, batch, m, f);
    assert_eq!(got, want, "{what}: top_counts differ");
}

#[test]
fn formula_top_counts_matches_twin() {
    let device = dev();
    // Fixture: cand from a real gather over M = 8; the top holds valid
    // slots, MAX padding and an out-of-range slot (M + 2). The source ids
    // are arbitrary: the kernel reads only the slot.
    let table = FormulaTable::from_compositions(fixture_compositions().into_iter()).unwrap();
    let (b, m, f) = (2usize, 8usize, 4usize);
    let window: Vec<u32> = vec![
        0, 1, 1, 2, u32::MAX, 0, 2, 1, 3, 0, 4, 2, u32::MAX, 0, 5, 1,
        6, 1, 7, 0, u32::MAX, 0, 8, 2, 9, 1, u32::MAX, 0, u32::MAX, 0, 10, 1,
    ];
    let search_flat = upload_search_table(&table, &device)
        .try_to_vec()
        .unwrap();
    let cand = twin::formula_gather(
        &window,
        &search_flat,
        &table_counts_flat(&table),
        b,
        m,
        table.len(),
    );
    let top: Vec<u32> = vec![
        0, 0, 0, 5, u32::MAX, u32::MAX, 0, (m + 2) as u32, // valid, valid, pad, OOB
        0, 1, u32::MAX, u32::MAX, 0, 6, 0, 3, // valid, pad, valid, valid
    ];
    check_top_counts(&top, &cand, b, m, f, "fixture");
    // The padding and out-of-range entries really are zeros.
    let want = twin::formula_top_counts(&top, &cand, b, m, f);
    assert_eq!(&want[2 * 10..3 * 10], &[0; 10], "MAX slot pads with 0");
    assert_eq!(&want[3 * 10..4 * 10], &[0; 10], "out-of-range slot pads with 0");

    // Randomised top/cand: valid slots, MAX padding and out-of-range slots.
    let (b, m, f) = (3usize, 12usize, 4usize);
    let mut rng = Rng::seeded(31);
    let u = rng.uniform_vec(b * m * 2 + b * f, 0.0, 1.0);
    let mut rand_window = Vec::with_capacity(b * m * 2);
    for i in 0..b * m {
        let row = if u[i * 2 + 1] < 0.15 {
            u32::MAX
        } else {
            ((u[i * 2] * table.len() as f32) as usize % table.len()) as u32
        };
        rand_window.push(row);
        rand_window.push((u[i * 2] * 3.0) as u32);
    }
    let cand = twin::formula_gather(
        &rand_window,
        &search_flat,
        &table_counts_flat(&table),
        b,
        m,
        table.len(),
    );
    let mut rand_top = Vec::with_capacity(b * f * 2);
    for i in 0..b * f {
        let v = u[b * m * 2 + i];
        let slot = if v < 0.6 {
            (v * 100.0) as u32 % m as u32 // valid slot
        } else if v < 0.8 {
            u32::MAX // padding
        } else {
            m as u32 + 1 + ((v * 100.0) as u32 % 3) // out of range
        };
        rand_top.push(0); // source id is ignored by the kernel
        rand_top.push(slot);
    }
    check_top_counts(&rand_top, &cand, b, m, f, "random");

    // Shape errors are Error::Shape, never panics.
    let top_t = IdTensor::from_slice(&top, vec![2, 4, 2], &device).unwrap();
    let cand_t = IdTensor::from_slice(&cand, vec![b, m, 13], &device).unwrap();
    let mut out_t = IdTensor::empty(vec![2, 4, 10], &device);
    // Top last dimension 3. (The [2, 4] batch/F prefix matches `out_t` so
    // the failure is the rank-3 width, not a batch mismatch.)
    let bad_top = IdTensor::from_slice(&vec![0u32; 2 * 4 * 3], vec![2, 4, 3], &device).unwrap();
    let cand2 = IdTensor::from_slice(&cand[..2 * m * 13], vec![2, m, 13], &device).unwrap();
    assert_shape(
        ms2::formula_top_counts(&bad_top, &cand2, &mut out_t),
        "top dims",
    );
    // Cand last dimension 12.
    let bad_cand = IdTensor::from_slice(&vec![0u32; 2 * m * 12], vec![2, m, 12], &device).unwrap();
    assert_shape(
        ms2::formula_top_counts(&top_t, &bad_cand, &mut out_t),
        "cand dims",
    );
    // Output last dimension 9. (Cand/top agree on B = 3 here.)
    let top3 = IdTensor::from_slice(&rand_top, vec![b, f, 2], &device).unwrap();
    let mut bad_out = IdTensor::empty(vec![b, f, 9], &device);
    assert_shape(
        ms2::formula_top_counts(&top3, &cand_t, &mut bad_out),
        "top_counts dims",
    );
    // Cand rank 2.
    let flat_cand = IdTensor::from_slice(&cand[..2 * m * 13], vec![2 * m, 13], &device).unwrap();
    assert_shape(
        ms2::formula_top_counts(&top_t, &flat_cand, &mut out_t),
        "cand rank",
    );
}

/// Run [`ms2::gold_slot`] on poisoned output and compare with
/// [`twin::gold_slot`] exactly; returns the device slots.
fn check_gold(cand: &[u32], gold: &[u32], batch: usize, m: usize, what: &str) -> Vec<u32> {
    let device = dev();
    let cand_t = IdTensor::from_slice(cand, vec![batch, m, 13], &device).unwrap();
    let gold_t = IdTensor::from_slice(gold, vec![batch, 10], &device).unwrap();
    let mut out =
        IdTensor::from_slice(&vec![0xDEAD_BEEFu32; batch], vec![batch], &device).unwrap();
    ms2::gold_slot(&cand_t, &gold_t, &mut out).unwrap();
    check_launches(&device).unwrap();
    let got = out.try_to_vec().unwrap();
    let want = twin::gold_slot(cand, gold, batch, m);
    assert_eq!(got, want, "{what}: gold slots differ");
    got
}

/// The 10 `u32` counts of one table row.
fn row_counts(table: &FormulaTable, row: usize) -> Vec<u32> {
    table.composition(row).iter().map(|&c| u32::from(c)).collect()
}

#[test]
fn gold_slot_matches_twin_and_edges() {
    let device = dev();
    // M = 8 over the three-row cluster: 3 scored slots then padding.
    let (table, precursor) = cluster_table();
    let m = 8usize;
    let meta: Vec<u32> = vec![meta_row(precursor, 50, 1, 1000)]
        .into_iter()
        .flat_map(|r| r.into_iter())
        .collect();
    let (window, _) =
        twin::formula_window(&table, &meta, 1, m, table.max_error(), u32::MAX, u32::MAX);
    let search_flat = upload_search_table(&table, &device)
        .try_to_vec()
        .unwrap();
    let cand = twin::formula_gather(
        &window,
        &search_flat,
        &table_counts_flat(&table),
        1,
        m,
        table.len(),
    );
    // Gold at slot 0 (the first joined row) and at the last scored slot.
    let slot0_row = window[0] as usize;
    assert_eq!(window[1], 1, "slot 0 is flagged");
    assert_eq!(check_gold(&cand, &row_counts(&table, slot0_row), 1, m, "slot 0"), vec![0]);
    let slot2_row = window[2 * 2] as usize;
    assert_eq!(window[2 * 2 + 1], 1, "slot 2 is flagged");
    assert_eq!(check_gold(&cand, &row_counts(&table, slot2_row), 1, m, "last slot"), vec![2]);
    // Gold matching the padding counts (all zero, flag 0) is MAX: padding
    // never matches, even though its counts equal the query.
    assert_eq!(check_gold(&cand, &vec![0u32; 10], 1, m, "padding"), vec![u32::MAX]);
    // An absent composition is MAX.
    let absent: Vec<u32> = comp(9, 9, 9, 9).iter().map(|&c| u32::from(c)).collect();
    assert_eq!(check_gold(&cand, &absent, 1, m, "absent"), vec![u32::MAX]);

    // Randomised golds over a dense table with provably unique inputs:
    // `from_compositions` dedups, so assert via a HashSet that the dedup
    // drops nothing and every gold can match at most one row.
    let mut comps = fixture_compositions();
    comps.extend(synthetic_compositions(5150, 300).into_iter());
    let mut seen = std::collections::HashSet::new();
    comps.retain(|c| seen.insert(*c));
    assert!(comps.len() > 200, "the dense table has rows");
    let big = FormulaTable::from_compositions(comps.into_iter()).unwrap();
    assert_eq!(big.len(), seen.len(), "no duplicate composition was dropped");
    let (b, m) = (8usize, 16usize);
    let mut rng = Rng::seeded(5152);
    let u = rng.uniform_vec(b * 3, 0.0, 1.0);
    let mut meta = Vec::with_capacity(b * 8);
    let mut gold_rows = Vec::with_capacity(b);
    for i in 0..b {
        if i % 2 == 0 {
            let row = (u[i * 3] * big.len() as f32) as usize % big.len();
            meta.extend_from_slice(&meta_row(big.mass(row) + H_NET, 50, 1, 200));
            gold_rows.push(row);
        } else {
            meta.extend_from_slice(&meta_row(50_000_000 + ((u[i * 3] * 400_000_000.0) as u32), 50, 1, 200));
            gold_rows.push(u32::MAX as usize); // absent gold, repaired below
        }
    }
    let (window, _) =
        twin::formula_window(&big, &meta, b, m, big.max_error(), u32::MAX, u32::MAX);
    let big_search = upload_search_table(&big, &device).try_to_vec().unwrap();
    let cand = twin::formula_gather(&window, &big_search, &table_counts_flat(&big), b, m, big.len());
    // Golds: a scored slot's own composition where one is scored (found by
    // scanning the flags), else an composition absent from the table.
    let mut gold = Vec::with_capacity(b * 10);
    let mut expect = Vec::with_capacity(b);
    for bi in 0..b {
        let scored: Vec<usize> = (0..m).filter(|&s| window[(bi * m + s) * 2 + 1] != 0).collect();
        if gold_rows[bi] != u32::MAX as usize && !scored.is_empty() {
            // The gold row's own slot when it is scored, else MAX.
            let row = gold_rows[bi];
            let slot = scored
                .iter()
                .find(|&&s| window[(bi * m + s) * 2] as usize == row)
                .copied();
            gold.extend_from_slice(&row_counts(&big, row));
            expect.push(slot.map(|s| s as u32).unwrap_or(u32::MAX));
        } else {
            gold.extend_from_slice(&absent);
            expect.push(u32::MAX);
        }
    }
    assert_eq!(check_gold(&cand, &gold, b, m, "random"), expect);
    assert!(expect.iter().any(|&s| s != u32::MAX), "some golds are scored");
    assert!(expect.iter().any(|&s| s == u32::MAX), "some golds are absent");

    // Shape errors are Error::Shape, never panics.
    let cand_t = IdTensor::from_slice(&cand, vec![b, m, 13], &device).unwrap();
    let gold_t = IdTensor::from_slice(&gold, vec![b, 10], &device).unwrap();
    let mut out = IdTensor::from_slice(&vec![0u32; b], vec![b], &device).unwrap();
    // Cand last dimension 12.
    let bad_cand = IdTensor::from_slice(&vec![0u32; b * m * 12], vec![b, m, 12], &device).unwrap();
    assert_shape(ms2::gold_slot(&bad_cand, &gold_t, &mut out), "cand dims");
    // Gold counts width 9.
    let bad_gold = IdTensor::from_slice(&vec![0u32; b * 9], vec![b, 9], &device).unwrap();
    assert_shape(ms2::gold_slot(&cand_t, &bad_gold, &mut out), "gold dims");
    // Output one element too long.
    let mut bad_out = IdTensor::from_slice(&vec![0u32; b + 1], vec![b + 1], &device).unwrap();
    assert_shape(ms2::gold_slot(&cand_t, &gold_t, &mut bad_out), "output len");
    // Cand rank 2.
    let flat = IdTensor::from_slice(&cand, vec![b * m, 13], &device).unwrap();
    assert_shape(ms2::gold_slot(&flat, &gold_t, &mut out), "cand rank");
}

#[test]
fn gold_slot_first_match_wins_on_duplicates() {
    // B1-fix finding 4: synthetic duplicate candidates built directly in
    // the candidate buffer — the first flagged match wins, even when a
    // later slot holds the same 10 counts.
    let device = dev();
    let (batch, m) = (2usize, 6usize);
    let gold_a: Vec<u32> = vec![6, 12, 0, 6, 0, 0, 0, 0, 0, 0];
    let gold_b: Vec<u32> = vec![3, 7, 1, 2, 0, 0, 0, 0, 0, 0];
    let other: Vec<u32> = vec![1, 4, 0, 0, 0, 0, 0, 0, 0, 0];
    let mut cand = vec![0u32; batch * m * 13];
    let put = |cand: &mut Vec<u32>, b: usize, mm: usize, counts: &[u32], flag: u32, src: u32| {
        let base = (b * m + mm) * 13;
        cand[base..base + 10].copy_from_slice(counts);
        cand[base + 10] = 100_000_000;
        cand[base + 11] = flag;
        cand[base + 12] = src;
    };
    // Spectrum 0: duplicates at slots 1 and 3 (both flagged), a decoy at 0,
    // padding at 2/4/5. First match (slot 1) must win.
    put(&mut cand, 0, 0, &other, 1, 9);
    put(&mut cand, 0, 1, &gold_a, 1, 3);
    put(&mut cand, 0, 2, &gold_a, 0, u32::MAX);
    put(&mut cand, 0, 3, &gold_a, 2, 5);
    put(&mut cand, 0, 4, &gold_a, 0, u32::MAX);
    put(&mut cand, 0, 5, &other, 0, u32::MAX);
    // Spectrum 1: the only flagged match is at slot 4 (ambiguous flag 2);
    // an unflagged identical row at slot 0 must not win.
    put(&mut cand, 1, 0, &gold_b, 0, u32::MAX);
    put(&mut cand, 1, 1, &other, 1, 1);
    put(&mut cand, 1, 2, &gold_b, 0, u32::MAX);
    put(&mut cand, 1, 3, &other, 0, u32::MAX);
    put(&mut cand, 1, 4, &gold_b, 2, 2);
    put(&mut cand, 1, 5, &other, 0, u32::MAX);
    let mut gold = vec![0u32; batch * 10];
    gold[..10].copy_from_slice(&gold_a);
    gold[10..].copy_from_slice(&gold_b);
    assert_eq!(check_gold(&cand, &gold, batch, m, "duplicates"), vec![1, 4]);
}

#[test]
fn cand_mask_matches_twin_every_element() {
    // B1-fix finding 4: direct poisoned-output every-element comparison of
    // `cand_mask` against its host twin, including ambiguous flags (2):
    // flags 1 and 2 both map to 1.0, flag 0 to 0.0. The kernel writes into
    // a caller-provided output poisoned with NaN first, so a lane the
    // kernel skips keeps a NaN and fails the comparison below.
    let device = dev();
    let (batch, m) = (2usize, 8usize);
    let mut cand = vec![0u32; batch * m * 13];
    let flags = [0u32, 1, 2, 0, 2, 1, 0, 3, 2, 0, 1, 0, 0, 2, 1, 7];
    for (i, &f) in flags.iter().enumerate() {
        let b = i / m;
        let mm = i % m;
        let base = (b * m + mm) * 13;
        for e in 0..10 {
            cand[base + e] = (i * 10 + e) as u32;
        }
        cand[base + 10] = 50_000_000 + i as u32;
        cand[base + 11] = f;
        cand[base + 12] = i as u32;
    }
    let cand_t = IdTensor::from_slice(&cand, vec![batch, m, 13], &device).unwrap();
    // Poisoned caller-provided output through the `cand_mask_into` launch
    // path: every lane must be overwritten, including zero-flag lanes.
    let mut out =
        Tensor::<R, E>::from_f32(&vec![f32::NAN; batch * m], vec![batch, m], &device).unwrap();
    ms2::cand_mask_into(&cand_t, &mut out).unwrap();
    check_launches(&device).unwrap();
    let got = out.try_to_f32().unwrap();
    let want = twin::cand_mask(&cand, batch, m);
    assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert_eq!(g.to_bits(), w.to_bits(), "mask element {i} (flag {})", flags[i]);
    }
    // Ambiguous flags really map to 1.0.
    for (i, &f) in flags.iter().enumerate() {
        let want_one = f != 0;
        assert_eq!(
            got[i].to_bits(),
            (if want_one { 1.0f32 } else { 0.0f32 }).to_bits(),
            "flag {f} at {i}"
        );
    }
    let flat = IdTensor::from_slice(&cand, vec![batch * m, 13], &device).unwrap();
    assert!(ms2::cand_mask::<R, E>(&flat).is_err());
    let narrow = IdTensor::from_slice(&cand[..batch * m * 12], vec![batch, m, 12], &device).unwrap();
    assert!(ms2::cand_mask::<R, E>(&narrow).is_err());
    // A caller-provided output of the wrong shape is refused, never
    // partially written.
    let mut short =
        Tensor::<R, E>::from_f32(&vec![0.0; batch * (m - 1)], vec![batch, m - 1], &device).unwrap();
    assert!(ms2::cand_mask_into(&cand_t, &mut short).is_err());
}

/// Device-vs-twin comparison of the V1 §1.2 table-source pipeline
/// (window → gather → top → top_counts) for one `(M, rows_scored_max)`
/// point; returns the device counters for the capacity assertions.
#[allow(clippy::too_many_arguments)]
fn check_table_pipeline(
    table: &FormulaTable,
    meta: &[u32],
    batch: usize,
    m: usize,
    f: usize,
    log_prob: &[f32],
    rows_visited_max: u32,
    rows_scored_max: u32,
    what: &str,
) -> Vec<u32> {
    let device = dev();
    let search = upload_search_table(table, &device);
    let counts = table_counts_flat(table);
    let counts_t = IdTensor::from_slice(&counts, vec![table.len(), 10], &device).unwrap();
    let meta_t = IdTensor::from_slice(meta, vec![batch, 8], &device).unwrap();
    let mut buffers = ms2::FormulaBuffers::<R, E>::poisoned(batch, m, f, &device).unwrap();
    ms2::formula_window(
        &search,
        &meta_t,
        table.max_error(),
        rows_visited_max,
        rows_scored_max,
        &buffers,
    )
    .unwrap();
    check_launches(&device).unwrap();
    ms2::formula_gather(&buffers.window, &search, &counts_t, &mut buffers.cand).unwrap();
    check_launches(&device).unwrap();
    let lp_t = Tensor::<R, E>::from_f32(log_prob, vec![batch, m], &device).unwrap();
    ms2::formula_top(&lp_t, &buffers.cand, &buffers).unwrap();
    check_launches(&device).unwrap();
    ms2::formula_top_counts(&buffers.top, &buffers.cand, &mut buffers.top_counts).unwrap();
    check_launches(&device).unwrap();
    let (got_window, got_counters, got_cand, got_top, got_lp, got_count, got_tc) = (
        buffers.window.try_to_vec().unwrap(),
        buffers.counters.try_to_vec().unwrap(),
        buffers.cand.try_to_vec().unwrap(),
        buffers.top.try_to_vec().unwrap(),
        buffers.top_log_prob.try_to_f32().unwrap(),
        buffers.top_count.try_to_vec().unwrap(),
        buffers.top_counts.try_to_vec().unwrap(),
    );
    let search_flat = search.try_to_vec().unwrap();
    let (want_window, want_counters) = twin::formula_window(
        table,
        meta,
        batch,
        m,
        table.max_error(),
        rows_visited_max,
        rows_scored_max,
    );
    let want_cand = twin::formula_gather(&want_window, &search_flat, &counts, batch, m, table.len());
    let (want_top, want_lp, want_count) =
        twin::formula_top_from_cand(log_prob, &want_cand, batch, m, f);
    let want_tc = twin::formula_top_counts(&want_top, &want_cand, batch, m, f);
    assert_windows_eq(&got_window, &got_counters, &want_window, &want_counters, what);
    assert_eq!(got_cand, want_cand, "{what}: cand differs");
    assert_eq!(got_top, want_top, "{what}: top differs");
    assert_bits_eq(&got_lp, &want_lp, &format!("{what}: top log-probs"));
    assert_eq!(got_count, want_count, "{what}: top counts differ");
    assert_eq!(got_tc, want_tc, "{what}: top_counts differ");
    got_counters
}

/// The capacity edge of one spectrum's counters: `FORMULA_SEARCH_EXHAUSTED`
/// exactly when `exhausted`, and `complete` (formula_support_complete) 1/0.
fn assert_capacity_edge(counters: &[u32], spectrum: usize, exhausted: bool, complete: bool, what: &str) {
    let status = counters[spectrum * 5 + 3];
    assert_eq!(
        status & request_status::FORMULA_SEARCH_EXHAUSTED != 0,
        exhausted,
        "{what}: exhausted bit"
    );
    assert_eq!(
        counters[spectrum * 5 + 4],
        u32::from(complete),
        "{what}: complete"
    );
}

#[test]
fn window_cand_top_counts_at_capacity() {
    // A dense C/H grid (~630 distinct compositions spanning ~270 Da): one
    // wide-uncertainty precursor joins every row, far past M = 128 and
    // M = 512. The second spectrum is absent (empty support throughout).
    let mut grid = Vec::new();
    for c in 0..=20u16 {
        for h in 1..=30u16 {
            grid.push(comp(c, h, 0, 0));
        }
    }
    let table = FormulaTable::from_compositions(grid.into_iter()).unwrap();
    assert!(table.len() >= 600, "the grid is dense: {}", table.len());
    let mid = table.len() / 2;
    let hit = precursor_of(table.mass(mid), 1);
    // The uncertainty bound (±400 Da) dwarfs the grid span, so every row is
    // inside the verdict's reach and joins as ambiguous.
    let meta: Vec<u32> = vec![
        meta_row(hit, 400_000_000, 1, 1000),
        meta_row(3_000_000_000, 50, 1, 200),
    ]
    .into_iter()
    .flat_map(|r| r.into_iter())
    .collect();
    // Learn the joined count with no effective cap.
    let (_, counters) = twin::formula_window(
        &table,
        &meta,
        2,
        2048,
        table.max_error(),
        u32::MAX,
        u32::MAX,
    );
    let joined = counters[1];
    assert_eq!(joined, table.len() as u32, "every grid row joins");
    assert!(joined > 512, "the window exceeds both capacities: {joined}");
    // Deterministic log-probs with ties (period 7); the absent spectrum's
    // values are irrelevant but must fill the shape.
    let lp = |m: usize| -> Vec<f32> { (0..2 * m).map(|i| -((i % 7) as f32) * 0.5).collect() };
    let f = 4usize;
    // M = 128 and M = 512: more rows join than fit, so the search exhausts
    // at capacity and the support is incomplete.
    for &m in &[128usize, 512] {
        let c = check_table_pipeline(
            &table,
            &meta,
            2,
            m,
            f,
            &lp(m),
            u32::MAX,
            u32::MAX,
            &format!("m-{m}"),
        );
        assert_capacity_edge(&c, 0, true, false, &format!("m-{m}"));
    }
    // joined == M: the window holds every joined row, so the search
    // completes with no exhausted bit.
    let m = joined as usize;
    let c = check_table_pipeline(&table, &meta, 2, m, f, &lp(m), u32::MAX, u32::MAX, "joined-eq-m");
    assert_capacity_edge(&c, 0, false, true, "joined-eq-m");
    // joined == M + 1 with cap M: the last row spills, so the search
    // exhausts and the support is incomplete.
    let m = joined as usize - 1;
    let c = check_table_pipeline(&table, &meta, 2, m, f, &lp(m), u32::MAX, u32::MAX, "joined-eq-m-plus-1");
    assert_capacity_edge(&c, 0, true, false, "joined-eq-m-plus-1");
    // The same two edges via the rows_scored_max cap with room to spare.
    let m = joined as usize + 8;
    let c = check_table_pipeline(&table, &meta, 2, m, f, &lp(m), u32::MAX, joined, "cap-eq-joined");
    assert_capacity_edge(&c, 0, false, true, "cap-eq-joined");
    let c = check_table_pipeline(
        &table,
        &meta,
        2,
        m,
        f,
        &lp(m),
        u32::MAX,
        joined - 1,
        "cap-eq-joined-minus-1",
    );
    assert_capacity_edge(&c, 0, true, false, "cap-eq-joined-minus-1");
}

#[test]
fn count_features_rejects_above_1023_at_upload() {
    let device = dev();
    // The 1023 boundary uploads; 1024 is refused with Error::Config (the
    // resident device log_table has 1024 entries, V1 §1.2).
    let mut at_bound = comp(1, 0, 0, 0);
    at_bound[1] = 1023;
    let ok_table = FormulaTable::from_compositions([at_bound]).unwrap();
    DeviceFormulaTable::<R, E>::upload(&ok_table, &device).unwrap();
    let mut over_bound = comp(1, 0, 0, 0);
    over_bound[1] = 1024;
    let bad_table =
        FormulaTable::from_compositions([comp(0, 2, 0, 1), over_bound]).unwrap();
    let r = DeviceFormulaTable::<R, E>::upload(&bad_table, &device);
    assert!(
        matches!(r, Err(mamba3::error::Error::Config(_))),
        "an H1024 row is Error::Config"
    );
}

/// Tiny full-model config for the V1 §1.2 gold-conditioning tests: `d = 16`
/// with the decoder-test encoder/decoder dims, `max_atoms` as given.
fn gold_test_model(max_atoms: u32) -> ModelConfig {
    let mut m = ModelConfig::v0();
    m.d_model = 16;
    m.n_peaks = 16;
    m.encoder_blocks = 1;
    m.decoder_blocks = 1;
    m.attention_heads = 2;
    m.encoder.d_model = 16;
    m.encoder.n_heads = 2;
    m.encoder.head_dim = 8;
    m.encoder.d_state = 8;
    m.encoder.n_groups = 2;
    m.decoder.d_model = 16;
    m.decoder.n_heads = 2;
    m.decoder.head_dim = 8;
    m.decoder.d_state = 8;
    m.decoder.n_groups = 2;
    m.max_atoms = max_atoms;
    m
}

/// Trainer config for the gold-conditioning tests: batch 2, the given slots,
/// formula weight 0.2 and the given teacher-forcing mode.
fn gold_test_train_config(slots: usize, conditioning: GoldFormulaConditioning) -> TrainConfig {
    TrainConfig {
        batch: 2,
        slots,
        lr: 3e-3,
        weight_decay: 0.1,
        formula_weight: 0.2,
        seed: 41,
        control: Control::None,
        grad_clip: None,
        gold_formula_conditioning: conditioning,
        formula_source: mamba3::models::ms2::contract::FormulaSource::Table,
        formula_window: 32,
        enum_lanes_max: 262_144,
        enum_lane_visits_max: 4_096,
        enum_dispatch_visits_max: 4_000_000,
        enum_fit_name: None,
        enum_fit_sha256: None,
        enum_fit_subset: None,
        lambda_assign: 0.0,
        ion_request_work_max: 268_435_456,
    }
}

/// Hand-built export spectrum: `n` peaks below the precursor, adduct 1.
fn gold_test_export(row: u64, precursor: u32, n: usize) -> ExportSpectrum {
    ExportSpectrum {
        row,
        spectrum_id: 9000 + row,
        adduct: 1,
        polarity: 1,
        precursor_mz_udalton: precursor,
        precursor_uncertainty_udalton: 50,
        raw_peak_count: n as u32,
        peak_id: (0..n as u32).collect(),
        mz_udalton: (0..n)
            .map(|k| precursor - ((k as u32 + 1) * 500_000))
            .collect(),
        intensity: vec![1.0; n],
        mz_uncertainty_udalton: 50,
        collision_energy_ev: 30.0,
        collision_energy_known: 1,
        energy_count: 1,
        instrument_class: 0,
    }
}

/// Dense table holding the in-window (glucose) and out-of-window (caffeine)
/// golds.
fn gold_conditioning_table() -> (FormulaTable, Composition, Composition) {
    let parent_in = comp(6, 12, 0, 6);
    let parent_out = comp(8, 10, 4, 2);
    let mut comps = fixture_compositions();
    comps.extend(synthetic_compositions(7, 48));
    let table = FormulaTable::from_compositions(comps.into_iter()).unwrap();
    for p in [&parent_in, &parent_out] {
        assert!(
            (0..table.len()).any(|r| table.composition(r) == p),
            "the table holds the gold"
        );
    }
    (table, parent_in, parent_out)
}

/// The table row of a composition in the table.
fn gold_table_row(table: &FormulaTable, want: &Composition) -> usize {
    (0..table.len())
        .find(|&r| table.composition(r) == want)
        .expect("gold row in the table")
}

/// Two unlabeled spectra sharing one precursor: spectrum 0's gold (glucose)
/// joins the window, spectrum 1's gold (caffeine, ~14 Da away) does not.
fn gold_conditioning_set(
    table: &FormulaTable,
    parent_in: Composition,
    parent_out: Composition,
) -> ExperimentSet {
    let precursor = precursor_of(table.mass(gold_table_row(table, &parent_in)), 1);
    let spectra = [parent_in, parent_out]
        .iter()
        .enumerate()
        .map(|(b, parent)| ExperimentSpectrum {
            molecule: b,
            spectrum: gold_test_export(b as u64, precursor, 10),
            parent: MolGraph::new(Vec::new(), Vec::new()).unwrap(),
            parent_composition: *parent,
            labels: None,
            domain: SpectrumDomain::InDomainUnlabeled,
        })
        .collect();
    ExperimentSet {
        name: "gold-conditioning".to_string(),
        source_sha256: "fixture".to_string(),
        molecules: vec!["in-window".to_string(), "out-of-window".to_string()],
        spectra,
    }
}

/// The trainer's conditioning inputs, recomputed with the trainer's own
/// weights (mirroring `Ms2Trainer::forward_with_donors` without changing
/// `src`): the device gold slots, the gold counts, the scored row
/// embeddings, the Composition embedding (device `count_features` path) and
/// the gathered scored-row embedding (ScoredRowOrZero path, unmasked).
struct GoldForward {
    slots: Vec<u32>,
    gold_counts: Vec<u32>,
    scored_embedding: Vec<f32>,
    e_composition: Vec<f32>,
    e_scored: Vec<f32>,
    d: usize,
}

/// Encode + window + score + gold slot + both teacher-forcing embeddings for
/// two spectra. `n_raw` 64 covers the hand-built peak lists.
fn gold_forward(trainer: &Ms2Trainer<R, E>, set: &ExperimentSet, table: &FormulaTable) -> GoldForward {
    let device = dev();
    let d = trainer.model.config.d_model as usize;
    let spectra_h = spectrum_batch_for(set, &[0, 1], 64).unwrap();
    let spectra = DeviceSpectra::upload(&spectra_h, &device).unwrap();
    let peaks =
        ms2::PeakBuffers::<R, E>::new(2, spectra.n_raw, trainer.model.config.n_peaks as usize, &device);
    let encoded = trainer
        .model
        .encoder
        .encode(&spectra, &peaks, Control::None)
        .unwrap();
    check_launches(&device).unwrap();
    let uploaded = DeviceFormulaTable::<R, E>::upload(table, &device).unwrap();
    let mut buffers = ms2::FormulaBuffers::<R, E>::new(2, TRAIN_WINDOW_M, 1, &device);
    ms2::formula_window(
        &uploaded.table,
        &spectra.meta,
        uploaded.max_error,
        u32::MAX,
        TRAIN_ROWS_SCORED_MAX,
        &buffers,
    )
    .unwrap();
    ms2::formula_gather(
        &buffers.window,
        &uploaded.table,
        &uploaded.counts,
        &mut buffers.cand,
    )
    .unwrap();
    ms2::count_features(
        &buffers.cand.reshape(vec![2 * TRAIN_WINDOW_M, 13]).unwrap(),
        &uploaded.log_table,
        &mut buffers
            .cand_feat
            .reshape(vec![2 * TRAIN_WINDOW_M, 10])
            .unwrap(),
        13,
    )
    .unwrap();
    check_launches(&device).unwrap();
    let scored = trainer.model.formula.score(&buffers, &encoded.pool).unwrap();
    check_launches(&device).unwrap();
    let mut gold_counts = vec![0u32; 2 * 10];
    for (b, s) in set.spectra.iter().enumerate() {
        for e in 0..10 {
            gold_counts[b * 10 + e] = u32::from(s.parent_composition[e]);
        }
    }
    let gold_counts_t = IdTensor::from_slice(&gold_counts, vec![2, 10], &device).unwrap();
    let mut gold_slot_t = IdTensor::empty(vec![2], &device);
    ms2::gold_slot(&buffers.cand, &gold_counts_t, &mut gold_slot_t).unwrap();
    check_launches(&device).unwrap();
    let slots = gold_slot_t.try_to_vec().unwrap();
    // Composition: the head embedding of the gold counts through the device
    // `count_features`, whether or not the search scored them.
    let mut gold_feat_buf = Tensor::empty(vec![2, 10], &device);
    ms2::count_features(&gold_counts_t, &uploaded.log_table, &mut gold_feat_buf, 10).unwrap();
    check_launches(&device).unwrap();
    let e_composition = trainer
        .model
        .formula
        .embed_rows(&Var::constant(gold_feat_buf))
        .unwrap()
        .try_to_f32()
        .unwrap();
    check_launches(&device).unwrap();
    // ScoredRowOrZero: the scored row's embedding, gathered at host-side
    // safe slots (the caller zeroes the absent rows).
    let safe: Vec<u32> = slots
        .iter()
        .map(|&s| if s == u32::MAX { 0 } else { s })
        .collect();
    let safe_t = IdTensor::from_slice(&safe, vec![2], &device).unwrap();
    let e_scored = Var::gather_tokens(&scored.embedding, &safe_t, 1)
        .unwrap()
        .reshape(vec![2, d])
        .unwrap()
        .try_to_f32()
        .unwrap();
    check_launches(&device).unwrap();
    let scored_embedding = scored.embedding.try_to_f32().unwrap();
    GoldForward {
        slots,
        gold_counts,
        scored_embedding,
        e_composition,
        e_scored,
        d,
    }
}

#[test]
fn gold_conditioning_composition_vs_row() {
    // V1 §1.2 teacher forcing: Composition conditions the decoder on the
    // head embedding of the gold counts for both spectra (non-zero even
    // when the gold is out of window); ScoredRowOrZero conditions on the
    // scored row's embedding in window and on zero out of window.
    let device = dev();
    let (table, parent_in, parent_out) = gold_conditioning_table();
    let set = gold_conditioning_set(&table, parent_in, parent_out);
    let model = gold_test_model(16);
    let mut trainer_comp: Ms2Trainer<R, E> = Ms2Trainer::new(
        &model,
        &table,
        &gold_test_train_config(2, GoldFormulaConditioning::Composition),
        &device,
    )
    .unwrap();
    let trainer_zero: Ms2Trainer<R, E> = Ms2Trainer::new(
        &model,
        &table,
        &gold_test_train_config(2, GoldFormulaConditioning::ScoredRowOrZero),
        &device,
    )
    .unwrap();
    // The real trainer forward pins the window status: spectrum 0's gold is
    // scored, spectrum 1's is absent.
    let eval = trainer_comp.teacher_eval(&set, &[0, 1]).unwrap();
    assert_eq!(eval.gold_slot.len(), 2);
    assert_ne!(
        eval.gold_slot[0],
        u32::MAX,
        "spectrum 0 gold is in window"
    );
    assert_eq!(
        eval.gold_slot[1],
        u32::MAX,
        "spectrum 1 gold is out of window"
    );
    let d = model.d_model as usize;
    let fwd = gold_forward(&trainer_comp, &set, &table);
    assert_eq!(
        fwd.slots, eval.gold_slot,
        "the recomputed gold slots match the trainer"
    );
    // Composition: the decoder input is the head embedding of the gold
    // counts, recomputed here from host features (`twin::count_features`
    // over the resident `log_table`) — exact bits, both spectra.
    let host_feat = twin::count_features(&fwd.gold_counts, &twin::log_table(), 2, 10);
    let host_feat_t = Tensor::<R, E>::from_f32(&host_feat, vec![2, 10], &device).unwrap();
    let expected = trainer_comp
        .model
        .formula
        .embed_rows(&Var::constant(host_feat_t))
        .unwrap()
        .try_to_f32()
        .unwrap();
    check_launches(&device).unwrap();
    assert_bits_eq(&fwd.e_composition, &expected, "composition e_cond");
    // The out-of-window Composition embedding is non-zero: teacher forcing
    // still conditions on the true parent.
    assert!(
        fwd.e_composition[d..].iter().any(|&v| v != 0.0),
        "the out-of-window composition embedding is non-zero"
    );
    // ScoredRowOrZero: the scored row's embedding in window, zero out. The
    // absent mask is applied host-side here (the zeroing the trainer does
    // on the device via its validity mask).
    let fwd_zero = gold_forward(&trainer_zero, &set, &table);
    assert_eq!(
        fwd_zero.slots, eval.gold_slot,
        "the slots agree across modes"
    );
    let mut e_masked = fwd_zero.e_scored.clone();
    for (b, &s) in fwd_zero.slots.iter().enumerate() {
        if s == u32::MAX {
            for v in &mut e_masked[b * d..(b + 1) * d] {
                *v = 0.0;
            }
        }
    }
    let s0 = fwd_zero.slots[0] as usize;
    for j in 0..d {
        assert_eq!(
            e_masked[j].to_bits(),
            fwd_zero.scored_embedding[s0 * d + j].to_bits(),
            "in-window scored row element {j}"
        );
        assert_eq!(
            e_masked[d + j].to_bits(),
            0.0f32.to_bits(),
            "out-of-window row is zero at {j}"
        );
    }
    check_launches(&device).unwrap();
}

#[test]
fn trainer_conditioning_bit_equality_at_m32_and_m128() {
    // B1-fix finding 3: through the ACTUAL trainer path
    // (`conditioning_for_test_with_window`, the same calls `forward` makes),
    // every element of the Composition conditioning embedding of an
    // in-window gold equals the scored row's embedding by `to_bits()`, at
    // M = 32 and M = 128. The gold path computes its embedding through the
    // same `embed_rows` operation on the same feature construction
    // (`count_features` over the resident `log_table`), so the same features
    // give the same bits; this test pins that gate.
    let device = dev();
    let (table, parent_in, parent_out) = gold_conditioning_table();
    let set = gold_conditioning_set(&table, parent_in, parent_out);
    let model = gold_test_model(16);
    for window_m in [32usize, 128] {
        let mut trainer: Ms2Trainer<R, E> = Ms2Trainer::new(
            &model,
            &table,
            &gold_test_train_config(2, GoldFormulaConditioning::Composition),
            &device,
        )
        .unwrap();
        let cond = trainer
            .conditioning_for_test_with_window(&set, &[0, 1], window_m)
            .unwrap();
        check_launches(&device).unwrap();
        assert_eq!(cond.window_m, window_m);
        assert_eq!(cond.spectra, 2);
        let d = cond.d_model;
        assert_ne!(cond.slots[0], u32::MAX, "M={window_m}: spectrum 0 gold is scored");
        let s0 = cond.slots[0] as usize;
        for j in 0..d {
            assert_eq!(
                cond.e_cond[j].to_bits(),
                cond.scored_embedding[(s0 * d) + j].to_bits(),
                "M={window_m}: in-window conditioning element {j}"
            );
        }
        // The out-of-window Composition embedding is non-zero: teacher
        // forcing still conditions on the true parent.
        assert!(
            cond.e_cond[d..].iter().any(|&v| v != 0.0),
            "M={window_m}: the out-of-window composition embedding is non-zero"
        );
    }
}

#[test]
fn trainer_conditioning_scored_row_or_zero_values() {
    // B1-fix finding 3, `ScoredRowOrZero` mode through the actual trainer
    // path (the hook runs the same forward prefix `step` runs): in-window
    // conditioning equals the scored row by `to_bits()`, out-of-window is
    // zero. The structural proof that this mode launches no gold-network
    // kernel lives in the counter-owning footprint binary (the
    // `ms2.gold_embed` tally scope is entered only by the `Composition`
    // branch), since this multi-test binary cannot isolate process-global
    // counters.
    let device = dev();
    let (table, parent_in, parent_out) = gold_conditioning_table();
    let set = gold_conditioning_set(&table, parent_in, parent_out);
    let model = gold_test_model(16);
    let mut trainer_zero: Ms2Trainer<R, E> = Ms2Trainer::new(
        &model,
        &table,
        &gold_test_train_config(2, GoldFormulaConditioning::ScoredRowOrZero),
        &device,
    )
    .unwrap();
    let cond = trainer_zero.conditioning_for_test(&set, &[0, 1]).unwrap();
    check_launches(&device).unwrap();
    let d = cond.d_model;
    assert_ne!(cond.slots[0], u32::MAX, "spectrum 0 gold is scored");
    assert_eq!(cond.slots[1], u32::MAX, "spectrum 1 gold is absent");
    let s0 = cond.slots[0] as usize;
    for j in 0..d {
        assert_eq!(
            cond.e_cond[j].to_bits(),
            cond.scored_embedding[s0 * d + j].to_bits(),
            "in-window scored-row element {j}"
        );
        // Out-of-window rows are zero values (the device validity mask
        // multiplies by 0.0, so the sign bit may be set: compare by value,
        // not by `to_bits()`; the bit gate above is the in-window one).
        assert!(
            cond.e_cond[d + j] == 0.0,
            "out-of-window row is zero at {j}, got {}",
            cond.e_cond[d + j]
        );
    }
    // ...and the Composition in-window row agrees with the scored row too.
    let mut trainer_comp: Ms2Trainer<R, E> = Ms2Trainer::new(
        &model,
        &table,
        &gold_test_train_config(2, GoldFormulaConditioning::Composition),
        &device,
    )
    .unwrap();
    let cond_comp = trainer_comp.conditioning_for_test(&set, &[0, 1]).unwrap();
    check_launches(&device).unwrap();
    let s0c = cond_comp.slots[0] as usize;
    for j in 0..d {
        assert_eq!(
            cond_comp.e_cond[j].to_bits(),
            cond_comp.scored_embedding[s0c * d + j].to_bits(),
            "composition in-window element {j}"
        );
    }
}

#[test]
fn trainer_conditioning_observes_production_mode_for_out_of_window_gold() {
    // B1-fix finding 3: the hook returns the conditioning embedding of the
    // production forward prefix (the exact tensor handed to
    // `decoder.teacher`), so this test fails if production passed the wrong
    // mode: with `Composition` the out-of-window gold still conditions on
    // its true-parent embedding (non-zero), while with `ScoredRowOrZero`
    // the same spectrum conditions on zero.
    let device = dev();
    let (table, parent_in, parent_out) = gold_conditioning_table();
    let set = gold_conditioning_set(&table, parent_in, parent_out);
    let model = gold_test_model(16);
    let mut trainer_comp: Ms2Trainer<R, E> = Ms2Trainer::new(
        &model,
        &table,
        &gold_test_train_config(2, GoldFormulaConditioning::Composition),
        &device,
    )
    .unwrap();
    let cond_comp = trainer_comp.conditioning_for_test(&set, &[0, 1]).unwrap();
    check_launches(&device).unwrap();
    let d = cond_comp.d_model;
    assert_eq!(
        cond_comp.slots[1],
        u32::MAX,
        "spectrum 1 gold is out of window"
    );
    assert!(
        cond_comp.e_cond[d..].iter().any(|&v| v != 0.0),
        "Composition conditions an out-of-window gold on a non-zero embedding"
    );
    let mut trainer_zero: Ms2Trainer<R, E> = Ms2Trainer::new(
        &model,
        &table,
        &gold_test_train_config(2, GoldFormulaConditioning::ScoredRowOrZero),
        &device,
    )
    .unwrap();
    let cond_zero = trainer_zero.conditioning_for_test(&set, &[0, 1]).unwrap();
    check_launches(&device).unwrap();
    assert_eq!(
        cond_zero.slots[1],
        u32::MAX,
        "spectrum 1 gold is out of window"
    );
    assert!(
        cond_zero.e_cond[d..].iter().all(|&v| v == 0.0),
        "ScoredRowOrZero conditions an out-of-window gold on zero"
    );
}

#[test]
fn trainer_conditioning_out_of_window_drives_row_gradients() {
    // B1-fix finding 3: an out-of-window gold gives a non-zero Composition
    // embedding whose graph loss sends a non-zero gradient into the row
    // network — through the actual trainer `step`, observed as a row-weight
    // change with decay disabled (a zero gradient would leave them bit-equal).
    let device = dev();
    let (table, parent_in, parent_out) = gold_conditioning_table();
    let model = gold_test_model(8);
    let atoms = model.max_atoms as usize;
    let closures = model.max_ring_closures;
    let limits = Limits::new(atoms, closures as usize).unwrap();
    let traces = [
        minimal_trace(limits, parent_in),
        minimal_trace(limits, parent_out),
    ];
    // Both spectra share the in-window precursor, so spectrum 1's gold
    // (caffeine) is out of window while labeled: its graph loss must still
    // flow through the Composition embedding into the row network.
    let precursor = precursor_of(table.mass(gold_table_row(&table, &parent_in)), 1);
    let spectra = [parent_in, parent_out]
        .iter()
        .enumerate()
        .map(|(b, parent)| ExperimentSpectrum {
            molecule: b,
            spectrum: gold_test_export(700 + b as u64, precursor, 10),
            parent: MolGraph::new(Vec::new(), Vec::new()).unwrap(),
            parent_composition: *parent,
            labels: Some(single_label(traces[b].clone())),
            domain: SpectrumDomain::InDomainLabeled,
        })
        .collect();
    let set = ExperimentSet {
        name: "oow-grad".to_string(),
        source_sha256: "fixture".to_string(),
        molecules: vec!["in".to_string(), "out".to_string()],
        spectra,
    };
    let mut cfg = gold_test_train_config(2, GoldFormulaConditioning::Composition);
    cfg.weight_decay = 0.0;
    let mut trainer: Ms2Trainer<R, E> = Ms2Trainer::new(&model, &table, &cfg, &device).unwrap();
    // The out-of-window conditioning row is non-zero through the real path.
    let cond = trainer.conditioning_for_test(&set, &[0, 1]).unwrap();
    check_launches(&device).unwrap();
    let d = cond.d_model;
    assert_eq!(cond.slots[1], u32::MAX, "spectrum 1 gold is out of window");
    assert!(
        cond.e_cond[d..].iter().any(|&v| v != 0.0),
        "the out-of-window conditioning embedding is non-zero"
    );
    // One optimizer step moves a row-network weight: the graph loss of the
    // out-of-window spectrum flows through its (non-zero) embedding.
    let before: Vec<(String, Vec<f32>)> = trainer
        .model
        .formula
        .named_parameters()
        .into_iter()
        .filter(|(n, _)| n.contains("row_in") || n.contains("row_out"))
        .map(|(n, p)| (n, p.value().to_f32()))
        .collect();
    assert!(!before.is_empty(), "the head exposes row params");
    trainer.request_report();
    // Step on the out-of-window spectrum alone: any row-weight movement
    // proves its graph loss flows through its (non-zero) embedding.
    let report = trainer.step(&set, &[1]).unwrap().expect("report");
    assert!(report.loss.is_finite(), "the step loss is finite");
    check_launches(&device).unwrap();
    let mut moved = false;
    for (name, old) in &before {
        let (_, param) = trainer
            .model
            .formula
            .named_parameters()
            .into_iter()
            .find(|(n, _)| n == name)
            .expect("the param still exists");
        let new = param.value().to_f32();
        if new.iter().zip(old.iter()).any(|(a, b)| a.to_bits() != b.to_bits()) {
            moved = true;
        }
    }
    assert!(moved, "a row-network weight moved: the graph loss drives it");
}

/// Shortest legal trace under `budget`: START, one legal root ADD_ATOM,
/// STOP (STOP is offered once an atom exists).
fn minimal_trace(limits: Limits, budget: Composition) -> Vec<Token> {
    let mut st = TraceState::new(limits, Some(budget));
    let start = Token {
        kind: START,
        atom_type: 0,
        bond: 0,
        pointer: 0,
    };
    assert!(st.is_legal(start), "START is legal");
    st.apply(start).unwrap();
    let mut root = None;
    for ty in 1..18u8 {
        let cand = Token {
            kind: ADD_ATOM,
            atom_type: ty,
            bond: 0,
            pointer: 0,
        };
        if st.is_legal(cand) {
            root = Some(cand);
            break;
        }
    }
    let root = root.expect("a legal root atom type");
    st.apply(root).unwrap();
    let stop = Token {
        kind: STOP,
        atom_type: 0,
        bond: 0,
        pointer: 0,
    };
    assert!(st.is_legal(stop), "STOP is legal after the root");
    vec![start, root, stop]
}

/// A single-target label with weight 1 on `trace`.
fn single_label(trace: Vec<Token>) -> Labels {
    Labels {
        embeddings: Vec::new(),
        graphs: 1,
        targets_before_cut: 1,
        targets: vec![Target {
            trace,
            weight: 1,
            q: 1.0,
            embeddings: Vec::new(),
            anchors: Vec::new(),
        }],
        dropped_weight: 0.0,
        cut_is_tied: false,
        explained_peaks: Vec::new(),
        ambiguous_hypotheses: 0,
        canonicalization_failures: 0,
    }
}

#[test]
fn composed_loss_finite_difference() {
    // V1 §1.2 Composition mode, tiny case (B = 2, slots 2, T = 14):
    // central finite differences of the composed loss
    // (`graph + 0.2 * formula`) against one row-network weight, with the
    // existing tolerance (2e-2 relative + 1e-3), not loosened.
    let device = dev();
    let (table, parent_a, parent_b) = gold_conditioning_table();
    let model = gold_test_model(8);
    let atoms = model.max_atoms as usize;
    let closures = model.max_ring_closures;
    let limits = Limits::new(atoms, closures as usize).unwrap();
    let t = limits.max_steps();
    assert_eq!((t, atoms), (14, 8));
    let traces = [
        minimal_trace(limits, parent_a),
        minimal_trace(limits, parent_b),
    ];
    let spectra = [parent_a, parent_b]
        .iter()
        .enumerate()
        .map(|(b, parent)| {
            let precursor = precursor_of(table.mass(gold_table_row(&table, parent)), 1);
            ExperimentSpectrum {
                molecule: b,
                spectrum: gold_test_export(700 + b as u64, precursor, 10),
                parent: MolGraph::new(Vec::new(), Vec::new()).unwrap(),
                parent_composition: *parent,
                labels: Some(single_label(traces[b].clone())),
                domain: SpectrumDomain::InDomainLabeled,
            }
        })
        .collect();
    let set = ExperimentSet {
        name: "composed-fd".to_string(),
        source_sha256: "fixture".to_string(),
        molecules: vec!["glucose".to_string(), "caffeine".to_string()],
        spectra,
    };
    let trainer: Ms2Trainer<R, E> = Ms2Trainer::new(
        &model,
        &table,
        &gold_test_train_config(2, GoldFormulaConditioning::Composition),
        &device,
    )
    .unwrap();
    let spectra_h = spectrum_batch_for(&set, &[0, 1], 64).unwrap();
    let targets_h = target_batch_for(&set, &[0, 1], 2, limits).unwrap();
    let spectra_b = DeviceSpectra::upload(&spectra_h, &device).unwrap();
    let targets: DeviceTargets<R, E> = targets_h.upload(&device).unwrap();
    let peaks =
        ms2::PeakBuffers::<R, E>::new(2, spectra_b.n_raw, model.n_peaks as usize, &device);
    let uploaded = DeviceFormulaTable::<R, E>::upload(&table, &device).unwrap();
    let mut buffers = ms2::FormulaBuffers::<R, E>::new(2, TRAIN_WINDOW_M, 1, &device);
    ms2::formula_window(
        &uploaded.table,
        &spectra_b.meta,
        uploaded.max_error,
        u32::MAX,
        TRAIN_ROWS_SCORED_MAX,
        &buffers,
    )
    .unwrap();
    ms2::formula_gather(
        &buffers.window,
        &uploaded.table,
        &uploaded.counts,
        &mut buffers.cand,
    )
    .unwrap();
    ms2::count_features(
        &buffers.cand.reshape(vec![2 * TRAIN_WINDOW_M, 13]).unwrap(),
        &uploaded.log_table,
        &mut buffers
            .cand_feat
            .reshape(vec![2 * TRAIN_WINDOW_M, 10])
            .unwrap(),
        13,
    )
    .unwrap();
    check_launches(&device).unwrap();
    let mut gold_counts = vec![0u32; 2 * 10];
    for (b, s) in set.spectra.iter().enumerate() {
        for e in 0..10 {
            gold_counts[b * 10 + e] = u32::from(s.parent_composition[e]);
        }
    }
    let gold_counts_t = IdTensor::from_slice(&gold_counts, vec![2, 10], &device).unwrap();
    let mut gold_slot_t = IdTensor::empty(vec![2], &device);
    ms2::gold_slot(&buffers.cand, &gold_counts_t, &mut gold_slot_t).unwrap();
    check_launches(&device).unwrap();
    let slots = gold_slot_t.try_to_vec().unwrap();
    assert!(
        slots.iter().all(|&s| s != u32::MAX),
        "both golds are scored: {slots:?}"
    );
    let constants = ms2::Ms2Constants::new(&device);
    let rows = 2 * 2;
    let replay_bufs = ms2::ReplayBuffers::poisoned(rows, t, atoms, &device).unwrap();
    ms2::grammar_replay(
        &targets.tokens,
        &targets.meta,
        &constants,
        atoms as u32,
        closures,
        &replay_bufs,
    )
    .unwrap();
    check_launches(&device).unwrap();
    let replay = ReplayView {
        replay: &replay_bufs.replay,
        atoms: &replay_bufs.atoms,
    };
    // Composition conditioning (V1 §1.2): the head embedding of the gold
    // counts. Re-encoded every call so a perturbation anywhere reaches the
    // loss through a fresh forward pass.
    let loss_of = || {
        let enc = trainer
            .model
            .encoder
            .encode(&spectra_b, &peaks, Control::None)
            .unwrap();
        let sc = trainer.model.formula.score(&buffers, &enc.pool).unwrap();
        let mut gold_feat_buf = Tensor::empty(vec![2, 10], &device);
        ms2::count_features(&gold_counts_t, &uploaded.log_table, &mut gold_feat_buf, 10).unwrap();
        let e_cond = trainer
            .model
            .formula
            .embed_rows(&Var::constant(gold_feat_buf))
            .unwrap();
        let tout = trainer
            .model
            .decoder
            .teacher(&enc, &e_cond, &targets, &replay)
            .unwrap();
        let graph = graph_loss(&tout, &targets.q, 2).unwrap();
        let formula_loss = trainer.model.formula.loss(&sc, &gold_slot_t).unwrap();
        graph
            .add(&formula_loss.mul_scalar(0.2))
            .unwrap()
            .try_to_f32()
            .unwrap()[0]
    };
    // Analytic gradients of the composed loss.
    let enc0 = trainer
        .model
        .encoder
        .encode(&spectra_b, &peaks, Control::None)
        .unwrap();
    let sc0 = trainer.model.formula.score(&buffers, &enc0.pool).unwrap();
    let mut gold_feat0 = Tensor::empty(vec![2, 10], &device);
    ms2::count_features(&gold_counts_t, &uploaded.log_table, &mut gold_feat0, 10).unwrap();
    let e_cond0 = trainer
        .model
        .formula
        .embed_rows(&Var::constant(gold_feat0))
        .unwrap();
    let tout0 = trainer
        .model
        .decoder
        .teacher(&enc0, &e_cond0, &targets, &replay)
        .unwrap();
    let graph0 = graph_loss(&tout0, &targets.q, 2).unwrap();
    let formula0 = trainer.model.formula.loss(&sc0, &gold_slot_t).unwrap();
    let total0 = graph0.add(&formula0.mul_scalar(0.2)).unwrap();
    let loss_v = total0.try_to_f32().unwrap()[0];
    assert!(loss_v.is_finite(), "the composed loss is finite");
    let grads = total0.backward_retain().unwrap();
    // One row-network entry: the largest |analytic gradient| among the
    // row_in/row_out weights, which feed both the formula score and the
    // Composition decoder input.
    let row_params: Vec<(String, mamba3::nn::param::Param<R, E>)> = trainer
        .model
        .formula
        .named_parameters()
        .into_iter()
        .filter(|(n, _)| n.contains("row_in") || n.contains("row_out"))
        .collect();
    assert!(
        !row_params.is_empty(),
        "the head exposes row_in/row_out params"
    );
    let mut best_name = String::new();
    let mut best_idx = 0usize;
    let mut best_abs = -1.0f32;
    for (name, param) in &row_params {
        let analytic = grads.get(param.id()).unwrap().to_f32();
        for (i, v) in analytic.iter().enumerate() {
            if v.abs() > best_abs {
                best_abs = v.abs();
                best_name = name.clone();
                best_idx = i;
            }
        }
    }
    let (_, param) = row_params
        .iter()
        .find(|(n, _)| *n == best_name)
        .expect("the top entry's param");
    let shape = param.shape().dims().to_vec();
    let base = param.value().to_f32();
    let analytic = grads.get(param.id()).unwrap().to_f32();
    let mut up = base.clone();
    up[best_idx] += 1e-2;
    param.set(Tensor::<R, E>::from_f32(&up, shape.clone(), &device).unwrap());
    let fu = loss_of();
    let mut down = base.clone();
    down[best_idx] -= 1e-2;
    param.set(Tensor::<R, E>::from_f32(&down, shape.clone(), &device).unwrap());
    let fd = loss_of();
    param.set(Tensor::<R, E>::from_f32(&base, shape.clone(), &device).unwrap());
    let numeric = (fu - fd) / 2e-2;
    let analytic_v = analytic[best_idx];
    assert!(
        numeric.abs() > 1e-4,
        "{best_name}[{best_idx}]: the check is not vacuous (numeric {numeric})"
    );
    assert!(
        (analytic_v - numeric).abs() <= 2e-2 * numeric.abs() + 1e-3,
        "{best_name}[{best_idx}]: analytic={analytic_v} numeric={numeric}"
    );
    check_launches(&device).unwrap();
}

/// TF reference: the flagged finite slots sorted by (score descending, slot
/// ascending), first `F` taken. Scores outside `-FINITE_MAX < s <
/// FINITE_MAX` never appear. Returns per-spectrum picks as (row, slot,
/// score).
fn tf_reference_picks(
    log_prob: &[f32],
    window: &[u32],
    b: usize,
    m: usize,
    f: usize,
) -> Vec<(u32, u32, f32)> {
    const FM: f32 = 3.0e38;
    let mut cands: Vec<(f32, usize)> = Vec::new();
    for slot in 0..m {
        if window[(b * m + slot) * 2 + 1] == 0 {
            continue;
        }
        let s = log_prob[b * m + slot];
        if !(s > -FM && s < FM) {
            continue;
        }
        cands.push((s, slot));
    }
    cands.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .expect("filtered scores are finite")
            .then_with(|| a.1.cmp(&b.1))
    });
    cands
        .into_iter()
        .take(f)
        .map(|(s, slot)| (window[(b * m + slot) * 2], slot as u32, s))
        .collect()
}

#[test]
fn formula_top_poisoned_random_windows_match_reference() {
    let device = dev();
    let f = 4usize;
    // Poison scores: NaN, infinities and out-of-domain finite extremes are
    // never selected; the domain edges (±3e38, exclusive) are excluded while
    // ±2.9e38 are included.
    let poisons: Vec<f32> = vec![
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        3.1e38,
        -3.1e38,
        3.0e38,
        -3.0e38,
        2.9e38,
        -2.9e38,
    ];
    for &m in &[32usize, 128, 512, 2048] {
        // Six spectra: random flags+ties, all-equal all-flagged, fewer
        // flagged than F, zero flagged, poisoned scores, random+poison mix.
        let batch = 6usize;
        let mut rng = Rng::seeded(1000 + m as u64);
        let u = rng.uniform_vec(batch * m * 2, 0.0, 1.0);
        let mut window = vec![0u32; batch * m * 2];
        let mut log_prob = vec![0.0f32; batch * m];
        for b in 0..batch {
            for slot in 0..m {
                let base = (b * m + slot) * 2;
                let r0 = u[(b * m + slot) * 2];
                let r1 = u[(b * m + slot) * 2 + 1];
                match b {
                    0 => {
                        // Random flags (0/1/2), quantised scores force ties.
                        let flag = (r0 * 3.0) as u32;
                        window[base] = if flag == 0 { u32::MAX } else { 1000 + slot as u32 };
                        window[base + 1] = flag;
                        log_prob[b * m + slot] = ((r1 * 8.0 - 4.0) * 2.0).round() / 2.0;
                    }
                    1 => {
                        // All-equal scores, all flagged (a worst-case row).
                        window[base] = 2000 + slot as u32;
                        window[base + 1] = 1;
                        log_prob[b * m + slot] = 0.25;
                    }
                    2 => {
                        // Fewer flagged slots than F: only slots 0 and 1.
                        if slot < 2 {
                            window[base] = 3000 + slot as u32;
                            window[base + 1] = if slot == 0 { 1 } else { 2 };
                            log_prob[b * m + slot] = 1.0 - slot as f32;
                        } else {
                            window[base] = u32::MAX;
                            window[base + 1] = 0;
                            log_prob[b * m + slot] = -99.0;
                        }
                    }
                    3 => {
                        // Zero flagged slots.
                        window[base] = u32::MAX;
                        window[base + 1] = 0;
                        log_prob[b * m + slot] = r1 * 2.0 - 1.0;
                    }
                    4 => {
                        // All flagged, poisoned scores cycling through the
                        // poison set with finite ties interleaved.
                        window[base] = 4000 + slot as u32;
                        window[base + 1] = if slot % 2 == 0 { 1 } else { 2 };
                        if slot % 3 == 2 {
                            log_prob[b * m + slot] = 0.5;
                        } else {
                            log_prob[b * m + slot] =
                                poisons[slot % poisons.len()];
                        }
                    }
                    _ => {
                        // Random flags with poison mixed in every seventh
                        // slot and quantised ties elsewhere.
                        let flag = (r0 * 3.0) as u32;
                        window[base] = if flag == 0 { u32::MAX } else { 5000 + slot as u32 };
                        window[base + 1] = flag;
                        if slot % 7 == 3 {
                            log_prob[b * m + slot] =
                                poisons[slot % poisons.len()];
                        } else {
                            log_prob[b * m + slot] =
                                ((r1 * 8.0 - 4.0) * 2.0).round() / 2.0;
                        }
                    }
                }
            }
        }
        let cand = cand_from_window(&window, batch, m);
        let lp_t = Tensor::<R, E>::from_f32(&log_prob, vec![batch, m], &device).unwrap();
        let cand_t = IdTensor::from_slice(&cand, vec![batch, m, 13], &device).unwrap();
        let out = ms2::FormulaBuffers::<R, E>::poisoned(batch, m, f, &device).unwrap();
        ms2::formula_top(&lp_t, &cand_t, &out).unwrap();
        check_launches(&device).unwrap();
        let (got_top, got_lp, got_count) = (
            out.top.try_to_vec().unwrap(),
            out.top_log_prob.try_to_f32().unwrap(),
            out.top_count.try_to_vec().unwrap(),
        );
        // Kernel versus both twins, every element.
        let (want_top, want_lp, want_count) =
            twin::formula_top(&log_prob, &window, batch, m, f);
        assert_eq!(got_top, want_top, "m={m}: kernel top differs from twin");
        assert_bits_eq(&got_lp, &want_lp, &format!("m={m}: kernel log-probs"));
        assert_eq!(got_count, want_count, "m={m}: kernel counts differ");
        let (want_c, want_c_lp, want_c_count) =
            twin::formula_top_from_cand(&log_prob, &cand, batch, m, f);
        assert_eq!(want_c, want_top, "m={m}: cand twin top differs");
        assert_bits_eq(&want_c_lp, &want_lp, &format!("m={m}: cand twin log-probs"));
        assert_eq!(want_c_count, want_count, "m={m}: cand twin counts differ");
        // Independent reference equals the twin, every element.
        for b in 0..batch {
            let picks = tf_reference_picks(&log_prob, &window, b, m, f);
            assert_eq!(
                want_count[b] as usize,
                picks.len(),
                "m={m} spectrum {b}: twin count differs from reference"
            );
            for (p, &(row, slot, s)) in picks.iter().enumerate() {
                assert_eq!(want_top[(b * f + p) * 2], row, "m={m} b={b} pick {p} row");
                assert_eq!(
                    want_top[(b * f + p) * 2 + 1],
                    slot,
                    "m={m} b={b} pick {p} slot"
                );
                assert_eq!(
                    want_lp[b * f + p].to_bits(),
                    s.to_bits(),
                    "m={m} b={b} pick {p} score"
                );
                // No poisoned score is ever selected.
                assert!(
                    s > -3.0e38 && s < 3.0e38,
                    "m={m} b={b} pick {p} selected out-of-domain {s}"
                );
            }
            for p in picks.len()..f {
                assert_eq!(
                    want_top[(b * f + p) * 2],
                    u32::MAX,
                    "m={m} b={b} padding {p} row"
                );
                assert_eq!(
                    want_top[(b * f + p) * 2 + 1],
                    u32::MAX,
                    "m={m} b={b} padding {p} slot"
                );
                assert_eq!(
                    want_lp[b * f + p].to_bits(),
                    0.0f32.to_bits(),
                    "m={m} b={b} padding {p} log-prob"
                );
            }
        }
        // Spot checks on the shaped spectra.
        assert_eq!(got_count[1], f as u32, "m={m}: all-equal row fills F");
        assert_eq!(got_top[(1 * f) * 2 + 1], 0, "m={m}: all-equal tie starts at slot 0");
        assert_eq!(got_top[(1 * f + 1) * 2 + 1], 1, "m={m}: all-equal tie order");
        assert_eq!(got_count[2], 2, "m={m}: fewer-flagged-than-F fills 2");
        assert_eq!(got_count[3], 0, "m={m}: zero-flagged fills 0");
    }
}

#[test]
fn formula_top_m2048_all_flagged_completes() {
    // The worst case: every slot flagged at M = 2048. With `F` successive
    // linear scans this is `F * M` comparisons per lane and completes
    // quickly on the CPU runtime.
    let device = dev();
    let (batch, m, f) = (2usize, 2048usize, 4usize);
    let mut rng = Rng::seeded(2048);
    let u = rng.uniform_vec(batch * m, -2.0, 0.0);
    let mut window = vec![0u32; batch * m * 2];
    for b in 0..batch {
        for slot in 0..m {
            window[(b * m + slot) * 2] = 7000 + slot as u32;
            window[(b * m + slot) * 2 + 1] = 1;
        }
    }
    let log_prob: Vec<f32> = u
        .iter()
        .map(|&v| (v * 2.0).round() / 2.0)
        .collect();
    let cand = cand_from_window(&window, batch, m);
    let lp_t = Tensor::<R, E>::from_f32(&log_prob, vec![batch, m], &device).unwrap();
    let cand_t = IdTensor::from_slice(&cand, vec![batch, m, 13], &device).unwrap();
    let out = ms2::FormulaBuffers::<R, E>::poisoned(batch, m, f, &device).unwrap();
    ms2::formula_top(&lp_t, &cand_t, &out).unwrap();
    check_launches(&device).unwrap();
    let (want_top, want_lp, want_count) = twin::formula_top(&log_prob, &window, batch, m, f);
    assert_eq!(out.top.try_to_vec().unwrap(), want_top);
    assert_bits_eq(&out.top_log_prob.try_to_f32().unwrap(), &want_lp, "worst-case log-probs");
    assert_eq!(out.top_count.try_to_vec().unwrap(), want_count);
    assert_eq!(want_count, vec![4, 4]);
}
