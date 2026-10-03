//! V0-A tests: the formula window and top-F kernels against their host
//! twins, the window mask, and the formula head (score, loss, gold slots).
//!
//! Every device call is followed by [`check_launches`], so a kernel that
//! failed to compile or run is an error rather than stale data.

#![cfg(feature = "backend")]

use mamba3::autograd::Var;
use mamba3::backend::{Device, check_launches};
use mamba3::backends::Auto;
use mamba3::models::ms2::chem::{Composition, composition_error_nda};
use mamba3::models::ms2::contract::{ModelConfig, request_status};
use mamba3::models::ms2::formula::{FormulaTable, WindowQuery};
use mamba3::models::ms2::formula_head::{
    DeviceFormulaTable, FormulaHead, gold_slots, gold_slots_host,
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
    let window_t = IdTensor::from_slice(&window, vec![2, m, 2], &device).unwrap();
    let lp_t = Tensor::<R, E>::from_f32(&log_prob, vec![2, m], &device).unwrap();
    let out = ms2::FormulaBuffers::<R, E>::poisoned(2, m, f, &device).unwrap();
    ms2::formula_top(&lp_t, &window_t, &out).unwrap();
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
    let window_t = IdTensor::from_slice(&got_window, vec![2, 16, 2], &device).unwrap();
    let lp_t = Tensor::<R, E>::from_f32(&lp, vec![2, 16], &device).unwrap();
    let out = ms2::FormulaBuffers::<R, E>::poisoned(2, 16, 4, &device).unwrap();
    ms2::formula_top(&lp_t, &window_t, &out).unwrap();
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
    // Both slots joined; the second holds NaN. The rank collides (NaN
    // compares false both ways), so one entry is written and the count
    // compacts to it: no crash, no out-of-bounds write, no padding slot
    // selected by `k mod top_count`.
    let (m, f) = (2usize, 2usize);
    let window: Vec<u32> = vec![40, 1, 41, 1];
    let log_prob: Vec<f32> = vec![0.0, f32::NAN];
    let window_t = IdTensor::from_slice(&window, vec![1, m, 2], &device).unwrap();
    let lp_t = Tensor::<R, E>::from_f32(&log_prob, vec![1, m], &device).unwrap();
    let out = ms2::FormulaBuffers::<R, E>::poisoned(1, m, f, &device).unwrap();
    ms2::formula_top(&lp_t, &window_t, &out).unwrap();
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
    // The mirrored order `[NaN, 0]`: both entries take rank 0, rank 1 is
    // never written, so `top_count` is the number of non-padding slots (1)
    // and every slot below it is a valid row index, on the kernel and the
    // twin.
    let log_prob: Vec<f32> = vec![f32::NAN, 0.0];
    let lp_t = Tensor::<R, E>::from_f32(&log_prob, vec![1, m], &device).unwrap();
    let out = ms2::FormulaBuffers::<R, E>::poisoned(1, m, f, &device).unwrap();
    ms2::formula_top(&lp_t, &window_t, &out).unwrap();
    check_launches(&device).unwrap();
    let (want_top, _, want_count) = twin::formula_top(&log_prob, &window, 1, m, f);
    assert_eq!(
        out.top.try_to_vec().unwrap(),
        want_top,
        "kernel matches twin"
    );
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
    let buffers = ms2::FormulaBuffers::<R, E>::new(2, m, 4, &device);
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
    let mut rng = Rng::seeded(17);
    let pool_host = rng.uniform_vec(2 * 8, -0.5, 0.5);
    let pool = Var::constant(Tensor::<R, E>::from_f32(&pool_host, vec![2, 8], &device).unwrap());
    let out = head.score(&uploaded, &buffers, &pool).unwrap();
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
    let buffers = ms2::FormulaBuffers::<R, E>::new(3, m, 4, &device);
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
        let out = head.score(&uploaded, &buffers, &pool_of(host)).unwrap();
        head.loss(&out, &gold_t).unwrap().to_f32()[0]
    };
    let out = head
        .score(&uploaded, &buffers, &pool_of(&pool_host))
        .unwrap();
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
