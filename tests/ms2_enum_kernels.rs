//! B2a tests: the four enumerating-formula kernels (`ms2_enum_count`,
//! `ms2_enum_offsets`, `ms2_enum_fill`, `ms2_cand_pad`) against their host
//! twins and the whole `enumerate_device_order` result.
//!
//! Every device call runs on poisoned outputs and is followed by
//! [`check_launches`], so a kernel that failed to compile or run is an error
//! rather than stale data: a dropped launch would leave the poison in place
//! and fail the element-wise comparison with the twin. Domains stay small
//! (the CPU runtime executes lanes on the host).

#![cfg(feature = "backend")]

use mamba3::backend::{Device, check_launches};
use mamba3::backends::Auto;
use mamba3::error::Error;
use mamba3::models::ms2::chem::{ELEMENTS, parent_mass, tolerance_u32};
use mamba3::models::ms2::contract::request_status;
use mamba3::models::ms2::contract::{SPECTRUM_SCHEMA_VERSION, SpectrumBatch};
use mamba3::models::ms2::formula_enum::{
    DEVICE_HALF_MAX, DeviceEnumLimits, ENUM_LANES_MAX_DEFAULT, EnumDomain, EnumQuery,
    LANE_EXHAUSTED_BIT, LANE_MODE_COUNT, LANE_MODE_FILL, LANE_RECORD_WORDS, LANE_VISITED_MASK,
    RatioBounds, build_enum_meta, enumerate_device_order, kernel_lane, kernel_offsets, kernel_pad,
    pack_device_bounds, rare_table, validate_enum_dispatch, ENUM_DISPATCH_VISITS_DEFAULT, ENUM_LANE_VISITS_DEFAULT,
};
use mamba3::models::ms2::{Composition, element_index};
use mamba3::tensor::ops::index::IdTensor;
use mamba3::tensor::ops::ms2_enum::{EnumChem, EnumLaunch, cand_pad, enum_count, enum_fill, enum_offsets};

type R = Auto;

fn dev() -> Device<R> {
    Device::<R>::default()
}

/// Poison pattern A: far from every legitimate value (counts, offsets and
/// counters stay small in these tests; visited words carry the top
/// exhausted bit only past billions of visits, which never happen here).
const POISON_A: u32 = u32::MAX - 7;
/// Poison pattern B: proves which kernel wrote which slot (see the capped
/// window test).
const POISON_B: u32 = u32::MAX - 13;

/// The fourteen chemistry scalars the lane twins take, read off the
/// production table (the kernels receive the same words as launch scalars).
fn chem14() -> [u32; 14] {
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

/// Build the 8-word spectrum meta row exactly as `enumerate_device_order`
/// derives it, or `None` when the host takes an early exit (unknown
/// precision, bad parent, wide window): those spectra never launch.
fn setup_meta(
    domain: &EnumDomain,
    query: &EnumQuery,
    budget: u32,
    cap: u32,
) -> Option<[u32; 8]> {
    if query.precursor_uncertainty == u32::MAX {
        return None;
    }
    let parent = parent_mass(query.precursor_mz, query.adduct).ok()?;
    let tol = tolerance_u32(query.precursor_mz, query.ppm_tenths).ok()?;
    let bound = query.precursor_uncertainty.saturating_add(1);
    let half = tol
        .saturating_add(bound)
        .saturating_add(domain.max_error());
    if half > DEVICE_HALF_MAX {
        return None;
    }
    Some([
        parent,
        tol,
        bound,
        parent.saturating_sub(half),
        parent.saturating_add(half),
        budget,
        cap,
        0,
    ])
}

/// Run the count twin for lane `r` of one spectrum into `stats`; returns the
/// `(joined, visited_word)` pair for convenience.
fn twin_count(
    meta: &[u32],
    packed: &[u32],
    rare_flat: &[u32],
    chem: &[u32; 14],
    r: usize,
    stats: &mut [u32],
) {
    let stats_len = stats.len() as u32;
    let _ = kernel_lane(
        meta,
        meta.len() as u32,
        packed,
        packed.len() as u32,
        rare_flat,
        rare_flat.len() as u32,
        0u32,
        r as u32 * 8u32,
        LANE_MODE_COUNT,
        0u32,
        0u32,
        stats,
        stats_len,
        0u32,
        r as u32 * 2u32,
        chem[0],
        chem[1],
        chem[2],
        chem[3],
        chem[4],
        chem[5],
        chem[6],
        chem[7],
        chem[8],
        chem[9],
        chem[10],
        chem[11],
        chem[12],
        chem[13],
        u32::MAX,
        meta[5],
    );
}

/// Run the fill twin for lane `r` of one spectrum into `out` at `out_base`.
/// Returns the raw visit count (without the exhausted bit) so capacity
/// exits are assertable.
#[allow(clippy::too_many_arguments)]
fn twin_fill(
    meta: &[u32],
    packed: &[u32],
    rare_flat: &[u32],
    chem: &[u32; 14],
    r: usize,
    offset: u32,
    cap: u32,
    out: &mut [u32],
    out_base: u32,
) -> u32 {
    let out_len = out.len() as u32;
    kernel_lane(
        meta,
        meta.len() as u32,
        packed,
        packed.len() as u32,
        rare_flat,
        rare_flat.len() as u32,
        0u32,
        r as u32 * 8u32,
        LANE_MODE_FILL,
        offset,
        cap,
        out,
        out_len,
        out_base,
        0u32,
        chem[0],
        chem[1],
        chem[2],
        chem[3],
        chem[4],
        chem[5],
        chem[6],
        chem[7],
        chem[8],
        chem[9],
        chem[10],
        chem[11],
        chem[12],
        chem[13],
        u32::MAX,
        meta[5],
    )
}

/// Run the offsets twin for one spectrum.
fn twin_offsets(stats: &[u32], meta: &[u32], n_p: usize, cap: u32) -> (Vec<u32>, Vec<u32>) {
    let mut offsets = vec![0u32; n_p];
    let mut counters = vec![0u32; 5];
    kernel_offsets(
        stats,
        stats.len() as u32,
        meta,
        meta.len() as u32,
        &mut offsets,
        n_p as u32,
        &mut counters,
        5u32,
        0u32,
        n_p as u32,
        cap,
        request_status::FORMULA_SEARCH_EXHAUSTED,
        request_status::FORMULA_ABSENT,
        u32::MAX,
    );
    (offsets, counters)
}

/// Decode one 13-word candidate record.
fn decode_record(words: &[u32]) -> ([u16; 10], u32, u32, u32) {
    let mut counts = [0u16; 10];
    for (i, c) in counts.iter_mut().enumerate() {
        *c = words[i] as u16;
    }
    (counts, words[10], words[11], words[12])
}

/// Outputs of one pipeline run.
struct PipeOut {
    lane_stats: Vec<u32>,
    offsets: Vec<u32>,
    counters: Vec<u32>,
    cand: Vec<u32>,
}

/// Run the four kernels over one `(B, P, M)` bucket. `meta_rows` carries the
/// per-spectrum words (budgets and the cap, already `<= M`, may differ per
/// row); `cap` is the scored cap passed to the offsets/fill wrappers.
/// Outputs start poisoned with `poison`; fill and pad run only when asked,
/// so authorship of every slot is attributable.
#[allow(clippy::too_many_arguments)]
fn run_pipeline(
    meta_rows: &[[u32; 8]],
    rare: &[[u32; 8]],
    packed: &[u32],
    m: usize,
    cap: u32,
    poison: u32,
    do_fill: bool,
    do_pad: bool,
) -> PipeOut {
    run_pipeline_capped(meta_rows, rare, packed, m, cap, cap, poison, do_fill, do_pad)
}

/// [`run_pipeline`] with separate wrapper caps for offsets and fill, so
/// mismatched dispatch capacities are exercisable (the effective cap is
/// `min(meta, wrapper, M)` in both lanes).
#[allow(clippy::too_many_arguments)]
fn run_pipeline_capped(
    meta_rows: &[[u32; 8]],
    rare: &[[u32; 8]],
    packed: &[u32],
    m: usize,
    offsets_cap: u32,
    fill_cap: u32,
    poison: u32,
    do_fill: bool,
    do_pad: bool,
) -> PipeOut {
    use mamba3::models::ms2::formula_enum::ENUM_LANES_MAX_DEFAULT;
    let device = dev();
    let b = meta_rows.len();
    let p = rare.len();
    assert!(
        offsets_cap <= m as u32 && fill_cap <= m as u32,
        "the scored caps fit the candidate width"
    );
    let chem = EnumChem::from_chemistry();
    let meta_flat: Vec<u32> = meta_rows.iter().flat_map(|r| r.iter().copied()).collect();
    let rare_flat: Vec<u32> = rare.iter().flat_map(|r| r.iter().copied()).collect();
    let meta_t = IdTensor::from_slice(&meta_flat, vec![b, 8], &device).unwrap();
    let rare_t = IdTensor::from_slice(&rare_flat, vec![p, 8], &device).unwrap();
    let bounds_t = IdTensor::from_slice(packed, vec![packed.len()], &device).unwrap();
    let stats_t =
        IdTensor::from_slice(&vec![poison; b * p * 2], vec![b * p, 2], &device).unwrap();
    enum_count(
        &meta_t,
        &rare_t,
        &bounds_t,
        &stats_t,
        &chem,
        ENUM_LANES_MAX_DEFAULT,
        ENUM_DISPATCH_VISITS_DEFAULT,
        ENUM_LANE_VISITS_DEFAULT,
    )
    .unwrap();
    check_launches(&device).unwrap();
    let offsets_t =
        IdTensor::from_slice(&vec![poison; b * p], vec![b * p], &device).unwrap();
    let counters_t =
        IdTensor::from_slice(&vec![poison; b * 5], vec![b, 5], &device).unwrap();
    enum_offsets(
        &stats_t,
        &meta_t,
        &offsets_t,
        &counters_t,
        offsets_cap,
        m,
        ENUM_LANES_MAX_DEFAULT,
    )
    .unwrap();
    check_launches(&device).unwrap();
    let cand_t =
        IdTensor::from_slice(&vec![poison; b * m * 13], vec![b, m, 13], &device).unwrap();
    if do_fill {
        enum_fill(
            &meta_t, &rare_t, &bounds_t, &offsets_t, &cand_t, &chem, fill_cap,
            ENUM_LANES_MAX_DEFAULT,
            ENUM_DISPATCH_VISITS_DEFAULT,
            ENUM_LANE_VISITS_DEFAULT,
        )
        .unwrap();
        check_launches(&device).unwrap();
    }
    if do_pad {
        cand_pad(&counters_t, &cand_t, p, ENUM_LANES_MAX_DEFAULT).unwrap();
        check_launches(&device).unwrap();
    }
    PipeOut {
        lane_stats: stats_t.try_to_vec().unwrap(),
        offsets: offsets_t.try_to_vec().unwrap(),
        counters: counters_t.try_to_vec().unwrap(),
        cand: cand_t.try_to_vec().unwrap(),
    }
}

/// Check one spectrum row end to end: `lane_stats` against the count twin
/// over all lanes, `offsets`/`counters` against the offsets twin, `cand`
/// against the fill twin below `scored` and the pad twin at or past it, and
/// everything against the whole `enumerate_device_order` result.
#[allow(clippy::too_many_arguments)]
fn check_row(
    domain: &EnumDomain,
    bounds: &RatioBounds,
    query: &EnumQuery,
    budget: u32,
    cap: u32,
    m: usize,
    meta: &[u32; 8],
    rare_flat: &[u32],
    packed: &[u32],
    chem: &[u32; 14],
    p: usize,
    out: &PipeOut,
    b: usize,
    what: &str,
) {
    // 1. Every lane_stats element against the count twin.
    let mut count_stats = vec![0u32; p * 2];
    for r in 0..p {
        twin_count(meta, packed, rare_flat, chem, r, &mut count_stats);
    }
    assert_eq!(
        &out.lane_stats[b * p * 2..(b + 1) * p * 2],
        &count_stats[..],
        "{what} row {b}: lane_stats against the count twin"
    );
    // 2. Every offsets/counters element against the offsets twin.
    let (offsets, counters) = twin_offsets(&count_stats, meta, p, cap);
    assert_eq!(
        &out.offsets[b * p..(b + 1) * p],
        &offsets[..],
        "{what} row {b}: offsets against the offsets twin"
    );
    assert_eq!(
        &out.counters[b * 5..(b + 1) * 5],
        &counters[..],
        "{what} row {b}: counters against the offsets twin"
    );
    // 3. Every cand element: fill records below `scored`, padding at/after.
    let scored = counters[2] as usize;
    assert!(
        scored <= m,
        "{what} row {b}: the scored prefix fits the candidate width"
    );
    let mut fill_buf = vec![0u32; scored * LANE_RECORD_WORDS as usize];
    for (r, off) in offsets.iter().enumerate() {
        twin_fill(meta, packed, rare_flat, chem, r, *off, cap, &mut fill_buf, 0);
    }
    let mut expect = vec![0u32; m * LANE_RECORD_WORDS as usize];
    expect[..fill_buf.len()].copy_from_slice(&fill_buf);
    let expect_len = expect.len() as u32;
    for slot in scored..m {
        kernel_pad(
            &mut expect,
            expect_len,
            0u32,
            slot as u32,
            scored as u32,
            m as u32,
            u32::MAX,
        );
    }
    assert_eq!(
        &out.cand[b * m * 13..(b + 1) * m * 13],
        &expect[..],
        "{what} row {b}: cand against the fill+pad twins"
    );
    // 4. The whole `enumerate_device_order` result.
    let found = enumerate_device_order(
        domain,
        bounds,
        query,
        &DeviceEnumLimits {
            lane_visits_max: budget,
            scored_cap: cap,
        },
    )
    .unwrap();
    assert_eq!(found.joined, counters[1], "{what} row {b}: joined");
    assert_eq!(found.visited, counters[0], "{what} row {b}: visited");
    assert_eq!(found.scored, scored as u32, "{what} row {b}: scored");
    assert_eq!(found.status, counters[3], "{what} row {b}: status");
    assert_eq!(
        found.complete,
        counters[4] != 0,
        "{what} row {b}: complete"
    );
    assert_eq!(
        found.exhausted,
        counters[4] == 0,
        "{what} row {b}: exhausted"
    );
    assert_eq!(
        found.absent,
        !found.exhausted && found.joined == 0,
        "{what} row {b}: absent"
    );
    for (r, lane) in found.lanes.iter().enumerate() {
        assert_eq!(lane.joined, count_stats[2 * r], "{what} row {b} lane {r}: joined");
        assert_eq!(
            lane.visited,
            count_stats[2 * r + 1] & LANE_VISITED_MASK,
            "{what} row {b} lane {r}: visited"
        );
        assert_eq!(
            lane.exhausted,
            count_stats[2 * r + 1] & LANE_EXHAUSTED_BIT != 0,
            "{what} row {b} lane {r}: exhausted"
        );
    }
    assert_eq!(
        found.compositions.len(),
        scored,
        "{what} row {b}: scored length"
    );
    for i in 0..scored {
        let base = i * LANE_RECORD_WORDS as usize;
        let (counts, mass, flag, source) = decode_record(&expect[base..base + 13]);
        assert_eq!(source, u32::MAX, "{what} row {b} slot {i}: source id");
        assert_eq!(mass, found.masses[i], "{what} row {b} slot {i}: mass");
        assert_eq!(counts, found.compositions[i], "{what} row {b} slot {i}: counts");
        assert_eq!(flag as u8, found.flags[i], "{what} row {b} slot {i}: flag");
    }
}

// ---------------------------------------------------------------------------
// Tiny brute-force domains.
// ---------------------------------------------------------------------------

/// Tiny C/N/O domain with a closed hydrogen range (brute-forceable).
fn tiny_domain() -> EnumDomain {
    EnumDomain {
        version: "test-tiny".to_string(),
        heavy_caps: [2, 1, 1, 0, 0, 0, 0, 0, 0],
        heavy_max: 4,
        hydrogen_min: 0,
        hydrogen_max: 6,
    }
}

/// Net hydrogen shift of contract §4.3, `m_H - m_e`, as test literals.
const H_NET: u32 = 1_007_825 - 549;

/// Protonated (`[M+H]+`) precursor m/z of a neutral integer mass.
fn protonated(mass: u32) -> u32 {
    mass + H_NET
}

/// Deprotonated (`[M-H]-`) precursor m/z of a neutral integer mass.
fn deprotonated(mass: u32) -> u32 {
    mass - H_NET
}

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

fn brute_mass(c: &Composition) -> u32 {
    let mut total: u64 = 0;
    for (e, n) in c.iter().enumerate() {
        total += u64::from(*n) * u64::from(BRUTE_MASS[e]);
    }
    assert!(total <= u64::from(u32::MAX), "brute mass fits u32");
    total as u32
}

/// A C/H/N/O composition; every other element is zero.
fn comp(c: u16, h: u16, n: u16, o: u16) -> Composition {
    [c, h, n, o, 0, 0, 0, 0, 0, 0]
}

#[test]
fn tiny_enum_kernels_match_twins_batched() {
    // Brute-force-scale C/N/O queries, both adducts, two tolerances, three
    // precursor deltas, alternating lane budgets — batched into one B = 24
    // pipeline so batching itself is exercised. Every output element is
    // compared with the host twin.
    let domain = tiny_domain();
    let subset: Vec<Composition> = vec![
        comp(1, 4, 0, 0),
        comp(1, 2, 0, 0),
        comp(0, 2, 1, 1),
        comp(2, 0, 0, 0),
        comp(1, 0, 0, 1),
    ];
    let bounds = RatioBounds::fit(subset.iter().copied(), 0).unwrap();
    let packed = pack_device_bounds(&domain, &bounds).unwrap();
    let rare = rare_table(&domain, &bounds).unwrap();
    assert_eq!(rare.len(), 1, "no rare elements here: a single lane");
    let p = rare.len();
    let rare_flat: Vec<u32> = rare.iter().flat_map(|r| r.iter().copied()).collect();
    let chem = chem14();
    let bases: Vec<Composition> = vec![
        comp(1, 4, 0, 0),
        comp(2, 6, 0, 1),
        comp(0, 2, 1, 1),
        comp(2, 0, 0, 0),
    ];
    let mut queries = Vec::new();
    let mut budgets = Vec::new();
    for base in &bases {
        for adduct in [1u16, 2] {
            let anchor = if adduct == 1 {
                protonated(brute_mass(base))
            } else {
                deprotonated(brute_mass(base))
            };
            for delta in [-1000i64, 0, 1000] {
                queries.push(EnumQuery {
                    precursor_mz: (anchor as i64 + delta) as u32,
                    adduct,
                    ppm_tenths: 200,
                    precursor_uncertainty: 500,
                });
                budgets.push(if queries.len() % 2 == 0 { 5000 } else { u32::MAX });
            }
        }
    }
    assert_eq!(queries.len(), 24);
    let m = 32usize;
    let cap = 32u32;
    let mut meta_rows = Vec::new();
    for (q, budget) in queries.iter().zip(budgets.iter()) {
        meta_rows.push(setup_meta(&domain, q, *budget, cap).expect("launchable row"));
    }
    let out = run_pipeline(&meta_rows, &rare, &packed, m, cap, POISON_A, true, true);
    for (b, ((query, budget), meta)) in queries
        .iter()
        .zip(budgets.iter())
        .zip(meta_rows.iter())
        .enumerate()
    {
        check_row(
            &domain, &bounds, query, *budget, cap, m, meta, &rare_flat, &packed, &chem, p,
            &out, b, "tiny",
        );
    }
}

#[test]
fn tiny_enum_batch_independence_b3() {
    // B = 3 with a different query per row: each row equals its single-row
    // run, element for element, across all four outputs.
    let domain = tiny_domain();
    let subset: Vec<Composition> =
        vec![comp(1, 4, 0, 0), comp(0, 2, 1, 1), comp(2, 0, 0, 0)];
    let bounds = RatioBounds::fit(subset.iter().copied(), 0).unwrap();
    let packed = pack_device_bounds(&domain, &bounds).unwrap();
    let rare = rare_table(&domain, &bounds).unwrap();
    let m = 32usize;
    let cap = 32u32;
    let queries = [
        EnumQuery {
            precursor_mz: protonated(brute_mass(&comp(1, 4, 0, 0))),
            adduct: 1,
            ppm_tenths: 1000,
            precursor_uncertainty: 500,
        },
        EnumQuery {
            precursor_mz: deprotonated(brute_mass(&comp(0, 2, 1, 1))),
            adduct: 2,
            ppm_tenths: 100,
            precursor_uncertainty: 50,
        },
        EnumQuery {
            precursor_mz: protonated(brute_mass(&comp(2, 0, 0, 0))) + 500,
            adduct: 1,
            ppm_tenths: 200,
            precursor_uncertainty: 5000,
        },
    ];
    let budgets = [u32::MAX, 5000, u32::MAX];
    let meta_rows: Vec<[u32; 8]> = queries
        .iter()
        .zip(budgets.iter())
        .map(|(q, budget)| setup_meta(&domain, q, *budget, cap).expect("launchable row"))
        .collect();
    let joint = run_pipeline(&meta_rows, &rare, &packed, m, cap, POISON_A, true, true);
    let p = rare.len();
    for (b, meta) in meta_rows.iter().enumerate() {
        let single = run_pipeline(&[*meta], &rare, &packed, m, cap, POISON_A, true, true);
        assert_eq!(
            &joint.lane_stats[b * p * 2..(b + 1) * p * 2],
            &single.lane_stats[..],
            "row {b}: lane_stats batch independence"
        );
        assert_eq!(
            &joint.offsets[b * p..(b + 1) * p],
            &single.offsets[..],
            "row {b}: offsets batch independence"
        );
        assert_eq!(
            &joint.counters[b * 5..(b + 1) * 5],
            &single.counters[..],
            "row {b}: counters batch independence"
        );
        assert_eq!(
            &joint.cand[b * m * 13..(b + 1) * m * 13],
            &single.cand[..],
            "row {b}: cand batch independence"
        );
    }
}

// ---------------------------------------------------------------------------
// All nine heavy elements across the cases.
// ---------------------------------------------------------------------------

/// Small domain admitting every heavy element (C, N, O, F, P, S, Cl, Br, I).
fn nine_domain() -> EnumDomain {
    EnumDomain {
        version: "test-nine".to_string(),
        heavy_caps: [2, 1, 2, 1, 1, 1, 1, 1, 1],
        heavy_max: 8,
        hydrogen_min: 0,
        hydrogen_max: 8,
    }
}

/// One exact-filter-passing carrier per heavy element (each query joins its
/// own centre, so the union covers all nine).
fn nine_subset() -> Vec<Composition> {
    vec![
        [1, 4, 0, 0, 0, 0, 0, 0, 0, 0], // C
        [1, 5, 1, 0, 0, 0, 0, 0, 0, 0], // N
        [2, 2, 0, 2, 0, 0, 0, 0, 0, 0], // O
        [1, 3, 0, 0, 1, 0, 0, 0, 0, 0], // F
        [1, 5, 0, 0, 0, 1, 0, 0, 0, 0], // P
        [2, 6, 0, 0, 0, 0, 1, 0, 0, 0], // S
        [1, 3, 0, 0, 0, 0, 0, 1, 0, 0], // Cl
        [1, 3, 0, 0, 0, 0, 0, 0, 1, 0], // Br
        [1, 3, 0, 0, 0, 0, 0, 0, 0, 1], // I
    ]
}

#[test]
fn nine_heavy_elements_join_across_cases() {
    let domain = nine_domain();
    let subset = nine_subset();
    let bounds = RatioBounds::fit(subset.iter().copied(), 0).unwrap();
    assert!(
        mamba3::models::ms2::formula_enum::validate_device_artifacts(&domain, &bounds).is_ok()
    );
    let packed = pack_device_bounds(&domain, &bounds).unwrap();
    let rare = rare_table(&domain, &bounds).unwrap();
    assert!(!rare.is_empty() && rare.len() <= 256, "P stays small on CPU");
    let p = rare.len();
    let rare_flat: Vec<u32> = rare.iter().flat_map(|r| r.iter().copied()).collect();
    let chem = chem14();
    // One query per carrier (both adducts for the first two), batched.
    let mut queries = Vec::new();
    for (i, carrier) in subset.iter().enumerate() {
        for adduct in [1u16, 2] {
            if adduct == 2 && i > 1 {
                continue;
            }
            let anchor = if adduct == 1 {
                protonated(brute_mass(carrier))
            } else {
                deprotonated(brute_mass(carrier))
            };
            queries.push((
                EnumQuery {
                    precursor_mz: anchor,
                    adduct,
                    ppm_tenths: 1000,
                    precursor_uncertainty: 500,
                },
                *carrier,
            ));
        }
    }
    let m = 64usize;
    let cap = 64u32;
    let meta_rows: Vec<[u32; 8]> = queries
        .iter()
        .map(|(q, _)| setup_meta(&domain, q, u32::MAX, cap).expect("launchable row"))
        .collect();
    let out = run_pipeline(&meta_rows, &rare, &packed, m, cap, POISON_A, true, true);
    let mut covered = [false; 9];
    for (b, ((query, carrier), meta)) in queries.iter().zip(meta_rows.iter()).enumerate() {
        check_row(
            &domain, &bounds, query, u32::MAX, cap, m, meta, &rare_flat, &packed, &chem,
            p, &out, b, "nine",
        );
        let scored = out.counters[b * 5 + 2] as usize;
        let mut carrier_joined = false;
        for i in 0..scored {
            let base = (b * m + i) * 13;
            let (counts, _, _, _) = decode_record(&out.cand[base..base + 13]);
            if counts == *carrier {
                carrier_joined = true;
            }
            for (e, present) in covered.iter_mut().enumerate() {
                let element = [0usize, 2, 3, 4, 5, 6, 7, 8, 9][e];
                if counts[element] > 0 {
                    *present = true;
                }
            }
        }
        assert!(
            carrier_joined,
            "row {b}: the query joins its own carrier centre"
        );
    }
    assert!(
        covered.iter().all(|c| *c),
        "every heavy element joins somewhere: {covered:?}"
    );
}

// ---------------------------------------------------------------------------
// Chemistry-fixture sweep.
// ---------------------------------------------------------------------------

fn fixture_compositions() -> Vec<Composition> {
    let text = std::fs::read_to_string(
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/ms2/chemistry_v0.json"),
    )
    .expect("fixture readable");
    let fixture: serde_json::Value = serde_json::from_str(&text).expect("fixture parses");
    let mut comps: Vec<Composition> = fixture["molecules"]
        .as_array()
        .expect("molecules")
        .iter()
        .map(|mol| {
            let mut c: Composition = [0; 10];
            for (symbol, count) in mol["formula"].as_object().expect("formula") {
                c[element_index(symbol).expect("known element")] =
                    count.as_u64().unwrap() as u16;
            }
            c
        })
        .collect();
    comps.sort_by_key(brute_mass);
    comps
}

#[test]
fn fixture_sweep_kernels_match_device_order() {
    // A sweep of fixture-derived queries, both adducts, batched: every
    // output element against the twins and the whole device-order result.
    let mut comps = fixture_compositions();
    comps.truncate(6);
    let domain = EnumDomain::from_compositions(comps.iter().copied(), 0).unwrap();
    let bounds = RatioBounds::fit(comps.iter().copied(), 0).unwrap();
    assert!(
        mamba3::models::ms2::formula_enum::validate_device_artifacts(&domain, &bounds).is_ok()
    );
    let packed = pack_device_bounds(&domain, &bounds).unwrap();
    let rare = rare_table(&domain, &bounds).unwrap();
    assert!(!rare.is_empty() && rare.len() <= 256, "P stays small on CPU");
    let p = rare.len();
    let rare_flat: Vec<u32> = rare.iter().flat_map(|r| r.iter().copied()).collect();
    let chem = chem14();
    let mut queries = Vec::new();
    for gold in &comps {
        for adduct in [1u16, 2] {
            let anchor = if adduct == 1 {
                protonated(brute_mass(gold))
            } else {
                deprotonated(brute_mass(gold))
            };
            queries.push(EnumQuery {
                precursor_mz: anchor,
                adduct,
                ppm_tenths: 200,
                precursor_uncertainty: 50,
            });
        }
    }
    let m = 128usize;
    let cap = 128u32;
    let meta_rows: Vec<[u32; 8]> = queries
        .iter()
        .map(|q| setup_meta(&domain, q, u32::MAX, cap).expect("launchable row"))
        .collect();
    let out = run_pipeline(&meta_rows, &rare, &packed, m, cap, POISON_A, true, true);
    let mut compared = 0usize;
    let mut gold_joined = 0usize;
    for (b, (query, meta)) in queries.iter().zip(meta_rows.iter()).enumerate() {
        let found = enumerate_device_order(
            &domain,
            &bounds,
            query,
            &DeviceEnumLimits {
                lane_visits_max: u32::MAX,
                scored_cap: cap,
            },
        )
        .unwrap();
        if found.exhausted {
            continue;
        }
        check_row(
            &domain, &bounds, query, u32::MAX, cap, m, meta, &rare_flat, &packed, &chem,
            p, &out, b, "fixture",
        );
        compared += 1;
        if found.scored_contains(&comps[b / 2]) {
            gold_joined += 1;
        }
    }
    assert!(compared > 0, "non-vacuous sweep");
    assert!(gold_joined > 0, "some gold joins");
}

// ---------------------------------------------------------------------------
// Lane budgets: 0, 1 and exact fit.
// ---------------------------------------------------------------------------

#[test]
fn lane_budgets_zero_one_exact_fit() {
    // One rare lane keeps attribution trivial; the open run fixes the exact
    // visit count, and budgets 0 / 1 / exact / exact − 1 behave.
    let domain = tiny_domain();
    let subset: Vec<Composition> = vec![comp(1, 4, 0, 0), comp(0, 2, 1, 1), comp(2, 0, 0, 0)];
    let bounds = RatioBounds::fit(subset.iter().copied(), 0).unwrap();
    let packed = pack_device_bounds(&domain, &bounds).unwrap();
    let rare = rare_table(&domain, &bounds).unwrap();
    assert_eq!(rare.len(), 1);
    let p = 1usize;
    let rare_flat: Vec<u32> = rare.iter().flat_map(|r| r.iter().copied()).collect();
    let chem = chem14();
    let base = comp(1, 4, 0, 0);
    let query = EnumQuery {
        precursor_mz: protonated(brute_mass(&base)),
        adduct: 1,
        ppm_tenths: 1000,
        precursor_uncertainty: 500_000,
    };
    let m = 64usize;
    let cap = 64u32;
    // Open run through the kernels: the reference visit count.
    let meta_open = setup_meta(&domain, &query, u32::MAX, cap).expect("launchable");
    let open = run_pipeline(&[meta_open], &rare, &packed, m, cap, POISON_A, true, true);
    check_row(
        &domain, &bounds, &query, u32::MAX, cap, m, &meta_open, &rare_flat, &packed, &chem,
        p, &open, 0, "budget-open",
    );
    let visits_open = open.counters[0];
    assert!(visits_open > 1, "the window visits several vectors");
    assert_eq!(open.counters[4], 1, "the open run completes");
    // Budget 0: nothing visited, nothing joined, the lane exhausted.
    let meta_zero = setup_meta(&domain, &query, 0, cap).expect("launchable");
    let zero = run_pipeline(&[meta_zero], &rare, &packed, m, cap, POISON_A, true, true);
    assert_eq!(&zero.lane_stats, &[0, LANE_EXHAUSTED_BIT]);
    assert_eq!(&zero.offsets, &[0]);
    assert_eq!(
        &zero.counters,
        &[
            0,
            0,
            0,
            request_status::FORMULA_SEARCH_EXHAUSTED,
            0
        ]
    );
    for slot in 0..m {
        let base = slot * 13;
        assert_eq!(&zero.cand[base..base + 12], &[0u32; 12]);
        assert_eq!(zero.cand[base + 12], u32::MAX);
    }
    // Budget 1: exactly one visit, then exhaustion.
    let meta_one = setup_meta(&domain, &query, 1, cap).expect("launchable");
    let one = run_pipeline(&[meta_one], &rare, &packed, m, cap, POISON_A, true, true);
    check_row(
        &domain, &bounds, &query, 1, cap, m, &meta_one, &rare_flat, &packed, &chem, p,
        &one, 0, "budget-one",
    );
    assert_eq!(one.counters[0], 1, "exactly one visit at budget 1");
    assert_eq!(
        one.counters[3],
        request_status::FORMULA_SEARCH_EXHAUSTED,
        "budget 1 exhausts"
    );
    // Exact-fit budget reproduces the open run with no exhaustion.
    let meta_fit = setup_meta(&domain, &query, visits_open, cap).expect("launchable");
    let fitted = run_pipeline(&[meta_fit], &rare, &packed, m, cap, POISON_A, true, true);
    check_row(
        &domain, &bounds, &query, visits_open, cap, m, &meta_fit, &rare_flat, &packed,
        &chem, p, &fitted, 0, "budget-fit",
    );
    assert_eq!(fitted.counters[4], 1, "exact-fit budget completes");
    assert_eq!(fitted.lane_stats, open.lane_stats);
    assert_eq!(fitted.offsets, open.offsets);
    assert_eq!(fitted.cand, open.cand);
    // One below the exact fit exhausts with exactly that many visits.
    let meta_tight =
        setup_meta(&domain, &query, visits_open - 1, cap).expect("launchable");
    let tight = run_pipeline(&[meta_tight], &rare, &packed, m, cap, POISON_A, true, true);
    check_row(
        &domain,
        &bounds,
        &query,
        visits_open - 1,
        cap,
        m,
        &meta_tight,
        &rare_flat,
        &packed,
        &chem,
        p,
        &tight,
        0,
        "budget-tight",
    );
    assert_eq!(
        tight.counters[3],
        request_status::FORMULA_SEARCH_EXHAUSTED,
        "one below the exact fit exhausts"
    );
    assert_eq!(tight.counters[0], visits_open - 1);
}

// ---------------------------------------------------------------------------
// Capped window: fill and pad cover every slot exactly once.
// ---------------------------------------------------------------------------

/// Rare-tiny domain with an F/S tail for the multi-row capped query.
fn capped_domain() -> EnumDomain {
    EnumDomain {
        version: "test-capped".to_string(),
        heavy_caps: [3, 1, 2, 2, 0, 1, 0, 0, 0],
        heavy_max: 6,
        hydrogen_min: 0,
        hydrogen_max: 8,
    }
}

fn capped_subset() -> Vec<Composition> {
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

#[test]
fn capped_window_covers_slots_exactly_once() {
    // joined > M with M = 1: the fill kernel owns slot 0 (poison A survives
    // everywhere else), the pad kernel owns every later slot (poison B
    // survives in slot 0). Two patterns, both directions.
    let domain = capped_domain();
    let subset = capped_subset();
    let bounds = RatioBounds::fit(subset.iter().copied(), 0).unwrap();
    let packed = pack_device_bounds(&domain, &bounds).unwrap();
    let rare = rare_table(&domain, &bounds).unwrap();
    let center: Composition = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0];
    let query = EnumQuery {
        precursor_mz: protonated(brute_mass(&center)),
        adduct: 1,
        ppm_tenths: 1000,
        precursor_uncertainty: 100_000,
    };
    let open = enumerate_device_order(
        &domain,
        &bounds,
        &query,
        &DeviceEnumLimits {
            lane_visits_max: u32::MAX,
            scored_cap: u32::MAX,
        },
    )
    .unwrap();
    assert!(!open.exhausted);
    assert!(
        open.joined >= 2,
        "the capped query joins several rows: {}",
        open.joined
    );
    let m = 1usize;
    let cap = 1u32;
    let meta = setup_meta(&domain, &query, u32::MAX, cap).expect("launchable");
    // Phase 1 (pattern A): fill only. Slot 0 holds rank 0; the poison
    // survives in every later slot, proving fill wrote nothing there.
    let filled = run_pipeline(&[meta], &rare, &packed, m, cap, POISON_A, true, false);
    assert_eq!(filled.counters[2], 1, "scored is the cap");
    assert_eq!(filled.counters[1], open.joined, "joined counts every join");
    assert_eq!(
        filled.counters[3],
        request_status::FORMULA_SEARCH_EXHAUSTED,
        "truncation exhausts"
    );
    assert_eq!(filled.counters[4], 0, "truncation clears complete");
    let (counts, mass, flag, source) = decode_record(&filled.cand[0..13]);
    assert_eq!(mass, open.masses[0], "slot 0 is rank 0");
    assert_eq!(counts, open.compositions[0]);
    assert_eq!(flag as u8, open.flags[0]);
    assert_eq!(source, u32::MAX);
    // Phase 2 (pattern B): pad only. Slot 0 keeps pattern B, proving pad
    // wrote nothing below `scored`; every later slot is padding.
    let padded = run_pipeline(&[meta], &rare, &packed, m, cap, POISON_B, false, true);
    assert_eq!(&padded.cand[0..13], &[POISON_B; 13]);
    // A full run covers every slot exactly once with no poison left.
    let full = run_pipeline(&[meta], &rare, &packed, m, cap, POISON_A, true, true);
    assert!(
        !full.cand.contains(&POISON_A),
        "no poison survives the fill+pad cover"
    );
    assert_eq!(&full.cand[0..13], &filled.cand[0..13]);
}

// ---------------------------------------------------------------------------
// Host-setup exits: wide window, mass overflow, unknown precision.
// ---------------------------------------------------------------------------

#[test]
fn wide_window_joins_nothing_exhausted() {
    // `half` above 1,511,737 is `formula_search_exhausted` with nothing
    // joined; the lanes never launch (a wide window admits unbounded
    // hydrogen ranges, so there is no meta row to launch with).
    let domain = tiny_domain();
    let subset: Vec<Composition> = vec![comp(1, 4, 0, 0), comp(2, 0, 0, 0)];
    let bounds = RatioBounds::fit(subset.iter().copied(), 0).unwrap();
    let ch4 = comp(1, 4, 0, 0);
    assert_eq!(domain.max_error(), 1);
    let query = EnumQuery {
        precursor_mz: protonated(brute_mass(&ch4)),
        adduct: 1,
        ppm_tenths: 0,
        precursor_uncertainty: DEVICE_HALF_MAX - 1,
    };
    assert_eq!(setup_meta(&domain, &query, u32::MAX, 32), None);
    let wide = enumerate_device_order(
        &domain,
        &bounds,
        &query,
        &DeviceEnumLimits {
            lane_visits_max: u32::MAX,
            scored_cap: 32,
        },
    )
    .unwrap();
    assert!(wide.exhausted);
    assert_eq!(wide.joined, 0);
    assert_eq!(wide.scored, 0);
    assert!(!wide.absent);
    assert!(!wide.complete);
    assert!(wide.lanes.iter().all(|l| l.visited == 0 && l.joined == 0));
}

#[test]
fn mass_overflow_never_launches() {
    // An invalid parent mass is `mass_overflow` with no search: precursor 0
    // under adduct 1 underflows the parent mass, and `u32::MAX` under
    // adduct 2 overflows it.
    let domain = tiny_domain();
    let subset: Vec<Composition> = vec![comp(1, 4, 0, 0)];
    let bounds = RatioBounds::fit(subset.iter().copied(), 0).unwrap();
    for query in [
        EnumQuery {
            precursor_mz: 0,
            adduct: 1,
            ppm_tenths: 200,
            precursor_uncertainty: 50,
        },
        EnumQuery {
            precursor_mz: u32::MAX,
            adduct: 2,
            ppm_tenths: 200,
            precursor_uncertainty: 50,
        },
    ] {
        assert_eq!(setup_meta(&domain, &query, u32::MAX, 32), None);
        let found = enumerate_device_order(
            &domain,
            &bounds,
            &query,
            &DeviceEnumLimits {
                lane_visits_max: u32::MAX,
                scored_cap: 32,
            },
        )
        .unwrap();
        assert_eq!(found.parent_mass, None);
        assert_eq!(found.status, request_status::MASS_OVERFLOW);
        assert_eq!(found.joined, 0);
        assert!(!found.exhausted);
    }
}

#[test]
fn unknown_precision_never_launches() {
    // `u32::MAX` precursor uncertainty is the unknown-precision sentinel:
    // nothing is searched, `exact_mass_unavailable` with `formula_absent`.
    let domain = tiny_domain();
    let subset: Vec<Composition> = vec![comp(1, 4, 0, 0)];
    let bounds = RatioBounds::fit(subset.iter().copied(), 0).unwrap();
    let query = EnumQuery {
        precursor_mz: protonated(brute_mass(&comp(1, 4, 0, 0))),
        adduct: 1,
        ppm_tenths: 200,
        precursor_uncertainty: u32::MAX,
    };
    assert_eq!(setup_meta(&domain, &query, u32::MAX, 32), None);
    let found = enumerate_device_order(
        &domain,
        &bounds,
        &query,
        &DeviceEnumLimits {
            lane_visits_max: u32::MAX,
            scored_cap: 32,
        },
    )
    .unwrap();
    assert_eq!(
        found.status,
        request_status::EXACT_MASS_UNAVAILABLE | request_status::FORMULA_ABSENT
    );
    assert!(found.absent);
    assert!(!found.exhausted);
}

// ---------------------------------------------------------------------------
// Failed rows and batch independence.
// ---------------------------------------------------------------------------

#[test]
fn failed_row_batch_independence_b3() {
    // B = 3 with different queries per row and a budget-0 failed middle row
    // (the device-side encoding of a row that failed upstream: its lanes
    // join nothing and report exhaustion). Good rows equal their single-row
    // runs; the failed row joins nothing, pads fully and reports exhausted.
    let domain = tiny_domain();
    let subset: Vec<Composition> =
        vec![comp(1, 4, 0, 0), comp(0, 2, 1, 1), comp(2, 0, 0, 0)];
    let bounds = RatioBounds::fit(subset.iter().copied(), 0).unwrap();
    let packed = pack_device_bounds(&domain, &bounds).unwrap();
    let rare = rare_table(&domain, &bounds).unwrap();
    let p = rare.len();
    let m = 32usize;
    let cap = 32u32;
    let good_a = EnumQuery {
        precursor_mz: protonated(brute_mass(&comp(1, 4, 0, 0))),
        adduct: 1,
        ppm_tenths: 1000,
        precursor_uncertainty: 500,
    };
    let good_b = EnumQuery {
        precursor_mz: deprotonated(brute_mass(&comp(2, 0, 0, 0))),
        adduct: 2,
        ppm_tenths: 200,
        precursor_uncertainty: 500,
    };
    let mut meta_failed =
        setup_meta(&domain, &good_a, u32::MAX, cap).expect("launchable row");
    meta_failed[5] = 0;
    let meta_rows = [
        setup_meta(&domain, &good_a, u32::MAX, cap).expect("launchable row"),
        meta_failed,
        setup_meta(&domain, &good_b, u32::MAX, cap).expect("launchable row"),
    ];
    let joint = run_pipeline(&meta_rows, &rare, &packed, m, cap, POISON_A, true, true);
    let queries = [good_a, good_a, good_b];
    let budgets = [u32::MAX, 0, u32::MAX];
    let rows = [0usize, 1usize, 2usize];
    let rare_flat: Vec<u32> = rare.iter().flat_map(|r| r.iter().copied()).collect();
    let chem = chem14();
    for (b, row) in rows.iter().enumerate() {
        if *row != 1 {
            let single = run_pipeline(
                &[meta_rows[*row]],
                &rare,
                &packed,
                m,
                cap,
                POISON_A,
                true,
                true,
            );
            assert_eq!(
                &joint.lane_stats[*row * p * 2..(*row + 1) * p * 2],
                &single.lane_stats[..],
                "good row {row}: batch independence"
            );
            assert_eq!(
                &joint.cand[*row * m * 13..(*row + 1) * m * 13],
                &single.cand[..],
                "good row {row}: cand batch independence"
            );
        }
        check_row(
            &domain,
            &bounds,
            &queries[b],
            budgets[b],
            cap,
            m,
            &meta_rows[*row],
            &rare_flat,
            &packed,
            &chem,
            p,
            &joint,
            *row,
            "failed-batch",
        );
    }
    // The failed row: every lane `(0 joined, exhausted)`, exhausted
    // counters, a fully padded candidate row.
    assert_eq!(
        &joint.lane_stats[p * 2..2 * p * 2],
        &vec![0u32; p * 2]
            .iter()
            .enumerate()
            .map(|(i, _)| if i % 2 == 0 { 0 } else { LANE_EXHAUSTED_BIT })
            .collect::<Vec<u32>>()[..]
    );
    assert_eq!(
        &joint.counters[5..10],
        &[0, 0, 0, request_status::FORMULA_SEARCH_EXHAUSTED, 0]
    );
    for slot in 0..m {
        let base = (m * 13) + slot * 13;
        assert_eq!(&joint.cand[base..base + 12], &[0u32; 12]);
        assert_eq!(joint.cand[base + 12], u32::MAX);
    }
}

// ---------------------------------------------------------------------------
// Shape errors.
// ---------------------------------------------------------------------------

#[test]
fn shape_errors_are_shape() {
    let device = dev();
    let chem = EnumChem::from_chemistry();
    let ok = |data: &[u32], shape: Vec<usize>| IdTensor::from_slice(data, shape, &device).unwrap();
    let meta = ok(&[0u32; 2 * 8], vec![2, 8]);
    let rare = ok(&[0u32; 3 * 8], vec![3, 8]);
    let bounds = ok(&[0u32; 64], vec![64]);
    let stats = ok(&[0u32; 6 * 2], vec![6, 2]);
    let offsets = ok(&[0u32; 6], vec![6]);
    let counters = ok(&[0u32; 2 * 5], vec![2, 5]);
    let cand = ok(&[0u32; 2 * 4 * 13], vec![2, 4, 13]);
    let is_shape = |r: Result<(), Error>| assert!(matches!(r, Err(Error::Shape(_))));
    // enum_count.
    enum_count(&meta, &rare, &bounds, &stats, &chem, u32::MAX, u32::MAX, 4096).unwrap();
    is_shape(enum_count(
        &ok(&[0u32; 2], vec![2]),
        &rare,
        &bounds,
        &stats,
        &chem,
        u32::MAX,
        u32::MAX,
        4096,
    ));
    is_shape(enum_count(
        &ok(&[0u32; 2 * 7], vec![2, 7]),
        &rare,
        &bounds,
        &stats,
        &chem,
        u32::MAX,
        u32::MAX,
        4096,
    ));
    is_shape(enum_count(
        &meta,
        &ok(&[0u32; 3 * 7], vec![3, 7]),
        &bounds,
        &stats,
        &chem,
        u32::MAX,
        u32::MAX,
        4096,
    ));
    is_shape(enum_count(
        &meta,
        &rare,
        &ok(&[0u32; 64], vec![8, 8]),
        &stats,
        &chem,
        u32::MAX,
        u32::MAX,
        4096,
    ));
    is_shape(enum_count(
        &meta,
        &rare,
        &bounds,
        &ok(&[0u32; 6 * 2], vec![2, 3, 2]),
        &chem,
        u32::MAX,
        u32::MAX,
        4096,
    ));
    is_shape(enum_count(
        &meta,
        &rare,
        &bounds,
        &ok(&[0u32; 5 * 2], vec![5, 2]),
        &chem,
        u32::MAX,
        u32::MAX,
        4096,
    ));
    // enum_offsets.
    enum_offsets(&stats, &meta, &offsets, &counters, 4, 4, u32::MAX).unwrap();
    is_shape(enum_offsets(
        &ok(&[0u32; 6], vec![6]),
        &meta,
        &offsets,
        &counters,
        4,
        4,
        u32::MAX,
    ));
    is_shape(enum_offsets(
        &stats,
        &meta,
        &ok(&[0u32; 6], vec![2, 3]),
        &counters,
        4,
        4,
        u32::MAX,
    ));
    is_shape(enum_offsets(
        &stats,
        &meta,
        &ok(&[0u32; 5], vec![5]),
        &counters,
        4,
        4,
        u32::MAX,
    ));
    is_shape(enum_offsets(
        &stats,
        &meta,
        &offsets,
        &ok(&[0u32; 2 * 4], vec![2, 4]),
        4,
        4,
        u32::MAX,
    ));
    // enum_fill.
    enum_fill(&meta, &rare, &bounds, &offsets, &cand, &chem, 4, u32::MAX, u32::MAX, 4096).unwrap();
    is_shape(enum_fill(
        &meta,
        &rare,
        &bounds,
        &offsets,
        &ok(&[0u32; 8], vec![2, 4]),
        &chem,
        4,
        u32::MAX,
        u32::MAX,
        4096,
    ));
    is_shape(enum_fill(
        &meta,
        &rare,
        &bounds,
        &offsets,
        &ok(&[0u32; 2 * 4 * 12], vec![2, 4, 12]),
        &chem,
        4,
        u32::MAX,
        u32::MAX,
        4096,
    ));
    // cand_pad.
    cand_pad(&counters, &cand, 3, u32::MAX).unwrap();
    is_shape(cand_pad(
        &ok(&[0u32; 2], vec![2]),
        &cand,
        3,
        u32::MAX,
    ));
    is_shape(cand_pad(
        &counters,
        &ok(&[0u32; 2 * 4 * 13], vec![2, 13, 4]),
        3,
        u32::MAX,
    ));
}

// ---------------------------------------------------------------------------
// EF finding 1: one effective scored cap across offsets, fill and pad.
// ---------------------------------------------------------------------------

/// The reviewer's O2/S domain: O=2, S=1, every other heavy count and
/// hydrogen zero, heavy maximum 2. Bounds fitted from O2 and S give two rare
/// lanes (S absent / S present); the query joins both, in different lanes.
fn o2s_domain() -> EnumDomain {
    EnumDomain {
        version: "test-o2s".to_string(),
        heavy_caps: [0, 0, 2, 0, 0, 1, 0, 0, 0],
        heavy_max: 2,
        hydrogen_min: 0,
        hydrogen_max: 0,
    }
}

fn o2s_subset() -> Vec<Composition> {
    vec![
        [0, 0, 0, 2, 0, 0, 0, 0, 0, 0], // O2
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0], // S
    ]
}

/// The reviewer's query: precursor 32,979,347 (protonated S), adduct 1,
/// ppm 1000, uncertainty 100,000. Both O2 and S join (ambiguous).
fn o2s_query() -> EnumQuery {
    let s: Composition = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0];
    let precursor = protonated(brute_mass(&s));
    assert_eq!(precursor, 32_979_347, "the reviewer's precursor");
    EnumQuery {
        precursor_mz: precursor,
        adduct: 1,
        ppm_tenths: 1000,
        precursor_uncertainty: 100_000,
    }
}

fn o2s_setup() -> (EnumDomain, RatioBounds, Vec<[u32; 8]>, Vec<u32>, EnumQuery) {
    let domain = o2s_domain();
    let subset = o2s_subset();
    let bounds = RatioBounds::fit(subset.iter().copied(), 0).unwrap();
    assert!(
        mamba3::models::ms2::formula_enum::validate_device_artifacts(&domain, &bounds).is_ok()
    );
    let packed = pack_device_bounds(&domain, &bounds).unwrap();
    let rare = rare_table(&domain, &bounds).unwrap();
    assert_eq!(rare.len(), 2, "O2 and S join in different rare lanes");
    let query = o2s_query();
    (domain, bounds, rare, packed, query)
}

/// Meta row with an explicit per-spectrum cap word (word 6), budget open.
fn meta_capped(domain: &EnumDomain, query: &EnumQuery, meta_cap: u32) -> [u32; 8] {
    let mut meta = setup_meta(domain, query, u32::MAX, meta_cap).expect("launchable row");
    meta[6] = meta_cap;
    meta
}

#[test]
fn single_writer_cap_unification_o2_s() {
    // Fill-only and pad-only launches with differing caps (0, 1, < joined,
    // == joined, > joined, and M > scored): every slot has exactly one
    // writer. The effective cap is `min(meta word 6, wrapper cap, M)` in
    // offsets, fill and pad alike; a cap-0 spectrum gets no fill write.
    let (domain, bounds, rare, packed, query) = o2s_setup();
    let p = rare.len();
    let rare_flat: Vec<u32> = rare.iter().flat_map(|r| r.iter().copied()).collect();
    let chem = chem14();
    // Open reference: joined J == 2, both ambiguous.
    let open_meta = meta_capped(&domain, &query, 4);
    let open = run_pipeline(&[open_meta], &rare, &packed, 4, 4, POISON_A, true, true);
    assert_eq!(open.counters[1], 2, "O2 and S both join");
    assert_eq!(open.counters[2], 2, "scored is joined below the cap");
    let o2: Composition = [0, 0, 0, 2, 0, 0, 0, 0, 0, 0];
    let s: Composition = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0];
    for (slot, want) in [(0usize, o2), (1usize, s)] {
        let (counts, _, flag, source) = decode_record(&open.cand[slot * 13..slot * 13 + 13]);
        assert_eq!(counts, want, "slot {slot} holds its rank");
        assert_eq!(flag, 2, "slot {slot} is ambiguous");
        assert_eq!(source, u32::MAX);
    }
    // (M, meta cap, wrapper cap, expected scored).
    let scenarios: Vec<(usize, u32, u32, u32)> = vec![
        (4, 0, 0, 0),
        (4, 1, 1, 1),
        (4, 2, 2, 2),
        (4, 3, 3, 2),
        (4, 1, 2, 1),
        (4, 2, 1, 1),
        (2, 2, 2, 2),
    ];
    for (m, meta_cap, wrap_cap, scored) in scenarios {
        let what = format!("M={m} meta={meta_cap} wrap={wrap_cap}");
        let meta = meta_capped(&domain, &query, meta_cap);
        // Fill only (poison A): ranks below scored hold records, the rest
        // keeps the poison, proving fill wrote nothing there.
        let filled = run_pipeline_capped(
            &[meta], &rare, &packed, m, wrap_cap, wrap_cap, POISON_A, true, false,
        );
        assert_eq!(filled.counters[2], scored, "{what}: scored");
        assert_eq!(filled.counters[1], 2, "{what}: joined counts every join");
        for slot in 0..m {
            let words = &filled.cand[slot * 13..slot * 13 + 13];
            if (slot as u32) < scored {
                assert_eq!(
                    words,
                    &open.cand[slot * 13..slot * 13 + 13],
                    "{what}: fill slot {slot} is rank {slot}"
                );
            } else {
                assert_eq!(words, &[POISON_A; 13], "{what}: fill owns nothing at {slot}");
            }
        }
        // Pad only (poison B): slots below scored keep the poison, proving
        // pad wrote nothing there; slots at or after scored are padding.
        let padded = run_pipeline_capped(
            &[meta], &rare, &packed, m, wrap_cap, wrap_cap, POISON_B, false, true,
        );
        for slot in 0..m {
            let words = &padded.cand[slot * 13..slot * 13 + 13];
            if (slot as u32) < scored {
                assert_eq!(words, &[POISON_B; 13], "{what}: pad owns nothing at {slot}");
            } else {
                assert_eq!(&words[..12], &[0u32; 12], "{what}: padding at {slot}");
                assert_eq!(words[12], u32::MAX, "{what}: padding source at {slot}");
            }
        }
        // Full run: no poison survives; slots match fill below scored and
        // pad at or after it.
        let full = run_pipeline_capped(
            &[meta], &rare, &packed, m, wrap_cap, wrap_cap, POISON_A, true, true,
        );
        assert!(
            !full.cand.contains(&POISON_A),
            "{what}: no poison survives fill+pad"
        );
        assert_eq!(
            &full.cand[..],
            &{
                let mut expect = filled.cand.clone();
                let base = scored as usize * 13;
                expect[base..].copy_from_slice(&padded.cand[base..]);
                expect
            }[..],
            "{what}: full cover is fill then pad"
        );
        // The twins agree under the same (possibly mismatched) caps: the
        // fill twin reads the meta word too, so its records match.
        let mut count_stats = vec![0u32; p * 2];
        for r in 0..p {
            twin_count(&meta, &packed, &rare_flat, &chem, r, &mut count_stats);
        }
        let (offsets, _) = twin_offsets(&count_stats, &meta, p, wrap_cap);
        let eff = meta_cap.min(wrap_cap).min(m as u32);
        assert_eq!(offsets, vec![0, eff.min(1)], "{what}: clamped offsets");
        let mut twin_buf = vec![0u32; scored as usize * 13];
        for (r, off) in offsets.iter().enumerate() {
            twin_fill(
                &meta, &packed, &rare_flat, &chem, r, *off, wrap_cap, &mut twin_buf, 0,
            );
        }
        assert_eq!(
            &twin_buf[..],
            &filled.cand[..twin_buf.len()],
            "{what}: kernel fill matches the twin"
        );
    }
}

#[test]
fn per_row_caps_vary_independently() {
    // B = 2 with per-row meta caps 0 and 2 under one wrapper cap: row 0 gets
    // no fill write at all while row 1 scores normally. Fill owns ranks
    // below each row's scored count, pad the rest, per row.
    let (_, _, rare, packed, query) = o2s_setup();
    let domain = o2s_domain();
    let m = 4usize;
    let wrap = 2u32;
    let meta_rows = [meta_capped(&domain, &query, 0), meta_capped(&domain, &query, 2)];
    let filled = run_pipeline_capped(&meta_rows, &rare, &packed, m, wrap, wrap, POISON_A, true, false);
    assert_eq!(&filled.counters[0..5], &[2, 2, 0, request_status::FORMULA_SEARCH_EXHAUSTED, 0]);
    assert_eq!(filled.counters[7], 2, "row 1 scores both ranks");
    assert_eq!(
        &filled.cand[0..m * 13],
        &vec![POISON_A; m * 13][..],
        "row 0 (cap 0): fill writes nothing"
    );
    assert_ne!(
        &filled.cand[m * 13..2 * m * 13],
        &vec![POISON_A; m * 13][..],
        "row 1 (cap 2): fill writes its ranks"
    );
    let padded = run_pipeline_capped(&meta_rows, &rare, &packed, m, wrap, wrap, POISON_B, false, true);
    assert_eq!(
        &padded.cand[m * 13..m * 13 + 2 * 13],
        &vec![POISON_B; 2 * 13][..],
        "row 1: pad owns nothing below scored"
    );
    for slot in 0..m {
        let words = &padded.cand[slot * 13..slot * 13 + 13];
        assert_eq!(&words[..12], &[0u32; 12], "row 0: every slot padded");
        assert_eq!(words[12], u32::MAX);
    }
    let full = run_pipeline_capped(&meta_rows, &rare, &packed, m, wrap, wrap, POISON_A, true, true);
    assert!(
        !full.cand.contains(&POISON_A),
        "no poison survives either row"
    );
}

// ---------------------------------------------------------------------------
// EF finding 4: fill takes the capacity exits; outputs are unchanged.
// ---------------------------------------------------------------------------

#[test]
fn fill_capacity_exits_keep_outputs() {
    // On the O2/S domain with cap 1: the S lane starts at offset 1, at the
    // cap, so its fill visits are 0 (count visits are 1); the O2 lane stops
    // after its first join. Total fill visits (1) are below count visits
    // (2), while every written record is unchanged. With cap 0 no fill lane
    // visits anything and nothing is written.
    let (domain, _, rare, packed, query) = o2s_setup();
    let p = rare.len();
    let rare_flat: Vec<u32> = rare.iter().flat_map(|r| r.iter().copied()).collect();
    let chem = chem14();
    let meta = meta_capped(&domain, &query, 1);
    let mut count_stats = vec![0u32; p * 2];
    for r in 0..p {
        twin_count(&meta, &packed, &rare_flat, &chem, r, &mut count_stats);
    }
    assert_eq!(
        &count_stats,
        &[1, 1, 1, 1],
        "count mode enumerates both lanes fully"
    );
    let (offsets, counters) = twin_offsets(&count_stats, &meta, p, 1);
    assert_eq!(offsets, vec![0, 1]);
    assert_eq!(counters[2], 1);
    let mut buf0 = vec![0u32; 13];
    let v0 = twin_fill(&meta, &packed, &rare_flat, &chem, 0, offsets[0], 1, &mut buf0, 0);
    let mut buf1 = vec![0u32; 13];
    let v1 = twin_fill(&meta, &packed, &rare_flat, &chem, 1, offsets[1], 1, &mut buf1, 0);
    assert_eq!(v1, 0, "the lane starting at the cap visits nothing");
    assert!(
        v0 < count_stats[1] + count_stats[3],
        "fill stops at the first excluded rank"
    );
    assert_eq!(v0 + v1, 1);
    // Cap 0: no fill lane visits anything.
    let meta0 = meta_capped(&domain, &query, 0);
    let mut z0 = vec![0u32; 13];
    let mut z1 = vec![0u32; 13];
    assert_eq!(
        twin_fill(&meta, &packed, &rare_flat, &chem, 0, 0, 0, &mut z0, 0),
        0,
        "cap 0: no visits"
    );
    assert_eq!(
        twin_fill(&meta0, &packed, &rare_flat, &chem, 1, 0, 0, &mut z1, 0),
        0,
        "cap 0: no visits"
    );
    // Kernel outputs are unchanged: cap == joined reproduces the open
    // prefix, and the kernels match the twins above.
    let open = run_pipeline(&[meta_capped(&domain, &query, 4)], &rare, &packed, 4, 4, POISON_A, true, true);
    let capped = run_pipeline(&[meta_capped(&domain, &query, 2)], &rare, &packed, 4, 2, POISON_A, true, true);
    assert_eq!(&capped.cand[..2 * 13], &open.cand[..2 * 13]);
    assert_eq!(&buf0[..], &capped.cand[..13]);
}

// ---------------------------------------------------------------------------
// EF finding 5: the lane ceiling and the standalone offsets length.
// ---------------------------------------------------------------------------

#[test]
fn dispatch_ceiling_refused_before_launch() {
    use mamba3::models::ms2::formula_enum::validate_enum_dispatch;
    // The reviewer's dispatch (B = 17, P = 16,384: 278,528 lanes) is refused
    // by every wrapper before any launch.
    let device = dev();
    let chem = EnumChem::from_chemistry();
    let ok = |data: &[u32], shape: Vec<usize>| IdTensor::from_slice(data, shape, &device).unwrap();
    let meta = ok(&[0u32; 2 * 8], vec![2, 8]);
    let rare = ok(&[0u32; 3 * 8], vec![3, 8]);
    let bounds = ok(&[0u32; 64], vec![64]);
    let stats = ok(&[0u32; 6 * 2], vec![6, 2]);
    let offsets = ok(&[0u32; 6], vec![6]);
    let counters = ok(&[0u32; 2 * 5], vec![2, 5]);
    let cand = ok(&[0u32; 2 * 4 * 13], vec![2, 4, 13]);
    // B * P = 6 above a ceiling of 5: all four wrappers refuse.
    assert!(enum_count(&meta, &rare, &bounds, &stats, &chem, 5, u32::MAX, 4096).is_err());
    assert!(enum_offsets(&stats, &meta, &offsets, &counters, 4, 4, 5).is_err());
    assert!(enum_fill(&meta, &rare, &bounds, &offsets, &cand, &chem, 4, 5, u32::MAX, 4096).is_err());
    assert!(cand_pad(&counters, &cand, 3, 5).is_err());
    // The check function itself refuses the reviewer's case with nothing
    // allocated.
    assert!(validate_enum_dispatch(17, 16_384, 32, ENUM_LANES_MAX_DEFAULT).is_err());
    // At the ceiling the same shapes launch.
    enum_count(&meta, &rare, &bounds, &stats, &chem, 6, u32::MAX, 4096).unwrap();
    enum_offsets(&stats, &meta, &offsets, &counters, 4, 4, 6).unwrap();
    enum_fill(&meta, &rare, &bounds, &offsets, &cand, &chem, 4, 6, u32::MAX, 4096).unwrap();
    cand_pad(&counters, &cand, 3, 6).unwrap();
    check_launches(&device).unwrap();
}

#[test]
fn standalone_fill_rejects_bad_offsets_length() {
    // A standalone `enum_fill` with an offsets buffer whose length is not
    // `B * P` is `Error::Shape` before any lane can wrap an address.
    let device = dev();
    let chem = EnumChem::from_chemistry();
    let ok = |data: &[u32], shape: Vec<usize>| IdTensor::from_slice(data, shape, &device).unwrap();
    let meta = ok(&[0u32; 8], vec![1, 8]);
    let rare = ok(&[0u32; 8], vec![1, 8]);
    let bounds = ok(&[0u32; 64], vec![64]);
    let cand = ok(&[0u32; 1 * 4 * 13], vec![1, 4, 13]);
    let bad_offsets = ok(&[0u32; 2], vec![2]);
    let r = enum_fill(&meta, &rare, &bounds, &bad_offsets, &cand, &chem, 4, u32::MAX, u32::MAX, 4096);
    assert!(matches!(r, Err(Error::Shape(_))));
}

// ---------------------------------------------------------------------------
// EF finding 7 (kernels): saturation offsets, CH2/CH4, 2^20, failed rows.
// ---------------------------------------------------------------------------

#[test]
fn offsets_kernel_on_saturation_fixture() {
    // The synthetic saturation `lane_stats` launched directly through the
    // `enum_offsets` kernel (not just the twin): saturation alone exhausts
    // with `complete` cleared.
    let device = dev();
    let stats = IdTensor::from_slice(
        &[0u32, 0x7FFF_FFFF, 0, 0x7FFF_FFFF, 0, 0x7FFF_FFFF],
        vec![3, 2],
        &device,
    )
    .unwrap();
    let meta = IdTensor::from_slice(&[0, 0, 0, 0, 0, 0, u32::MAX, 0], vec![1, 8], &device)
        .unwrap();
    let offsets = IdTensor::from_slice(&[POISON_A; 3], vec![3], &device).unwrap();
    let counters = IdTensor::from_slice(&[POISON_A; 5], vec![1, 5], &device).unwrap();
    enum_offsets(&stats, &meta, &offsets, &counters, u32::MAX, 32, u32::MAX).unwrap();
    check_launches(&device).unwrap();
    assert_eq!(
        counters.try_to_vec().unwrap(),
        vec![
            u32::MAX - 1,
            0,
            0,
            request_status::FORMULA_SEARCH_EXHAUSTED,
            0
        ]
    );
    assert_eq!(offsets.try_to_vec().unwrap(), vec![0, 0, 0]);
    // Joined saturation likewise exhausts through the kernel.
    let stats = IdTensor::from_slice(
        &[0x7FFF_FFFFu32, 0, 0x7FFF_FFFF, 0, 0x7FFF_FFFF, 0],
        vec![3, 2],
        &device,
    )
    .unwrap();
    let offsets = IdTensor::from_slice(&[POISON_A; 3], vec![3], &device).unwrap();
    let counters = IdTensor::from_slice(&[POISON_A; 5], vec![1, 5], &device).unwrap();
    enum_offsets(&stats, &meta, &offsets, &counters, u32::MAX, 32, u32::MAX).unwrap();
    check_launches(&device).unwrap();
    let counters = counters.try_to_vec().unwrap();
    assert_eq!(counters[1], u32::MAX - 1, "joined saturates");
    assert_eq!(counters[3], request_status::FORMULA_SEARCH_EXHAUSTED);
    assert_eq!(counters[4], 0);
}

#[test]
fn two_hydrogens_ch2_ch4_on_kernels() {
    // The CH2/CH4 window through the real kernels: both hydrogen counts join
    // as ambiguous, every output element matching the twins.
    let ch2: Composition = [1, 2, 0, 0, 0, 0, 0, 0, 0, 0];
    let ch4: Composition = [1, 4, 0, 0, 0, 0, 0, 0, 0, 0];
    let domain = EnumDomain::from_compositions([ch2, ch4], 0).unwrap();
    let bounds = RatioBounds::fit([ch2, ch4], 0).unwrap();
    let packed = pack_device_bounds(&domain, &bounds).unwrap();
    let rare = rare_table(&domain, &bounds).unwrap();
    let p = rare.len();
    let rare_flat: Vec<u32> = rare.iter().flat_map(|r| r.iter().copied()).collect();
    let chem = chem14();
    let parent = 15_023_475u32;
    let query = EnumQuery {
        precursor_mz: parent + 1_007_825 - 549,
        adduct: 1,
        ppm_tenths: 1000,
        precursor_uncertainty: 1_007_000,
    };
    let m = 32usize;
    let cap = 32u32;
    let meta = setup_meta(&domain, &query, u32::MAX, cap).expect("launchable row");
    let out = run_pipeline(&[meta], &rare, &packed, m, cap, POISON_A, true, true);
    check_row(
        &domain, &bounds, &query, u32::MAX, cap, m, &meta, &rare_flat, &packed, &chem, p,
        &out, 0, "ch2-ch4",
    );
    for want in [ch2, ch4] {
        let mut at = None;
        for i in 0..out.counters[2] as usize {
            let (counts, _, flag, _) = decode_record(&out.cand[i * 13..i * 13 + 13]);
            if counts == want {
                at = Some((i, flag));
            }
        }
        let (i, flag) = at.unwrap_or_else(|| panic!("{want:?} joins"));
        assert_eq!(flag, 2, "slot {i} is ambiguous");
    }
}

#[test]
fn ratio_two_to_twenty_on_kernels() {
    // Ratio factors exactly 2^20 through the kernels: the 2^20 products
    // (<= 1,072,693,248) stay exact, the gold joins, every element matches
    // the twins.
    let ch4: Composition = [1, 4, 0, 0, 0, 0, 0, 0, 0, 0];
    let domain = EnumDomain::from_compositions([ch4], 0).unwrap();
    let mut bounds = RatioBounds::fit([ch4], 0).unwrap();
    for k in 0..6 {
        bounds.ratio_lo_num[k] = 0;
        bounds.ratio_lo_den[k] = 1 << 20;
        bounds.ratio_hi_num[k] = 1 << 20;
        bounds.ratio_hi_den[k] = 1;
    }
    let packed = pack_device_bounds(&domain, &bounds).unwrap();
    let rare = rare_table(&domain, &bounds).unwrap();
    let p = rare.len();
    let rare_flat: Vec<u32> = rare.iter().flat_map(|r| r.iter().copied()).collect();
    let chem = chem14();
    let query = EnumQuery {
        precursor_mz: protonated(brute_mass(&ch4)),
        adduct: 1,
        ppm_tenths: 200,
        precursor_uncertainty: 50,
    };
    let m = 32usize;
    let cap = 32u32;
    let meta = setup_meta(&domain, &query, u32::MAX, cap).expect("launchable row");
    let out = run_pipeline(&[meta], &rare, &packed, m, cap, POISON_A, true, true);
    check_row(
        &domain, &bounds, &query, u32::MAX, cap, m, &meta, &rare_flat, &packed, &chem, p,
        &out, 0, "ratio-2^20",
    );
    let scored = out.counters[2] as usize;
    let mut gold = false;
    for i in 0..scored {
        let (counts, _, _, _) = decode_record(&out.cand[i * 13..i * 13 + 13]);
        if counts == ch4 {
            gold = true;
        }
    }
    assert!(gold, "gold joins with 2^20 ratio factors");
}

#[test]
fn failed_row_through_build_enum_meta() {
    // A real failed-request row (`peak_count == 0`, fatal host status)
    // through the actual metadata preparation used by generation
    // (`build_enum_meta`), on the kernels. The enum lanes still run for the
    // failed row (failure is enforced downstream by `peak_count == 0`, which
    // keeps `init_trajectories` from starting it); what the kernels must
    // guarantee is batch independence and single-writer coverage.
    // C2H5NO (59.0 Da, precursor 60.0 Da: inside the 50–2000 Da request
    // domain, unlike the lighter tiny-domain carriers) with a subset that
    // admits it.
    let good: Composition = [2, 5, 1, 1, 0, 0, 0, 0, 0, 0];
    let domain = tiny_domain();
    let subset: Vec<Composition> = vec![
        good,
        [1, 4, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 2, 1, 1, 0, 0, 0, 0, 0, 0],
        [2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ];
    let bounds = RatioBounds::fit(subset.iter().copied(), 0).unwrap();
    let packed = pack_device_bounds(&domain, &bounds).unwrap();
    let rare = rare_table(&domain, &bounds).unwrap();
    let p = rare.len();
    let n_raw = 64usize;
    let precursor = protonated(brute_mass(&good));
    let mut peak_id = vec![u32::MAX; 2 * n_raw];
    let mut mz = vec![0u32; 2 * n_raw];
    let mut intensity = vec![0.0f32; 2 * n_raw];
    for i in 0..10 {
        peak_id[i] = i as u32;
        mz[i] = 60_000_000 + i as u32;
        intensity[i] = 1.0;
    }
    let batch = SpectrumBatch {
        schema_version: SPECTRUM_SCHEMA_VERSION,
        n_raw: n_raw as u32,
        spectrum_id: vec![901, 902],
        raw_peak_count: vec![10, 0],
        peak_count: vec![10, 0],
        peak_id,
        mz_udalton: mz,
        intensity,
        intensity_scale: 0,
        mz_uncertainty_udalton: vec![50, 50],
        precursor_mz_udalton: vec![precursor, precursor],
        precursor_uncertainty_udalton: vec![500, 500],
        adduct: vec![1, 1],
        polarity: vec![1, 1],
        collision_energy_ev: vec![30.0, 30.0],
        collision_energy_known: vec![1, 1],
        energy_count: vec![1, 1],
        fragment_tolerance_ppm_tenths: vec![0, 0],
        precursor_tolerance_ppm_tenths: vec![0, 0],
        instrument_class: vec![0, 0],
    };
    let statuses = batch.validate().unwrap();
    assert_eq!(statuses[0], 0);
    assert!(
        statuses[1] & request_status::FATAL_MASK != 0,
        "peak_count 0 is a fatal host status"
    );
    let scored_cap = 32u32;
    let meta_host = build_enum_meta(&batch, domain.max_error(), u32::MAX, scored_cap);
    assert_eq!(meta_host.len(), 2 * 8);
    let meta_rows: Vec<[u32; 8]> = vec![
        meta_host[0..8].try_into().unwrap(),
        meta_host[8..16].try_into().unwrap(),
    ];
    let m = 32usize;
    let joint = run_pipeline(&meta_rows, &rare, &packed, m, scored_cap, POISON_A, true, true);
    // Batch independence: each row equals its single-row run.
    for (b, meta) in meta_rows.iter().enumerate() {
        let single = run_pipeline(&[*meta], &rare, &packed, m, scored_cap, POISON_A, true, true);
        assert_eq!(
            &joint.lane_stats[b * p * 2..(b + 1) * p * 2],
            &single.lane_stats[..],
            "row {b}: lane_stats batch independence"
        );
        assert_eq!(
            &joint.cand[b * m * 13..(b + 1) * m * 13],
            &single.cand[..],
            "row {b}: cand batch independence"
        );
    }
    // Single-writer cover on both rows: no poison survives fill+pad.
    assert!(
        !joint.cand.contains(&POISON_A),
        "no poison survives on either row"
    );
    // The failed row searched like its identical good twin (same query, same
    // lanes): the enum stage does not special-case it.
    assert_eq!(
        &joint.cand[0..m * 13],
        &joint.cand[m * 13..2 * m * 13],
        "identical queries join identically, failed or not"
    );
}

#[test]
fn bounded_dispatch_chunked_equals_unchunked() {
    // D10: one count/fill launch covers at most lanes_per_dispatch lanes;
    // chunked launches over contiguous ranges equal one unchunked launch
    // element-wise, for several chunk sizes including 1 lane per dispatch
    // and a chunk larger than B*P. The number of launches equals
    // ceil(B*P / lanes_per_dispatch), a function of bucket shape and config.
    use mamba3::models::ms2::formula_enum::{
        ENUM_LANES_MAX_DEFAULT, enum_dispatches, enum_lanes_per_dispatch,
    };
    // Pure helper checks (launch count formula, defaults give 976 lanes/launch).
    assert_eq!(enum_lanes_per_dispatch(4_000_000, 4_096), 976);
    assert_eq!(enum_lanes_per_dispatch(4_000_000, 65_536), 61);
    assert_eq!(enum_lanes_per_dispatch(100, 4096), 1);
    assert_eq!(enum_lanes_per_dispatch(0, 4096), 1);
    assert_eq!(enum_dispatches(0, 976), 0);
    assert_eq!(enum_dispatches(10, 976), 1);
    assert_eq!(enum_dispatches(976, 976), 1);
    assert_eq!(enum_dispatches(977, 976), 2);
    assert_eq!(enum_dispatches(10, 1), 10);
    assert_eq!(enum_dispatches(10, 1000), 1);
    // Wrapper equivalence on a tiny valid setup: B=1, P=3, M=8, empty search
    // (meta lo>hi, zero bounds) still exercises chunking through count/fill.
    let meta1: [[u32; 8]; 1] = [[0, 0, 0, 1, 0, 4096, 8, 0]];
    let rare1: [[u32; 8]; 3] = [[0, 0, 0, 0, 0, 0, 0, 0]; 3];
    let packed1 = vec![0u32; 64];
    let unchunked = run_pipeline_capped(&meta1, &rare1, &packed1, 8, 8, 8, 0xDEAD, true, true);
    for (dispatch, lane_visits) in [(1u32, 4096u32), (4_000_000u32, 4_096u32), (100u32, 1u32)] {
        let per = enum_lanes_per_dispatch(dispatch, lane_visits);
        let n = enum_dispatches(3, per);
        assert_eq!(n, (3 + per - 1) / per.max(1));
        assert_eq!(n, enum_dispatches(3, per), "launches equal the formula");
        let device = dev();
        let chem = EnumChem::from_chemistry();
        let meta_flat: Vec<u32> = meta1.iter().flat_map(|r| r.iter().copied()).collect();
        let rare_flat: Vec<u32> = rare1.iter().flat_map(|r| r.iter().copied()).collect();
        let meta_t = IdTensor::from_slice(&meta_flat, vec![1, 8], &device).unwrap();
        let rare_t = IdTensor::from_slice(&rare_flat, vec![3, 8], &device).unwrap();
        let bounds_t = IdTensor::from_slice(&packed1, vec![packed1.len()], &device).unwrap();
        let stats_t = IdTensor::from_slice(&vec![0xDEAD; 1 * 3 * 2], vec![3, 2], &device).unwrap();
        enum_count(&meta_t, &rare_t, &bounds_t, &stats_t, &chem, ENUM_LANES_MAX_DEFAULT, dispatch, lane_visits).unwrap();
        check_launches(&device).unwrap();
        let offsets_t = IdTensor::from_slice(&vec![0xDEAD; 3], vec![3], &device).unwrap();
        let counters_t = IdTensor::from_slice(&vec![0xDEAD; 5], vec![1, 5], &device).unwrap();
        enum_offsets(&stats_t, &meta_t, &offsets_t, &counters_t, 8, 8, ENUM_LANES_MAX_DEFAULT).unwrap();
        check_launches(&device).unwrap();
        let cand_t = IdTensor::from_slice(&vec![0xDEAD; 1 * 8 * 13], vec![1, 8, 13], &device).unwrap();
        enum_fill(&meta_t, &rare_t, &bounds_t, &offsets_t, &cand_t, &chem, 8, ENUM_LANES_MAX_DEFAULT, dispatch, lane_visits).unwrap();
        check_launches(&device).unwrap();
        cand_pad(&counters_t, &cand_t, 3, ENUM_LANES_MAX_DEFAULT).unwrap();
        check_launches(&device).unwrap();
        let chunked_stats = stats_t.try_to_vec().unwrap();
        let chunked_offsets = offsets_t.try_to_vec().unwrap();
        let chunked_counters = counters_t.try_to_vec().unwrap();
        let chunked_cand = cand_t.try_to_vec().unwrap();
        assert_eq!(chunked_stats, unchunked.lane_stats, "lane_stats chunked==unchunked (dispatch {dispatch})");
        assert_eq!(chunked_offsets, unchunked.offsets, "offsets chunked==unchunked");
        assert_eq!(chunked_counters, unchunked.counters, "counters chunked==unchunked");
        assert_eq!(chunked_cand, unchunked.cand, "cand chunked==unchunked");
    }
}

#[test]
fn bounded_dispatch_flush_per_chunk_keeps_results_bit_identical() {
    // EF2: every count/fill dispatch chunk is flushed (submitted) as its own
    // GPU job with `check_launches` — one flush per launch — which bounds
    // the work of one GPU job (a precautionary work bound; no claim that
    // batching was measured to cause any reset). The flushed chunked run
    // still equals the unchunked twin on every element, at several chunk
    // sizes, through both the raw wrappers and `EnumLaunch::count`/`fill`.
    //
    // Counter note: this asserts result equality only, not
    // `launch_count`/`read_count`. The flush path (`check_launches` is
    // `client.flush()`, `src/backend.rs`) increments none of the crate's
    // counters, so launch/read counts are unchanged by construction; this
    // binary does not read process-global counters because its tests share
    // one process without a common serialisation lock.
    use mamba3::models::ms2::formula_enum::{
        ENUM_LANES_MAX_DEFAULT, enum_dispatches, enum_lanes_per_dispatch,
    };
    let meta1: [[u32; 8]; 1] = [[0, 0, 0, 1, 0, 4096, 8, 0]];
    let rare1: [[u32; 8]; 3] = [[0, 0, 0, 0, 0, 0, 0, 0]; 3];
    let packed1 = vec![0u32; 64];
    let unchunked = run_pipeline_capped(&meta1, &rare1, &packed1, 8, 8, 8, 0xDEAD, true, true);
    for (dispatch, lane_visits) in [(1u32, 4096u32), (8192u32, 4096u32), (4_000_000u32, 4_096u32), (100u32, 1u32)] {
        let per = enum_lanes_per_dispatch(dispatch, lane_visits);
        let n = enum_dispatches(3, per);
        assert_eq!(n, (3 + per - 1) / per.max(1));
        let device = dev();
        let launch = EnumLaunch::from_chemistry();
        let meta_flat: Vec<u32> = meta1.iter().flat_map(|r| r.iter().copied()).collect();
        let rare_flat: Vec<u32> = rare1.iter().flat_map(|r| r.iter().copied()).collect();
        let meta_t = IdTensor::from_slice(&meta_flat, vec![1, 8], &device).unwrap();
        let rare_t = IdTensor::from_slice(&rare_flat, vec![3, 8], &device).unwrap();
        let bounds_t = IdTensor::from_slice(&packed1, vec![packed1.len()], &device).unwrap();
        let stats_t = IdTensor::from_slice(&vec![0xDEAD; 1 * 3 * 2], vec![3, 2], &device).unwrap();
        launch
            .count(&meta_t, &rare_t, &bounds_t, &stats_t, ENUM_LANES_MAX_DEFAULT, dispatch, lane_visits)
            .unwrap();
        check_launches(&device).unwrap();
        let offsets_t = IdTensor::from_slice(&vec![0xDEAD; 3], vec![3], &device).unwrap();
        let counters_t = IdTensor::from_slice(&vec![0xDEAD; 5], vec![1, 5], &device).unwrap();
        enum_offsets(&stats_t, &meta_t, &offsets_t, &counters_t, 8, 8, ENUM_LANES_MAX_DEFAULT).unwrap();
        check_launches(&device).unwrap();
        let cand_t = IdTensor::from_slice(&vec![0xDEAD; 1 * 8 * 13], vec![1, 8, 13], &device).unwrap();
        launch
            .fill(&meta_t, &rare_t, &bounds_t, &offsets_t, &cand_t, 8, ENUM_LANES_MAX_DEFAULT, dispatch, lane_visits)
            .unwrap();
        check_launches(&device).unwrap();
        cand_pad(&counters_t, &cand_t, 3, ENUM_LANES_MAX_DEFAULT).unwrap();
        check_launches(&device).unwrap();
        assert_eq!(stats_t.try_to_vec().unwrap(), unchunked.lane_stats, "lane_stats flushed-chunked==unchunked (dispatch {dispatch})");
        assert_eq!(offsets_t.try_to_vec().unwrap(), unchunked.offsets, "offsets flushed-chunked==unchunked");
        assert_eq!(counters_t.try_to_vec().unwrap(), unchunked.counters, "counters flushed-chunked==unchunked");
        assert_eq!(cand_t.try_to_vec().unwrap(), unchunked.cand, "cand flushed-chunked==unchunked");
    }
}

/// Chunked run of one `(B, P, M)` bucket with explicit dispatch budgets, for
/// the productive chunk-equivalence test below.
#[allow(clippy::too_many_arguments)]
fn run_pipeline_dispatch(
    meta_rows: &[[u32; 8]],
    rare: &[[u32; 8]],
    packed: &[u32],
    m: usize,
    cap: u32,
    dispatch_visits: u32,
    lane_visits: u32,
    poison: u32,
) -> PipeOut {
    use mamba3::models::ms2::formula_enum::ENUM_LANES_MAX_DEFAULT;
    let device = dev();
    let b = meta_rows.len();
    let p = rare.len();
    let chem = EnumChem::from_chemistry();
    let launch = EnumLaunch::from_chemistry();
    let meta_flat: Vec<u32> = meta_rows.iter().flat_map(|r| r.iter().copied()).collect();
    let rare_flat: Vec<u32> = rare.iter().flat_map(|r| r.iter().copied()).collect();
    let meta_t = IdTensor::from_slice(&meta_flat, vec![b, 8], &device).unwrap();
    let rare_t = IdTensor::from_slice(&rare_flat, vec![p, 8], &device).unwrap();
    let bounds_t = IdTensor::from_slice(packed, vec![packed.len()], &device).unwrap();
    let stats_t =
        IdTensor::from_slice(&vec![poison; b * p * 2], vec![b * p, 2], &device).unwrap();
    launch
        .count(
            &meta_t,
            &rare_t,
            &bounds_t,
            &stats_t,
            ENUM_LANES_MAX_DEFAULT,
            dispatch_visits,
            lane_visits,
        )
        .unwrap();
    check_launches(&device).unwrap();
    let offsets_t = IdTensor::from_slice(&vec![poison; b * p], vec![b * p], &device).unwrap();
    let counters_t =
        IdTensor::from_slice(&vec![poison; b * 5], vec![b, 5], &device).unwrap();
    enum_offsets(&stats_t, &meta_t, &offsets_t, &counters_t, cap, m, ENUM_LANES_MAX_DEFAULT)
        .unwrap();
    check_launches(&device).unwrap();
    let cand_t =
        IdTensor::from_slice(&vec![poison; b * m * 13], vec![b, m, 13], &device).unwrap();
    launch
        .fill(
            &meta_t,
            &rare_t,
            &bounds_t,
            &offsets_t,
            &cand_t,
            cap,
            ENUM_LANES_MAX_DEFAULT,
            dispatch_visits,
            lane_visits,
        )
        .unwrap();
    check_launches(&device).unwrap();
    cand_pad(&counters_t, &cand_t, p, ENUM_LANES_MAX_DEFAULT).unwrap();
    check_launches(&device).unwrap();
    PipeOut {
        lane_stats: stats_t.try_to_vec().unwrap(),
        offsets: offsets_t.try_to_vec().unwrap(),
        counters: counters_t.try_to_vec().unwrap(),
        cand: cand_t.try_to_vec().unwrap(),
    }
}

/// Query joining only the S lane of the O2/S domain: the S precursor with a
/// narrow (50-unit) uncertainty, so O2 (about 990,000 units away) is far
/// outside the window.
fn s_only_query() -> EnumQuery {
    let s: Composition = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0];
    EnumQuery {
        precursor_mz: protonated(brute_mass(&s)),
        adduct: 1,
        ppm_tenths: 1000,
        precursor_uncertainty: 50,
    }
}

#[test]
fn bounded_dispatch_chunked_equals_unchunked_productive() {
    // Chunk equivalence with PRODUCTIVE lanes (finding R1-B2): two spectra
    // over the O2/S domain, the first joining both rare lanes (O2 and S),
    // the second joining only S — distinct rare rows with candidates written
    // by fill. Chunk sizes 1, 2, one that does not divide B*P = 4 (3, whose
    // first chunk crosses the spectrum boundary) are compared element by
    // element against the unchunked run on poisoned outputs. A regression
    // using the local lane index in fill alone fails here (fill writes real
    // records whose absolute indices differ per chunk).
    let (domain, _bounds, rare, packed, query_both) = o2s_setup();
    let query_s = s_only_query();
    let meta0 = setup_meta(&domain, &query_both, 4096, 4).expect("launchable row");
    let meta1 = setup_meta(&domain, &query_s, 4096, 4).expect("launchable row");
    let metas = [meta0, meta1];
    let unchunked = run_pipeline_capped(&metas, &rare, &packed, 4, 4, 4, 0xDEAD, true, true);
    // Both spectra are productive and distinct: spectrum 0 joins both lanes,
    // spectrum 1 joins only S; fill wrote real records (no poison survives
    // in the written prefix).
    assert_eq!(unchunked.counters[1], 2, "spectrum 0 joins O2 and S");
    assert_eq!(unchunked.counters[6], 1, "spectrum 1 joins S only");
    assert_eq!(unchunked.counters[2], 2, "spectrum 0 scores 2");
    assert_eq!(unchunked.counters[7], 1, "spectrum 1 scores 1");
    assert!(
        !unchunked.cand[..4 * 13].iter().all(|&w| w == 0xDEAD),
        "fill wrote candidates for spectrum 0"
    );
    // (dispatch_visits, lane_visits, lanes_per_dispatch): 1, 2, and 3 (which
    // neither divides B*P = 4 nor respects the spectrum boundary).
    for (dispatch, lane_visits, per) in
        [(1u32, 4096u32, 1usize), (8192u32, 4096u32, 2usize), (12288u32, 4096u32, 3usize)]
    {
        assert_eq!(
            mamba3::models::ms2::formula_enum::enum_lanes_per_dispatch(dispatch, lane_visits),
            per,
            "lanes per dispatch"
        );
        let chunked =
            run_pipeline_dispatch(&metas, &rare, &packed, 4, 4, dispatch, lane_visits, 0xDEAD);
        assert_eq!(
            chunked.lane_stats, unchunked.lane_stats,
            "lane_stats chunked==unchunked (per {per})"
        );
        assert_eq!(
            chunked.offsets, unchunked.offsets,
            "offsets chunked==unchunked (per {per})"
        );
        assert_eq!(
            chunked.counters, unchunked.counters,
            "counters chunked==unchunked (per {per})"
        );
        assert_eq!(chunked.cand, unchunked.cand, "cand chunked==unchunked (per {per})");
    }
}

/// Rich-visit fixture for the lane-bound test: bounds fitted from C2H6 and
/// N2H2 (the reviewer's pair) with a 50,000-unit uncertainty around C2H6, so
/// lanes admit many heavy-vector visits.
fn rich_visit_setup() -> (EnumDomain, Vec<[u32; 8]>, Vec<u32>, EnumQuery) {
    use mamba3::models::ms2::formula_enum::EnumDomain;
    let c2h6: Composition = [2, 6, 0, 0, 0, 0, 0, 0, 0, 0];
    let n2h2: Composition = [0, 2, 2, 0, 0, 0, 0, 0, 0, 0];
    let comps = vec![c2h6, n2h2];
    let domain = EnumDomain::from_compositions(comps.iter().copied(), 0).unwrap();
    let bounds =
        mamba3::models::ms2::formula_enum::RatioBounds::fit(comps.iter().copied(), 0).unwrap();
    let packed = pack_device_bounds(&domain, &bounds).unwrap();
    let rare = rare_table(&domain, &bounds).unwrap();
    assert!(!rare.is_empty(), "the fixture domain has resident lanes");
    let query = EnumQuery {
        precursor_mz: protonated(brute_mass(&c2h6)),
        adduct: 1,
        ppm_tenths: 1000,
        precursor_uncertainty: 50_000,
    };
    (domain, rare, packed, query)
}

#[test]
fn lane_visit_bound_enforced_in_count_and_fill() {
    // The lane enforces `min(metadata budget, lane_visits_max)` (finding
    // R2-C1): metadata budget 4096 with wrapper bound 1 gives exactly one
    // visit per lane with the exhausted bit set, and count and fill agree.
    // Without the enforcement the wrapper bound only sizes chunks, so lanes
    // perform their many natural visits in the "one-visit" dispatch.
    let (domain, rare, packed, query) = rich_visit_setup();
    let p = rare.len();
    let meta = setup_meta(&domain, &query, 4096, 8).expect("launchable row");
    let metas = [meta];
    // Unclamped reference: some lane naturally visits more than once (else
    // the fixture cannot tell enforcement from nature).
    let open = run_pipeline_capped(&metas, &rare, &packed, 8, 8, 8, 0xDEAD, true, true);
    let max_visits = open
        .lane_stats
        .iter()
        .skip(1)
        .step_by(2)
        .map(|w| w & LANE_VISITED_MASK)
        .max()
        .unwrap();
    assert!(
        max_visits > 1,
        "the fixture admits multi-visit lanes (max {max_visits})"
    );
    // Clamped run: metadata budget 4096, wrapper bound 1, one-visit dispatch.
    let clamped = run_pipeline_dispatch(&metas, &rare, &packed, 8, 8, 1, 1, 0xDEAD);
    for r in 0..p {
        let word = clamped.lane_stats[r * 2 + 1];
        let natural = open.lane_stats[r * 2 + 1] & LANE_VISITED_MASK;
        if natural == 0 {
            continue;
        }
        assert_eq!(
            word & LANE_VISITED_MASK,
            1,
            "lane {r}: exactly one visit under bound 1 (natural {natural})"
        );
        assert!(
            word & LANE_EXHAUSTED_BIT != 0,
            "lane {r}: exhausted bit set under bound 1"
        );
    }
    // The same bound as metadata (budget 1, wrapper 4096) gives the identical
    // lane statistics: the wrapper bound is enforced exactly like metadata.
    let mut meta_one = meta;
    meta_one[5] = 1;
    let via_meta = run_pipeline_capped(&[meta_one], &rare, &packed, 8, 8, 8, 0xDEAD, true, true);
    assert_eq!(
        clamped.lane_stats, via_meta.lane_stats,
        "wrapper bound 1 == metadata budget 1"
    );
    assert_eq!(clamped.cand, via_meta.cand, "fill agrees under either bound");
    // Count and fill agree record for record under the bound: the count twin
    // on the budget-1 metadata reproduces the device count output (the twin
    // enforces `min(meta budget, lane budget)` with the meta word, exactly
    // like the kernel).
    let chem = chem14();
    let rare_flat: Vec<u32> = rare.iter().flat_map(|row| row.iter().copied()).collect();
    let mut twin_stats = vec![0u32; p * 2];
    for r in 0..p {
        twin_count(&meta_one, &packed, &rare_flat, &chem, r, &mut twin_stats);
    }
    assert_eq!(
        twin_stats, clamped.lane_stats,
        "count twin with lane budget 1 matches the device"
    );
}
