//! Dispatch-size bench for the enumerating formula source (task T6 §6).
//!
//! For a batch of `B` spectra placed (a) at real precursors from
//! `--data <export>` and (b) at an adversarial worst case (precursor near
//! the top of the artifacts' mass domain, widest tolerance the contract
//! allows — 1000 tenths of a ppm — so that as many lanes as possible reach
//! their visit budget), times count and fill (`N` back-to-back calls with
//! one drain, pipelined average) for `enum_dispatch_visits_max` in
//! {4e6, 1.6e7, 6.4e7, 2.56e8, 2^29}, printing for each the launches, the
//! wall time per call, the total visits (from `counters`) and the number of
//! exhausted lanes.
//!
//! This changes no default: it prints the table and states in the summary
//! what the worst-case single-launch time is on the cpu runtime (the
//! supervisor measures the GPU and decides).
//!
//! Usage:
//! ```text
//! cargo run --release --no-default-features --features cpu --example bench_ms2_enum -- \
//!   --data <export.json> [--b 8] [--n 5] [--m 32]
//! ```

use std::path::PathBuf;
use std::time::Instant;

use mamba3::backend::Device;
use mamba3::backends::Auto;
use mamba3::models::ms2::chem::composition_mass;
use mamba3::models::ms2::experiment::{ExperimentSet, spectrum_batch_for};
use mamba3::models::ms2::formula_enum::{
    DEVICE_HALF_MAX, EnumDomain, RatioBounds, build_enum_meta, enum_lanes_per_dispatch,
};
use mamba3::models::ms2::formula_head::DeviceEnumArtifacts;
use mamba3::models::ms2::targets::RecipeLimits;
use mamba3::tensor::ops::index::IdTensor;
use mamba3::tensor::ops::ms2_enum::{EnumLaunch, cand_pad, enum_offsets};

type R = Auto;

fn usage() -> ! {
    eprintln!(
        "usage: bench_ms2_enum --data <export.json> [--b 8] [--n 5] [--m 32]"
    );
    std::process::exit(2);
}

fn fail(msg: String) -> ! {
    eprintln!("bench_ms2_enum: {msg}");
    std::process::exit(1);
}

/// Raw peak capacity bucket covering these spectra.
fn bucket_n_raw(set: &ExperimentSet, indices: &[usize]) -> u32 {
    let mut longest = 0usize;
    for &i in indices {
        longest = longest.max(set.spectra[i].spectrum.peak_id.len());
    }
    for bucket in [64u32, 128, 256, 512] {
        if (longest as u64) <= u64::from(bucket) {
            return bucket;
        }
    }
    fail(format!(
        "longest peak list {longest} exceeds the 512-peak bucket"
    ));
}

fn main() {
    let mut data: Option<PathBuf> = None;
    let mut b = 8usize;
    let mut n = 5usize;
    let mut m = 32usize;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut next = || args.next().unwrap_or_else(|| usage());
        match arg.as_str() {
            "--data" => data = Some(PathBuf::from(next())),
            "--b" => b = next().parse().unwrap_or_else(|_| usage()),
            "--n" => n = next().parse().unwrap_or_else(|_| usage()),
            "--m" => m = next().parse().unwrap_or_else(|_| usage()),
            _ => usage(),
        }
    }
    let Some(data_path) = data else { usage() };
    if b == 0 || n == 0 || !matches!(m, 32 | 128 | 512 | 2048) {
        usage();
    }
    let lane_visits_max = 4_096u32;
    let lanes_max = 262_144u32;
    let scored_cap = 4096u32.min(m as u32);

    let device = Device::<R>::default();
    let set = ExperimentSet::load(&data_path, &RecipeLimits::V0)
        .unwrap_or_else(|e| fail(format!("cannot load {}: {e}", data_path.display())));
    let mut labeled = set.labeled();
    if labeled.is_empty() {
        fail("export holds no labeled spectrum".to_string());
    }
    labeled.truncate(b);
    let b = labeled.len();
    let mut comps = Vec::with_capacity(set.spectra.len());
    for s in &set.spectra {
        comps.push(s.parent_composition);
    }
    let domain = EnumDomain::from_compositions(comps.clone(), 0)
        .unwrap_or_else(|e| fail(format!("cannot fit EnumDomain: {e}")));
    let bounds = RatioBounds::fit(comps.clone(), 0)
        .unwrap_or_else(|e| fail(format!("cannot fit RatioBounds: {e}")));
    let artifacts = DeviceEnumArtifacts::upload(&domain, &bounds, &device)
        .unwrap_or_else(|e| fail(format!("cannot upload artifacts: {e}")));
    let p = artifacts.p;
    println!(
        "bench_ms2_enum: B={b} P={p} M={m} lane_visits_max={lane_visits_max} domain_max_error={}",
        artifacts.domain_max_error
    );

    // (a) real precursors from the export.
    let n_raw = bucket_n_raw(&set, &labeled);
    let real_batch = spectrum_batch_for(&set, &labeled, n_raw)
        .unwrap_or_else(|e| fail(format!("cannot build real batch: {e}")));
    // (b) adversarial worst case: precursor near the top of the artifacts'
    // mass domain (heaviest fitted composition, protonated) with the widest
    // tolerance the contract allows (1000 tenths of a ppm), so that as many
    // lanes as possible reach their visit budget. The uncertainty stays
    // small so the window stays inside `DEVICE_HALF_MAX` (past it the lane
    // budget is 0 and lanes do no work at all — the trivial case, not the
    // worst one).
    let top_mass = comps
        .iter()
        .map(|c| composition_mass(c).unwrap_or(0))
        .max()
        .unwrap_or(0);
    let mut worst_batch = real_batch.clone();
    worst_batch.precursor_mz_udalton = vec![top_mass + 1_007_825 - 549; b];
    worst_batch.precursor_uncertainty_udalton = vec![50; b];
    worst_batch.adduct = vec![1; b];
    worst_batch.precursor_tolerance_ppm_tenths = vec![1000; b];

    let cases: [(&str, mamba3::models::ms2::contract::SpectrumBatch); 2] =
        [("real", real_batch), ("worst", worst_batch)];
    let dispatches: [u32; 5] =
        [4_000_000, 16_000_000, 64_000_000, 256_000_000, 536_870_912];
    println!(
        "{:<6} {:>11} {:>7} {:>12} {:>12} {:>12} {:>12} {:>10}",
        "case", "dispatch", "launches", "count_ms", "fill_ms", "visits", "joined", "exhausted"
    );
    // Worst-case single-launch wall time on this runtime (the supervisor
    // measures the GPU and decides the default from it).
    let mut worst_single_ms = 0.0f64;
    for (name, batch) in &cases {
        let meta_host = build_enum_meta(
            batch,
            artifacts.domain_max_error,
            lane_visits_max,
            scored_cap,
        );
        let half_ok = meta_host
            .chunks_exact(8)
            .all(|r| r[4].saturating_sub(r[3]) <= DEVICE_HALF_MAX || r[5] == 0);
        let _ = half_ok;
        let meta_t = IdTensor::from_slice(&meta_host, vec![b, 8], &device)
            .unwrap_or_else(|e| fail(format!("meta upload: {e}")));
        let launch = EnumLaunch::from_chemistry();
        for &dispatch in &dispatches {
            let per = enum_lanes_per_dispatch(dispatch, lane_visits_max);
            let launches = (b * p).div_ceil(per);
            // Count: N back-to-back calls with one drain (pipelined
            // average). The stats buffer is rewritten by every call.
            let stats_t = IdTensor::empty(vec![b * p, 2], &device);
            device.synchronize();
            let t0 = Instant::now();
            for _ in 0..n {
                launch
                    .count(
                        &meta_t,
                        &artifacts.rare,
                        &artifacts.bounds,
                        &stats_t,
                        lanes_max,
                        dispatch,
                        lane_visits_max,
                    )
                    .unwrap_or_else(|e| fail(format!("count: {e}")));
            }
            device.synchronize();
            let count_ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
            // Offsets once (not timed), then fill N back-to-back calls
            // with one drain.
            let offsets_t = IdTensor::empty(vec![b * p], &device);
            let counters_t = IdTensor::empty(vec![b, 5], &device);
            let cand_t = IdTensor::empty(vec![b, m, 13], &device);
            enum_offsets(
                &stats_t,
                &meta_t,
                &offsets_t,
                &counters_t,
                scored_cap,
                m,
                lanes_max,
            )
            .unwrap_or_else(|e| fail(format!("offsets: {e}")));
            device.synchronize();
            let t0 = Instant::now();
            for _ in 0..n {
                launch
                    .fill(
                        &meta_t,
                        &artifacts.rare,
                        &artifacts.bounds,
                        &offsets_t,
                        &cand_t,
                        scored_cap,
                        lanes_max,
                        dispatch,
                        lane_visits_max,
                    )
                    .unwrap_or_else(|e| fail(format!("fill: {e}")));
            }
            device.synchronize();
            let fill_ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
            cand_pad(&counters_t, &cand_t, p, lanes_max)
                .unwrap_or_else(|e| fail(format!("pad: {e}")));
            // Totals from `counters` (visited, joined) and the exhausted
            // lane count from `lane_stats` (top bit of the visited word).
            let counters = counters_t
                .try_to_vec()
                .unwrap_or_else(|e| fail(format!("counters read: {e}")));
            let stats = stats_t
                .try_to_vec()
                .unwrap_or_else(|e| fail(format!("lane_stats read: {e}")));
            let mut visits = 0u64;
            let mut joined = 0u64;
            for i in 0..b {
                visits += u64::from(counters[i * 5]);
                joined += u64::from(counters[i * 5 + 1]);
            }
            let exhausted = stats
                .chunks_exact(2)
                .filter(|w| w[1] & (1 << 31) != 0)
                .count();
            println!(
                "{:<6} {:>11} {:>7} {:>12.3} {:>12.3} {:>12} {:>12} {:>10}",
                name, dispatch, launches, count_ms, fill_ms, visits, joined, exhausted
            );
            if launches == 1 {
                worst_single_ms = worst_single_ms.max(count_ms + fill_ms);
            }
        }
    }
    println!(
        "summary: worst-case single-launch count+fill time on {} is {worst_single_ms:.3} ms/call (pipelined average over {n}; the default enum_dispatch_visits_max is unchanged)",
        std::any::type_name::<R>()
    );
}
