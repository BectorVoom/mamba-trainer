//! Dispatch-size bench for the formula-evidence stage (task T6B §6).
//!
//! For a batch of `B` spectra with `M` enumerated candidates each, placed
//! (a) at real precursors with real peaks from `--data <export>` and (b) at
//! an adversarial worst case (hydrogen-rich, heteroatom-rich candidates near
//! the top of the artifacts' mass domain; 32 eligible peaks; the widest
//! fragment tolerance the contract allows — 1000 tenths of a ppm), times
//! `formula_evidence` (`N` back-to-back calls with one drain, pipelined
//! average) for each `--dispatch` bound (default `{2^24, 2^26, 2^28, 2^30,
//! 2^32, u64::MAX}`), printing for each the launches, the wall time per call
//! and the total hydrogen trials (the hidden trial-counting twin on the host
//! for the same inputs when `--host-count` asks for it, otherwise the
//! per-lane bound).
//!
//! GPU usability (task T7 part 1): the host trial counter used to run before
//! the first timed line, so on large shapes the tool sat on one CPU core
//! with no GPU activity and never printed. Now `--host-count off` (the
//! default) never runs it, the header and each case banner print (and flush)
//! before any long computation, `--case`/`--dispatch` select a subset, and a
//! watchdog thread prints `TIMEOUT ...` and exits with code 3 instead of
//! hanging silently when one setting exceeds `--timeout-seconds`.
//!
//! Each setting's line is flushed as soon as it is measured, so a hang at
//! one setting leaves the earlier lines in the log.
//!
//! This changes no default: it prints the table and states in the summary
//! what the worst-case single-launch time is on the cpu runtime (the
//! supervisor measures the GPU and decides).
//!
//! Usage:
//! ```text
//! cargo run --release --no-default-features --features cpu --example bench_ms2_evidence -- \
//!   --data <export.json> [--b 4] [--n 5] [--m 128] [--case both] \
//!   [--host-count off] [--dispatch 16777216,4294967296,max] [--timeout-seconds 120]
//! ```
//!
//! Second pass per setting: one extra drained call (`synchronize`, one
//! `formula_evidence`, `synchronize`) is timed as `max_single_ms`. When the
//! setting runs a single launch that is exactly the longest single launch;
//! with several launches it is the drained whole-call time, i.e. an upper
//! bound on the longest launch (the library chunks internally, so one
//! outside call cannot time one launch of many without changing the
//! library, which this tool must not do).

use std::io::Write;
use std::path::PathBuf;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::{Duration, Instant};

use mamba3::backend::Device;
use mamba3::backends::Auto;
use mamba3::models::ms2::batch::DeviceSpectra;
use mamba3::models::ms2::chem::{composition_mass, tolerance};
use mamba3::models::ms2::experiment::{ExperimentSet, spectrum_batch_for};
use mamba3::models::ms2::formula_enum::{EnumDomain, RatioBounds, build_enum_meta};
use mamba3::models::ms2::formula_evidence::{
    EVIDENCE_PEAKS, formula_evidence as host_evidence, formula_evidence_lane_trials,
};
use mamba3::models::ms2::formula_head::DeviceEnumArtifacts;
use mamba3::models::ms2::targets::RecipeLimits;
use mamba3::tensor::Tensor;
use mamba3::tensor::ops::index::{IdTensor, read_all};
use mamba3::tensor::ops::ms2::{PeakBuffers, peak_select};
use mamba3::tensor::ops::ms2_enum::{EnumLaunch, cand_pad, enum_offsets};
use mamba3::tensor::ops::ms2_formula_evidence::{evidence_peaks, formula_evidence};

type R = Auto;
type E = f32;

/// Work budget per evidence lane (the production default; the bench does not
/// change it).
const WORK_MAX: u32 = 2_048;
/// Kept-peak capacity the evidence peaks are selected from.
const N_KEEP: usize = 128;
/// Widest fragment tolerance the contract allows, in tenths of a ppm.
const WIDEST_PPM_TENTHS: u32 = 1000;
/// Host trial counting runs only when the bounded estimate is at most this
/// many hydrogen trials; otherwise the per-lane bound is reported. Used only
/// under `--host-count auto` (today's threshold rule).
const TRIAL_AFFORDABLE_MAX: u64 = 200_000_000;
/// Default dispatch bounds, as in the original table.
const DEFAULT_DISPATCHES: [u64; 6] = [1 << 24, 1 << 26, 1 << 28, 1 << 30, 1 << 32, u64::MAX];

#[derive(Clone, Copy, PartialEq, Eq)]
enum HostCount {
    On,
    Off,
    Auto,
}

impl HostCount {
    fn name(self) -> &'static str {
        match self {
            HostCount::On => "on",
            HostCount::Off => "off",
            HostCount::Auto => "auto",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum CaseSel {
    Real,
    Adversarial,
    Both,
}

fn usage() -> ! {
    eprintln!(
        "usage: bench_ms2_evidence --data <export.json> [--b 4] [--n 5] [--m 128] [--case real|adversarial|both] [--host-count on|off|auto] [--dispatch <comma list, 'max' for u64::MAX>] [--timeout-seconds S]"
    );
    std::process::exit(2);
}

fn fail(msg: String) -> ! {
    eprintln!("bench_ms2_evidence: {msg}");
    std::process::exit(1);
}

fn flush_stdout() {
    std::io::stdout()
        .flush()
        .unwrap_or_else(|e| fail(format!("flush: {e}")));
}

/// Arm a watchdog for one timed setting: after `timeout_secs` seconds without
/// `done` being set it prints the `TIMEOUT` line, flushes and exits the
/// process with code 3. The device call itself cannot be cancelled, so the
/// watchdog does not interrupt it — it only guarantees the tool never hangs
/// silently. A zero timeout arms nothing.
fn arm_watchdog(case: &str, dispatch: u64, timeout_secs: u64) -> Arc<AtomicBool> {
    let done = Arc::new(AtomicBool::new(false));
    if timeout_secs == 0 {
        return done;
    }
    let flag = done.clone();
    let case = case.to_string();
    std::thread::spawn(move || {
        let step = Duration::from_millis(100);
        let mut waited = Duration::ZERO;
        let limit = Duration::from_secs(timeout_secs);
        loop {
            std::thread::sleep(step);
            if flag.load(Ordering::SeqCst) {
                return;
            }
            waited += step;
            if waited >= limit {
                println!("TIMEOUT case={case} dispatch={dispatch} after {timeout_secs} s");
                flush_stdout();
                std::process::exit(3);
            }
        }
    });
    done
}

fn parse_dispatch_list(s: &str) -> Vec<u64> {
    let mut out = Vec::new();
    for tok in s.split(',') {
        let tok = tok.trim();
        if tok.is_empty() {
            usage();
        }
        let v = if tok == "max"
            || tok == "u64max"
            || tok == "u64::MAX"
            || tok == "18446744073709551615"
        {
            u64::MAX
        } else if let Ok(v) = tok.parse::<u64>() {
            v
        } else {
            usage();
        };
        if !out.contains(&v) {
            out.push(v);
        }
    }
    if out.is_empty() {
        usage();
    }
    out
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

/// Per-lane worst-case hydrogen-trial bound of the `formula_evidence`
/// dispatch sizing: `work_max * P * trials_bound` with `trials_bound` from
/// the host-known `h_cap_max` / `tol_max` (the wrapper's documented sizing,
/// mirrored here so the printed launch counts match the kernel's).
fn per_lane_bound(work_max: u32, p: usize, h_cap_max: u32, tol_max: u32) -> u64 {
    let s_max = (u64::from(h_cap_max) * 7_825 + 2 * u64::from(tol_max)) / 1_000_000;
    let per_s = (2 * u64::from(tol_max)) / 7_825 + 2;
    let wrapped = (s_max + 1) * per_s;
    let trials_bound = wrapped.min(u64::from(h_cap_max) + 1).max(1);
    (work_max as u64)
        .checked_mul(p as u64)
        .and_then(|v| v.checked_mul(trials_bound))
        .expect("per-lane bound fits u64")
        .max(1)
}

/// Launches of one `formula_evidence` call over `lanes` lanes at this
/// dispatch bound (the wrapper's `ceil(lanes / per)` chunking).
fn launches_for(lanes: usize, dispatch: u64, lane_bound: u64) -> usize {
    if lanes == 0 {
        return 0;
    }
    let per = (dispatch / lane_bound).max(1) as usize;
    lanes.div_ceil(per)
}

fn main() {
    let mut data: Option<PathBuf> = None;
    let mut b = 4usize;
    let mut n = 5usize;
    let mut m = 128usize;
    let mut host_count = HostCount::Off;
    let mut case_sel = CaseSel::Both;
    let mut dispatches: Vec<u64> = DEFAULT_DISPATCHES.to_vec();
    let mut timeout_secs = 120u64;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut next = || args.next().unwrap_or_else(|| usage());
        match arg.as_str() {
            "--data" => data = Some(PathBuf::from(next())),
            "--b" => b = next().parse().unwrap_or_else(|_| usage()),
            "--n" => n = next().parse().unwrap_or_else(|_| usage()),
            "--m" => m = next().parse().unwrap_or_else(|_| usage()),
            "--host-count" => match next().as_str() {
                "on" => host_count = HostCount::On,
                "off" => host_count = HostCount::Off,
                "auto" => host_count = HostCount::Auto,
                _ => usage(),
            },
            "--case" => match next().as_str() {
                "real" => case_sel = CaseSel::Real,
                "adversarial" | "worst" => case_sel = CaseSel::Adversarial,
                "both" => case_sel = CaseSel::Both,
                _ => usage(),
            },
            "--dispatch" => dispatches = parse_dispatch_list(&next()),
            "--timeout-seconds" => timeout_secs = next().parse().unwrap_or_else(|_| usage()),
            _ => usage(),
        }
    }
    let Some(data_path) = data else { usage() };
    if b == 0 || n == 0 || !matches!(m, 32 | 128 | 512 | 2048) {
        usage();
    }
    let case_name = match case_sel {
        CaseSel::Real => "real",
        CaseSel::Adversarial => "adversarial",
        CaseSel::Both => "both",
    };
    // Header first, before any long computation, so a later hang still leaves
    // the configuration in the log.
    println!(
        "bench_ms2_evidence: B={b} M={m} N={n} N_KEEP={N_KEEP} P={EVIDENCE_PEAKS} work_max={WORK_MAX} host-count={} case={case_name} timeout-seconds={timeout_secs} dispatches=[{}]",
        host_count.name(),
        dispatches
            .iter()
            .map(|d| if *d == u64::MAX {
                "max".to_string()
            } else {
                d.to_string()
            })
            .collect::<Vec<_>>()
            .join(","),
    );
    flush_stdout();

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
    let h_cap_max = artifacts.hydrogen_cap_max();
    let lanes_max = 262_144u32;
    let lane_visits_max = 4_096u32;
    let scored_cap = 4096u32.min(m as u32);
    let lanes = b * m;
    println!(
        "setup: h_cap_max={h_cap_max} lanes={lanes} (B={b} M={m}); max_single_ms is the drained single call (exactly the longest launch when launches=1, else a whole-call upper bound)"
    );
    println!(
        "{:<12} {:>11} {:>7} {:>12} {:>12} {:>16} {:>6}",
        "case", "dispatch", "launches", "ms_per_call", "max_single", "h_trials", "src"
    );
    flush_stdout();
    let mut worst_single_ms = 0.0f64;

    let want_real = matches!(case_sel, CaseSel::Real | CaseSel::Both);
    let want_adv = matches!(case_sel, CaseSel::Adversarial | CaseSel::Both);

    // (a) real batch: precursors and peaks from the export, enumerated on
    // the device to `M` candidates, evidence peaks from the production
    // `evidence_peaks`.
    //
    // Built only when the real case is selected, so `--case adversarial`
    // never pays for it.
    let mut real_bufs = None;
    if want_real {
        let n_raw = bucket_n_raw(&set, &labeled);
        let real_batch = spectrum_batch_for(&set, &labeled, n_raw)
            .unwrap_or_else(|e| fail(format!("cannot build real batch: {e}")));
        let spectra = DeviceSpectra::upload(&real_batch, &device)
            .unwrap_or_else(|e| fail(format!("real upload: {e}")));
        let peaks = PeakBuffers::<R, E>::new(b, n_raw as usize, N_KEEP, &device);
        peak_select(
            &spectra.mz,
            &spectra.intensity,
            &spectra.meta,
            spectra.intensity_scale,
            &peaks,
        )
        .unwrap_or_else(|e| fail(format!("peak_select: {e}")));
        let meta_host = build_enum_meta(
            &real_batch,
            artifacts.domain_max_error,
            lane_visits_max,
            scored_cap,
        );
        let meta_t = IdTensor::from_slice(&meta_host, vec![b, 8], &device)
            .unwrap_or_else(|e| fail(format!("meta upload: {e}")));
        let enum_launch = EnumLaunch::from_chemistry();
        let lane_stats = IdTensor::empty(vec![b * artifacts.p, 2], &device);
        let offsets_t = IdTensor::empty(vec![b * artifacts.p], &device);
        let counters_t = IdTensor::empty(vec![b, 5], &device);
        let cand_t = IdTensor::empty(vec![b, m, 13], &device);
        enum_launch
            .count(
                &meta_t,
                &artifacts.rare,
                &artifacts.bounds,
                &lane_stats,
                lanes_max,
                u32::MAX,
                lane_visits_max,
            )
            .unwrap_or_else(|e| fail(format!("enum count: {e}")));
        enum_offsets(
            &lane_stats,
            &meta_t,
            &offsets_t,
            &counters_t,
            scored_cap,
            m,
            lanes_max,
        )
        .unwrap_or_else(|e| fail(format!("enum offsets: {e}")));
        enum_launch
            .fill(
                &meta_t,
                &artifacts.rare,
                &artifacts.bounds,
                &offsets_t,
                &cand_t,
                scored_cap,
                lanes_max,
                u32::MAX,
                lane_visits_max,
            )
            .unwrap_or_else(|e| fail(format!("enum fill: {e}")));
        cand_pad(&counters_t, &cand_t, artifacts.p, lanes_max)
            .unwrap_or_else(|e| fail(format!("enum pad: {e}")));
        let spec_t = spectra
            .evidence_spec(&device)
            .unwrap_or_else(|e| fail(format!("spec: {e}")));
        let mut ev_peaks_t = IdTensor::empty(vec![b, EVIDENCE_PEAKS, 4], &device);
        let mut ev_w_t = Tensor::<R, E>::empty(vec![b, EVIDENCE_PEAKS], &device);
        evidence_peaks(
            &peaks.kept,
            &peaks.kept_f,
            &spectra.meta,
            &spec_t,
            &mut ev_peaks_t,
            &mut ev_w_t,
        )
        .unwrap_or_else(|e| fail(format!("evidence_peaks: {e}")));
        let tol_max = spectra.uploaded_tol_max();
        let cand_ev_t = Tensor::<R, E>::empty(vec![b, m, 4], &device);
        real_bufs = Some((
            spectra, cand_t, ev_peaks_t, ev_w_t, spec_t, cand_ev_t, tol_max,
        ));
    }

    if let Some((spectra, cand_t, ev_peaks_t, ev_w_t, spec_t, mut cand_ev_t, tol_max)) = real_bufs {
        let lane_bound = per_lane_bound(WORK_MAX, EVIDENCE_PEAKS, h_cap_max, tol_max);
        let est = (lanes as u64).saturating_mul(lane_bound);
        // Banner before any long computation of this case (host counting,
        // timing), so the log shows where a hang sits.
        println!(
            "== case real: lanes={lanes} per_lane_bound={lane_bound} tol_max={tol_max} host-count={}",
            host_count.name()
        );
        flush_stdout();
        let count_exact = match host_count {
            HostCount::On => true,
            HostCount::Off => false,
            HostCount::Auto => est <= TRIAL_AFFORDABLE_MAX,
        };
        // Host copies for the trial-counting twin, read only when the mode
        // asks for exact trials (under `off` no host twin runs at all).
        let host_copies = if count_exact {
            let (id_bufs, float_bufs) =
                read_all(&[&cand_t, &ev_peaks_t, &spectra.meta, &spec_t], &[&ev_w_t])
                    .unwrap_or_else(|e| fail(format!("host read: {e}")));
            Some((id_bufs, float_bufs))
        } else {
            None
        };
        // Total hydrogen trials: the hidden trial-counting twin on the host
        // for the same inputs when asked for, otherwise the per-lane bound.
        let (trials, src): (u64, &'static str) = match host_copies {
            Some((ref id_bufs, ref float_bufs)) => {
                let (real_cand, real_ev_peaks, real_meta, real_spec) =
                    (&id_bufs[0], &id_bufs[1], &id_bufs[2], &id_bufs[3]);
                let real_ev_w = &float_bufs[0];
                let mut total = 0u64;
                // Scratch sized as the full [B, M, 4] row-major buffer:
                // the lane writes its row at the absolute lane index.
                let mut scratch = vec![0.0f32; lanes * 4];
                for lane in 0..lanes {
                    let bb = lane / m;
                    let mm = lane % m;
                    let mut t = 0u64;
                    formula_evidence_lane_trials(
                        real_cand,
                        real_ev_peaks,
                        real_ev_w,
                        real_meta,
                        real_spec,
                        bb as u32,
                        mm as u32,
                        m as u32,
                        EVIDENCE_PEAKS as u32,
                        WORK_MAX,
                        h_cap_max,
                        &mut scratch,
                        &mut t,
                    );
                    total += t;
                }
                (total, "exact")
            }
            None => (est, "bound"),
        };
        for &dispatch in &dispatches {
            let launches = launches_for(lanes, dispatch, lane_bound);
            let watch = arm_watchdog("real", dispatch, timeout_secs);
            device.synchronize();
            let t0 = Instant::now();
            for _ in 0..n {
                formula_evidence(
                    &cand_t,
                    &ev_peaks_t,
                    &ev_w_t,
                    &spectra.meta,
                    &spec_t,
                    &mut cand_ev_t,
                    WORK_MAX,
                    dispatch,
                    h_cap_max,
                    tol_max,
                )
                .unwrap_or_else(|e| fail(format!("formula_evidence: {e}")));
            }
            device.synchronize();
            let ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
            // Second, separate drained pass: one call between two drains.
            // With one launch this is exactly the longest single launch; with
            // several it is the whole-call upper bound.
            device.synchronize();
            let t1 = Instant::now();
            formula_evidence(
                &cand_t,
                &ev_peaks_t,
                &ev_w_t,
                &spectra.meta,
                &spec_t,
                &mut cand_ev_t,
                WORK_MAX,
                dispatch,
                h_cap_max,
                tol_max,
            )
            .unwrap_or_else(|e| fail(format!("formula_evidence: {e}")));
            device.synchronize();
            let single_ms = t1.elapsed().as_secs_f64() * 1000.0;
            watch.store(true, Ordering::SeqCst);
            println!(
                "{:<12} {:>11} {:>7} {:>12.3} {:>12.3} {:>16} {:>6}",
                "real", dispatch, launches, ms, single_ms, trials, src
            );
            flush_stdout();
            if launches == 1 {
                worst_single_ms = worst_single_ms.max(single_ms);
            }
        }
    }

    // (b) adversarial worst case: hydrogen-rich, heteroatom-rich candidates
    // near the top of the domain mass, 32 eligible peaks, the widest
    // fragment tolerance the contract allows.
    //
    // When the host counter is off the 40-round host-twin perturbation below
    // is skipped entirely, so this construction carries NO guarantee that no
    // peak is explained (and no early exit happens): instead the measured
    // explained fraction is read back from the device result once after the
    // timed calls and printed (see the `adversarial evidence check` line).
    // With `--host-count on` (or `auto` while affordable) the historical
    // verification runs and the peaks explain nothing by construction.
    if want_adv {
        let top_mass = comps
            .iter()
            .map(|c| composition_mass(c).unwrap_or(0))
            .max()
            .unwrap_or(0);
        let adv_h = h_cap_max.saturating_sub(3).min(120).max(8);
        // Heteroatom-rich counts with a radix product above WORK_MAX so every
        // lane runs all WORK_MAX visits; carbon closes the mass near the top.
        let (an, ao, af, ap, ass, acl, abr) = (12u32, 12, 3, 2, 4, 2, 1);
        let heavy_mass = an * 14_003_074
            + ao * 15_994_915
            + af * 18_998_403
            + ap * 30_973_762
            + ass * 31_972_071
            + acl * 34_968_853
            + abr * 78_918_338;
        let h_mass = adv_h * 1_007_825;
        let rest = heavy_mass.saturating_add(h_mass);
        let adv_c = top_mass
            .saturating_sub(rest)
            .saturating_sub(1_007_825 - 549)
            / 12_000_000;
        let adv_c = adv_c.clamp(8, 400);
        let adv_counts = [adv_c, adv_h, an, ao, af, ap, ass, acl, abr, 0];
        let adv_mass: u32 = adv_counts
            .iter()
            .zip([
                12_000_000u32,
                1_007_825,
                14_003_074,
                15_994_915,
                18_998_403,
                30_973_762,
                31_972_071,
                34_968_853,
                78_918_338,
                126_904_472,
            ])
            .map(|(&c, m)| c.saturating_mul(m))
            .fold(0u32, |a, x| a.saturating_add(x));
        let mut adv_cand = vec![0u32; b * m * 13];
        for i in 0..b * m {
            for (e, &c) in adv_counts.iter().enumerate() {
                adv_cand[i * 13 + e] = c;
            }
            adv_cand[i * 13 + 10] = adv_mass;
            adv_cand[i * 13 + 11] = 1;
            adv_cand[i * 13 + 12] = u32::MAX;
        }
        let adv_prec = adv_mass + 1_007_825 - 549;
        let mut adv_meta = vec![0u32; b * 8];
        for i in 0..b {
            adv_meta[i * 8] = 32;
            adv_meta[i * 8 + 1] = adv_prec;
            adv_meta[i * 8 + 2] = 50;
            adv_meta[i * 8 + 3] = 1;
            adv_meta[i * 8 + 4] = WIDEST_PPM_TENTHS;
            adv_meta[i * 8 + 5] = 200;
            adv_meta[i * 8 + 6] = (i + 1) as u32;
            adv_meta[i * 8 + 7] = 0;
        }
        let mut adv_spec = vec![0u32; b * 2];
        for i in 0..b {
            adv_spec[i * 2] = 50;
        }
        // 32 peaks spread over 300..1400 Da; tolerances at the widest ppm.
        let mut adv_ev_peaks = vec![0u32; b * EVIDENCE_PEAKS * 4];
        let mut adv_ev_w = vec![0f32; b * EVIDENCE_PEAKS];
        for i in 0..b {
            for s in 0..EVIDENCE_PEAKS {
                let mz = 300_000_000u32 + (s as u32) * 34_000_000 + (i as u32) * 1_000_000;
                let t = mz + 549;
                let tol = tolerance(mz, WIDEST_PPM_TENTHS);
                adv_ev_peaks[(i * EVIDENCE_PEAKS + s) * 4] = s as u32;
                adv_ev_peaks[(i * EVIDENCE_PEAKS + s) * 4 + 1] = t;
                adv_ev_peaks[(i * EVIDENCE_PEAKS + s) * 4 + 2] = tol;
                adv_ev_peaks[(i * EVIDENCE_PEAKS + s) * 4 + 3] = 1;
                adv_ev_w[i * EVIDENCE_PEAKS + s] = 1.0 / EVIDENCE_PEAKS as f32;
            }
        }
        let adv_tol_max: u32 = (0..b * EVIDENCE_PEAKS)
            .map(|s| adv_ev_peaks[s * 4 + 2])
            .max()
            .unwrap_or(0);
        let adv_lane_bound = per_lane_bound(WORK_MAX, EVIDENCE_PEAKS, h_cap_max, adv_tol_max);
        let adv_est = (lanes as u64).saturating_mul(adv_lane_bound);
        // Banner before any long computation of this case.
        println!(
            "== case adversarial: lanes={lanes} per_lane_bound={adv_lane_bound} tol_max={adv_tol_max} adv_mass={adv_mass} top_mass={top_mass} host-count={}",
            host_count.name()
        );
        flush_stdout();
        let verify_with_host = match host_count {
            HostCount::On => true,
            HostCount::Off => false,
            HostCount::Auto => adv_est <= TRIAL_AFFORDABLE_MAX,
        };
        if verify_with_host {
            // Verify nothing is explained; shift every target until exact
            // (the reachable set is sparse, so this converges in a few
            // rounds).
            for _ in 0..40 {
                let out = host_evidence(
                    &adv_cand,
                    &adv_ev_peaks,
                    &adv_ev_w,
                    &adv_meta,
                    &adv_spec,
                    b,
                    m,
                    EVIDENCE_PEAKS,
                    WORK_MAX,
                    h_cap_max,
                );
                let mut explained_any = false;
                for v in out.chunks_exact(4) {
                    if v[0] != 0.0 {
                        explained_any = true;
                        break;
                    }
                }
                if !explained_any {
                    break;
                }
                for w in adv_ev_peaks.chunks_exact_mut(4) {
                    if w[3] == 1 {
                        w[1] = w[1].wrapping_add(3_000_007);
                    }
                }
            }
        }
        let adv_cand_t = IdTensor::from_slice(&adv_cand, vec![b, m, 13], &device)
            .unwrap_or_else(|e| fail(format!("adv cand upload: {e}")));
        let adv_peaks_t = IdTensor::from_slice(&adv_ev_peaks, vec![b, EVIDENCE_PEAKS, 4], &device)
            .unwrap_or_else(|e| fail(format!("adv peaks upload: {e}")));
        let adv_w_t = Tensor::<R, E>::from_f32(&adv_ev_w, vec![b, EVIDENCE_PEAKS], &device)
            .unwrap_or_else(|e| fail(format!("adv w upload: {e}")));
        let adv_meta_t = IdTensor::from_slice(&adv_meta, vec![b, 8], &device)
            .unwrap_or_else(|e| fail(format!("adv meta upload: {e}")));
        let adv_spec_t = IdTensor::from_slice(&adv_spec, vec![b, 2], &device)
            .unwrap_or_else(|e| fail(format!("adv spec upload: {e}")));
        let mut adv_ev_t = Tensor::<R, E>::empty(vec![b, m, 4], &device);
        // The adversarial case has no exact host trial counter (as before):
        // the per-lane bound is always reported.
        let (adv_trials, adv_src): (u64, &'static str) = (adv_est, "bound");
        for &dispatch in &dispatches {
            let launches = launches_for(lanes, dispatch, adv_lane_bound);
            let watch = arm_watchdog("adversarial", dispatch, timeout_secs);
            device.synchronize();
            let t0 = Instant::now();
            for _ in 0..n {
                formula_evidence(
                    &adv_cand_t,
                    &adv_peaks_t,
                    &adv_w_t,
                    &adv_meta_t,
                    &adv_spec_t,
                    &mut adv_ev_t,
                    WORK_MAX,
                    dispatch,
                    h_cap_max,
                    adv_tol_max,
                )
                .unwrap_or_else(|e| fail(format!("formula_evidence: {e}")));
            }
            device.synchronize();
            let ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
            device.synchronize();
            let t1 = Instant::now();
            formula_evidence(
                &adv_cand_t,
                &adv_peaks_t,
                &adv_w_t,
                &adv_meta_t,
                &adv_spec_t,
                &mut adv_ev_t,
                WORK_MAX,
                dispatch,
                h_cap_max,
                adv_tol_max,
            )
            .unwrap_or_else(|e| fail(format!("formula_evidence: {e}")));
            device.synchronize();
            let single_ms = t1.elapsed().as_secs_f64() * 1000.0;
            watch.store(true, Ordering::SeqCst);
            println!(
                "{:<12} {:>11} {:>7} {:>12.3} {:>12.3} {:>16} {:>6}",
                "adversarial", dispatch, launches, ms, single_ms, adv_trials, adv_src
            );
            flush_stdout();
            if launches == 1 {
                worst_single_ms = worst_single_ms.max(single_ms);
            }
        }
        // No host-twin guarantee when the verification above was skipped, so
        // report what the device actually computed: one read of `cand_ev`
        // after the timed calls (columns: explained-peak count, explained
        // weight, evidence count, complete flag).
        device.synchronize();
        match adv_ev_t.try_to_f32() {
            Ok(ev) => {
                let mut explained_sum = 0.0f64;
                let mut complete = 0u64;
                for v in ev.chunks_exact(4) {
                    explained_sum += v[0] as f64;
                    if v[3] != 0.0 {
                        complete += 1;
                    }
                }
                let mean_explained = explained_sum / lanes.max(1) as f64;
                let complete_frac = complete as f64 / lanes.max(1) as f64;
                println!(
                    "adversarial evidence check: mean_explained={mean_explained:.4} complete_frac={complete_frac:.4} (one device read of cand_ev after timing; verified_exact={verify_with_host})"
                );
            }
            Err(e) => {
                println!("adversarial evidence check: unreadable ({e})");
            }
        }
        flush_stdout();
    }

    println!(
        "summary: worst-case single-launch formula_evidence time on {} is {worst_single_ms:.3} ms/call (drained single call with launches=1, pipelined average over {n} otherwise; the default formula_evidence_dispatch_max is unchanged)",
        std::any::type_name::<R>()
    );
    flush_stdout();
}
