//! P3-A tests: MS2 peak-selection kernels, peak features and adapters against
//! their host twins, and the twin against the P1 `filter_peaks` reference.
//!
//! Every device call is followed by [`check_launches`], so a kernel that
//! failed to compile or run is an error rather than stale data.

#![cfg(feature = "backend")]

use mamba3::autograd::Var;
use mamba3::backend::{Device, check_launches};
use mamba3::backends::Auto;
use mamba3::models::ms2::targets::filter_peaks;
use mamba3::models::ms2::twin;
use mamba3::tensor::Tensor;
use mamba3::tensor::ops::index::IdTensor;
use mamba3::tensor::ops::ms2;
use mamba3::tensor::ops::random::Rng;

type R = Auto;

fn dev() -> Device<R> {
    Device::<R>::default()
}

/// Float comparison of the task: `1e-5` absolute for values in `[-1, 1]`,
/// `1e-5` relative otherwise.
fn assert_close(actual: &[f32], expected: &[f32], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        let ok = if e.abs() <= 1.0 {
            (a - e).abs() <= 1e-5
        } else {
            (a - e).abs() <= 1e-5 * e.abs()
        };
        assert!(ok, "{what}: index {i} got {a}, want {e}");
    }
}

fn assert_ids(actual: &[u32], expected: &[u32], what: &str) {
    assert_eq!(actual, expected, "{what} differs");
}

fn upload_ids(data: &[u32], shape: Vec<usize>, device: &Device<R>) -> IdTensor<R> {
    IdTensor::from_slice(data, shape, device).unwrap()
}

fn upload_f(data: &[f32], shape: Vec<usize>, device: &Device<R>) -> Tensor<R, f32> {
    Tensor::<R, f32>::from_f32(data, shape, device).unwrap()
}

/// One random batch in the task's intensity scheme: base intensities sit well
/// above the `1e-3` floor (relative `>= 6.6e-3`), so only the explicit
/// patches below exercise the threshold, with clear margins on both sides.
fn random_batch(
    seed: u64,
    batch: usize,
    n_raw: usize,
    counts: &[usize],
    precursor_base: u32,
) -> (Vec<u32>, Vec<f32>, Vec<u32>) {
    let mut rng = Rng::seeded(seed);
    let mz_host: Vec<f32> = rng.uniform_vec(batch * n_raw, 50_000_000.0, 1_450_000_000.0);
    let u_host: Vec<f32> = rng.uniform_vec(batch * n_raw, 0.0, 1.0);
    let mut mz = vec![0u32; batch * n_raw];
    let mut intensity = vec![0.0f32; batch * n_raw];
    let mut meta = vec![0u32; batch * 8];
    for b in 0..batch {
        let precursor = precursor_base + b as u32 * 100_000_000;
        for i in 0..counts[b] {
            mz[b * n_raw + i] = mz_host[b * n_raw + i] as u32;
            let u = u_host[b * n_raw + i];
            intensity[b * n_raw + i] = 0.02 + 3.0 * u * u;
        }
        meta[b * 8] = counts[b] as u32;
        meta[b * 8 + 1] = precursor;
        meta[b * 8 + 2] = 50;
        meta[b * 8 + 3] = 1;
        meta[b * 8 + 4] = 100;
        meta[b * 8 + 5] = 200;
        meta[b * 8 + 6] = 0;
        meta[b * 8 + 7] = 0;
    }
    (mz, intensity, meta)
}

/// Run device selection on poisoned buffers and read every output back.
struct Selected {
    stats: Vec<f32>,
    rank: Vec<u32>,
    position: Vec<u32>,
    kept: Vec<u32>,
    kept_f: Vec<f32>,
    summary: Vec<u32>,
}

fn run_select(
    mz: &[u32],
    intensity: &[f32],
    meta: &[u32],
    batch: usize,
    n_raw: usize,
    n_keep: usize,
    scale: u32,
) -> Selected {
    let device = dev();
    let mz_t = IdTensor::from_slice(mz, vec![batch, n_raw], &device).unwrap();
    let int_t = Tensor::<R, f32>::from_f32(intensity, vec![batch, n_raw], &device).unwrap();
    let meta_t = IdTensor::from_slice(meta, vec![batch, 8], &device).unwrap();
    let out = ms2::PeakBuffers::poisoned(batch, n_raw, n_keep, &device).unwrap();
    ms2::peak_select(&mz_t, &int_t, &meta_t, scale, &out).unwrap();
    check_launches(&device).unwrap();
    let stats = out.stats.try_to_f32().unwrap();
    let rank = out.rank.try_to_vec().unwrap();
    let position = out.position.try_to_vec().unwrap();
    let kept = out.kept.try_to_vec().unwrap();
    let kept_f = out.kept_f.try_to_f32().unwrap();
    let summary = out.summary.try_to_vec().unwrap();
    Selected {
        stats,
        rank,
        position,
        kept,
        kept_f,
        summary,
    }
}

fn check_select(
    mz: &[u32],
    intensity: &[f32],
    meta: &[u32],
    batch: usize,
    n_raw: usize,
    n_keep: usize,
    scale: u32,
    what: &str,
) {
    let got = run_select(mz, intensity, meta, batch, n_raw, n_keep, scale);
    let twin = twin::peak_select(mz, intensity, meta, batch, n_raw, n_keep, scale);
    assert_ids(&got.rank, &twin.rank, &format!("{what} rank"));
    assert_ids(&got.position, &twin.position, &format!("{what} position"));
    assert_ids(&got.kept, &twin.kept, &format!("{what} kept"));
    assert_ids(&got.summary, &twin.summary, &format!("{what} summary"));
    assert_close(&got.stats, &twin.stats, &format!("{what} stats"));
    assert_close(&got.kept_f, &twin.kept_f, &format!("{what} kept_f"));
}

#[test]
fn peak_selection_random_64_matches_twin() {
    let (batch, n_raw, n_keep) = (5, 64, 16);
    let counts = [0usize, 1, 16, 17, 64];
    let (mut mz, mut intensity, meta) = random_batch(7, batch, n_raw, &counts, 400_000_000);
    // Spectrum 4 carries the adversarial patches: duplicate m/z, equal
    // intensities, peaks above the precursor bound (one exactly on it), and
    // intensities far below and clearly above the floor.
    let b = 4;
    let precursor = meta[b * 8 + 1];
    mz[b * n_raw + 1] = mz[b * n_raw];
    mz[b * n_raw + 2] = mz[b * n_raw];
    for i in 3..6 {
        intensity[b * n_raw + i] = 5.0;
    }
    mz[b * n_raw + 7] = precursor + 3_000_000;
    mz[b * n_raw + 8] = precursor + 2_000_000;
    mz[b * n_raw + 9] = precursor + 2_000_001;
    let top = intensity[b * n_raw..(b + 1) * n_raw]
        .iter()
        .fold(0.0f32, |a, &v| a.max(v));
    intensity[b * n_raw + 10] = top * 1e-6;
    intensity[b * n_raw + 11] = top * 2e-3;
    check_select(&mz, &intensity, &meta, batch, n_raw, n_keep, 0, "case-a");
}

#[test]
fn peak_selection_clamps_over_capacity_count() {
    // Invalid metadata (`peak_count = n_raw + 5`) must never index past the
    // inputs: the kernel clamps to `n_raw`, and the twin does the same.
    let (batch, n_raw, n_keep) = (2, 16, 8);
    let counts = [16usize, 16];
    let (mz, intensity, mut meta) = random_batch(7, batch, n_raw, &counts, 400_000_000);
    meta[0] = n_raw as u32 + 5;
    meta[8] = n_raw as u32 + 5;
    check_select(
        &mz,
        &intensity,
        &meta,
        batch,
        n_raw,
        n_keep,
        0,
        "over-capacity",
    );
}

#[test]
fn peak_select_rejects_rank_one_mz() {
    let device = dev();
    let mz = upload_ids(&[1u32; 8], vec![8], &device);
    let int_t = upload_f(&[1.0f32; 8], vec![1, 8], &device);
    let meta = upload_ids(
        &[8u32, 400_000_000, 50, 1, 100, 200, 0, 0],
        vec![1, 8],
        &device,
    );
    let out = ms2::PeakBuffers::new(1, 8, 4, &device);
    assert!(ms2::peak_select(&mz, &int_t, &meta, 0, &out).is_err());
}

#[test]
fn peak_select_rejects_bad_meta_width() {
    let device = dev();
    let mz = upload_ids(&[1u32; 8], vec![1, 8], &device);
    let int_t = upload_f(&[1.0f32; 8], vec![1, 8], &device);
    let meta = upload_ids(&[0u32; 7], vec![1, 7], &device);
    let out = ms2::PeakBuffers::new(1, 8, 4, &device);
    assert!(ms2::peak_select(&mz, &int_t, &meta, 0, &out).is_err());
}

#[test]
fn peak_select_rejects_bad_kept_last_dim() {
    let device = dev();
    let mz = upload_ids(&[1u32; 8], vec![1, 8], &device);
    let int_t = upload_f(&[1.0f32; 8], vec![1, 8], &device);
    let meta = upload_ids(
        &[8u32, 400_000_000, 50, 1, 100, 200, 0, 0],
        vec![1, 8],
        &device,
    );
    let out = ms2::PeakBuffers::<R, f32>::poisoned(1, 8, 4, &device).unwrap();
    let bad_kept = upload_ids(&[0u32; 1 * 4 * 2], vec![1, 4, 2], &device);
    let bad = ms2::PeakBuffers {
        stats: out.stats,
        rank: out.rank,
        position: out.position,
        kept: bad_kept,
        kept_f: out.kept_f,
        summary: out.summary,
    };
    assert!(ms2::peak_select(&mz, &int_t, &meta, 0, &bad).is_err());
}

#[test]
fn peak_select_rejects_bad_summary_shape() {
    let device = dev();
    let mz = upload_ids(&[1u32; 8], vec![1, 8], &device);
    let int_t = upload_f(&[1.0f32; 8], vec![1, 8], &device);
    let meta = upload_ids(
        &[8u32, 400_000_000, 50, 1, 100, 200, 0, 0],
        vec![1, 8],
        &device,
    );
    let out = ms2::PeakBuffers::<R, f32>::poisoned(1, 8, 4, &device).unwrap();
    let bad_summary = upload_ids(&[0u32; 3], vec![1, 3], &device);
    let bad = ms2::PeakBuffers {
        stats: out.stats,
        rank: out.rank,
        position: out.position,
        kept: out.kept,
        kept_f: out.kept_f,
        summary: bad_summary,
    };
    assert!(ms2::peak_select(&mz, &int_t, &meta, 0, &bad).is_err());
}

#[test]
fn peak_features_rejects_rank_two_kept() {
    let device = dev();
    let waves = ms2::Ms2Constants::new(&device);
    let kept = upload_ids(&[0u32; 8], vec![2, 4], &device);
    let kept_f = upload_f(&[0.0f32; 8], vec![1, 4, 2], &device);
    let meta = upload_ids(&[0u32; 8], vec![1, 8], &device);
    let out = upload_f(&[0.0f32; 1 * 4 * 71], vec![1, 4, 71], &device);
    assert!(ms2::peak_features(&kept, &kept_f, &meta, &waves, &out).is_err());
}

#[test]
fn peak_features_rejects_bad_kept_last_dim() {
    let device = dev();
    let waves = ms2::Ms2Constants::new(&device);
    let kept = upload_ids(&[0u32; 1 * 4 * 2], vec![1, 4, 2], &device);
    let kept_f = upload_f(&[0.0f32; 8], vec![1, 4, 2], &device);
    let meta = upload_ids(&[0u32; 8], vec![1, 8], &device);
    let out = upload_f(&[0.0f32; 1 * 4 * 71], vec![1, 4, 71], &device);
    assert!(ms2::peak_features(&kept, &kept_f, &meta, &waves, &out).is_err());
}

#[test]
fn peak_features_rejects_bad_out_width() {
    let device = dev();
    let waves = ms2::Ms2Constants::new(&device);
    let kept = upload_ids(&[0u32; 1 * 4 * 3], vec![1, 4, 3], &device);
    let kept_f = upload_f(&[0.0f32; 8], vec![1, 4, 2], &device);
    let meta = upload_ids(&[0u32; 8], vec![1, 8], &device);
    let out = upload_f(&[0.0f32; 1 * 4 * 70], vec![1, 4, 70], &device);
    assert!(ms2::peak_features(&kept, &kept_f, &meta, &waves, &out).is_err());
}

#[test]
fn peak_selection_random_512_matches_twin() {
    let (batch, n_raw, n_keep) = (3, 512, 128);
    let counts = [64usize, 511, 512];
    let (mut mz, mut intensity, meta) = random_batch(1234, batch, n_raw, &counts, 300_000_000);
    let b = 2;
    let precursor = meta[b * 8 + 1];
    mz[b * n_raw + 1] = mz[b * n_raw];
    intensity[b * n_raw + 2] = intensity[b * n_raw + 3];
    mz[b * n_raw + 4] = precursor + 10_000_000;
    let top = intensity[b * n_raw..(b + 1) * n_raw]
        .iter()
        .fold(0.0f32, |a, &v| a.max(v));
    intensity[b * n_raw + 5] = top * 1e-7;
    check_select(&mz, &intensity, &meta, batch, n_raw, n_keep, 0, "case-b");
}

#[test]
fn peak_selection_scale_one_overflow_is_ineligible() {
    let (batch, n_raw, n_keep) = (2, 32, 8);
    let counts = [32usize, 32];
    let (mut mz, mut intensity, meta) = random_batch(99, batch, n_raw, &counts, 500_000_000);
    // Square-root scale inputs live in (0, 1]; one overflowing square.
    for v in intensity.iter_mut() {
        *v = (*v / 3.02).clamp(0.0, 1.0);
    }
    intensity[n_raw + 3] = 1e30;
    mz[n_raw + 3] = meta[8 + 1] - 1_000_000;
    check_select(&mz, &intensity, &meta, batch, n_raw, n_keep, 1, "case-c");
}

#[test]
fn peak_selection_all_zero_intensities() {
    let (batch, n_raw, n_keep) = (2, 16, 8);
    let counts = [16usize, 5];
    let (mz, _, meta) = random_batch(5, batch, n_raw, &counts, 400_000_000);
    let intensity = vec![0.0f32; batch * n_raw];
    let got = run_select(&mz, &intensity, &meta, batch, n_raw, n_keep, 0);
    let twin = twin::peak_select(&mz, &intensity, &meta, batch, n_raw, n_keep, 0);
    assert_ids(&got.rank, &twin.rank, "case-d rank");
    assert_ids(&got.position, &twin.position, "case-d position");
    assert_ids(&got.kept, &twin.kept, "case-d kept");
    assert_ids(&got.summary, &twin.summary, "case-d summary");
    assert_close(&got.stats, &twin.stats, "case-d stats");
    assert_close(&got.kept_f, &twin.kept_f, "case-d kept_f");
    assert!(got.rank.iter().all(|&r| r == u32::MAX));
    assert!(got.summary.iter().all(|&s| s == 0));
    assert!(got.stats.iter().all(|&s| s == 0.0));
}

#[test]
fn peak_selection_all_above_precursor() {
    let (batch, n_raw, n_keep) = (2, 16, 8);
    let counts = [16usize, 16];
    let (_, intensity, mut meta) = random_batch(6, batch, n_raw, &counts, 400_000_000);
    let mut mz = vec![0u32; batch * n_raw];
    for b in 0..batch {
        for i in 0..n_raw {
            mz[b * n_raw + i] = meta[b * 8 + 1] + 5_000_000 + i as u32;
        }
        meta[b * 8 + 2] = 50;
    }
    check_select(&mz, &intensity, &meta, batch, n_raw, n_keep, 0, "case-e");
}

#[test]
fn peak_selection_ignores_poisoned_padding() {
    let (batch, n_raw, n_keep) = (5, 64, 16);
    let counts = [0usize, 1, 16, 17, 64];
    let (mz, intensity, meta) = random_batch(7, batch, n_raw, &counts, 400_000_000);
    let clean = run_select(&mz, &intensity, &meta, batch, n_raw, n_keep, 0);
    // Same valid peaks, but every padding slot holds NaN intensity and a
    // hostile m/z: neither may reach an output.
    let mut mz_bad = mz.clone();
    let mut int_bad = intensity.clone();
    for b in 0..batch {
        for i in counts[b]..n_raw {
            mz_bad[b * n_raw + i] = if i % 2 == 0 { u32::MAX } else { 0 };
            int_bad[b * n_raw + i] = f32::NAN;
        }
    }
    let dirty = run_select(&mz_bad, &int_bad, &meta, batch, n_raw, n_keep, 0);
    assert_ids(&dirty.rank, &clean.rank, "padding rank");
    assert_ids(&dirty.position, &clean.position, "padding position");
    assert_ids(&dirty.kept, &clean.kept, "padding kept");
    assert_ids(&dirty.summary, &clean.summary, "padding summary");
    for (name, d, c) in [
        ("stats", &dirty.stats, &clean.stats),
        ("kept_f", &dirty.kept_f, &clean.kept_f),
    ] {
        assert_eq!(d.len(), c.len(), "{name} length");
        for (i, (a, e)) in d.iter().zip(c.iter()).enumerate() {
            assert!(
                a.to_bits() == e.to_bits(),
                "{name}[{i}] differs with poisoned padding: {a} vs {e}"
            );
        }
    }
}

#[test]
fn twin_matches_filter_peaks_plus_top_n() {
    for (seed, batch, n_raw, n_keep, counts) in [
        (7u64, 5usize, 64usize, 16usize, vec![0, 1, 16, 17, 64]),
        (1234u64, 3usize, 512usize, 128usize, vec![64, 511, 512]),
    ] {
        let (mut mz, mut intensity, meta) = random_batch(seed, batch, n_raw, &counts, 400_000_000);
        if seed == 7 {
            let b = 4;
            let precursor = meta[b * 8 + 1];
            mz[b * n_raw + 1] = mz[b * n_raw];
            mz[b * n_raw + 7] = precursor + 3_000_000;
            let top = intensity[b * n_raw..(b + 1) * n_raw]
                .iter()
                .fold(0.0f32, |a, &v| a.max(v));
            intensity[b * n_raw + 10] = top * 1e-6;
        }
        let twin = twin::peak_select(&mz, &intensity, &meta, batch, n_raw, n_keep, 0);
        for b in 0..batch {
            let base = b * n_raw;
            let ids: Vec<u32> = (0..n_raw as u32).collect();
            let int_f64: Vec<f64> = intensity[base..base + n_raw]
                .iter()
                .map(|&v| v as f64)
                .collect();
            let filtered =
                filter_peaks(&ids, &mz[base..base + n_raw], &int_f64, meta[b * 8 + 1]).unwrap();
            // The twin's kept set is the filter output: same threshold, same bound.
            let kept_n = twin.rank[base..base + n_raw]
                .iter()
                .filter(|&&r| r != u32::MAX)
                .count();
            assert_eq!(kept_n, filtered.len(), "spectrum {b}: kept count");
            // ... and the selected set is its `n_keep` most intense, ties by index.
            let mut by_int: Vec<(u32, f64)> =
                filtered.iter().map(|p| (p.id, p.intensity)).collect();
            by_int.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            by_int.truncate(n_keep);
            let mut want: Vec<u32> = by_int.iter().map(|&(id, _)| id).collect();
            want.sort_unstable();
            let mut got: Vec<u32> = (0..n_raw as u32)
                .filter(|&i| twin.position[base + i as usize] != u32::MAX)
                .collect();
            got.sort_unstable();
            assert_eq!(got, want, "spectrum {b}: selected set");
            // Kept slots run in increasing m/z, ties by index.
            let len = twin.summary[b * 2] as usize;
            assert_eq!(len, got.len(), "spectrum {b}: len");
            let mut order: Vec<(u32, u32)> =
                got.iter().map(|&i| (mz[base + i as usize], i)).collect();
            order.sort_unstable();
            for (p, &(_, i)) in order.iter().enumerate() {
                let slot = (b * n_keep + p) * 3;
                assert_eq!(twin.kept[slot], i, "spectrum {b} slot {p}: raw index");
                assert_eq!(
                    twin.kept[slot + 2],
                    (len - 1 - p) as u32,
                    "spectrum {b} slot {p}: reverse"
                );
            }
            // Truncation bit and retained fraction agree with the reference.
            let truncated = filtered.len() > n_keep;
            assert_eq!(
                twin.summary[b * 2 + 1] & (1 << 19) != 0,
                truncated,
                "spectrum {b}: truncation bit"
            );
            if filtered.is_empty() {
                assert_eq!(twin.stats[b * 3 + 2], 0.0, "spectrum {b}: empty retained");
            } else {
                let kept_sum: f64 = filtered.iter().map(|p| p.intensity).sum();
                let sel_sum: f64 = by_int.iter().map(|&(_, r)| r).sum();
                let want_ret = (sel_sum / kept_sum) as f32;
                let got_ret = twin.stats[b * 3 + 2];
                assert!(
                    (got_ret - want_ret).abs() <= 1e-5,
                    "spectrum {b}: retained {got_ret} vs {want_ret}"
                );
            }
        }
    }
}

#[test]
fn peak_selection_batch_permutation() {
    let (batch, n_raw, n_keep) = (5, 64, 16);
    let counts = [0usize, 1, 16, 17, 64];
    let (mz, intensity, meta) = random_batch(7, batch, n_raw, &counts, 400_000_000);
    let base = run_select(&mz, &intensity, &meta, batch, n_raw, n_keep, 0);
    let perm = [4usize, 2, 0, 3, 1];
    let take_rows = |v: &[u32], width: usize| {
        let mut out = vec![0u32; batch * width];
        for (nb, &ob) in perm.iter().enumerate() {
            out[nb * width..(nb + 1) * width].copy_from_slice(&v[ob * width..(ob + 1) * width]);
        }
        out
    };
    let take_frows = |v: &[f32], width: usize| {
        let mut out = vec![0.0f32; batch * width];
        for (nb, &ob) in perm.iter().enumerate() {
            out[nb * width..(nb + 1) * width].copy_from_slice(&v[ob * width..(ob + 1) * width]);
        }
        out
    };
    let mut mz_p = vec![0u32; batch * n_raw];
    let mut int_p = vec![0.0f32; batch * n_raw];
    let mut meta_p = vec![0u32; batch * 8];
    for (nb, &ob) in perm.iter().enumerate() {
        mz_p[nb * n_raw..(nb + 1) * n_raw].copy_from_slice(&mz[ob * n_raw..(ob + 1) * n_raw]);
        int_p[nb * n_raw..(nb + 1) * n_raw]
            .copy_from_slice(&intensity[ob * n_raw..(ob + 1) * n_raw]);
        meta_p[nb * 8..(nb + 1) * 8].copy_from_slice(&meta[ob * 8..(ob + 1) * 8]);
    }
    let got = run_select(&mz_p, &int_p, &meta_p, batch, n_raw, n_keep, 0);
    assert_ids(&got.rank, &take_rows(&base.rank, n_raw), "perm rank");
    assert_ids(
        &got.position,
        &take_rows(&base.position, n_raw),
        "perm position",
    );
    assert_ids(&got.kept, &take_rows(&base.kept, n_keep * 3), "perm kept");
    assert_ids(&got.summary, &take_rows(&base.summary, 2), "perm summary");
    assert_close(&got.stats, &take_frows(&base.stats, 3), "perm stats");
    assert_close(
        &got.kept_f,
        &take_frows(&base.kept_f, n_keep * 2),
        "perm kept_f",
    );
}

fn run_features(
    kept: &[u32],
    kept_f: &[f32],
    meta: &[u32],
    batch: usize,
    n_keep: usize,
) -> Vec<f32> {
    let device = dev();
    let waves = ms2::Ms2Constants::new(&device);
    let kept_t = IdTensor::from_slice(kept, vec![batch, n_keep, 3], &device).unwrap();
    let kept_f_t = Tensor::from_f32(kept_f, vec![batch, n_keep, 2], &device).unwrap();
    let meta_t = IdTensor::from_slice(meta, vec![batch, 8], &device).unwrap();
    let out = Tensor::<R, f32>::from_f32(
        &vec![f32::NAN; batch * n_keep * 71],
        vec![batch, n_keep, 71],
        &device,
    )
    .unwrap();
    ms2::peak_features(&kept_t, &kept_f_t, &meta_t, &waves, &out).unwrap();
    check_launches(&device).unwrap();
    out.try_to_f32().unwrap()
}

#[test]
fn peak_features_match_twin() {
    let (batch, n_raw, n_keep) = (5, 64, 16);
    let counts = [0usize, 1, 16, 17, 64];
    let (mz, intensity, meta) = random_batch(7, batch, n_raw, &counts, 400_000_000);
    let twin_sel = twin::peak_select(&mz, &intensity, &meta, batch, n_raw, n_keep, 0);
    let got = run_features(&twin_sel.kept, &twin_sel.kept_f, &meta, batch, n_keep);
    let want = twin::peak_features(&twin_sel.kept, &twin_sel.kept_f, &meta, batch, n_keep);
    assert_close(&got, &want, "features");
    // Padding slots are exactly zero; valid pairs sit on the unit circle.
    for b in 0..batch {
        for p in 0..n_keep {
            let base = (b * n_keep + p) * 71;
            if twin_sel.kept[(b * n_keep + p) * 3] == u32::MAX {
                for f in 0..71 {
                    assert!(
                        got[base + f] == 0.0,
                        "padding slot ({b}, {p}) feature {f} is {}",
                        got[base + f]
                    );
                }
            } else {
                for f in 3..7 {
                    assert!(got[base + f].is_finite(), "slot ({b}, {p}) feature {f}");
                }
                for k in 0..16 {
                    for off in [7usize, 39] {
                        let (s, c) = (got[base + off + 2 * k], got[base + off + 1 + 2 * k]);
                        assert!(
                            ((s * s + c * c) - 1.0).abs() <= 1e-4,
                            "slot ({b}, {p}) pair {k} at {off}: sin^2+cos^2 off"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn peak_features_hand_values() {
    // m = 100 Da, c = 200 Da, w_0 = 0.01 Da divides both: phase 0.
    let (batch, n_keep) = (1, 4);
    let kept = vec![
        5,
        100_000_000,
        0,
        u32::MAX,
        0,
        1,
        u32::MAX,
        0,
        2,
        u32::MAX,
        0,
        3,
    ];
    let kept_f = vec![0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let meta = vec![1, 200_000_000, 50, 1, 100, 200, 0, 0];
    let got = run_features(&kept, &kept_f, &meta, batch, n_keep);
    let want = twin::peak_features(&kept, &kept_f, &meta, batch, n_keep);
    assert_close(&got, &want, "hand features");
    assert_close(&got[0..3], &[0.1, 0.1, 0.5], "hand scalars");
    assert_close(&got[7..9], &[0.0, 1.0], "hand m fourier");
    assert_close(&got[39..41], &[0.0, 1.0], "hand v fourier");
}

#[test]
fn select_valid_matches_twin() {
    let device = dev();
    let x = vec![
        1.0,
        2.0,
        f32::NAN,
        1e30,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        f32::NAN,
        9.0,
        10.0,
    ];
    let valid = vec![1.0f32, 0.0, 1.0, 1.0, 0.0, 1.0];
    let x_t = upload_f(&x, vec![2, 3, 2], &device);
    let v_t = upload_f(&valid, vec![2, 3], &device);
    let out = ms2::select_valid(&x_t, &v_t).unwrap();
    check_launches(&device).unwrap();
    let got = out.try_to_f32().unwrap();
    let want = twin::select_valid(&x, &valid, 2);
    assert_close(&got, &want, "select_valid");
    // Invalid rows are exactly +0.0 even where the input is NaN or huge.
    for row in [1usize, 4] {
        for c in 0..2 {
            assert!(
                got[row * 2 + c].to_bits() == 0,
                "invalid element is not +0.0: {}",
                got[row * 2 + c]
            );
        }
    }
}

#[test]
fn bits_to_mask_matches_twin() {
    let device = dev();
    let words = [0u32, 1, 0x8000_0001, u32::MAX, 0x1234_5678];
    for width in [5usize, 18, 32] {
        let bits = upload_ids(&words, vec![words.len()], &device);
        let out: Tensor<R, f32> = ms2::bits_to_mask(&bits, width).unwrap();
        check_launches(&device).unwrap();
        let got = out.try_to_f32().unwrap();
        let want = twin::bits_to_mask(&words, width);
        assert_close(&got, &want, &format!("bits_to_mask width {width}"));
    }
}

#[test]
fn safe_ids_replaces_sentinel() {
    let device = dev();
    let ids = upload_ids(&[0, 5, u32::MAX, u32::MAX - 1, 42], vec![5], &device);
    let out = ms2::safe_ids(&ids, 7).unwrap();
    check_launches(&device).unwrap();
    assert_ids(
        &out.try_to_vec().unwrap(),
        &[0, 5, 7, u32::MAX - 1, 42],
        "safe_ids",
    );
}

#[test]
fn lookup_forward_and_backward_match_twin() {
    let device = dev();
    let table = vec![
        1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];
    let ids = vec![2u32, 0, u32::MAX, 3, 7];
    let table_t = upload_f(&table, vec![4, 3], &device);
    let ids_t = upload_ids(&ids, vec![ids.len()], &device);
    let out = ms2::lookup(&table_t, &ids_t).unwrap();
    check_launches(&device).unwrap();
    let got = out.try_to_f32().unwrap();
    assert_close(&got, &twin::lookup(&table, 3, &ids), "lookup");
    let grad: Vec<f32> = (0..15).map(|i| 0.25 * (i as f32 + 1.0)).collect();
    let grad_t = upload_f(&grad, vec![5, 3], &device);
    let back = ms2::lookup_backward(&grad_t, &ids_t, 4).unwrap();
    check_launches(&device).unwrap();
    let got_back = back.try_to_f32().unwrap();
    assert_close(
        &got_back,
        &twin::lookup_backward(&grad, 3, &ids, 4),
        "lookup_backward",
    );
}

/// More rows than one scan group holds (a device with planes then sums the
/// groups in a second launch), with a count neither the group nor the
/// eight-row round divides, an id out of range and a table row no id names.
#[test]
fn lookup_backward_over_many_rows_matches_twin() {
    let device = dev();
    let (rows, d, table_rows) = (203usize, 5usize, 7usize);
    let ids: Vec<u32> = (0..rows)
        .map(|r| match r % 11 {
            10 => u32::MAX,
            k => ((r * 5 + k) % (table_rows - 1)) as u32,
        })
        .collect();
    let grad: Vec<f32> = (0..rows * d)
        .map(|i| ((i * 37 % 101) as f32 - 50.0) / 64.0)
        .collect();
    let grad_t = upload_f(&grad, vec![rows, d], &device);
    let ids_t = upload_ids(&ids, vec![rows], &device);
    let back = ms2::lookup_backward(&grad_t, &ids_t, table_rows).unwrap();
    check_launches(&device).unwrap();
    assert_eq!(back.shape().dims(), [table_rows, d]);
    assert_close(
        &back.try_to_f32().unwrap(),
        &twin::lookup_backward(&grad, d, &ids, table_rows),
        "lookup_backward over many rows",
    );
}

#[test]
fn lookup_backward_accumulates_repeated_ids() {
    // Repeated ids with distinct upstream gradients accumulate, not overwrite.
    let device = dev();
    let ids = upload_ids(&[1u32, 2, 1, 0, 1, 2], vec![6], &device);
    let grad: Vec<f32> = (0..18).map(|i| 0.5 * (i as f32) + 1.0).collect();
    let grad_t = upload_f(&grad, vec![6, 3], &device);
    let back = ms2::lookup_backward(&grad_t, &ids, 3).unwrap();
    check_launches(&device).unwrap();
    let got = back.try_to_f32().unwrap();
    let want = twin::lookup_backward(&grad, 3, &[1, 2, 1, 0, 1, 2], 3);
    assert_close(&got, &want, "lookup_backward repeated ids");
    // Rows 0 and 6 of the gradient both land in table row 1.
    assert_close(
        &got[3..6],
        &[
            grad[0] + grad[6] + grad[12],
            grad[1] + grad[7] + grad[13],
            grad[2] + grad[8] + grad[14],
        ],
        "accumulation",
    );
}

/// Ids uploaded with a host copy take the bucket path: same sums as the
/// device scan and the twin, including an out-of-range id and a table
/// row no id names.
#[test]
fn lookup_backward_from_host_ids_matches_twin() {
    let device = dev();
    let (rows, d, table_rows) = (203usize, 5usize, 7usize);
    let ids: Vec<u32> = (0..rows)
        .map(|r| match r % 11 {
            10 => u32::MAX,
            k => ((r * 5 + k) % (table_rows - 1)) as u32,
        })
        .collect();
    let grad: Vec<f32> = (0..rows * d)
        .map(|i| ((i * 37 % 101) as f32 - 50.0) / 64.0)
        .collect();
    let grad_t = upload_f(&grad, vec![rows, d], &device);
    let ids_t = IdTensor::from_host(ids.clone(), vec![rows], &device).unwrap();
    assert_eq!(ids_t.host(), Some(&ids[..]));
    let back = ms2::lookup_backward(&grad_t, &ids_t, table_rows).unwrap();
    check_launches(&device).unwrap();
    assert_eq!(back.shape().dims(), [table_rows, d]);
    assert_close(
        &back.try_to_f32().unwrap(),
        &twin::lookup_backward(&grad, d, &ids, table_rows),
        "lookup_backward from host ids",
    );
    // Reshape and clone keep the host copy.
    assert_eq!(ids_t.reshape(vec![1, rows]).unwrap().host(), Some(&ids[..]));
    assert_eq!(ids_t.clone().host(), Some(&ids[..]));
}

#[test]
fn ms2_lookup_and_select_valid_gradients_match_finite_differences() {
    let device = dev();
    // Scalar loss: sum(select_valid(lookup(table, ids) * w, valid)).
    let table = vec![0.5f32, -1.0, 2.0, 0.25, 1.5, -0.5];
    let w = vec![1.0f32, 0.5, -1.0, 2.0, 0.25, -0.5, 1.5, 0.75];
    let ids = upload_ids(&[0u32, 2, 1, u32::MAX], vec![4], &device);
    let valid_t = upload_f(&[1.0f32, 1.0, 0.0, 1.0], vec![4], &device);
    let loss_of = |t: &[f32], w: &[f32]| {
        let table_v = Var::constant(upload_f(t, vec![3, 2], &device));
        let w_v = Var::constant(upload_f(w, vec![4, 2], &device));
        let looked = Var::ms2_lookup(&table_v, &ids).unwrap();
        let scaled = looked.mul(&w_v).unwrap();
        scaled.ms2_select_valid(&valid_t).unwrap().sum().unwrap()
    };
    let table_v = Var::traced(upload_f(&table, vec![3, 2], &device));
    let w_v = Var::traced(upload_f(&w, vec![4, 2], &device));
    let looked = Var::ms2_lookup(&table_v, &ids).unwrap();
    let scaled = looked.mul(&w_v).unwrap();
    let loss = scaled.ms2_select_valid(&valid_t).unwrap().sum().unwrap();
    check_launches(&device).unwrap();
    let grads = loss.backward_retain().unwrap();
    let grad_t = grads
        .node(table_v.node().unwrap())
        .expect("gradient reaches the table")
        .to_f32();
    let grad_w = grads
        .node(w_v.node().unwrap())
        .expect("gradient reaches w")
        .to_f32();
    let central = |base: &[f32], i: usize, is_table: bool| {
        let mut up = base.to_vec();
        up[i] += 1e-2;
        let mut down = base.to_vec();
        down[i] -= 1e-2;
        let (fu, fd) = if is_table {
            (loss_of(&up, &w).to_f32()[0], loss_of(&down, &w).to_f32()[0])
        } else {
            (
                loss_of(&table, &up).to_f32()[0],
                loss_of(&table, &down).to_f32()[0],
            )
        };
        (fu - fd) / 2e-2
    };
    for (i, &a) in grad_t.iter().enumerate() {
        let n = central(&table, i, true);
        assert!(
            (a - n).abs() <= 2e-2 * n.abs() + 1e-3,
            "table grad[{i}]: analytic={a} numeric={n}"
        );
    }
    for (i, &a) in grad_w.iter().enumerate() {
        let n = central(&w, i, false);
        assert!(
            (a - n).abs() <= 2e-2 * n.abs() + 1e-3,
            "w grad[{i}]: analytic={a} numeric={n}"
        );
    }
}
