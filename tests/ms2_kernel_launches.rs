//! P3-A test: exact launch counts and the zero-read property of the MS2
//! kernels and adapters.
//!
//! This is the only test in its binary on purpose: the launch and read
//! counters are process-global, and any test running beside it would add to
//! them.

#![cfg(feature = "backend")]

use mamba3::autograd::Var;
use mamba3::backend::{
    Device, check_launches, launch_count, read_count, reset_launch_count, reset_read_count,
    reset_transfer_counters, runtime_read_count,
};
use mamba3::backends::Auto;
use mamba3::tensor::Tensor;
use mamba3::tensor::ops::index::IdTensor;
use mamba3::tensor::ops::ms2;

type R = Auto;

#[test]
fn ms2_kernel_launch_and_read_counts() {
    let device = Device::<R>::default();
    let (batch, n_raw, n_keep) = (2usize, 8usize, 4usize);
    let mz_host: Vec<u32> = (0..batch * n_raw)
        .map(|i| 100_000_000 + i as u32 * 1000)
        .collect();
    let int_host: Vec<f32> = (0..batch * n_raw).map(|i| 0.1 + i as f32 * 0.01).collect();
    let mut meta_host = vec![0u32; batch * 8];
    for b in 0..batch {
        meta_host[b * 8] = n_raw as u32;
        meta_host[b * 8 + 1] = 500_000_000;
    }
    let mz = IdTensor::from_slice(&mz_host, vec![batch, n_raw], &device).unwrap();
    let int_t = Tensor::from_f32(&int_host, vec![batch, n_raw], &device).unwrap();
    let meta = IdTensor::from_slice(&meta_host, vec![batch, 8], &device).unwrap();
    let bufs = ms2::PeakBuffers::poisoned(batch, n_raw, n_keep, &device).unwrap();
    let feats = Tensor::<R, f32>::empty(vec![batch, n_keep, ms2::PEAK_FEATURES], &device);
    let x = Tensor::<R, f32>::from_f32(&vec![1.0f32; 12], vec![2, 3, 2], &device).unwrap();
    let valid = Tensor::<R, f32>::from_f32(&vec![1.0f32; 6], vec![2, 3], &device).unwrap();
    let bits = IdTensor::from_slice(&[3u32, 5], vec![2], &device).unwrap();
    let table = Tensor::<R, f32>::from_f32(&vec![1.0f32; 12], vec![4, 3], &device).unwrap();
    let ids = IdTensor::from_slice(&[0u32, 1, 2, 3], vec![4], &device).unwrap();
    let grad = Tensor::<R, f32>::from_f32(&vec![1.0f32; 12], vec![4, 3], &device).unwrap();

    // Warm each operation once: a warmed call performs no setup work.
    let waves = ms2::Ms2Constants::new(&device);
    ms2::peak_select(&mz, &int_t, &meta, 0, &bufs).unwrap();
    ms2::peak_features(&bufs.kept, &bufs.kept_f, &meta, &waves, &feats).unwrap();
    let _: Tensor<R, f32> = ms2::select_valid(&x, &valid).unwrap();
    let _: Tensor<R, f32> = ms2::bits_to_mask(&bits, 5).unwrap();
    let _: Tensor<R, f32> = ms2::lookup(&table, &ids).unwrap();
    let _: Tensor<R, f32> = ms2::lookup_backward(&grad, &ids, 4).unwrap();
    let _: IdTensor<R> = ms2::safe_ids(&ids, 0).unwrap();
    let warm_table = Var::traced(table.clone());
    let warm_loss = Var::ms2_lookup(&warm_table, &ids).unwrap().sum().unwrap();
    warm_loss.backward().unwrap();
    check_launches(&device).unwrap();

    reset_launch_count();
    reset_read_count();
    reset_transfer_counters();

    let delta = |before: usize| launch_count() - before;
    let mut before = launch_count();
    ms2::peak_select(&mz, &int_t, &meta, 0, &bufs).unwrap();
    assert_eq!(delta(before), 5, "peak_select launches");
    before = launch_count();
    ms2::peak_features(&bufs.kept, &bufs.kept_f, &meta, &waves, &feats).unwrap();
    assert_eq!(delta(before), 1, "peak_features launches");
    before = launch_count();
    let _: Tensor<R, f32> = ms2::select_valid(&x, &valid).unwrap();
    assert_eq!(delta(before), 1, "select_valid launches");
    before = launch_count();
    let _: Tensor<R, f32> = ms2::bits_to_mask(&bits, 5).unwrap();
    assert_eq!(delta(before), 1, "bits_to_mask launches");
    before = launch_count();
    let _: Tensor<R, f32> = ms2::lookup(&table, &ids).unwrap();
    assert_eq!(delta(before), 1, "lookup launches");
    before = launch_count();
    let _: Tensor<R, f32> = ms2::lookup_backward(&grad, &ids, 4).unwrap();
    assert_eq!(delta(before), 1, "lookup_backward launches");
    before = launch_count();
    let _: IdTensor<R> = ms2::safe_ids(&ids, 0).unwrap();
    assert_eq!(delta(before), 1, "safe_ids launches");

    // No device-to-host read anywhere above, nor in a tracked lookup
    // forward and backward.
    assert_eq!(read_count(), 0, "step-sync reads");
    assert_eq!(runtime_read_count(), 0, "total reads");
    let tracked = Var::traced(table.clone());
    let loss = Var::ms2_lookup(&tracked, &ids).unwrap().sum().unwrap();
    loss.backward().unwrap();
    check_launches(&device).unwrap();
    assert_eq!(read_count(), 0, "step-sync reads after Var backward");
    assert_eq!(runtime_read_count(), 0, "total reads after Var backward");
}
