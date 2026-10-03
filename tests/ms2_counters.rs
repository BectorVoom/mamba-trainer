//! P2-A test: the transfer counters in `mamba3::backend`.
//!
//! This is the only test in its binary on purpose. It reads the process-wide
//! upload, allocation and read counters, and any test running beside it would
//! add to them — a false failure that says nothing about the counters. Cargo
//! gives each integration test file its own process, so keeping this one alone
//! is what makes the numbers mean what they say.
//!
//! Covers the four transfer counters (`runtime_read_count`, `download_bytes`,
//! `upload_bytes`, `allocation_calls`), the `read_count` boundary and
//! `memory_snapshot`.

#![cfg(feature = "backend")]

use mamba3::backend::{
    Device, allocation_calls, download_bytes, memory_snapshot, read_count, reset_read_count,
    reset_transfer_counters, runtime_read_count, upload_bytes,
};
use mamba3::backends::Auto;
use mamba3::tensor::Tensor;
use mamba3::tensor::ops::index::{IdTensor, read_all};

type R = Auto;

#[test]
fn transfer_counters_track_uploads_allocations_and_reads() {
    let device = Device::<R>::default();
    reset_transfer_counters();
    reset_read_count();

    // Uploading a 100-element f32 tensor moves exactly 400 bytes in one buffer.
    let before_upload = upload_bytes();
    let before_alloc = allocation_calls();
    let tensor = Tensor::<R, f32>::from_data(&vec![1.0; 100], vec![100], &device).unwrap();
    assert_eq!(upload_bytes() - before_upload, 400);
    assert_eq!(allocation_calls() - before_alloc, 1);

    // Ten u32 ids move exactly 40 bytes.
    let before_upload = upload_bytes();
    let _ids = IdTensor::<R>::from_slice(&vec![0; 10], vec![10], &device).unwrap();
    assert_eq!(upload_bytes() - before_upload, 40);

    // An empty buffer allocates without uploading.
    let before_upload = upload_bytes();
    let before_alloc = allocation_calls();
    let _empty = Tensor::<R, f32>::empty(vec![7], &device);
    assert_eq!(allocation_calls() - before_alloc, 1);
    assert_eq!(upload_bytes() - before_upload, 0);

    // One host read is one runtime read, one step-sync read, and the bytes.
    let before_runtime = runtime_read_count();
    let before_read = read_count();
    let before_download = download_bytes();
    let _ = tensor.to_f32();
    assert_eq!(runtime_read_count() - before_runtime, 1);
    assert_eq!(read_count() - before_read, 1);
    assert!(download_bytes() - before_download >= 400);

    // Two tensors read together on one stream pay one runtime read.
    let other = Tensor::<R, f32>::from_data(&vec![2.0; 50], vec![50], &device).unwrap();
    let before_runtime = runtime_read_count();
    let (_ids, floats) = read_all(&[], &[&tensor, &other]).unwrap();
    assert_eq!(floats.len(), 2);
    assert_eq!(runtime_read_count() - before_runtime, 1);

    // Resetting zeroes the four transfer counters and leaves read_count alone.
    let reads = read_count();
    assert!(reads > 0);
    reset_transfer_counters();
    assert_eq!(runtime_read_count(), 0);
    assert_eq!(download_bytes(), 0);
    assert_eq!(upload_bytes(), 0);
    assert_eq!(allocation_calls(), 0);
    assert_eq!(read_count(), reads);

    // A runtime that reports memory has a reserved high-water mark above use.
    match memory_snapshot(&device) {
        Some(snapshot) => {
            assert!(snapshot.bytes_reserved >= snapshot.bytes_in_use);
            println!("memory_snapshot reports: {snapshot:?}");
        }
        None => println!("memory_snapshot: the runtime does not report memory"),
    }
}
