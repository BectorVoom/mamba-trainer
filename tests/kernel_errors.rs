//! A kernel the device cannot compile is an error, never a buffer of zeros.
//!
//! CubeCL launches return nothing, and on wgpu a shader the validator rejects is
//! simply not dispatched: its output keeps whatever the allocation held. Before
//! `backend::check_launches` existed, that is how masking was silently ignored on
//! WGSL and how every continuous distribution returned zeros there. These tests
//! pin the contract with a kernel that is invalid on WGSL by construction — it
//! writes an infinite float literal — and valid on runtimes that accept one.

#![cfg(feature = "backend")]

use cubecl::prelude::*;
use mamba3::backend::{Device, check_launches};
use mamba3::backends::Auto;
use mamba3::error::Error;
use mamba3::tensor::Tensor;

type R = Auto;

/// Writes a non-finite literal, which WGSL cannot spell (`f32(inf)` is not a
/// WGSL expression) and every other runtime here compiles.
#[cube(launch_unchecked)]
fn infinity_literal_kernel(out: &mut Array<f32>) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = f32::INFINITY;
    }
}

fn launch_invalid(device: &Device<R>, n: usize) -> Tensor<R, f32> {
    let out = Tensor::<R, f32>::zeros(vec![n], device);
    unsafe {
        infinity_literal_kernel::launch_unchecked::<R>(
            device.client(),
            CubeCount::Static(n as u32, 1, 1),
            CubeDim::new_1d(1),
            out.arg(),
        );
    }
    out
}

/// Whether this runtime rejects non-finite float literals in kernel source.
fn rejects_non_finite_literals(device: &Device<R>) -> bool {
    device.name().contains("wgsl")
}

/// The launch is reported as an error that names the kernel, or it ran — and
/// then the buffer holds what the kernel wrote. It never holds the zeros it
/// was allocated with while reporting success.
#[test]
fn a_kernel_that_fails_to_compile_is_reported_at_the_next_read() {
    let device = Device::<R>::default();
    let out = launch_invalid(&device, 8);
    match out.try_to_f32() {
        Ok(values) => {
            assert!(
                !rejects_non_finite_literals(&device),
                "{} cannot compile an infinite literal, yet the read succeeded with {values:?}",
                device.name()
            );
            assert!(values.iter().all(|v| *v == f32::INFINITY), "{values:?}");
            println!(
                "{} compiles non-finite literals; checked that the kernel ran instead",
                device.name()
            );
        }
        Err(err) => {
            assert!(matches!(err, Error::Backend(_)), "{err:?}");
            let message = err.to_string();
            assert!(
                message.contains("infinity_literal_kernel"),
                "the error should name the kernel: {message}"
            );
            assert!(rejects_non_finite_literals(&device), "{message}");
            println!("reported: {message}");
        }
    }

    // The failure is reported once, not forever: the stream is usable again.
    let fine = Tensor::<R, f32>::from_f32(&[1.0, 2.0], vec![2], &device).unwrap();
    assert_eq!(fine.try_to_f32().unwrap(), vec![1.0, 2.0]);
}

#[test]
fn the_infallible_read_panics_instead_of_returning_zeros() {
    let device = Device::<R>::default();
    if !rejects_non_finite_literals(&device) {
        println!(
            "skipped: {} compiles the literal, so there is no failure to observe",
            device.name()
        );
        return;
    }
    let out = launch_invalid(&device, 4);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| out.to_f32()));
    let payload = result.expect_err("reading a failed kernel's output must not succeed");
    let message = payload
        .downcast_ref::<String>()
        .cloned()
        .unwrap_or_default();
    assert!(message.contains("infinity_literal_kernel"), "{message}");
}

#[test]
fn synchronisation_reports_the_failure_rather_than_discarding_it() {
    let device = Device::<R>::default();
    if !rejects_non_finite_literals(&device) {
        println!(
            "skipped: {} compiles the literal, so there is no failure to observe",
            device.name()
        );
        return;
    }
    let _out = launch_invalid(&device, 4);
    let err = device
        .try_synchronize()
        .expect_err("a sync after a failed launch must fail");
    assert!(err.to_string().contains("infinity_literal_kernel"), "{err}");

    let _out = launch_invalid(&device, 4);
    let err = check_launches(&device).expect_err("the explicit check must fail too");
    assert!(err.to_string().contains("infinity_literal_kernel"), "{err}");
    check_launches(&device).expect("and pass once the failure has been reported");
}
