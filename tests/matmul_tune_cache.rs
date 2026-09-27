//! The matmul tuner's on-disk memory: a shape tuned in this process is written
//! to the cache file, one line keyed by device and shape. Alone in its binary
//! because it points the cache at a private directory through the environment.
//! On the CPU runtime nothing is tuned (the simple kernel always runs), so there
//! is nothing to record and the test only checks that nothing broke.

#![cfg(feature = "backend")]

use mamba3::backend::Device;
use mamba3::backends::Auto;
use mamba3::tensor::Tensor;
use mamba3::tensor::ops::matmul::matmul;

type R = Auto;

#[test]
fn tuned_shapes_are_recorded() {
    let dir = std::env::temp_dir().join(format!("mamba3-tune-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    // SAFETY: set before any other thread of this binary reads the environment;
    // this is the binary's only test.
    unsafe { std::env::set_var("MAMBA3_TUNE_CACHE_DIR", &dir) };

    let dev = Device::<R>::default();
    let (m, k, n) = (37usize, 24usize, 20usize);
    let a = Tensor::<R, f32>::from_f32(&vec![0.5; m * k], vec![m, k], &dev).unwrap();
    let b = Tensor::<R, f32>::from_f32(&vec![0.25; k * n], vec![k, n], &dev).unwrap();
    let out = matmul(&a, &b).unwrap().to_f32();
    assert!(out.iter().all(|v| (v - 3.0).abs() < 1e-4), "0.5 * 0.25 * 24 = 3");

    let gpu = dev.client().properties().hardware.plane_size_max > 1;
    let recorded = std::fs::read_dir(&dir)
        .ok()
        .into_iter()
        .flatten()
        .filter_map(|e| std::fs::read_to_string(e.ok()?.path()).ok())
        .collect::<String>();
    if gpu {
        assert!(
            recorded.contains(&format!("|1,{m},{n},{k},0,0,f32,f32|")),
            "no line for the tuned shape in {recorded:?}"
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}
