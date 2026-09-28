//! K9: `sum_dim` on the model's shapes is one launch; the long-axis case
//! still splits.
//!
//! Alone in its binary: `launch_count` is process-wide, so a test running
//! beside this one in the same binary adds its own launches between the reset
//! and the read (it failed that way inside `tests/tensor.rs`, reading 4 for 1).

#![cfg(feature = "backend")]

use mamba3::backend::Device;
use mamba3::backends::Auto;
use mamba3::tensor::Tensor;
use mamba3::tensor::ops::reduce;

type R = Auto;

fn dev() -> Device<R> {
    Device::<R>::default()
}

#[test]
fn sum_dim_single_pass_pins() {
    use mamba3::backend::{launch_count, reset_launch_count};
    let device = dev();
    let one_launch = |shape: Vec<usize>, axis: usize| {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|i| (i % 97) as f32 * 0.01).collect();
        let input = Tensor::<R, f32>::from_f32(&data, shape, &device).unwrap();
        let _ = reduce::sum_dim(&input, axis).unwrap();
        reset_launch_count();
        let out = reduce::sum_dim(&input, axis).unwrap();
        let launches = launch_count();
        assert!(!out.to_f32().is_empty());
        launches
    };
    assert_eq!(one_launch(vec![15360, 128], 1), 1);
    assert_eq!(one_launch(vec![128, 120, 128], 1), 1);
    assert_eq!(one_launch(vec![7680, 101], 1), 1);
    assert_eq!(one_launch(vec![3840, 128], 1), 1);
    // Long axis, few outputs: still two passes.
    assert!(one_launch(vec![2048, 1024], 1) >= 2);
}
