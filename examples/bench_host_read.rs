//! What a rollout step's host edges cost: the reads out and the uploads in.
//!
//! `Rollout.step` in the Python bindings reads three `[envs]` buffers back —
//! actions, values, log-probabilities — and uploads one or three buffers first.
//! On wgpu a read is a queue submit, a staging map and a blocking poll for the
//! map's callback, so three reads are three waits. `ComputeClient::read` takes a
//! `Vec` of handles and services them under one submit and one wait (see the
//! CubeCL manual, "Launch Overhead & Host–Device Transfers" §3); this measures
//! whether that is worth having, and the same for one `create_tensors` against
//! separate `create_from_slice` uploads.
//!
//! Interleaved and minimum-of-N, as in `bench_meta_upload`: run-to-run noise on
//! this hardware hides anything smaller than a large win.
//!
//! ```text
//! cargo run --release --features wgpu --example bench_host_read
//! ```

use std::time::{Duration, Instant};

use cubecl::bytes::Bytes;
use cubecl::prelude::CubeElement;
use cubecl::server::{MemoryLayoutDescriptor, MemoryLayoutStrategy};
use mamba3::prelude::*;

type R = mamba3::backends::Auto;

const ENVS: usize = 32;
const OBS_DIM: usize = 6;
const ACTIONS: usize = 4;
const INNER: usize = 200;
const SAMPLES: usize = 15;

fn each(d: Duration) -> f64 {
    d.as_secs_f64() * 1e6 / INNER as f64
}

fn main() -> Result<()> {
    let device = Device::<R>::default();
    let client = device.client();
    println!("backend: {}\n", device.name());

    let row: Vec<f32> = (0..ENVS).map(|i| i as f32).collect();
    let obs: Vec<f32> = (0..ENVS * OBS_DIM).map(|i| i as f32).collect();
    let mask = vec![1.0f32; ENVS * ACTIONS];
    let handles: Vec<_> = (0..3)
        .map(|_| client.create_from_slice(f32::as_bytes(&row)))
        .collect();
    device.synchronize();

    let mut separate = Duration::MAX;
    let mut batched = Duration::MAX;
    let mut single = Duration::MAX;
    let mut upload_separate = Duration::MAX;
    let mut upload_batched = Duration::MAX;
    for _ in 0..SAMPLES {
        let started = Instant::now();
        for _ in 0..INNER {
            for handle in &handles {
                std::hint::black_box(client.read_one_unchecked(handle.clone()));
            }
        }
        separate = separate.min(started.elapsed());

        let started = Instant::now();
        for _ in 0..INNER {
            std::hint::black_box(client.read(handles.clone()));
        }
        batched = batched.min(started.elapsed());

        let started = Instant::now();
        for _ in 0..INNER {
            std::hint::black_box(client.read_one_unchecked(handles[0].clone()));
        }
        single = single.min(started.elapsed());

        let started = Instant::now();
        for _ in 0..INNER {
            std::hint::black_box(client.create_from_slice(f32::as_bytes(&obs)));
            std::hint::black_box(client.create_from_slice(f32::as_bytes(&row)));
            std::hint::black_box(client.create_from_slice(f32::as_bytes(&mask)));
        }
        device.synchronize();
        upload_separate = upload_separate.min(started.elapsed());

        let started = Instant::now();
        for _ in 0..INNER {
            let layout = |len: usize| {
                MemoryLayoutDescriptor::new(MemoryLayoutStrategy::Contiguous, [len * 4].into(), 1)
            };
            std::hint::black_box(client.create_tensors(vec![
                (layout(obs.len()), Bytes::from_elems(obs.clone())),
                (layout(row.len()), Bytes::from_elems(row.clone())),
                (layout(mask.len()), Bytes::from_elems(mask.clone())),
            ]));
        }
        device.synchronize();
        upload_batched = upload_batched.min(started.elapsed());
    }

    println!("{:<44} {:>9.1} us", "one read", each(single));
    println!(
        "{:<44} {:>9.1} us",
        "three reads, one at a time",
        each(separate)
    );
    println!(
        "{:<44} {:>9.1} us",
        "three reads, one client.read",
        each(batched)
    );
    println!(
        "{:<44} {:>9.1} us",
        "three uploads, create_from_slice each",
        each(upload_separate)
    );
    println!(
        "{:<44} {:>9.1} us",
        "three uploads, one create_tensors",
        each(upload_batched)
    );
    Ok(())
}
