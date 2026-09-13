//! What a broadcast operation's metadata upload costs.
//!
//! Every broadcasting binary op and every strided copy uploads a small
//! `[shape, strides]` buffer with `create_from_slice` before it launches — about a
//! hundred bytes, but once per operation, and a rollout step performs eleven of
//! them. `PLAN.md` D.2 proposes caching those buffers, and says to skip it if the
//! upload turns out to be noise. This is the measurement that decides.
//!
//! The comparison is an upload against the launch it precedes, interleaved and
//! minimum-of-N for the same reason `bench_split` is: nothing smaller than a large
//! win survives a single wall-clock sample on this hardware.
//!
//! ```text
//! cargo run --release --features wgpu --example bench_meta_upload
//! ```

use std::time::{Duration, Instant};

use cubecl::prelude::{CubeElement, Runtime};
use mamba3::prelude::*;
use mamba3::tensor::ops::elemwise;

type R = mamba3::backends::Auto;

const INNER: usize = 500;
const SAMPLES: usize = 30;

fn main() -> Result<()> {
    let device = Device::<R>::default();
    let client = device.client();
    println!("backend: {}\n", device.name());

    // The shape of a rollout step's metadata: rank 5 packed as shape plus two
    // stride vectors, which is what `pack_broadcast_meta` produces for the
    // `[batch, 1, heads, 1, state]` bias broadcasts in the mixer's projection.
    let meta: Vec<u32> = (0..15).collect();

    // A broadcasting add at the same shape those biases use, for scale: this is one
    // whole operation, upload included.
    let big = Tensor::<R, f32>::zeros(vec![32, 1, 4, 8, 8], &device);
    let bias = Tensor::<R, f32>::zeros(vec![1, 1, 4, 1, 8], &device);

    let mut uncached = Duration::MAX;
    let mut cached = Duration::MAX;
    let mut upload = Duration::MAX;
    for _ in 0..SAMPLES {
        let started = Instant::now();
        for _ in 0..INNER {
            std::hint::black_box(client.create_from_slice(u32::as_bytes(&meta)));
        }
        device.synchronize();
        upload = upload.min(started.elapsed());

        // Clearing the cache before each operation is what the code did before it
        // had one: the metadata is uploaded again every time. The clear itself is a
        // one-entry hash map drop, which is nothing beside the upload it forces.
        let started = Instant::now();
        for _ in 0..INNER {
            mamba3::backend::clear_meta_cache();
            std::hint::black_box(elemwise::add(&big, &bias)?);
        }
        device.synchronize();
        uncached = uncached.min(started.elapsed());

        let started = Instant::now();
        for _ in 0..INNER {
            std::hint::black_box(elemwise::add(&big, &bias)?);
        }
        device.synchronize();
        cached = cached.min(started.elapsed());
    }

    let each = |d: Duration| d.as_secs_f64() * 1e6 / INNER as f64;
    println!("{:<36} {:>9.2} us", "create_from_slice (metadata alone)", each(upload));
    println!("{:<36} {:>9.2} us", "broadcasting add, uploading again", each(uncached));
    println!("{:<36} {:>9.2} us", "broadcasting add, cached metadata", each(cached));
    println!(
        "\ncaching the metadata takes {:.0}% off the operation ({:.1}x)",
        100.0 * (each(uncached) - each(cached)) / each(uncached),
        each(uncached) / each(cached),
    );
    Ok(())
}
