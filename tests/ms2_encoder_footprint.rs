#![cfg(feature = "backend")]

//! Footprint of upload + encode: warmed calls do no reads, launch counts are
//! stable, and the upload moves exactly the five tensors.

use mamba3::backend::{
    Device, check_launches, launch_count, reset_launch_count, reset_read_count,
    reset_transfer_counters, runtime_read_count, upload_bytes,
};
use mamba3::backends::Auto;
use mamba3::models::ms2::batch::DeviceSpectra;
use mamba3::models::ms2::contract::{Control, ModelConfig, SCHEMA_VERSION, SPECTRUM_SCHEMA_VERSION, SpectrumBatch};
use mamba3::models::ms2::encoder::Ms2Encoder;
use mamba3::tensor::ops::ms2;
use mamba3::tensor::ops::random::Rng;

type R = Auto;

#[test]
fn ms2_encoder_footprint() {
    let device = Device::<R>::default();
    let mut model = ModelConfig::v0();
    model.d_model = 16;
    model.n_peaks = 16;
    model.encoder_blocks = 2;
    model.encoder.d_model = 16;
    model.decoder.d_model = 16;
    model.encoder.n_heads = 2;
    model.encoder.head_dim = 8;
    model.encoder.d_state = 8;
    model.encoder.n_groups = 2;
    model.decoder.n_heads = 2;
    model.decoder.head_dim = 8;
    model.decoder.d_state = 8;
    model.decoder.n_groups = 2;
    let mut rng = Rng::seeded(7);
    let encoder = Ms2Encoder::<R, f32>::init(&model, &device, &mut rng).unwrap();
    let (batch_n, n_raw) = (2usize, 64usize);
    let batch = SpectrumBatch {
        schema_version: SPECTRUM_SCHEMA_VERSION,
        n_raw: n_raw as u32,
        spectrum_id: vec![1, 2],
        raw_peak_count: vec![10, 8],
        peak_count: vec![10, 8],
        peak_id: {
            let mut v = vec![u32::MAX; batch_n * n_raw];
            for b in 0..batch_n {
                for i in 0..10 {
                    v[b * n_raw + i] = i as u32;
                }
            }
            v
        },
        mz_udalton: {
            let mut v = vec![0u32; batch_n * n_raw];
            for b in 0..batch_n {
                for i in 0..10 {
                    v[b * n_raw + i] = 100_000_000 + i as u32 * 10_000_000;
                }
            }
            v
        },
        intensity: {
            let mut v = vec![0.0f32; batch_n * n_raw];
            for b in 0..batch_n {
                for i in 0..10 {
                    v[b * n_raw + i] = 1.0;
                }
            }
            v
        },
        intensity_scale: 0,
        mz_uncertainty_udalton: vec![50, 50],
        precursor_mz_udalton: vec![400_000_000, 410_000_000],
        precursor_uncertainty_udalton: vec![50, 50],
        adduct: vec![1, 1],
        polarity: vec![1, 1],
        collision_energy_ev: vec![30.0, 30.0],
        collision_energy_known: vec![1, 1],
        energy_count: vec![1, 1],
        fragment_tolerance_ppm_tenths: vec![0, 0],
        precursor_tolerance_ppm_tenths: vec![0, 0],
        instrument_class: vec![0, 0],
    };
    let warmed = |encoder: &Ms2Encoder<R, f32>| {
        let spectra = DeviceSpectra::<R, f32>::upload(&batch, &device).unwrap();
        let peaks = ms2::PeakBuffers::<R, f32>::new(batch_n, n_raw, 16, &device);
        let _ = encoder.encode(&spectra, &peaks, Control::None).unwrap();
        check_launches(&device).unwrap();
    };
    warmed(&encoder);
    warmed(&encoder);
    // Previous warmed call's launch cost.
    reset_launch_count();
    reset_read_count();
    reset_transfer_counters();
    let before = launch_count();
    let spectra = DeviceSpectra::<R, f32>::upload(&batch, &device).unwrap();
    let peaks = ms2::PeakBuffers::<R, f32>::new(batch_n, n_raw, 16, &device);
    let _ = encoder.encode(&spectra, &peaks, Control::None).unwrap();
    check_launches(&device).unwrap();
    let prev_launches = launch_count() - before;
    let prev_reads = runtime_read_count();
    // One more warmed call: same launch cost, no reads, five uploads.
    reset_launch_count();
    reset_read_count();
    reset_transfer_counters();
    let before = launch_count();
    let spectra = DeviceSpectra::<R, f32>::upload(&batch, &device).unwrap();
    // The upload alone moves exactly the five tensors.
    let upload_only = upload_bytes();
    let want_upload = (batch_n * n_raw * 4
        + batch_n * n_raw * 4
        + batch_n * 8 * 4
        + batch_n * 4 * 4
        + batch_n * 2 * 4) as u64;
    assert_eq!(
        upload_only, want_upload,
        "upload moves exactly the five tensors"
    );
    let peaks = ms2::PeakBuffers::<R, f32>::new(batch_n, n_raw, 16, &device);
    let _ = encoder.encode(&spectra, &peaks, Control::None).unwrap();
    check_launches(&device).unwrap();
    let launches = launch_count() - before;
    println!("encode launches per call: {launches} (previous {prev_launches})");
    assert_eq!(launches, prev_launches, "launch count stable");
    assert_eq!(runtime_read_count(), 0, "warmed encode does no reads");
    assert_eq!(prev_reads, 0, "previous warmed call did no reads");
    // A warmed encode performs no upload: the wavelength table is resident
    // in the encoder's constants, and everything else allocates uninitialised.
    reset_transfer_counters();
    let peaks = ms2::PeakBuffers::<R, f32>::new(batch_n, n_raw, 16, &device);
    let _ = encoder.encode(&spectra, &peaks, Control::None).unwrap();
    check_launches(&device).unwrap();
    // The fused scan (GPU-like devices) uploads nothing. The CPU runtime runs the
    // chunked scan, whose `Tensor::strict_causal_mask` uploads a small constant
    // on every call (library behaviour in `ssd_chunked`, not MS2's): bound it.
    let uploaded = upload_bytes();
    println!("warmed encode upload bytes: {uploaded}");
    if device.client().properties().hardware.plane_size_max > 1 {
        assert_eq!(uploaded, 0, "warmed encode does no upload");
    } else {
        assert!(uploaded <= 1024, "warmed encode uploads only the chunked scan's mask: {uploaded} bytes");
    }
    assert_eq!(runtime_read_count(), 0, "warmed encode does no reads");
}
