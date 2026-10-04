#![cfg(feature = "backend")]

use mamba3::autograd::Var;
use mamba3::backend::{Device, check_launches};
use mamba3::backends::Auto;
use mamba3::models::ms2::batch::{DeviceSpectra, rotate_peaks};
use mamba3::models::ms2::contract::{Control, ModelConfig, SCHEMA_VERSION, SPECTRUM_SCHEMA_VERSION, SpectrumBatch};
use mamba3::models::ms2::encoder::Ms2Encoder;
use mamba3::models::ms2::twin;
use mamba3::nn::Module;
use mamba3::tensor::Tensor;
use mamba3::tensor::ops::index::IdTensor;
use mamba3::tensor::ops::ms2;
use mamba3::tensor::ops::random::Rng;

type R = Auto;

fn dev() -> Device<R> {
    Device::<R>::default()
}

fn assert_close(actual: &[f32], expected: &[f32], tol: f32, what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        let ok = if e.abs() <= 1.0 {
            (a - e).abs() <= tol
        } else {
            (a - e).abs() <= tol * e.abs()
        };
        assert!(ok, "{what}: index {i} got {a}, want {e}");
    }
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn small_config(d: u32, n_peaks: u32, blocks: u32) -> ModelConfig {
    let mut m = ModelConfig::v0();
    m.d_model = d;
    m.n_peaks = n_peaks;
    m.encoder_blocks = blocks;
    m.encoder.d_model = d as usize;
    m.decoder.d_model = d as usize;
    if d == 16 {
        m.encoder.n_heads = 2;
        m.encoder.head_dim = 8;
        m.encoder.d_state = 8;
        m.encoder.n_groups = 2;
        m.decoder.n_heads = 2;
        m.decoder.head_dim = 8;
        m.decoder.d_state = 8;
        m.decoder.n_groups = 2;
    } else if d == 8 {
        m.encoder.n_heads = 2;
        m.encoder.head_dim = 4;
        m.encoder.d_state = 4;
        m.encoder.n_groups = 2;
        m.decoder.n_heads = 2;
        m.decoder.head_dim = 4;
        m.decoder.d_state = 4;
        m.decoder.n_groups = 2;
    }
    m
}

fn make_batch(
    spectrum_ids: &[u64],
    n_raw: usize,
    peak_counts: &[u32],
    precursor_base: u32,
    seed: u64,
) -> SpectrumBatch {
    let b = spectrum_ids.len();
    let mut rng = Rng::seeded(seed);
    let mut peak_id = vec![u32::MAX; b * n_raw];
    let mut mz = vec![0u32; b * n_raw];
    let mut intensity = vec![0.0f32; b * n_raw];
    let mut peak_count = vec![0u32; b];
    let mut raw_peak_count = vec![0u32; b];
    for (bi, &count) in peak_counts.iter().enumerate() {
        let count = count as usize;
        peak_count[bi] = count as u32;
        raw_peak_count[bi] = count as u32;
        let precursor = precursor_base + bi as u32 * 10_000_000;
        for i in 0..count {
            peak_id[bi * n_raw + i] = i as u32;
            let f = rng.uniform_vec(1, 60_000_000.0, (precursor - 5_000_000) as f32)[0] as u32;
            mz[bi * n_raw + i] = f.max(50_000_001);
            let u = rng.uniform_vec(1, 0.0, 1.0)[0];
            intensity[bi * n_raw + i] = 0.5 + 2.0 * u;
        }
    }
    let mut precursor = vec![0u32; b];
    for (bi, _) in spectrum_ids.iter().enumerate() {
        precursor[bi] = precursor_base + bi as u32 * 10_000_000;
    }
    SpectrumBatch {
        schema_version: SPECTRUM_SCHEMA_VERSION,
        n_raw: n_raw as u32,
        spectrum_id: spectrum_ids.to_vec(),
        raw_peak_count,
        peak_count,
        peak_id,
        mz_udalton: mz,
        intensity,
        intensity_scale: 0,
        mz_uncertainty_udalton: vec![50; b],
        precursor_mz_udalton: precursor,
        precursor_uncertainty_udalton: vec![50; b],
        adduct: vec![1; b],
        polarity: vec![1; b],
        collision_energy_ev: vec![30.0; b],
        collision_energy_known: vec![1; b],
        energy_count: vec![1; b],
        fragment_tolerance_ppm_tenths: vec![0; b],
        precursor_tolerance_ppm_tenths: vec![0; b],
        instrument_class: vec![0; b],
    }
}

#[test]
fn meta_features_and_kept_column_equal_twins() {
    let device = dev();
    let waves = ms2::Ms2Constants::new(&device);
    let (batch, n_raw, n_keep) = (3usize, 16usize, 8usize);
    let mut meta = vec![0u32; batch * 8];
    let mut energy = vec![0.0f32; batch * 2];
    for b in 0..batch {
        meta[b * 8] = n_raw as u32;
        meta[b * 8 + 1] = 300_000_000 + b as u32 * 50_000_000;
        meta[b * 8 + 2] = 50;
        meta[b * 8 + 3] = 1;
        meta[b * 8 + 4] = 100;
        meta[b * 8 + 5] = 200;
        meta[b * 8 + 6] = b as u32;
        meta[b * 8 + 7] = 0;
        energy[b * 2] = if b == 1 { 0.0 } else { 30.0 + b as f32 * 500.0 };
        energy[b * 2 + 1] = if b == 1 { 0.0 } else { 1.0 };
    }
    // Unknown energy with a huge stored value is still 0; known clips at 400.
    energy[0] = 1000.0;
    let meta_t = IdTensor::from_slice(&meta, vec![batch, 8], &device).unwrap();
    let energy_t = Tensor::<R, f32>::from_f32(&energy, vec![batch, 2], &device).unwrap();
    let out =
        Tensor::<R, f32>::from_f32(&vec![f32::NAN; batch * 34], vec![batch, 34], &device).unwrap();
    ms2::meta_features(&meta_t, &energy_t, &waves, &out).unwrap();
    check_launches(&device).unwrap();
    let got = out.try_to_f32().unwrap();
    let want = twin::meta_features(&meta, &energy, batch);
    assert_close(&got, &want, 1e-5, "meta_features");
    // Kept column.
    let kept: Vec<u32> = (0..batch * n_keep * 3).map(|i| i as u32).collect();
    let kept_t = IdTensor::from_slice(&kept, vec![batch, n_keep, 3], &device).unwrap();
    for col in 0..3 {
        let got_c = ms2::kept_column(&kept_t, col).unwrap();
        check_launches(&device).unwrap();
        let got_v = got_c.try_to_vec().unwrap();
        let want_v = twin::kept_column(&kept, batch, n_keep, col);
        assert_eq!(got_v, want_v, "kept_column {col}");
    }
    let _ = n_raw;
}

#[test]
fn encoder_shapes_finite_and_pool() {
    let device = dev();
    let config = small_config(16, 16, 2);
    let mut rng = Rng::seeded(11);
    let encoder = Ms2Encoder::<R, f32>::init(&config, &device, &mut rng).unwrap();
    let batch = make_batch(&[101, 102, 103], 64, &[10, 0, 5], 400_000_000, 21);
    let spectra = DeviceSpectra::<R, f32>::upload(&batch, &device).unwrap();
    let peaks = ms2::PeakBuffers::<R, f32>::new(3, 64, 16, &device);
    let out = encoder.encode(&spectra, &peaks, Control::None).unwrap();
    check_launches(&device).unwrap();
    let (b, n, d) = (3usize, 16usize, 16usize);
    assert_eq!(out.x.dims(), &[b, n, d]);
    assert_eq!(out.valid.dims(), &[b, n]);
    assert_eq!(out.memory.dims(), &[b, 1 + n, d]);
    assert_eq!(out.memory_mask.dims(), &[b, 1 + n]);
    assert_eq!(out.pool.dims(), &[b, d]);
    assert_eq!(out.context.dims(), &[b, d]);
    let x = out.x.try_to_f32().unwrap();
    let valid = out.valid.try_to_f32().unwrap();
    let memory = out.memory.try_to_f32().unwrap();
    let mask = out.memory_mask.try_to_f32().unwrap();
    let pool = out.pool.try_to_f32().unwrap();
    let context = out.context.try_to_f32().unwrap();
    for (name, v) in [
        ("x", &x),
        ("memory", &memory),
        ("pool", &pool),
        ("context", &context),
    ] {
        assert!(v.iter().all(|x| x.is_finite()), "{name} non-finite");
    }
    // x exactly 0 at padding.
    for bi in 0..b {
        for p in 0..n {
            if valid[bi * n + p] == 0.0 {
                for c in 0..d {
                    assert!(
                        x[(bi * n + p) * d + c].to_bits() == 0,
                        "padding ({bi},{p}) not +0.0"
                    );
                }
            }
        }
    }
    // memory_mask is [1; valid].
    for bi in 0..b {
        assert_eq!(mask[bi * (1 + n)], 1.0, "mask first column");
        for p in 0..n {
            assert_eq!(
                mask[bi * (1 + n) + 1 + p],
                valid[bi * n + p],
                "mask peak {p}"
            );
        }
    }
    // pool equals masked mean plus g from read-back x and context.
    for bi in 0..b {
        let len = valid[bi * n..(bi + 1) * n]
            .iter()
            .map(|&v| v as usize)
            .sum::<usize>()
            .max(1) as f32;
        for c in 0..d {
            let mut s = 0.0f32;
            for p in 0..n {
                s += x[(bi * n + p) * d + c];
            }
            let want = s / len + context[bi * d + c];
            let got = pool[bi * d + c];
            let ok = if want.abs() <= 1.0 {
                (got - want).abs() <= 1e-4
            } else {
                (got - want).abs() <= 1e-4 * want.abs()
            };
            assert!(ok, "pool ({bi},{c}): got {got}, want {want}");
        }
    }
}

#[test]
fn encoder_padding_independence() {
    let device = dev();
    let config = small_config(16, 16, 2);
    let mut rng = Rng::seeded(31);
    let encoder = Ms2Encoder::<R, f32>::init(&config, &device, &mut rng).unwrap();
    let mut base = make_batch(&[201, 202], 64, &[10, 8], 400_000_000, 33);
    // Dirty variant: poisoned padding plus two extra ineligible peaks appended.
    let mut dirty = base.clone();
    for b in 0..2 {
        let old_count = base.peak_count[b] as usize;
        let precursor = base.precursor_mz_udalton[b];
        // Two ineligible peaks: one above precursor + 2 Da, one below the floor.
        dirty.peak_count[b] = (old_count + 2) as u32;
        dirty.raw_peak_count[b] = (old_count + 2) as u32;
        dirty.peak_id[b * 64 + old_count] = old_count as u32;
        dirty.peak_id[b * 64 + old_count + 1] = (old_count + 1) as u32;
        dirty.mz_udalton[b * 64 + old_count] = precursor + 5_000_000;
        dirty.intensity[b * 64 + old_count] = 1.0;
        dirty.mz_udalton[b * 64 + old_count + 1] = 150_000_000;
        dirty.intensity[b * 64 + old_count + 1] = 1e-9;
        for i in (old_count + 2)..64 {
            dirty.mz_udalton[b * 64 + i] = u32::MAX;
            dirty.intensity[b * 64 + i] = f32::NAN;
            dirty.peak_id[b * 64 + i] = 1000 + i as u32;
        }
        // Keep ids strictly increasing over the valid range.
        for i in 0..(old_count + 2) {
            dirty.peak_id[b * 64 + i] = i as u32;
        }
    }
    // Fix raw counts for the base padding (already zero, but poison dirty only).
    for b in 0..2 {
        for i in (base.peak_count[b] as usize)..64 {
            base.mz_udalton[b * 64 + i] = 0;
            base.intensity[b * 64 + i] = 0.0;
        }
    }
    let run = |batch: &SpectrumBatch| {
        let spectra = DeviceSpectra::<R, f32>::upload(batch, &device).unwrap();
        let peaks = ms2::PeakBuffers::<R, f32>::new(batch.len(), 64, 16, &device);
        let out = encoder.encode(&spectra, &peaks, Control::None).unwrap();
        check_launches(&device).unwrap();
        (
            out.memory.try_to_f32().unwrap(),
            out.pool.try_to_f32().unwrap(),
        )
    };
    let (mem_base, pool_base) = run(&base);
    let (mem_dirty, pool_dirty) = run(&dirty);
    assert_eq!(mem_base.len(), mem_dirty.len());
    for (i, (a, b)) in mem_base.iter().zip(&mem_dirty).enumerate() {
        assert!(
            a.to_bits() == b.to_bits(),
            "memory[{i}] differs with poisoned padding: {a} vs {b}"
        );
    }
    for (i, (a, b)) in pool_base.iter().zip(&pool_dirty).enumerate() {
        assert!(
            a.to_bits() == b.to_bits(),
            "pool[{i}] differs with poisoned padding: {a} vs {b}"
        );
    }
}

#[test]
fn encoder_batch_independence_and_permutation() {
    let device = dev();
    let config = small_config(16, 16, 2);
    let mut rng = Rng::seeded(41);
    let encoder = Ms2Encoder::<R, f32>::init(&config, &device, &mut rng).unwrap();
    let single = make_batch(&[301], 64, &[9], 400_000_000, 51);
    // Same peaks as `single` at position 1 of a batch of 3 (ids differ elsewhere).
    let mut batch3 = make_batch(&[301, 302, 303], 64, &[9, 7, 4], 400_000_000, 52);
    // Copy single's peaks into position 1.
    for i in 0..64 {
        batch3.peak_id[1 * 64 + i] = single.peak_id[i];
        batch3.mz_udalton[1 * 64 + i] = single.mz_udalton[i];
        batch3.intensity[1 * 64 + i] = single.intensity[i];
    }
    batch3.peak_count[1] = single.peak_count[0];
    batch3.raw_peak_count[1] = single.raw_peak_count[0];
    batch3.precursor_mz_udalton[1] = single.precursor_mz_udalton[0];
    batch3.adduct[1] = single.adduct[0];
    batch3.polarity[1] = single.polarity[0];
    batch3.collision_energy_ev[1] = single.collision_energy_ev[0];
    batch3.collision_energy_known[1] = single.collision_energy_known[0];
    batch3.energy_count[1] = single.energy_count[0];
    batch3.mz_uncertainty_udalton[1] = single.mz_uncertainty_udalton[0];
    batch3.precursor_uncertainty_udalton[1] = single.precursor_uncertainty_udalton[0];
    let run = |batch: &SpectrumBatch| {
        let spectra = DeviceSpectra::<R, f32>::upload(batch, &device).unwrap();
        let peaks = ms2::PeakBuffers::<R, f32>::new(batch.len(), 64, 16, &device);
        let out = encoder.encode(&spectra, &peaks, Control::None).unwrap();
        check_launches(&device).unwrap();
        (
            out.memory.try_to_f32().unwrap(),
            out.pool.try_to_f32().unwrap(),
        )
    };
    let (mem_single, pool_single) = run(&single);
    let (mem_batch, pool_batch) = run(&batch3);
    let d = 16usize;
    let n1 = 1 + 16;
    // Position 1 of the batch matches the single.
    for c in 0..d {
        for p in 0..n1 {
            let a = mem_single[p * d + c];
            let b = mem_batch[(1 * n1 + p) * d + c];
            assert!(
                (a - b).abs() <= 1e-5 * (1.0 + b.abs()),
                "batch independence memory[{p},{c}]: {a} vs {b}"
            );
        }
        let a = pool_single[c];
        let b = pool_batch[1 * d + c];
        assert!(
            (a - b).abs() <= 1e-5 * (1.0 + b.abs()),
            "batch independence pool[{c}]: {a} vs {b}"
        );
    }
    // Permuting the batch permutes the outputs.
    let mut perm = batch3.clone();
    let order = [2usize, 0, 1];
    let permute_rows = |v: &mut Vec<u32>, width: usize| {
        let old = v.clone();
        for (nb, &ob) in order.iter().enumerate() {
            v[nb * width..(nb + 1) * width].copy_from_slice(&old[ob * width..(ob + 1) * width]);
        }
    };
    permute_rows(&mut perm.peak_count, 1);
    permute_rows(&mut perm.raw_peak_count, 1);
    permute_rows(&mut perm.peak_id, 64);
    permute_rows(&mut perm.mz_udalton, 64);
    perm.spectrum_id = order.iter().map(|&o| batch3.spectrum_id[o]).collect();
    permute_rows(&mut perm.mz_uncertainty_udalton, 1);
    permute_rows(&mut perm.precursor_mz_udalton, 1);
    permute_rows(&mut perm.precursor_uncertainty_udalton, 1);
    let mut ad: Vec<u32> = perm.adduct.iter().map(|&v| v as u32).collect();
    permute_rows(&mut ad, 1);
    perm.adduct = ad.iter().map(|&v| v as u16).collect();
    // Permute remaining per-spectrum fields directly.
    let mut intensity_p = vec![0.0f32; 3 * 64];
    for (nb, &ob) in order.iter().enumerate() {
        intensity_p[nb * 64..(nb + 1) * 64]
            .copy_from_slice(&batch3.intensity[ob * 64..(ob + 1) * 64]);
    }
    perm.intensity = intensity_p;
    let mut ce = perm.collision_energy_ev.clone();
    let mut known = perm.collision_energy_known.clone();
    let mut ecount = perm.energy_count.clone();
    let mut frag = perm.fragment_tolerance_ppm_tenths.clone();
    let mut prec = perm.precursor_tolerance_ppm_tenths.clone();
    let mut inst = perm.instrument_class.clone();
    let mut polv = perm.polarity.clone();
    for (nb, &ob) in order.iter().enumerate() {
        ce[nb] = batch3.collision_energy_ev[ob];
        known[nb] = batch3.collision_energy_known[ob];
        ecount[nb] = batch3.energy_count[ob];
        frag[nb] = batch3.fragment_tolerance_ppm_tenths[ob];
        prec[nb] = batch3.precursor_tolerance_ppm_tenths[ob];
        inst[nb] = batch3.instrument_class[ob];
        polv[nb] = batch3.polarity[ob];
    }
    perm.collision_energy_ev = ce;
    perm.collision_energy_known = known;
    perm.energy_count = ecount;
    perm.fragment_tolerance_ppm_tenths = frag;
    perm.precursor_tolerance_ppm_tenths = prec;
    perm.instrument_class = inst;
    perm.polarity = polv;
    let (mem_perm, pool_perm) = run(&perm);
    for (nb, &ob) in order.iter().enumerate() {
        for c in 0..d {
            for p in 0..n1 {
                let a = mem_batch[(ob * n1 + p) * d + c];
                let b = mem_perm[(nb * n1 + p) * d + c];
                assert!(
                    (a - b).abs() <= 1e-5 * (1.0 + b.abs()),
                    "perm memory[{nb},{p},{c}]"
                );
            }
            let a = pool_batch[ob * d + c];
            let b = pool_perm[nb * d + c];
            assert!(
                (a - b).abs() <= 1e-5 * (1.0 + b.abs()),
                "perm pool[{nb},{c}]"
            );
        }
    }
}

#[test]
fn encoder_bidirectional_block_matches_stepped_reference() {
    let device = dev();
    let config = small_config(16, 16, 2);
    let mut rng = Rng::seeded(61);
    let encoder = Ms2Encoder::<R, f32>::init(&config, &device, &mut rng).unwrap();
    for &len in &[1usize, 5, 16] {
        let batch = make_batch(&[401], 64, &[len as u32], 500_000_000, 70 + len as u64);
        let spectra = DeviceSpectra::<R, f32>::upload(&batch, &device).unwrap();
        let peaks = ms2::PeakBuffers::<R, f32>::new(1, 64, 16, &device);
        let x0 = encoder.embed(&spectra, &peaks, Control::None).unwrap();
        check_launches(&device).unwrap();
        let summary = peaks.summary.try_to_vec().unwrap();
        assert_eq!(summary[0] as usize, len, "valid length {len}");
        let reverse_ids = ms2::kept_column(&peaks.kept, 2).unwrap();
        check_launches(&device).unwrap();
        let (fwd, bwd) = encoder.block(0);
        // Parallel block-0 output.
        let f = fwd.apply(&x0).unwrap();
        let rev = Var::gather_tokens(&x0, &reverse_ids, 16).unwrap();
        let b = bwd.apply(&rev).unwrap();
        let r = Var::gather_tokens(&b, &reverse_ids, 16).unwrap();
        let parallel = f.add(&r).unwrap().sub(&x0).unwrap();
        check_launches(&device).unwrap();
        let p_vals = parallel.try_to_f32().unwrap();
        // Stepped reference.
        let mut fwd_cache = fwd.empty_cache(1, &device);
        let mut fwd_outs: Vec<Vec<f32>> = Vec::new();
        for pos in 0..len {
            let inp = x0.slice(1, pos, 1).unwrap();
            let (out, next) = fwd.step(&inp, &fwd_cache).unwrap();
            fwd_cache = next;
            fwd_outs.push(out.try_to_f32().unwrap());
        }
        let mut bwd_cache = bwd.empty_cache(1, &device);
        let mut bwd_rev: Vec<Vec<f32>> = Vec::new();
        for k in 0..len {
            let pos = len - 1 - k;
            let inp = x0.slice(1, pos, 1).unwrap();
            let (out, next) = bwd.step(&inp, &bwd_cache).unwrap();
            bwd_cache = next;
            bwd_rev.push(out.try_to_f32().unwrap());
        }
        let x0_vals = x0.try_to_f32().unwrap();
        check_launches(&device).unwrap();
        let d = 16usize;
        let mut worst = 0.0f32;
        for p in 0..len {
            let fp = &fwd_outs[p];
            let rp = &bwd_rev[len - 1 - p];
            for c in 0..d {
                let want = fp[c] + rp[c] - x0_vals[p * d + c];
                let got = p_vals[p * d + c];
                worst = worst.max((got - want).abs() / (1.0 + want.abs()));
            }
        }
        assert!(
            worst <= 1e-4,
            "len {len}: stepped reference differs by {worst}"
        );
    }
}

#[test]
fn mamba3_block_parity_for_contracted_config() {
    use mamba3::models::mamba3::Mamba3BlockConfig;
    let device = dev();
    let ssm = ModelConfig::v0().encoder;
    let mut rng = Rng::seeded(81);
    let block = Mamba3BlockConfig::new(ssm)
        .init::<R, f32>(&device, &mut rng)
        .unwrap();
    let (b, t, d) = (1usize, 12usize, 128usize);
    let mut rng2 = Rng::seeded(82);
    let data = rng2.uniform_vec(b * t * d, -0.5, 0.5);
    let input = Var::constant(Tensor::<R, f32>::from_f32(&data, vec![b, t, d], &device).unwrap());
    let cache0 = block.empty_cache(b, &device);
    let (parallel, state_parallel) = block.apply_with_state(&input, Some(&cache0)).unwrap();
    check_launches(&device).unwrap();
    let p_vals = parallel.try_to_f32().unwrap();
    let mut cache = block.empty_cache(b, &device);
    let mut steps: Vec<Vec<f32>> = Vec::new();
    for pos in 0..t {
        let inp = input.slice(1, pos, 1).unwrap();
        let (out, next) = block.step(&inp, &cache).unwrap();
        cache = next;
        steps.push(out.try_to_f32().unwrap());
    }
    check_launches(&device).unwrap();
    let mut flat: Vec<f32> = Vec::with_capacity(b * t * d);
    for s in &steps {
        flat.extend_from_slice(s);
    }
    let worst_out = max_abs_diff(&flat, &p_vals) / 129.0;
    assert!(
        max_abs_diff(&flat, &p_vals) <= 1e-4 * 129.0,
        "apply vs steps output differs: worst {worst_out}"
    );
    // Final carries.
    let sp = state_parallel.expect("want_state gives a carry");
    let h_p = sp.ssm.h.try_to_f32().unwrap();
    let h_s = cache.ssm.h.try_to_f32().unwrap();
    let u_p = sp.ssm.last_u.try_to_f32().unwrap();
    let u_s = cache.ssm.last_u.try_to_f32().unwrap();
    assert!(
        max_abs_diff(&h_p, &h_s)
            <= 1e-4 * (1.0 + h_p.iter().map(|v| v.abs()).fold(0.0f32, f32::max)),
        "carry h differs: {}",
        max_abs_diff(&h_p, &h_s)
    );
    assert!(
        max_abs_diff(&u_p, &u_s)
            <= 1e-4 * (1.0 + u_p.iter().map(|v| v.abs()).fold(0.0f32, f32::max)),
        "carry last_u differs: {}",
        max_abs_diff(&u_p, &u_s)
    );
    let a_p = sp.ssm.angle.as_ref().unwrap().try_to_f32().unwrap();
    let a_s = cache.ssm.angle.as_ref().unwrap().try_to_f32().unwrap();
    assert!(
        max_abs_diff(&a_p, &a_s)
            <= 1e-4 * (1.0 + a_p.iter().map(|v| v.abs()).fold(0.0f32, f32::max)),
        "carry angle differs: {}",
        max_abs_diff(&a_p, &a_s)
    );
}

#[test]
fn encoder_gradients_match_finite_differences() {
    let device = dev();
    let config = small_config(8, 6, 1);
    let mut rng = Rng::seeded(91);
    let encoder = Ms2Encoder::<R, f32>::init(&config, &device, &mut rng).unwrap();
    let batch = make_batch(&[501, 502], 64, &[6, 5], 400_000_000, 93);
    let w_host: Vec<f32> = (0..2 * 8).map(|i| 0.1 * (i as f32) - 0.5).collect();
    let w_t = Tensor::<R, f32>::from_f32(&w_host, vec![2, 8], &device).unwrap();
    let loss_of = |enc: &Ms2Encoder<R, f32>| {
        let spectra = DeviceSpectra::<R, f32>::upload(&batch, &device).unwrap();
        let peaks = ms2::PeakBuffers::<R, f32>::new(2, 64, 6, &device);
        let out = enc.encode(&spectra, &peaks, Control::None).unwrap();
        out.pool
            .mul(&Var::constant(w_t.clone()))
            .unwrap()
            .sum()
            .unwrap()
    };
    // Analytic gradients.
    let spectra = DeviceSpectra::<R, f32>::upload(&batch, &device).unwrap();
    let peaks = ms2::PeakBuffers::<R, f32>::new(2, 64, 6, &device);
    let out = encoder.encode(&spectra, &peaks, Control::None).unwrap();
    let loss = out
        .pool
        .mul(&Var::constant(w_t.clone()))
        .unwrap()
        .sum()
        .unwrap();
    check_launches(&device).unwrap();
    let grads = loss.backward_retain().unwrap();
    let params = encoder.named_parameters();
    let find = |name: &str| {
        params
            .iter()
            .find(|(n, _)| n == name)
            .unwrap_or_else(|| panic!("missing param {name}"))
            .1
            .clone()
    };
    let targets = [
        "peak_in.weight",
        "blocks.0.forward.mixer.in_proj.weight",
        "blocks.0.backward.mixer.in_proj.weight",
        "adduct",
    ];
    for name in targets {
        let param = find(name);
        let analytic = grads
            .get(param.id())
            .unwrap_or_else(|| panic!("no gradient for {name}"))
            .to_f32();
        let shape = param.shape().dims().to_vec();
        let base = param.value().to_f32();
        for &idx in &[0usize, 1, 2] {
            assert!(idx < base.len(), "{name} too small");
            let mut up = base.clone();
            up[idx] += 1e-2;
            param.set(Tensor::<R, f32>::from_f32(&up, shape.clone(), &device).unwrap());
            let fu = loss_of(&encoder).to_f32()[0];
            let mut down = base.clone();
            down[idx] -= 1e-2;
            param.set(Tensor::<R, f32>::from_f32(&down, shape.clone(), &device).unwrap());
            let fd = loss_of(&encoder).to_f32()[0];
            param.set(Tensor::<R, f32>::from_f32(&base, shape.clone(), &device).unwrap());
            let numeric = (fu - fd) / 2e-2;
            let a = analytic[idx];
            assert!(
                (a - numeric).abs() <= 3e-2 * numeric.abs() + 2e-3,
                "{name}[{idx}]: analytic={a} numeric={numeric}"
            );
        }
    }
    check_launches(&device).unwrap();
}

#[test]
fn encoder_controls() {
    let device = dev();
    let config = small_config(16, 16, 2);
    let mut rng = Rng::seeded(101);
    let encoder = Ms2Encoder::<R, f32>::init(&config, &device, &mut rng).unwrap();
    let batch = make_batch(&[601, 602], 64, &[8, 6], 400_000_000, 103);
    let run = |control: Control| {
        let spectra = DeviceSpectra::<R, f32>::upload(&batch, &device).unwrap();
        let peaks = ms2::PeakBuffers::<R, f32>::new(2, 64, 16, &device);
        let out = encoder.encode(&spectra, &peaks, control).unwrap();
        check_launches(&device).unwrap();
        (
            out.memory_mask.try_to_f32().unwrap(),
            out.pool.try_to_f32().unwrap(),
            out.context.try_to_f32().unwrap(),
        )
    };
    let (mask_meta, pool_meta, ctx_meta) = run(Control::MetadataOnly);
    for b in 0..2 {
        assert_eq!(mask_meta[b * 17], 1.0);
        for p in 1..17 {
            assert_eq!(mask_meta[b * 17 + p], 0.0, "MetadataOnly mask ({b},{p})");
        }
    }
    assert_close(&pool_meta, &ctx_meta, 1e-5, "MetadataOnly pool == context");
    // rotate_peaks moves exactly the listed fields.
    let src = make_batch(&[701, 702, 703], 64, &[5, 6, 7], 400_000_000, 107);
    let rotated = rotate_peaks(&src);
    let b = 3usize;
    let n_raw = 64usize;
    for dst in 0..b {
        let s = (dst + 1) % b;
        assert_eq!(rotated.peak_count[dst], src.peak_count[s]);
        assert_eq!(rotated.raw_peak_count[dst], src.raw_peak_count[s]);
        assert_eq!(
            rotated.peak_id[dst * n_raw..(dst + 1) * n_raw],
            src.peak_id[s * n_raw..(s + 1) * n_raw]
        );
        assert_eq!(
            rotated.mz_udalton[dst * n_raw..(dst + 1) * n_raw],
            src.mz_udalton[s * n_raw..(s + 1) * n_raw]
        );
        assert_eq!(
            rotated.intensity[dst * n_raw..(dst + 1) * n_raw],
            src.intensity[s * n_raw..(s + 1) * n_raw]
        );
        assert_eq!(
            rotated.mz_uncertainty_udalton[dst],
            src.mz_uncertainty_udalton[s]
        );
        // Everything else kept.
        assert_eq!(rotated.spectrum_id[dst], src.spectrum_id[dst]);
        assert_eq!(
            rotated.precursor_mz_udalton[dst],
            src.precursor_mz_udalton[dst]
        );
        assert_eq!(rotated.adduct[dst], src.adduct[dst]);
        assert_eq!(rotated.polarity[dst], src.polarity[dst]);
        assert_eq!(
            rotated.collision_energy_ev[dst],
            src.collision_energy_ev[dst]
        );
        assert_eq!(
            rotated.collision_energy_known[dst],
            src.collision_energy_known[dst]
        );
        assert_eq!(rotated.energy_count[dst], src.energy_count[dst]);
    }
    // Unknown energy uses row 0 and feature 0 is 0.
    let mut unk = make_batch(&[801], 64, &[4], 400_000_000, 109);
    unk.collision_energy_known[0] = 0;
    unk.collision_energy_ev[0] = 99.0;
    unk.energy_count[0] = 0;
    let spectra = DeviceSpectra::<R, f32>::upload(&unk, &device).unwrap();
    let meta_ids = spectra.meta_ids.try_to_vec().unwrap();
    assert_eq!(meta_ids[3], 0, "unknown energy_known row");
    let energy = spectra.energy.try_to_f32().unwrap();
    assert_eq!(energy[0], 0.0);
    assert_eq!(energy[1], 0.0);
    let meta = spectra.meta.try_to_vec().unwrap();
    let got_meta = {
        let waves_here = ms2::Ms2Constants::new(&device);
        let meta_t = IdTensor::from_slice(&meta, vec![1, 8], &device).unwrap();
        let energy_t = Tensor::<R, f32>::from_f32(&energy, vec![1, 2], &device).unwrap();
        let out = Tensor::<R, f32>::from_f32(&vec![f32::NAN; 34], vec![1, 34], &device).unwrap();
        ms2::meta_features(&meta_t, &energy_t, &waves_here, &out).unwrap();
        check_launches(&device).unwrap();
        out.try_to_f32().unwrap()
    };
    assert_eq!(got_meta[0].to_bits(), 0, "unknown energy feature 0");
    let want = twin::meta_features(&meta, &energy, 1);
    assert_eq!(want[0].to_bits(), 0);
}

#[test]
fn encoder_long_input() {
    let device = dev();
    let config = ModelConfig::v0();
    let mut rng = Rng::seeded(121);
    let encoder = Ms2Encoder::<R, f32>::init(&config, &device, &mut rng).unwrap();
    let batch = make_batch(&[901, 902], 512, &[400, 1], 600_000_000, 123);
    let spectra = DeviceSpectra::<R, f32>::upload(&batch, &device).unwrap();
    let peaks = ms2::PeakBuffers::<R, f32>::new(2, 512, 128, &device);
    let out = encoder.encode(&spectra, &peaks, Control::None).unwrap();
    check_launches(&device).unwrap();
    let memory = out.memory.try_to_f32().unwrap();
    let pool = out.pool.try_to_f32().unwrap();
    assert!(memory.iter().all(|v| v.is_finite()), "long memory finite");
    assert!(pool.iter().all(|v| v.is_finite()), "long pool finite");
    let summary = peaks.summary.try_to_vec().unwrap();
    assert_eq!(summary[0], 128, "400 peaks keep 128");
    assert_eq!(summary[2], 1, "1 peak keeps 1");
    assert_eq!(summary[1] & (1 << 19), 1 << 19, "truncation bit");
    assert_eq!(summary[3] & (1 << 19), 0, "no truncation");
}
