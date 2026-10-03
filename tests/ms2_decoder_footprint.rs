//! V0-B footprint: one warmed training step performs no device read, and two
//! consecutive warmed steps launch the same kernels.

#![cfg(feature = "backend")]

use mamba3::backend::{
    check_launches, launch_count, reset_launch_count, reset_transfer_counters, runtime_read_count,
};
use mamba3::backends::Auto;
use mamba3::models::ms2::batch::DeviceSpectra;
use mamba3::models::ms2::chem::{Composition, composition_mass};
use mamba3::models::ms2::contract::{Control, ModelConfig, SCHEMA_VERSION, SpectrumBatch};
use mamba3::models::ms2::decoder::{Ms2Decoder, ReplayView, graph_loss};
use mamba3::models::ms2::encoder::Ms2Encoder;
use mamba3::models::ms2::grammar::{Limits, Token};
use mamba3::models::ms2::targets::{Labels, Target};
use mamba3::models::ms2::targets_batch::TargetBatch;
use mamba3::nn::Module;
use mamba3::tensor::ops::ms2::{self, Ms2Constants, ReplayBuffers};
use mamba3::tensor::ops::random::Rng;
use mamba3::train::optim::{AdamWConfig, Optimizer};

type R = Auto;
type E = f32;

fn tiny_config() -> ModelConfig {
    let mut m = ModelConfig::v0();
    m.d_model = 16;
    m.n_peaks = 16;
    m.encoder_blocks = 1;
    m.decoder_blocks = 1;
    m.attention_heads = 2;
    m.encoder.d_model = 16;
    m.encoder.n_heads = 2;
    m.encoder.head_dim = 8;
    m.encoder.d_state = 8;
    m.encoder.n_groups = 2;
    m.decoder.d_model = 16;
    m.decoder.n_heads = 2;
    m.decoder.head_dim = 8;
    m.decoder.d_state = 8;
    m.decoder.n_groups = 2;
    m
}

#[test]
fn warmed_training_step_reads_nothing_and_launches_constantly() {
    let device = mamba3::backend::Device::<R>::default();
    let model = tiny_config();
    let mut rng = Rng::seeded(97);
    let encoder = Ms2Encoder::init(&model, &device, &mut rng).unwrap();
    let decoder = Ms2Decoder::init(&model, &device, &mut rng).unwrap();
    // Two spectra, one single-target label each (ethanol-ish traces).
    let trace0 = vec![
        Token {
            kind: 1,
            atom_type: 0,
            bond: 0,
            pointer: 0,
        },
        Token {
            kind: 2,
            atom_type: 2,
            bond: 0,
            pointer: 0,
        },
        Token {
            kind: 2,
            atom_type: 3,
            bond: 1,
            pointer: 0,
        },
        Token {
            kind: 2,
            atom_type: 4,
            bond: 1,
            pointer: 1,
        },
        Token {
            kind: 4,
            atom_type: 0,
            bond: 0,
            pointer: 0,
        },
    ];
    let trace1 = vec![
        Token {
            kind: 1,
            atom_type: 0,
            bond: 0,
            pointer: 0,
        },
        Token {
            kind: 2,
            atom_type: 8,
            bond: 0,
            pointer: 0,
        },
        Token {
            kind: 2,
            atom_type: 2,
            bond: 1,
            pointer: 0,
        },
        Token {
            kind: 4,
            atom_type: 0,
            bond: 0,
            pointer: 0,
        },
    ];
    let mk_labels = |trace: Vec<Token>| Labels {
        embeddings: Vec::new(),
        graphs: 1,
        targets_before_cut: 1,
        targets: vec![Target {
            trace,
            weight: 1,
            q: 1.0,
            embeddings: Vec::new(),
            anchors: Vec::new(),
        }],
        dropped_weight: 0.0,
        cut_is_tied: false,
        explained_peaks: Vec::new(),
        ambiguous_hypotheses: 0,
        canonicalization_failures: 0,
    };
    let lab0 = mk_labels(trace0);
    let lab1 = mk_labels(trace1);
    // Parent budgets covering the traces above.
    let parent0: Composition = [3, 6, 0, 0, 0, 0, 0, 0, 0, 0];
    let parent1: Composition = [1, 1, 0, 1, 0, 0, 0, 0, 0, 0];
    let parents = vec![parent0, parent1];
    let refs: Vec<Option<&Labels>> = vec![Some(&lab0), Some(&lab1)];
    let batch = TargetBatch::build(&refs, &parents, 2, Limits::V0).unwrap();
    let targets = batch.upload(&device).unwrap();
    let t = Limits::V0.max_steps();
    let constants = Ms2Constants::new(&device);
    let buffers = ReplayBuffers::poisoned(4, t, 16, &device).unwrap();
    ms2::grammar_replay(&targets.tokens, &targets.meta, &constants, 16, 4, &buffers).unwrap();
    check_launches(&device).unwrap();
    let replay = ReplayView {
        replay: &buffers.replay,
        atoms: &buffers.atoms,
    };
    // Two spectra with valid peaks and precursors.
    let n_raw = 64usize;
    let mut rng = Rng::seeded(98);
    let mut peak_id = vec![u32::MAX; 2 * n_raw];
    let mut mz = vec![0u32; 2 * n_raw];
    let mut intensity = vec![0.0f32; 2 * n_raw];
    for b in 0..2 {
        for i in 0..16 {
            peak_id[b * n_raw + i] = i as u32;
            mz[b * n_raw + i] = 60_000_000 + (rng.uniform_vec(1, 0.0, 100_000_000.0)[0] as u32);
            intensity[b * n_raw + i] = 0.5 + rng.uniform_vec(1, 0.0, 2.0)[0];
        }
    }
    let pmass: Vec<u32> = parents
        .iter()
        .map(|c| composition_mass(c).unwrap())
        .collect();
    let spectra_batch = SpectrumBatch {
        schema_version: SCHEMA_VERSION,
        n_raw: n_raw as u32,
        spectrum_id: vec![201, 202],
        raw_peak_count: vec![16, 16],
        peak_count: vec![16, 16],
        peak_id,
        mz_udalton: mz,
        intensity,
        intensity_scale: 0,
        mz_uncertainty_udalton: vec![50, 50],
        precursor_mz_udalton: vec![pmass[0] + 1_007_825 - 549, pmass[1] + 1_007_825 - 549],
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
    let spectra = DeviceSpectra::upload(&spectra_batch, &device).unwrap();
    let peaks = ms2::PeakBuffers::<R, E>::new(2, n_raw, 16, &device);
    let mut opt = AdamWConfig::builder()
        .learning_rate(3e-3)
        .build()
        .init::<R, E>();
    let mut params = encoder.named_parameters();
    params.extend(decoder.named_parameters());
    let only_values: Vec<mamba3::nn::param::Param<R, E>> =
        params.iter().map(|(_, p)| p.clone()).collect();
    // One warmed training step (encode + teacher + loss + backward + AdamW
    // step), the loss never read.
    let mut step_once = || {
        let encoded = encoder.encode(&spectra, &peaks, Control::None).unwrap();
        let out = decoder
            .teacher(&encoded, &encoded.pool, &targets, &replay)
            .unwrap();
        let loss = graph_loss(&out, &targets.q, 2).unwrap();
        let grads = loss.backward_retain().unwrap();
        opt.step(&only_values, &grads).unwrap();
        check_launches(&device).unwrap();
    };
    step_once();
    // Warmed production: both counters advance only at no reporting boundary.
    reset_transfer_counters();
    reset_launch_count();
    let reads_before = runtime_read_count();
    let launches_before = launch_count();
    step_once();
    let reads_after_first = runtime_read_count();
    let launches_after_first = launch_count();
    step_once();
    let reads_after_second = runtime_read_count();
    let launches_after_second = launch_count();
    println!(
        "warmed training step: reads +{} then +{}, launches +{} then +{}",
        reads_after_first - reads_before,
        reads_after_second - reads_after_first,
        launches_after_first - launches_before,
        launches_after_second - launches_after_first,
    );
    assert_eq!(
        reads_after_first - reads_before,
        0,
        "first warmed step reads nothing"
    );
    assert_eq!(
        reads_after_second - reads_after_first,
        0,
        "second warmed step reads nothing"
    );
    assert_eq!(
        launches_after_first - launches_before,
        launches_after_second - launches_after_first,
        "launch count delta is constant across warmed steps"
    );
}
