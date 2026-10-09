//! Spectral evidence for the completion model: fragment peaks, adduct and
//! neutral mass (`completion_spectrum`).
//!
//! Hand-built molecules only (atom type ids from
//! [`chem::ATOM_TYPES`](mamba3::models::ms2::chem::ATOM_TYPES)): the nine
//! molecules of `ms2_completion_fingerprint.rs` with made-up peak lists (the
//! tests do not need real spectra). Every device call is followed by
//! [`check_launches`].

#![cfg(feature = "backend")]

use mamba3::backend::{Device, check_launches};
use mamba3::backends::Auto;
use mamba3::models::ms2::chem::{Composition, composition_mass};
use mamba3::models::ms2::completion_data::{ExtractionConfig, PatternSource, same_identity};
use mamba3::models::ms2::completion_fingerprint::FingerprintBatch;
use mamba3::models::ms2::completion_model::{
    CompletionGenerationConfig, CompletionModel, CompletionModelConfig, CompletionRequest,
    CompletionTrainConfig, CompletionTrainer, PatternBatch, SubstructureSemantics,
};
use mamba3::models::ms2::completion_spectrum::{
    COMPLETION_ADDUCTS, SPECTRUM_ADDUCT_ROWS, SpectrumBatch, SpectrumEncoder, SpectrumEvidence,
    completion_adduct_by_name, neutral_mass_of, precursor_mz_of,
};
use mamba3::models::ms2::grammar::{
    CANONICAL_WORK_LIMIT, Limits, Token, canonical_trace, replay_exact,
};
use mamba3::models::ms2::graph::MolGraph;
use mamba3::models::ms2::targets_batch::TargetBatch;
use mamba3::models::ms2::twin;
use mamba3::nn::Module;
use mamba3::tensor::ops::ms2::{META_FEATURES, META_WIDTH, Ms2Constants, PEAK_FEATURES};
use mamba3::tensor::ops::random::Rng;

type R = Auto;
type E = f32;

static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn serial() -> std::sync::MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

fn dev() -> Device<R> {
    Device::<R>::default()
}

fn limits() -> Limits {
    Limits::new(16, 4).unwrap()
}

fn nine_molecules() -> Vec<MolGraph> {
    let chain =
        |atoms: Vec<u8>, bonds: Vec<(usize, usize, u8)>| MolGraph::new(atoms, bonds).unwrap();
    vec![
        // ethanol, dimethyl ether (C2H6O isomers)
        chain(vec![4, 3, 9], vec![(0, 1, 1), (1, 2, 1)]),
        chain(vec![4, 8, 4], vec![(0, 1, 1), (1, 2, 1)]),
        // propan-1-ol, propan-2-ol, methoxyethane (C3H8O isomers)
        chain(vec![4, 3, 3, 9], vec![(0, 1, 1), (1, 2, 1), (2, 3, 1)]),
        chain(vec![4, 2, 4, 9], vec![(0, 1, 1), (1, 2, 1), (1, 3, 1)]),
        chain(vec![4, 8, 3, 4], vec![(0, 1, 1), (1, 2, 1), (2, 3, 1)]),
        // ethylamine, dimethylamine (C2H7N isomers)
        chain(vec![4, 3, 7], vec![(0, 1, 1), (1, 2, 1)]),
        chain(vec![4, 6, 4], vec![(0, 1, 1), (1, 2, 1)]),
        // cyclopropane, kekulized methylbenzene
        chain(vec![3, 3, 3], vec![(0, 1, 1), (1, 2, 1), (2, 0, 1)]),
        chain(
            vec![1, 2, 2, 2, 2, 2, 4],
            vec![
                (0, 1, 1),
                (1, 2, 2),
                (2, 3, 1),
                (3, 4, 2),
                (4, 5, 1),
                (5, 0, 2),
                (0, 6, 1),
            ],
        ),
    ]
}

fn trace_and_composition(graph: &MolGraph) -> (Vec<Token>, Composition) {
    let canonical = canonical_trace(graph, limits(), CANONICAL_WORK_LIMIT).unwrap();
    let composition = graph.composition();
    let end = replay_exact(&canonical.trace, limits(), composition).unwrap();
    assert!(end.stopped() && end.is_complete());
    (canonical.trace, composition)
}

/// Made-up evidence for the nine molecules. Molecule `i` owns three peaks of
/// its own, except that the first isomer pair (ethanol, dimethyl ether)
/// shares one peak list and one neutral mass and differs only in the adduct,
/// and the amine pair shares its adduct and mass and differs only in peaks.
fn nine_evidence() -> Vec<SpectrumEvidence> {
    let molecules = nine_molecules();
    (0..9usize)
        .map(|i| {
            let neutral = composition_mass(&molecules[i].composition()).unwrap();
            let adduct: u16 = match i {
                0 => 1,
                1 => 3,
                _ => 1,
            };
            let own = if i == 1 { 0 } else { i as u32 };
            let peaks = vec![
                (15_000_000 + 3_100_000 * own, 100.0),
                (22_000_000 + 2_700_000 * own, 40.0 + own as f32),
                (31_000_000 + 1_900_000 * own, 10.0),
            ];
            SpectrumEvidence {
                peaks,
                precursor_mz: precursor_mz_of(neutral, adduct).unwrap(),
                adduct,
                neutral_mass: neutral,
            }
        })
        .collect()
}

fn spectrum_config() -> CompletionModelConfig {
    let mut config = CompletionModelConfig::small();
    config.spectrum_slots = 8;
    config
}

fn train_config() -> CompletionTrainConfig {
    let extraction = ExtractionConfig {
        min_patterns: 0,
        max_patterns: 0,
        ..ExtractionConfig::default()
    };
    CompletionTrainConfig {
        lr: 3e-3,
        weight_decay: 0.0,
        grad_clip: None,
        seed: 1,
        extraction_seed: 11,
        pattern_source: PatternSource::RandomPatches(extraction.clone()),
        extraction,
        fingerprint_mode: None,
        fingerprint_threshold: 0.1,
    }
}

fn gen_config(trajectories: u32, seed: u64) -> CompletionGenerationConfig {
    CompletionGenerationConfig {
        trajectories,
        temperature: 1.0,
        seed,
        returned: 25,
        containment_node_limit: 100_000,
        identity_work_limit: 100_000,
        condition_on_patterns: true,
        substructure_semantics: SubstructureSemantics::Contained,
    }
}

#[test]
fn adduct_conversions_are_exact_integers_and_round_trip() {
    // Shifts against the defining exact masses, rounded once to micro-dalton.
    let electron = 0.000548579909065f64;
    let exact = [
        ("[M+H]+", 1.00782503223 - electron),
        ("[M-H]-", -(1.00782503223 - electron)),
        ("[M+Na]+", 22.98976928 - electron),
        ("[M+NH4]+", 14.00307400443 + 4.0 * 1.00782503223 - electron),
        ("[M+K]+", 38.96370649 - electron),
    ];
    assert_eq!(COMPLETION_ADDUCTS.len() + 1, SPECTRUM_ADDUCT_ROWS);
    for (name, shift) in exact {
        let adduct = completion_adduct_by_name(name).expect(name);
        assert_eq!(
            adduct.shift_uda,
            (shift * 1e6).round() as i64,
            "{name} shift"
        );
    }
    // Ids 1 and 2 are the V0 adducts.
    for v0 in mamba3::models::ms2::chem::ADDUCTS.iter() {
        assert_eq!(completion_adduct_by_name(v0.name).unwrap().id, v0.id);
    }
    // Ethanol C2H6O: 46.041865 neutral, 47.049141 as [M+H]+, 69.031086 as [M+Na]+.
    let ethanol = composition_mass(&nine_molecules()[0].composition()).unwrap();
    assert_eq!(ethanol, 46_041_865);
    assert_eq!(precursor_mz_of(ethanol, 1), Some(47_049_141));
    assert_eq!(precursor_mz_of(ethanol, 3), Some(69_031_086));
    for adduct in COMPLETION_ADDUCTS.iter() {
        let mz = precursor_mz_of(ethanol, adduct.id).unwrap();
        assert_eq!(
            neutral_mass_of(mz, adduct.id),
            Some(ethanol),
            "{} round trip",
            adduct.name
        );
    }
    // Unknown adducts and results outside the positive range are `None`.
    assert_eq!(neutral_mass_of(47_049_141, 0), None);
    assert_eq!(neutral_mass_of(47_049_141, 99), None);
    assert_eq!(neutral_mass_of(1_000_000, 3), None);
    assert_eq!(precursor_mz_of(500_000, 2), None);
    assert_eq!(precursor_mz_of(u32::MAX, 3), None);
}

#[test]
fn evidence_validation_and_peak_selection() {
    let evidence = SpectrumEvidence {
        peaks: vec![
            (50_000_000, 5.0),
            (20_000_000, 50.0),
            (70_000_000, 0.0),
            (30_000_000, 25.0),
            (40_000_000, 50.0),
        ],
        precursor_mz: 80_000_000,
        adduct: 1,
        neutral_mass: 78_992_724,
    };
    evidence.validate().unwrap();
    // All four positive peaks, ascending m/z, relative to the strongest.
    assert_eq!(
        evidence.selected(8),
        vec![
            (20_000_000, 1.0),
            (30_000_000, 0.5),
            (40_000_000, 1.0),
            (50_000_000, 0.1),
        ]
    );
    assert_eq!(evidence.dropped(8), 0);
    // Two slots keep the two strongest (the tie resolves to the lower m/z
    // first, but both tied peaks fit), still ascending in m/z.
    assert_eq!(
        evidence.selected(2),
        vec![(20_000_000, 1.0), (40_000_000, 1.0)]
    );
    assert_eq!(evidence.dropped(2), 2);
    // One slot: the tie between the two 50.0 peaks goes to the lower m/z.
    assert_eq!(evidence.selected(1), vec![(20_000_000, 1.0)]);
    // Input order does not matter.
    let mut reversed = evidence.clone();
    reversed.peaks.reverse();
    assert_eq!(reversed.selected(8), evidence.selected(8));

    let bad = |change: fn(&mut SpectrumEvidence)| {
        let mut item = evidence.clone();
        change(&mut item);
        item.validate().unwrap_err().to_string()
    };
    assert!(bad(|e| e.precursor_mz = 0).contains("precursor_mz"));
    assert!(bad(|e| e.neutral_mass = 0).contains("neutral_mass"));
    assert!(bad(|e| e.adduct = 6).contains("adduct id 6"));
    assert!(bad(|e| e.peaks[0].0 = 0).contains("m/z 0"));
    assert!(bad(|e| e.peaks[1].1 = f32::NAN).contains("intensity"));
    assert!(bad(|e| e.peaks[1].1 = -1.0).contains("intensity"));
    // Adduct 0 (unknown) and an empty peak list are valid evidence.
    let mut unknown = evidence.clone();
    unknown.adduct = 0;
    unknown.peaks.clear();
    unknown.validate().unwrap();
    assert!(unknown.selected(4).is_empty());
}

#[test]
fn batch_features_follow_the_encoder_contract() {
    let evidence = nine_evidence();
    let slots = 4usize;
    let batch =
        SpectrumBatch::build(&[Some(&evidence[2]), None, Some(&evidence[8])], slots).unwrap();
    batch.validate().unwrap();
    assert_eq!(batch.queries, 3);
    assert_eq!(batch.features.len(), 3 * slots * PEAK_FEATURES);
    assert_eq!(batch.meta.len(), 3 * META_FEATURES);
    assert_eq!(batch.present, vec![1.0, 0.0, 1.0]);
    assert_eq!(batch.adduct_ids, vec![1, 0, 1]);
    // Three peaks each: three valid slots, one padding slot.
    assert_eq!(&batch.valid[0..slots], &[1.0, 1.0, 1.0, 0.0]);
    assert_eq!(&batch.valid[slots..2 * slots], &[0.0; 4]);
    // The absent query is exact zeros everywhere.
    assert!(
        batch.features[slots * PEAK_FEATURES..2 * slots * PEAK_FEATURES]
            .iter()
            .all(|&v| v == 0.0)
    );
    assert!(
        batch.meta[META_FEATURES..2 * META_FEATURES]
            .iter()
            .all(|&v| v == 0.0)
    );
    // Query 0 against a direct call of the contract twins.
    let selected = evidence[2].selected(slots);
    let mut kept = vec![u32::MAX; slots * 3];
    let mut kept_f = vec![0.0f32; slots * 2];
    for (s, &(mz, relative)) in selected.iter().enumerate() {
        kept[s * 3] = s as u32;
        kept[s * 3 + 1] = mz;
        kept[s * 3 + 2] = (selected.len() - 1 - s) as u32;
        kept_f[s * 2] = relative;
        kept_f[s * 2 + 1] = 1.0;
    }
    let mut meta = vec![0u32; META_WIDTH];
    meta[1] = evidence[2].precursor_mz;
    let want = twin::peak_features(&kept, &kept_f, &meta, 1, slots);
    assert_eq!(&batch.features[0..slots * PEAK_FEATURES], want.as_slice());
    // Feature 0 is the m/z in kilo-dalton, feature 1 the neutral loss from
    // the precursor, feature 3 the relative intensity.
    let (mz0, rel0) = selected[0];
    assert_eq!(batch.features[0], mz0 as f32 * 1e-9);
    assert_eq!(
        batch.features[1],
        (evidence[2].precursor_mz as f32 - mz0 as f32) * 1e-9
    );
    assert_eq!(batch.features[3], rel0);
    // Mass features: the neutral mass sits where the contract keeps the
    // precursor, and there is no collision energy.
    let mut mass_meta = vec![0u32; META_WIDTH];
    mass_meta[1] = evidence[2].neutral_mass;
    let want_meta = twin::meta_features(&mass_meta, &[0.0, 0.0], 1);
    assert_eq!(&batch.meta[0..META_FEATURES], want_meta.as_slice());
    assert_eq!(batch.meta[0], 0.0);
    assert_eq!(batch.meta[1], evidence[2].neutral_mass as f32 * 1e-9);
    // More peaks than slots: only the strongest are kept.
    let narrow = SpectrumBatch::build(&[Some(&evidence[2])], 2).unwrap();
    assert_eq!(narrow.valid, vec![1.0, 1.0]);
    // Hand-edited arrays are rejected.
    let mut broken = batch.clone();
    broken.valid[slots] = 1.0;
    assert!(
        broken
            .validate()
            .unwrap_err()
            .to_string()
            .contains("no evidence")
    );
    let mut broken = batch.clone();
    broken.adduct_ids[0] = SPECTRUM_ADDUCT_ROWS as u32;
    assert!(broken.validate().is_err());
    assert!(SpectrumBatch::build(&[], slots).is_err());
}

#[test]
fn encoder_zeroes_padding_and_absent_queries_and_rows_are_independent() {
    let _lock = serial();
    let device = dev();
    let mut rng = Rng::seeded(41);
    let d = 16usize;
    let slots = 6usize;
    let encoder: SpectrumEncoder<R, E> = SpectrumEncoder::init(d, &device, &mut rng);
    check_launches(&device).unwrap();
    let evidence = nine_evidence();
    let pair = SpectrumBatch::build(&[Some(&evidence[3]), None], slots).unwrap();
    let out = encoder.encode(&pair, &device).unwrap();
    check_launches(&device).unwrap();
    let states = out.states.try_to_f32().unwrap();
    let pooled = out.pooled.try_to_f32().unwrap();
    assert!(states.iter().chain(pooled.iter()).all(|v| v.is_finite()));
    // Query 0 holds three peaks: its three padding slots are exact zeros.
    for s in 3..slots {
        assert!(
            states[s * d..(s + 1) * d].iter().all(|&v| v == 0.0),
            "padding slot {s}"
        );
    }
    assert!(states[0..3 * d].iter().any(|&v| v != 0.0));
    // The absent query is exact zeros in its states and its pooled vector.
    assert!(states[slots * d..].iter().all(|&v| v == 0.0));
    assert!(pooled[d..].iter().all(|&v| v == 0.0));
    assert!(pooled[..d].iter().any(|&v| v != 0.0));
    // Row independence: the same evidence alone gives the same row.
    let alone = SpectrumBatch::build(&[Some(&evidence[3])], slots).unwrap();
    let out_alone = encoder.encode(&alone, &device).unwrap();
    check_launches(&device).unwrap();
    let pooled_alone = out_alone.pooled.try_to_f32().unwrap();
    for (x, y) in pooled[..d].iter().zip(pooled_alone.iter()) {
        assert!(
            (x - y).abs() <= 1e-5 * y.abs().max(1.0),
            "row differs: {x} vs {y}"
        );
    }
    // The adduct and the mass reach the pooled vector on their own: no
    // peaks, two adducts, two masses.
    let pooled_of = |adduct: u16, neutral: u32| {
        let item = SpectrumEvidence {
            peaks: Vec::new(),
            precursor_mz: precursor_mz_of(neutral, adduct).unwrap(),
            adduct,
            neutral_mass: neutral,
        };
        let batch = SpectrumBatch::build(&[Some(&item)], slots).unwrap();
        let out = encoder.encode(&batch, &device).unwrap();
        check_launches(&device).unwrap();
        out.pooled.try_to_f32().unwrap()
    };
    let base = pooled_of(1, 180_063_388);
    assert!(base.iter().any(|&v| v != 0.0));
    assert_ne!(
        base,
        pooled_of(3, 180_063_388),
        "adduct changes the pooled vector"
    );
    assert_ne!(
        base,
        pooled_of(1, 194_079_038),
        "neutral mass changes the pooled vector"
    );
}

#[test]
fn absent_evidence_changes_nothing_and_missing_encoder_rejects_evidence() {
    let _lock = serial();
    let device = dev();
    let molecules = nine_molecules();
    let evidence = nine_evidence();
    let (trace, composition) = trace_and_composition(&molecules[2]);
    let patterns: Vec<&[MolGraph]> = vec![&[]];
    let traces = vec![trace.as_slice()];
    let compositions = vec![composition];
    let batch = PatternBatch::build(&patterns, &compositions).unwrap();
    let targets = TargetBatch::build_exact(&traces, &compositions, limits()).unwrap();
    let constants = Ms2Constants::new(&device);
    // The spectrum encoder is drawn after every other weight, so the same
    // seed gives the same substructure encoder and decoder with and without
    // it. Without evidence the encoder contributes exact zeros to the
    // composition row and fully masked memory slots, so the loss agrees.
    let plain_config = CompletionModelConfig::small();
    let mut rng = Rng::seeded(53);
    let plain: CompletionModel<R, E> =
        CompletionModel::init(&plain_config, &device, &mut rng).unwrap();
    let mut rng = Rng::seeded(53);
    let with: CompletionModel<R, E> =
        CompletionModel::init(&spectrum_config(), &device, &mut rng).unwrap();
    check_launches(&device).unwrap();
    assert!(with.has_spectrum() && !plain.has_spectrum());
    assert_eq!(with.spectrum_slots(), 8);
    let no_fp = FingerprintBatch::empty(1, 0).unwrap();
    let (_, loss_plain) = plain
        .teacher(&batch, &targets, &constants, &device)
        .unwrap();
    let (_, loss_absent) = with
        .teacher_with_evidence(&batch, &no_fp, None, &targets, &constants, &device)
        .unwrap();
    let absent_batch = SpectrumBatch::empty(1, 8).unwrap();
    let (_, loss_empty) = with
        .teacher_with_evidence(
            &batch,
            &no_fp,
            Some(&absent_batch),
            &targets,
            &constants,
            &device,
        )
        .unwrap();
    check_launches(&device).unwrap();
    let a = loss_plain.try_to_f32().unwrap()[0];
    let b = loss_absent.try_to_f32().unwrap()[0];
    let c = loss_empty.try_to_f32().unwrap()[0];
    assert!(
        (a - b).abs() <= 1e-4 * a.abs().max(1.0),
        "absent evidence changed the loss: {a} vs {b}"
    );
    assert_eq!(b, c, "None and an all-absent batch are the same input");
    // Present evidence does change it.
    let present = SpectrumBatch::build(&[Some(&evidence[2])], 8).unwrap();
    let (_, loss_present) = with
        .teacher_with_evidence(
            &batch,
            &no_fp,
            Some(&present),
            &targets,
            &constants,
            &device,
        )
        .unwrap();
    check_launches(&device).unwrap();
    assert_ne!(loss_present.try_to_f32().unwrap()[0], b);
    // A model without the encoder rejects evidence instead of ignoring it.
    let error = plain
        .teacher_with_evidence(
            &batch,
            &no_fp,
            Some(&present),
            &targets,
            &constants,
            &device,
        )
        .err()
        .expect("evidence without an encoder is an error")
        .to_string();
    assert!(error.contains("spectrum_slots = 0"), "{error}");
    // A slot count other than the model's is rejected too.
    let wrong = SpectrumBatch::build(&[Some(&evidence[2])], 4).unwrap();
    let error = with
        .teacher_with_evidence(&batch, &no_fp, Some(&wrong), &targets, &constants, &device)
        .err()
        .expect("wrong slot count is an error")
        .to_string();
    assert!(error.contains("4 slots"), "{error}");
    let mut config = spectrum_config();
    config.spectrum_slots = 513;
    assert!(config.validate().is_err());
}

#[test]
fn gradients_reach_every_spectrum_parameter() {
    let _lock = serial();
    let device = dev();
    let mut rng = Rng::seeded(61);
    let model: CompletionModel<R, E> =
        CompletionModel::init(&spectrum_config(), &device, &mut rng).unwrap();
    let molecules = nine_molecules();
    let evidence = nine_evidence();
    let pairs: Vec<(Vec<Token>, Composition)> = [0usize, 8]
        .iter()
        .map(|&i| trace_and_composition(&molecules[i]))
        .collect();
    let patterns: Vec<&[MolGraph]> = vec![&[], &[]];
    let traces: Vec<&[Token]> = pairs.iter().map(|p| p.0.as_slice()).collect();
    let compositions: Vec<Composition> = pairs.iter().map(|p| p.1).collect();
    let batch = PatternBatch::build(&patterns, &compositions).unwrap();
    let targets = TargetBatch::build_exact(&traces, &compositions, limits()).unwrap();
    let spectra = SpectrumBatch::build(&[Some(&evidence[0]), Some(&evidence[8])], 8).unwrap();
    let no_fp = FingerprintBatch::empty(2, 0).unwrap();
    let constants = Ms2Constants::new(&device);
    let (_, loss) = model
        .teacher_with_evidence(
            &batch,
            &no_fp,
            Some(&spectra),
            &targets,
            &constants,
            &device,
        )
        .unwrap();
    check_launches(&device).unwrap();
    let grads = loss.backward_retain().unwrap();
    check_launches(&device).unwrap();
    let named = model.named_parameters();
    let spectrum: Vec<_> = named
        .iter()
        .filter(|(name, _)| name.starts_with("spectrum_encoder."))
        .collect();
    // peak_in, peak_out, meta_in, meta_out, pool_in (weight + bias each), the
    // adduct table, and two rounds of three linears plus a norm weight.
    for expected in [
        "spectrum_encoder.peak_in.weight",
        "spectrum_encoder.peak_out.weight",
        "spectrum_encoder.adduct_emb",
        "spectrum_encoder.meta_in.weight",
        "spectrum_encoder.meta_out.weight",
        "spectrum_encoder.pool_in.weight",
        "spectrum_encoder.round.0.own.weight",
        "spectrum_encoder.round.1.mean.weight",
    ] {
        assert!(
            spectrum.iter().any(|(name, _)| name == expected),
            "{expected} is a named parameter; have {:?}",
            spectrum.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>()
        );
    }
    for (name, param) in &spectrum {
        let grad = grads
            .get(param.id())
            .unwrap_or_else(|| panic!("{name} has no gradient"));
        let values = grad.try_to_f32().unwrap();
        assert!(
            values.iter().all(|v| v.is_finite()),
            "{name} gradient is not finite"
        );
        assert!(
            values.iter().any(|&v| v != 0.0),
            "{name} gradient is all zero"
        );
    }
    check_launches(&device).unwrap();
}

#[test]
fn spectrum_overfits_nine_molecules_and_generation_follows_the_evidence() {
    let _lock = serial();
    let device = dev();
    let molecules = nine_molecules();
    let evidence = nine_evidence();
    let pairs: Vec<(Vec<Token>, Composition)> =
        molecules.iter().map(trace_and_composition).collect();
    let empty: Vec<Vec<MolGraph>> = molecules.iter().map(|_| Vec::new()).collect();
    let refs: Vec<&[MolGraph]> = empty.iter().map(Vec::as_slice).collect();
    let traces: Vec<&[Token]> = pairs.iter().map(|p| p.0.as_slice()).collect();
    let compositions: Vec<Composition> = pairs.iter().map(|p| p.1).collect();
    let spectra: Vec<Option<&SpectrumEvidence>> = evidence.iter().map(Some).collect();
    let mut trainer =
        CompletionTrainer::<R, E>::new(&spectrum_config(), &train_config(), &device).unwrap();
    let mean =
        |values: &[f32]| values.iter().map(|&v| f64::from(v)).sum::<f64>() / values.len() as f64;
    let initial = mean(
        &trainer
            .teacher_eval_with_evidence(&refs, None, &spectra, &traces, &compositions)
            .unwrap(),
    );
    check_launches(&device).unwrap();
    let mut last = initial;
    for step in 0..400 {
        trainer
            .step_with_evidence(&refs, None, &spectra, &traces, &compositions)
            .unwrap();
        if (step + 1) % 20 == 0 {
            check_launches(&device).unwrap();
            last = mean(
                &trainer
                    .teacher_eval_with_evidence(&refs, None, &spectra, &traces, &compositions)
                    .unwrap(),
            );
            // Stop before every NLL saturates at 0 and the comparisons tie.
            if last < 0.15 * initial {
                break;
            }
        }
    }
    println!("spectrum overfit: initial {initial:.3} final {last:.3}");
    assert!(
        last < 0.30 * initial,
        "trained NLL {last:.3} is not below 30% of {initial:.3}"
    );
    // Own-versus-other evidence. The C2H6O pair shares its peaks and mass
    // and differs only in the adduct; the C2H7N pair shares adduct and mass
    // and differs only in peaks; the C3H8O molecules differ in peaks.
    let mut nll_with = |molecule: usize, source: usize| -> f32 {
        trainer
            .teacher_eval_with_evidence(
                &[&[]],
                None,
                &[Some(&evidence[source])],
                &[pairs[molecule].0.as_slice()],
                &[pairs[molecule].1],
            )
            .unwrap()[0]
    };
    for (molecule, other) in [
        (0usize, 1usize),
        (1, 0),
        (5, 6),
        (6, 5),
        (2, 3),
        (3, 4),
        (4, 2),
    ] {
        let own = nll_with(molecule, molecule);
        let foreign = nll_with(molecule, other);
        assert!(
            own < foreign,
            "molecule {molecule}: NLL {own} with its own evidence is not below {foreign} with molecule {other}'s"
        );
    }
    check_launches(&device).unwrap();
    // Sampling: the same composition with each isomer's evidence returns
    // that isomer first, and every returned candidate has exactly the
    // requested composition (the mass constraint is the grammar's).
    let model = trainer.model();
    let constants = Ms2Constants::new(&device);
    let requests: Vec<CompletionRequest> = [0usize, 1]
        .iter()
        .map(|&i| CompletionRequest {
            id: 100 + i as u64,
            composition: compositions[i],
            patterns: &[],
            acceptance_patterns: None,
            fingerprint: None,
        })
        .collect();
    let chosen = [Some(&evidence[0]), Some(&evidence[1])];
    let outcomes = model
        .generate_with_spectra(&requests, &chosen, &gen_config(32, 7), &constants, &device)
        .unwrap();
    check_launches(&device).unwrap();
    assert_eq!(compositions[0], compositions[1], "the pair is isomeric");
    for (i, outcome) in outcomes.iter().enumerate() {
        assert!(
            !outcome.candidates.is_empty(),
            "query {i} returned no candidate"
        );
        for candidate in &outcome.candidates {
            assert_eq!(
                candidate.graph.composition(),
                compositions[i],
                "candidate composition"
            );
            assert_eq!(
                composition_mass(&candidate.graph.composition()).unwrap(),
                evidence[i].neutral_mass,
                "candidate mass equals the input neutral mass"
            );
        }
        assert_eq!(
            same_identity(&outcome.candidates[0].graph, &molecules[i], 100_000),
            Some(true),
            "query {i}: the top candidate is the molecule its evidence was trained with"
        );
    }
    // The evidence is what separates them: the two outcomes differ at rank 1.
    assert_eq!(
        same_identity(
            &outcomes[0].candidates[0].graph,
            &outcomes[1].candidates[0].graph,
            100_000
        ),
        Some(false)
    );
    // Evidence for a model without the encoder is rejected by `generate`.
    let mut rng = Rng::seeded(3);
    let plain: CompletionModel<R, E> =
        CompletionModel::init(&CompletionModelConfig::small(), &device, &mut rng).unwrap();
    let error = plain
        .generate_with_spectra(&requests, &chosen, &gen_config(4, 7), &constants, &device)
        .err()
        .expect("evidence without an encoder is an error")
        .to_string();
    assert!(error.contains("no spectrum encoder"), "{error}");
    let error = model
        .generate_with_spectra(
            &requests,
            &chosen[..1],
            &gen_config(4, 7),
            &constants,
            &device,
        )
        .err()
        .expect("length mismatch is an error")
        .to_string();
    assert!(error.contains("1 spectra for 2 requests"), "{error}");
}

/// A `molecular-completion-generate-v1` request for C2H6O with no
/// substructures and the given spectral evidence.
fn json_request(id: &str, spectrum: serde_json::Value) -> serde_json::Value {
    serde_json::json!({
        "protocol": "molecular-completion-generate-v1",
        "id": id,
        "provenance": "synthetic test",
        "mass_role": "target_molecule",
        "composition": {"C": 2, "H": 6, "O": 1},
        "substructures": [],
        "generation": {"trajectories": 32, "temperature": 1.0, "seed": 7, "returned": 25},
        "spectrum": spectrum,
    })
}

fn json_spectrum(evidence: &SpectrumEvidence, adduct: &str) -> serde_json::Value {
    serde_json::json!({
        "peaks": evidence.peaks.iter().map(|&(mz, i)| serde_json::json!([mz, i])).collect::<Vec<_>>(),
        "precursor_mz_uda": evidence.precursor_mz,
        "adduct": adduct,
    })
}

#[test]
fn json_protocol_carries_spectral_evidence() {
    use mamba3::models::ms2::completion_api::CompletionService;
    let _lock = serial();
    let device = dev();
    let molecules = nine_molecules();
    let evidence = nine_evidence();
    // The committed fixture model has no spectrum encoder: evidence is
    // `unsupported_input` naming the field, never ignored.
    let fixture = format!(
        "{}/tests/fixtures/ms2/completion_tiny.ckpt",
        env!("CARGO_MANIFEST_DIR")
    );
    let plain = CompletionService::<R>::load(std::path::Path::new(&fixture), &device).unwrap();
    let request = json_request("spectrum-on-plain", json_spectrum(&evidence[0], "[M+H]+"));
    let out: serde_json::Value =
        serde_json::from_str(&plain.generate_json(&request.to_string()).unwrap()).unwrap();
    assert_eq!(out["status"], "unsupported_input", "{out}");
    assert_eq!(out["unsupported"]["limit"], "spectrum");
    assert_eq!(out["accounting"]["trajectories"], 0);
    check_launches(&device).unwrap();
    // Schema rules.
    let rejected = |change: fn(&mut serde_json::Value)| -> String {
        let mut request = json_request("bad", json_spectrum(&nine_evidence()[0], "[M+H]+"));
        change(&mut request["spectrum"]);
        plain
            .generate_json(&request.to_string())
            .unwrap_err()
            .to_string()
    };
    assert!(rejected(|s| s["extra"] = serde_json::json!(1)).contains("extra"));
    assert!(rejected(|s| s["adduct"] = serde_json::json!("[M+Li]+")).contains("[M+Li]+"));
    assert!(
        rejected(|s| s["adduct"] = serde_json::json!("unknown"))
            .contains("neutral_mass_uda is required")
    );
    assert!(rejected(|s| s["peaks"][0][0] = serde_json::json!(0)).contains("peaks[0][0]"));
    assert!(rejected(|s| s["peaks"][1][1] = serde_json::json!(-1.0)).contains("peaks[1][1]"));
    assert!(
        rejected(|s| s["precursor_mz_uda"] = serde_json::json!(0)).contains("precursor_mz_uda")
    );
    assert!(rejected(|s| s["peaks"] = serde_json::json!("none")).contains("peaks must be a list"));

    // A model with the encoder, trained as in the overfit test, served
    // through the JSON protocol: each isomer's evidence returns that isomer.
    let pairs: Vec<(Vec<Token>, Composition)> =
        molecules.iter().map(trace_and_composition).collect();
    let empty: Vec<Vec<MolGraph>> = molecules.iter().map(|_| Vec::new()).collect();
    let refs: Vec<&[MolGraph]> = empty.iter().map(Vec::as_slice).collect();
    let traces: Vec<&[Token]> = pairs.iter().map(|p| p.0.as_slice()).collect();
    let compositions: Vec<Composition> = pairs.iter().map(|p| p.1).collect();
    let spectra: Vec<Option<&SpectrumEvidence>> = evidence.iter().map(Some).collect();
    let mut trainer =
        CompletionTrainer::<R, E>::new(&spectrum_config(), &train_config(), &device).unwrap();
    for _ in 0..300 {
        trainer
            .step_with_evidence(&refs, None, &spectra, &traces, &compositions)
            .unwrap();
    }
    check_launches(&device).unwrap();
    let dir = std::env::temp_dir().join(format!("mamba3-spectrum-json-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("spectrum.ckpt");
    trainer.save(&path).unwrap();
    let service = CompletionService::<R>::load(&path, &device).unwrap();
    let top_of = |index: usize, adduct: &str| -> (serde_json::Value, MolGraph) {
        let request = json_request(
            &format!("isomer-{index}"),
            json_spectrum(&evidence[index], adduct),
        );
        let out: serde_json::Value =
            serde_json::from_str(&service.generate_json(&request.to_string()).unwrap()).unwrap();
        assert_eq!(out["status"], "ok", "{out}");
        let top = &out["candidates"][0];
        let atoms: Vec<u8> = top["atoms"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u8)
            .collect();
        let bonds: Vec<(usize, usize, u8)> = top["bonds"]
            .as_array()
            .unwrap()
            .iter()
            .map(|b| {
                (
                    b[0].as_u64().unwrap() as usize,
                    b[1].as_u64().unwrap() as usize,
                    b[2].as_u64().unwrap() as u8,
                )
            })
            .collect();
        (out.clone(), MolGraph::new(atoms, bonds).unwrap())
    };
    let (out_ethanol, top_ethanol) = top_of(0, "[M+H]+");
    let (out_ether, top_ether) = top_of(1, "[M+Na]+");
    check_launches(&device).unwrap();
    assert_eq!(
        same_identity(&top_ethanol, &molecules[0], 100_000),
        Some(true)
    );
    assert_eq!(
        same_identity(&top_ether, &molecules[1], 100_000),
        Some(true)
    );
    // The echo: peaks used, the adduct, and the neutral mass derived from
    // the precursor because the request omitted it.
    assert_eq!(out_ethanol["spectrum"]["peaks_used"], 3);
    assert_eq!(out_ethanol["spectrum"]["peaks_dropped"], 0);
    assert_eq!(out_ethanol["spectrum"]["adduct"], "[M+H]+");
    assert_eq!(
        out_ethanol["spectrum"]["neutral_mass_uda"],
        evidence[0].neutral_mass
    );
    assert_eq!(out_ether["spectrum"]["adduct"], "[M+Na]+");
    assert_eq!(
        out_ether["spectrum"]["neutral_mass_uda"],
        evidence[1].neutral_mass
    );
    // The evidence is part of the request hash, and a request without it
    // still works on the same model and carries no echo.
    assert_ne!(out_ethanol["input_hash"], out_ether["input_hash"]);
    let mut bare = json_request("bare", serde_json::json!(null));
    bare.as_object_mut().unwrap().remove("spectrum");
    let out: serde_json::Value =
        serde_json::from_str(&service.generate_json(&bare.to_string()).unwrap()).unwrap();
    assert_eq!(out["status"], "ok");
    assert!(out.get("spectrum").is_none());
    std::fs::remove_dir_all(&dir).unwrap();
}
