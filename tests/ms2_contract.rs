//! P1-B tests: `SpectrumBatch` validation, schema round-trips and the
//! generation/candidate/model/domain configs of contract §§3 and 8.

use mamba3::models::ms2::contract::{
    CandidateBatch, ChemistryDomain, Control, GenerationConfig, GenerationMode, ModelConfig,
    NO_FORMULA, SCHEMA_VERSION, SpectrumBatch, candidate_status, request_status,
};

/// A valid 2-spectrum batch: spectrum 0 exercises a known energy and an
/// explicit fragment tolerance, spectrum 1 an unknown energy and defaults.
fn valid_batch() -> SpectrumBatch {
    let n_raw = 64usize;
    let mut peak_id = vec![u32::MAX; 2 * n_raw];
    peak_id[0..3].copy_from_slice(&[0, 1, 2]);
    peak_id[n_raw..n_raw + 2].copy_from_slice(&[0, 1]);
    let mut mz = vec![0u32; 2 * n_raw];
    mz[0..3].copy_from_slice(&[100_000_000, 200_000_000, 300_000_000]);
    mz[n_raw..n_raw + 2].copy_from_slice(&[150_000_000, 250_000_000]);
    let mut intensity = vec![0.0f32; 2 * n_raw];
    intensity[0..3].copy_from_slice(&[1.0, 0.5, 0.25]);
    intensity[n_raw..n_raw + 2].copy_from_slice(&[1.0, 0.8]);
    SpectrumBatch {
        schema_version: SCHEMA_VERSION,
        n_raw: n_raw as u32,
        spectrum_id: vec![11, 22],
        raw_peak_count: vec![3, 2],
        peak_count: vec![3, 2],
        peak_id,
        mz_udalton: mz,
        intensity,
        intensity_scale: 0,
        mz_uncertainty_udalton: vec![50, 50],
        precursor_mz_udalton: vec![400_000_000, 500_000_000],
        precursor_uncertainty_udalton: vec![50, 50],
        adduct: vec![1, 2],
        polarity: vec![1, -1],
        collision_energy_ev: vec![30.0, 0.0],
        collision_energy_known: vec![1, 0],
        energy_count: vec![1, 0],
        fragment_tolerance_ppm_tenths: vec![100, 0],
        precursor_tolerance_ppm_tenths: vec![0, 200],
        instrument_class: vec![1, 0],
    }
}

#[test]
fn valid_batch_has_no_status() {
    let b = valid_batch();
    assert_eq!(b.len(), 2);
    assert!(!b.is_empty());
    assert_eq!(b.validate().unwrap(), vec![0, 0]);
    assert_eq!(b.fragment_tolerance(0), 100);
    assert_eq!(b.fragment_tolerance(1), 100);
    assert_eq!(b.precursor_tolerance(0), 200);
    assert_eq!(b.precursor_tolerance(1), 200);
}

#[test]
fn intensity_defects() {
    use request_status as s;
    // NaN intensity.
    let mut b = valid_batch();
    b.intensity[1] = f32::NAN;
    assert_eq!(b.validate().unwrap(), vec![s::NONFINITE_INPUT, 0]);
    // Infinite intensity.
    let mut b = valid_batch();
    b.intensity[1] = f32::INFINITY;
    assert_eq!(b.validate().unwrap(), vec![s::NONFINITE_INPUT, 0]);
    // Negative intensity.
    let mut b = valid_batch();
    b.intensity[1] = -0.5;
    assert_eq!(b.validate().unwrap(), vec![s::NEGATIVE_INTENSITY, 0]);
    // All-zero intensities.
    let mut b = valid_batch();
    b.intensity[0..3].copy_from_slice(&[0.0, 0.0, 0.0]);
    assert_eq!(b.validate().unwrap(), vec![s::EMPTY_SPECTRUM, 0]);
    // No peaks at all.
    let mut b = valid_batch();
    b.peak_count[0] = 0;
    b.raw_peak_count[0] = 0;
    assert_eq!(b.validate().unwrap(), vec![s::EMPTY_SPECTRUM, 0]);
}

#[test]
fn peak_metadata_defects() {
    use request_status as s;
    // A valid peak with m/z 0.
    let mut b = valid_batch();
    b.mz_udalton[1] = 0;
    assert_eq!(b.validate().unwrap(), vec![s::INVALID_PEAK, 0]);
    // Unknown adduct.
    let mut b = valid_batch();
    b.adduct[0] = 0;
    assert_eq!(b.validate().unwrap(), vec![s::INSUFFICIENT_METADATA, 0]);
    // Adduct outside the domain.
    let mut b = valid_batch();
    b.adduct[0] = 99;
    assert_eq!(b.validate().unwrap(), vec![s::UNSUPPORTED_ADDUCT, 0]);
    // Invalid polarity.
    let mut b = valid_batch();
    b.polarity[0] = 0;
    assert_eq!(b.validate().unwrap(), vec![s::INVALID_POLARITY, 0]);
    // Polarity/adduct sign conflict (negative polarity with [M+H]+).
    let mut b = valid_batch();
    b.polarity[0] = -1;
    assert_eq!(b.validate().unwrap(), vec![s::POLARITY_ADDUCT_CONFLICT, 0]);
    // No conflict check against an unknown adduct: only the metadata bit.
    let mut b = valid_batch();
    b.polarity[0] = -1;
    b.adduct[0] = 0;
    assert_eq!(b.validate().unwrap(), vec![s::INSUFFICIENT_METADATA, 0]);
}

#[test]
fn energy_known_flag_gates_the_value() {
    use request_status as s;
    // Unknown energy with a stored 0.0: not read, no bit.
    let b = valid_batch();
    assert_eq!(b.validate().unwrap()[1], 0);
    // Known energy of exactly 0.0 eV: a measured zero, valid.
    let mut b = valid_batch();
    b.collision_energy_known[0] = 1;
    b.collision_energy_ev[0] = 0.0;
    assert_eq!(b.validate().unwrap()[0], 0);
    // Known NaN energy.
    let mut b = valid_batch();
    b.collision_energy_ev[0] = f32::NAN;
    assert_eq!(b.validate().unwrap(), vec![s::NONFINITE_INPUT, 0]);
    // Known negative energy.
    let mut b = valid_batch();
    b.collision_energy_ev[0] = -1.0;
    assert_eq!(b.validate().unwrap(), vec![s::NONFINITE_INPUT, 0]);
}

#[test]
fn precursor_range_and_capacity() {
    use request_status as s;
    for precursor in [49_999_999u32, 2_000_000_001] {
        let mut b = valid_batch();
        b.precursor_mz_udalton[0] = precursor;
        assert_eq!(
            b.validate().unwrap(),
            vec![s::PRECURSOR_OUT_OF_RANGE, 0],
            "precursor {precursor}"
        );
    }
    // Boundaries are inclusive.
    for precursor in [50_000_000u32, 2_000_000_000] {
        let mut b = valid_batch();
        b.precursor_mz_udalton[0] = precursor;
        assert_eq!(b.validate().unwrap()[0], 0, "precursor {precursor}");
    }
    // peak_count past n_raw: over capacity, checked over the first n_raw.
    let mut b = valid_batch();
    let n_raw = b.n_raw as usize;
    b.peak_count[0] = n_raw as u32 + 1;
    b.raw_peak_count[0] = n_raw as u32 + 1;
    for k in 0..n_raw {
        b.peak_id[k] = k as u32;
        b.mz_udalton[k] = 100_000_000 + k as u32;
        b.intensity[k] = 1.0;
    }
    assert_eq!(b.validate().unwrap(), vec![s::OVER_CAPACITY, 0]);
}

#[test]
fn warnings_are_not_fatal() {
    use request_status as s;
    // More raw peaks than supplied: truncation warning only.
    let mut b = valid_batch();
    b.raw_peak_count[0] = 10;
    assert_eq!(b.validate().unwrap(), vec![s::RAW_TRUNCATED, 0]);
    // Unknown peak precision: warning only.
    let mut b = valid_batch();
    b.mz_uncertainty_udalton[1] = u32::MAX;
    assert_eq!(b.validate().unwrap(), vec![0, s::EXACT_MASS_UNAVAILABLE]);
    // Unknown precursor precision: warning only.
    let mut b = valid_batch();
    b.precursor_uncertainty_udalton[1] = u32::MAX;
    assert_eq!(b.validate().unwrap(), vec![0, s::EXACT_MASS_UNAVAILABLE]);
    // Two peaks at one m/z are legal.
    let mut b = valid_batch();
    b.mz_udalton[1] = b.mz_udalton[0];
    assert_eq!(b.validate().unwrap(), vec![0, 0]);
}

#[test]
fn padding_is_never_read() {
    // Poison in every padding slot: NaN intensity and m/z 0 past peak_count.
    let mut b = valid_batch();
    let n_raw = b.n_raw as usize;
    for k in 3..n_raw {
        b.intensity[k] = f32::NAN;
        b.mz_udalton[k] = 0;
        b.peak_id[k] = 0;
    }
    for k in 2..n_raw {
        b.intensity[n_raw + k] = f32::NAN;
        b.mz_udalton[n_raw + k] = 0;
        b.peak_id[n_raw + k] = 0;
    }
    assert_eq!(b.validate().unwrap(), vec![0, 0]);
}

#[test]
fn defects_combine() {
    use request_status as s;
    let mut b = valid_batch();
    b.intensity[0] = -1.0;
    b.mz_udalton[1] = 0;
    b.polarity[0] = 0;
    let bits = b.validate().unwrap()[0];
    assert_eq!(
        bits,
        s::NEGATIVE_INTENSITY | s::INVALID_PEAK | s::INVALID_POLARITY
    );
    assert_eq!(
        request_status::names(bits),
        vec!["negative_intensity", "invalid_peak", "invalid_polarity"]
    );
}

#[test]
fn malformed_batches_are_config_errors() {
    let mut b = valid_batch();
    b.schema_version = 2;
    let err = b.validate().unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("schema_version"), "{msg}");
    assert!(msg.contains('2') && msg.contains('1'), "{msg}");

    let mut b = valid_batch();
    b.n_raw = 100;
    let err = b.validate().unwrap_err();
    assert!(err.to_string().contains("n_raw"), "{err}");

    let mut b = valid_batch();
    b.peak_id.pop();
    let err = b.validate().unwrap_err();
    assert!(err.to_string().contains("peak_id"), "{err}");

    let mut b = valid_batch();
    b.intensity.pop();
    let err = b.validate().unwrap_err();
    assert!(err.to_string().contains("intensity"), "{err}");

    let mut b = valid_batch();
    b.spectrum_id[1] = b.spectrum_id[0];
    let err = b.validate().unwrap_err();
    assert!(err.to_string().contains("spectrum_id"), "{err}");

    let mut b = valid_batch();
    b.peak_id[1] = b.peak_id[0];
    let err = b.validate().unwrap_err();
    assert!(err.to_string().contains("peak_id"), "{err}");

    let mut b = valid_batch();
    b.intensity_scale = 2;
    let err = b.validate().unwrap_err();
    assert!(err.to_string().contains("intensity_scale"), "{err}");

    let mut b = valid_batch();
    b.collision_energy_known[0] = 2;
    let err = b.validate().unwrap_err();
    assert!(err.to_string().contains("collision_energy_known"), "{err}");

    let mut b = valid_batch();
    b.energy_count[0] = 9;
    let err = b.validate().unwrap_err();
    assert!(err.to_string().contains("energy_count"), "{err}");

    let mut b = valid_batch();
    b.fragment_tolerance_ppm_tenths[0] = 1001;
    let err = b.validate().unwrap_err();
    assert!(
        err.to_string().contains("fragment_tolerance_ppm_tenths"),
        "{err}"
    );

    let mut b = valid_batch();
    b.precursor_tolerance_ppm_tenths[1] = 1001;
    let err = b.validate().unwrap_err();
    assert!(
        err.to_string().contains("precursor_tolerance_ppm_tenths"),
        "{err}"
    );

    let mut b = valid_batch();
    b.instrument_class[0] = 5;
    let err = b.validate().unwrap_err();
    assert!(err.to_string().contains("instrument_class"), "{err}");
}

#[test]
fn schemas_round_trip() {
    let b = valid_batch();
    let back: SpectrumBatch = serde_json::from_str(&serde_json::to_string(&b).unwrap()).unwrap();
    assert_eq!(back, b);

    let g = GenerationConfig::default();
    assert_eq!(g.schema_version, SCHEMA_VERSION);
    assert_eq!(g.formula_rows_visited_max, u32::MAX);
    let back: GenerationConfig = serde_json::from_str(&serde_json::to_string(&g).unwrap()).unwrap();
    assert_eq!(back, g);

    let c = CandidateBatch::empty(&[7, 8], 2, 4, 3, 4);
    assert_eq!(c.batch, 2);
    assert_eq!(c.trajectory, vec![0, 1, 0, 1]);
    assert_eq!(c.spectrum_id, vec![7, 7, 8, 8]);
    assert_eq!(c.max_ring_closures, 4);
    assert!(c.length.iter().all(|&l| l == 0));
    assert!(c.formula_row.iter().all(|&r| r == NO_FORMULA));
    assert!(
        c.status
            .iter()
            .all(|&s| s == candidate_status::REQUEST_FAILED)
    );
    assert!(c.formula_support_complete.iter().all(|&v| v == 0));
    assert!(c.formula_mass_retained.iter().all(|&v| v == 0.0));
    let back: CandidateBatch = serde_json::from_str(&serde_json::to_string(&c).unwrap()).unwrap();
    assert_eq!(back, c);

    let m = ModelConfig::v0();
    assert_eq!(m.schema_version, SCHEMA_VERSION);
    let back: ModelConfig = serde_json::from_str(&serde_json::to_string(&m).unwrap()).unwrap();
    assert_eq!(back, m);

    let d = ChemistryDomain::v0();
    assert_eq!(d.schema_version, SCHEMA_VERSION);
    let back: ChemistryDomain = serde_json::from_str(&serde_json::to_string(&d).unwrap()).unwrap();
    assert_eq!(back, d);
    assert!(d.check_compatible(&back).is_ok());
    assert!(d.validate().is_ok());
}

#[test]
fn every_schema_rejects_version_2() {
    let mut b = valid_batch();
    b.schema_version = 2;
    let err = b.validate().unwrap_err();
    assert!(err.to_string().contains("schema_version"), "{err}");
    assert!(
        err.to_string().contains('2') && err.to_string().contains('1'),
        "{err}"
    );

    let mut g = GenerationConfig::default();
    g.schema_version = 2;
    let err = g.validate(16, 4).unwrap_err();
    assert!(err.to_string().contains("schema_version"), "{err}");
    assert!(
        err.to_string().contains('2') && err.to_string().contains('1'),
        "{err}"
    );

    let mut c = CandidateBatch::empty(&[7], 1, 4, 2, 0);
    c.schema_version = 2;
    let err = c.validate().unwrap_err();
    assert!(err.to_string().contains("schema_version"), "{err}");
    assert!(
        err.to_string().contains('2') && err.to_string().contains('1'),
        "{err}"
    );

    let mut m = ModelConfig::v0();
    m.schema_version = 2;
    let err = m.validate().unwrap_err();
    assert!(err.to_string().contains("schema_version"), "{err}");
    assert!(
        err.to_string().contains('2') && err.to_string().contains('1'),
        "{err}"
    );

    let mut d = ChemistryDomain::v0();
    d.schema_version = 2;
    let err = d.validate().unwrap_err();
    assert!(err.to_string().contains("schema_version"), "{err}");
    assert!(
        err.to_string().contains('2') && err.to_string().contains('1'),
        "{err}"
    );
}

#[test]
fn generation_config_ranges() {
    assert!(GenerationConfig::default().validate(16, 4).is_ok());
    let mut g = GenerationConfig::default();
    g.trajectories = 0;
    assert!(g.validate(16, 4).is_err());
    g.trajectories = 65;
    assert!(g.validate(16, 4).is_err());
    g = GenerationConfig::default();
    g.formulas = 0;
    assert!(g.validate(16, 4).is_err());
    g.formulas = 9;
    assert!(g.validate(16, 4).is_err());
    for temperature in [0.0f32, 4.5, f32::NAN, f32::INFINITY] {
        g = GenerationConfig::default();
        g.temperature = temperature;
        assert!(g.validate(16, 4).is_err(), "temperature {temperature}");
    }
    g = GenerationConfig::default();
    g.max_steps = 21;
    assert!(g.validate(16, 4).is_err());
    g.max_steps = 65;
    assert!(g.validate(16, 4).is_err());
    g = GenerationConfig::default();
    g.mode = GenerationMode::Beam;
    let err = g.validate(16, 4).unwrap_err();
    assert!(matches!(err, mamba3::error::Error::Unsupported(_)), "{err}");
    // The step floor follows the structure limits: 2 + 16 + 4 = 22.
    g = GenerationConfig::default();
    assert!(g.validate(32, 8).is_err());
    g.max_steps = 42;
    assert!(g.validate(32, 8).is_ok());
    // Control variants validate like None.
    for control in [
        Control::None,
        Control::ShuffledSpectrum,
        Control::MetadataOnly,
    ] {
        g = GenerationConfig::default();
        g.control = control;
        assert!(g.validate(16, 4).is_ok());
    }
    // The default visited limit is u32::MAX (no limit).
    assert_eq!(
        GenerationConfig::default().formula_rows_visited_max,
        u32::MAX
    );
}

#[test]
fn structure_prior_control_schema() {
    // The structure prior serializes as `structure_prior`, validates like the
    // other controls and round-trips through `GenerationConfig`.
    assert_eq!(
        serde_json::to_string(&Control::StructurePrior).unwrap(),
        "\"structure_prior\""
    );
    let back: Control = serde_json::from_str("\"structure_prior\"").unwrap();
    assert_eq!(back, Control::StructurePrior);
    let mut g = GenerationConfig::default();
    g.control = Control::StructurePrior;
    assert!(g.validate(16, 4).is_ok());
    let back: GenerationConfig = serde_json::from_str(&serde_json::to_string(&g).unwrap()).unwrap();
    assert_eq!(back, g);
}

#[test]
fn generation_config_checked_conversions() {
    use mamba3::error::Error;
    // A `usize` that does not fit `u32`, or whose sum overflows, is a config
    // error, never a truncation (no panic in debug or release).
    let g = GenerationConfig::default();
    let err = g.validate(usize::MAX, 4).unwrap_err();
    assert!(matches!(err, Error::Config(_)), "{err}");
    let err = g.validate(16, usize::MAX).unwrap_err();
    assert!(matches!(err, Error::Config(_)), "{err}");
}

#[test]
fn spectrum_batch_peak_counts_and_ids() {
    // raw_peak_count below peak_count is malformed.
    let mut b = valid_batch();
    b.raw_peak_count[0] = 2;
    assert_eq!(b.peak_count[0], 3);
    let err = b.validate().unwrap_err();
    assert!(err.to_string().contains("raw_peak_count"), "{err}");
    // A valid peak id at or above raw_peak_count is malformed.
    let mut b = valid_batch();
    b.peak_id[2] = 3;
    let err = b.validate().unwrap_err();
    assert!(err.to_string().contains("peak_id"), "{err}");
    // So is a valid peak carrying the u32::MAX padding id.
    let mut b = valid_batch();
    b.peak_count[0] = 1;
    b.raw_peak_count[0] = 1;
    b.peak_id[0] = u32::MAX;
    let err = b.validate().unwrap_err();
    assert!(err.to_string().contains("peak_id"), "{err}");
}

fn finished_record() -> CandidateBatch {
    // One spectrum, one trajectory: START, ADD_ATOM C H3 (root),
    // ADD_ATOM C H3 with a single bond to atom 0, STOP. Both residual
    // valences are 0, so the open valence is `[0, 0]`.
    let mut c = CandidateBatch::empty(&[7], 1, 5, 2, 0);
    c.length[0] = 4;
    c.actions[0..16].copy_from_slice(&[1, 0, 0, 0, 2, 4, 0, 0, 2, 4, 1, 0, 4, 0, 0, 0]);
    c.status[0] = candidate_status::FINISHED;
    c
}

#[test]
fn candidate_batch_empty_validates() {
    // A failed request carries a fatal status on every spectrum.
    let mut c = CandidateBatch::empty(&[7, 8], 2, 4, 3, 4);
    c.request_status = vec![
        request_status::EMPTY_SPECTRUM,
        request_status::OVER_CAPACITY,
    ];
    c.validate().unwrap();
    CandidateBatch::empty(&[], 2, 4, 3, 4).validate().unwrap();
    finished_record().validate().unwrap();
}

#[test]
fn candidate_batch_violations() {
    // Non-PAD past length.
    let mut c = finished_record();
    c.actions[16] = 2;
    assert!(c.validate().is_err());
    // Finished without STOP.
    let mut c = CandidateBatch::empty(&[7], 1, 4, 2, 0);
    c.length[0] = 1;
    c.actions[0..4].copy_from_slice(&[1, 0, 0, 0]);
    c.status[0] = candidate_status::FINISHED;
    assert!(c.validate().is_err());
    // Finished and truncated together.
    let mut c = finished_record();
    c.status[0] = candidate_status::FINISHED | candidate_status::TRUNCATED;
    assert!(c.validate().is_err());
    // Truncated with length short of max_steps.
    let mut c = CandidateBatch::empty(&[7], 1, 4, 2, 0);
    c.length[0] = 3;
    c.status[0] = candidate_status::TRUNCATED;
    assert!(c.validate().is_err());
    // NaN log-probability.
    let mut c = finished_record();
    c.trace_log_prob[0] = f32::NAN;
    assert!(c.validate().is_err());
    let mut c = finished_record();
    c.formula_log_prob[0] = f32::NAN;
    assert!(c.validate().is_err());
    // Lengths inconsistent with the dims.
    let mut c = finished_record();
    c.length.pop();
    assert!(c.validate().is_err());
}

#[test]
fn candidate_batch_rule_violations() {
    // Rule 1: record order. Trajectory must be r % K.
    let mut c = finished_record();
    c.trajectory[0] = 1;
    let err = c.validate().unwrap_err();
    assert!(err.to_string().contains("order"), "{err}");
    // Rule 1: the K records of a spectrum share one spectrum_id.
    let c = CandidateBatch::empty(&[7, 7], 1, 4, 2, 0);
    let err = c.validate().unwrap_err();
    assert!(err.to_string().contains("order"), "{err}");
    // Rule 2: the emitted prefix replays legally (STOP with fields set).
    let mut c = finished_record();
    c.actions[15] = 1;
    let err = c.validate().unwrap_err();
    assert!(err.to_string().contains("replay"), "{err}");
    // Rule 2: every emitted token field fits u8.
    let mut c = finished_record();
    c.actions[5] = 256;
    let err = c.validate().unwrap_err();
    assert!(err.to_string().contains("replay"), "{err}");
    // Rule 3: a finished record's open valence equals the replayed residuals.
    let mut c = finished_record();
    c.open_valence[0] = 1;
    let err = c.validate().unwrap_err();
    assert!(err.to_string().contains("open valence"), "{err}");
    // Rule 4: a request_failed record belongs to a fatal spectrum.
    let c = CandidateBatch::empty(&[7], 1, 4, 2, 0);
    let err = c.validate().unwrap_err();
    assert!(err.to_string().contains("request_failed"), "{err}");
    // Rule 4: a fatal spectrum fails on all K records.
    let mut c = finished_record();
    c.request_status[0] = request_status::EMPTY_SPECTRUM;
    let err = c.validate().unwrap_err();
    assert!(err.to_string().contains("request_failed"), "{err}");
    // Rule 5: the V0 constants are 0.
    let mut c = finished_record();
    c.attachment_partition[0] = 1;
    let err = c.validate().unwrap_err();
    assert!(err.to_string().contains("constant"), "{err}");
    // Rule 6: the oracle formula carries log-probability 0.
    let mut c = finished_record();
    c.status[0] |= candidate_status::FORMULA_SOURCE_ORACLE;
    c.formula_log_prob[0] = -1.0;
    let err = c.validate().unwrap_err();
    assert!(err.to_string().contains("oracle"), "{err}");
    // Rule 6: log-probabilities are finite and <= 1e-4.
    let mut c = finished_record();
    c.trace_log_prob[0] = f32::INFINITY;
    let err = c.validate().unwrap_err();
    assert!(err.to_string().contains("range"), "{err}");
    // Rule 7: retained fractions are in [0, 1 + 1e-4].
    let mut c = finished_record();
    c.intensity_retained[0] = 2.0;
    let err = c.validate().unwrap_err();
    assert!(err.to_string().contains("range"), "{err}");
    let mut c = finished_record();
    c.formula_mass_retained[0] = -0.5;
    let err = c.validate().unwrap_err();
    assert!(err.to_string().contains("range"), "{err}");
    // Rule 7: support completeness is 0 or 1, and 1 needs scored == joined.
    let mut c = finished_record();
    c.formula_support_complete[0] = 2;
    let err = c.validate().unwrap_err();
    assert!(err.to_string().contains("range"), "{err}");
    let mut c = finished_record();
    c.formula_support_complete[0] = 1;
    c.rows_joined[0] = 3;
    c.rows_scored[0] = 2;
    c.rows_visited[0] = 3;
    let err = c.validate().unwrap_err();
    assert!(err.to_string().contains("range"), "{err}");
    // Rule 7: rows_scored <= rows_joined <= rows_visited.
    let mut c = finished_record();
    c.rows_visited[0] = 1;
    c.rows_joined[0] = 2;
    let err = c.validate().unwrap_err();
    assert!(err.to_string().contains("range"), "{err}");
}

#[test]
fn model_config_v0_and_validation() {
    let m = ModelConfig::v0();
    assert_eq!(m.version, "ms2-model-v0");
    assert_eq!(m.chemistry, mamba3::models::ms2::CHEMISTRY_VERSION);
    assert_eq!((m.n_peaks, m.d_model), (128, 128));
    assert_eq!(m.encoder.d_inner(), 256);
    assert_eq!(m.decoder.d_inner(), 256);
    assert!(m.encoder.conv_kernel.is_none());
    assert_eq!((m.encoder_blocks, m.decoder_blocks), (2, 2));
    assert_eq!(m.attention_heads, 4);
    assert_eq!(m.fourier_features, 16);
    assert_eq!((m.max_atoms, m.max_ring_closures), (16, 4));
    assert_eq!((m.energy_scale_ev, m.energy_clip_ev), (100.0, 400.0));
    assert!(m.validate().is_ok());
    // Wrong chemistry version.
    let mut bad = m.clone();
    bad.chemistry = "ms2-chem-v9".to_string();
    assert!(bad.validate().is_err());
    // Split widths.
    let mut bad = m.clone();
    bad.d_model = 64;
    assert!(bad.validate().is_err());
    // Atom limit outside 1..=32.
    let mut bad = m.clone();
    bad.max_atoms = 0;
    assert!(bad.validate().is_err());
    bad.max_atoms = 33;
    assert!(bad.validate().is_err());
}

#[test]
fn chemistry_domain_compatibility() {
    let d = ChemistryDomain::v0();
    assert_eq!(d.version, "ms2-chem-v0.1");
    assert_eq!(d.mass_scale, 1_000_000);
    assert_eq!(d.elements.len(), 10);
    assert_eq!(d.atom_types.len(), 17);
    assert_eq!(d.bond_orders, vec![1, 2, 3]);
    assert_eq!(d.adducts.len(), 2);
    assert_eq!(d.max_hydrogen_shift, 2);
    let mut other = d.clone();
    other.recipe = "other".to_string();
    let err = d.check_compatible(&other).unwrap_err();
    assert!(err.to_string().contains("'recipe'"), "{err}");
    // The first differing field wins.
    let mut other = d.clone();
    other.version = "x".to_string();
    other.recipe = "other".to_string();
    let err = d.check_compatible(&other).unwrap_err();
    assert!(err.to_string().contains("'version'"), "{err}");
}

#[test]
fn status_names_and_masks() {
    assert_eq!(
        request_status::names(request_status::EMPTY_SPECTRUM | request_status::OVER_CAPACITY),
        vec!["empty_spectrum", "over_capacity"]
    );
    assert_eq!(
        request_status::names(
            request_status::RAW_TRUNCATED | request_status::EXACT_MASS_UNAVAILABLE
        ),
        vec!["raw_truncated", "exact_mass_unavailable"]
    );
    assert_eq!(request_status::names(0), Vec::<&str>::new());
    assert_eq!(request_status::FATAL_MASK, 0x0000_FFFF);
    assert_ne!(
        request_status::EMPTY_SPECTRUM & request_status::FATAL_MASK,
        0
    );
    assert_eq!(
        request_status::RAW_TRUNCATED & request_status::FATAL_MASK,
        0
    );
    assert_eq!(
        request_status::PEAKS_TRUNCATED & request_status::FATAL_MASK,
        0
    );
    assert_eq!(
        candidate_status::names(candidate_status::FINISHED | candidate_status::REQUEST_FAILED),
        vec!["finished", "request_failed"]
    );
    assert_eq!(
        candidate_status::names(
            candidate_status::TRUNCATED
                | candidate_status::NO_VALID_ACTION
                | candidate_status::INVALID_FINAL
                | candidate_status::DUPLICATE_TRACE
                | candidate_status::FORMULA_SOURCE_ORACLE
        ),
        vec![
            "truncated",
            "no_valid_action",
            "invalid_final",
            "duplicate_trace",
            "formula_source_oracle"
        ]
    );
}
