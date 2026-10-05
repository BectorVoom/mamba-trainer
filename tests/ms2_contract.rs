//! P1-B tests: `SpectrumBatch` validation, schema round-trips and the
//! generation/candidate/model/domain configs of contract §§3 and 8.

use mamba3::models::ms2::contract::{
    CandidateBatch, ChemistryDomain, Control, FormulaSource, GenerationConfig, GenerationMode,
    ModelConfig, NO_FORMULA, SCHEMA_VERSION, SCHEMA_VERSION_V1, SPECTRUM_SCHEMA_VERSION,
    SpectrumBatch, candidate_status, request_status,
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
        schema_version: SPECTRUM_SCHEMA_VERSION,
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
    assert_eq!(d.schema_version, SPECTRUM_SCHEMA_VERSION);
    let back: ChemistryDomain = serde_json::from_str(&serde_json::to_string(&d).unwrap()).unwrap();
    assert_eq!(back, d);
    assert!(d.check_compatible(&back).is_ok());
    assert!(d.validate().is_ok());
}

#[test]
fn every_schema_rejects_version_2() {
    // V1 §1.2: `SpectrumBatch` and `ChemistryDomain` stay at version 1 and
    // still reject 2; `GenerationConfig`, `CandidateBatch` and `ModelConfig`
    // move to version 2 (1 still loads, 3 is rejected).
    let mut b = valid_batch();
    b.schema_version = 2;
    let err = b.validate().unwrap_err();
    assert!(err.to_string().contains("schema_version"), "{err}");
    assert!(
        err.to_string().contains('2') && err.to_string().contains('1'),
        "{err}"
    );

    // Version-1 documents of the three V1 schemas still load and validate.
    let mut g1 = GenerationConfig::default();
    g1.schema_version = SCHEMA_VERSION_V1;
    g1.formula_source = FormulaSource::Table;
    g1.formula_window = 32;
    assert!(g1.validate(16, 4).is_ok());
    let mut c1 = CandidateBatch::empty(&[7], 1, 4, 2, 0);
    c1.schema_version = SCHEMA_VERSION_V1;
    c1.formula_counts.clear();
    c1.formula_source.clear();
    c1.formula_rank.clear();
    c1.evidence_count.clear();
    c1.evidence_peak_id.clear();
    c1.evidence_hypothesis.clear();
    c1.evidence_shift.clear();
    c1.evidence_residual.clear();
    c1.evidence_log_prob.clear();
    c1.request_status = vec![request_status::EMPTY_SPECTRUM];
    assert!(c1.validate().is_ok());
    let mut m1 = ModelConfig::v0();
    m1.schema_version = SCHEMA_VERSION_V1;
    assert!(m1.validate().is_ok());

    // Version 3 is rejected by all three.
    let mut g = GenerationConfig::default();
    g.schema_version = 3;
    let err = g.validate(16, 4).unwrap_err();
    assert!(err.to_string().contains("schema_version"), "{err}");
    assert!(
        err.to_string().contains('3') && err.to_string().contains('1'),
        "{err}"
    );

    let mut c = CandidateBatch::empty(&[7], 1, 4, 2, 0);
    c.schema_version = 3;
    let err = c.validate().unwrap_err();
    assert!(err.to_string().contains("schema_version"), "{err}");
    assert!(
        err.to_string().contains('3') && err.to_string().contains('1'),
        "{err}"
    );

    let mut m = ModelConfig::v0();
    m.schema_version = 3;
    let err = m.validate().unwrap_err();
    assert!(err.to_string().contains("schema_version"), "{err}");
    assert!(
        err.to_string().contains('3') && err.to_string().contains('1'),
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

/// `finished_record` with real formula provenance (finding R1-C6): rank 0
/// below rows_scored 1, nonzero counts, a real table row. The base for rule
/// tests that need validity up to the rule under test — a finished record
/// without a formula is corrupt, not a valid base.
fn finished_scored_record() -> CandidateBatch {
    let mut c = finished_record();
    c.formula_rank[0] = 0;
    c.formula_counts[0] = 6;
    c.formula_row[0] = 5;
    c.rows_visited[0] = 1;
    c.rows_joined[0] = 1;
    c.rows_scored[0] = 1;
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
    // A finished record needs real formula provenance (finding R1-C6): the
    // bare `finished_record` (no formula) no longer validates on its own.
    assert!(finished_record().validate().is_err());
    let mut with_formula = finished_record();
    with_formula.formula_rank[0] = 0;
    with_formula.formula_counts[0] = 6;
    with_formula.formula_row[0] = 5;
    with_formula.rows_visited[0] = 1;
    with_formula.rows_joined[0] = 1;
    with_formula.rows_scored[0] = 1;
    with_formula.validate().unwrap();
}

#[test]
fn finished_record_without_formula_is_rejected() {
    // Finding R1-C6: a finished graph with zero counts and MAX row/rank is
    // corrupt, not formula-less — no trajectory starts without a formula.
    // The reviewer's corruption (finished graph, zero counts, MAX row/rank,
    // zero counters, nonfatal status) is rejected, with or without search
    // counters.
    let bare = finished_record();
    assert!(bare.request_status[0] & request_status::FATAL_MASK == 0);
    let err = bare.validate().unwrap_err();
    assert!(
        err.to_string().contains("no formula provenance"),
        "finished without formula names provenance: {err}"
    );
    // The full reviewer corruption: zero search counters and all.
    let mut full = finished_record();
    full.rows_visited[0] = 0;
    full.rows_joined[0] = 0;
    full.rows_scored[0] = 0;
    let err = full.validate().unwrap_err();
    assert!(
        err.to_string().contains("no formula provenance"),
        "full corruption names provenance: {err}"
    );
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
    let mut c = finished_scored_record();
    c.trajectory[0] = 1;
    let err = c.validate().unwrap_err();
    assert!(err.to_string().contains("order"), "{err}");
    // Rule 1: the K records of a spectrum share one spectrum_id.
    let c = CandidateBatch::empty(&[7, 7], 1, 4, 2, 0);
    let err = c.validate().unwrap_err();
    assert!(err.to_string().contains("order"), "{err}");
    // Rule 2: the emitted prefix replays legally (STOP with fields set).
    let mut c = finished_scored_record();
    c.actions[15] = 1;
    let err = c.validate().unwrap_err();
    assert!(err.to_string().contains("replay"), "{err}");
    // Rule 2: every emitted token field fits u8.
    let mut c = finished_scored_record();
    c.actions[5] = 256;
    let err = c.validate().unwrap_err();
    assert!(err.to_string().contains("replay"), "{err}");
    // Rule 3: a finished record's open valence equals the replayed residuals.
    let mut c = finished_scored_record();
    c.open_valence[0] = 1;
    let err = c.validate().unwrap_err();
    assert!(err.to_string().contains("open valence"), "{err}");
    // Rule 4: a request_failed record belongs to a fatal spectrum or to one
    // with no scored formula (contracts §9 abstention, e.g. exhausted). An
    // empty batch (rows_scored 0, failed, not fatal) is valid for abstention.
    CandidateBatch::empty(&[7], 1, 4, 2, 0).validate().unwrap();
    // Failed with scored formulas and no fatal bit still errors.
    let mut c = finished_record();
    c.rows_visited[0] = 2;
    c.rows_joined[0] = 2;
    c.rows_scored[0] = 2;
    c.formula_rank[0] = 1;
    c.formula_counts[0] = 6;
    c.formula_row[0] = 5;
    c.status[0] = candidate_status::REQUEST_FAILED;
    c.length[0] = 0;
    for v in c.actions.iter_mut() { *v = 0; }
    for v in c.open_valence.iter_mut() { *v = 0; }
    let err = c.validate().unwrap_err();
    assert!(err.to_string().contains("request_failed"), "{err}");
    // Rule 4: a fatal spectrum fails on all K records.
    let mut c = finished_scored_record();
    c.request_status[0] = request_status::EMPTY_SPECTRUM;
    let err = c.validate().unwrap_err();
    assert!(err.to_string().contains("request_failed"), "{err}");
    // Rule 5: the V0 constants are 0.
    let mut c = finished_scored_record();
    c.attachment_partition[0] = 1;
    let err = c.validate().unwrap_err();
    assert!(err.to_string().contains("constant"), "{err}");
    // Rule 6: the oracle formula carries log-probability 0.
    let mut c = finished_scored_record();
    c.status[0] |= candidate_status::FORMULA_SOURCE_ORACLE;
    c.formula_log_prob[0] = -1.0;
    let err = c.validate().unwrap_err();
    assert!(err.to_string().contains("oracle"), "{err}");
    // Rule 6: log-probabilities are finite and <= 1e-4.
    let mut c = finished_scored_record();
    c.trace_log_prob[0] = f32::INFINITY;
    let err = c.validate().unwrap_err();
    assert!(err.to_string().contains("range"), "{err}");
    // Rule 7: retained fractions are in [0, 1 + 1e-4].
    let mut c = finished_scored_record();
    c.intensity_retained[0] = 2.0;
    let err = c.validate().unwrap_err();
    assert!(err.to_string().contains("range"), "{err}");
    let mut c = finished_scored_record();
    c.formula_mass_retained[0] = -0.5;
    let err = c.validate().unwrap_err();
    assert!(err.to_string().contains("range"), "{err}");
    // Rule 7: support completeness is 0 or 1, and 1 needs scored == joined.
    let mut c = finished_scored_record();
    c.formula_support_complete[0] = 2;
    let err = c.validate().unwrap_err();
    assert!(err.to_string().contains("range"), "{err}");
    let mut c = finished_scored_record();
    c.formula_support_complete[0] = 1;
    c.rows_joined[0] = 3;
    c.rows_scored[0] = 2;
    c.rows_visited[0] = 3;
    let err = c.validate().unwrap_err();
    assert!(err.to_string().contains("range"), "{err}");
    // Rule 7: rows_scored <= rows_joined <= rows_visited.
    // (rank 0 needs rows_scored >= 1, so the scored base keeps scored 1:
    // scored 1 <= joined 2, and the table joined 2 > visited 1 still fires.)
    let mut c = finished_scored_record();
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
    // Ring-closure limit above 8 (V1 §3.1).
    let mut bad = m.clone();
    bad.max_ring_closures = 9;
    assert!(bad.validate().is_err());
    // Decoder blocks outside 1..=4 (V1 §3.1).
    let mut bad = m.clone();
    bad.decoder_blocks = 0;
    assert!(bad.validate().is_err());
    bad.decoder_blocks = 5;
    assert!(bad.validate().is_err());
}

#[test]
fn model_config_v1_candidate_shape() {
    // V1 §3.1: the candidate shape restricted to what exists — A = 32,
    // R_max = 8, 4 decoder blocks, decoder d_model/d_inner set apart from
    // the encoder's — validates, while the encoder and peak cap stay V0. It
    // is for tests only, not a default.
    let m = ModelConfig::v1_candidate();
    assert_eq!((m.max_atoms, m.max_ring_closures), (32, 8));
    assert_eq!((m.encoder_blocks, m.decoder_blocks), (2, 4));
    assert_eq!(m.decoder.d_model, m.encoder.d_model);
    assert_eq!((m.d_model, m.n_peaks), (128, 128));
    assert_ne!(m.decoder.d_inner(), m.encoder.d_inner());
    assert!(m.validate().is_ok());
    // The differing decoder inner width is accepted, not tied to the
    // encoder's: 8 heads x 16 channels against 4 x 64.
    assert_eq!(m.encoder.d_inner(), 256);
    assert_eq!(m.decoder.d_inner(), 128);
    // The derived horizon T = 2 + A + R_max = 42 validates for generation.
    let mut g = GenerationConfig::default();
    g.max_steps = 42;
    assert!(g.validate(32, 8).is_ok());
    assert!(g.validate(16, 4).is_ok());
    g.max_steps = 41;
    assert!(g.validate(32, 8).is_err());
}

/// A batch with caller-chosen lengths and statuses for the V1 §3.3 work
/// tests: `work` reads only those two fields (plus the shape), so the trace
/// contents are dummy zeros.
fn work_batch(lengths: &[u32], statuses: &[u32], max_steps: usize) -> CandidateBatch {
    let k = lengths.len();
    assert_eq!(statuses.len(), k, "one status per record");
    CandidateBatch {
        schema_version: SCHEMA_VERSION,
        batch: 1,
        trajectories: k,
        max_steps,
        max_atoms: 4,
        max_ring_closures: 0,
        spectrum_id: vec![911; k],
        trajectory: (0..k as u32).collect(),
        actions: vec![0; k * max_steps * 4],
        length: lengths.to_vec(),
        formula_row: vec![NO_FORMULA; k],
        formula_log_prob: vec![0.0; k],
        trace_log_prob: vec![0.0; k],
        open_valence: vec![0; k * 4],
        attachment_partition: vec![0; k],
        status: statuses.to_vec(),
        evidence_status: vec![0; k],
        evidence_count: vec![0; k],
        evidence_peak_id: vec![0; (k) * 4],
        evidence_hypothesis: vec![0; (k) * 4],
        evidence_shift: vec![0; (k) * 4],
        evidence_residual: vec![0; (k) * 4],
        evidence_log_prob: vec![0.0; (k) * 4],
        identity_resolution: vec![0; k],
        request_status: vec![0],
        rows_visited: vec![0],
        rows_joined: vec![0],
        rows_scored: vec![0],
        formula_support_complete: vec![0],
        formula_mass_retained: vec![0.0],
        peaks_kept: vec![0],
        intensity_retained: vec![0.0],
        formula_counts: vec![0; k * 10],
        formula_source: vec![0],
        formula_rank: vec![NO_FORMULA; k],
    }
}

#[test]
fn generation_work_hand_built_batches() {
    // V1 §3.3 on a hand-built batch (T = 6): finished early, failed at the
    // root, failed mid-trace, truncated, never started, finished late.
    use candidate_status as cs;
    let batch = work_batch(
        &[3, 1, 4, 6, 0, 5],
        &[
            cs::FINISHED,
            cs::NO_VALID_ACTION,
            cs::NO_VALID_ACTION,
            cs::TRUNCATED,
            cs::REQUEST_FAILED,
            cs::FINISHED,
        ],
        6,
    );
    let work = batch.work();
    assert_eq!(work.submitted, 6 * 5, "submitted B * K * (T - 1) steps");
    assert_eq!(work.steps.len(), 5, "one entry per step t in 1..T");
    // Active per step: started trajectories with length > t, plus the
    // no_valid_action trajectories with length == t.
    let active: Vec<usize> = work.steps.iter().map(|s| s.active).collect();
    assert_eq!(active, vec![5, 4, 3, 3, 1], "active invocations per step");
    for (i, s) in work.steps.iter().enumerate() {
        assert_eq!(s.step, i + 1, "step t runs over 1..T");
        assert_eq!(s.active + s.inactive, 6, "active + inactive is B * K");
    }
    assert_eq!(work.active_total, 16, "sum of active over the steps");
    assert!(
        (work.active_fraction - 16.0 / 30.0).abs() < 1e-6,
        "active fraction {}, want {}",
        work.active_fraction,
        16.0 / 30.0,
    );
    // The reconciliation identity: the active sum is the emitted non-START
    // tokens (length - 1 per started record) plus the failure detections.
    let non_start: u32 = [3u32, 1, 4, 6, 0, 5]
        .iter()
        .map(|&len| len.saturating_sub(1))
        .sum();
    assert_eq!(
        work.active_total as u32,
        non_start + 2,
        "failures detected at t = 1 and t = 4"
    );
}

#[test]
fn generation_work_never_started_batch() {
    // A failed request (all length 0, `request_failed`): nothing ever ran,
    // so every step is inactive and the fraction is 0, not NaN.
    let batch = CandidateBatch::empty(&[77], 4, 6, 4, 0);
    assert!(batch.length.iter().all(|&len| len == 0));
    let work = batch.work();
    assert_eq!(work.submitted, 4 * 5);
    assert_eq!(work.active_total, 0);
    assert_eq!(work.active_fraction, 0.0);
    for s in &work.steps {
        assert_eq!((s.active, s.inactive), (0, 4));
    }
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

#[test]
fn schema_v1_json_loads_and_validates() {
    use mamba3::error::Error;
    use serde_json::Value;

    // GenerationConfig: v2 round trip, v1 JSON (missing new fields default
    // to Table/32), v3 rejected.
    let g = GenerationConfig::default();
    let back: GenerationConfig = serde_json::from_str(&serde_json::to_string(&g).unwrap()).unwrap();
    assert_eq!(back, g);
    assert!(back.validate(16, 4).is_ok());
    let mut v = serde_json::to_value(&g).unwrap();
    v["schema_version"] = Value::from(SCHEMA_VERSION_V1);
    v.as_object_mut().unwrap().remove("formula_source");
    v.as_object_mut().unwrap().remove("formula_window");
    let g1: GenerationConfig = serde_json::from_value(v).unwrap();
    assert_eq!(g1.schema_version, SCHEMA_VERSION_V1);
    assert_eq!(g1.formula_source, FormulaSource::Table);
    assert_eq!(g1.formula_window, 32);
    assert!(g1.validate(16, 4).is_ok());
    // Explicit Table/32 on a v1 document also loads and validates.
    let mut g1e = GenerationConfig::default();
    g1e.schema_version = SCHEMA_VERSION_V1;
    g1e.formula_source = FormulaSource::Table;
    g1e.formula_window = 32;
    assert!(g1e.validate(16, 4).is_ok());
    let mut g3 = GenerationConfig::default();
    g3.schema_version = 3;
    let g3v = serde_json::to_value(&g3).unwrap();
    let g3back: GenerationConfig = serde_json::from_value(g3v).unwrap();
    let err = g3back.validate(16, 4).unwrap_err();
    assert!(matches!(err, Error::Config(_)), "{err}");

    // CandidateBatch: v2 round trip, v1 JSON (three new fields absent),
    // v3 rejected.
    let mut c = CandidateBatch::empty(&[7], 1, 4, 2, 0);
    c.request_status = vec![request_status::EMPTY_SPECTRUM];
    let back: CandidateBatch = serde_json::from_str(&serde_json::to_string(&c).unwrap()).unwrap();
    assert_eq!(back, c);
    assert!(back.validate().is_ok());
    let mut v = serde_json::to_value(&c).unwrap();
    v["schema_version"] = Value::from(SCHEMA_VERSION_V1);
    for f in ["formula_counts", "formula_source", "formula_rank", "evidence_count", "evidence_peak_id", "evidence_hypothesis", "evidence_shift", "evidence_residual", "evidence_log_prob"] {
        v.as_object_mut().unwrap().remove(f);
    }
    let c1: CandidateBatch = serde_json::from_value(v).unwrap();
    assert!(c1.formula_counts.is_empty());
    assert!(c1.formula_source.is_empty());
    assert!(c1.formula_rank.is_empty());
    assert!(c1.evidence_count.is_empty());
    assert!(c1.validate().is_ok());
    let mut c3 = CandidateBatch::empty(&[7], 1, 4, 2, 0);
    c3.request_status = vec![request_status::EMPTY_SPECTRUM];
    c3.schema_version = 3;
    let c3v = serde_json::to_value(&c3).unwrap();
    let c3back: CandidateBatch = serde_json::from_value(c3v).unwrap();
    let err = c3back.validate().unwrap_err();
    assert!(matches!(err, Error::Config(_)), "{err}");

    // ModelConfig: v2 round trip, v1 JSON, v3 rejected.
    let m = ModelConfig::v0();
    let back: ModelConfig = serde_json::from_str(&serde_json::to_string(&m).unwrap()).unwrap();
    assert_eq!(back, m);
    assert!(back.validate().is_ok());
    let mut v = serde_json::to_value(&m).unwrap();
    v["schema_version"] = Value::from(SCHEMA_VERSION_V1);
    let m1: ModelConfig = serde_json::from_value(v).unwrap();
    assert_eq!(m1.schema_version, SCHEMA_VERSION_V1);
    assert!(m1.validate().is_ok());
    let mut m3 = ModelConfig::v0();
    m3.schema_version = 3;
    let m3v = serde_json::to_value(&m3).unwrap();
    let m3back: ModelConfig = serde_json::from_value(m3v).unwrap();
    let err = m3back.validate().unwrap_err();
    assert!(matches!(err, Error::Config(_)), "{err}");
}

#[test]
fn train_config_missing_field_defaults_to_row() {
    use mamba3::models::ms2::train::{GoldFormulaConditioning, TrainConfig};
    use serde_json::Value;
    assert_eq!(
        TrainConfig::default().gold_formula_conditioning,
        GoldFormulaConditioning::ScoredRowOrZero
    );
    let mut v = serde_json::to_value(&TrainConfig::default()).unwrap();
    v.as_object_mut()
        .unwrap()
        .remove("gold_formula_conditioning");
    assert!(
        !v.as_object()
            .unwrap()
            .contains_key("gold_formula_conditioning"),
        "V0 JSON has no gold_formula_conditioning field"
    );
    let back: TrainConfig = serde_json::from_value::<Value>(v).map_or_else(
        |e| panic!("V0 TrainConfig JSON loads: {e}"),
        |vv| serde_json::from_value(vv).unwrap(),
    );
    assert_eq!(
        back.gold_formula_conditioning,
        GoldFormulaConditioning::ScoredRowOrZero
    );
    assert!(back.validate().is_ok());
}

#[test]
fn candidate_new_fields_rules() {
    // Base: a finished record with real formula provenance validates
    // (a finished record without a formula is corrupt, finding R1-C6).
    let ok = finished_scored_record();
    assert!(ok.validate().is_ok());
    // Length rules (v2): each new field must match its documented length.
    let mut bad = finished_record();
    bad.formula_counts.pop();
    assert!(bad.validate().is_err());
    let mut bad = finished_record();
    bad.formula_source.clear();
    assert!(bad.validate().is_err());
    let mut bad = finished_record();
    bad.formula_rank.pop();
    assert!(bad.validate().is_err());
    // Range rule: formula_source is 0 (table) or 1 (enumeration).
    let mut bad = finished_record();
    bad.formula_source[0] = 2;
    let err = bad.validate().unwrap_err();
    assert!(err.to_string().contains("formula_source"), "{err}");
    // Enumeration rule: source == 1 implies formula_row == MAX.
    let mut en = finished_scored_record();
    en.formula_source[0] = 1;
    en.formula_row[0] = NO_FORMULA;
    assert!(en.validate().is_ok(), "source 1 with row MAX validates");
    let mut bad = finished_record();
    bad.formula_source[0] = 1;
    bad.formula_row[0] = 0;
    let err = bad.validate().unwrap_err();
    assert!(err.to_string().contains("formula_row"), "{err}");
    // Empty rule, positive: a scored formula has a rank, nonzero counts and
    // a row. The rank is a slot in the spectrum's scored support, so the
    // fixture carries rows_scored/joined/visited above it (6/6/6).
    let mut scored = finished_record();
    scored.rows_visited[0] = 6;
    scored.rows_joined[0] = 6;
    scored.rows_scored[0] = 6;
    scored.formula_rank[0] = 5;
    scored.formula_counts[0] = 6;
    scored.formula_row[0] = 5;
    assert!(scored.validate().is_ok());
    // Empty rule, negative: all-zero counts disagree with a set rank.
    let mut bad = finished_record();
    bad.formula_rank[0] = 5;
    let err = bad.validate().unwrap_err();
    assert!(err.to_string().contains("formula_rank"), "{err}");
    // Empty rule, negative: nonzero counts disagree with rank MAX.
    let mut bad = finished_record();
    bad.formula_counts[0] = 1;
    let err = bad.validate().unwrap_err();
    assert!(err.to_string().contains("formula_rank"), "{err}");
    // Empty rule, negative: rank MAX (no formula) needs row MAX too.
    let mut bad = finished_record();
    bad.formula_row[0] = 0;
    let err = bad.validate().unwrap_err();
    assert!(err.to_string().contains("formula_row"), "{err}");
    // Provenance, negative: a real rank must lie below the spectrum's
    // rows_scored. The fixture above has rows_scored 0, so any real rank
    // fails; a rank at or above rows_scored fails too.
    let mut bad = finished_record();
    bad.rows_visited[0] = 6;
    bad.rows_joined[0] = 6;
    bad.rows_scored[0] = 0;
    bad.formula_rank[0] = 5;
    bad.formula_counts[0] = 6;
    bad.formula_row[0] = 5;
    let err = bad.validate().unwrap_err();
    assert!(err.to_string().contains("rows_scored"), "{err}");
    let mut bad = finished_record();
    bad.rows_visited[0] = 6;
    bad.rows_joined[0] = 6;
    bad.rows_scored[0] = 5;
    bad.formula_rank[0] = 5;
    bad.formula_counts[0] = 6;
    bad.formula_row[0] = 5;
    let err = bad.validate().unwrap_err();
    assert!(err.to_string().contains("rows_scored"), "{err}");
    // Provenance, negative: a table-source record with a formula must name
    // a real table row (not u32::MAX).
    let mut bad = finished_record();
    bad.rows_visited[0] = 6;
    bad.rows_joined[0] = 6;
    bad.rows_scored[0] = 6;
    bad.formula_rank[0] = 2;
    bad.formula_counts[0] = 6;
    bad.formula_row[0] = NO_FORMULA;
    let err = bad.validate().unwrap_err();
    assert!(err.to_string().contains("formula_row"), "{err}");
    // Provenance, positive: an enumeration-source record with a formula has
    // rank below rows_scored, nonzero counts and row MAX.
    let mut en_scored = finished_record();
    en_scored.formula_source[0] = 1;
    en_scored.rows_visited[0] = 6;
    en_scored.rows_joined[0] = 6;
    en_scored.rows_scored[0] = 6;
    en_scored.formula_rank[0] = 2;
    en_scored.formula_counts[0] = 6;
    en_scored.formula_row[0] = NO_FORMULA;
    assert!(en_scored.validate().is_ok());
}

#[test]
fn generation_config_enumerate_supported() {
    // I1: `Enumerate` now validates (V1 §1.4); the lane limits must be
    // non-zero.
    use mamba3::error::Error;
    let mut g = GenerationConfig::default();
    g.formula_source = FormulaSource::Enumerate;
    assert!(g.validate(16, 4).is_ok());
    g.enum_lanes_max = 0;
    let err = g.validate(16, 4).unwrap_err();
    assert!(matches!(err, Error::Config(_)), "{err}");
    g.enum_lanes_max = 262_144;
    g.enum_lane_visits_max = 0;
    let err = g.validate(16, 4).unwrap_err();
    assert!(matches!(err, Error::Config(_)), "{err}");
}

#[test]
fn enumeration_counters_allow_four_h_per_visit() {
    // D1: `visited = 1, joined = 2` is legal for enumeration (one heavy
    // vector joining two hydrogen counts) but illegal for the table source
    // (`joined <= visited`). `scored <= joined` is required for both.
    let mut e = finished_scored_record();
    e.formula_source[0] = 1;
    e.formula_row[0] = NO_FORMULA;
    e.rows_visited[0] = 1;
    e.rows_joined[0] = 2;
    e.rows_scored[0] = 2;
    assert!(e.validate().is_ok(), "enumeration visited=1 joined=2 passes");
    let mut t = finished_scored_record();
    t.rows_visited[0] = 1;
    t.rows_joined[0] = 2;
    t.rows_scored[0] = 2;
    let err = t.validate().unwrap_err();
    assert!(err.to_string().contains("table source"), "{err}");
    // Enumeration bound: 5 joins from 1 visit exceeds 4 per vector.
    let mut bad = finished_scored_record();
    bad.formula_source[0] = 1;
    bad.formula_row[0] = NO_FORMULA;
    bad.rows_visited[0] = 1;
    bad.rows_joined[0] = 5;
    bad.rows_scored[0] = 5;
    bad.formula_rank[0] = 2;
    let err = bad.validate().unwrap_err();
    assert!(err.to_string().contains("4 * rows_visited"), "{err}");
    // Saturated enumeration counters are lower bounds carrying exhaustion.
    // No scored formula: use an empty (failed) batch shape for the saturated
    // case (failed records with rows_scored == 0 are valid without a fatal
    // bit, contracts §9 abstention).
    let mut sat_empty = CandidateBatch::empty(&[7], 1, 5, 2, 0);
    sat_empty.formula_source[0] = 1;
    sat_empty.rows_visited[0] = u32::MAX - 1;
    sat_empty.rows_joined[0] = u32::MAX - 1;
    sat_empty.rows_scored[0] = 0;
    sat_empty.request_status[0] = request_status::FORMULA_SEARCH_EXHAUSTED;
    assert!(sat_empty.validate().is_ok());
    // Saturated without exhaustion is rejected.
    let mut sat_bad = CandidateBatch::empty(&[7], 1, 5, 2, 0);
    sat_bad.formula_source[0] = 1;
    sat_bad.rows_visited[0] = u32::MAX - 1;
    sat_bad.rows_joined[0] = 0;
    sat_bad.rows_scored[0] = 0;
    sat_bad.request_status[0] = 0;
    assert!(sat_bad.validate().is_err());
}

#[test]
fn exhausted_enumeration_cannot_claim_completeness() {
    // Finding N3: `formula_support_complete == 1` is rejected when the
    // request carries `FORMULA_SEARCH_EXHAUSTED` or a counter is saturated
    // (contracts §9: `complete` is 1 exactly for a completed search). The
    // reviewer's input on a valid filled enumeration record.
    let mut base = finished_scored_record();
    base.formula_source[0] = 1;
    base.formula_row[0] = NO_FORMULA;
    base.formula_support_complete[0] = 1;
    assert!(base.validate().is_ok(), "valid enumeration record validates");
    // The reviewer corruption: joined == scored == 1 with provenance, then
    // saturated visited, exhaustion and a kept completeness claim.
    let mut bad = base.clone();
    bad.rows_visited[0] = u32::MAX - 1;
    bad.rows_joined[0] = 1;
    bad.rows_scored[0] = 1;
    bad.request_status[0] |= request_status::FORMULA_SEARCH_EXHAUSTED;
    bad.formula_support_complete[0] = 1;
    let err = bad.validate().unwrap_err();
    assert!(
        err.to_string().contains("formula_support_complete"),
        "exhausted completeness names completeness: {err}"
    );
    // Exhaustion alone (saturated counters, cleared completeness) stays
    // legal: the rule targets the false claim, not the search outcome.
    let mut ok = bad.clone();
    ok.formula_support_complete[0] = 0;
    assert!(ok.validate().is_ok(), "cleared completeness validates");
    // Exhaustion without saturation is also rejected with completeness set.
    let mut bad2 = base.clone();
    bad2.request_status[0] |= request_status::FORMULA_SEARCH_EXHAUSTED;
    bad2.formula_support_complete[0] = 1;
    let err = bad2.validate().unwrap_err();
    assert!(
        err.to_string().contains("formula_support_complete"),
        "exhausted completeness names completeness: {err}"
    );
}
