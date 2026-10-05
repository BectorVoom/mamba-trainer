//! Host-only tests for V0 metrics (contracts §10).

use std::path::PathBuf;

use serde_json::Value;

use mamba3::models::ms2::contain::Containment;
use mamba3::models::ms2::contract::{CandidateBatch, candidate_status};
use mamba3::models::ms2::dataset::ExportSpectrum;
use mamba3::models::ms2::experiment::{ExperimentSet, SpectrumDomain};
use mamba3::models::ms2::grammar::{Limits, Token, replay};
use mamba3::models::ms2::metrics::{
    CandidateEval, SpectrumEval, evaluate_candidates, summarize, teacher_nll_per_token,
};
use mamba3::models::ms2::targets::{Candidates, Peak, RecipeLimits};
use mamba3::models::ms2::{MolGraph, RawAtom, RawMolecule};

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/ms2/chemistry_v0.json")
}

fn fixture() -> Value {
    let text = std::fs::read_to_string(fixture_path()).expect("fixture readable");
    serde_json::from_str(&text).expect("fixture parses")
}

fn raw_molecule(m: &Value) -> RawMolecule {
    let atoms = m["raw_atoms"]
        .as_array()
        .expect("raw_atoms")
        .iter()
        .map(|a| RawAtom {
            element: a["element"].as_str().expect("element").to_string(),
            charge: a["charge"].as_i64().expect("charge") as i32,
            hydrogens: a["hydrogens"].as_u64().expect("hydrogens") as u8,
            isotope: a["isotope"].as_u64().expect("isotope") as u32,
            radical_electrons: a["radical_electrons"].as_u64().expect("radical") as u8,
            valence: a["valence"].as_u64().expect("valence") as u8,
        })
        .collect();
    let bonds = m["bonds"]
        .as_array()
        .expect("bonds")
        .iter()
        .map(|b| {
            let b = b.as_array().expect("bond triple");
            (
                b[0].as_u64().expect("a") as usize,
                b[1].as_u64().expect("b") as usize,
                b[2].as_u64().expect("order") as u8,
            )
        })
        .collect();
    RawMolecule { atoms, bonds }
}

fn graph_of(m: &Value) -> MolGraph {
    raw_molecule(m)
        .to_graph()
        .expect("in-domain molecule builds")
}

fn peaks_of(s: &Value) -> Vec<Peak> {
    s["peaks"]
        .as_array()
        .expect("peaks")
        .iter()
        .map(|p| {
            let p = p.as_array().expect("peak triple");
            Peak {
                id: p[0].as_u64().expect("peak id") as u32,
                mz: p[1].as_u64().expect("mz") as u32,
                intensity: p[2].as_f64().expect("intensity"),
            }
        })
        .collect()
}

fn molecule_by_name<'a>(f: &'a Value, name: &str) -> &'a Value {
    f["molecules"]
        .as_array()
        .expect("molecules")
        .iter()
        .find(|m| m["name"].as_str().unwrap() == name)
        .unwrap_or_else(|| panic!("molecule {name}"))
}

fn cand(
    finished: bool,
    valid: bool,
    duplicate: bool,
    atoms: usize,
    contained: Containment,
) -> CandidateEval {
    CandidateEval {
        finished,
        valid,
        duplicate,
        atoms,
        contained,
        canonical: None,
    }
}

fn spec(
    molecule: usize,
    domain: SpectrumDomain,
    candidates: Vec<CandidateEval>,
    abstained: bool,
    recall: Option<bool>,
    q_found: f64,
) -> SpectrumEval {
    SpectrumEval {
        molecule,
        domain,
        candidates,
        abstained,
        formula_recall: recall,
        q_found,
        q_found_by_stratum: [0.0; 3],
        q_total_by_stratum: [0.0; 3],
    }
}

/// A [`SpectrumEval`] with per-stratum `q` masses.
fn spec_strata(
    molecule: usize,
    domain: SpectrumDomain,
    candidates: Vec<CandidateEval>,
    abstained: bool,
    recall: Option<bool>,
    q_found: f64,
    found: [f64; 3],
    total: [f64; 3],
) -> SpectrumEval {
    SpectrumEval {
        molecule,
        domain,
        candidates,
        abstained,
        formula_recall: recall,
        q_found,
        q_found_by_stratum: found,
        q_total_by_stratum: total,
    }
}

fn labeled_domain() -> SpectrumDomain {
    SpectrumDomain::InDomainLabeled
}

#[test]
fn per_molecule_aggregation() {
    // Molecule 0 has two spectra (precision 1 and 0 => 0.5); molecule 1 has
    // one spectrum (precision 1 => 1.0); overall (0.5 + 1.0) / 2 = 0.75.
    let evals = vec![
        spec(
            0,
            labeled_domain(),
            vec![cand(true, true, false, 8, Containment::Contained)],
            false,
            None,
            1.0,
        ),
        spec(
            0,
            labeled_domain(),
            vec![cand(true, true, false, 8, Containment::NotContained)],
            false,
            None,
            0.0,
        ),
        spec(
            1,
            labeled_domain(),
            vec![cand(true, true, false, 8, Containment::Contained)],
            false,
            None,
            1.0,
        ),
    ];
    let s = summarize(&evals, 1, 200, 7);
    assert!(
        (s.precision.overall.point - 0.75).abs() < 1e-12,
        "precision {}",
        s.precision.overall.point
    );
    assert!(
        (s.coverage.overall.point - 0.75).abs() < 1e-12,
        "coverage {}",
        s.coverage.overall.point
    );
    assert_eq!(s.n_spectra, 3);
    assert_eq!(s.n_molecules, 2);
}

#[test]
fn full_and_conditional_denominators() {
    // Full: molecules means [1, 0, 0] => 1/3. Conditional: only the labeled
    // molecule => 1. Unlabeled and out-of-domain are misses in full, excluded
    // from conditional.
    let evals = vec![
        spec(
            0,
            labeled_domain(),
            vec![cand(true, true, false, 8, Containment::Contained)],
            false,
            None,
            1.0,
        ),
        spec(
            1,
            SpectrumDomain::InDomainUnlabeled,
            vec![],
            true,
            None,
            0.0,
        ),
        spec(
            2,
            SpectrumDomain::OutOfDomain("unsupported_adduct".to_string()),
            vec![],
            true,
            None,
            0.0,
        ),
    ];
    let s = summarize(&evals, 1, 200, 7);
    assert!(
        (s.coverage.overall.point - 1.0 / 3.0).abs() < 1e-12,
        "full {}",
        s.coverage.overall.point
    );
    assert!(
        (s.coverage_conditional.overall.point - 1.0).abs() < 1e-12,
        "cond {}",
        s.coverage_conditional.overall.point
    );
}

#[test]
fn duplicates_excluded_from_precision() {
    // Two finished with the same trace: the second is a duplicate, so the
    // distinct denominator is 1 and precision is 1, while uniqueness is 1/2.
    let evals = vec![spec(
        0,
        labeled_domain(),
        vec![
            cand(true, true, false, 8, Containment::Contained),
            cand(true, true, true, 8, Containment::Contained),
        ],
        false,
        None,
        1.0,
    )];
    let s = summarize(&evals, 2, 200, 7);
    assert!((s.precision.overall.point - 1.0).abs() < 1e-12);
    assert!((s.uniqueness.point - 0.5).abs() < 1e-12);
}

#[test]
fn strata_boundaries() {
    // 5 in 3-5, 6 and 9 in 6-9, 10 in 10-16. Strata points: 1, 1/2, 0.
    let evals = vec![spec(
        0,
        labeled_domain(),
        vec![
            cand(true, true, false, 5, Containment::Contained),
            cand(true, true, false, 6, Containment::NotContained),
            cand(true, true, false, 9, Containment::Contained),
            cand(true, true, false, 10, Containment::NotContained),
        ],
        false,
        None,
        0.5,
    )];
    let s = summarize(&evals, 4, 200, 7);
    assert!(
        (s.precision.overall.point - 0.5).abs() < 1e-12,
        "overall {}",
        s.precision.overall.point
    );
    assert!(
        (s.precision.s3_5.point - 1.0).abs() < 1e-12,
        "3-5 {}",
        s.precision.s3_5.point
    );
    assert!(
        (s.precision.s6_9.point - 0.5).abs() < 1e-12,
        "6-9 {}",
        s.precision.s6_9.point
    );
    assert!(
        (s.precision.s10_16.point - 0.0).abs() < 1e-12,
        "10-16 {}",
        s.precision.s10_16.point
    );
}

#[test]
fn every_metric_known_answers() {
    // Distinct = c0, c1, c2 (c3 is a duplicate of c0).
    // precision 1/3, validity 4/4, uniqueness 3/4, size-aware (8/16)/3,
    // work-limit 1/3, recall 1, abstention 0, coverage 0.6.
    let evals = vec![spec(
        0,
        labeled_domain(),
        vec![
            cand(true, true, false, 8, Containment::Contained),
            cand(true, true, false, 4, Containment::NotContained),
            cand(true, true, false, 6, Containment::WorkLimit),
            cand(true, true, true, 8, Containment::Contained),
        ],
        false,
        Some(true),
        0.6,
    )];
    let s = summarize(&evals, 4, 200, 7);
    assert!(
        (s.precision.overall.point - 1.0 / 3.0).abs() < 1e-12,
        "prec {}",
        s.precision.overall.point
    );
    assert!(
        (s.validity.point - 1.0).abs() < 1e-12,
        "val {}",
        s.validity.point
    );
    assert!(
        (s.uniqueness.point - 0.75).abs() < 1e-12,
        "uniq {}",
        s.uniqueness.point
    );
    assert!(
        (s.size_aware_precision.overall.point - (0.5 / 3.0)).abs() < 1e-12,
        "saw {}",
        s.size_aware_precision.overall.point
    );
    assert!(
        (s.worklimit_rate.overall.point - 1.0 / 3.0).abs() < 1e-12,
        "wlr {}",
        s.worklimit_rate.overall.point
    );
    assert!((s.formula_recall.overall.point - 1.0).abs() < 1e-12);
    assert!((s.abstention.overall.point - 0.0).abs() < 1e-12);
    assert!((s.coverage.overall.point - 0.6).abs() < 1e-12);
}

#[test]
fn bootstrap_constant_collapses_and_determinism() {
    let evals = vec![
        spec(
            0,
            labeled_domain(),
            vec![cand(true, true, false, 8, Containment::Contained)],
            false,
            Some(true),
            1.0,
        ),
        spec(
            1,
            labeled_domain(),
            vec![cand(true, true, false, 8, Containment::Contained)],
            false,
            Some(true),
            1.0,
        ),
    ];
    let a = summarize(&evals, 1, 500, 42);
    assert!((a.precision.overall.point - 1.0).abs() < 1e-12);
    assert!((a.precision.overall.lo - 1.0).abs() < 1e-12);
    assert!((a.precision.overall.hi - 1.0).abs() < 1e-12);
    let b = summarize(&evals, 1, 500, 42);
    assert_eq!(a, b, "same seed determinism");
    // Non-constant intervals still bracket the point.
    let mixed = vec![
        spec(
            0,
            labeled_domain(),
            vec![cand(true, true, false, 8, Containment::Contained)],
            false,
            None,
            1.0,
        ),
        spec(
            1,
            labeled_domain(),
            vec![cand(true, true, false, 8, Containment::NotContained)],
            false,
            None,
            0.0,
        ),
    ];
    let m1 = summarize(&mixed, 1, 500, 11);
    let m2 = summarize(&mixed, 1, 500, 11);
    assert_eq!(m1, m2, "seeded determinism");
    assert!(m1.precision.overall.lo <= m1.precision.overall.point + 1e-12);
    assert!(m1.precision.overall.point <= m1.precision.overall.hi + 1e-12);
}

#[test]
fn teacher_nll_known_answers() {
    // Spectrum 0: (0.5*1 + 0.5*3) / (0.5*2 + 0.5*2) = 1. Spectrum 1:
    // (1*2) / (1*4) = 0.5. One molecule => 0.75; unlabeled excluded.
    let (point, (lo, hi)) = teacher_nll_per_token(
        &[1.0, 3.0, 2.0, 99.0],
        &[0.5, 0.5, 1.0, 0.0],
        &[2, 2, 4, 0],
        2,
        2,
        &[0, 0],
    );
    assert!((point - 0.75).abs() < 1e-6, "point {point}");
    assert!(lo <= point && point <= hi, "interval [{lo}, {hi}]");
    // All unlabeled (zero denominators) => zeros, no panic.
    let (point, _) = teacher_nll_per_token(&[1.0], &[0.0], &[0], 1, 1, &[0]);
    assert_eq!(point, 0.0);
}

/// Build a valid [`CandidateBatch`] from raw traces (one spectrum set).
fn batch_from_traces(spectrum_ids: &[u64], traces: &[Vec<Token>]) -> CandidateBatch {
    let b = spectrum_ids.len();
    let k = traces.len() / b;
    let max_steps = Limits::V0.max_steps();
    let max_atoms = Limits::V0.max_atoms();
    let mut batch = CandidateBatch::empty(spectrum_ids, k, max_steps, max_atoms, 4);
    for (r, trace) in traces.iter().enumerate() {
        let base = r * max_steps * 4;
        for (step, tok) in trace.iter().enumerate() {
            batch.actions[base + step * 4] = u32::from(tok.kind);
            batch.actions[base + step * 4 + 1] = u32::from(tok.atom_type);
            batch.actions[base + step * 4 + 2] = u32::from(tok.bond);
            batch.actions[base + step * 4 + 3] = u32::from(tok.pointer);
        }
        batch.length[r] = trace.len() as u32;
        batch.formula_row[r] = 0;
        batch.formula_rank[r] = 0;
        // Every finished record needs real formula provenance: a rank below
        // `rows_scored` with non-zero counts (a finished graph with zero
        // counts and MAX row/rank is corrupt, not formula-less).
        batch.formula_counts[r * 10] = 1;
        batch.status[r] = candidate_status::FINISHED;
        let state = replay(trace, Limits::V0, None).expect("trace replays");
        let residual = state.residual_valence();
        for (i, v) in residual.iter().enumerate() {
            batch.open_valence[r * max_atoms + i] = *v;
        }
    }
    for bb in 0..b {
        batch.rows_visited[bb] = 1;
        batch.rows_joined[bb] = 1;
        batch.rows_scored[bb] = 1;
    }
    batch.validate().expect("hand-built batch validates");
    batch
}

fn dummy_export_spectrum(spectrum_id: u64) -> ExportSpectrum {
    ExportSpectrum {
        row: spectrum_id,
        spectrum_id,
        adduct: 1,
        polarity: 1,
        precursor_mz_udalton: 100_000_000,
        precursor_uncertainty_udalton: 50,
        raw_peak_count: 1,
        peak_id: vec![0],
        mz_udalton: vec![100_000_000],
        intensity: vec![1.0],
        mz_uncertainty_udalton: 50,
        collision_energy_ev: 0.0,
        collision_energy_known: 0,
        energy_count: 0,
        instrument_class: 0,
    }
}

#[test]
fn evaluate_candidates_contained_and_target_matched() {
    let f = fixture();
    let ethanol = molecule_by_name(&f, "ethanol");
    let benzene = molecule_by_name(&f, "benzene");
    let parent_a = graph_of(ethanol);
    let parent_b = graph_of(benzene);
    // Labels from the fixture's first ethanol spectrum: at least one target.
    let s0 = &ethanol["spectra"].as_array().expect("spectra")[0];
    let peaks = peaks_of(s0);
    let adduct = s0["adduct"].as_u64().unwrap() as u16;
    let ppm = s0["ppm_tenths"].as_u64().unwrap() as u32;
    let unc = s0["mz_uncertainty"].as_u64().unwrap() as u32;
    let cands = Candidates::new(&parent_a, &RecipeLimits::V0).expect("candidates");
    let labels = cands.label(&peaks, adduct, ppm, unc).expect("labels");
    assert!(!labels.targets.is_empty(), "ethanol has targets");
    let hit_trace = labels.targets[0].trace.clone();
    let hit_q = labels.targets[0].q;
    // A benzene whole-graph trace cannot fit in ethanol (6 vs 3 atoms).
    let miss_graph =
        MolGraph::new(parent_b.atoms().to_vec(), parent_b.bonds().to_vec()).expect("rebuild");
    let miss_trace = mamba3::models::ms2::canonical_trace(
        &miss_graph,
        Limits::V0,
        mamba3::models::ms2::grammar::CANONICAL_WORK_LIMIT,
    )
    .expect("benzene canonicalizes")
    .trace;
    let parent_a_comp = parent_a.composition();
    let set = ExperimentSet {
        name: "test".to_string(),
        source_sha256: "test".to_string(),
        molecules: vec!["ethanol".to_string()],
        spectra: vec![mamba3::models::ms2::experiment::ExperimentSpectrum {
            molecule: 0,
            spectrum: dummy_export_spectrum(1001),
            parent: parent_a,
            parent_composition: parent_a_comp,
            labels: Some(labels),
            domain: SpectrumDomain::InDomainLabeled,
        }],
    };
    let batch = batch_from_traces(&[1001], &[hit_trace.clone(), miss_trace]);
    let evals = evaluate_candidates(&set, &[0], &batch, 1_000_000).expect("evaluates");
    assert_eq!(evals.len(), 1);
    let e = &evals[0];
    assert_eq!(e.candidates.len(), 2);
    assert!(e.candidates[0].finished && e.candidates[0].valid);
    assert!(!e.candidates[0].duplicate);
    assert_eq!(e.candidates[0].contained, Containment::Contained);
    assert_eq!(e.candidates[0].canonical, Some(hit_trace));
    // The benzene graph exceeds the ethanol formula budget, so it is invalid
    // with no graph: still finished, still not contained, still unmatched.
    assert!(e.candidates[1].finished);
    assert_eq!(e.candidates[1].contained, Containment::NotContained);
    assert!(!e.abstained);
    assert!(
        (e.q_found - hit_q).abs() < 1e-12,
        "q_found {} vs {hit_q}",
        e.q_found
    );
    // Strata: the found masses sum to `q_found` (every recipe target has 3–16
    // atoms), the totals sum to the kept `q` mass of 1, and no found mass
    // exceeds its stratum total.
    let found_sum: f64 = e.q_found_by_stratum.iter().sum();
    let total_sum: f64 = e.q_total_by_stratum.iter().sum();
    assert!(
        (found_sum - e.q_found).abs() < 1e-12,
        "strata found {found_sum} vs {}",
        e.q_found
    );
    assert!(
        (total_sum - 1.0).abs() < 1e-12,
        "strata totals {total_sum} vs 1"
    );
    for s in 0..3 {
        assert!(
            e.q_found_by_stratum[s] <= e.q_total_by_stratum[s] + 1e-12,
            "stratum {s}: found more than total"
        );
    }
}

#[test]
fn coverage_strata_known_answers() {
    // One spectrum: strata found/total = 0.5/0.5, 0/0.5, 0.25/0.25, so the
    // stratum points are 1, 0, 1 and the overall is 0.75. The conditional
    // copies the labeled spectrum's ratios.
    let evals = vec![spec_strata(
        0,
        labeled_domain(),
        vec![cand(true, true, false, 8, Containment::Contained)],
        false,
        None,
        0.75,
        [0.5, 0.0, 0.25],
        [0.5, 0.5, 0.25],
    )];
    let s = summarize(&evals, 1, 200, 7);
    assert!((s.coverage.overall.point - 0.75).abs() < 1e-12);
    assert!((s.coverage.s3_5.point - 1.0).abs() < 1e-12);
    assert!((s.coverage.s6_9.point - 0.0).abs() < 1e-12);
    assert!((s.coverage.s10_16.point - 1.0).abs() < 1e-12);
    assert!((s.coverage_conditional.s6_9.point - 0.0).abs() < 1e-12);
}

#[test]
fn coverage_strata_skip_empty_strata() {
    // Molecule 0's spectrum holds no 10–16-atom target (total 0): the stratum
    // is undefined there, so the 10–16 point comes from molecule 1 alone.
    // Molecule means: overall (0.5 + 1.0) / 2 = 0.75; 3–5 (1 + 0) / 2 = 0.5.
    let evals = vec![
        spec_strata(
            0,
            labeled_domain(),
            vec![cand(true, true, false, 4, Containment::Contained)],
            false,
            None,
            0.5,
            [0.5, 0.0, 0.0],
            [0.5, 0.5, 0.0],
        ),
        spec_strata(
            1,
            labeled_domain(),
            vec![cand(true, true, false, 12, Containment::Contained)],
            false,
            None,
            1.0,
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 1.0],
        ),
    ];
    let s = summarize(&evals, 1, 200, 7);
    assert!(
        (s.coverage.overall.point - 0.75).abs() < 1e-12,
        "overall {}",
        s.coverage.overall.point
    );
    assert!(
        (s.coverage.s3_5.point - 0.5).abs() < 1e-12,
        "3-5 {}",
        s.coverage.s3_5.point
    );
    assert!(
        (s.coverage.s10_16.point - 1.0).abs() < 1e-12,
        "10-16 {}",
        s.coverage.s10_16.point
    );
}

#[test]
fn formula_recall_excludes_none() {
    // Two spectra of one molecule (Some(true), None) plus one of another
    // (Some(false)): molecule means 1 and 0, overall 0.5.
    let evals = vec![
        spec(
            0,
            labeled_domain(),
            vec![cand(true, true, false, 8, Containment::Contained)],
            false,
            Some(true),
            1.0,
        ),
        spec(
            0,
            labeled_domain(),
            vec![cand(true, true, false, 8, Containment::Contained)],
            false,
            None,
            1.0,
        ),
        spec(
            1,
            labeled_domain(),
            vec![cand(true, true, false, 8, Containment::NotContained)],
            false,
            Some(false),
            0.0,
        ),
    ];
    let s = summarize(&evals, 1, 200, 7);
    assert!(
        (s.formula_recall.overall.point - 0.5).abs() < 1e-12,
        "recall {}",
        s.formula_recall.overall.point
    );
}

#[test]
fn flag_metrics_have_null_strata() {
    // Formula recall and abstention are per-spectrum flags with no size:
    // the overall value is reported and every size stratum is null, never
    // a copy of the overall.
    let evals = vec![
        spec(
            0,
            labeled_domain(),
            vec![cand(true, true, false, 8, Containment::Contained)],
            false,
            Some(true),
            1.0,
        ),
        spec(1, labeled_domain(), vec![], true, Some(false), 0.0),
    ];
    let s = summarize(&evals, 1, 200, 7);
    assert!((s.formula_recall.overall.point - 0.5).abs() < 1e-12);
    assert!((s.abstention.overall.point - 0.5).abs() < 1e-12);
    assert!(s.formula_recall.s3_5.is_none());
    assert!(s.formula_recall.s6_9.is_none());
    assert!(s.formula_recall.s10_16.is_none());
    assert!(s.abstention.s3_5.is_none());
    assert!(s.abstention.s6_9.is_none());
    assert!(s.abstention.s10_16.is_none());
    let json = serde_json::to_value(&s).expect("summary serializes");
    for metric in ["formula_recall", "abstention"] {
        for stratum in ["s3_5", "s6_9", "s10_16"] {
            assert!(
                json[metric][stratum].is_null(),
                "{metric}.{stratum} is not null: {}",
                json[metric][stratum]
            );
        }
    }
}

#[test]
fn validity_and_uniqueness_are_overall_only() {
    // One finished-and-valid plus one unfinished candidate: the overall
    // validity is 0.5. The old size strata divided by finished-in-range and
    // read 1.0 here, answering a different question, so the strata are
    // dropped (no stratum keys in the JSON) rather than reported.
    let evals = vec![spec(
        0,
        labeled_domain(),
        vec![
            cand(true, true, false, 8, Containment::Contained),
            cand(false, false, false, 0, Containment::NotContained),
        ],
        false,
        None,
        0.5,
    )];
    let s = summarize(&evals, 2, 200, 7);
    assert!(
        (s.validity.point - 0.5).abs() < 1e-12,
        "validity {}",
        s.validity.point
    );
    assert!(
        (s.uniqueness.point - 1.0).abs() < 1e-12,
        "uniqueness {}",
        s.uniqueness.point
    );
    let json = serde_json::to_value(&s).expect("summary serializes");
    for metric in ["validity", "uniqueness"] {
        let obj = json[metric].as_object().expect("metric is an object");
        assert!(
            !obj.contains_key("s3_5") && !obj.contains_key("s6_9") && !obj.contains_key("s10_16"),
            "{metric} must not carry size strata: {obj:?}"
        );
    }
}

#[test]
fn export_provenance_is_allow_list_only() {
    use mamba3::models::ms2::experiment::export_provenance;

    // A raw export header in the shape of `tools/ms2/export_casmi.py`
    // output: allow-listed scalar provenance plus a `molecules` array
    // carrying per-molecule and per-spectrum rows (SMILES, spectrum ids,
    // peaks) that must never reach the report. The padding makes the raw
    // header well over 2 kB, so a whole-header copy would fail the size
    // check too.
    let raw = serde_json::json!({
        "schema_version": 1,
        "chemistry": "ms2-chem-v0",
        "rdkit": "2024.03.1",
        "source": "casmi",
        "seed": 7,
        "n_raw": 512,
        "spectra_per_molecule": 4,
        "spectrum_sampling": {"method": "top_n", "n": 4},
        "skipped_spectra": {"precursor_out_of_range": 3},
        "subset": "train",
        "molecules": [
            {
                "key": "mol0",
                "smiles": "CCO",
                "spectra": [
                    {
                        "spectrum_id": 1001,
                        "mz_udalton": [100_000_000, 200_000_000],
                        "intensity": [1.0, 2.0],
                        "padding": "x".repeat(4096),
                    }
                ],
            }
        ],
    });
    let prov = export_provenance(&raw, "overfit_train.json", "abc123");
    // No banned key anywhere in the output, at any depth.
    fn keys(value: &serde_json::Value, out: &mut Vec<String>) {
        match value {
            serde_json::Value::Object(map) => {
                for (k, v) in map {
                    out.push(k.clone());
                    keys(v, out);
                }
            }
            serde_json::Value::Array(items) => {
                for v in items {
                    keys(v, out);
                }
            }
            _ => {}
        }
    }
    let mut found = Vec::new();
    keys(&prov, &mut found);
    for banned in [
        "molecules",
        "smiles",
        "spectra",
        "spectrum_id",
        "mz_udalton",
    ] {
        assert!(
            !found.iter().any(|k| k == banned),
            "banned key {banned} in provenance: {found:?}"
        );
    }
    // The allow-list survives, with the file name and hash.
    assert_eq!(prov["file"], serde_json::json!("overfit_train.json"));
    assert_eq!(prov["source_sha256"], serde_json::json!("abc123"));
    assert_eq!(prov["n_raw"], serde_json::json!(512));
    assert_eq!(prov["subset"], serde_json::json!("train"));
    let text = serde_json::to_string(&prov).expect("provenance serializes");
    assert!(
        text.len() < 2048,
        "provenance serializes to {} bytes, over 2 kB",
        text.len()
    );
}
