//! Zero reads on a warmed non-report training step with `Enumerate` (own
//! binary: reads the process-global read counter).

#![cfg(feature = "backend")]

use mamba3::backend::{Device, read_count, reset_read_count};
use mamba3::backends::Auto;
use mamba3::models::ms2::chem::Composition;
use mamba3::models::ms2::dataset::ExportSpectrum;
use mamba3::models::ms2::experiment::{ExperimentSet, ExperimentSpectrum, SpectrumDomain};
use mamba3::models::ms2::contract::{FormulaSource, SPECTRUM_SCHEMA_VERSION, SpectrumBatch};
use mamba3::models::ms2::formula::FormulaTable;
use mamba3::models::ms2::formula_enum::{EnumDomain, RatioBounds};
use mamba3::models::ms2::graph::MolGraph;
use mamba3::models::ms2::train::{GoldFormulaConditioning, Ms2Trainer, TrainConfig};
use mamba3::models::ms2::contract::ModelConfig;

type R = Auto;
type E = f32;

#[test]
fn enumerate_training_warmed_step_reads_zero() {
    let device = Device::<R>::default();
    let comps: Vec<Composition> = vec![[6, 6, 0, 0, 0, 0, 0, 0, 0, 0], [3, 7, 1, 2, 0, 0, 0, 0, 0, 0]];
    let domain = EnumDomain::from_compositions(comps.clone(), 0).unwrap();
    let bounds = RatioBounds::fit(comps.clone(), 0).unwrap();
    let table = FormulaTable::from_compositions(comps.clone().into_iter()).unwrap();
    let train_config = TrainConfig {
        batch: 2,
        slots: 2,
        gold_formula_conditioning: GoldFormulaConditioning::Composition,
        formula_source: FormulaSource::Enumerate,
        formula_window: 32,
                    lambda_assign: 0.0,
..TrainConfig::default()
    };
    let mut trainer =
        Ms2Trainer::<R, E>::new(&ModelConfig::v0(), &table, &train_config, &device).unwrap();
    trainer.upload_enum_artifacts(&domain, &bounds).unwrap();
    // Minimal experiment set.
    let n_raw = 64;
    let precursors: Vec<u32> = comps
        .iter()
        .map(|c| mamba3::models::ms2::chem::composition_mass(c).unwrap() + 1_007_825 - 549)
        .collect();
    let batch = SpectrumBatch {
        schema_version: SPECTRUM_SCHEMA_VERSION,
        n_raw: n_raw as u32,
        spectrum_id: vec![1, 2],
        raw_peak_count: vec![8, 8],
        peak_count: vec![8, 8],
        peak_id: (0..128).map(|i| (i % 64) as u32).collect(),
        mz_udalton: vec![60_000_000; 128],
        intensity: vec![1.0; 128],
        intensity_scale: 0,
        mz_uncertainty_udalton: vec![50, 50],
        precursor_mz_udalton: precursors,
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
    let spectra = (0..2)
        .map(|i| {
            let base = i * n_raw;
            let np = 8;
            ExperimentSpectrum {
                molecule: i,
                spectrum: ExportSpectrum {
                    row: i as u64,
                    spectrum_id: batch.spectrum_id[i],
                    adduct: 1,
                    polarity: 1,
                    precursor_mz_udalton: batch.precursor_mz_udalton[i],
                    precursor_uncertainty_udalton: 50,
                    raw_peak_count: batch.raw_peak_count[i],
                    peak_id: batch.peak_id[base..base + np].to_vec(),
                    mz_udalton: batch.mz_udalton[base..base + np].to_vec(),
                    intensity: batch.intensity[base..base + np].iter().map(|&v| v as f64).collect(),
                    mz_uncertainty_udalton: 50,
                    collision_energy_ev: 30.0,
                    collision_energy_known: 1,
                    energy_count: 1,
                    instrument_class: 0,
                },
                parent: MolGraph::new(Vec::new(), Vec::new()).expect("empty graph builds"),
                parent_composition: comps[i],
                labels: None,
                domain: SpectrumDomain::InDomainUnlabeled,
            }
        })
        .collect();
    let set = ExperimentSet {
        name: "enum-train-fp".to_string(),
        source_sha256: "synthetic".to_string(),
        molecules: vec!["mol0".to_string(), "mol1".to_string()],
        spectra,
    };
    let indices = vec![0usize, 1];
    for _ in 0..2 {
        let _ = trainer.step(&set, &indices).unwrap();
    }
    device.synchronize();
    reset_read_count();
    let _ = trainer.step(&set, &indices).unwrap();
    device.synchronize();
    assert_eq!(read_count(), 0, "warmed non-report enumerate step reads zero");
}
