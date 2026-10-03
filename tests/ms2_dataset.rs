//! P1-C tests: the `export_casmi.py` dataset loader, the split preparation of
//! [`Candidates`], the [`spectrum_batch`] adapter and [`percentile`]. The
//! export file under test is built from `tests/fixtures/ms2/chemistry_v0.json`
//! so every molecule and peak list is fixture-derived.

use std::path::PathBuf;

use serde_json::Value;

use mamba3::models::ms2::dataset::{
    ExportFile, ExportMolecule, ExportSpectrum, percentile, spectrum_batch,
};
use mamba3::models::ms2::targets::{Candidates, Peak, RecipeLimits, pseudo_labels};
use mamba3::models::ms2::{CHEMISTRY_VERSION, MolGraph};

/// Tolerance of the frozen recipe, in tenths of a ppm.
const PPM_TENTHS: u32 = 100;

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/ms2/chemistry_v0.json")
}

fn fixture() -> Value {
    let text = std::fs::read_to_string(fixture_path()).expect("fixture readable");
    serde_json::from_str(&text).expect("fixture parses")
}

fn atoms_of(m: &Value) -> Vec<u8> {
    m["atoms"]
        .as_array()
        .expect("atoms")
        .iter()
        .map(|a| a.as_u64().expect("atom type") as u8)
        .collect()
}

fn bonds_of(m: &Value) -> Vec<(usize, usize, u8)> {
    m["bonds"]
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
        .collect()
}

/// One export spectrum per fixture molecule, from its first fixture spectrum:
///
/// the fixture peaks become `peak_id`/`mz_udalton`/`intensity`, the precursor
/// sits 1 Da above every peak (so the §2 bound keeps all peaks) and the
/// uncertainty is the fixture's own.
fn export_spectrum(m: &Value, row: u64) -> ExportSpectrum {
    let s = &m["spectra"].as_array().expect("spectra")[0];
    let peaks = s["peaks"].as_array().expect("peaks");
    let ids: Vec<u32> = peaks
        .iter()
        .map(|p| p.as_array().expect("peak")[0].as_u64().expect("id") as u32)
        .collect();
    let mz: Vec<u32> = peaks
        .iter()
        .map(|p| p.as_array().expect("peak")[1].as_u64().expect("mz") as u32)
        .collect();
    let intensity: Vec<f64> = peaks
        .iter()
        .map(|p| p.as_array().expect("peak")[2].as_f64().expect("intensity"))
        .collect();
    // 1 Da above every peak (so the §2 bound keeps all peaks), clamped into
    // the 50–2000 Da request range so the batch validates cleanly.
    let precursor = (mz.iter().max().copied().unwrap_or(0) + 1_000_000).max(50_000_000);
    let adduct = s["adduct"].as_u64().expect("adduct") as u16;
    // The fixture ids are synthetic but satisfy the contract invariant
    // `peak_id < raw_peak_count`, so the raw count covers the largest id.
    let raw_peak_count = ids.iter().max().copied().unwrap_or(0) + 1;
    ExportSpectrum {
        row,
        spectrum_id: row,
        adduct,
        // The fixture only holds the two V0 adducts, whose charges are ±1.
        polarity: if adduct == 1 { 1 } else { -1 },
        precursor_mz_udalton: precursor,
        precursor_uncertainty_udalton: 50,
        raw_peak_count,
        peak_id: ids,
        mz_udalton: mz,
        intensity,
        mz_uncertainty_udalton: s["mz_uncertainty"].as_u64().expect("uncertainty") as u32,
        collision_energy_ev: 0.0,
        collision_energy_known: 0,
        energy_count: 0,
        instrument_class: 0,
    }
}

/// The export file under test: every fixture molecule with one spectrum.
fn export_file() -> ExportFile {
    let f = fixture();
    let molecules = f["molecules"].as_array().expect("molecules");
    let mut out = ExportFile {
        schema_version: 1,
        chemistry: CHEMISTRY_VERSION.to_string(),
        rdkit: "test".to_string(),
        source: "test".to_string(),
        seed: 0,
        n_raw: 64,
        spectra_per_molecule: 1,
        skipped_spectra: Default::default(),
        subset: "test".to_string(),
        molecules: Vec::new(),
    };
    for (i, m) in molecules.iter().enumerate() {
        // Cubane holds no fixture spectrum, so it contributes no export row.
        if m["spectra"].as_array().expect("spectra").is_empty() {
            continue;
        }
        out.molecules.push(ExportMolecule {
            key: m["name"].as_str().expect("name").to_string(),
            identity_group: 0,
            fold_identity: 2,
            atoms: atoms_of(m),
            bonds: bonds_of(m),
            spectra: vec![export_spectrum(m, i as u64)],
        });
    }
    out
}

fn graph_of(m: &Value) -> MolGraph {
    MolGraph::new(atoms_of(m), bonds_of(m)).expect("in-domain molecule builds")
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

#[test]
fn export_round_trip_and_rejections() {
    let file = export_file();
    let text = serde_json::to_string(&file).expect("export serializes");
    let back = ExportFile::from_json(&text).expect("export parses");
    assert_eq!(back, file, "round trip");

    // A wrong chemistry names the problem and both versions.
    let mut bad = serde_json::to_value(&file).expect("export value");
    bad["chemistry"] = Value::String("ms2-chem-vX".to_string());
    let err = ExportFile::from_json(&serde_json::to_string(&bad).unwrap()).unwrap_err();
    assert!(err.to_string().contains("chemistry"), "{err}");
    assert!(err.to_string().contains(CHEMISTRY_VERSION), "{err}");

    // A wrong schema version names the problem and both versions.
    let mut bad = serde_json::to_value(&file).expect("export value");
    bad["schema_version"] = Value::Number(2.into());
    let err = ExportFile::from_json(&serde_json::to_string(&bad).unwrap()).unwrap_err();
    assert!(err.to_string().contains("schema_version"), "{err}");

    // Arrays of different lengths name the problem.
    let mut bad = serde_json::to_value(&file).expect("export value");
    bad["molecules"][0]["spectra"][0]["intensity"]
        .as_array_mut()
        .expect("intensity")
        .pop();
    let err = ExportFile::from_json(&serde_json::to_string(&bad).unwrap()).unwrap_err();
    assert!(err.to_string().contains("length"), "{err}");

    // More peaks than `n_raw` names the capacity.
    let mut bad = serde_json::to_value(&file).expect("export value");
    bad["n_raw"] = Value::Number(1.into());
    let err = ExportFile::from_json(&serde_json::to_string(&bad).unwrap()).unwrap_err();
    assert!(err.to_string().contains("n_raw"), "{err}");
}

#[test]
fn candidates_label_matches_pseudo_labels() {
    let f = fixture();
    for m in f["molecules"].as_array().expect("molecules") {
        let name = m["name"].as_str().unwrap();
        let graph = graph_of(m);
        // One preparation reused for both spectra of the molecule.
        let shared = Candidates::new(&graph, &RecipeLimits::V0).unwrap();
        assert_eq!(
            shared.embeddings().len(),
            pseudo_labels(&graph, &[], 1, PPM_TENTHS, 50, &RecipeLimits::V0)
                .unwrap()
                .embeddings
                .len(),
            "{name} embedding count"
        );
        for s in m["spectra"].as_array().expect("spectra") {
            let adduct = s["adduct"].as_u64().unwrap() as u16;
            let uncertainty = s["mz_uncertainty"].as_u64().unwrap() as u32;
            let peaks = peaks_of(s);
            let fresh = Candidates::new(&graph, &RecipeLimits::V0).unwrap();
            let via_shared = shared
                .label(&peaks, adduct, PPM_TENTHS, uncertainty)
                .unwrap();
            let via_fresh = fresh
                .label(&peaks, adduct, PPM_TENTHS, uncertainty)
                .unwrap();
            let via_direct = pseudo_labels(
                &graph,
                &peaks,
                adduct,
                PPM_TENTHS,
                uncertainty,
                &RecipeLimits::V0,
            )
            .unwrap();
            for (tag, labels) in [
                ("shared", &via_shared),
                ("fresh", &via_fresh),
                ("direct", &via_direct),
            ] {
                assert_eq!(
                    labels.embeddings, via_direct.embeddings,
                    "{name} {tag} embeddings"
                );
                assert_eq!(labels.graphs, via_direct.graphs, "{name} {tag} graphs");
                assert_eq!(
                    labels.targets_before_cut, via_direct.targets_before_cut,
                    "{name} {tag} targets_before_cut"
                );
                assert_eq!(
                    labels.targets.len(),
                    via_direct.targets.len(),
                    "{name} {tag} target count"
                );
                for (a, b) in labels.targets.iter().zip(via_direct.targets.iter()) {
                    assert_eq!(a.trace, b.trace, "{name} {tag} trace");
                    assert_eq!(a.weight, b.weight, "{name} {tag} weight");
                    assert_eq!(a.q, b.q, "{name} {tag} q");
                    assert_eq!(a.embeddings, b.embeddings, "{name} {tag} members");
                    assert_eq!(a.anchors, b.anchors, "{name} {tag} anchors");
                }
                assert_eq!(
                    labels.dropped_weight, via_direct.dropped_weight,
                    "{name} {tag} dropped_weight"
                );
                assert_eq!(
                    labels.explained_peaks, via_direct.explained_peaks,
                    "{name} {tag} explained_peaks"
                );
                assert_eq!(
                    labels.ambiguous_hypotheses, via_direct.ambiguous_hypotheses,
                    "{name} {tag} ambiguous"
                );
                assert_eq!(
                    labels.canonicalization_failures, via_direct.canonicalization_failures,
                    "{name} {tag} failures"
                );
            }
        }
    }
}

#[test]
fn spectrum_batch_pads_and_validates() {
    let file = export_file();
    let items: Vec<&ExportSpectrum> = file.molecules[..2].iter().map(|m| &m.spectra[0]).collect();
    let batch = spectrum_batch(&items, 64).expect("batch builds");
    assert_eq!(batch.n_raw, 64);
    assert_eq!(batch.peak_id.len(), 2 * 64);
    assert_eq!(batch.mz_udalton.len(), 2 * 64);
    assert_eq!(batch.intensity.len(), 2 * 64);
    for (b, s) in items.iter().enumerate() {
        let base = b * 64;
        let n = s.peak_id.len();
        assert_eq!(&batch.peak_id[base..base + n], &s.peak_id[..]);
        assert_eq!(&batch.mz_udalton[base..base + n], &s.mz_udalton[..]);
        assert_eq!(batch.raw_peak_count[b], s.raw_peak_count);
        assert_eq!(batch.peak_count[b], n as u32);
        // Padding slots carry the sentinel id and zeros.
        assert!(
            batch.peak_id[base + n..base + 64]
                .iter()
                .all(|&id| id == u32::MAX),
            "padding ids"
        );
        assert!(
            batch.mz_udalton[base + n..base + 64]
                .iter()
                .all(|&mz| mz == 0),
            "padding m/z"
        );
        assert!(
            batch.intensity[base + n..base + 64]
                .iter()
                .all(|&it| it == 0.0),
            "padding intensity"
        );
    }
    // The fixture ids are sparse, so the raw count covering the largest id
    // exceeds the supplied count: the truncation warning is set, nothing else.
    assert_eq!(
        batch.validate().expect("batch validates"),
        vec![
            mamba3::models::ms2::contract::request_status::RAW_TRUNCATED,
            mamba3::models::ms2::contract::request_status::RAW_TRUNCATED
        ]
    );

    // A spectrum with more peaks than `n_raw` is an error.
    let mut too_many = items[0].clone();
    too_many.peak_id = (0..65).collect();
    too_many.mz_udalton = vec![100_000_000; 65];
    too_many.intensity = vec![1.0; 65];
    let err = spectrum_batch(&[&too_many], 64).unwrap_err();
    assert!(err.to_string().contains("n_raw"), "{err}");
}

#[test]
fn targets_before_cut_bounds_targets() {
    let f = fixture();
    for m in f["molecules"].as_array().expect("molecules") {
        let name = m["name"].as_str().unwrap();
        let graph = graph_of(m);
        let candidates = Candidates::new(&graph, &RecipeLimits::V0).unwrap();
        for s in m["spectra"].as_array().expect("spectra") {
            let adduct = s["adduct"].as_u64().unwrap() as u16;
            let uncertainty = s["mz_uncertainty"].as_u64().unwrap() as u32;
            let labels = candidates
                .label(&peaks_of(s), adduct, PPM_TENTHS, uncertainty)
                .unwrap();
            assert!(
                labels.targets_before_cut >= labels.targets.len(),
                "{name} cut keeps at most the weighted graphs"
            );
            if labels.targets_before_cut <= RecipeLimits::V0.max_targets {
                assert_eq!(
                    labels.targets_before_cut,
                    labels.targets.len(),
                    "{name} no cut without overflow"
                );
                // The kept sum re-adds the same weights, so the dropped
                // fraction is zero up to summation order.
                assert!(
                    labels.dropped_weight.abs() < 1e-12,
                    "{name} nothing dropped: {}",
                    labels.dropped_weight
                );
            }
        }
    }
}

#[test]
fn percentile_hand_values() {
    let sorted = [1.0, 2.0, 3.0, 4.0];
    assert_eq!(percentile(&sorted, 50.0), 2.5);
    assert!((percentile(&sorted, 95.0) - 3.85).abs() < 1e-12);
    assert_eq!(percentile(&sorted, 100.0), 4.0);
    assert_eq!(percentile(&sorted, 0.0), 1.0);
    assert_eq!(percentile(&[7.0], 50.0), 7.0);
    assert_eq!(percentile(&[], 50.0), 0.0);
}

#[test]
fn kept_targets_expose_report_detail() {
    // The `targets_detail` report rows of `examples/ms2_label_report.rs` are
    // built from `Labels`: each kept target's embeddings map to sorted atom
    // lists, the list sorted, with an integer weight and sorted anchors.
    let f = fixture();
    for m in f["molecules"].as_array().expect("molecules") {
        let name = m["name"].as_str().unwrap();
        let graph = graph_of(m);
        let candidates = Candidates::new(&graph, &RecipeLimits::V0).unwrap();
        for s in m["spectra"].as_array().expect("spectra") {
            let adduct = s["adduct"].as_u64().unwrap() as u16;
            let uncertainty = s["mz_uncertainty"].as_u64().unwrap() as u32;
            let labels = candidates
                .label(&peaks_of(s), adduct, PPM_TENTHS, uncertainty)
                .unwrap();
            for t in &labels.targets {
                let mut embeddings: Vec<Vec<usize>> = t
                    .embeddings
                    .iter()
                    .map(|&i| {
                        let mut atoms = labels.embeddings[i].atoms.clone();
                        atoms.sort();
                        atoms
                    })
                    .collect();
                embeddings.sort();
                assert!(!embeddings.is_empty(), "{name} detail embeddings");
                assert!(
                    embeddings.iter().all(|e| e.len() >= 3),
                    "{name} detail sizes"
                );
                let mut anchors = t.anchors.clone();
                anchors.sort();
                assert_eq!(t.anchors, anchors, "{name} detail anchors sorted");
                let _ = (t.weight, t.q);
            }
        }
    }
}
