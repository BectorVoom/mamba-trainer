//! P1-B tests: pseudo-labels, the formula table and loss edges. Every
//! fixture-derived expected value comes from
//! `tests/fixtures/ms2/chemistry_v0.json`; hand-built cases state their
//! arithmetic inline.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use serde_json::Value;

use mamba3::models::ms2::contract::{UNKNOWN_UNCERTAINTY, request_status};
use mamba3::models::ms2::formula::{
    FORMULA_ROW_BYTES, FormulaTable, LOSSES, WindowQuery, loss_edges,
};
use mamba3::models::ms2::targets::{
    Peak, RecipeLimits, enumerate_embeddings, filter_peaks, intensity_units, peak_share,
    pseudo_labels,
};
use mamba3::models::ms2::{
    Composition, MolGraph, RawAtom, RawMolecule, composition_mass, element_index, tolerance,
};

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/ms2/chemistry_v0.json")
}

fn fixture() -> Value {
    let text = std::fs::read_to_string(fixture_path()).expect("fixture readable");
    serde_json::from_str(&text).expect("fixture parses")
}

fn as_u32(v: &Value) -> u32 {
    v.as_u64().expect("u32 in fixture") as u32
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

fn members_of(g: &Value) -> Vec<usize> {
    g["atoms"]
        .as_array()
        .expect("atoms")
        .iter()
        .map(|a| a.as_u64().expect("atom") as usize)
        .collect()
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

fn full_limits() -> RecipeLimits {
    RecipeLimits {
        max_targets: usize::MAX,
        ..RecipeLimits::V0
    }
}

#[test]
fn embeddings_match_fixture_subgraphs() {
    let f = fixture();
    for m in f["molecules"].as_array().expect("molecules") {
        let name = m["name"].as_str().unwrap();
        let graph = graph_of(m);
        let found = enumerate_embeddings(&graph, &RecipeLimits::V0);
        let expected = m["subgraphs"].as_array().expect("subgraphs");
        assert_eq!(found.len(), expected.len(), "{name} embedding count");
        for (emb, g) in found.iter().zip(expected.iter()) {
            assert_eq!(emb.atoms, members_of(g), "{name} atom set");
            assert_eq!(
                emb.boundary,
                g["boundary"].as_u64().unwrap() as usize,
                "{name} boundary"
            );
            assert_eq!(
                emb.closures,
                g["closures"].as_u64().unwrap() as usize,
                "{name} closures"
            );
            let sub = graph.induced(&emb.atoms).expect("induced subgraph");
            assert_eq!(
                composition_mass(&sub.composition()).unwrap(),
                as_u32(&g["mass_udalton"]),
                "{name} {:?} mass",
                emb.atoms
            );
        }
    }
}

#[test]
fn pseudo_labels_match_fixture() {
    let f = fixture();
    for m in f["molecules"].as_array().expect("molecules") {
        let name = m["name"].as_str().unwrap();
        let spectra = m["spectra"].as_array().expect("spectra");
        if spectra.is_empty() {
            continue;
        }
        let graph = graph_of(m);
        let class_of: BTreeMap<Vec<usize>, u64> = m["subgraphs"]
            .as_array()
            .expect("subgraphs")
            .iter()
            .map(|g| (members_of(g), g["class"].as_u64().unwrap()))
            .collect();
        for s in spectra {
            let adduct = s["adduct"].as_u64().unwrap() as u16;
            let ppm = as_u32(&s["ppm_tenths"]);
            let uncertainty = as_u32(&s["mz_uncertainty"]);
            let peaks = peaks_of(s);
            let labels =
                pseudo_labels(&graph, &peaks, adduct, ppm, uncertainty, &full_limits()).unwrap();
            assert_eq!(
                labels.canonicalization_failures, 0,
                "{name} adduct {adduct}"
            );
            assert_eq!(
                labels.graphs,
                m["identity_classes"].as_u64().unwrap() as usize,
                "{name} graph count"
            );
            let explained: Vec<u32> = s["explained_peaks"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u32)
                .collect();
            assert_eq!(labels.explained_peaks, explained, "{name} adduct {adduct}");
            assert_eq!(
                labels.ambiguous_hypotheses,
                s["ambiguous_hypotheses"].as_u64().unwrap() as usize,
                "{name} adduct {adduct} ambiguous"
            );
            let expected_targets = s["targets"].as_array().expect("targets");
            assert_eq!(
                labels.targets.len(),
                expected_targets.len(),
                "{name} adduct {adduct} target count"
            );
            for t in &labels.targets {
                assert!(!t.embeddings.is_empty(), "{name} target has embeddings");
                assert!(!t.anchors.is_empty(), "{name} target has anchors");
                // Sorted, deduplicated anchors.
                let mut sorted = t.anchors.clone();
                sorted.sort();
                sorted.dedup();
                assert_eq!(t.anchors, sorted, "{name} anchors sorted");
                let cls = class_of[&labels.embeddings[t.embeddings[0]].atoms];
                let expected = expected_targets
                    .iter()
                    .find(|e| e["class"].as_u64().unwrap() == cls)
                    .unwrap_or_else(|| panic!("{name} class {cls} in fixture targets"));
                // Integer weights compare exactly; q within 1e-12; anchors exactly.
                assert_eq!(
                    t.weight,
                    expected["weight"].as_u64().unwrap(),
                    "{name} class {cls} weight"
                );
                assert!(
                    (t.q - expected["q"].as_f64().unwrap()).abs() < 1e-12,
                    "{name} class {cls} q: {} vs {}",
                    t.q,
                    expected["q"]
                );
                let fixture_anchors: Vec<(u32, i32)> = expected["anchors"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|a| {
                        let a = a.as_array().expect("anchor pair");
                        (a[0].as_u64().unwrap() as u32, a[1].as_i64().unwrap() as i32)
                    })
                    .collect();
                assert_eq!(t.anchors, fixture_anchors, "{name} class {cls} anchors");
            }
        }
    }
}

/// First spectrum (molecule name, spectrum index) with at least four targets
/// and distinct 3rd/4th weights, so the top-3 cut is unambiguous.
fn retention_case(f: &Value) -> (String, usize) {
    for m in f["molecules"].as_array().expect("molecules") {
        let name = m["name"].as_str().unwrap().to_string();
        for (i, s) in m["spectra"].as_array().expect("spectra").iter().enumerate() {
            let targets = s["targets"].as_array().expect("targets");
            if targets.len() >= 4 {
                let (q2, q3) = (
                    targets[2]["q"].as_f64().unwrap(),
                    targets[3]["q"].as_f64().unwrap(),
                );
                if (q2 - q3).abs() > 1e-12 {
                    return (name, i);
                }
            }
        }
    }
    panic!("no fixture spectrum with a clean top-3 cut");
}

#[test]
fn retention_keeps_heaviest_and_renormalises() {
    let f = fixture();
    let molecules = f["molecules"].as_array().expect("molecules");
    let (name, index) = retention_case(&f);
    let m = molecules
        .iter()
        .find(|m| m["name"].as_str().unwrap() == name)
        .unwrap();
    let s = &m["spectra"].as_array().expect("spectra")[index];
    let graph = graph_of(m);
    let peaks = peaks_of(s);
    let adduct = s["adduct"].as_u64().unwrap() as u16;
    let ppm = as_u32(&s["ppm_tenths"]);
    let uncertainty = as_u32(&s["mz_uncertainty"]);
    let expected = s["targets"].as_array().expect("targets");

    let cut = RecipeLimits {
        max_targets: 3,
        ..RecipeLimits::V0
    };
    let labels = pseudo_labels(&graph, &peaks, adduct, ppm, uncertainty, &cut).unwrap();
    assert_eq!(labels.targets.len(), 3, "{name} keeps 3");
    let class_of: BTreeMap<Vec<usize>, u64> = m["subgraphs"]
        .as_array()
        .expect("subgraphs")
        .iter()
        .map(|g| (members_of(g), g["class"].as_u64().unwrap()))
        .collect();
    let kept_classes: Vec<u64> = labels
        .targets
        .iter()
        .map(|t| class_of[&labels.embeddings[t.embeddings[0]].atoms])
        .collect();
    let expected_classes: Vec<u64> = expected[..3]
        .iter()
        .map(|e| e["class"].as_u64().unwrap())
        .collect();
    assert_eq!(
        kept_classes, expected_classes,
        "{name} keeps the 3 heaviest"
    );
    let q_sum: f64 = labels.targets.iter().map(|t| t.q).sum();
    assert!((q_sum - 1.0).abs() < 1e-12, "{name} q renormalised");
    // Fixture q values sum to 1, so they are the weight fractions: the
    // dropped fraction is 1 minus the kept fixture q mass.
    let kept_q: f64 = expected[..3].iter().map(|e| e["q"].as_f64().unwrap()).sum();
    assert!(
        (labels.dropped_weight - (1.0 - kept_q)).abs() < 1e-12,
        "{name} dropped weight: {} vs {}",
        labels.dropped_weight,
        1.0 - kept_q
    );

    // The default cap drops nothing when there are at most 16 targets.
    let full = pseudo_labels(&graph, &peaks, adduct, ppm, uncertainty, &RecipeLimits::V0).unwrap();
    assert!(expected.len() <= 16);
    assert_eq!(
        full.targets.len(),
        expected.len(),
        "{name} default keeps all"
    );
    assert_eq!(full.dropped_weight, 0.0, "{name} nothing dropped");
}

/// Fixture molecule by name, with its subgraph members-to-class map.
fn special_molecule<'a>(
    f: &'a Value,
    name: &str,
) -> (MolGraph, BTreeMap<Vec<usize>, u64>, &'a Value) {
    let m = f["molecules"]
        .as_array()
        .expect("molecules")
        .iter()
        .find(|m| m["name"].as_str().unwrap() == name)
        .unwrap_or_else(|| panic!("molecule {name}"));
    let class_of: BTreeMap<Vec<usize>, u64> = m["subgraphs"]
        .as_array()
        .expect("subgraphs")
        .iter()
        .map(|g| (members_of(g), g["class"].as_u64().unwrap()))
        .collect();
    (graph_of(m), class_of, m)
}

/// Class of a kept target, through its first embedding's atom set.
fn target_class(
    labels: &mamba3::models::ms2::targets::Labels,
    class_of: &BTreeMap<Vec<usize>, u64>,
    t: &mamba3::models::ms2::targets::Target,
) -> u64 {
    class_of[&labels.embeddings[t.embeddings[0]].atoms]
}

#[test]
fn special_retention_tie_breaks_by_trace() {
    let f = fixture();
    let spec = &f["special_spectra"]["retention_tie"];
    let mol_name = spec["molecule"].as_str().unwrap();
    let (graph, class_of, _) = special_molecule(&f, mol_name);
    let peaks = peaks_of(spec);
    let adduct = spec["adduct"].as_u64().unwrap() as u16;
    let ppm = as_u32(&spec["ppm_tenths"]);
    let uncertainty = as_u32(&spec["mz_uncertainty"]);
    // With max_targets = 1 the kept target is the fixture's kept class.
    let cut = RecipeLimits {
        max_targets: 1,
        ..RecipeLimits::V0
    };
    let labels = pseudo_labels(&graph, &peaks, adduct, ppm, uncertainty, &cut).unwrap();
    assert_eq!(labels.targets.len(), 1, "one target kept");
    assert_eq!(
        target_class(&labels, &class_of, &labels.targets[0]),
        spec["kept_classes"][0].as_u64().unwrap(),
        "tie winner"
    );
    assert!(
        (labels.dropped_weight - spec["dropped_weight"].as_f64().unwrap()).abs() < 1e-12,
        "dropped weight: {} vs {}",
        labels.dropped_weight,
        spec["dropped_weight"]
    );
    // Without the cut the two tied graphs carry equal integer weights.
    let full = pseudo_labels(&graph, &peaks, adduct, ppm, uncertainty, &full_limits()).unwrap();
    let tied: Vec<u64> = spec["tied_classes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c.as_u64().unwrap())
        .collect();
    let mut weights: Vec<u64> = Vec::new();
    for t in &full.targets {
        if tied.contains(&target_class(&full, &class_of, t)) {
            weights.push(t.weight);
        }
    }
    assert_eq!(weights.len(), 2, "both tied graphs weighted");
    assert_eq!(weights[0], weights[1], "tied integer weights");
    assert!(!full.cut_is_tied, "two graphs never tie the 16-cut");
}

#[test]
fn special_over_sixteen_keeps_heaviest() {
    let f = fixture();
    let spec = &f["special_spectra"]["over_sixteen"];
    let mol_name = spec["molecule"].as_str().unwrap();
    let (graph, class_of, _) = special_molecule(&f, mol_name);
    let peaks = peaks_of(spec);
    let adduct = spec["adduct"].as_u64().unwrap() as u16;
    let ppm = as_u32(&spec["ppm_tenths"]);
    let uncertainty = as_u32(&spec["mz_uncertainty"]);
    let labels =
        pseudo_labels(&graph, &peaks, adduct, ppm, uncertainty, &RecipeLimits::V0).unwrap();
    assert_eq!(labels.targets.len(), 16, "sixteen kept");
    let kept: BTreeSet<u64> = labels
        .targets
        .iter()
        .map(|t| target_class(&labels, &class_of, t))
        .collect();
    let expected: BTreeSet<u64> = spec["kept_classes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c.as_u64().unwrap())
        .collect();
    assert_eq!(kept, expected, "the 16 heaviest classes");
    assert_eq!(
        labels.targets_before_cut,
        spec["targets"].as_array().unwrap().len(),
        "targets before cut"
    );
    assert!(
        (labels.dropped_weight - spec["dropped_weight"].as_f64().unwrap()).abs() < 1e-12,
        "dropped weight: {} vs {}",
        labels.dropped_weight,
        spec["dropped_weight"]
    );
}

#[test]
fn special_differing_boundary_shift_needs_two_bonds() {
    let f = fixture();
    let spec = &f["special_spectra"]["differing_boundary"];
    let mol_name = spec["molecule"].as_str().unwrap();
    let (graph, class_of, _) = special_molecule(&f, mol_name);
    let peaks = peaks_of(spec);
    let adduct = spec["adduct"].as_u64().unwrap() as u16;
    let ppm = as_u32(&spec["ppm_tenths"]);
    let uncertainty = as_u32(&spec["mz_uncertainty"]);
    let labels = pseudo_labels(&graph, &peaks, adduct, ppm, uncertainty, &full_limits()).unwrap();
    let want = spec["class"].as_u64().unwrap();
    let target = labels
        .targets
        .iter()
        .find(|t| target_class(&labels, &class_of, t) == want)
        .unwrap_or_else(|| panic!("class {want} has a target"));
    assert!(
        target.anchors.iter().any(|(_, s)| *s == -2),
        "shift -2 anchor: {:?}",
        target.anchors
    );
    let mut boundaries: Vec<usize> = target
        .embeddings
        .iter()
        .map(|&i| labels.embeddings[i].boundary)
        .collect();
    boundaries.sort();
    let expected: Vec<usize> = spec["boundaries"]
        .as_array()
        .unwrap()
        .iter()
        .map(|b| b.as_u64().unwrap() as usize)
        .collect();
    assert_eq!(boundaries, expected, "embedding boundaries");
}

#[test]
fn special_unknown_uncertainty_labels_nothing() {
    let f = fixture();
    let spec = &f["special_spectra"]["unknown_uncertainty"];
    let mol_name = spec["molecule"].as_str().unwrap();
    let (graph, _, _) = special_molecule(&f, mol_name);
    let peaks = peaks_of(spec);
    let adduct = spec["adduct"].as_u64().unwrap() as u16;
    let ppm = as_u32(&spec["ppm_tenths"]);
    let labels = pseudo_labels(
        &graph,
        &peaks,
        adduct,
        ppm,
        UNKNOWN_UNCERTAINTY,
        &RecipeLimits::V0,
    )
    .unwrap();
    assert!(labels.targets.is_empty(), "no targets");
    assert!(labels.explained_peaks.is_empty(), "no explained peaks");
    assert_eq!(labels.ambiguous_hypotheses, 0, "nothing counted");
}

#[test]
fn unknown_uncertainty_accepts_nothing() {
    let f = fixture();
    let molecules = f["molecules"].as_array().expect("molecules");
    let aspirin = molecules
        .iter()
        .find(|m| m["name"].as_str().unwrap() == "aspirin")
        .unwrap();
    let graph = graph_of(aspirin);
    let s = &aspirin["spectra"].as_array().expect("spectra")[0];
    assert!(!s["targets"].as_array().unwrap().is_empty());
    let peaks = peaks_of(s);
    let adduct = s["adduct"].as_u64().unwrap() as u16;
    let ppm = as_u32(&s["ppm_tenths"]);
    for uncertainty in [UNKNOWN_UNCERTAINTY, 1_000_000] {
        let labels =
            pseudo_labels(&graph, &peaks, adduct, ppm, uncertainty, &RecipeLimits::V0).unwrap();
        assert!(
            labels.targets.is_empty(),
            "uncertainty {uncertainty}: no target"
        );
        assert!(labels.explained_peaks.is_empty());
    }
}

#[test]
fn filter_peaks_hand_cases() {
    // The precursor bound keeps exactly precursor + 2 Da.
    let precursor = 100_000_000u32;
    let out = filter_peaks(
        &[0, 1, 2],
        &[precursor + 2_000_000, precursor + 2_000_001, precursor],
        &[1.0, 1.0, 0.5],
        precursor,
    )
    .unwrap();
    assert_eq!(out.iter().map(|p| p.id).collect::<Vec<_>>(), vec![0, 2]);
    assert_eq!(out[0].intensity, 1.0);
    assert_eq!(out[1].intensity, 0.5);
    // m/z 0 is dropped.
    let out = filter_peaks(&[0, 1], &[0, precursor], &[1.0, 1.0], precursor).unwrap();
    assert_eq!(out.iter().map(|p| p.id).collect::<Vec<_>>(), vec![1]);
    // The 1e-3 floor is relative to the max of the kept peaks.
    let out = filter_peaks(
        &[0, 1, 2, 3],
        &[precursor - 3, precursor - 2, precursor - 1, precursor],
        &[1.0, 0.001, 0.0005, 0.25],
        precursor,
    )
    .unwrap();
    assert_eq!(out.iter().map(|p| p.id).collect::<Vec<_>>(), vec![0, 1, 3]);
    assert_eq!(out[1].intensity, 0.001);
    // All-zero intensities give nothing.
    assert!(
        filter_peaks(&[0], &[precursor], &[0.0], precursor)
            .unwrap()
            .is_empty()
    );
    assert!(filter_peaks(&[], &[], &[], precursor).unwrap().is_empty());
    // Order and ids are preserved, intensities rescaled by the kept max.
    let out = filter_peaks(
        &[7, 3, 9],
        &[precursor - 2, precursor - 1, precursor],
        &[0.5, 1.0, 0.25],
        precursor,
    )
    .unwrap();
    assert_eq!(out.iter().map(|p| p.id).collect::<Vec<_>>(), vec![7, 3, 9]);
    assert_eq!(
        out.iter().map(|p| p.intensity).collect::<Vec<_>>(),
        vec![0.5, 1.0, 0.25]
    );
}

#[test]
fn filter_peaks_rejects_invalid_intensities() {
    let precursor = 100_000_000u32;
    // NaN, infinite and negative intensities are config errors naming the index.
    for (tag, value) in [
        ("nan", f64::NAN),
        ("infinite", f64::INFINITY),
        ("negative", -1.0),
    ] {
        let err = filter_peaks(
            &[0, 1],
            &[precursor - 1, precursor],
            &[1.0, value],
            precursor,
        )
        .expect_err("invalid intensity fails");
        assert!(err.to_string().contains('1'), "{tag}: {err}");
    }
    // Slices of different lengths are config errors, never a panic.
    assert!(
        filter_peaks(&[0, 1], &[precursor], &[1.0, 1.0], precursor).is_err(),
        "ids/mz mismatch"
    );
    assert!(
        filter_peaks(&[0], &[precursor], &[1.0, 1.0], precursor).is_err(),
        "ids/intensity mismatch"
    );
}

#[test]
fn peak_share_splits_exactly() {
    assert_eq!(intensity_units(1.0), 1 << 20);
    // The split factor is divisible by every count up to 16, so each count's
    // shares add back to the uncut weight.
    for graphs in 1..=16usize {
        assert_eq!(
            graphs as u64 * peak_share(1.0, graphs).unwrap(),
            peak_share(1.0, 1).unwrap(),
            "count {graphs}"
        );
    }
    assert_eq!(6 * peak_share(1.0, 6).unwrap(), peak_share(1.0, 1).unwrap());
    assert!(peak_share(1.0, 0).is_err());
}

fn formula_of(m: &Value) -> Composition {
    let mut c: Composition = [0; 10];
    for (symbol, count) in m["formula"].as_object().expect("formula") {
        c[element_index(symbol).expect("known element")] = count.as_u64().unwrap() as u16;
    }
    c
}

fn fixture_table() -> (FormulaTable, Vec<Composition>) {
    let f = fixture();
    let compositions: Vec<Composition> = f["molecules"]
        .as_array()
        .expect("molecules")
        .iter()
        .map(formula_of)
        .collect();
    let table = FormulaTable::from_compositions(compositions.iter().copied()).unwrap();
    (table, compositions)
}

#[test]
fn formula_table_from_fixtures_sorts_and_deduplicates() {
    let (table, compositions) = fixture_table();
    // 28 molecules, one shared formula (the two butanols).
    assert_eq!(compositions.len(), 28);
    assert_eq!(table.len(), 27);
    assert_eq!(table.bytes(), table.len() * FORMULA_ROW_BYTES);
    assert!(!table.is_empty());
    let mut keys: Vec<(u32, Composition)> = Vec::new();
    for i in 0..table.len() {
        keys.push((table.mass(i), *table.composition(i)));
        assert_eq!(
            composition_mass(table.composition(i)).unwrap(),
            table.mass(i),
            "row {i} mass"
        );
    }
    let mut sorted = keys.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(keys, sorted, "rows sorted and deduplicated");
    let butanol: Composition = {
        let f = fixture();
        let molecules = f["molecules"].as_array().expect("molecules");
        formula_of(
            molecules
                .iter()
                .find(|m| m["name"].as_str().unwrap() == "2-butanol")
                .unwrap(),
        )
    };
    let shared = keys.iter().filter(|(_, c)| *c == butanol).count();
    assert_eq!(shared, 1, "the butanols share one row");
    assert!(table.max_error() > 0);
    assert_eq!(FormulaTable::default().max_error(), 0);
    assert!(FormulaTable::default().is_empty());

    let text = table.to_json();
    let back = FormulaTable::from_json(&text).unwrap();
    assert_eq!(back.len(), table.len());
    for i in 0..table.len() {
        assert_eq!(back.mass(i), table.mass(i), "row {i} round trip");
        assert_eq!(
            back.composition(i),
            table.composition(i),
            "row {i} round trip"
        );
    }
}

#[test]
fn formula_table_from_json_rejects() {
    let (table, _) = fixture_table();
    let good = table.to_json();
    // A wrong mass names its row.
    let mut bad: Value = serde_json::from_str(&good).unwrap();
    bad["rows"][0][0] = serde_json::json!(1u64);
    let err = FormulaTable::from_json(&serde_json::to_string(&bad).unwrap()).unwrap_err();
    assert!(err.to_string().contains("row 0"), "{err}");
    assert!(err.to_string().contains("mass"), "{err}");
    // Unsorted rows name the row.
    let mut bad: Value = serde_json::from_str(&good).unwrap();
    let rows = bad["rows"].as_array().unwrap().clone();
    let mut swapped = rows.clone();
    swapped.swap(0, 1);
    bad["rows"] = Value::Array(swapped);
    let err = FormulaTable::from_json(&serde_json::to_string(&bad).unwrap()).unwrap_err();
    assert!(err.to_string().contains("row 1"), "{err}");
    // A wrong element order names the problem.
    let mut bad: Value = serde_json::from_str(&good).unwrap();
    let mut elements = bad["elements"].as_array().unwrap().clone();
    elements.swap(0, 1);
    bad["elements"] = Value::Array(elements);
    let err = FormulaTable::from_json(&serde_json::to_string(&bad).unwrap()).unwrap_err();
    assert!(err.to_string().contains("element"), "{err}");
}

fn comp_of(c: u16, h: u16, n: u16, o: u16) -> Composition {
    [c, h, n, o, 0, 0, 0, 0, 0, 0]
}

/// `[M+H]+` precursor m/z for a neutral composition.
fn protonated(c: &Composition) -> u32 {
    use mamba3::models::ms2::{ELECTRON_MASS, ELEMENTS, HYDROGEN};
    composition_mass(c).unwrap() + ELEMENTS[HYDROGEN].mass - ELECTRON_MASS
}

#[test]
fn formula_window_joins_absent_and_limits() {
    // (a) A precursor computed from a table row's mass joins that row.
    let ethanol_like = comp_of(2, 6, 0, 1);
    let table = FormulaTable::from_compositions([ethanol_like]).unwrap();
    let query = WindowQuery {
        precursor_mz: protonated(&ethanol_like),
        adduct: 1,
        ppm_tenths: 200,
        precursor_uncertainty: 50,
        rows_visited_max: u32::MAX,
        rows_scored_max: u32::MAX,
    };
    let found = table.window(&query);
    assert_eq!(found.joined, vec![0]);
    assert_eq!(found.ambiguous, vec![false]);
    assert!(!found.absent);
    assert!(!found.exhausted);
    assert_eq!(found.rows_scored, found.rows_joined);
    assert_eq!(found.rows_joined, 1);
    assert_eq!(found.first, 0);

    // (b) A precursor far from every row is absent, not exhausted.
    let far = WindowQuery {
        precursor_mz: 500_000_000,
        ..query
    };
    let found = table.window(&far);
    assert!(found.absent);
    assert!(!found.exhausted);
    assert!(found.joined.is_empty());

    // (c) A visit cap of 1 exhausts instead of answering.
    let capped = WindowQuery {
        rows_visited_max: 1,
        ..query
    };
    let found = table.window(&capped);
    assert!(found.exhausted);
    assert!(!found.absent);
    assert_eq!(found.rows_visited, 1);

    // (e) Unknown precursor precision visits nothing and is absent.
    let unknown = WindowQuery {
        precursor_uncertainty: UNKNOWN_UNCERTAINTY,
        ..query
    };
    let found = table.window(&unknown);
    assert!(found.absent);
    assert!(!found.exhausted);
    assert_eq!(found.rows_visited, 0);
    assert!(found.joined.is_empty());
}

#[test]
fn formula_window_partial_visit_keeps_counters() {
    // A visit cap that stops the row scan part-way keeps the counters and
    // the rows joined so far: exhausted, not complete, not absent, with the
    // exhausted status bit.
    // Two close rows (C3 vs SH4, 3371 units apart) both join at 100 ppm
    // around their midpoint; stopping the row scan after the first row keeps
    // that row and its counters: exhausted, not complete, not absent.
    let c3: Composition = [3, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let mut sh4: Composition = [0; 10];
    sh4[element_index("H").unwrap()] = 4;
    sh4[element_index("S").unwrap()] = 1;
    let table = FormulaTable::from_compositions([c3, sh4]).unwrap();
    let parent = (table.mass(0) + table.mass(1)) / 2;
    let base = WindowQuery {
        precursor_mz: parent + 1_007_825 - 549,
        adduct: 1,
        ppm_tenths: 1000,
        precursor_uncertainty: 0,
        rows_visited_max: u32::MAX,
        rows_scored_max: u32::MAX,
    };
    let full = table.window(&base);
    assert_eq!(full.rows_joined, 2, "both rows join");
    assert!(full.complete, "full search completes");
    let mut cap = None;
    for visit in 1..full.rows_visited {
        let query = WindowQuery {
            rows_visited_max: visit,
            ..base
        };
        let found = table.window(&query);
        if found.exhausted && found.rows_joined == 1 {
            cap = Some((visit, found));
            break;
        }
    }
    let (visit, found) = cap.expect("a cap stops the scan after one row");
    assert_eq!(found.rows_visited, visit, "visited stops at the cap");
    assert!(found.rows_joined >= 1, "joined rows kept");
    assert!(found.exhausted);
    assert!(!found.complete);
    assert!(!found.absent);
    assert_eq!(found.status, request_status::FORMULA_SEARCH_EXHAUSTED);
    assert_eq!(
        found.rows_scored,
        found.rows_joined.min(base.rows_scored_max)
    );
    assert_eq!(found.joined.len() as u32, found.rows_scored);
}

#[test]
fn formula_window_status_bits() {
    let ethanol_like = comp_of(2, 6, 0, 1);
    let table = FormulaTable::from_compositions([ethanol_like]).unwrap();
    let query = WindowQuery {
        precursor_mz: protonated(&ethanol_like),
        adduct: 1,
        ppm_tenths: 200,
        precursor_uncertainty: 50,
        rows_visited_max: u32::MAX,
        rows_scored_max: u32::MAX,
    };
    // A completed search that joins is complete with no status bit.
    let found = table.window(&query);
    assert!(found.complete);
    assert!(!found.exhausted);
    assert!(!found.absent);
    assert_eq!(found.status, 0);
    // A completed search that joins nothing is absent.
    let far = WindowQuery {
        precursor_mz: 500_000_000,
        ..query
    };
    let found = table.window(&far);
    assert!(found.complete);
    assert!(found.absent);
    assert_eq!(found.status, request_status::FORMULA_ABSENT);
    // Precursor 0 under adduct 1 underflows the parent mass: mass overflow,
    // not absence.
    let overflow = WindowQuery {
        precursor_mz: 0,
        ..query
    };
    let found = table.window(&overflow);
    assert_eq!(found.status, request_status::MASS_OVERFLOW);
    assert!(!found.absent);
    assert!(!found.complete);
    assert!(found.parent_mass.is_none());
    // The unknown-precision sentinel skips the search with both bits.
    let unknown = WindowQuery {
        precursor_uncertainty: UNKNOWN_UNCERTAINTY,
        ..query
    };
    let found = table.window(&unknown);
    assert_eq!(
        found.status,
        request_status::EXACT_MASS_UNAVAILABLE | request_status::FORMULA_ABSENT
    );
    assert!(found.absent);
    assert!(!found.exhausted);
}

#[test]
fn formula_window_scored_cap_truncates() {
    // (d) Two close rows (C3 vs SH4, 3371 units apart) both join at 100 ppm
    // around their midpoint; a scored cap of 1 keeps the first row.
    let c3: Composition = [3, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let mut sh4: Composition = [0; 10];
    sh4[element_index("H").unwrap()] = 4;
    sh4[element_index("S").unwrap()] = 1;
    let table = FormulaTable::from_compositions([c3, sh4]).unwrap();
    assert_eq!(table.len(), 2);
    let m0 = table.mass(0);
    let m1 = table.mass(1);
    assert_eq!(m0, 36_000_000);
    assert_eq!(m1, 36_003_371);
    let parent = (m0 + m1) / 2;
    let precursor = parent + 1_007_825 - 549;
    let query = WindowQuery {
        precursor_mz: precursor,
        adduct: 1,
        ppm_tenths: 1000,
        precursor_uncertainty: 0,
        rows_visited_max: u32::MAX,
        rows_scored_max: 1,
    };
    let found = table.window(&query);
    assert_eq!(found.rows_joined, 2, "both rows join");
    assert_eq!(found.rows_scored, 1);
    assert!(found.exhausted);
    assert!(!found.absent);
    assert_eq!(found.joined, vec![0]);
    assert_eq!(found.ambiguous, vec![false]);
    // Without the cap both rows are kept.
    let open = WindowQuery {
        rows_scored_max: u32::MAX,
        ..query
    };
    let found = table.window(&open);
    assert_eq!(found.joined, vec![0, 1]);
    assert!(!found.exhausted);
    assert_eq!(found.rows_scored, found.rows_joined);
}

#[test]
fn formula_window_visited_counter_matches_recount() {
    // (f) The visited counter equals a brute-force recount of the same
    // halving loops plus one per row in the window.
    let (table, _) = fixture_table();
    let cases = [
        (150_000_000u32, 1u16, 200u32, 50u32),
        (300_000_000, 1, 100, 0),
        (500_000_000, 2, 1000, 500),
        (50_000_000, 1, 1, 0),
    ];
    for (precursor_mz, adduct, ppm_tenths, precursor_uncertainty) in cases {
        let query = WindowQuery {
            precursor_mz,
            adduct,
            ppm_tenths,
            precursor_uncertainty,
            rows_visited_max: u32::MAX,
            rows_scored_max: u32::MAX,
        };
        let found = table.window(&query);
        let parent = mamba3::models::ms2::parent_mass(precursor_mz, adduct).unwrap();
        let tol = tolerance(precursor_mz, ppm_tenths);
        let bound = precursor_uncertainty + 1;
        let width = tol as u64 + bound as u64 + table.max_error() as u64;
        let lo_key = (parent as u64).saturating_sub(width);
        let hi_key = (parent as u64).saturating_add(width.min(u64::from(u32::MAX)));
        let n = table.len();
        let masses: Vec<u32> = (0..n).map(|r| table.mass(r)).collect();
        let mut recount = 0u32;
        let mut lo = 0usize;
        let mut hi = n;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            recount += 1;
            if (masses[mid] as u64) < lo_key {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        let first = lo;
        let mut blo = 0usize;
        let mut bhi = n;
        while blo < bhi {
            let mid = blo + (bhi - blo) / 2;
            recount += 1;
            if (masses[mid] as u64) <= hi_key {
                blo = mid + 1;
            } else {
                bhi = mid;
            }
        }
        recount += (blo - first) as u32;
        assert_eq!(found.first, first, "precursor {precursor_mz}");
        assert_eq!(found.rows_visited, recount, "precursor {precursor_mz}");
    }
}

#[test]
fn formula_window_flags_ambiguous_rows() {
    // (g) A parent one unit past the tolerance edge joins as Ambiguous:
    // r + E > tol while r <= tol + E (E >= 1 always, via the adduct bound).
    let c = comp_of(2, 4, 0, 1);
    let table = FormulaTable::from_compositions([c]).unwrap();
    let mass = table.mass(0);
    let mut parent = mass + 5;
    for _ in 0..3 {
        let precursor = parent + 1_007_825 - 549;
        let tol = tolerance(precursor, 1);
        parent = mass + tol + 1;
    }
    let precursor = parent + 1_007_825 - 549;
    let tol = tolerance(precursor, 1);
    assert_eq!(parent - mass, tol + 1);
    let query = WindowQuery {
        precursor_mz: precursor,
        adduct: 1,
        ppm_tenths: 1,
        precursor_uncertainty: 0,
        rows_visited_max: u32::MAX,
        rows_scored_max: u32::MAX,
    };
    let found = table.window(&query);
    assert_eq!(found.joined, vec![0]);
    assert_eq!(found.ambiguous, vec![true]);
    assert!(!found.absent);
}

#[test]
fn loss_edges_finds_named_losses() {
    // Peaks: a base, a water loss above it, a CO loss above it, and a far peak.
    let water = composition_mass(&LOSSES[1].composition).unwrap();
    let co = composition_mass(&LOSSES[3].composition).unwrap();
    assert!(water < co);
    let p0 = 100_000_000u32;
    let mz = vec![p0, p0 + water, p0 + co, 200_000_000];
    let edges = loss_edges(&mz, 100, 50, 8).unwrap();
    assert_eq!(edges.degree, 8);
    assert_eq!(edges.overflow, vec![false; 4]);
    // Destination 0 sees the water source (loss 1) then the CO source (loss 3).
    assert_eq!(edges.source[0 * 8], 1);
    assert_eq!(edges.loss[0 * 8], 1);
    assert_eq!(edges.source[0 * 8 + 1], 2);
    assert_eq!(edges.loss[0 * 8 + 1], 3);
    assert_eq!(edges.source[0 * 8 + 2], u32::MAX);
    // No other destination has an edge.
    for i in 1..4 {
        for slot in 0..8 {
            assert_eq!(edges.source[i * 8 + slot], u32::MAX, "peak {i} slot {slot}");
        }
        assert!(!edges.overflow[i]);
    }
    // degree 1 keeps the first candidate and flags the overflow.
    let edges = loss_edges(&mz, 100, 50, 1).unwrap();
    assert_eq!(edges.source[0], 1);
    assert_eq!(edges.loss[0], 1);
    assert!(edges.overflow[0]);
    assert!(!edges.overflow[1]);
}

#[test]
fn loss_edges_ties_and_errors() {
    // Symmetric sources around a water loss: equal residuals order by source.
    let water = composition_mass(&LOSSES[1].composition).unwrap();
    let p0 = 100_000_000u32;
    let mz = vec![p0, p0 + water - 100, p0 + water + 100];
    let edges = loss_edges(&mz, 100, 50, 2).unwrap();
    assert_eq!(edges.source[0 * 2], 1);
    assert_eq!(edges.source[0 * 2 + 1], 2);
    assert_eq!(edges.loss[0 * 2], 1);
    assert_eq!(edges.loss[0 * 2 + 1], 1);
    assert!(!edges.overflow[0]);
    // Equal residuals across losses order by loss index: exact water and CO
    // gaps from one destination both have residual 0.
    let co = composition_mass(&LOSSES[3].composition).unwrap();
    let mz = vec![p0, p0 + water, p0 + co];
    let edges = loss_edges(&mz, 100, 50, 2).unwrap();
    assert_eq!(
        (edges.source[0], edges.loss[0]),
        (1, 1),
        "water (loss 1) before CO (loss 3)"
    );
    assert_eq!((edges.source[1], edges.loss[1]), (2, 3));
    // A descending list is an error.
    assert!(loss_edges(&[200_000_000, 100_000_000], 100, 50, 2).is_err());
    // An ascending list with a duplicate is accepted (gaps of 0 match nothing).
    let edges = loss_edges(&[p0, p0, p0 + water], 100, 50, 2).unwrap();
    assert_eq!(edges.source[0 * 2], 2);
}
