//! Batch stereo reports for the RDKit cross-check tool.
//!
//! Reads one JSON document on stdin:
//!
//! ```json
//! {"graphs": [{"atoms": [4, 2, 3], "bonds": [[0, 1, 1]]}],
//!  "max_elements": 12, "max_automorphisms": 20000,
//!  "work_limit": 100000, "max_expanded": 64}
//! ```
//!
//! (`graphs` is required; every limit is optional and defaults to
//! [`StereoLimits::default`](mamba3::models::ms2::stereo::StereoLimits)
//! except `max_expanded`, which defaults to 64 here so the cross-check sees
//! every isomer.) Prints one JSON document on stdout with `version` and one
//! `reports` entry per graph: the potential elements (tetrahedral centres
//! with ligand lists, double bonds with reference ligands; `"H"` for
//! hydrogen, `"lone_pair"` for a nitrogen lone pair), the stereogenic
//! indices among them, counts, resolution, the automorphism count and the
//! canonical isomers as 0/1 vectors over the stereogenic elements
//! (`raw_assignments` / `distinct_stereoisomers` are `null` when
//! unresolved). Atom indices are the input's; graphs that fail
//! `MolGraph::new` give an `unresolved: graph_rebuild` entry, never a
//! crash.
//!
//! Usage:
//! ```text
//! cargo run --release --no-default-features --features cpu \
//!   --example stereo_report < graphs.json > reports.json
//! ```

use mamba3::models::ms2::graph::MolGraph;
use mamba3::models::ms2::stereo::{Ligand, STEREO_VERSION, StereoElement, StereoLimits, perceive};
use serde_json::{Value, json};

/// One ligand as JSON: heavy atoms by index, `"H"`, `"lone_pair"`.
fn ligand_json(ligand: &Ligand) -> Value {
    match ligand {
        Ligand::Atom(i) => json!(*i),
        Ligand::Hydrogen => json!("H"),
        Ligand::LonePair => json!("lone_pair"),
    }
}

/// One potential element as JSON.
fn element_json(element: &StereoElement) -> Value {
    match element {
        StereoElement::Tetrahedral { atom, ligands } => json!({
            "kind": "tetrahedral",
            "atom": atom,
            "ligands": ligands.iter().map(ligand_json).collect::<Vec<_>>(),
        }),
        StereoElement::DoubleBond { a, b, ref_a, ref_b } => json!({
            "kind": "double_bond",
            "atoms": [a, b],
            "reference": [ligand_json(ref_a), ligand_json(ref_b)],
        }),
    }
}

fn main() -> mamba3::error::Result<()> {
    let mut input = String::new();
    std::io::Read::read_to_string(&mut std::io::stdin(), &mut input).map_err(|e| {
        mamba3::error::Error::config(format!("stereo_report: reading stdin failed: {e}"))
    })?;
    let doc: Value = serde_json::from_str(&input).map_err(|e| {
        mamba3::error::Error::config(format!("stereo_report: stdin is not valid JSON: {e}"))
    })?;
    let obj = doc.as_object().ok_or_else(|| {
        mamba3::error::Error::config("stereo_report: stdin must be a JSON object".to_string())
    })?;
    let graphs = obj.get("graphs").and_then(Value::as_array).ok_or_else(|| {
        mamba3::error::Error::config("stereo_report: missing 'graphs' list".to_string())
    })?;
    let get_limit = |key: &str, default: usize| -> mamba3::error::Result<usize> {
        match obj.get(key) {
            None => Ok(default),
            Some(v) => v.as_u64().map(|n| n as usize).ok_or_else(|| {
                mamba3::error::Error::config(format!(
                    "stereo_report: '{key}' must be a non-negative integer"
                ))
            }),
        }
    };
    let limits = StereoLimits {
        max_elements: get_limit("max_elements", 10)?,
        max_automorphisms: get_limit("max_automorphisms", 20_000)?,
        work_limit: get_limit("work_limit", 100_000)?,
        max_expanded: get_limit("max_expanded", 64)?,
    };
    let mut reports = Vec::with_capacity(graphs.len());
    for (i, g) in graphs.iter().enumerate() {
        let entry: Result<mamba3::models::ms2::stereo::StereoReport, String> =
            (|| {
                let atoms: Vec<u8> = g
                    .get("atoms")
                    .and_then(Value::as_array)
                    .ok_or_else(|| format!("graph[{i}]: missing 'atoms' list"))?
                    .iter()
                    .map(|v| {
                        v.as_u64()
                            .and_then(|n| u8::try_from(n).ok())
                            .ok_or_else(|| format!("graph[{i}]: atom type entries must fit u8"))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let bonds: Vec<(usize, usize, u8)> =
                    g.get("bonds")
                        .and_then(Value::as_array)
                        .ok_or_else(|| format!("graph[{i}]: missing 'bonds' list"))?
                        .iter()
                        .map(|b| {
                            let t = b
                                .as_array()
                                .ok_or_else(|| format!("graph[{i}]: bond entries must be lists"))?;
                            if t.len() != 3 {
                                return Err(format!("graph[{i}]: bond entries must have 3 fields"));
                            }
                            let a = t[0].as_u64().ok_or_else(|| {
                                format!("graph[{i}]: bond endpoints must be integers")
                            })? as usize;
                            let c = t[1].as_u64().ok_or_else(|| {
                                format!("graph[{i}]: bond endpoints must be integers")
                            })? as usize;
                            let o = t[2]
                                .as_u64()
                                .and_then(|n| u8::try_from(n).ok())
                                .ok_or_else(|| format!("graph[{i}]: bond orders must fit u8"))?;
                            Ok((a, c, o))
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                let graph = MolGraph::new(atoms, bonds)
                    .map_err(|e| format!("graph[{i}] does not build: {e}"))?;
                Ok(perceive(&graph, &limits))
            })();
        match entry {
            Ok(report) => reports.push(json!({
                "potential": report.potential.iter().map(element_json).collect::<Vec<_>>(),
                "stereogenic": report.elements.iter().map(|e| e.potential_index).collect::<Vec<_>>(),
                "not_stereogenic": report.not_stereogenic,
                "unsupported": report.unsupported,
                "raw_assignments": report.raw_assignments,
                "distinct_stereoisomers": report.distinct,
                "resolution": report.resolution.text(),
                "automorphisms": report.automorphisms,
                "isomers": report.isomers,
                "isomers_truncated": report.isomers_truncated,
            })),
            Err(e) => reports.push(json!({
                "potential": [],
                "stereogenic": [],
                "not_stereogenic": 0,
                "unsupported": [],
                "raw_assignments": null,
                "distinct_stereoisomers": null,
                "resolution": "unresolved: graph_rebuild",
                "automorphisms": null,
                "isomers": [],
                "isomers_truncated": false,
                "error": e.to_string(),
            })),
        }
    }
    println!(
        "{}",
        serde_json::to_string(&json!({"version": STEREO_VERSION, "reports": reports}))
            .expect("reports serialize")
    );
    Ok(())
}
