//! Functional-group report: typed graphs on stdin, aromatic flags and groups out.
//!
//! Reads `{"atoms": [...], "bonds": [...]}` JSON lines on stdin and prints
//! one JSON line per molecule: `{"aromatic_atoms": [...], "groups": [[...]]}`.

use mamba3::models::ms2::functional_groups::{aromatic_atoms, functional_groups};
use mamba3::models::ms2::graph::MolGraph;

fn main() {
    let stdin = std::io::read_to_string(std::io::stdin()).unwrap_or_default();
    let mut out = String::new();
    for line in stdin.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let value: serde_json::Value =
            serde_json::from_str(line).expect("functional_groups_report: invalid JSON line");
        let atoms: Vec<u8> = value
            .get("atoms")
            .and_then(|v| v.as_array())
            .expect("functional_groups_report: missing atoms")
            .iter()
            .map(|a| a.as_u64().expect("atom") as u8)
            .collect();
        let bonds: Vec<(usize, usize, u8)> = value
            .get("bonds")
            .and_then(|v| v.as_array())
            .expect("functional_groups_report: missing bonds")
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
        let graph = MolGraph::new(atoms, bonds).expect("functional_groups_report: invalid graph");
        let aromatic = aromatic_atoms(&graph);
        let groups =
            functional_groups(&graph).expect("functional_groups_report: functional groups failed");
        let line = serde_json::json!({
            "aromatic_atoms": aromatic,
            "groups": groups.iter().map(|g| &g.atoms).collect::<Vec<_>>(),
        });
        out.push_str(&serde_json::to_string(&line).expect("serialize"));
        out.push('\n');
    }
    print!("{out}");
}
