"""Pseudo-label statistics of an exported subset, by the Python reference (P1.9 cross-check).

Reads a file written by `export_casmi.py`, applies recipe `q-cut-v1` with
`ms2_reference.py` and writes per-spectrum results and their aggregates. The Rust
reference writes the same report from the same file (`examples/ms2_label_report`);
the two are compared field by field with `--compare`.

Graph identity differs between the two on purpose: RDKit canonical fragment
SMILES with atoms tagged by type id here, canonical traces in Rust. Equal counts
on every spectrum are the evidence that the two identities agree.

    uv run --project /Users/ods/Documents/Enveda_CASMI python tools/ms2/label_report.py \
        --input <export>.json --out bench/results/ms2/labels_python.json [--compare <rust report>.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger

import ms2_reference as ref

RDLogger.DisableLog("rdApp.*")
PPM_TENTHS = 100
BOND = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}


def mol_from_graph(atoms, bonds):
    """An RDKit molecule with the exact hydrogens and bond orders of the exported graph."""
    mol = Chem.RWMol()
    for t in atoms:
        element, h, _ = ref.ATOM_TYPES[t]
        a = Chem.Atom(element)
        a.SetNumExplicitHs(h)
        a.SetNoImplicit(True)
        mol.AddAtom(a)
    for a, b, order in bonds:
        mol.AddBond(a, b, BOND[order])
    mol.UpdatePropertyCache(strict=False)
    return mol


def spectrum_report(mol, atoms, subgraphs, spectrum):
    keep, rel = ref.filter_peaks(spectrum["mz_udalton"], spectrum["intensity"], spectrum["precursor_mz_udalton"])
    peaks = [(spectrum["mz_udalton"][i], r) for i, r in zip(keep, rel)]
    ids = [spectrum["peak_id"][i] for i in keep]
    weight, anchors, ambiguous, explained = ref.targets(
        peaks, subgraphs, spectrum["adduct"], PPM_TENTHS, spectrum["mz_uncertainty_udalton"], ids)
    # Ties in weight are broken by the graph's smallest embedding, a key both
    # implementations can compute (the Rust reference breaks them by canonical trace;
    # the comparison below is per graph, so the order of tied graphs does not matter,
    # only which graphs survive the cut, which the report records).
    members = {}
    for g in subgraphs:
        members.setdefault(g["class"], []).append(list(g["atoms"]))
    kept, dropped = ref.retain(weight, order_key=lambda cls: min(members[cls]))
    by_id = dict(zip(ids, peaks))
    total = sum(r for _, r in peaks)
    detail = sorted(
        ({"embeddings": sorted(members[cls]), "weight": weight[cls], "q": float(q),
          "anchors": sorted(anchors[cls])} for cls, q in kept.items()),
        key=lambda t: t["embeddings"])
    return {
        "row": spectrum["row"], "peaks": len(peaks), "graphs": len(members),
        "embeddings": len(subgraphs), "targets_before_cut": len(weight), "targets": len(kept),
        "explained_peaks": len(explained),
        "explained_intensity": float(sum(by_id[p][1] for p in explained) / total) if total else 0.0,
        "dropped_weight": float(dropped), "ambiguous_hypotheses": ambiguous,
        "q_sorted": sorted((float(q) for q in kept.values()), reverse=True),
        "anchors": sum(len(anchors[c]) for c in kept),
        "explained_peak_ids": sorted(explained),
        "cut_is_tied": cut_is_tied(weight),
        "targets_detail": detail,
    }


def cut_is_tied(weight) -> bool:
    """Whether the top-16 cut falls between two graphs of equal weight."""
    ranked = sorted(weight.values(), reverse=True)
    return len(ranked) > ref.MAX_TARGETS and ranked[ref.MAX_TARGETS - 1] == ranked[ref.MAX_TARGETS]


def aggregate(rows):
    def q(key):
        v = np.array([r[key] for r in rows], dtype=np.float64)
        return {"mean": float(v.mean()), "p50": float(np.percentile(v, 50)), "p95": float(np.percentile(v, 95)),
                "max": float(v.max())}
    return {
        "spectra": len(rows), "labeled_fraction": float(np.mean([r["targets"] > 0 for r in rows])),
        "peaks": int(sum(r["peaks"] for r in rows)), "explained_peaks": int(sum(r["explained_peaks"] for r in rows)),
        "explained_intensity": q("explained_intensity"), "targets": q("targets"),
        "targets_before_cut": q("targets_before_cut"), "dropped_weight": q("dropped_weight"),
        "embeddings": q("embeddings"), "graphs": q("graphs"),
        "ambiguous_hypotheses": int(sum(r["ambiguous_hypotheses"] for r in rows)),
        "embeddings_per_graph": float(sum(r["embeddings"] for r in rows) / max(1, sum(r["graphs"] for r in rows))),
    }


def compare(mine, other):
    """Per-spectrum differences between two reports.

    Strict: the two reports must hold the same unique rows and every spectrum the
    same fields. Integers and lists compare exactly, floats within 1e-9. A spectrum
    whose top-16 cut is tied is compared on everything except which tied graphs were
    kept (the two implementations break that tie by different keys), and counted.
    """
    diffs = []
    rows_mine = [r["row"] for r in mine["spectra"]]
    rows_other = [r["row"] for r in other["spectra"]]
    if len(set(rows_mine)) != len(rows_mine) or len(set(rows_other)) != len(rows_other):
        diffs.append({"field": "duplicate rows"})
    if set(rows_mine) != set(rows_other):
        diffs.append({"field": "row sets differ", "only_python": sorted(set(rows_mine) - set(rows_other))[:10],
                      "only_other": sorted(set(rows_other) - set(rows_mine))[:10]})
    theirs = {r["row"]: r for r in other["spectra"]}
    tied = 0

    def same(a, b):
        if isinstance(a, float) or isinstance(b, float):
            return isinstance(a, (int, float)) and isinstance(b, (int, float)) and abs(a - b) <= 1e-9
        if isinstance(a, list):
            return isinstance(b, list) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
        if isinstance(a, dict):
            return isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
        return a == b

    for r in mine["spectra"]:
        o = theirs.get(r["row"])
        if o is None:
            continue
        if set(r) != set(o):
            diffs.append({"row": r["row"], "field": "field sets differ",
                          "only_python": sorted(set(r) - set(o)), "only_other": sorted(set(o) - set(r))})
            continue
        skip = set()
        if r["cut_is_tied"]:
            tied += 1
            skip = {"targets_detail", "anchors", "q_sorted"}
        for key, value in r.items():
            if key not in skip and not same(value, o[key]):
                diffs.append({"row": r["row"], "field": key, "python": value, "other": o[key]})
    return diffs, tied


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--compare", type=Path, default=None)
    args = ap.parse_args()

    data = json.loads(args.input.read_text())
    rows = []
    for m in data["molecules"]:
        atoms, bonds = m["atoms"], m["bonds"]
        mol = mol_from_graph(atoms, bonds)
        found = ref.enumerate_subgraphs(atoms, bonds)
        subgraphs = [{"atoms": members, "boundary": boundary, "closures": closures,
                      "counts": ref.composition([atoms[i] for i in members]),
                      "class": ref.fragment_smiles(mol, atoms, members)}
                     for members, (boundary, closures) in sorted(found.items())]
        for s in m["spectra"]:
            rows.append(spectrum_report(mol, atoms, subgraphs, s))
    rows.sort(key=lambda r: r["row"])
    report = {"schema_version": 1, "implementation": "python ms2_reference (RDKit identity)",
              "input": args.input.name, "recipe": "q-cut-v1", "ppm_tenths": PPM_TENTHS,
              "aggregate": aggregate(rows), "spectra": rows}
    if args.compare:
        other = json.loads(args.compare.read_text())
        diffs, tied = compare(report, other)
        report["comparison"] = {"other": other.get("implementation"), "differences": len(diffs),
                                "spectra_with_tied_cut": tied,
                                "first": diffs[:20], "other_aggregate": other.get("aggregate")}
        print(f"{len(diffs)} differing fields against {args.compare.name}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report["aggregate"], indent=1))


if __name__ == "__main__":
    main()
