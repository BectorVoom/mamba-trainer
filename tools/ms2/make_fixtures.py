"""Independent chemistry fixtures for the MS2 CPU oracle (P1 of docs/MS2_SUBSTRUCTURE_TASKS.md).

Everything the Rust reference in `src/models/ms2/` is tested against is derived
here a second time, from docs/MS2_CONTRACTS.md, with RDKit and Python integers:
domain classification, integer masses, adduct algebra, tolerances, bond-cut
subgraph enumeration, graph identity (RDKit canonical fragment SMILES), canonical
traces (exhaustive search, no pruning), per-prefix legality masks and the
pseudo-label weights of synthetic spectra. No value is copied from Rust output.

The molecules are well-known public structures written here as SMILES, and the
spectra are synthetic, so the fixture carries no licensed data.

    uv run --project /Users/ods/Documents/Enveda_CASMI python tools/ms2/make_fixtures.py \
        --out tests/fixtures/ms2/chemistry_v0.json
"""
from __future__ import annotations

import argparse
import json
import random
from decimal import Decimal
from collections import Counter
from pathlib import Path

import rdkit
from rdkit import Chem
from rdkit.Chem import Descriptors

from ms2_reference import *  # noqa: F401,F403 - the fixture is a dump of the whole reference

IN_DOMAIN = [
    ("ethanol", "CCO"),
    ("acetic acid", "CC(=O)O"),
    ("benzene", "c1ccccc1"),
    ("cyclopropane", "C1CC1"),
    ("neopentane", "CC(C)(C)C"),
    ("aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
    ("paracetamol", "CC(=O)Nc1ccc(O)cc1"),
    ("caffeine", "Cn1cnc2c1c(=O)n(C)c(=O)n2C"),
    ("nicotine", "CN1CCCC1c1cccnc1"),
    ("ibuprofen", "CC(C)Cc1ccc(cc1)C(C)C(=O)O"),
    ("methionine", "CSCCC(N)C(=O)O"),
    ("cysteine", "NC(CS)C(=O)O"),
    ("sulfanilamide", "Nc1ccc(cc1)S(N)(=O)=O"),
    ("glyphosate", "OC(=O)CNCP(O)(O)=O"),
    ("fluoxetine", "CNCCC(Oc1ccc(cc1)C(F)(F)F)c1ccccc1"),
    ("chloramphenicol-like", "OCC(NC(=O)C(Cl)Cl)C(O)c1ccccc1"),
    ("bromobenzene", "Brc1ccccc1"),
    ("iodoform", "IC(I)I"),
    ("adamantane", "C1C2CC3CC1CC(C2)C3"),
    ("cubane", "C12C3C4C1C5C2C3C45"),
    ("acetonitrile", "CC#N"),
    ("naphthalene", "c1ccc2ccccc2c1"),
    ("glucose", "OCC1OC(O)C(O)C(O)C1O"),
    ("2-butanol", "CCC(C)O"),
    ("isobutanol", "CC(C)CO"),
    # 16 atoms and 4 ring closures: the longest legal trace (22 tokens).
    ("pyrene", "c1cc2ccc3cccc4ccc(c1)c2c34"),
    # CH3-C-CH3 occurs twice as the same labeled graph, once with one boundary bond
    # (the double bond leaves) and once with two.
    ("methylheptenol", "CC(C)=CCC(C)(C)O"),
    # Stereo centres: identity must ignore them.
    ("stereo diol", "CC[C@H](O)CCCC[C@@H](O)CC"),
]
OUT_OF_DOMAIN = [
    ("nitrobenzene", "[O-][N+](=O)c1ccccc1"),
    ("tetramethylammonium", "C[N+](C)(C)C"),
    ("trimethylsilanol", "C[Si](C)(C)O"),
    ("deuterochloroform", "[2H]C(Cl)(Cl)Cl"),
    ("methyl radical", "[CH3]"),
    ("two components", "CC(=O)O.CCO"),
    ("thioanisole S-oxide", "CS(=O)c1ccccc1"),
    ("phosphine", "CP(C)C"),
]


def labeled_subgraphs(atoms, subgraphs):
    return [dict(g, counts=composition([atoms[i] for i in g["atoms"]])) for g in subgraphs]


def spectrum_record(peaks, labeled, adduct_id, ppm_tenths, uncertainty, max_targets=None):
    """One fixture spectrum: peaks as [peak_id, m/z, intensity] and the reference labels."""
    peaks = sorted(peaks)
    dedup = []
    for mz, it in peaks:
        if not dedup or dedup[-1][0] != mz:
            dedup.append([mz, it])
    # Ids are deliberately not positions, so a positional anchor is caught.
    ids = [2 * i + 3 for i in range(len(dedup))]
    weight, anchors, ambiguous, explained = targets(dedup, labeled, adduct_id, ppm_tenths, uncertainty, ids)
    ranked = sorted(weight.items(), key=lambda kv: (-kv[1], kv[0]))
    total = sum(weight.values())
    record = {
        "adduct": adduct_id, "ppm_tenths": ppm_tenths, "mz_uncertainty": uncertainty,
        "peaks": [[i, mz, float(it)] for i, (mz, it) in zip(ids, dedup)],
        "ambiguous_hypotheses": ambiguous, "explained_peaks": sorted(explained),
        "targets": [{"class": cls, "weight": w, "q": w / total, "anchors": sorted(anchors[cls])}
                    for cls, w in ranked],
    }
    if max_targets is not None:
        record["max_targets"] = max_targets
    return record, weight


def build_molecule(name, smiles, rng, with_traces):
    mol = kekulized(smiles)
    assert classify(mol) == [], (name, classify(mol))
    atoms, bonds = graph_of(mol)
    counts = composition(atoms)
    found = enumerate_subgraphs(atoms, bonds)
    classes = {}
    subgraphs = []
    for members in sorted(found):
        boundary, closures = found[members]
        smi = fragment_smiles(mol, atoms, members)
        cls = classes.setdefault(smi, len(classes))
        sub_counts = composition([atoms[i] for i in members])
        entry = {"atoms": list(members), "boundary": boundary, "closures": closures,
                 "mass_udalton": mass_of(sub_counts), "class": cls}
        if with_traces and len(members) <= 9:
            types = [atoms[i] for i in members]
            trace = canonical_trace(types, sub_adjacency(members, bonds))
            masks, residual = legality_masks(trace, dict(counts))
            entry["canonical_trace"] = [list(t) for t in trace]
            entry["legal_masks_parent_budget"] = masks
            entry["legal_masks_no_budget"] = legality_masks(trace, None)[0]
            entry["open_valence_canonical_order"] = residual
        subgraphs.append(entry)
    formula = {e: counts[e] for e in ELEMENT_ORDER if counts[e]}
    exact = sum(Decimal(EXACT[e]) * n for e, n in counts.items())
    record = {
        "name": name, "smiles": smiles, "raw_atoms": raw_atoms(mol), "atoms": atoms, "bonds": bonds,
        "formula": formula, "mass_exact": str(exact), "mass_udalton": mass_of(counts),
        "mass_error_nda": error_nda(counts),
        "rdkit_mass": f"{Descriptors.ExactMolWt(Chem.MolFromSmiles(smiles)):.9f}",
        "identity_classes": len(classes), "subgraphs": subgraphs,
    }
    # One legal (not canonical) trace of the whole molecule, when it fits the grammar:
    # legality masks for graphs too large for the exhaustive canonical search.
    ring_closures = len(bonds) - (len(atoms) - 1)
    if len(atoms) <= MAX_ATOMS and ring_closures <= MAX_CLOSURES:
        whole = first_bfs_trace(atoms, sub_adjacency(tuple(range(len(atoms))), bonds))
        masks, residual = legality_masks(whole, None)
        record["whole_trace"] = {"trace": [list(t) for t in whole], "legal_masks_no_budget": masks,
                                 "legal_masks_parent_budget": legality_masks(whole, dict(counts))[0],
                                 "open_valence": residual}
    labeled = labeled_subgraphs(atoms, subgraphs)
    # A synthetic spectrum: ions of sampled subgraphs at chosen offsets from their exact m/z.
    spectra = []
    if subgraphs:
        for adduct_id in (1, 2):
            ppm_tenths = 100
            peaks = []
            picks = rng.sample(subgraphs, min(6, len(subgraphs)))
            for j, g in enumerate(picks):
                g_counts = composition([atoms[i] for i in g["atoms"]])
                limit = min(g["boundary"], MAX_SHIFT)
                s = rng.randint(-limit, limit)
                hyp = ion(g_counts, adduct_id, s)
                if hyp is None:
                    continue
                mz, err = hyp
                tol_at = tolerance(mz, ppm_tenths)
                # exact, just inside, on the ambiguity band, and just outside the tolerance
                offset = [0, tol_at // 2, tol_at, tol_at + err + 2, -(tol_at // 3), 0][j % 6]
                peaks.append([mz + offset, (j + 1) / 8])
            peaks.append([peaks[0][0] + 373_100 if peaks else 100_373_100, 0.5])  # noise
            # Half a unit of the fourth decimal for the positive spectrum, none for the negative.
            uncertainty = 50 if adduct_id == 1 else 0
            spectra.append(spectrum_record(peaks, labeled, adduct_id, ppm_tenths, uncertainty)[0])
    record["spectra"] = spectra
    return record, labeled, atoms


def special_spectra(molecules, context):
    """Hand-directed spectra for the cases random sampling does not reach."""
    out = {}
    by_name = {m["name"]: (m, *context[m["name"]]) for m in molecules}

    # Retention tie: two graphs with canonical traces, one exclusive peak each of equal
    # intensity. With one target kept, the smaller trace wins.
    record, labeled, atoms = by_name["ibuprofen"]
    traced = {g["class"]: g for g in record["subgraphs"] if "canonical_trace" in g}
    done = False
    for a in sorted(traced):
        for b in sorted(traced):
            if a >= b or done:
                continue
            peaks = [[ion(composition([atoms[i] for i in traced[c]["atoms"]]), 1, 0)[0], 0.5] for c in (a, b)]
            if peaks[0][0] == peaks[1][0]:
                continue
            spec, weight = spectrum_record(peaks, labeled, 1, 100, 0, max_targets=1)
            top = sorted(weight.values(), reverse=True)
            if len(weight) == 2 and top[0] == top[1]:
                winner = min((a, b), key=lambda c: traced[c]["canonical_trace"])
                spec.update({"molecule": "ibuprofen", "tied_classes": [a, b], "kept_classes": [winner],
                             "dropped_weight": 0.5})
                out["retention_tie"] = spec
                done = True
    assert done, "no exclusive tie pair found"

    # More than 16 graphs with weight: the cut keeps the 16 heaviest.
    record, labeled, atoms = by_name["chloramphenicol-like"]
    seen, peaks = set(), []
    for g in record["subgraphs"]:
        mz = ion(composition([atoms[i] for i in g["atoms"]]), 1, 0)[0]
        if mz in seen or len(peaks) >= 24:
            continue
        seen.add(mz)
        peaks.append([mz, (len(peaks) + 8) / 32])
    spec, weight = spectrum_record(peaks, labeled, 1, 100, 0, max_targets=MAX_TARGETS)
    ranked = sorted(weight.items(), key=lambda kv: (-kv[1], kv[0]))
    assert len(ranked) > MAX_TARGETS and ranked[MAX_TARGETS - 1][1] != ranked[MAX_TARGETS][1], "cut is ambiguous"
    kept = ranked[:MAX_TARGETS]
    spec.update({"molecule": "chloramphenicol-like", "kept_classes": [c for c, _ in kept],
                 "dropped_weight": 1.0 - sum(w for _, w in kept) / sum(weight.values())})
    out["over_sixteen"] = spec

    # One graph, two embeddings with different boundary counts: a peak two hydrogens
    # below the plain ion is reachable only through the embedding with two boundary bonds.
    record, labeled, atoms = by_name["methylheptenol"]
    by_class = {}
    for g in record["subgraphs"]:
        by_class.setdefault(g["class"], []).append(g)
    cls, group = next((c, gs) for c, gs in sorted(by_class.items()) if len({g["boundary"] for g in gs}) > 1)
    wide = max(group, key=lambda g: g["boundary"])
    assert wide["boundary"] == 2 and min(g["boundary"] for g in group) == 1
    counts = composition([atoms[i] for i in wide["atoms"]])
    peaks = [[ion(counts, 1, -2)[0], 1.0], [ion(counts, 1, 0)[0], 0.25]]
    spec, weight = spectrum_record(peaks, labeled, 1, 100, 0)
    assert any(t["class"] == cls and any(s == -2 for _, s in t["anchors"]) for t in spec["targets"])
    spec.update({"molecule": "methylheptenol", "class": cls, "boundaries": sorted(g["boundary"] for g in group)})
    out["differing_boundary"] = spec

    # Unknown observation precision: nothing is decided, so nothing is labeled or counted.
    record, labeled, atoms = by_name["aspirin"]
    g = record["subgraphs"][0]
    peaks = [[ion(composition([atoms[i] for i in g["atoms"]]), 1, 0)[0], 1.0]]
    spec, weight = spectrum_record(peaks, labeled, 1, 100, UNKNOWN_UNCERTAINTY)
    assert not weight and spec["ambiguous_hypotheses"] == 0
    spec["molecule"] = "aspirin"
    out["unknown_uncertainty"] = spec
    return out


def ion_cases():
    """Ion m/z against exact decimal arithmetic, with the error interval in nano-units."""
    cases = []
    electron = Decimal(ELECTRON_EXACT)
    for formula in ({"C": 6, "H": 12}, {"C": 9, "H": 8, "O": 4}, {"C": 2, "H": 3, "Cl": 3, "O": 2},
                    {"C": 16, "H": 10}, {"S": 1, "O": 3}, {"C": 1, "H": 1, "I": 3}, {"C": 3, "H": 8, "N": 1, "O": 5, "P": 1},
                    {"C": 6, "H": 5, "Br": 1}, {"C": 2, "F": 4}, {"O": 2}):
        counts = Counter(formula)
        for adduct_id, (_, h_a, z) in ADDUCTS.items():
            for shift in (-2, -1, 0, 1, 2):
                hyp = ion(counts, adduct_id, shift)
                case = {"formula": formula, "adduct": adduct_id, "shift": shift}
                if hyp is None:
                    case["none"] = True
                else:
                    exact = (sum(Decimal(EXACT[e]) * n for e, n in counts.items())
                             + (h_a + shift) * Decimal(EXACT["H"]) - z * electron) * 10**9
                    case.update({"mz_udalton": hyp[0], "error_udalton": hyp[1],
                                 "exact_nda_floor": int(exact.to_integral_value(rounding="ROUND_FLOOR")),
                                 "exact_nda_ceil": int(exact.to_integral_value(rounding="ROUND_CEILING"))})
                    assert hyp[0] * 1000 - hyp[1] * 1000 <= case["exact_nda_floor"]
                    assert case["exact_nda_ceil"] <= hyp[0] * 1000 + hyp[1] * 1000
                cases.append(case)
    return cases


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rng = random.Random(20261002)

    built = [build_molecule(n, s, rng, with_traces=True) for n, s in IN_DOMAIN]
    molecules = [record for record, _, _ in built]
    context = {record["name"]: (labeled, atoms) for record, labeled, atoms in built}
    outside = []
    for name, smiles in OUT_OF_DOMAIN:
        mol = Chem.MolFromSmiles(smiles)
        Chem.Kekulize(mol, clearAromaticFlags=True)
        reasons = classify(mol)
        assert reasons, name
        bonds = sorted((min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()), max(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
                        int(b.GetBondTypeAsDouble())) for b in mol.GetBonds())
        outside.append({"name": name, "smiles": smiles, "raw_atoms": raw_atoms(mol),
                        "bonds": [list(b) for b in bonds], "reasons": reasons})

    adduct_cases = []
    for text in ("301.007300", "195.087652", "50.000000", "1999.999999", "89.0244", "330.1237"):
        mz = to_int(text)
        adduct_cases.append({
            "mz_decimal": text, "mz_udalton": mz,
            "parent_udalton": {"1": mz - MASS["H"] + ELECTRON, "2": mz + MASS["H"] - ELECTRON},
        })
    tolerance_cases = [
        {"mz_udalton": mz, "ppm_tenths": t, "tolerance_udalton": tolerance(mz, t)}
        for mz in (0, 1, 9_999, 10_000, 50_000_000, 330_123_700, 1_999_999_999, 4_294_967_295)
        for t in (1, 50, 100, 200, 999, 1000)
    ]
    decision_cases = []
    for observed, computed, error, tol in [
        (100_000_500, 100_000_000, 3, 1000), (100_000_997, 100_000_000, 3, 1000),
        (100_000_998, 100_000_000, 3, 1000), (100_001_003, 100_000_000, 3, 1000),
        (100_001_004, 100_000_000, 3, 1000), (99_998_996, 100_000_000, 3, 1000),
        (100_000_000, 100_000_000, 0, 0), (100_000_001, 100_000_000, 0, 0),
    ]:
        decision_cases.append({"observed": observed, "computed": computed, "error": error, "tolerance": tol,
                               "verdict": decide(observed, computed, error, tol)})

    for name, trace in INVALID_TRACES:
        assert first_illegal_step(trace) is not None, name

    # The root step under budgets that leave few or no atom types.
    root_budget_cases = []
    for budget in ({}, {"H": 4}, {"O": 1}, {"O": 1, "H": 1}, {"C": 1, "H": 2}, {"S": 1}, {"C": 2, "H": 6, "O": 1}):
        state = Replay(dict(budget))
        state.apply((START, 0, 0, 0))
        masks, _ = state.masks((ADD, 1, 0, 0))
        root_budget_cases.append({"budget": budget, "kinds": masks[0], "atom_types": masks[1]})

    report = {
        "schema_version": 2, "chemistry": "ms2-chem-v0.1", "grammar": "grammar-bfs-v1",
        "traversal": "bfs-canon-v1", "recipe": "q-cut-v1", "rdkit": rdkit.__version__,
        "mass_scale": SCALE,
        "elements": [{"symbol": e, "exact": EXACT[e], "udalton": MASS[e], "residual_nda": RESIDUAL[e]}
                     for e in ELEMENT_ORDER],
        "electron": {"exact": ELECTRON_EXACT, "udalton": ELECTRON, "residual_nda": ELECTRON_RESIDUAL},
        "atom_types": [{"id": i, "element": e, "hydrogens": h, "valence": v} for i, (e, h, v) in ATOM_TYPES.items()],
        "adducts": [{"id": i, "name": n, "hydrogens": h, "charge": z} for i, (n, h, z) in ADDUCTS.items()],
        "limits": {"max_atoms": MAX_ATOMS, "min_atoms": MIN_ATOMS, "max_closures": MAX_CLOSURES,
                   "max_cuts": MAX_CUTS, "max_shift": MAX_SHIFT, "max_targets": MAX_TARGETS},
        "weight_units": {"intensity": INTENSITY_UNITS, "split": SPLIT_UNITS},
        "token_kinds": {"pad": PAD, "start": START, "add_atom": ADD, "close_ring": CLOSE, "stop": STOP},
        "molecules": molecules, "out_of_domain": outside,
        "invalid_traces": [
            {"name": name, "trace": [list(t) for t in trace], "first_illegal_step": first_illegal_step(trace)}
            for name, trace in INVALID_TRACES
        ],
        "root_budget_cases": root_budget_cases,
        "special_spectra": special_spectra(molecules, context),
        "ion_cases": ion_cases(),
        "ion_errors": [
            {"formula": {"C": 400}, "adduct": 1, "shift": 0, "error": "mass_overflow"},
            {"formula": {"C": 1, "H": 4}, "adduct": 9, "shift": 0, "error": "unsupported_adduct"},
        ],
        "parent_mass_errors": [{"precursor_mz_udalton": 0, "adduct": 1}, {"precursor_mz_udalton": 4_294_967_295, "adduct": 2}],
        "adduct_cases": adduct_cases, "tolerance_cases": tolerance_cases, "decision_cases": decision_cases,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, separators=(",", ":")))
    n_sub = sum(len(m["subgraphs"]) for m in molecules)
    n_tr = sum(1 for m in molecules for g in m["subgraphs"] if "canonical_trace" in g)
    print(f"wrote {args.out}: {len(molecules)} molecules, {n_sub} subgraphs, {n_tr} canonical traces, "
          f"{args.out.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
