"""P0 dataset audit for the MS2-to-substructure plan (docs/MS2_SUBSTRUCTURE_TASKS.md).

Reads the CASMI 2026 training file and the identity-disjoint folds and writes the
counts docs/MS2_CONTRACTS.md is frozen from: metadata conventions (P0.1, P0.4),
chemistry coverage (P0.3, P0.7) and the split inventory (P0.5). The structure
domain is the frozen predicate of `ms2_reference.classify`, the same one the
fixtures and the pilot use. Nothing here is an accuracy result.

The atom-type vocabulary was chosen from the training and validation folds only
(`fold_identity != 0`); the report gives the type counts of those folds and of the
test fold separately so that choice can be checked.

    uv run --project /Users/ods/Documents/Enveda_CASMI python tools/ms2/audit_casmi.py \
        --data /Users/ods/Documents/Enveda_CASMI/kobayashi/exp-casmi-26-from-spectra-to-structures/data \
        --out bench/results/ms2/casmi_audit.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import rdkit
from rdkit import Chem, RDLogger

from ms2_reference import ADDUCTS, EXACT, MASS, classify, mz_uncertainty, raw_atoms, stored_decimals

RDLogger.DisableLog("rdApp.*")

V0_ADDUCTS = tuple(name for name, _, _ in ADDUCTS.values())
# The ten adducts of the hidden test set (chem.TEST_ADDUCTS of the CASMI engine): the V1 adduct domain.
V1_ADDUCTS = ("[M+H]+", "[M+NH4]+", "[M-H2O+H]+", "[M-2H2O+H]+", "[M+Na]+", "[M+K]+",
              "[M-H]-", "[M-H2O-H]-", "[M+CH2O2-H]-", "[M+Cl]-")
MAX_PEAKS = 160  # FPNet's prep_peaks cap
PRECURSOR_RANGE = (50.0, 2000.0)
TEST_FOLD = 0


def quantiles(values, qs=(0, 1, 5, 25, 50, 75, 95, 99, 99.9, 100)) -> dict:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {}
    return {f"p{q}": float(np.percentile(values, q)) for q in qs}


def counter_table(counter: Counter) -> list:
    total = sum(counter.values())
    return [
        {"value": None if k is None else str(k), "count": int(v), "fraction": v / total}
        for k, v in counter.most_common()
    ]


def instrument_class(name) -> str:
    """Contract section 1: case-insensitive match on `instrument_type`."""
    v = (name or "").lower()
    if "timstof" in v:
        return "timstof"
    if "orbitrap" in v or "qft" in v or "itft" in v:
        return "orbitrap"
    if "tof" in v:
        return "qtof"
    return "other"


def audit_structures(structures_path: Path) -> tuple[dict, set, set]:
    t = pq.read_table(structures_path, columns=[
        "inchikey14", "smiles", "fold_identity", "fold_scaffold", "n_spectra", "identity_group"]).to_pandas()
    reasons = Counter()
    only_reason = Counter()
    types = {"train_validation": Counter(), "test": Counter()}
    type_structures = {"train_validation": Counter(), "test": Counter()}
    bond_types = Counter(); charges = Counter(); elements = Counter()
    heavy, rings = [], []
    in_v0 = np.zeros(len(t), bool)
    in_v1 = np.zeros(len(t), bool)
    for i, (smi, fold) in enumerate(zip(t.smiles.values, t.fold_identity.values)):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            reasons["unparsable"] += 1
            continue
        try:
            Chem.Kekulize(mol, clearAromaticFlags=True)
        except Exception:
            reasons["kekulize_failed"] += 1
            continue
        found = classify(mol)
        atoms = raw_atoms(mol)
        for a in atoms:
            elements[a["element"]] += 1
            charges[a["charge"]] += 1
        for b in mol.GetBonds():
            bond_types[str(b.GetBondType())] += 1
        heavy.append(mol.GetNumHeavyAtoms())
        rings.append(mol.GetNumBonds() - (mol.GetNumAtoms() - 1))
        # V1 structure domain: the same elements, formal charges of -1, 0, +1, any atom type.
        v1_bad = set(found) - {"formal_charge", "unsupported_atom_type"}
        if not v1_bad and all(abs(a["charge"]) <= 1 for a in atoms):
            in_v1[i] = True
        # Atom types of structures with no defect other than (possibly) an unlisted type:
        # the population the vocabulary is chosen from.
        if not set(found) - {"unsupported_atom_type"}:
            split = "test" if fold == TEST_FOLD else "train_validation"
            local = [(a["element"], a["hydrogens"], a["valence"]) for a in atoms]
            for key in local:
                types[split][key] += 1
            for key in set(local):
                type_structures[split][key] += 1
        if found:
            for r in found:
                reasons[r] += 1
            if len(found) == 1:
                only_reason[found[0]] += 1
            # An explicit hydrogen vertex is reported under element_outside_domain; split it out.
            outside = {a["element"] for a in atoms if a["element"] not in MASS or a["element"] == "H"}
            if outside == {"H"}:
                reasons["element_outside_domain_only_explicit_hydrogen"] += 1
            continue
        in_v0[i] = True
    spectra = t.n_spectra.values
    all_types = sorted(set(types["train_validation"]) | set(types["test"]),
                       key=lambda k: -types["train_validation"][k])
    report = {
        "structures": int(len(t)),
        "in_v0_structure_domain": int(in_v0.sum()),
        "in_v0_structure_domain_fraction": float(in_v0.mean()),
        "spectra_of_v0_structures_fraction": float(spectra[in_v0].sum() / spectra.sum()),
        "in_v1_structure_domain": int(in_v1.sum()),
        "exclusion_reasons_structures": {k: int(v) for k, v in reasons.most_common()},
        "structures_with_exactly_one_reason": {k: int(v) for k, v in only_reason.most_common()},
        "element_atom_counts": {k: int(v) for k, v in elements.most_common()},
        "formal_charge_atom_counts": {str(k): int(v) for k, v in sorted(charges.items())},
        "bond_type_counts_after_kekulize": {k: int(v) for k, v in bond_types.most_common()},
        "heavy_atoms": quantiles(heavy),
        "ring_closures": quantiles(rings),
        "joint_atom_types": [
            {"element": e, "hydrogens": h, "valence": v,
             "train_validation_atoms": int(types["train_validation"][(e, h, v)]),
             "train_validation_structures": int(type_structures["train_validation"][(e, h, v)]),
             "test_atoms": int(types["test"][(e, h, v)]),
             "test_structures": int(type_structures["test"][(e, h, v)])}
            for e, h, v in all_types
        ],
        "validation_fold_parts_in_v0_domain": {
            name: int(((t.fold_identity == 1) & (t.identity_group % 3 == part) & in_v0).sum())
            for part, name in enumerate(("validation", "ranking", "calibration"))
        },
        "identity_groups": int(t.identity_group.nunique()),
        "folds": {
            name: {
                str(k): {"structures": int((t[name] == k).sum()),
                         "in_v0_domain": int(((t[name] == k) & in_v0).sum())}
                for k in sorted(t[name].unique())
            }
            for name in ("fold_identity", "fold_scaffold")
        },
    }
    # Keyed by the structure's own SMILES: 1,612 InChIKey14 values cover more than one
    # stored structure (tautomers), and those need not share a domain status.
    return report, set(t.smiles.values[in_v0]), set(t.smiles.values[in_v1])


def audit_spectra(train_path: Path, peak_groups: int, v0_keys: set, v1_keys: set) -> dict:
    f = pq.ParquetFile(train_path)
    meta_cols = [
        "ingest_lib", "ionization_mode", "instrument_type", "adduct", "precursor_mz",
        "precursor_error_ppm", "num_peaks", "collision_energy_ev", "collision_energy_orig_units",
        "normalized_smiles",
    ]
    counters = {k: Counter() for k in (
        "ingest_lib", "ionization_mode", "instrument_type", "adduct",
        "collision_energy_orig_units", "energy_count", "instrument_class")}
    v0 = Counter(); v1 = Counter()
    units_known = Counter()
    prec, ppm, npk, ce_all = [], [], [], []
    rows = 0
    for g in range(f.metadata.num_row_groups):
        t = f.read_row_group(g, columns=meta_cols)
        rows += t.num_rows
        for name in ("ingest_lib", "ionization_mode", "instrument_type", "adduct",
                     "collision_energy_orig_units"):
            for e in pc.value_counts(t[name]).to_pylist():
                counters[name][e["values"]] += e["counts"]
        prec.append(t["precursor_mz"].to_numpy()); ppm.append(t["precursor_error_ppm"].to_numpy())
        npk.append(t["num_peaks"].to_numpy())
        for name in t["instrument_type"].to_pylist():
            counters["instrument_class"][instrument_class(name)] += 1
        for c, u in zip(t["collision_energy_ev"].to_pylist(), t["collision_energy_orig_units"].to_pylist()):
            n = 0 if c is None else len(c)
            counters["energy_count"][n] += 1
            units_known[(u, n > 0)] += 1
            if c:
                ce_all.extend(c)
        # Request domain: one exclusion per spectrum, the first that applies, in this order:
        # adduct, polarity, structure, precursor range.
        for k, a, m, p, e in zip(t["normalized_smiles"].to_pylist(), t["adduct"].to_pylist(),
                                 t["ionization_mode"].to_pylist(), t["precursor_mz"].to_numpy(),
                                 t["precursor_error_ppm"].to_numpy()):
            want = {"positive": "+", "negative": "-"}.get(m, "?")
            for domain, adducts, keys in ((v0, V0_ADDUCTS, v0_keys), (v1, V1_ADDUCTS, v1_keys)):
                domain["spectra"] += 1
                if a not in adducts:
                    domain["excluded_adduct"] += 1
                elif not a.endswith(want):
                    domain["excluded_polarity_conflict"] += 1
                elif k not in keys:
                    domain["excluded_structure"] += 1
                elif not (np.isfinite(p) and PRECURSOR_RANGE[0] <= p <= PRECURSOR_RANGE[1]):
                    domain["excluded_precursor_range"] += 1
                else:
                    domain["in_domain"] += 1
                    for tol in (5, 10, 20, 50):
                        domain[f"in_domain_precursor_within_{tol}ppm"] += bool(np.isfinite(e) and abs(e) <= tol)
    prec = np.concatenate(prec); ppm = np.concatenate(ppm); npk = np.concatenate(npk)

    # Peak-level statistics need the two list columns, most of the 3 GB: sample evenly
    # spaced row groups rather than reading them all.
    groups = sorted(set(np.linspace(0, f.metadata.num_row_groups - 1, peak_groups).round().astype(int).tolist()))
    peak = Counter(); decimals = Counter(); uncertainty = Counter()
    mz_max, kept_after_prep, f32_err_ppm, retained_intensity = [], [], [], []
    for g in groups:
        t = f.read_row_group(int(g), columns=["ms2_mzs", "ms2_normalized_intensities", "precursor_mz"])
        for mz, it, p in zip(t["ms2_mzs"].to_pylist(), t["ms2_normalized_intensities"].to_pylist(),
                             t["precursor_mz"].to_numpy()):
            peak["spectra"] += 1
            if not mz:
                peak["empty"] += 1
                continue
            mz = np.asarray(mz, np.float64); it = np.asarray(it, np.float64)
            peak["peaks"] += mz.size
            peak["unsorted"] += bool(np.any(np.diff(mz) < 0))
            peak["has_duplicate_mz"] += bool(np.any(np.diff(np.sort(mz)) == 0))
            peak["nonfinite"] += bool(not (np.all(np.isfinite(mz)) and np.all(np.isfinite(it))))
            peak["negative_intensity"] += bool(np.any(it < 0))
            peak["zero_max_intensity"] += bool(it.max() <= 0)
            peak["max_intensity_not_one"] += bool(abs(it.max() - 1.0) > 1e-6)
            peak["peak_above_precursor_plus_2"] += bool(np.any(mz > p + 2.0))
            peak["over_512_raw_peaks"] += bool(mz.size > 512)
            mz_max.append(float(mz.max()))
            # prep_peaks, reproduced, to measure what the FPNet contract discards.
            keep = (mz <= p + 2.0) & (mz > 0)
            m2, i2 = mz[keep], it[keep]
            if m2.size == 0 or i2.max() <= 0:
                peak["empty_after_prep"] += 1
                continue
            i2 = i2 / i2.max()
            keep = i2 >= 1e-3
            m2, i2 = m2[keep], i2[keep]
            total = float(i2.sum())
            if m2.size > MAX_PEAKS:
                sel = np.argsort(-i2)[:MAX_PEAKS]
                peak["truncated_by_cap_160"] += 1
                retained_intensity.append(float(i2[sel].sum()) / total)
                m2 = m2[sel]
            peak["truncated_by_cap_128"] += bool(keep.sum() > 128)
            kept_after_prep.append(m2.size)
            err = np.abs(m2.astype(np.float32).astype(np.float64) - m2) / m2 * 1e6
            f32_err_ppm.append(float(err.max()))
            if peak["spectra"] % 20 == 0:
                d = stored_decimals(mz.tolist())
                decimals[d] += 1
                uncertainty[mz_uncertainty(d)] += 1
    return {
        "rows": int(rows),
        "row_groups": int(f.metadata.num_row_groups),
        "exclusion_order": ["adduct", "polarity_conflict", "structure", "precursor_range"],
        "v0_adducts": list(V0_ADDUCTS), "v0_request_domain": {k: int(v) for k, v in v0.items()},
        "v1_adducts": list(V1_ADDUCTS), "v1_request_domain": {k: int(v) for k, v in v1.items()},
        "categorical": {k: counter_table(v) for k, v in counters.items()},
        "energy_units_by_known_value": [
            {"units": u, "has_ev_values": k, "count": int(v)} for (u, k), v in units_known.most_common()],
        "precursor_mz": quantiles(prec),
        "abs_precursor_error_ppm": quantiles(np.abs(ppm)),
        "precursor_error_ppm_missing": int(np.isnan(ppm).sum()),
        "num_peaks": quantiles(npk),
        "collision_energy_ev": quantiles(ce_all),
        "peak_sample": {
            "row_groups_sampled": groups,
            "flags": {k: int(v) for k, v in peak.items()},
            "mz_max": quantiles(mz_max),
            "peaks_kept_by_prep_peaks": quantiles(kept_after_prep),
            "f32_cast_error_ppm_max_per_spectrum": quantiles(f32_err_ppm),
            "retained_intensity_when_truncated_at_160": quantiles(retained_intensity),
            "stored_decimals_per_spectrum": {str(k): int(v) for k, v in sorted(decimals.items())},
            "mz_uncertainty_udalton_per_spectrum": {str(k): int(v) for k, v in sorted(uncertainty.items())},
        },
    }


def mass_table_check() -> list:
    """The contract's element masses against RDKit's periodic table, in micro-dalton."""
    table = Chem.GetPeriodicTable()
    return [
        {"symbol": e, "contract_exact": EXACT[e], "contract_udalton": MASS[e],
         "rdkit": repr(table.GetMostCommonIsotopeMass(e)),
         "rdkit_minus_contract_udalton": (table.GetMostCommonIsotopeMass(e) - float(EXACT[e])) * 1e6}
        for e in EXACT
    ]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--peak-groups", type=int, default=4, help="row groups sampled for peak-level statistics")
    ap.add_argument("--skip-hash", action="store_true")
    args = ap.parse_args()

    train = args.data / "raw" / "train.parquet"
    structures = args.data / "folds" / "structures.parquet"
    structure_report, v0_keys, v1_keys = audit_structures(structures)
    report = {
        "schema_version": 3,
        "rdkit": rdkit.__version__,
        "train_file": str(train),
        "train_sha256": None if args.skip_hash else sha256(train),
        "structures_file": str(structures),
        "mass_table_check": mass_table_check(),
        "spectra": audit_spectra(train, args.peak_groups, v0_keys, v1_keys),
        "structures": structure_report,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
