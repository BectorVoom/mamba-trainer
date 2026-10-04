"""Formula table from MassSpecGym train formulas united with ChEBI 3-star formulas.

Third candidate source for plan item P4.9 (docs/MS2_SUBSTRUCTURE_TASKS.md):
the distinct molecular formulas of the in-domain MassSpecGym `train` structures
(exactly the rows `formula_table_msgym.py` builds) united with the formulas of
an independent public structure database (ChEBI 3-star SDF). No validation or
test structure is ever read to build rows; validation-fold in-domain structures
only supply the coverage denominator, computed exactly as `formula_table_msgym`
does. `test`-fold TSV rows are skipped at parse time and never read.

Row schema, sort and table JSON schema match `formula_table.py --table-out`
via the shared `build_table` / `table_payload` / `build_report` helpers.
`src/models/ms2/formula.rs` `FormulaTable::from_json` only checks
`elements`/`rows` (no version string), so the report `version` differs from
the train-only table's (`ms2-formula-v0`) to tell model configs apart.

    PYTHONPATH=tools/ms2 uv run --with rdkit --with numpy --with pyarrow \
        python tools/ms2/formula_table_db.py --out /tmp/ms2_formula_table_db_report.json \
        --table-out data/ms2/formula_table_db_v1.json
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors

import ms2_reference as ref
from formula_table import build_report, build_table, table_payload
from formula_table_msgym import collect as collect_msgym

RDLogger.DisableLog("rdApp.*")

VERSION = "ms2-formula-db-v1"
U16_MAX = 2**16 - 1

# Rejection reasons in the order they are checked; a record counts once,
# under the first reason that rejects it.
REJECTION_ORDER = (
    "parse_error",
    "disconnected",
    "formal_charge",
    "radical",
    "isotope_label",
    "element_outside_domain",
    "formula_mismatch",
    "count_overflow",
    "mass_over_limit",
    "no_carbon",
)

_FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")


def parse_formula(text):
    """{element: count} from a plain molecular-formula string (no charge/isotope)."""
    out = {}
    for element, number in _FORMULA_TOKEN.findall(text):
        out[element] = out.get(element, 0) + (int(number) if number else 1)
    return out


def collect_chebi(sdf_path, max_mass_da):
    """Distinct formula keys from the ChEBI SDF plus rejection counts.

    A record contributes its formula (heavy-atom counts plus total hydrogens,
    implicit and explicit, via `GetTotalNumHs()` on a `RemoveHs` copy; any
    residual explicit H is counted directly) when every check passes.
    This is a FORMULA source, not a structure-domain claim: `ref.classify`
    is deliberately NOT applied (ChEBI atom types need not be in the V0
    atom-type vocabulary). Only element/charge/count/mass/carbon rules apply.
    The composition's integer mass comes from `ms2_reference`'s mass function,
    cross-checked against RDKit's `CalcMolFormula`.
    """
    allowed = set(ref.ELEMENT_ORDER)  # the 10 domain elements, C H N O F P S Cl Br I
    max_mass = int(round(max_mass_da * ref.SCALE))
    rows = set()
    rejections = Counter()
    records, accepted = 0, 0
    with gzip.open(sdf_path, "rb") as fh:
        suppl = Chem.ForwardSDMolSupplier(fh)
        while True:
            try:
                mol = next(suppl)
            except StopIteration:
                break
            except Exception:
                records += 1
                rejections["parse_error"] += 1
                continue
            records += 1
            if mol is None:
                rejections["parse_error"] += 1
                continue
            try:
                fragments = Chem.GetMolFrags(mol)
            except Exception:
                rejections["parse_error"] += 1
                continue
            if len(fragments) != 1:
                rejections["disconnected"] += 1
                continue
            atoms = list(mol.GetAtoms())
            if Chem.GetFormalCharge(mol) != 0 or any(a.GetFormalCharge() != 0 for a in atoms):
                rejections["formal_charge"] += 1
                continue
            if any(a.GetNumRadicalElectrons() != 0 for a in atoms):
                rejections["radical"] += 1
                continue
            if any(a.GetIsotope() != 0 for a in atoms):
                rejections["isotope_label"] += 1
                continue
            if any(a.GetSymbol() not in allowed for a in atoms):
                rejections["element_outside_domain"] += 1
                continue
            try:
                bare = Chem.RemoveHs(mol)
            except Exception:
                rejections["parse_error"] += 1
                continue
            counts = {}
            for a in bare.GetAtoms():
                symbol = a.GetSymbol()
                if symbol == "H":
                    # Residual explicit H (RemoveHs leaves Hs it cannot merge;
                    # the heavy neighbour's GetTotalNumHs() does not count them).
                    counts["H"] = counts.get("H", 0) + 1
                else:
                    counts[symbol] = counts.get(symbol, 0) + 1
                    counts["H"] = counts.get("H", 0) + a.GetTotalNumHs()
            try:
                formula_counts = parse_formula(rdMolDescriptors.CalcMolFormula(mol))
            except Exception:
                formula_counts = None
            if formula_counts is None or any(
                counts.get(e, 0) != n for e, n in formula_counts.items()
            ) or any(e not in formula_counts for e in counts if counts[e]):
                rejections["formula_mismatch"] += 1
                continue
            key = tuple(counts.get(e, 0) for e in ref.ELEMENT_ORDER)
            if any(n > U16_MAX for n in key):
                rejections["count_overflow"] += 1
                continue
            mass = ref.mass_of(counts)
            if mass != sum(n * ref.MASS[e] for n, e in zip(key, ref.ELEMENT_ORDER)):
                rejections["formula_mismatch"] += 1
                continue
            if mass > max_mass:
                rejections["mass_over_limit"] += 1
                continue
            if counts.get("C", 0) < 1:
                rejections["no_carbon"] += 1
                continue
            rows.add(key)
            accepted += 1
    return rows, records, accepted, {r: int(rejections.get(r, 0)) for r in REJECTION_ORDER}


def parse_sources(text):
    sources = [s.strip() for s in text.split(",") if s.strip()]
    if not sources or set(sources) - {"train", "chebi"}:
        raise SystemExit("--sources must be a non-empty subset of {train, chebi}")
    return sorted(set(sources))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", type=Path, default=Path("data/pinned/MassSpecGym1.5.tsv"))
    ap.add_argument("--sdf", type=Path, default=Path("data/pinned/chebi_3_stars.sdf.gz"))
    ap.add_argument("--sources", type=str, default="train,chebi")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--table-out", type=Path, default=None)
    ap.add_argument("--max-mass-da", type=float, default=2000)
    args = ap.parse_args()
    sources = parse_sources(args.sources)

    # TSV side: exactly the train rows and validation keys of
    # formula_table_msgym.py (test rows skipped at parse time, never read).
    # Always collected: validation coverage needs the TSV even for
    # `--sources chebi`, and per-source coverages need both key sets.
    train_keys, train_structures, validation = collect_msgym(args.tsv)

    chebi_keys, chebi_records, chebi_accepted = set(), 0, 0
    chebi_rejections = {r: 0 for r in REJECTION_ORDER}
    if "chebi" in sources:
        chebi_keys, chebi_records, chebi_accepted, chebi_rejections = collect_chebi(
            args.sdf, args.max_mass_da
        )

    parts = []
    if "train" in sources:
        parts.append(train_keys)
    if "chebi" in sources:
        parts.append(chebi_keys)
    union = set().union(*parts) if parts else set()

    if sources == ["chebi", "train"]:
        source = "in-domain MassSpecGym1.5 structures of fold train + ChEBI 3-star formulas"
    elif sources == ["train"]:
        source = "in-domain MassSpecGym1.5 structures of fold train"
    else:
        source = "ChEBI 3-star formulas"

    table = build_table(union)
    payload = table_payload(table)
    report = build_report(table, union, train_structures, validation, source, payload)
    # Rust from_json checks only elements/rows, so the version string marks
    # this table apart from the train-only one; provenance rides alongside.
    report["version"] = VERSION
    report["sources"] = sources
    report["provenance"] = "formula_table_db"
    report["tsv"] = str(args.tsv)
    report["sdf"] = str(args.sdf) if "chebi" in sources else None
    report["max_mass_da"] = args.max_mass_da
    report["train_rows"] = len(train_keys)
    report["chebi_rows"] = len(chebi_keys)
    report["union_rows"] = len(union)
    report["validation_fold_formula_in_train"] = float(np.mean([k in train_keys for k in validation]))
    report["validation_fold_formula_in_chebi"] = float(np.mean([k in chebi_keys for k in validation]))
    report["chebi_records"] = chebi_records
    report["chebi_accepted"] = chebi_accepted
    report["chebi_rejections"] = chebi_rejections
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    if args.table_out:
        args.table_out.parent.mkdir(parents=True, exist_ok=True)
        args.table_out.write_text(payload)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
