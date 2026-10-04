"""The V0 formula table (docs/MS2_CONTRACTS.md section 9) from MassSpecGym.

The MassSpecGym twin of `formula_table.py`: rows are the distinct molecular
formulas of the in-domain structures of `fold == "train"`, same sort, same
table JSON schema (`--table-out`), same report JSON schema (`--out`), with
validation coverage computed on `fold == "val"`. Rows whose fold is `test`
are skipped at parse time and never read. Table construction and reporting
reuse `build_table`, `table_payload` and `build_report` from `formula_table.py`.

    PYTHONPATH=tools/ms2 uv run --with rdkit --with numpy --with pyarrow \
        python tools/ms2/formula_table_msgym.py --out /tmp/ms2_formula_table_msgym_report.json \
        --table-out data/ms2/formula_table_msgym_v0.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from rdkit import Chem, RDLogger

import ms2_reference as ref
from formula_table import build_report, build_table, table_payload

RDLogger.DisableLog("rdApp.*")

SPLIT_MSGYM_V1 = "msgym-split-v1"
SPLIT_PARTS = ("fit", "rank", "calibration", "report")


def normalize_identity(raw: str | None) -> str | None:
    """Normalised molecule-identity block of an InChIKey field.

    The identity is the text before the first ``-`` (so a full
    27-character InChIKey and its 14-character first block share an
    identity), upper-cased. Valid only when the block is exactly 14
    letters A-Z; otherwise None (the row is skipped as invalid).
    """
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    block = text.split("-", 1)[0].strip().upper()
    if len(block) != 14:
        return None
    if any(not ("A" <= c <= "Z") for c in block):
        return None
    return block


def part_of(key14: str, fold: str) -> str:
    """The `msgym-split-v1` part of an InChIKey first block and TSV fold.

    The key is normalised with `normalize_identity` first, so a full
    InChIKey and its 14-character block hash identically and land in the
    same part. Same rule as `export_msgym.part_of`, duplicated here so this
    tool does not import the exporter: `h` is the first 8 bytes of
    `sha256(block)` as a big-endian integer; in the train fold `rank` when
    `h % 5 == 0` else `fit`, in the val fold `calibration` when `h % 2 == 0`
    else `report`.
    """
    block = normalize_identity(key14)
    if block is None:
        raise ValueError(f"invalid InChIKey identity block {key14!r}")
    h = int.from_bytes(hashlib.sha256(block.encode()).digest()[:8], "big")
    if fold == "train":
        return "rank" if h % 5 == 0 else "fit"
    if fold == "val":
        return "calibration" if h % 2 == 0 else "report"
    raise ValueError(f"unknown fold {fold!r}")


def collect(tsv_path, molecule_filter=None):
    """Distinct train formula keys, in-domain train structure count, validation keys.

    Exactly the structures `main` used before this helper existed: distinct
    SMILES of `fold == "train"` (in-domain only) for rows, distinct SMILES of
    `fold == "val"` (in-domain only) for coverage. `fold == "test"` rows are
    skipped at parse time and never read. Molecule identity is the NORMALISED
    first InChIKey block (`normalize_identity`): rows with an invalid block
    are skipped; an identity block seen in more than one non-test fold is a
    fold conflict and every SMILES of that block is excluded from rows (and
    from coverage). With `molecule_filter(smiles, inchikey14, fold)` set, a
    structure is kept only when the filter returns true (the key passed is
    the normalised block); with the default `None` the result is unchanged
    on conflict-free, valid inputs.
    """
    with open(tsv_path, "r") as fh:
        header = fh.readline().rstrip("\n").split("\t")
    col = {name: i for i, name in enumerate(header)}
    for needed in ("smiles", "fold"):
        if needed not in col:
            raise SystemExit(f"{tsv_path} has no {needed!r} column")
    if molecule_filter is not None and "inchikey" not in col:
        raise SystemExit(f"{tsv_path} has no 'inchikey' column")

    # Distinct structures per fold; test rows are skipped at parse time.
    # Identity bookkeeping mirrors export_msgym.py: normalised blocks only.
    train_smiles, val_smiles = set(), set()
    smiles_key14: dict[str, str] = {}
    smiles_folds: dict[str, set[str]] = {}
    smiles_blocks: dict[str, set[str]] = {}
    identity_folds: dict[str, set[str]] = {}
    identity_smiles: dict[str, set[str]] = {}
    has_key_col = "inchikey" in col
    with open(tsv_path, "r") as fh:
        fh.readline()
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            fold = parts[col["fold"]]
            if fold == "test":
                continue
            if fold not in ("train", "val"):
                continue
            raw = parts[col["inchikey"]] if has_key_col else ""
            block = normalize_identity(raw)
            if block is None:
                continue
            smi = parts[col["smiles"]]
            smiles_folds.setdefault(smi, set()).add(fold)
            smiles_blocks.setdefault(smi, set()).add(block)
            identity_folds.setdefault(block, set()).add(fold)
            identity_smiles.setdefault(block, set()).add(smi)
            smiles_key14.setdefault(smi, block)
            if fold == "train":
                train_smiles.add(smi)
            elif fold == "val":
                val_smiles.add(smi)
    # Exclude fold conflicts: a SMILES in both non-test folds, and every
    # SMILES of a cross-fold identity block. (A SMILES with two blocks in
    # the same fold keeps its first block, preserving legacy outputs.)
    conflict_identities = {b for b, folds in identity_folds.items() if len(folds) > 1}
    excluded: set[str] = set()
    for smi, folds in smiles_folds.items():
        if len(folds) > 1:
            excluded.add(smi)
    for b in conflict_identities:
        excluded |= identity_smiles[b]
    train_smiles -= excluded
    val_smiles -= excluded
    if molecule_filter is not None:
        train_smiles = {s for s in train_smiles
                        if molecule_filter(s, smiles_key14.get(s, ""), "train")}
        val_smiles = {s for s in val_smiles
                      if molecule_filter(s, smiles_key14.get(s, ""), "val")}

    rows, train_structures = set(), 0
    for smi in sorted(train_smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            Chem.Kekulize(mol, clearAromaticFlags=True)
        except Exception:
            continue
        if ref.classify(mol):
            continue
        atoms, _ = ref.graph_of(mol)
        counts = ref.composition(atoms)
        rows.add(tuple(counts[e] for e in ref.ELEMENT_ORDER))
        train_structures += 1
    validation = []
    for smi in sorted(val_smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            Chem.Kekulize(mol, clearAromaticFlags=True)
        except Exception:
            continue
        if ref.classify(mol):
            continue
        atoms, _ = ref.graph_of(mol)
        counts = ref.composition(atoms)
        validation.append(tuple(counts[e] for e in ref.ELEMENT_ORDER))
    return rows, train_structures, validation


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", type=Path, default=Path("data/pinned/MassSpecGym1.5.tsv"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--table-out", type=Path, default=None)
    ap.add_argument("--split", choices=[SPLIT_MSGYM_V1], default=None)
    ap.add_argument("--part", choices=list(SPLIT_PARTS), default=None,
                    help="with --split, build rows from this part only "
                    "(validation coverage stays on the full val fold)")
    ap.add_argument("--formula-table-from", choices=list(SPLIT_PARTS), default=None,
                    help="alias for --part: build rows from this part only")
    args = ap.parse_args()

    requested = args.formula_table_from or args.part
    if (args.part is not None and args.formula_table_from is not None
            and args.part != args.formula_table_from):
        ap.error("--part and --formula-table-from conflict "
                 f"({args.part!r} versus {args.formula_table_from!r})")
    if args.split is not None and requested is None:
        ap.error("--split requires --part or --formula-table-from")
    if args.split is None and (args.part is not None or args.formula_table_from is not None):
        ap.error("--part/--formula-table-from requires --split")
    if args.split is not None and requested is not None \
            and requested not in ("fit", "rank"):
        ap.error(f"--part {requested!r} cannot supply table rows: "
                 f"calibration and report are evaluation data; "
                 f"only fit and rank (train-fold parts) may supply rows")
    molecule_filter = None
    source = "in-domain MassSpecGym1.5 structures of fold train"
    if requested is not None:
        wanted = requested
        def molecule_filter(smiles, key14, fold, _wanted=wanted):  # noqa: B023
            if fold == "val":
                return True
            return part_of(key14, fold) == _wanted
        source = (f"in-domain MassSpecGym1.5 structures of "
                  f"{SPLIT_MSGYM_V1} {wanted} (validation coverage on fold val)")

    rows, train_structures, validation = collect(args.tsv, molecule_filter)

    table = build_table(rows)
    payload = table_payload(table)
    report = build_report(table, rows, train_structures, validation,
                          source, payload)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    if args.table_out:
        args.table_out.parent.mkdir(parents=True, exist_ok=True)
        args.table_out.write_text(payload)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
